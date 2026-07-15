"""Model selection + the three deployment gates.

Offline RL cannot be model-selected on TD loss: a critic can drive its own loss to
zero while producing a useless policy. We have something better and use it — the
tree is fixed, so the learned policy is evaluated EXACTLY by rolling it out in the
offline env. That is real policy evaluation, not an off-policy estimator.

    primary   : coverage at the declared reference budget (10 * delta_root)
    tie-break : 1. regret over solved instances (lower)
                2. earlier checkpoint (prefer the simpler model)

NOT TD loss. NOT mean Q. NOT doom rate — that is provably 0 on solvable data
(completeness proposition), so it can never break a tie.

SPLITS ARE INSTANCE-LEVEL AND WITHIN-CONFIGURATION.
Row-level splits leak: rows from one tree share nodes, and a model can memorise a
tree it has partially seen. And a CROSS-configuration split is worse than useless:
node ids are hashes of the fluent set, so two configurations share no vocabulary
(measured 0.0% id overlap) and the node channel is provably pure noise. Any
cross-config number is noise. `assert_within_config` enforces it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .planner_config import planner_flags
from .tree import TreeInstance


def config_of(instance_name: str) -> str:
    """`CC_2_3_4__pl_7` -> `CC_2_3_4`. The fluent vocabulary is per-configuration."""
    return instance_name.rsplit("__pl_", 1)[0]


def assert_within_config(train: Sequence[str], val: Sequence[str],
                         test: Sequence[str] = (),
                         *, allow_cross_config: bool = False) -> None:
    """The guardrail. A cross-configuration split makes the NODE CHANNEL noise.

    Measured: 0.0% node-id overlap across configurations (they share no fluent
    vocabulary); 15-46% within one. Training on CC_2_2_x and testing on CC_2_3_x is
    the trap that produced a whole day of unreadable numbers.

    WHY 0.0%, PRECISELY (measured 2026-07-15): on HASHED the node id IS the feature
    -- `_prepare_node_features` feeds `hash / 2^63` to input_proj, and
    `node_label_embedding` takes `hash % 4096`. BOTH node channels are pure functions
    of the hash, and `boost::hash_range` destroys fluent structure by construction.
    The underlying domains actually share 50-87% of their FLUENTS (10 common to all 5
    CC configs); it is the hashing, not the domain, that removes the signal.

    SCOPE OF THE CLAIM: only the node channel is noise. The model also consumes
    `edge_attr` (agent/edge labels, shared vocabulary) and `edge_index` (topology),
    which are config-independent. A cross-config run is therefore a STRUCTURE-ONLY
    FLOOR -- weak, but not meaningless. `allow_cross_config=True` opts into exactly
    that, deliberately and labelled; the default stays a hard raise because the trap
    above is real and silent.

    (On BITMASK the node feature is the fluent bitmask, so cross-config transfer is
    genuinely available there -- that is the real test, not this floor.)
    """
    cfgs = {config_of(n) for n in list(train) + list(val) + list(test)}
    if len(cfgs) <= 1:
        return
    if allow_cross_config:
        print(
            f"[WARNING] CROSS-CONFIG RUN (exploratory): {sorted(cfgs)}. On HASHED the "
            f"node channel carries ZERO cross-config signal -- the id is a hash and "
            f"configurations share 0.0% of ids. Any gap measured here comes from "
            f"TOPOLOGY + EDGE LABELS alone; this is a STRUCTURE-ONLY FLOOR, not a "
            f"transfer result. Do not report it as one."
        )
        return
    raise ValueError(
        f"cross-configuration split: {sorted(cfgs)}. Node ids are hashes of the "
        f"fluent set, so different configurations share NO vocabulary (0.0% id "
        f"overlap measured) and the node channel is provably pure noise. Splits "
        f"must stay within one configuration. Pass --allow-cross-config to opt in "
        f"deliberately (the run is then stamped exploratory)."
    )


def split_instances(
    names: Sequence[str], val_frac: float = 0.2, seed: int = 0,
    val_instances: Optional[Sequence[str]] = None,
    *, allow_cross_config: bool = False,
) -> Tuple[List[str], List[str]]:
    """Instance-level split with a fixed seed, recorded in the manifest."""
    import random
    if val_instances is not None:
        val = [n for n in names if n in set(val_instances)]
        train = [n for n in names if n not in set(val_instances)]
    else:
        rng = random.Random(seed)
        shuf = sorted(names)
        rng.shuffle(shuf)
        k = max(1, int(round(val_frac * len(shuf))))
        val, train = shuf[:k], shuf[k:]
    if not train or not val:
        raise ValueError(f"degenerate split: train={train} val={val}")
    assert_within_config(train, val, allow_cross_config=allow_cross_config)
    return sorted(train), sorted(val)


@dataclass(order=False)
class Candidate:
    step: int
    frames: int
    coverage: float
    regret: Optional[float]
    doom: float
    payload: Dict = field(default_factory=dict)

    def key(self):
        """Sort key. Higher coverage first; then LOWER regret; then EARLIER step.

        `regret is None` (nothing solved) must sort worst, not best.
        """
        return (-self.coverage,
                (self.regret if self.regret is not None else float("inf")),
                self.step)


def select(candidates: Sequence[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("no checkpoints to select from")
    return sorted(candidates, key=lambda c: c.key())[0]


# ----------------------------------------------------------- the gates ------

@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def gate_onnx_parity(check: Callable[[], Tuple[bool, str]]) -> GateResult:
    """Gate 1: onnxruntime reproduces PyTorch to 1e-5, both encodings, all context
    modes. Covered by tests/test_onnx_contract.py (55 tests)."""
    ok, detail = check()
    return GateResult("onnx_parity", ok, detail)


PLANNER_EXPANSIONS_RE = r"Nodes expanded:\s*(\d+)"


def run_planner_expansions(
    deep_exe: str | Path,
    problem_file: str | Path,
    onnx_path: str | Path,
    fringe_size: int,
    separated: bool,
    repo_root: str | Path = ".",
    strong_equality: bool = True,
    timeout_s: int = 600,
) -> Optional[int]:
    """Invoke the REAL planner and parse its expansion count.

    Returns None if the planner failed or printed no count — the caller must treat
    that as a gate failure, never as "no data".

    The ONNX must match the encoding: a merged export takes 5 inputs and a
    separated one 9, and `FringeEvalRL.tpp:413` rejects a mismatch outright
    ("model expects 5 input tensors but C++ prepared 9"). `--dataset_separated`
    is what makes the C++ append the goal tensors.
    """
    if not Path(deep_exe).exists():
        raise FileNotFoundError(
            f"planner binary not found: {deep_exe}. The env-fidelity gate cannot be "
            f"armed without it -- build it (cmake-build-release-nn) or the gate is "
            f"scaffolding. A MISSING BINARY IS A SETUP ERROR, not a model failure: "
            f"returning 'no count' here would silently fail the gate and hide the "
            f"real cause."
        )
    if not Path(problem_file).exists():
        raise FileNotFoundError(f"problem file not found: {problem_file}")
    cmd = [str(deep_exe), str(problem_file), "-b", "-c",
           "--search", "RL",
           "--RL_model", str(Path(onnx_path).resolve()),
           "--RL_fringe_size", str(int(fringe_size)),
           "--RL_exploitation", str(planner_flags(fringe_size)["RL_exploitation"]),
           "--RL_exploration", "0"]
    if separated:
        cmd.append("--dataset_separated")
    if strong_equality:
        cmd.append("--strong_equality")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           cwd=str(repo_root))
    except (subprocess.TimeoutExpired, OSError):
        # A crash or a timeout IS a gate failure (unlike a missing binary, which is
        # a setup error and raises above).
        return None
    import re
    m = re.search(PLANNER_EXPANSIONS_RE, r.stdout or "")
    return int(m.group(1)) if m else None


def gate_env_fidelity(
    offline_expansions: Sequence[int],
    planner_expansions: Sequence[int],
    tolerance_frac: float = 0.10,
    min_expansions: int = 20,
) -> GateResult:
    """Gate 2: THE most important test.

    The C++ planner and the offline env must agree on expansion counts for the same
    policy on the same instances. If they disagree, the offline env is a wrong model
    of the planner and every number we report is meaningless.

    The tolerance is not eyeballed: the offline env replays a DFS spanning tree of a
    hash-deduplicated DAG, while the live planner dedups against the states it has
    actually visited. A child that is fresh in the tree may already be visited live
    (and dropped), so the counts can legitimately differ. The threshold is the
    measured median of that divergence; exceeding it means the disagreement is not
    tree-replay drift.
    """
    if not offline_expansions or len(offline_expansions) != len(planner_expansions):
        return GateResult("env_fidelity", False,
                          f"nothing to compare: offline={len(offline_expansions)} "
                          f"planner={len(planner_expansions)}")
    if any(p is None for p in planner_expansions):
        return GateResult("env_fidelity", False,
                          f"the planner produced no expansion count for "
                          f"{sum(1 for p in planner_expansions if p is None)} instance(s); "
                          f"a missing count is a FAILURE, never 'no data'")
    # A fractional tolerance is meaningless on tiny searches: at 7 expansions a
    # +-1 difference is 14%. Instances below `min_expansions` cannot discriminate
    # tree-replay drift from a real modelling error, so they must not be scored --
    # silently averaging them in would let a trivial instance pass or fail the gate
    # for arithmetic reasons.
    usable = [(a, b) for a, b in zip(offline_expansions, planner_expansions)
              if b >= min_expansions]
    if not usable:
        return GateResult(
            "env_fidelity", False,
            f"no instance reaches {min_expansions} planner expansions "
            f"(largest={max(planner_expansions)}); a fractional tolerance cannot "
            f"discriminate at this scale. Use harder fidelity instances. "
            f"offline={list(offline_expansions)} planner={list(planner_expansions)}")
    devs = [abs(a - b) / max(1, b) for a, b in usable]
    med = sorted(devs)[len(devs) // 2]
    ok = med <= tolerance_frac
    return GateResult(
        "env_fidelity", ok,
        f"median |offline-planner|/planner = {med:.3f} over {len(usable)}/"
        f"{len(offline_expansions)} instances with >= {min_expansions} expansions "
        f"(tolerance {tolerance_frac:.3f}); per-instance {[round(d,3) for d in devs]}; "
        f"offline={list(offline_expansions)} planner={list(planner_expansions)}",
    )


def gate_beats_baselines(
    model_regret: Optional[float],
    baselines: Dict[str, Optional[float]],
    exclude: Sequence[str] = ("hfs_oracle",),
) -> GateResult:
    """Gate 3: beat bfs / dfs / random AND the two-head baseline on val regret.

    NOT required to beat hfs_oracle -- that is the clairvoyant ceiling (it ranks by
    delta, the answer to the problem) and cannot be beaten. Failing this does not
    block export; it prints a prominent warning, because "we shipped a model that
    loses to dfs" must be visible rather than silent.
    """
    if model_regret is None:
        return GateResult("beats_baselines", False, "model solved nothing")
    lost = {k: v for k, v in baselines.items()
            if k not in exclude and v is not None and v <= model_regret}
    return GateResult(
        "beats_baselines", not lost,
        (f"model regret {model_regret:.2f}; LOSES TO {lost}" if lost
         else f"model regret {model_regret:.2f} beats {sorted(set(baselines) - set(exclude))}"),
    )


# --------------------------------------------------------- the sidecar ------

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_selection_sidecar(
    onnx_path: str | Path,
    selected: Candidate,
    *,
    train_instances: Sequence[str],
    val_instances: Sequence[str],
    fringe_size: int,
    kind_of_data: str,
    model: str,
    gamma: float,
    reward_scale: float,
    baselines: Dict[str, Optional[float]],
    gates: Sequence[GateResult],
    excluded_unsolvable: Sequence[Dict] = (),
    test_regret: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """The sidecar records everything needed to reproduce and to trust the export.

    Includes the exact C++ flags the model must be launched with: the offline MDP
    models one expansion per ONNX call, which only matches the planner at
    RL_node_to_add == 1, and that is a function of --RL_exploitation.
    """
    p = Path(onnx_path).with_suffix(".selection.json")
    doc = {
        "checkpoint": selected.step,
        "frames": selected.frames,
        "val_coverage_at_reference_budget": selected.coverage,
        "val_regret_lower_bound": selected.regret,
        "val_doom": selected.doom,
        "test_regret_lower_bound": test_regret,
        "n_train_instances": len(train_instances),
        "n_val_instances": len(val_instances),
        "train_instances": sorted(train_instances),
        "val_instances": sorted(val_instances),
        "configuration": config_of(val_instances[0]) if val_instances else None,
        "fringe_size": fringe_size,
        "kind_of_data": kind_of_data,
        "model": model,
        "gamma": gamma,
        "reward_scale": reward_scale,
        "baselines_val_regret_lower_bound": baselines,
        "planner_flags": planner_flags(fringe_size),
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates],
        "all_gates_passed": all(g.passed for g in gates),
        "excluded_unsolvable_instances": list(excluded_unsolvable),
        "git_sha": git_sha(),
    }
    if extra:
        doc.update(extra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str))
    return p


def onnx_path_for(exp_dir: str | Path, domain: str, fringe_size: int) -> Path:
    """`<exp_dir>/_models/<domain>/frontier_policy_<F>.onnx` — bulk_coverage_run.py
    depends on this exact naming, and the C++ checks logits length == F at load."""
    return Path(exp_dir) / "_models" / domain / f"frontier_policy_{int(fringe_size)}.onnx"
