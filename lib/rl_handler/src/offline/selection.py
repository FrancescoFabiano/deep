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
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .planner_config import planner_flags
from .tree import INF_DELTA, TreeInstance


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
    # The scalar the SMOOTHED selector maximises (higher = better). PRIMARY path
    # (held-out test instances): coverage. FALLBACK path (held-out trajectories):
    # held-out top1-oracle-agreement -- a low-variance ranking signal, because
    # coverage at n=15 rollouts is too noisy to argmax over (the floor run showed
    # argmax picks a lucky single rollout). None => fall back to the coverage/regret
    # key below.
    select_score: Optional[float] = None

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


def select_smoothed(candidates: Sequence[Candidate], window: int = 3) -> Candidate:
    """Pick the checkpoint maximising the WINDOW-MEAN of `select_score` over `window`
    consecutive checkpoints -- NOT a single argmax draw.

    The floor run proved argmax-on-a-single-draw selects noise: coverage bounced
    0.73-1.00 with no trend at n=15 rollouts (1 rollout = 0.067), and the rule
    reported a lucky peak. Averaging over a window of checkpoints cuts selection
    variance ~sqrt(window) for free. Ties -> EARLIEST checkpoint (a model that peaked
    early and held is preferred to a late fluke).

    Requires select_score set on every candidate. If none is set, this raises -- use
    plain select() for the coverage/regret key instead.
    """
    if not candidates:
        raise ValueError("no checkpoints to select from")
    if any(c.select_score is None for c in candidates):
        raise ValueError(
            "select_smoothed needs select_score on every candidate; use select() for "
            "the coverage/regret key."
        )
    cands = sorted(candidates, key=lambda c: c.step)
    scores = [c.select_score for c in cands]
    best_i, best_mean = 0, float("-inf")
    for i in range(len(cands)):
        lo = max(0, i - window + 1)
        w = scores[lo:i + 1]
        m = sum(w) / len(w)
        if m > best_mean:                 # strict > -> earliest wins ties
            best_mean, best_i = m, i
    return cands[best_i]


# ------------------------------------------- held-out trajectory split -------

def split_trajectories(
    rows: Sequence,
    frac: float = 0.10,
    seed: int = 0,
    min_heldout: int = 1,
) -> Tuple[List, List, Dict]:
    """Hold out ~`frac` of EACH instance's TRAJECTORIES (whole rollouts) for eval.

    A trajectory is one behaviour rollout, keyed (instance, policy, seed); every
    counterfactual action of a frontier AND every consecutive frontier of that
    rollout share the key (verified in dataset._emit), so holding out whole
    trajectories avoids the two leaks a per-frontier split would cause:
      1. a frontier's K actions never straddle train/eval;
      2. consecutive frontiers (which differ by ONE expansion and are near-identical)
         never straddle train/eval -- a held-out frontier is never a one-step
         neighbour of a trained one.

    Every instance keeps >=1 trajectory in TRAIN (no problem is dropped from training)
    and, if it has >=2, contributes >=1 to EVAL (so it is represented on both sides).
    Splits per-instance so the eval covers every instance, not a random subset.

    Returns (train_rows, heldout_rows, manifest). The manifest records the fraction,
    the seed, the per-instance train/eval counts, and the exact held-out
    (instance, policy, seed) list -- so the split is reproducible and auditable, and
    a too-thin instance is visible.
    """
    by_inst: Dict[str, set] = defaultdict(set)
    for r in rows:
        by_inst[r.instance].add((r.policy, r.seed))

    rng = random.Random(seed)
    heldout_keys: set = set()          # (instance, policy, seed)
    per_instance: Dict[str, Dict] = {}
    for inst in sorted(by_inst):
        trajs = sorted(by_inst[inst])
        rng.shuffle(trajs)
        n = len(trajs)
        k = math.ceil(frac * n)
        if n >= 2:
            k = max(k, min_heldout)
        k = min(k, n - 1)              # never hold out ALL: keep >=1 trajectory in train
        for pol, sd in trajs[:k]:
            heldout_keys.add((inst, pol, sd))
        per_instance[inst] = {"n_traj": n, "n_train": n - k, "n_eval": k}

    def key(r):
        return (r.instance, r.policy, r.seed)

    train_rows = [r for r in rows if key(r) not in heldout_keys]
    heldout_rows = [r for r in rows if key(r) in heldout_keys]
    manifest = {
        "split": "held_out_trajectories",
        "frac": frac,
        "seed": seed,
        "min_heldout": min_heldout,
        "n_train_rows": len(train_rows),
        "n_heldout_rows": len(heldout_rows),
        "heldout_trajectories": sorted(list(t) for t in heldout_keys),
        "per_instance": per_instance,
        "instances_with_no_eval": [i for i, c in per_instance.items() if c["n_eval"] == 0],
    }
    return train_rows, heldout_rows, manifest


def heldout_top1(
    heldout_rows: Sequence,
    rank_for: Callable[[str, Sequence[int]], Sequence[int]],
    instances_by_name: Dict[str, TreeInstance],
) -> Tuple[float, int]:
    """Mean top1-oracle-agreement over the UNIQUE held-out frontiers.

    The deployment decision IS a ranking -- expand the lowest-delta node, discard the
    highest -- so top1 (does the ranker put a minimum-delta node first?) is the
    ranking analogue of what is deployed. It is scored identically for the RL model
    and for a baseline (both expose `rank_for(instance, beam) -> ranking`), so the
    comparison is matched-n on the SAME held-out frontiers -- no max-of-20-vs-single
    bias.

    `rank_for(name, beam) -> ranking` (slot indices, best first).
    Degenerate frontiers are skipped: a singleton beam or one with no viable node
    cannot discriminate a good ranker from a bad one. Returns (mean_top1, n_scored).
    """
    seen: set = set()
    hits = n = 0
    for r in heldout_rows:
        if getattr(r, "forced", False):
            continue                     # forced states have no choice to rank
        fkey = (r.instance, tuple(r.obs))
        if fkey in seen:
            continue
        seen.add(fkey)
        beam = list(r.obs)
        inst = instances_by_name[r.instance]
        deltas = [inst.delta[v] for v in beam]
        if len(beam) < 2 or all(d == INF_DELTA for d in deltas):
            continue
        ranking = rank_for(r.instance, beam)
        best = min(deltas)
        hits += 1 if deltas[ranking[0]] == best else 0
        n += 1
    return (hits / n if n else 0.0), n


# ----------------------------------------- full-ranking metrics --------------
# top1 asks only "is the best first?". But the policy also DISCARDS the worst kappa,
# so the WHOLE ordering matters. We have ground-truth delta for every node, so measure
# the full ranking. TIE-SAFETY is load-bearing: delta has massive ties (many nodes
# share inf = dead subtree, several share a finite value), and two GENUINELY
# EQUIVALENT nodes ordered either way must score the SAME -- a strict-order metric
# would invent a phantom penalty. NDCG-with-gains and the softmax divergence are
# tie-safe by construction; Kendall is reported tie-aware (tau-b) and never headlined.

def _gain(d: float) -> float:
    """delta -> gain, decreasing, tie-safe, inf-safe. A dead node (inf) has gain 0;
    equal delta -> equal gain (so any order within a tie scores identically)."""
    return 0.0 if d >= INF_DELTA else 1.0 / (1.0 + d)


def _ndcg(ranking: Sequence[int], gains: Sequence[float]) -> float:
    """NDCG of the ranking against delta-based gains -- rank correlation WEIGHTED
    TOWARD THE EXTREMES (a mistake at top/bottom hurts more than one in the middle,
    matching a task that acts on the ends). Tie-safe: swapping two equal-gain nodes
    leaves DCG unchanged (equal numerators). All-zero gains (all dead) -> 1.0."""
    dcg = sum(gains[ranking[i]] / math.log2(i + 2) for i in range(len(ranking)))
    ideal = sorted(gains, reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    return (dcg / idcg) if idcg > 0 else 1.0


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _js_divergence(logits: Sequence[float], deltas: Sequence[float],
                   temp: float = 1.0) -> float:
    """Jensen-Shannon divergence (bits, symmetric, in [0,1]) between the model's
    preference distribution softmax(logits) and the ORACLE's softmax(-delta): where
    the model puts its preference MASS vs where the oracle does. Tie-safe: equal delta
    -> equal target mass. A dead node (inf) gets ~0 oracle mass. Oracle-vs-oracle = 0."""
    P = _softmax([l / temp for l in logits])
    negd = [(-(1e9) if d >= INF_DELTA else -d) / temp for d in deltas]
    Q = _softmax(negd)
    M = [(p + q) / 2.0 for p, q in zip(P, Q)]

    def _kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


def ranking_metrics_for_frontier(
    deltas: Sequence[float],
    ranking: Optional[Sequence[int]] = None,
    logits: Optional[Sequence[float]] = None,
) -> Dict[str, Optional[float]]:
    """All ranking metrics for ONE frontier. Pass `logits` (model: enables the
    softmax divergence) or `ranking` (baseline). Returns per-frontier values; the
    caller averages. `js` is None when only a ranking is available."""
    if ranking is None:
        if logits is None:
            raise ValueError("need ranking or logits")
        ranking = sorted(range(len(deltas)), key=lambda k: -logits[k])
    best = min(deltas)
    top = ranking[0]
    gains = [_gain(d) for d in deltas]
    # FAMILY 1 -- top-focused
    top1 = 1.0 if deltas[top] == best else 0.0
    top_finite = deltas[top] < INF_DELTA
    picked_dead = 1.0 if (deltas[top] >= INF_DELTA and best < INF_DELTA) else 0.0
    regret_at_decision = (deltas[top] - best) if (top_finite and best < INF_DELTA) else None
    # FAMILY 2 -- full-ranking
    ndcg = _ndcg(ranking, gains)
    js = _js_divergence(logits, deltas) if logits is not None else None
    tau = None
    if len(deltas) > 2:
        from scipy.stats import kendalltau       # tau-b: tie-aware
        # model order as a score (higher = ranked earlier), vs -delta
        order_score = [0.0] * len(ranking)
        for pos, slot in enumerate(ranking):
            order_score[slot] = -pos
        t = kendalltau(order_score, [-d if d < INF_DELTA else -1e9 for d in deltas]).statistic
        tau = float(t) if t == t else None       # nan -> None (all-tie)
    return {"top1": top1, "regret_at_decision": regret_at_decision,
            "picked_dead": picked_dead, "ndcg": ndcg, "js": js, "kendall_tau": tau}


def heldout_ranking_metrics(
    heldout_rows: Sequence,
    instances_by_name: Dict[str, TreeInstance],
    rank_for: Optional[Callable[[str, Sequence[int]], Sequence[int]]] = None,
    logits_for: Optional[Callable[[str, Sequence[int]], Sequence[float]]] = None,
) -> Dict[str, object]:
    """Aggregate the full ranking-metric family over the UNIQUE held-out frontiers,
    matched-n with baselines. Pass `logits_for` for the MODEL (enables js), or
    `rank_for` for a baseline. Same frontiers, same skipping rules as heldout_top1."""
    seen: set = set()
    acc: Dict[str, list] = defaultdict(list)
    n = 0
    for r in heldout_rows:
        if getattr(r, "forced", False):
            continue
        fkey = (r.instance, tuple(r.obs))
        if fkey in seen:
            continue
        seen.add(fkey)
        beam = list(r.obs)
        inst = instances_by_name[r.instance]
        deltas = [inst.delta[v] for v in beam]
        if len(beam) < 2 or all(d >= INF_DELTA for d in deltas):
            continue
        if logits_for is not None:
            m = ranking_metrics_for_frontier(deltas, logits=list(logits_for(r.instance, beam)))
        else:
            m = ranking_metrics_for_frontier(deltas, ranking=list(rank_for(r.instance, beam)))
        for k, v in m.items():
            if v is not None:
                acc[k].append(v)
        n += 1
    out: Dict[str, object] = {"n": n}
    for k in ("top1", "regret_at_decision", "picked_dead", "ndcg", "js", "kendall_tau"):
        out[k] = (sum(acc[k]) / len(acc[k])) if acc[k] else None
    return out


_RANK_KEYS = ("top1", "regret_at_decision", "picked_dead", "ndcg", "js", "kendall_tau")


def _order_score(ranking: Sequence[int], n: int) -> list:
    """ranking (slots best-first) -> per-slot score (higher = ranked earlier), for tau-b."""
    s = [0.0] * n
    for pos, slot in enumerate(ranking):
        s[slot] = -pos
    return s


def heldout_ranking_micro_macro(
    heldout_rows: Sequence,
    instances_by_name: Dict[str, TreeInstance],
    logits_for: Callable[[str, Sequence[int]], Sequence[float]],
    agreement_rankers: Optional[Dict[str, Callable[[str, Sequence[int]], Sequence[int]]]] = None,
) -> Dict[str, object]:
    """Fix 3 (micro/macro) + Fix 2 (policy agreement) in ONE forward pass over the unique
    held-out frontiers. Returns the pooled MICRO metrics (identical to
    `heldout_ranking_metrics`, so the selector is unchanged), per-instance metrics + MACRO
    + EFFECTIVE INSTANCE COUNT, and -- when `agreement_rankers` is given -- top1/tau-b
    agreement of the MODEL's ranking with EACH behaviour policy's ranking. The single pass
    is what keeps checkpoint eval from getting slower: model logits are computed once and
    reused for micro, macro, and agreement.

    Recording only: selection stays on micro. `agreement_rankers` must be built with a
    FIXED tie-break seed by the caller (recorded in telemetry) so the curve carries no
    tie-break noise -- every behaviour policy uses the mandatory random tie-break.
    """
    seen: set = set()
    pooled: Dict[str, list] = defaultdict(list)
    per: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    ag_top1: Dict[str, list] = defaultdict(list)
    ag_taub: Dict[str, list] = defaultdict(list)
    n_pool = 0
    n_inst: Dict[str, int] = defaultdict(int)
    for r in heldout_rows:
        if getattr(r, "forced", False):
            continue
        fkey = (r.instance, tuple(r.obs))
        if fkey in seen:
            continue
        seen.add(fkey)
        beam = list(r.obs)
        inst = instances_by_name[r.instance]
        deltas = [inst.delta[v] for v in beam]
        if len(beam) < 2 or all(d >= INF_DELTA for d in deltas):
            continue
        logits = list(logits_for(r.instance, beam))
        m = ranking_metrics_for_frontier(deltas, logits=logits)
        for k, v in m.items():
            if v is not None:
                pooled[k].append(v)
                per[r.instance][k].append(v)
        n_pool += 1
        n_inst[r.instance] += 1
        if agreement_rankers:
            model_rank = sorted(range(len(beam)), key=lambda k: -logits[k])
            for pol, rank_fn in agreement_rankers.items():
                pr = list(rank_fn(r.instance, beam))
                ag_top1[pol].append(1.0 if model_rank[0] == pr[0] else 0.0)
                if len(beam) > 2:
                    from scipy.stats import kendalltau
                    t = kendalltau(_order_score(model_rank, len(beam)),
                                   _order_score(pr, len(beam))).statistic
                    if t == t:  # not NaN
                        ag_taub[pol].append(t)

    def _mean(d, k):
        return (sum(d[k]) / len(d[k])) if d[k] else None

    def _m(lst):
        return (sum(lst) / len(lst)) if lst else None

    micro = {"n": n_pool, **{k: _mean(pooled, k) for k in _RANK_KEYS}}
    per_instance = {i: {"n": n_inst[i], **{k: _mean(per[i], k) for k in _RANK_KEYS}}
                    for i in per}
    ndcgs = [pi["ndcg"] for pi in per_instance.values() if pi["ndcg"] is not None]
    top1s = [pi["top1"] for pi in per_instance.values() if pi["top1"] is not None]
    out = {
        "micro": micro,
        "per_instance": per_instance,
        "ndcg_macro": (sum(ndcgs) / len(ndcgs)) if ndcgs else None,
        "top1_macro": (sum(top1s) / len(top1s)) if top1s else None,
        "effective_instance_count": len(per_instance),
    }
    if agreement_rankers:
        out["agreement"] = {pol: {"top1": _m(ag_top1[pol]), "taub": _m(ag_taub[pol])}
                            for pol in agreement_rankers}
    return out


# ----------------------------------------------------------- the gates ------

@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    armed: bool = True
    """False = the gate did not RUN (a precondition was absent), which is NOT the same
    as running and failing. Conflating them made an unarmed env-fidelity gate print
    'FAIL -- NOT ARMED' and fail the whole run, so disarming a gate we deliberately
    do not want (planner deployment out of scope) would abort the launcher at step 2.
    A gate that did not measure anything cannot have a verdict: callers must count
    only `armed and not passed` as failure."""
    blocking: bool = True
    """False = a failure is a WARNING, not a run failure. `beats_baselines` is
    non-blocking by design ("failing does not block export; it prints a prominent
    warning" -- a weak model is still a valid, recordable result, e.g. the HASHED
    floor null where the model ties random). Only env-fidelity/onnx-parity are
    validity checks that must block. The run fails only on `armed and blocking and not
    passed` -- otherwise making beats_baselines honest (top1, not the optimistic
    train-set regret) would abort the launcher on exactly the floor result we want to
    record."""


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
    # KNOWN BUG -- PARKED, NOT FIXED (2026-07-15). This uses planner_flags() for
    # RL_exploitation but DISCARDS the rest of what it returns, including
    # "RL_heuristics": "RNG". The C++ default is MIN, so the planner refills the
    # reservoir deterministically while the offline env models RANDOM refill -- the
    # exact invariant planner_config exists to enforce ("refill must be RANDOM
    # (--RL_heuristics RNG), the only mode reproducible offline"). Different refill
    # rules cannot produce matching expansion counts, so the gate compares two
    # different algorithms and fails by construction.
    #
    # It went unnoticed because the export device leak (run.py) crashed every run
    # before this ran: gate 2 had NEVER executed. Its one and only verdict --
    # "FAIL, median 0.738 over 1/3 instances, offline=[27,30,27] planner=[14,9,103]"
    # -- is this bug, not evidence the env is unfaithful. n=1, and the discrepancy
    # flips sign across instances.
    #
    # NOT fixed because planner deployment is out of scope; the gate is unarmed
    # (--deep-exe defaults to None). When deployment returns: pass every flag
    # planner_flags() computes, not one of them -- and note RNG refill makes the gate
    # stochastic across seeds, which needs its own thought. Scrutinize the first real
    # verdict; this is still untested code.
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
    names: Optional[Sequence[str]] = None,
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
        missing = [(names[i] if names and i < len(names) else f"#{i}")
                   for i, p in enumerate(planner_expansions) if p is None]
        return GateResult("env_fidelity", False,
                          f"the planner produced no expansion count for "
                          f"{len(missing)} instance(s) {missing}; "
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
    model_score: Optional[float],
    baselines: Dict[str, Optional[float]],
    exclude: Sequence[str] = ("hfs_oracle",),
    *,
    higher_is_better: bool = False,
    metric: str = "regret",
) -> GateResult:
    """Gate 3: beat bfs / dfs / random on the SELECTION metric, on the SAME held-out
    set the model was selected on.

    metric/direction MUST match the eval mode, or the gate passes on the wrong number:
      FALLBACK (held-out trajectories): metric='heldout_top1', higher_is_better=True.
        regret here is a TRAIN-SET rollout estimate -- optimistic (the model trained on
        those instances), which is exactly the number selection was moved away from.
      PRIMARY (test CSVs): metric='regret', higher_is_better=False -- held-out transfer.

    NOT required to beat hfs_oracle -- the clairvoyant ceiling (it ranks by delta, the
    answer) and unbeatable. Failing does not block export; it warns, because "we shipped
    a model that loses to dfs" must be visible rather than silent.
    """
    if model_score is None:
        return GateResult("beats_baselines", False, "model solved nothing", blocking=False)
    # "lost to k" = baseline is at least as good as the model on this metric.
    def loses_to(v: float) -> bool:
        return (v >= model_score) if higher_is_better else (v <= model_score)
    lost = {k: v for k, v in baselines.items()
            if k not in exclude and v is not None and loses_to(v)}
    beat = sorted(set(baselines) - set(exclude))
    return GateResult(
        "beats_baselines", not lost,
        (f"model {metric} {model_score:.3f}; LOSES TO {lost}" if lost
         else f"model {metric} {model_score:.3f} beats {beat}"),
        blocking=False,   # a weak model is a valid, recordable result -- warn, don't abort
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
