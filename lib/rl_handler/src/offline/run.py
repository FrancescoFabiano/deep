"""The harness: dataset -> train -> telemetry -> selection -> export -> 3 gates.

ONE code path for RL and the baseline. `--model {dqn,cql,two_head}` is the only
difference, so the comparison cannot drift into two harnesses.

REPRESENTATION-AGNOSTIC: nothing here branches on the dataset type.
WITHIN-CONFIG ONLY: `assert_within_config` raises otherwise.
USABLE DATA ONLY: instances absent from `usable_pool.json` never enter.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..models.frontier_policy import FrontierPolicyNetwork
from ..models.two_head_baseline import TwoHeadBaselineNetwork
from ..trainer import RLFrontierTrainer
from .batching import default_device, pack_single
from .dataset import generate_dataset
from .determinism import determinism_report, set_determinism
from .encoder import InstanceCache
from .env import DEFAULT_GAMMA, default_expansion_cap, reference_budget
from .usability import build_usable_pool, fidelity_names, usable_names
from .planner_config import exploitation_for, planner_flags
from .policies import BEHAVIOUR_POLICIES, make_policy
from .qlearning import QTrainer, TrainConfig, default_reward_scale
from .selection import (
    Candidate,
    GateResult,
    assert_within_config,
    gate_beats_baselines,
    gate_env_fidelity,
    heldout_top1,
    onnx_path_for,
    run_planner_expansions,
    select,
    select_smoothed,
    split_instances,
    split_trajectories,
    write_selection_sidecar,
)
from .telemetry import CheckpointRecord, TelemetryWriter, evaluate_split
from .tree import load_tree_instance, partition_solvable

SELECTION_WINDOW = 3       # checkpoints averaged in the smoothed selector
HELDOUT_TRAJ_FRAC = 0.10   # fraction of each instance's rollouts held out for eval

MODELS = ("dqn", "cql", "two_head")


@dataclass
class RunConfig:
    """One (domain, seed, F) cell. `final_launcher.sh` owns the cell iteration;
    `train_models.py` owns domain/seed; this owns one cell."""
    train_csvs: List[Path]
    dir_save_model: Path
    test_csvs: List[Path] = None
    fringe_size: int = 4
    model: str = "dqn"
    kind_of_data: str = "merged"
    context_mode: str = "mean_pool"
    attn_heads: int = 4
    attn_layers: int = 1
    counterfactual: str = "all"
    n_refill_samples: int = 1
    behaviour_policies: Optional[List[str]] = None
    target_sync: int = 500
    max_grad_norm: float = 10.0
    frames: int = 2000
    n_checkpoints: int = 5
    seed: int = 0
    seeds_per_policy: int = 3
    eval_seeds: int = 5
    val_frac: float = 0.34
    val_instances: Optional[List[str]] = None
    # Opt into a cross-configuration (global) split. Default OFF: the hard raise is
    # the right default (see selection.assert_within_config). On HASHED this buys a
    # STRUCTURE-ONLY FLOOR -- the node channel is a hash and carries 0.0% cross-config
    # signal, so any gap comes from topology + edge labels alone. Stamped exploratory.
    allow_cross_config: bool = False
    hidden_dim: int = 64
    gnn_layers: int = 2
    lr: float = 1e-4
    batch_size: int = 64
    cql_alpha: float = 0.0
    gamma: float = DEFAULT_GAMMA    # 0.9999: paper's discounted reward ~= SSP limit
    reward_scale: Optional[float] = None
    eval_expansion_cap: Optional[int] = None
    fidelity_instances: int = 3
    deep_exe: Optional[str] = None
    device: Optional[str] = None
    export_onnx: bool = True
    dataset_type: str = "HASHED"          # opaque; BITMASK needs no change here


def _net(cfg: RunConfig, max_delta: float):
    kw = dict(node_input_dim=1, hidden_dim=cfg.hidden_dim, gnn_layers=cfg.gnn_layers,
              dataset_type=cfg.dataset_type, context_mode=cfg.context_mode,
              use_goal_separate_input=(cfg.kind_of_data == "separated"))
    if cfg.model == "two_head":
        return TwoHeadBaselineNetwork(max_delta=max_delta, **kw)
    return FrontierPolicyNetwork(**kw)


def load_pool(cfg: RunConfig, repo_root: Path) -> tuple[List, Dict[str, InstanceCache], Dict]:
    """Load ONLY usable instances from the CSVs train_models.py handed us.

    The gatekeeper stands between generation and training: a tree whose root cannot
    reach a goal is a WRONG PROBLEM, not noisy data. The shipped discard=0.4 tables
    have their shallow goals deleted, so training on them produces clean,
    self-consistent, meaningless numbers.

    The gate certifies USABILITY (root reaches a goal, tree is non-trivial and
    scorable), NOT that distances are the planner's optimal -- see `usability.py`.
    """
    csvs = [Path(p) for p in cfg.train_csvs]
    missing = [str(p) for p in csvs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing generation tables: {missing}")
    loaded = [load_tree_instance(p, name=p.parent.name, kind_of_data=cfg.kind_of_data)
              for p in csvs]
    solvable, unsolvable = partition_solvable(loaded)
    pool_path = _fringe_dir(cfg).parent / "usable_pool.json"
    pool = (json.loads(pool_path.read_text()) if pool_path.exists()
            else build_usable_pool(solvable, out_path=pool_path))
    keep = set(usable_names(pool))
    insts = [i for i in solvable if i.name in keep]
    if not insts:
        raise ValueError(
            f"no USABLE instance among {[p.parent.name for p in csvs]}. "
            f"Regenerate at --dataset_discard_factor 0 -- the shipped discard=0.4 "
            f"tables have their shallow goals deleted and are a DIFFERENT search "
            f"problem. See usable_pool.json for the per-instance reason."
        )
    caches = {i.name: InstanceCache.from_paths(
        i.state_paths_abs(repo_root),
        cache_file=_fringe_dir(cfg).parent / "cache" / f"{i.name}.pt",
        verbose=False) for i in insts}
    return insts, caches, {"pool": pool, "unsolvable": [u.name for u in unsolvable]}


def _load_test_instances(cfg: RunConfig, repo_root: Path,
                         caches: Dict[str, InstanceCache]) -> List:
    """The PRIMARY eval path: held-out TEST instances (cross-instance transfer).

    Returns the solvable test instances and adds their caches in place. Empty when no
    test CSVs are given -- the FALLBACK (held-out trajectories) then applies. The two
    coexist by design: test-CSV instances get cross-instance transfer eval, the rest
    get trajectory holdout, and the two claims are never blurred.
    """
    if not cfg.test_csvs:
        return []
    csvs = [Path(p) for p in cfg.test_csvs]
    loaded = [load_tree_instance(p, name=p.parent.name, kind_of_data=cfg.kind_of_data)
              for p in csvs if p.exists()]
    solvable, _ = partition_solvable(loaded)
    for i in solvable:
        caches.setdefault(i.name, InstanceCache.from_paths(
            i.state_paths_abs(repo_root),
            cache_file=_fringe_dir(cfg).parent / "cache" / f"{i.name}.pt",
            verbose=False))
    return solvable


def run(cfg: RunConfig, repo_root: Path) -> Dict[str, object]:
    if cfg.model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg.model!r}")
    set_determinism(cfg.seed, strict=True)
    device = cfg.device or default_device()
    run_dir = _fringe_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    insts, caches, meta = load_pool(cfg, repo_root)
    by_name = {i.name: i for i in insts}

    # ---- EVAL DESIGN: PRIMARY (held-out test instances) or FALLBACK (held-out
    #      trajectories). Constraint: no instance is dropped from TRAINING, so we do
    #      NOT split instances into train/val. Every usable instance trains.
    #        PRIMARY  (test CSVs present): coverage/regret on the held-out TEST
    #          instances -> genuine cross-instance TRANSFER; selection on coverage.
    #        FALLBACK (no test CSVs, the common case since test CSVs cannot be
    #          generated everywhere): hold out ~10% of each instance's TRAJECTORIES,
    #          select on held-out-frontier top1-oracle-agreement (a low-variance
    #          ranking signal). coverage/regret is still REPORTED but only as a
    #          train-set rollout estimate -- NOT held out, and NOT the selection
    #          target. The two are never blurred: transfer is claimed only on PRIMARY.
    train_i = insts
    test_i = _load_test_instances(cfg, repo_root, caches)
    for t in test_i:
        by_name.setdefault(t.name, t)
    eval_mode = "test_instances" if test_i else "held_out_trajectories"

    # The guardrail still applies: training ONE model across configurations is the
    # cross-config decision (node ids are fluent-set hashes, 0.0% overlap on HASHED).
    # No instance-level train/val split any more, so the whole training set is the set
    # that must be within-config unless the run opts in.
    all_names = [i.name for i in train_i] + [i.name for i in test_i]
    assert_within_config(all_names, [], allow_cross_config=cfg.allow_cross_config)

    cap = cfg.eval_expansion_cap or default_expansion_cap(insts)
    scale = cfg.reward_scale if cfg.reward_scale is not None else default_reward_scale(train_i)
    max_delta = max((d for i in train_i for d in i.delta if d != float("inf")), default=64.0)

    print(f"[run] F={cfg.fringe_size} model={cfg.model} "
          f"{cfg.kind_of_data}/{cfg.context_mode} device={device} -> {run_dir}")
    print(f"[run] eval_mode={eval_mode}  train_instances={len(train_i)}  "
          f"test_instances={len(test_i)}  cap={cap} scale={scale:.4f}")

    rows, dsum = generate_dataset(train_i, cfg.fringe_size,
                                  policies=cfg.behaviour_policies or BEHAVIOUR_POLICIES,
                                  seeds_per_policy=cfg.seeds_per_policy,
                                  expansion_cap=cap,
                                  counterfactual=cfg.counterfactual,
                                  n_refill_samples=cfg.n_refill_samples)

    # FALLBACK: split the transitions by trajectory so held-out frontiers are never
    # one-step neighbours of trained ones. PRIMARY: all rows train; eval is the
    # test instances.
    if eval_mode == "held_out_trajectories":
        train_rows, heldout_rows, split_manifest = split_trajectories(
            rows, frac=HELDOUT_TRAJ_FRAC, seed=cfg.seed)
        cov_instances = train_i          # coverage rollout is a TRAIN-SET estimate
        print(f"[run] held-out-trajectory split: {split_manifest['n_train_rows']} train "
              f"/ {split_manifest['n_heldout_rows']} eval rows; "
              f"per-instance {split_manifest['per_instance']}")
        if split_manifest["instances_with_no_eval"]:
            print(f"[run] WARNING instances too thin to eval: "
                  f"{split_manifest['instances_with_no_eval']}")
    else:
        train_rows, heldout_rows, split_manifest = rows, [], {"split": "test_instances"}
        cov_instances = test_i           # coverage rollout is genuine held-out transfer

    tel = TelemetryWriter(run_dir / "telemetry.jsonl")

    def rank_for_policy(name, beam, pol):
        return make_policy(by_name[name], pol, seed=0)(list(beam))

    # ---- baselines, reported on their OWN before any comparison. Matched-n with RL:
    #      the SAME held-out frontiers (fallback) or the SAME test instances (primary).
    baselines: Dict[str, Optional[float]] = {}
    baseline_top1: Dict[str, float] = {}
    for b in BEHAVIOUR_POLICIES:
        out = evaluate_split(cov_instances, lambda n, _b=b: make_policy(by_name[n], _b, seed=0),
                             cfg.fringe_size, seeds=cfg.eval_seeds, expansion_cap=cap)
        baselines[b] = out["regret_mean_lower_bound"]
        if eval_mode == "held_out_trajectories":
            t1, _ = heldout_top1(heldout_rows,
                                 lambda n, beam, _b=b: rank_for_policy(n, beam, _b), by_name)
            baseline_top1[b] = t1
            out["heldout_top1"] = t1
        tel.append(CheckpointRecord(step=-1, frames=0, split=f"baseline:{b}", payload=out))
    print(f"[run] baselines regret={baselines}"
          + (f"  heldout_top1={baseline_top1}" if baseline_top1 else ""))

    net = _net(cfg, max_delta).to(device)
    if cfg.model == "two_head":
        from .baseline import train_two_head_baseline
        beams = []
        seen = set()
        for r in train_rows:                 # held-out frontiers must not train the baseline either
            if r.forced:
                continue
            k = (r.instance, tuple(r.obs))
            if k not in seen:
                seen.add(k)
                beams.append((r.instance, r.obs))
        net, _ = train_two_head_baseline(train_i, caches, beams, cfg.fringe_size,
                                         steps=cfg.frames, device=device, seed=cfg.seed,
                                         hidden_dim=cfg.hidden_dim,
                                         context_mode=cfg.context_mode)
        trainer = None
    else:
        # QTrainer sees ONLY train_rows -- the held-out trajectories never enter training.
        trainer = QTrainer(net, _net(cfg, max_delta), train_i, caches, train_rows,
                           TrainConfig(fringe_size=cfg.fringe_size, gamma=cfg.gamma,
                                       lr=cfg.lr, batch_size=cfg.batch_size,
                                       model=cfg.model, cql_alpha=cfg.cql_alpha,
                                       reward_scale=scale, expansion_cap=cap,
                                       seed=cfg.seed, device=device))

    def score_for(name, beam):
        p = pack_single(caches[name], beam, len(beam), device)
        with torch.no_grad():
            return net(node_features=p["node_features"], edge_index=p["edge_index"],
                       edge_attr=p["edge_attr"], membership=p["membership"],
                       candidate_batch=None, mask=p["mask"]).cpu().tolist()

    def policy_for(name):
        return lambda beam: sorted(range(len(beam)), key=lambda k: -score_for(name, beam)[k])

    def rl_rank(name, beam):                 # the RL ranker, for heldout_top1
        return policy_for(name)(list(beam))

    # ---- train with checkpoints ----
    cands: List[Candidate] = []
    every = max(1, cfg.frames // cfg.n_checkpoints)
    for step in range(1, cfg.frames + 1):
        if trainer is not None:
            log = trainer.step()
        if step % every == 0 or step == cfg.frames:
            # coverage/regret: held-out transfer on PRIMARY, train-set estimate on
            # FALLBACK (labelled in the sidecar -- never reported as transfer there).
            out = evaluate_split(cov_instances, policy_for, cfg.fringe_size,
                                 seeds=cfg.eval_seeds, expansion_cap=cap, score_for=score_for)
            if trainer is not None:
                out.update({k: log[k] for k in ("td_loss", "q_mean", "q_max", "grad_norm", "lr")})
                out.update(trainer.q_vs_qstar(train_rows[:200]))
            out["dataset"] = dsum
            out["coverage_is_transfer"] = (eval_mode == "test_instances")

            # THE SELECTION SIGNAL. FALLBACK: held-out-frontier top1 (low variance,
            # leak-free, matched-n with baselines). PRIMARY: coverage on held-out
            # instances. Smoothed over a window at select time -- never argmax.
            if eval_mode == "held_out_trajectories":
                t1, n_ho = heldout_top1(heldout_rows, rl_rank, by_name)
                out["heldout_top1"] = t1
                out["heldout_n"] = n_ho
                sel_score = t1
            else:
                sel_score = out["coverage_at_reference_budget"]

            tel.append(CheckpointRecord(step=step, frames=step * cfg.batch_size,
                                        split="val", payload=out))
            cands.append(Candidate(step=step, frames=step * cfg.batch_size,
                                   coverage=out["coverage_at_reference_budget"],
                                   regret=out["regret_mean_lower_bound"],
                                   doom=out["doom_rate"], select_score=sel_score))
            torch.save(net.state_dict(), run_dir / "checkpoints" / f"ckpt_{step}.pt"
                       if (run_dir / "checkpoints").exists()
                       else _mk(run_dir / "checkpoints") / f"ckpt_{step}.pt")
            sig = (f"heldout_top1={out['heldout_top1']:.3f}"
                   if eval_mode == "held_out_trajectories"
                   else f"coverage={out['coverage_at_reference_budget']:.2f}")
            print(f"[run] step {step:5d} {sig} "
                  f"regret={out['regret_mean_lower_bound']} auc={out.get('viability_auc')}")

    # Smoothed selection on the held-out signal -- NOT a single argmax draw (the floor
    # run showed argmax picks a lucky rollout).
    best = select_smoothed(cands, window=SELECTION_WINDOW)
    print(f"[run] selected checkpoint {best.step} "
          f"(select_score={best.select_score:.3f}, coverage {best.coverage:.2f}, "
          f"regret {best.regret}) via {eval_mode}, window={SELECTION_WINDOW}")
    net.load_state_dict(torch.load(run_dir / "checkpoints" / f"ckpt_{best.step}.pt"))

    # ---- export ONLY the selected checkpoint ----
    gates = []
    # THE CONTRACT: train_models.py copies this to
    # <exp_dir>/_models/<domain>/frontier_policy_<F>.onnx, which
    # bulk_coverage_run.py reads and the C++ checks logits-length against.
    onnx = run_dir / f"frontier_policy_{cfg.fringe_size}_best_by_expansions.onnx"
    if cfg.export_onnx:
        # EXPORT A COPY. RLFrontierTrainer.__init__ does `self.model = model.to(device)`
        # and nn.Module.to() mutates IN PLACE, so passing the live `net` with
        # device="cpu" permanently moved the training model off the GPU. The very next
        # block (the env-fidelity gate) calls score_for, which still packs to `device`
        # -- so every run that reached the export died with "mat1 is on cuda:0,
        # different from other tensors on cpu". --deep-exe defaults to the deep binary,
        # so the gate is armed by default and this fired on ANY complete run: gate 2 had
        # never once executed. Export must not mutate the training model.
        RLFrontierTrainer(model=copy.deepcopy(net), device="cpu",
                          kind_of_data=cfg.kind_of_data).to_onnx(
            onnx, node_input_dim=1, onnx_frontier_size=cfg.fringe_size)
        print(f"[run] exported {onnx}")

        # GATE 2 -- only instances the gate can actually SCORE
        fid = [n for n in fidelity_names(meta["pool"]) if n in by_name][:cfg.fidelity_instances]
        if cfg.deep_exe and fid:
            off, live = [], []
            for n in fid:
                out = evaluate_split([by_name[n]], policy_for, cfg.fringe_size, seeds=1,
                                     expansion_cap=cap)
                off.append(int(out["expansions_mean"] or 0))
                live.append(run_planner_expansions(
                    cfg.deep_exe, _problem_for(repo_root, n), onnx,
                    cfg.fringe_size, separated=(cfg.kind_of_data == "separated"),
                    repo_root=repo_root))
            gates.append(gate_env_fidelity(off, live))
        else:
            # armed=False: the gate did not RUN. Not a verdict, so it must not fail
            # the run -- planner deployment is out of scope and --deep-exe defaults
            # to None, so this is the EXPECTED state, not a problem.
            gates.append(GateResult(
                "env_fidelity", False,
                "NOT ARMED: no --deep-exe given (planner deployment out of scope; "
                "the RL-vs-baseline claim is made IN THE OFFLINE ENV)" if not cfg.deep_exe
                else "NOT ARMED: no usable instance reaches 20 expansions",
                armed=False))
        gates.append(gate_beats_baselines(best.regret, baselines))
        for g in gates:
            verdict = "PASS" if g.passed else ("FAIL" if g.armed else "SKIP")
            print(f"[gate] {g.name}: {verdict} -- {g.detail}")

        write_selection_sidecar(
            onnx, best,
            train_instances=[i.name for i in train_i],
            val_instances=[i.name for i in test_i],
            fringe_size=cfg.fringe_size, kind_of_data=cfg.kind_of_data,
            model=cfg.model, gamma=cfg.gamma, reward_scale=scale,
            baselines=baselines, gates=gates,
            excluded_unsolvable=meta["pool"]["excluded"],
            extra={"determinism": determinism_report(),
                   "context_mode": cfg.context_mode,
                   "dataset_type": cfg.dataset_type,
                   "eval_mode": eval_mode,
                   "selection_window": SELECTION_WINDOW,
                   "selection_metric": ("heldout_top1"
                                        if eval_mode == "held_out_trajectories"
                                        else "coverage_at_reference_budget"),
                   "split_manifest": split_manifest,
                   "baseline_heldout_top1": baseline_top1 or None,
                   "coverage_note": (
                       "coverage/regret is a TRAIN-SET rollout estimate (rolled from "
                       "root on trained instances), NOT held out. Selection used "
                       "held-out-trajectory top1. Transfer is NOT claimed here."
                       if eval_mode == "held_out_trajectories"
                       else "coverage/regret is held-out cross-instance TRANSFER "
                            "(test CSVs)."),
                   "cross_config": cfg.allow_cross_config,
                   "exploratory": cfg.allow_cross_config,
                   "cross_config_note": (
                       "Cross-configuration (global) run. On HASHED the node channel "
                       "is a hash and carries 0.0% cross-config signal, so any gap "
                       "here comes from TOPOLOGY + EDGE LABELS alone -- a "
                       "STRUCTURE-ONLY FLOOR, not a transfer result."
                   ) if cfg.allow_cross_config else None,
                   "refill_note": ("The offline env models refill as its own "
                                   "stochastic transition; it does not pin a C++ "
                                   "RefillMode. Whoever deploys should confirm the "
                                   "planner's refill matches their intent.")})
    return {"selected": best, "baselines": baselines, "gates": gates, "run_dir": run_dir}


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fringe_dir(cfg: RunConfig) -> Path:
    """`_fringe<F>` appended to --dir-save-model: the contract train_models.py
    installs from (`<dir>_fringe<F>/frontier_policy_<F>_best_by_expansions.onnx`)."""
    d = Path(cfg.dir_save_model)
    return d.with_name(d.name + f"_fringe{cfg.fringe_size}")


def _problem_for(repo_root: Path, instance: str) -> Path:
    for c in repo_root.glob(f"exp/rl_exp/*/*/Training/{instance}.txt"):
        return c
    for c in repo_root.glob(f"exp/all/**/{instance}.txt"):
        return c
    raise FileNotFoundError(f"no problem file for {instance}")
