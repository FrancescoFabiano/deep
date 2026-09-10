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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..models.frontier_policy import FrontierPolicyNetwork
from ..models.two_head_baseline import TwoHeadBaselineNetwork
from ..trainer import RLFrontierTrainer
from .batching import assert_goal_mode_consistent, default_device, pack_single
from .encoder import load_goal_graph
from .dataset import generate_dataset
from .determinism import determinism_report, set_determinism
from .encoder import InstanceCache
from .env import DEFAULT_GAMMA, default_expansion_cap, reference_budget
from .frozen_eval import (
    FROZEN_SEED_OFFSET,
    build_frozen_fringes,
    save_frozen,
    training_overlap,
)
from .usability import build_usable_pool, fidelity_names, pool_names, usable_names
from .planner_config import exploitation_for, planner_flags
from .policies import (
    BEHAVIOUR_POLICIES,
    TRACE_POLICY,
    TRACE_PREFIX,
    make_policy,
    trace_policies_of,
)
from .strategies import STRATEGIES, dir_name
from .unify import UnifiedInstance, report_unified, unify_instances
from .qlearning import QTrainer, TrainConfig, default_reward_scale
from .selection import (
    Candidate,
    GateResult,
    assert_within_config,
    gate_beats_baselines,
    gate_env_fidelity,
    heldout_ranking_metrics,
    heldout_ranking_micro_macro,
    heldout_top1,
    onnx_path_for,
    rankable_slots,
    run_planner_expansions,
    run_planner_metrics,
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
# Bump when the telemetry gains a new per-checkpoint metric. regenerate_figures.py
# reads it to detect a run that predates a metric and trigger the self-validated
# backfill (rather than silently emitting a blank figure). 1: coverage/regret only.
# 2: +heldout_top1/train_top1. 3: +return_mean +full ranking family (ndcg/js/...).
# 4: +macro/per-instance NDCG + effective_instance_count (Fix 3) + per-policy agreement
#    top1/tau-b vs bfs/dfs/hfs/random (Fix 2).
# 5: +eval_fringes ("held_out_trajectories" | "frozen_random") + unified stamp. In
#    unified mode the heldout_* fields are measured on the FROZEN random-fringe set
#    (frozen_eval.py), not on held-out trajectories -- same metric family, same keys,
#    different fringes; the stamp says which.
METRICS_SCHEMA = 5

EVAL_HELDOUT = "held_out_trajectories"
EVAL_FROZEN = "frozen_random_fringes"
EVAL_TEST = "test_instances"
# Fix 2: fixed tie-break seed for the policy-agreement metrics. Every behaviour policy
# uses the mandatory random tie-break sigma~=(sigma,u); pinning the seed keeps the
# agreement CURVE from carrying tie-break noise. Recorded in telemetry per checkpoint.
AGREEMENT_TIEBREAK_SEED = 12345

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
    # BUDGET IS EPOCHS, NOT STEPS. `--frames` was the step count, so the same
    # number meant a different number of passes over the data at every F (100k
    # frames = 321 epochs at F=4 but 44 at F=32, because the dataset grows with
    # F). Epochs fix the comparison: S is derived per run, after the held-out
    # split, from the size of the pool the trainer actually sees.
    epochs: float = 100.0
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
    # Draw distribution (see qlearning.TrainConfig). Default is the historical
    # behaviour, so the ablation runs both arms from one binary.
    sampler: str = "proportional"
    sampler_k: float = 12.0
    # Export the FINAL checkpoint (step S) instead of the one the selector picks.
    # For controlled comparisons: two arms must be scored at the same step, and
    # the smoothed selector runs on heldout_ndcg, which is a retired signal.
    select_final: bool = False
    # UNIFIED mode (unify.py): merge every strategy's tree of one problem into ONE
    # graph of content-unique states (delta recomputed on the union), train on
    # EVERY trajectory of EVERY behaviour policy (`trace:<s>` per strategy), and
    # evaluate the ranking metrics on a FROZEN set of random-policy fringes drawn
    # once (frozen_eval.py). Off = the per-strategy trees + held-out trajectories.
    unified: bool = False
    frozen_eval_m: int = 128                # frozen fringes per instance (fewer if fewer exist)
    frozen_eval_rollouts: int = 128         # random rollouts pooled per instance
    frozen_eval_seed: Optional[int] = None  # default: seed + FROZEN_SEED_OFFSET
    # FILL the non-full beams (dataset.py module docstring): from every non-full
    # decision state of a behaviour rollout, fill_k extra rollouts are grown (random
    # expansions, no rows) to a full beam and then rolled under the same behaviour
    # policy (rows, flagged filled=True). Off = the parent rollouts only, byte-
    # identical to before. Meant for unified graphs, where every strategy's
    # expansions are available to the growth.
    fill_fringes: bool = False
    fill_k: int = 4


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
    # One tree per (instance, strategy): the loader names it `<inst>@<strat>` from
    # the table's location, so two strategies of one problem never collide.
    loaded = [load_tree_instance(p, kind_of_data=cfg.kind_of_data) for p in csvs]
    _report_trees(loaded)
    if cfg.unified:
        loaded = _unify(cfg, loaded, repo_root)
    solvable, unsolvable = partition_solvable(loaded)
    # The unified pool is keyed by `<inst>@unified` names and judged on the union
    # graph, so it gets its own file: the per-strategy pool the launcher's gate 1c
    # writes stays valid for non-unified runs of the same batch.
    pool_path = _fringe_dir(cfg).parent / (
        "usable_pool_unified.json" if cfg.unified else "usable_pool.json")
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else None
    if pool is not None and not {i.name for i in loaded} <= set(pool_names(pool)):
        # A pool written for a different tree set (a pre-strategy run keyed by bare
        # instance names, or a batch that since gained a strategy) would silently
        # empty the training set. Rebuild instead of trusting it.
        print(f"[run] usable_pool.json at {pool_path} does not cover the loaded trees "
              f"-- rebuilding it")
        pool = None
    if pool is None:
        pool = build_usable_pool(solvable, out_path=pool_path)
    keep = set(usable_names(pool))
    insts = [i for i in solvable if i.name in keep]
    if not insts:
        raise ValueError(
            f"no USABLE tree among {[i.name for i in loaded]}. "
            f"Regenerate at --dataset_discard_factor 0 -- the shipped discard=0.4 "
            f"tables have their shallow goals deleted and are a DIFFERENT search "
            f"problem. See usable_pool.json for the per-instance reason."
        )
    caches = {i.name: InstanceCache.from_paths(
        i.state_paths_abs(repo_root),
        cache_file=_fringe_dir(cfg).parent / "cache" / f"{i.name}.pt",
        verbose=False) for i in insts}
    goals = _load_goals(cfg, csvs, insts)
    return insts, caches, {"pool": pool, "unsolvable": [u.name for u in unsolvable],
                           "goals": goals}


def _load_goals(cfg: RunConfig, csvs, insts) -> Optional[Dict[str, object]]:
    """S2 -- validate mode against data at load time. Separated: every usable instance
    MUST have a readable ``goal_tree.dot`` beside its state CSV (same dir); a missing one
    FAILS LOUDLY with the instance name + expected path rather than falling back to None
    (that silent fallback is exactly the bug this change removes). Merged: no goal is
    loaded and None is returned.

    Data-layout difference: separated generation (launcher ``--no_goal`` / GEN_FLAG) emits
    the goal as a SEPARATE per-instance ``goal_tree.dot``; merged folds the goal into each
    state graph, so no separate goal artifact is expected.
    """
    if cfg.kind_of_data != "separated":
        return None
    goals: Dict[str, object] = {}
    for i in insts:
        # Beside THIS tree's CSV: two strategies of one instance live in two folders,
        # each with its own goal_tree.dot, so the lookup keys on the tree, not the
        # instance name.
        gp = Path(i.csv_path).parent / "goal_tree.dot"
        if not gp.exists():
            raise FileNotFoundError(
                f"separated mode (kind_of_data=separated) requires a goal_tree.dot for "
                f"every usable instance, but it is missing for {i.name!r}: expected "
                f"{gp}. Regenerate the data in separated mode, or run merged. "
                f"(Refusing to fall back to a goal-less run -- that was the bug.)"
            )
        goals[i.name] = load_goal_graph(gp)
    print(f"[run] separated: loaded goal_tree.dot for {len(goals)} instances (goal is "
          f"threaded through training, target net, and eval)")
    return goals


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
    loaded = [load_tree_instance(p, kind_of_data=cfg.kind_of_data)
              for p in csvs if p.exists()]
    if cfg.unified and loaded:
        loaded = _unify(cfg, loaded, repo_root)
    solvable, _ = partition_solvable(loaded)
    for i in solvable:
        caches.setdefault(i.name, InstanceCache.from_paths(
            i.state_paths_abs(repo_root),
            cache_file=_fringe_dir(cfg).parent / "cache" / f"{i.name}.pt",
            verbose=False))
    return solvable


def _unify(cfg: RunConfig, trees: List, repo_root: Path) -> List[UnifiedInstance]:
    """Per-strategy trees -> one unified graph per problem (unify.py). Fingerprints
    are cached per tree under the domain's cache dir; the merge report is printed
    here and travels to the sidecar via each instance's `manifest`."""
    out = unify_instances(trees, repo_root,
                          cache_dir=_fringe_dir(cfg).parent / "cache" / "fp")
    print(f"[unify] {len(trees)} tree(s) -> {len(out)} unified graph(s)")
    print(report_unified(out))
    return out


def run(cfg: RunConfig, repo_root: Path) -> Dict[str, object]:
    if cfg.model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg.model!r}")
    set_determinism(cfg.seed, strict=True)
    device = cfg.device or default_device()
    run_dir = _fringe_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    insts, caches, meta = load_pool(cfg, repo_root)
    by_name = {i.name: i for i in insts}
    goals = meta["goals"]   # {name: goal StateGraph} in separated mode, else None

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
    if goals is not None and test_i:
        # separated PRIMARY path: test instances score through score_for too, so they
        # need goals as well (same FAIL-LOUD contract as train).
        goals.update(_load_goals(cfg, list(cfg.test_csvs or []), test_i))
    for t in test_i:
        by_name.setdefault(t.name, t)
    #        UNIFIED (cfg.unified): every trajectory trains; the ranking metrics and
    #          selection run on the FROZEN random-fringe set (frozen_eval.py).
    #          coverage/regret is rolled on the test instances when given (transfer)
    #          and on the train instances otherwise (train-set estimate).
    if cfg.unified:
        eval_mode = EVAL_FROZEN
    else:
        eval_mode = EVAL_TEST if test_i else EVAL_HELDOUT
    ranking_eval = eval_mode in (EVAL_HELDOUT, EVAL_FROZEN)

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

    # Behaviour policies for the dataset. Default `trace`: each tree is rolled out
    # under the expansion order of the search that generated it (its pi_b). Every
    # loaded table carries a trace (DOT names are creation-indexed); a tree without
    # one is a data defect and make_policy raises with the tree's name.
    behaviour = list(cfg.behaviour_policies or (TRACE_POLICY,))
    if cfg.fill_fringes and not cfg.unified:
        print("[run] WARNING --fill-fringes on PER-STRATEGY trees: a growth expansion "
              "of a censored leaf is a fabricated dead end (only the rows expanding "
              "such leaves are dropped, the grown beam itself stays). The fill is "
              "designed for --unified graphs.")
    print(f"[run] behaviour policies={behaviour}  strategies="
          f"{sorted({_strat_label(i) for i in train_i if i.strategy})}"
          + ("  (unified: `trace` expands to "
             f"{sorted({p for i in train_i for p in trace_policies_of(i)})})"
             if cfg.unified else ""))
    rows, dsum = generate_dataset(train_i, cfg.fringe_size,
                                  policies=behaviour,
                                  seeds_per_policy=cfg.seeds_per_policy,
                                  expansion_cap=cap,
                                  counterfactual=cfg.counterfactual,
                                  n_refill_samples=cfg.n_refill_samples,
                                  gamma=cfg.gamma,
                                  fill_fringes=cfg.fill_fringes,
                                  fill_k=cfg.fill_k)

    # H2: per-instance ROW SHARE at assembly. One instance owning 48-60% of the gradient
    # (batch1 pl_7) must be VISIBLE in the log, not require forensics.
    from collections import Counter as _Counter
    _rc = _Counter(r.instance for r in rows)
    _ntot = max(1, len(rows))
    print(f"[run] dataset row shares ({len(rows)} rows over {len(_rc)} instances):")
    for _name, _c in sorted(_rc.items(), key=lambda kv: -kv[1]):
        _flag = "   <-- DOMINATES" if _c / _ntot >= 0.40 else ""
        print(f"[run]   {_name:24} {_c:>8} rows  {100 * _c / _ntot:5.1f}%{_flag}")

    # FALLBACK: split the transitions by trajectory so held-out frontiers are never
    # one-step neighbours of trained ones. PRIMARY: all rows train; eval is the
    # test instances.
    if eval_mode == EVAL_HELDOUT:
        train_rows, heldout_rows, split_manifest = split_trajectories(
            rows, frac=HELDOUT_TRAJ_FRAC, seed=cfg.seed)
        cov_instances = train_i          # coverage rollout is a TRAIN-SET estimate
        print(f"[run] held-out-trajectory split: {split_manifest['n_train_rows']} train "
              f"/ {split_manifest['n_heldout_rows']} eval rows; "
              f"per-instance {split_manifest['per_instance']}")
        if split_manifest["instances_with_no_eval"]:
            print(f"[run] WARNING instances too thin to eval: "
                  f"{split_manifest['instances_with_no_eval']}")
    elif eval_mode == EVAL_FROZEN:
        # EVERY trajectory trains. The ranking metrics run on M random-policy
        # fringes per instance, drawn ONCE here from a pinned seed and written to
        # disk, so every checkpoint (and every run sharing the seed) is scored on
        # the same beams.
        # WHICH graphs the random rollouts run on (2026-09-10): the unified TEST
        # graphs when test tables were generated (cross-instance: the frozen fringes
        # then come from problems the model never trains on), else the unified
        # TRAIN graphs (same problems; the overlap below says how much is shared).
        fseed = (cfg.frozen_eval_seed if cfg.frozen_eval_seed is not None
                 else cfg.seed + FROZEN_SEED_OFFSET)
        frozen_src = test_i if test_i else train_i
        heldout_rows, fmanifest = build_frozen_fringes(
            frozen_src, cfg.fringe_size, seed=fseed, expansion_cap=cap,
            m_per_instance=cfg.frozen_eval_m,
            rollouts_per_instance=cfg.frozen_eval_rollouts, gamma=cfg.gamma)
        fmanifest["source"] = "test_instances" if test_i else "train_instances"
        fmanifest["source_instances"] = sorted(i.name for i in frozen_src)
        fmanifest["training_overlap"] = training_overlap(heldout_rows, rows)
        save_frozen(run_dir / "frozen_eval.json", heldout_rows, fmanifest)
        train_rows = rows
        split_manifest = {"split": EVAL_FROZEN, "n_train_rows": len(rows),
                          "n_heldout_rows": len(heldout_rows),
                          "frozen_eval": {k: v for k, v in fmanifest.items()
                                          if k != "per_instance"},
                          "per_instance": fmanifest["per_instance"],
                          "instances_with_no_eval": fmanifest["instances_with_none"]}
        cov_instances = test_i or train_i
        ov = fmanifest["training_overlap"]
        print(f"[run] frozen eval set: {len(heldout_rows)} fringes over "
              f"{fmanifest['n_instances']} {fmanifest['source'].replace('_', ' ')} "
              f"(seed {fseed}, M={cfg.frozen_eval_m}, "
              f"{cfg.frozen_eval_rollouts} random rollouts each) -> "
              f"{run_dir / 'frozen_eval.json'}")
        print(f"[run] frozen-vs-training overlap: exact beam "
              f"{100 * ov['exact_beam_match_frac']:.1f}%, beam-as-set "
              f"{100 * ov['beam_as_set_match_frac']:.1f}%, states seen "
              f"{100 * (ov['state_seen_in_training_frac'] or 0):.1f}%")
        if fmanifest["instances_with_none"]:
            print(f"[run] WARNING instances with NO frozen fringe: "
                  f"{fmanifest['instances_with_none']}")
    else:
        train_rows, heldout_rows, split_manifest = rows, [], {"split": EVAL_TEST}
        cov_instances = test_i           # coverage rollout is genuine held-out transfer
    if ranking_eval:
        # H3: instances_with_no_eval counts TRAJECTORIES, but a metric-blind instance can
        # get eval trajectories yet contribute ZERO frontiers that survive the ranking
        # filter (len>=2, not-all-INF, deduped). Count SCORABLE frontiers and warn loudly
        # -- this is what silently hid pl_3 / CC_3_3_3__pl_4 (0 scorable) behind a '[]'.
        _scorable = {i.name: 0 for i in train_i}
        _seen_fr: set = set()
        for _r in heldout_rows:
            if getattr(_r, "forced", False):
                continue
            _k = (_r.instance, tuple(_r.obs))
            if _k in _seen_fr:
                continue
            _seen_fr.add(_k)
            # Censored-aware, mirroring selection.rankable_slots exactly: a
            # frontier is scorable iff >=2 NON-CENSORED slots survive and not
            # all of them are INF.
            _it = by_name[_r.instance]
            _keep = rankable_slots(_it, _r.obs)
            _d = [_it.delta[_r.obs[k]] for k in _keep]
            if len(_keep) >= 2 and not all(x >= float("inf") for x in _d):
                _scorable[_r.instance] = _scorable.get(_r.instance, 0) + 1
        _blind = sorted(n for n, c in _scorable.items() if c == 0)
        split_manifest["instances_with_no_scorable_frontier"] = _blind
        if _blind:
            print(f"[run] WARNING {len(_blind)} instance(s) contribute ZERO scorable "
                  f"held-out frontiers (invisible to the ranking metric, no matter the "
                  f"aggregation): {_blind}")

    tel = TelemetryWriter(run_dir / "telemetry.jsonl")

    def rank_for_policy(name, beam, pol):
        return make_policy(by_name[name], pol, seed=0)(list(beam))

    # ---- baselines, reported on their OWN before any comparison. Matched-n with RL:
    #      the SAME held-out frontiers (fallback) or the SAME test instances (primary).
    baselines: Dict[str, Optional[float]] = {}
    baseline_top1: Dict[str, float] = {}
    baseline_ndcg: Dict[str, float] = {}
    # The synthetic rankings are the reference points; `trace` (the generating
    # search replayed) joins them when every evaluated tree carries a trace -- it is
    # the "what pi_b itself does under this F" number the RL must beat to matter.
    if cfg.unified:
        # UNIFIED: the comparison is model vs the clairvoyant oracle on the SAME
        # frozen fringes, nothing else -- no synthetic bfs/dfs/random rankers and no
        # trace rollouts (2026-09-10). The oracle is the ceiling, not a competitor.
        baseline_policies = ["hfs_oracle"]
    else:
        baseline_policies = list(BEHAVIOUR_POLICIES)
    if not cfg.unified and cov_instances and all(i.has_trace for i in cov_instances):
        # per-strategy trees: `trace`; unified graphs: every `trace:<s>` that ALL
        # evaluated graphs carry (a strategy generated for one problem only would
        # otherwise raise inside make_policy on the others).
        common = set(trace_policies_of(cov_instances[0]))
        for i in cov_instances[1:]:
            common &= set(trace_policies_of(i))
        baseline_policies += [p for p in trace_policies_of(cov_instances[0]) if p in common]
    # The trace of a behaviour policy ranks a beam by "which of these did THAT
    # search expand first". On a FROZEN random fringe most nodes were never on that
    # search's path, so the ranking is mostly ties: it is not a comparator there and
    # is left out of the ranking table (it keeps its rollout baseline).
    ranking_baselines = [b for b in baseline_policies
                         if not (eval_mode == EVAL_FROZEN and b.startswith(TRACE_PREFIX))]
    for b in baseline_policies:
        out = evaluate_split(cov_instances, lambda n, _b=b: make_policy(by_name[n], _b, seed=0),
                             cfg.fringe_size, seeds=cfg.eval_seeds, expansion_cap=cap,
                             gamma=cfg.gamma)
        baselines[b] = out["regret_mean_lower_bound"]
        if ranking_eval and b in ranking_baselines:
            # the FULL ranking-metric family, matched-n on the held-out frontiers
            # (baselines expose a ranking, so no softmax divergence for them).
            rm = heldout_ranking_metrics(
                heldout_rows, by_name,
                rank_for=lambda n, beam, _b=b: rank_for_policy(n, beam, _b))
            baseline_top1[b] = rm["top1"]
            baseline_ndcg[b] = rm["ndcg"]
            out.update({f"heldout_{k}": rm[k] for k in
                        ("top1", "ndcg", "regret_at_decision", "picked_dead", "kendall_tau")})
        tel.append(CheckpointRecord(step=-1, frames=0, split=f"baseline:{b}", payload=out))
    print(f"[run] baselines regret={baselines}"
          + (f"  top1={baseline_top1}  ndcg={baseline_ndcg}" if baseline_top1 else ""))

    # ---- budget: epochs -> steps, once N is known ----
    # N is |train_rows|, i.e. AFTER the held-out split: the held-out trajectories
    # never enter training, so counting them would overstate the budget.
    N_train = len(train_rows)
    S = int(math.ceil(cfg.epochs * N_train / cfg.batch_size))
    D = S * cfg.batch_size
    print(f"[budget] epochs={cfg.epochs:g}  N={N_train:,}  batch_size={cfg.batch_size}"
          f"  S={S:,} steps  D={D:,} draws")

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
                                         steps=S, device=device, seed=cfg.seed,
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
                                       seed=cfg.seed, device=device,
                                       sampler=cfg.sampler, sampler_k=cfg.sampler_k),
                           goals=goals)
        print(trainer.allocation_report(D))

    def score_for(name, beam):
        goal_graph = goals[name] if goals is not None else None
        assert_goal_mode_consistent(net, goal_graph is not None)
        p = pack_single(caches[name], beam, len(beam), device, goal_graph=goal_graph)
        with torch.no_grad():
            return net(node_features=p["node_features"], edge_index=p["edge_index"],
                       edge_attr=p["edge_attr"], membership=p["membership"],
                       candidate_batch=None, mask=p["mask"],
                       goal_node_features=p.get("goal_node_features"),
                       goal_edge_index=p.get("goal_edge_index"),
                       goal_edge_attr=p.get("goal_edge_attr"),
                       goal_batch=p.get("goal_batch")).cpu().tolist()

    def policy_for(name):
        return lambda beam: sorted(range(len(beam)), key=lambda k: -score_for(name, beam)[k])

    def rl_rank(name, beam):                 # the RL ranker, for heldout_top1
        return policy_for(name)(list(beam))

    # ---- train with checkpoints ----
    cands: List[Candidate] = []
    # EXACT count: `S // n_checkpoints` is integer division, so it overshot
    # whenever n_checkpoints did not divide the budget (100k/7 gave 8 records,
    # 100k/3 gave 4). With S derived from epochs the divisibility essentially
    # never holds, so the miss would become the norm. Enumerate the steps
    # instead: exactly n_checkpoints of them (modulo collisions at tiny S), and
    # the last is exactly S.
    ckpt_steps = sorted({max(1, round(i * S / cfg.n_checkpoints))
                         for i in range(1, cfg.n_checkpoints + 1)})
    # Progress bar: only when attached to a terminal. Through train_models.run_one the
    # child is piped (not a tty), so tqdm disables itself -- no \r spam in the logs;
    # the per-checkpoint prints stream instead (run_one runs the child with -u).
    import sys as _sys
    from tqdm import tqdm
    # The bar reports EPOCHS, the unit the budget is now expressed in; the loop
    # is still over steps and the checkpoint line below still reports steps and
    # draws, so train_models.py's parser is unaffected.
    bar = tqdm(total=float(cfg.epochs), desc=f"train F={cfg.fringe_size}",
               unit="epoch", disable=not _sys.stderr.isatty(),
               dynamic_ncols=True, leave=False)
    _ckpt_set = set(ckpt_steps)
    import time as _time
    t_train, t_eval = 0.0, 0.0     # wall split: trainer.step vs per-checkpoint eval
    for step in range(1, S + 1):
        if trainer is not None:
            _t0 = _time.perf_counter()
            log = trainer.step()
            t_train += _time.perf_counter() - _t0
        bar.update(cfg.batch_size / N_train)
        if step in _ckpt_set:
            _te = _time.perf_counter()
            # coverage/regret: held-out transfer on PRIMARY, train-set estimate on
            # FALLBACK (labelled in the sidecar -- never reported as transfer there).
            out = evaluate_split(cov_instances, policy_for, cfg.fringe_size,
                                 seeds=cfg.eval_seeds, expansion_cap=cap,
                                 score_for=score_for, gamma=cfg.gamma)
            if trainer is not None:
                # materialise the on-device tensors HERE (at the checkpoint), not
                # every step -- this is the only place the values are read.
                out.update({k: float(log[k]) for k in ("td_loss", "q_mean", "q_max", "grad_norm")})
                out["lr"] = log["lr"]
                out.update(trainer.q_vs_qstar(train_rows[:200]))
            out["dataset"] = dsum
            out["coverage_is_transfer"] = bool(test_i)
            out["metrics_schema"] = METRICS_SCHEMA
            out["eval_fringes"] = ("frozen_random" if eval_mode == EVAL_FROZEN
                                   else eval_mode)
            out["unified"] = bool(cfg.unified)

            # THE SELECTION SIGNAL. FALLBACK: held-out-frontier top1 (low variance,
            # leak-free, matched-n with baselines). PRIMARY: coverage on held-out
            # instances. Smoothed over a window at select time -- never argmax.
            if ranking_eval:
                # FULL ranking-metric family for the model (logits enable the softmax
                # divergence). top1 asks only "best first?"; ndcg/js read the WHOLE
                # ordering, which is what DISCARD needs (it removes the worst kappa).
                # ONE pass -> pooled (micro, the selector) + per-instance + macro (Fix 3)
                # + per-policy agreement (Fix 2, fixed tie-break seed).
                _agree_rankers = {
                    p: (lambda name, beam, _p=p: make_policy(
                        by_name[name], _p, seed=AGREEMENT_TIEBREAK_SEED)(list(beam)))
                    for p in ranking_baselines
                }
                rmm = heldout_ranking_micro_macro(
                    heldout_rows, by_name, logits_for=score_for,
                    agreement_rankers=_agree_rankers)
                rm = rmm["micro"]     # selection stays on MICRO -- unchanged
                out["heldout_n"] = rm["n"]
                for k in ("top1", "ndcg", "js", "regret_at_decision", "picked_dead",
                          "kendall_tau"):
                    out[f"heldout_{k}"] = rm[k]
                out["heldout_top1"] = rm["top1"]      # kept for back-compat / selection
                # Fix 3 (RECORD only, do NOT select on these):
                out["heldout_ndcg_macro"] = rmm["ndcg_macro"]
                out["heldout_top1_macro"] = rmm["top1_macro"]
                out["heldout_per_instance_ndcg"] = {i: v["ndcg"]
                                                    for i, v in rmm["per_instance"].items()}
                out["effective_instance_count"] = rmm["effective_instance_count"]
                # Fix 2: per-policy agreement (top1 + tau-b), RECORD only.
                out["agree_tiebreak_seed"] = AGREEMENT_TIEBREAK_SEED
                for _pol, _a in rmm["agreement"].items():
                    out[f"heldout_agree_top1_{_pol}"] = _a["top1"]
                    out[f"heldout_agree_taub_{_pol}"] = _a["taub"]
                # TRAIN-frontier metrics, same family, for the overfitting gap: train
                # up while held-out flat = overfitting, on the metrics the fallback can
                # split (return can't be held out -- env.reset() is root-only).
                rm_tr = heldout_ranking_metrics(
                    heldout_rows and train_rows[:len(heldout_rows)], by_name,
                    logits_for=score_for)
                out["train_top1"] = rm_tr["top1"]
                out["train_ndcg"] = rm_tr["ndcg"]
                # SELECTION on NDCG, not top1. A/B on F=4's saved checkpoints: NDCG is
                # ~2.2x smoother across checkpoints (sd 0.011 vs 0.026) because it reads
                # the WHOLE ranking, not just slot 1 -- a lower-variance, tie-safe
                # selection signal. top1/ndcg picked different checkpoints (55k vs 100k).
                sel_score = rm["ndcg"]
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
                   if ranking_eval
                   else f"coverage={out['coverage_at_reference_budget']:.2f}")
            bar.set_postfix_str(
                f"{sig} td={out.get('td_loss', float('nan')):.4f}", refresh=False)
            # STABLE, PARSEABLE per-checkpoint line: train_models.py (the parent, which is
            # a tty) parses `F=/step=/frames=/total=/ndcg=/td=` to drive its progress bar --
            # a bar in this child cannot render through the parent's line-buffered pipe.
            # Do not reorder/rename these fields without updating the parser.
            _ndcg = out.get("heldout_ndcg")
            print(f"[run] ckpt F={cfg.fringe_size} step={step} "
                  f"frames={step * cfg.batch_size} total={D} "
                  f"ndcg={'nan' if _ndcg is None else round(_ndcg, 4)} "
                  f"td={out.get('td_loss', float('nan')):.4f}")
            t_eval += _time.perf_counter() - _te
    bar.close()
    print(f"[timing] training {t_train:.1f}s over {S} steps "
          f"({1000 * t_train / max(1, S):.2f} ms/step); "
          f"per-checkpoint eval {t_eval:.1f}s over {len(_ckpt_set)} checkpoints "
          f"({t_eval / max(1, len(_ckpt_set)):.1f} s/ckpt)")

    # Smoothed selection on the held-out signal -- NOT a single argmax draw (the floor
    # run showed argmax picks a lucky rollout).
    best = cands[-1] if cfg.select_final else select_smoothed(cands, window=SELECTION_WINDOW)
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
            off, live, plen = [], [], []
            for n in fid:
                out = evaluate_split([by_name[n]], policy_for, cfg.fringe_size, seeds=1,
                                     expansion_cap=cap, gamma=cfg.gamma)
                off.append(int(out["expansions_mean"] or 0))
                # plan_length beside expansions: fewer expansions is only a WIN if
                # the plan is still optimal, so the quality check needs both, and
                # both must survive a gate failure.
                exp_n, plen_n = run_planner_metrics(
                    cfg.deep_exe, _problem_for(repo_root, by_name[n].instance), onnx,
                    cfg.fringe_size, separated=(cfg.kind_of_data == "separated"),
                    repo_root=repo_root)
                live.append(exp_n)
                plen.append(plen_n)
            # PERSIST BEFORE GATING: a None is a measurement (the planner aborted
            # on that instance), not an absence. A failed gate must not discard it.
            (run_dir / "planner_expansions.json").write_text(json.dumps({
                "instances": fid, "offline": off, "planner": live,
                "plan_length": plen,
                "fringe_size": cfg.fringe_size, "checkpoint": best.step,
            }, indent=1))
            gates.append(gate_env_fidelity(off, live, names=fid))
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
        # Gate on the SELECTION metric, matched-n on the SAME held-out set: FALLBACK
        # compares held-out top1 (higher better) vs each baseline's top1; PRIMARY
        # compares held-out-transfer regret. Gating on the fallback's train-set regret
        # would PASS on the optimistic number selection was moved away from.
        if eval_mode == EVAL_FROZEN:
            # No competitor baseline in unified mode: report the gap to the oracle
            # ceiling on the frozen set for the record (armed=False: not a verdict).
            _sel = next((c for c in cands if c.step == best.step), best)
            _rec = tel.read()
            _row = next((r for r in _rec if r.get("step") == best.step
                         and r.get("split") == "val"), {})
            gates.append(GateResult(
                "oracle_gap", True,
                f"frozen set n={_row.get('heldout_n')}: model ndcg "
                f"{best.select_score:.3f} top1 {_row.get('heldout_top1')} "
                f"regret_at_decision {_row.get('heldout_regret_at_decision')} "
                f"picked_dead {_row.get('heldout_picked_dead')} vs hfs_oracle "
                f"ndcg {baseline_ndcg.get('hfs_oracle')} (ceiling; no competitor "
                f"baseline is evaluated in unified mode)",
                armed=False, blocking=False))
        elif ranking_eval:
            gates.append(gate_beats_baselines(
                best.select_score, baseline_ndcg,
                higher_is_better=True, metric="heldout_ndcg"))
        else:
            gates.append(gate_beats_baselines(best.regret, baselines, metric="regret"))
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
                   "eval_fringes": ("frozen_random" if eval_mode == EVAL_FROZEN
                                    else eval_mode),
                   "unified": bool(cfg.unified),
                   "unified_manifest": ({i.name: i.manifest for i in train_i
                                         if isinstance(i, UnifiedInstance)}
                                        if cfg.unified else None),
                   "behaviour_policies_rolled": sorted(dsum.get("policies", [])),
                   "fill_fringes": bool(cfg.fill_fringes),
                   "fill_k": int(cfg.fill_k) if cfg.fill_fringes else 0,
                   "fill": dsum.get("fill"),
                   "beam_full_frac": {
                       "all_decision_states": dsum.get("full_beam_state_frac"),
                       "parent": dsum.get("parent_full_beam_state_frac"),
                       "fill": dsum.get("fill_full_beam_state_frac"),
                       "n_fill_rows": dsum.get("n_fill_rows"),
                   },
                   "selection_window": SELECTION_WINDOW,
                   "selection_metric": ("heldout_ndcg"
                                        if ranking_eval
                                        else "coverage_at_reference_budget"),
                   "split_manifest": split_manifest,
                   "baseline_heldout_top1": baseline_top1 or None,
                   "baseline_heldout_ndcg": baseline_ndcg or None,
                   "coverage_note": (
                       "coverage/regret is a TRAIN-SET rollout estimate (rolled from "
                       "root on trained instances), NOT held out. Selection used "
                       "held-out-trajectory top1. Transfer is NOT claimed here."
                       if eval_mode == EVAL_HELDOUT
                       else ("UNIFIED: every trajectory trained. Ranking metrics "
                             "(heldout_*) are on the FROZEN random-fringe set "
                             "(frozen_eval.json); coverage/regret is "
                             + ("held-out TRANSFER on the test CSVs."
                                if test_i else
                                "a TRAIN-SET rollout estimate. Transfer is NOT "
                                "claimed here."))
                       if eval_mode == EVAL_FROZEN
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

    # Diagnostic figures from the telemetry we just wrote (neutral; read-only).
    try:
        import subprocess as _sp
        _sp.run([_sys.executable,
                 str(repo_root / "scripts/rl_exp/plot_diagnostics.py"),
                 str(run_dir / "telemetry.jsonl"), "--fringe", str(cfg.fringe_size)],
                check=False, capture_output=True)
    except Exception as e:                       # plotting must never fail a run
        print(f"[run] (figures skipped: {e})")

    return {"selected": best, "baselines": baselines, "gates": gates, "run_dir": run_dir}


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _report_trees(trees: Sequence) -> None:
    """One line per (instance, strategy) tree, BEFORE any gate: a degenerate tree
    (HFS on CC is ~99% goal states because the generator keeps expanding goals; a
    BFS cut by the visit cap is mostly unexpanded frontier) must be visible here,
    not discovered from a regret number three stages later."""
    print(f"[run] {len(trees)} tree(s) loaded  (instance @ strategy: states, goals, "
          f"expanded, delta_root, censored)")
    for t in trees:
        s = t.stats()
        dr = "inf" if t.delta_root == float("inf") else f"{t.delta_root:.0f}"
        strat = _strat_label(t)
        print(f"[run]   {t.instance:24} @ {strat:6} states={s['n_reachable']:>6} "
              f"goals={s['n_goal_states']:>6} ({100 * s['goal_density']:5.1f}%) "
              f"expanded={t.n_expanded:>6}{'' if t.has_trace else ' (NO TRACE)'} "
              f"delta_root={dr:>4} censored={100 * s['censored_frac']:5.1f}%")


def _strat_label(t) -> str:
    """`BFS` for a generator strategy, the raw tag otherwise (`unified`)."""
    if not t.strategy:
        return "?"
    return dir_name(t.strategy) if t.strategy in STRATEGIES else str(t.strategy)


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
