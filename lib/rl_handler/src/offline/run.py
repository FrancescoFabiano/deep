"""The harness: dataset -> train -> telemetry -> selection -> export -> 3 gates.

ONE code path for RL and the baseline. `--model {dqn,cql,two_head}` is the only
difference, so the comparison cannot drift into two harnesses.

REPRESENTATION-AGNOSTIC: nothing here branches on the dataset type.
WITHIN-CONFIG ONLY: `assert_within_config` raises otherwise.
FAITHFUL DATA ONLY: instances absent from `faithful_pool.json` never enter.
"""

from __future__ import annotations

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
from .env import default_expansion_cap, reference_budget
from .faithfulness import build_faithful_pool, faithful_names, fidelity_names
from .planner_config import exploitation_for, planner_flags
from .policies import BEHAVIOUR_POLICIES, make_policy
from .qlearning import QTrainer, TrainConfig, default_reward_scale
from .selection import (
    Candidate,
    assert_within_config,
    gate_beats_baselines,
    gate_env_fidelity,
    onnx_path_for,
    run_planner_expansions,
    select,
    split_instances,
    write_selection_sidecar,
)
from .telemetry import CheckpointRecord, TelemetryWriter, evaluate_split
from .tree import load_tree_instance, partition_solvable

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
    hidden_dim: int = 64
    gnn_layers: int = 2
    lr: float = 1e-4
    batch_size: int = 64
    cql_alpha: float = 0.0
    gamma: float = 1.0
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
    """Load ONLY faithful instances from the CSVs train_models.py handed us.

    The gatekeeper stands between generation and training: an unfaithful tree is a
    WRONG PROBLEM, not noisy data. The shipped discard=0.4 tables have their
    shallow goals deleted (delta_root 10 vs a true optimal of 4), so training on
    them produces clean, self-consistent, meaningless numbers.
    """
    csvs = [Path(p) for p in cfg.train_csvs]
    missing = [str(p) for p in csvs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing generation tables: {missing}")
    loaded = [load_tree_instance(p, name=p.parent.name, kind_of_data=cfg.kind_of_data)
              for p in csvs]
    solvable, unsolvable = partition_solvable(loaded)
    pool_path = _fringe_dir(cfg).parent / "faithful_pool.json"
    pool = (json.loads(pool_path.read_text()) if pool_path.exists()
            else build_faithful_pool(solvable, out_path=pool_path))
    keep = set(faithful_names(pool))
    insts = [i for i in solvable if i.name in keep]
    if not insts:
        raise ValueError(
            f"no FAITHFUL instance among {[p.parent.name for p in csvs]}. "
            f"Regenerate at --dataset_discard_factor 0 -- the shipped discard=0.4 "
            f"tables have their shallow goals deleted (delta_root 10 vs a true "
            f"optimal of 4 on CC_2_2_3__pl_4) and are a DIFFERENT search problem. "
            f"See faithful_pool.json for the per-instance reason."
        )
    caches = {i.name: InstanceCache.from_paths(
        i.state_paths_abs(repo_root),
        cache_file=_fringe_dir(cfg).parent / "cache" / f"{i.name}.pt",
        verbose=False) for i in insts}
    return insts, caches, {"pool": pool, "unsolvable": [u.name for u in unsolvable]}


def run(cfg: RunConfig, repo_root: Path) -> Dict[str, object]:
    if cfg.model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg.model!r}")
    set_determinism(cfg.seed, strict=True)
    device = cfg.device or default_device()
    run_dir = _fringe_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    insts, caches, meta = load_pool(cfg, repo_root)
    by_name = {i.name: i for i in insts}
    train_n, val_n = split_instances([i.name for i in insts], cfg.val_frac,
                                     cfg.seed, cfg.val_instances)
    assert_within_config(train_n, val_n)                     # the guardrail
    train_i = [by_name[n] for n in train_n]
    val_i = [by_name[n] for n in val_n]
    cap = cfg.eval_expansion_cap or default_expansion_cap(insts)
    scale = cfg.reward_scale if cfg.reward_scale is not None else default_reward_scale(train_i)
    max_delta = max((d for i in train_i for d in i.delta if d != float("inf")), default=64.0)

    print(f"[run] F={cfg.fringe_size} model={cfg.model} "
          f"{cfg.kind_of_data}/{cfg.context_mode} device={device} -> {run_dir}")
    print(f"[run] faithful: train={train_n} val={val_n}  cap={cap} scale={scale:.4f}")

    rows, dsum = generate_dataset(train_i, cfg.fringe_size,
                                  policies=cfg.behaviour_policies or BEHAVIOUR_POLICIES,
                                  seeds_per_policy=cfg.seeds_per_policy,
                                  expansion_cap=cap,
                                  counterfactual=cfg.counterfactual,
                                  n_refill_samples=cfg.n_refill_samples)
    tel = TelemetryWriter(run_dir / "telemetry.jsonl")

    # ---- baselines, reported on their OWN before any comparison ----
    baselines: Dict[str, Optional[float]] = {}
    for b in BEHAVIOUR_POLICIES:
        out = evaluate_split(val_i, lambda n, _b=b: make_policy(by_name[n], _b, seed=0),
                             cfg.fringe_size, seeds=cfg.eval_seeds, expansion_cap=cap)
        baselines[b] = out["regret_mean_lower_bound"]
        tel.append(CheckpointRecord(step=-1, frames=0, split=f"baseline:{b}", payload=out))
    print(f"[run] baselines (val regret lower bound): {baselines}")

    net = _net(cfg, max_delta).to(device)
    if cfg.model == "two_head":
        from .baseline import train_two_head_baseline
        beams = []
        seen = set()
        for r in rows:
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
        trainer = QTrainer(net, _net(cfg, max_delta), train_i, caches, rows,
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

    # ---- train with checkpoints ----
    cands: List[Candidate] = []
    every = max(1, cfg.frames // cfg.n_checkpoints)
    for step in range(1, cfg.frames + 1):
        if trainer is not None:
            log = trainer.step()
        if step % every == 0 or step == cfg.frames:
            out = evaluate_split(val_i, policy_for, cfg.fringe_size, seeds=cfg.eval_seeds,
                                 expansion_cap=cap, score_for=score_for)
            if trainer is not None:
                out.update({k: log[k] for k in ("td_loss", "q_mean", "q_max", "grad_norm", "lr")})
                out.update(trainer.q_vs_qstar(rows[:200]))
            out["dataset"] = dsum
            tel.append(CheckpointRecord(step=step, frames=step * cfg.batch_size,
                                        split="val", payload=out))
            tr = evaluate_split(train_i[:2], policy_for, cfg.fringe_size, seeds=2,
                                expansion_cap=cap)
            tel.append(CheckpointRecord(step=step, frames=step * cfg.batch_size,
                                        split="train", payload=tr))
            cands.append(Candidate(step=step, frames=step * cfg.batch_size,
                                   coverage=out["coverage_at_reference_budget"],
                                   regret=out["regret_mean_lower_bound"],
                                   doom=out["doom_rate"]))
            torch.save(net.state_dict(), run_dir / "checkpoints" / f"ckpt_{step}.pt"
                       if (run_dir / "checkpoints").exists()
                       else _mk(run_dir / "checkpoints") / f"ckpt_{step}.pt")
            print(f"[run] step {step:5d} coverage={out['coverage_at_reference_budget']:.2f} "
                  f"regret={out['regret_mean_lower_bound']} "
                  f"auc={out.get('viability_auc')}")

    best = select(cands)
    print(f"[run] selected checkpoint {best.step} (coverage {best.coverage:.2f}, "
          f"regret {best.regret})")
    net.load_state_dict(torch.load(run_dir / "checkpoints" / f"ckpt_{best.step}.pt"))

    # ---- export ONLY the selected checkpoint ----
    gates = []
    # THE CONTRACT: train_models.py copies this to
    # <exp_dir>/_models/<domain>/frontier_policy_<F>.onnx, which
    # bulk_coverage_run.py reads and the C++ checks logits-length against.
    onnx = run_dir / f"frontier_policy_{cfg.fringe_size}_best_by_expansions.onnx"
    if cfg.export_onnx:
        RLFrontierTrainer(model=net, device="cpu", kind_of_data=cfg.kind_of_data).to_onnx(
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
            gates.append(type(gate_beats_baselines(1.0, {}))(
                "env_fidelity", False,
                "NOT ARMED: no --deep-exe given" if not cfg.deep_exe
                else "NOT ARMED: no faithful instance reaches 20 expansions"))
        gates.append(gate_beats_baselines(best.regret, baselines))
        for g in gates:
            print(f"[gate] {g.name}: {'PASS' if g.passed else 'FAIL'} -- {g.detail}")

        write_selection_sidecar(
            onnx, best, train_instances=train_n, val_instances=val_n,
            fringe_size=cfg.fringe_size, kind_of_data=cfg.kind_of_data,
            model=cfg.model, gamma=cfg.gamma, reward_scale=scale,
            baselines=baselines, gates=gates,
            excluded_unsolvable=meta["pool"]["excluded"],
            extra={"determinism": determinism_report(),
                   "context_mode": cfg.context_mode,
                   "dataset_type": cfg.dataset_type,
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
