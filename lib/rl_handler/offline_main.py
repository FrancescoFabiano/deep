#!/usr/bin/env python3
"""Entry point: one command runs dataset -> train -> telemetry -> selection ->
export -> gates.

    python3 offline_main.py --exp-dir exp/rl_exp/faithful --domain CC \
        --fringe-sizes 4 --model dqn --kind-of-data merged

`--model {dqn,cql,two_head}` is the ONLY difference between the RL arm and the
baseline: one harness, so the comparison cannot drift into two code paths.

PRUNED, and why (the brief's delete list, plus what this branch retired):
  --use-regimes/--regimes/--sweep-mode/--list-cells  regime + sweep machinery
  --signal-mode/--rank-variant/--lambda-ord/--target-centering
                                                     the old softmax-ranking objective
  --same-distance/--bfs-exclude-usable-frac/--eval-exploration-nodes
  --train-expansion-cap/--eval-refill-seeds/--target-tau/--stratified-replay
  --aux-lambda
  --epsilon-schedule       offline: there is no exploration. Behaviour randomness
                           lives entirely in data generation.
  --random-pct             superseded by --behaviour-policies
  --val-csv                superseded by the instance-level split
  --refill-mode            the env models refill as its own stochastic transition
                           (--n-refill-samples); we do not pin a C++ RefillMode here
  --reward-mode            DOOM is unreachable on solvable instances (completeness
                           proposition) and unsolvable ones are filtered at load, so
                           the `legacy` ablation has no regime to run in
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from src.offline.run import RunConfig, run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- data ----
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--kind-of-data", choices=["merged", "separated"], default="merged",
                   help="a different STATE REPRESENTATION, not a flag to prune: the "
                        "goal inlined per state vs a separate goal_tree.dot fed as 4 "
                        "extra tensors. Driven by this flag, NEVER by the presence of "
                        "the CSV `Goal` column (merged populates it too).")
    p.add_argument("--dataset-type", default="HASHED",
                   help="opaque to the pipeline; BITMASK needs no harness change")
    p.add_argument("--fringe-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--behaviour-policies", nargs="+",
                   default=["bfs", "dfs", "hfs_oracle", "random"])
    p.add_argument("--seeds-per-policy", type=int, default=3)
    p.add_argument("--counterfactual-actions", choices=["all", "none"], default="all")
    p.add_argument("--n-refill-samples", type=int, default=1)
    # ---- split (WITHIN-CONFIG, enforced in code) ----
    p.add_argument("--val-frac", type=float, default=0.34)
    p.add_argument("--val-instances", nargs="*", default=None)
    p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument("--eval-expansion-cap", type=int, default=None)
    # ---- model / trainer ----
    p.add_argument("--model", choices=["dqn", "cql", "two_head"], default="dqn")
    p.add_argument("--cql-alpha", type=float, default=0.0)
    p.add_argument("--context-mode", choices=["none", "mean_pool", "self_attention"],
                   default="mean_pool")
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="1.0 IS the objective: every policy is proper (completeness "
                        "proposition), so this is a stochastic shortest path problem. "
                        "gamma<1 truncates the objective and warns loudly; it is an "
                        "ablation axis, not a tuning knob.")
    p.add_argument("--reward-scale", type=float, default=None,
                   help="default 1/median(delta_root over train). Pure rescaling; it "
                        "cannot change the optimal policy. Human-facing numbers stay "
                        "unscaled.")
    p.add_argument("--frames", type=int, default=2000)
    p.add_argument("--n-checkpoints", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    # ---- export + gates ----
    p.add_argument("--export-onnx", dest="export_onnx", action="store_true", default=True)
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    p.add_argument("--fidelity-instances", type=int, default=3)
    p.add_argument("--deep-exe", default=str(REPO / "cmake-build-release-nn/bin/deep"),
                   help="the env-fidelity gate is UNARMED without this")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    rc = 0
    for F in a.fringe_sizes:
        cfg = RunConfig(
            exp_dir=Path(a.exp_dir), domain=a.domain, fringe_size=F, model=a.model,
            kind_of_data=a.kind_of_data, context_mode=a.context_mode,
            frames=a.frames, n_checkpoints=a.n_checkpoints, seed=a.seed,
            seeds_per_policy=a.seeds_per_policy, eval_seeds=a.eval_seeds,
            val_frac=a.val_frac, val_instances=a.val_instances,
            hidden_dim=a.hidden_dim, gnn_layers=a.gnn_layers, lr=a.lr,
            batch_size=a.batch_size, cql_alpha=a.cql_alpha, gamma=a.gamma,
            reward_scale=a.reward_scale, eval_expansion_cap=a.eval_expansion_cap,
            fidelity_instances=a.fidelity_instances, deep_exe=a.deep_exe,
            device=a.device, export_onnx=a.export_onnx, dataset_type=a.dataset_type,
        )
        out = run(cfg, REPO)
        if any(not g.passed for g in out["gates"]):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
