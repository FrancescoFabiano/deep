#!/usr/bin/env python3
"""Train one cell: dataset -> train -> telemetry -> selection -> export -> gates.

THIS IS NOT THE TOP-LEVEL ENTRY POINT. `final_launcher.sh` owns the cell iteration
(one cell per invocation, via env vars) and calls `scripts/rl_exp/train_models.py`,
which enumerates domains/seeds and invokes THIS per (domain, seed), forwarding
unknown args verbatim. This script owns exactly one (domain, seed) cell.

    train_models.py -> offline_main.py --seed S --dir-save-model <dir>/seedS \
                                       --train-csv ... [--test-csv ...] \
                                       [--kind-of-data separated] --model M \
                                       <forwarded> --fringe-sizes F...

Export contract (train_models.py then installs it to the deploy path):
    <dir-save-model>_fringe<F>/frontier_policy_<F>_best_by_expansions.onnx
    -> copied to <exp_dir>/_models/<domain>/frontier_policy_<F>.onnx

EVERY FLAG HERE IS CONSUMED. A flag the trainer ignores is the same latent bug as
one it rejects -- it just fails silently. Dropped, with the reason:

  --pad-closed/--no-pad-closed  padding the beam with CLOSED states gives the model
                                actions the planner can never take -- a train/deploy
                                mismatch the fidelity gate would flag. A defect, not
                                a feature.
  --target-tau                  Polyak. At gamma=1 the trainer uses a hard target
                                sync + the |Q| > 3x cap divergence alarm; a Polyak
                                option the trainer never reads is dead surface.
  --lr-schedule / --lr-min      gamma=0.99 stability machinery. The trainer honours
                                no schedule at gamma=1.
  --epsilon-schedule            offline: there is no exploration. Behaviour
                                randomness lives entirely in data generation.
  --random-pct                  superseded by --behaviour-policies
  --val-csv                     superseded by the instance-level split
  --refill-mode                 the env models refill as its own stochastic
                                transition (--n-refill-samples)
  --reward-mode                 DOOM is unreachable on solvable instances
                                (completeness proposition) and unsolvable ones are
                                filtered at load, so `legacy` has nothing to ablate
  --use-regimes/--regimes/--sweep-mode/--list-cells/--signal-mode/--rank-variant/
  --lambda-ord/--target-centering/--same-distance/--bfs-exclude-usable-frac/
  --eval-exploration-nodes/--train-expansion-cap/--eval-refill-seeds/
  --stratified-replay/--aux-lambda      regime + sweep + old-objective machinery
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
    # ---- what train_models.py sends ----
    p.add_argument("--train-csv", nargs="+", required=True,
                   help="generation-table CSVs (train). Instance-level split is "
                        "carved from these; WITHIN-CONFIG is enforced in code.")
    p.add_argument("--test-csv", nargs="*", default=None,
                   help="touched exactly once, at the end; never selects")
    p.add_argument("--dir-save-model", required=True,
                   help="`_fringe<F>` is appended per fringe size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kind-of-data", choices=["merged", "separated"], default="merged",
                   help="a different STATE REPRESENTATION, not a flag to prune: the "
                        "goal inlined per state vs a separate goal_tree.dot fed as 4 "
                        "extra tensors. Drives goal loading; NEVER the presence of "
                        "the CSV `Goal` column (merged populates it too).")
    p.add_argument("--no_goal", dest="no_goal", action="store_true",
                   help="the launcher's spelling of --kind-of-data separated; kept "
                        "as an alias so its invocation does not have to change")
    p.add_argument("--model", choices=["dqn", "cql", "two_head"], default="dqn")
    p.add_argument("--fringe-sizes", type=int, nargs="+", default=[4])
    # ---- trainer ----
    p.add_argument("--frames", type=int, default=2000)
    p.add_argument("--n-checkpoints", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-sync", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--cql-alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.9999,
                   help="Default 0.9999: the paper's discounted reward, numerically "
                        "~= the gamma=1 SSP limit (V*=-delta to <0.2%% for delta << "
                        "horizon 10000). gamma=1 stays valid as the SSP ablation. "
                        "assert_gamma enforces cap < 1/(1-gamma) so the objective "
                        "cannot silently saturate.")
    p.add_argument("--reward-scale", type=float, default=None,
                   help="default 1/median(delta_root over train). Pure rescaling; "
                        "human-facing numbers stay unscaled.")
    # ---- model ----
    p.add_argument("--context-mode", choices=["none", "mean_pool", "self_attention"],
                   default="mean_pool")
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--dataset-type", default="HASHED",
                   help="opaque to the pipeline; BITMASK needs no change here")
    p.add_argument("--use-global-context", dest="use_global_context",
                   action="store_true", default=None)
    p.add_argument("--no-global-context", dest="use_global_context",
                   action="store_false")
    # ---- data generation (offline dataset) ----
    p.add_argument("--behaviour-policies", nargs="+",
                   default=["bfs", "dfs", "hfs_oracle", "random"])
    p.add_argument("--seeds-per-policy", type=int, default=3)
    p.add_argument("--counterfactual-actions", choices=["all", "none"], default="all")
    p.add_argument("--n-refill-samples", type=int, default=1)
    # ---- split / eval ----
    p.add_argument("--val-frac", type=float, default=0.34)
    p.add_argument("--val-instances", nargs="*", default=None)
    p.add_argument(
        "--allow-cross-config", action="store_true",
        help="Opt into a cross-configuration (global) split. OFF by default: the "
             "guard hard-raises, because node ids are fluent-set hashes and "
             "configurations share 0.0%% of them. On HASHED this buys a "
             "STRUCTURE-ONLY FLOOR -- the node channel contributes nothing, so any "
             "gap comes from topology + edge labels alone. The run is stamped "
             "exploratory=true, cross_config=true. Not a transfer result.")
    p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument("--eval-expansion-cap", type=int, default=None)
    p.add_argument("--device", default=None)
    # ---- export + gates ----
    p.add_argument("--export-onnx", dest="export_onnx", action="store_true", default=True)
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    p.add_argument("--fidelity-instances", type=int, default=3)
    p.add_argument("--deep-exe", default=None,
                   help="Path to the deep binary. ARMS the env-fidelity gate, which "
                        "replays the policy in the REAL planner and compares expansion "
                        "counts. Default None = UNARMED: planner deployment is out of "
                        "scope, and the RL-vs-baseline claim is made IN THE OFFLINE "
                        "ENV, which is what we measure. Pass a path only when planner "
                        "deployment is back in scope -- and see the note in run.py: "
                        "the gate does not pass --RL_heuristics RNG, so its first "
                        "verdict was measured against a MIN-refill planner and is not "
                        "to be trusted until that is fixed.")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    kind = "separated" if a.no_goal else a.kind_of_data
    ctx = a.context_mode
    if a.use_global_context is False and ctx == "mean_pool":
        ctx = "none"          # the deprecated bool, honoured
    rc = 0
    for F in a.fringe_sizes:
        cfg = RunConfig(
            train_csvs=[Path(p) for p in a.train_csv],
            test_csvs=[Path(p) for p in (a.test_csv or [])],
            dir_save_model=Path(a.dir_save_model),
            fringe_size=F, model=a.model, kind_of_data=kind, context_mode=ctx,
            attn_heads=a.attn_heads, attn_layers=a.attn_layers,
            frames=a.frames, n_checkpoints=a.n_checkpoints, seed=a.seed,
            seeds_per_policy=a.seeds_per_policy,
            counterfactual=a.counterfactual_actions,
            n_refill_samples=a.n_refill_samples,
            eval_seeds=a.eval_seeds, val_frac=a.val_frac,
            val_instances=a.val_instances,
            allow_cross_config=a.allow_cross_config, hidden_dim=a.hidden_dim,
            gnn_layers=a.gnn_layers, lr=a.lr, batch_size=a.batch_size,
            target_sync=a.target_sync, max_grad_norm=a.max_grad_norm,
            cql_alpha=a.cql_alpha, gamma=a.gamma, reward_scale=a.reward_scale,
            eval_expansion_cap=a.eval_expansion_cap,
            fidelity_instances=a.fidelity_instances, deep_exe=a.deep_exe,
            device=a.device, export_onnx=a.export_onnx, dataset_type=a.dataset_type,
            behaviour_policies=a.behaviour_policies,
        )
        out = run(cfg, REPO)
        # Only an ARMED gate can fail the run. An unarmed gate did not measure
        # anything, so it has no verdict -- counting it as failure meant deliberately
        # disarming the env-fidelity gate (planner deployment out of scope) would
        # abort the launcher at step 2.
        if any(g.armed and not g.passed for g in out["gates"]):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
