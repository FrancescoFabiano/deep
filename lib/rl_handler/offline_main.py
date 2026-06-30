"""Offline RL (Double-DQN fringe ranking) entry point. See DESIGN.md.

Example (from lib/rl_handler):
  python offline_main.py --frames 100000 --seed 0 \
      --dir-save-model ../../exp/rl_exp/offline_rl/seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation BEFORE torch initializes CUDA. The
# separated-mode dual-GNN forward leaves >1 GiB reserved-but-unallocated on a
# small (8 GiB) GPU, which tips F=64 (2x-wider beam) over at its first update.
# expandable_segments lets the allocator grow segments instead of fragmenting.
# setdefault so an explicit caller env still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.frontier_policy import FrontierPolicyNetwork  # noqa: E402
from src.offline.dqn import EpsilonSchedule, OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache, load_goal_graph  # noqa: E402
from src.offline.sweep_cells import resolve_cells  # noqa: E402
from src.offline.tree_env import load_tree_instance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN = [
    "out/NN/Training/CC_2_2_4__pl_7/CC_2_2_4__pl_7_depth_25.csv",
    "out/NN/Training/CC_3_2_3__pl_6/CC_3_2_3__pl_6_depth_25.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline DQN fringe-ranking trainer")
    p.add_argument("--train-csv", nargs="+", default=DEFAULT_TRAIN)
    # Default EMPTY: with no val, selection falls back to TRAIN (and is forced to
    # train in the regime/study path regardless). nargs="*" so `--val-csv` with
    # no values is also empty. A val set is no longer the study's selection
    # signal — held-out instances go to --test-csv (diagnostic only).
    p.add_argument("--val-csv", nargs="*", default=[])
    p.add_argument(
        "--test-csv",
        nargs="+",
        default=[],
        help="Held-out DIAGNOSTIC test instances (regime-shaped, off-distribution "
        "vs the deploy-faithful TRAIN selection eval). Consumed by the per-regime "
        "diagnostic surface (default-on); never feeds model selection.",
    )
    p.add_argument("--frames", type=int, default=100_000)
    p.add_argument("--n-checkpoints", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=[0.99],
        help="Sweep axis (list): discount factor(s). Length 1 = fixed; length >1 "
        "= swept by the runner. A real training invocation must resolve to ONE "
        "gamma (single-cell contract).",
    )
    p.add_argument(
        "--epsilon-schedule",
        type=str,
        default="1.0,0.05,0.5",
        help="start,end,frac-of-frames for the linear decay",
    )
    p.add_argument(
        "--fringe-sizes",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Train one model per fringe size, sequentially, reusing the same "
        "parsed instances/caches. Default: 32 64.",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--replay-capacity", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--target-sync", type=int, default=1_000)
    # 1 gradient update per 4 env frames (classic DQN ratio); the GNN forward
    # is kernel-launch-bound on these tiny graphs, so this sets the fps.
    p.add_argument("--update-every", type=int, default=4)
    p.add_argument("--eval-expansion-cap", type=int, default=2_000)
    p.add_argument(
        "--signal-mode",
        type=str,
        default="basic",
        choices=["basic", "pbrs", "exact-return", "aux", "rank-sup", "rank-rl"],
        help="Training signal: 'basic' = current Double-DQN (default, unchanged); "
        "'pbrs' = potential-based d* shaping (keeps bootstrap); 'exact-return' = "
        "supervise chosen slot on -d* (no bootstrap); 'aux' = TD + lambda*MSE(-d*) "
        "auxiliary head on the shared trunk (head excluded from ONNX export); "
        "'rank-sup' = supervised ORDER loss (no bootstrap, range-free); 'rank-rl' "
        "= scale-invariant DQN (rank-fraction reward / fringe-centred target). "
        "All modes export the identical frontier_policy_<F>.onnx contract.",
    )
    p.add_argument(
        "--rank-variant",
        type=str,
        default=None,
        choices=["pairwise", "listwise", "reward", "advantage"],
        help="Variant for the rank objectives. rank-sup: 'pairwise' (default, "
        "logistic over d*-ordered slot pairs) or 'listwise' (rank-normalised soft "
        "target). rank-rl: 'reward' (default, rank-fraction reward in [0,1]) or "
        "'advantage' (fringe-centred target). Ignored for the other signal modes.",
    )
    p.add_argument(
        "--aux-lambda",
        type=float,
        default=1.0,
        help="Weight of the auxiliary -d* loss when --signal-mode=aux.",
    )
    p.add_argument(
        "--eval-refill-seeds",
        type=int,
        default=1,
        help="Average each val instance's greedy rollout over this many "
        "reservoir-refill seeds for robust (low-noise) checkpoint selection "
        "and convergence curves. Default: 1 (single-seed, prior behavior).",
    )
    # ---- P1: parallel fringe-composition regimes (value-only) ----
    # Regimes are now the DEFAULT path: the four composition regimes
    # {dfs,bfs,hfs,random} train into one shared replay+model. --use-regimes is
    # accepted (no-op, kept for back-compat); --no-regimes is the escape hatch
    # back to the single-source pipeline (debug only).
    p.add_argument(
        "--use-regimes",
        dest="use_regimes",
        action="store_true",
        default=True,
        help="Default ON. The four composition regimes train into one shared "
        "replay+model. Kept for back-compat; the regime path is now the default.",
    )
    p.add_argument(
        "--no-regimes",
        dest="use_regimes",
        action="store_false",
        help="Escape hatch: use the single-source (non-regime) pipeline. Debug "
        "only — the study path is regimes-default.",
    )
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["dfs", "bfs", "hfs", "random"],
        help="Which composition regimes to run (subset of the four). Default: "
        "all four. --mixture-weights, if given, must align with this order.",
    )
    p.add_argument(
        "--mixture-weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-regime mixture weights aligned with --regimes (default uniform). "
        "Renormalised per instance over that instance's available regimes.",
    )
    p.add_argument(
        "--target-centering",
        nargs="+",
        type=str,
        default=["absolute"],
        choices=["absolute", "fringe_mean"],
        help="Sweep axis (list). Value-only objective arm: 'absolute' -> "
        "signal_mode=basic; 'fringe_mean' -> signal_mode=rank-rl+advantage "
        "(per-fringe centred target). Length 1 = fixed; a real training run must "
        "resolve to ONE centering (single-cell contract).",
    )
    p.add_argument(
        "--same-distance",
        dest="same_distance_keep",
        action="store_true",
        default=False,
        help="KEEP fringes whose nodes all share one distance-from-goal value. "
        "Default (flag absent) FILTERS them out of the replay (per-regime drop "
        "rate is logged either way).",
    )
    p.add_argument(
        "--bfs-exclude-usable-frac",
        type=float,
        default=0.02,
        help="Exclude an instance from the BFS regime ONLY when its realized "
        "BFS-usable fraction (same-depth F-windows with >=2 distinct d*) is below "
        "this (depth->d* collinear; ~0 usable same-depth fringes). Default 0.02.",
    )
    p.add_argument(
        "--eval-exploration-nodes",
        type=int,
        default=None,
        help="Random-exploration beam slots in the deploy-faithful heuristic eval "
        "(default floor(F*0.1), matching the planner default).",
    )
    p.add_argument(
        "--train-expansion-cap",
        type=int,
        default=None,
        help="Per-episode expansion horizon for the redraw training envs "
        "(default 2*n_states, the safety cap). Lower it to cycle regimes faster.",
    )
    p.add_argument(
        "--lambda-ord",
        nargs="+",
        type=float,
        default=[0.0],
        help="Sweep axis (list). P2 order-auxiliary weight: L = L_val + "
        "lambda_ord * L_ord (pairwise order over non-padded slots) on the single "
        "head. 0.0 = value-only (byte-identical to no order term); >0 = "
        "value+order arm. A real training run must resolve to ONE lambda.",
    )
    p.add_argument(
        "--sweep-mode",
        type=str,
        default="one_at_a_time",
        choices=["product", "one_at_a_time"],
        help="How the runner expands the (gamma, lambda-ord, target-centering, "
        "fringe-sizes) axis lists into cells. one_at_a_time (default): baseline = "
        "first element of each list, vary one axis at a time off it (de-duped "
        "union). product: full Cartesian. offline_main only uses this to emit the "
        "resolved-cell manifest; it always trains a single cell.",
    )
    p.add_argument(
        "--list-cells",
        action="store_true",
        default=False,
        help="Resolve the axis lists under --sweep-mode, write the cell manifest "
        "(<dir-save-model>_cells_manifest.json), print it, and exit WITHOUT "
        "training. The runner / dry runs use this to introspect the cell set.",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--dataset-type",
        type=str,
        default="HASHED",
        choices=["MAPPED", "HASHED", "BITMASK"],
        help="Node-label representation; must match the generator's "
        "--dataset_type. Default: HASHED.",
    )
    p.add_argument(
        "--kind-of-data",
        type=str,
        default="merged",
        choices=["merged", "separated"],
        help="merged = goal inlined into each state DOT (default); separated = "
        "goal in a per-instance goal_tree.dot (CSV `Goal` column), fed through "
        "the separate goal input. Separated exports the 9-input ONNX.",
    )
    p.add_argument(
        "--use-goal-separate-input",
        dest="use_goal_separate_input",
        action="store_true",
        default=None,
        help="Force the separate goal GNN input. Defaults to True when "
        "--kind-of-data=separated, False for merged.",
    )
    p.add_argument(
        "--no-use-goal-separate-input",
        dest="use_goal_separate_input",
        action="store_false",
    )
    p.add_argument("--dir-save-model", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--export-onnx", action="store_true", default=True)
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def _build_model(
    dataset_type: str, use_goal_separate_input: bool
) -> FrontierPolicyNetwork:
    """A fresh model with the production architecture (matches deployed
    frontier_policy exports). The architecture is fringe-size-agnostic: F only
    sets the env beam width and the exported ONNX's symbolic output length, so
    one freshly-built model per fringe size is correct and reuses the parsed
    instances/caches."""
    return FrontierPolicyNetwork(
        node_input_dim=1,
        hidden_dim=128,
        gnn_layers=3,
        conv_type="gine",
        pooling_type="mean",
        dataset_type=dataset_type,
        edge_emb_dim=32,
        num_edge_labels=128,
        num_node_labels=4096,
        use_global_context=True,
        mlp_depth=2,
        use_goal_separate_input=use_goal_separate_input,
    )


def main() -> None:
    args = parse_args()

    # Resolve the separate-goal switch: explicit flag wins, else it tracks
    # kind_of_data (separated => on). Guard the contradictory combo early.
    use_goal_separate_input = (
        (args.kind_of_data == "separated")
        if args.use_goal_separate_input is None
        else bool(args.use_goal_separate_input)
    )
    if args.kind_of_data == "merged" and use_goal_separate_input:
        raise SystemExit(
            "merged data has the goal inlined into each state DOT; "
            "--use-goal-separate-input requires --kind-of-data=separated."
        )

    # Resolve the four sweep axes (gamma, lambda-ord, target-centering,
    # fringe-sizes) into the cell set under --sweep-mode, and persist it as the
    # manifest. The RUNNER expands these into per-cell invocations; offline_main
    # itself always trains a SINGLE cell (single-cell contract enforced below).
    cells = resolve_cells(
        args.gamma, args.lambda_ord, args.target_centering, args.fringe_sizes,
        mode=args.sweep_mode,
    )
    manifest = {
        "sweep_mode": args.sweep_mode,
        "axes": {
            "gamma": args.gamma, "lambda_ord": args.lambda_ord,
            "target_centering": args.target_centering,
            "fringe_sizes": [int(f) for f in args.fringe_sizes],
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    manifest_path = Path(str(args.dir_save_model) + "_cells_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    if args.list_cells:
        print(json.dumps(manifest, indent=2))
        print(f"[manifest] {manifest_path} ({len(cells)} cells, "
              f"mode={args.sweep_mode})")
        return  # introspection only — no training

    # Single-cell training contract: gamma / lambda-ord / target-centering must
    # each resolve to ONE value here (the runner passes singletons per cell).
    # --fringe-sizes may stay a list — offline_main still trains one model per
    # fringe size sequentially on the shared parsed data (existing behavior);
    # the runner gives one F per cell so each F lands in its own output dir.
    for name, vals in (("--gamma", args.gamma), ("--lambda-ord", args.lambda_ord),
                       ("--target-centering", args.target_centering)):
        if len(vals) != 1:
            raise SystemExit(
                f"{name} has {len(vals)} values; a direct training invocation "
                "must resolve to a single cell. Use --list-cells to emit the "
                "manifest, or scripts/sweeps/run_sensitivity.py to expand the "
                "sweep into per-cell runs."
            )
    gamma = float(args.gamma[0])
    lambda_ord = float(args.lambda_ord[0])
    target_centering = str(args.target_centering[0])

    # Parse instances + caches ONCE — the data is fringe-independent, so the
    # whole fringe sweep reuses it (no regeneration per fringe size). The order
    # is train, then val, then DIAGNOSTIC test (off-distribution, never selects).
    instances, caches, goal_graphs = [], [], []
    csvs = [*args.train_csv, *args.val_csv, *args.test_csv]
    pbar = tqdm(
        csvs,
        desc="preparing instances",
        #unit="inst",
        #dynamic_ncols=True,
        # disable=not sys.stderr.isatty(),  # clean logs under nohup/pipes
    )
    for csv in pbar:
        csv_path = _resolve(csv)
        inst = load_tree_instance(csv_path, kind_of_data=args.kind_of_data)
        pbar.set_postfix_str(f"{inst.name}: {inst.n_states} states")
        cache_file = csv_path.parent / "graph_cache_offline_v1.pt"
        cache = InstanceCache.from_paths(inst.state_paths_abs(REPO), cache_file)
        instances.append(inst)
        caches.append(cache)
        # Separated mode: parse this instance's goal_tree.dot once (the model
        # feeds it through the separate goal input). merged => None.
        if use_goal_separate_input:
            gp = inst.goal_path_abs(REPO)
            if gp is None:
                raise SystemExit(
                    f"{csv_path}: separated mode but no `Goal` path on the "
                    "instance; cannot load goal_tree.dot."
                )
            goal_graphs.append(load_goal_graph(gp))
        else:
            goal_graphs.append(None)

    n_train, n_val = len(args.train_csv), len(args.val_csv)
    train_ids = list(range(n_train))
    val_ids = list(range(n_train, n_train + n_val))
    test_ids = list(range(n_train + n_val, len(instances)))  # diagnostic only
    # Selection is ALWAYS train-based now (no val-based path remains).
    sel = "TRAIN (deploy-faithful; held-out is diagnostic-only)"
    print(f"[split] train={[instances[i].name for i in train_ids]} "
          f"val={[instances[i].name for i in val_ids]} "
          f"diag_test={[instances[i].name for i in test_ids]} | selection={sel}")
    if val_ids:
        print("[warn] --val-csv given but val is NOT a selection set (selection is "
              "train-based) and is not in the diagnostic surface — it is parsed "
              "but unused. Prefer --test-csv for held-out diagnostics.")
    if test_ids and not args.use_regimes:
        print("[warn] --test-csv given with --no-regimes; the diagnostic "
              "per-regime surface is regime-only, so test instances are ignored.")

    # Record the resolved TRAIN/TEST split (by instance name) into the per-seed
    # manifest, written once at run start (before training) so it travels with
    # the run and survives a kill. Merged into the existing _cells_manifest.json
    # (which already follows the seed{seed} convention since --dir-save-model
    # ends in seed{seed}).
    manifest["seed"] = int(args.seed)
    manifest["fringe_sizes"] = [int(f) for f in args.fringe_sizes]
    manifest["regimes"] = list(args.regimes) if args.use_regimes else []
    manifest["train"] = [instances[i].name for i in train_ids]
    manifest["test"] = [instances[i].name for i in test_ids]
    manifest["selection"] = "train"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[manifest] {manifest_path} (train={len(train_ids)} test={len(test_ids)} "
          f"selection=train)")

    # One trained model per fringe size, sequentially, on the same parsed data.
    for fringe in args.fringe_sizes:
        F = int(fringe)
        out_f = Path(str(args.dir_save_model) + f"_fringe{F}")
        out_f.mkdir(parents=True, exist_ok=True)
        print(f"[fringe {F}] training -> {out_f}")

        # Control per-fringe model init: FrontierPolicyNetwork.__init__ draws its
        # layer weights from the global torch RNG and does NOT seed it, and the
        # global RNG advances across the fringe loop (model build + a full
        # training run). Without re-seeding here, F=64 would init from a
        # different RNG state than F=32, confounding any fringe-size effect with
        # random-init variance. Re-seed so the ONLY difference across fringe runs
        # is the beam width. (The trainer re-seeds its own training RNG in
        # __init__, so post-build divergence is from beam binding + training,
        # which is intended.)
        torch.manual_seed(args.seed)
        if (args.device or "").startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

        # Fresh model + trainer per fringe; instances/caches are reused.
        model = _build_model(args.dataset_type, use_goal_separate_input)
        common = dict(
            model=model,
            instances=instances,
            caches=caches,
            train_ids=train_ids,
            val_ids=val_ids,
            fringe_size=F,
            gamma=gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            warmup=args.warmup,
            target_sync=args.target_sync,
            update_every=args.update_every,
            eval_expansion_cap=args.eval_expansion_cap,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            eval_refill_seeds=args.eval_refill_seeds,
            lambda_ord=lambda_ord,
            goal_graphs=goal_graphs if use_goal_separate_input else None,
            # Train-based selection is the ALWAYS path now: checkpoints are picked
            # on the TRAIN deploy-faithful metric, held-out instances are
            # diagnostic only. There is no val-based selection path anymore.
            select_on_train=True,
        )
        if args.use_regimes:
            from src.offline.regime_trainer import RegimeDQNTrainer  # noqa: E402

            mix = None
            if args.mixture_weights is not None:
                if len(args.mixture_weights) != len(args.regimes):
                    raise SystemExit(
                        "--mixture-weights must align 1:1 with --regimes "
                        f"({len(args.mixture_weights)} vs {len(args.regimes)})."
                    )
                mix = dict(zip(args.regimes, args.mixture_weights))
            trainer = RegimeDQNTrainer(
                **common,
                regimes=args.regimes,
                mixture_weights=mix,
                target_centering=target_centering,
                same_distance_keep=args.same_distance_keep,
                bfs_exclude_usable_frac=args.bfs_exclude_usable_frac,
                eval_exploration_nodes=args.eval_exploration_nodes,
                train_expansion_cap=args.train_expansion_cap,
                diag_test_ids=test_ids,
            )
        else:
            trainer = OfflineDQNTrainer(
                **common,
                signal_mode=args.signal_mode,
                aux_lambda=args.aux_lambda,
                rank_variant=args.rank_variant,
            )

        with (out_f / "args.json").open("w") as fh:
            json.dump(
                {**vars(args), "fringe_size": F,
                 "resolved_cell": {
                     "gamma": gamma, "lambda_ord": lambda_ord,
                     "target_centering": target_centering, "fringe_size": F}},
                fh, indent=2,
            )

        eps = EpsilonSchedule.parse(args.epsilon_schedule, args.frames)
        trainer.train(
            frames=args.frames,
            n_checkpoints=args.n_checkpoints,
            epsilon=eps,
            out_dir=out_f,
        )

        if args.export_onnx:
            from src.trainer import RLFrontierTrainer

            for tag in ("best_by_expansions", "best_by_spearman"):
                ckpt = out_f / f"{tag}.pt"
                if not ckpt.exists():
                    continue
                loaded = RLFrontierTrainer.load_model(ckpt, device="cpu")
                export_trainer = RLFrontierTrainer(
                    model=loaded, kind_of_data=args.kind_of_data, device="cpu"
                )
                onnx_path = out_f / f"frontier_policy_{F}_{tag}.onnx"
                export_trainer.to_onnx(
                    onnx_path,
                    node_input_dim=1,
                    onnx_frontier_size=F,
                )
                print(f"[export] {onnx_path}")

        # Per-run (single-seed) plots, written into the same dir as the export.
        # Done AFTER the model save + ONNX export and wrapped so a plotting
        # failure can never cost the trained model. Cross-seed IQM bands stay in
        # offline_analysis.py; these are single-seed only.
        try:
            if os.environ.get("RL_FORCE_PLOT_ERROR"):  # test hook for the guard
                raise RuntimeError("forced plot error (RL_FORCE_PLOT_ERROR)")
            from src.offline.plots import plot_seed_curves, plot_seed_val_curves

            history_file = out_f / "history.json"
            plot_seed_curves(history_file, out_f, seed=args.seed, fringe=F)
            plot_seed_val_curves(history_file, out_f, seed=args.seed, fringe=F)
            pngs = [
                f"training_curves_seed{args.seed}_fringe{F}.png",
                f"val_metrics_seed{args.seed}_fringe{F}.png",
            ]
            print(f"[plots] {out_f}: {', '.join(pngs)}")
        except Exception as exc:  # non-fatal: model + ONNX are already saved
            print(f"[plots] WARN failed to render per-run plots: {exc}")

        # Release this fringe's GPU memory before the next fringe builds its own
        # model/trainer. Without this F=32's model/target/optimizer/activation
        # pools persist into F=64 (a 2x-wider beam needs more) and OOM the GPU
        # even though each fringe alone fits. ALL references must drop: the local
        # `model`, `trainer` (holds model+target), AND `common` (the kwargs dict
        # still references `model`). gc.collect() breaks any model<->trainer
        # cycle so the refcount actually hits zero before empty_cache().
        del trainer, model, common
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[done]")


if __name__ == "__main__":
    main()
