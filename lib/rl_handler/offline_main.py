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

from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.frontier_policy import FrontierPolicyNetwork  # noqa: E402
from src.offline.dqn import EpsilonSchedule, OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache  # noqa: E402
from src.offline.tree_env import load_tree_instance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN = [
    "out/NN/Training/CC_2_2_4__pl_7/CC_2_2_4__pl_7_depth_25.csv",
    "out/NN/Training/CC_3_2_3__pl_6/CC_3_2_3__pl_6_depth_25.csv",
]
DEFAULT_VAL = [
    "out/NN/Training/CC_3_2_3__pl_7/CC_3_2_3__pl_7_depth_25.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline DQN fringe-ranking trainer")
    p.add_argument("--train-csv", nargs="+", default=DEFAULT_TRAIN)
    p.add_argument("--val-csv", nargs="+", default=DEFAULT_VAL)
    p.add_argument("--frames", type=int, default=100_000)
    p.add_argument("--n-checkpoints", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.99)
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
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dir-save-model", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--export-onnx", action="store_true", default=True)
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def _build_model() -> FrontierPolicyNetwork:
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
        dataset_type="HASHED",  # TODO
        edge_emb_dim=32,
        num_edge_labels=128,
        num_node_labels=4096,
        use_global_context=True,
        mlp_depth=2,
        use_goal_separate_input=False,  # TODO
    )


def main() -> None:
    args = parse_args()

    # Parse instances + caches ONCE — the data is fringe-independent, so the
    # whole fringe sweep reuses it (no regeneration per fringe size).
    instances, caches = [], []
    csvs = [*args.train_csv, *args.val_csv]
    pbar = tqdm(
        csvs,
        desc="preparing instances",
        #unit="inst",
        #dynamic_ncols=True,
        # disable=not sys.stderr.isatty(),  # clean logs under nohup/pipes
    )
    for csv in pbar:
        csv_path = _resolve(csv)
        inst = load_tree_instance(csv_path)
        pbar.set_postfix_str(f"{inst.name}: {inst.n_states} states")
        cache_file = csv_path.parent / "graph_cache_offline_v1.pt"
        cache = InstanceCache.from_paths(inst.state_paths_abs(REPO), cache_file)
        instances.append(inst)
        caches.append(cache)

    train_ids = list(range(len(args.train_csv)))
    val_ids = list(range(len(args.train_csv), len(instances)))
    print(f"[split] train={[instances[i].name for i in train_ids]} "
          f"val={[instances[i].name for i in val_ids]}")

    # One trained model per fringe size, sequentially, on the same parsed data.
    for fringe in args.fringe_sizes:
        F = int(fringe)
        out_f = Path(str(args.dir_save_model) + f"_fringe{F}")
        out_f.mkdir(parents=True, exist_ok=True)
        print(f"[fringe {F}] training -> {out_f}")

        # Fresh model + trainer per fringe; instances/caches are reused.
        model = _build_model()
        trainer = OfflineDQNTrainer(
            model=model,
            instances=instances,
            caches=caches,
            train_ids=train_ids,
            val_ids=val_ids,
            fringe_size=F,
            gamma=args.gamma,
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
        )

        with (out_f / "args.json").open("w") as fh:
            json.dump({**vars(args), "fringe_size": F}, fh, indent=2)

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
                    model=loaded, kind_of_data="merged", device="cpu"
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

    print("[done]")


if __name__ == "__main__":
    main()
