"""Distance-estimator trainer: beams in, per-state distances out, ONNX for the planner.

Trains one model per fringe size on the planner-shaped beams of the given
generation tables (src/beams.py), then exports it through the shared
FringeEvalRL contract (lib/deep_nn/contract.py) as ``distance_estimator_<F>.onnx``:
deploy with ``--search RL --heuristics RL_H --RL_model <file> --RL_fringe_size F``
(plus ``--dataset_separated`` for a separated model).

The train/test split is the data: ``--train-csv`` tables train, ``--test-csv``
tables are evaluated only.  Driven by scripts/gnn_exp/train_models.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # this handler's `src`
sys.path.append(str(HERE.parent))     # lib/deep_nn, shared with rl_handler


from deep_nn.policies import BEHAVIOUR_POLICIES  # noqa: E402
from src.beams import build_datasets  # noqa: E402
from src.utils import DistanceEstimatorModel, seed_everything  # noqa: E402

REPO_ROOT = HERE.parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", nargs="+", required=True, help="generation tables to train on")
    p.add_argument("--test-csv", nargs="*", default=None, help="held-out tables: evaluated, never trained on")
    p.add_argument("--dir-save-model", required=True, help="`_fringe<F>` is appended per fringe size")
    p.add_argument("--kind-of-data", choices=["merged", "separated"], default="merged",
                   help="the planner's state representation; separated is derived from merged DOTs")
    p.add_argument("--dataset-type", "--dataset_type", default="HASHED",
                   help="only HASHED is deployable through FringeEvalRL (issue #1)")
    p.add_argument("--fringe-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64, help="beams per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--behaviour-policies", nargs="+", default=list(BEHAVIOUR_POLICIES))
    p.add_argument("--seeds-per-policy", type=int, default=3)
    p.add_argument("--unreachable-state-value", type=float, default=1e6,
                   help="the table's distance of a state the generator could not reach a goal from")
    p.add_argument("--model-name", default="distance_estimator")
    p.add_argument("--no-export-onnx", action="store_true")
    return p.parse_args(argv)


def train_one(args, fringe_size: int) -> Path:
    seed_everything(args.seed)
    out_dir = Path(f"{args.dir_save_model}_fringe{fringe_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test, params = build_datasets(
        args.train_csv, args.test_csv, args.kind_of_data, fringe_size, args.unreachable_state_value,
        cache_dir=Path(args.dir_save_model).parent / "cache", repo_root=REPO_ROOT,
        policies=args.behaviour_policies, seeds_per_policy=args.seeds_per_policy)
    print(f"[gnn] F={fringe_size} {args.kind_of_data}: {len(train)} train beams, "
          f"{len(test) if test else 0} test beams, scaling {params}")

    m = DistanceEstimatorModel(lr=args.lr, hidden_dim=args.hidden_dim,
                               use_goal=(args.kind_of_data == "separated"))
    train_loader = train.loader(args.batch_size, shuffle=True, seed=args.seed)
    val_loader = test.loader(args.batch_size, shuffle=False) if test else train.loader(args.batch_size, shuffle=False)
    m.train(train_loader, val_loader, n_epochs=args.epochs, checkpoint_dir=str(out_dir),
            model_name=args.model_name)
    m.load_model(out_dir / f"{args.model_name}.pt")
    metrics = m.evaluate(val_loader)
    print(f"[gnn] F={fringe_size} best checkpoint on {'test' if test else 'train'}: "
          + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    info = {**vars(args), "fringe_size": fringe_size, **{f"scaling_{k}": v for k, v in params.items()},
            **{f"eval_{k}": v for k, v in metrics.items()}}
    onnx_path = out_dir / f"{args.model_name}_{fringe_size}.onnx"
    if not args.no_export_onnx:
        m.to_onnx(onnx_path, fringe_size, params)
        check = m.verify_onnx(onnx_path, [train.planner_feed(i) for i in range(min(32, len(train)))], params)
        print(f"[gnn] exported {onnx_path}: contract OK, torch vs onnxruntime on "
              f"{check['n_checked']} beams max |diff| = {check['max_abs_diff']:.2e}")
        info.update(onnx=str(onnx_path), **{f"onnx_{k}": v for k, v in check.items()})
    (out_dir / f"{args.model_name}_info.txt").write_text(
        "".join(f"{k} = {v}\n" for k, v in info.items()))
    return onnx_path


def main(argv=None):
    args = parse_args(argv)
    if args.dataset_type.upper() != "HASHED":
        raise SystemExit(f"--dataset-type {args.dataset_type}: FringeEvalRL deploys HASHED models only (issue #1)")
    for F in args.fringe_sizes:
        train_one(args, F)


if __name__ == "__main__":
    main()
