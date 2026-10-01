"""Distance-estimator trainer: beams in, per-state distances out, ONNX for the planner.

The model scores one state at a time, so the fringe size is not a property of the
model: it is trained ONCE on beams of ``--train-fringe-size`` states and exported
from the same weights once per ``--fringe-sizes`` entry (the pooled size F is
baked into the ONNX graph) as ``distance_estimator_<F>.onnx``, plus one per-state
export ``distance_estimator_state.onnx`` (+ ``_C.txt``) for HFS / A*. Exports go
through the shared FringeEvalRL contract (lib/deep_nn/contract.py): deploy with
``--search RL --heuristics RL_H --RL_model <file> --RL_fringe_size F`` (plus
``--dataset_separated`` for a separated model).

The train/test split is the data: ``--train-csv`` tables train, ``--test-csv``
tables are evaluated only.  Driven by scripts/trial.py train.
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
    p.add_argument("--dir-save-model", required=True, help="checkpoint, exports and info go here")
    p.add_argument("--kind-of-data", choices=["merged", "separated"], default="merged",
                   help="the planner's state representation; separated is derived from merged DOTs")
    p.add_argument("--dataset-type", "--dataset_type", default="HASHED",
                   help="only HASHED is deployable through FringeEvalRL (issue #1)")
    p.add_argument("--fringe-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
                   help="one ONNX export per F, all from the same trained model")
    p.add_argument("--train-fringe-size", type=int, default=4, help="beam width of the training batches")
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


def train_and_export(args) -> Path:
    seed_everything(args.seed)
    out_dir = Path(args.dir_save_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    F_train = int(args.train_fringe_size)
    train, test, params = build_datasets(
        args.train_csv, args.test_csv, args.kind_of_data, F_train, args.unreachable_state_value,
        cache_dir=out_dir.parent / "cache", repo_root=REPO_ROOT,
        policies=args.behaviour_policies, seeds_per_policy=args.seeds_per_policy)
    print(f"[gnn] train F={F_train} {args.kind_of_data}: {len(train)} train beams, "
          f"{len(test) if test else 0} test beams, scaling {params}")

    m = DistanceEstimatorModel(lr=args.lr, hidden_dim=args.hidden_dim,
                               use_goal=(args.kind_of_data == "separated"))
    train_loader = train.loader(args.batch_size, shuffle=True, seed=args.seed)
    val_loader = test.loader(args.batch_size, shuffle=False) if test else train.loader(args.batch_size, shuffle=False)
    m.train(train_loader, val_loader, n_epochs=args.epochs, checkpoint_dir=str(out_dir),
            model_name=args.model_name)
    m.load_model(out_dir / f"{args.model_name}.pt")
    metrics = m.evaluate(val_loader)
    print(f"[gnn] best checkpoint on {'test' if test else 'train'}: "
          + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    info = {**vars(args), **{f"scaling_{k}": v for k, v in params.items()},
            **{f"eval_{k}": v for k, v in metrics.items()}}
    if not args.no_export_onnx:
        for F in sorted(set(int(f) for f in args.fringe_sizes)):
            onnx_path = out_dir / f"{args.model_name}_{F}.onnx"
            m.to_onnx(onnx_path, F, params)
            # parity on real beams re-packed at THIS F (truncated when F < the training width)
            feeds = [train.planner_feed(i, fringe_size=F) for i in range(min(32, len(train)))]
            check = m.verify_onnx(onnx_path, feeds, params)
            print(f"[gnn] exported {onnx_path}: contract OK, torch vs onnxruntime on "
                  f"{check['n_checked']} beams max |diff| = {check['max_abs_diff']:.2e}")
            info.update({f"onnx_F{F}": str(onnx_path), **{f"onnx_F{F}_{k}": v for k, v in check.items()}})
        if not m.model.use_goal:   # HFS/A* consumer: one state at a time, scaling inverted by the planner
            info["state_onnx"] = str(m.to_onnx_state(out_dir / f"{args.model_name}_state.onnx", params))
    (out_dir / f"{args.model_name}_info.txt").write_text(
        "".join(f"{k} = {v}\n" for k, v in info.items()))
    return out_dir


def main(argv=None):
    args = parse_args(argv)
    if args.dataset_type.upper() != "HASHED":
        raise SystemExit(f"--dataset-type {args.dataset_type}: FringeEvalRL deploys HASHED models only (issue #1)")
    train_and_export(args)


if __name__ == "__main__":
    main()
