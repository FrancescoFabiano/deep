"""gnn_handler_plus entry point.

Same CLI as lib/gnn_handler plus four flags (see README.md):
  --dynamic-max-depth   (default true)
  --include-unreachable (default true)   [+ --unreachable-cap]
  --ckpt-metric {spearman,val_loss} (default spearman)
  --heuristic-weight W  (default 1.0)

Run:  .venv/bin/python lib/gnn_handler_plus/__main__.py <baseline args> [...]

NOTE on data: the baseline build drops unreachable states at BUILD time
(GraphDataPipeline remove_unreachable_goal_states=True), so a samples.pt
produced by the baseline contains none.  --include-unreachable therefore
needs a samples.pt built by THIS entry point (it builds with the filter
off); pass --build-data true once per dataset dir.  A loud error is raised
if the loaded samples.pt has no unreachable rows while the flag is on.
"""
import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Only the plus dir goes on sys.path: `src` must bind to the PLUS package,
# whose __init__ extends its module search path into the baseline's src/.
# Putting lib/gnn_handler on sys.path too would let the baseline's `src`
# shadow ours depending on insertion order — never do it.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

from src.preprocessing import GraphDataPipeline  # noqa: E402  (baseline)
from src.utils import (  # noqa: E402  (baseline)
    KEYWORD_BITMASK,
    KEYWORD_HASHED,
    KEYWORD_MAPPED,
    get_dataloaders,
    print_values,
    seed_everything,
)

from src.export import record_params_in_history, write_c_file  # noqa: E402
from src.sample_prep import count_unreachable, prepare_samples_plus  # noqa: E402
from src.training import select_model_plus  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="gnn_handler_plus: baseline pipeline + flagged method changes."
    )
    # ---- baseline-compatible arguments (same names/defaults) -------------
    parser.add_argument("--subset-train", action="extend", nargs="+", type=str,
                        default=[])
    parser.add_argument("--model-name", default="distance_estimator", type=str)
    parser.add_argument("--normalization-constants-name", default="C", type=str)
    parser.add_argument("--folder-raw-data", type=str, default="out/NN/Training")
    parser.add_argument("--unreachable-state-value", type=int, default=1000000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-percentage-per-class", type=float, default=0.5)
    parser.add_argument("--dir-save-data", type=str, default="data")
    parser.add_argument("--dir-save-model", type=str, default="models")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-data", type=str2bool, default=True)
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--dataset_type",
                        choices=[KEYWORD_MAPPED, KEYWORD_HASHED, KEYWORD_BITMASK],
                        default=KEYWORD_HASHED)
    parser.add_argument("--kind-of-data", type=str,
                        choices=["merged", "separated"], default="merged")
    parser.add_argument("--use-goal", type=str2bool, default=False)
    parser.add_argument("--use-depth", type=str2bool, default=False)
    parser.add_argument("--verbose", type=str2bool, default=False)

    # ---- plus flags -------------------------------------------------------
    parser.add_argument(
        "--dynamic-max-depth", type=str2bool, default=True,
        help="MAX_DEPTH = ceil(1.1 x max raw reachable train distance) "
             "instead of the hardcoded 50.")
    parser.add_argument(
        "--include-unreachable", type=str2bool, default=True,
        help="Keep sentinel-distance states with target = MAX_DEPTH instead "
             "of filtering them out of training.")
    parser.add_argument(
        "--unreachable-cap", type=float, default=0.25,
        help="If unreachable states exceed this fraction of the train set, "
             "subsample them down to it.")
    parser.add_argument(
        "--ckpt-metric", choices=["spearman", "val_loss"], default="spearman",
        help="Metric that selects the exported checkpoint; the val_loss "
             "selection is always saved alongside as *_by_val_loss.pt.")
    parser.add_argument(
        "--heuristic-weight", type=float, default=1.0,
        help="W: write slope x W into distance_estimator_C.txt at export "
             "(h_planner ~ h/W, bounded-suboptimality knob).")
    parser.add_argument(
        "--unreachable-loss-weight", type=float, default=1.0,
        help="U: MSE weight for unreachable samples (loss = sum(w e^2)/sum(w)); "
             "counters their ~0.3%% gradient share under plain MSE.")

    return parser.parse_args()


def main(args):
    seed_everything(args.seed)

    path_data = args.dir_save_data
    path_model = args.dir_save_model
    if args.experiment_name:
        path_data += "/" + args.experiment_name
        path_model += "/" + args.experiment_name
    os.makedirs(path_data, exist_ok=True)
    data_path = path_data + "/samples.pt"

    print("\n*********************** gnn_handler_plus ***********************")
    print(f"subset_train: {args.subset_train} | {args.dataset_type} | "
          f"{args.kind_of_data} | goal: {args.use_goal} | depth: {args.use_depth} "
          f"| train: {args.train} | build: {args.build_data}")
    print(f"[plus] dynamic_max_depth={args.dynamic_max_depth}  "
          f"include_unreachable={args.include_unreachable} "
          f"(cap {args.unreachable_cap})  ckpt_metric={args.ckpt_metric}  "
          f"heuristic_weight={args.heuristic_weight}")

    if args.build_data:
        pipe = GraphDataPipeline(
            folder_data=args.folder_raw_data,
            list_subset_train=args.subset_train,
            dataset_type=args.dataset_type,
            kind_of_data=args.kind_of_data,
            unreachable_state_value=args.unreachable_state_value,
            max_percentage_per_class=args.max_percentage_per_class,
            test_size=args.test_size,
            use_goal=args.use_goal,
            use_depth=args.use_depth,
            random_state=args.seed,
            # The one build-time divergence: keep unreachable rows in
            # samples.pt; the include/filter decision is made at load time
            # by prepare_samples_plus, so one build serves both settings.
            remove_unreachable_goal_states=False,
        )
        pipe.save(out_dir=data_path, extra_params={"plus": True})

    data = torch.load(data_path, weights_only=False)
    train_samples = data["train_samples"].copy()
    test_samples = data["test_samples"].copy()

    if args.include_unreachable:
        n_ur = count_unreachable(train_samples, args.unreachable_state_value) \
             + count_unreachable(test_samples, args.unreachable_state_value)
        if n_ur == 0:
            raise SystemExit(
                "[plus] --include-unreachable is on but samples.pt contains "
                "no unreachable rows — it was built by the baseline (which "
                "filters them at build time). Rebuild once with "
                "--build-data true (plus builds keep them)."
            )

    train_c, test_c, params = prepare_samples_plus(
        train_samples,
        test_samples,
        args.unreachable_state_value,
        dynamic_max_depth=args.dynamic_max_depth,
        include_unreachable=args.include_unreachable,
        unreachable_cap=args.unreachable_cap,
        seed=args.seed,
    )
    print(f"[plus] scaling params: {params}")

    if args.verbose:
        print("Train values:"); print_values(train_c)
        print("Test values:");  print_values(test_c)

    train_loader, val_loader = get_dataloaders(
        train_c, test_c, batch_size=args.batch_size
    )

    m = select_model_plus(
        args.use_goal, args.use_depth,
        bitmask=args.dataset_type == KEYWORD_BITMASK,
    )
    # INV-1 instrumentation: let evaluate() split metrics by reachability.
    # Unreachable samples carry the scaled target f(MAX_DEPTH); no reachable
    # state can alias it (max reachable distance < MAX_DEPTH by headroom).
    m.unreachable_target_value = (
        params["max_depth"] * params["slope"] + params["intercept"]
    )
    m.scale_params = params
    m.unreachable_loss_weight = args.unreachable_loss_weight
    if args.unreachable_loss_weight != 1.0 and args.include_unreachable:
        n_ur = params.get("train_unreachable_kept", 0)
        n_tot = len(train_c)
        u = args.unreachable_loss_weight
        mass = u * n_ur / (u * n_ur + (n_tot - n_ur))
        params["unreachable_loss_weight"] = u
        params["unreachable_effective_loss_mass"] = mass
        print(f"[plus] unreachable loss weight U={u}: effective loss mass "
              f"{100 * mass:.2f}% ({n_ur}/{n_tot} samples)")

    if args.train:
        m.train(
            train_loader,
            val_loader,
            n_epochs=args.n_train_epochs,
            checkpoint_dir=path_model,
            model_name=args.model_name,
            ckpt_metric=args.ckpt_metric,
        )

    # Load the chosen-metric checkpoint (the file the planner consumes) and
    # export ONNX from it — identical interface to the baseline export.
    m.load_model(f"{path_model}/{args.model_name}.pt")

    write_c_file(
        path_model, args.model_name, args.normalization_constants_name,
        params, heuristic_weight=args.heuristic_weight,
    )
    record_params_in_history(path_model, params, args.heuristic_weight)

    kwargs = {"th": params["slope"] / 2}
    metrics = m.evaluate(val_loader, verbose=args.verbose, **kwargs)
    print(f"[plus] exported-checkpoint val metrics: "
          f"{ {k: round(v, 4) for k, v in metrics.items()} }")

    onnx_model_path = f"{path_model}/{args.model_name}.onnx"
    m.to_onnx(onnx_model_path, args.use_goal, args.use_depth)

    with open(f"{path_model}/{args.model_name}_info.txt", "w",
              encoding="utf-8") as fh:
        for name, value in vars(args).items():
            fh.write(f"{name} = {value}\n")

    return onnx_model_path


if __name__ == "__main__":
    main(parse_args())
