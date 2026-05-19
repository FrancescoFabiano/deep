import argparse
import concurrent.futures
import multiprocessing
import os
import re
import subprocess
import time

from requests.compat import str


def find_training_data_folders(batch_root):
    models_root = os.path.join(batch_root, "_models")
    training_data_folders = []
    for root, dirs, _ in os.walk(models_root):
        if "training_data" in dirs:
            training_data_folders.append(os.path.join(root, "training_data"))
    return training_data_folders


def parse_onnx_frontier_size_values(raw_value):
    text = str(raw_value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("--onnx-frontier-size requires at least one value.")

    values = []
    seen = set()
    for token in [tok for tok in re.split(r"[,\s]+", text) if tok]:
        size = int(token)
        if size <= 0:
            raise ValueError("--onnx-frontier-size values must be > 0.")
        if size not in seen:
            seen.add(size)
            values.append(size)
    return values


def run_training(
    training_data_folder,
    no_goal,
    dataset_type,
    batch_size,
    eval_batch_size,
    n_train_epochs,
    n_checkpoints_evaluation,
    edge_label_buckets,
    max_regular_distance_for_reward,
    n_max_dataset_queries,
    onnx_frontier_sizes,
    train_random_frontier_ratio,
    train_random_frontier_with_failure_ratio,
    lr,
    weight_decay,
    max_grad_norm,
    early_stopping_patience_evals,
    failure_reward_value,
    train_frontier_jaccard_threshold,
    eval_frontier_jaccard_threshold,
    max_failure_states_per_dataset,
    build_data,
    build_eval_data,
    evaluate,
):
    if not os.path.isdir(training_data_folder):
        print(f"[ERROR] Missing training folder: {training_data_folder}")
        return

    instance_names = sorted(
        name
        for name in os.listdir(training_data_folder)
        if os.path.isdir(os.path.join(training_data_folder, name))
    )
    if not instance_names:
        print(f"[WARNING] Empty training folder: {training_data_folder}")
        return

    model_dir = os.path.dirname(training_data_folder)
    cmd = [
        "python3",
        "lib/rl_handler/__main__.py",
        "--folder-raw-data",
        training_data_folder,
        "--subset-train",
        *instance_names,
        "--dir-save-model",
        model_dir,
        "--dir-save-data",
        model_dir,
        "--dataset_type",
        dataset_type,
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(eval_batch_size),
        "--n-train-epochs",
        str(n_train_epochs),
        "--n-checkpoints-evaluation",
        str(n_checkpoints_evaluation),
        "--K",
        str(edge_label_buckets),
        "--build-data",
        str(bool(build_data)).lower(),
        "--failure-reward-value",
        str(failure_reward_value),
        "--max-regular-distance-for-reward",
        str(max_regular_distance_for_reward),
        "--n-max-dataset-queries",
        str(n_max_dataset_queries),
        "--onnx-frontier-size",
        *[str(v) for v in onnx_frontier_sizes],
        "--train-random-frontier-ratio",
        str(train_random_frontier_ratio),
        "--train-random-frontier-with-failure-ratio",
        str(train_random_frontier_with_failure_ratio),
        "--build-eval-data",
        str(bool(build_eval_data)).lower(),
        "--evaluate",
        str(bool(evaluate)).lower(),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--max-grad-norm",
        str(max_grad_norm),
        "--early-stopping-patience-evals",
        str(early_stopping_patience_evals),
        "--train-frontier-jaccard-threshold",
        str(train_frontier_jaccard_threshold),
        "--eval-frontier-jaccard-threshold",
        str(eval_frontier_jaccard_threshold),
        "--max-failure-states-per-dataset",
        str(max_failure_states_per_dataset),
    ]
    if no_goal:
        cmd.extend(["--kind-of-data", "separated"])
    print(" ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    prefix = f"[{os.path.basename(model_dir)}]".ljust(20)
    print(f"{prefix} Running RL training...")

    last_print = 0.0
    for line in iter(process.stdout.readline, ""):
        now = time.time()
        if now - last_print >= 15:
            print(f"{prefix} {line.strip()}")
            last_print = now
    process.stdout.close()
    rc = process.wait()
    if rc == 0:
        print(f"{prefix} [SUCCESS]")
    else:
        print(f"{prefix} [ERROR] return code {rc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL model training in parallel for all _models/*/training_data folders."
    )
    parser.add_argument("batch_root")
    parser.add_argument("--no_goal", action="store_true")
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation dataloaders.",
    )
    parser.add_argument("--n-train-epochs", type=int, default=100)
    parser.add_argument(
        "--n-checkpoints-evaluation",
        type=int,
        default=5,
        help="Number of uniformly-distributed evaluation checkpoints during training.",
    )
    parser.add_argument(
        "--K",
        "--edge-label-buckets",
        dest="edge_label_buckets",
        type=int,
        default=128,
        help=(
            "Fixed number of edge-label buckets (K) passed to RL model "
            "edge embeddings."
        ),
    )
    parser.add_argument("--max-regular-distance-for-reward", type=float, default=50.0)
    parser.add_argument(
        "--n-max-dataset-queries",
        type=int,
        default=500,
        help="Max generated evaluation strategy frontiers per dataset.",
    )
    parser.add_argument(
        "--onnx-frontier-size",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help=(
            "Frontier size cap used during training/evaluation (and optional ONNX export). "
            "Provide one or more integers (e.g. 32 or 16 32 64)."
        ),
    )
    parser.add_argument(
        "--train-random-frontier-ratio",
        type=float,
        default=0.1,
        help=(
            "Random frontiers generated as a ratio of finalized "
            "greedy/conservative/common frontiers."
        ),
    )
    parser.add_argument(
        "--train-random-frontier-with-failure-ratio",
        type=float,
        default=0.4,
        help=(
            "Among random frontiers, target this fraction to include "
            "at least one failure state."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--failure-reward-value", type=float, default=-1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=50)
    parser.add_argument(
        "--train-frontier-jaccard-threshold",
        type=float,
        default=0.6,
        help="Drop near-duplicate train frontiers with Jaccard similarity >= threshold.",
    )
    parser.add_argument(
        "--eval-frontier-jaccard-threshold",
        type=float,
        default=0.3,
        help="Drop near-duplicate evaluation frontiers with Jaccard similarity >= threshold.",
    )
    parser.add_argument(
        "--build-data",
        choices=["true", "false"],
        default="true",
        help="Whether to rebuild training samples before training.",
    )
    parser.add_argument(
        "--build-eval-data",
        choices=["true", "false"],
        default="true",
        help=(
            "Whether to rebuild evaluation query definitions "
            "(train/test x random/fifo/stress) before evaluation."
        ),
    )
    parser.add_argument(
        "--evaluate",
        choices=["true", "false"],
        default="true",
        help=(
            "If evaluation data are present, perform strategy evaluation on train/test splits."
        ),
    )
    parser.add_argument(
        "--max-failure-states-per-dataset",
        type=float,
        default=1.0,
        help=(
            "Keep all no-failure train frontiers; keep with-failure frontiers up to this ratio "
            "relative to no-failure ones."
        ),
    )
    args = parser.parse_args()
    if args.train_random_frontier_ratio < 0.0 or args.train_random_frontier_ratio > 1.0:
        raise ValueError("--train-random-frontier-ratio must be in [0.0, 1.0].")
    if (
        args.train_random_frontier_with_failure_ratio < 0.0
        or args.train_random_frontier_with_failure_ratio > 1.0
    ):
        raise ValueError("--train-random-frontier-with-failure-ratio must be in [0.0, 1.0].")
    if (
        args.train_frontier_jaccard_threshold < 0.0
        or args.train_frontier_jaccard_threshold > 1.0
    ):
        raise ValueError("--train-frontier-jaccard-threshold must be in [0.0, 1.0].")
    if (
        args.eval_frontier_jaccard_threshold < 0.0
        or args.eval_frontier_jaccard_threshold > 1.0
    ):
        raise ValueError("--eval-frontier-jaccard-threshold must be in [0.0, 1.0].")
    if (
        args.max_failure_states_per_dataset < 0.0
        or args.max_failure_states_per_dataset > 1.0
    ):
        raise ValueError("--max-failure-states-per-dataset must be in [0.0, 1.0].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0.")
    if args.n_train_epochs <= 0:
        raise ValueError("--n-train-epochs must be > 0.")
    if args.n_checkpoints_evaluation <= 0:
        raise ValueError("--n-checkpoints-evaluation must be > 0.")
    if args.n_checkpoints_evaluation > args.n_train_epochs:
        raise ValueError(
            "--n-checkpoints-evaluation cannot exceed --n-train-epochs."
        )
    onnx_frontier_sizes = parse_onnx_frontier_size_values(args.onnx_frontier_size)
    folders = find_training_data_folders(args.batch_root)
    if not folders:
        print(f"[ERROR] No training_data folders found in {args.batch_root}/_models/")
        return

    max_workers = min(multiprocessing.cpu_count(), len(folders))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_training,
                folder,
                args.no_goal,
                args.dataset_type,
                args.batch_size,
                args.eval_batch_size,
                args.n_train_epochs,
                args.n_checkpoints_evaluation,
                args.edge_label_buckets,
                args.max_regular_distance_for_reward,
                args.n_max_dataset_queries,
                onnx_frontier_sizes,
                args.train_random_frontier_ratio,
                args.train_random_frontier_with_failure_ratio,
                args.lr,
                args.weight_decay,
                args.max_grad_norm,
                args.early_stopping_patience_evals,
                args.failure_reward_value,
                args.train_frontier_jaccard_threshold,
                args.eval_frontier_jaccard_threshold,
                args.max_failure_states_per_dataset,
                args.build_data.lower() == "true",
                args.build_eval_data.lower() == "true",
                args.evaluate.lower() == "true",
            )
            for folder in folders
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
