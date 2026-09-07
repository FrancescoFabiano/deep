import os
import sys
import time
import argparse
import subprocess
import concurrent.futures
import multiprocessing

# The multi-strategy generator's directory level. The GNN distance estimator
# reads the flat S_DFS layout only (one state in, one score out; no strategies).
STRATEGY_DIRS = {"BFS", "DFS", "S_DFS", "HFS"}


def find_training_data_folders(batch_root):
    models_root = os.path.join(batch_root, "_models")
    training_data_folders = []

    for root, dirs, _ in os.walk(models_root):
        if "training_data" in dirs:
            training_data_path = os.path.join(root, "training_data")
            training_data_folders.append(training_data_path)

    return training_data_folders


def run_training(training_data_folder, batch_root, no_goal, dataset_type):
    if not os.path.isdir(training_data_folder):
        print(f"[ERROR] Training data folder not found: {training_data_folder}")
        return

    instance_names = sorted(
        [
            name
            for name in os.listdir(training_data_folder)
            if os.path.isdir(os.path.join(training_data_folder, name))
        ]
    )
    if not instance_names:
        print(f"[WARNING] No training instances found in {training_data_folder}")
        return
    strat = sorted(set(instance_names) & STRATEGY_DIRS)
    if strat:
        print(f"[ERROR] {training_data_folder} holds the per-strategy layout {strat}; the "
              f"GNN reads the flat <training_data>/<instance>/ layout (generate WITHOUT "
              f"--dataset-generation). Skipping.")
        return

    model_dir = os.path.dirname(training_data_folder)

    cmd = [
        sys.executable,          # the interpreter running this script, not whatever `python3` is
        "lib/gnn_handler/__main__.py",
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
    ]

    if no_goal:
        # Separated data: the state DOT is goal-free, so the instance's
        # goal_tree.dot is fed as a second graph (both halves are required by
        # the trainer; sending only the first was why this path never ran).
        cmd += ["--kind-of-data", "separated", "--use-goal", "true"]

    print(" ".join(cmd))
    # print(f"[INFO] Launching training for {training_data_folder}")

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        prefix = f"[{os.path.basename(model_dir)}]".ljust(20)
        print(f"{prefix} Showing one log line every 15 seconds...")

        last_print_time = 0  # epoch time
        from collections import deque
        recent = deque(maxlen=40)   # the tail of the child's output, for failures

        for line in iter(process.stdout.readline, ""):
            recent.append(line.rstrip())
            now = time.time()
            if now - last_print_time >= 15:
                print(f"{prefix} {line.strip()}")
                last_print_time = now

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            print(f"{prefix} [SUCCESS] Training completed.")
        else:
            # A swallowed traceback is a failure nobody can act on: show the tail.
            print(f"{prefix} [ERROR] Training failed with code {return_code}; last "
                  f"{len(recent)} lines of its output:")
            for ln in recent:
                print(f"{prefix}   {ln}")

    except Exception as e:
        print(f"[ERROR] Unexpected error during training {training_data_folder}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run training in parallel for all training_data folders under a batch root."
    )
    parser.add_argument(
        "batch_root", help="Path to batch folder (e.g., exp/gnn_exp/batch1)"
    )
    parser.add_argument(
        "--no_goal",
        action="store_true",
        help="Set '--kind-of-data separated' for model training",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["MAPPED", "HASHED", "BITMASK"],
        default="HASHED",
        help="Specifies how node labels are represented in dataset generation. Options: MAPPED (compact integer mapping), HASHED (standard hashing), or BITMASK (bitmask representation of fluents and goals).",
    )

    args = parser.parse_args()

    training_data_folders = find_training_data_folders(args.batch_root)
    if not training_data_folders:
        print(
            f"[ERROR] No training_data folders found under: {args.batch_root}/_models/"
        )
        return

    max_workers = min(multiprocessing.cpu_count(), len(training_data_folders))
    # print(f"[INFO] Running with up to {max_workers} parallel threads.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_training, folder, args.batch_root, args.no_goal, args.dataset_type
            )
            for folder in training_data_folders
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # trigger exceptions if any


if __name__ == "__main__":
    main()
