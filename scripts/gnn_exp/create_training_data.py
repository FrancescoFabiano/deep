import os
import argparse
import subprocess
import shutil
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from datetime import datetime

LOG_LOCK = threading.Lock()


def create_models_folder(base_folder, domain_name):
    models_folder = os.path.join(base_folder, "_models", domain_name, "training_data")
    os.makedirs(models_folder, exist_ok=True)
    return models_folder


def ensure_failed_dirs(base_folder, domain_name):
    failed_root = os.path.join(base_folder, "_models", domain_name, "_failed")
    logs_dir = os.path.join(failed_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return failed_root, logs_dir


def write_failed_record(failed_root, record):
    """Append a single JSON record to failed_items.jsonl in a thread-safe way."""
    path = os.path.join(failed_root, "failed_items.jsonl")
    with LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_instance_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def save_attempt_log(logs_dir, instance_name, seed, output):
    """
    Save an attempt log (stdout/stderr merged) into the global logs_dir.
    Returns the absolute path to the log file.
    """
    safe_instance = _safe_instance_name(instance_name)
    log_path = os.path.join(logs_dir, f"{safe_instance}__seed_{seed}.log")
    with LOG_LOCK:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output or "")
    return os.path.abspath(log_path)


def run_cpp_once(
    deep_exe, file_path, no_goal, depth, discard_factor, seed, dataset_type
):
    """Run the C++ tool once with a given seed. Returns (exit_code, output_string)."""
    command = [
        deep_exe,
        file_path,
        "-b",       #Bisimulation mode
        "--dataset",
        "--dataset_depth",
        str(depth),
        "--dataset_discard_factor",
        str(discard_factor),
        "--dataset_seed",
        str(seed),
        "--dataset_type",
        str(dataset_type),
    ]
    if no_goal:
        command.append("--dataset_separated")

    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout


def postprocess_and_move_output(cpp_output_folder, output_target):
    if os.path.exists(output_target):
        shutil.rmtree(output_target)
    shutil.move(cpp_output_folder, output_target)

    # Fix CSV paths that reference the original C++ output folder
    project_root = os.getcwd()
    old_path_rel = os.path.relpath(cpp_output_folder, project_root).replace("\\", "/")
    new_path_rel = os.path.relpath(output_target, project_root).replace("\\", "/")

    for f in os.listdir(output_target):
        if f.endswith(".csv"):
            csv_path = os.path.join(output_target, f)
            try:
                with open(csv_path, "r", encoding="utf-8") as file:
                    content = file.read()
                updated = content.replace(old_path_rel, new_path_rel)
                if updated != content:
                    with open(csv_path, "w", encoding="utf-8") as file:
                        file.write(updated)
            except Exception as e:
                print(f"[WARNING] Could not update CSV {csv_path}: {e}")


def _write_dataset_summary_logs(
    dataset_dir, instance_name, success_seed, failed_attempts, attempt_logs
):
    """
    Create per-dataset logs:
      - <dataset>/logs/  (move attempt logs here)
      - <dataset>/dataset_log.json (summary with success + failed)
      - <dataset>/SEEDS.txt (quick summary)
    """
    os.makedirs(dataset_dir, exist_ok=True)
    per_dataset_logs_dir = os.path.join(dataset_dir, "logs")
    os.makedirs(per_dataset_logs_dir, exist_ok=True)

    moved_logs = []
    for lp in attempt_logs:
        if not lp:
            continue
        try:
            # Keep same filename, move/copy into per-dataset logs
            dst = os.path.join(per_dataset_logs_dir, os.path.basename(lp))
            # Move if source exists and not already in place
            if os.path.abspath(lp) != os.path.abspath(dst) and os.path.exists(lp):
                shutil.move(lp, dst)
            moved_logs.append(dst)
        except Exception as e:
            print(f"[WARNING] Could not move log {lp} to dataset logs: {e}")

    # JSON summary
    summary = {
        "instance": instance_name,
        "success_seed": success_seed,
        "failed_seeds": failed_attempts,
        "attempt_logs": moved_logs,
        #        "time_utc": datetime.utcnow().isoformat() + "Z",
    }
    summary_json = os.path.join(dataset_dir, "dataset_log.json")
    with LOG_LOCK:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # Quick text summary
    seeds_txt = os.path.join(dataset_dir, "SEEDS.txt")
    with LOG_LOCK:
        with open(seeds_txt, "w", encoding="utf-8") as f:
            f.write(f"Instance: {instance_name}\n")
            f.write(f"SUCCESS SEED: {success_seed}\n")
            if failed_attempts:
                f.write("FAILED SEEDS:\n")
                for s in failed_attempts:
                    f.write(f"  - {s}\n")
            else:
                f.write("FAILED SEEDS: (none)\n")


def process_file_with_retries(
    deep_exe,
    file_path,
    target_folder,
    no_goal,
    depth,
    discard_factor,
    seeds,
    dataset_type,
    failed_root,
    global_logs_dir,
):
    file_name = os.path.basename(file_path)
    instance_name = os.path.splitext(file_name)[0]
    output_target = os.path.join(target_folder, instance_name)

    failed_seeds = []
    attempt_log_paths = []

    for attempt_idx, seed in enumerate(seeds, start=1):
        try:
            exit_code, output = run_cpp_once(
                deep_exe, file_path, no_goal, depth, discard_factor, seed, dataset_type
            )
        except Exception as e:
            exit_code, output = -999, f"Python exception while invoking C++: {e}"

        # Always save a log per attempt (global, then later moved into the dataset folder on success)
        log_path = save_attempt_log(global_logs_dir, instance_name, seed, output)
        attempt_log_paths.append(log_path)

        if exit_code in (0, 2):
            # Success – find where the tool wrote its output
            match = re.search(r"Dataset stored in (.+?) folder\.", output or "")
            if not match:
                print(
                    f"[WARNING] Could not find output folder in C++ output for {file_name} (seed {seed})."
                )
                return  # Stop – don’t retry for missing folder

            cpp_output_folder = match.group(1).strip()
            try:
                postprocess_and_move_output(cpp_output_folder, output_target)
                # Now that the dataset is in its final folder, create per-dataset logs + summary and move attempt logs into it
                _write_dataset_summary_logs(
                    dataset_dir=output_target,
                    instance_name=instance_name,
                    success_seed=seed,
                    failed_attempts=failed_seeds,
                    attempt_logs=attempt_log_paths,
                )
                print(f"[SUCCESS] {file_name} with seed {seed} → {output_target}")
                return
            except Exception as e:
                print(
                    f"[WARNING] Moving/patching failed for {file_name} (seed {seed}): {e}."
                )
                return  # Stop – don’t retry for patching failure

        elif exit_code == 3:
            # Retry with the next seed
            print(
                f"[RETRY] No goals found in {file_name} (seed {seed}), trying next seed..."
            )
            failed_seeds.append(seed)
            if attempt_idx < len(seeds):
                continue
            else:
                # All seeds exhausted → record as no goals
                write_failed_record(
                    failed_root,
                    {
                        "instance": instance_name,
                        "file": file_path,
                        "reason": "no_goals_all_seeds",
                        "seeds_tried": list(seeds),
                        #                    "time_utc": datetime.utcnow().isoformat() + "Z",
                    },
                )
                print(
                    f"[FAILED AFTER RETRIES] {file_name} → no goals with all seeds {seeds}."
                )
                return

        else:
            # Any other failure: don’t retry
            print(
                f"[FAILURE] {file_name} (seed {seed}) exit {exit_code}. See log: {log_path}"
            )
            write_failed_record(
                failed_root,
                {
                    "instance": instance_name,
                    "file": file_path,
                    "reason": f"failure_exit_{exit_code}",
                    "seed": seed,
                    "attempt": attempt_idx,
                    "log_path": log_path,
                    #               "time_utc": datetime.utcnow().isoformat() + "Z",
                },
            )
            return


def run_cpp_on_training_files_multithreaded(
    deep_exe,
    training_folder,
    models_folder,
    no_goal,
    depth,
    discard_factor,
    seeds,
    dataset_type,
    failed_root,
    logs_dir,
):
    if not os.path.isdir(training_folder):
        raise FileNotFoundError(f"Training folder not found: {training_folder}")

    files = [
        os.path.join(training_folder, f)
        for f in os.listdir(training_folder)
        if os.path.isfile(os.path.join(training_folder, f))
    ]

    # Determine number of threads to use
    max_threads = min(8, max(1, (os.cpu_count() or 4) - 2))

    def wrapper(file_path):
        time.sleep(5)  # Delay start like in original code
        process_file_with_retries(
            deep_exe,
            file_path,
            models_folder,
            no_goal,
            depth,
            discard_factor,
            seeds,
            dataset_type,
            failed_root,
            logs_dir,
        )

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(wrapper, file_path) for file_path in files]
        for future in as_completed(futures):
            future.result()  # Raise exceptions if any occurred


def parse_seeds(seeds_arg):
    if not seeds_arg:
        return [42, 1337, 2024]
    if isinstance(seeds_arg, (list, tuple)):
        return [int(x) for x in seeds_arg]
    # Comma or space separated
    parts = re.split(r"[,\s]+", str(seeds_arg).strip())
    return [int(p) for p in parts if p != ""]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run deep executable on all files in <base_folder>/<domain_name>/Training and "
            "store results in <base_folder>/_models/<domain_name>/training_data/.\n"
            "On failure, retry each problem with alternative seeds and record failures.\n"
            "Also, for every successfully saved dataset, write per-dataset logs and a seed summary inside the dataset folder.\n"
            "IMPORTANT: Run from the root of the project folder."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "base_folder",
        help="Root folder containing the <domain_name>/Training and _models/ folders",
    )
    parser.add_argument(
        "domain_name",
        help="Domain name (folder under base_folder containing 'Training')",
    )
    parser.add_argument("deep_exe", help="Path to the deep C++ executable")
    parser.add_argument(
        "--no_goal",
        action="store_true",
        help="Run with the --dataset_separated argument",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=25,
        help="Depth for dataset generation (default: 25)",
    )
    parser.add_argument(
        "--discard_factor",
        dest="discard_factor",
        type=float,
        default=0.4,
        help="Maximum discard factor (default: 0.4)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,1337,2024,23,31,47,59,73,89,101,137,149",
        help="Comma/space-separated list of seeds to try on failure (order respected). Default: 42,1337,2024,23,31,47,59,73,89,101,137,149",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["MAPPED", "HASHED", "BITMASK"],
        default="HASHED",
        help="Specifies how node labels are represented in dataset generation. Options: MAPPED (compact integer mapping), HASHED (standard hashing), or BITMASK (bitmask representation of fluents and goals).",
    )

    args = parser.parse_args()

    domain_folder = os.path.join(args.base_folder, args.domain_name)
    training_folder = os.path.join(domain_folder, "Training")

    models_folder = create_models_folder(args.base_folder, args.domain_name)
    failed_root, logs_dir = ensure_failed_dirs(args.base_folder, args.domain_name)
    seeds = parse_seeds(args.seeds)

    print(f"[INFO] Using seeds: {seeds}")
    print(
        f"[INFO] Failed attempts and global attempt logs will be stored in: {failed_root}"
    )
    print(
        f"[INFO] On success, per-dataset logs and seed summaries will be placed inside each dataset folder."
    )

    run_cpp_on_training_files_multithreaded(
        args.deep_exe,
        training_folder,
        models_folder,
        args.no_goal,
        args.depth,
        args.discard_factor,
        seeds,
        args.dataset_type,
        failed_root,
        logs_dir,
    )


if __name__ == "__main__":
    main()
