import os
import time
import argparse
import subprocess
import concurrent.futures
import multiprocessing

def find_training_data_folders(batch_root):
    models_root = os.path.join(batch_root, "_models")
    training_data_folders = []

    for root, dirs, _ in os.walk(models_root):
        if 'training_data' in dirs:
            training_data_path = os.path.join(root, 'training_data')
            training_data_folders.append(training_data_path)

    return training_data_folders

def run_training(training_data_folder, batch_root):
    if not os.path.isdir(training_data_folder):
        print(f"[ERROR] Training data folder not found: {training_data_folder}")
        return

    instance_names = sorted([
        name for name in os.listdir(training_data_folder)
        if os.path.isdir(os.path.join(training_data_folder, name))
    ])
    if not instance_names:
        print(f"[WARNING] No training instances found in {training_data_folder}")
        return

    model_dir = os.path.dirname(training_data_folder)


    cmd = [
        "python3", "lib/epistemic_rl/__main__.py",
        "--folder-raw-data", training_data_folder,
        "--subset-train", *instance_names,
        "--dir-save-model", model_dir,
    ]

    #print(f"[INFO] Launching training for {training_data_folder}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        prefix = f"[{os.path.basename(model_dir)}]".ljust(20)
        print(f"{prefix} Showing one log line every 15 seconds...")

        last_print_time = 0  # epoch time

        for line in iter(process.stdout.readline, ''):
            now = time.time()
            if now - last_print_time >= 15:
                print(f"{prefix} {line.strip()}")
                last_print_time = now

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            print(f"{prefix} [SUCCESS] Training completed.")
        else:
            print(f"{prefix} [ERROR] Training failed with code {return_code}")

    except Exception as e:
        print(f"[ERROR] Unexpected error during training {training_data_folder}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run training in parallel for all training_data folders under a batch root."
    )
    parser.add_argument(
        "batch_root",
        help="Path to batch folder (e.g., exp/aaai26/batch1)"
    )

    args = parser.parse_args()

    training_data_folders = find_training_data_folders(args.batch_root)
    if not training_data_folders:
        print(f"[ERROR] No training_data folders found under: {args.batch_root}/_models/")
        return

    max_workers = min(multiprocessing.cpu_count(), len(training_data_folders))
    #print(f"[INFO] Running with up to {max_workers} parallel threads.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_training, folder, args.batch_root)
            for folder in training_data_folders
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # trigger exceptions if any

if __name__ == "__main__":
    main()
