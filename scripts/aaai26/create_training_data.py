import os
import argparse
import subprocess
import shutil
import re
import threading
import time

def create_models_folder(base_folder, domain_name):
    models_folder = os.path.join(base_folder, "_models", domain_name, "training_data")
    os.makedirs(models_folder, exist_ok=True)
    return models_folder

def process_file(deep_exe, file_path, target_folder, no_goal):
    file_name = os.path.basename(file_path)
    instance_name = os.path.splitext(file_name)[0]
    output_target = os.path.join(target_folder, instance_name)

    command = [
        deep_exe,
        file_path,
        "--dataset",
        "--dataset_depth", "25"
    ]

    if not no_goal:
        command.append("--dataset_merged")

    print(f"Command is {command}")

    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        output = result.stdout
        exit_code = result.returncode

        if exit_code == 3:
            print(f"[ERROR] No goals found in file {file_name}. Please regenerate this training instance.")
            return
        elif exit_code not in (0, 2):
            print(f"[FAILURE] Command failed for {file_name} with exit code {exit_code}")
            print(f"Output:\n{output}")
            return

        match = re.search(r'Dataset stored in (.+?) folder\.', output)
        if not match:
            print(f"[WARNING] Could not find output folder in C++ output for {file_name}. Skipping.")
            return

        cpp_output_folder = match.group(1).strip()

        if os.path.exists(output_target):
            shutil.rmtree(output_target)
        shutil.move(cpp_output_folder, output_target)

        project_root = os.getcwd()
        old_path_rel = os.path.relpath(cpp_output_folder, project_root).replace("\\", "/")
        new_path_rel = os.path.relpath(output_target, project_root).replace("\\", "/")

        # Update the only CSV file in the folder with relative path
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
                        # print(f"[UPDATED] CSV: {csv_path}")
                except Exception as e:
                    print(f"[WARNING] Could not update CSV {csv_path}: {e}")

        print(f"[SUCCESS] Generated training data in: {output_target}")

    except Exception as e:
        print(f"[ERROR] Unexpected error for {file_name}: {e}")


def run_cpp_on_training_files_multithreaded(deep_exe, training_folder, models_folder, no_goal):
    if not os.path.isdir(training_folder):
        raise FileNotFoundError(f"Training folder not found: {training_folder}")

    files = [
        os.path.join(training_folder, f)
        for f in os.listdir(training_folder)
        if os.path.isfile(os.path.join(training_folder, f))
    ]

    threads = []
    for i, file_path in enumerate(files):
        thread = threading.Thread(
            target=process_file,
            args=(deep_exe, file_path, models_folder, no_goal)
        )
        threads.append((thread, i))

    for thread, i in threads:
        thread.start()
        time.sleep(5)  # delay start to avoid C++ simultaneous folder creation

    for thread, _ in threads:
        thread.join()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run deep executable on all files in <base_folder>/<domain_name>/Training and "
            "store results in <base_folder>/_models/<domain_name>/training_data/.\n"
            "IMPORTANT: This script is assumed to be run from the root of the project folder."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("base_folder", help="Root folder containing the <domain_name>/Training and _models/ folders")
    parser.add_argument("domain_name", help="Domain name (folder under base_folder containing 'Training')")
    parser.add_argument("deep_exe", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Run without the --dataset_merged argument")


    args = parser.parse_args()

    #print(">>> Running from project root. Make sure the path structure is correct.")

    domain_folder = os.path.join(args.base_folder, args.domain_name)
    training_folder = os.path.join(domain_folder, "Training")
    models_root = os.path.join(args.base_folder, "_models", args.domain_name)
    #trained_model_file = os.path.join(models_root, "distance_estimator.onnx")

    #if os.path.isfile(trained_model_file):
    #    print(f"Model already exists in {models_root}. Skipping all files.")
    #    return

    models_folder = create_models_folder(args.base_folder, args.domain_name)
    run_cpp_on_training_files_multithreaded(args.deep_exe, training_folder, models_folder, args.no_goal)

if __name__ == "__main__":
    main()
