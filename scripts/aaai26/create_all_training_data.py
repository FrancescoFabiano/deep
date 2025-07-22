import os
import argparse
import subprocess

def find_domains_with_training(batch_dir):
    """Find all immediate subdirectories of batch_dir that contain a 'Training' folder."""
    domains = []
    for entry in os.listdir(batch_dir):
        full_path = os.path.join(batch_dir, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, "Training")):
            domains.append(entry)
    return domains

def main():
    parser = argparse.ArgumentParser(
        description="Run create_models.py on all domain folders inside a batch folder."
    )
    parser.add_argument("batch_path", help="Path to the batch folder (e.g., exp/aaai26/exp_batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Omit --dataset_merged from C++ execution")

    args = parser.parse_args()
    batch_path = os.path.abspath(args.batch_path)

    if not os.path.isdir(batch_path):
        print(f"Error: {batch_path} is not a valid directory.")
        return

    domains = find_domains_with_training(batch_path)
    if not domains:
        print("No domains with Training folders found.")
        return

    print(f"Found {len(domains)} domain(s) with Training folders.")

    for domain_name in domains:
        domain_path = os.path.join(batch_path, domain_name)
        print(f"\n=== Processing: {domain_name} ===")

        cmd = [
            "python",
            "scripts/create_models.py",
            domain_path,
            domain_name,
            args.deep_exe
        ]

        if args.no_goal:
            cmd.append("--no_goal")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {domain_name}: {e}")

if __name__ == "__main__":
    main()
