import os
import argparse
import subprocess

def find_domains_with_training(batch_dir):
    """Recursively find all folders under batch_dir that contain a 'Training' folder."""
    domains = []
    for root, dirs, files in os.walk(batch_dir):
        if "Training" in dirs:
            rel_path = os.path.relpath(root, batch_dir)
            domains.append(rel_path)
    return sorted(domains)

def main():
    parser = argparse.ArgumentParser(
        description="Run the per-domain training-data generator on all domain folders inside a batch folder."
    )
    parser.add_argument("batch_path", help="Path to the batch folder (e.g., exp/gnn_exp/batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Add --dataset_separated to the C++ execution")
    parser.add_argument("--depth", type=int, default=25, help="Depth for dataset generation (default: 25)")
    parser.add_argument("--discard_factor", type=float, default=0.4, help="Maximum discard factor (default: 0.4)")
    # Forwarded to the per-domain script (which generates per-instance random seeds)
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed used to generate per-instance seeds (default: 42)")
    parser.add_argument("--max_retries", type=int, default=15, help="Maximum attempts per instance in the called script (default: 15)")
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED",
                        help="How node labels are represented: MAPPED, HASHED, or BITMASK.")
    # Path to the called script (your adapted one)
    parser.add_argument("--script_path", default="scripts/gnn_exp/create_training_data.py",
                        help="Path to the per-domain Python script to invoke.")

    args = parser.parse_args()
    batch_path = os.path.abspath(args.batch_path)

    if not os.path.isdir(batch_path):
        print(f"Error: {batch_path} is not a valid directory.")
        return

    domains = find_domains_with_training(batch_path)
    if not domains:
        print("No domains with Training folders found (even recursively).")
        return

    print(f"Found {len(domains)} domain(s) with Training folders.")


    print(f"[INFO] Base RNG seed: {args.seed}")
    print(f"[INFO] Max retries per instance: {args.max_retries}")
    print(
        f"[INFO] Failed attempts and global attempt logs will be stored in: {batch_path}/_models/<domain_name>/_failed"
    )
    print(
        f"[INFO] On success, per-dataset logs and seed summaries will be placed inside each dataset folder."
    )
    print(
        f"[INFO] Successful seeds will be appended to: {batch_path}/_models/<domain_name>/training_data/seeds.txt"
    )

    for domain_rel_path in domains:
        domain_name = domain_rel_path  # relative path like 'foo/bar'
        print(f"\n=== Processing: {domain_name} ===")

        # Call the adapted per-domain script; it handles seed generation, retries, and logging.
        cmd = [
            "python3",
            args.script_path,
            batch_path,             # base_folder
            domain_name,            # domain_name
            args.deep_exe,          # deep_exe
            "--depth", str(args.depth),
            "--discard_factor", str(args.discard_factor),
            "--seed", str(args.seed),
            "--max_retries", str(args.max_retries),
            "--dataset_type", str(args.dataset_type),
        ]
        if args.no_goal:
            cmd.append("--no_goal")

        try:
            # Let the per-domain script print its own detailed progress & logs
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Per-domain script failed for {domain_name} (exit {e.returncode}).")

    print("\n=== Batch complete ===")

if __name__ == "__main__":
    main()
