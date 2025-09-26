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
    return domains

def main():
    parser = argparse.ArgumentParser(
        description="Run create_training_data.py on all domain folders inside a batch folder."
    )
    parser.add_argument("batch_path", help="Path to the batch folder (e.g., exp/gnn_exp/batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Add --dataset_separated from C++ execution")
    parser.add_argument("--depth", type=int, default=25, help="Depth for dataset generation (default: 25)")
    parser.add_argument("--discard_factor", dest="discard_factor", type=float, default=0.4, help="Maximum discard factor (default: 0.4)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,1337,2024,23,31,47,59,73,89,101,137,149",
        help="Comma/space-separated list of seeds to try on failure (order respected). Default: 42,1337,2024,23,31,47,59,73,89,101,137,149"
    )
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED", help="Specifies how node labels are represented in dataset generation. Options: MAPPED (compact integer mapping), HASHED (standard hashing), or BITMASK (bitmask representation of fluents and goals).")

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

    for domain_rel_path in domains:
        domain_abs_path = os.path.join(batch_path, domain_rel_path)
        domain_name = domain_rel_path  # relative path like 'foo/bar'

        print(f"\n=== Processing: {domain_name} ===")

        cmd = [
            "python3",
            "scripts/gnn_exp/create_training_data.py",
            batch_path,
            domain_name,
            args.deep_exe,
            "--depth", str(args.depth),
            "--discard_factor", str(args.discard_factor),
            "--seeds", str(args.seeds),
            "--dataset_type", str(args.dataset_type),
        ]

        if args.no_goal:
            cmd.append("--no_goal")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {domain_name}: {e}")

if __name__ == "__main__":
    main()
