import os
import argparse
import subprocess

def find_domains_with_training(batch_dir, training_folder):
    """Recursively find all folders under batch_dir that contain the selected training folder."""
    domains = []
    for root, dirs, files in os.walk(batch_dir):
        if training_folder in dirs:
            rel_path = os.path.relpath(root, batch_dir)
            domains.append(rel_path)
    return sorted(domains)

def _parse_depth_map(spec: str) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = int(v)
    return out


def _depth_for(domain_name: str, args) -> int:
    """Per-domain depth. AUTHORITATIVE when --depth-map is given: a domain absent
    from the map FAILS LOUDLY rather than silently inheriting --depth.

    Why no fallback: --depth's historical value is 40, which is the UNFAITHFUL
    configuration for CC -- it is what blew past --dataset-max-creation and poisoned
    the tree (past the ceiling non-goals are dropped AND their parents inherit
    1e6 = unreachable). A new CC-like domain silently getting 40 would reintroduce
    that artifact for that domain, and the faithfulness gate might not catch it if
    the instance happens to enumerate. A loud failure beats a convenient default
    that can silently be wrong.
    """
    dm = _parse_depth_map(args.depth_map)
    if not dm:
        return args.depth          # no map given at all -> legacy single-depth mode
    if domain_name not in dm:
        raise SystemExit(
            f"[FATAL] domain '{domain_name}' has no depth in --depth-map "
            f"({args.depth_map!r}). Add one (CC-like ~25, SC-like ~40) -- refusing "
            f"to guess. Depth is what keeps generation under "
            f"--dataset-max-creation, and the historical default 40 is the "
            f"UNFAITHFUL setting for CC."
        )
    return dm[domain_name]


def main():
    parser = argparse.ArgumentParser(
        description="Run the per-domain training-data generator on all domain folders inside a batch folder."
    )
    parser.add_argument("batch_path", help="Path to the batch folder (e.g., exp/gnn_exp/batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Add --dataset_separated to the C++ execution")
    parser.add_argument("--strong_equality", action="store_true", help="Add --strong_equality to the C++ execution")
    parser.add_argument("--depth", type=int, default=25, help="Depth for dataset generation (default: 25)")
    parser.add_argument(
        "--discard_factor", type=float, default=0.4,
        help="Maximum discard factor (default: 0.4, unchanged -- this script is "
             "SHARED with gnn_exp and its default must not move under that "
             "pipeline's feet; callers that want faithful trees pass 0 explicitly). "
             "WARNING: the discard is BIASED, not uniform -- its probability rises "
             "with depth and gains +0.2 immediately after a goal is found "
             "(TrainingDataset.tpp:714-730), so it preferentially deletes the "
             "SHALLOW GOALS that make an instance easy, while writing the discarded "
             "states to the CSV as childless leaves. Measured on CC_2_2_3__pl_4 "
             "(planner BFS true optimal = 4): 0.4 -> delta_root 10, sterile 30 pct; "
             "0 -> delta_root 4, sterile 2.9 pct. Bound the tree with --depth/"
             "--depth-map instead, which keeps the solution path.")
    parser.add_argument(
        "--depth-map", dest="depth_map", default="",
        help="per-domain depth override, e.g. 'CC:25,SC:40,SCRich:40'. Falls back "
             "to --depth for domains not listed. Depth is what keeps generation "
             "under --dataset-max-creation (a HARD stop that also POISONS: past it "
             "non-goals are dropped AND their parents inherit 1e6 = unreachable), "
             "so it is per-domain: CC optimals are ~4-7, SC needs 40.")
    # Forwarded to the per-domain script (which generates per-instance random seeds)
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed used to generate per-instance seeds (default: 42)")
    parser.add_argument("--max_retries", type=int, default=15, help="Maximum attempts per instance in the called script (default: 15)")
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED",
                        help="How node labels are represented: MAPPED, HASHED, or BITMASK.")
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        default="training_data",
        help="Output dataset folder name under _models/<domain_name>/ (default: training_data)",
    )
    parser.add_argument(
        "--dataset-max-creation",
        dest="dataset_max_creation",
        type=int,
        default=60000,
        help="Maximum number of creations for dataset generation (default: 60000)",
    )
    parser.add_argument(
        "--training-folder",
        dest="training_folder",
        default="Training",
        help="Input folder name under each domain containing training instances (default: Training)",
    )
    # Path to the called script (your adapted one)
    parser.add_argument("--script_path", default="scripts/gnn_exp/create_training_data.py",
                        help="Path to the per-domain Python script to invoke.")

    args = parser.parse_args()
    batch_path = os.path.abspath(args.batch_path)

    if not os.path.isdir(batch_path):
        print(f"Error: {batch_path} is not a valid directory.")
        return

    domains = find_domains_with_training(batch_path, args.training_folder)
    if not domains:
        print(
            f"No domains with '{args.training_folder}' folders found (even recursively)."
        )
        return

    print(
        f"Found {len(domains)} domain(s) with '{args.training_folder}' folders."
    )


    print(f"[INFO] Base RNG seed: {args.seed}")
    print(f"[INFO] Max retries per instance: {args.max_retries}")
    print(
        f"[INFO] Failed attempts and global attempt logs will be stored in: {batch_path}/_models/<domain_name>/_failed"
    )
    print(
        f"[INFO] On success, per-dataset logs and seed summaries will be placed inside each dataset folder."
    )
    print(
        f"[INFO] Successful seeds will be appended to: {batch_path}/_models/<domain_name>/{args.dataset_name}/seeds.txt"
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
            "--depth", str(_depth_for(domain_name, args)),
            "--discard_factor", str(args.discard_factor),
            "--seed", str(args.seed),
            "--max_retries", str(args.max_retries),
            "--dataset_type", str(args.dataset_type),
            "--dataset-name", str(args.dataset_name),
            "--dataset-max-creation", str(args.dataset_max_creation),
            "--training-folder", str(args.training_folder),
        ]
        if args.no_goal:
            cmd.append("--no_goal")
        if args.strong_equality:
            cmd.append("--strong_equality")

        try:
            # Let the per-domain script print its own detailed progress & logs
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Per-domain script failed for {domain_name} (exit {e.returncode}).")

    print("\n=== Batch complete ===")

if __name__ == "__main__":
    main()
