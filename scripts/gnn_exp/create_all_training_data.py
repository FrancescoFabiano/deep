"""Generate training tables for every domain in a batch. THE one place generation
parameters live -- depth, discard, max_creation, the deep flags. `final_launcher.sh`
drives it and passes them explicitly. Nothing in lib/rl_handler re-specifies how to
generate; rl_handler only CONSUMES trees and CHECKS them (the faithfulness gate:
delta_root == optimal, poisoned_frac, fidelity floor). Scripts generate, the gate
validates, and they share no parameter.

DEPTH ALONE DOES NOT BOUND THE TREE  (measured 2026-07-15, CC_2_3_4__pl_7, depth 25)

Generating that instance at --depth 25 prints:

    Approximate number of nodes (exp(log)) = 1.15477e+40
    Threshold number of nodes = 100000
    Decision: using SPARSE DFS.

The tree is ~35 orders of magnitude past the ceiling, so the DFS is truncated. It
burns its visit budget in the DEEP region, records a goal at depth 22, and never
reaches the true optimal at depth 7. Result: delta_root=22 vs optimal 7 -- genuine
unfaithfulness, not a measurement artifact. Confirmed two ways: strict BFS returns
exactly 7 (combined_results/batch2/CC/bfs/train_BFS_strict.csv), and --strong_equality
never moved the optimal on any rung pl_3..pl_6 (it does change the search -- pl_5
expands 277 nodes weak vs 274 strict -- it just does not change the answer).

THE BINDING CEILING IS THE VISIT COUNT, NOT max_creation
`TrainingDataset.tpp:677` poisons on EITHER counter:

    if (m_current_nodes >= m_threshold_node_generation ||   // VISITS, default 100000
        m_added_to_dataset >= m_max_threshold_node_creation) {   // WRITES, 50000
      if (state.is_goal()) { add_to_dataset(...); return 0; }
      return m_failed_state;            // 1e6 = unreachable; does NOT recurse
    }

Past it a non-goal returns m_failed_state WITHOUT recursing, so any goal below it is
never reached. `m_threshold_node_generation` is set by --dataset_max_generation
(ArgumentParser.cpp:216) which NO caller passes -- it sits at its 100000 default while
we carefully thread the non-binding max_creation through.

Read a small dataset as the SYMPTOM, not the all-clear: hitting the 100k VISIT ceiling
stops all further additions, which is precisely what leaves the table stranded under
the 50k write cap. "Under 50k, so the ceiling did not bite" is exactly backwards.

SO: depth must be tight enough that the REACHABLE set fits under 100k visits. That is
a PER-INSTANCE property, not per-domain -- branching varies within a domain (CC_2_2_3
survives depth 25 only because low branching lets DFS reach the shallow goal before
the budget dies; CC_2_3_4, branching ~40, does not). --depth-map is per-domain and so
cannot express this; the open question is what depth (or visit budget) makes CC_2_3_4
faithful. The faithfulness gate is what tells you when you have it.
"""

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
        help="per-domain depth, e.g. 'CC:25,SC:40,SCRich:40'. AUTHORITATIVE when "
             "given: a domain absent from the map FAILS LOUDLY, it does NOT fall "
             "back to --depth (40 is the UNFAITHFUL setting for CC). Omit the map "
             "entirely for legacy single---depth mode. NOTE depth is necessary but "
             "NOT sufficient to bound the tree: the binding ceiling is the 100000 "
             "VISIT count (--dataset_max_generation, never passed), and CC_2_3_4 at "
             "depth 25 still estimates 1.15e40 nodes -> SPARSE DFS -> the true "
             "optimal is never reached. See the module docstring.")
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
        required=True,
        help="REQUIRED. Max nodes WRITTEN to the dataset (--dataset_max_creation). "
             "No default: a default here is a second opinion about how data was "
             "generated, and it drifts from what the caller actually passes (this "
             "one said 60000 while every launcher passed 50000).",
    )
    parser.add_argument(
        "--dataset-max-generation",
        dest="dataset_max_generation",
        type=int,
        required=True,
        help="REQUIRED. Max nodes VISITED (--dataset_max_generation). The BINDING "
             "ceiling -- see the module docstring. It used to sit unseen at its "
             "100000 C++ default while only max-creation was threaded through.",
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
            "--dataset-max-generation", str(args.dataset_max_generation),
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
