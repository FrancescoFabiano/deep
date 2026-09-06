"""Generate training tables for every domain in a batch. THE one place generation
parameters live -- depth, discard, max_creation, max_generation, the seed, the deep
flags. `final_launcher.sh` drives it and passes them explicitly. Nothing in
lib/rl_handler re-specifies how to generate; rl_handler only CONSUMES trees and CHECKS
them (the USABILITY gate: delta_root finite, non-trivial, scorable -- see
`src/offline/usability.py`). Scripts generate, the gate validates, and they share no
parameter.

THE GENERATOR CANNOT PRODUCE SHORTEST-PATH DISTANCES  (settled 2026-07-15)

Do not tune depth or budget hoping to make delta_root match the instance's optimal.
It cannot be done, and the reason is structural, not a knob.

`TrainingDataset.tpp` runs a depth-bounded DFS whose memo is keyed by STATE ALONE:

    if (m_visited_states.contains(state)) { return m_states_scores[state]; }

DFS dives first, so a state is typically discovered DEEP, where no depth budget
remains -- it is recorded as a childless leaf scored 1e6. When the search later
reaches that same state SHALLOW, with room to expand, the memo returns the stale deep
verdict and its subtree is never explored. Each state is written once, at its
DISCOVERY depth, and (because the file name is minted per VISIT, not per state) the
recorded structure collapses to a DFS spanning tree. So delta_root >= the true
optimal, equal only where the DFS happened to sample a shortest path first.

Measured on CC_2_2_3__pl_4 (true optimal 4; depth 40, discard 0, identical flags --
seed alone varied): delta_root = 14 / 6 / 7 for seeds 42 / 43 / 44. The value tracks
the RNG, not the instance.

BUDGET IS NOT THE LEVER -- the earlier "the 100k VISIT ceiling truncates the DFS"
diagnosis was WRONG. On CC_2_3_4__pl_7 at depth 25, raising --dataset_max_generation
100000 -> 250000 bought 4.2x the goals and 2.2x the states and moved delta_root by
ZERO (22 both times) -- while poisoned_frac went DOWN (0.0056 -> 0.0042). A starved
tree poisons MORE near its ceiling; this poisons less. There is no starvation
signature. Nor is depth the lever: depth 9 -> delta_root 9, depth 8 and 7 -> no goal
found AT ALL (23 states, vs the 7592 BFS needs to reach the depth-7 goal).

WHAT THE PARAMETERS ARE ACTUALLY FOR
  --dataset_discard_factor 0  -- MUST be 0. At 0.4 the biased discard deletes shallow
      goals outright; that is a different search problem, not noise.
  --dataset_max_generation    -- VISIT ceiling (m_current_nodes, default 100000).
      Hard-capped at ~290k in practice: the DOT file counter is incremented per VISIT
      and its name is built with `std::string(6 - digits, '0')`, which underflows an
      unsigned size_t at file #1,000,000 and aborts with std::length_error. Dots per
      visit is depth-dependent (~3.3x at depth 25, ~1580x at depth 9), so watch the
      file count rather than trusting a ratio.
  --dataset_max_creation      -- WRITE cap (m_added_to_dataset).
  --dataset_seed              -- LOAD-BEARING: the tree is a seed-dependent sample, so
      the seed is part of the data's identity and is fingerprinted in DATASPEC.

Both ceilings poison identically (`TrainingDataset.tpp`, top of dfs_worker): past
either, a non-goal returns m_failed_state (1e6) WITHOUT recursing. Note h*=1e6 does
NOT imply a ceiling fired -- an ordinary non-goal leaf at the depth bound scores 1e6
too, which is why the gate treats poisoned_frac as a diagnostic, not an exclusion.
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
    that artifact for that domain, and the usability gate might not catch it if the
    instance still reaches a goal. A loud failure beats a convenient default that can
    silently be wrong.
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


GENERATION_CHOICES = ("BFS", "DFS", "S_DFS", "HFS")


def normalize_generation(value):
    """Map user spellings (s-dfs, S-DFS, bfs, ...) onto the C++ enum names."""
    v = value.strip().upper().replace("-", "_")
    if v not in GENERATION_CHOICES:
        raise argparse.ArgumentTypeError(
            f"invalid --dataset-generation {value!r}; expected one of {GENERATION_CHOICES}")
    return v


# Discard factor defaults, PER STRATEGY. Only the stochastic DFS worker reads
# --dataset_discard_factor (DFS and S_DFS differ in nothing else; BFS/HFS never
# discard), so the knob is an S_DFS parameter:
#   S_DFS                -> 0.6  (project default, set 2026-09-06)
#   BFS / DFS / HFS      -> 0    (ignored by the C++; passed as 0 so the log is honest)
#   flag omitted (legacy) -> 0.4 (the gnn_exp pipeline's historical default, untouched)
# An explicit --discard_factor overrides all three.
S_DFS_DEFAULT_DISCARD = 0.6
LEGACY_DEFAULT_DISCARD = 0.4


def _discard_for(generation, args):
    if args.discard_factor is not None:
        return args.discard_factor
    if generation is None:
        return LEGACY_DEFAULT_DISCARD
    return S_DFS_DEFAULT_DISCARD if generation == "S_DFS" else 0.0


def parse_generation_list(values):
    """`--dataset-generation BFS HFS`, `BFS,HFS`, `all` -> C++ enum names in canonical
    order, deduplicated. Empty/None -> [] (flag not passed; C++ default, flat layout).

    The user picks WHICH behaviour policies exist by picking what to generate here:
    one tree per strategy per instance, each under <dataset>/<STRAT>/<instance>/.
    """
    flat = []
    for v in values or []:
        flat.extend(x for x in str(v).replace(",", " ").split() if x)
    if not flat:
        return []
    if any(x.lower() == "all" for x in flat):
        return list(GENERATION_CHOICES)
    got = {normalize_generation(x) for x in flat}
    return [g for g in GENERATION_CHOICES if g in got]


def main():
    parser = argparse.ArgumentParser(
        description="Run the per-domain training-data generator on all domain folders inside a batch folder."
    )
    parser.add_argument("batch_path", help="Path to the batch folder (e.g., exp/gnn_exp/batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep", help="Path to the deep C++ executable")
    parser.add_argument("--no_goal", action="store_true", help="Add --dataset_separated to the C++ execution")
    parser.add_argument("--strong_equality", action="store_true", help="Add --strong_equality to the C++ execution")
    parser.add_argument("--depth", "--dataset-depth", dest="depth", type=int, default=25,
                        help="Depth for dataset generation (default: 25). --dataset-depth is an alias.")
    parser.add_argument("--dataset-generation", dest="dataset_generation", default=None, nargs="+",
                        help="Search strategies used by the C++ tool to build the trees "
                             "(--dataset_generation): any of BFS, DFS, S_DFS (stochastic DFS; "
                             "'S-DFS' accepted), HFS (heuristic first, see --heuristics), or "
                             "'all'. Several may be given (space- or comma-separated): the "
                             "generator runs ONCE PER STRATEGY per instance and writes each to "
                             "<dataset>/<STRAT>/<instance>/, so the behaviour policies coexist "
                             "and rl_handler trains on whichever were generated (restrict "
                             "further with train_models.py --strategies). Omitted: the flag "
                             "is not passed, the C++ default applies (S_DFS at the time of "
                             "writing) and the legacy flat layout is kept.")
    parser.add_argument("--heuristics", dest="heuristics", default=None, help='Heuristic forwarded as --heuristics; only meaningful with --dataset-generation HFS (C++ default SUBGOALS).')
    parser.add_argument(
        "--discard_factor", type=float, default=None,
        help="Maximum discard factor. Read ONLY by the stochastic DFS worker, so it is "
             "an S_DFS parameter. Default when omitted: 0.6 for S_DFS, 0 for BFS/DFS/HFS "
             "(they never discard), and 0.4 when --dataset-generation is omitted too "
             "(legacy flat layout; this script is SHARED with gnn_exp and that "
             "pipeline's default must not move under its feet). An explicit value "
             "applies to every strategy. "
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
    try:
        generations = parse_generation_list(args.dataset_generation)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    if args.heuristics and "HFS" not in generations:
        print(f"[WARNING] --heuristics {args.heuristics} is only used by HFS, which is not "
              f"among the requested strategies {generations or '(C++ default)'}.")
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


    print(f"[INFO] Dataset generation strategies: "
          f"{' '.join(generations) if generations else '(not passed -> C++ default, flat layout)'}")
    print("[INFO] Discard factor per strategy: "
          + ", ".join(f"{g or 'legacy'}={_discard_for(g, args)}" for g in (generations or [None])))
    print(f"[INFO] Base RNG seed: {args.seed}")
    print(f"[INFO] Max retries per instance: {args.max_retries}")
    print(
        f"[INFO] Failed attempts and global attempt logs will be stored in: {batch_path}/_models/<domain_name>/_failed"
    )
    print(
        f"[INFO] On success, per-dataset logs and seed summaries will be placed inside each dataset folder."
    )
    print(
        f"[INFO] Successful seeds will be appended to: {batch_path}/_models/<domain_name>/"
        f"{args.dataset_name}/{'<STRAT>/' if generations else ''}seeds.txt"
    )

    # None = "flag not passed" (one legacy pass); otherwise one pass per strategy.
    passes = generations or [None]
    for domain_rel_path in domains:
      domain_name = domain_rel_path  # relative path like 'foo/bar'
      for generation in passes:
        print(f"\n=== Processing: {domain_name}"
              f"{f'  [strategy {generation}]' if generation else ''} ===")

        # Call the adapted per-domain script; it handles seed generation, retries, and logging.
        cmd = [
            "python3",
            args.script_path,
            batch_path,             # base_folder
            domain_name,            # domain_name
            args.deep_exe,          # deep_exe
            "--depth", str(_depth_for(domain_name, args)),
            "--discard_factor", str(_discard_for(generation, args)),
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
        if generation:
            cmd += ["--dataset-generation", generation]
        if args.heuristics and generation == "HFS":
            cmd += ["--heuristics", args.heuristics]

        try:
            # Let the per-domain script print its own detailed progress & logs
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Per-domain script failed for {domain_name}"
                  f"{f' [{generation}]' if generation else ''} (exit {e.returncode}).")

    print("\n=== Batch complete ===")

if __name__ == "__main__":
    main()
