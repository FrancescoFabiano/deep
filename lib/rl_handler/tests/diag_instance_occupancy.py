"""Diagnostic: rank instances by max live frontier — the fringe-size IV filter.

For every instance under a given exp root, measure the MAX live frontier with a
cheap, policy-free reference rollout: the BFS open-list maximum
(tree_env.bfs_frontier_max).  That is an UPPER BOUND on how many states any
policy can keep live, so if even BFS stays below the beam width F, no learned
policy will ever bind the beam and the F=32-vs-F=64 comparison is moot by
construction on that instance.

Use this as the selection filter for the diversity round: keep only instances
whose bfs_frontier_max exceeds the beam.

Discovers instances generically (any *_depth_*.csv under a training_data dir);
does not hardcode batch0.  Degrades gracefully ([skip], exit 0) when no data is
on disk, like the other diagnostics.

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_instance_occupancy.py [exp_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.tree_env import bfs_frontier_max, load_tree_instance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BEAMS = (32, 64)

# Fixed-depth globs (never walk into the huge leaf dirs); first that hits wins,
# matching discover_dots() in test_offline_encoder.
_CSV_GLOBS = (
    "**/_models/*/training_data/*/*_depth_*.csv",
    "**/training_data/*/*_depth_*.csv",
    "**/*_depth_*.csv",
)


def discover_csvs(root: Path) -> list[Path]:
    for pattern in _CSV_GLOBS:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits
    return []


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "exp"
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        print(f"[skip] exp root not found: {root}")
        return

    csvs = discover_csvs(root)
    if not csvs:
        print(f"[skip] no *_depth_*.csv instance tables under {root}; "
              "nothing to rank.")
        return

    rows = []
    for csv_path in csvs:
        try:
            inst = load_tree_instance(csv_path)
        except Exception as exc:  # malformed table -> note + skip, never crash
            print(f"[skip] {csv_path.name}: {exc}")
            continue
        st = inst.stats()
        fmax = bfs_frontier_max(inst)
        rows.append({
            "name": inst.name,
            "n_states": inst.n_states,
            "root_branching": st["root_branching"],
            "branching_mean": st["branching_internal_mean"],
            "bfs_frontier_max": fmax,
        })

    if not rows:
        print("[skip] no instance tables parsed successfully; nothing to rank.")
        return

    rows.sort(key=lambda r: r["bfs_frontier_max"], reverse=True)

    hdr = (f"{'instance':<26}{'n_states':>10}{'root_br':>9}{'branch_mean':>13}"
           f"{'bfs_front_max':>15}{'binds_32?':>11}{'binds_64?':>11}")
    print(hdr)
    print("-" * len(hdr))
    n_bind = {b: 0 for b in BEAMS}
    for r in rows:
        b32 = "yes" if r["bfs_frontier_max"] >= 32 else "no"
        b64 = "yes" if r["bfs_frontier_max"] >= 64 else "no"
        n_bind[32] += int(b32 == "yes")
        n_bind[64] += int(b64 == "yes")
        print(f"{r['name']:<26}{r['n_states']:>10}{r['root_branching']:>9}"
              f"{r['branching_mean']:>13.2f}{r['bfs_frontier_max']:>15}"
              f"{b32:>11}{b64:>11}")
    print("-" * len(hdr))
    print(f"{len(rows)} instances; binds_32: {n_bind[32]}  binds_64: {n_bind[64]} "
          f"(keep only binds_>=F instances for the fringe-size diversity round — "
          f"others leave the beam-width IV inert).")


if __name__ == "__main__":
    main()
