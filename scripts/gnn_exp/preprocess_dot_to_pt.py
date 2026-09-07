#!/usr/bin/env python3
"""Pre-build the per-instance graph caches the distance-estimator trainer uses.

Optional: the trainer (lib/gnn_handler/__main__.py) builds the same caches on
first use. Run this after Stage 1 to pay the parsing once, in parallel, before
any training run. For every instance folder under
    <batch>/_models/<domain>/training_data/<instance>/
every DOT the table references (states, and goal_tree.dot in separated mode)
is parsed once with the fast planner-DOT parser into flat tensors and stored as
    <batch>/_models/<domain>/cache/<instance>.<DATASET_TYPE>.pt
(src/graph_store.GraphStore). A cache is reused only if it lists exactly the
table's DOT paths in the same mode, so it cannot go stale silently.

Usage:
  .venv/bin/python scripts/gnn_exp/preprocess_dot_to_pt.py exp/gnn_exp/batch0 \
      [--dataset_type HASHED] [--workers 16]
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

from src.graph_store import GraphStore, dot_paths_in_csv  # noqa: E402
from src.utils import KEYWORD_BITMASK  # noqa: E402


def find_instance_dirs(batch_root: Path):
    """Instance dirs = the ones holding a CSV directly under */training_data/."""
    for root, dirs, fnames in os.walk(batch_root / "_models"):
        if Path(root).parent.name == "training_data" and any(f.endswith(".csv") for f in fnames):
            yield Path(root)


def build_cache(instance_dir: Path, dataset_type: str, workers: int) -> None:
    csvs = sorted(instance_dir.glob("*.csv"))
    if len(csvs) != 1:
        print(f"  [skip] {instance_dir}: expected one table, found {len(csvs)}")
        return
    # Goal DOTs only exist in separated mode; merged tables still name them.
    paths = [p for p in dot_paths_in_csv(csvs[0]) if Path(p).is_file()]
    if not paths:
        print(f"  [skip] no DOT files referenced by {csvs[0].name}")
        return
    cache = instance_dir.parent.parent / "cache" / f"{instance_dir.name}.{dataset_type}.pt"
    t0 = time.perf_counter()
    store = GraphStore.from_paths(paths, bitmask=dataset_type == KEYWORD_BITMASK,
                                  cache_file=cache, workers=workers, verbose=False)
    print(f"  [ok] {instance_dir.name}: {len(store)} graphs -> {cache} "
          f"({cache.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_root", type=Path, help="e.g. exp/gnn_exp/batch0")
    ap.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    instance_dirs = list(find_instance_dirs(args.batch_root))
    if not instance_dirs:
        sys.exit(f"No training instances found under {args.batch_root}/_models")
    print(f"Building {args.dataset_type} graph caches for {len(instance_dirs)} instance(s), "
          f"{args.workers} workers")
    for d in instance_dirs:
        build_cache(d, args.dataset_type, args.workers)


if __name__ == "__main__":
    main()
