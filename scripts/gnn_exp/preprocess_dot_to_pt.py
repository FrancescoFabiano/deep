#!/usr/bin/env python3
"""Pre-serialize planner DOT files into PyG Data objects (.pt cache).

Run this once after Stage 1 (training-data generation).  For every training
instance folder (the directories holding the per-instance CSV +
RawFiles/...), all referenced DOT files are parsed into PyG Data objects —
the exact objects preprocess_sample() would build — and stored as a single
dict {resolved_dot_path: Data} in

    <instance_dir>/graph_cache_<DATASET_TYPE>.pt

GraphDataPipeline._load_samples() picks the cache up automatically and skips
DOT parsing entirely; without the cache it falls back to the (fast) parser,
so this step is optional.

Usage:
  .venv/bin/python scripts/gnn_exp/preprocess_dot_to_pt.py exp/gnn_exp/batch0 \
      [--dataset_type HASHED] [--workers 16]
"""
import argparse
import concurrent.futures
import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

import torch  # noqa: E402

from src.utils import KEYWORD_BITMASK, load_graph  # noqa: E402

_BITMASK = False  # set per-process via initializer


def _init_worker(bitmask: bool):
    global _BITMASK
    _BITMASK = bitmask


def _parse_chunk(paths):
    return {p: load_graph(p, bitmask=_BITMASK) for p in paths}


def find_instance_dirs(batch_root: Path):
    """Instance dirs are the ones holding a CSV under */training_data/."""
    for root, dirs, fnames in os.walk(batch_root / "_models"):
        if Path(root).parent.name == "training_data" and any(
            f.endswith(".csv") for f in fnames
        ):
            yield Path(root)


def collect_dot_paths(instance_dir: Path):
    """Union of state-graph and goal-graph paths referenced by the CSV."""
    paths = set()
    for csv_path in instance_dir.glob("*.csv"):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ("File Path", "Goal"):
                    p = (row.get(col) or "").strip()
                    if p.endswith(".dot"):
                        paths.add(str(Path(p).resolve()))
    return sorted(p for p in paths if Path(p).is_file())


def build_cache(instance_dir: Path, dataset_type: str, workers: int):
    paths = collect_dot_paths(instance_dir)
    if not paths:
        print(f"  [skip] no DOT files referenced by CSV in {instance_dir}")
        return
    t0 = time.perf_counter()
    cache = {}
    bitmask = dataset_type == KEYWORD_BITMASK
    chunk = max(64, len(paths) // (workers * 8) or 1)
    chunks = [paths[i : i + chunk] for i in range(0, len(paths), chunk)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(bitmask,)
    ) as ex:
        for part in ex.map(_parse_chunk, chunks):
            cache.update(part)
    out = instance_dir / f"graph_cache_{dataset_type}.pt"
    torch.save(cache, out)
    dt = time.perf_counter() - t0
    print(
        f"  [ok] {instance_dir.name}: {len(cache)} graphs -> {out.name} "
        f"({out.stat().st_size / 1e6:.1f} MB) in {dt:.1f} s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_root", type=Path, help="e.g. exp/gnn_exp/batch0")
    ap.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"],
                    default="HASHED")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    instance_dirs = list(find_instance_dirs(args.batch_root))
    if not instance_dirs:
        sys.exit(f"No training instances found under {args.batch_root}/_models")
    print(f"Building {args.dataset_type} graph caches for "
          f"{len(instance_dirs)} instance(s), {args.workers} workers")
    for d in instance_dirs:
        build_cache(d, args.dataset_type, args.workers)


if __name__ == "__main__":
    main()
