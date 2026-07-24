#!/usr/bin/env python3
"""Profile the dataloader-preparation phase of GNN training (Stage 2).

Times each stage of the DOT -> PyG Data pipeline separately:
  (a) file I/O + pydot parsing            (_load_dot, first half)
  (b) pydot -> NetworkX conversion        (_load_dot, second half)
  (c) attr normalization + from_networkx  (_nx_to_pyg, first half)
  (d) uint64_to_signed_int64 + tensor     (_nx_to_pyg, second half)
  (e) graph_collate_fn batching           (DataLoader collate)

Usage:
  .venv/bin/python scripts/gnn_exp/profile_preprocessing.py \
      [--data-dir exp/.../hash_merged] [--n 200] [--batch-size 256]
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

import networkx as nx  # noqa: E402
import pydot  # noqa: E402
import torch  # noqa: E402
from torch_geometric.utils import from_networkx  # noqa: E402

from src.utils import (  # noqa: E402
    _parse_dot_fast,
    graph_collate_fn,
    uint64_to_signed_int64,
)

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "exp/gnn_exp/batch0/_models/CC/training_data/CC_2_2_3__pl_4/RawFiles/hash_merged"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("*.dot"))[: args.n]
    if not files:
        sys.exit(f"No .dot files in {args.data_dir}")
    print(f"Profiling {len(files)} DOT files from {args.data_dir}\n")

    t_read, t_pydot, t_nx, t_fromnx, t_ids, t_collate = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_nodes_tot, n_edges_tot = 0, 0
    samples = []

    t_total0 = time.perf_counter()
    for path in files:
        # (a) file I/O + pydot parsing
        t0 = time.perf_counter()
        src = path.read_text()
        t1 = time.perf_counter()
        dot = pydot.graph_from_dot_data(src)[0]
        t2 = time.perf_counter()
        # (b) pydot -> NetworkX
        G = nx.nx_pydot.from_pydot(dot)
        t3 = time.perf_counter()
        # (c) attr normalization + from_networkx (mirrors _nx_to_pyg)
        for _, d in G.nodes(data=True):
            d["shape"] = {"circle": 0, "doublecircle": 1}.get(d.get("shape", "circle"), 0)
        for _, _, d in G.edges(data=True):
            d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))
        data = from_networkx(G)
        data.edge_index = data.edge_index.long()
        data.edge_attr = data.edge_label.view(-1, 1).float()
        t4 = time.perf_counter()
        # (d) uint64 -> signed int64 + tensor creation
        raw_ids = [int(n) for n in G.nodes()]
        signed_ids = uint64_to_signed_int64(raw_ids)
        data.node_names = torch.tensor(signed_ids, dtype=torch.int64)
        t5 = time.perf_counter()

        t_read += t1 - t0
        t_pydot += t2 - t1
        t_nx += t3 - t2
        t_fromnx += t4 - t3
        t_ids += t5 - t4
        n_nodes_tot += data.num_nodes
        n_edges_tot += data.edge_index.size(1)
        data.name = path.stem
        samples.append(
            {
                "state_graph": data,
                "depth": torch.tensor([1]),
                "target": torch.tensor([1.0]),
            }
        )

    # (e) collate into batches
    t0 = time.perf_counter()
    for i in range(0, len(samples), args.batch_size):
        graph_collate_fn(samples[i : i + args.batch_size])
    t_collate = time.perf_counter() - t0
    t_total = time.perf_counter() - t_total0

    n = len(files)
    print(f"Graph size: avg {n_nodes_tot / n:.1f} nodes, {n_edges_tot / n:.1f} edges\n")
    rows = [
        ("(a) file read", t_read),
        ("(a) pydot parse", t_pydot),
        ("(b) nx from_pydot", t_nx),
        ("(c) from_networkx + attrs", t_fromnx),
        ("(d) uint64->int64 + tensor", t_ids),
        ("(e) graph_collate_fn", t_collate),
    ]
    print(f"{'stage':<30}{'total [s]':>12}{'per-sample [ms]':>18}{'share':>9}")
    print("-" * 69)
    for name, t in rows:
        print(f"{name:<30}{t:>12.3f}{1000 * t / n:>18.3f}{100 * t / t_total:>8.1f}%")
    print("-" * 69)
    print(f"{'TOTAL':<30}{t_total:>12.3f}{1000 * t_total / n:>18.3f}{'100.0%':>9}")
    print(f"\nThroughput: {n / t_total:.1f} samples/s "
          f"(extrapolated to 60k samples: {60000 * t_total / n / 60:.1f} min)")

    # ---- fast path: _parse_dot_fast (regex parser, no pydot/NetworkX) ----
    t0 = time.perf_counter()
    fast_samples = []
    for path in files:
        d = _parse_dot_fast(path.read_text())
        d.name = path.stem
        fast_samples.append(
            {"state_graph": d, "depth": torch.tensor([1]), "target": torch.tensor([1.0])}
        )
    t_fast = time.perf_counter() - t0
    print(f"\nFAST PATH (_parse_dot_fast, incl. file read): {t_fast:.3f} s total, "
          f"{1000 * t_fast / n:.3f} ms/sample -> {n / t_fast:.0f} samples/s")
    print(f"Speed-up vs old path: {t_total / t_fast:.1f}x "
          f"(extrapolated to 60k samples: {60000 * t_fast / n:.1f} s)")

    # ---- cache path: pre-serialized Data objects (graph_cache_*.pt) ----
    cache_file = args.data_dir.parents[1] / "graph_cache_HASHED.pt"
    if not cache_file.is_file():
        print(f"\nCACHE PATH: skipped — {cache_file} not found "
              f"(build with scripts/gnn_exp/preprocess_dot_to_pt.py)")
        return
    t0 = time.perf_counter()
    cache = torch.load(cache_file, weights_only=False)
    t_load = time.perf_counter() - t0
    keys = [str(p.resolve()) for p in files]
    assert all(k in cache for k in keys), "cache is missing profiled files"
    t0 = time.perf_counter()
    cache_samples = []
    for path, key in zip(files, keys):
        d = cache[key]
        d.name = path.stem
        cache_samples.append(
            {"state_graph": d, "depth": torch.tensor([1]), "target": torch.tensor([1.0])}
        )
    t_cache = time.perf_counter() - t0
    print(f"\nCACHE PATH (lookup + sample wrap): {t_cache:.3f} s total, "
          f"{1000 * t_cache / n:.3f} ms/sample -> {n / t_cache:.0f} samples/s")
    print(f"Speed-up vs old path: {t_total / t_cache:.1f}x")
    print(f"(one-off torch.load of {len(cache)}-graph cache: {t_load:.1f} s — "
          f"amortized {1000 * t_load / len(cache):.3f} ms/graph)")

    print("\n| Path                          | ms/sample | speedup vs pydot |")
    print("|-------------------------------|-----------|------------------|")
    print(f"| Old (pydot + NetworkX)        | {1000 * t_total / n:>9.3f} | 1x               |")
    print(f"| Fast (regex _parse_dot_fast)  | {1000 * t_fast / n:>9.3f} | {t_total / t_fast:>5.0f}x           |")
    print(f"| Cache (.pt lookup)            | {1000 * t_cache / n:>9.3f} | {t_total / t_cache:>5.0f}x           |")
    print(f"| Cache incl. amortized load    | {1000 * (t_cache + t_load * n / len(cache)) / n:>9.3f} | "
          f"{t_total / (t_cache + t_load * n / len(cache)):>5.0f}x           |")


if __name__ == "__main__":
    main()
