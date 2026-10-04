"""CPU latency of an exported model on real states, the way the planner runs it.

    python lib/deep_nn/bench_onnx.py exp/trials/hard/models/gossip/gnn_F16.onnx \
        --cache exp/trials/hard/models/gossip/gnn/cache/gos-04-all-deceived@bfs.merged.pt \
        --batch 1 16 --n 200 [--threads 1] [--profile]

Reads the state graphs from a handler cache (InstanceCache .pt), packs them through
the shared contract (`pack_fringe`, so the feeds are byte-identical to what
FringeEvalRL / GraphNN send), runs onnxruntime on the CPU with `--threads` intra-op
threads and reports the per-state latency. A `gnn_state.onnx` (per-state export,
GraphNN inputs) is detected from its input names. `--profile` adds the share of
each ONNX op type from onnxruntime's own profiler.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_nn.pack import pack_fringe  # noqa: E402

STATE_ONLY = ["node_features", "edge_index", "edge_attr", "batch", "pointed_ids"]


def load_states(cache_path: Path):
    c = torch.load(cache_path, weights_only=False)
    return c.states if hasattr(c, "states") else c["states"]


def feeds_for(states, names, batch: int, n: int, fringe: int = 1, seed: int = 0):
    """`n` feeds of `batch` states each (`batch` ignored for the per-state export)."""
    rng = np.random.default_rng(seed)
    out = []
    per_state = "batch" in names
    for _ in range(n):
        if per_state:
            g = states[int(rng.integers(len(states)))]
            f = {"node_features": g.node_ids.numpy(), "edge_index": g.edge_index.numpy(),
                 "edge_attr": g.edge_attr.numpy().reshape(-1, 1), "batch": np.zeros(g.n_nodes, dtype=np.int64),
                 "pointed_ids": g.pointed_ids.numpy()}
        else:
            # K = batch occupied slots of an F-wide export (F from the file name), padded like the planner
            ids = [int(i) for i in rng.integers(len(states), size=batch)]
            p = pack_fringe(_Cache(states), ids, max(batch, fringe))
            f = {k: v.numpy() for k, v in p.items()}
        out.append({k: v for k, v in f.items() if k in names})
    return out


class _Cache:
    def __init__(self, states):
        self.states = states


def session(path: Path, threads: int, profile: bool):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(tempfile.gettempdir()) / "ort_bench")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(path: Path, states, batch: int, n: int, threads: int, profile: bool, warmup: int = 10):
    sess = session(path, threads, profile)
    names = [i.name for i in sess.get_inputs()]
    per_state = "batch" in names
    m = re.search(r"_F(\d+)\.onnx$", path.name)
    fringe = int(m.group(1)) if m else batch
    if not per_state and batch > fringe:
        return None                                   # an F-wide export cannot take more than F states
    feeds = feeds_for(states, names, batch, n + warmup, fringe=fringe)
    for f in feeds[:warmup]:
        sess.run(None, f)
    times = []
    for f in feeds[warmup:]:
        t = time.perf_counter()
        sess.run(None, f)
        times.append(time.perf_counter() - t)
    k = 1 if per_state else batch
    ms = [1000 * t / k for t in times]
    res = {"model": path.name, "batch": 1 if per_state else batch, "n_calls": n,
           "per_state_ms_median": statistics.median(ms), "per_state_ms_p90": float(np.percentile(ms, 90)),
           "per_call_ms_median": statistics.median(times) * 1000}
    if profile:
        res["ops"] = op_shares(Path(sess.end_profiling()))
    return res


def op_shares(profile_json: Path, top: int = 6):
    """Share of the kernel time per ONNX op type (model-run entries only)."""
    by_op = defaultdict(float)
    for ev in json.loads(profile_json.read_text()):
        if ev.get("cat") == "Node" and ev.get("name", "").endswith("_kernel_time"):
            by_op[ev["args"].get("op_name", "?")] += ev.get("dur", 0)
    total = sum(by_op.values()) or 1.0
    ranked = sorted(by_op.items(), key=lambda kv: -kv[1])[:top]
    return {op: round(100 * d / total, 1) for op, d in ranked}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--cache", type=Path, required=True, help="InstanceCache .pt with the states to score")
    p.add_argument("--batch", type=int, nargs="+", default=[1])
    p.add_argument("--n", type=int, default=200, help="timed calls per setting")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args(argv)
    states = load_states(a.cache)
    print(f"[bench] {len(states)} states from {a.cache.name}; onnxruntime {ort.__version__}, {a.threads} thread(s)")
    for m in a.models:
        for b in a.batch:
            r = bench(m, states, b, a.n, a.threads, a.profile)
            if r is None:
                continue
            line = (f"[bench] {r['model']:<22} batch {r['batch']:>2}: {r['per_state_ms_median']:7.2f} ms/state "
                    f"(p90 {r['per_state_ms_p90']:7.2f}), {r['per_call_ms_median']:8.2f} ms/call")
            if "ops" in r:
                line += "  ops: " + ", ".join(f"{k} {v}%" for k, v in r["ops"].items())
            print(line)
            if "batch" in [i.name for i in session(m, 1, False).get_inputs()]:
                break        # per-state export: batch is meaningless


if __name__ == "__main__":
    main()
