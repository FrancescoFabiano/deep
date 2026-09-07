"""Flat graph storage: parse every DOT once, cache it, assemble batches by index.

WHY (2026-09-07)
The previous data path turned every CSV row into a PyG `Data` object, pickled
all of them into `samples.pt`, and had a `DataLoader` re-collate them into a
`Batch` on every step. Building that was the expensive part of a run, and it
was rebuilt for every experiment. lib/rl_handler's encoder showed the cheaper
direction: parse each instance's DOT files ONCE with the regex parser, keep
them as a few flat tensors, cache those to disk, and build a batch by index
arithmetic. This module does that for the distance estimator.

WHAT THE STORE HOLDS (all graphs of one or many instances, concatenated)
    node_ids   int64 [N_tot]          (HASHED / MAPPED: the two's-complement
                                       fold of the 64-bit world hash)
    node_bits  uint8 [N_tot, B]        (BITMASK: B = 42 = C++ BITMASK_DIM)
    edge_index int64 [2, E_tot]        LOCAL node indices (0..n_g-1 per graph)
    edge_attr  int64 [E_tot]           raw integer edge labels (agent ids)
    node_ptr   int64 [G+1], edge_ptr int64 [G+1]   graph g owns
                                       nodes node_ptr[g]:node_ptr[g+1], edges
                                       edge_ptr[g]:edge_ptr[g+1]
    paths      list[str]               graph g <-> its DOT path (as written in
                                       the CSV, repo-root relative)

`pack(indices)` gathers a batch: node/edge ranges concatenated, edge indices
shifted by the cumulative node offset, a `batch` vector, and `edge_attr` as
float32 [E,1] -- exactly the attributes `DistanceEstimator._encode_graph`
reads off a PyG Batch (`node_names`/`node_bits`, `edge_index`, `edge_attr`,
`batch`), so the model is unchanged.

FIDELITY WITH THE PLANNER
The planner (GraphNN.tpp) builds, per state, int64 node ids [N], int64
edge_index [2,E], int64 edge_attr [E,1] and an all-zero int64 batch [N] (or
uint8 [N,42] bits under BITMASK). `pack` produces the same values; the ONNX
self-check in utils.verify_onnx_export feeds a store graph in that exact form.
Node ORDER within a graph is first appearance while scanning the edge lines,
which is also how the C++ assigns symbolic ids while adding edges.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# The planner's bitmask width: MAX_REPETITION_BITS(14) + MAX_FLUENT_NUMBER(18)
# + GOAL_ENCODING_BITS(10), src/utilities/Define.h. The model is built with
# bit_input=42 (utils.select_model) and the C++ feeds uint8 [N, 42].
BITMASK_DIM = 42

_INT64_MIN = -(1 << 63)
_UINT64_MOD = 1 << 64

# `u -> v [label="3"];` -- node tokens may be quoted, signed, or bare.
_DOT_EDGE_RE = re.compile(
    r'^\s*"?(-?[0-9A-Za-z_]+)"?\s*->\s*"?(-?[0-9A-Za-z_]+)"?\s*'
    r'(?:\[\s*label\s*=\s*("?-?\d+"?)\s*\])?\s*;?\s*$'
)
_DOT_HEADER_RE = re.compile(r"^\s*digraph\s+([0-9A-Za-z_]+)\s*\{\s*$")
_DOT_CLOSE_RE = re.compile(r"^\s*\}\s*$")

CACHE_FORMAT = 1


def fold_uint64_to_int64(values: Sequence[int]) -> torch.Tensor:
    """IDs in [-2^63, 2^64-1] -> int64 preserving the 64-bit pattern (the planner
    used to print hashes unsigned and now prints them signed; both fold to the
    same tensor value, which is what CreateTensor<int64_t> receives)."""
    for v in values:
        if v < _INT64_MIN or v > _UINT64_MOD - 1:
            raise ValueError(f"Node ID {v} outside [-2^63, 2^64-1].")
    masked = [v & 0xFFFFFFFFFFFFFFFF for v in values]
    raw = np.fromiter(masked, dtype=np.uint64, count=len(masked))
    return torch.from_numpy(raw.view(np.int64).copy())


@dataclass
class FlatGraph:
    """One parsed DOT with local node indexing."""

    node_ids: Optional[torch.Tensor]    # int64 [n]      (ids mode)
    node_bits: Optional[torch.Tensor]   # uint8 [n, B]   (bitmask mode)
    edge_index: torch.Tensor            # int64 [2, e]
    edge_attr: torch.Tensor             # int64 [e]

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.numel() if self.node_ids is not None else self.node_bits.shape[0])


def parse_dot_text(src: str, bitmask: bool = False) -> Optional[FlatGraph]:
    """Planner DOT text -> FlatGraph, or None if the text is outside the
    planner's restricted grammar (then the caller falls back to pydot).

    Same node order and edge order as utils._parse_dot_fast / networkx:
    nodes by first appearance over edge lines, edges grouped by source node in
    node order, then by target in first-appearance order, then by label.
    """
    graph_name = None
    node_idx: Dict[str, int] = {}
    adj: Dict[int, Dict[int, List[str]]] = {}
    for line in src.splitlines():
        m = _DOT_EDGE_RE.match(line)
        if m is not None:
            u, v, raw_label = m.group(1), m.group(2), m.group(3)
            if raw_label is None:
                return None
            ui = node_idx.setdefault(u, len(node_idx))
            vi = node_idx.setdefault(v, len(node_idx))
            adj.setdefault(ui, {}).setdefault(vi, []).append(raw_label)
            continue
        if not line.strip():
            continue
        if graph_name is None:
            hm = _DOT_HEADER_RE.match(line)
            if hm is not None:
                graph_name = hm.group(1)
                continue
            return None
        if _DOT_CLOSE_RE.match(line):
            continue
        return None
    if graph_name is None:
        return None

    srcs: List[int] = []
    dsts: List[int] = []
    labels: List[int] = []
    for ui in range(len(node_idx)):
        for vi, labs in adj.get(ui, {}).items():
            for lab in labs:
                srcs.append(ui)
                dsts.append(vi)
                labels.append(int(lab.replace('"', "")))
    edge_index = torch.tensor([srcs, dsts], dtype=torch.int64).reshape(2, -1)
    edge_attr = torch.tensor(labels, dtype=torch.int64)

    nodes = list(node_idx)
    if bitmask:
        if not nodes:
            return None
        bit_len = len(nodes[0])
        if any(len(n) != bit_len for n in nodes):
            raise ValueError("Inconsistent bit length across nodes.")
        joined = "".join(nodes)
        bits = np.frombuffer(joined.encode("ascii"), dtype=np.uint8) - ord("0")
        if not ((bits == 0) | (bits == 1)).all():
            raise ValueError("Node labels are not 0/1 bitstrings.")
        node_bits = torch.from_numpy(bits.reshape(len(nodes), bit_len).astype(np.uint8))
        return FlatGraph(None, node_bits, edge_index, edge_attr)
    return FlatGraph(fold_uint64_to_int64([int(n) for n in nodes]), None, edge_index, edge_attr)


def parse_dot_file(path: str | Path, bitmask: bool = False) -> FlatGraph:
    """Fast path, with the general pydot path as fallback (rare: DOTs with node
    statements or attributes the planner never writes)."""
    text = Path(path).read_text()
    g = parse_dot_text(text, bitmask=bitmask)
    if g is not None:
        return g
    from src.utils import _load_dot, _nx_to_pyg  # general path, heavy imports
    data = _nx_to_pyg(_load_dot(Path(path)), bitmask=bitmask)
    edge_attr = data.edge_attr.view(-1).round().to(torch.int64)
    if bitmask:
        return FlatGraph(None, data.node_bits.to(torch.uint8), data.edge_index.to(torch.int64), edge_attr)
    return FlatGraph(data.node_names.to(torch.int64), None, data.edge_index.to(torch.int64), edge_attr)


# --------------------------------------------------------------------------
# multiprocess parsing
# --------------------------------------------------------------------------

_W_BITMASK = False


def _init_worker(bitmask: bool) -> None:
    global _W_BITMASK
    _W_BITMASK = bitmask


def _parse_chunk(paths: List[str]) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]]:
    out = []
    for p in paths:
        g = parse_dot_file(p, bitmask=_W_BITMASK)
        out.append((
            None if g.node_ids is None else g.node_ids.numpy(),
            None if g.node_bits is None else g.node_bits.numpy(),
            g.edge_index.numpy(),
            g.edge_attr.numpy(),
        ))
    return out


class GraphBatch:
    """What the model reads: the PyG-Batch attributes DistanceEstimator uses."""

    __slots__ = ("node_names", "node_bits", "edge_index", "edge_attr", "batch", "num_graphs")

    def __init__(self, node_names, node_bits, edge_index, edge_attr, batch, num_graphs):
        self.node_names = node_names        # int64 [N] or None
        self.node_bits = node_bits          # uint8 [N, B] or None
        self.edge_index = edge_index        # int64 [2, E]
        self.edge_attr = edge_attr          # float32 [E, 1]
        self.batch = batch                  # int64 [N]
        self.num_graphs = num_graphs

    def to(self, device) -> "GraphBatch":
        f = lambda t: None if t is None else t.to(device)  # noqa: E731
        return GraphBatch(f(self.node_names), f(self.node_bits), f(self.edge_index),
                          f(self.edge_attr), f(self.batch), self.num_graphs)


class GraphStore:
    """All graphs of a set of DOT files as flat tensors; batches by index."""

    def __init__(self, paths: List[str], node_ids: Optional[torch.Tensor],
                 node_bits: Optional[torch.Tensor], edge_index: torch.Tensor,
                 edge_attr: torch.Tensor, node_ptr: torch.Tensor, edge_ptr: torch.Tensor):
        self.paths = list(paths)
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.node_ids = node_ids
        self.node_bits = node_bits
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_ptr = node_ptr
        self.edge_ptr = edge_ptr
        self.device = edge_index.device

    # ---- construction -----------------------------------------------------

    @property
    def bitmask(self) -> bool:
        return self.node_bits is not None

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def from_graphs(paths: List[str], graphs: Iterable[FlatGraph]) -> "GraphStore":
        ids, bits, eis, eas = [], [], [], []
        node_ptr = [0]
        edge_ptr = [0]
        for g in graphs:
            if g.node_ids is not None:
                ids.append(g.node_ids)
            else:
                bits.append(g.node_bits)
            eis.append(g.edge_index)
            eas.append(g.edge_attr)
            node_ptr.append(node_ptr[-1] + g.n_nodes)
            edge_ptr.append(edge_ptr[-1] + int(g.edge_attr.numel()))
        if len(node_ptr) - 1 != len(paths):
            raise ValueError("paths/graphs length mismatch")
        bitmask = bool(bits) and not ids
        return GraphStore(
            paths,
            None if bitmask else (torch.cat(ids) if ids else torch.zeros(0, dtype=torch.int64)),
            (torch.cat(bits) if bits else torch.zeros((0, BITMASK_DIM), dtype=torch.uint8)) if bitmask else None,
            torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.int64),
            torch.cat(eas) if eas else torch.zeros(0, dtype=torch.int64),
            torch.tensor(node_ptr, dtype=torch.int64),
            torch.tensor(edge_ptr, dtype=torch.int64),
        )

    @staticmethod
    def parse(paths: Sequence[str], bitmask: bool = False, workers: Optional[int] = None,
              verbose: bool = True) -> "GraphStore":
        paths = list(paths)
        t0 = time.perf_counter()
        workers = workers or max(1, min(os.cpu_count() or 1, 16))
        if len(paths) < 512 or workers == 1:
            _init_worker(bitmask)
            parsed = _parse_chunk(paths)
        else:
            chunk = max(64, len(paths) // (workers * 8))
            chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
            parsed = []
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_worker, initargs=(bitmask,)) as ex:
                for part in ex.map(_parse_chunk, chunks):
                    parsed.extend(part)
        graphs = [FlatGraph(None if a is None else torch.from_numpy(a),
                            None if b is None else torch.from_numpy(b),
                            torch.from_numpy(c), torch.from_numpy(d)) for a, b, c, d in parsed]
        store = GraphStore.from_graphs(paths, graphs)
        if verbose:
            dt = time.perf_counter() - t0
            print(f"[graph_store] parsed {len(paths)} DOT files in {dt:.1f}s "
                  f"({store.node_ptr[-1].item()} nodes, {store.edge_ptr[-1].item()} edges)")
        return store

    @staticmethod
    def from_paths(paths: Sequence[str], bitmask: bool = False,
                   cache_file: Optional[str | Path] = None, workers: Optional[int] = None,
                   verbose: bool = True) -> "GraphStore":
        """Load from `cache_file` if it holds exactly these paths in this mode,
        else parse and (re)write the cache. Paths are compared as given (the
        CSV's repo-root-relative strings), so the cache moves with the batch."""
        paths = list(paths)
        if cache_file is not None:
            cache_file = Path(cache_file)
            if cache_file.is_file():
                try:
                    payload = torch.load(cache_file, weights_only=False)
                except Exception as e:  # corrupt cache: rebuild, never fail the run
                    if verbose:
                        print(f"[graph_store] cache {cache_file} unreadable ({e}); rebuilding")
                    payload = None
                if (payload is not None and payload.get("format") == CACHE_FORMAT
                        and payload.get("bitmask") == bitmask and payload.get("paths") == paths):
                    store = GraphStore(payload["paths"], payload.get("node_ids"), payload.get("node_bits"),
                                       payload["edge_index"], payload["edge_attr"],
                                       payload["node_ptr"], payload["edge_ptr"])
                    if verbose:
                        print(f"[graph_store] loaded {len(store)} graphs from {cache_file}")
                    return store
                if verbose and payload is not None:
                    print(f"[graph_store] cache {cache_file} does not match the requested "
                          f"paths/mode; rebuilding")
        store = GraphStore.parse(paths, bitmask=bitmask, workers=workers, verbose=verbose)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": CACHE_FORMAT, "bitmask": bitmask, "paths": store.paths,
                        "node_ids": store.node_ids, "node_bits": store.node_bits,
                        "edge_index": store.edge_index, "edge_attr": store.edge_attr,
                        "node_ptr": store.node_ptr, "edge_ptr": store.edge_ptr}, cache_file)
            if verbose:
                print(f"[graph_store] cached -> {cache_file}")
        return store

    @staticmethod
    def concat(stores: Sequence["GraphStore"]) -> "GraphStore":
        """One index space over several per-instance stores (duplicates of a
        path, e.g. a shared goal DOT, keep their first occurrence)."""
        stores = [s for s in stores if len(s)]
        if not stores:
            raise ValueError("no graphs")
        bitmask = stores[0].bitmask
        if any(s.bitmask != bitmask for s in stores):
            raise ValueError("cannot concat ids and bitmask stores")
        graphs: List[FlatGraph] = []
        paths: List[str] = []
        seen = set()
        for s in stores:
            for g in range(len(s)):
                if s.paths[g] in seen:
                    continue
                seen.add(s.paths[g])
                paths.append(s.paths[g])
                graphs.append(s.graph(g))
        return GraphStore.from_graphs(paths, graphs)

    # ---- access -------------------------------------------------------------

    def graph(self, g: int) -> FlatGraph:
        a, b = int(self.node_ptr[g]), int(self.node_ptr[g + 1])
        c, d = int(self.edge_ptr[g]), int(self.edge_ptr[g + 1])
        return FlatGraph(
            None if self.node_ids is None else self.node_ids[a:b],
            None if self.node_bits is None else self.node_bits[a:b],
            self.edge_index[:, c:d], self.edge_attr[c:d],
        )

    def to(self, device) -> "GraphStore":
        self.device = torch.device(device)
        for name in ("node_ids", "node_bits", "edge_index", "edge_attr", "node_ptr", "edge_ptr"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self

    def pack(self, indices: Sequence[int] | torch.Tensor) -> GraphBatch:
        """Gather graphs `indices` into one batch (index arithmetic only)."""
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device).view(-1)
        n_starts = self.node_ptr[idx]
        n_counts = self.node_ptr[idx + 1] - n_starts
        e_starts = self.edge_ptr[idx]
        e_counts = self.edge_ptr[idx + 1] - e_starts
        node_sel = _ranges(n_starts, n_counts)
        edge_sel = _ranges(e_starts, e_counts)
        # graph id per node / per edge
        batch = torch.repeat_interleave(torch.arange(idx.numel(), device=self.device), n_counts)
        edge_graph = torch.repeat_interleave(torch.arange(idx.numel(), device=self.device), e_counts)
        # local -> batch-global node index: add the cumulative node offset of the edge's graph
        offsets = torch.cumsum(n_counts, 0) - n_counts
        edge_index = self.edge_index[:, edge_sel] + offsets[edge_graph].unsqueeze(0)
        return GraphBatch(
            None if self.node_ids is None else self.node_ids[node_sel],
            None if self.node_bits is None else self.node_bits[node_sel],
            edge_index,
            self.edge_attr[edge_sel].to(torch.float32).view(-1, 1),
            batch,
            int(idx.numel()),
        )

    def planner_feed(self, g: int, prefix: str = "state") -> Dict[str, np.ndarray]:
        """The exact tensors the C++ planner feeds for ONE graph, as numpy:
        int64 ids [N] (or uint8 bits [N,42]), int64 edge_index [2,E], int64
        edge_attr [E,1], int64 all-zero batch [N]. Used by the ONNX self-check."""
        fg = self.graph(g)
        feed = {}
        if fg.node_ids is not None:
            feed[f"{prefix}_node_ids"] = fg.node_ids.cpu().numpy().astype(np.int64)
        else:
            feed[f"{prefix}_node_bits"] = fg.node_bits.cpu().numpy().astype(np.uint8)
        feed[f"{prefix}_edge_index"] = fg.edge_index.cpu().numpy().astype(np.int64)
        feed[f"{prefix}_edge_attr"] = fg.edge_attr.cpu().numpy().astype(np.int64).reshape(-1, 1)
        feed[f"{prefix}_batch"] = np.zeros(fg.n_nodes, dtype=np.int64)
        return feed


def _ranges(starts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Concatenate [s, s+c) ranges without a Python loop."""
    total = int(counts.sum())
    if total == 0:
        return torch.zeros(0, dtype=torch.int64, device=starts.device)
    rep_starts = torch.repeat_interleave(starts, counts)
    within = torch.arange(total, device=starts.device) - torch.repeat_interleave(
        torch.cumsum(counts, 0) - counts, counts)
    return rep_starts + within


def dot_paths_in_csv(csv_path: str | Path, columns: Sequence[str] = ("File Path", "Goal")) -> List[str]:
    """All DOT paths a generation table references, in row order, deduplicated,
    as written (repo-root relative). `init.dot` (the root's dummy predecessor)
    is never written to disk and is skipped."""
    import csv as _csv
    out: List[str] = []
    seen = set()
    with Path(csv_path).open(newline="") as fh:
        for row in _csv.DictReader(fh):
            for c in columns:
                v = (row.get(c) or "").strip()
                if v.endswith(".dot") and Path(v).name != "init.dot" and v not in seen:
                    seen.add(v)
                    out.append(v)
    return out
