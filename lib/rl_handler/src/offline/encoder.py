"""Fringe encoder mirroring the C++ packing in FringeEvalRL::fringe_to_tensor_minimal.

CONTRACT (see lib/rl_handler/DESIGN.md §1), HASHED + merged mode:
  node_features int64 [N]   concat of per-state signed node IDs, fringe order
  edge_index    int64 [2,E] per-state edges shifted by cumulative node offset
  edge_attr     int64 [E]   raw integer edge labels
  membership    int64 [N]   fringe-slot index repeated per node
  mask          uint8 [F]   1 for active slots 0..K-1, 0 padding

The fast DOT parser below is adapted from lib/gnn_handler/src/utils.py
(_parse_dot_fast / uint64_ids_to_int64_tensor).  A direct import is not
possible: both gnn_handler and rl_handler ship a top-level package named
`src`, so importing gnn_handler.src.utils from rl_handler's context collides
(and would also pull sklearn/model deps).  Logic and ID handling are kept
identical: node order = first appearance scanning edges, uint64 IDs folded to
int64 via two's complement.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_INT64_MIN = -(1 << 63)
_UINT64_MOD = 1 << 64

# `u -> v [label="3"];` — node tokens may be quoted, signed, or bare.
_DOT_EDGE_RE = re.compile(
    r'^\s*"?(-?[0-9A-Za-z_]+)"?\s*->\s*"?(-?[0-9A-Za-z_]+)"?\s*'
    r'(?:\[\s*label\s*=\s*("?-?\d+"?)\s*\])?\s*;?\s*$'
)
_DOT_HEADER_RE = re.compile(r"^\s*digraph\s+([0-9A-Za-z_]+)\s*\{\s*$")
_DOT_CLOSE_RE = re.compile(r"^\s*\}\s*$")


def uint64_ids_to_int64(values: Sequence[int]) -> torch.Tensor:
    """Fold IDs in [-2^63, 2^64-1] to int64 preserving the 64-bit pattern."""
    for v in values:
        if v < _INT64_MIN or v > _UINT64_MOD - 1:
            raise ValueError(f"Node ID {v} outside [-2^63, 2^64-1].")
    masked = [v & 0xFFFFFFFFFFFFFFFF for v in values]
    raw = np.fromiter(masked, dtype=np.uint64, count=len(masked))
    return torch.from_numpy(raw.view(np.int64).copy())


@dataclass
class StateGraph:
    """One planner state as contract-ready tensors (local node indexing)."""

    node_ids: torch.Tensor  # int64 [n]
    edge_index: torch.Tensor  # int64 [2, e]
    edge_attr: torch.Tensor  # int64 [e]


def parse_dot_fast(src: str) -> Optional[StateGraph]:
    """Parse a planner DOT string. Returns None on non-conforming grammar."""
    graph_name = None
    node_idx: Dict[str, int] = {}
    # Adjacency in NetworkX MultiDiGraph order (edges grouped by source node
    # in insertion order, then by target in first-edge order) so the flattened
    # edge list is bit-identical to the production pydot/networkx loader.
    adj: Dict[int, Dict[int, List[int]]] = {}

    for line in src.splitlines():
        m = _DOT_EDGE_RE.match(line)
        if m is not None:
            u, v, raw_label = m.group(1), m.group(2), m.group(3)
            if raw_label is None:
                return None
            ui = node_idx.setdefault(u, len(node_idx))
            vi = node_idx.setdefault(v, len(node_idx))
            adj.setdefault(ui, {}).setdefault(vi, []).append(
                int(raw_label.replace('"', ""))
            )
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
                labels.append(lab)

    return StateGraph(
        node_ids=uint64_ids_to_int64([int(n) for n in node_idx]),
        edge_index=torch.tensor([srcs, dsts], dtype=torch.int64),
        edge_attr=torch.tensor(labels, dtype=torch.int64),
    )


def load_state_graph(path: str | Path) -> StateGraph:
    graph = parse_dot_fast(Path(path).read_text())
    if graph is None:
        raise ValueError(f"DOT file does not match planner grammar: {path}")
    return graph


def load_goal_graph(path: str | Path) -> StateGraph:
    """Parse a per-instance ``goal_tree.dot`` (separated mode).

    The goal DOT is a single ``digraph G { ... }`` whose edges carry integer
    ``label=``s — the exact grammar ``parse_dot_fast`` accepts — so the goal is
    encoded with the SAME node-id folding and edge-attr handling as state
    graphs (the model feeds both through one shared GINE encoder).
    """
    graph = parse_dot_fast(Path(path).read_text())
    if graph is None:
        raise ValueError(f"goal DOT does not match planner grammar: {path}")
    if int(graph.node_ids.numel()) == 0:
        # _pool_goal aligns one pooled goal row per fringe via goal_batch; a
        # 0-node goal would contribute no row and shift every later fringe's
        # goal_emb out of step with candidate_batch.  Refuse it early.
        raise ValueError(f"goal DOT has 0 nodes (empty goal graph): {path}")
    return graph


class InstanceCache:
    """All states of one instance parsed once; fringe assembly is index math.

    State graphs are stored densely by integer state id (assigned by the
    tree-environment loader).  A .pt cache file skips re-parsing on reuse.
    """

    CACHE_VERSION = 1

    def __init__(self, states: List[StateGraph]):
        self.states = states
        self.n_nodes = torch.tensor(
            [int(s.node_ids.numel()) for s in states], dtype=torch.long
        )

    @classmethod
    def from_paths(
        cls,
        dot_paths: Sequence[str],
        cache_file: Optional[Path] = None,
        verbose: bool = True,
    ) -> "InstanceCache":
        if cache_file is not None and cache_file.exists():
            payload = torch.load(cache_file, weights_only=False)
            if (
                payload.get("version") == cls.CACHE_VERSION
                and payload.get("paths") == list(map(str, dot_paths))
            ):
                return cls(payload["states"])
        t0 = time.time()
        states = [load_state_graph(p) for p in dot_paths]
        if verbose:
            dt = time.time() - t0
            print(
                f"[encoder] parsed {len(states)} DOT files in {dt:.1f}s "
                f"({len(states) / max(dt, 1e-9):.0f} files/s)"
            )
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": cls.CACHE_VERSION,
                    "paths": list(map(str, dot_paths)),
                    "states": states,
                },
                cache_file,
            )
        return cls(states)


def pack_fringe(
    cache: InstanceCache,
    state_indices: Sequence[int],
    fringe_size: int,
) -> Dict[str, torch.Tensor]:
    """Pack one fringe exactly like FringeEvalRL::fringe_to_tensor_minimal.

    Returns the 5 contract tensors (merged mode).  Slot order = the order of
    `state_indices`; `mask[k]=1` for k < len(state_indices).
    """
    k = len(state_indices)
    if k == 0:
        raise ValueError("Cannot pack an empty fringe.")
    if k > int(fringe_size):
        raise ValueError(f"Fringe has {k} states > fringe_size {fringe_size}.")

    node_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    membership_parts: List[torch.Tensor] = []
    node_offset = 0
    for slot, si in enumerate(state_indices):
        g = cache.states[int(si)]
        n = int(g.node_ids.numel())
        node_parts.append(g.node_ids)
        membership_parts.append(torch.full((n,), slot, dtype=torch.int64))
        if g.edge_index.numel() > 0:
            edge_index_parts.append(g.edge_index + node_offset)
            edge_attr_parts.append(g.edge_attr)
        node_offset += n

    mask = torch.zeros(int(fringe_size), dtype=torch.uint8)
    mask[:k] = 1
    return {
        "node_features": torch.cat(node_parts, dim=0),
        "edge_index": (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=torch.int64)
        ),
        "edge_attr": (
            torch.cat(edge_attr_parts, dim=0)
            if edge_attr_parts
            else torch.zeros((0,), dtype=torch.int64)
        ),
        "membership": torch.cat(membership_parts, dim=0),
        "mask": mask,
    }


class GlobalFlatCache:
    """All instances' state graphs in flat device buffers for O(1)-call packing.

    Produces tensors identical to pack_fringe_batch (verified by test), but
    assembly is a handful of vectorized gathers instead of thousands of small
    torch.cat calls — and lives on the training device, so per-step host →
    device traffic is just the state-id list.
    """

    def __init__(
        self,
        caches: Sequence["InstanceCache"],
        device: str | torch.device = "cpu",
        goal_graphs: Optional[Sequence[Optional[StateGraph]]] = None,
    ):
        self.device = torch.device(device)
        # Resident graph buffers stay on the HOST regardless of the compute
        # device: pinning every instance's full state graph on the training GPU
        # OOMs once large instances enter the split (~30k-state instances ->
        # several GiB resident). pack() gathers the per-batch slice on the host
        # and moves only that small slice to self.device, so GPU residency is
        # ~model + one batch. Output tensors are byte-identical (device aside).
        self.store = torch.device("cpu")
        node_vals: List[torch.Tensor] = []
        e_src: List[torch.Tensor] = []
        e_dst: List[torch.Tensor] = []
        e_attr: List[torch.Tensor] = []
        n_len: List[int] = []
        e_len: List[int] = []
        self.inst_offset: List[int] = []
        total_states = 0
        for cache in caches:
            self.inst_offset.append(total_states)
            for g in cache.states:
                node_vals.append(g.node_ids)
                e_src.append(g.edge_index[0])
                e_dst.append(g.edge_index[1])
                e_attr.append(g.edge_attr)
                n_len.append(int(g.node_ids.numel()))
                e_len.append(int(g.edge_attr.numel()))
            total_states += len(cache.states)

        d = self.store
        self.node_vals = torch.cat(node_vals).to(d)
        self.edge_src = torch.cat(e_src).to(d)  # local node indices
        self.edge_dst = torch.cat(e_dst).to(d)
        self.edge_attr = torch.cat(e_attr).to(d)
        self.node_len = torch.tensor(n_len, dtype=torch.long, device=d)
        self.edge_len = torch.tensor(e_len, dtype=torch.long, device=d)
        self.node_start = torch.cumsum(self.node_len, 0) - self.node_len
        self.edge_start = torch.cumsum(self.edge_len, 0) - self.edge_len

        # Separated mode: one goal graph per INSTANCE (not per state), folded the
        # same way and indexed by instance id.  pack() duplicates the right
        # instance's goal per fringe so goal_emb[candidate_batch] aligns.
        self.has_goal = goal_graphs is not None and any(
            g is not None for g in goal_graphs
        )
        if self.has_goal:
            if len(goal_graphs) != len(caches):
                raise ValueError(
                    f"goal_graphs ({len(goal_graphs)}) must align with caches "
                    f"({len(caches)})."
                )
            if any(g is None for g in goal_graphs):
                raise ValueError(
                    "separated mode requires a goal graph for every instance; "
                    "got a None among them."
                )
            if any(int(g.node_ids.numel()) == 0 for g in goal_graphs):
                # Every fringe must contribute >=1 goal node so goal_emb rows
                # stay aligned with candidate_batch (see _pool_goal).
                raise ValueError(
                    "separated mode requires every goal graph to have >=1 node; "
                    "got an empty goal graph."
                )
            gnode_vals: List[torch.Tensor] = []
            ge_src: List[torch.Tensor] = []
            ge_dst: List[torch.Tensor] = []
            ge_attr: List[torch.Tensor] = []
            gn_len: List[int] = []
            ge_len: List[int] = []
            for g in goal_graphs:
                gnode_vals.append(g.node_ids)
                ge_src.append(g.edge_index[0])
                ge_dst.append(g.edge_index[1])
                ge_attr.append(g.edge_attr)
                gn_len.append(int(g.node_ids.numel()))
                ge_len.append(int(g.edge_attr.numel()))
            self.goal_node_vals = torch.cat(gnode_vals).to(d)
            self.goal_edge_src = torch.cat(ge_src).to(d)
            self.goal_edge_dst = torch.cat(ge_dst).to(d)
            self.goal_edge_attr = torch.cat(ge_attr).to(d)
            self.goal_node_len = torch.tensor(gn_len, dtype=torch.long, device=d)
            self.goal_edge_len = torch.tensor(ge_len, dtype=torch.long, device=d)
            self.goal_node_start = (
                torch.cumsum(self.goal_node_len, 0) - self.goal_node_len
            )
            self.goal_edge_start = (
                torch.cumsum(self.goal_edge_len, 0) - self.goal_edge_len
            )

    def gid(self, inst_idx: int, state_id: int) -> int:
        return self.inst_offset[inst_idx] + int(state_id)

    @staticmethod
    def _gather_index(start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """cat([arange(s, s+l) for s, l]) fully vectorized."""
        total = int(length.sum())
        excl = torch.cumsum(length, 0) - length
        rep_start = torch.repeat_interleave(start, length)
        rep_excl = torch.repeat_interleave(excl, length)
        intra = torch.arange(total, device=start.device) - rep_excl
        return rep_start + intra

    def pack(
        self,
        state_gids: torch.Tensor,  # long [S] — all fringes' states, slot order
        fringe_lens: torch.Tensor,  # long [B]
        fringe_inst: Optional[torch.Tensor] = None,  # long [B] — instance per fringe
    ) -> Dict[str, torch.Tensor]:
        # Gather the per-batch slice on the host (where the resident buffers
        # live), then move only the assembled batch to the compute device below.
        d = self.store
        state_gids = state_gids.to(d)
        fringe_lens = fringe_lens.to(d)
        nl = self.node_len[state_gids]
        el = self.edge_len[state_gids]
        s_count = int(state_gids.numel())

        node_gather = self._gather_index(self.node_start[state_gids], nl)
        node_features = self.node_vals[node_gather]
        membership = torch.repeat_interleave(
            torch.arange(s_count, device=d), nl
        )

        node_offsets = torch.cumsum(nl, 0) - nl  # per-state node base
        edge_gather = self._gather_index(self.edge_start[state_gids], el)
        shift = torch.repeat_interleave(node_offsets, el)
        edge_index = torch.stack(
            [self.edge_src[edge_gather] + shift, self.edge_dst[edge_gather] + shift]
        )
        edge_attr = self.edge_attr[edge_gather]

        candidate_batch = torch.repeat_interleave(
            torch.arange(int(fringe_lens.numel()), device=d), fringe_lens
        )
        ptr = torch.zeros(int(fringe_lens.numel()) + 1, dtype=torch.long, device=d)
        ptr[1:] = torch.cumsum(fringe_lens, 0)
        out = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "membership": membership,
            "candidate_batch": candidate_batch,
            "fringe_ptr": ptr,
        }
        if self.has_goal and fringe_inst is not None:
            fringe_inst = fringe_inst.to(d)
            b_count = int(fringe_inst.numel())
            gnl = self.goal_node_len[fringe_inst]  # [B]
            gel = self.goal_edge_len[fringe_inst]
            goal_node_gather = self._gather_index(
                self.goal_node_start[fringe_inst], gnl
            )
            goal_node_features = self.goal_node_vals[goal_node_gather]
            # goal_batch = fringe index per goal node -> goal_emb is [B, D] and
            # goal_emb[candidate_batch] aligns one goal per candidate. With B=1
            # (inference) this is all-zeros, matching FringeEvalRL.
            goal_batch = torch.repeat_interleave(
                torch.arange(b_count, device=d), gnl
            )
            goal_node_offsets = torch.cumsum(gnl, 0) - gnl
            goal_edge_gather = self._gather_index(
                self.goal_edge_start[fringe_inst], gel
            )
            gshift = torch.repeat_interleave(goal_node_offsets, gel)
            goal_edge_index = torch.stack(
                [
                    self.goal_edge_src[goal_edge_gather] + gshift,
                    self.goal_edge_dst[goal_edge_gather] + gshift,
                ]
            )
            goal_edge_attr = self.goal_edge_attr[goal_edge_gather]
            out["goal_node_features"] = goal_node_features
            out["goal_edge_index"] = goal_edge_index
            out["goal_edge_attr"] = goal_edge_attr
            out["goal_batch"] = goal_batch
        # Move the assembled batch (small: ~one beam's worth of nodes/edges) to
        # the compute device. The big resident buffers never leave the host.
        if self.device != self.store:
            out = {k: v.to(self.device) for k, v in out.items()}
        return out


def segment_argmax(values: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Global index of the max element of each ptr-delimited segment.

    Vectorized (no per-segment .item() syncs): scatter-amax per segment, then
    the first position attaining it.
    """
    n_seg = int(ptr.numel() - 1)
    seg = torch.repeat_interleave(
        torch.arange(n_seg, device=values.device), (ptr[1:] - ptr[:-1]).to(values.device)
    )
    seg_max = torch.full((n_seg,), float("-inf"), device=values.device)
    seg_max = seg_max.scatter_reduce(0, seg, values, reduce="amax")
    is_max = values == seg_max[seg]
    pos = torch.arange(values.numel(), device=values.device)
    big = values.numel() + 1
    first = torch.full((n_seg,), big, dtype=torch.long, device=values.device)
    first = first.scatter_reduce(0, seg[is_max], pos[is_max], reduce="amin")
    return first


def pack_fringe_batch(
    fringes: Sequence[Tuple[InstanceCache, Sequence[int]]],
    goal_graphs: Optional[Sequence[Optional[StateGraph]]] = None,
) -> Dict[str, torch.Tensor]:
    """Pack B fringes into one disconnected mega-graph for batched training.

    Mirrors src/data.py::frontier_collate_fn semantics: `membership` holds the
    *global* candidate index, `candidate_batch` maps candidate -> fringe, and
    `fringe_ptr` delimits per-fringe candidate segments in the logits vector.

    `goal_graphs` (separated mode): one StateGraph PER FRINGE (the fringe's
    instance goal), aligned with `fringes`.  When given, emits
    `goal_node_features, goal_edge_index, goal_edge_attr, goal_batch` with
    `goal_batch=b` for fringe b's goal nodes — the same contract as
    GlobalFlatCache.pack (this is the slow reference path the parity test
    checks against).
    """
    node_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    membership_parts: List[torch.Tensor] = []
    candidate_batch_parts: List[torch.Tensor] = []
    ptr = [0]
    node_offset = 0
    cand_offset = 0
    for b, (cache, state_indices) in enumerate(fringes):
        for local_slot, si in enumerate(state_indices):
            g = cache.states[int(si)]
            n = int(g.node_ids.numel())
            node_parts.append(g.node_ids)
            membership_parts.append(
                torch.full((n,), cand_offset + local_slot, dtype=torch.int64)
            )
            if g.edge_index.numel() > 0:
                edge_index_parts.append(g.edge_index + node_offset)
                edge_attr_parts.append(g.edge_attr)
            node_offset += n
        k = len(state_indices)
        candidate_batch_parts.append(torch.full((k,), b, dtype=torch.int64))
        cand_offset += k
        ptr.append(cand_offset)

    out = {
        "node_features": torch.cat(node_parts, dim=0),
        "edge_index": (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=torch.int64)
        ),
        "edge_attr": (
            torch.cat(edge_attr_parts, dim=0)
            if edge_attr_parts
            else torch.zeros((0,), dtype=torch.int64)
        ),
        "membership": torch.cat(membership_parts, dim=0),
        "candidate_batch": torch.cat(candidate_batch_parts, dim=0),
        "fringe_ptr": torch.tensor(ptr, dtype=torch.int64),
    }

    if goal_graphs is not None and any(g is not None for g in goal_graphs):
        if len(goal_graphs) != len(fringes):
            raise ValueError(
                f"goal_graphs ({len(goal_graphs)}) must align with fringes "
                f"({len(fringes)})."
            )
        goal_node_parts: List[torch.Tensor] = []
        goal_edge_index_parts: List[torch.Tensor] = []
        goal_edge_attr_parts: List[torch.Tensor] = []
        goal_batch_parts: List[torch.Tensor] = []
        goal_node_offset = 0
        for b, g in enumerate(goal_graphs):
            if g is None:
                raise ValueError(
                    "separated mode requires a goal graph for every fringe; "
                    f"got None at fringe {b}."
                )
            gn = int(g.node_ids.numel())
            if gn == 0:
                # >=1 goal node per fringe keeps goal_batch rows aligned with
                # candidate_batch (see _pool_goal).
                raise ValueError(
                    f"separated mode requires >=1 goal node per fringe; "
                    f"fringe {b} has an empty goal graph."
                )
            goal_node_parts.append(g.node_ids)
            goal_batch_parts.append(torch.full((gn,), b, dtype=torch.int64))
            if g.edge_index.numel() > 0:
                goal_edge_index_parts.append(g.edge_index + goal_node_offset)
                goal_edge_attr_parts.append(g.edge_attr)
            goal_node_offset += gn
        out["goal_node_features"] = torch.cat(goal_node_parts, dim=0)
        out["goal_edge_index"] = (
            torch.cat(goal_edge_index_parts, dim=1)
            if goal_edge_index_parts
            else torch.zeros((2, 0), dtype=torch.int64)
        )
        out["goal_edge_attr"] = (
            torch.cat(goal_edge_attr_parts, dim=0)
            if goal_edge_attr_parts
            else torch.zeros((0,), dtype=torch.int64)
        )
        out["goal_batch"] = torch.cat(goal_batch_parts, dim=0)

    return out
