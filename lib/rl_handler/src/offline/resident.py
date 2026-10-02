"""Every state graph of every instance in a few concatenated tensors, so a batch of
beams is packed by index arithmetic instead of a Python loop over slots.

`pack_batch` (batching.py) builds a batch with ~batch_size x F small `torch.cat`s
and host->device copies. `ResidentTrees.pack` produces the SAME tensors (byte
identical, checked in tests/test_resident.py) with ~10 vectorised ops on the device
the trees live on and no per-step copy but the slot ids.

MEASURED (2026-10-01, idle GPU, F=8, batch 32, two packs per step): blocks-world
pack 4.4 -> 3.2 ms of an ~18 ms step; muddy-child 50 -> 44 ms of ~325 ms, with its
3.5 GB of trees forced to a CPU store. No net step-time gain, so the trainer keeps it
OFF by default (`--resident-packing`). Kept because it is correct and the right
shape if packing ever dominates again (e.g. far larger batches).

Layout: states of all instances concatenated in (instance, state id) order; per
state the node, edge and pointed counts and their start offsets. Edges keep the
per-state LOCAL node indices and are shifted to the batch layout at pack time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .encoder import InstanceCache, StateGraph, pack_goal_tensors


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, 0) - x


class ResidentTrees:
    """`device`: where the packed batches go. `store`: where the trees live and the
    gathers run -- the same device, or "cpu" when the trees are too big for the GPU
    (muddy-child's 28 trees are 3.5 GB; the batch is then packed on the CPU with the
    same vectorised ops and copied once)."""

    def __init__(self, caches: Dict[str, InstanceCache], device, store=None):
        self.device = device
        self.store = device if store is None else store
        device = self.store
        self.base: Dict[str, int] = {}            # instance -> global id of its state 0
        nodes: List[torch.Tensor] = []
        edges: List[torch.Tensor] = []
        attrs: List[torch.Tensor] = []
        pointed: List[torch.Tensor] = []
        n_nodes: List[int] = []
        n_edges: List[int] = []
        n_pointed: List[int] = []
        for name, cache in caches.items():
            self.base[name] = len(n_nodes)
            for g in cache.states:
                nodes.append(g.node_ids)
                edges.append(g.edge_index)
                attrs.append(g.edge_attr)
                pointed.append(g.pointed_ids)
                n_nodes.append(g.n_nodes)
                n_edges.append(int(g.edge_index.shape[1]))
                n_pointed.append(int(g.pointed_ids.numel()))
        if not n_nodes:
            raise ValueError("no state graphs to make resident")
        self._bytes = sum(t.numel() * t.element_size() for t in nodes + edges + attrs + pointed)
        self.node_ids = torch.cat(nodes).to(device)                 # [sum n]
        self.edge_index = torch.cat(edges, dim=1).to(device)        # [2, sum e], local indices
        self.edge_attr = torch.cat(attrs).to(device)                # [sum e]
        self.pointed_ids = torch.cat(pointed).to(device)            # [sum p], local indices
        # counts on the CPU (sizes are needed as Python ints) and on the device (gathers)
        self._n_nodes_cpu = torch.tensor(n_nodes, dtype=torch.int64)
        self._n_edges_cpu = torch.tensor(n_edges, dtype=torch.int64)
        self._n_pointed_cpu = torch.tensor(n_pointed, dtype=torch.int64)
        self.n_nodes = self._n_nodes_cpu.to(device)
        self.n_edges = self._n_edges_cpu.to(device)
        self.n_pointed = self._n_pointed_cpu.to(device)
        self.node_start = _exclusive_cumsum(self.n_nodes)
        self.edge_start = _exclusive_cumsum(self.n_edges)
        self.pointed_start = _exclusive_cumsum(self.n_pointed)

    @property
    def n_states(self) -> int:
        return int(self._n_nodes_cpu.numel())

    def memory_bytes(self) -> int:
        return self._bytes

    @staticmethod
    def size_bytes(caches: Dict[str, InstanceCache]) -> int:
        return sum(t.numel() * t.element_size() for c in caches.values() for g in c.states
                   for t in (g.node_ids, g.edge_index, g.edge_attr, g.pointed_ids))

    def global_ids(self, picks: Sequence[Tuple[str, Sequence[int]]]) -> Tuple[torch.Tensor, List[int]]:
        """Global state id of every slot, in pick order, and the beam sizes."""
        ids: List[int] = []
        sizes: List[int] = []
        for name, beam in picks:
            b = self.base[name]
            ids.extend(b + int(s) for s in beam)
            sizes.append(len(beam))
        return torch.tensor(ids, dtype=torch.int64), sizes

    @staticmethod
    def _ranges(count: torch.Tensor, total: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For per-slot counts: segment id of each element, its position within the
        segment, and the segment's start in the output."""
        start = _exclusive_cumsum(count)
        seg = torch.repeat_interleave(torch.arange(count.numel(), device=device), count, output_size=total)
        pos = torch.arange(total, device=device) - start[seg]
        return seg, pos, start

    def pack(self, picks: Sequence[Tuple[str, Sequence[int]]],
             goal_graphs: Optional[Sequence[StateGraph]] = None) -> Dict[str, torch.Tensor]:
        """Same keys and values as `batching.pack_batch` (on `self.device`)."""
        out = self._pack_on_store(picks)
        if str(self.store) != str(self.device):
            out = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in out.items()}
        if goal_graphs is not None:
            if len(goal_graphs) != len(picks):
                raise ValueError(f"goal_graphs ({len(goal_graphs)}) must align with picks ({len(picks)}).")
            out.update({k: v.to(self.device) for k, v in pack_goal_tensors(goal_graphs).items()})
        return out

    def _pack_on_store(self, picks) -> Dict[str, torch.Tensor]:
        dev = self.store
        sid_cpu, sizes = self.global_ids(picks)
        total_n = int(self._n_nodes_cpu[sid_cpu].sum())
        total_e = int(self._n_edges_cpu[sid_cpu].sum())
        total_p = int(self._n_pointed_cpu[sid_cpu].sum())
        sid = sid_cpu.to(dev, non_blocking=True)

        cnt_n = self.n_nodes[sid]
        membership, pos_n, new_node_start = self._ranges(cnt_n, total_n, dev)
        node_idx = self.node_start[sid][membership] + pos_n
        node_features = self.node_ids[node_idx]

        cnt_e = self.n_edges[sid]
        eseg, pos_e, _ = self._ranges(cnt_e, total_e, dev)
        edge_idx = self.edge_start[sid][eseg] + pos_e
        edge_index = self.edge_index[:, edge_idx] + new_node_start[eseg]
        edge_attr = self.edge_attr[edge_idx]

        cnt_p = self.n_pointed[sid]
        pseg, pos_p, _ = self._ranges(cnt_p, total_p, dev)
        p_idx = self.pointed_start[sid][pseg] + pos_p
        pointed_ids = self.pointed_ids[p_idx] + new_node_start[pseg]

        beam_sizes = torch.tensor(sizes, dtype=torch.int64)
        slot_offset = _exclusive_cumsum(beam_sizes)
        candidate_batch = torch.repeat_interleave(torch.arange(len(sizes)), beam_sizes)
        out = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "membership": membership,
            "pointed_ids": pointed_ids,
            "candidate_batch": candidate_batch.to(dev, non_blocking=True),
            "slot_offset": slot_offset.to(dev, non_blocking=True),
            "n_slots": int(sid_cpu.numel()),
            "beam_sizes": beam_sizes.to(dev, non_blocking=True),
        }
        return out
