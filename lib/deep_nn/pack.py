"""FringeEvalRL::fringe_to_tensor_minimal, in Python.

One fringe of K <= F states becomes ONE graph:

    node_features  int64 [N]    per-state node ids, concatenated in slot order
    edge_index     int64 [2,E]  per-state edges shifted by the state's node offset
    edge_attr      int64 [E]    edge labels
    membership     int64 [N]    slot index of each node
    pointed_ids    int64 [P]    designated worlds, shifted like the edges
    mask           uint8 [F]    1 for the K occupied slots

Separated mode adds the instance goal (``pack_goal_tensors``) as four more
tensors, ``goal_batch`` all zeros -- the planner sends one goal per call.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .dot import StateGraph


def pack_fringe(cache, state_indices: Sequence[int], fringe_size: int) -> Dict[str, torch.Tensor]:
    """Slot k holds ``cache.states[state_indices[k]]``; K = len(state_indices) <= F."""
    k = len(state_indices)
    if k == 0:
        raise ValueError("cannot pack an empty fringe")
    if k > int(fringe_size):
        raise ValueError(f"fringe has {k} states > fringe_size {fringe_size}")

    nodes, edges, attrs, member, pointed = [], [], [], [], []
    offset = 0
    for slot, si in enumerate(state_indices):
        g: StateGraph = cache.states[int(si)]
        n = g.n_nodes
        if n == 0:
            # The C++ never produces one; only a truncated DOT can. Its pooled
            # slot would be silently all-zero, so refuse it here.
            raise ValueError(f"state graph for slot {slot} (state id {int(si)}) has 0 nodes")
        nodes.append(g.node_ids)
        member.append(torch.full((n,), slot, dtype=torch.int64))
        edges.append(g.edge_index + offset)
        attrs.append(g.edge_attr)
        pointed.append(g.pointed_ids + offset)
        offset += n

    mask = torch.zeros(int(fringe_size), dtype=torch.uint8)
    mask[:k] = 1
    return {
        "node_features": torch.cat(nodes),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(attrs),
        "membership": torch.cat(member),
        "pointed_ids": torch.cat(pointed),
        "mask": mask,
    }


def pack_goal_tensors(goal_graphs: Sequence[StateGraph]) -> Dict[str, torch.Tensor]:
    """The four goal tensors for B fringes; ``goal_batch = b`` marks fringe b's goal
    nodes so one pooled goal row aligns with each fringe."""
    nodes, edges, attrs, batch = [], [], [], []
    offset = 0
    for b, g in enumerate(goal_graphs):
        if g is None or g.n_nodes == 0:
            raise ValueError(f"separated mode needs a non-empty goal graph for every fringe (fringe {b})")
        nodes.append(g.node_ids)
        batch.append(torch.full((g.n_nodes,), b, dtype=torch.int64))
        edges.append(g.edge_index + offset)
        attrs.append(g.edge_attr)
        offset += g.n_nodes
    return {
        "goal_node_features": torch.cat(nodes),
        "goal_edge_index": torch.cat(edges, dim=1),
        "goal_edge_attr": torch.cat(attrs),
        "goal_batch": torch.cat(batch),
    }


def pack_fringes(fringes: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Many packed fringes in one graph for batched training. ``membership``
    continues across fringes (global slot ids), ``candidate_batch[j]`` is the
    fringe of slot j, ``slot_offset[b]`` the first slot of fringe b.  The mask
    is dropped: every slot in a batch is occupied."""
    keys = ("node_features", "edge_index", "edge_attr", "membership", "pointed_ids")
    parts: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
    cand, slot_offset = [], []
    node_off = slot_off = 0
    for b, p in enumerate(fringes):
        k = int(p["mask"].sum())
        parts["node_features"].append(p["node_features"])
        parts["edge_index"].append(p["edge_index"] + node_off)
        parts["edge_attr"].append(p["edge_attr"])
        parts["membership"].append(p["membership"] + slot_off)
        parts["pointed_ids"].append(p["pointed_ids"] + node_off)
        cand.append(torch.full((k,), b, dtype=torch.int64))
        slot_offset.append(slot_off)
        node_off += int(p["node_features"].numel())
        slot_off += k
    out = {k: torch.cat(v, dim=1 if k == "edge_index" else 0) for k, v in parts.items()}
    out["candidate_batch"] = torch.cat(cand)
    out["slot_offset"] = torch.tensor(slot_offset, dtype=torch.int64)
    out["n_slots"] = slot_off
    return out
