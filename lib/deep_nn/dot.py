"""Planner DOT files -> tensors. One parser for both learned heuristics.

The C++ writes one DOT per state (HelperPrint::print_dataset_format). In MERGED
mode the file is, in this order:

    0 -> 1 [label="2"];            epsilon -> goal parent          (TO_GOAL)
    <goal subtree edges>           goal parent -> ...  (labels >= 4)
    0 -> w [label="3"];            epsilon -> each designated world (TO_STATE)
    u -> v [label="<agent>"];      belief edges

The planner's tensor (GraphNN::state_to_tensor_minimal) numbers the nodes by
first appearance in that same order, so node index i here is symbolic id i
there, and `pointed_ids` are the targets of the TO_STATE edges -- the rule the
C++ DEBUG check (GraphNN::check_tensor_against_dot) applies.

SEPARATED mode: a separated-generated DOT holds the belief edges only and so
cannot say which worlds are designated (issue #2).  HASHED world ids do not
depend on the mode, so `separated_view` derives the planner's separated tensors
(state without epsilon/goal nodes, plus the goal graph) from a MERGED file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

TO_GOAL = 2    # TrainingDataset::m_to_goal_edge_id_int
TO_STATE = 3   # TrainingDataset::m_to_state_edge_id_int

_INT64_MIN = -(1 << 63)
_UINT64_MAX = (1 << 64) - 1

# `u -> v [label="3"];` -- node tokens may be quoted, signed, or bare.
_EDGE_RE = re.compile(
    r'^\s*"?(-?[0-9A-Za-z_]+)"?\s*->\s*"?(-?[0-9A-Za-z_]+)"?\s*'
    r'\[\s*label\s*=\s*"?(-?\d+)"?\s*\]\s*;?\s*$'
)
_HEADER_RE = re.compile(r"^\s*digraph\s+[0-9A-Za-z_]+\s*\{\s*$")
_CLOSE_RE = re.compile(r"^\s*\}\s*$")


def uint64_ids_to_int64(values: Sequence[int]) -> torch.Tensor:
    """Fold ids in [-2^63, 2^64-1] to int64, keeping the 64-bit pattern (the
    planner casts its uint64 hashes the same way)."""
    for v in values:
        if v < _INT64_MIN or v > _UINT64_MAX:
            raise ValueError(f"node id {v} outside [-2^63, 2^64-1]")
    raw = np.fromiter((v & _UINT64_MAX for v in values), dtype=np.uint64, count=len(values))
    return torch.from_numpy(raw.view(np.int64).copy())


def _empty_ids() -> torch.Tensor:
    return torch.zeros(0, dtype=torch.int64)


@dataclass
class StateGraph:
    """One planner state (or goal graph) with local node indexing."""

    node_ids: torch.Tensor      # int64 [n]
    edge_index: torch.Tensor    # int64 [2, e]
    edge_attr: torch.Tensor     # int64 [e]
    pointed_ids: torch.Tensor = field(default_factory=_empty_ids)  # int64 [p], indices into node_ids

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.numel())


def parse_dot(text: str) -> StateGraph:
    """Parse one planner DOT. Node and edge order are the file's order."""
    node_idx: dict[str, int] = {}
    src, dst, labels = [], [], []
    header = False
    for line in text.splitlines():
        m = _EDGE_RE.match(line)
        if m is not None:
            u, v, lab = m.groups()
            src.append(node_idx.setdefault(u, len(node_idx)))
            dst.append(node_idx.setdefault(v, len(node_idx)))
            labels.append(int(lab))
        elif _HEADER_RE.match(line):
            header = True
        elif line.strip() and not _CLOSE_RE.match(line):
            raise ValueError(f"line outside the planner DOT grammar: {line!r}")
    if not header:
        raise ValueError("missing `digraph <name> {` header")
    edge_index = torch.tensor([src, dst], dtype=torch.int64).reshape(2, -1)
    edge_attr = torch.tensor(labels, dtype=torch.int64)
    pointed = edge_index[1, edge_attr == TO_STATE]
    return StateGraph(uint64_ids_to_int64([int(t) for t in node_idx]), edge_index,
                      edge_attr, pointed)


def separated_view(g: StateGraph) -> Tuple[StateGraph, StateGraph]:
    """(state, goal) as the planner builds them in --dataset_separated mode, from
    a MERGED graph: the state keeps the world nodes only (designated first, then
    belief order -- the planner's separated numbering) and the goal is the goal
    subtree with its parent as node 0 (GraphNN::populate_with_goal)."""
    src, dst = g.edge_index
    to_goal = (g.edge_attr == TO_GOAL).nonzero().view(-1)
    if to_goal.numel() != 1:
        raise ValueError(
            "not a merged DOT (no epsilon -> goal edge). Separated-generated files "
            "carry no designated worlds (issue #2): generate the data in merged mode "
            "and derive the separated view from it.")
    eps, goal_parent = int(src[to_goal]), int(dst[to_goal])
    n = g.n_nodes
    goal = torch.zeros(n, dtype=torch.bool)
    goal[goal_parent] = True
    while True:  # closure over the goal subtree; converges in its depth
        grown = goal.clone()
        grown[dst[goal[src]]] = True
        if torch.equal(grown, goal):
            break
        goal = grown
    goal_edge = goal[src]
    keep = ~goal
    keep[eps] = False
    state_edge = keep[src] & keep[dst]
    return (_subgraph(g, keep, state_edge, g.pointed_ids),
            _subgraph(g, goal, goal_edge, _empty_ids()))


def _subgraph(g: StateGraph, nodes: torch.Tensor, edges: torch.Tensor,
              pointed: torch.Tensor) -> StateGraph:
    new_index = torch.cumsum(nodes.to(torch.int64), 0) - 1
    return StateGraph(g.node_ids[nodes], new_index[g.edge_index[:, edges]],
                      g.edge_attr[edges], new_index[pointed])


def load_state_graph(path: str | Path) -> StateGraph:
    try:
        return parse_dot(Path(path).read_text())
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def load_goal_graph(path: str | Path) -> StateGraph:
    """A generator-written ``goal_tree.dot`` (same grammar, no designated worlds)."""
    g = load_state_graph(path)
    if g.n_nodes == 0:
        raise ValueError(f"goal DOT has 0 nodes: {path}")
    return g
