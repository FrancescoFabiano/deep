"""All states of one instance parsed once; beam assembly is then index math."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .dot import StateGraph, load_state_graph, separated_view


class InstanceCache:
    """State graphs indexed by the tree's state id, plus (separated mode) the
    instance's goal graph derived from the same merged DOTs.

    A ``.pt`` file skips re-parsing; it is rebuilt when the path list, the
    mode, or this format version changes.
    """

    CACHE_VERSION = 2

    def __init__(self, states: List[StateGraph], goal: Optional[StateGraph] = None):
        self.states = states
        self.goal = goal
        self.n_nodes = torch.tensor([s.n_nodes for s in states], dtype=torch.long)

    @classmethod
    def from_paths(
        cls,
        dot_paths: Sequence[str],
        separated: bool = False,
        cache_file: Optional[Path] = None,
        verbose: bool = True,
    ) -> "InstanceCache":
        paths = list(map(str, dot_paths))
        key = {"version": cls.CACHE_VERSION, "paths": paths, "separated": bool(separated)}
        if cache_file is not None and cache_file.exists():
            try:
                payload = torch.load(cache_file, weights_only=False)
            except Exception:
                payload = {}
            if all(payload.get(k) == v for k, v in key.items()):
                return cls(payload["states"], payload.get("goal"))

        t0 = time.time()
        graphs = [load_state_graph(p) for p in paths]
        goal = None
        if separated:
            views = [separated_view(g) for g in graphs]
            graphs = [s for s, _ in views]
            goal = views[0][1]
            for p, (_, gl) in zip(paths, views):
                if not (torch.equal(gl.node_ids, goal.node_ids)
                        and torch.equal(gl.edge_index, goal.edge_index)
                        and torch.equal(gl.edge_attr, goal.edge_attr)):
                    raise ValueError(f"{p}: goal subgraph differs from {paths[0]}")
        if verbose:
            dt = time.time() - t0
            print(f"[cache] parsed {len(graphs)} DOT files in {dt:.1f}s "
                  f"({len(graphs) / max(dt, 1e-9):.0f} files/s)")
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**key, "states": graphs, "goal": goal}, cache_file)
        return cls(graphs, goal)
