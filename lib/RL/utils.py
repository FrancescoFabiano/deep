from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pydot

import torch
from torch_geometric.data import Data, HeteroData

F_NODE = 1  # hashed‑ID (not degree)
F_EDGE = 1  # dummy weight
_TWO_64 = 2**64  # constant so it is computed once


def _hash64(label) -> float:
    if int(label) == -1:
        return -1.0
    else:
        uid = np.uint64(int(label))
        return float(uid) / _TWO_64  # ∈ [0,1)


# --------------------------------------------------------------------
# fixed‑width node & edge feats
# --------------------------------------------------------------------
def _to_pyg_tensors(G: nx.Graph, node_to_idx: dict[int, int]):
    N = G.number_of_nodes()

    # node‑features
    x = torch.zeros((N, F_NODE), dtype=torch.float)
    x[:, 0] = torch.tensor([int(n) for n in G.nodes], dtype=torch.float)
    x[:, 0] = torch.tensor([_hash64(n) for n in G.nodes], dtype=torch.float)
    # x[:, 1] = torch.tensor([G.degree(n) for n in G.nodes], dtype=torch.float)

    # edges
    src, dst = (
        zip(*[(node_to_idx[comb[0]], node_to_idx[comb[1]]) for comb in G.edges])
        if G.edges
        else ([], [])
    )
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # edge_attr  = torch.ones((len(src), F_EDGE), dtype=torch.float)  # weight=1
    edge_attr = torch.zeros((len(src), 1), dtype=torch.float)
    for c, (u, v, data) in enumerate(G.edges(data=True)):
        edge_attr[c] = int(data["label"].strip('"'))
    return x, edge_index, edge_attr


def _build_hetero_sample(
    G_state: nx.Graph, G_goal: nx.Graph, depth: float, dist_from_goal: float = None
):
    idx_state = {n: i for i, n in enumerate(G_state.nodes)}
    idx_goal = {n: i for i, n in enumerate(G_goal.nodes)}

    data = HeteroData()

    x_s, ei_s, ea_s = _to_pyg_tensors(G_state, idx_state)
    x_g, ei_g, ea_g = _to_pyg_tensors(G_goal, idx_goal)
    # ── node features ────────────────────────────────────────────
    data["state"].x = x_s
    data["goal"].x = x_g

    # ── homogeneous edges  state→state , goal→goal ───────────────
    data["state", "to", "state"].edge_index = ei_s
    data["state", "to", "state"].edge_attr = ea_s

    data["goal", "to", "goal"].edge_index = ei_g
    data["goal", "to", "goal"].edge_attr = ea_g

    # ── alignment edges state⇄goal (as before) ───────────────────
    src = [idx_state[n] for n in G_state.nodes if n in idx_goal]
    dst = [idx_goal[n] for n in G_state.nodes if n in idx_goal]

    def make_edges(s, d):
        ei = torch.tensor([s, d], dtype=torch.long)
        ea = torch.ones((len(s), F_EDGE), dtype=torch.float)
        if len(s) == 0:  # keep schema if empty
            ei = torch.empty((2, 0), dtype=torch.long)
            ea = torch.empty((0, F_EDGE), dtype=torch.float)
        return ei, ea

    ei_fwd, ea_fwd = make_edges(src, dst)
    ei_rev, ea_rev = make_edges(dst, src)

    data["state", "matches", "goal"].edge_index = ei_fwd
    data["state", "matches", "goal"].edge_attr = ea_fwd
    data["goal", "rev_matches", "state"].edge_index = ei_rev
    data["goal", "rev_matches", "state"].edge_attr = ea_rev

    # ── graph‑level attrs ────────────────────────────────────────
    data.depth = torch.tensor([depth], dtype=torch.float)
    if dist_from_goal is not None:
        data.y = torch.tensor([dist_from_goal], dtype=torch.float)
    return data


def _build_state_sample(
    G_state: nx.Graph, depth: float, dist_from_goal: float | None = None
):
    idx = {n: i for i, n in enumerate(G_state.nodes)}
    x, ei, ea = _to_pyg_tensors(G_state, idx)

    data = Data()
    data.x = x
    data.edge_index = ei
    data.edge_attr = ea
    data.depth = torch.tensor([depth], dtype=torch.float)
    if dist_from_goal is not None:  # 3️⃣  use the *right* variable
        data.y = torch.tensor([dist_from_goal], dtype=torch.float)
    return data


def build_sample(G_state, depth, dist=None, G_goal=None):
    if G_goal is not None:
        return _build_hetero_sample(G_state, G_goal, depth, dist)
    else:
        return _build_state_sample(G_state, depth, dist)


def predict_single(sample_graph, model, device="cpu"):
    """
    sample_graph : G_s (+ G_g if USE_GOAL)   →  PyG Data/HeteroData
    model        : trained StateOnlyGNN or HeteroGNN
    returns      : scalar prediction (Python float)
    """
    model.eval()
    with torch.no_grad():
        g = _ensure_batch_field(_to_device(sample_graph, device))
        out = model(g)  # model knows whether g is Data or HeteroData
        return out.item()


def _to_device(d, device):
    """Move a Data / HeteroData object to CUDA/CPU recursively."""
    return d.to(device, non_blocking=True)


def _ensure_batch_field(d):
    """
    PyG Data/HeteroData created by hand lack the .batch vectors
    expected by global_mean_pool.  DataLoader adds them automatically,
    but for single‑graph inference we add them here.
    """
    if isinstance(d, Data):
        if not hasattr(d, "batch"):
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=d.x.device)
    else:  # HeteroData
        for node_type in d.node_types:
            store = d[node_type]
            if not hasattr(store, "batch"):
                store.batch = torch.zeros(
                    store.num_nodes, dtype=torch.long, device=store.x.device
                )
    return d


def load_nx_graph(dot_path: str, if_goal: bool):
    if not if_goal:
        pydot_graph, = pydot.graph_from_dot_file(dot_path)
        return nx.DiGraph(nx.nx_pydot.from_pydot(pydot_graph))
    else:
        dot_path = Path(dot_path)
        fixed_path = dot_path.with_name("fixed_" + dot_path.name)

        # Quote negative node IDs like -1
        with open(dot_path, "r") as f:
            content = f.read()

        # Add quotes around negative numbers only when they are node identifiers
        content_fixed = re.sub(r'(?<![\w"]) (-\d+)(?![\w"])', r'"\1"', content)

        # Save to a new file
        with open(fixed_path, "w") as f:
            f.write(content_fixed)

        (dot_graph,) = pydot.graph_from_dot_file(fixed_path)
        return nx.drawing.nx_pydot.from_pydot(dot_graph)
