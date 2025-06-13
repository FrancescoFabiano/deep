from __future__ import annotations

import argparse
import re
from pathlib import Path
import random

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import from_networkx


def _preprocess_sample(
        state_path: str,
        depth: int,
        target: int | None = None,
        goal_path: str | None = None,
):
    """
    • Converts DOT graphs to PyG objects.
    • Packs them (plus depth / optional target) into a dict the collate_fn
      already understands.
    """

    # --------------------------------------------------------------------------- #
    #  helper ─ load a DOT file and normalise node labels                         #
    # --------------------------------------------------------------------------- #
    def _load_graph_from_dot(dot_path: str, is_goal: bool = False) -> nx.DiGraph:
        """
        Reads a .dot file into a NetworkX DiGraph.
        Goal graphs sometimes contain negative node IDs; when `is_goal=True`
        we quote them so `pydot` parses correctly.
        """
        dot_path = Path(dot_path)

        if is_goal:
            fixed_path = dot_path.with_name("fixed_" + dot_path.name)
            with open(dot_path, "r") as f:
                content = f.read()
            # quote bare negative numbers that serve as node IDs
            content = re.sub(r'(?<![\w"]) (-\d+)(?![\w"])', r'"\1"', content)
            fixed_path.write_text(content)
            dot_path = fixed_path  # parse the patched file

        G_raw = nx.DiGraph(nx.nx_pydot.read_dot(str(dot_path)))
        return nx.convert_node_labels_to_integers(G_raw, label_attribute="orig_id")

    # --------------------------------------------------------------------------- #
    #  helper ─ convert NetworkX → PyG Data with safe tensor features             #
    # --------------------------------------------------------------------------- #
    def _nx_to_pyg(
            G: nx.DiGraph,
            node_attr_keys: list[str] = ("shape",),
            edge_attr_keys: list[str] = ("label",),
    ):
        """
        • Ensures every requested attribute exists (default 0).
        • Maps *all* non-numeric values to consecutive integers.
        • Guarantees `data.x` (and `data.edge_attr`, if requested) are tensors.
        """
        # ---- ensure uniform attribute sets ------------------------------------
        for _, d in G.nodes(data=True):
            for k in node_attr_keys:
                d.setdefault(k, 0)
        for _, _, d in G.edges(data=True):
            for k in edge_attr_keys:
                d.setdefault(k, 0)

        # ---- numeric-or-categorical → int -------------------------------------
        def _to_int(val, vocab: dict):
            # strip quotes that pydot keeps around strings
            s = str(val).strip('"')
            # numeric?
            if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
                return int(float(s))
            # categorical
            if s not in vocab:
                vocab[s] = len(vocab)
            return vocab[s]

        # per-attribute vocabularies
        node_vocab = {k: {} for k in node_attr_keys}
        edge_vocab = {k: {} for k in edge_attr_keys}

        for _, d in G.nodes(data=True):
            for k in node_attr_keys:
                d[k] = _to_int(d[k], node_vocab[k])

        for _, _, d in G.edges(data=True):
            for k in edge_attr_keys:
                d[k] = _to_int(d[k], edge_vocab[k])

        # ---- basic conversion --------------------------------------------------
        data = from_networkx(G)  # *without* grouping → no list-of-lists bug

        # ---- build node feature matrix ----------------------------------------
        node_feats = [
            [d[k] for k in node_attr_keys] for _, d in G.nodes(data=True)
        ]  # shape (N, len(node_attr_keys))
        data.x = torch.tensor(node_feats, dtype=torch.long)

        # ---- build edge features  (optional; comment out if unused) -----------
        if edge_attr_keys and G.number_of_edges() > 0:
            edge_feats = [
                [d[k] for k in edge_attr_keys] for *_, d in G.edges(data=True)
            ]
            data.edge_attr = torch.tensor(edge_feats, dtype=torch.long)

        return data

    # state graph -----------------------------------------------------------
    G_state = _load_graph_from_dot(state_path, is_goal=False)
    data_state = _nx_to_pyg(G_state)

    sample = {
        "state_graph": data_state,
        "depth": torch.tensor([depth], dtype=torch.long),
    }

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float32)

    # goal graph (optional) --------------------------------------------------
    if goal_path is not None:
        G_goal = _load_graph_from_dot(goal_path, is_goal=True)
        sample["goal_graph"] = _nx_to_pyg(G_goal)

    return sample


def _build_sample(state_path: str, depth: int, goal_path: str = None):
    return _preprocess_sample(
        state_path=state_path,
        depth=depth,
        goal_path=goal_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ░░  8.  PUBLIC PREDICTION APIS  ░░
# ──────────────────────────────────────────────────────────────────────────────
def predict_from_paths_single(
        model, state_path: str, depth: int, goal_path: str = None, device: str = "cpu"
):
    """
    Predicts distance for *one* new (state_path, depth, goal_path?) triple.
    """
    sample = _build_sample(state_path, depth, goal_path)
    with open("error.log", "a+") as f:
        print(f"path: {sample}", file=f)

    return _predict_single(model, sample, device)


def _to_device(batch_dict, dev):
    for k, v in batch_dict.items():
        if v is None:
            continue
        batch_dict[k] = v.to(dev)
    return batch_dict


class DistanceEstimator(nn.Module):
    """
    A light GNN → MLP that predicts the distance-to-goal.
    If USE_GOAL is True it embeds both graphs and concatenates them.
    """

    def __init__(
            self,
            in_channels_node: int,
            hidden_dim: int = 64,
            use_goal: bool = True,
    ):
        super().__init__()
        self.use_goal = use_goal

        # ── GNN for the *state* graph ────────────────────
        self.state_gnn1 = SAGEConv(in_channels_node, hidden_dim)
        self.state_gnn2 = SAGEConv(hidden_dim, hidden_dim)

        # ── (optional) GNN for the *goal* graph ──────────
        if use_goal:
            self.goal_gnn1 = SAGEConv(in_channels_node, hidden_dim)
            self.goal_gnn2 = SAGEConv(hidden_dim, hidden_dim)

        # ── “head”: graph-embedding + depth  →  distance ─
        # (+1 because we append the scalar depth)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * (2 if use_goal else 1) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # helper -----------------------------------------------------
    def _encode(self, data_batch, conv1, conv2):
        x = data_batch.x.float()
        edge_index = data_batch.edge_index
        x = F.relu(conv1(x, edge_index))
        x = F.relu(conv2(x, edge_index))
        return global_mean_pool(x, data_batch.batch)  # shape: [batch, hidden]

    # forward ----------------------------------------------------
    def forward(self, batch_dict):
        # ─ state graph ─
        state_emb = self._encode(
            batch_dict["state_graph"], self.state_gnn1, self.state_gnn2
        )

        # ─ goal graph (optional) ─
        if self.use_goal and batch_dict["goal_graph"] is not None:
            goal_emb = self._encode(
                batch_dict["goal_graph"], self.goal_gnn1, self.goal_gnn2
            )
            graph_emb = torch.cat([state_emb, goal_emb], dim=1)
        else:
            graph_emb = state_emb

        # ─ scalar depth (robust normalisation) ───────────────────────────
        depth = batch_dict["depth"].float()  # shape (B, 1)

        # Use population std (unbiased=False) so it is 0 instead of NaN for B=1
        depth_mean = depth.mean()
        depth_std = depth.std(unbiased=False)

        # If depth_std is zero (all depths identical) avoid dividing by 0
        depth = (depth - depth_mean) / (depth_std + 1e-6)

        # ─ concatenate and predict ─
        features = torch.cat([graph_emb, depth], dim=1)
        return self.mlp(features).squeeze(1)  # shape: [batch]


def _graph_collate_fn(batch):
    """
    • Works for both training/eval (samples include 'target')
      and inference (no 'target').
    • Keeps API identical for the model; the training loop just checks
      if 'target' is in the batch dict before computing the loss.
    """
    collated = {
        "state_graph": Batch.from_data_list([b["state_graph"] for b in batch]),
        "goal_graph": (
            Batch.from_data_list([b["goal_graph"] for b in batch])
            if "goal_graph" in batch[0]
            else None
        ),
        "depth": torch.stack([b["depth"] for b in batch]),
    }

    if "target" in batch[0]:  # ← only in training / evaluation
        collated["target"] = torch.stack([b["target"] for b in batch])

    return collated


def _predict_single(model, sample_dict, device):
    """
    `sample_dict` is *one* preprocessed sample (same structure as in training).
    """
    model.to(device)
    batch = _graph_collate_fn([sample_dict])
    batch = _to_device(batch, device)

    with torch.no_grad():
        return model(batch).item()


def load_distance_estimator(path: str = "distance_estimator.pt", map_to: str = None):
    """
    Returns a ready-to-use `DistanceEstimator` in eval mode.
    """
    map_to = map_to or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=map_to)

    model = DistanceEstimator(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_to).eval()

    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Run GNN heuristic prediction on a DOT graph.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("depth", type=int, help="Search depth parameter")
    parser.add_argument("goal_file", type=str, help="Path to goal DOT graph")
    parser.add_argument("model_file", type=str, help="Path to GGN model file")
    args = parser.parse_args()

    model = load_distance_estimator(args.model_file)

    # goal_path = model_dir + "/CC_2_2_3__pl_3.dot"

    pred = predict_from_paths_single(
        model,
        state_path=args.path,
        depth=args.depth,
        goal_path=args.goal_file,
        device=device,
    )

    with open("error.log", "a+") as f:
        print(f"path: {args.path}", file=f)
        print(f"depth: {args.depth}", file=f)
        print(f"depth: {args.goal_file}", file=f)
        print(f"pred {pred} - {int(pred)}", file=f)
        print("************************", file=f)

    pred = int(pred)

    print(f"{pred:.6f}")

if __name__ == "__main__":
    main()
