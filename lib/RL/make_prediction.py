#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.nn import GCNConv, global_mean_pool, HeteroConv, SAGEConv

# Custom utility modules assumed in local project
from utils import build_sample, predict_single, load_nx_graph


class BasicGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.g1 = GCNConv(-1, hidden_dim)
        self.g2 = GCNConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim + 1, 1)

    def forward(self, data):
        x = F.relu(self.g1(data.x, data.edge_index))
        x = F.relu(self.g2(x, data.edge_index))
        pooled = global_mean_pool(x, data.batch)  # [B, H]
        z = torch.cat([pooled, data.depth.unsqueeze(-1)], dim=-1)
        return self.out(z).squeeze(-1)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HeteroConv({
                ("state", "to", "state"): GCNConv(-1, hidden_dim),
                ("goal", "to", "goal"): GCNConv(-1, hidden_dim),
                ("state", "matches", "goal"): SAGEConv((-1, -1), hidden_dim),
                ("goal", "rev_matches", "state"): SAGEConv((-1, -1), hidden_dim),
            }, aggr="mean") for _ in range(2)
        ])
        self.out = torch.nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, data):
        x_dict, ei_dict = data.x_dict, data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, ei_dict)
        s = global_mean_pool(x_dict["state"], data["state"].batch)
        g = global_mean_pool(x_dict["goal"], data["goal"].batch)
        z = torch.cat([s, g, data.depth.unsqueeze(-1)], dim=-1)
        return self.out(z).squeeze(-1)


def _make_model(use_goal: bool):
    return HeteroGNN() if use_goal else BasicGNN()

def main():

    parser = argparse.ArgumentParser(description="Run GNN heuristic prediction on a DOT graph.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("depth", type=int, help="Search depth parameter")
    parser.add_argument("goal_file", type=str, help="Path to goal DOT graph")
    parser.add_argument("model_file", type=str, help="Path to GGN model file")

    args = parser.parse_args()

    USE_GOAL = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load state and goal graphs
    G_s = load_nx_graph(args.path, False)
    G_g = load_nx_graph(args.goal_file, True) if USE_GOAL else None

    with open("error.log", "w") as f:
        print(f"Model file {args.model_file}.", file=f)
        print(f"depth file {args.depth}.", file=f)
        print(f"goal_file {args.goal_file}.", file=f)
        print(f"State file {args.path}.", file=f)

    # Load model
    model = _make_model(USE_GOAL)
    if not os.path.exists(args.model_file):
        with open("error.log", "w") as f:
            print(f"Model file {args.model_file} does not exist.", file=f)
        sys.exit(1)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.to(device)

    # Build input and predict
    sample = build_sample(G_s, args.depth, None, G_g)
    pred_value = predict_single(sample, model, device=device)

    # Print for C++ consumption
    print(f"{pred_value:.6f}")


if __name__ == "__main__":
    main()
