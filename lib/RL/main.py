import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, HeteroConv, SAGEConv

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
        self.convs = torch.nn.ModuleList(
            [
                HeteroConv(
                    {
                        ("state", "to", "state"): GCNConv(-1, hidden_dim),
                        ("goal", "to", "goal"): GCNConv(-1, hidden_dim),
                        ("state", "matches", "goal"): SAGEConv((-1, -1), hidden_dim),
                        ("goal", "rev_matches", "state"): SAGEConv(
                            (-1, -1), hidden_dim
                        ),
                    },
                    aggr="mean",
                )
                for _ in range(2)
            ]
        )
        self.out = torch.nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, data):
        x_dict, ei_dict = data.x_dict, data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, ei_dict)
        s = global_mean_pool(x_dict["state"], data["state"].batch)  # [B,H]
        g = global_mean_pool(x_dict["goal"], data["goal"].batch)  # [B,H]
        z = torch.cat([s, g, data.depth.unsqueeze(-1)], dim=-1)
        return self.out(z).squeeze(-1)


def _make_model(if_use_goal: bool):
    return HeteroGNN() if if_use_goal else BasicGNN()


def main_prediction() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Run GNN heuristic prediction on a DOT graph."
    )
    parser.add_argument("path", type=str, help="Path (long) to graph file")
    parser.add_argument("depth", type=int, help="Depth parameter")
    parser.add_argument("n_agents", type=int, help="Number of agents")
    parser.add_argument(
        "goal_file",
        type=str,
        help="The file containing the goal description in graph format",
    )
    args = parser.parse_args()

    # TODO: with need to know if we are working with/without hashing and with/without gaol
    USE_GOAL = True
    USE_HASH = True

    # Giovanni I've added the parsing of the goal file path as well. This does not need to be executed on every run for
    # the prediction as it is shared in the problem adding compilation error so you notice this line above:)

    s_dot_path = args.path

    G_s = load_nx_graph(s_dot_path, False)

    subdir = s_dot_path.split("out/state/")[1].split("/")[0]
    model_dir = os.path.join("lib", "RL", "results", subdir)

    if USE_GOAL:
        # g_dot_path = args.goal_file
        g_dot_path = (
            "lib/RL/results/CC_3_2_3__pl_6_poss_hashing_goal/CC_3_2_3__pl_6_goal_tree.dot"
        )
        G_g = load_nx_graph(g_dot_path, True)
    else:
        G_g = None

    if USE_HASH:
        model_dir += "_hashing"
    else:
        model_dir += "_mapping"

    if USE_GOAL:
        model_dir += "_goal"
    else:
        model_dir += "_nogoal"

    model_file = os.path.join(model_dir, "gnn_predictor.pt")

    model = _make_model(USE_GOAL)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    # 4) Predict
    depth = int(args.depth)
    sample = build_sample(G_s, depth, None, G_g)
    pred_value = predict_single(sample, model, device=device)

    # 5) Output
    with open("prediction.tmp", "w", encoding="utf-8") as fp:
        fp.write(f"VALUE:{pred_value}\n")


if __name__ == "__main__":
    main_prediction()
