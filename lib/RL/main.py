import json
import os
from datetime import datetime

import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
from tqdm import tqdm

import argparse


from utils import create_data_from_graph, predict_from_graph


class GNN(torch.nn.Module):
    def __init__(
        self, node_input_dim, edge_input_dim, hidden_dim=64, use_edge_attr=False
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        if self.use_edge_attr:
            # MLP for layer 1: must map edge_attr (dim=edge_input_dim)
            # => shape [node_input_dim * hidden_dim] = [2 * 64] = 128
            self.edge_mlp1 = torch.nn.Sequential(
                torch.nn.Linear(edge_input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, node_input_dim * hidden_dim),
            )
            self.conv1 = NNConv(
                in_channels=node_input_dim,
                out_channels=hidden_dim,
                nn=self.edge_mlp1,
                aggr="mean",
            )

            # MLP for layer 2: now in_channels=hidden_dim=64 => we need output [64 * 64] = 4096
            self.edge_mlp2 = torch.nn.Sequential(
                torch.nn.Linear(edge_input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.conv2 = NNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                nn=self.edge_mlp2,
                aggr="mean",
            )

        else:
            self.conv1 = GCNConv(node_input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.lin = Linear(hidden_dim, 1)  # Final linear to predict scalar

    def forward(self, x, edge_index, edge_attr, batch):
        if self.use_edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)


def generate_simulation_name(prefix: str = "analysis") -> str:
    """
    Generates a simulation name based on the current date and time.

    Args:
        prefix (str): Prefix for the simulation name (default is "Simulation").

    Returns:
        str: A string containing the prefix and a timestamp.
    """
    # Get the current date and time
    now = datetime.now()
    # Format the date and time into a readable string
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Combine the prefix and timestamp
    simulation_name = f"{prefix}_{timestamp}"
    return simulation_name

def main_training():
    path_data = "./data/ex_d_6.csv"

    simulation_path = generate_simulation_name()
    os.makedirs(simulation_path, exist_ok=True)

    start_df = pd.read_csv(path_data)
    start_df = start_df  # [:1000]
    all_samples = []

    for n, rows in tqdm(
        start_df.iterrows(), total=len(start_df), desc="pre-processing data..."
    ):
        # state = rows["State"]
        path_graph = rows[" Path"]
        graph = nx.Graph(nx.nx_pydot.read_dot(path_graph))
        depth = rows[" Depth"]
        dist_from_goal = rows[" Distance From Goal"]

        data_obj = create_data_from_graph(graph, dist_from_goal, depth)
        all_samples.append(data_obj)

    torch.save(all_samples, f"{simulation_path}/complete_dataset.pt")

    loaded_samples = torch.load(
        f"{simulation_path}/complete_dataset.pt", weights_only=False
    )

    # Define split sizes
    train_size = int(0.8 * len(loaded_samples))
    val_size = len(loaded_samples) - train_size
    # Split
    train_dataset, val_dataset = random_split(loaded_samples, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)
    " ************************************************************************************************************* "
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_edge_attr = True  # or False

    # Suppose each node has dim=2 (from create_data_from_graph).
    node_input_dim = 2

    # If you have n distinct edge labels => edge_attr has shape [num_edges, n].
    # Let’s say from your data the dimension is 'num_labels'. If you’re not sure,
    # you can check the shape of `data.edge_attr`.
    edge_input_dim = 0
    if use_edge_attr:
        # Possibly read from a sample:
        example_data = loaded_samples[0]
        edge_input_dim = example_data.edge_attr.size(1)  # num_labels

    model = GNN(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=64,
        use_edge_attr=use_edge_attr,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    num_epochs = 1000
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training model...")
    for epoch in pbar:
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    torch.save(model.state_dict(), f"{simulation_path}/complete_gnn_predictor.pt")
    " ************************************************************************************************************* "

    model.load_state_dict(torch.load(f"{simulation_path}/complete_gnn_predictor.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred_int = torch.round(pred)
            all_preds.append(pred_int.cpu())
            all_targets.append(batch.y.view(-1).cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mse = F.mse_loss(preds, targets)
    print(f"\nEvaluation MSE: {mse.item():.4f}")

    file_to_save = {"preds": preds.cpu().numpy(), "targets": targets.cpu().numpy()}

    # Convert all arrays to lists before saving
    data_serializable = {k: v.tolist() for k, v in file_to_save.items()}

    # Save to JSON file
    with open("data.json", "w") as f:
        json.dump(data_serializable, f)

    " ************************************************************************************************************* "
    print("\n EXAMPLE OF USAGE IN REAL-WORLD SETTINGS: ")
    rows = start_df.iloc[0, :]
    path_graph = rows[" Path"]
    depth = rows[" Depth"]

    pred = predict_from_graph(model, path_graph, depth, device)

    true = rows[" Distance From Goal"]
    print("Pred: :", pred)
    print("True: :", true)

def main_prediction():
    
    parser = argparse.ArgumentParser(description='Process a graph file path and a depth value.')
    parser.add_argument('path', type=str, help='Path to the graph file')
    parser.add_argument('depth', type=int, help='Depth for graph traversal or analysis')

    args = parser.parse_args()

    graph_path = args.path
    depth = args.depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = GNN(
        node_input_dim=2,
        edge_input_dim=3,
        hidden_dim=64,
        use_edge_attr=True).to(device)
    
    model_path = "lib/RL/results/no_goal_formula"
    model.load_state_dict(torch.load(f"{model_path}/complete_gnn_predictor.pt", map_location=torch.device('cpu')))

    pred = predict_from_graph(model, graph_path, depth, device)

    # Write to output file
    with open('prediction.tmp', 'w') as f:
        f.write(f"VALUE:{pred}\n")

if __name__ == "__main__":
    main_prediction()