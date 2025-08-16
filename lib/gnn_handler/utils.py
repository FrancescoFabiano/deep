from typing import Dict, Tuple

import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data


def process_input_graph(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Converts a NetworkX graph with string edge labels (like 'a,b,c')
    into an adjacency matrix and an edge feature tensor.

    Args:
        G (nx.Graph): A NetworkX graph with string-labeled edges.

    Returns:
        adj_matrix (np.ndarray): Binary adjacency matrix [num_nodes, num_nodes]
        edge_attr_matrix (np.ndarray): Edge attributes as one-hot label matrix [num_nodes, num_nodes, num_labels]
        node_to_idx (Dict[str, int]): Mapping from node name to matrix index
    """

    # 1. Get nodes and index mapping
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # 2. Extract all unique labels
    unique_labels = set()
    for _, _, data in G.edges(data=True):
        label_str = data.get("label", "").strip('"')
        if label_str:
            unique_labels.update(label_str.split(","))

    label_list = sorted(list(unique_labels))  # consistent order
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # 3. Initialize matrices
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_attr_matrix = np.zeros((num_nodes, num_nodes, num_labels), dtype=int)

    # 4. Populate matrices
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = 1

        label_str = data.get("label", "").strip('"')
        if label_str:
            for label in label_str.split(","):
                if label in label_to_idx:
                    # print(u, v, label)
                    edge_attr_matrix[i, j, label_to_idx[label]] = 1
    """print("\n")
    print(adj_matrix)
    print(edge_attr_matrix)
    print(node_to_idx)"""
    return adj_matrix, edge_attr_matrix, node_to_idx


def create_data_from_graph(G: nx.DiGraph, dist_from_goal: int, depth: int) -> Data:
    """
    Processes the given directed graph and returns a PyG Data object
    ready to be batched or fed to a GNN.

    Args:
        G (nx.DiGraph): The input graph (nodes, edges with 'label' attributes).
        dist_from_goal (int): The scalar we want to predict (target).
        depth (int): An additional scalar that may be useful as a feature.

    Returns:
        Data: torch_geometric.data.Data object with the fields:
            - x: node features [num_nodes, node_feat_dim]
            - edge_index: [2, num_edges], describing graph connectivity
            - edge_attr: [num_edges, edge_feat_dim], describing edge features
            - y: [1], the scalar target
    """
    # 1) Process the graph => adjacency & edge_attr_matrix
    adj_matrix, edge_attr_matrix, node_to_idx = process_input_graph(G)
    # num_nodes = adj_matrix.shape[0]

    # 2) NODE FEATURES
    # Example: each node gets [1.0, depth]
    x_list = []
    # count = 0
    for node, idx in node_to_idx.items():
        x_list.append([int(node), int(depth)])
        # count += 1
    x = torch.tensor(x_list, dtype=torch.float)  # shape: [num_nodes, 2]

    # 3) EDGE INDEX
    # For a directed adjacency matrix, nonzero entry (i, j) => edge from i to j
    edge_index_np = np.array(np.nonzero(adj_matrix))  # shape: [2, num_edges]
    edge_index = torch.from_numpy(edge_index_np).long()

    # 4) EDGE ATTRIBUTES (Fix the warning)
    # Gather all edge_attr_matrix rows in one NumPy array:
    src_indices, tgt_indices = edge_index_np
    edge_attr_np = edge_attr_matrix[
        src_indices, tgt_indices, :
    ]  # shape: [num_edges, num_labels]
    edge_attr = torch.from_numpy(edge_attr_np).float()

    # 5) TARGET
    y = torch.tensor([dist_from_goal], dtype=torch.float)  # shape: [1]

    # 6) Build final Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


def predict_from_graph(
    model: torch.nn.Module,
    G: nx.DiGraph,
    depth: int,
    device: torch.device = "cuda",
) -> float:
    """
    Loads a graph from a file, processes it, and runs inference with the trained model.

    Args:
        model (torch.nn.Module): The trained GNN model.
        G (nx.DiGraph): graph.
        depth (int): Depth to include as node feature.
        device (torch.device): The device to run inference on.

    Returns:
        float: The predicted distance from the goal.
    """

    # 2. Dummy target value (not used for inference, just for Data creation)
    dummy_dist_from_goal = 0

    # 3. Create PyG Data object
    data_obj = create_data_from_graph(G, dummy_dist_from_goal, depth)

    # 4. Move data to the appropriate device and add batch index
    data_obj = data_obj.to(device)
    data_obj.batch = torch.zeros(
        data_obj.num_nodes, dtype=torch.long, device=device
    )  # single graph = batch 0

    # 5. Set model to eval mode and predict
    model.eval()
    with torch.no_grad():
        pred = model(
            data_obj.x, data_obj.edge_index, data_obj.edge_attr, data_obj.batch
        )
        pred_int = torch.round(pred)

    return pred_int.item()
