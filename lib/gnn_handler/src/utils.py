from __future__ import annotations

import math
import os
import random

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pydot
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

from src.model import BaseModel
from src.models.distance_estimator import (
    DistanceEstimator,
    OnnxDistanceEstimatorWrapper,
)


class PrecomputedGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloaders(
    train_samples,
    eval_samples,
    batch_size=256,
    shuffle=True,
    seed=42,
    num_workers=0,
):
    # Torch RNG shared by both loaders so shuffling is reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    def _build_loader(samples, is_train):
        return DataLoader(
            PrecomputedGraphDataset(samples),
            batch_size=batch_size,
            shuffle=(shuffle and is_train),
            collate_fn=graph_collate_fn,
            num_workers=num_workers,
            generator=g,
            worker_init_fn=lambda wid: seed_everything(seed + wid),
        )

    return _build_loader(train_samples, True), _build_loader(eval_samples, False)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_dot(path: Path) -> nx.DiGraph:
    src = path.read_text()
    dot = pydot.graph_from_dot_data(src)[0]
    return nx.nx_pydot.from_pydot(dot)


def plot_graph(G: nx.Graph):
    assert isinstance(G, nx.MultiDiGraph)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=800)
    nx.draw_networkx_labels(G, pos)
    for i, (u, v, k, d) in enumerate(G.edges(keys=True, data=True)):
        rad = 0.2 * (k + 1)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            arrows=True,
        )
        x, y = (pos[u] + pos[v]) / 2
        plt.text(x, y + rad, str(d.get("label", "")), fontsize=9, color="red")
    plt.axis("off")
    plt.show()


def diagnose_data(data):
    print("\n=== PyG Graph Diagnostics ===")

    # Node info
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Node indices: {list(range(data.num_nodes))}")
    print(f"Node names: {data.node_names}")
    if hasattr(data, "x"):
        print(f"Node features - shape node (x):\n{data.x}")
    else:
        print("No node features ('x') set.")

    # Edge info
    print(f"Number of edges: {data.num_edges}")
    print(f"Edge index (source -> target):\n{data.edge_index}")

    # Edge labels
    if hasattr(data, "edge_attr"):
        print(f"Edge attributes (labels):\n{data.edge_attr.squeeze()}")
    else:
        print("No edge attributes ('edge_attr') set.")

    # Edge labels
    if hasattr(data, "edge_type"):
        print(f"Edge type :\n{data.edge_type.squeeze()}")
    else:
        print("No edge type ('edge_type') set.")

    # Check graph consistency
    src_nodes = data.edge_index[0].tolist()
    tgt_nodes = data.edge_index[1].tolist()
    edge_labels = data.edge_attr.squeeze().tolist()

    print("--- Edge Summary ---")
    for i, (src, tgt, label) in enumerate(zip(src_nodes, tgt_nodes, edge_labels)):
        print(f"Edge {i}: {src} -> {tgt} | Label: {label}")


def _nx_to_pyg(G: nx.DiGraph, plot: bool = False, diagnose: bool = False) -> Data:
    for n, data in G.nodes(data=True):
        data["shape"] = {"circle": 0, "doublecircle": 1}.get(
            data.get("shape", "circle"), 0
        )
    for u, v, d in G.edges(data=True):
        d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))
    # optional plotting
    if plot:
        plot_graph(G)

    # convert to a PyG Data object
    data = from_networkx(G)

    # 1) Directed edges only; no duplication
    # Edge labels come from d["label"]
    edge_labels = data.edge_label.view(-1, 1).float()  # [E,1]
    data.edge_index = data.edge_index.long()
    data.edge_attr = edge_labels

    # 2) Node IDs → float tensor
    raw_ids = [int(n) for n in G.nodes()]
    data.node_names = torch.tensor(raw_ids, dtype=torch.float)

    # 3) Clean up unused fields if you like
    # del data.edge_label, data.edge_type, data.x

    if diagnose:
        diagnose_data(data)

    return data


def preprocess_sample(
    state_path: str,
    depth: int | None = None,
    target: int | None = None,
    goal_path: Optional[str] = None,
    if_plot_graph: bool = False,
    if_diagnose: bool = False,
) -> Dict[str, Any]:
    Gs = _load_dot(Path(state_path))
    if if_plot_graph:
        plot_graph(Gs)
    ds = _nx_to_pyg(Gs)
    assert ds.edge_index.size(1) == ds.edge_attr.size(
        0
    ), f"Mismatch: {ds.edge_index.size(1)} edges vs {ds.edge_attr.size(0)} attrs"

    if if_diagnose:
        diagnose_data(ds)
    ds.name = Path(state_path).stem

    sample = {"state_graph": ds}

    if depth is not None:
        sample["depth"] = torch.tensor([depth])

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float)

    if goal_path is not None:
        Gg = _load_dot(Path(goal_path))
        if if_plot_graph:
            plot_graph(Gg)
        dg = _nx_to_pyg(Gg)
        assert dg.edge_index.size(1) == dg.edge_attr.size(
            0
        ), f"Mismatch: {dg.edge_index.size(1)} edges vs {dg.edge_attr.size(0)} attrs"
        if if_diagnose:
            diagnose_data(dg)
        dg.name = Path(goal_path).stem
        sample["goal_graph"] = dg

    return sample


def graph_collate_fn(batch):
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
        "depth": (
            torch.stack([b["depth"] for b in batch]) if "depth" in batch[0] else None
        ),
    }

    if "target" in batch[0]:  # ← only in training / evaluation
        collated["target"] = torch.stack([b["target"] for b in batch])

    return collated


class DistanceEstimatorModel(BaseModel):
    def __init__(
        self,
        estimator_cls,  # ← a class, e.g. DistanceEstimator or ReachabilityClassifier
        *estimator_args,
        **estimator_kwargs,
    ):
        # instantiate whatever class you passed in:
        self.estimator_cls = estimator_cls(*estimator_args, **estimator_kwargs)

        # now call your base initializer
        super().__init__(model=self.estimator_cls, optimizer_kwargs={"lr": 1e-3})
        self.criterion = nn.MSELoss()

    def _compute_loss(self, batch):
        preds = self.model(batch)  # [B]
        targets = batch["target"].view(-1)  # [B]
        return self.criterion(preds, targets)

    def evaluate(
        self, loader: torch.utils.data.DataLoader, verbose: bool = False, **kwargs
    ) -> dict:
        """
        Basic regression evaluation over all samples.
        Returns:
            - val_loss: mean squared error over entire set
            - mse: same as val_loss
            - rmse: square root of mse
            - mae: mean absolute error
            - r2: R² score
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        th = kwargs.get("th", 0.1)

        c, tot = 0, 0
        if verbose:
            print("\n Errors:")

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(batch).view(-1).cpu().tolist()
                targets = batch["target"].view(-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets)

                if verbose:
                    for i, pred in enumerate(preds):
                        if not (pred - th < targets[i] < pred + th):
                            print(f"{c}) pred:{pred} | target:{targets[i]}")
                            c += 1
                        tot += 1
        if verbose:
            print(f"#errors: {c}/{tot} - {(c / tot) * 100:.2f} %")

        mse = mean_squared_error(all_targets, all_preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        return {
            "val_loss": 1 - r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def _save_full_checkpoint(self, path, **metrics):
        """
        Save model + config + metrics into one file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = self.model.get_checkpoint()
        ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def load_model(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location=self.device)
        self.model = self.estimator_cls.load_model(ckpt)
        self.model.to(self.device)

    def predict_batch(self, batch):
        self.model.eval()
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            preds = self.model(batch)
        return preds.cpu()

    def predict_single(
        self, state_dot: str, depth: int | None = None, goal_dot: str | None = None
    ):
        """
        Wraps the preprocessing and returns a dict with:
          - predicted_distance: float
        """
        # 1) preprocess
        sample = preprocess_sample(
            state_path=state_dot,
            depth=depth,
            target=None,
            goal_path=goal_dot,
        )
        batch = graph_collate_fn([sample])
        # 2) predict
        pred = self.predict_batch(batch).item()
        return pred

    def _get_onnx_wrapper(self):
        return OnnxDistanceEstimatorWrapper

    def try_onnx(
        self,
        onnx_path: str | Path,
        state_dot_files: Sequence[str | Path],
        depths: Sequence[float | int] = None,
        goal_dot_files: Optional[Sequence[str | Path]] = None,
    ) -> np.ndarray:
        """
        Run a **batch** of (state, goal, depth) samples through ONNX Runtime.

        Parameters
        ----------
        state_dot_files : list[str]   – DOT files of *state* graphs
        depths          : list[int]   – raw depth values (same length as batch)
        goal_dot_files  : list[str] | None – optional DOT files of *goal* graphs
        """
        import onnxruntime as ort

        feed = preprocess_for_onnx(state_dot_files, depths, goal_dot_files)
        # inference ------- --------------------------------------------------------
        ort_sess = ort.InferenceSession(str(onnx_path))
        distance = ort_sess.run(["distance"], feed)[0]  # → np.ndarray  shape [B]
        return distance


def preprocess_for_onnx(
    state_dot_files: Sequence[str | Path],
    depths: Sequence[float | int] = None,
    goal_dot_files: Optional[Sequence[str | Path]] = None,
):
    def _parse_dot(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """read DOT → PyG tensors (node-ids, edge_index, edge_attr)"""

        G = _load_dot(path)
        for n, d in G.nodes(data=True):
            d["shape"] = {"circle": 0, "doublecircle": 1}.get(
                d.get("shape", "circle"), 0
            )
        for u, v, d in G.edges(data=True):
            d["edge_label"] = int(str(d.get("label", "0")).strip('"'))

        data = from_networkx(G)
        node_ids = torch.tensor([int(x) for x in G.nodes()], dtype=torch.float32)
        return node_ids, data.edge_index.long(), data.edge_label.view(-1, 1).float()

    # ------------- build a *concatenated* big graph --------------------
    #
    #    Each sample keeps its own graph-id via the *_batch* arrays;
    #    edge-indices are offset so that node indexing stays correct.
    #
    s_nodes, s_edges, s_attrs, s_batch = [], [], [], []
    g_nodes, g_edges, g_attrs, g_batch = [], [], [], []

    cum_state = 0
    cum_goal = 0
    # print(state_dot_files)
    for g_idx, s_file in enumerate(state_dot_files):
        n_id, e_idx, e_attr = _parse_dot(Path(s_file))
        s_nodes.append(n_id)
        s_edges.append(e_idx + cum_state)  # offset
        s_attrs.append(e_attr)
        s_batch.append(torch.full((n_id.size(0),), g_idx, dtype=torch.int64))
        """print("n_id: ", n_id)
        print("e_idx: ", e_idx)
        print("e_attr: ", e_attr)
        print("g_idx: ", g_idx)
        print(torch.full((n_id.size(0),), g_idx, dtype=torch.int64))"""
        cum_state += n_id.size(0)

    # print(f"\n {s_batch}")

    if goal_dot_files is not None:
        for g_idx, g_file in enumerate(goal_dot_files):
            n_id, e_idx, e_attr = _parse_dot(Path(g_file))
            g_nodes.append(n_id)
            g_edges.append(e_idx + cum_goal)  # offset
            g_attrs.append(e_attr)
            g_batch.append(torch.full((n_id.size(0),), g_idx, dtype=torch.int64))
            cum_goal += n_id.size(0)

    # concatenate everything -------------------------------------------------
    state_node_names = torch.cat(s_nodes)  # [Ns]
    state_edge_index = torch.cat(s_edges, dim=1)  # [2,Es]
    state_edge_attr = torch.cat(s_attrs)  # [Es,1]
    state_batch = torch.cat(s_batch)  # [Ns]

    """print(f"state_node_names: ", state_node_names.shape)
    print(f"state_edge_index: ", state_edge_index.shape)
    print(f"state_edge_attr: ", state_edge_attr.shape)
    print(f"state_batch: ", state_batch.shape)"""

    # ----- pack everything that is always present -----
    feed = {
        "state_node_names": state_node_names.numpy(),
        "state_edge_index": state_edge_index.numpy(),
        "state_edge_attr": state_edge_attr.numpy(),
        "state_batch": state_batch.numpy(),
    }

    # ----- depth is optional -------------------------------------------
    if depths is not None:
        if len(depths) != len(state_dot_files):
            raise ValueError(
                f"len(depths)={len(depths)} must equal #graphs={len(state_dot_files)}"
            )
        depth_tensor = torch.as_tensor(depths, dtype=torch.float32)
        feed["depth"] = depth_tensor.numpy()
    else:
        # Either leave it out (common when the downstream ONNX graph
        # has an optional input) or put a default:
        # feed["depth"] = np.zeros(len(state_dot_files), dtype=np.float32)
        pass

    if goal_dot_files is not None:
        goal_node_names = torch.cat(g_nodes)  # [Ng]
        goal_edge_index = torch.cat(g_edges, dim=1)  # [2,Eg]
        goal_edge_attr = torch.cat(g_attrs)  # [Eg,1]
        goal_batch = torch.cat(g_batch)  # [Ng]

        feed["goal_node_names"] = goal_node_names.numpy()
        feed["goal_edge_index"] = goal_edge_index.numpy()
        feed["goal_edge_attr"] = goal_edge_attr.numpy()
        feed["goal_batch"] = goal_batch.numpy()

    return feed


def select_model(
    model_name: str = "distance_estimator",
    use_goal: bool = True,
    use_depth: bool = True,
):

    if model_name == "distance_estimator":
        model = DistanceEstimatorModel(
            DistanceEstimator, use_goal=use_goal, use_depth=use_depth
        )
        return model
    else:
        raise NotImplementedError


def print_values(samples):
    d = {}
    for s in samples:
        ss = s["target"].item()
        if ss in d.keys():
            d[ss] += 1
        else:
            d[ss] = 1

    x = ""
    z = 0
    for target in sorted(d):
        x += f"| Target {target}: {d[target]}"
        z += d[target]
    print(x, " -- Total samples: ", z)


def f(value, slope, min_value_nn, if_forward: bool = True):
    if if_forward:
        return value * slope + min_value_nn
    else:
        return (value - min_value_nn) / slope


def prepare_samples(
    t_s_copy: List[Dict], t_t_copy: List[Dict], unreachable_state_value
):

    t_s_copy = [s for s in t_s_copy if s["target"].item() != unreachable_state_value]
    t_t_copy = [s for s in t_t_copy if s["target"].item() != unreachable_state_value]

    """def find_max(sss):
        max_v = -1
        for s in sss:
            v = s["target"].item()
            if v > max_v and v != UNREACHABLE_STATE_VALUE:
                max_v = v
        return max_v

    max_train = find_max(t_s_copy)
    max_test = find_max(t_t_copy)

    max_tot = max(max_train, max_test)"""

    MIN_DEPTH = 0
    MAX_DEPTH = 50  # if max_tot * 2 > 50 else max_tot * 2

    MIN_V_NN = 1e-3
    MAX_V_NN = 1 - MIN_V_NN

    slope = (MAX_V_NN - MIN_V_NN) / (MAX_DEPTH - MIN_DEPTH)

    params = {"slope": slope, "intercept": MIN_V_NN}

    for s in t_s_copy:
        v = s["target"].item()
        if v != unreachable_state_value:
            s["target"] = torch.tensor(f(v, slope, MIN_V_NN), dtype=torch.float)
        else:
            s["target"] = torch.tensor(f(MAX_DEPTH, slope, MIN_V_NN), dtype=torch.float)

    for s in t_t_copy:
        v = s["target"].item()
        if v != unreachable_state_value:
            s["target"] = torch.tensor(f(v, slope, MIN_V_NN), dtype=torch.float)
        else:
            s["target"] = torch.tensor(f(MAX_DEPTH, slope, MIN_V_NN), dtype=torch.float)

    return t_s_copy, t_t_copy, params
