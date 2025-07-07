from __future__ import annotations

import math
import os
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pydot
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import from_networkx
from tqdm import tqdm

N_EPOCHS_DEFAULT = 200


TWO_48_MINUS_1 = float(2**48 - 1)


class DistanceEstimator(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            use_goal: bool = True,
            use_depth: bool = True,  # ← NEW
            node_emb_dim: int = 32,
            edge_emb_dim: int = 32,
    ):
        super().__init__()
        self.use_goal = use_goal
        self.use_depth = use_depth  # ← NEW

        # ────────────────────────────────────────────────────────── embeddings
        self.id_mlp = nn.Sequential(
            nn.Linear(1, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # ──────────────────────────────────────────────────────── GINE (state)
        self.state_conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(node_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_emb_dim,
        )
        self.state_conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_emb_dim,
        )

        # ───────────────────────────────────────────────────────── GINE (goal)
        if use_goal:
            self.goal_conv1 = GINEConv(
                nn.Sequential(
                    nn.Linear(node_emb_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_emb_dim,
            )
            self.goal_conv2 = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_emb_dim,
            )

        # ───────────────────────────────────────────────────────── regressor
        in_dim = hidden_dim * (2 if use_goal else 1) + (1 if use_depth else 0)
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # --------------------------------------------------------------- encode
    def _encode(self, graph, conv1, conv2):
        raw = graph.node_names.to(graph.edge_index.device).float()
        x = self.id_mlp((raw / TWO_48_MINUS_1).clamp(0.0, 1.0).view(-1, 1))

        e = self.edge_mlp(graph.edge_attr.to(x.device).float())
        x = F.relu(conv1(x, graph.edge_index, e))
        x = F.relu(conv2(x, graph.edge_index, e))
        return global_mean_pool(x, graph.batch.clone())

    # ---------------------------------------------------------------- forward
    def forward(self, batch_dict):
        s = self._encode(batch_dict["state_graph"], self.state_conv1, self.state_conv2)

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            g = self._encode(batch_dict["goal_graph"], self.goal_conv1, self.goal_conv2)
            rep = torch.cat([s, g], dim=1)
        else:
            rep = s

        # depth (optional)
        if self.use_depth:
            d = batch_dict.get("depth")
            if d is None:
                # allow inference without depth – pad with zeros
                d = torch.zeros(len(rep), 1, dtype=rep.dtype, device=rep.device)
            else:
                d = d.float().view(-1, 1)
            z = torch.cat([rep, d], dim=1)
        else:
            z = rep

        return self.regressor(z).squeeze(1)

    # -------------------------------------------------------------- checkpoint
    def get_checkpoint(self) -> Dict:
        cfg = {
            "hidden_dim": self.state_conv1.nn[0].out_features,
            "node_emb_dim": self.id_mlp[0].out_features,
            "edge_emb_dim": self.edge_mlp[0].out_features,
            "use_goal": self.use_goal,
            "use_depth": self.use_depth,  # ← NEW
        }
        return {"state_dict": self.state_dict(), "config": cfg}

    @classmethod
    def load_model(cls, ckpt: Dict):
        cfg = ckpt["config"]
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            use_goal=cfg["use_goal"],
            use_depth=cfg.get("use_depth", True),  # ← NEW
            node_emb_dim=cfg["node_emb_dim"],
            edge_emb_dim=cfg["edge_emb_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model



class OnnxDistanceEstimatorWrapper(nn.Module):
    def __init__(self, core: "DistanceEstimator"):
        super().__init__()
        self.core = core

    # ------------------------------ util
    @staticmethod
    def _global_mean(x, batch):
        B = batch.max() + 1
        sums = torch.zeros(B, x.size(1), dtype=x.dtype, device=x.device)
        sums = sums.index_add(0, batch, x)
        cnts = torch.zeros_like(sums[:, :1])
        cnts = cnts.index_add(
            0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(1)
        )
        return sums / cnts.clamp(min=1.0)

    # ------------------------------ encoder
    def _encode_raw(self, node_ids, edge_index, edge_attr, batch, conv1, conv2):
        x = self.core.id_mlp(
            torch.clamp((node_ids.float() + 2.0) / TWO_48_MINUS_1, 0.0, 1.0).unsqueeze(
                1
            )
        )
        e = self.core.edge_mlp(edge_attr.float())
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))
        return self._global_mean(x, batch)

    # ------------------------------ forward
    def forward(
            self,
            s_node_ids,
            s_edge_index,
            s_edge_attr,
            s_batch,
            depth,
            g_node_ids,
            g_edge_index,
            g_edge_attr,
            g_batch,
    ):
        s_emb = self._encode_raw(
            s_node_ids,
            s_edge_index,
            s_edge_attr,
            s_batch,
            self.core.state_conv1,
            self.core.state_conv2,
        )

        if self.core.use_goal and g_node_ids.numel() > 0:
            g_emb = self._encode_raw(
                g_node_ids,
                g_edge_index,
                g_edge_attr,
                g_batch,
                self.core.goal_conv1,
                self.core.goal_conv2,
            )
            rep = torch.cat([s_emb, g_emb], dim=1)
        else:
            rep = s_emb

        # depth handling identical to core
        if self.core.use_depth:
            if depth is None or depth.numel() == 0:
                depth = torch.zeros(len(rep), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            z = torch.cat([rep, depth], dim=1)
        else:
            z = rep

        return self.core.regressor(z).squeeze(1)


class BaseModel(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device | str = None,
            optimizer_cls: type = torch.optim.Adam,
            optimizer_kwargs: dict = None,
    ):
        """
        Args:
            model: your nn.Module
            device: 'cuda' / 'cpu' or torch.device.  If None, auto‐selects.
            optimizer_cls: optimizer class (default Adam)
            optimizer_kwargs: dict of kwargs to pass to optimizer (e.g. {'lr':1e-3})
        """
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            n_epochs: int = N_EPOCHS_DEFAULT,
            checkpoint_dir: str = ".",
            **kwargs,
    ) -> None:
        """
        Generic training loop.  Subclasses must implement:
          - self._compute_loss(batch)
          - self.evaluate(loader, **eval_kwargs) → metrics_dict
          - self._save_full_checkpoint(path, **ckpt_kwargs)

        Args:
            train_loader, val_loader: usual DataLoaders
            n_epochs: total epochs
            checkpoint_dir: where to save best model
            **kwargs: passed through to evaluate()
        """
        best_metric = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)
        pbar = tqdm(range(n_epochs), desc="training...")

        history: dict[str, list[float]] = defaultdict(list)

        for _ in pbar:
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader, **kwargs)

            pbar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                **{k: f"{v:.4f}" for k, v in val_metrics.items()},
            )

            if val_metrics["val_loss"] < best_metric:
                best_path = f"{checkpoint_dir}/best.pt"
                self._save_full_checkpoint(best_path)

                best_metric = val_metrics["val_loss"]

            # ←–– dynamically record *all* metrics returned
            for name, value in val_metrics.items():
                history[name].append(value)

        self.save_and_plot_metrics(history, checkpoint_dir, n_epochs)

    def _move_batch_to_device(self, batch: dict) -> dict:
        for k, v in batch.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            else:  # assume PyG Batch
                batch[k] = v.to(self.device)
        return batch

    @abstractmethod
    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Given a batch, compute and return a scalar loss Tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
            self,
            loader: torch.utils.data.DataLoader,
            **kwargs,
    ) -> dict:
        """
        Run the model in eval mode over `loader` and return a dict of metrics,
        e.g. {'val_loss': ..., 'rmse': ..., 'accuracy': ...}
        """
        raise NotImplementedError

    @abstractmethod
    def _save_full_checkpoint(self, path: str, **metrics) -> None:
        """
        Save model state_dict and any config needed, into `path`.
        You get access to all val‐metrics as well.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path_ckpt: str) -> None:
        """
        Load weights from `path_ckpt` into self.model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, batch: dict) -> torch.Tensor:
        """
        Run forward on a batch, return raw outputs or processed predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_single(self, *args, **kwargs):
        """
        Predict on a single example (e.g. file paths or tensors),
        """
        raise NotImplementedError

    @staticmethod
    def save_and_plot_metrics(history, checkpoint_dir, n_epochs):

        # — after training: plot everything you tracked —
        epochs = range(n_epochs)

        # 1) Loss curves (anything with 'loss' in name)
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/train_loss.png")
        plt.show()

        # 2) All other metrics
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" not in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Metric")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/validation_metrics.png")
        plt.show()

        return history

    def to_onnx(self, onnx_path: str | Path, use_goal: bool = True) -> None:
        """
        Export the *trained* DistanceEstimator to ONNX with **dynamic** sizes:

            • any number of graphs     (batch axis)
            • any number of nodes/edges in each graph

        The exported model therefore runs both single-graph *and* mini-batch
        inference in ONNX Runtime.
        """
        onnx_wrapper = self._get_onnx_wrapper()

        onnx_path = Path(onnx_path)
        self.model.eval()

        # dummy – just to trace the graph; real sizes don’t matter
        N_s, E_s = 3, 4  # state  graph:  3 nodes – 4 edges
        N_g, E_g = 3, 4  # goal   graph:  3 nodes – 4 edges
        B = 2  # two graphs in the mini-batch

        dummy_inputs = (
            torch.arange(N_s, dtype=torch.float32),  # state_node_names
            torch.zeros((2, E_s), dtype=torch.int64),  # state_edge_index
            torch.zeros((E_s, 1), dtype=torch.float32),  # state_edge_attr
            torch.tensor([0, 0, 1], dtype=torch.int64),  # state_batch
            torch.zeros(B, dtype=torch.float32),  # depth  (B × 1)
            torch.arange(N_g, dtype=torch.float32),  # goal_node_names
            torch.zeros((2, E_g), dtype=torch.int64),  # goal_edge_index
            torch.zeros((E_g, 1), dtype=torch.float32),  # goal_edge_attr
            torch.tensor([0, 0, 1], dtype=torch.int64),  # goal_batch
        )

        torch.onnx.export(
            onnx_wrapper(self.model).cpu(),  # << wrapped core
            dummy_inputs,
            onnx_path.as_posix(),
            opset_version=18,  # ScatterND w/ reduction
            input_names=[
                "state_node_names",
                "state_edge_index",
                "state_edge_attr",
                "state_batch",
                "depth",
                "goal_node_names",
                "goal_edge_index",
                "goal_edge_attr",
                "goal_batch",
            ],
            output_names=["distance"],
            dynamic_axes={  # ← make everything var-len
                "state_node_names": {0: "Ns"},
                "state_edge_index": {1: "Es"},
                "state_edge_attr": {0: "Es"},
                "state_batch": {0: "Ns"},
                "depth": {0: "B"},  # B = batch size
                "goal_node_names": {0: "Ng"},
                "goal_edge_index": {1: "Eg"},
                "goal_edge_attr": {0: "Eg"},
                "goal_batch": {0: "Ng"},
                "distance": {0: "B"},
            },
        )
        print("ONNX export complete ✅")

    @abstractmethod
    def _get_onnx_wrapper(self):
        raise NotImplementedError


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


def merge_rare(y, threshold=2):
    # Count labels
    from collections import Counter

    cnt = Counter(y)
    # Any label with fewer than `threshold` occurrences becomes “other”
    return ["other" if cnt[label] < threshold else label for label in y]


def split_samples(samples, test_ratio=0.2, seed=42):
    # Extract the list of labels
    y = [sample["target"] for sample in samples]
    y_merged = merge_rare(y, threshold=2)

    # Now split, stratifying on y
    train, test = train_test_split(
        samples,
        test_size=test_ratio,
        shuffle=True,
        random_state=seed,
        stratify=y_merged,
    )
    return train, test


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_NEG_ID_RE = re.compile(r'(?<![\w"])-\d+(?![\w"])')


def _load_dot(path: Path, quote_neg: bool) -> nx.DiGraph:
    src = path.read_text()
    if quote_neg:
        src = _NEG_ID_RE.sub(lambda m: f'"{m.group(0)}"', src)
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
    fix = "merged" in state_path
    Gs = _load_dot(Path(state_path), fix)
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
        Gg = _load_dot(Path(goal_path), True)
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
    for b in batch:
        print(b["state_graph"])
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

    def evaluate(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict:
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

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(batch).view(-1).cpu().tolist()
                targets = batch["target"].view(-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets)

        # now compute metrics via sklearn
        mse = mean_squared_error(all_targets, all_preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        return {
            "val_loss": mse,  # MSE as the main loss
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
        distance = ort_sess.run(None, feed)[0]  # → np.ndarray  shape [B]
        return distance


def preprocess_for_onnx(
        state_dot_files: Sequence[str | Path],
        depths: Sequence[float | int],
        goal_dot_files: Optional[Sequence[str | Path]] = None,
):
    def _parse_dot(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """read DOT → PyG tensors (node-ids, edge_index, edge_attr)"""

        G = _load_dot(path, True)
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

    # ----- pack everything that is always present -----
    feed = {
        "state_node_names": state_node_names.numpy(),
        "state_edge_index": state_edge_index.numpy(),
        "state_edge_attr": state_edge_attr.numpy(),
        "state_batch": state_batch.numpy(),
    }

    # ----- depth is optional -------------------------------------------
    if depths:
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
