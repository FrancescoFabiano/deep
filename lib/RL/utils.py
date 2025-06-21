from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import pydot
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

N_EPOCHS_DEFAULT = 500


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

    def save_and_plot_metrics(self, history, checkpoint_dir, n_epochs):

        # — after training: plot everything you tracked —
        epochs = range(n_epochs)

        # 1) Loss curves (anything with 'loss' in name)
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.title("Loss Curves")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/loss.png")
        plt.show()

        # 2) All other metrics
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" not in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend(loc="best")
        plt.title("Other Metrics")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/metrics.png")
        plt.show()

        return history

class DistanceEstimator(nn.Module):
    def __init__(
            self,
            in_channels_node: int,
            hidden_dim: int = 64,
            use_goal: bool = True,
            num_relations: int = 2**5,
    ):
        super().__init__()
        self.use_goal = use_goal

        self.state_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
        self.state_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        if use_goal:
            self.goal_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
            self.goal_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * (2 if use_goal else 1) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data_batch, conv1, conv2):
        x = data_batch.x.float()
        edge_index = data_batch.edge_index
        edge_type = data_batch.edge_type
        x = F.relu(conv1(x, edge_index, edge_type))
        x = F.relu(conv2(x, edge_index, edge_type))
        return global_mean_pool(x, data_batch.batch)

    def forward(self, batch_dict):
        state_emb = self._encode(
            batch_dict["state_graph"], self.state_gnn1, self.state_gnn2
        )

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            goal_emb = self._encode(
                batch_dict["goal_graph"], self.goal_gnn1, self.goal_gnn2
            )
            graph_emb = torch.cat([state_emb, goal_emb], dim=1)
        else:
            graph_emb = state_emb

        depth = batch_dict["depth"].float().view(-1, 1)
        depth = (depth - depth.mean()) / (depth.std(unbiased=False) + 1e-6)

        z = torch.cat([graph_emb, depth], dim=1)
        pred = self.regressor(z).squeeze(1)
        return pred

class ReachabilityClassifier(nn.Module):
    def __init__(
            self,
            in_channels_node: int,
            hidden_dim: int = 64,
            use_goal: bool = True,
            num_relations: int = 2**5,
    ):
        super().__init__()
        self.use_goal = use_goal

        self.state_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
        self.state_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        if use_goal:
            self.goal_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
            self.goal_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (2 if use_goal else 1) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data_batch, conv1, conv2):
        x = data_batch.x.float()
        edge_index = data_batch.edge_index
        edge_type = data_batch.edge_type
        x = F.relu(conv1(x, edge_index, edge_type))
        x = F.relu(conv2(x, edge_index, edge_type))
        return global_mean_pool(x, data_batch.batch)

    def forward(self, batch_dict):
        state_emb = self._encode(
            batch_dict["state_graph"], self.state_gnn1, self.state_gnn2
        )

        if self.use_goal and batch_dict["goal_graph"] is not None:
            goal_emb = self._encode(
                batch_dict["goal_graph"], self.goal_gnn1, self.goal_gnn2
            )
            graph_emb = torch.cat([state_emb, goal_emb], dim=1)
        else:
            graph_emb = state_emb

        depth = batch_dict["depth"].float().view(-1, 1)
        depth = (depth - depth.mean()) / (depth.std(unbiased=False) + 1e-6)

        z = torch.cat([graph_emb, depth], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits


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


def split_samples(samples, test_ratio=0.2, seed=42):  # ▶ NEW/CHANGED
    return train_test_split(
        samples, test_size=test_ratio, shuffle=True, random_state=seed, stratify=None
    )


def seed_everything(seed: int = 42):  # ▶ NEW/CHANGED
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
    print("Nodes:", data.num_nodes, data.node_names)
    print("Edges:", data.num_edges, data.edge_index)
    if hasattr(data, "edge_attr"):
        print("Edge attr:", data.edge_attr.squeeze())


def _nx_to_pyg(G: nx.DiGraph, plot: bool = False, diagnose: bool = False):
    for n, data in G.nodes(data=True):
        data["shape"] = {"circle": 0, "doublecircle": 1}.get(
            data.get("shape", "circle"), 0
        )
    for u, v, d in G.edges(data=True):
        d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))

    if plot:
        plot_graph(G)

    data = from_networkx(G)
    data.x = torch.tensor([[G.nodes[n]["shape"]] for n in G.nodes()], dtype=torch.float)
    data.node_names = list(G.nodes())
    data.edge_index = data.edge_index.long()
    data.edge_attr = data.edge_label.view(-1, 1)
    data.edge_type = data.edge_label

    if diagnose:
        diagnose_data(data)

    return data


def preprocess_sample(
        state_path: str,
        depth: int,
        target: int | None = None,
        goal_path: Optional[str] = None,
) -> Dict[str, Any]:
    fix = "merged" in state_path
    Gs = _load_dot(Path(state_path), fix)
    ds = _nx_to_pyg(Gs)
    ds.name = Path(state_path).stem

    sample = {"state_graph": ds, "depth": torch.tensor([depth])}

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float)

    if goal_path is not None:
        Gg = _load_dot(Path(goal_path), True)
        dg = _nx_to_pyg(Gg)
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
        "depth": torch.stack([b["depth"] for b in batch]),
    }

    if "target" in batch[0]:  # ← only in training / evaluation
        collated["target"] = torch.stack([b["target"] for b in batch])

    return collated


class DistanceEstimatorModel(BaseModel):
    def __init__(self, *estimator_args, **estimator_kwargs):
        estimator = DistanceEstimator(*estimator_args, **estimator_kwargs)
        super().__init__(model=estimator, optimizer_kwargs={"lr": 1e-3})
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
        cfg = {
            "in_channels_node": self.model.state_gnn1.in_channels,
            "hidden_dim": self.model.state_gnn1.out_channels,
            "use_goal": self.model.use_goal,
            "num_relations": self.model.state_gnn1.num_relations,
        }
        ckpt = {
            "state_dict": self.model.state_dict(),
            "config": cfg,
            "metrics": metrics,
        }
        torch.save(ckpt, path)

    def load_model(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location=self.device)
        # rebuild model in case it's not already the right shape
        cfg = ckpt.get("config", {})
        self.model = DistanceEstimator(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_batch(self, batch):
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            preds = self.model(batch)
        return preds.cpu()

    def predict_single(self, state_dot: str, depth: int, goal_dot: str | None = None):
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


class ReachabilityClassifierModel(BaseModel):
    def __init__(self, *classifier_args, **classifier_kwargs):
        classifier = ReachabilityClassifier(*classifier_args, **classifier_kwargs)
        super().__init__(model=classifier, optimizer_kwargs={"lr": 1e-3})
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def _compute_loss(self, batch):
        logits = self.model(batch).view(-1)
        targets = batch["target"].view(-1).float()  # assume 0/1
        return self.criterion(logits, targets)

    def evaluate(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """
        Evaluate a binary reachability classifier.
        Assumes batch["target"] is already 0/1 (float or long).
        Returns:
          - val_loss: average BCE loss over all samples
          - accuracy, precision, recall, f1: classification metrics
        """
        self.model.eval()
        loss_accum = 0.0
        n = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                logits = self.model(batch).view(-1)  # raw scores
                targets = batch["target"].view(-1).float()  # 0.0 or 1.0

                # sum‐reduction BCE
                loss_accum += F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="sum"
                ).item()
                n += targets.size(0)

                # compute predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().long().tolist())

        val_loss = loss_accum / n if n > 0 else float("nan")
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        return {
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _save_full_checkpoint(self, path: str, **metrics):
        """
        Save model + config + metrics into one file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # pull config from the classifier architecture
        cfg = {
            "in_channels_node": self.model.state_gnn1.in_channels,
            "hidden_dim": self.model.state_gnn1.out_channels,
            "use_goal": self.model.use_goal,
            "num_relations": self.model.state_gnn1.num_relations,
        }
        ckpt = {
            "state_dict": self.model.state_dict(),
            "config": cfg,
            "metrics": metrics,
        }
        torch.save(ckpt, path)

    def load_model(self, path_ckpt: str):
        ckpt = torch.load(path_ckpt, map_location=self.device)
        cfg = ckpt.get("config", {})
        # rebuild classifier with saved config
        self.model = ReachabilityClassifier(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_batch(self, batch: dict) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            logits = self.model(batch).view(-1)
            return torch.sigmoid(logits).cpu()  # returns probabilities

    def predict_single(
            self,
            state_dot: str,
            depth: int,
            goal_dot: str | None = None,
            threshold: float = 0.5,
    ) -> dict:
        """
        Preprocess one sample and return:
          - prob_reachable: float
          - reachable: bool
        """
        sample = preprocess_sample(
            state_path=state_dot, depth=depth, target=None, goal_path=goal_dot
        )
        batch = graph_collate_fn([sample])
        prob = self.predict_batch(batch).item()
        return {"prob_reachable": prob, "reachable": prob >= threshold}


def select_model(model_name: str = "distance_estimator", use_goal: bool = True):

    if model_name == "distance_estimator":
        mm = DistanceEstimatorModel(
            in_channels_node=1,
            hidden_dim=64,
            use_goal=use_goal,
            num_relations=64,
        )
        return mm

    elif model_name == "reachability_classifier":
        mm = ReachabilityClassifierModel(
            in_channels_node=1,
            hidden_dim=64,
            use_goal=use_goal,
            num_relations=64,
        )
        return mm
    else:
        raise NotImplementedError
