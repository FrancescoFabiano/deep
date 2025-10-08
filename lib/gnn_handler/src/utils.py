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
    OnnxDistanceEstimatorWrapperBits,
    OnnxDistanceEstimatorWrapperIds,
)

KEYWORD_MAPPED = "MAPPED"
KEYWORD_HASHED = "HASHED"
KEYWORD_BITMASK = "BITMASK"


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


def _nx_to_pyg(
    G: nx.DiGraph, plot: bool = False, diagnose: bool = False, bitmask: bool = False
) -> Data:
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

    if bitmask:
        # Preserve the same node order that from_networkx() used
        nodes = list(G.nodes())

        # Convert each node label into a list of bits
        def to_bits(n: object, bit_len: int | None) -> list[int]:
            if isinstance(n, str):
                s = n.strip()
                if not set(s) <= {"0", "1"}:
                    raise ValueError(f"Node '{n}' is not a 0/1 bitstring.")
                if bit_len is not None and len(s) != bit_len:
                    raise ValueError(
                        f"Inconsistent bit length for '{n}': {len(s)} vs {bit_len}."
                    )
                return [int(ch) for ch in s]
            elif isinstance(n, (list, tuple)):
                bits = [int(b) for b in n]
                if not set(bits) <= {0, 1}:
                    raise ValueError(f"Node '{n}' contains non-binary values.")
                if bit_len is not None and len(bits) != bit_len:
                    raise ValueError(f"Inconsistent bit length for '{n}'.")
                return bits
            elif isinstance(n, int):
                # If nodes are integers, require a fixed length via G.graph['bit_len']
                if bit_len is None:
                    raise ValueError(
                        "bit_len must be provided in G.graph['bit_len'] for int node labels."
                    )
                return [int(ch) for ch in format(n, f"0{bit_len}b")]
            else:
                raise TypeError(f"Unsupported node label type: {type(n)}")

        # Infer fixed length: prefer explicit G.graph['bit_len']; otherwise from first string/sequence
        explicit_len = G.graph.get("bit_len", None)
        first = nodes[0]
        inferred_len = len(first) if isinstance(first, (str, list, tuple)) else None
        BIT_LEN = explicit_len if explicit_len is not None else inferred_len
        if BIT_LEN is None:
            raise ValueError(
                "Cannot infer bit length. Set G.graph['bit_len'] or use string/sequence bit labels."
            )

        bit_rows = [
            to_bits(n, BIT_LEN) for n in nodes
        ]  # List[List[int]] shape [N, BIT_LEN]

        # Store as compact boolean tensor [N, BIT_LEN]
        data.node_bits = torch.tensor(bit_rows, dtype=torch.bool)

        # (Optional) also keep an integer hash/id for convenience: [N]
        weights = 2 ** torch.arange(BIT_LEN - 1, -1, -1)  # msb…lsb
        data.node_bitint = (data.node_bits.to(torch.int64) * weights).sum(dim=1)

        # (Optional) keep the original labels for reference (as a python list)
        data.node_labels = nodes

        # (Optional) if you still want a float tensor like the non-bitmask branch:
        data.node_names = data.node_bitint.to(torch.float32)

    else:
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
    bitmask: bool = False,
    if_plot_graph: bool = False,
    if_diagnose: bool = False,
) -> Dict[str, Any]:
    Gs = _load_dot(Path(state_path))
    if if_plot_graph:
        plot_graph(Gs)
    # print("Gs.nodes = ", Gs.nodes())
    ds = _nx_to_pyg(Gs, bitmask=bitmask)
    assert ds.edge_index.size(1) == ds.edge_attr.size(
        0
    ), f"Mismatch: {ds.edge_index.size(1)} edges vs {ds.edge_attr.size(0)} attrs"

    if if_diagnose:
        diagnose_data(ds)
    ds.name = Path(state_path).stem

    sample = {"state_graph": ds}
    # print("ds: ", ds)
    if depth is not None:
        sample["depth"] = torch.tensor([depth])

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float)

    if goal_path is not None:
        Gg = _load_dot(Path(goal_path))
        if if_plot_graph:
            plot_graph(Gg)
        dg = _nx_to_pyg(Gg, bitmask=bitmask)
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
        self,
        state_dot: str,
        depth: int | None = None,
        goal_dot: str | None = None,
        bitmask: bool = False,
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
            bitmask=bitmask,
        )
        batch = graph_collate_fn([sample])
        # 2) predict
        pred = self.predict_batch(batch).item()
        return pred

    def _get_onnx_wrapper(self):
        core = self.model
        return (
            OnnxDistanceEstimatorWrapperBits
            if getattr(core, "bit_input", None) is not None
            else OnnxDistanceEstimatorWrapperIds
        )

    def try_onnx(
        self,
        onnx_path: str | Path,
        state_dot_files: Sequence[str | Path],
        depths: Sequence[float | int] = None,
        goal_dot_files: Optional[Sequence[str | Path]] = None,
        bitmask: bool = False,
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

        feed = preprocess_for_onnx(
            state_dot_files, depths, goal_dot_files, bitmask=bitmask
        )
        # inference ------- --------------------------------------------------------
        ort_sess = ort.InferenceSession(str(onnx_path))
        distance = ort_sess.run(["distance"], feed)[0]  # → np.ndarray  shape [B]
        return distance

    def to_onnx(
        self, onnx_path: str | Path, with_goal: bool = False, with_depth: bool = False
    ) -> None:
        Wrapper = self._get_onnx_wrapper()

        onnx_path = Path(onnx_path)

        Ns, Es = 3, 4
        Ng, Eg = 2, 3  # dummy goal sizes
        input_names, dummy_inputs, dynamic_axes = [], [], {}

        if getattr(self.model, "bit_input", None) is None:
            # IDs path — first input: float32 [Ns]
            input_names += [
                "state_node_ids",
                "state_edge_index",
                "state_edge_attr",
                "state_batch",
            ]
            dummy_inputs += [
                torch.arange(Ns, dtype=torch.float32),  # [Ns]
                torch.zeros((2, Es), dtype=torch.int64),  # [2, Es]
                torch.zeros((Es, 1), dtype=torch.float32),  # [Es, 1]
                torch.zeros(Ns, dtype=torch.int64),  # [Ns]
            ]
            dynamic_axes.update(
                {
                    "state_node_ids": {0: "Ns"},
                    "state_edge_index": {1: "Es"},
                    "state_edge_attr": {0: "Es"},
                    "state_batch": {0: "Ns"},
                }
            )
            if with_goal:
                input_names += [
                    "goal_node_ids",
                    "goal_edge_index",
                    "goal_edge_attr",
                    "goal_batch",
                ]
                dummy_inputs += [
                    torch.arange(Ng, dtype=torch.float32),  # [Ng]
                    torch.zeros((2, Eg), dtype=torch.int64),  # [2, Eg]
                    torch.zeros((Eg, 1), dtype=torch.float32),  # [Eg, 1]
                    torch.zeros(Ng, dtype=torch.int64),  # [Ng]
                ]
                dynamic_axes.update(
                    {
                        "goal_node_ids": {0: "Ng"},
                        "goal_edge_index": {1: "Eg"},
                        "goal_edge_attr": {0: "Eg"},
                        "goal_batch": {0: "Ng"},
                    }
                )
        else:
            # Bitmask path — first input: uint8 [Ns, bit_len]
            bit_len = int(self.model.bit_input)  # must match C++ m_bitmask_size
            input_names += [
                "state_node_bits",
                "state_edge_index",
                "state_edge_attr",
                "state_batch",
            ]
            dummy_inputs += [
                torch.zeros((Ns, bit_len), dtype=torch.uint8),  # [Ns, bit_len]
                torch.zeros((2, Es), dtype=torch.int64),  # [2, Es]
                torch.zeros((Es, 1), dtype=torch.float32),  # [Es, 1]
                torch.zeros(Ns, dtype=torch.int64),  # [Ns]
            ]
            dynamic_axes.update(
                {
                    "state_node_bits": {0: "Ns"},  # feature dim static
                    "state_edge_index": {1: "Es"},
                    "state_edge_attr": {0: "Es"},
                    "state_batch": {0: "Ns"},
                }
            )
            if with_goal:
                input_names += [
                    "goal_node_bits",
                    "goal_edge_index",
                    "goal_edge_attr",
                    "goal_batch",
                ]
                dummy_inputs += [
                    torch.zeros((Ng, bit_len), dtype=torch.uint8),  # [Ng, bit_len]
                    torch.zeros((2, Eg), dtype=torch.int64),  # [2, Eg]
                    torch.zeros((Eg, 1), dtype=torch.float32),  # [Eg, 1]
                    torch.zeros(Ng, dtype=torch.int64),  # [Ng]
                ]
                dynamic_axes.update(
                    {
                        "goal_node_bits": {0: "Ng"},
                        "goal_edge_index": {1: "Eg"},
                        "goal_edge_attr": {0: "Eg"},
                        "goal_batch": {0: "Ng"},
                    }
                )

        # depth (orthogonal to goal)
        if with_depth:
            input_names.append("depth")
            dummy_inputs.append(torch.zeros(1, dtype=torch.float32))  # B=1 for tracing
            dynamic_axes["depth"] = {
                0: "B"
            }  # aligns with number of graphs in the batch

        dynamic_axes["distance"] = {0: "B"}

        wrapper = Wrapper(self.model).eval()
        torch.onnx.export(
            wrapper.cpu(),
            tuple(dummy_inputs),
            onnx_path.as_posix(),
            opset_version=18,
            input_names=input_names,
            output_names=["distance"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
        )


def preprocess_for_onnx(
    state_dot_files: Sequence[str | Path],
    depths: Sequence[float | int] | None = None,
    goal_dot_files: Optional[Sequence[str | Path]] = None,
    *,
    bitmask: bool = True,
    bit_len: int = 64,  # must match ONNX input feature dim (your model prints [-1, 64])
) -> Dict[str, np.ndarray]:
    """
    Build ONNX feed dict for a batch of graphs.
    - Bitmask=True -> produces uint8 node feature matrices with keys: state_node_bits / goal_node_bits
    - Bitmask=False -> produces float32 node id vectors with keys: state_node_ids / goal_node_ids

    Returns numpy arrays with exact dtypes/shapes expected by your ONNX:
      state_node_bits: uint8  [Ns, bit_len]   (or state_node_ids: float32 [Ns])
      state_edge_index: int64 [2, Es]
      state_edge_attr: float32 [Es, 1]
      state_batch: int64      [Ns]
      (optional) goal_* equivalents
      (optional) depth: float32 [B], where B == len(state_dot_files)
    """

    def _parse_dot_bits(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """read DOT -> (node_bits_u8 [N,bit_len], edge_index [2,E], edge_attr [E,1])"""
        G = _load_dot(path)

        # normalize attrs
        for _, d in G.nodes(data=True):
            d["shape"] = {"circle": 0, "doublecircle": 1}.get(
                d.get("shape", "circle"), 0
            )
        for _, _, d in G.edges(data=True):
            d["edge_label"] = int(str(d.get("label", "0")).strip('"'))

        # edge tensors from PyG
        data = from_networkx(G)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_label.view(-1, 1).float()

        # nodes: keep order consistent with networkx iteration
        nodes = list(G.nodes())
        # convert node labels to bit rows
        rows = []
        for n in nodes:
            if isinstance(n, str):
                s = n.strip()
                if not set(s) <= {"0", "1"}:
                    raise ValueError(f"Node '{n}' is not a 0/1 bitstring.")
                if len(s) != bit_len:
                    raise ValueError(
                        f"Bit length mismatch for '{n}': got {len(s)}, expected {bit_len}."
                    )
                rows.append([1 if c == "1" else 0 for c in s])
            elif isinstance(n, (list, tuple)):
                bits = [int(b) for b in n]
                if not set(bits) <= {0, 1}:
                    raise ValueError(f"Node '{n}' contains non-binary values.")
                if len(bits) != bit_len:
                    raise ValueError(
                        f"Bit length mismatch for '{n}': got {len(bits)}, expected {bit_len}."
                    )
                rows.append(bits)
            else:
                raise TypeError(
                    f"Unsupported node label type for bitmask mode: {type(n)}"
                )
        node_bits = torch.tensor(rows, dtype=torch.uint8)  # ONNX expects uint8

        return node_bits, edge_index, edge_attr

    def _parse_dot_ids(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """read DOT -> (node_ids_f32 [N], edge_index [2,E], edge_attr [E,1])"""
        G = _load_dot(path)
        for _, d in G.nodes(data=True):
            d["shape"] = {"circle": 0, "doublecircle": 1}.get(
                d.get("shape", "circle"), 0
            )
        for _, _, d in G.edges(data=True):
            d["edge_label"] = int(str(d.get("label", "0")).strip('"'))

        data = from_networkx(G)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_label.view(-1, 1).float()

        # node ids as float32 vector
        node_ids = torch.tensor([float(int(x)) for x in G.nodes()], dtype=torch.float32)
        return node_ids, edge_index, edge_attr

    # collectors for concatenation
    s_nodes, s_edges, s_attrs, s_batch = [], [], [], []
    g_nodes, g_edges, g_attrs, g_batch = [], [], [], []

    cum_state = 0
    for g_idx, s_file in enumerate(state_dot_files):
        if bitmask:
            n_feat, e_idx, e_attr = _parse_dot_bits(Path(s_file))
            # offset edges by current node count
            if e_idx.numel() > 0:
                e_idx = e_idx + cum_state
        else:
            n_feat, e_idx, e_attr = _parse_dot_ids(Path(s_file))
            if e_idx.numel() > 0:
                e_idx = e_idx + cum_state

        s_nodes.append(n_feat)
        s_edges.append(e_idx)
        s_attrs.append(e_attr)
        s_batch.append(torch.full((n_feat.size(0),), g_idx, dtype=torch.int64))
        cum_state += n_feat.size(0)

    if goal_dot_files is not None:
        cum_goal = 0
        for g_idx, g_file in enumerate(goal_dot_files):
            if bitmask:
                n_feat, e_idx, e_attr = _parse_dot_bits(Path(g_file))
                if e_idx.numel() > 0:
                    e_idx = e_idx + cum_goal
            else:
                n_feat, e_idx, e_attr = _parse_dot_ids(Path(g_file))
                if e_idx.numel() > 0:
                    e_idx = e_idx + cum_goal

            g_nodes.append(n_feat)
            g_edges.append(e_idx)
            g_attrs.append(e_attr)
            g_batch.append(torch.full((n_feat.size(0),), g_idx, dtype=torch.int64))
            cum_goal += n_feat.size(0)

    # concat state
    if bitmask:
        state_node_bits = (
            torch.cat(s_nodes, dim=0)
            if s_nodes
            else torch.zeros((0, bit_len), dtype=torch.uint8)
        )
    else:
        state_node_ids = (
            torch.cat(s_nodes) if s_nodes else torch.zeros((0,), dtype=torch.float32)
        )

    state_edge_index = (
        torch.cat(s_edges, dim=1) if s_edges else torch.zeros((2, 0), dtype=torch.int64)
    )
    state_edge_attr = (
        torch.cat(s_attrs, dim=0)
        if s_attrs
        else torch.zeros((0, 1), dtype=torch.float32)
    )
    state_batch = (
        torch.cat(s_batch, dim=0) if s_batch else torch.zeros((0,), dtype=torch.int64)
    )

    # build feed dict with exact names/dtypes the ONNX expects
    feed: Dict[str, np.ndarray] = {}
    if bitmask:
        feed["state_node_bits"] = state_node_bits.numpy()  # uint8 [Ns, bit_len]
    else:
        feed["state_node_ids"] = state_node_ids.numpy()  # float32 [Ns]
    feed["state_edge_index"] = state_edge_index.numpy()  # int64 [2, Es]
    feed["state_edge_attr"] = state_edge_attr.numpy()  # float32 [Es, 1]
    feed["state_batch"] = state_batch.numpy()  # int64 [Ns]

    # optional depth: float32 [B] where B = number of *state* graphs
    if depths is not None:
        if len(depths) != len(state_dot_files):
            raise ValueError(
                f"len(depths)={len(depths)} must equal #graphs={len(state_dot_files)}"
            )
        depth_tensor = torch.as_tensor(depths, dtype=torch.float32)
        feed["depth"] = depth_tensor.numpy()

    # optional goal
    if goal_dot_files is not None:
        if bitmask:
            goal_node_bits = (
                torch.cat(g_nodes, dim=0)
                if g_nodes
                else torch.zeros((0, bit_len), dtype=torch.uint8)
            )
            feed["goal_node_bits"] = goal_node_bits.numpy()  # uint8 [Ng, bit_len]
        else:
            goal_node_ids = (
                torch.cat(g_nodes)
                if g_nodes
                else torch.zeros((0,), dtype=torch.float32)
            )
            feed["goal_node_ids"] = goal_node_ids.numpy()  # float32 [Ng]

        goal_edge_index = (
            torch.cat(g_edges, dim=1)
            if g_edges
            else torch.zeros((2, 0), dtype=torch.int64)
        )
        goal_edge_attr = (
            torch.cat(g_attrs, dim=0)
            if g_attrs
            else torch.zeros((0, 1), dtype=torch.float32)
        )
        goal_batch = (
            torch.cat(g_batch, dim=0)
            if g_batch
            else torch.zeros((0,), dtype=torch.int64)
        )

        feed["goal_edge_index"] = goal_edge_index.numpy()  # int64 [2, Eg]
        feed["goal_edge_attr"] = goal_edge_attr.numpy()  # float32 [Eg, 1]
        feed["goal_batch"] = goal_batch.numpy()  # int64 [Ng]

        # sanity: if bitmask, state/goal bit widths must match ONNX feature dim
        if bitmask and state_node_bits.size(1) != bit_len:
            raise ValueError("State bit width != bit_len")
        if bitmask and g_nodes and goal_node_bits.size(1) != bit_len:
            raise ValueError("Goal bit width != bit_len")

    # extra sanity checks to avoid Gather OOB:
    Ns = state_node_bits.shape[0] if bitmask else state_node_ids.shape[0]
    if state_edge_index.numel() > 0:
        max_idx = int(state_edge_index.max().item())
        if max_idx >= Ns:
            raise ValueError(f"state_edge_index has node id {max_idx} >= Ns={Ns}")

    if (
        goal_dot_files is not None
        and (goal_edge_index := torch.from_numpy(feed["goal_edge_index"])).numel() > 0
    ):
        Ng = (
            feed["goal_node_bits"].shape[0]
            if bitmask
            else feed["goal_node_ids"].shape[0]
        )
        max_idx_g = int(goal_edge_index.max().item())
        if max_idx_g >= Ng:
            raise ValueError(f"goal_edge_index has node id {max_idx_g} >= Ng={Ng}")

    return feed


def select_model(
    model_name: str = "distance_estimator",
    use_goal: bool = True,
    use_depth: bool = True,
    bitmask: bool = False,
):

    if model_name == "distance_estimator":
        model = DistanceEstimatorModel(
            DistanceEstimator,
            use_goal=use_goal,
            use_depth=use_depth,
            bit_input=40 if bitmask else None,
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
