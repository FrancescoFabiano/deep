from __future__ import annotations

import math
import os
import random
import re

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pydot
import torch

from matplotlib import pyplot as plt
from scipy.stats import spearmanr
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

# Two's-complement reinterpretation constants.
# NetworkX node labels are 64-bit hash values the planner prints either
# unsigned (historical data: [0, 2^64-1]) or signed (current data:
# [-2^63, 2^63-1]).  Unsigned values in [2^63, 2^64-1] are folded into
# signed int64 by subtracting 2^64 — the bit pattern is identical; only the
# sign interpretation changes.
_UINT64_MOD  = 2**64
_INT64_MIN   = -(2**63)
_INT64_MAX   = 2**63   # first value that needs reinterpretation


def uint64_to_signed_int64(values: list[int]) -> list[int]:
    """Reinterpret a list of 64-bit node-ID ints as signed int64.

    Accepts the union of signed and unsigned 64-bit ranges, i.e. any value
    in [-2^63, 2^64-1]:
      - [-2^63, 2^63-1]  : already valid int64, returned unchanged
      - [2^63,  2^64-1]  : unsigned form, mapped to [-2^63, -1] via two's
                           complement (identical bit pattern)
    Values outside [-2^63, 2^64-1] raise ValueError.

    The C++ planner historically printed node hashes unsigned and now prints
    them signed; both forms of the same hash map to the same value here.

    NOTE: kept as the readable reference implementation (used by the
    validation/profiling scripts); production code paths use the vectorized
    uint64_ids_to_int64_tensor() instead.
    """
    out = []
    for v in values:
        if v < _INT64_MIN or v >= _UINT64_MOD:
            raise ValueError(
                f"Node ID {v} is outside [-2^63, 2^64-1]."
            )
        out.append(v if v < _INT64_MAX else v - _UINT64_MOD)
    return out


def uint64_ids_to_int64_tensor(values: Sequence[int]) -> torch.Tensor:
    """Convert node-ID ints to an int64 tensor preserving the 64-bit pattern.

    Vectorized equivalent of uint64_to_signed_int64 + torch.tensor(...).
    Accepts the union of signed and unsigned 64-bit ranges, i.e. any value
    in [-2^63, 2^64-1]:
      - [-2^63, 2^63-1]  : already valid int64, returned unchanged
      - [2^63,  2^64-1]  : unsigned form, folded to negative via two's
                           complement (identical bit pattern)
    Values outside [-2^63, 2^64-1] raise ValueError.

    The C++ planner historically printed node hashes unsigned and now prints
    them signed; both forms of the same hash map to the same tensor value.
    """
    for v in values:
        if v < _INT64_MIN or v > _UINT64_MOD - 1:
            raise ValueError(f"Node ID {v} outside [-2^63, 2^64-1].")
    # Mask to 64 bits with exact Python ints (handles negatives), then
    # reinterpret the raw bits as int64 via a zero-copy view — identical to
    # the two's-complement fold done element-wise in uint64_to_signed_int64,
    # but without per-element Python arithmetic on the fold itself.
    masked = [v & 0xFFFFFFFFFFFFFFFF for v in values]
    raw = np.fromiter(masked, dtype=np.uint64, count=len(masked))
    # .copy() so the tensor owns its memory (raw goes out of scope here)
    return torch.from_numpy(raw.view(np.int64).copy())


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
    graphs = pydot.graph_from_dot_data(src)
    if not graphs:
        # pydot returns None for files it can't parse — notably its grammar
        # rejects bare negative numerals as node IDs, which the planner now
        # emits (signed int64 hashes).
        raise ValueError(
            f"pydot could not parse {path}. If the file contains signed "
            "(negative) node IDs, it should have been handled by "
            "_parse_dot_fast; check why the fast parser rejected it."
        )
    return nx.nx_pydot.from_pydot(graphs[0])


# ---------------------------------------------------------------------------
# Fast DOT -> PyG path
# ---------------------------------------------------------------------------
# The planner emits DOT files with a fixed, trivial shape:
#
#   digraph G {
#     <u> -> <v> [label="<int>"];
#     ...
#   }
#
# pydot's pure-Python parser takes ~21 ms per file on these graphs — 97% of
# the whole dataloader-preparation time.  _parse_dot_fast() parses this
# restricted grammar with one regex per line and builds the PyG Data object
# directly, reproducing *exactly* what _load_dot() + _nx_to_pyg() produce:
#
#   * node order  = order of first appearance while scanning edges
#     (source endpoint before target), which is how nx.nx_pydot.from_pydot
#     inserts nodes;
#   * edge order  = MultiDiGraph adjacency order: edges grouped by source
#     node in node-insertion order, then by target in first-edge order, then
#     parallel edges in file order — which is the order from_networkx()
#     iterates G.edges();
#   * same Data fields, dtypes and values (edge_index, edge_attr, edge_label,
#     label, shape, name, num_nodes, node_names — plus node_bits/node_bitint/
#     node_labels in bitmask mode).
#
# Any line that doesn't match the restricted grammar (explicit node
# statements, subgraphs, extra attributes, ...) makes the function return
# None and the caller falls back to the general pydot/NetworkX path, so
# semantics never change.

# `u -> v [label="3"];`  — node tokens may be quoted; label may be unquoted.
# Node tokens may carry a leading minus: the planner now prints node-ID
# hashes as *signed* int64, so bare negative numerals appear in DOT files.
_DOT_EDGE_RE = re.compile(
    r'^\s*"?(-?[0-9A-Za-z_]+)"?\s*->\s*"?(-?[0-9A-Za-z_]+)"?\s*'
    r'(?:\[\s*label\s*=\s*("?-?\d+"?)\s*\])?\s*;?\s*$'
)
_DOT_HEADER_RE = re.compile(r"^\s*digraph\s+([0-9A-Za-z_]+)\s*\{\s*$")
_DOT_CLOSE_RE = re.compile(r"^\s*\}\s*$")


def _parse_dot_fast(src: str, bitmask: bool = False) -> Optional[Data]:
    """Parse a planner DOT string directly into a PyG Data object.

    Returns None when the input doesn't match the planner's restricted DOT
    grammar; the caller must then fall back to _load_dot() + _nx_to_pyg().
    """
    graph_name = None
    node_idx: Dict[str, int] = {}
    # adjacency in NetworkX MultiDiGraph order: u_idx -> {v_idx: [raw labels]}
    adj: Dict[int, Dict[int, List[str]]] = {}

    for line in src.splitlines():
        m = _DOT_EDGE_RE.match(line)
        if m is not None:
            u, v, raw_label = m.group(1), m.group(2), m.group(3)
            if raw_label is None:
                # Unlabeled edges change the attribute layout that
                # from_networkx() would produce — use the general path.
                return None
            ui = node_idx.setdefault(u, len(node_idx))
            vi = node_idx.setdefault(v, len(node_idx))
            adj.setdefault(ui, {}).setdefault(vi, []).append(raw_label)
            continue
        if not line.strip():
            continue
        if graph_name is None:
            hm = _DOT_HEADER_RE.match(line)
            if hm is not None:
                graph_name = hm.group(1)
                continue
            return None  # first significant line is not a digraph header
        if _DOT_CLOSE_RE.match(line):
            continue
        return None  # node statements, subgraphs, extra attrs, ... -> pydot

    if graph_name is None:
        return None

    # Flatten adjacency in the exact order MultiDiGraph.edges() iterates.
    srcs: List[int] = []
    dsts: List[int] = []
    raw_labels: List[str] = []
    for ui in range(len(node_idx)):
        for vi, labels in adj.get(ui, {}).items():
            for lab in labels:
                srcs.append(ui)
                dsts.append(vi)
                raw_labels.append(lab)

    data = Data()
    data.edge_index = torch.tensor([srcs, dsts], dtype=torch.int64)
    edge_label = torch.tensor(
        [int(lab.replace('"', "")) for lab in raw_labels], dtype=torch.int64
    )
    data.edge_attr = edge_label.view(-1, 1).float()
    # Extra fields kept for parity with the from_networkx() output.
    data.edge_label = edge_label
    data.label = raw_labels
    data.shape = torch.zeros(len(node_idx), dtype=torch.int64)
    data.name = graph_name
    data.num_nodes = len(node_idx)

    nodes = list(node_idx)  # insertion order == from_networkx node order
    if bitmask:
        if not nodes:
            return None
        bit_len = len(nodes[0])
        if any(len(n) != bit_len for n in nodes):
            raise ValueError("Inconsistent bit length across nodes.")
        joined = "".join(nodes)
        bits_u8 = np.frombuffer(joined.encode("ascii"), dtype=np.uint8) - ord("0")
        if not ((bits_u8 == 0) | (bits_u8 == 1)).all():
            raise ValueError("Node labels are not 0/1 bitstrings.")
        data.node_bits = torch.from_numpy(
            bits_u8.reshape(len(nodes), bit_len).astype(bool)
        )
        weights = 2 ** torch.arange(bit_len - 1, -1, -1)  # msb…lsb
        data.node_bitint = (data.node_bits.to(torch.int64) * weights).sum(dim=1)
        data.node_labels = nodes
        data.node_names = data.node_bitint
    else:
        data.node_names = uint64_ids_to_int64_tensor([int(n) for n in nodes])

    return data


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

        # (Optional) if you still want a scalar id tensor alongside node_bits,
        # keep it as int64 — float32 would lose precision for |id| > 2^24.
        data.node_names = data.node_bitint  # already int64

    else:
        # 2) Node IDs → int64 tensor.
        # Node labels from NetworkX may be unsigned 64-bit hash values that fall
        # in [2^63, 2^64-1] — valid uint64 but they overflow signed int64.
        # We reinterpret them as signed int64 via two's complement (same bit
        # pattern, different sign interpretation) so torch.int64 never overflows.
        # normalize_int64_ids() in model.py then maps [-2^63, 2^63-1] → [-1, 1]
        # over float64, so the full 64-bit uniqueness is preserved end-to-end.
        raw_ids = [int(n) for n in G.nodes()]
        data.node_names = uint64_ids_to_int64_tensor(raw_ids)

    # 3) Clean up unused fields if you like
    # del data.edge_label, data.edge_type, data.x

    if diagnose:
        diagnose_data(data)

    return data


def load_graph(
    path: str | Path,
    bitmask: bool = False,
    if_plot_graph: bool = False,
    graph_cache: Optional[Dict[str, Data]] = None,
) -> Data:
    """Load one DOT file into a PyG Data object.

    Resolution order:
      1. graph_cache lookup (pre-serialized Data, see preprocess_dot_to_pt.py)
      2. _parse_dot_fast() — regex parser for the planner's restricted grammar
      3. _load_dot() + _nx_to_pyg() — general pydot/NetworkX fallback

    All three produce identical Data objects for planner-generated files.
    """
    if graph_cache is not None and not if_plot_graph:
        cached = graph_cache.get(str(path))
        if cached is None:  # cache keys are resolved absolute paths
            cached = graph_cache.get(str(Path(path).resolve()))
        if cached is not None:
            return cached

    if if_plot_graph:  # plotting needs the NetworkX graph: general path only
        G = _load_dot(Path(path))
        plot_graph(G)
        return _nx_to_pyg(G, bitmask=bitmask)

    data = _parse_dot_fast(Path(path).read_text(), bitmask=bitmask)
    if data is None:  # DOT construct outside the planner grammar
        data = _nx_to_pyg(_load_dot(Path(path)), bitmask=bitmask)
    return data


def preprocess_sample(
    state_path: str,
    depth: int | None = None,
    target: int | None = None,
    goal_path: Optional[str] = None,
    bitmask: bool = False,
    if_plot_graph: bool = False,
    if_diagnose: bool = False,
    graph_cache: Optional[Dict[str, Data]] = None,
) -> Dict[str, Any]:
    ds = load_graph(
        state_path, bitmask=bitmask, if_plot_graph=if_plot_graph, graph_cache=graph_cache
    )
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
        dg = load_graph(
            goal_path,
            bitmask=bitmask,
            if_plot_graph=if_plot_graph,
            graph_cache=graph_cache,
        )
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
        # Rank correlation between predictions and targets — the quantity A*
        # actually consumes (node ordering), as opposed to R² (calibration).
        # NaN (e.g. constant predictions) is reported as 0.0 so the training
        # history stays JSON-serializable and plottable.
        rho = spearmanr(all_preds, all_targets).statistic
        spearman = float(rho) if not math.isnan(rho) else 0.0

        return {
            "val_loss": 1 - r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": spearman,
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
            # IDs path — node IDs as int64 (raw two's-complement hash bits,
            # exactly what the C++ planner feeds via CreateTensor<int64_t>).
            # The wrapper casts to float64 in-graph before normalization, so
            # full int64 precision survives the process boundary.  Edge
            # attributes are int64 too (C++ CreateTensor<int64_t>) and are
            # cast to float32 in-graph.
            input_names += [
                "state_node_ids",
                "state_edge_index",
                "state_edge_attr",
                "state_batch",
            ]
            dummy_inputs += [
                torch.arange(Ns, dtype=torch.int64),         # [Ns]
                torch.zeros((2, Es), dtype=torch.int64),     # [2, Es]
                torch.zeros((Es, 1), dtype=torch.int64),     # [Es, 1]
                torch.zeros(Ns, dtype=torch.int64),          # [Ns]
            ]
            dynamic_axes.update(
                {
                    "state_node_ids":     {0: "Ns"},
                    "state_edge_index":   {1: "Es"},
                    "state_edge_attr":    {0: "Es"},
                    "state_batch":        {0: "Ns"},
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
                    torch.arange(Ng, dtype=torch.int64),         # [Ng]
                    torch.zeros((2, Eg), dtype=torch.int64),     # [2, Eg]
                    torch.zeros((Eg, 1), dtype=torch.int64),     # [Eg, 1]
                    torch.zeros(Ng, dtype=torch.int64),          # [Ng]
                ]
                dynamic_axes.update(
                    {
                        "goal_node_ids":     {0: "Ng"},
                        "goal_edge_index":   {1: "Eg"},
                        "goal_edge_attr":    {0: "Eg"},
                        "goal_batch":        {0: "Ng"},
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
                torch.zeros((Es, 1), dtype=torch.int64),  # [Es, 1] — C++ feeds int64
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
                    torch.zeros((Eg, 1), dtype=torch.int64),  # [Eg, 1] — C++ feeds int64
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

        # The wrapper holds a *reference* to self.model, so wrapper.cpu()
        # moves the shared core model to CPU.  Restore the original device
        # afterwards (even on export failure) so later predict_* calls don't
        # hit a CPU/CUDA device mismatch.
        original_device = next(self.model.parameters()).device
        wrapper = Wrapper(self.model).eval()
        try:
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
        finally:
            self.model.to(original_device)


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
    - Bitmask=False -> produces int64 node id vectors with keys: state_node_ids / goal_node_ids

    Returns numpy arrays with the exact dtypes/shapes the ONNX model expects
    (matching what the C++ planner feeds via CreateTensor<...>):
      state_node_bits: uint8  [Ns, bit_len]   (or state_node_ids: int64 [Ns])
      state_edge_index: int64 [2, Es]
      state_edge_attr: int64  [Es, 1]
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
        edge_attr = data.edge_label.view(-1, 1).long()  # int64, cast in-graph

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
        """read DOT -> (node_ids int64 [N], edge_index [2,E], edge_attr int64 [E,1])

        Node IDs are returned as int64 (uint64 hash bits reinterpreted via
        two's complement) — the same raw representation the C++ planner feeds
        with CreateTensor<int64_t>.  The ONNX graph itself casts them to
        float64 before normalize_int64_ids(), so no precision is lost here.
        """
        G = _load_dot(path)
        for _, d in G.nodes(data=True):
            d["shape"] = {"circle": 0, "doublecircle": 1}.get(
                d.get("shape", "circle"), 0
            )
        for _, _, d in G.edges(data=True):
            d["edge_label"] = int(str(d.get("label", "0")).strip('"'))

        data = from_networkx(G)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_label.view(-1, 1).long()  # int64, cast in-graph

        node_ids = uint64_ids_to_int64_tensor([int(x) for x in G.nodes()])
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
            torch.cat(s_nodes) if s_nodes else torch.zeros((0,), dtype=torch.int64)
        )

    state_edge_index = (
        torch.cat(s_edges, dim=1) if s_edges else torch.zeros((2, 0), dtype=torch.int64)
    )
    state_edge_attr = (
        torch.cat(s_attrs, dim=0)
        if s_attrs
        else torch.zeros((0, 1), dtype=torch.int64)
    )
    state_batch = (
        torch.cat(s_batch, dim=0) if s_batch else torch.zeros((0,), dtype=torch.int64)
    )

    # build feed dict with exact names/dtypes the ONNX expects
    feed: Dict[str, np.ndarray] = {}
    if bitmask:
        feed["state_node_bits"] = state_node_bits.numpy()  # uint8 [Ns, bit_len]
    else:
        feed["state_node_ids"] = state_node_ids.numpy()  # int64 [Ns]
    feed["state_edge_index"] = state_edge_index.numpy()  # int64 [2, Es]
    feed["state_edge_attr"] = state_edge_attr.numpy()  # int64 [Es, 1]
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
            feed["goal_node_bits"]    = goal_node_bits.numpy()  # uint8  [Ng, bit_len]
        else:
            goal_node_ids = (
                torch.cat(g_nodes)
                if g_nodes
                else torch.zeros((0,), dtype=torch.int64)
            )
            feed["goal_node_ids"] = goal_node_ids.numpy()  # int64 [Ng]

        goal_edge_index = (
            torch.cat(g_edges, dim=1)
            if g_edges
            else torch.zeros((2, 0), dtype=torch.int64)
        )
        goal_edge_attr = (
            torch.cat(g_attrs, dim=0)
            if g_attrs
            else torch.zeros((0, 1), dtype=torch.int64)
        )
        goal_batch = (
            torch.cat(g_batch, dim=0)
            if g_batch
            else torch.zeros((0,), dtype=torch.int64)
        )

        feed["goal_edge_index"] = goal_edge_index.numpy()  # int64 [2, Eg]
        feed["goal_edge_attr"] = goal_edge_attr.numpy()  # int64 [Eg, 1]
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
            bit_input=42 if bitmask else None,
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

    MIN_DEPTH = 0
    # Data-driven target scaling: a hardcoded MAX_DEPTH=50 wasted half the
    # sigmoid range when the deepest observed training distance was 23
    # (mean scaled target 0.0105 — a weak signal at an awkward sigmoid
    # operating point).  Use the max distance actually present in the
    # training targets plus 10% headroom for unseen-depth states.  The
    # resulting slope/intercept are written to distance_estimator_C.txt,
    # which the C++ planner reads to invert the scaling — so training and
    # planner-side interpretation stay consistent automatically.
    max_train_dist = max(s["target"].item() for s in t_s_copy)
    MAX_DEPTH = math.ceil(1.1 * max_train_dist)

    MIN_V_NN = 1e-3
    MAX_V_NN = 1 - MIN_V_NN

    slope = (MAX_V_NN - MIN_V_NN) / (MAX_DEPTH - MIN_DEPTH)

    params = {"slope": slope, "intercept": MIN_V_NN, "max_depth": MAX_DEPTH}

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
