"""Target scaling, the trainer wrapper around DistanceEstimator, and its export."""

from __future__ import annotations

import copy

import math
import os
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from deep_nn import contract
from deep_nn.dense import set_dense_aggregation
from src.model import BaseModel
from src.models.distance_estimator import (
    DistanceEstimator,
    OnnxNegativeDistance,
    OnnxNegativeDistanceSeparated,
    OnnxScaledDistance,
)

# The historical fixed range of the target map, used when the generation depth
# cannot be read off the table name.
LEGACY_MAX_DEPTH = 50
MIN_V_NN = 1e-3

# The C++ GraphNN per-state consumer (HFS/A* with --heuristics GNN), by position.
STATE_INPUTS = ("node_features", "edge_index", "edge_attr", "batch", "pointed_ids")
MODEL_INPUTS = ("node_features", "edge_index", "edge_attr", "membership", "pointed_ids",
                "candidate_batch", "mask", "goal_node_features", "goal_edge_index",
                "goal_edge_attr", "goal_batch")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalization_params(max_depth: int) -> Dict[str, float]:
    """target = distance * slope + intercept, in (0, 1); the export inverts it."""
    return {"slope": (1.0 - 2 * MIN_V_NN) / float(max_depth), "intercept": MIN_V_NN}


def scale_distance(distance: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    return (distance * params["slope"] + params["intercept"]).to(torch.float32)


def constant_file(state_onnx: str | Path) -> Path:
    """`distance_estimator_4_state.onnx` -> `distance_estimator_4_state_C.txt`."""
    p = Path(state_onnx)
    return p.with_name(p.stem + "_C.txt")


def model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k in MODEL_INPUTS}


class DistanceEstimatorModel(BaseModel):
    """MSE regression on the scaled distance of every LABELLED slot (a state
    the generator marked unreachable has no distance and is masked out)."""

    def __init__(self, lr: float = 1e-3, device=None, **estimator_kwargs):
        super().__init__(model=DistanceEstimator(**estimator_kwargs), device=device,
                         optimizer_kwargs={"lr": lr})
        self.criterion = nn.MSELoss()

    def _compute_loss(self, batch):
        m = batch["label_mask"]
        preds = self.model(**model_inputs(batch))
        return self.criterion(preds[m], batch["target"][m])

    def evaluate(self, loader, verbose: bool = False, **kwargs) -> dict:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                m = batch["label_mask"]
                preds.extend(self.model(**model_inputs(batch))[m].cpu().tolist())
                targets.extend(batch["target"][m].cpu().tolist())
        mse = mean_squared_error(targets, preds)
        rho = spearmanr(preds, targets).statistic
        r2 = r2_score(targets, preds)
        return {
            "val_loss": 1 - r2, "mse": mse, "rmse": math.sqrt(mse),
            "mae": mean_absolute_error(targets, preds), "r2": r2,
            "spearman": 0.0 if math.isnan(rho) else float(rho),
        }

    def _save_full_checkpoint(self, path, **metrics):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = self.model.get_checkpoint()
        ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def load_model(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location=self.device, weights_only=False)
        self.model = DistanceEstimator.load_model(ckpt).to(self.device)

    def predict_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(**model_inputs(self._move_batch_to_device(batch))).cpu()

    def _export_core(self, aggregation: str, fringe_size: int = 1):
        """A CPU copy of the model for export; `dense` swaps the scatters for one-hot
        matmuls (deep_nn.dense: blocked per slot when the export is F >= 4 wide), the
        live model is never touched."""
        if aggregation not in ("scatter", "dense"):
            raise ValueError(f"aggregation must be scatter or dense, got {aggregation!r}")
        core = copy.deepcopy(self.model).cpu().eval()
        return set_dense_aggregation(core, aggregation == "dense", blocked=int(fringe_size) >= 4)

    def to_onnx(self, onnx_path: str | Path, fringe_size: int, params: Dict[str, float],
                aggregation: str = "scatter") -> Path:
        """Export through the shared planner contract, scores = -distance."""
        wrapper_cls = OnnxNegativeDistanceSeparated if self.model.use_goal else OnnxNegativeDistance
        wrapper = wrapper_cls(self._export_core(aggregation, fringe_size), params["slope"], params["intercept"],
                              fringe_size)
        return contract.export(wrapper, onnx_path, fringe_size, separated=self.model.use_goal)

    def to_onnx_state(self, onnx_path: str | Path, params: Dict[str, float],
                      aggregation: str = "scatter") -> Path:
        """Per-state export for the C++ GraphNN consumer, plus the constant file
        (``<stem>_C.txt``) it inverts the target scaling with."""
        if self.model.use_goal:
            raise ValueError("the GraphNN consumer supports merged models only")
        core = self._export_core(aggregation)
        onnx_path = Path(onnx_path)
        n, e = 8, 12
        dummy = (torch.zeros(n, dtype=torch.int64), torch.zeros((2, e), dtype=torch.int64),
                 torch.zeros((e, 1), dtype=torch.int64), torch.zeros(n, dtype=torch.int64),
                 torch.arange(2, dtype=torch.int64))
        axes = {"node_features": {0: "N"}, "edge_index": {1: "E"}, "edge_attr": {0: "E"},
                "batch": {0: "N"}, "pointed_ids": {0: "P"}}
        try:
            torch.onnx.export(OnnxScaledDistance(core).eval().cpu(), dummy, onnx_path.as_posix(),
                              opset_version=contract.OPSET, dynamo=False, input_names=list(STATE_INPUTS),
                              output_names=["scaled_distance"], dynamic_axes=axes, do_constant_folding=False)
        finally:
            self.model.to(self.device)
        import onnx
        names = [i.name for i in onnx.load(str(onnx_path)).graph.input]
        if names != list(STATE_INPUTS):
            raise ValueError(f"{onnx_path}: inputs {names} != {STATE_INPUTS} (an unused input was pruned)")
        constant_file(onnx_path).write_text(f"slope = {params['slope']}\nintercept = {params['intercept']}\n")
        return onnx_path

    def verify_onnx(self, onnx_path: str | Path, feeds: Sequence[Dict[str, torch.Tensor]],
                    params: Dict[str, float]) -> Dict[str, float]:
        """PyTorch vs onnxruntime on real single fringes, as the planner feeds them."""
        self.model.eval().cpu()

        def scores(feed):
            d = self.model(**model_inputs(feed))
            return -(d - params["intercept"]) / params["slope"]

        try:
            return contract.check_parity(onnx_path, scores, feeds)
        finally:
            self.model.to(self.device)
