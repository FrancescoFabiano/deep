"""Target scaling, the trainer wrapper around DistanceEstimator, and its export."""

from __future__ import annotations

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
from src.model import BaseModel
from src.models.distance_estimator import (
    DistanceEstimator,
    OnnxNegativeDistance,
    OnnxNegativeDistanceSeparated,
)

# The historical fixed range of the target map, used when the generation depth
# cannot be read off the table name.
LEGACY_MAX_DEPTH = 50
MIN_V_NN = 1e-3

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

    def to_onnx(self, onnx_path: str | Path, fringe_size: int, params: Dict[str, float]) -> Path:
        """Export through the shared planner contract, scores = -distance."""
        wrapper_cls = OnnxNegativeDistanceSeparated if self.model.use_goal else OnnxNegativeDistance
        wrapper = wrapper_cls(self.model, params["slope"], params["intercept"])
        try:
            return contract.export(wrapper, onnx_path, fringe_size, separated=self.model.use_goal)
        finally:
            self.model.to(self.device)

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
