"""Checkpoint-selection override: pick the best epoch by a chosen metric.

The baseline selects the checkpoint by lowest val_loss (= 1 - R², a
calibration metric), but A* consumes the heuristic's ORDERING — Spearman
rank correlation is the better proxy (observed: last-epoch Spearman 0.9096
vs best-by-val_loss 0.8865 on the same run).

DistanceEstimatorModelPlus.train() mirrors BaseModel.train() but:
  * tracks the best epoch by `ckpt_metric` ("spearman": higher is better,
    "val_loss": lower is better) and writes it to {model_name}.pt — the file
    the pipeline exports to ONNX and the planner consumes;
  * ALWAYS also saves {model_name}_by_val_loss.pt (baseline criterion) so
    both selections can be compared in-planner without retraining;
  * keeps the baseline's {model_name}_last.pt and history bookkeeping, and
    records best_epoch / best_epoch_by_val_loss / ckpt_metric in the
    history JSON.

Everything else (loss, evaluate(), checkpointing format, ONNX export) is
inherited from the baseline DistanceEstimatorModel.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from src.utils import DistanceEstimatorModel

_METRICS = {
    # name -> (direction sign; +1 = higher is better)
    "spearman": +1,
    "val_loss": -1,
}


def _metric_block(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Baseline metric set (val_loss/mse/rmse/mae/r2/spearman) on tensors."""
    p, t = preds.numpy(), targets.numpy()
    mse = mean_squared_error(t, p)
    r2 = r2_score(t, p)
    rho = spearmanr(p, t).statistic
    return {
        "val_loss": 1 - r2,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(t, p),
        "r2": r2,
        "spearman": float(rho) if not math.isnan(rho) else 0.0,
    }


class DistanceEstimatorModelPlus(DistanceEstimatorModel):

    # Wired by the entry point after prepare_samples_plus(): the scaled
    # target value of unreachable states (f(MAX_DEPTH)) and the scaling
    # params.  No reachable state can alias the value — max reachable
    # distance is < MAX_DEPTH by the 1.1 headroom.  When left None the
    # class behaves exactly like the baseline (single metric set).
    unreachable_target_value: float | None = None
    scale_params: dict | None = None

    def evaluate(self, loader, verbose: bool = False, **kwargs) -> dict:
        """Baseline metrics on ALL states, plus reachable-only (`*_reach`)
        and unreachable-only diagnostics when unreachable states exist.

        Rationale (INV-1): a handful of unreachable val states (target at
        the top of the range) can demolish squared-error metrics — plus's
        all-states val R² read 0.34 while the reachable subset was
        unaffected — so headline numbers and any val_loss-based selection
        must be interpretable per subset.
        """
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                preds.append(self.model(batch).view(-1).cpu())
                targets.append(batch["target"].view(-1).cpu())
        p, t = torch.cat(preds), torch.cat(targets)

        metrics = _metric_block(p, t)

        if self.unreachable_target_value is not None:
            ur = torch.isclose(
                t, torch.tensor(float(self.unreachable_target_value)),
                atol=1e-6,
            )
            if bool(ur.any()):
                metrics.update({
                    f"{k}_reach": v
                    for k, v in _metric_block(p[~ur], t[~ur]).items()
                })
                metrics["n_unreach"] = int(ur.sum())
                metrics["mae_unreach"] = float((p[ur] - t[ur]).abs().mean())
                if self.scale_params is not None:
                    s = self.scale_params
                    metrics["pred_dist_mean_unreach"] = float(
                        (p[ur].mean() - s["intercept"]) / s["slope"]
                    )
            else:  # no unreachable in this loader: reach == all
                metrics.update(
                    {f"{k}_reach": v for k, v in list(metrics.items())}
                )
                metrics["n_unreach"] = 0
        return metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model_name: str = "model",
        n_epochs: int = 200,
        checkpoint_dir: str = ".",
        ckpt_metric: str = "spearman",
        **kwargs,
    ) -> None:
        if ckpt_metric not in _METRICS:
            raise ValueError(
                f"ckpt_metric must be one of {sorted(_METRICS)}, "
                f"got {ckpt_metric!r}"
            )
        sign = _METRICS[ckpt_metric]

        best_score = -float("inf")
        best_epoch = -1
        best_val_loss = float("inf")
        best_epoch_by_val_loss = -1

        os.makedirs(checkpoint_dir, exist_ok=True)
        pbar = tqdm(range(n_epochs), desc="training(plus)...")
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in pbar:
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

            # Primary checkpoint: chosen metric -> {model_name}.pt (the file
            # the pipeline exports to ONNX and the planner consumes).
            score = sign * val_metrics[ckpt_metric]
            if score > best_score:
                self._save_full_checkpoint(f"{checkpoint_dir}/{model_name}.pt")
                best_score = score
                best_epoch = epoch

            # Secondary checkpoint: baseline criterion, for A/B comparison.
            if val_metrics["val_loss"] < best_val_loss:
                self._save_full_checkpoint(
                    f"{checkpoint_dir}/{model_name}_by_val_loss.pt"
                )
                best_val_loss = val_metrics["val_loss"]
                best_epoch_by_val_loss = epoch

            for name, value in val_metrics.items():
                history[name].append(value)
            history["train_loss"].append(avg_loss)

        self._save_full_checkpoint(f"{checkpoint_dir}/{model_name}_last.pt")

        history_extra = {
            "best_epoch": best_epoch,
            "ckpt_metric": ckpt_metric,
            "best_epoch_by_val_loss": best_epoch_by_val_loss,
        }
        # Reuse the baseline's JSON+plot writer, then append the extra keys.
        self.save_and_plot_metrics(history, checkpoint_dir, n_epochs, best_epoch)
        import json
        path = f"{checkpoint_dir}/history_losses.json"
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload.update(history_extra)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)


def select_model_plus(use_goal: bool, use_depth: bool, bitmask: bool):
    """Mirror of baseline select_model() that returns the plus subclass."""
    from src.models.distance_estimator import DistanceEstimator

    return DistanceEstimatorModelPlus(
        DistanceEstimator,
        use_goal=use_goal,
        use_depth=use_depth,
        bit_input=42 if bitmask else None,
    )
