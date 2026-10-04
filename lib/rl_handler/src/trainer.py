"""Contract surface of the deployed planner model.

This module owns everything the deployed C++ planner contract depends on:
the ONNX export wrappers (the contract's positional inputs, lib/deep_nn/
contract.py), checkpoint save/load (payload format), and the `to_onnx`
export path — reused verbatim by the offline exporter as the contract
guarantee.  Training logic lives in src/offline/qlearning.py.

Do not rename RLFrontierTrainer, the wrapper classes, or any export symbol:
checkpoints and the ONNX graph structure reference them.
"""

from __future__ import annotations

import copy

import torch
from pathlib import Path
from torch import nn
from typing import Dict, Optional

from deep_nn import contract
from deep_nn.dense import set_dense_aggregation
from src.models.frontier_policy import FrontierPolicyNetwork

FAILURE_EPS = 1e-9
FAILURE_REWARD_VALUE = -1.0


class OnnxFrontierPolicyWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        pointed_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            pointed_ids=pointed_ids,
            candidate_batch=None,
            mask=mask,
        )


class OnnxFrontierPolicySeparatedWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        pointed_ids: torch.Tensor,
        goal_node_features: torch.Tensor,
        goal_edge_index: torch.Tensor,
        goal_edge_attr: torch.Tensor,
        goal_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            pointed_ids=pointed_ids,
            candidate_batch=None,
            mask=mask,
            goal_node_features=goal_node_features,
            goal_edge_index=goal_edge_index,
            goal_edge_attr=goal_edge_attr,
            goal_batch=goal_batch,
        )


class RLFrontierTrainer:
    """Model holder with the contract-relevant operations (save/load/export).

    The constructor keeps its historical signature because existing
    checkpoints and callers (offline_main.py, offline_analysis.py) depend
    on it.
    """

    def __init__(
        self,
        model: FrontierPolicyNetwork,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
        reward_formulation: str = "negative_distance",
        kind_of_data: str = "merged",
        max_grad_norm: float = 0.0,
        m_failed_state: Optional[float] = None,
        **_: object,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.reward_formulation = reward_formulation
        self.kind_of_data = kind_of_data
        self.max_grad_norm = float(max_grad_norm)
        self.m_failed_state = (
            float(m_failed_state) if m_failed_state is not None else float("inf")
        )
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0.")

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target_device = self._model_device()
        self.device = target_device
        out = {}
        for k, v in batch.items():
            out[k] = v.to(target_device) if isinstance(v, torch.Tensor) else v
        return out

    def save_model(self, out_path: str | Path, metrics: Optional[Dict[str, float]] = None):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "node_input_dim": self.model.encoder.input_proj.in_features,
                "hidden_dim": self.model.encoder.input_proj.out_features,
                "gnn_layers": len(self.model.encoder.layers),
                "conv_type": self.model.encoder.conv_type,
                "pooling_type": self.model.pooling_type,
                "dataset_type": self.model.dataset_type,
                "edge_emb_dim": self.model.edge_emb_dim,
                "num_edge_labels": self.model.num_edge_labels,
                "num_node_labels": self.model.num_node_labels,
                "use_global_context": self.model.use_global_context,
                "context_mode": self.model.context_mode,
                "attn_heads": self.model.attn_heads,
                "attn_layers": self.model.attn_layers,
                "mlp_depth": (len(self.model.policy_head) - 1) // 2,
                "use_goal_separate_input": self.model.use_goal_separate_input,
            },
            "metrics": metrics or {},
        }
        torch.save(payload, out_path)

    @staticmethod
    def load_model(path: str | Path, device: Optional[str | torch.device] = None) -> FrontierPolicyNetwork:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = payload["config"]
        cfg.setdefault("num_node_labels", 4096)
        model = FrontierPolicyNetwork(**cfg)
        # A checkpoint from before the `pointed_ids` input lacks the (zero-
        # initialised) designated-world vector; it re-exports with identical scores.
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        if unexpected or set(missing) - {"encoder.pointed.weight"}:
            raise RuntimeError(f"{path}: state_dict mismatch (missing {missing}, unexpected {unexpected})")
        model.eval()
        return model

    def to_onnx(
        self,
        out_path: str | Path,
        node_input_dim: int,
        onnx_frontier_size: int = 32,
        aggregation: str = "scatter",
    ) -> None:
        """Export through the shared planner contract (deep_nn.contract). `aggregation`
        = dense swaps the scatters for one-hot matmuls (deep_nn.dense) on a copy."""
        if str(self.model.dataset_type).upper() != "HASHED":
            raise ValueError(
                f"dataset_type {self.model.dataset_type}: FringeEvalRL deploys HASHED "
                f"models only (issue #1)")
        separated = self.kind_of_data == "separated" and self.model.use_goal_separate_input
        wrapper_cls = OnnxFrontierPolicySeparatedWrapper if separated else OnnxFrontierPolicyWrapper
        model_was_training = bool(self.model.training)
        model_device = self._model_device()
        if aggregation not in ("scatter", "dense"):
            raise ValueError(f"aggregation must be scatter or dense, got {aggregation!r}")
        core = (self.model if aggregation == "scatter"
                else set_dense_aggregation(copy.deepcopy(self.model), True, blocked=int(onnx_frontier_size) >= 4))
        try:
            contract.export(wrapper_cls(core), out_path, onnx_frontier_size, separated)
        finally:
            self.model.to(model_device)
            self.model.train(model_was_training)
            self.device = model_device
