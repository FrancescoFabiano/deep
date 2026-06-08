"""Contract surface of the deployed planner model.

This module owns everything the deployed C++ planner contract depends on:
the ONNX export wrappers (input names/order/dtypes/dynamic axes consumed by
FringeEvalRL), checkpoint save/load (payload format), and the `to_onnx`
export path — reused verbatim by the offline exporter as the contract
guarantee.  Training logic lives in src/offline/dqn.py.

Do not rename RLFrontierTrainer, the wrapper classes, or any export symbol:
checkpoints and the ONNX graph structure reference them.
"""

from __future__ import annotations

import torch
from pathlib import Path
from torch import nn
from typing import Dict, Optional

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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
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
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def to_onnx(
        self,
        out_path: str | Path,
        node_input_dim: int,
        onnx_frontier_size: int = 32,
    ) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # ONNX tracing may specialize pooling/scatter dimensions to the traced
        # candidate range; keep this comfortably above common frontier sizes.
        n_candidates = int(onnx_frontier_size)
        if n_candidates <= 0:
            raise ValueError("onnx_frontier_size must be > 0.")
        n_nodes, n_edges = max(8, n_candidates), 12
        dataset_type = str(self.model.dataset_type).upper()
        raw_node_input_dim = int(node_input_dim)
        node_tensor_dtype = (
            torch.float32
            if dataset_type == "BITMASK"
            else torch.int64
        )
        if dataset_type == "HASHED":
            node_features_dummy = torch.zeros((n_nodes,), dtype=torch.int64)
        else:
            node_features_dummy = torch.zeros((n_nodes, raw_node_input_dim), dtype=node_tensor_dtype)
        model_was_training = bool(self.model.training)
        model_device = self._model_device()
        try:
            if self.kind_of_data == "separated" and self.model.use_goal_separate_input:
                wrapper = OnnxFrontierPolicySeparatedWrapper(self.model).eval().cpu()
                n_goal_nodes, n_goal_edges = 6, 8
                if dataset_type == "HASHED":
                    goal_node_features_dummy = torch.zeros((n_goal_nodes,), dtype=torch.int64)
                else:
                    goal_node_features_dummy = torch.zeros(
                        (n_goal_nodes, raw_node_input_dim),
                        dtype=node_tensor_dtype,
                    )
                dummy_inputs = (
                    node_features_dummy,
                    torch.zeros((2, n_edges), dtype=torch.int64),
                    torch.zeros((n_edges,), dtype=torch.int64),
                    torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                    goal_node_features_dummy,
                    torch.zeros((2, n_goal_edges), dtype=torch.int64),
                    torch.zeros((n_goal_edges,), dtype=torch.int64),
                    torch.zeros((n_goal_nodes,), dtype=torch.int64),
                    torch.ones((n_candidates,), dtype=torch.uint8),
                )
                input_names = [
                    "node_features",
                    "edge_index",
                    "edge_attr",
                    "membership",
                    "goal_node_features",
                    "goal_edge_index",
                    "goal_edge_attr",
                    "goal_batch",
                    "mask",
                ]
                dynamic_axes = {
                    "node_features": {0: "N"},
                    "edge_index": {1: "E"},
                    "edge_attr": {0: "E"},
                    "membership": {0: "N"},
                    "goal_node_features": {0: "GN"},
                    "goal_edge_index": {1: "GE"},
                    "goal_edge_attr": {0: "GE"},
                    "goal_batch": {0: "GN"},
                    "mask": {0: "F"},
                    "logits": {0: "F"},
                }
            else:
                wrapper = OnnxFrontierPolicyWrapper(self.model).eval().cpu()
                dummy_inputs = (
                    node_features_dummy,
                    torch.zeros((2, n_edges), dtype=torch.int64),
                    torch.zeros((n_edges,), dtype=torch.int64),
                    torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                    torch.ones((n_candidates,), dtype=torch.uint8),
                )
                input_names = [
                    "node_features",
                    "edge_index",
                    "edge_attr",
                    "membership",
                    "mask",
                ]
                dynamic_axes = {
                    "node_features": {0: "N"},
                    "edge_index": {1: "E"},
                    "edge_attr": {0: "E"},
                    "membership": {0: "N"},
                    "mask": {0: "F"},
                    "logits": {0: "F"},
                }

            torch.onnx.export(
                wrapper,
                dummy_inputs,
                out_path.as_posix(),
                opset_version=18,
                dynamo=False,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=False,
            )
        finally:
            self.model.to(model_device)
            self.model.train(model_was_training)
            self.device = model_device
