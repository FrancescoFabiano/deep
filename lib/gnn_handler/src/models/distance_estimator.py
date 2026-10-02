"""Per-state distance estimator on planner fringes (the FringeEvalRL contract).

Every state of a fringe goes through the same GINE stack and is mean-pooled per
slot (`membership`), so a slot's distance depends on its own graph only: the
fringe is the batching unit the planner uses, not a context.  Separated mode
encodes the instance goal with a second GINE stack and concatenates it to every
slot.  The regressor ends in a sigmoid: the training target is
``distance * slope + intercept`` in (0, 1) (utils.normalization_params) and the
export wrapper inverts it, so the planner receives ``-distance``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

from deep_nn.features import PointedEmbedding

# floats: the TorchScript exporter cannot hold a Python int above int64 max
INT64_MIN = -(2.0 ** 63)
INT64_RANGE = 2.0 ** 64


def normalize_int64_ids(raw: torch.Tensor) -> torch.Tensor:
    """int64 hash ids (given as float64, so no bit is lost) -> float32 [N, 1] in [-1, 1]."""
    assert raw.dtype == torch.float64, "cast ids to float64 before normalizing"
    normalized = 2.0 * (raw - INT64_MIN) / INT64_RANGE - 1.0
    return normalized.clamp(-1.0, 1.0).to(torch.float32).view(-1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


def _gine(in_dim: int, hidden_dim: int, edge_dim: int) -> GINEConv:
    return GINEConv(
        nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
        edge_dim=edge_dim,
    )


class DistanceEstimator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        use_goal: bool = False,
        node_emb_dim: int = 64,
        edge_emb_dim: int = 32,
        regressor_hidden_dim: int = 128,
        regressor_blocks: int = 3,
        regressor_dropout: float = 0.2,
    ):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, use_goal=use_goal, node_emb_dim=node_emb_dim,
                           edge_emb_dim=edge_emb_dim, regressor_hidden_dim=regressor_hidden_dim,
                           regressor_blocks=regressor_blocks, regressor_dropout=regressor_dropout)
        self.use_goal = use_goal
        self.id_mlp = nn.Sequential(nn.Linear(1, node_emb_dim), nn.ReLU(), nn.Linear(node_emb_dim, node_emb_dim))
        self.pointed = PointedEmbedding(node_emb_dim)
        self.edge_mlp = nn.Sequential(nn.Linear(1, edge_emb_dim), nn.ReLU(), nn.Linear(edge_emb_dim, edge_emb_dim))
        self.state_conv1 = _gine(node_emb_dim, hidden_dim, edge_emb_dim)
        self.state_conv2 = _gine(hidden_dim, hidden_dim, edge_emb_dim)
        if use_goal:
            self.goal_conv1 = _gine(node_emb_dim, hidden_dim, edge_emb_dim)
            self.goal_conv2 = _gine(hidden_dim, hidden_dim, edge_emb_dim)
        layers = [nn.Linear(hidden_dim * (2 if use_goal else 1), regressor_hidden_dim), nn.ReLU()]
        layers += [ResidualBlock(regressor_hidden_dim, regressor_dropout) for _ in range(regressor_blocks)]
        layers += [nn.Linear(regressor_hidden_dim, 1), nn.Sigmoid()]
        self.regressor = nn.Sequential(*layers)

    def _encode(self, node_ids, edge_index, edge_attr, batch, conv1, conv2, pointed_ids=None):
        x = self.id_mlp(normalize_int64_ids(node_ids.to(torch.float64)))
        if pointed_ids is not None:
            x = self.pointed(x, pointed_ids)
        e = self.edge_mlp(edge_attr.to(torch.float32).view(-1, 1))
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))
        return global_mean_pool(x, batch)   # [slots, hidden]; tracing bakes slots = F

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        pointed_ids: torch.Tensor,
        candidate_batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,   # uint8 [F]; a slot never sees the others
        goal_node_features: Optional[torch.Tensor] = None,
        goal_edge_index: Optional[torch.Tensor] = None,
        goal_edge_attr: Optional[torch.Tensor] = None,
        goal_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled distance in (0, 1) per slot. Training packs B fringes in one
        graph (`candidate_batch` = fringe of each slot); the planner sends one."""
        rep = self._encode(node_features, edge_index, edge_attr, membership,
                           self.state_conv1, self.state_conv2, pointed_ids)
        if self.use_goal != (goal_node_features is not None):
            raise ValueError(f"use_goal={self.use_goal} but goal tensors "
                             f"{'given' if goal_node_features is not None else 'missing'}")
        if self.use_goal:
            goal = self._encode(goal_node_features, goal_edge_index, goal_edge_attr, goal_batch,
                                self.goal_conv1, self.goal_conv2)
            goal = goal[candidate_batch] if candidate_batch is not None else goal.expand(rep.size(0), -1)
            rep = torch.cat([rep, goal], dim=1)
        out = self.regressor(rep).squeeze(1)
        if mask is not None:
            # A padded slot is "as far as it gets" (scaled distance 1.0). The
            # planner never reads it; this keeps `mask` a live input at export.
            out = out.masked_fill(~mask[: out.size(0)].to(torch.bool), 1.0)
        return out

    def get_checkpoint(self) -> Dict:
        return {"state_dict": self.state_dict(), "config": dict(self.config)}

    @classmethod
    def load_model(cls, ckpt: Dict) -> "DistanceEstimator":
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()


class OnnxNegativeDistance(nn.Module):
    """Planner-side scores: ``-(sigmoid_out - intercept) / slope`` = ``-distance``.
    rankScores sorts descending, so the shortest estimated distance is expanded
    first; no clamp, so near-goal slots do not collapse into ties (issue #5)."""

    def __init__(self, core: DistanceEstimator, slope: float, intercept: float, fringe_size: int):
        super().__init__()
        self.core, self.slope, self.intercept = core, float(slope), float(intercept)
        self.fringe_size = int(fringe_size)

    def scores(self, scaled: torch.Tensor) -> torch.Tensor:
        # reshape to the constant F: the planner refuses an output whose length ORT
        # cannot infer statically, which the traced pooling leaves open at F=1
        return (-(scaled - self.intercept) / self.slope).reshape(self.fringe_size)

    def forward(self, node_features, edge_index, edge_attr, membership, pointed_ids, mask):
        return self.scores(self.core(node_features, edge_index, edge_attr, membership, pointed_ids, mask=mask))


class OnnxNegativeDistanceSeparated(OnnxNegativeDistance):
    def forward(self, node_features, edge_index, edge_attr, membership, pointed_ids,
                goal_node_features, goal_edge_index, goal_edge_attr, goal_batch, mask):
        return self.scores(self.core(
            node_features, edge_index, edge_attr, membership, pointed_ids, mask=mask,
            goal_node_features=goal_node_features, goal_edge_index=goal_edge_index,
            goal_edge_attr=goal_edge_attr, goal_batch=goal_batch))


class OnnxScaledDistance(nn.Module):
    """One state for the C++ ``GraphNN`` consumer (``--heuristics GNN`` under HFS
    or A*): inputs ``node_features, edge_index, edge_attr [E,1], batch,
    pointed_ids``; output the scaled distance ``[1]``.  The planner inverts the
    scaling with the constant file (``slope = ``, ``intercept = ``) and rounds."""

    def __init__(self, core: DistanceEstimator):
        super().__init__()
        self.core = core

    def forward(self, node_features, edge_index, edge_attr, batch, pointed_ids):
        return self.core(node_features, edge_index, edge_attr, batch, pointed_ids)
