from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

# Int64 range constants for normalizing scalar node IDs from [-2^63, 2^63-1] to [-1, 1].
INT64_MIN   = float(-(2**63))       # -9223372036854775808
INT64_MAX   = float(2**63 - 1)      #  9223372036854775807
INT64_RANGE = INT64_MAX - INT64_MIN  # 2^64 - 1  (denominator for linear map)


def normalize_int64_ids(raw: torch.Tensor) -> torch.Tensor:
    """
    Map a 1-D tensor of int64 scalar node IDs to the [-1, 1] range.

    The input tensor must be float64 (torch.float64) to preserve full int64
    precision — float32 only has a 23-bit mantissa and would silently round
    any ID whose absolute value exceeds ~2^24, making large IDs
    indistinguishable from one another.

    Formula (linear min-max rescaling):
        normalized = 2 * (x - INT64_MIN) / INT64_RANGE - 1

    Edge values:  x = INT64_MIN  →  -1.0
                  x = INT64_MAX  →  +1.0
    """
    assert raw.dtype == torch.float64, (
        "normalize_int64_ids expects float64 input to avoid precision loss. "
        "Cast with .to(torch.float64) before calling."
    )
    normalized = 2.0 * (raw - INT64_MIN) / INT64_RANGE - 1.0
    return normalized.clamp(-1.0, 1.0).to(torch.float32).view(-1, 1)


class ResidualBlock(nn.Module):
    """A simple residual block with BatchNorm, ReLU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.lin2(out)
        out = self.bn2(out)
        return F.relu(out + identity)


class DistanceEstimator(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            use_goal: bool = True,
            use_depth: bool = True,
            node_emb_dim: int = 64,
            edge_emb_dim: int = 32,
            regressor_hidden_dim: int = 128,
            regressor_blocks: int = 3,
            regressor_dropout: float = 0.2,
            min_value: float = 1e-3,
            # Number of bits for bitmask node IDs.
            # Set to None to use the scalar int64 path (node_names).
            bit_input: Optional[int] = None,
    ):
        super().__init__()
        self.use_goal  = use_goal
        self.use_depth = use_depth
        self.min_value = min_value
        self.bit_input = bit_input

        # Node ID embedding: accepts either a bit-vector [N, bit_input]
        # or a normalized scalar id [N, 1].
        id_in_features = bit_input if bit_input is not None else 1
        self.id_mlp = nn.Sequential(
            nn.Linear(id_in_features, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim),
        )

        # Edge attribute embedding: scalar edge weight → edge_emb_dim.
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # GINE message-passing layers for the current state graph.
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

        # GINE message-passing layers for the goal graph (optional).
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

        # Regressor: graph embedding(s) [+ optional depth] → scalar in (0, 1).
        regressor_in_dim = hidden_dim * (2 if use_goal else 1) + (1 if use_depth else 0)
        regressor_layers = [nn.Linear(regressor_in_dim, regressor_hidden_dim), nn.ReLU()]
        for _ in range(regressor_blocks):
            regressor_layers.append(ResidualBlock(regressor_hidden_dim, regressor_dropout))
        regressor_layers.append(nn.Linear(regressor_hidden_dim, 1))
        regressor_layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor_layers)

    def _encode_graph(self, graph, conv1, conv2) -> torch.Tensor:
        """
        Embed a graph into a fixed-size vector via two GINE layers + global mean pool.

        Node features are derived from either:
          - node_bits  [N, bit_input]  (bitmask path, preferred)
          - node_names [N]             (scalar int64 path, cast to float64 first)
        """
        device = graph.edge_index.device

        if hasattr(graph, "node_bits") and self.bit_input is not None:
            # Bitmask path: binary vectors, no precision concerns.
            node_features = graph.node_bits.to(torch.float32).to(device)
        else:
            # Scalar int64 path: cast to float64 BEFORE normalization to
            # preserve all 64 bits of precision across the full int64 range.
            node_ids_f64 = graph.node_names.to(device).to(torch.float64)
            node_features = normalize_int64_ids(node_ids_f64)  # returns float32 [N,1]

        node_emb = self.id_mlp(node_features)
        edge_emb = self.edge_mlp(graph.edge_attr.to(device).float())

        node_emb = F.relu(conv1(node_emb, graph.edge_index, edge_emb))
        node_emb = F.relu(conv2(node_emb, graph.edge_index, edge_emb))
        return global_mean_pool(node_emb, graph.batch.clone())

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_emb = self._encode_graph(
            batch_dict["state_graph"], self.state_conv1, self.state_conv2
        )

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            goal_emb = self._encode_graph(
                batch_dict["goal_graph"], self.goal_conv1, self.goal_conv2
            )
            rep = torch.cat([state_emb, goal_emb], dim=1)
        else:
            rep = state_emb

        if self.use_depth:
            depth = batch_dict.get("depth")
            if depth is None:
                depth = torch.zeros(len(rep), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            rep = torch.cat([rep, depth], dim=1)

        out = self.regressor(rep).squeeze(1)
        # Clamp only at inference: the sigmoid already bounds (0, 1), and
        # clamping during training kills the gradient whenever the bound
        # binds.  With heavily zero-skewed targets the whole batch can sink
        # below min_value early on, freezing training permanently (observed:
        # val metrics bit-identical from epoch 3 for 200 epochs).
        if self.training:
            return out
        return out.clamp(min=self.min_value, max=1 - self.min_value)

    def get_checkpoint(self) -> Dict:
        cfg = {
            "hidden_dim":    self.state_conv1.nn[0].out_features,
            "node_emb_dim":  self.id_mlp[0].out_features,
            "edge_emb_dim":  self.edge_mlp[0].out_features,
            "use_goal":      self.use_goal,
            "use_depth":     self.use_depth,
            "bit_input":     self.bit_input,
        }
        return {"state_dict": self.state_dict(), "config": cfg}

    @classmethod
    def load_model(cls, ckpt: Dict) -> "DistanceEstimator":
        cfg   = ckpt["config"]
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            use_goal=cfg["use_goal"],
            use_depth=cfg.get("use_depth", True),
            node_emb_dim=cfg["node_emb_dim"],
            edge_emb_dim=cfg["edge_emb_dim"],
            bit_input=cfg.get("bit_input", None),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# ONNX export wrappers
# ---------------------------------------------------------------------------
# These wrappers replicate _encode_graph with explicit tensor arguments
# (no Data objects) so ONNX can trace a static graph.  Both wrappers share
# the same _global_mean helper that avoids scatter_mean (not always supported
# by ONNX opsets < 16).
# ---------------------------------------------------------------------------

def _global_mean_pool_onnx(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    ONNX-friendly global mean pooling via scatter_add.
    Equivalent to torch_geometric.nn.global_mean_pool but fully traceable.

    Earlier versions used F.one_hot(batch, num_classes=num_graphs), but the
    dynamo-based ONNX exporter (default since torch 2.9) rejects it:
    one_hot needs num_classes as a *specialized Python int*, while
    batch.max()+1 is data-dependent.  scatter_add only uses num_graphs to
    pre-allocate the output tensor, which the exporter handles as a dynamic
    (unbacked) size — and it maps to ONNX ScatterElements (opset >= 16).
    """
    num_graphs = batch.max() + 1
    sums = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
    # Tell the dynamo exporter the pooled batch is never empty; without this
    # hint the downstream BatchNorm1d guards on `numel == 0` over the unbacked
    # size and torch.export fails.  In eager mode shape[0] is a plain int, so
    # the assertion is a no-op.  Skipped under the legacy TorchScript tracer
    # (dynamo=False), where shape[0] is a traced Tensor that _check rejects —
    # the legacy exporter needs no hint anyway.
    if not torch.jit.is_tracing():
        torch._check(sums.shape[0] >= 1)
    sums = sums.scatter_add(0, batch.unsqueeze(1).expand_as(x), x)     # [G, D]
    counts = torch.zeros(sums.shape[0], 1, dtype=x.dtype, device=x.device)
    counts = counts.scatter_add(0, batch.unsqueeze(1), torch.ones_like(x[:, :1]))
    return sums / torch.clamp(counts, min=1.0)                          # [G, D]


class OnnxDistanceEstimatorWrapperIds(nn.Module):
    """
    ONNX wrapper for the scalar int64 node-ID path (bit_input=None).

    Inputs are flat tensors rather than PyG Data objects so that ONNX can
    trace a static computation graph.  Node IDs are passed as **int64**
    (raw two's-complement hash bits — exactly what the C++ planner feeds via
    CreateTensor<int64_t>) and cast to float64 *inside* the graph before
    normalize_int64_ids(), mirroring the training path bit-for-bit.  Edge
    attributes likewise arrive as int64 (C++ CreateTensor<int64_t>) and are
    cast to float32 in-graph.
    """

    def __init__(self, core: DistanceEstimator):
        super().__init__()
        assert core.bit_input is None, (
            "OnnxDistanceEstimatorWrapperIds requires a model trained without "
            "bitmask node features (bit_input=None).  Use "
            "OnnxDistanceEstimatorWrapperBits for bit_input models."
        )
        self.core = core

    def _encode_raw(
            self,
            node_ids:     torch.Tensor,   # [N]  int64 — raw two's-complement bits
            edge_index:   torch.Tensor,   # [2, E]
            edge_attr:    torch.Tensor,   # [E, 1]  int64
            batch:        torch.Tensor,   # [N]
            conv1:        GINEConv,
            conv2:        GINEConv,
    ) -> torch.Tensor:
        # int64 -> float64 inside the graph: same cast the training path does
        # (model.py `.to(torch.float64)`), so precision behaviour is identical.
        node_features = normalize_int64_ids(node_ids.to(torch.float64))  # [N,1]
        node_emb      = self.core.id_mlp(node_features)
        edge_emb      = self.core.edge_mlp(edge_attr.float())
        node_emb      = F.relu(conv1(node_emb, edge_index, edge_emb))
        node_emb      = F.relu(conv2(node_emb, edge_index, edge_emb))
        return _global_mean_pool_onnx(node_emb, batch)

    def forward(
            self,
            state_node_ids:     torch.Tensor,            # [Ns]  int64
            state_edge_index:   torch.Tensor,            # [2, Es]
            state_edge_attr:    torch.Tensor,            # [Es, 1]  int64
            state_batch:        torch.Tensor,            # [Ns]
            goal_node_ids:      torch.Tensor | None = None,
            goal_edge_index:    torch.Tensor | None = None,
            goal_edge_attr:     torch.Tensor | None = None,
            goal_batch:         torch.Tensor | None = None,
            depth:              torch.Tensor | None = None,
    ) -> torch.Tensor:
        rep = self._encode_raw(
            state_node_ids, state_edge_index, state_edge_attr, state_batch,
            self.core.state_conv1, self.core.state_conv2,
        )

        if (
                self.core.use_goal
                and goal_node_ids is not None
                and goal_node_ids.numel() > 0
        ):
            goal_emb = self._encode_raw(
                goal_node_ids, goal_edge_index, goal_edge_attr, goal_batch,
                self.core.goal_conv1, self.core.goal_conv2,
            )
            rep = torch.cat([rep, goal_emb], dim=1)

        if self.core.use_depth:
            if depth is None or depth.numel() == 0:
                depth = torch.zeros(rep.size(0), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            rep = torch.cat([rep, depth], dim=1)

        out = self.core.regressor(rep).squeeze(1)
        return out.clamp(min=self.core.min_value, max=1 - self.core.min_value)


class OnnxDistanceEstimatorWrapperBits(nn.Module):
    """
    ONNX wrapper for the bitmask node-feature path (bit_input is not None).

    Node IDs are represented as binary vectors of length bit_input, so there
    are no floating-point precision concerns.
    """

    def __init__(self, core: DistanceEstimator):
        super().__init__()
        assert core.bit_input is not None, (
            "OnnxDistanceEstimatorWrapperBits requires a model trained with "
            "bitmask node features (bit_input != None).  Use "
            "OnnxDistanceEstimatorWrapperIds for scalar-id models."
        )
        self.core = core

    def _encode_raw(
            self,
            node_bits_u8: torch.Tensor,   # [N, bit_input]  uint8 / bool
            edge_index:   torch.Tensor,   # [2, E]
            edge_attr:    torch.Tensor,   # [E, 1]
            batch:        torch.Tensor,   # [N]
            conv1:        GINEConv,
            conv2:        GINEConv,
    ) -> torch.Tensor:
        node_emb = self.core.id_mlp(node_bits_u8.to(torch.float32))
        edge_emb = self.core.edge_mlp(edge_attr.float())
        node_emb = F.relu(conv1(node_emb, edge_index, edge_emb))
        node_emb = F.relu(conv2(node_emb, edge_index, edge_emb))
        return _global_mean_pool_onnx(node_emb, batch)

    def forward(
            self,
            state_node_bits_u8: torch.Tensor,            # [Ns, bit_input]
            state_edge_index:   torch.Tensor,             # [2, Es]
            state_edge_attr:    torch.Tensor,             # [Es, 1]
            state_batch:        torch.Tensor,             # [Ns]
            goal_node_bits_u8:  torch.Tensor | None = None,
            goal_edge_index:    torch.Tensor | None = None,
            goal_edge_attr:     torch.Tensor | None = None,
            goal_batch:         torch.Tensor | None = None,
            depth:              torch.Tensor | None = None,
    ) -> torch.Tensor:
        rep = self._encode_raw(
            state_node_bits_u8, state_edge_index, state_edge_attr, state_batch,
            self.core.state_conv1, self.core.state_conv2,
        )

        if (
                self.core.use_goal
                and goal_node_bits_u8 is not None
                and goal_node_bits_u8.numel() > 0
        ):
            goal_emb = self._encode_raw(
                goal_node_bits_u8, goal_edge_index, goal_edge_attr, goal_batch,
                self.core.goal_conv1, self.core.goal_conv2,
            )
            rep = torch.cat([rep, goal_emb], dim=1)

        if self.core.use_depth:
            if depth is None or depth.numel() == 0:
                depth = torch.zeros(rep.size(0), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            rep = torch.cat([rep, depth], dim=1)

        out = self.core.regressor(rep).squeeze(1)
        return out.clamp(min=self.core.min_value, max=1 - self.core.min_value)