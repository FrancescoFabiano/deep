from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

TWO_48_MINUS_1 = float(2**48 - 1)


class ResidualBlock(nn.Module):
    """A simple residual block with BatchNorm, ReLU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
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
        # NEW: number of bits for bitmask node IDs; if None, use old scalar id path
        bit_input: Optional[int] = None,
    ):
        super().__init__()
        self.use_goal = use_goal
        self.use_depth = use_depth
        self.min_value = min_value
        self.bit_input = bit_input

        # ───────── node & edge embeddings
        in_features = bit_input if bit_input is not None else 1
        self.id_mlp = nn.Sequential(
            nn.Linear(in_features, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # ───────── GINE encoders
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

        in_dim = hidden_dim * (2 if use_goal else 1) + (1 if use_depth else 0)
        layers = [nn.Linear(in_dim, regressor_hidden_dim), nn.ReLU()]
        for _ in range(regressor_blocks):
            layers.append(ResidualBlock(regressor_hidden_dim, regressor_dropout))
        layers.append(nn.Linear(regressor_hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*layers)

    def _encode(self, graph, conv1, conv2):
        # Prefer bitmask if present; otherwise, fall back to scalar IDs.
        if hasattr(graph, "node_bits") and self.bit_input is not None:
            # node_bits: [N, bit_input], already 0/1
            raw = graph.node_bits.to(torch.float32).to(graph.edge_index.device)
        else:
            # node_names: [N] scalar -> normalize and view as [N,1]
            raw1d = graph.node_names.to(graph.edge_index.device).float()
            raw = (raw1d / TWO_48_MINUS_1).clamp(0.0, 1.0).view(-1, 1)

        x = self.id_mlp(raw)
        e = self.edge_mlp(graph.edge_attr.to(x.device).float())
        x = F.relu(conv1(x, graph.edge_index, e))
        x = F.relu(conv2(x, graph.edge_index, e))
        return global_mean_pool(x, graph.batch.clone())

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = self._encode(batch_dict["state_graph"], self.state_conv1, self.state_conv2)
        if self.use_goal and batch_dict.get("goal_graph") is not None:
            g = self._encode(batch_dict["goal_graph"], self.goal_conv1, self.goal_conv2)
            rep = torch.cat([s, g], dim=1)
        else:
            rep = s

        if self.use_depth:
            d = batch_dict.get("depth")
            if d is None:
                d = torch.zeros(len(rep), 1, dtype=rep.dtype, device=rep.device)
            else:
                d = d.float().view(-1, 1)
            rep = torch.cat([rep, d], dim=1)

        out = self.regressor(rep).squeeze(1)
        return out.clamp(min=self.min_value, max=1 - self.min_value)

    def get_checkpoint(self) -> Dict:
        cfg = {
            "hidden_dim": self.state_conv1.nn[0].out_features,
            "node_emb_dim": self.id_mlp[0].out_features,
            "edge_emb_dim": self.edge_mlp[0].out_features,
            "use_goal": self.use_goal,
            "use_depth": self.use_depth,
            "bit_input": self.bit_input,  # NEW
        }
        return {"state_dict": self.state_dict(), "config": cfg}

    @classmethod
    def load_model(cls, ckpt: Dict):
        cfg = ckpt["config"]
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            use_goal=cfg["use_goal"],
            use_depth=cfg.get("use_depth", True),
            node_emb_dim=cfg["node_emb_dim"],
            edge_emb_dim=cfg["edge_emb_dim"],
            bit_input=cfg.get("bit_input", None),  # NEW
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


class OnnxDistanceEstimatorWrapperIds(nn.Module):
    def __init__(self, core: DistanceEstimator):
        super().__init__()
        assert core.bit_input is None
        self.core = core

    @staticmethod
    def _global_mean(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        G = batch.max() + 1
        oh = torch.nn.functional.one_hot(batch, num_classes=G).float()
        sums = oh.T @ x
        cnts = oh.T @ torch.ones_like(x[:, :1])
        return sums / torch.clamp(cnts, min=1.0)

    def _encode_raw(self, node_ids_f32, edge_index, edge_attr, batch, conv1, conv2):
        x = self.core.id_mlp(
            torch.clamp(node_ids_f32 / TWO_48_MINUS_1, 0, 1).unsqueeze(1)
        )
        e = self.core.edge_mlp(edge_attr.float())
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))
        return self._global_mean(x, batch)

    def forward(
        self,
        s_node_ids_f32,
        s_edge_index,
        s_edge_attr,
        s_batch,
        # goal (optional by export flag)
        g_node_ids_f32=None,
        g_edge_index=None,
        g_edge_attr=None,
        g_batch=None,
        # depth (optional by export flag)
        depth: torch.Tensor | None = None,
    ):
        rep = self._encode_raw(
            s_node_ids_f32,
            s_edge_index,
            s_edge_attr,
            s_batch,
            self.core.state_conv1,
            self.core.state_conv2,
        )

        if (
            self.core.use_goal
            and g_node_ids_f32 is not None
            and g_node_ids_f32.numel() > 0
        ):
            g_emb = self._encode_raw(
                g_node_ids_f32,
                g_edge_index,
                g_edge_attr,
                g_batch,
                self.core.goal_conv1,
                self.core.goal_conv2,
            )
            rep = torch.cat([rep, g_emb], dim=1)

        if self.core.use_depth:
            if depth is None or depth.numel() == 0:
                depth = torch.zeros(rep.size(0), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            rep = torch.cat([rep, depth], dim=1)

        out = self.core.regressor(rep).squeeze(1)
        return out.clamp(min=self.core.min_value, max=1 - self.core.min_value)


class OnnxDistanceEstimatorWrapperBits(nn.Module):
    def __init__(self, core: DistanceEstimator):
        super().__init__()
        assert core.bit_input is not None
        self.core = core

    @staticmethod
    def _global_mean(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        G = batch.max() + 1
        oh = torch.nn.functional.one_hot(batch, num_classes=G).float()
        sums = oh.T @ x
        cnts = oh.T @ torch.ones_like(x[:, :1])
        return sums / torch.clamp(cnts, min=1.0)

    def _encode_raw(self, node_bits_u8, edge_index, edge_attr, batch, conv1, conv2):
        x = self.core.id_mlp(node_bits_u8.to(torch.float32))
        e = self.core.edge_mlp(edge_attr.float())
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))
        return self._global_mean(x, batch)

    def forward(
        self,
        s_node_bits_u8,
        s_edge_index,
        s_edge_attr,
        s_batch,
        g_node_bits_u8=None,
        g_edge_index=None,
        g_edge_attr=None,
        g_batch=None,
        depth: torch.Tensor | None = None,
    ):
        rep = self._encode_raw(
            s_node_bits_u8,
            s_edge_index,
            s_edge_attr,
            s_batch,
            self.core.state_conv1,
            self.core.state_conv2,
        )

        if (
            self.core.use_goal
            and g_node_bits_u8 is not None
            and g_node_bits_u8.numel() > 0
        ):
            g_emb = self._encode_raw(
                g_node_bits_u8,
                g_edge_index,
                g_edge_attr,
                g_batch,
                self.core.goal_conv1,
                self.core.goal_conv2,
            )
            rep = torch.cat([rep, g_emb], dim=1)

        if self.core.use_depth:
            if depth is None or depth.numel() == 0:
                depth = torch.zeros(rep.size(0), 1, dtype=rep.dtype, device=rep.device)
            else:
                depth = depth.float().view(-1, 1)
            rep = torch.cat([rep, depth], dim=1)

        out = self.core.regressor(rep).squeeze(1)
        return out.clamp(min=self.core.min_value, max=1 - self.core.min_value)
