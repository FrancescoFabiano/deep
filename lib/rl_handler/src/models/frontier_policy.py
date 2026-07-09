"""Deployed network architecture (DO NOT rename exported symbols).

FrontierPolicyNetwork is the model behind every frontier_policy_<F>.onnx the
C++ planner loads (FringeEvalRL): GNN encoder with in-graph HASHED int64
normalization + per-fringe-slot score head with optional frontier context.
Used by the offline DQN trainer (src/offline/dqn.py). Checkpoint configs
reference the constructor signature; the ONNX contract depends on the
forward semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    RGCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from typing import Optional

DATASET_TYPE_HASHED = "HASHED"
DATASET_TYPE_MAPPED = "MAPPED"
DATASET_TYPE_BITMASK = "BITMASK"
VALID_DATASET_TYPES = {
    DATASET_TYPE_HASHED,
    DATASET_TYPE_MAPPED,
    DATASET_TYPE_BITMASK,
}
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
I64_ABS_MIN_FLOAT = float(1 << 63)
I64_MAX_FLOAT = float(I64_MAX)


VALID_CONTEXT_MODES = {"mean_pool", "self_attention", "none"}


def _build_mlp(in_dim: int, hidden_dim: int, depth: int, out_dim: int) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for _ in range(max(1, depth)):
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.ReLU())
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


class FringeAttention(nn.Module):
    """Multi-head self-attention over fringe state embeddings.

    Each state attends to all other states in ITS fringe (not across fringes in
    the training batch).  Variable-size fringes are padded into ``[B, max_K, D]``
    with a key-padding mask so padding slots are ignored by attention.

    Two entry points:
      * ``forward(z, candidate_batch)`` — batched training path.  ``candidate_batch``
        maps each pooled state to its fringe index; the disconnected fringes are
        padded/unpadded around the transformer stack.
      * ``forward_single(z, mask)`` — single-fringe inference/ONNX path (batch=1).
        No ``_pad`` loop, so it traces cleanly; ``mask`` (1=active, 0=padding)
        becomes the key-padding mask.  This is the path the exported ONNX runs,
        because the C++ contract feeds one fringe with ``candidate_batch=None``.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.n_heads,
                    dim_feedforward=self.hidden_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )

    def forward(self, z: torch.Tensor, candidate_batch: torch.Tensor) -> torch.Tensor:
        """Batched training path.

        Args:
            z: ``[N, D]`` flat state embeddings (all fringes concatenated).
            candidate_batch: ``[N]`` fringe index per state (0..B-1).

        Returns:
            ``[N, D]`` — each state enriched by attending to its fringe-mates.
        """
        padded, key_padding, positions = self._pad(z, candidate_batch)
        # src_key_padding_mask convention: True = IGNORE that position.
        for layer in self.layers:
            padded = layer(padded, src_key_padding_mask=key_padding)
        return self._unpad(padded, candidate_batch, positions)

    def forward_single(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single-fringe inference/ONNX path: ``z`` is ``[M, D]`` for one fringe.

        ``mask`` (1=active, 0=padding) is turned into a ``[1, M]`` key-padding
        mask (True = ignore).  No padding assembly, so this is trace-friendly for
        ONNX export at batch=1.
        """
        z_seq = z.unsqueeze(0)  # [1, M, D]
        key_padding = None
        if mask is not None:
            key_padding = (~mask.to(dtype=torch.bool).view(1, -1))
        for layer in self.layers:
            z_seq = layer(z_seq, src_key_padding_mask=key_padding)
        return z_seq.squeeze(0)

    def _pad(self, z: torch.Tensor, candidate_batch: torch.Tensor):
        """Flat ``[N, D]`` -> padded ``[B, max_K, D]`` + key-padding mask ``[B, max_K]``.

        ``candidate_batch`` is produced by ``repeat_interleave`` in the packers,
        so it is sorted with contiguous per-fringe groups.  The intra-fringe
        position of every state is therefore computed vectorially (argsort-based,
        robust even if the grouping were ever reordered) — no per-element Python
        loop / CUDA sync.
        """
        device = z.device
        N = z.size(0)
        D = z.size(1)
        B = int(candidate_batch.max().item()) + 1
        sizes = torch.bincount(candidate_batch, minlength=B)
        max_K = int(sizes.max().item())

        # position of each state within its fringe (0..size_b-1)
        offsets = torch.cumsum(sizes, 0) - sizes  # [B] group start rank
        order = torch.argsort(candidate_batch, stable=True)
        ranks_sorted = (
            torch.arange(N, device=device) - offsets[candidate_batch[order]]
        )
        positions = torch.empty(N, dtype=torch.long, device=device)
        positions[order] = ranks_sorted

        padded = z.new_zeros(B, max_K, D)
        key_padding = torch.ones(B, max_K, dtype=torch.bool, device=device)  # True=ignore
        padded[candidate_batch, positions] = z
        key_padding[candidate_batch, positions] = False
        return padded, key_padding, positions

    def _unpad(
        self,
        padded: torch.Tensor,
        candidate_batch: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Padded ``[B, max_K, D]`` -> flat ``[N, D]`` (gather back each state)."""
        return padded[candidate_batch, positions]


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gine",
        dataset_type: str = DATASET_TYPE_HASHED,
        edge_emb_dim: int = 32,
        num_edge_labels: int = 256,
        num_node_labels: int = 4096,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.dataset_type = str(dataset_type).upper()
        if self.dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(
                f"Unsupported dataset_type '{dataset_type}'. "
                f"Expected one of {sorted(VALID_DATASET_TYPES)}."
            )
        self.node_input_dim = int(node_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.edge_emb_dim = int(edge_emb_dim)
        self.num_edge_labels = max(1, int(num_edge_labels))
        self.num_node_labels = max(1, int(num_node_labels))
        self.input_proj = nn.Linear(self.node_input_dim, self.hidden_dim)
        self.input_proj_refine = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_label_embedding = nn.Embedding(self.num_node_labels, self.hidden_dim)
        self.edge_embedding = nn.Embedding(self.num_edge_labels, self.edge_emb_dim)
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_emb_dim, self.edge_emb_dim),
            nn.ReLU(),
            nn.Linear(self.edge_emb_dim, self.edge_emb_dim),
        )
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if self.conv_type == "gine":
                self.layers.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        edge_dim=self.edge_emb_dim,
                    )
                )
            elif self.conv_type == "rgcn":
                self.layers.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=self.num_edge_labels)
                )
            elif self.conv_type == "gcn":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

    def _prepare_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        is_tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        x = node_features
        if x.dim() == 1:
            x = x.view(-1, 1)

        if self.dataset_type == DATASET_TYPE_HASHED:
            if (not is_tracing) and x.size(-1) != 1:
                raise ValueError(
                    "HASHED dataset expects scalar signed int64 IDs with shape "
                    f"[N] or [N, 1], got {tuple(x.shape)}."
                )
            raw = x[:, 0].to(torch.float64)
            if not is_tracing:
                if bool((raw < float(I64_MIN)).any().item()) or bool(
                    (raw > float(I64_MAX)).any().item()
                ):
                    raise ValueError(
                        f"HASHED scalar IDs must be in int64 range [{I64_MIN}, {I64_MAX}]."
                    )

            # Signed-aware normalization in-model (kept inside ONNX at export time):
            # [-2^63, -1] -> [-1, 0), [0, 2^63-1] -> [0, 1].
            x = torch.where(
                raw >= 0.0,
                raw / I64_MAX_FLOAT,
                raw / I64_ABS_MIN_FLOAT,
            ).to(torch.float32).view(-1, 1)
            x = x.clamp(-1.0, 1.0)
        elif self.dataset_type in {DATASET_TYPE_MAPPED, DATASET_TYPE_BITMASK}:
            x = x.to(torch.float32)
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        if (not is_tracing) and x.size(-1) != self.node_input_dim:
            raise ValueError(
                f"node feature dim mismatch: got {x.size(-1)} expected {self.node_input_dim}"
            )
        return x

    def _edge_label_ids(
        self,
        edge_attr: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        is_tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if (not is_tracing) and edge_attr.dim() == 2 and edge_attr.size(1) != 1:
            raise ValueError(
                f"edge_attr must be categorical IDs with shape [E] or [E, 1], got {tuple(edge_attr.shape)}."
            )
        if (not is_tracing) and edge_attr.dim() not in {1, 2}:
            raise ValueError(
                f"edge_attr must be 1D/2D categorical IDs, got rank {edge_attr.dim()}."
            )

        raw_ids = edge_attr.reshape(-1)
        edge_ids = raw_ids.to(device=device, dtype=torch.long)

        # Deterministic OOV-safe bucketing applied in-model (and therefore exported to ONNX):
        # bucket = abs(edge_id) % K.
        min_i64 = torch.iinfo(torch.long).min
        safe_edge_ids = torch.where(
            edge_ids == min_i64,
            torch.zeros_like(edge_ids),
            edge_ids,
        )
        return torch.remainder(torch.abs(safe_edge_ids), self.num_edge_labels)

    def _node_label_ids(
        self,
        node_features: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        is_tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        x = node_features
        if x.dim() == 1:
            x = x.view(-1, 1)
        if (not is_tracing) and x.size(0) == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)

        if self.dataset_type == DATASET_TYPE_HASHED:
            if (not is_tracing) and x.size(-1) != 1:
                raise ValueError(
                    f"HASHED dataset expects scalar signed int64 IDs, got shape {tuple(x.shape)}."
                )
            raw = x[:, 0].to(device=device, dtype=torch.long)
            return torch.remainder(raw, self.num_node_labels).view(-1)

        if self.dataset_type == DATASET_TYPE_MAPPED:
            if (not is_tracing) and x.size(-1) != 1:
                raise ValueError(
                    f"{self.dataset_type} dataset expects scalar node IDs for label embedding, got shape {tuple(x.shape)}."
                )
            raw = x[:, 0].to(device=device, dtype=torch.long)
            return torch.remainder(raw, self.num_node_labels).view(-1)

        if self.dataset_type == DATASET_TYPE_BITMASK:
            bits = (x > 0.5).to(device=device, dtype=torch.long)
            if bits.dim() != 2:
                bits = bits.view(bits.size(0), -1)
            n_bits = int(bits.size(1))
            if n_bits == 0:
                return torch.zeros((bits.size(0),), dtype=torch.long, device=device)
            max_bits = min(n_bits, 31)
            weights = (2 ** torch.arange(max_bits, device=device, dtype=torch.long)).view(
                1, -1
            )
            label_ids = (bits[:, :max_bits] * weights).sum(dim=1)
            if n_bits > max_bits:
                tail = bits[:, max_bits:].sum(dim=1) * 131
                label_ids = label_ids + tail
            return torch.remainder(label_ids, self.num_node_labels).view(-1)

        raise ValueError(f"Unsupported dataset_type for node label embedding: {self.dataset_type}")

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        raw_nodes = node_features
        if raw_nodes.dim() == 1:
            raw_nodes = raw_nodes.view(-1, 1)
        x = self._prepare_node_features(raw_nodes)
        x = self.input_proj(x)
        x = self.input_proj_refine(x)
        node_ids = self._node_label_ids(raw_nodes, device=edge_index.device)
        x = x + self.node_label_embedding(node_ids)
        edge_ids = self._edge_label_ids(edge_attr, device=edge_index.device)
        shared_edge_attr = None
        if self.conv_type == "gine":
            shared_edge_attr = self.edge_embedding(edge_ids)
            shared_edge_attr = self.edge_proj(shared_edge_attr)

        for conv in self.layers:
            if isinstance(conv, GINEConv):
                if shared_edge_attr is None:
                    raise RuntimeError("Expected shared edge attributes for GINEConv.")
                x = F.relu(conv(x, edge_index, shared_edge_attr))
            elif isinstance(conv, RGCNConv):
                x = F.relu(conv(x, edge_index, edge_ids))
            else:
                x = F.relu(conv(x, edge_index))
        return x


class FrontierPolicyNetwork(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        conv_type: str = "gine",
        pooling_type: str = "mean",
        dataset_type: str = DATASET_TYPE_HASHED,
        edge_emb_dim: int = 32,
        num_edge_labels: int = 256,
        num_node_labels: int = 4096,
        use_global_context: Optional[bool] = None,
        context_mode: Optional[str] = None,
        attn_heads: int = 4,
        attn_layers: int = 1,
        mlp_depth: int = 2,
        use_goal_separate_input: bool = False,
    ):
        super().__init__()
        # Resolve the frontier-context mode.  `context_mode` is authoritative when
        # given (new checkpoints carry it); otherwise fall back to the deprecated
        # `use_global_context` bool (True->mean_pool, False->none) so old configs /
        # callers keep working; default (both unset) is mean_pool.
        if context_mode is None:
            if use_global_context is None:
                context_mode = "mean_pool"
            else:
                context_mode = "mean_pool" if use_global_context else "none"
        context_mode = str(context_mode)
        if context_mode not in VALID_CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {sorted(VALID_CONTEXT_MODES)}, "
                f"got {context_mode!r}."
            )
        self.context_mode = context_mode
        self.attn_heads = int(attn_heads)
        self.attn_layers = int(attn_layers)
        # Kept for backward-compat serialization / callers that read it: True iff
        # the head concatenates mean-pool frontier context.
        self.use_global_context = context_mode == "mean_pool"
        self.pooling_type = pooling_type
        self.use_goal_separate_input = use_goal_separate_input
        self.dataset_type = str(dataset_type).upper()
        self.edge_emb_dim = int(edge_emb_dim)
        self.num_edge_labels = max(1, int(num_edge_labels))
        self.num_node_labels = max(1, int(num_node_labels))
        self.encoder = GNNEncoder(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
            dataset_type=self.dataset_type,
            edge_emb_dim=self.edge_emb_dim,
            num_edge_labels=self.num_edge_labels,
            num_node_labels=self.num_node_labels,
        )
        # mean_pool and self_attention both concat a second hidden-width vector
        # ([z_i || ctx_i] resp. [z_i || z'_i]), so head_in matches for a fair A/B;
        # none scores each state from its own embedding only.
        if self.context_mode in ("mean_pool", "self_attention"):
            head_in = hidden_dim * 2
        else:  # "none"
            head_in = hidden_dim
        if self.use_goal_separate_input:
            head_in += hidden_dim
        self.fringe_attention: Optional[FringeAttention] = None
        if self.context_mode == "self_attention":
            self.fringe_attention = FringeAttention(
                hidden_dim=hidden_dim,
                n_heads=self.attn_heads,
                n_layers=self.attn_layers,
            )
        self.policy_head = _build_mlp(
            in_dim=head_in,
            hidden_dim=hidden_dim,
            depth=mlp_depth,
            out_dim=1,
        )

    def _pool_nodes(
        self,
        node_embeddings: torch.Tensor,
        membership: torch.Tensor,
        pool_node_index: Optional[torch.Tensor] = None,
        pool_membership: Optional[torch.Tensor] = None,
        expected_size: Optional[int] = None,
    ) -> torch.Tensor:
        is_tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if membership.dim() != 1:
            raise ValueError(f"membership must be 1D, got shape {tuple(membership.shape)}")
        if expected_size is not None and int(expected_size) < 0:
            raise ValueError(f"expected_size must be non-negative, got {expected_size}")

        if (pool_node_index is None) != (pool_membership is None):
            raise ValueError(
                "pool_node_index and pool_membership must be provided together."
            )
        if pool_node_index is not None and pool_membership is not None:
            if pool_node_index.dim() != 1 or pool_membership.dim() != 1:
                raise ValueError(
                    "pool_node_index and pool_membership must be 1D tensors."
                )
            if pool_node_index.numel() != pool_membership.numel():
                raise ValueError(
                    "pool_node_index and pool_membership must have identical lengths."
                )
            if pool_node_index.numel() > 0:
                if int(pool_node_index.min().item()) < 0:
                    raise ValueError("pool_node_index contains negative indices.")
                if int(pool_node_index.max().item()) >= int(node_embeddings.size(0)):
                    raise ValueError(
                        f"pool_node_index out of bounds: max={int(pool_node_index.max().item())} "
                        f"n_nodes={int(node_embeddings.size(0))}"
                    )
            if pool_membership.numel() > 0 and int(pool_membership.min().item()) < 0:
                raise ValueError("pool_membership contains negative indices.")
            x = node_embeddings[pool_node_index]
            m = pool_membership
        else:
            mask = membership >= 0
            x = node_embeddings[mask]
            m = membership[mask]
        if m.dim() != 1:
            raise ValueError(f"pool membership vector must be 1D, got shape {tuple(m.shape)}")
        if (not is_tracing) and x.size(0) != m.numel():
            raise ValueError(
                f"Pooled tensor and membership mismatch: x.size(0)={x.size(0)} m.numel()={m.numel()}"
            )

        size = int(expected_size) if expected_size is not None else None
        if self.pooling_type == "mean":
            pooled = global_mean_pool(x, m, size=size)
        elif self.pooling_type == "sum":
            pooled = global_add_pool(x, m, size=size)
        elif self.pooling_type == "max":
            pooled = global_max_pool(x, m, size=size)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")
        if expected_size is not None and pooled.size(0) != int(expected_size):
            raise AssertionError(
                f"Pooled candidate size mismatch: pooled.size(0)={pooled.size(0)} "
                f"expected_size={int(expected_size)}"
            )
        return pooled

    def _pool_goal(
        self,
        goal_node_embeddings: torch.Tensor,
        goal_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(goal_node_embeddings, goal_batch)
        if self.pooling_type == "sum":
            return global_add_pool(goal_node_embeddings, goal_batch)
        if self.pooling_type == "max":
            return global_max_pool(goal_node_embeddings, goal_batch)
        raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

    def _contextualize(
        self,
        z: torch.Tensor,
        candidate_batch: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.context_mode == "none":
            return z
        if self.context_mode == "self_attention":
            # Inference/ONNX contract feeds one fringe with candidate_batch=None;
            # use the mask as the key-padding mask (single-fringe, trace-friendly).
            if candidate_batch is None:
                z_att = self.fringe_attention.forward_single(z, mask)
                return torch.cat([z, z_att], dim=-1)
            if candidate_batch.dim() != 1:
                raise ValueError(
                    f"candidate_batch must be 1D, got shape {tuple(candidate_batch.shape)}"
                )
            if candidate_batch.numel() == 0:
                return torch.cat([z, z.new_zeros(z.shape)], dim=-1)
            if int(candidate_batch.min().item()) < 0:
                raise ValueError("candidate_batch contains negative frontier indices.")
            z_att = self.fringe_attention(z, candidate_batch)
            return torch.cat([z, z_att], dim=-1)
        # context_mode == "mean_pool" (existing behavior, unchanged below)
        if candidate_batch is None:
            if mask is not None:
                is_tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
                mask_vec = mask.to(device=z.device, dtype=torch.bool).view(-1)
                if (not is_tracing) and mask_vec.numel() != z.size(0):
                    raise ValueError(
                        "mask and pooled candidates must have identical lengths when "
                        "candidate_batch is None."
                    )
                active = mask_vec.to(dtype=z.dtype).unsqueeze(-1)
                denom = active.sum(dim=0, keepdim=True).clamp_min(1.0)
                ctx_vec = (z * active).sum(dim=0, keepdim=True) / denom
                ctx = ctx_vec.expand_as(z)
                return torch.cat([z, ctx], dim=-1)
            ctx = z.mean(dim=0, keepdim=True).expand_as(z)
            return torch.cat([z, ctx], dim=-1)
        if candidate_batch.dim() != 1:
            raise ValueError(
                f"candidate_batch must be 1D, got shape {tuple(candidate_batch.shape)}"
            )
        if candidate_batch.numel() > 0 and int(candidate_batch.min().item()) < 0:
            raise ValueError("candidate_batch contains negative frontier indices.")

        assert z.size(0) == candidate_batch.numel(), (
            f"Mismatch before global_mean_pool: "
            f"z.size(0)={z.size(0)} candidate_batch.numel()={candidate_batch.numel()} "
            f"candidate_batch.min()={int(candidate_batch.min()) if candidate_batch.numel() > 0 else 'NA'} "
            f"candidate_batch.max()={int(candidate_batch.max()) if candidate_batch.numel() > 0 else 'NA'}"
        )

        if candidate_batch.numel() == 0:
            ctx = z.new_zeros(z.shape)
            return torch.cat([z, ctx], dim=-1)
        n_frontiers = int(candidate_batch.max().item()) + 1
        ctx_per_frontier = global_mean_pool(z, candidate_batch, size=n_frontiers)
        if int(candidate_batch.max().item()) >= ctx_per_frontier.size(0):
            raise ValueError(
                f"candidate_batch index out of range: max={int(candidate_batch.max().item())} "
                f"n_frontiers={ctx_per_frontier.size(0)}"
            )
        ctx = ctx_per_frontier[candidate_batch]
        return torch.cat([z, ctx], dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        candidate_batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        goal_node_features: Optional[torch.Tensor] = None,
        goal_edge_index: Optional[torch.Tensor] = None,
        goal_edge_attr: Optional[torch.Tensor] = None,
        goal_batch: Optional[torch.Tensor] = None,
        pool_node_index: Optional[torch.Tensor] = None,
        pool_membership: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_emb = self.encoder(node_features, edge_index, edge_attr)
        expected_num_candidates = (
            int(candidate_batch.numel()) if candidate_batch is not None else None
        )
        z = self._pool_nodes(
            node_emb,
            membership,
            pool_node_index=pool_node_index,
            pool_membership=pool_membership,
            expected_size=expected_num_candidates,
        )
        h = self._contextualize(z, candidate_batch, mask=mask)
        if self.use_goal_separate_input:
            if (
                goal_node_features is not None
                and goal_edge_index is not None
                and goal_edge_attr is not None
                and goal_batch is not None
            ):
                goal_node_emb = self.encoder(
                    goal_node_features,
                    goal_edge_index,
                    goal_edge_attr,
                )
                goal_emb = self._pool_goal(goal_node_emb, goal_batch)
                if candidate_batch is None:
                    goal_per_candidate = goal_emb.mean(dim=0, keepdim=True).expand(h.size(0), -1)
                else:
                    goal_per_candidate = goal_emb[candidate_batch]
            else:
                goal_per_candidate = torch.zeros(
                    (h.size(0), self.encoder.input_proj.out_features),
                    device=h.device,
                    dtype=h.dtype,
                )
            h = torch.cat([h, goal_per_candidate], dim=-1)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(torch.bool), -1e9)
        return logits
