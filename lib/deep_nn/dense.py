"""Scatter-free (dense one-hot) aggregation for ONNX export.

onnxruntime's CPU ``ScatterElements(reduction=add)`` -- what PyG's sum aggregation,
``global_*_pool`` and ``PointedEmbedding`` trace to -- is a scalar loop and took
45-70% of a single-thread evaluation (lib/deep_nn/bench_onnx.py, 2026-10-02).
Every one of those reductions is ``onehot @ values``: a membership matrix built
in-graph from the index tensor with ``Equal`` + ``Cast``, then a ``MatMul`` that
runs in the BLAS kernels. The dense form costs rows x columns instead of
columns, so it pays off for small graphs (gossip: 33 nodes x 430 edges) and not
for huge ones (muddy-child: ~460 x ~9000) -- an export-time choice, measured per
domain, never a training-time one. Values are identical up to float rounding;
``contract.check_parity`` runs against the scatter model at export.

``set_dense_aggregation(model, True)`` swaps every ``GINEConv`` for ``DenseGINE``
(same submodules, same weights) and flips the ``dense_aggregation`` flag on the
blocks that pool or mark (they consult it in forward). Inputs, outputs and the
C++ contract are unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv


def onehot_rows(index: torch.Tensor, n_rows, dtype: torch.dtype) -> torch.Tensor:
    """[n_rows, len(index)] with a 1 where row == index[col]; exports as Equal + Cast."""
    rows = torch.arange(n_rows, device=index.device)      # n_rows: int, or a traced 0-d size
    return (rows.unsqueeze(1) == index.unsqueeze(0)).to(dtype)


def dense_scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """sum of src[j] into out[index[j]] -- scatter_add(0) as onehot @ src. src [E, D] -> [n, D].
    Cost n x E: fine for one graph, quadratic in the batch for F packed slots."""
    return onehot_rows(index, n, src.dtype) @ src


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, 0) - x


def blocked_scatter_sum(src: torch.Tensor, index: torch.Tensor, membership: torch.Tensor,
                        n_slots: int) -> torch.Tensor:
    """`dense_scatter_sum` per slot: the F slots' graphs are aggregated with a batched
    [F, Nmax, Emax] one-hot, so the cost is F x Nmax x Emax (linear in F) instead of
    (F Nmax) x (F Emax). Padding rows are gathered from an appended zero row -- no
    scatter anywhere.

    ASSUMES the planner's packing (pack_fringe / FringeEvalRL::fringe_to_tensor_minimal):
    the nodes of slot f are contiguous in slot order, and so are its edges. Both hold
    by construction of the contract; `contract.check_parity` at export verifies the
    result on real feeds.

    src [E, D] messages, index [E] target node of each message (global ids),
    membership [N] slot of each node, n_slots = the number of OCCUPIED slots (an int,
    or a traced 0-d tensor such as membership.max() + 1 so that a half-empty fringe
    costs K x Nmax x Emax, not F x Nmax x Emax).
    """
    N, E, D = membership.numel(), src.size(0), src.size(1)
    dev, dtype = src.device, src.dtype
    slot_e = membership[index]                                     # [E]
    oh_n = onehot_rows(membership, n_slots, torch.int64)           # [F, N]
    oh_e = onehot_rows(slot_e, n_slots, torch.int64)               # [F, E]
    cnt_n, cnt_e = oh_n.sum(dim=1), oh_e.sum(dim=1)                # [F]
    start_n, start_e = _exclusive_cumsum(cnt_n), _exclusive_cumsum(cnt_e)
    n_max, e_max = cnt_n.max(), cnt_e.max()
    j = torch.arange(n_max, device=dev)                            # [Nmax]
    k = torch.arange(e_max, device=dev)                            # [Emax]
    idx_e = start_e.unsqueeze(1) + k.unsqueeze(0)                  # [F, Emax]
    valid_e = k.unsqueeze(0) < cnt_e.unsqueeze(1)
    idx_e = torch.where(valid_e, idx_e, torch.full_like(idx_e, E))  # padding -> the zero row
    src_pad = torch.cat([src, src.new_zeros(1, D)], dim=0)          # [E + 1, D]
    msg_b = src_pad[idx_e]                                          # [F, Emax, D]
    dst_local = index - start_n[slot_e]                             # [E] node index within its slot
    dst_pad = torch.cat([dst_local, dst_local.new_full((1,), -1)])[idx_e]   # [F, Emax], -1 = padding
    a = (j.view(1, -1, 1) == dst_pad.unsqueeze(1)).to(dtype)        # [F, Nmax, Emax]
    agg_b = torch.matmul(a, msg_b)                                  # [F, Nmax, D]
    flat = membership * n_max + (torch.arange(N, device=dev) - start_n[membership])
    return agg_b.reshape(-1, D)[flat]                               # [N, D]


def dense_add_pool(x: torch.Tensor, batch: torch.Tensor, size) -> torch.Tensor:
    return onehot_rows(batch, size, x.dtype) @ x


def dense_mean_pool(x: torch.Tensor, batch: torch.Tensor, size) -> torch.Tensor:
    a = onehot_rows(batch, size, x.dtype)
    return (a @ x) / a.sum(dim=1, keepdim=True).clamp(min=1)


def pool_size(batch: torch.Tensor, size=None) -> int:
    """PyG's rule: ``size`` when given, else int(batch.max()) + 1 -- a Python int, so a
    trace bakes the dummy's F in, exactly as global_*_pool does (output length F even
    when fewer slots are occupied)."""
    return int(size) if size is not None else int(batch.max()) + 1


class DenseGINE(GINEConv):
    """A trained ``GINEConv`` with the sum aggregation as a dense matmul.

    Subclass so ``isinstance(conv, GINEConv)`` dispatch in the models still holds;
    ``nn``, ``lin`` and ``eps`` are the original modules/tensors, not copies.
    """

    def __init__(self, conv: GINEConv):
        nn.Module.__init__(self)
        self.nn = conv.nn
        self.lin = conv.lin
        self.eps = conv.eps
        self.initial_eps = getattr(conv, "initial_eps", 0.0)
        self.blocking = None        # (membership, n_slots) set by the caller for the blocked form

    def forward(self, x, edge_index, edge_attr=None, size=None):   # noqa: D102 - mirrors GINEConv
        src, dst = edge_index[0], edge_index[1]
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        msg = (x[src] + edge_attr).relu()                     # GINEConv.message
        if self.blocking is not None:
            out = blocked_scatter_sum(msg, dst, *self.blocking)
        else:
            out = dense_scatter_sum(msg, dst, x.size(0))      # aggr="add"
        out = out + (1 + self.eps) * x
        return self.nn(out)


def set_blocking(model: nn.Module, membership) -> None:
    """Tell every DenseGINE of `model` the slot layout of the graph it is about to see
    (export-time; the models call this at the top of their encoder when dense). The
    occupied-slot count is membership.max() + 1, left as a tensor so the trace keeps
    it dynamic. `membership=None` returns the convs to the plain dense form."""
    for mod in model.modules():
        if isinstance(mod, DenseGINE):
            mod.blocking = None if membership is None else (membership, membership.max() + 1)


def set_dense_aggregation(model: nn.Module, enabled: bool = True, blocked: bool = False) -> nn.Module:
    """Swap GINEConv <-> DenseGINE in place and set ``dense_aggregation`` (and
    ``dense_blocked``) on every module that defines them. Returns ``model``.

    Measured on hard/gossip (33 nodes, 430 edges per state; CPU, 1 thread):
      scatter        1.41 ms/state at batch 1, 1.90 at a full batch of 16
      dense          0.83 at batch 1 (-41%), 3.62 at 16 (one-hot is (F N) x (F E))
      dense blocked  1.44 at batch 1, 1.31 at 16 (-31%; ~0.6 ms of fixed op overhead)
    so: plain dense for the per-state and F=1 exports, blocked for F >= 4."""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if enabled and type(child) is GINEConv:
                setattr(mod, child_name, DenseGINE(child))
            elif not enabled and isinstance(child, DenseGINE):
                # GINEConv.__init__ resets the parameters of the nn it is given: build it
                # around an Identity, then hand back the trained submodules untouched.
                conv = GINEConv(nn.Identity(), eps=float(child.initial_eps), train_eps=False)
                conv.nn, conv.lin, conv.eps = child.nn, child.lin, child.eps
                setattr(mod, child_name, conv)
    for mod in model.modules():
        if hasattr(mod, "dense_aggregation"):
            mod.dense_aggregation = bool(enabled)
        if hasattr(mod, "dense_blocked"):
            mod.dense_blocked = bool(enabled and blocked)
    return model
