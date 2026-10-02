"""deep_nn.dense: the one-hot forms equal PyG's scatters, plain and blocked, and the
model swap keeps the weights."""
import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from deep_nn.dense import (DenseGINE, blocked_scatter_sum, dense_add_pool, dense_mean_pool,
                           dense_scatter_sum, set_blocking, set_dense_aggregation)
from deep_nn.features import PointedEmbedding


def _packed(seed=0, sizes=(5, 1, 7, 3), d=6):
    """A planner-shaped packed fringe: nodes and edges contiguous per slot."""
    g = torch.Generator().manual_seed(seed)
    membership, edges = [], []
    start = 0
    for f, n in enumerate(sizes):
        membership.append(torch.full((n,), f))
        e = int(torch.randint(0, 3 * n + 1, (1,), generator=g))
        edges.append(torch.randint(0, n, (2, e), generator=g) + start)
        start += n
    membership = torch.cat(membership)
    edge_index = torch.cat(edges, dim=1)
    x = torch.randn(start, d, generator=g)
    edge_attr = torch.randn(edge_index.shape[1], d, generator=g)
    return x, edge_index, edge_attr, membership


def test_dense_and_blocked_scatter_sum_match_scatter_add():
    x, ei, ea, mem = _packed()
    msg = (x[ei[0]] + ea).relu()
    want = torch.zeros_like(x).index_add_(0, ei[1], msg)
    assert torch.allclose(dense_scatter_sum(msg, ei[1], x.size(0)), want, atol=1e-6)
    assert torch.allclose(blocked_scatter_sum(msg, ei[1], mem, 4), want, atol=1e-6)
    assert torch.allclose(blocked_scatter_sum(msg, ei[1], mem, mem.max() + 1), want, atol=1e-6)


def test_dense_pools_match_pyg():
    x, _, _, mem = _packed()
    assert torch.allclose(dense_mean_pool(x, mem, 4), global_mean_pool(x, mem), atol=1e-6)
    assert torch.allclose(dense_add_pool(x, mem, 4), global_add_pool(x, mem), atol=1e-6)
    # a baked F larger than the occupied slots: extra rows are zero (mean of nothing)
    assert dense_mean_pool(x, mem, 6).shape == (6, x.size(1))
    assert torch.all(dense_mean_pool(x, mem, 6)[4:] == 0)


def test_dense_gine_matches_gineconv_plain_and_blocked():
    torch.manual_seed(1)
    x, ei, ea, mem = _packed(d=6)
    conv = GINEConv(nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 8)), edge_dim=6).eval()
    want = conv(x, ei, ea)
    dense = DenseGINE(conv)
    assert torch.allclose(dense(x, ei, ea), want, atol=1e-5)
    dense.blocking = (mem, mem.max() + 1)
    assert torch.allclose(dense(x, ei, ea), want, atol=1e-5)
    assert isinstance(dense, GINEConv)


def test_pointed_embedding_dense_matches_scatter():
    torch.manual_seed(2)
    x = torch.randn(9, 4)
    pe = PointedEmbedding(4)
    with torch.no_grad():
        pe.weight.copy_(torch.randn(4))
    ids = torch.tensor([0, 3, 3, 8])
    want = pe(x, ids)
    pe.dense_aggregation = True
    assert torch.allclose(pe(x, ids), want, atol=1e-6)


def test_set_dense_aggregation_swaps_convs_and_flags_and_restores():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([GINEConv(nn.Linear(4, 4), edge_dim=4) for _ in range(2)])
            self.pointed = PointedEmbedding(4)
            self.dense_aggregation = False
            self.dense_blocked = False
    m = M()
    w = [c.nn.weight.clone() for c in m.convs]
    set_dense_aggregation(m, True, blocked=True)
    assert all(isinstance(c, DenseGINE) for c in m.convs)
    assert m.dense_aggregation and m.dense_blocked and m.pointed.dense_aggregation
    set_blocking(m, torch.tensor([0, 0, 1]))
    assert all(c.blocking is not None for c in m.convs)
    set_dense_aggregation(m, False)
    assert all(type(c) is GINEConv for c in m.convs)
    assert not m.dense_aggregation and not m.pointed.dense_aggregation
    assert all(torch.equal(c.nn.weight, w0) for c, w0 in zip(m.convs, w))
