"""Loader fixes: high-bit HASHED id folding parity + 0-node state-graph refusal."""

from __future__ import annotations

import pytest
import torch

from src.common.no_pyg_loader import fold_uint64_to_int64, load_graph_tensors_no_pyg
from src.offline.encoder import (
    InstanceCache,
    StateGraph,
    pack_fringe,
    parse_dot_fast,
    uint64_ids_to_int64,
)

HIGH_BIT = 1 << 63                       # smallest uint64 the old loader rejected
HIGH = (1 << 64) - 5


def test_fold_matches_fast_parser_folding():
    for v in (0, 1, HIGH_BIT, HIGH, (1 << 63) - 1, -5):
        folded = fold_uint64_to_int64(v, context="test")
        ref = uint64_ids_to_int64([v]).item()
        assert folded == ref, f"fold mismatch for {v}: {folded} != {ref}"


def test_fold_rejects_out_of_range():
    with pytest.raises(ValueError):
        fold_uint64_to_int64(1 << 64, context="test")
    with pytest.raises(ValueError):
        fold_uint64_to_int64(-(1 << 63) - 1, context="test")


def test_no_pyg_loads_high_bit_hashed_dot(tmp_path):
    """~half of uniform 64-bit hashes have the high bit set; the parity
    reference loader must load them exactly like the fast parser."""
    dot = f'digraph G {{\n{HIGH_BIT} -> {HIGH} [label="3"];\n}}\n'
    p = tmp_path / "s.dot"
    p.write_text(dot)
    ref = load_graph_tensors_no_pyg(str(p), "HASHED")
    fast = parse_dot_fast(dot)
    assert torch.equal(ref.node_features.view(-1), fast.node_ids)
    assert torch.equal(ref.edge_attr.view(-1), fast.edge_attr)


def test_pack_fringe_refuses_zero_node_state():
    empty = StateGraph(
        node_ids=torch.zeros((0,), dtype=torch.int64),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        edge_attr=torch.zeros((0,), dtype=torch.int64),
    )
    ok = parse_dot_fast('digraph G {\n1 -> 2 [label="0"];\n}\n')
    cache = InstanceCache([ok, empty])
    with pytest.raises(ValueError, match="0 nodes"):
        pack_fringe(cache, [0, 1], fringe_size=4)
