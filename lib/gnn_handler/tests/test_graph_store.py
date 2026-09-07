"""The graph store: parse once, cache, pack by index -- and stay identical to
the per-row PyG path it replaces."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from conftest import BITS_DOT, GOAL_DOT, STATE_DOT
from src.graph_store import (
    BITMASK_DIM,
    GraphStore,
    dot_paths_in_csv,
    fold_uint64_to_int64,
    parse_dot_file,
    parse_dot_text,
)
from src.utils import load_graph


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_fold_keeps_the_bit_pattern():
    t = fold_uint64_to_int64([0, 2**63, 2**64 - 1, -1, -(2**63)])
    assert t.dtype == torch.int64
    assert t.tolist() == [0, -(2**63), -1, -1, -(2**63)]


def test_parse_matches_the_legacy_pyg_parser(tmp_path):
    p = _write(tmp_path, "s.dot", STATE_DOT)
    flat = parse_dot_text(STATE_DOT)
    legacy = load_graph(p)
    assert torch.equal(flat.node_ids, legacy.node_names)
    assert torch.equal(flat.edge_index, legacy.edge_index)
    assert torch.equal(flat.edge_attr.to(torch.float32).view(-1, 1), legacy.edge_attr)
    assert flat.n_nodes == 3 and flat.edge_attr.tolist() == [8, 8, 9, 9, 10]


def test_bitmask_parse_is_uint8_with_the_planner_width(tmp_path):
    flat = parse_dot_text(BITS_DOT, bitmask=True)
    assert flat.node_bits.dtype == torch.uint8 and flat.node_bits.shape == (2, BITMASK_DIM)
    assert flat.node_bits[0].sum() == 1 and flat.node_bits[1].sum() == BITMASK_DIM


def test_pack_equals_batch_from_data_list(tmp_path):
    s = _write(tmp_path, "s.dot", STATE_DOT)
    g = _write(tmp_path, "g.dot", GOAL_DOT)
    store = GraphStore.parse([s, g, s], verbose=False)
    assert len(store) == 3
    packed = store.pack([2, 1, 0])
    ref = Batch.from_data_list([load_graph(s), load_graph(g), load_graph(s)])
    assert torch.equal(packed.node_names, ref.node_names)
    assert torch.equal(packed.edge_index, ref.edge_index)
    assert torch.equal(packed.edge_attr, ref.edge_attr)
    assert torch.equal(packed.batch, ref.batch)
    assert packed.num_graphs == 3


def test_pack_handles_an_edgeless_graph(tmp_path):
    lone = _write(tmp_path, "lone.dot", "digraph G {\n}\n")
    s = _write(tmp_path, "s.dot", STATE_DOT)
    store = GraphStore.parse([s, lone], verbose=False)
    assert store.graph(1).n_nodes == 0
    packed = store.pack([1, 0])
    assert packed.batch.tolist() == [1, 1, 1]      # lone graph contributes no node
    assert packed.edge_index.min() == 0


def test_cache_roundtrip_and_staleness(tmp_path):
    s = _write(tmp_path, "s.dot", STATE_DOT)
    g = _write(tmp_path, "g.dot", GOAL_DOT)
    cache = tmp_path / "cache" / "x.HASHED.pt"
    a = GraphStore.from_paths([s, g], cache_file=cache, verbose=False)
    assert cache.is_file()
    b = GraphStore.from_paths([s, g], cache_file=cache, verbose=False)
    assert torch.equal(a.node_ids, b.node_ids) and a.paths == b.paths
    # a different path list must NOT be served from the old cache
    c = GraphStore.from_paths([g], cache_file=cache, verbose=False)
    assert len(c) == 1 and c.paths == [g]


def test_planner_feed_is_the_cpp_tensor_set(tmp_path):
    s = _write(tmp_path, "s.dot", STATE_DOT)
    store = GraphStore.parse([s], verbose=False)
    feed = store.planner_feed(0)
    assert list(feed) == ["state_node_ids", "state_edge_index", "state_edge_attr", "state_batch"]
    assert feed["state_node_ids"].dtype.name == "int64" and feed["state_node_ids"].shape == (3,)
    assert feed["state_edge_index"].shape == (2, 5)
    assert feed["state_edge_attr"].dtype.name == "int64" and feed["state_edge_attr"].shape == (5, 1)
    assert feed["state_batch"].tolist() == [0, 0, 0]


def test_concat_dedups_shared_goal(tmp_path):
    s1 = _write(tmp_path, "s1.dot", STATE_DOT)
    s2 = _write(tmp_path, "s2.dot", STATE_DOT.replace('"8"', '"7"'))
    g = _write(tmp_path, "g.dot", GOAL_DOT)
    a = GraphStore.parse([s1, g], verbose=False)
    b = GraphStore.parse([s2, g], verbose=False)
    c = GraphStore.concat([a, b])
    assert c.paths == [s1, g, s2]
    assert c.graph(c.index[g]).edge_attr.tolist() == [5, 5, 5]


def test_dot_paths_in_csv_skips_init_and_dedups(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(
        "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action\n"
        "a/000001.dot,0,0000000004,a/goal_tree.dot,a/init.dot,0\n"
        "a/000002.dot,1,3,a/goal_tree.dot,a/000001.dot,5\n"
    )
    assert dot_paths_in_csv(csv) == ["a/000001.dot", "a/goal_tree.dot", "a/000002.dot"]
    assert dot_paths_in_csv(csv, columns=("File Path",)) == ["a/000001.dot", "a/000002.dot"]


def test_fallback_parser_agrees_on_a_dot_with_node_statements(tmp_path):
    text = 'digraph G {\n  5 [shape=circle];\n  5 -> 6 [label="2"];\n}\n'
    p = _write(tmp_path, "n.dot", text)
    assert parse_dot_text(text) is None
    flat = parse_dot_file(p)
    assert flat.node_ids.tolist() == [5, 6] and flat.edge_attr.tolist() == [2]
