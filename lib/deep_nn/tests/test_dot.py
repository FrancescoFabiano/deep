import pytest
import torch

from conftest import MERGED_DOT
from deep_nn.dot import TO_STATE, parse_dot, separated_view, uint64_ids_to_int64


def test_merged_parse_follows_the_planner_numbering():
    g = parse_dot(MERGED_DOT)
    # nodes by first appearance over the file = GraphNN's symbolic ids
    assert g.node_ids[:4].tolist() == [0, 1, 19, 20]
    assert g.node_ids[4].item() == 7 and g.node_ids[5].item() == -8011962737897461503
    assert g.node_ids[-1].item() == uint64_ids_to_int64([18446744073709551615]).item() == -1
    assert g.edge_index.shape == (2, 10) and g.edge_attr.tolist()[:2] == [2, 5]
    # designated worlds = targets of the label-3 edges (GraphNN::check_tensor_against_dot)
    assert g.pointed_ids.tolist() == g.edge_index[1, g.edge_attr == TO_STATE].tolist() == [5, 6]


def test_separated_view_matches_the_planner_separated_tensors():
    state, goal = separated_view(parse_dot(MERGED_DOT))
    # state: designated worlds first (in epsilon-edge order), then belief order; no epsilon/goal nodes
    assert state.node_ids.tolist() == [-8011962737897461503, -77, 3010661539059282004, -1]
    assert state.pointed_ids.tolist() == [0, 1]
    assert state.edge_attr.tolist() == [7, 8, 7, 8]
    assert state.edge_index.tolist() == [[2, 2, 0, 3], [2, 0, 3, 2]]
    # goal: the subtree only, parent as node 0, no designated worlds
    assert goal.node_ids.tolist() == [1, 19, 20, 7]
    assert goal.edge_index.tolist() == [[0, 1, 2], [1, 2, 3]] and goal.edge_attr.tolist() == [5, 5, 5]
    assert goal.pointed_ids.numel() == 0


def test_separated_generated_file_is_refused():
    with pytest.raises(ValueError, match="not a merged DOT"):
        separated_view(parse_dot('digraph G {\n  5 -> 6 [label="7"];\n}\n'))


def test_grammar_violations_raise():
    with pytest.raises(ValueError):
        parse_dot('digraph G {\n  5 -> 6;\n}\n')          # no label
    with pytest.raises(ValueError):
        parse_dot('  5 -> 6 [label="7"];\n')               # no header


def test_high_bit_ids_fold_like_the_planner_cast():
    assert uint64_ids_to_int64([1 << 63]).item() == -(1 << 63)
    assert torch.equal(uint64_ids_to_int64([-5, 5]), torch.tensor([-5, 5]))
