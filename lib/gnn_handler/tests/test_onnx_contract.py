"""What leaves the trainer must be what the planner feeds.

GraphNN.tpp builds, per state: int64 node ids [N] (uint8 [N,42] under BITMASK),
int64 edge_index [2,E], int64 edge_attr [E,1], int64 all-zero batch [N]; in
separated mode the same four for the goal. It pairs them POSITIONALLY with the
model's declared inputs and reads output[0] as float.
"""

from __future__ import annotations

import onnx
import pytest
import torch

from conftest import BITS_DOT, GOAL_DOT, STATE_DOT
from src.graph_store import BITMASK_DIM, GraphStore
from src.utils import ONNX_INPUTS, assert_onnx_contract, select_model


def _store(tmp_path, bitmask=False):
    if bitmask:
        a = tmp_path / "a.dot"; a.write_text(BITS_DOT)
        b = tmp_path / "b.dot"; b.write_text(BITS_DOT.replace('"3"', '"9"'))
    else:
        a = tmp_path / "a.dot"; a.write_text(STATE_DOT)
        b = tmp_path / "b.dot"; b.write_text(GOAL_DOT)
    return GraphStore.parse([str(a), str(b)], bitmask=bitmask, verbose=False)


@pytest.mark.parametrize("with_goal", [False, True])
def test_export_is_single_file_with_the_planner_inputs(tmp_path, with_goal):
    m = select_model("distance_estimator", use_goal=with_goal, use_depth=False)
    p = tmp_path / "distance_estimator.onnx"
    m.to_onnx(p, with_goal=with_goal, with_depth=False)
    assert p.is_file() and not (tmp_path / "distance_estimator.onnx.data").exists()
    g = onnx.load(str(p))
    assert [i.name for i in g.graph.input] == ONNX_INPUTS[("ids", with_goal)]
    assert [o.name for o in g.graph.output] == ["distance"]
    types = {i.name: i.type.tensor_type.elem_type for i in g.graph.input}
    assert all(t == onnx.TensorProto.INT64 for t in types.values())
    assert g.opset_import[0].version == 18


def test_bitmask_export_declares_uint8_42(tmp_path):
    m = select_model("distance_estimator", use_goal=False, use_depth=False, bitmask=True)
    p = tmp_path / "d.onnx"
    m.to_onnx(p, with_goal=False, with_depth=False)
    g = onnx.load(str(p))
    first = g.graph.input[0]
    assert first.name == "state_node_bits"
    assert first.type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert first.type.tensor_type.shape.dim[1].dim_value == BITMASK_DIM


def test_depth_models_are_refused(tmp_path):
    m = select_model("distance_estimator", use_goal=False, use_depth=True)
    with pytest.raises(ValueError, match="never feeds a depth"):
        m.to_onnx(tmp_path / "d.onnx", with_goal=False, with_depth=True)


@pytest.mark.parametrize("with_goal", [False, True])
def test_onnx_matches_torch_on_planner_style_single_state_feeds(tmp_path, with_goal):
    torch.manual_seed(0)
    m = select_model("distance_estimator", use_goal=with_goal, use_depth=False)
    p = tmp_path / "d.onnx"
    m.to_onnx(p, with_goal=with_goal, with_depth=False)
    store = _store(tmp_path)
    out = m.verify_onnx(p, store, [0, 1], [1, 1] if with_goal else None, atol=1e-4)
    assert out["n_checked"] == 2 and out["max_abs_diff"] <= 1e-4


def test_bitmask_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    m = select_model("distance_estimator", use_goal=False, use_depth=False, bitmask=True)
    p = tmp_path / "d.onnx"
    m.to_onnx(p, with_goal=False, with_depth=False)
    store = _store(tmp_path, bitmask=True)
    out = m.verify_onnx(p, store, [0, 1], None, atol=1e-4)
    assert out["max_abs_diff"] <= 1e-4


def test_contract_check_rejects_a_foreign_input_set(tmp_path):
    m = select_model("distance_estimator", use_goal=True, use_depth=False)
    p = tmp_path / "d.onnx"
    m.to_onnx(p, with_goal=True, with_depth=False)
    with pytest.raises(RuntimeError, match="planner contract"):
        assert_onnx_contract(p, with_goal=False, bitmask=False)


def test_the_stale_june_models_fail_the_contract():
    """Existing exp/gnn_exp artifacts declare FLOAT node ids under an old name;
    the current planner feeds int64 and aborts on them. The check must say so."""
    from conftest import REPO_ROOT
    old = REPO_ROOT / "exp/gnn_exp/batch1/_models/CC/distance_estimator.onnx"
    if not old.is_file():
        pytest.skip("no batch1 artifact")
    with pytest.raises(RuntimeError):
        assert_onnx_contract(old, with_goal=False, bitmask=False)
