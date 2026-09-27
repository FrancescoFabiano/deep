"""The GNN through the FringeEvalRL contract: inputs, -distance output, parity."""

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from conftest import Cache, random_graph
from deep_nn import contract
from deep_nn.pack import pack_fringe, pack_goal_tensors
from src.utils import STATE_INPUTS, DistanceEstimatorModel, constant_file, model_inputs, normalization_params


def _model(separated: bool) -> DistanceEstimatorModel:
    torch.manual_seed(0)
    return DistanceEstimatorModel(hidden_dim=16, node_emb_dim=8, edge_emb_dim=4,
                                  regressor_hidden_dim=16, regressor_blocks=1, use_goal=separated, device="cpu")


def _feed(k: int, F: int, separated: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    cache = Cache([random_graph(rng) for _ in range(k)])
    p = pack_fringe(cache, list(range(k)), F)
    if separated:
        p.update(pack_goal_tensors([random_graph(rng, n=3, e=4)]))
    return p


@pytest.mark.parametrize("separated", [False, True])
def test_export_declares_the_contract_inputs(tmp_path, separated):
    m = _model(separated)
    path = m.to_onnx(tmp_path / "d.onnx", 8, normalization_params(10))
    names = [i.name for i in onnx.load(str(path)).graph.input]
    assert names == contract.input_names(separated)
    assert "pointed_ids" in names
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert sess.get_outputs()[0].shape == [8]        # ORT resolves the slot axis to F


@pytest.mark.parametrize("separated", [False, True])
def test_scores_are_minus_distance_and_match_torch(tmp_path, separated):
    m = _model(separated)
    params = normalization_params(10)
    path = m.to_onnx(tmp_path / "d.onnx", 8, params)
    for k in (8, 3, 1):                               # full beam and short beams
        feed = _feed(k, 8, separated, seed=k)
        got = contract.run_onnx(path, feed)[:k]
        with torch.no_grad():
            d = m.model(**model_inputs(feed))
        want = (-(d - params["intercept"]) / params["slope"]).numpy()
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
        assert (got <= 0).all() and np.isfinite(got).all()
    assert m.verify_onnx(path, [_feed(4, 8, separated, seed=9)], params)["max_abs_diff"] < 1e-5


def test_a_slot_scores_from_its_own_graph_only(tmp_path):
    m = _model(False)
    path = m.to_onnx(tmp_path / "d.onnx", 8, normalization_params(10))
    full = contract.run_onnx(path, _feed(4, 8, False))
    fewer = contract.run_onnx(path, _feed(3, 8, False))
    np.testing.assert_allclose(fewer[:3], full[:3], atol=1e-6)


def test_designated_worlds_reach_the_score():
    m = _model(False)
    with torch.no_grad():
        m.model.pointed.weight.fill_(0.5)
        feed = _feed(2, 2, False)
        a = m.model(**model_inputs(feed))
        feed["pointed_ids"] = torch.tensor([1, 6])
        b = m.model(**model_inputs(feed))
    assert not torch.allclose(a, b)


def test_non_hashed_export_is_refused(tmp_path):
    import importlib.util
    from conftest import PKG_ROOT
    spec = importlib.util.spec_from_file_location("gnn_main", PKG_ROOT / "__main__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main = mod.main
    with pytest.raises(SystemExit, match="HASHED"):
        main(["--train-csv", "x.csv", "--dir-save-model", str(tmp_path), "--dataset-type", "BITMASK"])


def test_state_export_matches_torch_and_writes_constants(tmp_path):
    m = _model(False)
    params = normalization_params(10)
    path = m.to_onnx_state(tmp_path / "s.onnx", params)
    assert [i.name for i in onnx.load(str(path)).graph.input] == list(STATE_INPUTS)
    feed = _feed(1, 1, False)          # one state: membership is all zeros, like the planner's batch
    got = contract.run_onnx(path, {"node_features": feed["node_features"], "edge_index": feed["edge_index"],
                                   "edge_attr": feed["edge_attr"].view(-1, 1), "batch": feed["membership"],
                                   "pointed_ids": feed["pointed_ids"]})
    with torch.no_grad():
        want = m.model(**model_inputs(feed)).numpy()
    assert got.shape == (1,) and abs(float(got[0]) - float(want[0])) < 1e-5
    text = constant_file(path).read_text()
    assert f"slope = {params['slope']}" in text and f"intercept = {params['intercept']}" in text
