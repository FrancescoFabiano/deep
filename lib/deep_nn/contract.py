"""The ONNX contract with FringeEvalRL (src/heuristics/strategies/neuralnets).

The planner pairs its tensors with the model's inputs BY POSITION (it reads the
names from the session and passes them back), so order and dtype are the
contract and names are documentation:

    merged    (6): node_features, edge_index, edge_attr, membership, pointed_ids, mask
    separated(10): ... pointed_ids, goal_node_features, goal_edge_index,
                   goal_edge_attr, goal_batch, mask
    output       : scores float32 [F]

`rankScores` sorts the scores DESCENDING and expands the first slot, so higher
means better and only the order within one call matters.  A model that outputs
a distance exports ``-distance``.  The planner exits on an input-count mismatch
(FringeEvalModelLoadError) and reads output[0..K) for K occupied slots.

Only HASHED data reaches this path today (issue #1), so node ids are int64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

STATE_INPUTS = ["node_features", "edge_index", "edge_attr", "membership", "pointed_ids"]
GOAL_INPUTS = ["goal_node_features", "goal_edge_index", "goal_edge_attr", "goal_batch"]
MASK = "mask"
OUTPUT = "scores"
OPSET = 18


def input_names(separated: bool) -> List[str]:
    return STATE_INPUTS + (GOAL_INPUTS if separated else []) + [MASK]


def dummy_inputs(fringe_size: int, separated: bool) -> tuple:
    """Trace with every slot occupied: the exporter bakes the pooled size F into
    the graph, which is what makes the output length F at K < F."""
    F = int(fringe_size)
    n, e, p = max(8, F), 12, 2
    state = (
        torch.zeros(n, dtype=torch.int64),
        torch.zeros((2, e), dtype=torch.int64),
        torch.zeros(e, dtype=torch.int64),
        torch.arange(n, dtype=torch.int64) % F,
        torch.arange(p, dtype=torch.int64),
    )
    gn, ge = 6, 8
    goal = (
        torch.zeros(gn, dtype=torch.int64),
        torch.zeros((2, ge), dtype=torch.int64),
        torch.zeros(ge, dtype=torch.int64),
        torch.zeros(gn, dtype=torch.int64),
    ) if separated else ()
    return state + goal + (torch.ones(F, dtype=torch.uint8),)


def dynamic_axes(separated: bool) -> Dict[str, Dict[int, str]]:
    axes = {"node_features": {0: "N"}, "edge_index": {1: "E"}, "edge_attr": {0: "E"},
            "membership": {0: "N"}, "pointed_ids": {0: "P"}, MASK: {0: "F"}, OUTPUT: {0: "F"}}
    if separated:
        axes.update({"goal_node_features": {0: "GN"}, "goal_edge_index": {1: "GE"},
                     "goal_edge_attr": {0: "GE"}, "goal_batch": {0: "GN"}})
    return axes


def export(wrapper: nn.Module, out_path: str | Path, fringe_size: int, separated: bool) -> Path:
    """Export ``wrapper`` (forward = the contract's positional inputs, returns
    scores [F]) as ONE file and verify the contract on it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False pins the TorchScript exporter: a single self-contained file.
    # The dynamo exporter writes the weights to `<name>.onnx.data`, which breaks
    # as soon as the .onnx is copied on its own.
    torch.onnx.export(
        wrapper.eval().cpu(), dummy_inputs(fringe_size, separated), out_path.as_posix(),
        opset_version=OPSET, dynamo=False, input_names=input_names(separated),
        output_names=[OUTPUT], dynamic_axes=dynamic_axes(separated), do_constant_folding=False,
    )
    sidecar = out_path.with_name(out_path.name + ".data")
    if sidecar.exists():
        raise RuntimeError(f"export wrote external data {sidecar}; the planner loads one file")
    assert_contract(out_path, separated)
    assert_static_output(out_path, int(fringe_size))
    return out_path


def assert_static_output(onnx_path: str | Path, fringe_size: int) -> None:
    """FringeEvalRL refuses a model whose output length onnxruntime cannot infer
    statically (exit 905 "invalid/dynamic frontier dimension"). PyG's baked pool
    size made it static by accident; the dense (blocked) aggregation does not, so
    the wrappers reshape to F and this check keeps it that way."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    shape = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"]).get_outputs()[0].shape
    if shape != [fringe_size]:
        raise ValueError(f"{onnx_path}: output shape {shape} is not the static [{fringe_size}] the planner requires")


def assert_contract(onnx_path: str | Path, separated: bool) -> Dict[str, object]:
    """Names in order, dtypes, one float output. Raises on any deviation."""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    names = [i.name for i in m.graph.input]
    want = input_names(separated)
    if names != want:
        raise ValueError(f"{onnx_path}: inputs {names} != contract {want} "
                         f"(an input the model never uses is pruned at export)")
    for i in m.graph.input:
        et = i.type.tensor_type.elem_type
        exp = onnx.TensorProto.UINT8 if i.name == MASK else onnx.TensorProto.INT64
        if et != exp:
            raise ValueError(f"{onnx_path}: input {i.name} has dtype {et}, expected {exp}")
    outs = [o.name for o in m.graph.output]
    if len(outs) != 1 or m.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"{onnx_path}: expected one float32 output, got {outs}")
    return {"inputs": names, "output": outs[0]}


def run_onnx(onnx_path: str | Path, feed: Dict[str, torch.Tensor | np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    arrays = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
              for k, v in feed.items() if k in names}
    return sess.run(None, arrays)[0]


def check_parity(onnx_path: str | Path, torch_scores, feeds: Sequence[Dict[str, torch.Tensor]],
                 atol: float = 1e-5) -> Dict[str, float]:
    """``torch_scores(feed) -> [K]`` must match the ONNX output's first K
    entries on every feed; the scores must also be finite (issue #5: NaN and
    ties are undefined behaviour in rankScores)."""
    worst = 0.0
    for feed in feeds:
        k = int(feed[MASK].sum())
        got = run_onnx(onnx_path, feed)[:k]
        if not np.isfinite(got).all():
            raise ValueError(f"{onnx_path}: non-finite scores {got}")
        with torch.no_grad():
            want = torch_scores(feed).detach().cpu().numpy()[:k]
        worst = max(worst, float(np.abs(got - want).max()))
        if worst > atol:
            raise ValueError(f"{onnx_path}: onnxruntime and PyTorch differ by {worst:.2e} > {atol}")
    return {"n_checked": len(feeds), "max_abs_diff": worst}
