"""C-file export with a first-class heuristic weight.

The planner computes  distance = round((onnx_out - intercept) / slope)
from `distance_estimator_C.txt`.  Writing slope' = W x slope therefore
scales the heuristic to h/W — the bounded-suboptimality knob measured in
REPORT_sweep_and_maxdepth.md (the previous sessions edited the file by
hand; here it is a recorded, reproducible parameter).

C-file format contract (planner-compatible): the `slope = ...` and
`intercept = ...` lines come FIRST; the C++ parser (GraphNN.tpp) stops
after reading both, so the provenance lines that follow are ignored by the
planner but keep the model <-> C-file pairing reconstructable.
"""
from __future__ import annotations

import json
import os
from typing import Dict


def write_c_file(
    path_model: str,
    model_name: str,
    constants_name: str,
    params: Dict,
    heuristic_weight: float = 1.0,
) -> str:
    """Write distance_estimator_C.txt with slope x W; return the path."""
    path = f"{path_model}/{model_name}_{constants_name}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        # Planner-consumed lines first (parser stops after these two):
        fh.write(f"slope = {params['slope'] * heuristic_weight}\n")
        fh.write(f"intercept = {params['intercept']}\n")
        # Provenance (ignored by the planner):
        fh.write(f"max_depth = {params['max_depth']}\n")
        fh.write(f"raw_slope = {params['slope']}\n")
        fh.write(f"heuristic_weight = {heuristic_weight}\n")
    return path


def record_params_in_history(
    path_model: str, params: Dict, heuristic_weight: float
) -> None:
    """Merge scaling params + W into history_losses.json (post-training)."""
    path = f"{path_model}/history_losses.json"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload.update(params)
    payload["heuristic_weight"] = heuristic_weight
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)
