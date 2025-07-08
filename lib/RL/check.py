from __future__ import annotations

import argparse
from utils import select_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real-time c++ planner score.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("pytorch_model_file", type=str, help="PyTorch Path to GGN model file")
    parser.add_argument("onnx_model_file", type=str, help="ONNX Path to GGN model file")
    parser.add_argument("c_score", type=float, help="Score get from real-time planner")
    parser.add_argument("threshold", type=float, help="Threshold percentage to print warning", default=0.1)
    parser.add_argument("use_goal", type=bool, help="Use goal dot as additional feature", default=False)
    parser.add_argument("use_depth", type=bool, help="Use depth as additional feature", default=False)
    args = parser.parse_args()

    path_state = args.path
    pytorch_model_file = args.pytorch_model_file
    onnx_model_file = args.onnx_model_file

    c_score = args.c_score
    th = args.threshold

    USE_GOAL = args.use_goal
    USE_DEPTH = args.use_depth

    model = select_model("distance_estimator", USE_GOAL, USE_DEPTH)
    model.load_model(pytorch_model_file)

    pytorch_pred = model.predict_single(path_state)
    python_onnx_pred = model.try_onnx(onnx_model_file, state_dot_files=[path_state])[0]

    c_score_perc = c_score * th

    if not(c_score - c_score_perc < pytorch_pred < c_score + c_score_perc):
        print(f"[WARNING] Planner output: {c_score:.3f} vs. PyTorch ouput: {pytorch_pred:.3f} | Difference: {((pytorch_pred-c_score)/c_score)*100:.2f}%")

    if not(c_score - c_score_perc < python_onnx_pred < c_score + c_score_perc):
        print(f"[WARNING] Planner output: {c_score:.3f} vs. Python-ONNX ouput: {python_onnx_pred:.3f} | Difference: {((python_onnx_pred-c_score)/c_score)*100:.2f}%")

    print(f"PyTorch: {pytorch_pred:.3f}")
    print(f"Python ONNX: {python_onnx_pred:.3f}")

if __name__ == "__main__":
    main()