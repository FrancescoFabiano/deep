from __future__ import annotations

import argparse
from utils import select_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real-time C++ planner score.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("pytorch_model_file", type=str, help="PyTorch Path to GGN model file")
    parser.add_argument("onnx_model_file", type=str, help="ONNX Path to GGN model file")
    parser.add_argument("use_goal", type=bool, help="Use goal dot as additional feature", default=False)
    parser.add_argument("use_depth", type=bool, help="Use depth as additional feature", default=False)
    parser.add_argument("output_file", type=str, default="prediction_results.out", help="File to write results to")
    args = parser.parse_args()

    path_state = args.path
    pytorch_model_file = args.pytorch_model_file
    onnx_model_file = args.onnx_model_file

    USE_GOAL = args.use_goal
    USE_DEPTH = args.use_depth

    model = select_model("distance_estimator", USE_GOAL, USE_DEPTH)
    model.load_model(pytorch_model_file)

    pytorch_pred = model.predict_single(path_state)
    python_onnx_pred = model.try_onnx(onnx_model_file, state_dot_files=[path_state])


    output_lines = []
    output_lines.append(f"PyTorch: {pytorch_pred:.3f}")
    output_lines.append(f"ONNX: {python_onnx_pred:.3f}")

    # Write to file
    with open(args.output_file, "w") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
