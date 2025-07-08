from __future__ import annotations

import argparse
from utils import select_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real-time C++ planner score.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("pytorch_model_file", type=str, help="PyTorch Path to GGN model file")
    parser.add_argument("onnx_model_file", type=str, help="ONNX Path to GGN model file")
    parser.add_argument("output_file", type=str, help="File to write results to")
    args = parser.parse_args()

    # Write parsed arguments to info.out
    with open("info.out", "w") as info_file:
        for arg_name, arg_value in vars(args).items():
            info_file.write(f"{arg_name}: {arg_value}\n")

    path_state = args.path
    pytorch_model_file = args.pytorch_model_file
    onnx_model_file = args.onnx_model_file

    USE_GOAL = False
    USE_DEPTH = False

    model = select_model("distance_estimator", USE_GOAL, USE_DEPTH)
    model.load_model(pytorch_model_file)

    pytorch_pred = model.predict_single(path_state)
    python_onnx_pred = model.try_onnx(onnx_model_file, state_dot_files=[path_state])[0]

    output_lines = []

    output_lines.append(f"PyTorch: {pytorch_pred:.3f}")
    output_lines.append(f"ONNX: {python_onnx_pred:.3f}")

    output_file = args.output_file

    # Write to prediction output file
    with open(output_file, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
