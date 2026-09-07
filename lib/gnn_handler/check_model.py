"""Spot-check a trained distance estimator: torch vs ONNX vs table label on
random training states of every instance under <batch>/_models/<domain>/.

Reads the normalisation from distance_estimator_C.txt (what the planner reads),
finds the table by its depth pattern (the generator now embeds the strategy in
the file name), and uses the planner's 42-bit width in BITMASK mode.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import pandas as pd
from pathlib import Path

from tqdm import tqdm

from src.utils import select_model


def read_constants(path: str) -> tuple[float, float]:
    """`slope = ...` / `intercept = ...` -- the two lines the C++ parses."""
    vals = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = float(v)
    return vals["slope"], vals["intercept"]


def get_subfolders(root: str):
    return [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]


def get_distance_from_goal(df: pd.DataFrame, file_name: str) -> int:

    # Ensure required columns exist
    if "File Path" not in df.columns or "Distance From Goal" not in df.columns:
        raise ValueError(
            "CSV must contain 'File Path' and 'Distance From Goal' columns"
        )

    # Filter the row where File Path matches the given name
    row = df.loc[df["File Path"] == file_name]

    if row.empty:
        raise ValueError(f"No entry found for file path: {file_name}")

    # Extract the value (convert to int)
    value = int(row["Distance From Goal"].iloc[0])
    return value


def sample_dot_files(folder, n: int, recursive: bool = False) -> list[Path]:
    folder = Path(folder)  # ensure it's a Path
    if recursive:
        dot_files = sorted(folder.rglob("*.dot"))
    else:
        dot_files = sorted(folder.glob("*.dot"))

    if len(dot_files) < n:
        n = len(dot_files)

    return random.sample(dot_files, n)


def run_test(
    model_path: str,
    onnx_model_path: str,
    state_dot: str,
    ground_truth: int,
    goal_dot: str | None = None,
    depth: int | None = None,
    bitmask: bool = False,
    constants: tuple[float, float] = (0.01996, 0.001),
):
    slope, intercept = constants
    model = select_model("distance_estimator", goal_dot is not None, False, bitmask=bitmask)
    model.load_model(model_path)

    # PyTorch
    out = model.predict_single(
        state_dot, depth=depth, goal_dot=goal_dot, bitmask=bitmask
    )
    final_out = int((out - intercept) / slope)

    # ONNX
    out_onnx = model.try_onnx(
        onnx_model_path, [state_dot], bitmask=bitmask,
        goal_dot_files=[goal_dot] if goal_dot is not None else None,
    )
    # Ensure scalar
    out_onnx_scalar = (
        float(out_onnx[0]) if hasattr(out_onnx, "__len__") else float(out_onnx)
    )
    final_out_onnx = int((out_onnx_scalar - intercept) / slope)
    # print(f"ONNX output: {out_onnx} -> {final_out_onnx}")

    # Compare with an absolute tolerance (in bins)
    tol_bins = max(1, int(0.1 * abs(final_out)))
    diff_bins = abs(final_out_onnx - final_out)
    assert diff_bins <= tol_bins, (
        f"ONNX and PyTorch are not aligned: onnx={final_out_onnx}, torch={final_out}, "
        f"diff_bins={diff_bins}, tol_bins={tol_bins}, raw_onnx={out_onnx_scalar}, raw_torch={out}"
    )

    # Ground-truth check (keep your threshold)
    if abs(ground_truth - final_out) > 1:
        print(f"Error: {ground_truth} vs. {final_out}")


def main(path_experiment_batch: str, n_trials: int):
    models_path = os.path.join(path_experiment_batch, "_models")
    domains = get_subfolders(models_path)

    for domain in domains:
        domain_problems = get_subfolders(os.path.join(domain, "training_data"))

        constants = read_constants(os.path.join(domain, "distance_estimator_C.txt"))
        for problem in domain_problems:
            print(f"Problem: {problem}")
            problem_name = os.path.basename(problem)

            tables = sorted(glob.glob(os.path.join(problem, "*_depth_*.csv")))
            if len(tables) != 1:
                print(f"[WARN] expected one generation table in {problem}, found {len(tables)}")
                continue
            csv_file = pd.read_csv(tables[0])
            goal_dot = os.path.join(problem, "goal_tree.dot")
            separated = os.path.isdir(os.path.join(problem, "RawFiles", "hash_separated")) \
                and os.path.isfile(goal_dot)

            raw_dir = os.path.join(problem, "RawFiles")
            # choose first subfolder inside RawFiles
            subdirs = get_subfolders(raw_dir)
            if not subdirs:
                print(f"[WARN] No subfolders in {raw_dir}")
                continue

            state_root = subdirs[0]
            if_bitmask = os.path.basename(state_root).startswith("mask")
            samples_root = sample_dot_files(state_root, n_trials)

            for state_dot in tqdm(samples_root, desc=problem_name):
                model_path = os.path.join(domain, "distance_estimator.pt")
                onnx_path = os.path.join(domain, "distance_estimator.onnx")

                ground_truth = get_distance_from_goal(csv_file, str(state_dot))

                run_test(
                    model_path,
                    onnx_path,
                    str(state_dot),
                    ground_truth,
                    goal_dot=goal_dot if separated else None,
                    bitmask=if_bitmask,
                    constants=constants,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_experiment_batch",
        type=str,
        help="Path to the experiment batch (e.g., exp/gnn_exp/batch_CC)",
    )
    parser.add_argument(
        "--n_trials",
        default=20,
        type=int,
        help="Number of trials to run",
    )
    args = parser.parse_args()
    main(args.path_experiment_batch, args.n_trials)
