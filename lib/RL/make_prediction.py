from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from utils import select_model


def double_prediction(
        path_models: str,
        path_dot_state: str,
        depth: int,
        goal_dot_state: str = None,
        use_goal: bool = True,
        DEFAULT_UNREACHABLE_STATE_VALUE: float = 100000.0,
) -> Tuple[float, float]:

    HERE = Path(__file__).resolve().parent    # → /.../lib/RL

    # point at the models directory next to this file
    path_models = HERE / "models"               # → /.../lib/RL/models

    classifier = select_model("reachability_classifier", use_goal)

    classifier.load_model(f"{path_models}/classifier.pt")

    if use_goal:
        reachability = classifier.predict_single(path_dot_state, depth, goal_dot_state)["prob_reachable"]
    else:
        reachability = classifier.predict_single(path_dot_state, depth)["prob_reachable"]
    #{"prob_reachable": prob, "reachable": prob >= threshold}

    if reachability > 0.5:
        estimator = select_model("distance_estimator", use_goal)
        estimator.load_model(f"{path_models}/estimator.pt")
        if use_goal:
            return estimator.predict_single(path_dot_state, depth, goal_dot_state), reachability
        else:
            return estimator.predict_single(path_dot_state, depth), reachability
    else:
        return DEFAULT_UNREACHABLE_STATE_VALUE, reachability

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNN heuristic prediction on a DOT graph.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("depth", type=int, help="Search depth parameter")
    parser.add_argument("goal_file", type=str, help="Path to goal DOT graph")
    parser.add_argument("model_file", type=str, help="Path to GGN model file")
    args = parser.parse_args()

    path_state = args.path
    depth = args.depth
    path_goal = args.goal_file
    model_file = args.model_file

    pred, reachability = double_prediction(model_file, path_state, depth, path_goal)

    with open("error.log", "a+") as f:
        print(f"path: {args.path}", file=f)
        print(f"depth: {args.depth}", file=f)
        print(f"depth: {args.goal_file}", file=f)
        print(f"probability reachable: {reachability}", file=f)
        print(f"distance from goal {pred} -> {int(pred)}", file=f)
        print("************************", file=f)

    pred = int(pred)

    print(f"{pred:.6f}")

if __name__ == "__main__":
    main()
