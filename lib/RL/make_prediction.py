from __future__ import annotations

import argparse
from utils import select_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNN heuristic prediction on a DOT graph.")
    parser.add_argument("path", type=str, help="Path to state DOT graph")
    parser.add_argument("depth", type=int, help="Search depth parameter")
    parser.add_argument("goal_file", type=str, help="Path to goal DOT graph")
    parser.add_argument("model_file", type=str, help="Path to GNN model file")
    parser.add_argument("use_goal", type=bool, help="Use goal dot as additional feature", default=False)
    parser.add_argument("use_depth", type=bool, help="Use depth as additional feature", default=False)
    args = parser.parse_args()

    path_state = args.path
    depth = args.depth
    path_goal = args.goal_file
    model_file = args.model_file

    USE_GOAL = args.use_goal
    USE_DEPTH = args.use_depth

    model = select_model("distance_estimator", USE_GOAL, USE_DEPTH)
    model.load_model(model_file)

    pred = model.predict_single(path_state, depth if USE_DEPTH else None, path_goal if USE_GOAL else None)

    print(f"{pred:.3f}")

if __name__ == "__main__":
    main()
