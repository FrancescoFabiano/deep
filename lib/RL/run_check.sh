#!/bin/bash

# Usage: ./run_check.sh path_to_dot pt_model onnx_model c_score use_goal use_depth output_file

# Activate the Python virtual environment
# ── Activate the virtual environment in that directory ──────────────────────────
VENV_PATH=".venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "No .venv found at: $VENV_PATH"
  exit 1
fi

source "$VENV_PATH"


# Forward all arguments to the Python script
python3 lib/RL/check.py "$1" "$2" "$3" "$4" "$5" "$6"