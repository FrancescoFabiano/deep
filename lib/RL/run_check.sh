#!/bin/bash

# Usage: ./run_check.sh path_to_dot pt_model onnx_model use_goal use_depth

set -e  # Exit immediately if any command fails

# Activate the Python virtual environment
VENV_PATH=".venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "No .venv found at: $VENV_PATH"
  exit 1
fi

source "$VENV_PATH"

# Check number of arguments
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 path_to_dot pt_model onnx_model"
  exit 1
fi

# Fixed output file name
OUTPUT_FILE="prediction_results.out"

# Call the Python script with the fixed output file name
python3 lib/RL/check.py "$1" "$2" "$3" "$OUTPUT_FILE"
