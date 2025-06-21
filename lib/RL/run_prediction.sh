#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <dot_file> <depth> <goal_file> <model_file>"
  exit 1
fi

DOT_FILE="$1"
DEPTH="$2"
GOALFILE="$3"
MODELFILE="$4"


# ── Activate the virtual environment in that directory ──────────────────────────
VENV_PATH=".venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "No .venv found at: $VENV_PATH"
  exit 1
fi

source "$VENV_PATH"

# ── Run the Python script ───────────────────────────────────────────────────────
python3 -u "./lib/RL/make_prediction.py" "$DOT_FILE" "$DEPTH" "$GOALFILE" "$MODELFILE"
