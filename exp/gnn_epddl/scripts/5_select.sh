#!/bin/bash
# Step 5: per run, the checkpoint with the most validation problems solved, then the smallest sum of log10(nodes)
# (unsolved = 2 x cap), then the later epoch.
set -euo pipefail
source "$(dirname "$0")/env.sh"
gnn select --log "$LOG" --out "$WORK/selection.csv"
