#!/bin/bash
# Step 3: deep's search trees of the split's training instances (BFS, else DFS seeds 42-44; depth 25, at most
# 60000 written / 100000 visited states, HASHED, --ranker_encoding) and their training sets (new3.pt).
# --trim deletes each raw tree once its new3.pt is built (the trees take several GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
gnn data --instances "$INSTANCES" --split "$SPLIT" --data "$DATA" --deep "$DEEP" --jobs "${JOBS:-4}" --trim
