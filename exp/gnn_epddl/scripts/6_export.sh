#!/bin/bash
# Step 6: export the selected checkpoints into exp/gnn_epddl/models in the folder layout deep reads:
#   models/cpu/  fast export (segment sums; --ranker_model)       models/gpu/  standard export (scatter_add; --ranker_model_gpu)
#   per-domain/<domain>.onnx for pf-<domain>, general.onnx for final-all, subset.onnx for final-sub; each with its .vocab.
# The export checks use the search-and-rescue training sets as sample graphs, as in the experiments.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=${OUT:-$EXP/models}
tail -n +2 "$WORK/selection.csv" | while IFS=, read -r run ckpt _; do
  case "$run" in
    pf-*) rel=per-domain/${run#pf-}.onnx ;;
    final-all) rel=general.onnx ;;
    final-sub) rel=subset.onnx ;;
    *) continue ;;
  esac
  for kind in cpu:fast gpu:standard; do
    mkdir -p "$(dirname "$OUT/${kind%%:*}/$rel")"
    gnn export --checkpoint "$ckpt" --out "$OUT/${kind%%:*}/$rel" --kind "${kind#*:}" --sample-data "$DATA/search-and-rescue"
  done
done
