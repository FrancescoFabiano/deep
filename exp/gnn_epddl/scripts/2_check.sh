#!/bin/bash
# Step 2: check every generated problem: plank parses it, it is not a benchmark problem, deep's BFS solves it.
# Needs PLANK (plank binary) and BENCHMARKS (benchmark root: <tier>/<family>/problems/*.epddl).
# Training problems were checked with 120 s / 6 GiB, validation problems with 600 s / 24 GiB (TIMEOUT, MEM_GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${PLANK:?set PLANK}" "${BENCHMARKS:?set BENCHMARKS}"
mkdir -p "$WORK/check"
for d in "$INSTANCES"/*/*/problems; do
  fam=$(basename "$(dirname "$d")"); tier=$(basename "$(dirname "$(dirname "$d")")")
  gnn check --instances "$INSTANCES" --tier "$tier" --family "$fam" --deep "$DEEP" --plank "$PLANK" \
      --benchmarks "$BENCHMARKS" --timeout "${TIMEOUT:-120}" --mem-gb "${MEM_GB:-6}" --jobs "${JOBS:-3}" \
      > "$WORK/check/$tier-$fam.csv"
done
