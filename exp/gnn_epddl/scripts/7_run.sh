#!/bin/bash
# Step 7: the planner runs on the 100 IPC 2026 benchmark problems, through the competition runner
# (iepc2026-runner, scripts/run-parallel.sh): 600 s CPU and 16 GiB per problem, one instance, 30 jobs at a time.
# Usage: RUNNER=<runner checkout> ./exp/gnn_epddl/scripts/7_run.sh [config ...]   (default: every config)
# Configs (all with -b -c --fast-comparison; models from exp/gnn_epddl/models):
#   bfs              breadth-first search
#   hfs-cpu          per-domain ranker, greedy best-first search, CPU (models/cpu)
#   hfs-cpu-gpu      as hfs-cpu, states with >= 10000 edges on the GPU (models/gpu); needs a CUDA build (DEEP_GPU, default DEEP),
#                    and runs JOBS_GPU (6) jobs at a time with GPU_MEM_MIB (6000) MiB of GPU memory each: many processes
#                    scoring very large states at once can exhaust the GPU's memory
#   general subset no-spreading no-puzzle no-physical   one ranker for every domain, greedy best-first search, CPU
#   beam8 beam16     per-domain ranker ranking the RL beam (width 8 or 16, reservoir), CPU
#   untrained        the untrained network (control), greedy best-first search, CPU
# PROBLEM_LIST=<file> (benchmark paths, one per line, relative to the runner) restricts every config to those
# problems, e.g. for a smoke test. Results: $WORK/runs/<config>/results.csv (runner format).
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${RUNNER:?set RUNNER to a checkout of the competition runner}"
DEEP_GPU=${DEEP_GPU:-$DEEP}   # a CUDA build (./build.sh nn use_gpu); it also runs the CPU-only configs, which force the CPU
JOBS=${JOBS:-30}
JOBS_GPU=${JOBS_GPU:-6}
GPU_MEM_MIB=${GPU_MEM_MIB:-6000}
M=$EXP/models
BASE="-b -c --fast-comparison --onnx_threads 1"
HFS="$BASE -s HFS -u GNN"
beam() { echo "$BASE -s RL -u RL_H --RL_fringe_size $1 --RL_exploration 10 --RL_exploitation 70 --RL_seed 42 --onnx_device cpu --ranker_model $M/cpu"; }
args() {
  case $1 in
    bfs) echo "-b -c --fast-comparison -s BFS" ;;
    hfs-cpu) echo "$HFS --onnx_device cpu --ranker_model $M/cpu" ;;
    hfs-cpu-gpu) echo "$HFS --ranker_model $M/cpu --ranker_model_gpu $M/gpu --onnx_gpu_mem_limit $GPU_MEM_MIB" ;;
    general | subset | no-spreading | no-puzzle | no-physical | untrained) echo "$HFS --onnx_device cpu --ranker_model $M/cpu/$1.onnx" ;;
    beam8) beam 8 ;;
    beam16) beam 16 ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(bfs hfs-cpu hfs-cpu-gpu general subset no-spreading no-puzzle no-physical beam8 beam16 untrained)
for c in "${CONFIGS[@]}"; do
  bin=$DEEP; jobs=$JOBS
  [ "$c" = hfs-cpu-gpu ] && { bin=$DEEP_GPU; jobs=$JOBS_GPU; }
  out=$WORK/runs/$c; mkdir -p "$out"
  echo "== $c $(date)"
  (cd "$RUNNER" && env TEAMS=baseline-deep DEEP_BIN="$bin" DEEP_TEAM="gnn-epddl-$c" DEEP_ARGS="$(args "$c")" \
     PROBLEM_LIST="${PROBLEM_LIST:-}" INSTANCES=1 SOLVE_TIMEOUT=600 MEM_LIMIT_MIB=16384 JOBS="$jobs" ARCHIVE=0 GIT_PUSH=0 LOG_DIR="$out" \
     ./scripts/run-parallel.sh) > "$out/runner.log" 2>&1
done
