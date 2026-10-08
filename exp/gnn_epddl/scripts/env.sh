#!/bin/bash
# Common settings of the replication scripts; source it from the repository root (every script does).
# Override any variable in the environment, e.g. DEEP=/path/to/deep ./exp/gnn_epddl/scripts/3_data.sh
export ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export PYTHONPATH=$ROOT/lib/gnn_epddl/src${PYTHONPATH:+:$PYTHONPATH}
export PY=${PY:-python3}
export DEEP=${DEEP:-$ROOT/cmake-build-release-nn/bin/deep}     # deep built with ONNX support (build.sh nn)
export EXP=$ROOT/exp/gnn_epddl
export INSTANCES=$EXP/instances
export SPLIT=${SPLIT:-$EXP/split.csv}
export WORK=${WORK:-$EXP/work}                                  # data, checkpoints, logs (large; not versioned)
export DATA=${DATA:-$WORK/data}
export CKPT=${CKPT:-$WORK/ckpt}
export LOG=${LOG:-$WORK/train.csv}
export DEVICE=${DEVICE:-cuda}                                   # training device; validation searches run on the CPU
gnn() { "$PY" -m gnn_epddl "$@"; }
