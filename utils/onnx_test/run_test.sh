#!/bin/bash
set -euo pipefail

echo "Running ONNX Runtime sanity test..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONNXRUNTIME_DIR="${REPO_ROOT}/lib/onnxruntime"
BUILD_DIR="${SCRIPT_DIR}/build"
MODEL_PATH="${SCRIPT_DIR}/model.onnx"

cd "$SCRIPT_DIR"

if [[ ! -f "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" ]]; then
    echo "[ERROR] ONNX Runtime headers not found in ${ONNXRUNTIME_DIR}/include."
    echo "Build deep with neural-network support first."
    exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    ONNX_LIB="${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
else
    ONNX_LIB="${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
fi

if [[ ! -f "$ONNX_LIB" ]]; then
    echo "[ERROR] ONNX Runtime library not found at ${ONNX_LIB}."
    echo "Build deep with neural-network support first."
    exit 1
fi

# Check if model.onnx exists, if not run python script to create it
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "model.onnx not found. Attempting to create with Python script..."

    if command -v python3 &>/dev/null; then
        python3 model_creation.py
    elif command -v python &>/dev/null; then
        python model_creation.py
    else
        echo "[ERROR] Python is not installed or not in PATH."
        exit 1
    fi

    # Verify model creation
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "[ERROR] model_creation.py did not create model.onnx"
        exit 1
    fi
else
    echo "Found existing model.onnx"
fi

mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR"

if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif [[ "$(uname -s)" == "Darwin" ]]; then
    JOBS=$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
else
    JOBS=1
fi

cmake --build "$BUILD_DIR" --parallel "$JOBS"

if ./build/onnx_test; then
    echo "ONNX Runtime test ran successfully."
else
    echo "[ERROR] ONNX Runtime test failed."
    exit 1
fi
