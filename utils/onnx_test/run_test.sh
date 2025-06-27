#!/bin/bash
set -e

echo "Running ONNX Runtime sanity test..."

cd "$(dirname "$0")"

# Check if model.onnx exists, if not run python script to create it
if [[ ! -f "model.onnx" ]]; then
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
    if [[ ! -f "model.onnx" ]]; then
        echo "[ERROR] model_creation.py did not create model.onnx"
        exit 1
    fi
else
    echo "Found existing model.onnx"
fi

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
make -j$(nproc)

if ./test_onnx; then
    echo "ONNX Runtime test ran successfully."
else
    echo "[ERROR] ONNX Runtime test failed."
    exit 1
fi
