#!/bin/bash
set -e

echo "Running libtorch sanity test..."

cd "$(dirname "$0")"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
make -j$(nproc)

if ./test_libtorch; then
    echo "LibTorch test ran successfully."
else
    echo "[ERROR] LibTorch test failed."
    exit 1
fi
