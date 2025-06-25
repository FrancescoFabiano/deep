#!/bin/bash

set -e

# Help message
show_usage() {
    echo "Usage: ./build.sh [nn] [debug] [use_gpu] [force_gpu] [no_torch_test]"
    echo "  Options:"
    echo "    nn              Enable neural networks (downloads libtorch if not present)"
    echo "    debug           Build with Debug flags (default is Release)"
    echo "    use_gpu         Use GPU-accelerated libtorch (requires NVIDIA GPU and CUDA installed)"
    echo "    force_gpu       Force install of GPU libtorch without checking for CUDA or GPU"
    echo "    no_torch_test   Skip running the torch test after build"
    echo ""
    echo "  Examples:"
    echo "    ./build.sh                          # Release without NN"
    echo "    ./build.sh nn                       # Release with NN using CPU libtorch"
    echo "    ./build.sh debug nn                 # Debug with NN using CPU libtorch"
    echo "    ./build.sh nn use_gpu               # Release with NN using GPU libtorch (if available)"
    echo "    ./build.sh nn force_gpu             # Force GPU libtorch download (e.g. for cross-compilation)"
    echo "    ./build.sh nn no_torch_test         # Build with NN but skip torch test"
}

# Default options
BUILD_TYPE="Release"
ENABLE_NN="OFF"
USE_GPU="OFF"
FORCE_GPU="OFF"
TORCH_TEST="ON"

# Parse args
for arg in "$@"; do
    case "${arg,,}" in
        nn)
            ENABLE_NN="ON"
            ;;
        debug)
            BUILD_TYPE="Debug"
            ;;
        use_gpu)
            USE_GPU="ON"
            ;;
        force_gpu)
            FORCE_GPU="ON"
            ;;
        no_torch_test)
            TORCH_TEST="OFF"
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            show_usage
            exit 1
            ;;
    esac
done

# Construct build directory name
BUILD_DIR="cmake-build"
[[ "$BUILD_TYPE" == "Debug" ]] && BUILD_DIR+="-debug"
[[ "$ENABLE_NN" == "ON" ]] && BUILD_DIR+="-nn"

# Detect GPU and CUDA
HAS_GPU="unknown"
HAS_CUDA="false"

if command -v lspci >/dev/null 2>&1; then
    if lspci | grep -i nvidia >/dev/null 2>&1; then
        HAS_GPU="true"
    else
        HAS_GPU="false"
    fi
else
    echo "WARNING: 'lspci' not found. Cannot determine if a GPU is present."
    echo "To install: sudo apt install pciutils"
fi

if command -v nvcc >/dev/null 2>&1; then
    HAS_CUDA="true"
fi

# Safety checks
if [[ "$USE_GPU" == "ON" && "$FORCE_GPU" != "ON" ]]; then
    if [[ "$HAS_GPU" != "true" ]]; then
        echo "ERROR: 'use_gpu' flag passed, but no NVIDIA GPU was detected or GPU detection failed."
        if ! command -v lspci >/dev/null 2>&1; then
            echo "It looks like 'lspci' is not installed. Please install 'pciutils' package."
            echo "  On Debian/Ubuntu: sudo apt install pciutils"
            echo "  On Fedora: sudo dnf install pciutils"
            echo "  On Arch Linux: sudo pacman -S pciutils"
        else
            echo "Ensure a supported NVIDIA GPU is available."
        fi
        echo "Alternatively, use 'force_gpu' to override this check."
        exit 1
    elif [[ "$HAS_CUDA" != "true" ]]; then
        echo "ERROR: 'use_gpu' flag passed, GPU found but CUDA toolkit is NOT installed."
        echo "  Please install CUDA: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
elif [[ "$HAS_GPU" == "true" && "$ENABLE_NN" == "ON" && "$USE_GPU" != "ON" && "$FORCE_GPU" != "ON" ]]; then
    echo "WARNING: NVIDIA GPU detected but 'use_gpu' or 'force_gpu' flag not passed."
    echo "  You are about to use CPU-only libtorch despite having a GPU."
    echo "  To enable GPU acceleration, re-run the script with 'use_gpu' or 'force_gpu'."
    read -p "Do you want to continue with CPU-only libtorch? [y/N]: " confirm
    confirm="${confirm,,}"
    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# Download libtorch if needed
TORCH_DIR="lib/libtorch"
if [[ "$ENABLE_NN" == "ON" && ! -d "$TORCH_DIR" ]]; then
    mkdir -p lib
    echo "libtorch not found. Preparing to download..."

    if [[ "$USE_GPU" == "ON" || "$FORCE_GPU" == "ON" ]]; then
        echo "Downloading GPU-enabled libtorch (default: CUDA 11.8). To use a different CUDA version, edit this script."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip"
    else
        echo "Downloading CPU-only libtorch..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip"
    fi

    curl -L -o lib/libtorch.zip "$LIBTORCH_URL"
    unzip lib/libtorch.zip -d lib/
    rm lib/libtorch.zip
    echo "libtorch installed in $TORCH_DIR"
fi

# Create and enter build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring: BUILD_TYPE=$BUILD_TYPE | ENABLE_NEURALNETS=$ENABLE_NN"
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_NEURALNETS=$ENABLE_NN ..

# Build
echo "Building..."
make -j$(nproc)

echo "Build complete: $BUILD_DIR"

cd ..

#Testing some parts
if [[ "$ENABLE_NN" == "ON" && "$TORCH_TEST" == "ON" ]]; then
    echo ""
    echo "Running torch test (utils/torch_test)..."
    ./utils/torch_test/run_test.sh
fi
