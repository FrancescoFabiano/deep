#!/bin/bash

set -e

# --------------------------
# Help message
# --------------------------
show_usage() {
    echo "Usage: ./build.sh [nn] [debug] [use_gpu] [force_gpu] [install_all]"
    echo ""
    echo "Options:"
    echo "  nn              Enable neural networks (downloads ONNX Runtime if not present)"
    echo "  debug           Build with Debug flags (default is Release)"
    echo "  use_gpu         Use GPU-accelerated ONNX Runtime (requires NVIDIA GPU and CUDA installed)"
    echo "  force_gpu       Force install of GPU ONNX Runtime without checking for CUDA or GPU"
    echo "  install_all     Automatically install required system packages (requires sudo)"
    echo "  no_onnx_test   Skip running the onnx tests before build"

    echo ""
    echo "Examples:"
    echo "  ./build.sh                       # Release build without NN"
    echo "  ./build.sh nn                    # Release build with ONNX Runtime (CPU)"
    echo "  ./build.sh debug nn              # Debug build with ONNX Runtime (CPU)"
    echo "  ./build.sh nn use_gpu            # Use ONNX Runtime with CUDA if compatible"
    echo "  sudo ./build.sh install_all      # Install required packages"
}

# --------------------------
# OS Detection & Restriction
# --------------------------
OS="$(uname -s)"
if [[ "$OS" != "Linux" ]]; then
    echo "ERROR: This build script is currently supported only on Linux."
    exit 1
fi

# --------------------------
# Default options
# --------------------------
BUILD_TYPE="Release"
ENABLE_NN="OFF"
USE_GPU="OFF"
FORCE_GPU="OFF"
ONNX_TEST="ON"
INSTALL_ALL="OFF"

for arg in "$@"; do
    case "${arg,,}" in
        nn) ENABLE_NN="ON" ;;
        debug) BUILD_TYPE="Debug" ;;
        use_gpu) USE_GPU="ON" ;;
        force_gpu) FORCE_GPU="ON" ;;
        no_onnx_test) ONNX_TEST="OFF" ;;
        install_all) INSTALL_ALL="ON" ;;
        -h|--help) show_usage; exit 0 ;;
        *) echo "Unknown option: $arg"; show_usage; exit 1 ;;
    esac
done

# --------------------------
# Sudo/root check
# --------------------------
IS_ROOT=0
if [[ "$EUID" -eq 0 ]]; then
    IS_ROOT=1
elif command -v sudo &>/dev/null; then
    IS_ROOT=1
fi

# --------------------------
# Package check
# --------------------------
REQUIRED_PACKAGES=(build-essential cmake bison flex libboost-dev unzip curl)
MISSING=()

echo "Checking for required system packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "All required packages are installed."
else
    echo "Missing packages:"
    for pkg in "${MISSING[@]}"; do echo "  - $pkg"; done

    if [[ "$INSTALL_ALL" == "ON" ]]; then
        if [[ "$IS_ROOT" -eq 1 ]]; then
            echo "Installing missing packages..."
            sudo apt-get update
            sudo apt-get install -y "${MISSING[@]}"
        else
            echo "ERROR: Missing sudo/root for installing packages."
            exit 1
        fi
    else
        read -p "Install missing packages now? [y/N]: " confirm
        confirm="${confirm,,}"
        if [[ "$confirm" == "y" || "$confirm" == "yes" ]]; then
            if [[ "$IS_ROOT" -eq 1 ]]; then
                sudo apt-get update
                sudo apt-get install -y "${MISSING[@]}"
            else
                echo "ERROR: Cannot install without sudo/root."
                exit 1
            fi
        else
            echo "Skipping package installation."
        fi
    fi
fi

# --------------------------
# GPU/CUDA check
# --------------------------
HAS_GPU="unknown"
HAS_CUDA="false"

if command -v lspci &>/dev/null; then
    if lspci | grep -i nvidia &>/dev/null; then
        HAS_GPU="true"
    else
        HAS_GPU="false"
    fi
fi

if command -v nvcc &>/dev/null; then
    HAS_CUDA="true"
fi

if [[ "$USE_GPU" == "ON" && "$FORCE_GPU" != "ON" ]]; then
    if [[ "$HAS_GPU" != "true" ]]; then
        echo "ERROR: 'use_gpu' requested but no NVIDIA GPU detected."
        exit 1
    elif [[ "$HAS_CUDA" != "true" ]]; then
        echo "ERROR: NVIDIA GPU found but CUDA not installed."
        echo "Install CUDA from https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
fi

# --------------------------
# Architecture detection for ONNX Runtime
# --------------------------
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)
        ONNX_ARCH="linux-x64"
        ;;
    aarch64 | arm64)
        ONNX_ARCH="linux-aarch64"
        ;;
    *)
        echo "ERROR: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# ONNX Runtime version to install
ONNX_VER="1.22.0"

# --------------------------
# Download ONNX Runtime
# --------------------------
ONNX_DIR="lib/onnxruntime"
if [[ "$ENABLE_NN" == "ON" && ! -d "$ONNX_DIR" ]]; then
    mkdir -p lib
    echo "Downloading ONNX Runtime for architecture $ONNX_ARCH..."

    if [[ "$USE_GPU" == "ON" || "$FORCE_GPU" == "ON" ]]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-${ONNX_ARCH}-gpu-${ONNX_VER}.tgz"
    else
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-${ONNX_ARCH}-${ONNX_VER}.tgz"
    fi

    echo "Downloading from: $ONNX_URL"
    curl -fL "$ONNX_URL" -o lib/onnxruntime.tgz

    if [[ ! -f lib/onnxruntime.tgz ]]; then
        echo "ERROR: Failed to download ONNX Runtime archive."
        exit 1
    fi

    echo "Extracting ONNX Runtime..."
    tar -xzf lib/onnxruntime.tgz -C lib/
    rm lib/onnxruntime.tgz

    # Find extracted directory (the tarball creates a directory starting with onnxruntime-)
    EXTRACTED_DIR=$(ls -d lib/onnxruntime-* | head -1)
    if [[ -z "$EXTRACTED_DIR" ]]; then
        echo "ERROR: Could not find extracted ONNX Runtime directory."
        exit 1
    fi

    # Rename to a consistent path
    mv "$EXTRACTED_DIR" "$ONNX_DIR"

    echo "ONNX Runtime installed to $ONNX_DIR"
fi


# --------------------------
# ONNX Runtime test
# --------------------------
if [[ "$ENABLE_NN" == "ON" && "$ONNX_TEST" == "ON" ]]; then
    echo
    echo "============================"
    echo "Running ONNX Runtime test..."
    echo "============================"

    ./utils/onnx_test/run_test.sh
fi



# --------------------------
# Configure and build
# --------------------------
BUILD_DIR="cmake-build"
[[ "$BUILD_TYPE" == "Debug" ]] && BUILD_DIR+="-debug"
[[ "$ENABLE_NN" == "ON" ]] && BUILD_DIR+="-nn"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_NEURALNETS=$ENABLE_NN -DENABLE_CUDA=$USE_GPU ..

echo "Compiling..."
make -j$(nproc)

cd ..
echo "Build completed successfully."