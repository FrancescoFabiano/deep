#!/bin/bash

set -e

# --------------------------
# Help message
# --------------------------
show_usage() {
    echo "Usage: ./build.sh [nn] [debug] [use_gpu] [force_gpu] [no_torch_test] [install_all]"
    echo ""
    echo "Options:"
    echo "  nn              Enable neural networks (downloads libtorch if not present)"
    echo "  debug           Build with Debug flags (default is Release)"
    echo "  use_gpu         Use GPU-accelerated libtorch (requires NVIDIA GPU and CUDA installed)"
    echo "  force_gpu       Force install of GPU libtorch without checking for CUDA or GPU"
    echo "  no_torch_test   Skip running the torch test after build"
    echo "  install_all     Automatically install required system packages (requires sudo)"
    echo ""
    echo "Examples:"
    echo "  ./build.sh                          # Release build without NN"
    echo "  ./build.sh nn                       # Release with NN using CPU libtorch"
    echo "  ./build.sh debug nn                 # Debug with NN using CPU libtorch"
    echo "  ./build.sh nn use_gpu               # Use GPU libtorch if CUDA is installed"
    echo "  sudo ./build.sh install_all         # Install required system packages automatically"
}

# --------------------------
# Default options
# --------------------------
BUILD_TYPE="Release"
ENABLE_NN="OFF"
USE_GPU="OFF"
FORCE_GPU="OFF"
TORCH_TEST="ON"
INSTALL_ALL="OFF"

for arg in "$@"; do
    case "${arg,,}" in
        nn) ENABLE_NN="ON" ;;
        debug) BUILD_TYPE="Debug" ;;
        use_gpu) USE_GPU="ON" ;;
        force_gpu) FORCE_GPU="ON" ;;
        no_torch_test) TORCH_TEST="OFF" ;;
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
REQUIRED_PACKAGES=(build-essential cmake bison flex libboost-dev)
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
            echo "Installing missing packages automatically..."
            sudo apt-get update
            sudo apt-get install -y "${MISSING[@]}"
        else
            echo "ERROR: 'install_all' was specified, but this script is not running with sufficient privileges."
            echo ""
            echo "To install dependencies either:"
            echo "  1) Re-run this script with sudo:"
            echo "     sudo ./build.sh install_all [ARGS]"
            echo ""
            echo "  OR"
            echo "  2) Manually install the missing packages:"
            echo "     sudo apt-get install ${MISSING[*]}"
            echo ""
            exit 1
        fi
    else
        read -p "Do you want to install the missing packages now? [y/N]: " confirm
        confirm="${confirm,,}"
        if [[ "$confirm" == "y" || "$confirm" == "yes" ]]; then
            if [[ "$IS_ROOT" -eq 1 ]]; then
                sudo apt-get update
                sudo apt-get install -y "${MISSING[@]}"
            else
                 echo "ERROR: Cannot install packages without sudo/root privileges."
                 echo ""
                 echo "To install dependencies either:"
                 echo "  1) Re-run this script with sudo:"
                 echo "     sudo ./build.sh install_all [ARGS]"
                 echo ""
                 echo "  OR"
                 echo "  2) Manually install the missing packages:"
                 echo "     sudo apt-get install ${MISSING[*]}"
                 echo ""
                 exit 1
            fi
        else
            echo "Skipping package installation. Build may fail if dependencies are missing."
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
        echo "Use 'force_gpu' to override this check if you know your setup is compatible."
        exit 1
    elif [[ "$HAS_CUDA" != "true" ]]; then
        echo "ERROR: NVIDIA GPU found but CUDA toolkit is not installed."
        echo "Please install CUDA from https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
elif [[ "$HAS_GPU" == "true" && "$ENABLE_NN" == "ON" && "$USE_GPU" != "ON" && "$FORCE_GPU" != "ON" ]]; then
    echo "WARNING: NVIDIA GPU detected but 'use_gpu' or 'force_gpu' flag not set."
    read -p "Continue with CPU-only libtorch? [y/N]: " confirm
    confirm="${confirm,,}"
    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# --------------------------
# Download libtorch if needed
# --------------------------
TORCH_DIR="lib/libtorch"
if [[ "$ENABLE_NN" == "ON" && ! -d "$TORCH_DIR" ]]; then
    mkdir -p lib
    echo "Downloading libtorch..."

    if [[ "$USE_GPU" == "ON" || "$FORCE_GPU" == "ON" ]]; then
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip"
    else
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip"
    fi

    curl -L -o lib/libtorch.zip "$LIBTORCH_URL"
    unzip lib/libtorch.zip -d lib/
    rm lib/libtorch.zip
    echo "libtorch installed to $TORCH_DIR"
fi

# --------------------------
# Configure and build
# --------------------------
BUILD_DIR="cmake-build"
[[ "$BUILD_TYPE" == "Debug" ]] && BUILD_DIR+="-debug"
[[ "$ENABLE_NN" == "ON" ]] && BUILD_DIR+="-nn"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake configuration..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_NEURALNETS=$ENABLE_NN ..

echo "Compiling..."
make -j$(nproc)

cd ..

# --------------------------
# Torch test
# --------------------------
if [[ "$ENABLE_NN" == "ON" && "$TORCH_TEST" == "ON" ]]; then
    echo "Running torch test..."
    ./utils/torch_test/run_test.sh
fi

echo "Build completed successfully."
