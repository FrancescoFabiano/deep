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
    echo "  use_gpu         Use GPU ONNX Runtime (Linux + NVIDIA only)"
    echo "  force_gpu       Force GPU ONNX install (Linux only)"
    echo "  install_all     Install required system packages (apt on Linux, Homebrew on macOS)"
    echo "  no_onnx_test    Skip ONNX tests"
}

# --------------------------
# OS Detection
# --------------------------
OS="$(uname -s)"
if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
    echo "ERROR: Unsupported OS: $OS"
    exit 1
fi

echo "Detected OS: $OS"

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
    arg_lc="$(echo "$arg" | tr '[:upper:]' '[:lower:]')"
    case "$arg_lc" in
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
# macOS GPU restriction
# --------------------------
if [[ "$OS" == "Darwin" && ("$USE_GPU" == "ON" || "$FORCE_GPU" == "ON") ]]; then
    echo "WARNING: CUDA/NVIDIA not supported on macOS. Disabling GPU."
    USE_GPU="OFF"
    FORCE_GPU="OFF"
fi

# --------------------------
# Package installation
# --------------------------
if [[ "$OS" == "Linux" ]]; then
    REQUIRED_PACKAGES=(build-essential cmake bison flex libboost-dev unzip curl)

    MISSING=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            MISSING+=("$pkg")
        fi
    done

    if [[ ${#MISSING[@]} -ne 0 ]]; then
        echo "Missing packages: ${MISSING[*]}"
        if [[ "$INSTALL_ALL" == "ON" ]]; then
            sudo apt-get update
            sudo apt-get install -y "${MISSING[@]}"
        fi
    else
        echo "All required packages installed."
    fi

elif [[ "$OS" == "Darwin" ]]; then
    # Xcode Command Line Tools provide the Apple Clang compiler and make.
    if ! xcode-select -p &>/dev/null; then
        echo "ERROR: Xcode Command Line Tools are required."
        echo "Install them with: xcode-select --install"
        exit 1
    fi

    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew is required on macOS."
        echo 'Install it with:'
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi

    REQUIRED_PACKAGES=(cmake bison flex boost unzip curl)
    MISSING=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! brew list "$pkg" &>/dev/null; then
            MISSING+=("$pkg")
        fi
    done

    if [[ ${#MISSING[@]} -ne 0 ]]; then
        echo "Missing Homebrew packages: ${MISSING[*]}"
        if [[ "$INSTALL_ALL" == "ON" ]]; then
            brew install "${MISSING[@]}"
        else
            echo "Run './build.sh install_all' (plus any other desired options) to install them automatically."
            exit 1
        fi
    else
        echo "All required Homebrew packages installed."
    fi

    # Homebrew's bison/flex are keg-only on common macOS/Homebrew setups.
    # Put their binaries first and expose their prefixes to CMake/pkg-config.
    export PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
    export CMAKE_PREFIX_PATH="$(brew --prefix boost);$(brew --prefix bison);$(brew --prefix flex)${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
fi

# --------------------------
# GPU/CUDA check (Linux only)
# --------------------------
if [[ "$OS" == "Linux" ]]; then
    HAS_GPU="false"
    HAS_CUDA="false"

    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        HAS_GPU="true"
    fi

    if command -v nvcc &>/dev/null; then
        HAS_CUDA="true"
    fi

    if [[ "$USE_GPU" == "ON" && "$FORCE_GPU" != "ON" ]]; then
        if [[ "$HAS_GPU" != "true" ]]; then
            echo "ERROR: No NVIDIA GPU detected."
            exit 1
        elif [[ "$HAS_CUDA" != "true" ]]; then
            echo "ERROR: CUDA not installed."
            exit 1
        fi
    fi
fi

# --------------------------
# Architecture detection
# --------------------------
ARCH="$(uname -m)"

if [[ "$OS" == "Linux" ]]; then
    case "$ARCH" in
        x86_64) ONNX_ARCH="linux-x64" ;;
        aarch64 | arm64) ONNX_ARCH="linux-aarch64" ;;
        *) echo "Unsupported arch: $ARCH"; exit 1 ;;
    esac
else
    case "$ARCH" in
        x86_64) ONNX_ARCH="osx-x86_64" ;;
        arm64) ONNX_ARCH="osx-arm64" ;;
        *) echo "Unsupported macOS arch: $ARCH"; exit 1 ;;
    esac
fi

ONNX_VER="1.22.0"
ONNX_DIR="lib/onnxruntime"

# --------------------------
# Download ONNX Runtime
# --------------------------
if [[ "$ENABLE_NN" == "ON" && ! -d "$ONNX_DIR" ]]; then
    mkdir -p lib

    if [[ "$OS" == "Linux" && ("$USE_GPU" == "ON" || "$FORCE_GPU" == "ON") ]]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-${ONNX_ARCH}-gpu-${ONNX_VER}.tgz"
    else
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-${ONNX_ARCH}-${ONNX_VER}.tgz"
    fi

    echo "Downloading ONNX Runtime..."
    curl -fL "$ONNX_URL" -o lib/onnxruntime.tgz

    tar -xzf lib/onnxruntime.tgz -C lib/
    rm lib/onnxruntime.tgz

    EXTRACTED_DIR=$(ls -d lib/onnxruntime-* | head -1)
    mv "$EXTRACTED_DIR" "$ONNX_DIR"

    echo "ONNX installed at $ONNX_DIR"
fi

# --------------------------
# ONNX Test
# --------------------------
if [[ "$ENABLE_NN" == "ON" && "$ONNX_TEST" == "ON" ]]; then
    if [[ -x ./utils/onnx_test/run_test.sh ]]; then
        if [[ "$OS" == "Darwin" ]]; then
            echo "Running ONNX test on macOS..."
            # Allow the test executable to resolve libonnxruntime.dylib from the downloaded runtime.
            DYLD_LIBRARY_PATH="$PWD/$ONNX_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
                ./utils/onnx_test/run_test.sh
        else
            ./utils/onnx_test/run_test.sh
        fi
    else
        echo "WARNING: ./utils/onnx_test/run_test.sh is missing or not executable; skipping ONNX test."
    fi
fi

# --------------------------
# Build setup
# --------------------------
BUILD_DIR="cmake-build"
[[ "$BUILD_TYPE" == "Debug" ]] && BUILD_DIR+="-debug"
[[ "$BUILD_TYPE" == "Release" ]] && BUILD_DIR+="-release"
[[ "$ENABLE_NN" == "ON" ]] && BUILD_DIR+="-nn"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DENABLE_NEURALNETS=$ENABLE_NN \
      -DENABLE_CUDA=$USE_GPU ..

# --------------------------
# Parallel build
# --------------------------
if [[ "$OS" == "Linux" ]]; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.ncpu)
fi

echo "Compiling with $JOBS threads..."
make -j"$JOBS"

cd ..
echo "Build completed successfully."