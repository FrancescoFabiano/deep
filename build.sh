#!/bin/bash

# Default to Debug if no argument is provided
BUILD_TYPE=${1:-Debug}

# Define build directory based on build type
BUILD_DIR="cmake-build-${BUILD_TYPE,,}"

# Create build directory if it doesn't exist
mkdir -p $BUILD_DIR

# Navigate to the build directory
cd $BUILD_DIR

# Run CMake with the specified build type
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

# Build the project using all available CPU cores
make -j$(nproc)

# Navigate back to the project root
cd ..