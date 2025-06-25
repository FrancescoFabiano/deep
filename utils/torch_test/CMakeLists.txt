cmake_minimum_required(VERSION 3.10)
project(TestLibTorch)

set(CMAKE_CXX_STANDARD 17)

# Path to Torch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/../../lib/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(test_libtorch test_libtorch.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
target_compile_options(test_libtorch PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=1")