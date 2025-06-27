cmake_minimum_required(VERSION 3.14)
project(onnxruntime_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to ONNX Runtime installation - adjust this to your system
set(ONNXRUNTIME_DIR "../../lib/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(onnx_test test_onnx.cpp)

target_link_libraries(onnx_test PRIVATE onnxruntime)

# For Linux: link pthread and dl if needed
if(UNIX)
    target_link_libraries(onnx_test PRIVATE pthread dl)
endif()
