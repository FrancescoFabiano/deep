cmake_minimum_required(VERSION 3.14)
project(onnxruntime_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    ONNXRUNTIME_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/onnxruntime"
    CACHE PATH
    "Path to the ONNX Runtime installation"
)

if(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found under '${ONNXRUNTIME_DIR}/include'. "
        "Build deep with neural-network support first."
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR
        "ONNX Runtime library not found at '${ONNXRUNTIME_LIB}'. "
        "Build deep with neural-network support first."
    )
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(
    onnxruntime
    PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
)

add_executable(onnx_test onnx_test.cpp)

target_link_libraries(onnx_test PRIVATE onnxruntime)

if(UNIX)
    target_link_libraries(onnx_test PRIVATE pthread dl)
endif()

set_target_properties(
    onnx_test
    PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    SKIP_BUILD_RPATH FALSE
    SKIP_INSTALL_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
)
