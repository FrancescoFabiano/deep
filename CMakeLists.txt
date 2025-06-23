cmake_minimum_required(VERSION 3.14)
project(deep VERSION 1.0 LANGUAGES CXX)

# === Options ===
option(ENABLE_NEURALNETS "Enable neural networks (requires libtorch)" OFF)

# === Build directories ===
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# === Default build type ===
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# === Include directories ===
include_directories(
        src
        src/actions
        src/argparse
        src/bisimulation
        src/domain
        src/formulae
        src/heuristics
        src/heuristics/strategies
        src/heuristics/strategies/epg
        src/parse
        src/states
        src/states/representations/kripke
        src/states/representations/kripke/helpers
        src/utilities
        src/search
        lib/bisimulation
)

# === Templated headers (always included for IDE visibility) ===
set(TEMPLATIC_HEADERS
        src/heuristics/HeuristicsManager.h
        src/heuristics/strategies/SatisfiedGoals.h
        src/heuristics/strategies/epg/StateLevel.h
        src/heuristics/strategies/epg/PlanningGraph.h
        src/heuristics/strategies/neuralnets/TrainingDataset.h
        src/search/SpaceSearcher.h
        src/search/search_strategies/BreadthFirst.h
        src/search/search_strategies/DepthFirst.h
        src/search/search_strategies/BestFirst.h
        src/states/State.h
        src/utilities/SetHelper.h
)

# === Optional neural net headers ===
if(ENABLE_NEURALNETS)
    list(APPEND TEMPLATIC_HEADERS
            src/heuristics/strategies/neuralnets/GraphNN.h
    )
endif()

set(SOURCES ${TEMPLATIC_HEADERS})

# === Automatically collect all .cpp sources ===
get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${INCLUDE_DIRS})
    file(GLOB TMP_SOURCES CONFIGURE_DEPENDS "${dir}/*.cpp")
    list(APPEND SOURCES ${TMP_SOURCES})
endforeach()

# === Flex and Bison ===
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

FLEX_TARGET(Lexer src/parse/lcp.lex ${BUILD_DIR}/lex.cpp)
BISON_TARGET(Parser src/parse/lcp.y ${BUILD_DIR}/bison.cpp)
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

list(APPEND SOURCES ${FLEX_Lexer_OUTPUTS} ${BISON_Parser_OUTPUTS})

# === External libraries ===
add_subdirectory(lib/CLI11)

# === Neural Net (Torch) Setup ===
if(ENABLE_NEURALNETS)
    message(STATUS "Neural networks are ENABLED")

    set(TORCH_DIR "${CMAKE_SOURCE_DIR}/lib/libtorch")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

    find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
    if(NOT Torch_FOUND)
        message(FATAL_ERROR "Torch not found in ${TORCH_DIR}. Make sure it's downloaded.")
    endif()

    include_directories(${TORCH_INCLUDE_DIRS})

    # === CUDA Detection ===
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
    else()
        if(Torch_LIBRARIES MATCHES "cuda")
            message(FATAL_ERROR "CUDA not found, but GPU-enabled libtorch is being used. Please install CUDA toolkit.")
        else()
            message(STATUS "Using CPU-only libtorch.")
        endif()
    endif()
endif()


# === Executable ===
add_executable(deep ${SOURCES})

# === Compiler flags ===
target_compile_options(deep PRIVATE -Wall -Wextra -Wpedantic -ansi -Wfatal-errors -pthread -std=c++20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    target_compile_options(deep PRIVATE -g)
    target_compile_definitions(deep PRIVATE DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    target_compile_options(deep PRIVATE -O3)
endif()

# === Link libraries ===
target_link_libraries(deep PRIVATE pthread rt CLI11::CLI11)

if(ENABLE_NEURALNETS)
    target_link_libraries(deep PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(deep PRIVATE USE_NEURALNETS)
endif()

# === Output directories ===
set_target_properties(deep PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}
)

# === Ensure output directories ===
add_custom_target(prepare_build_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
        COMMENT "Ensuring build and bin directories exist"
)
add_dependencies(deep prepare_build_dir)

# === Cleaning ===
add_custom_target(clean_build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILD_DIR}
        COMMENT "Cleaning build and binary directories"
)

add_custom_target(clean_out
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/out
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.tmp
        COMMENT "Cleaning output directory and temporary files"
)

add_custom_target(clear
        DEPENDS clean_build clean_out
        COMMENT "Clearing all build and output files"
)

add_custom_target(fresh
        DEPENDS clear
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/doxygen/Docs
        COMMENT "Performing a fresh cleanup"
)
