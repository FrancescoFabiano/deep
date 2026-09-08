cmake_minimum_required(VERSION 3.14)
project(deep VERSION 1.0 LANGUAGES CXX)

# === Checks ===

# Check that uint64_t is actually 64 bits for the hashing function and ONNX usage
if(CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR "This project requires a 64-bit architecture.")
endif()


# === Options ===

option(ENABLE_NEURALNETS "Enable use of ONNX Runtime" OFF)
option(ENABLE_CUDA "Enable CUDA support for ONNX Runtime" OFF)


# === Use ccache if available ===

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()


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


# === Templatic headers ===

set(TEMPLATIC_HEADERS
        src/heuristics/HeuristicsManager.h
        src/heuristics/strategies/SatisfiedGoals.h
        src/heuristics/strategies/epg/StateLevel.h
        src/heuristics/strategies/epg/PlanningGraph.h
        src/heuristics/strategies/neuralnets/TrainingDataset.h
        src/search/SpaceSearcher.h
        src/search/search_strategies/BreadthFirst.h
        src/search/search_strategies/DepthFirst.h
        src/search/search_strategies/best_first/BestFirst.h
        src/search/search_strategies/best_first/HeuristicFirst.h
        src/search/search_strategies/best_first/Astar.h
        src/states/State.h
        src/utilities/SetHelper.h
)

if(ENABLE_NEURALNETS)
    list(APPEND TEMPLATIC_HEADERS
            src/heuristics/strategies/neuralnets/GraphNN.h
            src/heuristics/strategies/neuralnets/FringeEvalRL.h
            src/search/search_strategies/best_first/RL_BestFirst.h
    )
endif()

set(SOURCES ${TEMPLATIC_HEADERS})


# === Automatically collect all .cpp sources ===

get_property(
        INCLUDE_DIRS
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY INCLUDE_DIRECTORIES
)

foreach(dir ${INCLUDE_DIRS})
    file(GLOB TMP_SOURCES CONFIGURE_DEPENDS "${dir}/*.cpp")
    list(APPEND SOURCES ${TMP_SOURCES})
endforeach()


# === Flex and Bison ===

# Keep the legacy Deep parser while EPDDL integration is developed.
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

FLEX_TARGET(
        Lexer
        src/parse/lcp.lex
        ${BUILD_DIR}/lex.cpp
)

BISON_TARGET(
        Parser
        src/parse/lcp.y
        ${BUILD_DIR}/bison.cpp
)

ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

list(APPEND SOURCES
        ${FLEX_Lexer_OUTPUTS}
        ${BISON_Parser_OUTPUTS}
)


# === Doxygen ===

find_package(Doxygen OPTIONAL_COMPONENTS dot)


# === Executable ===

add_executable(deep ${SOURCES})


# === External libraries ===

set(XXHASH_BUILD_XXHSUM OFF)
set(BUILD_SHARED_LIBS OFF)

# Boost must be available before Plank targets are configured.
find_package(Boost CONFIG REQUIRED)

add_subdirectory(lib/CLI11)

add_subdirectory(
        lib/xxHash/build/cmake
        ${CMAKE_BINARY_DIR}/xxhash_build
        EXCLUDE_FROM_ALL
)


# === Plank EPDDL frontend ===
#
# Plank is used only for parsing, type checking and grounding EPDDL.
# Deep will convert the resulting Plank/DEL representation into its
# own native data structures before planning.

add_subdirectory(
        lib/plank/src/lib/del
        ${CMAKE_BINARY_DIR}/plank_del
        EXCLUDE_FROM_ALL
)

add_subdirectory(
        lib/plank/src/lib/epddl
        ${CMAKE_BINARY_DIR}/plank_epddl
        EXCLUDE_FROM_ALL
)

# Plank headers use Boost directly (e.g. boost::dynamic_bitset).
# Make Boost available while compiling both Plank libraries and to
# consumers of their public headers.
target_link_libraries(del_lib PUBLIC Boost::boost)
target_link_libraries(epddl_lib PUBLIC Boost::boost)


# === Precompiled Header ===

target_precompile_headers(
        deep
        PRIVATE
        src/utilities/pch.h
)


# === Neural Net (ONNX) Runtime Setup ===

if(ENABLE_NEURALNETS)
    message(STATUS "ONNX inference is ENABLED")

    set(
            ONNXRUNTIME_DIR
            "${CMAKE_SOURCE_DIR}/lib/onnxruntime"
            CACHE PATH
            "Path to ONNX Runtime"
    )

    if(APPLE)
        set(
                ONNXRUNTIME_LIB
                "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
        )
    else()
        set(
                ONNXRUNTIME_LIB
                "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
        )
    endif()

    # Add ONNX Runtime as an IMPORTED shared library target.
    add_library(onnxruntime SHARED IMPORTED GLOBAL)

    set_target_properties(
            onnxruntime
            PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
    )

    # Enable runtime inference.
    target_compile_definitions(
            deep
            PRIVATE
            USE_NEURALNETS
    )

    target_link_libraries(
            deep
            PRIVATE
            onnxruntime
    )

    # Embed runtime path for libonnxruntime.
    set_target_properties(
            deep
            PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
            SKIP_BUILD_RPATH FALSE
            SKIP_INSTALL_RPATH FALSE
            BUILD_WITH_INSTALL_RPATH TRUE
    )

    if(ENABLE_CUDA)
        find_package(CUDAToolkit QUIET)

        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA support ENABLED and CUDA Toolkit FOUND")

            target_compile_definitions(
                    deep
                    PRIVATE
                    USE_CUDA
            )

            target_link_libraries(
                    deep
                    PRIVATE
                    CUDA::cudart
                    CUDA::cuda_driver
            )
        else()
            message(
                    WARNING
                    "ENABLE_CUDA is ON but CUDA Toolkit not found - falling back to CPU mode"
            )
        endif()
    else()
        message(STATUS "CUDA support is DISABLED")
    endif()
endif()


# === Compiler flags ===

target_compile_options(
        deep
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wfatal-errors
        -pthread
        -std=c++20
)

if(CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(STATUS "Building in Debug mode")

    target_compile_options(
            deep
            PRIVATE
            -g
    )

    target_compile_definitions(
            deep
            PRIVATE
            DEBUG
    )

elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr][Ee][Ll][Ee][Aa][Ss][Ee]$")
    message(STATUS "Building in Release mode")

    target_compile_options(
            deep
            PRIVATE
            -O3
    )
endif()


# === Link libraries ===

find_package(Threads REQUIRED)

target_link_libraries(
        deep
        PRIVATE
        Threads::Threads
        CLI11::CLI11
        xxHash::xxhash
        Boost::boost
        del_lib
        epddl_lib
)


# librt exists only on Linux.
if(UNIX AND NOT APPLE)
    target_link_libraries(
            deep
            PRIVATE
            rt
    )
endif()


# === Output directories ===

set_target_properties(
        deep
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}
)


# === Ensure output directories ===

add_custom_target(
        prepare_build_dir
        ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
        COMMENT "Ensuring build and bin directories exist"
)

add_dependencies(
        deep
        prepare_build_dir
)


# === Cleaning ===

add_custom_target(
        clean_build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILD_DIR}
        COMMENT "Cleaning build and binary directories"
)

add_custom_target(
        clean_out
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/out
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.tmp
        COMMENT "Cleaning output directory and temporary files"
)

add_custom_target(
        clear
        DEPENDS
        clean_build
        clean_out
        COMMENT "Clearing all build and output files"
)

add_custom_target(
        fresh
        DEPENDS
        clear
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/docs/html
        COMMENT "Performing a fresh cleanup"
)


# === Doxygen Documentation ===

if(DOXYGEN_FOUND)
    set(
            DOXYGEN_INPUT_DIR
            ${CMAKE_SOURCE_DIR}/src
    )

    set(
            DOXYGEN_OUTPUT_DIR
            ${CMAKE_SOURCE_DIR}/docs
    )

    set(
            DOXYGEN_CONFIG_FILE
            ${CMAKE_BINARY_DIR}/Doxyfile
    )

    configure_file(
            ${CMAKE_SOURCE_DIR}/docs/Doxyfile.in
            ${DOXYGEN_CONFIG_FILE}
            @ONLY
    )

    add_custom_target(
            doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
    )
else()
    message(
            STATUS
            "Doxygen not found. Documentation target will be skipped."
    )
endif()