cmake_minimum_required(VERSION 3.14)

project(deep VERSION 1.0 LANGUAGES CXX)

# Define build directories
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# Include directories
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

# Automatically include all source (.cpp) files from include directories
# and manually specify header files containing template implementations

# Header files with template implementations that must be included in the build
set(TEMPLATIC_HEADERS
        src/bisimulation/VectorBisWrapper.h
        src/heuristics/HeuristicsManager.h
        src/heuristics/strategies/neuralnets/GraphNN.h
        src/heuristics/strategies/neuralnets/TrainingDataset.h
        src/heuristics/strategies/SatisfiedGoals.h
        src/heuristics/strategies/epg/StateLevel.h
        src/heuristics/strategies/epg/PlanningGraph.h
        src/search/SpaceSearcher.h
        src/search/search_strategies/BreadthFirst.h
        src/search/search_strategies/DepthFirst.h
        src/search/search_strategies/BestFirst.h
        src/states/State.h
        src/utilities/SetHelper.h
)

# Start the list of source files with those headers
# Note: These aren't compiled directly but included for clarity in IDEs and some build systems
set(SOURCES ${TEMPLATIC_HEADERS})

# Append all .cpp source files found in include directories
get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${INCLUDE_DIRS})
    file(GLOB TMP_SOURCES CONFIGURE_DEPENDS "${dir}/*.cpp")
    list(APPEND SOURCES ${TMP_SOURCES})
endforeach()

# External library
add_subdirectory(lib/CLI11)

# Flex and Bison integration
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

FLEX_TARGET(Lexer src/parse/lcp.lex ${BUILD_DIR}/lex.cpp)
BISON_TARGET(Parser src/parse/lcp.y ${BUILD_DIR}/bison.cpp)
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

# Add generated parser and lexer files to the build
list(APPEND SOURCES ${FLEX_Lexer_OUTPUTS} ${BISON_Parser_OUTPUTS})

# Create the executable
add_executable(deep ${SOURCES})

# Compiler flags
target_compile_options(deep PRIVATE -g -Wall -Wextra -Wpedantic -ansi -Wfatal-errors -pthread -std=c++20)

# Link libraries
target_link_libraries(deep PRIVATE pthread rt CLI11::CLI11)

# Debug macro
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(deep PRIVATE DEBUG)
endif()

# Output directories
set_target_properties(deep PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}
)

# Ensure build and bin directories exist
add_custom_target(prepare_build_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
        COMMENT "Ensuring build and bin directories exist"
)
add_dependencies(deep prepare_build_dir)

# Custom cleaning targets
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