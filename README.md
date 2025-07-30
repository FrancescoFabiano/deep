# deep

**deep** (Dynamic Epistemic logic-basEd Planner) is a multi-agent epistemic planner that operates over the full scope of mA*, leveraging optimized search algorithms and heuristics. It is designed for scalable planning in multi-agent domains, supporting advanced reasoning over knowledge and belief.

For a more detailed overview, please take a look at the works referenced in the Bibliography section at the end of this README.

## AAAI 2026
To replicate the experiments, please follow the instructions provided in the exp/aaai26 directory, using the included pre-trained models.
Alternatively, scripts for generating the training data and training the models are also provided.
Explanations of this procedure are deferred to C++ code comments.

Note that since the repository has been anonymized and compressed into a ZIP archive, Git submodule functionality is not supported.
Versions of the required external libraries are included within the archive.

To build please execute the building script with `nn` argument to activate ONNX support and, therefore, GNN-based heuristics.

## Features

- Multi-agent epistemic planning using Dynamic Epistemic Logic (DEL)
- Optimized search strategies: BFS, DFS, A*, etc.
- Support for heuristics, including optional neural network-based heuristics
- Modular and extensible C++ codebase
- Templated heuristics and flexible state representations

> EPDDL support is planned but not yet integrated. The current version works with the `mA*` language.

## Requirements

- C++20 compiler (e.g., g++ 10+, clang 11+)
- CMake 3.14 or higher
- Boost (header-only)
- Bison and Flex
- Python 3.6+ (for optional scripts and comparisons)
- ONNX Runtime (installed automatically through build script) (optional, for neural network heuristics)
- LaTeX (optional, for PDF state visualization)
- Doxygen (optional, for documentation generation)
- GraphViz  (optional, for PDF state visualization and documentation generation)


> CUDA is **not required**. GPU support via ONNX Runtime with CUDA is available but **not tested**.

### Linux Dependencies

Install required tools and libraries:

    sudo apt-get install build-essential cmake bison flex libboost-dev unzip curl

## Installation

### Clone the repository

    git clone --recurse-submodules https://github.com/FrancescoFabiano/deep.git
    cd deep

The `--recurse-submodules` flag ensures that all submodules (like CLI11) are cloned as well.
> Installation procedure of all the external libraries is deferred to the respective READMEs. The external libraries are found in the `lib/' folder.

### Build the project

Use the build script with `-h` to view all available options:

    ./build.sh -h

For example, a simple debug build:

    ./build.sh debug

Or a release build with neural networks enabled (the one used for AAAI testing):

    ./build.sh nn

For more granular building options, please refer to the next two sections where the building process is explained without using the script.
Note that the script abstracts away many details, so if you want to use more granular building options, please make sure to follow the installation procedure in `build.sh` to ensure you have all the necessary components.
##### CMake Build Options

You can customize the build using these options:

- `ENABLE_NEURALNETS=ON` – Enable ONNX Runtime for heuristic inference
- `ENABLE_CUDA=ON`       – Attempt to enable CUDA inference if CUDA Toolkit is found

Example:

    cmake .. -DENABLE_NEURALNETS=ON -DENABLE_CUDA=OFF

> CUDA acceleration is supported if available, but **not required or tested**.


#### Make

Common `make` targets after building:

    make             # Compile the planner
    make doxygen     # Generate documentation (see docs/html/index.html)
    make clean       # Remove build files
    make clean_build # Remove build and binary directories
    make clean_out   # Remove output files (e.g., PDF visualizations)
    make clear       # Clean both build and output files
    make fresh       # Clean everything, including documentation

If built in a different folder from root, for example `cmake-build-release-nn`, run with -C option like this for documentation generation:

    make -C cmake-build-release-nn doxygen

## Execution Examples

Assuming the binary is located at `./cmake-build-release-nn/bin/deep`, here are some example commands you can run in your terminal:

```console
# Use '-h' or '--help' to see all options (this option is automatically activated when wrong arguments are parsed)
./cmake-build-release-nn/bin/deep -h

# Find a plan for exp/example.txt using BFS
./cmake-build-release-nn/bin/deep exp/example.txt

# Plan using the heuristic 'SUBGOALS' and 'Heuristics First Search' search
./cmake-build-release-nn/bin/deep exp/example.txt -s HFS --heuristics SUBGOALS

# Plan using the 'GNN' heuristic and 'Astar' search with the `--dataset_merged` flag 
# on the default model. This includes the goal in the e-state representation for GNN.
./cmake-build-release-nn/bin/deep exp/example.txt -s Astar --heuristics GNN --dataset_merged

# Plan using the 'GNN' heuristic and 'Astar' search with the `--dataset_merged` flag 
# on the default model. Useful for exmaples and models in exp/batch3
./cmake-build-release-nn/bin/deep exp/example.txt -s Astar --heuristics GNN

# Execute actions [open_a, peek_a] step by step
./cmake-build-release-nn/bin/deep exp/example.txt --execute_actions open_a peek_a

# Run 3 planner configurations in parallel (portfolio search)
./cmake-build-release-nn/bin/deep exp/example.txt --portfolio_threads 3
```


## Scripts

Helper scripts for testing and debugging are located in the `scripts/` directory.
All scripts contain a `-h` or `--help` option to display usage information.

## Benchmarks

Benchmarks are available in the `exp/` directory.

## Citation

If you use deep in your research, please cite:
Anonymized

## Bibliography

Anonymized

## License

This project is licensed under the GNU General Public License v3.0 – see the LICENSE file for details.
