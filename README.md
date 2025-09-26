# deep

**deep** (Dynamic Epistemic logic-basEd Planner) is a multi-agent epistemic planner that operates over the full scope of mA*, leveraging optimized search algorithms and heuristics. It is designed for scalable planning in multi-agent domains, supporting advanced reasoning over knowledge and belief.

For a more detailed overview, please take a look at the works referenced in the Bibliography section at the end of this README.

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

Or a release build with neural networks enabled (the one used for GNN testing):

    ./build.sh nn

For more granular building options, please refer to the next two sections where the building process is explained without using the script.
Note that the script abstracts away many details, so if you want to use more granular building options, please make sure to follow the installation procedure in `build.sh` to ensure you have all the necessary components.

> Note: This script and some dependencies may require execute permissions. Use the `chmod` command or a similar tool to grant the necessary permissions.

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

# Plan using the 'GNN' heuristic and 'Astar' search
# on the default model. This includes the goal in the e-state representation for GNN.
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

### Main References Experiments -- GNN Testing
To replicate the experiments, please follow the instructions provided in the exp/gnn_exp directory, using the included pre-trained models.
Alternatively, scripts for generating the training data and training the models are also provided.
Explanations of this procedure are deferred to C++ code comments.

To build please execute the building script with `nn` argument to activate ONNX support and, therefore, GNN-based heuristics like this:

```console
./build.sh nn
```

## Citation

If you use deep in your research, please cite:

```bibtex'
@misc{briglia2025scaling,
      title={Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics}, 
      author={Giovanni Briglia and Francesco Fabiano and Stefano Mariani},
      year={2025},
      eprint={2508.12840},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.12840}, 
}
```

## Bibliography

#### Strong integration of heuristics
- Fabiano, F., Platt, T., Son, T. C., & Pontelli, E. (2024).  
  *𝓗-EFP: Bridging Efficiency in Multi-agent Epistemic Planning with Heuristics.*  
  In *PRIMA 2024: Principles and Practice of Multi-Agent Systems* (pp. 81–86).  
  DOI: [10.1007/978-3-031-77367-9_7](https://doi.org/10.1007/978-3-031-77367-9_7)


#### EFP 2.0 with updated transition function and multiple e-state representations
- Fabiano, F., Burigana, A., Dovier, A., & Pontelli, E. (2020).  
  *EFP 2.0: A Multi-Agent Epistemic Solver with Multiple E-State Representations.*  
  In *Proceedings of the 30th International Conference on Automated Planning and Scheduling (ICAPS 2020)*, Nancy, France (pp. 101–109).  
  [Link](https://ojs.aaai.org/index.php/ICAPS/article/view/6650)


#### Foundational work on optimizing the code and transition function
- Burigana, A., Fabiano, F., Dovier, A., & Pontelli, E. (2020).  
  *Modelling Multi-Agent Epistemic Planning in ASP.*  
  *Theory and Practice of Logic Programming*, 20(5), 593–608.  
  DOI: [10.1017/S1471068420000289](https://doi.org/10.1017/S1471068420000289)


- Fabiano, F., Riouak, I., Dovier, A., & Pontelli, E. (2019).  
  *Non-Well-Founded Set Based Multi-Agent Epistemic Action Language.*  
  In *Proceedings of the 34th Italian Conference on Computational Logic (CILC 2019)*, Trieste, Italy (pp. 242–259).  
  [Link](https://ceur-ws.org/Vol-2396/paper38.pdf)



#### EFP version 1.0
- Le, T., Fabiano, F., Son, T. C., & Pontelli, E. (2018).  
  *EFP and PG-EFP: Epistemic Forward Search Planners in Multi-Agent Domains.*  
  In *Proceedings of the 28th International Conference on Automated Planning and Scheduling (ICAPS 2018)*, Delft, Netherlands (pp. 161–170).  
  [Link](https://aaai.org/ocs/index.php/ICAPS/ICAPS18/paper/view/17733)


#### Works on EPDDL (final version to come)
- Burigana, A. & Fabiano, F. (2022).  
  *The Epistemic Planning Domain Definition Language (Short Paper).*  
  In *Proceedings of the 10th Italian Workshop on Planning and Scheduling (IPS 2022)*, Udine, Italy.  
  [Link](https://ceur-ws.org/Vol-3345/paper5_2497.pdf)

- Fabiano, F., Srivastava, B., Lenchner, J., Horesh, L., Rossi, F., & Ganapini, M. B. (2021).  
  *E-PDDL: A Standardized Way of Defining Epistemic Planning Problems.*  
  *CoRR*, abs/2107.08739.  
  [arXiv](https://arxiv.org/abs/2107.08739)



#### Work on mA* (language currently used)
- Baral, C., Gelfond, G., Pontelli, E., & Son, T. C. (2015).  
  *An action language for multi-agent domains: Foundations.*  
  *CoRR*, abs/1511.01960.


## Known Issues

- [ ] When running long action lists without bisimulation, a `NULL` world is generated in the state.

  ### Steps to Reproduce
  1. Run the following action list:  
     left_b, b_check_1, right_a, b_check_1, a_check_3, tell_b_b1_1, a_check_3, a_check_3, b_check_1, a_check_3

  2. On this file:  
     exp/gnn_exp/batch1/CC/Test/CC_2_2_3__pl_3.txt

  3. Using the command:

     ```console
     ./cmake-build-debug/bin/deep exp/gnn_exp/batch1/CC/Test/CC_2_2_3__pl_3.txt \
       -v \
       -e \
       --execute_actions \
       left_b, \
       b_check_1, \
       right_a, \
       b_check_1, \
       a_check_3, \
       tell_b_b1_1, \
       a_check_3, \
       a_check_3, \
       b_check_1, \
       a_check_3
     ```

  ### Expected Behavior
  - The state should remain valid throughout execution.
  - No `NULL` world should be generated.

  ### Actual Behavior
  - A `NULL` world is generated in the state during execution.

  ### Additional Notes
  - Make sure to comment out the `return true` after a goal is found in order to print the final state.

  ### Environment
  - **Binary:** ./cmake-build-debug/bin/deep
  - **Dataset/File:** exp/gnn_exp/batch1/CC/Test/CC_2_2_3__pl_3.txt
  - **Execution Mode:** --execute_actions
  - **Platform:** Ubuntu 22.04

## License

This project is licensed under the GNU General Public License v3.0 – see the LICENSE file for details.
