# deep

### Dynamic Epistemic logic-basEd Planner

**deep** is a C++20 multi-agent epistemic planner based on **EPDDL** and **Dynamic Epistemic Logic (DEL)**.

Given an epistemic planning domain and problem, deep grounds the task, constructs the corresponding epistemic state, and searches for a sequence of DEL actions that satisfies the goal.

**[EPDDL Guideline](https://arxiv.org/abs/2601.20969)** ·
**[IPC 2026 Benchmarks](https://github.com/ipc2026-epistemic/benchmarks)** ·
**[GNN Heuristics](https://arxiv.org/abs/2508.12840)**

---

## At a glance

```mermaid
flowchart LR

    subgraph IN["1. World Specification"]
        D["Domain"]
        P["Problem"]
        L["Libraries"]
    end

    subgraph GROUND["2. Symbolic Grounding"]
        G["Plank"]
        T["Grounded Task"]
    end

    subgraph CORE["3. Epistemic Planning Loop"]
        S0["Initial Model"]
        F["Frontier"]
        A["DEL Action"]
        U["Product Update"]
        S1["Next Model"]
        Q{"Goal?"}

        S0 --> F
        F --> A
        A --> U
        U --> S1
        S1 --> Q
        Q -- "No" --> F
    end

    H["Heuristics"]
    X["Visited Check + Bisimulation"]
    R["Plan"]

    D --> G
    P --> G
    L --> G
    G --> T

    T --> S0
    Q -- "Yes" --> R

    H -. "guide" .-> F
    X -. "reduce" .-> S1

    classDef file fill:#edf4ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a;
    classDef engine fill:#f5edff,stroke:#8b5cf6,stroke-width:2px,color:#5b21b6;
    classDef task fill:#eef2ff,stroke:#6366f1,stroke-width:2px,color:#3730a3;
    classDef state fill:#ecfdf5,stroke:#22c55e,stroke-width:2px,color:#166534;
    classDef action fill:#ecfeff,stroke:#06b6d4,stroke-width:2px,color:#155e75;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#9a3412;
    classDef assist fill:#fff8e8,stroke:#eab308,stroke-width:2px,stroke-dasharray:7 4,color:#854d0e;
    classDef result fill:#fff1f2,stroke:#e11d48,stroke-width:3px,color:#9f1239;

    class D,P,L file;
    class G engine;
    class T task;
    class S0,F,S1 state;
    class A,U action;
    class Q decision;
    class H,X assist;
    class R result;
```

The core planner uses **Plank** to parse and ground EPDDL tasks. Search operates over Kripke states, with **DEL event-model update** providing the transition semantics. Heuristics and state-space reduction techniques can optionally guide and reduce the search.

### Features

- EPDDL domains, problems, and action libraries
- DEL-based epistemic transitions
- Multi-agent Kripke-state search
- BFS, DFS, IDFS, heuristic search, A*, and RL-oriented search
- Optional visited-state checking and bisimulation
- GNN-based heuristic evaluation
- Dataset-generation infrastructure
- Native ONNX Runtime inference
- Optional Linux/NVIDIA GPU inference

---

## Quick Start

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/FrancescoFabiano/deep.git
cd deep
```

If you already cloned the repository without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### 2. Install dependencies and build

#### Linux

Install dependencies manually:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libboost-dev unzip curl
./build.sh
```

Or let the build script install them for you:

```bash
./build.sh install_all
```

The default Linux build produces:

```text
cmake-build-release/bin/deep
```

#### macOS

Install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Install the required packages through Homebrew:

```bash
brew install cmake boost unzip curl
./build.sh
```

Or let the build script install the missing packages:

```bash
./build.sh install_all
```

The default macOS build also produces:

```text
cmake-build-release/bin/deep
```

### 3. Run a first problem

The core interface is:

```text
deep <domain.epddl> <problem.epddl> [options]
```

Minimal example:

```bash
./cmake-build-release/bin/deep path/to/domain.epddl path/to/problem.epddl
```

With an action library:

```bash
./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl
```

---

## Build Reference

### External dependencies

deep uses the following external components:

| Component | Purpose | Repository |
| --- | --- | --- |
| **Plank** | EPDDL parsing, grounding, and DEL infrastructure | [a-burigana/plank](https://github.com/a-burigana/plank) |
| **xxHash** | Fast hashing | [Cyan4973/xxHash](https://github.com/Cyan4973/xxHash) |
| **CLI11** | Command-line interface | [CLIUtils/CLI11](https://github.com/CLIUtils/CLI11) |
| **IPC 2026 Benchmarks** | EPDDL benchmark suite | [ipc2026-epistemic/benchmarks](https://github.com/ipc2026-epistemic/benchmarks) |

These dependencies are versioned as Git submodules.

Neural-network builds additionally use [ONNX Runtime](https://github.com/microsoft/onnxruntime), which is downloaded automatically when required.

### System requirements

deep requires:

- a C++20 compiler;
- CMake 3.14 or newer;
- Boost;
- Git;
- `curl`;
- `unzip`.

### Build

If you only need the default build, use:

```bash
./build.sh
```

This creates:

```text
cmake-build-release/bin/deep
```

Common build variants:

| Command | Use when you want... |
| --- | --- |
| `./build.sh` | build the default Release binary |
| `./build.sh debug` | build with Debug flags |
| `./build.sh nn` | enable ONNX Runtime / neural-network support |
| `./build.sh debug nn` | combine Debug and neural-network support |
| `./build.sh verify` | enable extra correctness checks |
| `./build.sh install_all` | install required system packages first |
| `./build.sh nn use_gpu` | use CUDA-backed ONNX Runtime on Linux/NVIDIA |
| `./build.sh -h` | print the full build-script help |

Build flags are unordered, so combinations such as `./build.sh install_all debug nn` are valid.

Output directories follow the selected mode:

- Release: `cmake-build-release/`
- Debug: `cmake-build-debug/`
- Release + `nn`: `cmake-build-release-nn/`
- Debug + `nn`: `cmake-build-debug-nn/`

---

## Running deep

### Common patterns

| Goal | Command pattern |
| --- | --- |
| Plan with no libraries | `./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl` |
| Plan with one library | `./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl` |
| Plan with multiple libraries | `./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIB_A.epddl --act_lib LIB_B.epddl` |
| Enable visited-state checking | add `-c` |
| Enable bisimulation | add `-b` |
| Use a specific search | add `-s BFS`, `-s DFS`, `-s HFS`, or `-s Astar` |
| Use SUBGOALS with heuristic search | add `-u SUBGOALS` |
| Execute a known action sequence | add `-e -a action1 action2 ...` |

### Example commands

Planning with a library:

```bash
./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl
```

Planning with visited-state checking and bisimulation:

```bash
./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl -c -b
```

Heuristic search with SUBGOALS:

```bash
./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl -s Astar -u SUBGOALS
```

Direct plan execution:

```bash
./cmake-build-release/bin/deep DOMAIN.epddl PROBLEM.epddl --act_lib LIBRARY.epddl -e -a open_A peek_A
```

### CI smoke tests

The execution workflow validates the planner against
`utils/smoke_cases.tsv`, where each row specifies:

- an EPDDL domain file;
- an EPDDL problem file;
- an optional action library;
- optional grounded actions for `--execute_actions`.

Run the same smoke tests locally with:

```bash
./utils/smoke_test.sh cmake-build-release/bin/deep
```

For the complete CLI reference:

```bash
./cmake-build-release/bin/deep -h
```

---

## IPC 2026 benchmarks

The IPC 2026 Epistemic Planning benchmark suite is included as a submodule at:

```text
exp/ipc2026-benchmarks/
```

Upstream:

**[ipc2026-epistemic/benchmarks](https://github.com/ipc2026-epistemic/benchmarks)**

Initialize it together with the other submodules:

```bash
git submodule update --init --recursive
```

Inspect the available instances with:

```bash
find exp/ipc2026-benchmarks -type f \( -name '*.epddl' -o -name '*.pddl' \) | head -50
```

A benchmark can then be run as:

```bash
./cmake-build-release/bin/deep exp/ipc2026-benchmarks/<benchmark>/domain.epddl exp/ipc2026-benchmarks/<benchmark>/problems/<problem>.epddl --act_lib exp/ipc2026-benchmarks/<benchmark>/<library>.epddl
```

Use the actual domain, problem, and library filenames contained in the checked-out benchmark revision.

---

## Architecture

### EPDDL and DEL

deep uses **EPDDL** as its input language and **Dynamic Epistemic Logic** as its transition semantics.

EPDDL domains, problems, and action libraries are processed using **Plank**,
which provides the parsing, grounding, model-checking, and DEL infrastructure
required by the planner.

The grounded task provides:

- the initial epistemic state;
- the epistemic goal;
- the grounded DEL actions available during search.

Search states are represented as **Kripke structures** with designated worlds,
and successor states are obtained by DEL product update with the corresponding
grounded action models.

For each grounded action, deep stores:

- the action's events and designated events;
- postconditions and preconditions for each event;
- one event relation for each grounded EPDDL observability type;
- formula-valued observability conditions selecting which relation applies to
  each agent in the current source state.

During successor generation, deep first resolves one observability type per
agent on the full source epistemic state, then expands the reachable product
model from designated `(world, event)` pairs.

### Search

deep provides several uninformed, heuristic, and learned search strategies.

State-space management can optionally use visited-state checking and bisimulation contraction. The exact configuration is controlled through the command-line interface.

### GNN heuristics

Kripke structures are naturally represented as labelled directed graphs. deep can translate epistemic states into graph tensors and evaluate them using GNN models through ONNX Runtime.

```text
Kripke state
     │
     ▼
Graph tensor
     │
     ▼
GNN
     │
     ▼
Heuristic estimate
```

The representation describes the epistemic state itself—including its graph structure and designated worlds—and is therefore not intrinsically tied to a particular EPDDL action fragment.

The effectiveness of a trained model nevertheless depends on its training distribution and should be evaluated on the target domains.

### mA* heuristics

> **Note:** Specialized heuristics designed for the mA* fragment should currently be used only for tasks known to satisfy the assumptions required by those heuristics.

Automatic recognition of compatible EPDDL fragments is planned. Until then, these heuristics should not be treated as general-purpose heuristics for arbitrary EPDDL tasks.

---

## Experiments

The current IPC benchmark source is:

```text
exp/ipc2026-benchmarks/
```

Experimental results should record the relevant planner configuration, including:

- benchmark revision;
- search strategy and heuristic;
- visited-state and bisimulation settings;
- comparison mode;
- neural model, when applicable;
- time and memory limits.

Experiment, testing, and dataset-generation utilities are available under:

```text
scripts/
```

---

## Known limitations

- Applicability of mA*-specific heuristics is not yet detected automatically.
- Learned models may require evaluation or retraining when moving to substantially different domain distributions.
- GPU ONNX inference currently targets Linux/NVIDIA environments.

---

## Future work

Current development directions include:

- **automatic EPDDL fragment detection**, allowing deep to enable compatible specialized heuristics automatically;
- **GNN/RL training and evaluation** on EPDDL and IPC benchmark distributions;
- improved generalization of learned heuristics across domains;
- further optimization of epistemic state generation, storage, and search.

---

## Repository structure

```text
deep/
├── src/                     planner implementation
│   ├── actions/
│   ├── bisimulation/
│   ├── domain/
│   ├── formulae/
│   ├── heuristics/
│   ├── parse/
│   ├── search/
│   ├── states/
│   └── utilities/
│
├── lib/
│   ├── plank/               EPDDL / DEL infrastructure
│   ├── CLI11/
│   ├── xxHash/
│   └── onnxruntime/         optional NN runtime
│
├── exp/
│   └── ipc2026-benchmarks/
│
├── scripts/
├── utils/
├── CMakeLists.txt
└── build.sh
```

---

## References

### deep / GNN-derived heuristics

```bibtex
@article{briglia2025scaling,
  title   = {Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics},
  author  = {Briglia, Giovanni and Fabiano, Francesco and Mariani, Stefano},
  journal = {arXiv preprint arXiv:2508.12840},
  year    = {2025}
}
```

### EPDDL

```bibtex
@article{burigana2026epddl,
  title   = {The Epistemic Planning Domain Definition Language: Official Guideline},
  author  = {Burigana, Alessandro and Fabiano, Francesco},
  journal = {arXiv preprint arXiv:2601.20969},
  year    = {2026}
}
```

### Related work

- A. Burigana and F. Fabiano. *The Epistemic Planning Domain Definition Language.* IPS 2022.
- F. Fabiano et al. *E-PDDL: A Standardized Way of Defining Epistemic Planning Problems.* 2021.
- F. Fabiano et al. *H-EFP: Bridging Efficiency in Multi-agent Epistemic Planning with Heuristics.* PRIMA 2024.
- F. Fabiano et al. *EFP 2.0: A Multi-Agent Epistemic Solver with Multiple E-State Representations.* ICAPS 2020.

---

## License

deep is distributed under the **GNU General Public License v3.0**.

See [`LICENSE`](LICENSE) for details.
