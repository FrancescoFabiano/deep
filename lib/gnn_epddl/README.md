## About the Package

`gnn_epddl` trains the learned sibling ranker that deep uses as its GNN
heuristic (`-u GNN --ranker_model <file|folder>`). A state (a Kripke
structure plus the goal) is encoded as a typed graph; a 5-layer GINE network
scores it, higher = closer to the goal; under HFS the planner expands the
best-scored state first. The network is trained with a pairwise ranking loss
on states of deep's own search trees, and the checkpoint of a run is chosen by
greedy best-first search on held-out validation problems.

Training and validation use generated instances only; the benchmark problems
are test problems and are never trained or validated on. The instances, the
split and the replication scripts are in `exp/gnn_epddl/`.

## 1. Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=lib/gnn_epddl/src
```

deep must be built with ONNX support (`build.sh nn`); every deep run of this
package passes `--ranker_encoding`, and validation searches use
`--expand_server`. The deep binary is always a parameter (`--deep`, default
`cmake-build-release-nn/bin/deep`).

## 2. Usage

One command per step, `python -m gnn_epddl <command> --help` for every option.
`exp/gnn_epddl/scripts/` runs them in order with the settings of the
experiments (steps 1-6), and step 7 runs the planner with the exported models
on the benchmark problems through the competition runner.

| Step | Command | Output |
|---|---|---|
| generate | `generate --out exp/gnn_epddl/instances` | problem files `<tier>/<family>/problems/*.epddl` |
| check | `check --instances ... --tier T --family F --plank P --benchmarks B` | CSV: parse, duplicate of a benchmark problem, BFS plan length and nodes |
| data | `data --instances ... --split split.csv --data D` | per training instance `D/<family>/<short tier>-<problem>/`: deep's tree (`out/`), `fluent_names.csv`, `goal_ops.csv`, `new3.pt` |
| build | `build D/<family>/<dir> ...` | `new3.pt` from an existing tree |
| train | `train --family F \| --all \| --families a,b --name N \| --subfamily F S --split ... --data D --instances ... --ckpt C --log L` | checkpoints `C/<run>/s<seed>-e<epoch>.pt`, one log row per evaluation |
| select | `select --log L --out selection.csv` | the chosen checkpoint of every run |
| export | `export --checkpoint ck --out m.onnx --kind standard\|fast --sample-data D/search-and-rescue` | `m.onnx` and its sidecar `m.onnx.vocab` |
| search | `search --checkpoint ck --instances ... --family F --tier T --problem P` | one greedy best-first search (nodes, plan length) |

### Instances and split

`generators/` holds one module per domain; each writes its problems
deterministically (fixed seeds) from templates of the benchmark problems
(same domain files). `check` verifies that plank parses a problem, that it is
not a benchmark problem (comments and whitespace ignored) and that deep's BFS
solves it. `split.csv` lists, per family, the training instances (label
`train`) and the validation instances (label `validation`); the column
`reason` is the sub-family.

### Training data (`dataset.py`)

deep's dataset mode (`--dataset`, depth 25, HASHED states, at most 60000
written / 100000 visited states, `-b -c`) on each training instance: BFS
first, else DFS with seeds 42, 43, 44 (a run is killed above 12 GiB or after
600 s). From the tree: the distance of every state to the goal, sibling
groups (children of one state with different distances, at most 3000) and
global pairs (two states with different distances, at most 2000), sampled with
seed 0, saved as `new3.pt`.

### Encoding (`encoding.py`)

Nodes: goal tree, worlds (designated worlds have their own type), agents,
fluents and objects; edges: beliefs by agent rank, goal links, world-to-fluent
`holds` edges and fluent-to-argument edges, every edge also reversed with its
own type. Predicate names are hidden (every predicate is one `fluent` type).
The vocabulary (`vocab.json`) covers the seven benchmark domains; for a new
domain pass `--vocab <writable copy>` to every command.

### Model and training (`model.py`, `train.py`)

Node and edge type embeddings, 5 residual GINE layers (width 96) each followed
by LayerNorm, readout [sum / 50, mean, mean over designated worlds], MLP.
Loss: binary cross-entropy on score(a) - score(b) over sibling pairs and
global pairs, Adam 1e-3, batches of 64, seed 0 (or `--seed`). Every
`--eval-every` epochs (10; and at the last epoch, 50) the checkpoint is saved
and every validation problem of the run's families is searched with it (2000
expansions / 120 s, gossip 3000 / 300 s).

Runs: `--family F` (per-domain model, run `pf-F`), `--all` (every family,
`final-all`), `--families a,b,...` (a subset, `final-sub` or `--name`),
`--subfamily F S` (one sub-family of F, `sf-F-S`).

### Selection (`selection.py`)

Per run, over all its seeds: most validation problems solved, then the
smallest sum of log10(expanded nodes) with an unsolved problem counted as
2 x cap, then the later epoch.

### Export (`export.py`)

Two ONNX forms of the same weights (opset 17; inputs `x`, `edge_index`,
`edge_attr`, `pmask`; output `score`):

* `standard`: the per-node message sums as scatter_add (for the GPU,
  `--ranker_model_gpu`);
* `fast`: edges sorted by target once per state and the sums taken as
  differences of a float64 running sum (for the CPU, `--ranker_model`;
  ONNX Runtime's ScatterElements dominated the CPU time).

Each export is checked against the PyTorch Geometric network and against ONNX
Runtime, and written with its sidecar `<model>.onnx.vocab` (node-type
vocabulary, `#anon`, `#drop_holds`, `#scale`, `#ops`) that deep reads.

### deep interface

* `--ranker_encoding`: the dataset DOT files get an edge (label 4) from every
  world to each fluent true in it; deep writes `fluent_names.csv` (fluent id,
  grounded name) and `goal_ops.csv` (goal-tree node id, operator) in its
  working folder.
* `--expand_server <folder>`: deep writes the initial state as
  `<folder>/0.dot` and prints `@@ready 0 <is_goal>`; each stdin line
  `expand <id>` writes every new successor as `<folder>/<id>.dot` and prints
  `@@ <id> <action> <is_goal> <is_new>` per successor, then `@@end`.
* `--ranker_model <file|folder>`: deploys an export; a folder holds
  `per-domain/<domain name>.onnx` and `general.onnx`. With `-s HFS -u GNN`
  the ranker guides greedy best-first search; with `-s RL -u RL_H` it ranks
  the RL beam (relative ranks within the beam).
* `--ranker_model_gpu <file|folder>` and `--ranker_gpu_edges N` (CUDA build):
  states with at least N edges (default 10000) are scored with the standard
  export on the GPU, smaller ones with `--ranker_model` on the CPU.

## 3. Tests

```
cd lib/gnn_epddl && python -m pytest tests -q
```

`tests/test_generators.py` regenerates every problem and compares it byte for
byte with `exp/gnn_epddl/instances` (no torch needed).
