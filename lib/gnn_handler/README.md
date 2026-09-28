## About the Project

A GNN distance estimator for dynamic epistemic search: every state (a Kripke
structure) of a planner **fringe** gets an estimated distance to the goal, and
the planner expands the closest one.  It is deployed through the same C++
consumer as the RL fringe ranker (`FringeEvalRL`), so the two models are
interchangeable at the planner: only the network and its training objective
differ.  Everything between the generator's tables and the ONNX file --
DOT parsing, beam construction, packing, the ONNX contract -- lives in
`lib/deep_nn` and is shared with `lib/rl_handler`.

## 1. Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Usage

### Training data

Generation tables produced by `deep --dataset` (merged mode, `HASHED`), in the
batch layout the RL handler uses:

```
<batch>/_models/<domain>/training_data/<instance>/<instance>_<STRAT>_depth_<D>.csv
<batch>/_models/<domain>/test_data/<instance>/...
```

The train/test split *is* these folders: `training_data` trains,
`test_data` is evaluated only.  Separated models are trained from the same
merged DOTs: the separated state and goal tensors are derived from them
(`deep_nn.dot.separated_view`), because a separated-generated DOT records no
designated worlds (GitHub issue #2).

### Train

```
python lib/gnn_handler/__main__.py \
    --train-csv <tables...> --test-csv <tables...> \
    --dir-save-model <batch>/_models/<domain>/seed42 \
    --kind-of-data merged|separated --fringe-sizes 4 8 16 32 --epochs 200
```

or, for every domain of a batch, `python scripts/gnn_exp/train_models.py <batch>`.

One model is trained per fringe size `F` on the planner-shaped beams of the
behaviour rollouts (`src/beams.py`); the loss is MSE on the scaled distance
`distance * slope + intercept` of every labelled slot (a state the generator
marked unreachable is packed but not a target).  The best checkpoint on the
test beams (or on the train beams when there is no test data) is exported as
`<dir>_fringe<F>/distance_estimator_<F>.onnx` and checked against
onnxruntime on real beams.

### Deploy

```
deep <domain> <problem> --act_lib <lib> -b -c --search RL --heuristics RL_H \
     --RL_model distance_estimator_<F>.onnx --RL_fringe_size <F> [--dataset_separated]
```

The ONNX follows the `FringeEvalRL` contract (`lib/deep_nn/contract.py`):
inputs `node_features, edge_index, edge_attr, membership, pointed_ids,
[goal_node_features, goal_edge_index, goal_edge_attr, goal_batch], mask`,
output `scores[F] = -distance` (the planner sorts descending).  No constant
file is needed: the scaling is inverted inside the graph.

The planner accepts HASHED, MAPPED and BITMASK fringes (issue #1 fixed upstream,
commit a9c7e66); the Python exporter still produces HASHED models only, the
BITMASK layout is specified in `docs/python_followups.md`.

For HFS/A* (`--heuristics GNN`) the same model is exported per state as
`distance_estimator_<F>_state.onnx` plus `_state_C.txt` (slope/intercept);
inputs `node_features, edge_index, edge_attr [E,1], batch, pointed_ids`.

## 3. Tests

```
cd lib/gnn_handler && ../../.venv/bin/python -m pytest tests -q
```
