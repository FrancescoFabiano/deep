# Python-side follow-ups for the C++ ONNX / dataset changes

This file tracks what the Python code (`lib/gnn_handler`, `lib/rl_handler`,
`lib/deep_nn` in the fork, `scripts/`) must change to match the C++ planner on
`deep-2.0`. The C++ is the reference; nothing here has been changed on the
Python side yet.

Issue numbers refer to `Giovannibriglia/deep_forked`.

## 1. GNN models have no `pointed_ids` input (blocking)

`GraphNN::run_inference` feeds `pointed_ids` since commit 405d76f, but the
`lib/gnn_handler` contract (`ONNX_INPUTS` in `lib/gnn_handler/src/utils.py`)
still declares 4 state inputs. Since fix #4 the planner stops with

```
[ERROR] ONNX input count mismatch: model expects 4 input tensors but C++ prepared 5 (check that the model was exported for this dataset type/mode and takes pointed_ids; see docs/python_followups.md).
Error code: 899
```

(RL uses code 905 for the same check. Exit codes above 255 reach the shell
modulo 256: 899 shows up as 131, 905 as 137.)

for every shipped model (`lib/gnn_handler/models/distance_estimator.onnx`,
`exp/mAstar/gnn_exp/*/_models/*`). They must be re-exported with `pointed_ids`.

`pointed_ids` holds the **row indices** (in the node tensor) of the designated
worlds, not their hashed ids; the same holds for HASHED, MAPPED and BITMASK.

## 2. GNN input order expected by the planner

Inputs are paired **positionally** with the model's declared inputs.

| # | name (suggested)  | dtype | shape                              | notes |
|---|-------------------|-------|------------------------------------|-------|
| 0 | `state_node_ids`  | int64 | `[N]`                              | HASHED / MAPPED |
| 0 | `state_node_bits` | uint8 | `[N, B]`                           | BITMASK, see §3 |
| 1 | `state_edge_index`| int64 | `[2, E]`                           | |
| 2 | `state_edge_attr` | int64 | `[E, 1]`                           | |
| 3 | `state_batch`     | int64 | `[N]`                              | all zeros |
| 4 | `pointed_ids`     | int64 | `[K]`                              | row indices into the state nodes |
| 5 | `goal_node_ids`   | int64 | `[M]`                              | separated only |
| 6 | `goal_edge_index` | int64 | `[2, F]`                           | separated only |
| 7 | `goal_edge_attr`  | int64 | `[F, 1]`                           | separated only |
| 8 | `goal_batch`      | int64 | `[M]`                              | separated only, all zeros |

Merged: 5 inputs (the goal is part of the state graph through the epsilon
node). Separated: 9 inputs. Separated GNN inference was enabled by fix #3.

## 3. BITMASK width depends on the mode

The per-node bitmask width is not always 42 (`BITMASK_DIM`):

| mode      | state node bits                                        | goal nodes |
|-----------|--------------------------------------------------------|------------|
| merged    | 14 repetition + 18 fluents + 10 goal = **42**          | uint8 bits, 42 wide, inside the state graph |
| separated | 14 repetition + 18 fluents = **32**                    | **int64 decimal ids** (`goal_node_ids`), as written in the separated goal DOT |

(`MAX_REPETITION_BITS`, `MAX_FLUENT_NUMBER`, `GOAL_ENCODING_BITS` in
`src/utilities/Define.h`.)

`lib/gnn_handler` hard-codes `BITMASK_DIM = 42` and names the separated goal
input `goal_node_bits`; separated BITMASK models must instead take
`state_node_bits` `[N, 32]` and int64 `goal_node_ids`.

## 4. RL (FringeEvalRL) contract differences

`FringeEvalRL::get_score` feeds, in order: `node_ids [N]`, `edge_index [2,E]`,
`edge_attr [E]`, `membership [N]`, `pointed_ids [K]`, then (separated only)
`goal_node_ids [M]`, `goal_edge_index [2,F]`, `goal_edge_attr [F]`,
`goal_batch [M]`, and finally the active-state mask `uint8 [fringe_size]`.

Note that `edge_attr` / `goal_edge_attr` are **1-D** here and `[E, 1]` in the
GNN path.

All three dataset types are accepted (issue #1). The first input depends on
the type; everything else is unchanged:

| dataset          | `node_ids` / `node_bits`   | goal nodes (separated) |
|------------------|----------------------------|------------------------|
| HASHED / MAPPED  | int64 `[N]`                | int64 `goal_node_ids`  |
| BITMASK merged   | uint8 `[N, 42]`            | (none, merged)         |
| BITMASK separated| uint8 `[N, 32]`            | int64 `goal_node_ids`  |

The states of the fringe are concatenated in order: rows of state `i` come
after those of state `i-1`, and `membership` gives each row's state index.
The Python side needs a bitmask variant of `pack_fringe` producing this
layout.

## 5. Other pending items

- **#6 (done in C++, no Python change needed):** goals with group knowledge
  no longer abort dataset generation. The goal DOT writes `E_G phi` as the
  conjunction of `B_a phi` for `a` in `G` (an AND-style node with one belief
  subtree per agent), `TRUE` as a leaf and `FALSE` as `NOT(TRUE)`. Goal DOTs of
  domains without these formulas are unchanged; gossip-like domains now
  produce data.

- **#5 (done in C++):** non-finite fringe scores are no longer fatal; the
  planner prints a `[WARNING]` and ranks them last. Ties are broken by fringe
  index. The export-time non-finite check proposed in the issue is still
  useful.
- **#2 (done in C++):** separated state DOTs now contain one line per
  designated world, before the edges, with the same id spelling as the edge
  lines (decimal, mapped id or bitmask string):

  ```
    <world_id> [shape=doublecircle];
  ```

  Every designated world appears, even one with no belief edge. The order
  matches `pointed_ids`. Merged DOTs are unchanged (still `epsilon -> w`
  edges). The separated DOT parser should build `pointed_ids` from these lines
  instead of deriving them from the merged twin.
- **#7 (done in C++):** dataset generation now exits with code **4**
  (`DatasetNoGoalFound`) when no goal is reached; the CSV is kept and always
  has its depth-0 root row (DFS / S_DFS used to drop it). Code 2 is success,
  3 means goals were found but fewer rows than the minimum.
  `scripts/pipeline/data.py` should check `returncode == 4` instead of matching
  `No goals found`.
