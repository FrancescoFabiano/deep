# Separated encoding — status

The `separated` encoding stores the state Kripke graph and the goal graph
**separately**: each state DOT is goal-free and the per-instance goal lives in
its own `goal_tree.dot`. This contrasts with `merged`, where the goal subgraph is
inlined into every state DOT. Both are first-class paths in `lib/rl_handler`
(offline RL fringe-ranking) and `lib/gnn_handler[_plus]` (distance estimator).

## Data contract
- Generation: `--no_goal` in `scripts/gnn_exp/create_all_training_data.py` →
  `--dataset_separated` on the C++ `deep` binary.
- CSV columns: `File Path, Depth, Distance From Goal, Goal, File Path
  Predecessor, Action`. The **`Goal` column is the path to `goal_tree.dot`** (in
  both modes), repo-root-relative after the script's path rewrite.
- Separated state DOTs live under `RawFiles/<TYPE>_separated/` and are goal-free
  (hashed node ids, no small-integer goal nodes). `goal_tree.dot` is a single
  per-instance `digraph G { … }`; every edge carries an integer `label=`, so it
  parses with the same `parse_dot_fast` / id-folding as state graphs.
- `merged` does **not** write `goal_tree.dot` (the `Goal` column still names one,
  but the file is absent). Goal loading is therefore driven by `kind_of_data`,
  never by the column's presence: separated ⇒ load+feed the separate goal;
  merged ⇒ goal is inlined, never feed it again (no double-count).

## rl_handler — 9-input ONNX contract
Separated export (`trainer.to_onnx`, `kind_of_data="separated"`) and the C++
`FringeEvalRL` separated branch agree on input order:

```
node_features, edge_index, edge_attr, membership,
goal_node_features, goal_edge_index, goal_edge_attr, goal_batch, mask
```

`goal_batch` maps each goal node to its fringe index, so `goal_emb[candidate_batch]`
aligns one pooled goal per candidate. At inference there is one goal graph per
call, so **`goal_batch` is all-zeros** (B=1) — matching `FringeEvalRL`, which
zero-fills `goal_state_batch`. Every fringe must contribute ≥1 goal node (see the
empty-goal guard); an empty goal would shift later fringes' `goal_emb` rows out
of step with `candidate_batch`.

## Deployment status
- **rl_handler = wired & runs at solve time (C1 confirmed).** A separated
  `frontier_policy_32.onnx` loads and runs under
  `deep <inst> -b --search RL --RL_model <onnx> --RL_fringe_size 32
  --dataset_separated` with no input-count mismatch (release binary, mtime
  2026-06-04). The goal tensor is built at **solve** time, not only under
  `--dataset`: the `GraphNN` constructor calls `populate_with_goal()` →
  `fill_graph_tensor(m_goal_graph_tensor)` when separated, and `m_goal_string` is
  set unconditionally by `TrainingDataset::generate_goal_tree_subgraph(false)`.
  `FringeEvalRL` pulls it via `get_goal_tensor()`. No empty-goal gap.
- **Unverified:** byte-level goal-node-ordering parity between the Python
  encoder's goal packing and the C++ `m_goal_graph_tensor` remains pending a C++
  fringe/goal tensor dump. Inference succeeds, but exact node-order equivalence
  is not yet asserted.
- **gnn_handler / gnn_handler_plus = training + export ready; deployment pending
  the C++ `run_inference` separated branch.** A separated distance-estimator ONNX
  exports the goal inputs (`goal_node_ids, goal_edge_index, goal_edge_attr,
  goal_batch`). Deployment requires the C++ `GraphNN::run_inference` separated
  branch to feed `get_goal_tensor()` into them, mirroring `FringeEvalRL` (today
  `run_inference` early-exits on `--dataset_separated`). The goal tensor is
  already built at solve time, so training/export here is the ready precursor.

## Hardening notes
- **Empty-goal guard:** `load_goal_graph` rejects a 0-node goal; the
  `GlobalFlatCache` / `pack_fringe_batch` all-or-none validation requires every
  goal graph to have ≥1 node.
- **`_read_csv` header-driven parse:** `gnn_handler` reads the generation table
  by header, so the 6-column row's `Goal` field is the clean `goal_tree.dot` path
  (the old fixed `split(",", 3)` folded `Predecessor,Action` into `Goal`).
