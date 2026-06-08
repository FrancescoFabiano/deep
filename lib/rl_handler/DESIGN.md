# DESIGN — Offline RL fringe-ranking trainer (`offline-RL-test` branch)

Goal: replace the pointwise softmax-over-frontier objective in `lib/rl_handler`
with a real Q-learning trainer over the search-fringe MDP, simulated inside
trees reconstructed from the generation tables. The trained model must be a
drop-in replacement for the existing `frontier_policy_<F>.onnx` consumed by the
C++ planner. **No C++ changes.**

## 1. CONTRACT (derived read-only from the C++ consumer)

Consumer chain: `SpaceSearcher.tpp` → `RL_BestFirst.h` → `FringeEvalRL.{h,tpp}`
(ONNX Runtime). Verified empirically: `cmake-build-release-nn/bin/deep` solves
`CC_2_2_3__pl_3` with `--search RL --RL_model frontier_policy_32.onnx`.

**ONNX model I/O** (merged mode = what batch0_merged uses; `HASHED` dataset):

| input | dtype | shape | C++ source (`fringe_to_tensor_minimal`) |
|---|---|---|---|
| `node_features` | int64 | `[N]` | concat of each state's `real_node_ids` (signed hashed IDs), fringe order |
| `edge_index` | int64 | `[2,E]` | per-state `edge_src`/`edge_dst` shifted by cumulative node offset; row0=src, row1=dst |
| `edge_attr` | int64 | `[E]` | raw integer edge labels, aligned with edges |
| `membership` | int64 | `[N]` | state index (0..K−1) repeated per node, fringe order |
| `mask` | uint8 | `[F]` | `active_states`: 1 for slots `0..K−1`, 0 padding up to `F = --RL_fringe_size` (**last input**) |

Output: `logits` float32 `[F]` (symbolic dim in the export; accepted by the
deployed binary). Separated mode inserts `goal_node_features/goal_edge_index/
goal_edge_attr/goal_batch` between `membership` and `mask`.

**Interpretation:** C++ `rankScores` sorts scores descending and assigns rank i
as the heuristic value (lower = expanded first) ⇒ **higher logit = expand
sooner**; effective action = argmax over the first K logits. No constants
sidecar exists for this model family — the only sacred pairing is logits length
`F` ↔ `--RL_fringe_size` (checked at load) and file naming
`_models/<domain>/frontier_policy_<F>.onnx` (assumed by
`scripts/rl_exp/bulk_coverage_run.py`).

**In-model encoding** (already inside the exported graph, must stay): HASHED
signed-aware normalization `id/2^63 → [-1,1]` + node-label embedding
`id mod num_node_labels (=4096)` + edge buckets `abs(id) mod K (=128)`;
inactive slots masked to −1e9.

**Export path:** reuse `RLFrontierTrainer.to_onnx` *verbatim* (opset 18,
tracing, same input names/order/dynamic axes). This is the strongest possible
contract guarantee — same code that produced the working production exports.

**Planner loop facts** (defaults `F=32`, exploitation 70%, exploration 10%):
successors accumulate until ≥ `int(F*0.7)=22` (or queue empty), then
`push_vector`: remaining queue → reservoir, new beam = new successors first
(overflow → reservoir) + reservoir refill (heuristic-best or random + 3 random
exploration slots), beam re-scored in one ONNX call. Goal test happens at
*successor generation* (`successor.is_goal()` returns before pushing).

## 2. MDP — fringe-ranking formulation (offline, reconstructed tree)

Environment = the search tree reconstructed from a generation table
(`File Path, Depth, Distance From Goal, Goal, File Path Predecessor, Action`;
rows are edges `pred --Action--> state`). Reconstruction conventions (in
`src/offline/tree_env.py::load_tree_instance`): on duplicate `File Path`
rows the min-depth row wins; the root is the unique Depth-0 state (its CSV
predecessor is a dummy `init.dot`); edges whose predecessor never appears as
a state (<1% generator anomalies) are dropped as unreachable orphans.

- **State:** current fringe (≤ F tree nodes) + reservoir + visited set.
- **Action:** index k of one *active* fringe slot → expand that node.
- **Transition (mirrors `RL_BestFirst` with `RefillMode::RANDOM`):** expanded
  node's unvisited children become the new beam (first F; overflow →
  reservoir); leftover beam states → reservoir; free slots refilled uniformly
  at random from the reservoir (seeded RNG).
- **Reward:** `0` and terminal if any generated child has `d*=0` (goal found at
  generation, exactly like the C++ loop); `−1` otherwise.
- **Terminal:** goal generated, or fringe+reservoir exhausted (dead end), or
  safety cap (#expansions ≤ #states). Return = −(#expansions before the
  goal-generating one) ⇒ **node economy is the objective itself.**
- **Episodes:** start at each instance's root; one tree per episode; train
  instances cycled round-robin.
- **Divergence noted:** the deployed planner re-scores every ~22 expansions
  with stale ranks in between; the env re-scores every expansion. Greedy
  selection from a fixed scorer is rank-consistent, so the gap is benign; the
  Phase-5 planner smoke run measures it directly.
- `d*` (table distance; unreachable = 1e6) is **evaluation oracle only**, never
  a training input.

## 3. Score parameterization & algorithm

Q(fringe, k) = `logits[k]` of `FrontierPolicyNetwork` — shared GNN encoder over
all fringe states packed in one graph + per-state head (+ frontier mean-context,
as in production). One forward pass scores the whole variable-size fringe; this
is exactly the shape the C++ packing implies. An accurate −d*(s) is one optimal
scorer, but Q-learning may learn fringe-selection behavior beyond it — that is
the point of optimizing long-term return directly.

**Network reuse decision:** the brief suggested importing `lib/gnn_handler`'s
encoder; we instead reuse `lib/rl_handler`'s own `FrontierPolicyNetwork`
because (a) it *defines* the deployed contract (in-graph HASHED normalization —
itself a port of gnn_handler's I64 handling, identical constants), and (b) its
`to_onnx` is the proven export path. Same hyperparameters as production exports
(gine ×3, hidden 128, mean pool, K=128, 4096 node labels, global context).

**Algorithm: Double DQN over the fringe MDP.**
- Behavior policy: ε-greedy over Q (ε linear 1.0 → 0.05 over the first 50% of
  frames, then flat; `--epsilon-schedule start,end,frac`); start-state = root.
- Replay buffer (compact: instance id + fringe node-ids + action + reward +
  next-fringe node-ids + done flag; tensors re-assembled from the per-instance
  cache, never stored raw). Uniform sampling, capacity 5e4.
- Targets: `y = r + γ·(1−done)·Q_target(s', argmax_k Q_online(s',k))`; Huber
  loss; target net hard-synced every 1000 frames; Adam 1e-4 (smaller and
  steadier than the 3e-4 supervised default — bootstrapped targets);
  grad-clip 1.0.
- **γ = 0.99** (flag `--gamma`). With −1/step rewards Q ∈ [−100, 0]; relative
  ranking pressure is what matters, absolute scale is bounded by construction.

## 4. Dead ends / collapse safety (gnn_exp failure catalogue)

Fringe exhaustion is terminal **with no bonus and no special target**: the TD
target is just `r = −1` with `done=1`. No 1e6-style constants enter training
anywhere (the table's `1e6` marker is used only by the eval oracle). This
avoids (i) the bias-attractor (no huge-magnitude regression target to collapse
onto), (ii) saturation (Q bounded in [−1/(1−γ), 0]), and (iii) the
constant-output degenerate optimum (a constant Q is *not* a fixed point of the
Bellman update when returns differ across states). Checkpoint logging watches
the known signatures: score mean/min/max, dead-end-vs-reachable score split,
val Spearman → if a seed collapses, stop, diagnose against
`REPORT_plus_investigations.md`, record — no silent seed rerolling.

## 5. Training & evaluation protocol

Frames = env steps; `--frames 1e5`, `--n-checkpoints 20`, `--seed`, `--gamma`,
`--epsilon-schedule`. Data: CC tables under `out/NN/Training` (the batch0/CC
source data). Degenerate tables `CC_2_3_4__pl_7{,_1}` (pure chains, no goal
rows) are excluded. Instance-level split: train = {CC_2_2_4__pl_7,
CC_3_2_3__pl_6}, val = {CC_3_2_3__pl_7}. At every checkpoint, on val:
1. greedy-rollout #expansions per instance (vs BFS count and plan length),
2. Spearman(score, −d*) over val states (single-state fringes; unreachable d*
   reported both excluded and as max+1),
3. score statistics incl. dead-end vs reachable split.
Best checkpoint by val greedy #expansions (also keep best-by-Spearman). Export
best via `to_onnx` at F ∈ {32} (smoke), consumed unchanged by
`bulk_coverage_run.py`. Production: 1e5 frames × 3 seeds (nohup), IQM ± IQR-std
bands over checkpoints for both metrics.

New code lives in `lib/rl_handler/src/offline/` (encoder cache, tree env,
replay, DQN trainer) + entry point `lib/rl_handler/offline_main.py`. The
existing supervised pipeline is untouched.
