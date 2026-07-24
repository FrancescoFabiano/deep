# Unattended run report — best-checkpoint saving, full training, timeout-600 eval

Session date: 2026-06-04/05. Branch: `Reinforcement-Learning`.
All tasks completed; no unrecovered failures.

## Task 1 — Best-val checkpoint saving

**Finding: the pipeline was ALREADY saving the best-validation epoch, not the last.**

- `BaseModel.train` (lib/gnn_handler/src/model.py) writes `{model_name}.pt`
  every time `val_loss` improves, so the `.pt` on disk is the best epoch.
- `__main__.py` reloads that `.pt` (`m.load_model(...)`) *before* calling
  `to_onnx()`, so the ONNX export already used the best weights too.

What was missing and got added (commit `5536a5e`):
- `best_epoch` recorded and stored in `history_losses.json` (kept out of the
  per-epoch plot series).
- `{model_name}_last.pt` (last-epoch weights) saved for reference.
- 3-epoch sanity run confirmed: `best_epoch` == argmin(val_loss), `_last.pt`
  written, ONNX export succeeded. No interface change.

## Task 2 — Commits (local only, no push)

| Hash | Message |
|---|---|
| `220602a` | Fix clamp gradient trap; per-epoch Spearman; frozen-training diagnostics |
| `5536a5e` | Save best-val checkpoint; export ONNX from best epoch |

Committed: `lib/gnn_handler/**` (Python only), `scripts/gnn_exp/**`,
`.gitignore` (now ignores `exp/**/_models/`, `exp/**/_results*/`).
Excluded: `lib/CLI11` submodule pointer, staged `exp/.../CC/*.txt` problem
files (left staged, untouched), all model/data artifacts.

## Task 3 — Full 200-epoch training (clamp fix active)

Run: 200 epochs, batch 1024, AdamW lr=1e-3, `--build-data false`
(reusing the validated 200,325/40,057 sample split). Exit code 0.

| | epoch | val R² | Spearman | MAE | RMSE |
|---|---|---|---|---|---|
| **Best (val_loss)** | **193** | **0.9184** | 0.8780 | 0.00150 | 0.00441 |
| Last | 199 | 0.9135 | 0.8722 | 0.00160 | — |

Trajectory (mean per 50-epoch window):

| epochs | mean R² | mean Spearman |
|---|---|---|
| 0–50 | 0.410 | 0.623 |
| 50–100 | 0.779 | 0.819 |
| 100–150 | 0.881 | 0.869 |
| 150–200 | 0.903 | 0.880 |

- All 200 val_loss values unique → no refreeze; clamp-trap fix holds at scale.
- Prediction spread on full val set (best ckpt, via
  `diagnose_frozen_training.py`): std = 0.0150 > 0, mean 0.0104 ≈ target
  mean, max 0.157 ≈ target max. 44.7% of predictions at the inference clamp
  floor — legitimate, as 64.4% of targets ARE 0.001 (depth 0).
- **More training warranted?** Mildly: the top-5 val_loss epochs are
  {193, 197, 157, 190, 196} — still inching up at 200, but the last
  50-epoch window gains only ~0.02 mean R² over the previous one. An LR
  schedule (cosine/warmup) is likely worth more than raw extra epochs.
- vs the broken run: R² −0.3721 (constant output) → **0.9184**; Spearman
  undefined → **0.878**.

## Task 4 — Bulk evaluation, timeout 600 (paper-comparable)

Old timeout-60 results from the broken model preserved verbatim at
`exp/gnn_exp/batch0/_results_t60_broken_model/`. New runs in
`exp/gnn_exp/batch0/_results/{Astar_GNN,BFS}/`. Both: 4 threads, `-c -b`,
GNN adds `-s Astar -u GNN`.

### Coverage summary

| Config | Coverage | Avg length | Avg nodes | Avg total (ms) |
|---|---|---|---|---|
| **A\*(GNN) t600, fixed model** | **25/26 (96%)** | 6.20 | 845 | 3345 |
| BFS t600 | **26/26 (100%)** | 5.12 (optimal) | 1459 | 6310 |
| A\*(GNN) t60, broken model | 23/26 (88%) | 5.04 (optimal) | — | — |
| BFS t60 | 25/26 (96%) | 5.04 (optimal) | 1156 | 3505 |

GNN's only timeout: `CC_2_3_4__pl_7` (BFS needs 9031 expansions / 81 s on it;
at the GNN's ~2.5–7 ms/node ONNX cost that instance still doesn't fit 600 s
unless node count collapses, and the heuristic didn't collapse it).

### Per-instance results, timeout 600 (Goal / Length / Nodes / Total ms)

| Instance | GNN Goal | GNN Len | GNN Nodes | GNN ms | BFS Len | BFS Nodes | BFS ms | Node ratio BFS/GNN |
|---|---|---|---|---|---|---|---|---|
| CC_2_2_3__pl_3 | Yes | 4 | 8 | 249 | 3 | 9 | 9 | 1.1 |
| CC_2_2_3__pl_4 | Yes | 6 | 11 | 93 | 4 | 17 | 16 | 1.5 |
| CC_2_2_3__pl_5 | Yes | 5 | 8 | 76 | 5 | 78 | 79 | 9.8 |
| CC_2_2_3__pl_6 | Yes | 6 | 11 | 54 | 6 | 283 | 240 | 25.7 |
| CC_2_2_3__pl_7 | Yes | 9 | 477 | 3519 | 7 | 1032 | 887 | 2.2 |
| CC_2_2_3__pl_8 | Yes | 8 | 1419 | 9482 | 8 | 1957 | 1689 | 1.4 |
| CC_2_2_4__pl_3 | Yes | 3 | 11 | 608 | 3 | 6 | 20 | 0.5 |
| CC_2_2_4__pl_4 | Yes | 5 | 9 | 329 | 4 | 24 | 77 | 2.7 |
| CC_2_2_4__pl_5 | Yes | 8 | 25 | 192 | 5 | 171 | 651 | 6.8 |
| CC_2_2_4__pl_6 | Yes | 6 | 13 | 643 | 6 | 824 | 1692 | 63.4 |
| CC_2_2_4__pl_7 | Yes | 8 | 74 | 1929 | 7 | 4163 | 5995 | 56.3 |
| CC_2_3_4__pl_3 | Yes | 3 | 14 | 929 | 3 | 6 | 187 | 0.4 |
| CC_2_3_4__pl_4 | Yes | 4 | 60 | 5941 | 4 | 29 | 869 | 0.5 |
| CC_2_3_4__pl_5 | Yes | 8 | 9 | 788 | 5 | 277 | 7639 | 30.8 |
| CC_2_3_4__pl_6 | Yes | 8 | 22 | 1245 | 6 | 1611 | 20603 | 73.2 |
| CC_2_3_4__pl_7 | **TO** | – | – | – | 7 | 9031 | 80984 | – |
| CC_3_2_3__pl_3 | Yes | 5 | 21 | 204 | 3 | 8 | 11 | 0.4 |
| CC_3_2_3__pl_4 | Yes | 6 | 13 | 76 | 4 | 30 | 46 | 2.3 |
| CC_3_2_3__pl_5 | Yes | 7 | 16 | 129 | 5 | 177 | 239 | 11.1 |
| CC_3_2_3__pl_6 | Yes | 6 | 6 | 190 | 6 | 468 | 613 | 78.0 |
| CC_3_2_3__pl_7 | Yes | 11 | 18154 | 44751 | 7 | 3361 | 5630 | **0.19** |
| CC_3_3_3__pl_3 | Yes | 4 | 5 | 129 | 3 | 8 | 42 | 1.6 |
| CC_3_3_3__pl_4 | Yes | 4 | 9 | 338 | 4 | 39 | 103 | 4.3 |
| CC_3_3_3__pl_5 | Yes | 6 | 240 | 2643 | 5 | 538 | 1269 | 2.2 |
| CC_3_3_3__pl_6 | Yes | 6 | 65 | 1643 | 6 | 1643 | 3938 | 25.3 |
| CC_3_3_3__pl_7 | Yes | 9 | 428 | 7435 | 7 | 12137 | 30544 | 28.4 |

### Node-expansion analysis (25 instances both solved)

- Total: GNN 21,128 vs BFS 28,896 expansions (1.37×). The aggregate is
  dominated by one outlier: `CC_3_2_3__pl_7`, where the heuristic misleads
  (GNN 18,154 nodes for a length-11 plan vs BFS 3,361 for length 7).
- **Excluding that outlier: GNN 2,974 vs BFS 25,535 — 8.6× fewer expansions
  overall; median per-instance ratio ≈ 2.7×, max 78× (`CC_3_2_3__pl_6`).**
- GNN expands fewer nodes than BFS on 19/25 instances; ratios grow with
  depth within each family — exactly what a useful distance estimate should do.

### Plan quality (the trade-off)

A* with this learned heuristic is **not admissible**: 14/25 plans are
suboptimal (avg +1.16 steps over optimal; worst +4 on `CC_3_2_3__pl_7`).
The broken-model run had all-optimal plans only because a constant heuristic
degenerates to uniform-cost search. Anyone comparing against the appendix
should report lengths, not just coverage.

### vs the earlier timeout-60 broken-model run

- **Coverage**: 23/26 → **25/26** (newly solved: `CC_2_3_4__pl_6` 1.2 s,
  `CC_3_3_3__pl_7` 7.4 s — both former timeouts; `CC_2_3_4__pl_7` remains TO).
- **Nodes** (broken → fixed): `CC_2_2_4__pl_7` 1844→74 (25× fewer),
  `CC_3_3_3__pl_6` 1949→65 (30×), `CC_2_3_4__pl_5` 127→9 (14×),
  `CC_2_2_3__pl_8` 1581→1419 (~par), `CC_3_2_3__pl_7` 1762→18154
  (10× worse, the misleading-heuristic outlier).
- **Runtime**: avg total 3345 ms vs broken-run avgs in the tens of seconds on
  deep instances (e.g. `CC_2_3_4__pl_5` 54.8 s→0.8 s) — fewer expansions
  directly buy down the ~2.5–7 ms/node ONNX cost.
- The broken run's "optimal plans, fewer nodes than BFS" pattern was indeed
  the uniform-cost + tie-breaking artifact suspected earlier; the fixed model
  shows the real (and much larger) node reductions, at the cost of optimality.

## Failures / notes

- No task failures. One benign `UserWarning` from the dynamo ONNX exporter
  about `dynamic_axes` (same as previous successful exports).
- `lib/CLI11` shows as modified (submodule pointer) — left uncommitted on
  purpose; not part of this work.
- Artifacts: model + ONNX in `exp/gnn_exp/batch0/_models/CC/`
  (`distance_estimator.pt` = best epoch 193, `_last.pt` = epoch 199,
  `history_losses.json` has per-epoch R²/Spearman + `best_epoch`).
- Open question for next session: the `CC_3_2_3__pl_7` outlier and the 14/25
  suboptimal plans suggest trying weighted-A*-style calibration, an
  admissibility-leaning loss (penalize overestimation more), or the
  percentile-based target scaling proposed in the previous session.
