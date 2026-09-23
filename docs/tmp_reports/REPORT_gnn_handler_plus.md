# Report — gnn_handler frozen as baseline; gnn_handler_plus built and measured

Session date: 2026-06-05. Branch: `Reinforcement-Learning`. All evals: batch0/CC, timeout 600, threads 4.

## Commits (local only)

| Hash | Message |
|---|---|
| `d73fd06` | Revert "Data-driven target scaling: MAX_DEPTH = ceil(1.1 x max seen)" — baseline is now published method + correctness/infrastructure fixes only; verified `MAX_DEPTH = 50` restored and a 3-epoch baseline smoke run passes |
| `e9d90a6` | Add gnn_handler_plus: dynamic MAX_DEPTH, unreachable-as-max, Spearman ckpt, first-class heuristic weight (flag-controlled) |

`lib/gnn_handler_plus` is a thin override package: `src/__init__.py` extends its module path into `lib/gnn_handler/src`, so only `sample_prep.py` / `training.py` / `export.py` (+ entry point and README) live in plus; parsing, model, ONNX and the data pipeline are imported from the baseline, never copied. The planner interface (ONNX inputs, C-file format) is identical; both variants are drop-in interchangeable.

## Unreachable-state counts (resolving the open question)

Raw training CSVs: **726 / 332,590 rows (0.2%)** — per instance: pl_4-family ≤0.1%, `CC_2_3_4__pl_7` highest at 6.5% (189/2,897). After the build pipeline's balancing/split, the plus-built `samples.pt` holds **603/201,099 train (0.30%)** and **123/40,210 test (0.31%)**. Far below the 25% cap — `--unreachable-cap` never triggered; the majority-class worry is moot at this scale.

Important build detail discovered: the baseline drops unreachable rows **at build time** (`GraphDataPipeline(remove_unreachable_goal_states=True)`), so a baseline `samples.pt` contains none. Plus builds with the filter off (one rebuild per dataset dir; a loud error explains this if `--include-unreachable` meets a baseline-built file).

### Did the model learn "unreachable = far"? Partially.

Predicted distance on known-unreachable states (target = 26):

| | baseline ep193 | plus @15 ep | plus @200 ep |
|---|---|---|---|
| in-distribution (val) | 0.59 | 1.75 | **5.50 mean / 1.69 median** |
| OOD (pl_7 post-mortem states) | 0.59 | 0.82 | 2.28 (vs 0.81 for distance-0 states) |

Direction is right — dead ends now rank above goal-adjacent states even OOD — but far from the target: 603 states (0.3%) under plain MSE are not enough gradient mass; even *train* unreachables underfit (mean 4.07). This also explains the low val R² (0.34): the 123 val unreachables contribute huge squared errors while barely moving Spearman (0.8634). **Loss upweighting for unreachables is the clear next step (added to README future work).**

## Training (plus, 200 epochs, spearman checkpoint)

best_epoch(spearman) = 132: Spearman **0.8634**, R² 0.34 (depressed by unreachables, see above), MAE 0.00925. best_epoch(val_loss) = 168: Spearman 0.8542. Both checkpoints + ONNX exported; `max_depth=26`, `raw_slope=0.038385`, `W` recorded in C-file and history. Note: plus val metrics include unreachable states, so they are not directly comparable to baseline/C-model numbers.

## Full comparison — 7 configurations, timeout 600

Common-solved set = 24 instances (same set for all). BFS reference: 26/26, 28,896 nodes (incl. `CC_2_3_4__pl_7`), all optimal.

| Config | Coverage | Total nodes | Median | #subopt | avg + | worst | TO |
|---|---|---|---|---|---|---|---|
| baseline w=1.0 | 25/26 | 20,651 | 13 | 14/24 | +1.12 | +4 | 2_3_4_pl_7 |
| baseline w=2.0 | 25/26 | 7,537 | 21 | 14/24 | +1.00 | +4 | 2_3_4_pl_7 |
| dyn-MAX_DEPTH (C) | 25/26 | 2,856 | 11 | 16/24 | +1.33 | +3 | 2_2_3_pl_7 |
| C + w=2.0 | 26/26 | 3,826 | 16 | 9/24 | **+0.42** | **+2** | — |
| **plus W=1.0** | **26/26** | **2,227** | 15 | 17/24 | +1.54 | +6 | — |
| plus W=2.0 | 26/26 | 4,857 | 27 | 15/24 | +1.04 | +6 | — |
| plus val_loss-ckpt W=1.0 | 25/26 | 8,521 | 16 | 16/24 | +1.83 | +9 | 2_3_4_pl_7 |

### The three traded-timeout instances

| Instance (opt) | baseline | C | C+w2 | plus W=1 | plus W=2 | plus vloss |
|---|---|---|---|---|---|---|
| CC_3_2_3__pl_7 (7) | +4 / 18,154 n | +1 / 216 | +1 / 120 | +2 / 68 | **+1 / 36** | +3 / 41 |
| CC_2_3_4__pl_7 (7) | TO | +4 / 118 | **+2 / 68** | +6 / 479 | +3 / 247 | TO |
| CC_2_2_3__pl_7 (7) | +2 / 477 | TO | +2 / 541 | **+2 / 168** | +2 / 717 | +8 / 239,951 (!) |

plus is the only family that solves all three at W=1.0 *and* W=2.0.

## Per-flag attribution (where separable)

- **`--dynamic-max-depth`** (baseline → C): nodes ÷7.2, first solve of `CC_2_3_4__pl_7` family-wide; small training-metric gains. The single biggest lever.
- **`--ckpt-metric spearman`** (plus-vloss → plus, same training run, only the exported epoch differs): coverage 25→26, nodes 8,521→2,227, avg +1.83→+1.54, worst +9→+6, and it removes a pathological 239,951-node / 560 s near-timeout solve. **Decisive in-planner win for ranking-based selection.**
- **`--include-unreachable`** (C → plus-vloss, both val_loss-selected): nodes 2,856→8,521 and quality −0.50 — i.e. as currently implemented (target-only, plain MSE) the unreachable signal *alone* HURT under val_loss selection; it pays off only combined with the Spearman checkpoint (plus W=1.0 beats C on coverage and nodes). The two flags interact through what the selection metric rewards; attribution here is partly confounded by the val set itself changing (unreachables included).
- **`--heuristic-weight`** on plus: W=2.0 *worsens* nodes (2,227→4,857) but improves plans (+1.54→+1.04). **Answer to the stated question: yes — the unreachable fix (plus the rest of plus) removes the node-count benefit W=2.0 had on the C model; plus@W=1.0 already exceeds C+w=2.0 on coverage and expansions.** W remains useful purely as a plan-quality knob on plus.

## Recommendation — production defaults

For the current model generation:

- **Best plan quality + full coverage: C + w=2.0** equivalent, i.e. plus with `--include-unreachable false --ckpt-metric val_loss --heuristic-weight 2.0` … but that forfeits plus's coverage robustness on the pl_7 trio. avg +0.42 / worst +2 is unmatched.
- **Best expansions + full coverage (recommended default): plus as-is at `--heuristic-weight 1.0`** (`--dynamic-max-depth true --include-unreachable true --ckpt-metric spearman`): 26/26, lowest node count measured (2,227), 12.9× fewer expansions than BFS, every instance under 3.4 s. Accept avg +1.54.
- Middle ground: plus at `W=2.0` (26/26, +1.04) if plan quality matters more than the ~2× node increase.

Flag the suboptimality trade-off in any paper comparison; with inadmissible heuristics, coverage and expansions are only half the story.

## Failures / notes

- Plus import mechanism initially shadowed its own `src` package by also inserting the baseline dir into `sys.path` — caught by the `--help` smoke test, fixed before commit (`__main__.py` now inserts only the plus dir; the package-path extension reaches the baseline).
- The renamed `by_val_loss` ONNX references its external-data file by original name; fixed by staging `distance_estimator_by_val_loss.onnx.data` under its original name during eval 3 (cleaned up after).
- 15-epoch validation could not confirm "unreachable predicts HIGH" (mean 1.5 in-dist); at 200 epochs partially confirmed (mean 5.5 in-dist, correct ordering OOD) — documented above; full fix needs loss upweighting (future work).
- Model↔C-file pairing audit: every results dir contains its exact `distance_estimator_C_used.txt`; `_models/CC/` restored to the baseline ep193 pairing (diff-verified) at the end; plus artifacts isolated in `_models_plus/CC/`; protected dirs (`CC_baseline_ep193/`, all `_results*/`) untouched except for adding new sibling results dirs.
- The user-staged `exp/.../CC/*.txt` files were temporarily stashed to let `git revert` run (dirty-index refusal) and restored with `git stash pop --index` — staging area byte-identical afterwards.
