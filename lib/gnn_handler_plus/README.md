# gnn_handler_plus

A thin override layer on top of `lib/gnn_handler` (the invariant baseline =
published method + correctness/infrastructure fixes only). This package
imports the baseline and overrides exactly three points — target
preparation, checkpoint selection, C-file export — behind flags, so every
method change is individually attributable and reversible. Nothing from the
baseline is copied; `src/__init__.py` extends the package path into
`lib/gnn_handler/src`, so parsing, the model, ONNX wrappers and the data
pipeline always come from the baseline.

From the planner's point of view both variants are drop-in interchangeable:
same ONNX input names/dtypes, same `distance_estimator_C.txt` format
(`slope = ...` and `intercept = ...` first; the C++ parser stops there, the
provenance lines that follow are ignored).

```bash
.venv/bin/python lib/gnn_handler_plus/__main__.py \
  --folder-raw-data ... --subset-train ... \
  --dir-save-model exp/gnn_exp/batch0/_models_plus/CC \
  --dir-save-data  exp/gnn_exp/batch0/_models_plus/CC \
  --dataset_type HASHED \
  [--dynamic-max-depth true] [--include-unreachable true]
  [--unreachable-cap 0.25] [--ckpt-metric spearman] [--heuristic-weight 1.0]
```

NOTE: `--include-unreachable` requires a `samples.pt` built by THIS entry
point (the baseline build drops unreachable rows at build time); run once
with `--build-data true` per dataset directory.

## Default configuration: the node-economy champion

The argparse defaults are not placeholders — they are the **measured
node-economy champion** on batch0/CC at t600, per the single-convention
table in `REPORT_plus_investigations.md` (cross-investigation synthesis):
running this entry point with **no plus flags at all** reproduces the
"plus W=1.0" row — **26/26 coverage, 2,874 total expanded nodes**
(13.2× under BFS's 37,927, 7.4× under the fixed baseline's 21,128),
plan quality **+1.73 avg / +6 worst** over BFS-optimal.

| Flag | Default (champion) |
|---|---|
| `--dynamic-max-depth` | `true` |
| `--include-unreachable` | `true` |
| `--unreachable-cap` | `0.25` |
| `--unreachable-loss-weight` | `1.0` |
| `--ckpt-metric` | `spearman` |
| `--heuristic-weight` | `1.0` |

These defaults are guarded by `tests/test_defaults.py` — changing any of
them fails the test, because "plus with no flags" is the reference point
every number in the report compares against.

**Two caveats (verbatim from the synthesis):**

1. **Node economy is bought at a plan-quality cost.** For quality, use
   W=2.0 on the same trained model (+1.15 avg / +6 worst at 5,821 nodes)
   or the C+w=2.0 recipe (`--include-unreachable false --ckpt-metric
   val_loss --heuristic-weight 2.0`; 26/26, +0.54/+2 at 4,435 nodes).
2. **Defaults are validated on CC only.** On Grapevine these defaults
   COLLAPSE in training (INV-5), and `--dynamic-max-depth` /
   `--include-unreachable` need guards before cross-domain use — see the
   three-mechanism failure catalogue in `REPORT_plus_investigations.md`
   (bias attractor → `--include-unreachable`; saturation collapse →
   `--dynamic-max-depth`).

**Presets** — the two named operating points, with exact flag sets so
future runs can cite them unambiguously:

| Preset | Exact flags | Measured (batch0/CC t600) |
|---|---|---|
| **node-economy default** | *(none — all defaults)* ≡ `--dynamic-max-depth true --include-unreachable true --unreachable-cap 0.25 --unreachable-loss-weight 1.0 --ckpt-metric spearman --heuristic-weight 1.0` | 26/26, 2,874 nodes, +1.73 avg / +6 worst |
| **quality preset** | `--include-unreachable false --ckpt-metric val_loss --heuristic-weight 2.0` (the C+w=2.0 recipe; other flags at default) | 26/26, 4,435 nodes, +0.54 avg / +2 worst |

Middle ground on the same champion-trained model (no retraining, export
only): `--heuristic-weight 2.0` with everything else default → 26/26,
5,821 nodes, +1.15/+6. Never W=1.5 (dominated; see synthesis).

## (i) Why this package exists — problems observed in gnn_handler

All findings are documented in `REPORT_unattended_run.md` and
`REPORT_sweep_and_maxdepth.md` (repo root), on the batch0/CC domain.

1. **Hardcoded `MAX_DEPTH = 50` wastes the sigmoid range.** The deepest
   distance in the training data is 23, so scaled targets averaged 0.0105 —
   a weak signal at an awkward sigmoid operating point (logit ≈ −4.6).
   Downstream, the outlier post-mortem showed true distances 1–9 collapsing
   into h ∈ {0,1,2} (band compression) on `CC_3_2_3__pl_7`. Retraining with
   data-driven scaling (MAX_DEPTH 26) improved every training metric and cut
   total node expansions 7.2× on the common-solved set.

2. **Unreachable states are filtered from training → dead ends look like
   goals.** The build pipeline drops sentinel-distance (1000000) rows, so
   the model has never seen a dead end; the post-mortem measured mean
   predicted distance **0.59** on known-unreachable states (ideal: the top
   of the range). On `CC_3_2_3__pl_7` this actively misled A*: 18,154
   expansions and a +4 plan where BFS needs 3,361. Counted in the raw CSVs:
   726 unreachable rows / 332,590 total (0.2%) — small enough that a cap is
   a safety net, not a daily constraint.

3. **Checkpoint selected by calibration, consumed as ranking.** The baseline
   keeps the lowest-val_loss (1−R²) epoch, but A* consumes the *ordering* of
   h values. On the dynamic-MAX_DEPTH run the last epoch had Spearman
   0.9096 while the exported best-by-val_loss checkpoint had 0.8865 — the
   exported model was not the best-ranking model the run produced.

4. **The learned heuristic is inadmissible and per-instance brittle.**
   14–15/25 plans were suboptimal at w=1 (avg +1.12); worse, configurations
   trade timeouts on *different* instances (`CC_2_3_4__pl_7` vs
   `CC_2_2_3__pl_7`) — per-instance variance, not average quality, is the
   dominant deployment risk. The heuristic-weight knob (h/W) measured in the
   sweep was the single most effective mitigation (w=2.0: fewer nodes AND
   better plans), but lived in a hand-edited file.

5. **Targets are heavily zero-skewed.** 64.4% of training targets are
   distance 0 (within-class cap is 50% per instance CSV before splitting),
   diluting gradient signal for deep-state discrimination — visible as the
   near-goal band compression of (1).

6. *(history note)* The output clamp (`clamp(min=1e-3)`) silently zeroed all
   gradients once the whole batch sank below the floor, freezing training at
   epoch 3 for 197 epochs (constant val R² −0.3721). Fixed in the baseline
   itself (training-mode bypass) since it is a bug, not a method choice —
   recorded here because several published-era results predate the fix.

## (ii) What plus changes (all flag-controlled)

| Flag | Default | Addresses | Measured effect (where known) |
|---|---|---|---|
| `--dynamic-max-depth` | true | (1) range waste | R² 0.9184→0.9209, Spearman 0.8780→0.8865, nodes ÷7.2 on common set; first-ever solve of `CC_2_3_4__pl_7` (REPORT_sweep_and_maxdepth, Task C) |
| `--include-unreachable` (+ `--unreachable-cap 0.25`) | true | (2) dead-end blind spot | new in plus — validated by predicting HIGH on known-unreachable post-mortem states |
| `--ckpt-metric spearman` (always also saves `*_by_val_loss.pt`) | spearman | (3) ranking/calibration mismatch | new in plus — both checkpoints saved for in-planner A/B |
| `--heuristic-weight W` | 1.0 | (4) inadmissibility trade-off | from the sweep: C+w=2.0 gave 26/26 coverage, 9/24 subopt, avg +0.42 (best config measured); W recorded in C-file + history for reproducibility |

Reference comparison (timeout 600, common-solved 24 instances, from
REPORT_sweep_and_maxdepth.md):

| Config | Coverage | Nodes | #subopt | avg + | worst |
|---|---|---|---|---|---|
| baseline w=1.0 | 25/26 | 20,651 | 14/24 | +1.12 | +4 |
| w=2.0 (post-hoc) | 25/26 | 7,537 | 14/24 | +1.00 | +4 |
| dynamic MAX_DEPTH | 25/26 | 2,856 | 16/24 | +1.33 | +3 |
| dynamic + w=2.0 | **26/26** | 3,826 | **9/24** | **+0.42** | **+2** |

The W default stays 1.0: the unreachable fix may remove exactly the
misleading-low-h errors that made w=2.0 win, so the right W must be
re-measured per model generation, not inherited.

## (iii) Perspectives / future work (NOT implemented)

- **Asymmetric loss** penalizing overestimation more than underestimation —
  pushes the regressor toward admissibility instead of post-hoc dividing.
- **Distance-0 downweighting / stratified batches** — attack the 64.4%
  zero-skew at the loss level rather than by undersampling files.
- **Ranking losses** (pairwise/listwise, e.g. RankNet-style): A* consumes
  ordering; learning-to-rank matches the consumer better than MSE
  regression. Spearman checkpointing (flag 3) is the cheap first step.
- **LR schedule** (cosine decay / warmup): convergence is front-loaded;
  epochs 150–200 gain ~0.02 R². Likely cheaper than more epochs.
- **Uncertainty-aware search**: ensemble or MC-dropout variance as a
  per-instance confidence signal; low confidence → fall back to BFS.
  Directly attacks the per-instance-variance risk (problem 4).
- **BITMASK path revisit**: HASHED squeezes 64-bit node identity through a
  float64 normalize (53-bit mantissa) — bitmask inputs are lossless and the
  infrastructure already exists (`bit_input`), but need planner-side
  bitmask emission for the same instances.
- **Planner-side batched ONNX + GPU execution provider** *(out of scope
  here: requires C++ changes)*: per-node inference costs 2.5–7 ms vs BFS's
  1–2 ms per expansion and dominates wall-clock; batching frontier
  evaluations would change the coverage economics entirely.

## Separated encoding

Inherits the baseline data path (`src.preprocessing` resolves to
`lib/gnn_handler`), so `--kind-of-data separated` requires `--use-goal true` and
feeds the per-instance `goal_tree.dot` as a separate goal graph. The separated
ONNX exports the goal inputs; training + export are **ready**, deployment pending
the C++ `GraphNN::run_inference` separated branch (mirror `FringeEvalRL`; the
goal tensor is already built at solve time). See `lib/rl_handler/SEPARATED.md`.
