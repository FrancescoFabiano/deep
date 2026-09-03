# Report — heuristic-weight sweep, outlier post-mortem, data-driven MAX_DEPTH

Session date: 2026-06-05. Branch: `Reinforcement-Learning`. Baseline model (epoch 193) preserved at `exp/gnn_exp/batch0/_models/CC_baseline_ep193/`.

## Task A — Outlier post-mortem: CC_3_2_3__pl_7

Ground truth generated with the planner's own dataset mode (`--dataset --dataset_depth 25 --dataset_discard_factor 0.4 --dataset_type HASHED`), same as training data. Diagnostic kept at `scripts/gnn_exp/postmortem_outlier.py`. Note: `CC_2_2_4__pl_7` needed a seed retry (seed 42 → "No goals found", exit 3; seed 1337 → OK), mirroring the retry loop the training generator uses.

### Ranking quality (Spearman pred vs true, unreachable states excluded)

Instance

n states

Spearman

Search outcome at w=1.0

**CC_3_2_3__pl_7 (outlier)**

10,661

**−0.055**

18,154 nodes, len 11 (+4)

CC_3_2_3__pl_6 (control)

25,958

0.191

6 nodes, optimal

CC_2_2_4__pl_7 (control)

22,318

**0.606**

74 nodes (BFS: 4,163), +1

pl_7 is a genuine **ranking** outlier — the heuristic is uncorrelated/slightly inverted with true distance on this instance's state space, not merely unlucky in tie-breaking.

### Where it goes wrong

-   **Shortest-path profile** (true 19→0): long range is fine (19/18/17/16 predicted 18.5/18.1/17.4/15.3), but the **critical band true=1..9 collapses to h ∈ {0,1,2}** with 9 inversions — A* is told "you're almost there" nearly everywhere, which produces exactly the observed plateau-thrashing.
-   **Bucket stats** (pred_dist mean ± std): true 0 → 0.80±0.42, 1 → 0.51±0.53, 2-3 → 0.70±0.63, 4-5 → 0.80±0.63, 6+ → 1.24±1.95. Buckets 0 and 4-5 have the SAME mean — the model cannot discriminate distance at all here, and is even mildly inverted at the bottom (dist-1 mean < dist-0 mean).
-   **Unreachable states** get mean pred_dist 0.59 (min 0.00) — dead subtrees look like goals, actively pulling the search in.

### Hypothesis check: out-of-distribution?

**Rejected for graph size**: outlier states are 20±1 nodes / ~56 edges; training max is 62 nodes / 668 edges (CC_3_2_3-family max: 32 / 231). 0% of outlier states exceed training maxima.

Most likely cause: with HASHED node IDs, node features are arbitrary hash positions in [−1,1]; generalization to unseen states can only come from structure/edge labels. The training subset contains CC_3_2_3 only up to pl_5; pl_7's deeper state space is *functionally* novel (new hash content), even though graph sizes are in range. Note the val-set Spearman (0.878) is dominated by depth-0/1 states from training instances — per-unseen-instance ranking is much weaker (0.19–0.61), with pl_7 the extreme.

## Task B — Heuristic-weight sweep via the C file (model unchanged)

Method: `slope' = w × slope` written to the live `distance_estimator_C.txt` (h_planner = round((onnx−intercept)/slope') ≈ h/w), bulk eval at timeout 600, results archived with the exact C file used (`_results_slope_w*/distance_estimator_C_used.txt`). Original C file restored and verified by diff afterwards. BFS optimal lengths = pl_N.

w

Coverage

Total nodes*

Median nodes*

# subopt

avg +steps

worst

eff. max h

CC_3_2_3__pl_7

CC_2_3_4__pl_7

1.0 (baseline)

25/26

21,128

14

15/25

+1.16

+4

50

len 11 / 18,154 n

TO

1.25

25/26

27,846

21

17/25

+1.28

+4

40

len 11 / 2,878 n

TO

1.5

25/26

**8,839**

28

18/25

+1.20

**+3**

33

len 9 / **1,278 n**

TO

**2.0**

25/26

**8,280**

21

**15/25**

**+1.04**

+4

25

len 9 / 1,825 n

TO

* over the 25 instances solved by all configurations (same set in every run).

Findings:

-   **w=2.0 dominates the baseline on BOTH axes**: 2.55× fewer total expansions AND slightly better plan quality (+1.04 vs +1.16 avg). The usual weighted-A* trade-off doesn't bite here because the heuristic's absolute scale is already unreliable (Task A); discounting it mostly de-weights its *errors*.
-   The sweep is non-monotone: w=1.25 is *worse* than baseline on nodes (27,846) — small reweighting reshuffles plateau tie-breaking without flattening the misleading peaks.
-   The Task-A outlier improves dramatically under any w ≥ 1.25 (18,154 → 1,278–2,878 nodes) — consistent with its failure mode being overcommitment to wrong low-h plateaus.
-   `CC_2_3_4__pl_7` never completes at any w: not a weighting problem (per-node ONNX cost × required expansions exceeds 600 s).
-   Quantization: effective max h = round(0.998/slope′) shrinks 50 → 25 as w grows; with observed distances ≤ 23 this is harmless here, but it bounds how far w can be pushed on deeper domains.
-   Discrepancy note: the previous report said 14/25 suboptimal at w=1.0; systematic re-parse gives **15/25** (same avg +1.16) — the earlier manual count missed one instance.

**Recommendation: w = 2.0** is the best operating point for this model (fewest nodes, best plan quality, coverage unchanged). w=1.5 is a close second and has the better worst case (+3).

## Task C — Data-driven MAX_DEPTH = ceil(1.1 × max train distance)

Code change (commit `b35292e`): `prepare_samples` now computes `MAX_DEPTH = ceil(1.1 × max raw train distance)` (here: 23 → **26**, slope 0.01996 → **0.03838**, ~1.92× steeper) instead of the hardcoded 50; the constants flow into `distance_estimator_C.txt` automatically (planner parses slope/intercept and ignores the extra `max_depth` line — verified read-only against `GraphNN.tpp`), and are now also recorded in `history_losses.json`. Verified `samples.pt` stores RAW distances (max 23), so `--build-data false` was valid for the retrain.

### Training metrics (200 epochs, vs baseline)

| | best epoch | val R² | Spearman | MAE (steps)* |
|---|---|---|---|---|
| Baseline (MAX_DEPTH=50) | 193 | 0.9184 | 0.8780 | 0.0752 |
| **Dynamic (MAX_DEPTH=26)** | 193 | **0.9209** | **0.8865** | **0.0724** |

\* MAE divided by each model's own slope — raw MAE values are not comparable across scalings.

Small but consistent improvement on all three metrics (and last-epoch Spearman reached 0.9096, the highest seen anywhere).

### Evaluation, timeout 600 (`_results_maxdepth_dynamic/`)

Per-run coverage and metrics over the 24 instances solved by ALL configurations:

| Config | Coverage | TO instance(s) | Total nodes | Median | # subopt | avg +steps | worst |
|---|---|---|---|---|---|---|---|
| baseline w=1.0 | 25/26 | CC_2_3_4__pl_7 | 20,651 | 13 | 14/24 | +1.12 | +4 |
| w=2.0 (Task B) | 25/26 | CC_2_3_4__pl_7 | 7,537 | 21 | 14/24 | +1.00 | +4 |
| **dynamic MAX_DEPTH (C)** | 25/26 | CC_2_2_3__pl_7 | **2,856** | **11** | 16/24 | +1.33 | **+3** |
| **C + w=2.0 (composition)** | **26/26** | — | 3,826 | 16 | **9/24** | **+0.42** | **+2** |

- The C model **solves `CC_2_3_4__pl_7` for the first time in any configuration** (118 nodes, 1.6 s; the t60-broken, baseline, and all w-sweep runs timed out on it; BFS needs 9,031 nodes / 81 s). The Task-A outlier `CC_3_2_3__pl_7` drops from 18,154 nodes (+4) to 216 (+1).
- But it trades a new timeout: `CC_2_2_3__pl_7` (which baseline solved with 477 nodes) — heuristic quality remains instance-idiosyncratic; per-instance variance is the dominant risk, not average quality.
- 7.2× fewer total expansions than baseline on the common set, with slightly worse average suboptimality (+1.33 vs +1.12).

**Should dynamic MAX_DEPTH become the default? Yes** — it is a strict improvement on training metrics, dramatically reduces expansions, solves the hardest instance, and removes a magic constant. Its one regression (the new TO) is eliminated by composing with w=2.0 (below).

## Interaction note (B × C)

**They compose, and the composition is the best configuration tested.** An extra eval was run with the C model and w=2.0 applied to *its own* slope (0.03838 × 2 = 0.07677, archived in `_results_maxdepth_w200/` with the C file used):

- **Coverage 26/26 — the only full-coverage GNN configuration** (BFS-equivalent coverage with 7.6× fewer total expansions than BFS on the common set).
- **Best plan quality of all configs: 9/24 suboptimal, avg +0.42, worst +2** — better than even unweighted baseline (+1.12). Discounting h primarily de-weights the heuristic's *errors* (Task A showed the absolute scale is unreliable), and the better-ranked C model loses even less.
- Both problem instances solve fast: `CC_3_2_3__pl_7` len 8 / 120 nodes, `CC_2_3_4__pl_7` len 9 / 68 nodes, and C-alone's regression `CC_2_2_3__pl_7` len 9 / 541 nodes.

Two caveats: (1) w values are relative to each model's own slope — Task B divides a fixed model's output scale, Task C retrains the sigmoid to use the range; baseline-w settings must not be reused verbatim on the C model. (2) Task C's slope is coincidentally ≈ the baseline's w=1.92 point, but the mechanisms differ (retrained representation vs post-hoc rescale) and so do the results (C alone: 2,856 nodes / +1.33 vs B w=2.0: 7,537 / +1.00).

**Recommended production setting: dynamic-MAX_DEPTH model with w=2.0 on its slope** (i.e. C file slope ×2). If optimality matters more than coverage, C+w≈1 with a fallback to BFS on timeout is the alternative.

## Failures / notes

-   `CC_2_2_4__pl_7` dataset generation failed on seed 42 (exit 3, "No goals found"); succeeded on retry with seed 1337 — same behavior class the training-data generator handles with its retry loop. Not a bug.
-   One commit slip, immediately corrected: the post-mortem-script commit briefly swept in 11 pre-staged `exp/.../CC/*.txt` problem files from the index; soft-reset and recommitted script-only (`6d556a5`). The `.txt` files remain staged exactly as they were before the session.
-   Commits this session (local only): `b35292e` (dynamic MAX_DEPTH), `6d556a5` (post-mortem diagnostic).
-   Model/C-file pairing audit: live `_models/CC/` holds the dynamic-MAX_DEPTH model with ITS C file (restored and diff-verified after each sweep); baseline pair preserved in `_models/CC_baseline_ep193/`; every sweep results dir contains the exact `distance_estimator_C_used.txt` it ran with.
-   The B sweep numbers in this report use the 24-instance common-solved set (all four configs), so baseline rows differ slightly from the Task-B table above, which uses the 25-instance set common to the four w-runs only.