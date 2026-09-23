# gnn_handler_plus investigations (INV-1 … INV-5)

Session date: 2026-06-05/06. Branch: `Reinforcement-Learning`. All evals: batch0/CC, timeout 600, threads 4, planner binary unchanged. Hypotheses below were written and committed to this file BEFORE the corresponding runs were launched.

Reference numbers (from REPORT_gnn_handler_plus.md, common-solved 24): plus W=1.0 = 26/26 / 2,227 nodes / +1.54 avg / +6 worst; plus W=2.0 = 26/26 / 4,857 / +1.04 / +6; C+w=2.0 = 26/26 / 3,826 / +0.42 / +2; plus val R² 0.34 all-states, Spearman 0.8634; unreachable val states n=123, predicted mean 5.5 (target 26).

## INV-1 — Reachable/unreachable metric split (instrumentation, no retrain)

**Addressed problem:** plus's headline val R² collapsed 0.92 → 0.34 — an artifact of 123 unreachable val states (target ≈ 26 in distance units, predicted mean ≈ 5.5) demolishing a squared-error metric while all-states Spearman stayed 0.8634. Anyone reading `history_losses.json` would conclude plus trains worse than C; and val_loss-based checkpoint selection is dominated by those 123 points — the in-planner pathology of the val_loss checkpoint (239,951-node solve on `CC_2_2_3__pl_7`) is plausibly this mechanism.

**Why now:** every later investigation (INV-2/3/5) needs honest per-subset metrics to attribute effects; and the spearman-vs-val_loss checkpoint comparison needs grounding in what val_loss was actually measuring.

**Hypothesis (expected outcome):** plus reachable-only val R² ≈ 0.90 (within a few points of C's 0.92), i.e. the R² collapse is attributable ENTIRELY to the unreachable subset; the C model's val set contains zero unreachable states (its samples.pt was baseline-built). Secondary: the val_loss-selected plus checkpoint will not look better than the spearman-selected one on reachable-only metrics.

**Method:** extend `DistanceEstimatorModelPlus.evaluate()` to compute every metric three ways — all / reachable-only (`*_reach`) / unreachable-only (`mae_unreach`, `pred_dist_mean_unreach`, `n_unreach`) — identifying unreachable samples by their scaled target = f(MAX_DEPTH) (no reachable state can alias it: max reachable distance 23 < 26 by the 1.1 headroom). Flows into history_losses.json for all future runs. Recompute for the EXISTING checkpoints (plus spearman-selected, plus val_loss-selected) on the plus val split, and for the C model (`_models/CC_maxdepth_ep193/`) on its own (baseline-built) val split. No training.

**Real outcome:**

| Checkpoint | R² (all) | R² (reach) | Spearman (all) | Spearman (reach) | MAE (reach) | unreach pred mean (target 26) |
|---|---|---|---|---|---|---|
| plus spearman-ckpt (ep132) | 0.3400 | **0.6164** | 0.8634 | 0.8631 | 0.0069 | 5.50 |
| plus val_loss-ckpt (ep168) | 0.3611 | **0.7594** | 0.8542 | 0.8538 | 0.0063 | 5.27 |
| C model (ep193) | 0.9209 | 0.9209 | 0.8865 | 0.8865 | 0.0028 | n/a (0 unreachable — confirmed) |

**Verdict:** PARTIAL — the unreachable subset explains only part of the R² collapse. Removing the 123 unreachable states lifts plus R² from 0.34 to 0.62 (spearman ckpt) / 0.76 (val_loss ckpt) — still far from C's 0.92. So including unreachables in TRAINING degraded reachable-state CALIBRATION broadly (consistent with INV-2's upward-bias mechanism), while RANKING stayed nearly intact (Spearman_reach 0.863 vs C's 0.887). The secondary hypothesis also fell: the val_loss checkpoint is *better* calibrated on reachable states (R² 0.76, MAE 0.0063) than the spearman one, yet was decisively worse in-planner — the strongest evidence yet that calibration metrics simply do not predict planner performance; ordering does. New belief: (a) split metrics are mandatory for any unreachable-trained model; (b) plus's quality regression is a training-distribution effect, not a metric artifact — INV-2 directly tests it.

## INV-2 — Unreachable ablation: attribute plus's plan-quality regression

**Addressed problem:** plus W=1.0 wins coverage and nodes (26/26, 2,227) but loses plan quality (+1.54 avg / +6 worst) vs C+w=2.0 (+0.42 / +2). Proposed mechanism: training dead ends toward target=26 pushes the model to OVERESTIMATE ambiguous states generally — aggressive pruning (fewer nodes) and inadmissibility (worse plans) are two faces of the same upward bias. Also unresolved: whether the 26/26 coverage follows the unreachable flag or the Spearman checkpoint (last session's C-vs-plus comparisons confounded the two).

**Why now:** decides whether `--include-unreachable` stays in the recommended default set, and cleanly separates its effect from `--ckpt-metric spearman` (this run keeps spearman selection while removing only the unreachable signal).

**Hypothesis (expected outcome):** plan quality recovers substantially — avg + ≤ +1.0 and worst ≤ +4 on the common-24 — and node count rises moderately (2,227 → 3,000–5,000). Coverage prediction: 26/26 retained (i.e. I expect the Spearman checkpoint, not the unreachable signal, is what buys the 26th instance — `CC_2_3_4__pl_7` stays solved). If coverage drops, the lost instance identifies what unreachable inclusion was buying.

**Method:** train plus 200 epochs, `--include-unreachable false`, all other flags default (dynamic-max-depth true, ckpt-metric spearman, W=1.0), same subset/seed/batch; model dir `_models_plus/CC_nounreach` (existing plus model untouched — INV-4 needs it); reuse plus-built samples.pt (`--build-data false`; the include flag filters at load). Eval t600 → `_results_plus_nounreach_w100/`. Three-way compare: plus-full, plus-no-unreach, C+w=2.0, with INV-1 split metrics.

**Real outcome:** (run executed after a session crash; partial `CC_nounreach` from the crashed session was deleted and retrained from scratch, 200 epochs, 1h05.) Training confirmed the load-time filter: `train_unreachable_kept = 0`, 123 val unreachables also dropped, MAX_DEPTH stays 26 (dynamic scaling uses reachable max 23 — predictions scale-comparable to plus-full). Exported spearman ckpt = ep92: **Spearman 0.9150, R² 0.8986, MAE 0.0035** (reachable-only by construction — directly comparable to C's 0.8865/0.9209; nounreach *beats C on ranking* and approaches it on calibration). val_loss-best ep192: Spearman 0.8948, R² 0.9263.

Eval (t600, W=1.0): **26/26**, common-24: **avg + = +1.04, worst +4, 14/24 subopt, 6,504 nodes** (all-26: 7,197). Per-instance: `CC_2_3_4__pl_7` solved at +3/395 (plus-full: +6/479); `CC_2_2_3__pl_7` +1/298 — the instance the C model TIMED OUT on; node outlier `CC_2_3_4__pl_5` 2,156 nodes (33% of the common-24 total).

| Config | Coverage | Nodes (c24) | avg + | worst |
|---|---|---|---|---|
| plus-full W=1.0 | 26/26 | 2,227 | +1.54 | +6 |
| **plus-nounreach W=1.0** | **26/26** | 6,504 | **+1.04** | **+4** |
| C+w=2.0 | 26/26 | 3,826 | +0.42 | +2 |
| dyn-MAX_DEPTH (C) | 25/26 | 2,856 | +1.33 | +3 |

**Verdict:** LARGELY CONFIRMED — with one quantitative miss in the expected direction. Quality recovered as predicted (avg +1.54 → +1.04, right at the ≤+1.0 bound; worst +6 → +4, met), and coverage stayed 26/26 with `CC_2_3_4__pl_7` solved — confirming the secondary prediction that the **Spearman checkpoint, not the unreachable signal, buys the 26th instance**. Sharper still: nounreach differs from the C model essentially only by ckpt-metric (both train on zero unreachables, same dynamic scaling), and it solves `CC_2_2_3__pl_7` where C timed out — the cleanest attribution yet of coverage to spearman selection. The miss: nodes rose to 6,504, above the predicted 3,000–5,000 band (2.9× plus-full) — the overestimation mechanism is confirmed in direction (no dead-end signal → less aggressive pruning → more nodes + better admissibility) but its magnitude was underestimated; unreachable training was buying ~⅔ of plus's node advantage. Consequence for the default set: `--include-unreachable` is a genuine speed/quality dial, not a free win — keep it default-on for node-economy, but INV-3 (amplify) now decides whether the dial has a usable upper setting.

## INV-3 — Unreachable loss upweighting

**Addressed problem:** the dead-end lesson is starved: 726/332,590 raw samples (0.2%; 603/201,099 = 0.2999% in the train split) carry ~0.3% of MSE gradient mass. The model learned the ORDERING (unreachable 2.28 vs distance-0 0.81 even OOD) but not the MAGNITUDE (in-dist predicted mean 5.5 vs target 26). Dead-end avoidance was the root-cause fix indicated by the CC_3_2_3__pl_7 post-mortem; partial learning is partial protection.

**Why now:** INV-2 measures removing the signal; this measures amplifying it — together they bracket the dial and tell us whether "unreachable-as-max + enough gradient" is a viable default or inherently quality-toxic (per the INV-2 overestimation mechanism).

**Hypothesis (expected outcome):** with U chosen for ~10% effective loss mass (U=35 → exact mass 35×0.0029989/(35×0.0029989+0.9970) = 9.5%), unreachable predicted mean rises 5.5 → >15; node count equal or LOWER than plus-full (≤2,227; sharper dead-end pruning); plan quality equal or slightly worse (avg + in +1.5–2.0) — the pruning/admissibility trade-off SHARPENS rather than disappears. Reachable-only R²/Spearman drop by ≤0.02 (mild collateral).

**Method:** new plus flag `--unreachable-loss-weight U` (default 1.0): weighted MSE in `_compute_loss` — unreachable samples (identified by scaled target = f(MAX_DEPTH), same rule as INV-1) get weight U, loss = Σw·e²/Σw. Record exact effective mass from the actual counts. Train 200 epochs with U=35, other flags default, model dir `_models_plus/CC_uw`; eval t600 W=1.0 → `_results_plus_uw_w100/`. INV-1 metrics verify the magnitude shift and collateral.

**Real outcome (training metrics; planner eval below):** training completed 200 epochs (survived a session crash; verified post-hoc: full 200-entry history, all checkpoints/ONNX present). Exact effective loss mass: 35×603 / (35×603 + 200,496) = **9.52%** — right at the 9.5% design point. Best-spearman checkpoint at epoch 163 (the saved/staged model). Metrics there, vs plus-full (best ep 133) and nounreach (best ep 92) on the identical split:

| Metric (best ckpt) | plus-full (U=1) | nounreach | **uw (U=35)** | hypothesis |
|---|---|---|---|---|
| Spearman (full val) | 0.863 | 0.915 | **0.699** | drop ≤0.02 |
| Spearman_reach | 0.863 (INV-1) | 0.915 | **0.698** | drop ≤0.02 |
| R² (full val) | 0.34 | 0.899 | **−3.78** | — |
| R²_reach | 0.62 (INV-1) | 0.899 | **−17.99** | drop ≤0.02 |
| MAE_reach | 0.0063 (INV-1) | 0.0035 | **0.0650** | mild |
| pred mean (unreach, target 26) | 5.5 | n/a | **9.74** (10.1 at ep 200) | >15 |

The magnitude shift happened in the predicted direction but stalled at ~⅓ of target range (5.5 → 9.7, not >15), while collateral damage was catastrophic, not mild: reachable ranking lost 0.165 Spearman and reachable calibration went from R² +0.62 to −18 (MAE_reach 10× worse). At 9.5% loss mass the 603 unreachable samples don't just rebalance the gradient — they dominate early optimization (best val_loss epoch is epoch 1; the network never recovers a calibrated reachable regression afterwards).

**Real outcome (planner, t600 W=1.0 → `_results_plus_uw_w100/`):** coverage **25/26** — uw is the first plus variant to LOSE an instance: `CC_2_3_4__pl_7` genuinely timed out (run wall-time 21:20→21:32 confirms a full 600 s burn), the same instance whose coverage INV-2 attributed to the Spearman checkpoint. The spearman ckpt was used here too — so a Spearman of 0.699 is below what that mechanism needs; checkpoint selection can't rescue a heuristic this degraded. Quality: avg +1.96 over BFS-optimal on the 25 solved (worst +7 on `CC_2_3_4__pl_5`; 18/25 instances suboptimal) — inside the predicted +1.5–2.0 band, but at its bad edge. Nodes: **22,215 total** over 25 solved vs plus-full's 2,874 over 26 (same parser/convention; report's earlier 2,227/6,504 figures used a slightly different aggregation — relative ordering unaffected) — the prediction was equal-or-LOWER, reality is **~8× higher**, with blowups on `CC_3_3_3__pl_7` (8,674) and `CC_2_2_3__pl_8` (6,448). The "sharper dead-end pruning" mechanism never materialized: degrading the reachable-state heuristic (R²_reach −18) costs far more expansions than better dead-end magnitudes could save.

**Verdict:** REFUTED — on three of four predictions (magnitude stalled at 9.7 vs >15; collateral catastrophic vs ≤0.02 Spearman; nodes 8× higher vs equal-or-lower), with only the quality band technically hit (+1.96 ∈ +1.5–2.0) and that for the wrong reason: quality degraded via a globally worse heuristic, not via a sharpened pruning trade-off. Combined with INV-2, the dial is now fully bracketed and the answer is asymmetric: U=0 trades ~2.5× nodes for +0.5 quality; U=1 (default) is the node-economy sweet spot; U=35 (~10% loss mass) is toxic in BOTH directions — more nodes AND worse quality AND lost coverage. The 0.3%-of-gradient "starvation" framing was wrong: the unreachable lesson is cheap to learn partially and ruinous to force fully, because at high weight 603 constant-target samples act as a bias attractor that overwhelms the 200k-sample regression (best val_loss at epoch 1 — the network effectively never trains past the unreachable prior). If a usable upper setting exists it is at single-digit U (~1–3% mass), but the INV-2/INV-3 bracket gives no reason to expect it beats the default. New belief: `--unreachable-loss-weight` stays default 1.0; the flag's value is as a measurement instrument, not a tuning knob.

## INV-4 — W frontier on the plus model (no retrain)

**Addressed problem:** W is now first-class and recorded, but the quality/speed frontier between the two measured points (W=1.0: 2,227 / +1.54; W=2.0: 4,857 / +1.04) is unmapped, and the baseline sweep proved non-monotonicity is possible (w=1.25 was WORSE than w=1.0 on nodes: 27,846 vs 21,128).

**Why now:** if a knee exists near W≈1.5 it changes the recommended default; and monotonicity (or not) on the plus model tells us whether the baseline's w=1.25 anomaly was model-specific plateau noise or a general phenomenon.

**Hypothesis (expected outcome):** roughly monotone between the endpoints — avg + falls and nodes rise with W — with a usable knee near W≈1.5 (avg + ≈ +1.2 at ≤3,500 total nodes, coverage 26/26 throughout). If non-monotonicity reappears, the flipped instances will be among the three known traders (`CC_3_2_3__pl_7`, `CC_2_3_4__pl_7`, `CC_2_2_3__pl_7`).

**Method:** EXISTING plus model held fixed (spearman ckpt; not the INV-2/3 models — one variable at a time). Stage model+C into `_models/CC/`, evals at W ∈ {1.25, 1.5} (slope × W in the C file, raw_slope/W recorded) → `_results_plus_w125/`, `_results_plus_w150/`; restore baseline pairing after. Serialized against trainings — planner timings are a measured quantity.

**Real outcome:** both evals ran t600, serialized, plus model staged with slope = raw_slope × W (`distance_estimator_C_used.txt` archived in each results dir); baseline pairing (`CC_baseline_ep193`) restored to `_models/CC/` afterwards. Convention note: all four W points below were recomputed from the per-instance tables with ONE parser (total nodes over solved; avg + vs BFS-optimal over the 26-instance Combined split). These differ slightly from the endpoint figures quoted earlier in this report (+1.54/2,227 and +1.04/4,857 — a different aggregation by an earlier session, exact provenance not reconstructed); the frontier SHAPE is what this investigation tests and all its points share one convention.

| W | Coverage | Total nodes | Avg + | Worst + |
|---|---|---|---|---|
| 1.00 | 26/26 | 2,874 | +1.73 | +6 |
| 1.25 | 26/26 | 3,348 | +1.62 | +6 |
| 1.50 | 26/26 | **5,817** | +1.35 | +6 |
| 2.00 | 26/26 | 5,821 | +1.15 | +6 |

Aggregate frontier is monotone in both coordinates (avg + falls, nodes rise with W) — the baseline's w=1.25 node anomaly did NOT reappear at the aggregate level. But it reappears per-instance, and exactly where predicted: among the three known traders, node counts wiggle non-monotonically across W (`CC_2_2_3__pl_7`: 168→81→972→717; `CC_3_2_3__pl_7`: 68→35→77→36), while `CC_2_3_4__pl_7` improves steadily (479→362→88→247 nodes, len 13→12→10→10). The striking structural feature: nodes jump +74% between W=1.25 and W=1.5 (3,348→5,817), then PLATEAU — W=2.0 costs the same nodes as W=1.5 but is strictly better on quality (+1.15 vs +1.35). W=1.5 is a dominated point.

**Verdict:** PARTIALLY CONFIRMED — monotone-between-endpoints held and coverage stayed 26/26 throughout, but the hypothesized knee at W≈1.5 (avg + ≈ +1.2 at ≤3,500 nodes) is refuted: W=1.5 delivers +1.35 at 5,817 nodes, missing the node bound by 66% and dominated outright by W=2.0. The efficient frontier among tested points is {1.0, 1.25, 2.0}: W=1.25 buys a small quality gain (+1.73→+1.62) for +16% nodes; everything past 1.25 pays the full ~2× node cost, so once you leave the node-economy regime there is no reason to stop short of W=2.0. The per-instance prediction scored cleanly: non-monotonicity, where present, was confined to the predicted trio. Recommended defaults: W=1.0/1.25 when node economy matters, W=2.0 when quality matters; do not ship W=1.5.

## INV-5 (conditional, run last) — Second domain

**Addressed problem:** every number so far is one domain (CC). The traded-timeout pattern across configs says per-instance variance is the dominant risk — cross-domain generalization is untested and is where the method must eventually stand.

**Why now:** last open external-validity question before any paper claim; also tests whether plus's flag defaults transfer or were tuned-to-CC.

**Hypothesis (expected outcome):** plus ≥ baseline on coverage and strictly fewer nodes in the new domain; the domain's unreachable ratio and max depth differ from CC's (0.2%, 23) and will be recorded as the parameters that scale each flag's leverage. Budget guard: if 1-instance generation × instance count projects > ~2 h, stop and write the schedule estimate instead.

**Method:** discover candidate domains under `exp/` (paper domains; batch1 expectations); pick ONE with a usable instance set; generate training data via `create_training_data.py` tooling for a modest subset (measure one instance first, extrapolate); train baseline AND plus 200 epochs each; eval both + BFS t600 → `_results_<domain>_{baseline,plus,bfs}/`.

**Real outcome (in progress — training phase):** domain picked: **Grapevine** (batch1; 4 train + 12 test instances; 7.7 GB raw training data already on disk, so generation was skipped and the budget guard passed trivially). Training subset: the 3 instances with raw data (`Grapevine_3__pl_3`, `Grapevine_3__pl_4`, `Grapevine_4__pl_2`), 46,274 train / 9,252 val samples, same seed/batch as CC (42/1024).

Domain parameters (the quantities this hypothesis said to record): max raw train distance 19 → dynamic MAX_DEPTH **21** (CC: 23→26); unreachable ratio **0.77%** train AND test (357/46,274; 71/9,252) — **2.6× CC's 0.30%**; the 0.25 cap does not bind (all 357 kept). Effective loss mass at U=1: count-share 0.77%, but ACTUAL epoch-1 MSE share ≈ **90%** (unreachable targets sit at scaled 1.0 vs near-zero initial predictions: mse_total 0.0085 vs mse_reach 0.0008 → unreachable contribution ≈ 0.0077/0.0085).

First result, and it is structural: **plus training COLLAPSED on Grapevine; baseline did not.** Plus (all defaults, 200 epochs): validation metrics froze bit-for-bit from epoch 22 (23 unique values in 200 epochs), predictions constant ≈0; `diagnose_frozen_training.py` (adapted to the plus model dir) shows 99.96% of val predictions pinned at the 0.001 clamp floor with pre-clamp outputs at ~0 — the original clamp-trap signature, reproduced despite the 220602a gradient fix (train_loss kept moving, 200 unique values; the collapse route is output saturation, not a dead gradient). Exported ckpt is merely the best pre-collapse epoch (17; Spearman 0.10 — unusable). Baseline (gnn_handler, hardcoded MAX_DEPTH=50, unreachables filtered): healthy past the collapse point (ep 27: Spearman 0.48, R² 0.37, improving) — completing as of this writing. This is the INV-3 bias-attractor mechanism predicted from data composition: at 2.6× CC's unreachable ratio, the constant-target mass (≈90% of initial loss) pulls the regressor into the clamp floor before the reachable signal can establish itself; CC at 0.3% survived the same dynamics, Grapevine at 0.77% does not. Next step (per the INV-3 finding, seed retry deferred): isolate with `--include-unreachable false` — if plus-minus-unreachables trains healthily, the collapse is attributable to data composition, not to dynamic scaling or other plus machinery; a seed retry afterwards as robustness check on whichever config trains.

**Isolation result (`--include-unreachable false`, same seed/data, model dir `Grapevine_plus_nounreach`):** a clean dose-response, sharper than the binary attribution sought. Plus-nounreach trains healthily through epoch ~142 — best-spearman ckpt at **epoch 77: Spearman 0.819, R² 0.53, MAE 0.0068** (usable; exported) — then ALSO collapses to the constant-output state from epoch 143 (val_loss pinned at 1.2660 = the constant-≈0-prediction value; Spearman 0). Three configs, one domain, one seed: plus-default collapses at ep 22; plus-nounreach at ep 143; baseline (static MAX_DEPTH=50, no unreachables) never, in 200 epochs. So the attribution splits: the unreachable mass (90% of initial loss) ACCELERATES the collapse ~7×, but the underlying instability is present without any unreachables and differs from baseline essentially only by dynamic MAX_DEPTH (21 vs 50, i.e., slope 0.0475 vs ~0.02 — a 2.4× steeper target scaling). Grapevine is collapse-prone under plus's steeper scaling in a way CC never showed; unreachable-as-max turns a late-training failure into an immediate one. Practical consequence: the spearman-ckpt mechanism partially insures against late collapse (ep-77 export is healthy), but cannot insure against early collapse (plus-default's ep-17 export at Spearman 0.10). Seed-7 retry of the healthy config running as robustness check; evals (baseline / plus-nounreach / BFS, t600) follow, serialized.

**Seed-7 retry (plus-nounreach, identical config/data, seed 7):** TOTAL collapse from epoch 1 — all 200 val_loss entries bit-for-bit identical at 1.2660 (the constant-≈0-output value); the model never learned anything (best "Spearman" 0.0 at ep 1). Plus training on Grapevine is therefore a **seed lottery**: seed 42 bought 142 healthy epochs (and a good ep-77 export), seed 7 bought zero. Combined with the dose-response above, the picture is: under plus's dynamic-MAX_DEPTH scaling Grapevine sits near a collapse basin whose entry time is seed-dependent (ep 1 / ep 143) and which unreachable-as-max mass reliably accelerates (ep 22 at seed 42). Baseline's shallower static scaling kept seed 42 out of the basin entirely (baseline robustness across seeds: untested). The cross-domain transfer question is thereby answered at the training level before any planner number: plus's defaults do NOT transfer to Grapevine — `--include-unreachable` had to be disabled to get any usable model, and even that is seed-fragile. Planner evals (t600: BFS / baseline / plus-nounreach-s42-ep77) will quantify what the surviving artifacts deliver.

**Planner evals: SKIPPED BY DECISION.** The t600 planner evals (BFS / baseline / plus-nounreach-s42-ep77 on Grapevine) were cancelled after repeated session crashes killed the eval runs partway (empty `out/coverage_results/run_1…run_21` debris removed). This is a decision, not an accident of scheduling: the hypothesis is already decidable on the training-level evidence alone, and the verdict below rests entirely on it. What planner evals would have added — quantifying what the one surviving healthy artifact (plus-nounreach seed-42 ep-77, Spearman 0.819) delivers in-planner on Grapevine — is **deferred work**, recorded in the next-round list of the synthesis, not silently omitted.

**Verdict:** REFUTED — at the training level, which is upstream of everything the hypothesis claimed. The hypothesis ("plus ≥ baseline on coverage and strictly fewer nodes in the new domain") presupposes that plus's defaults produce a usable model on the new domain; on Grapevine they do not, and the evidence is a clean dose-response plus a seed lottery:

- **Collapse dose-response (one domain, one seed, three configs):** plus-default collapses at **epoch 22**; plus with `--include-unreachable false` collapses at **epoch 143**; baseline (static MAX_DEPTH=50, no unreachables) **never collapses** in 200 epochs. The same constant-output clamp-floor signature each time (99.96% of val predictions pinned at the 0.001 floor, pre-clamp outputs ≈ 0).
- **Seed lottery:** the one config that produced a usable export (plus-nounreach, seed 42, ep-77 ckpt: Spearman 0.819 / R² 0.53) collapses from **epoch 1** at seed 7 — 200 bit-for-bit identical val_loss entries, best "Spearman" 0.0. Seed 42 bought 142 healthy epochs; seed 7 bought zero. No robust plus model was obtainable on Grapevine.
- **Prime suspect — dynamic MAX_DEPTH, with unreachables as accelerant:** plus-nounreach differs from baseline essentially only by dynamic MAX_DEPTH (21 vs 50, i.e. 2.4× steeper target scaling), and it still collapses — so the underlying instability is the steeper scaling, present with zero unreachable samples. Unreachable-as-max mass (0.77% of samples but ≈90% of epoch-1 MSE on this domain, 2.6× CC's ratio) does not cause the basin; it **accelerates entry ~7×** (ep 143 → ep 22).
- The spearman-ckpt mechanism partially insures against *late* collapse (the ep-77 export is healthy) but cannot insure against *early* collapse (plus-default's ep-17 export, Spearman 0.10, unusable).

So the cross-domain transfer question is answered before any planner number: **plus's defaults do not transfer to Grapevine** — `--include-unreachable` had to be disabled to get any usable model, and even that config is seed-fragile. The domain parameters the hypothesis said to record (unreachable ratio 0.77% vs CC's 0.30%; max depth 19→21 vs 23→26) are exactly the quantities that scale the failure. Planner-level quantification of the surviving ep-77 artifact: deferred (synthesis, next-round list).

## Cross-investigation synthesis

### The single-convention table (all batch0/CC, t600, one parser)

Every number below was produced by ONE parser (`/tmp/parse_inv.py`) over the per-instance result tables, under this convention, stated verbatim:

> Convention (the ONE convention all numbers use):
> - Source: the per-instance "Combined" table of each `<domain>_combined.tex`.
> - Coverage: instances with Goal=Yes out of all instances in the Combined split.
> - Total nodes: sum of NodesExpanded over SOLVED instances only.
> - Plan-quality: delta = PlanLength − optimal, where optimal = BFS t600 plan length for the same instance (fallback: the instance's `pl_N` suffix when BFS timed out).
> - avg+ / worst+ / #subopt computed over SOLVED instances only.

This supersedes the mixed aggregations quoted earlier in this report and in REPORT_gnn_handler_plus.md / REPORT_sweep_and_maxdepth.md (common-24 vs all-solved, manual vs scripted counts). Where a number here differs from an earlier section, this table is authoritative; the earlier sections' *relative orderings* all survive the re-parse.

| Config | Coverage | Total nodes | avg+ | worst+ | #subopt | Timed out |
|---|---|---|---|---|---|---|
| baseline w=1.0 | 25/26 | 21,128 | +1.16 | +4 | 15/25 | CC_2_3_4__pl_7 |
| baseline w=1.25 | 25/26 | 27,846 | +1.28 | +4 | 17/25 | CC_2_3_4__pl_7 |
| baseline w=1.5 | 25/26 | 8,839 | +1.20 | +3 | 18/25 | CC_2_3_4__pl_7 |
| baseline w=2.0 | 25/26 | 8,280 | +1.04 | +4 | 15/25 | CC_2_3_4__pl_7 |
| C (dyn-MAX_DEPTH) w=1.0 | 25/26 | 2,974 | +1.44 | +4 | 17/25 | CC_2_2_3__pl_7 |
| **C (dyn-MAX_DEPTH) w=2.0** | **26/26** | 4,435 | **+0.54** | **+2** | **11/26** | — |
| plus W=1.0 | 26/26 | **2,874** | +1.73 | +6 | 19/26 | — |
| plus W=1.25 | 26/26 | 3,348 | +1.62 | +6 | 19/26 | — |
| plus W=1.5 | 26/26 | 5,817 | +1.35 | +6 | 17/26 | — |
| plus W=2.0 | 26/26 | 5,821 | +1.15 | +6 | 17/26 | — |
| plus val_loss-ckpt W=1.0 | 25/26 | 248,472 | +2.08 | +9 | 17/25 | CC_2_3_4__pl_7 |
| plus-nounreach W=1.0 | 26/26 | 7,197 | +1.12 | +4 | 16/26 | — |
| plus-uw U=35 W=1.0 | 25/26 | 22,215 | +1.96 | +7 | 18/25 | CC_2_3_4__pl_7 |

(`_results_t60_broken_model/` excluded: different timeout, known-broken model.) Headline reading: **C+w=2.0 dominates on quality** (+0.54 avg, +2 worst, full coverage at 4,435 nodes); **plus W=1.0 dominates on node economy** (2,874 nodes, full coverage) at a real quality cost (+1.73/+6); the val_loss-checkpoint row is the single worst configuration ever measured (248k nodes, 86× the spearman ckpt on the same weights) — the starkest single demonstration that calibration-metric checkpoint selection does not select for planner performance.

### Recommended default flag set (with domain-dependent caveats)

| Flag | Recommended default | Transfer status |
|---|---|---|
| `--ckpt-metric spearman` | **ON** | **Transfers.** On CC it buys the 26th instance (INV-2 attribution: nounreach vs C differ only by ckpt-metric and nounreach solves `CC_2_2_3__pl_7` where C times out); on Grapevine it rescued the only usable export (ep-77, pre-collapse). Caveat: it insures against *late* collapse only — it cannot rescue a run that collapses early (Grapevine plus-default ep-17 export, Spearman 0.10), and it needs the underlying Spearman to be high enough (0.699 on CC-uw was below what the mechanism needs; coverage was lost anyway). |
| W (heuristic weight, first-class) | **W=1.0/1.25 for node economy; W=2.0 for quality. Never W=1.5.** | **Transfers as a mechanism** (post-hoc rescale, no retraining — domain enters only through the model). On the plus frontier W=1.5 is dominated (5,817 nodes for +1.35 vs W=2.0's 5,821 for +1.15); per-instance non-monotonicity is confined to the three known trader instances. Per-model, not per-family: w values are relative to each model's own slope; baseline-w settings must not be reused verbatim on C/plus models. |
| `--dynamic-max-depth` | ON for CC-like domains, **NEEDS A GUARD** in general | **Does not transfer unguarded.** Prime suspect for the Grapevine collapse: 2.4× steeper target scaling puts Grapevine near a collapse basin baseline never enters (plus-nounreach collapses ep 143 with zero unreachables; baseline never). On CC it is a strict win (21,128 → 2,974 nodes at w=1.0). |
| `--include-unreachable` | ON for CC-like domains, **NEEDS A GUARD** in general | **Does not transfer unguarded.** A genuine speed/quality dial on CC (off: 2.5× nodes, +0.6 better quality; on: node-economy sweet spot), but an accelerant on Grapevine (collapse ep 143 → ep 22; 0.77% of samples carrying ≈90% of epoch-1 loss). Guard ideas in the next-round list. |
| `--unreachable-loss-weight` | **1.0, never tune up** | INV-2/INV-3 bracket the dial: U=0 trades nodes for quality, U=1 is the sweet spot, U=35 is toxic in both directions (8× nodes AND worse quality AND lost coverage). Value of the flag is as a measurement instrument. |

### Three-mechanism failure catalogue

All observed training failures across both domains reduce to three mechanisms; each has a distinct signature and a distinct (partly shipped, partly proposed) fix:

1. **Clamp gradient trap** — predictions pinned at the 0.001 clamp floor with *dead gradients*: the clamp's zero-derivative region eats the backward signal, train_loss freezes. Signature: train AND val metrics frozen. **Fix (shipped):** the 220602a gradient fix (straight-through/leaky clamp). Verified still effective: in the Grapevine collapses train_loss kept moving (200 unique values) — the trap itself no longer bites.
2. **Bias attractor** — a mass of constant-target (unreachable-as-max) samples dominates early optimization and pulls the regressor to a constant output before the reachable signal establishes itself. Signature: best val_loss at ~epoch 1, predictions constant, severity scaling with unreachable loss share (CC U=35: 9.5% mass → degraded but trains; Grapevine U=1: ≈90% epoch-1 mass → collapse at ep 22). **Fix (proposed, next round):** error-magnitude-based unreachable guards — soft target at max_seen+margin instead of the scale ceiling, warmup-phase exclusion of unreachables, Huber loss on the unreachable subset. Addressed by next-round item 2.
3. **Saturation collapse** — output saturation at the clamp floor *with live gradients*: the steeper dynamic-MAX_DEPTH scaling (Grapevine slope 0.0475 vs baseline ~0.02) drives pre-clamp outputs to ≈0 mid-training and the run never recovers; entry time is seed-dependent. Signature: train_loss moving, val predictions constant from some epoch (22/143/1 depending on config and seed); never observed under baseline's shallow scaling. **Fix (proposed, next round):** LR warmup — mechanism-matched, because entry into the basin happens when early large steps under a steep target scaling overshoot into saturation. Addressed by next-round item 1.

Mapping: the gradient fix killed mechanism 1; mechanism 2 is unreachable-specific and is what `--include-unreachable`'s guard must address; mechanism 3 is scaling-specific and is what `--dynamic-max-depth`'s guard must address. On CC none of 2/3 ever fired (shallow enough effective scaling, 0.3% unreachable ratio) — which is exactly why single-domain validation missed both.

### Seed fragility

The Grapevine seed lottery (seed 42: 142 healthy epochs and a usable ep-77 export; seed 7: collapse from epoch 1, nothing learned) means *no plus result on a new domain can be trusted from a single seed*. CC results were all single-seed (42) — they are now known to sit in a regime where the collapse basin is not nearby, but that was luck of domain parameters, not robustness. Baseline's cross-seed robustness is also untested (only seed 42 measured). Any future cross-domain claim needs ≥3 seeds per config, and "trains healthily" must be a measured property (collapse-epoch distribution), not an assumption.

### Prioritized next round

1. **LR warmup** (first — mechanism-matched to the saturation collapse, the prime suspect): linear warmup over the first ~5–10 epochs before the cosine/plateau schedule. Directly targets basin entry: every observed collapse (ep 1 / 22 / 143) is an excursion into output saturation that early large steps make likelier under steep scaling. Cheap to implement, testable on Grapevine seed 7 (the known ep-1 collapse) as a one-run falsifier.
2. **Error-magnitude-based unreachable guards** (second — targets the bias attractor): (a) soft unreachable target at `max_seen + margin` rather than the scale ceiling (shrinks the constant-target error mass that dominates epoch-1 loss); (b) exclude unreachables during warmup, include after the reachable regression is established; (c) Huber loss on unreachable samples (bounds their per-sample gradient). Success metric: Grapevine plus-default trains past epoch 143 at seed 42.
3. **Deferred from INV-5:** Grapevine planner evals (BFS / baseline / plus-nounreach-s42-ep77, t600) to quantify what the surviving artifact delivers in-planner.
4. **Multi-seed protocol:** ≥3 seeds per config on any new domain; report collapse-epoch distribution alongside metrics.
5. **Single-digit U probe** (low priority): INV-3 suggests if a usable upper `--unreachable-loss-weight` exists it is at ~1–3% loss mass, with no bracket evidence it beats the default — only worth running if dead-end pruning becomes the binding constraint on a new domain.
