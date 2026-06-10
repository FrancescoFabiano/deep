# RL pipeline — known issues / tech-debt ledger

Living doc, updated each session. Status: [DONE <commit>] · [PENDING] ·
[DECISION] · [NOTE] · [BLOCKING] · [PARTIAL] · [RUNNING] · [INSIGHT].

## Top-line state
The fringe-size question (does F=64 beat F=32 on node economy?) is UNRESOLVED,
and now understood to sit downstream of two blockers: (1) TRAINING INSTABILITY
(primary — F1, CONFIRMED by v2) and (2) DATA THINNESS (batch0 has ~4 binding
instances — #5/#6). `diversity_v2` ran (batch-128 / 5e5 / 3 seeds / binders-in-
train) and did NOT tame the instability: 5/6 runs fail the convergence gate and
the inter-seed range (~180-237) dwarfs any fringe effect (~11). Verdict: "not
resolvable; stabilize the trainer (S) first" — the fringe comparison is blocked,
not answered.

## Findings & experiments
F1. [INSIGHT — ROOT CAUSE] Training is unstable. The v1 val curve thrashed
    non-monotonically (269 -> 790 -> 99) across checkpoints. A fixed converged
    model scores identically across refill seeds [0..4] (spread 0), so this is
    NOT eval noise — it is real policy thrash between checkpoints. Consequence:
    best_by_expansions is a noisy MAX over a thrashing curve, so every prior
    fringe comparison (25-vs-41, 22-vs-15, 17-vs-43) compared two noisy maxes —
    which produced the sign-flips. Instability subsumes the init-variance story.
    The trainer itself (target-sync rate, LR, clamp / Double-DQN dynamics — cf.
    Phase A failure modes) is the gating problem.
F2. [INSIGHT] The deployment fringe is REGIME-DEPENDENT: it binds early (a
    poor/random policy at high epsilon accumulates a reservoir -> fringe fills
    to F) and goes inert late (a converged policy dives narrow -> fringe ~1-11).
    So the fringe-size lever acts during the EARLY exploration phase; the window
    closes at convergence. Corollaries: (a) --eval-refill-seeds>1 is justified
    (refill noise is real early); (b) the converged comparison is regime-matched
    (both fringes inert at the endpoint), so it measures whether wider-beam
    EARLY training produced better FINAL weights.
A.  [PARTIAL] Experiment A (init-controlled SC re-run): prior "F=64 wins"
    (22-vs-15) did NOT replicate — it reversed to F=32->17, F=64->43 under
    controlled init. Resolves "is the prior result robust?" -> NO. Does NOT
    identify the true direction (confounded by F=64 under-training at equal
    frames #3, and now understood as instability F1). train binds=true,
    val binds=false at convergence.
B.  [INSIGHT] Fringe size self-attenuates with policy quality (good policy dives
    narrow). Effect, if any, is training-time, not deployment-time.
C.  [INSIGHT] When the beam never binds, F=32 == F=64 by construction.
v2. [RESOLVED — NULL/BLOCKED] diversity_v2 — batch 128, 5e5 frames,
    --eval-refill-seeds 5, seeds 0/1/2, fringes 32/64, fresh symlinked dir
    (batch0 untouched). TRAIN = 3 binders {SC_10_8__pl_15, SC_9_11__pl_8,
    SC_4_2__pl_7}; VAL = SC_8_10__pl_6 (headroom 154) + 5 lower-headroom. train
    binds=true (IV active). opt_sum=52 bfs_sum=233 on the 6 val instances.
    VERDICT: fringe-size UNRESOLVED — both gates FAILED.
    - CONVERGENCE GATE: 5/6 runs FAIL (best_val @best_frame, final-4 spread):
      s0f32 324@500k sp35 PASS; s0f64 117@200k sp167; s1f32 106@50k sp464;
      s1f64 136@300k sp674; s2f32 87@50k sp1086; s2f64 296@500k sp478. Best-
      early + thrashing tails => best_by_expansions is a noisy max, not a
      converged value.
    - VARIANCE GATE: F=32 [324,106,87] mean 172 range 237; F=64 [117,136,296]
      mean 183 range 179. F-gap |172-183| = 11 << inter-seed range ~180-237
      (effect ~20x smaller than seed noise) => indistinguishable.
    - FALSIFIER H (bigger F -> fewer): REJECTED (F=64 mean 183 not below F=32
      172). A real null.
    Lands on the pre-registered "not resolvable; stabilize the trainer first".

## Correctness & methodology
1.  [DONE f4cd867] Per-fringe model init controlled (manual_seed before each
    _build_model). Proven via init-hash equality + counterfactual.
2.  [PENDING] Broader RNG / reproducibility audit.
3.  [BLOCKING] Equal frame budget under-trains the wider beam (A: F=64 best@10k).
    v2 uses 5e5 + the convergence gate; scale further if a fringe is best-late.
S.  [BLOCKING — CONFIRMED v2] Training stability (the trainer itself) is THE
    blocker. v1 thrashed; v2 (batch 128, 5e5 frames, 3 seeds, binders-in-train)
    STILL thrashed — 5/6 runs fail the convergence gate (tail spreads up to
    1086), inter-seed range ~180-237 >> any fringe effect (~11), converged level
    no better than BFS. No fringe comparison is possible until the trainer is
    stabilized. NEXT STEP before any further fringe runs: target-sync rate, LR,
    reward/Q clamp, Double-DQN dynamics; add a stability metric (e.g. val-curve
    monotonicity / tail variance) as a first-class gate.

## Experimental design / data
4.  [DONE f4cd867] Fringe-occupancy logging (val+train history.json, [occupancy]
    line, plot panel) + BFS-based instance selector. REFINEMENT: BFS frontier is
    an UPPER BOUND — bfs<F => inert; bfs>=F => may bind, confirm with real
    occupancy.
5.  [PENDING] CC val degenerate (bfs_front=2) -> no signal.
6.  [APPLIED v2 / PENDING] >=3 seeds applied; but val discrimination effectively
    rests on ~1-2 high-headroom instances (batch0 ceiling). Needs more
    high-branching instances for cross-instance generalization.
14. [APPLIED v2] Fresh/versioned output dirs (was: experiment A overwrote the
    batch0 baseline). diversity_v2 symlinks training_data; never writes batch0.

## Features added this arc
- `--n-val K` multi-instance validation split [0f64864].
- `--eval-refill-seeds K` averaged validation eval [caa003b] (near no-op once
  converged/inert, but real during early binding — see F2).

## Logging / UX
7.  [PENDING] tqdm off-TTY guard commented out + import sys removed (12d15ea) ->
    bar doesn't auto-disable under the driver pipe. Uncomment + re-add import.
8.  [PENDING] Durable unbuffering (PYTHONUNBUFFERED in run_one + flushes) not
    built; using the env-var prefix.

## Test / CI hygiene
9.  [PENDING] test_offline_encoder::test_onnx_compat tracebacks on the absent
    frontier_policy_32.onnx fixture instead of skipping.

## Model / export quirks
10. [PENDING] Exported ONNX has a fixed scatter dim = training fringe size
    (logits labeled symbolic but a 32-model rejects N>32). No cross-width deploy
    via ONNX, only via .pt.
11. [DECISION] Two # TODOs in _build_model (committed): dataset_type="HASHED",
    use_goal_separate_input=False.
12. [NOTE] --batch-size default 64->512 rode into 12d15ea; experiments since use
    explicit batch (A: 64, v2: 128).

## Housekeeping
13. [PENDING] Stale seed42/last.pt from a pre-last.pt-removal run.

---
Priority order: S + #3 (stabilize/converge training) gate everything; then
#6/#5 (data diversity) gate generalization. The fringe-size verdict is only
trustworthy once v2 clears both the convergence and variance gates.
