# RL pipeline — known issues / tech-debt ledger

Living doc, updated each session. Status: [DONE <commit>] · [PENDING] ·
[DECISION] · [NOTE] · [BLOCKING] · [PARTIAL] · [RUNNING] · [INSIGHT].

## Top-line state
The fringe-size question (does F=64 beat F=32 on node economy?) is UNRESOLVED,
and now understood to sit downstream of two blockers: (1) the OBJECTIVE
UNDER-DETERMINES THE WITHIN-FRINGE RANKING (primary — S, root-caused; the
optimizer is healthy, the order is just never pinned by a signal-free reward)
and (2) DATA THINNESS (batch0 has ~4 binding instances — #5/#6). `diversity_v2`
(batch-128 / 5e5 / 3 seeds / binders-in-train) did NOT resolve it: 5/6 runs fail
the convergence gate and the inter-seed range (~180-237) dwarfs any fringe
effect (~11) — because three seeds converge to near-independent RANKERS
(tau~0.13) despite identical value-convergence. Verdict: "not resolvable until
the objective injects within-fringe ranking signal (S)" — blocked, not answered.
Next: a supervision-availability audit decides the loss family, then a rank head.

## Findings & experiments
F1. [INSIGHT — SUBSUMED BY S] The val curve thrashes (v1: 269 -> 790 -> 99).
    A fixed converged model scores identically across refill seeds [0..4]
    (spread 0), so this is NOT eval noise — it is real policy thrash between
    checkpoints. Consequence: best_by_expansions is a noisy MAX over a thrashing
    curve, so every prior fringe comparison (25-vs-41, 22-vs-15, 17-vs-43)
    compared two noisy maxes — which produced the sign-flips. NOTE: the original
    read ("the trainer/optimizer is unstable") was WRONG — Path B exonerated the
    optimizer; the thrash is the policy RANKING changing between checkpoints
    because the objective never pins it (see S, now the confirmed root cause).
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
S.  [ROOT CAUSE — CONFIRMED] The OBJECTIVE UNDER-DETERMINES THE WITHIN-FRINGE
    RANKING. (Was mis-framed as "trainer/optimizer instability"; the optimizer
    is exonerated.) v1/v2 val thrashed (5/6 fail convergence, inter-seed range
    ~180-237 >> fringe effect ~11, converged level no better than BFS) NOT
    because the learner diverges but because the within-fringe ORDER is never
    pinned. The -1/step reward is constant across fringe picks => zero ordering
    signal; the only other channel (the bootstrap) is too weak on thin offline
    data to fix the argmax. Lever is the OBJECTIVE/SIGNAL, NOT target-sync / LR /
    clip.
    OPTIMIZER HEALTHY ON EVERY AXIS (Path B, RL_LOG_STABILITY): q_max ~13
    BOUNDED (no monotone growth => not the deadly-triad/explosion mode), q_mean
    smooth & seed-invariant, td_loss -> ~0, none of the 3 Phase-A divergence
    modes fits. grad_norm exceeds the 1.0 clip 70-77% of checkpoints => a
    SECONDARY knob (LR/clip interplay), not the cause.
    EVIDENCE: (i) within-seed dissociation — seed0_f32 had td_loss & q_mean flat
    over 250k->500k while val went 1993->324 (value loss converged, ranking not
    pinned); (ii) cross-seed probe [0f23c59] — 3 seeds at identical value-
    convergence give near-independent rankers (Kendall tau ~0.13, top-1 argmax
    agreement 52-66%, disagreement broad and growing with fringe width).
    THREE NOTES:
    - q_max is bounded but POSITIVE (~+12) in an MDP whose return <= 0 (rewards
      in {-1, 0}; goal reward verified = 0.0 in tree_env reset/step) => bounded
      OVERESTIMATION, not a reward-sign bug; harmless except insofar as it
      scrambles order. Watch it, don't chase it.
    - target_online_l2 logging is phase-locked to the 1000-frame hard sync
      (checkpoint frames are multiples of target_sync => probe lands right after
      a sync => always 0). To be useful, sample off-sync
      (frame % target_sync == target_sync//2). Low priority (optimizer cleared).
    - probe 0f23c59 compared DIFFERENT-FRAME best checkpoints, so stage drift is
      conflated with seed (agreement falls with frame-gap). The clean stage-
      matched comparison needs saved final/periodic weights (last.pt was dropped
      in 12d15ea) — fold into the next training run.
    NEXT (separate prompt, gated on the supervision audit): inject within-fringe
    ranking signal that BYPASSES the bootstrap — likely a listwise softmax-CE
    rank head (correct = on-path / argmin-d*) sharing the GINE trunk ALONGSIDE
    the Q-head; or supervised -d* regression if d* is fully available. NOT a
    DQN-knob sweep.

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

## Generalization (held-out SC_Mix test eval)
G.  [INSIGHT — FIRST GENUINELY HELD-OUT RESULT] On the SC_Mix generalization run
    (--signal-mode sweep basic/pbrs/exact-return/aux, held-out Test/ instances),
    the generalization frontier is DEPTH and it is SIGNAL-INDEPENDENT. Five sub-
    findings:
    - DEPTH RANKING COLLAPSES SIGNAL-INDEPENDENTLY. On the depth sub-group
      (high-pl SC_Multi), rank-accuracy vs d* is 0.08-0.13 for ALL FOUR signals
      (basic/pbrs/exact/aux) — i.e. ~1/N, near-random. No signal's within-fringe
      ordering transfers to deeper fringes than training saw.
    - RICHNESS TRANSFERS FOR ALL (rank-acc 0.55-0.76; aux best at 0.76). So the
      collapse is specific to depth, not a global OOD failure.
    - THE NODE-ECONOMY ORDERING IS A SECONDARY OVERCONFIDENCE EFFECT on top of
      uniformly-broken depth ranking. exact-return's sharp-but-WRONG scores
      mislead the deep search catastrophically — it is NOT better OOD ranking.
      In-dist <-> OOD FLIP: exact-return best in-dist (108 expansions) / worst
      OOD (231); basic least-bad OOD. The d*-signals OVERFIT.
    - MECHANISM (well-supported hypothesis): depth needs OUT-OF-RANGE d* targets
      (train d* <= ~10, test ~17); absolute-value learners cannot extrapolate a
      range they never saw. Richness keeps targets IN-RANGE while presenting
      novel features the hashed encoder absorbs -> transfers. So the
      generalization wall is the TARGET RANGE, not the input representation.
    - CAVEATS: cross-seed IQR unreliable at n=3 (basic swung 388->9 between runs)
      — do NOT lean on variance; coverage saturated ~1.0 (we measure efficiency,
      not solvability); refill ~moot on SC_Mix; PBRS middling, not a standout.
    STRATEGIC: the lever is NOT more signal-tweaking (all four hit the same depth
    wall). The next test is HORN B — a SCALE-INVARIANT / RANK objective:
    deployment uses only ORDER, which is range-free, so an order-based objective
    could extrapolate depth where absolute-value (d*-regression / TD) ones can't.
    Pursued on branch `rank-objective` (rank-sup {pairwise,listwise}, rank-rl
    {reward,advantage}); read primarily on DEPTH rank-accuracy vs the 0.08-0.13
    wall.
G-PREREG. [PRE-REGISTERED — payoff sweep, written BEFORE launch] One harness
    run: methods = {basic, exact-return, rank-sup-pairwise, rank-sup-listwise,
    rank-rl-reward, rank-rl-advantage}; 3 seeds; 100k frames; F=32; batch 128;
    cuda. basic/exact-return re-anchored under the same per-(method,seed) init
    control (also a reproduction check of the prior SC_Mix TEST depth rank-acc:
    basic ~0.13, exact ~0.08). Held-out SC_Mix TEST eval, measure-only on last.pt.
    PRIOR TEST depth rank-acc (reproduction targets, heuristic/random refill):
    basic 0.129/0.116, pbrs 0.113/0.116, exact 0.084/0.071, aux 0.089/0.084;
    rich: basic 0.642, exact 0.626, aux 0.757 (transfer baseline to preserve).
    - H1 (Horn B works): >=1 rank method lifts TEST DEPTH rank-acc to > 0.25 IQM
      across seeds, clearly separated from basic's ~0.13. FALSIFIER: all four stay
      within seed-noise of 0.13 -> order does not transfer across depth -> Horn B
      REJECTED, lever forced to Horn A (curriculum/bootstrap).
    - H2 (wall is SCALE not LEVEL): the fully range-free variants {rank-sup-
      pairwise, rank-sup-listwise, rank-rl-reward} extrapolate depth better than
      the partial {rank-rl-advantage} (level-removed, scale-retained). reward >>
      advantage -> wall is target scale/range; advantage ~= reward -> level-
      removal sufficed; all fail -> deeper than scale.
    - EXPECTED, NOT a failure: rank methods likely WORSE on in-dist node-economy
      (they optimise order, not the -1/step economy). The test is OOD depth rank-
      acc, not in-dist economy.
    - DEGENERACY GATE: advantage dry-run logit-std was 0.015 (near-flat). Harness
      now reports per-(method,group) TEST logit-std; if advantage stays near-
      degenerate on TEST, its depth number is UNINTERPRETABLE (cannot separate
      "scale wall" from "collapsed logits"), NOT a clean H2 readout.
    - STANDING CAVEATS: n=3 -> IQR~=range (don't lean on variance); 100k may
      under-train the rank methods (if loss/rank-acc still moving -> 200k follow-up
      on survivors); EXPLORATORY signal comparison; TEST lacks its rich x deep
      corner (depth=base-repr, rich=low-pl, no rich-and-deep instance).
G-PREREG-AMEND (recorded before run): ~70 min/run x 18 ~= 21h -> staged. Run a
    1-seed ranks-first SCREEN (~7h); the registered 3-seed IQM H1/H2 test runs
    only on methods clearing the wall at seed 0. Asymmetric stop: seed-0 positive
    -> full test; seed-0 null does NOT reject Horn B -> confirm with >=1 more seed
    first (ledger S cross-seed divergence). Watch: the rank objectives may be MORE
    seed-stable than basic precisely because they inject the explicit ordering
    signal -1/step lacked (the S root cause) -> divergence risk smaller for these
    methods; readable in whether seed-0 depth rank-acc sits where expected vs
    scatters. Screen order: rank-sup-pairwise/listwise, rank-rl-reward/advantage,
    then basic + exact-return (reproduction anchors).
G-SCREEN. [RESULT — 1-seed ranks-first SCREEN, seed0, 100k, F=32, batch128, cuda]
    HELD-OUT TEST depth rank-acc (heuristic/random refill), wall = ~0.13:
      rank-sup-pairwise  0.444 / 0.444   (rich 0.897, logit_std ~24, cov 6/6)
      rank-sup-listwise  0.206 / 0.204   (rich 0.838, logit_std 0.25, cov 6/6)
      basic              0.137 / 0.115   (rich 0.470)            [reproduces ~0.13]
      rank-rl-advantage  0.076 / 0.032   (rich 0.690, logit_std 0.9-2.7, cov 5/6,4/6)
      rank-rl-reward     0.067 / 0.066   (rich 0.388, logit_std ~5)
      exact-return       0.061 / 0.042   (rich 0.867)            [reproduces floor]
    H1 (Horn B works): CONFIRMED at seed0 via the SUPERVISED path. rank-sup-
      pairwise 0.444 clears the 0.25 threshold and is 3.4x the wall — the FIRST
      objective to transfer within-fringe ORDER across depth (every absolute-value
      signal sat at 0.08-0.13). rank-sup-listwise 0.206 lifts above the wall but
      below 0.25 (borderline).
    H2 (range-free >> advantage): REJECTED AS FRAMED. The fully range-free reward
      (0.067) did NOT beat the partial advantage (0.076) — both bootstrap variants
      sit AT THE FLOOR. The decisive axis is NOT scale-vs-level but BOOTSTRAP vs
      SUPERVISED: keeping the Double-DQN bootstrap fails depth even with a range-
      free reward, because the gamma-discounted value it bootstraps still
      ACCUMULATES WITH DEPTH and reintroduces the out-of-range target. Only the
      NO-BOOTSTRAP supervised order objectives transfer — exactly the channel S
      flagged as the one that never pinned within-fringe order. Within supervised,
      the pairwise margin loss >> the listwise soft-CE.
    DEGENERACY GATE: NOT triggered — advantage TEST logit_std 0.9-2.7 (the 0.015
      was only the 600-frame dry run); its floor depth-acc is a REAL, interpretable
      failure, not collapsed logits. advantage is also the WORST method: it loses
      coverage on depth (0.83/0.67 heur/rand) — fails to solve some instances.
    BONUS (not a failure as feared): rank-sup-pairwise is ALSO the most node-
      ECONOMICAL on TEST depth (43.8 expansions vs basic 135, exact 456, advantage
      743) AND best in-dist (val regret 0.456 ~= exact 0.452 best; val econ 81 best).
      The "rank methods worse on economy" caveat did NOT bite pairwise — a better
      OOD ranker is also cheaper. Reproduction: basic depth 0.137~=prior 0.129;
      exact depth 0.061 vs prior 0.084 (single-seed scatter, same floor verdict).
    PROMOTION (asymmetric rule, G-PREREG-AMEND): rank-sup-pairwise -> full 3-seed
      registered H1 test (clear seed0 positive). rank-sup-listwise -> promote with
      >=1 confirming seed (above wall, sub-0.25). rank-rl-{reward,advantage} are
      seed0 nulls (non-degenerate -> real); the "bootstrap fails depth" claim
      should be confirmed with >=1 more seed before it is called definitive.
    CAVEATS: single seed (screen, not the registered test); 100k (rank methods may
      still be under-trained -> 200k follow-up on survivors); EXPLORATORY; TEST
      lacks the rich x deep corner.
G-3SEED. [RESULT — registered 3-seed H1 test (seeds 0/1/2, 100k, F=32, batch128)]
    Promoted survivors pairwise + listwise + basic anchor (basic seeds1/2 reused
    from the baseline via --skip-train; valid under deterministic per-(method,seed)
    init, byte-identical basic path). HELD-OUT TEST depth rank-acc (IQM, IQR=range
    at n=3), wall ~0.12:
      rank-sup-pairwise  0.426  (IQR 0.126; seeds 0.444/0.480/0.354; econ 44.5)
      rank-sup-listwise  0.322  (IQR 0.339; HIGHLY seed-variable; econ 76.2)
      basic              0.124  (IQR 0.030; seeds 0.137/0.106/0.129; econ 149.2)
    H1 CONFIRMED at the registered level for pairwise: depth IQM 0.426 clears 0.25,
    every seed clears 0.25, and ZERO overlap with basic (pairwise min 0.354 > basic
    max 0.137). pairwise is also the most node-ECONOMICAL on depth (44.5 vs basic
    149.2), robustly. listwise IQM 0.322 also clears 0.25 but IQR 0.339 (~range) =>
    a NOISY secondary (one seed near wall, one ~0.5+), not robust like pairwise.
    SEED STABILITY (the S watch-point): pairwise DEPTH never collapses to the wall
    (0.354-0.480); basic pinned at floor (~0.10-0.14). The ordering signal that
    -1/step lacked (S root cause) makes the SURVIVING axis (depth order) seed-
    stable, while basic-as-ranker stays at the floor. RICH is indistinct at 3 seeds
    (pairwise 0.598 / listwise 0.572 / basic 0.588, all high IQR) -> depth is the
    discriminating axis, rich is seed-noise.
G-PARTA. [RESULT — depth-extrapolation decay probe, seed0 eval-only, ~2 inst/bin]
    TEST depth rank-acc by pl-bin (pairwise / basic), wall ~0.13:
      pl 11-12: 0.541 / 0.171   pl 13-14: 0.255 / 0.104   pl 15-17: 0.642 / 0.140
    pairwise does NOT decay toward the wall as pl grows; the DEEPEST bin (15-17,
    furthest from train pl<=10) is its BEST (rank-acc 0.642, econ 27.5). The 13-14
    dip (0.255) is non-monotonic -> 2-inst/bin noise, not a slope. Favors order
    being DEPTH-INVARIANT (true extrapolation) over adjacent-only interpolation —
    a noisy HINT, not a verdict. Part B (train pl<=6, gap pl7-10 held out entirely,
    test pl11-17) is the real test; pl<=6 pool=10 supports the widest gap (train 8
    + val 2). Plot: sensitive_analysis/partA_depth_decay.png.
G-PARTB. [RESULT — wider-gap extrapolation test, 1-seed, 100k, dir SC_Mix_gap]
    Train = pl<=6 (8 inst + 2 val), pl 7-10 HELD OUT ENTIRELY, test = existing
    depth (pl 11-17) + rich. Held-out depth rank-acc (wall ~0.12-0.20 here):
      rank-sup-pairwise 0.198 (econ 87.8, logit_std 25.1, non-degen)
      basic             0.203 (econ 83.0)
      rank-sup-listwise 0.175 (econ 96.0)
    VERDICT: INTERPOLATION, not true extrapolation. The pairwise depth ADVANTAGE
    over basic COLLAPSES from +0.30 with full data (0.426 vs 0.124, train pl<=10)
    to ~0 under the gap (0.198 vs 0.203, train pl<=6). Removing the adjacent
    pl 7-10 kills the transfer -> seed-0/3-seed 0.426 was largely INTERPOLATION to
    the adjacent pl 9-10, NOT depth-invariant extrapolation. The range-free order
    loss buys transfer to ADJACENT unseen depth, not unbounded extrapolation —
    consistent with the optimal within-fringe ORDER itself shifting with depth in
    ways a shallow-trained model never sees. Part A's "deepest bin best / no decay"
    was WITHIN the small full-data gap and is OVERTURNED by this controlled gap.
    RICH is preserved (pairwise 0.632 / listwise 0.628 > basic 0.529) because rich
    test pl 2-7 overlaps the pl<=6 train range (no gap for rich).
    CAVEATS: 1-seed gap SCREEN; basic's gap depth 0.203 is single-seed and ABOVE
    its full-data 3-seed IQM 0.124 (shallow-only training may be less depth-
    overconfident, or seed-luck) -> the robust claim is the DIFFERENTIAL collapse
    (pairwise no longer beats basic), not the absolute levels. Pushing depth past
    this data floor needs the planner-coverage tier (deeper d*-labelled instances),
    out of scope. STRATEGIC: Horn B gives ADJACENT-depth order transfer (real, ~3x
    the wall in-range) but not far extrapolation; closing the true depth gap is a
    DATA/curriculum problem (Horn A), not solvable by the objective alone.

---
Priority order: S + #3 (stabilize/converge training) gate everything; then
#6/#5 (data diversity) gate generalization. The fringe-size verdict is only
trustworthy once v2 clears both the convergence and variance gates. G reframes
the open frontier as DEPTH/target-range and motivates the rank-objective arc.
