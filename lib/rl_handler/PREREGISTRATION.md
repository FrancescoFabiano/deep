# Pre-registration — the mixing-fraction hypothesis

**Written and committed BEFORE the n=29 run. Do not edit after seeing results;
append an OUTCOME section instead.**

## Hypothesis

The fraction of 1-WL frame classes that contain BOTH a viable (`delta < inf`) and
a sterile (`delta = inf`) state predicts whether a topology-only heuristic can
rank on that instance. A class straddling the viability boundary forces any
topology-only model to score a live and a dead subtree identically.

Motivating observation (**n=3, not a result**):

| instance | WL classes | mixing-fraction | oracle[wl] regret |
|---|---|---|---|
| `SC_R_10_10__pl_10` | 329 | 13.1% | 1.00 |
| `CC_2_3_4__pl_7` | 371 | 14.3% | 6.00 |
| `CC_2_2_3__pl_6` | 350 | 28.9% | 123.67 |

Note the WL **class count** does NOT track regret (329/371/350 → 1/6/124). Only
the mixing-fraction does. That is the hypothesis under test.

## Measured quantities (per solvable instance)

- `mixing_fraction` — fraction of 1-WL frame classes (3 iterations, edge labels
  used, node ids IGNORED) containing both a viable and a sterile state.
- `oracle_wl_self` — the **in-sample** topology ceiling: rank each beam by the
  mean `delta` of the instance's own states in the same WL class. This is the best
  ANY topology-only model could do on this instance, given perfect knowledge of the
  WL→delta map. It needs no sibling instance, so it is defined for every instance
  (n≈29) and isolates the representation's resolving power from transfer.
- `random_regret` — the difficulty control (see confound).
- Normalisation: **`regret / delta_root`**. Raw regret is not comparable across
  instances whose `delta_root` spans 2..37; an easy instance's 124 and a hard
  one's 6 are not on the same scale, and that alone could manufacture or hide the
  correlation.

Rollouts: F=8, 3 seeds, cap 800, determinism enforced.

## Acceptance criteria — PRIMARY

Spearman rho between `mixing_fraction` and `oracle_wl_self_regret / delta_root`
across all solvable instances.

- **PASS**: `|rho| > 0.6` AND the sign is positive (more mixing → worse) AND rho
  stays `> 0.6` after dropping the 3 most extreme points (by |residual|).
- **WEAK**: `0.3 < |rho| <= 0.6`, or it drops below 0.6 when the extremes are
  removed → report as a correlation with caveats, NOT a headline.
- **FAIL**: `|rho| <= 0.3`, or the sign flips → the n=3 monotonicity was a lever,
  not a trend. Say so and drop the claim.

## Acceptance criteria — CONFOUND

`mixing_fraction` may be a proxy for instance **triviality**. `CC_2_2_3__pl_6` has
`random` regret 1.33 — everything wins there, so "oracle[wl] fails" might mean
"the frame is useless" OR "the instance is so easy that WL's coarseness finally
bites where it is cheap to".

Control: fit `oracle_wl_self_regret/delta_root ~ mixing_fraction + random_regret/delta_root`
(OLS), and report the **partial** Spearman of `mixing_fraction` controlling for
`random_regret/delta_root`.

- **PASS**: partial `|rho| > 0.4` with the sign preserved.
- **FAIL**: partial `|rho| <= 0.4` → it was triviality wearing a costume. Report
  that, and the hypothesis is dead.

Both PRIMARY and CONFOUND must pass for this to be a headline.

## What this is NOT

Not a claim about cross-configuration transfer (that is settled and negative, see
DESIGN §1.1). Not a claim that the frame is uninformative in general — the SCRich
inversion (`oracle[wl]` 1.00 beats `oracle[id_knn]` 9.00) is a **counterexample**
and stays in the writeup: the discriminating channel differs by domain (valuation
for CC, frame for SC), which is precisely why one fixed encoder struggles and why
the BITMASK gap matters.

## Standing artifact (independent of the above)

Per family, permanently reported alongside the baselines:

    oracle[wl]      the topology ceiling
    oracle[id_knn]  a GENEROUS ceiling for a model keyed on hashed ids (kNN in
                    id-space; a real net does worse)
    dfs / random / bfs / hfs_oracle

The **gap** between `oracle[wl]` and `oracle[id_knn]` says which channel carries
the signal for that family — and therefore whether a model can win before any GPU
is spent. The RL question only bites where `id_knn << wl` (signal in the
valuation) AND the beam binds AND `delta-hat` error is moderate.

---

# OUTCOME (appended after the run; the above is unedited)

**Both criteria FAIL. The hypothesis is dead.** n=22, not 29 — seven instances
(Assemble x3, Grapevine x4) were skipped because their RawFiles no longer exist on
disk (rotated away by the running launcher), not for any statistical reason.

```
PRIMARY   Spearman(mixing, oracle_wl_self/delta_root) = 0.415   (needed >0.6)  FAIL
          after dropping the 3 most extreme points    = 0.308                  FAIL

CONFOUND  Spearman(mixing, random_regret/delta_root)  = 0.601
          PARTIAL (mixing, oracle_wl | difficulty)    = -0.109  (needed >0.4)  FAIL
                                                                sign FLIPPED
```

## Verdict

`mixing_fraction` is a **difficulty proxy**. It correlates 0.60 with `random`'s
normalised regret, and controlling for that leaves a partial rho of -0.11 with the
sign inverted. The n=3 monotonicity (13.1% -> 1.0, 14.3% -> 6.0, 28.9% -> 123.7)
was a lever, not a trend. PRIMARY fails on its own terms even before the confound.

Do not report the mixing-fraction as a predictor. Do not resurrect it without a
new pre-registration and a fixed oracle (below).

## A flaw in the oracle, recorded because it cuts against the measurement

`oracle_wl_self` used `1e6` as the sterile stand-in inside a class's MEAN delta.
One sterile state in a class of ten viable ones gives a mean of ~90909: the
sentinel dominates and every mixed class is ranked last wholesale. That is not the
Bayes-optimal predictor given the representation; it is a broken estimator.

Visible in the data: `CC_2_3_4__pl_7` scores 79.0 in-sample here but 6.0 under the
sibling-trained transfer oracle. An in-sample oracle can never legitimately lose to
a transfer oracle, so the estimator is wrong, not the instance.

This is the SAME mistake as the two-head baseline's score inversion: an unbounded
sterile value swamping a distance term. The correct form is the one that fix used —
viability must strictly dominate distance:

    score(class) = (max_delta + 1) * P(viable | class) - E[delta | class, viable]

What survives the flaw: PRIMARY fails independently of oracle calibration, and
`Spearman(mixing, random_regret) = 0.601` never touches the oracle at all. The
partial correlation does depend on it, so treat -0.109 as suggestive rather than
decisive. The verdict does not rest on it.

## Consequences

- The 1-WL work is NOT a centerpiece. It is a set of measurements
  (DESIGN 1.1.1) whose interpretation did not survive n=22.
- The SCRich inversion (frame 1.00 beats ids 9.00) stands — it is a direct
  measurement, not a correlation, and remains the honest counterexample: the
  discriminating channel differs by domain.
- The transfer results (DESIGN 1.1) stand: 0.0% cross-configuration id overlap is
  a structural fact, not a statistic.
- The per-family pre-flight table (oracle[wl] vs oracle[id_knn] vs baselines)
  remains worth having, but MUST be rebuilt on the corrected oracle before any
  claim rests on it.

---

# Sampler ablation (proportional vs capped) on batch1_1/CC — PRE-REGISTERED LIMITATION

Recorded 2026-07-21, BEFORE any arm was run.

**The deepest fidelity bin contains exactly one instance.** `--fidelity-instances 5`
takes all five fidelity-usable instances in the CC pool, whose plan lengths are
pl=4 (`CC_3_2_3__pl_4`), pl=5 (`CC_2_2_4__pl_5`, `CC_3_2_3__pl_5`), pl=6
(`CC_2_2_3__pl_6`), pl=7 (`CC_2_3_4__pl_7`). The remaining three usable instances
are excluded from planner-side scoring because BFS reaches < 20 expansions.

So the deepest bin is `CC_2_3_4__pl_7` alone — which is **also the pool's dominant
instance (62% of rows at F=8) and the one the cap reduces most**. Any depth-related
verdict from this ablation is therefore CONFOUNDED between "deep" and "this
particular instance", and is provisional. Three seeds give three measurements of one
instance, not three instances: seed replication does not repair this.

**pl=5 is the only bin with within-bin replication** (2 instances). A depth claim
that survives only at pl=7 and not at pl=5 should be read as an instance effect
until a second deep instance exists.

This does not change the pre-registered verdict rule; it bounds what the rule can
conclude.

**The run is a STRUCTURE-ONLY FLOOR.** `--allow-cross-config` is required to train
the CC pool at all: its eight instances span five configurations (`CC_2_2_3`,
`CC_2_2_4`, `CC_2_3_4`, `CC_3_2_3`, `CC_3_3_3`), and on HASHED the node ids are
hashes of the fluent set, so different configurations share **0% vocabulary**. The
model therefore learns from topology and edge labels alone; the node channel
carries no cross-configuration signal.

Both arms inherit this equally, so the RELATIVE comparison the verdict rule rests on
remains valid. The ABSOLUTE numbers are a floor, not the method's performance, and
must not be reported as such.
