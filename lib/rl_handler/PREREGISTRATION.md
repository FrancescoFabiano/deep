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
