# Offline-RL sensitivity study

A list-driven sweep over four training axes for the multi-regime Double-DQN
fringe ranker (`lib/rl_handler`, `--use-regimes`). It pairs the existing
deploy-faithful model-selection eval with a **new, strictly non-selecting**
per-regime diagnostic surface, so we can read how each axis reshapes the model's
fringe ordering on train and held-out problems **without** letting any of that
leak into checkpoint selection.

> **Provisional / underpowered.** The headline wide-gap verdict (train→test) is
> read off **3 seeds** on a **thin split**; treat directions as suggestive, not
> conclusive. Split pinning is still open.

---

## Axes

`offline_main.py` takes all four as lists (length 1 = fixed, length >1 = swept);
the runner expands them into cells. `--sweep-mode` chooses the combine rule:

- **`one_at_a_time`** (default): baseline = the first element of every list; vary
  one axis at a time off that baseline (the de-duplicated union of single-axis
  variations). Cell count is linear in the axis lengths — each non-baseline cell
  isolates one axis' effect.
- **`product`**: full Cartesian product.

`offline_main` always trains **one cell** per invocation (single-cell contract);
expansion lives only in the runner. `--list-cells` emits the resolved manifest
and exits.

### `--gamma` (discount)
- **Controls** the bootstrap horizon **and** the unreachable-node penalty depth:
  the finite worst-value floor is `floor_v = -1/(1-gamma)`, so an unreachable
  (d\*=∞) node's target deepens with gamma (`-1` at γ=0 → `-100` at γ=0.99).
- **Expected direction:** larger gamma = longer effective horizon **and** a
  harsher unreachable penalty; the two move together.
- **Caveats:** a gamma trend **conflates horizon with unreachable-penalty
  depth** — you cannot attribute a change to one without the other. `gamma=0` is
  a no-bootstrap bandit that **cannot** produce the discounted value deployment
  consumes — it is **diagnostic only, not deployable**.

### `--lambda-ord` (order auxiliary weight)
- **Controls** the weight of the pairwise ORDER loss added to the value loss on
  the single head: `L = L_val + lambda_ord · L_ord` over non-padded slots.
- **Expected direction:** `L_ord` is **gamma-invariant** (the sweep moves only
  `L_val` with gamma), so as `lambda → large` the objective approaches the
  **supervised ranker** (interpolation regime).
- **Caveats:** read **jointly with gamma** — λ trades off against the
  gamma-scaled value term, so the same λ means different things at different
  gamma.

### `--fringe-sizes` (beam width F)
- **Controls** two things at once: order-pair supply scales **~F²**, and F sets
  the **padded/faithful partition** — when `F > fmax` (the instance's max live
  frontier) the beam must be padded from closed nodes, thinning the per-regime
  signal on that instance.
- **Expected direction:** larger F = more order pairs but more padded instances.
- **Caveats:** read **per-problem, not aggregate** — F's effect is dominated by
  which instances flip to padded, which the aggregate hides.

### `--target-centering` (`absolute` | `fringe_mean`)
- **Controls** the value arm: `absolute` → `signal_mode=basic`; `fringe_mean` →
  `signal_mode=rank-rl` + `advantage` (per-fringe centred target).
- **This is the level-shift diagnostic.** P0c found the **train→test gap is a
  +3–8 d\* LEVEL shift**; centring removes the level, so `fringe_mean` may
  **extrapolate** the shift away even **without** an order auxiliary.
- **Caveat:** keep `absolute` in the comparison to **show the shift bites** — if
  only centred arms are run, the level shift is silently absorbed and the
  diagnostic loses its point.

---

## The a → b → c arms

The value/order arms, in increasing structure:

- **a — `absolute`, λ=0:** plain Double-DQN value target (`signal_mode=basic`).
  Absolute d\* level is learned; exposes the train→test level shift.
- **b — centred (`fringe_mean`), λ=0:** per-fringe-centred target removes the
  absolute level but **not** the scale (target spread still grows with d\* gaps)
  — a *partial* scale fix.
- **c — centred, λ>0:** centred value **plus** the pairwise order auxiliary on
  the same head; pushes toward the range-free supervised ranker.

---

## Dual eval surface (the critical separation)

- **SELECTION — unchanged.** The deploy-faithful eval (children-first seating +
  heuristic reservoir refill) on **val** is the **sole** checkpoint-selection
  signal (`best_by_expansions`, `best_by_spearman`). Nothing in the diagnostic
  surface feeds it.
- **DIAGNOSTIC — new, never selects.** At each checkpoint, for the **train** set
  and (if `--test-csv` is given) the **test** set, for **each regime** in
  `{dfs, bfs, hfs_m0, hfs_m1, random}`, the *current* model is rolled out greedily
  over that regime's full-beam-redraw fringes. Per `(regime × problem)` we record
  raw: node economy (expansions-to-goal), goal rate, the −1-stream return, the
  order usable-pair fraction + post-mask skip fraction, realized pad-fill, and
  the padded/faithful flag.
  - Stored under `history.json["diag_per_regime"][split]`, explicitly labelled
    non-selecting, computed **after** the selection logic each checkpoint so it
    structurally cannot influence it.
  - **Test envs are regime-shaped → off-distribution vs deployment** (deployment
    seats children first; the regimes do a full-beam redraw). They measure
    ordering quality on held-out problems, **not** deployable performance.

---

## Reporting & the normalized-aggregate plot convention

- **Per-problem tables** (`diag_per_regime_{train,test}.csv`): raw metrics, one
  row per `problem × regime × checkpoint`. Problems are kept **separated** in
  storage; aggregation happens only at plot time.
- **Plots** (`diag_per_regime_{split}_fringe<F>.png`, matplotlib → PNG, no
  display): one lineplot per split, **one line per regime** = the aggregate over
  problems, **per-problem-normalized then meaned** so deep problems don't
  dominate:
  - **node economy** = `expansions / optimal-expansions` per problem (optimal =
    shallowest-goal depth from the tree), averaged across problems;
  - **reward** = `return / optimal-expansions` per problem, averaged likewise.

  Raw stays in the table; the plotted aggregate is the normalized ratio.

---

## Runner + collector

- `scripts/sweeps/run_sensitivity.py` — reads the axis lists + `--sweep-mode`,
  expands to **cells × seeds**, builds one `offline_main` CLI per `(cell, seed)`
  with a distinct output dir tagged by axis coords, and writes
  `sweep_manifest.json`. **Safe by default:** prints the plan and exits unless
  `--execute` is passed; launches are bounded by `--max-parallel`. It never
  launches the full set on its own.
- `scripts/sweeps/collect_sensitivity.py` — walks the run dirs and emits
  (i) `collected_diag.csv`, a tidy `cell × seed × problem × regime × frame`
  table of the diagnostic metrics, and (ii) `axis_summary.csv`, the per-axis
  **median/IQR of the SELECTION metric** (deploy-faithful val expansions). No
  plotting (Part 3 plots are per-run).

---

## How to run (dry run first)

```bash
# 1. inspect the cell set for either mode (no training)
python lib/rl_handler/offline_main.py --list-cells \
    --gamma 0.0 0.9 0.99 --lambda-ord 0.0 1.0 \
    --target-centering absolute fringe_mean --fringe-sizes 32 64 \
    --sweep-mode one_at_a_time --dir-save-model /tmp/sa --use-regimes

# 2. plan the sweep (prints commands; nothing launches without --execute)
python scripts/sweeps/run_sensitivity.py \
    --gamma 0.0 0.9 0.99 --lambda-ord 0.0 1.0 \
    --target-centering absolute fringe_mean --fringe-sizes 32 64 \
    --sweep-mode one_at_a_time --seeds 0 1 2 \
    --train-csv <a.csv> <b.csv> --val-csv <c.csv> --test-csv <d.csv> \
    --out-root exp/rl_exp/sensitivity/run1 --frames 30000

# 3. (when ready) add --execute --max-parallel N to launch, then collect:
python scripts/sweeps/collect_sensitivity.py \
    --out-root exp/rl_exp/sensitivity/run1
```

---

## PRE-REGISTRATION — CC binding-pool regime study (2026-06-29)

Recorded **before** the 18-run batch is interpreted. The falsifiers and scope
below are fixed at write time; the Results block at the end is sealed PENDING and
filled only after the runs finish — the hypotheses above it are not edited
post-hoc.

### AMENDMENT 1 — frame budget 100k → 40k (2026-06-29, batch in progress, NO results seen)
Committed as its own commit **before** STEP 5 collected anything, with the 18 runs
already launched but no `collected_*.csv` written and 0 runs finished — so this
amendment's timestamp honestly precedes all 40k data.

- **Change:** `--frames 100000` (as originally pre-registered) → `--frames 40000`,
  `--n-checkpoints 20` (2k-frame resolution, unchanged from prior runs, so the
  early peak is still resolved).
- **Why, stated honestly:** the cut was **informed by prior results** — earlier
  CC runs showed val metrics (expansions minimum ~frame 4000, Spearman peak
  ~6000) **peaking ~4–6k and degrading thereafter**, so 100k was largely spent
  overfitting past the binding checkpoint. 40k retains margin past the peak to
  confirm the degradation is real without the wasted tail. This is a
  results-informed budget choice, recorded here rather than applied silently.
- **Added diagnostic (also pre-results):** a per-regime **rank-fidelity** figure —
  tie-aware Spearman(model slot score, −d\*) over order-eligible slots (non-pad,
  finite d\*), two panels (train | eval) × five regime lines, structural_nonresult
  excluded — plus a companion expansions (`node_economy_ratio`) figure. These are
  non-selecting diagnostics; they do not change selection, the ONNX contract, or
  the dqn loss math.
- **Expected reading, fixed now so it is not re-interpreted later:** the 1500-frame
  smoke already showed rank_spearman **identical across regimes and slightly
  negative** — the same early-training tie seen at every scale. The realistic 40k
  outcome is five low (ρ well under 0.2), tangled lines, possibly weak-positive
  then declining. **Tangled-within-seed-noise = the 4-binder underpower verdict,
  NOT "regimes don't matter."** **Weak-positive-then-declining ρ = the overfitting
  signature, NOT a training bug.** These readings are set before the numbers.

### AMENDMENT 2 — corrected split: val/test carved from held-out test_data (2026-06-30, batch killed, NO results collected)
Committed as its own doc-only commit **after killing the prior 40k batch and
before any corrected-split data exists** (the prior run's outputs + its
`collected_*.csv` were moved aside to `_sensitivity_cc_wrongsplit_40k/`; the
`sensitivity_cc/` working dir holds no `collected_*.csv`), so this amendment
honestly precedes all corrected-split results.

- **The methodological error being fixed:** the original split (AMENDMENT-0 table
  above) drew **val and test from the same 4-binder pool** and put binders on the
  train side too — so val was not genuinely out-of-train, and pulling val/test out
  of the pool shrank the training set. Selection on a same-pool val cannot measure
  generalization honestly.
- **Corrected split (fmax-keyed, val/test genuinely held out):**
  - **TRAIN = ALL `training_data` instances, kept whole** (no held-out val carved
    from train): CC_2_2_3__pl_4, CC_2_2_3__pl_6, CC_2_2_4__pl_5, CC_2_3_4__pl_3,
    CC_2_3_4__pl_7. (Binding signal comes from the one binder pl_7/fmax=442; the
    four sub-F instances still contribute value supervision.)
  - **VAL = one held-out binder from `test_data`:** CC_2_2_4__pl_6 (fmax=242) —
    the selection set, now genuinely out-of-train.
  - **TEST (diagnostic, never selects) = the remaining `test_data` binders:**
    CC_2_2_4__pl_7 (268), CC_2_3_4__pl_6 (120). **Families mixed on val/test**
    (CC_2_2_4 and CC_2_3_4 both represented) — no pure-family holdout.
  - `test_data` non-binders excluded: CC_2_2_3__pl_7 (fmax 26), pl_8 (fmax 2).
  - All eight instances confirmed to carry `goal_tree.dot` + a non-empty `Goal`
    column (separated mode) before launch.
- **NO-VAL FALLBACK (recorded; not exercised by this batch, which HAS a val):**
  if no held-out instance exists at all, model selection falls back to **TRAIN
  deploy-faithful performance** (`selection_on_train=true` in the checkpoint
  summary). This is **weaker** — it selects on the fit set and is blind to
  generalization — and is wired only so the no-test-data case runs instead of
  selecting on a degenerate empty-val metric.
- **Unchanged from earlier (reaffirmed):** the a→b→c arms and falsifiers (F1: b≤a
  within 3-seed IQR ⇒ centring inert; F2: c≤b ⇒ order-aux inert); tangled
  rank-fidelity lines = 4-binder underpower, NOT "regimes don't matter";
  weak-positive-then-declining ρ = overfitting; the 40k budget from AMENDMENT 1;
  deploy = val-argmin checkpoint, falsifier reads the SELECTED checkpoint not last.

### Why this run exists (the prior failure)
The earlier CC run was **structurally inert on the train side**: auto-split
(`train_models.py`) held out the only training-side binder (`CC_2_3_4__pl_7`,
fmax=442) as val, leaving four sub-F training instances. The occupancy panels
that "hit F" were the **val rollout** of that one bushy instance, not "CC binds."
Every training fringe was sub-F, so the regime treatment could not fire on a
single training step. This run fixes that by **keying the split on per-instance
`fmax = bfs_frontier_max`, not on domain.**

### Split (fmax-keyed; explicit `--train/--val/--test-csv`, NOT auto-split)
Only **4 CC instances bind** at F (fmax ≥ F at F=32; all four also ≥ 64):

| role | instance | fmax | n_states | family |
|---|---|---:|---:|---|
| TRAIN | CC_2_3_4__pl_6 | 120 | 10,072 | CC_2_3_4 |
| TRAIN | CC_2_2_4__pl_7 | 268 | 4,823 | CC_2_2_4 |
| VAL (selection; deploy-faithful) | CC_2_2_4__pl_6 | 242 | 10,099 | CC_2_2_4 |
| TEST (diagnostic only; never selects) | CC_2_3_4__pl_7 | 442 | 2,897 | CC_2_3_4 |

In-distribution split: both families present on train; binders on **both** sides.
Files live under `exp/rl_exp/batch0_train/_models/CC/{training_data,test_data}/`
(dir name is irrelevant — the path is what matters); all four carry
`goal_tree.dot` + a fully-populated `Goal` column (separated mode).

### CONFOUND (recorded up front)
In **this** pool, fmax is **inversely correlated with tree size**: the four
binders are the *small* trees (2.9k–10k states); the sub-F instances are the
~30k-state deep-narrow ones. So any regime / extrapolation effect measured here
is **confounded with tree size** — the defensible claim is *"regime effect on
small-bushy CC trees,"* **not** a general one. fmax tracks branching family
(`CC_2_3_4`=442, `CC_2_2_4`=268; `CC_2_2_3` never binds, max fmax 26), not depth.

### ARMS + FALSIFIERS (stated BEFORE reading results)
Three arms, the a→b→c decomposition (gamma=0.99 fixed, F∈{32,64}, seeds {42,43,44};
3×2×3 = 18 runs, `--frames 100000`, all `--use-regimes` separated):

- **a** — `--target-centering absolute   --lambda-ord 0.0` (plain value)
- **b** — `--target-centering fringe_mean --lambda-ord 0.0` (centred value)
- **c** — `--target-centering fringe_mean --lambda-ord 0.5` (centred + order aux)

**Falsifiers** (selection metric = deploy-faithful val `total_expansions`, lower
is better; "within IQR" = the 3-seed spread ≈ range, so direction-only):
- **F1:** if **b ≤ a within 3-seed IQR**, centring (absolute-level removal) is
  **inert** on this pool.
- **F2:** if **c ≤ b within 3-seed IQR**, the order auxiliary
  (scale-invariance) **adds nothing** beyond centring.

A real effect requires the inequality to clear the 3-seed spread; with n=3 we read
**direction only**, never fine IQR structure.

### Selection facts pinned in advance
- **"More frames ≠ better."** Deployed = the **val-argmin** checkpoint
  (`best_by_expansions.pt`), **not** `last.pt`. Prior CC runs degraded past the
  binding frame (val-expansions minimum ~frame 4000; Spearman peak ~6000; both
  worse by 10000). On a proper binding split the val-argmin should now land on or
  after the binding frame rather than a pre-bind one — itself a recorded check.
- **The falsifier reads the SELECTED checkpoint, not the last one.**

### SCOPE (and what is DEFERRED)
- This is the **in-distribution REGIME study**, adjudicable on 4 binders. It is
  **NOT** the extrapolation falsifier — that needs a larger binding pool (more
  bushy-CC generation, gated on system inode reclaim) and is **out of scope here**.
- **DEFERRED** for the same small-pool reason: gamma sweeps, lambda-ord sweeps,
  and fringe-size as an axis. This run fixes gamma=0.99 and treats F as a
  confirm-at-64, not a swept axis.

### Smoke gate (passed 2026-06-29, before launch)
One arm-c F=32 seed-42 run, `--frames 1500`: reached checkpoints + exited SUCCESS
(no `order_loss`/KeyError); `history.json["diag_per_regime"]` populated for train
**and** test; rows carry `fmax`; the tie-classifier returned **real_convergence
×9, structural_nonresult = 0** (all binders, fmax ≥ F); `best_by_expansions.pt`
written. Gate green → 18-run batch launched to `exp/rl_exp/sensitivity_cc/`.

### Results (P3) — SEALED PENDING
> To be filled after the 18-run batch (`exp/rl_exp/sensitivity_cc/`) completes,
> via `scripts/sweeps/collect_sensitivity.py --out-root exp/rl_exp/sensitivity_cc`.
> Will record: (1) SELECTION table — per (arm, F, seed) `best_by_expansions` val
> metric **with the frame it was saved at** (not last.pt) + best Spearman;
> (2) per-arm median[IQR] across seeds — the a→b→c readout vs **F1/F2** above;
> (3) DIAGNOSTIC — per-regime node-economy ratio on train+test with
> `structural_nonresult` excluded (expected ~0 here) + tie-class counts.
> The hypotheses above this line are **not** edited when results land.
