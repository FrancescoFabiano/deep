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
