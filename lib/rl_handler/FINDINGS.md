# FINDINGS — `Final-Reinforcement-Learning`

What we learned and what is load-bearing. (`DESIGN.md` is how it works;
`PREREGISTRATION.md` is the one hypothesis we tested formally.)

Every number here is traceable to a commit and, where applicable, to
`preflight_table.json`.

---

## 0. CORRECTION BLOCK — claims measured on subsampled trees, SUPERSEDED

**Read this before citing any number below.** The shipped generation tables were
produced with `--dataset_discard_factor 0.4`. `TrainingDataset.tpp:743-756`
randomly discards states, **writes them to the CSV anyway**, and never expands
their subtrees:

```cpp
if (m_dis(m_gen) < discard_probability) {
    add_to_dataset(this_state_filename, depth, current_score, predecessor, action);
    return current_score;                    // subtree NEVER expanded
}
```

So a discarded state appears as a **childless leaf**, indistinguishable from a
genuine dead end. And the discard is **BIASED, not uniform**
(`TrainingDataset.tpp:714-730`): the probability rises with depth, and gains
`+0.2` **immediately after a goal is found**. It therefore preferentially deletes
the *shallow goals that make an instance easy*. The offline env was not modelling
a noisier version of the planner's problem — it was modelling a **harder problem
with the easy solutions cut out**.

Measured, `CC_2_2_3__pl_4` (planner BFS: true optimal plan length **4**, 17 expansions):

| generation | n states | delta_root | h*_root | sterile |
|---|---|---|---|---|
| `discard_factor 0.4` (shipped) | 2,759 | **10** | 4 | **30.0%** |
| `discard_factor 0` (regenerated) | 50,146 | **4** ✓ | 4 | **2.9%** |

And across instances, reconstructed vs the planner's true optimal:

| instance | planner BFS optimal | `delta_root` | `h*_root` |
|---|---|---|---|
| CC_2_2_3__pl_4 | 4 | 10 | 4 |
| CC_2_2_3__pl_6 | 6 | 12 | 10 |
| CC_2_3_4__pl_7 | **7** | **34** | **29** |
| CC_3_2_3__pl_5 | 5 | 18 | 17 |

### What this retracts

- **"34.8% of `CC_2_3_4__pl_7`'s nodes are sterile"** — SUPERSEDED. Mostly a
  discard artifact. The faithful figure on `pl_4` is **2.9%**, not 30%. The entire
  sterile population was inflated by discarded states written as childless leaves.
- **The eviction / compounding-loop MAGNITUDES** (bfs 60% sterile expansions,
  |R| growth to 74, 43.9 evictions) — SUPERSEDED; measured on the inflated tree.
  The **MECHANISM** (eviction ⟺ non-argmin ∧ the beam binds) is structural and
  survives; the numbers do not. Re-measure on faithful data.
- **`delta` and `h*`** are BOTH distances in the *sampled* space, 4–5× the true
  optimal. `delta >= h*` still holds (tree vs DAG — `h*` is memoised across
  parents, `delta` follows the DFS spanning tree), but the "selecting on `h*`
  charges 5 expansions of phantom regret" framing **understated it**: both were
  wrong relative to the planner, not just `delta`.
- **`preflight_table.json` regret values** — measured on sampled trees. The
  *structure* of the verdict (which channel carries the signal) may survive; every
  regret value needs re-measuring on faithful data before it is cited.

Old numbers are kept, not deleted, marked *"measured on discard=0.4 sampled trees,
superseded."* The retraction is part of the record.

### What is unaffected

The pipeline is representation- **and data-**agnostic, and this is that property
paying off a second time: the fix is upstream data and **no pipeline code changes**.
Untouched: the completeness proposition, the eviction *mechanism*, γ=1 via SSP, the
ONNX contract, determinism, the trainer, the telemetry, the selection logic, the
scoring helper.

---

## 1. The instrument

| component | commit | tests |
|---|---|---|
| planner-config invariant (`RL_node_to_add == 1`) | `62b270e` | 14 |
| tree reconstruction + `delta` oracle | `283e001` | 6 |
| reservoir MDP + behaviour policies | `8ef9957` | 40 |
| terminated/truncated split, coverage-first | `6689c47` | — |
| ONNX contract gate | `bd8b489` | 55 |
| γ = 1 (SSP) | `896c2ed` | 12 |
| counterfactual dataset + F8 diagnostics | `e093450` | 17 |
| failure/eviction telemetry | `046d4e7` | 22 |
| two-head baseline | `6e52f8d`, `2db42d1` | 8 |
| determinism | `305ec01` | — |
| shared `viability_dominant_score` | `e7d83f9` | 16 |
| Double-DQN + CQL trainer | `0770f5a` | 19 |
| telemetry → JSONL | `a765ac0` | 10 |
| selection + gates | `99b5fc7` | 19 |

**290 tests.** The fidelity gate is armed and, on the shipped data, **correctly
failing** — which is the gate succeeding.

---

## 2. Proposition — the wrapper makes the search complete

If `delta(root) < inf`, some root child has `delta = d-1`; it is a goal (success) or
it enters `B ∪ R`; the reservoir never discards it, so it stays open until expanded,
and expanding it exposes a `delta = d-2` node. By induction `B ∪ R` always contains
a viable node.

**DOOM ⟺ `delta(root) = inf` ⟺ the instance has no reachable goal.**

Verified: min viable count in `B ∪ R` across 36 rollouts is **1**, never 0; doom
rate **0.00** for all four behaviour policies. Consequences: `doom_rate` is dead as
a selection tie-break; the expansion cap is the only non-success terminal (so it
must **truncate**, not terminate); and **every policy is proper**, which licenses
γ = 1 (stochastic shortest path, Bertsekas & Tsitsiklis). At γ = 1, `V*(s) = -delta(s)`
exactly and `G_succ(k) = -k` — no horizon, no cap coupling, and no
inability to distinguish a slow search from a hopeless one (at γ=0.99 a 1000- and a
2000-expansion search both return ≈ -100).

## 3. Proposition — V\* is an optimistic bound

`V*(B,R) = -min delta over B ∪ R` assumes an evicted node returns for free. It does
not. Demonstrated: `0 -> 1,2,3` with `delta(3)=1` the unique optimum and F=2 — node
3 overflows to R at reset and π\* pays regret on **60/60 seeds** while V\* claims -1.
Therefore `regret = k - delta(root)` is a **LOWER BOUND**, V\* is never a training
target, and `Q*(s, non-argmin)` has no closed form.

## 4. The eviction mechanism (structure survives; magnitudes superseded)

`push_vector` dumps the **whole** unexpanded beam into R regardless of what `v` was,
then refills at random:

```
eviction  ⟺  (expanded != argmin)  AND  (the beam binds)
```

Sterility is the most *common* cause, not the mechanism — a **viable** non-argmin
expansion evicts the argmin identically (tested). π\* never evicts (it always
expands the argmin); `dfs` never evicts at F=32 (its open set peaks at 18 < 32, so
nothing binds). **Eviction requires binding**, so the compounding loop is exactly
the regime where F matters.

Consequent asymmetry, which is what critic calibration must split on:
`Q*(s, argmin) = -delta(s)` **exact**; `Q*(s, v != argmin) < -1 - delta(s)`, an
optimistic upper bound. Split on **argmin**, not sterility — viable non-argmin slots
evict too. Exactness is **per-state** (the argmin's `delta-1` child must land in the
beam, index < F), not the far-too-conservative per-instance `F >= b_max`.

## 5. Transfer: blocked upstream by the frozen C++

`FringeEvalRL` accepts only **HASHED** node ids (`FringeEvalRL.tpp:199-213` exits for
BITMASK/MAPPED). The id is `boost::hash_range` over the fluent set
(`KripkeWorld.cpp:43`), and a hash **destroys the metric structure of the valuation
space**: worlds differing in one fluent get unrelated ids. Measured id overlap:

- **0.0%** across configurations (no shared fluent vocabulary — structural, not a bug)
- **15–46%** within a configuration (`CC_2_3_4__pl_7` vs `__pl_3`: 45.9%)

The transferable **BITMASK** channel (fluent bitstring, Hamming-structured,
`MAX_FLUENT_NUMBER = 18`) is implemented for `GraphNN` (`GraphNN.tpp:316,565,611`)
and **unimplemented for the RL path**. `frontier_policy.py` already carries the
matching branch, unused.

**This is the one-line lever**: enabling BITMASK in `fringe_to_tensor_minimal` is
the difference between "here is why it cannot transfer" and "here is transfer."
The pipeline is built to run on it unchanged (ast-guarded: no dataset-type branch in
trainer/telemetry/selection/export).

## 6. The discriminating channel differs BY DOMAIN — the honest counterexample

Direct measurement, survives the correction block's *structure* (values need
re-measuring):

- `SC_R_10_10__pl_10`: `oracle[wl]` **1.00** BEATS `oracle[id_knn]` **9.00** — the
  **frame** carries the signal.
- `CC_2_3_4__pl_7`: `id_knn == wl == 6.00` — topology suffices.

So *"the signal is in the valuation"* is **false in general**. A model needs **both**
channels; the frozen RL path reliably has one. This is why a single fixed encoder
struggles, and it is the counterexample to our own story — kept deliberately.

## 7. Pre-flight: the RL band is empty on HASHED data

`preflight_table.json` — leave-one-out within all 7 configurations with >=2 generated
instances, 15 held-out evaluations *(regret values superseded; see §0)*:

```
9  topology suffices (id_knn == wl)      2  FRAME (wl < id_knn)
4  nothing transfers                     0  VALUATION (id_knn << wl)  <-- empty
```

The eviction-tax thesis needs a representation that transfers with **moderate**
error. At n=15 that regime does not occur: transfer is either good enough that
topology alone suffices, or absent. **The band is not thin — it is empty.** This is
a verdict about HASHED *data*, not about the code, and it is expected to change the
moment BITMASK feeds the RL path.

## 8. Method notes that cost us

- **Determinism**: CUDA `scatter_add` atomics made `manual_seed` insufficient — the
  identical config gave held-out regret **252.3 and 93.0**. Solved (not mitigated)
  by `use_deterministic_algorithms(True, strict)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`:
  train+backward now bit-identical, no op raising. Multi-seed still mandatory for
  claims — a seed is a model, not a dice roll.
- **Five claims died at n<=3**, each killed by a larger n or a control. The
  mixing-fraction hypothesis died in minutes because the criteria were committed
  first (`PREREGISTRATION.md`): Spearman 0.415 vs a required 0.6, and a partial rho
  of **-0.11** after controlling for difficulty. It was a difficulty proxy.
- **One bug, three times**: an unbounded sterile value swamping a distance term
  (two-head score → `viability_auc` 0.048; the WL oracle's `1e6` sentinel in a class
  mean; and it would have recurred in the pre-flight). Now one helper, one test file
  (`scoring.py`, `e7d83f9`).
- **Never gate on a proxy when the true objective is measurable.** A topology-only
  model sat at `viability_auc` 0.568 (picking sterile at the base rate) while landing
  26 expansions from the ceiling and beating every baseline. Gating on the proxy
  would have killed the best model produced.
