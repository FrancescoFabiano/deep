# DESIGN — offline RL fringe-ranking trainer (`Final-Reinforcement-Learning`)

Learn a fringe-ranking policy for the C++ epistemic planner, trained offline in
trees reconstructed from the generation tables, exported as a drop-in
`frontier_policy_<F>.onnx`.

**The C++ under `src/` is frozen. The Python bends to it, never the reverse.**
Everything below is downstream of that.

---

## 1. CONTRACT (read-only, derived from the C++ consumer)

Consumer chain: `SpaceSearcher.tpp` → `RL_BestFirst.h` → `FringeEvalRL.{h,tpp}`
(ONNX Runtime).

**ONNX I/O** (`HASHED` dataset):

| # | input | dtype | shape | C++ source |
|---|---|---|---|---|
| 0 | `node_features` | int64 | `[N]` | concat of each state's `real_node_ids`, fringe order |
| 1 | `edge_index` | int64 | `[2,E]` | per-state `edge_src`/`edge_dst` + cumulative node offset |
| 2 | `edge_attr` | int64 | `[E]` | raw integer edge labels |
| 3 | `membership` | int64 | `[N]` | state index repeated per node |
| 4–7 | `goal_node_features`, `goal_edge_index`, `goal_edge_attr`, `goal_batch` | int64 | | **separated only** |
| last | `mask` | uint8 | `[F]` | `active_states`; 1 for slots `0..K-1` |

Output: `logits` float32 `[F]`. **Higher logit = expanded sooner** (`rankScores`
sorts descending, assigns rank `i` as the heuristic value; `StateComparator` is a
min-heap on it, so rank 0 = argmax logit is popped first).

Three facts that are easy to get wrong:

- **Names are not the contract; ORDER is.** `FringeEvalRL.tpp:426` reads the input
  names *from the loaded session* and pairs them **positionally** with the tensors
  it built. Renaming an input silently still works. Reordering silently produces
  garbage. Tests pin order + dtype.
- **The symbolic `F` loads because ORT resolves it.** The proto declares `logits`
  with `dim_param 'F'`, but ORT shape-inference resolves it to the concrete traced
  value before `FringeEvalRL.tpp:120` reads it — which is why the static-dim check
  (which exits on a dynamic dim) passes. `FringeEvalRL.tpp:131` then hard-fails
  unless the length equals `--RL_fringe_size`.
- **Tracing bakes `size=F` into the pooling.** The exported graph always emits `F`
  slots: at `K < F` absent candidates pool from zero rows and are masked to
  `-1e9`. Eager does *not* bake — it pools to `K` and wants a length-`K` mask. The
  two are different functions; only their **active** logits agree (verified to
  1e-5, all context modes). This is load-bearing: the offline env scores in eager
  PyTorch while the planner runs the ONNX, and `K < F` is the state at the first
  `push_vector`.

In-model encoding stays inside the graph: HASHED signed normalisation
`id/2^63 → [-1,1]`, node-label embedding `id mod 4096`, edge buckets
`abs(id) mod 128`. Export reuses `RLFrontierTrainer.to_onnx` **verbatim**
(opset 18, tracing, same names/order/dynamic axes).

File naming `<exp_dir>/_models/<domain>/frontier_policy_<F>.onnx`
(`bulk_coverage_run.py` depends on it).

---

## 2. THE DEPLOYMENT CONFIG IS PART OF THE MODEL

The offline MDP models **one expansion per ONNX call**. The planner only behaves
that way when the driver's accumulation threshold is 1:

```
SpaceSearcher.tpp:179      if (fringe_RL.size() >= RL_node_to_add || empty())
                               push_vector(fringe_RL);
Configuration.cpp:121-130  RL_node_to_add = int(F * RL_exploitation / 100.0)
```

With the C++ defaults (F=32, exploitation=70) that is **22**: the planner expands
~22/b nodes per ONNX call using *stale* ranks in between. Driving it to 1 restores
one-expansion-per-call at every F:

| F | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `--RL_exploitation` | 25 | 13 | 7 | 4 | 2 |

Feasible only for **F ∈ [2, 199]** (needs an integer `e` with `100 <= F*e <= 199`;
at F>=200 even `e=1` gives `int(F/100) >= 2`). Export **refuses** on a mismatch.

**Refill must be `--RL_heuristics RNG`** (→ `RefillMode::RANDOM`). MIN/MAX/AVG map
to `RefillMode::HEURISTIC`, whose reservoir order comes from `Heuristics::RL_H`
folding *stored past RL ranks* (`HeuristicsManager.tpp:139-164`) — not
reconstructible from a generation table. The C++ default is MIN, so the launcher
passes RNG explicitly. `--RL_exploration 0` (inert under RNG anyway).

---

## 3. THE MDP

```
s = (B, R)     B = beam, |B| <= F   (the candidates the net scores)
               R = reservoir        (every other open, unexpanded node)
a = v ∈ B                           (which slot to expand)
```

The agent observes **only B** — that is what the ONNX takes. `R` is environment
state, so the observation is a strict subset of the state and **the learned policy
operates under partial observability**. The environment must still carry `R` or
the transition is undefined and the critic fits dynamics that do not exist.

`F` is a **scoring window, not a beam width**. An RL policy has no completeness
property; `RL_BestFirst` wraps it in a structure that never discards a node.

### Transition (mirrors `push_vector`, order included)

```
b_v >= 1:  R' = R ∪ (B\{v}) ∪ children[F:]        # drain BEFORE overflow
           B' = children[:F] + refill(R')          # refill AFTER, at random
           -> rescore point, |A(s')| = |B'|

b_v == 0:  B' = B \ {v},  R' = R                   # NO rebuild, NO refill
           -> FORCED: the planner pops the stale rank; the model is NOT called
```

The dead-end path is not a rounding error. `fringe_RL` stays empty, so `0 >= 1` is
false and `push_vector` never fires. Dead ends are **17.5%** of `CC_2_3_4__pl_7`
and **17%** of visited states are forced. An env that rescored here would silently
diverge. The env therefore carries the model's full priority order, not just the
chosen action.

Also modelled: `peek()`'s **unscored reservoir pull** (`RL_BestFirst.h:127-142`) —
when consecutive dead ends drain the beam while R is non-empty, one random
reservoir node is pushed straight into the queue, never scored.

### Termination

```
SUCC   : a generated child is a goal    → r = 0,  terminated
DOOM   : B ∪ R empty / all non-viable   → absorbing, terminated
TRUNC  : expansion cap                  → r = -1, TRUNCATED (bootstrap!)
```

---

## 4. PROPOSITION 1 — the wrapper makes the search complete

> If `delta(root) < inf` then some root child has `delta = d-1`. It is either a
> goal (success) or it enters `B ∪ R`. **The reservoir never discards it**, so it
> stays open until expanded, and expanding it exposes a `delta = d-2` node. By
> induction `B ∪ R` always contains a viable node, so it can be neither empty nor
> entirely non-viable.
>
> **DOOM ⟺ delta(root) = inf ⟺ the instance has no reachable goal.**

Empirically: the minimum viable count in `B ∪ R` across 36 real rollouts is **1**,
never 0; doom rate is **0.00** for all four behaviour policies on
`CC_2_3_4__pl_7`.

This is the reservoir doing the job it exists for. Four consequences:

1. `doom_rate` is provably 0 on solvable data — **dead as a selection tie-break**.
2. The absorbing doom reward never fires in training. Unsolvable instances are
   filtered at load (logged by name in the manifest) and the env refuses to
   construct on one, so the doom branch is **dead code on every env that can
   exist**. What stays live is the assert: doom firing on a solvable instance
   means the transition is dropping nodes.
3. The expansion cap is the only non-success terminal → §5.
4. **It licenses γ = 1** → §6.

---

## 5. REWARD, AND THE BUG THAT WAS FIXED

```
r(s,a) = 0     if SUCC
       = -1    otherwise
```

The previous env made the cap **terminal** with reward -1, so a truncation
returned `-m`: **stopping early scored better than searching**, and offline there
is no exploration to discover that this is an artifact. The old design attributed
this to "fringe+reservoir exhaustion (dead end)", but by Proposition 1 that never
fires — **the cap was the only trigger**.

The fix is the terminated/truncated split, not a discount:

```
y = r + (1 - terminated) * max_a' Q(s', a')        # NOT (1 - done)
```

The cap has no deployment counterpart *as a terminal state*: the planner does not
stop and declare failure at our training budget, and by Proposition 1 it would
eventually succeed. A capped rollout is one **we stopped watching**. Treating it
as terminal says a state 3000 expansions deep is worth -1 when it is really worth
`-E[remaining expansions]`; that leaks backwards and makes the critic
systematically optimistic about long searches (Pardo et al. 2018). Truncation
therefore returns the successor beam to bootstrap from.

---

## 6. γ = 1 — the objective is undiscounted

By Proposition 1 every policy reaches a goal on a solvable instance: **every
policy is proper**. Costs are strictly positive, the process terminates w.p. 1
under any policy, and there is no absorbing failure state to escape into. That is
the stochastic shortest path setting (Bertsekas & Tsitsiklis) — the undiscounted
Bellman operator has a unique fixed point. Discounting exists to make
*non-terminating* processes well-posed. Ours terminates.

```
V*(s)     = -delta(s)   exactly
G_succ(k) = -k          linear, no saturation, no horizon, no cap coupling
```

And it removes a distortion: at γ=0.99 the horizon is 100 expansions, so a
1000- and a 2000-expansion search both return ≈ -100 — the critic is
**structurally unable to tell a slow search from a hopeless one**. Even the real
baseline spread on CC (oracle 34 … bfs 231) compresses to <10 apart. We are
minimising expansions; γ<1 stops counting them.

`--gamma` stays as an **ablation axis**, warning loudly below 1.

---

## 7. `delta` vs `h*` — they are not the same

- `h*(v)` = CSV `Distance From Goal`: true distance in the **deduplicated state
  DAG**.
- `delta(v)` = distance to the nearest goal **inside the reconstructed tree**:
  `0` if goal, `inf` if non-goal leaf, else `1 + min_c delta(c)`.

The reconstruction is a **DFS spanning tree of that DAG** (the parent pointer is
the first discoverer in DFS order), so tree paths overestimate:

| instance | h*(root) | delta(root) | gap |
|---|---|---|---|
| `CC_2_3_4__pl_7` | 29 | 34 | 5 |
| `SC_R_10_10__pl_10` | 10 | 24 | 14 |

`delta >= h*` node-wise on both (0 violations). The MDP moves only within the
tree, so **delta is the correct reference for V\* and regret** — selecting on `h*`
would charge 5 expansions of phantom regret on CC. Both are environment-side only:
never features, never targets.

---

## 8. FAILURE IS NODE-LEVEL

Proposition 1 killed episode DOOM. It did **not** kill failure. **34.8%** of
`CC_2_3_4__pl_7`'s reachable nodes have `delta = inf` — their whole subtree
contains no goal. Those are the failure states, they sit in every beam, and
avoiding them is the primary thing the policy must learn.

### Eviction — the compounding loop

`push_vector` dumps the **whole** unexpanded beam into R regardless of what `v`
was, then refills at random. So:

```
eviction  ⟺  (expanded != argmin)  AND  (the beam binds)
```

Sterility is the most *common* cause, not the mechanism. Expanding a non-argmin
node evicts the argmin into R, from which it returns only by a random draw —
against a reservoir that just grew. **The cost of a wrong pick is not -1**; it is
-1 plus the expected recovery wait, and that wait grows with |R|.

Measured on `CC_2_3_4__pl_7`, F=32, 8 seeds:

| policy | expansions | sterile exp. | beam sterile | evictions | recovery | mean \|R\| | max \|R\| |
|---|---|---|---|---|---|---|---|
| `hfs_oracle` | 34 | **0.0%** | 13.2% | 0.0 | – | 0.0 | 0 |
| `dfs` | 98 | 43.0% | 31.7% | 0.0 | – | 0.0 | 0 |
| `random` | 195 | 49.3% | 50.0% | 33.8 | 2.4 | 11.8 | 47 |
| `bfs` | 229 | **60.0%** | 60.8% | 43.9 | 2.9 | 18.6 | 74 |

π\*'s beam is 13.2% sterile — sterile nodes *are* in the beam, it just never
expands them, which is why the ranking task is real. `dfs` evicts **nothing**
despite 43% sterile expansions: its open set peaks at 18 < 32, so nothing binds.
**Eviction requires binding**, so the compounding loop is exactly the regime where
F matters.

---

## 9. PROPOSITION 2 — V\* is an optimistic bound

`V*(B,R) = -min delta over B ∪ R` assumes an evicted node comes back for free. It
does not. Demonstrated: with `0 -> 1,2,3` where `delta(3)=1` is the unique optimum
and F=2, node 3 **overflows to R** at reset; π\* pays regret on **60/60 seeds**
while V\* claims -1.

Therefore:

- **`regret = k - delta(root)` is a LOWER BOUND on true regret.** Axis label:
  `regret (lower bound)`, with the caveat in the caption.
- **V\* is never a training target.** Diagnostics and selection only.
- **`Q*(s,v)` for a non-argmin `v` has no closed form** — it depends on the refill
  process. The critic must learn it by bootstrapping through real stochastic
  transitions. This is the first thing here a regressor on `delta` cannot do.

### The Q\* asymmetry (F4)

```
Q*(s, argmin)      = -delta(s)        EXACT
Q*(s, v != argmin) < -1 - delta(s)    optimistic UPPER bound
```

Exactness is **per-state**, not per-instance: it needs the argmin's
delta-decreasing child to land in the beam (index < F) rather than overflow. The
per-instance `F >= b_max` test is far too conservative — `b_max` is driven by a few
high-branching nodes while the good child almost always sits early in the child
order. At F=8, **100%** of viable internal nodes are exact on instances with
`b_max` = 8, 20, 21 and 23, where the per-instance test would reject three of four.

---

## 10. WHAT THE RL IS FOR (the falsifiable claim)

Under a **perfect** delta, greedy-on-delta is optimal, never expands a non-argmin,
never evicts, and pays no tax — π\* at 34 expansions with 0 evictions is exactly
that, measured. **Under perfect information there is nothing for RL to add.**

The learned policy has `delta-hat` and will sometimes expand a non-argmin. **The
eviction tax is the cost of that error, and it compounds.** A regressor minimises
prediction error; it has no representation of what a prediction error *costs*. A
critic bootstrapping through real transitions does, because the cost is literally
in the returns it fits.

**The baseline** (mandatory, `models/two_head_baseline.py`): same network, same
encoder, same context mode, same head width — only the loss differs. Two heads on
exact labels, **no clip and no sentinel** (any clip would be a hyperparameter we
chose, making the comparison unfalsifiable):

```
viability : p(v) = P[delta(v) < inf]    BCE, all slots
distance  : d_hat(v) ~ delta(v)         MSE, VIABLE slots only
score     : -d_hat(v) + logit p(v)      (one scalar -> same ONNX contract)
```

Evaluated in the **same env** — same reservoir, same random refill, same F, same
seeds. Not a clean best-first, or the eviction tax vanishes from the baseline by
construction.

Prediction:

| metric | expectation |
|---|---|
| `viability_auc` | **TIE** (the baseline optimises it directly) |
| `expansions_sterile_frac` | TIE, or near |
| `eviction_events`, recovery, \|R\| | **RL strictly better** |
| `regret` | RL better, **concentrated in the binding cohort**, growing with \|R\|; indistinguishable on non-binding instances |

**If eviction is not where the RL wins, the thesis is wrong** and this becomes a
paper about a two-head learned heuristic for MEP — which would be a perfectly good
paper. Better to find that early than defend it at review.

---

## 11. THE FRINGE KNOB IS POLICY-DEPENDENT

Occupancy must be **measured**, not bounded. The old `bfs_frontier_max` documented
itself as "a policy-free UPPER BOUND on how many states can be simultaneously
live". **It is not.** The search stops when a goal is *generated*, so a policy that
finds a goal later expands more and accumulates a larger open set: on CC the
statistic says 105 while bfs actually reaches 117 and `hfs_oracle` only 17.

An instance whose open set never exceeds F **cannot exercise F at all** — the beam
never fills, refill never fires:

| F | binding | inert |
|---|---|---|
| 4 | **15/29** | 14 |
| 8 | 11/29 | 18 |
| 16 | 6/29 | 23 |
| 32 | 4/29 | 25 |
| 64 | 2/29 | 27 |

This is **not a defect of the instance pool** — it is the result. The beam binds
precisely when the ranker is bad. **F is a safety net for a bad ranker**; a good
ranker never floods the reservoir and does not need one. So F7's honest story is
not "bigger F is better" but *"the better the ranker, the less F matters"*, and the
learned policy's occupancy **falling** over training is itself a finding.

Consequences:
- F-sweep / context aggregates are computed over the **binding cohort only**, with
  `n` stated in every title. Inert instances stay in **coverage**, where they are
  valid instances.
- Instances with `max_open <= 2` have `|A(s)| <= 2` — essentially no ranking to do.
  Kept for coverage, **excluded from ranking claims**; both `n`s reported.
- The sweep's centre of gravity is **low F**, with F>=32 as the control arm.

The claim to defend: *the learned ranker beats the baselines where ranking
matters, and does not hurt where it doesn't.*

---

## 12. MODEL SELECTION

**Instance-level splits, never row-level** (rows from one tree share nodes; a
model can memorise a tree it has partially seen). Split on instance name with a
fixed seed, recorded in the manifest. `test_data` is touched exactly once.

**The cap is not a semantic threshold.** Any single cap is an arbitrary line
through the policy spread (F=32 on CC: oracle 34, dfs 81, random 199, bfs 231), and
whichever we pick decides which baselines "fail". So the cap is a generous
backstop — `max(2000, 50 * max delta(root))`, logged — and the reported quantity is
a **coverage curve over budget** (a cactus plot, F1'), which makes the budget a
parameter of the *reader*.

Selection, per `(domain, F, kind_of_data, model)` cell:

```
primary   : coverage at the declared reference budget (10 * delta(root),
            difficulty-relative; declared once, never tuned)
tie-break : 1. regret over solved instances (lower)
            2. earlier checkpoint
```

Not TD loss. Not mean Q. `doom_rate` is provably 0 and is not a tie-break.

### Gates before an ONNX is deployable

1. **ONNX parity** — onnxruntime reproduces PyTorch to 1e-5, both encodings, all
   three context modes. *(green)*
2. **Env fidelity** — the C++ planner and the offline env agree on expansion
   counts within a tolerance derived from measured tree-replay-vs-live-dedup
   divergence. If they disagree, every number we report is meaningless.
3. **Beats the baselines** — including the two-head baseline, on the binding
   cohort. Warn prominently if not; do not silently export.

---

## 13. TELEMETRY

Per checkpoint, appended to `telemetry.jsonl`; **figures are generated from the
JSONL**, never coupled to the training loop.

```
step, frames, split
coverage@budget, coverage_curve, regret_mean/median/p90 (lower bound),
  expansions_mean, success_rate, doom_rate (== 0), timeout_rate
viability_auc, expansions_sterile_frac, beam_sterile_frac       # F10
eviction_events, eviction_recovery_steps, reservoir_size mean/max
top1_oracle_agreement, spearman_logits_vs_delta
q_vs_qstar_r2 (argmin slots), qstar_naive_residual + corr(|R|)  # F4
td_loss, q_mean, q_max, grad_norm, lr
forced_state_frac, actions_per_state, adv_hist                  # constant, once
```

`viability_auc` returns **None** on all-viable/all-sterile beams rather than
fabricating 0.5 — with 11 instances at `max_open <= 2`, averaging a fabricated 0.5
would drag the aggregate toward "learned nothing".

**Gate: if `viability_auc` is flat at 0.5, stop and report.** Nothing downstream is
worth plotting.

### Figures

| | |
|---|---|
| **F1'** | `coverage_vs_budget.png` — cactus plot, one curve per policy, one panel per F |
| **F2** | `train_vs_val.png` — divergence = overfitting to the training trees |
| **F3** | `oracle_agreement.png` — top1 + spearman vs frames |
| **F4** | `critic_calibration.png` — **A**: Q(s,argmin) vs -delta (diagonal). **B**: residual below the naive bound for non-argmin, vs \|R\| — **this is the paper** |
| **F5** | `expansions_distribution.png` — violin per policy at the selected checkpoint |
| **F6** | `truncation_ablation.png` — `terminal_at_cap` vs `truncated`; **drop it if the arms are indistinguishable** and put Proposition 1 in its place |
| **F7** | `fringe_size_sweep.png` — binding cohort only, n in the title; overlay occupancy |
| **F8** | `instance_diagnostics.png` — b_v hist, \|L_d\|, forced chain, delta vs h*, densities, **measured** occupancy; flags inert instances |
| **F9** | `context_mode_comparison.png` — none / mean_pool / self_attention |
| **F10** | `sterile_avoidance.png` — `expansions_sterile_frac` + `viability_auc` vs frames, with behaviour policies as reference lines; second panel: mean/max \|R\| falling toward the π\* line |

Every figure: axis labels with units, legend, `n` in the title, and
`regret (lower bound)` never unqualified.

---

## 14. TRAINER

`|A(s)| <= F`, so `max_a Q` and `logsumexp` are trivial: **Double-DQN** (workhorse)
+ **CQL** as the conservatism ablation (`--cql-alpha`, 0 valid).

**No IQL.** It was proposed when the action was believed to be `(v, D)` —
combinatorial, so `max_a`/`logsumexp` were infeasible. The reservoir killed that
premise. Worse, IQL is now actively wrong here: its purpose is to never evaluate
out-of-distribution actions, but counterfactual expansion gives **every** action in
every visited state with the exact successor — there is no off-support action. And
its advantage-weighted extraction can only reweight actions the behaviour took,
which **excludes precisely the counterfactual rows where the eviction tax lives**.

**No exploration.** This is offline; behaviour randomness lives entirely in data
generation.

`--reward-scale` defaults to `1 / median(delta_root over train instances)`
(measured **12** on the current pool — CC's 34 is an outlier, not the median). Pure
rescaling — it cannot change the optimal policy — but it keeps Q in a sane range at
γ=1. `1/expansion_cap` would put the per-step signal at 5e-4, below init noise.
Everything human-facing reports **unscaled expansions**.

γ=1 gives no contraction, so TD with function approximation can drift. Three
guards: the exact `Q*` reference as a divergence alarm (`q_vs_qstar_r2`, `q_max`,
abort if `|Q| > 3x` cap), reward scaling, and Double-DQN + target network. **If it
diverges anyway that is a RESULT** (γ ∈ {1.0, 0.999, 0.99} figure), not a silent
fallback.

---

## 15. DATA

Generation-table CSVs, unchanged:
`File Path, Depth, Distance From Goal, Goal, File Path Predecessor, Action` under
`<exp_dir>/_models/<domain>/{training_data,test_data}/<instance>/`.

Reconstruction: duplicate `File Path` → min-depth row wins; root = the unique
Depth-0 state; edges whose predecessor never appears are dropped as orphans
(<1%). Verified a **strict tree**: every node has exactly one parent, every edge is
depth+1, no cycles. `delta` is nonetheless computed by multi-source BFS from the
goal set over reversed edges — the shortest-path fixpoint — so it cannot silently
go wrong if that ever stops holding.

**merged / separated is a first-class axis**, not a flag to prune: a different
state representation (goal inlined into every state DOT vs. a separate
`goal_tree.dot` fed as 4 extra tensors). Goal loading is driven by `kind_of_data`,
**never** by the presence of the `Goal` column — merged runs populate that column
too, but do not write the file.

Behaviour policies: one mechanism, priority `sigma(v)`, **randomised tie-breaking
mandatory** (without it each deterministic policy yields one trajectory per
instance and the dataset collapses):

| policy | `sigma(v)` |
|---|---|
| `bfs` | `depth(v)` |
| `dfs` | `-depth(v)` |
| `hfs_oracle` | `delta(v)` |
| `random` | `U(0,1)` |

**`hfs_oracle` is π\*, not a baseline.** It ranks by `delta` — the answer to the
problem — so it is clairvoyant and not executable at deployment. It is (a) the
evaluation **ceiling** and (b) expert demonstrations. The name stays `hfs_oracle`
everywhere so it cannot be mistaken for a competitor.

Dataset: **counterfactual action expansion**. The tree is a model and
`|A(s)| <= F <= 64`, so enumerate every action per state (do not sample) and follow
**one** successor (branching on all `|B|` would re-enumerate the search space).
Measured: actions/state 1.00 → 5.02 at F=8 on CC. Under random refill the successor
is a random variable: emit `--n-refill-samples` draws as **separate rows**, never
averaged.
