"""Fringe-composition regimes (P1, value-only). See DESIGN.md §2 + the P1 brief.

The deployed planner and the existing FringeEnv seat *fresh children first* and
fill only the residual beam slots from the reservoir (children-first, mirrors
RL_BestFirst::push_vector). These regimes instead do a FULL-BEAM-REDRAW: at every
expansion the entire F-beam is rebuilt from the live pool

    P = fresh-children  ∪  reservoir

by a per-regime rule, always exactly min(F, |P|) slots, no repetition within a
fringe. They are TRAIN-DISTRIBUTION GENERATORS only — deployment still seats
children first, so model selection uses the deploy-faithful eval (regime_trainer
.greedy_rollout with refill_mode='heuristic'), never a redraw rollout.

Nothing here touches the model / target / ONNX contract. d* and depth are read
off the TreeInstance oracle (encoder node_features are hashed ids only), exactly
as P0 confirmed; d* is the evaluation oracle and only shapes the *training state
distribution* here, consistent with the rank objectives that already read it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from src.offline.tree_env import UNREACHABLE_DISTANCE, FringeEnv, TreeInstance

REGIMES = ("dfs", "bfs", "hfs_m0", "hfs_m1", "random")
_BIG = 1 << 60


def _round_hu(x: float) -> int:
    """Round half-up (0.5 -> 1), matching the P0b/P0c slot allocation."""
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------
# Per-instance static structure (computed once at load; never per step)
# ---------------------------------------------------------------------------

def reachable_from_root(inst: TreeInstance) -> set:
    seen = {inst.root_id}
    stack = [inst.root_id]
    while stack:
        s = stack.pop()
        for c in inst.children[s]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def eligible_nodes(inst: TreeInstance) -> List[int]:
    """E_i = reachable(root) ∧ not goal ∧ 1 <= d* < 1e6, in state-id order."""
    out = []
    for s in sorted(reachable_from_root(inst)):
        d = inst.distance[s]
        if (not inst.is_goal[s]) and 1.0 <= d < UNREACHABLE_DISTANCE:
            out.append(s)
    return out


def dfs_preorder_rank(inst: TreeInstance) -> List[int]:
    """rank[s] = position of s in a DFS preorder from the root following CSV
    child order (generation order). Unreached states get _BIG (sorted last)."""
    rank = [_BIG] * inst.n_states
    seen = {inst.root_id}
    stack = [inst.root_id]
    order: List[int] = []
    while stack:
        s = stack.pop()
        order.append(s)
        for c in reversed(inst.children[s]):  # reversed -> children visited in CSV order
            if c not in seen:
                seen.add(c)
                stack.append(c)
    for r, s in enumerate(order):
        rank[s] = r
    return rank


def depth_dstar_var_explained(inst: TreeInstance, elig: Sequence[int]) -> float:
    """1 - Var[d* | depth] / Var[d*] over eligible nodes (the P0c collinearity
    statistic). ~1 => depth determines d* => same-depth fringes are mono-d*."""
    if len(elig) < 2:
        return float("nan")
    ds = [inst.distance[s] for s in elig]
    n = len(ds)
    mean = sum(ds) / n
    tot = sum((d - mean) ** 2 for d in ds) / n
    if tot <= 0:
        return 1.0
    by_depth: Dict[int, List[float]] = defaultdict(list)
    for s in elig:
        by_depth[inst.depth[s]].append(inst.distance[s])
    within = 0.0
    for v in by_depth.values():
        m = sum(v) / len(v)
        within += sum((d - m) ** 2 for d in v)  # sum, not mean (pooled)
    within /= n
    return 1.0 - within / tot


def bfs_usable_fraction(inst: TreeInstance, elig: Sequence[int], F: int) -> tuple:
    """Realized BFS-usable supply at this F (P0c): tile each depth's eligible
    nodes (id order) into size-F windows, a window is 'usable' iff it has >=2
    distinct d*. Returns (usable, total). total==0 -> no same-depth F-window."""
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for s in elig:  # elig already id-sorted
        by_depth[inst.depth[s]].append(s)
    usable = total = 0
    for v in by_depth.values():
        for t in range(len(v) // F):
            chunk = v[t * F:(t + 1) * F]
            total += 1
            if len({inst.distance[s] for s in chunk}) >= 2:
                usable += 1
    return usable, total


def single_distinct_dstar(inst: TreeInstance, fringe: Sequence[int]) -> bool:
    """True iff every node in the fringe shares one distance-from-goal value
    (the same-distance filter target). Empty/size-1 fringes count as single."""
    if len(fringe) <= 1:
        return True
    it = iter(fringe)
    first = inst.distance[next(it)]
    return all(inst.distance[s] == first for s in it)


# ---------------------------------------------------------------------------
# HFS composition diagnostics (m0 vs m1 tail coverage — the P1 result)
# ---------------------------------------------------------------------------

class HFSDiag:
    """Accumulates, over a checkpoint window, the realized-vs-intended d*-bucket
    composition for one HFS regime so stv_real/stv_true + unsampled tail mass
    are reportable (the m0/m1 contrast).

    A bucket is 'starved' on a step iff 0 < p_b*F < 0.5 (round-half-up gives it 0
    intended slots). stv_true = starved present-mass fraction; stv_real = realized
    sampled-frequency from starved buckets. m1's min-1 floor should drive both the
    starved present-mass and the unsampled tail toward 0 relative to m0."""

    def __init__(self) -> None:
        self.present_mass: Dict[int, int] = defaultdict(int)   # Σ_steps |bucket_b|
        self.realized: Dict[int, int] = defaultdict(int)       # Σ_steps slots taken from b
        self.present_total: int = 0
        self.realized_total: int = 0
        self.starved_present: int = 0                          # present mass on starved steps
        self.starved_realized: int = 0                         # realized slots on starved steps
        self.n_steps: int = 0

    def record(self, p: Dict[int, float], present: Dict[int, int],
               realized: Dict[int, int], F: int) -> None:
        self.n_steps += 1
        for b, m in present.items():
            self.present_mass[b] += m
            self.present_total += m
            r = realized.get(b, 0)
            self.realized[b] += r
            self.realized_total += r
            if 0.0 < p[b] * F < 0.5:               # starved this step
                self.starved_present += m
                self.starved_realized += r

    def summary(self) -> Dict[str, object]:
        present_buckets = set(self.present_mass)
        sampled_buckets = {b for b, c in self.realized.items() if c > 0}
        unsampled = present_buckets - sampled_buckets
        return {
            "n_steps": self.n_steps,
            "n_buckets_present": len(present_buckets),
            "stv_true": round(self.starved_present / self.present_total, 4)
            if self.present_total else 0.0,
            "stv_real": round(self.starved_realized / self.realized_total, 4)
            if self.realized_total else 0.0,
            "n_unsampled_buckets": len(unsampled),
            "unsampled_mass": int(sum(self.present_mass[b] for b in unsampled)),
        }


# ---------------------------------------------------------------------------
# Beam selection
# ---------------------------------------------------------------------------

def select_beam(
    regime: str,
    pool: Sequence[int],
    inst: TreeInstance,
    F: int,
    rng,
    dfs_rank: Sequence[int],
    hfs_diag: Optional[HFSDiag] = None,
) -> List[int]:
    """Rebuild the F-beam from the live pool by the regime rule. Returns EXACTLY
    min(F, |pool|) unique state ids (the pool is unique by construction).

    Post-condition guard: when |P| <= F every regime returns the whole pool (no
    selection — DFS/BFS/HFS_m0/HFS_m1/random are identical there); when |P| > F
    the regime emits F slots. The trailing top-up is a defensive net so a future
    regime-rule edit can never silently emit a short beam (the hfs_m1 small-pool
    regression class) — it is a no-op on every current path (test_select_beam)."""
    P = list(pool)
    target = min(F, len(P))
    if len(P) <= F:
        return P                                   # whole pool, all regimes alike
    if regime == "dfs":
        beam = sorted(P, key=lambda s: (dfs_rank[s], s))[:F]
    elif regime == "bfs":
        beam = sorted(P, key=lambda s: (inst.depth[s], s))[:F]
    elif regime == "random":
        beam = rng.sample(P, F)
    elif regime in ("hfs_m0", "hfs_m1"):
        beam = _hfs_select(regime, P, inst, F, rng, hfs_diag)
    else:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")

    if len(beam) < target:                         # defensive top-up (never short)
        seen = set(beam)
        for s in P:
            if s not in seen:
                beam.append(s)
                seen.add(s)
                if len(beam) >= target:
                    break
    return beam[:target]


def _hfs_select(regime, P, inst, F, rng, diag) -> List[int]:
    """Histogram-matched beam over exact d* buckets.

    m0: slots_b = round(p_b*F); starved buckets (round->0) get nothing.
    m1: min-1-slot floor on every non-empty bucket (prioritise the F nearest-goal
        buckets when #buckets > F); remaining slots proportional to p_b.
    Both: clamp by availability, reconcile to F (trim largest-mass buckets m1 /
    largest-slot buckets m0), sample without replacement, then residual-fill to F
    from the remaining pool (ascending-d* finite first, then anything). Single
    draw per beam, so 'nearest-bucket borrow on depletion' reduces to the
    residual fill (no cross-fringe depletion within one beam)."""
    finite = [s for s in P if 1.0 <= inst.distance[s] < UNREACHABLE_DISTANCE]
    if not finite:
        return rng.sample(P, min(F, len(P)))
    buckets: Dict[int, List[int]] = defaultdict(list)
    for s in finite:
        buckets[int(inst.distance[s])].append(s)
    bkeys = sorted(buckets)
    n = len(finite)
    p = {b: len(buckets[b]) / n for b in bkeys}

    if regime == "hfs_m0":
        slots = {b: _round_hu(p[b] * F) for b in bkeys}
    else:  # hfs_m1
        slots = {b: 0 for b in bkeys}
        if len(bkeys) > F:
            for b in sorted(bkeys)[:F]:           # F nearest-goal buckets get the floor
                slots[b] = 1
        else:
            for b in bkeys:
                slots[b] = max(1, _round_hu(p[b] * F))

    for b in bkeys:                               # clamp by availability
        slots[b] = min(slots[b], len(buckets[b]))

    # reconcile to exactly F (finite pool permitting)
    total = sum(slots.values())
    if total > F:
        trim_order = (
            sorted(bkeys, key=lambda b: (-len(buckets[b]), b))   # m1: trim largest mass
            if regime == "hfs_m1"
            else sorted(bkeys, key=lambda b: (-slots[b], b))     # m0: trim largest slot
        )
        floor_b = 1 if (regime == "hfs_m1" and len(bkeys) <= F) else 0
        i = 0
        guard = 0
        while total > F and guard < 100000:
            b = trim_order[i % len(trim_order)]
            if slots[b] > floor_b:
                slots[b] -= 1
                total -= 1
            i += 1
            guard += 1

    chosen: List[int] = []
    realized: Dict[int, int] = defaultdict(int)
    for b in bkeys:
        k = slots[b]
        if k <= 0:
            continue
        picks = rng.sample(buckets[b], k)
        chosen.extend(picks)
        realized[b] += k

    if len(chosen) < F:                           # residual fill
        chosen_set = set(chosen)
        rem_finite = [s for b in bkeys for s in buckets[b] if s not in chosen_set]
        rng.shuffle(rem_finite)
        for s in rem_finite:
            if len(chosen) >= F:
                break
            chosen.append(s)
            realized[int(inst.distance[s])] += 1
        if len(chosen) < F:
            chosen_set = set(chosen)
            rem_other = [s for s in P if s not in chosen_set]
            rng.shuffle(rem_other)
            chosen.extend(rem_other[: F - len(chosen)])

    chosen = chosen[:F]
    if diag is not None:
        present = {b: len(buckets[b]) for b in bkeys}
        diag.record(p, present, realized, F)
    return chosen


# ---------------------------------------------------------------------------
# Redraw environment
# ---------------------------------------------------------------------------

class RedrawFringeEnv(FringeEnv):
    """FringeEnv whose beam is FULL-BEAM-REDRAWN by a composition regime.

    Reuses reset()/_generate_children unchanged; overrides _rebuild_beam (both
    reset() and step() call it with the fresh children) so the whole live pool
    (fresh ∪ reservoir) is re-composed every expansion — no children-first
    privilege.

    AUTO-PADDING (self-activating when ``pad_to_F`` is set, i.e. F > the
    instance's bfs_frontier_max): on instances whose live frontier can never
    reach F, the regimes would be indistinguishable (below F all five return the
    whole pool). Padding tops the beam up to F from CLOSED (already-expanded)
    nodes — drawn by the SAME regime rule, after the live pool — so each regime
    composes a real F-slot beam. Closed nodes are reachable, non-goal, may be
    unreachable-to-goal (allowed, floored downstream); GOALS NEVER enter a beam
    (they terminate at generation, are never expanded, so never closed). Re-
    expanding a closed pad node yields no fresh children but is charged a normal
    expansion / -1 reward (node economy), tracked as a pad expansion.
    """

    def __init__(self, instance, fringe_size, seed, regime, dfs_rank,
                 expansion_cap=None, hfs_diag=None, pad_to_F=False):
        super().__init__(instance, fringe_size=fringe_size, seed=seed,
                         expansion_cap=expansion_cap, refill_mode="random")
        self.regime = regime
        self.dfs_rank = dfs_rank
        self.hfs_diag = hfs_diag
        self.pad_to_F = bool(pad_to_F)
        self.last_pool_size = 0
        # closed = already-expanded nodes (pad source); pad accounting (windowed).
        self.closed: set[int] = set()
        self.n_pad_slots_filled = 0
        self.n_pad_expansions = 0
        # per-slot pad flag, aligned with self.fringe (True = pad-filled slot).
        self.pad_flags: List[bool] = []

    def reset(self, seed=None) -> "StepResult":
        # root is expansion 1 => it is closed before the first beam is built.
        self.closed = {self.instance.root_id}
        self.pad_flags = []
        return super().reset(seed)

    def step(self, action: int) -> "StepResult":
        # Charge pad accounting BEFORE delegating: peek the chosen slot, mark it
        # closed (or count a re-expansion of an already-closed pad node). On
        # faithful (non-padding) envs this is inert other than bookkeeping that
        # never affects dynamics. super().step pops the slot + calls _rebuild_beam.
        if self.pad_to_F and 0 <= action < len(self.fringe):
            chosen = self.fringe[action]
            if chosen in self.closed:
                self.n_pad_expansions += 1
            else:
                self.closed.add(chosen)
        return super().step(action)

    def reset_pad_counters(self) -> None:
        self.n_pad_slots_filled = 0
        self.n_pad_expansions = 0

    def _rebuild_beam(self, new_states: List[int]) -> None:
        # old beam -> reservoir, then recompose the whole F-beam from the live pool.
        self.reservoir.extend(self.fringe)
        self.fringe = []
        pool = list(new_states) + self.reservoir   # both sides disjoint, unique
        self.last_pool_size = len(pool)            # live pool size (bind metric)
        beam = select_beam(
            self.regime, pool, self.instance, self.fringe_size,
            self.rng, self.dfs_rank, self.hfs_diag,
        )
        sel = set(beam)
        self.reservoir = [s for s in pool if s not in sel]
        pad_flags = [False] * len(beam)            # live composition slots
        if self.pad_to_F and len(beam) < self.fringe_size:
            # Live frontier < F: top up from reservoir (live, normally empty here)
            # then CLOSED nodes, selected by the regime rule so regimes still
            # differ on the padded slots. No goal can be present in either source.
            need = self.fringe_size - len(beam)
            inbeam = set(beam)
            pad_src = [s for s in self.reservoir if s not in inbeam]
            pad_src += [s for s in self.closed if s not in inbeam]
            if pad_src:
                pad = select_beam(self.regime, pad_src, self.instance, need,
                                  self.rng, self.dfs_rank, None)
                beam = beam + pad
                pad_flags += [True] * len(pad)     # appended slots are pad-filled
                self.n_pad_slots_filled += len(pad)
                padset = set(pad)
                self.reservoir = [s for s in self.reservoir if s not in padset]
        self.fringe = beam
        self.pad_flags = pad_flags

    def _info(self, goal_found: bool) -> Dict[str, object]:
        info = super()._info(goal_found)
        info["pool_size"] = self.last_pool_size
        info["regime"] = self.regime
        info["padded"] = self.pad_to_F
        # pad_flags aligned with the CURRENT self.fringe (= the StepResult.fringe
        # captured alongside this info), so the trainer attaches a slot-aligned
        # pad_mask at push without any timing hazard.
        info["pad_flags"] = list(self.pad_flags)
        return info
