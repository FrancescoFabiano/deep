"""Telemetry metrics. All of these need `delta`, so they exist ONLY offline.

The two that go in the paper:

  expansions_sterile_frac  fraction of expansions spent on delta=inf nodes.
                           Measured on CC_2_3_4__pl_7 at F=32: hfs_oracle 0.0%,
                           dfs 43%, random 49%, bfs 60%. The 197-expansion gap
                           between bfs (231) and hfs_oracle (34) is almost
                           entirely sterile subtree -- that gap IS the task.

  viability_auc            AUC of logit vs (delta < inf) over active slots.
                           Ranking sterile nodes last is a BINARY problem the
                           model must solve before anything else. At 0.5 the
                           model has learned nothing, whatever the TD loss says.

Together they separate "learned to avoid failure" from "got lucky on
expansions".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .tree import INF_DELTA, TreeInstance


def _average_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks, ties averaged (Mann-Whitney convention)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def viability_auc(
    logits: Sequence[float],
    deltas: Sequence[float],
) -> Optional[float]:
    """P(logit of a random VIABLE slot > logit of a random STERILE slot).

    Mann-Whitney U / (n_viable * n_sterile), ties counted as 0.5. Returns None
    when the beam is all-viable or all-sterile (no comparison exists) so callers
    can skip rather than average a fabricated 0.5 in.

    0.5 = the model cannot tell a dead subtree from a live one.
    1.0 = it ranks every viable candidate above every sterile one.
    """
    viable = [i for i, d in enumerate(deltas) if d != INF_DELTA]
    sterile = [i for i, d in enumerate(deltas) if d == INF_DELTA]
    if not viable or not sterile:
        return None
    ranks = _average_ranks(list(logits))
    r_viable = sum(ranks[i] for i in viable)
    n_v, n_s = len(viable), len(sterile)
    u = r_viable - n_v * (n_v + 1) / 2.0
    return u / (n_v * n_s)


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Rank correlation. None when either side is constant (undefined)."""
    n = len(a)
    if n < 2:
        return None
    ra, rb = _average_ranks(list(a)), _average_ranks(list(b))
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def top1_oracle_agreement(logits: Sequence[float], deltas: Sequence[float]) -> float:
    """P[argmax logits == argmin delta] over the active slots."""
    if not logits:
        return 0.0
    return float(
        max(range(len(logits)), key=lambda i: logits[i])
        == min(range(len(deltas)), key=lambda i: deltas[i])
    )


def spearman_logits_vs_delta(
    logits: Sequence[float], deltas: Sequence[float]
) -> Optional[float]:
    """Correlation of the model's ranking with -delta on one beam.

    Sterile slots are INF_DELTA; they are kept (ranking them last is the point),
    with inf mapped to a finite sentinel above every finite delta so the rank
    order is well defined.
    """
    finite = [d for d in deltas if d != INF_DELTA]
    hi = (max(finite) + 1.0) if finite else 1.0
    d = [hi if x == INF_DELTA else x for x in deltas]
    return spearman([-x for x in logits], d)


def beam_metrics(
    instance: TreeInstance,
    beam: Sequence[int],
    logits: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Per-state ranking diagnostics. Averaged over states by the caller.

    CENSORED slots (fix 1A) are excluded from every judged metric: their
    delta=inf is a generation artifact, so counting them as sterile would score
    the model against an invented label. They are reported separately as
    beam_censored_frac.
    """
    keep = [k for k, v in enumerate(beam) if not instance.censored[v]]
    deltas = [instance.delta[beam[k]] for k in keep]
    lg = [logits[k] for k in keep]
    n_all = len(beam)
    return {
        "viability_auc": viability_auc(lg, deltas),
        "top1_oracle_agreement": (
            top1_oracle_agreement(lg, deltas) if keep else 0.0
        ),
        "spearman_logits_vs_delta": spearman_logits_vs_delta(lg, deltas),
        "beam_sterile_frac": (
            sum(1 for d in deltas if d == INF_DELTA) / n_all if n_all else 0.0
        ),
        "beam_censored_frac": (
            (n_all - len(keep)) / n_all if n_all else 0.0
        ),
    }


def mean_ignoring_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """None means 'undefined here', not 'zero' -- never average it in."""
    vs = [v for v in values if v is not None]
    return (sum(vs) / len(vs)) if vs else None


# ------------------------------- Q* reference for critic calibration (F4) ----
#
# EVICTION IS NOT STERILE-SPECIFIC. push_vector dumps the whole unexpanded beam
# into R regardless of what v was, so:
#
#     eviction  <=>  (expanded != argmin)  AND  (the beam binds)
#
# Sterile expansions are the most COMMON cause, not the mechanism. That is why
# hfs_oracle has 0 evictions despite a 13.2%-sterile beam (it always expands the
# argmin), and why dfs has 0 at F=32 despite 43% sterile expansions (|B u R| <= F,
# so refill takes everything back).
#
# The consequence for calibration is an ASYMMETRY:
#
#     Q*(s, argmin)      = -delta(s)        EXACT   (conditions below)
#     Q*(s, v != argmin) < -1 - delta(s)    STRICTLY worse than the naive value,
#                                           by the expected recovery wait, which
#                                           has no closed form.
#
# So the naive Q from delta is exact on the argmin slot and an OPTIMISTIC UPPER
# BOUND everywhere else. Split F4 on argmin, NOT on sterility: viable non-argmin
# slots evict too, so pooling them into a "clean" R^2 would still punish a
# correct critic.


def best_child_index(instance: TreeInstance, v: int) -> Optional[int]:
    """Index in ch(v) of the first delta-decreasing child, or None if v is a goal
    or sterile."""
    d = instance.delta[v]
    if d == INF_DELTA or d == 0.0:
        return None
    for i, c in enumerate(instance.children[v]):
        if instance.delta[c] == d - 1.0:
            return i
    return None


def q_star_argmin_is_exact(instance: TreeInstance, v: int, fringe_size: int) -> bool:
    """Is Q*(s, v) == -delta(s) exact for the argmin slot v at this F?

    Expanding the argmin puts ch(v)[:F] in the new beam and OVERFLOWS the rest to
    R. The chain continues for free only if v's delta-decreasing child lands in
    the beam, i.e. its index is < F. Otherwise even the argmin pays a recovery
    wait and Q*(s,v) < -delta(s).

    This is the PER-STATE condition. The per-instance `F >= b_max` test is far
    too conservative: b_max is driven by a few high-branching nodes, while the
    good child almost always sits early in the child order. Measured -- at F=8,
    100% of viable internal nodes satisfy this on CC_2_3_4__pl_7 (b_max 8),
    CC_3_3_3__pl_4 (b_max 21), Grapevine_4__pl_2 (b_max 20) and SC_R_10_10__pl_7
    (b_max 23), where the per-instance test would have rejected three of the four.
    At F=4 it still holds for 97-100% of nodes.
    """
    i = best_child_index(instance, v)
    return i is not None and i < int(fringe_size)


def q_star_naive(
    instance: TreeInstance,
    beam: Sequence[int],
    reservoir: Sequence[int],
    action: int,
) -> Optional[float]:
    """The delta-derived reference value for Q(s, beam[action]).

    argmin slot     -> -delta(s)        (exact where q_star_argmin_is_exact)
    non-argmin slot -> -1 - delta(s)    (an optimistic UPPER bound: the true
                                         value is lower by the eviction wait)

    None when the open set has no viable node (no reference exists).
    """
    d = -instance.v_star(beam, reservoir)      # = min delta over B u R
    if d == INF_DELTA:
        return None
    v = beam[action]
    return -d if instance.delta[v] == d else -1.0 - d


def is_argmin_action(
    instance: TreeInstance, beam: Sequence[int], reservoir: Sequence[int], action: int
) -> bool:
    d = -instance.v_star(beam, reservoir)
    return d != INF_DELTA and instance.delta[beam[action]] == d


def qstar_naive_residual(
    q: float,
    instance: TreeInstance,
    beam: Sequence[int],
    reservoir: Sequence[int],
    action: int,
) -> Optional[float]:
    """Q(s,v) - naive(s,v) for a NON-argmin v: how far below the optimistic bound
    the critic places the action.

    F4 panel B, and the paper's point: expect ~0 when the beam does not bind, and
    growing negative as |R| grows. A critic that learns this has learned the COST
    OF BEING WRONG -- something a delta-regressor structurally cannot represent,
    because it minimises prediction error and has no notion of what an error
    costs.
    """
    if is_argmin_action(instance, beam, reservoir, action):
        return None
    naive = q_star_naive(instance, beam, reservoir, action)
    return None if naive is None else q - naive


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def r2(pred: Sequence[float], truth: Sequence[float]) -> Optional[float]:
    """Coefficient of determination against the 1:1 line (not a refit)."""
    n = len(truth)
    if n < 2:
        return None
    mt = sum(truth) / n
    ss_res = sum((p - t) ** 2 for p, t in zip(pred, truth))
    ss_tot = sum((t - mt) ** 2 for t in truth)
    return None if ss_tot == 0 else 1.0 - ss_res / ss_tot
