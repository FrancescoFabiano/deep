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
    """Per-state ranking diagnostics. Averaged over states by the caller."""
    deltas = [instance.delta[v] for v in beam]
    return {
        "viability_auc": viability_auc(logits, deltas),
        "top1_oracle_agreement": top1_oracle_agreement(logits, deltas),
        "spearman_logits_vs_delta": spearman_logits_vs_delta(logits, deltas),
        "beam_sterile_frac": (
            sum(1 for d in deltas if d == INF_DELTA) / len(deltas) if deltas else 0.0
        ),
    }


def mean_ignoring_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """None means 'undefined here', not 'zero' -- never average it in."""
    vs = [v for v in values if v is not None]
    return (sum(vs) / len(vs)) if vs else None
