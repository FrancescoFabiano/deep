"""Sensitivity-sweep cell resolution (shared by offline_main + the runner).

Pure-python (stdlib only) so the runner can import it for ``--dry-run`` without
pulling torch.  A *cell* is one concrete training config along the four sweep
axes:

    gamma (float), lambda_ord (float), target_centering (str), fringe_size (int)

``offline_main`` runs exactly ONE cell per invocation (the single-cell contract);
this module is the shared expander both the runner and offline_main's
``--list-cells`` manifest emission use, so the two never drift.

Combine modes
-------------
- ``product``       : full Cartesian product of the four axis lists.
- ``one_at_a_time`` : the BASELINE is the first element of every list; we vary
  ONE axis at a time off that baseline (the union of single-axis variations),
  de-duplicated, baseline first.  This keeps the cell count linear in the axis
  lengths instead of multiplicative, which is what the sensitivity study wants
  (each non-baseline cell isolates one axis' effect).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

# Axis order is fixed everywhere (manifest, tags, dedup keys) so a cell's
# identity is stable across the runner, offline_main and the collector.
AXES = ("gamma", "lambda_ord", "target_centering", "fringe_size")
SWEEP_MODES = ("product", "one_at_a_time")


def _cell(gamma, lam, cen, F) -> Dict[str, object]:
    return {
        "gamma": float(gamma),
        "lambda_ord": float(lam),
        "target_centering": str(cen),
        "fringe_size": int(F),
    }


def _key(c: Dict[str, object]) -> tuple:
    return (c["gamma"], c["lambda_ord"], c["target_centering"], c["fringe_size"])


def resolve_cells(
    gammas: Sequence[float],
    lambdas: Sequence[float],
    centerings: Sequence[str],
    fringes: Sequence[int],
    mode: str = "one_at_a_time",
) -> List[Dict[str, object]]:
    """Expand the four axis lists into concrete cells under ``mode``.

    Order is deterministic and the result is de-duplicated (preserving first
    appearance).  Empty axis lists are rejected — every axis needs a baseline.
    """
    if mode not in SWEEP_MODES:
        raise ValueError(f"sweep mode must be one of {SWEEP_MODES}, got {mode!r}")
    lists = {"gamma": list(gammas), "lambda_ord": list(lambdas),
             "target_centering": list(centerings), "fringe_size": list(fringes)}
    for name, vals in lists.items():
        if not vals:
            raise ValueError(f"axis {name!r} is empty; every axis needs >=1 value")

    cells: List[Dict[str, object]] = []
    seen: set = set()

    def add(g, l, c, f) -> None:
        cell = _cell(g, l, c, f)
        k = _key(cell)
        if k not in seen:
            seen.add(k)
            cells.append(cell)

    if mode == "product":
        for g, l, c, f in itertools.product(
            lists["gamma"], lists["lambda_ord"],
            lists["target_centering"], lists["fringe_size"]
        ):
            add(g, l, c, f)
        return cells

    # one_at_a_time: baseline = first element of every axis, then single-axis
    # variations off that baseline (union, baseline first).
    g0, l0, c0, f0 = (lists["gamma"][0], lists["lambda_ord"][0],
                      lists["target_centering"][0], lists["fringe_size"][0])
    add(g0, l0, c0, f0)                       # baseline
    for g in lists["gamma"][1:]:
        add(g, l0, c0, f0)
    for l in lists["lambda_ord"][1:]:
        add(g0, l, c0, f0)
    for c in lists["target_centering"][1:]:
        add(g0, l0, c, f0)
    for f in lists["fringe_size"][1:]:
        add(g0, l0, c0, f)
    return cells


def cell_tag(cell: Dict[str, object]) -> str:
    """Filesystem-safe coordinate tag for a cell's output dir (no seed)."""
    g = f"{cell['gamma']:g}".replace(".", "p").replace("-", "m")
    l = f"{cell['lambda_ord']:g}".replace(".", "p").replace("-", "m")
    return f"g{g}_l{l}_{cell['target_centering']}_F{int(cell['fringe_size'])}"


def axis_of_variation(cell: Dict[str, object], baseline: Dict[str, object]) -> str:
    """Which single axis a cell varies vs the baseline ('baseline' if none, or
    'multi' if more than one differs — only possible under product mode)."""
    diff = [a for a in AXES if cell[a] != baseline[a]]
    if not diff:
        return "baseline"
    return diff[0] if len(diff) == 1 else "multi"
