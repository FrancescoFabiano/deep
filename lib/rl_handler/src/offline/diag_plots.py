"""Diagnostic per-regime plots (Part 3 of the sensitivity surface).

NON-SELECTING: these visualise the diagnostic per-(regime × problem) rollouts,
which never feed model selection (selection = the deploy-faithful val eval).

Convention — NORMALIZED AGGREGATE so deep problems don't dominate:
  * node economy is plotted as  expansions / optimal-expansions  per problem
    (optimal = shallowest-goal depth from the tree), averaged across problems;
  * reward (the -1-stream return) is plotted as  return / optimal-expansions
    per problem, averaged across problems.
One lineplot per split (train / test), ONE LINE PER REGIME = that aggregate over
problems at each checkpoint frame. Raw per-problem metrics stay in the CSV
tables; only the normalized ratios are aggregated here, and only at plot time.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _aggregate(rows: List[Dict[str, object]], ratio_key: str):
    """frame -> regime -> mean(ratio over problems) (None ratios dropped)."""
    acc: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(ratio_key)
        if v is None:
            continue
        acc[int(r["frame"])][str(r["regime"])].append(float(v))
    frames = sorted(acc)
    regimes = sorted({rg for f in acc for rg in acc[f]})
    series: Dict[str, List[Optional[float]]] = {}
    for rg in regimes:
        series[rg] = [
            (sum(acc[f][rg]) / len(acc[f][rg])) if acc[f].get(rg) else None
            for f in frames
        ]
    return frames, series


def _plot_split(rows: List[Dict[str, object]], out_path: Path, split: str,
                fringe: Optional[int]) -> bool:
    if not rows:
        return False
    ftag = f" | F={fringe}" if fringe is not None else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
    panels = [
        ("node_economy_ratio", "node economy = expansions / optimal (lower=better)"),
        ("ret_ratio", "reward = return / optimal (higher=better)"),
    ]
    any_line = False
    for ax, (key, title) in zip(axes, panels):
        frames, series = _aggregate(rows, key)
        for rg, ys in series.items():
            xs = [f for f, y in zip(frames, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, yv, marker="o", label=rg)
                any_line = True
        ax.set_title(f"{split}{ftag} | {title}")
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="regime (agg over problems)")
    fig.suptitle(f"DIAGNOSTIC (non-selecting) — {split} split", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return any_line


def plot_diag_curves(
    diag_rows: Dict[str, List[Dict[str, object]]],
    out_dir: Path,
    fringe: Optional[int] = None,
) -> List[str]:
    """Render one normalized-aggregate diagnostic plot per non-empty split.
    Returns the list of written filenames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    suffix = f"_fringe{fringe}" if fringe is not None else ""
    for split in ("train", "test"):
        rows = diag_rows.get(split, [])
        name = f"diag_per_regime_{split}{suffix}.png"
        if _plot_split(rows, out_dir / name, split, fringe):
            written.append(name)
    return written
