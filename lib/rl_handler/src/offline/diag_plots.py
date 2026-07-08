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


def _drop_structural_nonresult(
    rows: List[Dict[str, object]], fringe: Optional[int]
) -> List[Dict[str, object]]:
    """Drop rows whose (problem, frame) group is a STRUCTURAL_NONRESULT: every
    regime's node_economy EXACTLY equal (a tie) AND fmax < F (the beam can't
    bind, so there is no real ordering to measure and the lines would flatten
    falsely). Mirrors collect_sensitivity._classify_ties. fringe None or fmax
    missing -> cannot classify -> keep (never silently drop)."""
    if fringe is None or not rows:
        return rows
    groups: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("problem"), r.get("frame"))].append(r)
    keep: List[Dict[str, object]] = []
    for g in groups.values():
        econ = [r.get("node_economy") for r in g]
        fmax = g[0].get("fmax")
        tie = len(set(econ)) == 1
        structural = tie and (fmax is not None) and (fmax < fringe)
        if not structural:
            keep.extend(g)
    return keep


def _plot_metric_two_panel(
    diag_rows: Dict[str, List[Dict[str, object]]], out_path: Path,
    metric_key: str, ylabel: str, suptitle: str, fringe: Optional[int],
) -> bool:
    """One figure, TWO panels (train | eval=test), FOUR regime lines (dfs/bfs/
    hfs/random) = per-regime aggregate of `metric_key`
    (mean over problems, None dropped) vs frame. structural_nonresult groups
    excluded. Returns True if any line was drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
    any_line = False
    ftag = f" | F={fringe}" if fringe is not None else ""
    for ax, split in zip(axes, ("train", "test")):
        rows = _drop_structural_nonresult(diag_rows.get(split, []), fringe)
        frames, series = _aggregate(rows, metric_key)
        for rg, ys in series.items():
            xs = [f for f, y in zip(frames, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, yv, marker="o", label=rg)
                any_line = True
        eval_tag = "eval (test)" if split == "test" else "train"
        ax.set_title(f"{eval_tag}{ftag}")
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="regime (agg over problems)")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return any_line


def _iqm_band(values: List[float]):
    """(iqm, lo, hi): iqm = mean of the interquartile (p25–p75) values; the band
    is iqm ± std computed OVER THOSE SAME interquartile values ("iqm ± iqr-std").
    None if empty; a zero-width band when too few (<4) to trim a quartile."""
    ys = sorted(float(v) for v in values)
    n = len(ys)
    if n == 0:
        return None
    if n < 4:
        m = sum(ys) / n
        return (m, m, m)
    lo = n // 4
    mid = ys[lo:n - lo] or ys
    k = len(mid)
    mean = sum(mid) / k
    std = (sum((x - mean) ** 2 for x in mid) / k) ** 0.5
    return (mean, mean - std, mean + std)


def _plot_regret_two_panel(
    diag_rows: Dict[str, List[Dict[str, object]]], out_path: Path,
    fringe: Optional[int],
) -> bool:
    """TOP-1 REGRET figure: two panels (train | eval=test), four regime lines.
    Line = IQM of per-beam regret POOLED OVER BEAMS across the split's problems;
    band = IQM ± std over the interquartile beams. structural_nonresult groups
    excluded. Lower=better, 0=oracle. The band is OVER-BEAMS (not over-problems —
    at 1–2 binders an over-problems band is degenerate)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
    any_line = False
    ftag = f" | F={fringe}" if fringe is not None else ""
    for ax, split in zip(axes, ("train", "test")):
        rows = _drop_structural_nonresult(diag_rows.get(split, []), fringe)
        # pool the raw per-beam regrets by (frame, regime) across problems
        acc: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            beams = r.get("top1_regret_beams") or []
            if beams:
                acc[int(r["frame"])][str(r["regime"])].extend(float(b) for b in beams)
        frames = sorted(acc)
        regimes = sorted({rg for f in acc for rg in acc[f]})
        for rg in regimes:
            xs, ys, los, his = [], [], [], []
            for f in frames:
                band = _iqm_band(acc[f].get(rg, []))
                if band is None:
                    continue
                m, lo, hi = band
                xs.append(f); ys.append(m); los.append(lo); his.append(hi)
            if xs:
                ax.plot(xs, ys, marker="o", label=rg)
                ax.fill_between(xs, los, his, alpha=0.15)
                any_line = True
        eval_tag = "eval (test)" if split == "test" else "train"
        ax.set_title(f"{eval_tag}{ftag}")
        ax.set_xlabel("frame")
        ax.set_ylabel("top-1 regret d* (0=oracle, lower=better)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="regime (IQM±IQR-std over beams)")
    fig.suptitle(
        "TOP-1 REGRET (non-selecting) — IQM ± IQR-std OVER BEAMS, eligible slots; "
        "lower=better, 0=oracle", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return any_line


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

    # The three per-regime figures (train | eval panels, four regime lines, each
    # rendered whenever any binding data exists): does it RANK well (b), does good
    # rank convert to fewer expansions (a), and how far is the top pick from the
    # oracle (c, top-1 regret).
    rf_name = f"diag_rank_fidelity{suffix}.png"
    if _plot_metric_two_panel(
        diag_rows, out_dir / rf_name, "rank_spearman",
        "Spearman(score, -d*)",
        "RANK FIDELITY (non-selecting) — Spearman(score, -d*) per regime, "
        "eligible slots (finite d*)",
        fringe,
    ):
        written.append(rf_name)

    ex_name = f"diag_expansions{suffix}.png"
    if _plot_metric_two_panel(
        diag_rows, out_dir / ex_name, "node_economy_ratio",
        "expansions / optimal (lower=better)",
        "NODE ECONOMY (non-selecting) — expansions / optimal per regime",
        fringe,
    ):
        written.append(ex_name)

    rg_name = f"diag_top1_regret{suffix}.png"
    if _plot_regret_two_panel(diag_rows, out_dir / rg_name, fringe):
        written.append(rg_name)
    return written


def plot_fringe_occupancy_histogram(occ_diag, out_path, fringe, pad_closed):
    """Stacked bar chart of model-input fringe occupancy (n_fringe), by regime.

    x = occupancy 1..F, y = #fringes at that occupancy, stacked one colour per
    regime. With pad_closed=True the mass concentrates at F (padding fills the
    beam); with pad_closed=False it spreads low (the natural, mostly-sparse
    distribution). `occ_diag` is a tree_env.OccupancyDiagnostics. Returns the path
    written, or None when there is nothing to plot.
    """
    F = int(fringe)
    hist = getattr(occ_diag, "hist", {}) or {}
    regimes = sorted(hist)
    if not regimes:
        return None
    # bins 1..F (occupancy 0 is only a transient empty-fringe artefact; keep the
    # readable 1..F range the brief asks for).
    xs = list(range(1, F + 1))
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(max(6.0, F * 0.22), 4.0))
    bottom = [0.0] * len(xs)
    for rg in regimes:
        h = hist[rg]
        ys = [h[k] if k < len(h) else 0 for k in xs]
        ax.bar(xs, ys, bottom=bottom, width=0.9, label=rg)
        bottom = [b + y for b, y in zip(bottom, ys)]
    ax.set_xlabel("fringe occupancy (n_fringe = open + closed-padding)")
    ax.set_ylabel("count of fringes")
    ax.set_title(f"Fringe occupancy (F={F}, pad_closed={pad_closed})")
    ax.axvline(F, color="k", linestyle=":", linewidth=1, alpha=0.6)
    ax.legend(title="regime", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
