"""
Fringe-size comparison plots, per domain.

For each method (heuristic + strictness, WITHOUT fringe in the label) compares the
search metrics across fringe sizes (BFS=0, fringe_16, fringe_32, ...). Grouped bar
charts, one bar per fringe size, one subplot per split variant (train / test / all).

Two modes:
  - "all"    : each fringe's metrics computed over that fringe's OWN solved set,
               plus a per-fringe success rate.
  - "common" : metrics computed over the INTERSECTION of instances solved by every
               fringe of that method (fair per-instance comparison; no success rate,
               it is 1.0 by construction).

Output: {results_dir}/analysis/fringe_comparison/fringe_cmp_{metric}_{mode}.png
  4 PNGs for "all" (nodes, planlength, time, success_rate)
  3 PNGs for "common" (nodes, planlength, time)  ->  7 total per domain.

Reuses the per-instance CSV parsing pattern from plot_by_pl.py (read_instances, num).

Usage:
    python3 scripts/rl_exp/plot_fringe_comparison.py <results_dir>
where <results_dir> is the domain-scoped results dir,
e.g. combined_results/batch_separated_dqn_strict/CC
"""

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Lower is better for all of these; SuccessRate (higher is better) handled separately.
METRICS_ALL = ["NodesExpanded", "PlanLength", "TotalExecutionTime", "SuccessRate"]
METRICS_COMMON = ["NodesExpanded", "PlanLength", "TotalExecutionTime"]
SPLITS = ["train", "test", "all"]

METRIC_SHORT = {
    "NodesExpanded": "nodes",
    "PlanLength": "planlength",
    "TotalExecutionTime": "time",
    "SuccessRate": "success_rate",
}

# Distinct marker per heuristic in the trend plots (cycled if >10 heuristics).
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
# Metrics whose trend y-axis uses a log scale (symlog fallback if any value <= 0).
LOG_METRICS = ("NodesExpanded", "TotalExecutionTime")


def num(s):
    """Parse a metric cell to float; return None for blank / '-' / non-numeric."""
    s = (s or "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_instances(csv_path: Path):
    """Yield per-instance dict rows, stopping at the blank line / stats header."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = None
        for fields in reader:
            if not fields or all(c.strip() == "" for c in fields):
                break
            if header is None:
                if fields[0].strip() != "File":
                    return []  # not the expected per-instance layout
                header = [c.strip() for c in fields]
                continue
            if fields[0].strip() == "Solved":
                break
            rows.append(dict(zip(header, fields)))
    return rows


def parse_method_and_fringe(csv_path: Path):
    """(split, method, fringe) from filename + parent dir, or None if not a summary CSV.

    Method label carries approach + strictness but NOT fringe:
      train_RL-C_PG_strict.csv under fringe_32/ -> ("train", "RL-C_PG_strict", 32)
      train_BFS_strict.csv      under bfs/       -> ("train", "BFS_strict",   0)
    """
    stem = csv_path.stem
    if stem.startswith("train_"):
        split, rest = "train", stem[len("train_"):]
    elif stem.startswith("test_"):
        split, rest = "test", stem[len("test_"):]
    else:
        return None

    if rest.endswith("_nostrict"):
        strict, approach = "nostrict", rest[:-len("_nostrict")]
    elif rest.endswith("_strict"):
        strict, approach = "strict", rest[:-len("_strict")]
    else:
        strict, approach = None, rest

    method = f"{approach}_{strict}" if strict else approach

    parent = csv_path.parent.name
    if parent.startswith("fringe_"):
        try:
            fringe = int(parent[len("fringe_"):])
        except ValueError:
            return None
    elif parent == "bfs":
        fringe = 0
    else:
        return None

    return split, method, fringe


def load_records(results_dir: Path):
    """Flatten every per-instance row across all summary CSVs into records."""
    records = []
    for csv_path in sorted(results_dir.rglob("*.csv")):
        if csv_path.name == "aggregate.csv":
            continue
        if "analysis" in csv_path.parts:
            continue
        parsed = parse_method_and_fringe(csv_path)
        if parsed is None:
            continue
        split, method, fringe = parsed
        for row in read_instances(csv_path):
            records.append({
                "method": method,
                "split": split,
                "fringe": fringe,
                "instance": row.get("File", "").strip(),
                "solved": row.get("GoalFound", "").strip() == "Yes",
                "NodesExpanded": num(row.get("NodesExpanded")),
                "TotalExecutionTime": num(row.get("TotalExecutionTime")),
                "PlanLength": num(row.get("PlanLength")),
            })
    return records


def _mean_std(pool, metric):
    vals = [r[metric] for r in pool if r[metric] is not None]
    if not vals:
        return (None, None)
    return (float(np.mean(vals)), float(np.std(vals)))


def aggregate(records, split_variant, mode):
    """method -> fringe -> {metric: (value, err)} for one split variant + mode."""
    subset = [r for r in records
              if split_variant == "all" or r["split"] == split_variant]
    result = {}
    for method in sorted(set(r["method"] for r in subset)):
        mrecs = [r for r in subset if r["method"] == method]
        fringes = sorted(set(r["fringe"] for r in mrecs))
        result[method] = {}

        common = None
        if mode == "common":
            solved_sets = {
                fr: set(r["instance"] for r in mrecs
                        if r["fringe"] == fr and r["solved"])
                for fr in fringes
            }
            if len(fringes) <= 1:
                common = solved_sets[fringes[0]] if fringes else set()
            else:
                common = set.intersection(*(solved_sets[fr] for fr in fringes))

        for fr in fringes:
            frecs = [r for r in mrecs if r["fringe"] == fr]
            entry = {}
            if mode == "all":
                n_total = len(frecs)
                n_solved = sum(1 for r in frecs if r["solved"])
                entry["SuccessRate"] = (
                    (n_solved / n_total) if n_total else None, None)
                pool = [r for r in frecs if r["solved"]]
            else:  # common
                pool = [r for r in frecs
                        if r["solved"] and r["instance"] in common]
            for metric in ("NodesExpanded", "PlanLength", "TotalExecutionTime"):
                entry[metric] = _mean_std(pool, metric)
            result[method][fr] = entry
    return result


def order_methods(methods, agg_all, metric):
    """Sort methods by mean metric across fringes on the 'all' split.

    best = lowest, except SuccessRate where best = highest. Methods with no data
    for this metric on the 'all' split go last (alphabetical)."""
    higher_better = metric == "SuccessRate"
    keyed = []
    for m in methods:
        vals = [
            entry.get(metric, (None, None))[0]
            for entry in agg_all.get(m, {}).values()
            if entry.get(metric, (None, None))[0] is not None
        ]
        keyed.append((m, float(np.mean(vals)) if vals else None))
    with_data = sorted(
        [(m, k) for m, k in keyed if k is not None],
        key=lambda x: x[1], reverse=higher_better)
    without = sorted(m for m, k in keyed if k is None)
    return [m for m, _ in with_data] + without


def make_figure(records, methods, fringe_sizes, metric, mode, domain, out):
    aggs = {sv: aggregate(records, sv, mode) for sv in SPLITS}
    ordered = order_methods(methods, aggs["all"], metric)

    n_fringes = len(fringe_sizes)
    bar_width = 0.8 / n_fringes
    offsets = np.arange(n_fringes) * bar_width - (n_fringes - 1) * bar_width / 2
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        1, 3, figsize=(max(18, len(ordered) * 1.8), 7), sharey=True)

    x = np.arange(len(ordered))
    for i, sv in enumerate(SPLITS):
        ax = axes[i]
        agg = aggs[sv]
        plotted = 0
        for fi, fr in enumerate(fringe_sizes):
            xpos, values, errs = [], [], []
            for mi, m in enumerate(ordered):
                entry = agg.get(m, {}).get(fr)
                if not entry:
                    continue
                v, e = entry.get(metric, (None, None))
                if v is None:
                    continue
                xpos.append(x[mi] + offsets[fi])
                values.append(v)
                errs.append(e if e is not None else 0.0)
            if not xpos:
                continue
            plotted += len(xpos)
            color = cmap(fi % 10)
            if metric == "SuccessRate":
                ax.bar(xpos, values, bar_width, color=color)
            else:
                ax.bar(xpos, values, bar_width, yerr=errs, capsize=3, color=color)

        ax.set_title(sv)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel(metric)
        if i == 1:
            ax.set_xlabel("Method")
        if plotted == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, alpha=0.6)

    legend_handles = [
        Patch(color=cmap(fi % 10),
              label=("BFS" if fr == 0 else f"F={fr}"))
        for fi, fr in enumerate(fringe_sizes)
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.suptitle(f"{metric} by fringe size ({mode} solved) — {domain}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f"fringe_cmp_{METRIC_SHORT[metric]}_{mode}.png"
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)
    print(f"[OK] saved {fname} ({len(ordered)} methods, {n_fringes} fringes)")


def _median_iqr(pool, metric):
    vals = [r[metric] for r in pool if r[metric] is not None]
    if not vals:
        return (None, None, None)
    return (float(np.median(vals)),
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)))


def aggregate_percentile(records, split_variant, mode):
    """method -> fringe -> {metric: (median, q25, q75)} for one split variant + mode.

    Same solved/common-set logic as aggregate(), but carries median + IQR instead
    of mean + std. SuccessRate is stored as (ratio, None, None) in 'all' mode."""
    subset = [r for r in records
              if split_variant == "all" or r["split"] == split_variant]
    result = {}
    for method in sorted(set(r["method"] for r in subset)):
        mrecs = [r for r in subset if r["method"] == method]
        fringes = sorted(set(r["fringe"] for r in mrecs))
        result[method] = {}

        common = None
        if mode == "common":
            solved_sets = {
                fr: set(r["instance"] for r in mrecs
                        if r["fringe"] == fr and r["solved"])
                for fr in fringes
            }
            if len(fringes) <= 1:
                common = solved_sets[fringes[0]] if fringes else set()
            else:
                common = set.intersection(*(solved_sets[fr] for fr in fringes))

        for fr in fringes:
            frecs = [r for r in mrecs if r["fringe"] == fr]
            entry = {}
            if mode == "all":
                n_total = len(frecs)
                n_solved = sum(1 for r in frecs if r["solved"])
                entry["SuccessRate"] = (
                    (n_solved / n_total) if n_total else None, None, None)
                pool = [r for r in frecs if r["solved"]]
            else:  # common
                pool = [r for r in frecs
                        if r["solved"] and r["instance"] in common]
            for metric in ("NodesExpanded", "PlanLength", "TotalExecutionTime"):
                entry[metric] = _median_iqr(pool, metric)
            result[method][fr] = entry
    return result


def make_trend_figure(records, fringe_sizes, metric, mode, domain, out):
    """2x3 (or 1x3) trend figure: rows = strictness, cols = split; lines per heuristic.

    Median metric vs fringe size with IQR bands. BFS is a horizontal dashed
    reference (single fringe=0 point). Log y-scale for nodes / time."""
    methods = sorted(set(r["method"] for r in records))
    heuristics = sorted(set(
        m[:-len("_strict")] if m.endswith("_strict") else m[:-len("_nostrict")]
        for m in methods))
    strictnesses = [s for s in ("strict", "nostrict")
                    if any(m.endswith("_" + s) for m in methods)]

    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(heuristics))))
    color_map = {h: palette[i] for i, h in enumerate(heuristics)}
    marker_map = {h: MARKERS[i % len(MARKERS)] for i, h in enumerate(heuristics)}

    aggs = {sv: aggregate_percentile(records, sv, mode) for sv in SPLITS}

    # Decide the y-scale once for the whole figure (sharey='row' forbids mixing).
    use_symlog = False
    if metric in LOG_METRICS:
        allvals = [
            entry.get(metric, (None,))[0]
            for a in aggs.values() for byfr in a.values() for entry in byfr.values()
            if entry.get(metric, (None,))[0] is not None
        ]
        use_symlog = any(v <= 0 for v in allvals)

    nrows = len(strictnesses)
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 5 * nrows), sharey="row")
    axes = np.atleast_2d(axes)

    for row, strictness in enumerate(strictnesses):
        for col, sv in enumerate(SPLITS):
            ax = axes[row, col]
            a = aggs[sv]
            plotted = 0
            for heur in heuristics:
                byfr = a.get(f"{heur}_{strictness}")
                if not byfr:
                    continue
                if heur == "BFS":
                    e = byfr.get(0)
                    med = e.get(metric, (None,))[0] if e else None
                    if med is not None:
                        ax.axhline(y=med, color="gray", linestyle="--",
                                   linewidth=1.2, label="BFS", alpha=0.7)
                        plotted += 1
                    continue
                xs, meds, q25s, q75s = [], [], [], []
                for f in sorted(fr for fr in byfr if fr != 0):
                    med, lo, hi = byfr[f].get(metric, (None, None, None))
                    if med is None:
                        continue
                    xs.append(f)
                    meds.append(med)
                    q25s.append(lo if lo is not None else med)
                    q75s.append(hi if hi is not None else med)
                if not xs:
                    continue
                plotted += 1
                ax.plot(xs, meds, marker=marker_map[heur], color=color_map[heur],
                        label=heur, linewidth=1.5, markersize=6)
                if metric != "SuccessRate":
                    ax.fill_between(xs, q25s, q75s, color=color_map[heur], alpha=0.15)

            if metric in LOG_METRICS:
                if use_symlog:
                    ax.set_yscale("symlog", linthresh=1)
                else:
                    ax.set_yscale("log")

            ax.set_xticks(fringe_sizes)
            ax.set_xticklabels(["BFS" if f == 0 else str(f) for f in fringe_sizes])
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(sv)
            if col == 0:
                ax.set_ylabel(f"{strictness}\n{metric}")
            if row == nrows - 1:
                ax.set_xlabel("Fringe size")
            if plotted == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, alpha=0.6)

    # One shared legend (deduped across subplots so every heuristic + BFS appears).
    handles, labels, seen = [], [], set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.12, 0.5), fontsize=9)

    fig.suptitle(f"{metric} trend by fringe size ({mode} solved) — {domain}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])

    fname = f"fringe_trend_{METRIC_SHORT[metric]}_{mode}.png"
    # bbox_inches='tight' so the out-of-axes legend (x=1.12) is not clipped.
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {fname} ({len(heuristics)} heuristics, "
          f"{len(strictnesses)} strictness groups, {len(fringe_sizes)} fringes)")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: plot_fringe_comparison.py <results_dir>")
    results_dir = Path(sys.argv[1])
    domain = results_dir.name
    out = results_dir / "analysis" / "fringe_comparison"
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir)
    if not records:
        sys.exit(f"[plot_fringe_comparison] no per-instance records under {results_dir}")

    methods = sorted(set(r["method"] for r in records))
    fringe_sizes = sorted(set(r["fringe"] for r in records))

    for mode, metrics in (("all", METRICS_ALL), ("common", METRICS_COMMON)):
        for metric in metrics:
            make_figure(records, methods, fringe_sizes, metric, mode, domain, out)

    print(f"[OK] fringe comparison complete for domain={domain}")

    for mode, metrics in (("all", METRICS_ALL), ("common", METRICS_COMMON)):
        for metric in metrics:
            make_trend_figure(records, fringe_sizes, metric, mode, domain, out)

    print(f"[OK] fringe trend plots complete for domain={domain}")


if __name__ == "__main__":
    main()
