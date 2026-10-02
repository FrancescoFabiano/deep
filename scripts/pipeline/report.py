"""Stage 4: results/results.csv -> report/tables/*.tex *.csv and report/figures/*.png.

Two aggregations of nodes expanded and total time, per split:
  common  IQM over the problems EVERY method solved (same set for all columns, so
          the reduction vs BFS is apples to apples);
  solved  IQM over each method's OWN solved problems, with that count (the set
          differs per column; it is what a method achieves on what it can solve).
Plus coverage (solved/total) and one per-instance table of nodes expanded (best in
bold). Figures per domain: coverage vs F, nodes vs F for both aggregations, nodes
vs BFS nodes. The reading of the numbers belongs in the paper, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Color follows the method, in a fixed order; BFS is the neutral baseline.
COLORS = {"BFS": "#52514e", "RL": "#2a78d6", "GNN_RL": "#eb6834", "GNN_Astar": "#1baf7a", "GNN_HFS": "#eda100"}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}
METRICS = {"nodes": "nodes_expanded", "time": "total_ms"}
UNITS = {"nodes": "nodes expanded", "time": "total time in ms"}


def run(cfg: Config) -> None:
    df = pd.read_csv(cfg.results_file)
    df["solved"] = df["status"] == "SOLVED"
    df["label"] = [m if F == 0 else f"{m}@{F}" for m, F in zip(df["method"], df["F"])]
    labels = sorted(df["label"].unique(), key=lambda s: (s != "BFS", s.split("@")[0], int(s.split("@")[1]) if "@" in s else 0))
    tables, figures = cfg.report_dir / "tables", cfg.report_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    for split, part in df.groupby("split"):
        _write(coverage(part, labels), tables / f"coverage_{split}", f"Solved problems ({split})", f"tab:coverage_{split}")
        for name, col in METRICS.items():
            _write(common_iqm(part, labels, col), tables / f"{name}_common_{split}",
                   f"IQM {UNITS[name]} on the problems solved by every method, reduction vs BFS ({split})",
                   f"tab:{name}_common_{split}")
            _write(solved_iqm(part, labels, col), tables / f"{name}_solved_{split}",
                   f"IQM {UNITS[name]} on each method's own solved problems (count in parentheses) ({split})",
                   f"tab:{name}_solved_{split}")
    _write(per_instance(df, labels), tables / "per_instance", "Nodes expanded per problem (best in bold)", "tab:per_instance")

    for domain, part in df.groupby("domain"):
        plot_vs_F(part, figures / f"{domain}_coverage_vs_F.png", metric="coverage")
        plot_vs_F(part, figures / f"{domain}_nodes_common_vs_F.png", metric="common")
        plot_vs_F(part, figures / f"{domain}_nodes_solved_vs_F.png", metric="solved")
        plot_vs_bfs(part, figures / f"{domain}_nodes_vs_bfs.png")
    print(f"[report] tables -> {tables}\n[report] figures -> {figures}")


# ---------------------------------------------------------------- tables
def coverage(df, labels) -> pd.DataFrame:
    g = df.groupby(["domain", "label"])["solved"].agg(["sum", "count"])
    cells = g.apply(lambda r: f"{int(r['sum'])}/{int(r['count'])}", axis=1).unstack("label")
    tot = df.groupby("label")["solved"].agg(["sum", "count"])
    cells.loc["all"] = [f"{int(tot.loc[l, 'sum'])}/{int(tot.loc[l, 'count'])}" if l in tot.index else "" for l in cells.columns]
    return cells.reindex(columns=[l for l in labels if l in cells.columns]).fillna("")


def common_iqm(df, labels, metric) -> pd.DataFrame:
    """One row per domain: IQM over the problems every method solved, reduction vs BFS."""
    rows = {}
    for domain, part in df.groupby("domain"):
        common = common_problems(part)
        sub = part[part["problem"].isin(common)]
        base = iqm(sub.loc[sub["label"] == "BFS", metric])
        row = {"n": str(len(common))}
        for label in labels:
            vals = sub.loc[sub["label"] == label, metric]
            if len(vals) == 0:
                continue
            v = iqm(vals)
            row[label] = f"{v:.0f}" if label == "BFS" or not base else f"{v:.0f} ({100 * (base - v) / base:+.0f}%)"
        rows[domain] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["n", *labels]).fillna("")


def solved_iqm(df, labels, metric) -> pd.DataFrame:
    """One row per domain: IQM over each method's own solved problems, with the count."""
    rows = {}
    for domain, part in df.groupby("domain"):
        row = {}
        for label in labels:
            vals = part.loc[(part["label"] == label) & part["solved"], metric]
            row[label] = f"{iqm(vals):.0f} ({len(vals)})" if len(vals) else "-- (0)"
        rows[domain] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=labels).fillna("")


def per_instance(df, labels) -> pd.DataFrame:
    idx = ["domain", "split", "problem"]
    nodes = df.pivot_table(index=idx, columns="label", values="nodes_expanded", aggfunc="first")
    nodes = nodes.where(df.pivot_table(index=idx, columns="label", values="solved", aggfunc="first").astype(bool))
    nodes = nodes.reindex(columns=[l for l in labels if l in nodes.columns])
    out = nodes.copy().astype(object)
    for i, row in nodes.iterrows():
        best = row.min()
        out.loc[i] = [("--" if pd.isna(v) else (f"\\textbf{{{int(v)}}}" if v == best else f"{int(v)}")) for v in row]
    return out


def iqm(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    lo, hi = np.percentile(v, [25, 75])
    return float(v[(v >= lo) & (v <= hi)].mean())


def common_problems(df) -> set:
    solved = df[df["solved"]].groupby("label")["problem"].agg(set)
    return set.intersection(*solved) if len(solved) == len(df["label"].unique()) and len(solved) else set()


def _write(df: pd.DataFrame, stem, caption: str, label: str) -> None:
    df.to_csv(stem.with_suffix(".csv"))
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    cell = lambda v: str(v) if str(v).startswith("\\") else esc(v)      # \textbf{...} cells are already LaTeX
    index_names = [n or "" for n in df.index.names]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" * len(index_names) + "r" * len(df.columns) + "}", r"\toprule",
             " & ".join(esc(c) for c in [*index_names, *df.columns]) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        lines.append(" & ".join([*(esc(i) for i in idx), *(cell(v) for v in row)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    stem.with_suffix(".tex").write_text("\n".join(lines))


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e1",
                         "axes.grid": True, "axes.axisbelow": True, "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 10, "legend.frameon": False})
    return plt


def _learned(df):
    return [m for m in COLORS if m != "BFS" and m in set(df["method"])]


def _value(rows, metric: str, common: set) -> float:
    """The y value of one (method, F) group of rows under the chosen aggregation."""
    if metric == "coverage":
        return 100 * rows["solved"].mean()
    if metric == "common":
        return iqm(rows.loc[rows["problem"].isin(common), "nodes_expanded"])
    return iqm(rows.loc[rows["solved"], "nodes_expanded"])           # solved: own solved set


def plot_vs_F(df, path, metric: str) -> None:
    """metric: coverage (solved %), common (IQM nodes, problems solved by all), solved (IQM nodes, own solved)."""
    plt = _style()
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.4), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        part = df[df["split"] == split]
        common = common_problems(part) if metric == "common" else set()
        ax.axhline(_value(part[part["method"] == "BFS"], metric, common), color=COLORS["BFS"], ls="--", lw=2, label="BFS")
        for method in _learned(part):
            pts = [(F, _value(g, metric, common)) for F, g in part[part["method"] == method].groupby("F")]
            ax.plot(*zip(*pts), color=COLORS[method], lw=2, marker="o", ms=7, label=method)
        n = f"n={len(common)} common" if metric == "common" else f"n={part['problem'].nunique()}"
        ax.set_title(f"{split}: {metric} ({n})", loc="left")
        Fs = sorted(part.loc[part["F"] > 0, "F"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(Fs)
        ax.set_xticklabels([str(x) for x in Fs])
        ax.set_xlabel("fringe size F")
        if metric != "coverage":
            ax.set_yscale("log")
    axes[0][0].set_ylabel("solved %" if metric == "coverage" else "IQM nodes expanded")
    axes[0][-1].legend(loc="best")
    fig.suptitle(df["domain"].iloc[0], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_vs_bfs(df, path) -> None:
    plt = _style()
    methods = _learned(df)
    if not methods:
        return
    bfs = df[(df["method"] == "BFS") & df["solved"]].set_index("problem")["nodes_expanded"]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.8 * len(methods), 3.6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        part = df[(df["method"] == method) & df["solved"] & df["problem"].isin(bfs.index)]
        for F, g in part.groupby("F"):
            for split, gg in g.groupby("split"):
                ax.scatter(bfs[gg["problem"]], gg["nodes_expanded"], s=40, marker=MARKERS.get(F, "o"),
                           facecolors=COLORS[method] if split == "train" else "none", edgecolors=COLORS[method],
                           linewidths=1.5, label=f"F={F} {split}")
        lim = [1, max(1, bfs.max(), part["nodes_expanded"].max()) * 1.5]
        ax.plot(lim, lim, color=COLORS["BFS"], ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(method, loc="left"); ax.set_xlabel("BFS nodes expanded")
    axes[0][0].set_ylabel("nodes expanded")
    axes[0][-1].legend(loc="lower right", fontsize=8, title="filled = train, hollow = test")
    fig.suptitle(df["domain"].iloc[0], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
