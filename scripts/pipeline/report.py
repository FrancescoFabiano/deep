"""Stage 4: results/results.csv -> report/tables/*.tex *.csv and report/figures/*.png.

Tables (per split): coverage (solved/total), nodes and time (IQM on the problems
every method solved, with the reduction vs BFS), and one per-instance table of
nodes expanded (best in bold). Figures per domain: coverage vs F, IQM nodes vs F,
nodes vs BFS nodes. The reading of the numbers belongs in the paper, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Color follows the method, in a fixed order; BFS is the neutral baseline.
COLORS = {"BFS": "#52514e", "RL": "#2a78d6", "GNN_RL": "#eb6834", "GNN_Astar": "#1baf7a", "GNN_HFS": "#eda100"}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}


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
        _write(common_iqm(part, labels, "nodes_expanded"), tables / f"nodes_{split}",
               f"IQM nodes expanded on commonly solved problems, reduction vs BFS ({split})", f"tab:nodes_{split}")
        _write(common_iqm(part, labels, "total_ms"), tables / f"time_{split}",
               f"IQM total time in ms on commonly solved problems, reduction vs BFS ({split})", f"tab:time_{split}")
    _write(per_instance(df, labels), tables / "per_instance", "Nodes expanded per problem (best in bold)", "tab:per_instance")

    for domain, part in df.groupby("domain"):
        plot_vs_F(part, figures / f"{domain}_coverage_vs_F.png", metric="coverage")
        plot_vs_F(part, figures / f"{domain}_nodes_vs_F.png", metric="nodes")
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
    rows = {}
    for domain, part in df.groupby("domain"):
        common = _common_problems(part)
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


def per_instance(df, labels) -> pd.DataFrame:
    nodes = df.pivot_table(index=["domain", "split", "problem"], columns="label", values="nodes_expanded", aggfunc="first")
    nodes = nodes.where(df.pivot_table(index=["domain", "split", "problem"], columns="label", values="solved", aggfunc="first").astype(bool))
    nodes = nodes.reindex(columns=[l for l in labels if l in nodes.columns])
    out = nodes.copy().astype(object)
    for idx, row in nodes.iterrows():
        best = row.min()
        out.loc[idx] = [("--" if pd.isna(v) else (f"\\textbf{{{int(v)}}}" if v == best else f"{int(v)}")) for v in row]
    return out


def iqm(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    lo, hi = np.percentile(v, [25, 75])
    return float(v[(v >= lo) & (v <= hi)].mean())


def _common_problems(df) -> set:
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


def plot_vs_F(df, path, metric: str) -> None:
    plt = _style()
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.4), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        part = df[df["split"] == split]
        common = _common_problems(part) if metric == "nodes" else None
        bfs = part[part["method"] == "BFS"]
        base = 100 * bfs["solved"].mean() if metric == "coverage" else iqm(bfs[bfs["problem"].isin(common)]["nodes_expanded"])
        ax.axhline(base, color=COLORS["BFS"], ls="--", lw=2, label="BFS")
        for method in _learned(part):
            pts = []
            for F, g in part[part["method"] == method].groupby("F"):
                if metric == "coverage":
                    pts.append((F, 100 * g["solved"].mean()))
                else:
                    pts.append((F, iqm(g[g["problem"].isin(common)]["nodes_expanded"])))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=COLORS[method], lw=2, marker="o", ms=7, label=method)
        n = part.groupby("label")["problem"].nunique().max()
        ax.set_title(f"{split}: {metric} ({'n=' + str(len(common)) + ' common' if common is not None else f'n={n}'})", loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(part.loc[part["F"] > 0, "F"].unique()))
        ax.set_xticklabels([str(x) for x in sorted(part.loc[part["F"] > 0, "F"].unique())])
        ax.set_xlabel("fringe size F")
        if metric == "nodes":
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
