"""Diagnostic plots for one fringe run's telemetry.jsonl. Reads ONLY what is on disk.

NEUTRAL by design: titles state what is plotted (quantity, F, n), axes are labelled,
and nothing here interprets the numbers. The figures are evidence; the reading of them
lives in the report, not on the plot.

Writes, into <telemetry_dir>/figures/:
  selection_audit.png     the SELECTION metric vs checkpoint (held-out top1, or
                          coverage on the primary path), the smoothing window, the
                          selected checkpoint, and baselines as reference lines.
  coverage_vs_checkpoint.png
  regret_vs_checkpoint.png
  training_health.png     td_loss + grad_norm (twin axis), regret below.
  heldout_top1_bars.png   model vs each baseline on the SAME held-out frontiers
                          (matched-n) -- only on the held-out-trajectory path.

Frontier-size distribution is a separate, dataset-level figure (plot_frontier_dist).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_STYLE = {           # colour-blind safe, distinguishable in greyscale
    "hfs_oracle": ("#000000", "-.", "hfs_oracle"),
    "random":     ("#D55E00", "--", "random"),
    "bfs":        ("#0072B2", ":",  "bfs"),
    "dfs":        ("#009E73", (0, (3, 1, 1, 1)), "dfs"),
}
RL_C = "#CC79A7"
WINDOW = 3


def load(path: Path):
    rows = [json.loads(l) for l in path.open()]
    val = sorted([r for r in rows if r["split"] == "val" and r["step"] > 0],
                 key=lambda r: r["step"])
    base = {r["split"].split(":", 1)[1]: r for r in rows if r["step"] == -1}
    return val, base


def _win_mean(ys, i, window=WINDOW):
    lo = max(0, i - window + 1)
    w = ys[lo:i + 1]
    return sum(w) / len(w)


def _baselines(ax, base, field):
    for name, (c, ls, label) in BASE_STYLE.items():
        if name not in base or base[name].get(field) is None:
            continue
        ax.axhline(base[name][field], color=c, ls=ls, lw=1.5,
                   label=f"{label}={base[name][field]:.3g}", zorder=1)


def fig_selection_audit(val, base, out: Path, F: int, metric: str, field: str):
    steps = [r["step"] for r in val]
    ys = [r[field] for r in val]
    win = [_win_mean(ys, i) for i in range(len(ys))]
    best_i = max(range(len(win)), key=lambda i: (win[i], -steps[i]))
    n_ho = val[-1].get("heldout_n")

    fig, ax = plt.subplots(figsize=(9, 5))
    _baselines(ax, base, field)
    ax.plot(steps, ys, "o-", color=RL_C, lw=1.6, ms=4, label="model (per checkpoint)")
    ax.plot(steps, win, "-", color=RL_C, lw=2.4, alpha=.5,
            label=f"model ({WINDOW}-checkpoint window mean)")
    ax.plot(steps[best_i], win[best_i], "*", ms=18, color=RL_C, mec="black", mew=.8,
            zorder=5, label=f"selected (step {steps[best_i]})")
    ax.set_xlabel("training step")
    ax.set_ylabel(metric)
    n = f", n={n_ho} held-out frontiers" if n_ho else ""
    ax.set_title(f"Selection metric vs checkpoint  (F={F}{n})", fontsize=11)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="best", framealpha=.95)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_line(val, base, out: Path, F: int, field: str, ylabel: str):
    steps = [r["step"] for r in val]
    ys = [r.get(field) for r in val]
    fig, ax = plt.subplots(figsize=(9, 5))
    _baselines(ax, base, field)
    ax.plot(steps, ys, "o-", color=RL_C, lw=1.6, ms=4, label="model (per checkpoint)")
    finite = [y for y in ys if y is not None]
    if finite:
        ax.axhline(st.mean(finite), color=RL_C, lw=1.0, alpha=.5,
                   label=f"model mean={st.mean(finite):.3g}")
    ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs checkpoint  (F={F})", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="best", framealpha=.95)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_health(val, out: Path, F: int):
    steps = [r["step"] for r in val]
    td = [r.get("td_loss") for r in val]
    gn = [r.get("grad_norm") for r in val]
    reg = [r.get("regret_mean_lower_bound") for r in val]
    if not any(td):
        return
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.semilogy(steps, td, "o-", color="#0072B2", lw=1.6, ms=4, label="td_loss (log)")
    ax1.set_ylabel("td_loss", color="#0072B2"); ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax2 = ax1.twinx()
    ax2.plot(steps, gn, "s--", color="#009E73", lw=1.4, ms=4, label="grad_norm")
    ax2.set_ylabel("grad_norm", color="#009E73"); ax2.tick_params(axis="y", labelcolor="#009E73")
    ax1.set_title(f"Training health: td_loss, grad_norm  (F={F})", fontsize=11)
    ax1.grid(alpha=.25)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax3.plot(steps, reg, "o-", color=RL_C, lw=1.6, ms=4)
    ax3.set_ylabel("regret"); ax3.set_xlabel("training step"); ax3.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_top1_bars(val, base, out: Path, F: int):
    if "heldout_top1" not in val[-1]:
        return
    if not any("heldout_top1" in b for b in base.values()):
        return
    steps = [r["step"] for r in val]
    ys = [r["heldout_top1"] for r in val]
    win = [_win_mean(ys, i) for i in range(len(ys))]
    best_i = max(range(len(win)), key=lambda i: (win[i], -steps[i]))
    names = [n for n in ("hfs_oracle", "bfs", "random", "dfs")
             if n in base and base[n].get("heldout_top1") is not None]
    heights = [base[n]["heldout_top1"] for n in names] + [ys[best_i]]
    cols = [BASE_STYLE[n][0] for n in names] + [RL_C]
    bars = names + ["model"]
    n_ho = val[-1].get("heldout_n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bars, heights, color=cols, edgecolor="white")
    for i, h in enumerate(heights):
        ax.text(i, h + .01, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("held-out top1 (oracle agreement)")
    ax.set_ylim(0, 1.05)
    n = f", n={n_ho} frontiers" if n_ho else ""
    ax.set_title(f"Held-out top1: model (selected) vs baselines, matched-n  (F={F}{n})",
                 fontsize=11)
    ax.grid(alpha=.2, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry", type=Path)
    ap.add_argument("--fringe", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    out = a.out_dir or a.telemetry.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    val, base = load(a.telemetry)
    if not val:
        print("[plot] no val checkpoints in telemetry"); return
    F = a.fringe
    held = "heldout_top1" in val[-1]
    metric = "held-out top1 (oracle agreement)" if held else "coverage @ reference budget"
    field = "heldout_top1" if held else "coverage_at_reference_budget"

    fig_selection_audit(val, base, out / "selection_audit.png", F, metric, field)
    fig_line(val, base, out / "coverage_vs_checkpoint.png", F,
             "coverage_at_reference_budget", "coverage @ reference budget")
    fig_line(val, base, out / "regret_vs_checkpoint.png", F,
             "regret_mean_lower_bound", "regret (mean lower bound)")
    fig_health(val, out / "training_health.png", F)
    fig_top1_bars(val, base, out / "heldout_top1_bars.png", F)

    print(f"[plot] wrote figures to {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.stat().st_size:>8} bytes  {p.name}")


if __name__ == "__main__":
    main()
