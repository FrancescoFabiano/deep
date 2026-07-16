"""Plot a fringe run's telemetry.jsonl. Reads ONLY what is already on disk.

Written because the F=4 HASHED floor run crashed at the ONNX export (the device
leak, run.py:259) BEFORE the figure step -- train -> export -> gates -> figures --
so the data was complete but unplotted. No GPU, no re-run: telemetry.jsonl already
carries the baselines (step=-1) and every checkpoint.

Three figures, and together they ARE the floor result:
  1. coverage vs checkpoint  -- RL bounces with no trend; the argmax selection rule
     picks a noise peak and compares it to each baseline's single draw. This is the
     visual argument for the selection fix.
  2. regret vs checkpoint    -- RL's mean sits ABOVE random: the floor null.
  3. td_loss + grad_norm     -- healthy convergence WHILE regret stays bad: the
     "fits the hash target, generalises nowhere" signature.
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
    "hfs_oracle": ("#000000", "-.", "oracle (clairvoyant)"),
    "random":     ("#D55E00", "--", "random"),
    "bfs":        ("#0072B2", ":",  "bfs"),
    "dfs":        ("#009E73", (0, (3, 1, 1, 1)), "dfs"),
}


def load(path: Path):
    rows = [json.loads(l) for l in path.open()]
    val = sorted([r for r in rows if r["split"] == "val" and r["step"] > 0],
                 key=lambda r: r["step"])
    base = {r["split"].split(":", 1)[1]: r for r in rows if r["step"] == -1}
    return val, base


def _baselines(ax, base, field):
    for name, (c, ls, label) in BASE_STYLE.items():
        if name not in base:
            continue
        y = base[name][field]
        if y is None:
            continue
        ax.axhline(y, color=c, ls=ls, lw=1.6, label=f"{label} = {y:.3g}", zorder=1)


def fig_coverage(val, base, out: Path, n_roll: int):
    steps = [r["step"] for r in val]
    cov = [r["coverage_at_reference_budget"] for r in val]
    mean, sd, mx = st.mean(cov), st.stdev(cov), max(cov)

    fig, ax = plt.subplots(figsize=(9, 5))
    _baselines(ax, base, "coverage_at_reference_budget")
    ax.plot(steps, cov, "o-", color="#CC79A7", lw=1.8, ms=5, zorder=3, label="RL (dqn) per checkpoint")
    ax.axhline(mean, color="#CC79A7", lw=1.2, alpha=.55, zorder=2,
               label=f"RL mean = {mean:.3f} $\\pm$ {sd:.3f}")
    # the checkpoints argmax selection would pick
    peaks = [(s, c) for s, c in zip(steps, cov) if c == mx]
    ax.plot([p[0] for p in peaks], [p[1] for p in peaks], "*", ms=17, color="#CC79A7",
            mec="black", mew=.8, zorder=4,
            label=f"argmax draws ({len(peaks)}/{len(cov)}) -- selection reports one of these")
    rnd = base["random"]["coverage_at_reference_budget"]
    # THE bias, drawn: selection reports the argmax (above random); the model's actual
    # mean is BELOW random. Same data, opposite conclusions.
    ax.annotate("", xy=(102000, mx), xytext=(102000, mean),
                arrowprops=dict(arrowstyle="<->", lw=1.4, color="black"))
    ax.text(103000, (mx + mean) / 2,
            f"selection reports {mx:.3f}\n(max of {len(cov)} noisy draws)\n"
            f"but RL mean is {mean:.3f}\n= BELOW random {rnd:.3f}",
            fontsize=8.5, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=.4", fc="#FFF6E5", ec="grey"))
    ax.set_xlabel("training step"); ax.set_ylabel("coverage @ reference budget")
    ax.set_title("F=4 HASHED floor: RL coverage has NO trend -- argmax selects noise\n"
                 f"n={n_roll} rollouts/checkpoint, 1 rollout = {1/n_roll:.3f} coverage",
                 fontsize=11)
    ax.set_ylim(.65, 1.06); ax.set_xlim(0, 152000); ax.grid(alpha=.25)
    # park the legend in the empty right margin: the data lives at x < 100k, and a
    # legend over it would hide the 60k dip to 0.73 -- the very point of the figure
    ax.legend(fontsize=7.5, loc="lower right", framealpha=.95)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_regret(val, base, out: Path):
    steps = [r["step"] for r in val]
    reg = [r["regret_mean_lower_bound"] for r in val]
    mean, sd = st.mean(reg), st.stdev(reg)

    fig, ax = plt.subplots(figsize=(9, 5))
    _baselines(ax, base, "regret_mean_lower_bound")
    ax.plot(steps, reg, "o-", color="#CC79A7", lw=1.8, ms=5, zorder=3, label="RL (dqn) per checkpoint")
    ax.axhline(mean, color="#CC79A7", lw=1.2, alpha=.55, zorder=2,
               label=f"RL mean = {mean:.1f} $\\pm$ {sd:.1f}")
    rnd = base["random"]["regret_mean_lower_bound"]
    ax.fill_between([min(steps), max(steps)], rnd, mean, color="#D55E00", alpha=.10, zorder=0)
    ax.annotate(f"RL is {mean - rnd:.0f} expansions WORSE than random\n"
                f"-- the floor null: no cross-config signal on HASHED",
                xy=(steps[len(steps) // 2], (mean + rnd) / 2), fontsize=9,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="grey", alpha=.9))
    ax.set_xlabel("training step"); ax.set_ylabel("regret (mean lower bound, expansions)")
    ax.set_title("F=4 HASHED floor: RL regret sits ABOVE random -- lower is better",
                 fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper right", framealpha=.95)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_health(val, out: Path):
    steps = [r["step"] for r in val]
    td = [r["td_loss"] for r in val]
    gn = [r["grad_norm"] for r in val]
    reg = [r["regret_mean_lower_bound"] for r in val]

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.semilogy(steps, td, "o-", color="#0072B2", lw=1.8, ms=4, label="td_loss (log)")
    ax1.set_ylabel("td_loss", color="#0072B2"); ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax2 = ax1.twinx()
    ax2.plot(steps, gn, "s--", color="#009E73", lw=1.5, ms=4, label="grad_norm")
    ax2.set_ylabel("grad_norm", color="#009E73"); ax2.tick_params(axis="y", labelcolor="#009E73")
    ax1.set_title("F=4 HASHED floor: training converges cleanly WHILE regret stays bad\n"
                  "= fits the hash target, generalises nowhere", fontsize=11)
    ax1.grid(alpha=.25)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.annotate(f"td_loss {td[0]:.4f} $\\to$ {td[-1]:.4f}", xy=(steps[-1], td[-1]),
                 xytext=(steps[len(steps)//2], td[0]), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=1.1))

    ax3.plot(steps, reg, "o-", color="#CC79A7", lw=1.8, ms=4)
    ax3.set_ylabel("regret"); ax3.set_xlabel("training step"); ax3.grid(alpha=.25)
    ax3.set_ylim(min(reg) - 12, max(reg) + 6)   # headroom so the label clears the peak
    ax3.text(.015, .06, f"regret: no improvement (mean {st.mean(reg):.0f} $\\pm$ {st.stdev(reg):.0f})",
             transform=ax3.transAxes, ha="left", va="bottom", fontsize=9, color="#CC79A7")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    out = a.out_dir or a.telemetry.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    val, base = load(a.telemetry)
    n_roll = val[0].get("n_rollouts", 15)
    fig_coverage(val, base, out / "coverage_vs_checkpoint.png", n_roll)
    fig_regret(val, base, out / "regret_vs_checkpoint.png")
    fig_health(val, out / "training_health.png")
    print(f"[plot] wrote 3 figures to {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.stat().st_size:>8} bytes  {p.name}")


if __name__ == "__main__":
    main()
