"""Frontier-length distribution across F, from a batch's training-data CSVs.

Dataset-level (not per-run): builds the offline dataset for each F and histograms
|frontier| = |A(s)|. NEUTRAL title/labels; the reading lives in the report.

Usage: plot_frontier_dist.py <batch>/_models/<domain>/training_data [--fringes 4 8 16 32]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "lib" / "rl_handler"))
from src.offline.tree import load_tree_instance          # noqa: E402
from src.offline.dataset import generate_dataset          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("training_data", type=Path, help="<batch>/_models/<dom>/training_data")
    ap.add_argument("--fringes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--mode", default="separated")
    ap.add_argument("--seeds-per-policy", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    csvs = sorted(a.training_data.glob("*/*_depth_*.csv"))
    insts = [load_tree_instance(p, name=p.parent.name, kind_of_data=a.mode) for p in csvs]

    ncol = 2
    nrow = (len(a.fringes) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, F in zip(axes, a.fringes):
        rows, _ = generate_dataset(insts, fringe_size=F, seeds_per_policy=a.seeds_per_policy,
                                   expansion_cap=900, verbose=False)
        na = np.array([r.n_actions for r in rows])
        sat = 100 * np.mean(na == F)
        starve = 100 * np.mean(na < F)
        forced = 100 * np.mean(na == 1)
        ax.hist(na, bins=np.arange(0.5, F + 2.5), color="#0072B2", edgecolor="white")
        ax.axvline(F, color="#D55E00", ls="--", lw=2, label=f"F={F}")
        ax.set_title(f"F={F}: {sat:.0f}% at F, {starve:.0f}% below F, {forced:.0f}% forced",
                     fontsize=10)
        ax.set_xlabel("|frontier| = |A(s)|"); ax.set_ylabel("transitions")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ax in axes[len(a.fringes):]:
        ax.axis("off")
    fig.suptitle(f"Frontier-length distribution per F  ({a.training_data.parent.name}, "
                 f"{len(insts)} instances)", fontsize=11)
    fig.tight_layout()
    out = a.out or a.training_data.parent / "figures" / "frontier_length_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
