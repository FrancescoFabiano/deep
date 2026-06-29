"""Plot helpers for the offline-RL runs (Phase 4).

- per-seed training curves (TD loss, mean Q, episode return) over frames
- IQM ± IQR-std bands across seeds over checkpoints, for both eval metrics
- score(s) vs -d*(s) scatter at the best checkpoint

IQM = mean of values inside [q1, q3]; dispersion = std of that trimmed set
(same convention as RLFrontierTrainer._iqm_iqr_stats).  A 3-seed band is
crude — annotated on the plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def iqm_iqr_std(values: Sequence[float]) -> tuple[float, float]:
    t = torch.tensor([float(v) for v in values], dtype=torch.float32)
    if t.numel() == 0:
        return 0.0, 0.0
    q1 = torch.quantile(t, 0.25)
    q3 = torch.quantile(t, 0.75)
    trimmed = t[(t >= q1) & (t <= q3)]
    if trimmed.numel() == 0:
        trimmed = t
    return float(trimmed.mean()), float(trimmed.std(unbiased=False))


def _moving_avg(xs: List[float], k: int = 25) -> List[float]:
    out, acc = [], 0.0
    from collections import deque

    win: deque = deque()
    for x in xs:
        win.append(x)
        acc += x
        if len(win) > k:
            acc -= win.popleft()
        out.append(acc / len(win))
    return out


def plot_seed_curves(
    history_file: Path, out_dir: Path, seed: int, fringe: Optional[int] = None
) -> None:
    suffix = f"_fringe{fringe}" if fringe is not None else ""
    payload = json.loads(Path(history_file).read_text())
    h = payload["history"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Episode-level panels need per-episode history (episode_frame/return/
    # expansions). The multi-regime trainer logs no episodes, so degrade to the
    # two always-present loss/score panels rather than failing the whole figure
    # — these visualizations must always render.
    has_episodes = all(
        k in h and h[k]
        for k in ("episode_frame", "episode_return", "episode_expansions")
    )
    n_panels = 4 if has_episodes else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=200)
    axes[0].plot(h["frame"], h["td_loss"])
    axes[0].set_title(f"seed {seed} | TD loss")
    axes[0].set_xlabel("frame")
    axes[0].set_yscale("log")
    axes[1].plot(h["frame"], h["q_mean"], label="Q(s,a) mean")
    axes[1].plot(h["frame"], h["target_mean"], label="target mean", alpha=0.7)
    axes[1].set_title(f"seed {seed} | mean score")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    if has_episodes:
        axes[2].plot(
            h["episode_frame"],
            _moving_avg([float(r) for r in h["episode_return"]]),
            label="return (MA-25)",
        )
        axes[2].set_title(f"seed {seed} | episode return")
        axes[2].set_xlabel("frame")
        axes[2].legend()
        axes[3].plot(
            h["episode_frame"],
            _moving_avg([float(e) for e in h["episode_expansions"]]),
            label="expansions (MA-25)",
        )
        axes[3].set_title(f"seed {seed} | episode expansions")
        axes[3].set_xlabel("frame")
        axes[3].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"training_curves_seed{seed}{suffix}.png")
    plt.close(fig)


def plot_seed_val_curves(
    history_file: Path, out_dir: Path, seed: int, fringe: Optional[int] = None
) -> None:
    """Single-seed eval-metric curves over checkpoint frames.

    Companion to plot_seed_curves for per-run (single-seed) output; the
    cross-seed IQM bands stay in plot_iqm_bands_across_seeds.
    """
    suffix = f"_fringe{fringe}" if fringe is not None else ""
    payload = json.loads(Path(history_file).read_text())
    ckpts = payload["checkpoints"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [c["frame"] for c in ckpts]
    exps = [c["summary"]["val_total_expansions"] for c in ckpts]
    rhos = [c["summary"]["val_spearman_all"] for c in ckpts]

    # Occupancy panel only when the history carries it (older runs predate the
    # val_occupancy field — degrade to the original 2-panel layout).
    has_occ = bool(ckpts) and all(
        c["summary"].get("val_occupancy") for c in ckpts
    )
    n_panels = 3 if has_occ else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), dpi=200)
    axes[0].plot(frames, exps, marker="o")
    axes[0].set_title(f"seed {seed} | val greedy #expansions (lower=better)")
    axes[0].set_xlabel("frame")
    axes[1].plot(frames, rhos, marker="o")
    axes[1].set_title(f"seed {seed} | val Spearman(score, -d*) incl. unreachable")
    axes[1].set_xlabel("frame")
    if has_occ:
        occ = [c["summary"]["val_occupancy"] for c in ckpts]
        axes[2].plot(frames, [o["max"] for o in occ], marker="o", label="max")
        axes[2].plot(frames, [o["p99"] for o in occ], marker=".", label="p99", alpha=0.8)
        axes[2].plot(frames, [o["p90"] for o in occ], marker=".", label="p90", alpha=0.8)
        axes[2].plot(frames, [o["mean"] for o in occ], marker=".", label="mean", alpha=0.8)
        if fringe is not None:
            axes[2].axhline(
                float(fringe), ls="--", c="red",
                label=f"F={fringe} (beam binds above)",
            )
        axes[2].set_title(f"seed {seed} | val fringe occupancy vs F")
        axes[2].set_xlabel("frame")
        axes[2].set_ylabel("live beam members")
        axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"val_metrics_seed{seed}{suffix}.png")
    plt.close(fig)


def plot_iqm_bands_across_seeds(
    history_files: Dict[int, Path],
    out_dir: Path,
    bfs_reference: Optional[float] = None,
    optimal_reference: Optional[float] = None,
) -> None:
    """IQM ± IQR-std across seeds at each checkpoint, both eval metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = {
        seed: json.loads(Path(f).read_text())["checkpoints"]
        for seed, f in history_files.items()
    }
    n_ck = min(len(c) for c in per_seed.values())
    frames = [per_seed[next(iter(per_seed))][k]["frame"] for k in range(n_ck)]

    metrics = {
        "val_total_expansions": "val greedy #expansions (lower=better)",
        "val_spearman_all": "val Spearman(score, -d*) incl. unreachable",
    }
    for key, label in metrics.items():
        iqms, bands = [], []
        for k in range(n_ck):
            vals = [
                per_seed[s][k]["summary"][key]
                for s in per_seed
                if per_seed[s][k]["summary"][key] is not None
            ]
            m, sd = iqm_iqr_std(vals)
            iqms.append(m)
            bands.append(sd)
        plt.figure(figsize=(8, 4.5), dpi=200)
        lo = [m - s for m, s in zip(iqms, bands)]
        hi = [m + s for m, s in zip(iqms, bands)]
        plt.plot(frames, iqms, marker="o", label="IQM across seeds")
        plt.fill_between(frames, lo, hi, alpha=0.25, label="± IQR-std")
        if key == "val_total_expansions":
            if bfs_reference is not None:
                plt.axhline(bfs_reference, ls="--", c="gray", label="BFS")
            if optimal_reference is not None:
                plt.axhline(optimal_reference, ls=":", c="green", label="optimal")
        plt.xlabel("frame")
        plt.ylabel(label)
        plt.title(f"{label} | IQM ± IQR-std over {len(per_seed)} seeds (crude band)")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"iqm_{key}.png")
        plt.close()


def plot_score_vs_dstar(
    scores: Sequence[float],
    d_star: Sequence[float],
    unreachable_mask: Sequence[bool],
    out_path: Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=200)
    s = torch.tensor([float(x) for x in scores])
    d = torch.tensor([float(x) for x in d_star])
    m = torch.tensor([bool(x) for x in unreachable_mask])
    plt.scatter(
        (-d[~m]).numpy(), s[~m].numpy(), s=4, alpha=0.3, label="reachable"
    )
    if bool(m.any()):
        plt.scatter(
            torch.full((int(m.sum()),), float((-d[~m]).min()) - 2).numpy(),
            s[m].numpy(),
            s=4,
            alpha=0.3,
            c="red",
            label="unreachable (placed left)",
        )
    plt.xlabel("-d*(s)")
    plt.ylabel("score(s)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
