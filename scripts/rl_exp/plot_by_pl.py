"""
Per-plan-length line plots, per domain.

x-axis = plan length (pl) parsed from the instance File name (e.g. CC_2_3_4__pl_7.txt -> 7).
Each line = one approach (heuristic + fringe + strict). Three metrics, each in three
split variants (train / test / train+test combined) -> 9 PNGs per domain, written to
the domain's analysis/ dir.

Reads the PER-INSTANCE section of every summary CSV under the results dir
(header starts with "File", terminated by the blank line before the "Solved,Total,..."
stats row). Skips aggregate.csv and anything under analysis/.

Usage:
    python3 scripts/rl_exp/plot_by_pl.py [results_dir]
results_dir defaults to "combined_results" (back-compat). When called from the
pipeline it is already batch+domain scoped, e.g. combined_results/batchX/CC.
"""

import sys
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# An approach line is shown only if its OVERALL success rate across all pl values
# in that split is >= this threshold. Filters entire lines, not individual points.
SUCCESS_RATE_THRESHOLD = 0.999

# (column in the CSV / y-axis label, output filename prefix)
METRICS = [
    ("NodesExpanded", "nodes_by_pl"),
    ("TotalExecutionTime", "time_by_pl"),
    ("PlanLength", "planlength_by_pl"),
]

SPLITS = ["train", "test", "all"]

PL_RE = re.compile(r"__pl_(\d+)")


def parse_label(csv_path: Path):
    """(split, line_label) from filename + parent dir, or None if not a summary CSV.

    train_RL-SUBGOALS_strict.csv under fringe_32/ -> ("train", "RL-SUBGOALS_f32_strict")
    train_BFS_strict.csv         under bfs/        -> ("train", "BFS_strict")
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

    parent = csv_path.parent.name
    if parent.startswith("fringe_"):
        fringe = parent[len("fringe_"):]
        core = f"{approach}_f{fringe}"
    else:
        # bfs (or any non-fringe dir): no fringe component
        core = approach

    label = f"{core}_{strict}" if strict else core
    return split, label


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
            # blank line -> end of the per-instance section
            if not fields or all(c.strip() == "" for c in fields):
                break
            if header is None:
                if fields[0].strip() != "File":
                    return []  # not the expected per-instance layout
                header = [c.strip() for c in fields]
                continue
            # defensive: also stop if the stats table starts without a blank line
            if fields[0].strip() == "Solved":
                break
            rows.append(dict(zip(header, fields)))
    return rows


def load_records(results_dir: Path):
    """Flatten every per-instance row across all summary CSVs into records."""
    records = []
    labels = set()
    for csv_path in sorted(results_dir.rglob("*.csv")):
        if csv_path.name == "aggregate.csv":
            continue
        if "analysis" in csv_path.parts:
            continue
        parsed = parse_label(csv_path)
        if parsed is None:
            continue
        split, label = parsed
        for row in read_instances(csv_path):
            m = PL_RE.search(row.get("File", ""))
            if not m:
                continue  # no pl -> can't place on the x-axis
            records.append({
                "label": label,
                "split": split,
                "file": row.get("File", "").strip(),
                "pl": int(m.group(1)),
                "solved": row.get("GoalFound", "").strip() == "Yes",
                "NodesExpanded": num(row.get("NodesExpanded")),
                "TotalExecutionTime": num(row.get("TotalExecutionTime")),
                "PlanLength": num(row.get("PlanLength")),
            })
            labels.add(label)
    return records, labels


def survivors_for_split(subset):
    """Labels whose overall success rate across all pl in this split >= threshold."""
    total = defaultdict(int)
    solved = defaultdict(int)
    for r in subset:
        total[r["label"]] += 1
        if r["solved"]:
            solved[r["label"]] += 1
    return {
        lbl for lbl, tot in total.items()
        if tot > 0 and solved[lbl] / tot >= SUCCESS_RATE_THRESHOLD
    }


def common_solved_files(subset, labels):
    """Files solved by EVERY approach in `labels` (the fair, common-solved set).

    Only files attempted by all of `labels` and solved by all of them qualify, so
    every plotted line is averaged over the exact same instances — no approach gets
    an easier subset by silently dropping the instances it failed on.
    """
    attempted = defaultdict(set)
    solved = defaultdict(set)
    for r in subset:
        if r["label"] not in labels:
            continue
        attempted[r["file"]].add(r["label"])
        if r["solved"]:
            solved[r["file"]].add(r["label"])
    return {
        f for f, labs in attempted.items()
        if labs >= labels and solved[f] >= labels
    }


def make_figure(subset, survivors, metric, variant, domain, color_map, out):
    mname, fprefix = metric
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_x = set()
    for lbl in sorted(survivors):
        # mean/std over SOLVED instances only, grouped by pl
        by_pl = defaultdict(list)
        for r in subset:
            if r["label"] != lbl or not r["solved"]:
                continue
            v = r[mname]
            if v is None:
                continue
            by_pl[r["pl"]].append(v)
        if not by_pl:
            continue

        xs = sorted(by_pl)
        means = np.array([np.mean(by_pl[x]) for x in xs])
        stds = np.array([np.std(by_pl[x]) for x in xs])  # population std (0 for singletons)
        color = color_map[lbl]
        linestyle = "--" if "bfs_" in lbl else "-"

        ax.plot(
            xs,
            means,
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle=linestyle,
            label=lbl,
            color=color,
        )

        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18)
        plotted_x.update(xs)

    ax.set_title(f"{mname} by plan length ({variant}) — {domain}")
    ax.set_xlabel("plan length (pl)")
    ax.set_ylabel(mname)
    ax.grid(True, alpha=0.3)

    if plotted_x:
        ax.set_xticks(sorted(plotted_x))

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        if len(handles) <= 6:
            ax.legend(fontsize=8)
        else:
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "no approaches >= success threshold",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, alpha=0.6)

    fig.tight_layout()
    dst = out / f"{fprefix}_{variant}.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {dst.name} ({len(handles)} approaches)")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("combined_results")
    domain = results_dir.name
    out = results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    records, labels = load_records(results_dir)

    # global color map: same approach -> same color in every figure
    labels_sorted = sorted(labels)
    n = max(1, len(labels_sorted))
    palette = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, n))
    color_map = {lbl: palette[i] for i, lbl in enumerate(labels_sorted)}

    for variant in SPLITS:
        subset = [r for r in records if variant == "all" or r["split"] == variant]
        survivors = survivors_for_split(subset)
        for metric in METRICS:
            make_figure(subset, survivors, metric, variant, domain, color_map, out)

    # --- common-solved figures ---
    # Fair head-to-head across all splits: restrict to the instances solved by
    # EVERY approach, so the per-pl means for nodes / plan length / time compare the
    # exact same instances (no approach gets an easier subset by dropping failures).
    # Unlike the split figures, this uses all approaches, not just the near-100%
    # survivors, since the whole point is a like-for-like multi-approach comparison.
    # Produces 3 extra PNGs (one per metric).
    all_labels = set(labels)
    common = common_solved_files(records, all_labels)
    common_subset = [r for r in records if r["file"] in common]
    print(f"[info] common-solved set: {len(common)} instances across {len(all_labels)} approaches")
    for metric in METRICS:
        make_figure(common_subset, all_labels, metric, "common_solved", domain, color_map, out)

    print(f"[OK] plot_by_pl complete for domain={domain}")


if __name__ == "__main__":
    main()
