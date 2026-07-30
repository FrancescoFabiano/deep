"""
RL-vs-BFS summary figure, per domain and batch.

Given a batch+domain scoped results dir (e.g. combined_results/batchX/CC), compare
every learned approach against the BFS baseline on the SAME instances (BFS-solved
AND approach-solved), so the comparison is apples-to-apples. One PNG is written to
the domain's analysis/ dir; the batch and domain are read from the path.

For each approach the best fringe is picked (highest test success rate, then fewest
nodes), matching the ordering used by the heatmap scripts. Four panels are drawn:

  1. Coverage (solved %) — train and test bars, with the BFS coverage as a line.
  2. Node reduction vs BFS — median BFS_nodes / RL_nodes on commonly-solved test
     instances (>1 means the approach expands fewer nodes than BFS).
  3. Plan optimality — fraction of commonly-solved test instances whose plan length
     equals the (optimal) BFS plan length.
  4. Speed vs BFS — median BFS_time / RL_time on commonly-solved test instances
     (>1 means the approach is faster in wall-clock than BFS).

The figure carries only data (bars, values, reference lines, axis labels); no prose
annotations.

Usage:
    python3 scripts/rl_exp/plot_rl_vs_bfs.py [results_dir]
results_dir defaults to "combined_results" (back-compat).
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def num(s):
    """Parse a metric cell to float; None for blank / '-' / non-numeric."""
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
        header = None
        for fields in csv.reader(f):
            if not fields or all(c.strip() == "" for c in fields):
                break
            if header is None:
                if fields[0].strip() != "File":
                    return []
                header = [c.strip() for c in fields]
                continue
            if fields[0].strip() == "Solved":
                break
            rows.append(dict(zip(header, fields)))
    return rows


def parse_label(csv_path: Path):
    """(split, is_bfs, fringe, approach) or None if not a summary CSV."""
    stem = csv_path.stem
    if stem.startswith("train_"):
        split, rest = "train", stem[len("train_"):]
    elif stem.startswith("test_"):
        split, rest = "test", stem[len("test_"):]
    else:
        return None

    if rest.endswith("_nostrict"):
        rest = rest[:-len("_nostrict")]
        strict = "nostrict"
    elif rest.endswith("_strict"):
        rest = rest[:-len("_strict")]
        strict = "strict"
    else:
        strict = None

    parent = csv_path.parent.name
    is_bfs = parent == "bfs" or rest.startswith("BFS")
    fringe = int(parent[len("fringe_"):]) if parent.startswith("fringe_") else 0

    approach = rest if strict is None else f"{rest}_{strict}"
    return split, is_bfs, fringe, approach


def load(results_dir: Path):
    """Flatten every per-instance row into records."""
    records = []
    for csv_path in sorted(results_dir.rglob("*.csv")):
        if csv_path.name == "aggregate.csv" or "analysis" in csv_path.parts:
            continue
        parsed = parse_label(csv_path)
        if parsed is None:
            continue
        split, is_bfs, fringe, approach = parsed
        for row in read_instances(csv_path):
            records.append({
                "split": split,
                "is_bfs": is_bfs,
                "fringe": fringe,
                "approach": approach,
                "file": row.get("File", "").strip(),
                "solved": row.get("GoalFound", "").strip() == "Yes",
                "nodes": num(row.get("NodesExpanded")),
                "plan": num(row.get("PlanLength")),
                "time": num(row.get("TotalExecutionTime")),
            })
    return records


def coverage(records, pred):
    total = sum(1 for r in records if pred(r))
    solved = sum(1 for r in records if pred(r) and r["solved"])
    return (100.0 * solved / total) if total else 0.0


def best_fringe_per_approach(records, eval_split):
    """For each learned approach, the fringe with the highest `eval_split` success
    rate, tie-broken by fewest mean nodes on solved `eval_split` instances."""
    approaches = sorted({r["approach"] for r in records if not r["is_bfs"]})
    chosen = {}
    for appr in approaches:
        best = None
        for fr in sorted({r["fringe"] for r in records
                          if r["approach"] == appr and not r["is_bfs"]}):
            sub = [r for r in records
                   if r["approach"] == appr and r["fringe"] == fr
                   and not r["is_bfs"] and r["split"] == eval_split]
            if not sub:
                continue
            cov = coverage(sub, lambda r: True)
            nodes = [r["nodes"] for r in sub if r["solved"] and r["nodes"] is not None]
            mean_nodes = float(np.mean(nodes)) if nodes else float("inf")
            key = (cov, -mean_nodes)  # higher cov, then fewer nodes
            if best is None or key > best[0]:
                best = (key, fr)
        if best is not None:
            chosen[appr] = best[1]
    return chosen


def compare_vs_bfs(records, appr, fringe, split="test"):
    """Median node reduction, plan-optimal fraction and median speedup vs BFS on
    the instances both BFS and this approach solved in `split`."""
    bfs = {r["file"]: r for r in records if r["is_bfs"] and r["split"] == split}
    node_red, plan_opt, speed = [], [], []
    for r in records:
        if r["is_bfs"] or r["approach"] != appr or r["fringe"] != fringe:
            continue
        if r["split"] != split or not r["solved"]:
            continue
        b = bfs.get(r["file"])
        if not b or not b["solved"]:
            continue
        if r["nodes"] and b["nodes"]:
            node_red.append(b["nodes"] / r["nodes"])
        if r["plan"] and b["plan"]:
            plan_opt.append(1.0 if r["plan"] <= b["plan"] + 1e-9 else 0.0)
        if r["time"] and b["time"] and b["time"] > 0:
            speed.append(b["time"] / r["time"])
    return {
        "node_red": float(np.median(node_red)) if node_red else np.nan,
        "plan_opt": 100.0 * float(np.mean(plan_opt)) if plan_opt else np.nan,
        "speed": float(np.median(speed)) if speed else np.nan,
        "n": len(node_red),
    }


def bars(ax, labels, values, color, ref=None, ref_label=None, fmt="{:.0f}"):
    x = np.arange(len(labels))
    vals = np.array([v if v is not None and not np.isnan(v) else 0.0 for v in values])
    ax.bar(x, vals, color=color, width=0.7)
    for xi, v in zip(x, values):
        if v is not None and not np.isnan(v):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
    if ref is not None:
        ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, label=ref_label)
        if ref_label:
            ax.legend(fontsize=8, loc="best")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("combined_results")
    domain = results_dir.name
    batch = results_dir.parent.name
    out = results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    records = load(results_dir)
    if not records:
        print(f"[WARN] no summary CSVs under {results_dir}")
        return

    # evaluate on the held-out test split; fall back to train for train-only domains
    has_test = any(r["split"] == "test" for r in records)
    eval_split = "test" if has_test else "train"

    chosen = best_fringe_per_approach(records, eval_split)
    if not chosen:
        print(f"[WARN] no learned approaches under {results_dir}")
        return

    # order approaches by eval-split coverage (desc), then node reduction (desc)
    rows = []
    for appr, fr in chosen.items():
        cov_eval = coverage(records, lambda r: (
            r["approach"] == appr and r["fringe"] == fr and not r["is_bfs"]
            and r["split"] == eval_split))
        cov_train = coverage(records, lambda r: (
            r["approach"] == appr and r["fringe"] == fr and not r["is_bfs"]
            and r["split"] == "train"))
        cmp = compare_vs_bfs(records, appr, fr, eval_split)
        rows.append({
            "label": f"{appr} (f{fr})",
            "cov_train": cov_train,
            "cov_eval": cov_eval,
            **cmp,
        })
    rows.sort(key=lambda d: (d["cov_eval"],
                             d["node_red"] if not np.isnan(d["node_red"]) else -1),
              reverse=True)

    bfs_cov_train = coverage(records, lambda r: r["is_bfs"] and r["split"] == "train")
    bfs_cov_eval = coverage(records, lambda r: r["is_bfs"] and r["split"] == eval_split)

    labels = [d["label"] for d in rows]

    fig, axes = plt.subplots(2, 2, figsize=(max(11, 0.7 * len(labels) + 6), 10))
    (ax_cov, ax_nodes), (ax_plan, ax_time) = axes

    # panel 1: coverage (train + eval split) with BFS reference lines
    x = np.arange(len(labels))
    if eval_split != "train":
        w = 0.38
        ax_cov.bar(x - w / 2, [d["cov_train"] for d in rows], w, color="#8ecae6", label="RL train")
        ax_cov.bar(x + w / 2, [d["cov_eval"] for d in rows], w, color="#219ebc",
                   label=f"RL {eval_split}")
        annot_x = x + w / 2
    else:
        ax_cov.bar(x, [d["cov_eval"] for d in rows], 0.6, color="#219ebc", label="RL train")
        annot_x = x
    ax_cov.axhline(bfs_cov_eval, color="black", linestyle="--", linewidth=1.2,
                   label=f"BFS {eval_split} ({bfs_cov_eval:.0f}%)")
    if eval_split != "train":
        ax_cov.axhline(bfs_cov_train, color="gray", linestyle=":", linewidth=1.2,
                       label=f"BFS train ({bfs_cov_train:.0f}%)")
    for xi, d in zip(annot_x, rows):
        ax_cov.text(xi, d["cov_eval"], f"{d['cov_eval']:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_cov.set_ylabel("coverage (%)")
    ax_cov.set_title("Coverage")
    ax_cov.set_ylim(0, 105)
    ax_cov.grid(True, axis="y", alpha=0.3)
    ax_cov.legend(fontsize=8, loc="lower right")

    # panel 2: node reduction vs BFS (eval split, common), 1.0 = BFS
    bars(ax_nodes, labels, [d["node_red"] for d in rows], "#ffb703",
         ref=1.0, ref_label="BFS (=1)", fmt="{:.1f}")
    ax_nodes.set_ylabel("BFS nodes / RL nodes (median)")
    ax_nodes.set_title(f"Node reduction vs BFS  —  {eval_split}, commonly solved")

    # panel 3: plan optimality (% == BFS optimal), eval split common
    bars(ax_plan, labels, [d["plan_opt"] for d in rows], "#90be6d",
         ref=100.0, ref_label="BFS optimal (100%)", fmt="{:.0f}")
    ax_plan.set_ylabel("plans == BFS optimal (%)")
    ax_plan.set_title(f"Plan optimality  —  {eval_split}, commonly solved")
    ax_plan.set_ylim(0, 105)

    # panel 4: speed vs BFS (eval split, common), 1.0 = BFS
    bars(ax_time, labels, [d["speed"] for d in rows], "#f4978e",
         ref=1.0, ref_label="BFS (=1)", fmt="{:.2f}")
    ax_time.set_ylabel("BFS time / RL time (median)")
    ax_time.set_title(f"Speed vs BFS  —  {eval_split}, commonly solved")

    fig.suptitle(f"RL vs BFS — {batch} / {domain}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    dst = out / "rl_vs_bfs.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    fig.savefig(out / "rl_vs_bfs.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {dst} ({len(labels)} approaches)")


if __name__ == "__main__":
    main()
