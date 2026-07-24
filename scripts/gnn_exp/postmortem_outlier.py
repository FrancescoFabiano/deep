#!/usr/bin/env python3
"""Post-mortem of a misleading-heuristic instance (default: CC_3_2_3__pl_7).

Takes planner-generated ground-truth datasets (dataset mode: CSV with
`File Path` / `Distance From Goal` + DOT files) for the outlier instance and
a couple of control instances, runs a trained checkpoint on every state, and
reports:

  [1] Spearman(pred, true) per instance — is the outlier bad at *ranking*,
      or only in search behavior?
  [2] Prediction profile along a shortest path to the goal (true distance
      k, k-1, ..., 0) — monotone? inverted where?
  [3] Prediction stats per true-distance bucket (0, 1, 2-3, 4-5, 6+) — does
      the model collapse deep states into mid-range predictions?
  [4] OOD check: node/edge-count distribution of the instance's states vs
      the training set (full, and same-family-only subset).

Usage:
  .venv/bin/python scripts/gnn_exp/postmortem_outlier.py \
      --outlier <dataset_dir> --controls <dataset_dir> [<dataset_dir> ...] \
      [--model-dir exp/gnn_exp/batch0/_models/CC_baseline_ep193] \
      [--train-sample 5000]

A "dataset dir" is the folder the planner's --dataset mode produced (or a
copy): it must contain one *_depth_*.csv and the referenced .dot files.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.utils import (  # noqa: E402
    _parse_dot_fast,
    graph_collate_fn,
    preprocess_sample,
    seed_everything,
    select_model,
)

TRAIN_ROOT = REPO_ROOT / "exp/gnn_exp/batch0/_models/CC/training_data"
BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 99)]
UNREACHABLE = 1000000  # planner's sentinel for states with no path to goal


def read_constants(model_dir: Path):
    txt = (model_dir / "distance_estimator_C.txt").read_text()
    consts = dict(
        line.replace(" ", "").split("=") for line in txt.strip().splitlines())
    return float(consts["slope"]), float(consts["intercept"])


def load_dataset(ds_dir: Path):
    """Return rows of (state_path, true_distance, predecessor_path), deduped."""
    csvs = sorted(ds_dir.glob("*_depth_*.csv"))
    assert csvs, f"no *_depth_*.csv in {ds_dir}"
    rows, seen = [], set()
    with open(csvs[0], newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r["File Path"]
            if p in seen:
                continue
            seen.add(p)
            rows.append((p, int(r["Distance From Goal"]),
                         r.get("File Path Predecessor", "")))
    return rows


def resolve(path_str: str, ds_dir: Path) -> Path:
    """CSV paths may be relative to the repo root or to the original output
    folder; fall back to basename lookup inside the dataset dir."""
    p = Path(path_str)
    if p.is_file():
        return p
    q = REPO_ROOT / path_str
    if q.is_file():
        return q
    hits = list(ds_dir.rglob(p.name))
    assert hits, f"cannot resolve {path_str} under {ds_dir}"
    return hits[0]


def predict_all(model, rows, ds_dir, batch_size=1024):
    preds = []
    samples = []
    for path_str, _, _ in rows:
        samples.append(preprocess_sample(str(resolve(path_str, ds_dir))))
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = graph_collate_fn(samples[i:i + batch_size])
            preds.append(model.predict_batch(batch).view(-1))
    return torch.cat(preds)


def shortest_chain(rows):
    """Longest predecessor chain whose true distance increases by exactly 1
    per backward step, ending at a distance-0 state — i.e. a shortest-path
    segment goal-ward.  Returns the list path-to-goal order (dist k..0)."""
    info = {p: (d, pred) for p, d, pred in rows}
    best = []
    for p, d, _ in rows:
        if d != 0:
            continue
        chain = [p]
        cur = p
        while True:
            pred = info[cur][1]
            if pred not in info:
                break
            if info[pred][0] != info[cur][0] + 1:
                break
            chain.append(pred)
            cur = pred
        if len(chain) > len(best):
            best = chain
    return list(reversed(best))  # farthest state first: dist k, ..., 0


def graph_sizes(paths):
    nodes, edges = [], []
    for p in paths:
        data = _parse_dot_fast(Path(p).read_text())
        assert data is not None, f"fast parser rejected {p}"
        nodes.append(data.num_nodes)
        edges.append(data.edge_index.size(1))
    return np.array(nodes), np.array(edges)


def pct(a):
    return (f"mean={a.mean():.1f}  p50={np.percentile(a, 50):.0f}  "
            f"p95={np.percentile(a, 95):.0f}  p99={np.percentile(a, 99):.0f}  "
            f"max={a.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier", required=True, type=Path)
    ap.add_argument("--controls", nargs="+", type=Path, default=[])
    ap.add_argument("--model-dir", type=Path,
                    default=REPO_ROOT / "exp/gnn_exp/batch0/_models/CC_baseline_ep193")
    ap.add_argument("--train-sample", type=int, default=5000)
    args = ap.parse_args()

    slope, intercept = read_constants(args.model_dir)
    print(f"Model: {args.model_dir}  (slope={slope}, intercept={intercept})")

    seed_everything(42)
    m = select_model("distance_estimator", use_goal=False, use_depth=False)
    m.load_model(str(args.model_dir / "distance_estimator.pt"))
    m.model.eval()

    # ---- [1] Spearman per instance --------------------------------------
    # Unreachable states (sentinel distance 1000000) are excluded from the
    # headline Spearman — training filters them out the same way — but the
    # all-states value is reported too, since A* does see those states.
    print("\n[1] Spearman(pred, true) per instance")
    per_inst = {}
    for ds in [args.outlier] + args.controls:
        rows = load_dataset(ds)
        preds = predict_all(m, rows, ds)
        true = torch.tensor([d for _, d, _ in rows], dtype=torch.float)
        reach = true < UNREACHABLE
        rho = spearmanr(preds[reach].numpy(), true[reach].numpy()).statistic
        rho_all = spearmanr(preds.numpy(), true.numpy()).statistic
        per_inst[ds.name] = (rows, preds, true)
        print(f"  {ds.name:20s}: n={len(rows):6d} "
              f"(unreachable {int((~reach).sum()):4d})  "
              f"spearman={rho:.4f}  incl-unreachable={rho_all:.4f}  "
              f"reachable-dist max={int(true[reach].max())}")

    rows, preds, true = per_inst[args.outlier.name]
    reach = true < UNREACHABLE
    ur_pred = (preds[~reach] - intercept) / slope
    if (~reach).sum() > 0:
        print(f"  outlier unreachable states: mean pred_dist "
              f"{ur_pred.mean():.2f} (ideally high) min {ur_pred.min():.2f}")
    pred_dist = (preds - intercept) / slope          # what the planner sees
    pred_h = torch.round(pred_dist)                  # after C++ round()

    # ---- [2] Shortest-path profile ---------------------------------------
    print("\n[2] Prediction profile along a shortest path (outlier)")
    chain = shortest_chain(rows)
    idx = {p: i for i, (p, _, _) in enumerate(rows)}
    print(f"  chain length: {len(chain)} states "
          f"(true dist {len(chain)-1} .. 0)")
    prev = None
    for p in chain:
        i = idx[p]
        d = int(true[i])
        flag = ""
        if prev is not None and pred_dist[i] > prev:
            flag = "   <-- INVERSION (pred goes UP toward goal)"
        print(f"    true={d}  pred_dist={pred_dist[i]:7.3f}  "
              f"h=round={int(pred_h[i])}{flag}")
        prev = pred_dist[i]

    # ---- [3] Bucket stats -------------------------------------------------
    print("\n[3] Outlier prediction stats per true-distance bucket")
    print(f"  {'bucket':>8} {'n':>6} {'pred_dist mean':>14} {'std':>7} "
          f"{'h-mode':>7}")
    for lo, hi in BUCKETS:
        mask = (true >= lo) & (true <= hi)
        if mask.sum() == 0:
            continue
        pd_b = pred_dist[mask]
        h_b = pred_h[mask].to(torch.int64)
        mode = int(torch.mode(h_b).values)
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f"{lo}+")
        print(f"  {label:>8} {int(mask.sum()):>6} {pd_b.mean():>14.3f} "
              f"{pd_b.std():>7.3f} {mode:>7}")

    # ---- [4] OOD check ----------------------------------------------------
    print("\n[4] OOD check: graph sizes (nodes / edges)")
    out_paths = [resolve(p, args.outlier) for p, _, _ in rows]
    on, oe = graph_sizes(out_paths)
    print(f"  outlier states   nodes: {pct(on)}")
    print(f"                   edges: {pct(oe)}")

    rng = random.Random(42)
    all_train = []
    fam = args.outlier.name.rsplit("__", 1)[0]  # e.g. CC_3_2_3
    fam_train = []
    for inst in sorted(TRAIN_ROOT.iterdir()):
        merged = inst / "RawFiles" / "hash_merged"
        if merged.is_dir():
            fs = sorted(merged.glob("*.dot"))
            all_train += fs
            if inst.name.startswith(fam):
                fam_train += fs
    for label, pool in [("training (all)", all_train),
                        (f"training ({fam}*)", fam_train)]:
        sample = rng.sample(pool, min(args.train_sample, len(pool)))
        tn, te = graph_sizes(sample)
        print(f"  {label:18s} nodes: {pct(tn)}   (n={len(sample)})")
        print(f"  {'':18s} edges: {pct(te)}")
        print(f"  {'':18s} outlier states beyond this max: "
              f"nodes {100*(on > tn.max()).mean():.1f}%  "
              f"edges {100*(oe > te.max()).mean():.1f}%")

    # Bonus: Spearman restricted to in-range vs out-of-range states
    sample = rng.sample(all_train, min(args.train_sample, len(all_train)))
    tn, _ = graph_sizes(sample)
    in_range = torch.tensor(on <= tn.max()) & reach
    for label, mask in [("in-range", in_range), ("out-of-range", ~in_range & reach)]:
        if mask.sum() >= 10 and len(torch.unique(true[mask])) > 1:
            rho = spearmanr(preds[mask].numpy(), true[mask].numpy()).statistic
            print(f"  outlier spearman on {label} states "
                  f"(n={int(mask.sum())}): {rho:.4f}")


if __name__ == "__main__":
    main()
