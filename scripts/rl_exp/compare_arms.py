#!/usr/bin/env python3
"""Side-by-side comparison of two (or more) F=32 arms by context mode.

For each arm: best checkpoint = min val_total_expansions across checkpoints.
Reports best val_total, its frame + val-Spearman, and per-instance greedy
expansions vs BFS/optimal. Usage:
  python3 scripts/rl_exp/compare_arms.py NAME=DIR [NAME=DIR ...]
DIR = the seedNN_fringeF run dir containing history.json.
"""
import json, sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("usage: compare_arms.py NAME=path/to/seed42_fringe32 [...]")

arms = {}
for spec in sys.argv[1:]:
    name, _, path = spec.partition("=")
    h = json.load(open(Path(path) / "history.json"))
    cks = h["checkpoints"]
    best = min(cks, key=lambda c: c["summary"]["val_total_expansions"])
    final = max(cks, key=lambda c: c["summary"]["frame"])
    arms[name] = {"best": best, "final": final, "n": len(cks),
                  "cfg": h.get("config", {})}

names = list(arms)
print("=== best checkpoint (min val_total_expansions across checkpoints) ===")
print(f"{'arm':<16} {'best_val_total':>14} {'@frame':>8} {'spearman_all':>13} "
      f"{'final_val_total':>16}")
for n in names:
    b = arms[n]["best"]["summary"]; f = arms[n]["final"]["summary"]
    print(f"{n:<16} {b['val_total_expansions']:>14.0f} {b['frame']:>8} "
          f"{b.get('val_spearman_all', float('nan')):>13.3f} "
          f"{f['val_total_expansions']:>16.0f}")

# per-instance greedy at each arm's best checkpoint
print("\n=== per-instance greedy expansions @ each arm's best-val checkpoint ===")
insts = sorted(arms[names[0]]["best"]["val_per_instance"].keys())
opt, bfs = {}, {}
cols = {n: {} for n in names}
for n in names:
    vpi = arms[n]["best"]["val_per_instance"]
    for i, rec in vpi.items():
        cols[n][i] = rec.get("greedy_mean_expansions")
        opt[i] = rec.get("optimal_expansions"); bfs[i] = rec.get("bfs_expansions")
hdr = f"{'instance':<16} {'opt':>4} {'bfs':>5} " + " ".join(f"{n[:12]:>12}" for n in names)
print(hdr)
for i in insts:
    row = " ".join(f"{cols[n].get(i, float('nan')):>12.0f}"
                   if cols[n].get(i) is not None else f"{'-':>12}" for n in names)
    print(f"{i:<16} {opt.get(i,'-'):>4} {bfs.get(i,'-'):>5} {row}")
sums = {n: sum(v for v in cols[n].values() if v is not None) for n in names}
print(f"{'SUM(greedy)':<16} {sum(opt.values()):>4} {sum(bfs.values()):>5} "
      + " ".join(f"{sums[n]:>12.0f}" for n in names))

# verdict
if len(names) == 2:
    a, b = names
    da = sums[a] - sums[b]
    win = a if sums[a] < sums[b] else b
    print(f"\n=== verdict: {win} wins on per-instance greedy sum "
          f"({sums[a]:.0f} {a} vs {sums[b]:.0f} {b}, Δ={abs(da):.0f}) ===")
