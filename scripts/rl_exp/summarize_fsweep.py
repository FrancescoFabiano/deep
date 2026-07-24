#!/usr/bin/env python3
"""Summarize the F-sweep (or any seedNN_fringeF run set): per-F best
val_total_expansions + per-instance greedy vs BFS vs optimal.

Best checkpoint = min val_total_expansions across the saved checkpoints.
Usage: python3 scripts/rl_exp/summarize_fsweep.py <models_root_with_seedNN_fringeF_dirs>
       (default: exp/rl_exp/retrain_fsweep/_models/CC)
"""
import json, sys, re
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "exp/rl_exp/retrain_fsweep/_models/CC")
dirs = sorted(root.glob("seed*_fringe*"),
              key=lambda p: int(re.search(r"fringe(\d+)", p.name).group(1)))
if not dirs:
    sys.exit(f"no seed*_fringe* dirs under {root}")

per_f = {}
for d in dirs:
    F = int(re.search(r"fringe(\d+)", d.name).group(1))
    hj = d / "history.json"
    if not hj.exists():
        print(f"[F={F:>3}] no history.json (not started)"); continue
    h = json.load(open(hj))
    cks = h.get("checkpoints", [])
    if not cks:
        print(f"[F={F:>3}] history.json but 0 checkpoints yet"); continue
    onnx = list(d.glob("frontier_policy_*_best_by_expansions.onnx"))
    status = "DONE" if onnx else f"IN-PROGRESS ({len(cks)} ckpts)"
    best = min(cks, key=lambda c: c["summary"]["val_total_expansions"])
    per_f[F] = (best, status, len(cks))

# ---- table 1: per-F best val_total_expansions ----
print("\n=== per-F best val_total_expansions (min over checkpoints) ===")
print(f"{'F':>4} {'status':<20} {'best_val_total':>14} {'@frame':>8} {'spearman':>9}")
for F in sorted(per_f):
    best, status, n = per_f[F]
    s = best["summary"]
    print(f"{F:>4} {status:<20} {s['val_total_expansions']:>14.0f} "
          f"{s['frame']:>8} {s.get('val_spearman_all', float('nan')):>9.3f}")

# ---- table 2: per-instance greedy at each F's best checkpoint ----
insts = None
for F in sorted(per_f):
    best = per_f[F][0]
    vpi = best["val_per_instance"]
    if insts is None:
        insts = sorted(vpi.keys())
        hdr = f"{'instance':<16} {'opt':>4} {'bfs':>5} " + " ".join(f"F{F:>4}" for F in sorted(per_f))
        print("\n=== per-instance greedy expansions at each F's best checkpoint ===")
        print(hdr)
    # build rows once we have all Fs; collect into dict
per_inst_rows = {i: {} for i in (insts or [])}
opt = {}; bfs = {}
for F in sorted(per_f):
    vpi = per_f[F][0]["val_per_instance"]
    for i, rec in vpi.items():
        per_inst_rows.setdefault(i, {})[F] = rec.get("greedy_mean_expansions")
        opt[i] = rec.get("optimal_expansions"); bfs[i] = rec.get("bfs_expansions")
for i in sorted(per_inst_rows):
    cells = " ".join(f"{per_inst_rows[i].get(F, float('nan')):>5.0f}"
                     if per_inst_rows[i].get(F) is not None else f"{'-':>5}"
                     for F in sorted(per_f))
    print(f"{i:<16} {opt.get(i,'-'):>4} {bfs.get(i,'-'):>5} {cells}")

# totals row
tot = {F: sum(v for v in per_inst_rows_col.values() if v is not None)
       for F, per_inst_rows_col in
       {F: {i: per_inst_rows[i].get(F) for i in per_inst_rows} for F in sorted(per_f)}.items()}
print(f"{'SUM(greedy)':<16} {sum(v for v in opt.values() if isinstance(v,(int,float))):>4} "
      f"{sum(v for v in bfs.values() if isinstance(v,(int,float))):>5} "
      + " ".join(f"{tot[F]:>5.0f}" for F in sorted(per_f)))
