#!/usr/bin/env python3
"""Compare C++ deployment eval (bulk_coverage_run summary CSVs) across arms.

Reads exp/rl_exp/attn_ab/eval_results/{arm}_{split}.csv (per-instance rows then
a blank line then a Solved,Total,... summary row) for arms mean_pool and
self_attention on splits Test/Training, and prints the comparison tables +
per-metric verdict. Read-only; no source edits.
"""
import csv, sys
from pathlib import Path

OUT = Path("exp/rl_exp/attn_ab/eval_results")
ARMS = ["mean_pool", "self_attention"]
SPLITS = ["Train", "Test"]          # display order
FILEMAP = {"Train": "Training", "Test": "Test"}

def load(arm, split):
    p = OUT / f"{arm}_{FILEMAP[split]}.csv"
    if not p.exists():
        return None
    rows, summary, in_summary, hdr2 = [], None, False, None
    for line in p.read_text().splitlines():
        if not line.strip():
            in_summary = True; continue
        if in_summary and hdr2 is None:
            hdr2 = line.split(","); continue
        if in_summary:
            summary = dict(zip(hdr2, line.split(","))); continue
        rows.append(line)
    reader = csv.DictReader(rows)
    per = {r["File"].replace(".txt", ""): r for r in reader}
    return {"per": per, "summary": summary}

data = {(a, s): load(a, s) for a in ARMS for s in SPLITS}

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def agg(d):
    """Recompute from per-instance rows (solved-only means)."""
    per = d["per"]
    solved = [r for r in per.values() if r["GoalFound"] == "Yes"]
    def mean(col):
        xs = [num(r[col]) for r in solved if num(r[col]) is not None]
        return sum(xs) / len(xs) if xs else None
    return {
        "solved": len(solved), "total": len(per),
        "nodes": mean("NodesExpanded"), "plan": mean("PlanLength"),
        "time_ms": mean("TotalExecutionTime"),
    }

print("=== C++ deployment eval: RL search (--heuristics RL_H --RL_heuristics MAX), F=32, strict, 200s cap ===\n")
# ---- summary table ----
hdr = f"{'metric':<20}" + "".join(f"{a[:14]+'/'+s:>18}" for a in ARMS for s in SPLITS)
print(hdr)
metrics = [("solved", lambda g: f"{g['solved']}/{g['total']}"),
           ("mean_nodes", lambda g: f"{g['nodes']:.1f}" if g['nodes'] is not None else "-"),
           ("mean_plan_len", lambda g: f"{g['plan']:.2f}" if g['plan'] is not None else "-"),
           ("mean_time_ms", lambda g: f"{g['time_ms']:.0f}" if g['time_ms'] is not None else "-")]
G = {(a, s): (agg(data[(a, s)]) if data[(a, s)] else None) for a in ARMS for s in SPLITS}
for name, fmt in metrics:
    row = f"{name:<20}"
    for a in ARMS:
        for s in SPLITS:
            row += f"{(fmt(G[(a,s)]) if G[(a,s)] else 'NA'):>18}"
    print(row)

# ---- per-instance Test table ----
print("\n=== per-instance NodesExpanded (Test) — G=goal, . = solved, X=unsolved ===")
mp, sa = data[("mean_pool", "Test")], data[("self_attention", "Test")]
if mp and sa:
    insts = sorted(set(mp["per"]) | set(sa["per"]))
    print(f"{'instance':<18} {'mean_pool':>12} {'self_attn':>12}   winner")
    tot_mp = tot_sa = 0; both = 0; wins_mp = wins_sa = 0
    for i in insts:
        rmp, rsa = mp["per"].get(i), sa["per"].get(i)
        def cell(r):
            if r is None: return "  -"
            if r["GoalFound"] != "Yes": return f"X({r['GoalFound']})"
            return r["NodesExpanded"]
        cmp = ""
        if rmp and rsa and rmp["GoalFound"] == "Yes" == rsa["GoalFound"]:
            nmp, nsa = int(rmp["NodesExpanded"]), int(rsa["NodesExpanded"])
            tot_mp += nmp; tot_sa += nsa; both += 1
            if nsa < nmp: cmp = "self_attn"; wins_sa += 1
            elif nmp < nsa: cmp = "mean_pool"; wins_mp += 1
            else: cmp = "tie"
        print(f"{i:<18} {cell(rmp):>12} {cell(rsa):>12}   {cmp}")
    print(f"\n  both-solved instances: {both}/{len(insts)} | "
          f"sum nodes (both-solved only): mean_pool={tot_mp}  self_attn={tot_sa}")
    print(f"  per-instance wins: self_attn={wins_sa}  mean_pool={wins_mp}")

# ---- verdict ----
print("\n=== VERDICT ===")
for s in SPLITS:
    gm, gs = G[("mean_pool", s)], G[("self_attention", s)]
    if not (gm and gs): continue
    def win(k, lower=True):
        vm, vs = gm[k], gs[k]
        if vm is None or vs is None: return "NA"
        if vm == vs: return "tie"
        better = (vs < vm) if lower else (vs > vm)
        return "self_attention" if better else "mean_pool"
    print(f" [{s}] success: {'self_attention' if gs['solved']>gm['solved'] else ('mean_pool' if gm['solved']>gs['solved'] else 'tie')}"
          f" ({gm['solved']}/{gm['total']} vs {gs['solved']}/{gs['total']}) | "
          f"nodes: {win('nodes')} | plan_len: {win('plan')} | time: {win('time_ms')}")
print("\n=== C++ EVAL COMPLETE ===")
