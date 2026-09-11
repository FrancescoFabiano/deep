#!/usr/bin/env python3
"""Side-by-side comparison of two batches' PLANNER results (pipeline.py output).

    python3 scripts/rl_exp/compare_batches.py combined_results/batch1_cc_strat/CC \
                                             combined_results/batch1_cc_unified/CC \
                                             [--label-a old --label-b new] [--out cmp.csv]

Reads each side's per-run summary CSVs (`<split>_<label>_<strict>.csv` under
`bfs/` and `fringe_<F>/`), i.e. one row per problem instance, and compares the
two sides CELL BY CELL (same split, same search config, same F):

  * solved counts per side;
  * on the instances BOTH sides solved: mean nodes expanded and mean plan length
    per side, and how many instances each side expanded fewer nodes on;
  * instances solved by one side only.

Nothing here is a verdict: the table is the evidence, the reading lives in the
report. BFS rows are included as the fixed reference (they should be identical on
both sides -- if they are not, the two batches did not run the same problems).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _read_instance_table(p: Path) -> dict:
    """{instance: row} from the instance section of a summary CSV (stops at the
    blank line that precedes the stats section)."""
    out = {}
    with p.open() as fh:
        rdr = csv.reader(fh)
        header = None
        for row in rdr:
            if not row or not any(row):
                break
            if header is None:
                header = row
                continue
            if len(row) != len(header):
                break
            d = dict(zip(header, row))
            out[d["File"]] = d
    return out


def _cells(root: Path) -> dict:
    """{(split, label, fringe, strict): {instance: row}}"""
    cells = {}
    for p in sorted(root.glob("*/*.csv")):
        d = p.parent.name
        if d == "bfs":
            F = 0
        elif d.startswith("fringe_"):
            F = int(d.split("_", 1)[1])
        else:
            continue
        name = p.stem                      # test_RL-H-RNG_strict
        split, _, rest = name.partition("_")
        label, _, strict = rest.rpartition("_")
        cells[(split, label, F, strict)] = _read_instance_table(p)
    return cells


def _solved(row) -> bool:
    return row.get("GoalFound", "").strip().lower() == "yes"


def _num(row, k):
    try:
        return float(row[k])
    except (KeyError, ValueError):
        return None


def compare(a: dict, b: dict, la: str, lb: str):
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], k[2], k[1], k[3]))
    rows = []
    for k in keys:
        ra, rb = a.get(k, {}), b.get(k, {})
        insts = sorted(set(ra) | set(rb))
        sa = {i for i in insts if i in ra and _solved(ra[i])}
        sb = {i for i in insts if i in rb and _solved(rb[i])}
        both = sorted(sa & sb)
        na = [_num(ra[i], "NodesExpanded") for i in both]
        nb = [_num(rb[i], "NodesExpanded") for i in both]
        pa = [_num(ra[i], "PlanLength") for i in both]
        pb = [_num(rb[i], "PlanLength") for i in both]
        fewer_a = sum(1 for x, y in zip(na, nb) if x is not None and y is not None and x < y)
        fewer_b = sum(1 for x, y in zip(na, nb) if x is not None and y is not None and y < x)
        rows.append({
            "split": k[0], "config": k[1], "fringe": k[2], "strict": k[3],
            "n_instances": len(insts),
            f"solved_{la}": len(sa), f"solved_{lb}": len(sb),
            "n_common_solved": len(both),
            f"nodes_{la}": _mean(na), f"nodes_{lb}": _mean(nb),
            f"fewer_nodes_{la}": fewer_a, f"fewer_nodes_{lb}": fewer_b,
            f"plan_{la}": _mean(pa), f"plan_{lb}": _mean(pb),
            f"only_{la}": ";".join(sorted(sa - sb)),
            f"only_{lb}": ";".join(sorted(sb - sa)),
        })
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", type=Path, default=None, help="write the table as CSV")
    args = ap.parse_args()
    la, lb = args.label_a, args.label_b
    ca, cb = _cells(args.a), _cells(args.b)
    if not ca or not cb:
        sys.exit(f"no summary CSVs under {args.a} ({len(ca)}) / {args.b} ({len(cb)})")
    rows = compare(ca, cb, la, lb)
    cols = ["split", "config", "fringe", "strict", "n_instances",
            f"solved_{la}", f"solved_{lb}", "n_common_solved",
            f"nodes_{la}", f"nodes_{lb}", f"fewer_nodes_{la}", f"fewer_nodes_{lb}",
            f"plan_{la}", f"plan_{lb}"]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
    # totals over the RL cells (BFS is the reference, not an arm)
    rl = [r for r in rows if r["fringe"] > 0]
    tot = lambda k: sum(r[k] or 0 for r in rl)
    print(f"\nRL cells: {len(rl)}; total solved {la}={tot(f'solved_{la}')} "
          f"{lb}={tot(f'solved_{lb}')}; fewer nodes on common-solved: "
          f"{la}={tot(f'fewer_nodes_{la}')} {lb}={tot(f'fewer_nodes_{lb}')} "
          f"(ties not counted)")
    diff = [r for r in rows if r["fringe"] > 0 and r[f"only_{la}"] or r[f"only_{lb}"]]
    for r in diff:
        if r[f"only_{la}"] or r[f"only_{lb}"]:
            print(f"  {r['split']} {r['config']} F={r['fringe']}: only {la}: "
                  f"[{r[f'only_{la}']}]  only {lb}: [{r[f'only_{lb}']}]")
    if args.out:
        with args.out.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
