#!/usr/bin/env python3
"""Sensitivity-sweep collector: walk a sweep's run dirs and emit two tables.

(i)  collected_diag.csv     — tidy long format, one row per
     cell × seed × problem × regime × checkpoint-frame, carrying the DIAGNOSTIC
     (non-selecting) metrics straight from each run's history.json.
(ii) axis_summary.csv       — per axis × axis-value, the median + IQR of the
     SELECTION metric (deploy-faithful val expansions) across seeds. This is the
     headline: it isolates each axis' effect on the metric that actually picks
     checkpoints.

Also writes collected_selection.csv (per cell × seed: best + last selection
metric). No plotting here — Part 3 plots are per-run.

Cell coordinates (gamma, lambda_ord, target_centering, fringe_size) + seed come
from each run's args.json["resolved_cell"]; the diagnostic rows + selection
metric come from history.json. Runs missing either file are skipped with a note.

Run:  python scripts/sweeps/collect_sensitivity.py --out-root <sweep dir>
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "lib" / "rl_handler"
sys.path.insert(0, str(RL))

from src.offline.sweep_cells import AXES  # noqa: E402

SELECT_KEY = "val_total_expansions"  # deploy-faithful selection metric


def _iqr(vals):
    """(q1, q3, iqr) by linear interpolation; degrades on tiny n."""
    xs = sorted(float(v) for v in vals)
    n = len(xs)
    if n == 0:
        return (None, None, None)
    if n == 1:
        return (xs[0], xs[0], 0.0)

    def q(p):
        pos = p * (n - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, n - 1)
        return xs[lo] + frac * (xs[hi] - xs[lo])

    q1, q3 = q(0.25), q(0.75)
    return (q1, q3, q3 - q1)


def _classify_ties(diag_rows):
    """Tag each diag row in place with regime_tie_class and return
    {group_key: class}. group_key = (gamma, lambda_ord, target_centering,
    fringe_size, seed, split, problem, frame) — the regime rows for one problem
    at one checkpoint of one run. See the call site for the classification rule."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in diag_rows:
        key = (r.get("gamma"), r.get("lambda_ord"), r.get("target_centering"),
               r.get("fringe_size"), r.get("seed"), r.get("split"),
               r.get("problem"), r.get("frame"))
        groups[key].append(r)
    group_class = {}
    for key, rows in groups.items():
        F = key[3]
        econ = [rr.get("node_economy") for rr in rows]
        tie = len(set(econ)) == 1            # EXACT integer equality, no epsilon
        fmax = rows[0].get("fmax")
        if not tie:
            klass = "distinct"
        elif fmax is None:                   # pre-fmax run: cause unknowable
            klass = "tie_unknown_fmax"
        elif F is not None and fmax < F:
            klass = "structural_nonresult"   # beam can't bind -> not a finding
        else:
            klass = "real_convergence"       # converged with room to differ
        group_class[key] = klass
        for rr in rows:
            rr["regime_tie_class"] = klass
    return group_class


def discover_runs(out_root: Path):
    """Yield (args.json dict, history.json dict, run_dir) for every run dir that
    has BOTH files (offline_main writes them per fringe into <dir>_fringe<F>)."""
    for hist in sorted(out_root.rglob("history.json")):
        run_dir = hist.parent
        ajson = run_dir / "args.json"
        if not ajson.exists():
            print(f"[skip] {run_dir}: no args.json")
            continue
        try:
            a = json.loads(ajson.read_text())
            h = json.loads(hist.read_text())
        except Exception as exc:
            print(f"[skip] {run_dir}: parse error {exc}")
            continue
        yield a, h, run_dir


def selection_metrics(history: dict):
    """(best, last) of the selection metric over checkpoints (None if absent)."""
    cks = history.get("checkpoints", [])
    vals = [c["summary"].get(SELECT_KEY) for c in cks
            if c.get("summary", {}).get(SELECT_KEY) is not None]
    if not vals:
        return None, None
    return min(vals), vals[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Collect a sensitivity sweep")
    p.add_argument("--out-root", required=True)
    args = p.parse_args()
    out_root = Path(args.out_root)
    if not out_root.exists():
        sys.exit(f"out-root does not exist: {out_root}")

    sel_rows = []   # per cell × seed
    diag_rows = []  # per cell × seed × problem × regime × frame
    n_runs = 0
    for ajson, hist, run_dir in discover_runs(out_root):
        n_runs += 1
        cell = ajson.get("resolved_cell") or {}
        seed = ajson.get("seed")
        coords = {a: cell.get(a) for a in AXES}
        best, last = selection_metrics(hist)
        sel_rows.append({**coords, "seed": seed, "run_dir": str(run_dir),
                         "best_val_expansions": best, "last_val_expansions": last})
        diag = hist.get("diag_per_regime", {})
        for split in ("train", "test"):
            for row in diag.get(split, []):
                diag_rows.append({**coords, "seed": seed, **row})

    # ---- regime-tie classification ----
    # Group the regime rows for ONE problem at ONE checkpoint of ONE run, i.e.
    # key = (cell coords, seed, split, problem, frame). A *tie* = every regime's
    # node_economy (integer expansions) is EXACTLY equal (exact equality only —
    # the structural case produces identical integers; an epsilon would falsely
    # flag genuine near-ties). The cause is what matters:
    #   fmax <  F  -> structural_nonresult : all regimes returned the same sub-F
    #                 pool (the beam can't bind), so they CAN'T differ — NOT a
    #                 finding; excluded from the regime-comparison aggregate.
    #   fmax >= F  -> real_convergence     : regimes had room to differ and still
    #                 agreed — a real finding; KEPT.
    # Never collapse the two tie causes — that distinction is the whole point.
    group_class = _classify_ties(diag_rows)

    # per-(cell, seed) prevalence of each class, counted over problem-checkpoint
    # GROUPS (not individual regime rows), so the count reflects how many
    # problem-checkpoints were non-results, not 5× that.
    prevalence: dict = {}
    for gkey, klass in group_class.items():
        ck_seed = gkey[:5]  # (gamma, lambda_ord, target_centering, fringe_size, seed)
        d = prevalence.setdefault(ck_seed, {"structural_nonresult": 0,
                                            "real_convergence": 0, "distinct": 0,
                                            "tie_unknown_fmax": 0})
        d[klass] = d.get(klass, 0) + 1

    # ---- (i) tidy diagnostic table ----
    diag_path = out_root / "collected_diag.csv"
    if diag_rows:
        cols = list(AXES) + ["seed"] + [c for c in diag_rows[0]
                                        if c not in AXES and c != "seed"]
        with diag_path.open("w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in diag_rows:
                w.writerow({k: r.get(k) for k in cols})

    # ---- selection table (+ tie-class prevalence per cell×seed) ----
    # The selection metric itself NEVER used the diag rows, so it is unchanged;
    # we only append the diagnostic tie-class GROUP counts so the prevalence of
    # structural non-results is visible alongside each cell's selection value.
    sel_path = out_root / "collected_selection.csv"
    sel_cols = list(AXES) + ["seed", "best_val_expansions", "last_val_expansions",
                             "n_structural_nonresult", "n_real_convergence",
                             "n_distinct", "run_dir"]
    with sel_path.open("w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=sel_cols)
        w.writeheader()
        for r in sel_rows:
            ck_seed = tuple(r.get(a) for a in AXES) + (r.get("seed"),)
            pv = prevalence.get(ck_seed, {})
            out = {k: r.get(k) for k in sel_cols}
            out["n_structural_nonresult"] = pv.get("structural_nonresult", 0)
            out["n_real_convergence"] = pv.get("real_convergence", 0)
            out["n_distinct"] = pv.get("distinct", 0)
            w.writerow(out)

    # ---- (ii) per-axis median/IQR of the selection metric ----
    # Marginal over each axis value: group every (cell, seed) best-selection
    # value by that axis' value. In one_at_a_time the baseline cell sits in every
    # axis' group, which is the correct shared reference for a marginal.
    axis_rows = []
    for axis in AXES:
        groups: dict = {}
        for r in sel_rows:
            if r.get("best_val_expansions") is None:
                continue
            groups.setdefault(r[axis], []).append(r["best_val_expansions"])
        for val, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
            q1, q3, iqr = _iqr(vals)
            axis_rows.append({
                "axis": axis, "value": val, "n": len(vals),
                "median_best_val_expansions": round(median(vals), 3),
                "q1": (round(q1, 3) if q1 is not None else None),
                "q3": (round(q3, 3) if q3 is not None else None),
                "iqr": (round(iqr, 3) if iqr is not None else None),
            })
    axis_path = out_root / "axis_summary.csv"
    with axis_path.open("w", newline="") as fh:
        w = csvmod.DictWriter(
            fh, fieldnames=["axis", "value", "n",
                            "median_best_val_expansions", "q1", "q3", "iqr"])
        w.writeheader()
        for r in axis_rows:
            w.writerow(r)

    # ---- regime-comparison aggregate (EXCLUDES structural_nonresult) ----
    # The point of the study: compare regimes on the normalized node-economy
    # ratio. Structural non-results (the beam can't bind) would wash out real
    # differences, so they are dropped here; their prevalence stays visible in
    # collected_selection.csv. Grouped per (cell × split × regime).
    agg_path = out_root / "collected_diag_regime_agg.csv"
    agg_rows = []
    if diag_rows:
        agg: dict = {}
        for r in diag_rows:
            if r.get("regime_tie_class") == "structural_nonresult":
                continue  # not a finding — excluded from the regime comparison
            key = (tuple(r.get(a) for a in AXES), r.get("split"), r.get("regime"))
            ne = r.get("node_economy_ratio")
            if ne is None:
                continue
            agg.setdefault(key, []).append(float(ne))
        for (coords, split, regime), vals in sorted(
                agg.items(), key=lambda kv: tuple(str(x) for x in kv[0][0]) +
                (str(kv[0][1]), str(kv[0][2]))):
            row = dict(zip(AXES, coords))
            row.update({"split": split, "regime": regime, "n_kept": len(vals),
                        "median_node_economy_ratio": round(median(vals), 4)})
            agg_rows.append(row)
        with agg_path.open("w", newline="") as fh:
            w = csvmod.DictWriter(
                fh, fieldnames=list(AXES) + ["split", "regime", "n_kept",
                                             "median_node_economy_ratio"])
            w.writeheader()
            for r in agg_rows:
                w.writerow(r)

    n_struct = sum(1 for k in group_class.values() if k == "structural_nonresult")
    n_real = sum(1 for k in group_class.values() if k == "real_convergence")
    n_dist = sum(1 for k in group_class.values() if k == "distinct")
    n_unk = sum(1 for k in group_class.values() if k == "tie_unknown_fmax")
    print(f"[collect] {n_runs} runs, {len(sel_rows)} cell×seed, "
          f"{len(diag_rows)} diag rows")
    print(f"[collect] wrote {sel_path}")
    if diag_rows:
        print(f"[collect] wrote {diag_path}")
        print(f"[collect] wrote {agg_path}")
    print(f"[collect] wrote {axis_path}")
    print(f"[collect] regime-tie groups: distinct={n_dist} "
          f"real_convergence={n_real} structural_nonresult={n_struct}"
          + (f" tie_unknown_fmax={n_unk}" if n_unk else "")
          + "  (structural_nonresult EXCLUDED from the regime aggregate)")
    print("[collect] per-axis median[IQR] of selection metric "
          f"({SELECT_KEY}, lower=better; diag rows never feed this):")
    for r in axis_rows:
        print(f"    {r['axis']:16s} {str(r['value']):12s} "
              f"median={r['median_best_val_expansions']} "
              f"IQR={r['iqr']} (n={r['n']})")


if __name__ == "__main__":
    main()
