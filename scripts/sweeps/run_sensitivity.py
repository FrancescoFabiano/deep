#!/usr/bin/env python3
"""Sensitivity-sweep runner: expand the four axis lists into cells × seeds and
build one offline_main invocation per (cell, seed).

The expansion logic is SHARED with offline_main (src/offline/sweep_cells.py), so
the runner and the in-process manifest never drift. offline_main itself always
trains a single cell; this runner is the only place the sweep fans out.

SAFE BY DEFAULT: without --execute it prints the plan and exits (a dry run). Pass
--execute to actually launch, bounded by --max-parallel. Use --dry-run to force
print-only even alongside other flags.

Example (dry run, both modes are just a flag flip):
  python scripts/sweeps/run_sensitivity.py \
      --gamma 0.0 0.9 0.99 --lambda-ord 0.0 1.0 \
      --target-centering absolute fringe_mean --fringe-sizes 32 64 \
      --sweep-mode one_at_a_time --seeds 0 1 2 \
      --train-csv <a.csv> <b.csv> --val-csv <c.csv> \
      --out-root exp/rl_exp/sensitivity/run1 --frames 30000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "lib" / "rl_handler"
sys.path.insert(0, str(RL))

from src.offline.sweep_cells import (  # noqa: E402
    AXES,
    cell_tag,
    resolve_cells,
)

OFFLINE_MAIN = RL / "offline_main.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity sweep runner")
    # axes
    p.add_argument("--gamma", nargs="+", type=float, default=[0.99])
    p.add_argument("--lambda-ord", nargs="+", type=float, default=[0.0])
    p.add_argument("--target-centering", nargs="+",
                   choices=["absolute", "fringe_mean"], default=["absolute"])
    p.add_argument("--fringe-sizes", nargs="+", type=int, default=[32])
    p.add_argument("--sweep-mode", choices=["product", "one_at_a_time"],
                   default="one_at_a_time")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # data + budget (forwarded to offline_main)
    p.add_argument("--train-csv", nargs="+", required=True)
    p.add_argument("--val-csv", nargs="+", required=True)
    p.add_argument("--test-csv", nargs="+", default=[])
    p.add_argument("--frames", type=int, default=30_000)
    p.add_argument("--n-checkpoints", type=int, default=10)
    p.add_argument("--kind-of-data", choices=["merged", "separated"],
                   default="merged")
    p.add_argument("--dataset-type", choices=["MAPPED", "HASHED", "BITMASK"],
                   default="HASHED")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="Everything after --extra is forwarded verbatim to offline_main "
        "(e.g. --extra --train-expansion-cap 200 --warmup 100).",
    )
    # control
    p.add_argument("--out-root", required=True,
                   help="Root dir for the sweep; each run lands in "
                   "<out-root>/<cell_tag>/seed<s>.")
    p.add_argument("--max-parallel", type=int, default=2,
                   help="Concurrency cap on launched offline_main subprocesses.")
    p.add_argument("--execute", action="store_true", default=False,
                   help="Actually launch. Without it the runner only prints the "
                   "plan (safe default — never launches the full set).")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Force print-only even if --execute is also passed.")
    p.add_argument("--no-regimes", action="store_true", default=False,
                   help="Omit --use-regimes (the diagnostic surface is regime-"
                   "only; default ON for the sweep).")
    return p.parse_args()


def build_plan(args) -> list:
    cells = resolve_cells(args.gamma, args.lambda_ord, args.target_centering,
                          args.fringe_sizes, mode=args.sweep_mode)
    out_root = Path(args.out_root)
    plan = []
    for cell in cells:
        tag = cell_tag(cell)
        for seed in args.seeds:
            out_dir = out_root / tag / f"seed{seed}"
            cmd = [sys.executable, str(OFFLINE_MAIN)]
            if not args.no_regimes:
                cmd.append("--use-regimes")
            cmd += ["--train-csv", *args.train_csv]
            cmd += ["--val-csv", *args.val_csv]
            if args.test_csv:
                cmd += ["--test-csv", *args.test_csv]
            cmd += ["--gamma", repr(cell["gamma"])]
            cmd += ["--lambda-ord", repr(cell["lambda_ord"])]
            cmd += ["--target-centering", str(cell["target_centering"])]
            cmd += ["--fringe-sizes", str(int(cell["fringe_size"]))]
            cmd += ["--frames", str(args.frames)]
            cmd += ["--n-checkpoints", str(args.n_checkpoints)]
            cmd += ["--seed", str(seed)]
            cmd += ["--kind-of-data", args.kind_of_data]
            cmd += ["--dataset-type", args.dataset_type]
            if args.device:
                cmd += ["--device", args.device]
            cmd += ["--dir-save-model", str(out_dir)]
            if args.extra:
                cmd += list(args.extra)
            plan.append({"cell": cell, "tag": tag, "seed": seed,
                         "out_dir": str(out_dir), "cmd": cmd})
    return cells, plan


def _run_one(item) -> tuple:
    proc = subprocess.run(item["cmd"], cwd=str(REPO))
    return item["tag"], item["seed"], proc.returncode


def main() -> None:
    args = parse_args()
    cells, plan = build_plan(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sweep_mode": args.sweep_mode,
        "axes": {"gamma": args.gamma, "lambda_ord": args.lambda_ord,
                 "target_centering": args.target_centering,
                 "fringe_sizes": [int(f) for f in args.fringe_sizes]},
        "seeds": args.seeds,
        "n_cells": len(cells),
        "n_runs": len(plan),
        "cells": cells,
        "runs": [{"tag": p["tag"], "seed": p["seed"], "out_dir": p["out_dir"],
                  "cell": p["cell"]} for p in plan],
    }
    (out_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[plan] sweep_mode={args.sweep_mode}  "
          f"{len(cells)} cells × {len(args.seeds)} seeds = {len(plan)} runs")
    print(f"[plan] axes: gamma={args.gamma} lambda_ord={args.lambda_ord} "
          f"target_centering={args.target_centering} "
          f"fringe_sizes={manifest['axes']['fringe_sizes']}")
    for p in plan:
        print(f"  [{p['tag']} seed{p['seed']}] -> {p['out_dir']}")
        print("    " + " ".join(p["cmd"]))
    print(f"[manifest] {out_root / 'sweep_manifest.json'}")

    if args.dry_run or not args.execute:
        print("[dry-run] nothing launched. Re-run with --execute to launch "
              f"(cap --max-parallel={args.max_parallel}).")
        return

    print(f"[execute] launching {len(plan)} runs, max_parallel={args.max_parallel}")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        futs = [ex.submit(_run_one, item) for item in plan]
        for fut in as_completed(futs):
            tag, seed, rc = fut.result()
            status = "ok" if rc == 0 else f"FAIL rc={rc}"
            print(f"[done] {tag} seed{seed}: {status}")
            results.append((tag, seed, rc))
    n_fail = sum(1 for _, _, rc in results if rc != 0)
    print(f"[execute] complete: {len(results) - n_fail} ok, {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
