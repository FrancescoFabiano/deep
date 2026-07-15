#!/usr/bin/env python3
"""Generate FAITHFUL training tables, then gate them.

    python3 scripts/rl_exp/generate_faithful.py <out_dir> --domains CC SCRich

No discarding; per-domain depth bounds the tree under the fixed 50k ceiling.
Runs the faithfulness check immediately and writes `faithful_pool.json`.

WHY (see lib/rl_handler/src/offline/generation.py for the full account):
the shipped tables used --dataset_discard_factor 0.4, whose discard is BIASED
(+0.2 right after a goal is found) and therefore deletes the SHALLOW GOALS that make
an instance easy, while writing the discarded states to the CSV as childless leaves.
Measured on CC_2_2_3__pl_4 (planner BFS true optimal = 4):
    discard 0.4 -> delta_root 10, sterile 30.0%   (optimal path ABSENT)
    discard 0   -> delta_root  4, sterile  2.9%   (optimal path present)

DO NOT raise --dataset_max_creation: it is a hard stop that POISONS (past it,
non-goals are dropped AND their parents inherit 1e6 = unreachable). Depth is what
keeps the count under the ceiling.

Coordinates with the machine: `nice`s every planner call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lib" / "rl_handler"))

from src.offline.faithfulness import build_faithful_pool  # noqa: E402
from src.offline.generation import (  # noqa: E402
    depth_for,
    domain_of,
    generation_argv,
    generation_manifest,
)
from src.offline.tree import load_tree_instance  # noqa: E402

PROBLEM_DIRS = {
    "CC": REPO / "exp/all/CC",
    "SC": REPO / "exp/all/SC_Multi",
    "SCRich": REPO / "exp/all/SC_Multi_Rich",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--domains", nargs="+", default=["CC"])
    ap.add_argument("--families", nargs="*", default=None,
                    help="e.g. CC_2_3_4 CC_2_2_3; default: every family in the domain")
    ap.add_argument("--deep-exe", default=str(REPO / "cmake-build-release-nn/bin/deep"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-type", default="HASHED",
                    help="opaque to the pipeline; BITMASK needs no downstream change")
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    problems = []
    for dom in a.domains:
        d = PROBLEM_DIRS.get(dom)
        if d is None or not d.exists():
            print(f"[gen] no problem dir for domain {dom!r}", file=sys.stderr)
            continue
        for fam_dir in sorted(d.iterdir()):
            if not fam_dir.is_dir():
                continue
            if a.families and fam_dir.name not in a.families:
                continue
            for p in sorted(fam_dir.glob("*.txt")):
                if "wrong" in p.stem:
                    continue
                problems.append(p)

    print(f"[gen] {len(problems)} problems across {a.domains}")
    manifests = []
    for p in problems:
        name = p.stem
        depth = depth_for(name)
        argv = [a.deep_exe] + generation_argv(p, name, seed=a.seed,
                                              dataset_type=a.dataset_type)
        print(f"[gen] {name:24s} domain={domain_of(name):8s} depth={depth}")
        if a.dry_run:
            print("       " + " ".join(argv))
            continue
        wd = out / name
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = subprocess.run(["nice", "-n", str(a.nice)] + argv, cwd=str(wd),
                           capture_output=True, text=True, timeout=a.timeout_s)
        manifests.append(generation_manifest(
            name, seconds=round(time.time() - t0, 1), returncode=r.returncode))
        if r.returncode != 0:
            print(f"       FAILED rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}")

    if a.dry_run:
        return 0
    (out / "generation_manifest.json").write_text(json.dumps(manifests, indent=1))

    # ---- gate the output: nothing trains on data that fails this ----
    insts = []
    for csv in sorted(out.glob("**/*_depth_*.csv")):
        try:
            insts.append(load_tree_instance(csv, name=csv.parent.name))
        except Exception as e:
            print(f"[gen] unreadable {csv}: {e}")
    build_faithful_pool(insts, out_path=out / "faithful_pool.json")
    print(f"[gen] wrote {out/'faithful_pool.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
