#!/usr/bin/env python3
"""One experiment = one folder under exp/trials holding a trial.toml.

    python scripts/trial.py data   exp/trials/basic [--domains gossip] [--strategies BFS HFS]
    python scripts/trial.py train  exp/trials/basic
    python scripts/trial.py infer  exp/trials/basic
    python scripts/trial.py report exp/trials/basic
    python scripts/trial.py all    exp/trials/basic

Every stage reads the same config and writes only inside the trial folder:
data/ (generation trees), models/ (ONNX), results/results.csv, report/ (tables, figures).
Stages are idempotent: what is already on disk is skipped, so a crashed run resumes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import config, data, infer, report, train  # noqa: E402

STAGES = {"data": data.run, "train": train.run, "infer": infer.run, "report": report.run}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("trial_dir")
    p.add_argument("--domains", nargs="+", help="restrict to these domains (default: all in instances/)")
    p.add_argument("--strategies", nargs="+", help="data: generate only these strategies")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)     # progress lines reach a redirected log as they happen
    cfg = config.load(Path(a.trial_dir), domains=a.domains, strategies=a.strategies, dry_run=a.dry_run)
    for name in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"===== {name} =====")
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
