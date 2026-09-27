"""Train the GNN distance estimator for every domain of a batch.

Twin of scripts/rl_exp/train_models.py: the same data discovery (one generation
table per (instance, strategy) under <batch>/_models/<domain>/training_data,
held-out tables under test_data) drives lib/gnn_handler/__main__.py instead of
the RL trainer.  TRAIN = all of training_data, TEST = all of test_data; nothing
is carved out.  Flags this script does not know are forwarded verbatim.

    python scripts/gnn_exp/train_models.py exp/gnn_exp/batch1 --fringe-sizes 4 8 -- --epochs 50

The exported ``distance_estimator_<F>.onnx`` is installed beside the RL one
(``<batch>/_models/<domain>/``); deploy it with scripts/rl_exp/bulk_coverage_run.py
and ``RL_MODEL_BASENAME=distance_estimator``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_MAIN = REPO_ROOT / "lib" / "gnn_handler" / "__main__.py"
sys.path.insert(0, str(REPO_ROOT / "lib"))
from deep_nn.strategies import parse_strategy_list  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts" / "rl_exp"))
from train_models import domain_test_csvs, domain_train_csvs, find_domains  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_root", help="batch folder holding _models/<domain>/training_data")
    p.add_argument("--no_goal", action="store_true", help="train separated models (--kind-of-data separated)")
    p.add_argument("--fringe-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="+", default=None,
                   help="restrict to these generation strategies (default: every table on disk)")
    args, forwarded = p.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    strategies = parse_strategy_list(args.strategies) if args.strategies else None

    models_root = Path(args.batch_root) / "_models"
    domains = find_domains(models_root)
    if not domains:
        raise SystemExit(f"no _models/<domain>/training_data under {args.batch_root}")

    for domain in domains:
        train_csvs = domain_train_csvs(models_root, domain, strategies)
        test_csvs = domain_test_csvs(models_root, domain, strategies)
        if not train_csvs:
            print(f"[{domain}] no training_data tables, skipped")
            continue
        domain_dir = models_root / domain
        cmd = [sys.executable, str(GNN_MAIN), "--seed", str(args.seed),
               "--dir-save-model", str(domain_dir / f"seed{args.seed}"),
               "--train-csv", *(str(c.resolve()) for c in train_csvs)]
        if test_csvs:
            cmd += ["--test-csv", *(str(c.resolve()) for c in test_csvs)]
        if args.no_goal:
            cmd += ["--kind-of-data", "separated"]
        cmd += ["--fringe-sizes", *map(str, args.fringe_sizes), *forwarded]
        print(f"[{domain}] {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            raise SystemExit(f"[{domain}] gnn_handler exited with code {rc}")
        for F in args.fringe_sizes:
            exported = domain_dir / f"seed{args.seed}_fringe{F}" / f"distance_estimator_{F}.onnx"
            if exported.exists():
                shutil.copy2(exported, domain_dir / exported.name)
                print(f"[{domain}] installed {domain_dir / exported.name}")


if __name__ == "__main__":
    main()
