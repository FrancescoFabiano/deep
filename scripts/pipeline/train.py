"""Stage 2: one model per (domain, kind, F) from the train-split trees.

rl  -> lib/rl_handler/offline_main.py   (fringe ranker, one run per F)
gnn -> lib/gnn_handler/__main__.py      (distance estimator; one run exports every F)
Installed flat under models/<domain>/:
    rl_F<F>.onnx                         RL search + RL_H heuristic
    gnn_F<F>.onnx                        RL (beam) search with the GNN as ranker
    gnn_F<F>_state.onnx + _state_C.txt   HFS / A* with --heuristics GNN
Flags the trainers accept but this file does not name go in [train.rl].extra / [train.gnn].extra.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from . import instances
from .config import REPO, Config

sys.path.insert(0, str(REPO / "lib"))
from deep_nn.strategies import discover_tables, select_tables  # noqa: E402

TRAINERS = {"rl": REPO / "lib" / "rl_handler" / "offline_main.py",
            "gnn": REPO / "lib" / "gnn_handler" / "__main__.py"}
# Every EPDDL problem is its own configuration; the RL trainer refuses such a split by default.
FIXED_FLAGS = {"rl": ["--allow-cross-config"], "gnn": []}
EXPORTS = {   # (file the trainer writes under run_fringe<F>/, installed name)
    "rl": [("frontier_policy_{F}_best_by_expansions.onnx", "rl_F{F}.onnx")],
    "gnn": [("distance_estimator_{F}.onnx", "gnn_F{F}.onnx"),
            ("distance_estimator_{F}_state.onnx", "gnn_F{F}_state.onnx"),
            ("distance_estimator_{F}_state_C.txt", "gnn_F{F}_state_C.txt")],
}


def run(cfg: Config) -> None:
    insts = instances.load(cfg)
    for domain in sorted({i.domain for i in insts}):
        train_csvs, test_csvs = _tables(cfg, insts, domain)
        if not train_csvs:
            print(f"[train] {domain}: no train trees under {cfg.data_dir / domain}, skipped")
            continue
        for kind in cfg.train["models"]:
            _train(cfg, domain, kind, train_csvs, test_csvs)


def _tables(cfg, insts, domain):
    tables = discover_tables(cfg.data_dir / domain)
    strategies = None if cfg.train["strategies"] == "all" else cfg.train["strategies"]

    def pick(split):
        mine = {i.problem for i in insts if i.domain == domain and i.split == split}
        sub = {p: t for p, t in tables.items() if p in mine}
        return select_tables(sub, strategies, verbose=False) if sub else []
    return pick("train"), pick("test")


def _train(cfg, domain, kind, train_csvs, test_csvs) -> None:
    out = cfg.models_dir / domain
    installed = [(out / dst.format(F=F)) for F in cfg.train["fringe_sizes"] for _, dst in EXPORTS[kind]]
    if all(p.exists() for p in installed):
        print(f"[train] {domain}/{kind}: models present, skipped")
        return
    t = cfg.train
    cmd = [sys.executable, str(TRAINERS[kind]), "--train-csv", *map(str, train_csvs),
           "--dir-save-model", str(out / kind / "run"), "--fringe-sizes", *map(str, t["fringe_sizes"]),
           "--epochs", str(t["epochs"]), "--batch-size", str(t["batch_size"]), "--seed", str(t["seed"]),
           "--dataset-type", cfg.data["dataset_type"], *FIXED_FLAGS[kind], *map(str, t.get(kind, {}).get("extra", []))]
    if test_csvs:
        cmd += ["--test-csv", *map(str, test_csvs)]
    print(f"[train] {domain}/{kind}: {len(train_csvs)} train tables, {len(test_csvs)} test tables")
    print(" ".join(cmd))
    if cfg.dry_run:
        return
    subprocess.run(cmd, cwd=REPO, check=True)
    for F in t["fringe_sizes"]:
        for src, dst in EXPORTS[kind]:
            shutil.copy2(out / kind / f"run_fringe{F}" / src.format(F=F), out / dst.format(F=F))
    print(f"[train] {domain}/{kind}: installed {[p.name for p in installed]}")
