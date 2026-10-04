"""Stage 2: models from the train-split trees, per domain, or with [train].pooled from
every domain's train trees at once (models/pooled/).

rl  -> lib/rl_handler/offline_main.py   fringe ranker: one run per [train].fringe_sizes entry
gnn -> lib/gnn_handler/__main__.py      per-state distance estimator: ONE run, exported at
                                        every GNN_FRINGE_SIZES width from the same weights
Installed flat under models/<domain>/ (models/pooled/ when [train].pooled):
    rl_F<F>.onnx                         RL search + RL_H heuristic
    gnn_F<F>.onnx                        RL (beam) search with the GNN as ranker
    gnn_state.onnx + gnn_state_C.txt     HFS / A* with --heuristics GNN (one state at a time)
Flags the trainers accept but this file does not name go in [train.rl].extra / [train.gnn].extra.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from . import instances
from .config import POOLED_DIR, REPO, Config

sys.path.insert(0, str(REPO / "lib"))
from deep_nn.strategies import discover_tables, select_tables  # noqa: E402

TRAINERS = {"rl": REPO / "lib" / "rl_handler" / "offline_main.py",
            "gnn": REPO / "lib" / "gnn_handler" / "__main__.py"}
# Every EPDDL problem is its own configuration; the RL trainer refuses such a split by default.
FIXED_FLAGS = {"rl": ["--allow-cross-config"], "gnn": []}
AGGREGATION = "dense"   # [train].aggregation default: ONNX sum aggregations as one-hot matmuls (deep_nn.dense)
GNN_FRINGE_SIZES = (1, 4, 8, 16, 32)   # the GNN scores states, not fringes: every width is exported
EXPORTS = {   # per F: (file the trainer writes, installed name); rl under run_fringe<F>/, gnn under run/
    "rl": [("frontier_policy_{F}_best_by_expansions.onnx", "rl_F{F}.onnx")],
    "gnn": [("distance_estimator_{F}.onnx", "gnn_F{F}.onnx")],
}
GNN_STATE = [("distance_estimator_state.onnx", "gnn_state.onnx"),
             ("distance_estimator_state_C.txt", "gnn_state_C.txt")]


def run(cfg: Config) -> None:
    insts = instances.load(cfg)
    domains = sorted({i.domain for i in insts})
    # [train].pooled: one model set from every domain's train trees, installed under models/pooled/
    groups = [(POOLED_DIR, domains)] if cfg.pooled else [(d, [d]) for d in domains]
    failed: list[str] = []            # a failed run must not block the other domains / F
    for label, members in groups:
        # the trainers key trees by problem name, so a pooled trial needs names unique across domains
        names = [i.problem for i in insts if i.domain in members]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise SystemExit(f"[train] {label}: problem names repeat across domains {dupes}; rename the problem files")
        train_csvs, test_csvs = [], []
        for domain in members:
            tr, te = _tables(cfg, insts, domain)
            train_csvs += tr
            test_csvs += te
        if not train_csvs:
            print(f"[train] {label}: no train trees under {cfg.data_dir}, skipped")
            continue
        for kind in cfg.train["models"]:
            failed += _train(cfg, label, kind, train_csvs, test_csvs)
    if failed:
        raise SystemExit(f"[train] {len(failed)} run(s) failed, the rest were trained: {failed}")


def _tables(cfg, insts, domain):
    tables = discover_tables(cfg.data_dir / domain)
    strategies = None if cfg.train["strategies"] == "all" else cfg.train["strategies"]

    def pick(split):
        mine = {i.problem for i in insts if i.domain == domain and i.split == split}
        sub = {p: t for p, t in tables.items() if p in mine}
        return select_tables(sub, strategies, verbose=False) if sub else []
    return pick("train"), pick("test")


def _train(cfg, label, kind, train_csvs, test_csvs) -> list[str]:
    """The RL trainer batches batch_size x F fringe states per step, so it runs once per F
    with the batch capped by [train].max_batch_states (and resumes per F). The GNN is
    per-state: one run, every F exported. Returns the tags of the runs that failed."""
    t = cfg.train
    groups = [[F] for F in t["fringe_sizes"]] if kind == "rl" else [list(GNN_FRINGE_SIZES)]
    failed = []
    for Fs in groups:
        try:
            _train_group(cfg, label, kind, Fs, train_csvs, test_csvs)
        except subprocess.CalledProcessError as e:
            tag = f"{label}/{kind}" + (f"@F{Fs[0]}" if len(Fs) == 1 else "")
            print(f"[train] FAIL {tag}: trainer exited {e.returncode}")
            failed.append(tag)
    return failed


def _batch_size(cfg, kind, F) -> int:
    t = cfg.train
    if kind == "rl" and "max_batch_states" in t:
        return max(1, min(int(t["batch_size"]), int(t["max_batch_states"]) // F))
    return int(t["batch_size"])


def _train_group(cfg, label, kind, Fs, train_csvs, test_csvs) -> None:
    out = cfg.models_dir / label             # the domain, or POOLED_DIR
    tag = f"{label}/{kind}" + (f"@F{Fs[0]}" if len(Fs) == 1 else "")
    installed = [(out / dst.format(F=F)) for F in Fs for _, dst in EXPORTS[kind]]
    if kind == "gnn":
        installed += [out / dst for _, dst in GNN_STATE]
    if all(p.exists() for p in installed):
        print(f"[train] {tag}: models present, skipped")
        return
    t = cfg.train
    batch = _batch_size(cfg, kind, Fs[0])
    cmd = [sys.executable, str(TRAINERS[kind]), "--train-csv", *map(str, train_csvs),
           "--dir-save-model", str(out / kind / "run"), "--fringe-sizes", *map(str, Fs),
           "--epochs", str(t["epochs"]), "--batch-size", str(batch), "--seed", str(t["seed"]),
           "--dataset-type", cfg.data["dataset_type"], "--aggregation", str(t.get("aggregation", AGGREGATION)),
           *FIXED_FLAGS[kind], *map(str, t.get(kind, {}).get("extra", []))]
    if kind == "rl" and "ckpt_every" in t:
        cmd += ["--ckpt-every", str(int(t["ckpt_every"]))]
    if test_csvs:
        cmd += ["--test-csv", *map(str, test_csvs)]
    print(f"[train] {tag}: {len(train_csvs)} train tables, {len(test_csvs)} test tables, batch {batch}")
    print(" ".join(cmd))
    if cfg.dry_run:
        return
    subprocess.run(cmd, cwd=REPO, check=True)
    for F in Fs:
        src_dir = out / kind / (f"run_fringe{F}" if kind == "rl" else "run")
        for src, dst in EXPORTS[kind]:
            shutil.copy2(src_dir / src.format(F=F), out / dst.format(F=F))
    if kind == "gnn":
        for src, dst in GNN_STATE:
            shutil.copy2(out / kind / "run" / src, out / dst)
    print(f"[train] {tag}: installed {[p.name for p in installed]}")


def reexport(cfg: Config) -> None:
    """`trial.py export`: re-export every installed model from its run dir with the
    trial's [train].aggregation (same weights, another ONNX form) and reinstall it.
    RL: offline_main --export-from run_fringe<F>; GNN: __main__ --export-only."""
    insts = instances.load(cfg)
    domains = sorted({i.domain for i in insts})
    labels = [POOLED_DIR] if cfg.pooled else domains
    t = cfg.train
    agg = str(t.get("aggregation", AGGREGATION))
    failed: list[str] = []

    def run(tag, cmd, install):
        print(f"[export] {tag} -> {agg}")
        if cfg.dry_run:
            print(" ".join(cmd))
            return
        try:
            subprocess.run(cmd, cwd=REPO, check=True)
            install()
        except subprocess.CalledProcessError as e:
            print(f"[export] FAIL {tag}: exited {e.returncode}")
            failed.append(tag)

    for label in labels:
        out = cfg.models_dir / label
        if "rl" in t["models"]:
            for F in t["fringe_sizes"]:
                run_dir = out / "rl" / f"run_fringe{F}"
                if not (out / f"rl_F{F}.onnx").exists() or not run_dir.is_dir():
                    continue
                cmd = [sys.executable, str(TRAINERS["rl"]), "--export-from", str(run_dir), "--aggregation", agg,
                       "--dataset-type", cfg.data["dataset_type"], *map(str, t.get("rl", {}).get("extra", []))]
                run(f"{label}/rl@F{F}", cmd,
                    lambda F=F, run_dir=run_dir: shutil.copy2(run_dir / EXPORTS["rl"][0][0].format(F=F), out / f"rl_F{F}.onnx"))
        if "gnn" in t["models"] and (out / "gnn" / "run" / "distance_estimator.pt").exists():
            members = domains if cfg.pooled else [label]
            train_csvs = [c for d in members for c in _tables(cfg, insts, d)[0]]
            # CPU: the export needs no GPU, and a training job may be holding it
            cmd = [sys.executable, str(TRAINERS["gnn"]), "--train-csv", *map(str, train_csvs),
                   "--dir-save-model", str(out / "gnn" / "run"), "--fringe-sizes", *map(str, GNN_FRINGE_SIZES),
                   "--epochs", str(t["epochs"]), "--batch-size", str(t["batch_size"]), "--seed", str(t["seed"]),
                   "--dataset-type", cfg.data["dataset_type"], "--aggregation", agg, "--export-only", "--device", "cpu",
                   *map(str, t.get("gnn", {}).get("extra", []))]

            def install_gnn(out=out):
                for F in GNN_FRINGE_SIZES:
                    shutil.copy2(out / "gnn" / "run" / EXPORTS["gnn"][0][0].format(F=F), out / f"gnn_F{F}.onnx")
                for src, dst in GNN_STATE:
                    shutil.copy2(out / "gnn" / "run" / src, out / dst)
            run(f"{label}/gnn", cmd, install_gnn)
    if failed:
        raise SystemExit(f"[export] {len(failed)} run(s) failed, the rest were re-exported: {failed}")
