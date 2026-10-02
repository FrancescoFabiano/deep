"""Stage 1: one generation tree per (problem, strategy) -> data/<domain>/<STRAT>/<problem>/.

The planner writes to out/NN/Training/<name>/ under its cwd, so every job runs in a
private work dir, and the result is moved into place with the CSV paths rewritten
to be repo-relative (what lib/deep_nn resolves them against). data/.dataspec pins the
generation parameters: a later run with different ones must start from an empty data/.
"""
from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from . import deep, instances
from .config import REPO, Config

SUCCESS = (0, 2, 3)              # ExitHandler: found goal / not planning mode / ... with warning
NO_GOAL = "No goals found"       # rc 4 (DatasetNoGoalFound) since upstream 31217ba; the text covers older binaries
SEEDLESS = ("BFS", "HFS")        # deterministic: retrying with another seed reproduces the tree
FINGERPRINT = ("strategies", "hfs_heuristic", "depth", "depth_overrides", "seed", "discard_factor",
               "max_generation", "max_creation", "dataset_type")


def run(cfg: Config) -> None:
    _check_dataspec(cfg)
    insts = [i for i in instances.load(cfg) if i.split == "train" or cfg.data.get("generate_test", False)]
    jobs = [(i, s) for i in insts for s in cfg.strategies if not _done(cfg, i, s)]
    print(f"[data] {len(jobs)} trees to generate, {len(insts) * len(cfg.strategies) - len(jobs)} already present")
    with ThreadPoolExecutor(cfg.workers) as ex:
        list(ex.map(lambda job: _generate(cfg, *job), jobs))


def target(cfg: Config, inst: instances.Instance, strategy: str):
    return cfg.data_dir / inst.domain / strategy / inst.problem


def _done(cfg, inst, strategy) -> bool:
    return any(target(cfg, inst, strategy).glob("*_depth_*.csv"))


def _argv(cfg, inst, strategy, seed) -> list[str]:
    d = cfg.data
    argv = deep.base_argv(cfg, inst) + [
        "--dataset", "--dataset_generation", strategy, "--dataset_depth", str(cfg.depth(inst.domain)),
        "--dataset_seed", str(seed), "--dataset_max_creation", str(d["max_creation"]),
        "--dataset_max_generation", str(d["max_generation"]), "--dataset_type", d["dataset_type"]]
    if strategy == "HFS":
        argv += ["--heuristics", d["hfs_heuristic"]]
    if strategy == "S_DFS":
        argv += ["--dataset_discard_factor", str(d["discard_factor"])]
    return argv


def _generate(cfg, inst, strategy) -> None:
    tgt = target(cfg, inst, strategy)
    work = tgt.parent / f".work_{inst.problem}"
    attempts = 1 if strategy in SEEDLESS else int(cfg.data.get("max_retries", 3))
    for k in range(attempts):
        argv = _argv(cfg, inst, strategy, cfg.data["seed"] + k)
        if cfg.dry_run:
            print(" ".join(argv))
            return
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        res = deep.run(argv, timeout_s=cfg.data["timeout_s"], mem_gb=cfg.mem_gb, cwd=work)
        m = re.search(r"Dataset stored in (.+?) folder\.", res.out)
        if res.limit is None and res.rc in SUCCESS and m and NO_GOAL not in res.out:
            _install(work / m.group(1), tgt, m.group(1))
            (tgt / "generation.log").write_text(" ".join(argv) + "\n\n" + res.out)
            shutil.rmtree(work, ignore_errors=True)
            print(f"[data] ok   {inst.domain}/{strategy}/{inst.problem} ({res.wall_s:.0f}s)")
            return
        (tgt.parent / f"{inst.problem}.failed.log").write_text(" ".join(argv) + "\n\n" + res.out)
        why = res.limit or ("no goal" if NO_GOAL in res.out else f"rc={res.rc}")
        print(f"[data] FAIL {inst.domain}/{strategy}/{inst.problem} {why} ({res.wall_s:.0f}s)")
    shutil.rmtree(work, ignore_errors=True)


def _install(src, tgt, printed: str) -> None:
    """Move the generator's folder into place; the CSV paths become repo-relative."""
    shutil.rmtree(tgt, ignore_errors=True)
    shutil.move(str(src), str(tgt))
    rel = tgt.resolve().relative_to(REPO).as_posix()
    for csv in tgt.glob("*.csv"):
        csv.write_text(csv.read_text().replace(printed.rstrip("/"), rel))


def _check_dataspec(cfg) -> None:
    spec = "\n".join(f"{k}={_flat(cfg.data.get(k))}" for k in FINGERPRINT) + "\n"
    path = cfg.data_dir / ".dataspec"
    if path.exists() and path.read_text() != spec:
        raise SystemExit(f"{path} was generated with different [data] parameters:\n{path.read_text()}"
                         f"now:\n{spec}Delete {cfg.data_dir} (or restore the parameters) before generating.")
    if not cfg.dry_run:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(spec)


def _flat(v) -> str:
    if isinstance(v, dict):
        return ",".join(f"{k}:{x}" for k, x in sorted(v.items()))
    return ",".join(map(str, v)) if isinstance(v, list) else str(v)
