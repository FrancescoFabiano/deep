"""Stage 3: BFS and every trained model on every instance -> results/results.csv.

One row per (problem, method, F); rows already present are not rerun. Methods come
from what models/<domain>/ holds: RL (rl_F*.onnx), GNN_RL (the GNN ranking the RL
beam), GNN_<search> for each [inference].gnn_searches (the per-state export).
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from . import deep, instances
from .config import REPO, Config

COLUMNS = ["domain", "split", "problem", "method", "F", "status", "plan_length", "nodes_expanded",
           "init_ms", "search_ms", "total_ms", "wall_s"]
STATUS = {0: "SOLVED", 1: "NO_PLAN"}


@dataclass(frozen=True)
class Method:
    name: str
    F: int
    flags: tuple


def run(cfg: Config) -> None:
    insts = instances.load(cfg)
    done = _done(cfg)
    jobs = [(i, m) for i in insts for m in methods(cfg, i.domain) if (i.domain, i.problem, m.name, m.F) not in done]
    print(f"[infer] {len(jobs)} runs to do, {len(done)} already in {cfg.results_file}")
    if cfg.dry_run:
        for inst, m in jobs:
            print(" ".join(_argv(cfg, inst, m)))
        return
    cfg.results_file.parent.mkdir(parents=True, exist_ok=True)
    new = not cfg.results_file.exists()
    with cfg.results_file.open("a", newline="") as f, ThreadPoolExecutor(cfg.workers) as ex:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for fut in as_completed([ex.submit(_one, cfg, inst, m) for inst, m in jobs]):
            w.writerow(fut.result())
            f.flush()


def methods(cfg: Config, domain: str) -> list[Method]:
    inf = cfg.inference
    beam = ("--RL_exploration", str(inf["rl_exploration"]), "--RL_exploitation", str(inf["rl_exploitation"]))
    out = [Method("BFS", 0, ("-s", "BFS"))]
    for F, onnx in _models(cfg, domain, "rl"):
        out.append(Method("RL", F, ("-s", "RL", "-u", "RL_H", "--RL_model", str(onnx), "--RL_fringe_size", str(F), *beam)))
    for F, onnx in _models(cfg, domain, "gnn"):
        out.append(Method("GNN_RL", F, ("-s", "RL", "-u", "RL_H", "--RL_model", str(onnx), "--RL_fringe_size", str(F), *beam)))
        state = onnx.with_name(f"gnn_F{F}_state.onnx")
        for search in inf["gnn_searches"]:
            out.append(Method(f"GNN_{search}", F, ("-s", search, "-u", "GNN", "--GNN_model", str(state),
                                                   "--GNN_constant_file", str(state.with_name(state.stem + "_C.txt")))))
    return out


def _models(cfg, domain, kind):
    found = [(re.fullmatch(rf"{kind}_F(\d+)\.onnx", p.name), p) for p in (cfg.models_dir / domain).glob(f"{kind}_F*.onnx")]
    return sorted((int(m.group(1)), p) for m, p in found if m)


def _argv(cfg, inst, m: Method) -> list[str]:
    return deep.base_argv(cfg, inst) + ["-r", *m.flags]


def _one(cfg, inst, m: Method) -> dict:
    res = deep.run(_argv(cfg, inst, m), timeout_s=cfg.inference["timeout_s"], mem_gb=cfg.mem_gb, cwd=REPO)
    status = res.limit or STATUS.get(res.rc, f"ERROR_{res.rc}")
    row = dict(domain=inst.domain, split=inst.split, problem=inst.problem, method=m.name, F=m.F, status=status,
               plan_length=res.field("Plan length"), nodes_expanded=res.field("Nodes expanded"),
               init_ms=res.field("Initial state construction"), search_ms=res.field("Search time"),
               total_ms=res.field("Total execution time"), wall_s=round(res.wall_s, 2))
    if status != "SOLVED":
        log = cfg.results_file.parent / "logs" / f"{inst.domain}__{inst.problem}__{m.name}_F{m.F}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(" ".join(_argv(cfg, inst, m)) + "\n\n" + res.out)
    print(f"[infer] {status:8} {inst.domain}/{inst.problem} {m.name}@{m.F} nodes={row['nodes_expanded']} ({res.wall_s:.1f}s)")
    return row


def _done(cfg) -> set:
    if not cfg.results_file.exists():
        return set()
    with cfg.results_file.open() as f:
        return {(r["domain"], r["problem"], r["method"], int(r["F"])) for r in csv.DictReader(f)}
