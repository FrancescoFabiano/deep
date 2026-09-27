"""Domains and problems of a trial, and the frozen train/test split (split.csv).

instances/<domain>/domain.epddl + problems/*.epddl. The split is computed once per
domain -- problems sorted by name (or shuffled with [split].shuffle_seed), the first
ceil(train_pct%) train, the rest test -- and written to split.csv, which every
stage then reads. Edit or delete that file to change the split.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class Instance:
    domain: str
    problem: str
    split: str
    domain_file: Path
    problem_file: Path

    @property
    def key(self) -> tuple[str, str]:
        return self.domain, self.problem


def load(cfg: Config) -> list[Instance]:
    if not cfg.split_file.exists():
        _write_split(cfg)
    with cfg.split_file.open() as f:
        rows = list(csv.DictReader(f))
    if cfg.domains:
        unknown = set(cfg.domains) - {r["domain"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown domains {sorted(unknown)}; split.csv has {sorted({r['domain'] for r in rows})}")
        rows = [r for r in rows if r["domain"] in cfg.domains]
    return [Instance(r["domain"], r["problem"], r["split"],
                     cfg.instances_dir / r["domain"] / "domain.epddl",
                     cfg.instances_dir / r["domain"] / "problems" / f"{r['problem']}.epddl") for r in rows]


def _write_split(cfg: Config) -> None:
    rows = []
    for dom in sorted(p for p in cfg.instances_dir.iterdir() if (p / "domain.epddl").is_file()):
        problems = sorted(p.stem for p in (dom / "problems").glob("*.epddl"))
        if "shuffle_seed" in cfg.split:
            random.Random(cfg.split["shuffle_seed"]).shuffle(problems)
        n_train = math.ceil(len(problems) * cfg.split["train_pct"] / 100)
        rows += [(dom.name, prob, "train" if k < n_train else "test") for k, prob in enumerate(problems)]
    with cfg.split_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "problem", "split"])
        w.writerows(rows)
    print(f"[split] wrote {cfg.split_file} ({len(rows)} problems)")
