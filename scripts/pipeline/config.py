"""trial.toml -> a validated Config with resolved paths."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STRATEGIES = ("BFS", "DFS", "S_DFS", "HFS")
MODELS = ("rl", "gnn")
GNN_SEARCHES = ("Astar", "HFS")
REQUIRED = {
    "trial": ["deep_exe"],
    "split": ["train_pct"],
    "data": ["strategies", "depth", "seed", "max_generation", "max_creation", "dataset_type", "timeout_s"],
    "train": ["models", "strategies", "fringe_sizes", "epochs", "batch_size", "seed"],
    "inference": ["timeout_s", "rl_exploration", "rl_exploitation", "gnn_searches"],
}


@dataclass
class Config:
    trial_dir: Path
    trial: dict
    split: dict
    data: dict
    train: dict
    inference: dict
    domains: list[str] | None      # CLI restriction, None = all
    strategies: list[str]          # strategies to generate (CLI or [data].strategies)
    dry_run: bool

    @property
    def deep_exe(self) -> Path: return REPO / self.trial["deep_exe"]
    @property
    def workers(self) -> int: return int(self.trial.get("workers", 4))
    @property
    def mem_gb(self) -> float: return float(self.trial.get("mem_gb", 8))
    @property
    def instances_dir(self) -> Path: return self.trial_dir / "instances"
    @property
    def act_lib(self) -> Path: return self.instances_dir / "act_lib.epddl"
    @property
    def split_file(self) -> Path: return self.trial_dir / "split.csv"
    @property
    def data_dir(self) -> Path: return self.trial_dir / "data"
    @property
    def models_dir(self) -> Path: return self.trial_dir / "models"
    @property
    def results_file(self) -> Path: return self.trial_dir / "results" / "results.csv"
    @property
    def report_dir(self) -> Path: return self.trial_dir / "report"

    def depth(self, domain: str) -> int:
        return int(self.data.get("depth_overrides", {}).get(domain, self.data["depth"]))


def load(trial_dir: Path, domains=None, strategies=None, models=None, workers=None, dry_run=False) -> Config:
    raw = tomllib.loads((trial_dir / "trial.toml").read_text())
    for section, keys in REQUIRED.items():
        missing = [k for k in keys if k not in raw.get(section, {})]
        if missing:
            raise SystemExit(f"trial.toml [{section}] is missing {missing}")
    if models:
        raw["train"]["models"] = list(models)
    if workers:
        raw["trial"]["workers"] = int(workers)
    cfg = Config(trial_dir, raw["trial"], raw["split"], raw["data"], raw["train"], raw["inference"],
                 domains, list(strategies or raw["data"]["strategies"]), dry_run)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise SystemExit(f"trial.toml: {msg}")
    d, t, i = cfg.data, cfg.train, cfg.inference
    check(cfg.dry_run or cfg.deep_exe.is_file(), f"deep_exe not found: {cfg.deep_exe}")
    check(cfg.act_lib.is_file(), f"missing {cfg.act_lib}")
    check(0 < cfg.split["train_pct"] < 100, "[split] train_pct must be in (0, 100)")
    check(set(d["strategies"]) <= set(STRATEGIES), f"[data] strategies must be among {STRATEGIES}")
    check(set(cfg.strategies) <= set(d["strategies"]), f"--strategies must be among [data] strategies {d['strategies']}")
    check("HFS" not in d["strategies"] or "hfs_heuristic" in d, "[data] HFS needs hfs_heuristic")
    check("S_DFS" not in d["strategies"] or "discard_factor" in d, "[data] S_DFS needs discard_factor")
    check(set(t["models"]) <= set(MODELS), f"[train] models must be among {MODELS}")
    check(t["strategies"] == "all" or set(t["strategies"]) <= set(d["strategies"]),
          "[train] strategies must be 'all' or a subset of [data] strategies")
    check(i["rl_exploration"] + i["rl_exploitation"] < 100, "[inference] rl_exploration + rl_exploitation must be < 100")
    check(set(i["gnn_searches"]) <= set(GNN_SEARCHES), f"[inference] gnn_searches must be among {GNN_SEARCHES}")
