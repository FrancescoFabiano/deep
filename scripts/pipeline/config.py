"""trial.toml -> a validated Config with resolved paths."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STRATEGIES = ("BFS", "DFS", "S_DFS", "HFS")
MODELS = ("rl", "gnn")
GNN_SEARCHES = ("Astar", "HFS")
POOLED_DIR = "pooled"          # models/<POOLED_DIR>/ holds the one model set of a [train].pooled trial
# [train].aggregation (the ONNX form of the sum aggregations) -> the folder its exports are installed in
AGGREGATION_DIRS = {"dense": "dense", "scatter": "scattered"}
AGGREGATION = "dense"          # default: one-hot matmuls (deep_nn.dense), -40% CPU latency on small graphs
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
    def pooled(self) -> bool: return bool(self.train.get("pooled", False))
    @property
    def aggregation(self) -> str: return str(self.train.get("aggregation", AGGREGATION))
    @property
    def exports(self) -> list[str]:
        """The ONNX forms stage 2 installs: every form unless [train].exports narrows it."""
        return list(self.train.get("exports", AGGREGATION_DIRS))

    def act_lib_for(self, domain: str) -> Path:
        """instances/<domain>/act_lib.epddl when the domain ships its own, else the shared one."""
        own = self.instances_dir / domain / "act_lib.epddl"
        return own if own.is_file() else self.act_lib

    def model_dir(self, domain: str) -> Path:
        """Where stage 2 installs and stage 3 finds a domain's models (one shared dir when pooled),
        in the ONNX form of [train].aggregation: models/<domain>/{dense,scattered}/."""
        return self.models_dir / (POOLED_DIR if self.pooled else domain) / AGGREGATION_DIRS[self.aggregation]
    @property
    def results_file(self) -> Path: return self.trial_dir / "results" / "results.csv"
    @property
    def report_dir(self) -> Path: return self.trial_dir / "report"

    def depth(self, domain: str) -> int:
        return int(self.data.get("depth_overrides", {}).get(domain, self.data["depth"]))


def load(trial_dir: Path, domains=None, strategies=None, models=None, workers=None, dry_run=False,
         aggregation=None) -> Config:
    raw = tomllib.loads((trial_dir / "trial.toml").read_text())
    for section, keys in REQUIRED.items():
        missing = [k for k in keys if k not in raw.get(section, {})]
        if missing:
            raise SystemExit(f"trial.toml [{section}] is missing {missing}")
    if models:
        raw["train"]["models"] = list(models)
    if workers:
        raw["trial"]["workers"] = int(workers)
    if aggregation:
        raw["train"]["aggregation"] = aggregation
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
    domains = [p.name for p in cfg.instances_dir.iterdir() if (p / "domain.epddl").is_file()] if cfg.instances_dir.is_dir() else []
    check(domains, f"no <domain>/domain.epddl under {cfg.instances_dir}")
    no_lib = [d for d in domains if not cfg.act_lib_for(d).is_file()]
    check(not no_lib, f"no action library for {no_lib}: add instances/act_lib.epddl or instances/<domain>/act_lib.epddl")
    check(0 < cfg.split["train_pct"] < 100, "[split] train_pct must be in (0, 100)")
    check(set(d["strategies"]) <= set(STRATEGIES), f"[data] strategies must be among {STRATEGIES}")
    check(set(cfg.strategies) <= set(d["strategies"]), f"--strategies must be among [data] strategies {d['strategies']}")
    check("HFS" not in d["strategies"] or "hfs_heuristic" in d, "[data] HFS needs hfs_heuristic")
    check("S_DFS" not in d["strategies"] or "discard_factor" in d, "[data] S_DFS needs discard_factor")
    check(set(t["models"]) <= set(MODELS), f"[train] models must be among {MODELS}")
    check(cfg.aggregation in AGGREGATION_DIRS, f"[train] aggregation must be among {sorted(AGGREGATION_DIRS)}")
    check(set(cfg.exports) <= set(AGGREGATION_DIRS), f"[train] exports must be among {sorted(AGGREGATION_DIRS)}")
    check(t["strategies"] == "all" or set(t["strategies"]) <= set(d["strategies"]),
          "[train] strategies must be 'all' or a subset of [data] strategies")
    check(i["rl_exploration"] + i["rl_exploitation"] < 100, "[inference] rl_exploration + rl_exploitation must be < 100")
    check(set(i["gnn_searches"]) <= set(GNN_SEARCHES), f"[inference] gnn_searches must be among {GNN_SEARCHES}")
