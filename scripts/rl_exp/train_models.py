"""Per-experiment training driver for the offline RL fringe-ranking pipeline.

Twin of scripts/gnn_exp/train_models.py, adapted for lib/rl_handler's offline
Double-DQN trainer (lib/rl_handler/offline_main.py).  It enumerates domains
under an experiment root, builds each domain's train/val split from its
generation-table CSVs, runs offline_main.py per (domain, seed), and — for a
single-seed run — installs the exported ONNX where the eval consumer
(scripts/rl_exp/bulk_coverage_run.py) looks for it:

    <exp_dir>/_models/<domain>/frontier_policy_<F>.onnx

Domain/data convention (same as gnn_exp):
    <exp_dir>/_models/<domain>/training_data/<instance>/<instance>_depth_*.csv

Train/val split: instances are sorted; the last one is held out as validation,
the rest are training (single-instance domains use it as both, with a warning).

KWARGS PASS-THROUGH: this script owns only orchestration flags (exp dir,
--domains, --seeds, output root).  Every offline_main.py flag (--frames,
--n-checkpoints, --gamma, --epsilon-schedule, --fringe-size, --device, ...) is
forwarded verbatim via the unknown-args remainder; offline_main.py validates
them, and a non-zero exit there fails this script loudly.  The fully resolved
command is echoed before each launch so runs reproduce from the log.

Examples
--------
# All domains found under the experiment root, defaults, single seed (42):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged

# One domain, short run, forwarding offline_main flags after the orchestration
# args (note --frames/--fringe-size are passed straight through):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \\
    --domains CC -- --frames 100000 --n-checkpoints 20 --fringe-size 32

# Multi-seed production run (each seed in its own subdir; no auto-install):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \\
    --domains CC --seeds 0 1 2 -- --frames 100000 --gamma 0.99
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_MAIN = REPO_ROOT / "lib" / "rl_handler" / "offline_main.py"


def find_domains(models_root: Path) -> list[str]:
    """Domains = subdirs of <exp_dir>/_models that contain a training_data dir."""
    if not models_root.is_dir():
        return []
    domains = []
    for child in sorted(p for p in models_root.iterdir() if p.is_dir()):
        if (child / "training_data").is_dir():
            domains.append(child.name)
    return domains


def domain_instance_csvs(models_root: Path, domain: str) -> list[Path]:
    """All per-instance generation tables for a domain, sorted by instance name."""
    training_data = models_root / domain / "training_data"
    csvs: list[Path] = []
    for inst_dir in sorted(p for p in training_data.iterdir() if p.is_dir()):
        matches = sorted(inst_dir.glob(f"{inst_dir.name}_depth_*.csv"))
        if not matches:
            matches = sorted(inst_dir.glob("*_depth_*.csv"))
        if matches:
            csvs.append(matches[0])
    return csvs


def split_train_val(csvs: list[Path]) -> tuple[list[Path], list[Path]]:
    """Hold out the last instance as validation; the rest train."""
    if len(csvs) <= 1:
        return csvs, csvs  # caller warns
    return csvs[:-1], csvs[-1:]


def fringe_size_from_forwarded(forwarded: list[str], default: int = 32) -> int:
    """Read --fringe-size from the forwarded remainder to name the install file.

    Non-destructive: the flag stays in `forwarded` and still reaches
    offline_main.py; we only peek at it so the installed ONNX matches the
    `frontier_policy_<F>.onnx` name the eval consumer expects.
    """
    for i, tok in enumerate(forwarded):
        if tok == "--fringe-size" and i + 1 < len(forwarded):
            return int(forwarded[i + 1])
        if tok.startswith("--fringe-size="):
            return int(tok.split("=", 1)[1])
    return default


def user_supplied_csvs(forwarded: list[str]) -> bool:
    return any(
        tok == "--train-csv" or tok == "--val-csv"
        or tok.startswith("--train-csv=") or tok.startswith("--val-csv=")
        for tok in forwarded
    )


def run_one(cmd: list[str], prefix: str) -> int:
    """Run offline_main.py, streaming one log line every ~15s; return exit code.

    The 15s throttle keeps long training runs readable, but would hide a
    fast-failing error (e.g. a rejected kwarg).  So the most recent lines are
    kept in a ring buffer and dumped verbatim on a non-zero exit — kwargs
    errors are never silent.
    """
    print(" ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_print = 0.0
    recent: deque[str] = deque(maxlen=40)
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        recent.append(line.rstrip())
        now = time.time()
        if now - last_print >= 15:
            print(f"{prefix} {recent[-1]}", flush=True)
            last_print = now
    process.stdout.close()
    rc = process.wait()
    if rc != 0:
        print(f"{prefix} ---- offline_main.py output (last {len(recent)} lines) ----")
        for line in recent:
            print(f"{prefix} {line}", flush=True)
        print(f"{prefix} ---- end output ----", flush=True)
    return rc


def train_domain(
    exp_dir: Path,
    models_root: Path,
    domain: str,
    seeds: list[int],
    forwarded: list[str],
) -> None:
    csvs = domain_instance_csvs(models_root, domain)
    if not csvs:
        print(f"[WARNING] No instance CSVs for domain '{domain}', skipping.")
        return

    auto_split = not user_supplied_csvs(forwarded)
    train_csvs, val_csvs = split_train_val(csvs)
    if auto_split and len(csvs) == 1:
        print(
            f"[WARNING] domain '{domain}' has a single instance; "
            "using it as both train and val."
        )

    fringe = fringe_size_from_forwarded(forwarded)
    domain_model_dir = models_root / domain

    for seed in seeds:
        seed_dir = domain_model_dir / f"seed{seed}"
        prefix = f"[{domain}/seed{seed}]".ljust(22)

        cmd = [
            sys.executable,
            str(OFFLINE_MAIN),
            "--seed",
            str(seed),
            "--dir-save-model",
            str(seed_dir),
        ]
        if auto_split:
            cmd += ["--train-csv", *(str(p.resolve()) for p in train_csvs)]
            cmd += ["--val-csv", *(str(p.resolve()) for p in val_csvs)]
        cmd += forwarded

        rc = run_one(cmd, prefix)
        if rc != 0:
            # Fail loudly: propagate offline_main's non-zero exit immediately
            # (covers rejected kwargs and training errors alike).
            print(f"{prefix} [ERROR] offline_main.py exited with code {rc}")
            sys.exit(rc)
        print(f"{prefix} [SUCCESS]")

    # Single-seed run: install the objective-best export where the eval
    # consumer expects it, mirroring gnn_exp's zero-manual-step flow.
    if len(seeds) == 1:
        seed_dir = domain_model_dir / f"seed{seeds[0]}"
        exported = seed_dir / f"frontier_policy_{fringe}_best_by_expansions.onnx"
        if not exported.exists():
            print(
                f"[WARNING] expected export not found: {exported} "
                "(did you pass --no-export-onnx?); skipping install."
            )
            return
        installed = domain_model_dir / f"frontier_policy_{fringe}.onnx"
        shutil.copy2(exported, installed)
        print(f"[{domain}] installed {installed}")
    else:
        print(
            f"[{domain}] multi-seed run ({len(seeds)} seeds): no model auto-"
            "installed. Use offline_analysis.py to pick a seed, then copy its "
            f"frontier_policy_<F>_best_by_expansions.onnx to "
            f"{domain_model_dir / 'frontier_policy_<F>.onnx'}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the offline RL fringe-ranking model per domain under an "
            "experiment root. Orchestration flags are below; any other flags "
            "(after them, optionally separated by --) are forwarded verbatim "
            "to lib/rl_handler/offline_main.py."
        )
    )
    parser.add_argument(
        "exp_dir",
        help="Experiment root (e.g. exp/rl_exp/batch0_merged); models read "
        "from / written to <exp_dir>/_models/<domain>/",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to train (default: all under <exp_dir>/_models with a "
        "training_data dir).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds; one subdir per seed. A single seed also installs the "
        "exported model under _models/<domain>/. Default: 42.",
    )
    args, forwarded = parser.parse_known_args()
    # Allow an explicit `--` separator before the forwarded block.
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    exp_dir = Path(args.exp_dir)
    models_root = exp_dir / "_models"
    if not models_root.is_dir():
        print(f"[ERROR] No _models dir under: {exp_dir}")
        sys.exit(1)

    domains = args.domains or find_domains(models_root)
    if not domains:
        print(
            f"[ERROR] No domains (with a training_data dir) found under "
            f"{models_root}."
        )
        sys.exit(1)

    print(f"[INFO] exp_dir={exp_dir} domains={domains} seeds={args.seeds}")
    if forwarded:
        print(f"[INFO] forwarding to offline_main.py: {' '.join(forwarded)}")

    for domain in domains:
        train_domain(exp_dir, models_root, domain, args.seeds, forwarded)


if __name__ == "__main__":
    main()
