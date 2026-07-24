"""Per-experiment training driver for the offline RL fringe-ranking pipeline.

Twin of scripts/gnn_exp/train_models.py, adapted for lib/rl_handler's offline
Double-DQN trainer (lib/rl_handler/offline_main.py).  It enumerates domains
under an experiment root, builds each domain's train/val split from its
generation-table CSVs, runs offline_main.py per (domain, seed), and — for a
single-seed run — installs the exported ONNX where the eval consumer
(scripts/rl_exp/bulk_coverage_run.py) looks for it:

    <exp_dir>/_models/<domain>/frontier_policy_<F>.onnx

Domain/data convention:
    <exp_dir>/_models/<domain>/training_data/<instance>/<instance>_depth_*.csv
    <exp_dir>/_models/<domain>/test_data/<instance>/<instance>_depth_*.csv

Split: NO auto-split. TRAIN = all of training_data (kept whole); held-out TEST =
all of test_data (diagnostic only — never selects; selection is TRAIN-based in
offline_main's regime path). The script never carves a val set out of train and
never passes --val-csv. Sweep axes (--gamma/--lambda-ord/--target-centering/
--fringe-sizes) and --use-regimes/--regimes are forwarded verbatim to
offline_main; run one invocation per arm (single-cell contract).

KWARGS PASS-THROUGH: this script owns orchestration flags (exp dir, --domains,
--seeds, --fringe-sizes, output root).  Every other offline_main.py flag
(--frames, --n-checkpoints, --gamma, --epsilon-schedule, --device, ...) is
forwarded verbatim via the unknown-args remainder; offline_main.py validates
them, and a non-zero exit there fails this script loudly.  --fringe-sizes is a
known flag here, so it is passed through deliberately (one trained+installed
model per fringe size).  The fully resolved command is echoed before each
launch so runs reproduce from the log.

Examples
--------
# All domains found under the experiment root, defaults, single seed (42):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged

# One domain, short run, both fringe sizes, forwarding offline_main flags after
# the orchestration args (--frames is passed straight through):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \\
    --domains CC --fringe-sizes 32 64 -- --frames 100000 --n-checkpoints 20

# Multi-seed production run (each seed in its own subdir; no auto-install):
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \\
    --domains CC --seeds 0 1 2 -- --frames 100000 --gamma 0.99
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import os
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_MAIN = REPO_ROOT / "lib" / "rl_handler" / "offline_main.py"


# fe3d2a9 ("dead paths") deleted these four while leaving every call site, so this
# module raised NameError before training a single frame and final_launcher.sh's
# step 2 had never once executed. Restored verbatim from fe3d2a9^ -- no redesign.


def find_domains(models_root: Path) -> list[str]:
    """Domains = subdirs of <exp_dir>/_models that contain a training_data dir."""
    if not models_root.is_dir():
        return []
    domains = []
    for child in sorted(p for p in models_root.iterdir() if p.is_dir()):
        if (child / "training_data").is_dir():
            domains.append(child.name)
    return domains


def _instance_csvs(subdir: Path) -> list[Path]:
    """All per-instance generation tables under a data subdir, sorted by name."""
    csvs: list[Path] = []
    if not subdir.is_dir():
        return csvs
    for inst_dir in sorted(p for p in subdir.iterdir() if p.is_dir()):
        matches = sorted(inst_dir.glob(f"{inst_dir.name}_depth_*.csv"))
        if not matches:
            matches = sorted(inst_dir.glob("*_depth_*.csv"))
        if matches:
            csvs.append(matches[0])
    return csvs


def domain_train_csvs(models_root: Path, domain: str) -> list[Path]:
    """TRAIN instances = everything under <domain>/training_data, kept whole.
    NO auto-split: the script never carves a val set out of train."""
    return _instance_csvs(models_root / domain / "training_data")


def domain_test_csvs(models_root: Path, domain: str) -> list[Path]:
    """Held-out diagnostic TEST instances = everything under <domain>/test_data
    (never feeds selection; only the per-regime diagnostic plots / final numbers)."""
    return _instance_csvs(models_root / domain / "test_data")


def run_one(cmd: list[str], prefix: str, domain: str = "?") -> int:
    """Run offline_main.py with a PARENT-side tqdm progress bar; return exit code.

    The bar lives here, not in the child: the child is piped (never a tty), so a
    bar there cannot render through this line-buffered readline loop (B0/B1). We
    parse the child's stable per-checkpoint line
    `[run] ckpt F=.. step=.. frames=.. total=.. ndcg=.. td=..` to advance ONE bar
    over FRAMES, resetting per fringe (one offline_main child trains ALL
    --fringe-sizes in sequence). Every other child line -- the H2 row-share table,
    the H3 zero-scorable warning, goals-loaded, gate results, errors -- is routed
    through tqdm.write so it prints above the bar without corrupting it.

    TTY GUARD: when this parent's stderr is NOT a tty (backgrounded / tee'd to a
    log), a \r bar would be thousands of control chars, so we fall back to plain
    line output -- every child line is printed. The child stays bounded: its own
    bar disables itself when piped, so only newline prints stream here. On a
    non-zero exit the last lines are dumped so kwargs errors are never silent.
    """
    if cmd and "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print(" ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=env,
    )
    import re
    from tqdm import tqdm
    is_tty = sys.stderr.isatty()
    ckpt_re = re.compile(
        r"\[run\] ckpt F=(\d+) step=(\d+) frames=(\d+) total=(\d+) ndcg=(\S+) td=(\S+)")
    recent: deque[str] = deque(maxlen=40)
    bar = None
    cur_F = None
    assert process.stdout is not None
    for raw in iter(process.stdout.readline, ""):
        line = raw.rstrip()
        recent.append(line)
        m = ckpt_re.search(line)
        if m and is_tty:
            F, _step, frames, total, ndcg, td = m.groups()
            if bar is None or F != cur_F:
                if bar is not None:
                    bar.close()
                bar = tqdm(total=int(total), desc=f"{domain}-{F}fringe", unit="frame",
                           unit_scale=True, dynamic_ncols=True, leave=True)
                cur_F = F
            bar.n = int(frames)
            bar.set_postfix_str(f"ndcg={ndcg} td={td}", refresh=False)
            bar.refresh()
        elif is_tty and bar is not None:
            bar.write(f"{prefix} {line}")   # non-checkpoint line, above the bar
        else:
            print(f"{prefix} {line}", flush=True)   # startup lines, and the non-tty path
    if bar is not None:
        bar.close()
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
    fringe_sizes: list[int],
    forwarded: list[str],
    no_goal: bool = False,
    model: str = "dqn",
) -> None:
    # NO auto-split: TRAIN = all training_data (kept whole); held-out TEST = all
    # test_data (diagnostic only). Selection is TRAIN-based in offline_main's
    # regime path; the script never supplies --val-csv.
    train_csvs = domain_train_csvs(models_root, domain)
    test_csvs = domain_test_csvs(models_root, domain)
    if not train_csvs:
        print(f"[WARNING] No training_data CSVs for domain '{domain}', skipping.")
        return
    print(
        f"[split] domain '{domain}': "
        f"train={[p.parent.name for p in train_csvs]} "
        f"test(diagnostic)={[p.parent.name for p in test_csvs]}"
    )
    if not test_csvs:
        print(f"[INFO] domain '{domain}' has no test_data; running train-only "
              "(selection is train-based; no held-out diagnostic plots).")

    domain_model_dir = models_root / domain

    for seed in seeds:
        # Base dir; offline_main.py appends `_fringe{F}` per fringe size.
        seed_dir = domain_model_dir / f"seed{seed}"
        prefix = f"[{domain}/seed{seed}]".ljust(22)

        cmd = [
            sys.executable,
            str(OFFLINE_MAIN),
            "--seed",
            str(seed),
            "--dir-save-model",
            str(seed_dir),
            "--train-csv", *(str(p.resolve()) for p in train_csvs),
        ]
        if test_csvs:
            cmd += ["--test-csv", *(str(p.resolve()) for p in test_csvs)]
        if no_goal:
            cmd += ["--kind-of-data", "separated"]
        # Forward the offline RL model choice; offline_main reads --cql-alpha from
        # the forwarded remainder (it is not a train_models flag), so the study
        # can sweep alpha. --model dqn is the default and byte-identical to before.
        cmd += ["--model", model]
        cmd += forwarded
        # --fringe-sizes is a known flag here, so parse_known_args strips it
        # from the forwarded remainder; pass it through deliberately.
        cmd += ["--fringe-sizes", *(str(F) for F in fringe_sizes)]

        rc = run_one(cmd, prefix, domain=domain)
        if rc != 0:
            # Fail loudly: propagate offline_main's non-zero exit immediately
            # (covers rejected kwargs and training errors alike).
            print(f"{prefix} [ERROR] offline_main.py exited with code {rc}")
            sys.exit(rc)
        print(f"{prefix} [SUCCESS]")

    # Single-seed run: install the objective-best export for EACH fringe size
    # where the eval consumer expects it, mirroring gnn_exp's zero-manual-step
    # flow. Each fringe coexists under its own frontier_policy_<F>.onnx name.
    if len(seeds) == 1:
        for F in fringe_sizes:
            fringe_dir = domain_model_dir / f"seed{seeds[0]}_fringe{F}"
            exported = fringe_dir / f"frontier_policy_{F}_best_by_expansions.onnx"
            if not exported.exists():
                print(
                    f"[WARNING] expected export not found: {exported} "
                    "(did you pass --no-export-onnx?); skipping install."
                )
                continue
            installed = domain_model_dir / f"frontier_policy_{F}.onnx"
            shutil.copy2(exported, installed)
            print(f"[{domain}] installed {installed}")

            # Carry this fringe's per-run plots next to the deployed ONNX so the
            # single-seed view sits beside the model. The fringe-named PNGs keep
            # each fringe's plots distinct when copied up.
            copied = []
            for png in sorted(fringe_dir.glob("*.png")):
                dst = domain_model_dir / png.name
                shutil.copy2(png, dst)
                copied.append(dst.name)
            if copied:
                print(
                    f"[{domain}] copied plots to {domain_model_dir}: "
                    f"{', '.join(copied)}"
                )
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
    parser.add_argument(
        "--fringe-sizes",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Train one model per fringe size (forwarded to offline_main.py). "
        "Each F installs its own frontier_policy_<F>.onnx. Default: 32 64.",
    )
    parser.add_argument(
        "--model",
        choices=["dqn", "cql"],
        default="dqn",
        help="Offline RL loss. 'dqn' (default) = Double-DQN. 'cql' adds the "
        "conservative-Q penalty; sweep its weight by forwarding --cql-alpha "
        "(e.g. -- --cql-alpha 0.5) to offline_main. alpha=0 == dqn.",
    )
    parser.add_argument(
        "--no_goal",
        action="store_true",
        help="Pass '--kind-of-data separated' to offline_main.py (normal "
        "training only).",
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

    print(
        f"[INFO] exp_dir={exp_dir} model={args.model} domains={domains} "
        f"seeds={args.seeds} fringe_sizes={args.fringe_sizes}"
    )
    if forwarded:
        print(f"[INFO] forwarding to offline_main.py: {' '.join(forwarded)}")

    for domain in domains:
        train_domain(
            exp_dir, models_root, domain, args.seeds, args.fringe_sizes,
            forwarded, args.no_goal, args.model,
        )

    # BELT-AND-SUSPENDERS: regenerate figures for EVERY run in the batch, unconditionally,
    # after all training. run() already plots per-run at its end, but a crash before that
    # (e.g. an export failure) leaves complete telemetry unplotted -- this walks the
    # telemetry on disk and plots regardless, backfilling any missing metric from the
    # saved checkpoints (self-validated). CPU only; never fails the run.
    regen = REPO_ROOT / "scripts" / "rl_exp" / "regenerate_figures.py"
    print(f"[INFO] regenerating figures for all runs under {exp_dir}")
    subprocess.run([sys.executable, str(regen), str(exp_dir)], check=False)


if __name__ == "__main__":
    main()
