#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path):
    print(f"\n▶ Running:\n  {cmd}\n  (cwd: {cwd})")
    try:
        subprocess.run(cmd, cwd=str(cwd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✖ Command failed with exit code {e.returncode}\n", file=sys.stderr)
        sys.exit(e.returncode)


# ---- helpers ---------------------------------------------------------------


def _bulk_cmd(
    deep_exe, exp_dir, *, threads: int, binary_args: str, timeout: str
) -> str:
    """Build the shell command string for bulk_coverage_run.py."""
    return " ".join(
        [
            sys.executable,
            "scripts/gnn_exp/bulk_coverage_run.py",
            shlex.quote(str(deep_exe)),
            shlex.quote(str(exp_dir)),
            f"--threads {threads}",
            "--binary_args",
            shlex.quote(binary_args),
            f"--timeout {timeout}",
        ]
    )


def _base_args(
    dataset_type: str,
    *,
    with_search: str | None = None,
    with_upstream: str | None = None,
    extra: str = "",
) -> str:
    """
    Compose the `--binary_args` string.
    Common flags: `-c -b --dataset_type ...`.
    Optionally add `-s <search>` and `-u <upstream>`.
    """
    parts = ["-c", "-b", f"--dataset_type {dataset_type}"]
    if with_search:
        parts.extend(["-s", with_search])
    if with_upstream:
        parts.extend(["-u", with_upstream])
    if extra:
        parts.append(extra)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Run Steps 1–3 (create training data → train models → evaluate/aggregate) with hardcoded defaults."
    )
    parser.add_argument(
        "--bin",
        required=True,
        help="Path to the compiled deep executable (e.g., cmake-build-debug-nn/bin/deep)",
    )
    parser.add_argument(
        "--dataset_type",
        default="HASHED",
        choices=["MAPPED", "HASHED", "BITMASK"],
        help="State representation to use across Steps 1–3 (default: HASHED)",
    )
    parser.add_argument(
        "--exp_dir",
        default="exp/gnn_exp/batchTest",
        help="Experiment folder (default: exp/gnn_exp/batchTest)",
    )
    parser.add_argument(
        "--skip_gen", action="store_true", help="Skip Step 1 (Generate training data)"
    )
    parser.add_argument(
        "--skip_train", action="store_true", help="Skip Step 2 (Train GNN models)"
    )
    parser.add_argument(
        "--skip_inference", action="store_true", help="Skip Step 3 (Inference steps)"
    )
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--timeout", default="600", help="Timeout")
    parser.add_argument("--threads", default=8, help="Number of threads")

    args = parser.parse_args()

    repo_root = Path.cwd()  # ✅ always the directory you launched the script from
    deep_exe = Path(args.bin)
    exp_dir = Path(args.exp_dir)
    timeout = str(args.timeout)

    if not deep_exe.exists():
        print(f"✖ deep executable not found: {deep_exe}", file=sys.stderr)
        sys.exit(2)

    # Step 1: Generate training data
    if not args.skip_gen:
        step1_cmd = f"{sys.executable} scripts/gnn_exp/create_all_training_data.py {shlex.quote(str(exp_dir))} --deep_exe {shlex.quote(str(deep_exe))} --dataset_type {args.dataset_type}"
        run_cmd(step1_cmd, cwd=repo_root)
    else:
        print("\n⏭ Skipping Step 1 (Generate training data)")

    # Step 2: Train models
    if not args.skip_train:
        step2_cmd = f"{sys.executable} scripts/gnn_exp/train_models.py {shlex.quote(str(exp_dir))} --dataset_type {args.dataset_type}"
        run_cmd(step2_cmd, cwd=repo_root)
    else:
        print("\n⏭ Skipping Step 2 (Train GNN models)")

    if not args.skip_inference:
        # Always run: GNN with A* search
        jobs = [
            {
                "name": "step3a_gnn_astar",
                "threads": args.threads,
                "binary_args": _base_args(
                    args.dataset_type, with_search="Astar", with_upstream="GNN"
                ),
            }
        ]

        if args.ablation:
            # Ablation: GNN with HFS search
            jobs.append(
                {
                    "name": "step3e_gnn_hfs",
                    "threads": args.threads,
                    "binary_args": _base_args(
                        args.dataset_type, with_search="HFS", with_upstream="GNN"
                    ),
                }
            )
        else:
            # BFS (no -s/-u), HEFS and FULL parallel variants (with -p)
            jobs.extend(
                [
                    {
                        "name": "step3b_bfs",
                        "threads": args.threads,
                        "binary_args": _base_args(args.dataset_type),
                    },
                    {
                        "name": "step3c_hefs",
                        "threads": 1,
                        "binary_args": _base_args(args.dataset_type, extra="-p 5"),
                    },
                    {
                        "name": "step3d_full_parallel",
                        "threads": 1,
                        "binary_args": _base_args(args.dataset_type, extra="-p 6"),
                    },
                ]
            )

        # ---- execute --------------------------------------------------------------

        for j in jobs:
            cmd = _bulk_cmd(
                deep_exe,
                exp_dir,
                threads=j["threads"],
                binary_args=j["binary_args"],
                timeout=timeout,
            )
            run_cmd(cmd, cwd=repo_root)

        print("\n✅ Done. Results aggregated into the `_results` folder.")


# TODO: ablation: SOLO 3a 3e per ogni dataset_type / 100k generazione
# TODO: ricorda a Fra di sistemare readme batch ablation
# TODO: tables command

if __name__ == "__main__":
    main()
