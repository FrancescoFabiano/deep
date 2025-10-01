#!/usr/bin/env python3
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

def main():
    parser = argparse.ArgumentParser(
        description="Run Steps 1–3 (create training data → train models → evaluate/aggregate) with hardcoded defaults."
    )
    parser.add_argument("--bin", required=True,
                        help="Path to the compiled deep executable (e.g., cmake-build-debug-nn/bin/deep)")
    parser.add_argument("--dataset_type", default="HASHED", choices=["MAPPED", "HASHED", "BITMASK"],
                        help="State representation to use across Steps 1–3 (default: HASHED)")
    parser.add_argument("--exp_dir", default="exp/gnn_exp/batchTest",
                        help="Experiment folder (default: exp/gnn_exp/batchTest)")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip Step 1 (Generate training data)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip Step 2 (Train GNN models)")

    args = parser.parse_args()

    repo_root = Path.cwd()   # ✅ always the directory you launched the script from
    deep_exe = Path(args.bin)
    exp_dir = Path(args.exp_dir)

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

    # Step 3a: Run evaluation with GNN
    gnn_binary_args = f'-s Astar -u GNN -c -b --dataset_type {args.dataset_type}'
    step3a_cmd = " ".join([
        sys.executable, "scripts/gnn_exp/bulk_coverage_run.py",
        shlex.quote(str(deep_exe)), shlex.quote(str(exp_dir)),
        "--threads 8",
        "--binary_args", shlex.quote(gnn_binary_args),
        "--timeout 600",
    ])
    run_cmd(step3a_cmd, cwd=repo_root)

    # Step 3b: Run evaluation with BFS
    bfs_binary_args = f'-c -b --dataset_type {args.dataset_type}'
    step3b_cmd = " ".join([
        sys.executable, "scripts/gnn_exp/bulk_coverage_run.py",
        shlex.quote(str(deep_exe)), shlex.quote(str(exp_dir)),
        "--threads 8",
        "--binary_args", shlex.quote(bfs_binary_args),
        "--timeout 600",
    ])
    run_cmd(step3b_cmd, cwd=repo_root)

    # Step 3c: Run evaluation with H-EFP
    hefs_binary_args = f'-c -b --dataset_type {args.dataset_type} -p 5'
    step3c_cmd = " ".join([
        sys.executable, "scripts/gnn_exp/bulk_coverage_run.py",
        shlex.quote(str(deep_exe)), shlex.quote(str(exp_dir)),
        "--threads 1",
        "--binary_args", shlex.quote(hefs_binary_args),
        "--timeout 600",
    ])
    run_cmd(step3c_cmd, cwd=repo_root)

    # Step 3d: Run evaluation with FULL PARALLELISM
    full_binary_args = f'-c -b --dataset_type {args.dataset_type} -p 6'
    step3d_cmd = " ".join([
        sys.executable, "scripts/gnn_exp/bulk_coverage_run.py",
        shlex.quote(str(deep_exe)), shlex.quote(str(exp_dir)),
        "--threads 1",
        "--binary_args", shlex.quote(full_binary_args),
        "--timeout 600",
    ])
    run_cmd(step3d_cmd, cwd=repo_root)

    print("\n✅ Done. Results aggregated into the `_results` folder.")

if __name__ == "__main__":
    main()
