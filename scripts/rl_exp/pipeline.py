
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print(f"[PIPELINE] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def discover_domains(batch_root: Path):
    """A domain is any subdir of batch_root that contains a Training/ folder.
    Skips _models, __pycache__, and other dotted/underscored helper dirs."""
    domains = []
    for child in sorted(batch_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        if child.name == "__pycache__":
            continue
        if (child / "Training").is_dir():
            domains.append(child)
    return domains

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", help="e.g. exp/rl_exp/batch0_train")
    parser.add_argument("--separated", action="store_true",
                        help="run in separated (goal-free) mode; forwards --dataset_separated")
    parser.add_argument("--dry-run", action="store_true",
                        help="print discovered domains + run_all commands and exit (no simulation)")
    opts = parser.parse_args()

    batch_root = Path(opts.batch_root)
    if not batch_root.is_dir():
        sys.exit(f"[PIPELINE] batch_root not found: {batch_root}")

    domains = discover_domains(batch_root)
    if not domains:
        sys.exit(f"[PIPELINE] no domains (subdir with Training/) found under {batch_root}")

    sep = " --separated" if opts.separated else ""
    batch_name = batch_root.stem

    print(f"[PIPELINE] batch_root = {batch_root}")
    print(f"[PIPELINE] batch_name = {batch_name}")
    print(f"[PIPELINE] separated  = {opts.separated}")
    print(f"[PIPELINE] domains    = {[d.name for d in domains]}")

    # ---- per-domain: run the FULL chain scoped to combined_results/<batch>/<domain> ----
    # Each domain is fully self-contained: its own simulation CSVs, aggregate.csv,
    # analysis/, and plots. Running batchX then batchY never overwrites; CC and SC
    # never mix.
    for domain_path in domains:
        domain = domain_path.name
        scoped = f"combined_results/{batch_name}/{domain}"

        chain = [
            f"python3 scripts/rl_exp/run_all.py {domain_path}{sep} --out-dir {scoped}",
            f"python3 scripts/rl_exp/aggregate.py {scoped}",
            f"python3 scripts/rl_exp/analyze_results.py {scoped}",
            f"python3 scripts/rl_exp/advanced_analysis.py {scoped}",
            f"python3 scripts/rl_exp/plot_results.py {scoped}",
            f"python3 scripts/rl_exp/plot_results_best.py {scoped}",
            f"python3 scripts/rl_exp/plot_results_best_isolated.py {scoped}",
            f"python3 scripts/rl_exp/plot_by_pl.py {scoped}",
            f"python3 scripts/rl_exp/plot_fringe_comparison.py {scoped}",
        ]

        if opts.dry_run:
            print(f"[PIPELINE][DRY-RUN] domain={domain} -> {scoped}")
            for c in chain:
                print(f"[PIPELINE][DRY-RUN]   {c}")
        else:
            print(f"[PIPELINE] ===== domain={domain} -> {scoped} =====")
            for c in chain:
                run(c)

    if opts.dry_run:
        print("=== DRY-RUN COMPLETE (no simulation) ===")
        sys.exit(0)

    print("=== PIPELINE COMPLETE ===")
