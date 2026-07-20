import argparse, subprocess, shutil, shlex
from pathlib import Path

SCRIPT = "scripts/rl_exp/bulk_coverage_run.py"
BIN = "./cmake-build-release-nn/bin/deep"

#print("[DEBUG] Script started")

parser = argparse.ArgumentParser()
parser.add_argument("data", help="domain-level path, e.g. exp/rl_exp/batch0_train/CC")
parser.add_argument("--separated", action="store_true",
                    help="run in separated (goal-free) mode; adds --dataset_separated for the C++ binary")
parser.add_argument("--out-dir", default=None,
                    help="results dir. When given, it is already domain-scoped "
                         "(e.g. combined_results/batchX/CC) and results are written directly here. "
                         "When omitted (back-compat), defaults to combined_results/ with a per-domain subdir.")
cli = parser.parse_args()

DATA = Path(cli.data)
#print(f"[DEBUG] DATA path: {DATA} (exists={DATA.exists()})")

# separated (goal-free) mode is off by default -> merged behaviour, no extra flag
SEP = " --dataset_separated" if cli.separated else ""

# --out-dir given -> path is already domain-scoped, write straight to OUT.
# omitted -> legacy behaviour: combined_results/ with a per-domain subdir (OUT / domain).
SCOPED = cli.out_dir is not None
OUT = Path(cli.out_dir) if SCOPED else Path("combined_results")
OUT.mkdir(parents=True, exist_ok=True)
#print(f"[DEBUG] Output dir: {OUT.resolve()}")

FRINGES = [4, 8, 16, 32]
STRICT_FLAGS = [True]

def run(label, args, split_path, prefix, fringe, strict):
    #print("\n[DEBUG] ===== RUN START =====")

    domain = split_path.parent.name
    OUT_check = OUT if SCOPED else OUT / domain
    OUT_check.mkdir(parents=True, exist_ok=True)
    split = split_path.name

    uses_rl = "--search RL" in args

    model_path = None
    if uses_rl:
        data_root = split_path.parent.parent
        model_path = (data_root / "_models" / domain / f"frontier_policy_{fringe}.onnx").resolve()

    print(
        f"[RUN] domain={domain} | split={split} | "
        f"mode={label} | fringe={fringe} | strict={strict}"
    )

    print(f"[RUN] args: {args}")
    print(f"[RUN] path: {split_path}")

    if model_path:
        print(f"[RUN] model: {model_path}")
        if not model_path.exists():
            print(f"[WARNING] Model does NOT exist: {model_path}")
    else:
        print("[RUN] model: (none)")

    cmd = [
              "python3",
              SCRIPT,
              BIN,
              str(split_path),
              str(fringe),
              str(strict),
          ] + shlex.split(args)

    print(f"[RUN] command: {' '.join(map(str, cmd))}")

    subprocess.run(cmd, check=True)

    # ---- save results ----
    out_dir = OUT_check / ("bfs" if label == "BFS" else f"fringe_{fringe}")
    out_dir.mkdir(exist_ok=True)

    strict_tag = "strict" if strict else "nostrict"
    dst = out_dir / f"{prefix}_{label}_{strict_tag}.csv"

    if not Path("results/summary.csv").exists():
        raise FileNotFoundError("results/summary.csv missing")

    shutil.copy("results/summary.csv", dst)
    print(f"[DEBUG] saved -> {dst}")


# ---------- MAIN LOOP ----------

for strict in STRICT_FLAGS:
    #print(f"\n[DEBUG] ==== strict={strict} ====")

    for split in ["Training", "Test"]:
        split_path = DATA / split

        if not split_path.exists():
            print(f"[DEBUG] missing {split_path}, skipping")
            continue

        prefix = "train" if split == "Training" else "test"

        # ---- BFS (once) ----
        run("BFS", "--search BFS" + SEP, split_path, prefix, fringe=0, strict=strict)

        # ---- RL per fringe ----
        for fringe in FRINGES:
            #print(f"\n[DEBUG] ---- fringe={fringe} ----")

            for h in ["SUBGOALS","L_PG","C_PG","S_PG"]:
                run(
                    f"RL-{h}",
                    f"--search RL --heuristics {h}" + SEP,
                    split_path,
                    prefix,
                    fringe,
                    strict
                )

            for h in ["MIN","MAX","AVG","RNG"]:
                run(
                    f"RL-H-{h}",
                    f"--search RL --heuristics RL_H --RL_heuristics {h}" + SEP,
                    split_path,
                    prefix,
                    fringe,
                    strict
                )

#print("[DEBUG] Script finished")
