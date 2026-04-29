import subprocess, shutil, sys, shlex
from pathlib import Path

SCRIPT = "scripts/rl_exp/bulk_coverage_run.py"
BIN = "./cmake-build-release-nn/bin/deep"

#print("[DEBUG] Script started")

DATA = Path(sys.argv[1])
#print(f"[DEBUG] DATA path: {DATA} (exists={DATA.exists()})")

OUT = Path("combined_results")
OUT.mkdir(exist_ok=True)
#print(f"[DEBUG] Output dir: {OUT.resolve()}")

#FRINGES = [8, 16, 32, 64]
#STRICT_FLAGS = [True, False]
FRINGES = [32, 64]
STRICT_FLAGS = [True, False]

def run(label, args, split_path, prefix, fringe, strict):
    #print("\n[DEBUG] ===== RUN START =====")

    domain = split_path.parent.name
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
    out_dir = OUT / ("bfs" if label == "BFS" else f"fringe_{fringe}")
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
        run("BFS", "--search BFS", split_path, prefix, fringe=0, strict=strict)

        # ---- RL per fringe ----
        for fringe in FRINGES:
            #print(f"\n[DEBUG] ---- fringe={fringe} ----")

            for h in ["SUBGOALS","L_PG","C_PG","S_PG"]:
                run(
                    f"RL-{h}",
                    f"--search RL --heuristics {h}",
                    split_path,
                    prefix,
                    fringe,
                    strict
                )

            for h in ["MIN","MAX","AVG","RNG"]:
                run(
                    f"RL-H-{h}",
                    f"--search RL --heuristics RL_H --RL_heuristics {h}",
                    split_path,
                    prefix,
                    fringe,
                    strict
                )

#print("[DEBUG] Script finished")