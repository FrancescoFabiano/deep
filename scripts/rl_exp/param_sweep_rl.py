"""
RL exploration/exploitation parameter sweep, per instance.

Runs the `deep` planner over a grid of (instance x fringe x exploration/exploitation
split x seed) for a single RL heuristic, and records SOLVED / TIMEOUT / error plus
nodes / plan length / time. The goal is to see whether some VALID split beats the
production default (10/70) on instances that time out under it — i.e. whether the low
RL coverage is a bad fixed operating point rather than model quality.

Guardrails baked in (each was a real trap):
  * fringe must have a model: only F with an existing frontier_policy_F.onnx are run
    (a missing model exits code 905 / rc 137, which is NOT an OOM).
  * exploration + exploitation must be < 100 (strictly), else the binary exits code
    100 without ever searching.

Usage (defaults reproduce the CC timeout-prone sample):
    python3 scripts/rl_exp/param_sweep_rl.py \
        --batch-dir exp/rl_exp/batch1_1 --domain CC --split Test \
        --method AVG --fringes 16 32 \
        --pairs 10:70 10:85 20:60 30:50 \
        --seeds 94 --timeout 45 \
        --out /tmp/cc_param_sweep.csv

--instances defaults to a built-in CC timeout-prone list; pass names (without .txt)
or a path to a file with one instance name per line to override.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

BIN = "./cmake-build-release-nn/bin/deep"

# CC instances that time out most often in combined_results/batch1/CC (test split).
DEFAULT_CC_INSTANCES = [
    "CC_3_2_3__pl_7",
    "CC_2_3_4__pl_5",
    "CC_3_3_3__pl_6",
    "CC_2_3_4__pl_4",
    "CC_2_2_4__pl_7",
    "CC_2_3_4__pl_6",
]

# heuristic name -> the search flags that select it
RL_H_SET = {"MIN", "MAX", "AVG", "RNG"}


def method_flags(method):
    """Search flags for an RL method. AVG/MIN/MAX/RNG go through RL_H; the
    subgoal/potential heuristics (SUBGOALS/L_PG/C_PG/S_PG) are passed directly."""
    if method in RL_H_SET:
        return ["--search", "RL", "--heuristics", "RL_H", "--RL_heuristics", method]
    return ["--search", "RL", "--heuristics", method]


def extract(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_one(inst_path, model_path, method, fringe, explore, exploit, seed, strict,
            separated, timeout):
    """Run one configuration; return a result dict."""
    cmd = [BIN, str(inst_path), "-b", "-c",
           "--RL_fringe_size", str(fringe),
           "--RL_model", str(model_path),
           "--RL_exploration", str(explore),
           "--RL_exploitation", str(exploit),
           "--RL_seed", str(seed)]
    cmd += method_flags(method)
    if strict:
        cmd.append("--strong_equality")
    if separated:
        cmd.append("--dataset_separated")

    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, out, timed_out = None, (e.stdout or "") if isinstance(e.stdout, str) else "", True
    wall = round(time.time() - start, 2)

    if timed_out:
        result = "TIMEOUT"
    elif rc == 100:
        result = "ARG-ERROR"
    elif rc == 905 or rc == 137:
        result = "MODEL-ERROR"
    elif "Goal found" in out:
        result = "SOLVED"
    else:
        result = f"NO-GOAL(rc={rc})"

    return {
        "result": result,
        "nodes": extract(r"Nodes expanded:\s*(\d+)", out),
        "plan": extract(r"Plan length:\s*(\d+)", out),
        "time": extract(r"Total execution time:\s*(\d+)", out),
        "wall": wall,
    }


def load_instances(arg):
    if not arg:
        return list(DEFAULT_CC_INSTANCES)
    if len(arg) == 1 and Path(arg[0]).is_file():
        return [ln.strip() for ln in Path(arg[0]).read_text().splitlines() if ln.strip()]
    return [a[:-4] if a.endswith(".txt") else a for a in arg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="exp/rl_exp/batch1_1",
                    help="dir holding <domain>/<split>/*.txt and _models/<domain>/")
    ap.add_argument("--domain", default="CC")
    ap.add_argument("--split", default="Test", choices=["Test", "Training"])
    ap.add_argument("--methods", nargs="+", default=["AVG"],
                    help="RL heuristics: any of MIN|MAX|AVG|RNG|SUBGOALS|L_PG|C_PG|S_PG")
    ap.add_argument("--fringes", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--pairs", nargs="+", default=["10:70", "10:85", "20:60", "30:50"],
                    help="exploration:exploitation percentages (sum must be < 100)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[94])
    ap.add_argument("--instances", nargs="*", default=None,
                    help="instance names (no .txt) or a file; default = CC timeout-prone list")
    ap.add_argument("--timeout", type=int, default=45, help="per-run wall timeout (s)")
    ap.add_argument("--no-strict", action="store_true", help="drop --strong_equality")
    ap.add_argument("--no-separated", action="store_true", help="drop --dataset_separated")
    ap.add_argument("--out", default="param_sweep.csv")
    opts = ap.parse_args()

    batch = Path(opts.batch_dir)
    model_dir = batch / "_models" / opts.domain
    inst_dir = batch / opts.domain / opts.split

    # keep only fringes that actually have a model
    fringes = []
    for f in opts.fringes:
        if (model_dir / f"frontier_policy_{f}.onnx").exists():
            fringes.append(f)
        else:
            print(f"[skip] fringe {f}: no model {model_dir}/frontier_policy_{f}.onnx")
    if not fringes:
        sys.exit(f"[FATAL] no valid fringe models under {model_dir}")

    # parse + validate splits (exploration + exploitation < 100)
    pairs = []
    for p in opts.pairs:
        ex, xp = (int(x) for x in p.split(":"))
        if ex + xp >= 100:
            print(f"[skip] split {ex}/{xp}: exploration+exploitation must be < 100")
            continue
        pairs.append((ex, xp))
    if not pairs:
        sys.exit("[FATAL] no valid exploration/exploitation splits")

    instances = load_instances(opts.instances)

    rows = []
    total = (len(opts.methods) * len(instances) * len(fringes)
             * len(pairs) * len(opts.seeds))
    print(f"[sweep] domain={opts.domain} methods={opts.methods} "
          f"instances={len(instances)} fringes={fringes} splits={pairs} "
          f"seeds={opts.seeds} -> {total} runs, timeout={opts.timeout}s")

    n = 0
    for method in opts.methods:
        for inst in instances:
            inst_path = inst_dir / f"{inst}.txt"
            if not inst_path.exists():
                print(f"[skip] instance not found: {inst_path}")
                continue
            for f in fringes:
                model = (model_dir / f"frontier_policy_{f}.onnx").resolve()
                for (ex, xp) in pairs:
                    for seed in opts.seeds:
                        n += 1
                        r = run_one(inst_path, model, method, f, ex, xp, seed,
                                    strict=not opts.no_strict,
                                    separated=not opts.no_separated,
                                    timeout=opts.timeout)
                        row = {"domain": opts.domain, "method": method,
                               "instance": inst, "fringe": f, "explore": ex,
                               "exploit": xp, "seed": seed, **r}
                        rows.append(row)
                        tag = "DEFAULT" if (ex, xp) == (10, 70) else ""
                        print(f"[{n}/{total}] {method:8s} {inst:22s} F={f:<2} "
                              f"{ex}/{xp} s={seed} -> {r['result']:14s} "
                              f"nodes={r['nodes'] or '-':>6} "
                              f"plan={r['plan'] or '-':>3} {tag}")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "method", "instance",
                                           "fringe", "explore", "exploit", "seed",
                                           "result", "nodes", "plan", "time", "wall"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] wrote {len(rows)} rows -> {out}")

    solved = lambda r: r["result"] == "SOLVED"

    # --- headline 1: solve rate per split (is the default the best operating point?) ---
    print("\n=== solve rate per exploration/exploitation split ===")
    by_split = {}
    for r in rows:
        by_split.setdefault((r["explore"], r["exploit"]), []).append(r)
    ranked = sorted(by_split.items(),
                    key=lambda kv: sum(solved(r) for r in kv[1]) / len(kv[1]),
                    reverse=True)
    for (ex, xp), rs in ranked:
        s = sum(solved(r) for r in rs)
        tag = "  <-- pipeline DEFAULT" if (ex, xp) == (10, 70) else ""
        print(f"  {ex:>2}/{xp:<2}: {s:3d}/{len(rs)} = {100*s/len(rs):4.0f}%{tag}")

    # --- headline 2: seed instability (same config flips solve<->fail on seed) ---
    cells = {}
    for r in rows:
        cells.setdefault((r["instance"], r["fringe"], r["explore"], r["exploit"]),
                         {})[r["seed"]] = solved(r)
    multi = [v for v in cells.values() if len(v) > 1]
    flips = sum(1 for v in multi if len(set(v.values())) > 1)
    if multi:
        print(f"\n[seed instability] {flips}/{len(multi)} configs flip solve<->fail "
              f"across seeds ({100*flips/len(multi):.0f}%) — a fixed production seed "
              f"lands on one side of that coin flip.")

    # --- per (instance, fringe): does any non-default split beat the default? ---
    print("\n=== does a non-default split solve where 10/70 does not? (any seed) ===")
    by_if = {}
    for r in rows:
        by_if.setdefault((r["instance"], r["fringe"]), []).append(r)
    wins = 0
    for (inst, f), rs in sorted(by_if.items()):
        default_solved = any(solved(r) for r in rs
                             if (r["explore"], r["exploit"]) == (10, 70))
        alt_solved = sorted({f"{r['explore']}/{r['exploit']}" for r in rs
                             if solved(r) and (r["explore"], r["exploit"]) != (10, 70)})
        status = "default SOLVES" if default_solved else "default TIMEOUT/fail"
        note = f"  alt splits that solve: {alt_solved}" if alt_solved else ""
        if not default_solved and alt_solved:
            wins += 1
            note += "   <== BETTER SPLIT EXISTS"
        print(f"  {inst:22s} F={f:<2} {status}{note}")
    print(f"\n[summary] instances(xfringe) where default fails but a valid split solves: {wins}")


if __name__ == "__main__":
    main()
