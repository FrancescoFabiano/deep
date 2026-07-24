import csv, subprocess, re, shlex, threading, time
from pathlib import Path
from statistics import mean
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

NUMERIC_COLUMNS = [
    "PlanLength","NodesExpanded","TotalExecutionTime",
    "InitTime","SearchTime","ThreadOverhead"
]

TIMEOUT = 600


# ---------- FLAG BUILDER ----------
def build_flags(base_folder, domain_name, fringe, strict, extra_args):
    flags = ["-b", "-c"]

    # always pass fringe
    flags += ["--RL_fringe_size", str(fringe)]

    if strict:
        flags.append("--strong_equality")

    uses_rl = any(
        kw in extra_args
        for kw in ["--search RL", "--heuristics RL", "RL_H", "RL_heuristics"]
    )

    model_path = None

    if uses_rl:
        data_root = Path(base_folder).parent.parent
        model_path = data_root / "_models" / domain_name / f"frontier_policy_{fringe}.onnx"
        model_path = model_path.resolve()

        flags += ["--RL_model", str(model_path)]

    return flags, model_path, uses_rl


# ---------- RUN INSTANCE ----------
def run_instance(binary, file, args_list):
    thread_id = threading.get_ident()
    file = Path(file)

    #print(f"[DEBUG][T{thread_id}] START {file}")

    cmd = [str(binary), str(file)] + args_list
    #print(f"[DEBUG][T{thread_id}] CMD: {' '.join(map(str, cmd))}")

    start = time.time()

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT
        )
        out = res.stdout
        timeout_flag = False
        error_flag = res.returncode != 0

        if error_flag:
            print(f"\n[ERROR][T{thread_id}] Return code: {res.returncode}")
            print(out)

    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT][T{thread_id}] {file}")
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        timeout_flag = True
        error_flag = False
        print(out)

    except Exception as e:
        print(f"[ERROR][T{thread_id}] {file}: {e}")
        out = ""
        timeout_flag = False
        error_flag = True

    elapsed = round(time.time() - start, 2)

    def ex(p):
        m = re.search(p, out)
        return m.group(1) if m else "-"

    result = {
        "File": file.name,
        "GoalFound": "Yes" if ("Goal found" in out and not timeout_flag and not error_flag) else "No",
        "PlanLength": ex(r"Plan length:\s*(\d+)"),
        "NodesExpanded": ex(r"Nodes expanded:\s*(\d+)"),
        "TotalExecutionTime": ex(r"Total execution time:\s*(\d+)"),
        "InitTime": ex(r"Initial state.*?:\s*(\d+)"),
        "SearchTime": ex(r"Search time:\s*(\d+)"),
        "ThreadOverhead": ex(r"Thread.*?:\s*(\d+)"),
        "WallTime": elapsed,
        "Status": "OK"
    }

    if timeout_flag:
        result["GoalFound"] = "TO"
        result["Status"] = "TIMEOUT"
    elif error_flag:
        result["GoalFound"] = "ERR"
        result["Status"] = "ERROR"

    #print(f"[DEBUG][T{thread_id}] DONE {file} -> {result['GoalFound']}")
    return result


# ---------- STATS ----------
def compute_stats(rows):
    solved = [r for r in rows if r["GoalFound"] == "Yes"]
    stats = {"Solved": len(solved), "Total": len(rows)}

    for col in NUMERIC_COLUMNS:
        vals = [int(r[col]) for r in solved if r[col].isdigit()]
        stats[f"{col}_mean"] = round(mean(vals), 2) if vals else "-"

    stats["WallTime_mean"] = round(mean([r["WallTime"] for r in rows]), 2) if rows else "-"

    return stats


# ---------- MAIN ----------
def main(binary, folder, fringe, strict, extra_args):
    #print("[DEBUG] ===== START =====")

    folder_path = Path(folder)
    domain_name = folder_path.parent.name

    print(f"[RUN] domain={domain_name}")
    print(f"[RUN] fringe={fringe} | strict={strict}")
    print(f"[RUN] extra_args={extra_args}")

    flags, model_path, uses_rl = build_flags(folder_path, domain_name, fringe, strict, extra_args)
    args_list = flags + shlex.split(extra_args)

    print(f"[RUN] RL used: {uses_rl}")
    if model_path:
        print(f"[RUN] model: {model_path}")
        if not model_path.exists():
            print(f"[WARNING] missing model: {model_path}")
    else:
        print("[RUN] model: (none)")

    print(f"[RUN] final args: {' '.join(map(str, args_list))}")

    files = list(folder_path.rglob("*.txt"))
    #print(f"[DEBUG] Found {len(files)} problems")

    # max_threads = max(1, int(multiprocessing.cpu_count() * 0.9))
    max_threads = 8
    #print(f"[DEBUG] Using {max_threads} threads")

    results = []

    with ThreadPoolExecutor(max_workers=max_threads) as ex:
        futures = [ex.submit(run_instance, binary, f, args_list) for f in files]

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] Future failed: {e}")
            print(f"[DEBUG] Progress {i}/{len(files)}")

    Path("results").mkdir(exist_ok=True)
    csv_path = Path("results/summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "File","GoalFound","Status",
            "PlanLength","NodesExpanded","TotalExecutionTime",
            "InitTime","SearchTime","ThreadOverhead","WallTime"
        ])

        writer.writeheader()
        for r in sorted(results, key=lambda x: x["File"]):
            writer.writerow(r)

        f.write("\n")

        stats = compute_stats(results)
        summary_writer = csv.DictWriter(f, fieldnames=stats.keys())
        summary_writer.writeheader()
        summary_writer.writerow(stats)

    #print(f"[DEBUG] Results written to {csv_path}")
    #print("[DEBUG] ===== END =====")


# ---------- ENTRY ----------
if __name__ == "__main__":
    binary = sys.argv[1]
    folder = sys.argv[2]
    fringe = int(sys.argv[3])
    strict = sys.argv[4].lower() == "true"
    extra_args = " ".join(sys.argv[5:])

    main(binary, folder, fringe, strict, extra_args)
