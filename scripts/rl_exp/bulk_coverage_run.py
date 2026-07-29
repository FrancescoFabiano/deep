import csv, subprocess, re, shlex, threading, time
import os, signal
from pathlib import Path
from statistics import mean
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

NUMERIC_COLUMNS = [
    "PlanLength","NodesExpanded","TotalExecutionTime",
    "InitTime","SearchTime","ThreadOverhead"
]

TIMEOUT = 30
EXPLOITATION_PERCENTAGE = 10

# Per-instance resident-memory ceiling. When a planner run's RSS crosses this,
# it is killed and recorded as MEMOUT (analogous to TIMEOUT). RSS (physical RAM)
# is used rather than an RLIMIT_AS cap, because ONNX Runtime + threads reserve a
# large *virtual* address space that would trip an address-space limit far below
# real usage.
MEMORY_LIMIT_GB = 16
MEMORY_LIMIT_BYTES = int(MEMORY_LIMIT_GB * 1024 ** 3)
# How often the monitor thread samples RSS while a run is in flight (seconds).
MEMORY_POLL_INTERVAL = 0.25


def _rss_bytes(pid):
    """Resident set size of `pid` in bytes (0 if the process is gone).

    Threads share the address space, so VmRSS already covers a multithreaded
    planner. Child processes (not expected here) are not summed.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # format: "VmRSS:\t   123456 kB"
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return 0
    return 0


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
        # flags += ["--RL_exploration", str(EXPLORATION_PERCENTAGE)]
        flags += ["--RL_exploitation", str(EXPLOITATION_PERCENTAGE)]
        flags += ["--RL_adaptive"]
        flags += ["--RL_adaptive_signal", "budget"]

    return flags, model_path, uses_rl


# ---------- RUN INSTANCE ----------
def run_instance(binary, file, args_list):
    thread_id = threading.get_ident()
    file = Path(file)

    #print(f"[DEBUG][T{thread_id}] START {file}")

    cmd = [str(binary), str(file)] + args_list
    #print(f"[DEBUG][T{thread_id}] CMD: {' '.join(map(str, cmd))}")

    start = time.time()

    timeout_flag = False
    memout_flag = False
    error_flag = False
    out = ""

    try:
        # start_new_session=True puts the child in its own process group so we can
        # SIGKILL the whole group (child + any descendants) on timeout / memout.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        # Per-process RSS watchdog: kills THIS run if it crosses the 16 GB ceiling.
        memout_event = threading.Event()

        def _watch():
            while proc.poll() is None:
                if _rss_bytes(proc.pid) > MEMORY_LIMIT_BYTES:
                    memout_event.set()
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    return
                time.sleep(MEMORY_POLL_INTERVAL)

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

        try:
            out, _ = proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            out, _ = proc.communicate()
            if not memout_event.is_set():
                timeout_flag = True

        watcher.join(timeout=1.0)
        out = out or ""

        if memout_event.is_set():
            memout_flag = True
            print(f"[MEMOUT][T{thread_id}] {file} (> {MEMORY_LIMIT_GB} GB RSS)")
        elif timeout_flag:
            print(f"[TIMEOUT][T{thread_id}] {file}")
            print(out)
        else:
            error_flag = proc.returncode != 0
            if error_flag:
                print(f"\n[ERROR][T{thread_id}] Return code: {proc.returncode}")
                print(out)

    except Exception as e:
        print(f"[ERROR][T{thread_id}] {file}: {e}")
        out = ""
        timeout_flag = False
        memout_flag = False
        error_flag = True

    elapsed = round(time.time() - start, 2)

    def ex(p):
        m = re.search(p, out)
        return m.group(1) if m else "-"

    result = {
        "File": file.name,
        "GoalFound": "Yes" if ("Goal found" in out and not timeout_flag and not memout_flag and not error_flag) else "No",
        "PlanLength": ex(r"Plan length:\s*(\d+)"),
        "NodesExpanded": ex(r"Nodes expanded:\s*(\d+)"),
        "TotalExecutionTime": ex(r"Total execution time:\s*(\d+)"),
        "InitTime": ex(r"Initial state.*?:\s*(\d+)"),
        "SearchTime": ex(r"Search time:\s*(\d+)"),
        "ThreadOverhead": ex(r"Thread.*?:\s*(\d+)"),
        "WallTime": elapsed,
        "Status": "OK"
    }

    if memout_flag:
        result["GoalFound"] = "MO"
        result["Status"] = "MEMOUT"
    elif timeout_flag:
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
