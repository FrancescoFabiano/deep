import csv
import multiprocessing
import subprocess
import re
import shutil
import shlex
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
import argparse

NUMERIC_COLUMNS = ["PlanLength", "NodesExpanded", "TotalExecutionTime", "InitTime", "SearchTime", "ThreadOverhead"]

def latex_escape_underscores(s: str) -> str:
    return s.replace('_', r'\_')

SEARCH_NAME_MAP = {
    "Breadth First Search": "BFS",
    "Depth First Search": "DFS",
    "Heuristics First Search": "HFS",
    "Astar": "A*",
    "A Star": "A*",
    "AStar": "A*",
}

def parse_output(output: str):
    def extract(pat, default='-'):
        m = re.search(pat, output)
        return m.group(1).strip() if m else default

    m = re.search(r"Search used:\s*([A-Za-z ]+)(\([^)]+\))?", output)
    if m:
        raw_search = m.group(1).strip()
        parens = m.group(2) or ""
        mapped_search = SEARCH_NAME_MAP.get(raw_search, raw_search)
        search_used = f"{mapped_search}{parens}"
    else:
        search_used = "-"

    return {
        "GoalFound": "Yes" if "Goal found :)" in output else "No",
        "ActionExecuted": extract(r"Action executed:\s*(.*)"),
        "PlanLength": extract(r"Plan length:\s*(\d+)"),
        "SearchUsed": search_used,
        "NodesExpanded": extract(r"Nodes expanded:\s*(\d+)"),
        "TotalExecutionTime": extract(r"Total execution time:\s*(\d+)\s*ms"),
        "InitTime": extract(r"Initial state construction.*?:\s*(\d+)\s*ms"),
        "SearchTime": extract(r"Search time:\s*(\d+)\s*ms"),
        "ThreadOverhead": extract(r"Thread management overhead:\s*(\d+)\s*ms"),
    }

def run_instance(binary_path, filepath, binary_args, search_prefix, timeout):
    folder = Path(filepath).parent.name
    file_id = Path(filepath).stem
    cmd = [binary_path, filepath] + shlex.split(binary_args)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        output = res.stdout
        timeout_occurred = False
    except subprocess.TimeoutExpired as e:
        output = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        timeout_occurred = True

    metrics = parse_output(output)

    if timeout_occurred or "Goal found :)" not in output:
        metrics["GoalFound"] = "TO"

    if metrics["SearchUsed"] != "-" and search_prefix:
        metrics["SearchUsed"] = f"{search_prefix}-{metrics['SearchUsed']}"
    else:
        metrics["SearchUsed"] = f"{metrics['SearchUsed']}"

    return {"InputFolder": folder, "FileName": file_id, **metrics}

def find_problem_files_recursive(subfolder):
    return sorted(str(f) for f in Path(subfolder).rglob("*") if f.suffix in {".txt", ".eppdl"})

def compute_summary(rows):
    solved = sum(1 for r in rows if r.get("GoalFound") == "Yes")
    total = len(rows)
    percent = round((solved / total) * 100) if total else 0
    solved_str = f"{solved}/{total} ({percent}\\%)"

    averages = {}
    for col in NUMERIC_COLUMNS:
        vals = [int(r[col]) for r in rows if r[col].isdigit()]
        averages[col] = round(mean(vals), 2) if vals else "-"

    averages["Solved"] = solved_str
    return solved, total, averages

def format_table_header(caption):
    return [
        f"\\subsection*{{{caption}}}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Problem & Goal & Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) & Search \\\\",
        "\\midrule"
    ]

def format_table_footer(averages):
    return [
        f"\\textbf{{Averages:}} & {averages['Solved']} & {averages['PlanLength']} & {averages['NodesExpanded']} & {averages['TotalExecutionTime']} & {averages['InitTime']} & {averages['SearchTime']} & {averages['ThreadOverhead']} & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[0.7cm]"
    ]

def format_single_table(rows, averages, caption):
    lines = format_table_header(caption)
    rows_sorted = sorted(rows, key=lambda r: r["FileName"])
    for r in rows_sorted:
        lines.append(" & ".join([
            latex_escape_underscores(r["FileName"]),
            r["GoalFound"],
            r["PlanLength"],
            r["NodesExpanded"],
            r["TotalExecutionTime"],
            r["InitTime"],
            r["SearchTime"],
            r["ThreadOverhead"],
            latex_escape_underscores(r["SearchUsed"])
        ]) + " \\\\")
    lines.extend(format_table_footer(averages))
    return lines

def process_domain_split(binary_path, split_path, threads, binary_args, search_prefix, timeout):
    files = find_problem_files_recursive(split_path)
    if not files:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_instance, binary_path, f, binary_args, search_prefix, timeout) for f in files]
        for future in as_completed(futures):
            results.append(future.result())
    return results

def generate_domain_combined_tex(run_folder, domain_name, training_rows, test_rows, search_prefix):
    training_solved, training_total, training_avg = compute_summary(training_rows) if training_rows else (0, 0, {c: "-" for c in NUMERIC_COLUMNS})
    test_solved, test_total, test_avg = compute_summary(test_rows) if test_rows else (0, 0, {c: "-" for c in NUMERIC_COLUMNS})
    combined_rows = training_rows + test_rows
    combined_solved, combined_total, combined_avg = compute_summary(combined_rows) if combined_rows else (0, 0, {c: "-" for c in NUMERIC_COLUMNS})

    tex_lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}",
        f"\\section*{{Results for domain: \\texttt{{{latex_escape_underscores(domain_name)}}}}}",
        f"\\textbf{{Search prefix:}} {latex_escape_underscores(search_prefix)}",
        "\\\\[0.5cm]"
    ]

    if training_rows:
        tex_lines.extend(format_single_table(training_rows, training_avg, f"Training (Solved {training_solved}/{training_total})"))
    else:
        tex_lines.append("\\textit{No Training data available}\\\\[0.5cm]")

    if test_rows:
        tex_lines.extend(format_single_table(test_rows, test_avg, f"Test (Solved {test_solved}/{test_total})"))
    else:
        tex_lines.append("\\textit{No Test data available}\\\\[0.5cm]")

    if combined_rows:
        tex_lines.extend(format_single_table(combined_rows, combined_avg, f"Combined (Solved {combined_solved}/{combined_total})"))
    else:
        tex_lines.append("\\textit{No Combined data available}\\\\[0.5cm]")

    tex_lines.append("\\end{document}")

    tex_path = run_folder / f"{domain_name}_combined.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    compile_latex_to_pdf(tex_path)

def compile_latex_to_pdf(tex_path: Path):
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            print(f"\nLaTeX compilation failed for {tex_path}")
            print(proc.stdout)
            print(proc.stderr)
        else:
            for ext in [".aux", ".log", ".out"]:
                f = tex_path.with_suffix(ext)
                if f.exists():
                    f.unlink()
    except FileNotFoundError:
        print("Error: 'pdflatex' not found. Please install a LaTeX distribution.")

def generate_monolithic_combined_tex(run_folder, all_training, all_test, search_prefix):
    from statistics import mean

    def compute_avg(rows):
        if not rows:
            return {c: "-" for c in NUMERIC_COLUMNS}
        return compute_summary(rows)[2]

    training_avg = compute_avg(all_training)
    test_avg = compute_avg(all_test)
    combined_rows = all_training + all_test
    combined_avg = compute_avg(combined_rows)

    def format_rows(rows):
        lines = []
        for r in sorted(rows, key=lambda x: (x["InputFolder"], x["FileName"])):
            fields = [
                latex_escape_underscores(r.get("Mode", "")),      # Mode: Training or Test
                latex_escape_underscores(r.get("Domain", "")),    # Domain folder name
                latex_escape_underscores(r["FileName"]),
                r["GoalFound"],
                r["PlanLength"],
                r["NodesExpanded"],
                r["TotalExecutionTime"],
                r["InitTime"],
                r["SearchTime"],
                r["ThreadOverhead"],
                latex_escape_underscores(r["SearchUsed"])
            ]
            lines.append(" & ".join(fields) + " \\\\")
        return lines

    def format_footer(avg):
        return [
            f"\\textbf{{Averages:}} & & & {avg['Solved']} & {avg['PlanLength']} & {avg['NodesExpanded']} & "
            f"{avg['TotalExecutionTime']} & {avg['InitTime']} & {avg['SearchTime']} & "
            f"{avg['ThreadOverhead']} & \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\newpage"
        ]

    tex_lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage[a4paper, landscape, margin=1in]{geometry}",
        "\\begin{document}",
        "\\section*{Monolithic Combined Results (All Domains)}",
        f"\\textbf{{Search prefix:}} {latex_escape_underscores(search_prefix)}",
        "\\\\[0.5cm]"
    ]

    for split_name, rows, avg in [
        ("Training", all_training, training_avg),
        ("Test", all_test, test_avg),
        ("Combined (Training + Test)", combined_rows, combined_avg)
    ]:
        tex_lines.extend([
            f"\\subsection*{{{split_name}}}",
            # Add one more 'l' for the mode column
            "\\begin{tabular}{lllcccccccc}",
            "\\toprule",
            # Add "Mode" column header first
            "Mode & Domain & Problem & Goal & Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) & Search \\\\",
            "\\midrule"
        ])
        tex_lines.extend(format_rows(rows))
        tex_lines.extend(format_footer(avg))

    tex_lines.append("\\end{document}")

    tex_path = run_folder / "monolithic_combined_results.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    compile_latex_to_pdf(tex_path)


def main_all_domains(binary_path, parent_folder, threads, binary_args, search_prefix, timeout):
    run_folder = get_next_run_folder()

    all_training_global = []
    all_test_global = []
    domain_results = []

    has_heuristics = "-u GNN" in binary_args or "--heuristics GNN" in binary_args

    for domain_folder in sorted(Path(parent_folder).iterdir()):
        if not domain_folder.is_dir() or domain_folder.name.startswith('_'):
            continue  # Skip folders starting with underscore

        domain_name = domain_folder.name
        print(f"\nProcessing domain: {domain_name}")

        training_path = domain_folder / "Training"
        test_path = domain_folder / "Test"

        # Append model file path if heuristics used
        domain_binary_args = binary_args
        if has_heuristics:
            model_path = Path(parent_folder) / "_models" / domain_name / "distance_estimator.onnx"
            domain_binary_args += f" --GNN_model {model_path}"
            constant_path = Path(parent_folder) / "_models" / domain_name / "distance_estimator_C.txt"
            domain_binary_args += f" --GNN_constant_file {constant_path}"

        print(f"binary_args for {domain_name}: {domain_binary_args}")

        training_rows = process_domain_split(binary_path, training_path, threads, domain_binary_args, search_prefix, timeout) if training_path.exists() else []
        test_rows = process_domain_split(binary_path, test_path, threads, domain_binary_args, search_prefix, timeout) if test_path.exists() else []

        domain_results.append((domain_name, training_rows, test_rows))

        for r in training_rows:
            r["Domain"] = domain_name
            r["Mode"] = "Training"
        all_training_global.extend(training_rows)

        for r in test_rows:
            r["Domain"] = domain_name
            r["Mode"] = "Test"
        all_test_global.extend(test_rows)

        generate_domain_combined_tex(run_folder, domain_name, training_rows, test_rows, search_prefix)

    generate_monolithic_combined_tex(run_folder, all_training_global, all_test_global, search_prefix)

    generate_monolithic_combined_tex(run_folder, all_training_global, all_test_global, search_prefix)



    # Move all output files to a results folder under parent_folder
    search_val, heuristics_val = extract_search_heuristics(binary_args)
    parent_path = Path(parent_folder)
    dest_dir = get_unique_results_dir(parent_path, search_val, heuristics_val)

    for item in run_folder.iterdir():
        if item.name.endswith("_combined.tex") or item.name.endswith("_combined.pdf"):
            # Domain-specific result; move into its own subfolder
            domain_name = item.stem.replace("_combined", "")
            domain_subdir = dest_dir / domain_name
            domain_subdir.mkdir(exist_ok=True)
            shutil.move(str(item), str(domain_subdir / item.name))
        else:
            # Monolithic or shared result; move directly into root results dir
            shutil.move(str(item), str(dest_dir / item.name))

    print(f"\nAll results exported to {dest_dir}")
    print(f"LaTeX Tables:\n"
          f"  ├─ training_results.tex\n"
          f"  ├─ test_results.tex\n"
          f"  ├─ all_results_combined.tex\n"
          f"  └─ summary_averages_*.tex")
    print(f"CSV Files:\n"
          f"  ├─ all_results_combined.csv\n"
          f"  └─ summary_averages.csv")


def get_next_run_folder(base="out/coverage_results"):
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    runs = [int(d.name.split('_')[1]) for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_") and d.name.split('_')[1].isdigit()]
    next_run = max(runs, default=0) + 1
    run_folder = base_path / f"run_{next_run}"
    run_folder.mkdir()
    return run_folder

def parse_portfolio_threads(binary_args):
    match = re.search(r"(?:-p|--portfolio_threads)\s+(\d+)", binary_args)
    return int(match.group(1)) if match else 1

def get_arg_parser():
    p = argparse.ArgumentParser(
        description="Run planner binary on all problem files in domain subfolders, gather results and generate combined LaTeX tables.",
        epilog="""Example usage:
  python coverage_run.py ./cmake-build-debug-nn/bin/deep exp/aaai26/batch1 \\
    --threads 2 \\
    --binary_args "-b -c -p 3" \\
    --timeout 600
This will run the planner with 2 threads on all .txt and .eppdl files under ./exp/coverage/*,
passing '-b -c -p 3' to the binary, prefixing search names with 'P-', and timing out each run after 600 seconds.
""")
    p.add_argument("binary", help="Path to planner binary executable")
    p.add_argument("parent_folder", help="Parent folder containing domain subfolders with problem files")
    p.add_argument("--threads", type=int, default=1, help="Number of parallel thread workers")
    p.add_argument("--binary_args", type=str, default="", help="Additional args to pass to the binary")
    p.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds")
    return p

def extract_search_heuristics(binary_args: str):
    search_match = re.search(r"(?:-s|--search)\s+(\S+)", binary_args)
    heuristics_match = re.search(r"(?:-u|--heuristics)\s+(\S+)", binary_args)

    search = search_match.group(1) if search_match else "BFS"
    heuristics = heuristics_match.group(1) if heuristics_match else ""
    return search, heuristics

def get_unique_results_dir(base_dir: Path, search: str, heuristics: str) -> Path:
    base_name = f"{search}_{heuristics}" if heuristics else f"{search}"
    results_dir = base_dir / "_results" / base_name
    counter = 1
    while results_dir.exists():
        results_dir = base_dir / "_results" / f"{base_name}_{counter}"
        counter += 1
    results_dir.mkdir(parents=True)
    return results_dir

if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    available_cores = multiprocessing.cpu_count()
    portfolio_threads = parse_portfolio_threads(args.binary_args)

    total_threads_used = args.threads * portfolio_threads
    if total_threads_used > available_cores - 1:
        print(f"Error: Total threads requested ({args.threads} * {portfolio_threads} = {total_threads_used}) "
              f"exceeds available cores - 1 ({available_cores - 1}). Reduce --threads or --portfolio_threads.")
        sys.exit(1)

    search_prefix = "P" if portfolio_threads > 1 else ""

    main_all_domains(
        binary_path=args.binary,
        parent_folder=args.parent_folder,
        threads=args.threads,
        binary_args=args.binary_args,
        search_prefix=search_prefix,
        timeout=args.timeout
    )
