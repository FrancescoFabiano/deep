import csv
import multiprocessing
import subprocess
import re
import shlex
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
import argparse

NUMERIC_COLUMNS = ["PlanLength", "NodesExpanded", "TotalExecutionTime", "InitTime", "SearchTime", "ThreadOverhead"]

def latex_escape_underscores(s: str) -> str:
  return s.replace('_', r'\_')

def parse_output(output: str):
  def extract(pat, default='-'):
    m = re.search(pat, output)
    return m.group(1).strip() if m else default
  return {
    "GoalFound": "Yes" if "Goal found :)" in output else "No",
    "ActionExecuted": extract(r"Action executed:\s*(.*)"),
    "PlanLength": extract(r"Plan length:\s*(\d+)"),
    "SearchUsed": extract(r"Search used:\s*(.*?)\s*(?:\(|$)"),
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
  except subprocess.TimeoutExpired as e:
    output = (e.stdout.decode('utf-8', errors='replace') if e.stdout else "")
  metrics = parse_output(output)
  if "Goal found :)" not in output:
    metrics["GoalFound"] = "TO"
  if metrics["SearchUsed"] != "-" and search_prefix:
    metrics["SearchUsed"] = f"{search_prefix}-{metrics['SearchUsed']}"
  elif search_prefix and metrics["GoalFound"] == "TO":
    metrics["SearchUsed"] = f"{search_prefix}-TO"
  elif metrics["GoalFound"] == "TO":
    metrics["SearchUsed"] = "TO"
  return {"InputFolder": folder, "FileName": file_id, **metrics}

def get_next_run_folder(base="out/coverage_results"):
  base_path = Path(base)
  base_path.mkdir(parents=True, exist_ok=True)
  runs = [int(d.name.split('_')[1]) for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_") and d.name.split('_')[1].isdigit()]
  next_run = max(runs, default=0) + 1
  run_folder = base_path / f"run_{next_run}"
  run_folder.mkdir()
  return run_folder

def find_problem_files_recursive(subfolder):
  return sorted(str(f) for f in Path(subfolder).rglob("*") if f.suffix in {".txt", ".eppdl"})

def compute_summary(rows):
  solved = sum(r["GoalFound"] == "Yes" for r in rows)
  averages = {}
  for col in NUMERIC_COLUMNS:
    vals = [int(r[col]) for r in rows if r[col].isdigit()]
    averages[col] = round(mean(vals), 2) if vals else "-"
  return solved, averages

def _format_table_header(solved_total, caption):
  return [
    f"\\subsection*{{Domain: \\texttt{{{caption}}}}}",
    f"\\textbf{{Solved:}} {solved_total['solved']} / {solved_total['total']}",
    "\\\\[0.3cm]",
    "\\begin{tabular}{lcccccccc}",
    "\\toprule",
    "Problem & Goal & Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) & Search \\\\",
    "\\midrule"
  ]

def _format_table_footer(averages):
  return [
    "\\bottomrule",
    "\\end{tabular}",
    "\\\\[0.5cm]",
    "\\textbf{Averages:}",
    "\\\\[0.5cm]",
    "\\begin{tabular}{cccccc}",
    "\\toprule",
    "Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) \\\\",
    "\\midrule",
    " " + " & ".join(str(averages[col]) for col in NUMERIC_COLUMNS) + " \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\\\[1cm]"
  ]

def format_latex_table_fragment(domain_name, rows, solved, averages):
  domain_esc = latex_escape_underscores(domain_name)
  rows_sorted = sorted(rows, key=lambda r: r["FileName"])
  lines = _format_table_header({'solved': solved, 'total': len(rows)}, domain_esc)
  for r in rows_sorted:
    lines.append(" & ".join([
      latex_escape_underscores(r["FileName"]),
      r["GoalFound"], r["PlanLength"], r["NodesExpanded"], r["TotalExecutionTime"],
      r["InitTime"], r["SearchTime"], r["ThreadOverhead"],
      latex_escape_underscores(r["SearchUsed"])
    ]) + " \\\\")
  lines += _format_table_footer(averages)
  return "\n".join(lines)

def format_latex_table(domain_name, rows, solved, averages):
  domain_esc = latex_escape_underscores(domain_name)
  rows_sorted = sorted(rows, key=lambda r: r["FileName"])
  header = [
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage[margin=1in]{geometry}",
    "\\usepackage[T1]{fontenc}",
    "\\usepackage{lmodern}",
    "\\begin{document}",
  ]
  header += _format_table_header({'solved': solved, 'total': len(rows)}, domain_esc)
  body = [
    " & ".join([
      latex_escape_underscores(r["FileName"]), r["GoalFound"], r["PlanLength"], r["NodesExpanded"],
      r["TotalExecutionTime"], r["InitTime"], r["SearchTime"], r["ThreadOverhead"],
      latex_escape_underscores(r["SearchUsed"])
    ]) + " \\\\" for r in rows_sorted
  ]
  footer = _format_table_footer(averages)
  footer.append("\\end{document}")
  return "\n".join(header + body + footer)

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
      # Cleanup auxiliary files
      for ext in [".aux", ".log", ".out"]:
        f = tex_path.with_suffix(ext)
        if f.exists():
          f.unlink()
  except FileNotFoundError:
    print("Error: 'pdflatex' not found. Please install a LaTeX distribution.")

def process_domain(binary_path, domain_path, threads, binary_args, search_prefix, timeout, out_folder):
  files = find_problem_files_recursive(domain_path)
  if not files:
    print(f"Skipping {domain_path}, no files found.")
    return None
  domain_name = Path(domain_path).name
  print(f"\nProcessing {domain_name} ({len(files)} files)")

  results = []
  with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = [executor.submit(run_instance, binary_path, f, binary_args, search_prefix, timeout) for f in files]
    for future in as_completed(futures):
      results.append(future.result())

  # Save CSV
  csv_path = out_folder / f"{domain_name}_results.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

  solved, averages = compute_summary(results)

  # Save standalone LaTeX and compile
  tex_path = out_folder / f"{domain_name}_table.tex"
  tex_path.write_text(format_latex_table(domain_name, results, solved, averages), encoding='utf-8')
  compile_latex_to_pdf(tex_path)

  return format_latex_table_fragment(domain_name, results, solved, averages)

def main_all_domains(binary_path, parent_folder, output_tex, threads, binary_args, search_prefix, timeout):
  run_folder = get_next_run_folder()
  print(f"Saving results to {run_folder}")

  sections = []
  for subfolder in sorted(Path(parent_folder).iterdir()):
    if subfolder.is_dir():
      sec = process_domain(binary_path, subfolder, threads, binary_args, search_prefix, timeout, run_folder)
      if sec:
        sections.append(sec)

  monolithic_path = run_folder / output_tex
  monolithic_path.write_text(
    "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
    + "\n\n".join(sections) + "\n\\end{document}\n",
    encoding='utf-8'
  )
  compile_latex_to_pdf(monolithic_path)
  print(f"\nAll LaTeX and CSV files generated.\nMonolithic LaTeX: {monolithic_path}")

def get_arg_parser():
  p = argparse.ArgumentParser(
    description="Run planner binary on all problem files in domain subfolders, gather results and generate LaTeX tables.",
    epilog="""Example usage:
  python scripts/coverage_run.py ./cmake-build-debug-nn/bin/deep ./exp/coverage \\
    --threads 2 \\
    --binary_args "-b -c -p 3" \\
    --search_prefix "Portfolio" \\
    --timeout 600
This will run the planner with 2 threads on all .txt and .eppdl files under ./exp/coverage/*,
passing '-b -c -p 3' to the binary, prefixing search names with 'Portfolio-', and timing out each run after 600 seconds.
""")
  p.add_argument("binary_path", help="Path to planner binary")
  p.add_argument("parent_folder", help="Parent folder containing domain subfolders")
  p.add_argument("--threads", type=int, default=4, help="Number of parallel threads (default: 4)")
  p.add_argument("--output_tex", default="results_tables.tex", help="Output monolithic LaTeX filename")
  p.add_argument("--binary_args", default="", help="Extra arguments to pass to the binary (single string)")
  p.add_argument("--search_prefix", default="", help="Prefix to prepend to search type in results")
  p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for each planner run (default: 30)")
  return p

def extract_portfolio_threads(binary_args: str) -> int:
  """Extract the value of --portfolio_threads or -p from binary_args."""
  long_match = re.search(r"--portfolio_threads\s+(\d+)", binary_args)
  short_match = re.search(r"-p\s+(\d+)", binary_args)
  if long_match:
    return int(long_match.group(1))
  elif short_match:
    return int(short_match.group(1))
  return 1  # default if not specified

if __name__ == "__main__":
  args = get_arg_parser().parse_args()

  portfolio_threads = extract_portfolio_threads(args.binary_args)
  total_threads_needed = args.threads * portfolio_threads
  available_threads = multiprocessing.cpu_count()
  reserved_threads = 1  # You can increase this if needed

  if total_threads_needed >= available_threads - reserved_threads:
    print(f"\nError: Requested total threads ({args.threads} × {portfolio_threads} = {total_threads_needed}) "
          f"exceeds available CPU threads ({available_threads}) minus reserved ({reserved_threads}).")
    print("Reduce --threads or (-p,--portfolio_threads) in binary_args.")
    sys.exit(1)

  main_all_domains(
    args.binary_path, args.parent_folder, args.output_tex, args.threads,
    args.binary_args, args.search_prefix, args.timeout
  )