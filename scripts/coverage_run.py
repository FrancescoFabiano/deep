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
    timeout_occurred = False
  except subprocess.TimeoutExpired as e:
    # Get whatever output was captured before timeout (if any)
    output = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
    timeout_occurred = True

  metrics = parse_output(output)

  if timeout_occurred or "Goal found :)" not in output:
    # Mark GoalFound as TO on timeout or if goal not found
    metrics["GoalFound"] = "TO"

  # Prepend search_prefix if given and SearchUsed is not "-"
  if metrics["SearchUsed"] != "-" and search_prefix:
    metrics["SearchUsed"] = f"{search_prefix}-{metrics['SearchUsed']}"
  else:
    metrics["SearchUsed"] = f"{metrics['SearchUsed']}"

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
  total = len(rows)
  averages = {}
  for col in NUMERIC_COLUMNS:
    vals = [int(r[col]) for r in rows if r[col].isdigit()]
    averages[col] = round(mean(vals), 2) if vals else "-"
  return solved, total, averages

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

  solved, total, averages = compute_summary(results)
  timeouts = sum(r["GoalFound"] == "TO" for r in results)
  print(f"{domain_name}: Solved = {solved}, Timeouts = {timeouts}, Total = {total}")

  # Save standalone LaTeX and compile
  tex_path = out_folder / f"{domain_name}_table.tex"
  tex_path.write_text(format_latex_table(domain_name, results, solved, averages), encoding='utf-8')
  compile_latex_to_pdf(tex_path)

  return format_latex_table_fragment(domain_name, results, solved, averages)

def format_combined_table(all_results, search_used_label):
  lines = [
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage[margin=1in,landscape]{geometry}",  # landscape mode
    "\\begin{document}",
    "\\section*{Combined Results (All Domains)}",
    f"\\textbf{{Search used:}} {latex_escape_underscores(search_used_label)} \\\\",
    "\\\\[0.3cm]",
    "\\begin{tabular}{llcccccccc}",  # one extra column for solved ratio
    "\\toprule",
    "Domain & Problem & Goal & Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) & Solved Ratio \\\\",
    "\\midrule"
  ]

  for domain_name, rows, solved, total, averages, timeouts in all_results:
    domain_esc = latex_escape_underscores(domain_name)
    solved_ratio = f"{solved}/{total} = {round(solved/total*100, 2)}\\%" if total > 0 else "-"
    for r in sorted(rows, key=lambda r: r["FileName"]):
      lines.append(" & ".join([
        domain_esc,
        latex_escape_underscores(r["FileName"]),
        r["GoalFound"], r["PlanLength"], r["NodesExpanded"], r["TotalExecutionTime"],
        r["InitTime"], r["SearchTime"], r["ThreadOverhead"],
        solved_ratio
      ]) + " \\\\")
  lines += ["\\bottomrule", "\\end{tabular}", "\\end{document}"]
  return "\n".join(lines)

def save_combined_csv(all_results, output_path: Path):
  combined_path = output_path / "all_results_combined.csv"
  with open(combined_path, "w", newline="") as f:
    fieldnames = ["Domain", "FileName", "GoalFound", "PlanLength", "NodesExpanded", "TotalExecutionTime",
                  "InitTime", "SearchTime", "ThreadOverhead", "SearchUsed"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for domain_name, rows, _, _, _, _ in all_results:
      for row in rows:
        row_out = {key: row[key] for key in fieldnames if key != "Domain"}
        row_out["Domain"] = domain_name
        writer.writerow(row_out)

def save_summary_csv(all_results, output_path: Path):
  summary_path = output_path / "summary_averages.csv"
  with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Domain", "SolvedRatio"] + NUMERIC_COLUMNS
    writer.writerow(header)

    total_vals = {col: [] for col in NUMERIC_COLUMNS}
    total_solved = 0
    total_problems = 0

    for domain_name, _, solved, total, averages, _ in all_results:
      solved_ratio = f"{solved}/{total} = {round(solved/total*100, 2)}%" if total > 0 else "-"
      row = [domain_name, solved_ratio]
      for col in NUMERIC_COLUMNS:
        val = averages[col]
        if isinstance(val, (int, float)):
          row.append(val)
          total_vals[col].append(val)
        else:
          row.append("-")
      writer.writerow(row)
      total_solved += solved
      total_problems += total

    total_avg = [round(mean(total_vals[col]), 2) if total_vals[col] else "-" for col in NUMERIC_COLUMNS]
    total_ratio = f"{total_solved}/{total_problems} = {round(total_solved/total_problems*100, 2)}%" if total_problems > 0 else "-"
    writer.writerow(["Total Average", total_ratio] + total_avg)

def reorganize_files(run_folder: Path, all_results):
  for domain_name, _, _, _, _, _ in all_results:
    subdir = run_folder / domain_name
    subdir.mkdir(exist_ok=True)

    # Move CSV
    csv_file = run_folder / f"{domain_name}_results.csv"
    if csv_file.exists():
      csv_file.rename(subdir / csv_file.name)

    # Move TeX
    tex_file = run_folder / f"{domain_name}_table.tex"
    if tex_file.exists():
      tex_file.rename(subdir / tex_file.name)

    # Move PDF (generated by pdflatex)
    pdf_file = run_folder / f"{domain_name}_table.pdf"
    if pdf_file.exists():
      pdf_file.rename(subdir / pdf_file.name)


def export_csvs(run_folder: Path, all_results):
  save_combined_csv(all_results, run_folder)
  save_summary_csv(all_results, run_folder)
  reorganize_files(run_folder, all_results)


def main_all_domains(binary_path, parent_folder, output_tex, threads, binary_args, search_prefix, timeout):
  run_folder = get_next_run_folder()
  print(f"Saving results to {run_folder}")

  all_results = []
  sections = []

  for subfolder in sorted(Path(parent_folder).iterdir()):
    if subfolder.is_dir():
      res = process_domain(binary_path, subfolder, threads, binary_args, search_prefix, timeout, run_folder)
      if res:
        domain_name = Path(subfolder).name
        rows = list(csv.DictReader(open(run_folder / f"{domain_name}_results.csv")))
        solved, total, averages = compute_summary(rows)
        timeouts = sum(r["GoalFound"] == "TO" for r in rows)
        all_results.append((domain_name, rows, solved, total, averages, timeouts))
        sections.append(res)

  monolithic_path = run_folder / output_tex
  monolithic_path.write_text(
    "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
    + "\n\n".join(sections) + "\n\\end{document}\n",
    encoding='utf-8'
  )
  compile_latex_to_pdf(monolithic_path)

  # Combined LaTeX + summary tables
  if args.search_prefix:
    search_used_label = args.search_prefix
  else:
  # fallback: first domain's first instance search used or "-"
    search_used_label = "-"
    if all_results:
      first_domain_rows = all_results[0][1]
      if first_domain_rows:
        search_used_label = first_domain_rows[0].get("SearchUsed", "-")

  combined_table = format_combined_table(all_results, search_used_label)
  combined_path = run_folder / "all_results_combined.tex"
  combined_path.write_text(combined_table, encoding='utf-8')
  compile_latex_to_pdf(combined_path)

  summary_table = format_summary_table(all_results, search_used_label)
  summary_path = run_folder / "summary_averages.tex"
  summary_path.write_text(summary_table, encoding='utf-8')
  compile_latex_to_pdf(summary_path)

  # Export CSVs and organize
  export_csvs(run_folder, all_results)

  print(f"\nAll results exported to {run_folder}")
  print(f"LaTeX Tables:\n  ├─ {monolithic_path}\n  ├─ {combined_path}\n  └─ {summary_path}")
  print(f"CSV Files:\n  ├─ all_results_combined.csv\n  └─ summary_averages.csv")


def format_summary_table(all_results,search_used_label):
  lines = [
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage[margin=1in]{geometry}",
    "\\begin{document}",
    "\\section*{Domain Summary Averages}",
    f"\\textbf{{Search used:}} {latex_escape_underscores(search_used_label)} \\\\",
    "\\\\[0.3cm]",
    "\\begin{tabular}{lccccccc}",
    "\\toprule",
    "Domain & Solved Ratio & Length & Nodes & Total (ms) & Init (ms) & Search (ms) & Overhead (ms) \\\\",
    "\\midrule"
  ]

  total_vals = {col: [] for col in NUMERIC_COLUMNS}
  total_solved = 0
  total_problems = 0

  for domain_name, _, solved, total, averages, _ in all_results:
    domain_esc = latex_escape_underscores(domain_name)
    solved_ratio = f"{solved}/{total} = {round(solved/total*100, 2)}\\%" if total > 0 else "-"
    line_vals = []
    for col in NUMERIC_COLUMNS:
      val = averages[col]
      if isinstance(val, (int, float)):
        line_vals.append(str(val))
        total_vals[col].append(val)
      else:
        line_vals.append("-")
    lines.append(f"{domain_esc} & {solved_ratio} & " + " & ".join(line_vals) + " \\\\")
    total_solved += solved
    total_problems += total

  total_avg = [
    f"{round(mean(total_vals[col]), 2) if total_vals[col] else '-'}"
    for col in NUMERIC_COLUMNS
  ]
  total_ratio = f"{total_solved}/{total_problems} = {round(total_solved/total_problems*100, 2)}\\%" if total_problems > 0 else "-"

  lines += [
    "\\midrule",
    f"\\textbf{{Total Average}} & {total_ratio} & " + " & ".join(total_avg) + " \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{document}"
  ]
  return "\n".join(lines)


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
  p.add_argument("binary", help="Path to planner binary executable")
  p.add_argument("parent_folder", help="Parent folder containing domain subfolders with problem files")
  p.add_argument("--threads", type=int, default=4, help="Number of parallel threads")
  p.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds")
  p.add_argument("--binary_args", default="", help="Additional command line arguments for binary")
  p.add_argument("--search_prefix", default="", help="Prefix for search names in output")
  p.add_argument("--output_tex", default="all_domains_summary.tex", help="Filename for combined LaTeX summary")
  return p


if __name__ == "__main__":
  args = get_arg_parser().parse_args()
  main_all_domains(args.binary, args.parent_folder, args.output_tex, args.threads,
                   args.binary_args, args.search_prefix, args.timeout)
