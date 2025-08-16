#!/usr/bin/env python3
import re
from textwrap import dedent

# ← EDIT THIS LIST: each tuple is (train_tex, test_tex, DomainName)
DOMAIN_TRIPLES = [
    ("../../exp/gnn_exp/final_results_ok/batch1/Assemble_comparison_train.tex",  "../../exp/gnn_exp/final_results_ok/batch1/Assemble_comparison_test.tex",  "Assemble"),
    ("../../exp/gnn_exp/final_results_ok/batch1/CC_comparison_train.tex",        "../../exp/gnn_exp/final_results_ok/batch1/CC_comparison_test.tex",        "CC"),
]

# Defaults: you can also change these if you like
CAPTION = "Batch\\#1 — IQM (com) node counts and percent reduction"
LABEL   = "tab:batch1_res"

OUTPUT_FILE = "out_b1.tex"

def parse_iqm_com(path):
    """
    Find the line containing 'IQM' and '(com)' in the .tex file and
    extract the LAST two integers on that line (A*GNN, BFS).
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'IQM' in line and '(com)' in line:
                nums = re.findall(r'(\d+)', line)
                if len(nums) < 2:
                    raise ValueError(f"Could not find two integers on IQM(com) line in {path!r}")
                # take the last two numbers
                return map(int, [nums[2], nums[8]])
    raise ValueError(f"No IQM(com) line found in {path!r}")

def pct_reduction(a, b):
    """Compute percentage reduction from b down to a."""
    return 0.0 if b == 0 else (b - a) / b * 100

def generate_latex_table(triples, caption, label, output_file):
    # collect rows
    rows = []
    for train_tex, test_tex, domain in triples:
        a_tr, b_tr = parse_iqm_com(train_tex)
        a_te, b_te = parse_iqm_com(test_tex)
        pr_tr = pct_reduction(a_tr, b_tr)
        pr_te = pct_reduction(a_te, b_te)
        rows.append((domain, a_tr, a_te, b_tr, b_te, pr_tr, pr_te))

    # compute averages
    count      = len(rows)
    sum_a_tr   = sum(r[1] for r in rows)
    sum_a_te   = sum(r[2] for r in rows)
    sum_b_tr   = sum(r[3] for r in rows)
    sum_b_te   = sum(r[4] for r in rows)
    sum_pr_tr  = sum(r[5] for r in rows)
    sum_pr_te  = sum(r[6] for r in rows)
    avg_row = (
        "Average",
        sum_a_tr / count,
        sum_a_te / count,
        sum_b_tr / count,
        sum_b_te / count,
        sum_pr_tr / count,
        sum_pr_te / count
    )

    # build LaTeX lines
    out = [dedent(r"""
        \begin{table}[!ht]
          \small
          \centering
          \begin{tabular}{c|cc|cc|cc}
            \textbf{Domain}
              & \multicolumn{2}{c|}{\textbf{A*GNN}}
              & \multicolumn{2}{c|}{\textbf{BFS}}
              & \multicolumn{2}{c}{\textbf{\% Reduction}} \\
            \cline{2-7}
            & Train & Test & Train & Test & Train & Test \\
            \hline
    """).lstrip()]

    for domain, a_tr, a_te, b_tr, b_te, pr_tr, pr_te in rows:
        safe_domain = domain.replace('_', '\\_')
        out.append(
            f"    {safe_domain} & "
            f"{int(a_tr):d} & {int(a_te):d} & "
            f"{int(b_tr):d} & {int(b_te):d} & "
            f"{int(pr_tr):d}\\% & {int(pr_te):d}\\% \\\\"
        )

    # add average separator and row
    out.append("    \\hline")
    out.append(
        f"    Average & "
        f"{int(avg_row[1]):d} & {int(avg_row[2]):d} & "
        f"{int(avg_row[3]):d} & {int(avg_row[4]):d} & "
        f"{int(avg_row[5]):d}\\% & {int(avg_row[6]):d}\\% \\\\"
    )

    out.append(dedent(f"""
          \\end{{tabular}}
          \\caption{{{caption}}}
          \\label{{{label}}}
        \\end{{table}}
    """))

    final_table = "\n".join(out)

    # write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_table)
    print(f"Wrote LaTeX table (with average row) to '{output_file}'")


if __name__ == "__main__":
    generate_latex_table(DOMAIN_TRIPLES, CAPTION, LABEL, OUTPUT_FILE)
