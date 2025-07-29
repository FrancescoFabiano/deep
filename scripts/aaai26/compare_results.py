import os
import re

import numpy as np
import pandas as pd

def extract_table_data(tex_content: str, section_name: str):
    """
    Extracts data from a specific section like 'Training' or 'Test'.
    """
    # Extract only the content inside \subsection*{<section_name>}
    pattern = re.compile(rf"\\subsection\*{{{re.escape(section_name)}.*?}}(.*?)\\subsection\*", re.DOTALL)
    match = pattern.search(tex_content + "\\subsection*{END}")  # Add dummy endpoint
    if not match:
        return pd.DataFrame(columns=["Problem", "Length", "Nodes", "Total (ms)"])

    section = match.group(1)
    data = []
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") or line.startswith("Problem") or "Averages:" in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 8:
            continue
        problem, goal, length, nodes, total = parts[0], parts[1], parts[2], parts[3], parts[4]
        if length != "-" and nodes != "-" and total != "-":
            try:
                data.append((problem, int(length), int(nodes), int(total)))
            except ValueError:
                continue
    return pd.DataFrame(data, columns=["Problem", "Length", "Nodes", "Total (ms)"])


def get_common_success(df, suffix1, suffix2):
    """Filter rows where both algorithms solved the problem."""
    solved_mask_1 = df[f"Total (ms){suffix1}"].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))
    solved_mask_2 = df[f"Total (ms){suffix2}"].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))
    return df[solved_mask_1 & solved_mask_2]

def safe_stat(mean, std):
    """Format mean ± std, replacing nan with '--'."""
    if np.isnan(std):
        return f"{mean:.2f} ± --"
    return f"{mean:.2f} ± {std:.2f}"


def compute_avg_std_safe(df, suffix=""):
    stats = {}
    for col in ["Length", "Nodes", "Total (ms)"]:
        col_name = f"{col}{suffix}"
        if col_name in df.columns:
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(values) > 0:
                stats[col] = safe_stat(values.mean(), values.std())
            else:
                stats[col] = "--"
        else:
            stats[col] = "--"
    return stats


def compute_iqm_iqmstd_safe(df, suffix=""):
    stats = {}
    for col in ["Length", "Nodes", "Total (ms)"]:
        col_name = f"{col}{suffix}"
        if col_name in df.columns:
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(values) > 0:
                q25 = values.quantile(0.25)
                q75 = values.quantile(0.75)
                iqm_values = values[(values >= q25) & (values <= q75)]
                if len(iqm_values) > 1:
                    stats[col] = safe_stat(iqm_values.mean(), iqm_values.std())
                else:
                    stats[col] = safe_stat(values.mean(), values.std())
            else:
                stats[col] = "--"
        else:
            stats[col] = "--"
    return stats


def append_summary_rows_to_latex(df, f, name1, name2):
    suffix1 = f" ({name1})"
    suffix2 = f" ({name2})"

    def write_stat_row(label, stats1, stats2):
        f.write(f"\\textbf{{{label}}} & {stats1['Length']} & {stats1['Nodes']} & {stats1['Total (ms)']} & "
                f"{stats2['Length']} & {stats2['Nodes']} & {stats2['Total (ms)']} \\\\\n")

    common_df = get_common_success(df, suffix1, suffix2)

    write_stat_row(f"Avg ± Std", compute_avg_std_safe(df, suffix1), compute_avg_std_safe(df, suffix2))
    write_stat_row(f"Avg ± Std***", compute_avg_std_safe(common_df, suffix1), compute_avg_std_safe(common_df, suffix2))
    write_stat_row(f"IQM ± IQMStd", compute_iqm_iqmstd_safe(df, suffix1), compute_iqm_iqmstd_safe(df, suffix2))
    write_stat_row(f"IQM ± IQMStd***", compute_iqm_iqmstd_safe(common_df, suffix1), compute_iqm_iqmstd_safe(common_df, suffix2))


def merge_and_generate_latex(df1, df2, name1="Astar_GNN", name2="BFS",
                             output_file="compare_results.tex",
                             label:str=None,
                             problem_name:str=None):
    # Normalize problem names for merging
    df1["Problem"] = df1["Problem"].str.strip().str.replace(r"\\_", "_", regex=True)
    df2["Problem"] = df2["Problem"].str.strip().str.replace(r"\\_", "_", regex=True)

    df = pd.merge(df1, df2, on="Problem", how="outer", suffixes=(f" ({name1})", f" ({name2})"))
    df = df.sort_values("Problem")

    p_name = "Problem" if problem_name is None else problem_name

    with open(output_file, "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l|ccc|ccc}\n")
        f.write("\\multirow{2}{*}{\\textbf{" + p_name.replace("_", r"\_") + "}} & "
                "\\multicolumn{3}{c|}{\\textbf{" + name1.replace("_", r"\_") + "}} & "
                "\\multicolumn{3}{c}{\\textbf{" + name2.replace("_", r"\_") + "}} \\\\\n")
        f.write("\cline{2-7}\n")
        f.write("& Length & Nodes & Total (ms) & Length & Nodes & Total (ms) \\\\\n")
        f.write("\\hline\n")

        for _, row in df.iterrows():
            # Escape underscores in problem names
            escaped_problem = row["Problem"].replace("_", r"\_")
            row_values = [escaped_problem]
            for suffix in [f" ({name1})", f" ({name2})"]:
                for col in ["Length", "Nodes", "Total (ms)"]:
                    val = row.get(f"{col}{suffix}", "-")
                    row_values.append(str(val) if pd.notna(val) else "-")
            f.write(" & ".join(row_values) + " \\\\\n")

        # Summary stats
        f.write("\\hline\n")
        append_summary_rows_to_latex(df, f, name1, name2)

        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison}\n")
        if label is not None:
            f.write(f"\\label{{tab:{label}}}\n")
        f.write("\\end{table}\n")

def main(main_dir:str= "exp/aaai26", file_res:str="exp/aaai26/final_results_ok"):
    batches = {"batch1": ["Assemble", "CC", "CoinBox", "Grapevine", "SC"],
               "batch2": ["SC_Four_Rooms"],
               "batch3": ["CC_2_2_4__pl_7", "CC_3_2_3__pl_6", "CoinBox_pl__3", "CoinBox_pl__5"],
               "batch4":["SCRich"]}

    os.makedirs(file_res, exist_ok=True)

    for batch_n, list_of_domains in batches.items():

        for domain in list_of_domains:

            a_star_gnn_file = f"{main_dir}/{batch_n}/_results/Astar_GNN/{domain}/{domain}_combined.tex"
            bfs_file = f"{main_dir}/{batch_n}/_results/BFS/{domain}/{domain}_combined.tex"

            with (open(a_star_gnn_file, "r") as f_astar, open(bfs_file, "r") as f_bfs):

                astar_content = f_astar.read()
                bfs_content = f_bfs.read()

            df_astar_train = extract_table_data(astar_content, "Training")
            df_bfs_train = extract_table_data(bfs_content, "Training")

            df_astar_test = extract_table_data(astar_content, "Test")
            df_bfs_test = extract_table_data(bfs_content, "Test")

            output_file_train = f"{file_res}/{batch_n}/{domain}_comparison_train.tex"
            output_file_test = f"{file_res}/{batch_n}/{domain}_comparison_test.tex"

            os.makedirs(os.path.dirname(output_file_train), exist_ok=True)
            os.makedirs(os.path.dirname(output_file_test), exist_ok=True)

            problem_name_train = f"{batch_n}-{domain}-Train"
            problem_name_test = f"{batch_n}-{domain}-Test"

            merge_and_generate_latex(df_astar_train, df_bfs_train,
                                     name1="Astar_GNN", name2="BFS",
                                     output_file=output_file_train,
                                     label=f"{batch_n}_{domain}_comparison_train",
                                     problem_name=problem_name_train)

            merge_and_generate_latex(df_astar_test, df_bfs_test,
                                     name1="Astar_GNN", name2="BFS",
                                     output_file=output_file_test,
                                     label=f"{batch_n}_{domain}_comparison_test",
                                     problem_name=problem_name_test)

if __name__ == "__main__":
    main()