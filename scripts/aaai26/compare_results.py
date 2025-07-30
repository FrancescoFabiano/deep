import os

import pandas as pd
import numpy as np
import re



def extract_table_data(tex_content: str, section_name: str, include_search: bool = False):
    """
    Extracts data from a specific section like 'Training' or 'Test', optionally including the 'Search' column.

    Args:
        tex_content (str): Full LaTeX file content.
        section_name (str): Section title to extract (e.g., "Training", "Test").
        include_search (bool): Whether to extract the 'Search' column as well.

    Returns:
        pd.DataFrame: Columns = Problem, Length, Nodes, Total (ms), [Search]
    """
    # Extract only the content inside \subsection*{<section_name>}
    pattern = re.compile(rf"\\subsection\*{{{re.escape(section_name)}.*?}}(.*?)\\subsection\*", re.DOTALL)
    match = pattern.search(tex_content + "\\subsection*{END}")  # Add dummy endpoint
    if not match:
        columns = ["Problem", "Length", "Nodes", "Total (ms)"] + (["Search"] if include_search else [])
        return pd.DataFrame(columns=columns)

    section = match.group(1)
    data = []

    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") or line.startswith("Problem") or "Averages:" in line:
            continue

        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 8:
            continue

        problem = parts[0]
        length = parts[2]
        nodes = parts[3]
        total = parts[4]
        search = parts[8] if include_search and len(parts) > 8 else None

        # Convert "-" to NaN
        length_val = int(length) if length != "-" else np.nan
        nodes_val = int(nodes) if nodes != "-" else np.nan
        total_val = int(total) if total != "-" else np.nan

        row = [problem, length_val, nodes_val, total_val]
        if include_search:
            row.append(search.strip() if search else "--")

        data.append(row)

    columns = ["Problem", "Length", "Nodes", "Total (ms)"]
    if include_search:
        columns.append("Search")

    return pd.DataFrame(data, columns=columns)


def safe_stat(mean, std):
    return f"{round(mean):d} ± --" if np.isnan(std) else f"{round(mean):d} ± {round(std):d}"


def compute_stats(df, col):
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return "--"
    return safe_stat(vals.mean(), vals.std())


def compute_iqm(df, col):
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return "--"
    q25 = vals.quantile(0.25)
    q75 = vals.quantile(0.75)
    iqm_vals = vals[(vals >= q25) & (vals <= q75)]
    if len(iqm_vals) <= 1:
        return compute_stats(df, col)
    return safe_stat(iqm_vals.mean(), iqm_vals.std())


def get_common_solved_problems(dfs):
    """
    Return the set of Problem IDs for which every DataFrame in `dfs`
    has non-null entries in Length, Nodes, and Total (ms).
    """
    numeric = ["Length", "Nodes", "Total (ms)"]
    common = set(dfs[0].dropna(subset=numeric)["Problem"])
    for df in dfs[1:]:
        common &= set(df.dropna(subset=numeric)["Problem"])
    return common


def merge_and_generate_latex(
        df_astar: pd.DataFrame,
        df_bfs: pd.DataFrame,
        custom_name: str,
        output_path: str,
        df_custom: pd.DataFrame = None,
        label_tab: str = None,
        caption: str = "Comparison"
):
    # 1️⃣ build the set of problems solved by all
    if df_custom is not None:
        common_solved = get_common_solved_problems([df_astar, df_bfs, df_custom])
    else:
        common_solved = get_common_solved_problems([df_astar, df_bfs])

    # 2️⃣ merge just for printing rows
    merged = pd.merge(
        df_astar, df_bfs, on="Problem",
        suffixes=(" (Astar_GNN)", " (BFS)")
    )
    if df_custom is not None:
        merged = pd.merge(
            merged, df_custom, on="Problem",
            suffixes=("", f" ({custom_name})")
        )
    merged = merged.sort_values("Problem")

    # 3️⃣ which algs we have
    algs = ["Astar_GNN", "BFS"]
    if df_custom is not None:
        algs.append(custom_name)

    with open(output_path, "w") as f:
        # -- header --
        f.write(r"\begin{table}[!ht]" + "\n")
        f.write(r"\centering" + "\n")


        if df_custom is not None:
            f.write(r"\scriptsize" + "\n")
            # Full three-column header
            f.write(r"\begin{tabular}{l|ccc|ccc|cccc}" + "\n")
            f.write(
                rf"\multirow{{2}}{{*}}{{\textbf{{{"Problem".replace('_', r'\_')}}}}} "
                r"& \multicolumn{3}{c|}{\textbf{Astar\_GNN}} "
                r"& \multicolumn{3}{c|}{\textbf{BFS}} "
                rf"& \multicolumn{{4}}{{c}}{{\textbf{{{custom_name.replace('_', r'\_')}}}}} \\" + "\n"
            )
            f.write(r"\cline{2-11}" + "\n")
            f.write(
                r"& Length & Nodes & Total (ms) "
                r"& Length & Nodes & Total (ms) "
                r"& Length & Nodes & Total (ms) & Search \\" + "\n"
            )
        else:
            f.write(r"\footnotesize" + "\n")
            f.write(r"\begin{tabular}{l|ccc|ccc}" + "\n")
            f.write(
                rf"\multirow{2}{{*}}{{\textbf{{{"Problem".replace('_', r'\_')}}}}} "
                r"& \multicolumn{3}{c|}{\textbf{Astar\_GNN}} "
                r"& \multicolumn{3}{c}{\textbf{BFS}} " r"\\" + "\n"
            )
            f.write(r"\cline{2-7}" + "\n")
            f.write(
                r"& Length & Nodes & Total (ms) "
                r"& Length & Nodes & Total (ms) " r"\\" + "\n"
            )

        f.write(r"\hline" + "\n")

        # -- data rows --
        for _, row in merged.iterrows():
            prob = re.sub(r"\\_", "_", row["Problem"]).replace("_", r"\_")
            parts = [
                row["Length (Astar_GNN)"], row["Nodes (Astar_GNN)"], row["Total (ms) (Astar_GNN)"],
                row["Length (BFS)"],       row["Nodes (BFS)"],       row["Total (ms) (BFS)"]
            ]
            if df_custom is not None:
                parts += [
                    row["Length"], row["Nodes"], row["Total (ms)"],
                    str(row.get("Search", "-")).replace("_", r"\_")
                ]
                line = prob + " & " + " & ".join(str(x) for x in parts)
            else:
                line = prob + " & " + " & ".join(str(x) for x in parts) + r" \\"
            f.write(line + "\n")

        f.write(r"\hline" + "\n")

        # -- FULL stats on raw dfs --
        for label, func in [("Avg ± Std (all)", compute_stats), ("IQM ± IQRStd (all)", compute_iqm)]:
            f.write(rf"\textbf{{{label}}}")
            for alg in algs:
                src = {"Astar_GNN": df_astar, "BFS": df_bfs, custom_name: df_custom}[alg]
                for col in ["Length", "Nodes", "Total (ms)"]:
                    f.write(" & " + func(src, col))
                if alg == custom_name:
                    f.write(" & --")
            f.write(" \\\\\n")

        # -- COMMON stats on raw dfs (filtered) --
        for label, func in [
            ("Avg ± Std (com)", compute_stats),
            ("IQM ± IQRStd (com)", compute_iqm)
        ]:
            f.write(rf"\textbf{{{label}}}")
            for alg in algs:
                src = {"Astar_GNN": df_astar, "BFS": df_bfs, custom_name: df_custom}[alg]
                filt = src[src["Problem"].isin(common_solved)].dropna(subset=["Length","Nodes","Total (ms)"])
                for col in ["Length", "Nodes", "Total (ms)"]:
                    f.write(" & " + func(filt, col))
                if alg == custom_name:
                    f.write(" & --")
            f.write(" \\\\\n")

        # -- footer --
        f.write(r"\end{tabular}" + "\n")
        f.write(rf"\caption{{{custom_name.replace("_", "-")}}}" + "\n")
        if label_tab:
            f.write(rf"\label{{tab:{label_tab}}}" + "\n")
        f.write(r"\end{table}" + "\n")



def run_comparison(astar_path, bfs_path, custom_name, output_path, custom_path:str=None,section="Training", label:str= "comparison"):
    with open(astar_path) as f: astar_tex = f.read()
    with open(bfs_path) as f: bfs_tex = f.read()
    if custom_path is not None:
        with open(custom_path) as f: custom_tex = f.read()

    df_astar = extract_table_data(astar_tex, section, include_search=False)
    df_bfs = extract_table_data(bfs_tex, section, include_search=False)
    if custom_path is not None:
        df_custom = extract_table_data(custom_tex, section, include_search=True)

    if custom_path is not None:
        assert df_astar is not None and df_bfs is not None and df_custom is not None, "Missing section data"
    else:
        assert df_astar is not None and df_bfs is not None, "Missing section data"

    if custom_path is not None:
        merge_and_generate_latex(df_astar, df_bfs, custom_name, output_path, df_custom=df_custom, label_tab=label, caption=f"Comparison {section}")
    else:
        merge_and_generate_latex(df_astar, df_bfs, custom_name, output_path, label_tab=label, caption=f"Comparison {section}")

def main(main_dir:str= "exp/aaai26", file_res:str="exp/aaai26/final_results_ok"):
    batches = {
        "batch1": ["Assemble", "CC", "CoinBox", "Grapevine", "SC"],
        "batch2": ["SC_Four_Rooms"],
        "batch3": ["CC_2_2_4__pl_7", "CC_3_2_3__pl_6", "CoinBox_pl__3", "CoinBox_pl__5"],
        "batch4":["SCRich"],
        "batch5": ["All", "CC-CoinBox", "CC-CoinBox-Grapevine", "CC-Grapevine"],
        "batch5_partial": ["CC-CoinBox", "CC-CoinBox-Grapevine", "CC-Grapevine"],
    }

    os.makedirs(file_res, exist_ok=True)

    for batch_n, list_of_domains in batches.items():

        for domain in list_of_domains:

            a_star_gnn_file = f"{main_dir}/{batch_n}/_results/Astar_GNN/{domain}/{domain}_combined.tex"
            bfs_file = f"{main_dir}/{batch_n}/_results/BFS/{domain}/{domain}_combined.tex"

            if batch_n == "batch5" or batch_n == "batch5_partial":
                p5_file = f"{main_dir}/{batch_n}_p5/_results/portfolio/{domain}/{domain}_combined.tex"
            else:
                p5_file = None

            output_file_train = f"{file_res}/{batch_n}/{domain}_comparison_train.tex"
            output_file_test = f"{file_res}/{batch_n}/{domain}_comparison_test.tex"

            os.makedirs(os.path.dirname(output_file_train), exist_ok=True)
            os.makedirs(os.path.dirname(output_file_test), exist_ok=True)

            problem_name_train = f"{batch_n}-{domain}-Train"
            problem_name_test = f"{batch_n}-{domain}-Test"

            caption_train = f"{batch_n}_{domain}_comparison_train"
            caption_test = f"{batch_n}_{domain}_comparison_test"

            run_comparison(a_star_gnn_file, bfs_file, problem_name_train, output_file_train, p5_file, "Train", caption_train)
            run_comparison(a_star_gnn_file, bfs_file, problem_name_test, output_file_test, p5_file, "Test", caption_test)


if __name__ == "__main__":
    main(main_dir="../../exp/aaai26", file_res="../../exp/aaai26/reports")