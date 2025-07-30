import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import iqr
import seaborn as sns

sns.set(style="whitegrid")

def extract_pl_value_unescaped(problem_name):
    problem_name_clean = problem_name.replace("\\_", "_")
    match = re.search(r"__pl_(\d+)", problem_name_clean)
    return int(match.group(1)) if match else None

def compute_iqm_iqrstd(values):
    values = np.sort(values)
    q1, q3 = np.percentile(values, [25, 75])
    mid_vals = values[(values >= q1) & (values <= q3)]
    if len(mid_vals) == 0:
        return np.nan, np.nan
    iqm = np.mean(mid_vals)
    iqr_std = iqr(mid_vals) / 2
    return iqm, iqr_std

def plot_iqm_fill_by_pl(df, algo_name, color):
    """Group‐by‐pl plotting."""
    x_vals, iqm_vals, std_vals = [], [], []
    for pl in sorted(df['pl'].unique()):
        nodes = df.loc[df['pl'] == pl, 'Nodes']
        if nodes.empty:
            continue
        iqm, std = compute_iqm_iqrstd(nodes)
        if np.isnan(iqm):
            continue
        x_vals.append(pl)
        iqm_vals.append(iqm)
        std_vals.append(std)

    x = np.array(x_vals)
    y = np.array(iqm_vals)
    s = np.array(std_vals)

    plt.plot(x, y, marker='o', color=color, label=algo_name)
    plt.fill_between(x, y - s, y + s, alpha=0.2, color=color)

def plot_iqm_overall(series, algo_name, color, xpos):
    """Single‐point IQM±std plotting."""
    vals = series.dropna().values
    if vals.size == 0:
        return
    if vals.size < 4:
        iqm = vals.mean()
        std = vals.std(ddof=0)
    else:
        iqm, std = compute_iqm_iqrstd(vals)
    plt.errorbar(xpos, iqm, yerr=std,
                 fmt='o', capsize=5,
                 color=color,
                 label=algo_name)

def parse_latex_table_from_text(tex_text, n):
    rows = []

    if n == 3:
        for line in tex_text.splitlines():
            line = line.strip()
            if line.startswith("\\") or line == "" or "±" in line or "Avg" in line or "IQM" in line:
                continue
            if "&" in line and "\\" in line:
                parts = [part.strip().replace("\\\\", "") for part in line.split("&")]
                if len(parts) != 11:
                    continue
                problem = parts[0]
                try:
                    row = [
                        problem,
                        float(parts[2]) if parts[2] != "nan" else np.nan,  # Nodes_astar
                        float(parts[5]) if parts[5] != "nan" else np.nan,  # Nodes_bfs
                        float(parts[8]) if parts[8] != "nan" else np.nan,  # Nodes_p5
                        parts[10].strip()
                    ]
                    rows.append(row)
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["Problem", "Nodes_astar", "Nodes_bfs", "Nodes_p5", "Search"])
    elif n == 2:
        for line in tex_text.splitlines():
            line = line.strip()
            if line.startswith("\\") or line == "" or "±" in line or "Avg" in line or "IQM" in line:
                continue
            if "&" in line and "\\" in line:
                parts = [part.strip().replace("\\\\", "") for part in line.split("&")]
                if len(parts) != 7:
                    continue
                problem = parts[0]
                try:
                    row = [
                        problem,
                        float(parts[2]) if parts[2] != "nan" else np.nan,  # Nodes_astar
                        float(parts[5]) if parts[5] != "nan" else np.nan,  # Nodes_bfs
                    ]
                    rows.append(row)
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["Problem", "Nodes_astar", "Nodes_bfs"])

def main_plot_scaling(tex_path,
                      n:int=2,
                      title:str="comparison",
                      figname:str="fig_scalability_comparison.pdf"):
    # 1) Read & parse
    with open(tex_path, "r") as f:
        tex = f.read()
    df = parse_latex_table_from_text(tex, n)

    # 2) Detect any __pl_<num> in 'Problem'
    df['pl'] = df['Problem'].apply(extract_pl_value_unescaped)
    has_pl = df['pl'].notnull().any()

    # 3) Prepare colors
    colors = {'Astar_GNN': 'blue', 'BFS': 'orange', 'Best_P5': 'red'}

    plt.figure(figsize=(5,3), dpi=500)

    if has_pl:
        # — per‐pl plotting path —
        df = df.dropna(subset=['pl'])
        # build per-algo DataFrames
        astar_df = (df[['pl','Nodes_astar']]
                    .rename(columns={'Nodes_astar':'Nodes'})
                    .dropna())
        astar_df['Algo'] = 'Astar_GNN'

        bfs_df   = (df[['pl','Nodes_bfs']]
                    .rename(columns={'Nodes_bfs':'Nodes'})
                    .dropna())
        bfs_df['Algo']   = 'BFS'

        # optional third algorithm
        p5_df = None
        if n == 3 and 'Nodes_p5' in df.columns:
            p5_df = (df[['pl','Nodes_p5']]
                     .rename(columns={'Nodes_p5':'Nodes'})
                     .dropna())
            p5_df['Algo'] = 'Best_P5'

        # Plot each
        plot_iqm_fill_by_pl(astar_df, 'Astar_GNN', colors['Astar_GNN'])
        plot_iqm_fill_by_pl(bfs_df,      'BFS',       colors['BFS'])
        if p5_df is not None:
            plot_iqm_fill_by_pl(p5_df,   'Best_P5',  colors['Best_P5'])

        plt.xlabel("Plan Length")

    else:
        # — overall plotting path —
        # we assume your parser has already dropped summary rows (±, Avg, IQM)
        astar = df['Nodes_astar']
        bfs   = df['Nodes_bfs']

        plot_iqm_overall(astar, 'Astar_GNN', colors['Astar_GNN'], xpos=0)
        plot_iqm_overall(bfs,   'BFS',       colors['BFS'],       xpos=1)

        plt.xticks([0,1], ['Astar_GNN','BFS'])
        plt.xlabel("Algorithm")

    # 4) Common finishing touches
    plt.ylabel("#Expanded Nodes")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

if __name__ == "__main__":
    # Change this to the path to your .tex file
    tex_path = "../../exp/aaai26/final_results_ok/batch5/CC-CoinBox_comparison_test.tex"
    main_plot_scaling(tex_path, n=3)
