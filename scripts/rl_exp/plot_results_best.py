import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df = pd.read_csv("combined_results/aggregate.csv")

# --- clean ---
df["NodesExpanded_mean"] = pd.to_numeric(df["NodesExpanded_mean"], errors="coerce")
df["Solved"] = pd.to_numeric(df["Solved"], errors="coerce")
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

# normalize strict
df["Strict"] = df["Strict"].map({
    "strict": True,
    "nostrict": False,
    True: True,
    False: False
})

# --- labels ---
df["Approach"] = df["Heuristic"] + "-" + df["Strict"].map({
    True: "strict",
    False: "nostrict"
})

# solved %
df["SolvedPct"] = (df["Solved"] / df["Total"]) * 100

# split
df["Split"] = df["Config"].apply(
    lambda x: "train" if x.startswith("train") else "test"
)

out = Path("combined_results/analysis")
out.mkdir(exist_ok=True)


# --------------------------------------------------
# 🔥 SELECT TOP 10 PER DATASET
# --------------------------------------------------

def get_top(data):
    ranking = (
        data.groupby("Approach")
        .apply(lambda g: pd.Series({
            "Solved": g["Solved"].sum(),
            "Total": g["Total"].sum(),
            "NodesExpanded_mean": np.average(
                g["NodesExpanded_mean"],
                weights=g["Total"]
            ),
            "SolvedPct": (g["Solved"].sum() / g["Total"].sum()) * 100
        }), include_groups=False)
        .reset_index()
    )

    ranking = ranking.sort_values(
        by=["Solved", "NodesExpanded_mean"],
        ascending=[False, True]
    )

    top = ranking.head(8)

    # --- ALWAYS include BFS ---
    bfs = ranking[ranking["Approach"].str.contains("BFS")]
    top = pd.concat([top, bfs]).drop_duplicates(subset="Approach")

    return top


# --------------------------------------------------
# 🎨 HEATMAP FUNCTION
# --------------------------------------------------

def plot_heatmap(data, name):
    top = get_top(data)

    # map solved % into label
    label_map = {
        row["Approach"]: f"{row['Approach']} ({row['SolvedPct']:.1f}%)"
        for _, row in top.iterrows()
    }

    df_top = data[data["Approach"].isin(top["Approach"])]

    pivot = df_top.pivot_table(
        index="Approach",
        columns="Fringe",
        values="NodesExpanded_mean",
        aggfunc="mean"
    )

    # ordering
    pivot = pivot.reindex(index=top["Approach"])
    pivot = pivot[sorted(pivot.columns)]

    # rename rows with % solved
    pivot.index = [label_map[a] for a in pivot.index]

    # --- plot ---
    plt.figure(figsize=(10, 8), dpi=500)
    im = plt.imshow(pivot, aspect="auto", cmap="plasma_r")

    plt.xticks(range(len(pivot.columns)), pivot.columns, fontsize=10)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)

    cbar = plt.colorbar(im)
    cbar.set_label("Nodes Expanded (mean)", fontsize=11)

    plt.title(f"Top Approaches – {name}", fontsize=14)
    plt.xlabel("Fringe", fontsize=12)
    plt.ylabel("Approach (Solved %)", fontsize=12)

    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                plt.text(j, i, int(val), ha="center", va="center", fontsize=6)
            else:
                plt.text(j, i, "-", ha="center", va="center", fontsize=6, alpha=0.5)

    plt.tight_layout()

    plt.savefig(out / f"heatmap_top10_{name}.png", dpi=300)
    plt.savefig(out / f"heatmap_top10_{name}.pdf")

    print(f"[OK] saved heatmap_top10_{name}")


# --------------------------------------------------
# 📊 GENERATE HEATMAPS
# --------------------------------------------------

plot_heatmap(df, "all")
plot_heatmap(df[df["Split"] == "train"], "train")
plot_heatmap(df[df["Split"] == "test"], "test")