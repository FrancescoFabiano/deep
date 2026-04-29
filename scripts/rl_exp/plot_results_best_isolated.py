import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --------------------------------------------------
# LOAD + CLEAN
# --------------------------------------------------

df = pd.read_csv("combined_results/aggregate.csv")

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

# labels
df["Approach"] = df["Heuristic"] + "-" + df["Strict"].map({
    True: "strict",
    False: "nostrict"
})

df["SolvedPct"] = (df["Solved"] / df["Total"]) * 100

# split
df["Split"] = df["Config"].apply(
    lambda x: "train" if x.startswith("train") else "test"
)

out = Path("combined_results/analysis")
out.mkdir(exist_ok=True)


# --------------------------------------------------
# TOP 10 SELECTION
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

    # always include BFS
    bfs = ranking[ranking["Approach"].str.contains("BFS")]
    top = pd.concat([top, bfs]).drop_duplicates(subset="Approach")

    return top


# --------------------------------------------------
# PLOT FUNCTION (BEST FRINGE ONLY)
# --------------------------------------------------

def plot_best_fringe(data, name):
    top = get_top(data)

    df_top = data[data["Approach"].isin(top["Approach"])]

    # select best fringe per approach
    best_rows = (
        df_top.groupby(["Approach", "Fringe"])
        .apply(lambda g: pd.Series({
            "NodesExpanded_mean": np.average(
                g["NodesExpanded_mean"],
                weights=g["Total"]
            ),
            "Solved": g["Solved"].sum(),
            "Total": g["Total"].sum(),
            "SolvedPct": (g["Solved"].sum() / g["Total"].sum()) * 100
        }), include_groups=False)
        .reset_index()
    )

    best_rows = (
        best_rows.sort_values(
            ["Solved", "NodesExpanded_mean"],
            ascending=[False, True]
        )
        .groupby("Approach")
        .first()
        .reset_index()
    )

    # build labels with % and fringe
    labels = []
    for _, row in best_rows.iterrows():
        label = f"{row['Approach']} ({row['SolvedPct']:.1f}%, f={int(row['Fringe'])})"
        labels.append(label)

    pivot = best_rows.set_index("Approach")[["NodesExpanded_mean"]]

    # preserve ordering
    pivot = pivot.reindex(best_rows["Approach"])
    pivot.index = labels

    # --------------------------------------------------
    # PLOT
    # --------------------------------------------------

    plt.figure(figsize=(10, 8), dpi=500)

    im = plt.imshow(pivot, aspect="auto", cmap="plasma_r")

    plt.xticks([0], ["Best Fringe"], fontsize=10)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)

    cbar = plt.colorbar(im)
    cbar.set_label("Nodes Expanded (mean)", fontsize=11)

    plt.title(f"Top Approaches (Best Fringe) – {name}", fontsize=14)
    plt.ylabel("Approach (Solved %, Fringe)", fontsize=12)

    # annotate values
    for i in range(pivot.shape[0]):
        val = pivot.iloc[i, 0]
        if not np.isnan(val):
            plt.text(0, i, int(val), ha="center", va="center", fontsize=7)

    plt.tight_layout()

    plt.savefig(out / f"best_fringe_{name}.png", dpi=300)
    plt.savefig(out / f"best_fringe_{name}.pdf")

    print(f"[OK] saved best_fringe_{name}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

plot_best_fringe(df, "all")
plot_best_fringe(df[df["Split"] == "train"], "train")
plot_best_fringe(df[df["Split"] == "test"], "test")