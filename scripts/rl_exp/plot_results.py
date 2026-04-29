import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("combined_results/aggregate.csv")

# --- clean ---
df["NodesExpanded_mean"] = pd.to_numeric(df["NodesExpanded_mean"], errors="coerce")

# normalize Strict
df["Strict"] = df["Strict"].map({
    "strict": True,
    "nostrict": False,
    True: True,
    False: False
})

# --- combine heuristic + strict into one label ---
df["Approach"] = df["Heuristic"] + "-" + df["Strict"].astype(str)

out = Path("combined_results/analysis")
out.mkdir(exist_ok=True)

# --- PIVOT ---
pivot = df.pivot_table(
    index="Approach",
    columns="Fringe",
    values="NodesExpanded_mean",
    aggfunc="mean"
)

# sort nicely
pivot = pivot.sort_index()
pivot = pivot[sorted(pivot.columns)]

# --- PLOT ---
plt.figure(figsize=(10, 8), dpi=500)

plt.imshow(pivot, aspect="auto", cmap="plasma_r")

plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)

plt.colorbar(label="NodesExpanded_mean")

plt.title("Nodes Expanded Heatmap (All Approaches)")
plt.xlabel("Fringe")
plt.ylabel("Approach (Heuristic-Strict)")

# annotate values (optional but very useful)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if pd.notna(val):
            plt.text(j, i, int(val), ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.savefig(out / "heatmap_nodes_all.png")

print("[OK] unified heatmap generated")