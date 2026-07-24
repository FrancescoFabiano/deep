import sys
import pandas as pd
from pathlib import Path

# optional positional arg: results dir (batch+domain scoped when called from
# pipeline.py, e.g. combined_results/batchX/CC). default = combined_results.
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("combined_results")
df = pd.read_csv(RESULTS / "aggregate.csv")

# --- numeric cleanup ---
df["NodesExpanded_mean"] = pd.to_numeric(df["NodesExpanded_mean"], errors="coerce")
df["TotalExecutionTime_mean"] = pd.to_numeric(df["TotalExecutionTime_mean"], errors="coerce")

df["SolvedPct"] = df["Solved"] / df["Total"] * 100

out = RESULTS / "analysis"
out.mkdir(parents=True, exist_ok=True)

# --- BEST PER GROUP (min nodes) ---
best = (
    df.sort_values("NodesExpanded_mean")
    .groupby(["Fringe","Strict"])
    .first()
    .reset_index()
)

best.to_csv(out / "best_by_nodes.csv", index=False)

# --- FULL GROUPED STATS ---
grouped = (
    df.groupby(["Heuristic","Fringe","Strict"])
    .mean(numeric_only=True)
    .reset_index()
)

grouped.to_csv(out / "grouped_summary.csv", index=False)

#print("[OK] analysis complete")