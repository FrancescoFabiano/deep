import csv, sys
from pathlib import Path

# optional positional arg: results dir (already batch+domain scoped when called
# from pipeline.py, e.g. combined_results/batchX/CC). default = combined_results.
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("combined_results")
OUT = DIR / "aggregate.csv"


# ---------- PARSE STATS FROM MIXED CSV ----------
def extract_stats_row(csv_path: Path):
    """
    Extract stats section from summary.csv that contains:
    instance table + blank line + stats table
    """
    try:
        with open(csv_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        header_idx = None

        # find stats header (robust pattern)
        for i, line in enumerate(lines):
            if "Solved" in line and "Total" in line:
                header_idx = i
                break

        if header_idx is None or header_idx + 1 >= len(lines):
            print(f"[WARN] stats section not found: {csv_path}")
            return None

        headers = lines[header_idx].split(",")
        values = lines[header_idx + 1].split(",")

        if len(headers) != len(values):
            print(f"[WARN] malformed stats row: {csv_path}")
            return None

        row = dict(zip(headers, values))

        return row

    except Exception as e:
        print(f"[ERROR] failed parsing {csv_path}: {e}")
        return None


# ---------- METADATA EXTRACTION ----------
def extract_metadata(path: Path):
    name = path.stem  # e.g. train_RL-H-AVG_nostrict

    # --- Fringe ---
    fringe = next(
        (p.replace("fringe_", "") for p in path.parts if p.startswith("fringe_")),
        "0"
    )

    # --- Strict ---
    if "_nostrict" in name:
        strict = False
    elif "_strict" in name:
        strict = True
    else:
        strict = None

    # --- Heuristic ---
    core = name.replace("train_", "").replace("test_", "")
    core = core.replace("_strict", "").replace("_nostrict", "")

    if core == "BFS":
        heuristic = "BFS"
    elif core.startswith("RL-H-"):
        heuristic = core.replace("RL-H-", "")
    elif core.startswith("RL-"):
        heuristic = core.replace("RL-", "")
    else:
        heuristic = core

    return {
        "Heuristic": heuristic,
        "Fringe": int(fringe),
        "Strict": strict,
        "Config": name,
    }


# ---------- MAIN ----------
def main():
    rows = []

    for f in DIR.rglob("*.csv"):
        if f.name == "aggregate.csv":
            continue

        stats = extract_stats_row(f)
        if stats is None:
            continue

        meta = extract_metadata(f)

        combined = {**meta, **stats}
        rows.append(combined)

    if not rows:
        raise RuntimeError("No valid data found to aggregate")

    # --- normalize keys ---
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # preferred ordering
    priority = [
        "Config", "Heuristic", "Fringe", "Strict",
        "Solved", "Total"
    ]

    stat_keys = sorted(k for k in all_keys if k not in priority)
    fieldnames = priority + stat_keys

    # --- write ---
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    #print(f"[OK] Aggregated {len(rows)} files -> {OUT}")


if __name__ == "__main__":
    main()