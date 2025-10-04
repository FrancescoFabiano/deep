#!/usr/bin/env python3
# scripts/tables_tex_management/1_combine_results_in_csv.py

import os
import re
import pandas as pd
from typing import List, Tuple
import argparse

# Regex to extract just the "Combined (Training + Test)" subsection
SUBSECTION_RE = re.compile(
    r"\\subsection\*\{Combined\s*\(Training\s*\+\s*Test\)\}(.*?)(?=\\subsection\*|\Z)",
    re.DOTALL,
)

# Regex to extract tabular environments
TABULAR_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)


def clean_cell(s: str) -> str:
    """Strip common LaTeX formatting from a cell."""
    s = s.strip()
    # unwrap simple formatting
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\small\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\footnotesize\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\scriptsize\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathsf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    # escaped characters
    s = (
        s.replace("\\&", "&")
        .replace("\\_", "_")
        .replace("\\%", "%")
        .replace("\\#", "#")
        .replace("\\$", "$")
    )
    # inline math
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    # common commands
    s = s.replace(r"\pm", "±").replace(r"\,", " ").replace("~", " ")
    # generic command remover
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


PL_TAIL_RE = re.compile(r".*_(\d+)$")


def _extract_pl(problem: str) -> int | None:
    if not isinstance(problem, str):
        return None
    m = PL_TAIL_RE.match(problem.strip())
    return int(m.group(1)) if m else None


def _ltr_underscore_natural_key(problem: str):
    """
    Tokenize by '_' (ignore empty tokens from double '__'),
    compare left-to-right with numeric tokens as ints and others as lowercase strings.
    """
    if not isinstance(problem, str):
        return ()
    toks = [t for t in problem.split("_") if t != ""]
    key = []
    for t in toks:
        if t.isdigit():
            key.append((1, int(t)))
        else:
            key.append((0, t.lower()))
    return tuple(key)


def parse_tabular_to_df(tabular_tex: str) -> pd.DataFrame:
    """Parse a LaTeX tabular environment into a DataFrame."""
    body = []
    for ln in tabular_tex.splitlines():
        ln = ln.strip()
        if (
            ln.startswith(r"\begin{tabular}")
            or ln.startswith(r"\end{tabular}")
            or ln == r"\hline"
        ):
            continue
        if ln:
            body.append(ln)

    # Reconstruct rows by looking for lines ending with '\\'
    rows, cur = [], ""
    for ln in body:
        cur += (" " + ln) if cur else ln
        if cur.endswith(r"\\"):
            rows.append(cur)
            cur = ""
    if cur:
        rows.append(cur)

    if not rows:
        return pd.DataFrame()

    # Header
    header_line = rows[0]
    headers = [clean_cell(x) for x in header_line[:-2].split("&")]  # drop trailing '\\'

    # Data
    data = []
    for ln in rows[1:]:
        if r"\hline" in ln or "&" not in ln or not ln.endswith(r"\\"):
            continue
        cells = [clean_cell(x) for x in ln[:-2].split("&")]
        if len(cells) == len(headers):
            data.append(cells)

    return (
        pd.DataFrame(data, columns=headers) if data else pd.DataFrame(columns=headers)
    )


def extract_combined_tables(tex: str) -> list[pd.DataFrame]:
    """From a full .tex file, extract all tables under the 'Combined (Training + Test)' subsection."""
    out = []
    for block in SUBSECTION_RE.findall(tex):
        for tabular_tex in TABULAR_RE.findall(block):
            df = parse_tabular_to_df(tabular_tex)
            if not df.empty:
                out.append(df)
    return out


def combine_tables(tex_paths: List[str]) -> Tuple[pd.DataFrame, int, dict]:
    """
    Parse all given tex files and combine their 'Combined (Training + Test)' tables
    into a single long DataFrame with a source file column, numeric 'pl',
    and a left-to-right underscore-natural problem key for global ordering.

    Returns:
      long_df, total_rows_read, per_file_rows_read (dict path->count)
    """
    frames = []
    total_rows_read = 0
    per_file_counts: dict[str, int] = {}

    for fp in tex_paths:
        file_rows = 0
        try:
            with open(fp, "r", encoding="utf-8") as f:
                tex = f.read()
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue

        tables = extract_combined_tables(tex)
        if not tables:
            print(f"[INFO] {fp}: no 'Combined (Training + Test)' tabulars found.")
        for tdf in tables:
            clean_df = normalize_and_filter(tdf)
            file_rows += len(clean_df)
            if not clean_df.empty:
                clean_df["__source_file__"] = os.path.basename(fp)
                clean_df["pl"] = clean_df["Problem"].map(_extract_pl)
                clean_df["__prob_key__"] = clean_df["Problem"].map(
                    _ltr_underscore_natural_key
                )
                frames.append(clean_df)

        per_file_counts[fp] = file_rows
        total_rows_read += file_rows
        print(f"[INFO] {fp}: rows with results read = {file_rows}")

    if not frames:
        return pd.DataFrame(), total_rows_read, per_file_counts

    long_df = pd.concat(frames, ignore_index=True)

    # De-duplicate per key + Search, then sort primarily by Problem key (global)
    before = len(long_df)
    long_df = long_df.sort_values(
        ["__prob_key__", "Mode", "Domain", "Search"]
    ).drop_duplicates(["Mode", "Domain", "Problem", "Search"], keep="last")
    after = len(long_df)
    if after < before:
        print(
            f"[INFO] Deduplicated combined rows: {before} -> {after} (-{before-after})"
        )

    return long_df, total_rows_read, per_file_counts


def normalize_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all rows (including those with Search='', '-'), so that problems
    unsolved by every search still appear in the final CSV.

    Required columns (exact names):
      Mode, Domain, Problem, Goal, Length, Nodes, Total (ms), Init (ms),
      Search (ms), Overhead (ms), Search
    Drop summary rows (e.g., 'Averages:').
    """
    needed = [
        "Mode",
        "Domain",
        "Problem",
        "Goal",
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
        "Search",
    ]
    if any(c not in df.columns for c in needed):
        # Header mismatch → skip this table
        return pd.DataFrame(columns=needed)

    # Remove summary rows
    df = df[~df["Mode"].astype(str).str.contains(r"^Averages:", na=False)].copy()

    # Trim whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # IMPORTANT CHANGE: do NOT filter out rows based on 'Search' anymore.
    # We keep even Search in {"", "-"} to preserve problems unsolved by all searches.

    # Numeric coercion
    num_cols = [
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[needed].copy()


def to_pivoted_csv(long_df: pd.DataFrame, output_csv: str) -> int:
    """
    Build a wide CSV that preserves ALL problems, even if no search solved them.
    Strategy:
      1) Build a base index of unique (Mode, Domain, Problem, pl).
      2) Discover valid search labels = unique Search values excluding "" and "-".
      3) For each search label, left-join its metrics onto the base.
    Returns: number of rows written.
    """
    if long_df.empty:
        raise RuntimeError(
            "No valid 'Combined (Training + Test)' rows found in provided files."
        )

    # Ensure helpers exist
    if "pl" not in long_df.columns:
        long_df["pl"] = long_df["Problem"].map(_extract_pl)
    if "__prob_key__" not in long_df.columns:
        long_df["__prob_key__"] = long_df["Problem"].map(_ltr_underscore_natural_key)

    # Base index: ALL unique instances we saw (even from rows with Search in {"","-"})
    id_cols = ["Mode", "Domain", "Problem", "pl"]
    base = (
        long_df[id_cols + ["__prob_key__"]]
        .drop_duplicates(id_cols + ["__prob_key__"])
        .sort_values(["__prob_key__", "Mode", "Domain"], kind="mergesort")
        .drop(columns="__prob_key__")
        .reset_index(drop=True)
    )

    # Valid search labels for column-block creation (ignore "" and "-")
    raw_searches = (
        long_df["Search"].astype(str).str.strip().replace({"nan": ""}).fillna("")
    )
    valid_searches = sorted({s for s in raw_searches.unique() if s not in ("", "-")})

    # Metrics to attach per search
    metric_cols = [
        "Goal",
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
    ]

    wide = base.copy()

    for s in valid_searches:
        sub = long_df[long_df["Search"].astype(str).str.strip() == s].copy()
        if sub.empty:
            # Nothing for this search; still create empty columns for consistency
            for m in metric_cols:
                wide[f"{m} {{{s}}}"] = pd.NA
            continue

        # Deduplicate per instance for this search (keep last)
        sub = sub.sort_values(["Mode", "Domain", "Problem"]).drop_duplicates(
            ["Mode", "Domain", "Problem"], keep="last"
        )

        attach = sub[id_cols + metric_cols].copy()
        # Rename metric columns with {search} suffix
        attach = attach.rename(columns={m: f"{m} {{{s}}}" for m in metric_cols})

        wide = wide.merge(attach, on=id_cols, how="left")

    # Save CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    wide.to_csv(output_csv, index=False)
    rows_written = len(wide)
    print(
        f"[OK] Wrote CSV: {os.path.abspath(output_csv)}  (rows_written={rows_written}, cols={len(wide.columns)})"
    )
    return rows_written


def _print_search_distribution(long_df: pd.DataFrame):
    print("[INFO] Rows per Search value (including empty/'-'):")
    counts = (
        long_df.groupby(long_df["Search"].fillna(""))
        .size()
        .sort_values(ascending=False)
    )
    for k, v in counts.items():
        disp = k if k not in ("",) else "<EMPTY>"
        print(f"        - {disp}: {v}")


def main(tex_paths: List[str], output_csv: str):
    checked_paths = []
    for fp in tex_paths:
        if not os.path.exists(fp):
            print(f"[WARN] File not found: {fp}")
        else:
            checked_paths.append(fp)

    if not checked_paths:
        print("[ERROR] No valid files to process.")
        return

    long_df, total_rows_read, per_file_counts = combine_tables(checked_paths)
    _print_search_distribution(long_df)  # <-- add this
    print(f"[INFO] Total rows with results read (all files): {total_rows_read}")

    if long_df.empty:
        print(
            "[ERROR] No rows survived normalization/filtering. CSV will NOT be written."
        )
        return

    rows_written = to_pivoted_csv(long_df, output_csv)
    print(f"[INFO] Rows with results written to CSV: {rows_written}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine LaTeX results into a single CSV (with counts)."
    )
    parser.add_argument(
        "--experiment_set",
        default="gnn_exp",
        type=str,
        help="Experiment set name",
    )
    parser.add_argument(
        "--experiment_batch",
        type=str,
        required=True,
        help="Experiment batch name (used to locate _results/*/monolithic_combined_results.tex)",
    )
    parser.add_argument(
        "--out_dir_name",
        default="final_reports",
        type=str,
        help="Directory under exp/<set>/<out_dir_name>/<batch> to place combined_results.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    EXAMPLE:
    python3 scripts/tables_tex_management/1_combine_results_in_csv.py \
      --experiment_set gnn_exp \
      --experiment_batch batch_test \
      --out_dir_name final_reports
    """
    args = parse_args()

    experiment_set = args.experiment_set
    experiment_batch = args.experiment_batch
    out_dir_name = args.out_dir_name

    exp_dir = f"./exp/{experiment_set}/{experiment_batch}"
    output_csv_dir = f"./exp/{experiment_set}/{out_dir_name}/{experiment_batch}"
    os.makedirs(output_csv_dir, exist_ok=True)

    main(
        tex_paths=[
            f"{exp_dir}/_results/Astar_GNN/monolithic_combined_results.tex",
            f"{exp_dir}/_results/HFS_GNN/monolithic_combined_results.tex",
        ],
        output_csv=f"{output_csv_dir}/combined_results.csv",
    )
