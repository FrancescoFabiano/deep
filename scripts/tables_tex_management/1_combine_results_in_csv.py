import os
import re
import pandas as pd
from typing import List
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


def normalize_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns of interest and valid rows.
    Required columns (exact names):
      Mode, Domain, Problem, Goal, Length, Nodes, Total (ms), Init (ms),
      Search (ms), Overhead (ms), Search
    Drop summary rows (e.g., 'Averages:') and invalid Search values ('', '-').
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

    # Valid search values
    df = df[df["Search"].notna() & (df["Search"] != "") & (df["Search"] != "-")]

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


def combine_tables(tex_paths: List[str]) -> pd.DataFrame:
    """
    Parse all given tex files and combine their 'Combined (Training + Test)' tables
    into a single long DataFrame with a source file column, numeric 'pl',
    and a left-to-right underscore-natural problem key for global ordering.
    """
    frames = []
    for fp in tex_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                tex = f.read()
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue

        for tdf in extract_combined_tables(tex):
            clean_df = normalize_and_filter(tdf)
            if not clean_df.empty:
                clean_df["__source_file__"] = os.path.basename(fp)
                clean_df["pl"] = clean_df["Problem"].map(_extract_pl)
                clean_df["__prob_key__"] = clean_df["Problem"].map(
                    _ltr_underscore_natural_key
                )
                frames.append(clean_df)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)

    # De-duplicate per key + Search, then **sort primarily by Problem key** (global),
    # so rows won't be block-grouped by Mode anymore.
    long_df = long_df.sort_values(
        ["__prob_key__", "Mode", "Domain", "Search"]
    ).drop_duplicates(["Mode", "Domain", "Problem", "Search"], keep="last")
    return long_df


def to_pivoted_csv(long_df: pd.DataFrame, output_csv: str):
    """
    Pivot the long_df so each Search becomes its own block of columns,
    include 'pl' before per-Search results, and order rows globally by Problem.
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

    # Keys (pl appears before metric blocks as requested)
    id_cols = ["Mode", "Domain", "Problem", "pl"]
    value_cols = [
        "Goal",
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
    ]

    wide = long_df.set_index(id_cols + ["Search"])[value_cols].unstack("Search")
    wide.columns = [f"{metric} {{{search}}}" for metric, search in wide.columns]
    wide = wide.reset_index()

    # Column order: keys first, then per-Search blocks
    search_names = sorted(
        {c.split("{", 1)[1].rstrip("}") for c in wide.columns if "{" in c}
    )
    metrics_order = [
        "Goal",
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
    ]
    ordered_cols = id_cols[:]
    for s in search_names:
        for m in metrics_order:
            col = f"{m} {{{s}}}"
            if col in wide.columns:
                ordered_cols.append(col)
    for c in wide.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    wide = wide[ordered_cols]

    # **Global Problem ordering** (natural, left-to-right on '_' tokens)
    wide["__prob_key__"] = wide["Problem"].map(_ltr_underscore_natural_key)
    wide = wide.sort_values(
        ["__prob_key__", "Mode", "Domain"], kind="mergesort"
    ).reset_index(drop=True)
    wide = wide.drop(columns="__prob_key__")

    wide.to_csv(output_csv, index=False)
    print(
        f"[OK] Wrote: {os.path.abspath(output_csv)}  (rows={len(wide)}, cols={len(wide.columns)})"
    )


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

    long_df = combine_tables(checked_paths)
    to_pivoted_csv(long_df, output_csv)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or build data for the distance estimator model."
    )
    parser.add_argument(
        "--experiment_set",
        default="gnn_exp",
        type=str,
        help="Name for the distance estimator model",
    )
    parser.add_argument(
        "--experiment_batch",
        type=str,
        help="Name for the distance estimator model",
    )
    parser.add_argument(
        "--out_dir_name",
        default="final_reports",
        type=str,
        help="Name for the distance estimator model",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """
    EXAMPLE OF USAGE:
    python3 scripts/tables_tex_management/1_combine_results_in_csv.py --experiment_set "gnn_exp" --experiment_batch "batch_test" --out_dir_name "final_reports"
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
            f"{exp_dir}/_results/BFS/monolithic_combined_results.tex",
        ],
        output_csv=f"{output_csv_dir}/combined_results.csv",
    )
