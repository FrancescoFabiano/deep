#!/usr/bin/env python3
# scripts/tables_tex_management/1_combine_results_in_csv.py
from __future__ import annotations

import os
import re
import argparse
from typing import List, Tuple, Dict

import pandas as pd

# ─────────────────────────── LaTeX parsing helpers ───────────────────────────

SUBSECTION_RE = re.compile(
    r"\\subsection\*\{Combined\s*\(Training\s*\+\s*Test\)\}(.*?)(?=\\subsection\*|\Z)",
    re.DOTALL,
)
TABULAR_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)


def clean_cell(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\small\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\footnotesize\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\scriptsize\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathsf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = (
        s.replace("\\&", "&")
        .replace("\\_", "_")
        .replace("\\%", "%")
        .replace("\\#", "#")
        .replace("\\$", "$")
    )
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = s.replace(r"\pm", "±").replace(r"\,", " ").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_tabular_to_df(tabular_tex: str) -> pd.DataFrame:
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
    header_line = rows[0]
    headers = [clean_cell(x) for x in header_line[:-2].split("&")]
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


def extract_combined_tables(tex: str) -> List[pd.DataFrame]:
    out = []
    for block in SUBSECTION_RE.findall(tex):
        for tab in TABULAR_RE.findall(block):
            df = parse_tabular_to_df(tab)
            if not df.empty:
                out.append(df)
    return out


# ─────────────────────────── utilities ───────────────────────────

PL_TAIL_RE = re.compile(r".*_(\d+)$")


def _extract_pl(problem: str):
    if not isinstance(problem, str):
        return None
    m = PL_TAIL_RE.match(problem.strip())
    return int(m.group(1)) if m else None


def _ltr_underscore_natural_key(s: str):
    if not isinstance(s, str):
        return ()
    toks = [t for t in s.split("_") if t != ""]
    key = []
    for t in toks:
        if t.isdigit():
            key.append((1, int(t)))
        else:
            key.append((0, t.lower()))
    return tuple(key)


def _source_label_from_path(fp: str) -> str | None:
    p = fp.replace("\\", "/")
    pl = p.lower()
    if "/_results/astar_gnn/" in pl:
        return "AstarGNN"
    if "/_results/bfs/" in pl:
        return "BFS"
    if "/_results/hfs_gnn/" in pl:  # <-- NEW: map HFS_GNN
        return "HFS"
    if "/_results/portfolio_1/" in pl or "/_results/portfolio-1/" in pl:
        return "p6"
    if "/_results/portfolio/" in pl:
        return "p5"
    return None


# ─────────────────────────── normalize & combine ───────────────────────────

NEEDED_COLS = [
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


def normalize_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    if any(c not in df.columns for c in NEEDED_COLS):
        return pd.DataFrame(columns=NEEDED_COLS)
    # drop summary rows
    df = df[~df["Mode"].astype(str).str.contains(r"^Averages:", na=False)].copy()
    # trim
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # numeric coercion (keep NaN if not numeric)
    for c in [
        "Length",
        "Nodes",
        "Total (ms)",
        "Init (ms)",
        "Search (ms)",
        "Overhead (ms)",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[NEEDED_COLS].copy()


def combine_tables(tex_paths: List[str]) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    frames = []
    total_rows_read = 0
    per_file_counts: Dict[str, int] = {}

    for fp in tex_paths:
        file_rows = 0
        try:
            with open(fp, "r", encoding="utf-8") as f:
                tex = f.read()
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue

        label = _source_label_from_path(fp)
        tables = extract_combined_tables(tex)
        if not tables:
            print(f"[INFO] {fp}: no 'Combined (Training + Test)' tabulars found.")
        for tdf in tables:
            clean_df = normalize_and_filter(tdf)
            file_rows += len(clean_df)
            if not clean_df.empty:
                clean_df["__source_file__"] = os.path.basename(fp)
                clean_df["__source_label__"] = label
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

    # de-dup per (Mode,Domain,Problem,Search,__source_label__) keep last, stable sort
    before = len(long_df)
    long_df = long_df.sort_values(
        ["__prob_key__", "Mode", "Domain", "Search", "__source_label__"]
    ).drop_duplicates(
        ["Mode", "Domain", "Problem", "Search", "__source_label__"], keep="last"
    )
    after = len(long_df)
    if after < before:
        print(
            f"[INFO] Deduplicated combined rows: {before} -> {after} (-{before-after})"
        )

    return long_df, total_rows_read, per_file_counts


# ─────────────────────────── wide builder (per-file blocks) ───────────────────────────

METRICS = [
    "Goal",
    "Length",
    "Nodes",
    "Total (ms)",
    "Init (ms)",
    "Search (ms)",
    "Overhead (ms)",
]

ORDERED_LABELS = ["AstarGNN", "BFS", "HFS", "p5", "p6"]  # presentation order


def to_pivoted_csv(long_df: pd.DataFrame, output_csv: str) -> int:
    if long_df.empty:
        raise RuntimeError(
            "No valid 'Combined (Training + Test)' rows found in provided files."
        )

    # base unique keys
    if "pl" not in long_df.columns:
        long_df["pl"] = long_df["Problem"].map(_extract_pl)
    if "__prob_key__" not in long_df.columns:
        long_df["__prob_key__"] = long_df["Problem"].map(_ltr_underscore_natural_key)

    id_cols = ["Mode", "Domain", "Problem", "pl"]
    base = (
        long_df[id_cols + ["__prob_key__"]]
        .drop_duplicates()
        .sort_values(["__prob_key__", "Mode", "Domain"], kind="mergesort")
        .drop(columns="__prob_key__")
        .reset_index(drop=True)
    )

    wide = base.copy()

    present_labels = [
        lbl
        for lbl in ORDERED_LABELS
        if lbl in set(long_df["__source_label__"].dropna())
    ]

    for lbl in present_labels:
        sub = long_df[long_df["__source_label__"] == lbl].copy()
        if sub.empty:
            # still create empty columns for consistency
            for m in METRICS:
                wide[f"{m} {{{lbl}}}"] = pd.NA
            # p5/p6 extra Search {lbl}
            if lbl in ("p5", "p6"):
                wide[f"Search {{{lbl}}}"] = pd.NA
            continue

        # dedup per instance for this source
        sub = sub.sort_values(["Mode", "Domain", "Problem"]).drop_duplicates(
            ["Mode", "Domain", "Problem"], keep="last"
        )
        attach = sub[id_cols + METRICS + ["Search"]].copy()

        # rename metrics to "{lbl}"
        rename_map = {m: f"{m} {{{lbl}}}" for m in METRICS}
        # Only p5/p6 should expose Search {lbl}
        if lbl in ("p5", "p6"):
            rename_map["Search"] = f"Search {{{lbl}}}"
        else:
            attach = attach.drop(columns=["Search"])

        attach = attach.rename(columns=rename_map)
        wide = wide.merge(attach, on=id_cols, how="left")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    wide.to_csv(output_csv, index=False)
    print(
        f"[OK] Wrote CSV: {os.path.abspath(output_csv)}  (rows_written={len(wide)}, cols={len(wide.columns)})"
    )
    return len(wide)


# ─────────────────────────── CLI glue ───────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine LaTeX results into a CSV (per-file metric blocks + p5/p6 Search)."
    )
    p.add_argument("--experiment_set", default="gnn_exp", type=str)
    p.add_argument("--experiment_batch", required=True, type=str)
    p.add_argument("--out_dir_name", default="final_reports", type=str)
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Use only Astar_GNN and HFS_GNN inputs (legacy behavior).",
    )
    return p.parse_args()


def main(tex_paths: List[str], output_csv: str):
    checked = [fp for fp in tex_paths if os.path.exists(fp)]
    missing = [fp for fp in tex_paths if not os.path.exists(fp)]
    for fp in missing:
        print(f"[WARN] File not found: {fp}")
    if not checked:
        print("[ERROR] No valid files to process.")
        return

    long_df, total_rows_read, per_file_counts = combine_tables(checked)

    # Per-file counts
    for fp, n in per_file_counts.items():
        print(f"[INFO] {fp}: rows with results read = {n}")
    print(f"[INFO] Total rows with results read (all files): {total_rows_read}")

    if long_df.empty:
        print(
            "[ERROR] No rows survived normalization/filtering. CSV will NOT be written."
        )
        return

    rows_written = to_pivoted_csv(long_df, output_csv)
    print(f"[INFO] Rows with results written to CSV: {rows_written}")


if __name__ == "__main__":
    """
    EXAMPLE:
    python3 scripts/tables_tex_management/1_combine_results_in_csv.py \
      --experiment_set gnn_exp \
      --experiment_batch batch_test \
      --out_dir_name final_reports
    """
    args = parse_args()
    exp_dir = f"./exp/{args.experiment_set}/{args.experiment_batch}"
    out_dir = f"./exp/{args.experiment_set}/{args.out_dir_name}/{args.experiment_batch}"
    os.makedirs(out_dir, exist_ok=True)

    output_csv = f"{out_dir}/combined_results.csv"

    if args.ablation:
        tex_paths = [
            f"{exp_dir}/_results/Astar_GNN/monolithic_combined_results.tex",
            f"{exp_dir}/_results/HFS_GNN/monolithic_combined_results.tex",
        ]
    else:
        tex_paths = [
            f"{exp_dir}/_results/Astar_GNN/monolithic_combined_results.tex",
            f"{exp_dir}/_results/BFS/monolithic_combined_results.tex",
            f"{exp_dir}/_results/portfolio/monolithic_combined_results.tex",  # → p5
            f"{exp_dir}/_results/portfolio_1/monolithic_combined_results.tex",  # → p6
        ]
    main(tex_paths=tex_paths, output_csv=output_csv)
