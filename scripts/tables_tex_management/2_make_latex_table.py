#!/usr/bin/env python3
# scripts/tables_tex_management/2_make_latex_table.py
#
# Builds ONE longtable from combined_results.csv, aggregating all domains,
# with variable-width blocks per search:
#   - normal searches (e.g., A*(GNN), BFS, HFS): 3 cols (Len, Nodes, Time)
#   - p5/p6: 4 cols (Len, Nodes, Time, search)  <-- shows Search {p5}/{p6}
#
# Example:
#   python3 scripts/tables_tex_management/2_make_latex_table.py \
#     --csv ./exp/gnn_exp/final_reports/batch_test/combined_results.csv \
#     --mode Test \
#     --domain Grapevine \
#     --search "\GNNres=Astar_GNN" \
#     --search "\BFSres=BFS" \
#     --search "\Pfive=p5" \
#     --search "\Psix=p6"
from __future__ import annotations

import os
import re
import difflib
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────── PREAMBLE (inline) ───────────────────────────────
# If your main .tex already defines these, LaTeX will ignore duplicates.
PREAMBLE_TEX = r"""% ====== Auto-inlined preamble for the monolithic results table ======
\documentclass{article}
\usepackage[a4paper, landscape, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}

% --- Display labels/macros (single, non-conflicting definitions) ---
\newcommand{\hyperstyle}[1]{\ensuremath{\mathsf{#1}}}
\newcommand{\resstyle}[1]{\texttt{#1}}

\newcommand{\BFSres}{BFS}
\newcommand{\GNNres}{A*(GNN)}
\newcommand{\HFSres}{HFS} % not used in this table but left for completeness

\newcommand{\planLength}{Length.}
\newcommand{\nodesExp}{Nodes}
\newcommand{\solvingTime}{Time}
\newcommand{\unsolvedColumn}{--}
\newcommand{\myTO}{TO}

\newcommand{\myAvg}{avg}
\newcommand{\myStd}{std}
\newcommand{\IQM}{iqm}
\newcommand{\IQR}{iqr}
\newcommand{\allInstances}{all}
\newcommand{\onlyInCommon}{comm}

\newcommand{\Pfive}{P5}   % previously undefined in the source
\newcommand{\Psix}{P6}    % previously undefined in the source

% ====== End preamble ======

\begin{document}
"""


def _tail(path: Path, n=80) -> str:
    try:
        with path.open("r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"(could not read log: {e})"


def compile_tex(
    source_path: str,
    use_latexmk: bool = True,
    biblio: str | None = None,  # None | "bibtex" | "biber"
    shell_escape: bool = False,
    clean_aux: bool = False,
):
    src = Path(source_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    workdir = src.parent
    basename = src.stem  # e.g., "combined_results_Assemble_..."
    # NOTE: LaTeX can compile a file with any extension (even .txt) if it contains a proper preamble.

    pdf_path = workdir / f"{basename}.pdf"
    log_path = workdir / f"{basename}.log"
    shell_flag = ["-shell-escape"] if shell_escape else []

    def run(cmd):
        return subprocess.run(
            cmd, cwd=workdir, check=True, capture_output=True, text=True
        )

    try:
        if use_latexmk and shutil.which("latexmk"):
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", *shell_flag, src.name]
            run(cmd)
        else:
            # manual pipeline
            run(["pdflatex", "-interaction=nonstopmode", *shell_flag, src.name])
            if biblio == "bibtex":
                # bibtex expects AUX basename (no extension)
                run(["bibtex", basename])
            elif biblio == "biber":
                run(["biber", basename])
            # second pass (and a third to settle refs if biblio used)
            run(["pdflatex", "-interaction=nonstopmode", *shell_flag, src.name])
            if biblio:
                run(["pdflatex", "-interaction=nonstopmode", *shell_flag, src.name])

        if not pdf_path.exists():
            raise RuntimeError(
                "LaTeX completed but PDF not found. Check the log for details."
            )

        if clean_aux:
            for ext in (
                ".aux",
                ".bbl",
                ".blg",
                ".bcf",
                ".run.xml",
                ".out",
                ".toc",
                ".log",
                ".fls",
                ".fdb_latexmk",
                ".synctex.gz",
            ):
                p = workdir / f"{basename}{ext}"
                if p.exists():
                    p.unlink()

        print(f"✅ PDF generated: {pdf_path}")
        return pdf_path

    except subprocess.CalledProcessError as e:
        # Show a concise error plus tail of the .log (or stderr if no log yet)
        print("❌ LaTeX compilation failed.")
        if log_path.exists():
            print(
                "\n─── Log tail ─────────────────────────────────────────────────────────"
            )
            print(_tail(log_path, 80))
            print(
                "───────────────────────────────────────────────────────────────────────"
            )
        else:
            # If LaTeX died before writing a log, show stderr
            print(
                "\n─── Compiler stderr ───────────────────────────────────────────────────"
            )
            print(e.stderr)
            print(
                "───────────────────────────────────────────────────────────────────────"
            )
        # Bubble up a clearer error for callers
        raise RuntimeError(
            f"LaTeX failed; see {log_path if log_path.exists() else 'stderr above'}"
        ) from e


# ───────────────────────────── VERBOSITY ─────────────────────────────
def vinfo(msg: str):
    print(f"[INFO] {msg}")


def vwarn(msg: str):
    print(f"[WARN] {msg}")


def vok(msg: str):
    print(f"[OK] {msg}")


# ────────────────────────── SORT / PRETTY ───────────────────────────
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


def _pretty_instance_name(problem: str) -> str:
    return (
        ""
        if not isinstance(problem, str)
        else problem.replace("__pl_", "-pl_").replace("_", "\_")
    )


# ───────────────────────────── STATS ────────────────────────────────
def _percentile(x: np.ndarray, q: List[float]) -> Tuple[float, float]:
    try:
        return tuple(np.percentile(x, q, method="linear"))
    except TypeError:
        return tuple(np.percentile(x, q, interpolation="linear"))


def _iqm(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    x = np.sort(values.astype(float))
    q1, q3 = _percentile(x, [25, 75])
    mid = x[(x >= q1) & (x <= q3)]
    return float(np.mean(mid if mid.size else x))


def _iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    q1, q3 = _percentile(values.astype(float), [25, 75])
    return float(q3 - q1)


def _avg_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    return (float(np.mean(values)), float(np.std(values, ddof=0)))


def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NaN"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.0f}"


def _fmt_stat_pair(mu: float, sigma: float) -> str:
    if np.isnan(mu) or np.isnan(sigma):
        return "NaN $\\pm$ NaN"
    return f"{_fmt_num(mu)} $\\pm$ {_fmt_num(sigma)}"


# ───────────────────────── COLUMN HELPERS ───────────────────────────
def discover_search_labels(df: pd.DataFrame) -> List[str]:
    return [
        col[len("Goal {") : -1]
        for col in df.columns
        if col.startswith("Goal {") and col.endswith("}")
    ]


def _normalize_label_core(s: str) -> str:
    s = s.lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"\*{2,}", "*", s)
    return s


def _normalize_label_all(s: str) -> List[str]:
    base = s.strip()
    variants = set()
    queue = {base}
    for _ in range(2):
        new_items = set()
        for t in queue:
            new_items.add(t)
            t1 = re.sub(r"star", "*", t, flags=re.IGNORECASE)
            t2 = re.sub(r"\*", "star", t1)
            new_items.add(t1)
            new_items.add(t2)
        queue = new_items
    more = set()
    for t in queue:
        more.add(t)
        more.add(t.replace("(GNN)", "GNN"))
        more.add(t.replace("GNN", "(GNN)"))
        more.add(t.replace("(gnn)", "gnn"))
        more.add(t.replace("gnn", "(gnn)"))
    queue |= more
    for t in queue:
        core = _normalize_label_core(t)
        keep_star = re.sub(r"[^a-z0-9\*]+", "", core)
        no_star = keep_star.replace("*", "")
        variants.add(keep_star)
        variants.add(no_star)
        variants.add(keep_star.replace("a*", "astar"))
        variants.add(keep_star.replace("astar", "a*"))
    ordered = sorted(variants, key=lambda x: (-len(x), x))
    return ordered


def resolve_search_label(df: pd.DataFrame, requested: str) -> str:
    available = discover_search_labels(df)
    if not available:
        raise RuntimeError(
            "No 'Goal {…}' columns found in CSV. Check your combined_results.csv."
        )
    vinfo(f"Requested search label: '{requested}'")
    vinfo(f"Discovered labels in CSV ({len(available)}): {sorted(available)}")
    req_clean = requested.strip()
    if req_clean in available:
        vok(f"Using exact match for '{req_clean}'.")
        return req_clean
    lowers = {a.lower(): a for a in available}
    if req_clean.lower() in lowers:
        chosen = lowers[req_clean.lower()]
        vok(f"Case-insensitive match: '{req_clean}' → '{chosen}'.")
        return chosen
    req_norms = _normalize_label_all(req_clean)
    avail_norm_map = {}
    for a in available:
        for norm in _normalize_label_all(a):
            avail_norm_map.setdefault(norm, set()).add(a)
    for rn in req_norms:
        if rn in avail_norm_map:
            candidates = sorted(avail_norm_map[rn], key=lambda x: (-len(x), x))
            chosen = candidates[0]
            vok(f"Normalized match: '{req_clean}' → '{chosen}' via key '{rn}'")
            return chosen
    cand = difflib.get_close_matches(req_clean, available, n=1, cutoff=0.0)
    if cand:
        chosen = cand[0]
        vwarn(f"Fuzzy match: '{req_clean}' → '{chosen}'.")
        return chosen
    raise RuntimeError(
        f"Requested search '{req_clean}' not found. Available: {sorted(available)}"
    )


def _cols(search: str) -> dict:
    """
    For every resolved CSV label 'search', map the needed columns.
    If search in {p5, p6}, also expose the extra 'Search {search}' column.
    """
    out = {
        "goal": f"Goal {{{search}}}",
        "length": f"Length {{{search}}}",
        "nodes": f"Nodes {{{search}}}",
        "time": f"Total (ms) {{{search}}}",  # or "Search (ms)" if you prefer
    }
    if search in {"p5", "p6"}:
        out["search_str"] = f"Search {{{search}}}"  # extra visible column for p5/p6
    else:
        out["search_str"] = None
    return out


# ───────────────────────────── CORE BUILD ───────────────────────────
def _collect(df: pd.DataFrame, cols: dict, mask: pd.Series):
    L = pd.to_numeric(df.loc[mask, cols["length"]], errors="coerce").to_numpy()
    N = pd.to_numeric(df.loc[mask, cols["nodes"]], errors="coerce").to_numpy()
    T = pd.to_numeric(df.loc[mask, cols["time"]], errors="coerce").to_numpy()
    L = L[~np.isnan(L)]
    N = N[~np.isnan(N)]
    T = T[~np.isnan(T)]
    return L, N, T


def _block_width(cols: dict) -> int:
    """3 metrics (Len, Nodes, Time) + 1 extra if p5/p6 has search_str."""
    return 3 + (1 if cols.get("search_str") else 0)


def _build_header_blocks(colmaps: Dict[str, dict]) -> Tuple[str, str, int, str]:
    """
    Returns:
      first_hdr, second_hdr, cline_end, colspec
    Header adapts per macro width: 3 for normal, 4 for p5/p6 (adds 'search').
    """
    first = r"\multirow2{*}{\textbf{Instance Name}}"
    second_cells = []
    colspec_parts = ["l"]
    total_data_cols = 0

    macros = list(colmaps.keys())
    for i, macro in enumerate(macros):
        cols = colmaps[macro]
        w = _block_width(cols)
        # right bar between macro blocks
        bar = "|" if i < len(macros) - 1 else ""
        first += f" & \\multicolumn{{{w}}}{{c{bar}}}{{{macro}}}"
        # second row labels for this macro
        second_cells.append(r"\planLength & \nodesExp & \solvingTime [ms]")
        if cols.get("search_str"):
            second_cells.append("search")  # plain text header for p5/p6 extra col
        # colspec
        colspec_parts.append("|")
        colspec_parts.append("c" * w)
        total_data_cols += w

    first_hdr = first
    second_hdr = "& " + " & ".join(second_cells)
    cline_end = 1 + total_data_cols
    colspec = "".join(colspec_parts)
    return first_hdr, second_hdr, cline_end, colspec


def build_table_body_and_stats(
    df: pd.DataFrame, colmaps: Dict[str, dict]
) -> Tuple[List[str], List[str]]:
    body_rows = []
    for _, r in df.iterrows():
        row_cells = [_pretty_instance_name(r["Problem"])]
        for cols in colmaps.values():
            goal_ok = str(r.get(cols["goal"], "")) == "Yes"
            if goal_ok:
                l = _fmt_num(r.get(cols["length"]))
                n = _fmt_num(r.get(cols["nodes"]))
                t = _fmt_num(r.get(cols["time"]))
            else:
                l = n = r"\unsolvedColumn"
                t = r"\myTO"
            row_cells += [l, n, t]
            # extra visible 'search' cell for p5/p6
            if cols.get("search_str"):
                s_val = r.get(cols["search_str"], "")
                row_cells.append(str(s_val) if pd.notna(s_val) else "")
        body_rows.append(" & ".join(row_cells) + r" \\")

    # STATS / FOOTER
    masks = {}
    metrics_all = {}
    solved_counts = {}
    for macro, cols in colmaps.items():
        m = df[cols["goal"]].astype(str).eq("Yes")
        masks[macro] = m
        solved_counts[macro] = int(m.sum())
        metrics_all[macro] = _collect(df, cols, m)
    total_instances = len(df)

    # Common solved mask across all macros
    common_mask = None
    for macro in colmaps.keys():
        common_mask = (
            masks[macro] if common_mask is None else (common_mask & masks[macro])
        )

    metrics_common = {
        macro: _collect(df, cols, common_mask) for macro, cols in colmaps.items()
    }

    def stats_pack(triple):
        L, N, T = triple
        return (
            _avg_std(L),
            _avg_std(N),
            _avg_std(T),
            _iqm(L),
            _iqm(N),
            _iqm(T),
            _iqr(L),
            _iqr(N),
            _iqr(T),
        )

    def fmt_block(st):
        (avgL, avgN, avgT, iqmL, iqmN, iqmT, iqrL, iqrN, iqrT) = st
        avg = f"{_fmt_stat_pair(*avgL)} & {_fmt_stat_pair(*avgN)} & {_fmt_stat_pair(*avgT)}"
        iqm = f"{_fmt_num(iqmL)} $\\pm$ {_fmt_num(iqrL)} & {_fmt_num(iqmN)} $\\pm$ {_fmt_num(iqrN)} & {_fmt_num(iqmT)} $\\pm$ {_fmt_num(iqrT)}"
        return avg, iqm

    footer = []

    # avg±std (all)
    line = r"\myAvg  $\pm$ \myStd \hfill (\allInstances)"
    for _, cols in colmaps.items():
        avg_all, _ = fmt_block(stats_pack(metrics_all[_]))
        # (we can't index metrics_all by cols; recompute in loop properly)
    footer.clear()
    # rebuild lines properly:
    line = r"\myAvg  $\pm$ \myStd \hfill (\allInstances)"
    for macro, cols in colmaps.items():
        avg_all, _ = fmt_block(stats_pack(metrics_all[macro]))
        line += " & " + avg_all
        if cols.get("search_str"):
            line += " & "  # pad the extra p5/p6 column
    footer.append(line + r" \\")

    line = r"\IQM $\pm$ \IQR \hfill (\allInstances)"
    for macro, cols in colmaps.items():
        _, iqm_all = fmt_block(stats_pack(metrics_all[macro]))
        line += " & " + iqm_all
        if cols.get("search_str"):
            line += " & "
    footer.append(line + r" \\")

    line = r"\myAvg  $\pm$ \myStd \hfill (\onlyInCommon)"
    for macro, cols in colmaps.items():
        avg_com, _ = fmt_block(stats_pack(metrics_common[macro]))
        line += " & " + avg_com
        if cols.get("search_str"):
            line += " & "
    footer.append(line + r" \\")

    line = r"\IQM $\pm$ \IQR \hfill (\onlyInCommon)"
    for macro, cols in colmaps.items():
        _, iqm_com = fmt_block(stats_pack(metrics_common[macro]))
        line += " & " + iqm_com
        if cols.get("search_str"):
            line += " & "
    footer.append(line + r" \\")

    # Solved Instances — span over each macro width
    solved = r"Solved Instances"
    macros = list(colmaps.keys())
    for i, (macro, cols) in enumerate(colmaps.items()):
        cnt = solved_counts[macro]
        pct = f"{(100.0*cnt/total_instances):.2f}\\%"
        bar = "|" if i < len(macros) - 1 else ""
        width = _block_width(cols)
        solved += (
            rf" & \multicolumn{{{width}}}{{c{bar}}}{{{cnt}/{total_instances} ({pct})}}"
        )
    footer.append(solved)

    return body_rows, footer


def render_longtable_block(
    colspec: str,
    first_hdr: str,
    second_hdr: str,
    cline_end: int,
    body_rows: List[str],
    footer_rows: List[str],
) -> str:
    lines = []
    lines.append(r"\begin{longtable}[!ht]{" + colspec + r"}")
    lines.append(r"\centering")
    lines.append(first_hdr + r" \\")
    lines.append(rf"\cline{{2-{cline_end}}}")
    lines.append(second_hdr + r" \\")
    lines.append(r"\hline")
    lines.extend(body_rows)
    lines.append(r"\hline")
    lines.extend(footer_rows)
    return "\n".join(lines)


# ───────────────────────── ARGPARSE / MAIN ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Make ONE longtable LaTeX comparison (aggregating ALL domains) from combined_results.csv (with inlined preamble)."
    )
    p.add_argument("--csv", required=True, help="Path to combined_results.csv")
    p.add_argument(
        "--mode",
        required=True,
        choices=["Training", "Test", "Training and Test"],
        help="Which rows to include (aggregates ALL domains).",
    )
    p.add_argument(
        "--domain",
        default="",
        help="Domain name placeholder used in section title and caption (e.g., 'Grapevine' or 'ALL DOMAINS').",
    )
    p.add_argument(
        "--search",
        action="append",
        required=True,
        help=r"Repeatable: header macro and requested label, e.g. --search '\GNNres=Astar_GNN' (use '\Pfive=p5' and/or '\Psix=p6' to show the extra search column).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the .txt (default: next to CSV)",
    )
    p.add_argument(
        "--out-path", default=None, help="Full output .txt path (overrides --out-dir)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    CSV_PATH = args.csv
    MODE = args.mode
    DOMAIN = args.domain

    # Parse searches
    SEARCHES: Dict[str, str] = {}
    for item in args.search:
        if "=" not in item:
            raise SystemExit(f"[ERROR] --search expects 'Macro=Label', got: {item}")
        macro, label = item.split("=", 1)
        macro = macro.strip()
        label = label.strip()
        if not macro or not label:
            raise SystemExit(f"[ERROR] Invalid --search entry: {item}")
        SEARCHES[macro] = label

    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"[ERROR] CSV not found: {CSV_PATH}")

    vinfo(f"Reading CSV: {CSV_PATH}")
    df_all = pd.read_csv(CSV_PATH)

    for c in ["Mode", "Problem", "Domain"]:
        if c not in df_all.columns:
            raise SystemExit(f"[ERROR] CSV must contain '{c}' column.")
    vinfo(f"Discovered search labels in CSV: {sorted(discover_search_labels(df_all))}")

    # Mode slice (aggregate all domains inside it)
    if MODE == "Training and Test":
        df_mode = df_all[df_all["Mode"].astype(str).isin(["Training", "Test"])].copy()
    else:
        df_mode = df_all[df_all["Mode"].astype(str) == MODE].copy()

    if df_mode.empty:
        raise SystemExit(f"[ERROR] No rows with Mode='{MODE}' in CSV.")

    # Resolve searches
    resolved = {}
    colmaps = {}
    for macro, requested in SEARCHES.items():
        try:
            label_res = resolve_search_label(df_mode, requested)
        except RuntimeError as e:
            vwarn(str(e))
            continue
        resolved[macro] = label_res
        colmaps[macro] = _cols(label_res)
        # sanity: if p5/p6 requested, ensure extra col exists (warn otherwise)
        if label_res in {"p5", "p6"}:
            extra = colmaps[macro]["search_str"]
            if extra not in df_mode.columns:
                vwarn(
                    f"CSV missing expected column '{extra}' for '{label_res}'. That extra 'search' cell will be empty."
                )

    if not colmaps:
        raise SystemExit(
            "[ERROR] None of the requested searches were found in the CSV columns."
        )

    vinfo(
        "Resolved search labels: "
        + ", ".join([f"{m}:{r}" for m, r in resolved.items()])
    )

    # Global sort by Problem
    df = df_mode.copy()
    df["__prob_key__"] = df["Problem"].map(_ltr_underscore_natural_key)
    df = df.sort_values(["__prob_key__"], kind="mergesort").drop(columns="__prob_key__")

    # Build headers/body/footer with variable widths and render
    first_hdr, second_hdr, cline_end, colspec = _build_header_blocks(colmaps)
    body_rows, footer_rows = build_table_body_and_stats(df, colmaps)

    # Section title (before the table)
    section_line = r"\section*{Monolithic Results}"

    # Table
    table_block = render_longtable_block(
        colspec, first_hdr, second_hdr, cline_end, body_rows, footer_rows
    )

    # Derive batch_name from CSV path (directory containing the CSV file)
    batch_name = os.path.basename(os.path.dirname(os.path.abspath(CSV_PATH)))

    # Caption + label AFTER the table
    if MODE == "Test":
        caption = (
            r"Comparison of execution on the "
            + "{"
            + DOMAIN
            + "}"
            + r" domain over the \textbf{Test} instances. The model used by \GNNres has been "
            r"trained using the instances reported in the previous \textbf{Train} Table."
        )
        label = (
            r"tab:"
            + "{"
            + batch_name
            + "}"
            + "_"
            + "{"
            + DOMAIN
            + "}"
            + "_comparison_test"
        )
    elif MODE == "Training":
        caption = (
            r"Comparison of execution on the "
            + "{"
            + DOMAIN
            + "}"
            + r" domain over the \textbf{Train} instances. The model used by \GNNres has been "
            r"trained using the instances reported in this Table."
        )
        label = (
            r"tab:"
            + "{"
            + batch_name
            + "}"
            + "_"
            + "{"
            + DOMAIN
            + "}"
            + "_comparison_train"
        )
    else:
        caption = (
            r"Comparison of execution on the "
            + "{"
            + DOMAIN
            + "}"
            + r" domain over the \textbf{Train and Test} instances."
        )
        label = (
            r"tab:"
            + "{"
            + batch_name
            + "}"
            + "_"
            + "{"
            + DOMAIN
            + "}"
            + "_comparison_train_and_test"
        )

    # Assemble final .tex content
    parts = []
    parts.append(PREAMBLE_TEX.rstrip() + "\n")  # <-- inlined preamble
    parts.append(section_line)
    parts.append(table_block)
    parts.append(r"\\")
    parts.append(r"\caption{" + caption + r"}")
    parts.append(r"\label{" + label + r"}")
    parts.append(r"\end{longtable}")
    parts.append(r"\end{document}")
    final_tex = "\n".join(parts) + "\n"

    # Output path
    domains = sorted(set(df["Domain"].astype(str).dropna()))
    domains_joined = (
        "_".join(re.sub(r"[^A-Za-z0-9_-]+", "_", d) for d in domains)
        if domains
        else "ALLDOMAINS"
    )
    mode_sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", MODE)

    if args.out_path:
        out_path = args.out_path
        out_dir = os.path.dirname(out_path) or "."
    else:
        base = os.path.splitext(os.path.basename(CSV_PATH))[0]
        out_dir = args.out_dir or (os.path.dirname(CSV_PATH) or ".")
        out_path = os.path.join(
            out_dir, f"{base}_{domains_joined}_{mode_sanitized}.txt"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_tex)
    vok(f"Saved ONE LaTeX table: {out_path}")

    compile_tex(out_path)


if __name__ == "__main__":
    """
    python3 scripts/tables_tex_management/2_make_latex_table.py \
      --csv ./out/res/_tot_comparison/final_reports/combined_results/combined_results.csv \
      --mode Training and Test \
      --search "\GNNres=Astar_GNN" \
      --search "\BFSres=BFS" \
      --search "\Pfive=p5" \
      --search "\Psix=p6"
    """
    main()

    """
    sudo apt-get update
    sudo apt-get install texlive-latex-extra
    """
