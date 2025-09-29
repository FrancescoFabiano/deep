import os
import re
import difflib
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def vinfo(msg: str):
    print(f"[INFO] {msg}")


def vwarn(msg: str):
    print(f"[WARN] {msg}")


def vok(msg: str):
    print(f"[OK] {msg}")


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
    return "" if not isinstance(problem, str) else problem.replace("__pl_", "-pl_")


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
    return {
        "goal": f"Goal {{{search}}}",
        "length": f"Length {{{search}}}",
        "nodes": f"Nodes {{{search}}}",
        "time": f"Total (ms) {{{search}}}",  # change to Search (ms) if needed
    }


def _collect(df: pd.DataFrame, cols: dict, mask: pd.Series):
    L = pd.to_numeric(df.loc[mask, cols["length"]], errors="coerce").to_numpy()
    N = pd.to_numeric(df.loc[mask, cols["nodes"]], errors="coerce").to_numpy()
    T = pd.to_numeric(df.loc[mask, cols["time"]], errors="coerce").to_numpy()
    L = L[~np.isnan(L)]
    N = N[~np.isnan(N)]
    T = T[~np.isnan(T)]
    return L, N, T


def build_table_multi(
    df_mode: pd.DataFrame, searches: Dict[str, str], caption: str, label: str
) -> str:
    if df_mode.empty:
        raise RuntimeError("No rows for the selected Mode in the CSV.")
    resolved = {}
    colmaps = {}
    for macro, requested in searches.items():
        try:
            label_res = resolve_search_label(df_mode, requested)
        except RuntimeError as e:
            vwarn(str(e))
            continue
        resolved[macro] = label_res
        colmaps[macro] = _cols(label_res)
    if not colmaps:
        raise RuntimeError(
            "None of the requested searches were found in the CSV columns."
        )
    vinfo(
        "Resolved search labels: "
        + ", ".join([f"{m}:{r}" for m, r in resolved.items()])
    )

    df = df_mode.copy()
    df["__prob_key__"] = df["Problem"].map(_ltr_underscore_natural_key)
    df = df.sort_values(["__prob_key__"], kind="mergesort").drop(columns="__prob_key__")

    colspec = "l|" + "|".join(["ccc"] * len(colmaps))
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{" + colspec + r"}")

    first_line = r"\multirow2{*}{\textbf{Instance Name}}"
    for i, macro in enumerate(colmaps.keys()):
        bar = "|" if i < len(colmaps) - 1 else ""
        first_line += f" & \\multicolumn{{3}}{{c{bar}}}{{{macro}}}"
    lines.append(first_line + r" \\")
    cline_end = 1 + 3 * len(colmaps)
    lines.append(rf"\cline{{2-{cline_end}}}")
    subhdr = r"& \planLength & \nodesExp & \solvingTime [ms]"
    for _ in list(colmaps.keys())[1:]:
        subhdr += r" & \planLength & \nodesExp & \solvingTime [ms]"
    lines.append(subhdr + r" \\")
    lines.append(r"\hline")

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
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\hline")

    masks = {}
    metrics_all = {}
    solved_counts = {}
    for macro, cols in colmaps.items():
        mask = df[cols["goal"]].astype(str).eq("Yes")
        masks[macro] = mask
        solved_counts[macro] = int(mask.sum())
        metrics_all[macro] = _collect(df, cols, mask)

    total_instances = len(df)
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

    line = r"\myAvg  $\pm$ \myStd \hfill (\allInstances)"
    for macro in colmaps.keys():
        avg_all, _ = fmt_block(stats_pack(metrics_all[macro]))
        line += " & " + avg_all
    lines.append(line + r" \\")
    line = r"\IQM $\pm$ \IQR \hfill (\allInstances)"
    for macro in colmaps.keys():
        _, iqm_all = fmt_block(stats_pack(metrics_all[macro]))
        line += " & " + iqm_all
    lines.append(line + r" \\")
    line = r"\myAvg  $\pm$ \myStd \hfill (\onlyInCommon)"
    for macro in colmaps.keys():
        avg_com, _ = fmt_block(stats_pack(metrics_common[macro]))
        line += " & " + avg_com
    lines.append(line + r" \\")
    line = r"\IQM $\pm$ \IQR \hfill (\onlyInCommon)"
    for macro in colmaps.keys():
        _, iqm_com = fmt_block(stats_pack(metrics_common[macro]))
        line += " & " + iqm_com
    lines.append(line + r" \\")
    lines.append(r"\hline")

    solved = r"Solved Instances"
    for i, macro in enumerate(colmaps.keys()):
        cnt = solved_counts[macro]
        pct = f"{(100.0*cnt/total_instances):.2f}\\%"
        bar = "|" if i < len(colmaps) - 1 else ""
        solved += rf" & \multicolumn{{3}}{{c{bar}}}{{{cnt}/{total_instances} ({pct})}}"
    lines.append(solved)

    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ───────────────────────── ARGPARSE / MAIN ─────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Make a single LaTeX comparison table from combined_results.csv"
    )
    p.add_argument("--csv", required=True, help="Path to combined_results.csv")
    # NEW: includes 'Train and Test'
    p.add_argument(
        "--mode",
        required=True,
        choices=["Train", "Test", "Train and Test"],
        help="Which rows to include (aggregates ALL domains).",
    )
    p.add_argument(
        "--search",
        action="append",
        required=True,
        help=r"Repeatable: header macro and requested label, e.g. --search '\GNNres=Astar_GNN'. "
        r"Use quotes to protect backslashes.",
    )
    p.add_argument(
        "--caption",
        default=None,
        help="Custom caption (optional). {MODE} will be expanded if present.",
    )
    p.add_argument("--label", default=None, help="Custom LaTeX label (optional).")
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
    MODE = args.mode  # can be "Train", "Test", or "Train and Test"

    # parse searches
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

    # Slice by mode (supporting "Train and Test")
    if MODE == "Train and Test":
        df_mode = df_all[df_all["Mode"].astype(str).isin(["Train", "Test"])].copy()
    else:
        df_mode = df_all[df_all["Mode"].astype(str) == MODE].copy()

    if df_mode.empty:
        raise SystemExit(f"[ERROR] No rows with Mode='{MODE}' in CSV.")

    # Caption / label (MODE token preserved exactly, including "Train and Test")
    CAPTION = args.caption or (
        f"Comparison over \\textbf{{{MODE}}} instances across all domains. "
        "The model used by \\GNNres has been trained using the instances reported "
        "in the previous \\textbf{Train} Table."
    )
    LABEL = args.label or f"tab:combined_comparison_{MODE.lower().replace(' ', '_')}"

    # Build table
    vinfo(
        f"Building SINGLE table for Mode='{MODE}' (aggregating all domains); "
        f"instances: {len(df_mode)}; searches: {list(SEARCHES.keys())}"
    )
    try:
        latex = build_table_multi(
            df_mode=df_mode, searches=SEARCHES, caption=CAPTION, label=LABEL
        )
    except RuntimeError as e:
        raise SystemExit(f"[ERROR] {e}")

    # ---- Output path logic (uses unique domains & mode) ----
    # Extract unique domains from the filtered data
    domains = sorted(set(df_mode["Domain"].astype(str).dropna()))
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
        f.write(latex)
    vok(f"Saved ONE LaTeX table: {out_path}")


if __name__ == "__main__":
    """
    python3 scripts/tables_tex_management/2_make_latex_table.py --csv ./exp/gnn_exp/final_reports/batch_test/combined_results.csv     --mode Test     --search "\GNNres=Astar_GNN"     --search "\BFSres=BFS"
    """
    main()
