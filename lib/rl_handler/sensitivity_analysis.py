"""Sensitivity analysis over the four d*-aware training signals.

Trains methods x seeds under IDENTICAL config (same instances/frames/batch/
eval-refill, and the same per-(method,seed) init via offline_main's manual_seed-
before-build, so seed s shares its init across methods and only the SIGNAL
differs), then compares them on a fixed validation fringe set. The diagnosed
failure (ledger S) is variance, so the headline metric is the CROSS-SEED IQR of
the tie-aware oracle top-1 regret.

Methods: basic (current Double-DQN), pbrs (potential d* shaping), exact-return
(supervise -d*, no bootstrap), aux (TD + lambda*MSE(-d*)). See offline_main
--signal-mode.

Outputs into <dir-save-model>/sensitive_analysis/: table.{csv,md}, plots, a
README with the pre-registered expectation, and one labeled ONNX per run
(frontier_policy_<F>_{method}_s{seed}.onnx; aux head excluded, shape unchanged).

Graceful: prints [skip] and exits 0 when inputs are absent.

Run from lib/rl_handler, e.g.:
  ../../.venv/bin/python sensitivity_analysis.py \
      --train-csv <binder.csv> --val-csv <a.csv> <b.csv> \
      --dir-save-model /tmp/sa --frames 30000 --seeds 0 1 2 3 4 --device cuda
"""

from __future__ import annotations

import argparse
import csv as csvmod
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.offline.encoder import InstanceCache, pack_fringe_batch  # noqa: E402
from src.offline.tree_env import (  # noqa: E402
    UNREACHABLE_DISTANCE,
    FringeEnv,
    bfs_expansions,
    load_tree_instance,
)
from src.trainer import RLFrontierTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OFFLINE_MAIN = Path(__file__).resolve().parent / "offline_main.py"
METHODS = ["basic", "pbrs", "exact-return", "aux"]
# pre-registered: regret & cross-seed IQR fall basic >= aux >= pbrs >= exact-return
PREREG = "basic >= aux >= pbrs >= exact-return"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="d*-signal sensitivity analysis")
    p.add_argument("--train-csv", nargs="+", required=True)
    p.add_argument("--val-csv", nargs="+", required=True)
    p.add_argument("--dir-save-model", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fringe-size", type=int, default=32)
    p.add_argument("--frames", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-checkpoints", type=int, default=10)
    p.add_argument("--eval-refill-seeds", type=int, default=3)
    p.add_argument("--aux-lambda", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--n-fringes", type=int, default=300, help="fixed val fringe set size")
    p.add_argument("--refill-mode", choices=["heuristic", "random"], default="heuristic",
                   help="beam refill for the node-economy rollout. 'heuristic' (default) "
                   "mirrors RL_BestFirst's deployment MIN refill; 'random' the RNG mode.")
    p.add_argument("--rl-exploration-pct", type=float, default=10.0,
                   help="random-exploration %% of the beam during heuristic refill "
                   "(deployment default 10; exploration_nodes = floor(F * pct/100)).")
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing runs; only re-evaluate + plot")
    p.add_argument("--test-data-dir", default=None,
                   help="held-out test_data dir (default: <dir-save-model>/test_data). "
                   "If present, runs measure-only TEST eval on last.pt (refill sweep "
                   "MIN+RNG); never feeds training/selection. Absent -> skipped.")
    return p.parse_args()


def _abs(paths):
    out = []
    for p in paths:
        pp = Path(p)
        out.append(str(pp if pp.is_absolute() else (REPO / pp)))
    return out


# ---------------- fixed validation fringe set (model-independent) ----------------

def load_inst_cache(csv_path: str):
    cp = Path(csv_path)
    inst = load_tree_instance(cp)
    cache = InstanceCache.from_paths(
        inst.state_paths_abs(REPO), cp.parent / "graph_cache_offline_v1.pt", verbose=False
    )
    return inst, cache


def collect_fringes(inst, fringe_size, cap, n_env_seeds=12):
    seen, out = set(), []
    for es in range(n_env_seeds):
        env = FringeEnv(inst, fringe_size=fringe_size, seed=es, expansion_cap=2000)
        rng = random.Random(1000 + es)
        res = env.reset(seed=es)
        while not res.done and len(out) < cap:
            fr = tuple(res.fringe)
            if len(fr) >= 2 and fr not in seen:
                seen.add(fr)
                out.append(fr)
            res = env.step(rng.randrange(len(res.fringe)))
        if len(out) >= cap:
            break
    return out


@torch.no_grad()
def score(model, cache, state_ids):
    packed = pack_fringe_batch([(cache, list(state_ids))])
    return model(
        node_features=packed["node_features"],
        edge_index=packed["edge_index"],
        edge_attr=packed["edge_attr"],
        membership=packed["membership"],
        candidate_batch=packed["candidate_batch"],
    ).cpu().numpy()


def oracle_regret(model, fringe_set):
    """Tie-aware oracle top-1 regret = fraction of fringes where the model
    argmax is NOT a d*-minimiser (ties: any minimiser counts as correct)."""
    miss = 0
    for inst, cache, fr in fringe_set:
        d = np.array([inst.distance[s] for s in fr], dtype=float)
        if not (d < UNREACHABLE_DISTANCE).any():
            continue
        v = score(model, cache, fr)
        miss += int(d[int(np.argmax(v))] != d.min())
    return miss / max(1, len(fringe_set))


def node_economy(model, val_insts, val_caches, fringe_size, refill_mode,
                 exploration_nodes, seeds=(0, 1, 2)):
    """Mean greedy val expansions over refill seeds, under the DEPLOYMENT refill
    mode (default 'heuristic' = RL_BestFirst's MIN refill: model picks the best
    reservoir states + a small random-exploration budget)."""

    @torch.no_grad()
    def rollout(inst, cache, sd):
        sfn = (lambda ids: score(model, cache, ids)) if refill_mode == "heuristic" else None
        env = FringeEnv(
            inst, fringe_size=fringe_size, seed=sd, expansion_cap=2000,
            refill_mode=refill_mode, reservoir_score_fn=sfn,
            exploration_nodes=exploration_nodes,
        )
        res = env.reset(seed=sd)
        while not res.done:
            v = score(model, cache, res.fringe)
            res = env.step(int(np.argmax(v)))
        return int(res.info["expansions"])

    tot = 0.0
    for inst, cache in zip(val_insts, val_caches):
        tot += float(np.mean([rollout(inst, cache, sd) for sd in seeds]))
    return tot


def tail_spread(run_dir: Path):
    import json
    h = run_dir / "history.json"
    if not h.exists():
        return float("nan")
    cks = json.loads(h.read_text())["checkpoints"]
    last4 = [c["summary"]["val_total_expansions"] for c in cks[-4:]]
    return float(max(last4) - min(last4)) if last4 else float("nan")


def iqm_iqr(xs):
    xs = [x for x in xs if not np.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 4:
        return float(np.mean(xs)), float(max(xs) - min(xs))
    q1, q3 = np.percentile(xs, 25), np.percentile(xs, 75)
    trim = [x for x in xs if q1 <= x <= q3] or xs
    return float(np.mean(trim)), float(q3 - q1)


# ----------------------------- HELD-OUT TEST EVAL -----------------------------
# test_data is measured-and-reported ONLY: evaluated on last.pt (final-frame
# weights, NO checkpoint selection), never feeding gradients/selection. Refill
# sweep mirrors the planner's RL_heuristics modes: heuristic (MIN) + random (RNG).
REFILL_SWEEP = ("heuristic", "random")


@torch.no_grad()
def _test_rollout(model, inst, cache, F, refill_mode, expl_nodes, sd, cap=2000):
    """One greedy rollout on a test instance under `refill_mode`. Returns
    (solved, expansions, rank_hits, rank_total) where a 'rank hit' is the model
    argmax being a d*-minimiser of the live beam (rank accuracy vs d*)."""
    sfn = (lambda ids: score(model, cache, ids)) if refill_mode == "heuristic" else None
    env = FringeEnv(inst, fringe_size=F, seed=sd, expansion_cap=cap,
                    refill_mode=refill_mode, reservoir_score_fn=sfn,
                    exploration_nodes=expl_nodes)
    res = env.reset(seed=sd)
    rh = rt = 0
    while not res.done:
        fr = res.fringe
        k = int(np.argmax(score(model, cache, fr)))
        d = np.array([inst.distance[s] for s in fr], dtype=float)
        if (d < UNREACHABLE_DISTANCE).any():
            rt += 1
            rh += int(d[k] == d.min())
        res = env.step(k)
    return bool(res.info["goal_found"]), int(res.info["expansions"]), rh, rt


def _test_group(name: str) -> str:
    """Sub-group of a held-out test instance: 'rich' (SC_R_ richness axis) vs
    'depth' (high-pl base-repr axis). A signal can pass one axis and fail the
    other, so we never let the aggregate hide it."""
    return "rich" if name.startswith("SC_R_") else "depth"


def _rich_encoding_sanity(insts, caches, models):
    """Confirm rich SC_R instances produce NON-degenerate scores (not all-equal
    logits). A degenerate score there is a representation/encoding bug, not a
    generalization result. Returns a human line."""
    rich = [(i, c) for i, c in zip(insts, caches) if _test_group(i.name) == "rich"]
    if not rich:
        return "[test] rich-encoding sanity: no rich (SC_R_) instances present."
    inst, cache = rich[0]
    # a multi-candidate fringe from this rich instance (root children)
    fr = list(inst.children[inst.root_id])[:8]
    if len(fr) < 2:
        fr = list(range(min(8, inst.n_states)))
    stds = []
    for m in models:
        v = score(m, cache, fr)
        stds.append(float(np.std(v)))
    mx = max(stds) if stds else 0.0
    ok = mx > 1e-4
    return (f"[test] rich-encoding sanity on {inst.name} (N={len(fr)}): "
            f"max logit-std across models = {mx:.4g} -> "
            f"{'OK non-degenerate' if ok else 'DEGENERATE (encoding problem, NOT a result)'}")


def test_eval(sa_dir, test_dir, args, F, expl_nodes):
    """Held-out TEST metrics on last.pt across the refill sweep, broken out by
    sub-group (depth vs richness) and aggregate. Pure measure-and-report."""
    import csv as _csv

    insts, caches = [], []
    for d in sorted(Path(test_dir).iterdir()):
        if not d.is_dir():
            continue
        csvs = sorted(d.glob(f"{d.name}_depth_*.csv"))
        if not csvs:
            continue
        try:
            inst = load_tree_instance(csvs[0])
            cache = InstanceCache.from_paths(
                inst.state_paths_abs(REPO), d / "graph_cache_offline_v1.pt", verbose=False)
        except Exception as e:
            print(f"[test] skip {d.name}: {e}")
            continue
        insts.append(inst)
        caches.append(cache)
    if not insts:
        print("[test] no usable test instances in test_data; skipping test eval.")
        return
    groups = {i.name: _test_group(i.name) for i in insts}
    n_by_g = {g: sum(1 for v in groups.values() if v == g) for g in set(groups.values())}
    print(f"[test] {len(insts)} held-out TEST instances | eval on last.pt (NO selection) "
          f"| refill sweep {REFILL_SWEEP} | sub-groups {n_by_g}")

    # one row per (method, seed, refill, group) where group in {all, depth, rich}
    rows = []
    first_models = []  # seed0 model per method, for the encoding sanity check
    for method in args.methods:
        for seed in args.seeds:
            last = sa_dir / method / f"seed{seed}_fringe{F}" / "last.pt"
            if not last.exists():
                continue
            model = RLFrontierTrainer.load_model(last, device="cpu")
            if seed == args.seeds[0]:
                first_models.append(model)
            for refill in REFILL_SWEEP:
                acc = {g: {"cov": 0, "n": 0, "econ": [], "rh": 0, "rt": 0}
                       for g in ("all", "depth", "rich")}
                for inst, cache in zip(insts, caches):
                    s, e, h, t = _test_rollout(model, inst, cache, F, refill, expl_nodes, sd=0)
                    for gg in ("all", groups[inst.name]):
                        a = acc[gg]
                        a["n"] += 1
                        a["cov"] += int(s)
                        if s:
                            a["econ"].append(e)
                        a["rh"] += h
                        a["rt"] += t
                for gg, a in acc.items():
                    if a["n"] == 0:
                        continue
                    rows.append({
                        "method": method, "seed": seed, "refill": refill, "group": gg,
                        "coverage": a["cov"] / a["n"],
                        "node_econ_solved": float(np.mean(a["econ"])) if a["econ"] else float("nan"),
                        "rank_acc": a["rh"] / a["rt"] if a["rt"] else float("nan"),
                        "n_inst": a["n"],
                    })
            print(f"[test] {method} s{seed}: "
                  + " ".join(f"{r['refill']}/{r['group']} cov={r['coverage']:.2f}"
                             for r in rows if r["method"] == method and r["seed"] == seed
                             and r["group"] == "all"))

    print(_rich_encoding_sanity(insts, caches, first_models))

    with (sa_dir / "test_table.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["method", "seed", "refill", "group",
                                            "coverage", "node_econ_solved", "rank_acc", "n_inst"])
        w.writeheader()
        w.writerows(rows)

    # per-group BFS/optimal context
    ctx = {}
    for g in ("all", "depth", "rich"):
        gi = [i for i in insts if g == "all" or groups[i.name] == g]
        if gi:
            ctx[g] = (sum(bfs_expansions(i)["expansions"] for i in gi),
                      sum((i.optimal_expansions() or 0) for i in gi), len(gi))

    md = ["# HELD-OUT TEST metrics (last.pt; measure-and-report only)\n",
          "Final-frame last.pt (NO checkpoint selection); never fed training/selection. "
          "Sub-groups: depth = high-pl SC_Multi (base repr), rich = low-pl SC_R "
          "(richness). EXPLORATORY: Test compares signals, not an unbiased single number.\n",
          "context: " + "; ".join(f"{g}: n={ctx[g][2]} BFS={ctx[g][0]} opt={ctx[g][1]}"
                                   for g in ctx) + "\n",
          "| group | method | refill | coverage IQM | cov IQR | econ_solved IQM | "
          "econ IQR | rank_acc IQM | n_seed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in ("all", "depth", "rich"):
        for method in args.methods:
            for refill in REFILL_SWEEP:
                sub = [r for r in rows if r["group"] == g and r["method"] == method
                       and r["refill"] == refill]
                if not sub:
                    continue
                cm, ci = iqm_iqr([r["coverage"] for r in sub])
                em, ei = iqm_iqr([r["node_econ_solved"] for r in sub])
                rm, _ = iqm_iqr([r["rank_acc"] for r in sub])
                md.append(f"| {g} | {method} | {refill} | {cm:.2f} | {ci:.2f} | "
                          f"{em:.1f} | {ei:.1f} | {rm:.3f} | {len(sub)} |")
    (sa_dir / "test_table.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
    _test_plot(sa_dir, rows, args)
    print(f"[test] wrote test_table.{{csv,md}} + test_*.png to {sa_dir}")


def _test_plot(sa_dir, rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping test plot: {e}")
        return
    groups = ("depth", "rich")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    for gi, g in enumerate(groups):
        ax = axes[gi]
        x = 0
        labels = []
        for method in args.methods:
            for refill in REFILL_SWEEP:
                cs = [r["coverage"] for r in rows if r["group"] == g
                      and r["method"] == method and r["refill"] == refill]
                if not cs:
                    continue
                ax.scatter([x] * len(cs), cs, alpha=0.6, s=40)
                ax.scatter([x], [np.mean(cs)], marker="_", s=400, c="k")
                labels.append(f"{method[:5]}/{refill[:3]}")
                x += 1
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"held-out TEST coverage — {g} sub-group")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sa_dir / "test_coverage_by_group.png")
    plt.close(fig)


def run_one(args, method, seed, sa_dir) -> None:
    # offline_main appends `_fringe{F}` to --dir-save-model; pass the base.
    run_base = sa_dir / method / f"seed{seed}"
    cmd = [
        sys.executable, str(OFFLINE_MAIN),
        "--train-csv", *_abs(args.train_csv),
        "--val-csv", *_abs(args.val_csv),
        "--fringe-sizes", str(args.fringe_size),
        "--frames", str(args.frames),
        "--n-checkpoints", str(args.n_checkpoints),
        "--batch-size", str(args.batch_size),
        "--eval-refill-seeds", str(args.eval_refill_seeds),
        "--signal-mode", method, "--aux-lambda", str(args.aux_lambda),
        "--seed", str(seed), "--no-export-onnx",
        "--dir-save-model", str(run_base),
    ]
    if args.device:
        cmd += ["--device", args.device]
    print(f"  [train] {method} seed{seed} -> {run_base}_fringe{args.fringe_size}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    # prerequisite data
    for c in _abs(args.train_csv) + _abs(args.val_csv):
        if not Path(c).exists():
            print(f"[skip] missing csv {c}")
            return
        if not (Path(c).parent / "graph_cache_offline_v1.pt").exists():
            print(f"[skip] missing graph cache for {Path(c).parent.name}")
            return

    F = args.fringe_size
    expl_nodes = int(F * args.rl_exploration_pct / 100.0)  # deployment refill budget
    sa_dir = Path(args.dir_save_model) / "sensitive_analysis"
    sa_dir.mkdir(parents=True, exist_ok=True)

    # fixed validation fringe set (from val instances; model-independent)
    val_insts, val_caches = [], []
    for c in _abs(args.val_csv):
        inst, cache = load_inst_cache(c)
        val_insts.append(inst)
        val_caches.append(cache)
    fringe_set = []
    per = max(1, args.n_fringes // len(val_insts))
    for inst, cache in zip(val_insts, val_caches):
        for fr in collect_fringes(inst, F, per):
            fringe_set.append((inst, cache, fr))
    opt = sum(i.optimal_expansions() for i in val_insts)
    bfs = sum(bfs_expansions(i)["expansions"] for i in val_insts)
    print(f"[setup] {len(fringe_set)} fixed val fringes | val optimal={opt} bfs={bfs} | "
          f"methods={args.methods} seeds={args.seeds} | "
          f"refill={args.refill_mode} (explore_nodes={expl_nodes})")

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            # offline_main writes to <base>_fringe{F}
            run_dir = sa_dir / method / f"seed{seed}_fringe{F}"
            if not args.skip_train or not (run_dir / "last.pt").exists():
                run_one(args, method, seed, sa_dir)
            last = run_dir / "last.pt"
            if not last.exists():
                print(f"  [warn] no last.pt for {method} seed{seed}; skipping eval")
                continue
            model = RLFrontierTrainer.load_model(last, device="cpu")
            regret = oracle_regret(model, fringe_set)
            econ = node_economy(model, val_insts, val_caches, F,
                                args.refill_mode, expl_nodes)
            spread = tail_spread(run_dir)
            rows.append({"method": method, "seed": seed, "regret": regret,
                         "node_economy": econ, "tail_spread": spread})
            # labeled ONNX export of the stage-matched final weights
            try:
                exp = RLFrontierTrainer(model=model, kind_of_data="merged", device="cpu")
                onnx_path = sa_dir / f"frontier_policy_{F}_{method}_s{seed}.onnx"
                exp.to_onnx(onnx_path, node_input_dim=1, onnx_frontier_size=F)
            except Exception as e:
                print(f"  [warn] onnx export failed {method} seed{seed}: {e}")
            print(f"  [eval] {method} seed{seed}: regret={regret:.3f} "
                  f"node_econ={econ:.1f} tail_spread={spread:.1f}", flush=True)

    if not rows:
        print("[skip] no runs produced weights; nothing to aggregate.")
        return

    _write_outputs(sa_dir, rows, args, opt, bfs)

    # Held-out TEST eval (measure-and-report only, on last.pt). Auto-located at
    # <dir-save-model>/test_data unless overridden; skipped if absent so domains
    # without a test set don't error.
    test_dir = Path(args.test_data_dir) if args.test_data_dir else (
        Path(args.dir_save_model) / "test_data")
    if test_dir.is_dir():
        test_eval(sa_dir, test_dir, args, F, expl_nodes)
    else:
        print(f"[test] no test_data at {test_dir}; skipping held-out test eval.")


def _write_outputs(sa_dir, rows, args, opt, bfs):
    # CSV
    with (sa_dir / "table.csv").open("w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=["method", "seed", "regret",
                                              "node_economy", "tail_spread"])
        w.writeheader()
        w.writerows(rows)

    # aggregate per method
    agg = []
    for method in args.methods:
        rs = [r["regret"] for r in rows if r["method"] == method]
        es = [r["node_economy"] for r in rows if r["method"] == method]
        if not rs:
            continue
        rm, riqr = iqm_iqr(rs)
        em, eiqr = iqm_iqr(es)
        agg.append((method, rm, riqr, em, eiqr, len(rs)))

    # markdown table
    md = ["# d*-signal sensitivity analysis\n",
          f"Config: F={args.fringe_size}, frames={args.frames}, batch={args.batch_size}, "
          f"seeds={args.seeds}. Val optimal={opt}, BFS={bfs}.\n",
          f"Pre-registered (regret & cross-seed IQR should fall): **{PREREG}**.\n",
          "| method | regret IQM | regret IQR (cross-seed, HEADLINE) | "
          "node_econ IQM | node_econ IQR | n |",
          "|---|---|---|---|---|---|"]
    for method, rm, riqr, em, eiqr, n in agg:
        md.append(f"| {method} | {rm:.3f} | {riqr:.3f} | {em:.1f} | {eiqr:.1f} | {n} |")
    md.append(f"\nBFS node economy baseline = {bfs}; optimal = {opt}.\n")
    (sa_dir / "table.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))

    _plots(sa_dir, rows, args, bfs, opt)

    readme = (
        "# sensitive_analysis\n\n"
        "Four d*-aware training signals trained under identical config + matched "
        "per-(method,seed) init; only the SIGNAL differs.\n\n"
        f"PRIMARY metric: DEPLOYMENT node economy = greedy val expansions under the "
        f"'{args.refill_mode}' beam refill that mirrors RL_BestFirst (the C++ "
        f"within-fringe width-F beam ranker; vs BFS={bfs}, optimal={opt}). Aggregated "
        "per method as IQM +/- cross-seed IQR; the CROSS-SEED IQR is the headline "
        "because the diagnosed failure (ledger S) is variance.\n\n"
        f"Pre-registered expectation: the d*-signals beat **basic** on deployment node "
        f"economy and its cross-seed IQR, falling in the order **{PREREG}** (basic "
        "worst, increasing target discriminativeness).\n\n"
        "Falsifier: a d*-signal that does NOT beat basic under deployment refill is a "
        "real null (the supervision quality, not the under-determination diagnosis, "
        "would be wrong). Any deviation from the pre-registered order is itself a result.\n\n"
        "SECONDARY (why-diagnostic only, NOT primary): tie-aware oracle regret on the "
        "off-distribution random-walk fringe set (it saturates ~0.9 and dissociates "
        "from economy; kept for continuity, not for the verdict).\n\n"
        "Weights compared are last.pt (stage-matched final frame). ONNX exports are "
        "labeled frontier_policy_<F>_{method}_s{seed}.onnx (aux head excluded; output "
        "shape unchanged; confirmed to LOAD in the C++ planner, non-separated). These "
        "models live ONLY here, never installed to the deploy path.\n"
    )
    (sa_dir / "README.md").write_text(readme)
    print(f"\n[done] wrote table.csv/md, plots, README to {sa_dir}")


def _plots(sa_dir, rows, args, bfs, opt):
    try:
        import json

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping plots: {e}")
        return
    methods = [m for m in args.methods if any(r["method"] == m for r in rows)]
    F = args.fringe_size

    # 1) node-economy distribution per method (strip) + BFS/optimal lines
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for x, m in enumerate(methods):
        es = [r["node_economy"] for r in rows if r["method"] == m]
        ax.scatter([x] * len(es), es, alpha=0.6, s=40)
        ax.scatter([x], [np.mean(es)], marker="_", s=700, c="k")
    ax.axhline(bfs, ls="--", c="gray", label=f"BFS={bfs}")
    ax.axhline(opt, ls=":", c="green", label=f"optimal={opt}")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20)
    ax.set_title(f"deployment node economy per seed ({args.refill_mode} refill; lower=better)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sa_dir / "node_economy_dist.png")
    plt.close(fig)

    # 2) learning curves: in-training val expansions vs frame, cross-seed band
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for m in methods:
        curves = []
        frames = None
        for s in args.seeds:
            h = sa_dir / m / f"seed{s}_fringe{F}" / "history.json"
            if not h.exists():
                continue
            cks = json.loads(h.read_text())["checkpoints"]
            frames = [c["summary"]["frame"] for c in cks]
            curves.append([c["summary"]["val_total_expansions"] for c in cks])
        if not curves or frames is None:
            continue
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        fr = frames[:n]
        med = np.median(arr, 0)
        ax.plot(fr, med, marker="o", label=m)
        ax.fill_between(fr, arr.min(0), arr.max(0), alpha=0.18)
    ax.axhline(bfs, ls="--", c="gray", label=f"BFS={bfs}")
    ax.set_xlabel("frame")
    ax.set_ylabel("val expansions (in-training eval)")
    ax.set_title("learning curves: node economy vs frames (median + min-max band)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sa_dir / "learning_curves.png")
    plt.close(fig)

    # 3) cross-seed IQR bar (the headline: variance is the diagnosed failure)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    iqrs = []
    for m in methods:
        es = [r["node_economy"] for r in rows if r["method"] == m]
        _, iqr = iqm_iqr(es)
        iqrs.append(iqr)
    ax.bar(range(len(methods)), iqrs, color="tab:red", alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20)
    ax.set_title("cross-seed IQR of node economy (HEADLINE; lower=more stable)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(sa_dir / "cross_seed_iqr.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
