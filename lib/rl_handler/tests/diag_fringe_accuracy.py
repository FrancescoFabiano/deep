"""Diagnostic: which fringe size is "more accurate"?

Two distinct notions, reported separately, over whichever domains/fringes are
on disk under exp/rl_exp/batch0/_models/<dom>/seed<seed>_fringe<F>/.

Metric 1 — node economy (the deployment objective): read from history.json the
best_by_expansions checkpoint (reconstructed with the trainer's exact strict-<
rule: earliest checkpoint attaining the min val_total_expansions). Report
val_total_expansions (THE objective), val_spearman_all (CALIBRATION, not
accuracy), best_frame, plus optimal/BFS sums for context.

Metric 2 — within-fringe agreement (isolates scorer quality): instrument a
greedy (eps=0) rollout reusing the real FringeEnv + OfflineDQNTrainer.greedy_action
with the best_by_expansions.pt weights. At each expansion step, with the live
fringe members and their eval-only d* oracle (inst.distance), record whether the
model's argmax slot is a MINIMUM-d* member, and the d*-rank of the picked state.

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_fringe_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.dqn import OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache  # noqa: E402
from src.offline.tree_env import FringeEnv, load_tree_instance  # noqa: E402
from src.trainer import RLFrontierTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "exp/rl_exp/batch0/_models"
SEEDS_M2 = [0, 1, 2, 3, 4]  # rollout RANDOM-refill seeds aggregated for Metric 2


def discover() -> list[tuple[str, int, Path]]:
    """(domain, fringe, seed_dir) for every seed*_fringe* with a history.json."""
    found = []
    for dom_dir in sorted(MODELS.glob("*")):
        if not dom_dir.is_dir():
            continue
        for sd in sorted(dom_dir.glob("seed*_fringe*")):
            if (sd / "history.json").exists():
                f = int(sd.name.split("fringe")[-1])
                found.append((dom_dir.name, f, sd))
    return found


def best_by_expansions_ckpt(history: dict) -> dict:
    """Replay the trainer's selection: earliest checkpoint attaining the
    minimum val_total_expansions (strict < update, so ties -> earliest)."""
    best = None
    best_exp = float("inf")
    for ck in history["checkpoints"]:
        ve = ck["summary"]["val_total_expansions"]
        if ve < best_exp:
            best_exp = ve
            best = ck
    return best


def m2_prereqs_missing(seed_dir: Path) -> str | None:
    """None if Metric 2 can run for this seed dir, else what's missing.

    Metric 2 needs the best_by_expansions.pt weights, args.json, and for each
    val instance both the CSV table and its graph cache on disk (the DOT raw
    files are untracked and typically gone, so the cache must be present)."""
    if not (seed_dir / "best_by_expansions.pt").exists():
        return f"{seed_dir.name}: best_by_expansions.pt missing"
    args_path = seed_dir / "args.json"
    if not args_path.exists():
        return f"{seed_dir.name}: args.json missing"
    args = json.loads(args_path.read_text())
    for csv in args.get("val_csv", []):
        csv_path = Path(csv)
        if not csv_path.is_absolute():
            csv_path = REPO / csv_path
        if not csv_path.exists():
            return f"{seed_dir.name}: val csv missing ({csv_path.name})"
        if not (csv_path.parent / "graph_cache_offline_v1.pt").exists():
            return f"{seed_dir.name}: graph cache missing ({csv_path.parent.name})"
    return None


def metric1(seed_dir: Path) -> dict | None:
    history = json.loads((seed_dir / "history.json").read_text())
    ck = best_by_expansions_ckpt(history)
    if ck is None:
        return None
    per = ck["val_per_instance"]
    opt = sum(
        v["optimal_expansions"] for v in per.values()
        if v["optimal_expansions"] is not None
    )
    bfs = sum(v["bfs_expansions"] for v in per.values())
    return {
        "val_expansions": ck["summary"]["val_total_expansions"],
        "optimal": opt,
        "bfs": bfs,
        "val_spearman": ck["summary"]["val_spearman_all"],
        "best_frame": ck["summary"]["frame"],
        "frames_total": history["history"]["frame"][-1] if history["history"]["frame"] else None,
    }


def build_trainer(seed_dir: Path, dom: str, fringe: int):
    """Load the val instance + best_by_expansions.pt into an OfflineDQNTrainer
    (only val needed; train_ids empty)."""
    args = json.loads((seed_dir / "args.json").read_text())
    val_csv = args["val_csv"]
    instances, caches = [], []
    for csv in val_csv:
        csv_path = Path(csv)
        if not csv_path.is_absolute():
            csv_path = REPO / csv_path
        inst = load_tree_instance(csv_path)
        cache = InstanceCache.from_paths(
            inst.state_paths_abs(REPO),
            csv_path.parent / "graph_cache_offline_v1.pt",
            verbose=False,
        )
        instances.append(inst)
        caches.append(cache)
    model = RLFrontierTrainer.load_model(seed_dir / "best_by_expansions.pt", device="cpu")
    trainer = OfflineDQNTrainer(
        model=model,
        instances=instances,
        caches=caches,
        train_ids=[],
        val_ids=list(range(len(instances))),
        fringe_size=fringe,
        eval_expansion_cap=args.get("eval_expansion_cap", 2000),
        seed=args.get("seed", 42),
        device="cpu",
    )
    return trainer, instances


def metric2(seed_dir: Path, dom: str, fringe: int) -> dict:
    trainer, instances = build_trainer(seed_dir, dom, fringe)
    n_hits = n_steps = n_nontrivial = nontrivial_hits = 0
    rank_norm_sum = rank_sum = 0.0
    for vid, inst in enumerate(instances):
        dist = inst.distance
        for seed in SEEDS_M2:
            env = FringeEnv(
                inst, fringe_size=fringe, seed=seed,
                expansion_cap=trainer.eval_expansion_cap,
            )
            res = env.reset(seed=seed)
            while not res.done:
                fr = res.fringe
                picked = trainer.greedy_action(vid, fr)
                ds = [dist[s] for s in fr]
                dmin = min(ds)
                dmax = max(ds)
                pd = ds[picked]
                # d*-rank of the picked state (1 = best); ties share the better rank
                rank = 1 + sum(1 for d in ds if d < pd)
                n_steps += 1
                rank_sum += rank
                rank_norm_sum += rank / len(fr)
                hit = pd <= dmin  # picked a minimum-d* member
                n_hits += int(hit)
                if dmax > dmin:  # fringe actually offers a choice that matters
                    n_nontrivial += 1
                    nontrivial_hits += int(hit)
                res = env.step(picked)
    return {
        "agreement_rate": n_hits / n_steps if n_steps else float("nan"),
        "mean_rank": rank_sum / n_steps if n_steps else float("nan"),
        "mean_rank_norm": rank_norm_sum / n_steps if n_steps else float("nan"),
        "n_steps": n_steps,
        "agreement_nontrivial": (
            nontrivial_hits / n_nontrivial if n_nontrivial else float("nan")
        ),
        "n_nontrivial": n_nontrivial,
    }


def main() -> None:
    runs = discover()
    if not runs:
        print("No trained per-fringe models found. Run the production sweep first.")
        return

    m1, m2 = {}, {}
    for dom, f, sd in runs:
        r1 = metric1(sd)
        if r1 is None:
            print(f"[skip] {dom} F{f}: no usable checkpoints in history.json")
        else:
            m1[(dom, f)] = r1
        missing = m2_prereqs_missing(sd)
        if missing is not None:
            print(f"[skip] {dom} F{f} Metric 2: {missing}")
        else:
            m2[(dom, f)] = metric2(sd, dom, f)

    if not m1 and not m2:
        print("[skip] no run has usable inputs for either metric; nothing to do.")
        return

    print("\n=== Metric 1 — node economy (val greedy expansions; lower = better) ===")
    h = f"{'dom':<6}{'F':>4}{'val_exp':>9}{'optimal':>9}{'BFS':>7}{'spearman(calib)':>18}{'best_frame':>12}"
    print(h)
    print("-" * len(h))
    for (dom, f), r in sorted(m1.items()):
        sp = f"{r['val_spearman']:.4f}" if r["val_spearman"] is not None else "n/a"
        print(f"{dom:<6}{f:>4}{r['val_expansions']:>9}{r['optimal']:>9}{r['bfs']:>7}"
              f"{sp:>18}{r['best_frame']:>12}")

    print("\n=== Metric 2 — within-fringe agreement (greedy rollout, d* oracle) ===")
    h2 = (f"{'dom':<6}{'F':>4}{'agree_rate':>12}{'mean_rank':>11}"
          f"{'rank_norm':>11}{'agree_nontriv':>15}{'n_steps':>9}{'n_nontriv':>11}")
    print(h2)
    print("-" * len(h2))
    for (dom, f), r in sorted(m2.items()):
        ant = f"{r['agreement_nontrivial']:.3f}" if r["n_nontrivial"] else "n/a"
        print(f"{dom:<6}{f:>4}{r['agreement_rate']:>12.3f}{r['mean_rank']:>11.2f}"
              f"{r['mean_rank_norm']:>11.3f}{ant:>15}{r['n_steps']:>9}{r['n_nontrivial']:>11}")
    print(f"\n(Metric 2 aggregates rollouts over refill seeds {SEEDS_M2}; "
          f"agree_nontriv restricts to steps where the fringe d* spread is non-zero.)")


if __name__ == "__main__":
    main()
