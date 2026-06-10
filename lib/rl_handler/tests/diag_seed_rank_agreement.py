"""Diagnostic: do independently-seeded converged models AGREE on the within-
fringe ranking?

Falsifier for the reframed ledger-S hypothesis: the offline TD objective
under-determines the within-fringe *order*. The -1/step reward is constant
across fringe picks, so it carries no ranking signal; all ordering flows through
a weak/high-variance bootstrap. If true, three seeds that all reach td~=0 /
q_mean~=-5.6 should nonetheless converge to DIFFERENT rankers — low top-1
(argmax) agreement and low/moderate Kendall tau on the slot scores.

Method (read-only, .pt weights — not ONNX, which has a fixed scatter dim):
  1. Build a FIXED common set of multi-candidate fringe snapshots (N>=2) from SC
     binders, via a model-INDEPENDENT random walk (fixed RNG) so every seed
     scores the identical fringes.
  2. Score each fringe with each seed's best_by_expansions.pt -> one logit/slot.
  3. Per seed-pair, aggregate top-1 agreement (deployment-relevant: greedy uses
     argmax) and mean Kendall tau, stratified by fringe width N.
  4. Breadth: distribution of per-fringe 3-way argmax consensus (is disagreement
     broad or concentrated on a few high-leverage fringes?).

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_seed_rank_agreement.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.encoder import InstanceCache, pack_fringe_batch  # noqa: E402
from src.offline.tree_env import FringeEnv, load_tree_instance  # noqa: E402
from src.trainer import RLFrontierTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "exp/rl_exp/diversity_v2/_models/SC"
TRAIN_DATA = REPO / "exp/rl_exp/batch0/_models/SC/training_data"
BINDERS = ["SC_10_8__pl_15", "SC_9_11__pl_8"]  # high-branching SC binders
SEEDS = [0, 1, 2]
N_ENV_SEEDS = 12      # model-independent random-walk seeds for fringe collection
MAX_FRINGES = 400     # cap on distinct multi-candidate snapshots
WIDTH_BINS = [(2, 2), (3, 4), (5, 8), (9, 16), (17, 32)]


def model_pt(seed: int, F: int) -> Path:
    return MODELS / f"seed{seed}_fringe{F}" / "best_by_expansions.pt"


def models_present(F: int) -> bool:
    return all(model_pt(s, F).exists() for s in SEEDS)


def load_binder(name: str):
    csv = TRAIN_DATA / name / f"{name}_depth_25.csv"
    cache_file = TRAIN_DATA / name / "graph_cache_offline_v1.pt"
    if not csv.exists() or not cache_file.exists():
        return None
    inst = load_tree_instance(csv)
    cache = InstanceCache.from_paths(inst.state_paths_abs(REPO), cache_file, verbose=False)
    return inst, cache


def collect_fringes(inst, fringe_size: int, cap: int):
    """Fixed, model-independent multi-candidate fringe snapshots via a random
    walk (fixed RNG per env seed). Returns list of (state_ids tuple)."""
    seen = set()
    out = []
    for es in range(N_ENV_SEEDS):
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
def score(model, cache, state_ids) -> np.ndarray:
    packed = pack_fringe_batch([(cache, list(state_ids))])
    logits = model(
        node_features=packed["node_features"],
        edge_index=packed["edge_index"],
        edge_attr=packed["edge_attr"],
        membership=packed["membership"],
        candidate_batch=packed["candidate_batch"],
    )
    return logits.cpu().numpy()


def width_bin(n: int) -> str:
    for lo, hi in WIDTH_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return f">{WIDTH_BINS[-1][1]}"


def run_fringe_size(F: int) -> None:
    print(f"\n########## F={F} models (best_by_expansions.pt) ##########")
    models = {s: RLFrontierTrainer.load_model(model_pt(s, F), device="cpu") for s in SEEDS}

    # Fixed fringe set + per-model scores. Cap PER BINDER so both contribute
    # (one binder's wide fringes would otherwise fill the global cap alone).
    per_binder = max(1, MAX_FRINGES // len(BINDERS))
    fringes = []           # (cache, state_ids, N)
    for name in BINDERS:
        loaded = load_binder(name)
        if loaded is None:
            print(f"[skip] binder {name}: csv/cache absent")
            continue
        inst, cache = loaded
        snaps = collect_fringes(inst, fringe_size=F, cap=per_binder)
        for fr in snaps:
            fringes.append((cache, fr, len(fr)))
        print(f"  {name}: collected {len(snaps)} multi-candidate fringes")
    if not fringes:
        print("[skip] no multi-candidate fringes collected; nothing to compare.")
        return

    # argmax slot + score vector per (model, fringe)
    argmax = {s: [] for s in SEEDS}
    scores = {s: [] for s in SEEDS}
    widths = []
    for cache, fr, n in fringes:
        widths.append(n)
        for s in SEEDS:
            v = score(models[s], cache, fr)
            argmax[s].append(int(np.argmax(v)))
            scores[s].append(v)
    widths = np.array(widths)
    n_fr = len(fringes)
    print(f"  total fringes scored: {n_fr}  (width: min {widths.min()} "
          f"max {widths.max()} median {int(np.median(widths))})")

    pairs = [(0, 1), (0, 2), (1, 2)]
    # ---- overall + per-pair top-1 agreement & mean Kendall tau ----
    print(f"\n  {'pair':<8}{'top1_agree':>12}{'mean_tau':>10}")
    top1_by_pair = {}
    for a, b in pairs:
        same = np.array([argmax[a][i] == argmax[b][i] for i in range(n_fr)])
        taus = []
        for i in range(n_fr):
            t = kendalltau(scores[a][i], scores[b][i]).statistic
            if not np.isnan(t):
                taus.append(t)
        top1_by_pair[(a, b)] = same
        print(f"  {f'{a}-{b}':<8}{same.mean():>11.1%}{np.mean(taus):>10.3f}")
    all_same = np.vstack([top1_by_pair[p] for p in pairs])  # 3 x n_fr
    mean_top1 = all_same.mean()
    print(f"  {'MEAN':<8}{mean_top1:>11.1%}"
          f"{'  (across the 3 pairs; spread '+f'{all_same.mean(1).min():.0%}-{all_same.mean(1).max():.0%})':>10}")

    # ---- width-stratified top-1 agreement (mean over the 3 pairs) ----
    print("\n  width-stratified top-1 agreement (mean over 3 pairs):")
    print(f"  {'width':<8}{'n_fr':>6}{'top1':>9}")
    for lo, hi in WIDTH_BINS:
        mask = (widths >= lo) & (widths <= hi)
        if not mask.any():
            continue
        lab = f"{lo}-{hi}" if lo != hi else f"{lo}"
        print(f"  {lab:<8}{int(mask.sum()):>6}{all_same[:, mask].mean():>8.1%}")

    # ---- Task 4: breadth — per-fringe 3-way argmax consensus ----
    # consensus = how many distinct argmax slots the 3 seeds produce per fringe.
    n_distinct = np.array([len({argmax[0][i], argmax[1][i], argmax[2][i]}) for i in range(n_fr)])
    full = (n_distinct == 1).mean()   # all 3 agree
    split2 = (n_distinct == 2).mean()  # 2 agree, 1 differs
    split3 = (n_distinct == 3).mean()  # all 3 differ
    print("\n  breadth of disagreement (3-way argmax consensus per fringe):")
    print(f"    all-3-agree: {full:.1%}   2-of-3: {split2:.1%}   all-3-differ: {split3:.1%}")
    disagree_frac = 1 - full
    if disagree_frac >= 0.4:
        verdict = ("HYPOTHESIS SUPPORTED — low top-1 agreement, broad disagreement: "
                   "the seeds converge to different rankers despite td~=0.")
    elif disagree_frac <= 0.15:
        verdict = ("HYPOTHESIS REJECTED — seeds largely agree on argmax: the ranker is "
                   "seed-consistent; val variance comes from elsewhere.")
    else:
        verdict = ("MIXED — partial disagreement; ranker is somewhat seed-dependent "
                   "(revisit with more fringes / both fringe sizes).")
    print(f"\n  CALL [F={F}]: mean top-1 agreement {mean_top1:.0%}, "
          f"{disagree_frac:.0%} of fringes lack full consensus -> {verdict}")


def main() -> None:
    any_run = False
    for F in (32, 64):
        if not models_present(F):
            print(f"[skip] F={F}: not all of seed{{0,1,2}}_fringe{F}/best_by_expansions.pt present")
            continue
        any_run = True
        run_fringe_size(F)
    if not any_run:
        print("[skip] no diversity_v2 seed models on disk; nothing to compare.")


if __name__ == "__main__":
    main()
