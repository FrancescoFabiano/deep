"""Diagnostic: what per-fringe-element SUPERVISION is derivable offline?

Ledger S is root-caused (the objective under-determines the within-fringe
ranking). The fix injects ranking signal — but the loss family depends on what
target is actually available per fringe element WITHOUT paying the expansions we
are trying to avoid. This audits that, on multi-candidate fringe snapshots from
SC binders. Read-only: no training, no reward/model/contract change.

Reports:
  1. d* coverage — fraction of fringe elements with an EXACT distance-to-goal.
     (d* is precomputed in the generation table's "Distance From Goal" column and
     loaded into inst.distance; unreachable states carry +inf.)
  2. Available labels + coverage — d*, depth, on-optimal-path flag (derived from
     depth+d*), goal flag; and what is NOT present (planner f-value / expansion
     order are not in the table). Subtree size is derivable but unstored.
  3. Within-fringe target quality — on fringes with >=2 labelled elements: unique
     best (argmin d*) vs ties; on-path element present/unique; and how often the
     current v2 model's argmax already matches the oracle best (warm-start gap).

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_fringe_supervision_audit.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.encoder import InstanceCache, pack_fringe_batch  # noqa: E402
from src.offline.tree_env import (  # noqa: E402
    UNREACHABLE_DISTANCE,
    FringeEnv,
    load_tree_instance,
)
from src.trainer import RLFrontierTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "exp/rl_exp/diversity_v2/_models/SC"
TRAIN_DATA = REPO / "exp/rl_exp/batch0/_models/SC/training_data"
BINDERS = ["SC_10_8__pl_15", "SC_9_11__pl_8"]
SEEDS = [0, 1, 2]
FRINGE_SIZE = 32
N_ENV_SEEDS = 12
PER_BINDER = 200


def load_binder(name: str):
    csv = TRAIN_DATA / name / f"{name}_depth_25.csv"
    cache_file = TRAIN_DATA / name / "graph_cache_offline_v1.pt"
    if not csv.exists() or not cache_file.exists():
        return None
    inst = load_tree_instance(csv)
    cache = InstanceCache.from_paths(inst.state_paths_abs(REPO), cache_file, verbose=False)
    return inst, cache


def collect_fringes(inst, fringe_size: int, cap: int):
    seen, out = set(), []
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


def on_path(inst, s: int) -> bool:
    """s lies on a shortest root->goal path iff depth(s)+d*(s)==d*(root)."""
    droot = inst.distance[inst.root_id]
    if droot >= UNREACHABLE_DISTANCE or inst.distance[s] >= UNREACHABLE_DISTANCE:
        return False
    return inst.depth[s] + inst.distance[s] == droot


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


def main() -> None:
    # Collect fringes per binder, keep (inst, cache, state_ids).
    fringes = []
    for name in BINDERS:
        loaded = load_binder(name)
        if loaded is None:
            print(f"[skip] binder {name}: csv/cache absent")
            continue
        inst, cache = loaded
        snaps = collect_fringes(inst, FRINGE_SIZE, PER_BINDER)
        for fr in snaps:
            fringes.append((inst, cache, fr))
        print(f"  {name}: {len(snaps)} multi-candidate fringes "
              f"(n_states={inst.n_states}, root_d*={inst.distance[inst.root_id]})")
    if not fringes:
        print("[skip] no fringes collected; nothing to audit.")
        return

    # ---- 1 & 2: per-element label coverage ----
    n_elem = sum(len(fr) for _, _, fr in fringes)
    finite_d = sum(
        1 for inst, _, fr in fringes for s in fr
        if inst.distance[s] < UNREACHABLE_DISTANCE
    )
    print(f"\n[1] d* coverage: {finite_d}/{n_elem} fringe elements have EXACT d* "
          f"({100 * finite_d / n_elem:.1f}%); {n_elem - finite_d} unreachable (d*=inf).")
    print("[2] available per-element labels + coverage:")
    print(f"      d* (distance-to-goal)     : {100 * finite_d / n_elem:5.1f}%  (table 'Distance From Goal')")
    print("      depth (from root)         : 100.0%  (table 'Depth')")
    print(f"      on-optimal-path flag      : {100 * finite_d / n_elem:5.1f}%  (derived: depth+d*==root_d*; needs finite d*)")
    print("      goal flag                 : 100.0%  (table 'Goal')")
    print("      planner f-value / expand-order : ABSENT  (not a table column)")
    print("      subtree size              : derivable (tree walk), not stored")

    # ---- 3: within-fringe target quality (fringes with >=2 finite-d* elems) ----
    usable = []  # (inst, cache, fr, d-array)
    for inst, cache, fr in fringes:
        d = np.array([inst.distance[s] for s in fr], dtype=float)
        if (d < UNREACHABLE_DISTANCE).sum() >= 2:
            usable.append((inst, cache, fr, d))
    print(f"\n[3] within-fringe target quality (on {len(usable)}/{len(fringes)} "
          f"fringes with >=2 finite-d* elements):")
    uniq_best = 0
    onpath_present = onpath_unique = 0
    for inst, _, fr, d in usable:
        dmin = d.min()
        uniq_best += int((d == dmin).sum() == 1)
        npath = sum(1 for s in fr if on_path(inst, s))
        onpath_present += int(npath >= 1)
        onpath_unique += int(npath == 1)
    nu = len(usable)
    print(f"      unique argmin-d* (clean best) : {uniq_best}/{nu} ({100 * uniq_best / nu:.1f}%); "
          f"ties at the min on the rest")
    print(f"      >=1 on-path element present   : {onpath_present}/{nu} ({100 * onpath_present / nu:.1f}%)")
    print(f"      exactly-1 on-path element     : {onpath_unique}/{nu} ({100 * onpath_unique / nu:.1f}%)")

    # current-model argmax vs oracle (argmin-d*) — warm-start gap
    have_models = all((MODELS / f"seed{s}_fringe{FRINGE_SIZE}" / "best_by_expansions.pt").exists()
                      for s in SEEDS)
    if have_models:
        print("      current-model argmax == oracle argmin-d* (warm-start gap):")
        for s in SEEDS:
            model = RLFrontierTrainer.load_model(
                MODELS / f"seed{s}_fringe{FRINGE_SIZE}" / "best_by_expansions.pt", device="cpu"
            )
            match = 0
            for inst, cache, fr, d in usable:
                v = score(model, cache, fr)
                # oracle best = argmin d*; ties => match if model picks any min
                match += int(d[int(np.argmax(v))] == d.min())
            print(f"        seed{s}_f{FRINGE_SIZE}: {match}/{nu} ({100 * match / nu:.1f}%)")
    else:
        print("      (current-model warm-start gap: SKIP — v2 seed models absent)")

    # ---- call ----
    cov = finite_d / n_elem
    clean = onpath_present / nu if nu else 0
    print("\nCALL — supervision available:")
    if cov >= 0.95:
        print(f"  d* is ~fully available ({cov:.0%}) and precomputed (no re-expansion). "
              "Supports POINTWISE -d* regression (simplest, no bootstrap) AND "
              "LISTWISE softmax-CE on the argmin-d* element. Unreachable (~few %) "
              "carry d*=inf = a valid 'worst' label.")
    else:
        print(f"  d* partial ({cov:.0%}); the on-path binary "
              f"(present on {clean:.0%} of fringes) is the robust fallback -> "
              "LISTWISE softmax-CE on on-path, or PAIRWISE margin.")


if __name__ == "__main__":
    main()
