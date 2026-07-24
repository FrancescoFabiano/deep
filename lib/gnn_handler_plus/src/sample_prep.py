"""Target preparation overrides: dynamic MAX_DEPTH + unreachable-as-max.

Diverges from baseline `src.utils.prepare_samples` in two flag-controlled
ways (see README):

  dynamic_max_depth  : MAX_DEPTH = ceil(1.1 x max raw REACHABLE train
                       distance) instead of the hardcoded 50.
  include_unreachable: sentinel-distance states (unreachable_state_value)
                       are kept with target = MAX_DEPTH instead of being
                       filtered out, so dead ends train toward the TOP of
                       the output range rather than being a blind spot the
                       model fills with near-zero predictions.

The returned params dict carries everything needed to reconstruct the
scaling (slope, intercept, max_depth, raw counts) and flows into
distance_estimator_C.txt / history_losses.json via the plus entry point.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch

# Baseline scaling helper (same f(value) = value * slope + intercept).
from src.utils import f

MIN_V_NN = 1e-3
MAX_V_NN = 1 - MIN_V_NN
HARDCODED_MAX_DEPTH = 50  # baseline value, used when dynamic_max_depth=False


def count_unreachable(samples: List[Dict], unreachable_state_value: int) -> int:
    return sum(
        1 for s in samples if s["target"].item() == unreachable_state_value
    )


def prepare_samples_plus(
    t_s_copy: List[Dict],
    t_t_copy: List[Dict],
    unreachable_state_value: int,
    *,
    dynamic_max_depth: bool = True,
    include_unreachable: bool = True,
    unreachable_cap: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Scale targets into (0, 1); returns (train, test, params).

    Mirrors the baseline contract of prepare_samples() but:
      * optionally keeps unreachable states (target := MAX_DEPTH);
      * if unreachable states would exceed `unreachable_cap` of the final
        TRAIN set, they are subsampled down to the cap (deterministic on
        `seed`) — swapping a dead-end blind spot for a new majority-class
        imbalance would repeat the distance-0 mistake;
      * optionally derives MAX_DEPTH from the data.
    """
    n_ur_train = count_unreachable(t_s_copy, unreachable_state_value)
    n_ur_test = count_unreachable(t_t_copy, unreachable_state_value)
    counts = {
        "train_total": len(t_s_copy),
        "train_unreachable": n_ur_train,
        "test_total": len(t_t_copy),
        "test_unreachable": n_ur_test,
    }
    print(
        f"[plus] unreachable states: train {n_ur_train}/{len(t_s_copy)} "
        f"({100 * n_ur_train / max(len(t_s_copy), 1):.2f}%), "
        f"test {n_ur_test}/{len(t_t_copy)} "
        f"({100 * n_ur_test / max(len(t_t_copy), 1):.2f}%)"
    )

    if include_unreachable:
        cap_n = int(unreachable_cap * len(t_s_copy))
        if n_ur_train > cap_n:
            rng = random.Random(seed)
            ur = [s for s in t_s_copy
                  if s["target"].item() == unreachable_state_value]
            keep = set(id(s) for s in rng.sample(ur, cap_n))
            t_s_copy = [
                s for s in t_s_copy
                if s["target"].item() != unreachable_state_value
                or id(s) in keep
            ]
            print(f"[plus] unreachable capped: {n_ur_train} -> {cap_n} "
                  f"({100 * unreachable_cap:.0f}% of train)")
            counts["train_unreachable_kept"] = cap_n
        else:
            counts["train_unreachable_kept"] = n_ur_train
        # Test set is never capped: metrics should see the true mix.
    else:  # baseline behavior: filter the sentinel states out entirely
        t_s_copy = [s for s in t_s_copy
                    if s["target"].item() != unreachable_state_value]
        t_t_copy = [s for s in t_t_copy
                    if s["target"].item() != unreachable_state_value]
        counts["train_unreachable_kept"] = 0

    reachable_train = [
        s["target"].item() for s in t_s_copy
        if s["target"].item() != unreachable_state_value
    ]
    max_train_dist = max(reachable_train)
    if dynamic_max_depth:
        # 10% headroom over the deepest observed REACHABLE train distance.
        max_depth = math.ceil(1.1 * max_train_dist)
    else:
        max_depth = HARDCODED_MAX_DEPTH

    slope = (MAX_V_NN - MIN_V_NN) / max_depth

    params = {
        "slope": slope,
        "intercept": MIN_V_NN,
        "max_depth": max_depth,
        "max_train_distance": max_train_dist,
        **counts,
    }

    for split in (t_s_copy, t_t_copy):
        for s in split:
            v = s["target"].item()
            if v == unreachable_state_value:
                v = max_depth  # dead ends train toward the top of the range
            s["target"] = torch.tensor(
                f(v, slope, MIN_V_NN), dtype=torch.float
            )

    return t_s_copy, t_t_copy, params
