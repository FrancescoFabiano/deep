"""Unit tests for src/offline/regimes.select_beam (P1 Task 1).

Guarantees the beam-size invariant across all five regimes in all three pool
regimes (|P| < F, |P| == F, |P| > F):
  - returns EXACTLY min(F, |P|) ids, unique, all drawn from the pool;
  - when |P| <= F every regime returns the WHOLE pool (no selection) — this is
    the hfs_m1 small-pool regression guard;
  - when |P| > F the regime rule is honored (DFS preorder front-window, BFS
    (depth,id) front-window, HFS exact count, random = uniform F-subset).

Run from lib/rl_handler:  ../../.venv/bin/python tests/test_select_beam.py
(also discoverable as test_* functions by pytest.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.regimes import (  # noqa: E402
    REGIMES,
    HFSDiag,
    UNREACHABLE_DISTANCE,
    select_beam,
)


class _FakeInstance:
    """Minimal stand-in exposing only what select_beam reads: .distance/.depth
    indexed by state id, and .n_states."""

    def __init__(self, distance, depth):
        self.distance = list(distance)
        self.depth = list(depth)
        self.n_states = len(distance)


def _make(n, seed=0):
    """A pool of n ids with varied d* (incl. some unreachable) and depths."""
    rng = random.Random(seed)
    distance, depth = [], []
    for _ in range(n):
        if rng.random() < 0.15:
            distance.append(UNREACHABLE_DISTANCE)
        else:
            distance.append(float(rng.randint(1, rng.choice([1, 2, 3, 8, 40]))))
        depth.append(rng.randint(0, 25))
    inst = _FakeInstance(distance, depth)
    pool = list(range(n))
    dfs_rank = list(range(n))
    rng.shuffle(dfs_rank)
    return inst, pool, dfs_rank


def _check_basic(beam, pool, F):
    assert len(beam) == min(F, len(pool)), (
        f"beam size {len(beam)} != min(F={F}, |P|={len(pool)})"
    )
    assert len(set(beam)) == len(beam), "beam has duplicate slots"
    assert set(beam) <= set(pool), "beam contains ids outside the pool"


def test_all_regimes_all_pool_regimes():
    """min(F,|P|) unique members for every regime in |P|<F, ==F, >F."""
    for F in (4, 8, 32, 64):
        for n in (F - 1, F, F + 1, 3 * F):
            if n < 1:
                continue
            inst, pool, rank = _make(n, seed=n + F)
            for regime in REGIMES:
                diag = HFSDiag() if regime.startswith("hfs") else None
                beam = select_beam(regime, pool, inst, F, random.Random(1), rank, diag)
                _check_basic(beam, pool, F)


def test_subF_returns_whole_pool_for_every_regime():
    """|P| <= F: every regime (incl. hfs_m1) returns the WHOLE pool."""
    for F in (8, 32, 64):
        for n in (1, 2, F // 2, F - 1, F):
            inst, pool, rank = _make(n, seed=100 + n + F)
            for regime in REGIMES:
                beam = select_beam(regime, pool, inst, F, random.Random(2), rank)
                assert set(beam) == set(pool), (
                    f"{regime} did not return the whole pool at |P|={n}<=F={F}"
                )
                assert len(beam) == n


def test_hfs_m1_subF_equals_whole_pool_explicit():
    """Targeted regression: hfs_m1 == whole pool when |P| <= F (the reported bug)."""
    F = 32
    for n in (1, 5, 17, 31, 32):
        inst, pool, rank = _make(n, seed=7 * n)
        beam = select_beam("hfs_m1", pool, inst, F, random.Random(0), rank, HFSDiag())
        assert sorted(beam) == sorted(pool), f"hfs_m1 short/altered pool at |P|={n}"


def test_dfs_rule_front_window():
    """|P|>F DFS beam = the F smallest-(dfs_rank,id) pool members."""
    F = 32
    inst, pool, rank = _make(5 * F, seed=11)
    beam = select_beam("dfs", pool, inst, F, random.Random(0), rank)
    expected = sorted(pool, key=lambda s: (rank[s], s))[:F]
    assert beam == expected


def test_bfs_rule_front_window():
    """|P|>F BFS beam = the F smallest-(depth,id) pool members."""
    F = 32
    inst, pool, rank = _make(5 * F, seed=13)
    beam = select_beam("bfs", pool, inst, F, random.Random(0), rank)
    expected = sorted(pool, key=lambda s: (inst.depth[s], s))[:F]
    assert beam == expected


def test_random_rule_is_subset_sizeF():
    F = 32
    inst, pool, rank = _make(5 * F, seed=17)
    beam = select_beam("random", pool, inst, F, random.Random(0), rank)
    _check_basic(beam, pool, F)


def test_hfs_emits_exactly_F_with_heavy_unreachable():
    """Adversarial: mostly-unreachable pool, few finite buckets, |P|>F -> still F."""
    F = 32
    n = 4 * F
    rng = random.Random(3)
    # 90% unreachable, the rest concentrated in 3 small finite buckets
    distance = []
    for _ in range(n):
        distance.append(UNREACHABLE_DISTANCE if rng.random() < 0.9
                        else float(rng.choice([1, 2, 5])))
    depth = [rng.randint(0, 25) for _ in range(n)]
    inst = _FakeInstance(distance, depth)
    pool = list(range(n))
    rank = list(range(n))
    for regime in ("hfs_m0", "hfs_m1"):
        beam = select_beam(regime, pool, inst, F, random.Random(0), rank, HFSDiag())
        _check_basic(beam, pool, F)


def test_hfs_diag_records_when_pool_binds():
    """HFS diag accumulates only on |P|>F draws (the bucketing path)."""
    F = 32
    inst, pool, rank = _make(4 * F, seed=23)
    diag = HFSDiag()
    select_beam("hfs_m0", pool, inst, F, random.Random(0), rank, diag)
    assert diag.n_steps == 1 and diag.present_total > 0


def main():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} select_beam checks passed.")


if __name__ == "__main__":
    main()
