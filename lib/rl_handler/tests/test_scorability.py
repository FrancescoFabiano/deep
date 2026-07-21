"""Scorability is a pool criterion, and it is NOT min_states in disguise.

The regression these lock down: `min_states` was standing in for "can the ranking
metric read this instance", and it does not correlate with that in either
direction -- a 50k-state instance can yield 0 scorable frontiers while a 23-state
one yields 18. An instance with 0 trains fine but is invisible to every held-out
ranking metric and to checkpoint selection.
"""
from __future__ import annotations

import pytest

from src.offline.usability import build_usable_pool, check_instance, scorability

from conftest import make_tree


def _chain(n: int, name: str = "chain"):
    """Branchless chain: every frontier is a singleton, so nothing is rankable."""
    children = [[i + 1] if i + 1 < n else [] for i in range(n)]
    return make_tree(children, goals=[n - 1], name=name)


def _bush(name: str = "bush"):
    """Branching tree with goals and dead ends -- rankable frontiers throughout."""
    children = [
        [1, 2, 3],      # 0
        [4, 5],         # 1
        [],             # 2  dead
        [6],            # 3
        [7, 8],         # 4
        [],             # 5  dead
        [9],            # 6
        [],             # 7  goal
        [],             # 8  dead
        [],             # 9  goal
    ]
    return make_tree(children, goals=[7, 9], name=name)


def test_singleton_chain_is_metric_blind_however_large():
    """A big tree can be completely unrankable: size is not the criterion."""
    s = scorability(_chain(400))
    assert s["scorable_frontiers"] == 0
    assert s["singleton_frontier_frac"] == pytest.approx(1.0)


def test_branching_tree_is_scorable_however_small():
    """...and a small one can be rankable. The two criteria are independent."""
    s = scorability(_bush())
    assert s["scorable_frontiers"] > 0


def test_metric_blind_is_a_flag_not_an_exclusion_by_default():
    """Default must not change any existing pool's composition."""
    v = check_instance(_chain(400))
    assert v.metric_blind is True
    assert v.usable is True
    assert any("METRIC-BLIND" in r for r in v.reasons)


def test_require_scorable_promotes_it_to_an_exclusion():
    v = check_instance(_chain(400), require_scorable=True)
    assert v.metric_blind is True
    assert v.usable is False


def test_pool_counts_ranking_usable_separately_from_usable():
    # min_states=1 so the size criterion cannot interfere: the point of this test
    # is that "trains" and "can be scored" are counted independently.
    pool = build_usable_pool([_chain(400), _bush()], out_path=None, verbose=False,
                             min_states=1)
    assert pool["n_usable"] == 2            # both still train
    assert pool["n_metric_blind"] == 1      # only one can be scored
    assert pool["n_usable_for_ranking"] == 1


def test_contrastive_counts_finite_vs_inf_as_a_signal():
    """`[1.0, inf]` IS a ranking label -- the bush's frontiers mix dead ends with
    live branches. Requiring two distinct FINITE deltas instead scores CoinBox at
    a bogus 0% when its true contrast rate is 93.4%."""
    s = scorability(_bush())
    assert s["contrastive_frontiers"] > 0


def test_probe_is_deterministic():
    """No RNG in the verdict -- the retired delta_root criterion swung 14/6/7 on
    the seed alone, and a gate must not do that."""
    inst = _bush()
    assert scorability(inst) == scorability(inst)
