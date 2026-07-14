"""Test 2 + 3 — the reward, at gamma = 1.

WHY GAMMA = 1
The completeness proposition says every policy reaches a goal on a solvable
instance, i.e. EVERY POLICY IS PROPER. Costs are strictly positive (-1 per
expansion), the process terminates w.p. 1 under any policy, and there is no
absorbing failure state to escape into. That is the stochastic shortest path
setting (Bertsekas & Tsitsiklis): the undiscounted Bellman operator has a unique
fixed point. Discounting exists to make NON-terminating processes well-posed;
ours terminates.

    r(s,a) = 0     if SUCC
           = -1    otherwise
    V*(s)  = -delta(s)   exactly
    G_succ(k) = -k       linear in k, no saturation, ever

DOOM keeps a finite absorbing penalty but is UNREACHABLE: doom <=> delta(root) =
inf, and unsolvable instances are filtered at load.

THE BUG THAT WAS FIXED
The previous env made the cap terminal with reward -1, so a truncation returned
-m: stopping early scored better than searching. The fix is the terminated /
truncated split (see test_env_transition), not a discount.
"""

from __future__ import annotations

import warnings

import pytest

from src.offline.env import assert_gamma, doom_penalty, g_succ
from src.offline.tree import max_success_expansions


# ------------------------------------------------- gamma = 1: the objective ---

def test_g_succ_is_linear_at_gamma_one():
    """-k exactly: the return IS the negative expansion count."""
    for k in range(1, 500):
        assert g_succ(k, 1.0) == -float(k)


def test_g_succ_strictly_decreasing_at_gamma_one():
    """The objective is fewest expansions, at every scale -- no horizon."""
    for k in range(1, 5000):
        assert g_succ(k + 1, 1.0) < g_succ(k, 1.0)


def test_gamma_one_never_saturates():
    """The property gamma<1 cannot have: a 1000-expansion search and a
    2000-expansion one differ by exactly 1000, not by ~0."""
    assert g_succ(2000, 1.0) - g_succ(1000, 1.0) == -1000.0


def test_gamma_one_is_accepted_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_gamma(1.0)


def test_discounting_truncates_the_objective():
    """The distortion gamma=1 avoids, made concrete.

    At gamma=0.99 the effective horizon is 100 expansions. A 1000-expansion
    search and a 2000-expansion one both return ~-100: the critic is
    STRUCTURALLY unable to represent the difference between a slow search and a
    hopeless one. We are minimising expansions; gamma<1 stops counting them.
    """
    slow = g_succ(1000, 0.99)
    hopeless = g_succ(2000, 0.99)
    assert abs(slow - hopeless) < 0.01, "expected discounted returns to collapse"
    # ...whereas undiscounted they are 1000 apart.
    assert abs(g_succ(1000, 1.0) - g_succ(2000, 1.0)) == 1000.0

    # And the real baseline spread on CC (oracle 34 ... bfs 231) is compressed:
    assert abs(g_succ(231, 0.99) - g_succ(1000, 0.99)) < 10.0


def test_gamma_below_one_warns_loudly():
    with pytest.warns(RuntimeWarning, match="TRUNCATES the objective"):
        assert_gamma(0.99)
    with pytest.warns(RuntimeWarning):
        assert_gamma(0.999)


def test_gamma_out_of_range_is_rejected():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="gamma must be in"):
            assert_gamma(bad)


# ----------------------------------------------------------- doom penalty ----

def test_doom_penalty_is_finite_and_budget_shaped():
    """-1/(1-gamma) is +inf at gamma=1 and is gone with the discounting it
    belonged to. -expansion_cap is finite, defensible as 'the worst cost the
    budget admits', and unreachable by construction."""
    assert doom_penalty(2000) == -2000.0
    assert doom_penalty(500) == -500.0


def test_doom_penalty_is_worse_than_any_admissible_success():
    """Any success within the budget must beat doom."""
    cap = 2000
    for k in range(1, cap):
        assert g_succ(k, 1.0) > doom_penalty(cap)


def test_legacy_reward_mode_is_the_known_broken_one():
    assert doom_penalty(2000, "legacy") == -1.0
    with pytest.raises(ValueError, match="reward_mode"):
        doom_penalty(2000, "nonsense")


# ---------------------------------------------------------- K_max from data ---

def test_k_max_is_measured_from_data_not_worst_case(shipped_instances):
    """max delta(root) over the shipped tables. At gamma=1 this no longer
    constrains anything -- it is reported as a data statistic only."""
    assert max_success_expansions(list(shipped_instances.values())) == 34


def test_k_max_undefined_without_a_solvable_instance():
    from conftest import make_tree
    with pytest.raises(ValueError, match="no solvable instance"):
        max_success_expansions([make_tree([[1], []], goals=[], name="goalless")])
