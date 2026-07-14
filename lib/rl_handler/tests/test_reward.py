"""Tests 2 and 3 — the reward fix, and the gamma/K_max separation assertion.

THE BUG BEING FIXED
The previous env gave a dead end "terminal with no bonus", so a failure returned
-m where m is however many steps you survived: fail after 3 -> -3, succeed after
20 -> -20. Under that reward FAILING FAST IS OPTIMAL, and offline there is no
exploration to discover that this is an artifact -- the critic learns exactly it.

    r(s,a) = 0            if SUCC
           = -1/(1-gamma) if DOOM   (absorbing, done=True)
           = -1           otherwise
"""

from __future__ import annotations

import math

import pytest

from src.offline.env import (
    assert_gamma_separates,
    doom_penalty,
    g_doom,
    g_succ,
)
from src.offline.tree import max_success_expansions

GAMMAS = (0.9, 0.95, 0.99, 0.995)


def _horizon(gamma: float) -> int:
    """1/(1-gamma) -- the longest K_max this gamma is allowed to be used with."""
    return int(1.0 / (1.0 - gamma))


def _return_of_doom_after(m: int, gamma: float, reward_mode: str) -> float:
    """Exactly the reward sequence the env emits for an episode that dooms on
    its m-th transition: (m-1) steps of -1, then the terminal doom reward."""
    g = 0.0
    disc = 1.0
    for _ in range(m - 1):
        g += disc * -1.0
        disc *= gamma
    return g + disc * doom_penalty(gamma, reward_mode)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_g_doom_independent_of_when_it_happens(gamma):
    """The whole point: no incentive to fail early."""
    ref = g_doom(gamma)
    for m in range(1, 60):
        got = _return_of_doom_after(m, gamma, "absorbing")
        assert math.isclose(got, ref, rel_tol=1e-9), (
            f"gamma={gamma} m={m}: doom return {got} != {ref}; the return of a "
            f"failure must not depend on how long you survived."
        )
    assert math.isclose(ref, -1.0 / (1.0 - gamma))


@pytest.mark.parametrize("gamma", GAMMAS)
def test_any_success_beats_any_failure(gamma):
    """G_succ(k) > G_doom for every k within the horizon this gamma is valid for.

    Mathematically the strict inequality holds for ANY finite k, since
    G_succ(k) - G_doom = gamma^k/(1-gamma) > 0. In float it holds only while
    gamma^k has not underflowed relative to 1 -- see the saturation test below,
    which is exactly why assert_gamma_separates exists.
    """
    for k in range(1, _horizon(gamma) + 1):
        assert g_succ(k, gamma) > g_doom(gamma), f"gamma={gamma} k={k}"


@pytest.mark.parametrize("gamma", GAMMAS)
def test_g_succ_strictly_decreasing_in_k(gamma):
    """The objective is fewest expansions."""
    for k in range(1, _horizon(gamma) + 1):
        assert g_succ(k + 1, gamma) < g_succ(k, gamma), f"gamma={gamma} k={k}"


@pytest.mark.parametrize("gamma", GAMMAS)
def test_returns_saturate_beyond_the_horizon(gamma):
    """WHY 1/(1-gamma) > K_max is required.

    Far beyond the horizon, gamma^k underflows against 1 and G_succ(k) becomes
    float-INDISTINGUISHABLE from G_doom: a success and a failure carry the same
    return, and the critic cannot tell them apart. The requirement is about this
    numerical separation, not about ordering (which never breaks in exact math).
    """
    k = 50 * _horizon(gamma)
    assert g_succ(k, gamma) == g_doom(gamma), (
        f"expected float saturation at k={k} for gamma={gamma}"
    )
    # ...and it does NOT saturate inside the horizon, which is the point.
    assert g_succ(_horizon(gamma), gamma) > g_doom(gamma)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_doom_penalty_is_the_delta_to_infinity_limit(gamma):
    """The penalty is not a free hyperparameter: it is the value of never
    reaching a goal, i.e. lim_{k->inf} G_succ(k) under gamma^inf = 0."""
    assert math.isclose(g_succ(10_000, gamma), doom_penalty(gamma), rel_tol=1e-6)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_legacy_reward_makes_failing_fast_optimal(gamma):
    """F6's premise, asserted rather than asserted-in-a-comment: under legacy the
    return of a failure is -m, so dooming sooner scores strictly better."""
    returns = [_return_of_doom_after(m, gamma, "legacy") for m in range(1, 20)]
    assert all(a > b for a, b in zip(returns, returns[1:])), (
        "legacy doom return must strictly decrease in m -- that IS the bug"
    )
    with pytest.raises(ValueError, match="m-dependent"):
        g_doom(gamma, "legacy")


# ------------------------------------------------------ test 3: gamma/K_max ---

def test_gamma_assertion_fails_loudly_when_too_small():
    with pytest.raises(ValueError, match="too small"):
        assert_gamma_separates(gamma=0.9, k_max=34)   # horizon 10 < 34
    # and the message tells you what to do about it
    try:
        assert_gamma_separates(gamma=0.9, k_max=34)
    except ValueError as e:
        assert "K_max" in str(e) and "Fix: gamma >" in str(e)


def test_gamma_assertion_passes_when_horizon_clears_k_max():
    assert_gamma_separates(gamma=0.99, k_max=34)      # horizon 100 > 34
    assert_gamma_separates(gamma=0.99, k_max=99)
    with pytest.raises(ValueError):
        assert_gamma_separates(gamma=0.99, k_max=100)  # horizon 100, not > 100


def test_gamma_assertion_against_real_k_max(shipped_instances):
    """K_max measured from the data, NOT from a worst-case node count.

    max delta(root) over the shipped tables is 34 (CC). The worst-case internal
    node count is ~4.5k, which would force gamma ~ 0.9998 for no benefit -- that
    bound is useless and is deliberately not what we assert.
    """
    insts = list(shipped_instances.values())
    k_max = max_success_expansions(insts)
    assert k_max == 34, f"expected K_max=34 (CC delta(root)), got {k_max}"
    assert_gamma_separates(gamma=0.99, k_max=k_max)   # the default gamma clears it
    with pytest.raises(ValueError):
        assert_gamma_separates(gamma=0.95, k_max=k_max)  # horizon 20 < 34


def test_k_max_undefined_without_a_solvable_instance():
    from conftest import make_tree
    with pytest.raises(ValueError, match="no solvable instance"):
        max_success_expansions([make_tree([[1], []], goals=[], name="goalless")])
