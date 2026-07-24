"""Test 2 + 3 — the reward.

DEFAULT gamma = 0.9999 (the paper's discounted form), with gamma = 1 kept as the
SSP ablation. The two are numerically ~identical here and BOTH are defensible:

  gamma = 1 (SSP): every policy is proper (completeness proposition -> every policy
    reaches a goal on a solvable instance), costs are strictly positive, the process
    terminates w.p.1, no absorbing failure to escape into. Stochastic-shortest-path
    (Bertsekas & Tsitsiklis): the undiscounted operator has a unique fixed point,
    V*(s) = -delta(s) EXACTLY, G_succ(k) = -k linear.

  gamma = 0.9999 (default): the paper's discounted reward. r=-1/step and gamma give
    G_succ(k) = -(1-gamma^k)/(1-gamma), V*(s) = -(1-gamma^delta)/(1-gamma). For
    delta << horizon (=10000) these equal the SSP limit to <0.2%. A presentation/
    conformance choice, numerically ~= gamma=1, NOT a change of objective.

    r(s,a) = 0                        SUCCESS
           = -1/(1-gamma)             GENUINE DOOM (absorbing, terminated)
           = -1                       every non-terminal step, INCLUDING a TIMEOUT
                                      (timeout is truncation: -1 + bootstrap)

DOOM is UNREACHABLE on the gated pool: doom <=> delta(root)=inf, filtered at load.
DOMINANCE invariant (assert_gamma): expansion_cap < 1/(1-gamma), so a success at
the cap still dominates doom -- past the horizon the objective saturates.

THE BUG THAT WAS FIXED (still relevant)
The previous env made the cap terminal with reward -1, so a truncation returned -m:
stopping early scored better than searching. The fix is the terminated / truncated
split (see test_env_transition), NOT a discount -- and a TIMEOUT must never get the
doom penalty (it bootstraps).
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


def test_gamma_default_is_the_discounted_paper_form():
    """DEFAULT_GAMMA=0.9999: the paper's gamma<1 formulation, numerically ~= the
    gamma=1 SSP limit (V*=-delta to <0.2% for delta << horizon)."""
    from src.offline.env import DEFAULT_GAMMA
    assert DEFAULT_GAMMA == 0.9999


def test_dominance_check_raises_when_cap_reaches_the_horizon():
    """assert_gamma encodes the invariant a success must dominate doom:
    expansion_cap < 1/(1-gamma). Past the horizon the objective saturates and a slow
    search is indistinguishable from a hopeless one. gamma=0.99 -> horizon 100, so a
    cap of 2000 must fail loudly rather than silently flatten the objective."""
    with pytest.raises(ValueError, match="horizon"):
        assert_gamma(0.99, expansion_cap=2000)      # horizon 100 < cap 2000
    assert_gamma(0.9999, expansion_cap=2000)         # horizon 10000 > cap 2000: OK
    assert_gamma(0.99, expansion_cap=50)             # horizon 100 > cap 50: OK
    assert_gamma(0.9999)                             # no cap given -> gamma-only check


def test_gamma_out_of_range_is_rejected():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="gamma must be in"):
            assert_gamma(bad)


def test_gamma_one_stays_valid_as_the_ssp_ablation():
    """gamma=1 is the well-posed SSP objective (every policy proper -> V*=-delta),
    not a hack. It must remain a valid ablation axis, and its cap-based doom means no
    dominance check applies (the -1/(1-gamma) horizon is infinite)."""
    assert_gamma(1.0)
    assert_gamma(1.0, expansion_cap=2000)


# ----------------------------------------------------------- doom penalty ----

def test_doom_penalty_is_the_paper_absorbing_return_at_gamma_below_one():
    """gamma<1: doom = -1/(1-gamma), the paper's absorbing-failure value."""
    assert doom_penalty(2000, 0.9999) == pytest.approx(-10000.0)
    assert doom_penalty(2000, 0.999) == pytest.approx(-1000.0)


def test_doom_penalty_is_finite_cap_at_gamma_one():
    """gamma=1: -1/(1-gamma) is -inf, so the SSP ablation uses -expansion_cap as a
    finite stand-in for 'the worst cost the budget admits'."""
    assert doom_penalty(2000, 1.0) == -2000.0
    assert doom_penalty(500, 1.0) == -500.0


def test_doom_penalty_is_worse_than_any_admissible_success():
    """Any success within the budget must beat doom, at the deployed gamma."""
    cap, g = 2000, 0.9999
    dp = doom_penalty(cap, g)
    for k in (1, 24, 100, 1000, cap - 1):
        assert g_succ(k, g) > dp, f"success@{k} must dominate doom"


def test_legacy_reward_mode_is_the_known_broken_one():
    assert doom_penalty(2000, 0.9999, "legacy") == -1.0
    with pytest.raises(ValueError, match="reward_mode"):
        doom_penalty(2000, 0.9999, "nonsense")


# ---------------------------------------------------------- K_max from data ---

def test_k_max_is_measured_from_data_not_worst_case(shipped_instances):
    """K_max is MEASURED from the tables, not assumed. At gamma=1 it constrains
    nothing -- it is reported as a data statistic only.

    Asserted as an invariant, not a literal: this pinned 34, which is one seed's
    delta_root on one discard=0.4-era table, not a property of the instances (the
    tree is a seed-dependent DFS sample -- delta_root moved 14/6/7 across seeds on
    CC_2_2_3__pl_4). What must hold is that K_max IS the max delta_root over the
    solvable instances, whatever the tables happen to be.
    """
    insts = list(shipped_instances.values())
    k = max_success_expansions(insts)
    expected = max(i.delta_root for i in insts if i.solvable())
    assert k == expected, "K_max must be the max delta_root over solvable instances"
    assert float(k).is_integer() and k >= 0


def test_k_max_undefined_without_a_solvable_instance():
    from conftest import make_tree
    with pytest.raises(ValueError, match="no solvable instance"):
        max_success_expansions([make_tree([[1], []], goals=[], name="goalless")])


# ================================================================================
#  gamma = 0.9999: the paper's discounted reward (B). The six required properties.
# ================================================================================

def _pool():
    """The gated batch3 CC instances -- real trees for the pool-level checks."""
    from pathlib import Path
    from src.offline.tree import load_tree_instance
    root = Path(__file__).resolve().parents[3] / "exp/rl_exp/batch3/_models/CC/training_data"
    csvs = sorted(root.glob("*/*_depth_25.csv"))
    if not csvs:
        pytest.skip("batch3 CC training data not present")
    return [load_tree_instance(p, name=p.parent.name, kind_of_data="separated")
            for p in csvs]


def test_B_success_at_k_returns_the_paper_formula():
    """r=-1/step summed under gamma gives -(1-gamma^k)/(1-gamma)."""
    g = 0.9999
    for k in (1, 3, 24, 100, 2000):
        assert g_succ(k, g) == pytest.approx(-(1 - g ** k) / (1 - g))
    # numerically ~= the gamma=1 limit for small k (the whole point)
    assert g_succ(24, g) == pytest.approx(-24, abs=0.05)


def test_B_genuine_doom_returns_minus_one_over_one_minus_gamma():
    assert doom_penalty(2000, 0.9999) == pytest.approx(-10000.0)


def test_B_timeout_is_minus_one_and_bootstraps_UNCHANGED():
    """PIN so no one 'fixes' a truncated timeout into a terminal doom later.
    A cap hit returns the ordinary step -1, terminated=False, truncated=True -- its
    value comes from the bootstrapped continuation, not a catastrophic penalty.

    Driven directly at the env: step until the cap fires, then inspect that exact
    StepResult (a tight cap on a real tree forces the truncation branch, env.py:583).
    """
    from src.offline.env import FringeEnv
    from src.offline.policies import make_policy
    inst = _pool()[0]
    env = FringeEnv(inst, fringe_size=4, seed=0, expansion_cap=3)
    pol = make_policy(inst, "dfs", seed=0)
    res = env.reset(seed=0)
    saw_timeout = False
    while not res.done:
        ranking = None if env.forced else pol(list(env.fringe))
        action = env.forced_action if env.forced else ranking[0]
        res = env.step(action, ranking)
        if res.info["outcome"] == "timeout":
            saw_timeout = True
            assert res.reward == -1.0, "timeout reward must be the step -1, not doom"
            assert res.terminated is False, "timeout must NOT be terminated"
            assert res.truncated is True, "timeout IS truncation"
            assert res.fringe, "the successor fringe must be returned to bootstrap"
    assert saw_timeout, "cap=3 on this tree must produce a timeout to pin"


def test_B_oracle_hits_regret_zero_under_the_new_reward():
    """THE correctness anchor. regret = k - delta_root is in EXPANSION units and is
    gamma-invariant, so the clairvoyant oracle must still spend exactly delta_root
    and score regret 0 -- the reward change must not perturb this."""
    from src.offline.env import FringeEnv, rollout
    from src.offline.policies import make_policy
    for inst in _pool()[:4]:
        for seed in range(3):
            env = FringeEnv(inst, fringe_size=32, seed=seed, expansion_cap=900)
            r = rollout(env, make_policy(inst, "hfs_oracle", seed=seed), seed=seed)
            assert r["solved"], f"{inst.name} seed {seed}: oracle failed"
            assert r["expansions"] == inst.delta_root
            assert r["regret"] == 0.0


def test_B_dominance_holds_on_the_actual_pool():
    """Every reachable success return dominates doom, at gamma=0.9999, cap 2000."""
    from src.offline.env import default_expansion_cap
    g = 0.9999
    insts = _pool()
    cap = default_expansion_cap(insts)
    dp = doom_penalty(cap, g)
    assert cap < 1 / (1 - g), "cap must sit below the horizon"
    for k in range(1, cap):                # every k a success could take
        assert g_succ(k, g) > dp


def test_B_v_star_uses_the_discounted_formula_and_matches_the_limit():
    """tree.v_star(gamma<1) = -(1-gamma^d)/(1-gamma); ~= -d for d << horizon."""
    inst = _pool()[0]
    root = inst.root_id
    d = inst.delta_root
    v1 = inst.v_star([root], gamma=1.0)
    vg = inst.v_star([root], gamma=0.9999)
    assert v1 == pytest.approx(-d)
    assert vg == pytest.approx(-(1 - 0.9999 ** d) / (1 - 0.9999))
    assert vg == pytest.approx(v1, abs=0.05), "discounted V* ~= SSP limit for small d"
