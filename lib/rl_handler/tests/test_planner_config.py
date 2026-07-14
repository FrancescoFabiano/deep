"""The deployment-config invariant that makes the offline MDP faithful.

int(F * RL_exploitation / 100) must be 1 for every F in the sweep, or the planner
rescores only every RL_node_to_add successors and the offline env -- which models
one expansion per ONNX call -- is a wrong model of it.
"""

from __future__ import annotations

import pytest

from src.offline.planner_config import (
    F_MAX,
    F_MIN,
    assert_one_expansion_per_call,
    assert_rl_heuristics,
    exploitation_for,
    exploration_nodes,
    planner_argv,
    planner_flags,
    rl_node_to_add,
)

SWEEP = [4, 8, 16, 32, 64]


@pytest.mark.parametrize("F", SWEEP)
def test_one_expansion_per_onnx_call_for_every_sweep_F(F):
    e = exploitation_for(F)
    assert rl_node_to_add(F, e) == 1, (
        f"F={F} exploitation={e} gives RL_node_to_add="
        f"{rl_node_to_add(F, e)}; the env models 1 expansion per ONNX call"
    )
    assert_one_expansion_per_call(F, e)


def test_the_mapping_matches_the_agreed_table():
    assert {F: exploitation_for(F) for F in SWEEP} == {4: 25, 8: 13, 16: 7, 32: 4, 64: 2}


@pytest.mark.parametrize("F", SWEEP)
def test_cpp_sum_constraint_holds(F):
    """ArgumentParser.cpp:113 -- exploration + exploitation must be < 100."""
    f = planner_flags(F)
    assert f["RL_exploration"] + f["RL_exploitation"] < 100


@pytest.mark.parametrize("F", SWEEP)
def test_exploration_slots_are_inert(F):
    """--RL_exploration 0 => exploration_nodes 0. In RNG mode all refill is
    random anyway, so this only removes a confound."""
    f = planner_flags(F)
    assert exploration_nodes(F, f["RL_exploration"]) == 0


def test_cpp_defaults_would_break_the_env():
    """The reason this module exists: the shipped defaults are F=32,
    exploitation=70 -> the planner expands ~22/b nodes per ONNX call."""
    assert rl_node_to_add(32, 70) == 22
    with pytest.raises(ValueError, match="one expansion per ONNX call"):
        assert_one_expansion_per_call(32, 70)


def test_mismatch_error_prints_both_numbers():
    try:
        assert_one_expansion_per_call(32, 70)
    except ValueError as e:
        m = str(e)
        assert "--RL_fringe_size   = 32" in m
        assert "--RL_exploitation  = 70" in m
        assert "must be 1" in m
        assert "--RL_exploitation 4" in m, "must say how to fix it"


def test_rl_node_to_add_mirrors_cpp_float_truncation():
    """Configuration.cpp:128 -- static_cast<int>(F * exploitation / 100.0)."""
    assert rl_node_to_add(4, 25) == 1      # exactly 1.00
    assert rl_node_to_add(8, 13) == 1      # 1.04
    assert rl_node_to_add(64, 2) == 1      # 1.28
    assert rl_node_to_add(4, 49) == 1      # 1.96, still 1
    assert rl_node_to_add(4, 50) == 2      # 2.00
    assert rl_node_to_add(8, 12) == 0      # 0.96 -> fires on EVERY expansion


def test_F_outside_the_feasible_range_is_rejected():
    """RL_node_to_add == 1 needs an integer e with 100 <= F*e <= 199, which only
    has a solution for F in [2, 199]."""
    assert (F_MIN, F_MAX) == (2, 199)
    with pytest.raises(ValueError, match="cannot give one expansion"):
        exploitation_for(1)
    with pytest.raises(ValueError, match="cannot give one expansion"):
        exploitation_for(200)
    exploitation_for(199)  # the boundary is feasible


def test_only_rng_refill_is_accepted():
    assert_rl_heuristics("rng")
    assert_rl_heuristics("RNG")
    for bad in ("min", "MAX", "avg"):
        with pytest.raises(ValueError, match="RefillMode::HEURISTIC"):
            assert_rl_heuristics(bad)


@pytest.mark.parametrize("F", SWEEP)
def test_planner_argv_is_the_exact_cpp_launch(F):
    argv = planner_argv(F)
    assert argv[:2] == ["--RL_fringe_size", str(F)]
    assert "--RL_heuristics" in argv and argv[argv.index("--RL_heuristics") + 1] == "RNG"
    assert argv[argv.index("--RL_exploitation") + 1] == str(exploitation_for(F))
    assert argv[argv.index("--RL_exploration") + 1] == "0"


def test_env_refuses_a_config_that_breaks_the_invariant(t19):
    from src.offline.env import FringeEnv
    with pytest.raises(ValueError, match="one expansion per ONNX call"):
        FringeEnv(t19, fringe_size=32, exploitation=70)
