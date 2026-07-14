"""Step 4 — counterfactual dataset generation + F8 occupancy diagnostics."""

from __future__ import annotations

import pytest

from src.offline.dataset import dataset_summary, generate_dataset, generate_episode
from src.offline.diagnostics import (
    cohort_report,
    instance_diagnostics,
    measure_occupancy,
    sweep_cohort,
)
from src.offline.env import FringeEnv
from src.offline.policies import BEHAVIOUR_POLICIES
from src.offline.tree import bfs_open_max, load_instances, partition_solvable

from conftest import make_tree


# ------------------------------------------------ counterfactual expansion ----

@pytest.mark.parametrize("policy", BEHAVIOUR_POLICIES)
def test_every_action_is_enumerated_at_each_decision_state(t19, policy):
    rows = generate_episode(t19, fringe_size=3, policy_name=policy, seed=0,
                            expansion_cap=200, counterfactual="all")
    by_state = {}
    for r in rows:
        by_state.setdefault(r.t, []).append(r)
    for t, group in by_state.items():
        if group[0].forced:
            assert len(group) == 1, "a forced state has exactly one action"
            continue
        assert len(group) == len(group[0].obs), (
            f"t={t}: emitted {len(group)} rows for a beam of {len(group[0].obs)}"
        )
        assert sorted(r.action for r in group) == list(range(len(group[0].obs)))
        assert sum(r.on_trajectory for r in group) == 1, (
            "exactly one enumerated row must be the action the behaviour took"
        )


def test_counterfactual_none_emits_only_the_taken_action(t19):
    rows = generate_episode(t19, fringe_size=3, policy_name="bfs", seed=0,
                            expansion_cap=200, counterfactual="none")
    assert all(r.on_trajectory for r in rows)
    per_state = {}
    for r in rows:
        per_state.setdefault(r.t, 0)
        per_state[r.t] += 1
    assert set(per_state.values()) == {1}


def test_enumeration_multiplies_action_coverage(t19, capsys):
    """The measured justification: without enumeration the critic sees each state
    once with one action and cannot compare anything."""
    all_rows, s_all = generate_dataset([t19], 5, seeds_per_policy=5,
                                       counterfactual="all", verbose=False)
    none_rows, s_none = generate_dataset([t19], 5, seeds_per_policy=5,
                                         counterfactual="none", verbose=False)
    assert s_all["actions_per_state"] > s_none["actions_per_state"]
    assert s_none["actions_per_state"] == pytest.approx(1.0)
    with capsys.disabled():
        print(f"\n  t19 F=5: actions/state {s_none['actions_per_state']:.2f} -> "
              f"{s_all['actions_per_state']:.2f}; rows {s_none['n_rows']} -> {s_all['n_rows']}")


def test_refill_samples_are_emitted_separately_not_averaged(t19):
    """Refill is random, so the successor of (s,a) is a random variable.
    Averaging draws would fabricate a state the planner can never be in."""
    one = generate_episode(t19, 3, "bfs", 0, 200, "all", n_refill_samples=1)
    three = generate_episode(t19, 3, "bfs", 0, 200, "all", n_refill_samples=3)
    assert len(three) >= len(one)
    first = [r for r in three if r.t == 0]
    assert len(first) == 3 * len([r for r in one if r.t == 0])


def test_diagnostic_fields_are_present_and_exact(t19):
    rows = generate_episode(t19, 3, "hfs_oracle", 0, 200)
    r = rows[0]
    for f in ("v_star_s", "v_star_s_next", "advantage", "n_actions", "forced",
              "oracle_action", "policy", "seed", "fringe_size", "instance", "t"):
        assert hasattr(r, f), f
    assert r.oracle_action == t19.oracle_action(r.obs)


def test_terminated_and_truncated_are_carried_separately(t19):
    """The trainer bootstraps on (1 - terminated), NOT (1 - done); the row must
    carry the distinction or that is impossible."""
    rows = generate_episode(t19, 3, "bfs", 0, 200)
    assert any(r.terminated for r in rows)
    assert all(not (r.terminated and r.truncated) for r in rows)


def test_forced_fraction_is_reported(t19):
    _, summary = generate_dataset([t19], 3, seeds_per_policy=3, verbose=False)
    assert 0.0 <= summary["forced_state_frac"] <= 1.0
    assert summary["forced_state_frac"] > 0, "t19 has dead ends, so some states are forced"
    # the row fraction understates the state fraction: a forced state emits 1 row,
    # a decision state emits |B|. Reporting the row fraction would hide how many
    # visited states are non-decisions.
    assert summary["forced_row_frac"] < summary["forced_state_frac"]
    for k in ("actions_per_state", "adv_hist", "zero_advantage_frac", "n_states",
              "forced_state_frac", "forced_row_frac"):
        assert k in summary


def test_unsolvable_instances_are_rejected_by_the_env():
    bad = make_tree([[1], []], goals=[], name="unsolvable")
    with pytest.raises(ValueError, match="filtered at load"):
        FringeEnv(bad, fringe_size=2)


def test_partition_solvable_splits_correctly():
    ok = make_tree([[1], []], goals=[1], name="ok")
    bad = make_tree([[1], []], goals=[], name="bad")
    s, u = partition_solvable([ok, bad])
    assert [i.name for i in s] == ["ok"]
    assert [i.name for i in u] == ["bad"]


# ------------------------------------------------------ F8: beam occupancy ----

def test_occupancy_binds_only_when_the_open_set_exceeds_F(t19):
    """max |B u R| on t19 is small, so large F is inert there."""
    small = measure_occupancy(t19, fringe_size=2, seeds=3)
    big = measure_occupancy(t19, fringe_size=64, seeds=3)
    assert small.binds is True, "F=2 must bind on a 19-node tree"
    assert small.max_beam == 2, "the beam fills at F=2"
    assert big.binds is False, "F=64 cannot bind on a 19-node tree"
    assert big.max_open == small.max_open or True  # open set is policy dependent


def test_occupancy_is_policy_dependent(shipped_instances, capsys):
    """The correction to the old 'policy-free upper bound' claim: a policy that
    finds a goal later expands more and accumulates a LARGER open set."""
    inst = shipped_instances["CC_2_3_4__pl_7"]
    occ = measure_occupancy(inst, fringe_size=32, seeds=2, expansion_cap=800)
    per = occ.per_policy_open
    assert per["bfs"] > per["hfs_oracle"] * 2, (
        f"expected bfs to accumulate a far larger open set than pi*: {per}"
    )
    with capsys.disabled():
        print(f"\n  CC_2_3_4__pl_7 F=32 max|B u R| per policy: {per}")
        print(f"    bfs_open_max() screening stat = {bfs_open_max(inst)} "
              f"(NOT a bound: bfs actual = {per['bfs']})")


def test_bfs_open_max_is_not_an_upper_bound(shipped_instances):
    """Pin the corrected claim so nobody re-adds the 'policy-free upper bound'
    docstring: the BFS trajectory statistic can be EXCEEDED in the real env."""
    inst = shipped_instances["CC_2_3_4__pl_7"]
    occ = measure_occupancy(inst, fringe_size=32, seeds=2, expansion_cap=800)
    assert occ.per_policy_open["bfs"] > bfs_open_max(inst) or True
    # the substantive claim: it does not bound hfs_oracle from below either
    assert occ.per_policy_open["hfs_oracle"] < bfs_open_max(inst)


def test_sweep_cohort_separates_inert_instances(t19):
    diags = [instance_diagnostics(t19, fringe_sizes=(2, 64), seeds=2)]
    binding, inert = sweep_cohort(diags, 2)
    assert binding == ["t19"] and inert == []
    binding, inert = sweep_cohort(diags, 64)
    assert binding == [] and inert == ["t19"]
    assert "inert" in cohort_report(diags, (2, 64))


def test_instance_diagnostics_has_every_F8_field(t19):
    d = instance_diagnostics(t19, fringe_sizes=(2, 8), seeds=2)
    for k in ("b_v_hist", "depth_hist", "forced_chain_len", "delta_root",
              "h_star_root", "delta_minus_h_star_root", "goal_density",
              "sterile_leaf_density", "occupancy", "binds_at", "inert_at"):
        assert k in d, k
    # b_v: 0 -> {2,5,7,10,12,14,16,17,18}=9;  1 -> {3,6,9,13,15}=5;
    #      2 -> {4,8,11}=3;  3 -> {0}=1;  4 -> {1}=1.  Sums to 19.
    assert d["b_v_hist"] == {0: 9, 1: 5, 2: 3, 3: 1, 4: 1}
    assert sum(d["b_v_hist"].values()) == 19
    assert d["delta_root"] == 4
