"""Model selection + the three deployment gates."""

from __future__ import annotations

import json

import pytest

from src.offline.selection import (
    Candidate,
    assert_within_config,
    config_of,
    gate_beats_baselines,
    gate_env_fidelity,
    onnx_path_for,
    select,
    split_instances,
    write_selection_sidecar,
)


# ------------------------------------------------------ the guardrail --------

def test_config_of():
    assert config_of("CC_2_3_4__pl_7") == "CC_2_3_4"
    assert config_of("SC_R_10_10__pl_10") == "SC_R_10_10"


def test_cross_configuration_splits_are_rejected():
    """THE trap: node ids are fluent-set hashes, so two configurations share NO
    vocabulary (0.0% overlap measured) and the node channel is pure noise."""
    with pytest.raises(ValueError, match="cross-configuration"):
        assert_within_config(["CC_2_2_3__pl_4"], ["CC_2_3_4__pl_7"])
    with pytest.raises(ValueError, match="0.0% id"):
        assert_within_config(["CC_2_2_3__pl_4"], ["CC_2_3_4__pl_7"])


def test_within_configuration_splits_are_accepted():
    assert_within_config(["CC_2_3_4__pl_3"], ["CC_2_3_4__pl_7"])


def test_split_instances_is_instance_level_and_within_config():
    tr, va = split_instances(
        ["CC_2_3_4__pl_3", "CC_2_3_4__pl_5", "CC_2_3_4__pl_6", "CC_2_3_4__pl_7"],
        val_frac=0.25, seed=0)
    assert set(tr) & set(va) == set(), "train and val must be disjoint"
    assert len(va) == 1 and len(tr) == 3
    with pytest.raises(ValueError, match="cross-configuration"):
        split_instances(["CC_2_3_4__pl_3", "CC_2_2_3__pl_4"], val_frac=0.5, seed=0)


def test_explicit_val_instances_honoured():
    tr, va = split_instances(["CC_2_3_4__pl_3", "CC_2_3_4__pl_7"],
                             val_instances=["CC_2_3_4__pl_7"])
    assert va == ["CC_2_3_4__pl_7"] and tr == ["CC_2_3_4__pl_3"]


def test_degenerate_split_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        split_instances(["CC_2_3_4__pl_3"], val_frac=0.99)


# -------------------------------------------------------- the metric ---------

def test_primary_is_coverage_then_regret_then_earlier():
    a = Candidate(step=1, frames=10, coverage=0.5, regret=1.0, doom=0.0)
    b = Candidate(step=2, frames=20, coverage=0.9, regret=8.0, doom=0.0)
    # coverage dominates: 100% at regret 8 beats 50% at regret 1
    assert select([a, b]) is b
    c = Candidate(step=3, frames=30, coverage=0.9, regret=2.0, doom=0.0)
    assert select([a, b, c]) is c            # same coverage -> lower regret
    d = Candidate(step=9, frames=90, coverage=0.9, regret=2.0, doom=0.0)
    assert select([d, c]) is c               # tie -> earlier checkpoint


def test_a_checkpoint_that_solved_nothing_sorts_worst():
    good = Candidate(step=2, frames=20, coverage=0.1, regret=50.0, doom=0.0)
    none = Candidate(step=1, frames=10, coverage=0.1, regret=None, doom=0.0)
    assert select([none, good]) is good, "regret=None must not win the tie-break"


def test_td_loss_is_not_a_selection_input():
    """A critic can drive its own loss to zero and produce a useless policy."""
    import inspect
    from src.offline import selection
    src = inspect.getsource(selection.Candidate)
    assert "td" not in src.lower() and "loss" not in src.lower()


def test_doom_is_not_a_tiebreak():
    """doom_rate is provably 0 on solvable data, so it can never break a tie."""
    a = Candidate(step=1, frames=10, coverage=0.9, regret=5.0, doom=0.0)
    b = Candidate(step=2, frames=20, coverage=0.9, regret=5.0, doom=0.9)
    assert select([b, a]) is a               # decided by step, not by doom


# -------------------------------------------------------- the gates ----------

def test_env_fidelity_passes_when_counts_agree():
    g = gate_env_fidelity([100, 50, 20], [100, 50, 20])
    assert g.passed and "0.000" in g.detail


def test_env_fidelity_fails_loudly_and_prints_both_traces():
    g = gate_env_fidelity([100, 50], [10, 5])
    assert not g.passed
    assert "offline=[100, 50]" in g.detail and "planner=[10, 5]" in g.detail


def test_env_fidelity_tolerates_tree_replay_drift():
    """The offline env replays a DFS spanning tree of a deduplicated DAG while the
    planner dedups live, so small divergence is expected, not a bug."""
    assert gate_env_fidelity([105, 52], [100, 50], tolerance_frac=0.10).passed
    assert not gate_env_fidelity([150, 80], [100, 50], tolerance_frac=0.10).passed


def test_env_fidelity_fails_on_empty_input():
    assert not gate_env_fidelity([], []).passed


def test_beats_baselines_excludes_the_clairvoyant_ceiling():
    """hfs_oracle ranks by delta -- the answer to the problem. Not beatable."""
    g = gate_beats_baselines(5.0, {"dfs": 10.0, "random": 20.0, "bfs": 50.0,
                                   "hfs_oracle": 0.0})
    assert g.passed, "must not require beating the clairvoyant ceiling"


def test_beats_baselines_fails_and_names_the_loser():
    g = gate_beats_baselines(60.0, {"dfs": 56.0, "random": 69.0})
    assert not g.passed and "dfs" in g.detail


def test_beats_baselines_fails_when_nothing_solved():
    assert not gate_beats_baselines(None, {"dfs": 1.0}).passed


# ------------------------------------------------------- the sidecar ---------

def test_onnx_naming_matches_what_the_cpp_and_bulk_runner_expect(tmp_path):
    p = onnx_path_for(tmp_path, "CC", 8)
    assert p.name == "frontier_policy_8.onnx"
    assert p.parent.name == "CC" and p.parent.parent.name == "_models"


def test_sidecar_records_the_planner_flags_and_gates(tmp_path):
    sel = Candidate(step=14, frames=70000, coverage=1.0, regret=6.2, doom=0.0)
    p = write_selection_sidecar(
        tmp_path / "_models" / "CC" / "frontier_policy_8.onnx", sel,
        train_instances=["CC_2_3_4__pl_3"], val_instances=["CC_2_3_4__pl_7"],
        fringe_size=8, kind_of_data="merged", model="dqn", gamma=1.0,
        reward_scale=1 / 12,
        baselines={"dfs": 56.0, "random": 69.3, "bfs": 177.3, "hfs_oracle": 0.0},
        gates=[gate_env_fidelity([100], [100]), gate_beats_baselines(6.2, {"dfs": 56.0})],
        excluded_unsolvable=[{"instance": "X", "reason": "delta(root)=inf"}],
    )
    doc = json.loads(p.read_text())
    assert doc["checkpoint"] == 14
    assert doc["configuration"] == "CC_2_3_4"
    assert doc["all_gates_passed"] is True
    # the exact C++ launch is part of the model: one expansion per ONNX call only
    # holds at RL_node_to_add == 1
    assert doc["planner_flags"]["RL_node_to_add"] == 1
    assert doc["planner_flags"]["RL_exploitation"] == 13      # F=8
    assert doc["planner_flags"]["RL_heuristics"] == "RNG"
    assert doc["val_regret_lower_bound"] == 6.2
    assert "regret" not in doc, "unqualified regret must not appear"
    assert doc["excluded_unsolvable_instances"]
    assert doc["git_sha"]


# ------------------------------------- the fidelity gate, armed for real -----

def test_a_missing_planner_count_is_a_failure_not_no_data():
    """If the planner crashed or printed nothing, that is a FAILED gate. Treating
    it as 'no data' would let a broken invocation pass silently."""
    g = gate_env_fidelity([100, 50], [100, None])
    assert not g.passed and "never 'no data'" in g.detail


def test_tiny_searches_cannot_score_the_gate():
    """A fractional tolerance is meaningless at 7 expansions: +-1 is 14%. Measured
    on the regenerated CC_2_2_3__pl_4: offline 5 vs planner 7 -> 28.6%, which says
    nothing about whether the env models the planner."""
    g = gate_env_fidelity([5], [7], min_expansions=20)
    assert not g.passed
    assert "cannot discriminate at this scale" in g.detail
    assert "harder fidelity instances" in g.detail


def test_only_instances_above_the_floor_are_scored():
    # the 5-vs-7 pair must be EXCLUDED, not averaged in
    g = gate_env_fidelity([5, 105], [7, 100], min_expansions=20, tolerance_frac=0.10)
    assert g.passed, "the trivial pair must not drag the median"
    assert "1/2 instances" in g.detail


def test_missing_binary_is_a_setup_error_not_a_silent_gate_failure(tmp_path):
    """A missing `deep` binary must RAISE. Returning "no count" would fail the gate
    and hide the real cause -- an unarmed gate must be visibly unarmed."""
    from src.offline.selection import run_planner_expansions
    with pytest.raises(FileNotFoundError, match="planner binary not found"):
        run_planner_expansions(tmp_path / "nope", tmp_path / "p.txt",
                               tmp_path / "m.onnx", 8, separated=True,
                               repo_root=tmp_path, timeout_s=5)


def test_missing_problem_file_raises(tmp_path):
    from src.offline.selection import run_planner_expansions
    (tmp_path / "deep").write_text("#!/bin/sh\n"); (tmp_path / "deep").chmod(0o755)
    with pytest.raises(FileNotFoundError, match="problem file not found"):
        run_planner_expansions(tmp_path / "deep", tmp_path / "nope.txt",
                               tmp_path / "m.onnx", 8, separated=True,
                               repo_root=tmp_path, timeout_s=5)


def test_planner_that_prints_no_count_fails_the_gate(tmp_path):
    """A crash (e.g. the 5-vs-9 input mismatch) yields no count -> gate FAILS."""
    from src.offline.selection import run_planner_expansions
    exe = tmp_path / "deep"; exe.write_text("#!/bin/sh\necho '[ERROR] boom'\n"); exe.chmod(0o755)
    prob = tmp_path / "p.txt"; prob.write_text("x")
    assert run_planner_expansions(exe, prob, tmp_path / "m.onnx", 8, separated=True,
                                  repo_root=tmp_path, timeout_s=10) is None


def test_planner_count_is_parsed(tmp_path):
    from src.offline.selection import run_planner_expansions
    exe = tmp_path / "deep"
    exe.write_text("#!/bin/sh\necho 'Goal found :)'\necho '  Nodes expanded: 9031'\n")
    exe.chmod(0o755)
    prob = tmp_path / "p.txt"; prob.write_text("x")
    assert run_planner_expansions(exe, prob, tmp_path / "m.onnx", 8, separated=True,
                                  repo_root=tmp_path, timeout_s=10) == 9031
