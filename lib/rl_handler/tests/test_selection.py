"""Model selection + the three deployment gates."""

from __future__ import annotations

import json

import pytest

from src.offline.selection import (
    Candidate,
    GateResult,
    assert_within_config,
    config_of,
    gate_beats_baselines,
    gate_env_fidelity,
    heldout_top1,
    onnx_path_for,
    select,
    select_smoothed,
    split_instances,
    split_trajectories,
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


def test_cross_config_is_opt_in_and_the_default_is_still_a_hard_raise():
    """The opt-in must not weaken the default. A caller that says nothing still
    gets the raise -- the trap is real and silent, so it stays default-on."""
    with pytest.raises(ValueError, match="cross-configuration"):
        assert_within_config(["CC_2_2_3__pl_4"], ["CC_2_3_4__pl_7"])
    with pytest.raises(ValueError, match="--allow-cross-config"):
        assert_within_config(["CC_2_2_3__pl_4"], ["CC_2_3_4__pl_7"])


def test_allow_cross_config_permits_the_split_and_warns_loudly(capsys):
    """Opting in is allowed but must never be quiet: the run is a STRUCTURE-ONLY
    FLOOR on HASHED (node channel = hash, 0.0% cross-config overlap), and the
    warning is what stops the number being reported as transfer."""
    assert_within_config(["CC_2_2_3__pl_4"], ["CC_2_3_4__pl_7"],
                         allow_cross_config=True)
    out = capsys.readouterr().out
    assert "CROSS-CONFIG RUN" in out and "exploratory" in out
    assert "FLOOR" in out, "the floor framing must be stated, not implied"


def test_split_instances_forwards_the_opt_in():
    """The flag must reach the split, not just the bare assert -- split_instances
    calls the guard itself."""
    names = ["CC_2_2_3__pl_4", "CC_2_3_4__pl_7", "CC_3_2_3__pl_5"]
    with pytest.raises(ValueError, match="cross-configuration"):
        split_instances(names, val_frac=0.34, seed=0)
    train, val = split_instances(names, val_frac=0.34, seed=0,
                                 allow_cross_config=True)
    assert sorted(train + val) == sorted(names)


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


def test_beats_baselines_top1_direction_is_higher_is_better():
    """FALLBACK gates on held-out top1, where HIGHER wins (opposite of regret).
    A model above every baseline's top1 passes; at or below any baseline it loses."""
    base = {"bfs": 0.38, "dfs": 0.20, "random": 0.29, "hfs_oracle": 1.0}
    win = gate_beats_baselines(0.42, base, higher_is_better=True, metric="heldout_top1")
    assert win.passed and "heldout_top1" in win.detail
    # RL == random (the floor null) must LOSE, not pass -- 0.29 does not beat 0.29
    floor = gate_beats_baselines(0.29, base, higher_is_better=True, metric="heldout_top1")
    assert not floor.passed and "random" in floor.detail
    # and it must NOT be required to beat the oracle's perfect 1.0
    assert "hfs_oracle" not in floor.detail


def test_beats_baselines_regret_still_defaults_to_lower_is_better():
    """PRIMARY path unchanged: regret, lower wins."""
    assert gate_beats_baselines(5.0, {"dfs": 10.0, "random": 20.0}).passed
    assert not gate_beats_baselines(20.0, {"dfs": 10.0}).passed


def test_beats_baselines_is_non_blocking():
    """A weak model is a valid, recordable result (the HASHED floor null ties random),
    so beats_baselines WARNS -- it must not abort the run. Only an armed, BLOCKING,
    failed gate fails the run."""
    lost = gate_beats_baselines(0.29, {"bfs": 0.38}, higher_is_better=True,
                                metric="heldout_top1")
    assert not lost.passed and lost.armed and lost.blocking is False
    fails = lambda gs: any(g.armed and g.blocking and not g.passed for g in gs)
    assert not fails([lost]), "a failed beats_baselines must NOT fail the run"
    # a failed env-fidelity (blocking) DOES fail the run
    blk = GateResult("env_fidelity", False, "counts disagree")   # blocking default True
    assert fails([blk])


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


def test_an_unarmed_gate_is_not_a_failed_gate():
    """A gate that did not RUN has no verdict. Conflating 'not measured' with
    'measured and failed' made the unarmed env-fidelity gate print `FAIL -- NOT
    ARMED` and fail the whole run, so disarming a gate we deliberately do not want
    (planner deployment out of scope) would abort the launcher at step 2.

    Callers must count only `armed and not passed` as failure.
    """
    unarmed = GateResult("env_fidelity", False, "NOT ARMED: no --deep-exe given",
                         armed=False)
    failed = GateResult("env_fidelity", False, "median 0.738 > tolerance 0.100")
    passed = GateResult("beats_baselines", True, "beats bfs/dfs/random")

    assert not unarmed.armed and unarmed.passed is False
    assert failed.armed and passed.armed, "gates are armed unless said otherwise"

    fails = lambda gs: any(g.armed and not g.passed for g in gs)
    assert not fails([unarmed, passed]), "an unarmed gate must NOT fail the run"
    assert fails([failed, passed]), "a genuinely failed gate MUST fail the run"


# ------------------------------------------- held-out trajectory split -------

from dataclasses import dataclass as _dc


@_dc
class _Row:
    instance: str
    policy: str
    seed: int
    obs: tuple = (0, 1)
    forced: bool = False


def _rollout_rows(instance, policy, seed, n_frontiers=3):
    """Fake one rollout: n_frontiers, each with 2 counterfactual action rows."""
    out = []
    for t in range(n_frontiers):
        for a in range(2):
            out.append(_Row(instance, policy, seed, obs=(t, t + 1)))
    return out


def _pool_rows(instances, policies=("bfs", "dfs", "hfs_oracle", "random"), seeds=(0, 1, 2)):
    from src.offline.selection import split_trajectories  # noqa
    rows = []
    for inst in instances:
        for p in policies:
            for s in seeds:
                rows += _rollout_rows(inst, p, s)
    return rows


def test_split_holds_out_whole_trajectories_no_leak():
    """THE leak test. No held-out (instance,policy,seed) trajectory may share its
    tuple with any TRAIN row -- the clean, checkable guarantee (a frontier
    composition can recur across rollouts, so we pin the rollout-level invariant)."""
    from src.offline.selection import split_trajectories
    rows = _pool_rows(["CC_2_3_4__pl_7", "CC_2_2_3__pl_4"])
    train, held, man = split_trajectories(rows, frac=0.10, seed=0)
    train_keys = {(r.instance, r.policy, r.seed) for r in train}
    held_keys = {(r.instance, r.policy, r.seed) for r in held}
    assert train_keys and held_keys
    assert train_keys.isdisjoint(held_keys), "a held-out rollout leaked into train"


def test_split_keeps_every_instance_in_both_train_and_eval():
    """Constraint: no problem dropped from training; and every instance is evaluable.
    12 rollouts/instance (4 policies x 3 seeds) -> ceil(0.10*12)=2 held out, 10 train."""
    from src.offline.selection import split_trajectories
    insts = ["CC_2_3_4__pl_7", "CC_2_2_3__pl_4", "CC_3_2_3__pl_5"]
    _, _, man = split_trajectories(_pool_rows(insts), frac=0.10, seed=0)
    for inst in insts:
        c = man["per_instance"][inst]
        assert c["n_train"] >= 1, f"{inst} dropped from TRAIN"
        assert c["n_eval"] >= 1, f"{inst} absent from EVAL"
        assert c["n_train"] + c["n_eval"] == c["n_traj"]
    assert man["per_instance"]["CC_2_3_4__pl_7"]["n_eval"] == 2   # ceil(0.1*12)
    assert man["instances_with_no_eval"] == []


def test_split_never_holds_out_all_of_a_thin_instances_trajectories():
    """An instance with 2 rollouts: 1 held out, 1 kept in train -- never 0 in train."""
    from src.offline.selection import split_trajectories
    rows = _pool_rows(["X__pl_1"], policies=("bfs",), seeds=(0, 1))
    _, _, man = split_trajectories(rows, frac=0.99, seed=0)   # even at 99%
    c = man["per_instance"]["X__pl_1"]
    assert c["n_train"] == 1 and c["n_eval"] == 1


def test_split_is_reproducible_and_manifest_records_it():
    from src.offline.selection import split_trajectories
    rows = _pool_rows(["A__pl_1", "B__pl_2"])
    _, h1, m1 = split_trajectories(rows, frac=0.2, seed=7)
    _, h2, m2 = split_trajectories(rows, frac=0.2, seed=7)
    assert m1["heldout_trajectories"] == m2["heldout_trajectories"]
    assert m1["frac"] == 0.2 and m1["seed"] == 7
    # a different seed shuffles differently
    _, _, m3 = split_trajectories(rows, frac=0.2, seed=8)
    # (may coincide by chance on tiny sets; assert the recorded seed differs)
    assert m3["seed"] == 8


def test_smoothed_selection_prefers_a_sustained_peak_over_a_lucky_spike():
    """window-mean, not argmax. A single spiked checkpoint must NOT win over a
    sustained plateau -- this is the exact bias the floor run exposed."""
    from src.offline.selection import Candidate, select_smoothed
    def C(step, score):
        return Candidate(step=step, frames=step, coverage=0.0, regret=0.0, doom=0.0,
                         select_score=score)
    # a lone spike at step 2, vs a sustained high plateau at steps 4-6
    cands = [C(1, 0.5), C(2, 1.0), C(3, 0.5), C(4, 0.9), C(5, 0.9), C(6, 0.9)]
    chosen = select_smoothed(cands, window=3)
    assert chosen.step in (5, 6), "must pick the sustained plateau, not the spike"
    # argmax would have picked the step-2 spike
    assert max(cands, key=lambda c: c.select_score).step == 2


def test_smoothed_selection_needs_a_score_on_every_candidate():
    from src.offline.selection import Candidate, select_smoothed
    cands = [Candidate(step=1, frames=1, coverage=1.0, regret=0.0, doom=0.0)]  # no score
    with pytest.raises(ValueError, match="select_score"):
        select_smoothed(cands)


def test_heldout_top1_scores_rankers_on_the_same_frontiers():
    """Matched-n: a perfect ranker (oracle) scores 1.0, an adversarial ranker 0.0,
    on the identical held-out frontiers."""
    from src.offline.selection import heldout_top1
    from conftest import make_tree
    inst = make_tree([[1, 2, 3], [], [], []], goals=[1], name="G__pl_1")
    # frontier = slots {1,2,3}; slot 1 is the goal (delta 0), 2/3 are sterile (inf)
    held = [_Row("G__pl_1", "bfs", 0, obs=(1, 2, 3))]
    inst_by = {"G__pl_1": inst}
    good = lambda name, beam: sorted(range(len(beam)), key=lambda k: inst.delta[beam[k]])
    bad = lambda name, beam: sorted(range(len(beam)), key=lambda k: -inst.delta[beam[k]])
    g, ng = heldout_top1(held, good, inst_by)
    b, nb = heldout_top1(held, bad, inst_by)
    assert ng == nb == 1
    assert g == 1.0 and b == 0.0
