"""The faithfulness gatekeeper: nothing trains on data that fails this.

The gate VALIDATES OUTPUT. It owns no generation parameter and does not know how the
tree was built -- that lives in scripts/gnn_exp/create_all_training_data.py. The
independence is the point: the generator picks the knobs, the gate asks only whether
the RESULT is faithful, so a wrong knob cannot silence its own alarm.
"""

from __future__ import annotations

import json

from src.offline.faithfulness import (
    MIN_EXPANSIONS_FOR_FIDELITY,
    bfs_expansions,
    build_faithful_pool,
    check_instance,
    expected_optimal,
    faithful_names,
    fidelity_names,
)

from conftest import make_tree


def _named(children, goals, name):
    return make_tree(children, goals, name=name)


# ------------------------------------- how the check knows its target --------

def test_expected_optimal_is_the_pl_in_the_name():
    """Verified against the planner: strict BFS returns exactly pl_N on every CC
    instance measured (combined_results/batch2/CC/bfs/*_BFS_strict.csv, 26/26).
    --strong_equality does not move it (checked pl_3..pl_6 on CC_2_3_4)."""
    assert expected_optimal("CC_2_3_4__pl_7") == 7
    assert expected_optimal("SC_R_10_10__pl_10") == 10
    assert expected_optimal("no_plan_length") is None


def test_the_gate_owns_no_generation_parameter():
    """ARCHITECTURE. Two places encoding generation params is what let
    DEPTH_BY_DOMAIN=25 outlive the generator's real setting, so any code reading it
    was silently wrong. The gate must stay a pure check: it reads a tree and judges
    it, and cannot be made to agree with a bad knob by sharing that knob."""
    import src.offline.faithfulness as f
    for attr in ("DEPTH_BY_DOMAIN", "DEFAULT_DEPTH", "MAX_CREATION", "DISCARD_FACTOR",
                 "depth_for", "generation_argv", "generation_manifest"):
        assert not hasattr(f, attr), (
            f"{attr} is a GENERATION parameter; it belongs in "
            f"create_all_training_data.py, not in the check that validates its output"
        )
    v = check_instance(_named([[1], [2], []], [2], "X_1_1__pl_2"), min_expansions=1)
    assert not hasattr(v, "dataset_depth"), "the verdict must not carry the depth"


# ----------------------------------------- the discard-artifact test ---------

def test_faithful_instance_passes():
    # 0 -> 1 -> 2(goal): delta_root == 2 == pl_2
    t = _named([[1], [2], []], [2], "X_1_1__pl_2")
    v = check_instance(t, min_expansions=1)
    assert v.faithful and v.delta_root == 2 and v.expected_optimal == 2


def test_delta_root_mismatch_is_excluded_as_a_discard_artifact():
    """THE test. Shipped data: CC_2_2_3__pl_4 had delta_root 10 vs optimal 4 --
    the biased discard had deleted the shallow goals. The same check later caught
    CC_2_3_4__pl_7 at delta_root 22 vs optimal 7 at discard 0, where a generation
    ceiling truncated the DFS instead -- one check, either cause."""
    t = _named([[1], [2], [3], [4], []], [4], "X_1_1__pl_2")   # delta_root 4, pl says 2
    v = check_instance(t)
    assert not v.faithful
    assert any("!= known optimal 2" in r for r in v.reasons)
    assert any("solution path is ABSENT" in r for r in v.reasons)


def test_unsolvable_is_excluded():
    v = check_instance(_named([[1], []], [], "X__pl_3"))
    assert not v.faithful and any("no goal reachable" in r for r in v.reasons)


def test_tiny_instance_is_flagged_not_excluded():
    """Trivial instances stay in the pool for coverage; they just cannot
    discriminate for the fidelity gate (at 7 expansions, +-1 is 14%)."""
    t = _named([[1], [2], []], [2], "X__pl_2")
    v = check_instance(t)
    assert v.faithful, "a tiny instance is still FAITHFUL"
    assert not v.usable_for_fidelity
    assert any("too small for the fidelity gate" in r for r in v.reasons)


def test_bfs_expansions_counts_to_goal_generation():
    # 0 -> 1,2 ; 2 -> 3(goal). BFS: expand 0 (1), expand 1 (2), expand 2 -> goal (3)
    t = _named([[1, 2], [], [3], []], [3], "X__pl_2")
    assert bfs_expansions(t) == 3


def test_pool_json_lists_faithful_excluded_and_why(tmp_path):
    good = _named([[1], [2], []], [2], "G_1__pl_2")
    bad = _named([[1], [2], [3], []], [3], "B_1__pl_1")     # delta_root 3 != 1
    doc = build_faithful_pool([good, bad], out_path=tmp_path / "faithful_pool.json",
                              verbose=False)
    assert doc["n_faithful"] == 1 and doc["n_excluded"] == 1
    assert faithful_names(doc) == ["G_1__pl_2"]
    assert doc["excluded"][0]["instance"] == "B_1__pl_1"
    assert doc["excluded"][0]["reasons"], "every exclusion must carry a reason"
    on_disk = json.loads((tmp_path / "faithful_pool.json").read_text())
    assert on_disk["n_faithful"] == 1


def test_fidelity_names_are_a_subset_of_faithful():
    tiny = _named([[1], [2], []], [2], "T__pl_2")
    doc = build_faithful_pool([tiny], verbose=False)
    assert faithful_names(doc) == ["T__pl_2"]
    assert fidelity_names(doc) == [], "too small to score fidelity"
