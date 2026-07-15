"""Generation config (per-domain depth) + the faithfulness gatekeeper."""

from __future__ import annotations

import json

import pytest

from src.offline.faithfulness import (
    MIN_EXPANSIONS_FOR_FIDELITY,
    bfs_expansions,
    build_faithful_pool,
    check_instance,
    faithful_names,
    fidelity_names,
)
from src.offline.generation import (
    DISCARD_FACTOR,
    MAX_CREATION,
    assert_depth_contains_optimal,
    depth_for,
    domain_of,
    expected_optimal,
    generation_argv,
    generation_manifest,
)

from conftest import make_tree


# ------------------------------------------------- per-domain depth ----------

def test_depth_is_domain_dependent():
    assert depth_for("CC_2_3_4__pl_7") == 25
    assert depth_for("CC_2_2_3__pl_4") == 25
    assert depth_for("SC_10_8__pl_15") == 40
    assert depth_for("SC_R_10_10__pl_10") == 40


def test_domain_of():
    assert domain_of("SC_R_10_10__pl_10") == "SCRich"
    assert domain_of("SC_9_11__pl_8") == "SC"
    assert domain_of("CC_2_3_4__pl_7") == "CC"


def test_expected_optimal_is_the_pl_in_the_name():
    """Verified against the planner: BFS returns exactly pl_N on CC_2_2_3__pl_4 (4),
    __pl_6 (6), CC_2_3_4__pl_7 (7), CC_3_2_3__pl_5 (5)."""
    assert expected_optimal("CC_2_3_4__pl_7") == 7
    assert expected_optimal("SC_R_10_10__pl_10") == 10
    assert expected_optimal("no_plan_length") is None


def test_depth_must_contain_the_optimal():
    assert_depth_contains_optimal("CC_2_3_4__pl_7")      # 25 >= 7, fine
    assert_depth_contains_optimal("SC_R_10_10__pl_10")   # 40 >= 10, fine
    import src.offline.generation as g
    old = dict(g.DEPTH_BY_DOMAIN)
    try:
        g.DEPTH_BY_DOMAIN["CC"] = 3          # below the pl_7 optimal
        with pytest.raises(ValueError, match="BELOW the known optimal"):
            assert_depth_contains_optimal("CC_2_3_4__pl_7")
    finally:
        g.DEPTH_BY_DOMAIN.clear(); g.DEPTH_BY_DOMAIN.update(old)


def test_generation_argv_never_discards_and_pins_the_ceiling():
    argv = generation_argv("p.txt", "CC_2_3_4__pl_7")
    assert argv[argv.index("--dataset_discard_factor") + 1] == "0", (
        "the biased discard is the artifact this config exists to remove"
    )
    assert argv[argv.index("--dataset_depth") + 1] == "25"
    assert argv[argv.index("--dataset_max_creation") + 1] == str(MAX_CREATION)
    assert DISCARD_FACTOR == 0 and MAX_CREATION == 50000


def test_generation_argv_is_representation_agnostic():
    """dataset_type passes through opaquely: HASHED today, BITMASK when the C++
    implements it -- no change here or downstream."""
    a = generation_argv("p.txt", "CC_2_3_4__pl_7", dataset_type="BITMASK")
    assert a[a.index("--dataset_type") + 1] == "BITMASK"


def test_manifest_records_the_depth_per_instance():
    m = generation_manifest("SC_R_10_10__pl_10")
    assert m["dataset_depth"] == 40 and m["domain"] == "SCRich"
    assert m["expected_optimal_plan_length"] == 10
    assert m["dataset_discard_factor"] == 0


# ----------------------------------------- the discard-artifact test ---------

def _named(children, goals, name):
    t = make_tree(children, goals, name=name)
    return t


def test_faithful_instance_passes():
    # 0 -> 1 -> 2(goal): delta_root == 2 == pl_2
    t = _named([[1], [2], []], [2], "X_1_1__pl_2")
    v = check_instance(t, min_expansions=1)
    assert v.faithful and v.delta_root == 2 and v.expected_optimal == 2


def test_delta_root_mismatch_is_excluded_as_a_discard_artifact():
    """THE test. Shipped data: CC_2_2_3__pl_4 had delta_root 10 vs optimal 4 --
    the biased discard had deleted the shallow goals."""
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
