"""The usability gatekeeper: nothing trains on data that fails this.

The gate VALIDATES OUTPUT. It owns no generation parameter and does not know how the
tree was built -- that lives in scripts/gnn_exp/create_all_training_data.py. The
independence is the point: the generator picks the knobs, the gate asks only whether
the RESULT is usable, so a wrong knob cannot silence its own alarm.

It certifies USABILITY (the root reaches a goal, the tree is non-trivial and
scorable), NOT that distances are the planner's optimal -- the generator's DFS cannot
deliver that. See src/offline/usability.py.
"""

from __future__ import annotations

import json

from src.offline.usability import (
    MIN_EXPANSIONS_FOR_FIDELITY,
    MIN_STATES,
    bfs_expansions,
    build_usable_pool,
    check_instance,
    expected_optimal,
    fidelity_names,
    usable_names,
)

from conftest import make_tree


def _named(children, goals, name):
    return make_tree(children, goals, name=name)


def _chain(n: int, name: str):
    """A 0->1->...->(n-1) chain whose last node is the goal. n states, delta_root n-1."""
    children = [[i + 1] for i in range(n - 1)] + [[]]
    return _named(children, [n - 1], name)


# ------------------------------------- how the check knows its target --------

def test_expected_optimal_is_the_pl_in_the_name():
    """Confirmed by the planner: combined_results/batch2/CC/bfs/train_BFS_strict.csv
    gives CC_2_3_4__pl_7 -> 7 and CC_2_2_3__pl_4 -> 4. Reported as a DIAGNOSTIC; it
    is no longer a gate."""
    assert expected_optimal("CC_2_3_4__pl_7") == 7
    assert expected_optimal("SC_R_10_10__pl_10") == 10
    assert expected_optimal("no_plan_length") is None


def test_the_gate_owns_no_generation_parameter():
    """ARCHITECTURE. Two places encoding generation params is what let
    DEPTH_BY_DOMAIN=25 outlive the generator's real setting, so any code reading it
    was silently wrong. The gate must stay a pure check: it reads a tree and judges
    it, and cannot be made to agree with a bad knob by sharing that knob."""
    import src.offline.usability as u
    for attr in ("DEPTH_BY_DOMAIN", "DEFAULT_DEPTH", "MAX_CREATION", "DISCARD_FACTOR",
                 "SEED", "MAX_GENERATION", "depth_for", "generation_argv",
                 "generation_manifest"):
        assert not hasattr(u, attr), (
            f"{attr} is a GENERATION parameter; it belongs in "
            f"create_all_training_data.py, not in the check that validates its output"
        )
    v = check_instance(_named([[1], [2], []], [2], "X_1_1__pl_2"),
                       min_states=1, min_expansions=1)
    assert not hasattr(v, "dataset_depth"), "the verdict must not carry the depth"


# --------------------------------- the retired optimality criterion ----------

def test_delta_root_mismatch_is_NOT_an_exclusion():
    """THE reframe (2026-07-15). `delta_root == pl_N` tested DFS LUCK, not the tree.

    The generator is a depth-bounded DFS whose memo is keyed by state alone, so it
    records DISCOVERY depths: delta_root >= the true optimal, equal only where the
    DFS sampled a shortest path first. Measured on CC_2_2_3__pl_4 (true optimal 4,
    identical flags, seed alone varied): delta_root = 14/6/7 for seeds 42/43/44.

    A tree whose delta_root exceeds the known optimal is EXPECTED, not broken. It
    stays in the pool; the mismatch is reported as a diagnostic.
    """
    t = _named([[1], [2], [3], [4], []], [4], "X_1_1__pl_2")   # delta_root 4, pl says 2
    v = check_instance(t, min_states=1)
    assert v.usable, "a delta_root above the known optimal must NOT be excluded"
    assert v.delta_root_matches_optimal is False
    assert any("[diagnostic]" in r and "did not sample a shortest path" in r
               for r in v.reasons)


def test_delta_root_matching_the_optimal_is_reported_but_grants_nothing():
    """The converse: a match is luck, not proof. It is recorded, not rewarded."""
    t = _named([[1], [2], []], [2], "X_1_1__pl_2")   # delta_root 2 == pl_2
    v = check_instance(t, min_states=1, min_expansions=1)
    assert v.usable and v.delta_root_matches_optimal is True


def test_poisoned_frac_is_a_diagnostic_not_a_gate():
    """h*=1e6 conflates a real ceiling truncation with an ordinary non-goal leaf at
    the depth bound. At depth 9 on CC_2_3_4__pl_7, 145 of 154 poisoned rows were
    plain frontier leaves and no ceiling ever fired -- excluding on it rejects healthy
    shallow trees."""
    t = _chain(60, "P__pl_59")
    t.h_star[3] = 1e6          # a poisoned node, whatever the cause
    v = check_instance(t)
    assert v.poisoned_frac > 0
    assert v.usable, "poisoning must never exclude"
    assert any("[diagnostic]" in r and "poisoned fraction" in r for r in v.reasons)


# ------------------------------------------- what IS still a gate ------------

def test_unsolvable_is_excluded():
    """delta(root)=inf is DOOM: no goal reachable, nothing for the agent to find.
    This is the completeness condition, and it is what a DFS CAN certify."""
    v = check_instance(_named([[1], []], [], "X__pl_3"))
    assert not v.usable and any("no goal reachable" in r for r in v.reasons)


def test_a_tree_below_the_state_floor_is_excluded():
    """Non-triviality: too few states and there is nothing to train on."""
    v = check_instance(_chain(5, "S__pl_4"))
    assert not v.usable
    assert any(f"< {MIN_STATES}" in r and "too small to train on" in r
               for r in v.reasons)


def test_a_tree_above_the_state_floor_passes():
    v = check_instance(_chain(MIN_STATES + 10, f"B__pl_{MIN_STATES + 9}"))
    assert v.usable, "a non-trivial, goal-reaching tree is USABLE"


def test_tiny_instance_is_flagged_not_excluded():
    """The fidelity floor stays a FLAG: trivial instances keep coverage, they just
    cannot discriminate for the fidelity gate (at 7 expansions, +-1 is 14%)."""
    t = _named([[1], [2], []], [2], "X__pl_2")
    v = check_instance(t, min_states=1)
    assert v.usable, "a tiny instance is still USABLE"
    assert not v.usable_for_fidelity
    assert any("too small for the fidelity gate" in r for r in v.reasons)


def test_bfs_expansions_counts_to_goal_generation():
    # 0 -> 1,2 ; 2 -> 3(goal). BFS: expand 0 (1), expand 1 (2), expand 2 -> goal (3)
    t = _named([[1, 2], [], [3], []], [3], "X__pl_2")
    assert bfs_expansions(t) == 3


# --------------------------------------------------- the pool ----------------

def test_pool_json_lists_usable_excluded_and_why(tmp_path):
    good = _chain(60, "G_1__pl_59")
    doomed = _named([[1], []], [], "B_1__pl_1")      # no goal -> DOOM
    doc = build_usable_pool([good, doomed], out_path=tmp_path / "usable_pool.json",
                            verbose=False)
    assert doc["n_usable"] == 1 and doc["n_excluded"] == 1
    assert usable_names(doc) == ["G_1__pl_59"]
    assert doc["excluded"][0]["instance"] == "B_1__pl_1"
    assert doc["excluded"][0]["reasons"], "every exclusion must carry a reason"
    on_disk = json.loads((tmp_path / "usable_pool.json").read_text())
    assert on_disk["n_usable"] == 1


def test_pool_records_the_distance_semantics(tmp_path):
    """The pool is where the next reader meets this data. It must say, in the file,
    that delta is tree-exact and NOT the instance's optimal -- otherwise the retired
    claim sneaks back in via the artifact."""
    doc = build_usable_pool([_chain(60, "G__pl_59")], verbose=False)
    sem = doc["distance_semantics"]
    assert "WITHIN the generated tree" in sem
    assert "NOT the planner's" in sem
    assert "upper bound" in sem
    assert "DIAGNOSTIC ONLY" in doc["criteria"]["delta_root_equals_known_optimal"]


def test_fidelity_names_are_a_subset_of_usable():
    tiny = _named([[1], [2], []], [2], "T__pl_2")
    doc = build_usable_pool([tiny], verbose=False, min_states=1)
    assert usable_names(doc) == ["T__pl_2"]
    assert fidelity_names(doc) == [], "too small to score fidelity"
