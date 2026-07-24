"""Test 1 — delta >= h* node-wise on both shipped CSVs; log the gap.

WHY THIS MATTERS
The CSV `Distance From Goal` (h*) is the true distance in the hash-deduplicated
state DAG. The MDP moves inside the reconstructed tree, whose parent pointers are
DFS first-discoverers, so tree distance (delta) OVERESTIMATES DAG distance. If
selection used h* it would charge the policy regret it cannot possibly avoid --
5 expansions on CC, 14 on SCRich -- purely as an artifact of data generation.
"""

from __future__ import annotations

from src.offline.tree import INF_DELTA, UNREACHABLE_DISTANCE, compute_delta

from conftest import T19_CHILDREN, T19_GOALS, make_tree


def test_delta_ge_h_star_nodewise(shipped_instances, capsys):
    for name, inst in shipped_instances.items():
        reachable = inst._reachable()
        violations = [
            v for v in reachable
            if inst.delta[v] != INF_DELTA and inst.delta[v] < inst.h_star[v]
        ]
        assert not violations, (
            f"{name}: delta < h* on {len(violations)} nodes, e.g. "
            f"{[(v, inst.delta[v], inst.h_star[v]) for v in violations[:5]]}"
        )
        gaps = [
            inst.delta[v] - inst.h_star[v]
            for v in reachable
            if inst.delta[v] != INF_DELTA
        ]
        with capsys.disabled():
            print(
                f"\n  {name}: n={len(reachable)} finite-delta={len(gaps)} "
                f"h*(root)={inst.h_star[inst.root_id]:.0f} "
                f"delta(root)={inst.delta_root:.0f} "
                f"root gap={inst.delta_root - inst.h_star[inst.root_id]:.0f} | "
                f"node gap mean={sum(gaps)/len(gaps):.2f} max={max(gaps):.0f}"
            )


def test_root_distances_are_well_formed(shipped_instances):
    """INVARIANTS, not literals.

    This test used to pin the brief's exact pairs -- CC_2_3_4__pl_7 (29, 34),
    SC_R_10_10__pl_10 (10, 24). Those numbers are NOT properties of the instances:
    they are properties of ONE generated table. The tree is a seed-dependent DFS
    sample, so delta_root moves with the seed (measured on CC_2_2_3__pl_4, true
    optimal 4, identical flags: 14/6/7 for seeds 42/43/44), and the brief's values
    came from the discard=0.4 era whose data the project now rejects outright. The
    test therefore broke the moment a table was regenerated -- correctly, but for a
    reason that says nothing about the code.

    What IS invariant, for any table the gate admits: both distances exist, are
    finite non-negative integers, and delta >= h* at the root (the reconstruction is
    a DFS spanning tree of the DAG, so tree distance overestimates DAG distance).
    """
    for name, inst in shipped_instances.items():
        h = inst.h_star[inst.root_id]
        d = inst.delta_root
        assert d != INF_DELTA, f"{name}: delta(root) must be finite (no DOOM)"
        assert 0 <= h < UNREACHABLE_DISTANCE, f"{name}: h*(root)={h} out of range"
        assert d >= 0, f"{name}: delta(root)={d} must be non-negative"
        assert float(d).is_integer(), f"{name}: delta(root)={d} must be integral"
        assert d >= h, (
            f"{name}: delta(root)={d} < h*(root)={h} -- tree distance can never "
            f"undercut DAG distance"
        )


def test_delta_root_equals_shallowest_goal_depth(shipped_instances):
    """delta(root) is the shallowest goal's depth -- the goal test fires at
    generation, so expanding its parent ends the episode."""
    for name, inst in shipped_instances.items():
        reachable = inst._reachable()
        shallowest = min(inst.depth[i] for i in reachable if inst.is_goal[i])
        assert inst.delta_root == shallowest, name


def test_delta_recurrence_on_synthetic(t19):
    """delta = 0 at goals, inf at non-goal leaves, 1 + min_c delta(c) otherwise."""
    assert t19.delta_root == 4.0
    for g in T19_GOALS:
        assert t19.delta[g] == 0.0
    for d in (2, 5, 7, 10, 12, 16):
        assert t19.delta[d] == INF_DELTA, f"non-goal leaf {d} must be inf"
    for v, cs in enumerate(T19_CHILDREN):
        if t19.is_goal[v] or not cs:
            continue
        m = min(t19.delta[c] for c in cs)
        assert t19.delta[v] == (1.0 + m if m != INF_DELTA else INF_DELTA), v


def test_delta_is_inf_when_no_goal_reachable():
    """A subtree with no goal has delta = inf, not a large finite number."""
    inst = make_tree([[1, 2], [], []], goals=[], name="goalless")
    assert inst.delta_root == INF_DELTA
    assert not inst.solvable()


def test_compute_delta_multi_source():
    """Nearest goal wins, not the first one found."""
    #   0 -> 1 -> 2 (goal)      delta(0) via 1 = 2
    #   0 -> 3 (goal)           delta(0) via 3 = 1
    delta = compute_delta([[1, 3], [2], [], []], [False, False, True, True])
    assert delta[0] == 1.0
    assert delta[1] == 1.0
    assert delta[3] == 0.0
