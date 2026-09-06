"""The `trace` behaviour: replay the generator's own expansion order.

The generation table records, through the DOT creation counter, the order in which
the search expanded states (children are created the moment their parent is
expanded). `trace` ranks a beam by that order, so a rollout follows pi_b as
closely as the F-window and the random refill allow.
"""

from __future__ import annotations

import pytest

from conftest import make_tree
from src.offline.dataset import generate_dataset
from src.offline.env import FringeEnv, rollout
from src.offline.policies import (
    ALL_POLICIES,
    BEHAVIOUR_POLICIES,
    TRACE_POLICY,
    make_policy,
    trace_ranking,
)
from src.offline.tree import (
    INF_DELTA,
    UNREACHABLE_DISTANCE,
    compute_censored,
    compute_expansion_order,
    dot_indices,
    provably_expanded,
)

#   0 -> 1, 2, 3       goal 6 under 2 (depth 2); 5 under 3 is a real dead end
#   1 -> 4             4 is a leaf
#   2 -> 6 (GOAL)
#   3 -> 5             5 childless non-goal
CHILDREN = [[1, 2, 3], [4], [6], [5], [], [], []]
GOALS = [6]
# BFS creation order: root 0 (1), then 1,2,3 (2,3,4), then 4 child of 1 (5),
# 6 child of 2 (6), 5 child of 3 (7).  Indexed BY STATE ID below.
BFS_INDEX = [1, 2, 3, 4, 5, 7, 6]


def test_dot_indices_parse_the_creation_counter():
    assert dot_indices(["x/RawFiles/hash_merged/000023.dot", "y/000001.dot"]) == [23, 1]
    assert dot_indices(["x/000023.dot", "y/init.dot"]) is None


def test_expansion_order_bfs():
    time, rank = compute_expansion_order(CHILDREN, BFS_INDEX)
    # expansion_time(v) = min index of a child created after v
    assert time == [2, 5, 6, 7, None, None, None]
    assert rank == [0, 1, 2, 3, None, None, None]


def test_expansion_order_dfs():
    # DFS: visit 0(1) -> 1(2) -> 4(3) back -> 2(4) -> 6(5) back -> 3(6) -> 5(7)
    idx = [1, 2, 4, 6, 3, 7, 5]
    time, rank = compute_expansion_order(CHILDREN, idx)
    assert time == [2, 3, 5, 7, None, None, None]
    assert [rank[v] for v in (0, 1, 2, 3)] == [0, 1, 2, 3]


def test_re_discovery_edges_with_smaller_index_are_ignored():
    """The DFS worker records a dedup'd re-discovery as an edge to an OLDER state;
    that child's index says nothing about when the parent was expanded."""
    children = [[1, 2], [2], []]           # 1 -> 2 re-discovers 2 (created earlier)
    idx = [1, 3, 2]
    time, rank = compute_expansion_order(children, idx)
    assert time == [2, None, None]         # 1's only child predates 1 -> unexpanded
    assert rank == [0, None, None]


# ----------------------------------------------------- provably expanded ----

def test_fifo_strategies_prove_everything_before_the_last_expansion():
    time, _ = compute_expansion_order(CHILDREN, BFS_INDEX)
    depth = [0, 1, 1, 1, 2, 2, 2]
    exp = provably_expanded("bfs", BFS_INDEX, time, depth, depth_bound=None)
    # last expanded = 3 (index 4); 4,6,5 have indices 5,6,7 > 4 -> unexpanded
    assert exp == [True, True, True, True, False, False, False]


def test_fifo_proof_respects_the_depth_bound():
    """A childless state created before the last expansion is proven expanded by
    the FIFO argument -- unless it sits AT the depth bound, where the generator
    refuses to expand, so nothing is proven."""
    children = [[1, 2], [], [3], []]
    idx = [1, 2, 3, 4]
    depth = [0, 1, 1, 2]
    time, _ = compute_expansion_order(children, idx)     # 0 -> 2, 2 -> 4; last = idx 3
    assert provably_expanded("dfs", idx, time, depth, depth_bound=None)[1] is True
    assert provably_expanded("dfs", idx, time, depth, depth_bound=1)[1] is False
    assert provably_expanded("dfs", idx, time, depth, depth_bound=1)[0] is True


def test_hfs_proves_only_states_with_children():
    time, _ = compute_expansion_order(CHILDREN, BFS_INDEX)
    depth = [0, 1, 1, 1, 2, 2, 2]
    exp = provably_expanded("hfs", BFS_INDEX, time, depth)
    assert exp == [t is not None for t in time]


def test_unexpanded_leaf_is_censored_not_sterile():
    """A BFS frontier cut by the VISIT cap: childless, non-goal, well below the
    depth bound. The depth rule calls it a dead end; the expansion rule knows the
    generator never looked at it."""
    delta = [2.0, INF_DELTA, 1.0, INF_DELTA, INF_DELTA, INF_DELTA, 0.0]
    h = [d if d != INF_DELTA else UNREACHABLE_DISTANCE for d in delta]
    is_goal = [False] * 6 + [True]
    depth = [0, 1, 1, 1, 2, 2, 2]
    without = compute_censored(CHILDREN, is_goal, delta, h, depth, depth_bound=20)
    assert without[4] is False and without[5] is False        # labelled sterile
    expanded = [True, True, True, True, False, False, False]  # 4 and 5 never expanded
    with_ = compute_censored(CHILDREN, is_goal, delta, h, depth, depth_bound=20,
                             expanded=expanded)
    assert with_[4] is True and with_[5] is True
    assert with_[1] is True and with_[3] is True, "propagates to inf-delta parents"
    assert with_[0] is False and with_[2] is False


# ----------------------------------------------------------- the policy -----

def _bfs_tree():
    return make_tree(CHILDREN, GOALS, name="bfs_tree", dot_index=BFS_INDEX, strategy="bfs")


def test_trace_is_in_all_policies_but_not_a_synthetic_baseline():
    assert TRACE_POLICY in ALL_POLICIES and TRACE_POLICY not in BEHAVIOUR_POLICIES


def test_trace_requires_a_trace(t19):
    with pytest.raises(ValueError, match="carries no generation trace"):
        make_policy(t19, TRACE_POLICY, seed=0)


def test_trace_ranking_follows_the_generator_order():
    t = _bfs_tree()
    # beam holding 3, 1, 2 (slots 0,1,2): generator expanded 1 then 2 then 3
    assert trace_ranking(t, [3, 1, 2]) == [1, 2, 0]
    pol = make_policy(t, TRACE_POLICY, seed=0)
    assert pol([3, 1, 2])[0] == 1


def test_unexpanded_states_rank_last_and_tie():
    t = _bfs_tree()
    r = trace_ranking(t, [4, 5, 3])            # 4,5 never expanded; 3 expanded 4th
    assert r[0] == 2
    seen = {tuple(make_policy(t, TRACE_POLICY, seed=s)([4, 5])) for s in range(20)}
    assert seen == {(0, 1), (1, 0)}, "ties among unexpanded states are broken at random"


def test_trace_rollout_reproduces_bfs_on_a_bfs_tree():
    """On a BFS-generated tree with F large enough to hold the open set, replaying
    the trace IS breadth-first: same expansion count as the synthetic bfs ranking."""
    t = _bfs_tree()
    env = FringeEnv(t, fringe_size=8, seed=0, gamma=1.0, expansion_cap=50)
    out = rollout(env, make_policy(t, TRACE_POLICY, seed=0), seed=0)
    assert out["solved"]
    # root (1) + expand 1 (2) + expand 2 -> generates goal 6: 3 expansions
    assert out["expansions"] == 3


def test_generate_dataset_accepts_trace_and_reports_duplicates():
    t = _bfs_tree()
    rows, s = generate_dataset([t], fringe_size=8, policies=(TRACE_POLICY,),
                               seeds_per_policy=3, expansion_cap=50, verbose=False)
    assert rows and all(r.policy == TRACE_POLICY for r in rows)
    # open set never exceeds F=8 on a 7-node tree: every seed walks the same path
    assert s["n_trajectories"] == 3 and s["n_distinct_trajectories"] == 1
    assert s["duplicate_trajectory_frac"] == pytest.approx(2 / 3)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown behaviour policy"):
        generate_dataset([_bfs_tree()], 4, policies=("greedy",), verbose=False)


# ------------------------------------------------------ real tables ---------

def test_real_tables_carry_a_trace_rooted_at_the_root(strategy_tables, capsys):
    for s, t in strategy_tables.items():
        assert t.strategy == s and t.has_trace, s
        assert t.expansion_rank[t.root_id] == 0, f"{s}: root is expanded first"
        assert t.n_expanded > 0
        # a child is never created before its parent
        assert all(t.expansion_rank[c] is None or t.expansion_rank[c] > t.expansion_rank[v]
                   for v in range(t.n_states) if t.expansion_rank[v] is not None
                   for c in t.children[v])
    with capsys.disabled():
        print("\n  " + "\n  ".join(
            f"{s:5} states={t.n_states} expanded={t.n_expanded} delta_root={t.delta_root:.0f} "
            f"goal_density={t.stats()['goal_density']:.3f} censored={t.stats()['censored_frac']:.3f}"
            for s, t in strategy_tables.items()))


def test_bfs_table_frontier_is_censored_not_sterile(strategy_tables):
    """A BFS run cut at 1000 visits leaves thousands of childless states far below
    the depth bound. Every one the generator never expanded must be censored."""
    t = strategy_tables["bfs"]
    frontier = [v for v in range(t.n_states)
                if not t.children[v] and not t.is_goal[v] and not t.expanded[v]]
    assert frontier, "the smoke BFS table has an unexpanded frontier"
    assert all(t.censored[v] for v in frontier)
    assert not any(t.provably_sterile(v) for v in frontier)


def test_trace_replays_each_strategy_in_its_own_tree(strategy_tables):
    """Same problem, four searches: replaying HFS's order reaches the goal in fewer
    expansions than replaying BFS's (the heuristic prefers goal-ward states), and
    every replay solves."""
    got = {}
    for s, t in strategy_tables.items():
        env = FringeEnv(t, fringe_size=32, seed=0, gamma=1.0, expansion_cap=2000)
        out = rollout(env, make_policy(t, TRACE_POLICY, seed=0), seed=0)
        assert out["solved"], s
        got[s] = out["expansions"]
    assert got["hfs"] < got["bfs"], got
