"""unify.py: content-unique states across behaviour-policy trees, labels recomputed
on the union, per-strategy traces preserved."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from conftest import make_tree
from src.offline.dataset import generate_dataset
from src.offline.env import FringeEnv, rollout
from src.offline.policies import expand_policies, make_policy, trace_policies_of
from src.offline.tree import INF_DELTA, load_tree_instance
from src.offline.unify import (
    UnifiedInstance,
    fingerprint_dot,
    fingerprint_dot_bytes,
    fingerprints_for_tree,
    unified_tree_name,
    unify_instances,
    unify_trees,
)

# ----------------------------------------------------------------- fingerprints --

DOT_A = b'digraph G {\n  1 -> 2 [label="8"];\n  2 -> 1 [label="9"];\n}\n'
DOT_A_REORDERED = b'digraph G {\n  2 -> 1 [label="9"];\n  1 -> 2 [label="8"];\n}\n'
DOT_B = b'digraph G {\n  1 -> 2 [label="8"];\n  2 -> 2 [label="9"];\n}\n'


def test_fingerprint_is_order_independent_and_content_sensitive():
    assert fingerprint_dot_bytes(DOT_A) == fingerprint_dot_bytes(DOT_A_REORDERED)
    assert fingerprint_dot_bytes(DOT_A) != fingerprint_dot_bytes(DOT_B)
    # whitespace and the header/closing lines do not matter
    assert fingerprint_dot_bytes(b"digraph G {\n1 -> 2 [label=\"8\"];\n2 -> 1 [label=\"9\"];\n}") \
        == fingerprint_dot_bytes(DOT_A)


def test_fingerprint_dot_reads_files(tmp_path):
    p = tmp_path / "x.dot"
    p.write_bytes(DOT_A)
    assert fingerprint_dot(p) == fingerprint_dot_bytes(DOT_A)


# ------------------------------------------------------------ synthetic merge --

def _fps(tree, content):
    """content[v] -> digest per state id."""
    return [fingerprint_dot_bytes(content[v]) for v in range(tree.n_states)]


def _dot(k: int) -> bytes:
    return f'digraph G {{\n  1 -> {k} [label="8"];\n}}\n'.encode()


def test_merge_collapses_shared_states_and_unions_edges():
    # tree A (bfs): 0 -> 1, 2 ; 1 -> 3(goal)      states by content: 0=r,1=a,2=b,3=g
    # tree B (dfs): 0 -> 2', 4 ; 2' -> 5(goal)    2' is the same content as A's 2
    A = make_tree([[1, 2], [3], [], []], goals=[3], name="P@bfs", strategy="bfs",
                  dot_index=[1, 2, 3, 4])
    B = make_tree([[1, 2], [3], [], []], goals=[3], name="P@dfs", strategy="dfs",
                  dot_index=[1, 2, 3, 4])
    A.instance = B.instance = "P"
    contentA = {0: _dot(0), 1: _dot(1), 2: _dot(2), 3: _dot(3)}
    contentB = {0: _dot(0), 1: _dot(2), 2: _dot(4), 3: _dot(5)}   # B's state 1 == A's 2
    u = unify_trees([A, B], [_fps(A, contentA), _fps(B, contentB)])
    assert isinstance(u, UnifiedInstance)
    assert u.name == unified_tree_name("P") and u.strategy == "unified"
    # 8 files -> 6 unique states (root and A2/B1 shared)
    assert u.n_states == 6
    assert u.manifest["n_files"] == 8 and u.manifest["n_shared_states"] == 2
    assert u.manifest["n_in_all_trees"] == 2
    # the shared state now has children from BOTH trees: none from A, goal 5 from B
    shared = u.n_trees_of.index(2, 1)          # the non-root shared id
    assert u.n_trees_of[u.root_id] == 2
    assert len(u.children[shared]) == 1 and u.is_goal[u.children[shared][0]]
    # delta on the union: A's dead-end leaf 2 is B's goal parent -> delta 1, not inf
    assert u.delta[shared] == 1.0
    assert u.delta_root == 2.0
    # the union equals the per-tree MIN here (A: inf, B: 1) -- a disagreement, not
    # an improvement; improvement needs two trees' paths to COMBINE (next test)
    assert u.manifest["n_shared_delta_disagree"] == 1
    assert u.manifest["n_delta_improved_by_union"] == 0
    # both traces survive, keyed by canonical id
    assert set(u.traces) == {"bfs", "dfs"}
    assert u.traces["bfs"][u.root_id] == 0 and u.traces["dfs"][u.root_id] == 0
    assert u.traces["bfs"][shared] is None            # A never expanded it
    assert u.traces["dfs"][shared] is not None        # B did
    assert u.has_trace and trace_policies_of(u) == ["trace:bfs", "trace:dfs"]


def test_union_delta_beats_the_per_tree_minimum_when_paths_combine():
    # A: 0 -> 1 -> 2 -> 3 -> 4(goal)   delta_A(1) = 3, delta_A(0) = 4
    # B: 0 -> 5 -> 6(goal)             5 has the CONTENT of A's 2; delta_B(0) = 2
    # union: 1 -> 2(=5) -> 6(goal)     delta(1) = 2 < 3 (A's only label for it);
    #                                  root = 2 = B's label (min, not an improvement)
    A = make_tree([[1], [2], [3], [4], []], goals=[4], name="P@bfs", strategy="bfs")
    B = make_tree([[1], [2], []], goals=[2], name="P@dfs", strategy="dfs")
    A.instance = B.instance = "P"
    cA = {0: _dot(0), 1: _dot(1), 2: _dot(2), 3: _dot(3), 4: _dot(4)}
    cB = {0: _dot(0), 1: _dot(2), 2: _dot(6)}
    u = unify_trees([A, B], [_fps(A, cA), _fps(B, cB)])
    assert u.delta[1] == 2.0 and u.delta_root == 2.0
    assert u.manifest["n_delta_improved_by_union"] == 1      # state 1 only
    assert u.manifest["delta_root_per_tree"] == {"P@bfs": 4.0, "P@dfs": 2.0}
    assert u.manifest["delta_root_unified"] == 2.0


def test_root_mismatch_is_refused():
    A = make_tree([[1], []], goals=[1], name="P@bfs", strategy="bfs")
    B = make_tree([[1], []], goals=[1], name="P@dfs", strategy="dfs")
    A.instance = B.instance = "P"
    with pytest.raises(ValueError, match="root"):
        unify_trees([A, B], [_fps(A, {0: _dot(0), 1: _dot(1)}),
                             _fps(B, {0: _dot(9), 1: _dot(1)})])


def test_different_problems_are_refused():
    A = make_tree([[1], []], goals=[1], name="P@bfs", strategy="bfs")
    B = make_tree([[1], []], goals=[1], name="Q@dfs", strategy="dfs")
    A.instance, B.instance = "P", "Q"
    with pytest.raises(ValueError, match="different problems"):
        unify_trees([A, B], [_fps(A, {0: _dot(0), 1: _dot(1)}),
                             _fps(B, {0: _dot(0), 1: _dot(1)})])


def test_within_tree_duplicates_collapse_and_self_loops_are_dropped():
    # one tree where states 1 and 2 have the same content, and 2's child 3 has the
    # content of 1 (a re-reached state): merge -> a cycle 1 -> 1 dropped as self-loop
    T = make_tree([[1, 2], [], [3], []], goals=[], name="P@bfs", strategy="bfs")
    T.instance = "P"
    content = {0: _dot(0), 1: _dot(1), 2: _dot(1), 3: _dot(1)}
    u = unify_trees([T], [_fps(T, content)])
    assert u.n_states == 2
    assert u.manifest["n_within_tree_duplicates"] == 2
    assert u.manifest["n_self_loops_dropped"] == 1
    assert u.initial_forced_chain_len() == 1          # cycle-safe


def test_goal_or_and_conflicts_are_counted():
    A = make_tree([[1], []], goals=[1], name="P@bfs", strategy="bfs")
    B = make_tree([[1], []], goals=[], name="P@dfs", strategy="dfs", h_star=[1e6, 1e6])
    A.instance = B.instance = "P"
    c = {0: _dot(0), 1: _dot(1)}
    u = unify_trees([A, B], [_fps(A, c), _fps(B, c)])
    assert u.is_goal[1] is True
    assert u.manifest["n_goal_conflicts"] == 1
    assert u.h_star[1] == 0.0                         # min over trees


def test_expanded_is_or_and_censoring_is_recomputed():
    # A: 0 -> 1 (leaf, unexpanded: BFS cut)     B: 0 -> 1 -> 2(goal)
    A = make_tree([[1], []], goals=[], name="P@bfs", strategy="bfs", h_star=[1e6, 1e6],
                  dot_index=[1, 2])
    B = make_tree([[1], [2], []], goals=[2], name="P@dfs", strategy="dfs",
                  dot_index=[1, 2, 3])
    A.instance = B.instance = "P"
    A.expanded = [True, False]
    B.expanded = [True, True, False]
    u = unify_trees([A, B], [_fps(A, {0: _dot(0), 1: _dot(1)}),
                             _fps(B, {0: _dot(0), 1: _dot(1), 2: _dot(2)})])
    assert u.expanded == [True, True, False]
    # A's censored leaf is B's expanded internal node with a goal child: finite delta
    assert u.delta[1] == 1.0 and not u.censored[1]


def test_unified_graph_rolls_out_and_generates_rows_per_trace():
    A = make_tree([[1, 2], [3], [], []], goals=[3], name="P@bfs", strategy="bfs",
                  dot_index=[1, 2, 3, 4])
    B = make_tree([[1, 2], [3], [], []], goals=[3], name="P@dfs", strategy="dfs",
                  dot_index=[1, 2, 3, 4])
    A.instance = B.instance = "P"
    contentA = {0: _dot(0), 1: _dot(1), 2: _dot(2), 3: _dot(3)}
    contentB = {0: _dot(0), 1: _dot(2), 2: _dot(4), 3: _dot(5)}
    u = unify_trees([A, B], [_fps(A, contentA), _fps(B, contentB)])
    env = FringeEnv(u, fringe_size=4, seed=0, expansion_cap=50)
    r = rollout(env, make_policy(u, "hfs_oracle", seed=0), seed=0)
    assert r["solved"] and r["expansions"] == u.delta_root
    # `trace` expands to one rollout family per strategy on a unified graph
    assert expand_policies(u, ["trace"]) == ["trace:bfs", "trace:dfs"]
    rows, summ = generate_dataset([u], 4, policies=("trace",), seeds_per_policy=1,
                                  expansion_cap=50, verbose=False)
    assert set(summ["policies"]) == {"trace:bfs", "trace:dfs"}
    assert all(r.instance == u.name for r in rows)
    with pytest.raises(ValueError, match="ambiguous"):
        make_policy(u, "trace")
    with pytest.raises(ValueError, match="no 'hfs' trace"):
        make_policy(u, "trace:hfs")


# --------------------------------------------------------- real tables on disk --

def _write_table(inst_dir: Path, strat_token: str, rows, goal_path: Path):
    inst_dir.mkdir(parents=True, exist_ok=True)
    raw = inst_dir / "RawFiles" / "hash_separated"
    raw.mkdir(parents=True)
    csv_path = inst_dir / f"{inst_dir.name}_{strat_token}_depth_25.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["File Path", "Depth", "Distance From Goal", "Goal",
                    "File Path Predecessor", "Action"])
        for idx, depth, dist, pred_idx, content in rows:
            p = raw / f"{idx:06d}.dot"
            p.write_bytes(content)
            pred = (raw / f"{pred_idx:06d}.dot") if pred_idx else (raw / "init.dot")
            w.writerow([str(p), depth, dist, str(goal_path), str(pred), 1])
    return csv_path


def test_unify_instances_from_tables_caches_fingerprints(tmp_path):
    goal = tmp_path / "goal_tree.dot"
    goal.write_bytes(b"digraph G {\n}\n")
    # BFS: root(1) -> a(2), b(3); a -> g(4)
    bfs = _write_table(tmp_path / "BFS" / "P__pl_2", "BFS", [
        (1, 0, 2, None, _dot(0)), (2, 1, 1, 1, _dot(1)), (3, 1, 1e6, 1, _dot(2)),
        (4, 2, 0, 2, _dot(3))], goal)
    # DFS: root(1) -> b(2) -> g2(3)  (b has the content of BFS's dead leaf)
    dfs = _write_table(tmp_path / "DFS" / "P__pl_2", "DFS", [
        (1, 0, 2, None, _dot(0)), (2, 1, 1, 1, _dot(2)), (3, 2, 0, 2, _dot(9))], goal)
    trees = [load_tree_instance(p, kind_of_data="separated") for p in (bfs, dfs)]
    assert {t.strategy for t in trees} == {"bfs", "dfs"}
    cache = tmp_path / "cache" / "fp"
    us = unify_instances(trees, repo_root=tmp_path, cache_dir=cache, verbose=False)
    assert len(us) == 1
    u = us[0]
    assert u.instance == "P__pl_2" and u.n_states == 5   # 7 files, root + b shared
    assert u.manifest["n_shared_states"] == 2
    # the BFS dead leaf b is now one step from DFS's goal
    b = [v for v in range(u.n_states) if u.n_trees_of[v] == 2 and v != u.root_id][0]
    assert u.delta[b] == 1.0
    assert (cache / "P__pl_2@bfs.fp.json").exists()
    # second call hits the cache and yields the same digests
    again = fingerprints_for_tree(trees[0], tmp_path, cache, verbose=False)
    assert again == fingerprints_for_tree(trees[0], tmp_path, None, verbose=False)
    # representative paths resolve (one DOT per unique state)
    for p in u.state_paths_abs(tmp_path):
        assert Path(p).exists()
    assert u.goal_path == str(goal)
