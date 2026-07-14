"""Tests 4, 5, 6 + the dead-end path — the env is a faithful RL_BestFirst.

The golden traces below are HAND-DERIVED from RL_BestFirst.h / SpaceSearcher.tpp,
not recorded from this implementation. They are set up so refill has no freedom
(the beam is already full, or the reservoir holds exactly one node), which makes
(B, R) exact despite RefillMode::RANDOM.
"""

from __future__ import annotations

import pytest

from src.offline.env import FringeEnv
from src.offline.policies import BEHAVIOUR_POLICIES, make_policy, oracle_ranking
from src.offline.tree import INF_DELTA

from conftest import T19_DEAD, T19_GOALS, make_tree


def _env(inst, F, **kw):
    kw.setdefault("gamma", 1.0)
    return FringeEnv(inst, fringe_size=F, **kw)


# ------------------------------------------- test 5: golden (B, R) traces -----

def test_golden_trace_oracle_F2(t19):
    """Hand-derived trace. F=2, so every rebuild fills the beam from children
    alone and refill never fires -> fully deterministic despite random refill.

      reset: expand 0 -> fresh [1,2,3]; batch=[1,2]; overflow [3] -> R
             B=[1,2]              R=[3]
      exp 1 (delta 3 < inf):      B=[4,5]  R=[3, 2, 6,7]
             drain [2]->R, batch=[4,5], overflow [6,7]->R
      exp 4 (delta 2):            B=[9,10] R=[3,2,6,7, 5]
      exp 9 (delta 1): child 14 is a GOAL -> success at expansion 4 = delta(root)
    """
    env = _env(t19, 2)
    r = env.reset()
    assert env.fringe == [1, 2] and env.reservoir == [3]
    assert r.reward == 0.0 and not r.done and not env.forced

    r = env.step(0, ranking=[0, 1])          # expand 1
    assert env.fringe == [4, 5]
    assert env.reservoir == [3, 2, 6, 7]     # drain precedes overflow
    assert r.reward == -1.0 and not r.done

    r = env.step(0, ranking=[0, 1])          # expand 4
    assert env.fringe == [9, 10]
    assert env.reservoir == [3, 2, 6, 7, 5]

    r = env.step(0, ranking=[0, 1])          # expand 9 -> generates goal 14
    assert r.done and r.reward == 0.0
    assert r.info["outcome"] == "success"
    assert r.info["expansions"] == 4 == t19.delta_root


def test_golden_trace_dead_end_does_not_refill_or_rescore(t19):
    """THE dead-end assertion.

    From B=[1,2], R=[3], expanding 2 (b_v == 0) leaves fringe_RL empty, so
    `0 >= 1` is false and push_vector never fires:
      - B' = B \\ {2} = [1]   -- NOT topped back up to F=2 from R
      - R' = R = [3]         -- untouched
      - the successor is FORCED: the planner pops the stale rank-2, no model call
    """
    env = _env(t19, 2)
    env.reset()
    assert env.fringe == [1, 2] and env.reservoir == [3]

    r = env.step(1, ranking=[0, 1])          # expand 2, the dead end

    assert env.fringe == [1], "dead end must not rebuild the beam"
    assert env.reservoir == [3], "dead end must not touch the reservoir"
    assert len(env.fringe) < env.fringe_size, (
        "the beam is left BELOW F: refilling here would be the classic wrong model"
    )
    assert env.forced is True and r.info["forced"] is True
    assert env.forced_action == 0
    assert env.available_actions() == [0], "|A(s)| == 1 at a forced state"
    assert env.n_actions() == 1
    assert r.reward == -1.0 and not r.done

    # the forced step then proceeds normally and rescores (b_1 = 4 >= 1)
    r = env.step(0)                           # no ranking needed: not consulted
    assert env.fringe == [4, 5]
    assert env.reservoir == [3, 6, 7]
    assert env.forced is False
    assert env.n_forced_steps == 1


def test_forced_state_rejects_a_different_action(t19):
    env = _env(t19, 2)
    env.reset()
    env.step(1, ranking=[0, 1])               # dead end -> forced
    assert env.forced
    # B is [1] so slot 0 is the only in-range slot; use a tree where B is larger
    inst = make_tree([[1, 2, 3], [], [4], [5], [], []], goals=[4], name="wide")
    e2 = _env(inst, 3)
    e2.reset()                                 # B=[1,2,3]
    e2.step(0, ranking=[0, 1, 2])              # expand 1 (dead end) -> forced on [2,3]
    assert e2.forced and e2.forced_action == 0
    with pytest.raises(ValueError, match="forced"):
        e2.step(1)


def test_rescore_state_requires_a_ranking(t19):
    env = _env(t19, 2)
    env.reset()
    with pytest.raises(ValueError, match="ranking is required"):
        env.step(0)
    with pytest.raises(ValueError, match="permutation"):
        env.step(0, ranking=[0, 0])


def test_unscored_reservoir_pull_is_reachable(capsys):
    """RL_BestFirst::peek() lines 127-142, reached via consecutive dead ends.

      0 -> a(dead), b(dead), c(->goal);  F=2  =>  B=[a,b], R=[c]
      expand a: dead end, no rescore     ->  B=[b],  R=[c],  forced
      expand b: dead end, B empties but R is non-empty, so `empty()` is false and
                push_vector still never fires -> peek() pushes ONE random
                reservoir node straight into the queue, NEVER SCORED.
    """
    inst = make_tree([[1, 2, 3], [], [], [4], []], goals=[4], name="pull")
    env = _env(inst, 2)
    env.reset()
    assert env.fringe == [1, 2] and env.reservoir == [3]

    env.step(0, ranking=[0, 1])               # expand 1 (dead)
    assert env.fringe == [2] and env.reservoir == [3] and env.forced

    r = env.step(0)                            # expand 2 (dead) -> beam empties
    assert env.n_unscored_pulls == 1, "peek() must pull from the reservoir"
    assert env.fringe == [3], "the pulled node was never scored by the model"
    assert env.reservoir == []
    assert env.forced and env.available_actions() == [0]
    assert not r.done, "reservoir non-empty => not doom"

    r = env.step(0)                            # expand 3 -> generates goal 4
    assert r.info["outcome"] == "success"
    with capsys.disabled():
        print(f"\n  unscored pulls reachable: {env.n_unscored_pulls} on a 5-node tree")


def test_doom_is_unreachable_by_construction():
    """The theorem's operational form.

    DOOM <=> delta(root) = inf, and the env refuses to construct on an unsolvable
    instance (they are filtered at load). So the doom branch is DEAD CODE on
    every env that can exist -- which is the point, not an oversight. What
    remains live is the ASSERT: if doom ever fires on a solvable instance, the
    transition is dropping nodes and we want to hear about it immediately.
    """
    unsolvable = make_tree([[1, 2], [], []], goals=[], name="doomed")
    assert not unsolvable.solvable()
    with pytest.raises(ValueError, match="filtered at load"):
        _env(unsolvable, 2)


def test_a_beam_that_runs_out_is_not_doom_on_a_solvable_instance(t19):
    """Consecutive dead ends drain the beam, but the reservoir hands a node back
    (peek()'s unscored pull) -- so the search continues rather than dooming."""
    inst = make_tree([[1, 2, 3], [], [], [4], []], goals=[4], name="drain")
    env = _env(inst, 2)
    env.reset()
    env.step(0, ranking=[0, 1])
    r = env.step(0)
    assert r.info["outcome"] != "doom"
    assert env.n_unscored_pulls == 1


def test_root_is_never_goal_tested():
    """SpaceSearcher only tests successors, so a goal root is not detected."""
    inst = make_tree([[1], []], goals=[0], name="goalroot")
    env = _env(inst, 2)
    r = env.reset()
    assert r.info["outcome"] != "success"


@pytest.mark.parametrize("policy_name", BEHAVIOUR_POLICIES)
def test_transition_invariants_all_four_policies(t19, policy_name):
    """Test 5, stochastic-refill half: with F small the reservoir engages and the
    exact (B,R) depends on the RNG, so assert the mechanics that must hold on
    EVERY step for every policy, rather than a recorded trace."""
    for seed in range(25):
        env = _env(t19, 3, seed=seed)
        pol = make_policy(t19, policy_name, seed=seed)
        res = env.reset(seed=seed)
        steps = 0
        while not res.done:
            open_set = set(env.fringe) | set(env.reservoir)
            assert len(env.fringe) <= env.fringe_size
            assert not (set(env.fringe) & set(env.reservoir)), "B and R must be disjoint"
            assert len(env.fringe) == len(set(env.fringe)), "no duplicate beam slots"
            assert open_set <= env.visited, "open nodes must all have been generated"
            if env.forced:
                action, ranking = env.forced_action, None
            else:
                ranking = pol(env.fringe)
                action = ranking[0]
            res = env.step(action, ranking)
            steps += 1
            assert steps < 100, "episode did not terminate"
        assert res.info["outcome"] in ("success", "doom", "timeout")


def test_bfs_expands_in_nondecreasing_depth(t19):
    """With F >= max open the beam holds every open node, so `bfs` degenerates to
    a true breadth-first order -- including across the stale dead-end path, where
    the retained ranking is still the BFS one."""
    for seed in range(20):
        env = _env(t19, 19, seed=seed)
        pol = make_policy(t19, "bfs", seed=seed)
        res = env.reset(seed=seed)
        depths = [t19.depth[t19.root_id]]
        while not res.done:
            if env.forced:
                action, ranking = env.forced_action, None
            else:
                ranking = pol(env.fringe)
                action = ranking[0]
            depths.append(t19.depth[env.fringe[action]])
            res = env.step(action, ranking)
        assert depths == sorted(depths), f"seed {seed}: bfs expanded {depths}"


def test_dfs_dives(t19):
    """`dfs` expands the deepest beam member; it must never expand a node
    shallower than one it could have taken."""
    for seed in range(20):
        env = _env(t19, 19, seed=seed)
        pol = make_policy(t19, "dfs", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                action, ranking = env.forced_action, None
            else:
                ranking = pol(env.fringe)
                action = ranking[0]
                chosen = env.fringe[action]
                assert t19.depth[chosen] == max(t19.depth[v] for v in env.fringe)
            res = env.step(action, ranking)


# ------------------------------------------------ test 4: hfs_oracle = pi* -----

def test_hfs_oracle_spends_exactly_delta_root(t19):
    """F=19 >= max open, so the reservoir always drains back into the beam and
    refill has no effect on WHICH node is available -- 'deterministic refill'.
    Under those conditions pi* expands exactly delta(root) nodes."""
    for seed in range(30):
        env = _env(t19, 19, seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
            else:
                ranking = oracle_ranking(t19, env.fringe)
                res = env.step(ranking[0], ranking)
        assert res.info["outcome"] == "success", f"seed {seed}: hfs_oracle doomed"
        assert res.info["expansions"] == t19.delta_root == 4
        assert t19.regret(res.info["expansions"]) == 0.0


def test_hfs_oracle_never_dooms_with_random_tiebreaks(t19):
    """The tie-broken behaviour policy is still clairvoyant: any argmin-delta
    choice is equally optimal, so ties cannot cost expansions."""
    for seed in range(50):
        env = _env(t19, 19, seed=seed)
        pol = make_policy(t19, "hfs_oracle", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
            else:
                ranking = pol(env.fringe)
                res = env.step(ranking[0], ranking)
        assert res.info["outcome"] == "success"
        assert res.info["expansions"] == t19.delta_root


def test_hfs_oracle_on_a_deeper_synthetic_tree():
    """A chain with decoys: delta(root) = 5, and pi* must not be distracted."""
    #  0 -> 1(dead), 2 ; 2 -> 3(dead), 4 ; 4 -> 5 ; 5 -> 6 ; 6 -> 7(goal)
    ch = [[1, 2], [], [3, 4], [], [5], [6], [7], []]
    inst = make_tree(ch, goals=[7], name="chain")
    assert inst.delta_root == 5
    for seed in range(20):
        env = _env(inst, 8, seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
            else:
                rk = oracle_ranking(inst, env.fringe)
                res = env.step(rk[0], rk)
        assert res.info["outcome"] == "success"
        assert res.info["expansions"] == 5


def test_v_star_is_only_an_optimistic_bound_under_random_refill(capsys):
    """Documented consequence of RefillMode::RANDOM -- demonstrated, not asserted.

        0 -> 1, 2, 3      delta: 1->2, 2->2, 3->1   =>  delta(root) = 2
        F = 2  =>  batch = children[:2] = [1,2], and 3 OVERFLOWS to the reservoir

    The unique optimal move (expand 3, whose child is a goal) is not in the beam
    at all: the policy cannot take it, and whether 3 ever comes back is a random
    refill draw rather than a decision. So even pi* pays regret on EVERY seed,
    while V*(B,R) = -min delta over B u R = -1 claims one expansion suffices.

    That gap is exactly why V* is an optimistic BOUND and why every figure using
    it must say so. (On t19 this never triggers -- its good children happen not
    to overflow at F=2 -- which is why the phenomenon needs its own fixture
    rather than a hopeful loop over an unrelated tree.)
    """
    ch = [[1, 2, 3], [4], [6], [8], [5], [], [7], [], []]
    inst = make_tree(ch, goals=[5, 7, 8], name="overflow")
    assert inst.delta_root == 2

    misses = 0
    for seed in range(60):
        env = _env(inst, 2, seed=seed)
        res = env.reset(seed=seed)
        assert env.fringe == [1, 2] and env.reservoir == [3], "node 3 must overflow"
        assert env.v_star() == -1.0, "V* counts the reservoir, so it claims -1"
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
            else:
                rk = oracle_ranking(inst, env.fringe)
                res = env.step(rk[0], rk)
        assert res.info["outcome"] == "success"
        assert res.info["expansions"] >= inst.delta_root, "delta(root) is a floor"
        if res.info["expansions"] > inst.delta_root:
            misses += 1
    assert misses == 60, (
        "pi* must miss delta(root) on every seed here: the optimal node sits in "
        "the reservoir and refill is not under the policy's control"
    )
    with capsys.disabled():
        print(f"\n  pi* at F=2 misses delta(root) on {misses}/60 seeds "
              f"(optimal node overflowed to R; refill is not policy-controlled)")


def test_v_star_includes_the_reservoir(t19):
    env = _env(t19, 2)
    env.reset()                                # B=[1,2]  R=[3]
    # delta: 1->3, 2->inf, 3->3
    assert env.v_star() == -3.0
    # a node parked in R still counts: drop the beam's best and V* is unchanged
    assert t19.v_star([2], [3]) == -3.0
    assert t19.v_star([2], []) == -INF_DELTA


# ------------------------------------------ test 6: reservoir conservation ----

@pytest.mark.parametrize("policy_name", BEHAVIOUR_POLICIES)
@pytest.mark.parametrize("F", [2, 3, 5, 19])
def test_no_node_is_ever_lost(t19, policy_name, F):
    """|B u R u expanded| == |generated so far| at EVERY step.

    This is the completeness guarantee in one number: the reservoir is why an
    RL policy with no completeness property can be deployed at all.
    """
    for seed in range(15):
        env = _env(t19, F, seed=seed)
        pol = make_policy(t19, policy_name, seed=seed)
        res = env.reset(seed=seed)
        assert env.live_node_count() == len(env.visited)
        while not res.done:
            if env.forced:
                action, ranking = env.forced_action, None
            else:
                ranking = pol(env.fringe)
                action = ranking[0]
            res = env.step(action, ranking)
            assert env.live_node_count() == len(env.visited), (
                f"{policy_name} F={F} seed={seed}: node lost at expansion "
                f"{env.expansions}"
            )


def test_expanded_beam_and_reservoir_partition_the_generated_set(t19):
    for seed in range(10):
        env = _env(t19, 3, seed=seed)
        pol = make_policy(t19, "random", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            b, r = set(env.fringe), set(env.reservoir)
            assert not (b & r)
            assert (b | r) <= env.visited
            if env.forced:
                res = env.step(env.forced_action)
            else:
                rk = pol(env.fringe)
                res = env.step(rk[0], rk)


def test_goals_never_enter_the_beam(t19):
    """The goal test precedes the visited insert (SpaceSearcher.tpp:162), so a
    goal ends the search at generation and is never a beam member."""
    for seed in range(20):
        env = _env(t19, 5, seed=seed)
        pol = make_policy(t19, "random", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            assert not any(t19.is_goal[v] for v in env.fringe)
            assert not any(t19.is_goal[v] for v in env.reservoir)
            if env.forced:
                res = env.step(env.forced_action)
            else:
                rk = pol(env.fringe)
                res = env.step(rk[0], rk)


def test_every_node_expanded_at_most_once(t19):
    for seed in range(15):
        env = _env(t19, 3, seed=seed)
        pol = make_policy(t19, "dfs", seed=seed)
        res = env.reset(seed=seed)
        seen = [t19.root_id]
        while not res.done:
            if env.forced:
                action, ranking = env.forced_action, None
            else:
                ranking = pol(env.fringe)
                action = ranking[0]
            seen.append(env.fringe[action])
            res = env.step(action, ranking)
        assert len(seen) == len(set(seen)), "a node was expanded twice"


# ------------------------------------- terminated vs truncated (bootstrap) ---

def test_truncation_is_not_termination_and_returns_a_successor():
    """The cap must NOT be terminal.

    A capped rollout is one we stopped watching, not a search that failed: by the
    completeness proposition the planner would have kept going and eventually
    succeeded. So the critic must bootstrap, which means the successor fringe has
    to come back -- an empty one would leave nothing to bootstrap from.
    """
    N = 40                                   # 0 -> 1 -> ... -> 39 (goal)
    ch = [[i + 1] for i in range(N - 1)] + [[]]
    inst = make_tree(ch, goals=[N - 1], name="deep")
    assert inst.delta_root == N - 1
    env = FringeEnv(inst, fringe_size=4, gamma=1.0, seed=0, expansion_cap=5)
    res = env.reset(seed=0)
    while not res.done:
        rk = list(range(len(env.fringe)))
        res = env.step(rk[0], rk)
    assert res.truncated is True
    assert res.terminated is False, "the cap must never set terminated"
    assert res.done is True, "but the loop must still stop"
    assert res.fringe, "truncation must return the successor beam to bootstrap from"
    assert res.reward == -1.0, "a truncated step is an ordinary step"
    assert res.info["outcome"] == "timeout"


def test_success_is_terminated():
    env = FringeEnv(make_tree([[1], []], goals=[1], name="s"), fringe_size=2)
    r = env.reset()
    assert r.terminated and not r.truncated and r.info["outcome"] == "success"
    assert r.reward == 0.0


def test_doom_on_a_solvable_instance_is_an_env_bug(t19, monkeypatch):
    """The completeness proposition made executable: if DOOM ever fires on an
    instance with delta(root) < inf, the transition is dropping nodes. Fail
    loudly rather than absorb it into a reward."""
    env = _env(t19, 2)
    env.reset()
    # Simulate a transition that loses the reservoir (the bug this guards).
    env.fringe = []
    env.reservoir = []
    with pytest.raises(AssertionError, match="DOOM fired on SOLVABLE"):
        env._doom()


def test_coverage_aggregation_never_averages_unsolved_runs():
    from src.offline.env import aggregate_rollouts
    rs = [
        {"solved": True, "regret": 2.0, "expansions": 10, "outcome": "success", "truncated": False},
        {"solved": False, "regret": None, "expansions": 99, "outcome": "timeout", "truncated": True},
    ]
    agg = aggregate_rollouts(rs)
    assert agg["coverage"] == 0.5
    assert agg["timeout_rate"] == 0.5
    assert agg["regret_mean"] == 2.0, "the truncated run must not enter the mean"
    assert agg["expansions_mean"] == 10.0


# ------------------------------------------- real-data regression (test 4) ----

def test_hfs_oracle_hits_delta_root_on_real_data(shipped_instances, capsys):
    """The strongest validation the env has: on the real CC tree, pi* spends
    exactly delta(root) = 34 expansions with regret 0, at the deployed F=32."""
    inst = shipped_instances["CC_2_3_4__pl_7"]
    from src.offline.env import rollout
    from src.offline.policies import make_policy
    for seed in range(10):
        env = FringeEnv(inst, fringe_size=32, seed=seed, gamma=1.0,
                        expansion_cap=900)
        r = rollout(env, make_policy(inst, "hfs_oracle", seed=seed), seed=seed)
        assert r["solved"], f"seed {seed}: pi* failed on real data"
        assert r["expansions"] == inst.delta_root == 34
        assert r["regret"] == 0.0
    with capsys.disabled():
        print(f"\n  CC_2_3_4__pl_7 F=32: hfs_oracle = {inst.delta_root:.0f} "
              f"expansions, regret 0 on 10/10 seeds")


def test_doom_is_zero_on_real_data_for_every_policy(shipped_instances, capsys):
    """The completeness proposition, checked on the real tree rather than argued."""
    from src.offline.env import aggregate_rollouts, rollout
    from src.offline.policies import make_policy
    inst = shipped_instances["CC_2_3_4__pl_7"]
    for name in BEHAVIOUR_POLICIES:
        rs = [
            rollout(
                FringeEnv(inst, fringe_size=32, seed=s, gamma=1.0, expansion_cap=900),
                make_policy(inst, name, seed=s), seed=s,
            )
            for s in range(8)
        ]
        agg = aggregate_rollouts(rs)
        assert agg["doom_rate"] == 0.0, (
            f"{name} doomed on a solvable instance -- the reservoir should make "
            f"that impossible"
        )
        with capsys.disabled():
            print(f"    {name:11s} coverage={agg['coverage']:.2f} "
                  f"doom={agg['doom_rate']:.2f} regret={agg['regret_mean']}")


def test_dead_end_rate_on_t19_is_representative(t19):
    """The fixture deliberately mirrors the real data's hot dead-end path
    (CC_2_3_4__pl_7 is 17.5% dead ends)."""
    n_reach = len(t19._reachable())
    assert len(T19_DEAD) / n_reach > 0.15
    assert len(T19_GOALS) == 3
    assert sorted({len(c) for c in t19.children}) == [0, 1, 2, 3, 4]
