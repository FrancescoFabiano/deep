"""Test 7 — counterfactual action expansion.

The tree is fixed and fully in memory, so the transition is a MODEL. |A(s)| <= F
<= 64, so every action can be ENUMERATED rather than sampled: from each visited
state emit one row per beam slot, then follow only ONE successor (branching on
all |B| would re-enumerate the search space).

This is what gives on-distribution STATES (from the behaviour policies) with full
ACTION coverage (from enumeration). Without it the critic sees each state once
with one action and has nothing to compare.
"""

from __future__ import annotations

import pytest

from src.offline.env import FringeEnv
from src.offline.policies import make_policy

from conftest import make_tree


def _env(inst, F, seed=0):
    # gamma left at the default (0.9999): these tests are about counterfactual
    # enumeration, not the objective. gamma=0.99 no longer validates against the
    # default cap (horizon 100 < cap 2000 fails the dominance check in assert_gamma).
    return FringeEnv(inst, fringe_size=F, seed=seed)


def enumerate_actions(env: FringeEnv, ranking):
    """Every action from `env`'s current state, via cloned model steps.

    Each clone carries the same RNG state, so the successors differ because the
    ACTION differed, not because the refill draw differed.
    """
    rows = []
    for a in range(len(env.fringe)):
        c = env.clone()
        res = c.step(a, ranking=list(ranking))
        rows.append({
            "action": a,
            "state": env.fringe[a],
            "obs": list(env.fringe),
            "obs_next": list(res.fringe),
            "reward": res.reward,
            "done": res.done,
            "outcome": res.info["outcome"],
            "reservoir_next": list(c.reservoir),
            "v_star_s": env.v_star(),
            "v_star_s_next": c.v_star(),
        })
    return rows


def test_one_row_per_beam_slot(t19):
    env = _env(t19, 3)
    env.reset()
    ranking = list(range(len(env.fringe)))
    rows = enumerate_actions(env, ranking)
    assert len(rows) == len(env.fringe) == 3
    assert [r["action"] for r in rows] == [0, 1, 2]
    assert {r["state"] for r in rows} == set(env.fringe)


def test_successors_are_distinct(t19):
    """Different actions must lead to different successors -- otherwise the rows
    carry no comparative signal."""
    env = _env(t19, 3)
    env.reset()
    rows = enumerate_actions(env, list(range(len(env.fringe))))
    sigs = [(tuple(r["obs_next"]), tuple(r["reservoir_next"]), r["outcome"]) for r in rows]
    assert len(set(sigs)) == len(sigs), f"non-distinct successors: {sigs}"


def test_exactly_one_row_matches_the_followed_trajectory(t19):
    for seed in range(20):
        env = _env(t19, 3, seed=seed)
        pol = make_policy(t19, "bfs", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
                continue
            ranking = pol(env.fringe)
            rows = enumerate_actions(env, ranking)
            taken = ranking[0]
            matches = [r for r in rows if r["action"] == taken]
            assert len(matches) == 1

            follow = env.step(taken, ranking)
            assert matches[0]["obs_next"] == list(follow.fringe), (
                "the enumerated row for the taken action must equal the followed "
                "successor -- clone() and step() must agree, RNG included"
            )
            assert matches[0]["reward"] == follow.reward
            assert matches[0]["done"] == follow.done
            res = follow


def test_enumeration_does_not_mutate_the_parent(t19):
    env = _env(t19, 3)
    env.reset()
    before = (list(env.fringe), list(env.reservoir), set(env.visited), env.expansions)
    enumerate_actions(env, list(range(len(env.fringe))))
    after = (list(env.fringe), list(env.reservoir), set(env.visited), env.expansions)
    assert before == after, "cloning must not touch the parent env"


def test_clone_is_independent_and_rng_faithful(t19):
    env = _env(t19, 3, seed=7)
    env.reset(seed=7)
    a = env.clone()
    b = env.clone()
    ra = a.step(0, ranking=[0, 1, 2])
    rb = b.step(0, ranking=[0, 1, 2])
    assert ra.fringe == rb.fringe, "clones must face the same refill draw"
    assert a.reservoir == b.reservoir
    assert env.expansions == 1


def test_forced_states_carry_zero_advantage(t19):
    """`forced` rows are valid Bellman backups for the critic and useless for the
    actor: there is exactly one action, so no comparison exists."""
    inst = make_tree([[1, 2, 3], [], [4], [5], [], []], goals=[4], name="wide")
    env = _env(inst, 3)
    env.reset()
    env.step(0, ranking=[0, 1, 2])          # expand 1 (dead end) -> forced
    assert env.forced
    assert env.n_actions() == 1
    assert len(env.available_actions()) == 1


def test_advantage_is_computable_from_v_star(t19):
    """advantage = -1 + V*(s') - V*(s), exact and diagnostic-only."""
    env = _env(t19, 3)
    env.reset()
    rows = enumerate_actions(env, list(range(len(env.fringe))))
    for r in rows:
        if r["outcome"] == "success":
            continue
        adv = -1.0 + r["v_star_s_next"] - r["v_star_s"]
        assert adv <= 1e-9, (
            "V* cannot improve by more than one expansion's worth per step"
        )


def test_action_coverage_beats_single_action_rows(t19, capsys):
    """The measured point of enumeration: actions/state goes from ~1 to ~|B|."""
    n_states = 0
    n_rows = 0
    for seed in range(30):
        env = _env(t19, 5, seed=seed)
        pol = make_policy(t19, "random", seed=seed)
        res = env.reset(seed=seed)
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
                continue
            ranking = pol(env.fringe)
            n_rows += len(enumerate_actions(env, ranking))
            n_states += 1
            res = env.step(ranking[0], ranking)
    per_state = n_rows / max(1, n_states)
    with capsys.disabled():
        print(f"\n  t19 F=5: {n_states} states -> {n_rows} rows "
              f"({per_state:.2f} actions/state vs 1.0 without enumeration)")
    assert per_state > 1.5
