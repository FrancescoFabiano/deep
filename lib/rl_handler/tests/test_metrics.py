"""Node-level failure metrics: sterile expansions, eviction, viability AUC.

The completeness proposition removed episode DOOM. It did NOT remove failure:
~35% of CC_2_3_4__pl_7's nodes have delta = inf (their whole subtree contains no
goal). Those are the failure states, they sit in every beam, and avoiding them is
the primary thing the policy must learn.
"""

from __future__ import annotations

import pytest

from src.offline.env import FringeEnv, rollout
from src.offline.metrics import (
    beam_metrics,
    mean_ignoring_none,
    spearman_logits_vs_delta,
    top1_oracle_agreement,
    viability_auc,
)
from src.offline.policies import make_policy
from src.offline.tree import INF_DELTA

from conftest import make_tree


# ------------------------------------------------------- viability AUC -------

def test_viability_auc_is_one_when_viable_ranked_first():
    # higher logit = expanded sooner, so viable slots must carry higher logits
    logits = [5.0, 4.0, -1.0, -2.0]
    deltas = [1.0, 2.0, INF_DELTA, INF_DELTA]
    assert viability_auc(logits, deltas) == 1.0


def test_viability_auc_is_zero_when_perfectly_inverted():
    logits = [-1.0, -2.0, 5.0, 4.0]
    deltas = [1.0, 2.0, INF_DELTA, INF_DELTA]
    assert viability_auc(logits, deltas) == 0.0


def test_viability_auc_is_half_for_an_uninformative_model():
    """The number that says 'learned nothing', whatever the TD loss says."""
    logits = [1.0, 1.0, 1.0, 1.0]
    deltas = [1.0, INF_DELTA, 2.0, INF_DELTA]
    assert viability_auc(logits, deltas) == 0.5


def test_viability_auc_is_none_when_no_comparison_exists():
    """All-viable or all-sterile beams have no AUC. Returning 0.5 would
    fabricate a data point and drag the average toward 'learned nothing'."""
    assert viability_auc([1.0, 2.0], [1.0, 2.0]) is None
    assert viability_auc([1.0, 2.0], [INF_DELTA, INF_DELTA]) is None
    assert mean_ignoring_none([None, 1.0, None]) == 1.0
    assert mean_ignoring_none([None, None]) is None


def test_viability_auc_handles_ties_as_half():
    logits = [1.0, 1.0]
    deltas = [1.0, INF_DELTA]
    assert viability_auc(logits, deltas) == 0.5


# ------------------------------------------------------ ranking metrics ------

def test_top1_oracle_agreement():
    assert top1_oracle_agreement([0.1, 9.0, 0.2], [5.0, 1.0, 4.0]) == 1.0
    assert top1_oracle_agreement([9.0, 0.1, 0.2], [5.0, 1.0, 4.0]) == 0.0


def test_spearman_is_one_for_a_perfect_ranker():
    logits = [3.0, 2.0, 1.0]
    deltas = [1.0, 2.0, 3.0]
    assert spearman_logits_vs_delta(logits, deltas) == pytest.approx(1.0)


def test_spearman_maps_inf_above_every_finite_delta():
    """Sterile slots are kept, not dropped: ranking them last IS the task."""
    logits = [3.0, 2.0, 1.0]
    deltas = [1.0, 2.0, INF_DELTA]
    assert spearman_logits_vs_delta(logits, deltas) == pytest.approx(1.0)


def test_spearman_is_none_when_undefined():
    assert spearman_logits_vs_delta([1.0, 1.0], [1.0, 1.0]) is None


def test_beam_metrics_bundle(t19):
    beam = [1, 2, 3]                      # delta 3, inf, 3
    m = beam_metrics(t19, beam, [1.0, -1.0, 0.5])
    assert m["viability_auc"] == 1.0
    assert m["beam_sterile_frac"] == pytest.approx(1 / 3)


# ------------------------------------------- sterile expansions + eviction ---

def test_oracle_never_expands_a_sterile_node(t19):
    from src.offline.policies import oracle_ranking
    env = FringeEnv(t19, fringe_size=19, seed=0, expansion_cap=200)
    res = env.reset(seed=0)
    while not res.done:
        if env.forced:
            res = env.step(env.forced_action)
        else:
            rk = oracle_ranking(t19, env.fringe)
            res = env.step(rk[0], rk)
    assert env.n_sterile_expansions == 0


def test_sterile_beam_is_nonempty_even_for_the_oracle(t19):
    """The task is real: sterile nodes ARE in the beam; pi* just never expands
    them. If the beam were all-viable there would be nothing to rank."""
    env = FringeEnv(t19, fringe_size=19, seed=0, expansion_cap=200)
    env.reset(seed=0)
    assert env.beam_sterile_frac() > 0.0


def test_eviction_requires_the_beam_to_bind():
    """No binding, no eviction: if the open set fits in F, nothing is ever pushed
    out. This is why the compounding loop is exactly the regime where F matters
    -- measured: dfs on CC at F=32 spends 43% of expansions sterile yet evicts
    nothing, because its open set peaks at 18 < 32."""
    #   0 -> 1(good, delta 1), 2(sterile), 3(sterile)
    ch = [[1, 2, 3], [4], [], [], []]
    inst = make_tree(ch, goals=[4], name="evict")
    # F=3 holds the whole beam -> expanding a sterile node cannot evict node 1
    env = FringeEnv(inst, fringe_size=3, seed=0, expansion_cap=50)
    env.reset(seed=0)
    env.step(1, ranking=[1, 0, 2])          # expand sterile node 2 (dead end)
    assert env.evictions == [], "nothing can be evicted when the beam does not bind"


def test_eviction_is_recorded_and_recovered(capsys):
    """Expanding a sterile node with children floods the beam and pushes the good
    node into R, from which only a random draw returns it."""
    #   0 -> 1(good), 2(sterile with 2 sterile children)
    ch = [[1, 2], [3], [4, 5], [], [], []]
    inst = make_tree(ch, goals=[3], name="evict2")
    seen = 0
    for s in range(20):
        env = FringeEnv(inst, fringe_size=2, seed=s, expansion_cap=50)
        env.reset(seed=s)                    # B=[1,2], R=[]
        assert env.fringe == [1, 2]
        # expand sterile 2 -> children [4,5] fill B at F=2 -> node 1 evicted to R
        env.step(1, ranking=[0, 1])
        if 1 in env.reservoir:
            seen += 1
            assert env.fringe == [4, 5]
    assert seen == 20, "the good node must be evicted on every seed"
    with capsys.disabled():
        print("\n  eviction: expanding a sterile node flooded B with its sterile "
              "children and pushed the only viable node into R")


def test_rollout_reports_the_failure_and_eviction_telemetry(t19):
    env = FringeEnv(t19, fringe_size=3, seed=0, expansion_cap=200)
    r = rollout(env, make_policy(t19, "random", seed=0), seed=0)
    for k in ("expansions_sterile_frac", "beam_sterile_frac", "eviction_events",
              "eviction_recovery_steps_mean", "eviction_never_recovered",
              "reservoir_size_mean", "reservoir_size_max", "beam_size_max"):
        assert k in r, k
    assert 0.0 <= r["expansions_sterile_frac"] <= 1.0


def test_sterile_expansion_rate_separates_oracle_from_bfs(shipped_instances, capsys):
    """The headline of F10, on real data: this is the clearest single measure of
    whether the policy learned anything."""
    inst = shipped_instances["CC_2_3_4__pl_7"]
    out = {}
    for name in ("hfs_oracle", "bfs"):
        rs = [
            rollout(
                FringeEnv(inst, fringe_size=32, seed=s, expansion_cap=1200),
                make_policy(inst, name, seed=s), seed=s,
            )
            for s in range(3)
        ]
        out[name] = sum(r["expansions_sterile_frac"] for r in rs) / len(rs)
    assert out["hfs_oracle"] == 0.0, "pi* must never expand a dead subtree"
    assert out["bfs"] > 0.4, f"bfs should waste heavily on sterile nodes: {out}"
    with capsys.disabled():
        print(f"\n  CC F=32 expansions_sterile_frac: hfs_oracle="
              f"{out['hfs_oracle']:.3f}  bfs={out['bfs']:.3f}")
