"""Fix 1A: censored (generation-artifact) vs provably sterile delta=inf.

A depth-bound leaf records b_v=0 / delta=inf because GENERATION stopped, not
because the state is a dead end -- at deployment it expands normally and may
sit on the shortest path (usability.py: 145/154 "poisoned" rows on CC pl_7).
Training must not learn "avoid v" from that label, and metrics must not judge
the model against it.
"""

from __future__ import annotations

import pytest
import torch

from conftest import make_tree
from src.offline.dataset import generate_dataset
from src.offline.env import FringeEnv, g_succ, realized_success_return, rollout
from src.offline.policies import make_policy, oracle_ranking
from src.offline.selection import project_ranking, rankable_slots
from src.offline.tree import INF_DELTA, UNREACHABLE_DISTANCE, compute_censored
from src.models.two_head_baseline import two_head_loss


# ---- the censoring fixture ---------------------------------------------------
#
#   0 d0 -> 1, 2, 5
#   1 d1 -> 3                       delta=inf  (subtree ends at the bound)
#   2 d1 -> 4                       delta=2    (goal below)
#   3 d2 -> []   BOUND-HIT LEAF     delta=inf, h*=1e6, depth == max depth
#   4 d2 -> 6    GOAL PARENT        delta=1
#   5 d1 -> []   GENUINE DEAD END   delta=inf, h*=1e6, depth < max depth
#   6 d3 GOAL
#
# depth_bound = max depth = 3?  No: node 6 is a goal at depth 3; node 3 sits at
# depth 2.  Use an explicit depth_bound=2 so the test does not depend on the
# max-depth fallback picking up the goal's depth.
CHILDREN = [[1, 2, 5], [3], [4], [], [6], [], []]
GOALS = [6]


def _censored_tree():
    inst = make_tree(CHILDREN, GOALS, name="cens")
    # Real generator marks unreachable rows with 1e6 (not inf).
    inst.h_star = [d if d != INF_DELTA else UNREACHABLE_DISTANCE for d in inst.delta]
    inst.censored = compute_censored(
        inst.children, inst.is_goal, inst.delta, inst.h_star, inst.depth,
        depth_bound=2,
    )
    return inst


def test_bound_hit_leaf_is_censored_and_propagates():
    inst = _censored_tree()
    assert inst.censored[3] is True          # bound-hit leaf
    assert inst.censored[1] is True          # inf ancestor of a censored leaf
    assert inst.censored[5] is False         # genuine dead end below the bound
    assert inst.censored[0] is False         # finite delta -> never censored
    assert inst.censored[2] is False
    assert inst.provably_sterile(5) and not inst.provably_sterile(3)


def test_h_star_contradiction_is_censored_at_any_depth():
    inst = make_tree(CHILDREN, GOALS, name="contra")
    h = [d if d != INF_DELTA else UNREACHABLE_DISTANCE for d in inst.delta]
    h[5] = 4.0        # generator says a goal IS reachable from 5 in the DAG
    censored = compute_censored(inst.children, inst.is_goal, inst.delta, h,
                                inst.depth, depth_bound=99)
    assert censored[5] is True               # contradiction beats depth
    assert censored[3] is False              # below bound, h*=1e6: stays sterile


def test_oracle_prefers_censored_over_provably_sterile():
    inst = _censored_tree()
    # beam of [censored leaf 3, sterile leaf 5, viable 4]
    rank = oracle_ranking(inst, [3, 5, 4])
    assert rank == [2, 0, 1]                 # finite < censored < sterile
    pol = make_policy(inst, "hfs_oracle", seed=0)
    assert pol([3, 5, 4])[0] == 2


def test_generate_dataset_drops_censored_leaf_rows():
    inst = _censored_tree()
    rows, summary = generate_dataset(
        [inst], fringe_size=4, policies=("bfs",), seeds_per_policy=2,
        expansion_cap=50, drop_censored=False, verbose=False)
    censored_rows = [r for r in rows if r.censored_expansion]
    assert censored_rows, "fixture must exercise a censored-leaf expansion"
    assert all(inst.censored[r.obs[r.action]] for r in censored_rows)

    kept, summary2 = generate_dataset(
        [inst], fringe_size=4, policies=("bfs",), seeds_per_policy=2,
        expansion_cap=50, drop_censored=True, verbose=False)
    assert not any(r.censored_expansion for r in kept)
    assert summary2["n_censored_rows"] == len(censored_rows)
    # dropping labels must not change how many rollouts/states were visited
    assert summary2["censored_rows_dropped"] is True


def test_rankable_slots_and_projection():
    inst = _censored_tree()
    beam = [4, 3, 5, 2]                      # viable, censored, sterile, viable
    keep = rankable_slots(inst, beam)
    assert keep == [0, 2, 3]
    # full-beam ranking [1, 3, 0, 2] -> censored slot 1 removed, order kept,
    # slots remapped to positions within keep
    assert project_ranking([1, 3, 0, 2], keep) == [2, 0, 1]


def test_two_head_loss_masks_censored_slots():
    p_logit = torch.tensor([0.0, 5.0, -5.0])
    d_hat = torch.tensor([1.0, 2.0, 3.0])
    viable = torch.tensor([1.0, 0.0, 0.0])   # slot 1 is censored "non-viable"
    delta = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([True, False, True])
    loss_m, log_m = two_head_loss(p_logit, d_hat, viable, delta, label_mask=mask)
    loss_k, _ = two_head_loss(p_logit[mask], d_hat[mask], viable[mask], delta[mask])
    assert log_m["n_censored"] == 1
    assert torch.isclose(loss_m, loss_k)


def test_env_counts_censored_expansions_separately():
    inst = _censored_tree()
    env = FringeEnv(inst, fringe_size=4, seed=0, expansion_cap=50)
    out = rollout(env, make_policy(inst, "bfs", seed=0), seed=0)
    assert out["n_sterile_expansions"] + out["n_censored_expansions"] <= out["expansions"]
    # the fixture has exactly one provably sterile node (5) and one censored
    # branch (1 -> 3); bfs expands everything above the goal depth
    assert out["n_censored_expansions"] >= 1
    assert out["n_sterile_expansions"] >= 1


def test_realized_success_return_convention():
    assert realized_success_return(3, 1.0) == g_succ(1, 1.0) == -1.0
    assert realized_success_return(2, 1.0) == 0.0
    inst = make_tree([[1], [2], [3], []], [3], name="chain")  # delta_root = 3
    env = FringeEnv(inst, fringe_size=4, seed=0, gamma=1.0, expansion_cap=50)
    out = rollout(env, make_policy(inst, "hfs_oracle", seed=0), seed=0)
    assert out["solved"] and out["expansions"] == 3
    assert out["return"] == realized_success_return(3, 1.0)
