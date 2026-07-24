"""Behaviour policies for offline data generation.

Four behaviours, ONE mechanism, parameterised by a priority score sigma(v)
(lower = preferred):

    bfs         sigma(v) = depth(v)
    dfs         sigma(v) = -depth(v)
    hfs_oracle  sigma(v) = delta(v)
    random      sigma(v) = U(0,1)

RANDOMISED TIE-BREAKING IS MANDATORY
------------------------------------
    v = argmin over B of sigma~(v),  sigma~(v) = (sigma(v), u_v),  u_v ~ U(0,1)

Without it every deterministic behaviour yields exactly one trajectory per
instance and the dataset collapses. Measured on CC_2_3_4__pl_7: deterministic
tie-breaking gives DFS a 0% doom rate; random tie-breaking gives 52%.

hfs_oracle IS NOT A BASELINE
----------------------------
It ranks by `delta`, the answer to the problem, so it is CLAIRVOYANT: it is pi*,
not a competitor, and it is not executable at deployment. It serves as (a) the
evaluation ceiling and (b) expert demonstrations. The name stays `hfs_oracle` in
the code and in every plot legend so it cannot be mistaken for a baseline.

A policy returns a RANKING (slot indices, best first), not a single action: the
planner keeps the whole priority order and reuses it on the stale-rank dead-end
path, so the env needs all of it.
"""

from __future__ import annotations

import random
from typing import Callable, List, Sequence

from .tree import TreeInstance

BEHAVIOUR_POLICIES = ("bfs", "dfs", "hfs_oracle", "random")

# A ranking policy: (beam: Sequence[int]) -> List[int] slot indices, best first.
RankingPolicy = Callable[[Sequence[int]], List[int]]


def _sigma_fn(instance: TreeInstance, name: str, rng: random.Random):
    if name == "bfs":
        return lambda v: float(instance.depth[v])
    if name == "dfs":
        return lambda v: -float(instance.depth[v])
    if name == "hfs_oracle":
        return lambda v: float(instance.delta[v])
    if name == "random":
        return lambda v: rng.random()
    raise ValueError(f"unknown behaviour policy {name!r}; expected one of {BEHAVIOUR_POLICIES}")


def make_policy(
    instance: TreeInstance,
    name: str,
    seed: int = 0,
) -> RankingPolicy:
    """Build a ranking policy over `instance` with random tie-breaking.

    Ties are broken by an independent U(0,1) draw per candidate per state, so
    `sigma~ = (sigma, u)` compared lexicographically. Re-drawing per state (not
    once per node) is what makes repeated seeds explore genuinely different
    trajectories through the same tree.
    """
    rng = random.Random(seed)
    sigma = _sigma_fn(instance, name, rng)

    def _policy(beam: Sequence[int]) -> List[int]:
        keyed = [(sigma(v), rng.random(), k) for k, v in enumerate(beam)]
        keyed.sort()
        return [k for _, _, k in keyed]

    return _policy


def oracle_ranking(instance: TreeInstance, beam: Sequence[int]) -> List[int]:
    """Deterministic clairvoyant ranking by delta (ties -> lowest slot).

    Used by the diagnostics that need pi* without behaviour randomness, e.g. the
    synthetic-tree test asserting hfs_oracle spends exactly delta(root)
    expansions.
    """
    return sorted(range(len(beam)), key=lambda k: (instance.delta[beam[k]], k))
