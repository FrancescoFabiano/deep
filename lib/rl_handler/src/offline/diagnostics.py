"""F8 — per-instance diagnostics, computed ONCE before training.

The headline is beam occupancy. The beam holds at most min(F, |B u R|), so an
instance whose open set never exceeds F cannot exercise the fringe knob AT ALL:
the beam never fills, refill never fires, and F is inert. Such an instance still
contributes to every F-sweep aggregate, silently dragging it toward "F does not
matter" -- which would be a conclusion about the instance pool, not about F.

Occupancy is POLICY-DEPENDENT and must be measured, not bounded. The search stops
when a goal is generated, so a policy that finds a goal later expands more nodes
and (the reservoir discarding nothing) accumulates a larger open set. Measured on
CC_2_3_4__pl_7 at F=32: max |B u R| is 117 under bfs but only 17 under
hfs_oracle. A BFS-trajectory statistic is a screening heuristic, not a bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .env import FringeEnv, default_expansion_cap
from .policies import BEHAVIOUR_POLICIES, make_policy
from .tree import INF_DELTA, TreeInstance, bfs_open_max


@dataclass
class Occupancy:
    fringe_size: int
    max_beam: int          # max |B| -- reaches F iff the beam binds
    max_open: int          # max |B u R| -- the live-node count
    binds: bool            # max_open > F: the fringe knob can actually move
    per_policy_open: Dict[str, int]


def measure_occupancy(
    instance: TreeInstance,
    fringe_size: int,
    policies: Iterable[str] = BEHAVIOUR_POLICIES,
    seeds: int = 4,
    expansion_cap: Optional[int] = None,
) -> Occupancy:
    """Roll the real behaviour policies in the real env and record occupancy.

    This is the authoritative binds/inert verdict: `binds` is True iff SOME
    behaviour policy drives the open set past F, i.e. iff refill can ever fire.
    """
    cap = expansion_cap if expansion_cap is not None else default_expansion_cap([instance])
    per_policy: Dict[str, int] = {}
    max_beam = 0
    for name in policies:
        best_open = 0
        for s in range(int(seeds)):
            env = FringeEnv(instance, fringe_size=fringe_size, seed=s, gamma=1.0,
                            expansion_cap=cap)
            pol = make_policy(instance, name, seed=s)
            res = env.reset(seed=s)
            best_open = max(best_open, len(env.fringe) + len(env.reservoir))
            max_beam = max(max_beam, len(env.fringe))
            while not res.done:
                if env.forced:
                    res = env.step(env.forced_action)
                else:
                    rk = pol(env.fringe)
                    res = env.step(rk[0], rk)
                best_open = max(best_open, len(env.fringe) + len(env.reservoir))
                max_beam = max(max_beam, len(env.fringe))
        per_policy[name] = best_open
    max_open = max(per_policy.values()) if per_policy else 0
    return Occupancy(
        fringe_size=int(fringe_size),
        max_beam=max_beam,
        max_open=max_open,
        binds=max_open > int(fringe_size),
        per_policy_open=per_policy,
    )


def instance_diagnostics(
    instance: TreeInstance,
    fringe_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    seeds: int = 4,
) -> Dict[str, object]:
    """The full F8 record for one instance."""
    reachable = instance._reachable()
    b_v = [len(instance.children[i]) for i in reachable]
    b_hist: Dict[int, int] = {}
    for b in b_v:
        b_hist[b] = b_hist.get(b, 0) + 1

    depth_hist: Dict[int, int] = {}
    for i in reachable:
        depth_hist[instance.depth[i]] = depth_hist.get(instance.depth[i], 0) + 1

    occ = {F: measure_occupancy(instance, F, seeds=seeds) for F in fringe_sizes}
    stats = instance.stats()
    return {
        "instance": instance.name,
        "n_states": instance.n_states,
        "n_reachable": len(reachable),
        "b_v_hist": dict(sorted(b_hist.items())),
        "branching_mean": stats["branching_internal_mean"],
        "depth_hist": dict(sorted(depth_hist.items())),
        "depth_max": stats["depth_max"],
        "forced_chain_len": stats["forced_chain_len"],
        "delta_root": instance.delta_root,
        "h_star_root": instance.h_star[instance.root_id],
        "delta_minus_h_star_root": stats["delta_minus_h_star_root"],
        "goal_density": stats["goal_density"],
        "sterile_leaf_density": stats["sterile_leaf_density"],
        "n_delta_inf": stats["n_delta_inf"],
        "bfs_open_max": bfs_open_max(instance),   # screening only, not a bound
        "occupancy": {
            F: {
                "max_beam": o.max_beam,
                "max_open": o.max_open,
                "binds": o.binds,
                "per_policy_open": o.per_policy_open,
            }
            for F, o in occ.items()
        },
        "binds_at": {F: o.binds for F, o in occ.items()},
        "inert_at": sorted([F for F, o in occ.items() if not o.binds]),
    }


def sweep_cohort(
    diags: Sequence[Dict[str, object]], fringe_size: int
) -> tuple[List[str], List[str]]:
    """(binding, inert) instance names at this F.

    F-sweep aggregates (F7) and the context comparison (F9) must be computed over
    the BINDING cohort only -- an inert instance cannot express any effect of F or
    of context over a beam it never fills. Inert instances stay in the COVERAGE
    numbers, where they remain perfectly valid instances.
    """
    binding = [d["instance"] for d in diags if d["binds_at"].get(fringe_size)]
    inert = [d["instance"] for d in diags if not d["binds_at"].get(fringe_size)]
    return binding, inert


def cohort_report(diags: Sequence[Dict[str, object]],
                  fringe_sizes: Sequence[int] = (4, 8, 16, 32, 64)) -> str:
    """Human-readable inert/binding split. Print it before any F-sweep claim."""
    lines = ["Fringe-knob cohort (an instance is INERT at F if max |B u R| <= F):"]
    for F in fringe_sizes:
        binding, inert = sweep_cohort(diags, F)
        n = len(binding) + len(inert)
        lines.append(
            f"  F={F:3d}: {len(binding):3d}/{n} binding, {len(inert):3d} inert"
            + (f"  -> F-sweep cohort: {', '.join(sorted(binding))}" if binding
               else "  -> NO instance can exercise F here; the sweep is meaningless")
        )
    return "\n".join(lines)
