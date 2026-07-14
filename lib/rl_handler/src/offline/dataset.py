"""Offline dataset generation by counterfactual action expansion.

The tree is fixed and fully in memory, so the transition is a MODEL. |A(s)| <= F
<= 64, so every action can be ENUMERATED rather than sampled:

    for pi in {bfs, dfs, hfs_oracle, random}:
      for seed in seeds:
        (B, R) <- reset()
        while not terminal:
            for v in B:                  # ALL actions -- emit a row for each
                emit(model_step(B, R, v))
            v_taken = pi(B)              # follow only ONE successor
            (B, R) <- step(B, R, v_taken)

Following a single successor is the point: branching on all |B| actions would
re-enumerate the search space. This yields on-distribution STATES (from the
behaviour policies) with full ACTION coverage (from enumeration). Without it the
critic sees each state once with one action and has nothing to compare.

FORCED STATES
A dead end leaves the planner with no choice (it pops the stale rank; the model
is not called). Those rows are valid Bellman backups for the critic and carry
ZERO information for the actor, so they are emitted, flagged, and reported --
if they dominate, the dataset is mostly non-decisions and that must be visible.

DIAGNOSTIC FIELDS ARE NEVER TRAINING TARGETS. V*, advantage and oracle_action are
computed from `delta` and exist only for telemetry and model selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .env import FringeEnv, default_expansion_cap
from .policies import BEHAVIOUR_POLICIES, make_policy
from .tree import TreeInstance

COUNTERFACTUAL_MODES = ("all", "none")


@dataclass
class Transition:
    """One (s, a, r, s') row. `obs`/`obs_next` are beam state-id lists; tensor
    packing happens later (offline/encoder.pack_fringe)."""

    instance: str
    policy: str
    seed: int
    fringe_size: int
    t: int                        # expansion index of the source state
    obs: List[int]
    action: int
    reward: float
    obs_next: List[int]
    terminated: bool              # SUCC / genuine DOOM -> do NOT bootstrap
    truncated: bool               # cap -> DO bootstrap
    outcome: str
    # --- diagnostics only: never features, never targets ---
    v_star_s: float
    v_star_s_next: float
    advantage: float              # -1 + V*(s') - V*(s)
    n_actions: int
    forced: bool
    oracle_action: int
    on_trajectory: bool           # True iff this is the action the behaviour took

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def _emit(env: FringeEnv, ranking: Sequence[int], action: int, t: int,
          policy: str, seed: int, on_traj: bool) -> Transition:
    """Model-step a CLONE so the parent env is untouched."""
    inst = env.instance
    v_s = env.v_star()
    n_actions = env.n_actions()
    forced = env.forced
    oracle_a = inst.oracle_action(env.fringe)

    c = env.clone()
    res = c.step(action, ranking=list(ranking))
    v_sn = c.v_star() if not res.terminated else 0.0

    return Transition(
        instance=inst.name,
        policy=policy,
        seed=seed,
        fringe_size=env.fringe_size,
        t=t,
        obs=list(env.fringe),
        action=int(action),
        reward=float(res.reward),
        obs_next=list(res.fringe),
        terminated=bool(res.terminated),
        truncated=bool(res.truncated),
        outcome=str(res.info["outcome"]),
        v_star_s=v_s,
        v_star_s_next=v_sn,
        advantage=(-1.0 + v_sn - v_s) if not res.terminated else 0.0,
        n_actions=n_actions,
        forced=forced,
        oracle_action=int(oracle_a),
        on_trajectory=bool(on_traj),
    )


def generate_episode(
    instance: TreeInstance,
    fringe_size: int,
    policy_name: str,
    seed: int,
    expansion_cap: int,
    counterfactual: str = "all",
    n_refill_samples: int = 1,
) -> List[Transition]:
    """One behaviour rollout with every action enumerated at each state.

    `n_refill_samples`: refill is uniformly random (RefillMode::RANDOM), so the
    successor of (s, a) is a random variable. Sample it n times and emit each
    draw as its OWN row -- averaging them would fabricate a successor state that
    the planner can never be in.
    """
    if counterfactual not in COUNTERFACTUAL_MODES:
        raise ValueError(f"counterfactual must be one of {COUNTERFACTUAL_MODES}")

    env = FringeEnv(instance, fringe_size=fringe_size, seed=seed, gamma=1.0,
                    expansion_cap=expansion_cap)
    pol = make_policy(instance, policy_name, seed=seed)
    rows: List[Transition] = []
    res = env.reset(seed=seed)
    t = 0

    while not res.done:
        if env.forced:
            # The planner does not call the model here; there is exactly one
            # action. Emit it once (a valid backup) and move on.
            ranking = [env.forced_action]
            taken = env.forced_action
            actions: List[int] = [taken]
        else:
            ranking = list(pol(env.fringe))
            taken = ranking[0]
            actions = (
                list(range(len(env.fringe))) if counterfactual == "all" else [taken]
            )

        for a in actions:
            for _ in range(max(1, int(n_refill_samples))):
                rows.append(_emit(env, ranking, a, t, policy_name, seed,
                                  on_traj=(a == taken)))

        res = env.step(taken, None if env.forced else ranking)
        t += 1

    return rows


def generate_dataset(
    instances: Sequence[TreeInstance],
    fringe_size: int,
    policies: Iterable[str] = BEHAVIOUR_POLICIES,
    seeds_per_policy: int = 10,
    expansion_cap: Optional[int] = None,
    counterfactual: str = "all",
    n_refill_samples: int = 1,
    verbose: bool = True,
) -> tuple[List[Transition], Dict[str, object]]:
    """Full offline dataset + the composition summary that must be reported."""
    policies = list(policies)
    for p in policies:
        if p not in BEHAVIOUR_POLICIES:
            raise ValueError(f"unknown behaviour policy {p!r}")
    cap = int(expansion_cap) if expansion_cap is not None else default_expansion_cap(instances)

    rows: List[Transition] = []
    for inst in instances:
        for pol in policies:
            for s in range(int(seeds_per_policy)):
                rows.extend(generate_episode(
                    inst, fringe_size, pol, s, cap, counterfactual, n_refill_samples,
                ))
    summary = dataset_summary(rows, fringe_size, cap)
    if verbose:
        print(
            f"[dataset] F={fringe_size} n_rows={summary['n_rows']} "
            f"states={summary['n_states']} "
            f"actions/state={summary['actions_per_state']:.2f} "
            f"forced_frac={summary['forced_frac']:.3f} "
            f"truncated_frac={summary['truncated_frac']:.4f}"
        )
    return rows, summary


def dataset_summary(rows: Sequence[Transition], fringe_size: int,
                    expansion_cap: int) -> Dict[str, object]:
    """Composition figures the run summary must surface.

    `forced_frac` is the one to watch: forced rows are non-decisions, so if they
    dominate, the dataset is mostly states where the policy had no choice and the
    actor has almost nothing to learn from. That must be visible, not buried.
    """
    n = len(rows)
    states = {(r.instance, r.policy, r.seed, r.t) for r in rows}
    n_states = max(1, len(states))
    forced = sum(1 for r in rows if r.forced)
    on_traj = sum(1 for r in rows if r.on_trajectory)
    advs = [r.advantage for r in rows if not r.forced and not r.terminated]
    zero_adv = sum(1 for a in advs if abs(a) < 1e-9)
    return {
        "n_rows": n,
        "n_states": len(states),
        "fringe_size": fringe_size,
        "expansion_cap": expansion_cap,
        "actions_per_state": n / n_states,
        "forced_frac": forced / max(1, n),
        "on_trajectory_frac": on_traj / max(1, n),
        "terminated_frac": sum(1 for r in rows if r.terminated) / max(1, n),
        "truncated_frac": sum(1 for r in rows if r.truncated) / max(1, n),
        "success_frac": sum(1 for r in rows if r.outcome == "success") / max(1, n),
        "doom_frac": sum(1 for r in rows if r.outcome == "doom") / max(1, n),
        "zero_advantage_frac": (zero_adv / len(advs)) if advs else 0.0,
        "adv_hist": _hist(advs),
        "n_instances": len({r.instance for r in rows}),
        "policies": sorted({r.policy for r in rows}),
    }


def _hist(values: Sequence[float], bins: int = 10) -> Dict[str, int]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {f"{lo:.2f}": len(values)}
    out: Dict[str, int] = {}
    for v in values:
        k = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        key = f"[{lo + k * (hi - lo) / bins:.2f},{lo + (k + 1) * (hi - lo) / bins:.2f})"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
