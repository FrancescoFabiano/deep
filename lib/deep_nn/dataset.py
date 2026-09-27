"""Offline dataset generation by counterfactual action expansion.

The tree is fixed and fully in memory, so the transition is a MODEL. |A(s)| <= F
<= 64, so every action can be ENUMERATED rather than sampled:

    for tree in trees:                       # one tree per (instance, pi_b)
      for pi in behaviour policies:          # default: `trace` = replay pi_b's own order
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

FILLING THE NON-FULL BEAMS (`fill_fringes`, 2026-09-10)
--------------------------------------------------------
The beam is FULL only while the open set B u R holds >= F nodes. A behaviour trace
on the CC unified graphs reaches that late or never: at F=32, 26% of the decision
states have a full beam (all of them from the BFS trace on the two larger
problems), at F=16 55%, at F=8 90%. The model therefore trains mostly on beams
narrower than the window it is deployed with at F >= 16.

Padding a beam with states that are not open (closed, random) was tried and
retired (--pad-closed): it hands the model actions the planner can never take.
The LEGAL way to widen a beam is to EXPAND MORE before recording decisions -- a
frontier grows only by expanding, and every expansion is a real search state.
So with `fill_fringes` every non-full decision state of a behaviour rollout
becomes a BRANCH POINT, and from each one `fill_k` extra episodes are rolled:

    1. GROW (no rows): from the branch point's exact (B, R, visited) state, expand
       uniformly random beam slots until the beam is full. A goal generated during
       growth ends the search as it would in the planner, and the attempt is
       DISCARDED (nothing was recorded); so is an attempt whose open set cannot
       reach F within the growth budget.
    2. ROLL (rows): from the grown state, continue with the SAME behaviour policy
       (the trace), emitting every counterfactual action exactly like the parent
       episode. Every beam on this stretch is full while the open set stays >= F.

The parent trajectory is untouched (its small beams are real: every search starts
that way, at deployment too). Branch rows are flagged `filled=True` and carry a
distinct `seed` (FILL_SEED_STRIDE * branch + parent seed) so trajectory-keyed
consumers ((instance, policy, seed)) see them as their own rollouts. Labels need
nothing new: delta is known for every state of the graph, so V*, oracle_action
and advantage are exact on the grown beams -- on a UNIFIED graph (unify.py),
which is where this is meant to run: every strategy's expansions are available
there, so a random growth expansion has real children wherever any search went.
On a per-strategy tree a growth expansion of a censored leaf is a fabricated
dead end, exactly the artifact drop_censored exists for.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .env import FringeEnv, default_expansion_cap
from .policies import (
    ALL_POLICIES,
    BEHAVIOUR_POLICIES,
    expand_policies,
    is_policy_name,
    make_policy,
)
from .tree import TreeInstance

COUNTERFACTUAL_MODES = ("all", "none")

# Branch rows get seed = FILL_SEED_STRIDE * (branch_index + 1) + parent_seed, so a
# branch is its own trajectory for every (instance, policy, seed)-keyed consumer
# and never collides with a parent seed (seeds_per_policy << the stride).
FILL_SEED_STRIDE = 1_000_000
# Growth budget per branch attempt, in expansions per unit of F: from the root's
# 4 children with branching ~3 the open set passes F=32 in ~14 expansions; the
# budget is generous so only a graph whose open set CANNOT reach F trips it.
FILL_GROW_BUDGET_PER_F = 8
DEFAULT_FILL_K = 4


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
    # True iff the action expands a CENSORED depth-bound leaf: the tree records
    # a dead end there but the real planner would expand it normally, so the
    # transition's outcome is unknowable from the data. Dropped from training
    # by generate_dataset (drop_censored=True) -- keeping it would teach
    # "avoid v" from a label that is a generation artifact (fix 1A).
    censored_expansion: bool = False
    # True iff the row comes from a fill BRANCH (module docstring): a rollout of the
    # same behaviour policy started from a non-full decision state of the parent
    # trajectory after random growth to a full beam. Diagnostics only.
    filled: bool = False

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def _emit(env: FringeEnv, ranking: Sequence[int], action: int, t: int,
          policy: str, seed: int, on_traj: bool, filled: bool = False) -> Transition:
    """Model-step a CLONE so the parent env is untouched."""
    inst = env.instance
    v_s = env.v_star()
    n_actions = env.n_actions()
    forced = env.forced
    oracle_a = inst.oracle_action(env.fringe)
    node = env.fringe[action]
    # Censored bound-hit LEAF: the tree records b_v=0 / delta=inf there purely
    # because generation stopped (depth bound), so the replayed "dead end"
    # transition is a fabrication. Internal censored nodes keep their rows --
    # their replay transitions (children exist) are real tree dynamics.
    censored_leaf = inst.censored[node] and not inst.children[node]

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
        censored_expansion=bool(censored_leaf),
        filled=bool(filled),
    )


def _roll(env: FringeEnv, pol, policy_name: str, row_seed: int,
          counterfactual: str, n_refill_samples: int, rows: List[Transition],
          filled: bool = False,
          branch_points: Optional[List[FringeEnv]] = None) -> None:
    """Follow `pol` from the env's CURRENT state to the end of the episode,
    emitting every enumerated action at every state into `rows`.

    `t` is the expansion index of the source state (reset = expansion 1 = t 0),
    so it stays unique along a branch that starts deep in the search.
    `branch_points`, when given, collects a clone of every NON-FULL decision
    state passed on the way (the fill branch points).
    """
    while not env.done:
        t = env.expansions - 1
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
            if branch_points is not None and len(env.fringe) < env.fringe_size:
                branch_points.append(env.clone())

        for a in actions:
            for _ in range(max(1, int(n_refill_samples))):
                rows.append(_emit(env, ranking, a, t, policy_name, row_seed,
                                  on_traj=(a == taken), filled=filled))

        env.step(taken, None if env.forced else ranking)


def grow_to_full(env: FringeEnv, rng: random.Random,
                 max_expansions: Optional[int] = None) -> str:
    """Expand uniformly random beam slots (NO rows) until the beam is full.

    Returns "full" on success, else why the attempt must be discarded:
      "goal"   a goal was generated (the planner would have stopped: nothing
               after it is a search state),
      "ended"  the episode hit the cap or doom before the beam filled,
      "budget" the growth budget ran out (the open set cannot reach F here).
    The env is left in whatever state the growth reached; a discarded attempt's
    env must not be rolled.
    """
    budget = (int(max_expansions) if max_expansions is not None
              else FILL_GROW_BUDGET_PER_F * env.fringe_size)
    steps = 0
    while len(env.fringe) < env.fringe_size:
        if env.done:
            return "ended"
        if steps >= budget:
            return "budget"
        if env.forced:
            res = env.step(env.forced_action, None)
        else:
            ranking = list(range(len(env.fringe)))
            rng.shuffle(ranking)
            res = env.step(ranking[0], ranking)
        steps += 1
        if res.terminated:
            return "goal" if res.info["outcome"] == "success" else "ended"
        if res.truncated:
            return "ended"
    return "full"


def generate_episode(
    instance: TreeInstance,
    fringe_size: int,
    policy_name: str,
    seed: int,
    expansion_cap: int,
    counterfactual: str = "all",
    n_refill_samples: int = 1,
    gamma: float = 1.0,
    fill_fringes: bool = False,
    fill_k: int = DEFAULT_FILL_K,
    fill_stats: Optional[Dict[str, int]] = None,
) -> List[Transition]:
    """One behaviour rollout with every action enumerated at each state.

    `n_refill_samples`: refill is uniformly random (RefillMode::RANDOM), so the
    successor of (s, a) is a random variable. Sample it n times and emit each
    draw as its OWN row -- averaging them would fabricate a successor state that
    the planner can never be in.

    `gamma`: the discount factor for the data-generating env. THE TRAINING
    PIPELINE PASSES THE ARGPARSER'S --gamma (run.py forwards RunConfig.gamma),
    so the data-generating env and the trainer share one objective -- it was a
    hardcoded 1.0 before, silently disagreeing with the trainer's 0.9999.
    The DEFAULT stays 1.0 for standalone structural probes (e.g. usability.
    scorability), whose caps can exceed the gamma<1 dominance horizon and which
    never train anything: rewards are -1/0 either way, gamma only sets the
    (unreachable) doom penalty and arms the dominance guard.

    `fill_fringes` / `fill_k`: the fill branches of the module docstring --
    `fill_k` extra rollouts per NON-FULL decision state of this episode, each
    grown to a full beam first (no rows) and then rolled under the same
    behaviour policy (rows, `filled=True`). False = the parent rollout only,
    byte-identical to before. `fill_stats` (optional dict) accumulates the
    branch-point / attempt / discard counts across calls.
    """
    if counterfactual not in COUNTERFACTUAL_MODES:
        raise ValueError(f"counterfactual must be one of {COUNTERFACTUAL_MODES}")
    if fill_fringes and int(fill_k) < 1:
        raise ValueError(f"fill_k must be >= 1 when fill_fringes is on, got {fill_k}")

    env = FringeEnv(instance, fringe_size=fringe_size, seed=seed, gamma=gamma,
                    expansion_cap=expansion_cap)
    pol = make_policy(instance, policy_name, seed=seed)
    rows: List[Transition] = []
    env.reset(seed=seed)
    branch_points: Optional[List[FringeEnv]] = [] if fill_fringes else None
    _roll(env, pol, policy_name, seed, counterfactual, n_refill_samples, rows,
          filled=False, branch_points=branch_points)

    if not fill_fringes:
        return rows

    stats = fill_stats if fill_stats is not None else {}
    for k in ("branch_points", "attempts", "kept", "discarded_goal",
              "discarded_ended", "discarded_budget"):
        stats.setdefault(k, 0)
    stats["branch_points"] += len(branch_points or [])
    n_branch = 0
    for bp_index, snap in enumerate(branch_points or []):
        for j in range(int(fill_k)):
            stats["attempts"] += 1
            # One RNG per attempt for BOTH the growth choices and the env's refill
            # draws: the K attempts from one branch point must diverge, and a
            # clone carries the parent's RNG state.
            attempt_seed = (seed * 7919 + bp_index) * 4096 + j
            b_env = snap.clone()
            b_env.rng.seed(attempt_seed)
            why = grow_to_full(b_env, random.Random(attempt_seed))
            if why != "full":
                stats[f"discarded_{why}"] += 1
                continue
            stats["kept"] += 1
            n_branch += 1
            row_seed = FILL_SEED_STRIDE * n_branch + seed
            b_pol = make_policy(instance, policy_name, seed=attempt_seed)
            _roll(b_env, b_pol, policy_name, row_seed, counterfactual,
                  n_refill_samples, rows, filled=True, branch_points=None)
    return rows


def generate_dataset(
    instances: Sequence[TreeInstance],
    fringe_size: int,
    policies: Iterable[str] = BEHAVIOUR_POLICIES,
    seeds_per_policy: int = 10,
    expansion_cap: Optional[int] = None,
    counterfactual: str = "all",
    n_refill_samples: int = 1,
    gamma: float = 1.0,
    drop_censored: bool = True,
    verbose: bool = True,
    fill_fringes: bool = False,
    fill_k: int = DEFAULT_FILL_K,
) -> tuple[List[Transition], Dict[str, object]]:
    """Full offline dataset + the composition summary that must be reported.

    `drop_censored`: rows whose action expands a censored bound-hit leaf are
    DROPPED from the returned pool (the count is in the summary) -- their
    recorded "dead end" outcome is a generation artifact, and training on it
    teaches the policy to avoid states that may sit on the shortest path at
    deployment. The behaviour rollout still traverses them (the env dynamics
    are unchanged); only the training LABEL is withheld. False keeps the old
    behaviour for ablation.

    `fill_fringes` / `fill_k`: see generate_episode and the module docstring.
    Off by default; the summary then carries zero fill counts.
    """
    policies = list(policies)
    for p in policies:
        if not is_policy_name(p):
            raise ValueError(f"unknown behaviour policy {p!r}; expected one of "
                             f"{ALL_POLICIES} or trace:<strategy>")
    cap = int(expansion_cap) if expansion_cap is not None else default_expansion_cap(instances)

    rows: List[Transition] = []
    fill_stats: Dict[str, int] = {}
    for inst in instances:
        # On a unified graph `trace` means EVERY behaviour policy that built it:
        # one rollout family per `trace:<s>` (see policies.expand_policies).
        for pol in expand_policies(inst, policies):
            for s in range(int(seeds_per_policy)):
                rows.extend(generate_episode(
                    inst, fringe_size, pol, s, cap, counterfactual,
                    n_refill_samples, gamma,
                    fill_fringes=fill_fringes, fill_k=fill_k,
                    fill_stats=fill_stats,
                ))
    n_censored_rows = sum(1 for r in rows if r.censored_expansion)
    if drop_censored and n_censored_rows:
        rows = [r for r in rows if not r.censored_expansion]
    summary = dataset_summary(rows, fringe_size, cap)
    summary["n_censored_rows"] = n_censored_rows
    summary["censored_rows_dropped"] = bool(drop_censored)
    summary["fill_fringes"] = bool(fill_fringes)
    summary["fill_k"] = int(fill_k) if fill_fringes else 0
    summary["fill"] = {k: int(fill_stats.get(k, 0)) for k in (
        "branch_points", "attempts", "kept", "discarded_goal", "discarded_ended",
        "discarded_budget")}
    if verbose:
        print(
            f"[dataset] F={fringe_size} n_rows={summary['n_rows']} "
            f"states={summary['n_states']} "
            f"actions/state={summary['actions_per_state']:.2f} "
            f"forced_states={summary['forced_state_frac']:.3f} "
            f"full_beam_states={summary['full_beam_state_frac']:.3f} "
            f"truncated_frac={summary['truncated_frac']:.4f} "
            f"censored_rows={'dropped ' if drop_censored else ''}{n_censored_rows} "
            f"trajectories={summary['n_trajectories']} "
            f"(distinct {summary['n_distinct_trajectories']})"
        )
        if fill_fringes:
            fs = summary["fill"]
            print(
                f"[dataset] fill (k={fill_k}): branch_points={fs['branch_points']} "
                f"attempts={fs['attempts']} kept={fs['kept']} "
                f"discarded(goal={fs['discarded_goal']} ended={fs['discarded_ended']} "
                f"budget={fs['discarded_budget']}) -> fill_rows={summary['n_fill_rows']} "
                f"({100 * summary['fill_row_frac']:.1f}% of rows), "
                f"fill decision states={summary['n_fill_decision_states']} "
                f"(full {100 * summary['fill_full_beam_state_frac']:.1f}%); "
                f"parent decision states full "
                f"{100 * summary['parent_full_beam_state_frac']:.1f}%"
            )
        if summary["duplicate_trajectory_frac"] > 0.5:
            print(f"[dataset] WARNING {100 * summary['duplicate_trajectory_frac']:.0f}% of "
                  f"the rollouts duplicate another rollout of the same (tree, policy): a "
                  f"deterministic behaviour on a tree whose open set never exceeds F={fringe_size} "
                  f"yields the same trajectory at every seed. More seeds add rows, not "
                  f"information, and a held-out trajectory may be a copy of a trained one.")
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
    # The ROW fraction badly understates how many STATES are non-decisions: a
    # forced state emits 1 row while a decision state emits |B| ~ 5. Measured on
    # CC at F=8: 3.4% of rows are forced but 17% of states are -- and 17% is the
    # number that matters, because it is the share of visited states where the
    # policy had no choice. It also tracks the 17.5% dead-end rate, as it should.
    forced_states = {
        (r.instance, r.policy, r.seed, r.t) for r in rows if r.forced
    }
    on_traj = sum(1 for r in rows if r.on_trajectory)
    advs = [r.advantage for r in rows if not r.forced and not r.terminated]
    zero_adv = sum(1 for a in advs if abs(a) < 1e-9)
    # Trajectory identity = the followed sequence of (beam AS A SET, expanded state).
    # Slot ORDER is refill noise (a node parked in R comes back at a random slot) and
    # must not make two identical searches look different. Two seeds of a
    # deterministic behaviour coincide unless refill (open set > F) or a tie
    # separated them; with `trace` this is the normal case on small trees.
    traj: Dict[tuple, List[tuple]] = {}
    for r in rows:
        if r.on_trajectory:
            traj.setdefault((r.instance, r.policy, r.seed), []).append(
                (r.t, tuple(sorted(r.obs)), r.obs[r.action]))
    distinct: Dict[tuple, set] = {}
    for (inst, pol, _seed), steps in traj.items():
        distinct.setdefault((inst, pol), set()).add(tuple(sorted(steps)))
    n_traj = len(traj)
    n_distinct = sum(len(s) for s in distinct.values())
    # Beam width at the DECISION states (forced states excluded: the planner does
    # not score there). "Full" = exactly F slots -- the window the model is
    # deployed with. Reported for parent and fill rows separately, so the effect
    # of fill_fringes is a number in the log, not an inference.
    decision: Dict[tuple, tuple] = {}
    for r in rows:
        if not r.forced:
            decision[(r.instance, r.policy, r.seed, r.t)] = (len(r.obs), r.filled)
    n_dec = len(decision)
    n_full = sum(1 for w, _ in decision.values() if w >= fringe_size)
    n_dec_fill = sum(1 for _, f in decision.values() if f)
    n_full_fill = sum(1 for w, f in decision.values() if f and w >= fringe_size)
    n_dec_parent = n_dec - n_dec_fill
    n_full_parent = n_full - n_full_fill
    n_fill_rows = sum(1 for r in rows if r.filled)
    return {
        "n_decision_states": n_dec,
        "full_beam_state_frac": (n_full / n_dec) if n_dec else 0.0,
        "n_parent_decision_states": n_dec_parent,
        "parent_full_beam_state_frac": (n_full_parent / n_dec_parent) if n_dec_parent else 0.0,
        "n_fill_decision_states": n_dec_fill,
        "fill_full_beam_state_frac": (n_full_fill / n_dec_fill) if n_dec_fill else 0.0,
        "n_fill_rows": n_fill_rows,
        "fill_row_frac": n_fill_rows / max(1, n),
        "n_trajectories": n_traj,
        "n_distinct_trajectories": n_distinct,
        "duplicate_trajectory_frac": (1.0 - n_distinct / n_traj) if n_traj else 0.0,
        "n_rows": n,
        "n_states": len(states),
        "fringe_size": fringe_size,
        "expansion_cap": expansion_cap,
        "actions_per_state": n / n_states,
        # forced_state_frac is the one to report: the share of visited STATES
        # where the policy had no choice. forced_row_frac is kept only so the two
        # cannot be confused.
        "forced_state_frac": len(forced_states) / n_states,
        "forced_row_frac": forced / max(1, n),
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
