"""The offline fringe MDP — a faithful model of RL_BestFirst + SpaceSearcher.

WHY THERE IS A RESERVOIR
------------------------
An RL policy has no completeness property, so it cannot be deployed as a search
strategy on its own. `RL_BestFirst` wraps it in a structure that never discards a
node: everything the beam does not hold is parked in a reservoir and can come
back. The policy *ranks a window*; the reservoir guarantees nothing is lost.
`F` (--RL_fringe_size) is therefore a SCORING WINDOW, not a beam width.

    s = (B, R)    B = beam, |B| <= F   (the candidates the net scores)
                  R = reservoir        (every other open, unexpanded node)

The agent observes only B — that is what the ONNX takes. R is environment state.
The observation is a strict subset of the state, so **the learned policy operates
under partial observability**. The environment must still carry R or the
transition is undefined and the critic fits dynamics that do not exist.

ONE EXPANSION PER ONNX CALL — AND WHY IT IS NOT FREE
----------------------------------------------------
The driver batches successors and only rescores when enough have accumulated:

    SpaceSearcher.tpp:179   if (fringe_RL.size() >= RL_node_to_add || empty())
                                push_vector(fringe_RL);

With the C++ defaults RL_node_to_add is 22, and the planner expands ~22/b nodes
per ONNX call using STALE ranks in between. This env models one expansion per
call, which is only faithful when RL_node_to_add == 1 — see planner_config.
Training and deployment must both be launched with the matching
--RL_exploitation, and export refuses otherwise.

THE DEAD-END PATH IS NOT A ROUNDING ERROR
-----------------------------------------
Even at RL_node_to_add == 1, an expansion that generates NO children leaves
fringe_RL empty, so `0 >= 1` is false and push_vector never fires. The planner
does not rebuild, does not refill and does not rescore — it pops the next node by
the STALE ranks it already has. Dead ends are ~18% of nodes on CC_2_3_4__pl_7, so
this path is hot, and an env that rescored here would silently diverge.

    b_v >= 1:  B' = children[:F] + refill(R'),  R' = R u (B\\{v}) u children[F:]
               -> rescore point, |A(s')| = |B'|
    b_v == 0:  B' = B \\ {v},  R' = R,  no refill, no rescore
               -> FORCED: the action is the stale argmax over B', |A(s')| = 1

`forced` states are valid Bellman backups for the critic and carry zero
information for the actor.

REFILL IS RANDOM, AND THAT IS FORCED ON US
------------------------------------------
Only --RL_heuristics RNG (-> RefillMode::RANDOM) is reproducible offline. The
HEURISTIC modes order the reservoir by Heuristics::RL_H, which folds the
planner's stored past RL ranks — not reconstructible from a generation table.
Consequence: the transition is stochastic and V* is an OPTIMISTIC BOUND.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .planner_config import assert_one_expansion_per_call, exploitation_for
from .tree import INF_DELTA, TreeInstance

REWARD_MODES = ("absorbing", "legacy")


# ---------------------------------------------------------------- reward ----

def doom_penalty(gamma: float, reward_mode: str = "absorbing") -> float:
    """Terminal reward on DOOM.

    absorbing: -1/(1-gamma). This is NOT a free hyperparameter — it is the
    delta->inf limit of V* under gamma^inf = 0, i.e. the value of never
    reaching a goal. It makes the return of a failure independent of when it
    happens (see G_doom below).

    legacy: -1, the known-broken reward kept only for the F6 ablation.
    """
    if reward_mode == "legacy":
        return -1.0
    if reward_mode != "absorbing":
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    return -1.0 / (1.0 - gamma)


def g_succ(k: int, gamma: float) -> float:
    """Return of a success after k expansions: -(1 - gamma^k)/(1-gamma)."""
    return -(1.0 - gamma ** k) / (1.0 - gamma)


def g_doom(gamma: float, reward_mode: str = "absorbing") -> float:
    """Return of a failure that dooms after m expansions.

    Under `absorbing` this is exactly -1/(1-gamma) for EVERY m:

        G = -(1-gamma^m)/(1-gamma) + gamma^m * (-1/(1-gamma))
          = -[1 - gamma^m + gamma^m]/(1-gamma)
          = -1/(1-gamma)

    which is the whole point: there is no incentive to fail early. Under
    `legacy` the return is -m, so failing sooner scores better and the optimal
    policy is to fail as fast as possible.
    """
    if reward_mode == "legacy":
        raise ValueError(
            "g_doom is m-dependent under legacy (that is the bug): G = -m."
        )
    return -1.0 / (1.0 - gamma)


def assert_gamma_separates(gamma: float, k_max: int) -> None:
    """Require 1/(1-gamma) > K_max, with K_max measured from the data.

    This is NOT about ordering — G_succ(k) > G_doom holds for any gamma in
    (0,1) and any finite k. It is about NUMERICAL SEPARATION: G_succ(k) ->
    G_doom as k grows, so once the discount saturates over the horizon the data
    actually contains, successes become indistinguishable from failures in
    float. Requiring the effective horizon 1/(1-gamma) to exceed the longest
    observed success keeps the gap meaningful.

    K_max is max delta(root) over the solvable instances — the longest observed
    success. It is deliberately NOT the worst-case internal-node count, which
    would be a useless bound forcing gamma ~ 0.998.
    """
    horizon = 1.0 / (1.0 - gamma)
    if not horizon > k_max:
        raise ValueError(
            f"gamma={gamma} is too small for this data.\n"
            f"  effective horizon 1/(1-gamma) = {horizon:.2f}\n"
            f"  K_max (longest observed success, max delta(root)) = {k_max}\n"
            f"  Require 1/(1-gamma) > K_max so the discount has not saturated\n"
            f"  over the horizon the data contains; otherwise G_succ(K_max) is\n"
            f"  numerically indistinguishable from G_doom.\n"
            f"  Fix: gamma > {1.0 - 1.0 / (k_max + 1):.5f}."
        )


# ------------------------------------------------------------ step result ----

@dataclass
class StepResult:
    fringe: List[int]           # the beam, in ONNX packing order; action indexes this
    reward: float
    done: bool
    info: Dict[str, object] = field(default_factory=dict)


# -------------------------------------------------------------------- env ----

class FringeEnv:
    """One episode = one instance's tree. Observations are state-id lists;
    tensor packing is the caller's job (offline/encoder.py).

    Contract with the caller:
      - at a RESCORE state (`forced == False`) the caller must pass `ranking`,
        the full priority order over the beam (slot indices, best first). The
        planner keeps that order in its priority queue and reuses it if the
        expansion turns out to be a dead end, so the env cannot infer it from
        `action` alone.
      - at a FORCED state (`forced == True`) the action is predetermined
        (`forced_action`) and `ranking` is ignored: the planner does not call
        the model here.
    """

    def __init__(
        self,
        instance: TreeInstance,
        fringe_size: int = 32,
        seed: int = 0,
        gamma: float = 0.99,
        reward_mode: str = "absorbing",
        expansion_cap: Optional[int] = None,
        exploitation: Optional[int] = None,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.instance = instance
        self.fringe_size = int(fringe_size)
        self.gamma = float(gamma)
        self.reward_mode = reward_mode
        self.rng = random.Random(seed)
        self.seed = seed

        # The env models one expansion per ONNX call; that only matches the
        # planner when int(F * exploitation / 100) == 1. Check it here so a
        # mis-parameterised env cannot silently produce numbers.
        self.exploitation = (
            int(exploitation) if exploitation is not None else exploitation_for(self.fringe_size)
        )
        assert_one_expansion_per_call(self.fringe_size, self.exploitation)

        self.expansion_cap = (
            int(expansion_cap) if expansion_cap is not None else 4 * instance.n_states
        )
        self.doom_reward = doom_penalty(self.gamma, self.reward_mode)

        self.fringe: List[int] = []
        self.order: List[int] = []      # state ids, best first (the priority queue)
        self.reservoir: List[int] = []
        self.visited: set[int] = set()
        self.expansions = 0
        self.forced = False
        self.forced_action: Optional[int] = None
        self.done = True
        # Diagnostics
        self.n_forced_steps = 0
        self.n_unscored_pulls = 0

    # ---- observation helpers ----

    def available_actions(self) -> List[int]:
        """Slot indices the planner could actually take from this state."""
        if self.done:
            return []
        if self.forced:
            return [self.forced_action]  # type: ignore[list-item]
        return list(range(len(self.fringe)))

    def n_actions(self) -> int:
        return len(self.available_actions())

    def v_star(self) -> float:
        """Optimistic bound (reservoir included, random refill not controllable)."""
        return self.instance.v_star(self.fringe, self.reservoir)

    # ---- core mechanics ----

    def _generate_children(self, v: int) -> tuple[List[int], bool]:
        """Mirror SpaceSearcher.tpp:150-178.

        The goal test precedes the visited check (line 162 returns before line
        167 inserts), so a goal is never recorded as visited and never enters
        the beam: reaching it ends the search at *generation*.
        """
        fresh: List[int] = []
        for c in self.instance.children[v]:
            if self.instance.is_goal[c]:
                return fresh, True
            if c in self.visited:
                continue
            self.visited.add(c)
            fresh.append(c)
        return fresh, False

    def _rebuild_beam(self, new_states: List[int]) -> None:
        """Mirror RL_BestFirst::push_vector (RL_BestFirst.h:51-89).

        Order matters and matches the C++ exactly:
          1. drain the remaining beam into the reservoir  (lines 56-61)
          2. new states fill the batch first, overflow -> reservoir (66-73)
          3. refill the free slots from the reservoir     (76)
        Step 1 precedes step 3, so a just-demoted beam member is eligible to be
        drawn straight back — that is the planner's behaviour, not an accident.
        """
        self.reservoir.extend(self.fringe)
        self.fringe = []
        batch = list(new_states[: self.fringe_size])
        self.reservoir.extend(new_states[self.fringe_size:])
        self._refill_random(batch)
        self.fringe = batch
        self.order = []  # awaiting a fresh ranking

    def _refill_random(self, batch: List[int]) -> None:
        """RL_BestFirst::refill_beam_random — uniform draw without replacement."""
        while len(batch) < self.fringe_size and self.reservoir:
            i = self.rng.randrange(len(self.reservoir))
            self.reservoir[i], self.reservoir[-1] = self.reservoir[-1], self.reservoir[i]
            batch.append(self.reservoir.pop())

    def _pull_unscored(self) -> None:
        """Mirror RL_BestFirst::peek() (lines 127-142).

        Reachable: consecutive dead-end pops can drain the beam while the
        reservoir is still full. push_vector never fired (no children), and
        `empty()` is false (reservoir non-empty), so the driver loops, peek()
        finds search_space empty and pushes ONE random reservoir node straight
        into the queue — never scored by the model. The next expansion is that
        node, with no choice involved.
        """
        i = self.rng.randrange(len(self.reservoir))
        self.reservoir[i], self.reservoir[-1] = self.reservoir[-1], self.reservoir[i]
        v = self.reservoir.pop()
        self.fringe = [v]
        self.order = [v]
        self.n_unscored_pulls += 1

    def _set_forced(self) -> None:
        self.forced = True
        self.forced_action = self.fringe.index(self.order[0])

    def _terminal(self, reward: float, kind: str) -> StepResult:
        self.done = True
        return StepResult([], reward, True, self._info(outcome=kind))

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Expand the root (forced, counts as expansion 1).

        The initial state is never goal-tested: the C++ only tests successors
        (SpaceSearcher.tpp:162), so a goal root would not be detected there
        either.
        """
        if seed is not None:
            self.rng.seed(seed)
        self.fringe = []
        self.order = []
        self.reservoir = []
        self.visited = {self.instance.root_id}
        self.expansions = 1
        self.forced = False
        self.forced_action = None
        self.n_forced_steps = 0
        self.n_unscored_pulls = 0
        self.done = False

        fresh, goal = self._generate_children(self.instance.root_id)
        if goal:
            return self._terminal(0.0, "success")
        self._rebuild_beam(fresh)
        if not self.fringe:
            return self._terminal(self.doom_reward, "doom")
        return StepResult(list(self.fringe), 0.0, False, self._info(outcome="running"))

    def step(
        self,
        action: int,
        ranking: Optional[Sequence[int]] = None,
    ) -> StepResult:
        """Expand slot `action`.

        `ranking`: slot indices over the CURRENT beam in priority order (best
        first) — required at rescore states. It need NOT start with `action`:
        counterfactual dataset rows expand a non-argmax slot while keeping the
        model's ranking, which is exactly what determines the stale order if
        that expansion dead-ends.
        """
        if self.done:
            raise RuntimeError("step() on a finished episode; call reset().")
        if not (0 <= action < len(self.fringe)):
            raise IndexError(f"action {action} out of beam range {len(self.fringe)}")

        if self.forced:
            if action != self.forced_action:
                raise ValueError(
                    f"state is forced (stale-rank continuation after a dead end): "
                    f"the planner has no choice here and would expand slot "
                    f"{self.forced_action}, got {action}."
                )
            self.n_forced_steps += 1
        else:
            if ranking is None:
                raise ValueError(
                    "ranking is required at a rescore state: the planner keeps the "
                    "model's full priority order and reuses it if this expansion "
                    "dead-ends, so the env cannot infer it from `action` alone."
                )
            if sorted(ranking) != list(range(len(self.fringe))):
                raise ValueError(
                    f"ranking must be a permutation of the {len(self.fringe)} beam "
                    f"slots, got {list(ranking)!r}"
                )
            self.order = [self.fringe[i] for i in ranking]

        v = self.fringe.pop(action)
        self.order.remove(v)
        self.expansions += 1

        fresh, goal = self._generate_children(v)
        if goal:
            return self._terminal(0.0, "success")

        if fresh:
            # RESCORE PATH: push_vector fires (RL_node_to_add == 1).
            self._rebuild_beam(fresh)
            self.forced = False
            self.forced_action = None
        else:
            # STALE PATH: b_v == 0, so fringe_RL stays empty, `0 >= 1` is false
            # and push_vector never fires. No rebuild, no refill, no rescore —
            # the planner pops the next node by the ranks it already holds.
            if not self.fringe:
                if not self.reservoir:
                    return self._terminal(self.doom_reward, "doom")
                self._pull_unscored()
            self._set_forced()

        if not self.fringe:
            return self._terminal(self.doom_reward, "doom")
        if self.expansions >= self.expansion_cap:
            # Truncation, NOT doom: the search space is not exhausted, we simply
            # stopped looking. Bootstrapping (not the absorbing penalty) is the
            # correct backup here, so it is reported separately.
            self.done = True
            return StepResult([], -1.0, True, self._info(outcome="timeout"))
        return StepResult(list(self.fringe), -1.0, False, self._info(outcome="running"))

    def _info(self, outcome: str) -> Dict[str, object]:
        return {
            "outcome": outcome,               # success | doom | timeout | running
            "expansions": self.expansions,
            "goal_found": outcome == "success",
            "fringe_len": len(self.fringe),
            "reservoir_len": len(self.reservoir),
            "forced": self.forced and outcome == "running",
            "n_actions": self.n_actions(),
            "n_forced_steps": self.n_forced_steps,
            "n_unscored_pulls": self.n_unscored_pulls,
        }

    def clone(self) -> "FringeEnv":
        """A independent copy sharing the (immutable) instance.

        The tree is fixed and fully in memory, so the transition is a MODEL: the
        dataset generator enumerates every action from a state by cloning and
        stepping each clone. The RNG state is copied too, so every counterfactual
        branch faces the same refill draw and the successors differ only because
        the action differed.
        """
        c = FringeEnv.__new__(FringeEnv)
        c.instance = self.instance  # immutable, shared on purpose
        c.fringe_size = self.fringe_size
        c.gamma = self.gamma
        c.reward_mode = self.reward_mode
        c.seed = self.seed
        c.exploitation = self.exploitation
        c.expansion_cap = self.expansion_cap
        c.doom_reward = self.doom_reward
        c.rng = random.Random()
        c.rng.setstate(self.rng.getstate())
        c.fringe = list(self.fringe)
        c.order = list(self.order)
        c.reservoir = list(self.reservoir)
        c.visited = set(self.visited)
        c.expansions = self.expansions
        c.forced = self.forced
        c.forced_action = self.forced_action
        c.done = self.done
        c.n_forced_steps = self.n_forced_steps
        c.n_unscored_pulls = self.n_unscored_pulls
        return c

    # ---- invariant used by the reservoir test ----

    def live_node_count(self) -> int:
        """|B u R u expanded| — must equal |generated so far| at every step.

        The completeness guarantee in one number: with the reservoir, no node is
        ever dropped. `visited` is exactly the generated set (goals excepted,
        which end the episode), and the root is seeded into it at reset.
        """
        return len(set(self.fringe) | set(self.reservoir) | self._expanded())

    def _expanded(self) -> set[int]:
        return self.visited - set(self.fringe) - set(self.reservoir)


def rollout(
    env: FringeEnv,
    policy,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run one episode. `policy(fringe) -> ranking` (slot indices, best first).

    The action taken is ranking[0] — exactly what the planner does (it pops the
    priority-queue top, i.e. the argmax logit). At forced states the policy is
    not consulted: the planner does not call the model there either.
    """
    res = env.reset(seed=seed)
    total = 0.0
    disc = 0.0
    g = 1.0
    while not res.done:
        if env.forced:
            action, ranking = env.forced_action, None
        else:
            ranking = list(policy(res.fringe))
            action = ranking[0]
        res = env.step(action, ranking)
        total += res.reward
        disc += g * res.reward
        g *= env.gamma
    return {
        "expansions": res.info["expansions"],
        "outcome": res.info["outcome"],
        "goal_found": res.info["goal_found"],
        "return": total,
        "discounted_return": disc,
        "regret": (
            env.instance.regret(int(res.info["expansions"]))
            if res.info["outcome"] == "success" else None
        ),
        "n_forced_steps": res.info["n_forced_steps"],
        "n_unscored_pulls": res.info["n_unscored_pulls"],
    }
