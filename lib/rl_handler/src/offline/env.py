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
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .planner_config import assert_one_expansion_per_call, exploitation_for
from .tree import INF_DELTA, TreeInstance

# The rollout ceiling is a GENEROUS backstop, not a semantic threshold. Any
# single cap is an arbitrary line through the spread of policy costs (measured at
# F=32 on CC: hfs_oracle 34, dfs 81, random 199, bfs 231), and whichever we pick
# decides which baselines "fail". So the cap only bounds compute; the reported
# quantity is a COVERAGE CURVE over budget (a cactus plot), which makes the
# budget a parameter of the reader rather than of the experiment.
CAP_FLOOR = 2000
CAP_DELTA_MULTIPLIER = 50

# Selection budget: difficulty-relative, so an easy instance does not get the
# same absolute slack as a hard one. Declared once in the manifest, never tuned.
REFERENCE_BUDGET_MULTIPLIER = 10


def default_expansion_cap(instances: Iterable[TreeInstance]) -> int:
    """max(CAP_FLOOR, 50 * max delta(root)). Log it; never tune it."""
    deltas = [int(i.delta_root) for i in instances if i.solvable()]
    if not deltas:
        return CAP_FLOOR
    return max(CAP_FLOOR, CAP_DELTA_MULTIPLIER * max(deltas))


def reference_budget(instance: TreeInstance) -> int:
    """The declared budget coverage is measured at: 10 * delta(root)."""
    return int(REFERENCE_BUDGET_MULTIPLIER * instance.delta_root)

REWARD_MODES = ("absorbing", "legacy")

# THE OBJECTIVE IS UNDISCOUNTED. See assert_gamma().
DEFAULT_GAMMA = 0.9999


# ---------------------------------------------------------------- reward ----

def doom_penalty(expansion_cap: int, gamma: float = DEFAULT_GAMMA,
                 reward_mode: str = "absorbing") -> float:
    """Terminal reward on GENUINE DOOM (a real dead end -- no viable node remains).

    absorbing, gamma < 1: -1/(1-gamma) -- the paper's absorbing-failure return, the
    worst value the discounted objective admits. At gamma=0.9999 this is -10000.
    UNREACHABLE by construction on the gated pool: the completeness proposition makes
    DOOM equivalent to delta(root)=inf, and those instances are filtered at load.

    absorbing, gamma == 1: -expansion_cap. The -1/(1-gamma) form is -inf at gamma=1,
    so the undiscounted SSP ablation uses a finite stand-in for "the worst cost the
    budget admits". (gamma=1 stays a valid ablation axis; it is no longer the default.)

    This is for genuine doom ONLY. A TIMEOUT is truncation, not failure: it returns
    the ordinary step -1 and BOOTSTRAPS (env.step, `truncated=True`), so it must never
    receive this penalty -- doing so would declare a budget-exhausted state a
    catastrophic terminal and reintroduce the deep-search pessimism (Pardo 2018).

    legacy: -1, the known-broken reward kept only for the reward ablation.
    """
    if reward_mode == "legacy":
        return -1.0
    if reward_mode != "absorbing":
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    if gamma >= 1.0:
        return -float(expansion_cap)
    return -1.0 / (1.0 - gamma)


def g_succ(k: int, gamma: float = DEFAULT_GAMMA) -> float:
    """Return of a success after k expansions.

    gamma = 1:  -k                       linear, no saturation, ever
    gamma < 1:  -(1 - gamma^k)/(1-gamma)
    """
    if gamma == 1.0:
        return -float(k)
    return -(1.0 - gamma ** k) / (1.0 - gamma)


def assert_gamma(gamma: float, expansion_cap: Optional[int] = None) -> None:
    """Validate gamma, and (if a cap is given) the DOMINANCE invariant the paper's
    reward needs.

    THE OBJECTIVE, AND WHY BOTH gamma ARE DEFENSIBLE. Every policy reaches a goal on
    a solvable instance (see FringeEnv._doom), so EVERY POLICY IS PROPER: costs are
    strictly positive (-1/expansion), the process terminates w.p.1, there is no
    absorbing failure to escape into. That is the stochastic-shortest-path setting
    (Bertsekas & Tsitsiklis), where the undiscounted operator is well-posed and
    V*(s) = -delta(s) EXACTLY. gamma=1 is therefore correct, not a hack, and stays a
    valid ablation axis.

    gamma=0.9999 (the default) matches the PAPER'S discounted formulation while
    staying numerically ~= the SSP limit: for delta 3-24 against a horizon of 10000,
    V* = -(1-gamma^delta)/(1-gamma) equals -delta to <0.2%. It buys the paper's
    formula on the page and produces the same numbers -- a presentation/conformance
    choice, not a change of objective.

    THE DOMINANCE INVARIANT. The paper's guarantee (a success always beats doom)
    only holds if every reachable success return -(1-gamma^k)/(1-gamma) stays above
    G_doom = -1/(1-gamma). The worst reachable k is the expansion cap, so the
    invariant is:

        expansion_cap < 1/(1 - gamma)          (cap strictly below the horizon)

    Past the horizon the discounted return saturates and a slow search becomes
    indistinguishable from a hopeless one -- the exact collapse gamma<1 risks. At
    gamma=0.9999 the horizon is 10000 and the cap is 2000, so it passes with a wide
    margin. This is checked HERE so a future gamma/cap combo that breaks it fails
    loudly rather than silently flattening the objective.
    """
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if gamma < 1.0 and expansion_cap is not None:
        horizon = 1.0 / (1.0 - gamma)
        if expansion_cap >= horizon:
            raise ValueError(
                f"gamma={gamma} gives horizon 1/(1-gamma)={horizon:.0f}, but "
                f"expansion_cap={expansion_cap} >= horizon. A success at the cap "
                f"returns {g_succ(int(expansion_cap), gamma):.1f}, which does not "
                f"dominate doom {-horizon:.0f} with margin: past the horizon the "
                f"discounted objective saturates and a slow search is "
                f"indistinguishable from a hopeless one. Lower the cap below "
                f"{horizon:.0f}, or move gamma closer to 1."
            )


# ------------------------------------------------------------ step result ----

@dataclass
class StepResult:
    """Gymnasium-style terminated/truncated split. The distinction is NOT
    cosmetic -- it decides whether the critic bootstraps:

        y = r + gamma * (1 - terminated) * max_a' Q(s', a')      # NOT (1 - done)

    `truncated` means we stopped watching, not that the search failed. By the
    completeness proposition the planner would have kept going and eventually
    succeeded, so a capped state is worth -E[remaining expansions], not -1.
    Treating it as terminal leaks that -1 backwards and makes the critic
    systematically optimistic about deep searches (Pardo et al. 2018).
    """

    fringe: List[int]           # the beam, in ONNX packing order; action indexes this
    reward: float
    terminated: bool            # SUCC, or genuine DOOM on an unsolvable instance
    truncated: bool             # expansion cap only
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Loop control only. NEVER use this in a Bellman target."""
        return self.terminated or self.truncated


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
        gamma: float = DEFAULT_GAMMA,
        reward_mode: str = "absorbing",
        expansion_cap: Optional[int] = None,
        exploitation: Optional[int] = None,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        if not instance.solvable():
            raise ValueError(
                f"instance {instance.name!r} has delta(root)=inf (no reachable "
                f"goal). Unsolvable instances must be filtered at load time "
                f"(tree.partition_solvable) -- they contribute one immediate-doom "
                f"transition and teach nothing about ranking."
            )
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
            int(expansion_cap) if expansion_cap is not None
            else default_expansion_cap([instance])
        )
        # Validate gamma AND the dominance invariant now that the cap is known:
        # a success at the cap must dominate doom (cap < 1/(1-gamma)).
        assert_gamma(self.gamma, self.expansion_cap)
        self.doom_reward = doom_penalty(self.expansion_cap, self.gamma, self.reward_mode)

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
        # Eviction tracking -- see _track_eviction.
        self.evictions: List[Dict[str, object]] = []
        self._pending_eviction: Dict[int, int] = {}   # node -> expansion evicted at
        self.reservoir_sizes: List[int] = []
        self.beam_sizes: List[int] = []
        self.n_sterile_expansions = 0

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
        """OPTIMISTIC BOUND, and now we know how loose.

        -min delta over B u R assumes a reservoir node comes back for free. It
        does not: it returns only by a random draw against a growing reservoir
        (see _track_eviction). So regret = k - delta(root) is a LOWER BOUND on
        true regret -- every figure using it must say so on the axis, not in a
        caption. Never a training target.
        """
        return self.instance.v_star(self.fringe, self.reservoir, gamma=self.gamma)

    def _record_occupancy(self) -> None:
        self.reservoir_sizes.append(len(self.reservoir))
        self.beam_sizes.append(len(self.fringe))

    def beam_sterile_frac(self) -> float:
        """Fraction of the current beam whose subtree contains no goal."""
        if not self.fringe:
            return 0.0
        n = sum(1 for v in self.fringe if self.instance.delta[v] == INF_DELTA)
        return n / len(self.fringe)

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

    def _best_viable(self, nodes: Sequence[int]) -> Optional[int]:
        best, bd = None, INF_DELTA
        for v in nodes:
            if self.instance.delta[v] < bd:
                best, bd = v, self.instance.delta[v]
        return best

    def _track_eviction(self, best_before: Optional[int], expanded: int) -> None:
        """Record the beam's best viable node being pushed out into the reservoir.

        THE COMPOUNDING LOOP. push_vector does not keep the beam: it dumps the
        whole unexpanded beam into R, fills B with ch(v), and refills leftover
        slots AT RANDOM from R. So expanding a sterile v both floods B with v's
        sterile children AND EVICTS the good node into R, from which it returns
        only by a random draw -- against a reservoir that just grew.

        The cost of a sterile expansion is therefore NOT -1: it is -1 plus the
        expected wait to draw the good node back, and that wait grows with |R|.
        This is why Q*(s,v) for sterile v has no closed form (it depends on the
        refill process) and must be LEARNED by bootstrapping through real
        stochastic transitions -- the first thing here a regressor on delta
        cannot do.
        """
        if best_before is None or best_before == expanded:
            return
        if best_before in self.fringe:
            return                       # survived the rebuild; no eviction
        if best_before in self.reservoir and best_before not in self._pending_eviction:
            self._pending_eviction[best_before] = self.expansions

    def _resolve_recoveries(self) -> None:
        for node in list(self._pending_eviction):
            if node in self.fringe:
                at = self._pending_eviction.pop(node)
                self.evictions.append({
                    "node": node,
                    "evicted_at": at,
                    "recovered_at": self.expansions,
                    "recovery_steps": self.expansions - at,
                    "recovered": True,
                })

    def _finalize_evictions(self) -> None:
        """Never-recovered evictions are the expensive ones; count them."""
        for node, at in self._pending_eviction.items():
            self.evictions.append({
                "node": node,
                "evicted_at": at,
                "recovered_at": None,
                "recovery_steps": self.expansions - at,
                "recovered": False,
            })
        self._pending_eviction = {}

    def _terminated(self, reward: float, kind: str) -> StepResult:
        self.done = True
        return StepResult([], reward, terminated=True, truncated=False,
                          info=self._info(outcome=kind))

    def _doom(self) -> StepResult:
        """Genuine exhaustion. On a SOLVABLE instance this is unreachable.

        PROPOSITION (completeness of the wrapper). If delta(root) < inf then some
        root child has delta = d-1; it is either a goal (success) or it enters
        B u R. The reservoir never discards it, so it stays open until expanded,
        and expanding it exposes a delta = d-2 node. By induction B u R always
        contains a viable node, so it can be neither empty nor entirely
        non-viable. Hence DOOM <=> delta(root) = inf.

        So reaching here on a solvable instance is an ENV BUG, not a bad episode:
        fail loudly rather than absorb it into a reward.
        """
        if self.instance.solvable():
            raise AssertionError(
                f"DOOM fired on SOLVABLE instance {self.instance.name!r} "
                f"(delta(root)={self.instance.delta_root}) at expansion "
                f"{self.expansions}. The reservoir guarantees B u R always holds "
                f"a viable node, so this is an environment bug -- the transition "
                f"is dropping nodes. |B|={len(self.fringe)} |R|={len(self.reservoir)} "
                f"|visited|={len(self.visited)}"
            )
        return self._terminated(self.doom_reward, "doom")

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
        self.evictions = []
        self._pending_eviction = {}
        self.reservoir_sizes = []
        self.beam_sizes = []
        self.n_sterile_expansions = 0

        fresh, goal = self._generate_children(self.instance.root_id)
        if goal:
            return self._terminated(0.0, "success")
        self._rebuild_beam(fresh)
        self._record_occupancy()
        if not self.fringe:
            return self._doom()
        return StepResult(list(self.fringe), 0.0, terminated=False, truncated=False,
                          info=self._info(outcome="running"))

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

        best_before = self._best_viable(self.fringe)
        v = self.fringe.pop(action)
        self.order.remove(v)
        self.expansions += 1
        if self.instance.delta[v] == INF_DELTA:
            # A sterile expansion: v's whole subtree contains no goal. This is
            # node-level FAILURE -- the completeness proposition removed episode
            # doom, it did not remove failure. ~35% of CC_2_3_4__pl_7's nodes are
            # sterile, and the 197-expansion gap between bfs (231) and
            # hfs_oracle (34) is almost entirely spent in them. Avoiding these is
            # the primary thing the policy must learn.
            self.n_sterile_expansions += 1

        fresh, goal = self._generate_children(v)
        if goal:
            self._finalize_evictions()
            return self._terminated(0.0, "success")

        if fresh:
            # RESCORE PATH: push_vector fires (RL_node_to_add == 1).
            self._rebuild_beam(fresh)
            self._track_eviction(best_before, v)
            self.forced = False
            self.forced_action = None
        else:
            # STALE PATH: b_v == 0, so fringe_RL stays empty, `0 >= 1` is false
            # and push_vector never fires. No rebuild, no refill, no rescore —
            # the planner pops the next node by the ranks it already holds.
            if not self.fringe:
                if not self.reservoir:
                    return self._doom()
                self._pull_unscored()
            self._set_forced()

        self._resolve_recoveries()
        self._record_occupancy()

        if not self.fringe:
            return self._doom()
        if self.expansions >= self.expansion_cap:
            self._finalize_evictions()
            # TRUNCATION, not failure. The cap has no deployment counterpart as a
            # terminal state: the planner does not stop and declare failure at our
            # training budget, and by the completeness proposition it would
            # eventually succeed. So this is a rollout we stopped watching.
            # The successor fringe IS returned: the critic must bootstrap
            # max_a' Q(s',a') from it, otherwise the -1 leaks backwards and the
            # critic becomes optimistic about deep searches.
            self.done = True
            return StepResult(list(self.fringe), -1.0, terminated=False, truncated=True,
                              info=self._info(outcome="timeout"))
        return StepResult(list(self.fringe), -1.0, terminated=False, truncated=False,
                          info=self._info(outcome="running"))

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
        c.evictions = [dict(e) for e in self.evictions]
        c._pending_eviction = dict(self._pending_eviction)
        c.reservoir_sizes = list(self.reservoir_sizes)
        c.beam_sizes = list(self.beam_sizes)
        c.n_sterile_expansions = self.n_sterile_expansions
        # Guard against silent drift: this hand-rolled copy bypasses __init__, so
        # any field added later is missing here and the clone diverges from the
        # parent in a way that surfaces far from the cause.
        missing = set(vars(self)) - set(vars(c))
        if missing:
            raise AssertionError(
                f"FringeEnv.clone() does not copy {sorted(missing)}; add them "
                f"above or the counterfactual successors will be wrong."
            )
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
    sterile_beam = []
    while not res.done:
        sterile_beam.append(env.beam_sterile_frac())
        if env.forced:
            action, ranking = env.forced_action, None
        else:
            ranking = list(policy(res.fringe))
            action = ranking[0]
        res = env.step(action, ranking)
        total += res.reward
        disc += g * res.reward
        g *= env.gamma
    solved = res.info["outcome"] == "success"
    recs = [e for e in env.evictions if e["recovered"]]
    return {
        "expansions": res.info["expansions"],
        "outcome": res.info["outcome"],
        # `solved` is the coverage indicator: % of instances solved within the
        # cap. It is the PRIMARY selection metric -- a policy that solves 60% at
        # regret 2 is not better than one that solves 100% at regret 8, and a
        # mean regret over solved instances alone would hide exactly that.
        "solved": solved,
        "terminated": res.terminated,
        "truncated": res.truncated,
        # regret and expansions are meaningful only on SOLVED instances; None
        # here so an aggregator cannot silently average a truncation in.
        "regret": env.instance.regret(int(res.info["expansions"])) if solved else None,
        "return": total,
        "discounted_return": disc,
        "n_forced_steps": res.info["n_forced_steps"],
        "n_unscored_pulls": res.info["n_unscored_pulls"],
        # --- node-level FAILURE and the compounding loop (F10) ---
        # expansions_sterile_frac is the single clearest measure of whether the
        # policy learned anything: hfs_oracle should be ~0, bfs large.
        "expansions_sterile_frac": env.n_sterile_expansions / max(1, env.expansions),
        "n_sterile_expansions": env.n_sterile_expansions,
        "beam_sterile_frac": (sum(sterile_beam) / len(sterile_beam)) if sterile_beam else 0.0,
        "eviction_events": len(env.evictions),
        "eviction_recovery_steps_mean": (
            sum(e["recovery_steps"] for e in recs) / len(recs) if recs else None
        ),
        "eviction_never_recovered": sum(1 for e in env.evictions if not e["recovered"]),
        "reservoir_size_mean": (
            sum(env.reservoir_sizes) / len(env.reservoir_sizes) if env.reservoir_sizes else 0.0
        ),
        "reservoir_size_max": max(env.reservoir_sizes) if env.reservoir_sizes else 0,
        "beam_size_mean": (
            sum(env.beam_sizes) / len(env.beam_sizes) if env.beam_sizes else 0.0
        ),
        "beam_size_max": max(env.beam_sizes) if env.beam_sizes else 0,
    }


def coverage_at(rollouts, budget: int) -> float:
    """% solved within `budget` expansions. One point on the cactus plot."""
    rs = list(rollouts)
    if not rs:
        return 0.0
    return sum(1 for r in rs if r["solved"] and r["expansions"] <= budget) / len(rs)


def coverage_curve(rollouts, budgets: Sequence[int]) -> List[Dict[str, float]]:
    """The F1' cactus plot's data: coverage as a function of budget.

    This is what planner papers compare on, and it is the honest answer to "which
    cap?": report the whole curve and let the reader pick the budget.
    """
    return [{"budget": int(b), "coverage": coverage_at(rollouts, b)} for b in budgets]


def aggregate_rollouts(rollouts, reference_budget: Optional[int] = None) -> Dict[str, object]:
    """Coverage-first aggregation. Never averages expansions over unsolved runs.

    `coverage` (at the declared reference budget when given, else at the rollout
    cap) is the PRIMARY selection metric; regret over solved instances is the
    tie-break. doom_rate is reported but is provably 0 on solvable data.
    """
    rs = list(rollouts)
    n = max(1, len(rs))
    solved = [r for r in rs if r["solved"]]
    if reference_budget is not None:
        scored = [r for r in solved if r["expansions"] <= reference_budget]
    else:
        scored = solved
    regrets = [r["regret"] for r in scored]
    exps = [r["expansions"] for r in scored]
    returns = [r["return"] for r in rs if "return" in r]
    return {
        "n": len(rs),
        "coverage": len(scored) / n,
        "reference_budget": reference_budget,
        "doom_rate": sum(1 for r in rs if r["outcome"] == "doom") / n,
        "timeout_rate": sum(1 for r in rs if r["truncated"]) / n,
        # over SOLVED-WITHIN-BUDGET only
        "regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
        "expansions_mean": (sum(exps) / len(exps)) if exps else None,
        # the objective itself, over ALL rollouts (a timeout's return is real cost)
        "return_mean": (sum(returns) / len(returns)) if returns else None,
        "n_solved": len(scored),
    }
