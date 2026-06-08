"""Reconstructed-tree fringe environment (offline MDP). See DESIGN.md §2.

The generation table (File Path, Depth, Distance From Goal, Goal,
File Path Predecessor, Action) is reconstructed into the search tree following
src/data.py::_build_tree_structures conventions (min-depth parent wins; root =
predecessor that never appears as a child).  Episodes simulate the C++
RL_BestFirst loop with RefillMode::RANDOM:

  - expansion pops one *chosen* fringe slot (the action),
  - its unvisited children become the new beam (first F), overflow and the
    leftover old beam go to the reservoir, free slots are refilled uniformly
    at random from the reservoir,
  - reward 0 + terminal when a generated child is the goal (the C++ goal test
    runs at successor generation), else reward -1,
  - terminal with no bonus on fringe+reservoir exhaustion (dead end).

Ground-truth distance d* (unreachable = 1e6) is kept as EVALUATION ORACLE
ONLY — it never enters training.
"""

from __future__ import annotations

import csv
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

UNREACHABLE_DISTANCE = 1e6


@dataclass
class TreeInstance:
    name: str
    csv_path: str
    state_paths: List[str]  # index = state id
    depth: List[int]
    distance: List[float]  # d* oracle; UNREACHABLE_DISTANCE marker kept as-is
    is_goal: List[bool]
    children: List[List[int]]  # state id -> child state ids (edge order = CSV)
    root_id: int  # the unique Depth-0 state (its CSV predecessor is a dummy)
    n_orphan_states: int  # states not reachable from the root (table anomalies)

    @property
    def n_states(self) -> int:
        return len(self.state_paths)

    def state_paths_abs(self, repo_root: str | Path) -> List[str]:
        """CSV File Path values are repo-root-relative; resolve them."""
        root = Path(repo_root)
        return [
            p if Path(p).is_absolute() else str(root / p) for p in self.state_paths
        ]

    def stats(self) -> Dict[str, object]:
        n = self.n_states
        n_goal = sum(self.is_goal)
        n_unreachable = sum(1 for d in self.distance if d >= UNREACHABLE_DISTANCE)
        out_deg = [len(c) for c in self.children]
        internal = [d for d in out_deg if d > 0]
        n_dead_leaves = sum(
            1
            for sid in range(n)
            if not self.children[sid] and not self.is_goal[sid]
        )
        depth_hist: Dict[int, int] = {}
        for d in self.depth:
            depth_hist[d] = depth_hist.get(d, 0) + 1
        return {
            "n_states": n,
            "n_goal_states": int(n_goal),
            "n_unreachable_states": int(n_unreachable),
            "n_dead_end_leaves": int(n_dead_leaves),
            "branching_internal_mean": (
                sum(internal) / len(internal) if internal else 0.0
            ),
            "root_branching": len(self.children[self.root_id]),
            "n_orphan_states": int(self.n_orphan_states),
            "depth_min": min(self.depth) if self.depth else 0,
            "depth_max": max(self.depth) if self.depth else 0,
            "depth_hist": dict(sorted(depth_hist.items())),
            "optimal_expansions": self.optimal_expansions(),
        }

    def optimal_expansions(self) -> Optional[int]:
        """Min #expansions to generate a goal = depth of shallowest goal.

        Expanding the root (depth 0) is expansion 1 and generates depth-1
        states; expanding the goal's parent (depth d*-1... i.e. depth(g)-1)
        is expansion depth(g) on a perfectly ranked run.
        """
        goal_depths = [self.depth[i] for i in range(self.n_states) if self.is_goal[i]]
        return min(goal_depths) if goal_depths else None


def load_tree_instance(csv_path: str | Path, name: Optional[str] = None) -> TreeInstance:
    csv_path = Path(csv_path)
    rows = list(csv.DictReader(csv_path.open()))

    path_to_id: Dict[str, int] = {}
    state_paths: List[str] = []
    depth: List[int] = []
    distance: List[float] = []
    # First pass: assign ids in row order; min-depth row wins on duplicates
    # (mirrors src/data.py::_build_tree_structures).
    for r in rows:
        p = r["File Path"].strip()
        d_i = int(r["Depth"])
        dist_f = float(r["Distance From Goal"])
        sid = path_to_id.get(p)
        if sid is None:
            path_to_id[p] = len(state_paths)
            state_paths.append(p)
            depth.append(d_i)
            distance.append(dist_f)
        elif d_i < depth[sid]:
            depth[sid] = d_i
            distance[sid] = dist_f

    children: List[List[int]] = [[] for _ in state_paths]
    n_orphan_edges = 0
    for r in rows:
        child = path_to_id[r["File Path"].strip()]
        pred = r["File Path Predecessor"].strip()
        pid = path_to_id.get(pred)
        if pid is not None:
            if child not in children[pid]:
                children[pid].append(child)
        else:
            # Predecessor never recorded as a state.  The root's dummy
            # `init.dot` predecessor lands here by construction; a handful of
            # deep rows do too (generator-side dedup anomalies, <1% of edges)
            # — their subtrees are unreachable and simply never enter
            # episodes.
            n_orphan_edges += 1

    # Root: the unique Depth-0 state.
    root_ids = [i for i, d in enumerate(depth) if d == 0]
    if len(root_ids) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly 1 depth-0 root state, got "
            f"{len(root_ids)}"
        )
    root_id = root_ids[0]

    # Reachability from the root (stats only).
    reachable = {root_id}
    stack = [root_id]
    while stack:
        s = stack.pop()
        for c in children[s]:
            if c not in reachable:
                reachable.add(c)
                stack.append(c)

    return TreeInstance(
        name=name or csv_path.parent.name,
        csv_path=str(csv_path),
        state_paths=state_paths,
        depth=depth,
        distance=distance,
        is_goal=[d == 0.0 for d in distance],
        children=children,
        root_id=root_id,
        n_orphan_states=len(state_paths) - len(reachable),
    )


@dataclass
class StepResult:
    fringe: List[int]
    reward: float
    done: bool
    info: Dict[str, object] = field(default_factory=dict)


class FringeEnv:
    """Gym-style fringe MDP over one reconstructed tree (state ids only).

    Observations are lists of state ids; tensor encoding is the caller's job
    (src/offline/encoder.pack_fringe on the instance's cache).
    """

    def __init__(
        self,
        instance: TreeInstance,
        fringe_size: int = 32,
        seed: int = 0,
        expansion_cap: Optional[int] = None,
    ):
        self.instance = instance
        self.fringe_size = int(fringe_size)
        self.rng = random.Random(seed)
        self.expansion_cap = (
            int(expansion_cap) if expansion_cap is not None else 2 * instance.n_states
        )
        self.fringe: List[int] = []
        self.reservoir: List[int] = []
        self.visited: set[int] = set()
        self.expansions = 0
        self.done = True

    def _generate_children(self, child_ids: Sequence[int]) -> Tuple[List[int], bool]:
        """Visited-dedup + goal test at generation (mirrors SpaceSearcher)."""
        fresh: List[int] = []
        for c in child_ids:
            if c in self.visited:
                continue
            self.visited.add(c)
            if self.instance.is_goal[c]:
                return fresh, True
            fresh.append(c)
        return fresh, False

    def _rebuild_beam(self, new_states: List[int]) -> None:
        """push_vector semantics: old beam -> reservoir, new states first
        (overflow -> reservoir), then uniform random refill."""
        self.reservoir.extend(self.fringe)
        self.fringe = []
        beam = new_states[: self.fringe_size]
        self.reservoir.extend(new_states[self.fringe_size:])
        while len(beam) < self.fringe_size and self.reservoir:
            idx = self.rng.randrange(len(self.reservoir))
            self.reservoir[idx], self.reservoir[-1] = (
                self.reservoir[-1],
                self.reservoir[idx],
            )
            beam.append(self.reservoir.pop())
        self.fringe = beam

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Expand the root (forced, counts as expansion 1)."""
        if seed is not None:
            self.rng.seed(seed)
        self.fringe = []
        self.reservoir = []
        self.visited = {self.instance.root_id}
        self.expansions = 1
        fresh, goal = self._generate_children(
            self.instance.children[self.instance.root_id]
        )
        if goal:
            self.done = True
            return StepResult([], 0.0, True, self._info(goal_found=True))
        self._rebuild_beam(fresh)
        self.done = not self.fringe
        return StepResult(
            list(self.fringe), 0.0, self.done, self._info(goal_found=False)
        )

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() on a finished episode; call reset().")
        if not (0 <= action < len(self.fringe)):
            raise IndexError(f"action {action} out of fringe range {len(self.fringe)}")

        chosen = self.fringe.pop(action)
        self.expansions += 1
        fresh, goal = self._generate_children(self.instance.children[chosen])
        if goal:
            self.done = True
            return StepResult([], 0.0, True, self._info(goal_found=True))

        self._rebuild_beam(fresh)
        if not self.fringe or self.expansions >= self.expansion_cap:
            self.done = True
            return StepResult([], -1.0, True, self._info(goal_found=False))
        return StepResult(list(self.fringe), -1.0, False, self._info(goal_found=False))

    def _info(self, goal_found: bool) -> Dict[str, object]:
        return {
            "expansions": self.expansions,
            "goal_found": goal_found,
            "fringe_len": len(self.fringe),
            "reservoir_len": len(self.reservoir),
        }


def rollout(
    env: FringeEnv,
    policy,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run one episode; policy(fringe: List[int]) -> action index."""
    res = env.reset(seed=seed)
    total_reward = 0.0
    while not res.done:
        action = policy(res.fringe)
        res = env.step(action)
        total_reward += res.reward
    return {
        "expansions": res.info["expansions"],
        "goal_found": res.info["goal_found"],
        "return": total_reward,
    }


def bfs_expansions(instance: TreeInstance, cap: Optional[int] = None) -> Dict[str, object]:
    """Reference: uncapped FIFO search on the tree, goal test at generation."""
    cap = cap if cap is not None else 4 * instance.n_states
    visited: set[int] = {instance.root_id}
    queue: deque[int] = deque()
    expansions = 1  # root expansion
    for c in instance.children[instance.root_id]:
        if c in visited:
            continue
        visited.add(c)
        if instance.is_goal[c]:
            return {"expansions": expansions, "goal_found": True}
        queue.append(c)
    while queue and expansions < cap:
        s = queue.popleft()
        expansions += 1
        for c in instance.children[s]:
            if c in visited:
                continue
            visited.add(c)
            if instance.is_goal[c]:
                return {"expansions": expansions, "goal_found": True}
            queue.append(c)
    return {"expansions": expansions, "goal_found": False}


def oracle_policy(instance: TreeInstance):
    """Greedy on ground-truth -d* (eval reference only)."""

    def _policy(fringe: List[int]) -> int:
        return min(range(len(fringe)), key=lambda k: instance.distance[fringe[k]])

    return _policy
