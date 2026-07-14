"""Reconstructed search tree + oracle quantities (environment-side only).

The generation table
    File Path, Depth, Distance From Goal, Goal, File Path Predecessor, Action
is reconstructed into the search tree the offline MDP moves inside.

TWO DISTANCES, AND THEY ARE NOT THE SAME
----------------------------------------
`h*(v)` is the CSV `Distance From Goal` column: the true distance in the
*hash-deduplicated state DAG*.

`delta(v)` is the distance to the nearest goal *inside the reconstructed tree*:

    delta(v) = 0                  if v is a goal
             = inf                if v is a non-goal leaf
             = 1 + min_c delta(c) otherwise

They differ because the reconstruction is a DFS spanning tree of that DAG: the
`File Path Predecessor` pointer records the *first discoverer in DFS order*, so
tree paths overestimate DAG distance. Measured on the shipped tables:

    CC_2_3_4__pl_7     h*(root) = 29   delta(root) = 34
    SC_R_10_10__pl_10  h*(root) = 10   delta(root) = 24

and `delta >= h*` node-wise on both (0 violations).

The MDP moves only within the reconstructed tree, so **delta is the correct
reference for V* and for regret**. Using the CSV column instead would charge the
policy 5 expansions of regret on CC that are an artifact of data generation.

Both quantities are ENVIRONMENT-SIDE ONLY: diagnostics, oracle behaviour and
model selection. Neither is ever a feature or a training target.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# CSV marker for "no path to a goal in the DAG".
UNREACHABLE_DISTANCE = 1e6

# delta for a node with no goal anywhere in its subtree.
INF_DELTA = float("inf")


@dataclass
class TreeInstance:
    """One reconstructed instance. State ids are indices into `state_paths`."""

    name: str
    csv_path: str
    state_paths: List[str]          # id -> repo-relative DOT path
    depth: List[int]
    h_star: List[float]             # CSV `Distance From Goal` (DAG distance)
    is_goal: List[bool]
    children: List[List[int]]       # id -> child ids, CSV edge order
    root_id: int
    n_orphan_states: int            # states unreachable from the root
    delta: List[float]              # tree distance to nearest goal (INF_DELTA if none)
    # Separated mode only: repo-relative path to this instance's goal_tree.dot.
    # Driven by kind_of_data, NEVER by the presence of the `Goal` column (the
    # column is populated in merged runs too, but the file is not written there).
    goal_path: Optional[str] = None

    @property
    def n_states(self) -> int:
        return len(self.state_paths)

    @property
    def delta_root(self) -> float:
        """Fewest expansions any policy can reach a goal in, on this tree.

        The goal test fires at *generation* (SpaceSearcher.tpp:162), so expanding
        the goal's parent ends the episode: delta(root) expansions exactly.
        """
        return self.delta[self.root_id]

    def solvable(self) -> bool:
        return self.delta_root != INF_DELTA

    def state_paths_abs(self, repo_root: str | Path) -> List[str]:
        root = Path(repo_root)
        return [p if Path(p).is_absolute() else str(root / p) for p in self.state_paths]

    def goal_path_abs(self, repo_root: str | Path) -> Optional[str]:
        if self.goal_path is None:
            return None
        p = Path(self.goal_path)
        return str(p) if p.is_absolute() else str(Path(repo_root) / p)

    # ---- oracle quantities (diagnostics / selection only) ----

    def v_star(self, beam: Iterable[int], reservoir: Iterable[int] = ()) -> float:
        """V*(B, R) = -min delta over viable v in B u R.

        The reservoir IS included: with the completeness wrapper a node parked in
        R is not lost, so the attainable optimum must account for it.

        OPTIMISTIC BOUND, not the attainable optimum. Refill is uniformly random
        (RefillMode::RANDOM) and therefore not under the policy's control, so a
        low-delta node sitting in R may never be handed back to the beam. Every
        figure and log that uses this must say so.
        """
        best = INF_DELTA
        for v in beam:
            if self.delta[v] < best:
                best = self.delta[v]
        for v in reservoir:
            if self.delta[v] < best:
                best = self.delta[v]
        return -best if best != INF_DELTA else -INF_DELTA

    def regret(self, expansions: int) -> float:
        """regret(pi) = k_pi - delta(root), in expansions. Lower is better."""
        return float(expansions) - float(self.delta_root)

    def oracle_action(self, beam: Sequence[int]) -> int:
        """argmin delta over the beam (ties -> lowest slot). Clairvoyant."""
        return min(range(len(beam)), key=lambda k: self.delta[beam[k]])

    def stats(self) -> Dict[str, object]:
        n = self.n_states
        reachable = self._reachable()
        out_deg = [len(c) for c in self.children]
        internal = [d for i, d in enumerate(out_deg) if d > 0 and i in reachable]
        dead = [i for i in reachable if not self.children[i] and not self.is_goal[i]]
        depth_hist: Dict[int, int] = {}
        for i in reachable:
            depth_hist[self.depth[i]] = depth_hist.get(self.depth[i], 0) + 1
        n_reach = max(1, len(reachable))
        return {
            "n_states": n,
            "n_reachable": len(reachable),
            "n_goal_states": int(sum(1 for i in reachable if self.is_goal[i])),
            "goal_density": round(sum(1 for i in reachable if self.is_goal[i]) / n_reach, 4),
            "n_dead_end_leaves": len(dead),
            "sterile_leaf_density": round(len(dead) / n_reach, 4),
            "n_delta_inf": int(sum(1 for i in reachable if self.delta[i] == INF_DELTA)),
            "branching_internal_mean": (sum(internal) / len(internal)) if internal else 0.0,
            "root_branching": len(self.children[self.root_id]),
            "n_orphan_states": int(self.n_orphan_states),
            "depth_min": min((self.depth[i] for i in reachable), default=0),
            "depth_max": max((self.depth[i] for i in reachable), default=0),
            "depth_hist": dict(sorted(depth_hist.items())),
            "h_star_root": self.h_star[self.root_id],
            "delta_root": self.delta_root,
            # The artifact the brief warns about: tree depth overestimates DAG
            # distance, so selecting on h* would charge phantom regret.
            "delta_minus_h_star_root": (
                self.delta_root - self.h_star[self.root_id]
                if self.solvable() else None
            ),
            "forced_chain_len": self.initial_forced_chain_len(),
        }

    def initial_forced_chain_len(self) -> int:
        """Length of the leading b_v == 1 chain from the root.

        Expansions along it are non-decisions: |A(s)| == 1, so they dilute any
        aggregate that averages over states.
        """
        v, n = self.root_id, 0
        while len(self.children[v]) == 1:
            n += 1
            v = self.children[v][0]
        return n

    def _reachable(self) -> set[int]:
        seen = {self.root_id}
        stack = [self.root_id]
        while stack:
            for c in self.children[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen


def compute_delta(
    children: Sequence[Sequence[int]],
    is_goal: Sequence[bool],
) -> List[float]:
    """delta(v) = tree distance from v to the nearest goal in its subtree.

    Implemented as a multi-source BFS from the goal set over *reversed* edges,
    which is the shortest-path fixpoint of the brief's recurrence:

        delta(v) = 0 if goal; inf if non-goal leaf; else 1 + min_c delta(c)

    A reversed-DFS-preorder fold would also work on a strict tree, but only there
    — this formulation is correct on any graph and does not silently produce
    wrong numbers if the reconstruction ever stops being a tree.
    """
    n = len(children)
    parents: List[List[int]] = [[] for _ in range(n)]
    for v, cs in enumerate(children):
        for c in cs:
            parents[c].append(v)

    delta: List[float] = [INF_DELTA] * n
    q: deque[int] = deque()
    for v in range(n):
        if is_goal[v]:
            delta[v] = 0.0
            q.append(v)
    while q:
        v = q.popleft()
        for p in parents[v]:
            if delta[p] == INF_DELTA:
                delta[p] = delta[v] + 1.0
                q.append(p)
    return delta


def load_tree_instance(
    csv_path: str | Path,
    name: Optional[str] = None,
    kind_of_data: str = "merged",
) -> TreeInstance:
    """Reconstruct one instance from its generation table.

    Conventions (preserved from the previous loader):
      - duplicate `File Path` rows: the min-depth row wins;
      - the root is the unique Depth-0 state (its CSV predecessor is a dummy
        `init.dot` that never appears as a state);
      - edges whose predecessor never appears as a state are dropped as orphans
        (<1% generator anomalies); their subtrees never enter episodes.
    """
    csv_path = Path(csv_path)
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path}: empty generation table.")

    if kind_of_data not in ("merged", "separated"):
        raise ValueError(f"kind_of_data must be merged|separated, got {kind_of_data!r}")

    # Goal graph loading is driven by kind_of_data ONLY. In merged mode the
    # `Goal` column is still populated but goal_tree.dot is not written (the goal
    # is inlined into every state DOT), so capturing it there would feed a file
    # that does not exist.
    goal_path: Optional[str] = None
    if kind_of_data == "separated":
        raw = rows[0].get("Goal")
        if raw is None or not raw.strip():
            raise ValueError(
                f"{csv_path}: separated mode requires a non-empty `Goal` column."
            )
        goal_path = raw.strip()

    path_to_id: Dict[str, int] = {}
    state_paths: List[str] = []
    depth: List[int] = []
    h_star: List[float] = []
    for r in rows:
        p = r["File Path"].strip()
        d = int(r["Depth"])
        dist = float(r["Distance From Goal"])
        sid = path_to_id.get(p)
        if sid is None:
            path_to_id[p] = len(state_paths)
            state_paths.append(p)
            depth.append(d)
            h_star.append(dist)
        elif d < depth[sid]:
            depth[sid] = d
            h_star[sid] = dist

    children: List[List[int]] = [[] for _ in state_paths]
    n_orphan_edges = 0
    for r in rows:
        c = path_to_id[r["File Path"].strip()]
        pid = path_to_id.get(r["File Path Predecessor"].strip())
        if pid is None:
            n_orphan_edges += 1
            continue
        if c not in children[pid]:
            children[pid].append(c)

    root_ids = [i for i, d in enumerate(depth) if d == 0]
    if len(root_ids) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly 1 depth-0 root state, got {len(root_ids)}"
        )
    root_id = root_ids[0]

    is_goal = [d == 0.0 for d in h_star]
    delta = compute_delta(children, is_goal)

    reachable = {root_id}
    stack = [root_id]
    while stack:
        for c in children[stack.pop()]:
            if c not in reachable:
                reachable.add(c)
                stack.append(c)

    return TreeInstance(
        name=name or csv_path.parent.name,
        csv_path=str(csv_path),
        state_paths=state_paths,
        depth=depth,
        h_star=h_star,
        is_goal=is_goal,
        children=children,
        root_id=root_id,
        n_orphan_states=len(state_paths) - len(reachable),
        delta=delta,
        goal_path=goal_path,
    )


def max_success_expansions(instances: Sequence[TreeInstance]) -> int:
    """K_max = the longest reachable success across the data = max delta(root)."""
    ks = [int(i.delta_root) for i in instances if i.solvable()]
    if not ks:
        raise ValueError("no solvable instance: K_max is undefined.")
    return max(ks)


def partition_solvable(
    instances: Sequence[TreeInstance],
) -> tuple[List[TreeInstance], List[TreeInstance]]:
    """Split into (solvable, unsolvable). Unsolvable = delta(root) = inf.

    An unsolvable instance has no reachable goal, so by the completeness
    proposition (see env._doom) it produces exactly ONE useful transition -- an
    immediate doom -- and teaches nothing about ranking. Including it would
    dilute coverage and regret aggregates with an instance no policy can ever
    solve. Callers must log the excluded names and count them in the manifest
    rather than dropping them silently.
    """
    ok = [i for i in instances if i.solvable()]
    bad = [i for i in instances if not i.solvable()]
    return ok, bad
