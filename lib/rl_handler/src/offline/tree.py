"""Reconstructed search tree + oracle quantities (environment-side only).

The generation table
    File Path, Depth, Distance From Goal, Goal, File Path Predecessor, Action
is reconstructed into the search tree the offline MDP moves inside.

TWO DISTANCES, AND THEY ARE NOT THE SAME
----------------------------------------
`h*(v)` is the CSV `Distance From Goal` column, as computed by the generator.
WARNING: It is NOT "the true distance in the state DAG", despite what it looks
like: the generator's memo is keyed by STATE ALONE inside a depth-bounded DFS
(`TrainingDataset.tpp`), so a state locked as a 1e6 leaf on a deep first visit
returns that stale verdict when later reached shallower with room to expand. `h*`
is therefore a DFS-discovery artifact too, not a DAG shortest distance.

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

WHAT delta IS, AND WHAT IT IS NOT (state this before making any claim)
----------------------------------------------------------------------
`delta` is the EXACT shortest-path distance to the nearest goal WITHIN the
generated tree -- `compute_delta` is a multi-source BFS from the goal set over
reversed edges, correct on any graph. The MDP's action space IS that tree, so the
labels are exact for the object the agent navigates.

`delta` is NOT the planner's optimal plan length for the instance. The tree is a
SEED-DEPENDENT DFS SAMPLE of the true state space, so root-to-goal distance in the
tree is an upper bound on the true optimal, tight only where the DFS happened to
sample a shortest path. Measured on CC_2_2_3__pl_4 (true optimal 4, identical flags,
seed alone varied): delta_root = 14 (seed 42) / 6 (seed 43) / 7 (seed 44). None is 4.

So: never read delta_root as "the optimal plan length", and never claim the data is
shortest-path-faithful -- the generator's algorithm cannot deliver that (see
`usability.py`, which retired the `delta_root == pl_N` criterion for this reason).
A comparison between arms run on the SAME tree with the SAME labels stays valid
regardless: relative ranking never needed absolute optimality.

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

# CENSORED vs STERILE -- the distinction fix-1A introduces.
# ---------------------------------------------------------
# delta(v) = inf conflates two different facts:
#   PROVABLY STERILE: every path from v ends in a genuine dead end (a leaf the
#     generator stopped at because it had NO fresh successors). The planner
#     really would waste expansions in that subtree.
#   CENSORED: the verdict is an artifact of generation, not a fact about the
#     search space. Two cases:
#       (a) a non-goal leaf sitting AT the generation depth bound -- the
#           generator stopped writing children there, but at deployment the
#           state expands normally and may sit on the shortest path
#           (usability.py: 145/154 "poisoned" rows on CC pl_7 were exactly
#           this);
#       (b) delta(v)=inf while h*(v) is FINITE -- the generator itself asserts
#           a goal is reachable from v in the DAG; the spanning-tree
#           reconstruction merely lacks the edge.
#     Censoredness propagates upward through inf-delta ancestors: an internal
#     node whose inf verdict rests on a censored subtree is itself unknown,
#     not proven dead.
# Training must not teach "avoid v" from a censored label: that is inventing
# a target (see two_head_baseline's masked MSE for the same principle).


def compute_censored(
    children: Sequence[Sequence[int]],
    is_goal: Sequence[bool],
    delta: Sequence[float],
    h_star: Sequence[float],
    depth: Sequence[int],
    depth_bound: Optional[int] = None,
) -> List[bool]:
    """censored[v] = delta(v) is inf but the tree cannot PROVE sterility.

    Seeds:
      * bound-hit leaves: non-goal leaf, delta=inf, depth >= depth_bound;
      * label contradictions: delta=inf but h* finite (the generator says a
        goal IS reachable from v in the DAG).
    Propagation: upward through parents whose delta is inf (a finite-delta
    parent has a proven goal path; its verdict does not depend on the
    censored branch).

    `depth_bound` defaults to max(depth) -- a conservative stand-in for the
    generator's --dataset_depth, which the CSV does not record (the dataspec
    does; callers that know it should pass it). Conservative in the safe
    direction: a genuine dead end at exactly the max depth is treated as
    unknown rather than a genuine dead end mislabeled as such.
    """
    n = len(children)
    if depth_bound is None:
        depth_bound = max(depth) if n else 0
    censored = [False] * n
    q: deque[int] = deque()
    for v in range(n):
        if delta[v] != INF_DELTA:
            continue
        bound_hit_leaf = (
            not children[v] and not is_goal[v] and depth[v] >= depth_bound
        )
        contradicted = h_star[v] < UNREACHABLE_DISTANCE
        if bound_hit_leaf or contradicted:
            censored[v] = True
            q.append(v)

    parents: List[List[int]] = [[] for _ in range(n)]
    for v, cs in enumerate(children):
        for c in cs:
            parents[c].append(v)
    while q:
        v = q.popleft()
        for p in parents[v]:
            if delta[p] == INF_DELTA and not censored[p]:
                censored[p] = True
                q.append(p)
    return censored


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
    # censored[v]: delta(v)=inf is a generation artifact (depth-bound leaf /
    # h*-contradicted), NOT proven sterility -- see compute_censored. None (e.g.
    # a hand-built test instance) normalises to all-False in __post_init__.
    censored: Optional[List[bool]] = None
    # Edge rows whose predecessor never appears as a state (dropped at load).
    # Surfaced so a generator regression cannot silently discard structure.
    n_orphan_edges: int = 0
    n_edge_rows: int = 0

    def __post_init__(self) -> None:
        if self.censored is None:
            self.censored = [False] * len(self.state_paths)

    @property
    def n_states(self) -> int:
        return len(self.state_paths)

    def is_censored(self, v: int) -> bool:
        return bool(self.censored[v])

    def provably_sterile(self, v: int) -> bool:
        """delta=inf AND the verdict does not rest on a censored subtree."""
        return self.delta[v] == INF_DELTA and not self.censored[v]

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

    def v_star(self, beam: Iterable[int], reservoir: Iterable[int] = (),
               gamma: float = 1.0) -> float:
        """V*(B, R) = the discounted return of reaching the nearest goal in B u R.

        gamma = 1:  -min delta          (the SSP limit, in expansion units)
        gamma < 1:  -(1 - gamma^d)/(1-gamma)  with d = min delta -- the paper's V*.

        For d << 1/(1-gamma) these agree to <0.2% (d=24, gamma=0.9999: -23.97 vs -24),
        so the discounted form is the paper's formula producing the SSP numbers.

        The reservoir IS included: with the completeness wrapper a node parked in R is
        not lost, so the attainable optimum must account for it.

        OPTIMISTIC BOUND, not the attainable optimum. Refill is uniformly random
        (RefillMode::RANDOM) and therefore not under the policy's control, so a
        low-delta node sitting in R may never be handed back to the beam. Every figure
        and log that uses this must say so.
        """
        best = INF_DELTA
        for v in beam:
            if self.delta[v] < best:
                best = self.delta[v]
        for v in reservoir:
            if self.delta[v] < best:
                best = self.delta[v]
        if best == INF_DELTA:
            return -INF_DELTA
        if gamma >= 1.0:
            return -float(best)
        return -(1.0 - gamma ** best) / (1.0 - gamma)

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
            "n_censored": int(sum(1 for i in reachable if self.censored[i])),
            "censored_frac": round(
                sum(1 for i in reachable if self.censored[i]) / n_reach, 4),
            "n_orphan_edges": int(self.n_orphan_edges),
            "orphan_edge_frac": round(
                self.n_orphan_edges / max(1, self.n_edge_rows), 4),
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
    depth_bound: Optional[int] = None,
) -> TreeInstance:
    """Reconstruct one instance from its generation table.

    Conventions (preserved from the previous loader):
      - duplicate `File Path` rows: the min-depth row wins;
      - the root is the unique Depth-0 state (its CSV predecessor is a dummy
        `init.dot` that never appears as a state);
      - edges whose predecessor never appears as a state are dropped as orphans
        (<1% generator anomalies, now COUNTED on the instance so a regression
        past that rate is visible); their subtrees never enter episodes.

    `depth_bound`: the generator's --dataset_depth for censoring (see
    compute_censored). The CSV does not record it; the batch .dataspec does.
    Defaults to max observed depth -- conservative, see compute_censored.
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
    censored = compute_censored(children, is_goal, delta, h_star, depth,
                                depth_bound=depth_bound)

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
        censored=censored,
        n_orphan_edges=n_orphan_edges,
        n_edge_rows=len(rows),
    )


def bfs_open_max(instance: TreeInstance) -> int:
    """Max |open list| along an uncapped BFS trajectory.

    NOT A BOUND ON OTHER POLICIES -- and the previous implementation's docstring
    claiming it was "a policy-free UPPER BOUND on how many states can be
    simultaneously live" is wrong. The search stops when a goal is GENERATED, so
    a policy that finds a goal later expands more nodes and (because the
    reservoir discards nothing) accumulates a LARGER open set. Measured on
    CC_2_3_4__pl_7: this function returns 105, while the env's actual max |B u R|
    is 117 under bfs and only 17-18 under hfs_oracle/dfs -- which finish sooner.

    Keep this as a cheap screening statistic only. The honest occupancy figure is
    diagnostics.measure_occupancy(), which rolls out the real behaviour policies
    in the real env.

    Goal test at generation, mirroring the planner.
    """
    visited = {instance.root_id}
    queue: deque[int] = deque()
    fmax = 0
    for c in instance.children[instance.root_id]:
        if c in visited:
            continue
        visited.add(c)
        queue.append(c)
        if instance.is_goal[c]:
            return max(fmax, len(queue))
    fmax = max(fmax, len(queue))
    while queue:
        v = queue.popleft()
        for c in instance.children[v]:
            if c in visited:
                continue
            visited.add(c)
            queue.append(c)
            if instance.is_goal[c]:
                return max(fmax, len(queue))
        fmax = max(fmax, len(queue))
    return fmax


def bfs_binds_at(instance: TreeInstance, fringe_size: int) -> bool:
    """Cheap screen only -- see bfs_open_max on why this is not authoritative.
    Use diagnostics.measure_occupancy() for the real binds/inert verdict."""
    return bfs_open_max(instance) > int(fringe_size)


def max_success_expansions(instances: Sequence[TreeInstance]) -> int:
    """K_max = the longest reachable success across the data = max delta(root)."""
    ks = [int(i.delta_root) for i in instances if i.solvable()]
    if not ks:
        raise ValueError("no solvable instance: K_max is undefined.")
    return max(ks)


def load_instances(
    csv_paths: Sequence[str | Path],
    kind_of_data: str = "merged",
    verbose: bool = True,
) -> tuple[List[TreeInstance], List[Dict[str, object]]]:
    """Load many instances and GATE OUT the unsolvable ones.

    Nothing downstream may see an unsolvable tree: by the completeness
    proposition it produces exactly one transition (immediate doom) and teaches
    nothing about ranking, while diluting every coverage and regret aggregate.

    Returns (solvable, excluded_records). Callers MUST put `excluded_records`
    in the run manifest -- an instance silently dropped is an instance nobody
    knows was dropped.
    """
    loaded = [
        load_tree_instance(p, name=Path(p).parent.name, kind_of_data=kind_of_data)
        for p in csv_paths
    ]
    ok, bad = partition_solvable(loaded)
    excluded = [
        {
            "instance": i.name,
            "csv_path": i.csv_path,
            "reason": "delta(root)=inf: no goal reachable in the reconstructed tree",
            "n_states": i.n_states,
            "h_star_root": i.h_star[i.root_id],
        }
        for i in bad
    ]
    if verbose and excluded:
        print(
            f"[tree] EXCLUDED {len(excluded)} unsolvable instance(s) of "
            f"{len(loaded)}: {', '.join(e['instance'] for e in excluded)}"
        )
    if not ok:
        raise ValueError(
            f"every one of the {len(loaded)} instances is unsolvable "
            f"(delta(root)=inf); nothing to train on."
        )
    return ok, excluded


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
