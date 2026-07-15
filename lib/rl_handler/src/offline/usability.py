"""The gatekeeper between generation and training. Nothing trains on data that
fails this.

THIS MODULE VALIDATES OUTPUT; IT DOES NOT GENERATE
It reads a tree and judges it. It owns NO generation parameter -- no depth, no
discard, no ceiling -- and deliberately does not know them. Those live in exactly one
place, `scripts/gnn_exp/create_all_training_data.py`, which the launcher drives. The
independence is the point: the generator picks the knobs, this asks only whether the
RESULT is usable, so a wrong knob cannot also silence its own alarm.

WHAT THIS GATE CERTIFIES -- AND WHAT IT DOES NOT
It certifies USABILITY, not optimality:
  * the root reaches a goal inside the tree (delta_root finite -- not DOOM);
  * the tree is non-trivial (enough states for the MDP to be worth training on);
  * it is scorable (BFS reaches >= min_expansions -- a FLAG, not an exclusion).

It does NOT certify that the tree's distances are the planner's optimal plan length.
It cannot, and neither can anything else downstream -- see DISTANCE SEMANTICS below.

WHY THE `delta_root == pl_N` CRITERION WAS RETIRED (2026-07-15)
It tested DFS luck, not the tree. The generator is a depth-bounded DFS whose memo
(`m_visited_states`, keyed by STATE ALONE) locks a state at its first-discovery depth
and returns that stale verdict when the state is later reached shallower with room to
expand. First discovery is systematically deep, so the recorded structure is a DFS
spanning tree of DISCOVERY depths. `delta_root` is therefore >= the true optimal,
equal only where the DFS happened to sample a shortest path first.

Measured on CC_2_2_3__pl_4 (true optimal 4; depth 40, discard 0, identical flags --
seed alone varied):

    seed 42 -> delta_root 14     seed 43 -> delta_root 6     seed 44 -> delta_root 7

A criterion whose verdict swings 14/6/7 on a quantity whose truth is a fixed 4 is
measuring the RNG, not the data. The instance it once stamped "faithful" (delta_root
4) simply got lucky. Root cause is in frozen C++ (`TrainingDataset.tpp`: depth-blind
memo, plus a filename minted per visit rather than per state, which collapses the
recorded DAG into a spanning tree); a real fix needs a shortest-path search, which is
a different generator. So the gate stops asking the generator for something its
algorithm cannot deliver.

`poisoned_frac` is likewise NOT a gate, only a diagnostic. `h* >= 1e6` conflates two
unrelated causes: a real generation ceiling truncating the tree, AND an ordinary
non-goal leaf sitting at the depth bound (both are scored 1e6). At depth 9 on
CC_2_3_4__pl_7, 145 of 154 poisoned rows were plain frontier leaves and no ceiling
ever fired -- excluding on it rejects healthy shallow trees.

DISTANCE SEMANTICS (state this before making any claim)
`delta` is the exact shortest-path distance to the nearest goal WITHIN the generated
tree, which is the MDP's action space. It is NOT the planner's optimal plan length for
the instance -- the tree is a seed-dependent DFS sample of the true state space, and
root-to-goal distance in the tree is an upper bound on the true optimal, tight only
where the DFS sampled a shortest path. Labels are exact for the tree the agent
navigates; they do not certify instance-level optimality. A comparison between arms
run on the SAME tree with the SAME labels is valid regardless -- relative ranking never
needed absolute optimality.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .tree import INF_DELTA, UNREACHABLE_DISTANCE, TreeInstance


def expected_optimal(instance_name: str) -> Optional[int]:
    """`CC_2_3_4__pl_7` -> 7. The instance's CONSTRUCTED optimal plan length, which
    the planner confirms (combined_results/<batch>/CC/bfs/*_BFS_strict.csv:
    CC_2_3_4__pl_7 -> 7, CC_2_2_3__pl_4 -> 4).

    Reported as a DIAGNOSTIC only. It is not a gate: the generated tree's delta_root
    is a DFS-discovery upper bound on it (see the module docstring), so a mismatch
    says the DFS did not sample a shortest path -- not that the tree is unusable.
    """
    m = re.search(r"__pl_(\d+)", str(instance_name))
    return int(m.group(1)) if m else None


# Below this the tree is too small for the MDP to be worth training on. The smallest
# real tree observed is 133 states (CC_2_3_4__pl_7 at depth 9); degenerate cases are
# far smaller (that instance at depth 7 yields 23 states and no goal at all, and is
# already excluded by the DOOM check).
MIN_STATES = 50
# Below this the fidelity gate cannot discriminate tree-replay drift from a real
# modelling error: at 7 expansions a +-1 difference is 14%, so a fractional tolerance
# is meaningless.
MIN_EXPANSIONS_FOR_FIDELITY = 20
# Reported, never gated on -- see the module docstring.
POISONED_FRAC_NOTE_ABOVE = 0.01


@dataclass
class InstanceVerdict:
    instance: str
    usable: bool
    reasons: List[str] = field(default_factory=list)
    delta_root: Optional[float] = None
    expected_optimal: Optional[int] = None
    delta_root_matches_optimal: Optional[bool] = None  # diagnostic only
    poisoned_frac: float = 0.0  # diagnostic only
    sterile_frac: float = 0.0
    n_states: int = 0
    bfs_expansions: Optional[int] = None
    usable_for_fidelity: bool = False


def bfs_expansions(instance: TreeInstance, cap: int = 200000) -> Optional[int]:
    """Expansions an uncapped BFS needs to GENERATE a goal, in the tree.

    Goal test at generation, mirroring `SpaceSearcher.tpp:162`.
    """
    visited = {instance.root_id}
    q: deque[int] = deque()
    n = 1
    for c in instance.children[instance.root_id]:
        if c in visited:
            continue
        visited.add(c)
        if instance.is_goal[c]:
            return n
        q.append(c)
    while q and n < cap:
        v = q.popleft()
        n += 1
        for c in instance.children[v]:
            if c in visited:
                continue
            visited.add(c)
            if instance.is_goal[c]:
                return n
            q.append(c)
    return None


def check_instance(
    inst: TreeInstance,
    *,
    min_states: int = MIN_STATES,
    min_expansions: int = MIN_EXPANSIONS_FOR_FIDELITY,
) -> InstanceVerdict:
    """Is this tree USABLE for training? (Not: are its distances optimal.)"""
    reach = inst._reachable()
    n = max(1, len(reach))
    poisoned = sum(1 for v in reach if inst.h_star[v] >= UNREACHABLE_DISTANCE)
    sterile = sum(1 for v in reach if inst.delta[v] == INF_DELTA)
    opt = expected_optimal(inst.name)
    v = InstanceVerdict(
        instance=inst.name,
        usable=True,
        delta_root=(None if inst.delta_root == INF_DELTA else float(inst.delta_root)),
        expected_optimal=opt,
        poisoned_frac=poisoned / n,
        sterile_frac=sterile / n,
        n_states=len(reach),
    )

    # 1. THE COMPLETENESS/USABILITY CONDITION: the root must reach a goal.
    #    delta(root)=inf is DOOM -- there is nothing for the agent to find.
    if not inst.solvable():
        v.usable = False
        v.reasons.append("delta(root)=inf: no goal reachable in the tree")
        return v

    # 2. non-trivial enough to be worth training on
    if v.n_states < min_states:
        v.usable = False
        v.reasons.append(
            f"only {v.n_states} states (< {min_states}): tree too small to train on"
        )

    # DIAGNOSTIC (never a gate): how far the DFS's sampled distance sits above the
    # instance's known optimal. A mismatch means the DFS did not sample a shortest
    # path; it does NOT mean the tree is unusable. See the module docstring.
    if opt is not None and v.delta_root is not None:
        v.delta_root_matches_optimal = int(inst.delta_root) == opt
        if not v.delta_root_matches_optimal:
            v.reasons.append(
                f"[diagnostic] delta_root={inst.delta_root:.0f} vs known optimal "
                f"{opt}: the DFS did not sample a shortest path (expected; not an "
                f"exclusion)"
            )
    if v.poisoned_frac > POISONED_FRAC_NOTE_ABOVE:
        v.reasons.append(
            f"[diagnostic] poisoned fraction {v.poisoned_frac:.3f}: states with "
            f"h*=1e6 -- either a generation ceiling truncated the tree or these are "
            f"ordinary non-goal leaves at the depth bound (not an exclusion)"
        )

    # 3. can it discriminate for the fidelity gate? (a FLAG, not an exclusion)
    v.bfs_expansions = bfs_expansions(inst)
    v.usable_for_fidelity = bool(v.bfs_expansions and v.bfs_expansions >= min_expansions)
    if not v.usable_for_fidelity:
        v.reasons.append(
            f"BFS reaches only {v.bfs_expansions} expansions (< {min_expansions}): "
            f"too small for the fidelity gate to score; kept for coverage"
        )
    return v


def build_usable_pool(
    instances: Sequence[TreeInstance],
    out_path: Optional[str | Path] = None,
    verbose: bool = True,
    *,
    min_states: int = MIN_STATES,
    min_expansions: int = MIN_EXPANSIONS_FOR_FIDELITY,
) -> Dict[str, object]:
    """Partition into usable / excluded, with a reason for every exclusion.

    Writes `usable_pool.json`. Nothing downstream may train on an instance absent
    from `usable`.
    """
    verdicts = [check_instance(i, min_states=min_states, min_expansions=min_expansions)
                for i in instances]
    usable = [v for v in verdicts if v.usable]
    excluded = [v for v in verdicts if not v.usable]
    doc = {
        "n_total": len(verdicts),
        "n_usable": len(usable),
        "n_excluded": len(excluded),
        "n_usable_for_fidelity": sum(1 for v in usable if v.usable_for_fidelity),
        "criteria": {
            "delta_root_finite": True,
            "min_states": min_states,
            "min_expansions_for_fidelity": min_expansions,
            "delta_root_equals_known_optimal": (
                "DIAGNOSTIC ONLY -- retired as a gate: it tested DFS luck, not the "
                "tree (delta_root swung 14/6/7 across seeds on a fixed optimal of 4)"
            ),
            "poisoned_frac": (
                "DIAGNOSTIC ONLY -- conflates ceiling truncation with ordinary "
                "depth-bound leaves"
            ),
        },
        "distance_semantics": (
            "delta is the exact shortest-path distance to the nearest goal WITHIN the "
            "generated tree, which is the MDP's action space. It is NOT the planner's "
            "optimal plan length: the tree is a seed-dependent DFS sample of the true "
            "state space, so root-to-goal distance in the tree is an upper bound on "
            "the true optimal, tight only where the DFS sampled a shortest path."
        ),
        "usable": [asdict(v) for v in usable],
        "excluded": [asdict(v) for v in excluded],
    }
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1))
    if verbose:
        print(f"[usability] {len(usable)}/{len(verdicts)} usable; "
              f"{doc['n_usable_for_fidelity']} usable for the fidelity gate")
        for v in excluded:
            print(f"  EXCLUDED {v.instance}: {'; '.join(v.reasons)}")
    return doc


def usable_names(pool: Dict[str, object]) -> List[str]:
    return [v["instance"] for v in pool["usable"]]


def fidelity_names(pool: Dict[str, object]) -> List[str]:
    """The instances the fidelity gate may actually score."""
    return [v["instance"] for v in pool["usable"] if v["usable_for_fidelity"]]
