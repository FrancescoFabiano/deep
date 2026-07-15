"""The gatekeeper between generation and training. Nothing trains on data that
fails this.

THIS MODULE VALIDATES OUTPUT; IT DOES NOT GENERATE
It reads a tree and judges it. It owns NO generation parameter -- no depth, no
discard, no ceiling -- and deliberately does not know them. Those live in exactly one
place, `scripts/gnn_exp/create_all_training_data.py`, which the launcher drives. The
independence is the point: the generator picks the knobs, this asks only whether the
RESULT is faithful, so a wrong knob cannot also silence its own alarm.

THE DISCARD-ARTIFACT TEST, AUTOMATED
The instances are constructed with a known optimal plan length (`__pl_N`), and the
planner confirms it (strict BFS returns exactly N -- 26/26 CC rows in
combined_results/<batch>/CC/bfs/*_BFS_strict.csv). So a faithful table MUST satisfy
`delta_root == N`. On the shipped `discard_factor 0.4` data it did not:

    CC_2_2_3__pl_4   delta_root 10  vs true optimal 4
    CC_2_3_4__pl_7   delta_root 34  vs true optimal 7

That is the biased discard having deleted the shallow goals. This check turns that
diagnosis into an automatic exclusion so the artifact can never silently return.

It earns its keep beyond the discard, too: at discard 0 it still caught
`CC_2_3_4__pl_7` at depth 25 with delta_root 22 vs optimal 7 -- there the DFS was
truncated by the VISIT ceiling and never reached the shallow goal. Same symptom, a
different cause, and the check did not need to know which.

THREE CRITERIA
  1. delta_root == expected optimal      -> the solution path is PRESENT
  2. poisoned fraction below threshold   -> the generator's ceiling did not bite.
                                            Past it, non-goals are dropped AND
                                            parents inherit 1e6 = "unreachable".
                                            (Two ceilings can do this; the binding
                                            one is the 100k VISIT count. Which one
                                            fired is the generator's problem, not
                                            this module's -- 1e6 is 1e6.)
  3. BFS reaches >= min_expansions       -> the fidelity gate can actually SCORE it;
                                            at 7 expansions a +-1 difference is 14%,
                                            so a fractional tolerance is meaningless

(3) is a FLAG, not an exclusion: trivial instances stay in the pool for coverage,
they just cannot discriminate for fidelity.
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
    """`CC_2_3_4__pl_7` -> 7. The instances are CONSTRUCTED with a known optimal
    plan length, and the planner confirms it: strict BFS returns exactly `pl_N` on
    every CC instance measured (combined_results/<batch>/CC/bfs/*_BFS_strict.csv,
    26/26 rows). `--strong_equality` does not move it either -- verified pl_3..pl_6
    on CC_2_3_4, where the flag demonstrably changes the search (pl_5: 277 nodes
    weak vs 274 strict) without changing the answer.

    This is what lets the check know its target WITHOUT probing the planner, and
    without knowing anything about how the tree was generated.
    """
    m = re.search(r"__pl_(\d+)", str(instance_name))
    return int(m.group(1)) if m else None


# Past a generation ceiling the generator drops non-goals and poisons their parents
# to 1e6. A little is tolerable (0.1% measured on the pl_4 regen); a lot means the
# tree is truncated and the instance is not faithful.
MAX_POISONED_FRAC = 0.01
# Below this the fidelity gate cannot discriminate tree-replay drift from a real
# modelling error.
MIN_EXPANSIONS_FOR_FIDELITY = 20


@dataclass
class InstanceVerdict:
    instance: str
    faithful: bool
    reasons: List[str] = field(default_factory=list)
    delta_root: Optional[float] = None
    expected_optimal: Optional[int] = None
    poisoned_frac: float = 0.0
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
    max_poisoned_frac: float = MAX_POISONED_FRAC,
    min_expansions: int = MIN_EXPANSIONS_FOR_FIDELITY,
) -> InstanceVerdict:
    reach = inst._reachable()
    n = max(1, len(reach))
    poisoned = sum(1 for v in reach if inst.h_star[v] >= UNREACHABLE_DISTANCE)
    sterile = sum(1 for v in reach if inst.delta[v] == INF_DELTA)
    opt = expected_optimal(inst.name)
    v = InstanceVerdict(
        instance=inst.name,
        faithful=True,
        delta_root=(None if inst.delta_root == INF_DELTA else float(inst.delta_root)),
        expected_optimal=opt,
        poisoned_frac=poisoned / n,
        sterile_frac=sterile / n,
        n_states=len(reach),
    )

    if not inst.solvable():
        v.faithful = False
        v.reasons.append("delta(root)=inf: no goal reachable in the tree")
        return v

    # 1. THE DISCARD-ARTIFACT TEST
    if opt is None:
        v.reasons.append("no __pl_N in the name; cannot verify the optimal is present")
    elif int(inst.delta_root) != opt:
        v.faithful = False
        v.reasons.append(
            f"delta_root={inst.delta_root:.0f} != known optimal {opt}: the solution "
            f"path is ABSENT (biased discard deletes shallow goals) or the depth "
            f"bound cut it"
        )

    # 2. did the ceiling bite?
    if v.poisoned_frac > max_poisoned_frac:
        v.faithful = False
        v.reasons.append(
            f"poisoned fraction {v.poisoned_frac:.3f} > {max_poisoned_frac}: a "
            f"generation ceiling truncated the tree and marked live nodes "
            f"unreachable (h*=1e6)"
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


def build_faithful_pool(
    instances: Sequence[TreeInstance],
    out_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Partition into faithful / excluded, with a reason for every exclusion.

    Writes `faithful_pool.json`. Nothing downstream may train on an instance absent
    from `faithful`.
    """
    verdicts = [check_instance(i) for i in instances]
    faithful = [v for v in verdicts if v.faithful]
    excluded = [v for v in verdicts if not v.faithful]
    doc = {
        "n_total": len(verdicts),
        "n_faithful": len(faithful),
        "n_excluded": len(excluded),
        "n_usable_for_fidelity": sum(1 for v in faithful if v.usable_for_fidelity),
        "criteria": {
            "delta_root_equals_known_optimal": True,
            "max_poisoned_frac": MAX_POISONED_FRAC,
            "min_expansions_for_fidelity": MIN_EXPANSIONS_FOR_FIDELITY,
        },
        "faithful": [asdict(v) for v in faithful],
        "excluded": [asdict(v) for v in excluded],
    }
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1))
    if verbose:
        print(f"[faithfulness] {len(faithful)}/{len(verdicts)} faithful; "
              f"{doc['n_usable_for_fidelity']} usable for the fidelity gate")
        for v in excluded:
            print(f"  EXCLUDED {v.instance}: {'; '.join(v.reasons)}")
    return doc


def faithful_names(pool: Dict[str, object]) -> List[str]:
    return [v["instance"] for v in pool["faithful"]]


def fidelity_names(pool: Dict[str, object]) -> List[str]:
    """The instances the fidelity gate may actually score."""
    return [v["instance"] for v in pool["faithful"] if v["usable_for_fidelity"]]
