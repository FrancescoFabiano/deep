"""Generation config for FAITHFUL trees. No discarding; depth bounds the tree.

WHY THIS EXISTS
The shipped tables used `--dataset_discard_factor 0.4`. The discard is BIASED
(`TrainingDataset.tpp:714-730`): its probability rises with depth and gains +0.2
immediately after a goal is found, so it preferentially deletes the SHALLOW GOALS
that make an instance easy. Discarded states are still written to the CSV as
childless leaves, indistinguishable from genuine dead ends. Measured on
`CC_2_2_3__pl_4` (planner BFS: true optimal = 4):

    discard 0.4:  n=2759   delta_root=10  sterile 30.0%    <- optimal path ABSENT
    discard 0  :  n=50146  delta_root= 4  sterile  2.9%    <- optimal path present

So the env was not modelling a noisier planner — it was modelling a HARDER problem
with the easy solutions cut out.

THE FIX IS DEPTH, NOT A BIGGER CEILING
`--dataset_max_creation` is a HARD STOP that POISONS (`TrainingDataset.tpp:677-684`):
past the cap, non-goals are dropped from the CSV entirely AND their parents inherit
`m_failed_state` (1e6), marking live nodes unreachable. Raising it just moves the
poisoning. Instead, bound the DEPTH so full enumeration fits under the cap:

    a depth bound that CONTAINS the optimal keeps the whole solution path while
    cutting the tree where it no longer matters

That is faithful (the optimal is present, nothing is silently deleted) AND
tractable (the node count stays under the ceiling, so the poisoning never bites).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

# Depth is DOMAIN-DEPENDENT. CC optimals are ~4-7, so 25 contains them with wide
# margin; SC/SCRich genuinely need the depth.
DEPTH_BY_DOMAIN: Dict[str, int] = {
    "CC": 25,
    "SC": 40,
    "SCRich": 40,
}
DEFAULT_DEPTH = 25

# Fixed by directive: do NOT sweep the ceiling. Depth is what keeps generation
# under it.
MAX_CREATION = 50000
DISCARD_FACTOR = 0          # the whole point: no biased deletion


def domain_of(instance_or_domain: str) -> str:
    """`SC_R_10_10__pl_10` -> `SCRich`; `CC_2_3_4__pl_7` -> `CC`.

    Keys off the instance prefix so one launch can cover a mixed pool.
    """
    s = str(instance_or_domain)
    if s in DEPTH_BY_DOMAIN:
        return s
    if s.startswith("SC_R"):
        return "SCRich"
    if s.startswith("SC_"):
        return "SC"
    if s.startswith("CC_"):
        return "CC"
    return s.split("_")[0]


def depth_for(instance_or_domain: str) -> int:
    """Per-domain depth lookup. Logged per instance in the manifest."""
    return DEPTH_BY_DOMAIN.get(domain_of(instance_or_domain), DEFAULT_DEPTH)


def expected_optimal(instance_name: str) -> Optional[int]:
    """`CC_2_3_4__pl_7` -> 7. The instances are CONSTRUCTED with a known optimal
    plan length, and the planner confirms it: BFS returns exactly `pl_N` on
    CC_2_2_3__pl_4 (4), __pl_6 (6), CC_2_3_4__pl_7 (7), CC_3_2_3__pl_5 (5).

    This is what makes the faithfulness check possible without a BFS probe.
    """
    m = re.search(r"__pl_(\d+)", str(instance_name))
    return int(m.group(1)) if m else None


def assert_depth_contains_optimal(instance_name: str) -> None:
    """A depth bound is only faithful if it CONTAINS the optimal path."""
    opt = expected_optimal(instance_name)
    d = depth_for(instance_name)
    if opt is not None and d < opt:
        raise ValueError(
            f"{instance_name}: --dataset_depth {d} is BELOW the known optimal plan "
            f"length {opt}. The solution path would be cut off and delta_root would "
            f"be inf — the exact unfaithfulness this config exists to remove."
        )


def generation_argv(
    problem_file: str | Path,
    instance_name: Optional[str] = None,
    *,
    separated: bool = True,
    strong_equality: bool = True,
    seed: int = 42,
    dataset_type: str = "HASHED",
) -> List[str]:
    """The exact `deep` argv for a faithful table.

    `dataset_type` is passed through opaquely: HASHED today, BITMASK the moment
    `FringeEvalRL` implements it — no change here or downstream.
    """
    name = instance_name or Path(problem_file).stem
    assert_depth_contains_optimal(name)
    argv = [
        str(problem_file), "-b", "-c", "--dataset",
        "--dataset_depth", str(depth_for(name)),
        "--dataset_discard_factor", str(DISCARD_FACTOR),
        "--dataset_seed", str(seed),
        "--dataset_max_creation", str(MAX_CREATION),
        "--dataset_type", dataset_type,
    ]
    if separated:
        argv.append("--dataset_separated")
    if strong_equality:
        argv.append("--strong_equality")
    return argv


def generation_manifest(instance_name: str, **extra) -> Dict[str, object]:
    """Recorded per instance so a table's provenance is never in doubt."""
    d = {
        "instance": instance_name,
        "domain": domain_of(instance_name),
        "dataset_depth": depth_for(instance_name),
        "dataset_discard_factor": DISCARD_FACTOR,
        "dataset_max_creation": MAX_CREATION,
        "expected_optimal_plan_length": expected_optimal(instance_name),
    }
    d.update(extra)
    return d
