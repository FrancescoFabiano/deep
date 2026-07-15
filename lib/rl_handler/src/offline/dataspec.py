"""The data fingerprint that guards `DATA_SOURCE` reuse.

WHY THIS IS THE MOST IMPORTANT GUARD IN THE PIPELINE

`final_launcher.sh` lets a run symlink another batch's `training_data` instead of
regenerating (it saves hours), and checks compatibility against a `.dataspec`
string. The original fingerprint was:

    mode=<merged|separated>;strict=<yes|no>;depth=<N>;max_creation=<N>

It does NOT mention the discard factor. So a `discard=0` run could symlink a
`discard=0.4` batch, PASS the reuse check, train on trees whose shallow goals were
deleted, and report clean, self-consistent numbers. That is the worst failure mode
available here: silent, invisible, and it reintroduces the exact artifact that cost
two turns to find.

Measured, `CC_2_2_3__pl_4` (planner BFS true optimal = 4):
    discard 0.4 -> delta_root 10, sterile 30.0%   (the optimal path is ABSENT)
    discard 0   -> delta_root  4, sterile  2.9%

The depth is in for the same reason: a CC-depth-25 run must not reuse CC-depth-40
data. Depth is PER DOMAIN, so a single `depth=40` scalar can no longer describe a
mixed batch — the fingerprint carries the whole map.

THIS MODULE OWNS NO PARAMETER, IT ONLY FINGERPRINTS ONE
Every value here is passed IN by the caller that actually generated the data
(`final_launcher.sh`, whose `DEPTH_MAP` feeds both this fingerprint and
`create_all_training_data.py --depth-map`). There are deliberately NO defaults: a
default here would be a second opinion about how data was generated, and when it
drifted from the generator's real setting this guard would happily bless data it had
mis-described. Describing data you did not generate is only safe if you are told.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Union


def _render_depth_map(depth_map: Union[str, Mapping[str, int]]) -> str:
    """Accepts the launcher's `CC:25,SC:40` string verbatim, or a mapping."""
    if isinstance(depth_map, str):
        return depth_map.strip()
    return ",".join(f"{k}:{v}" for k, v in sorted(depth_map.items()))


def make_dataspec(
    mode: str,
    strict: str,
    depth_map: Union[str, Mapping[str, int]],
    *,
    discard_factor: float,
    max_creation: int,
    max_generation: int,
) -> str:
    """The fingerprint. Any run reusing this data must match it EXACTLY.

    `mode`           : merged | separated   (the state representation)
    `strict`         : yes | no             (--strong_equality)
    `depth_map`      : `CC:25,SC:40` or {"CC": 25, "SC": 40} -- what the GENERATOR used
    `max_creation`   : the WRITE ceiling (--dataset_max_creation)
    `max_generation` : the VISIT ceiling (--dataset_max_generation) -- the BINDING one

    BOTH ceilings are fingerprinted. The visit ceiling decides WHICH goals the DFS
    ever reaches before it starts poisoning, so two batches generated at different
    visit budgets are genuinely different data even at identical depth and discard.
    It is also the one that was invisible: it sat at its 100000 C++ default while
    only max_creation was threaded through.
    """
    # "," inside the map, NOT ";" -- ";" is the FIELD separator, and using it here
    # made parse_dataspec truncate `depth_map=CC:25;SC:40` to `CC:25`, so a CC-only
    # run compared EQUAL to a CC+SC batch and would have reused it. Caught by
    # test_a_mixed_batch_spec_differs_from_a_cc_only_one.
    dm = _render_depth_map(depth_map)
    return (f"mode={mode};strict={strict};discard={discard_factor};"
            f"depth_map={dm};max_creation={max_creation};"
            f"max_generation={max_generation}")


def parse_dataspec(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in spec.strip().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def compatible(have: str, want: str) -> tuple[bool, Optional[str]]:
    """(ok, why_not). Every field must match; a missing field is a mismatch.

    An OLD spec (no `discard=` key) is INCOMPATIBLE with any new one by
    construction — that is the point. Old batches were generated at discard 0.4,
    and letting them through on "the key is absent, assume it's fine" is exactly
    the silent path this module exists to close.
    """
    h, w = parse_dataspec(have), parse_dataspec(want)
    if "discard" not in h:
        return False, (
            "the source .dataspec predates discard fingerprinting, so it was almost "
            "certainly generated at --dataset_discard_factor 0.4, whose BIASED "
            "discard deletes the shallow goals that make an instance easy "
            "(delta_root 10 vs a true optimal of 4 on CC_2_2_3__pl_4). Regenerate; "
            "do not reuse."
        )
    for k in sorted(set(h) | set(w)):
        if h.get(k) != w.get(k):
            return False, f"{k}: source={h.get(k)!r} wanted={w.get(k)!r}"
    return True, None
