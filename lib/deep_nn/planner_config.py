"""Deployment-config invariants that the offline MDP depends on.

The offline environment models **one expansion per ONNX call**. The deployed
planner only behaves that way when the driver loop's successor-accumulation
threshold is exactly 1. That threshold is not a constant — it is derived from
two C++ flags:

    SpaceSearcher.tpp:179     if (fringe_RL.size() >= RL_node_to_add || empty())
                                  push_vector(fringe_RL);
    Configuration.cpp:121-130 RL_node_to_add = int(F * RL_exploitation / 100.0)

With the C++ defaults (F=32, exploitation=70) RL_node_to_add is 22: the planner
expands ~22/b nodes per ONNX call, using *stale* ranks in between, and the
offline MDP would be a wrong model of it. Driving RL_node_to_add to 1 restores
one-expansion-per-call at every F, which is what makes the offline env faithful.

This module owns that mapping and the assertion. Nothing here changes the C++;
it only computes the flag values the planner must be launched with, and refuses
to let a model be exported against a config that would not hold the invariant.

See also `rl_heuristics`: refill must be RANDOM (`--RL_heuristics RNG`), the only
mode reproducible offline — see `env.py` for why.
"""

from __future__ import annotations

from math import ceil
from typing import Dict

# The only --RL_heuristics value whose refill is reproducible offline.
# MIN/MAX/AVG all map to RefillMode::HEURISTIC, whose reservoir order depends on
# Heuristics::RL_H folding *stored past RL ranks* (HeuristicsManager.tpp:139-164)
# — reproducing that offline means simulating the planner's whole heuristic
# history, which we explicitly do not do.
REQUIRED_RL_HEURISTICS = "rng"

# --RL_exploration must be 0: in RNG mode all refill is random anyway, so the
# exploration budget is inert, and 0 keeps `exploitation + exploration < 100`
# (ArgumentParser.cpp:113) satisfiable at every F.
REQUIRED_RL_EXPLORATION = 0


def rl_node_to_add(fringe_size: int, exploitation: int) -> int:
    """Mirror Configuration::get_succesors_to_analyze() exactly.

    C++ does float division then truncates toward zero:
        static_cast<int>(F * exploitation / 100.0)
    """
    return int(fringe_size * exploitation / 100.0)


def exploration_nodes(fringe_size: int, exploration: int) -> int:
    """Mirror Configuration::get_exploration_nodes() exactly."""
    return int(fringe_size * exploration / 100.0)


# RL_node_to_add == 1 requires an INTEGER exploitation e with 100 <= F*e <= 199
# (int() truncates). That has a solution only for F in [2, 199]:
#   F = 1    -> needs e = 100, rejected by ArgumentParser.cpp:113 (sum < 100)
#   F >= 200 -> even e = 1 gives int(F/100) >= 2
F_MIN = 2
F_MAX = 199


def exploitation_for(fringe_size: int) -> int:
    """Smallest --RL_exploitation making RL_node_to_add == 1 at this F.

    int(F*e/100) == 1  <=>  100 <= F*e <= 199, so the smallest integer is
    ceil(100/F). Picking the smallest keeps `e + exploration < 100` slack.
    """
    F = int(fringe_size)
    if not (F_MIN <= F <= F_MAX):
        raise ValueError(
            f"fringe_size={F} cannot give one expansion per ONNX call.\n"
            f"  RL_node_to_add = int(F * exploitation / 100) == 1 needs an integer\n"
            f"  exploitation with 100 <= F*exploitation <= 199, which only has a\n"
            f"  solution for F in [{F_MIN}, {F_MAX}].\n"
            + (
                f"  F=1 would need --RL_exploitation 100, which the C++ rejects\n"
                f"  (exploration + exploitation must be < 100, ArgumentParser.cpp:113).\n"
                if F < F_MIN else
                f"  At F={F} even --RL_exploitation 1 gives "
                f"int({F}/100) = {rl_node_to_add(F, 1)} >= 2, so the planner would\n"
                f"  rescore only every {rl_node_to_add(F, 1)} successors and the\n"
                f"  offline MDP would not model it.\n"
            )
        )
    e = ceil(100 / F)
    if e + REQUIRED_RL_EXPLORATION >= 100:
        raise ValueError(
            f"fringe_size={F} needs --RL_exploitation {e}, which violates the "
            f"C++ constraint exploration + exploitation < 100."
        )
    got = rl_node_to_add(F, e)
    if got != 1:
        raise AssertionError(
            f"internal: exploitation_for({F}) computed {e} but "
            f"RL_node_to_add == {got} != 1"
        )
    return e


def assert_one_expansion_per_call(fringe_size: int, exploitation: int) -> None:
    """Refuse to proceed unless the planner will rescore after every expansion.

    A silent mismatch here invalidates every number the run reports: the offline
    env would model one expansion per ONNX call while the planner does several,
    so regret, oracle agreement and the fidelity gate would all be measuring a
    planner that does not exist. Fail loudly, printing both numbers.
    """
    F = int(fringe_size)
    e = int(exploitation)
    n = rl_node_to_add(F, e)
    if n != 1:
        raise ValueError(
            "Planner config does not give one expansion per ONNX call.\n"
            f"  --RL_fringe_size   = {F}\n"
            f"  --RL_exploitation  = {e}\n"
            f"  => RL_node_to_add  = int({F} * {e} / 100) = {n}  (must be 1)\n"
            f"  The offline MDP models one expansion per ONNX call; at "
            f"RL_node_to_add={n} the planner would expand ~{n}/b nodes per call "
            f"using stale ranks in between, and every reported number would be "
            f"meaningless.\n"
            f"  Fix: launch with --RL_exploitation {exploitation_for(F)} at F={F}."
        )


def assert_rl_heuristics(value: str) -> None:
    """Only RNG (-> RefillMode::RANDOM) is reproducible offline."""
    v = str(value).lower()
    if v != REQUIRED_RL_HEURISTICS:
        raise ValueError(
            f"--rl-heuristics must be {REQUIRED_RL_HEURISTICS!r}, got {value!r}.\n"
            f"  MIN/MAX/AVG map to RefillMode::HEURISTIC (RL_BestFirst.h:25), whose\n"
            f"  reservoir order comes from Heuristics::RL_H folding stored past RL\n"
            f"  ranks (HeuristicsManager.tpp:139-164). Reproducing that offline\n"
            f"  requires simulating the planner's entire heuristic history, which\n"
            f"  this trainer does not do — so a model trained here would be\n"
            f"  selected against an environment that is not the deployed planner.\n"
            f"  Deploy with --RL_heuristics RNG."
        )


def planner_flags(fringe_size: int) -> Dict[str, object]:
    """The exact C++ flags a run at this F must be launched with.

    Recorded verbatim in the selection sidecar so a deployed ONNX always carries
    the config it was validated under.
    """
    F = int(fringe_size)
    e = exploitation_for(F)
    assert_one_expansion_per_call(F, e)
    return {
        "RL_fringe_size": F,
        "RL_exploitation": e,
        "RL_exploration": REQUIRED_RL_EXPLORATION,
        "RL_heuristics": "RNG",
        "RL_node_to_add": rl_node_to_add(F, e),
        "exploration_nodes": exploration_nodes(F, REQUIRED_RL_EXPLORATION),
    }


def planner_argv(fringe_size: int) -> list[str]:
    """`planner_flags` as a C++ argv fragment (fidelity gate + launcher)."""
    f = planner_flags(fringe_size)
    return [
        "--RL_fringe_size", str(f["RL_fringe_size"]),
        "--RL_exploitation", str(f["RL_exploitation"]),
        "--RL_exploration", str(f["RL_exploration"]),
        "--RL_heuristics", str(f["RL_heuristics"]),
    ]
