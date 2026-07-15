"""The launcher <-> entry-point seam. A TEST, not a read.

final_launcher.sh is the durable interface: it owns cell iteration (one cell per
invocation, via env vars) and calls train_models.py, which invokes offline_main.py
per (domain, seed) forwarding unknown args verbatim.

THE INVARIANT: the two must agree on EXACTLY one flag set.
  - every flag the launcher sends, offline_main.py must ACCEPT
  - nothing offline_main.py accepts may be dead surface
A launcher passing a flag the entry point ignores is the same latent bug as one it
rejects -- it just fails silently instead of loudly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
OFFLINE_MAIN = HERE / "offline_main.py"


def _parser():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_om", OFFLINE_MAIN)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_om"] = m
    spec.loader.exec_module(m)
    return m.build_parser()


def accepted_flags() -> set[str]:
    out = set()
    for a in _parser()._actions:
        out.update(a.option_strings)
    return out


# ---- what final_launcher.sh sends, AFTER the reconciliation ----
# TRAIN_FLAG   : --no_goal        (separated cells only)
# TRAIN_EXTRA  : --context-mode X [--cql-alpha A]
# plus         : --fringe-sizes / --model / --batch-size / --frames / --n-checkpoints
# train_models.py adds: --seed / --dir-save-model / --train-csv / --test-csv /
#                       --kind-of-data separated
LAUNCHER_SENDS = [
    "--fringe-sizes", "--model", "--no_goal", "--batch-size", "--frames",
    "--n-checkpoints", "--context-mode", "--cql-alpha",
]
TRAIN_MODELS_SENDS = [
    "--seed", "--dir-save-model", "--train-csv", "--test-csv", "--kind-of-data",
    "--model", "--fringe-sizes",
]

# Flags the launcher used to send that the trainer no longer consumes. It must STOP
# sending these; each is dead for a stated reason.
RETIRED = {
    "--random-pct": "superseded by --behaviour-policies",
    "--target-tau": "Polyak; at gamma=1 the trainer uses hard target sync + the "
                    "|Q| > 3x cap alarm",
    "--lr-schedule": "gamma=0.99 stability machinery; the trainer honours no schedule",
    "--lr-min": "same",
    "--no-pad-closed": "padding the beam with CLOSED states gives the model actions "
                       "the planner can never take -- a train/deploy mismatch",
    "--stratified-replay": "deleted with the regime machinery",
    "--epsilon-schedule": "offline: there is no exploration",
}


@pytest.mark.parametrize("flag", LAUNCHER_SENDS)
def test_offline_main_accepts_every_flag_the_launcher_sends(flag):
    assert flag in accepted_flags(), (
        f"final_launcher.sh sends {flag} but offline_main.py rejects it -- the "
        f"pipeline breaks loudly at step 2"
    )


@pytest.mark.parametrize("flag", TRAIN_MODELS_SENDS)
def test_offline_main_accepts_every_flag_train_models_sends(flag):
    assert flag in accepted_flags(), f"train_models.py sends {flag}; offline_main rejects it"


@pytest.mark.parametrize("flag", sorted(RETIRED))
def test_retired_flags_are_gone_from_the_entry_point(flag):
    """If offline_main still ACCEPTED these it would be dead surface -- the silent
    half of the bug."""
    assert flag not in accepted_flags(), f"{flag} is dead surface: {RETIRED[flag]}"


def test_the_no_goal_alias_is_kept_for_compatibility():
    """The ONE alias: the launcher's existing spelling of --kind-of-data separated.
    Changing it would ripple into how the launcher is invoked."""
    a = _parser().parse_args(["--train-csv", "x.csv", "--dir-save-model", "d", "--no_goal"])
    assert a.no_goal is True
    a2 = _parser().parse_args(["--train-csv", "x.csv", "--dir-save-model", "d"])
    assert a2.no_goal is False and a2.kind_of_data == "merged"


def test_the_launchers_full_separated_invocation_is_accepted():
    """The exact argv for MODE=separated STRICT=yes ALGO=cql CTX=self_attention,
    after the reconciliation. Must parse without error."""
    a = _parser().parse_args([
        "--seed", "0", "--dir-save-model", "/tmp/d/seed0",
        "--train-csv", "/tmp/a.csv", "/tmp/b.csv",
        "--kind-of-data", "separated", "--model", "cql",
        "--no_goal", "--batch-size", "64", "--frames", "100000",
        "--n-checkpoints", "20", "--context-mode", "self_attention",
        "--cql-alpha", "1.0", "--fringe-sizes", "4", "8", "16", "32",
    ])
    assert a.model == "cql" and a.no_goal and a.context_mode == "self_attention"
    assert a.fringe_sizes == [4, 8, 16, 32] and a.cql_alpha == 1.0


def test_the_launchers_merged_invocation_is_accepted():
    a = _parser().parse_args([
        "--seed", "0", "--dir-save-model", "/tmp/d/seed0", "--train-csv", "/tmp/a.csv",
        "--model", "dqn", "--batch-size", "64", "--frames", "100000",
        "--n-checkpoints", "20", "--context-mode", "mean_pool",
        "--fringe-sizes", "4",
    ])
    assert not a.no_goal and a.kind_of_data == "merged"


def test_help_exits_clean():
    r = subprocess.run([sys.executable, str(OFFLINE_MAIN), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
