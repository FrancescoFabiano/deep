"""The depth map is AUTHORITATIVE, and it lives in the GENERATOR.

`scripts/gnn_exp/create_all_training_data.py` is the ONE place generation parameters
live; `final_launcher.sh` passes DEPTH_MAP to it explicitly. rl_handler holds no
second copy (that is what these tests, and test_faithfulness's architecture test,
protect).

NOTE depth is necessary but NOT sufficient to bound the tree. CC_2_3_4 at depth 25
still estimates 1.15e40 nodes -> SPARSE DFS -> the DFS burns the 100k VISIT ceiling
(--dataset_max_generation, which no caller passes) in the deep region and records a
goal at depth 22 while the true optimal at 7 is never reached. See that module's
docstring; the faithfulness gate is what catches it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _cad():
    """Load scripts/gnn_exp/create_all_training_data.py as a module."""
    p = Path(__file__).resolve().parents[3] / "scripts/gnn_exp/create_all_training_data.py"
    spec = importlib.util.spec_from_file_location("_cad", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    def __init__(self, depth_map="", depth=40):
        self.depth_map = depth_map
        self.depth = depth


def test_depth_map_is_used_when_present():
    m = _cad()
    a = _Args("CC:25,SC:40,SCRich:40")
    assert m._depth_for("CC", a) == 25
    assert m._depth_for("SC", a) == 40
    assert m._depth_for("SCRich", a) == 40


def test_a_domain_absent_from_the_map_FAILS_LOUDLY_and_does_not_default_to_40():
    """THE test. 'A domain absent from the map silently gets 40' is the exact bug
    shape this rework existed to kill: a default that is wrong for the new regime,
    applied invisibly, producing self-consistent data nobody flagged. 40 is the
    UNFAITHFUL setting for CC -- it is what blew past the ceiling and poisoned."""
    m = _cad()
    a = _Args("CC:25,SC:40")          # Grapevine absent
    with pytest.raises(SystemExit) as e:
        m._depth_for("Grapevine", a)
    msg = str(e.value)
    assert "no depth in --depth-map" in msg
    assert "refusing to guess" in msg
    assert "UNFAITHFUL" in msg


def test_no_map_at_all_keeps_the_legacy_single_depth_behaviour():
    """create_all_training_data.py is SHARED with gnn_exp; a caller that passes no
    map must behave exactly as before."""
    m = _cad()
    assert m._depth_for("Anything", _Args("", depth=25)) == 25


def test_parse_depth_map():
    m = _cad()
    assert m._parse_depth_map("CC:25,SC:40") == {"CC": 25, "SC": 40}
    assert m._parse_depth_map("") == {}
    assert m._parse_depth_map(" CC : 25 ") == {"CC": 25}
