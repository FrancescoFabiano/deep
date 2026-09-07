"""The generator's multi-strategy flag lives in scripts/gnn_exp (the ONE place
generation parameters live). rl_handler only reads what it produced."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/gnn_exp"


def _load(name):
    p = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_generation_list_spellings_and_all():
    m = _load("create_all_training_data.py")
    assert m.parse_generation_list(["BFS", "hfs"]) == ["BFS", "HFS"]
    assert m.parse_generation_list(["bfs,s-dfs"]) == ["BFS", "S_DFS"]
    assert m.parse_generation_list(["all"]) == ["BFS", "DFS", "S_DFS", "HFS"]
    assert m.parse_generation_list(None) == [] and m.parse_generation_list([]) == []
    with pytest.raises(argparse.ArgumentTypeError):
        m.parse_generation_list(["random"])


def test_strategy_target_folder_nests_only_when_a_strategy_is_passed():
    """Nested <dataset>/<STRAT>/ when the flag is given (strategies coexist); the
    legacy flat layout when it is not, so the gnn_exp pipeline -- which never passes
    the flag -- keeps the layout it expects."""
    m = _load("create_training_data.py")
    assert m.strategy_target_folder("/x/training_data", "BFS") == "/x/training_data/BFS"
    assert m.strategy_target_folder("/x/training_data", None) == "/x/training_data"


def test_seedless_strategies_are_bfs_and_hfs():
    """BFS and HFS use no RNG: retrying them with another seed reproduces the same
    tree byte for byte (CC_2_3_4__pl_7 BFS ~3 min + >1 GB per attempt)."""
    m = _load("create_training_data.py")
    assert set(m.SEEDLESS_GENERATIONS) == {"BFS", "HFS"}
    assert set(m.GENERATION_CHOICES) == {"BFS", "DFS", "S_DFS", "HFS"}


def test_the_strategy_dir_names_match_rl_handlers():
    """The generator writes <STRAT>/ with the C++ enum spelling and rl_handler reads
    it back: the two lists must be one list."""
    from src.offline.strategies import STRATEGIES, dir_name
    m = _load("create_all_training_data.py")
    assert [dir_name(s) for s in STRATEGIES] == list(m.GENERATION_CHOICES)


def test_discard_default_is_per_strategy():
    """0.6 for S_DFS (project default), 0 for the searches that never discard, 0.4
    when the strategy flag is omitted (gnn_exp's legacy default), explicit wins."""
    m = _load("create_all_training_data.py")

    class A:
        def __init__(self, d=None):
            self.discard_factor = d
    assert m._discard_for("S_DFS", A()) == 0.6
    assert m._discard_for("BFS", A()) == 0.0
    assert m._discard_for("DFS", A()) == 0.0
    assert m._discard_for("HFS", A()) == 0.0
    assert m._discard_for(None, A()) == 0.4
    assert m._discard_for("S_DFS", A(0.1)) == 0.1 and m._discard_for("BFS", A(0.1)) == 0.1
