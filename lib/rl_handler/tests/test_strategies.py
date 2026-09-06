"""The strategy layer: names, the two on-disk layouts, and the refusal to guess.

One tree per (instance, generation strategy). The user decides which behaviour
policies exist by deciding what to GENERATE; this module is what lets the trainer
find exactly those trees and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.offline.strategies import (
    LEGACY_STRATEGY,
    STRATEGIES,
    available_strategies,
    depth_from_csv_name,
    dir_name,
    discover_tables,
    normalize_strategy,
    parse_strategy_list,
    select_tables,
    split_tree_name,
    strategy_from_csv_name,
    strategy_of_table,
    tree_name,
)

HEADER = "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action\n"


def _table(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "a.dot,0,1,g.dot,init.dot,0\n")
    return path


# ------------------------------------------------------------ names ---------

@pytest.mark.parametrize("spelling,want", [
    ("BFS", "bfs"), ("bfs", "bfs"), ("S-DFS", "s_dfs"), ("s_dfs", "s_dfs"),
    ("S_DFS", "s_dfs"), ("Hfs", "hfs"), (" dfs ", "dfs"),
])
def test_normalize_accepts_every_spelling(spelling, want):
    assert normalize_strategy(spelling) == want


def test_normalize_rejects_junk():
    with pytest.raises(ValueError, match="unknown generation strategy"):
        normalize_strategy("random")   # a synthetic ranking, not a generator


def test_dir_names_are_the_cpp_enum_spellings():
    assert [dir_name(s) for s in STRATEGIES] == ["BFS", "DFS", "S_DFS", "HFS"]


@pytest.mark.parametrize("fname,want", [
    ("CC_2_2_3__pl_4_BFS_depth_20.csv", "bfs"),
    ("CC_2_2_3__pl_4_DFS_depth_20.csv", "dfs"),
    ("CC_2_2_3__pl_4_S_DFS_depth_20.csv", "s_dfs"),
    ("CC_2_2_3__pl_4_HFS_SUBGOALS__depth_20.csv", "hfs"),     # the double underscore
    ("CC_2_2_3__pl_4_HFS_L_PG__depth_25.csv", "hfs"),
    ("CC_2_2_3__pl_4_depth_25.csv", None),                    # legacy: no token
    ("SC_R_10_10__pl_10_depth_40.csv", None),
])
def test_strategy_token_in_the_csv_name(fname, want):
    assert strategy_from_csv_name(fname) == want


def test_depth_from_csv_name():
    assert depth_from_csv_name("CC_2_2_3__pl_4_HFS_SUBGOALS__depth_20.csv") == 20
    assert depth_from_csv_name("x_depth_25.csv") == 25


def test_tree_name_keeps_config_and_pl_parsable():
    """`config_of` rsplits on `__pl_`; `expected_optimal` regexes `__pl_(\\d+)`.
    The strategy suffix must not break either."""
    from src.offline.selection import config_of
    from src.offline.usability import expected_optimal
    n = tree_name("CC_2_3_4__pl_7", "BFS")
    assert n == "CC_2_3_4__pl_7@bfs"
    assert config_of(n) == "CC_2_3_4"
    assert expected_optimal(n) == 7
    assert split_tree_name(n) == ("CC_2_3_4__pl_7", "bfs")
    assert split_tree_name("CC_2_3_4__pl_7") == ("CC_2_3_4__pl_7", None)


def test_parse_strategy_list():
    assert parse_strategy_list(["BFS", "hfs"]) == ["bfs", "hfs"]
    assert parse_strategy_list(["BFS,S-DFS"]) == ["bfs", "s_dfs"]
    assert parse_strategy_list(["all"]) == list(STRATEGIES)
    assert parse_strategy_list(["HFS", "bfs", "HFS"]) == ["bfs", "hfs"]   # canonical order, dedup
    with pytest.raises(ValueError):
        parse_strategy_list(["bfs", "random"])


# --------------------------------------------------------- discovery --------

def test_strategy_layout_is_discovered(tmp_path):
    td = tmp_path / "training_data"
    a = _table(td / "BFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_BFS_depth_20.csv")
    b = _table(td / "HFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_HFS_SUBGOALS__depth_20.csv")
    c = _table(td / "BFS" / "CC_2_2_3__pl_6" / "CC_2_2_3__pl_6_BFS_depth_20.csv")
    (td / "BFS" / "seeds.txt").write_text("CC_2_2_3__pl_4,1\n")    # a file, not an instance
    got = discover_tables(td)
    assert got == {"CC_2_2_3__pl_4": {"bfs": a, "hfs": b}, "CC_2_2_3__pl_6": {"bfs": c}}
    assert available_strategies(got) == ["bfs", "hfs"]
    for p in (a, b, c):
        assert strategy_of_table(p) == strategy_from_csv_name(p.name)


def test_legacy_flat_layout_is_s_dfs(tmp_path):
    td = tmp_path / "training_data"
    a = _table(td / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_depth_25.csv")
    (td / "seeds.txt").write_text("")
    got = discover_tables(td)
    assert got == {"CC_2_2_3__pl_4": {LEGACY_STRATEGY: a}}
    assert strategy_of_table(a) == "s_dfs"


def test_flat_layout_with_a_token_reads_the_token(tmp_path):
    """A single-strategy run that was not nested (the flag was passed to the
    per-domain script by hand) still identifies itself by the file name."""
    td = tmp_path / "training_data"
    a = _table(td / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_BFS_depth_20.csv")
    assert discover_tables(td) == {"CC_2_2_3__pl_4": {"bfs": a}}


def test_two_tables_for_one_strategy_is_an_error_not_first_match(tmp_path):
    """The previous resolver took matches[0]; `_BFS_` sorts before `_S_DFS_`, so a
    folder holding both would silently have trained BFS. Refuse instead."""
    td = tmp_path / "training_data"
    _table(td / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_S_DFS_depth_20.csv")
    _table(td / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_S_DFS_depth_25.csv")
    with pytest.raises(ValueError, match="two tables for strategy S_DFS"):
        discover_tables(td)


def test_folder_and_name_disagreeing_is_an_error(tmp_path):
    td = tmp_path / "training_data"
    _table(td / "BFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_HFS_SUBGOALS__depth_20.csv")
    with pytest.raises(ValueError, match="moved between strategy folders"):
        discover_tables(td)


def test_mixed_layout_is_an_error(tmp_path):
    td = tmp_path / "training_data"
    _table(td / "BFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_BFS_depth_20.csv")
    _table(td / "CC_2_2_3__pl_6" / "CC_2_2_3__pl_6_depth_25.csv")
    with pytest.raises(ValueError, match="mixes the strategy layout"):
        discover_tables(td)


def test_missing_dir_is_empty():
    assert discover_tables(Path("/nonexistent/training_data")) == {}


# --------------------------------------------------------- selection --------

def _two_strategy_dir(tmp_path):
    td = tmp_path / "training_data"
    a = _table(td / "BFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_BFS_depth_20.csv")
    b = _table(td / "HFS" / "CC_2_2_3__pl_4" / "CC_2_2_3__pl_4_HFS_SUBGOALS__depth_20.csv")
    c = _table(td / "BFS" / "CC_2_2_3__pl_6" / "CC_2_2_3__pl_6_BFS_depth_20.csv")
    return td, a, b, c


def test_default_selection_is_everything_generated(tmp_path):
    td, a, b, c = _two_strategy_dir(tmp_path)
    assert select_tables(discover_tables(td), None, verbose=False) == [a, b, c]


def test_selection_restricts_to_the_requested_strategies(tmp_path, capsys):
    td, a, b, c = _two_strategy_dir(tmp_path)
    assert select_tables(discover_tables(td), ["hfs"]) == [b]
    out = capsys.readouterr().out
    assert "CC_2_2_3__pl_6: no HFS table" in out, "an instance lacking it is reported"


def test_requesting_a_strategy_that_was_never_generated_is_an_error(tmp_path):
    """THE contract: the user picks pi_b by generating it. Asking for one that is
    not on disk must not fall back to whatever is."""
    td, *_ = _two_strategy_dir(tmp_path)
    with pytest.raises(ValueError, match="never generated here"):
        select_tables(discover_tables(td), ["bfs", "s-dfs"])
