"""Regression test for GraphDataPipeline._read_csv header-driven parsing.

The generation table grew from 4 to 6 columns
  File Path, Depth, Distance From Goal, Goal, File Path Predecessor, Action
A fixed `split(",", 3)` folded Predecessor+Action into the `Goal` field, which
broke separated goal loading (the `Goal` column is the goal_tree.dot path).
`_read_csv` now parses by HEADER, so:
  * the 6-column row yields a clean goal_tree.dot path (no pollution), and
  * a row whose columns are in a different order still maps correctly.

`_read_csv` references no instance attributes, so we call it with a dummy self.

Run from lib/gnn_handler:  ../../.venv/bin/python tests/test_read_csv.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import GraphDataPipeline  # noqa: E402

GOAL = "exp/x/_models/D/training_data/INST/goal_tree.dot"
STATE = "exp/x/_models/D/training_data/INST/RawFiles/hash_separated/000023.dot"
PRED = "exp/x/_models/D/training_data/INST/RawFiles/hash_separated/000022.dot"


def _write(text: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    f.write(text)
    f.close()
    return Path(f.name)


def test_six_column_goal_unpolluted() -> None:
    csv = _write(
        "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action\n"
        f"{STATE},17,0,{GOAL},{PRED},2\n"
    )
    df = GraphDataPipeline._read_csv(SimpleNamespace(), csv)
    assert df["Goal"].iloc[0] == GOAL, df["Goal"].iloc[0]
    assert "," not in df["Goal"].iloc[0]
    assert df["File Path"].iloc[0] == STATE
    assert int(df["Depth"].iloc[0]) == 17
    assert int(df["Distance From Goal"].iloc[0]) == 0
    print("PASS test_six_column_goal_unpolluted -> Goal =", df["Goal"].iloc[0])


def test_reordered_columns_map_correctly() -> None:
    # Same fields, different column order; header-driven parse must still work.
    csv = _write(
        "Action,Goal,File Path,File Path Predecessor,Distance From Goal,Depth\n"
        f"2,{GOAL},{STATE},{PRED},0,17\n"
    )
    df = GraphDataPipeline._read_csv(SimpleNamespace(), csv)
    assert df["Goal"].iloc[0] == GOAL
    assert df["File Path"].iloc[0] == STATE
    assert int(df["Depth"].iloc[0]) == 17
    assert int(df["Distance From Goal"].iloc[0]) == 0
    print("PASS test_reordered_columns_map_correctly -> Goal =", df["Goal"].iloc[0])


if __name__ == "__main__":
    test_six_column_goal_unpolluted()
    test_reordered_columns_map_correctly()
    print("\nALL _read_csv TESTS PASSED")
