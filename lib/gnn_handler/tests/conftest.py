"""Run from lib/gnn_handler:  ../../.venv/bin/python -m pytest tests -q

`lib/gnn_handler` and `lib/rl_handler` BOTH define a top-level package `src`,
so tests must put THIS package root first on sys.path and be run from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

REPO_ROOT = PKG_ROOT.parents[1]

# A planner-style state DOT: 64-bit hash ids (one above 2^63, as the older
# generator printed unsigned), agent labels, self-loops and multi-edges.
STATE_DOT = """digraph G {
  -6451783155724511960 -> -6451783155724511960 [label="8"];
  -6451783155724511960 -> -2964014973223465598 [label="8"];
  -6451783155724511960 -> 18446744073709551615 [label="9"];
  -2964014973223465598 -> -6451783155724511960 [label="9"];
  18446744073709551615 -> 18446744073709551615 [label="10"];
}
"""
GOAL_DOT = """digraph G {
  1 -> 24 [label="5"];
  24 -> 8 [label="5"];
  8 -> 24 [label="5"];
}
"""
BITS_DOT = "digraph G {\n" + "\n".join(
    f'  {"0" * 41}1 -> {"1" * 42} [label="{k}"];' for k in (3, 4)
) + "\n}\n"
