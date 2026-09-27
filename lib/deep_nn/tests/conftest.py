"""Run from lib/deep_nn:  ../../.venv/bin/python -m pytest tests -q"""

from __future__ import annotations

import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[2]
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

# A merged planner DOT, exactly as HelperPrint::print_dataset_format writes it:
# epsilon(0) -> goal parent(1), the goal subtree, epsilon -> designated world,
# then the belief edges (agent labels 7, 8). One world (-77) is designated but
# has no belief edge, which a separated-generated file would lose.
MERGED_DOT = """digraph G {
  0 -> 1 [label="2"];
  1 -> 19 [label="5"];
  19 -> 20 [label="5"];
  20 -> 7 [label="5"];
  0 -> -8011962737897461503 [label="3"];
  0 -> -77 [label="3"];
  3010661539059282004 -> 3010661539059282004 [label="7"];
  3010661539059282004 -> -8011962737897461503 [label="8"];
  -8011962737897461503 -> 18446744073709551615 [label="7"];
  18446744073709551615 -> 3010661539059282004 [label="8"];
}
"""
