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
sys.path.append(str(PKG_ROOT.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from deep_nn.dot import StateGraph  # noqa: E402

HEADER = "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action\n"

# A merged planner DOT (epsilon -> goal parent, goal subtree, epsilon -> designated
# world, belief edges); the belief labels vary per state in the fixtures.
MERGED_DOT = """digraph G {
  0 -> 1 [label="2"];
  1 -> 19 [label="5"];
  19 -> 7 [label="5"];
  0 -> -8011962737897461503 [label="3"];
  3010661539059282004 -> 3010661539059282004 [label="7"];
  3010661539059282004 -> -8011962737897461503 [label="8"];
  -8011962737897461503 -> 18446744073709551615 [label="7"];
  18446744073709551615 -> 3010661539059282004 [label="8"];
}
"""


def random_graph(rng: np.random.Generator, n: int = 5, e: int = 7) -> StateGraph:
    return StateGraph(
        node_ids=torch.from_numpy(rng.integers(-(2**62), 2**62, size=n, dtype=np.int64).copy()),
        edge_index=torch.from_numpy(rng.integers(0, n, size=(2, e), dtype=np.int64).copy()),
        edge_attr=torch.from_numpy(rng.integers(4, 40, size=e, dtype=np.int64).copy()),
        pointed_ids=torch.tensor([0], dtype=torch.int64),
    )


class Cache:
    def __init__(self, states, goal=None):
        self.states, self.goal = states, goal


def write_instance(root: Path, name: str, n_states: int = 8) -> Path:
    """A tiny generation table + merged DOTs: a chain root -> 1 -> ... with the
    last state a goal and one dead-end sibling (distance 1e6)."""
    inst = root / name
    raw = inst / "RawFiles" / "hash_merged"
    raw.mkdir(parents=True)
    rows = []
    paths = [raw / f"{i + 1:06d}.dot" for i in range(n_states)]
    for i, p in enumerate(paths):
        p.write_text(MERGED_DOT.replace('"7"', f'"{7 + i}"'))
    goal = n_states - 2
    for i, p in enumerate(paths):
        if i <= goal:
            pred = raw / "init.dot" if i == 0 else paths[i - 1]
            rows.append(f"{p},{i},{goal - i:010d},{inst / 'goal_tree.dot'},{pred},0\n")
        else:                                       # dead end hanging off the root
            rows.append(f"{p},1,{1000000:010d},{inst / 'goal_tree.dot'},{paths[0]},1\n")
    csv = inst / f"{name}_S_DFS_depth_10.csv"
    csv.write_text(HEADER + "".join(rows))
    return csv
