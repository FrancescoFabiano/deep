"""Shared fixtures. Run from lib/rl_handler:  ../../.venv/bin/python -m pytest tests -q

`lib/gnn_handler` and `lib/rl_handler` BOTH define a top-level package `src`, so
tests must put THIS package root first on sys.path and be run from here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

REPO_ROOT = PKG_ROOT.parents[1]

from src.offline.strategies import LEGACY_STRATEGY, dir_name, strategy_of_table  # noqa: E402
from src.offline.tree import (  # noqa: E402
    TreeInstance,
    compute_delta,
    compute_expansion_order,
    load_tree_instance,
)

# The two generation tables the design's numbers are pinned to. Located by
# SEARCH, not by a fixed path: `exp/rl_exp/` holds batch dirs that the data-
# generation launcher creates and rotates away while it runs, so hardcoding
# `batch<N>` makes these tests fail for reasons that have nothing to do with the
# code. The generator is reproducible -- a regenerated CC_2_3_4__pl_7 reproduces
# n=4382, h*(root)=29, delta(root)=34 exactly -- so any batch's copy will do.
SHIPPED_INSTANCES = ("CC_2_3_4__pl_7", "SC_R_10_10__pl_10")


# Where generation tables may live: the repo's batches (rotated by the launcher) and
# the strategy smoke runs (out of tree: /tmp is RAM here, so they sit in /var/tmp).
TABLE_ROOTS = [REPO_ROOT / "exp", Path("/var/tmp")]


def find_generation_table(instance: str, strategy: str = LEGACY_STRATEGY) -> Optional[Path]:
    """Newest table of `instance` generated with `strategy` (default: the legacy
    S_DFS tables the design's numbers are pinned to), in either layout:
    `.../training_data/<STRAT>/<instance>/*_depth_*.csv` or the flat legacy one."""
    hits = []
    for root in TABLE_ROOTS:
        if not root.is_dir():
            continue
        for p in root.glob(f"*/*/_models/*/training_data/**/{instance}/*_depth_*.csv"):
            try:
                if strategy_of_table(p) == strategy:
                    hits.append(p)
            except ValueError:
                continue
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def make_tree(
    children: Sequence[Sequence[int]],
    goals: Sequence[int],
    name: str = "synthetic",
    root_id: int = 0,
    h_star: Optional[Sequence[float]] = None,
    dot_index: Optional[Sequence[int]] = None,
    strategy: Optional[str] = None,
) -> TreeInstance:
    """Build a TreeInstance directly from a child-list spec (no CSV).

    Depth is derived by BFS from the root. `h_star` defaults to `delta`: on a
    strict tree the DAG distance and the tree distance coincide, so synthetic
    fixtures satisfy `delta >= h*` with equality. The delta-vs-h* GAP is a
    property of the real generator's DFS-spanning-tree reconstruction and is
    tested against the shipped CSVs instead.

    `dot_index`: per-state creation counter (what the real DOT names carry). When
    given, the generator's expansion order is recovered from it exactly as the
    loader does, so the `trace` behaviour can be tested on a synthetic tree.
    """
    n = len(children)
    expansion_rank = None
    if dot_index is not None:
        _, expansion_rank = compute_expansion_order(children, list(dot_index))
    is_goal = [i in set(goals) for i in range(n)]
    depth = [-1] * n
    depth[root_id] = 0
    stack = [root_id]
    while stack:
        v = stack.pop()
        for c in children[v]:
            if depth[c] == -1:
                depth[c] = depth[v] + 1
                stack.append(c)
    delta = compute_delta(children, is_goal)
    return TreeInstance(
        name=name,
        csv_path=f"<{name}>",
        state_paths=[f"{name}/{i}.dot" for i in range(n)],
        depth=[max(0, d) for d in depth],
        h_star=list(h_star) if h_star is not None else list(delta),
        is_goal=is_goal,
        children=[list(c) for c in children],
        root_id=root_id,
        n_orphan_states=0,
        delta=delta,
        strategy=strategy,
        expansion_rank=expansion_rank,
    )


# --- the 19-node transition-parity fixture (test 5) ---------------------------
#
#   0 d0 -> 1,2,3        b=3          delta=4
#   1 d1 -> 4,5,6,7      b=4          delta=3
#   2 d1 -> []           b=0  DEAD    delta=inf
#   3 d1 -> 8            b=1          delta=3
#   4 d2 -> 9,10         b=2          delta=2
#   5 d2 -> []           b=0  DEAD    delta=inf
#   6 d2 -> 11           b=1          delta=3
#   7 d2 -> []           b=0  DEAD    delta=inf
#   8 d2 -> 12,13        b=2          delta=2
#   9 d3 -> 14           b=1          delta=1
#  10 d3 -> []           b=0  DEAD    delta=inf
#  11 d3 -> 15,16        b=2          delta=2
#  12 d3 -> []           b=0  DEAD    delta=inf
#  13 d3 -> 17           b=1          delta=1
#  14 d4    GOAL                      delta=0
#  15 d4 -> 18           b=1          delta=1
#  16 d4 -> []           b=0  DEAD    delta=inf
#  17 d4    GOAL                      delta=0
#  18 d5    GOAL                      delta=0
#
# b_v in {0,1,2,3,4}; 3 goals; 6 dead ends. delta(root) = 4.
T19_CHILDREN: List[List[int]] = [
    [1, 2, 3],      # 0
    [4, 5, 6, 7],   # 1
    [],             # 2  dead
    [8],            # 3
    [9, 10],        # 4
    [],             # 5  dead
    [11],           # 6
    [],             # 7  dead
    [12, 13],       # 8
    [14],           # 9
    [],             # 10 dead
    [15, 16],       # 11
    [],             # 12 dead
    [17],           # 13
    [],             # 14 GOAL
    [18],           # 15
    [],             # 16 dead
    [],             # 17 GOAL
    [],             # 18 GOAL
]
T19_GOALS = [14, 17, 18]
T19_DEAD = [2, 5, 7, 10, 12, 16]


@pytest.fixture(scope="session")
def t19() -> TreeInstance:
    return make_tree(T19_CHILDREN, T19_GOALS, name="t19")


@pytest.fixture(scope="session")
def shipped_instances() -> Dict[str, TreeInstance]:
    out: Dict[str, TreeInstance] = {}
    missing = []
    for name in SHIPPED_INSTANCES:
        p = find_generation_table(name)
        if p is None:
            missing.append(name)
            continue
        out[name] = load_tree_instance(p, name=name, kind_of_data="merged")
    if missing:
        pytest.skip(
            f"generation tables not found for {missing}; regenerate with "
            f"scripts/gnn_exp/create_all_training_data.py"
        )
    return out


# One real table per strategy of ONE instance, when the strategy smoke runs exist
# (/var/tmp/smoke_batch1_{BFS,DFS,S-DFS,HFS}, merged, 1000 visits, depth 20). Tests
# that need them skip otherwise: they are the only place all four searches were run
# on the same problem.
STRATEGY_INSTANCE = "CC_2_2_3__pl_4"


@pytest.fixture(scope="session")
def strategy_tables() -> Dict[str, TreeInstance]:
    out: Dict[str, TreeInstance] = {}
    for s in ("bfs", "dfs", "s_dfs", "hfs"):
        p = find_generation_table(STRATEGY_INSTANCE, s)
        if p is not None:
            out[s] = load_tree_instance(p, kind_of_data="merged")
    if len(out) < 4:
        pytest.skip(f"need {STRATEGY_INSTANCE} tables for all four strategies "
                    f"(have {[dir_name(s) for s in out]}); run the strategy smoke "
                    f"generation into /var/tmp/smoke_batch1_<STRAT>")
    return out
