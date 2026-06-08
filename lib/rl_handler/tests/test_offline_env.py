"""Phase-2 verification for src/offline/tree_env.py on real CC tables.

Sanity ladder (must hold if env mechanics are right):
  optimal <= oracle(-d*) rollout <= random rollout (in #expansions, typically)
and BFS provides the uninformed-but-systematic reference.

Run from lib/rl_handler:  ../../.venv/bin/python tests/test_offline_env.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.tree_env import (  # noqa: E402
    FringeEnv,
    bfs_expansions,
    load_tree_instance,
    oracle_policy,
    rollout,
)

REPO = Path(__file__).resolve().parents[3]

CSVS = {
    "CC_2_2_4__pl_7": "out/NN/Training/CC_2_2_4__pl_7/CC_2_2_4__pl_7_depth_25.csv",
    "CC_3_2_3__pl_6": "out/NN/Training/CC_3_2_3__pl_6/CC_3_2_3__pl_6_depth_25.csv",
    "CC_3_2_3__pl_7": "out/NN/Training/CC_3_2_3__pl_7/CC_3_2_3__pl_7_depth_25.csv",
}


def main() -> None:
    rng = random.Random(0)
    for name, rel in CSVS.items():
        inst = load_tree_instance(REPO / rel, name=name)
        st = inst.stats()
        print(f"\n=== {name} ===")
        for k in (
            "n_states", "n_goal_states", "n_unreachable_states",
            "n_dead_end_leaves", "branching_internal_mean", "root_branching",
            "depth_max", "optimal_expansions",
        ):
            print(f"  {k}: {st[k]}")

        bfs = bfs_expansions(inst)
        print(f"  BFS: {bfs}")

        env = FringeEnv(inst, fringe_size=32, seed=123)
        orc = [rollout(env, oracle_policy(inst), seed=1000 + i) for i in range(5)]
        rnd = [
            rollout(env, lambda f: rng.randrange(len(f)), seed=2000 + i)
            for i in range(5)
        ]
        orc_exp = [r["expansions"] for r in orc]
        rnd_exp = [r["expansions"] for r in rnd]
        print(f"  oracle(-d*) expansions over 5 seeds: {orc_exp} "
              f"(goal_found={all(r['goal_found'] for r in orc)})")
        print(f"  random      expansions over 5 seeds: {rnd_exp} "
              f"(goal_found={all(r['goal_found'] for r in rnd)})")

        opt = st["optimal_expansions"]
        assert all(r["goal_found"] for r in orc), "oracle must reach the goal"
        assert min(orc_exp) >= opt, "no rollout can beat the optimal"
        assert mean(orc_exp) <= mean(rnd_exp), "oracle must beat random on average"

        # Determinism: same seed -> same trajectory length under oracle.
        a = rollout(env, oracle_policy(inst), seed=42)
        b = rollout(env, oracle_policy(inst), seed=42)
        assert a == b, "env must be deterministic given the seed"
    print("\nAll Phase-2 env checks passed.")


if __name__ == "__main__":
    main()
