"""Unit tests for flooring unreachables into the objective (P2-prep, Task 1a).

Covers all five inclusion sites + the same-distance equivalence class:
  - floor_v = -1/(1-gamma);
  - exact-return value target keeps an unreachable-chosen transition at floor_v;
  - order-aux (_rank_sup_loss) puts a finite node above an unreachable and ties
    two unreachables (all-unreachable fringe contributes nothing);
  - _rank_fraction_reward counts unreachables in the denominator as worst;
  - _phi (PBRS) returns floor_v for an all-unreachable fringe;
  - an all-unreachable fringe is dropped by the same-distance default.

Run from lib/rl_handler:  ../../.venv/bin/python tests/test_floor_unreachable.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.models.frontier_policy import FrontierPolicyNetwork  # noqa: E402
from src.offline.dqn import OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache, StateGraph  # noqa: E402
from src.offline.regimes import single_distinct_dstar  # noqa: E402
from src.offline.replay import Transition  # noqa: E402
from src.offline.tree_env import UNREACHABLE_DISTANCE, TreeInstance  # noqa: E402

INF = UNREACHABLE_DISTANCE


def make_instance() -> TreeInstance:
    # ids:    0     1     2     3      4(goal) 5
    distance = [3.0, 2.0, 1.0, INF, 0.0, INF]
    depth = [0, 1, 2, 3, 1, 2]
    is_goal = [d == 0.0 for d in distance]
    children = [[1, 2, 4], [2], [], [], [], []]
    return TreeInstance(
        name="fake", csv_path="x", state_paths=[f"p{i}" for i in range(6)],
        depth=depth, distance=distance, is_goal=is_goal, children=children,
        root_id=0, n_orphan_states=0,
    )


def make_cache(n: int) -> InstanceCache:
    states = [
        StateGraph(
            node_ids=torch.tensor([100 + i], dtype=torch.int64),
            edge_index=torch.zeros((2, 0), dtype=torch.int64),
            edge_attr=torch.zeros((0,), dtype=torch.int64),
        )
        for i in range(n)
    ]
    return InstanceCache(states)


def make_trainer(signal_mode="basic", gamma=0.99, rank_variant=None):
    inst = make_instance()
    model = FrontierPolicyNetwork(
        node_input_dim=1, hidden_dim=16, gnn_layers=1, conv_type="gine",
        pooling_type="mean", num_edge_labels=8, num_node_labels=64,
    )
    tr = OfflineDQNTrainer(
        model=model, instances=[inst], caches=[make_cache(inst.n_states)],
        train_ids=[0], val_ids=[0], fringe_size=4, gamma=gamma, batch_size=4,
        warmup=1, device="cpu", signal_mode=signal_mode, rank_variant=rank_variant,
    )
    return tr, inst


# ---- a lightweight stub for the pure d*-logic methods (no model needed) ----
class _Stub:
    # the real per-fringe helpers (need only self.instances/.device/.rank_variant)
    _order_eligible_slots = OfflineDQNTrainer._order_eligible_slots
    _pairwise_term = OfflineDQNTrainer._pairwise_term

    def __init__(self, inst, floor_v, rank_variant="pairwise"):
        self.instances = [inst]
        self.floor_v = floor_v
        self.device = torch.device("cpu")
        self.rank_variant = rank_variant


def _T(fringe, action):
    return Transition(inst=0, fringe=tuple(fringe), action=action,
                      reward=-1.0, next_fringe=(), done=True)


def test_floor_v_matches_gamma():
    for gamma, want in ((0.99, -100.0), (0.9, -10.0), (0.5, -2.0)):
        tr, _ = make_trainer(gamma=gamma)
        assert abs(tr.floor_v - want) < 1e-6, (gamma, tr.floor_v, want)


def test_exact_return_target_keeps_unreachable_at_floor_v():
    tr, _ = make_trainer(signal_mode="exact-return", gamma=0.99)
    # chosen node id=3 is unreachable -> target must be floor_v (-100), not dropped
    for _ in range(8):
        tr.replay.push(_T((1, 3), action=1))
    out = tr._update()
    assert abs(out["target_mean"] - tr.floor_v) < 1e-4, out["target_mean"]
    # control: a finite chosen node (id=2, d*=1) -> target -1
    tr2, _ = make_trainer(signal_mode="exact-return", gamma=0.99)
    for _ in range(8):
        tr2.replay.push(_T((1, 2), action=1))
    out2 = tr2._update()
    assert abs(out2["target_mean"] - (-1.0)) < 1e-4, out2["target_mean"]


def test_rank_sup_finite_above_unreachable():
    inst = make_instance()
    stub = _Stub(inst, floor_v=-100.0, rank_variant="pairwise")
    batch = [_T((1, 3, 2), action=0)]   # d* = [2, inf, 1]
    ptr = torch.tensor([0, 3])
    # order-respecting logits: best=id2(d1, pos2) high, worst=id3(inf, pos1) low
    good = torch.tensor([0.0, -5.0, 5.0])
    bad = torch.tensor([0.0, 5.0, -5.0])   # unreachable on top -> violates order
    loss_good = OfflineDQNTrainer._rank_sup_loss(stub, batch, good, ptr)
    loss_bad = OfflineDQNTrainer._rank_sup_loss(stub, batch, bad, ptr)
    assert float(loss_good) < float(loss_bad), (float(loss_good), float(loss_bad))
    # the finite-vs-unreachable pair must contribute (loss strictly > 0 for bad)
    assert float(loss_bad) > 0.0


def test_rank_sup_all_unreachable_contributes_nothing():
    inst = make_instance()
    stub = _Stub(inst, floor_v=-100.0, rank_variant="pairwise")
    batch = [_T((3, 5), action=0)]      # both unreachable -> tie, <2 distinct
    ptr = torch.tensor([0, 2])
    logits = torch.tensor([3.0, -2.0])
    loss = OfflineDQNTrainer._rank_sup_loss(stub, batch, logits, ptr)
    assert float(loss) == 0.0, float(loss)


def test_rank_fraction_reward_counts_unreachables():
    inst = make_instance()
    stub = _Stub(inst, floor_v=-100.0)
    # fringe ids (2,1,3) -> d* = [1, 2, inf]
    assert OfflineDQNTrainer._rank_fraction_reward(stub, _T((2, 1, 3), 0)) == 1.0  # best
    assert OfflineDQNTrainer._rank_fraction_reward(stub, _T((2, 1, 3), 2)) == 0.0  # unreachable
    # chosen id1 (d*=2): strictly worse = only the unreachable -> 1/2
    assert abs(OfflineDQNTrainer._rank_fraction_reward(stub, _T((2, 1, 3), 1)) - 0.5) < 1e-9


def test_phi_all_unreachable_is_floor_v():
    inst = make_instance()
    stub = _Stub(inst, floor_v=-100.0)
    assert OfflineDQNTrainer._phi(stub, 0, (1, 2)) == -1.0       # best finite d*=1 -> -1
    assert OfflineDQNTrainer._phi(stub, 0, (3,)) == -100.0       # all unreachable -> floor_v
    assert OfflineDQNTrainer._phi(stub, 0, (3, 5)) == -100.0


def test_same_distance_drops_all_unreachable():
    inst = make_instance()
    assert single_distinct_dstar(inst, (3, 5)) is True           # all-unreachable -> dropped
    assert single_distinct_dstar(inst, (1, 3)) is False          # finite + unreachable -> kept
    assert single_distinct_dstar(inst, (1, 2)) is False          # two finite distinct -> kept


def main():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} floor-unreachable checks passed.")


if __name__ == "__main__":
    main()
