"""Unit tests for the P2 order auxiliary (no-padding beams).

Covers:
  - the order-eligible slot set is EVERY slot (beams are live-pool only; there is
    no padding), with unreachable d* mapped to a worst-sorting sentinel;
  - the <2-distinct-eligible-d* skip still applies (a beam with no d* spread —
    e.g. all-unreachable — carries no order signal and is skipped);
  - lambda_ord == 0 never enters the order path (byte-identical value-only);
  - lambda_ord > 0 produces an order gradient that lifts a finite-d* slot's logit
    above a larger-d* slot's, and populates the usable-pair / skip instrumentation.

Run from lib/rl_handler:  ../../.venv/bin/python tests/test_order_aux.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.models.frontier_policy import FrontierPolicyNetwork  # noqa: E402
from src.offline.dqn import OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache, StateGraph  # noqa: E402
from src.offline.replay import ReplayBuffer, Transition  # noqa: E402
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
    return InstanceCache([
        StateGraph(
            node_ids=torch.tensor([100 + i], dtype=torch.int64),
            edge_index=torch.zeros((2, 0), dtype=torch.int64),
            edge_attr=torch.zeros((0,), dtype=torch.int64),
        )
        for i in range(n)
    ])


def make_trainer(lambda_ord=0.0):
    inst = make_instance()
    model = FrontierPolicyNetwork(
        node_input_dim=1, hidden_dim=16, gnn_layers=1, conv_type="gine",
        pooling_type="mean", num_edge_labels=8, num_node_labels=64,
    )
    return OfflineDQNTrainer(
        model=model, instances=[inst], caches=[make_cache(inst.n_states)],
        train_ids=[0], val_ids=[0], fringe_size=8, gamma=0.99, batch_size=4,
        warmup=1, device="cpu", signal_mode="rank-rl", rank_variant="advantage",
        lambda_ord=lambda_ord,
    ), inst


class _Stub:
    _order_eligible_slots = OfflineDQNTrainer._order_eligible_slots
    _pairwise_term = OfflineDQNTrainer._pairwise_term

    def __init__(self, inst):
        self.instances = [inst]
        self.device = torch.device("cpu")


def _T(fringe, action):
    return Transition(inst=0, fringe=tuple(fringe), action=action, reward=-1.0,
                      next_fringe=(), done=True, regime="dfs")


def test_transition_roundtrips_replay():
    buf = ReplayBuffer(8, seed=0)
    buf.push(_T((1, 2, 3), 0))
    s = buf.sample(1)[0]
    assert s.fringe == (1, 2, 3) and s.regime == "dfs"


def test_order_eligible_is_every_slot():
    stub = _Stub(make_instance())
    # fringe ids (2,1,3): d* = 1, 2, INF -> all slots eligible; unreachable id 3
    # maps to the worst-sorting sentinel = max(finite)+1 = 3.0.
    pos, dv = stub._order_eligible_slots(_T((2, 1, 3), 0))
    assert pos == [0, 1, 2], pos
    assert dv == [1.0, 2.0, 3.0], dv


def test_pairwise_over_all_slots():
    stub = _Stub(make_instance())
    # ids (2,1) -> d* 1,2 -> exactly one strictly-ordered pair.
    pos, dv = stub._order_eligible_slots(_T((2, 1), 0))
    sl = torch.tensor([0.5, -0.5], requires_grad=True)
    term, npairs = stub._pairwise_term(sl, torch.tensor(dv))
    assert npairs == 1 and term is not None


def test_skip_on_no_dstar_spread():
    stub = _Stub(make_instance())
    # ids (3,5): both unreachable -> both map to the same sentinel -> tie -> skip.
    pos, dv = stub._order_eligible_slots(_T((3, 5), 0))
    assert pos == [0, 1]
    assert len(set(dv)) < 2, dv
    # a fringe with finite spread does NOT skip
    _, dv2 = stub._order_eligible_slots(_T((2, 1, 3), 0))
    assert len(set(dv2)) >= 2


def test_lambda0_never_enters_order_path():
    tr, _ = make_trainer(lambda_ord=0.0)
    for _ in range(8):
        tr.replay.push(_T((1, 2, 3), 0))
    out = tr._update()
    assert out["order_loss"] == 0.0
    assert tr._order_seen_fringes == 0             # _order_aux_loss not called
    assert tr.order_stats()["order_usable_pairs"] == 0


def test_lambda_pos_order_gradient_and_instrumentation():
    # gradient direction: pair (d=1 better than d=2) lifts slot0, lowers slot1.
    stub = _Stub(make_instance())
    sl = torch.zeros(2, requires_grad=True)
    dl = torch.tensor([1.0, 2.0])
    term, _ = stub._pairwise_term(sl, dl)
    term.backward()
    assert float(sl.grad[0]) < 0.0 < float(sl.grad[1])   # SGD raises l0 above l1

    # integration: lambda>0 trains and populates the order instrumentation.
    tr, _ = make_trainer(lambda_ord=0.5)
    for _ in range(8):
        tr.replay.push(_T((1, 2, 3), 0))
    out = tr._update()
    assert tr._order_seen_fringes > 0
    st = tr.order_stats()
    # eligible slots = ids 1,2,3 (d*=2,1,sentinel) -> usable pairs > 0, 0 skips
    assert st["order_usable_pairs"] > 0
    assert st["order_skipped_fringes"] == 0
    assert "order_loss" in out


def main():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} order-aux checks passed.")


if __name__ == "__main__":
    main()
