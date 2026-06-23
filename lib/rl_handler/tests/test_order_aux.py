"""Unit tests for the P2 order auxiliary + padded-slot exclusion.

Covers:
  - pad_mask plumbs intact + slot-aligned through replay (push -> sample);
  - the order-eligible slot set excludes padded slots, so no pair ever involves a
    padded slot; a fringe of 2 live + N padded forms only the live-live pair(s);
  - the <2-distinct skip is evaluated on the ELIGIBLE (live) set, so 2 same-d*
    live + distinct-d* padded slots skip (would NOT skip on the raw fringe);
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


def _T(fringe, action, pad_mask=None):
    return Transition(inst=0, fringe=tuple(fringe), action=action, reward=-1.0,
                      next_fringe=(), done=True, regime="dfs", pad_mask=pad_mask)


def test_pad_mask_roundtrips_replay():
    buf = ReplayBuffer(8, seed=0)
    pm = (False, True, False)
    buf.push(_T((1, 2, 3), 0, pad_mask=pm))
    s = buf.sample(1)[0]
    assert s.pad_mask == pm and len(s.pad_mask) == len(s.fringe)
    # default None still works (single-source path)
    buf.push(_T((1, 2), 0, pad_mask=None))


def test_order_eligible_excludes_padded():
    stub = _Stub(make_instance())
    # fringe ids (2,1,3); mark slot 2 (id 3) padded
    pos, dv = stub._order_eligible_slots(_T((2, 1, 3), 0, pad_mask=(False, False, True)))
    assert pos == [0, 1], pos                      # padded position 2 excluded
    assert dv == [1.0, 2.0], dv                    # d* of ids 2 and 1


def test_pair_never_involves_padded():
    stub = _Stub(make_instance())
    # 2 live (ids 2,1 -> d* 1,2) + 3 padded (ids 0,3,5)
    t = _T((2, 1, 0, 3, 5), 0, pad_mask=(False, False, True, True, True))
    pos, dv = stub._order_eligible_slots(t)
    assert pos == [0, 1]                            # only the live slots survive
    sl = torch.tensor([0.5, -0.5], requires_grad=True)
    dl = torch.tensor(dv)
    term, npairs = stub._pairwise_term(sl, dl)
    assert npairs == 1                              # exactly the one live-live pair
    assert term is not None


def test_post_mask_skip_on_live_set():
    stub = _Stub(make_instance())
    # 2 live with the SAME d* (ids 3 and 5 are both unreachable -> tie) + padded
    # ids 1,2 (distinct finite d*). Raw fringe has >=2 distinct, but the ELIGIBLE
    # (live) set is a single tied value -> must skip.
    t = _T((3, 5, 1, 2), 0, pad_mask=(False, False, True, True))
    pos, dv = stub._order_eligible_slots(t)
    assert pos == [0, 1]
    assert len(set(dv)) < 2, dv                     # eligible set ties -> skip
    # sanity: the raw fringe (all live) would NOT skip
    pos2, dv2 = stub._order_eligible_slots(_T((3, 5, 1, 2), 0))
    assert len(set(dv2)) >= 2


def test_lambda0_never_enters_order_path():
    tr, _ = make_trainer(lambda_ord=0.0)
    for _ in range(8):
        tr.replay.push(_T((1, 2, 3), 0, pad_mask=(False, False, True)))
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
        tr.replay.push(_T((1, 2, 3), 0, pad_mask=(False, False, True)))
    out = tr._update()
    assert tr._order_seen_fringes > 0
    st = tr.order_stats()
    # eligible slots per fringe = ids 1,2 (d*=2,1) -> 1 usable pair, 0 skips
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
