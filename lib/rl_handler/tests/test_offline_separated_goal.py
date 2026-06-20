"""Parity tests for the `separated` encoding (per-instance goal_tree.dot).

Asserts the goal-graph packing matches the deployment contract (FringeEvalRL
separated branch + trainer.to_onnx separated export):

1. The exported separated ONNX input names equal the 9-input contract, in order:
     node_features, edge_index, edge_attr, membership,
     goal_node_features, goal_edge_index, goal_edge_attr, goal_batch, mask
2. goal_batch maps each fringe's goal nodes to that fringe's index, so
   goal_emb[candidate_batch] aligns one goal per candidate.
3. A single-fringe pack yields goal_batch all-zeros (inference parity with
   FringeEvalRL, which zero-fills goal_state_batch for one goal graph per call).
4. The slow reference packer (pack_fringe_batch) and the vectorized
   GlobalFlatCache.pack agree on every goal tensor.
5. merged packing is unchanged when no goal graphs are supplied (regression).

Run from lib/rl_handler:
  ../../.venv/bin/python tests/test_offline_separated_goal.py
  (or, where available)  pytest -q tests/test_offline_separated_goal.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.frontier_policy import FrontierPolicyNetwork  # noqa: E402
from src.offline.encoder import (  # noqa: E402
    GlobalFlatCache,
    InstanceCache,
    pack_fringe_batch,
    parse_dot_fast,
)
from src.trainer import RLFrontierTrainer  # noqa: E402

# Two instances, each with two states + a distinct goal graph.
INST0_STATES = [
    "digraph G {\n  5 -> -7 [label=\"2\"];\n  -7 -> 5 [label=\"3\"];\n}\n",
    "digraph G {\n  1 -> 2 [label=\"4\"];\n}\n",
]
INST1_STATES = [
    "digraph G {\n  9 -> 9 [label=\"8\"];\n}\n",
    "digraph G {\n  3 -> 4 [label=\"1\"];\n  4 -> 3 [label=\"1\"];\n}\n",
]
# Goal graphs: instance 0 has 3 nodes / 2 edges, instance 1 has 2 nodes / 1 edge.
GOAL0 = "digraph G {\n  10 -> 11 [label=\"5\"];\n  11 -> 12 [label=\"6\"];\n}\n"
GOAL1 = "digraph G {\n  20 -> 21 [label=\"7\"];\n}\n"

CONTRACT_INPUT_NAMES = [
    "node_features",
    "edge_index",
    "edge_attr",
    "membership",
    "goal_node_features",
    "goal_edge_index",
    "goal_edge_attr",
    "goal_batch",
    "mask",
]


def _caches_and_goals():
    c0 = InstanceCache([parse_dot_fast(s) for s in INST0_STATES])
    c1 = InstanceCache([parse_dot_fast(s) for s in INST1_STATES])
    g0 = parse_dot_fast(GOAL0)
    g1 = parse_dot_fast(GOAL1)
    return [c0, c1], [g0, g1]


def test_goal_batch_maps_to_fringe_index() -> None:
    """2-fringe batch: fringe 0 from inst 0, fringe 1 from inst 1."""
    caches, goals = _caches_and_goals()
    g0, g1 = goals
    # fringe b uses instance b's goal here, so goal_batch must be
    # [0]*n0 + [1]*n1 with n0 = goal0 nodes (3), n1 = goal1 nodes (2).
    out = pack_fringe_batch(
        [(caches[0], [0, 1]), (caches[1], [0, 1])],
        goal_graphs=[g0, g1],
    )
    n0 = int(g0.node_ids.numel())
    n1 = int(g1.node_ids.numel())
    assert out["goal_batch"].tolist() == [0] * n0 + [1] * n1
    # goal_node_features = concat of the two goals' node ids, fringe order.
    assert out["goal_node_features"].tolist() == (
        g0.node_ids.tolist() + g1.node_ids.tolist()
    )
    # goal_edge_index second fringe shifted by n0 nodes.
    expected_ei = torch.cat(
        [g0.edge_index, g1.edge_index + n0], dim=1
    )
    assert out["goal_edge_index"].tolist() == expected_ei.tolist()
    assert out["goal_edge_attr"].tolist() == (
        g0.edge_attr.tolist() + g1.edge_attr.tolist()
    )
    print("PASS test_goal_batch_maps_to_fringe_index")


def test_single_fringe_goal_batch_all_zeros() -> None:
    """Inference parity: one goal graph per call => goal_batch all-zeros."""
    caches, goals = _caches_and_goals()
    out = pack_fringe_batch([(caches[0], [0, 1])], goal_graphs=[goals[0]])
    gb = out["goal_batch"]
    assert gb.numel() == int(goals[0].node_ids.numel())
    assert torch.all(gb == 0).item()
    print("PASS test_single_fringe_goal_batch_all_zeros")


def test_flatcache_matches_reference_packer() -> None:
    """Vectorized GlobalFlatCache.pack == slow pack_fringe_batch, goals incl."""
    caches, goals = _caches_and_goals()
    flat = GlobalFlatCache(caches, device="cpu", goal_graphs=goals)

    # Fringe 0 -> inst 0 states [0,1]; fringe 1 -> inst 1 states [1,0].
    fringes = [(0, [0, 1]), (1, [1, 0])]
    state_gids = torch.tensor(
        [flat.gid(i, s) for i, states in fringes for s in states],
        dtype=torch.long,
    )
    fringe_lens = torch.tensor([len(s) for _, s in fringes], dtype=torch.long)
    fringe_inst = torch.tensor([i for i, _ in fringes], dtype=torch.long)
    fast = flat.pack(state_gids, fringe_lens, fringe_inst=fringe_inst)

    ref = pack_fringe_batch(
        [(caches[0], [0, 1]), (caches[1], [1, 0])],
        goal_graphs=[goals[0], goals[1]],
    )
    for k in (
        "node_features",
        "edge_index",
        "edge_attr",
        "membership",
        "candidate_batch",
        "goal_node_features",
        "goal_edge_index",
        "goal_edge_attr",
        "goal_batch",
    ):
        assert fast[k].tolist() == ref[k].tolist(), f"mismatch on {k}"
    print("PASS test_flatcache_matches_reference_packer")


def test_merged_packing_unchanged() -> None:
    """No goal graphs => no goal_* keys; tensors identical to pre-change."""
    caches, _ = _caches_and_goals()
    out = pack_fringe_batch([(caches[0], [0, 1]), (caches[1], [0, 1])])
    for k in (
        "goal_node_features",
        "goal_edge_index",
        "goal_edge_attr",
        "goal_batch",
    ):
        assert k not in out, f"merged pack unexpectedly emitted {k}"
    # GlobalFlatCache without goal graphs must also stay goal-free.
    flat = GlobalFlatCache(caches, device="cpu")
    assert flat.has_goal is False
    state_gids = torch.tensor(
        [flat.gid(0, 0), flat.gid(0, 1)], dtype=torch.long
    )
    packed = flat.pack(state_gids, torch.tensor([2], dtype=torch.long))
    assert "goal_batch" not in packed
    print("PASS test_merged_packing_unchanged")


def test_separated_onnx_input_names() -> None:
    """trainer.to_onnx separated export emits the 9-input contract in order."""
    import onnx

    model = FrontierPolicyNetwork(
        node_input_dim=1,
        hidden_dim=16,
        gnn_layers=2,
        conv_type="gine",
        pooling_type="mean",
        dataset_type="HASHED",
        edge_emb_dim=8,
        num_edge_labels=128,
        num_node_labels=4096,
        use_global_context=True,
        mlp_depth=2,
        use_goal_separate_input=True,
    )
    trainer = RLFrontierTrainer(model=model, kind_of_data="separated", device="cpu")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sep.onnx"
        trainer.to_onnx(out, node_input_dim=1, onnx_frontier_size=8)
        m = onnx.load(str(out))
        names = [i.name for i in m.graph.input]
    assert names == CONTRACT_INPUT_NAMES, names
    print("PASS test_separated_onnx_input_names ->", names)


if __name__ == "__main__":
    test_goal_batch_maps_to_fringe_index()
    test_single_fringe_goal_batch_all_zeros()
    test_flatcache_matches_reference_packer()
    test_merged_packing_unchanged()
    test_separated_onnx_input_names()
    print("\nALL SEPARATED-GOAL PARITY TESTS PASSED")
