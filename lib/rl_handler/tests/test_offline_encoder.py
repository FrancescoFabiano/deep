"""Phase-1 verification for src/offline/encoder.py.

1. Hand-derived contract check: pack a 3-state fringe from tiny DOT strings
   and compare every tensor against values derived by hand from
   FringeEvalRL::fringe_to_tensor_minimal.
2. Parser parity vs the production pydot loader on real DOT files.
3. ONNX compatibility: feed a packed fringe to the deployed
   frontier_policy_32.onnx and check logits shape/masking.
4. Alone-vs-batch invariance through the real network.

Run from lib/rl_handler:  ../../.venv/bin/python tests/test_offline_encoder.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.encoder import (  # noqa: E402
    InstanceCache,
    pack_fringe,
    pack_fringe_batch,
    parse_dot_fast,
)

REPO = Path(__file__).resolve().parents[3]

DOT_A = """digraph G {
  5 -> -7 [label="2"];
  -7 -> 5 [label="3"];
}
"""
DOT_B = """digraph G {
  1 -> 2 [label="4"];
}
"""
DOT_C = """digraph G {
  -9083301410840355011 -> 42 [label="1"];
}
"""


def test_hand_derived_contract() -> None:
    states = [parse_dot_fast(s) for s in (DOT_A, DOT_B, DOT_C)]
    assert all(s is not None for s in states)
    cache = InstanceCache(states)  # type: ignore[arg-type]
    out = pack_fringe(cache, [0, 1, 2], fringe_size=8)

    # Hand-derived (C++ packing): node order = first appearance per state,
    # concatenated in fringe order; edges shifted by cumulative node count.
    assert out["node_features"].tolist() == [5, -7, 1, 2, -9083301410840355011, 42]
    assert out["edge_index"].tolist() == [[0, 1, 2, 4], [1, 0, 3, 5]]
    assert out["edge_attr"].tolist() == [2, 3, 4, 1]
    assert out["membership"].tolist() == [0, 0, 1, 1, 2, 2]
    assert out["mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert out["node_features"].dtype == torch.int64
    assert out["edge_index"].dtype == torch.int64
    assert out["edge_attr"].dtype == torch.int64
    assert out["membership"].dtype == torch.int64
    assert out["mask"].dtype == torch.uint8
    print("[1] hand-derived contract check: OK")


def test_parser_parity_vs_pydot(sample_files: list[Path]) -> None:
    # Compare against the production no-pyg loader used to materialize
    # evaluation frontiers (same loader family that trained frontier_policy).
    from src.common.no_pyg_loader import load_graph_tensors_no_pyg

    for p in sample_files:
        fast = parse_dot_fast(p.read_text())
        assert fast is not None, f"fast parser rejected {p}"
        ref = load_graph_tensors_no_pyg(str(p), dataset_type="HASHED")
        assert fast.node_ids.tolist() == ref.node_names.view(-1).tolist(), p
        assert fast.edge_index.tolist() == ref.edge_index.tolist(), p
        assert fast.edge_attr.tolist() == ref.edge_attr.view(-1).tolist(), p
    print(f"[2] parser parity vs pydot loader on {len(sample_files)} files: OK")


def test_onnx_compat(cache: InstanceCache) -> None:
    import numpy as np
    import onnxruntime as ort

    model_path = (
        REPO / "exp/rl_exp/batch0_merged/_models/CC/frontier_policy_32.onnx"
    )
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    k = 5
    out = pack_fringe(cache, list(range(k)), fringe_size=32)
    feeds = {
        "node_features": out["node_features"].numpy(),
        "edge_index": out["edge_index"].numpy(),
        "edge_attr": out["edge_attr"].numpy(),
        "membership": out["membership"].numpy(),
        "mask": out["mask"].numpy(),
    }
    (logits,) = sess.run(["logits"], feeds)
    assert logits.shape == (32,), logits.shape
    active, padded = logits[:k], logits[k:]
    assert np.all(np.isfinite(active))
    assert np.all(padded <= -1e8), "padding slots must be masked to -1e9"
    assert len(set(np.round(active, 6))) > 1, "active logits should not be constant"
    print(f"[3] ONNX compat with deployed frontier_policy_32.onnx: OK "
          f"(active logits range [{active.min():.4f}, {active.max():.4f}])")


def test_alone_vs_batch(cache: InstanceCache) -> None:
    """A state's pooled embedding must not depend on fringe co-members.

    Verified through the real network with global context disabled at the
    embedding level: we compare per-candidate pooled embeddings (encoder +
    pooling), which is the part the packing affects.  With global context the
    *logits* legitimately depend on the fringe (documented in DESIGN.md).
    """
    from src.models.frontier_policy import FrontierPolicyNetwork

    torch.manual_seed(0)
    model = FrontierPolicyNetwork(
        node_input_dim=1, hidden_dim=32, gnn_layers=2, conv_type="gine",
        pooling_type="mean", dataset_type="HASHED", edge_emb_dim=8,
        num_edge_labels=128, num_node_labels=4096,
        use_global_context=True, mlp_depth=1,
    ).eval()

    def pooled(fringe_indices: list[int]) -> torch.Tensor:
        out = pack_fringe_batch([(cache, fringe_indices)])
        with torch.no_grad():
            emb = model.encoder(
                out["node_features"], out["edge_index"], out["edge_attr"]
            )
            return model._pool_nodes(
                emb, out["membership"],
                expected_size=int(out["candidate_batch"].numel()),
            )

    z_alone = pooled([7])
    z_batch = pooled([3, 7, 11, 2])
    diff = (z_alone[0] - z_batch[1]).abs().max().item()
    assert diff < 1e-5, f"alone-vs-batch embedding diff {diff}"
    print(f"[4] alone-vs-batch pooled-embedding invariance: OK (max diff {diff:.2e})")


def test_flat_cache_equivalence(cache: InstanceCache) -> None:
    """GlobalFlatCache.pack must reproduce pack_fringe_batch bit-exactly."""
    from src.offline.encoder import GlobalFlatCache, segment_argmax

    flat = GlobalFlatCache([cache], device="cpu")
    rng = random.Random(3)
    fringes = [
        rng.sample(range(len(cache.states)), rng.randint(1, 32)) for _ in range(17)
    ]
    ref = pack_fringe_batch([(cache, f) for f in fringes])
    gids = torch.tensor([s for f in fringes for s in f], dtype=torch.long)
    lens = torch.tensor([len(f) for f in fringes], dtype=torch.long)
    fast = flat.pack(gids, lens)
    for key in ("node_features", "edge_index", "edge_attr", "membership",
                "candidate_batch", "fringe_ptr"):
        assert torch.equal(ref[key], fast[key]), key
    # segment_argmax vs python argmax per segment
    vals = torch.randn(int(ref["fringe_ptr"][-1]))
    got = segment_argmax(vals, ref["fringe_ptr"])
    for i in range(len(fringes)):
        s, e = int(ref["fringe_ptr"][i]), int(ref["fringe_ptr"][i + 1])
        assert int(got[i]) == s + int(torch.argmax(vals[s:e]))
    print("[6] GlobalFlatCache.pack + segment_argmax equivalence: OK")


def benchmark(cache: InstanceCache) -> None:
    rng = random.Random(0)
    print("[5] packing benchmark (states/sec, CPU):")
    for fringe in (8, 64, 512):
        pool = list(range(len(cache.states)))
        n_rep = max(1, 4096 // fringe)
        batches = [rng.sample(pool, fringe) for _ in range(n_rep)]
        t0 = time.time()
        for b in batches:
            # fringe_size >= len(b); for >32 this is a training-style pack
            pack_fringe(cache, b, fringe_size=max(fringe, 32))
        dt = time.time() - t0
        sps = n_rep * fringe / dt
        print(f"    fringe={fringe:4d}: {sps:9.0f} states/s ({dt / n_rep * 1e3:.2f} ms/fringe)")


def main() -> None:
    test_hand_derived_contract()

    inst_dir = REPO / "out/NN/Training/CC_3_2_3__pl_7/RawFiles/hash_merged"
    files = sorted(inst_dir.glob("*.dot"))[:2000]
    rng = random.Random(1)
    parity_sample = rng.sample(files, 25)
    test_parser_parity_vs_pydot(parity_sample)

    cache = InstanceCache.from_paths([str(p) for p in files], cache_file=None)
    test_onnx_compat(cache)
    test_alone_vs_batch(cache)
    test_flat_cache_equivalence(cache)
    benchmark(cache)


if __name__ == "__main__":
    main()
