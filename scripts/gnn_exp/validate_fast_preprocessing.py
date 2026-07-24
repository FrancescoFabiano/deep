#!/usr/bin/env python3
"""Validate the fast DOT->PyG path, including signed (negative) node IDs.

The planner now prints node-ID hashes as *signed* int64, so DOT files may
contain bare negative numerals as node IDs.  pydot's grammar rejects those,
which means the old fast-vs-pydot comparison is only possible on files whose
IDs are all non-negative; on negative-ID files the fast parser is validated
against a small independent reference parser defined in this script.

Checks:
  1a. NEW-style files (negative IDs): _parse_dot_fast parses them, node_names
      dtype is int64, and node order / edge multiset / labels match an
      independent ad-hoc regex parser.
  1b. OLD-style files (all IDs non-negative): field-by-field equality vs the
      pydot path (_load_dot + _nx_to_pyg), as before.  Falls back to synthetic
      non-negative files if no old-style files remain.
  2.  Synthetic BITMASK graphs: old vs fast path (node_bits, node_bitint, ...).
  3.  uint64_ids_to_int64_tensor vs the scalar uint64_to_signed_int64 on
      boundary values of the accepted range [-2^63, 2^64-1], incl. errors.
      Round-trip: boundary ids written into a synthetic DOT string come back
      as exactly the expected int64 values from _parse_dot_fast.
  4.  normalize_int64_ids: signed and unsigned form of the same hash produce
      identical normalized output.
  5.  Model forward pass on batches built from both paths (old-style files)
      -> identical output.

Usage:
  .venv/bin/python scripts/gnn_exp/validate_fast_preprocessing.py [--n 300]
"""
import argparse
import random
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

import torch  # noqa: E402

from src.models.distance_estimator import normalize_int64_ids  # noqa: E402
from src.utils import (  # noqa: E402
    _load_dot,
    _nx_to_pyg,
    _parse_dot_fast,
    graph_collate_fn,
    seed_everything,
    select_model,
    uint64_ids_to_int64_tensor,
    uint64_to_signed_int64,
)

DATA_ROOT = REPO_ROOT / "exp/gnn_exp/batch0/_models/CC/training_data"

# Bare negative numeral used as a node ID (either endpoint of an edge).
_NEG_ID_RE = re.compile(r"(?:^|\s)-\d+\s*->|->\s*-\d+")

# Independent reference parser: one edge per line, signed integer IDs.
# Deliberately minimal and separate from src.utils._DOT_EDGE_RE.
_REF_EDGE_RE = re.compile(r'^\s*(-?\d+)\s*->\s*(-?\d+)\s*\[label="(-?\d+)"\]\s*;\s*$')


def _reference_parse(src: str):
    """Return (nodes_in_first_appearance_order, edge_list) from a planner DOT.

    Edge list entries are (u_id, v_id, label) with IDs folded to int64 via the
    scalar reference uint64_to_signed_int64 (so unsigned-form files compare
    on the same footing).
    """
    nodes, edges = [], []
    seen = set()
    for line in src.splitlines():
        m = _REF_EDGE_RE.match(line)
        if m is None:
            continue
        u, v, lab = (int(m.group(i)) for i in (1, 2, 3))
        u, v = uint64_to_signed_int64([u, v])
        for n in (u, v):
            if n not in seen:
                seen.add(n)
                nodes.append(n)
        edges.append((u, v, lab))
    return nodes, edges


def assert_data_equal(old, new, path, bitmask=False):
    assert old.num_nodes == new.num_nodes, f"{path}: num_nodes"
    assert torch.equal(old.edge_index, new.edge_index), f"{path}: edge_index"
    assert old.edge_index.dtype == new.edge_index.dtype, f"{path}: edge_index dtype"
    assert torch.equal(old.edge_attr, new.edge_attr), f"{path}: edge_attr"
    assert old.edge_attr.dtype == new.edge_attr.dtype, f"{path}: edge_attr dtype"
    assert torch.equal(old.edge_label, new.edge_label), f"{path}: edge_label"
    assert old.label == new.label, f"{path}: label list"
    assert torch.equal(old.shape, new.shape), f"{path}: shape"
    assert torch.equal(old.node_names, new.node_names), f"{path}: node_names"
    assert old.node_names.dtype == new.node_names.dtype, f"{path}: node_names dtype"
    if bitmask:
        assert torch.equal(old.node_bits, new.node_bits), f"{path}: node_bits"
        assert old.node_bits.dtype == new.node_bits.dtype, f"{path}: node_bits dtype"
        assert torch.equal(old.node_bitint, new.node_bitint), f"{path}: node_bitint"
        assert old.node_labels == new.node_labels, f"{path}: node_labels"


def _collect_files(n):
    """Sample up to n negative-ID files and n non-negative files (with text)."""
    files = []
    for inst in sorted(DATA_ROOT.iterdir()):
        merged = inst / "RawFiles" / "hash_merged"
        if merged.is_dir():
            files += sorted(merged.glob("*.dot"))
    rng = random.Random(42)
    rng.shuffle(files)
    neg, pos = [], []
    for p in files:
        if len(neg) >= n and len(pos) >= n:
            break
        src = p.read_text()
        bucket = neg if _NEG_ID_RE.search(src) else pos
        if len(bucket) < n:
            bucket.append((p, src))
    return neg, pos


def check_negative_id_files(neg_files):
    print(f"[1a] HASHED (signed IDs): fast path vs independent reference parser "
          f"on {len(neg_files)} files...")
    assert neg_files, "no negative-ID files found — expected new-style data"
    for p, src in neg_files:
        data = _parse_dot_fast(src)
        assert data is not None, f"{p}: fast parser rejected a planner file"
        assert data.node_names.dtype == torch.int64, f"{p}: node_names dtype"

        ref_nodes, ref_edges = _reference_parse(src)
        got_nodes = data.node_names.tolist()
        assert got_nodes == ref_nodes, f"{p}: node values/order mismatch"
        # Edge multiset (resolved to node IDs) must match the file, regardless
        # of the MultiDiGraph reordering the fast parser applies.
        got_edges = sorted(
            (got_nodes[u], got_nodes[v], int(lab))
            for u, v, lab in zip(
                data.edge_index[0].tolist(),
                data.edge_index[1].tolist(),
                data.edge_label.tolist(),
            )
        )
        assert got_edges == sorted(ref_edges), f"{p}: edge multiset mismatch"
        assert data.edge_attr.dtype == torch.float32, f"{p}: edge_attr dtype"
        assert data.num_nodes == len(ref_nodes), f"{p}: num_nodes"
    print("    OK — parsed, int64 node_names, nodes/edges match reference.")


def check_old_style_files(pos_files):
    samples_old, samples_new = [], []
    if pos_files:
        print(f"[1b] HASHED (unsigned IDs): old pydot path vs fast path on "
              f"{len(pos_files)} files...")
        pairs = [(p, src) for p, src in pos_files]
    else:
        print("[1b] HASHED: no non-negative real files left — using synthetic "
              "unsigned-ID graphs...")
        rng = random.Random(1)
        pairs = []
        td = tempfile.mkdtemp()
        for i in range(30):
            nodes = [rng.randrange(0, 2**64) for _ in range(rng.randint(2, 10))]
            lines = ["digraph G {"]
            for _ in range(rng.randint(1, 25)):
                u, v = rng.choice(nodes), rng.choice(nodes)
                lines.append(f'  {u} -> {v} [label="{rng.randint(0, 12)}"];')
            lines.append("}")
            p = Path(td) / f"syn_{i}.dot"
            src = "\n".join(lines) + "\n"
            p.write_text(src)
            pairs.append((p, src))
    for p, src in pairs:
        old = _nx_to_pyg(_load_dot(p))
        new = _parse_dot_fast(src)
        assert new is not None, f"{p}: fast parser fell back unexpectedly"
        assert_data_equal(old, new, p)
        old.name = new.name = p.stem
        samples_old.append({"state_graph": old, "depth": torch.tensor([3]),
                            "target": torch.tensor([2.0])})
        samples_new.append({"state_graph": new, "depth": torch.tensor([3]),
                            "target": torch.tensor([2.0])})
    print("    OK — all fields identical (values + dtypes).")
    return samples_old, samples_new


def check_bitmask():
    print("[2] BITMASK: comparing old vs fast path on synthetic bitstring graphs...")
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as td:
        for i in range(20):
            bit_len = rng.choice([8, 42, 64])
            nodes = list({"".join(rng.choice("01") for _ in range(bit_len))
                          for _ in range(rng.randint(2, 12))})
            lines = ["digraph G {"]
            for _ in range(rng.randint(1, 30)):
                u, v = rng.choice(nodes), rng.choice(nodes)
                lines.append(f'  {u} -> {v} [label="{rng.randint(0, 12)}"];')
            lines.append("}")
            p = Path(td) / f"bm_{i}.dot"
            p.write_text("\n".join(lines) + "\n")
            old = _nx_to_pyg(_load_dot(p), bitmask=True)
            new = _parse_dot_fast(p.read_text(), bitmask=True)
            assert new is not None, f"{p}: fast parser fell back unexpectedly"
            assert_data_equal(old, new, p, bitmask=True)
    print("    OK — node_bits / node_bitint / node_labels identical.")


def check_int64_conversion():
    print("[3] int64 conversion: boundary values, errors, DOT round-trip...")
    # Full accepted range [-2^63, 2^64-1]: signed values pass through,
    # unsigned values fold via two's complement.
    vals = [-(2**63), -1, 0, 1, 2**24, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1,
            17262573695215226039, -6185577020032993166]
    ref = torch.tensor(uint64_to_signed_int64(vals), dtype=torch.int64)
    fast = uint64_ids_to_int64_tensor(vals)
    assert torch.equal(ref, fast) and fast.dtype == torch.int64
    # Signed values must be unchanged; unsigned must fold to the same bits.
    assert uint64_to_signed_int64([-(2**63), -1, 0, 2**63 - 1]) == \
        [-(2**63), -1, 0, 2**63 - 1]
    assert uint64_to_signed_int64([2**63, 2**64 - 1]) == [-(2**63), -1]
    for bad in ([-(2**63) - 1], [2**64]):
        for fn in (uint64_to_signed_int64, uint64_ids_to_int64_tensor):
            try:
                fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__}({bad}): expected ValueError")

    # Round-trip through a synthetic DOT string: signed boundary ids must come
    # back exactly; unsigned forms must come back as their signed fold.
    def roundtrip(ids):
        lines = ["digraph G {"]
        for u, v in zip(ids, ids[1:]):
            lines.append(f'  {u} -> {v} [label="1"];')
        lines.append("}")
        data = _parse_dot_fast("\n".join(lines) + "\n")
        assert data is not None, f"fast parser rejected ids {ids}"
        assert data.node_names.dtype == torch.int64
        return data.node_names.tolist()

    signed_ids = [-(2**63), -1, 0, 1, 2**63 - 1]
    assert roundtrip(signed_ids) == signed_ids
    assert roundtrip([2**63, 2**64 - 1]) == [-(2**63), -1]
    print("    OK — conversions, errors, and DOT round-trip all exact.")


def check_normalize_equivalence():
    print("[4] normalize_int64_ids: signed vs unsigned form of the same hash...")
    unsigned = [2**63, 2**63 + 1, 2**64 - 1, 17262573695215226039]
    signed = uint64_to_signed_int64(unsigned)
    out_u = normalize_int64_ids(
        uint64_ids_to_int64_tensor(unsigned).to(torch.float64))
    out_s = normalize_int64_ids(
        uint64_ids_to_int64_tensor(signed).to(torch.float64))
    assert torch.equal(out_u, out_s), "normalized outputs differ between forms"
    print("    OK — identical normalized output for both printed forms.")


def check_forward(samples_old, samples_new):
    # NOTE: comparison runs on CPU.  On CUDA the scatter-add inside GINEConv /
    # global_mean_pool is non-deterministic (atomicAdd ordering), so even the
    # *same* batch run twice differs by ~1 ulp — that is a property of the
    # GPU kernels, not of the preprocessing.
    print("[5] Forward pass (CPU): model(batch_old) vs model(batch_new)...")
    seed_everything(42)
    m = select_model("distance_estimator", use_goal=False, use_depth=False)
    core = m.model.cpu().eval()
    with torch.no_grad():
        for i in range(0, len(samples_old), 64):
            out_old = core(graph_collate_fn(samples_old[i : i + 64]))
            out_new = core(graph_collate_fn(samples_new[i : i + 64]))
            assert torch.equal(out_old, out_new), "forward outputs differ"
    print(f"    OK — outputs bit-identical on {len(samples_old)} samples.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    neg_files, pos_files = _collect_files(args.n)
    check_negative_id_files(neg_files)
    samples_old, samples_new = check_old_style_files(pos_files)
    check_bitmask()
    check_int64_conversion()
    check_normalize_equivalence()
    check_forward(samples_old, samples_new)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
