"""Pydot-based DOT -> tensor loader without torch_geometric ("no-pyg").

This module owns the reference DOT loader from the supervised pipeline that
produced the deployed production frontier_policy models (that pipeline was
removed; recover via git history, see README).  The offline pipeline keeps
its own fast regex parser (src/offline/encoder.py) and uses this loader ONLY
as the parity reference in tests/test_offline_encoder.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pydot
import torch

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

VALID_DATASET_TYPES = {"HASHED", "MAPPED", "BITMASK"}

# Signed-id datasets may contain bare negative node IDs (e.g. -123); pydot
# requires quoted negatives, so sanitize before parsing.
BARE_NEGATIVE_INT_RE = re.compile(r'(?<!["\w])-([0-9]+)(?!["\w])')


def strip_quotes(value: Any) -> str:
    return str(value).replace('"', "").strip()


def parse_numeric_node_label(node_obj: object) -> int:
    if isinstance(node_obj, bool):
        raise TypeError("Boolean node labels are not supported.")
    if isinstance(node_obj, int):
        return int(node_obj)
    if isinstance(node_obj, float):
        if not float(node_obj).is_integer():
            raise ValueError(f"Node label '{node_obj}' is not an integer.")
        return int(node_obj)
    if isinstance(node_obj, str):
        text = node_obj.strip()
        try:
            return int(text)
        except ValueError:
            parsed = float(text)
            if not parsed.is_integer():
                raise ValueError(f"Node label '{node_obj}' is not an integer.")
            return int(parsed)
    raise TypeError(f"Unsupported node label type: {type(node_obj)}")


def validate_int64_range(value: int, *, context: str) -> int:
    if value < I64_MIN or value > I64_MAX:
        raise ValueError(f"{context} is out of int64 range [{I64_MIN}, {I64_MAX}].")
    return int(value)


def fold_uint64_to_int64(value: int, *, context: str) -> int:
    """Fold IDs in [-2^63, 2^64-1] to int64 preserving the 64-bit pattern.

    Mirrors src/offline/encoder.uint64_ids_to_int64 (two's complement): HASHED
    ids are raw 64-bit hashes, so ~half of them exceed 2^63-1. The previous
    strict int64 check made this reference loader reject typical HASHED data --
    the exact folding path it exists to parity-test.
    """
    if value < I64_MIN or value > (1 << 64) - 1:
        raise ValueError(f"{context} is out of foldable range [-2^63, 2^64-1].")
    return int(value - (1 << 64)) if value > I64_MAX else int(value)


@dataclass
class EvalGraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_names: torch.Tensor


def load_graph_tensors_no_pyg(path: str, dataset_type: str) -> EvalGraphTensors:
    dataset_type_norm = str(dataset_type).upper()
    if dataset_type_norm not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of {sorted(VALID_DATASET_TYPES)}."
        )

    dot_src = Path(path).read_text()
    dot_src = BARE_NEGATIVE_INT_RE.sub(r'"-\1"', dot_src)
    parsed = pydot.graph_from_dot_data(dot_src)
    if not parsed:
        raise ValueError(f"Failed to parse DOT graph: {path}")
    dot = parsed[0]
    graph_nx = nx.nx_pydot.from_pydot(dot)

    nodes = list(graph_nx.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}

    if dataset_type_norm == "BITMASK":
        explicit_len = graph_nx.graph.get("bit_len", None)
        if explicit_len is not None:
            explicit_len = int(strip_quotes(explicit_len))

        first = nodes[0] if nodes else None
        inferred_len = len(first) if isinstance(first, (str, list, tuple)) else None
        bit_len = explicit_len if explicit_len is not None else inferred_len
        if bit_len is None:
            raise ValueError("Cannot infer bit length from graph nodes.")

        def _to_bits(node_obj: object, expected_len: int | None) -> list[int]:
            if isinstance(node_obj, str):
                s = node_obj.strip()
                if not set(s) <= {"0", "1"}:
                    raise ValueError(f"Node '{node_obj}' is not a bitstring.")
                if expected_len is not None and len(s) != expected_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return [int(ch) for ch in s]
            if isinstance(node_obj, (list, tuple)):
                bits = [int(x) for x in node_obj]
                if not set(bits) <= {0, 1}:
                    raise ValueError(f"Node '{node_obj}' has non-binary values.")
                if expected_len is not None and len(bits) != expected_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return bits
            if isinstance(node_obj, int):
                if expected_len is None:
                    raise ValueError("bit_len is required for int node labels.")
                return [int(ch) for ch in format(node_obj, f"0{expected_len}b")]
            raise TypeError(f"Unsupported node label type: {type(node_obj)}")

        rows = [_to_bits(node, bit_len) for node in nodes]
        node_bits = torch.tensor(rows, dtype=torch.bool)
        node_features = node_bits.to(torch.float32)
        node_names = node_features.clone()
    elif dataset_type_norm == "HASHED":
        raw_ids = [
            fold_uint64_to_int64(
                parse_numeric_node_label(node),
                context=f"Node '{node}'",
            )
            for node in nodes
        ]
        node_names = torch.tensor(raw_ids, dtype=torch.int64)
        node_features = node_names.view(-1, 1).to(torch.int64)
    else:
        raw_ids = [parse_numeric_node_label(node) for node in nodes]
        for node, raw_id in zip(nodes, raw_ids):
            if raw_id < I64_MIN or raw_id > I64_MAX:
                raise ValueError(
                    f"Node '{node}' is out of int64 range [{I64_MIN}, {I64_MAX}] "
                    "for MAPPED dataset."
                )
        node_names = torch.tensor(raw_ids, dtype=torch.int64)
        node_features = node_names.view(-1, 1).to(torch.int64)

    edge_rows: list[list[int]] = []
    edge_attrs: list[list[int]] = []
    for src, dst, edge_data in graph_nx.edges(data=True):
        src_idx = int(node_to_idx[src])
        dst_idx = int(node_to_idx[dst])
        edge_label = int(strip_quotes(edge_data.get("label", "0")))
        edge_rows.append([src_idx, dst_idx])
        edge_attrs.append([edge_label])

    if edge_rows:
        edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.int64)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.int64)

    return EvalGraphTensors(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_names=node_names,
    )
