"""Packing beams into batched graphs. Shared by the trainer and the baseline.

REPRESENTATION-AGNOSTIC. `node_features` are opaque int64s: whatever the C++
writes. Nothing here branches on HASHED vs BITMASK vs MAPPED, and nothing here may
ever start. When the BITMASK path is enabled in `FringeEvalRL`, the only thing that
changes is which integers arrive — no code in this module, the trainer, the
telemetry, the selection or the export.

Two packing modes, and they are different code paths in the model
(`_contextualize`, and `FringeAttention.forward` vs `forward_single`):

  pack_single : one fringe, mask of length K -- what the planner does at inference
  pack_batch  : many fringes in one graph via `candidate_batch` -- what training does

`test_batched_path_matches_single_path` asserts they agree to 1e-5, which is what
makes it safe to train through one and deploy through the other.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .encoder import InstanceCache, pack_fringe


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pack_single(
    cache: InstanceCache,
    beam: Sequence[int],
    fringe_size: int,
    device,
) -> Dict[str, torch.Tensor]:
    """One fringe, exactly as `FringeEvalRL::fringe_to_tensor_minimal` builds it."""
    p = pack_fringe(cache, list(beam), fringe_size)
    return {k: v.to(device) for k, v in p.items()}


def pack_batch(
    caches: Dict[str, InstanceCache],
    picks: Sequence[Tuple[str, Sequence[int]]],
    device,
) -> Dict[str, torch.Tensor]:
    """Many beams in ONE graph.

    `membership` continues across beams (global slot ids); `candidate_batch[j]`
    says which beam slot j belongs to. Also returns `slot_offset`, the index of
    each beam's first slot, so a caller can gather the slot for a chosen action.
    """
    nf, ei, ea, mem, cb = [], [], [], [], []
    slot_offset: List[int] = []
    node_off, slot_off = 0, 0
    for i, (name, beam) in enumerate(picks):
        p = pack_fringe(caches[name], list(beam), len(beam))
        nf.append(p["node_features"])
        ei.append(p["edge_index"] + node_off)
        ea.append(p["edge_attr"])
        mem.append(p["membership"] + slot_off)
        cb.append(torch.full((len(beam),), i, dtype=torch.int64))
        slot_offset.append(slot_off)
        node_off += int(p["node_features"].numel())
        slot_off += len(beam)
    return {
        "node_features": torch.cat(nf).to(device),
        "edge_index": torch.cat(ei, dim=1).to(device),
        "edge_attr": torch.cat(ea).to(device),
        "membership": torch.cat(mem).to(device),
        "candidate_batch": torch.cat(cb).to(device),
        "slot_offset": torch.tensor(slot_offset, dtype=torch.int64, device=device),
        "beam_sizes": torch.tensor([len(b) for _, b in picks], dtype=torch.int64, device=device),
        "n_slots": slot_off,
    }


def segment_max(values: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Max of `values` within each segment of `seg`. |A(s)| <= F, so this is the
    whole reason Double-DQN is trivial here and IQL is unnecessary."""
    out = values.new_full((n_seg,), float("-inf"))
    return out.scatter_reduce(0, seg, values, reduce="amax", include_self=True)


def segment_argmax(values: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Index (into `values`) of the max within each segment. Used for Double-DQN's
    online-argmax / target-evaluate split."""
    mx = segment_max(values, seg, n_seg)
    is_max = values == mx[seg]
    idx = torch.arange(values.numel(), device=values.device)
    big = torch.where(is_max, idx, torch.full_like(idx, values.numel()))
    out = big.new_full((n_seg,), values.numel())
    out = out.scatter_reduce(0, seg, big, reduce="amin", include_self=True)
    return out


def segment_logsumexp(values: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """logsumexp within each segment -- the CQL term. Also trivial because
    |A(s)| <= F."""
    mx = segment_max(values, seg, n_seg)
    shifted = torch.exp(values - mx[seg])
    summed = torch.zeros(n_seg, device=values.device, dtype=values.dtype)
    summed = summed.scatter_add(0, seg, shifted)
    return mx + torch.log(summed.clamp_min(1e-12))
