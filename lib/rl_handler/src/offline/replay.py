"""Compact replay buffer: transitions hold state *indices*, never tensors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Transition:
    inst: int  # index into the trainer's instance list
    fringe: Tuple[int, ...]  # state ids (slot order)
    action: int  # slot index expanded
    reward: float
    next_fringe: Tuple[int, ...]  # empty when done
    done: bool
    # Optional provenance tag = the fringe-composition regime that produced this
    # transition (P1: dfs/bfs/hfs/random). None for the single-source
    # pipeline, so existing constructions are unchanged. Never read by training
    # (_update is regime-agnostic) — it exists only for per-regime audits.
    regime: Optional[str] = None
    # Per-slot padded-ness, aligned with `fringe` slot order: True iff that slot
    # was PAD-FILLED (reservoir/closed top-up on an F>fmax instance) rather than a
    # live frontier member. Padded-ness is a property of (this fringe, this slot)
    # — the SAME node can be live in one fringe and pad-fill in another — so it
    # cannot be recovered from the id or d* downstream and must be carried here.
    # None => treated as all-False (the single-source / faithful path), so the
    # order-aux includes every slot exactly as before. The value objective never
    # reads it (padded slots stay value-supervised); only the order-aux masks it.
    pad_mask: Optional[Tuple[bool, ...]] = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.buf: List[Optional[Transition]] = [None] * self.capacity
        self.idx = 0
        self.size = 0
        self.rng = random.Random(seed)

    def push(self, t: Transition) -> None:
        self.buf[self.idx] = t
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        idxs = [self.rng.randrange(self.size) for _ in range(batch_size)]
        return [self.buf[i] for i in idxs]  # type: ignore[misc]

    def __len__(self) -> int:
        return self.size
