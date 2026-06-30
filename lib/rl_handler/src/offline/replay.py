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
