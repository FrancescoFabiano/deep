"""Compact replay buffer: transitions hold state *indices*, never tensors."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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
    # Pad-to-F support: number of LEADING slots in `fringe` / `next_fringe` that
    # are OPEN (live beam). Slots at index >= n_open are closed padding. DIAGNOSTIC
    # ONLY — nothing in the loss path reads these: padded closed states are treated
    # identically to open states (selectable, in the Q-loss, in the bootstrap max),
    # so no code masks or excludes by n_open. Kept so the occupancy CSV can report
    # open-vs-padded composition. None means "no padding" (every slot open).
    n_open: Optional[int] = None
    next_n_open: Optional[int] = None
    # d* (distance-to-goal) of the EXPANDED state = fringe[action]. Used only by
    # d*-stratified replay sampling to balance draws across d* buckets; the loss
    # path never reads it. None means "unlabelled" (bucketed as max_dstar).
    dstar: Optional[int] = None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int = 0,
        stratified: bool = False,
        max_dstar: int = 10,
    ):
        self.capacity = int(capacity)
        self.buf: List[Optional[Transition]] = [None] * self.capacity
        self.idx = 0
        self.size = 0
        self.rng = random.Random(seed)
        # d*-stratified sampling: draw an equal share of the batch from each
        # non-empty d* bucket, countering the near-goal skew of the raw data.
        self.stratified = bool(stratified)
        self.max_dstar = int(max_dstar)
        self._buckets: Dict[int, List[int]] = defaultdict(list)

    def push(self, t: Transition) -> None:
        self.buf[self.idx] = t
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _bucket_of(self, t: Transition) -> int:
        d = t.dstar if t.dstar is not None else self.max_dstar
        return min(int(d), self.max_dstar)

    def _rebuild_buckets(self) -> None:
        # O(size) rebuild over the live slots (buf holds None in unfilled slots
        # and stale entries in overwritten circular slots, so we never index past
        # self.size). Cheap: size <= capacity <= ~50k.
        self._buckets = defaultdict(list)
        for i in range(self.size):
            self._buckets[self._bucket_of(self.buf[i])].append(i)  # type: ignore[arg-type]

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.stratified:
            # Original uniform sampling (with replacement) — byte-identical to
            # the pre-stratified behaviour; keep the exact same RNG draws.
            idxs = [self.rng.randrange(self.size) for _ in range(batch_size)]
            return [self.buf[i] for i in idxs]  # type: ignore[misc]

        self._rebuild_buckets()
        active = {k: v for k, v in self._buckets.items() if v}
        if not active:
            idxs = [self.rng.randrange(self.size) for _ in range(batch_size)]
            return [self.buf[i] for i in idxs]  # type: ignore[misc]

        bucket_keys = sorted(active.keys())
        n_buckets = len(bucket_keys)
        per_bucket = batch_size // n_buckets
        remainder = batch_size % n_buckets

        idxs: List[int] = []
        for i, k in enumerate(bucket_keys):
            n = per_bucket + (1 if i < remainder else 0)
            if n <= 0:
                continue
            # With replacement so a small bucket can still meet its quota.
            idxs.extend(self.rng.choices(active[k], k=n))
        self.rng.shuffle(idxs)
        return [self.buf[i] for i in idxs[:batch_size]]  # type: ignore[misc]

    def __len__(self) -> int:
        return self.size
