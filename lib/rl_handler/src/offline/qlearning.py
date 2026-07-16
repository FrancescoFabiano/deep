"""Offline Double-DQN (+ CQL ablation) over the reservoir fringe MDP.

REPRESENTATION-AGNOSTIC BY CONSTRUCTION. `node_features` are opaque int64s. There
is no branch on HASHED / BITMASK / MAPPED anywhere in this file, and there must
never be: when the BITMASK path is enabled in `FringeEvalRL`, the only change is
which integers the C++ writes. The encoder already carries the per-dataset-type
branch; the training loop does not know or care which is active.

WHY DOUBLE-DQN AND NOT IQL
`|A(s)| <= F <= 64`, so `max_a Q` and `logsumexp_a Q` are trivial segment ops
(batching.py). IQL exists to avoid exactly those two operations when the action
space is combinatorial — which it was believed to be when the action was thought to
be `(v, D)`. The reservoir killed that premise. Worse, IQL is actively wrong here:
its purpose is to never evaluate out-of-distribution actions, but counterfactual
expansion supplies EVERY action in every visited state with the exact successor, so
there is no off-support action; and its advantage-weighted extraction can only
reweight actions the behaviour policy took, which excludes precisely the
counterfactual rows.

NO EXPLORATION. This is offline; behaviour randomness lives entirely in data
generation. There is no epsilon anywhere.

THE BOOTSTRAP IS ON `terminated`, NOT `done`
    y = r + (1 - terminated) * max_a' Q(s', a')
The expansion cap TRUNCATES: by the completeness proposition the planner would have
kept going and eventually succeeded, so a capped state is worth
-E[remaining expansions], not -1. Using `done` here leaks that -1 backwards and
makes the critic optimistic about deep searches (Pardo et al. 2018).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from .batching import (
    default_device,
    pack_batch,
    segment_argmax,
    segment_logsumexp,
    segment_max,
)
from .dataset import Transition
from .encoder import InstanceCache
from .env import DEFAULT_GAMMA, assert_gamma
from .metrics import is_argmin_action, r2
from .tree import INF_DELTA, TreeInstance

MODELS = ("dqn", "cql")


def default_reward_scale(instances: Sequence[TreeInstance]) -> float:
    """1 / median(delta_root over TRAIN instances).

    Pure rescaling — it cannot change the optimal policy — but at gamma=1 it keeps
    Q in a sane range for the value head. Measured median over the current pool is
    12, so returns land near [-1, -19]; CC's 34 is an outlier, not the median.

    NOT 1/expansion_cap: with cap=2000 the per-step signal would be 5e-4, below
    initialisation noise, and the critic would learn nothing for a long time.
    """
    ds = sorted(float(i.delta_root) for i in instances if i.solvable())
    if not ds:
        return 1.0
    m = ds[len(ds) // 2] if len(ds) % 2 else 0.5 * (ds[len(ds) // 2 - 1] + ds[len(ds) // 2])
    return 1.0 / max(1.0, m)


class DivergenceError(RuntimeError):
    """gamma=1 gives no contraction, so TD with function approximation can drift.
    We know Q* exactly, so drift is detectable rather than silent."""


@dataclass
class TrainConfig:
    fringe_size: int = 8
    gamma: float = DEFAULT_GAMMA
    lr: float = 1e-4
    batch_size: int = 64
    target_sync: int = 500
    max_grad_norm: float = 10.0
    model: str = "dqn"              # dqn | cql
    cql_alpha: float = 0.0          # 0 reduces cql to plain Double-DQN
    reward_scale: Optional[float] = None
    expansion_cap: int = 2000
    seed: int = 0
    device: Optional[str] = None
    # Divergence guard: |Q| beyond this multiple of the cap means drift, not learning.
    q_abort_multiple: float = 3.0


class QTrainer:
    """Double-DQN / CQL over pre-generated transitions.

    The transitions come from counterfactual expansion, so every state carries
    every action with its exact successor. Nothing here samples actions.
    """

    def __init__(
        self,
        model: nn.Module,
        target: nn.Module,
        instances: Sequence[TreeInstance],
        caches: Dict[str, InstanceCache],
        transitions: Sequence[Transition],
        cfg: TrainConfig,
    ):
        if cfg.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {cfg.model!r}")
        assert_gamma(cfg.gamma, cfg.expansion_cap)   # full dominance check: cap < horizon
        self.cfg = cfg
        self.device = cfg.device or default_device()
        self.model = model.to(self.device)
        self.target = target.to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.by_name = {i.name: i for i in instances}
        self.caches = caches
        self.data = list(transitions)
        if not self.data:
            raise ValueError("no transitions to train on")
        self.rng = random.Random(cfg.seed)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scale = (
            float(cfg.reward_scale) if cfg.reward_scale is not None
            else default_reward_scale(instances)
        )
        self.steps = 0
        # |Q| ceiling in SCALED units. The worst value a legitimate critic can
        # represent is the absorbing DOOM floor: at gamma<1 that is 1/(1-gamma)
        # (=10000 at 0.9999), NOT the cap -- a bootstrapped timeout is fixed-pointed
        # at doom (y=-1+gamma*(-1/(1-gamma)) = -1/(1-gamma)) and can go no lower. At
        # gamma=1 the SSP doom is -expansion_cap, so the cap IS the floor there.
        # Anything past q_abort_multiple * floor * scale is drift, not learning.
        worst_magnitude = (cfg.expansion_cap if cfg.gamma >= 1.0
                           else 1.0 / (1.0 - cfg.gamma))
        self.q_ceiling = cfg.q_abort_multiple * worst_magnitude * self.scale

    # ---- forward helpers -------------------------------------------------

    def _logits(self, net: nn.Module, picks) -> tuple[torch.Tensor, Dict]:
        p = pack_batch(self.caches, picks, self.device)
        out = net(
            node_features=p["node_features"], edge_index=p["edge_index"],
            edge_attr=p["edge_attr"], membership=p["membership"],
            candidate_batch=p["candidate_batch"], mask=None,
        )
        return out, p

    # ---- one gradient step ------------------------------------------------

    def step(self) -> Dict[str, float]:
        cfg = self.cfg
        batch = [self.data[self.rng.randrange(len(self.data))] for _ in range(cfg.batch_size)]

        picks = [(t.instance, t.obs) for t in batch]
        q_all, p = self._logits(self.model, picks)
        seg = p["candidate_batch"]
        n = len(batch)
        # Q(s, a_taken): slot_offset[i] + action[i]
        act_idx = p["slot_offset"] + torch.tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device)
        q_sa = q_all[act_idx]

        # --- targets ---
        # Only NON-terminated successors are bootstrapped from. Terminated rows
        # (SUCC / genuine DOOM) carry obs_next == [] and must not be packed.
        term = torch.tensor([t.terminated for t in batch], dtype=torch.bool, device=self.device)
        nxt_ids = [i for i, t in enumerate(batch) if not t.terminated and t.obs_next]
        y = torch.tensor([t.reward for t in batch], dtype=torch.float32,
                         device=self.device) * self.scale
        if nxt_ids:
            npicks = [(batch[i].instance, batch[i].obs_next) for i in nxt_ids]
            with torch.no_grad():
                q_next_online, pn = self._logits(self.model, npicks)
                q_next_target, _ = self._logits(self.target, npicks)
                nseg = pn["candidate_batch"]
                m = len(npicks)
                # DOUBLE-DQN: argmax from the ONLINE net, value from the TARGET net.
                a_star = segment_argmax(q_next_online, nseg, m)
                boot = q_next_target[a_star]
            y = y.clone()
            y[torch.tensor(nxt_ids, device=self.device)] += cfg.gamma * boot
        # y for terminated rows is just r (no bootstrap) -- (1 - terminated) applied
        # by construction above, NOT (1 - done): truncated rows DO bootstrap.

        td = nn.functional.smooth_l1_loss(q_sa, y)
        loss = td
        cql_term = torch.zeros((), device=self.device)
        if cfg.model == "cql" and cfg.cql_alpha > 0:
            # logsumexp_a Q(s,a) - Q(s,a_data): push down OOD actions. Trivial here
            # because |A(s)| <= F.
            lse = segment_logsumexp(q_all, seg, n)
            cql_term = (lse - q_sa).mean()
            loss = loss + cfg.cql_alpha * cql_term

        self.opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.opt.step()
        self.steps += 1
        if self.steps % cfg.target_sync == 0:
            self.target.load_state_dict(self.model.state_dict())

        q_max = float(q_all.detach().abs().max())
        if not math.isfinite(q_max) or q_max > self.q_ceiling:
            raise DivergenceError(
                f"|Q| = {q_max:.3f} (scaled) exceeds {self.q_ceiling:.3f} = "
                f"{cfg.q_abort_multiple} x expansion_cap x reward_scale at step "
                f"{self.steps}. gamma={cfg.gamma} gives no contraction, so this is "
                f"drift, not learning. Unscaled |Q| ~ {q_max / self.scale:.1f} "
                f"expansions against a cap of {cfg.expansion_cap}."
            )
        return {
            "step": self.steps,
            "td_loss": float(td.detach()),
            "cql": float(cql_term.detach()),
            "loss": float(loss.detach()),
            "q_mean": float(q_all.detach().mean()),
            "q_max": q_max,
            "q_max_unscaled": q_max / self.scale,
            "grad_norm": float(gn),
            "lr": self.opt.param_groups[0]["lr"],
        }

    # ---- critic calibration (F4) -----------------------------------------

    def q_vs_qstar(self, rows: Sequence[Transition]) -> Dict[str, Optional[float]]:
        """F4 panel A: Q(s, argmin) vs -delta(s), which is EXACT on the argmin slot.

        Split on ARGMIN, not on sterility: viable non-argmin slots evict too, so
        pooling them into a "clean" R^2 would punish a correct critic. Panel B (the
        residual below the naive bound for non-argmin actions) is computed by the
        telemetry, which owns the |R| pairing.
        """
        pred, truth = [], []
        for t in rows:
            inst = self.by_name.get(t.instance)
            if inst is None or not t.obs:
                continue
            if not is_argmin_action(inst, t.obs, [], t.action):
                continue
            d = inst.delta[t.obs[t.action]]
            if d == INF_DELTA:
                continue
            with torch.no_grad():
                q, p = self._logits(self.model, [(t.instance, t.obs)])
                pred.append(float(q[t.action]) / self.scale)
            truth.append(-float(d))
        if len(pred) < 2:
            return {"q_vs_qstar_r2": None, "q_vs_qstar_mae": None, "n": len(pred)}
        return {
            "q_vs_qstar_r2": r2(pred, truth),
            "q_vs_qstar_mae": sum(abs(a - b) for a, b in zip(pred, truth)) / len(pred),
            "n": len(pred),
        }

    # ---- the deployed policy ---------------------------------------------

    def greedy_policy(self, instance_name: str):
        """argmax logits over the active slots -- exactly what the planner pops."""
        cache = self.caches[instance_name]

        def _policy(beam: Sequence[int]) -> List[int]:
            from .batching import pack_single
            p = pack_single(cache, beam, len(beam), self.device)
            with torch.no_grad():
                s = self.model(
                    node_features=p["node_features"], edge_index=p["edge_index"],
                    edge_attr=p["edge_attr"], membership=p["membership"],
                    candidate_batch=None, mask=p["mask"],
                )
            sc = s.detach().cpu().tolist()
            return sorted(range(len(beam)), key=lambda k: -sc[k])

        return _policy
