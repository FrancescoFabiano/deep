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
    assert_goal_mode_consistent,
    default_device,
    pack_batch,
    pack_single,
    segment_argmax,
    segment_logsumexp,
    segment_max,
)
from .dataset import Transition
from .encoder import InstanceCache, StateGraph
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
    # Draw distribution over instances. "proportional" is the historical
    # behaviour (uniform over ROWS, so row share == gradient share); "capped"
    # draws an instance first, then a row uniformly within it. Neither filters
    # rows, drops instances, or touches `self.data` -- only the draw changes.
    sampler: str = "proportional"   # proportional | capped
    # The MAXIMUM multiple of its own natural share any instance may be lifted
    # to. Dimensionless, so it does not move with F or D. The cap is DERIVED
    # from it as the tightest value consistent with it, c*(k), so no (k, pool)
    # combination can be infeasible for k >= 1.
    sampler_k: float = 4.0
    reward_scale: Optional[float] = None
    expansion_cap: int = 2000
    seed: int = 0
    device: Optional[str] = None
    # Divergence guard: |Q| beyond this multiple of the cap means drift, not learning.
    q_abort_multiple: float = 3.0
    # How often the guard MATERIALISES |Q| (a GPU->CPU sync). 1 = every step (the
    # default; drift is caught immediately). Drift compounds over steps, so a larger
    # value (e.g. 100 in production) catches it a few steps later while removing the
    # last per-step sync -- it does not change the trained weights, only the abort
    # latency. The 5 logging fields never sync per step at all (returned as on-device
    # tensors, float()'d by the caller only at checkpoints).
    div_check_every: int = 1


SAMPLERS = ("proportional", "capped")


def _prop_then_clip(p: Dict[str, float], u: Dict[str, float]) -> Dict[str, float]:
    """Allocate 1.0 over instances: start proportional, clip anything above its
    ceiling, freeze it, renormalise the REST in proportion to p. Repeat.

    NOT water-filling: water-filling equalises everything below the cap and so
    returns the uniform allocation, which is a different intervention entirely
    (it lifts 12-row instances to 1/m). Here nothing is ever lifted -- an
    instance that cannot absorb mass simply stays small, and the tail keeps its
    relative weights. Terminates in at most len(p) rounds (one freeze each).
    """
    s = dict(p)
    frozen = {k for k, v in u.items() if v <= 0.0}
    for k in frozen:
        s[k] = 0.0
    for _ in range(len(p) + 1):
        free = [k for k in p if k not in frozen]
        if not free:
            break
        mass = 1.0 - sum(s[k] for k in frozen)
        denom = sum(p[k] for k in free)
        if denom <= 0.0:
            break
        for k in free:
            s[k] = mass * p[k] / denom
        over = [k for k in free if s[k] > u[k] + 1e-15]
        if not over:
            break
        for k in over:
            s[k] = u[k]
            frozen.add(k)
    return s


def _derive_cap(p: Dict[str, float], k: float) -> float:
    """c*(k) = min { c : sum_i min(c, k*p_i) >= 1 } -- the TIGHTEST cap that
    still admits an allocation in which nothing is lifted beyond k.

    The cap is derived, not declared: k (a repetition tolerance, dimensionless)
    is the knob, and c follows. A solution always exists for k >= 1, because at
    c = 1 the sum is sum_i k*p_i = k >= 1, so this construction cannot be
    infeasible -- there is no rejection path.

    sum_i min(c, k*p_i) is non-decreasing in c, so bisection is exact to
    tolerance. Returns the feasible (upper) side, so sum_i u_i >= 1 always.

    Guarded here as well as in the caller: below k=1 the bisection has no root
    and would silently return 1.0, which is a wrong answer rather than a
    refusal.
    """
    if k < 1.0:
        raise ValueError(
            f"k must be >= 1 for c*(k) to exist (at c=1 the sum is k), got {k:g}")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(min(mid, k * v) for v in p.values()) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


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
        goals: Optional[Dict[str, StateGraph]] = None,
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
        # S1/S2 boundary invariant, once at construction: a separated model MUST have a
        # goal map, a merged model MUST NOT. Online and target share `_logits`, so both
        # nets are covered by the same per-call assertion downstream.
        assert_goal_mode_consistent(self.model, goals is not None)
        self.goals = goals
        self.by_name = {i.name: i for i in instances}
        self.caches = caches
        self.data = list(transitions)
        if not self.data:
            raise ValueError("no transitions to train on")
        self.rng = random.Random(cfg.seed)
        self._build_allocation()
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
        self.div_check_every = max(1, int(cfg.div_check_every))

    # ---------- instance-stratified draw distribution ----------

    def _build_allocation(self) -> None:
        """One pass over self.data -> per-instance row indices, p, u, s.

        `Transition.instance` is a first-class field, so no re-scan and no
        schema change. self.data is READ, never mutated: every instance keeps
        every row, and |train_rows| is identical in both modes. The only thing
        that differs between proportional and capped is which row a draw lands
        on.
        """
        if self.cfg.sampler not in SAMPLERS:
            raise ValueError(
                f"sampler must be one of {SAMPLERS}, got {self.cfg.sampler!r}")
        k = float(self.cfg.sampler_k)
        if k < 1.0:
            raise ValueError(
                f"sampler_k must be >= 1 (an instance cannot be lifted to less "
                f"than its own share and still fill the simplex), got {k:g}")
        idx: Dict[str, List[int]] = {}
        for j, t in enumerate(self.data):
            idx.setdefault(t.instance, []).append(j)
        N = len(self.data)
        m = len(idx)
        if m == 0:
            raise ValueError("empty pool: no instances carry rows")
        p = {i: len(v) / N for i, v in idx.items()}
        c_star = _derive_cap(p, k)                    # derived from k, not from m
        u = {i: min(c_star, k * p[i]) for i in p}
        sum_u = sum(u.values())

        if self.cfg.sampler == "capped":
            s = _prop_then_clip(p, u)
        else:
            s = dict(p)                                # today's behaviour, exactly
        max_lift = max((s[i] / p[i]) for i in p if p[i] > 0)

        self._inst_names = sorted(idx)
        self._inst_rows = [idx[n] for n in self._inst_names]
        self._alloc_p = p
        self._alloc_s = s
        self._alloc_u = u
        self.allocation = {
            "mode": self.cfg.sampler, "m": m, "c_star": c_star, "k": k,
            "max_lift": max_lift, "sum_u": sum_u, "n_rows": N,
            "counts": {n: len(idx[n]) for n in idx},
            "p": p, "u": u, "s": s,
        }
        # cum_weights + random.choices is a C-level draw, so a capped step costs
        # the same as today's randrange in practice.
        cum, acc = [], 0.0
        for n in self._inst_names:
            acc += s[n]
            cum.append(acc)
        if cum:
            cum[-1] = 1.0
        self._inst_cum = cum

    def _draw_batch(self) -> List[Transition]:
        """batch_size rows. proportional: uniform over rows (unchanged).
        capped: instance ~ s, then a row uniformly within that instance."""
        n = self.cfg.batch_size
        if self.cfg.sampler == "proportional":
            return [self.data[self.rng.randrange(len(self.data))] for _ in range(n)]
        picks = self.rng.choices(range(len(self._inst_names)),
                                 cum_weights=self._inst_cum, k=n)
        out = []
        for i in picks:
            rows = self._inst_rows[i]
            out.append(self.data[rows[self.rng.randrange(len(rows))]])
        return out

    def allocation_report(self, total_draws: int) -> str:
        """The 2C block: what each instance will actually be shown."""
        a = self.allocation
        N, D = a["n_rows"], int(total_draws)
        E = D / N if N else 0.0
        head = (f"[sampler] mode={a['mode']}  k={a['k']:g}  "
                f"c*={a['c_star']:.4f}  m={a['m']}  "
                f"max_lift={a['max_lift']:.2f}  sum_u={a['sum_u']:.4f}")
        lines = [head,
                 f"[sampler] {'instance':24}{'n_i':>9}{'p_i':>8}{'s_i':>8}"
                 f"{'r_i':>10}{'r_i/E':>8}  binds   (E=D/N={E:,.1f})"]
        for name in sorted(a["s"], key=lambda x: -a["s"][x]):
            n_i, s_i, u_i, p_i = a["counts"][name], a["s"][name], a["u"][name], a["p"][name]
            r_i = s_i * D / n_i if n_i else 0.0
            if a["mode"] == "proportional" or s_i < u_i - 1e-12:
                binds = "-"
            elif u_i <= a["k"] * p_i - 1e-15:
                binds = "c*"
            else:
                binds = "k"
            lines.append(f"[sampler] {name:24}{n_i:>9,}{p_i:>8.4f}{s_i:>8.4f}"
                         f"{r_i:>10,.0f}{(r_i / E if E else 0):>8.2f}  {binds}")
        return "\n".join(lines)

    # ---- forward helpers -------------------------------------------------

    def _logits(self, net: nn.Module, picks) -> tuple[torch.Tensor, Dict]:
        # ONE goal decision, per instance, looked up by name (not stored per row).
        # Online and target both flow through here -> identical goal treatment, which
        # matters because double-DQN takes argmax(online) and value(target).
        goal_graphs = (
            [self.goals[name] for name, _ in picks] if self.goals is not None else None
        )
        assert_goal_mode_consistent(net, goal_graphs is not None)
        p = pack_batch(self.caches, picks, self.device, goal_graphs=goal_graphs)
        out = net(
            node_features=p["node_features"], edge_index=p["edge_index"],
            edge_attr=p["edge_attr"], membership=p["membership"],
            candidate_batch=p["candidate_batch"], mask=None,
            goal_node_features=p.get("goal_node_features"),
            goal_edge_index=p.get("goal_edge_index"),
            goal_edge_attr=p.get("goal_edge_attr"),
            goal_batch=p.get("goal_batch"),
        )
        return out, p

    # ---- one gradient step ------------------------------------------------

    def step(self) -> Dict[str, float]:
        cfg = self.cfg
        batch = self._draw_batch()

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

        # DIVERGENCE GUARD -- the only per-step sync, and only every div_check_every
        # steps. `q_max_t` stays on device; we materialise it (a GPU->CPU sync) solely
        # to compare against the ceiling. Drift compounds, so checking every N steps
        # catches it a few steps later at most and does NOT change the trained weights.
        q_max_t = q_all.detach().abs().max()
        if self.steps % self.div_check_every == 0:
            q_max = float(q_max_t)
            if not math.isfinite(q_max) or q_max > self.q_ceiling:
                raise DivergenceError(
                    f"|Q| = {q_max:.3f} (scaled) exceeds {self.q_ceiling:.3f} = "
                    f"{cfg.q_abort_multiple} x expansion_cap x reward_scale at step "
                    f"{self.steps}. gamma={cfg.gamma} gives no contraction, so this is "
                    f"drift, not learning. Unscaled |Q| ~ {q_max / self.scale:.1f} "
                    f"expansions against a cap of {cfg.expansion_cap}."
                )
        # The logging fields are returned as detached ON-DEVICE TENSORS -- NO per-step
        # sync. The caller float()s them ONLY at checkpoints (run.py), where the read
        # actually happens (~20x total, not once per step). Read-only observations:
        # the training math and its order are untouched, so removing these syncs is
        # bit-identical (test_step_is_bit_identical_with_and_without_logging_syncs).
        return {
            "step": self.steps,
            "td_loss": td.detach(),
            "cql": cql_term.detach(),
            "loss": loss.detach(),
            "q_mean": q_all.detach().mean(),
            "q_max": q_max_t,
            "q_max_unscaled": q_max_t / self.scale,
            "grad_norm": gn.detach() if torch.is_tensor(gn) else torch.as_tensor(gn),
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
        goal_graph = self.goals[instance_name] if self.goals is not None else None

        def _policy(beam: Sequence[int]) -> List[int]:
            assert_goal_mode_consistent(self.model, goal_graph is not None)
            p = pack_single(cache, beam, len(beam), self.device, goal_graph=goal_graph)
            with torch.no_grad():
                s = self.model(
                    node_features=p["node_features"], edge_index=p["edge_index"],
                    edge_attr=p["edge_attr"], membership=p["membership"],
                    candidate_batch=None, mask=p["mask"],
                    goal_node_features=p.get("goal_node_features"),
                    goal_edge_index=p.get("goal_edge_index"),
                    goal_edge_attr=p.get("goal_edge_attr"),
                    goal_batch=p.get("goal_batch"),
                )
            sc = s.detach().cpu().tolist()
            return sorted(range(len(beam)), key=lambda k: -sc[k])

        return _policy
