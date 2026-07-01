"""Double-DQN trainer over the fringe MDP. See DESIGN.md §3-§5.

Q(fringe, k) = FrontierPolicyNetwork logits[k]; one GNN forward scores the
whole fringe (shared weights + frontier context), exactly the deployed
parameterization.  Targets: y = r + gamma*(1-done)*Q_target(s', argmax_online).
No large-magnitude constants anywhere: dead ends terminate with y = r = -1.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from scipy.stats import spearmanr
from torch import nn
from tqdm import tqdm

from src.models.frontier_policy import FrontierPolicyNetwork
from src.offline.encoder import (
    GlobalFlatCache,
    InstanceCache,
    StateGraph,
    segment_argmax,
)
from src.offline.replay import ReplayBuffer, Transition
from src.offline.tree_env import (
    UNREACHABLE_DISTANCE,
    FringeEnv,
    OccupancyCounter,
    TreeInstance,
    bfs_expansions,
)


class EpsilonSchedule:
    """Linear start->end over `frac` of total frames, then flat."""

    def __init__(self, start: float, end: float, frac: float, total_frames: int):
        self.start = float(start)
        self.end = float(end)
        self.decay_frames = max(1, int(float(frac) * int(total_frames)))

    def __call__(self, frame: int) -> float:
        t = min(1.0, frame / self.decay_frames)
        return self.start + t * (self.end - self.start)

    @classmethod
    def parse(cls, spec: str, total_frames: int) -> "EpsilonSchedule":
        start, end, frac = (float(x) for x in spec.split(","))
        return cls(start, end, frac, total_frames)


class OfflineDQNTrainer:
    def __init__(
        self,
        model: FrontierPolicyNetwork,
        instances: Sequence[TreeInstance],
        caches: Sequence[InstanceCache],
        train_ids: Sequence[int],
        val_ids: Sequence[int],
        fringe_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        warmup: int = 1_000,
        target_sync: int = 1_000,
        update_every: int = 1,
        eval_expansion_cap: int = 2_000,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        device: Optional[str] = None,
        eval_refill_seeds: int = 1,
        signal_mode: str = "basic",
        aux_lambda: float = 1.0,
        rank_variant: Optional[str] = None,
        lambda_ord: float = 0.0,
        goal_graphs: Optional[Sequence[Optional["StateGraph"]]] = None,
        select_on_train: bool = False,
        cql_alpha: float = 0.0,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.target = FrontierPolicyNetwork(
            node_input_dim=1,
            hidden_dim=model.encoder.input_proj.out_features,
            gnn_layers=len(model.encoder.layers),
            conv_type=model.encoder.conv_type,
            pooling_type=model.pooling_type,
            dataset_type=model.dataset_type,
            edge_emb_dim=model.edge_emb_dim,
            num_edge_labels=model.num_edge_labels,
            num_node_labels=model.num_node_labels,
            use_global_context=model.use_global_context,
            mlp_depth=2,
            use_goal_separate_input=model.use_goal_separate_input,
        ).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.instances = list(instances)
        self.caches = list(caches)
        # Separated mode: one parsed goal graph per instance (aligned with
        # caches).  use_goal mirrors the model so the forward passes (online +
        # target) feed goal_* iff the architecture has the separate goal input.
        self.use_goal = bool(model.use_goal_separate_input)
        self.goal_graphs = list(goal_graphs) if goal_graphs is not None else None
        if self.use_goal and self.goal_graphs is None:
            raise ValueError(
                "use_goal_separate_input is on but no goal_graphs were passed; "
                "separated runs must supply one goal graph per instance."
            )
        self.train_ids = list(train_ids)
        self.val_ids = list(val_ids)
        # Selection set: when select_on_train is set (the regime/study default),
        # checkpoints are picked on the TRAIN deploy-faithful metric regardless of
        # whether a val set was supplied — the held-out instances are diagnostic
        # only. Otherwise selection uses val if present, else falls back to train.
        self.select_on_train = bool(select_on_train)
        self.fringe_size = int(fringe_size)
        self.gamma = float(gamma)
        # Maximally-bad finite return floor = -1/(1-gamma) (all -1 forever): the
        # value an unreachable (d*=inf) node deserves. Used to INCLUDE unreachables
        # in the d*-aware objectives (exact-return target / aux regression / PBRS
        # potential) instead of masking them out, and as the 'worst' anchor the
        # order objectives sort everything above. gamma<1 by construction (DESIGN
        # §3); guard the degenerate gamma>=1 so floor stays finite-negative.
        self.floor_v = (-1.0 / (1.0 - self.gamma)) if self.gamma < 1.0 else -1e9
        # P2 order-auxiliary weight: L = L_val + lambda_ord * L_ord (pairwise over
        # the order-eligible slots) on the SAME logits. 0 => the order term is
        # never even computed (byte-identical to the value-only path).
        self.lambda_ord = float(lambda_ord)
        # CQL conservative-Q weight: L = L_dqn + cql_alpha * (logsumexp_elig(Q) -
        # Q_taken). 0 => the term is never computed (byte-identical to DQN), same
        # guard pattern as lambda_ord.
        self.cql_alpha = float(cql_alpha)
        # Windowed order-aux instrumentation (reset by reset_order_stats): pairs
        # actually formed vs candidate pairs, and fringes skipped by the
        # post-mask <2-distinct guard, all over order-eligible slots only.
        self._order_usable_pairs = 0
        self._order_candidate_pairs = 0
        self._order_skipped_fringes = 0
        self._order_seen_fringes = 0
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.target_sync = int(target_sync)
        self.update_every = max(1, int(update_every))
        self.eval_expansion_cap = int(eval_expansion_cap)
        self.max_grad_norm = float(max_grad_norm)
        self.seed = int(seed)
        self.eval_refill_seeds = max(1, int(eval_refill_seeds))
        # Opt-in stability instrumentation (RL_LOG_STABILITY=1), default OFF so
        # normal runs are byte-identical. Pure read-only probes — no effect on
        # the model, contract, or export. See _stability_probe / train().
        self.log_stability = os.environ.get("RL_LOG_STABILITY") == "1"
        self._probe_batch: Optional[List[Transition]] = None
        self._grad_norm_ema: Optional[float] = None

        # d*-aware training signal selection (see _update). 'basic' is the
        # current Double-DQN path (default; byte-identical to before). The other
        # modes use the offline d* oracle (inst.distance) to inject within-fringe
        # ranking signal that the -1/step reward lacks (ledger S).
        # 'rank-sup'/'rank-rl' are the Horn-B SCALE-INVARIANT / ORDER objectives
        # (ledger G): deployment uses only the within-fringe ARGMAX, which is
        # range-free, so an order-based loss can extrapolate depth where the
        # absolute-value learners (exact-return / TD) overfit the train d* range.
        # rank-sup = supervised, no bootstrap (beside exact-return); rank-rl =
        # keeps the Double-DQN bootstrap + export contract (beside basic). Both
        # change ONLY the loss/target — the model + ONNX contract are untouched.
        valid = {"basic", "pbrs", "exact-return", "aux", "rank-sup", "rank-rl"}
        if signal_mode not in valid:
            raise ValueError(f"signal_mode must be one of {sorted(valid)}, got {signal_mode!r}")
        self.signal_mode = signal_mode
        self.aux_lambda = float(aux_lambda)
        # Per-mode variant. None -> the mode's default (pairwise / reward).
        _RANK_VARIANTS = {
            "rank-sup": {"pairwise", "listwise"},
            "rank-rl": {"reward", "advantage"},
        }
        _RANK_DEFAULT = {"rank-sup": "pairwise", "rank-rl": "reward"}
        if signal_mode in _RANK_VARIANTS:
            rv = rank_variant or _RANK_DEFAULT[signal_mode]
            if rv not in _RANK_VARIANTS[signal_mode]:
                raise ValueError(
                    f"rank_variant for {signal_mode!r} must be one of "
                    f"{sorted(_RANK_VARIANTS[signal_mode])}, got {rv!r}"
                )
            self.rank_variant = rv
        else:
            if rank_variant is not None:
                raise ValueError(
                    f"rank_variant is only valid for rank-sup/rank-rl, "
                    f"not signal_mode={signal_mode!r}"
                )
            self.rank_variant = None
        # Auxiliary -d* regression head on the shared GINE trunk, TRAINING-ONLY:
        # it lives on the trainer, never on self.model, so it is never exported
        # to ONNX (contract unchanged). Only built for 'aux' mode.
        self.aux_head: Optional[nn.Module] = None
        if self.signal_mode == "aux":
            h = self.model.encoder.input_proj.out_features
            self.aux_head = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)
            ).to(self.device)

        torch.manual_seed(seed)
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        params = list(self.model.parameters())
        if self.aux_head is not None:
            params += list(self.aux_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.flat = GlobalFlatCache(
            self.caches,
            device=self.device,
            goal_graphs=self.goal_graphs if self.use_goal else None,
        )
        self.envs = {
            i: FringeEnv(self.instances[i], fringe_size=fringe_size, seed=seed + i)
            for i in set(self.train_ids) | set(self.val_ids)
        }

    # ---------- forward helpers ----------

    def _pack(
        self, fringes: Sequence[Tuple[int, Sequence[int]]]
    ) -> Dict[str, torch.Tensor]:
        gids = torch.tensor(
            [
                self.flat.gid(i, s)
                for i, states in fringes
                for s in states
            ],
            dtype=torch.long,
        )
        lens = torch.tensor([len(s) for _, s in fringes], dtype=torch.long)
        # Separated mode: instance id per fringe drives per-fringe goal packing
        # (goal_batch=b), so goal_emb[candidate_batch] aligns one goal per
        # candidate. Single fringe -> goal_batch all-zeros (inference parity).
        fringe_inst = (
            torch.tensor([i for i, _ in fringes], dtype=torch.long)
            if self.use_goal
            else None
        )
        return self.flat.pack(gids, lens, fringe_inst=fringe_inst)

    def _forward(
        self,
        net: nn.Module,
        fringes: Sequence[Tuple[int, Sequence[int]]],
        packed: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score fringes [(inst_idx, state_ids), ...] -> (flat logits, ptr)."""
        if packed is None:
            packed = self._pack(fringes)
        goal_kwargs = {}
        if self.use_goal and "goal_node_features" in packed:
            goal_kwargs = {
                "goal_node_features": packed["goal_node_features"],
                "goal_edge_index": packed["goal_edge_index"],
                "goal_edge_attr": packed["goal_edge_attr"],
                "goal_batch": packed["goal_batch"],
            }
        logits = net(
            node_features=packed["node_features"],
            edge_index=packed["edge_index"],
            edge_attr=packed["edge_attr"],
            membership=packed["membership"],
            candidate_batch=packed["candidate_batch"],
            **goal_kwargs,
        )
        return logits, packed["fringe_ptr"]

    @torch.no_grad()
    def greedy_action(self, inst_idx: int, fringe: Sequence[int]) -> int:
        self.model.eval()
        logits, _ = self._forward(self.model, [(inst_idx, fringe)])
        return int(torch.argmax(logits).item())

    # ---------- training ----------

    def _update(self) -> Dict[str, float]:
        batch = self.replay.sample(self.batch_size)

        self.model.train()
        cur_packed = self._pack([(t.inst, t.fringe) for t in batch])
        cur_logits, cur_ptr = self._forward(self.model, [], packed=cur_packed)
        q_sa = cur_logits[
            cur_ptr[:-1]
            + torch.tensor([t.action for t in batch], device=self.device)
        ]

        aux_loss = 0.0
        order_loss = 0.0
        cql_loss = 0.0
        targets = None  # set by the value-based branches; None for rank-sup
        if self.signal_mode == "rank-sup":
            # SUPERVISED, NO BOOTSTRAP (beside exact-return). The loss is purely
            # ORDER-based over the full fringe's slot logits, so it carries no d*
            # magnitude -> range-free by construction (ledger G / Horn B).
            loss = self._rank_sup_loss(batch, cur_logits, cur_ptr)
        elif self.signal_mode == "exact-return":
            # Supervise the chosen slot on the exact (undiscounted) return
            # -d*(chosen node); no bootstrap. Unreachable nodes (d*=inf) are KEPT,
            # not masked: their target is floor_v = -1/(1-gamma) (worst possible
            # return), so the model learns they are maximally bad rather than
            # ignoring them. Order is all deployment uses, so y makes
            # argmax(Q) -> argmin(d*) = the oracle pick, unreachables last.
            y = []
            for t in batch:
                d = self.instances[t.inst].distance[t.fringe[t.action]]
                y.append(-d if d < UNREACHABLE_DISTANCE else self.floor_v)
            targets = torch.tensor(y, dtype=torch.float32, device=self.device)
            loss = nn.functional.smooth_l1_loss(q_sa, targets)
        else:
            # Bootstrap-based: basic / pbrs / aux / rank-rl. Reward r (or a mode-
            # specific reward) plus gamma * maxQ(s') via Double-DQN.
            if self.signal_mode == "pbrs":
                rew = []
                for t in batch:
                    phi_s = self._phi(t.inst, t.fringe)
                    phi_sp = 0.0 if t.done else self._phi(t.inst, t.next_fringe)
                    rew.append(t.reward + self.gamma * phi_sp - phi_s)
            elif self.signal_mode == "rank-rl" and self.rank_variant == "reward":
                # RANK-FRACTION REWARD in [0,1] (1 = chose the min-d* slot). Range-
                # free by construction: the per-step signal is a within-fringe
                # ORDER statistic, never an absolute d*. Keeps gamma / bootstrap.
                rew = [self._rank_fraction_reward(t) for t in batch]
            else:
                rew = [t.reward for t in batch]
            targets = torch.tensor(rew, dtype=torch.float32, device=self.device)
            live = [i for i, t in enumerate(batch) if not t.done]
            if live:
                next_packed = self._pack(
                    [(batch[i].inst, batch[i].next_fringe) for i in live]
                )
                with torch.no_grad():
                    self.model.eval()
                    on_logits, on_ptr = self._forward(self.model, [], packed=next_packed)
                    tg_logits, _ = self._forward(self.target, [], packed=next_packed)
                    self.model.train()
                    a_star = segment_argmax(on_logits, on_ptr)  # global positions
                    q_next = tg_logits[a_star]
                idx = torch.tensor(live, device=self.device)
                targets[idx] = targets[idx] + self.gamma * q_next
            if self.signal_mode == "rank-rl" and self.rank_variant == "advantage":
                # CENTRE each transition's target by its current fringe's mean
                # (target-net) logit -> a per-fringe baseline. Removes the absolute
                # d* LEVEL while preserving within-fringe order. PARTIAL scale-fix:
                # centring removes the level but NOT the scale (the spread of
                # targets still grows with d* gaps), unlike the rank-fraction
                # reward which is fully range-free. Loss-side only (no dueling /
                # architecture change) -> ONNX contract untouched.
                targets = targets - self._fringe_baseline(cur_packed, cur_ptr)
            loss = nn.functional.smooth_l1_loss(q_sa, targets)
            if self.signal_mode == "aux":
                aux_loss = self._aux_dstar_loss(batch, cur_packed)
                loss = loss + self.aux_lambda * aux_loss
            # P2 combined objective: add the pairwise ORDER auxiliary on the same
            # logits, weighted by lambda_ord. Computed (and back-propagated) ONLY
            # when lambda_ord > 0, so lambda_ord == 0 is byte-identical to the
            # value-only path. Padded slots are excluded inside _order_aux_loss.
            if self.lambda_ord > 0.0:
                order_loss = self._order_aux_loss(batch, cur_logits, cur_ptr)
                loss = loss + self.lambda_ord * order_loss

        # CQL conservative-Q penalty on the SAME logits, added to whatever loss
        # the mode produced. Computed (and back-propagated) ONLY when cql_alpha>0,
        # so cql_alpha==0 is byte-identical to the DQN path (same guard as the
        # order aux). Never touches the target / Double-DQN action selection.
        if self.cql_alpha > 0.0:
            cql_loss = self._cql_conservative_loss(batch, cur_logits, cur_ptr, q_sa)
            loss = loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            clip_params = list(self.model.parameters())
            if self.aux_head is not None:
                clip_params += list(self.aux_head.parameters())
            gn = nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
            # EMA of the PRE-clip global grad norm (clip_grad_norm_ returns it),
            # so the stability log can show whether grads constantly saturate the
            # clip. Free to capture; logged only when RL_LOG_STABILITY=1.
            g = float(gn)
            self._grad_norm_ema = g if self._grad_norm_ema is None else (
                0.98 * self._grad_norm_ema + 0.02 * g
            )
        self.optimizer.step()
        return {
            "td_loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "target_mean": (
                float("nan") if targets is None else float(targets.mean().item())
            ),
            "aux_loss": float(aux_loss) if self.signal_mode == "aux" else 0.0,
            "order_loss": (
                float(order_loss.detach()) if torch.is_tensor(order_loss) else 0.0
            ),
            "cql_loss": (
                float(cql_loss.detach()) if torch.is_tensor(cql_loss) else 0.0
            ),
        }

    # ---------- rank objectives (Horn B: scale-invariant / order) ----------

    def _order_eligible_slots(self, t):
        """The ORDER-ELIGIBLE slots of a transition's fringe: every slot (beams
        are live-pool only now — there is no padding). Returns (positions, dvals):
        positions index into the fringe's logit segment; dvals are the d* values
        with unreachable mapped to a sentinel strictly above every finite d*
        (= worst), per Task 1a. The <2-distinct-eligible-d* skip still applies
        downstream (a short beam with no d* spread carries no order signal)."""
        inst = self.instances[t.inst]
        raw = [inst.distance[s] for s in t.fringe]
        positions = list(range(len(raw)))
        finite = [d for d in raw if d < UNREACHABLE_DISTANCE]
        sentinel = (max(finite) + 1.0) if finite else 1.0
        dvals = [d if d < UNREACHABLE_DISTANCE else sentinel for d in raw]
        return positions, dvals

    def _pairwise_term(self, sl: torch.Tensor, dl: torch.Tensor):
        """Per-fringe pairwise logistic order loss over slots `sl` with d* `dl`:
        softplus(-(logit_i - logit_j)) for every (i,j) with d_i < d_j (strict; ties
        impose no constraint). Returns (term_or_None, n_pairs_formed)."""
        better = dl[:, None] < dl[None, :]                # i strictly better than j
        diff_s = sl[:, None] - sl[None, :]                # logit_i - logit_j (want > 0)
        sel = diff_s[better]
        npairs = int(better.sum().item())
        if sel.numel() == 0:
            return None, 0
        return nn.functional.softplus(-sel).mean(), npairs

    def _rank_sup_loss(
        self, batch, cur_logits: torch.Tensor, cur_ptr: torch.Tensor
    ) -> torch.Tensor:
        """Supervised ORDER loss over each fringe's ORDER-ELIGIBLE slot logits,
        sorted by d*. No bootstrap, no d* magnitude -> range-free. Padded slots are
        EXCLUDED (see _order_eligible_slots); the <2-distinct skip is evaluated on
        the eligible set only. Unreachables included via the worst-sorting
        sentinel. pairwise = logistic over d_i<d_j pairs; listwise = rank-normalised
        soft target."""
        terms: List[torch.Tensor] = []
        for b, t in enumerate(batch):
            lo, hi = int(cur_ptr[b].item()), int(cur_ptr[b + 1].item())
            seg = cur_logits[lo:hi]
            positions, dvals = self._order_eligible_slots(t)
            if len(set(dvals)) < 2:
                continue  # <2 distinct ELIGIBLE values -> no ordering signal
            idx = torch.tensor(positions, dtype=torch.long, device=self.device)
            sl = seg[idx]
            dl = torch.tensor(dvals, dtype=torch.float32, device=self.device)
            if self.rank_variant == "pairwise":
                term, _ = self._pairwise_term(sl, dl)
                if term is not None:
                    terms.append(term)
            else:  # listwise: rank-normalised soft target (range-free, tie-aware)
                n = sl.numel()
                cl = (dl[None, :] < dl[:, None]).sum(1).float()   # # strictly smaller
                ce = (dl[None, :] == dl[:, None]).sum(1).float()  # # equal (incl self)
                ranks = cl + (ce - 1.0) / 2.0
                w = (n - ranks)                                   # best->n, worst->1, >0
                target = w / w.sum()                              # ordinal only -> range-free
                logp = nn.functional.log_softmax(sl, dim=0)
                terms.append(-(target * logp).sum())
        if not terms:
            return cur_logits.sum() * 0.0  # keep grad path, contribute nothing
        return torch.stack(terms).mean()

    def _order_aux_loss(
        self, batch, cur_logits: torch.Tensor, cur_ptr: torch.Tensor
    ) -> torch.Tensor:
        """P2 order auxiliary: the PAIRWISE term of _rank_sup_loss over the
        ORDER-ELIGIBLE slots, on the SAME logits as the value loss. Beams are
        live-pool only (no padding); the <2-distinct skip runs on the eligible
        set. Accumulates windowed instrumentation (usable vs candidate pairs,
        skipped fringes) so tie-dominated batches are visible."""
        terms: List[torch.Tensor] = []
        for b, t in enumerate(batch):
            lo, hi = int(cur_ptr[b].item()), int(cur_ptr[b + 1].item())
            seg = cur_logits[lo:hi]
            positions, dvals = self._order_eligible_slots(t)
            self._order_seen_fringes += 1
            m = len(positions)
            self._order_candidate_pairs += m * (m - 1) // 2
            if len(set(dvals)) < 2:
                self._order_skipped_fringes += 1
                continue
            idx = torch.tensor(positions, dtype=torch.long, device=self.device)
            sl = seg[idx]
            dl = torch.tensor(dvals, dtype=torch.float32, device=self.device)
            term, npairs = self._pairwise_term(sl, dl)
            self._order_usable_pairs += npairs
            if term is not None:
                terms.append(term)
        if not terms:
            return cur_logits.sum() * 0.0
        return torch.stack(terms).mean()

    def _cql_conservative_loss(
        self, batch, cur_logits: torch.Tensor, cur_ptr: torch.Tensor,
        q_sa: torch.Tensor,
    ) -> torch.Tensor:
        """CQL conservative-Q penalty on the SAME logits as the value loss:
        mean over the batch of  logsumexp_{eligible slots}(Q_slot) - Q(taken slot).
        Pushes Q DOWN on the fringe's non-dataset slots (the logsumexp) and UP on
        the taken slot (subtracting q_sa), so the model stays conservative about
        actions the data never took. Eligibility = the SAME order-eligible slots
        the order loss / top-1 regret use (_order_eligible_slots), so the three
        never disagree on which slots count. torch.logsumexp for stability.
        Skips a transition with no eligible slots or whose taken slot is not
        eligible (mirrors the order-loss skip)."""
        terms: List[torch.Tensor] = []
        for b, t in enumerate(batch):
            lo, hi = int(cur_ptr[b].item()), int(cur_ptr[b + 1].item())
            seg = cur_logits[lo:hi]
            positions, _ = self._order_eligible_slots(t)
            if len(positions) < 1 or t.action not in positions:
                continue
            idx = torch.tensor(positions, dtype=torch.long, device=self.device)
            lse = torch.logsumexp(seg[idx], dim=0)     # soft-max over eligible Q
            terms.append(lse - q_sa[b])                # - Q(dataset action)
        if not terms:
            return cur_logits.sum() * 0.0
        return torch.stack(terms).mean()

    def reset_order_stats(self) -> None:
        self._order_usable_pairs = 0
        self._order_candidate_pairs = 0
        self._order_skipped_fringes = 0
        self._order_seen_fringes = 0

    def order_stats(self) -> Dict[str, object]:
        cp = max(1, self._order_candidate_pairs)
        sf = max(1, self._order_seen_fringes)
        return {
            "order_seen_fringes": self._order_seen_fringes,
            "order_usable_pairs": self._order_usable_pairs,
            "order_candidate_pairs": self._order_candidate_pairs,
            "order_usable_pair_frac": round(self._order_usable_pairs / cp, 4),
            "order_skipped_fringes": self._order_skipped_fringes,
            "order_skip_frac": round(self._order_skipped_fringes / sf, 4),
        }

    def _rank_fraction_reward(self, t) -> float:
        """Chosen slot's within-fringe d* rank fraction in [0,1]: 1 = picked the
        (a) min-d* slot, 0 = picked the max. Fraction of ALL slots strictly worse
        than the chosen one (unreachables INCLUDED as the worst class, counted in
        the denominator). Range-free per-step reward for rank-rl."""
        inst = self.instances[t.inst]
        ds = [inst.distance[s] for s in t.fringe]
        d_ch = ds[t.action]
        n = len(ds)
        if n <= 1:
            return 1.0  # the only viable pick is the best by construction
        if d_ch >= UNREACHABLE_DISTANCE:
            return 0.0  # chose an unreachable (worst) slot -> nothing strictly worse
        # finite chosen: strictly worse = larger-finite-d* slots + ALL unreachables
        worse = sum(
            1 for d in ds
            if d >= UNREACHABLE_DISTANCE or d > d_ch
        )
        return worse / (n - 1)

    @torch.no_grad()
    def _fringe_baseline(
        self, cur_packed: Dict[str, torch.Tensor], cur_ptr: torch.Tensor
    ) -> torch.Tensor:
        """Per-transition baseline = mean TARGET-net logit over the current
        fringe's slots (detached). Used by rank-rl 'advantage' to centre the
        target within the fringe. Returns a [batch] tensor aligned to cur_ptr."""
        self.target.eval()
        tg_logits, _ = self._forward(self.target, [], packed=cur_packed)
        n_seg = int(cur_ptr.numel() - 1)
        seg = torch.repeat_interleave(
            torch.arange(n_seg, device=self.device),
            (cur_ptr[1:] - cur_ptr[:-1]).to(self.device),
        )
        sums = torch.zeros(n_seg, device=self.device).scatter_add(0, seg, tg_logits)
        counts = (cur_ptr[1:] - cur_ptr[:-1]).to(self.device).float()
        return sums / counts

    def _phi(self, inst_idx: int, fringe: Sequence[int]) -> float:
        """PBRS state potential phi(fringe) = -min finite d* over its nodes
        (the best node's value). An all-unreachable fringe has no finite d*, so
        its potential is the worst-value floor floor_v (consistent with treating
        unreachables as maximally bad), not 0. State-only, as PBRS requires."""
        finite = [
            self.instances[inst_idx].distance[s]
            for s in fringe
            if self.instances[inst_idx].distance[s] < UNREACHABLE_DISTANCE
        ]
        return -min(finite) if finite else self.floor_v

    def _aux_dstar_loss(self, batch, cur_packed) -> torch.Tensor:
        """MSE(h_aux(z_candidate), -d*(candidate)) over the current fringes'
        candidates. z is the per-candidate pooled trunk embedding; gradients
        flow into the shared GINE trunk. h_aux is training-only (never exported).
        Candidate order matches pack: for t in batch, for s in t.fringe."""
        node_emb = self.model.encoder(
            cur_packed["node_features"], cur_packed["edge_index"], cur_packed["edge_attr"]
        )
        z = self.model._pool_nodes(
            node_emb, cur_packed["membership"],
            expected_size=int(cur_packed["candidate_batch"].numel()),
        )
        pred = self.aux_head(z).squeeze(-1)
        # Unreachable candidates are KEPT and regressed onto floor_v (worst value),
        # not masked out — the trunk learns an unreachable node is maximally bad.
        y = []
        for t in batch:
            inst = self.instances[t.inst]
            for s in t.fringe:
                d = inst.distance[s]
                y.append(-d if d < UNREACHABLE_DISTANCE else self.floor_v)
        tgt = torch.tensor(y, dtype=torch.float32, device=self.device)
        return nn.functional.mse_loss(pred, tgt)

    @torch.no_grad()
    def _stability_probe(self, frame: int) -> Dict[str, object]:
        """Read-only stability scalars on a FIXED probe batch (set once after
        warmup, held constant for the run so checkpoints are comparable).

        Logs the Q-magnitude / TD / target-divergence signature used to name the
        instability mode (Q-explosion vs oscillation vs saturation). Pure probe:
        no grads stepped, no model/contract/export touched.
        """
        b = self._probe_batch
        self.model.eval()
        cur_packed = self._pack([(t.inst, t.fringe) for t in b])
        cur_logits, cur_ptr = self._forward(self.model, [], packed=cur_packed)
        q_sa = cur_logits[
            cur_ptr[:-1] + torch.tensor([t.action for t in b], device=self.device)
        ]
        # net-divergence: RMS gap between online and target Q over the fixed
        # probe candidates (jumps at hard target-sync events if that's the mode).
        tg_cur, _ = self._forward(self.target, [], packed=cur_packed)
        target_online_l2 = float((cur_logits - tg_cur).pow(2).mean().sqrt().item())
        # Double-DQN TD on the same fixed probe (comparable across checkpoints).
        td = float("nan")
        live = [i for i, t in enumerate(b) if not t.done]
        if live:
            nxt = self._pack([(b[i].inst, b[i].next_fringe) for i in live])
            on_logits, on_ptr = self._forward(self.model, [], packed=nxt)
            tg_logits, _ = self._forward(self.target, [], packed=nxt)
            a_star = segment_argmax(on_logits, on_ptr)
            q_next = tg_logits[a_star]
            targets = torch.tensor(
                [t.reward for t in b], dtype=torch.float32, device=self.device
            )
            idx = torch.tensor(live, device=self.device)
            targets[idx] = targets[idx] + self.gamma * q_next
            td = float(nn.functional.smooth_l1_loss(q_sa, targets).item())
        return {
            "frame": int(frame),
            "q_mean": float(q_sa.mean().item()),
            "q_max": float(q_sa.abs().max().item()),
            "td_loss": td,
            "grad_norm": self._grad_norm_ema,
            "target_online_l2": target_online_l2,
        }

    def train(
        self,
        frames: int,
        n_checkpoints: int,
        epsilon: EpsilonSchedule,
        out_dir: Path,
    ) -> Dict[str, object]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_frames = sorted(
            {
                int(math.ceil((k + 1) * frames / n_checkpoints))
                for k in range(n_checkpoints)
            }
        )

        history: Dict[str, list] = {
            "frame": [], "td_loss": [], "q_mean": [], "target_mean": [],
            "aux_loss": [], "order_loss": [], "cql_loss": [], "epsilon": [],
            "episode_return": [],
            "episode_frame": [], "episode_expansions": [], "episode_goal": [],
            "episode_inst": [],
        }
        checkpoints: List[Dict[str, object]] = []
        stability: List[Dict[str, object]] = []
        best_val_exp = float("inf")
        best_spearman = -float("inf")

        env_order = list(self.train_ids)
        env_cursor = 0
        inst_idx = env_order[env_cursor]
        env = self.envs[inst_idx]
        res = env.reset(seed=self.seed)
        ep_return, ep_rng = 0.0, torch.Generator().manual_seed(self.seed + 1)
        # Live fringe occupancy over training episodes — cheap histogram, O(1)
        # per frame; surfaced per checkpoint so "does the beam bind past F?" is
        # readable for the train distribution too, not just the eval rollouts.
        train_occ = OccupancyCounter(self.fringe_size)

        loss_acc: List[Dict[str, float]] = []
        t0 = time.time()
        frame = 0
        last_td = last_q = float("nan")

        # Basic frame progress bar; auto-disabled off-TTY so nohup/piped logs
        # stay clean. [train]/[ckpt] lines go through pbar.write so they print
        # above the bar instead of breaking it.
        pbar = tqdm(
            total=frames,
            unit="frame",
            desc="training"
            #dynamic_ncols=True,
            #disable=not sys.stderr.isatty(),
            #mininterval=1.0,
        )

        while frame < frames:
            frame += 1
            eps = epsilon(frame)
            fringe = res.fringe
            train_occ.record(len(fringe), len(env.reservoir))
            if torch.rand((), generator=ep_rng).item() < eps:
                action = int(torch.randint(len(fringe), (1,), generator=ep_rng))
            else:
                action = self.greedy_action(inst_idx, fringe)

            nxt = env.step(action)
            ep_return += nxt.reward
            self.replay.push(
                Transition(
                    inst=inst_idx,
                    fringe=tuple(fringe),
                    action=action,
                    reward=nxt.reward,
                    next_fringe=tuple(nxt.fringe),
                    done=nxt.done,
                )
            )

            if nxt.done:
                history["episode_return"].append(ep_return)
                history["episode_frame"].append(frame)
                history["episode_expansions"].append(int(nxt.info["expansions"]))
                history["episode_goal"].append(bool(nxt.info["goal_found"]))
                history["episode_inst"].append(self.instances[inst_idx].name)
                ep_return = 0.0
                env_cursor = (env_cursor + 1) % len(env_order)
                inst_idx = env_order[env_cursor]
                env = self.envs[inst_idx]
                res = env.reset()
                while res.done:  # degenerate instant-goal reset
                    env_cursor = (env_cursor + 1) % len(env_order)
                    inst_idx = env_order[env_cursor]
                    env = self.envs[inst_idx]
                    res = env.reset()
            else:
                res = nxt

            pbar.update(1)
            pbar.set_postfix(
                {"eps": f"{eps:.3f}", "td": f"{last_td:.4f}", "q": f"{last_q:.2f}"}
            )

            if len(self.replay) >= self.warmup and frame % self.update_every == 0:
                # Fix the stability probe batch once, the first time we learn,
                # so its scalars are comparable across the whole run.
                if self.log_stability and self._probe_batch is None:
                    try:
                        self._probe_batch = self.replay.sample(self.batch_size)
                    except Exception:
                        self._probe_batch = None
                loss_acc.append(self._update())
            if frame % self.target_sync == 0:
                self.target.load_state_dict(self.model.state_dict())

            if frame % n_checkpoints == 0 and loss_acc:
                avg = {
                    k: sum(d[k] for d in loss_acc) / len(loss_acc)
                    for k in loss_acc[0]
                }
                history["frame"].append(frame)
                history["epsilon"].append(eps)
                for k, v in avg.items():
                    history[k].append(v)
                loss_acc = []
                last_td, last_q = avg["td_loss"], avg["q_mean"]
                fps = frame / (time.time() - t0)
                pbar.write(
                    f"[train] frame {frame}/{frames} eps={eps:.3f} "
                    f"td={last_td:.4f} q={last_q:.2f} ({fps:.0f} fps)"
                )

            if frame in ckpt_frames:
                ck = self.evaluate(frame)
                ck["summary"]["train_occupancy"] = train_occ.summary()
                checkpoints.append(ck)
                if self.log_stability and self._probe_batch is not None:
                    try:
                        sp = self._stability_probe(frame)
                        stability.append(sp)
                        pbar.write(
                            f"[stability] frame {frame} q_mean={sp['q_mean']:.3f} "
                            f"q_max={sp['q_max']:.3f} td={sp['td_loss']:.4f} "
                            f"grad_norm={sp['grad_norm']} "
                            f"target_online_l2={sp['target_online_l2']:.4f}"
                        )
                    except Exception as exc:  # instrumentation must never crash a run
                        pbar.write(f"[stability] WARN probe failed: {exc}")
                pbar.write(f"[ckpt] {json.dumps(ck['summary'])}")
                vo = ck["summary"]["val_occupancy"]
                pbar.write(
                    f"[occupancy] frame {frame} val fringe (F={self.fringe_size}): "
                    f"max={vo['max']} mean={vo['mean']} "
                    f"p50={vo['p50']} p90={vo['p90']} p99={vo['p99']} "
                    f"res_max={vo['reservoir_max']} "
                    f"binds={'YES' if vo['binds'] else 'no'}"
                )
                val_exp = ck["summary"]["val_total_expansions"]
                rho = ck["summary"]["val_spearman_all"]
                if val_exp < best_val_exp:
                    best_val_exp = val_exp
                    self._save_model(out_dir / "best_by_expansions.pt", frame, ck)
                if rho is not None and rho > best_spearman:
                    best_spearman = rho
                    self._save_model(out_dir / "best_by_spearman.pt", frame, ck)
                with (out_dir / "history.json").open("w") as fh:
                    json.dump(
                        {"history": history, "checkpoints": checkpoints,
                         "stability": stability}, fh, indent=1
                    )

        pbar.close()
        # Final (last-frame) weights, so cross-run comparisons can be STAGE-
        # MATCHED (same frame) instead of comparing different-frame best_by_*
        # checkpoints (restores the last.pt dropped in 12d15ea).
        self._save_model(out_dir / "last.pt", frame, None)
        with (out_dir / "history.json").open("w") as fh:
            json.dump({"history": history, "checkpoints": checkpoints,
                       "stability": stability}, fh, indent=1)
        return {"history": history, "checkpoints": checkpoints,
                "stability": stability}

    # ---------- evaluation ----------

    @torch.no_grad()
    def greedy_rollout(
        self,
        inst_idx: int,
        seed: int,
        occ_accum: Optional[OccupancyCounter] = None,
    ) -> Dict[str, object]:
        env = FringeEnv(
            self.instances[inst_idx],
            fringe_size=self.fringe_size,
            seed=seed,
            expansion_cap=self.eval_expansion_cap,
        )
        occ = OccupancyCounter(self.fringe_size)
        res = env.reset(seed=seed)
        while not res.done:
            occ.record(len(res.fringe), len(env.reservoir))
            res = env.step(self.greedy_action(inst_idx, res.fringe))
        if occ_accum is not None:
            occ_accum.merge(occ)
        return {
            "expansions": int(res.info["expansions"]),
            "goal_found": bool(res.info["goal_found"]),
            "capped": int(res.info["expansions"]) >= self.eval_expansion_cap,
            "occupancy": occ.summary(),
        }

    @torch.no_grad()
    def state_scores(
        self, inst_idx: int, state_ids: Sequence[int], chunk: int = 1024
    ) -> torch.Tensor:
        """Per-state score via singleton fringes (frontier ctx = own emb)."""
        self.model.eval()
        out: List[torch.Tensor] = []
        for c0 in range(0, len(state_ids), chunk):
            ids = state_ids[c0 : c0 + chunk]
            logits, _ = self._forward(
                self.model, [(inst_idx, [s]) for s in ids]
            )
            out.append(logits.cpu())
        return torch.cat(out)

    @torch.no_grad()
    def evaluate(self, frame: int, n_score_sample: int = 4096) -> Dict[str, object]:
        self.model.eval()
        per_instance: Dict[str, Dict[str, object]] = {}
        val_total = 0.0
        val_occ = OccupancyCounter(self.fringe_size)
        # SELECTION eval set. Train-based selection (the regime/study default,
        # self.select_on_train) picks the checkpoint on the TRAIN deploy-faithful
        # metric regardless of any val set — held-out instances are diagnostic
        # only (the test per-regime plots are the sole overfitting detector).
        # Otherwise: use val if supplied, else fall back to train. Train-based
        # selection is BLIND to generalization, so it is flagged loudly
        # (selection_on_train) in the summary. (train is always non-empty.)
        select_on_train = self.select_on_train or not self.val_ids
        eval_ids = self.train_ids if select_on_train else self.val_ids
        # Average each instance's greedy rollout over K refill seeds so the
        # convergence curve and best_by_expansions selection are robust to
        # reservoir-refill noise (a single seed per checkpoint made the v1 curve
        # oscillate wildly). K=1 reproduces the prior single-seed behavior.
        k = self.eval_refill_seeds
        for vid in eval_ids:
            inst = self.instances[vid]
            rolls = [
                self.greedy_rollout(vid, seed=10_000 + frame + s, occ_accum=val_occ)
                for s in range(k)
            ]
            exps = [int(r["expansions"]) for r in rolls]
            mean_exp = sum(exps) / len(exps)
            bfs = bfs_expansions(inst)
            per_instance[inst.name] = {
                "greedy": rolls[-1],  # representative single rollout
                "greedy_mean_expansions": round(mean_exp, 3),
                "greedy_expansions_per_seed": exps,
                "bfs_expansions": int(bfs["expansions"]),
                "optimal_expansions": inst.optimal_expansions(),
                "occupancy": rolls[-1]["occupancy"],
            }
            val_total += mean_exp
        val_total = round(val_total, 3)

        # Spearman + score stats on the selection set's states (singleton
        # fringes) — val if held out, else the train fallback set.
        rho_all = rho_reach = None
        score_stats: Dict[str, object] = {}
        if eval_ids:
            vid = eval_ids[0]
            inst = self.instances[vid]
            g = torch.Generator().manual_seed(self.seed + 7)
            n = inst.n_states
            sample = (
                torch.randperm(n, generator=g)[:n_score_sample].tolist()
                if n > n_score_sample
                else list(range(n))
            )
            scores = self.state_scores(vid, sample)
            d = torch.tensor([inst.distance[s] for s in sample])
            unreach = d >= UNREACHABLE_DISTANCE
            d_max_finite = d[~unreach].max() if bool((~unreach).any()) else torch.tensor(0.0)
            d_clamped = torch.where(unreach, d_max_finite + 1.0, d)
            rho_all = float(spearmanr(scores.numpy(), (-d_clamped).numpy()).statistic)
            if bool((~unreach).any()) and int((~unreach).sum()) > 2:
                rho_reach = float(
                    spearmanr(
                        scores[~unreach].numpy(), (-d[~unreach]).numpy()
                    ).statistic
                )
            score_stats = {
                "mean": float(scores.mean()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "std": float(scores.std()),
                "reachable_mean": (
                    float(scores[~unreach].mean()) if bool((~unreach).any()) else None
                ),
                "unreachable_mean": (
                    float(scores[unreach].mean()) if bool(unreach.any()) else None
                ),
                "n_sampled": len(sample),
                "n_unreachable_sampled": int(unreach.sum()),
            }

        train_rollouts = {
            self.instances[tid].name: self.greedy_rollout(tid, seed=20_000 + frame)
            for tid in self.train_ids
        }

        return {
            "frame": int(frame),
            "summary": {
                "frame": int(frame),
                "val_total_expansions": val_total,
                "val_eval_refill_seeds": k,
                "val_spearman_all": rho_all,
                "val_spearman_reachable": rho_reach,
                "val_occupancy": val_occ.summary(),
                "selection_on_train": select_on_train,
            },
            "val_per_instance": per_instance,
            "train_greedy": train_rollouts,
            "score_stats": score_stats,
        }

    # ---------- persistence ----------

    def _save_model(self, path: Path, frame: int, ckpt: Optional[Dict[str, object]]) -> None:
        """RLFrontierTrainer.load_model-compatible payload."""
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "node_input_dim": self.model.encoder.input_proj.in_features,
                "hidden_dim": self.model.encoder.input_proj.out_features,
                "gnn_layers": len(self.model.encoder.layers),
                "conv_type": self.model.encoder.conv_type,
                "pooling_type": self.model.pooling_type,
                "dataset_type": self.model.dataset_type,
                "edge_emb_dim": self.model.edge_emb_dim,
                "num_edge_labels": self.model.num_edge_labels,
                "num_node_labels": self.model.num_node_labels,
                "use_global_context": self.model.use_global_context,
                "mlp_depth": (len(self.model.policy_head) - 1) // 2,
                "use_goal_separate_input": self.model.use_goal_separate_input,
            },
            "metrics": {"frame": int(frame), "checkpoint": ckpt},
        }
        torch.save(payload, path)
