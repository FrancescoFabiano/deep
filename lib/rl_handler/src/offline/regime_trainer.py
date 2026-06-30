"""Multi-regime Double-DQN trainer (P1, value-only). See the P1 brief + DESIGN.md.

RegimeDQNTrainer subclasses OfflineDQNTrainer and changes ONLY the data-generation
side: instead of one FringeEnv per instance, it runs one RedrawFringeEnv per
(instance × available regime), interleaved by a weighted round-robin into the one
shared replay buffer and one shared model. It REUSES the base _update (loss /
target / Double-DQN), evaluate, _save_model and the whole FrontierPolicyNetwork /
ONNX export path verbatim — dqn.py and the model are byte-identical, so the diff
cannot touch logit/head/target/ONNX code (that is the point of subclassing rather
than editing dqn.train).

Value-only this phase:
  target_centering='absolute'   -> signal_mode='basic'                (arm a)
  target_centering='fringe_mean'-> signal_mode='rank-rl'+'advantage'  (arm b)
No order term (lambda_ord=0); P2 adds it.

Model selection is by the DEPLOY-FAITHFUL eval (greedy_rollout overridden to the
children-first refill_mode='heuristic' path, reservoir scored by the live model),
never by training-env return — the redraw regimes are train-distribution
generators only.
"""

from __future__ import annotations

import csv as csvmod
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from src.offline.dqn import OfflineDQNTrainer
from src.offline.regimes import (
    REGIMES,
    HFSDiag,
    RedrawFringeEnv,
    bfs_usable_fraction,
    depth_dstar_var_explained,
    dfs_preorder_rank,
    eligible_nodes,
    single_distinct_dstar,
)
from src.offline.replay import Transition
from src.offline.tree_env import (
    UNREACHABLE_DISTANCE,
    FringeEnv,
    OccupancyCounter,
    bfs_frontier_max,
)


# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Windowed per-regime instrumentation (reset each checkpoint)."""
    n_seen: int = 0            # fringes (states) observed
    n_full_F: int = 0          # fringes with exactly F slots (beam binds)
    n_usable: int = 0          # fringes with >=2 distinct d* (carry ranking signal)
    n_dropped: int = 0         # dropped by the same-distance filter
    n_dedup_drop: int = 0      # dropped by fringe-level (member-set) dedup
    pool_size_sum: int = 0     # Σ |P| at beam-build time
    n_pool_ge_F: int = 0       # steps where |P| >= F (redraw controls all F slots)
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    n_episodes: int = 0
    n_goal: int = 0

    def observe(self, fringe, inst, F, pool_size) -> None:
        self.n_seen += 1
        if len(fringe) == F:
            self.n_full_F += 1
        if len({inst.distance[s] for s in fringe}) >= 2:
            self.n_usable += 1
        self.pool_size_sum += int(pool_size)
        if pool_size >= F:
            self.n_pool_ge_F += 1

    def summary(self, F) -> Dict[str, object]:
        n = max(1, self.n_seen)
        ne = max(1, self.n_episodes)
        return {
            "n_fringes": self.n_seen,
            "frac_full_F": round(self.n_full_F / n, 3),
            "frac_pool_ge_F": round(self.n_pool_ge_F / n, 3),
            "mean_pool_size": round(self.pool_size_sum / n, 2),
            "usable_frac": round(self.n_usable / n, 3),
            "same_dist_drop_rate": round(self.n_dropped / (self.n_seen + self.n_dropped), 3)
            if (self.n_seen + self.n_dropped) else 0.0,
            "fringe_dedup_drops": self.n_dedup_drop,
            "n_episodes": self.n_episodes,
            "mean_ep_len": round(sum(self.episode_lengths) / ne, 2),
            "mean_ep_return": round(sum(self.episode_returns) / ne, 3),
            "goal_rate": round(self.n_goal / ne, 3),
        }


@dataclass
class _Env:
    inst: int
    regime: str
    env: RedrawFringeEnv


@dataclass
class _DiagProblem:
    """A diagnostic (NON-SELECTING) problem: one dataset instance carrying its
    own per-regime RedrawFringeEnv set, kept separate from the training envs so a
    diagnostic rollout never perturbs training RNG / episode state."""
    inst: int
    split: str                       # 'train' or 'test'
    name: str
    optimal: Optional[int]
    fmax: int                        # max simultaneous live frontier (policy-free)
    envs: Dict[str, RedrawFringeEnv]


# ---------------------------------------------------------------------------

class RegimeDQNTrainer(OfflineDQNTrainer):
    def __init__(
        self,
        *args,
        regimes: Sequence[str] = REGIMES,
        mixture_weights: Optional[Dict[str, float]] = None,
        target_centering: str = "absolute",
        same_distance_keep: bool = False,
        bfs_exclude_usable_frac: float = 0.02,
        eval_exploration_nodes: Optional[int] = None,
        train_expansion_cap: Optional[int] = None,
        diag_test_ids: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        if target_centering not in ("absolute", "fringe_mean"):
            raise ValueError(
                f"target_centering must be 'absolute' or 'fringe_mean', got "
                f"{target_centering!r}"
            )
        # Route value-only arms through the EXISTING signal modes (no new loss).
        if target_centering == "absolute":
            kwargs["signal_mode"] = "basic"
            kwargs.pop("rank_variant", None)
        else:
            kwargs["signal_mode"] = "rank-rl"
            kwargs["rank_variant"] = "advantage"
        super().__init__(*args, **kwargs)

        bad = [r for r in regimes if r not in REGIMES]
        if bad:
            raise ValueError(f"unknown regimes {bad}; valid: {REGIMES}")
        self.regimes = list(regimes)
        self.target_centering = target_centering
        self.same_distance_keep = bool(same_distance_keep)
        self.bfs_exclude_usable_frac = float(bfs_exclude_usable_frac)
        F = self.fringe_size
        self.eval_exploration_nodes = (
            int(eval_exploration_nodes)
            if eval_exploration_nodes is not None
            else max(1, int(math.floor(F * 0.1)))   # deploy default ~10%
        )
        self.train_expansion_cap = (
            int(train_expansion_cap) if train_expansion_cap is not None else None
        )
        w = mixture_weights or {r: 1.0 for r in self.regimes}
        self.mixture_weights = {r: float(w.get(r, 0.0)) for r in self.regimes}

        # Fringe-level dedup: a member-SET never enters replay twice (per
        # instance). Keyed by (inst, hash(frozenset(fringe))); persists for the run.
        self._seen_fringes: set = set()

        # ---- per-instance static structure + exclusions (computed predicates) ----
        self.dfs_rank: Dict[int, List[int]] = {}
        self.eligibility: Dict[int, Dict[str, object]] = {}
        self.fmax_by_inst: Dict[int, int] = {}
        envs: List[_Env] = []
        self.hfs_diags: Dict[str, HFSDiag] = {}
        included: List[int] = []
        excluded_subF: List[str] = []
        bfs_excluded: List[str] = []
        for i in self.train_ids:
            inst = self.instances[i]
            elig = eligible_nodes(inst)
            ve = depth_dstar_var_explained(inst, elig)
            usable, total = bfs_usable_fraction(inst, elig, F)
            uf = (usable / total) if total else 0.0
            sub_F = len(elig) < F
            bfs_out = (total == 0) or (uf < self.bfs_exclude_usable_frac)
            # fmax = max simultaneous live frontier (policy-free). F>fmax => the
            # live pool can never reach F, so the beam runs SHORT on this instance
            # (no fabrication; kept honest-short — see RedrawFringeEnv).
            fmax = bfs_frontier_max(inst)
            self.eligibility[i] = {
                "name": inst.name, "E": len(elig), "var_expl": round(ve, 4),
                "bfs_usable_frac": round(uf, 4), "sub_F": sub_F,
                "bfs_excluded": bfs_out, "fmax": int(fmax),
            }
            if sub_F:
                excluded_subF.append(inst.name)
                continue
            included.append(i)
            self.fmax_by_inst[i] = int(fmax)
            self.dfs_rank[i] = dfs_preorder_rank(inst)
            avail = [
                r for r in self.regimes
                if not (r == "bfs" and bfs_out)
            ]
            if bfs_out and "bfs" in self.regimes:
                bfs_excluded.append(inst.name)
            for r in avail:
                diag = None
                if r == "hfs":
                    diag = self.hfs_diags.setdefault(r, HFSDiag())
                env = RedrawFringeEnv(
                    inst, fringe_size=F, seed=self.seed + 1000 * i + hash(r) % 997,
                    regime=r, dfs_rank=self.dfs_rank[i],
                    expansion_cap=(self.train_expansion_cap
                                   if self.train_expansion_cap is not None
                                   else 2 * inst.n_states),
                    hfs_diag=diag,
                )
                envs.append(_Env(inst=i, regime=r, env=env))
        if not envs:
            raise ValueError(
                "no trainable (instance × regime) envs after eligibility filtering "
                f"(sub-F excluded: {excluded_subF})"
            )
        self.regime_envs = envs
        self.included_train = included
        self.excluded_subF = excluded_subF
        self.bfs_excluded_names = bfs_excluded
        # per-instance available regimes + per-instance renormalised weights
        self.avail_by_inst: Dict[int, List[str]] = {}
        self.regime_w_by_inst: Dict[int, List[float]] = {}
        for i in included:
            avail = sorted({e.regime for e in envs if e.inst == i})
            self.avail_by_inst[i] = avail
            ws = [self.mixture_weights[r] for r in avail]
            tot = sum(ws) or 1.0
            self.regime_w_by_inst[i] = [x / tot for x in ws]
        self._env_by_key: Dict[tuple, RedrawFringeEnv] = {
            (e.inst, e.regime): e.env for e in envs
        }
        self.regime_stats: Dict[str, RegimeStats] = {}
        self._reset_regime_window()

        # ---- DIAGNOSTIC (non-selecting) per-regime surface ----
        # Build a SEPARATE per-(problem × regime) RedrawFringeEnv set for the
        # TRAIN split (self.train_ids) and, if supplied, the diagnostic TEST
        # split (diag_test_ids). These envs are used ONLY by _diagnostic_surface
        # and never by the training loop or model selection — model selection is
        # the deploy-faithful val eval (evaluate/greedy_rollout), full stop. Test
        # envs are regime-shaped and therefore OFF-DISTRIBUTION vs deployment;
        # they exist purely to read how the current model orders fringes on held-
        # out problems, not to score it.
        self.diag_test_ids = list(diag_test_ids) if diag_test_ids else []
        self.diag_problems: List[_DiagProblem] = self._build_diag_problems()

    # ---- windowed instrumentation ----
    def _reset_regime_window(self) -> None:
        self.regime_stats = {r: RegimeStats() for r in self.regimes}
        for r in ("hfs",):
            if r in self.hfs_diags:
                new = HFSDiag()
                self.hfs_diags[r] = new
                for e in self.regime_envs:
                    if e.regime == r:
                        e.env.hfs_diag = new
        self.reset_order_stats()            # windowed order-aux pair/skip counters

    # ---- weighted (instance uniform × regime ∝ w) scheduler ----
    def _pick(self, rng: random.Random) -> tuple:
        i = rng.choice(self.included_train)
        avail = self.avail_by_inst[i]
        r = rng.choices(avail, weights=self.regime_w_by_inst[i], k=1)[0]
        return i, r

    # ---- deploy-faithful rollout (children-first + heuristic reservoir) ----
    @torch.no_grad()
    def greedy_rollout(self, inst_idx, seed, occ_accum=None):
        F = self.fringe_size

        def score_fn(res_ids):
            if not len(res_ids):
                return []
            return self.state_scores(inst_idx, list(res_ids)).tolist()

        env = FringeEnv(
            self.instances[inst_idx], fringe_size=F, seed=seed,
            expansion_cap=self.eval_expansion_cap,
            refill_mode="heuristic", reservoir_score_fn=score_fn,
            exploration_nodes=self.eval_exploration_nodes,
        )
        occ = OccupancyCounter(F)
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

    # ---- DIAGNOSTIC surface: per-(regime × problem), NON-SELECTING ----
    def _build_diag_problems(self) -> List[_DiagProblem]:
        """One _DiagProblem per dataset instance (train split + diagnostic test
        split), each with a fresh RedrawFringeEnv for EVERY regime in
        self.regimes (bfs included regardless of training eligibility — this is a
        diagnostic, off-distribution is fine). Static per-instance structure is
        recomputed here so the surface is self-contained and never reuses (and
        thus never disturbs) the training envs."""
        problems: List[_DiagProblem] = []
        F = self.fringe_size
        splits = [("train", self.train_ids), ("test", self.diag_test_ids)]
        for split, ids in splits:
            for i in ids:
                inst = self.instances[i]
                fmax = bfs_frontier_max(inst)
                rank = dfs_preorder_rank(inst)
                envs: Dict[str, RedrawFringeEnv] = {}
                for ri, r in enumerate(self.regimes):
                    diag = HFSDiag() if r == "hfs" else None
                    # fixed per (problem, regime) seed -> the curve over
                    # checkpoints reflects the MODEL changing, not env noise.
                    dseed = self.seed + 70_000 + 131 * i + 7 * ri
                    env = RedrawFringeEnv(
                        inst, fringe_size=F, seed=dseed,
                        regime=r, dfs_rank=rank,
                        expansion_cap=self.eval_expansion_cap,
                        hfs_diag=diag,
                    )
                    env.diag_seed = dseed
                    envs[r] = env
                problems.append(_DiagProblem(
                    inst=i, split=split, name=inst.name,
                    optimal=inst.optimal_expansions(),
                    fmax=int(fmax), envs=envs,
                ))
        return problems

    @torch.no_grad()
    def _diag_regime_rollout(self, inst_idx: int, env: RedrawFringeEnv,
                             seed: int) -> Dict[str, object]:
        """Greedy rollout of the CURRENT model on one regime env. Records node
        economy (raw expansions), goal, the -1-stream return, and the order
        usable-pair / skip fractions over the fringes the rollout actually
        visited (same eligibility rule as the order auxiliary). Beams are live-
        pool only (possibly short); there is no padding. Pure read: no grad, no
        replay, no training-state mutation."""
        self.model.eval()
        res = env.reset(seed=seed)
        ret = 0.0
        cand_pairs = usable_pairs = seen_fr = skip_fr = 0
        inst = self.instances[inst_idx]
        # Rank-fidelity accumulators: the model's slot SCORES vs -d* over the
        # ORDER-ELIGIBLE slots (finite d*; floored-unreachables excluded), pooled
        # across the fringes this rollout visits. One tie-aware Spearman per
        # (regime × problem × checkpoint) -> "is the d* rank preserved through
        # the Q-values, per environment".
        rf_scores: List[float] = []
        rf_neg_dstar: List[float] = []
        while not res.done:
            fringe = res.fringe
            c, u, skipped = self._order_pair_counts(inst_idx, fringe)
            cand_pairs += c
            usable_pairs += u
            seen_fr += 1
            skip_fr += skipped
            # One forward serves BOTH the rank-fidelity slot scores and the
            # greedy action (argmax) — no extra forward; behaviour is identical
            # to greedy_action.
            logits, _ = self._forward(self.model, [(inst_idx, fringe)])
            slot_scores = logits.detach().cpu()
            for k, s in enumerate(fringe):
                d = inst.distance[s]
                if d >= UNREACHABLE_DISTANCE:
                    continue
                rf_scores.append(float(slot_scores[k]))
                rf_neg_dstar.append(-float(d))
            res = env.step(int(torch.argmax(slot_scores).item()))
            ret += res.reward
        expansions = int(res.info["expansions"])
        goal = bool(res.info["goal_found"])
        # <2 distinct eligible d* -> no ordering to measure -> None (not plotted).
        # spearmanr can also return nan on degenerate (constant) scores -> None.
        rank_spearman = None
        if len(set(rf_neg_dstar)) >= 2:
            rho = float(spearmanr(rf_scores, rf_neg_dstar).statistic)
            rank_spearman = round(rho, 4) if math.isfinite(rho) else None
        return {
            "node_economy": expansions,
            "goal_found": goal,
            "goal_rate": 1.0 if goal else 0.0,
            "ret": round(ret, 3),
            "capped": expansions >= self.eval_expansion_cap,
            "rank_spearman": rank_spearman,
            "order_usable_pair_frac": round(usable_pairs / max(1, cand_pairs), 4),
            "order_skip_frac": round(skip_fr / max(1, seen_fr), 4),
        }

    def _order_pair_counts(self, inst_idx: int, fringe: Sequence[int]) -> tuple:
        """(candidate_pairs, usable_pairs, skipped) over the ORDER-ELIGIBLE slots
        of one fringe, mirroring dqn._order_eligible_slots: unreachable d* maps to
        a worst-sorting sentinel; <2 distinct eligible values -> skipped (no
        ordering signal). Usable pairs = strictly-ordered (d_i<d_j) pairs."""
        inst = self.instances[inst_idx]
        raw = [inst.distance[s] for s in fringe]
        positions = list(range(len(raw)))
        finite = [raw[k] for k in positions if raw[k] < UNREACHABLE_DISTANCE]
        sentinel = (max(finite) + 1.0) if finite else 1.0
        dvals = [raw[k] if raw[k] < UNREACHABLE_DISTANCE else sentinel
                 for k in positions]
        m = len(positions)
        cand = m * (m - 1) // 2
        if len(set(dvals)) < 2:
            return cand, 0, 1
        usable = sum(1 for a in range(m) for b in range(m) if dvals[a] < dvals[b])
        return cand, usable, 0

    def _diagnostic_surface(self, frame: int) -> Dict[str, List[Dict[str, object]]]:
        """At one checkpoint, run every (regime × problem) diagnostic rollout and
        return rows grouped by split. NEVER feeds checkpoint selection — the
        caller stores it under a clearly-labelled non-selecting field."""
        out: Dict[str, List[Dict[str, object]]] = {"train": [], "test": []}
        for p in self.diag_problems:
            for r in self.regimes:
                env = p.envs[r]
                roll = self._diag_regime_rollout(p.inst, env, seed=env.diag_seed)
                opt = p.optimal
                ne = roll["node_economy"]
                out[p.split].append({
                    "frame": int(frame),
                    "split": p.split,
                    "regime": r,
                    "problem": p.name,
                    "fmax": p.fmax,
                    "optimal_expansions": opt,
                    "node_economy": ne,
                    "node_economy_ratio": (round(ne / opt, 4) if opt else None),
                    "rank_spearman": roll["rank_spearman"],
                    "goal_found": roll["goal_found"],
                    "goal_rate": roll["goal_rate"],
                    "ret": roll["ret"],
                    "ret_ratio": (round(roll["ret"] / opt, 4) if opt else None),
                    "capped": roll["capped"],
                    "order_usable_pair_frac": roll["order_usable_pair_frac"],
                    "order_skip_frac": roll["order_skip_frac"],
                })
        return out

    _DIAG_COLS = (
        "frame", "split", "regime", "problem", "fmax",
        "optimal_expansions", "node_economy", "node_economy_ratio",
        "rank_spearman", "goal_found",
        "goal_rate", "ret", "ret_ratio", "capped", "order_usable_pair_frac",
        "order_skip_frac",
    )

    def _write_diag_tables(self, out_dir: Path,
                           diag_rows: Dict[str, List[Dict[str, object]]]) -> None:
        """Persist the accumulated per-problem×regime diagnostic rows (raw
        metrics, one row per problem×regime×checkpoint) to a CSV per split. Raw
        only — normalization happens at plot time."""
        for split, rows in diag_rows.items():
            if not rows:
                continue
            path = out_dir / f"diag_per_regime_{split}.csv"
            with path.open("w", newline="") as fh:
                w = csvmod.DictWriter(fh, fieldnames=list(self._DIAG_COLS))
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k) for k in self._DIAG_COLS})

    def _regime_instrumentation(self) -> Dict[str, object]:
        out = {"per_regime": {}, "hfs": {}, "eligibility": list(self.eligibility.values())}
        for r in self.regimes:
            if self.regime_stats[r].n_seen or self.regime_stats[r].n_dropped:
                out["per_regime"][r] = self.regime_stats[r].summary(self.fringe_size)
        for r, diag in self.hfs_diags.items():
            out["hfs"][r] = diag.summary()
        # per-instance fmax (live frontier ceiling). F>fmax => the beam runs short
        # on that instance (no padding); the short-ness is visible in the regime
        # composition table's poolGEF/fullF columns.
        out["fmax"] = [
            {"name": self.instances[i].name, "fmax": self.fmax_by_inst[i],
             "short": self.fmax_by_inst[i] < self.fringe_size}
            for i in self.included_train
        ]
        # P2 order-aux: usable-pair fraction + <2-distinct skip count over the
        # order-eligible slots. Zeroed when lambda_ord==0 (no order loss).
        out["order"] = {"lambda_ord": self.lambda_ord, **self.order_stats()}
        return out

    # ---- training loop: weighted round-robin over (instance × regime) ----
    def train(self, frames, n_checkpoints, epsilon, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_frames = sorted(
            {int(math.ceil((k + 1) * frames / n_checkpoints)) for k in range(n_checkpoints)}
        )
        history: Dict[str, list] = {
            "frame": [], "td_loss": [], "q_mean": [], "target_mean": [], "epsilon": [],
        }
        checkpoints: List[Dict[str, object]] = []
        # DIAGNOSTIC, NON-SELECTING: per-(regime × problem) rows accumulated over
        # checkpoints for the train + (optional) test splits. Kept in a SEPARATE
        # structure from `checkpoints` and never read by the best_* selection
        # below, which uses only the deploy-faithful val eval (ck["summary"]).
        diag_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "test": []}
        best_val_exp = float("inf")
        best_spearman = -float("inf")

        sched_rng = random.Random(self.seed + 99)
        ep_rng = torch.Generator().manual_seed(self.seed + 1)
        cur_i, cur_r = self._pick(sched_rng)
        env = self._env_by_key[(cur_i, cur_r)]
        res = env.reset(seed=self.seed)
        while res.done:
            cur_i, cur_r = self._pick(sched_rng)
            env = self._env_by_key[(cur_i, cur_r)]
            res = env.reset()
        ep_return, ep_len = 0.0, 0

        loss_acc: List[Dict[str, float]] = []
        t0 = time.time()
        frame = 0
        last_td = last_q = float("nan")
        pbar = tqdm(total=frames, unit="frame", desc=f"regime-train[{self.target_centering}]")

        while frame < frames:
            frame += 1
            eps = epsilon(frame)
            fringe = res.fringe
            # slot-aligned pad mask for THIS state fringe (captured from res.info
            inst = self.instances[cur_i]
            st = self.regime_stats[cur_r]
            st.observe(fringe, inst, self.fringe_size, env.last_pool_size)

            drop = (not self.same_distance_keep) and single_distinct_dstar(inst, fringe)
            if drop:
                st.n_dropped += 1
            # fringe-level dedup: the identical member SET must not enter replay
            # twice (intra-beam dedup already holds). Keyed per instance.
            dkey = (cur_i, hash(frozenset(fringe)))
            is_dup = dkey in self._seen_fringes
            if is_dup:
                st.n_dedup_drop += 1
            else:
                self._seen_fringes.add(dkey)

            if torch.rand((), generator=ep_rng).item() < eps:
                action = int(torch.randint(len(fringe), (1,), generator=ep_rng))
            else:
                action = self.greedy_action(cur_i, fringe)

            nxt = env.step(action)
            ep_return += nxt.reward
            ep_len += 1
            if not drop and not is_dup:
                self.replay.push(Transition(
                    inst=cur_i, fringe=tuple(fringe), action=action,
                    reward=nxt.reward, next_fringe=tuple(nxt.fringe),
                    done=nxt.done, regime=cur_r,
                ))

            if nxt.done:
                st.n_episodes += 1
                st.episode_returns.append(ep_return)
                st.episode_lengths.append(ep_len)
                if bool(nxt.info["goal_found"]):
                    st.n_goal += 1
                ep_return, ep_len = 0.0, 0
                cur_i, cur_r = self._pick(sched_rng)
                env = self._env_by_key[(cur_i, cur_r)]
                res = env.reset()
                while res.done:
                    cur_i, cur_r = self._pick(sched_rng)
                    env = self._env_by_key[(cur_i, cur_r)]
                    res = env.reset()
            else:
                res = nxt

            pbar.update(1)
            pbar.set_postfix({"eps": f"{eps:.3f}", "td": f"{last_td:.4f}", "q": f"{last_q:.2f}"})

            if len(self.replay) >= self.warmup and frame % self.update_every == 0:
                loss_acc.append(self._update())
            if frame % self.target_sync == 0:
                self.target.load_state_dict(self.model.state_dict())

            if frame % n_checkpoints == 0 and loss_acc:
                avg = {k: sum(d[k] for d in loss_acc) / len(loss_acc) for k in loss_acc[0]}
                history["frame"].append(frame)
                history["epsilon"].append(eps)
                for k in ("td_loss", "q_mean", "target_mean"):
                    history[k].append(avg[k])
                loss_acc = []
                last_td, last_q = avg["td_loss"], avg["q_mean"]
                fps = frame / (time.time() - t0)
                pbar.write(f"[train] frame {frame}/{frames} eps={eps:.3f} "
                           f"td={last_td:.4f} q={last_q:.2f} ({fps:.0f} fps)")

            if frame in ckpt_frames:
                ck = self.evaluate(frame)
                ck["regime_instrumentation"] = self._regime_instrumentation()
                checkpoints.append(ck)
                pbar.write(f"[ckpt] {json.dumps(ck['summary'])}")
                self._print_regime_tables(pbar, frame, ck["regime_instrumentation"])
                val_exp = ck["summary"]["val_total_expansions"]
                rho = ck["summary"]["val_spearman_all"]
                # ---- SELECTION (deploy-faithful val eval ONLY) ----
                if val_exp < best_val_exp:
                    best_val_exp = val_exp
                    self._save_model(out_dir / "best_by_expansions.pt", frame, ck)
                if rho is not None and rho > best_spearman:
                    best_spearman = rho
                    self._save_model(out_dir / "best_by_spearman.pt", frame, ck)
                # ---- DIAGNOSTIC surface (computed AFTER selection; cannot feed
                # it). Append this checkpoint's rows, persist the raw per-problem
                # tables, and print a compact summary. ----
                surf = self._diagnostic_surface(frame)
                for split in ("train", "test"):
                    diag_rows[split].extend(surf[split])
                self._write_diag_tables(out_dir, diag_rows)
                self._print_diag_tables(pbar, frame, surf)
                with (out_dir / "history.json").open("w") as fh:
                    json.dump({"history": history, "checkpoints": checkpoints,
                               "config": self._regime_config(),
                               "diag_per_regime": self._diag_payload(diag_rows)},
                              fh, indent=1)
                self._reset_regime_window()

        pbar.close()
        self._save_model(out_dir / "last.pt", frame, None)
        with (out_dir / "history.json").open("w") as fh:
            json.dump({"history": history, "checkpoints": checkpoints,
                       "config": self._regime_config(),
                       "diag_per_regime": self._diag_payload(diag_rows)},
                      fh, indent=1)
        # Per-run diagnostic plots (normalized aggregate per regime, per split).
        # Wrapped: a plotting failure must never cost the trained model.
        try:
            from src.offline.diag_plots import plot_diag_curves
            written = plot_diag_curves(diag_rows, out_dir, fringe=self.fringe_size)
            if written:
                tqdm.write(f"[diag-plots] {out_dir}: {', '.join(written)}")
        except Exception as exc:  # non-fatal
            tqdm.write(f"[diag-plots] WARN failed to render: {exc}")
        return {"history": history, "checkpoints": checkpoints,
                "diag_per_regime": self._diag_payload(diag_rows)}

    def _diag_payload(self, diag_rows: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
        """Wrap the accumulated diagnostic rows with an explicit non-selecting
        label so no downstream reader mistakes them for a selection signal."""
        return {
            "_note": (
                "DIAGNOSTIC ONLY — per-(regime × problem) rollouts of the current "
                "model; NEVER feeds checkpoint selection (selection = deploy-"
                "faithful val eval). Test split is regime-shaped / off-"
                "distribution vs deployment."
            ),
            "columns": list(self._DIAG_COLS),
            "train": diag_rows["train"],
            "test": diag_rows["test"],
        }

    @staticmethod
    def _print_diag_tables(pbar, frame, surf) -> None:
        for split in ("train", "test"):
            rows = surf.get(split, [])
            if not rows:
                continue
            # aggregate (normalized) per regime for a one-line readout
            by_r: Dict[str, List[Dict[str, object]]] = {}
            for row in rows:
                by_r.setdefault(row["regime"], []).append(row)
            pbar.write(f"[diag:{split}] frame {frame} (non-selecting) "
                       f"node-econ ratio / goal-rate per regime:")
            for r, rs in by_r.items():
                ratios = [x["node_economy_ratio"] for x in rs
                          if x["node_economy_ratio"] is not None]
                mean_ratio = (sum(ratios) / len(ratios)) if ratios else float("nan")
                goal = sum(x["goal_rate"] for x in rs) / len(rs)
                pbar.write(f"    {r:8s} ne_ratio={mean_ratio:6.2f} "
                           f"goal={goal:4.2f} (n={len(rs)})")

    def _regime_config(self) -> Dict[str, object]:
        return {
            "regimes": self.regimes,
            "mixture_weights": self.mixture_weights,
            "target_centering": self.target_centering,
            "signal_mode": self.signal_mode,
            "rank_variant": self.rank_variant,
            "same_distance_keep": self.same_distance_keep,
            "bfs_exclude_usable_frac": self.bfs_exclude_usable_frac,
            "eval_exploration_nodes": self.eval_exploration_nodes,
            "fringe_size": self.fringe_size,
            "included_train": [self.instances[i].name for i in self.included_train],
            "excluded_subF": self.excluded_subF,
            "bfs_excluded": self.bfs_excluded_names,
            "fmax_by_instance": {
                self.instances[i].name: self.fmax_by_inst[i]
                for i in self.included_train
            },
            "diag_split": {
                "train": [self.instances[i].name for i in self.train_ids],
                "test": [self.instances[i].name for i in self.diag_test_ids],
                "note": "diagnostic (non-selecting); selection = deploy-faithful val",
            },
        }

    @staticmethod
    def _print_regime_tables(pbar, frame, instr) -> None:
        pbar.write(f"[regime] frame {frame} per-regime composition:")
        hdr = (f"  {'regime':8s}{'nfr':>7}{'fullF':>7}{'poolGEF':>8}"
               f"{'mPool':>7}{'usable':>7}{'drop':>6}{'dedup':>7}"
               f"{'epLen':>7}{'epRet':>8}{'goal':>6}")
        pbar.write(hdr)
        for r, s in instr["per_regime"].items():
            pbar.write(
                f"  {r:8s}{s['n_fringes']:>7}{s['frac_full_F']:>7.2f}"
                f"{s['frac_pool_ge_F']:>8.2f}{s['mean_pool_size']:>7.1f}"
                f"{s['usable_frac']:>7.2f}{s['same_dist_drop_rate']:>6.2f}"
                f"{s['fringe_dedup_drops']:>7}"
                f"{s['mean_ep_len']:>7.1f}{s['mean_ep_return']:>8.1f}{s['goal_rate']:>6.2f}"
            )
        if instr["hfs"]:
            pbar.write("  HFS tail: "
                       + " | ".join(
                           f"{r}: stv_true={d['stv_true']} stv_real={d['stv_real']} "
                           f"unsmp_mass={d['unsampled_mass']}"
                           for r, d in instr["hfs"].items()))
        o = instr.get("order")
        if o is not None:
            pbar.write(
                f"  order-aux: lambda={o['lambda_ord']} "
                f"usable_pair_frac={o['order_usable_pair_frac']} "
                f"(pairs {o['order_usable_pairs']}/{o['order_candidate_pairs']}) "
                f"post-mask skip={o['order_skipped_fringes']}/{o['order_seen_fringes']} "
                f"({o['order_skip_frac']})"
            )
        short = [p for p in instr.get("fmax", []) if p.get("short")]
        if short:
            pbar.write("  short-beam instances (fmax < F, run short, no padding):")
            for p in short:
                pbar.write(f"    {p['name']:18s} fmax={p['fmax']:>4}")
