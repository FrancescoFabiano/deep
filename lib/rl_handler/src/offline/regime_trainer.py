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

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
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
from src.offline.tree_env import FringeEnv, OccupancyCounter, bfs_frontier_max


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
        self.padded_by_inst: Dict[int, bool] = {}
        envs: List[_Env] = []
        self.hfs_diags: Dict[str, HFSDiag] = {}
        included: List[int] = []
        excluded_subF: List[str] = []
        bfs_excluded: List[str] = []
        padded_names: List[str] = []
        for i in self.train_ids:
            inst = self.instances[i]
            elig = eligible_nodes(inst)
            ve = depth_dstar_var_explained(inst, elig)
            usable, total = bfs_usable_fraction(inst, elig, F)
            uf = (usable / total) if total else 0.0
            sub_F = len(elig) < F
            bfs_out = (total == 0) or (uf < self.bfs_exclude_usable_frac)
            # fmax = max simultaneous live frontier (policy-free). F>fmax => the
            # live pool can never reach F, so regimes are indistinguishable unless
            # we pad the beam from CLOSED nodes -> self-activating padding.
            fmax = bfs_frontier_max(inst)
            padded = (not sub_F) and (F > fmax)
            self.eligibility[i] = {
                "name": inst.name, "E": len(elig), "var_expl": round(ve, 4),
                "bfs_usable_frac": round(uf, 4), "sub_F": sub_F,
                "bfs_excluded": bfs_out, "fmax": int(fmax), "padded": padded,
            }
            if sub_F:
                excluded_subF.append(inst.name)
                continue
            included.append(i)
            self.fmax_by_inst[i] = int(fmax)
            self.padded_by_inst[i] = padded
            if padded:
                padded_names.append(inst.name)
            self.dfs_rank[i] = dfs_preorder_rank(inst)
            avail = [
                r for r in self.regimes
                if not (r == "bfs" and bfs_out)
            ]
            if bfs_out and "bfs" in self.regimes:
                bfs_excluded.append(inst.name)
            for r in avail:
                diag = None
                if r in ("hfs_m0", "hfs_m1"):
                    diag = self.hfs_diags.setdefault(r, HFSDiag())
                env = RedrawFringeEnv(
                    inst, fringe_size=F, seed=self.seed + 1000 * i + hash(r) % 997,
                    regime=r, dfs_rank=self.dfs_rank[i],
                    expansion_cap=(self.train_expansion_cap
                                   if self.train_expansion_cap is not None
                                   else 2 * inst.n_states),
                    hfs_diag=diag, pad_to_F=padded,
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
        self.padded_names = padded_names
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

    # ---- windowed instrumentation ----
    def _reset_regime_window(self) -> None:
        self.regime_stats = {r: RegimeStats() for r in self.regimes}
        for r in ("hfs_m0", "hfs_m1"):
            if r in self.hfs_diags:
                new = HFSDiag()
                self.hfs_diags[r] = new
                for e in self.regime_envs:
                    if e.regime == r:
                        e.env.hfs_diag = new
        for e in self.regime_envs:          # windowed pad counters
            e.env.reset_pad_counters()
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

    def _regime_instrumentation(self) -> Dict[str, object]:
        out = {"per_regime": {}, "hfs": {}, "eligibility": list(self.eligibility.values())}
        for r in self.regimes:
            if self.regime_stats[r].n_seen or self.regime_stats[r].n_dropped:
                out["per_regime"][r] = self.regime_stats[r].summary(self.fringe_size)
        for r, diag in self.hfs_diags.items():
            out["hfs"][r] = diag.summary()
        # per-instance padding (windowed): fmax, padded flag, charged pad
        # expansions + filled pad slots summed over that instance's regime envs.
        padcounts: Dict[int, Dict[str, int]] = {}
        for e in self.regime_envs:
            d = padcounts.setdefault(e.inst, {"pad_expansions": 0, "pad_slots_filled": 0})
            d["pad_expansions"] += e.env.n_pad_expansions
            d["pad_slots_filled"] += e.env.n_pad_slots_filled
        out["padding"] = [
            {
                "name": self.instances[i].name,
                "fmax": self.fmax_by_inst[i],
                "padded": self.padded_by_inst[i],
                "n_pad_expansions": padcounts.get(i, {}).get("pad_expansions", 0),
                "n_pad_slots_filled": padcounts.get(i, {}).get("pad_slots_filled", 0),
            }
            for i in self.included_train
        ]
        # P2 order-aux: usable-pair fraction + post-mask <2-distinct skip count,
        # over order-eligible (non-padded) slots. Zeroed when lambda_ord==0
        # (the order loss is never computed).
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
            # before env.step rebuilds the beam). Faithful envs => all-False.
            pad_mask = tuple(res.info.get("pad_flags", ()))
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
                    pad_mask=pad_mask if any(pad_mask) else None,
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
                if val_exp < best_val_exp:
                    best_val_exp = val_exp
                    self._save_model(out_dir / "best_by_expansions.pt", frame, ck)
                if rho is not None and rho > best_spearman:
                    best_spearman = rho
                    self._save_model(out_dir / "best_by_spearman.pt", frame, ck)
                with (out_dir / "history.json").open("w") as fh:
                    json.dump({"history": history, "checkpoints": checkpoints,
                               "config": self._regime_config()}, fh, indent=1)
                self._reset_regime_window()

        pbar.close()
        self._save_model(out_dir / "last.pt", frame, None)
        with (out_dir / "history.json").open("w") as fh:
            json.dump({"history": history, "checkpoints": checkpoints,
                       "config": self._regime_config()}, fh, indent=1)
        return {"history": history, "checkpoints": checkpoints}

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
            "padded_instances": self.padded_names,
            "fmax_by_instance": {
                self.instances[i].name: self.fmax_by_inst[i]
                for i in self.included_train
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
            pbar.write("  HFS tail (m0 vs m1): "
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
        pad_rows = [p for p in instr.get("padding", [])]
        if any(p["padded"] for p in pad_rows):
            pbar.write("  padding (instance fmax / padded / pad_exp / pad_slots):")
            for p in pad_rows:
                flag = "PAD" if p["padded"] else "faithful"
                pbar.write(f"    {p['name']:18s} fmax={p['fmax']:>4} {flag:>8} "
                           f"pad_exp={p['n_pad_expansions']:>5} "
                           f"pad_slots={p['n_pad_slots_filled']:>6}")
