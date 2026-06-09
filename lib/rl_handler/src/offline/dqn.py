"""Double-DQN trainer over the fringe MDP. See DESIGN.md §3-§5.

Q(fringe, k) = FrontierPolicyNetwork logits[k]; one GNN forward scores the
whole fringe (shared weights + frontier context), exactly the deployed
parameterization.  Targets: y = r + gamma*(1-done)*Q_target(s', argmax_online).
No large-magnitude constants anywhere: dead ends terminate with y = r = -1.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from scipy.stats import spearmanr
from torch import nn
from tqdm import tqdm

from src.models.frontier_policy import FrontierPolicyNetwork
from src.offline.encoder import GlobalFlatCache, InstanceCache, segment_argmax
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
        self.train_ids = list(train_ids)
        self.val_ids = list(val_ids)
        self.fringe_size = int(fringe_size)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.target_sync = int(target_sync)
        self.update_every = max(1, int(update_every))
        self.eval_expansion_cap = int(eval_expansion_cap)
        self.max_grad_norm = float(max_grad_norm)
        self.seed = int(seed)

        torch.manual_seed(seed)
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.flat = GlobalFlatCache(self.caches, device=self.device)
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
        return self.flat.pack(gids, lens)

    def _forward(
        self,
        net: nn.Module,
        fringes: Sequence[Tuple[int, Sequence[int]]],
        packed: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score fringes [(inst_idx, state_ids), ...] -> (flat logits, ptr)."""
        if packed is None:
            packed = self._pack(fringes)
        logits = net(
            node_features=packed["node_features"],
            edge_index=packed["edge_index"],
            edge_attr=packed["edge_attr"],
            membership=packed["membership"],
            candidate_batch=packed["candidate_batch"],
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

        targets = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
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

        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            "td_loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "target_mean": float(targets.mean().item()),
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
            "epsilon": [], "episode_return": [], "episode_frame": [],
            "episode_expansions": [], "episode_goal": [], "episode_inst": [],
        }
        checkpoints: List[Dict[str, object]] = []
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
                        {"history": history, "checkpoints": checkpoints}, fh, indent=1
                    )

        pbar.close()
        with (out_dir / "history.json").open("w") as fh:
            json.dump({"history": history, "checkpoints": checkpoints}, fh, indent=1)
        return {"history": history, "checkpoints": checkpoints}

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
        val_total = 0
        val_occ = OccupancyCounter(self.fringe_size)
        for vid in self.val_ids:
            inst = self.instances[vid]
            roll = self.greedy_rollout(vid, seed=10_000 + frame, occ_accum=val_occ)
            bfs = bfs_expansions(inst)
            per_instance[inst.name] = {
                "greedy": roll,
                "bfs_expansions": int(bfs["expansions"]),
                "optimal_expansions": inst.optimal_expansions(),
                "occupancy": roll["occupancy"],
            }
            val_total += int(roll["expansions"])

        # Spearman + score stats on val states (singleton fringes).
        rho_all = rho_reach = None
        score_stats: Dict[str, object] = {}
        if self.val_ids:
            vid = self.val_ids[0]
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
                "val_total_expansions": int(val_total),
                "val_spearman_all": rho_all,
                "val_spearman_reachable": rho_reach,
                "val_occupancy": val_occ.summary(),
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
