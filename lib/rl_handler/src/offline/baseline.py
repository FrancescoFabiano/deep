"""Train + evaluate the two-head delta baseline.

FAIRNESS CONDITIONS, all load-bearing:
  - SAME network (encoder, context mode, head width) as the RL model; only the
    loss differs.
  - SAME env at eval: same reservoir, same random refill, same F, same seeds. NOT
    a clean best-first -- otherwise the eviction tax vanishes from the baseline by
    construction, which is strawmanning in the other direction.
  - SAME states: trained on the beams the behaviour policies actually visit.
  - No clip, no sentinel, no tuned constant anywhere.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..models.two_head_baseline import TwoHeadBaselineNetwork, two_head_loss
from .encoder import InstanceCache, pack_fringe
from .env import FringeEnv, rollout
from .metrics import beam_metrics, mean_ignoring_none
from .tree import INF_DELTA, TreeInstance


def _pack(cache: InstanceCache, beam: Sequence[int], F: int, device) -> Dict[str, torch.Tensor]:
    p = pack_fringe(cache, list(beam), F)
    return {k: v.to(device) for k, v in p.items()}


def score_beam(
    model: TwoHeadBaselineNetwork,
    cache: InstanceCache,
    beam: Sequence[int],
    device,
) -> List[float]:
    """Score a beam exactly as the deployed net would: mask length == |B|."""
    p = _pack(cache, beam, len(beam), device)
    with torch.no_grad():
        s = model(
            node_features=p["node_features"], edge_index=p["edge_index"],
            edge_attr=p["edge_attr"], membership=p["membership"],
            candidate_batch=None, mask=p["mask"],
        )
    return s.detach().cpu().tolist()


def make_greedy_policy(model, cache, device):
    """Ranking policy from a scorer. argmax score == what the planner pops."""
    def _policy(beam: Sequence[int]) -> List[int]:
        s = score_beam(model, cache, beam, device)
        return sorted(range(len(beam)), key=lambda k: -s[k])
    return _policy


def train_two_head_baseline(
    instances: Sequence[TreeInstance],
    caches: Dict[str, InstanceCache],
    beams: Sequence[tuple],          # (instance_name, beam) -- on-distribution states
    fringe_size: int,
    steps: int = 2000,
    batch_beams: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    hidden_dim: int = 64,
    context_mode: str = "mean_pool",
    verbose: bool = True,
) -> tuple[TwoHeadBaselineNetwork, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    by_name = {i.name: i for i in instances}
    model = TwoHeadBaselineNetwork(
        node_input_dim=1, hidden_dim=hidden_dim, gnn_layers=2,
        dataset_type="HASHED", context_mode=context_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist: List[Dict[str, float]] = []

    for step in range(int(steps)):
        picks = [beams[rng.randrange(len(beams))] for _ in range(batch_beams)]
        opt.zero_grad()
        total = None
        logs = []
        for name, beam in picks:
            inst = by_name[name]
            cache = caches[name]
            p = _pack(cache, beam, len(beam), device)
            p_logit, d_hat = model.heads(
                p["node_features"], p["edge_index"], p["edge_attr"], p["membership"],
                candidate_batch=None, mask=p["mask"],
            )
            deltas = [inst.delta[v] for v in beam]
            viable = torch.tensor(
                [0.0 if d == INF_DELTA else 1.0 for d in deltas],
                dtype=torch.float32, device=device,
            )
            dt = torch.tensor(
                [0.0 if d == INF_DELTA else float(d) for d in deltas],
                dtype=torch.float32, device=device,
            )
            loss, log = two_head_loss(p_logit, d_hat, viable, dt)
            total = loss if total is None else total + loss
            logs.append(log)
        total = total / len(picks)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            rec = {
                "step": step,
                "loss": sum(l["loss"] for l in logs) / len(logs),
                "bce": sum(l["bce"] for l in logs) / len(logs),
                "mse": sum(l["mse"] for l in logs) / len(logs),
            }
            hist.append(rec)
            if verbose:
                print(f"  [baseline] step {step:5d}  loss={rec['loss']:.4f} "
                      f"bce={rec['bce']:.4f} mse={rec['mse']:.3f}")
    return model, hist


def evaluate_in_env(
    model,
    instances: Sequence[TreeInstance],
    caches: Dict[str, InstanceCache],
    fringe_size: int,
    seeds: int = 3,
    expansion_cap: int = 2000,
    device: str = "cpu",
) -> Dict[str, object]:
    """Roll the greedy policy in the REAL env -- reservoir, random refill and all.

    Also collects the ranking diagnostics per visited state so the baseline is
    reported on the same axes as the RL policy.
    """
    rows = []
    per_instance = {}
    auc, top1, sp = [], [], []
    for inst in instances:
        cache = caches[inst.name]
        pol = make_greedy_policy(model, cache, device)
        rs = []
        for s in range(seeds):
            env = FringeEnv(inst, fringe_size=fringe_size, seed=s, gamma=1.0,
                            expansion_cap=expansion_cap)
            # instrument the ranking metrics along the rollout
            res = env.reset(seed=s)
            while not res.done:
                if env.forced:
                    res = env.step(env.forced_action)
                    continue
                logits = score_beam(model, cache, env.fringe, device)
                m = beam_metrics(inst, env.fringe, logits)
                auc.append(m["viability_auc"])
                top1.append(m["top1_oracle_agreement"])
                sp.append(m["spearman_logits_vs_delta"])
                rk = sorted(range(len(env.fringe)), key=lambda k: -logits[k])
                res = env.step(rk[0], rk)
            rs.append({
                "expansions": res.info["expansions"],
                "solved": res.info["outcome"] == "success",
                "outcome": res.info["outcome"],
                "truncated": res.truncated,
                "regret": inst.regret(int(res.info["expansions"]))
                if res.info["outcome"] == "success" else None,
                "expansions_sterile_frac": env.n_sterile_expansions / max(1, env.expansions),
                "eviction_events": len(env.evictions),
                "eviction_never_recovered": sum(1 for e in env.evictions if not e["recovered"]),
                "reservoir_size_mean": (sum(env.reservoir_sizes) / len(env.reservoir_sizes))
                if env.reservoir_sizes else 0.0,
                "reservoir_size_max": max(env.reservoir_sizes) if env.reservoir_sizes else 0,
            })
        rows.extend(rs)
        per_instance[inst.name] = rs
    solved = [r for r in rows if r["solved"]]
    return {
        "n_rollouts": len(rows),
        "coverage": len(solved) / max(1, len(rows)),
        "regret_mean": (sum(r["regret"] for r in solved) / len(solved)) if solved else None,
        "expansions_mean": (sum(r["expansions"] for r in solved) / len(solved)) if solved else None,
        "expansions_sterile_frac": sum(r["expansions_sterile_frac"] for r in rows) / max(1, len(rows)),
        "eviction_events": sum(r["eviction_events"] for r in rows) / max(1, len(rows)),
        "eviction_never_recovered": sum(r["eviction_never_recovered"] for r in rows) / max(1, len(rows)),
        "reservoir_size_mean": sum(r["reservoir_size_mean"] for r in rows) / max(1, len(rows)),
        "reservoir_size_max": max((r["reservoir_size_max"] for r in rows), default=0),
        "viability_auc": mean_ignoring_none(auc),
        "top1_oracle_agreement": mean_ignoring_none(top1),
        "spearman_logits_vs_delta": mean_ignoring_none(sp),
        "per_instance": per_instance,
    }
