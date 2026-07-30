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
from .batching import default_device, pack_batch, pack_single
from .encoder import InstanceCache, pack_fringe
from .env import FringeEnv, rollout
from .metrics import beam_metrics, mean_ignoring_none
from .tree import INF_DELTA, TreeInstance


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pack(cache: InstanceCache, beam: Sequence[int], F: int, device) -> Dict[str, torch.Tensor]:
    p = pack_fringe(cache, list(beam), F)
    return {k: v.to(device) for k, v in p.items()}


def pack_beams_batch(
    caches: Dict[str, InstanceCache],
    picks: Sequence[tuple],           # (instance_name, beam)
    device,
) -> Dict[str, torch.Tensor]:
    """Pack many beams into ONE graph: membership continues across beams and
    `candidate_batch` says which beam each pooled slot belongs to.

    One forward for the whole minibatch instead of one per beam. The batched and
    single-fringe paths are different code paths in `_contextualize` (and in
    FringeAttention: forward vs forward_single) -- they are asserted equal to
    1e-5 by test_batched_path_matches_single_path, which is what makes this
    substitution safe.
    """
    nf, ei, ea, mem, cb = [], [], [], [], []
    node_off, slot_off = 0, 0
    for i, (name, beam) in enumerate(picks):
        p = pack_fringe(caches[name], list(beam), len(beam))
        nf.append(p["node_features"])
        ei.append(p["edge_index"] + node_off)
        ea.append(p["edge_attr"])
        mem.append(p["membership"] + slot_off)
        cb.append(torch.full((len(beam),), i, dtype=torch.int64))
        node_off += int(p["node_features"].numel())
        slot_off += len(beam)
    return {
        "node_features": torch.cat(nf).to(device),
        "edge_index": torch.cat(ei, dim=1).to(device),
        "edge_attr": torch.cat(ea).to(device),
        "membership": torch.cat(mem).to(device),
        "candidate_batch": torch.cat(cb).to(device),
    }


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
    batch_beams: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 0,
    hidden_dim: int = 64,
    context_mode: str = "mean_pool",
    verbose: bool = True,
) -> tuple[TwoHeadBaselineNetwork, List[Dict[str, float]]]:
    device = device or default_device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    by_name = {i.name: i for i in instances}
    # max_delta is DATA-DERIVED: the largest finite delta in the training
    # instances. It bounds the distance head so viability strictly dominates
    # distance in the score (see TwoHeadBaselineNetwork.forward). Not tuned.
    max_delta = max(
        (d for i in instances for d in i.delta if d != INF_DELTA), default=64.0
    )
    model = TwoHeadBaselineNetwork(
        node_input_dim=1, hidden_dim=hidden_dim, gnn_layers=2,
        dataset_type="HASHED", context_mode=context_mode,
        max_delta=float(max_delta),
    ).to(device)
    if verbose:
        print(f"  [baseline] max_delta (data-derived) = {max_delta:.0f}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist: List[Dict[str, float]] = []

    for step in range(int(steps)):
        picks = [beams[rng.randrange(len(beams))] for _ in range(batch_beams)]
        # ONE forward for the whole minibatch (batched == single to 1e-5).
        p = pack_beams_batch(caches, picks, device)
        deltas = [by_name[n].delta[v] for n, beam in picks for v in beam]
        viable = torch.tensor(
            [0.0 if d == INF_DELTA else 1.0 for d in deltas],
            dtype=torch.float32, device=device,
        )
        dt = torch.tensor(
            [0.0 if d == INF_DELTA else float(d) for d in deltas],
            dtype=torch.float32, device=device,
        )
        # Censored slots (fix 1A) carry no viability label -- their delta=inf
        # is a generation artifact, so the BCE must not read them as sterile.
        label_mask = torch.tensor(
            [not by_name[n].censored[v] for n, beam in picks for v in beam],
            dtype=torch.bool, device=device,
        )
        opt.zero_grad()
        p_logit, d_hat = model.heads(
            p["node_features"], p["edge_index"], p["edge_attr"], p["membership"],
            candidate_batch=p["candidate_batch"], mask=None,
        )
        loss, log = two_head_loss(p_logit, d_hat, viable, dt, label_mask=label_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % max(1, steps // 15) == 0 or step == steps - 1:
            rec = {"step": step, **{k: log[k] for k in ("loss", "bce", "mse")}}
            hist.append(rec)
            if verbose:
                print(f"  [baseline] step {step:5d}  loss={rec['loss']:.4f} "
                      f"bce={rec['bce']:.4f} mse={rec['mse']:.3f}", flush=True)
    return model, hist


def evaluate_in_env(
    model,
    instances: Sequence[TreeInstance],
    caches: Dict[str, InstanceCache],
    fringe_size: int,
    seeds: int = 3,
    expansion_cap: int = 2000,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Roll the greedy policy in the REAL env -- reservoir, random refill and all.

    Also collects the ranking diagnostics per visited state so the baseline is
    reported on the same axes as the RL policy.
    """
    device = device or default_device()
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
