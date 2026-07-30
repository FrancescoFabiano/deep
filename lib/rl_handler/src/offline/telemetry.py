"""Per-checkpoint telemetry -> telemetry.jsonl. Figures are generated FROM the
JSONL, never from the training loop, so they can be regenerated without retraining.

REPRESENTATION-AGNOSTIC: nothing here branches on the dataset type.

ALL sterile-aware scoring routes through scoring.py. No new sentinel — that bug has
already appeared three times in this project.

WHAT IS AND IS NOT A GATE
Gates are `coverage` and `regret`: the true objective, and it is measurable offline
because the tree is fixed. `viability_auc` and `spearman_logits_vs_delta` are
DIAGNOSTICS — they explain a result, they do not license one. A topology-only model
measured viability_auc 0.568 (picking sterile at the base rate) while landing 26
expansions from the clairvoyant ceiling and beating every baseline: gating on the
proxy would have killed the best model produced. Never gate on a proxy when the
true objective is measurable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .env import (
    FringeEnv,
    aggregate_rollouts,
    coverage_curve,
    reference_budget,
    rollout,
)
from .metrics import (
    beam_metrics,
    mean_ignoring_none,
    pearson,
    qstar_naive_residual,
)
from .tree import TreeInstance

# Budgets for the F1' cactus plot. Log-spaced: the reader picks the budget.
CACTUS_BUDGETS = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)


@dataclass
class CheckpointRecord:
    step: int
    frames: int
    split: str                      # train | val
    payload: Dict[str, object] = None

    def to_json(self) -> Dict[str, object]:
        d = {"step": self.step, "frames": self.frames, "split": self.split}
        d.update(self.payload or {})
        return d


def evaluate_split(
    instances: Sequence[TreeInstance],
    policy_for: Callable[[str], Callable],
    fringe_size: int,
    seeds: int = 5,
    expansion_cap: int = 2000,
    score_for: Optional[Callable] = None,
    gamma: float = 1.0,
) -> Dict[str, object]:
    """Roll the greedy policy in the REAL env and collect everything.

    `policy_for(instance_name) -> ranking policy`.
    `score_for(instance_name, beam) -> logits`, optional; when given, the ranking
    diagnostics (viability_auc, top1, spearman) and the F4 residual are collected
    along the trajectory.
    `gamma`: the run's discount factor, threaded from RunConfig so the eval env
    matches the trained objective (it affects only the unreachable doom penalty
    on gated data; the reported `return` stays the undiscounted reward sum).
    """
    rs: List[Dict] = []
    auc, top1, sp = [], [], []
    resid, resid_R = [], []
    per_instance: Dict[str, Dict] = {}

    for inst in instances:
        pol = policy_for(inst.name)
        inst_rs = []
        for s in range(seeds):
            env = FringeEnv(inst, fringe_size=fringe_size, seed=s, gamma=gamma,
                            expansion_cap=expansion_cap)
            res = env.reset(seed=s)
            ret = 0.0                        # accumulated reward = the RETURN (gamma=1)
            while not res.done:
                if env.forced:
                    res = env.step(env.forced_action)
                    ret += float(res.reward)
                    continue
                if score_for is not None:
                    lg = score_for(inst.name, env.fringe)
                    m = beam_metrics(inst, env.fringe, lg)
                    auc.append(m["viability_auc"])
                    top1.append(m["top1_oracle_agreement"])
                    sp.append(m["spearman_logits_vs_delta"])
                    # F4 panel B: how far below the naive bound a NON-argmin action
                    # sits, against |R|. The critic that learns this has learned the
                    # cost of being wrong -- what a delta-regressor cannot represent.
                    for a in range(len(env.fringe)):
                        r = qstar_naive_residual(lg[a], inst, env.fringe, env.reservoir, a)
                        if r is not None:
                            resid.append(r)
                            resid_R.append(float(len(env.reservoir)))
                    rk = sorted(range(len(env.fringe)), key=lambda k: -lg[k])
                else:
                    rk = list(pol(env.fringe))
                res = env.step(rk[0], rk)
                ret += float(res.reward)
            inst_rs.append(_row_from(env, res, inst, ret))
        rs.extend(inst_rs)
        per_instance[inst.name] = aggregate_rollouts(
            inst_rs, reference_budget=reference_budget(inst))

    # coverage at the DECLARED reference budget, per instance (difficulty-relative),
    # then averaged -- an absolute budget would give an easy instance the same slack
    # as a hard one.
    cov = (sum(per_instance[i.name]["coverage"] for i in instances) / max(1, len(instances)))
    agg = aggregate_rollouts(rs)
    out = {
        "coverage_at_reference_budget": cov,
        "coverage_curve": coverage_curve(rs, CACTUS_BUDGETS),
        "regret_mean_lower_bound": agg["regret_mean"],
        "regret_median_lower_bound": _pct(rs, 0.50),
        "regret_p90_lower_bound": _pct(rs, 0.90),
        "return_mean": agg["return_mean"],       # the objective, over ALL rollouts
        "expansions_mean": agg["expansions_mean"],
        "success_rate": agg["coverage"],
        "doom_rate": agg["doom_rate"],
        "timeout_rate": agg["timeout_rate"],
        "n_rollouts": len(rs),
        "n_instances": len(instances),
        # F10 -- node-level failure
        "expansions_sterile_frac": _mean(rs, "expansions_sterile_frac"),
        "beam_sterile_frac": _mean(rs, "beam_sterile_frac"),
        # the compounding loop
        "eviction_events": _mean(rs, "eviction_events"),
        "eviction_recovery_steps": mean_ignoring_none(
            [r["eviction_recovery_steps_mean"] for r in rs]),
        "eviction_never_recovered": _mean(rs, "eviction_never_recovered"),
        "reservoir_size_mean": _mean(rs, "reservoir_size_mean"),
        "reservoir_size_max": max((r["reservoir_size_max"] for r in rs), default=0),
        "beam_size_max": max((r["beam_size_max"] for r in rs), default=0),
        "per_instance": per_instance,
    }
    if score_for is not None:
        out.update({
            "viability_auc": mean_ignoring_none(auc),
            "top1_oracle_agreement": mean_ignoring_none(top1),
            "spearman_logits_vs_delta": mean_ignoring_none(sp),
            "qstar_naive_residual_mean": (sum(resid) / len(resid)) if resid else None,
            "qstar_naive_residual_corr_reservoir": (
                pearson(resid, resid_R) if len(resid) > 1 else None),
            "n_residual_samples": len(resid),
        })
    return out


def _row_from(env: FringeEnv, res, inst: TreeInstance, ret: float = 0.0) -> Dict:
    solved = res.info["outcome"] == "success"
    return {
        "instance": inst.name,
        "expansions": res.info["expansions"],
        "return": ret,                       # accumulated reward = the objective itself
        "solved": solved,
        "outcome": res.info["outcome"],
        "truncated": res.truncated,
        # LOWER BOUND: V* assumes an evicted node returns for free; it does not.
        "regret": inst.regret(int(res.info["expansions"])) if solved else None,
        "expansions_sterile_frac": env.n_sterile_expansions / max(1, env.expansions),
        "beam_sterile_frac": env.beam_sterile_frac(),
        "eviction_events": len(env.evictions),
        "eviction_recovery_steps_mean": (
            sum(e["recovery_steps"] for e in env.evictions if e["recovered"])
            / max(1, sum(1 for e in env.evictions if e["recovered"]))
            if any(e["recovered"] for e in env.evictions) else None),
        "eviction_never_recovered": sum(1 for e in env.evictions if not e["recovered"]),
        "reservoir_size_mean": (sum(env.reservoir_sizes) / len(env.reservoir_sizes))
        if env.reservoir_sizes else 0.0,
        "reservoir_size_max": max(env.reservoir_sizes) if env.reservoir_sizes else 0,
        "beam_size_max": max(env.beam_sizes) if env.beam_sizes else 0,
    }


def _mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return (sum(v) / len(v)) if v else None


def _pct(rs, q):
    v = sorted(r["regret"] for r in rs if r["regret"] is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))]


class TelemetryWriter:
    """One run's JSONL. TRUNCATES on construction: a run's telemetry.jsonl must
    contain ONLY that run's rows.

    It used to open in append mode, so a re-run into an existing fringe dir appended
    its rows to the STALE previous run's file (measured: an F=4 re-run left 44 old +
    24 new rows). Figures and analysis then mixed two runs -- and since selection reads
    the newest rows while a plot reads all rows, the old rows (a different eval_mode,
    no heldout_top1) silently corrupted the figures. Truncate once here; append per row.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")          # truncate: this run owns the file

    def append(self, rec: CheckpointRecord) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec.to_json(), default=_safe) + "\n")

    def read(self) -> List[Dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


def _safe(o):
    try:
        return float(o)
    except Exception:
        return str(o)
