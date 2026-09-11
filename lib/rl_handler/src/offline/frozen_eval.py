"""The FROZEN evaluation set: M random-policy fringes per instance, drawn once
before the first checkpoint and never changed.

WHY (2026-09-09)
----------------
With the unified graph every real trajectory trains (nothing is held out), so
the ranking metrics need a fringe set that is (a) not a training trajectory's
neighbour by construction and (b) IDENTICAL at every checkpoint and across runs
that share the seed -- apples to apples. Held-out trajectories gave neither once
they all train; a random-policy rollout on the unified graph gives both: it visits
the graph without favouring any of the N behaviour policies that built it, and
the seed pins it.

WHAT A FROZEN FRINGE IS
-----------------------
A real beam of the offline env (`FringeEnv`, the run's F, random refill) at a
RESCORE state of a `random`-behaviour rollout, kept only if the ranking metric can
read it: >= 2 rankable (non-censored) slots, not all dead, and >= 2 DISTINCT
deltas (a beam whose nodes are all equally good has no ranking to get right).
Consecutive beams of one rollout differ by a single expansion, so candidates are
pooled over several rollouts and M are drawn uniformly from the unique pool;
fewer than M candidates => all of them, and the shortfall is in the manifest.

The objects expose `.instance / .obs / .forced` -- the row protocol that
`selection.heldout_ranking_metrics` and friends already consume -- so the whole
ranking-metric family (top1, NDCG, JS divergence, regret at decision, picked_dead,
tau-b) and the per-policy agreement run on them unchanged.

TRAINING OVERLAP IS REPORTED, NOT HIDDEN
----------------------------------------
A frozen fringe may coincide with a beam some training row saw (same problem,
same graph). `training_overlap` measures exactly how often, at the fringe level
(exact beam match) and at the state level, and the numbers go to the sidecar.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .env import FringeEnv
from .policies import make_policy
from .selection import rankable_slots
from .tree import INF_DELTA, TreeInstance

FROZEN_EVAL_VERSION = 1
# Rollouts pooled per instance before drawing M fringes.
DEFAULT_ROLLOUTS_PER_INSTANCE = 128
DEFAULT_M_PER_INSTANCE = 128
# Offset added to the run seed when no explicit frozen-eval seed is given, so the
# eval rollouts never coincide with a behaviour rollout seeded 0..k.
FROZEN_SEED_OFFSET = 7919


@dataclass
class FrozenFringe:
    instance: str
    obs: List[int]
    seed: int                      # the random rollout it was taken from
    t: int                         # decision index inside that rollout
    forced: bool = False           # always False: forced states have no ranking

    def key(self) -> Tuple[str, Tuple[int, ...]]:
        return (self.instance, tuple(self.obs))


def is_contrastive(inst: TreeInstance, obs: Sequence[int]) -> bool:
    keep = rankable_slots(inst, obs)
    if len(keep) < 2:
        return False
    deltas = [inst.delta[obs[k]] for k in keep]
    if all(d >= INF_DELTA for d in deltas):
        return False
    return len(set(deltas)) >= 2


def collect_random_fringes(
    inst: TreeInstance,
    fringe_size: int,
    seeds: Iterable[int],
    expansion_cap: int,
    gamma: float = 1.0,
) -> Tuple[List[FrozenFringe], Dict[str, object]]:
    """Every UNIQUE contrastive rescore beam met by `random` rollouts at `seeds`."""
    out: List[FrozenFringe] = []
    seen: set = set()
    outcomes: Counter = Counter()
    n_decisions = 0
    for s in seeds:
        env = FringeEnv(inst, fringe_size=fringe_size, seed=s, gamma=gamma,
                        expansion_cap=expansion_cap)
        pol = make_policy(inst, "random", seed=s)
        res = env.reset(seed=s)
        t = 0
        while not res.done:
            if env.forced:
                res = env.step(env.forced_action)
            else:
                n_decisions += 1
                obs = list(env.fringe)
                k = (inst.name, tuple(obs))
                if k not in seen and is_contrastive(inst, obs):
                    seen.add(k)
                    out.append(FrozenFringe(inst.name, obs, int(s), t))
                ranking = list(pol(env.fringe))
                res = env.step(ranking[0], ranking)
            t += 1
        outcomes[str(res.info["outcome"])] += 1
    return out, {"n_decisions": n_decisions, "outcomes": dict(outcomes)}


def build_frozen_fringes(
    instances: Sequence[TreeInstance],
    fringe_size: int,
    seed: int,
    expansion_cap: int,
    m_per_instance: int = DEFAULT_M_PER_INSTANCE,
    rollouts_per_instance: int = DEFAULT_ROLLOUTS_PER_INSTANCE,
    gamma: float = 1.0,
    verbose: bool = True,
) -> Tuple[List[FrozenFringe], Dict[str, object]]:
    """M frozen fringes per instance + the manifest that makes them auditable."""
    fringes: List[FrozenFringe] = []
    per_instance: Dict[str, Dict[str, object]] = {}
    # Seeds are a function of the frozen seed and the instance's position in the
    # NAME-SORTED pool -- never of hash(), whose per-process randomisation would
    # make two runs with the same seed draw different fringes.
    for idx, inst in enumerate(sorted(instances, key=lambda i: i.name)):
        base = int(seed) * 100_003 + idx * 1_000
        seeds = [base + k for k in range(int(rollouts_per_instance))]
        cands, info = collect_random_fringes(inst, fringe_size, seeds, expansion_cap, gamma)
        rng = random.Random(f"{seed}:{inst.name}")
        if len(cands) > m_per_instance:
            chosen = rng.sample(cands, int(m_per_instance))
        else:
            chosen = list(cands)
        chosen.sort(key=lambda f: (f.seed, f.t))
        fringes.extend(chosen)
        t_hist: Counter = Counter()
        for f in chosen:
            t_hist[str(10 * (f.t // 10))] += 1
        sizes = [len(f.obs) for f in chosen]
        n_trees_of = getattr(inst, "n_trees_of", None)
        shared_frac = None
        if n_trees_of and chosen:
            states = [v for f in chosen for v in f.obs]
            shared_frac = sum(1 for v in states if n_trees_of[v] >= 2) / len(states)
        per_instance[inst.name] = {
            "n_rollouts": len(seeds),
            "seeds": seeds,
            "n_decisions": info["n_decisions"],
            "outcomes": info["outcomes"],
            "n_candidates": len(cands),
            "n_selected": len(chosen),
            "shortfall": max(0, int(m_per_instance) - len(cands)),
            "beam_size_mean": (sum(sizes) / len(sizes)) if sizes else None,
            "decision_index_hist": dict(sorted(t_hist.items(), key=lambda kv: int(kv[0]))),
            "depth_mean": (sum(inst.depth[v] for f in chosen for v in f.obs)
                           / max(1, sum(sizes))) if sizes else None,
            "shared_state_frac": shared_frac,
        }
        if verbose:
            print(f"[frozen] {inst.name}: {info['n_decisions']} decisions over "
                  f"{len(seeds)} random rollouts -> {len(cands)} unique contrastive "
                  f"fringes -> {len(chosen)} frozen"
                  + (f"  (SHORTFALL {per_instance[inst.name]['shortfall']})"
                     if per_instance[inst.name]["shortfall"] else ""))
    manifest = {
        "version": FROZEN_EVAL_VERSION,
        "kind": "frozen_random_fringes",
        "seed": int(seed),
        "fringe_size": int(fringe_size),
        "expansion_cap": int(expansion_cap),
        "m_per_instance": int(m_per_instance),
        "rollouts_per_instance": int(rollouts_per_instance),
        "n_fringes": len(fringes),
        "n_instances": len(instances),
        "instances_with_none": [n for n, p in per_instance.items() if p["n_selected"] == 0],
        "per_instance": per_instance,
    }
    return fringes, manifest


def training_overlap(fringes: Sequence[FrozenFringe], rows: Sequence) -> Dict[str, object]:
    """How much of the frozen set the training rows have seen: exact-beam matches
    (same instance, same slot order), set-matches (same instance, same beam as a
    set), and the share of frozen-fringe STATES that occur in any training beam."""
    exact = {(r.instance, tuple(r.obs)) for r in rows}
    as_set = {(r.instance, tuple(sorted(r.obs))) for r in rows}
    states: Dict[str, set] = {}
    for r in rows:
        states.setdefault(r.instance, set()).update(r.obs)
    n = max(1, len(fringes))
    n_exact = sum(1 for f in fringes if (f.instance, tuple(f.obs)) in exact)
    n_set = sum(1 for f in fringes if (f.instance, tuple(sorted(f.obs))) in as_set)
    tot = sum(len(f.obs) for f in fringes)
    n_states = sum(1 for f in fringes for v in f.obs if v in states.get(f.instance, ()))
    return {
        "n_fringes": len(fringes),
        "exact_beam_match_frac": n_exact / n,
        "beam_as_set_match_frac": n_set / n,
        "state_seen_in_training_frac": (n_states / tot) if tot else None,
    }


def save_frozen(path: str | Path, fringes: Sequence[FrozenFringe],
                manifest: Dict[str, object]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"manifest": manifest,
                             "fringes": [asdict(f) for f in fringes]}, indent=1))
    return p


def load_frozen(path: str | Path) -> Tuple[List[FrozenFringe], Dict[str, object]]:
    doc = json.loads(Path(path).read_text())
    fr = [FrozenFringe(instance=d["instance"], obs=list(d["obs"]), seed=int(d["seed"]),
                       t=int(d["t"]), forced=bool(d.get("forced", False)))
          for d in doc["fringes"]]
    return fr, doc["manifest"]
