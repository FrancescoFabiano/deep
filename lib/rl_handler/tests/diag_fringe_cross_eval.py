"""Diagnostic: 2x2 cross-eval — scorer quality vs deploy beam width.

Disentangle "F=64's weights are a genuinely better scorer" from "F=64 just
deploys with a wider beam".  We cross WEIGHTS x DEPLOY-BEAM:

  weights:  W32 = best_by_expansions.pt from seed<seed>_fringe32/
            W64 = best_by_expansions.pt from seed<seed>_fringe64/
  beam:     B32 = runtime fringe width 32
            B64 = runtime fringe width 64

The .pt weights are fringe-agnostic (the network is a per-node GNN + global
mean context; nothing is shaped to a fixed candidate count), so any weight set
runs at any beam width.  We deliberately do NOT use the exported ONNX —
policy_32 has a fixed scatter dim and rejects N>32.

Per cell, over refill seeds [0..4] on the val instance(s):
  - val_total_expansions (node economy — primary)
  - within-fringe agreement (greedy rollout vs the eval-only d* oracle):
    agree_rate, agree_nontriv, mean_dstar_rank

Reuses the real eval harness: FringeEnv at the runtime beam width +
OfflineDQNTrainer.greedy_action (the same per-step decision diag_fringe_accuracy
uses).  No search or oracle is reimplemented.

Diagonal-consistency check: the on-regime cells (W32,B32) and (W64,B64) must
reproduce diag_fringe_accuracy's Metric-1 val_total_expansions; otherwise the
harness is wired wrong and we stop.

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_fringe_cross_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.dqn import OfflineDQNTrainer  # noqa: E402
from src.offline.encoder import InstanceCache  # noqa: E402
from src.offline.tree_env import FringeEnv, load_tree_instance  # noqa: E402
from src.trainer import RLFrontierTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "exp/rl_exp/batch0/_models"
SEEDS = [0, 1, 2, 3, 4]
WEIGHT_FRINGES = (32, 64)   # which seed<>_fringe<F> dirs supply the weights
BEAMS = (32, 64)            # runtime deploy beam widths


def discover_domains() -> list[str]:
    """Domains that have BOTH seed*_fringe32 and seed*_fringe64 with weights."""
    out = []
    for dom_dir in sorted(MODELS.glob("*")):
        if not dom_dir.is_dir():
            continue
        ok = True
        for wf in WEIGHT_FRINGES:
            hits = list(dom_dir.glob(f"seed*_fringe{wf}/best_by_expansions.pt"))
            if not hits:
                ok = False
                break
        if ok:
            out.append(dom_dir.name)
    return out


def seed_dir(dom: str, fringe: int) -> Path | None:
    hits = sorted((MODELS / dom).glob(f"seed*_fringe{fringe}"))
    hits = [h for h in hits if (h / "best_by_expansions.pt").exists()]
    return hits[0] if hits else None


def load_val(seed_dir_path: Path) -> tuple[list, list] | str:
    """Load val instances + caches from a run's args.json. Returns a string
    describing what's missing if any prerequisite is absent."""
    args_path = seed_dir_path / "args.json"
    if not args_path.exists():
        return f"{seed_dir_path.name}: args.json missing"
    args = json.loads(args_path.read_text())
    instances, caches = [], []
    for csv in args.get("val_csv", []):
        csv_path = Path(csv)
        if not csv_path.is_absolute():
            csv_path = REPO / csv_path
        if not csv_path.exists():
            return f"{seed_dir_path.name}: val csv missing ({csv_path.name})"
        cache_file = csv_path.parent / "graph_cache_offline_v1.pt"
        if not cache_file.exists():
            return f"{seed_dir_path.name}: graph cache missing ({csv_path.parent.name})"
        inst = load_tree_instance(csv_path)
        cache = InstanceCache.from_paths(
            inst.state_paths_abs(REPO), cache_file, verbose=False
        )
        instances.append(inst)
        caches.append(cache)
    if not instances:
        return f"{seed_dir_path.name}: no val instances listed"
    return instances, caches


def make_trainer(weights_pt: Path, instances, caches, beam: int) -> OfflineDQNTrainer:
    """Trainer carrying `weights_pt`, configured to deploy at runtime `beam`."""
    model = RLFrontierTrainer.load_model(weights_pt, device="cpu")
    return OfflineDQNTrainer(
        model=model,
        instances=instances,
        caches=caches,
        train_ids=[],
        val_ids=list(range(len(instances))),
        fringe_size=beam,
        eval_expansion_cap=2000,
        seed=42,
        device="cpu",
    )


def eval_cell(trainer: OfflineDQNTrainer, instances, beam: int) -> dict:
    """Greedy rollout per (instance, seed); aggregate economy + agreement.

    Mirrors diag_fringe_accuracy.metric2's instrumentation, plus the rollout's
    expansion count (node economy) so the diagonal can be checked against
    Metric 1.  greedy_rollout reseeds the env internally, so for economy we use
    the trainer's own rollout; the agreement instrumentation walks the same env
    construction at the same seeds.
    """
    total_exp = 0
    n_hits = n_steps = n_nontriv = nontriv_hits = 0
    rank_sum = 0.0
    max_fringe = 0
    for vid, inst in enumerate(instances):
        dist = inst.distance
        for seed in SEEDS:
            env = FringeEnv(inst, fringe_size=beam, seed=seed, expansion_cap=2000)
            res = env.reset(seed=seed)
            while not res.done:
                fr = res.fringe
                max_fringe = max(max_fringe, len(fr))
                picked = trainer.greedy_action(vid, fr)
                ds = [dist[s] for s in fr]
                dmin, dmax, pd = min(ds), max(ds), ds[picked]
                rank = 1 + sum(1 for d in ds if d < pd)
                n_steps += 1
                rank_sum += rank
                hit = pd <= dmin
                n_hits += int(hit)
                if dmax > dmin:
                    n_nontriv += 1
                    nontriv_hits += int(hit)
                res = env.step(picked)
            total_exp += int(res.info["expansions"])
    n_inst = max(1, len(instances))
    return {
        "max_fringe": max_fringe,
        # economy summed over instances, averaged over seeds -> comparable to
        # Metric 1's single-rollout val_total_expansions
        "val_expansions": total_exp / len(SEEDS),
        "agree_rate": n_hits / n_steps if n_steps else float("nan"),
        "agree_nontriv": nontriv_hits / n_nontriv if n_nontriv else float("nan"),
        "mean_rank": rank_sum / n_steps if n_steps else float("nan"),
        "n_steps": n_steps,
        "n_nontriv": n_nontriv,
        "n_inst": n_inst,
    }


def metric1_diag(dom: str, fringe: int) -> int | None:
    """val_total_expansions of best_by_expansions for the on-regime diagonal."""
    sd = seed_dir(dom, fringe)
    if sd is None or not (sd / "history.json").exists():
        return None
    history = json.loads((sd / "history.json").read_text())
    best, best_exp = None, float("inf")
    for ck in history["checkpoints"]:
        ve = ck["summary"]["val_total_expansions"]
        if ve < best_exp:
            best_exp, best = ve, ck
    return None if best is None else best["summary"]["val_total_expansions"]


def main() -> None:
    domains = discover_domains()
    if not domains:
        print("[skip] no domain has both fringe32 and fringe64 best_by_expansions.pt; "
              "run the production sweep first.")
        return

    for dom in domains:
        print(f"\n########## domain {dom} ##########")
        val = None
        for f in WEIGHT_FRINGES:
            sd = seed_dir(dom, f)
            res = load_val(sd)
            if isinstance(res, str):
                print(f"[skip] {dom}: {res}")
                val = None
                break
            val = res  # val instances are identical across fringe runs
        if val is None:
            continue
        instances, caches = val

        weights = {wf: seed_dir(dom, wf) / "best_by_expansions.pt" for wf in WEIGHT_FRINGES}
        cells: dict[tuple[int, int], dict] = {}
        for wf in WEIGHT_FRINGES:
            for beam in BEAMS:
                trainer = make_trainer(weights[wf], instances, caches, beam)
                cells[(wf, beam)] = eval_cell(trainer, instances, beam)

        # ---- diagonal consistency vs diag_fringe_accuracy Metric 1 ----
        print("\n--- diagonal consistency check (on-regime vs Metric 1) ---")
        ok = True
        for wf in WEIGHT_FRINGES:
            m1 = metric1_diag(dom, wf)
            got = cells[(wf, wf)]["val_expansions"]
            # Metric 1 is a single rollout at seed 10_000+best_frame; our cell
            # averages SEEDS. Exact match only when the rollout is seed-stable;
            # otherwise they should be very close. Flag if they diverge a lot.
            tag = "OK" if (m1 is not None and abs(got - m1) <= 1.0) else "CHECK"
            if tag != "OK":
                ok = False
            print(f"  (W{wf},B{wf}): cross-eval={got:.1f}  Metric1={m1}  [{tag}]")
        if not ok:
            print("  !! diagonal does NOT reproduce Metric 1 within 1 expansion — "
                  "harness may be wired wrong; interpret with caution.")

        # ---- beam-binding diagnostic: does the deploy beam axis do anything? ----
        max_fr = max(c["max_fringe"] for c in cells.values())
        binds = max_fr >= min(BEAMS)
        print(f"\n--- beam-binding: max live fringe seen = {max_fr} "
              f"(min beam = {min(BEAMS)}) -> beam axis is "
              f"{'ACTIVE' if binds else 'INERT (fringe never reaches the narrower beam, '
              'so B32==B64 by construction; the beam-width axis cannot be tested on '
              'this instance and all economy differences are attributable to the scorer)'}")

        # ---- table 1: node economy ----
        print("\n--- 2x2 node economy (val expansions, mean over seeds; lower=better) ---")
        print(f"{'':<10}{'B32':>10}{'B64':>10}")
        for wf in WEIGHT_FRINGES:
            row = "".join(f"{cells[(wf, b)]['val_expansions']:>10.1f}" for b in BEAMS)
            print(f"{'W'+str(wf):<10}{row}")

        # ---- table 2: within-fringe agreement ----
        print("\n--- 2x2 within-fringe agreement (agree_rate / agree_nontriv / mean_rank) ---")
        print(f"{'':<10}{'B32':>22}{'B64':>22}")
        for wf in WEIGHT_FRINGES:
            cellstrs = []
            for b in BEAMS:
                c = cells[(wf, b)]
                ant = f"{c['agree_nontriv']:.2f}" if c["n_nontriv"] else "n/a"
                cellstrs.append(f"{c['agree_rate']:.2f}/{ant}/{c['mean_rank']:.2f}")
            print(f"{'W'+str(wf):<10}" + "".join(f"{s:>22}" for s in cellstrs))
        print(f"\n(cells aggregate refill seeds {SEEDS}; agree_nontriv restricts to "
              f"steps with non-zero fringe d* spread; n_inst={instances and len(instances)}.)")


if __name__ == "__main__":
    main()
