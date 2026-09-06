"""Regenerate diagnostic figures for one run, or every run in a batch -- CPU only.

Closes two ways figures go missing:
  1. A crash before the end-of-run() figure call (e.g. the export device leak) skips
     figures even though telemetry is complete on disk. This walks telemetry files that
     ALREADY exist and plots from them, so a crashed run still gets its figures.
  2. A run on stale code recorded telemetry without a metric (return_mean, heldout_ndcg,
     ...), so the figures would be blank. This DETECTS the missing metric (via the
     telemetry's metrics_schema stamp / field presence) and, if the run's checkpoints
     are on disk, recomputes it by re-evaluating each checkpoint -- SELF-VALIDATED
     against the recorded top1 (max mismatch must be ~0, else it aborts rather than
     writing wrong numbers). Manual /tmp backfill is no longer load-bearing.

Usage:
  regenerate_figures.py <batch_dir>                 # every seed*_fringe* under _models/*/
  regenerate_figures.py --telemetry <path> --fringe N
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lib" / "rl_handler"))

# The metrics the current figure set needs in each val row. Absence => backfill.
# NOTE (METRICS_SCHEMA=4, Fix 2/3): the macro/per-instance NDCG + effective_instance_count
# and the per-policy agreement (heldout_agree_top1/taub_*) fields are recorded LIVE by a
# schema-4 run and are NOT in this backfill list: agreement needs the behaviour-policy
# rankings at a fixed tie-break seed and is not cheaply reconstructable, so we do not
# fake it for pre-schema-4 runs. The agreement FIGURE (fig_policy_agreement) skips itself
# when those fields are absent, so old runs are not mis-drawn -- just left without it.
REQUIRED_VAL_FIELDS = ("heldout_top1", "heldout_ndcg", "heldout_js",
                       "return_mean", "train_ndcg")
PLOT = REPO / "scripts" / "rl_exp" / "plot_diagnostics.py"


def _fringe_of(fringe_dir: Path) -> int:
    m = re.search(r"fringe(\d+)", fringe_dir.name)
    return int(m.group(1)) if m else 0


def _seed_of(fringe_dir: Path) -> int:
    m = re.search(r"seed(\d+)", fringe_dir.name)
    return int(m.group(1)) if m else 42


def _missing_fields(tel_path: Path) -> set:
    rows = [json.loads(l) for l in tel_path.open() if l.strip()]
    val = [r for r in rows if r.get("split") == "val" and r.get("step", 0) > 0]
    if not val:
        return set(REQUIRED_VAL_FIELDS)
    last = val[-1]
    return {f for f in REQUIRED_VAL_FIELDS if last.get(f) is None}


def backfill(fringe_dir: Path) -> str:
    """Recompute missing per-checkpoint + baseline metrics from the saved checkpoints,
    self-validated against the recorded top1. Rewrites telemetry.jsonl in place.
    Returns a status string. Never writes if validation fails."""
    import torch
    from src.offline.tree import load_tree_instance
    from src.offline.encoder import InstanceCache
    from src.offline.batching import pack_single
    from src.offline.dataset import generate_dataset
    from src.offline.selection import split_trajectories, heldout_ranking_metrics
    from src.offline.telemetry import evaluate_split
    from src.offline.env import default_expansion_cap
    from src.offline.policies import make_policy
    from src.models.frontier_policy import FrontierPolicyNetwork

    torch.set_num_threads(6)
    dev = "cpu"
    F = _fringe_of(fringe_dir)
    seed = _seed_of(fringe_dir)
    domain_dir = fringe_dir.parent                      # _models/<domain>
    tel_path = fringe_dir / "telemetry.jsonl"
    ck_dir = fringe_dir / "checkpoints"
    sidecars = list(fringe_dir.glob("*.selection.json"))
    if not ck_dir.is_dir() or not any(ck_dir.glob("ckpt_*.pt")):
        return "SKIP (no checkpoints to backfill from)"
    kind = "separated"
    if sidecars:
        kind = json.loads(sidecars[0].read_text()).get("kind_of_data", "separated")

    root = domain_dir / "training_data"
    from src.offline.strategies import tables_under
    insts = [load_tree_instance(p, kind_of_data=kind)
             for p in tables_under(root, verbose=False)]
    by_name = {i.name: i for i in insts}
    caches = {i.name: InstanceCache.from_paths(
        i.state_paths_abs(REPO), cache_file=domain_dir / "cache" / f"{i.name}.pt",
        verbose=False) for i in insts}
    cap = default_expansion_cap(insts)
    # HELDOUT_TRAJ_FRAC=0.10, seeds_per_policy=3 are the run defaults; the self-validation
    # below catches any mismatch (top1 would not reproduce) before anything is written.
    rows, _ = generate_dataset(insts, F, seeds_per_policy=3, expansion_cap=cap, verbose=False)
    train_rows, held_rows, _ = split_trajectories(rows, frac=0.10, seed=seed)

    net = FrontierPolicyNetwork(node_input_dim=1, hidden_dim=64, gnn_layers=2,
                                dataset_type="HASHED", context_mode="self_attention",
                                use_goal_separate_input=(kind == "separated")).to(dev).eval()

    def score_for(name, beam):
        p = pack_single(caches[name], beam, len(beam), dev)
        with torch.no_grad():
            return net(node_features=p["node_features"], edge_index=p["edge_index"],
                       edge_attr=p["edge_attr"], membership=p["membership"],
                       candidate_batch=None, mask=p["mask"]).cpu().tolist()

    def policy_for(name):
        return lambda beam: sorted(range(len(beam)), key=lambda k: -score_for(name, beam)[k])

    tel = [json.loads(l) for l in tel_path.open() if l.strip()]
    val = {r["step"]: r for r in tel if r.get("split") == "val" and r.get("step", 0) > 0}
    mismatch = 0.0
    for step in sorted(val):
        ck = ck_dir / f"ckpt_{step}.pt"
        if not ck.exists():
            continue
        net.load_state_dict(torch.load(ck, map_location=dev))
        net.eval()
        out = evaluate_split(insts, policy_for, F, seeds=5, expansion_cap=cap, score_for=score_for)
        rm = heldout_ranking_metrics(held_rows, by_name, logits_for=score_for)
        rmtr = heldout_ranking_metrics(train_rows[:len(held_rows)], by_name, logits_for=score_for)
        if val[step].get("heldout_top1") is not None:
            mismatch = max(mismatch, abs(rm["top1"] - val[step]["heldout_top1"]))
        val[step]["return_mean"] = out["return_mean"]
        for k in ("top1", "ndcg", "js", "regret_at_decision", "picked_dead", "kendall_tau"):
            val[step][f"heldout_{k}"] = rm[k]
        val[step]["train_top1"] = rmtr["top1"]
        val[step]["train_ndcg"] = rmtr["ndcg"]
        val[step]["metrics_schema"] = 3
    if mismatch > 0.02:
        return f"ABORT (self-validation failed: top1 mismatch {mismatch:.4f} -- config guess wrong, NOT written)"
    for br in [r for r in tel if r.get("step") == -1]:
        b = br["split"].split(":")[1]
        o = evaluate_split(insts, lambda n, _b=b: make_policy(by_name[n], _b, seed=0),
                           F, seeds=5, expansion_cap=cap)
        br["return_mean"] = o["return_mean"]
        rm = heldout_ranking_metrics(
            held_rows, by_name,
            rank_for=lambda n, beam, _b=b: make_policy(by_name[n], _b, seed=0)(list(beam)))
        for k in ("top1", "ndcg", "regret_at_decision", "picked_dead", "kendall_tau"):
            br[f"heldout_{k}"] = rm[k]
    with tel_path.open("w") as fh:
        for r in tel:
            fh.write(json.dumps(r) + "\n")
    return f"BACKFILLED (self-validated, top1 reproduced to {mismatch:.4f})"


def process(fringe_dir: Path) -> None:
    tel = fringe_dir / "telemetry.jsonl"
    if not tel.exists():
        print(f"[regen] {fringe_dir.name}: no telemetry.jsonl -- skip")
        return
    missing = _missing_fields(tel)
    if missing:
        print(f"[regen] {fringe_dir.name}: telemetry missing {sorted(missing)} "
              f"(predates the metrics commit) -- backfilling from checkpoints")
        status = backfill(fringe_dir)
        print(f"[regen] {fringe_dir.name}: {status}")
        if status.startswith("ABORT") or status.startswith("SKIP"):
            print(f"[regen] {fringe_dir.name}: NOT plotting the missing-metric figures "
                  f"(would be blank/mislabeled)")
    # plot from whatever is on disk (best-effort; never fails the caller)
    r = subprocess.run([sys.executable, str(PLOT), str(tel),
                        "--fringe", str(_fringe_of(fringe_dir))],
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()
    print(f"[regen] {fringe_dir.name}: {'figures OK' if r.returncode == 0 else 'PLOT FAILED'}"
          + (f" ({tail[0]})" if tail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", type=Path, nargs="?",
                    help="a batch dir; walks _models/*/seed*_fringe*/")
    ap.add_argument("--telemetry", type=Path, help="a single telemetry.jsonl")
    ap.add_argument("--fringe", type=int, default=0)
    a = ap.parse_args()

    if a.telemetry:
        process(a.telemetry.parent)
        return
    if not a.batch_dir:
        ap.error("give a batch_dir or --telemetry")
    fringe_dirs = sorted((a.batch_dir / "_models").glob("*/seed*_fringe*"))
    fringe_dirs = [d for d in fringe_dirs if (d / "telemetry.jsonl").exists()]
    if not fringe_dirs:
        print(f"[regen] no runs with telemetry under {a.batch_dir}/_models/*/seed*_fringe*")
        return
    print(f"[regen] {len(fringe_dirs)} run(s) under {a.batch_dir}")
    for d in fringe_dirs:
        process(d)


if __name__ == "__main__":
    main()
