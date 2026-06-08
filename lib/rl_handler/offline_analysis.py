"""Post-training analysis for the offline-RL runs (Phases 4-5).

Produces, under <runs-root>/plots/:
  - per-seed training curves
  - IQM ± IQR-std bands across seeds for val #expansions and val Spearman
  - score vs -d* scatter at the overall-best checkpoint
and prints the comparison vs the supervised gnn_handler_plus champion
(distance_estimator.onnx) on the SAME val instance + protocol:
  - val Spearman(score, -d*)  (champion score = -predicted distance)
  - val greedy #expansions in the same FringeEnv (same seeds)

Run from lib/rl_handler:
  ../../.venv/bin/python offline_analysis.py --runs-root ../../exp/rl_exp/offline_rl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.offline.encoder import InstanceCache, pack_fringe_batch  # noqa: E402
from src.offline.plots import (  # noqa: E402
    plot_iqm_bands_across_seeds,
    plot_score_vs_dstar,
    plot_seed_curves,
)
from src.offline.tree_env import (  # noqa: E402
    UNREACHABLE_DISTANCE,
    FringeEnv,
    bfs_expansions,
    load_tree_instance,
    rollout,
)

REPO = Path(__file__).resolve().parents[2]
VAL_CSV = "out/NN/Training/CC_3_2_3__pl_7/CC_3_2_3__pl_7_depth_25.csv"
CHAMPION_ONNX = "exp/gnn_exp/batch0/_models_plus/CC/distance_estimator.onnx"
N_SCORE_SAMPLE = 4096
ROLLOUT_SEEDS = [11, 22, 33, 44, 55]


def champion_scorer(onnx_path: Path, cache: InstanceCache):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def score(state_ids: list[int], chunk: int = 512) -> np.ndarray:
        out = []
        for c0 in range(0, len(state_ids), chunk):
            ids = state_ids[c0 : c0 + chunk]
            packed = pack_fringe_batch([(cache, [s]) for s in ids])
            feeds = {
                "state_node_ids": packed["node_features"].numpy(),
                "state_edge_index": packed["edge_index"].numpy(),
                "state_edge_attr": packed["edge_attr"].view(-1, 1).numpy(),
                "state_batch": packed["membership"].numpy(),
            }
            (dist,) = sess.run(["distance"], feeds)
            out.append(-np.asarray(dist).reshape(-1))  # score = -predicted d
        return np.concatenate(out)

    return score


def dqn_scorer(ckpt: Path, cache: InstanceCache, device: str = "cpu"):
    from src.trainer import RLFrontierTrainer

    model = RLFrontierTrainer.load_model(ckpt, device=device).to(device).eval()

    def score(state_ids: list[int], chunk: int = 1024) -> np.ndarray:
        out = []
        with torch.no_grad():
            for c0 in range(0, len(state_ids), chunk):
                ids = state_ids[c0 : c0 + chunk]
                packed = pack_fringe_batch([(cache, [s]) for s in ids])
                logits = model(
                    node_features=packed["node_features"].to(device),
                    edge_index=packed["edge_index"].to(device),
                    edge_attr=packed["edge_attr"].to(device),
                    membership=packed["membership"].to(device),
                    candidate_batch=packed["candidate_batch"].to(device),
                )
                out.append(logits.cpu().numpy())
        return np.concatenate(out)

    return score


def fringe_scorer_policy(score_fn, cache: InstanceCache):
    """Greedy fringe policy from a per-state scorer (scores whole fringe)."""

    def policy(fringe: list[int]) -> int:
        return int(np.argmax(score_fn(fringe)))

    return policy


def dqn_fringe_policy(ckpt: Path, cache: InstanceCache, device: str = "cpu"):
    """True deployed semantics: score the fringe as ONE packed batch."""
    from src.trainer import RLFrontierTrainer

    model = RLFrontierTrainer.load_model(ckpt, device=device).to(device).eval()

    def policy(fringe: list[int]) -> int:
        packed = pack_fringe_batch([(cache, fringe)])
        with torch.no_grad():
            logits = model(
                node_features=packed["node_features"].to(device),
                edge_index=packed["edge_index"].to(device),
                edge_attr=packed["edge_attr"].to(device),
                membership=packed["membership"].to(device),
                candidate_batch=packed["candidate_batch"].to(device),
            )
        return int(torch.argmax(logits).item())

    return policy


def eval_scores(score_fn, inst, sample_ids):
    s = score_fn(sample_ids)
    d = np.array([inst.distance[i] for i in sample_ids])
    unreach = d >= UNREACHABLE_DISTANCE
    d_max = d[~unreach].max() if (~unreach).any() else 0.0
    d_cl = np.where(unreach, d_max + 1.0, d)
    rho_all = float(spearmanr(s, -d_cl).statistic)
    rho_reach = float(spearmanr(s[~unreach], -d[~unreach]).statistic)
    return s, d, unreach, rho_all, rho_reach


def eval_rollouts(policy, inst, fringe_size=32, cap=2000):
    outs = []
    for sd in ROLLOUT_SEEDS:
        env = FringeEnv(inst, fringe_size=fringe_size, seed=sd, expansion_cap=cap)
        outs.append(rollout(env, policy, seed=sd))
    exps = [int(o["expansions"]) for o in outs]
    return {
        "expansions_per_seed": exps,
        "mean": float(np.mean(exps)),
        "all_goal_found": all(o["goal_found"] for o in outs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    runs_root = Path(args.runs_root).resolve()
    plots_dir = runs_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- val instance + cache ---
    csv_path = REPO / VAL_CSV
    inst = load_tree_instance(csv_path)
    cache = InstanceCache.from_paths(
        inst.state_paths_abs(REPO), csv_path.parent / "graph_cache_offline_v1.pt"
    )
    bfs = bfs_expansions(inst)
    opt = inst.optimal_expansions()
    print(f"[refs] val={inst.name} optimal={opt} bfs={bfs['expansions']}")

    # --- per-seed curves + IQM bands ---
    histories = {}
    for s in args.seeds:
        hf = runs_root / f"seed{s}" / "history.json"
        if hf.exists():
            histories[s] = hf
            plot_seed_curves(hf, plots_dir, seed=s)
    if histories:
        plot_iqm_bands_across_seeds(
            histories, plots_dir,
            bfs_reference=float(bfs["expansions"]),
            optimal_reference=float(opt) if opt else None,
        )
        print(f"[plots] saved to {plots_dir}")

    # --- pick overall best checkpoint (val expansions) across seeds ---
    best = None
    for s, hf in histories.items():
        cks = json.loads(hf.read_text())["checkpoints"]
        for ck in cks:
            v = ck["summary"]["val_total_expansions"]
            if best is None or v < best[2]:
                best = (s, ck["frame"], v)
    print(f"[best] seed={best[0]} frame={best[1]} val_expansions={best[2]}")

    g = torch.Generator().manual_seed(49)
    n = inst.n_states
    sample = (
        torch.randperm(n, generator=g)[:N_SCORE_SAMPLE].tolist()
        if n > N_SCORE_SAMPLE
        else list(range(n))
    )

    results = {"references": {"optimal": opt, "bfs": int(bfs["expansions"])},
               "best_checkpoint": {"seed": best[0], "frame": best[1]}}

    # --- DQN best ---
    best_ckpt = runs_root / f"seed{best[0]}" / "best_by_expansions.pt"
    dqn_score = dqn_scorer(best_ckpt, cache, device=args.device)
    s_dqn, d, unreach, rho_all, rho_reach = eval_scores(dqn_score, inst, sample)
    dqn_roll = eval_rollouts(dqn_fringe_policy(best_ckpt, cache, args.device), inst)
    results["dqn_best"] = {
        "spearman_all": rho_all, "spearman_reachable": rho_reach, **dqn_roll,
    }
    print(f"[dqn ] spearman_all={rho_all:.4f} reach={rho_reach:.4f} "
          f"rollouts={dqn_roll['expansions_per_seed']}")
    plot_score_vs_dstar(
        s_dqn.tolist(), d.tolist(), unreach.tolist(),
        plots_dir / "score_vs_dstar_best_dqn.png",
        f"DQN best (seed {best[0]}, frame {best[1]}) | val {inst.name}",
    )

    # --- champion ---
    champ_path = REPO / CHAMPION_ONNX
    if champ_path.exists():
        ch_score = champion_scorer(champ_path, cache)
        s_ch, d, unreach, ch_all, ch_reach = eval_scores(ch_score, inst, sample)
        ch_roll = eval_rollouts(fringe_scorer_policy(ch_score, cache), inst)
        results["champion"] = {
            "spearman_all": ch_all, "spearman_reachable": ch_reach, **ch_roll,
        }
        print(f"[champ] spearman_all={ch_all:.4f} reach={ch_reach:.4f} "
              f"rollouts={ch_roll['expansions_per_seed']}")
        plot_score_vs_dstar(
            s_ch.tolist(), d.tolist(), unreach.tolist(),
            plots_dir / "score_vs_dstar_champion.png",
            f"gnn_handler_plus champion | val {inst.name}",
        )
    else:
        print(f"[champ] missing: {champ_path}")

    with (runs_root / "analysis_summary.json").open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] {runs_root / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
