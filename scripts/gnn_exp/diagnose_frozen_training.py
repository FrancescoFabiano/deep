#!/usr/bin/env python3
"""Diagnose why HASHED validation metrics are flat (R^2 = -0.3721 for 200 epochs).

Decisive checks, in order:
  1. Target stats: distance_estimator_C.txt + raw/scaled target distribution
     from samples.pt (prepare_samples applied exactly as __main__.py does).
  2. Prediction stats: load the trained checkpoint, run a few val batches,
     print prediction min/max/mean/std — plus the PRE-clamp sigmoid output
     (captured with a forward hook on model.regressor) so the clamp trap
     (out pinned at min_value=1e-3, zero gradient) is directly observable.
  3. If predictions vary: Spearman rank correlation preds-vs-targets on the
     full val set (the quantity A* consumes), reported alongside R^2.
  4. history_losses.json: are per-epoch val entries literally identical?

Usage:
  .venv/bin/python scripts/gnn_exp/diagnose_frozen_training.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib" / "gnn_handler"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from src.utils import (  # noqa: E402
    get_dataloaders,
    prepare_samples,
    seed_everything,
    select_model,
)

MODEL_DIR = REPO_ROOT / "exp/gnn_exp/batch0/_models/CC"
UNREACHABLE = 1000000
MIN_VALUE = 1e-3  # DistanceEstimator clamp floor


def stats(t: torch.Tensor) -> str:
    return (f"min={t.min().item():.6f}  max={t.max().item():.6f}  "
            f"mean={t.mean().item():.6f}  std={t.std().item():.6f}")


def step1_targets():
    print("=" * 70)
    print("[1] Target stats")
    print("=" * 70)
    c_txt = (MODEL_DIR / "distance_estimator_C.txt").read_text()
    print(f"distance_estimator_C.txt:\n{c_txt.strip()}")
    print("(C is slope/intercept from prepare_samples: MAX_DEPTH=50 hardcoded,"
          " so identical for old and new data — C did not change.)")

    data = torch.load(MODEL_DIR / "samples.pt", weights_only=False)
    train_s, test_s = data["train_samples"], data["test_samples"]
    print(f"\n#train={len(train_s)}  #test={len(test_s)}")

    raw_tr = torch.tensor([s["target"].item() for s in train_s])
    raw_te = torch.tensor([s["target"].item() for s in test_s])
    print(f"RAW train targets:  {stats(raw_tr)}")
    print(f"RAW test  targets:  {stats(raw_te)}")
    for name, raw in (("train", raw_tr), ("test", raw_te)):
        vals, counts = raw.unique(return_counts=True)
        head = "  ".join(f"{int(v)}:{int(c)}" for v, c in
                         list(zip(vals.tolist(), counts.tolist()))[:15])
        print(f"  {name} value counts (first 15): {head}")

    # Apply the exact runtime scaling __main__.py applies.
    train_c, test_c, params = prepare_samples(
        list(train_s), list(test_s), UNREACHABLE)
    sc_tr = torch.tensor([s["target"].item() for s in train_c])
    sc_te = torch.tensor([s["target"].item() for s in test_c])
    print(f"\nprepare_samples params: {params}")
    print(f"SCALED train targets: {stats(sc_tr)}")
    print(f"SCALED test  targets: {stats(sc_te)}")
    print(f"  share of scaled test targets == intercept (depth 0): "
          f"{(sc_te == params['intercept']).float().mean().item():.4f}")
    return train_c, test_c, params


def step2_predictions(train_c, test_c):
    print()
    print("=" * 70)
    print("[2] Prediction stats from the trained checkpoint")
    print("=" * 70)
    seed_everything(42)
    m = select_model("distance_estimator", use_goal=False, use_depth=False)
    m.load_model(str(MODEL_DIR / "distance_estimator.pt"))
    m.model.eval()

    # Capture the regressor (sigmoid) output BEFORE the final clamp.
    pre_clamp = []
    h = m.model.regressor.register_forward_hook(
        lambda mod, inp, out: pre_clamp.append(out.detach().cpu().view(-1)))

    _, val_loader = get_dataloaders(train_c, test_c, batch_size=1024)
    preds, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            p = m.predict_batch(batch).view(-1)
            preds.append(p)
            targets.append(batch["target"].view(-1).cpu())
            if i == 2:  # 3 batches for the per-batch report
                for j, (pp, raw) in enumerate(zip(
                        [t for t in preds], pre_clamp)):
                    print(f"  batch {j}: POST-clamp {stats(pp)}")
                    print(f"           PRE -clamp {stats(raw)}  "
                          f"(values < {MIN_VALUE:g}: "
                          f"{(raw < MIN_VALUE).float().mean().item()*100:.2f}%)")
    h.remove()
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    pre = torch.cat(pre_clamp)

    print(f"\nFULL val set ({len(preds)} samples):")
    print(f"  POST-clamp preds: {stats(preds)}")
    print(f"  PRE -clamp preds: {stats(pre)}")
    at_floor = (preds <= MIN_VALUE).float().mean().item()
    print(f"  preds exactly at clamp floor ({MIN_VALUE:g}): {at_floor*100:.2f}%")
    if preds.std().item() == 0.0:
        # float32 tolerance: torch.float32(1e-3) == 0.001000000047...
        if abs(preds[0].item() - MIN_VALUE) < 1e-9:
            print("  --> std == 0 at 1e-3: hypothesis (A) CONFIRMED (clamp trap)")
        else:
            print(f"  --> std == 0 at {preds[0].item():g}: constant output, "
                  "different cause")
    else:
        print("  --> std > 0: predictions vary; hypothesis (B) territory")
    return preds, targets


def step3_ranking(preds, targets):
    print()
    print("=" * 70)
    print("[3] Ranking vs calibration on the full val set")
    print("=" * 70)
    p, t = preds.numpy(), targets.numpy()
    r2 = r2_score(t, p)
    if p.std() == 0:
        print(f"  R^2 = {r2:.4f};  Spearman undefined (constant predictions)")
        return
    rho, pval = spearmanr(p, t)
    print(f"  R^2      = {r2:.4f}   (calibration — what the logs report)")
    print(f"  Spearman = {rho:.4f}  (ranking — what A* consumes; p={pval:.2e})")


def step4_history():
    print()
    print("=" * 70)
    print("[4] history_losses.json: bit-for-bit identical epochs?")
    print("=" * 70)
    hist = json.loads((MODEL_DIR / "history_losses.json").read_text())
    for k, v in hist.items():
        arr = np.asarray(v)
        n_unique = len(np.unique(arr))
        first_frozen = next(
            (i for i in range(1, len(v)) if v[i:] == [v[i]] * (len(v) - i)),
            len(v))
        print(f"  {k:11s}: {len(v)} epochs, {n_unique} unique values; "
              f"constant from epoch {first_frozen} onward "
              f"(value {v[-1]:.10f})")
    print("  (Training loop appends a fresh evaluate() result each epoch — "
        "identical entries mean the model output itself stopped changing.)")


def main():
    train_c, test_c, _ = step1_targets()
    preds, targets = step2_predictions(train_c, test_c)
    step3_ranking(preds, targets)
    step4_history()


if __name__ == "__main__":
    main()
