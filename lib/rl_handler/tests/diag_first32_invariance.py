"""Diagnostic: is the deployed fringe-ranking ONNX context-dependent?

Black-box test of the DEPLOYED frontier_policy_{32,64}.onnx via onnxruntime
(the artifact the C++ planner runs).  Question: given a fringe of 32 real
states, if we append 32 more (so the first 32 are byte-identical), do the
first-32 scores stay the same during inference?

Mechanism under test: the packed fringe is a DISCONNECTED graph, so message
passing cannot move information across states.  The only cross-state coupling
is the fringe-wide global context (use_global_context=True).  So appending
states should SHIFT the first-32 logits via the changed global mean; a
bit-identical result would falsify context-dependence.

Run from lib/rl_handler:
    ../../.venv/bin/python tests/diag_first32_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.encoder import InstanceCache, pack_fringe  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

# One cached instance per domain that holds >= 64 real states.
DOMAIN_CACHE = {
    "CC": "exp/rl_exp/batch0/_models/CC/training_data/{inst}/graph_cache_offline_v1.pt",
    "SC": "exp/rl_exp/batch0/_models/SC/training_data/{inst}/graph_cache_offline_v1.pt",
}
MODEL = "exp/rl_exp/batch0/_models/{dom}/frontier_policy_{f}.onnx"


def load_cache_with_ge64(dom: str) -> tuple[str, InstanceCache] | None:
    """First instance cache for `dom` with >= 64 states, or None if none exist."""
    import glob

    pat = DOMAIN_CACHE[dom].format(inst="*")
    for p in sorted(glob.glob(str(REPO / pat))):
        try:
            payload = torch.load(p, weights_only=False)
        except Exception:
            continue
        states = payload.get("states")
        if states is not None and len(states) >= 64:
            return Path(p).parent.name, InstanceCache(states)
    return None


def models_present(dom: str) -> bool:
    return all((REPO / MODEL.format(dom=dom, f=f)).exists() for f in (32, 64))


def feeds_from_pack(out: dict) -> dict:
    return {
        "node_features": out["node_features"].numpy(),
        "edge_index": out["edge_index"].numpy(),
        "edge_attr": out["edge_attr"].numpy(),
        "membership": out["membership"].numpy(),
        "mask": out["mask"].numpy(),
    }


def assert_leading_byte_identical(a: dict, b: dict, h: int) -> None:
    """The leading `h` slots of pack B must equal all of pack A exactly.

    membership/node_features/edge_attr for slots 0..h-1, and the edge_index
    columns belonging to those slots, must be bit-identical so any logit
    difference on those slots is purely the global-context effect.
    """
    nh = int((a["membership"] < h).sum())  # all of A's nodes are slots 0..h-1
    assert int((b["membership"] < h).sum()) == nh, "node count of leading half differs"
    assert torch.equal(a["node_features"], b["node_features"][:nh]), "node_features"
    assert torch.equal(a["membership"], b["membership"][:nh]), "membership"
    e_a = a["edge_index"].shape[1]
    assert torch.equal(a["edge_index"], b["edge_index"][:, :e_a]), "edge_index"
    assert torch.equal(a["edge_attr"], b["edge_attr"][:e_a]), "edge_attr"
    assert bool((b["edge_index"][:, :e_a] < nh).all()), "leading edges escape block"


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    num = 0
    den = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            num += 1 if sx == sy else -1
            den += 1
    return num / den if den else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def run_model(dom: str, f: int, cache: InstanceCache) -> dict:
    """Half-vs-full context test for the native-fringe-size `f` model.

    A packs the leading h=f//2 real states; B packs the full f real states.
    The leading h slots are byte-identical, so any logit shift on them is the
    global-context effect (the only cross-state coupling in the disconnected
    fringe graph).  We use half/full rather than 32/64 because each ONNX has a
    FIXED scatter/output dim equal to its native f — policy_32 rejects N>32.
    """
    sess = ort.InferenceSession(
        str(REPO / MODEL.format(dom=dom, f=f)), providers=["CPUExecutionProvider"]
    )

    h = f // 2
    idx = list(range(f))
    pack_a = pack_fringe(cache, idx[:h], fringe_size=f)   # h real states, padded
    pack_b = pack_fringe(cache, idx[:f], fringe_size=f)   # f real states, leading h == A
    assert_leading_byte_identical(pack_a, pack_b, h)

    (logits_a,) = sess.run(["logits"], feeds_from_pack(pack_a))
    (logits_b,) = sess.run(["logits"], feeds_from_pack(pack_b))
    a = logits_a[:h]
    b = logits_b[:h]

    max_abs_diff = float(np.max(np.abs(a - b)))
    return {
        "dom": dom,
        "f": f,
        "h": h,
        "max_abs_diff": max_abs_diff,
        "exact_equal": max_abs_diff < 1e-5,
        "kendall_tau": kendall_tau(a, b),
        "spearman": spearman(a, b),
        "argmax_same": int(np.argmax(a)) == int(np.argmax(b)),
        "a_argmax": int(np.argmax(a)),
        "b_argmax": int(np.argmax(b)),
    }


def control_permutation(dom: str, cache: InstanceCache) -> None:
    """Shuffle the 32 states, re-pack, re-run; logits must permute identically."""
    sess = ort.InferenceSession(
        str(REPO / MODEL.format(dom=dom, f=32)), providers=["CPUExecutionProvider"]
    )
    base = list(range(32))
    (logits_base,) = sess.run(
        ["logits"], feeds_from_pack(pack_fringe(cache, base, fringe_size=32))
    )
    perm = [base[i] for i in [13, 0, 7, 31, 2, 19, 5, 28, 11, 1, 24, 6, 30, 3,
                              17, 9, 22, 4, 26, 8, 15, 12, 29, 10, 20, 14, 27,
                              16, 25, 18, 23, 21]]
    (logits_perm,) = sess.run(
        ["logits"], feeds_from_pack(pack_fringe(cache, perm, fringe_size=32))
    )
    # logits_perm[slot] is the score of state perm[slot]; compare to base score
    inv = [perm.index(s) for s in base]
    follow = np.array([logits_perm[inv[s]] for s in range(32)])
    d = float(np.max(np.abs(follow - logits_base[:32])))
    assert d < 1e-4, f"[{dom}] permutation control failed: scores do not follow states (max diff {d})"
    print(f"  control [{dom}]: scores follow their state under permutation (max diff {d:.2e}) OK")


def main() -> None:
    rows = []
    for dom in ("CC", "SC"):
        if not models_present(dom):
            print(f"[skip] {dom}: deployed ONNX missing "
                  f"({MODEL.format(dom=dom, f=32)} / _64)")
            continue
        found = load_cache_with_ge64(dom)
        if found is None:
            print(f"[skip] {dom}: no graph cache with >=64 states on disk")
            continue
        inst, cache = found
        print(f"[{dom}] using instance {inst} ({len(cache.states)} states)")
        control_permutation(dom, cache)
        for f in (32, 64):
            rows.append(run_model(dom, f, cache))

    if not rows:
        print("[skip] no domain has both deployed ONNX and a >=64-state cache; "
              "nothing to compare.")
        return

    hdr = f"{'model':<14}{'shared':>8}{'max_abs_diff':>14}{'exact_eq':>10}{'kendall':>10}{'spearman':>10}{'argmax_same':>13}"
    print("\n(A = leading h states padded to f; B = full f states; shared = the h byte-identical slots)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        name = f"{r['dom']}/policy_{r['f']}"
        print(
            f"{name:<14}{r['h']:>8}{r['max_abs_diff']:>14.6e}{str(r['exact_equal']):>10}"
            f"{r['kendall_tau']:>10.4f}{r['spearman']:>10.4f}"
            f"{str(r['argmax_same']):>13}  (argmax {r['a_argmax']}->{r['b_argmax']})"
        )


if __name__ == "__main__":
    main()
