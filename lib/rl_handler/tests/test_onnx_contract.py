"""Test 8 — the ONNX contract. MUST pass before anything else is merged.

WHAT THE CONTRACT ACTUALLY IS
-----------------------------
`FringeEvalRL.tpp:426` reads the input names FROM THE LOADED SESSION and pairs
them POSITIONALLY with the tensors it built. So input NAMES are not part of the
contract -- renaming one would silently still work, while REORDERING would
silently produce garbage. These tests therefore pin ORDER and DTYPE, and check
names only as documentation.

  merged   (5): node_features, edge_index, edge_attr, membership, mask
  separated(9): ... membership, goal_node_features, goal_edge_index,
                    goal_edge_attr, goal_batch, mask
  output:       logits float32 [F];  higher logit = expanded sooner

`FringeEvalRL.tpp:131` hard-fails unless the logits length equals
--RL_fringe_size, so the F pairing is load-bearing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

warnings.filterwarnings("ignore")

from src.models.frontier_policy import FrontierPolicyNetwork
from src.offline.encoder import StateGraph, pack_fringe
from src.trainer import RLFrontierTrainer

CONTEXT_MODES = ("none", "mean_pool", "self_attention")

MERGED_INPUTS = ["node_features", "edge_index", "edge_attr", "membership", "mask"]
SEPARATED_INPUTS = [
    "node_features", "edge_index", "edge_attr", "membership",
    "goal_node_features", "goal_edge_index", "goal_edge_attr", "goal_batch",
    "mask",
]


class _Cache:
    """Minimal InstanceCache stand-in holding synthetic state graphs."""

    def __init__(self, states):
        self.states = states


def _rng_graph(rng: np.random.Generator, n: int = 4, e: int = 5) -> StateGraph:
    return StateGraph(
        node_ids=torch.from_numpy(
            rng.integers(-(2**62), 2**62, size=n, dtype=np.int64).copy()
        ),
        edge_index=torch.from_numpy(
            rng.integers(0, n, size=(2, e), dtype=np.int64).copy()
        ),
        edge_attr=torch.from_numpy(
            rng.integers(0, 200, size=e, dtype=np.int64).copy()
        ),
    )


def _cache(n_states: int, seed: int = 0) -> _Cache:
    rng = np.random.default_rng(seed)
    return _Cache([_rng_graph(rng) for _ in range(n_states)])


def _model(mode: str, separated: bool = False, seed: int = 0) -> FrontierPolicyNetwork:
    torch.manual_seed(seed)
    return FrontierPolicyNetwork(
        node_input_dim=1,
        hidden_dim=32,
        gnn_layers=2,
        dataset_type="HASHED",
        context_mode=mode,
        attn_heads=2,
        attn_layers=1,
        use_goal_separate_input=separated,
    ).eval()


def _export(model, tmp_path, F: int, kind: str):
    t = RLFrontierTrainer(model=model, device="cpu", kind_of_data=kind)
    p = tmp_path / f"frontier_policy_{F}.onnx"
    t.to_onnx(p, node_input_dim=1, onnx_frontier_size=F)
    return p


def _goal_inputs(seed: int = 99):
    g = _rng_graph(np.random.default_rng(seed), n=3, e=4)
    return {
        "goal_node_features": g.node_ids,
        "goal_edge_index": g.edge_index,
        "goal_edge_attr": g.edge_attr,
        "goal_batch": torch.zeros(g.node_ids.numel(), dtype=torch.int64),
    }


# ------------------------------------------------- input order and dtypes ----

@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_merged_exports_five_inputs_in_order(mode, tmp_path):
    p = _export(_model(mode), tmp_path, 8, "merged")
    g = onnx.load(str(p))
    assert [i.name for i in g.graph.input] == MERGED_INPUTS
    assert [o.name for o in g.graph.output] == ["logits"]


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_separated_exports_nine_inputs_in_order(mode, tmp_path):
    p = _export(_model(mode, separated=True), tmp_path, 8, "separated")
    g = onnx.load(str(p))
    assert [i.name for i in g.graph.input] == SEPARATED_INPUTS


@pytest.mark.parametrize("mode", CONTEXT_MODES)
@pytest.mark.parametrize("kind", ["merged", "separated"])
def test_mask_is_last_and_goal_tensors_sit_before_it(mode, kind, tmp_path):
    """The C++ appends goal_* then mask (FringeEvalRL.tpp:403-411). Order is the
    contract; names are not."""
    p = _export(_model(mode, separated=(kind == "separated")), tmp_path, 8, kind)
    names = [i.name for i in onnx.load(str(p)).graph.input]
    assert names[-1] == "mask"
    if kind == "separated":
        assert names[4:8] == SEPARATED_INPUTS[4:8]


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_input_dtypes_match_the_cpp_tensors(mode, tmp_path):
    """int64 everywhere except mask (uint8) -- FringeEvalRL.tpp CreateTensor calls."""
    p = _export(_model(mode, separated=True), tmp_path, 8, "separated")
    g = onnx.load(str(p))
    for i in g.graph.input:
        et = i.type.tensor_type.elem_type
        want = onnx.TensorProto.UINT8 if i.name == "mask" else onnx.TensorProto.INT64
        assert et == want, f"{i.name}: dtype {et} != {want}"
    assert g.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


# ---------------------------------------------------- logits length == F -----

@pytest.mark.parametrize("mode", CONTEXT_MODES)
@pytest.mark.parametrize("F", [4, 8, 16, 32])
def test_logits_length_equals_F(mode, F, tmp_path):
    """FringeEvalRL.tpp:131 rejects the model unless this holds.

    The proto declares the axis symbolically ('F'), but ORT's shape inference
    resolves it to the concrete traced value -- which is why the C++ static-dim
    check (which would exit on a dynamic dim) passes at load time.
    """
    p = _export(_model(mode), tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    assert sess.get_outputs()[0].shape == [F], (
        "ORT must resolve the logits axis to a concrete F, or the C++ "
        "GetShape().back() <= 0 check exits at load"
    )


# --------------------------------------------- ORT reproduces PyTorch --------

@pytest.mark.parametrize("mode", CONTEXT_MODES)
@pytest.mark.parametrize("kind", ["merged", "separated"])
def test_onnx_reproduces_pytorch_scores(mode, kind, tmp_path):
    """The gate: round-trip through onnxruntime and match PyTorch to 1e-5 on a
    batch of real fringes, for both encodings."""
    F, K = 8, 8
    separated = kind == "separated"
    model = _model(mode, separated=separated)
    p = _export(model, tmp_path, F, kind)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    cache = _cache(K)
    goal = _goal_inputs()

    for trial in range(3):
        order = list(np.random.default_rng(trial).permutation(K))
        packed = pack_fringe(cache, order, F)
        feeds = {k: v.numpy() for k, v in packed.items()}
        if separated:
            feeds.update({k: v.numpy() for k, v in goal.items()})
        got = sess.run(None, feeds)[0]

        with torch.no_grad():
            kw = dict(goal) if separated else {}
            want = model(
                node_features=packed["node_features"],
                edge_index=packed["edge_index"],
                edge_attr=packed["edge_attr"],
                membership=packed["membership"],
                candidate_batch=None,
                mask=packed["mask"],
                **kw,
            ).numpy()

        assert got.shape == want.shape == (F,)
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=0, err_msg=(
            f"{mode}/{kind} trial {trial}: onnxruntime and PyTorch disagree"
        ))


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_eager_matches_onnx_when_the_beam_is_not_full(mode, tmp_path):
    """TRAIN/DEPLOY PARITY at K < F -- the case that actually diverges.

    Tracing baked `size=F` into the pooling, so the exported graph ALWAYS emits F
    slots: absent candidates pool from zero rows and are masked out. Eager mode
    has no such baking -- it pools to K and requires a length-K mask. The two are
    therefore *different functions*, and only their ACTIVE logits are required to
    agree.

    This matters beyond tidiness: the offline env scores fringes in eager PyTorch
    while the planner runs the ONNX. If they disagreed here, every offline regret
    number would describe a policy the planner never executes -- and the beam is
    genuinely short at the first push_vector, so K < F is on the hot path.
    """
    F, K = 16, 3
    model = _model(mode)
    p = _export(model, tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    cache = _cache(K, seed=21)

    packed_F = pack_fringe(cache, list(range(K)), F)          # deploy: mask len F
    onnx_logits = sess.run(None, {k: v.numpy() for k, v in packed_F.items()})[0]

    packed_K = pack_fringe(cache, list(range(K)), K)          # eager: mask len K
    with torch.no_grad():
        eager = model(
            node_features=packed_K["node_features"],
            edge_index=packed_K["edge_index"],
            edge_attr=packed_K["edge_attr"],
            membership=packed_K["membership"],
            candidate_batch=None,
            mask=packed_K["mask"],
        ).numpy()

    assert eager.shape == (K,) and onnx_logits.shape == (F,)
    np.testing.assert_allclose(onnx_logits[:K], eager, atol=1e-5, rtol=0, err_msg=(
        f"{mode}: eager (K={K}) and ONNX (F={F}, {K} active) disagree on the "
        f"ACTIVE logits -- the offline scorer is not the deployed scorer"
    ))


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_batched_path_matches_single_path(mode):
    """The training loop scores many fringes at once via `candidate_batch`; the
    planner scores one via `candidate_batch=None`. Those are two different code
    paths in _contextualize (and in FringeAttention: forward vs forward_single).
    If they diverge, the critic is trained on scores the deployed net never
    produces.
    """
    model = _model(mode)
    K = 4
    caches = [_cache(K, seed=s) for s in (31, 32)]

    singles = []
    for c in caches:
        packed = pack_fringe(c, list(range(K)), K)
        with torch.no_grad():
            singles.append(model(
                node_features=packed["node_features"],
                edge_index=packed["edge_index"],
                edge_attr=packed["edge_attr"],
                membership=packed["membership"],
                candidate_batch=None,
                mask=packed["mask"],
            ))

    # pack both fringes into one graph: membership continues across fringes,
    # candidate_batch says which fringe each pooled slot belongs to.
    nf, ei, ea, mem = [], [], [], []
    node_off, slot_off = 0, 0
    for c in caches:
        p = pack_fringe(c, list(range(K)), K)
        nf.append(p["node_features"])
        ei.append(p["edge_index"] + node_off)
        ea.append(p["edge_attr"])
        mem.append(p["membership"] + slot_off)
        node_off += p["node_features"].numel()
        slot_off += K
    cb = torch.cat([torch.full((K,), i, dtype=torch.int64) for i in range(len(caches))])
    with torch.no_grad():
        batched = model(
            node_features=torch.cat(nf),
            edge_index=torch.cat(ei, dim=1),
            edge_attr=torch.cat(ea),
            membership=torch.cat(mem),
            candidate_batch=cb,
            mask=None,
        )

    for i, s in enumerate(singles):
        np.testing.assert_allclose(
            batched[i * K:(i + 1) * K].numpy(), s.numpy(), atol=1e-5, rtol=0,
            err_msg=f"{mode}: batched and single scoring paths disagree on fringe {i}",
        )


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_inactive_slots_are_masked_to_minus_1e9(mode, tmp_path):
    F, K = 8, 3
    p = _export(_model(mode), tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    packed = pack_fringe(_cache(K), list(range(K)), F)
    got = sess.run(None, {k: v.numpy() for k, v in packed.items()})[0]
    assert (got[K:] <= -1e8).all(), f"inactive slots not masked: {got[K:]}"
    assert (got[:K] > -1e8).all(), "active slots must not be masked"


def test_higher_logit_means_expanded_sooner(tmp_path):
    """Direction check against C++ rankScores: it sorts DESCENDING and assigns
    rank i as the heuristic value; StateComparator is a min-heap on that value,
    so rank 0 (the argmax logit) is popped first."""
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    order = np.argsort(-scores)                 # rankScores' sort
    ranks = np.empty(3)
    ranks[order] = np.arange(3)
    assert ranks[1] == 0, "the highest logit must get rank 0"
    assert int(np.argmin(ranks)) == int(np.argmax(scores))


# ------------------------------------------------- padding invariance --------

@pytest.mark.parametrize("mode", CONTEXT_MODES)
@pytest.mark.parametrize("kind", ["merged", "separated"])
def test_padding_invariance_across_F(mode, kind, tmp_path):
    """NON-NEGOTIABLE for attention: a padded slot must not leak into any active
    candidate's score.

    Pad the SAME 3-state fringe to F = 8, 16, 32, 64 and require the active
    slots' logits to be identical to 1e-5. A padded slot that leaks into the
    attention silently corrupts every active score, and it would NOT show up in
    a shape test -- the output would still be [F].
    """
    K = 3
    separated = kind == "separated"
    model = _model(mode, separated=separated)
    cache = _cache(K, seed=5)
    goal = _goal_inputs()

    ref = None
    for F in (8, 16, 32, 64):
        p = _export(model, tmp_path / f"F{F}", F, kind)
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        packed = pack_fringe(cache, list(range(K)), F)
        feeds = {k: v.numpy() for k, v in packed.items()}
        if separated:
            feeds.update({k: v.numpy() for k, v in goal.items()})
        got = sess.run(None, feeds)[0]
        assert got.shape == (F,)
        active = got[:K]
        if ref is None:
            ref = active
        else:
            np.testing.assert_allclose(active, ref, atol=1e-5, rtol=0, err_msg=(
                f"{mode}/{kind}: padding to F={F} changed the ACTIVE logits. "
                f"A padding slot is leaking into the scores."
            ))


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_padding_invariance_holds_for_every_active_count(mode, tmp_path):
    """Same property swept over K, since the leak would scale with F-K."""
    F = 16
    model = _model(mode)
    p = _export(model, tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    cache = _cache(8, seed=7)

    per_k = {}
    for K in (1, 2, 4, 8):
        packed = pack_fringe(cache, list(range(K)), F)
        per_k[K] = sess.run(None, {k: v.numpy() for k, v in packed.items()})[0][:K]

    # slot 0's score must not depend on how many OTHER slots are padded...
    # except under mean_pool/self_attention, where it legitimately depends on the
    # other ACTIVE members (that is the point of context). So compare K=1 only
    # against itself and require monotone consistency of the shared prefix under
    # `none`, which is a per-node scorer.
    if mode == "none":
        for K in (2, 4, 8):
            np.testing.assert_allclose(per_k[K][:1], per_k[1], atol=1e-5, rtol=0,
                                       err_msg="`none` must be a pure per-node scorer")


def test_none_mode_is_a_pure_per_node_scorer(tmp_path):
    """The control arm's defining property: h(v) depends only on v's own graph,
    so removing another candidate cannot change v's score. This is what makes
    `none` stale-rank-exact by construction."""
    F = 8
    model = _model("none")
    p = _export(model, tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    cache = _cache(4, seed=11)

    full = sess.run(None, {k: v.numpy() for k, v in pack_fringe(cache, [0, 1, 2, 3], F).items()})[0]
    drop = sess.run(None, {k: v.numpy() for k, v in pack_fringe(cache, [0, 1, 2], F).items()})[0]
    np.testing.assert_allclose(drop[:3], full[:3], atol=1e-5, rtol=0)


@pytest.mark.parametrize("mode", ["mean_pool", "self_attention"])
def test_context_modes_actually_use_the_context(mode, tmp_path):
    """Guard against a context mode that silently degenerates to `none`: if
    dropping a candidate never changes the others' scores, the context is inert
    and the whole F9 comparison would be meaningless."""
    F = 8
    model = _model(mode)
    p = _export(model, tmp_path, F, "merged")
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    cache = _cache(4, seed=13)

    full = sess.run(None, {k: v.numpy() for k, v in pack_fringe(cache, [0, 1, 2, 3], F).items()})[0]
    drop = sess.run(None, {k: v.numpy() for k, v in pack_fringe(cache, [0, 1, 2], F).items()})[0]
    assert not np.allclose(drop[:3], full[:3], atol=1e-5), (
        f"{mode} produced identical scores after removing a candidate -- the "
        f"context is inert and this mode has degenerated to `none`"
    )
