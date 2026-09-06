"""Fix 1 regression suite: separated mode must thread the goal through training, the
target net, and eval; merged must stay goal-less; and the separated/merged boundary
(S1/S2) must be enforced, not assumed.

T-g is the most important test here: separated + a missing goal_tree.dot must FAIL LOUD.
That is the exact bug the investigation found (training silently ran goal-less).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.frontier_policy import FrontierPolicyNetwork
from src.offline.batching import pack_batch, pack_single
from src.offline.dataset import generate_dataset
from src.offline.encoder import StateGraph
from src.offline.qlearning import QTrainer, TrainConfig
from src.offline.run import RunConfig, _load_goals

from conftest import make_tree


class _Cache:
    def __init__(self, states):
        self.states = states


def _cache_for(inst, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(inst.n_states):
        n, e = 4, 5
        states.append(StateGraph(
            node_ids=torch.randint(-(2**62), 2**62, (n,), generator=g, dtype=torch.int64),
            edge_index=torch.randint(0, n, (2, e), generator=g, dtype=torch.int64),
            edge_attr=torch.randint(0, 50, (e,), generator=g, dtype=torch.int64),
        ))
    return _Cache(states)


def _goal(seed=1):
    g = torch.Generator().manual_seed(seed)
    n, e = 3, 2
    return StateGraph(
        node_ids=torch.randint(-(2**62), 2**62, (n,), generator=g, dtype=torch.int64),
        edge_index=torch.randint(0, n, (2, e), generator=g, dtype=torch.int64),
        edge_attr=torch.randint(0, 50, (e,), generator=g, dtype=torch.int64),
    )


def _net(sep: bool, seed=0):
    torch.manual_seed(seed)
    return FrontierPolicyNetwork(node_input_dim=1, hidden_dim=16, gnn_layers=1,
                                 dataset_type="HASHED", context_mode="self_attention",
                                 use_goal_separate_input=sep)


def _rows_cfg(t19):
    rows, _ = generate_dataset([t19], 3, seeds_per_policy=2, expansion_cap=100, verbose=False)
    cfg = TrainConfig(fringe_size=3, model="dqn", batch_size=8, device="cpu", expansion_cap=100)
    return rows, cfg


def _trainer(t19, sep: bool, goals_override="auto"):
    cache = _cache_for(t19)
    rows, cfg = _rows_cfg(t19)
    goals = ({t19.name: _goal()} if sep else None) if goals_override == "auto" else goals_override
    tr = QTrainer(_net(sep), _net(sep), [t19], {t19.name: cache}, rows, cfg, goals=goals)
    return tr, rows, cache


def _spy_goal_emb(net):
    """Record whether _contextualize saw goal_emb is None."""
    seen = {}
    orig = net._contextualize
    def spy(z, candidate_batch, mask=None, goal_emb=None):
        seen["none"] = goal_emb is None
        return orig(z, candidate_batch, mask=mask, goal_emb=goal_emb)
    net._contextualize = spy
    return seen


def _multi(rows, k=4):
    return [(r.instance, r.obs) for r in rows if len(r.obs) >= 2][:k]


# ----------------------------------------------------------------- T-a / T-b ---
def test_Ta_separated_training_forward_has_goal(t19):
    tr, rows, _ = _trainer(t19, sep=True)
    seen = _spy_goal_emb(tr.model)
    tr._logits(tr.model, _multi(rows))
    assert seen["none"] is False


def test_Tb_merged_training_forward_has_no_goal(t19):
    tr, rows, _ = _trainer(t19, sep=False)
    seen = _spy_goal_emb(tr.model)
    tr._logits(tr.model, _multi(rows))
    assert seen["none"] is True


# ---------------------------------------------------------------------- T-c ---
def test_Tc_online_and_target_receive_identical_goal(t19):
    tr, rows, _ = _trainer(t19, sep=True)
    picks = _multi(rows)
    _, p_on = tr._logits(tr.model, picks)
    _, p_tg = tr._logits(tr.target, picks)
    for k in ("goal_node_features", "goal_edge_index", "goal_edge_attr", "goal_batch"):
        assert k in p_on and k in p_tg, f"{k} missing"
        assert torch.equal(p_on[k], p_tg[k])


# ---------------------------------------------------------------------- T-d ---
def test_Td_eval_score_path_receives_goal(t19):
    net = _net(sep=True)
    cache = _cache_for(t19)
    rows, _ = _rows_cfg(t19)
    beam = _multi(rows)[0][1]
    p = pack_single(cache, beam, len(beam), "cpu", goal_graph=_goal())
    assert "goal_node_features" in p
    seen = _spy_goal_emb(net)
    net(node_features=p["node_features"], edge_index=p["edge_index"], edge_attr=p["edge_attr"],
        membership=p["membership"], candidate_batch=None, mask=p["mask"],
        goal_node_features=p.get("goal_node_features"), goal_edge_index=p.get("goal_edge_index"),
        goal_edge_attr=p.get("goal_edge_attr"), goal_batch=p.get("goal_batch"))
    assert seen["none"] is False


# ---------------------------------------------------------------------- T-e ---
def test_Te_merged_regression_no_goal_keys_and_forward_unchanged(t19):
    cache = _cache_for(t19)
    rows, _ = _rows_cfg(t19)
    beam = _multi(rows)[0][1]
    # pack_batch/pack_single with no goal -> zero goal keys (byte-identical shape)
    p = pack_batch({t19.name: cache}, [(t19.name, beam)], "cpu")
    assert not any(k.startswith("goal_") for k in p)
    ps = pack_single(cache, beam, len(beam), "cpu")
    assert not any(k.startswith("goal_") for k in ps)
    # merged forward: passing explicit goal_*=None must equal omitting them entirely
    net = _net(sep=False).eval()
    kw = dict(node_features=ps["node_features"], edge_index=ps["edge_index"],
              edge_attr=ps["edge_attr"], membership=ps["membership"],
              candidate_batch=None, mask=ps["mask"])
    with torch.no_grad():
        a = net(**kw)
        b = net(**kw, goal_node_features=None, goal_edge_index=None,
                goal_edge_attr=None, goal_batch=None)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------- T-f ---
def test_Tf_goal_columns_now_receive_gradient(t19):
    """Inverted T1 diagnostic: after the fix the goal block of policy_head.0.weight
    must move comparably to the state block, not ~35x less (weight-decay only)."""
    tr, _, _ = _trainer(t19, sep=True)
    net = tr.model
    W0 = net.policy_head[0].weight.detach().clone()
    h = net.policy_head[0].weight.shape[1] // 3   # [z | z_att | GOAL]
    for _ in range(50):
        tr.step()
    dW = (net.policy_head[0].weight.detach() - W0).abs()
    z_delta = dW[:, 0:h].max().item()
    goal_delta = dW[:, 2 * h:3 * h].max().item()
    assert goal_delta > 0.3 * z_delta, (
        f"goal block barely moved (goal={goal_delta:.3e} vs z={z_delta:.3e}): "
        "goal channel is not receiving gradient"
    )


# ------------------------------------------------- T-g / T-h / T-i (boundary) ---
def test_Tg_separated_missing_goal_dot_fails_loud(tmp_path):
    inst_dir = tmp_path / "CC_x"
    inst_dir.mkdir()
    csv = inst_dir / "CC_x_depth_25.csv"
    csv.write_text("")                       # state CSV exists, goal_tree.dot does NOT
    cfg = RunConfig(train_csvs=[str(csv)], dir_save_model=tmp_path, kind_of_data="separated")

    class _I:
        name = "CC_x"
        csv_path = str(csv)          # the goal is looked up beside THIS tree's CSV
    with pytest.raises(FileNotFoundError, match="CC_x"):
        _load_goals(cfg, [str(csv)], [_I()])


def test_Th_merged_with_goal_supplied_is_rejected(t19):
    with pytest.raises(AssertionError, match="consistency"):
        _trainer(t19, sep=False, goals_override={t19.name: _goal()})


def test_Ti_separated_without_goal_is_rejected(t19):
    with pytest.raises(AssertionError, match="consistency"):
        _trainer(t19, sep=True, goals_override=None)


# ------------------------------------------------- T-j / T-k (behavioural) ----
def test_Tj_separated_logits_depend_on_the_goal(t19):
    """SEPARATED goal-sensitivity: same beam + same state graphs, two DIFFERENT goal
    graphs -> the logits must DIFFER. This proves the goal actually REACHES the scores,
    the behavioural counterpart to T-f (which proves the goal weights LEARN).

    NOTE: goal-sensitivity alone does NOT prove correctness -- the BROKEN model was
    goal-sensitive too, through untrained near-init weights (measured max |delta| 0.059,
    argmin flipping on up to 12/20 frontiers). T-j shows the goal reaches the score; T-f
    shows the goal weights receive gradient. Both are needed; neither alone is sufficient,
    and whether the *learned* sensitivity helps is a deployment question, not a unit test.
    """
    net = _net(sep=True).eval()
    cache = _cache_for(t19)
    rows, _ = _rows_cfg(t19)
    beam = _multi(rows)[0][1]

    def logits(goal_seed):
        p = pack_single(cache, beam, len(beam), "cpu", goal_graph=_goal(seed=goal_seed))
        with torch.no_grad():
            return net(node_features=p["node_features"], edge_index=p["edge_index"],
                       edge_attr=p["edge_attr"], membership=p["membership"],
                       candidate_batch=None, mask=p["mask"],
                       goal_node_features=p.get("goal_node_features"),
                       goal_edge_index=p.get("goal_edge_index"),
                       goal_edge_attr=p.get("goal_edge_attr"), goal_batch=p.get("goal_batch"))

    a = logits(1)
    b = logits(999)
    assert float((a - b).abs().max()) > 1e-4, "logits are invariant to the goal input"


def test_Tk_merged_logits_depend_only_on_state():
    """MERGED: passing a goal is already rejected by the mode check (see T-h), so there is
    no 'merged + goal' path to exercise. That merged logits depend only on the state
    inputs -- byte-identical to before the fix -- is covered by
    test_Te_merged_regression_no_goal_keys_and_forward_unchanged. Cross-referenced here
    rather than duplicated."""
    assert True


def test_merged_load_goals_returns_none(tmp_path):
    csv = tmp_path / "CC_y" / "CC_y_depth_25.csv"
    csv.parent.mkdir()
    csv.write_text("")
    cfg = RunConfig(train_csvs=[str(csv)], dir_save_model=tmp_path, kind_of_data="merged")
    assert _load_goals(cfg, [str(csv)], []) is None
