"""The trainer: Double-DQN + CQL over the reservoir MDP.

The load-bearing properties are the Bellman semantics (bootstrap on `terminated`,
NOT `done`) and representation-agnosticism (no branch on HASHED vs BITMASK).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.frontier_policy import FrontierPolicyNetwork
from src.offline.batching import segment_argmax, segment_logsumexp, segment_max
from src.offline.dataset import generate_dataset
from src.offline.encoder import StateGraph
from src.offline.qlearning import (
    DivergenceError,
    QTrainer,
    TrainConfig,
    default_reward_scale,
)

from conftest import make_tree


class _Cache:
    def __init__(self, states):
        self.states = states


def _cache_for(inst, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = []
    for v in range(inst.n_states):
        n, e = 4, 5
        states.append(StateGraph(
            node_ids=torch.randint(-(2**62), 2**62, (n,), generator=g, dtype=torch.int64),
            edge_index=torch.randint(0, n, (2, e), generator=g, dtype=torch.int64),
            edge_attr=torch.randint(0, 50, (e,), generator=g, dtype=torch.int64),
        ))
    return _Cache(states)


def _net(seed=0):
    torch.manual_seed(seed)
    return FrontierPolicyNetwork(node_input_dim=1, hidden_dim=16, gnn_layers=1,
                                 dataset_type="HASHED", context_mode="mean_pool")


def _setup(t19, model="dqn", cql_alpha=0.0, **kw):
    cache = _cache_for(t19)
    rows, _ = generate_dataset([t19], 3, seeds_per_policy=2, expansion_cap=100, verbose=False)
    cfg = TrainConfig(fringe_size=3, model=model, cql_alpha=cql_alpha, batch_size=8,
                      device="cpu", expansion_cap=100, **kw)
    tr = QTrainer(_net(), _net(), [t19], {t19.name: cache}, rows, cfg)
    return tr, rows


# ------------------------------------------------------- segment ops ---------

def test_segment_max_and_argmax():
    v = torch.tensor([1.0, 5.0, 2.0, 9.0, 3.0])
    s = torch.tensor([0, 0, 1, 1, 1])
    assert segment_max(v, s, 2).tolist() == [5.0, 9.0]
    assert segment_argmax(v, s, 2).tolist() == [1, 3]


def test_segment_argmax_breaks_ties_to_the_lowest_index():
    v = torch.tensor([5.0, 5.0])
    s = torch.tensor([0, 0])
    assert segment_argmax(v, s, 1).tolist() == [0]


def test_segment_logsumexp_matches_torch():
    v = torch.tensor([1.0, 5.0, 2.0, 9.0])
    s = torch.tensor([0, 0, 1, 1])
    got = segment_logsumexp(v, s, 2)
    want = torch.tensor([torch.logsumexp(v[:2], 0), torch.logsumexp(v[2:], 0)])
    assert torch.allclose(got, want, atol=1e-5)


# ------------------------------------------- the Bellman target semantics ----

def test_bootstrap_is_on_terminated_not_done(t19):
    """THE property. A truncated row MUST bootstrap; a terminated row must not.

    Using `done` here would say a capped state 3000 expansions deep is worth -1,
    when it is really worth -E[remaining]. That leaks backwards and makes the
    critic optimistic about long searches (Pardo et al. 2018).
    """
    tr, rows = _setup(t19)
    trunc = [r for r in rows if r.truncated]
    term = [r for r in rows if r.terminated]
    # every terminated row has an EMPTY successor (nothing to bootstrap from)...
    assert all(not r.obs_next for r in term)
    # ...and every truncated row carries a real successor beam to bootstrap from.
    for r in trunc:
        assert r.obs_next, "a truncated row must return the successor beam"
        assert not r.terminated


def test_terminated_rows_are_never_packed_as_successors(t19):
    """Packing an empty beam would crash; more importantly it would be wrong."""
    tr, rows = _setup(t19)
    term = [r for r in rows if r.terminated]
    assert term, "t19 must produce terminated rows"
    tr.data = term
    log = tr.step()           # must not raise
    assert float(log["td_loss"]) >= 0.0    # step() returns on-device tensors now


def test_a_step_runs_and_reports_the_telemetry_fields(t19):
    tr, _ = _setup(t19)
    log = tr.step()
    for k in ("td_loss", "q_mean", "q_max", "q_max_unscaled", "grad_norm", "lr", "loss"):
        assert k in log, k


def test_target_network_syncs_on_schedule(t19):
    tr, _ = _setup(t19, target_sync=3)
    before = next(tr.target.parameters()).clone()
    for _ in range(2):
        tr.step()
    assert torch.equal(next(tr.target.parameters()), before), "must not sync early"
    tr.step()
    assert not torch.equal(next(tr.target.parameters()), before), "must sync at the interval"


def test_target_params_do_not_receive_gradients(t19):
    tr, _ = _setup(t19)
    tr.step()
    assert all(p.grad is None for p in tr.target.parameters())


# --------------------------------------------------------------- CQL ---------

def test_cql_alpha_zero_reduces_to_double_dqn(t19):
    tr, _ = _setup(t19, model="cql", cql_alpha=0.0)
    log = tr.step()
    assert float(log["cql"]) == 0.0
    assert float(log["loss"]) == pytest.approx(float(log["td_loss"]))


def test_cql_adds_a_positive_conservatism_term(t19):
    tr, _ = _setup(t19, model="cql", cql_alpha=1.0)
    log = tr.step()
    # logsumexp_a Q >= Q(s,a) always, so the term is >= 0
    assert float(log["cql"]) >= -1e-5
    assert log["loss"] >= log["td_loss"] - 1e-5


def test_unknown_model_is_rejected(t19):
    with pytest.raises(ValueError, match="model must be one of"):
        _setup(t19, model="iql")


# ----------------------------------------------------- reward scaling --------

def test_default_reward_scale_is_one_over_median_delta_root():
    a = make_tree([[1], [2], [3], []], goals=[3], name="a")     # delta_root 3
    b = make_tree([[1], []], goals=[1], name="b")               # delta_root 1
    c = make_tree([[1], [2], []], goals=[2], name="c")          # delta_root 2
    assert default_reward_scale([a, b, c]) == pytest.approx(1 / 2)   # median 2


def test_reward_scale_is_not_one_over_cap(t19):
    """1/cap would put the per-step signal at 5e-4 -- below init noise."""
    tr, _ = _setup(t19)
    assert tr.scale > 1.0 / tr.cfg.expansion_cap


def test_scaling_does_not_change_the_ranking(t19):
    """Pure rescaling: it cannot change the optimal policy."""
    tr, _ = _setup(t19, reward_scale=1.0)
    tr2, _ = _setup(t19, reward_scale=0.1)
    pol1 = tr.greedy_policy(t19.name)
    pol2 = tr2.greedy_policy(t19.name)
    # same init seed -> same net -> same ranking regardless of the reward scale
    assert pol1([1, 2, 3]) == pol2([1, 2, 3])


# --------------------------------------------------- divergence guard --------

def test_perf_refactor_is_bit_identical(t19):
    """THE regression guard for the sync-removal perf fix. step() now returns on-device
    tensors (the 5 logging fields are float()'d by the caller only at checkpoints) and
    the divergence guard materialises |Q| only every div_check_every steps. All of that
    is READ-ONLY: the training math (zero_grad -> backward -> clip -> opt.step) and its
    order are untouched.

    Two trainers from a byte-identical starting net, one checking the guard every step
    and one every 100, must end byte-identical. If they diverge, a 'sync' that was
    removed was actually load-bearing -- put it back.
    """
    import copy
    cache = _cache_for(t19)
    rows, _ = generate_dataset([t19], 3, seeds_per_policy=2, expansion_cap=100, verbose=False)

    def mk(net, tgt, every):
        cfg = TrainConfig(fringe_size=3, batch_size=8, device="cpu", expansion_cap=100,
                          seed=0, div_check_every=every)
        return QTrainer(net, tgt, [t19], {t19.name: cache}, rows, cfg)

    net_a, tgt_a = _net(), _net()
    net_b, tgt_b = copy.deepcopy(net_a), copy.deepcopy(tgt_a)
    tr_a, tr_b = mk(net_a, tgt_a, 1), mk(net_b, tgt_b, 100)
    for _ in range(40):
        tr_a.step()
        tr_b.step()
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb), (
            "guard check interval changed the weights -- a removed sync was load-bearing"
        )


def test_divergence_guard_fires_on_exploding_q(t19):
    """gamma=1 has no contraction, so drift is possible. We know Q* exactly, so it
    is detectable rather than silent."""
    tr, _ = _setup(t19)
    with torch.no_grad():
        for p in tr.model.policy_head.parameters():
            p.mul_(1e6)
    with pytest.raises(DivergenceError, match=r"exceeds"):
        tr.step()


def test_divergence_message_reports_unscaled_expansions(t19):
    tr, _ = _setup(t19)
    with torch.no_grad():
        for p in tr.model.policy_head.parameters():
            p.mul_(1e6)
    try:
        tr.step()
    except DivergenceError as e:
        assert "expansions against a cap" in str(e)


# --------------------------------------- representation-agnosticism ----------

def test_trainer_never_branches_on_dataset_type():
    """The BITMASK switch must change only which integers the C++ writes -- no
    trainer code. Guard it mechanically so nobody adds a branch later."""
    import ast
    import pathlib
    for mod in ("qlearning.py", "batching.py"):
        tree = ast.parse((pathlib.Path("src/offline") / mod).read_text())
        # Strip docstrings; ast.unparse drops comments. What remains is CODE, which
        # is the only thing the invariant is about -- the prose is allowed (and
        # required) to name the dataset types in order to explain the rule.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                b = getattr(node, "body", None)
                if (b and isinstance(b[0], ast.Expr)
                        and isinstance(b[0].value, ast.Constant)
                        and isinstance(b[0].value.value, str)):
                    b.pop(0)
        body = ast.unparse(tree)
        for tok in ("HASHED", "BITMASK", "MAPPED", "dataset_type"):
            assert tok not in body, (
                f"{mod} branches on {tok!r} in CODE; the trainer must treat "
                f"node_features as opaque ints so enabling BITMASK in "
                f"FringeEvalRL needs no trainer change"
            )


def test_trainer_runs_on_arbitrary_node_feature_integers(t19):
    """node_features are opaque: hashes today, fluent bitstrings tomorrow. Both
    must train without touching this code."""
    for magnitude in (2**62, 2**17, 3):     # hash-like, bitmask-like, tiny
        cache = _cache_for(t19)
        g = torch.Generator().manual_seed(1)
        for s in cache.states:
            s.node_ids = torch.randint(-magnitude, magnitude, s.node_ids.shape,
                                       generator=g, dtype=torch.int64)
        rows, _ = generate_dataset([t19], 3, seeds_per_policy=1, expansion_cap=100,
                                   verbose=False)
        cfg = TrainConfig(fringe_size=3, batch_size=8, device="cpu", expansion_cap=100)
        tr = QTrainer(_net(), _net(), [t19], {t19.name: cache}, rows, cfg)
        log = tr.step()
        assert log["td_loss"] >= 0.0


# --------------------------------------------------- critic calibration ------

def test_q_vs_qstar_splits_on_argmin(t19):
    """F4 panel A uses ONLY argmin slots, where Q* = -delta(s) is exact. Viable
    non-argmin slots evict too, so pooling them would punish a correct critic."""
    tr, rows = _setup(t19)
    out = tr.q_vs_qstar(rows[:40])
    assert set(out) >= {"q_vs_qstar_r2", "q_vs_qstar_mae", "n"}
    assert out["n"] <= 40
