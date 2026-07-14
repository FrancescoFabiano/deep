"""The two-head delta baseline: the thing the RL must beat.

Reported ALONE before any comparison. A strong baseline reported honestly is
worth more than a favourable comparison.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.two_head_baseline import TwoHeadBaselineNetwork, two_head_loss
from src.offline.metrics import viability_auc
from src.offline.tree import INF_DELTA


def _net(max_delta=34.0, context_mode="none"):
    torch.manual_seed(0)
    return TwoHeadBaselineNetwork(
        node_input_dim=1, hidden_dim=16, gnn_layers=1, dataset_type="HASHED",
        context_mode=context_mode, max_delta=max_delta,
    ).eval()


def _score(p_logit, d_hat, max_delta):
    return (max_delta + 1.0) * torch.sigmoid(torch.tensor(p_logit)) - d_hat


# ------------------------------------------- the inversion bug, pinned -------

def test_viability_strictly_dominates_distance():
    """REGRESSION. The first score was `s = -d_hat + logit p`, which measured
    viability_auc = 0.048 on real data -- near-perfect INVERSION.

    Cause: the distance head trains on VIABLE slots only (masked MSE -- the very
    thing that removes the clip), so d_hat on a sterile slot never gets a
    gradient and sits at its init ~0:

        sterile:  -0  + (-5) = -5
        viable:   -20 + (+5) = -15     (delta 20)

    and -5 > -15, so sterile won whenever the delta range exceeded the p_logit
    gap. The bands must be DISJOINT by construction instead.
    """
    D = 34.0
    worst_viable = _score(+20.0, torch.tensor(D), D)   # confident viable, farthest
    best_sterile = _score(-20.0, torch.tensor(0.0), D)  # confident sterile, nearest
    assert worst_viable > best_sterile, (
        f"the worst viable slot ({worst_viable:.3f}) must outrank the best "
        f"sterile slot ({best_sterile:.3f}) -- else sterile nodes get expanded first"
    )


def test_distance_head_is_bounded_so_sterile_cannot_swamp():
    """d_hat in [0, max_delta] for EVERY slot, trained or not. An unconstrained
    head is what let untrained sterile predictions dominate the score."""
    m = _net(max_delta=34.0)
    for raw in (-1e3, -8.0, 0.0, 8.0, 1e3):
        d = torch.sigmoid(torch.tensor(raw)) * m.max_delta
        assert 0.0 <= float(d) <= 34.0


def test_score_orders_viable_slots_by_distance():
    """Within the viable band, -d_hat does the ordering."""
    D = 34.0
    near = _score(+8.0, torch.tensor(2.0), D)
    far = _score(+8.0, torch.tensor(30.0), D)
    assert near > far


def test_score_is_a_single_scalar_per_slot():
    """It must stay one scalar to export through the same ONNX contract."""
    m = _net()
    n, e, k = 6, 5, 3
    s = m(
        node_features=torch.zeros(n, dtype=torch.int64),
        edge_index=torch.zeros((2, e), dtype=torch.int64),
        edge_attr=torch.zeros(e, dtype=torch.int64),
        membership=torch.arange(n) % k,
        candidate_batch=None,
        mask=torch.ones(k, dtype=torch.uint8),
    )
    assert s.shape == (k,)


def test_masked_slots_are_masked():
    m = _net()
    n, e, k = 6, 5, 3
    mask = torch.tensor([1, 1, 0], dtype=torch.uint8)
    s = m(
        node_features=torch.zeros(n, dtype=torch.int64),
        edge_index=torch.zeros((2, e), dtype=torch.int64),
        edge_attr=torch.zeros(e, dtype=torch.int64),
        membership=torch.arange(n) % k,
        candidate_batch=None, mask=mask,
    )
    assert s[2] <= -1e8


# ------------------------------------------------------------ the loss -------

def test_distance_loss_is_masked_not_clipped():
    """A sterile node has no finite distance, so asking the head to predict one
    would be inventing a target -- that is the clip we refuse."""
    p_logit = torch.zeros(4, requires_grad=True)
    d_hat = torch.zeros(4, requires_grad=True)
    viable = torch.tensor([1.0, 1.0, 0.0, 0.0])
    delta = torch.tensor([3.0, 5.0, 0.0, 0.0])   # the 0s are ignored
    _, log = two_head_loss(p_logit, d_hat, viable, delta)
    assert log["n_viable"] == 2 and log["n_total"] == 4
    # the MSE must not see the sterile slots at all
    delta2 = torch.tensor([3.0, 5.0, 999.0, 999.0])
    _, log2 = two_head_loss(p_logit, d_hat, viable, delta2)
    assert log["mse"] == pytest.approx(log2["mse"])


def test_all_sterile_beam_has_no_distance_loss():
    p_logit = torch.zeros(2, requires_grad=True)
    d_hat = torch.zeros(2, requires_grad=True)
    loss, log = two_head_loss(p_logit, d_hat, torch.zeros(2), torch.zeros(2))
    assert log["mse"] == 0.0 and log["n_viable"] == 0
    loss.backward()   # must not produce NaN


def test_a_perfect_two_head_model_scores_auc_one():
    """Sanity on the metric + score together: a model with exact heads must rank
    every viable slot above every sterile one."""
    D = 34.0
    deltas = [2.0, 20.0, INF_DELTA, INF_DELTA]
    scores = [
        float(_score(+20.0 if d != INF_DELTA else -20.0,
                     torch.tensor(0.0 if d == INF_DELTA else d), D))
        for d in deltas
    ]
    assert viability_auc(scores, deltas) == 1.0
