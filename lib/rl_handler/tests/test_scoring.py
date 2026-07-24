"""The shared viability-dominant score: ONE implementation, ONE test.

This bug appeared THREE times in this project (two-head score, 1-WL oracle, and it
would have appeared in the pre-flight table). A sentinel or an untrained value
standing in for delta=inf swamps the distance term and the ranking INVERTS. This
test exists so it cannot happen a fourth time.
"""

from __future__ import annotations

import pytest

from src.offline.scoring import (
    bands_are_disjoint,
    score_from_class_stats,
    viability_dominant_score,
)


@pytest.mark.parametrize("max_delta", [1.0, 5.0, 12.0, 34.0, 64.0, 1000.0])
def test_worst_viable_outranks_best_sterile(max_delta):
    """THE invariant. However wrong d_hat is, a sterile slot must never outrank a
    viable one."""
    assert bands_are_disjoint(max_delta)
    worst_viable = viability_dominant_score(1.0, max_delta, max_delta)
    best_sterile = viability_dominant_score(0.0, 0.0, max_delta)
    assert worst_viable > best_sterile


@pytest.mark.parametrize("max_delta", [5.0, 34.0])
def test_bands_are_disjoint_over_the_whole_range(max_delta):
    viable = [viability_dominant_score(1.0, d, max_delta)
              for d in (0.0, max_delta / 2, max_delta)]
    sterile = [viability_dominant_score(0.0, d, max_delta)
               for d in (0.0, max_delta / 2, max_delta)]
    assert min(viable) > max(sterile)


def test_within_the_viable_band_distance_orders():
    assert (viability_dominant_score(1.0, 2.0, 34.0)
            > viability_dominant_score(1.0, 30.0, 34.0))


def test_the_original_inversion_is_impossible_now():
    """Regression on the exact numbers that inverted: an untrained sterile head
    predicting d_hat ~ 0 alongside a confident viable slot at delta 20."""
    D = 34.0
    sterile_untrained = viability_dominant_score(0.0, 0.0, D)    # d_hat stuck at 0
    viable_far = viability_dominant_score(1.0, 20.0, D)
    assert viable_far > sterile_untrained, (
        "the untrained-sterile-head case is exactly what inverted the old score"
    )


# ------------------------------------------------- bucket (oracle) scoring ----

def test_class_stats_never_averages_a_sentinel_into_the_distance():
    """Failure mode 2: one sterile state in a class of ten must NOT drag the
    class's distance to ~1e5. P(viable) carries sterility; delta never sees it."""
    D = 34.0
    mixed = score_from_class_stats(n_viable=9, n_total=10, mean_delta_viable=3.0, max_delta=D)
    pure = score_from_class_stats(n_viable=10, n_total=10, mean_delta_viable=3.0, max_delta=D)
    assert pure > mixed, "more sterile members must lower the score"
    # ...but a 90%-viable class must still beat an all-sterile one by a mile
    all_sterile = score_from_class_stats(n_viable=0, n_total=10, mean_delta_viable=None, max_delta=D)
    assert mixed > all_sterile
    # and the mixed class must not collapse to the sterile band
    assert mixed > viability_dominant_score(0.0, 0.0, D)


def test_class_stats_monotone_in_viability():
    D = 34.0
    xs = [score_from_class_stats(k, 10, 3.0, D) for k in range(11)]
    assert xs == sorted(xs), "score must increase with P(viable)"


def test_class_stats_monotone_in_distance():
    D = 34.0
    xs = [score_from_class_stats(10, 10, d, D) for d in (1.0, 5.0, 20.0, 34.0)]
    assert xs == sorted(xs, reverse=True), "score must decrease with distance"


def test_all_sterile_class_has_no_distance_to_speak_of():
    D = 34.0
    s = score_from_class_stats(n_viable=0, n_total=10, mean_delta_viable=None, max_delta=D)
    assert s == pytest.approx(viability_dominant_score(0.0, D, D))


def test_empty_bucket_is_worst_possible():
    D = 34.0
    assert score_from_class_stats(0, 0, None, D) == pytest.approx(-D)


def test_distance_is_clamped_to_max_delta():
    """An out-of-range mean (e.g. a held-out instance deeper than anything in
    train) must not push a viable slot into the sterile band."""
    D = 12.0
    s = score_from_class_stats(n_viable=5, n_total=5, mean_delta_viable=999.0, max_delta=D)
    assert s > viability_dominant_score(0.0, 0.0, D)
    assert s == pytest.approx(viability_dominant_score(1.0, D, D))
