"""The ONE place viability is combined with distance.

KNOWN FAILURE MODE FOR THIS PROJECT: a sentinel or an untrained value standing in
for `delta = inf` swamps the distance term, and the ranking INVERTS. It has now
appeared THREE times:

  1. two_head_baseline score `-d_hat + logit p`. The distance head trains on
     viable slots only (masked MSE), so d_hat on a sterile slot never receives a
     gradient and sits at ~0:
         sterile: -0  + (-5) = -5      viable: -20 + (+5) = -15
     -5 > -15, so sterile ranked FIRST. Measured viability_auc = 0.048 on real
     data -- near-perfect inversion.
  2. The 1-WL oracle used `1e6` as the sterile stand-in inside a class MEAN, so one
     sterile state in a class of ten produced a mean of ~90909 and every mixed
     class was ranked last wholesale. Visible: CC_2_3_4__pl_7 scored 79.0
     in-sample vs 6.0 under a sibling-trained transfer oracle -- an in-sample
     oracle cannot legitimately lose to a transfer oracle.
  3. It would have appeared again in the standing pre-flight table.

So there is exactly one implementation, with one test. Do not inline this
arithmetic anywhere else.

THE FORM

    score(v) = (max_delta + 1) * p_viable(v) - d_hat(v),   d_hat in [0, max_delta]

    viable  (p ~ 1): score in [1, max_delta + 1]
    sterile (p ~ 0): score in [-max_delta, 0]

The bands are DISJOINT by construction, so viability strictly dominates distance
and no sterile prediction can outrank a viable one however wrong d_hat is. Within
the viable band, -d_hat does the ordering. Higher score = expanded sooner, matching
the C++ contract.

`max_delta` is DATA-DERIVED (the largest finite delta in the training instances),
not tuned: it is a range constraint on a quantity whose range we know, NOT a
sentinel for `inf`, which is the thing we refuse to invent.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def viability_dominant_score(p_viable, d_hat, max_delta: float):
    """(max_delta + 1) * p_viable - d_hat. Works for floats or torch tensors.

    `p_viable` in [0,1]; `d_hat` MUST already be bounded to [0, max_delta] (the
    caller squashes it -- an unbounded d_hat re-opens the inversion).
    """
    return (float(max_delta) + 1.0) * p_viable - d_hat


def score_from_class_stats(
    n_viable: int,
    n_total: int,
    mean_delta_viable: float | None,
    max_delta: float,
) -> float:
    """Score a bucket of states (a 1-WL class, or a kNN neighbourhood).

    This is the Bayes-optimal-shaped predictor for RANKING given only bucket
    membership: P(viable | bucket) and E[delta | bucket, viable]. It deliberately
    does NOT average `delta` with a sentinel for the sterile members -- that is
    failure mode 2 above.
    """
    if n_total <= 0:
        return viability_dominant_score(0.0, max_delta, max_delta)
    p = n_viable / n_total
    d = max_delta if (mean_delta_viable is None or n_viable == 0) else min(
        float(mean_delta_viable), max_delta
    )
    return viability_dominant_score(p, d, max_delta)


def bands_are_disjoint(max_delta: float) -> bool:
    """The invariant the form exists to guarantee: the WORST viable slot must
    outrank the BEST sterile slot."""
    worst_viable = viability_dominant_score(1.0, max_delta, max_delta)
    best_sterile = viability_dominant_score(0.0, 0.0, max_delta)
    return worst_viable > best_sterile
