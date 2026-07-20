"""Fix 3: the single-pass micro/macro report. MICRO must be byte-identical to
`heldout_ranking_metrics` (the selector is unchanged); macro + effective-instance-count
are new recording-only fields."""
from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from src.offline.selection import (
    heldout_ranking_metrics,
    heldout_ranking_micro_macro,
)


class _Row:
    def __init__(self, instance, obs):
        self.instance = instance
        self.obs = obs
        self.forced = False


class _Inst:
    def __init__(self, delta):
        self.delta = delta


# two instances; instance B has a singleton frontier that the metric filter drops
_BY_NAME = {
    "A": _Inst([0.0, 1.0, 2.0, 3.0, 5.0]),
    "B": _Inst([0.0, 2.0, 4.0]),
}
_ROWS = [
    _Row("A", [1, 2, 3]),
    _Row("A", [0, 2, 4]),
    _Row("A", [1, 2, 3]),   # duplicate -> deduped
    _Row("B", [0, 1, 2]),
    _Row("B", [0]),         # len<2 -> skipped, so B still scorable via the row above
]


def _logits(name, beam):
    # deterministic, goal-free stand-in: score = -delta (a decent ranker)
    return [-_BY_NAME[name].delta[v] for v in beam]


def test_micro_equals_heldout_ranking_metrics_exactly():
    ref = heldout_ranking_metrics(_ROWS, _BY_NAME, logits_for=_logits)
    mm = heldout_ranking_micro_macro(_ROWS, _BY_NAME, logits_for=_logits)
    micro = mm["micro"]
    assert micro["n"] == ref["n"]
    for k in ("top1", "regret_at_decision", "picked_dead", "ndcg", "js", "kendall_tau"):
        assert micro[k] == ref[k], f"{k}: micro {micro[k]} != ref {ref[k]}"


def test_macro_and_effective_instance_count():
    mm = heldout_ranking_micro_macro(_ROWS, _BY_NAME, logits_for=_logits)
    # both A and B contribute >=1 scorable frontier
    assert set(mm["per_instance"]) == {"A", "B"}
    assert mm["effective_instance_count"] == 2
    # macro is the unweighted mean of per-instance NDCGs
    ndcgs = [mm["per_instance"][i]["ndcg"] for i in ("A", "B")]
    assert abs(mm["ndcg_macro"] - sum(ndcgs) / 2) < 1e-12


def test_agreement_metrics_top1_and_taub():
    # model ranks by -delta; oracle ranker ranks by delta (identical order) -> perfect
    # agreement; a reversed ranker is anti-correlated.
    rankers = {
        "oracle":  lambda name, beam: sorted(range(len(beam)), key=lambda k: _BY_NAME[name].delta[beam[k]]),
        "reverse": lambda name, beam: sorted(range(len(beam)), key=lambda k: -_BY_NAME[name].delta[beam[k]]),
    }
    mm = heldout_ranking_micro_macro(_ROWS, _BY_NAME, logits_for=_logits, agreement_rankers=rankers)
    ag = mm["agreement"]
    assert ag["oracle"]["top1"] == 1.0 and ag["oracle"]["taub"] > 0.99
    assert ag["reverse"]["top1"] == 0.0 and ag["reverse"]["taub"] < -0.99


def test_agreement_absent_when_no_rankers_given():
    mm = heldout_ranking_micro_macro(_ROWS, _BY_NAME, logits_for=_logits)
    assert "agreement" not in mm


def test_macro_differs_from_micro_when_instances_are_imbalanced():
    # A has 2 frontiers, B has 1 -> micro (frontier-weighted) != macro (instance-weighted)
    mm = heldout_ranking_micro_macro(_ROWS, _BY_NAME, logits_for=_logits)
    assert mm["micro"]["n"] == 3          # 2 from A + 1 from B (dup + singleton dropped)
    # not asserting inequality of values (they can coincide); assert the counts differ
    assert mm["per_instance"]["A"]["n"] == 2 and mm["per_instance"]["B"]["n"] == 1
