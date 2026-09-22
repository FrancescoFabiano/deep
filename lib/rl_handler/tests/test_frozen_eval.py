"""frozen_eval.py: M random-policy fringes per instance, drawn once, reproducible,
readable by the existing ranking metrics."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from conftest import T19_CHILDREN, T19_GOALS, make_tree
from src.offline.dataset import generate_dataset
from src.offline.frozen_eval import (
    FrozenFringe,
    build_frozen_fringes,
    collect_random_fringes,
    is_contrastive,
    load_frozen,
    save_frozen,
    training_overlap,
)
from src.offline.selection import heldout_ranking_metrics, heldout_ranking_micro_macro
from src.offline.tree import INF_DELTA


def _t19(name="t19"):
    return make_tree(T19_CHILDREN, T19_GOALS, name=name)


def test_contrastive_filter():
    t = _t19()
    assert is_contrastive(t, [1, 2])          # delta 3 vs inf
    assert not is_contrastive(t, [2, 5])      # both dead
    assert not is_contrastive(t, [9, 13])     # both delta 1: nothing to rank
    assert not is_contrastive(t, [1])         # singleton


def test_collect_yields_unique_contrastive_beams_only():
    t = _t19()
    fr, info = collect_random_fringes(t, fringe_size=4, seeds=[0, 1, 2],
                                      expansion_cap=100)
    assert fr and info["n_decisions"] >= len(fr)
    keys = {f.key() for f in fr}
    assert len(keys) == len(fr)
    for f in fr:
        assert is_contrastive(t, f.obs) and not f.forced and f.instance == t.name


def test_build_is_reproducible_and_capped_at_m():
    t = _t19()
    a, ma = build_frozen_fringes([t], 4, seed=7, expansion_cap=100, m_per_instance=3,
                                 rollouts_per_instance=4, verbose=False)
    b, mb = build_frozen_fringes([t], 4, seed=7, expansion_cap=100, m_per_instance=3,
                                 rollouts_per_instance=4, verbose=False)
    assert [f.key() for f in a] == [f.key() for f in b]
    assert len(a) <= 3 and ma["n_fringes"] == len(a)
    assert ma["per_instance"][t.name]["n_selected"] == len(a)
    c, _ = build_frozen_fringes([t], 4, seed=8, expansion_cap=100, m_per_instance=3,
                                rollouts_per_instance=4, verbose=False)
    assert [f.key() for f in a] != [f.key() for f in c] or len(a) == 0


def test_shortfall_is_recorded_when_fewer_than_m_exist():
    t = _t19()
    fr, m = build_frozen_fringes([t], 4, seed=1, expansion_cap=100, m_per_instance=10_000,
                                 rollouts_per_instance=2, verbose=False)
    p = m["per_instance"][t.name]
    assert p["shortfall"] == 10_000 - p["n_candidates"] and p["n_selected"] == p["n_candidates"]


def test_round_trip_and_metrics_consume_frozen_fringes(tmp_path):
    t = _t19()
    fr, m = build_frozen_fringes([t], 4, seed=3, expansion_cap=100, m_per_instance=8,
                                 rollouts_per_instance=4, verbose=False)
    p = save_frozen(tmp_path / "frozen_eval.json", fr, m)
    fr2, m2 = load_frozen(p)
    assert [f.key() for f in fr2] == [f.key() for f in fr] and m2["seed"] == 3
    by_name = {t.name: t}
    oracle = lambda name, beam: [-by_name[name].delta[v] if by_name[name].delta[v] < INF_DELTA
                                 else -1e9 for v in beam]
    rm = heldout_ranking_metrics(fr2, by_name, logits_for=oracle)
    assert rm["n"] == len(fr2) and rm["top1"] == 1.0 and rm["ndcg"] == 1.0
    mm = heldout_ranking_micro_macro(fr2, by_name, logits_for=oracle)
    assert mm["micro"]["n"] == len(fr2)


def test_training_overlap_is_measured_not_hidden():
    t = _t19()
    rows, _ = generate_dataset([t], 4, policies=("bfs",), seeds_per_policy=1,
                               expansion_cap=100, verbose=False)
    fr = [FrozenFringe(t.name, list(rows[0].obs), 0, 0),
          FrozenFringe(t.name, list(reversed(rows[0].obs)), 0, 1),
          FrozenFringe(t.name, [17, 18], 0, 2)]
    ov = training_overlap(fr, rows)
    assert ov["n_fringes"] == 3
    assert ov["exact_beam_match_frac"] == pytest.approx(1 / 3) or len(rows[0].obs) == 1
    assert ov["beam_as_set_match_frac"] >= ov["exact_beam_match_frac"]
    assert 0.0 <= ov["state_seen_in_training_frac"] <= 1.0
