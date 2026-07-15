"""Telemetry: the per-checkpoint record, and what is/isn't a gate."""

from __future__ import annotations

import json

import pytest

from src.offline.env import FringeEnv, coverage_at, coverage_curve, reference_budget
from src.offline.policies import make_policy, oracle_ranking
from src.offline.telemetry import (
    CACTUS_BUDGETS,
    CheckpointRecord,
    TelemetryWriter,
    evaluate_split,
)

from conftest import make_tree


def _pol_for(inst, name="random"):
    return lambda _n: make_policy(inst, name, seed=0)


def test_evaluate_split_reports_every_required_field(t19):
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=3, expansion_cap=200)
    for k in ("coverage_at_reference_budget", "coverage_curve",
              "regret_mean_lower_bound", "regret_median_lower_bound",
              "regret_p90_lower_bound", "expansions_mean", "success_rate",
              "doom_rate", "timeout_rate", "expansions_sterile_frac",
              "beam_sterile_frac", "eviction_events", "eviction_recovery_steps",
              "reservoir_size_mean", "reservoir_size_max", "per_instance"):
        assert k in out, k


def test_doom_rate_is_zero_by_the_completeness_proposition(t19):
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=3, expansion_cap=200)
    assert out["doom_rate"] == 0.0


def test_ranking_diagnostics_only_when_scores_are_supplied(t19):
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=2, expansion_cap=200)
    assert "viability_auc" not in out
    scores = {}

    def score_for(_n, beam):
        # a perfect scorer: -delta, so viability_auc must be 1.0 where defined
        return [-(1e6 if t19.delta[v] == float("inf") else t19.delta[v]) for v in beam]

    out2 = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=2,
                          expansion_cap=200, score_for=score_for)
    for k in ("viability_auc", "top1_oracle_agreement", "spearman_logits_vs_delta",
              "qstar_naive_residual_mean", "qstar_naive_residual_corr_reservoir"):
        assert k in out2, k
    assert out2["viability_auc"] == pytest.approx(1.0)


def test_coverage_is_measured_at_a_difficulty_relative_budget(t19):
    """10 * delta_root, declared once. An absolute budget would give an easy
    instance the same slack as a hard one."""
    assert reference_budget(t19) == 10 * int(t19.delta_root)


def test_coverage_curve_is_monotone_nondecreasing(t19):
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=3, expansion_cap=200)
    cov = [p["coverage"] for p in out["coverage_curve"]]
    assert cov == sorted(cov), "coverage cannot fall as the budget grows"
    assert [p["budget"] for p in out["coverage_curve"]] == list(CACTUS_BUDGETS)


def test_coverage_at_counts_only_solves_within_budget():
    rs = [{"solved": True, "expansions": 10}, {"solved": True, "expansions": 100},
          {"solved": False, "expansions": 5}]
    assert coverage_at(rs, 10) == pytest.approx(1 / 3)
    assert coverage_at(rs, 100) == pytest.approx(2 / 3)


def test_regret_is_recorded_as_a_lower_bound(t19):
    """V* assumes an evicted node returns for free; it does not. The field name
    must carry the caveat so no figure can drop it."""
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=2, expansion_cap=200)
    assert "regret_mean_lower_bound" in out
    assert "regret_mean" not in out, "unqualified 'regret' must not exist in the record"


def test_oracle_beats_random_on_the_recorded_metric(t19):
    """Sanity that the harness measures what it claims."""
    o = evaluate_split([t19], lambda _n: (lambda b: oracle_ranking(t19, b)),
                       fringe_size=19, seeds=3, expansion_cap=200)
    r = evaluate_split([t19], _pol_for(t19, "bfs"), fringe_size=19, seeds=3,
                       expansion_cap=200)
    assert o["regret_mean_lower_bound"] <= r["regret_mean_lower_bound"]
    assert o["expansions_sterile_frac"] == 0.0


def test_writer_roundtrips_jsonl(tmp_path, t19):
    w = TelemetryWriter(tmp_path / "telemetry.jsonl")
    out = evaluate_split([t19], _pol_for(t19), fringe_size=3, seeds=2, expansion_cap=200)
    w.append(CheckpointRecord(step=1, frames=10, split="val", payload=out))
    w.append(CheckpointRecord(step=2, frames=20, split="val", payload=out))
    got = w.read()
    assert len(got) == 2
    assert got[0]["step"] == 1 and got[0]["split"] == "val"
    assert "coverage_at_reference_budget" in got[0]
    # figures must be regenerable from the file alone
    assert json.loads(json.dumps(got)) == got


def test_telemetry_never_branches_on_dataset_type():
    import ast, pathlib
    tree = ast.parse(pathlib.Path("src/offline/telemetry.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = getattr(node, "body", None)
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                b.pop(0)
    body = ast.unparse(tree)
    for tok in ("HASHED", "BITMASK", "MAPPED", "dataset_type"):
        assert tok not in body
