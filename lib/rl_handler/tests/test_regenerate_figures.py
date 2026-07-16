"""The figure-regeneration + auto-backfill entrypoint. Manual /tmp backfill must not
be load-bearing: a crashed run (complete telemetry, no figures) and a stale-code run
(telemetry missing a metric) must both recover automatically. GPU-free unit coverage;
the full self-validated backfill is exercised end-to-end over batch3 separately."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "regen", REPO / "scripts" / "rl_exp" / "regenerate_figures.py")
regen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regen)


def _write_tel(path, val_extra):
    """A telemetry.jsonl with one baseline row + one val row carrying val_extra."""
    rows = [
        {"step": -1, "frames": 0, "split": "baseline:bfs",
         "coverage_at_reference_budget": 0.9, "regret_mean_lower_bound": 40.0},
        {"step": 5000, "frames": 320000, "split": "val",
         "coverage_at_reference_budget": 1.0, "regret_mean_lower_bound": 1.0, **val_extra},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


FULL = {"heldout_top1": 0.66, "heldout_ndcg": 0.88, "heldout_js": 0.4,
        "return_mean": -12.0, "train_ndcg": 0.90, "heldout_n": 200}


def test_missing_fields_detects_a_stale_run(tmp_path):
    complete = tmp_path / "a.jsonl"; _write_tel(complete, FULL)
    assert regen._missing_fields(complete) == set()
    stale = tmp_path / "b.jsonl"; _write_tel(stale, {"heldout_top1": 0.66})   # top1 only
    miss = regen._missing_fields(stale)
    assert "heldout_ndcg" in miss and "return_mean" in miss and "heldout_top1" not in miss


def test_fringe_and_seed_parsed_from_dirname(tmp_path):
    d = tmp_path / "seed42_fringe16"
    assert regen._fringe_of(d) == 16 and regen._seed_of(d) == 42


def test_a_stale_run_with_no_checkpoints_skips_cleanly_never_blank(tmp_path, capsys, monkeypatch):
    """Missing metric AND no checkpoints -> backfill returns SKIP, and we do NOT emit
    the missing-metric figures (a blank/mislabeled figure is worse than none)."""
    fd = tmp_path / "seed42_fringe4"; fd.mkdir()
    _write_tel(fd / "telemetry.jsonl", {"heldout_top1": 0.66})     # stale, no checkpoints/ dir
    monkeypatch.setattr(regen.subprocess, "run", lambda *a, **k: _Ok())
    regen.process(fd)
    out = capsys.readouterr().out
    assert "backfilling" in out and "SKIP" in out
    assert "NOT plotting" in out, "must refuse to emit blank missing-metric figures"


def test_a_complete_run_plots_without_backfill(tmp_path, capsys, monkeypatch):
    fd = tmp_path / "seed42_fringe8"; fd.mkdir()
    _write_tel(fd / "telemetry.jsonl", FULL)
    calls = []
    monkeypatch.setattr(regen.subprocess, "run", lambda *a, **k: calls.append(a) or _Ok())
    regen.process(fd)
    out = capsys.readouterr().out
    assert "backfilling" not in out and "figures OK" in out
    assert calls, "the plot subprocess must fire for a complete run"


def test_sweep_walk_processes_every_fringe_dir(tmp_path, capsys, monkeypatch):
    models = tmp_path / "_models" / "CC"
    for name in ("seed42_fringe4", "seed42_fringe32"):
        d = models / name; d.mkdir(parents=True); _write_tel(d / "telemetry.jsonl", FULL)
    (models / "cache").mkdir()                    # a non-fringe sibling must be ignored
    monkeypatch.setattr(regen.subprocess, "run", lambda *a, **k: _Ok())
    monkeypatch.setattr(regen.sys, "argv", ["regen", str(tmp_path)])
    regen.main()
    out = capsys.readouterr().out
    assert "2 run(s)" in out
    assert "seed42_fringe4" in out and "seed42_fringe32" in out


class _Ok:
    returncode = 0
    stdout = "[plot] wrote figures"
    stderr = ""
