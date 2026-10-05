"""Cheap contracts: config validation, the split rule, the flags stage 2 sends, stage 4 on a toy CSV.

    .venv/bin/python -m pytest scripts/pipeline/tests -q
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from pipeline import config, infer, instances, report, train  # noqa: E402

BASIC = config.REPO / "exp" / "trials" / "basic"


def _trial(tmp_path, **overrides):
    """A copy of basic's config with tiny instances; `overrides` patch [section] keys."""
    text = (BASIC / "trial.toml").read_text()
    for key, value in overrides.items():
        text = "\n".join(f"{key} = {value}" if line.split("=")[0].strip() == key else line for line in text.splitlines())
    (tmp_path / "trial.toml").write_text(text)
    inst = tmp_path / "instances"
    (inst / "d1" / "problems").mkdir(parents=True)
    (inst / "act_lib.epddl").write_text("")
    (inst / "d1" / "domain.epddl").write_text("")
    for k in range(10):
        (inst / "d1" / "problems" / f"p-{k:02d}.epddl").write_text("")
    return config.load(tmp_path, dry_run=True)


def test_basic_config_loads():
    cfg = config.load(BASIC, dry_run=True)
    assert cfg.strategies == cfg.data["strategies"]


@pytest.mark.parametrize("key,value", [("train_pct", 0), ("rl_exploitation", 95), ('gnn_searches', '["BFS"]')])
def test_invalid_config_is_refused(tmp_path, key, value):
    with pytest.raises(SystemExit):
        _trial(tmp_path, **{key: value})


def test_split_is_sorted_and_frozen(tmp_path):
    cfg = _trial(tmp_path)
    insts = instances.load(cfg)
    assert [i.split for i in insts] == ["train"] * 7 + ["test"] * 3
    assert [i.problem for i in insts][:2] == ["p-00", "p-01"]
    rows = list(csv.DictReader(cfg.split_file.open()))
    rows[0]["split"] = "test"
    with cfg.split_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert instances.load(cfg)[0].split == "test"      # the file, not the rule, is authoritative


def _accepted(main_py: Path) -> set:
    """Option strings of a trainer's parser. Both handlers own a top-level `src`
    package, so each load starts from a clean module cache and path."""
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(main_py.stem + "_mod", main_py)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(main_py.parent))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(str(main_py.parent))
    parser = m.build_parser() if hasattr(m, "build_parser") else None
    if parser is None:                       # gnn_handler: parse_args builds its parser inline
        import argparse
        real = argparse.ArgumentParser.parse_args
        captured = {}
        argparse.ArgumentParser.parse_args = lambda self, *a, **k: captured.setdefault("p", self) and self
        try:
            m.parse_args([])
        finally:
            argparse.ArgumentParser.parse_args = real
        parser = captured["p"]
    return {o for a in parser._actions for o in a.option_strings}


@pytest.mark.parametrize("kind", ["rl", "gnn"])
def test_trainers_accept_the_flags_stage2_sends(kind):
    sent = {"--train-csv", "--test-csv", "--dir-save-model", "--fringe-sizes", "--epochs", "--batch-size",
            "--seed", "--dataset-type", "--aggregation", *train.FIXED_FLAGS[kind]}
    missing = sent - _accepted(train.TRAINERS[kind])
    assert not missing, f"{kind} trainer rejects {missing}"


def test_act_lib_per_domain_overrides_the_shared_one(tmp_path):
    cfg = _trial(tmp_path)
    (cfg.instances_dir / "d2" / "problems").mkdir(parents=True)
    (cfg.instances_dir / "d2" / "domain.epddl").write_text("")
    (cfg.instances_dir / "d2" / "problems" / "q-00.epddl").write_text("")
    (cfg.instances_dir / "d2" / "act_lib.epddl").write_text("")
    libs = {i.domain: i.act_lib for i in instances.load(cfg)}
    assert libs["d1"] == cfg.act_lib and libs["d2"] == cfg.instances_dir / "d2" / "act_lib.epddl"
    cfg.act_lib.unlink()                                  # d1 now has no library at all
    with pytest.raises(SystemExit):
        config.load(tmp_path, dry_run=True)


def test_pooled_trains_once_and_refuses_repeated_problem_names(tmp_path, capsys):
    _trial(tmp_path)
    toml = tmp_path / "trial.toml"
    toml.write_text(toml.read_text().replace("[train]\n", "[train]\npooled = true\n"))
    cfg = config.load(tmp_path, dry_run=True)
    assert cfg.pooled and cfg.model_dir("d1") == cfg.models_dir / config.POOLED_DIR / "dense"
    (cfg.instances_dir / "d2" / "problems").mkdir(parents=True)
    (cfg.instances_dir / "d2" / "domain.epddl").write_text("")
    (cfg.instances_dir / "d2" / "problems" / "p-00.epddl").write_text("")   # same name as in d1
    cfg.split_file.unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="p-00"):
        train.run(cfg)
    (cfg.instances_dir / "d2" / "problems" / "p-00.epddl").rename(cfg.instances_dir / "d2" / "problems" / "q-00.epddl")
    cfg.split_file.unlink(missing_ok=True)
    for dom, prob in [("d1", "p-00"), ("d2", "q-00")]:
        d = cfg.data_dir / dom / "BFS" / prob
        d.mkdir(parents=True)
        (d / f"{prob}_BFS_depth_25.csv").write_text("")
    train.run(cfg)
    out = capsys.readouterr().out
    assert f"[train] {config.POOLED_DIR}/rl@F4: 2 train tables" in out and "[train] d1/" not in out


def test_methods_follow_the_installed_models(tmp_path):
    cfg = _trial(tmp_path)
    d = cfg.model_dir("d1")
    d.mkdir(parents=True)
    for name in ["rl_F4.onnx", "gnn_F1.onnx", "gnn_F8.onnx", "gnn_state.onnx", "gnn_state_C.txt"]:
        (d / name).write_text("")
    got = [(m.name, m.F) for m in infer.methods(cfg, "d1")]
    assert got == [("BFS", 0), ("RL", 4), ("GNN_RL", 1), ("GNN_RL", 8), ("GNN_Astar", 0)]


def test_report_on_toy_results(tmp_path):
    cfg = _trial(tmp_path)
    cfg.results_file.parent.mkdir(parents=True)
    rows = []
    for k in range(4):
        split = "train" if k < 3 else "test"
        rows.append(dict(domain="d1", split=split, problem=f"p-{k:02d}", method="BFS", F=0, status="SOLVED",
                         plan_length=3, nodes_expanded=100 * (k + 1), init_ms=1, search_ms=5, total_ms=6, wall_s=0.1))
        for F in (4, 8):
            rows.append(dict(domain="d1", split=split, problem=f"p-{k:02d}", method="RL", F=F,
                             status="SOLVED" if k < 3 else "TIMEOUT", plan_length=3, nodes_expanded=50 * (k + 1),
                             init_ms=1, search_ms=5, total_ms=6, wall_s=0.1))
    with cfg.results_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=infer.COLUMNS); w.writeheader(); w.writerows(rows)
    report.run(cfg)
    tex = (cfg.report_dir / "tables" / "nodes_common_train.tex").read_text()
    assert r"\begin{tabular}" in tex and "(+50\\%)" in tex
    solved = (cfg.report_dir / "tables" / "nodes_solved_test.csv").read_text().splitlines()[1]
    assert solved == "d1,400 (1),-- (0),-- (0)"        # own solved set: BFS solved p-03, RL nothing
    assert (cfg.report_dir / "tables" / "coverage_test.csv").read_text().splitlines()[1] == "d1,1/1,0/1,0/1"
    for name in ("coverage_vs_F", "nodes_common_vs_F", "nodes_solved_vs_F", "nodes_vs_bfs"):
        assert (cfg.report_dir / "figures" / f"d1_{name}.png").exists()


def test_a_failed_run_does_not_block_the_others(tmp_path, monkeypatch):
    _trial(tmp_path)
    cfg = config.load(tmp_path, dry_run=False)
    d = cfg.data_dir / "d1" / "BFS" / "p-00"
    d.mkdir(parents=True)
    (d / "p-00_BFS_depth_25.csv").write_text("")
    calls, exports = [], []

    def fake_run(cmd, **kw):
        if "--export-from" in cmd or "--export-only" in cmd:   # the other ONNX form: same files, rewritten
            exports.append(cmd[cmd.index("--aggregation") + 1])
            return
        F = int(cmd[cmd.index("--fringe-sizes") + 1])
        calls.append(F)
        if F == 4 and "offline_main" in cmd[1]:                # the RL F=4 run fails
            raise train.subprocess.CalledProcessError(1, cmd)
        run = Path(cmd[cmd.index("--dir-save-model") + 1])
        if "offline_main" in cmd[1]:
            (run.parent / f"run_fringe{F}").mkdir(parents=True, exist_ok=True)
            (run.parent / f"run_fringe{F}" / train.EXPORTS["rl"][0][0].format(F=F)).write_text("")
        else:                                              # the GNN: one run, every width exported
            run.mkdir(parents=True, exist_ok=True)
            for f in train.GNN_FRINGE_SIZES:
                (run / train.EXPORTS["gnn"][0][0].format(F=f)).write_text("")
            for src, _ in train.GNN_STATE:
                (run / src).write_text("")
            (run / "distance_estimator.pt").write_text("")
    monkeypatch.setattr(train.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="d1/rl@F4"):
        train.run(cfg)
    assert calls == [*cfg.train["fringe_sizes"], 1]        # every RL F tried, then the one GNN run
    out = cfg.models_dir / "d1" / "dense"                  # [train].aggregation default
    assert (out / "rl_F8.onnx").exists() and (out / "gnn_state.onnx").exists()
    assert all((out / f"gnn_F{f}.onnx").exists() for f in train.GNN_FRINGE_SIZES)
    assert not (out / "rl_F4.onnx").exists()
    # every other ONNX form of what was trained is exported from the same weights, next to the first
    sc = config.load(tmp_path, dry_run=False, aggregation="scatter")
    assert sc.model_dir("d1") == cfg.models_dir / "d1" / "scattered"
    assert exports == ["scatter"] * 3                      # RL F=8, F=16 and the GNN; nothing for the failed F=4
    assert (sc.model_dir("d1") / "rl_F8.onnx").exists() and (sc.model_dir("d1") / "gnn_state.onnx").exists()
    assert not (sc.model_dir("d1") / "rl_F4.onnx").exists()
    # asked for the other form: never a retrain of what is trained, and nothing left to export
    calls.clear(), exports.clear()
    with pytest.raises(SystemExit, match="d1/rl@F4"):
        train.run(sc)
    assert calls == [4] and exports == []                  # only the run that never finished is retried
    train.reexport(sc)                                     # `trial.py export`: rewrites the form it is asked for
    assert exports == ["scatter"] * 3
