"""Rows -> indices -> batches on a synthetic flat batch, both modes, plus the
trainer-facing guards (strategy layout, two tables)."""

from __future__ import annotations

import os

import pytest
import torch

from conftest import GOAL_DOT, STATE_DOT
from src.preprocessing import GraphDataPipeline, generation_depth_of
from src.utils import IndexBatcher, normalization_params, prepare_targets, select_model

HEADER = "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action\n"


def _instance(root, name, n_states=12, separated=False, token="S_DFS", depth=25):
    inst = root / name
    raw = inst / "RawFiles" / ("hash_separated" if separated else "hash_merged")
    raw.mkdir(parents=True)
    goal = inst / "goal_tree.dot"
    goal.write_text(GOAL_DOT)
    rows = []
    for i in range(n_states):
        p = raw / f"{i + 1:06d}.dot"
        # vary the graph a little so states differ
        p.write_text(STATE_DOT.replace('"8"', f'"{(i % 3) + 1}"'))
        dist = "0001000000" if i == n_states - 1 else f"{i % 4:010d}"   # one unreachable
        rows.append(f"{p},{i % 3},{dist},{goal},{raw / 'init.dot'},0\n")
    (inst / f"{name}_{token}_depth_{depth}.csv").write_text(HEADER + "".join(rows))
    return inst


def _pipe(root, kind, use_goal, monkeypatch):
    monkeypatch.chdir(root.parent)  # paths in the table are absolute here anyway
    return GraphDataPipeline(folder_data=str(root), list_subset_train=[], dataset_type="HASHED",
                             kind_of_data=kind, unreachable_state_value=1000000,
                             max_percentage_per_class=0.5, test_size=0.2, use_goal=use_goal,
                             use_depth=False, random_state=0)


@pytest.mark.parametrize("kind,sep", [("merged", False), ("separated", True)])
def test_pipeline_builds_indices_and_batches(tmp_path, monkeypatch, kind, sep):
    root = tmp_path / "training_data"
    _instance(root, "CC_2_2_3__pl_4", separated=sep)
    _instance(root, "CC_2_2_3__pl_6", separated=sep, token="BFS", depth=20)
    pipe = _pipe(root, kind, use_goal=sep, monkeypatch=monkeypatch)
    store = pipe.build_store(cache_dir=tmp_path / "cache", verbose=False)
    assert pipe.generation_depths == {"CC_2_2_3__pl_4": 25, "CC_2_2_3__pl_6": 20}
    assert len(os.listdir(tmp_path / "cache")) == 2
    n_rows = len(pipe.train_df) + len(pipe.test_df)
    assert n_rows == pipe.train_index["state"].numel() + pipe.test_index["state"].numel()
    assert ("goal" in pipe.train_index) == sep
    # unreachable rows were dropped by balancing already (remove_unreachable default)
    assert (pipe.train_index["target"] == 1000000).sum() == 0
    tr, te, params = prepare_targets(pipe.train_index, pipe.test_index, 1000000, max_depth=25)
    assert params == normalization_params(25)
    assert tr["target"].max() <= 1.0 and tr["target"].min() >= params["intercept"]
    batches = list(IndexBatcher(store, tr, batch_size=4, shuffle=True, seed=0))
    assert sum(b["target"].numel() for b in batches) == tr["state"].numel()
    b = batches[0]
    assert b["state_graph"].num_graphs == b["target"].numel()
    assert (b["goal_graph"] is not None) == sep
    # the model consumes the packed batch directly
    m = select_model("distance_estimator", use_goal=sep, use_depth=False)
    m.model.eval()
    with torch.no_grad():
        out = m.model(m._move_batch_to_device(b))
    assert out.shape == (b["target"].numel(),)


def test_strategy_layout_is_refused_with_an_explanation(tmp_path, monkeypatch):
    root = tmp_path / "training_data"
    _instance(root / "BFS", "CC_2_2_3__pl_4")
    with pytest.raises(ValueError, match="per-strategy layout"):
        _pipe(root, "merged", False, monkeypatch)


def test_two_tables_in_one_instance_is_an_error(tmp_path, monkeypatch):
    root = tmp_path / "training_data"
    inst = _instance(root, "CC_2_2_3__pl_4")
    (inst / "CC_2_2_3__pl_4_BFS_depth_25.csv").write_text(HEADER)
    with pytest.raises(ValueError, match="generation tables"):
        _pipe(root, "merged", False, monkeypatch)


def test_separated_requires_goal_and_merged_forbids_it(tmp_path, monkeypatch):
    root = tmp_path / "training_data"
    _instance(root, "CC_2_2_3__pl_4", separated=True)
    with pytest.raises(ValueError, match="requires use_goal"):
        _pipe(root, "separated", False, monkeypatch)


def test_generation_depth_from_name():
    assert generation_depth_of("CC_2_2_3__pl_4_HFS_SUBGOALS__depth_25.csv") == 25
    assert generation_depth_of("CC_2_2_3__pl_4_depth_40.csv") == 40
    assert generation_depth_of("whatever.csv") is None


def test_normalisation_is_invertible_the_planner_way():
    p = normalization_params(25)
    for d in (0, 4, 25):
        v = d * p["slope"] + p["intercept"]
        assert round((v - p["intercept"]) / p["slope"]) == d
