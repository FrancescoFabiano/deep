"""Beams from a generation table: targets, label mask, batching, end-to-end train."""

from pathlib import Path

import torch

from conftest import write_instance
from src.beams import build_datasets
from src.utils import DistanceEstimatorModel, MIN_V_NN


def _datasets(tmp_path, kind="merged", F=4):
    train = write_instance(tmp_path / "train", "A_1", n_states=8)
    test = write_instance(tmp_path / "test", "A_2", n_states=6)
    return build_datasets([train], [test], kind, F, 1e6, tmp_path / "cache", Path("/"),
                          seeds_per_policy=1)


def test_beams_carry_scaled_targets_and_a_label_mask(tmp_path):
    train, test, params = _datasets(tmp_path)
    assert params["slope"] == (1 - 2 * MIN_V_NN) / 10 and len(train) > 0 and len(test) > 0
    batch = train.collate([train[i] for i in range(min(3, len(train)))])
    assert batch["target"].numel() == batch["label_mask"].numel() == int(batch["membership"].max()) + 1
    assert batch["candidate_batch"].numel() == batch["target"].numel()
    assert batch["pointed_ids"].numel() >= batch["target"].numel()   # one designated world per state
    labelled = batch["target"][batch["label_mask"]]
    assert ((labelled > 0) & (labelled < 1)).all()
    # the dead-end state (distance 1e6) is packed but not a training target
    dead = [i for i, (_, obs) in enumerate(train.beams) if 7 in obs]
    assert dead and not train[dead[0]]["label_mask"].all()


def test_separated_beams_carry_the_derived_goal(tmp_path):
    train, _, _ = _datasets(tmp_path, kind="separated")
    batch = train.collate([train[0], train[1]])
    assert batch["goal_batch"].tolist() == [0] * 3 + [1] * 3
    assert batch["goal_node_features"][:3].tolist() == [1, 19, 7]
    feed = train.planner_feed(0)
    assert feed["goal_batch"].tolist() == [0, 0, 0] and feed["mask"].numel() == 4


def test_train_evaluate_export_round_trip(tmp_path):
    train, test, params = _datasets(tmp_path)
    m = DistanceEstimatorModel(hidden_dim=8, node_emb_dim=4, edge_emb_dim=4,
                               regressor_hidden_dim=8, regressor_blocks=1, device="cpu")
    m.train(train.loader(4, shuffle=True, seed=0), test.loader(4, shuffle=False),
            n_epochs=2, checkpoint_dir=str(tmp_path / "m"), model_name="d")
    m.load_model(tmp_path / "m" / "d.pt")
    metrics = m.evaluate(test.loader(4, shuffle=False))
    assert {"val_loss", "mse", "spearman"} <= metrics.keys()
    path = m.to_onnx(tmp_path / "m" / "d_4.onnx", 4, params)
    check = m.verify_onnx(path, [train.planner_feed(i) for i in range(min(4, len(train)))], params)
    assert check["max_abs_diff"] < 1e-5
