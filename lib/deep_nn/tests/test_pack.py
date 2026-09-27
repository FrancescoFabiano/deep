import pytest
import torch

from conftest import MERGED_DOT
from deep_nn.cache import InstanceCache
from deep_nn.dot import parse_dot
from deep_nn.pack import pack_fringe, pack_fringes, pack_goal_tensors


def _cache(n: int, separated: bool = False, tmp_path=None) -> InstanceCache:
    paths = []
    for i in range(n):
        p = tmp_path / f"{i + 1:06d}.dot"
        p.write_text(MERGED_DOT.replace('"7"', f'"{7 + i}"'))
        paths.append(str(p))
    return InstanceCache.from_paths(paths, separated=separated, verbose=False)


def test_pack_fringe_is_fringe_to_tensor_minimal(tmp_path):
    cache = _cache(3, tmp_path=tmp_path)
    n = cache.states[0].n_nodes
    p = pack_fringe(cache, [2, 0], fringe_size=4)
    assert p["node_features"].numel() == 2 * n
    assert p["membership"].tolist() == [0] * n + [1] * n
    # edges and designated worlds shift by the node offset of their state
    assert torch.equal(p["edge_index"][:, 10:], cache.states[0].edge_index + n)
    assert p["pointed_ids"].tolist() == [5, 6, 5 + n, 6 + n]
    assert p["mask"].tolist() == [1, 1, 0, 0] and p["mask"].dtype == torch.uint8
    for k in ("node_features", "edge_index", "edge_attr", "membership", "pointed_ids"):
        assert p[k].dtype == torch.int64


def test_pack_refuses_empty_and_overfull(tmp_path):
    cache = _cache(2, tmp_path=tmp_path)
    with pytest.raises(ValueError):
        pack_fringe(cache, [], 4)
    with pytest.raises(ValueError):
        pack_fringe(cache, [0, 1], 1)


def test_pack_fringes_continues_slots_and_offsets(tmp_path):
    cache = _cache(2, tmp_path=tmp_path)
    a, b = pack_fringe(cache, [0, 1], 2), pack_fringe(cache, [1], 2)
    out = pack_fringes([a, b])
    n = int(a["node_features"].numel())
    assert out["membership"].max().item() == 2 and out["n_slots"] == 3
    assert out["candidate_batch"].tolist() == [0, 0, 1]
    assert torch.equal(out["pointed_ids"][4:], b["pointed_ids"] + n)
    assert out["slot_offset"].tolist() == [0, 2]


def test_separated_cache_derives_one_goal_and_packs_it(tmp_path):
    cache = _cache(2, separated=True, tmp_path=tmp_path)
    assert cache.goal is not None and cache.states[0].pointed_ids.tolist() == [0, 1]
    g = pack_goal_tensors([cache.goal])
    assert g["goal_batch"].tolist() == [0] * 4 and g["goal_node_features"].tolist() == [1, 19, 20, 7]


def test_cache_file_round_trip_and_mode_key(tmp_path):
    paths = [str(tmp_path / "s.dot")]
    (tmp_path / "s.dot").write_text(MERGED_DOT)
    f = tmp_path / "c.pt"
    a = InstanceCache.from_paths(paths, cache_file=f, verbose=False)
    b = InstanceCache.from_paths(paths, cache_file=f, verbose=False)          # hit
    assert torch.equal(a.states[0].pointed_ids, b.states[0].pointed_ids)
    c = InstanceCache.from_paths(paths, separated=True, cache_file=f, verbose=False)  # miss: other mode
    assert c.goal is not None and c.states[0].n_nodes < a.states[0].n_nodes
