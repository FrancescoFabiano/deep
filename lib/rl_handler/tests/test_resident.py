"""ResidentTrees.pack must be byte-identical to batching.pack_batch, in merged and
separated mode, for beams of any size, including states with no pointed ids."""
import torch

from src.offline.batching import pack_batch
from src.offline.encoder import StateGraph
from src.offline.resident import ResidentTrees


class _Cache:
    def __init__(self, states, goal=None):
        self.states, self.goal = states, goal


def _graph(g, n, e, p):
    return StateGraph(
        node_ids=torch.randint(-(2**62), 2**62, (n,), generator=g, dtype=torch.int64),
        edge_index=torch.randint(0, n, (2, e), generator=g, dtype=torch.int64),
        edge_attr=torch.randint(0, 50, (e,), generator=g, dtype=torch.int64),
        pointed_ids=torch.randperm(n, generator=g)[:p].to(torch.int64),
    )


def _caches(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, name in enumerate(["a@bfs", "b@dfs", "c@hfs"]):
        states = [_graph(g, n=int(torch.randint(1, 9, (1,), generator=g)), e=int(torch.randint(0, 12, (1,), generator=g)),
                         p=int(torch.randint(0, 3, (1,), generator=g))) for _ in range(5 + k)]
        out[name] = _Cache(states, goal=_graph(g, 3, 2, 1))
    return out


def _picks(caches, seed=1):
    g = torch.Generator().manual_seed(seed)
    names = list(caches)
    picks = []
    for _ in range(40):
        name = names[int(torch.randint(0, len(names), (1,), generator=g))]
        n = len(caches[name].states)
        k = int(torch.randint(1, 7, (1,), generator=g))
        picks.append((name, [int(i) for i in torch.randint(0, n, (k,), generator=g)]))
    return picks


def _same(a, b):
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype, k
            assert torch.equal(a[k].cpu(), b[k].cpu()), k
        else:
            assert a[k] == b[k], k


def test_resident_pack_matches_pack_batch_merged():
    caches = _caches()
    picks = _picks(caches)
    _same(ResidentTrees(caches, "cpu").pack(picks), pack_batch(caches, picks, "cpu"))


def test_resident_pack_matches_pack_batch_separated():
    caches = _caches()
    picks = _picks(caches)
    goals = [caches[name].goal for name, _ in picks]
    _same(ResidentTrees(caches, "cpu").pack(picks, goal_graphs=goals),
          pack_batch(caches, picks, "cpu", goal_graphs=goals))


def test_resident_single_slot_and_repeated_state():
    caches = _caches()
    picks = [("a@bfs", [2]), ("a@bfs", [2, 2]), ("c@hfs", [0, 4, 0])]
    _same(ResidentTrees(caches, "cpu").pack(picks), pack_batch(caches, picks, "cpu"))
