"""Training beams for the distance estimator.

The estimator is deployed on the planner's beams (FringeEvalRL), so it trains
on beams of the same shape: the behaviour policies of deep_nn.dataset roll the
reservoir environment over each generation tree and every decision beam they
pass becomes one sample.  The per-state target is the table's
`Distance From Goal`, scaled to (0, 1); a state the generator marked
unreachable is kept in the beam (the planner scores it too) but excluded from
the loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from deep_nn.cache import InstanceCache
from deep_nn.dataset import generate_dataset
from deep_nn.pack import pack_fringe, pack_fringes, pack_goal_tensors
from deep_nn.policies import BEHAVIOUR_POLICIES
from deep_nn.strategies import depth_from_csv_name
from deep_nn.tree import TreeInstance, load_tree_instance, partition_solvable
from src.utils import LEGACY_MAX_DEPTH, normalization_params, scale_distance

Beam = Tuple[str, Tuple[int, ...]]   # (tree name, state ids in slot order)


def load_trees(csvs: Sequence[str | Path], kind_of_data: str, cache_dir: Path,
               repo_root: Path) -> Tuple[List[TreeInstance], Dict[str, InstanceCache]]:
    """Solvable trees + their parsed states. The DOTs are always merged; the
    separated view is derived in the cache (deep_nn.dot.separated_view)."""
    loaded = [load_tree_instance(p, kind_of_data=kind_of_data) for p in csvs]
    solvable, unsolvable = partition_solvable(loaded)
    for u in unsolvable:
        print(f"[beams] {u.name}: no goal reachable from the root, skipped")
    caches = {
        t.name: InstanceCache.from_paths(
            t.state_paths_abs(repo_root), separated=(kind_of_data == "separated"),
            cache_file=cache_dir / f"{t.name}.{kind_of_data}.pt", verbose=False)
        for t in solvable
    }
    return solvable, caches


def max_generation_depth(csvs: Sequence[str | Path]) -> int:
    depths = [depth_from_csv_name(Path(p).name) for p in csvs]
    return max([d for d in depths if d] or [LEGACY_MAX_DEPTH])


def collect_beams(trees: Sequence[TreeInstance], fringe_size: int,
                  policies: Sequence[str] = BEHAVIOUR_POLICIES, seeds_per_policy: int = 3,
                  verbose: bool = True) -> List[Beam]:
    """Every distinct decision beam the behaviour rollouts pass, in first-seen order."""
    rows, _ = generate_dataset(trees, fringe_size, policies=policies,
                               seeds_per_policy=seeds_per_policy, counterfactual="none",
                               verbose=verbose)
    return list(dict.fromkeys((r.instance, tuple(r.obs)) for r in rows))


class BeamDataset(Dataset):
    def __init__(self, beams: Sequence[Beam], trees: Sequence[TreeInstance],
                 caches: Dict[str, InstanceCache], fringe_size: int, unreachable: float,
                 params: Dict[str, float]):
        self.beams = list(beams)
        self.caches = caches
        self.fringe_size = int(fringe_size)
        self.separated = any(c.goal is not None for c in caches.values())
        self.target: Dict[str, torch.Tensor] = {}
        self.labelled: Dict[str, torch.Tensor] = {}
        for t in trees:
            h = torch.tensor(t.h_star, dtype=torch.float64)
            self.labelled[t.name] = h != float(unreachable)
            self.target[t.name] = scale_distance(h, params)

    def __len__(self) -> int:
        return len(self.beams)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        name, obs = self.beams[i]
        ids = torch.tensor(obs, dtype=torch.long)
        p = pack_fringe(self.caches[name], obs, self.fringe_size)
        p["target"] = self.target[name][ids]
        p["label_mask"] = self.labelled[name][ids]
        p["name"] = name
        return p

    def collate(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out = pack_fringes(items)
        out.pop("n_slots"), out.pop("slot_offset")
        out["target"] = torch.cat([p["target"] for p in items])
        out["label_mask"] = torch.cat([p["label_mask"] for p in items])
        if self.separated:
            out.update(pack_goal_tensors([self.caches[p["name"]].goal for p in items]))
        return out

    def loader(self, batch_size: int, shuffle: bool, seed: Optional[int] = None) -> DataLoader:
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate,
                          generator=gen)

    def planner_feed(self, i: int, fringe_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """One beam exactly as FringeEvalRL sends it (mask of length F, one goal).
        ``fringe_size`` re-packs the beam for an export of another width: the first
        F states when F is smaller than the beam, padding otherwise."""
        name, obs = self.beams[i]
        F = self.fringe_size if fringe_size is None else int(fringe_size)
        p = pack_fringe(self.caches[name], list(obs)[:F], F)
        if self.separated:
            p.update(pack_goal_tensors([self.caches[name].goal]))
        return {k: v for k, v in p.items() if isinstance(v, torch.Tensor)}


def build_datasets(train_csvs, test_csvs, kind_of_data: str, fringe_size: int, unreachable: float,
                   cache_dir: Path, repo_root: Path, policies=BEHAVIOUR_POLICIES,
                   seeds_per_policy: int = 3):
    """(train, test-or-None, params). Split = the folders the CSVs came from."""
    params = normalization_params(max_generation_depth(train_csvs))
    trees, caches = load_trees(train_csvs, kind_of_data, cache_dir, repo_root)
    train = BeamDataset(collect_beams(trees, fringe_size, policies, seeds_per_policy),
                        trees, caches, fringe_size, unreachable, params)
    test = None
    if test_csvs:
        t_trees, t_caches = load_trees(test_csvs, kind_of_data, cache_dir, repo_root)
        test = BeamDataset(collect_beams(t_trees, fringe_size, policies, seeds_per_policy),
                           t_trees, t_caches, fringe_size, unreachable, params)
    return train, test, params
