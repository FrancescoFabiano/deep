"""Generators of the training and validation instances.

The benchmark problems are test problems only: every model is trained and
validated on instances written by these generators. One module per domain;
each exposes ``problems()``, the list of the generated problems of that
domain as ``Problem(tier, family, name, text)``. All generators are
deterministic (fixed seeds), so re-running them gives byte-identical files.

The generated problems use the benchmark domain files unchanged; only problem
files are written here. Layout (the one deep's trials use, one folder per
tier)::

    <root>/<tier>/act_lib.epddl
    <root>/<tier>/<family>/domain.epddl
    <root>/<tier>/<family>/problems/<name>.epddl
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple


class Problem(NamedTuple):
    tier: str      # 1-basic | 2-intermediate | 3-hard
    family: str    # domain folder name, e.g. blocks-world
    name: str      # problem name (file name without .epddl)
    text: str      # file content


def _modules():
    from . import (active_muddy_child, blocks_world, cloud_scheduling, consecutive_numbers, gossip,
                   search_and_rescue, selective_communication)
    return {
        'active-muddy-child': active_muddy_child,
        'blocks-world': blocks_world,
        'cloud-scheduling': cloud_scheduling,
        'consecutive-numbers': consecutive_numbers,
        'gossip': gossip,
        'search-and-rescue': search_and_rescue,
        'selective-communication': selective_communication,
    }


FAMILIES = ['active-muddy-child', 'blocks-world', 'cloud-scheduling', 'consecutive-numbers', 'gossip',
            'search-and-rescue', 'selective-communication']


def problems(family: str) -> List[Problem]:
    """All generated problems of one family."""
    return _modules()[family].problems()


def problem_path(root: str, p: Problem) -> str:
    return os.path.join(root, p.tier, p.family, 'problems', p.name + '.epddl')


def write(root: str, families: List[str] = None) -> Dict[str, int]:
    """Writes the problems of the given families (default: all) under root; returns the count per family."""
    out = {}
    for fam in families or FAMILIES:
        ps = problems(fam)
        for p in ps:
            path = problem_path(root, p)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as fh:
                fh.write(p.text)
        out[fam] = len(ps)
    return out
