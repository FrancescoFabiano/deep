"""File layout shared by every step: instances, the split, per-instance data folders, deep's command line.

Instances (exp/gnn_epddl/instances, the layout of deep's trials, one folder per tier)::

    <instances>/<tier>/act_lib.epddl
    <instances>/<tier>/<family>/domain.epddl
    <instances>/<tier>/<family>/problems/<problem>.epddl

Split (exp/gnn_epddl/split.csv): columns family, tier, problem, label (train |
validation), reason (sub-family). Every benchmark problem is a test problem
and never appears here.

Per-instance data: ``<data>/<family>/<short tier>-<problem>/`` with the raw
tree written by deep (``out/``), ``fluent_names.csv``, ``goal_ops.csv`` and the
built training set ``new3.pt`` (short tier: basic | intermediate | hard).
"""

import csv
import os
from typing import Dict, List

TIERS = ['1-basic', '2-intermediate', '3-hard']


def short_tier(tier: str) -> str:
    """'2-intermediate' -> 'intermediate'."""
    return tier.split('-')[1]


def deep_path(deep: str) -> str:
    """deep runs in its own work folder: a path to the binary is made absolute (a bare name is looked up in PATH)."""
    return os.path.abspath(deep) if os.sep in deep else deep


def instance_files(instances: str, tier: str, family: str, problem: str) -> List[str]:
    """deep's positional arguments and library option for one problem (absolute paths)."""
    instances = os.path.abspath(instances)
    return [os.path.join(instances, tier, family, 'domain.epddl'),
            os.path.join(instances, tier, family, 'problems', problem + '.epddl'),
            '--act_lib', os.path.join(instances, tier, 'act_lib.epddl')]


def read_split(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def subfamily_split(rows: List[Dict[str, str]], family: str, sub: str) -> List[Dict[str, str]]:
    """The split with one family restricted to one sub-family (reason column); the other families are unchanged."""
    return [r for r in rows if r['family'] != family or r['reason'] == sub]


def data_dir(data: str, family: str, tier: str, problem: str) -> str:
    return os.path.join(data, family, f'{short_tier(tier)}-{problem}')


def inst_paths(rows: List[Dict[str, str]], data: str, family: str, label: str) -> List[str]:
    """new3.pt of the family's split rows with that label, in split order; instances without one are skipped."""
    out = []
    for r in rows:
        if r['family'] == family and r['label'] == label:
            p = os.path.join(data_dir(data, family, r['tier'], r['problem']), 'new3.pt')
            if os.path.exists(p):
                out.append(p)
    return out
