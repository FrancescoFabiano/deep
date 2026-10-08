"""The generators reproduce the shipped instances byte for byte (exp/gnn_epddl/instances).

Run from lib/gnn_epddl: python -m pytest tests -q   (no torch needed)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from gnn_epddl import generators  # noqa: E402

INSTANCES = os.path.join(HERE, '..', '..', '..', 'exp', 'gnn_epddl', 'instances')


def test_generated_problems_match_shipped_instances():
    made = set()
    for fam in generators.FAMILIES:
        for p in generators.problems(fam):
            path = generators.problem_path(INSTANCES, p)
            with open(path) as fh:
                assert fh.read() == p.text, path
            made.add(os.path.normpath(path))
    shipped = {os.path.normpath(os.path.join(d, f)) for d, _, fs in os.walk(INSTANCES) for f in fs
               if f.endswith('.epddl') and os.path.basename(d) == 'problems'}
    assert shipped == made, sorted(shipped ^ made)[:10]


def test_problem_names_unique():
    names = [(p.tier, p.family, p.name) for fam in generators.FAMILIES for p in generators.problems(fam)]
    assert len(names) == len(set(names))
