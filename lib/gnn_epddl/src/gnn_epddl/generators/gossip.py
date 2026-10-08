"""Generated gossip instances (tiers 1-basic, 2-intermediate, 3-hard).

Templated on the benchmark problems of each variant (templates/: gos-05-all
for 'all' and 'single', gos-04-{all,imp}-deceived for the deceived variants;
domain files unchanged). Varied:
  - number of agents (all/single: 3-6; deceived: 1-3 normal agents plus the
    detective and the impostor),
  - knowledge shared at the start: for chosen pairs (j, i),
    ``([C. All] ([Kw. j] (secret i)))`` is added to the initial state,
  - for 'single', the agent who must learn every secret.
Seeded; the sharing patterns are drawn in a fixed call order (``problems``),
and no two generated instances share (variant, size, pattern).
"""

import os
import random
import re

from . import Problem

FAMILY = 'gossip'
L = 'ABCDEFGHIJKLMNOP'
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def _template(name):
    with open(os.path.join(TEMPLATES, f'gossip-{name}.epddl')) as fh:
        return fh.read()


def shared_init(t, pairs):
    """Adds common knowledge that agent j knows whether secret i holds, for each (j, i)."""
    if not pairs:
        return t
    add = ''.join(f'\n            ([C. All] ([Kw. {j}] (secret {i})))' for j, i in pairs)
    i = t.index('(:init')
    k = t.index('(:and', i) + len('(:and')
    return t[:k] + add + t[k:]


class _Gen:
    def __init__(self):
        self.used = set()   # (tag, pairs) already drawn
        self.out = []

    def pick_pairs(self, agents, k, rng, tag):
        cand = [(j, i) for j in agents for i in agents if i != j]
        if not k:
            return []
        for _ in range(200):
            pairs = tuple(sorted(rng.sample(cand, min(k, len(cand)))))
            if (tag, pairs) not in self.used:
                self.used.add((tag, pairs))
                return list(pairs)
        raise ValueError(f'no new sharing pattern for {tag} with {k} pairs')

    def all_single(self, tier, n, variant, role, shared=0, seed=0, target='A'):
        src = _template('gos-05-all')
        ag = list(L[:n])
        pairs = self.pick_pairs(ag, shared, random.Random(seed), (tier, variant, n, target))
        name = (f'g-gos-{n:02d}-{variant}' + (f'-sh{shared}s{seed}' if shared else '')
                + (f'-t{target}' if variant == 'single' and target != 'A' else ''))
        t = re.sub(r'\(define \(problem [^)]*\)', f'(define (problem {name})', src, count=1)
        t = re.sub(r'\(:agents\s+[^)]*\)', '(:agents\n        ' + ' '.join(ag) + '\n    )', t, count=1)
        if variant == 'single':
            t = t.replace('([Kw. All] (secret ?i))', f'([Kw. {target}] (secret ?i))')
        self.out.append(Problem(tier, FAMILY, name, shared_init(t, pairs)))

    def deceived(self, var, normal, role, shared=0, seed=0):
        tier = '3-hard'
        src = _template(f'gos-04-{var}-deceived')
        norm = list(L[:normal])
        pairs = self.pick_pairs(norm, shared, random.Random(seed), (var, normal)) if normal > 1 else []
        n = normal + 2
        name = f'g-gos-{n:02d}-{var}-deceived' + (f'-sh{len(pairs)}s{seed}' if pairs else '')
        t = re.sub(r'\(define \(problem [^)]*\)', f'(define (problem {name})', src, count=1)
        t = re.sub(r'A B(\s+)- normal-agent', ' '.join(norm) + r'\1- normal-agent', t, count=1)
        self.out.append(Problem(tier, FAMILY, name, shared_init(t, pairs)))


def problems():
    g = _Gen()
    a, d = g.all_single, g.deceived
    # all: basic tier trains at 3-4 agents and validates at 5-6; intermediate states are larger: train 3, validate 4-5
    a('1-basic', 3, 'all', 'train', shared=1, seed=1); a('1-basic', 4, 'all', 'train'); a('1-basic', 4, 'all', 'train', shared=2, seed=2)
    a('2-intermediate', 3, 'all', 'train', shared=1, seed=1); a('2-intermediate', 3, 'all', 'train', shared=2, seed=2)
    a('2-intermediate', 3, 'all', 'train', shared=3, seed=3)
    a('1-basic', 5, 'all', 'validation', shared=2, seed=3); a('1-basic', 6, 'all', 'validation')
    a('2-intermediate', 4, 'all', 'validation', shared=2, seed=4); a('2-intermediate', 5, 'all', 'validation', shared=3, seed=4)
    # single (benchmark: 7, 9, 11 agents)
    a('1-basic', 3, 'single', 'train'); a('1-basic', 4, 'single', 'train', target='B'); a('1-basic', 5, 'single', 'train')
    a('2-intermediate', 3, 'single', 'train'); a('2-intermediate', 3, 'single', 'train', target='C')
    a('2-intermediate', 4, 'single', 'train', target='B')
    a('1-basic', 6, 'single', 'validation'); a('1-basic', 5, 'single', 'validation', shared=3, seed=6)
    a('2-intermediate', 4, 'single', 'validation', shared=2, seed=7); a('2-intermediate', 5, 'single', 'validation', target='C')
    # all-deceived: plans are short; train 1-3 normal agents, validate 3 normal agents with distinct sharing patterns
    d('all', 1, 'train'); d('all', 2, 'train', shared=1, seed=1); d('all', 2, 'train', shared=1, seed=2)
    d('all', 2, 'train', shared=2, seed=3); d('all', 3, 'train')
    for k, sd in ((1, 5), (2, 6), (3, 7), (1, 8), (2, 9)):
        d('all', 3, 'validation', shared=k, seed=sd)
    # imp-deceived: 5 agents are heavy; train 1-2 normal agents, validate 2 normal (other patterns) and 3 normal
    d('imp', 1, 'train'); d('imp', 2, 'train', shared=1, seed=1); d('imp', 2, 'train', shared=1, seed=2)
    d('imp', 2, 'validation', shared=2, seed=3); d('imp', 3, 'validation'); d('imp', 3, 'validation', shared=1, seed=5)
    d('imp', 3, 'validation', shared=2, seed=6); d('imp', 3, 'validation', shared=3, seed=7)
    return g.out
