"""Generated active-muddy-child instances (tier 1-basic).

Template: the benchmark problems amc-06-03-clean / amc-09-05-muddy (same
requirements, initial state and goal shape). Varied: the number of children n
and which k children are muddy (seeded); 'clean' = Child1 is not muddy,
'muddy' = Child1 is muddy. Training n = 4-8, validation n = 10-11; the
benchmark sizes n = 3, 6, 9, 12 are never used. Goal unchanged:
``([Kw. Child1] (muddy Child1))``.
"""

import random

from . import Problem

TIER, FAMILY = '1-basic', 'active-muddy-child'
TEMPLATE = """(define (problem {name})  ;; {n} children; {k} are muddy; child 1 is {c1} (generated, seed {seed})
    (:domain active-muddy-child)

    (:requirements
        :equality :list-comprehensions
        :finitary-S5-theories :modal-goals
    )

    (:agents
        {agents}
    )

    (:init
        (:and
            ;; Muddy children
            {muddy}
            ;; It is common knowledge that at least one child is muddy
            ([C. All]
                (exists (?i - agent)
                    (muddy ?i) ))
            ;; It is common knowledge that all children know whether the others are muddy
            (:forall (?i ?j - agent | (/= ?i ?j))
                ([C. All]
                    ([Kw. ?i] (muddy ?j)) )))
    )

    (:goal
        ([Kw. Child1] (muddy Child1))
    )
)
"""
# (n, k, intended role): training sizes small, validation larger than every training size
PLAN = {'clean': [(4, 1, 'train'), (4, 3, 'train'), (5, 2, 'train'), (7, 2, 'train'), (7, 5, 'train'), (8, 4, 'train'),
                  (10, 4, 'validation'), (10, 7, 'validation'), (11, 5, 'validation')],
        'muddy': [(4, 2, 'train'), (5, 3, 'train'), (5, 5, 'train'), (7, 3, 'train'), (7, 7, 'train'), (8, 6, 'train'),
                  (10, 5, 'validation'), (10, 10, 'validation'), (11, 6, 'validation')]}


def problems():
    out = []
    for var, plan in PLAN.items():
        for j, (n, k, _role) in enumerate(plan):
            seed = 1000 + 17 * j + (0 if var == 'clean' else 500)
            rng = random.Random(seed)
            others = [f'Child{i}' for i in range(2, n + 1)]
            mud = (['Child1'] if var == 'muddy' else []) + rng.sample(others, k - 1 if var == 'muddy' else k)
            mud = sorted(mud, key=lambda c: int(c[5:]))
            name = f'g-amc-{n:02d}-{k:02d}-{var}-s{seed}'
            text = TEMPLATE.format(name=name, n=n, k=k, c1='muddy' if var == 'muddy' else 'not muddy', seed=seed,
                                   agents=' '.join(f'Child{i}' for i in range(1, n + 1)),
                                   muddy=' '.join(f'(muddy {c})' for c in mud))
            out.append(Problem(TIER, FAMILY, name, text))
    return out
