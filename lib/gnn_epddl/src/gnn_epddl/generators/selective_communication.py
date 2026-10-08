"""Generated selective-communication instances (tier 2-intermediate).

Template: the benchmark problem sc-05-06 (two worlds w-info / w-not-info,
every agent considers both possible, designated w-info, goal
``([All] (info))``). Varied: n agents, m rooms in a line (the benchmark always
has m = n + 1 with agent i in room i) and the agents' rooms (seeded; several
agents may share a room). As in the domain's move effects, two agents are
close when they are in the same or adjacent rooms. Validation: 6 and 7 agents.
"""

import random

from . import Problem

TIER, FAMILY = '2-intermediate', 'selective-communication'
L = 'ABCDEFGHIJKLMNOP'


def problem(n, m, pos, name, seed):
    ag = L[:n]
    close = [f'(close {ag[i]} {ag[j]})' for i in range(n) for j in range(n) if i != j and abs(pos[i] - pos[j]) <= 1]
    at = ' '.join(f'(at {ag[i]} room{pos[i]})' for i in range(n))
    body = lambda info: '\n'.join(['                (:and'] + (['                    (info)'] if info else []) +
                                  [f'                    {at}'] + ([f'                    {" ".join(close)}'] if close else []) +
                                  ['                    (:forall (?i - agent)', '                        (close ?i ?i) ))'])
    rel = '\n             '.join(f'{a} (:forall (?x ?y - world) (?x ?y))' for a in ag)
    return f"""(define (problem {name})    ;; {n} agents; {m} rooms (generated, seed {seed})
    (:domain selective-communication)

    (:requirements
        :typing :equality :list-comprehensions
        :finitary-S5-theories :modal-goals :facts
    )

    (:agents
        {' '.join(ag)}
    )

    (:objects
        {' '.join(f'room{r}' for r in range(1, m + 1))} - room
    )

    (:facts-init
        {' '.join(f'(neighbor room{r} room{r + 1})' for r in range(1, m))}
        (leftmost room1)
        (rightmost room{m})
    )

    (:init
        :worlds (w-info w-not-info)
        :relations
            ({rel} )
        :labels
            (w-info
{body(True)}
             w-not-info
{body(False)})
        :designated
            (w-info)
    )

    (:goal
        ([All] (info))
    )
)
"""


# (n agents, m rooms, intended role); positions drawn per seed in rooms 1..m-1 (not the rightmost room)
PLAN = [(2, 4, 'train'), (3, 5, 'train'), (3, 6, 'train'), (4, 6, 'train'), (4, 7, 'train'), (5, 7, 'train'),
        (6, 8, 'validation'), (6, 9, 'validation'), (7, 9, 'validation'), (7, 10, 'validation')]


def problems():
    out = []
    for j, (n, m, _role) in enumerate(PLAN):
        seed = 2000 + 31 * j
        rng = random.Random(seed)
        pos = sorted(rng.randint(1, m - 1) for _ in range(n))
        name = f'g-sc-{n:02d}-{m:02d}-s{seed}'
        out.append(Problem(TIER, FAMILY, name, problem(n, m, pos, name, seed)))
    return out
