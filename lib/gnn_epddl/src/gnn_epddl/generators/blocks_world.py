"""Generated blocks-world instances (three sub-families, one per tier).

Each sub-family is templated on the benchmark problems of its tier; only
sizes, seeded towers and goal facts vary.
  classical (1-basic, as bw-05): n blocks, one world, relation (Robot (w w)),
      random initial towers; the goal is a random target arrangement
      (conjunction of on / on-table facts), as in the benchmark goals.
  epistemic (2-intermediate, as bw-3-05-2 ... bw-5-07-3): agents A, B, ... at
      tables t1 t2 t1 t2 t3 (benchmark order), W worlds with A uncertain
      between all of them and every other agent certain, one tower per table
      in every world, every agent focused on itself, A can move every block,
      all arms empty, w1 designated. Goal: facts of a random target
      arrangement placed as in the benchmark goals: a physical fact, beliefs
      of single agents ([B] f), a belief over a conjunction ([D] (and f g))
      and, for larger instances, a nested belief ([B] ([C] f)).
  clumsy (3-hard, as bw-3-3-2-clumsy / bw-3-5-2-clumsy): precise agents (A, B)
      at t1 t2, clumsy agents at every table, A uncertain, A and the clumsy
      agents can move every block; goals as in the clumsy benchmark problems:
      a clumsy agent's belief over a conjunction with disjunctions, common
      knowledge of a disjunction, or the epistemic goal shape.

The instance lists below are the kept instances: candidates of the same
generator that failed the checks (check.py) are not listed.
"""

import random

from . import Problem

FAMILY = 'blocks-world'
TIERS = {'classical': '1-basic', 'epistemic': '2-intermediate', 'clumsy': '3-hard'}
LETTERS = 'ABCDEFGH'
SEAT = ['t1', 't2', 't1', 't2', 't3', 't3', 't1', 't2']   # benchmark order of the agents' tables


def towers(blocks, k, rng):
    """Random partition of the blocks into exactly k non-empty towers (bottom -> top)."""
    b = blocks[:]
    rng.shuffle(b)
    cuts = sorted(rng.sample(range(1, len(b)), k - 1)) if k > 1 else []
    return [b[i:j] for i, j in zip([0] + cuts, cuts + [len(b)])]


def tower_facts(tw, tables=None, clear=True):
    f = []
    for i, t in enumerate(tw):
        f.append(f'(on-table {t[0]} {tables[i]})' if tables else f'(on-table {t[0]})')
        f += [f'(on {t[j]} {t[j - 1]})' for j in range(1, len(t))]
        if clear:
            f.append(f'(clear {t[-1]})')
    return f


def classical(n, seed):
    rng = random.Random(1000 + 97 * n + seed)
    blocks = [f'b{i}' for i in range(1, n + 1)]
    ini = towers(blocks, rng.randint(1, min(3, n)), rng)
    goal = towers(blocks, rng.randint(1, min(3, n)), rng)
    while sorted(map(tuple, goal)) == sorted(map(tuple, ini)):
        goal = towers(blocks, rng.randint(1, min(3, n)), rng)
    name = f'g-bw-{n:02d}-s{seed}'
    lab = '\n'.join('                    ' + x for x in tower_facts(ini) + ['(arm-empty)'])
    gl = '\n'.join('            ' + x for x in tower_facts(goal, clear=False))
    return name, f"""(define (problem {name})  ;; generated: {n} blocks, seed {seed} (training/validation only)
    (:domain blocks-world)

    (:requirements
        :finitary-S5-theories :typing :lists
    )

    (:objects
        {' '.join(blocks)} - block
    )

    (:init
        :worlds
            (w)
        :relations
            (Robot (w w))
        :labels
            (w
                (:and
{lab}) )
        :designated (w)
    )

    (:goal
        (and
{gl}
        )
    )
)
"""


def multi_world(agents, n, nt, nw, seed, clumsy):
    """Shared part of epistemic and clumsy: objects, agents, facts-init, worlds, relations, labels."""
    rng = random.Random(5000 + 1013 * len(agents) + 101 * n + 11 * nt + seed + (777 if clumsy else 0))
    blocks = [f'b{i}' for i in range(1, n + 1)]
    tables = [f't{i}' for i in range(1, nt + 1)]
    worlds = [f'w{i}' for i in range(1, nw + 1)]
    precise = agents[:2] if clumsy else agents
    clumsies = agents[2:] if clumsy else []
    at = [f'(at {a} {SEAT[i] if SEAT[i] in tables else "t1"})' for i, a in enumerate(precise)]
    for a in clumsies:
        at += [f'(at {a} {t})' for t in tables]
    movers = ['A'] + clumsies
    rel = []
    for a in agents:
        rel.append(f'{a} (:forall (?w1 ?w2 - world) (?w1 ?w2))' if a == 'A' else f'{a} (:forall (?w - world) (?w ?w))')
    labels, seen = [], set()
    for w in worlds:
        tw = towers(blocks, nt, rng)
        while tuple(map(tuple, tw)) in seen:
            tw = towers(blocks, nt, rng)
        seen.add(tuple(map(tuple, tw)))
        facts = [f'(focused-on {a} {a})' for a in agents] + [f'(can-move {m} {b})' for m in movers for b in blocks]
        facts += tower_facts(tw, tables) + [f'(arm-empty {a})' for a in agents]
        labels.append((w, facts, tw))
    return rng, blocks, tables, worlds, precise, clumsies, at, rel, labels


def render(name, comment, blocks, tables, agents_decl, at, worlds, rel, labels, goal):
    req = ':finitary-S5-theories :typing :equality\n        :lists :list-comprehensions :facts\n        :negative-goals :modal-goals :disjunctive-list-formulas'
    lab = '\n             '.join(f'{w}\n                (:and\n' + '\n'.join('                    ' + f for f in facts) + '\n                )'
                                for w, facts, _ in labels)
    return f"""(define (problem {name})      ;; {comment}
    (:domain blocks-world)

    (:requirements
        {req}
    )

    (:objects
        {' '.join(blocks)} - block
        {' '.join(tables)} - table
    )

    (:agents
        {agents_decl}
    )

    (:facts-init
{chr(10).join('        ' + x for x in at)}
    )

    (:init
        :worlds
            ({' '.join(worlds)})
        :relations
            ({chr(10).join(('             ' if i else '') + r for i, r in enumerate(rel))} )
        :labels
            ({lab}
            )
        :designated (w1)
    )

    (:goal
{goal}
    )
)
"""


def target_facts(blocks, tables, rng):
    tw = towers(blocks, len(tables), rng)
    return [f for f in tower_facts(tw, tables, clear=False)], tw


def epistemic(na, n, nt, nw, seed, nconj):
    agents = list(LETTERS[:na])
    rng, blocks, tables, worlds, precise, clumsies, at, rel, labels = multi_world(agents, n, nt, nw, seed, False)
    facts, _ = target_facts(blocks, tables, rng)
    rng.shuffle(facts)
    others = agents[1:]
    parts = [facts[0]]                                                  # a physical fact
    k = 1
    for j in range(nconj - 1):
        if k >= len(facts):
            break
        if j == 1 and k + 1 < len(facts):                               # a belief over a conjunction
            a = others[(j + seed) % len(others)]
            parts.append(f'([{a}]\n                (and\n                    {facts[k]}\n                    {facts[k + 1]} ))')
            k += 2
        elif j == 3 and len(others) >= 2:                               # a nested belief
            a, b = others[(j + seed) % len(others)], others[(j + seed + 1) % len(others)]
            parts.append(f'([{a}] ([{b}] {facts[k]}))')
            k += 1
        else:                                                           # a single agent's belief
            a = others[(j + seed) % len(others)]
            parts.append(f'([{a}] {facts[k]})')
            k += 1
    goal = '        (and\n' + '\n'.join('            ' + p for p in parts) + '\n        )'
    name = f'g-bw-{na}-{n:02d}-{nt}-w{nw}-s{seed}'
    return name, render(name, f'generated: {na} agents; {n} blocks; {nt} tables; {nw} worlds; seed {seed} (training/validation only)',
                        blocks, tables, ' '.join(agents), at, worlds, rel, labels, goal)


def clumsy(na, n, nt, nw, seed, shape):
    agents = list(LETTERS[:na])
    rng, blocks, tables, worlds, precise, clumsies, at, rel, labels = multi_world(agents, n, nt, nw, seed, True)
    decl = f"{' '.join(precise)} - precise-agent\n        {' '.join(clumsies)} - clumsy-agent"
    facts, tw = target_facts(blocks, tables, rng)
    c = clumsies[seed % len(clumsies)]
    if shape == 'disj':        # as bw-3-5-2-clumsy: a clumsy agent believes a conjunction of facts and disjunctions
        rng.shuffle(facts)
        alt = [f'(on-table {rng.choice(blocks)} {rng.choice(tables)})' for _ in facts]
        inner = [facts[0]] + [f'(or\n                    {facts[i]}\n                    {alt[i]} )' for i in range(1, min(len(facts), 3))]
        goal = f'        ([{c}]\n            (and\n' + '\n'.join('                ' + x for x in inner) + '))'
    elif shape == 'common':    # as bw-3-3-2-clumsy: common knowledge of a disjunction
        f = rng.choice(facts)
        g = f'(on-table {rng.choice(blocks)} {tables[-1]})'
        goal = f'        ([C. All]\n            (or\n                {f}\n                {g} ))'
    else:                      # 'beliefs': epistemic goal shape, with the clumsy agents' beliefs
        rng.shuffle(facts)
        others = agents[1:]
        parts = [facts[0]] + [f'([{others[(i + seed) % len(others)]}] {facts[i]})' for i in range(1, min(len(facts), 3))]
        goal = '        (and\n' + '\n'.join('            ' + p for p in parts) + '\n        )'
    name = f'g-bw-{na}-{n}-{nt}-clumsy-w{nw}-{shape}-s{seed}'
    return name, render(name, f'generated: {na} agents ({len(clumsies)} clumsy); {n} blocks; {nt} tables; {nw} worlds; goal {shape}; seed {seed} (training/validation only)',
                        blocks, tables, decl, at, worlds, rel, labels, goal)


# Kept instances: (n blocks, seeds); (agents, blocks, tables, worlds, goal parts, seeds); (agents, blocks, tables, worlds, goal shape, seeds)
CLASSICAL = [(n, (1, 2, 3)) for n in (4, 5, 6, 7, 8, 9, 10, 12)]
EPISTEMIC = [(3, 3, 2, 2, 2, (2,)), (3, 3, 2, 3, 3, (1,)), (3, 4, 2, 3, 3, (1,)), (4, 5, 2, 3, 4, (1, 2, 3)),
             (3, 6, 2, 2, 3, (4, 5, 6, 7)), (4, 5, 2, 2, 3, (4, 5, 6, 7))]
CLUMSY = [(3, 3, 2, 2, 'disj', (1, 2, 3)), (3, 3, 2, 3, 'common', (1, 2, 3)), (3, 5, 2, 2, 'disj', (1, 2, 3)),
          (4, 4, 2, 2, 'beliefs', (1,)), (4, 5, 2, 3, 'beliefs', (1, 2))]


def problems():
    out = []
    for n, seeds in CLASSICAL:
        for s in seeds:
            out.append(Problem(TIERS['classical'], FAMILY, *classical(n, s)))
    for na, n, nt, nw, nconj, seeds in EPISTEMIC:
        for s in seeds:
            out.append(Problem(TIERS['epistemic'], FAMILY, *epistemic(na, n, nt, nw, s, nconj)))
    for na, n, nt, nw, shape, seeds in CLUMSY:
        for s in seeds:
            out.append(Problem(TIERS['clumsy'], FAMILY, *clumsy(na, n, nt, nw, s, shape)))
    return out
