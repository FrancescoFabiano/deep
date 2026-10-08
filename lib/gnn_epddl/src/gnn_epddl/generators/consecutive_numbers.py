"""Generated consecutive-numbers instances (tier 1-basic).

Two templates from the benchmark: 2 agents (as cn-2-05: numbers n0..nN, A odd
/ B even assignable, B holds nb, goal ``([B] ([A] (has B nb)))``) and 3 agents
(as cn-3-10: assignable round-robin n mod 3, actual A=n(3k), B=n(3k+1),
C=n(3k+2), goal: B and C know that everyone knows their numbers). Varied: N,
B's number b (2 agents) and k (3 agents). Training N up to 19 (2 agents) and
13 (3 agents), validation N = 16-18; the benchmark sizes N = 5, 10, 15, 20 are
never used.
"""

from . import Problem

TIER, FAMILY = '1-basic', 'consecutive-numbers'
REQ = """    (:requirements
        :typing :equality :list-comprehensions :facts :finitary-S5-theories
        :negative-list-formulas :modal-goals
    )"""
COMMON = """            ; Agents have at most one number
            ([C. All]
                (forall (?i - agent
                         ?m ?n - number |
                            (/= ?m ?n))
                    (imply
                        (has ?i ?m)
                        (not (has ?i ?n)) )))
            ; Agents know their number
            (:forall (?i - agent
                      ?n - number)
                ([C. All]
                    ([Kw. ?i] (has ?i ?n)) )))"""


def two(N, b, name):
    nums = ' '.join(f'n{i}' for i in range(N + 1))
    cons = ' '.join(f'(consecutive n{i} n{i + 1})' for i in range(N))
    A = ' '.join(f'(assignable A n{i})' for i in range(1, N + 1, 2))
    B = ' '.join(f'(assignable B n{i})' for i in range(0, N + 1, 2))
    return f"""(define (problem {name})    ;; 2 agents, N = {N} (generated)
    (:domain consecutive-numbers)

{REQ}

    (:objects
        {nums} - number
    )

    (:agents A B)

    (:facts-init
        {cons}
        ;; Numbers assignable to A
        {A}
        ;; Numbers assignable to B
        {B}
    )

    (:init
        (:and
            ; Numbers assignment from agent B's perspective
            (has B n{b})
            ; A and B have each an assignable number, and the numbers are consecutive
            ([C. All]
                (and
                    (exists (?i ?j - agent
                             ?m ?n - number |
                        (and
                            (assignable ?i ?m)
                            (assignable ?j ?n)
                            (consecutive ?m ?n) ))
                        (and
                            (has ?i ?m)
                            (has ?j ?n) ))
                    (forall (?i - agent
                             ?n - number |
                                (not (assignable ?i ?n)) )
                        (not (has ?i ?n)) )))
{COMMON}
    )

    (:goal
        ([B] ([A] (has B n{b})))
    )
)
"""


def three(N, k, name):
    nums = ' '.join(f'n{i}' for i in range(N + 1))
    cons = ' '.join(f'(consecutive n{i} n{i + 1})' for i in range(N))
    asg = ' '.join(f'(assignable {"ABC"[i % 3]} n{i})' for i in range(N + 1))
    a, b, c = 3 * k, 3 * k + 1, 3 * k + 2
    return f"""(define (problem {name})    ;; 3 agents, N = {N} (generated)
    (:domain consecutive-numbers)

{REQ}

    (:objects
        {nums} - number
    )

    (:agents
        A B C
    )

    (:facts-init
        {cons}
        {asg}
    )

    (:init
        (:and
            ; Actual assignment
            (has A n{a})
            (has B n{b})
            (has C n{c})
            ; A, B and C have each an assignable number, and the numbers are consecutive
            ([C. All]
                (and
                    (exists (?i1 ?i2 ?i3 - agent
                             ?n1 ?n2 ?n3 - number |
                        (and
                            (assignable  ?i1 ?n1)
                            (assignable  ?i2 ?n2)
                            (assignable  ?i3 ?n3)
                            (consecutive ?n1 ?n2)
                            (consecutive ?n2 ?n3) ))
                        (and
                            (has ?i1 ?n1)
                            (has ?i2 ?n2)
                            (has ?i3 ?n3) ))
                    (forall (?i - agent
                             ?n - number |
                                (not (assignable ?i ?n)) )
                        (not (has ?i ?n)) )))
{COMMON}
    )

    (:goal
        (and
            ([B] ([All] (has B n{b})))
            ([C] ([All] (has C n{c}))) )
    )
)
"""


# (agents, N, b or k, intended role)
PLAN = [(2, 7, 4, 'train'), (2, 12, 6, 'train'), (2, 14, 8, 'train'), (2, 19, 10, 'train'), (3, 8, 1, 'train'),
        (3, 11, 2, 'train'), (3, 13, 3, 'train'),
        (2, 16, 8, 'validation'), (2, 17, 10, 'validation'), (2, 18, 12, 'validation'), (3, 16, 3, 'validation'),
        (3, 17, 4, 'validation')]


def problems():
    out = []
    for ag, N, p, _role in PLAN:
        name = f'g-cn-{ag}-{N:02d}-{"b" if ag == 2 else "k"}{p}'
        out.append(Problem(TIER, FAMILY, name, two(N, p, name) if ag == 2 else three(N, p, name)))
    return out
