"""Checks of the generated problems, before any of them is used.

For every problem: (1) plank parses it (``plank parse -d <domain> -p <problem>
-l <library>``); (2) its text differs from every benchmark problem of the
family (comments and whitespace ignored), so no test problem is trained or
validated on; (3) deep's breadth-first search solves it within a time and
memory budget (default 120 s / 6 GiB; validation problems were checked with
600 s / 24 GiB), reporting the plan length and the expanded nodes:
``deep <domain> <problem> --act_lib <library> -b -c --fast-comparison -s BFS``.

Benchmarks layout: <benchmarks>/<tier>/<family>/problems/*.epddl.
"""

import glob
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from . import layout

FIELDS = ['tier', 'family', 'problem', 'parse', 'duplicate', 'solved', 'plan_length', 'nodes', 'secs']


def norm(t):
    """Problem text without comments and with whitespace collapsed."""
    return re.sub(r'\s+', ' ', re.sub(r';[^\n]*', '', t)).strip()


def check(instances, tier, fam, prob, deep, plank, benchmarks, timeout=120, mem_gb=6.0, strategy='BFS'):
    dom, pr, _, lib = layout.instance_files(instances, tier, fam, prob)
    row = dict(tier=tier, family=fam, problem=prob, parse='', duplicate='', solved='', plan_length='', nodes='', secs='')
    p = subprocess.run([plank, 'parse', '-d', dom, '-p', pr, '-l', lib], capture_output=True, text=True, timeout=120)
    row['parse'] = 'ok' if p.returncode == 0 else 'FAIL'
    mine = norm(open(pr).read())
    row['duplicate'] = 'yes' if any(norm(open(f).read()) == mine
                                    for f in glob.glob(f'{benchmarks}/*/{fam}/problems/*.epddl')) else 'no'
    if row['parse'] != 'ok':
        return row
    t0 = time.time()
    cmd = (f'ulimit -v {int(mem_gb * 1048576)}; exec timeout {timeout} {layout.deep_path(deep)} {dom} {pr} --act_lib {lib} '
           f'-b -c --fast-comparison -s {strategy}')
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, cwd=cwd)
    out = r.stdout + r.stderr
    row['secs'] = round(time.time() - t0, 1)
    if 'Goal found' in out:
        row['solved'] = 'yes'
        m = re.search(r'Plan length:\s*(\d+)', out)
        row['plan_length'] = m.group(1) if m else ''
        m = re.search(r'Nodes expanded:\s*(\d+)', out)
        row['nodes'] = m.group(1) if m else ''
    else:
        row['solved'] = 'no-plan' if 'No goal found' in out else ('timeout' if r.returncode == 124 else f'error {r.returncode}')
    return row


def check_family(instances, tier, fam, problems, jobs=3, **kw):
    """Yields one row per problem (default: every problem of the family in that tier), jobs at a time."""
    probs = problems or sorted(os.path.basename(f)[:-6]
                               for f in glob.glob(os.path.join(instances, tier, fam, 'problems', '*.epddl')))
    with ThreadPoolExecutor(jobs) as ex:
        yield from ex.map(lambda p: check(instances, tier, fam, p, **kw), probs)
