"""Training data: deep's search trees of the training instances, and the per-instance training sets built from them.

1. Tree (generate_tree): deep's dataset mode on one problem,
       deep <domain> <problem> --act_lib <lib> -b -c --dataset --dataset_depth 25 --dataset_max_creation 60000
            --dataset_max_generation 100000 --dataset_type HASHED --ranker_encoding
            --dataset_generation <BFS|DFS> --dataset_seed <seed>
   run in <data>/<family>/<short tier>-<problem>/ (see layout). Attempts, in
   order: BFS (seed 42), then DFS with seeds 42, 43, 44; the first run that
   stores a dataset with a goal is kept. A run is killed above 12 GiB resident
   memory or after 600 s. Output: out/NN/Training/<run>/<...>_depth_25.csv (one
   row per state: its DOT file, its predecessor, its distance from the goal)
   and out/.../RawFiles/hash_merged/*.dot, plus fluent_names.csv and
   goal_ops.csv written by --ranker_encoding.

2. Training set (build_instance), saved as new3.pt in the same folder:
     graphs : encoding.graph of the sampled states (the states of the groups and pairs below)
     h      : their distances to the goal (finite labels only)
     groups : lists of state indices sharing one parent (>= 2 children with finite labels, not all equal),
              at most MAXG = 3000, sampled with seed 0
     glob   : global pairs (two labelled states of the tree with different labels), at most MAXP = 2000
     dist   : {sha1 of a state's sorted DOT lines: distance} for every labelled state
     agents, names, ops, family, problem
   Instances whose labels are all equal give no training set (None).
"""

import collections
import csv
import glob
import hashlib
import os
import random
import shutil
import subprocess
import time

import torch

from . import encoding, layout

DATASET_ARGS = ['-b', '-c', '--dataset', '--dataset_depth', '25', '--dataset_max_creation', '60000',
                '--dataset_max_generation', '100000', '--dataset_type', 'HASHED', '--ranker_encoding']
ATTEMPTS = [('BFS', 42), ('DFS', 42), ('DFS', 43), ('DFS', 44)]
MEM_KB, TIMEOUT_S = 12 * 2 ** 20, 600
MAXG, MAXP = 3000, 2000


def generate_tree(deep, instances, tier, family, problem, data):
    """Runs deep's dataset generation for one problem; -> (output folder, 'ok <strategy> seed <seed> (<s>s)' or 'FAIL')."""
    out = layout.data_dir(data, family, tier, problem)
    base = [layout.deep_path(deep), *layout.instance_files(instances, tier, family, problem), *DATASET_ARGS]
    for strat, seed in ATTEMPTS:
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        with open(out + '/gen.log', 'w') as log:
            pr = subprocess.Popen(base + ['--dataset_generation', strat, '--dataset_seed', str(seed)], cwd=out,
                                  stdout=log, stderr=subprocess.STDOUT)
            t0, status = time.time(), None
            while pr.poll() is None:
                time.sleep(1)
                rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(pr.pid)], capture_output=True, text=True).stdout
                if int(rss.strip() or 0) > MEM_KB:
                    pr.kill()
                    status = 'memout'
                elif time.time() - t0 > TIMEOUT_S:
                    pr.kill()
                    status = 'timeout'
        txt = open(out + '/gen.log').read()
        if status is None and 'Dataset stored' in txt and 'No goals found' not in txt:
            return out, f'ok {strat} seed {seed} ({time.time() - t0:.0f}s)'
        print(f'{family}/{problem}', strat, seed, status or 'no goal', flush=True)
    return out, 'FAIL'


def sha(path):
    """Content key of a state: its DOT lines, sorted (robust to line order)."""
    return hashlib.sha1(''.join(sorted(open(path).read().splitlines(True))).encode()).hexdigest()


def build_instance(pdir, save=True):
    """Builds pdir/new3.pt from the raw tree in pdir/out; -> (states, groups, global pairs, labelled states) or None."""
    csvp = glob.glob(pdir + '/out/NN/Training/*/*_depth_*.csv')
    if not csvp or not os.path.exists(pdir + '/fluent_names.csv'):
        return None
    with open(csvp[0]) as fh:
        rows = list(csv.DictReader(fh))
    base = os.path.dirname(csvp[0])
    names = encoding.read_names(pdir + '/fluent_names.csv')
    assert os.path.exists(pdir + '/goal_ops.csv'), pdir + ': no goal_ops.csv (run deep with --ranker_encoding)'
    ops = encoding.read_ops(pdir + '/goal_ops.csv')
    fp = lambda p: os.path.join(base, 'RawFiles/hash_merged', os.path.basename(p))
    h = {r['File Path']: float(r['Distance From Goal']) for r in rows if float(r['Distance From Goal']) < 1e5}
    if len(set(h.values())) < 2:
        return None
    kids = collections.defaultdict(list)
    for r in rows:
        q = r['File Path Predecessor']
        if q and q != r['File Path'] and r['File Path'] in h and r['File Path'] not in kids[q]:
            kids[q].append(r['File Path'])
    rng = random.Random(0)
    groups = [ks for ks in kids.values() if len(ks) >= 2 and len({h[k] for k in ks}) > 1]
    rng.shuffle(groups)
    groups = groups[:MAXG]
    keys = list(h)
    glob_ = []
    for _ in range(MAXP * 20):
        if len(glob_) >= MAXP or len(keys) < 2:
            break
        a, b = rng.sample(keys, 2)
        if h[a] != h[b]:
            glob_.append((a, b))
    agents = encoding.agent_ranks([fp(p) for p in keys[:50]])
    need = sorted({s for g in groups for s in g} | {s for pr in glob_ for s in pr})
    idx = {p: i for i, p in enumerate(need)}
    out = {'family': pdir.rstrip('/').split('/')[-2], 'problem': os.path.basename(pdir.rstrip('/')),
           'graphs': [encoding.graph(fp(p), names, agents, ops) for p in need], 'ops': ops,
           'h': [h[p] for p in need], 'groups': [[idx[s] for s in g] for g in groups],
           'glob': [(idx[a], idx[b]) for a, b in glob_],
           'dist': {sha(fp(p)): h[p] for p in h if os.path.exists(fp(p))}, 'agents': agents, 'names': names}
    if save:
        torch.save(out, pdir + '/new3.pt')
    return len(need), len(groups), len(glob_), len(out['dist'])
