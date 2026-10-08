"""Graph encoding of a planner state, as read by the network.

Input: one state in deep's dataset DOT format written with --ranker_encoding
(goal tree, designated worlds, belief edges labelled by agent, an edge with
label 4 from every world to each fluent true in it), plus the instance's
fluent_names.csv (fluent id -> grounded name) and goal_ops.csv (goal-tree node
id -> operator).

Nodes: every node of the DOT plus one node per object named in a fluent.
Node type (vocabulary, vocab.json): eps (label 0 node), root (node 1), W
(world), P (designated world), gop:<operator> (goal-tree node with an
operator; gop without), agent, obj, or pred:<predicate> for a fluent.
With anonymisation (used by every final model) every pred:<name> type is
mapped to the single type 'fluent' (anon_map), so a model never sees
predicate names.

Edges (typed, every edge also added reversed with type 2t+1): belief edges
between worlds by agent rank (B0..B7, the agents' DOT labels sorted as
integers), togoal (label 2), tostate (label 3), holds (label 4), goal (any
other label: goal structure), and fluent-to-argument edges by argument
position (A0..A3).

The vocabulary is global and grows when a new type is met (vid); the packaged
vocab.json holds every type of the seven benchmark domains, and the model's
embedding table has NV = 64 rows.
"""

import csv
import json
import os
import re

import torch

EDGE = re.compile(r'^\s*"?(-?[0-9A-Za-z_]+)"?\s*->\s*"?(-?[0-9A-Za-z_]+)"?\s*\[\s*label\s*=\s*"?(-?\d+)"?\s*\]')
BASE = ['pad', 'eps', 'root', 'W', 'P', 'gop', 'agent', 'obj']
ET = [f'B{i}' for i in range(8)] + ['togoal', 'tostate', 'holds', 'goal'] + [f'A{i}' for i in range(4)]
DEFAULT_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

vocab = {}        # node type -> id (insertion order = id order)
vocab_path = ''   # file it was loaded from


def load_vocab(path: str = DEFAULT_VOCAB):
    """(Re)loads the vocabulary in place, so every module sees the same dict."""
    global vocab_path
    vocab.clear()
    if os.path.exists(path):
        with open(path) as fh:
            vocab.update(json.load(fh))
    else:
        vocab.update({k: i for i, k in enumerate(BASE)})
    vocab_path = path


def save_vocab(path: str):
    with open(path, 'w') as fh:
        json.dump(vocab, fh, indent=0)


def vid(k: str) -> int:
    if k not in vocab:
        vocab[k] = len(vocab)
    return vocab[k]


load_vocab()


def world(t: str) -> bool:
    """World nodes have ids above 10^6 in absolute value."""
    return abs(int(t)) > 10 ** 6


def read_names(path: str):
    with open(path) as fh:
        return {r['id']: r['name'] for r in csv.DictReader(fh)}


def read_ops(path: str):
    """goal_ops.csv -> {node id: operator}; {} when the file is missing (a goal without operators writes it empty)."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r['id']: r['op'] for r in csv.DictReader(fh)}


def dot_edges(path: str):
    with open(path) as fh:
        return [m.groups() for m in (EDGE.match(l) for l in fh) if m]


def agent_ranks(dot_paths):
    """Agent label -> rank: the labels of world-to-world edges in these DOT files, sorted as integers."""
    ids = set()
    for p in dot_paths:
        for u, v, l in dot_edges(p):
            if world(u) and world(v):
                ids.add(l)
    return {a: i for i, a in enumerate(sorted(ids, key=int))}


def graph(path, names, agents, ops=None):
    """One state -> (x [N] node types, edge_index [2, E], edge types [E], designated world indices)."""
    E = dot_edges(path)
    pointed = {v for u, v, l in E if l == '3'}
    nodes = list(dict.fromkeys([x for u, v, l in E for x in (u, v)] + list(names)))
    objs = sorted({o for nm in names.values() for o in nm.split('_')[1:]})
    nodes += ['obj:' + o for o in objs]
    ix = {n: i for i, n in enumerate(nodes)}

    def nt(n):
        if n.startswith('obj:'):
            return 'obj'
        if world(n):
            return 'P' if n in pointed else 'W'
        if n == '0':
            return 'eps'
        if n == '1':
            return 'root'
        if n in names:
            return 'pred:' + names[n].split('_')[0]
        if n in agents:
            return 'agent'
        if ops and n in ops:
            return 'gop:' + ops[n]
        return 'gop'

    x = torch.tensor([vid(nt(n)) for n in nodes], dtype=torch.long)
    src, dst, typ = [], [], []

    def add(u, v, t):
        src.extend([ix[u], ix[v]])
        dst.extend([ix[v], ix[u]])
        typ.extend([2 * ET.index(t), 2 * ET.index(t) + 1])

    for u, v, l in E:
        if world(u) and world(v):
            add(u, v, f'B{min(agents.get(l, 7), 7)}')
        elif l == '2':
            add(u, v, 'togoal')
        elif l == '3':
            add(u, v, 'tostate')
        elif l == '4':
            add(u, v, 'holds')
        else:
            add(u, v, 'goal')
    for fid, nm in names.items():
        for i, o in enumerate(nm.split('_')[1:]):
            add(fid, 'obj:' + o, f'A{min(i, 3)}')
    return (x, torch.tensor([src, dst], dtype=torch.long), torch.tensor(typ, dtype=torch.long),
            torch.tensor([ix[p] for p in pointed], dtype=torch.long))


def anon_map():
    """Vocabulary id -> id with every pred:<name> mapped to the single 'fluent' type."""
    f = vid('fluent')
    n = max(vocab.values()) + 1
    m = torch.arange(n)
    for k, i in vocab.items():
        if k.startswith('pred:'):
            m[i] = f
    return m


def data(g):
    """(x, edge_index, edge types, designated) -> PyTorch Geometric Data with the designated-world mask pmask."""
    from torch_geometric.data import Data
    x, ei, ea, pt = g
    d = Data(x=x, edge_index=ei, edge_attr=ea)
    m = torch.zeros(x.size(0), dtype=torch.bool)
    m[pt] = True
    d.pmask = m
    return d


def prep(g, amap):
    """A stored graph -> network input: anonymised node types, as Data."""
    x, ei, ea, pt = g
    return data((amap[x], ei, ea, pt))
