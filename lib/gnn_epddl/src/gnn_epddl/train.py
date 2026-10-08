"""Training of one model (one run): pairwise ranking loss, validation by search, one checkpoint every eval_every epochs.

Data: the new3.pt of the run's families' training instances (split.csv label
'train'); node types anonymised (every predicate is 'fluent').
Training items: every pair of siblings with different distances in a sibling
group, plus the global pairs; target y = 1 when the first state is closer to
the goal. Loss: binary cross-entropy with logits on score(a) - score(b)
(pairwise logistic loss), Adam (lr 1e-3), batches of 64 items, the item list
shuffled every epoch with random.Random(seed); torch.manual_seed(seed) before
the network is built.

Every eval_every epochs (and at the last epoch) the network is copied to the
CPU, saved as <ckpt>/<run>/s<seed>-e<epoch>.pt, and every validation problem
of the run's families (split.csv label 'validation') is searched with it
(search.py; cap 3000 expansions / 300 s for gossip, 2000 / 120 s otherwise),
vjobs searches in parallel. One row per evaluation is appended to the log CSV:
run, families, seed, epoch, loss (mean batch loss of that epoch), groups,
pairs, val_top1 (sibling top-1 on validation training sets, nan when the
validation instances have none), val_solved, val_logsum (sum of log10 nodes,
an unsolved problem counts 2 x cap), val (per problem: nodes/plan length or X),
ckpt. select.py picks the checkpoint of a run from these rows.
"""

import copy
import csv
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor

import torch
from torch import nn

from . import encoding, layout
from .model import CFG, Net3, save_checkpoint

CAPS = {'gossip': (3000, 300.0)}   # (expansions, seconds) per validation search; other families (2000, 120)


def caps(fam):
    return CAPS.get(fam, (2000, 120.0))


def load(paths, amap):
    """-> list of (Data list, h list, groups, glob pairs), one per new3.pt."""
    out = []
    for p in paths:
        d = torch.load(p, weights_only=False)
        out.append(([encoding.prep(g, amap) for g in d['graphs']], d['h'], d['groups'], d['glob']))
    return out


def items(insts):
    """-> sibling groups [(Data list, h list)] and global pairs [(a, b, y)]."""
    G, P = [], []
    for D, h, groups, glob_ in insts:
        for g in groups:
            G.append(([D[i] for i in g], [h[i] for i in g]))
        for a, b in glob_:
            P.append((D[a], D[b], 1.0 if h[a] < h[b] else 0.0))
    return G, P


def sib_pairs(G):
    P = []
    for ds, hs in G:
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                if hs[i] != hs[j]:
                    P.append((ds[i], ds[j], 1.0 if hs[i] < hs[j] else 0.0))
    return P


def pair_loss(net, chunk, lf, device):
    from torch_geometric.data import Batch
    A = Batch.from_data_list([a for a, _, _ in chunk]).to(device)
    B = Batch.from_data_list([b for _, b, _ in chunk]).to(device)
    y = torch.tensor([y for *_, y in chunk]).to(device)
    return lf(net(A) - net(B), y)


def top1(net, G):
    """Share of sibling groups whose highest-scored child has the smallest distance."""
    from torch_geometric.data import Batch
    if not G:
        return float('nan')
    net.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(G), 64):
            chunk = G[i:i + 64]
            flat = [d for ds, _ in chunk for d in ds]
            s = net(Batch.from_data_list(flat))
            k = 0
            for ds, hs in chunk:
                sc = s[k:k + len(ds)]
                k += len(ds)
                ok += hs[int(sc.argmax())] == min(hs)
    return ok / len(G)


def vsearch(a):
    """One validation search in a worker process: rebuild the network on the CPU, search, -> (fam, prob, cap, solved, nodes, plan length)."""
    from .search import search
    fam, tier, prob, cfg, state, tag, deep, instances, work_root, vocab_path = a
    torch.set_num_threads(1)
    if vocab_path != encoding.vocab_path:
        encoding.load_vocab(vocab_path)
    net = Net3(**cfg)
    net.load_state_dict(state)
    net.eval()
    net.anon = True
    net.ops = True
    cap, wall = caps(fam)
    r = search(deep, instances, fam, tier, prob, net, node_cap=cap, wall=wall, work_root=work_root, tag=tag)
    return fam, prob, cap, r['status'] == 'solved', r['nodes'], r['plan_length']


def train_run(name, fams, rows, data, instances, deep, ckpt_dir, log_csv, epochs=50, eval_every=10, threads=4,
              device='cpu', seed=0, vjobs=8, work_root=None):
    """Trains run `name` on families `fams` with split rows `rows`."""
    torch.set_num_threads(threads)
    amap = encoding.anon_map()
    paths = [p for f in fams for p in layout.inst_paths(rows, data, f, 'train')]
    G, P = items(load(paths, amap))
    VG, _ = items(load([p for f in fams for p in layout.inst_paths(rows, data, f, 'validation')], amap))
    vprob = [(f, r['tier'], r['problem']) for f in fams for r in rows if r['family'] == f and r['label'] == 'validation']
    cfg = dict(CFG)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = Net3(**cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.BCEWithLogitsLoss()
    work = sib_pairs(G) + P
    os.makedirs(f'{ckpt_dir}/{name}', exist_ok=True)
    print(name, 'training instances', len(paths), 'sibling groups', len(G), 'pairs', len(work), 'validation problems',
          len(vprob), flush=True)
    # spawn, not fork: a forked worker cannot initialise CUDA once the parent has used it
    pool = ProcessPoolExecutor(vjobs, mp_context=multiprocessing.get_context('spawn'))
    for ep in range(1, epochs + 1):
        net.train()
        rng.shuffle(work)
        tot, n = 0.0, 0
        for i in range(0, len(work), 64):
            opt.zero_grad()
            loss = pair_loss(net, work[i:i + 64], lf, device)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        if ep % eval_every and ep != epochs:
            continue
        cpu = copy.deepcopy(net).cpu()
        path = f'{ckpt_dir}/{name}/s{seed}-e{ep}.pt'
        save_checkpoint(cpu, cfg, path)
        state = {k: v.clone() for k, v in cpu.state_dict().items()}
        jobs = [(f, t, p, cfg, state, f'f8-{name}-s{seed}-e{ep}-{p}', deep, instances, work_root, encoding.vocab_path)
                for f, t, p in vprob]
        res = list(pool.map(vsearch, jobs))
        solved = sum(ok for *_, ok, _, _ in res)
        logsum = sum(math.log10(max(nd, 1) if ok else 2 * cap) for _, _, cap, ok, nd, _ in res)
        per = ' '.join(f'{p}={nd}/{L}' if ok else f'{p}=X' for _, p, _, ok, nd, L in res)
        row = dict(run=name, families='+'.join(fams), seed=seed, epoch=ep, loss=round(tot / max(n, 1), 4), groups=len(G),
                   pairs=len(work), val_top1=round(top1(cpu, VG), 3), val_solved=f'{solved}/{len(vprob)}',
                   val_logsum=round(logsum, 3), val=per, ckpt=path)
        with open(log_csv, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(row)
        print(*row.values(), sep='\t', flush=True)
    pool.shutdown()
