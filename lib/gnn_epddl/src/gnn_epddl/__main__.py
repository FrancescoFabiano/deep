"""Command line: python -m gnn_epddl <command> [options]; one command per step of the pipeline.

  generate  write the generated problems (all families or --family)
  check     parse / duplicate / BFS check of generated problems (CSV on stdout)
  data      deep's search trees of the split's training instances, then their training sets (new3.pt)
  build     training sets (new3.pt) from existing raw trees
  train     train one run (per family, all families, a family subset or one sub-family)
  select    pick each run's checkpoint from the training log
  export    ONNX export of a checkpoint (standard = GPU form, fast = CPU form) with its .vocab sidecar
  search    one greedy best-first search with a checkpoint through deep's expand server

deep is always a parameter (--deep); every deep run uses --ranker_encoding.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import encoding, layout


def _families(rows):
    return sorted({r['family'] for r in rows})


def cmd_generate(a):
    from . import generators
    for fam, n in generators.write(a.out, a.family).items():
        print(f'{fam}: {n} problems')


def cmd_check(a):
    from . import check
    w = csv.DictWriter(sys.stdout, fieldnames=check.FIELDS)
    w.writeheader()
    for row in check.check_family(a.instances, a.tier, a.family, a.problems, jobs=a.jobs, deep=a.deep, plank=a.plank,
                                  benchmarks=a.benchmarks, timeout=a.timeout, mem_gb=a.mem_gb, strategy=a.strategy):
        w.writerow(row)
        sys.stdout.flush()


def _build(pdir, trim):
    from . import dataset
    import shutil
    r = dataset.build_instance(pdir)
    print(pdir, '(states, groups, global pairs, labelled states) =', r, flush=True)
    if trim and r is not None:
        shutil.rmtree(os.path.join(pdir, 'out'), ignore_errors=True)
    return r


def _save_vocab_if_grown(n0):
    if len(encoding.vocab) == n0:
        return
    new = list(encoding.vocab)[n0:]
    if os.path.abspath(encoding.vocab_path) == os.path.abspath(encoding.DEFAULT_VOCAB):
        sys.exit(f'new node types {new}: pass --vocab <writable copy of vocab.json> to every command')
    encoding.save_vocab(encoding.vocab_path)
    print('vocabulary', encoding.vocab_path, 'now has', len(encoding.vocab), 'types; new:', new)


def cmd_data(a):
    from . import dataset
    rows = [r for r in layout.read_split(a.split) if r['label'] == 'train' and (not a.family or r['family'] in a.family)]
    n0 = len(encoding.vocab)

    def one(r):
        out, status = dataset.generate_tree(a.deep, a.instances, r['tier'], r['family'], r['problem'], a.data)
        print(r['family'], r['tier'], r['problem'], status, flush=True)
        return out, status

    with ThreadPoolExecutor(a.jobs) as ex:
        done = list(ex.map(one, rows))
    for out, status in done:
        if status != 'FAIL':
            _build(out, a.trim)
    _save_vocab_if_grown(n0)


def cmd_build(a):
    n0 = len(encoding.vocab)
    for pdir in a.dirs:
        _build(pdir.rstrip('/'), a.trim)
    _save_vocab_if_grown(n0)


def cmd_train(a):
    from .train import train_run
    rows = layout.read_split(a.split)
    if a.family:
        fams, name = [a.family], f'pf-{a.family}'
    elif a.all:
        fams, name = _families(rows), 'final-all'
    elif a.families:
        fams, name = a.families.split(','), 'final-sub'
    else:
        fam, sub = a.subfamily
        fams, name = [fam], f'sf-{fam}-{sub}'
        rows = layout.subfamily_split(rows, fam, sub)
    assert set(fams) <= set(_families(rows)), fams
    train_run(a.name or name, fams, rows, a.data, a.instances, a.deep, a.ckpt, a.log, epochs=a.epochs,
              eval_every=a.eval_every, threads=a.threads, device=a.device, seed=a.seed, vjobs=a.vjobs, work_root=a.work)


def cmd_select(a):
    from .selection import select, write_selection
    sel = select(a.log, a.runs)
    write_selection(sel, a.out)
    for run, r in sorted(sel.items()):
        print('selected', run, r['ckpt'], 'epoch', r['epoch'], r['val_solved'], r['val_logsum'], flush=True)


def cmd_export(a):
    from .export import export, sample_graphs
    graphs = sample_graphs(a.sample_data)
    assert graphs, f'no new3.pt under {a.sample_data}'
    export(a.checkpoint, a.out, graphs, kind=a.kind)


def cmd_search(a):
    from .model import load_checkpoint
    from .search import search, write_row
    import torch
    torch.set_num_threads(a.threads)
    net = load_checkpoint(a.checkpoint)
    r = search(a.deep, a.instances, a.family, a.tier, a.problem, net, node_cap=a.node_cap, wall=a.wall,
               cap_gb=a.cap_gb, work_root=a.work, tag=a.tag)
    print(*(f'{k}={v}' for k, v in r.items()))
    if a.out:
        write_row(a.out, r)


def parser():
    p = argparse.ArgumentParser(prog='python -m gnn_epddl', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vocab', default=encoding.DEFAULT_VOCAB,
                   help='node-type vocabulary (default: the packaged vocab.json, which covers the seven benchmark domains)')
    sub = p.add_subparsers(dest='cmd', required=True)
    deep = dict(default='cmake-build-release-nn/bin/deep', help='deep binary (default: %(default)s)')

    s = sub.add_parser('generate', help='write the generated problems')
    s.add_argument('--out', required=True, help='instances root (<out>/<tier>/<family>/problems/)')
    s.add_argument('--family', nargs='*', help='only these families (default: all)')
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser('check', help='check generated problems; CSV on stdout')
    s.add_argument('--instances', required=True)
    s.add_argument('--tier', required=True)
    s.add_argument('--family', required=True)
    s.add_argument('problems', nargs='*', help='default: every problem of the family in that tier')
    s.add_argument('--deep', **deep)
    s.add_argument('--plank', required=True, help='plank binary (parser check)')
    s.add_argument('--benchmarks', required=True, help='benchmark root <tier>/<family>/problems/ (duplicate check)')
    s.add_argument('--timeout', type=int, default=120)
    s.add_argument('--mem-gb', type=float, default=6.0)
    s.add_argument('--strategy', default='BFS')
    s.add_argument('--jobs', type=int, default=3)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser('data', help="search trees and training sets of the split's training instances")
    s.add_argument('--instances', required=True)
    s.add_argument('--split', required=True)
    s.add_argument('--data', required=True, help='output root (<data>/<family>/<short tier>-<problem>/)')
    s.add_argument('--family', nargs='*', help='only these families')
    s.add_argument('--deep', **deep)
    s.add_argument('--jobs', type=int, default=4, help='deep runs in parallel')
    s.add_argument('--trim', action='store_true', help='delete the raw tree (out/) once new3.pt is built')
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser('build', help='training sets (new3.pt) from existing raw trees')
    s.add_argument('dirs', nargs='+', help='instance data folders (with out/, fluent_names.csv, goal_ops.csv)')
    s.add_argument('--trim', action='store_true')
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser('train', help='train one run')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--family', help='per-family model (run pf-<family>)')
    g.add_argument('--all', action='store_true', help='every family of the split (run final-all)')
    g.add_argument('--families', help='comma-separated family subset (run final-sub, or --name)')
    g.add_argument('--subfamily', nargs=2, metavar=('FAMILY', 'SUB'), help='one sub-family (run sf-<family>-<sub>)')
    s.add_argument('--name', help='run name (overrides the default)')
    s.add_argument('--split', required=True)
    s.add_argument('--data', required=True)
    s.add_argument('--instances', required=True)
    s.add_argument('--deep', **deep)
    s.add_argument('--ckpt', required=True, help='checkpoint root (<ckpt>/<run>/s<seed>-e<epoch>.pt)')
    s.add_argument('--log', required=True, help='training log CSV (rows appended)')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--epochs', type=int, default=50)
    s.add_argument('--eval-every', type=int, default=10)
    s.add_argument('--threads', type=int, default=4)
    s.add_argument('--vjobs', type=int, default=8, help='validation searches in parallel')
    s.add_argument('--device', default='cpu', help='training device (validation searches run on the CPU)')
    s.add_argument('--work', help='work folder of the validation searches (default: <tmp>/gnn-epddl-search)')
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser('select', help="pick each run's checkpoint from the training log")
    s.add_argument('--log', required=True)
    s.add_argument('--out', required=True, help='selection CSV')
    s.add_argument('--runs', nargs='*', help='only these runs')
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser('export', help='ONNX export of a checkpoint')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--out', required=True, help='<model>.onnx (the sidecar <model>.onnx.vocab is written next to it)')
    s.add_argument('--kind', choices=['standard', 'fast'], default='standard',
                   help='standard: scatter_add (GPU); fast: segment sums over sorted edges (CPU)')
    s.add_argument('--sample-data', required=True,
                   help='a family data folder for the checks (the experiments used <data>/search-and-rescue)')
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser('search', help='one validation-style search with a checkpoint')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--instances', required=True)
    s.add_argument('--family', required=True)
    s.add_argument('--tier', required=True)
    s.add_argument('--problem', required=True)
    s.add_argument('--deep', **deep)
    s.add_argument('--node-cap', type=int, default=None)
    s.add_argument('--wall', type=float, default=600.0)
    s.add_argument('--cap-gb', type=float, default=6.0)
    s.add_argument('--threads', type=int, default=1)
    s.add_argument('--work')
    s.add_argument('--tag', default='net')
    s.add_argument('--out', help='append the result row to this CSV')
    s.set_defaults(fn=cmd_search)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.vocab != encoding.vocab_path:
        encoding.load_vocab(a.vocab)
    a.fn(a)


if __name__ == '__main__':
    main()
