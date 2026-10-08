"""Checkpoint selection from the training log (rule fixed before any test run).

Per run (all seeds of the run together): the row with the most validation
problems solved, then the smallest val_logsum (sum of log10 nodes, unsolved =
2 x cap), then the later epoch; on a full tie the row that comes first in the
log is kept.
"""

import csv


def key(r):
    return int(r['val_solved'].split('/')[0]), -float(r['val_logsum']), int(r['epoch'])


def select(log_csv, runs=None):
    """-> {run: selected row}, for every run in the log (or only `runs`)."""
    best = {}
    with open(log_csv) as fh:
        for r in csv.DictReader(fh):
            if runs and r['run'] not in runs:
                continue
            k = key(r)
            if r['run'] not in best or k > best[r['run']][0]:
                best[r['run']] = (k, r)
    return {run: r for run, (_, r) in best.items()}


def write_selection(sel, out_csv, onnx_of=lambda run: ''):
    """Writes run, checkpoint, epoch, val_solved, val_logsum, val, onnx (sorted by run)."""
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'checkpoint', 'epoch', 'val_solved', 'val_logsum', 'val', 'onnx'])
        for run, r in sorted(sel.items()):
            w.writerow([run, r['ckpt'], r['epoch'], r['val_solved'], r['val_logsum'], r['val'], onnx_of(run)])
