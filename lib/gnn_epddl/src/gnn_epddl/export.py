"""ONNX export of a checkpoint for deep's ranker (--ranker_model / --ranker_model_gpu).

The network is re-expressed with plain tensor operations: the edge types are
categorical, so each GINE layer's edge projection lin(edge embedding) becomes
a fixed [types, H] table looked up per edge (identical result, no E x H x H
product). Two forms of the per-node sum of incoming messages:
  standard : scatter_add over the edge targets (for the GPU build).
  fast     : edges sorted by target once per state, then per layer the
             cumulative sum of the sorted messages in float64, differenced at
             the segment boundaries (for the CPU: ONNX Runtime's ScatterElements
             dominated the CPU time).
Inputs: x int64 [N] (anonymised node types), edge_index int64 [2, E],
edge_attr int64 [E], pmask float [N] (1 for designated worlds); output
score [1] (higher = closer to the goal). Opset 17.

Each export is checked against the PyTorch Geometric network and then
against ONNX Runtime on sample graphs, and written with the sidecar
<out>.vocab: one line '<node type> <id>' per vocabulary entry, then
'#anon', '#drop_holds', '#scale 1000' and '#ops'.
"""

import glob

import torch
from torch import nn

from . import encoding
from .model import Net3

SCALE = 1000


class ExportNet(nn.Module):
    """Standard form: scatter_add aggregation."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, edge_index, edge_attr, pmask):
        n = self.net
        h = n.inp(x)
        src, dst = edge_index[0], edge_index[1]
        for i, c in enumerate(n.convs):
            table = c.lin(n.edge.weight) if c.lin is not None else n.edge.weight
            ee = table[edge_attr]
            msg = torch.relu(h[src] + ee)
            agg = torch.zeros_like(h).scatter_add(0, dst.unsqueeze(1).expand(-1, h.size(1)), msg)
            h = h + torch.relu(c.nn(agg + (1 + c.eps) * h))
            if n.norms is not None:
                h = n.norms[i](h)
        return _readout(n, h, pmask)


class ExportNetSeg(nn.Module):
    """Fast form: segment sums over target-sorted edges (float64 running sums)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, edge_index, edge_attr, pmask):
        n = self.net
        h = n.inp(x)
        src, dst = edge_index[0], edge_index[1]
        perm = torch.argsort(dst)
        src, dst, ea = src[perm], dst[perm], edge_attr[perm]
        cnt = torch.zeros(h.size(0), dtype=torch.long).scatter_add(0, dst, torch.ones_like(dst))   # in-degree
        end = torch.cumsum(cnt, 0)
        start = end - cnt
        for i, c in enumerate(n.convs):
            table = c.lin(n.edge.weight) if c.lin is not None else n.edge.weight
            msg = torch.relu(h[src] + table[ea])
            cs = torch.cat([torch.zeros(1, h.size(1), dtype=torch.float64), torch.cumsum(msg.double(), 0)], 0)
            agg = (cs[end] - cs[start]).float()
            h = h + torch.relu(c.nn(agg + (1 + c.eps) * h))
            if n.norms is not None:
                h = n.norms[i](h)
        return _readout(n, h, pmask)


def _readout(n, h, pmask):
    pm = pmask.unsqueeze(1)
    parts = [h.sum(0, keepdim=True) / 50.0, h.mean(0, keepdim=True), (h * pm).sum(0, keepdim=True) / pm.sum().clamp(min=1)]
    return n.out(torch.cat(parts, 1)).reshape(1)


KINDS = {'standard': ExportNet, 'fast': ExportNetSeg}


def sample_graphs(family_data, n_inst=3, per_inst=20):
    """Check graphs: the first per_inst graphs of the first n_inst new3.pt (sorted by path) of one family's data folder."""
    G = []
    for p in sorted(glob.glob(f'{family_data}/*/new3.pt'))[:n_inst]:
        G += torch.load(p, weights_only=False)['graphs'][:per_inst]
    return G


def write_vocab_sidecar(path, anon, drop, ops):
    with open(path, 'w') as fh:
        for k, i in encoding.vocab.items():
            fh.write(f'{k} {i}\n')
        fh.write(f'#anon {int(anon)}\n#drop_holds {int(drop)}\n#scale {SCALE}\n#ops {int(ops)}\n')


def export(ckpt, out, graphs, kind='standard'):
    """Exports ckpt to out (+ out.vocab); graphs: sample (x, edge_index, edge types, designated) for the checks."""
    import onnxruntime as ort
    from torch_geometric.data import Batch
    d = torch.load(ckpt, weights_only=False)
    net = Net3(**d['cfg'])
    net.load_state_dict(d['state'])
    net.eval()
    anon, drop, ops = d.get('anon', False), d.get('drop_holds', False), d.get('ops', False)
    assert not drop, 'models without holds edges are not supported'
    ex = KINDS[kind](net).eval()
    amap = encoding.anon_map() if anon else None

    def feed(g):
        x, ei, ea, pt = g
        if amap is not None:
            x = amap[x]
        pm = torch.zeros(x.size(0))
        pm[pt] = 1
        return x, ei, ea, pt, pm

    worst = 0.0
    for g in graphs:
        x, ei, ea, pt, pm = feed(g)
        with torch.no_grad():
            a = net(Batch.from_data_list([encoding.data((x, ei, ea, pt))])).item()
            b = ex(x, ei, ea, pm).item()
        worst = max(worst, abs(a - b))
    print(f'torch: PyG vs plain ops, max |diff| over {len(graphs)} graphs = {worst:.2e}')
    # the trace input is the first sample graph as stored (types not anonymised; the trace does not depend on values)
    x, ei, ea, pt = graphs[0]
    pm = torch.zeros(x.size(0))
    pm[pt] = 1
    torch.onnx.export(ex, (x, ei, ea, pm), out, input_names=['x', 'edge_index', 'edge_attr', 'pmask'], output_names=['score'],
                      dynamic_axes={'x': {0: 'N'}, 'edge_index': {1: 'E'}, 'edge_attr': {0: 'E'}, 'pmask': {0: 'N'}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(out)
    worst = 0.0
    for g in graphs:
        x, ei, ea, pt, pm = feed(g)
        o = sess.run(None, {'x': x.numpy(), 'edge_index': ei.numpy(), 'edge_attr': ea.numpy(), 'pmask': pm.numpy()})[0][0]
        worst = max(worst, abs(float(o) - ex(x, ei, ea, pm).item()))
    print(f'onnx: plain ops vs ONNX Runtime, max |diff| = {worst:.2e}')
    write_vocab_sidecar(out + '.vocab', anon, drop, ops)
    print('wrote', out, out + '.vocab')
