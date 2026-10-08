"""The ranking network (configuration L5-ln of the experiments).

Embedding of the node type (NV = 64 types) and of the edge type (NE = 32), 5
GINE layers of width H = 96, each residual (h = h + relu(conv(h))) and
followed by a LayerNorm; readout [sum / 50, mean, mean over the designated
worlds] -> MLP -> one score per graph. Higher score = closer to the goal
(expanded first by the greedy search).

Checkpoints are dicts {'cfg': Net3 kwargs, 'state': state_dict, 'anon',
'drop_holds', 'ops'}; every final model has cfg = {'layers': 5, 'pool': 'all',
'norm': True}, anon = True, drop_holds = False, ops = True.
"""

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

H, NV, NE = 96, 64, 32
CFG = {'layers': 5, 'pool': 'all', 'norm': True}   # L5-ln


class Net3(nn.Module):
    def __init__(self, layers=5, pool='all', norm=True):
        super().__init__()
        assert pool == 'all', pool   # the only readout of the final models (the ablation's 'nomean' is not kept)
        self.pool = pool
        # Module creation order fixes the order of the random initialisation; keep it.
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(layers)]) if norm else None
        self.inp = nn.Embedding(NV, H)
        self.edge = nn.Embedding(NE, H)
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H)), edge_dim=H)
                                    for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, b):
        h = self.inp(b.x)
        e = self.edge(b.edge_attr)
        for i, c in enumerate(self.convs):
            h = h + torch.relu(c(h, b.edge_index, e))
            if self.norms is not None:
                h = self.norms[i](h)
        pm = b.pmask.float().unsqueeze(1)
        des = global_add_pool(h * pm, b.batch) / global_add_pool(pm, b.batch).clamp(min=1)
        parts = [global_add_pool(h, b.batch) / 50.0, global_mean_pool(h, b.batch), des]
        return self.out(torch.cat(parts, 1)).squeeze(1)


def load_checkpoint(path):
    """A checkpoint -> Net3 in eval mode, with its flags (anon, ops) as attributes."""
    d = torch.load(path, weights_only=False)
    assert not d.get('drop_holds', False), 'models without holds edges are not supported'
    net = Net3(**d['cfg'])
    net.load_state_dict(d['state'])
    net.eval()
    net.anon = d.get('anon', False)
    net.ops = d.get('ops', False)
    return net


def save_checkpoint(net, cfg, path):
    torch.save({'cfg': cfg, 'state': net.state_dict(), 'anon': True, 'drop_holds': False, 'ops': True}, path)
