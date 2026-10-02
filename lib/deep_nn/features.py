"""Model blocks shared by every network that consumes the planner contract."""

from __future__ import annotations

import torch
from torch import nn


class PointedEmbedding(nn.Module):
    """Adds a learned vector to the designated (pointed) worlds' node features.

    ``pointed_ids`` is the planner's 5th input (GraphTensor::pointed_ids), so
    every deployed network must consume it -- an input the traced graph does
    not use is dropped by the exporter, and the planner then rejects the
    model on input count.  The vector starts at zero: a checkpoint re-exported
    with this block scores exactly as before until it is trained further.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(dim)))
        self.dense_aggregation = False      # export-time switch, see deep_nn.dense

    def forward(self, x: torch.Tensor, pointed_ids: torch.Tensor) -> torch.Tensor:
        pointed_ids = pointed_ids.to(x.device)
        if self.dense_aggregation:
            from .dense import onehot_rows
            mark = onehot_rows(pointed_ids, x.size(0), x.dtype).sum(dim=1)   # Equal + ReduceSum
        else:
            ones = torch.ones(pointed_ids.numel(), dtype=x.dtype, device=x.device)
            mark = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            mark = mark.scatter_add(0, pointed_ids, ones)   # ONNX: ScatterElements(add)
        return x + mark.unsqueeze(1) * self.weight
