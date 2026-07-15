"""The baseline to beat: a two-head delta model. NO CLIP, NO FREE PARAMETER.

Regressing delta with a sentinel for inf is the wrong formulation, not a
formulation with a hard constant -- and any sentinel is a hyperparameter we chose,
so any result would invite "you picked a bad clip", which is unfalsifiable. So
sterility becomes a CLASSIFICATION problem (which it is) and distance a regression
over viable nodes only. Both heads train on exact labels we already have:

    viability head : p(v) = P[delta(v) < inf]     BCE, exact labels
    distance head  : d_hat(v) ~ delta(v)          MSE, VIABLE nodes only
    score          : s(v) = -d_hat(v) + logit p(v)

The score is a single scalar per slot, so this exports through the SAME ONNX
contract (higher score = expanded sooner). A sterile node drives logit p -> -inf
and is ranked last; among viable nodes logit p is near-constant and -d_hat does
the ordering.

This is a genuinely strong baseline -- it is what a competent person would build
given delta -- and that is the point. It shares the encoder, the context stack and
the head width with the RL model byte-for-byte; ONLY THE LOSS DIFFERS.

WHAT IT CAN AND CANNOT DO
It optimises viability directly, so it gets viability_auc for free and should TIE
the RL policy there. It can rank viable-before-sterile and near-before-far. What
neither head represents is what happens AFTER a wrong pick: the beam dumped into
R, the argmin evicted, recovery being a random draw against a growing reservoir.
Only a critic bootstrapping through real stochastic transitions has that, because
the cost is literally in the returns it fits. That is the whole comparison.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ..offline.scoring import viability_dominant_score
from .frontier_policy import FrontierPolicyNetwork, _build_mlp


class TwoHeadBaselineNetwork(FrontierPolicyNetwork):
    """FrontierPolicyNetwork with a 2-output head. Same encoder, same context.

    `max_delta` bounds the distance head. It is DATA-DERIVED (max delta over the
    training instances), not tuned -- see `forward` for why it is load-bearing.
    """

    def __init__(self, *args, mlp_depth: int = 2, max_delta: float = 64.0, **kwargs):
        super().__init__(*args, mlp_depth=mlp_depth, **kwargs)
        head_in = self.policy_head[0].in_features
        hidden = self.policy_head[0].out_features
        # out_dim 2: [viability_logit, raw_distance]
        self.policy_head = _build_mlp(head_in, hidden, mlp_depth, 2)
        self.max_delta = float(max_delta)

    def heads(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """(viability_logit, d_hat) per candidate slot -- the training surface.

        d_hat is squashed into [0, max_delta]. delta IS bounded by the data, so
        this is a range constraint on a quantity we know the range of -- not a
        sentinel for `inf`, which is the thing we refuse to invent.
        """
        mask = kwargs.pop("mask", None)
        h = self.head_features(*args, mask=mask, **kwargs)
        out = self.policy_head(h)
        p_logit = out[..., 0]
        d_hat = torch.sigmoid(out[..., 1]) * self.max_delta
        return p_logit, d_hat

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Combined score, masked -- the SAME contract as the RL model.

        Viability must strictly dominate distance; see offline/scoring.py for why
        (this is one of THREE places the same inversion bug appeared, so the
        arithmetic lives in exactly one helper and is not inlined here).
        """
        mask = kwargs.pop("mask", None)
        p_logit, d_hat = self.heads(*args, mask=mask, **kwargs)
        score = viability_dominant_score(torch.sigmoid(p_logit), d_hat, self.max_delta)
        if mask is not None:
            score = score.masked_fill(~mask.to(torch.bool), -1e9)
        return score


def two_head_loss(
    p_logit: torch.Tensor,
    d_hat: torch.Tensor,
    viable: torch.Tensor,      # float 1.0/0.0
    delta: torch.Tensor,       # finite where viable; ignored elsewhere
    distance_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """BCE on viability (all slots) + MSE on distance (VIABLE slots only).

    The MSE is masked rather than clipped: a sterile node has no finite distance,
    so asking the head to predict one would be inventing a target.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(p_logit, viable)
    m = viable > 0.5
    if m.any():
        mse = nn.functional.mse_loss(d_hat[m], delta[m])
    else:
        mse = d_hat.sum() * 0.0
    loss = bce + distance_weight * mse
    return loss, {
        "bce": float(bce.detach()),
        "mse": float(mse.detach()),
        "loss": float(loss.detach()),
        "n_viable": int(m.sum()),
        "n_total": int(viable.numel()),
    }
