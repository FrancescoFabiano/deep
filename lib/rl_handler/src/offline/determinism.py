"""Reproducibility. Call `set_determinism(seed)` before building any model.

WHY THIS MODULE EXISTS
CUDA scatter_add uses atomics, which sum in nondeterministic order. The GNN uses
scatter in every message-passing layer and in global_*_pool, so
`torch.manual_seed(s)` does NOT pin a result: the identical config produced
held-out regret 252.3 and 93.0 on two runs, and a 5-seed sweep put the with-hash
arm at 98.3 +/- 76.1. A single-run A/B here measures the RNG, not the change.

Two runs of the same config MUST agree bit-for-bit, or every comparison in this
project is unfalsifiable. Verified: with this enabled the forward is identical
across repeats; without it, it is not.

`CUBLAS_WORKSPACE_CONFIG` must be set BEFORE the first CUDA context is created,
so this module sets it at import time and callers must import it early.
"""

from __future__ import annotations

import os

# Must precede CUDA init; :4096:8 is the cuBLAS-documented deterministic setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_SEED = 0


def set_determinism(seed: int = DEFAULT_SEED, strict: bool = True) -> None:
    """Pin every RNG and force deterministic kernels.

    `strict=True` makes PyTorch RAISE on any op with no deterministic
    implementation, rather than silently continuing -- silence is what let the
    nondeterminism go unnoticed for a whole session. Use strict=False only to
    find out WHICH op is the offender, never to ship a result.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def determinism_report() -> dict:
    """What a run must record so its numbers can be reproduced."""
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
