"""gnn_handler_plus.src — thin override layer on top of gnn_handler.

This package contains ONLY the code that diverges from the baseline
(`lib/gnn_handler`).  Everything else — DOT parsing, PyG conversion, the
DistanceEstimator model, ONNX wrappers, data pipeline — is imported from the
baseline, never copied.

Mechanism: this `src` package extends its own module search path with the
baseline's `src/` directory.  Submodules that exist here (sample_prep,
training, export) resolve locally; everything else (`src.utils`,
`src.model`, `src.models.*`, `src.preprocessing`) falls through to
`lib/gnn_handler/src/`.  Zero edits to the baseline are required, and the
baseline keeps working standalone.
"""
from pathlib import Path

_BASELINE_SRC = Path(__file__).resolve().parents[2] / "gnn_handler" / "src"
if not _BASELINE_SRC.is_dir():  # pragma: no cover
    raise ImportError(
        f"gnn_handler baseline not found at {_BASELINE_SRC}; "
        "gnn_handler_plus must live next to gnn_handler under lib/."
    )
__path__.append(str(_BASELINE_SRC))
