__version__ = "0.0.1"

# lib/deep_nn is the data path shared with lib/rl_handler: make it importable
# wherever this package is (entry points, scripts and tests all import `src`).
import sys as _sys
from pathlib import Path as _Path

_LIB = str(_Path(__file__).resolve().parents[2])
if _LIB not in _sys.path:
    _sys.path.append(_LIB)
