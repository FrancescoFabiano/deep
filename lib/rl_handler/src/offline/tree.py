# Moved to lib/deep_nn (shared by the GNN and RL handlers); this name stays importable.
import sys
from deep_nn import tree as _m
sys.modules[__name__] = _m
