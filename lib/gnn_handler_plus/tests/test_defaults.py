"""Guard: the all-defaults run of gnn_handler_plus IS the node-economy champion.

Evidence: the single-convention table in REPORT_plus_investigations.md
("Cross-investigation synthesis"), batch0/CC t600 — plus W=1.0 is the
node-economy champion: 26/26 coverage, 2,874 total nodes (13.2x under BFS's
37,927, 7.4x under baseline w=1.0's 21,128), quality +1.73 avg / +6 worst.
That run used every plus flag at its argparse default, so the table below
must stay byte-identical to the parser defaults; any drift silently changes
what "plus, no flags" measures and breaks comparability with every number
in the report.

Run:  .venv/bin/python lib/gnn_handler_plus/tests/test_defaults.py
(or via pytest; no torch/data needed — only the parser is inspected).
"""
import ast
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1] / "__main__.py"

# Champion configuration = the synthesis table's "plus W=1.0" row.
CHAMPION_DEFAULTS = {
    "dynamic_max_depth": True,
    "include_unreachable": True,
    "unreachable_cap": 0.25,
    "unreachable_loss_weight": 1.0,
    "ckpt_metric": "spearman",
    "heuristic_weight": 1.0,
}


def _parser_defaults():
    """Extract add_argument defaults from __main__.py without importing it.

    Importing __main__.py pulls in torch and the baseline package; the AST
    walk keeps this test runnable anywhere (the defaults are all literals).
    """
    tree = ast.parse(_MAIN.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        dest = node.args[0].value.lstrip("-").replace("-", "_")
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                defaults[dest] = kw.value.value
    return defaults


def test_defaults_match_champion():
    defaults = _parser_defaults()
    errors = []
    for dest, expected in CHAMPION_DEFAULTS.items():
        if dest not in defaults:
            errors.append(f"--{dest.replace('_', '-')}: no literal default "
                          f"found in __main__.py (expected {expected!r})")
        elif defaults[dest] != expected or type(defaults[dest]) is not type(expected):
            errors.append(f"--{dest.replace('_', '-')}: default is "
                          f"{defaults[dest]!r}, champion is {expected!r}")
    assert not errors, (
        "gnn_handler_plus defaults drifted from the node-economy champion "
        "(REPORT_plus_investigations.md, single-convention table):\n  "
        + "\n  ".join(errors)
    )


def test_defaults_parse_to_champion():
    """End-to-end check through a real ArgumentParser when importable.

    Skipped (without failing) when torch/baseline imports are unavailable —
    the AST test above is the portable guard.
    """
    sys.path.insert(0, str(_MAIN.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_plus_main", _MAIN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError as exc:
        print(f"  (parser-level check skipped: {exc})")
        return
    finally:
        sys.path.remove(str(_MAIN.parent))

    old_argv = sys.argv
    sys.argv = ["__main__.py"]
    try:
        args = mod.parse_args()
    finally:
        sys.argv = old_argv
    for dest, expected in CHAMPION_DEFAULTS.items():
        actual = getattr(args, dest)
        assert actual == expected and type(actual) is type(expected), (
            f"--{dest.replace('_', '-')}: parsed default {actual!r} != "
            f"champion {expected!r}"
        )


if __name__ == "__main__":
    test_defaults_match_champion()
    print("OK  AST defaults == champion table")
    test_defaults_parse_to_champion()
    print("OK  parsed (no-flag) args == champion table")
