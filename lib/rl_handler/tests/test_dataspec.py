"""The reuse guard. NON-NEGOTIABLE: a discard=0 run must REFUSE discard=0.4 data.

This is the guard against the exact mistake that motivated the whole data rework.
The original fingerprint (mode;strict;depth;max_creation) omitted the discard
factor, so a faithful run could symlink biased data, pass the check, and report
clean self-consistent numbers on trees whose optimal path had been deleted.
"""

from __future__ import annotations

import pytest

from src.offline.dataspec import compatible, depth_map_for, make_dataspec, parse_dataspec


# ------------------------------------- THE test: cross-discard reuse ---------

def test_discard0_run_refuses_a_discard04_batch():
    """THE non-negotiable one."""
    have = make_dataspec("merged", "yes", ["CC"], discard_factor=0.4)
    want = make_dataspec("merged", "yes", ["CC"], discard_factor=0)
    ok, why = compatible(have, want)
    assert not ok, "a faithful run MUST NOT reuse biased-discard data"
    assert "discard" in why


def test_a_legacy_dataspec_without_discard_is_refused():
    """Old batches were generated at 0.4. An absent key must NOT read as 'fine' --
    that is the silent path this guard exists to close."""
    legacy = "mode=merged;strict=yes;depth=40;max_creation=50000"
    want = make_dataspec("merged", "yes", ["CC"])
    ok, why = compatible(legacy, want)
    assert not ok
    assert "predates discard fingerprinting" in why
    assert "0.4" in why and "Regenerate" in why


def test_identical_specs_are_compatible():
    s = make_dataspec("separated", "yes", ["CC"])
    ok, why = compatible(s, s)
    assert ok and why is None


# ------------------------------------- cross-depth reuse ---------------------

def test_cc_depth25_run_refuses_cc_depth40_data():
    """Depth is per-domain now; a CC-25 run must not reuse CC-40 data."""
    import src.offline.generation as g
    want = make_dataspec("merged", "yes", ["CC"])            # CC -> 25
    old = dict(g.DEPTH_BY_DOMAIN)
    try:
        g.DEPTH_BY_DOMAIN["CC"] = 40
        have = make_dataspec("merged", "yes", ["CC"])        # CC -> 40
    finally:
        g.DEPTH_BY_DOMAIN.clear(); g.DEPTH_BY_DOMAIN.update(old)
    ok, why = compatible(have, want)
    assert not ok and "depth_map" in why


def test_depth_map_is_per_domain_not_a_scalar():
    """A single depth= scalar cannot describe a mixed batch once CC=25 and SC=40."""
    dm = depth_map_for(["CC", "SC", "SCRich"])
    assert dm == {"CC": 25, "SC": 40, "SCRich": 40}
    spec = make_dataspec("merged", "yes", ["CC", "SCRich"])
    assert "depth_map=CC:25,SCRich:40" in spec


def test_depth_map_delimiter_does_not_collide_with_the_field_separator():
    """REGRESSION: depth_map used ";" internally, which is the FIELD separator, so
    parse_dataspec truncated `depth_map=CC:25;SC:40` to `CC:25` and a CC-only run
    compared EQUAL to a CC+SC batch -- it would have reused it silently."""
    d = parse_dataspec(make_dataspec("merged", "yes", ["CC", "SC"]))
    assert d["depth_map"] == "CC:25,SC:40", "the whole map must survive parsing"
    assert d["max_creation"] == "50000", "fields after depth_map must still parse"


def test_a_mixed_batch_spec_differs_from_a_cc_only_one():
    assert make_dataspec("merged", "yes", ["CC"]) != make_dataspec("merged", "yes", ["CC", "SC"])
    ok, _ = compatible(make_dataspec("merged", "yes", ["CC", "SC"]),
                       make_dataspec("merged", "yes", ["CC"]))
    assert not ok, "a CC-only run must not reuse a CC+SC batch's spec blindly"


# ------------------------------------- the original fields still guard -------

@pytest.mark.parametrize("a,b", [
    (("merged", "yes"), ("separated", "yes")),      # representation differs
    (("merged", "yes"), ("merged", "no")),          # strong_equality differs
])
def test_mode_and_strict_still_guard(a, b):
    ok, why = compatible(make_dataspec(*a, ["CC"]), make_dataspec(*b, ["CC"]))
    assert not ok and why


def test_max_creation_guards():
    ok, _ = compatible(make_dataspec("merged", "yes", ["CC"], max_creation=5000),
                       make_dataspec("merged", "yes", ["CC"], max_creation=50000))
    assert not ok


def test_roundtrip_parse():
    s = make_dataspec("separated", "no", ["CC", "SC"])
    d = parse_dataspec(s)
    assert d["mode"] == "separated" and d["strict"] == "no" and d["discard"] == "0"
    assert d["max_creation"] == "50000"
