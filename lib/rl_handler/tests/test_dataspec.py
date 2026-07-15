"""The reuse guard. NON-NEGOTIABLE: a discard=0 run must REFUSE discard=0.4 data.

This is the guard against the exact mistake that motivated the whole data rework.
The original fingerprint (mode;strict;depth;max_creation) omitted the discard
factor, so a faithful run could symlink biased data, pass the check, and report
clean self-consistent numbers on trees whose optimal path had been deleted.

Every value is passed IN: `make_dataspec` has no defaults, because a default here
would be a second opinion about how the data was generated. The generator
(`create_all_training_data.py`, driven by `final_launcher.sh`'s DEPTH_MAP) owns
those values; this module only fingerprints what it is told.
"""

from __future__ import annotations

import inspect

import pytest

from src.offline.dataspec import compatible, make_dataspec, parse_dataspec

# What a real run generates with. Stated here, not imported -- these tests pin the
# GUARD's behaviour, not the generator's current settings.
CLEAN = dict(discard_factor=0, max_creation=50000, max_generation=100000, seed=42)


def spec(mode="merged", strict="yes", depth_map="CC:25", **kw):
    return make_dataspec(mode, strict, depth_map, **{**CLEAN, **kw})


# ------------------------------------- THE test: cross-discard reuse ---------

def test_discard0_run_refuses_a_discard04_batch():
    """THE non-negotiable one."""
    ok, why = compatible(spec(discard_factor=0.4), spec(discard_factor=0))
    assert not ok, "a faithful run MUST NOT reuse biased-discard data"
    assert "discard" in why


def test_a_legacy_dataspec_without_discard_is_refused():
    """Old batches were generated at 0.4. An absent key must NOT read as 'fine' --
    that is the silent path this guard exists to close."""
    legacy = "mode=merged;strict=yes;depth=40;max_creation=50000"
    ok, why = compatible(legacy, spec())
    assert not ok
    assert "predates discard fingerprinting" in why
    assert "0.4" in why and "Regenerate" in why


def test_identical_specs_are_compatible():
    s = spec("separated", "yes")
    ok, why = compatible(s, s)
    assert ok and why is None


# ------------------------------------- the module owns no parameter ----------

def test_make_dataspec_has_no_defaults_for_generation_params():
    """A default here would be rl_handler holding a SECOND opinion about how data
    was generated -- exactly the drift that let DEPTH_BY_DOMAIN=25 outlive the
    generator's real setting. The caller that generated the data must say."""
    sig = inspect.signature(make_dataspec)
    for name in ("depth_map", "discard_factor", "max_creation", "max_generation",
                 "seed"):
        assert sig.parameters[name].default is inspect.Parameter.empty, (
            f"{name} must be required: this module fingerprints data, it does not "
            f"decide how data is made"
        )


def test_depth_map_accepts_the_launchers_string_verbatim():
    """final_launcher.sh builds DEPTH_MAP as `CC:25,SC:40,SCRich:40` and feeds the
    SAME string to create_all_training_data.py --depth-map. It must round-trip."""
    assert "depth_map=CC:25,SC:40,SCRich:40" in spec(depth_map="CC:25,SC:40,SCRich:40")
    assert "depth_map=CC:25,SC:40" in spec(depth_map={"CC": 25, "SC": 40})


# ------------------------------------- cross-depth reuse ---------------------

def test_cc_depth25_run_refuses_cc_depth40_data():
    """Depth is per-domain; a CC-25 run must not reuse CC-40 data."""
    ok, why = compatible(spec(depth_map="CC:40"), spec(depth_map="CC:25"))
    assert not ok and "depth_map" in why


def test_depth_map_is_per_domain_not_a_scalar():
    """A single depth= scalar cannot describe a mixed batch once CC=25 and SC=40."""
    assert "depth_map=CC:25,SCRich:40" in spec(depth_map={"CC": 25, "SCRich": 40})


def test_depth_map_delimiter_does_not_collide_with_the_field_separator():
    """REGRESSION: depth_map used ";" internally, which is the FIELD separator, so
    parse_dataspec truncated `depth_map=CC:25;SC:40` to `CC:25` and a CC-only run
    compared EQUAL to a CC+SC batch -- it would have reused it silently."""
    d = parse_dataspec(spec(depth_map="CC:25,SC:40"))
    assert d["depth_map"] == "CC:25,SC:40", "the whole map must survive parsing"
    assert d["max_creation"] == "50000", "fields after depth_map must still parse"


def test_a_mixed_batch_spec_differs_from_a_cc_only_one():
    assert spec(depth_map="CC:25") != spec(depth_map="CC:25,SC:40")
    ok, _ = compatible(spec(depth_map="CC:25,SC:40"), spec(depth_map="CC:25"))
    assert not ok, "a CC-only run must not reuse a CC+SC batch's spec blindly"


# ------------------------------------- the original fields still guard -------

@pytest.mark.parametrize("a,b", [
    (("merged", "yes"), ("separated", "yes")),      # representation differs
    (("merged", "yes"), ("merged", "no")),          # strong_equality differs
])
def test_mode_and_strict_still_guard(a, b):
    ok, why = compatible(spec(*a), spec(*b))
    assert not ok and why


def test_max_creation_guards():
    ok, _ = compatible(spec(max_creation=5000), spec(max_creation=50000))
    assert not ok


# ------------------------------------- the VISIT ceiling guards --------------

def test_max_generation_guards():
    """Two batches at different visit budgets are different data even at identical
    depth/discard: the budget changes how much of the space the DFS walks (measured
    on CC_2_3_4__pl_7 at depth 25, 100k -> 250k: 2.2x the states, 4.2x the goals)."""
    ok, why = compatible(spec(max_generation=100000), spec(max_generation=250000))
    assert not ok and "max_generation" in why


def test_a_spec_predating_the_visit_ceiling_is_refused():
    """Old specs carry no max_generation -- the invisible-knob problem. A missing
    field is a mismatch, never an 'assume it's fine'."""
    legacy = "mode=separated;strict=yes;discard=0;depth_map=CC:25;max_creation=50000"
    ok, why = compatible(legacy, spec("separated", "yes", depth_map="CC:25"))
    assert not ok and "max_generation" in why


# ------------------------------------- the SEED guards -----------------------

def test_seed_guards():
    """THE seed test. The generated tree is a seed-dependent DFS SAMPLE of the state
    space, so a different seed is a DIFFERENT SUBGRAPH with different labels on it --
    not a cosmetic difference. Measured on CC_2_2_3__pl_4 (true optimal 4, identical
    flags, seed alone varied): delta_root = 14 (seed 42) / 6 (43) / 7 (44). Reusing
    another seed's data is reusing different data."""
    ok, why = compatible(spec(seed=42), spec(seed=43))
    assert not ok, "a run MUST NOT reuse a batch generated at another seed"
    assert "seed" in why


def test_a_spec_predating_the_seed_is_refused():
    """Specs written before the seed was fingerprinted cannot say which sample they
    are. Absent means unknown, and unknown must never read as compatible."""
    legacy = ("mode=separated;strict=yes;discard=0;depth_map=CC:25;"
              "max_creation=50000;max_generation=100000")
    ok, why = compatible(legacy, spec("separated", "yes", depth_map="CC:25"))
    assert not ok and "seed" in why


def test_roundtrip_parse():
    d = parse_dataspec(spec("separated", "no", depth_map="CC:25,SC:40"))
    assert d["mode"] == "separated" and d["strict"] == "no" and d["discard"] == "0"
    assert d["max_creation"] == "50000"
    assert d["max_generation"] == "100000"
    assert d["seed"] == "42", "the last field must still parse"
