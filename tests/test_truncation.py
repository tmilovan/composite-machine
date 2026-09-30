"""The dimension cap is a default, and an explicit request overrides it.

MAX_ACTIVE_DIMS guards against dimension explosion in deep composition chains:
sin(atan(sin(atan(x)))) reaches 100k+ dimensions without it.  As a DEFAULT that
is right, and a caller who said nothing about depth should get the guard.

A caller who wrote `terms=300` has said something about depth.  Returning 60 and
saying nothing answers a question they did not ask, and it is the silent-wrong
shape rather than an honest refusal: before this was fixed, `sqrt(h - h*h,
terms=300)` and `terms=1000` returned byte-identical 60-term series, so a
convergence study flat-lined at 3.24e-03 and looked like a fact about the
mathematics.

_derivative_scope already took this position for extraction -- "a global order
cap set by the caller is an economy, not a ceiling on what an extraction may ask
for" -- and these tests hold the same rule for a direct call.

The second half covers the warning at the four places a term is actually
dropped.  Only where something is REMOVED, never where a bound is merely
recorded, or it would fire on every sin() in the library and be worth nothing.
"""
import math
import warnings

import pytest

import composite.composite_lib as cl
from composite.composite_lib import R, h, sqrt, sin, cos, atan, antiderivative
from composite.backends import config


#: The library default, restated because the fixture has to SET it.  Nine other
#: test modules assign cl.MAX_ACTIVE_DIMS = 10**9 at module import and never put
#: it back, so by the time this module runs the cap is whatever ran first and
#: the truncation path is untested -- two TypeErrors have hidden there.  Saving
#: and restoring is not enough: these tests were green alone and six of them
#: failed in the full suite, inheriting 10**9.
DEFAULT_CAP = 60


@pytest.fixture(autouse=True)
def _default_cap():
    """Put the cap at its default for the test, and back afterwards."""
    config.use_dict()
    cl.set_max_order(None)
    old = cl.MAX_ACTIVE_DIMS
    cl.MAX_ACTIVE_DIMS = DEFAULT_CAP
    yield
    cl.MAX_ACTIVE_DIMS = old
    cl.set_max_order(None)


def test_the_library_default_is_what_this_module_assumes():
    """If the default moves, DEFAULT_CAP above is stale and every threshold in
    this file is measuring the wrong thing."""
    import re
    src = open(cl.__file__).read()
    declared = int(re.search(r"^MAX_ACTIVE_DIMS = (\d+)", src, re.M).group(1))
    assert declared == DEFAULT_CAP, \
        f"library default is {declared}, this module assumes {DEFAULT_CAP}"


def _truncations(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn()
        return result, sum("dropped" in str(x.message) for x in w)


# --- an explicit terms= is a request ---------------------------------------

@pytest.mark.parametrize("terms", [100, 300, 1000])
def test_an_explicit_terms_is_honoured_past_the_cap(terms):
    assert cl.MAX_ACTIVE_DIMS == DEFAULT_CAP, "this test is about exceeding the default"
    s, dropped = _truncations(lambda: sqrt(h - h * h, terms=terms))
    assert len(s.coeffs_dict()) == terms, \
        f"asked {terms}, got {len(s.coeffs_dict())}"
    assert dropped == 0, "nothing should be truncated when the depth was requested"


def test_the_cap_is_put_back_afterwards():
    sqrt(h - h * h, terms=500)
    assert cl.MAX_ACTIVE_DIMS == DEFAULT_CAP, "the lift must not leak past the call"


def test_terms_given_positionally_counts_too():
    """sqrt(x, 300), not just sqrt(x, terms=300) -- the decorator reads both."""
    assert len(sqrt(h - h * h, 300).coeffs_dict()) == 300


def test_the_default_path_is_untouched():
    """Every terms= default in the module is 12 or 15 against a cap of 60, so
    the override is a no-op unless someone asks for more."""
    s, dropped = _truncations(lambda: sin(h))
    assert dropped == 0
    assert len(s.coeffs_dict()) <= DEFAULT_CAP
    assert cl.MAX_ACTIVE_DIMS == DEFAULT_CAP


def test_the_guard_still_bites_when_no_depth_was_asked_for():
    """The reason the cap exists.  Nobody asked for depth here, so it applies."""
    deep, dropped = _truncations(
        lambda: sin(atan(sin(atan(sin(atan(h + R(0.5))))))))
    assert dropped > 0, "a deep chain with no terms= must still be capped"
    assert len(deep.coeffs_dict()) <= DEFAULT_CAP


def test_the_requested_depth_actually_buys_accuracy():
    """The point of honouring it: the answer must keep improving.  Newton's pi
    integrated to 1 instead of 1/4 converges like n^(-3/2), so each 10x in terms
    is worth about 1.5 digits -- and it flat-lined before this was fixed."""
    errs = []
    for terms in (30, 300, 3000):
        s = sqrt(h - h * h, terms=terms)
        errs.append(abs(8 * antiderivative(s).eval_taylor(1.0) - math.pi))
    assert errs[0] > errs[1] > errs[2], f"not improving: {errs}"
    assert errs[2] < 1e-5, f"3000 terms should reach 1e-5, got {errs[2]:.1e}"


# --- the warning fires where a term is dropped, and only there --------------

def test_truncation_warns_when_the_order_cap_drops_terms():
    cl.set_max_order(None)
    big = sqrt(h - h * h, terms=40)
    cl.set_max_order(4)
    _, dropped = _truncations(lambda: big * big)
    assert dropped > 0, "set_max_order dropped terms and said nothing"


def test_truncation_warns_when_max_active_dims_drops_terms():
    _, dropped = _truncations(
        lambda: sin(atan(sin(atan(sin(atan(h + R(0.5))))))))
    assert dropped > 0


def test_no_warning_when_nothing_is_dropped():
    """A bound recorded is not a term removed.  If this starts failing, the
    warning has become noise and everyone will filter it out."""
    for label, fn in (("sin(h)", lambda: sin(h)),
                      ("cos(h)", lambda: cos(h)),
                      ("h*cos(h)/sin(h)", lambda: h * cos(h) / sin(h)),
                      ("sqrt(h - h*h)", lambda: sqrt(h - h * h))):
        _, dropped = _truncations(fn)
        assert dropped == 0, f"{label} warned with nothing to drop"


def test_a_lone_transcendental_is_not_touched_by_set_max_order():
    """Documenting what the knob does NOT do, because I assumed otherwise and
    reported a cap as mattering when it did not: set_max_order applies at
    _order_cap, which products pass through and a bare sin() does not."""
    cl.set_max_order(None)
    uncapped = len(sin(h).coeffs_dict())
    cl.set_max_order(4)
    assert len(sin(h).coeffs_dict()) == uncapped
