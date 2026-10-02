"""A written zero discloses itself the same way however it is spelled.

`R(0)` used to return |1|_-1 already converted, which skipped `_r1` -- and `_r1`
is where the R1 warning fires and where `_denot` is recorded. So `x*x + R(0)`
silently read d(1) = 7 instead of 6, with no warning and `denotation_order is
None`, while `x*x + 0` with identical semantics warned twice and recorded the
order. The EXPLICIT spelling was the undisclosed one.

`R(0)` now returns the latent |0|_0, so both spellings convert at the point of use
and disclose identically. ZERO is not built through `R`, names the infinitesimal
rather than being a written zero, and still carries no denotation.
"""
import warnings

import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO
from composite.backends import config


@pytest.fixture(autouse=True)
def dict_backend():
    config.use_dict()
    cl._refresh_constants()
    yield
    config.use_sparse_dense()
    cl._refresh_constants()


def build_and_watch(build):
    """Build INSIDE the catch, or construction-time warnings are missed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = R(3) + ZERO
        value = cl._ensure_composite(build(x))
        first = value.d(1)
        categories = sorted({c.category.__name__ for c in caught})
    return first, value.denotation_order, categories


def test_R0_discloses_exactly_as_a_bare_zero_does():
    explicit = build_and_watch(lambda x: x * x + R(0))
    bare = build_and_watch(lambda x: x * x + 0)
    assert explicit == bare, "R(0) gave %r, bare 0 gave %r" % (explicit, bare)
    derivative, denot, categories = explicit
    assert derivative == 7.0, "got %r, want the denoted 7.0" % derivative
    assert denot == 1.0, "denotation_order is %r, want 1.0" % denot
    assert "NotConventionalWarning" in categories, categories
    assert "UserWarning" in categories, categories


def test_a_zero_free_expression_still_discloses_nothing():
    derivative, denot, categories = build_and_watch(lambda x: x * x)
    assert (derivative, denot, categories) == (6.0, None, []), \
        "a clean expression reported %r" % ((derivative, denot, categories),)


def test_R0_is_the_latent_expressed_zero():
    assert R(0).coeffs_dict() == {0: 0.0}, R(0).coeffs_dict()
    assert R(0.0).coeffs_dict() == {0: 0.0}, R(0.0).coeffs_dict()
    # still the same NUMBER as the infinitesimal: both read through R1
    assert R(0) == ZERO
    assert R(0) == 0.0


def test_ZERO_is_not_a_written_zero_and_carries_no_denotation():
    # It names the infinitesimal. Marking it would make every ordinary seed
    # `R(a) + ZERO` report a denotation it does not have.
    assert ZERO.coeffs_dict() == {-1: 1.0}, ZERO.coeffs_dict()
    assert ZERO.denotation_order is None
    seed = R(3) + ZERO
    assert seed.denotation_order is None, "the plain seed reported a denotation"
    assert (seed * seed).d(1) == 6.0


def test_the_conversion_still_happens_and_still_shifts_the_jet():
    # The fix is about DISCLOSURE, not about changing the arithmetic.
    assert (R(5) + R(0)).coeffs_dict() == {0: 5.0, -1: 1.0}, (R(5) + R(0)).coeffs_dict()
    assert (R(5) * R(0)).coeffs_dict() == {-1: 5.0}, (R(5) * R(0)).coeffs_dict()


def test_NOTHING_remains_the_silent_escape():
    # Composite({}) is absence, adds no order, and so has nothing to disclose.
    derivative, denot, categories = build_and_watch(lambda x: x * x + Composite({}))
    assert derivative == 6.0, derivative
    assert denot is None and categories == [], (denot, categories)
