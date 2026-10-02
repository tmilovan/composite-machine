"""`to_ieee754`: the one well-defined exit to a system with an additive identity.

No single substitution `h = value` does this.  The two halves of the dimension
axis want opposite limits, and the tests that matter are the two that show why:
at the smallest representable float a cancellation leaves a subnormal crumb
rather than a zero, and at `h = 0` exactly a pole raises rather than giving the
infinity IEEE754 itself gives for `1/0`.
"""
import math
import sys

import pytest

import composite.composite_lib as cl
from composite.composite_lib import (
    Composite, R, ZERO, StandardPartUndefinedError, ln, sqrt)
from composite.backends import config

MIN_SUBNORMAL = 5e-324


@pytest.fixture(autouse=True)
def dict_backend():
    """The log-axis cases need vector dimensions."""
    config.use_dict()
    cl._refresh_constants()
    yield
    config.use_sparse_dense()
    cl._refresh_constants()


def test_the_three_regimes():
    cases = [
        # bounded: the standard part survives
        ("x*x at 3",      (R(3) + ZERO) * (R(3) + ZERO),      9.0),
        ("3 + h",         R(9) + ZERO,                        9.0),
        # infinitesimal: a true zero, which is the whole point
        ("ZERO",          ZERO,                               0.0),
        ("R(6) - R(6)",   R(6) - R(6),                        0.0),
        ("sqrt(ZERO)",    sqrt(ZERO),                         0.0),
        ("x*x - 9 at 3",  (R(3) + ZERO) * (R(3) + ZERO) - R(9), 0.0),
        # unbounded: signed infinity, as IEEE754 answers 1/0
        ("1/ZERO",        R(1) / ZERO,                        math.inf),
        ("-1/ZERO",       R(-1) / ZERO,                       -math.inf),
        ("1/ZERO**2",     R(1) / (ZERO * ZERO),               math.inf),
    ]
    for label, value, want in cases:
        got = value.to_ieee754()
        assert got == want, "%s: %s gave %r, want %r" % (label, value, got, want)


def test_the_log_axis_keeps_its_sign():
    # ln of a small positive number is a large NEGATIVE one, so the sign has to
    # come from the coefficient and not from the axis.
    assert ln(ZERO).to_ieee754() == -math.inf, ln(ZERO)
    assert ln(R(1) / ZERO).to_ieee754() == math.inf, ln(R(1) / ZERO)
    # 1/ln(h) reaches 0, so it is infinitesimal despite living on the log axis
    assert (R(1) / ln(ZERO)).to_ieee754() == 0.0, R(1) / ln(ZERO)
    # and where both axes appear, the pole dominates the logarithm
    both = R(1) / ZERO + ln(R(1) / ZERO)
    assert both.to_ieee754() == math.inf, both


def test_nothing_is_nan_and_an_expressed_zero_is_zero():
    # The distinction the whole system rests on has to survive the exit.
    # float has no absence, so nan is the nearest honest answer; 0.0 would
    # claim NOTHING was a zero.
    assert math.isnan(Composite({}).to_ieee754()), Composite({})
    assert Composite({0: 0.0}).to_ieee754() == 0.0, Composite({0: 0.0})


def test_it_agrees_with_st_wherever_st_is_defined():
    for value in (R(3) + ZERO, (R(3) + ZERO) * (R(3) + ZERO),
                  R(-2.5) + ZERO * R(7), (R(3) + ZERO) * (R(3) + ZERO) - R(9)):
        assert value.to_ieee754() == float(value.st()), \
            "%s: projection %r vs st %r" % (value, value.to_ieee754(), value.st())


def test_the_smallest_float_is_NOT_a_substitute():
    # The measurement this method exists because of: at h = 5e-324 a
    # cancellation comes back as a subnormal, so `6 - 6` is not 0 and the
    # additive identity is not restored.
    residue = R(6) - R(6)
    assert residue.coeffs_dict() == {-1: 6.0}, residue
    crumb = 6.0 * MIN_SUBNORMAL
    assert crumb != 0.0, "expected a representable subnormal, got %r" % crumb
    assert crumb == pytest.approx(2.9643938750474793e-323, rel=1e-12), crumb
    assert residue.to_ieee754() == 0.0, \
        "the projection must give a true zero, got %r" % residue.to_ieee754()


def test_float_stays_strict_on_the_unbounded():
    # Widening __float__ would make the silent-coercion hazard worse, so the
    # exception is deliberate: it is what catches `math.sin(composite)`.
    with pytest.raises(StandardPartUndefinedError):
        float(R(1) / ZERO)
    assert (R(1) / ZERO).to_ieee754() == math.inf


@pytest.mark.parametrize("backend", ["dict", "sparse_dense", "dense_series"])
def test_the_power_axis_projects_the_same_on_every_backend(backend):
    getattr(config, "use_" + backend)()
    cl._refresh_constants()
    assert ((R(3) + ZERO) * (R(3) + ZERO)).to_ieee754() == 9.0
    assert (R(6) - R(6)).to_ieee754() == 0.0
    assert (R(1) / ZERO).to_ieee754() == math.inf
