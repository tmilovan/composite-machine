"""limit() and the rounding residue of a cancellation floats cannot make exact.

(sin(tan x) - tan(sin x)) / (asin(atan x) - atan(asin x)) at 0 is 1.  Both sides
vanish to x^7; the numerator's x^3 and x^5 coefficients come out 2.8e-17 and
1.4e-17 instead of 0, because sin(tan x) and tan(sin x) reach them by different
float arithmetic.  Dividing by a denominator that starts at x^7 makes them pole
terms of about 8e-16, and limit() read their sign as divergence and returned
-INF.  It now drops terms above grade 0 that are all at most
LIMIT_ROUNDING_RESIDUE of the finite part, with a RoundingResidueWarning.
Each check prints got against known.
"""
import math
import warnings

import pytest

import composite.composite_lib as cl
from composite.composite_lib import (INF, LIMIT_ROUNDING_RESIDUE, R, RoundingResidueWarning,
                                     asin, atan, limit, sin, tan)


def run(f, at, dir="both"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = limit(f, as_x_to=at, dir=dir)
    return got, [w for w in caught if issubclass(w.category, RoundingResidueWarning)]


def test_seventh_order_cancellation_reads_one():
    got, warned = run(lambda x: (sin(tan(x)) - tan(sin(x))) / (asin(atan(x)) - atan(asin(x))), 0)
    print(f"\n  (sin tan - tan sin)/(asin atan - atan asin) at 0: got {got!r}  known 1   warned: {bool(warned)}")
    print(f"  {warned[0].message if warned else ''}")
    assert isinstance(got, float) and abs(got - 1.0) <= 1e-12
    assert len(warned) == 1


def test_ordinary_limits_are_untouched():
    for label, f, want in (("sin(x)/x", lambda x: sin(x) / x, 1.0),
                           ("(x^2 - 4)/(x - 2) at 2", lambda x: (x * x - R(4)) / (x - R(2)), 4.0)):
        at = 2 if "at 2" in label else 0
        got, warned = run(f, at)
        print(f"\n  {label}: got {got!r} known {want}  warned: {bool(warned)}")
        assert abs(got - want) <= 1e-12 and not warned


@pytest.mark.parametrize("c", [1.0, 1e-6, 1e-11])
def test_a_real_pole_is_still_divergence(c):
    got, warned = run(lambda x: R(c) / x + R(1), 0)
    print(f"\n  {c}/x + 1 at 0: got {got!r}  known +inf   warned: {bool(warned)}")
    assert got == INF and not warned


def test_a_pole_below_the_bound_is_read_as_residue_and_said():
    """The stated cost: a genuine infinite part below the bound is read the same
    way.  It is not silent."""
    c = LIMIT_ROUNDING_RESIDUE / 10
    got, warned = run(lambda x: R(c) / x + R(1), 0)
    print(f"\n  {c}/x + 1 at 0: got {got!r} (a real pole, below the bound)  warned: {bool(warned)}")
    assert got == 1.0 and len(warned) == 1


SEVENTH = lambda u: (sin(tan(u)) - tan(sin(u))) / (asin(atan(u)) - atan(asin(u)))


@pytest.mark.parametrize("dir", ["+", "-"])
def test_seventh_order_cancellation_one_sided(dir):
    got, warned = run(SEVENTH, 0, dir=dir)
    print(f"\n  the 7th-order limit from {dir}: got {got!r}  known 1   warned: {bool(warned)}")
    assert isinstance(got, float) and abs(got - 1.0) <= 1e-12


def test_seventh_order_cancellation_off_the_origin():
    got, warned = run(lambda x: SEVENTH(x - R(2)), 2)
    print(f"\n  the same limit in u = x - 2, at x = 2: got {got!r}  known 1   warned: {bool(warned)}")
    assert isinstance(got, float) and abs(got - 1.0) <= 1e-12
