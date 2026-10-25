"""CANCELLATION = "mirrored" (experimental, not the default).

As "convert_and_keep", except that a cancelled infinite grade k leaves its zero
at the mirror grade -k instead of 0, so depth(a - a) = depth(1/a - 1/a).
"""
import warnings

import pytest

from composite.composite_lib import R, ZERO, INF, ln
from tests.zero_modes import cancellation

warnings.simplefilter("ignore")

I2 = INF * INF

CASES = [
    # label, expression, terms under "mirrored", terms under "convert_and_keep"
    ("3 - 3", lambda: R(3) - R(3), {0: 0.0}, {0: 0.0}),
    ("3h - 3h", lambda: 3 * ZERO - 3 * ZERO, {-1: 0.0}, {-1: 0.0}),
    ("3INF - 3INF", lambda: 3 * INF - 3 * INF, {-1: 0.0}, {0: 0.0}),
    ("3INF^2 - 3INF^2", lambda: 3 * I2 - 3 * I2, {-2: 0.0}, {0: 0.0}),
    ("(3INF + 1) - 3INF", lambda: (3 * INF + 1) - 3 * INF, {0: 1.0, -1: 0.0}, {0: 1.0}),
    ("(3INF - 3INF) + 1", lambda: (3 * INF - 3 * INF) + 1, {0: 1.0, -2: 1.0}, {0: 1.0, -1: 1.0}),
    ("<|3|_2 |3|_1> - itself", lambda: (3 * I2 + 3 * INF) - (3 * I2 + 3 * INF),
     {-1: 0.0, -2: 0.0}, {0: 0.0}),
    ("INF^1.5 - INF^1.5", lambda: INF ** 1.5 - INF ** 1.5, {-1.5: 0.0}, {0: 0.0}),
]


@pytest.mark.parametrize("rule", ["mirrored", "convert_and_keep"])
@pytest.mark.parametrize("label,fn,mirrored,keep", CASES, ids=[c[0] for c in CASES])
def test_where_a_cancelled_grade_lands(rule, label, fn, mirrored, keep):
    want = mirrored if rule == "mirrored" else keep
    with cancellation(rule):
        got = fn().coeffs_dict()
    assert got == want, "[%s] %s: got %r, want %r" % (rule, label, got, want)


def test_log_axis_mirrors_to_the_reciprocal_grade():
    with cancellation("mirrored"):
        a = 3 * INF * ln(INF)
        got = {tuple(float(x) for x in d): v for d, v in (a - a).coeffs_dict().items()}
        recip = {tuple(float(x) for x in d) for d in (1 / a).coeffs_dict()}
    assert got == {(-1.0, -1.0): 0.0}, "got %r, want {(-1, -1): 0.0}" % got
    assert set(got) == recip, "cancelled at %r, reciprocal at %r" % (set(got), recip)


@pytest.mark.parametrize("a", [3 * INF, 3 * I2, R(3), 3 * ZERO], ids=["INF", "INF^2", "real", "h"])
def test_depth_of_a_cancellation_is_inversion_invariant(a):
    with cancellation("mirrored"):
        x, y = (a - a).coeffs_dict(), ((1 / a) - (1 / a)).coeffs_dict()
    assert set(x) == set(y), "a - a %r, 1/a - 1/a %r" % (x, y)


def test_a_cancelled_infinity_needs_one_more_division_to_reach_grade_0():
    # f = 3/(x-2) - 3/(x-2) is zero everywhere.  Under convert_and_keep one
    # division by (x-2) already puts 1 on the standard part; under mirrored
    # that takes two.
    for rule, want in (("convert_and_keep", ({0: 1.0}, {1: 1.0})),
                       ("mirrored", ({-1: 1.0}, {0: 1.0}))):
        with cancellation(rule):
            x = R(2) + ZERO
            f = 3 / (x - 2) - 3 / (x - 2)
            got = ((f / (x - 2)).coeffs_dict(), (f / ((x - 2) * (x - 2))).coeffs_dict())
        assert got == want, "[%s] f/(x-2), f/(x-2)^2: got %r, want %r" % (rule, got, want)
