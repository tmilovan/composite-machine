"""power() with an expressed zero inside s*ln(x), and its constant fast path.

power(x, s) is exp(s * ln(x)).  Two of its inputs put an EXPRESSED zero into
that product: a written exponent 0, and a base of exactly 1, whose ln is the
computed zero ln(1).  R1 converts either one, so the result is the denoted value
-- c**h for power(c, 0), exp(s*h) for (x/x)**s -- not a plain 1.

A constant base and exponent take math.pow instead of exp(s*ln(x)), which was
one ulp off: power(R(2), 0.5) was 1.414213562373095 where sqrt(R(2)) is
1.4142135623730951, so power(2, 0.5) - sqrt(2) left -2.2e-16 and the
cancellation residue a - a = a*h never formed.  The first version of that fast
path also took the two cases above and flattened them to 1; the rest of the
suite passed it, which is what this file is for.
"""
import math
import warnings

import pytest

import composite.composite_lib as cl
from composite.composite_lib import R, ZERO, power, sqrt
from composite.backends import config

TOL = 1e-12


@pytest.fixture(autouse=True, params=["sparse_dense", "dict"])
def backend(request):
    getattr(config, "use_" + request.param)()
    cl._refresh_constants()
    warnings.simplefilter("ignore", cl.NotConventionalWarning)
    yield request.param
    config.use_sparse_dense()
    cl._refresh_constants()


def close(got, want):
    return abs(got - want) <= TOL * max(1.0, abs(want))


@pytest.mark.parametrize("spelling", ["power", "**"])
def test_base_one_is_ln1_and_converts(spelling):
    x = R(0.7) + ZERO
    v = power(x / x, 1.5) if spelling == "power" else (x / x) ** 1.5
    # ln(x/x) = ln(1) is an expressed zero -> h, so this is exp(1.5 h)
    want = [1.0, 1.5, 2.25, 3.375]
    got = [v.st()] + [v.d(k) for k in (1, 2, 3)]
    assert all(close(g, w) for g, w in zip(got, want)), \
        "(x/x)**1.5 via %s: got %r, want %r" % (spelling, got, want)
    assert v.denotation_order is not None, \
        "the converted ln(1) must be disclosed, denotation_order is None"


@pytest.mark.parametrize("spelling", ["power", "**"])
def test_written_zero_exponent_converts(spelling):
    v = power(R(3), 0.0) if spelling == "power" else R(3) ** 0.0
    # the written 0 -> h, so this is 3**h = exp(h ln 3)
    L = math.log(3.0)
    want = [1.0, L, L * L, L ** 3]
    got = [v.st()] + [v.d(k) for k in (1, 2, 3)]
    assert all(close(g, w) for g, w in zip(got, want)), \
        "3**0.0 via %s: got %r, want 3**h, %r" % (spelling, got, want)


def test_constant_fast_path_is_correctly_rounded():
    got = [power(R(2), 0.5).st(), (R(2) ** 0.5).st(), power(R(2), R(0.5)).st()]
    want = math.pow(2.0, 0.5)
    assert all(g == want for g in got), \
        "power(2, 0.5) three spellings: got %r, want %r exactly" % (got, want)
    assert sqrt(R(2)).st() == want, "sqrt(R(2)) %r vs %r" % (sqrt(R(2)).st(), want)


def test_equal_constants_cancel_into_the_residue():
    d = power(R(2), 0.5) - sqrt(R(2))
    # a - a = a*h: the residue is sqrt(2) one grade down, not float dust
    got = d.coeffs_dict()
    assert set(got) == {-1} and close(got[-1], math.sqrt(2.0)), \
        "power(2, 0.5) - sqrt(2): got %r, want {-1: %r}" % (got, math.sqrt(2.0))


def test_negative_base_still_refuses():
    with pytest.raises(ValueError):
        power(R(-2), 0.5)


def test_zero_base_still_converts():
    got = power(R(0), 0.5).coeffs_dict()
    assert len(got) == 1 and close(list(got.values())[0], 1.0) \
        and float(list(got)[0]) == -0.5, \
        "power(R(0), 0.5): got %r, want h**0.5 = {-0.5: 1.0}" % got


def test_seeded_base_still_takes_the_series():
    x = R(0.7) + ZERO
    v = power(x, 0.5)
    want = [math.sqrt(0.7), 0.5 / math.sqrt(0.7), -0.25 * 0.7 ** -1.5]
    got = [v.st(), v.d(1), v.d(2)]
    assert all(close(g, w) for g, w in zip(got, want)), \
        "power(0.7 + h, 0.5): got %r, want %r" % (got, want)
