"""power and ** at a real or rational exponent: the depth a scope asks for, and a
bound reported instead of a claim of exactness.

Two gaps, both older than the exp and ln work.  _series_pow (Miller's
recurrence, every real exponent but 0.5) ran its default 15 terms whatever
depth was asked, so power(x, 1.5) and x ** 1.5 stopped at order 14 under
_derivative_scope(50, 52).  _rational_power (a Fraction exponent) built the
binomial series to the right depth but reported complete_order None -- exact --
so a read past its last power came back 0.0.

Tolerances relative and fixed before running: 1e-13 against mpmath.
"""
from fractions import Fraction

import mpmath as mp
import pytest

import composite.composite_lib as cl
from composite.backends import config
from tests.series_check import check_series
from composite.composite_lib import R, ZERO, power

mp.mp.dps = 60


def _ref(s, a, n):
    return [float(c) for c in mp.taylor(lambda t: t ** mp.mpf(s), mp.mpf(a), n)]


@pytest.mark.parametrize("label, build", [
    ("power(x, 1.5)", lambda x: power(x, 1.5)),
    ("x ** 1.5", lambda x: x ** 1.5),
    ("x ** Fraction(3, 2)", lambda x: x ** Fraction(3, 2)),
])
def test_real_and_rational_exponent_follow_the_scope(label, build):
    with cl._derivative_scope(50, 52):
        r = build(R(1.6) + ZERO)
    K = min(int(r.complete_order), 50)
    print(f"\n  {label}: complete_order {r.complete_order}")
    assert r.complete_order >= 50
    ref = _ref(1.5, 1.6, 50)
    check_series(f"{label} at 1.6, orders 0..{K}", r.coeffs_dict(), {-n: ref[n] for n in range(K + 1)}, 1e-13)


def test_default_depth_unchanged():
    x = R(1.6) + ZERO
    got = (power(x, 2.5).complete_order, (x ** 2.5).complete_order)
    print(f"\n  power(x, 2.5), x ** 2.5 at default depth: complete_order {got} (want 14, 14)")
    assert got == (14, 14)


def test_rational_exponent_reports_its_bound():
    r = (R(1.6) + ZERO) ** Fraction(3, 2)
    d = r.coeffs_dict()
    deepest = max(-float(g) for g in d)
    print(f"\n  (1.6 + h) ** (3/2): complete_order {r.complete_order}, deepest order {deepest:.0f}")
    assert r.complete_order is not None and r.complete_order == deepest
    ref = _ref(1.5, 1.6, int(deepest))
    check_series("(1.6 + h) ** (3/2), default depth", d, {-n: ref[n] for n in range(int(deepest) + 1)}, 1e-13)


def test_monomial_stays_exact():
    r = ZERO ** Fraction(1, 2)
    print(f"\n  h ** (1/2) = {r}  complete_order {r.complete_order} (want None: nothing was expanded)")
    assert r.complete_order is None and r.coeffs_dict() == {-0.5: 1.0}


@pytest.fixture
def fractional_backend():
    before = config.get_backend()
    config.use_fractional_numpy()
    cl._refresh_constants()
    yield
    config.set_backend(before)
    cl._refresh_constants()


def test_rational_exponent_on_the_lattice_reports_its_bound(fractional_backend):
    h = cl.ZERO
    r = (h ** 3 + h ** 4) ** Fraction(1, 5)
    d = r.coeffs_dict()
    lead = max(d)
    print(f"\n  (h^3 + h^4) ** (1/5): lead grade {lead}, complete_order {r.complete_order} "
          f"(want 15 + 3/5 = 15.6)")
    assert lead == Fraction(-3, 5)
    assert abs(float(r.complete_order) - 15.6) < 1e-12
