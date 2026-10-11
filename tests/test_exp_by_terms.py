"""exp on the power axis, built one factor per term of the argument.

exp(a + sum_g |c_g|_g) = e^a * prod_g exp(|c_g|_g), and a single term's powers
stay single terms, (|c|_g)^n = |c^n|_(gn), so each factor is written out.  These
pin the coefficients against closed forms or the mpmath oracle, the
completeness bound, the denotation marker, the unchanged log-axis route, and
the underflow case the old loop got wrong.

Tolerances are relative and fixed before running: 1e-14 where the answer is a
closed form reached by an n-step recurrence (about n rounding steps, n <= 52),
1e-13 where several factors are multiplied and compared with mpmath.
"""
import math

import mpmath as mp
import pytest

from tests.series_check import check_series
from composite.composite_lib import Composite, R, ZERO, exp, sin, sqrt, ln

mp.mp.dps = 40


def _taylor(f, x0, n):
    return [float(c) for c in mp.taylor(f, x0, n)]


def test_seeded_value_e_over_n_factorial():
    r = exp(R(1) + ZERO)
    check_series("exp(1 + h)", r.coeffs_dict(),
           {-n: math.e / math.factorial(n) for n in range(15)}, 1e-14)
    assert r.complete_order == 14


def test_monomial_53_terms():
    r = exp(R(0.5) * ZERO, 53)
    check_series("exp(0.5 h), 53 terms", r.coeffs_dict(),
           {-n: 0.5 ** n / math.factorial(n) for n in range(53)}, 1e-14)
    assert r.complete_order == 52


def test_negative_coefficient_with_standard_part():
    r = exp(R(3) - R(4) * ZERO)
    check_series("exp(3 - 4h)", r.coeffs_dict(),
           {-n: math.exp(3.0) * (-4.0) ** n / math.factorial(n) for n in range(15)}, 1e-14)


def test_half_grade():
    r = exp(sqrt(ZERO))
    check_series("exp(h^1/2)", r.coeffs_dict(),
           {-n / 2: 1.0 / math.factorial(n) for n in range(15)}, 1e-14)
    print(f"  complete_order {r.complete_order} (want 7)")
    assert r.complete_order == 7


def test_two_terms_against_mpmath():
    x = R(1.3) + ZERO
    r = exp(-(x * x))
    ref = _taylor(lambda t: mp.exp(-t * t), mp.mpf("1.3"), 14)
    check_series("exp(-(x^2)) at 1.3", r.coeffs_dict(), {-n: ref[n] for n in range(15)}, 1e-13)


def test_mixed_half_and_whole_grades_against_mpmath():
    r = exp(R(2) + sqrt(ZERO) + ZERO)
    ref = _taylor(lambda s: mp.exp(2 + s + s * s), 0, 14)       # s = h^1/2
    check_series("exp(2 + h^1/2 + h)", r.coeffs_dict(), {-n / 2: ref[n] for n in range(15)}, 1e-13)
    assert r.complete_order == 7


def test_series_argument_against_mpmath():
    r = exp(sin(R(0.7) + ZERO))
    ref = _taylor(lambda t: mp.exp(mp.sin(t)), mp.mpf("0.7"), 11)
    check_series("exp(sin x) at 0.7", r.coeffs_dict(), {-n: ref[n] for n in range(12)}, 1e-13)
    print(f"  complete_order {r.complete_order} (want 11, the argument's own bound)")
    assert r.complete_order == 11
    assert min(r.coeffs_dict()) >= -11, "no grade past the argument's bound"


def test_log_axis_route_unchanged():
    # h ln h = -h L with L = ln(1/h): exp(-h L) = sum (-1)^n h^n L^n / n!
    # A grade may come back scalar next to the vector ones (the plain 0 does,
    # depending on which backend state earlier imports left): d is (d, 0).
    r = exp(ZERO * ln(ZERO))
    vec = lambda d: d if isinstance(d, tuple) else (d, 0)
    got = {}
    for d, v in r.coeffs_dict().items():               # two spellings of one grade ADD
        k = (int(round(float(vec(d)[0]))), int(vec(d)[1]))
        got[k] = got.get(k, 0.0) + v
    check_series("exp(h ln h)", got, {(-n, n): (-1.0) ** n / math.factorial(n) for n in range(15)}, 1e-14)


def test_denotation_marker_carried():
    x = R(1) + ZERO + (R(6) - R(6))
    r = exp(x)
    print(f"\n  argument denotation_order {x.denotation_order}  result {r.denotation_order}")
    assert x.denotation_order == 1 and r.denotation_order == 1


def test_underflow_is_absent_not_a_manufactured_infinitesimal():
    # (1e-200)^2 underflows.  The old loop formed h**2 = |0.0|_-2, a wholly
    # zero composite, and R1 turned it into |1|_-3: exp(1e-200 h) came back
    # with coefficient 1.0 at grades -3, -6, -9, -12.
    r = exp(R(1e-200) * ZERO)
    d = r.coeffs_dict()
    print(f"\n  exp(1e-200 h) = {d}")
    assert d == {0: 1.0, -1: 1e-200}


def test_several_integer_terms_order_50_against_mpmath():
    # The recurrence path: exp(-(x*x)) at 1.3 was 3e-12 out at order 50 as a
    # product of exp(-2.6h) and exp(-h^2), whose alternating terms cancel.
    import composite.composite_lib as cl
    for label, a, fc, fm in (("exp(-(x^2)) at 1.3", 1.3, lambda x: exp(-(x * x)), lambda t: mp.exp(-t * t)),
                             ("exp(sin x) at 0.7", 0.7, lambda x: exp(sin(x)), lambda t: mp.exp(mp.sin(t)))):
        with cl._derivative_scope(50, 52):
            r = fc(R(a) + ZERO)
        K = min(int(r.complete_order), 50)
        print(f"\n  {label}: complete_order {r.complete_order}")
        assert r.complete_order >= 50
        ref = _taylor(fm, mp.mpf(a), 50)
        check_series(f"{label}, orders 0..{K}", r.coeffs_dict(), {-n: ref[n] for n in range(K + 1)}, 1e-13)
