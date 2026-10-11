"""ln on the power axis: a one-term argument written out, several terms through
ln(a) + antiderivative(x'/x).

The second path replaced summing powers of u = (x - a)/a, which loses digits
with order for any argument of several terms: ln(1 + x^2) at 0.7 was 3e-6 out
at order 50.  These pin the coefficients to order 50 against closed forms or
the mpmath oracle, the completeness bound, the unchanged log-axis route, the
denotation marker, and grade 0 left absent when ln(a) = 0.

Tolerances are relative and fixed before running: 1e-14 for a closed form,
1e-13 against mpmath for an argument of several terms.
"""
import math

import mpmath as mp

import composite.composite_lib as cl
from tests.series_check import check_series
from composite.composite_lib import R, ZERO, ln, sin, sqrt

mp.mp.dps = 60


def _deep(f, a):
    with cl._derivative_scope(50, 52):
        return f(R(a) + ZERO)


def test_one_term_closed_form():
    # ln(0.3 + h) = ln 0.3 + sum (-1)^(n+1) (h/0.3)^n / n
    r = ln(R(0.3) + ZERO)
    want = {0: math.log(0.3)}
    want.update({-n: (-1) ** (n + 1) / (n * 0.3 ** n) for n in range(1, 15)})
    check_series("ln(0.3 + h)", r.coeffs_dict(), want, 1e-14)
    assert r.complete_order == 14


def test_one_term_half_grade():
    # ln(2 + 3 h^1/2): grade -n/2 holds (-1)^(n+1) 1.5^n / n
    r = ln(R(2) + R(3) * sqrt(ZERO))
    want = {0: math.log(2.0)}
    want.update({-n / 2: (-1) ** (n + 1) * 1.5 ** n / n for n in range(1, 15)})
    check_series("ln(2 + 3 h^1/2)", r.coeffs_dict(), want, 1e-14)
    assert r.complete_order == 7


def test_unit_standard_part_leaves_grade_zero_absent():
    r = ln(R(1) + R(2) * ZERO)
    d = r.coeffs_dict()
    print(f"\n  ln(1 + 2h) grades {sorted(d)[-3:]}  grade 0 present: {0 in d}")
    assert 0 not in d
    r = ln(R(1) + ZERO + ZERO * ZERO)
    d = r.coeffs_dict()
    print(f"  ln(1 + h + h^2) grade 0 present: {0 in d}")
    assert 0 not in d


def test_several_terms_order_50_against_mpmath():
    for label, a, fx, fm in (
            ("ln(1 + x^2) at 0.7", 0.7, lambda x: R(1) + x * x, lambda x: mp.log(1 + x * x)),
            ("ln(2 + sin x) at 1.1", 1.1, lambda x: R(2) + sin(x), lambda x: mp.log(2 + mp.sin(x)))):
        r = _deep(lambda x: ln(fx(x)), a)
        ref = mp.taylor(fm, mp.mpf(a), 50)
        K = min(int(r.complete_order), 50)            # the reference runs to order 50
        check_series(f"{label}, orders 0..{K}", r.coeffs_dict(), {-n: float(ref[n]) for n in range(K + 1)}, 1e-13)
        print(f"  complete_order {r.complete_order}")
        assert K >= 50


def test_several_terms_half_grade_against_mpmath():
    r = ln(R(2) + sqrt(ZERO) + ZERO)
    ref = mp.taylor(lambda s: mp.log(2 + s + s * s), 0, 14)          # s = h^1/2
    check_series("ln(2 + h^1/2 + h)", r.coeffs_dict(), {-n / 2: float(ref[n]) for n in range(15)}, 1e-13)
    assert r.complete_order == 7


def test_bound_follows_the_argument():
    r = _deep(lambda x: ln(R(2) + sin(x)), 1.1)
    arg = _deep(lambda x: R(2) + sin(x), 1.1)
    print(f"\n  ln(2 + sin x) complete_order {r.complete_order}, its argument {arg.complete_order}")
    assert r.complete_order <= arg.complete_order
    assert min(r.coeffs_dict()) >= -r.complete_order


def test_log_axis_route_unchanged():
    # ln(2 + h ln h): the argument has a log-axis term, so the power loop runs
    r = ln(R(2) + ZERO * ln(ZERO))
    vec = lambda d: d if isinstance(d, tuple) else (d, 0)
    got = {}
    for d, v in r.coeffs_dict().items():               # two spellings of one grade ADD
        k = (int(round(float(vec(d)[0]))), int(vec(d)[1]))
        got[k] = got.get(k, 0.0) + v
    # h ln h = -h L, L = ln(1/h): ln(2 - hL) = ln 2 - sum (hL/2)^n / n
    want = {(0, 0): math.log(2.0)}
    want.update({(-n, n): -(0.5 ** n) / n for n in range(1, 15)})
    check_series("ln(2 + h ln h)", got, want, 1e-14)


def test_denotation_marker_carried():
    x = R(1.5) + ZERO + (R(6) - R(6))
    r = ln(x)
    print(f"\n  argument denotation_order {x.denotation_order}  result {r.denotation_order}")
    assert x.denotation_order == 1 and r.denotation_order == 1
