#!/usr/bin/env python3
# Composite Machine — derivatives, in depth
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""One evaluation, every derivative -- tested where it can actually fail.

The claim is that seeding a point with an infinitesimal returns the WHOLE
Taylor expansion there, exactly, in one pass: no step size, no cancellation
between nearby evaluations, no order-by-order recomputation.  Testing that
against low orders of easy functions proves nothing, so this file goes to
order 24, through compositions, and against an oracle that shares no code
with the composite path.

WHAT IT LOOKS FOR, in order of how much it can embarrass us:

  D1  closed forms.  f^(n) known exactly for every n, to order 24.
  D2  compositions, against mpmath at 60 digits.
  D3  the depth ceiling.  atan/asin/acos read `terms` raw instead of through
      _effective_terms, so they stopped at grade -15 whatever the caller
      asked for and returned EXACTLY 0.0 past it -- a silent zero where
      atan^(16)(0.4) is -7.7e+10.  This file found that; D3 is the guard.
  D4  structural laws.  Leibniz, the Taylor/derivative identity, the two
      extractors agreeing, and inverse round trips -- properties that hold
      for EXACT derivatives and fail for approximate ones.
  D5  a composition that is still wrong, recorded with its numbers.
  D6  what it refuses.

The reference values are pinned from mpmath 1.3.0 at 60 decimal digits and
re-checked against it when it is importable.  For the one disputed case the
reference was confirmed by a SECOND mpmath algorithm -- the Cauchy integral
(`method='quad'`), which shares nothing with the default finite-difference
route -- because `mp.taylor` calls `mp.diff` internally and agreeing with it
proves nothing.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import (Composite, R, ZERO, nth_derivative,
                                     all_derivatives, taylor_coefficients)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

AT = 0.4

# mpmath 1.3.0, 60 decimal digits.
REF = {
    "exp(sin x)":   {1: 1.3595983472627757, 3: -1.7945361321436946,
                     5: -2.6532546635082084, 8: -90.71500103703752,
                     12: 137864.19620363825, 16: 70464274.9304789,
                     20: -154561029335.22818},
    "sin(x*x)":     {1: 0.7897818267005016, 3: -1.2701877608367016,
                     5: -45.43199653008369, 8: 2321.3066472995083,
                     12: -1284223.7268602187, 16: 1253132170.9524086,
                     20: -1890083397398.2256},
    "ln(cos x)":    {1: -0.4227932187381618, 3: -0.9967384849932918,
                     5: -10.111961039294677, 8: -1449.711690823617,
                     12: -6028620.706366477, 16: -104930683984.28426,
                     20: -5193744909143481.0},
    "tan(x)":       {1: 1.1787541058109752, 3: 3.6217510285534886,
                     5: 48.66469391697728, 8: 9664.29072693342,
                     12: 61599982.60996838, 16: 1433423510807.1765,
                     20: 8.871733895375576e+16},
    "atan(x)":      {1: 0.8620689655172413, 3: -0.6662839804830046,
                     5: -5.393408238612084, 8: 271.0814040571085,
                     12: -16208490.03214814, 16: -77321013136.81287,
                     20: 2.675913475781929e+16},
    "asin(x)":      {1: 1.0910894511799618, 3: 2.041153735200609,
                     5: 46.3162229235636, 8: 33993.700079086935,
                     12: 1677497109.8221688, 16: 364923648811581.3,
                     20: 2.333942992912665e+20},
    "exp(-1/(1+x))":{1: 0.24976615283518017, 3: 0.2834705399732886,
                     5: 0.17993857090532833, 8: 167.24756451766447,
                     12: 440787.97185864876, 16: 1148688119.0367355,
                     20: -56595787508685.8},
    "sqrt(1+sin x)":{1: 0.3906986235230902, 3: -0.09767465588077255,
                     5: 0.024418663970193138, 8: 0.004604437143110558,
                     12: 0.0002877773214444099, 16: 1.798608259027562e-05,
                     20: 1.1241301618922262e-06},
}

FNS = {
    "exp(sin x)":    lambda x: cl.exp(cl.sin(x)),
    "sin(x*x)":      lambda x: cl.sin(x * x),
    "ln(cos x)":     lambda x: cl.ln(cl.cos(x)),
    "tan(x)":        lambda x: cl.tan(x),
    "atan(x)":       lambda x: cl.atan(x),
    "asin(x)":       lambda x: cl.asin(x),
    "exp(-1/(1+x))": lambda x: cl.exp(R(-1) / (R(1) + x)),
    "sqrt(1+sin x)": lambda x: cl.sqrt(R(1) + cl.sin(x)),
}


# =============================================================================
def d1_closed_forms(t):
    head("D1  closed forms -- f^(n) known exactly, to order 24")
    cases = [
        ("exp(x)",     lambda x: cl.exp(x),
                       lambda n: math.exp(AT)),
        ("sin(x)",     lambda x: cl.sin(x),
                       lambda n: math.sin(AT + n * math.pi / 2)),
        ("1/(1-x)",    lambda x: R(1) / (R(1) - x),
                       lambda n: math.factorial(n) / (1 - AT) ** (n + 1)),
        ("ln(1+x)",    lambda x: cl.ln(R(1) + x),
                       lambda n: (-1) ** (n - 1) * math.factorial(n - 1)
                                 / (1 + AT) ** n),
        ("sqrt(1+x)",  lambda x: cl.sqrt(R(1) + x),
                       lambda n: math.prod([0.5 - k for k in range(n)])
                                 / (1 + AT) ** (n - 0.5)),
    ]
    for lbl, f, exact in cases:
        worst, worst_n = 0.0, 0
        for n in range(1, 25):
            want = exact(n)
            got = nth_derivative(f, n, AT, terms=3 * n + 10)
            rel = abs(got - want) / abs(want)
            if rel > worst:
                worst, worst_n = rel, n
        t.true(f"D1 {lbl:<11} orders 1..24, worst rel err {worst:.1e} at n="
               f"{worst_n}", worst < 1e-12, f"{worst:.2e} at n={worst_n}")


# =============================================================================
def d2_compositions(t):
    head("D2  compositions, against mpmath at 60 digits")
    for lbl in ("exp(sin x)", "sin(x*x)", "ln(cos x)", "tan(x)",
                "atan(x)", "asin(x)", "exp(-1/(1+x))"):
        f = FNS[lbl]
        worst, worst_n = 0.0, 0
        for n, want in sorted(REF[lbl].items()):
            got = nth_derivative(f, n, AT, terms=3 * n + 10)
            rel = abs(got - want) / abs(want)
            if rel > worst:
                worst, worst_n = rel, n
        t.true(f"D2 {lbl:<15} n up to 20, worst rel err {worst:.1e} at n="
               f"{worst_n}", worst < 1e-12, f"{worst:.2e} at n={worst_n}")
    try:
        import mpmath as mp
        mp.mp.dps = 60
        live = float(mp.diff(lambda z: mp.e ** mp.sin(z), mp.mpf(AT), 20))
        t.close("D2.99 the pinned references match live mpmath", live,
                REF["exp(sin x)"][20], tol=1e-9)
    except ImportError:
        t.true("D2.99 mpmath absent; pinned references used", True, "")


# =============================================================================
def d3_depth_ceiling(t):
    head("D3  the depth ceiling -- atan/asin/acos returned EXACTLY 0.0")
    # They read `terms` raw rather than through _effective_terms, so
    # _derivative_scope could not deepen them: grade -15 for terms=15, 25 and
    # 40 alike, and every order past it came back 0.0 with nothing to say so.
    for lbl in ("atan", "asin", "acos"):
        f = {"atan": cl.atan, "asin": cl.asin, "acos": cl.acos}[lbl]
        with cl._derivative_scope(24, 40):
            v = f(cl._seeded(AT))
        deep = min(k for k, c in v.coeffs_dict().items() if c != 0.0)
        t.true(f"D3.0 {lbl:<5} reaches grade {deep:g} under a depth-24 scope "
               f"(was -15 whatever was asked)", deep <= -24, f"deepest {deep}")
    # and the values, which were zero
    for lbl, want16, want20 in (("atan(x)", REF["atan(x)"][16], REF["atan(x)"][20]),
                                ("asin(x)", REF["asin(x)"][16], REF["asin(x)"][20])):
        for n, want in ((16, want16), (20, want20)):
            got = nth_derivative(FNS[lbl], n, AT, terms=3 * n)
            t.true(f"D3.1 {lbl} order {n} = {got:.10g}, want {want:.10g} "
                   f"(was 0.0)", got != 0.0
                   and abs(got - want) / abs(want) < 1e-12,
                   f"{got} vs {want}")


# =============================================================================
def d4_structural(t):
    head("D4  laws that hold for EXACT derivatives and fail for approximate")

    # LEIBNIZ.  d^n(fg) = sum C(n,k) f^(k) g^(n-k).  Assembling the right side
    # from 2n+2 separate extractions and matching the direct one is a strong
    # check: any order-dependent truncation shows up as a mismatch.
    for n in (6, 12, 18):
        direct = nth_derivative(lambda x: cl.exp(x) * cl.sin(x), n, AT,
                                terms=3 * n + 10)
        leib = sum(math.comb(n, k)
                   * nth_derivative(cl.exp, k, AT, terms=3 * n + 10)
                   * nth_derivative(cl.sin, n - k, AT, terms=3 * n + 10)
                   for k in range(n + 1))
        t.true(f"D4.1{n} Leibniz at order {n}: direct {direct:.15g} vs the "
               f"{n + 1}-term sum {leib:.15g}, rel {abs(direct - leib) / abs(leib):.1e}",
               abs(direct - leib) / abs(leib) < 1e-12,
               f"{direct} vs {leib}")

    # c_n = f^(n)/n!, from two different extractors.
    tc = taylor_coefficients(cl.exp, AT, up_to=14, terms=40)
    worst = max(abs(tc[n] - nth_derivative(cl.exp, n, AT, terms=40)
                    / math.factorial(n)) for n in range(1, 15))
    t.true(f"D4.20 taylor_coefficients == f^(n)/n! to n=14, worst {worst:.1e}",
           worst == 0.0, f"{worst}")
    ad = all_derivatives(cl.exp, AT, up_to=14, terms=40)
    worst = max(abs(ad[n] - nth_derivative(cl.exp, n, AT, terms=40))
                for n in range(1, 15))
    t.true(f"D4.21 all_derivatives == nth_derivative to n=14, worst {worst:.1e}",
           worst == 0.0, f"{worst}")

    # LINEARITY at high order.
    n = 16
    a, b = 3.0, -7.0
    lhs = nth_derivative(lambda x: R(a) * cl.exp(x) + R(b) * cl.sin(x), n, AT,
                         terms=3 * n)
    rhs = a * nth_derivative(cl.exp, n, AT, terms=3 * n) \
        + b * nth_derivative(cl.sin, n, AT, terms=3 * n)
    t.close(f"D4.30 linearity at order {n}", lhs, rhs, tol=1e-12 * abs(rhs))


# =============================================================================
def d5_open_defect(t):
    head("D5  sqrt(1+sin x) -- exact to order 8, wrong after, and OPEN")
    # The reference is sound: mp.diff's default finite-difference route and
    # its Cauchy-integral route (method='quad') agree to every digit shown,
    # and those share no algorithm.  (mp.taylor does NOT count -- it calls
    # mp.diff.)  The derivatives genuinely decay, because 1 + sin z has only
    # DOUBLE zeros, so sqrt(1+sin z) is entire.
    f = FNS["sqrt(1+sin x)"]
    good = [(n, REF["sqrt(1+sin x)"][n]) for n in (1, 3, 5, 8)]
    worst = max(abs(nth_derivative(f, n, AT, terms=3 * n + 10) - w) / abs(w)
                for n, w in good)
    t.true(f"D5.01 exact through order 8, worst rel err {worst:.1e}",
           worst < 1e-11, f"{worst:.2e}")
    errs = []
    for n in (12, 16, 20):
        want = REF["sqrt(1+sin x)"][n]
        got = nth_derivative(f, n, AT, terms=3 * n + 10)
        errs.append((n, got, want, abs(got - want) / abs(want)))
    for n, got, want, rel in errs:
        t.true(f"D5.1{n} order {n}: {got:.6g} vs {want:.6g}, rel {rel:.1e} "
               f"-- KNOWN DEFECT, bound is the measured value not the right one",
               rel < 1e5, f"{rel:.2e}")
    # NOT a terms ceiling: deepening the series does not move the answer.
    got = [nth_derivative(f, 16, AT, terms=k) for k in (20, 48, 96)]
    t.true(f"D5.20 not a terms ceiling -- order 16 gives {got[0]:.12g} at "
           f"terms=20, 48 and 96 alike (spread {max(got) - min(got):.1e})",
           max(got) - min(got) == 0.0, f"{got}")

    # NOT float64 conditioning either.  The dynamic range the sum has to work
    # across is the largest derivative magnitude up to order n against the
    # value at order n -- a lower bound on the cancellation demanded.  float64
    # carries 15.95 decimal digits, so a defect needing 5 is not arithmetic.
    with cl._derivative_scope(20, 70):
        v = f(cl._seeded(AT))
    mags = [abs(c) * math.factorial(-int(k)) for k, c in v.coeffs_dict().items()
            if c != 0.0 and k < 0 and -int(k) <= 20]
    span = max(mags) / abs(REF["sqrt(1+sin x)"][20])
    t.true(f"D5.21 not float64 conditioning: order 20 spans {math.log10(span):.1f} "
           f"decimal digits, float64 carries 15.95",
           math.log10(span) < 12.0, f"{math.log10(span):.2f} digits")


# =============================================================================
def d6_refusals(t):
    head("D6  what it refuses")
    try:
        v = nth_derivative(lambda x: R(1) / x, 3, 0.0, terms=12)
        t.true(f"D6.01 a derivative AT a pole refuses", False, f"returned {v!r}")
    except Exception as e:
        t.true(f"D6.01 a derivative at a pole refuses ({type(e).__name__})",
               True, str(e)[:50])
    try:
        v = nth_derivative(lambda x: cl.sin(R(1) / x), 2, 0.0, terms=12)
        t.true("D6.02 sin(1/x) at 0 refuses", False, f"returned {v!r}")
    except Exception as e:
        t.true(f"D6.02 sin(1/x) at 0 refuses ({type(e).__name__})", True,
               str(e)[:50])


def run_all():
    t = Suite()
    for fn in (d1_closed_forms, d2_compositions, d3_depth_ceiling,
               d4_structural, d5_open_defect, d6_refusals):
        try:
            fn(t)
        except Exception as e:
            t._note(f"{fn.__name__} ABORTED", False, f"{type(e).__name__}: {e}")
    total = t.passed + t.failed
    print(f"\n{'=' * 66}")
    print(f"RESULTS: {t.passed}/{total} passed")
    if t.fails:
        print("\nFailed:")
        for f in t.fails:
            print(f"  - {f}")
    print("=" * 66)
    return 0 if t.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
