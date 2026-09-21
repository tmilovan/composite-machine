#!/usr/bin/env python3
# Composite Machine — singularity analysis
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Where a series stops converging, and what it does there -- against known answers.

`composite/singularity.py` takes the coefficients of a series and returns the
location and exponent of its nearest singularity.  Four different calculations
reduce to that one: the blow-up time and rate of a nonlinear ODE, the critical
point and exponent of a lattice model, the asymptotic growth of a counting
sequence, and the Borel singularity of a perturbation series.

Every case below has an answer known in closed form, so the suite can be wrong
rather than merely green.

WHAT IT LOOKS FOR, in order of how much it can embarrass us:

  S1  algebraic singularities, exponent known exactly.  Integer poles,
      square roots, and a deliberately non-rational exponent, on both sides
      of zero.

  S2  nonlinear ODE blow-up.  y' = y^p blows up at a time and a rate that are
      both known; the rate is the part numerical integration cannot give you.

  S3  analytic combinatorics.  Catalan, Motzkin, Schroeder, central binomial,
      Fibonacci and derangements -- sequences whose growth constants are
      textbook, including the amplitude, which is a second check on the
      exponent because a wrong exponent makes the amplitude drift.

  S4  negative controls.  An entire function has no singularity and the honest
      answer is None.  Returning a confident number for exp(z) would make
      every other result in this file worthless.

  S5  the failure modes that were actually hit while building this, each
      of which produced a WRONG answer rather than an error:
        - Froissart doublets: spurious poles with residue at the rounding
          floor, which cost 4 of 7 exponents before any filtering.  Mutation
          testing showed that the residue filter added for them is inert --
          removing it changes nothing on either route, because persistence
          and the radius filter reject the doublets first.  S5.01/02 pin the
          ANSWER on a doublet-prone series; no test credits the residue
          filter, because no test can.
        - symmetric singularities: tan has poles at both +-pi/2, and
          clustering on a median over candidates lands near zero and rejects
          both.
        - a spurious root nearer than the real one: clustering on the NEAREST
          candidate then returned z0 = 2.5e-12 for Catalan.  The radius of
          convergence is a property of the coefficients and settles it.
        - dynamic range: Catalan coefficients span fifteen decades, and an
          unnormalised least-squares fit let the few largest equations decide
          it, costing three digits of exponent.
        - direction: a blow-up TIME is forward in time, and the unrestricted
          answer for tan is whichever pole the approximants happen to favour.

  S6  the reported semantics: kind, confidence, and the aliases each field
      uses for the same exponent.

Tolerances are fixed at the measured accuracy, not widened until things pass.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.singularity import (
    analyse, blowup, series_solve, coefficient_asymptotics, radius, Singularity,
)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

PHI_INV = (5 ** 0.5 - 1) / 2          # 0.6180339887498949
INV_SQRT_PI = 1 / math.sqrt(math.pi)  # 0.5641895835477563


# ---- sequences, generated rather than pasted ------------------------------

def catalan(n):
    return [float(math.comb(2 * k, k) // (k + 1)) for k in range(n)]


def central_binomial(n):
    return [float(math.comb(2 * k, k)) for k in range(n)]


def motzkin(n):
    m = [1.0, 1.0]
    while len(m) < n:
        j = len(m) - 1
        m.append(m[j] + sum(m[k] * m[j - 1 - k] for k in range(j)))
    return m[:n]


def schroeder(n):
    r = [1.0]
    while len(r) < n:
        j = len(r)
        r.append(r[j - 1] + sum(r[k] * r[j - 1 - k] for k in range(j)))
    return r[:n]


def fibonacci(n):
    f = [0.0, 1.0]
    while len(f) < n:
        f.append(f[-1] + f[-2])
    return f[:n]


def derangement_ratio(n):
    """D_k / k!  ->  GF e^-z/(1-z): a simple pole at z = 1."""
    out, s = [], 0.0
    for k in range(n):
        s += (-1.0) ** k / math.factorial(k)
        out.append(s)
    return out


def binomial_series(gamma, xc, n):
    """(1 - x/xc)^-gamma, so beta = -gamma and z0 = xc."""
    return [math.prod((gamma + j) / (j + 1) for j in range(k)) / xc ** k
            for k in range(n)]


# ==========================================================================

def s1_algebraic(t):
    head("S1  algebraic singularities, exponent known exactly")
    for gamma, xc in ((1.75, 0.25), (1.2345, 0.2), (3.5, 2.0), (0.5, 1.0),
                      (1.0, 0.5), (2.0, 1.0), (1 / 3, 0.5), (4.0, 0.125)):
        s = analyse(binomial_series(gamma, xc, 32))
        if s is None:
            t.true("S1 (1-x/%g)^-%g found" % (xc, gamma), False, "returned None")
            continue
        t.close("S1 (1-x/%-5g)^-%-6g  z0" % (xc, gamma), s.location, xc, tol=1e-10)
        t.close("S1 (1-x/%-5g)^-%-6g  beta" % (xc, gamma), s.exponent, -gamma, tol=1e-9)
        t.close("S1 (1-x/%-5g)^-%-6g  gamma alias" % (xc, gamma),
                s.critical_exponent, gamma, tol=1e-9)


def s2_ode_blowup(t):
    head("S2  nonlinear ODE blow-up: the time AND the rate")
    #  y' = y^p, y(0)=y0  ->  t* = y0^(1-p)/(p-1),  y ~ (t*-t)^(-1/(p-1))
    for p, y0 in ((2, 1.0), (2, 2.0), (2, 10.0), (3, 1.0), (4, 1.0), (5, 1.0)):
        f = [0.0] * (p + 1)
        f[p] = 1.0
        t_star = y0 ** (1 - p) / (p - 1)
        alpha = 1.0 / (p - 1)
        s = blowup(f, y0, terms=30)
        if s is None:
            t.true("S2 y'=y^%d y0=%g" % (p, y0), False, "returned None")
            continue
        t.close("S2 y'=y^%d y0=%-4g  t*" % (p, y0), s.location, t_star, tol=1e-9)
        t.close("S2 y'=y^%d y0=%-4g  rate" % (p, y0), s.blowup_rate, alpha, tol=1e-6)
    #  y' = 1 + y^2, y(0)=0  ->  y = tan t, blows up at pi/2 with rate 1
    s = blowup([1.0, 0.0, 1.0], 0.0, terms=30)
    t.close("S2 y'=1+y^2  t* = pi/2", s.location, math.pi / 2, tol=1e-8)
    t.close("S2 y'=1+y^2  rate", s.blowup_rate, 1.0, tol=1e-6)
    t.true("S2 y'=1+y^2  t* is forward in time", s.location > 0,
           "got %.12g (tan has poles at both +-pi/2)" % s.location)
    #  the series itself must be right before anything read off it can be
    a = series_solve([1.0, 0.0, 1.0], 0.0, 8)
    t.close("S2 series_solve reproduces tan: a_1", a[1], 1.0, tol=1e-15)
    t.close("S2 series_solve reproduces tan: a_3", a[3], 1.0 / 3, tol=1e-15)
    t.close("S2 series_solve reproduces tan: a_5", a[5], 2.0 / 15, tol=1e-15)
    t.close("S2 series_solve reproduces tan: a_7", a[7], 17.0 / 315, tol=1e-15)


def s3_combinatorics(t):
    head("S3  analytic combinatorics: growth constants and amplitudes")
    #  name, coefficients, z0, beta, amplitude C, beta tolerance.
    #  Each tolerance is the accuracy that case actually delivers, not one
    #  number loose enough to cover the worst.  A single 1e-8 was slack enough
    #  to pass with a square rather than overdetermined least-squares fit,
    #  which is 90x worse on Motzkin -- the per-case bound is what makes the
    #  fit load-bearing.  Catalan is the least accurate of the six at 3.2e-12
    #  and is the reason a single tight bound does not work either.
    CASES = [
        ("Catalan", catalan(32), 0.25, 0.5, INV_SQRT_PI, 1e-11),
        ("central binomial", central_binomial(32), 0.25, -0.5, INV_SQRT_PI, 1e-14),
        ("Motzkin", motzkin(32), 1 / 3, 0.5,
         math.sqrt(3) * 3 / (2 * math.sqrt(math.pi)), 1e-12),
        ("large Schroeder", schroeder(32), 3 - 2 * math.sqrt(2), 0.5, None, 1e-12),
        ("Fibonacci", fibonacci(32), PHI_INV, -1.0, None, 1e-14),
        ("derangements D_n/n!", derangement_ratio(32), 1.0, -1.0, None, 1e-13),
    ]
    for name, c, z0, beta, amp, btol in CASES:
        s = analyse(c)
        if s is None:
            t.true("S3 %s found" % name, False, "returned None")
            continue
        t.close("S3 %-19s z0" % name, s.location, z0, tol=1e-9)
        t.close("S3 %-19s beta" % name, s.exponent, beta, tol=btol)
        if amp is not None:
            got = coefficient_asymptotics(c, s)[0]
            rel = abs(got - amp) / abs(amp)
            # The amplitude is an asymptotic fit and is term-count limited --
            # unlike the location and the exponent, which sit at 1e-14 here.
            # S3.conv below pins that it is a rate, not a wall.
            t.true("S3 %-19s amplitude" % name, rel < 1.5e-2,
                   "got %.12g  want %.12g  rel %.2e  (32 terms)" % (got, amp, rel))
    #  the amplitude must IMPROVE with term count; the exponent must not need to
    true_c = INV_SQRT_PI
    errs = []
    for n in (24, 48, 80):
        c = catalan(n)
        sg = analyse(c)
        errs.append(abs(coefficient_asymptotics(c, sg)[0] - true_c) / true_c)
        t.close("S3.conv Catalan beta at %d terms" % n, sg.exponent, 0.5, tol=1e-8)
    t.true("S3.conv amplitude converges with terms",
           errs[0] > errs[1] > errs[2],
           "rel err 24/48/80 terms: %.2e -> %.2e -> %.2e" % tuple(errs))

    #  the growth constant is 1/z0, which is what a combinatorialist quotes
    s = analyse(catalan(32))
    t.close("S3 Catalan growth constant 1/z0 = 4", 1 / s.location, 4.0, tol=1e-9)
    s = analyse(schroeder(32))
    t.close("S3 Schroeder growth 3+2sqrt2", 1 / s.location,
            3 + 2 * math.sqrt(2), tol=1e-8)


def s4_negative_controls(t):
    head("S4  negative controls: an entire function has no singularity")
    for name, c in (
            ("exp(z)", [1.0 / math.factorial(n) for n in range(30)]),
            ("cos(z)", [((-1) ** (n // 2) / math.factorial(n)) if n % 2 == 0 else 0.0
                        for n in range(30)]),
            ("sin(z)/z", [((-1) ** (n // 2) / math.factorial(n + 1)) if n % 2 == 0
                          else 0.0 for n in range(30)]),
            ("exp(z)/2 + 1", [1.0 + 0.5] + [0.5 / math.factorial(n) for n in range(1, 30)]),
            ("a polynomial", [1.0, 2.0, 3.0, 4.0] + [0.0] * 20),
            ("all zeros", [0.0] * 30),
            ("too few terms", [1.0, 1.0, 1.0]),
    ):
        got = analyse(c)
        t.true("S4 %-14s -> None" % name, got is None,
               "got %r" % (got,) if got is not None else "correct")


def s5_failure_modes(t):
    head("S5  the failure modes that produced WRONG answers, not errors")
    #  Froissart doublets.  A geometric series is exactly [1/1]; higher
    #  approximants grow spurious pole/zero pairs with residue ~1e-16.
    g, xc = 1.75, 0.25
    s = analyse(binomial_series(g, xc, 32))
    t.close("S5.01 doublet-prone series still exact (z0)", s.location, xc, tol=1e-10)
    t.close("S5.02 doublet-prone series still exact (beta)", s.exponent, -g, tol=1e-9)
    #  A spurious root nearer than the true one.  What rejects it is the
    #  radius FILTER, not the ordering: mutation testing showed that replacing
    #  the radius-ordered search with nearest-first leaves every test green,
    #  while removing the filter fails them.  The filter is the guard.
    s = analyse(catalan(32))
    t.true("S5.03 Catalan is not answered at the origin",
           abs(s.location - 0.25) < 1e-9,
           "got z0 = %.12g (clustering with no radius filter returned 2.5e-12)"
           % s.location)
    #  the radius of convergence is what rejects it, and must itself be right
    t.close("S5.04 radius of convergence, Catalan", radius(catalan(32)), 0.25, tol=0.1)
    t.close("S5.05 radius of convergence, Fibonacci", radius(fibonacci(32)),
            PHI_INV, tol=0.05)
    #  dynamic range
    c = catalan(32)
    t.true("S5.06 the series really does span 15 decades",
           c[31] / c[1] > 1e14, "ratio %.3g" % (c[31] / c[1]))
    #  symmetric poles
    s = blowup([1.0, 0.0, 1.0], 0.0, terms=30)
    t.true("S5.07 tan's two poles do not cancel each other out", s is not None,
           "a median over both lands at 0 and rejects both")
    #  direction
    t.true("S5.08 blow-up time is positive", s.location > 0, "got %.12g" % s.location)
    #  unrestricted analysis may legitimately answer on either side
    un = analyse(series_solve([1.0, 0.0, 1.0], 0.0, 30))
    t.close("S5.09 unrestricted |z0| is still pi/2", abs(un.location),
            math.pi / 2, tol=1e-8)


def s6_reported_semantics(t):
    head("S6  what the report says about itself")
    s = analyse(fibonacci(32))
    t.exact("S6.01 a simple pole is called one", s.kind, "pole")
    t.exact("S6.02 and its order is 1", s.pole_order, 1)
    t.exact("S6.03 confidence is high when both routes agree", s.confidence, "high")
    t.true("S6.04 the Pade cross-check found it too", s.cross_check is not None,
           "cross-check %r" % (s.cross_check,))
    s = analyse(catalan(32))
    t.exact("S6.05 a square root is called a branch point", s.kind, "branch point")
    t.exact("S6.06 and reports no pole order", s.pole_order, None)
    s = analyse(binomial_series(2.0, 1.0, 32))
    t.exact("S6.07 a double pole is order 2", s.pole_order, 2)
    s = analyse(catalan(32))
    t.close("S6.08 blowup_rate and critical_exponent are the same number",
            s.blowup_rate - s.critical_exponent, 0.0, tol=0.0)
    t.true("S6.09 describe() renders", isinstance(s.describe(), str))
    t.true("S6.10 repr() renders", "Singularity" in repr(s))
    t.true("S6.11 spread is reported, not hidden",
           isinstance(s.spread, float) and s.spread >= 0.0, "spread %.2e" % s.spread)


def run_all():
    t = Suite()
    for fn in (s1_algebraic, s2_ode_blowup, s3_combinatorics,
               s4_negative_controls, s5_failure_modes, s6_reported_semantics):
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
