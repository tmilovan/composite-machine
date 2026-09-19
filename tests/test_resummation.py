#!/usr/bin/env python3
# Composite Machine — Borel-Pade resummation
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Getting a NUMBER out of a series that diverges for every nonzero argument.

The Euler-Stieltjes integral is the case:

    S(eps) = int_0^inf exp(-t)/(1 + eps*t) dt  ~  sum (-1)^n n! eps^n

eps = 0 is an IRREGULAR singular point, and that is why it is the test.  At a
branch point the expansion determines the function; here it does not.  S and
S + C*exp(-1/eps) share every coefficient to all orders, because the expansion
of exp(-1/eps) is identically zero -- so no number of coefficients picks out
which function you have, and "compute more coefficients" is not a strategy.

The series also diverges factorially: partial sums at eps = 1 run
1, 0, 2, -4, 20, -100, 620, -4420, away from S(1) = 0.5963473623231941.

Borel-Pade goes around both problems.  b_n = c_n/n! divides the factorial out;
Pade turns the transform into a rational function WITHOUT needing to recognise
it in closed form; the Laplace integral brings it back.  The Pade denominator's
roots then locate the Borel singularity, and its residue gives the size of the
flat term -- an object with no representation in this value group, measured
numerically before anything can hold it.

Tolerances are tight because these results are exact or near-exact, and a loose
tolerance on an exact result is not a test.  Where a bound is looser it is
because the quantity is genuinely approximate, and it says so.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO
from composite.resummation import (borel, pade, polydiv, poly, degree, poles,
                                   dpoly, borel_singularity, flat_term, resum)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

TARGET = 0.5963473623231941          # S(1) = e * E1(1)
STIELTJES = [(-1) ** n * math.factorial(n) for n in range(24)]


# =============================================================================
def r1_polynomials(t):
    head("R1  polynomial arithmetic underneath: division with a remainder")
    # Pade is the extended Euclidean algorithm, which needs quotient AND
    # remainder.  The library's `/` is SERIES division -- it does not stop,
    # (1+z^2)/(1-z) runs on into z^-1, z^-2 -- so the polynomial quotient is
    # its non-negative part and the remainder is formed as A - Q*B.
    cases = [([1, 0, 1], [1, -1], "(1+z^2)/(1-z)"),
             ([1, -3, 2], [1, -1], "(1-3z+2z^2)/(1-z)"),
             ([0, 0, 0, 1], [1, 1], "z^3/(1+z)"),
             ([5], [1, 2], "5/(1+2z)")]
    for i, (a, b, lbl) in enumerate(cases, 1):
        A, B = poly(a), poly(b)
        Q, Rm = polydiv(A, B)
        dr, db = degree(Rm), degree(B)
        t.true(f"R1.0{i} {lbl}: deg(R) < deg(B)",
               dr is None or dr < db, f"deg R {dr}, deg B {db}")
    t.dims("R1.05 exact division leaves no remainder",
           polydiv(poly([1, -3, 2]), poly([1, -1]))[1], {})
    t.dims("R1.06 and the quotient is 1 - 2z",
           polydiv(poly([1, -3, 2]), poly([1, -1]))[0], {0: 1.0, 1: -2.0})


# =============================================================================
def r2_borel(t):
    head("R2  the Borel transform divides the factorial out")
    b = borel(STIELTJES)
    t.true("R2.01 b_n = (-1)^n for every n",
           all(abs(b[n] - (-1) ** n) < 1e-15 for n in range(len(b))),
           f"first six {b[:6]}")
    t.true("R2.02 the ORIGINAL series diverges at eps=1",
           abs(sum(STIELTJES[:8])) > 4000,
           f"partial sums {[sum(STIELTJES[:k+1]) for k in range(8)]}")
    t.true("R2.03 the TRANSFORMED one does not",
           all(abs(x) <= 1.0 for x in b), "every |b_n| == 1")


# =============================================================================
def r3_pade(t):
    head("R3  Pade recovers 1/(1+z) without being told it exists")
    b = borel(STIELTJES)
    for k in (1, 2, 3, 5, 8):
        P, Q = pade(b, k, k)
        t.dims(f"R3.{k:02d}a [{k}/{k}] numerator is 1", P, {0: 1.0})
        t.dims(f"R3.{k:02d}b [{k}/{k}] denominator is 1 + z", Q,
               {0: 1.0, 1: 1.0})
    t.close("R3.20 the pole sits at u = -1",
            poles(pade(b, 3, 3)[1])[0].real, -1.0)
    t.dims("R3.21 d/dz of 1 + z is 1", dpoly(poly([1.0, 1.0])), {0: 1.0})
    t.raises("R3.22 too few coefficients is an error, not a guess",
             ValueError, lambda: pade([1.0, -1.0], 3, 3))


# =============================================================================
def r4_value(t):
    head("R4  the number, from a series that cannot be summed")
    for k in (1, 2, 3, 5):
        v, _ = resum(STIELTJES, 1.0, k, k)
        t.close(f"R4.{k:02d} S(1) via [{k}/{k}] = {TARGET}", float(v), TARGET,
                tol=1e-15)
    # The Laplace step must run to infinity.  A finite cutoff looks harmless --
    # the tail past u = 40 is about 1e-19 -- and costs eight digits: measured
    # 0.5963473621 on [0, 40], an error of 2e-10 that tightening tol from 1e-10
    # to 1e-16 did not move.
    f = lambda u: cl.exp(-u) / (R(1) + u)
    t.close("R4.10 improper Laplace is exact",
            float(cl.integrate(f, 0.0, float('inf'))), TARGET, tol=1e-15)
    t.true("R4.11 a finite cutoff is NOT",
           abs(float(cl.integrate(f, 0.0, 40.0)) - TARGET) > 1e-11,
           f"got {float(cl.integrate(f, 0.0, 40.0))!r}")


# =============================================================================
def r5_flat(t):
    head("R5  the flat term -- an object with no grade, measured anyway")
    z0, res = borel_singularity(STIELTJES)
    t.close("R5.01 Borel singularity located at z0 = -1", z0.real, -1.0)
    t.close("R5.02 with residue 1", abs(res), 1.0)
    # For eps < 0 the pole lands on the integration ray and the Laplace
    # integral is ambiguous by the contour choice.  That ambiguity IS the flat
    # term exp(-1/eps), which the value group cannot hold -- powers and logs
    # have no slot below every power.  Here it is a number.
    for i, e in enumerate((-1.0, -0.5, -0.25, -0.1, -0.05), 1):
        u0, amb = flat_term(STIELTJES, e)
        want = math.pi / abs(e) * math.exp(-1 / abs(e))
        t.close(f"R5.1{i} ambiguity at eps={e} is (pi/|eps|)exp(-1/|eps|)",
                amb, want, tol=1e-14 * max(1.0, want))
    for e in (1.0, 0.1):
        u0, amb = flat_term(STIELTJES, e)
        t.true(f"R5.2 eps={e} > 0: pole off the ray, no ambiguity",
               u0 is None and amb == 0.0, f"u0 {u0}, amb {amb}")


# =============================================================================
def r6_end_to_end(t):
    head("R6  end to end: the integral itself, nothing typed in")
    eps = ZERO
    f = lambda tt: cl.exp(-tt) / (R(1) + eps * tt)

    def panel(x, dx):
        fx = cl._ensure_composite(f(x + dx / 2 + cl._perturbation_seed(1)))
        F = cl._antiderivative_axis(fx, 1)
        acc = {}
        for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
            for k, v in cl._eval_axis(F, hv, 1).items():
                acc[k] = acc.get(k, 0.0) + sign * v
        out = {}
        for k, v in acc.items():
            kk = k if isinstance(k, tuple) else (k,)
            if len(kk) > 1 and any(c != 0 for c in kk[1:]):
                continue
            pk = kk[0] if kk else 0
            out[pk] = out.get(pk, 0.0) + v
        return out

    T, N = 90.0, 2000
    tot = {}
    for i in range(N):
        for k, v in panel(T * i / N, T / N).items():
            tot[k] = tot.get(k, 0.0) + v
    derived = [tot.get(-n, 0.0) for n in range(12)]

    for n in range(8):
        want = (-1) ** n * math.factorial(n)
        t.close(f"R6.0{n} coefficient of eps^{n} derived from the integral",
                derived[n], want, tol=1e-8 * abs(want))
    for k in (1, 3, 5):
        v, (P, Q) = resum(derived, 1.0, k, k)
        t.close(f"R6.1{k} S(1) from DERIVED coefficients via [{k}/{k}]",
                float(v), TARGET, tol=1e-12)
        # The value alone is not enough: a [2/2] built on a contaminated
        # coefficient once returned the right number from an approximant whose
        # pole had drifted to 0.  The pole is what exposes that.
        t.close(f"R6.2{k} and its pole is still at -1", poles(Q)[0].real, -1.0,
                tol=1e-6)


def run_all():
    t = Suite()
    for fn in (r1_polynomials, r2_borel, r3_pade, r4_value, r5_flat,
               r6_end_to_end):
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
