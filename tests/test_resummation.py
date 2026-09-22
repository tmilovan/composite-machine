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
from decimal import Decimal, getcontext

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO
from composite.resummation import (borel, pade, polydiv, poly, degree, poles,
                                   dpoly, borel_singularity, flat_term, resum,
                                   classify_singularity, SingularityKind,
                                   _balance, resum_lateral, resum_median,
                                   _on_ray, _blocking, poles as _poles)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

TARGET = 0.5963473623231941          # S(1) = e * E1(1)
STIELTJES = [(-1) ** n * math.factorial(n) for n in range(24)]

# A second series whose Borel transform has a CUT instead of a pole.
#   B(u) = 1/sqrt(1+u) = sum b_n u^n,  b_n = (-1)^n (2n)! / (4^n (n!)^2)
#   c_n  = n! b_n                      -- still factorially divergent
#   S(1) = int_0^inf exp(-t)/sqrt(1+t) dt = e*sqrt(pi)*erfc(1)
# Pade cannot reproduce a cut at any finite order, so it lays a CHAIN of poles
# along it.  Everything downstream of the chain's leading pole -- most of all
# its residue -- is then an artefact of the order, not a property of B.
BRANCH = [math.factorial(n) * ((-1) ** n * math.factorial(2 * n)
                               / (4.0 ** n * math.factorial(n) ** 2))
          for n in range(26)]
BRANCH_TARGET = math.e * math.sqrt(math.pi) * math.erfc(1.0)


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


# =============================================================================
def r7_branch_point(t):
    head("R7  a CUT, not a pole -- where the singularity reader lied")

    # (a) The resummation itself.  Every earlier test ran on a Borel transform
    # that is RATIONAL, so [1/1] was exact and the Pade layer was never
    # exercised -- passing told us nothing about approximation.  1/sqrt(1+u) is
    # not rational, so here the order has to earn the digits.
    errs = []
    for i, k in enumerate((1, 2, 4, 6, 8, 10), 1):
        v, _ = resum(BRANCH, 1.0, k, k)
        errs.append(abs(float(v) - BRANCH_TARGET))
        t.true(f"R7.0{i} [{k}/{k}] -> {float(v):.14f}  "
               f"(want {BRANCH_TARGET:.14f}, err {errs[-1]:.2e})",
               errs[-1] < 2e-2, f"err {errs[-1]:.3e}")
    t.true(f"R7.09 error falls with order: {errs[0]:.1e} -> {errs[-1]:.1e}",
           errs[-1] < errs[0] / 1e6, f"{errs[0]:.3e} -> {errs[-1]:.3e}")

    # (b) What the reader used to do with it.  borel_singularity(BRANCH, 8, 8)
    # returned z0 = -1.008587 with residue 0.118657 -- a confident number for
    # an object that has no residue at all, and nothing on it to say so.
    kind, z0, res, det = classify_singularity(BRANCH, 8, 8)
    t.true(f"R7.10 classified CUT (got {kind!r}, z0 {z0.real:+.6f})",
           kind == SingularityKind.CUT, f"kind {kind!r}")
    t.true(f"R7.11 residue REFUSED (got {res!r}, was 0.118657)",
           res is None, f"residue {res!r}")
    t.close("R7.12 branch point still located near -1", z0.real, -1.0, tol=0.02)

    # The three signals it is read from, each a number.
    t.true(f"R7.13 a chain, not one pole: {det['roots']} roots",
           det["roots"] > 1, f"roots {det['roots']}")
    t.true(f"R7.14 nearest two clustered: gap/|z0| = {det['gap_ratio']:.4f}",
           det["gap_ratio"] < 0.5, f"gap ratio {det['gap_ratio']:.4f}")
    t.true(f"R7.15 leading root drifts with order: {det['drift']:.2e}",
           det["drift"] > 1e-9, f"drift {det['drift']:.3e}")

    # Drift, at full length: a pole sits still, a chain creeps inward.
    zs = [classify_singularity(BRANCH, k, k)[1].real for k in (4, 6, 8, 10)]
    t.true("R7.16 z0 across [4/4],[6/6],[8/8],[10/10] = "
           + ", ".join(f"{z:+.4f}" for z in zs),
           all(zs[i + 1] > zs[i] for i in range(len(zs) - 1)),
           f"not monotone inward: {zs}")

    # (c) The pole case must be untouched by any of this.
    kind, z0, res, det = classify_singularity(STIELTJES, 8, 8)
    t.true(f"R7.20 Stieltjes still classified POLE (got {kind!r})",
           kind == SingularityKind.POLE, f"kind {kind!r}")
    t.close("R7.21 at exactly -1", z0.real, -1.0)
    t.close("R7.22 with residue 1", abs(res), 1.0)
    t.true(f"R7.23 one root, no chain ({det['roots']})", det["roots"] == 1,
           f"roots {det['roots']}")
    zs = [classify_singularity(STIELTJES, k, k)[1].real for k in (2, 4, 6, 8)]
    t.true("R7.24 and it does NOT move with order: "
           + ", ".join(f"{z:+.6f}" for z in zs),
           all(abs(z + 1.0) < 1e-9 for z in zs), f"{zs}")

    # (d) flat_term downstream.  Its size IS a residue, so on a cut it has
    # nothing to compute -- it must say that rather than return 0.118657-based
    # arithmetic that looks like an answer.
    try:
        flat_term(BRANCH, -1.0, 8, 8)
        t.true("R7.30 flat_term refuses a residue it does not have", False,
               "returned a value")
    except NotImplementedError as e:
        t.true("R7.30 flat_term raises NotImplementedError on the cut",
               "BRANCH POINT" in str(e), str(e)[:60])
    u0, amb = flat_term(STIELTJES, -1.0, 8, 8)
    want = math.pi * math.exp(-1.0)
    t.close(f"R7.31 and is unchanged on the pole: {amb:.15f}", amb, want,
            tol=1e-14 * want)


# =============================================================================
def _bernoulli(nt=24):
    """B_n from ONE composite division: h/(exp(h)-1) is their generating
    function, and the library's division is series division."""
    q = ZERO / (cl.exp(ZERO, terms=nt) - R(1))
    return {-int(k): v * math.factorial(-int(k))
            for k, v in q.coeffs_dict().items() if 0 <= -int(k) <= nt}


def _odd_series(vals):
    """[a1, a2, ...] -> coefficients of sum a_n eps^(2n-1)."""
    c = [0.0] * (2 * len(vals))
    for n, v in enumerate(vals, 1):
        c[2 * n - 1] = v
    return c


def r8_stirling(t):
    head("R8  Stirling/Binet -- coefficients that span 20 decades")
    NT = 24
    B = _bernoulli(NT)
    for n, want in ((6, 1 / 42), (12, -691 / 2730), (20, -174611 / 330)):
        t.close(f"R8.0{n} B_{n} from composite division h/(exp(h)-1)",
                B[n], want, tol=1e-12 * abs(want))

    # mu(z) = lnGamma(z) - [(z-1/2)ln z - z + (1/2)ln 2pi],  Binet's function.
    a = [B[2 * n] / (2 * n * (2 * n - 1)) for n in range(1, NT // 2 + 1)]
    c = _odd_series(a)
    mu = lambda z: (math.lgamma(z)
                    - ((z - 0.5) * math.log(z) - z + 0.5 * math.log(2 * math.pi)))

    # The series diverges: at z=1 the best truncation is 2.9e-04 and every
    # later partial sum is worse.  Borel-Pade goes three orders past it.
    for i, (z, tol) in enumerate(((2.0, 1e-8), (1.0, 1e-6), (0.5, 1e-4)), 1):
        s, best = 0.0, None
        for n in range(1, len(a) + 1):
            s += a[n - 1] * z ** (-(2 * n - 1))
            best = abs(s - mu(z)) if best is None else min(best, abs(s - mu(z)))
        v, _ = resum(c, 1.0 / z, 6, 6)
        t.close(f"R8.1{i} mu({z}) via [6/6] (best partial sum was {best:.1e})",
                float(v), mu(z), tol=tol)

    # THE HANG.  Unbalanced, b_1 = 8.3e-02 and b_23 = -6.1e-21 were weighed
    # against one absolute tolerance; b_15 = -2.3e-14 and below were deleted as
    # "zero", poly(b[:15]) came back degree 13, and the Euclidean algorithm
    # cycled 13 -> 11 -> 13 -> 11 forever.  [7/7] did not fail, it HUNG.
    rho = _balance(borel(c))
    # Cauchy-Hadamard from 24 coefficients, so an ESTIMATE of the radius: it
    # overshoots 2*pi by 20%.  It does not need to be sharp -- its whole job is
    # to put the coefficients on one footing, and 20% off does that.
    t.true(f"R8.20 balance factor {rho:.6f} estimates the radius 2*pi = "
           f"{2 * math.pi:.6f}, {abs(rho - 2 * math.pi) / (2 * math.pi):.1%} high",
           abs(rho - 2 * math.pi) / (2 * math.pi) < 0.25,
           f"rho {rho:.6f} vs {2 * math.pi:.6f}")
    bs = [v * rho ** k for k, v in enumerate(borel(c))]
    lo = min(abs(v) for v in bs if v)
    hi = max(abs(v) for v in bs)
    t.true(f"R8.21 balanced coefficients span {lo:.1e}..{hi:.1e} "
           f"(raw: 6.1e-21..8.3e-02)", hi / lo < 1e3, f"spread {hi / lo:.1e}")
    errs = []
    for k in (6, 8, 10, 11):
        v, _ = resum(c, 1.0, k, k)
        errs.append(abs(float(v) - mu(1.0)))
        t.true(f"R8.3{k} [{k}/{k}] runs (it hung at [7/7]) -> {float(v):.15f}, "
               f"err {errs[-1]:.2e}", errs[-1] < 1e-6, f"err {errs[-1]:.2e}")
    t.true(f"R8.39 and keeps improving: {errs[0]:.1e} -> {errs[-1]:.1e}",
           errs[-1] < errs[0] / 100, f"{errs[0]:.2e} -> {errs[-1]:.2e}")

    # THE GUARD, tested on the input that actually hung -- balancing switched
    # off, so the Euclidean algorithm sees exactly what it saw before: degrees
    # read off coefficients 20 decades apart.  It must REFUSE, not cycle.
    import composite.resummation as _rs
    _real = _rs._balance
    _rs._balance = lambda b: 1.0
    try:
        pade(borel(c), 7, 7)
        t.true("R8.40 unbalanced [7/7] refuses instead of cycling 13->11->13",
               False, "returned an approximant instead of refusing")
    except ArithmeticError as e:
        t.true("R8.40 unbalanced [7/7] REFUSES instead of cycling 13->11->13: "
               + str(e)[:52] + "...", "stopped shrinking" in str(e), str(e)[:70])
    finally:
        _rs._balance = _real
    v, _ = resum(c, 1.0, 7, 7)
    t.close("R8.41 and balanced, the same [7/7] returns mu(1)", float(v),
            mu(1.0), tol=1e-7)

    # THE CLASSIFICATION.  Truth: B' is Binet's kernel, with SIMPLE POLES at
    # 2*pi*i*k of residue 1/(2*pi*i); B is its antiderivative, so B has
    # LOGARITHMS there -- branch points, no residue.  Confirmed independently
    # from the coefficients: k*|b_k|*(2pi)^k -> 2/(2pi) = 1/pi, which is what a
    # log branch point at radius 2pi gives and a pole does not.
    b = borel(c)
    for k in (17, 19, 21):
        t.close(f"R8.4{k} k|b_k|(2pi)^k -> 1/pi: a LOG at radius 2pi",
                k * abs(b[k]) * (2 * math.pi) ** k, 1 / math.pi, tol=1e-5)
    for L in (4, 6, 8, 10):
        kind, z0, res, det = classify_singularity(c, L, L)
        t.true(f"R8.5{L} [{L}/{L}] kind={kind} |z0|={abs(z0):.5f} "
               f"rate={det.get('rate', float('nan')):.3f} residue={res}",
               kind == SingularityKind.CUT and res is None,
               f"kind {kind!r}, residue {res!r}")
    t.true("R8.59 the residue SLIDES toward zero, as a branch point's must: "
           + ", ".join("%.4f" % classify_singularity(c, L, L)[3]["residue"].real
                       for L in (4, 6, 8, 10)),
           (classify_singularity(c, 10, 10)[3]["residue"].real
            < classify_singularity(c, 4, 4)[3]["residue"].real / 3), "not sliding")


def r9_pole_string(t):
    head("R9  a chain is not a verdict -- three kinds of it, told apart")
    NT = 24
    B = _bernoulli(NT)

    # (a) RATIONAL, a conjugate PAIR of simple poles: B(u) = 1/(1+(u/2pi)^2).
    # Pade reproduces it exactly, so the roots never move.  This is also the
    # case that showed the old clustering test was dead: rs[1] is rs[0]'s own
    # mirror, so the gap ratio was 2.000 by construction and said nothing.
    cr = [0.0] * 24
    for n in range(12):
        if 2 * n < 24:
            cr[2 * n] = math.factorial(2 * n) * ((-1) ** n / (2 * math.pi) ** (2 * n))
    for L in (3, 5, 8):
        kind, z0, res, det = classify_singularity(cr, L, L)
        t.true(f"R9.0{L} [{L}/{L}] conjugate pair -> {kind}, z0 = {abs(z0):.6f}i",
               kind == SingularityKind.POLE, f"kind {kind!r}")
        t.close(f"R9.1{L} at 2*pi*i", abs(z0), 2 * math.pi, tol=1e-9)
        t.close(f"R9.2{L} residue = -pi*i (exact, rational)", res.imag, -math.pi,
                tol=1e-9)
        t.true(f"R9.3{L} rs[1] is rs[0]'s mirror, so no gap ratio is quoted "
               f"(it was a constant 2.000)",
               det.get("conjugate_pair") is True and "gap_ratio" not in det,
               f"conjugate_pair={det.get('conjugate_pair')}, "
               f"gap_ratio={det.get('gap_ratio')}")

    # (b) an INFINITE STRING of simple poles: B(u) = 1/(e^u-1) - 1/u + 1/2,
    # poles at 2*pi*i*k with residue 1.  Pade lays a chain here TOO -- exactly
    # as it does over a branch cut -- so the chain itself decides nothing.
    cs = _odd_series([B[2 * n] / (2 * n) for n in range(1, NT // 2 + 1)])
    TGT = 0.5772156649015329 - 0.5      # S(1) = -mu'(1) = -(psi(1) + 1/2)
    for L in (6, 8, 11):
        v, _ = resum(cs, 1.0, L, L)
        t.close(f"R9.4{L} S(1) = -mu'(1) via [{L}/{L}]", float(v), TGT, tol=1e-7)
    for L in (4, 6):
        kind, z0, res, det = classify_singularity(cs, L, L)
        t.true(f"R9.5{L} [{L}/{L}] converging but not settled -> {kind} "
               f"(rate {det.get('rate', float('nan')):.3f}, residue still moving "
               f"{det.get('res_change', float('nan')):.1e})",
               kind == SingularityKind.UNRESOLVED and res is None,
               f"kind {kind!r}")
    for L in (8, 10, 11):
        kind, z0, res, det = classify_singularity(cs, L, L)
        t.true(f"R9.6{L} [{L}/{L}] settled -> {kind}", kind == SingularityKind.POLE,
               f"kind {kind!r}, why {det.get('why')}")
        t.close(f"R9.7{L} at 2*pi*i", abs(z0), 2 * math.pi, tol=1e-4)
        t.close(f"R9.8{L} residue = 1 (a pole INSIDE a chain, recovered)",
                res.real, 1.0, tol=1e-3)

    # (c) with nothing below to compare against, say so rather than guess.
    kind, z0, res, det = classify_singularity(cr, 2, 2)
    t.true(f"R9.90 [2/2] no lower order -> {kind}, not a confident 'cut'",
           kind == SingularityKind.UNRESOLVED, f"kind {kind!r}")
    ok, msg = _refuses(flat_term, cs, 1.0, 6, 6, word="not been resolved")
    t.true("R9.91 flat_term on the unresolved string: " + msg, ok, msg)
    ok, msg = _refuses(flat_term, _odd_series(
        [B[2 * n] / (2 * n * (2 * n - 1)) for n in range(1, NT // 2 + 1)]),
        1.0, 6, 6, word="BRANCH POINT")
    t.true("R9.92 flat_term on the log branch point: " + msg, ok, msg)


def _refuses(fn, *args, word="", **kw):
    """(did it refuse for the stated reason, what it actually said)."""
    try:
        v = fn(*args, **kw)
        return False, f"did NOT refuse, returned {v!r}"
    except NotImplementedError as e:
        txt = " ".join(str(e).split())
        return word in txt, f"refused with {txt[:58]!r}..."


# =============================================================================
def r10_painleve(t):
    head("R10  Painleve I -- a NONLINEAR source, on a blocked ray")
    # y'' = 6y^2 - t,  t -> +inf,  y = sum a_k t^((1-5k)/2).
    #   y''  contributes a_k (25k^2-1)/4 t^((-3-5k)/2)
    #   6y^2 contributes 6 S_m t^((2-5m)/2), S_m = sum_{i+j=m} a_i a_j
    # and the powers meet at m = k+1, so
    #   a_k (25k^2-1)/4 = 6 [ 2 a_0 a_{k+1} + sum_{i=1..k} a_i a_{k+1-i} ].
    # Every other series here came from an integral, which is linear.  This one
    # comes from a QUADRATIC recursion, and that sum over i is a convolution --
    # which is what composite multiplication IS.  So the composite computes it.
    N = 42
    a0 = 1.0 / math.sqrt(6.0)
    a = [a0]
    for k in range(N):
        Y = Composite({-i: v for i, v in enumerate(a)})   # y in u = t^(-5/2)
        conv = (Y * Y).coeffs_dict().get(-(k + 1), 0.0)   # only i,j <= k appear
        a.append((a[k] * (25 * k * k - 1) / 4.0 - 6.0 * conv) / (12.0 * a0))

    # Same recursion at 60 decimal digits, convolution written out by hand.
    getcontext().prec = 60
    d0 = Decimal(1) / Decimal(6).sqrt()
    d = [d0]
    for k in range(N):
        cv = sum((d[i] * d[k + 1 - i] for i in range(1, k + 1)), Decimal(0))
        d.append((d[k] * Decimal(25 * k * k - 1) / 4 - 6 * cv) / (12 * d0))

    t.close("R10.01 a_1 = -1/48 from the composite convolution", a[1], -1 / 48.0,
            tol=1e-15)
    for k in (5, 11, 20):
        t.close(f"R10.0{k} a_{k} vs the same recursion at 60 digits",
                a[k], float(d[k]), tol=1e-13 * abs(float(d[k])))

    # The action, THREE independent ways.
    # (i) WKB on the ODE: y = y0 + delta about y0 = sqrt(t/6) gives
    #     delta'' = 12 y0 delta = 2 sqrt(6) sqrt(t) delta, and the exponent is
    #     int sqrt(2 sqrt 6) t^(1/4) dt = (4/5) sqrt(2 sqrt 6) t^(5/4).
    A = 0.8 * math.sqrt(2 * math.sqrt(6))
    t.close("R10.10 A from WKB = (4/5)sqrt(2 sqrt 6)", A, 1.770691071520514,
            tol=1e-14)

    # (ii) from the coefficients: a_k ~ Gamma(2k+beta)/A^(2k), so the ratio
    #      |a_{k+1}/a_k| ~ (2k)(2k+1)/A^2.  The estimate falls as A + c/k;
    #      one Richardson step removes the c/k.
    est = [math.sqrt((2 * k) * (2 * k + 1) / abs(a[k + 1] / a[k]))
           for k in range(4, N)]
    k_last = 4 + len(est) - 1
    A_rich = est[-1] + (k_last - 1) * (est[-1] - est[-2])
    t.close(f"R10.11 A from coefficient growth: raw {est[-1]:.6f} at k={k_last}, "
            f"Richardson {A_rich:.6f}", A_rich, A, tol=1e-3)

    # (iii) from where optimal truncation stops: the smallest term of the
    #       series is exp(-A t^(5/4)) up to an algebraic prefactor, so the log
    #       ratio approaches 1.
    # Only where the series HAS its smallest term: the optimum sits near
    # k = A t^(5/4)/2, so a t large enough to put it past the last coefficient
    # measures the end of the list instead, and the ratio then falls away from
    # 1 for a reason that has nothing to do with A (t=40 gave 0.672 on 26
    # coefficients).  Skip those rather than read them.
    ratios = []
    for tt in (6.0, 10.0, 14.0, 20.0):
        terms = [abs(a[k] * tt ** ((1 - 5 * k) / 2.0)) for k in range(len(a))]
        bn = min(range(1, len(terms)), key=lambda i: terms[i])
        if bn >= len(terms) - 2:
            continue                      # optimum is past the coefficients
        ratios.append((tt, math.log(terms[bn]) / (-A * tt ** 1.25)))
    t.true("R10.12 A from optimal truncation: log(smallest term)/(-A t^5/4) = "
           + ", ".join(f"{r:.4f} (t={tt:g})" for tt, r in ratios) + " -> 1",
           len(ratios) >= 3 and ratios[-1][1] < ratios[0][1]
           and abs(ratios[-1][1] - 1) < 0.1, f"{ratios}")

    # WHAT KIND of singularity, from the coefficients alone.  If
    # B(v) ~ (1 - v/A)^(-beta) then b_n ~ n^(beta-1) A^-n, so the slope of
    # log|b_n|A^n against log n is beta-1.  beta = 1 is a pole.
    c = [0.0] * (2 * len(a))
    for k, v in enumerate(a):
        if 2 * k < len(c):
            c[2 * k] = v
    r = lambda k: abs(a[k]) * A ** (2 * k) / math.factorial(2 * k)
    slopes = []
    for k in (18, 22, 24):
        n0, n1 = 2 * (k - 2), 2 * k
        slopes.append((math.log(r(k)) - math.log(r(k - 2)))
                      / (math.log(n1) - math.log(n0)))
    t.true("R10.20 beta from coefficients = "
           + ", ".join(f"{sl + 1:+.4f}" for sl in slopes)
           + " -> -1/2: B(v) ~ (A-v)^(1/2), a SQUARE-ROOT branch point, "
             "so no residue exists",
           all(abs(sl + 1 + 0.5) < 0.02 for sl in slopes), f"{slopes}")

    # The series is EVEN in w = t^(-5/4), so odd orders have no approximant at
    # all -- every one raises "denominator vanishes at 0".  Stepping the
    # comparison order down by a fixed 1 landed on those, came back with no
    # history, and the verdict was then reached on no evidence.
    for L in (3, 5, 7):
        try:
            pade(borel(c), L, L)
            t.true(f"R10.3{L} [{L}/{L}] on an even series", False, "did not raise")
        except ValueError:
            t.true(f"R10.3{L} [{L}/{L}] has no approximant (even series) "
                   f"-- the walk skips it", True, "raised")
    for L in (6, 8, 10):
        kind, z0, res, det = classify_singularity(c, L, L)
        t.true(f"R10.4{L} [{L}/{L}] kind={kind} z0={z0.real:+.6f} "
               f"rate={det.get('rate', float('nan')):.3f} "
               f"(compared {det.get('orders_compared')} orders, skipping odd)",
               kind == SingularityKind.CUT and res is None
               and det.get("orders_compared") == 3,
               f"kind {kind!r}, res {res!r}, orders {det.get('orders_compared')}")
    zs = [abs(classify_singularity(c, L, L)[1]) for L in (4, 6, 8, 10)]
    t.true("R10.49 |z0| closes on A from above: "
           + ", ".join(f"{z:.6f}" for z in zs) + f" -> A = {A:.6f}",
           all(zs[i + 1] < zs[i] for i in range(len(zs) - 1)) and zs[-1] > A,
           f"{zs}")

    # THE RAY IS BLOCKED.  Every a_k past the first has one sign, so the Borel
    # singularity sits on the POSITIVE axis -- which is the Laplace path.  The
    # series is not Borel summable in this direction at all.
    t.true("R10.50 a_k all one sign past a_0 => the singularity is on the ray",
           all(v < 0 for v in a[1:12]), f"{a[1:5]}")
    for tt in (6.0, 10.0):
        w = tt ** -1.25
        ok, msg = _refuses(resum, c, w, 6, 6, word="BLOCKED")
        t.true(f"R10.6{int(tt)} resum at t={tt} refuses the blocked ray: {msg}",
               ok, msg)
    # It used to answer.  At [6/6] and t=10 it returned 0.40818214484148, which
    # is right to 8 digits; at [4/4] it returned 1.4e+43.  Nothing on either
    # said which was which.
    y_over_sqrt_t = sum(a[k] * 10.0 ** ((1 - 5 * k) / 2.0)
                        for k in range(17)) / math.sqrt(10.0)
    t.close("R10.70 (for the record) optimal truncation at t=10 gives "
            "y/sqrt(t) = 0.40818214", y_over_sqrt_t, 0.4081821448, tol=1e-9)

    # And the ambiguity cannot be sized either, for a reason that is not the
    # ray: a square-root branch point has no residue.  Both refusals stand.
    ok, msg = _refuses(flat_term, c, 10.0 ** -1.25, 8, 8, word="BRANCH POINT")
    t.true("R10.80 flat_term refuses: the exponent is exp(-A t^5/4) but the "
           "SIZE is not a residue. " + msg, ok, msg)


# =============================================================================
def _p1_coeffs(N=42):
    a0 = 1.0 / math.sqrt(6.0)
    a = [a0]
    for k in range(N):
        Y = Composite({-i: v for i, v in enumerate(a)})
        conv = (Y * Y).coeffs_dict().get(-(k + 1), 0.0)
        a.append((a[k] * (25 * k * k - 1) / 4.0 - 6.0 * conv) / (12.0 * a0))
    c = [0.0] * (2 * len(a))
    for k, v in enumerate(a):
        if 2 * k < len(c):
            c[2 * k] = v
    return a, c


def r11_lateral(t):
    head("R11  lateral summation -- a number off a blocked ray")

    # THE CALIBRATION.  Euler-Stieltjes at eps = -1 puts the pole at u0 = 1,
    # right on the Laplace path, and the answer is known in closed form:
    #   PV = e^-1 Ei(1),  and the two sides differ by 2*pi*i*Res.
    PV = math.exp(-1.0) * 1.8951178163559368
    IM = math.pi / math.e
    sp = resum_lateral(STIELTJES, -1.0, 8, 8, +1)
    sm = resum_lateral(STIELTJES, -1.0, 8, 8, -1)
    t.true(f"R11.01 S+ and S- are exact conjugates "
           f"(|S+ - conj(S-)| = {abs(sp - sm.conjugate()):.1e})",
           abs(sp - sm.conjugate()) < 1e-14, f"{abs(sp - sm.conjugate()):.2e}")
    med, amb = resum_median(STIELTJES, -1.0, 8, 8)
    t.close("R11.02 median = PV = e^-1 Ei(1)", med.real, PV, tol=1e-8)
    t.true(f"R11.03 median is real (Im = {med.imag:.1e})", abs(med.imag) < 1e-14,
           f"{med.imag:.2e}")
    t.close("R11.04 half the jump = pi/e, the flat term", amb, IM, tol=1e-8)
    # The same number by a completely different route: the RESIDUE.
    _, amb_res = flat_term(STIELTJES, -1.0, 8, 8)
    t.close(f"R11.05 and it agrees with the residue route ({amb_res:.15f})",
            amb, amb_res, tol=1e-8)

    # theta is free analytically and is not free numerically: the contour
    # passes the pole at distance u0 sin(theta), and a panel wider than that
    # expands a divergent Taylor series.  pi/3 and 1.2 rad both sit in the
    # good band; pi/6 does not.
    for th, ok in ((math.pi / 3, True), (1.2, True), (math.pi / 6, False)):
        v = resum_lateral(STIELTJES, -1.0, 8, 8, +1, theta=th)
        e = abs(v - complex(PV, IM))
        t.true(f"R11.1{int(math.degrees(th))} theta={math.degrees(th):.0f} deg "
               f"(pole at distance {math.sin(th):.3f}): err {e:.1e}"
               + ("" if ok else "  -- OUTSIDE the usable band, as measured"),
               (e < 1e-8) == ok, f"err {e:.2e}, expected ok={ok}")

    # PAINLEVE I.  resum refuses this ray; the contour gets a number anyway.
    a, c = _p1_coeffs()
    A = 0.8 * math.sqrt(2 * math.sqrt(6))

    def trunc(tt):
        tm = [a[k] * tt ** ((1 - 5 * k) / 2.0) for k in range(len(a))]
        k = min(range(1, len(tm)), key=lambda i: abs(tm[i]))
        return sum(tm[:k + 1]), abs(tm[k]), k

    ok, msg = _refuses(resum, c, 6.0 ** -1.25, 8, 8, word="BLOCKED")
    t.true("R11.20 resum still refuses the straight ray: " + msg, ok, msg)

    # Where optimal truncation is trustworthy, the lateral sum must match it.
    for tt, tol in ((14.0, 1e-9), (10.0, 1e-9)):
        med, _ = resum_median(c, tt ** -1.25, 8, 8)
        y = med.real * math.sqrt(tt)
        yt, er, k = trunc(tt)
        t.close(f"R11.2{int(tt)} y({tt}) lateral vs optimal truncation "
                f"(k*={k}, its error {er:.1e})", y, yt, tol=tol)

    # Where it is NOT: the orders must agree with each other, which is the
    # only convergence signal available -- the Painleve I tritronquee is a
    # SADDLE, so no initial-value integration can check it.  The exp(-A t^5/4)
    # mode grows as t decreases, amplifying double-precision noise by
    # exp(A(20^5/4 - 6^5/4)) = 1e+25 on the way down, and the growing mode
    # does the same on the way up.  There is no oracle to integrate to.
    for tt, want in ((5.0, 1e-8), (3.0, 1e-5)):
        vals = [resum_median(c, tt ** -1.25, L, L)[0].real * math.sqrt(tt)
                for L in (6, 8, 10)]
        spread = max(vals) - min(vals)
        yt, er, k = trunc(tt)
        t.true(f"R11.3{int(tt)} y({tt}) = {vals[1]:.12f}: orders [6/6],[8/8],"
               f"[10/10] agree to {spread:.1e}, inside truncation's own error "
               f"bar of {er:.1e} around {yt:.12f}",
               spread < want and abs(vals[1] - yt) < 3 * er,
               f"spread {spread:.2e}, |lateral-trunc| {abs(vals[1] - yt):.2e}, "
               f"trunc err {er:.2e}")

    # THE FLAT TERM, from the contour -- which flat_term cannot give here,
    # because a square-root branch point has no residue.  Its size must carry
    # the exponential the ODE predicted, exp(-A t^(5/4)).
    ratios = []
    for tt in (3.0, 4.0, 5.0, 6.0):
        _, amb = resum_median(c, tt ** -1.25, 8, 8)
        ratios.append(amb * math.sqrt(tt) / math.exp(-A * tt ** 1.25))
    spread = max(ratios) / min(ratios) - 1.0
    t.true("R11.40 |S+-S-|/2 / exp(-A t^5/4) = "
           + ", ".join(f"{r:.4f}" for r in ratios)
           + f" -- constant to {spread:.1%} while the flat term itself moves "
             f"from 9.2e-04 to 6.0e-08", spread < 0.10, f"spread {spread:.3f}")
    ok, msg = _refuses(flat_term, c, 6.0 ** -1.25, 8, 8, word="BRANCH POINT")
    t.true("R11.41 while the residue route still refuses, correctly: " + msg,
           ok, msg)


# =============================================================================
def r12_negative_controls(t):
    head("R12  negative controls -- three ways for the layer to be WRONG")
    import random

    # (1) A CONVERGENT SERIES.  sum eps^n = 1/(1-eps), Borel transform e^u,
    # entire -- singular NOWHERE.  A Pade denominator always has roots, and
    # the approximants to exp put one on the positive axis at odd order
    # (+4.644 at [3/3], +7.293 at [5/5]).  Reading roots alone, the layer
    # refused a sum nothing was blocking.
    conv = [1.0] * 24
    zs = [abs(_poles(pade(borel(conv), L, L)[1])[0]) for L in (2, 3, 4, 5, 6)]
    t.true("R12.01 an entire transform's Pade poles RECEDE: "
           + ", ".join(f"{z:.3f}" for z in zs)
           + " -- a real singularity converges instead",
           all(zs[i + 1] > zs[i] for i in range(len(zs) - 1)), f"{zs}")
    errs = []
    for L in (2, 3, 4, 5, 6):
        v, _ = resum(conv, 0.5, L, L)
        errs.append(abs(float(v) - 2.0))
        t.true(f"R12.1{L} [{L}/{L}] sums to {float(v):.15g}, err {errs[-1]:.1e} "
               f"(want 2.0) -- no order refuses", errs[-1] < 0.2, f"{float(v)}")
    t.true(f"R12.19 and it converges: {errs[0]:.1e} -> {errs[-1]:.1e}",
           errs[-1] < errs[0] / 100, f"{errs}")

    # (2) A SINGULARITY OFF THE RAY.  B(u) = 1/(1+u^2): poles at +-i, ninety
    # degrees from the Laplace path.  Must sum, must not refuse, must claim no
    # ambiguity.
    off = [0.0] * 26
    for k in range(13):
        if 2 * k < 26:
            off[2 * k] = float(math.factorial(2 * k)) * ((-1) ** k)
    # truth by a completely separate route: the defining integral
    ref = {1.0: 0.6214496242358132, 0.5: 0.7980419771883682,
           0.25: 0.9167702720981078}          # mpmath.quad, 50 dps
    for eps, want in sorted(ref.items()):
        v, _ = resum(off, eps, 8, 8)
        t.close(f"R12.2{int(100 * eps)} eps={eps}: {float(v):.15f} vs the "
                f"defining integral", float(v), want, tol=1e-12)
    kind, z0, res, det = classify_singularity(off, 8, 8)
    t.close("R12.28 and the singularity is found at +i, exactly", abs(z0), 1.0,
            tol=1e-12)
    t.true(f"R12.29 nothing on the ray, so flat_term claims no ambiguity: "
           f"{flat_term(off, 1.0, 8, 8)}", flat_term(off, 1.0, 8, 8) == (None, 0.0),
           f"{flat_term(off, 1.0, 8, 8)}")

    # (3) A FROISSART DOUBLET.  Relative noise on clean coefficients throws a
    # near-cancelling pole/zero pair; at 1e-11 it lands at -0.819/+0.825 and
    # the positive half blocked a sum good to 6.9e-13.
    for noise, tol in ((1e-14, 1e-12), (1e-11, 1e-6), (1e-8, 1e-3)):
        rng = random.Random(7)
        noisy = [c * (1 + noise * rng.uniform(-1, 1)) for c in STIELTJES]
        on = _on_ray(_poles(pade(borel(noisy), 8, 8)[1]), 1.0)
        v, _ = resum(noisy, 10, 10)[0] if False else resum(noisy, 1.0, 10, 10)
        e = abs(float(v) - TARGET)
        t.true(f"R12.3 noise {noise:g}: {len(on)} spurious root(s) on the ray, "
               f"and [10/10] still sums to {float(v):.12f}, err {e:.1e}",
               e < tol, f"err {e:.2e}")

    # THE DISCRIMINATOR.  A root that does not persist one order down is an
    # artefact; one that does is a singularity.  Relative distance to the
    # nearest root at L-1:
    #     Stieltjes genuine 0.000   sqrt branch point 0.002
    #     entire [3/3] 0.51         Froissart [8/8] 0.66
    b = borel(STIELTJES)
    t.true("R12.40 a GENUINE on-ray pole still blocks: Stieltjes at eps=-1",
           len(_blocking(b, 8, 8, -1.0, _poles(pade(b, 8, 8)[1]))) == 1,
           "should block")
    bc = borel(conv)
    t.true("R12.41 an ARTEFACT does not: the entire transform at [3/3]",
           _blocking(bc, 3, 3, 0.5, _poles(pade(bc, 3, 3)[1])) == [],
           "should not block")
    ok, msg = _refuses(resum, STIELTJES, -1.0, 8, 8, word="BLOCKED")
    t.true("R12.42 and the genuine case still refuses by name: " + msg, ok, msg)


def run_all():
    t = Suite()
    for fn in (r1_polynomials, r2_borel, r3_pade, r4_value, r5_flat,
               r6_end_to_end, r7_branch_point, r8_stirling,
               r9_pole_string, r10_painleve, r11_lateral,
               r12_negative_controls):
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
