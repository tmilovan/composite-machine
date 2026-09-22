#!/usr/bin/env python3
# Composite Machine — transseries: a value group that can hold exp(-1/h)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""The scale that powers and logs cannot reach.

Composite's value group is powers and iterated logarithms -- the
powers-and-logs fragment of a Hardy field.  It has NO slot below every power,
and exp(-1/h) is exactly there.  That is not a gap in the implementation; the
object genuinely does not exist in that group.

Which is why the same asymptotic series belongs to a whole family of
functions:

    S(eps)  and  S(eps) + C*exp(-1/eps)

have IDENTICAL coefficients to all orders, because exp(-1/eps) expands to
zero.  No number of coefficients separates them.  The resummation layer
measured the difference instead -- pi/e = 1.1557273497909217 for
Euler-Stieltjes at eps = -1, by two independent routes -- so there is now a
number to build against rather than a definition to argue with.

ORDERING COMES FIRST, and this file is ordering.  If exp(-1/h) does not
compare correctly against every power, nothing built on top of it is
recoverable -- and the comparison is the whole content of the claim that the
value group has been extended.  Arithmetic is only tested here as far as the
ordering questions need it.

The sector index is in UNITS OF THE ACTION: sector n carries exp(-n/h), so
the indices are integers and their addition is exact.  A float action would
put 3A on a grade that float addition cannot represent, and the library
refuses inexact grade addition -- correctly.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO
from composite.resummation import borel, SingularityKind
from composite.transseries import (Transseries, flat, sector, ts_exp, ts_st,
                                   is_infinitesimal, is_infinite,
                                   resum_sector, sector_coefficients,
                                   ts_d, ts_ln, from_series, action_from_growth)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

h = ZERO                      # the infinitesimal; 1/h is infinite


# =============================================================================
def t1_ordering(t):
    head("T1  ordering -- exp(-1/h) against every power of h")
    e1 = flat(1)              # exp(-1/h)
    big = ts_exp(R(1) / h)    # exp(+1/h)

    # 1. Below EVERY power, not merely below the ones we thought to write.
    for i, n in enumerate((1, 10, 1000), 1):
        t.true(f"T1.0{i} exp(-1/h) < h**{n}", e1 < Transseries.lift(h ** n),
               f"exp(-1/h) vs h**{n}")
    # A WRITTEN 0 IS AN EXPRESSED ZERO, which R1 sends to h -- so `> 0` asks
    # whether the flat term exceeds h, and it does not.  That is the same
    # statement as T1.01 and not a sign test at all; the sign is tested below
    # without going near a zero.
    t.true("T1.01b exp(-1/h) < 0, because a written 0 IS h (R1) and the flat "
           "term is below every power of it", e1 < 0, f"{e1}")
    t.true("T1.01c and its sign is positive, tested without a zero: "
           "exp(-1/h) > -exp(-1/h)", e1 > -e1, f"{e1} vs {-e1}")

    # 2. No power rescues it.  This is the property the whole extension exists
    # for: the sector outranks the power axis outright, so multiplying by
    # h**-1000 -- an INFINITE composite -- leaves it infinitesimal.
    scaled = e1 * Transseries.lift(h ** -1000)
    t.true("T1.02 exp(-1/h) * h**(-1000) is STILL infinitesimal",
           is_infinitesimal(scaled), f"{scaled}")
    t.true("T1.02b and still below h**1000", scaled < Transseries.lift(h ** 1000),
           f"{scaled}")

    # 3. It is not zero.  A power series says it is; that is the error.
    t.true("T1.03 exp(-1/h) == 0 is False", not (e1 == 0), f"{e1}")
    t.true("T1.03b and it is not equal to any power of h",
           not (e1 == Transseries.lift(h ** 1000)), f"{e1}")

    # 4. The other side: exp(+1/h) is beyond every power upward, so it has no
    # standard part at all.
    t.true("T1.04a exp(1/h) is infinite", is_infinite(big), f"{big}")
    t.true("T1.04b exp(1/h) > h**(-1000)", big > Transseries.lift(h ** -1000),
           f"{big}")
    try:
        v = ts_st(big)
        t.true(f"T1.04 st(exp(1/h)) refuses", False, f"returned {v!r}")
    except (ValueError, ArithmeticError) as e:
        t.true("T1.04 st(exp(1/h)) refuses: " + str(e)[:52], True, "raised")

    # 5. The family the coefficients cannot separate, separated.
    S = Transseries.lift(Composite({0: 1.0, -1: -1.0, -2: 2.0, -3: -6.0}))
    C = 1.1557273497909217                       # pi/e, MEASURED, not assumed
    S2 = S + flat(1) * C
    t.true("T1.05 S and S + C*exp(-1/h) compare UNEQUAL", not (S == S2),
           "they must differ")
    t.true("T1.05b with S the smaller (C > 0)", S < S2, "sign of the difference")
    t.true("T1.05c yet their power parts are identical",
           S.sectors[0] == S2.sectors[0], "sector 0 must be untouched")
    t.close("T1.05d and the difference is exactly C", (S2 - S).sectors[1].st(),
            C, tol=1e-15)


# =============================================================================
def t2_sector_arithmetic(t):
    head("T2  sectors multiply by ADDING -- exp(-m/h)exp(-n/h) = exp(-(m+n)/h)")
    t.true("T2.01 exp(-1/h)*exp(-1/h) == exp(-2/h)",
           flat(1) * flat(1) == flat(2), f"{flat(1) * flat(1)}")
    t.true("T2.02 exp(-2/h) < exp(-1/h): deeper sectors are smaller",
           flat(2) < flat(1), "sector order")
    t.true("T2.03 exp(1/h)*exp(-1/h) == 1", ts_exp(R(1) / h) * flat(1) == 1,
           f"{ts_exp(R(1) / h) * flat(1)}")
    t.true("T2.04 sector index must be an INTEGER (units of the action)",
           _refuses(ts_exp, R(-1.7706910715) / h), "a float action must refuse")
    # Within a sector the ordinary series still works, untouched.
    s = flat(1) * Transseries.lift(Composite({0: 2.0, -1: 3.0}))
    t.close("T2.05 a sector carries a full power series: coeff of h^1 is 3",
            s.sectors[1].coeffs_dict().get(-1, 0.0), 3.0, tol=1e-15)
    t.true("T2.06 and that series is an ordinary scalar-dim Composite "
           "(fast path preserved)",
           not any(isinstance(d, tuple) for d in s.sectors[1].coeffs_dict()),
           f"{list(s.sectors[1].coeffs_dict())}")


def _refuses(fn, *a, **k):
    try:
        fn(*a, **k)
        return False
    except (ValueError, ArithmeticError, NotImplementedError):
        return True


# =============================================================================
def t3_against_the_oracle(t):
    head("T3  the flat term is a MEASURED number, not a definition")
    from composite.resummation import resum_median, flat_term
    ST = [(-1) ** n * math.factorial(n) for n in range(24)]
    eps = -1.0
    _, amb = resum_median(ST, eps, 8, 8)      # from the contour
    _, amb_r = flat_term(ST, eps, 8, 8)       # from the residue
    exact = math.pi / math.e

    # The transseries that represents the ambiguity: the prefactor pi/|eps|
    # lives INSIDE sector 1 as an infinite composite, h^-1, and the sector
    # still outranks it.
    T = flat(1) * Transseries.lift(R(math.pi) / h)
    t.true("T3.01 the one-instanton term pi*h^-1*exp(-1/h) is infinitesimal "
           "although its prefactor is INFINITE", is_infinitesimal(T), f"{T}")
    val = T.evaluate(abs(eps))
    t.close(f"T3.02 evaluated at h={abs(eps)}: {val:.15f}", val, exact, tol=1e-15)
    t.close(f"T3.03 and that is what the contour measured ({amb:.15f})",
            val, amb, tol=1e-8)
    t.close(f"T3.04 and what the residue measured ({amb_r:.15f})",
            val, amb_r, tol=1e-14)
    # The point of the whole exercise, stated as the two groups disagreeing
    # about the SAME expression: the powers-and-logs group refuses exp(-1/h)
    # because it genuinely has nowhere to put it, and the transseries holds it.
    arg = R(-1) / h                       # -1/h, infinite
    try:
        v = cl.exp(arg)
        t.true("T3.05 the value group refuses exp(-1/h)", False,
               f"cl.exp returned {v!r} instead of refusing")
    except Exception as ex:
        t.true(f"T3.05 the value group REFUSES exp(-1/h) "
               f"({type(ex).__name__}), having no dimension below every power",
               True, str(ex)[:60])
    t.true("T3.06 and ts_exp(-1/h) places it: sector 1, == flat(1)",
           ts_exp(arg) == flat(1), f"{ts_exp(arg)}")
    t.true("T3.07 so the refusal was not a missing feature -- the two agree "
           "on sector 0 and differ only where the group had no room",
           ts_exp(R(-1) * h).sectors.keys() == {0},
           f"{ts_exp(R(-1) * h)}")


# =============================================================================
def t4_the_bridge(t):
    head("T4  the bridge -- a sector's series is DIVERGENT, so it is resummed")

    # Euler-Stieltjes on the branch where the Borel ray is blocked: at eps < 0
    # the series in h = |eps| is sum n! h^n, every coefficient positive, so the
    # Borel singularity sits at +1 -- ON the Laplace path.
    S0 = Composite({-n: float(math.factorial(n)) for n in range(24)})
    S1 = R(math.pi) / h                      # (pi/h) exp(-1/h)
    T = Transseries({0: S0, 1: S1})

    p, co = sector_coefficients(S1)
    t.true(f"T4.00 a sector's INFINITE prefactor factors out: pi*h^-1 reads as "
           f"h^-{p:g} * [{co[0]:.6f}]", p == 1.0 and abs(co[0] - math.pi) < 1e-15,
           f"p={p}, coeffs={co}")

    # What the bridge replaces.
    naive = T.evaluate(1.0, borel=False)
    t.true(f"T4.01 summed as written the sector is nonsense: {naive:.4g} "
           f"(the answer is 1.85)", naive > 1e20, f"{naive:.4e}")

    # What it gives instead.  e^-1 Ei(1) is the principal value of the blocked
    # integral and is known in closed form.
    PV = math.exp(-1.0) * 1.8951178163559368
    v, amb = resum_sector(S0, 1.0, 8, 8)
    t.close("T4.02 sector 0 resummed = e^-1 Ei(1)", v, PV, tol=1e-8)

    # THE CONSISTENCY THAT MATTERS.  Sector 0's resummation reports an
    # ambiguity, because its ray is blocked.  Sector 1 says how big the flat
    # term is.  These are computed from DIFFERENT halves of the transseries and
    # they must be the same number -- that is what makes the pair a complete
    # object rather than a series plus a decoration.
    for i, hv in enumerate((1.0, 0.5, 0.25), 1):
        _, a = resum_sector(S0, hv, 8, 8)
        s1 = math.exp(-1 / hv) * _eval_sector1(S1, hv)
        t.close(f"T4.1{i} h={hv}: sector 0's ambiguity {a:.15f} == sector 1's "
                f"value {s1:.15f}", a, s1, tol=1e-8)

    val, amb = T.evaluate_with_ambiguity(1.0, L=8, M=8)
    t.close("T4.20 evaluate() routes the divergent sector through Borel-Pade",
            val, PV + math.exp(-1.0) * math.pi, tol=1e-8)
    t.true(f"T4.21 and reports the ray was blocked (ambiguity {amb:.6f} > 0)",
           amb > 1e-3, f"{amb}")

    # UNITS OF THE ACTION.  The sector index counts exp(-1/h), so a series
    # whose action is A must be read in h = w/A.  That is a change of variable
    # and nothing else, which is checkable: it must not move the answer.
    a_k, A = _p1(), 0.8 * math.sqrt(2 * math.sqrt(6))
    for tt in (4.0, 6.0):
        w = tt ** -1.25
        raw = Composite({-2 * k: a_k[k] for k in range(len(a_k))})
        scaled = Composite({-2 * k: a_k[k] * A ** (2 * k) for k in range(len(a_k))})
        v_raw, amb_raw = resum_sector(raw, w, 8, 8)
        v_sc, amb_sc = resum_sector(scaled, w / A, 8, 8)
        t.close(f"T4.3{int(tt)} P1 at t={tt}: h=w/A gives the same value "
                f"({v_sc:.15f})", v_sc, v_raw, tol=1e-14)
        t.close(f"T4.4{int(tt)} and the same ambiguity ({amb_sc:.3e})",
                amb_sc, amb_raw, tol=1e-14 * max(1.0, amb_raw))
    from composite.resummation import borel, pade, poles
    bs = [a_k[k // 2] * A ** k if k % 2 == 0 else 0.0
          for k in range(2 * len(a_k))]
    z = abs(poles(pade(borel(bs), 8, 8)[1])[0])
    t.close(f"T4.50 and in those units the Borel singularity sits at |z0| = "
            f"{z:.4f}, which is [8/8]'s estimate of A over A "
            f"({1.8787124553114658 / A:.4f})", z, 1.8787124553114658 / A,
            tol=1e-3)


def _eval_sector1(c, hv):
    return sum(float(v) * hv ** (-float(d)) for d, v in c.coeffs_dict().items())


def _p1(N=42):
    a0 = 1.0 / math.sqrt(6.0)
    a = [a0]
    for k in range(N):
        Y = Composite({-i: v for i, v in enumerate(a)})
        conv = (Y * Y).coeffs_dict().get(-(k + 1), 0.0)
        a.append((a[k] * (25 * k * k - 1) / 4.0 - 6.0 * conv) / (12.0 * a0))
    return a


# =============================================================================
def t5_separability(t):
    head("T5  h=1 is the one place the flat term is not flat")
    # exp(-1/h) at h = 1 is 0.368 -- the same order as the answer, so float
    # addition loses nothing and the comparison proves less than it looks.
    # The real question is what happens once the term is exponentially small.
    S0 = Composite({-n: (-1.0) ** n * math.factorial(n) for n in range(24)})
    T = Transseries({0: S0, 1: R(1)})          # S + exp(-1/h)
    base = Transseries({0: S0})

    for hv in (0.1, 0.05, 0.01):
        flat_true = math.exp(-1 / hv)
        fp = T.flat_part(hv)
        t.close(f"T5.0{int(100 * hv):02d} flat_part({hv}) = {fp:.6e} "
                f"= exp(-1/{hv})", fp, flat_true, tol=1e-15 * flat_true)

    # Visible, but already decaying: recovering the flat term from the SUM
    # means subtracting two numbers of order 1, so what survives is bounded by
    # an ulp of those, not by the flat term's own size.  13 digits at h=0.1,
    # 8 at h=0.05, none at all at h=0.01.
    for hv in (0.1, 0.05):
        d = T.evaluate(hv) - base.evaluate(hv)
        want = math.exp(-1 / hv)
        floor = abs(base.evaluate(hv)) * 2.220446049250313e-16
        t.true(f"T5.1{int(100 * hv):02d} at h={hv} the sum still shows it: "
               f"{d:.9e} vs exp(-1/{hv}) = {want:.9e}, recovered to "
               f"{abs(d - want) / want:.1e} relative (cancellation floor "
               f"{floor / want:.1e})",
               abs(d - want) <= 2 * floor, f"|d-want| {abs(d - want):.2e}, "
               f"floor {floor:.2e}")

    # Gone: the sum does not move at all, bit for bit.
    hv = 0.01
    tot, bas = T.evaluate(hv), base.evaluate(hv)
    t.true(f"T5.200 at h=0.01 the sum is BIT-IDENTICAL with and without the "
           f"flat term ({tot!r})", tot == bas, f"{tot!r} vs {bas!r}")
    fp = T.flat_part(hv)
    t.true(f"T5.201 exp(-100) = {fp:.6e} is 28 orders below the grade-0 part's "
           f"last bit ({abs(bas) * 2.2e-16:.2e}), so the sum cannot carry it",
           0 < fp < abs(bas) * 2.2e-16 * 1e-20, f"flat {fp:.3e}, ulp "
           f"{abs(bas) * 2.2e-16:.3e}")
    t.true(f"T5.202 but the DECOMPOSITION still reads it: parts = "
           f"{{0: {T.evaluate_parts(hv)[0]:.12f}, 1: {T.evaluate_parts(hv)[1]:.3e}}}",
           T.evaluate_parts(hv)[1] == fp and fp != 0.0, f"{T.evaluate_parts(hv)}")
    parts = T.evaluate_parts(hv)
    t.true(f"T5.203 evaluate() is exactly the sum of the parts, and at h=0.01 "
           f"that sum has already discarded sector 1",
           sum(parts.values()) == tot and parts[0] == tot,
           f"sum {sum(parts.values())!r}, parts[0] {parts[0]!r}, tot {tot!r}")


# =============================================================================
def t6_closure(t):
    head("T6  derivation CLOSES on the new level -- an H-field, not a box")
    # ln sends a sector back to the power axis; it is ts_exp inverted.
    t.true("T6.01 ln(exp(-1/h)) == -1/h",
           ts_ln(flat(1)) == Transseries.lift(R(-1) / h), f"{ts_ln(flat(1))}")
    t.true("T6.02 ln(exp(1/h)) == 1/h",
           ts_ln(ts_exp(R(1) / h)) == Transseries.lift(R(1) / h),
           f"{ts_ln(ts_exp(R(1) / h))}")
    t.true("T6.03 and the round trip exp(ln(x)) == x",
           ts_exp(ts_ln(flat(2)).sectors[0]) == flat(2),
           f"{ts_exp(ts_ln(flat(2)).sectors[0])}")

    # THE STRUCTURAL TEST.  d/dh must map transseries to transseries, staying
    # on the exponential level rather than leaking off it.
    t.true("T6.10 d/dh exp(-1/h) == exp(-1/h)/h**2",
           ts_d(flat(1)) == flat(1) * Transseries.lift(R(1) / (h * h)),
           f"{ts_d(flat(1))}")
    t.true("T6.11 d/dh exp(-2/h) == 2*exp(-2/h)/h**2",
           ts_d(flat(2)) == flat(2) * Transseries.lift(R(2) / (h * h)),
           f"{ts_d(flat(2))}")
    t.true("T6.12 the sector is UNCHANGED by differentiation (closure)",
           set(ts_d(flat(1)).sectors) == {1}
           and set(ts_d(flat(3)).sectors) == {3},
           f"{set(ts_d(flat(3)).sectors)}")
    t.true("T6.13 d/dh h**3 == 3h**2, the power axis untouched",
           ts_d(Transseries.lift(h ** 3)) == Transseries.lift(R(3) * h ** 2),
           f"{ts_d(Transseries.lift(h ** 3))}")
    # d/dh [h exp(-1/h)] = (1 + 1/h) exp(-1/h)
    f = flat(1) * Transseries.lift(h)
    want = flat(1) * Transseries.lift(R(1) + R(1) / h)
    t.true("T6.14 d/dh [h*exp(-1/h)] == (1 + 1/h)exp(-1/h)", ts_d(f) == want,
           f"{ts_d(f)}")

    # Leibniz across the level: if the rule holds for a product of a sector
    # and a power series, derivation is a genuine derivation on the whole
    # object and not two rules glued together.
    g = Transseries.lift(h ** 2 + R(3) * h)
    t.true("T6.20 Leibniz holds across sectors: d(fg) == f'g + fg'",
           ts_d(f * g) == ts_d(f) * g + f * ts_d(g), f"{ts_d(f * g)}")

    t.true("T6.30 ln of a multi-sector transseries refuses (needs the "
           "dominant term factored out)",
           _refuses(ts_ln, flat(1) + Transseries.lift(R(1))), "should refuse")


# =============================================================================
# Ai'(z)/Ai(z) from mpmath 1.3.0 at 50 decimal digits -- an independent oracle
# that shares no code with anything here.  Pinned so the suite does not need
# the dependency; checked against live mpmath as well when it is importable.
AIRY_REF = {2.0: -1.5201633881848287, 3.0: -1.807422974977254,
            5.0: -2.283586660845399, 8.0: -2.858866034197276}

STIELTJES = [(-1) ** n * math.factorial(n) for n in range(24)]


def _stirling_series(nt=24):
    """Binet's series, Bernoulli numbers from one composite division."""
    q = ZERO / (cl.exp(ZERO, terms=nt) - R(1))
    B = {-int(k): v * math.factorial(-int(k))
         for k, v in q.coeffs_dict().items() if 0 <= -int(k) <= nt}
    a = [B[2 * n] / (2 * n * (2 * n - 1)) for n in range(1, nt // 2 + 1)]
    c = [0.0] * (2 * len(a))
    for n, v in enumerate(a, 1):
        c[2 * n - 1] = v
    return c


def _p1_even():
    """Painleve I's coefficients laid on even indices (a lacunary series)."""
    a = _p1()
    c = [0.0] * (2 * len(a))
    for k, v in enumerate(a):
        if 2 * k < len(c):
            c[2 * k] = v
    return c


def _airy_riccati(N=30):
    """Airy's asymptotic series, derived from the ODE rather than typed in.

    y'' = z y.  Put y = exp(int S dz), giving the RICCATI equation S' + S^2 = z,
    and expand S = sum c_k z^((1-3k)/2):

        S'  contributes c_k (1-3k)/2 * z^((-1-3k)/2)
        S^2 contributes (sum_{i+j=m} c_i c_j) z^((2-3m)/2)

    The powers meet at m = k+1, so

        c_k (1-3k)/2 + 2 c_0 c_{k+1} + sum_{i=1..k} c_i c_{k+1-i} = 0

    and that sum over i is a convolution -- which is composite multiplication.
    c_0 = -1 selects the decaying branch, so S is Ai'/Ai.
    """
    c = [-1.0]
    for k in range(N):
        S = Composite({-i: v for i, v in enumerate(c)})
        conv = (S * S).coeffs_dict().get(-(k + 1), 0.0)
        c.append(-(c[k] * (1 - 3 * k) / 2.0 + conv) / (2 * c[0]))
    return c


def t7_from_series(t):
    head("T7  from_series -- the PROBLEM builds the object, not a person")

    # (a) A resolved pole: the whole transseries comes out, flat term included.
    T, info = from_series(STIELTJES, 8, 8)
    t.true(f"T7.01 Euler-Stieltjes classified {info['kind']}, action "
           f"{info['action']:.12f}", info["kind"] == SingularityKind.POLE
           and abs(info["action"] - 1.0) < 1e-12, f"{info}")
    t.true(f"T7.02 both sectors built from the coefficient list alone: "
           f"{sorted(T.sectors)}", sorted(T.sectors) == [0, 1],
           f"{sorted(T.sectors)}")
    t.close(f"T7.03 sector 1 is (pi/h)exp(-1/h) -- pi DISCOVERED, not typed: "
            f"{T.sectors[1]}", info["stokes"], math.pi, tol=1e-12)
    for hv, want in ((1.0, math.pi / math.e), (0.5, math.pi / 0.5 * math.exp(-2))):
        t.close(f"T7.04 flat_part({hv}) = {T.flat_part(hv):.15f}",
                T.flat_part(hv), want, tol=1e-12 * want)

    # (b) AIRY.  The series is derived from y'' = z y by composite convolution;
    # the machinery is then asked to find the SECOND solution, which it has
    # never been told about.  Bi ~ e^{+zeta} and Ai ~ e^{-zeta} with
    # zeta = (2/3)z^(3/2), so they differ by e^{-2*zeta} = e^{-(4/3)/w} in the
    # expansion variable w = z^(-3/2).  The action must come out 4/3.
    c = _airy_riccati()
    t.close("T7.10 the derived series starts -1, -1/4, 5/32 (Riccati, by "
            "composite convolution)", c[1], -0.25, tol=1e-15)
    t.close("T7.11 and its third coefficient", c[2], 5.0 / 32.0, tol=1e-15)
    T2, i2 = from_series(c, 10, 10)
    t.true(f"T7.12 Airy's action found WITHOUT being told Bi exists: "
           f"{i2['action']:.8f} vs 4/3 = {4/3:.8f} (err "
           f"{abs(i2['action'] - 4/3):.1e}) -- that is exp(-2*zeta), the gap "
           f"to the other solution", abs(i2["action"] - 4 / 3) < 5e-3,
           f"{i2['action']}")
    t.true(f"T7.13 classified {i2['kind']} and the Stokes coefficient is "
           f"REFUSED ({i2['stokes']}) -- a branch point has no residue",
           i2["kind"] == SingularityKind.CUT and i2["stokes"] is None, f"{i2}")
    # beta -> 0 from the coefficients alone: a LOGARITHMIC branch point
    b = borel(c)
    A = 4.0 / 3.0
    sl = [(math.log(abs(b[k]) * A ** k) - math.log(abs(b[k - 4]) * A ** (k - 4)))
          / (math.log(k) - math.log(k - 4)) for k in (16, 20, 24)]
    t.true("T7.14 beta from the coefficients = "
           + ", ".join(f"{x + 1:+.4f}" for x in sl)
           + " -> 0: a LOGARITHMIC branch point", all(0 < x + 1 < 0.05 for x in sl),
           f"{sl}")

    # (c) the value, against mpmath -- no shared code with any of this
    for z, ref in sorted(AIRY_REF.items()):
        w = z ** -1.5
        got = T2.evaluate(w / i2["action"], L=8, M=8) * math.sqrt(z)
        terms = [abs(c[k] * z ** ((1 - 3 * k) / 2.0)) for k in range(len(c))]
        bn = min(range(1, len(terms)), key=lambda j: terms[j])
        tr = sum(c[k] * z ** ((1 - 3 * k) / 2.0) for k in range(bn + 1))
        t.true(f"T7.2{int(z)} Ai'/Ai at z={z}: {got:.15f} vs mpmath "
               f"{ref:.15f}, err {abs(got - ref):.1e}  (optimal truncation "
               f"{abs(tr - ref):.1e} at k*={bn})",
               abs(got - ref) < 1e-10, f"{got} vs {ref}")
    try:
        import mpmath as mp
        mp.mp.dps = 50
        live = float(mp.airyai(3, 1) / mp.airyai(3))
        t.close("T7.29 and the pinned reference matches live mpmath", live,
                AIRY_REF[3.0], tol=1e-15)
    except ImportError:
        t.true("T7.29 live mpmath unavailable; pinned references used", True, "")

    # (d) the action estimator, across every geometry we have
    cut = [math.factorial(n) * ((-1) ** n * math.factorial(2 * n)
                                / (4.0 ** n * math.factorial(n) ** 2))
           for n in range(26)]
    stir = _stirling_series()
    p1c = _p1_even()
    cases = [("Euler-Stieltjes", STIELTJES, 8, 1.0, 1e-12),
             ("sqrt branch cut", cut, 8, 1.0, 1e-3),
             ("Airy (Riccati)", c, 10, 4 / 3, 5e-3),
             ("Painleve I", p1c, 8, 0.8 * math.sqrt(2 * math.sqrt(6)), 1e-6),
             ("Stirling/Binet", stir, 8, 2 * math.pi, 1e-2)]
    for lbl, cs, L, truth, tol in cases:
        _, i = from_series(cs, L, L)
        t.true(f"T7.3 {lbl:<16} action {i['action']:.8f} vs {truth:.8f} "
               f"(err {abs(i['action'] - truth):.1e}; Pade alone gave "
               f"{i['action_pade']:.6f})", abs(i["action"] - truth) < tol,
               f"{i['action']} vs {truth}")
    # and the estimator must survive a LACUNARY series
    for lbl, cs in (("even (Painleve I)", p1c), ("odd (Stirling)", stir)):
        g = action_from_growth(cs)
        t.true(f"T7.4 stride detected on an {lbl} series -> {g!r}; reading "
               f"consecutive pairs would hit a zero every step",
               g is not None, f"action_from_growth returned {g!r}")


# =============================================================================
def t8_zero_sectors(t):
    head("T8  R1 on the OPERAND, R2 on the TERM -- a sector is a term")

    def d(c):
        return dict(sorted(c.coeffs_dict().items()))

    # R1 ACTS ON THE OPERAND, and the operand is the whole transseries.  Both
    # layers must therefore answer the same way, or `0` means one thing in a
    # Composite and another one call up.
    for lbl, comp, ts in (
            ("lift(0) + 5", Composite(0) + R(5),
             (Transseries.lift(0) + Transseries.lift(R(5))).sectors[0]),
            ("lift(0) * 5", Composite(0) * R(5),
             (Transseries.lift(0) * Transseries.lift(R(5))).sectors.get(0))):
        t.true(f"T8.0 {lbl}: Composite {d(comp)} == Transseries {d(ts)}",
               d(comp) == d(ts), f"{d(comp)} vs {d(ts)}")
    F = flat(1) - flat(1)
    c = Composite({0: 1.0})
    t.true(f"T8.05 a wholly zero transseries IS an operand and converts: "
           f"(flat-flat)+flat = {d((F + flat(1)).sectors[1])}, matching "
           f"(c-c)+R(1) = {d((c - c) + R(1))}",
           d((F + flat(1)).sectors[1]) == d((c - c) + R(1)),
           f"{d((F + flat(1)).sectors[1])}")

    # R2: A ZERO SECTOR SITTING AMONG NON-ZERO ONES IS A TERM, NOT AN OPERAND.
    # Per-sector arithmetic through Composite.__add__ made it an operand, and
    # R1 converted it -- an infinitesimal manufactured inside a number nobody
    # used as a zero.  Committed, and no suite caught it.
    z = Composite({0: 5.0}) - Composite({0: 5.0})
    A = Transseries({0: R(1), 1: z})
    B = Transseries({1: R(5)})
    two = Transseries({0: R(2)})
    for lbl, got, want in (("A + B", d((A + B).sectors[1]), {0: 5.0}),
                           ("A - B", d((A - B).sectors[1]), {0: -5.0}),
                           ("A * 2", d((A * two).sectors[1]), {0: 0.0})):
        t.true(f"T8.1 ({lbl}).sectors[1] = {got}, R2 says {want} "
               f"(was leaking |1|_-1)", got == want, f"{got} vs {want}")
    t.true("T8.20 and sector 0 is untouched by any of it",
           d((A + B).sectors[0]) == {0: 1.0}, f"{d((A + B).sectors[0])}")


def run_all():
    t = Suite()
    for fn in (t1_ordering, t2_sector_arithmetic, t3_against_the_oracle,
               t4_the_bridge, t5_separability, t6_closure,
               t7_from_series, t8_zero_sectors):
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
