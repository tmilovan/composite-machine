# composite/resummation.py
# Composite Machine — Borel-Pade resummation of divergent asymptotic series
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Get a NUMBER out of a series that diverges for every nonzero argument.

The Euler-Stieltjes integral is the case this exists for:

    S(eps) = int_0^inf exp(-t) / (1 + eps*t) dt  ~  sum (-1)^n n! eps^n

The coefficients are n!, so the terms grow without bound however small eps is.
Partial sums at eps = 1 go 1, 0, 2, -4, 20, -100, 620 -- summing coefficients
does not approach S(1) = 0.5963473623231941, it runs away from it.  eps = 0 is
an IRREGULAR singular point: unlike a branch point, the expansion does not
determine the function.  S and S + C*exp(-1/eps) share every coefficient to
all orders, because the expansion of exp(-1/eps) is identically zero.

So no number of coefficients answers the question, and the route is Borel:

    Borel transform    b_n = c_n / n!        divides the factorial out
    Pade approximant   B(z) ~ P(z)/Q(z)      rational, no recognition needed
    Laplace transform  S(eps) = int_0^inf exp(-u) B(eps*u) du

For this series b_n = (-1)^n, whose Pade is 1/(1+z) and whose Laplace integral
is the original integral -- the factorial has been absorbed by the transform.
Pade is used rather than summing b_n because it needs no closed form: it works
on series whose Borel transform has none, which is the general case.

WHY THE PADE IS A COMPOSITE COMPUTATION, not a linear solve.  Pade is usually
presented as a Toeplitz system, which would be float linear algebra wearing a
composite's clothes.  It is equivalently the extended Euclidean algorithm on
z^(L+M+1) and B(z), stopped when the remainder falls below degree L -- and a
composite with integer grades IS a polynomial, with convolve as polynomial
multiplication and deconvolve as long division.  So the whole approximant is
built from the library's own two core operations.

ASCENDING ENCODING.  A polynomial is stored with the coefficient of z^k at
grade +k, so the HIGHEST degree dominates and division proceeds leading-term
first, which is what long division means.  The descending encoding (z^k at
grade -k) makes division proceed from the constant term and produce an
ascending power series that never terminates -- correct for an asymptotic
expansion, useless for a remainder.

DIVISION HERE IS SERIES DIVISION.  Even ascending, `/` does not stop at a
remainder: (1+z^2)/(1-z) continues into z^-1, z^-2, ...  The polynomial
quotient is its non-negative part, and the remainder has to be formed as
A - Q*B.  Measured on four cases, every one gives deg(R) < deg(B).
"""
import cmath
import math

from .composite_lib import (Composite, integrate, exp as _cexp, exp, sin, cos)

_TOL = 1e-13


def _clean(c, scale=1.0):
    """Drop terms whose coefficient is numerically zero.

    `scale` sets what "zero" is measured against.  The default 1.0 is an
    ABSOLUTE tolerance, which is right only while the coefficients are O(1).
    Where they are not -- see _balance -- the caller must say so.

    The library keeps EXPRESSED zeros on purpose -- a dimension exists because
    the computation built it.  That is right for the arithmetic and wrong for
    a degree test, which must not see a leading 0*z^5 as degree 5.
    """
    return Composite({k: v for k, v in c.coeffs_dict().items()
                      if abs(v) > _TOL * scale})


def degree(c):
    """Highest grade with a nonzero coefficient, or None for the zero polynomial."""
    d = _clean(c).coeffs_dict()
    return max(d) if d else None


def poly(coeffs):
    """Polynomial from [c0, c1, ...] -- c_k on z^k, ascending encoding."""
    return Composite({k: float(v) for k, v in enumerate(coeffs)})


def polydiv(a, b):
    """Quotient and remainder, deg(R) < deg(B), from the library's division."""
    q = Composite({k: v for k, v in (a / b).coeffs_dict().items()
                   if k >= 0 and abs(v) > _TOL})
    return q, _clean(a - q * b)


def _balance(b):
    """Scale rho putting b_k * rho^k on one footing, by Cauchy-Hadamard.

    The radius of convergence of sum b_k z^k is 1/limsup|b_k|^(1/k), so scaling
    the variable by that radius makes every coefficient O(1).

    This is not cosmetic.  The Euclidean algorithm below decides the DEGREE of
    each remainder by comparing coefficients against one absolute tolerance,
    and that only means something while they share a magnitude.  For the
    Stirling/Binet series they do not: b_1 = 8.3e-02 and b_23 = -6.1e-21,
    because the Borel transform there has radius 2*pi and the coefficients
    decay like (2*pi)^-k.  Everything from b_15 = -2.3e-14 down was deleted as
    "numerically zero" although it was real data, poly(b[:15]) came back degree
    13 instead of 15, and the algorithm then cycled 13 -> 11 -> 13 -> 11
    FOREVER: [7/7] did not fail, it hung.

    Balanced, the same coefficients span 4.1e-02 .. 5.5e-01 and the orders run
    through: mu(1) improves 2.45e-07 at [6/6] to 6.08e-10 at [11/11].

    Returns 1.0 (no scaling) when the coefficients are already O(1), which is
    the Euler-Stieltjes case -- b_k = (-1)^k there, so nothing moves.
    """
    r = max((abs(v) ** (1.0 / k)
             for k, v in enumerate(b) if k >= 1 and v != 0.0), default=0.0)
    if not (r > 0.0) or math.isinf(r) or math.isnan(r):
        return 1.0
    return 1.0 / r


def borel(c):
    """b_n = c_n / n!  -- the transform that absorbs the factorial growth."""
    return [v / math.factorial(n) for n, v in enumerate(c)]


def pade(b, L, M):
    """Pade [L/M] of sum b_n z^n, by the extended Euclidean algorithm.

    Runs gcd steps on z^(L+M+1) and the truncated series, carrying the
    cofactor of the series; when the remainder drops to degree <= L the
    remainder IS P and the cofactor IS Q.  Normalised to Q(0) = 1 so the
    approximant is comparable across orders.
    """
    n = L + M + 1
    if len(b) < n:
        raise ValueError(f"Pade [{L}/{M}] needs {n} coefficients, got {len(b)}")

    # Run the whole algorithm in a BALANCED variable w = z/rho, where the
    # absolute tolerance the degree test uses is meaningful, and unscale the
    # result at the end.  See _balance for what an unbalanced run did.
    rho = _balance(b[:n])
    bs = [v * rho ** k for k, v in enumerate(b[:n])]

    r_prev = Composite({n: 1.0})          # w^(L+M+1)
    r_cur = poly(bs)
    s_prev = Composite({})                # cofactor of the series: 0
    s_cur = Composite({0: 1.0})           # then 1

    for _ in range(n + 2):                # backstop; the degree guard is exact
        dr = degree(r_cur)
        if dr is None or dr <= L:
            break
        q, r_next = polydiv(r_prev, r_cur)
        dn = degree(r_next)
        if dn is not None and dn >= dr:
            # The remainder must lose a degree every step or the algorithm
            # does not terminate.  When it does not, the degrees are being
            # read off coefficients that are numerically indistinguishable
            # from zero -- REFUSE, rather than cycle silently.
            raise ArithmeticError(
                f"Pade [{L}/{M}]: the Euclidean remainder stopped shrinking "
                f"(degree {dr} -> {dn}).  The coefficients cannot support "
                f"this order -- the degree test is reading numerical noise. "
                f"Try a lower order, or supply coefficients to more digits.")
        r_prev, r_cur = r_cur, r_next
        s_prev, s_cur = s_cur, _clean(s_prev - q * s_cur)
    else:
        raise ArithmeticError(
            f"Pade [{L}/{M}]: Euclidean algorithm did not terminate in "
            f"{n + 2} steps.")

    P, Q = _clean(r_cur), _clean(s_cur)
    q0 = Q.coeffs_dict().get(0, 0.0)
    if abs(q0) < _TOL:
        raise ValueError("Pade denominator vanishes at 0; try another (L, M)")
    # Back to z: P~(w) with w = z/rho is sum p_k z^k / rho^k.
    return (Composite({k: v / (q0 * rho ** k)
                       for k, v in P.coeffs_dict().items()}),
            Composite({k: v / (q0 * rho ** k)
                       for k, v in Q.coeffs_dict().items()}))


def _evaluate(p, z):
    """Evaluate a polynomial-composite at z, which may itself be a composite.

    Horner would need dense consecutive degrees; a Pade denominator is sparse
    in general, so powers are built incrementally by ascending degree and only
    the degrees present are touched.
    """
    d = _clean(p).coeffs_dict()
    if not d:
        return z * 0.0
    out = None
    power = None
    prev_k = 0
    for k in sorted(d):
        k = int(k)
        if power is None:
            power = z ** k if k else 1.0
        else:
            for _ in range(k - prev_k):
                power = power * z
        prev_k = k
        term = power * d[k] if k else d[k]
        out = term if out is None else out + term
    return out


def poles(Q):
    """Roots of the Pade denominator -- the Borel-plane singularities.

    For the Euler-Stieltjes series a pole converges to u = -1, and that
    distance from the origin IS the exponent of the flat term exp(-1/eps).
    So the approximant locates the object the value group cannot represent.
    """
    import numpy as _np
    raw = Q.coeffs_dict()
    if not raw:
        return []
    # Unscaling pade's balanced result spreads Q's coefficients over rho^M, so
    # "is the leading coefficient zero" is a RELATIVE question.  Asked
    # absolutely it silently lowers the degree and loses poles.
    d = _clean(Q, scale=max(abs(v) for v in raw.values())).coeffs_dict()
    if not d:
        return []
    deg = max(d)
    coef = [d.get(k, 0.0) for k in range(deg, -1, -1)]
    # Modulus alone does not order a symmetric set: an even series puts roots
    # at BOTH +A and -A, a real series puts them in conjugate pairs, and which
    # came back first was down to numpy.  z0 then flipped sign from order to
    # order (+2.146, -1.952, -1.879, +1.842) and every comparison across
    # orders was measuring the flip, not the singularity.
    return sorted(_np.roots(coef), key=lambda r: (abs(r), -r.real, -r.imag))


def _on_ray(rs, eps):
    """The roots that land on the Laplace path u > 0, nearest first.

    u0 = z/eps must be real and positive.  Asking this of the NEAREST root
    only was wrong for any symmetric set: Painleve I has roots at both +A and
    -A, and whichever of the two came back first decided whether the ambiguity
    was reported or silently called zero.
    """
    out = []
    for r in rs:
        if abs(getattr(r, "imag", 0.0)) > 1e-9 * max(1.0, abs(r)):
            continue
        u0 = float(getattr(r, "real", r)) / eps
        if u0 > 0:
            out.append(r)
    return sorted(out, key=abs)


def resum(c, eps, L=None, M=None, upper=float('inf')):
    """Borel-Pade sum of sum c_n eps^n.

    S(eps) = int_0^inf exp(-u) * P(eps*u)/Q(eps*u) du, with P/Q the Pade
    approximant of the Borel transform.  The integral runs on the library's
    quadrature.

    THE UPPER LIMIT IS INFINITY, not a large cutoff.  The tail past u = 40 is
    about 1e-19 here, so truncating looks harmless -- and costs eight digits:
    a finite panel run returned 0.5963473621 against 0.5963473623231941, an
    error of 2e-10 that tightening tol from 1e-10 to 1e-16 did not move.  The
    improper path returns all 16 digits exactly, because it handles the decay
    analytically instead of panelling it.
    """
    b = borel(c)
    if L is None or M is None:
        k = (len(b) - 1) // 2
        L, M = k, k
    P, Q = pade(b, L, M)

    blocked = _on_ray(poles(Q), eps)
    if blocked:
        u0 = blocked[0].real / eps
        raise NotImplementedError(
            f"the Laplace ray is BLOCKED: a Borel singularity sits at "
            f"{blocked[0].real:+.9f}, which for eps={eps!r} lands at u0="
            f"{u0:.6f} ON the integration path u > 0.  The sum is not defined "
            f"in this direction -- it is ambiguous by exactly the flat term, "
            f"and flat_term() will size it.  resum_lateral() / resum_median() "
            f"will sum it by swinging the contour to one side.  Quadrature "
            f"straight through the "
            f"pole returned 1.4e+43 at one order and the correct 0.40818 at "
            f"the next, with nothing to tell them apart.")

    def integrand(u):
        z = eps * u
        return _cexp(-u) * (_evaluate(P, z) / _evaluate(Q, z))

    return integrate(integrand, 0.0, upper), (P, Q)


def dpoly(p):
    """d/dz of a polynomial-composite: coefficient k*p_k moves to grade k-1."""
    return _clean(Composite({k - 1: k * v
                             for k, v in _clean(p).coeffs_dict().items()
                             if k >= 1}))


class SingularityKind:
    """What the Pade denominator's leading root actually represents."""
    POLE = "pole"              # a pole, resolved: the residue is a real number
    CUT = "cut"                # a branch point: there IS no residue
    UNRESOLVED = "unresolved"  # converging toward a pole, not there yet


# How much the residue may still move between consecutive orders and still
# count as settled.  Measured: a resolved pole moves by at most 5.1e-05 (the
# string case at its noise floor), a branch point by 11%-21%.  Four orders of
# magnitude of daylight; the threshold sits in the middle of it.
_RES_SETTLED = 1e-3

# How fast the leading root must be closing in to count as converging.  A pole
# being resolved: 0.138, 0.120, 0.083, 0.058, 0.043, 0.033 -- falling.  A
# branch point: 0.195, 0.416, 0.503, 0.635, 0.699, 0.745 -- rising toward 1,
# because Pade can only ever lay another link of the chain.  The two come
# within a factor of 1.4 of each other at low order, which is why this only
# ever chooses the LABEL: both CUT and UNRESOLVED refuse the residue.
_RATE_CONVERGING = 0.15


def _leading(b, L, M, near=None):
    """(z0, residue) of the Pade denominator's leading root at order [L/M].

    `near` pairs the root with one from another order.  Sorting by modulus
    alone cannot do that: a conjugate pair has ONE modulus, so which of the
    two comes back first is arbitrary, and comparing orders by index reads a
    mirror image as a drift of 2|z0|.
    """
    P, Q = pade(b, L, M)
    rs = poles(Q)
    if not rs:
        return None, None
    z0 = rs[0] if near is None else min(rs, key=lambda r: abs(r - near))
    dq = _evaluate(dpoly(Q), z0)
    if dq == 0:
        return z0, None
    return z0, _evaluate(P, z0) / dq


def classify_singularity(c, L=None, M=None):
    """Tell a resolved pole from a branch point, and say which.

    Pade reproduces a RATIONAL function exactly at the first order that fits,
    so an isolated pole shows up as a denominator whose root does not move at
    all.  Nothing else is ever reproduced exactly: for anything with infinitely
    many singularities Pade lays a CHAIN of poles and extends it order by
    order.  A chain looks the same whether it is covering a branch cut or a
    string of genuine poles, so the chain itself decides nothing.

    What decides it is the RATE.  Measured across five series, order by order:

      case                       drift/|z0|          rate d(L)/d(L-1)   residue
      -------------------------------------------------------------------------
      1/(1+u)         pole       0                   --                 1.000000
      1/(1+(u/2pi)^2) pole pair  0                   --                 -3.14159i
      1/(e^u-1)-...   poles at   1.95e-01..2.25e-08  .138 .083 .043 .033  ->1.000000
                      2*pi*i*k                        FALLING
      Stirling/Binet  LOGS at    6.04e-01..4.65e-03  .195 .503 .699 .745  1.13->0.060
                      2*pi*i*k                        RISING
      1/sqrt(1+u)     branch pt  2.06e-01..1.29e-03  .235 .496 .699 .723  0.44->0.084
                                                      RISING

    A pole's residue CONVERGES -- the chain is closing on something that is
    there.  A branch point's slides toward zero, because each link is carrying
    a share of a continuous density and the share shrinks as links are added.
    That is the signal, and the rate at which the root settles confirms it.

    Returns (kind, z0, residue_or_None, detail).  Only POLE gets a residue.
    Reporting one for a branch point returned 0.118657 for an object that has
    none, with nothing on it to say so.
    """
    b = borel(c)
    if L is None or M is None:
        k = (len(b) - 1) // 2
        L, M = k, k
    P, Q = pade(b, L, M)
    rs = poles(Q)
    if not rs:
        return SingularityKind.POLE, None, None, {"roots": 0,
                                                  "why": "no singularity found"}

    z0 = rs[0]
    dq = _evaluate(dpoly(Q), z0)
    res = _evaluate(P, z0) / dq if dq != 0 else None
    detail = {"roots": len(rs), "z0": z0, "residue": res}

    # A denominator that stopped short of its allowed degree means the
    # Euclidean algorithm terminated early: the transform IS rational and this
    # is the whole of it.
    if len(rs) == 1:
        detail["why"] = "single root: rational transform, reproduced exactly"
        return SingularityKind.POLE, z0, res, detail

    # Distance to the nearest root of a DIFFERENT modulus.  rs[1] is usually
    # z0's own conjugate, and measuring against a mirror image gave a gap
    # ratio of exactly 2.000 every time -- a dead signal.
    tol = 1e-9 * max(1.0, abs(z0))
    others = [r for r in rs[1:] if abs(abs(r) - abs(z0)) > tol]
    if others:
        detail["gap"] = abs(others[0] - z0)
        detail["gap_ratio"] = detail["gap"] / abs(z0) if abs(z0) else float("inf")
    detail["conjugate_pair"] = len(others) < len(rs) - 1

    # History: the same root, and its residue, at the two orders below.
    #
    # Walk DOWN past orders that have no approximant rather than giving up at
    # the first one.  A series in even powers only -- Painleve I's is, being a
    # series in w^2 -- has no [L/M] at odd L at all: every odd order raises
    # "denominator vanishes at 0".  Stepping by a fixed 1 landed on those and
    # came back with no history, so drift and rate were never computed and the
    # verdict was reached on nothing.  Consecutive entries may therefore be two
    # orders apart; the rate is then a two-step ratio, which SEPARATES THE
    # CASES MORE WIDELY (a geometric rate r shows up as r^2), so the same
    # threshold holds.
    hist = [(z0, res)]
    probe = L - 1
    while len(hist) < 3 and probe >= 1 and len(b) >= 2 * probe + 1:
        try:
            z, r = _leading(b, probe, M - (L - probe), near=hist[-1][0])
            if z is not None:
                hist.append((z, r))
        except (ArithmeticError, ValueError):
            pass
        probe -= 1
    hist = [h for h in hist if h[0] is not None]
    detail["orders_compared"] = len(hist)

    if len(hist) >= 2:
        detail["drift"] = abs(hist[1][0] - hist[0][0])
        detail["drift_ratio"] = detail["drift"] / abs(z0) if abs(z0) else 0.0
        if res is not None and hist[1][1] is not None:
            scale = max(abs(res), abs(hist[1][1]))
            detail["res_change"] = (abs(res - hist[1][1]) / scale
                                    if scale > 0 else 0.0)
    if len(hist) >= 3:
        prev = abs(hist[2][0] - hist[1][0])
        detail["rate"] = (detail["drift"] / prev) if prev > 0 else float("inf")

    # 1. The residue has settled -> there is something there with a strength,
    #    and we have it.  This is the strongest evidence available and it wins.
    rc = detail.get("res_change")
    if rc is not None and rc < _RES_SETTLED:
        detail["why"] = f"residue settled: moved {rc:.2e} < {_RES_SETTLED:g}"
        return SingularityKind.POLE, z0, res, detail

    # 2. Still moving.  Is it closing in, or laying another link?
    rate = detail.get("rate")
    clustered = detail.get("gap_ratio", float("inf")) < 0.5
    if rate is not None and rate < _RATE_CONVERGING and not clustered:
        detail["why"] = (f"root closing in (rate {rate:.3f}) but residue still "
                         f"moving ({rc:.2e}): raise the order")
        return SingularityKind.UNRESOLVED, z0, None, detail

    # 3. Nothing to judge on.  At the lowest orders there is no order below
    #    to compare against, and "cut" would be a confident answer built on no
    #    evidence -- it named the rational pair at [2/2] a cut, with roots
    #    sitting on +-2*pi*i exactly.
    if rc is None and rate is None and not clustered:
        detail["why"] = (f"no order below [{L}/{M}] to compare against: "
                         f"nothing has been shown to settle or to move")
        return SingularityKind.UNRESOLVED, z0, None, detail

    why = []
    if clustered:
        why.append(f"roots clustered along a ray (gap/|z0| "
                   f"{detail['gap_ratio']:.3f})")
    if rate is not None:
        why.append(f"root not converging (rate {rate:.3f})")
    if rc is not None:
        why.append(f"residue sliding ({rc:.2e} per order)")
    detail["why"] = "; ".join(why) or "chain of roots, nothing settled"
    return SingularityKind.CUT, z0, None, detail


def borel_singularity(c, L=None, M=None):
    """The nearest Borel-plane singularity, and its residue.

    The Pade denominator's roots ARE the singularities of the Borel transform:
    the approximant locates them without being told they exist.  For the
    Euler-Stieltjes series a root sits at z = -1 at every order from [1/1] up.

    Returns (z0, residue) with residue = P(z0) / Q'(z0), the standard formula
    for a simple pole.
    """
    b = borel(c)
    if L is None or M is None:
        k = (len(b) - 1) // 2
        L, M = k, k
    kind, z0, res, _ = classify_singularity(c, L, M)
    if kind != SingularityKind.POLE:
        # The LOCATION is still usable -- it is where the singularity is, and
        # it is what sets the exponent of the flat term.  The residue is not:
        # a branch point has none, and an unconverged chain's leading pole has
        # one that is still moving.
        return z0, None
    return z0, res


def flat_term(c, eps, L=None, M=None):
    """Size of the exponentially small ambiguity in the Borel sum.

    THIS IS THE OBJECT THE VALUE GROUP CANNOT HOLD, measured numerically.

    S and S + C*exp(-1/eps) have identical asymptotic expansions, so the
    coefficients cannot distinguish them -- but the Borel plane can.  The
    Laplace integral runs along u > 0 and the transform's pole sits at
    u0 = z0/eps; when eps has the sign that puts u0 ON that ray, the integral
    is ambiguous by the contour choice, and the ambiguity is exactly the flat
    term:

        residue of exp(-u) B(eps*u) at u0  =  exp(-u0) * Res_B(z0) / eps
        ambiguity                          =  pi * |that|

    For this series z0 = -1 and Res_B = 1, giving (pi/|eps|) * exp(-1/|eps|),
    which is what a transseries representation of exp(-1/eps) would have to
    reproduce.  So this is the oracle to build that against: a number for an
    object with no grade.

    Returns (u0, ambiguity), or (None, 0.0) when the pole is off the ray.
    """
    if L is None or M is None:
        k = (len(borel(c)) - 1) // 2
        L, M = k, k
    kind, z0, res, detail = classify_singularity(c, L, M)
    if z0 is None:
        return None, 0.0
    if kind == SingularityKind.CUT:
        raise NotImplementedError(
            f"the Borel transform has a BRANCH POINT at {z0!r}, not a pole "
            f"({detail.get('roots')} chained roots; {detail.get('why')}).  The "
            f"exponent of the flat term is still exp({z0!r}/eps), but its SIZE "
            f"is not a residue -- a branch point has none.  Returning one "
            f"would be a number with no meaning attached.")
    if kind == SingularityKind.UNRESOLVED:
        raise NotImplementedError(
            f"the singularity at {z0!r} has not been resolved at order "
            f"[{L}/{M}]: {detail.get('why')}.  The flat term's size is that "
            f"residue, so it cannot be quoted yet.")
    # Which singularity is ON the ray -- not necessarily the nearest one.
    P, Q = pade(borel(c), L, M)
    onray = _on_ray(poles(Q), eps)
    if not onray:
        return None, 0.0                    # nothing on the integration ray
    zr = onray[0]
    dq = _evaluate(dpoly(Q), zr)
    if dq == 0:
        return None, 0.0
    res_r = _evaluate(P, zr) / dq
    u0 = float(getattr(zr, "real", zr)) / eps
    return u0, math.pi * abs(res_r) * math.exp(-u0) / abs(eps)


# ---------------------------------------------------------------------------
# Lateral summation: getting a number when the Laplace ray is blocked.
# ---------------------------------------------------------------------------
#
# When a Borel singularity sits ON u > 0 the straight integral does not exist.
# Cauchy says the contour may be swung to arg u = theta instead, and the answer
# does not depend on which theta is used -- only on which SIDE the singularity
# was passed.  The two sides differ by exactly the flat term, which is the
# whole point: the ambiguity is not an error, it IS the invisible object.
#
# In the pair helpers below, None means NOTHING -- no such part exists.  It is
# NOT 0.0.  A bare Python zero meeting a Composite converts to |1|_-1 (R1), and
# it lands in the derivative grades, which is exactly what the panel
# integration reads.  Writing `d.get(k, 0.0)` for an absent coefficient did
# that, and the integral came back as 3.2e+03 where the answer is 0.697.

_THETA = math.pi / 3        # fallback; theta=None picks one from the geometry
_CLEARANCE = 1.2            # how far the contour must pass any singularity
_THETA_MIN, _THETA_MAX = 0.15, 1.2


def _ray_clearance(us, th, side):
    """Distance from the ray arg u = side*th to the nearest singularity."""
    best = float("inf")
    for u in us:
        if u == 0:
            return 0.0
        phi = cmath.phase(u) - side * th
        phi = abs((phi + math.pi) % (2 * math.pi) - math.pi)
        best = min(best, abs(u) if phi >= math.pi / 2
                   else abs(u) * math.sin(phi))
    return best


def _auto_theta(rs, eps, side=+1):
    """The SMALLEST swing that clears every singularity by _CLEARANCE.

    Both ends cost something, and both were measured:

      * too small and the contour passes close to a singularity, so a panel is
        wider than the Taylor radius and the composite quadrature expands a
        DIVERGENT series;
      * too large and the improper integrator loses digits on the OSCILLATION.
        exp(-u) along arg u = t oscillates tan(t) times per e-fold of decay,
        and at eps = +1 -- where the only pole sits at u = -1, nowhere near any
        ray tested -- the error runs 1.1e-16 at t = 0.1 and 0.3, 5.4e-15 at
        pi/4, 8.8e-12 at pi/3, 7.3e-10 at 1.2.  Tightening tol from 1e-10 to
        1e-15 moves none of those, so it is not panel resolution; and a FINITE
        upper limit is far worse, saturating at 2.4e-02 however large the
        cutoff, so the improper path is the good one.

    Clearance 1.2 is read off the blocked ray, pole at u0 = 1/|eps|:

        u0 = 1   best t = 1.20   clearance 0.93   rel err 2.6e-10
        u0 = 2   best t = 0.60   clearance 1.13   rel err 1.3e-15
        u0 = 4   best t = 0.40   clearance 1.56   rel err 4.8e-16

    Where the clearance is reachable the method is exact to machine precision.
    Only u0 near 1 pays, and it pays because sin(theta) <= 1 puts 1.2 out of
    reach there -- not because the contour is doing anything wrong.

    _THETA_MIN IS NOT COSMETIC.  Absolute clearance alone is not enough when
    the singularity is FAR out: the quadrature never refines panels where the
    integrand is ~1e-13, so a distant pole sits inside a wide panel however
    many units of clearance it has, and the divergent Taylor expansion there
    corrupts both laterals differently -- which shows up as a spurious
    ambiguity, since the ambiguity IS their difference.  Painleve I, ratio
    |S+-S-|/2 over exp(-A t^(5/4)), which must sit near 0.077:

        theta     0.05        0.10     0.20     0.30     0.60     1.00
        t=3       1.9e+05     0.0812   0.0812   0.0812   0.0812   0.0812
        t=4       1.4e+04     0.0788   0.0789   0.0789   0.0789   0.0789
        t=5       1.5e+07     0.0767   0.0767   0.0767   0.0767   0.0767
        t=6       3.3e+06     0.0751   0.0753   0.0753   0.0753   0.0763

    Clean at 0.10 and above at every t, wrong below it, with clearances of
    0.37 to 0.88 at 0.05 -- so the floor is angular, not metric.  0.15 leaves
    a margin, and costs nothing: the oscillation penalty is still below the
    last bit out to theta = 0.3.
    """
    us = [complex(r) / eps for r in rs]
    if not us:
        return _THETA_MIN
    n = 64
    grid = [_THETA_MIN + (_THETA_MAX - _THETA_MIN) * i / (n - 1)
            for i in range(n)]
    for th in grid:
        if _ray_clearance(us, th, side) >= _CLEARANCE:
            return th
    return max(grid, key=lambda th: _ray_clearance(us, th, side))

def _pm(x, y):
    return None if (x is None or y is None) else x * y


def _pa(x, y):
    return y if x is None else (x if y is None else x + y)


def _ps(x, y):
    if y is None:
        return x
    return -y if x is None else x - y


def _cmul(a, b):
    return (_ps(_pm(a[0], b[0]), _pm(a[1], b[1])),
            _pa(_pm(a[0], b[1]), _pm(a[1], b[0])))


def _cdiv(a, b):
    den = _pa(_pm(b[0], b[0]), _pm(b[1], b[1]))
    nr = _pa(_pm(a[0], b[0]), _pm(a[1], b[1]))
    ni = _ps(_pm(a[1], b[0]), _pm(a[0], b[1]))
    return (None if nr is None else nr / den,
            None if ni is None else ni / den)


def _eval_pair(p, z):
    """Horner at a complex point carried as a (real, imag) PAIR.

    The library's integrator hands the integrand a Composite, and a Composite
    does not multiply by a Python complex.  Two Composites do the job, and
    every operation stays inside the library's own arithmetic.
    """
    d = _clean(p).coeffs_dict()
    if not d:
        return (None, None)
    deg = max(d)
    acc = (d.get(deg), None)
    for k in range(deg - 1, -1, -1):
        acc = _cmul(acc, z)
        ck = d.get(k)                      # absent is NOTHING, never 0.0
        if ck is not None:
            acc = (_pa(acc[0], ck), acc[1])
    return acc


def resum_lateral(c, eps, L=None, M=None, side=+1, theta=None,
                  upper=float('inf'), param="arc", tol=1e-10):
    """Borel sum along arg u = side*theta, passing a blocked ray on one side.

    Returns a complex value.  side=+1 goes above the singularity, side=-1
    below; for a series with real coefficients the two are conjugates, and
    their difference is the flat term.

    THETA.  Cauchy makes the answer independent of theta as long as no
    singularity is crossed, so any theta in (0, pi/2) is the same number --
    analytically.  Numerically both ends cost something, and _auto_theta
    carries the measurements and picks the smallest swing that clears every
    singularity.  theta=None uses it; passing a number overrides it.
    """
    b = borel(c)
    if L is None or M is None:
        k = (len(b) - 1) // 2
        L, M = k, k
    P, Q = pade(b, L, M)
    if theta is None:
        theta = _auto_theta(poles(Q), eps, side)
    if not (0.0 < theta < math.pi / 2):
        raise ValueError(
            f"theta must lie strictly inside (0, pi/2); got {theta!r}.  At 0 "
            f"the contour IS the blocked ray, and a zero rotation multiplies a "
            f"Composite by Python 0.0, which converts to |1|_-1 (R1) and "
            f"quietly poisons the integrand -- it returned 1.2e-03 where the "
            f"answer is exact.")
    if param == "real":
        # Parameterise by the REAL PART of u, not by arc length along the ray:
        #     u = x*(1 + i*side*tan(theta)),   x from 0 to infinity
        # Same contour, same answer by Cauchy -- but exp(-u) now decays as
        # exp(-x), exactly as on the straight path, instead of exp(-x cos t).
        # Arc length makes the decay rate cos(theta) and the tail correspondingly
        # longer, which is where the digits were going.
        rc, rd = 1.0, side * math.tan(theta)
    elif param == "arc":
        rc, rd = math.cos(theta), side * math.sin(theta)
    else:
        raise ValueError(f"param must be 'real' or 'arc', got {param!r}")

    def pair(s):
        ur, ui = s * rc, s * rd            # u along arg u = side*theta
        e = exp(-ur)                       # exp(-u) = e^-ur (cos ui - i sin ui)
        E = (e * cos(ui), -(e * sin(ui)))
        z = (eps * ur, eps * ui)
        B = _cdiv(_eval_pair(P, z), _eval_pair(Q, z))
        return _cmul(_cmul(E, B), (rc, rd))   # times du/ds

    def part(i):
        return float(integrate(lambda s: (lambda v: 0.0 if v is None else v)(
            pair(s)[i]), 0.0, upper, tol=tol))

    return complex(part(0), part(1))


def resum_median(c, eps, L=None, M=None, theta=None, upper=float('inf')):
    """The two lateral sums, averaged -- and the flat term that separates them.

    Returns (median, ambiguity).  The median is the real solution the series
    describes; the ambiguity is |S+ - S-|/2, which is the SAME number
    flat_term() computes from the residue when the singularity is a pole, and
    which flat_term cannot give at all when it is a branch point.  Here it is
    measured from the contour instead, so it exists either way.
    """
    sp = resum_lateral(c, eps, L, M, +1, theta, upper)
    sm = resum_lateral(c, eps, L, M, -1, theta, upper)
    return (sp + sm) / 2.0, abs(sp - sm) / 2.0
