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
import math

from .composite_lib import Composite, integrate, exp as _cexp

_TOL = 1e-13


def _clean(c):
    """Drop terms whose coefficient is numerically zero.

    The library keeps EXPRESSED zeros on purpose -- a dimension exists because
    the computation built it.  That is right for the arithmetic and wrong for
    a degree test, which must not see a leading 0*z^5 as degree 5.
    """
    return Composite({k: v for k, v in c.coeffs_dict().items()
                      if abs(v) > _TOL})


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
    r_prev = Composite({n: 1.0})          # z^(L+M+1)
    r_cur = poly(b[:n])
    s_prev = Composite({})                # cofactor of the series: 0
    s_cur = Composite({0: 1.0})           # then 1

    while True:
        dr = degree(r_cur)
        if dr is None or dr <= L:
            break
        q, r_next = polydiv(r_prev, r_cur)
        r_prev, r_cur = r_cur, r_next
        s_prev, s_cur = s_cur, _clean(s_prev - q * s_cur)

    P, Q = _clean(r_cur), _clean(s_cur)
    q0 = Q.coeffs_dict().get(0, 0.0)
    if abs(q0) < _TOL:
        raise ValueError("Pade denominator vanishes at 0; try another (L, M)")
    return (Composite({k: v / q0 for k, v in P.coeffs_dict().items()}),
            Composite({k: v / q0 for k, v in Q.coeffs_dict().items()}))


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
    d = _clean(Q).coeffs_dict()
    if not d:
        return []
    deg = max(d)
    coef = [d.get(k, 0.0) for k in range(deg, -1, -1)]
    return sorted(_np.roots(coef), key=lambda r: abs(r))


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

    def integrand(u):
        z = eps * u
        return _cexp(-u) * (_evaluate(P, z) / _evaluate(Q, z))

    return integrate(integrand, 0.0, upper), (P, Q)


def dpoly(p):
    """d/dz of a polynomial-composite: coefficient k*p_k moves to grade k-1."""
    return _clean(Composite({k - 1: k * v
                             for k, v in _clean(p).coeffs_dict().items()
                             if k >= 1}))


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
    P, Q = pade(b, L, M)
    rs = poles(Q)
    if not rs:
        return None, None
    z0 = rs[0]
    return z0, _evaluate(P, z0) / _evaluate(dpoly(Q), z0)


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
    z0, res = borel_singularity(c, L, M)
    if z0 is None:
        return None, 0.0
    if abs(getattr(z0, "imag", 0.0)) > 1e-9:
        return None, 0.0                    # complex pole: not on the real ray
    z0 = float(getattr(z0, "real", z0))
    u0 = z0 / eps
    if u0 <= 0:
        return None, 0.0                    # pole off the integration ray
    return u0, math.pi * abs(res) * math.exp(-u0) / abs(eps)
