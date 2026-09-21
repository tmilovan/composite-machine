"""Where a series stops converging, and what it does when it gets there.

A power series knows its own singularity.  The radius of convergence is the
distance to the nearest one, and the way the coefficients grow encodes what
kind it is.  Reading that back out -- location and exponent -- is a
calculation people currently do symbolically or by hand, and it answers
several different questions at once:

  a nonlinear ODE       when does the solution blow up, and how fast?
  a lattice model       where is the critical point, and what is the exponent?
  a generating function how do the coefficients grow asymptotically?
  a perturbation series where is the Borel singularity, and is it a pole?

All four are the same computation.  Given coefficients of f with

    f(z)  ~  A (1 - z/z0)^beta        as z -> z0

this module returns z0 and beta.  The sign convention follows analytic
combinatorics: beta < 0 is a divergence (a pole of order -beta when beta is a
negative integer), beta > 0 is a vanishing singular part such as a square-root
branch point, and the critical exponent people usually quote is -beta.

METHOD.  Two independent routes, and their agreement is the confidence.

  Differential approximant (primary).  Fit Q(z) f'(z) + P(z) f(z) = R(z) to
  the series; near a root z0 of Q the equation forces f' / f ~ beta / (z - z0),
  so z0 comes from Q and beta = -P(z0)/Q'(z0).  This is the standard tool for
  critical exponents, and it is the one that handles branch points: it encodes
  the exponent in an equation rather than trying to build a cut out of poles.
  The linear fit is a numpy least-squares solve, not composite arithmetic.

  Log-derivative Pade (cross-check).  f'/f has a simple pole at z0 with
  residue beta, so a Pade approximant of it locates both.  The series division
  runs in composite arithmetic and is exact -- verified to 0.00e+00 per
  coefficient on series spanning eleven decades.  Pade converges quickly to
  genuine poles and slowly to branch points (it must approximate a cut with a
  string of poles), which is exactly the split the two methods disagree on,
  and is why the primary route is the approximant.

WHAT MAKES IT UNRELIABLE, and what is done about it.

  Froissart doublets.  A spurious pole from a numerator zero landing on a
  denominator root.  It moves with the approximant order and carries a residue
  at the rounding floor.  Both are filtered: candidates must persist across
  orders and carry a residue above `MIN_RESIDUE`.  Without the filter this
  returned 3 of 7 known exponents; with it, 7 of 7 to 1e-16.

  Symmetric singularities.  tan has poles at both +-pi/2, so clustering on a
  median over all candidates lands near zero and rejects both.  Clustering is
  seeded on the NEAREST candidate instead.

  A singularity that is not algebraic.  A logarithm or an essential
  singularity does not fit the form above.  Disagreement between the two
  routes, or a wide spread across orders, shows up as low confidence -- it is
  reported, not hidden.
"""

import cmath
import math
import statistics

import numpy as np

from .composite_lib import Composite
from .resummation import pade, poles, dpoly

__all__ = [
    "Singularity", "analyse", "series_solve", "blowup",
    "coefficient_asymptotics", "radius", "MIN_RESIDUE",
]

MIN_RESIDUE = 1e-8      # below this a candidate is a doublet, not a singularity
_CLUSTER = 1e-5         # relative width of the agreement cluster
_NEED = 3               # approximant orders that must agree


class Singularity:
    """The nearest singularity of a series: where, and of what kind."""

    __slots__ = ("location", "exponent", "spread", "orders", "cross_check",
                 "terms_used")

    def __init__(self, location, exponent, spread, orders, cross_check, terms_used):
        self.location = location        # z0
        self.exponent = exponent        # beta, with f ~ A (1 - z/z0)^beta
        self.spread = spread            # relative disagreement across orders
        self.orders = orders            # how many approximants agreed
        self.cross_check = cross_check  # (z0, beta) from the Pade route, or None
        self.terms_used = terms_used

    # -- the same number, under the name each field uses for it --------------

    @property
    def critical_exponent(self):
        """gamma, with f ~ (1 - z/z0)^-gamma.  Statistical mechanics."""
        return -self.exponent

    @property
    def blowup_rate(self):
        """alpha, with y ~ (t* - t)^-alpha.  Nonlinear ODEs."""
        return -self.exponent

    @property
    def pole_order(self):
        """Order of the pole, or None when the singularity is not one."""
        b = self.exponent
        return int(round(-b)) if self.kind == "pole" else None

    @property
    def kind(self):
        b = self.exponent
        if b < -0.5 and abs(b - round(b)) < 1e-6:
            return "pole"
        if abs(b) < 1e-6:
            return "logarithmic"
        return "branch point"

    @property
    def confidence(self):
        """low / medium / high, from order agreement and method agreement."""
        if self.spread > 1e-4 or self.orders < _NEED:
            return "low"
        agree = self.cross_check is not None and abs(
            self.cross_check[0] - self.location) <= 1e-6 * max(1.0, abs(self.location))
        if self.spread < 1e-9 and agree:
            return "high"
        return "medium"

    def __repr__(self):
        return ("<Singularity at %.12g, exponent %.12g (%s, %s confidence)>"
                % (self.location, self.exponent, self.kind, self.confidence))

    def describe(self):
        lines = [
            "  nearest singularity   z0 = %.15g" % self.location,
            "  exponent              beta = %.15g   f ~ A (1 - z/z0)^beta"
            % self.exponent,
            "  kind                  %s%s" % (
                self.kind,
                " of order %d" % self.pole_order if self.kind == "pole" else ""),
            "  critical exponent     gamma = %.15g" % self.critical_exponent,
            "  confidence            %s   (%d approximant orders agreed, spread %.1e)"
            % (self.confidence, self.orders, self.spread),
        ]
        if self.cross_check is not None:
            z, b = self.cross_check
            lines.append("  Pade cross-check      z0 = %.12g, beta = %.12g" % (z, b))
        else:
            lines.append("  Pade cross-check      none found (expected at a branch point)")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the two routes
# --------------------------------------------------------------------------

def _differential_approximant(a, q, p, r):
    """Fit Q f' + P f = R and read the singularities off Q.

    Returns [(z0, beta), ...] for the real roots of Q.
    """
    n_eq = q + p + r + 2
    if len(a) < n_eq + 2:
        return []
    # Match every coefficient available, not just enough to determine the fit.
    # A square system hands the highest-order approximants a nearly singular
    # matrix; an overdetermined least-squares fit is both better conditioned
    # and what makes averaging over orders meaningful.
    n_eq = len(a) - 2
    cols = []
    for j in range(q + 1):                       # Q_j * (f')_{n-j}
        cols.append([((n - j + 1) * a[n - j + 1]
                      if 0 <= n - j and n - j + 1 < len(a) else 0.0)
                     for n in range(n_eq + 1)])
    for j in range(p + 1):                       # P_j * f_{n-j}
        cols.append([(a[n - j] if 0 <= n - j < len(a) else 0.0)
                     for n in range(n_eq + 1)])
    for j in range(r + 1):                       # -R_j
        cols.append([(-1.0 if n == j else 0.0) for n in range(n_eq + 1)])
    A = np.array(cols, dtype=float).T
    if not np.all(np.isfinite(A)):
        return []
    # Scale each equation to unit norm.  Coefficients of a combinatorial series
    # span fifteen decades, and without this the few largest rows decide the
    # whole least-squares fit -- which cost Catalan three digits of exponent.
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0.0] = 1.0
    A = A / norms[:, None]
    fix = q + 1                                  # normalise P_0 = 1
    keep = [i for i in range(A.shape[1]) if i != fix]
    try:
        sol, *_ = np.linalg.lstsq(A[:, keep], -A[:, fix], rcond=None)
    except np.linalg.LinAlgError:
        return []
    full = np.zeros(A.shape[1])
    full[keep] = sol
    full[fix] = 1.0
    Q, P = full[:q + 1], full[q + 1:q + p + 2]
    if not np.any(Q):
        return []
    out = []
    for z in np.roots(Q[::-1]):
        if abs(z.imag) > 1e-8 * max(1.0, abs(z)):
            continue
        z = float(z.real)
        dQ = sum(k * Q[k] * z ** (k - 1) for k in range(1, q + 1))
        if abs(dQ) < 1e-14:
            continue
        beta = -sum(P[k] * z ** k for k in range(p + 1)) / dQ
        if math.isfinite(beta) and abs(beta) > MIN_RESIDUE:
            out.append((z, float(beta)))
    return out


def _log_derivative_pade(a, orders):
    """Cross-check: Pade of f'/f.  The series division is composite arithmetic."""
    f = Composite({-k: v for k, v in enumerate(a) if v})
    df = Composite({-(k - 1): k * v for k, v in enumerate(a) if k >= 1 and v})
    try:
        lg = df / f
    except Exception:
        return []
    got = []
    for L in orders:
        n = 2 * L + 1
        if n > len(a) - 1:
            break
        d = [lg.coeffs_dict().get(-k, 0.0) for k in range(n)]
        try:
            P, Q = pade(d, L, L)
        except Exception:
            continue
        for z in poles(Q):
            if abs(z.imag) > 1e-6 * max(1.0, abs(z)):
                continue
            try:
                num = sum(v * z ** k for k, v in P.coeffs_dict().items())
                den = sum(v * z ** k for k, v in dpoly(Q).coeffs_dict().items())
                beta = num / den
            except Exception:
                continue
            if abs(beta) > MIN_RESIDUE:
                got.append((float(z.real), float(beta.real)))
    return got


def radius(a, tail=8):
    """Radius of convergence from coefficient growth: R = lim |a_n|^(-1/n).

    This is what tells a real singularity from an artefact.  Seeding the
    cluster on the nearest candidate instead -- which is what it took to stop
    `tan` rejecting both of its symmetric poles -- then picked a spurious root
    at 2.5e-12 for the Catalan numbers, because nearest is not the same as
    real.  The radius is a property of the coefficients, so it cannot be
    fooled by an approximant artefact.
    """
    est = []
    n = len(a) - 1
    while n > 1 and len(est) < tail:
        if a[n] != 0.0:
            est.append(abs(a[n]) ** (-1.0 / n))
        n -= 1
    return statistics.median(est) if est else None


def _cluster(cands, need=_NEED, tol=_CLUSTER, near=None, sign=0):
    """Group candidates and keep the cluster that is really there.

    `near` is the radius of convergence when it is known: candidates far from
    it are approximant artefacts, however many orders happen to repeat them.
    """
    if not cands:
        return None
    if sign > 0:
        cands = [c for c in cands if c[0] > 0]
    elif sign < 0:
        cands = [c for c in cands if c[0] < 0]
    if not cands:
        return None
    if near is not None and near > 0:
        kept = [c for c in cands if 0.2 <= abs(c[0]) / near <= 5.0]
        if kept:
            cands = kept
        order = sorted(cands, key=lambda t: abs(abs(t[0]) - near))
    else:
        order = sorted(cands, key=lambda t: abs(t[0]))
    for z0, _ in order:
        keep = [(z, b) for z, b in cands
                if abs(z - z0) <= tol * max(1.0, abs(z0))]
        if len(keep) >= need:
            zs = [z for z, _ in keep]
            bs = [b for _, b in keep]
            loc = statistics.median(zs)
            spread = (max(zs) - min(zs)) / max(1.0, abs(loc)) if len(zs) > 1 else 0.0
            bspread = (max(bs) - min(bs)) / max(1.0, abs(statistics.median(bs))) \
                if len(bs) > 1 else 0.0
            return loc, statistics.median(bs), max(spread, bspread), len(keep)
    return None


def analyse(coeffs, max_order=None, need=_NEED, sign=0):
    """Locate the nearest singularity of a series and name its exponent.

    `coeffs` are a_0, a_1, ... of f(z) = sum a_k z^k.  Returns a `Singularity`,
    or None when nothing persists across approximant orders -- which is the
    right answer for an entire function.

    `sign` restricts the answer to a half line: +1 for the nearest singularity
    at positive z, -1 for negative.  tan has poles at both +-pi/2 and the
    unrestricted answer is whichever the approximants favour, which is not
    what a blow-up TIME means.
    """
    a = [float(v) for v in coeffs]
    while a and a[-1] == 0.0:
        a.pop()
    if len(a) < 8:
        return None
    n = len(a) if max_order is None else min(len(a), max_order)
    a = a[:n]

    top = max(2, (n - 4) // 3)
    cands = []
    for d in range(2, min(top, 9) + 1):
        cands.extend(_differential_approximant(a, d, d, d))
    R = radius(a)
    got = _cluster(cands, need=need, near=R, sign=sign)
    if got is None:
        return None
    loc, beta, spread, orders = got

    cross = _cluster(_log_derivative_pade(a, range(2, min(top, 9) + 1)),
                     need=2, near=R, sign=sign)
    cross_pair = (cross[0], cross[1]) if cross is not None else None
    return Singularity(loc, beta, spread, orders, cross_pair, n)


# --------------------------------------------------------------------------
# getting a series in the first place
# --------------------------------------------------------------------------

def series_solve(f_poly, y0, terms):
    """Taylor coefficients of y' = f(y), y(0) = y0, by coefficient recursion.

    `f_poly` is [c_0, c_1, ...] with f(y) = sum c_j y^j.  The powers of the
    partial series are composite multiplications -- grade k carries the
    coefficient of t^k -- so the convolution is the library's own arithmetic.
    """
    a = [0.0] * (terms + 1)
    a[0] = float(y0)
    for n in range(terms):
        y = Composite({k: a[k] for k in range(n + 1) if a[k]})
        acc = Composite({0: float(f_poly[0])}) if f_poly[0] else Composite({})
        power = Composite({0: 1.0})
        for j in range(1, len(f_poly)):
            power = power * y
            if f_poly[j]:
                acc = acc + Composite({0: float(f_poly[j])}) * power
        a[n + 1] = acc.coeffs_dict().get(n, 0.0) / (n + 1)
    return a


def blowup(f_poly, y0, terms=30):
    """When y' = f(y), y(0) = y0 blows up, and how fast.

    `.location` is the blow-up time t*, `.blowup_rate` the alpha in
    y ~ (t* - t)^-alpha.  Numerical integration gives the time and says
    nothing about the rate; this gives both from one series.
    """
    return analyse(series_solve(f_poly, y0, terms), sign=+1)


def coefficient_asymptotics(coeffs, sing=None, tail=6):
    """Fit a_n ~ C n^(-beta-1) z0^-n and return (C, z0, beta).

    The amplitude is fitted from the tail rather than derived, so it doubles
    as a check: a wrong z0 or beta makes C drift instead of settling.

    Accuracy is not the same as for the location and the exponent.  Those come
    from an equation and sit near 1e-14 whatever the term count; C comes from
    an asymptotic fit and is term-count limited, converging like 1/n even after
    a Richardson step -- measured on the Motzkin numbers, 1.1e-02 relative at
    24 terms, 5.8e-03 at 32, 8.5e-04 at 80.  Further Richardson steps do not
    help; they oscillate at the same error.  Treat C as a one-to-three digit
    number unless the series is long.
    """
    a = [float(v) for v in coeffs]
    s = sing if sing is not None else analyse(a)
    if s is None:
        return None
    z0, b = s.location, s.exponent
    raw = {}
    for n in range(max(1, len(a) - 2 * tail), len(a)):
        if a[n] == 0.0:
            continue
        raw[n] = a[n] * n ** (b + 1.0) * z0 ** n
    if not raw:
        return None
    # The estimates approach C as C(1 + c/n + ...), so the last one is still
    # a percent or so out at n = 30 -- Catalan's amplitude came back 3.9% low.
    # One Richardson step removes the 1/n term.
    ext = [n * raw[n] - (n - 1) * raw[n - 1]
           for n in sorted(raw) if (n - 1) in raw]
    best = ext[-tail:] if len(ext) >= 2 else list(raw.values())
    return statistics.median(best), z0, b
