#!/usr/bin/env python3
"""Global root finding on composites: exclusion by Taylor bound, then Householder.

A scan-and-bisect root finder samples the function and bisects wherever the sign
changes.  It silently misses every root that does not straddle a sample, and it
cannot tell you whether it found them all.  That is not a hypothetical: it is how
the Kerr spin inversion produced a wrong answer -- the coefficient function has
three roots and the bisection assumed monotonicity.

A composite knows more than a sample.  Seeding f with an infinitesimal returns
the whole Taylor expansion at that point in one evaluation, and two facts follow:

  EXCLUSION.  On |h| <= rad,  |f(mid+h)| >= |c0| - sum_{k>=1} |ck| rad^k.
              If that is positive the interval provably holds no root and the
              whole subtree is dropped.  Sampling can only ever say "no sign
              change here", which is not the same statement.

  UNIQUENESS. If  |c1| > sum_{k>=2} k |ck| rad^(k-1)  then f' cannot vanish on
              the interval, so f is monotone and holds AT MOST one root.  That
              turns "some roots somewhere" into "exactly these, one per box".

Both bounds truncate the series at `order`, so they are reliable rather than
rigorous.  Carrying an interval remainder term would make them rigorous -- that
is the Taylor-model construction, and it is the natural next step.

Polishing uses Householder of arbitrary order d (d=1 Newton, d=2 Halley),
which costs nothing extra here because the derivatives were already computed:

    x <- x + d * [ (1/f)^(d-1) / (1/f)^(d) ]      order d+1 convergence
"""
import math
import composite.composite_lib as cl


def taylor(f, at, order):
    """Taylor coefficients [c0, c1, ..., c_order] of f at `at`, one evaluation.

    c_k is f^(k)(at)/k!, read straight off dimension -k.  _min_terms is raised
    so transcendentals inside f expand deeply enough -- without it they use
    their own defaults (12 or 15) and the high coefficients come back 0.
    """
    old = cl._min_terms[0]
    cl._min_terms[0] = max(12, order + 2)
    try:
        r = f(cl._seeded(at))
        if not isinstance(r, cl.Composite):
            raise TypeError("f must return a Composite")
        return r
    finally:
        cl._min_terms[0] = old


def coeffs(F, order):
    """Taylor coefficients [c0..c_order] of a seeded result, c_k at dimension -k."""
    return [F.coeff(-k) for k in range(order + 1)]


def dseries(F, order):
    """Series of f' from the series of f:  c'_k = (k+1) c_(k+1).

    Differentiation is a shift-and-scale of the coefficients, so it stays inside
    the algebra.  Having it makes _monotone unnecessary: "f is monotone on this
    box" is exactly "f' has no root on this box", which is the exclusion test
    already written.  One concept instead of two.
    """
    c = coeffs(F, order + 1)
    return cl.Composite({-k: (k + 1) * c[k + 1] for k in range(order + 1)
                         if c[k + 1] != 0.0})


def _lead(x):
    """(leading dimension, |leading coefficient|) -- sorts composites by size.

    Dimension dominates: |1|_1 (an infinity) is larger than any |v|_0, which is
    larger than any |v|_-1 (an infinitesimal).  That ordering IS the answer to
    every degenerate case below; nothing needs a branch.
    """
    d = x.coeffs_dict()
    if not d:
        return (-10**9, 0.0)
    k = max(d)
    return (k, abs(d[k]))


def _coeff_ratio(c):
    """Ratio of successive coefficient magnitudes, as a COMPOSITE.

    NOT unconditionally a radius of convergence -- that is only one reading of
    it.  For a genuinely convergent series |c_k| ~ A/R^k, so the ratio estimates
    R.  For a TERMINATED series the same ratio is |0|_0/|0|_0, which both operands
    convert to |1|_-1, giving exactly |1|_0 = 1: the ratio of two equal
    first-order infinitesimals, lim x/x.  That is not a degenerate case being
    papered over, it is the correct value, and it says something true -- the two
    coefficients vanish to the SAME order.  Had they vanished to different
    orders the algebra would say so: (ZERO*ZERO)/ZERO is |1|_-1, infinitesimal.

    Naming it a radius was the one wrong thing here; the arithmetic never was.
    Callers must not read it as a radius without checking what else is true --
    _tail does not need to, because a vanishing c_n settles the bound on its own.
    """
    n = len(c) - 1
    ratios = [cl.R(abs(c[k-1])) / cl.R(abs(c[k]))
              for k in range(max(2, n - 3), n + 1)]
    return min(ratios, key=_lead) if ratios else cl.INF


def _tail(c, rad):
    """Geometric bound on the discarded terms, as a COMPOSITE.

        tail  =  |c_n| * rad^n * q / (1 - q),        q = rad / R

    Every degenerate case falls out of the dimension, with no special case:

      series terminated  -> c_n = 0, and R(0) is |1|_-1, so the whole product is
                            INFINITESIMAL (dim < 0).  Negligible, correctly.
      q = 1 exactly      -> 1-q is R(0) = |1|_-1, dividing by it lifts the result
                            to dim +1.  INFINITE, correctly: no bound exists.
      q < 1              -> everything sits at dim 0 and it is an ordinary number.

    The float version of this needed two hand-written guards and still produced a
    nan that hung the search.
    """
    n = len(c) - 1
    cn = max(abs(v) for v in c[max(0, n-2):])     # last few, so one stray zero
    q = cl.R(rad) / _coeff_ratio(c)               # does not kill the bound
    return cl.R(cn) * cl.R(rad ** n) * q / (cl.R(1.0) - q)


def _bound(c, rad):
    """Read the tail composite back as a float: inf, a real, or 0."""
    d, v = _lead(_tail(c, rad))
    if d > 0:
        return float('inf')                        # unbounded tail
    if d < 0:
        return 0.0                                 # infinitesimal tail
    return v


def _no_root(c, rad):
    """True when the Taylor bound proves f has no zero on [mid-rad, mid+rad].

    THIS IS FLOAT ARITHMETIC ON PURPOSE, AND IT IS ONLY SAFE WHILE THE
    COEFFICIENTS ARE SCALARS.

    The bound needs |c_k|, and absolute value is not a ring operation -- but it
    IS available in the algebra as sqrt(x*x), which works at every dimension
    including odd ones, because squaring makes the exponent even before sqrt
    halves it.  Measured: computing this whole bound in composites gives
    identical values at 11x the cost, since each abs becomes a series expansion.

    So floats win here for one reason only: `c` comes out of `.coeff()` as plain
    scalars, and abs() of a scalar has no degenerate case to get wrong.  THE DAY
    THE COEFFICIENTS CARRY ANYTHING ELSE -- lanes, an error budget, a symbolic
    parameter -- abs() becomes meaningless and this must move to sqrt(c*c).
    Widening the coefficient type without changing this function is a silent bug.

    Only the final comparison is genuinely outside any algebra: `>` is an order
    relation, not arithmetic.  Same for min/max in _condense.
    """
    t = _bound(c, rad)
    if not math.isfinite(t):
        return False
    return abs(c[0]) > sum(abs(c[k]) * rad**k for k in range(1, len(c))) + t


def _monotone(F, order, rad):
    """f' cannot vanish here -> f holds at most one root.  Exclusion, on f'."""
    return _no_root(coeffs(dseries(F, order), order), rad)


def householder(f, x, order=12, d=2, iters=60, tol=1e-15):
    """Householder iteration of order d (1=Newton, 2=Halley).  None if it leaves."""
    for _ in range(iters):
        c = coeffs(taylor(f, x, max(d + 1, 3)), max(d + 1, 3))
        if c[0] == 0.0:
            return x
        # derivatives of 1/f from the series of f, via composite division
        g = cl.Composite({-k: c[k] for k in range(len(c)) if c[k] != 0.0})
        if not g.coeffs_dict():
            return x
        inv = cl.Composite({0: 1.0}) / g
        num = inv.coeff(-(d-1)) * math.factorial(d-1)
        den = inv.coeff(-d) * math.factorial(d)
        # Let the algebra answer the zero case instead of guarding it.  R(0) is
        # |1|_-1, so R(num)/R(den) with den == 0 lands at dimension +1 -- an
        # infinity whose ORDER says how deep the zero was.  A float guard
        # (`if den == 0.0`) throws that away and returns nothing.
        ratio = cl.R(num) / cl.R(den)
        rd = ratio.coeffs_dict()
        if not rd or max(rd) != 0:
            return None                      # infinite or infinitesimal step
        step = d * ratio.coeff(0)
        x += step
        if abs(step) < tol * max(1.0, abs(x)):
            return x
    return x


def roots(f, a, b, order=12, tol=1e-11, max_depth=80, d=2):
    """Every root of f on [a, b].  Subdivide, drop provably-empty boxes, polish.

    Returns (roots, stats) where stats records how much of the interval was
    eliminated by the Taylor bound rather than searched -- if that number is low
    the order is too small for this function.
    """
    found, boxes, pruned, split = [], [(a, b, 0)], 0, 0
    while boxes:
        lo, hi, depth = boxes.pop()
        mid, rad = 0.5*(lo+hi), 0.5*(hi-lo)
        F = taylor(f, mid, order)
        c = coeffs(F, order)
        if _no_root(c, rad):
            pruned += 1
            continue
        if rad < tol or depth >= max_depth or _monotone(F, order, rad):
            r = householder(f, mid, order, d)
            if r is not None and lo - 1e-9 <= r <= hi + 1e-9:
                found.append(r)
            elif abs(c[0]) < 1e-10:
                found.append(mid)
            continue
        split += 1
        boxes.append((lo, mid, depth+1))
        boxes.append((mid, hi, depth+1))
    return _condense(f, found, order, tol), {"pruned": pruned, "split": split,
                                             "raw": len(found)}


def multiplicity(f, x, order=12):
    """Estimated order of the zero f is approaching at x.

    Near a root of multiplicity m the Newton step u = f/f' behaves like (x-r)/m,
    so u' -> 1/m.  Both f and f' are series, so u is ONE COMPOSITE DIVISION and
    u' is its coefficient at dimension -1.  There is no formula to derive.

    This used to be 1/(1 - 2 c0 c2 / c1^2) -- the same thing expanded by hand,
    which is where an algebra slip would live and which does not generalise.

    Counting "how many leading coefficients look like zero" instead is unusable:
    at a candidate 1e-8 from a double root, c1 is 2e-8, which no fixed threshold
    separates from a genuine zero.
    """
    F = taylor(f, x, max(3, order))
    Fp = dseries(F, max(3, order))
    if not Fp.coeffs_dict():
        return 1                             # f' has no terms at all
    u = F / Fp                               # u = f/f' ~ (x-r)/m near order-m root
    # m = 1/u'.  Done in the algebra: a zero u' gives |1|_1 rather than a
    # ZeroDivisionError, so the "is it zero" and "is it finite" tests both become
    # one question -- does the result sit at dimension 0.
    inv = cl.R(1.0) / cl.R(u.coeff(-1))
    d = inv.coeffs_dict()
    if not d or max(d) != 0:
        return 1
    m = round(inv.coeff(0))
    return m if 1 <= m <= 16 else 1


def _condense(f, found, order, tol):
    """Polish each candidate at its own multiplicity, THEN dedupe tightly.

    Order matters.  Clustering by distance first cannot work: the spray around a
    double root is wider than the gap between two distinct roots 1e-4 apart, so
    any tolerance either merges real roots or splits fake ones.  Modified Newton
    (x <- x - m f/f') restores full accuracy once m is known, and every member of
    a spray then lands on the same point, so a tight dedupe finishes the job.
    """
    polished = []
    for x in found:
        m = multiplicity(f, x, order)
        if m > 1:
            for _ in range(100):
                c = coeffs(taylor(f, x, max(m + 1, 3)), max(m + 1, 3))
                if c[1] == 0.0:
                    break
                step = -m * c[0] / c[1]
                x += step
                if abs(step) < tol * max(1.0, abs(x)):
                    break
        polished.append(x)
    polished.sort()
    out = []
    for r in polished:
        if not out or abs(r - out[-1]) > 1e-9 * max(1.0, abs(r)):
            out.append(r)
    return out
