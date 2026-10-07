# Composite Machine — Automatic Calculus via Dimensional Arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing available. Contact: tmilovan@fwd.hr

"""
composite_multivar.py -- Multi-Variable Calculus by Directional Composites
=========================================================================
Partial derivatives, gradients, Hessians, Jacobians, Laplacians, directional
derivatives, curl, divergence and limits of functions of several variables,
read off ORDINARY composites: one infinitesimal, used as implemented.

One composite carries one infinitesimal, so at a point it holds one number per
grade.  Along a direction c, f(x1 + c1 h, ..., xn + cn h) holds at grade k the
degree-k part of the Taylor polynomial evaluated at c:

    grade -k  =  sum over |a| = k  of  T_a * c1^a1 * ... * cn^an

Enough directions recover every T_a by a fixed linear combination (univariate
Taylor propagation with interpolation; Griewank, Utke and Walther, Math. Comp.
69, 2000).  Order K in n variables takes C(K+n-1, n-1) composites: n for a
gradient, n(n+1)/2 for a Hessian -- plus one check direction the jet must
predict, which refuses a point where f is not smooth.

Directions are (1, q_i2, ..., q_in) over the LOWER SET of index tuples with
i2 + ... + in <= K, q the Chebyshev-dyadic seed quantities (never 0, exact in
float64, never a simple ratio).  The ones with index sum <= k are unisolvent for
grade k, so each grade is a square system, solved once in exact rationals.  The
weights act on the composites' PARTS, never as composite sums: a partial sum
that is wholly zero would deposit a residue.

Functions are written with the composite_lib functions (sin, exp, ...).  These
are refused rather than read:
  - a float coercion (math.sin on a composite): ValueError "not composite";
  - an R1 residue (f - f, 0*f, f + 0): ResidueError.  In several variables the
    one infinitesimal is the direction's own parameter, and which variable a
    residue denotes is a convention of the direction set, not of the function.
    Write a zero as a plain 0;
  - a pole at the point (a term above grade 0 along a direction): PoleError.
    multivar_limit is the exception: an unbounded limit reads as +-inf;
  - a branch point (a non-integer grade, sqrt(x) at 0): BranchPointError;
  - a point where f is not smooth (a cone, a kink): NonSmoothError, from one
    extra check direction the jet must predict.

Replaced the MC implementation (tuple dimensions) on 2026-10-06; that is parked
in composite_multivar_mc.py.  See docs/MC Replacement - Directional Composites
(DRAFT).md.

Usage:
    from composite.composite_lib import sin
    from composite.composite_multivar import gradient_at, hessian_at, partial_derivative

    f = lambda x, y: x**2 * y + sin(y)
    gradient_at(f, [3, 2])                 # [12, 9 + cos(2)]
    hessian_at(f, [3, 2])                  # [[4, 6], [6, -sin(2)]]
    partial_derivative(f, [3, 2], [1, 1])  # 6

Author: Toni Milovan
"""
import math
import contextlib as _contextlib_dir
import functools as _functools_dir
import warnings as _warnings_dir
import itertools as _itertools_dir
from fractions import Fraction as _Fraction_dir
from typing import Callable, List


class PoleError(ValueError):
    """f is unbounded at the point: a directional composite has terms above
    grade 0.  Its finite grades are then Laurent coefficients that depend on the
    direction, and separating them gives numbers that look like derivatives and
    are not.  Measured: x*y/(x+y) at (1, -1) read as gradient [-0.83, 1.83]."""


class BranchPointError(ValueError):
    """A directional composite has a non-integer grade: sqrt(x) at x = 0 puts its
    content at grade -1/2.  The separation reads integer grades only, so it read
    nothing there -- measured: sqrt(x)*y at (0, 1) as gradient [0, 0], where
    d/dx is infinite.  The function is not differentiable at the point."""


class NonSmoothError(ValueError):
    """The jet does not predict a direction it was not built from.

    The separation fits a polynomial in the direction to each grade, and with
    exactly as many directions as unknowns the fit always succeeds -- the cone
    sqrt(x^2+y^2) at the origin read as gradient [1.21, 0].  One more direction,
    with its first component -1 so it lies in the half of the directions the set
    never covers, is predicted from the jet and compared.  Measured, as a
    fraction of the grade's size: smooth functions <= 7e-10 (to order 16, near
    a pole, at large points, removable singularities); cones and kinks 1.5 to 2.
    SMOOTHNESS_TOL = 1e-8 sits between them: 14x above the worst smooth case,
    eight orders below the cones.  It was 1e-6 and let d^20/dx^20 of exp(xy)
    through 13% wrong (disagreement 4.7e-8); 1e-8 refuses it.

    It is a smoothness check, not an error estimate: it measures the jet against
    a whole grade, and a single small high-order coefficient can be far worse
    (d^16/dx^16 of exp(xy) is 6e-6 off while the check reads 3.3e-10)."""


SMOOTHNESS_TOL = 1e-8


def _disagreement(actual, predicted, scale):
    return abs(actual - predicted) / scale if scale else 0.0


def _refuse_non_smooth(ratio, k, at):
    if ratio > SMOOTHNESS_TOL:
        raise NonSmoothError(
            "a check direction disagrees with the jet at %r, order %d, by %.1e of the "
            "grade's size (tolerance %.0e): either the function is not smooth there "
            "(a cone, a kink -- typically a disagreement near 1 at low order), or the "
            "order is beyond what float64 separation resolves for this many variables "
            "(a small disagreement at high order)" % (list(at), k, ratio, SMOOTHNESS_TOL))


def _mirror_check(plus, minus, at, upto):
    """f(at + v h) and f(at - v h) of a smooth f have grade -k equal up to the sign
    (-1)^k.  A cone or kink breaks it at grade -1."""
    for k in range(upto + 1):
        a, b = plus.get(-k, 0.0), minus.get(-k, 0.0)
        _refuse_non_smooth(_disagreement(a, (-1) ** k * b, max(abs(a), abs(b))), k, at)


class IncompleteJetError(ValueError):
    """A grade the function needs lies beyond what the arithmetic vouches for
    (Composite.complete_order), e.g. past a transcendental's `terms=` cap."""


@_contextlib_dir.contextmanager
def _reading_within_bounds():
    """Silence the truncation warning during the directional evaluations.

    A transcendental announces every term it drops past its cap (exp: order
    14).  These functions read only grades down to the order they were asked
    for, and _parts_of checks each composite's completeness against exactly
    that, so the warning would report something that is checked here instead.
    Only that warning is filtered; every other one passes through.
    """
    with _warnings_dir.catch_warnings():
        _warnings_dir.filterwarnings("ignore", message=r"\d+ terms? dropped", category=UserWarning)
        yield


def _parts_of(v, at, need):
    """A directional composite's coefficients, refusing an infinite part and a
    jet that is not complete to order `need`."""
    from composite.composite_lib import Composite
    if not isinstance(v, Composite):
        return {0: float(v)} if v != 0 else {}
    p = v.coeffs_dict()
    frac = sorted(g for g, c in p.items() if c != 0 and not float(g).is_integer())
    if frac:
        raise BranchPointError(
            "the function has a branch point at %r: grade %s along a direction, so it "
            "has no derivatives there" % (list(at), frac[-1]))
    # The pole first: dividing by an infinitesimal also lowers the completeness
    # bound, so at a pole the bound check would fire and name the wrong cause.
    # Only a nonzero coefficient is an infinite part.  A zero term above grade 0
    # is inert (R2): (exp(u) - 1)/u keeps the cancelled standard part of its
    # numerator as a zero term, and the division moves it to grade 1 as 0.0.
    up = sorted(g for g, c in p.items() if g > 0 and c != 0)
    if up:
        raise PoleError(
            "the function is unbounded at %r: grade %s along a direction, so it has "
            "no derivatives there" % (list(at), up[-1]))
    complete = v.complete_order() if callable(getattr(v, "complete_order", None)) else getattr(v, "_complete", None)
    if complete is not None and complete < need:
        raise IncompleteJetError(
            "order %d is needed at %r and the result is complete only to order %s; "
            "raise `terms=` on the transcendental that set the bound" % (need, list(at), complete))
    return p


def _lower_set(m, K):
    """Index tuples of length m with sum <= K, ordered by sum then lexicographic."""
    out = [t for t in _itertools_dir.product(range(K + 1), repeat=m) if sum(t) <= K]
    return sorted(out, key=lambda t: (sum(t), t))


def _no_pair_cancels(d):
    """No two components equal or opposite, so no x_i - x_j (at x_i0 = x_j0) and
    no x_i + x_j (at x_i0 = -x_j0) is wholly zero along d -- a cancellation
    there is an R1 residue on an ordinary function."""
    return all(d[i] != d[j] and d[i] != -d[j]
               for i in range(len(d)) for j in range(i + 1, len(d)))


@_functools_dir.lru_cache(maxsize=None)
def _coordinate_quantities(j, m):
    """m seed quantities for coordinate j >= 2: Chebyshev points in [-0.95, 0.95]
    rounded to ODD multiples of 1/2^(8+j).

    Each coordinate has its own denominator.  In lowest terms a quantity of
    coordinate j has denominator exactly 2^(8+j), so quantities of different
    coordinates are never equal or opposite, and none is +-1 (the first
    coordinate's).  One shared set made (1, q, q) and (1, q, -q) directions: y - z
    cancelled exactly wherever y0 = z0.  Coordinate 2 keeps the 1/1024 set of
    the 2D integrator (composite_lib._seed_quantities).  Distinct values within a
    coordinate keep the lower set unisolvent.
    """
    D = 2 ** (8 + j)
    out = []
    for k in range(m):
        x = 0.95 * math.cos(math.pi * (k + 0.5) / m)
        out.append(_Fraction_dir(round(x * D / 2) * 2 + (1 if x >= 0 else -1), D))
    if len(set(out)) != m:
        raise ValueError("_coordinate_quantities: %d points collide at 1/%d spacing" % (m, D))
    return tuple(out)


@_functools_dir.lru_cache(maxsize=None)
def _direction_set(n, K):
    """The directions for n variables to order K, as tuples of Fractions:
    (1, q2[i2], ..., qn[in]) over the lower set, each coordinate from its own
    quantity set (_coordinate_quantities)."""
    qs = [_coordinate_quantities(j, K + 1) for j in range(2, n + 1)]
    dirs = tuple((_Fraction_dir(1),) + tuple(q[i] for q, i in zip(qs, idx))
                 for idx in _lower_set(n - 1, K))
    assert all(_no_pair_cancels(d) for d in dirs)
    return dirs


@_functools_dir.lru_cache(maxsize=None)
def _check_direction(n):
    """(-1, e2, ..., en): the direction the jet is checked against.  First
    component -1, so it lies outside the half-space every _direction_set member
    is in; e_j = (5 * 2^(5+j) + 1) / 2^(8+j), about 0.63, a per-coordinate
    denominator as in _coordinate_quantities, so no two components are equal or
    opposite."""
    d = (_Fraction_dir(-1),) + tuple(_Fraction_dir(5 * 2 ** (5 + j) + 1, 2 ** (8 + j))
                                     for j in range(2, n + 1))
    assert _no_pair_cancels(d)
    return d


@_functools_dir.lru_cache(maxsize=None)
def _grade_weights(n, K, k):
    """For grade k: (monomials, weights).  T_a = sum_m weights[a][m] * grade_k(m),
    over the first C(k+n-1, n-1) directions, exact rationals."""
    dirs = _direction_set(n, K)
    mons = [(k - sum(b),) + b for b in _lower_set(n - 1, k) if sum(b) <= k]
    mons = [a for a in mons if a[0] >= 0]
    use = dirs[:len(mons)]
    # A[m][a] = c^a restricted to the chart c1 = 1: product over i >= 2 of c_i^a_i
    A = [[_prod_pow(d, a) for a in mons] for d in use]
    inv = _rational_inverse(A)                       # inv[a][m]
    return tuple(mons), tuple(tuple(float(x) for x in row) for row in inv)


def _prod_pow(d, a):
    out = _Fraction_dir(1)
    for c, e in zip(d[1:], a[1:]):
        out *= c ** e
    return out


def _rational_inverse(A):
    n = len(A)
    M = [list(row) + [_Fraction_dir(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        M[col] = [x / M[col][col] for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                M[r] = [x - M[r][col] * y for x, y in zip(M[r], M[col])]
    return [row[n:] for row in M]


def taylor_jets(f, at, order):
    """Every Taylor coefficient T_a of f at `at`, |a| <= order, from directional
    composites.  Returns ({a: T_a}, evaluations).  The evaluations are
    C(order+n-1, n-1) directions plus one check direction (NonSmoothError).

    f takes ordinary composites and uses composite_lib functions.  A float
    coercion inside f (math.*) is refused with ValueError.
    """
    from composite.composite_lib import (_seed_at, _derivative_scope,
                                         _refusing_float, _refusing_residue, FloatCoercionError, Composite)
    n = len(at)
    dirs = _direction_set(n, order)
    # The check direction is evaluated after the others have been read, so a
    # branch point or pole is named by _parts_of first: at sqrt(x), x = 0, the
    # check direction moves x negative and sqrt would refuse it with a domain
    # error that says nothing about why.
    scope = lambda: (_derivative_scope(order + 2, order + 2), _refusing_float(),
                     _refusing_residue(), _reading_within_bounds())
    try:
        with _contextlib_dir.ExitStack() as stack:
            for cm in scope():
                stack.enter_context(cm)
            vals = [f(*[_seed_at(x, float(c)) for x, c in zip(at, d)]) for d in dirs]
            parts = [_parts_of(v, at, order) for v in vals]
            check = _parts_of(f(*[_seed_at(x, float(c)) for x, c in zip(at, _check_direction(n))]),
                              at, order)
    except FloatCoercionError as e:
        raise ValueError("the function is not composite -- %s" % e) from None
    T = {}
    for k in range(order + 1):
        mons, W = _grade_weights(n, order, k)
        g = [p.get(-k, 0.0) for p in parts[:len(mons)]]
        for a, row in zip(mons, W):
            T[a] = sum(w * x for w, x in zip(row, g))
    # The check direction, predicted from the jet: grade -k along e is the
    # degree-k part of the Taylor polynomial at e.  Scale: the larger of the
    # actual value and the biggest single term of the prediction.
    e = [float(c) for c in _check_direction(n)]
    for k in range(order + 1):
        terms = [t * math.prod(q ** ai for q, ai in zip(e, a)) for a, t in T.items() if sum(a) == k]
        actual = check.get(-k, 0.0)
        scale = max([abs(actual)] + [abs(x) for x in terms])
        _refuse_non_smooth(_disagreement(actual, sum(terms), scale), k, at)
    return T, len(dirs) + 1


def _fact(a):
    out = 1
    for e in a:
        out *= math.factorial(e)
    return out


def partial_derivative(f, at: List[float], wrt: List[int]):
    """d^|wrt| f / dx1^wrt1 ... at `at`, from C(K+n-1, n-1) directional composites,
    K = sum(wrt)."""
    T, _ = taylor_jets(f, at, sum(wrt))
    return T[tuple(wrt)] * _fact(wrt)


def gradient_at(f, at: List[float]):
    """[df/dx1, ...] from n directional composites."""
    n = len(at)
    T, _ = taylor_jets(f, at, 1)
    return [T[tuple(int(i == j) for i in range(n))] for j in range(n)]


def hessian_at(f, at: List[float]):
    """[[d2f/dxi dxj]] from n(n+1)/2 directional composites."""
    n = len(at)
    T, _ = taylor_jets(f, at, 2)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a = [0] * n
            a[i] += 1
            a[j] += 1
            H[i][j] = T[tuple(a)] * _fact(a)
    return H


def jacobian_at(fs: List[Callable], at: List[float]):
    """[[dfi/dxj]]: one gradient per component, n directional composites each."""
    return [gradient_at(f, at) for f in fs]


@_functools_dir.lru_cache(maxsize=None)
def _orthonormal_exact(n):
    """Rows of a Householder reflection I - 2 u u^T / |u|^2, exact rationals: an
    orthonormal basis with NO zero entry, so every coordinate moves and none is
    a plain value -- a plain coordinate at 0 would be a written zero -- and no
    row with two entries equal or opposite (_no_pair_cancels).  u = (shift +
    step*i) is searched until both hold."""
    for step in range(1, 20):
        for shift in range(1, 50):
            u = [_Fraction_dir(shift + step * i) for i in range(n)]
            S = sum(x * x for x in u)
            H = [[(1 if i == j else 0) - 2 * u[i] * u[j] / S for j in range(n)] for i in range(n)]
            if all(x != 0 for row in H for x in row) and all(_no_pair_cancels(r) for r in H):
                return tuple(tuple(row) for row in H)
    raise ValueError("no Householder basis without zero or paired entries for n = %d" % n)


def _orthonormal_directions(n):
    """_orthonormal_exact as floats."""
    return tuple(tuple(float(x) for x in row) for row in _orthonormal_exact(n))


def laplacian_at(f, at: List[float]):
    """sum d2f/dxi2 from n directional composites: the trace of the Hessian is the
    sum of second directional derivatives along any orthonormal basis, and each
    of those is 2 * grade -2 of one composite.  One more composite, the first
    basis vector reversed, checks smoothness (NonSmoothError)."""
    from composite.composite_lib import (_seed_at, _derivative_scope,
                                         _refusing_float, _refusing_residue, FloatCoercionError, Composite)
    total = 0.0
    try:
        with _derivative_scope(4, 4), _refusing_float(), _refusing_residue(), _reading_within_bounds():
            for i, v in enumerate(_orthonormal_directions(len(at))):
                c = _parts_of(f(*[_seed_at(x, q) for x, q in zip(at, v)]), at, 2)
                total += 2.0 * c.get(-2, 0.0)
                if i == 0:      # smoothness: the opposite direction mirrors it (NonSmoothError)
                    _mirror_check(c, _parts_of(f(*[_seed_at(x, -q) for x, q in zip(at, v)]), at, 2), at, 2)
    except FloatCoercionError as e:
        raise ValueError("the function is not composite -- %s" % e) from None
    return total


def directional_derivative(f, at: List[float], direction: List[float]):
    """grad f . v_hat.  One composite along v_hat when every coordinate either moves
    or is nonzero; a coordinate that is 0 and does not move would enter as a
    written zero, so then it is the gradient (n composites) dotted with v_hat."""
    from composite.composite_lib import (_seed_at, _derivative_scope, R,
                                         _refusing_float, _refusing_residue, FloatCoercionError, Composite)
    norm = math.sqrt(sum(d * d for d in direction))
    v = [d / norm for d in direction]
    # Two cases go through the gradient (n composites) instead of one composite:
    # a coordinate that is 0 and does not move would be a written zero, and two
    # components equal or opposite make x_i -+ x_j wholly zero along v wherever
    # x_i0 = +-x_j0, an R1 residue on an ordinary function.
    moving = [d for d in direction if d != 0]
    if any(q == 0 and x == 0 for x, q in zip(at, v)) or not _no_pair_cancels(moving):
        return sum(g * q for g, q in zip(gradient_at(f, at), v))
    args = [_seed_at(x, q) if q != 0 else R(x) for x, q in zip(at, v)]
    back = [_seed_at(x, -q) if q != 0 else R(x) for x, q in zip(at, v)]
    try:
        with _derivative_scope(3, 3), _refusing_float(), _refusing_residue(), _reading_within_bounds():
            c = _parts_of(f(*args), at, 1)
            _mirror_check(c, _parts_of(f(*back), at, 1), at, 1)   # smoothness, one more composite
    except FloatCoercionError as e:
        raise ValueError("the function is not composite -- %s" % e) from None
    return c.get(-1, 0.0)


def multivar_limit(f, as_vars_to: List[float], rel_tol=1e-12):
    """lim f as the point is approached, read as the standard part along several
    directions.  If they disagree beyond rounding (rel_tol), the limit depends on
    the path and does not exist.  An unbounded result reads as +-inf."""
    from composite.composite_lib import (_seed_at, _derivative_scope, LimitDoesNotExistError,
                                         _refusing_float, _refusing_residue, FloatCoercionError, Composite)
    n = len(as_vars_to)
    dirs = _direction_set(n, 2) + _orthonormal_exact(n)
    vals = []
    try:
        with _derivative_scope(4, 4), _refusing_float(), _refusing_residue(), _reading_within_bounds():
            for d in dirs:
                c = f(*[_seed_at(x, float(q)) for x, q in zip(as_vars_to, d)])
                vals.append(c.to_ieee754() if isinstance(c, Composite) else float(c))
    except FloatCoercionError as e:
        raise ValueError("the function is not composite -- %s" % e) from None
    first = vals[0]
    for v in vals[1:]:
        same = (v == first) if (math.isinf(first) or math.isinf(v)) else \
               abs(v - first) <= rel_tol * max(1.0, abs(first))
        if not same:
            raise LimitDoesNotExistError(
                "the limit depends on the direction of approach: %r along one, %r along "
                "another" % (first, v))
    return first


def divergence_at(F: List[Callable], at: List[float]):
    """sum dFi/dxi: one gradient per component."""
    return sum(gradient_at(Fi, at)[i] for i, Fi in enumerate(F))


def curl_at(F: List[Callable], at: List[float]):
    """curl of a 3D field [Fx, Fy, Fz] at a point, from its Jacobian.

    A zero component is written as a plain 0 (lambda x, y, z: 0), not 0*x: a
    written zero times x is an R1 residue and is refused.
    """
    if len(at) != 3:
        raise ValueError("Curl requires 3D vector field")
    J = jacobian_at(F, at)               # J[i][j] = dFi/dxj
    return [J[2][1] - J[1][2], J[0][2] - J[2][0], J[1][0] - J[0][1]]


def double_integral(f, x_range, y_range, tol=1e-10):
    """Integral of f over the box x_range x y_range: composite_lib.integrate's
    2D path (the 2D meet on merged composites, separated from parts)."""
    from composite.composite_lib import integrate
    return integrate(f, x_range, y_range, tol=tol)
