"""Directional composites for partial derivatives (composite_multivar).

partial_derivative, gradient_at and hessian_at read every mixed
partial off ordinary one-variable composites evaluated along several directions
(docs/MC Replacement - Directional Composites (DRAFT).md).  The references are
mpmath.diff at 50 digits, the project's independent oracle.  Every check prints
got against known, pass or fail; tolerances are fixed.

The evaluation counts are the method's own: n directions for a gradient,
n(n+1)/2 for a Hessian, C(K+n-1, n-1) for order K.
"""
import math

import pytest

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 50

from composite.composite_lib import R, cos, exp, ln, sin, sqrt
from composite.composite_lib import LimitDoesNotExistError, ResidueError
from composite.composite_multivar import (_direction_set, _orthonormal_directions,
                                          _orthonormal_exact, curl_at, divergence_at,
                                          directional_derivative, gradient_at,
                                          hessian_at, jacobian_at,
                                          laplacian_at, multivar_limit,
                                          partial_derivative, taylor_jets)

TOL = 1e-13          # relative, against 50-digit references; measured worst 3.1e-14 (order-4 coefficients)

CASES = [
    # name, composite f, mpmath f, point, a third-order partial
    ("x^2 y", lambda x, y: x * x * y, lambda x, y: x * x * y, [3.0, 2.0], [2, 1]),
    ("exp(xy) sin z + x y^2 z^3",
     lambda x, y, z: exp(x * y) * sin(z) + x * y * y * z * z * z,
     lambda x, y, z: mpmath.exp(x * y) * mpmath.sin(z) + x * y ** 2 * z ** 3,
     [0.5, 1.2, 0.7], [1, 1, 1]),
    ("1/(1+x^2+y^2)", lambda x, y: R(1) / (R(1) + x * x + y * y),
     lambda x, y: 1 / (1 + x ** 2 + y ** 2), [0.3, -0.4], [2, 1]),
    ("sqrt(x^2+y^2+z^2)", lambda x, y, z: sqrt(x * x + y * y + z * z),
     lambda x, y, z: mpmath.sqrt(x ** 2 + y ** 2 + z ** 2), [1.0, 2.0, 2.0], [1, 1, 1]),
    ("ln(1+xy) + cos(x-y)", lambda x, y: ln(R(1) + x * y) + cos(x - y),
     lambda x, y: mpmath.log(1 + x * y) + mpmath.cos(x - y), [0.7, 0.2], [2, 1]),
    ("exp(x+y) sin(zw)", lambda x, y, z, w: exp(x + y) * sin(z * w),
     lambda x, y, z, w: mpmath.exp(x + y) * mpmath.sin(z * w), [0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1]),
]
IDS = [c[0] for c in CASES]


def ref(fm, at, orders):
    return float(mpmath.diff(fm, [mpmath.mpf(a) for a in at], tuple(orders)))


def check(label, got, want):
    err = abs(got - want) / max(1.0, abs(want))
    print(f"\n  {label:48s} got {got!r:24} known {want!r:24} rel err {err:.1e}")
    assert err <= TOL


@pytest.mark.parametrize("name,fc,fm,at,wrt", CASES, ids=IDS)
def test_gradient(name, fc, fm, at, wrt):
    got = gradient_at(fc, at)
    n = len(at)
    for i in range(n):
        check(f"{name} d/dx{i}", got[i], ref(fm, at, [int(j == i) for j in range(n)]))


@pytest.mark.parametrize("name,fc,fm,at,wrt", CASES, ids=IDS)
def test_hessian(name, fc, fm, at, wrt):
    got = hessian_at(fc, at)
    n = len(at)
    for i in range(n):
        for j in range(i, n):
            o = [0] * n
            o[i] += 1
            o[j] += 1
            check(f"{name} d2/dx{i}dx{j}", got[i][j], ref(fm, at, o))
            assert got[i][j] == got[j][i]


@pytest.mark.parametrize("name,fc,fm,at,wrt", CASES, ids=IDS)
def test_third_order_partial(name, fc, fm, at, wrt):
    check(f"{name} d^3 {wrt}", partial_derivative(fc, at, wrt), ref(fm, at, wrt))


def test_every_coefficient_to_order_4_in_three_variables():
    """35 Taylor coefficients from 15 directions, plus one smoothness check
    direction; T_a = d^a f / a!."""
    fc, fm, at = CASES[1][1], CASES[1][2], CASES[1][3]
    T, evals = taylor_jets(fc, at, 4)
    worst = 0.0
    for a, t in T.items():
        want = ref(fm, at, list(a)) / math.prod(math.factorial(e) for e in a)
        worst = max(worst, abs(t - want) / max(1.0, abs(want)))
    print(f"\n  order 4, 3 variables: {len(T)} coefficients from {evals} composites (15 + 1 check), worst rel err {worst:.1e}")
    assert len(T) == 35 and evals == 16
    assert worst <= TOL


@pytest.mark.parametrize("n,K,count", [(2, 1, 2), (3, 1, 3), (3, 2, 6), (4, 2, 10), (3, 4, 15)])
def test_direction_counts(n, K, count):
    dirs = _direction_set(n, K)
    print(f"\n  n={n} K={K}: {len(dirs)} directions, want C(K+n-1, n-1) = {count}")
    assert len(dirs) == count == math.comb(K + n - 1, n - 1)
    assert all(c != 0 for d in dirs for c in d)            # never a written zero


def test_separation_has_no_residue():
    """u*v has exactly four nonzero coefficients.  Summing the directional
    composites as composites, a wholly zero partial sum deposits a residue and
    writes spurious terms; the weights act on parts, so none appear."""
    T, _ = taylor_jets(lambda u, v: u * v, [1.0, 1.0], 4)
    nz = {a: t for a, t in T.items() if abs(t) > 1e-13}
    print(f"\n  u*v at (1,1): {nz}  known {{(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}}")
    assert set(nz) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert all(abs(t - 1.0) <= 1e-13 for t in nz.values())


def test_a_float_function_is_refused():
    with pytest.raises(ValueError, match="not composite"):
        gradient_at(lambda x, y: math.exp(x * y), [0.5, 1.0])


# --- step 2: Jacobian, Laplacian, directional derivative, limits ---------------------

def test_jacobian():
    s2 = math.sqrt(2) / 2
    for label, fs, at, want in (
            ("J[x^2+y, xy^2] at (1,2)", [lambda x, y: x * x + y, lambda x, y: x * y * y], [1, 2], [[2, 1], [4, 4]]),
            ("J[r cos t, r sin t] at (2,pi/4)", [lambda r, t: r * cos(t), lambda r, t: r * sin(t)],
             [2, math.pi / 4], [[s2, -2 * s2], [s2, 2 * s2]])):
        got = jacobian_at(fs, at)
        for i, row in enumerate(want):
            for j, w in enumerate(row):
                check(f"{label} [{i}][{j}]", got[i][j], w)


def test_orthonormal_directions_have_no_zero_entry():
    """A plain coordinate at 0 would be a written zero, so every direction moves
    every coordinate."""
    for n in (2, 3, 4, 5):
        H = _orthonormal_directions(n)
        orth = max(abs(sum(a * b for a, b in zip(H[i], H[j])) - (i == j)) for i in range(n) for j in range(n))
        print(f"\n  n={n}: max |H H^T - I| = {orth:.1e}, smallest |entry| = {min(abs(x) for r in H for x in r):.3f}")
        assert orth <= 1e-15 and all(x != 0 for r in H for x in r)


LAPLACIANS = [
    ("x^2+y^2 at (1,1)", lambda x, y: x * x + y * y, [1, 1], 4.0),
    ("x^3 y + x y^3 at (2,3)", lambda x, y: x ** 3 * y + x * y ** 3, [2, 3], 72.0),
    ("x^2+y^2+z^2 at the origin", lambda x, y, z: x * x + y * y + z * z, [0, 0, 0], 6.0),
    ("exp(x) sin(y) z at (1,2,3), harmonic", lambda x, y, z: exp(x) * sin(y) * z, [1, 2, 3], 0.0),
    # MC took 38.7 s for this one (sparse-dense, 2026-10-06); n composites here
    ("1/r at (1,2,2), harmonic", lambda x, y, z: R(1) / sqrt(x * x + y * y + z * z), [1, 2, 2], 0.0),
]


@pytest.mark.parametrize("label,f,at,want", LAPLACIANS, ids=[c[0] for c in LAPLACIANS])
def test_laplacian(label, f, at, want):
    check(f"laplacian {label}", laplacian_at(f, at), want)


def test_directional_derivative():
    check("d/dv exp(xy) at (1,2), v = (3,4)",
          directional_derivative(lambda x, y: exp(x * y), [1, 2], [3, 4]), (2 * 0.6 + 1 * 0.8) * math.exp(2))
    # a coordinate that is 0 and does not move: falls back to the gradient
    check("d/dv x*y at (0,2), v = (0,1)", directional_derivative(lambda x, y: x * y, [0, 2], [0, 1]), 0.0)


def test_limits():
    check("lim (x^2+y^2)/(x^2+y^2) at (0,0)",
          multivar_limit(lambda x, y: (x ** 2 + y ** 2) / (x ** 2 + y ** 2), [0, 0]), 1.0)
    check("lim (x^2 y - 2x^2)/(y - 2) at (1,2)",
          multivar_limit(lambda x, y: (x ** 2 * y - 2 * x ** 2) / (y - 2), [1, 2]), 1.0)
    check("lim sin(x^2+y^2)/(x^2+y^2) at (0,0)",
          multivar_limit(lambda x, y: sin(x * x + y * y) / (x * x + y * y), [0, 0]), 1.0)


def test_a_path_dependent_limit_is_refused():
    """xy/(x^2+y^2) at the origin is 1/2 along y = x and 0 along an axis: no limit.
    MC's multivar_limit returned 0.0 here."""
    with pytest.raises(LimitDoesNotExistError):
        multivar_limit(lambda x, y: x * y / (x * x + y * y), [0, 0])


# --- directions with two components equal or opposite ------------------------------
# A single shared quantity set gave 3-variable directions (1, q, q) and (1, q, -q):
# y - z was wholly zero along them wherever y0 = z0, and y + z wherever y0 = -z0.
# Before the residue refusal that read silently wrong (gradient of y - z at (1,2,2)
# as [0, 2.49, 0.49]); after it, ordinary functions were refused.  Each coordinate
# now has its own quantity set.

def _pairs_ok(d):
    return all(d[i] != d[j] and d[i] != -d[j] for i in range(len(d)) for j in range(i + 1, len(d)))


def test_no_direction_has_equal_or_opposite_components():
    for n in (2, 3, 4, 5):
        for K in (1, 2, 3, 4):
            dirs = _direction_set(n, K)
            bad = [d for d in dirs if not _pairs_ok(d)]
            print(f"\n  _direction_set n={n} K={K}: {len(bad)} of {len(dirs)} with equal/opposite components")
            assert not bad
    for n in range(2, 9):
        bad = [r for r in _orthonormal_exact(n) if not _pairs_ok(r)]
        print(f"\n  Householder n={n}: {len(bad)} of {n} rows with equal/opposite entries")
        assert not bad


PAIRED = [
    ("gradient y - z at (1,2,2)", lambda: gradient_at(lambda x, y, z: y - z, [1, 2, 2]), [0, 1, -1]),
    ("gradient x*(y - z) at (1,2,2)", lambda: gradient_at(lambda x, y, z: x * (y - z), [1, 2, 2]), [0, 1, -1]),
    ("gradient sin(y+z) at (1,0,0)", lambda: gradient_at(lambda x, y, z: sin(y + z), [1, 0, 0]), [0, 1, 1]),
    ("hessian (y-z)^2 at (1,2,2)", lambda: sum(hessian_at(lambda x, y, z: (y - z) * (y - z), [1, 2, 2]), []),
     [0, 0, 0, 0, 2, -2, 0, -2, 2]),
    ("curl [y-z, z-x, x-y] at (1,1,1)",
     lambda: curl_at([lambda x, y, z: y - z, lambda x, y, z: z - x, lambda x, y, z: x - y], [1, 1, 1]), [-2, -2, -2]),
    ("divergence [x, y-z, z] at (0,2,2)",
     lambda: [divergence_at([lambda x, y, z: x, lambda x, y, z: y - z, lambda x, y, z: z], [0, 2, 2])], [3]),
    ("limit (y-z)/(y+z) at (1,1,1)", lambda: [multivar_limit(lambda x, y, z: (y - z) / (y + z), [1, 1, 1])], [0]),
    ("laplacian sum (xi-xj)^2, 4 vars at (1,1,1,1)",
     lambda: [laplacian_at(lambda a, b, c, d: (a - b) * (a - b) + (a - c) * (a - c) + (a - d) * (a - d)
                           + (b - c) * (b - c) + (b - d) * (b - d) + (c - d) * (c - d), [1, 1, 1, 1])], [24]),
    ("dir deriv x - y at (2,2) along (1,1)", lambda: [directional_derivative(lambda x, y: x - y, [2, 2], [1, 1])], [0]),
    ("dir deriv x + y at (0,0) along (1,-1)", lambda: [directional_derivative(lambda x, y: x + y, [0, 0], [1, -1])], [0]),
    ("dir deriv y*z at (1,2,2) along (1,1,1)",
     lambda: [directional_derivative(lambda x, y, z: y * z, [1, 2, 2], [1, 1, 1])], [4 / math.sqrt(3)]),
]


@pytest.mark.parametrize("label,run,want", PAIRED, ids=[c[0] for c in PAIRED])
def test_equal_or_opposite_coordinates(label, run, want):
    got = run()
    for i, (g, w) in enumerate(zip(got, want)):
        check(f"{label} [{i}]", g, float(w))


# --- poles and removable singularities ----------------------------------------------
# At a pole each directional composite has an infinite part (grade > 0) and its finite
# grades are direction-dependent Laurent coefficients; separating them read
# x*y/(x+y) at (1, -1) as gradient [-0.83, 1.83] and Hessian 0.  Refused now.  A
# removable singularity has no infinite part and is read normally (JAX gives nan
# for both cases below, measured 2026-10-07).

from composite.composite_lib import R as _R
from composite.composite_multivar import PoleError

POLES = [
    ("gradient", lambda: gradient_at(lambda x, y: x * y / (x + y), [1, -1])),
    ("hessian", lambda: hessian_at(lambda x, y: x * y / (x + y), [1, -1])),
    ("partial d2/dxdy", lambda: partial_derivative(lambda x, y: x * y / (x + y), [1, -1], [1, 1])),
    ("laplacian", lambda: laplacian_at(lambda x, y, z: _R(1) / (x * x + y * y + z * z), [0, 0, 0])),
    ("directional", lambda: directional_derivative(lambda x, y: _R(1) / (x - 2 * y), [2, 1], [3, 4])),
]


@pytest.mark.parametrize("label,run", POLES, ids=[c[0] for c in POLES])
def test_a_pole_is_refused(label, run):
    with pytest.raises(PoleError) as e:
        run()
    print(f"\n  {label} at a pole: {type(e.value).__name__}: {str(e.value)[:70]}")


def test_an_unbounded_limit_still_reads_as_infinity():
    got = multivar_limit(lambda x, y: _R(1) / (x * x + y * y), [0, 0])
    print(f"\n  lim 1/(x^2+y^2) at (0,0): got {got!r}  known inf")
    assert got == math.inf


def test_removable_singularities():
    g = gradient_at(lambda x, y: (exp(x + 2 * y) - _R(1)) / (x + 2 * y), [0.0, 0.0])
    H = hessian_at(lambda x, y: (exp(x + 2 * y) - _R(1)) / (x + 2 * y), [0.0, 0.0])
    for lab, got, want in (("d/dx", g[0], 0.5), ("d/dy", g[1], 1.0), ("d2/dx2", H[0][0], 1 / 3),
                           ("d2/dxdy", H[0][1], 2 / 3), ("d2/dy2", H[1][1], 4 / 3)):
        check(f"(e^(x+2y)-1)/(x+2y) at 0 {lab}", got, want)
    H2 = hessian_at(lambda x, y: sin(x * x + y * y) / (x * x + y * y), [0.0, 0.0])
    for i in range(2):
        for j in range(2):
            check(f"sin(r^2)/r^2 at 0 H[{i}][{j}] (= 1 - r^4/6)", H2[i][j], 0.0)


# --- branch points and non-smooth points ---------------------------------------------
# Both used to read silently wrong (2026-10-07): sqrt(x)*y at (0,1) as gradient
# [0, 0], the cone sqrt(x^2+y^2) at the origin as [1.21, 0].  A branch point puts
# content at a non-integer grade, which the separation never reads; a cone or kink
# has integer grades whose dependence on the direction is not a polynomial, and with
# exactly as many directions as unknowns the fit always succeeds.  One check
# direction (first component -1) must be predicted by the jet; measured disagreement
# is <= 7e-10 of the grade for smooth functions and 1.5 to 2 for cones and kinks.

from composite.composite_multivar import BranchPointError, NonSmoothError

NON_DIFFERENTIABLE = [
    ("branch sqrt(x)*y at (0,1) gradient", lambda: gradient_at(lambda x, y: sqrt(x) * y, [0, 1]), BranchPointError),
    ("branch sqrt(x+y) at (1,-1) hessian", lambda: hessian_at(lambda x, y: sqrt(x + y), [1, -1]), BranchPointError),
    ("cone sqrt(x^2+y^2) at 0 gradient", lambda: gradient_at(lambda x, y: sqrt(x * x + y * y), [0, 0]), NonSmoothError),
    ("cone 3D at 0 hessian", lambda: hessian_at(lambda x, y, z: sqrt(x * x + y * y + z * z), [0, 0, 0]), NonSmoothError),
    ("kink sqrt((x-y)^2) at (1,1) gradient", lambda: gradient_at(lambda x, y: sqrt((x - y) * (x - y)), [1, 1]), NonSmoothError),
    ("kink sqrt(x^2)*y at (0,1) partial d/dx",
     lambda: partial_derivative(lambda x, y: sqrt(x * x) * y, [0, 1], [1, 0]), NonSmoothError),
    ("cone 3D at 0 laplacian", lambda: laplacian_at(lambda x, y, z: sqrt(x * x + y * y + z * z), [0, 0, 0]), NonSmoothError),
    ("cone at 0 directional along (3,4)",
     lambda: directional_derivative(lambda x, y: sqrt(x * x + y * y), [0, 0], [3, 4]), NonSmoothError),
]


@pytest.mark.parametrize("label,run,exc", NON_DIFFERENTIABLE, ids=[c[0] for c in NON_DIFFERENTIABLE])
def test_not_differentiable_is_refused(label, run, exc):
    with pytest.raises(exc) as e:
        run()
    print(f"\n  {label}: {type(e.value).__name__}: {str(e.value)[:80]}")


def test_smooth_neighbours_still_read():
    """Next to the refused points the same functions are smooth and read exactly."""
    g = gradient_at(lambda x, y: sqrt(x * x + y * y), [3, 4])
    check("cone at (3,4) d/dx", g[0], 0.6)
    check("cone at (3,4) d/dy", g[1], 0.8)
    g = gradient_at(lambda x, y: sqrt(x) * y, [4, 1])
    check("sqrt(x)*y at (4,1) d/dx", g[0], 0.25)
    check("sqrt(x)*y at (4,1) d/dy", g[1], 2.0)
    check("cone at (3,4) along (1,2)",
          directional_derivative(lambda x, y: sqrt(x * x + y * y), [3, 4], [1, 2]), (0.6 + 1.6) / math.sqrt(5))


def test_high_order_beyond_float_separation_is_refused():
    """The full directional set at order 20 in two variables: d^20/dx^20 of
    exp(xy) came out 13% off with the check at 4.7e-8; the tolerance is 1e-8, so
    taylor_jets refuses.  Order 12 still reads (7.8e-10)."""
    T, _ = taylor_jets(lambda x, y: exp(x * y), [0.5, 0.5], 12)
    want = 0.5 ** 12 * math.exp(0.25) / math.factorial(12)
    err = abs(T[(12, 0)] - want) / want
    print(f"\n  taylor_jets order 12, T(12,0): rel err {err:.1e}")
    assert err <= 1e-8
    with pytest.raises(NonSmoothError) as e:
        taylor_jets(lambda x, y: exp(x * y), [0.5, 0.5], 20)
    print(f"  taylor_jets order 20: {type(e.value).__name__}: {str(e.value)[:90]}")


# --- partial_derivative reads the segment in the variables it involves -----------------
# Only those move; the rest are fixed at y0 + c_y h**(K+1), an infinitesimal below every
# grade read.  The check direction carries the fixed coordinates at a different c, so a
# function that pulls c into the read grades (y/x at the origin) is refused.


SEGMENT_EXACT = [
    # label, f, at, wrt, known, composites
    ("d/dx x*y at (2,0)  [fixed 0 is not a written zero]", lambda x, y: x * y, [2, 0], [1, 0], 0.0, 2),
    ("d/dx x*(y-z) at (1,2,2)  [equal fixed coordinates]", lambda x, y, z: x * (y - z), [1, 2, 2], [1, 0, 0], 0.0, 2),
    ("d/dx x*(y+z) at (1,2,-2)  [opposite fixed coordinates]", lambda x, y, z: x * (y + z), [1, 2, -2], [1, 0, 0], 0.0, 2),
    ("d/dz sin(x)cos(y)z at the origin", lambda x, y, z: sin(x) * cos(y) * z, [0, 0, 0], [0, 0, 1], 0.0, 2),
    ("d/dx x*sin(y)/y at (1,0)  [0/0 in a fixed coordinate]", lambda x, y: x * sin(y) / y, [1, 0], [1, 0], 1.0, 2),
    ("d2/dx2 e^x (e^y-1)/y at (0,0)", lambda x, y: exp(x) * (exp(y) - _R(1)) / y, [0, 0], [2, 0], 1.0, 2),
    ("d/dy (x*y)/x at (0,2)", lambda x, y: (x * y) / x, [0, 2], [0, 1], 1.0, 2),
    ("d/dx y^2/x at (0,0)  [0 along y = 0; not jointly smooth]", lambda x, y: y * y / x, [0, 0], [1, 0], 0.0, 2),
    ("d20/dx20 exp(xy) at (.5,.5)", lambda x, y: exp(x * y), [0.5, 0.5], [20, 0], 0.5 ** 20 * math.exp(0.25), 2),
    ("d28/dx28 exp(xy) at (.5,.5)", lambda x, y: exp(x * y), [0.5, 0.5], [28, 0], 0.5 ** 28 * math.exp(0.25), 2),
]


@pytest.mark.parametrize("label,f,at,wrt,want,count", SEGMENT_EXACT, ids=[c[0] for c in SEGMENT_EXACT])
def test_segment_partials(label, f, at, wrt, want, count):
    from composite.composite_multivar import _taylor_segment
    moving = [i for i, a in enumerate(wrt) if a]
    _, n = _taylor_segment(f, at, moving, sum(wrt))
    check(f"{label} ({n} composites)", partial_derivative(f, at, wrt), want)
    assert n == count


def test_segment_mixed_partials_in_few_of_many_variables():
    fc = lambda x, y, z, w: exp(x * y) * sin(z + 2 * w)
    fm = lambda x, y, z, w: mpmath.exp(x * y) * mpmath.sin(z + 2 * w)
    at = [0.3, 0.7, 1.1, 0.5]
    for wrt in ([2, 2, 0, 0], [4, 4, 0, 0], [1, 0, 1, 0], [0, 3, 0, 3]):
        check(f"d{wrt} exp(xy)sin(z+2w)", partial_derivative(fc, at, wrt), ref(fm, at, wrt))


SEGMENT_REFUSED = [
    ("d/dx y/x at (0,0)  [c would leak into the read grades]", lambda x, y: y / x, [0, 0], [1, 0], NonSmoothError),
    ("d/dx sin(y)/x at (0,0)", lambda x, y: sin(y) / x, [0, 0], [1, 0], NonSmoothError),
    ("d/dx x*y/(x^2+y^2) at (0,0)", lambda x, y: x * y / (x * x + y * y), [0, 0], [1, 0], NonSmoothError),
    ("d3/dx3 y/x^3 at (0,0)", lambda x, y: y / (x * x * x), [0, 0], [3, 0], NonSmoothError),
    ("d/dx x/y at (1,0)  [pole in a fixed coordinate]", lambda x, y: x / y, [1, 0], [1, 0], PoleError),
    ("d/dx exp(xy)-exp(xy)  [residue]", lambda x, y: exp(x * y) - exp(x * y), [1, 1], [1, 0], ResidueError),
    ("d/dx x*y + 0  [written zero]", lambda x, y: x * y + 0, [1, 2], [1, 0], ResidueError),
    ("d/dx sqrt(x^2+y^2) at 0  [cone]", lambda x, y: sqrt(x * x + y * y), [0, 0], [1, 0], NonSmoothError),
    ("d/dx sqrt(x)*y at (0,1)  [branch]", lambda x, y: sqrt(x) * y, [0, 1], [1, 0], BranchPointError),
]


@pytest.mark.parametrize("label,f,at,wrt,exc", SEGMENT_REFUSED, ids=[c[0] for c in SEGMENT_REFUSED])
def test_segment_refusals(label, f, at, wrt, exc):
    with pytest.raises(exc) as e:
        partial_derivative(f, at, wrt)
    print(f"\n  {label}: {type(e.value).__name__}: {str(e.value)[:80]}")
    if exc is PoleError:
        assert "grade" not in str(e.value)      # the fixed coordinate's grade is not the pole's


# --- limits along curves, not only lines ------------------------------------------------
# x^2 y / (x^4 + y^2) at the origin is 0 along every straight line and 1/2 along
# y = x^2; the line-only multivar_limit returned 0.0 for it.  Paths are now
# x_i = q_i h^(w_i) for weight vectors over {1, 2, 3}, so the curves y ~ x^2, x^3,
# sqrt(x), ... are tried as well.

LIMITS_THAT_EXIST = [
    ("x^2 y / (x^2+y^2)", lambda x, y: x * x * y / (x * x + y * y), [0, 0], 0.0),
    ("(x^3+y^3) / (x^2+y^2)", lambda x, y: (x ** 3 + y ** 3) / (x * x + y * y), [0, 0], 0.0),
    ("sin(x^2+y^2) / (x^2+y^2)", lambda x, y: sin(x * x + y * y) / (x * x + y * y), [0, 0], 1.0),
    ("(1 - cos(x^2+y^2)) / (x^2+y^2)^2", lambda x, y: (R(1) - cos(x * x + y * y)) / ((x * x + y * y) ** 2), [0, 0], 0.5),
    ("(x^2 y - 2x^2)/(y - 2) at (1,2)", lambda x, y: (x * x * y - 2 * x * x) / (y - 2), [1, 2], 1.0),
    ("x y z / (x^2+y^2+z^2)", lambda x, y, z: x * y * z / (x * x + y * y + z * z), [0, 0, 0], 0.0),
    ("sin(u^2+v^2)/(u^2+v^2), u = x-1, v = y-2, at (1,2)",
     lambda x, y: sin((x - 1) ** 2 + (y - 2) ** 2) / ((x - 1) ** 2 + (y - 2) ** 2), [1, 2], 1.0),
]


@pytest.mark.parametrize("label,f,at,want", LIMITS_THAT_EXIST, ids=[c[0] for c in LIMITS_THAT_EXIST])
def test_limits_that_exist(label, f, at, want):
    check(f"lim {label}", multivar_limit(f, at), want)


LIMITS_THAT_DO_NOT = [
    ("x y / (x^2+y^2)  [lines disagree]", lambda x, y: x * y / (x * x + y * y), [0, 0]),
    ("x^2 y / (x^4+y^2)  [0 on lines, 1/2 on y = x^2]", lambda x, y: x * x * y / (x ** 4 + y * y), [0, 0]),
    ("x y^2 / (x^2+y^4)  [0 on lines, 1/2 on x = y^2]", lambda x, y: x * y * y / (x * x + y ** 4), [0, 0]),
    ("x^3 y / (x^6+y^2)  [0 on lines, 1/2 on y = x^3]", lambda x, y: x ** 3 * y / (x ** 6 + y * y), [0, 0]),
    ("x^2 y / (x^4+y^2) + z, 3 variables", lambda x, y, z: x * x * y / (x ** 4 + y * y) + z, [0, 0, 0]),
    ("u^2 v / (u^4+v^2), u = x-1, v = y-2, at (1,2)  [curve off the origin]",
     lambda x, y: (x - 1) ** 2 * (y - 2) / ((x - 1) ** 4 + (y - 2) ** 2), [1, 2]),
]


@pytest.mark.parametrize("label,f,at", LIMITS_THAT_DO_NOT, ids=[c[0] for c in LIMITS_THAT_DO_NOT])
def test_path_dependent_limits_are_refused(label, f, at):
    with pytest.raises(LimitDoesNotExistError) as e:
        multivar_limit(f, at)
    print(f"\n  {label}: {str(e.value)[:150]}")
