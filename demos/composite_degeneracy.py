#!/usr/bin/env python3
"""Degeneracy by arithmetic: four classical questions, one reading.

A negative grade is an order of vanishing.  That one fact turns a set of
unrelated-looking problems, each of which classically needs its own algorithm
and its own tolerance, into a subtraction followed by lead_order():

  1. RANK DEFICIENCY.  det(A + hB) vanishes to the order of the corank, so the
     grade of one determinant is the rank deficiency.  Classically: SVD, then a
     threshold deciding which singular values count as zero.
  2. ROOT MULTIPLICITY.  p(a + h) vanishes to the order of the multiplicity at
     a, given nothing but the expanded coefficients.  Classically: repeated GCD
     with the derivative, or root clustering with a tolerance.
  3. ORDER OF CONTACT.  (f - g)(a + h) vanishes to the order of contact between
     the two curves.  Classically: match derivatives one at a time and decide
     when one is close enough to zero.
  4. VERTEX OF A CURVE.  A vertex is where the osculating circle gains a degree
     of contact, so it shows up as the contact order going from 3 to 4.
     Classically: form the curvature, differentiate it, solve dk/ds = 0.

The answers are integers, so there is no threshold anywhere and no borderline
case to argue about.  And the leading COEFFICIENT carries the rest of the
answer: the first non-vanishing derivative for a root, the null-space projector
for a matrix.  The grade says how degenerate, the coefficient says in what way.

Section 5 shows the division of labour directly.  The inverse of a singular
matrix has pole order 1 no matter how deficient it is -- the grade saturates,
because a scalar order cannot hold a subspace -- and the corank reappears as
the RANK of the coefficient sitting at that grade.

WHAT THIS DOES NOT DO.  It reads the order of an exactly degenerate object.
Perturb the matrix entries by float noise and the corank is 0; perturb the
ellipse and it has no exact vertex.  Both answers are correct, and they are
why the comparison here is against symbolic computation rather than against
numerical linear algebra.  What the composite adds is that the order falls out
of ordinary arithmetic on the entries, with no algorithm named after the
question being asked.

One rule matters throughout: a coefficient that is not there is absent,
Composite({}), and never R(0).  A written zero is an expressed zero, so an
infinitesimal, and it converts -- which shifts the very grade being measured.
Section 2 shows the failure: the polynomial builder that writes its zero
coefficients reports multiplicity 1 for a root of multiplicity 5.
"""
import math
import random
import warnings

import numpy as np

import composite.composite_lib as cl
from composite.composite_lib import R, ZERO, Composite, sqrt, sin, tan, exp, ln

warnings.filterwarnings("ignore")        # R1 conversions are expected where shown

NOTHING = Composite({})
_results = []


def check(label, got, want, tol=0):
    """Integer answers are compared exactly; tol is for the coefficients."""
    err = abs(got - want)
    ok = err <= tol
    _results.append(ok)
    print(f"  {'OK ' if ok else 'BAD'} {label:44s} got {got: .12g}  want {want: .12g}  "
          f"err {err:.1e}  tol {tol:.0e}")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def order_and_coeff(value):
    """lead_order() plus the coefficient sitting at the leading grade."""
    v = cl._ensure_composite(value)
    ld = v.lead_dim()
    return v.lead_order(), (v.coeffs_dict().get(ld, 0.0) if ld is not None else 0.0)


def vanishing_order(f, at, depth=12):
    """Order of vanishing of f at `at`, from ONE evaluation."""
    with cl._derivative_scope(depth, depth + 8):
        return order_and_coeff(f(cl._seeded(at)))


# ---------------------------------------------------------------------------
# 1. Rank deficiency from the grade of a determinant
# ---------------------------------------------------------------------------
def det(M):
    """Cofactor expansion.  Composite arithmetic only, no pivoting, no tolerance."""
    n = len(M)
    if n == 1:
        return M[0][0]
    total = NOTHING
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = M[0][j] * det(minor)
        if j % 2:
            term = term * R(-1.0)
        total = total + term
    return total


def perturbed(A, B):
    """A + h*B as a matrix of composites."""
    n = len(A)
    return [[R(float(A[i][j])) + R(float(B[i][j])) * ZERO for j in range(n)]
            for i in range(n)]


def matrix_of_rank(n, rank, rng):
    """n x n integer matrix of exactly `rank`, as a sum of rank-1 outer products."""
    A = np.zeros((n, n))
    while int(np.linalg.matrix_rank(A)) != rank:
        A = sum(np.outer([rng.randint(-4, 4) for _ in range(n)],
                         [rng.randint(-4, 4) for _ in range(n)]) * 1.0
                for _ in range(rank))
    return A


def det_order(A, rng, directions=5):
    """Order of vanishing of det(A + hB), minimised over directions B.

    The order is the corank for a GENERIC B, and a particular B can be
    non-generic and report higher: at corank 1 the h coefficient is v.B.u for
    the left and right null vectors, and small random integers hit v.B.u == 0
    often enough to see it.  The generic order is the MINIMUM over directions,
    so probe a few.  This is not peeking at the answer -- the minimum is the
    definition of the generic value.
    """
    n = len(A)
    best = None
    for _ in range(directions):
        B = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        with cl._derivative_scope(n + 2, n + 6):
            order, _ = order_and_coeff(det(perturbed(A, B)))
        order = 0.0 if order is None else order
        best = order if best is None else min(best, order)
    return best


def demo_rank():
    section("1. RANK DEFICIENCY -- the grade of det(A + hB) is the corank")
    print("  A is exactly rank-deficient, B is a generic integer direction.")
    print("  det is one cofactor expansion in composite arithmetic.  No SVD,")
    print("  no singular values, no threshold deciding which are zero.")
    print("  The order is minimised over a few directions, because the generic")
    print("  order is the minimum and one direction can be non-generic.\n")
    rng = random.Random(7)
    for n in (3, 4, 5):
        for rank in range(1, n + 1):
            A = matrix_of_rank(n, rank, rng)
            corank = n - int(np.linalg.matrix_rank(A))
            check(f"n={n} rank={rank}: order of vanishing",
                  det_order(A, rng), corank)


# ---------------------------------------------------------------------------
# 2. Root multiplicity from the grade of p(a + h)
# ---------------------------------------------------------------------------
def poly_from_coeffs(coeffs, skip_zeros=True):
    """Horner-free build from an expanded coefficient list, low order first.

    `skip_zeros=False` is the bug this section demonstrates, kept deliberately.
    """
    def f(x):
        acc = NOTHING
        for k, c in enumerate(coeffs):
            if c == 0 and skip_zeros:
                continue                  # ABSENT.  R(0) here is an infinitesimal.
            acc = acc + R(float(c)) * x ** k
        return acc
    return f


def expanded_coeffs(root, m, other):
    """Coefficients of (x - root)^m * (x - other), low order first."""
    poly = [1.0]
    for _ in range(m):
        poly = np.convolve(poly, [1.0, -root]).tolist()
    poly = np.convolve(poly, [1.0, -other]).tolist()
    return list(reversed(poly))


def demo_multiplicity():
    section("2. ROOT MULTIPLICITY -- the grade of p(a + h) is the multiplicity")
    print("  p(x) = (x-2)^m (x+1), handed over as EXPANDED coefficients only, so")
    print("  no factored structure is visible.  The leading coefficient that")
    print("  comes back is p^(m)(2)/m!, which is 3 for every m here.\n")
    for m in range(1, 8):
        coeffs = expanded_coeffs(2.0, m, -1.0)
        order, lead = vanishing_order(poly_from_coeffs(coeffs), 2.0)
        check(f"(x-2)^{m}(x+1): order of vanishing", order or 0.0, m)
        check(f"(x-2)^{m}(x+1): leading coefficient", lead, 3.0, 1e-9)

    print("\n  The same builder, writing its zero coefficients as R(0) instead of")
    print("  skipping them.  R(0) is |1|-1, so it adds h*x^k to the polynomial and")
    print("  drags the measured order down to 1.  The composite is right about the")
    print("  function it was given; the function was not the one intended.")
    for m in (2, 5):
        coeffs = expanded_coeffs(2.0, m, -1.0)
        assert 0.0 in coeffs, "this m was chosen because its expansion has a zero"
        order, _ = vanishing_order(poly_from_coeffs(coeffs, skip_zeros=False), 2.0)
        print(f"      m={m}: R(0) for absent coefficients -> order {order}, "
              f"not {m}.  Absent is not zero.")


# ---------------------------------------------------------------------------
# 3 and 4. Order of contact, and the vertex
# ---------------------------------------------------------------------------
def contact_order(f, g, at, depth=12):
    return vanishing_order(lambda x: f(x) - g(x), at, depth)


def taylor_sin(n):
    """Taylor polynomial of sin to degree n.  Even terms are ABSENT, not zero."""
    def g(x):
        acc = NOTHING
        for k in range(1, n + 1, 2):
            acc = acc + R(((-1) ** ((k - 1) // 2)) / math.factorial(k)) * x ** k
        return acc
    return g


def demo_contact():
    section("3. ORDER OF CONTACT -- the grade of (f - g)(a + h)")
    print("  Two curves through the same point.  The order of vanishing of their")
    print("  difference is the order of contact, and the leading coefficient is")
    print("  the first term by which they part company.\n")
    rows = [
        ("x^2 and its tangent at 1",      lambda x: x * x,
         lambda x: R(2.0) * x - R(1.0),   1.0, 2,  1.0),
        ("sin and deg-1 Taylor at 0",     sin, taylor_sin(1), 0.0, 3, -1 / 6),
        ("sin and deg-3 Taylor at 0",     sin, taylor_sin(3), 0.0, 5,  1 / 120),
        ("sin and deg-5 Taylor at 0",     sin, taylor_sin(5), 0.0, 7, -1 / 5040),
        ("sin and tan at 0",              sin, tan,           0.0, 3, -0.5),
        ("exp and 1 + x + x^2/2 at 0",    exp,
         lambda x: R(1.0) + x + x * x / R(2.0), 0.0, 3, 1 / 6),
        ("ln(x) and x - 1 at 1",          ln,
         lambda x: x - R(1.0),            1.0, 2, -0.5),
    ]
    for label, f, g, at, want_order, want_coeff in rows:
        order, lead = contact_order(f, g, at)
        check(f"{label}: contact order", order or 0.0, want_order)
        check(f"{label}: leading coefficient", lead, want_coeff, 1e-9)


def demo_vertex():
    section("4. VERTEX -- contact with the osculating circle goes from 3 to 4")
    print("  y = x^2.  Its curvature is largest at x = 0, which is a vertex, and")
    print("  ordinary at x = 1.  The osculating circle has contact order 3 at a")
    print("  generic point and 4 at a vertex, so the vertex is visible in the")
    print("  grade.  No curvature is formed and dk/ds is never differentiated.\n")

    def parabola(x):
        return x * x

    def osc_circle(centre_x, centre_y, r2):
        def g(x):
            # A centre coordinate of 0 is ABSENT.  `x - R(0.0)` subtracts an
            # infinitesimal and moves the curve, which reported the vertex at
            # contact order 2 instead of 4 the first time this demo was run.
            dx = x if centre_x == 0 else x - R(centre_x)
            return R(centre_y) - sqrt(R(r2) - dx ** 2)
        return g

    # x = 0: k = 2, radius 1/2, centre (0, 1/2)
    at_vertex = osc_circle(0.0, 0.5, 0.25)
    # x = 1: y' = 2, y'' = 2, radius 5^(3/2)/2, centre (-4, 3.5), r^2 = 125/4
    at_generic = osc_circle(-4.0, 3.5, 31.25)

    order, lead = contact_order(parabola, at_generic, 1.0)
    check("generic point x=1: contact order", order or 0.0, 3)
    check("generic point x=1: leading coefficient", lead, -0.8, 1e-9)
    order, lead = contact_order(parabola, at_vertex, 0.0)
    check("vertex x=0: contact order", order or 0.0, 4)
    check("vertex x=0: leading coefficient", lead, -1.0, 1e-9)
    print("\n      3 at the ordinary point, 4 at the vertex, from one subtraction each.")


# ---------------------------------------------------------------------------
# 5. Where the grade saturates and the coefficient takes over
# ---------------------------------------------------------------------------
def inverse_residue(A, B):
    """Coefficient of 1/h in adj(A + hB) / det(A + hB)."""
    n = len(A)
    M = perturbed(A, B)
    D = det(M)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor = [[M[r][c] for c in range(n) if c != i] for r in range(n) if r != j]
            cof = det(minor)
            if (i + j) % 2:
                cof = cof * R(-1.0)
            res[i][j] = cl._ensure_composite(cof / D).coeffs_dict().get(1, 0.0)
    return res


def demo_saturation():
    section("5. WHERE THE GRADE RUNS OUT -- order in the grade, structure in the coefficient")
    print("  The inverse of a singular matrix has pole order 1 however deficient it")
    print("  is: a single scalar order cannot hold a subspace, so the grade")
    print("  saturates.  The corank reappears as the RANK of the coefficient at")
    print("  that grade, and that coefficient maps into the null space on both")
    print("  sides.  It is the singular part of the resolvent, obtained by")
    print("  dividing two cofactor expansions and reading one coefficient.\n")
    rng = random.Random(5)
    for n, rank in ((3, 2), (3, 1), (4, 3), (4, 2)):
        A = matrix_of_rank(n, rank, rng)
        B = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        corank = n - int(np.linalg.matrix_rank(A))
        with cl._derivative_scope(n + 2, n + 6):
            res = inverse_residue(A, B)
            dorder, _ = order_and_coeff(det(perturbed(A, B)))
        scale = max(1.0, float(np.abs(res).max()))
        check(f"n={n} corank={corank}: order of det", dorder or 0.0, corank)
        check(f"n={n} corank={corank}: rank of the residue",
              float(np.linalg.matrix_rank(res)), float(corank))
        check(f"n={n} corank={corank}: |A . residue| (in ker A)",
              float(np.abs(A @ res).max()) / scale, 0.0, 1e-9)
        check(f"n={n} corank={corank}: |residue . A| (in ker A^T)",
              float(np.abs(res @ A).max()) / scale, 0.0, 1e-9)


def main():
    print(__doc__.split("\n")[0])
    demo_rank()
    demo_multiplicity()
    demo_contact()
    demo_vertex()
    demo_saturation()
    print(f"\nRESULTS: {sum(_results)}/{len(_results)} checks passed")
    print("""
Four questions that classically need four algorithms and four tolerances:
rank by SVD with a threshold, multiplicity by GCD with the derivative, contact
by matching derivatives one at a time, a vertex by differentiating curvature.
Each one here is a subtraction and a lead_order(), and the answer is an integer
so there is nothing to threshold.  What the grade cannot hold, because it is
one number, sits in the coefficient at that grade instead.""")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
