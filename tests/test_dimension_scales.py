#!/usr/bin/env python3
# Composite Machine — fractional dimensions and the log scale
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Dimensions that are not integers, and dimensions that are not scalars.

Two capabilities, one theme: what a dimension is allowed to be.

  FRACTIONAL (dyadic).  Dimensions are float64, so sqrt(|c|_d) = |sqrt(c)|_(d/2)
  exists for odd d.  Every float64 IS a dyadic rational (m x 2^e), and the
  dyadics are closed under {+, -, /2} -- exactly composite's index operations,
  since multiply adds dimensions, divide subtracts, and sqrt halves.  So the
  representation and the reachable set coincide: nothing composite can do
  leaves it, and nothing outside it is representable.  An n-th root for n not a
  power of 2 WOULD leave it, which is why there is no cbrt.

  VECTOR (the log scale).  ln of an infinitesimal is ln(c) + d*ln(h), and ln(h)
  needs a dimension that is positive but smaller than EVERY power -- log x
  outgrows any constant and is outgrown by x^e for every e > 0.  No float sits
  there.  A vector dimension (power, log) does: the log is the minor component,
  so lexicographic order puts any log term below any power term, which is the
  dominance order.  Off by default; see LOG_SCALE.

Tolerances here are tight on purpose.  Most of these assertions are about the
ALGEBRA, which is exact, so they compare exactly; the few that route through a
transcendental use 1e-12.  A loose tolerance on an exact result is not a test.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, INF
from composite.backends import config
from composite.backends.dict_backend import DictBackend
from composite.backends.vector_dim_backend import VectorDimBackend


class Suite:
    def __init__(self):
        self.passed = self.failed = 0
        self.fails = []

    def _note(self, tag, ok, detail=""):
        self.passed += ok
        self.failed += (not ok)
        # Numbers on PASS too -- a tick alone cannot be audited.
        print(f"  {'OK  ' if ok else 'FAIL'} {tag}"
              + (f"   {detail}" if detail else ""))
        if not ok:
            self.fails.append(tag)

    # Every check takes a THUNK where the value could throw, so a regression
    # that raises is reported as one failed test rather than aborting the run.
    # Three of five mutation tests crashed the suite before this was added, and
    # a crash tells you far less than "18 failed, here they are".
    def _eval(self, tag, got):
        try:
            return (True, got() if callable(got) else got)
        except Exception as e:
            self._note(tag, False, f"raised {type(e).__name__}: {e}")
            return (False, None)

    def exact(self, tag, got, want):
        """Exact equality -- for anything the algebra should produce exactly."""
        ok, v = self._eval(tag, got)
        if ok:
            self._note(tag, v == want, f"got {v!r}  want {want!r}")

    def close(self, tag, got, want, tol=1e-12):
        ok, v = self._eval(tag, got)
        if not ok:
            return
        try:
            g = float(v.st()) if isinstance(v, Composite) else float(v)
        except Exception as e:
            self._note(tag, False, f"not numeric: {type(e).__name__}: {e}")
            return
        self._note(tag, abs(g - want) <= tol,
                   f"got {g!r}  want {want!r}  err {abs(g-want):.3e}  tol {tol:g}")

    def true(self, tag, cond, detail=""):
        ok, v = self._eval(tag, cond)
        if ok:
            self._note(tag, bool(v), detail)

    def raises(self, tag, exc, fn):
        try:
            got = fn()
            self._note(tag, False, f"expected {exc.__name__}, got {got!r}")
        except exc:
            self._note(tag, True)
        except Exception as e:
            self._note(tag, False, f"expected {exc.__name__}, got {type(e).__name__}: {e}")

    def dims(self, tag, c, want):
        """Compare the full {dim: coeff} map exactly."""
        ok, v = self._eval(tag, c)
        if not ok:
            return
        try:
            got = {k: x for k, x in v.coeffs_dict().items() if x != 0.0}
        except Exception as e:
            self._note(tag, False, f"raised {type(e).__name__}: {e}")
            return
        self._note(tag, got == want, f"got {got}  want {want}")


def head(s):
    print(f"\n{'=' * 66}\n{s}\n{'=' * 66}")


# =============================================================================
def dyadic_dimensions(t):
    head("1. FRACTIONAL DIMENSIONS — sqrt of an odd dimension")

    t.dims("D01 sqrt(ZERO) = |1|_-0.5", cl.sqrt(ZERO), {-0.5: 1.0})
    t.dims("D02 sqrt(INF)  = |1|_+0.5", cl.sqrt(INF), {0.5: 1.0})
    t.dims("D03 sqrt(|4|_-2) = |2|_-1", cl.sqrt(Composite({-2: 4.0})), {-1: 2.0})
    t.dims("D04 sqrt(|9|_0)  = |3|_0", cl.sqrt(R(9)), {0: 3.0})

    # nesting: each sqrt halves, and a dyadic stays dyadic
    x = ZERO
    for k in range(1, 6):
        x = cl.sqrt(x)
        t.dims(f"D05.{k} sqrt^{k}(ZERO) = |1|_-{2.0**-k}", x, {-(2.0 ** -k): 1.0})

    # and the tower closes back on itself
    q = cl.sqrt(cl.sqrt(cl.sqrt(ZERO)))            # h^(1/8)
    p = q
    for _ in range(3):
        p = p * p                                   # ^8
    t.exact("D06 (sqrt^3(ZERO))^8 == ZERO", p, ZERO)

    for nm, x in (("ZERO", ZERO), ("INF", INF), ("R(4)", R(4)),
                  ("ZERO^2", ZERO * ZERO)):
        r = cl.sqrt(x)
        t.exact(f"D07 sqrt({nm})^2 == {nm}", r * r, x)

    # the three limits the old integer-dimension sqrt got wrong
    _r = cl.limit_right(lambda x: cl.sqrt(x) / x, 0.0)
    t.true("D08 lim(x->0+) sqrt(x)/x is INFINITE, not 1.0",
           isinstance(_r, Composite) and _r.max_positive_dim() is not None,
           f"got {_r!r}")
    t.close("D09 lim(x->0+) x/sqrt(x) = 0",
            cl.limit_right(lambda x: x / cl.sqrt(x), 0.0), 0.0)
    t.close("D10 lim(x->0+) sqrt(x*x)/x = 1",
            cl.limit_right(lambda x: cl.sqrt(x * x) / x, 0.0), 1.0)


def closure(t):
    head("2. DYADIC CLOSURE — composite's own operations cannot leave ℤ[1/2]")

    def _well_conditioned(c):
        """A divisor whose leading coefficient dominates its lower terms."""
        d = {k: v for k, v in c.coeffs_dict().items() if v != 0.0}
        if not d:
            return False
        lead = d[max(d)]
        others = [abs(v) for k, v in d.items() if k != max(d)]
        return abs(lead) > 0 and (not others or abs(lead) >= max(others))

    def is_dyadic(d):
        if isinstance(d, tuple):
            d = d[0]
        f = float(d)
        return f == f and abs(f) != float("inf")      # every finite float is dyadic

    import random
    rng = random.Random(20260916)
    pool = [R(2), ZERO, INF, R(1) + ZERO, cl.sqrt(ZERO)]
    escaped = 0
    ops = 0
    for _ in range(600):
        x, y = rng.choice(pool), rng.choice(pool)
        op = rng.choice(["*", "/", "+", "-", "sqrt"])
        if op == "/" and not _well_conditioned(y):
            # Long division by a composite whose LEADING coefficient is smaller
            # than its lower terms diverges: each step multiplies the remainder
            # by their ratio, and 50 iterations of that overflows to inf, then
            # inf-inf gives nan.  That is a property of the division, not of
            # the dimension set, and this test is about the dimension set.
            continue
        try:
            r = {"*": lambda: x * y, "/": lambda: x / y, "+": lambda: x + y,
                 "-": lambda: x - y, "sqrt": lambda: cl.sqrt(x)}[op]()
        except Exception:
            continue
        ops += 1
        coeffs = r.coeffs_dict()
        if any(not is_dyadic(d) for d in coeffs):
            escaped += 1
        # Feed back only well-conditioned results.  Dividing by a composite
        # whose leading coefficient is smaller than its lower terms produces a
        # DIVERGENT quotient -- the library computes 50 terms of it and returns
        # them -- and compounding those reaches inf, then inf-inf gives nan.
        # That is a property of the division, not of the dimension set, and
        # letting it into the pool would test conditioning instead of closure.
        vals = [abs(v) for v in coeffs.values()]
        finite = all(v == v and v != float("inf") for v in vals)
        if finite and len(coeffs) < 10 and (not vals or max(vals) < 1e6):
            pool.append(r)
        pool = pool[-20:]
    t.true(f"C01 {ops} random ops over *,/,+,-,sqrt stay dyadic",
           escaped == 0, f"{escaped} escaped")

    # No operation over well-conditioned operands may produce nan or inf --
    # composite has no representation for either, so one appearing means an
    # escape to raw floats somewhere.
    rng2 = random.Random(99)
    p2 = [R(2), ZERO, INF, R(1) + ZERO, cl.sqrt(ZERO), R(3) + INF]
    nonfinite = 0
    for _ in range(400):
        x, y = rng2.choice(p2), rng2.choice(p2)
        op = rng2.choice(["*", "+", "-", "sqrt"])
        try:
            r = {"*": lambda: x * y, "+": lambda: x + y,
                 "-": lambda: x - y, "sqrt": lambda: cl.sqrt(x)}[op]()
        except Exception:
            continue
        if any(v != v or abs(v) == float("inf") for v in r.coeffs_dict().values()):
            nonfinite += 1
    t.true("C03 400 ops (no division) produce no nan or inf",
           nonfinite == 0, f"{nonfinite} produced non-finite coefficients")

    # halving an integer dimension repeatedly is exact for 50 levels
    d = 1.0
    ok = True
    for k in range(1, 51):
        d /= 2
        if d * (2 ** k) != 1.0:
            ok = False
    t.true("C02 halving is exact to 2^-50", ok)


def ordering(t):
    head("3. ORDERING — the Levi-Civita field's own specification")

    MINF = 5e-324
    MAXF = 1.797e308
    t.true("O01 eps < MIN_FLOAT", ZERO < R(MINF))
    t.true("O02 1-eps < 1 < 1+eps", (R(1) - ZERO < R(1)) and (R(1) < R(1) + ZERO))
    t.true("O03 eps/2 < eps < 7*eps", (ZERO / R(2) < ZERO) and (ZERO < R(7) * ZERO))
    t.true("O04 eps^2 < eps", ZERO * ZERO < ZERO)
    t.true("O05 eps < sqrt(eps)", ZERO < cl.sqrt(ZERO))
    t.true("O06 MAX_FLOAT < 1/eps", R(MAXF) < INF)
    t.true("O07 sqrt(1/eps) < 1/eps", cl.sqrt(INF) < INF)
    t.true("O08 1/eps < (1/eps)^2", INF < INF * INF)


def transcendentals(t):
    head("4. TRANSCENDENTALS ON A FRACTIONAL ARGUMENT")

    h = cl.sqrt(ZERO)                                  # h = eps^(1/2)
    # exp(h) = 1 + h + h^2/2 + h^3/6 ...   dims 0, -0.5, -1, -1.5
    e = cl.exp(h).coeffs_dict()
    t.close("T01 exp(sqrt(eps)) coeff at dim 0", e.get(0, 0.0), 1.0)
    t.close("T02 exp(sqrt(eps)) coeff at dim -0.5", e.get(-0.5, 0.0), 1.0)
    t.close("T03 exp(sqrt(eps)) coeff at dim -1", e.get(-1, 0.0), 0.5)
    t.close("T04 exp(sqrt(eps)) coeff at dim -1.5", e.get(-1.5, 0.0), 1 / 6)

    # sin(h) = h - h^3/6 + h^5/120   dims -0.5, -1.5, -2.5
    s = cl.sin(h).coeffs_dict()
    t.close("T05 sin(sqrt(eps)) at dim -0.5", s.get(-0.5, 0.0), 1.0)
    t.close("T06 sin(sqrt(eps)) at dim -1.5", s.get(-1.5, 0.0), -1 / 6)
    t.close("T07 sin(sqrt(eps)) at dim -2.5", s.get(-2.5, 0.0), 1 / 120)

    # cos(h) = 1 - h^2/2 + h^4/24   dims 0, -1, -2  (integers!)
    c = cl.cos(h).coeffs_dict()
    t.close("T08 cos(sqrt(eps)) at dim 0", c.get(0, 0.0), 1.0)
    t.close("T09 cos(sqrt(eps)) at dim -1", c.get(-1, 0.0), -0.5)
    t.close("T10 cos(sqrt(eps)) at dim -2", c.get(-2, 0.0), 1 / 24)

    # ln(1+h) = h - h^2/2 + h^3/3   dims -0.5, -1, -1.5
    l = cl.ln(R(1) + h).coeffs_dict()
    t.close("T11 ln(1+sqrt(eps)) at dim -0.5", l.get(-0.5, 0.0), 1.0)
    t.close("T12 ln(1+sqrt(eps)) at dim -1", l.get(-1, 0.0), -0.5)
    t.close("T13 ln(1+sqrt(eps)) at dim -1.5", l.get(-1.5, 0.0), 1 / 3)

    # half-dimensions annihilate in pairs
    t.dims("T14 sqrt(eps)^2 back to an integer dim", h * h, {-1: 1.0})


def guardrails(t):
    head("5. GUARDRAILS — silent collapse now raises")

    C = R(2) + ZERO                                    # distinguishable from 2.0
    t.raises("G01 R(Composite)", TypeError, lambda: R(C))
    t.raises("G02 Composite.real(Composite)", TypeError, lambda: Composite.real(C))
    t.raises("G03 _seeded(Composite)", TypeError, lambda: cl._seeded(C))
    t.raises("G04 derivative(at=Composite)", TypeError,
             lambda: cl.derivative(cl.exp, C))
    t.raises("G05 nth_derivative(at=Composite)", TypeError,
             lambda: cl.nth_derivative(cl.exp, 1, C))
    t.raises("G06 taylor_coefficients(at=Composite)", TypeError,
             lambda: cl.taylor_coefficients(cl.exp, C, 2))
    t.raises("G07 limit(at=infinitesimal)", TypeError, lambda: cl.limit(cl.exp, C))

    # but the documented uses still work
    t.close("G08 limit(f, INF) still allowed",
            cl.limit(lambda x: R(1) / x, INF), 0.0)
    t.close("G09 limit(f, R(2)) collapses losslessly", cl.limit(cl.exp, R(2)),
            math.exp(2))
    t.exact("G10 R(0) is still the canonical zero", R(0), ZERO)

    # power with a COMPOSITE exponent: used to collapse the exponent to st()
    t.close("G11 power(1+x, 1/x) = e", cl.power(R(1) + ZERO, R(1) / ZERO), math.e)
    t.close("G12 (1+x)**(1/x) = e", (R(1) + ZERO) ** (R(1) / ZERO), math.e)
    t.close("G13 power(4, 0.5) unchanged", cl.power(R(4), 0.5), 2.0)
    t.close("G14 power(2, 10) unchanged", cl.power(R(2), 10), 1024.0)

    # a pole is not a Taylor series
    t.raises("G15 taylor_coefficients at a pole", ValueError,
             lambda: cl.taylor_coefficients(
                 lambda x: R(1) / (R(1) - cl.cos(x)), at=0.0, up_to=3))

    # derivative order is no longer capped by MAX_ACTIVE_DIMS
    v = cl.nth_derivative(lambda x: cl.exp(x) / (R(1) + x * x), 70, 0.3, terms=78)
    t.true("G16 derivative order 70 is not silently zero", v != 0.0, f"got {v}")


def log_scale_off(t):
    head("6. LOG SCALE — on by default, and disableable")

    t.true("F01 LOG_SCALE defaults to True", cl.LOG_SCALE is True)
    t.dims("F02 ln(eps) answers by default", cl.ln(ZERO), {(0, 1): -1.0})

    # Setting it False restores the guardrail: ln RAISES rather than dropping
    # the scale, because dropping it made ln(h), ln(h^2) and ln(sqrt(h)) the
    # same object and produced wrong ratios, not merely lost structure.
    cl.LOG_SCALE = False
    try:
        t.raises("F03 ln(eps) raises when disabled", ValueError,
                 lambda: cl.ln(ZERO))
        t.raises("F04 ln(eps^2) raises when disabled", ValueError,
                 lambda: cl.ln(ZERO * ZERO))
        t.close("F05 ln(2) unaffected when disabled", cl.ln(R(2)), math.log(2))
    finally:
        cl.LOG_SCALE = True

    # ln of something with a standard part never needed the scale
    t.close("F06 ln(2) unaffected", cl.ln(R(2)), math.log(2))
    t.close("F07 ln(1+eps) unaffected", cl.ln(R(1) + ZERO), 0.0)


def log_scale_on(t):
    head("7. LOG SCALE — enabled")

    try:
        # ln now NAMES the scale: h, h^2, h^3 and sqrt(h) are four answers
        t.dims("F06 ln(h)      = -1*L", cl.ln(ZERO), {(0, 1): -1.0})
        t.dims("F07 ln(h^2)    = -2*L", cl.ln(ZERO * ZERO), {(0, 1): -2.0})
        t.dims("F08 ln(h^3)    = -3*L", cl.ln(ZERO * ZERO * ZERO), {(0, 1): -3.0})
        t.dims("F09 ln(sqrt h) = -0.5*L", cl.ln(cl.sqrt(ZERO)), {(0, 1): -0.5})
        t.dims("F10 ln(1/h)    = +1*L", cl.ln(INF), {(0, 1): 1.0})

        ln5 = cl.ln(R(5) * ZERO).coeffs_dict()
        t.close("F11 ln(5h) log part", ln5.get((0, 1), 0.0), -1.0)
        t.close("F12 ln(5h) real part", ln5.get((0, 0), 0.0), math.log(5))

        # exp inverts it, and the result returns to the scalar path
        for nm, x in (("h", ZERO), ("h^2", ZERO * ZERO),
                      ("h^3", ZERO * ZERO * ZERO), ("sqrt(h)", cl.sqrt(ZERO))):
            b = cl.exp(cl.ln(x))
            t.exact(f"F13 exp(ln({nm})) == {nm}", b, x)

        # the four answers that were WRONG before the log scale existed
        t.close("F14 ln(h)/ln(h^2)     = 1/2",
                cl.ln(ZERO) / cl.ln(ZERO * ZERO), 0.5)
        t.close("F15 ln(h^2)/ln(h)     = 2",
                cl.ln(ZERO * ZERO) / cl.ln(ZERO), 2.0)
        t.close("F16 ln(h^3)/ln(h)     = 3",
                cl.ln(ZERO * ZERO * ZERO) / cl.ln(ZERO), 3.0)
        t.close("F17 ln(h)/ln(sqrt(h)) = 2",
                cl.ln(ZERO) / cl.ln(cl.sqrt(ZERO)), 2.0)

        # 1/ln(h) was |1|_1 -- an INFINITY where the answer is 0
        t.close("F18 1/ln(h) is infinitesimal, not infinite",
                R(1) / cl.ln(ZERO), 0.0)
        t.dims("F19 1/ln(h) sits at a negative log level",
               R(1) / cl.ln(ZERO), {(0, -1): -1.0})

        # and the cases that were RIGHT stay right
        t.close("F20 h*ln(h) -> 0", ZERO * cl.ln(ZERO), 0.0)
        t.close("F21 h^2*ln(h) -> 0", ZERO * ZERO * cl.ln(ZERO), 0.0)
        t.dims("F22 h*ln(h) is x^-1 log^1", ZERO * cl.ln(ZERO), {(-1, 1): -1.0})
    finally:
        cl.LOG_SCALE = True


def demotion(t):
    head("8. DEMOTION — the vector path is used only while it is needed")

    try:
        vec = lambda c: type(c._backend).__name__ == "VectorDimBackend"

        # carries a log component -> stays on the vector backend
        t.true("M01 ln(h) stays vector", vec(cl.ln(ZERO)))
        t.true("M02 h*ln(h) stays vector", vec(ZERO * cl.ln(ZERO)))
        t.true("M03 1/ln(h) stays vector", vec(R(1) / cl.ln(ZERO)))

        # log components cancel -> returns to the scalar path
        d = cl.ln(ZERO) / cl.ln(ZERO * ZERO)
        t.true("M04 ln(h)/ln(h^2) demotes", not vec(d))
        t.dims("M05 and its dimension is scalar again", d, {0: 0.5})

        e = cl.exp(cl.ln(ZERO))
        t.true("M06 exp(ln(h)) demotes", not vec(e))
        t.dims("M07 and equals h", e, {-1: 1.0})

        # a demoted value is fully usable on the scalar path
        t.dims("M08 demoted * R(4)", d * R(4), {0: 2.0})
        t.dims("M09 demoted + R(1)", d + R(1), {0: 1.5})
        t.close("M10 exp(demoted)", cl.exp(d), math.exp(0.5))
    finally:
        cl.LOG_SCALE = True


def vector_backend(t):
    head("9. VECTOR DIMENSIONS — arithmetic and mixing")

    prev = config.get_backend()
    try:
        config.set_backend(VectorDimBackend())
        A = Composite({(1, 0): 2.0, (0, 1): 3.0})       # 2x + 3 log x
        B = Composite({(-1, 0): 5.0, (0, 1): 1.0})      # 5/x + log x

        # (2x + 3log x)(5/x + log x) = 10 + 15 log x/x + 2x log x + 3 log^2 x
        t.dims("V01 product adds dimensions componentwise", A * B,
               {(0, 0): 10.0, (-1, 1): 15.0, (1, 1): 2.0, (0, 2): 3.0})
        t.dims("V02 sum merges on equal dimensions", A + B,
               {(1, 0): 2.0, (0, 1): 4.0, (-1, 0): 5.0})
        t.dims("V03 single-term division shifts", A / Composite({(2, 0): 1.0}),
               {(-1, 0): 2.0, (-2, 1): 3.0})
        t.dims("V04 scalar multiply leaves dimensions", A * 3,
               {(1, 0): 6.0, (0, 1): 9.0})
        t.dims("V05 negate", -A, {(1, 0): -2.0, (0, 1): -3.0})
        t.true("V06 equality", A == A)
        t.true("V07 a power dominates a log", Composite({(1, 0): 1.0}) >
               Composite({(0, 9): 1.0}))
        t.true("V08 higher log level dominates lower",
               Composite({(0, 2): 1.0}) > Composite({(0, 1): 1.0}))
    finally:
        config.set_backend(prev)

    # mixing a vector composite with a scalar one, BOTH ways round
    prev = config.get_backend()
    try:
        config.use_sparse_dense()
        S = Composite({1: 2.0})                          # 2x, scalar
        config.set_backend(VectorDimBackend())
        V = Composite({(0, 1): 3.0})                     # 3 log x, vector
        t.dims("V09 vector * scalar", V * S, {(1, 1): 6.0})
        t.dims("V10 scalar * vector", S * V, {(1, 1): 6.0})
        t.dims("V11 vector + scalar", V + S, {(1, 0): 2.0, (0, 1): 3.0})
        t.dims("V12 scalar + vector", S + V, {(1, 0): 2.0, (0, 1): 3.0})
    finally:
        config.set_backend(prev)


def scalar_path_unchanged(t):
    head("10. THE SCALAR PATH IS UNTOUCHED — differential against DictBackend")

    import random
    rng = random.Random(4242)
    prev = config.get_backend()
    mismatch = 0
    try:
        for _ in range(1500):
            mk = lambda: {float(rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 2])):
                          float(rng.randint(-4, 4))
                          for _ in range(rng.randint(1, 4))}
            da, db = mk(), mk()
            config.use_sparse_dense()
            s = ((Composite(da) * Composite(db)).coeffs_dict(),
                 (Composite(da) + Composite(db)).coeffs_dict())
            config.use_dict()
            d = ((Composite(da) * Composite(db)).coeffs_dict(),
                 (Composite(da) + Composite(db)).coeffs_dict())
            if s != d:
                mismatch += 1
    finally:
        config.set_backend(prev)
    t.true("P01 1500 products+sums, sparse-dense vs dict, fractional dims",
           mismatch == 0, f"{mismatch} mismatches")

    # a vector composite carrying only powers must equal the scalar result
    prev = config.get_backend()
    bad = 0
    try:
        vb, db = VectorDimBackend(), DictBackend()
        from composite.backends.dict_backend import DictData
        for _ in range(1500):
            mk = lambda: {rng.randint(-4, 4): float(rng.randint(-4, 4))
                          for _ in range(rng.randint(1, 4))}
            ds, es = mk(), mk()
            sca = db.convolve(DictData(ds), DictData(es)).terms
            vec = vb.convolve(DictData({(k, 0): v for k, v in ds.items()}),
                              DictData({(k, 0): v for k, v in es.items()})).terms
            if {k[0]: v for k, v in vec.items()} != sca:
                bad += 1
    finally:
        config.set_backend(prev)
    t.true("P02 1500 products, vector dims with zero log == scalar dims",
           bad == 0, f"{bad} mismatches")


def limits_with_log_scale(t):
    head("11. LIMITS — what the log scale unblocks")

    # These raise with LOG_SCALE off, because ln of an infinitesimal has
    # nowhere to put the d*ln(h) term.  With it on they are answerable.
    blocked = [
        ("L01 lim(x->0+) sqrt(x)*ln(x) = 0",
         lambda x: cl.sqrt(x) * cl.ln(x), 0.0),
        ("L02 lim(x->0+) ln(x)/ln(x*x) = 1/2",
         lambda x: cl.ln(x) / cl.ln(x * x), 0.5),
        ("L03 lim(x->0+) ln(x*x)/ln(x) = 2",
         lambda x: cl.ln(x * x) / cl.ln(x), 2.0),
        ("L04 lim(x->0+) ln(x)/ln(sqrt(x)) = 2",
         lambda x: cl.ln(x) / cl.ln(cl.sqrt(x)), 2.0),
        ("L05 lim(x->0+) 1/ln(x) = 0",
         lambda x: R(1) / cl.ln(x), 0.0),
    ]
    # With LOG_SCALE off, ln of an infinitesimal RAISES -- that guardrail is
    # asserted directly below.  What limit_right() then does is a separate
    # question: it falls back to numeric extrapolation at real probe points,
    # where ln takes its ordinary series branch and never raises.  Whether that
    # fallback converges is not something the log scale controls, so requiring
    # an exception here tested the wrong thing.  It passed only because
    # ln(x*x) -- two-order h -- used to return incomplete high orders that
    # poisoned the extrapolation; once ln was truncated to the orders it
    # completes, L02 and L03 started converging to 13 digits.
    #
    # The property that actually matters is that DISABLING the log scale never
    # produces a silently WRONG answer.  Raise or be right, never neither.
    cl.LOG_SCALE = False
    try:
        for tag, f, want in blocked:
            nm = tag.replace("=", "raises or is correct when DISABLED, want")
            try:
                got = float(cl.limit_right(f, 0.0))
            except Exception:
                t.true(nm + " [raised]", True)
                continue
            t.close(nm + " [converged]", got, want, tol=1e-9)
        # the guardrail itself, unmediated by the limit machinery
        t.raises("L00 ln(ZERO) raises when log scale DISABLED", ValueError,
                 lambda: cl.ln(cl.ZERO))
        t.raises("L00b ln(ZERO*ZERO) raises when DISABLED", ValueError,
                 lambda: cl.ln(cl.ZERO * cl.ZERO))
    finally:
        cl.LOG_SCALE = True

    try:
        for tag, f, want in blocked:
            t.close(tag, cl.limit_right(f, 0.0), want, tol=1e-12)

        # x^x goes ALGEBRAIC rather than falling back to numeric sampling.
        # With the flag off the fallback lands 1.6e-6 short of 1.
        t.close("L06 lim(x->0+) x^x = 1 exactly",
                cl.limit_right(lambda x: x ** x, 0.0), 1.0, tol=0.0)
        t.close("L07 lim(x->0+) exp(x*ln x) = 1 exactly",
                cl.limit_right(lambda x: cl.exp(x * cl.ln(x)), 0.0), 1.0, tol=0.0)

        # and the ones that already worked are unchanged
        t.close("L08 lim(x->0+) x*ln(x) = 0",
                cl.limit_right(lambda x: x * cl.ln(x), 0.0), 0.0)
        t.close("L09 lim(x->0+) x^2*ln(x) = 0",
                cl.limit_right(lambda x: x * x * cl.ln(x), 0.0), 0.0)
        t.close("L10 lim(x->0+) x/ln(x) = 0",
                cl.limit_right(lambda x: x / cl.ln(x), 0.0), 0.0)
        t.close("L11 lim(x->0+) ln(1+x)/x = 1",
                cl.limit_right(lambda x: cl.ln(R(1) + x) / x, 0.0), 1.0)
    finally:
        cl.LOG_SCALE = True

    # exp of a MIXED power/log term is an ordinary series, not a scale change:
    # h*log(h) vanishes, so exp of it expands.  Refusing this is what left x^x
    # on the numeric fallback.
    try:
        r = cl.exp(cl._seeded(0.0) * cl.ln(cl._seeded(0.0)))
        d = r.coeffs_dict()
        t.close("L12 exp(h*ln h) constant term", d.get((0, 0), 0.0), 1.0)
        t.close("L13 exp(h*ln h) first order", d.get((-1, 1), 0.0), -1.0)
        t.close("L14 exp(h*ln h) second order", d.get((-2, 2), 0.0), 0.5)
        t.close("L15 exp(h*ln h) third order", d.get((-3, 3), 0.0), -1 / 6)
    finally:
        cl.LOG_SCALE = True


# =============================================================================
def run_all():
    print("#" * 66)
    print("# DIMENSION SCALES — fractional dimensions and the log scale")
    print("#" * 66)

    t = Suite()
    for fn in (dyadic_dimensions, closure, ordering, transcendentals,
               guardrails, log_scale_off, log_scale_on, demotion,
               vector_backend, scalar_path_unchanged,
               limits_with_log_scale):
        try:
            fn(t)
        except Exception as e:
            t._note(f"{fn.__name__} ABORTED", False,
                    f"{type(e).__name__}: {e}")
            cl.LOG_SCALE = True

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
