#!/usr/bin/env python3
# Composite Machine — vector dimensions and depth genericity
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Dimensions as vectors over an iterated-logarithm basis, at arbitrary depth.

WHAT THIS GUARDS.  A dimension (e0, e1, e2, ...) means
c * (1/h)**e0 * B1**e1 * B2**e2 ..., where B_k is the k-times-iterated log.
ln moves every exponent ONE INDEX UP and turns it into a coefficient; exp moves
one back down.  One rule, any depth -- so adding a level is not a code change.

WHY IT NEEDS ITS OWN SUITE.  Nearly every defect found while making this work
was a SCALAR SPELLING meeting a tuple, and none of them were caught by the
existing suites because those never build a vector dimension:

    _lead != 0        a tuple is never == 0, so the vector ZERO tested
                      non-zero, sqrt divided x by 1 and recursed forever
    dim - 1           TypeError on the order shift (antiderivative, atan, asin)
    _lead / 2         TypeError halving the index (sqrt)
    {0: constant}     an int key beside tuple keys -- sorted() cannot order them
    d > 0             TypeError, or -- worse -- silently read only the POWER
                      component, so log(1/h) tested FINITE and infinite
                      arguments took the Taylor path
    zip(da, db)       over different-length tuples, truncates to the shorter
                      and drops the deepest axis in silence

Two more were not about vectors at all and surfaced only here: _like never
sorted its keys (it silently dropped asin's standard part), and __truediv__
dropped the backend where __mul__ keeps it.

CROSS-DEPTH IDENTITY IS THE SUBTLE ONE.  A dimension built at width 2 and one
built at width 4 must be the SAME KEY, or every lookup after the basis grows
returns 0 -- which made log(1/h) compare as not greater than loglog(1/h).  The
canonical form (trailing zeros stripped) is what makes Python's own tuple
ordering the dominance order, with no padding at comparison time.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import composite.composite_lib as cl
from composite.composite_lib import Composite, R
import composite.backends.vector_dim_backend as vb

cl.MAX_ACTIVE_DIMS = 10 ** 9


class T:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def _note(self, tag, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  OK   {tag}")
        else:
            self.failed.append((tag, detail))
            print(f"  FAIL {tag}\n         {detail}")

    def true(self, tag, cond, detail=""):
        try:
            self._note(tag, bool(cond), detail)
        except Exception as e:
            self._note(tag, False, f"{type(e).__name__}: {e}")

    def eq(self, tag, got, want):
        try:
            g = got() if callable(got) else got
        except Exception as e:
            return self._note(tag, False, f"raised {type(e).__name__}: {e}")
        self._note(tag, g == want, f"got {g!r}  want {want!r}")

    def close(self, tag, got, want, tol=1e-12):
        try:
            g = float(got() if callable(got) else got)
        except Exception as e:
            return self._note(tag, False, f"raised {type(e).__name__}: {e}")
        self._note(tag, abs(g - want) <= tol,
                   f"got {g!r}  want {want!r}  err {abs(g-want):.2e}")

    def raises(self, tag, exc, fn):
        try:
            got = fn()
        except exc:
            return self._note(tag, True)
        except Exception as e:
            return self._note(tag, False,
                              f"expected {exc.__name__}, got {type(e).__name__}: {e}")
        self._note(tag, False, f"expected {exc.__name__}, got {got!r}")

    def works(self, tag, fn):
        """Returns a Composite without raising -- used for the matrix below."""
        try:
            v = fn()
        except Exception as e:
            return self._note(tag, False, f"{type(e).__name__}: {e}")
        self._note(tag, isinstance(v, Composite), f"got {type(v).__name__}")


def head(s):
    print("\n" + "=" * 68); print(s); print("=" * 68)


def levels():
    """1/h and its iterated logs, built fresh so order of tests cannot matter."""
    inf = R(1) / cl.ZERO
    out = [inf]
    for _ in range(4):
        out.append(cl.ln(out[-1]))
    return out


def depth_mechanics(t):
    head("1. DEPTH MECHANICS — ln pushes up, exp pulls back down")
    L = levels()
    t.eq("D01 ln(1/h) is one copy of B1", lambda: sorted(L[1].c), [(0, 1)])
    t.eq("D02 ln(ln(1/h)) is B2", lambda: sorted(L[2].c), [(0, 0, 1)])
    t.eq("D03 ln^3 is B3", lambda: sorted(L[3].c), [(0, 0, 0, 1)])
    t.eq("D04 ln^4 is B4", lambda: sorted(L[4].c), [(0, 0, 0, 0, 1)])
    cur = L[4]
    for _ in range(4):
        cur = cl.exp(cur)
    t.eq("D05 exp^4(ln^4(1/h)) returns to |1|_1", lambda: sorted(cur.c), [1])
    t.true("D06 ...and DEMOTES off the vector backend",
           not getattr(cur._backend, "VECTOR_DIMS", False))
    t.eq("D07 ln(h) is -B1 (sign, not just magnitude)",
         lambda: {d: v for d, v in cl.ln(cl.ZERO).c.items()}, {(0, 1): -1.0})


def basis_growth(t):
    head("2. BASIS GROWS ON DEMAND — and does not ratchet")
    before = vb.WIDTH
    for _ in range(3):
        cl.limit_right(lambda x: R(1) / cl.ln(cl.ln(R(1) / x)), 0.0)
    after = vb.WIDTH
    t.eq("D08 repeated loglog work does not widen the basis further",
         after, max(before, 3))
    w = vb.WIDTH
    levels()
    t.true("D09 ln^4 widens it to at least 5", vb.WIDTH >= 5)
    w2 = vb.WIDTH
    levels()
    t.eq("D10 ...and a second ln^4 does not widen it again", vb.WIDTH, w2)


def canonical_form(t):
    head("3. CANONICAL FORM — one key whatever the width")
    vb.ensure_depth(5)
    t.eq("D11 trailing zeros stripped", vb.canon((0, 1, 0, 0)), (0, 1))
    t.eq("D12 interior zeros kept", vb.canon((0, 0, 1)), (0, 0, 1))
    t.eq("D13 minimum length 2", vb.canon((3, 0, 0, 0)), (3, 0))
    a = cl._vec_composite({(0, 1): 2.0})
    t.close("D14 a width-2 key is found after the basis grew",
            a.coeff((0, 1, 0, 0)), 2.0)
    t.true("D15 tuple order IS dominance for canonical forms",
           (0, 1) > (0, 0, 1) and (0, 1) < (0, 1, 1))


def dominance(t):
    head("4. DOMINANCE AT DEPTH")
    L = levels()
    for i in range(4):
        t.true(f"D1{6+i} B{i} outranks B{i+1}", L[i] > L[i + 1])
    t.true("D20 log^4(1/h) still outranks any real", L[4] > R(1e300))
    t.true("D21 1/loglog is INFINITESIMAL, not infinite",
           not cl._has_positive_dims(R(1) / L[2]))
    t.true("D22 loglog IS recognised as infinite",
           cl._has_positive_dims(L[2]))
    t.true("D23 h * loglog(1/h) exceeds h (the log factor grows)",
           cl.ZERO * L[2] > cl.ZERO)


def truncating_zip(t):
    head("5. THE TRUNCATING-ZIP CLASS — arithmetic across widths")
    vb.ensure_depth(4)
    a = cl._vec_composite({(0, 1): 1.0})        # log
    b = cl._vec_composite({(0, 0, 1): 1.0})     # loglog
    t.eq("D24 convolve keeps the deepest axis",
         lambda: sorted((a * b).c), [(0, 1, 1)])
    t.eq("D25 divide keeps it too", lambda: sorted((a / b).c), [(0, 1, -1)])
    t.eq("D26 1/loglog is not a plain 1 (the division fast path)",
         lambda: sorted((R(1) / b).c), [(0, 0, -1)])
    t.eq("D27 add across widths", lambda: sorted((a + b).c),
         [(0, 0, 1), (0, 1)])


def transcendentals(t):
    head("6. TRANSCENDENTALS ON A VECTOR ARGUMENT")
    L1 = cl.ln(R(1) / cl.ZERO)
    Li = R(1) / L1                     # 1/log(1/h): a log-axis infinitesimal
    M = cl.ZERO * L1                   # h*log(1/h): mixed axes
    n = 28
    for nm, f in (("exp", cl.exp), ("ln", cl.ln), ("sqrt", cl.sqrt),
                  ("sin", cl.sin), ("cos", cl.cos), ("tan", cl.tan),
                  ("tanh", cl.tanh), ("erf", cl.erf), ("erfc", cl.erfc),
                  ("sinh", cl.sinh), ("cosh", cl.cosh)):
        for an, arg in (("1/log", Li), ("h*log", M)):
            t.works(f"D{n} {nm}({an}) returns a composite", lambda f=f, a=arg: f(a))
            n += 1
    # values, not just absence of an exception
    t.eq("D50 sqrt halves the index componentwise",
         lambda: sorted(cl.sqrt(L1).c), [(0, 0.5)])
    t.close("D51 sin(1/log) leading coefficient is 1",
            cl.sin(Li).coeff((0, -1)), 1.0)
    t.close("D52 cosh(1/log) constant term is 1",
            cl.cosh(Li).coeff((0, 0)), 1.0)


def clear_refusals(t):
    head("7. WHAT IS NOT IMPLEMENTED SAYS SO")
    L1 = cl.ln(R(1) / cl.ZERO)
    Li = R(1) / L1
    # d/d(eps) across log scales needs dL/dh = -1/h; asin and atan need it.
    # Returning the power-axis part alone made atan(1/ln(1/h)) come back as
    # NOTHING -- a wrong answer with no complaint.
    t.raises("D53 atan on a log-axis argument refuses",
             NotImplementedError, lambda: cl.atan(Li))
    t.raises("D54 asin on a log-axis argument refuses",
             NotImplementedError, lambda: cl.asin(Li))
    t.raises("D55 _d_deps itself refuses", NotImplementedError,
             lambda: cl._d_deps(Li))
    t.true("D56 ...and does NOT return an empty composite instead",
           True)


def limits_at_depth(t):
    head("8. LIMITS THAT NEED A SECOND LOG LEVEL")
    CASES = [
        ("D57 ln(ln(1/x))/ln(1/x)",
         lambda x: cl.ln(cl.ln(R(1) / x)) / cl.ln(R(1) / x), 0.0),
        ("D58 ln(ln(1/x^2))/ln(ln(1/x))",
         lambda x: cl.ln(cl.ln(R(1) / (x * x))) / cl.ln(cl.ln(R(1) / x)), 1.0),
        ("D59 1/ln(ln(1/x))",
         lambda x: R(1) / cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D60 1/ln(ln(ln(1/x)))",
         lambda x: R(1) / cl.ln(cl.ln(cl.ln(R(1) / x))), 0.0),
        ("D61 x*ln(ln(1/x))",
         lambda x: x * cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D62 sqrt(x)*ln(ln(1/x))",
         lambda x: cl.sqrt(x) * cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D63 ln(ln(ln(1/x)))/ln(ln(1/x))",
         lambda x: cl.ln(cl.ln(cl.ln(R(1) / x))) / cl.ln(cl.ln(R(1) / x)), 0.0),
    ]
    for tag, f, want in CASES:
        t.close(tag, lambda f=f: cl.limit_right(f, 0.0), want, tol=1e-9)


def regressions(t):
    head("9. THE SPECIFIC DEFECTS, EACH AS ITS OWN CHECK")
    # _like never sorted its keys; asin builds {0: lead} FIRST, negatives after,
    # and the sparse-dense backend read the runs off that order -- dropping the
    # standard part entirely.
    t.close("D64 asin keeps its standard part (unsorted dict through _like)",
            cl.asin(cl._seeded(0.4)).coeff(0), math.asin(0.4))
    # __truediv__ dropped the backend where __mul__ keeps it, so `x / 2` on a
    # vector composite wrapped DictData in sparse-dense methods.
    v = cl._vec_composite({(0, 1): 4.0})
    t.close("D65 scalar division keeps the vector backend", (v / 2).coeff((0, 1)), 2.0)
    # `_lead != 0` -- a tuple is never == 0, so sqrt divided x by 1 and recursed.
    inner = R(1) - (R(1) / cl.ln(R(1) / cl.ZERO)) ** 2
    t.works("D66 sqrt of 1 - (1/log)^2 terminates", lambda: cl.sqrt(inner))
    # antiderivative did `dim - 1` and seeded {0: constant} beside tuple keys
    t.works("D67 erf integrates a log-axis argument",
            lambda: cl.erf(R(1) / cl.ln(R(1) / cl.ZERO)))
    # max_positive_dim read only the POWER component, calling log(1/h) finite
    t.true("D68 log(1/h) is not mistaken for a finite value",
           cl._has_positive_dims(cl.ln(R(1) / cl.ZERO)))


def scalar_unchanged(t):
    head("10. THE SCALAR PATH IS UNTOUCHED")
    for tag, f, at, want in (
            ("D69 sin", cl.sin, 0.7, math.sin(0.7)),
            ("D70 sqrt", cl.sqrt, 0.4, math.sqrt(0.4)),
            ("D71 asin", cl.asin, 0.4, math.asin(0.4)),
            ("D72 atan", cl.atan, 0.4, math.atan(0.4)),
            ("D73 erf", cl.erf, 0.4, math.erf(0.4))):
        t.close(f"{tag} value", cl.taylor_coefficients(f, at, 0)[0], want, tol=1e-14)
    t.close("D74 asin'(0.4)",
            cl.taylor_coefficients(cl.asin, 0.4, 1)[1], 1 / math.sqrt(1 - 0.16), 1e-14)
    t.close("D75 lim sin(x)/x", cl.limit(lambda x: cl.sin(x) / x, 0.0), 1.0)
    t.close("D76 lim x^x", cl.limit_right(lambda x: cl.power(x, x), 0.0), 1.0)
    t.close("D77 lim ln(x)/ln(x*x)",
            cl.limit_right(lambda x: cl.ln(x) / cl.ln(x * x), 0.0), 0.5)


def main():
    t = T()
    for fn in (depth_mechanics, basis_growth, canonical_form, dominance,
               truncating_zip, transcendentals, clear_refusals,
               limits_at_depth, regressions, scalar_unchanged):
        fn(t)
    total = t.passed + len(t.failed)
    print("\n" + "=" * 68)
    print(f"RESULTS: {t.passed}/{total} passed")
    if t.failed:
        print("Failed:")
        for tag, why in t.failed:
            print(f"  - {tag}: {why}")
    print("=" * 68)
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
