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
        # Print the numbers on PASS as well as on fail.  Showing got/want only
        # when something breaks means a passing suite reveals nothing, and a
        # tolerance can be widened until the tick appears with no trace of it
        # in the output.  A green run has to be auditable from its own log.
        if ok:
            self.passed += 1
            print(f"  OK   {tag}" + (f"   {detail}" if detail else ""))
        else:
            self.failed.append((tag, detail))
            print(f"  FAIL {tag}\n         {detail}")

    def true(self, tag, cond, detail=""):
        # cond may be a thunk: an argument evaluated at the call site raises
        # BEFORE this try can catch it, which turned a failing check into a
        # crashed suite and hid every check after it.
        try:
            self._note(tag, bool(cond() if callable(cond) else cond), detail)
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
                   f"got {g!r}  want {want!r}  err {abs(g-want):.2e}  tol {tol:.1e}")

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
                  ("sinh", cl.sinh), ("cosh", cl.cosh),
                  ("atan", cl.atan), ("asin", cl.asin)):
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


def inverse_trig_on_log_axis(t):
    head("7. asin AND atan ACROSS SCALES")
    L1 = cl.ln(R(1) / cl.ZERO)
    u = R(1) / L1                      # 1/log(1/h): small, on the log axis
    M = cl.ZERO * L1                   # h*log(1/h): mixed axes
    # Maclaurin composition, which needs no derivative and no antiderivative
    # and so works on any axis.  The derivative route cannot generalise here:
    # integral of h**k * B1**m dh is not a single term unless k == -1, so there
    # is no dimension shift to apply.
    for tag, f, want in (
            ("atan", cl.atan, {(0, -1): 1.0, (0, -3): -1 / 3,
                               (0, -5): 1 / 5, (0, -7): -1 / 7}),
            ("asin", cl.asin, {(0, -1): 1.0, (0, -3): 1 / 6,
                               (0, -5): 3 / 40, (0, -7): 15 / 336})):
        got = {d: v for d, v in f(u).c.items()}
        for k, want_v in sorted(want.items(), reverse=True):
            t.close(f"D53 {tag}(1/log) coefficient at {k}",
                    got.get(k, 0.0), want_v, tol=1e-14)
    t.close("D54 tan(atan u) recovers u", cl.tan(cl.atan(u)).coeff((0, -1)), 1.0)
    t.close("D55 sin(asin u) recovers u", cl.sin(cl.asin(u)).coeff((0, -1)), 1.0)
    t.close("D56 atan on MIXED axes, leading term",
            cl.atan(M).coeff((-1, 1)), 1.0)
    t.close("D57 asin on MIXED axes, third-order term",
            cl.asin(M).coeff((-3, 3)), 1 / 6)

    # d/d(eps) across scales: dB1/dh = -1/h, dB_k/dh = -(1/h)*B1^-1...B_{k-1}^-1,
    # so one term becomes a SUM, one per axis it touches.  Returning only the
    # power-axis part made atan(1/ln(1/h)) come back as NOTHING.
    t.close("D58 d/de[h] = 1", cl._d_deps(cl.ZERO).coeff(0), 1.0)
    t.close("D59 d/de[h*h] = 2h", cl._d_deps(cl.ZERO * cl.ZERO).coeff(-1), 2.0)
    t.close("D60 d/de[1/B1] = (1/h)B1^-2", cl._d_deps(u).coeff((1, -2)), 1.0)
    t.close("D61 d/de[B1] = -1/h", cl._d_deps(L1).coeff(1), -1.0)
    t.close("D62 d/de[1/B1^2] = 2(1/h)B1^-3",
            cl._d_deps(R(1) / (L1 * L1)).coeff((1, -3)), 2.0)
    t.true("D63 a pure log term does NOT differentiate to nothing",
           len(cl._d_deps(u).c) > 0)
    # Differentiating loses an order.  Without this the bound is one too
    # generous and asin(sin x) claims order 12 on 11 sound ones -- caught only
    # by the identity suite before, which is too far from the cause.
    _sc = cl.sin(cl._seeded(0.7) ** 2)
    t.eq("D63b d/de records the lost order",
         lambda: cl._d_deps(_sc)._complete, _sc._complete - 1)
    t.true("D63c an EXACT input stays exact under d/de",
           cl._d_deps(cl._seeded(0.7) ** 3)._complete is None)

    # A TRUNCATED series must say so, on EVERY axis.  The Maclaurin route
    # skipped this and asin(h)/atan(h) reported _complete = None -- exact --
    # while being a 15-term truncation reaching order 29.  Fixing it on the
    # power axis alone still left the log axis claiming exactness, because
    # _infinitesimal_terms filtered on _dim_order, which reads only the power
    # component and calls (0,-1) order zero.
    for tag, r in (("D63d asin(h)", cl.asin(cl.ZERO)),
                   ("D63e atan(h)", cl.atan(cl.ZERO)),
                   ("D63f atan on the log axis", cl.atan(u)),
                   ("D63g asin on the log axis", cl.asin(u))):
        t.true(f"{tag} states a bound rather than claiming exact",
               r._complete is not None)
        top = max(cl._lead_order(d) for d in r.c) if r.c else 0
        # guard the comparison: a failing bound is None, and `int <= None`
        # raises, which would crash the run instead of reporting the failure
        t.true(f"{tag} bound covers what it produced",
               r._complete is not None and top <= r._complete,
               f"produced up to order {top}, bound {r._complete!r}")
    # _lead_order is the axis-aware order; _dim_order reads the power axis only
    t.eq("D63h _lead_order sees a log-axis infinitesimal",
         cl._lead_order((0, -1)), 1)
    t.eq("D63i _lead_order on a scalar dim", cl._lead_order(-3), 3)
    t.true("D63j _infinitesimal_terms includes log-axis terms",
           len(cl._infinitesimal_terms(u)) == 1)


def limits_at_depth(t):
    head("8. LIMITS THAT NEED A SECOND LOG LEVEL")
    CASES = [
        ("D64 ln(ln(1/x))/ln(1/x)",
         lambda x: cl.ln(cl.ln(R(1) / x)) / cl.ln(R(1) / x), 0.0),
        ("D65 ln(ln(1/x^2))/ln(ln(1/x))",
         lambda x: cl.ln(cl.ln(R(1) / (x * x))) / cl.ln(cl.ln(R(1) / x)), 1.0),
        ("D66 1/ln(ln(1/x))",
         lambda x: R(1) / cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D67 1/ln(ln(ln(1/x)))",
         lambda x: R(1) / cl.ln(cl.ln(cl.ln(R(1) / x))), 0.0),
        ("D68 x*ln(ln(1/x))",
         lambda x: x * cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D69 sqrt(x)*ln(ln(1/x))",
         lambda x: cl.sqrt(x) * cl.ln(cl.ln(R(1) / x)), 0.0),
        ("D70 ln(ln(ln(1/x)))/ln(ln(1/x))",
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


def vector_keyed_standard_part(t):
    """A vector-keyed value whose infinitesimal part is EMPTY.

    Every suite here builds values that carry a real log-axis term, so the
    one shape that was never tested is the shape the arithmetic produces
    constantly: subtract the log term back off and what is left is a plain
    number wearing vector keys.  Its dimension is the VECTOR zero (0, 0),
    and `d != 0` -- a tuple against an int -- called that infinitesimal.
    sin then built its series with a = st(x) while h still held the same
    standard part, so sin(5) returned sin(10): -0.5440211108893698.  More
    terms converged harder onto the wrong number, which is what ruled out
    truncation as the cause.
    """
    H = Composite({-1: 1.0})
    L = cl.ln(1 / H)
    S = (5.0 + L) - L                      # 5.0, vector-keyed, no infinitesimal
    t.true("D78 the vector zero is a tuple key",
           lambda: all(isinstance(k, tuple) for k in S.c))
    t.true("D79 ...and reads as having NO infinitesimal part",
           lambda: not cl._has_infinitesimal_part(S))
    # EVERY transcendental, not a sample: sin and cos were found by hand, and
    # the same gate turned out to be breaking nine others -- ln, sqrt, tan,
    # asin, acos, erf, erfc, normal_cdf and atan -- which went from 4/15 to
    # 15/15 correct on this shape without any of them being touched.  A loop
    # over the whole set is what makes that visible; a sample is what let it
    # sit unnoticed.
    S04 = (0.4 + L) - L
    for tag, fn, arg, want in (
            ("D80 sin",   cl.sin,   S, math.sin(5.0)),
            ("D81 cos",   cl.cos,   S, math.cos(5.0)),
            ("D82 exp",   cl.exp,   S, math.exp(5.0)),
            ("D83 sqrt",  cl.sqrt,  S, math.sqrt(5.0)),
            ("D87 ln",    cl.ln,    S, math.log(5.0)),
            ("D88 tan",   cl.tan,   S04, math.tan(0.4)),
            ("D89 atan",  cl.atan,  S04, math.atan(0.4)),
            ("D90 asin",  cl.asin,  S04, math.asin(0.4)),
            ("D91 acos",  cl.acos,  S04, math.acos(0.4)),
            ("D92 sinh",  cl.sinh,  S04, math.sinh(0.4)),
            ("D93 cosh",  cl.cosh,  S04, math.cosh(0.4)),
            ("D94 tanh",  cl.tanh,  S04, math.tanh(0.4)),
            ("D95 erf",   cl.erf,   S04, math.erf(0.4)),
            ("D96 erfc",  cl.erfc,  S04, math.erfc(0.4)),
            ("D97 normal_cdf", cl.normal_cdf, S04,
             0.5 * math.erfc(-0.4 / math.sqrt(2.0)))):
        t.close(f"{tag}(vector-keyed)",
                lambda fn=fn, arg=arg: fn(arg).st(), want, 1e-13)
    t.true("D84 |cos| <= 1 -- the old cos returned -1.133",
           lambda: abs(cl.cos(S).st()) <= 1.0)
    # raising the term count must not move a correct answer
    t.close("D85 sin is term-count stable", lambda: cl.sin(S, terms=40).st(),
            math.sin(5.0), 1e-13)
    # and the scalar spelling of the same value is unchanged
    t.close("D86 scalar 5.0 agrees", lambda: cl.sin(Composite({0: 5.0})).st(),
            math.sin(5.0), 1e-13)


def r1_on_a_vector_zero(t):
    """R1 -- |0|_d becomes |1|_(d-1) -- when d is a vector dimension.

    This sat one layer below every transcendental: the uplift itself spelled
    the shift `dims[0] - 1`, so a bare Python zero meeting ANY vector-keyed
    composite raised TypeError.  `0.0 + ln(1/h)` could not be evaluated.

    The shift goes on the POWER axis, not the leading axis.  (-1, 1) is below
    (0, 1) lexicographically, so the result is always infinitesimal; taking
    the leading axis instead would send the zero at (0, 1) to (0, 0), which is
    finite, and R1 must not produce a finite number.
    """
    H = Composite({-1: 1.0})
    L = cl.ln(1 / H)
    Z = L - L                                  # wholly zero, vector-keyed
    t.true("D98 L - L is wholly zero", lambda: cl._is_wholly_zero(Z))
    t.eq("D99 0.0 + L uplifts to h + L",
         lambda: str(0.0 + L), "|1|_(0,1) + |1|_(-1,0)")
    # LITERAL expected value, not str(0.0 + L): comparing one live result to
    # another live result asserts nothing -- both sides move together, so the
    # check passes whatever the code does.  It also crashes rather than
    # reports, because `want` is evaluated at the call site.
    t.eq("D100 R(0) + L agrees with the bare zero",
         lambda: str(R(0) + L), "|1|_(0,1) + |1|_(-1,0)")
    t.eq("D101 1/(h*L) lands at (1,-1)", lambda: str(1 / Z), "|1|_(1,-1)")
    t.true("D102 the uplift is strictly infinitesimal",
           lambda: cl._dim_negative(sorted((Z + 1).c)[0]))
    # R2: a zero among nonzero terms is a TERM, not an operand -- unchanged
    t.eq("D103 Z*2 stays a zero term", lambda: str(Z * 2), "|0|_(0,1)")
    # and the scalar spellings must not have moved at all
    t.eq("D104 scalar 1/0 unchanged", lambda: str(1 / R(0)), "|1|\u2081")
    t.eq("D105 scalar 0.0 + h unchanged",
         lambda: str(0.0 + H), "|2|\u208b\u2081")


def vector_keyed_with_infinitesimal(t):
    """Vector keys AND a genuine infinitesimal part -- the other untested shape.

    D78-D97 cover a vector-keyed value whose infinitesimal part is EMPTY, which
    every transcendental now handles by returning early.  This is the shape
    that still reaches the series machinery, and exp was wrong on it long after
    the early-return gate was fixed: its own filter kept the scalar spelling,
    so exp(0.4 + h*ln(1/h)) multiplied the standard part in twice and returned
    e**0.8.  The standard part being right is NOT enough evidence here -- the
    coefficients are checked too, because a double-count that lands on dim zero
    is exactly what a .st()-only check misses.
    """
    H = Composite({-1: 1.0})
    L = cl.ln(1 / H)
    V = 0.4 + H * L                            # 0.4 + h*ln(1/h), dim (-1, 1)
    t.true("D106 V has vector keys", lambda: any(isinstance(k, tuple) for k in V.c))
    t.true("D107 ...and a real infinitesimal part",
           lambda: cl._has_infinitesimal_part(V))
    e4 = math.exp(0.4)
    t.close("D108 exp(V) standard part", lambda: cl.exp(V).st(), e4, 1e-13)
    # exp(0.4 + u) = e**0.4 * (1 + u + u**2/2 + ...) with u = h*ln(1/h)
    for n, tag in ((1, "D109"), (2, "D110"), (3, "D111")):
        t.close(f"{tag} exp(V) coeff at (-{n},{n})",
                lambda n=n: cl.exp(V).coeff((-n, n)),
                e4 / math.factorial(n), 1e-12)
    for tag, fn, want in (("D112 sin", cl.sin, math.sin(0.4)),
                          ("D113 cos", cl.cos, math.cos(0.4)),
                          ("D114 tan", cl.tan, math.tan(0.4)),
                          ("D115 sqrt", cl.sqrt, math.sqrt(0.4)),
                          ("D116 tanh", cl.tanh, math.tanh(0.4))):
        t.close(f"{tag}(V)", lambda fn=fn: fn(V).st(), want, 1e-13)
    # the scalar analogue must be untouched
    xs = Composite({0: 0.4, -1: 1.0})
    for n, tag in ((0, "D117"), (1, "D118"), (2, "D119")):
        t.close(f"{tag} scalar exp(0.4+h) coeff {n}",
                lambda n=n: cl.exp(xs).coeff(-n),
                e4 / math.factorial(n), 1e-13)


def main():
    t = T()
    for fn in (depth_mechanics, basis_growth, canonical_form, dominance,
               truncating_zip, transcendentals, inverse_trig_on_log_axis,
               limits_at_depth, regressions, scalar_unchanged,
               vector_keyed_standard_part, r1_on_a_vector_zero,
               vector_keyed_with_infinitesimal):
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
