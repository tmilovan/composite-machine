#!/usr/bin/env python3
# Composite Machine — expressed zeros, absent terms, and the line between
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""A written zero is an EXPRESSED zero, and an expressed zero is an infinitesimal.

That is R1 and R6, and it is the system's content rather than a cost it
imposes:

    R1  a wholly zero operand converts:  0_d -> 1_(d-1)
    R6  there is no additive identity.  R(5) + 0_0 -> 5_0 + 1_-1, not 5_0

The distinction §0 draws is between a zero and an ABSENCE, not between a zero
and a zero that came from data.  A masked entry holding 0.0 holds zero; a node
at 0.0 is a node at zero; a coefficient that is 0.0 is a coefficient that is
zero.  All written, all expressed, all converting.

What is absent is a term nobody put there -- `d.get(k)` returning nothing, an
accumulator with nothing added yet.  The idiom for that is Composite({}), and
NOT a written 0.  Getting this backwards is what `d.get(k, 0.0)` did for a
Pade coefficient that did not exist: it manufactured a zero the mathematics
never expressed, and the lateral Borel integral returned 3224 for a value of
0.697.  Section Z5 pins that down.

This file was briefly written against the opposite rule -- a bare scalar zero
coercing to NOTHING -- which was reverted.  The premise was wrong (the event
R1 records is the EXPRESSION of the zero, and writing it is that event), and
the consequence was worse than the problem: a keyboard-reachable additive
identity is a conventional ring with an extra symbol attached, and two zeros
obeying different laws is worse than one obeying one.
"""
import math
import os
import random
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import StandardPartUndefinedError
from composite.composite_lib import _r1
from composite.composite_lib import (Composite, R, ZERO, is_zero, is_nothing,
                                     is_vanishing)
from composite.transseries import _d_composite
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

E = Composite({})                  # NOTHING -- an absence, not a number
Z0 = Composite({0: 0.0})           # an expressed zero
C = Composite({0: 5.0, -1: 3.0})


def _d(x):
    return dict(sorted(x.coeffs_dict().items()))


def _quiet(fn, *a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a)


# =============================================================================
def z1_expressed(t):
    head("Z1  a written zero is expressed, and converts")
    want = "|5|\u2080 + |1|\u208b\u2081"
    for i, (lbl, got) in enumerate((
            ("R(5) + 0        -- the spec's own example",
             str(_quiet(lambda: R(5) + 0))),
            ("R(5) + <0_0>    -- same number, written out", str(R(5) + Z0)),
            ("R(5) + R(0)     -- and the canonical spelling", str(R(5) + R(0)))), 1):
        t.true(f"Z1.0{i} {lbl}: {got}", got == want, f"got {got!r} want {want!r}")
    t.true(f"Z1.04 <0_0> + <0_0> -> 2_-1: {Z0 + Z0}",
           str(Z0 + Z0) == "|2|\u208b\u2081", f"{Z0 + Z0}")
    t.true(f"Z1.05 c * 0 converts rather than annihilating: "
           f"{_d(_quiet(lambda: C * 0))}",
           _d(_quiet(lambda: C * 0)) == {-2: 3.0, -1: 5.0},
           f"{_d(_quiet(lambda: C * 0))}")
    # 1 - 1 != 0, which is the same statement
    t.true(f"Z1.10 R(1) - R(1) is a wholly zero |0|_0", cl._is_wholly_zero(R(1) - R(1)),
           f"{R(1) - R(1)}")
    t.true(f"Z1.11 and it converts on next use: {(R(1) - R(1)) + R(5)}",
           str((R(1) - R(1)) + R(5)) == "|5|\u2080 + |1|\u208b\u2081",
           f"{(R(1) - R(1)) + R(5)}")
    t.true("Z1.12 ZERO - ZERO == ZERO**2", (ZERO - ZERO) == ZERO ** 2,
           f"{ZERO - ZERO}")


# =============================================================================
def z2_family(t):
    head("Z2  NOTHING, |0|_d, ZERO -- and the order they sit in")
    rows = [("<0_0> == ZERO", Z0 == ZERO, True, "both read through R1"),
            ("ZERO == 0.0", ZERO == 0.0, True, "0.0 is an expressed zero"),
            ("R(0) == 0", R(0) == 0, True, ""),
            ("E == 0.0", E == 0.0, False, "NOTHING is not a number"),
            ("E == ZERO", E == ZERO, False, ""),
            ("E == Composite({})", E == Composite({}), True, "")]
    for i, (lbl, got, want, why) in enumerate(rows, 1):
        t.true(f"Z2.0{i} {lbl} is {want}" + (f"  ({why})" if why else ""),
               got == want, f"got {got}, want {want}")

    t.true("Z2.10 NOTHING is not an additive identity dressed up -- it is an "
           "absence, and adding nothing is a no-op only because nothing is "
           "not a number (spec R6)", (E + R(5)) == R(5), f"{E + R(5)}")

    vals = {"E": E, "<0_0>": Z0, "ZERO": ZERO, "0.0": 0.0,
            "<0_-3>": Composite({-3: 0.0})}
    bad = [f"{ka}=={kb}=={kc}" for ka, a in vals.items() for kb, b in vals.items()
           for kc, c in vals.items()
           if (a == b) and (b == c) and not (a == c)]
    t.true(f"Z2.20 transitivity holds across all {len(vals)}^3 triples of zeros",
           not bad, f"{bad[:3]}")
    bad = [f"{ka},{kb}" for ka, a in vals.items() for kb, b in vals.items()
           if (a < b) and (b < a)]
    t.true("Z2.21 antisymmetry: never both a < b and b < a", not bad, f"{bad[:3]}")
    bad = [f"{ka},{kb}" for ka, a in vals.items() for kb, b in vals.items()
           if not ((a < b) or (b < a) or (a == b))]
    t.true("Z2.22 totality: every pair is <, > or ==", not bad, f"{bad[:3]}")


# =============================================================================
def z3_substitution(t):
    head("Z3  substitution: equal operands must stay equal under every f")
    # This is what caught d/dh reading coefficients raw: |0|_0 and ZERO are the
    # same number -- each converts under R1, so they are operationally
    # indistinguishable -- yet d/dh gave NOTHING for one and |1|_0 for the
    # other.  Equal in, unequal out.
    pairs = [("<0_0> vs ZERO", Z0, ZERO), ("E vs Composite({})", E, Composite({}))]
    fs = [("+c", lambda x: x + C), ("*c", lambda x: x * C),
          ("c-", lambda x: C - x), ("d/dh", _d_composite)]
    for pname, a, b in pairs:
        if not (a == b):
            t.true(f"Z3.00 {pname} equal (precondition)", False, "not equal")
            continue
        for fname, f in fs:
            fa, fb = _quiet(f, a), _quiet(f, b)
            t.true(f"Z3 {pname} under {fname}: {_d(fa)} == {_d(fb)}", fa == fb,
                   f"{_d(fa)} vs {_d(fb)}")


# =============================================================================
def z4_laws(t):
    head("Z4  the algebraic laws, zeros of every kind in the pool")
    rng = random.Random(20260920)

    def mk():
        r = rng.random()
        if r < 0.12:
            return Composite({})
        if r < 0.20:
            return Composite({rng.choice([0, -1, -2]): 0.0})
        if r < 0.26:
            x = Composite({rng.choice([0, -1]): rng.choice([1.0, 2.0, -3.0])})
            return x - x
        return Composite({rng.choice([0, -1, -2, 1]): rng.choice(
            [1.0, -2.0, 0.5, 3.0])})

    laws = {"a*1 = a": lambda a, b, c: a * R(1) == a,
            "a+b = b+a": lambda a, b, c: a + b == b + a,
            "a*b = b*a": lambda a, b, c: a * b == b * a,
            "(a+b)+c = a+(b+c)": lambda a, b, c: (a + b) + c == a + (b + c),
            "(a*b)*c = a*(b*c)": lambda a, b, c: (a * b) * c == a * (b * c),
            "(a+b)*c = a*c+b*c": lambda a, b, c: (a + b) * c == a * c + b * c}
    N = 2000
    fails = {k: 0 for k in laws}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N):
            a, b, c = mk(), mk(), mk()
            for k, f in laws.items():
                try:
                    if not f(a, b, c):
                        fails[k] += 1
                except Exception:
                    fails[k] += 1
    for k, n in fails.items():
        t.true(f"Z4 {k:<20} {N - n}/{N} hold", n == 0, f"{n} failures")


# =============================================================================
def z5_expressed_vs_absent(t):
    head("Z5  the real bug: a zero that was MANUFACTURED, not expressed")
    coeffs = {0: 2.0, 2: 5.0}          # no coefficient at 1 -- it does not exist

    # WRONG: invent one.  The scalar is written, so it is expressed, and it
    # converts.  Dimension 0 stays right and every derivative moves.
    acc = R(7)
    for k in (0, 1, 2):
        acc = _quiet(lambda a=acc, k=k: a + coeffs.get(k, 0.0))
    t.true(f"Z5.01 `d.get(k, 0.0)` for an ABSENT coefficient deposits |1|_-1: "
           f"{_d(acc)}", _d(acc).get(-1, 0.0) == 1.0, f"{_d(acc)}")

    # RIGHT: an absent term is absent.
    acc2 = R(7)
    for k in (0, 1, 2):
        v = coeffs.get(k)
        if v is not None:
            acc2 = acc2 + v
    t.true(f"Z5.02 `d.get(k)` and skip deposits nothing: {_d(acc2)}",
           _d(acc2) == {0: 14.0}, f"{_d(acc2)}")

    # and the accumulator idiom, which is the same mistake one line earlier
    t.true(f"Z5.10 `acc = 0` seeds an expressed zero: "
           f"{_d(_quiet(lambda: sum([C, C])))}",
           _d(_quiet(lambda: sum([C, C]))).get(-1) == 7.0,
           f"{_d(_quiet(lambda: sum([C, C])))}")
    t.true(f"Z5.11 Composite({{}}) seeds an absence: "
           f"{_d(sum([C, C], Composite({})))}",
           sum([C, C], Composite({})) == C * 2.0,
           f"{_d(sum([C, C], Composite({})))}")

    # the warning names the fix
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = C + 0
    msg = str(w[0].message) if w else ""
    t.true("Z5.20 and the bare-zero warning says so: " + msg[:46] + "...",
           "d.get(k)" in msg and "Composite({})" in msg, msg[:90])


# =============================================================================
def z6_naming(t):
    head("Z6  is_zero / is_vanishing / is_nothing")
    cases = [("E", E, True, True, False), ("<0_0>", Z0, True, False, True),
             ("c-c", C - C, True, False, True), ("ZERO", ZERO, False, False, False),
             ("c", C, False, False, False), ("0.0", 0.0, True, False, False)]
    for lbl, v, wz, wn, wv in cases:
        t.true(f"Z6 {lbl:<7} is_zero={is_zero(v)} is_nothing={is_nothing(v)} "
               f"is_vanishing={is_vanishing(v)}",
               (is_zero(v), is_nothing(v), is_vanishing(v)) == (wz, wn, wv),
               f"want {(wz, wn, wv)}")


def z7_equality_is_identity(t):
    head("Z7  equality is identity, not magnitude")
    # Reported from the REPL: (1+ZERO)-ZERO compared equal to R(1).  The left
    # side holds |0|_-1, a cancelled infinitesimal that R2 retains; the right
    # side has no grade -1 term at all.  __eq__ went through _compare, whose
    # read_dim returns 0.0 for an absent dimension, so the term was invisible.
    b = (1 + ZERO) - ZERO
    a = R(1)
    # NOT t.dims here: it drops zero coefficients ("if x != 0.0"), so it is
    # blind to the term this whole section is about.  Read the map directly.
    t.exact("Z7.01 the cancelled infinitesimal is retained (R2)",
            b.coeffs_dict(), {-1: 0.0, 0: 1.0})
    t.exact("Z7.02 and R(1) has no grade -1 term at all",
            a.coeffs_dict(), {0: 1.0})
    t.true("Z7.03 so they are NOT equal", a != b, "a=%s  b=%s" % (a, b))
    t.true("Z7.04 nor the other way round", not (b == a), "b == a")
    t.true("Z7.04b != is exactly the negation of ==",
           (a != b) == (not (a == b)) and (a != a) == (not (a == a)),
           "__ne__ had its own _compare and disagreed with __eq__")

    # What must NOT change: R1 still makes the two spellings of a zero one
    # number, and scalars still compare.
    t.true("Z7.05 |0|_0 == |1|_-1, the two spellings of a zero",
           Composite({0: 0.0}) == ZERO, "R1 is applied before comparing")
    t.true("Z7.06 R(1) == 1", R(1) == 1)
    t.true("Z7.07 R(1) == R(1)", R(1) == R(1))
    t.true("Z7.08 ZERO == ZERO", ZERO == ZERO)
    t.true("Z7.09 R(2) != R(3)", R(2) != R(3))
    t.true("Z7.10 a composite equals its own rebuild",
           b == Composite(b.coeffs_dict()), "%s" % b)

    # The ORDER is unaffected: a zero term changes no magnitude.
    t.true("Z7.11 the zero term does not make b bigger", not (b > a), "b > a")
    t.true("Z7.12 nor smaller", not (b < a), "b < a")
    t.true("Z7.13 R(1) < R(2) still", R(1) < R(2))
    t.true("Z7.14 ZERO < R(1) still", ZERO < R(1))

    # The price, pinned deliberately.  `==` is no longer the tie case of the
    # order: b and a are incomparable by magnitude AND unequal.  Anything
    # written `if x == y ... elif x < y ... else` now lands in `else`.
    t.true("Z7.15 TRICHOTOMY IS LOST, on purpose",
           (not (a < b)) and (not (a > b)) and (not (a == b)),
           "neither dominates and they are not equal -- this is the "
           "documented cost of Z7.03, not a regression")

    # DECIDED: zero TERMS are not resolved before comparing.
    #
    # R1 sends a zero OPERAND at grade d to |1|_(d-1), so |0|_-1 and |1|_-2
    # look like they should be the same number, and resolving before
    # comparison would also make equality agree with the order on this pair
    # (it currently says a > b while equality says unequal).  It was not done,
    # because resolution collides: in {0:1, -1:0, -2:1} the cancellation at
    # grade -1 lands on a real term at -2, so `1 + 0h + h**2` would become
    # `1 + 2h**2` -- a cancelled infinitesimal doubling a real one.  R2 already
    # says a zero among nonzero terms is a TERM and nothing happens to it;
    # these checks hold the comparison to that.
    cancelled = Composite({0: 1.0, -1: 0.0})     # 1, with an infinitesimal cancelled
    real_deep = Composite({0: 1.0, -2: 1.0})     # 1, with a real one two grades down
    t.true("Z7.17 a cancelled infinitesimal is NOT resolved onto the next grade",
           cancelled != real_deep, "%s vs %s" % (cancelled, real_deep))
    t.true("Z7.18 the order still separates them, by the real term",
           real_deep > cancelled, "the -2 term is 1.0 against nothing")
    t.exact("Z7.19 and R1 on the term ALONE would have resolved it",
            _r1(Composite({-1: 0.0})).coeffs_dict(), {-2: 1.0})
    t.true("Z7.20 which is why resolving would collide",
           Composite({0: 1.0, -1: 0.0, -2: 1.0}) != Composite({0: 1.0, -2: 2.0}),
           "resolution would make 1 + 0h + h**2 equal 1 + 2h**2")

    # Equality stays stable under arithmetic: equal things remain equal.
    for lbl, f in (("+ ZERO", lambda x: x + ZERO),
                   ("* 2", lambda x: x * 2),
                   ("* x", lambda x: x * x),
                   ("exp", lambda x: cl.exp(x))):
        p_, q_ = R(2), R(2)
        t.true("Z7.16 equal inputs stay equal through %s" % lbl, f(p_) == f(q_))


def z8_r1_reaches_the_transcendentals(t):
    head("Z8  R1 applies to a transcendental's argument, not just to arithmetic")
    # Arithmetic ran its operands through _operands, which ends in _r1.  The
    # transcendentals did not, so a cancellation arrived as |0|_0 instead of
    # |1|_-1 and the infinitesimal was thrown away at the door:
    #
    #     z = R(1) - R(1)          |0|_0
    #     z * R(2)   -> |2|_-1     R1 applied
    #     sqrt(z)    -> |0|_0      R1 skipped
    #
    # The physical case is the Dirac ground state, E = sqrt(1 - (Z*alpha)**2),
    # whose square-root branch point at Z*alpha = 1 is what says the solution
    # ceases to exist there.  With R1 skipped the argument reached sqrt as a
    # plain zero and the answer was 0 -- the branch point silently gone.
    z = R(1) - R(1)
    t.exact("Z8.01 a cancellation is still |0|_0 before use", z.coeffs_dict(),
            {0: 0.0})
    t.exact("Z8.02 and R1 converts it when applied", _r1(z).coeffs_dict(),
            {-1: 1.0})

    # THE PHYSICS CASE.
    Za = R(1.0)
    E = cl.sqrt(R(1) - Za * Za)
    t.exact("Z8.03 sqrt(1-(Z*alpha)^2) at Z*alpha=1 keeps the branch point",
            E.coeffs_dict(), {-0.5: 1.0})
    t.true("Z8.04 the exponent is one HALF grade -- that IS the branch point",
           list(E.coeffs_dict())[0] == -0.5, "got %s" % E)
    # Oracle computed, not transcribed: the first version of this hand-typed
    # 0.1410673597706336 from a 12-digit printout of 0.141067359797, and the
    # check failed against a correct answer.
    for za in (0.9, 0.99, 0.999):
        t.close("Z8.05 and below it the value is unchanged (Za=%g)" % za,
                cl.sqrt(R(1) - R(za) * R(za)),
                math.sqrt(1.0 - za * za), tol=1e-15)

    # The invariant, stated once: a cancellation and an explicit ZERO are the
    # same operand, so every function must answer them identically.
    for nm in ("sqrt", "exp", "ln", "sin", "cos", "tan", "sinh", "cosh",
               "tanh", "atan", "asin", "acos", "erf", "erfc"):
        f = getattr(cl, nm)
        try:
            a, b = f(z), f(ZERO)
            t.true("Z8.06 %-5s(R(1)-R(1)) == %s(ZERO)" % (nm, nm), a == b,
                   "%s  vs  %s" % (str(a)[:34], str(b)[:34]))
        except Exception as e:
            t.true("Z8.06 %-5s(R(1)-R(1)) == %s(ZERO)" % (nm, nm), False,
                   "raised %s: %s" % (type(e).__name__, str(e)[:40]))

    # And the series are the real ones, not an artefact of the conversion.
    t.close("Z8.07 exp(1-1) has the h coefficient of exp", cl.exp(z).d(1),
            1.0, tol=1e-12)
    t.exact("Z8.08 sin(1-1) leads at grade -1, as sin(h) does",
            min(cl.sin(z).coeffs_dict()), -11.0)
    t.close("Z8.09 cos(1-1) has cos's h^2 coefficient",
            cl.cos(z).coeffs_dict()[-2.0], -0.5, tol=1e-12)

    # What must NOT have changed.
    t.close("Z8.10 sqrt(4) is still 2", cl.sqrt(R(4)), 2.0, tol=1e-15)
    t.close("Z8.11 exp(1) is still e", cl.exp(R(1)), math.e, tol=1e-12)
    t.close("Z8.12 ln(2) unchanged", cl.ln(R(2)), math.log(2), tol=1e-12)
    t.close("Z8.13 sqrt(4+h) standard part", cl.sqrt(R(4) + ZERO), 2.0, tol=1e-15)
    t.close("Z8.14 sqrt(4+h) first coefficient is 1/(2*sqrt 4)",
            cl.sqrt(R(4) + ZERO).d(1), 0.25, tol=1e-12)
    t.raises("Z8.15 sqrt(-1) still refuses", ValueError, lambda: cl.sqrt(R(-1)))
    t.raises("Z8.16 ln(-1) still refuses", ValueError, lambda: cl.ln(R(-1)))
    t.true("Z8.17 NOTHING still propagates through sqrt",
           is_nothing(cl.sqrt(Composite({}))))
    # Z8.18 used to assert exp(NOTHING) is NOTHING.  That was wrong under R6
    # and Z9 now pins the correct rule; see z9_nothing_keeps_the_constant.
    t.close("Z8.18 exp(NOTHING) is 1, not NOTHING (see Z9)",
            cl.exp(Composite({})), 1.0, tol=1e-15)

    # DELIBERATE CHANGE, pinned so it is not mistaken for a leak: ln(0.0) used
    # to raise.  R(0.0) IS |1|_-1, and ln of an infinitesimal belongs on the
    # log axis, so it now returns that instead of refusing.  ln(-1) still
    # refuses, so only the zero case moved.
    r = cl.ln(R(0.0))
    t.true("Z8.19 ln(0.0) returns a log-axis object rather than raising",
           any(isinstance(d, tuple) for d in r.coeffs_dict()),
           "got %s -- ln(h), not a refusal" % r)
    t.true("Z8.20 and ln(0.0) agrees with ln(ZERO)", r == cl.ln(ZERO))


def z9_nothing_keeps_the_constant(t):
    head("Z9  f(nothing) keeps the term that does not carry the argument")
    # R6, spec line 107: "a + nothing = a holds because adding nothing is a
    # no-op".  Apply it term by term to a series and the answer falls out:
    #
    #     exp(nothing) = 1 + nothing + nothing + ... = 1
    #     sin(nothing) =     nothing + nothing + ... = nothing
    #
    # Every function short-circuited to Composite({}) before the series was
    # formed, so exp(nothing) was nothing.  Removing the short-circuit outright
    # is worse, not better: the scalar fast path below it reads st(nothing) as
    # 0 and returns Composite({0: f(0)}) -- an EXPRESSED zero -- which R1 then
    # converts on next use.  That is how acos(nothing) came back as
    # |1.5708|_0 + |-1|_-1, a spurious infinitesimal in a constant.  The
    # short-circuit is kept and returns the constant term instead.
    N = Composite({})
    for nm, want in (("exp", 1.0), ("cos", 1.0), ("cosh", 1.0),
                     ("erfc", 1.0), ("acos", math.pi / 2)):
        t.close("Z9.01 %-5s(nothing) = its constant term" % nm,
                getattr(cl, nm)(N), want, tol=1e-15)
    for nm in ("sin", "sinh", "tan", "tanh", "atan", "asin", "erf", "sqrt", "ln"):
        t.true("Z9.02 %-5s(nothing) = nothing, every term carries x" % nm,
               is_nothing(getattr(cl, nm)(N)),
               "got %s" % getattr(cl, nm)(N))

    # The failure mode the short-circuit prevents, pinned by its symptom.
    a = cl.acos(N)
    t.exact("Z9.03 acos(nothing) carries no spurious infinitesimal",
            sorted(a.coeffs_dict()), [0.0])
    t.true("Z9.04 and it is not an expressed zero anywhere",
           all(v != 0.0 for v in a.coeffs_dict().values()), "%s" % a)

    # It must be the CONSTANT, not f(0) evaluated on a zero: those differ,
    # because R(0.0) is |1|_-1 and carries an infinitesimal that nothing does not.
    t.true("Z9.05 exp(nothing) and exp(R(0.0)) are different numbers",
           cl.exp(N) != cl.exp(R(0.0)),
           "%s  vs  %s" % (cl.exp(N), str(cl.exp(R(0.0)))[:34]))
    t.exact("Z9.06 exp(nothing) is exactly 1, one term",
            cl.exp(N).coeffs_dict(), {0: 1.0})

    # R6 itself, restated on the operations that already obeyed it.
    t.true("Z9.07 R(5) + nothing = R(5)", (R(5) + N) == R(5))
    t.true("Z9.08 R(5) * nothing = nothing", is_nothing(R(5) * N))
    t.true("Z9.09 nothing + nothing = nothing", is_nothing(N + N))


def z10_division_by_an_unbounded_multi_axis_divisor(t):
    head("Z10  dividing by a divisor that spans axes and has no standard part")
    # Reachable only since Z8: ln(ZERO) used to raise, and now returns the
    # log-axis object |-1|_(0,1).  Adding an ordinary infinitesimal to it gives
    # a divisor spanning BOTH axes, dominated by ln(h), so it has no standard
    # part -- and __truediv__ asked for one anyway:
    #
    #     if _spans_multiple_axes(b) and b.st() != 0.0:
    #
    # st() does not answer that question for an unbounded value, it raises, so
    # the division itself crashed.  All twelve divisions by ln(h) + h failed
    # this way, although h / (ln(h) + h) is an ordinary infinitesimal.
    h = Composite({-1: 1.0})
    lg = cl.ln(ZERO)
    den = lg + h
    t.exact("Z10.01 the divisor really does span two axes",
            sorted(map(str, den.coeffs_dict())), ["(-1.0, 0)", "(0, 1)"])
    t.raises("Z10.02 and genuinely has no standard part",
             StandardPartUndefinedError, den.st)

    q = h / den
    t.close("Z10.03 h/(ln h + h) is an infinitesimal, not a crash",
            q.st(), 0.0, tol=0.0)
    lead = min(q.coeffs_dict(), key=lambda d: (-d[0], -d[1]))
    t.exact("Z10.04 and it leads where h/ln(h) does", lead, (-1.0, -1))
    t.exact("Z10.05 which is exactly h/ln(ZERO)'s only grade",
            list((h / lg).coeffs_dict()), [(-1.0, -1)])

    t.exact("Z10.06 the divisor over itself is exactly 1",
            (den / den).coeffs_dict(), {0: 1.0})
    t.close("Z10.07 R(1)/(ln h + h) is also an infinitesimal",
            (R(1) / den).st(), 0.0, tol=0.0)
    t.raises("Z10.08 INF/(ln h + h) stays unbounded, so st still refuses",
             StandardPartUndefinedError, (cl.INF / den).st)

    # No grade may appear twice in any of them.
    for lbl, v in (("h/den", q), ("R1/den", R(1) / den), ("den/den", den / den),
                   ("INF/den", cl.INF / den)):
        ks = list(v.coeffs_dict())
        t.true("Z10.09 %-8s has no repeated grade" % lbl,
               len(ks) == len(set(ks)), "grades %s" % ks[:6])

    # The guard must ANSWER, not raise -- that is the whole fix.
    t.true("Z10.10 the guard returns False rather than raising",
           cl._expandable_about_st(den) is False)
    t.true("Z10.11 and still returns True where the series IS usable",
           cl._expandable_about_st(R(2) + h) is True)
    t.true("Z10.12 and False for a zero standard part",
           cl._expandable_about_st(h) is False)

    # The reciprocal-series path it guards must be unaffected.
    t.close("Z10.13 1/(1+h) still expands about its standard part",
            (R(1) / (R(1) + h)).st(), 1.0, tol=1e-15)
    t.close("Z10.14 and 2/(2+h) likewise",
            (R(2) / (R(2) + h)).st(), 1.0, tol=1e-15)


def _dT():
    """An inert |0|_0 from a cancellation, over a half-integer series.

    This is the shape the physics hit: a remnant heat capacity whose
    denominator carries a retained zero at grade 0 (R2) and a long run of
    fractional grades below it.
    """
    terms = {0: 0.0}
    for k in range(1, 13):
        terms[-k / 2.0] = (-1) ** k * (k + 1) * 0.7
    return Composite(terms)


def z11_deconvolve_emits_unique_grades(t):
    head("Z11  division must not emit the same grade twice")
    # create_from_terms documents its precondition: "callers hand over sorted,
    # unique dimensions ... validating here is work on the hottest path in the
    # library".  deconvolve sorted but never merged, so it broke that contract.
    #
    # The loop drops r_dim from the remainder to stop a grade repeating on the
    # NEXT iteration, but each iteration adds b.dims + q_dim back in, so a
    # dropped grade can be reintroduced later, become the maximum again, and be
    # emitted a second time.  Integer grades rarely expose it.  Fractional ones
    # do, because b.dims + q_dim then interleaves with the remainder instead of
    # landing on grades already occupied.
    #
    # Left unmerged the duplicates reached coeffs_dict(), which builds a dict
    # and keeps only the LAST of them.  That is silent data loss, not a display
    # artefact: grade -1.5 held [-1.607142857142857, 1.4285714285714286] and
    # the dict reported only the second.
    h = Composite({-1: 1.0})
    dT = _dT()
    q = h / dT
    dims, vals = q._backend.to_arrays(q._data)
    keys = [float(d) for d in dims]
    t.true("Z11.01 no grade appears twice in raw storage",
           len(keys) == len(set(keys)),
           "%d entries, %d distinct" % (len(keys), len(set(keys))))
    t.true("Z11.02 so coeffs_dict drops nothing",
           len(q.coeffs_dict()) == len(keys),
           "dict %d vs storage %d" % (len(q.coeffs_dict()), len(keys)))
    t.close("Z11.03 grade -1.5 holds the SUM of what was emitted there",
            q.coeffs_dict()[-1.5], -0.17857142857142855, tol=1e-15)

    # The quotient must actually be one: (h/dT) * dT == h.
    back = (q * dT).coeffs_dict()
    t.close("Z11.04 (h/dT)*dT recovers h at grade -1", back.get(-1.0, 0.0),
            1.0, tol=1e-12)
    for g in (-0.5, -1.5, -2.0):
        t.close("Z11.05 and nothing at grade %g" % g, back.get(g, 0.0),
                0.0, tol=1e-12)

    # The inert zero is the trigger, so the two spellings must now agree on
    # every grade they share.
    noz = Composite({k: v for k, v in _dT().coeffs_dict().items() if k != 0})
    a, b = (h / dT).coeffs_dict(), (h / noz).coeffs_dict()
    # Neither a pure absolute nor a pure relative tolerance works here, and
    # both were tried.  The quotient spans 21 decades: leading terms are O(1),
    # the tail reaches 2.6e+07, and between them sit dust terms at 4e-14.
    # Absolute 1e-12 demands 19 significant digits of the tail and failed at
    # 7.45e-09 on a coefficient of -1.4e+07 (relative 5.31e-16, one ulp).
    # Relative 1e-14 then failed at 1.25e-01 -- on the dust, where two values
    # of 4e-14 differ by 5e-15 and the ratio means nothing.
    #
    # The mixed form is the honest one, with the absolute floor taken from the
    # computation's own scale rather than chosen: eps * (largest coefficient)
    # is the size of a rounding error anywhere in this series.
    shared = set(a) & set(b)
    scale = max(max(abs(v) for v in a.values()), max(abs(v) for v in b.values()))
    floor = 10.0 * 2.220446049250313e-16 * scale
    off = [(abs(a[k] - b[k]) - 1e-12 * max(abs(a[k]), abs(b[k])) - floor, k)
           for k in shared]
    t.true("Z11.06 with and without the inert |0|_0 agree to float precision",
           all(d <= 0.0 for d, _ in off),
           "%d shared grades, scale %.2e, floor %.2e, worst excess %.2e"
           % (len(shared), scale, floor, max((d for d, _ in off), default=0.0)))

    # Ordinary division is untouched -- the merge only fires when a grade
    # really was emitted twice.
    t.close("Z11.07 R(6)/R(3)", R(6) / R(3), 2.0, tol=1e-15)
    t.close("Z11.08 h/h", h / h, 1.0, tol=1e-15)
    t.close("Z11.09 (1+h)/(1+h)", (R(1) + h) / (R(1) + h), 1.0, tol=1e-15)
    t.close("Z11.10 1/(1+h) about its standard part",
            (R(1) / (R(1) + h)).st(), 1.0, tol=1e-15)


def run_all():
    t = Suite()
    for fn in (z1_expressed, z2_family, z3_substitution, z4_laws,
               z5_expressed_vs_absent, z6_naming, z7_equality_is_identity,
               z8_r1_reaches_the_transcendentals,
               z9_nothing_keeps_the_constant,
               z10_division_by_an_unbounded_multi_axis_divisor,
               z11_deconvolve_emits_unique_grades):
        try:
            fn(t)
        except Exception as e:
            t._note(f"{fn.__name__} ABORTED", False, f"{type(e).__name__}: {e}")
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
