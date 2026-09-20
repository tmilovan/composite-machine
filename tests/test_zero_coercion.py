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


def run_all():
    t = Suite()
    for fn in (z1_expressed, z2_family, z3_substitution, z4_laws,
               z5_expressed_vs_absent, z6_naming):
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
