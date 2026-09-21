#!/usr/bin/env python3
# Composite Machine — cancellation forensics
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Is the formula bad, or is the problem hard?  Tested where it can be wrong.

`composite/forensics.py` answers that question with two numbers: kappa, the
problem's own condition number, taken from the exact derivative the composite
returns alongside the value; and a forward error bound for the code as
written.  Both come from one seeded pass, with no step size anywhere.

The tool is only worth anything if the verdict is right and the bound holds,
so this file attacks both.

WHAT IT LOOKS FOR, in order of how much it can embarrass us:

  X1  verdicts against the textbook classification.  Nine formulas, four of
      them exact rewrites of the other five, each of which the numerical
      analysis literature already calls stable or not.  Disagreeing with it
      is the tool being wrong, not the literature.

  X2  the bound holds.  `predicted` must not fall below the error actually
      observed against a 50-digit oracle, at every point tested.  This is the
      falsifiable claim, and it caught two real defects while being written:
      a magnitude-only metric that missed `ln(1 + x)` entirely, and a total
      cancellation reported as the mildest verdict instead of the harshest.

  X3  kappa against condition numbers known in closed form.

  X4  kappa predicting a measured response.  Move the input by one ulp and
      the output must move by kappa times as much -- an end-to-end check that
      kappa means what the report says it means.

  X5  the expressed zero.  A data 0.0 in an additive position converts under
      R1, so the value survives and the derivative does not.  float64 cannot
      see this: the number is identical either way.  X5 pins the detection,
      the line attribution, and the derivative that actually moved.

  X6  correlated rounding.  A first-order bound that tracks magnitudes alone
      must assume every rounding is independent, so it double-counts the
      error a compensated algorithm deliberately cancels and calls Kahan's
      expm1 and log1p unstable.  Both read `unstable formula` before the
      error form was made affine.  X6 is the guard: a tool that defames the
      stable-algorithm literature is worse than no tool.

  X7  absorption, which a cancellation metric cannot see.  In `ln(1 + x)` the
      amplification of `1 + x` is 1.0 and the formula is still ruinous.  It
      was reported `stable` with a 2.22e-16 bound against a true error of
      8.27e-08 until absorption was scored separately.

  X8  two propagation rules that were wrong and are easy to get wrong again:
      reverse division, which double-swapped its operands and charged the
      numerator's error to the denominator, libelling a stable formula whose
      observed error was 0; and a computed zero, whose relative error is
      unbounded rather than absent.

  X9  what it refuses, and what it must not crash on.

Reference values are pinned from mpmath 1.3.0 at 60 decimal digits, evaluated
at the float input -- `mp.mpf(x)`, not `mp.mpf(repr(x))`, which re-reads the
decimal string as an exact decimal and so lands on a different point.  That
mistake reported an error of 2.21e-05 for a formula that is accurate to one
rounding, and the pinning below is at the float.  They are re-checked against
mpmath when it is importable.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, NotRepresentableError
from composite.forensics import (
    audit, compare, table, report, F, EPS,
    STABLE, ILL_CONDITIONED, UNSTABLE, DERIVATIVE_LOST, REFUSED,
)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9

try:
    import mpmath as mp
    mp.mp.dps = 60
except ImportError:
    mp = None

# mpmath 1.3.0 at 60 digits, evaluated at the float input.
TRUE = {
    "versine": {
        0.01: 0.49999583334722219742, 0.0001: 0.49999999958333333347,
        1e-05: 0.49999999999583333333, 1e-06: 0.49999999999995833333,
        1e-08: 0.49999999999999999583,
    },
    "expm1": {
        0.01: 0.010050167084168057752, 1e-06: 1.0000005000001666215e-6,
        1e-10: 1.0000000000500000364e-10,
    },
    "log1p": {
        0.01: 0.0099503308531680830543, 1e-06: 9.9999950000033328783e-7,
        1e-10: 9.9999999995000003644e-11,
    },
    "sqrtdiff": {
        100.0: 0.049875621120890270219, 1000000.0: 0.00049999987500006249996,
        10000000000.0: 4.999999999875e-6,
    },
    "tansin": {
        0.01: 5.0001250054168856551e-7, 0.0001: 5.000000012500000773e-13,
        1e-06: 5.0000000000012493212e-19,
    },
    "root": {10000.0: -0.00010000000100000002, 100000000.0: -1.0000000000000001e-8},
}

ORACLE = {
    "versine": lambda z: (1 - mp.cos(z)) / z ** 2,
    "expm1": lambda z: mp.e ** z - 1,
    "log1p": lambda z: mp.log(1 + z),
    "sqrtdiff": lambda z: mp.sqrt(z + 1) - mp.sqrt(z),
    "tansin": lambda z: mp.tan(z) - mp.sin(z),
    "root": lambda b: (-b + mp.sqrt(b ** 2 - 4)) / 2,
}


# --------------------------------------------------------------------------
# the formulas.  Each pair computes the same function; the stable member is an
# exact rewrite, never a truncated series -- the bound covers rounding and has
# nothing to say about terms a series left out.
# --------------------------------------------------------------------------

def versine_naive(x):
    return (1 - F.cos(x)) / (x * x)


def versine_stable(x):
    return 2 * F.sin(x / 2) ** 2 / (x * x)


def expm1_naive(x):
    return F.exp(x) - 1


def expm1_kahan(x):
    u = F.exp(x)
    return x * (u - 1) / F.ln(u)


def log1p_naive(x):
    return F.ln(1 + x)


def log1p_kahan(x):
    u = 1 + x
    return x * F.ln(u) / (u - 1)


def sqrtdiff_naive(x):
    return F.sqrt(x + 1) - F.sqrt(x)


def sqrtdiff_stable(x):
    return 1 / (F.sqrt(x + 1) + F.sqrt(x))


def tansin_naive(x):
    return F.tan(x) - F.sin(x)


def root_naive(b):
    return (-b + F.sqrt(b * b - 4)) / 2


def root_stable(b):
    return -2 / (b + F.sqrt(b * b - 4))


def amplifier(x):
    return 1 / (1 - x)


#  family, formula, spelling is stable?, points
BATTERY = [
    ("versine", "(1-cos x)/x^2", versine_naive, False, [1e-2, 1e-4, 1e-6]),
    ("versine", "2 sin^2(x/2)/x^2", versine_stable, True, [1e-2, 1e-4, 1e-6]),
    ("expm1", "exp(x)-1", expm1_naive, False, [1e-6, 1e-10]),
    ("expm1", "Kahan x(u-1)/ln u", expm1_kahan, True, [1e-2, 1e-6, 1e-10]),
    ("log1p", "ln(1+x)", log1p_naive, False, [1e-6, 1e-10]),
    ("log1p", "Kahan x ln(u)/(u-1)", log1p_kahan, True, [1e-2, 1e-6, 1e-10]),
    ("sqrtdiff", "sqrt(x+1)-sqrt(x)", sqrtdiff_naive, False, [1e6, 1e10]),
    ("sqrtdiff", "1/(sqrt(x+1)+sqrt x)", sqrtdiff_stable, True, [1e2, 1e6, 1e10]),
    ("tansin", "tan x - sin x", tansin_naive, False, [1e-4, 1e-6]),
    ("root", "(-b+sqrt(b^2-4))/2", root_naive, False, [1e8]),
    ("root", "-2/(b+sqrt(b^2-4))", root_stable, True, [1e4, 1e8]),
]


def _true(family, x):
    """The pinned 60-digit value, re-checked against mpmath when available."""
    v = TRUE[family][x]
    if mp is not None:
        fresh = float(ORACLE[family](mp.mpf(x)))
        if v != 0.0 and abs(fresh - v) / abs(v) > 1e-15:
            raise AssertionError("pinned reference drifted: %s at %g: %r vs %r"
                                 % (family, x, v, fresh))
    return v


# ==========================================================================

def x1_verdicts(t):
    head("X1  verdicts against the textbook classification")
    for family, label, fn, stable, pts in BATTERY:
        for x in pts:
            a = audit(fn, x, name=label)
            got = a.verdict
            want = STABLE if stable else UNSTABLE
            t.true("X1 %-22s at %-8.0e -> %s" % (label, x, want),
                   got == want,
                   "got %r  want %r   kappa %.3g  predicted %.3g"
                   % (got, want, a.kappa, a.predicted))


def x2_bound_holds(t):
    head("X2  the bound holds -- predicted >= observed, against 60-digit truth")
    worst = (0.0, None)
    for family, label, fn, stable, pts in BATTERY:
        for x in pts:
            a = audit(fn, x, name=label)
            truth = _true(family, x)
            obs = abs(a.value - truth) / abs(truth)
            ok = a.predicted >= obs * (1 - 1e-9)
            t.true("X2 %-22s at %-8.0e bound" % (label, x), ok,
                   "predicted %.3e  observed %.3e  ratio %.3g"
                   % (a.predicted, obs, (obs / a.predicted) if a.predicted else 0.0))
            if a.predicted and math.isfinite(a.predicted) and obs:
                r = obs / a.predicted
                if r > worst[0]:
                    worst = (r, "%s at %g" % (label, x))
    t.true("X2.tight  the bound is not vacuous somewhere", worst[0] > 0.01,
           "tightest %s: observed/predicted = %.3g" % (worst[1], worst[0]))


def x3_kappa_closed_form(t):
    head("X3  kappa against condition numbers known in closed form")
    #  f(x) = x^n           kappa = n
    #  f(x) = exp(x)        kappa = |x|
    #  f(x) = sqrt(x)       kappa = 1/2
    #  f(x) = ln(x)         kappa = 1/|ln x|
    #  f(x) = sin(x)        kappa = |x cot x|
    for label, fn, at, want in (
            ("x**3", lambda x: x ** 3, 2.0, 3.0),
            ("x**7", lambda x: x ** 7, 1.5, 7.0),
            ("exp(x)", lambda x: F.exp(x), 0.5, 0.5),
            ("exp(x)", lambda x: F.exp(x), 4.0, 4.0),
            ("sqrt(x)", lambda x: F.sqrt(x), 7.0, 0.5),
            ("ln(x)", lambda x: F.ln(x), 3.0, 0.91023922662683732),
            ("sin(x)", lambda x: F.sin(x), 1.2, 0.46653548324184588)):
        t.close("X3 kappa %-9s at %-5g" % (label, at),
                audit(fn, at).kappa, want, tol=1e-12)


def x4_kappa_predicts_response(t):
    head("X4  kappa predicts a measured response: perturb the input, watch the output")
    #  The probe size is per case and neither arbitrary nor uniform.  Too large
    #  and second-order terms enter; too small and the OUTPUT's own quantisation
    #  dominates -- a one-ulp probe on x**5 at 3 moves the result by six ulps,
    #  so a single ulp of output rounding is a 16% error in the ratio, and the
    #  first version of this check failed at 4.74 against kappa = 5 for that
    #  reason alone.  Each delta below is small against the curvature scale and
    #  large enough to move the output by many ulps.
    for label, fn, plain, at, delta, tol in (
            ("1/(1-x)", amplifier, lambda x: 1 / (1 - x), 1 - 1e-12, None, 2e-2),
            ("exp(x)", lambda x: F.exp(x), math.exp, 12.0, 1e-9, 1e-6),
            ("ln(x)", lambda x: F.ln(x), math.log, 1.0000001, 1e-12, 1e-4),
            ("tan(x)", lambda x: F.tan(x), math.tan, 1.5, 1e-9, 1e-6),
            ("x**5", lambda x: x ** 5, lambda x: x ** 5, 3.0, 1e-9, 1e-6),
            ("sqrt(x)", lambda x: F.sqrt(x), math.sqrt, 7.0, 1e-9, 1e-6)):
        k = audit(fn, at).kappa
        if delta is None:                   # a pole this close admits one ulp only
            nudged = math.nextafter(at, at * 2 + 1.0)
        else:
            nudged = at * (1.0 + delta)
        din = abs(nudged - at) / abs(at)
        v0, v1 = plain(at), plain(nudged)
        dout = abs(v1 - v0) / abs(v0)
        measured = dout / din
        rel = abs(measured - k) / max(abs(k), 1e-300)
        t.true("X4 %-9s at %-12g kappa predicts the response" % (label, at),
               rel <= tol,
               "kappa %.8g  measured %.8g  rel diff %.2e  tol %g"
               % (k, measured, rel, tol))


def x5_expressed_zero(t):
    head("X5  the expressed zero -- value survives, derivative does not")

    def with_zero(x):
        baseline = 0.0
        return (x * x - baseline) / x

    def without_zero(x):
        return (x * x) / x

    a, b = audit(with_zero, 3.0), audit(without_zero, 3.0)
    t.close("X5.01 value is untouched by the expressed zero", a.value, 3.0, tol=1e-12)
    t.close("X5.02 the clean spelling has f'(3) = 1", b.derivative, 1.0, tol=1e-12)
    t.true("X5.03 the zero moved the derivative off 1",
           abs(a.derivative - 1.0) > 0.1,
           "f'(3) = %.17g with the zero, %.17g without" % (a.derivative, b.derivative))
    t.exact("X5.04 verdict names it", a.verdict, DERIVATIVE_LOST)
    t.exact("X5.05 the clean spelling is stable", b.verdict, STABLE)
    t.true("X5.06 exactly one data-zero recorded", len(a.zeros) == 1,
           "found %d" % len(a.zeros))
    z = a.zeros[0]
    t.true("X5.07 attributed to this file", os.path.basename(z.file) == "test_forensics.py",
           "got %s" % z.where)
    t.true("X5.08 attributed to the line that wrote it",
           "baseline" in z.src and "x * x" in z.src, "source: %r" % z.src)
    # A zero-crossing INTERMEDIATE is not a data zero.  ln(S/K) at S = K is
    # exactly 0 and still carries the derivative 1/S, and flagging it reported
    # `derivative corrupted` for three correct Black-Scholes sensitivities.
    def d1(S, K, r, sig, T):
        return (F.ln(S / K) + (r + sig * sig / 2) * T) / (sig * F.sqrt(T))

    a = audit(lambda v: d1(v, 100.0, 0.05, 0.2, 1.0), 100.0, name="d1 wrt S")
    t.close("X5.10 zero-crossing intermediate keeps its derivative",
            a.derivative, 0.05, tol=1e-14)        # 1/(S sigma sqrt(T))
    t.exact("X5.11 and is not blamed as a data zero", a.verdict, STABLE)
    t.true("X5.12 no data-zero finding recorded for it", len(a.zeros) == 0,
           "found %d" % len(a.zeros))
    a = audit(lambda v: d1(100.0, 100.0, 0.05, v, 1.0), 0.2, name="d1 wrt sigma")
    t.close("X5.13 same formula, another variable", a.derivative, 4.25, tol=1e-14)

    t.true("X5.09 float64 shows nothing at all",
           (3.0 * 3.0 - 0.0) / 3.0 == (3.0 * 3.0) / 3.0,
           "both spellings give %.17g in float64" % ((3.0 * 3.0 - 0.0) / 3.0))


def x6_correlated_rounding(t):
    head("X6  correlated rounding -- compensated algorithms must read stable")
    for label, fn, pts in (("expm1 Kahan", expm1_kahan, [1e-2, 1e-6, 1e-10, 1e-14]),
                           ("log1p Kahan", log1p_kahan, [1e-2, 1e-6, 1e-10, 1e-14])):
        for x in pts:
            a = audit(fn, x, name=label)
            t.true("X6 %-12s at %-8.0e stays near the rounding floor"
                   % (label, x), a.predicted <= 100 * EPS,
                   "predicted %.3e = %.1f eps" % (a.predicted, a.predicted / EPS))
    # The naive spellings at the same points must NOT be excused.
    for label, fn, x in (("exp(x)-1", expm1_naive, 1e-10),
                         ("ln(1+x)", log1p_naive, 1e-10)):
        a = audit(fn, x, name=label)
        t.true("X6 %-12s at %-8.0e is still blamed" % (label, x),
               a.verdict == UNSTABLE and a.predicted > 1e-8,
               "predicted %.3e  verdict %r" % (a.predicted, a.verdict))


def x7_absorption(t):
    head("X7  absorption -- amplification 1.0, and ruinous anyway")
    a = audit(log1p_naive, 1e-10, name="ln(1+x)")
    t.true("X7.01 `1 + x` amplification really is ~1",
           abs((1.0 + 1e-10) + 0.0) and
           (1.0 + 1e-10) / abs(1.0 + 1e-10) < 1.0 + 1e-9,
           "(|a|+|b|)/|a+b| = %.6f -- a cancellation metric sees nothing"
           % ((1.0 + 1e-10) / (1.0 + 1e-10)))
    t.exact("X7.02 verdict is still unstable", a.verdict, UNSTABLE)
    t.true("X7.03 an absorption finding was recorded",
           len(a.of_kind("absorption")) >= 1,
           "found %d absorption, %d cancellation"
           % (len(a.of_kind("absorption")), len(a.of_kind("cancellation"))))
    truth = _true("log1p", 1e-10)
    obs = abs(a.value - truth) / abs(truth)
    t.true("X7.04 and the bound covers the real error", a.predicted >= obs,
           "predicted %.3e  observed %.3e  (a magnitude metric predicted %.3e)"
           % (a.predicted, obs, EPS))


def x8_propagation_rules(t):
    head("X8  two propagation rules that were wrong")
    # Reverse division: 1/(sqrt(x+1)+sqrt x) is exact to the last bit here.
    for x in (1e2, 1e6, 1e10):
        a = audit(sqrtdiff_stable, x, name="1/(sqrt(x+1)+sqrt x)")
        truth = _true("sqrtdiff", x)
        obs = abs(a.value - truth) / abs(truth)
        t.true("X8 reverse division at %-8.0e not libelled" % x,
               a.verdict == STABLE and a.predicted <= 100 * EPS,
               "predicted %.3e  observed %.3e  verdict %r"
               % (a.predicted, obs, a.verdict))
    # A computed zero: relative error is unbounded, not absent.
    a = audit(versine_naive, 1e-9, name="(1-cos x)/x^2")
    t.close("X8.zero value collapsed to 0", a.value, 0.0, tol=0.0)
    t.true("X8.zero predicted is infinite, not nan",
           math.isinf(a.predicted), "predicted %r" % a.predicted)
    t.exact("X8.zero verdict is the harshest, not the mildest", a.verdict, UNSTABLE)
    t.true("X8.zero kappa is undefined rather than infinite",
           math.isnan(a.kappa), "kappa %r" % a.kappa)


def x9_refusals_and_robustness(t):
    head("X9  refusals, and what it must not crash on")
    a = audit(lambda x: F.exp(1 / x), 0.0, name="exp(1/x) at 0")
    t.true("X9.01 exp of an unbounded argument is refused or flagged",
           a.verdict in (REFUSED, UNSTABLE, ILL_CONDITIONED),
           "verdict %r  %s" % (a.verdict, type(a.exception).__name__
                               if a.exception else ""))
    a = audit(lambda x: 1 / (x - 5.0), 5.0, name="1/(x-5) at 5")
    t.true("X9.02 a pole does not crash the audit",
           isinstance(a.verdict, str),
           "verdict %r  value %r" % (a.verdict, a.value))
    a = audit(lambda x: R(3.0), 2.0, name="constant")
    t.close("X9.03 a constant has kappa 0", a.kappa, 0.0, tol=1e-15)
    t.exact("X9.04 a constant is stable", a.verdict, STABLE)
    a = audit(lambda x: x, 0.0, name="identity at 0")
    t.true("X9.05 identity at the origin does not crash",
           isinstance(a.verdict, str), "verdict %r" % a.verdict)
    # Long chains must not blow up the affine form.
    def chain(x):
        acc = x
        for _ in range(200):
            acc = (acc + 1) * 1.000001 - 1
        return acc
    a = audit(chain, 1.0, name="200-step chain")
    t.true("X9.06 a 200-step chain stays bounded",
           math.isfinite(a.predicted) and a.predicted < 1e-6,
           "predicted %.3e after 200 steps" % a.predicted)
    t.true("X9.07 report() renders without raising",
           isinstance(report(audit(versine_naive, 1e-5)), str))
    t.true("X9.08 table() renders without raising",
           isinstance(table(compare({"v": versine_naive}, 1e-5)), str))


def run_all():
    t = Suite()
    for fn in (x1_verdicts, x2_bound_holds, x3_kappa_closed_form,
               x4_kappa_predicts_response, x5_expressed_zero,
               x6_correlated_rounding, x7_absorption, x8_propagation_rules,
               x9_refusals_and_robustness):
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
