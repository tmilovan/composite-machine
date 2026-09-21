"""Cancellation forensics -- is the formula bad, or is the problem hard?

float64 loses significance without saying so.  The useful question is not
whether a formula lost digits but whether it *had* to, because the two answers
call for opposite actions: rewrite the expression, or accept the error and stop
trying.  Telling them apart needs the condition number of the problem beside a
forward error bound for the code.

The composite supplies both.  Seeding the input with an infinitesimal gives
f'(x) from the same pass that computes f(x), which is what the condition number
needs, and supplies the elementary-function derivatives the error bound needs
at each node.  That matters more than it looks: the usual substitute for f' is
a finite difference, which is itself a subtraction of nearly equal numbers, so
it fails hardest on exactly the formulas worth diagnosing.  Section 4 measures
that.

Every prediction below is checked against mpmath at 50 digits, evaluated at the
float input rather than at a decimal re-reading of it.

    python demos/composite_forensics.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpmath as mp

from composite.forensics import audit, compare, table, report, F, EPS

mp.mp.dps = 50


def rule(title):
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


# ---------------------------------------------------------------------------
# Each pair computes the same mathematical function.  The stable spellings are
# exact rewrites, not truncated series -- the bound covers rounding, and would
# have nothing to say about terms a series left out.
# ---------------------------------------------------------------------------

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


def root_naive(b):
    """Smaller root of t^2 + b t + 1, the way it is usually written."""
    return (-b + F.sqrt(b * b - 4)) / 2


def root_stable(b):
    """Same root, reached through the product of the roots."""
    return -2 / (b + F.sqrt(b * b - 4))


def amplifier(x):
    """Nothing wrong with this formula.  The problem is simply hard."""
    return 1 / (1 - x)


def with_data_zero(x):
    """A zero that came from data, not from the mathematics."""
    baseline = 0.0
    return (x * x - baseline) / x


PAIRS = [
    ("versine   x = 1e-5", 1e-5,
     [("(1 - cos x)/x^2", versine_naive), ("2 sin^2(x/2)/x^2", versine_stable)],
     lambda z: (1 - mp.cos(z)) / z ** 2),
    ("expm1     x = 1e-10", 1e-10,
     [("exp(x) - 1", expm1_naive), ("Kahan  x(u-1)/ln u", expm1_kahan)],
     lambda z: mp.e ** z - 1),
    ("log1p     x = 1e-10", 1e-10,
     [("ln(1 + x)", log1p_naive), ("Kahan  x ln(u)/(u-1)", log1p_kahan)],
     lambda z: mp.log(1 + z)),
    ("quadratic b = 1e8", 1e8,
     [("(-b + sqrt(b^2-4))/2", root_naive), ("-2/(b + sqrt(b^2-4))", root_stable)],
     lambda b: (-b + mp.sqrt(b ** 2 - 4)) / 2),
]


def main():
    rule("1.  ONE FUNCTION, TWO SPELLINGS        predicted vs observed, oracle mpmath/50")

    for label, at, pair, ref in PAIRS:
        rows = compare(dict(pair), at,
                       reference=lambda v, _r=ref: float(_r(mp.mpf(v))))
        print()
        print(table(rows, "  " + label))

    print()
    print("  Same function in each pair, and the verdict separates the spelling that")
    print("  had to lose digits from the one that chose to.  `predicted` is an upper")
    print("  bound, so it sits above `observed`; the verdict is the claim, not the")
    print("  digits.  Both Kahan forms cancel their own rounding exactly, which a")
    print("  magnitude-only bound would miss and call unstable.")

    rule("2.  WHEN IT IS NOT THE FORMULA'S FAULT")
    at = 1 - 1e-12
    rows = compare({"1/(1 - x)": amplifier}, at,
                   reference=lambda v: float(1 / (1 - mp.mpf(v))))
    print()
    print(table(rows))
    a = rows[0][0]
    print()
    print("  Evaluated exactly as written, this formula is accurate -- observed error")
    print("  is at the rounding floor.  What kappa = %.3g says is what happens to the" % a.kappa)
    print("  input, so nudge x by a single ulp and watch the output:")
    nudged = math.nextafter(at, 2.0)
    v0, v1 = 1 / (1 - at), 1 / (1 - nudged)
    print()
    print("      x            %.17g" % at)
    print("      x + 1 ulp    %.17g      (relative move %.2e)"
          % (nudged, abs(nudged - at) / at))
    print("      1/(1-x)      %.17g" % v0)
    print("      1/(1-x')     %.17g      (relative move %.2e)"
          % (v1, abs(v1 - v0) / abs(v0)))
    print()
    print("  One ulp in, %.0e out. That is the problem, not the code, and the verdict"
          % (abs(v1 - v0) / abs(v0)))
    print("  is `%s`: no rewrite helps. Carry more precision, or reformulate" % a.verdict)
    print("  the problem so the subtraction never happens.")

    rule("3.  THE FAILURE float64 CANNOT SEE")
    a = audit(with_data_zero, 3.0, name="(x*x - baseline)/x")
    print()
    print(report(a))
    print()
    print("  In float64 this returns 3.0 and nothing looks wrong, because nothing is")
    print("  wrong with the value.  What broke is the sensitivity: the function is")
    print("  f(x) = x, so f'(3) = 1, and the expressed zero converted to an")
    print("  infinitesimal and carried the derivative to %.17g." % a.derivative)
    print("  Any stability radius, gradient or sensitivity read off that number is")
    print("  wrong, with no symptom anywhere in the value to warn you.")

    rule("4.  WHY THIS NEEDS THE COMPOSITE")
    print()
    print("  kappa needs f'(x).  The standard substitute is a finite difference --")
    print("  a subtraction of nearly equal numbers, which is the very fault being")
    print("  diagnosed.  Same formula, same points, three ways of getting f':")
    print()
    print("   x         kappa TRUE      composite       central difference (h = 1e-6 / 1e-8)")
    print("   " + "-" * 84)
    g = lambda z: (1 - mp.cos(z)) / z ** 2
    fl = lambda z: (1 - math.cos(z)) / (z * z)
    for x in (1e-3, 1e-4, 1e-5):
        k_true = abs(x * float(mp.diff(g, mp.mpf(x))) / float(g(mp.mpf(x))))
        k_comp = audit(versine_naive, x).kappa
        fd = [abs(x * ((fl(x + h) - fl(x - h)) / (2 * h)) / fl(x)) for h in (1e-6, 1e-8)]
        print("   %-9.0e %-15.6g %-15.6g %.3g / %.3g" % (x, k_true, k_comp, *fd))
    print()
    print("  The composite derivative degrades too -- it differentiates the program as")
    print("  written, float64 noise and all -- but stays orders of magnitude closer,")
    print("  and kappa stays small enough that the verdict holds.  The difference")
    print("  quotient does not: at x = 1e-5 it is off by eight orders, and it is")
    print("  wrongest exactly where you most need it to be right.")

    rule("SUMMARY")
    print()
    print("  Two numbers from one pass, no step size:")
    print("    kappa      what the problem forces you to lose")
    print("    predicted  what this spelling of it loses")
    print("  Their ratio is the verdict, and the ledger names the line.")
    print()


if __name__ == "__main__":
    main()
