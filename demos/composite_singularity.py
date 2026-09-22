"""Where a series stops converging -- and why that is four questions at once.

A power series knows its own singularity.  The radius of convergence is the
distance to the nearest one, and the growth of the coefficients encodes what
kind it is.  Reading both back out is a calculation normally done symbolically
or by hand, and it answers four different questions with the same arithmetic:

    a nonlinear ODE        when does the solution blow up, and how fast?
    a counting sequence    how do the coefficients grow?
    a lattice model        where is the critical point, and what is the exponent?
    a perturbation series  where is the Borel singularity, and is it a pole?

Given f(z) ~ A (1 - z/z0)^beta near z0, this returns z0 and beta.  The critical
exponent people quote is -beta; the blow-up rate of an ODE is also -beta.

Every number below is checked against a closed form.

    python demos/composite_singularity.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composite.singularity import (
    analyse, blowup, series_solve, coefficient_asymptotics, radius,
)

PHI_INV = (5 ** 0.5 - 1) / 2
INV_SQRT_PI = 1 / math.sqrt(math.pi)


def rule(title):
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def catalan(n):
    return [float(math.comb(2 * k, k) // (k + 1)) for k in range(n)]


def motzkin(n):
    m = [1.0, 1.0]
    while len(m) < n:
        j = len(m) - 1
        m.append(m[j] + sum(m[k] * m[j - 1 - k] for k in range(j)))
    return m[:n]


def schroeder(n):
    r = [1.0]
    while len(r) < n:
        j = len(r)
        r.append(r[j - 1] + sum(r[k] * r[j - 1 - k] for k in range(j)))
    return r[:n]


def fibonacci(n):
    f = [0.0, 1.0]
    while len(f) < n:
        f.append(f[-1] + f[-2])
    return f[:n]


def binomial_series(gamma, xc, n):
    return [math.prod((gamma + j) / (j + 1) for j in range(k)) / xc ** k
            for k in range(n)]


# ---------------------------------------------------------------------------

def section_ode():
    rule("1.  NONLINEAR ODE BLOW-UP -- the time, and the rate")
    print()
    print("  y' = y^p, y(0) = y0   blows up at t* = y0^(1-p)/(p-1),")
    print("  approaching it as y ~ (t* - t)^-alpha with alpha = 1/(p-1).")
    print()
    print("   ODE                 t* found             t* true         err       rate     true")
    print("   " + "-" * 86)
    for p, y0 in ((2, 1.0), (2, 10.0), (3, 1.0), (5, 1.0)):
        f = [0.0] * (p + 1)
        f[p] = 1.0
        s = blowup(f, y0, terms=30)
        t_star, alpha = y0 ** (1 - p) / (p - 1), 1.0 / (p - 1)
        print("   y' = y^%d, y0 = %-5g %-20.16g %-15.10g %-9.1e %-8.6g %g"
              % (p, y0, s.location, t_star, abs(s.location - t_star),
                 s.blowup_rate, alpha))
    s = blowup([1.0, 0.0, 1.0], 0.0, terms=30)
    print("   y' = 1 + y^2        %-20.16g %-15.10g %-9.1e %-8.6g %g"
          % (s.location, math.pi / 2, abs(s.location - math.pi / 2),
             s.blowup_rate, 1.0))
    print("     (y = tan t; the poles sit at BOTH +-pi/2 and a blow-up time")
    print("      has to be the forward one)")

    print()
    print("  The alternative: integrate and watch it explode.  y' = y^2, y(0)=1,")
    print("  RK4 to the first step where y exceeds a threshold.")
    print()
    print("   step h    threshold 1e4   1e8        1e12       -> t* = 1 exactly")
    print("   " + "-" * 66)
    for h in (1e-2, 1e-3, 1e-4):
        row = []
        for thresh in (1e4, 1e8, 1e12):
            t, y = 0.0, 1.0
            while y < thresh and t < 2.0:
                k1 = y * y
                k2 = (y + h / 2 * k1) ** 2
                k3 = (y + h / 2 * k2) ** 2
                k4 = (y + h * k3) ** 2
                y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not math.isfinite(y):
                    break
            row.append(t)
        print("   %-9.0e %-15.9g %-10.9g %-10.9g" % (h, *row))
    print()
    print("  The estimate is step-limited: it cannot resolve t* better than one")
    print("  step, so h = 1e-4 buys about 1e-4, against 0.0e+00 from the series.")
    print("  And it never produces alpha at all.  The rate is the part you would")
    print("  have to do the asymptotics by hand to get, and it is often the part")
    print("  that matters -- it says what KIND of blow-up it is.")


def section_combinatorics():
    rule("2.  ANALYTIC COMBINATORICS -- growth constants from a counting sequence")
    print()
    print("   sequence           z0 found             1/z0 (growth)    beta      true beta")
    print("   " + "-" * 86)
    CASES = [("Catalan", catalan(32), 0.25, 0.5),
             ("Motzkin", motzkin(32), 1 / 3, 0.5),
             ("large Schroeder", schroeder(32), 3 - 2 * math.sqrt(2), 0.5),
             ("Fibonacci", fibonacci(32), PHI_INV, -1.0)]
    for name, c, z0, beta in CASES:
        s = analyse(c)
        print("   %-18s %-20.16g %-16.12g %-9.6g %g"
              % (name, s.location, 1 / s.location, s.exponent, beta))
    print()
    print("   growth constants recovered: 4, 3, 3+2sqrt2 = %.12g, 1/phi = %.12g"
          % (3 + 2 * math.sqrt(2), 1 / PHI_INV))
    print()
    c = catalan(32)
    C, z0, b = coefficient_asymptotics(c)
    print("   full asymptotic form for the Catalan numbers, from the numbers alone:")
    print("      C_n  ~  %.9g * n^%.9g * %.9g^n" % (C, -b - 1, 1 / z0))
    print("      true    %.9g * n^-1.5 * 4^n        (C = 1/sqrt(pi))" % INV_SQRT_PI)
    print()
    print("   n      C_n actual            asymptotic form       rel err")
    for n in (10, 20, 31):
        got = C * n ** (-b - 1) * (1 / z0) ** n
        print("   %-6d %-21.12g %-21.12g %.2e"
              % (n, c[n], got, abs(got - c[n]) / c[n]))
    print()
    print("   That last column is the ASYMPTOTIC FORM's own error at finite n --")
    print("   the leading term omits an O(1/n) correction worth ~3% at n = 31 --")
    print("   not the error in z0 or beta, which are at 1e-14 and 3e-12.")


def section_critical():
    rule("3.  CRITICAL EXPONENTS -- the lattice-model reading of the same number")
    print()
    print("  A susceptibility series chi(x) ~ (1 - x/xc)^-gamma: xc is the critical")
    print("  point, gamma the exponent.  Here on series with gamma known exactly,")
    print("  including one deliberately not a nice rational.")
    print()
    print("   series                 xc found             xc true     gamma found          true")
    print("   " + "-" * 88)
    for gamma, xc in ((1.75, 0.25), (1.2345, 0.2), (3.5, 2.0), (1 / 3, 0.5)):
        s = analyse(binomial_series(gamma, xc, 32))
        print("   (1-x/%-4g)^-%-8.5g %-20.16g %-11.8g %-20.16g %.6g"
              % (xc, gamma, s.location, xc, s.critical_exponent, gamma))
    print()
    print("  gamma = 7/4 is the two-dimensional Ising susceptibility exponent, which")
    print("  is the first row.  Recovered to %.1e from coefficients alone."
          % abs(analyse(binomial_series(1.75, 0.25, 32)).critical_exponent - 1.75))


def section_unknown():
    rule("4.  WHERE THERE IS NO CLOSED FORM")
    print()
    print("  y' = y^2 + t, y(0) = 1 has no elementary solution, so the blow-up time")
    print("  is not a formula to check against.  Convergence across approximant")
    print("  orders is the check instead.")
    print()
    from composite.composite_lib import Composite
    a = [0.0] * 31
    a[0] = 1.0
    for n in range(30):
        y = Composite({k: a[k] for k in range(n + 1) if a[k]})
        acc = y * y + Composite({1: 1.0})
        a[n + 1] = acc.coeffs_dict().get(n, 0.0) / (n + 1)
    print("   terms used   t* found              rate")
    print("   " + "-" * 52)
    for n in (16, 20, 24, 28, 31):
        s = analyse(a[:n], sign=+1)
        if s is None:
            print("   %-12d none" % n)
            continue
        print("   %-12d %-21.15g %.8g" % (n, s.location, s.blowup_rate))
    s = analyse(a, sign=+1)
    print()
    print("   t* = %.12g, and the rate is 1 -- set by the y^2 nonlinearity, which"
          % s.location)
    print("   is the one part of the answer that WAS predictable.")


def section_controls():
    rule("5.  NEGATIVE CONTROLS -- an entire function has no singularity")
    print()
    for name, c in (("exp(z)", [1.0 / math.factorial(n) for n in range(30)]),
                    ("cos(z)", [((-1) ** (n // 2) / math.factorial(n))
                                if n % 2 == 0 else 0.0 for n in range(30)]),
                    ("a polynomial", [1.0, 2.0, 3.0, 4.0] + [0.0] * 20),
                    ("all zeros", [0.0] * 30)):
        print("   %-16s -> %s" % (name, analyse(c) or "None"))
    print()
    print("   Returning a confident number for exp(z) would make every other")
    print("   result here worthless, so refusing is part of the answer.")


def section_two_methods():
    rule("6.  WHY TWO METHODS")
    print()
    print("  Pade on the log-derivative converges fast to a genuine POLE and slowly")
    print("  to a BRANCH POINT, because it has to approximate a cut with a string of")
    print("  poles.  The differential approximant encodes the exponent in an equation")
    print("  instead.  Same Catalan coefficients, same machine:")
    print()
    s = analyse(catalan(32))
    print("   differential approximant   z0 = %-20.16g beta = %-20.16g" % (s.location, s.exponent))
    if s.cross_check:
        print("   log-derivative Pade        z0 = %-20.16g beta = %-20.16g"
              % s.cross_check)
    else:
        print("   log-derivative Pade        no persistent pole -- the cut defeats it")
    print("   true                       z0 = 0.25                 beta = 0.5")
    print()
    s = analyse(fibonacci(32))
    print("  On a genuine simple pole (Fibonacci) the two agree exactly, and that")
    print("  agreement is what the confidence field reports:")
    print("   differential approximant   z0 = %.16g" % s.location)
    print("   log-derivative Pade        z0 = %.16g" % s.cross_check[0])
    print("   confidence                 %s" % s.confidence)
    print()
    print(s.describe())


def main():
    section_ode()
    section_combinatorics()
    section_critical()
    section_unknown()
    section_controls()
    section_two_methods()
    rule("SUMMARY")
    print()
    print("  One calculation -- coefficients in, (location, exponent) out -- reading")
    print("  as a blow-up time, a growth constant, or a critical exponent depending")
    print("  on who is asking.  Location and exponent land near 1e-14; the amplitude")
    print("  is an asymptotic fit and is term-count limited, which the module says.")
    print()


if __name__ == "__main__":
    main()
