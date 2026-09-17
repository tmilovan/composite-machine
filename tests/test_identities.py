#!/usr/bin/env python3
# Composite Machine — algebraic identity coherence check
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Identities that must hold, computed through INDEPENDENT paths.

WHY THIS CATCHES WHAT THE OTHER SUITES DO NOT.  Most tests check a value
against a number someone wrote down, on a bare seeded variable.  That walks the
easy path: h spans one order, u' == 1, nothing is composed.  Defects that need
a COMPOSED argument survive it indefinitely.

Running this battery for the first time found four, in one pass:

  - asin and atan omitted the CHAIN RULE.  They form 1/sqrt(1-u^2) and
    antidifferentiate w.r.t. eps, but d/deps asin(u(eps)) = u'/sqrt(1-u^2).
    Correct only when u' == 1 -- i.e. a bare seeded variable, which is what
    every test used.  asin(2x) returned exactly HALF its true first derivative.
    asin(x*x) looked right at the first probe only because 2a == 1 at a = 0.5.
  - deconvolution reported "exact" when it had truncated, so (1/x)*x = 1
    diverged at order 50.
  - antiderivative dropped the completeness bound (it builds its dict directly,
    bypassing _truncate_order).
  - _d_deps did not record that differentiating LOSES an order, so asin(sin x)
    claimed order 12 on 11 sound ones.

COMPARISON IS WITHIN THE CLAIMED COMPLETENESS.  sin(xs) is sound to order 11;
squaring it gives orders up to 22, but those need sin's orders 12..22, which do
not exist.  Comparing them would fail a correct library -- the first version of
this check did exactly that and reported 3/16, all of it the checker's fault.
So the bound comes from Composite._complete, which is what it is for.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import composite.composite_lib as cl
from composite.composite_lib import Composite, R

cl.MAX_ACTIVE_DIMS = 10 ** 9
TOL = 1e-9


def _orders(c):
    return {-int(d): v for d, v in c.c.items()
            if not isinstance(d, tuple) and d <= 0}


def disagreement(lhs, rhs):
    """(worst relative disagreement, order compared to)."""
    a, b = _orders(lhs), _orders(rhs)
    bound = cl._tighter(getattr(lhs, "_complete", None),
                        getattr(rhs, "_complete", None))
    if bound is None:                       # both exact: every order counts
        bound = max(max(a, default=0), max(b, default=0))
    ref = max((abs(v) for v in list(a.values()) + list(b.values())), default=1.0)
    worst = 0.0
    for k in sorted(set(a) | set(b)):
        if k > bound:
            continue
        worst = max(worst, abs(a.get(k, 0.0) - b.get(k, 0.0)) / max(ref, 1e-300))
    return worst, bound


def build():
    x = cl._seeded(0.7)
    xs = x * x * 0.5            # COMPOSED: this is what makes the check bite
    y = cl._seeded(0.4)
    return [
        ("sin^2+cos^2=1", lambda: cl.sin(xs)*cl.sin(xs)+cl.cos(xs)*cl.cos(xs),
                          lambda: R(1.0)),
        ("cosh^2-sinh^2=1", lambda: cl.cosh(xs)*cl.cosh(xs)-cl.sinh(xs)*cl.sinh(xs),
                            lambda: R(1.0)),
        ("exp(a+b)=e^a e^b", lambda: cl.exp(x+y), lambda: cl.exp(x)*cl.exp(y)),
        ("ln(exp x)=x", lambda: cl.ln(cl.exp(xs)), lambda: xs),
        ("exp(ln x)=x", lambda: cl.exp(cl.ln(x)), lambda: x),
        ("sqrt(x)^2=x", lambda: cl.sqrt(x)*cl.sqrt(x), lambda: x),
        ("(1/x)*x=1", lambda: (R(1.0)/x)*x, lambda: R(1.0)),
        ("tan=sin/cos", lambda: cl.tan(xs), lambda: cl.sin(xs)/cl.cos(xs)),
        ("tanh=sinh/cosh", lambda: cl.tanh(xs), lambda: cl.sinh(xs)/cl.cosh(xs)),
        ("sin(2x)=2 sin cos", lambda: cl.sin(xs*2.0),
                              lambda: cl.sin(xs)*cl.cos(xs)*2.0),
        ("cos(2x)=1-2sin^2", lambda: cl.cos(xs*2.0),
                             lambda: R(1.0)-cl.sin(xs)*cl.sin(xs)*2.0),
        ("asin(sin x)=x", lambda: cl.asin(cl.sin(xs)), lambda: xs),
        ("sin(asin x)=x", lambda: cl.sin(cl.asin(xs)), lambda: xs),
        ("atan(tan x)=x", lambda: cl.atan(cl.tan(xs)), lambda: xs),
        ("tan(atan x)=x", lambda: cl.tan(cl.atan(xs)), lambda: xs),
        ("erf+erfc=1", lambda: cl.erf(xs)+cl.erfc(xs), lambda: R(1.0)),
        ("acos=pi/2-asin", lambda: cl.acos(xs), lambda: R(math.pi/2)-cl.asin(xs)),
        ("cosh=(e^x+e^-x)/2", lambda: cl.cosh(xs),
                              lambda: (cl.exp(xs)+cl.exp(-xs))/2.0),
        ("sinh=(e^x-e^-x)/2", lambda: cl.sinh(xs),
                              lambda: (cl.exp(xs)-cl.exp(-xs))/2.0),
    ]


def roundtrip():
    """d/dx of an antiderivative is the integrand again.

    antiderivative stores coeff/|new_dim| at new_dim = dim-1, so inverting it
    multiplies by |stored dim|, NOT |stored dim + 1| -- getting that wrong
    zeroes the constant term and reports a spurious 1.0 disagreement.
    """
    f = cl.exp(cl._seeded(0.7))
    F = cl.antiderivative(f)
    back = Composite({d + 1: c * abs(d) for d, c in F.c.items() if d <= -1})
    c = getattr(F, "_complete", None)
    back._complete = (c - 1) if c is not None else None
    return disagreement(back, f)


def main():
    print("=" * 66)
    print("ALGEBRAIC IDENTITIES — independent paths, composed arguments")
    print("=" * 66)
    print(f"  {'identity':>20}{'sound to':>10}{'disagreement':>16}")
    print("  " + "-" * 46)
    failures = []
    cases = build()
    for name, lhs, rhs in cases:
        try:
            w, b = disagreement(lhs(), rhs())
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name:>20}   ERROR {type(e).__name__}")
            continue
        if w >= TOL:
            failures.append((name, f"disagrees by {w:.2e} within order {b}"))
        print(f"  {name:>20}{b:>10}{w:>16.2e}"
              f"{'   DIVERGES' if w >= TOL else ''}")
    try:
        w, b = roundtrip()
        if w >= TOL:
            failures.append(("d/dx antideriv=f", f"disagrees by {w:.2e}"))
        print(f"  {'d/dx antideriv=f':>20}{b:>10}{w:>16.2e}"
              f"{'   DIVERGES' if w >= TOL else ''}")
    except Exception as e:
        failures.append(("d/dx antideriv=f", f"{type(e).__name__}: {e}"))

    total = len(cases) + 1
    print("\n" + "=" * 66)
    if failures:
        print(f"RESULTS: {total - len(failures)}/{total} passed")
        print("Failed:")
        for n, why in failures:
            print(f"  - {n}: {why}")
        print("=" * 66)
        return 1
    print(f"RESULTS: {total}/{total} passed")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
