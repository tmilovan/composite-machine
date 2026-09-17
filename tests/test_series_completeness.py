#!/usr/bin/env python3
# Composite Machine — series completeness audit
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Every transcendental must return only the orders it actually COMPLETES.

THE DEFECT THIS CATCHES.  A series built as `for n in range(1, terms)` over
`power = power * h` reaches orders it cannot finish whenever h spans more than
one order.  h**n starts at order n*m (m = lowest order in h), so after forming
powers 1..terms-1 every order at or below m*(terms-1) has all its
contributions and every order above it is missing those from the powers never
formed.  Those surplus orders are PARTIAL SUMS returned as finished
coefficients -- measured wrong by factors up to 5e+14.

WHY THE REST OF THE SUITE CANNOT SEE IT.  Two conditions have to coincide:

  1. h must span several orders, which needs a COMPOSED argument.  Tests call
     sqrt(_seeded(t)), where h = e is a single order and the bug is
     unreachable; it needs sqrt(x*x) or exp(-(x*x)).
  2. the consumer must sweep the whole coefficient dict.  taylor_coefficients
     raises the depth through _derivative_scope and never reads past the
     boundary, so derivatives look perfect.  antiderivative -- and integrate
     through it -- iterates everything, which is why this first surfaced as
     integrate(exp(-t^2), 0, 8) returning erf > 1.

Nine functions had it.  exp, sin and cos were fixed when the integrator bug was
traced; ln, sqrt, _reciprocal, tan, asin, acos and tanh were found by running
exactly the check below and fixed the same day.

THE CHECK.  Compute at the default depth, compute far deeper, and compare
coefficients present in both.  A coefficient that MOVES was never complete.
This needs no reference values and no tolerances of its own, so it covers any
function added later for free -- add it to CASES and it is audited.
"""
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import composite.composite_lib as cl

cl.MAX_ACTIVE_DIMS = 10 ** 9

DEEP = 90          # depth that any default-depth answer must already agree with
TOL = 1e-9         # relative move that counts as "was not complete"


def _orders(c):
    """{taylor order: coeff} for the scalar path; vector dims are not in scope."""
    return {-int(d): v for d, v in c.c.items()
            if not isinstance(d, tuple) and d <= 0}


def _audit(name, fn, arg):
    """(max_order, first_incomplete_order, relative_move) -- first is None if clean.

    RuntimeWarning is an error for the SHALLOW call only.  A backend overflow
    during ordinary use is a real defect -- asin emitted one on its way to
    returning 785 orders.  The deep call is a reference probe at a depth nobody
    runs at, and power's internal exp(0.5*ln x) legitimately overflows there;
    failing on that would be failing on the measuring instrument.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        shallow = fn(arg)
    old = cl._min_terms[0]
    cl._min_terms[0] = DEEP
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            deep = fn(arg)
    finally:
        cl._min_terms[0] = old
    a, b = _orders(shallow), _orders(deep)
    for k in sorted(set(a) & set(b)):
        if abs(b[k]) < 1e-300:
            continue
        rel = abs(a[k] - b[k]) / abs(b[k])
        if rel > TOL:
            return (max(a) if a else -1), k, rel
    return (max(a) if a else -1), None, 0.0


def build_cases():
    x = cl._seeded(1.3)
    composed = x * x               # h = 2.6e + e**2 -- TWO orders, the trigger
    small = x * x * 0.3            # same, inside the |arg| < 1 domain
    return [
        ("exp", cl.exp, composed),
        ("sin", cl.sin, composed),
        ("cos", cl.cos, composed),
        ("ln", cl.ln, composed),
        ("sqrt", cl.sqrt, composed),
        ("tan", cl.tan, small),
        ("atan", cl.atan, small),
        ("asin", cl.asin, small),
        ("acos", cl.acos, small),
        ("sinh", cl.sinh, composed),
        ("cosh", cl.cosh, composed),
        ("tanh", cl.tanh, small),
        ("erf", cl.erf, composed),
        ("erfc", cl.erfc, composed),
        ("_reciprocal", lambda a: cl._reciprocal(a, 15), composed),
        ("truediv", lambda a: cl.R(1.0) / a, composed),
        ("power", lambda a: cl.power(a, 0.5), composed),
    ]


def build_compositions():
    """KNOWN LIMITATION, reported but not failed -- see the note in main()."""
    x = cl._seeded(1.3)
    composed = x * x
    return [
        ("exp(sin)", lambda a: cl.exp(cl.sin(a)), composed),
        ("ln(cosh)", lambda a: cl.ln(cl.cosh(a)), composed),
        ("sqrt(exp)", lambda a: cl.sqrt(cl.exp(a)), composed),
        ("sin(exp)", lambda a: cl.sin(cl.exp(a)), composed),
        ("tanh(sin)", lambda a: cl.tanh(cl.sin(a)), composed),
    ]


def main():
    print("=" * 68)
    print("SERIES COMPLETENESS — every returned order must be finished")
    print("=" * 68)
    print(f"  {'function':>14}{'max order':>11}{'complete to':>13}"
          f"{'first bad':>11}{'rel move':>12}")
    print("  " + "-" * 61)

    failures = []
    for name, fn, arg in build_cases():
        try:
            mx, bad, rel = _audit(name, fn, arg)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name:>14}   ERROR {type(e).__name__}: {str(e)[:40]}")
            continue
        if bad is not None:
            failures.append((name, f"order {bad} moves by {rel:.2e}"))
        print(f"  {name:>14}{mx:>11}{(bad - 1 if bad else mx):>13}"
              f"{(bad if bad is not None else '-'):>11}"
              f"{(f'{rel:.1e}' if bad is not None else '-'):>12}"
              f"{'   INCOMPLETE' if bad is not None else ''}")

    # ---- compositions: ENFORCED since completeness became a carried property ----
    print("\n  COMPOSITIONS f(g(x)) -- enforced")
    print("  f(g(x)) can only be as complete as g, and g's bound cannot be")
    print("  INFERRED: _seeded(t) is exact with max order 1 while sin(x*x) is")
    print("  truncated with max order 11, and both merely look like 'max order")
    print("  K'.  So Composite._complete CARRIES it -- None meaning exact -- set")
    print("  by whatever truncates, propagated by every arithmetic op as the")
    print("  min of its operands, and read by each transcendental to cap its")
    print("  own reach.  exp(sin(x)) used to return 14 orders on 11 sound ones.\n")
    print(f"  {'composition':>14}{'max order':>11}{'sound to':>13}"
          f"{'first bad':>11}{'rel move':>12}")
    print("  " + "-" * 61)
    for name, fn, arg in build_compositions():
        try:
            mx, bad, rel = _audit(name, fn, arg)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name:>14}   ERROR {type(e).__name__}: {str(e)[:40]}")
            continue
        if bad is not None:
            failures.append((name, f"order {bad} moves by {rel:.2e}"))
        print(f"  {name:>14}{mx:>11}{(bad - 1 if bad else mx):>13}"
              f"{(bad if bad is not None else '-'):>11}"
              f"{(f'{rel:.1e}' if bad is not None else '-'):>12}"
              f"{'   INCOMPLETE' if bad is not None else ''}")

    total = len(build_cases()) + len(build_compositions())
    print("\n" + "=" * 68)
    if failures:
        print(f"RESULTS: {total - len(failures)}/{total} passed")
        print("Failed:")
        for n, why in failures:
            print(f"  - {n}: {why}")
        print("=" * 68)
        return 1
    print(f"RESULTS: {total}/{total} passed")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
