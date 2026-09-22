#!/usr/bin/env python3
# Composite Machine — singularity handling across the library
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""What every operation must do when its argument is not finite.

A library whose purpose is to compute AT a singular point has one obligation
above all others: never return a representable-looking value for something
outside the value group.  A wrong magnitude is worse than an error, because
every downstream comparison inherits it silently.

The value group here is powers and iterated logs -- dimensions are float64
(so dyadic fractional powers exist) and vector dimensions add the log axis,
where ln(h) is positive but below every power.  That is the powers-and-logs
fragment of a Hardy field.  Two families sit OUTSIDE it and must be refused
rather than approximated:

  FLAT / STEEP.  exp(-1/h) is nonzero and below every power; exp(1/h) is above
  every power.  No finite-rank dimension can name either.

  OSCILLATORY at an unbounded argument.  sin(1/h) is bounded but has no limit,
  so it is not eventually monotone and cannot live in ANY Hardy field, at any
  basis extension.  The correct answer is a permanent refusal.

The checks are grouped by the GRADE OF THE ARGUMENT, because that is the only
thing that decides whether a series evaluation is valid: a Maclaurin series is
sound exactly when the argument is (finite standard part) + (strictly negative
grade), and unsound otherwise.  Several checks below assert what the library
OUGHT to do and currently does not; they are written as the requirement, not
as the present behaviour, so the suite is the specification.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, INF
from test_dimension_scales import Suite, head


def lead(c):
    """Dominant grade, scalar or vector, as a comparable tuple."""
    d = {k: v for k, v in c.coeffs_dict().items() if v != 0.0}
    if not d:
        return None
    return max((k if isinstance(k, tuple) else (k, 0)) for k in d)


# =============================================================================
def s1_grades(t):
    head("S1  the objects, and their grades")
    t.dims("S1.01 h = |1|_-1", ZERO, {-1: 1.0})
    t.dims("S1.02 1/h = |1|_+1", R(1) / ZERO, {1: 1.0})
    t.true("S1.03 lead(h) is negative", lead(ZERO) < (0, 0),
           f"lead {lead(ZERO)}")
    t.true("S1.04 lead(1/h) is positive", lead(R(1) / ZERO) > (0, 0),
           f"lead {lead(R(1)/ZERO)}")
    t.true("S1.05 lead(1 + h) is 0 -- series-valid argument",
           lead(R(1) + ZERO) == (0, 0), f"lead {lead(R(1)+ZERO)}")
    t.true("S1.06 lead(1 + 1/h) is positive -- series-INVALID argument",
           lead(R(1) + R(1) / ZERO) > (0, 0),
           f"lead {lead(R(1)+R(1)/ZERO)}")


# =============================================================================
def s2_powers(t):
    head("S2  powers: the monomial path is right, the SUM path is not")
    t.dims("S2.01 sqrt(1/h) = |1|_+0.5", cl.sqrt(INF), {0.5: 1.0})
    t.dims("S2.02 sqrt(h)   = |1|_-0.5", cl.sqrt(ZERO), {-0.5: 1.0})
    t.dims("S2.03 (1 + 1/h)**2 is exact, three terms",
           (R(1) + INF) ** 2, {2: 1.0, 1: 2.0, 0: 1.0})
    t.dims("S2.04 1/(1 + 1/h) = h - h^2 + h^3 - ... (leading three)",
           Composite({k: v for k, v in
                      (R(1) / (R(1) + INF)).coeffs_dict().items()
                      if k >= -3}),
           {-1: 1.0, -2: -1.0, -3: 1.0})
    # (1 + 1/h)^(1/2) = h^(-1/2) * (1+h)^(1/2), so the dominant grade is +1/2.
    t.true("S2.05 (1 + 1/h)**0.5 has lead grade +0.5",
           lead((R(1) + INF) ** 0.5) == (0.5, 0),
           f"lead {lead((R(1)+INF)**0.5)}  want (0.5, 0)")
    t.true("S2.06 (1 + 1/h)**1.5 has lead grade +1.5",
           lead((R(1) + INF) ** 1.5) == (1.5, 0),
           f"lead {lead((R(1)+INF)**1.5)}  want (1.5, 0)")
    t.true("S2.07 (4 + 1/h)**0.5 has lead grade +0.5",
           lead((R(4) + INF) ** 0.5) == (0.5, 0),
           f"lead {lead((R(4)+INF)**0.5)}  want (0.5, 0)")


# =============================================================================
def s3_exp(t):
    head("S3  exp at positive grade -- outside the value group")
    # exp of a positive grade is not a large number, it is a different LEVEL.
    # exp(1/h) is above every power; exp(-1/h) is nonzero and below every
    # power -- the flat object, exp(-1/x^2), whose derivatives all vanish at
    # 0.  Naming either needs grades that are recursive expressions rather
    # than coordinates: transseries, not powers-and-logs.
    #
    # It used to apply the Maclaurin series regardless, so BOTH came back as
    # 1 - 1 + 1/2 - ... truncated at 15 terms with standard part 1.0 -- two
    # objects at opposite ends of the scale reported as the same number.
    t.raises("S3.01 exp(1/h) refuses -- above every power",
             Exception, lambda: cl.exp(INF))
    t.raises("S3.02 exp(-1/h) refuses -- below every power, nonzero",
             Exception, lambda: cl.exp(-INF))
    t.raises("S3.03 exp(1/h^2) refuses too -- any positive grade",
             Exception, lambda: cl.exp(R(1) / (ZERO * ZERO)))

    # The series path is untouched wherever it is sound: the argument must be
    # (finite standard part) + (strictly negative grade).
    t.close("S3.04 exp(h) standard part is 1", cl.exp(ZERO), 1.0)
    t.dims("S3.05 exp(h) is the series, not a collapse",
           Composite({k: v for k, v in cl.exp(ZERO).coeffs_dict().items()
                      if k >= -2}),
           {0: 1.0, -1: 1.0, -2: 0.5})
    t.close("S3.06 exp(1+h) standard part is e", cl.exp(R(1) + ZERO),
            math.e)

    # And the log axis still inverts, because (0,1) is not a power grade.
    t.dims("S3.07 exp(ln(1/h)) = |1|_1, the log axis survives",
           cl.exp(cl.ln(INF)), {1: 1.0})

    # THE TWO SIDES ARE NOT THE SAME, even though both refuse.  exp(-1/h) has
    # a well-defined standard part -- it is below every power, so the number
    # it approaches is unambiguously 0 -- and only its GRADE is unnameable.
    # exp(1/h) has no standard part at all.  Recorded because it is the
    # opening for a later rule: a refusal that knows which side it fell on
    # could answer lim exp(-1/x^2) = 0 exactly, without sampling.
    t.true("S3.08 the two sides are distinguishable at the point of refusal",
           _sign_of_dominant(INF) > 0 and _sign_of_dominant(-INF) < 0,
           f"exp(1/h) arg sign {_sign_of_dominant(INF)}, "
           f"exp(-1/h) arg sign {_sign_of_dominant(-INF)}")


def _sign_of_dominant(c):
    d = {k: v for k, v in c.coeffs_dict().items() if v != 0.0}
    if not d:
        return 0
    k = max(d, key=lambda k: (k if isinstance(k, tuple) else (k, 0)))
    return 1 if d[k] > 0 else -1


# =============================================================================
def s4_oscillatory(t):
    head("S4  oscillatory at an unbounded argument")
    # THE RULE: undefined at this dimension, so skip a dimension and keep the
    # value, exactly as R1 does for 1/0 -> |1|_1.  sin(|c|_d) = |sin(c)|_d:
    # the coefficient is not a claim about the value -- sin(1/h) does not
    # approach sin(1) -- it is MEMORY of what made the value undefined, and it
    # is what makes the skip revertible.  st() is then undefined because the
    # grade is positive, which is the point.
    #
    # Two requirements this does NOT meet, recorded rather than hidden:
    # the squeeze (x*sin(1/x) comes back |sin 1|_0 instead of 0, because
    # grades add and -1 + 1 cancels the damping) and the magnitude claim
    # (|sin| <= 1 can never be infinite, but a positive grade says it is).
    # No dimension map satisfies both -- st-undefined needs f(d) > 0 and an
    # honest magnitude needs f(d) <= 0.  See _bounded_at_inf.
    t.raises("S4.01 sin(1/h) refuses: the value is a range, not a point",
             Exception, lambda: cl.sin(INF))
    t.raises("S4.02 cos(1/h) refuses: the value is a range, not a point",
             Exception, lambda: cl.cos(INF))
    t.true("S4.03 sin(h) still works (argument has grade < 0)",
           abs(_st(cl.sin(ZERO))) < 1e-15,
           f"st {_st(cl.sin(ZERO))}")
    t.true("S4.04 sin(1 + h) still works",
           abs(_st(cl.sin(R(1) + ZERO)) - math.sin(1.0)) < 1e-12,
           f"st {_st(cl.sin(R(1)+ZERO))}  want {math.sin(1.0)}")


# =============================================================================
def s5_bounded(t):
    head("S5  bounded transcendentals at an unbounded argument")
    # atan(1/h) = pi/2 - h + h^3/3 - ...  The leading term alone is the limit,
    # not the value: dropping the correction silently loses a whole order.
    a = cl.atan(INF)
    t.close("S5.01 atan(1/h) standard part is pi/2", a, math.pi / 2)
    t.true("S5.02 atan(1/h) carries the -h correction",
           any(k == -1 and abs(v + 1.0) < 1e-9
               for k, v in a.coeffs_dict().items()),
           f"dims {a.coeffs_dict()}")
    # tanh(1/h) = 1 - 2exp(-2/h) + ...; the correction is exponentially flat,
    # so 1 is right to every representable order.
    t.dims("S5.03 tanh(1/h) = 1 exactly (remainder is flat)",
           cl.tanh(INF), {0: 1.0})


# =============================================================================
def s6_st(t):
    head("S6  st() must be partial, not total")
    t.raises("S6.01 st(1/h) raises -- no standard part exists",
             Exception, lambda: INF.st())
    t.raises("S6.02 st(ln(h)) raises -- negatively infinite",
             Exception, lambda: cl.ln(ZERO).st())
    t.close("S6.03 st(1 + h) = 1.0", R(1) + ZERO, 1.0)
    t.true("S6.04 coeff at grade 0 of 1/h is 0.0 and is NOT st()",
           (R(1) / ZERO).coeffs_dict().get(0, 0.0) == 0.0,
           "a coefficient read is total; a standard part is not")


# =============================================================================
def s7_log_axis(t):
    head("S7  the log axis -- ln of an infinitesimal")
    prev = getattr(cl, "LOG_SCALE", None)
    if prev is not None:
        cl.LOG_SCALE = True
    try:
        t.dims("S7.01 ln(h)   = |-1|_(0,1)", cl.ln(ZERO), {(0, 1): -1.0})
        t.dims("S7.02 ln(1/h) = |1|_(0,1)", cl.ln(INF), {(0, 1): 1.0})
        t.true("S7.03 the log grade is below every positive power",
               (0, 1) < (1, 0), "lexicographic: (0,1) < (1,0)")
        t.true("S7.04 the log grade is above grade 0",
               (0, 1) > (0, 0), "lexicographic: (0,1) > (0,0)")
    finally:
        if prev is not None:
            cl.LOG_SCALE = prev


# =============================================================================
def _try(f):
    try:
        return f()
    except Exception as e:
        return e


def _st(c):
    try:
        return c.st()
    except Exception as e:
        return f"<{type(e).__name__}>"


def _is_zero(c):
    return not any(v != 0.0 for v in c.coeffs_dict().values())


def _below(a, b):
    la, lb = lead(a), lead(b)
    return la is not None and lb is not None and la < lb


def _above(a, b):
    la, lb = lead(a), lead(b)
    return la is not None and lb is not None and la > lb


def _flat(c):
    return lead(c) is not None and lead(c) < (0, 0)


def run_all():
    t = Suite()
    for fn in (s1_grades, s2_powers, s3_exp, s4_oscillatory,
               s5_bounded, s6_st, s7_log_axis):
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
