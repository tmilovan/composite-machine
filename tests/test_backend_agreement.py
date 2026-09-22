#!/usr/bin/env python3
# Composite Machine — the three backends must agree
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""One number, three representations: they must not disagree about its value.

sparse-dense stores runs of contiguous dimensions, dict stores a plain map,
dense-series stores one contiguous frame at a fixed step.  Which one is active
is a performance choice and must never be a semantic one -- but three bugs
found by a black-hole thermodynamics calculation were each visible ONLY as a
disagreement between them, and every one produced confident wrong numbers
rather than an error.

They differ legitimately in ONE way, and the tests below allow for it: the
dense frame expresses every dimension in its span, zeros included, while the
other two express only dimensions that arose.  Under R2 an expressed zero term
is not the same object as an absent one, so agreement is asserted on the
NONZERO coefficients.

WHAT IT LOOKS FOR, in order of how much it can embarrass us:

  B1  the leading grade of a product.  sparse-dense builds a product from
      runs, and _merge_runs groups them by lattice because a run at 0,1,2 and
      one at 0.5,1.5 interleave and cannot share a dense array.  Flattening
      then concatenated run by run, so (1+h)(1-sqrt h) came out ordered
      [-1.5, -0.5, -1.0, 0.0] -- not sorted.  deconvolve reads its leading
      term as b.dims[nonzero][-1], took -0.5 instead of 0.0, and returned
      a series in POSITIVE grades: |1|_16.5 + |1|_16 + ... for a quantity
      whose true value is -h**0.5 - h + h**2 + ...

  B2  the length of a non-terminating quotient.  dense-series divided with a
      remainder array no longer than the dividend, so 1/(1+h) -- whose
      quotient continues below every grade the dividend has -- stopped after
      ONE coefficient and returned |1|_0.  The sparse backend does not hit it
      because its remainder grows as b.dims + q_dim adds new dimensions.

  B3  the phase of a lattice.  dense-series treated an equal STEP as the same
      lattice.  Two frames at step 1.0 with offsets 0 and -11.5 are half a
      step out of phase and no index holds both; add() placed each term at the
      nearest slot, moving R(1) from grade 0 to grade 0.5.  That put the
      leading term of a black-hole temperature at |0.0795775|_0.5 instead of
      |0.0795775|_0 -- an infinite quantity where a finite one belongs.

  B4  agreement across a battery, including the formulas that found B1-B3.

Each case is computed independently under each backend and compared after the
default is restored, so a failure cannot leave the process on the wrong one.
"""
import math
import os
import sys
import importlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from composite.backends import config as cfg
import composite.composite_lib as _cl
from test_dimension_scales import Suite, head

BACKENDS = ("use_sparse_dense", "use_dict", "use_dense_series")


def _under(backend, fn):
    """Run fn(cl, Composite, R) with `backend` active, then restore."""
    getattr(cfg, backend)()
    try:
        importlib.reload(_cl)
        return fn(_cl, _cl.Composite, _cl.R)
    finally:
        cfg.use_sparse_dense()
        importlib.reload(_cl)


def _nonzero(c):
    return {k: v for k, v in c.coeffs_dict().items() if v != 0.0}


def _top(d, n=4):
    return [(k, round(d[k], 9)) for k in sorted(d, reverse=True)[:n]]


#  the formulas, written once and run under each backend
def f_versine(cl, C, R):
    h = C({-1: 1.0})
    return _nonzero((R(1) + h) * (R(1) - cl.sqrt(h)))


def f_heat(cl, C, R):
    h = C({-1: 1.0})
    return _nonzero(h / ((R(1) + h) * (R(1) - cl.sqrt(h)) - R(1)))


def f_geometric(cl, C, R):
    h = C({-1: 1.0})
    return _nonzero(R(1) / (R(1) + h))


def f_one_minus_sqrt(cl, C, R):
    h = C({-1: 1.0}); M = R(1) + h
    return _nonzero(R(1) - cl.sqrt(R(1) - R(1) / (M * M)))


def f_hawking(cl, C, R):
    h = C({-1: 1.0}); M = R(1) + h
    return _nonzero((M / (4 * math.pi)) * (R(1) - cl.sqrt(R(1) - R(1) / (M * M))))


BATTERY = [("(1+h)(1-sqrt h)", f_versine),
           ("h / ((1+h)(1-sqrt h) - 1)", f_heat),
           ("1/(1+h)", f_geometric),
           ("1 - sqrt(1 - 1/M^2)", f_one_minus_sqrt),
           ("T(M) = (M/4pi)(1-sqrt(1-1/M^2))", f_hawking)]


def b1_product_leading_grade(t):
    head("B1  a product must not reorder its dimensions")
    d = _under("use_sparse_dense", f_versine)
    t.exact("B1.01 (1+h)(1-sqrt h) has the grades it should",
            sorted(d), [-1.5, -1.0, -0.5, 0.0])
    t.true("B1.02 and the flat form is SORTED",
           list(d) == sorted(d), "got %s" % list(d))
    q = _under("use_sparse_dense", f_heat)
    t.true("B1.03 so the quotient has no positive grade",
           not [k for k in q if k > 0], "positive grades %s" % [k for k in q if k > 0])
    for g, want in ((-0.5, -1.0), (-1.0, -1.0), (-2.0, 1.0),
                    (-2.5, 1.0), (-3.5, -1.0), (-4.0, -1.0)):
        t.close("B1.04 h/((1+h)(1-sqrt h)-1) at grade %-4g" % g,
                q.get(g, 0.0), want, tol=1e-12)


def b2_non_terminating_quotient(t):
    head("B2  a non-terminating quotient continues below the dividend")
    for bk in BACKENDS:
        d = _under(bk, f_geometric)
        t.true("B2.01 %-18s 1/(1+h) has many terms, not one" % bk,
               len(d) > 10, "got %d terms" % len(d))
        got = [d.get(-k) for k in range(6)]
        t.exact("B2.02 %-18s and they alternate 1,-1,1,..." % bk,
                got, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def b3_lattice_phase(t):
    head("B3  an equal step is not the same lattice")
    for bk in BACKENDS:
        d = _under(bk, f_one_minus_sqrt)
        t.close("B3.01 %-18s 1-sqrt(...) leads with 1 at grade 0" % bk,
                d.get(0.0, 0.0), 1.0, tol=1e-15)
        t.true("B3.02 %-18s and nothing sits at grade 0.5" % bk,
               0.5 not in d, "found %s at 0.5" % d.get(0.5))
        h = _under(bk, f_hawking)
        t.close("B3.03 %-18s T(M) leads with 1/(4pi) at grade 0" % bk,
                h.get(0.0, 0.0), 1.0 / (4 * math.pi), tol=1e-12)
        t.true("B3.04 %-18s T(M) has no positive grade" % bk,
               not [k for k in h if k > 0], "positive %s" % [k for k in h if k > 0])


def b4_battery(t):
    head("B4  every backend agrees on every nonzero coefficient")
    # A non-terminating quotient is cut at whatever depth each backend
    # reaches, and those differ legitimately: 1/((1+h)(1-sqrt h)-1) gives 50
    # nonzero terms down to grade -37 on sparse-dense and dict, and 34 down to
    # -25 on dense-series.  Demanding identical grade SETS fails on that and
    # would have to be relaxed to a tolerance, which hides the real property.
    #
    # What must hold instead: every coefficient they BOTH produce is the same,
    # and no backend SKIPS a grade inside the range it does cover.  Stopping
    # early is a representation choice; a hole in the middle is a bug, and a
    # hole is exactly what B1's misordered product produced.
    for name, fn in BATTERY:
        ref = _under("use_sparse_dense", fn)
        for bk in BACKENDS[1:]:
            got = _under(bk, fn)
            shared = set(ref) & set(got)
            worst = max((abs(ref[k] - got[k]) for k in shared), default=0.0)
            floor = min(got) if got else 0.0
            holes = sorted((k for k in ref if k >= floor and k not in got),
                           reverse=True)
            depth = len(shared)
            t.true("B4 %-32s %s" % (name[:32], bk.replace("use_", "")),
                   worst < 1e-9 and not holes and depth >= min(len(ref), 4),
                   "%d shared to grade %g, worst diff %.2e, holes %s"
                   % (depth, floor, worst, holes[:4] or "none"))


def b5_fractional_power_of_a_scaled_infinitesimal(t):
    head("B5  a fractional power of a SCALED infinitesimal, on every backend")
    # ZERO ** 0.5 worked everywhere; (ZERO/2) ** 0.5 raised TypeError on
    # DictBackend alone -- "object of type 'int' has no len()".
    #
    # ** routes through exp(n * ln(x)), and ln() puts a term on the LOG AXIS
    # whatever backend it was called on, so a scalar backend ends up holding
    # tuple dimensions next to scalar ones.  exp's remainder branch indexed
    # those keys positionally.  A unit coefficient missed it because its
    # remainder is empty; a scaled one did not.
    #
    # This is the half grade, and the half grade IS the branch point: the two
    # roots leaving a DOUBLE root under a coefficient perturbation go as
    # eps**(1/2), which no float can hold.  Losing it on one backend loses
    # exactly the case composites are there for.
    cases = [
        ("ZERO ** 0.5",           lambda cl, C, R: (C({-1: 1.0}) ** 0.5),      -0.5, 1.0),
        ("(ZERO/2) ** 0.5",       lambda cl, C, R: ((C({-1: 1.0}) / 2.0) ** 0.5),
         -0.5, 1.0 / math.sqrt(2.0)),
        ("(3*ZERO) ** 0.5",       lambda cl, C, R: ((R(3) * C({-1: 1.0})) ** 0.5),
         -0.5, math.sqrt(3.0)),
        ("ZERO ** 0.25",          lambda cl, C, R: (C({-1: 1.0}) ** 0.25),     -0.25, 1.0),
        ("(ZERO/2) ** 2",         lambda cl, C, R: ((C({-1: 1.0}) / 2.0) ** 2), -2.0, 0.25),
    ]
    for i, (lbl, fn, grade, coeff) in enumerate(cases, 1):
        vals = {}
        for bk in BACKENDS:
            try:
                vals[bk] = _under(bk, lambda cl, C, R: fn(cl, C, R).coeffs_dict())
            except Exception as e:
                vals[bk] = "%s: %s" % (type(e).__name__, e)
        bad = [bk for bk, v in vals.items() if not isinstance(v, dict)]
        t.true(f"B5.{i:02d} {lbl} evaluates on all three backends"
               + (f"  -- {bad}: {vals[bad[0]]}" if bad else ""),
               not bad, str(vals))
        if bad:
            continue
        got = {bk: v.get(grade) for bk, v in vals.items()}
        t.true(f"B5.{i:02d}b {lbl} -> grade {grade} on all three: "
               + ", ".join("%s %s" % (b, "%.12g" % g if g is not None else "MISSING")
                           for b, g in got.items()),
               all(g is not None for g in got.values()), str(vals))
        if all(g is not None for g in got.values()):
            worst = max(abs(g - coeff) for g in got.values())
            t.true(f"B5.{i:02d}c coefficient {coeff:.12g}, worst deviation {worst:.2e}",
                   worst <= 1e-15, str(got))


def run_all():
    t = Suite()
    for fn in (b1_product_leading_grade, b2_non_terminating_quotient,
               b5_fractional_power_of_a_scaled_infinitesimal,
               b3_lattice_phase, b4_battery):
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
