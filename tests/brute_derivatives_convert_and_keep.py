#!/usr/bin/env python3
"""Brute-force derivative battery for CANCELLATION = "convert_and_keep".

The same random expressions, seeds, tolerances and report as
brute_derivatives.py (which scores against the MECHANICAL rule).  Only the
composite-semantics reference `den` changes, built from Zero Rules v2 sections
1b and 1c rather than from the code:

    R1      a wholly zero operand of ANY operation converts, at its LOWEST
            grade: |0|_d -> |1|_(d-1), i.e. (x - a)**(1 - d)
    K1, K2  + and - included; the zero converts, then the sign applies
    N1, N2  + and - count: a cancellation leaves zeros, no residue
    N3      a cancelled infinite grade drops to |0|_0, so a wholly zero result
            never has its lowest grade above 0
    R3      x 1 and / 1 return the other operand untouched
    6.2     a function of a value with no infinitesimal part returns its value
            directly, one term at grade 0: ln(x / x) is <0_0>, not a series

Because a cancellation is counted, an identically zero result is a zero at
the lowest grade its operands carried, and it converts there when it is next
used: x - x is <0_0 0_-1> and converts to (x-a)**2, while 3 - 3 is <0_0> and
converts to (x-a).  So the reference tracks each value's SUPPORT, its lowest
grade, by the rules: x is -1, a constant 0, a product adds them (R5, Minkowski
sum), a power multiplies, + and - take the lower, a quotient by a single term
shifts.

Where the rules do not fix the support -- a transcendental of a non-constant
argument, a quotient by more than one term -- the depth is the library's
series length, a truncation choice.  A zero of unknown depth is modelled as
(x - a)**40: invisible at orders 0..6 whatever the true depth, as long as that
depth is at least 7.  Where such a DEEP zero would lead a divisor or the
argument of ln or a real power, its depth would decide the value, and the case
is not scored (SKIP "SupportDependent").

    python tests/brute_derivatives_convert_and_keep.py 1000 --seed 11
"""
import math, random, sys
import mpmath as mp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import brute_derivatives as bd
import composite.composite_lib as cl
from composite.composite_lib import R, power

DEEP = 40


class SupportDependent(ValueError):
    """The value depends on a series length the rules do not fix."""


def ev(t, seed, a, events):
    """(comp, cls, den), as brute_derivatives.ev, under convert_and_keep."""
    comp, cls_, den, _ = _ev(t, seed, a, events)
    return comp, cls_, den


def _info(low, mono, deep=False):
    # low: lowest grade carried (None = set by truncation); mono: the grade of
    # a single-term value, else None; deep: holds a zero of unknown depth.
    return {"low": low, "mono": mono, "deep": deep}


def _ev(t, seed, a, events):
    k = t[0]
    if k == "x":
        return seed, (lambda x: x), (lambda x: x), _info(-1, None)
    if k == "c":
        c = mp.mpf(t[1])
        return R(t[1]), (lambda x, c=c: c), (lambda x, c=c: c), _info(0, 0)

    def is0(f):
        return bd.ident_zero(f, a)

    def unit(f):
        return bd.ident_zero(lambda x: f(x) - 1, a)

    def conv(d, inf):
        """R1 on an operand: a wholly zero value converts at its lowest grade."""
        if not is0(d):
            return d, inf
        events.add("R1")
        if inf["low"] is None:
            events.add("deep")
            return (lambda x: (x - a) ** DEEP), _info(-DEEP, -DEEP, True)
        g = min(inf["low"], 0) - 1          # N3: never above grade 0
        return (lambda x, p=-g: (x - a) ** p), _info(g, g, inf["deep"])

    def lead_is_deep(d, inf):
        if not inf["deep"]:
            return False
        with mp.workdps(150):
            v = bd.safe(d, a)
            return not mp.isfinite(v) or abs(v) < mp.mpf("1e-120")

    def const(d, inf):
        """No infinitesimal part (§6.2): the value is constant near a.  x / x
        qualifies although it carries zero terms below grade 0."""
        if inf["deep"]:
            return False
        if inf["mono"] == 0:
            return True
        with mp.workdps(150):
            v = bd.safe(d, a)
        return mp.isfinite(v) and bd.ident_zero(lambda x: d(x) - v, a)

    if k == "u":
        fc, fm = bd.UN[t[1]]
        c, s, d, inf = _ev(t[2], seed, a, events)
        dd, di = conv(d, inf)
        if t[1] == "ln" and lead_is_deep(dd, di):
            raise SupportDependent("ln of a zero of unknown depth")
        out = _info(0, 0) if const(dd, di) else _info(None, None, di["deep"])
        return (fc(c), (lambda x: fm(s(x))) if s else None,
                (lambda x: fm(dd(x))), out)

    if k in ("ipow", "rpow"):
        c, s, d, inf = _ev(t[1], seed, a, events)
        p = t[2]
        dd, di = conv(d, inf)
        if k == "ipow":
            low = None if di["low"] is None else p * di["low"]
            mono = None if di["mono"] is None else p * di["mono"]
            return (c ** p, (lambda x: s(x) ** p) if s else None,
                    (lambda x: dd(x) ** p), _info(low, mono, di["deep"]))
        if lead_is_deep(dd, di):
            raise SupportDependent("real power of a zero of unknown depth")
        q = mp.mpf(p)
        lgi = _info(0, 0) if const(dd, di) else _info(None, None, di["deep"])
        lg, _ = conv(lambda x: mp.log(dd(x)), lgi)
        out = _info(0, 0) if const(dd, di) else _info(None, None, di["deep"])
        return (power(c, p), (lambda x: s(x) ** q) if s else None,
                (lambda x: mp.exp(q * lg(x))), out)

    c1, s1, d1, i1 = _ev(t[1], seed, a, events)
    c2, s2, d2, i2 = _ev(t[2], seed, a, events)
    both = s1 is not None and s2 is not None

    if k == "cpow":
        L, Li = conv(d1, i1)
        Rr, Ri = conv(d2, i2)
        if lead_is_deep(L, Li):
            raise SupportDependent("power base is a zero of unknown depth")
        lgi = _info(0, 0) if const(L, Li) else _info(None, None, Li["deep"])
        lg, _ = conv(lambda x: mp.log(L(x)), lgi)
        out = (_info(0, 0) if const(L, Li) and const(Rr, Ri)
               else _info(None, None, Li["deep"] or Ri["deep"]))
        return (power(c1, c2), (lambda x: s1(x) ** s2(x)) if both else None,
                (lambda x: mp.exp(Rr(x) * lg(x))), out)

    if k in "+-":
        L, Li = conv(d1, i1)                 # K1: a zero operand converts
        Rr, Ri = conv(d2, i2)
        sg = 1 if k == "+" else -1           # K2: then the sign applies
        comp = c1 + c2 if k == "+" else c1 - c2
        cls_ = (lambda x: s1(x) + sg * s2(x)) if both else None
        den = lambda x: L(x) + sg * Rr(x)    # N1: counted, no residue
        low = (None if Li["low"] is None or Ri["low"] is None
               else min(Li["low"], Ri["low"]))
        mono = Li["mono"] if Li["mono"] is not None and Li["mono"] == Ri["mono"] else None
        deep = Li["deep"] or Ri["deep"]
        if is0(den):
            events.add("cancel")
            if low is not None:
                low = min(low, 0)            # N3
        return comp, cls_, den, _info(low, mono, deep)

    if k == "*":
        if unit(d1):                         # R3: the other operand untouched
            return c1 * c2, (lambda x: s1(x) * s2(x)) if both else None, d2, i2
        if unit(d2):
            return c1 * c2, (lambda x: s1(x) * s2(x)) if both else None, d1, i1
        L, Li = conv(d1, i1)
        Rr, Ri = conv(d2, i2)
        low = (None if Li["low"] is None or Ri["low"] is None
               else Li["low"] + Ri["low"])
        mono = (None if Li["mono"] is None or Ri["mono"] is None
                else Li["mono"] + Ri["mono"])
        return (c1 * c2, (lambda x: s1(x) * s2(x)) if both else None,
                (lambda x: L(x) * Rr(x)), _info(low, mono, Li["deep"] or Ri["deep"]))

    if k == "/":
        cls_ = None
        if both and not bd.ident_zero(s2, a):
            cls_ = lambda x: s1(x) / s2(x)
        if unit(d2):
            return c1 / c2, cls_, d1, i1
        L, Li = conv(d1, i1)
        Rr, Ri = conv(d2, i2)
        if lead_is_deep(Rr, Ri):
            raise SupportDependent("division by a zero of unknown depth")
        if Ri["mono"] is not None:           # a single term: a shift
            low = None if Li["low"] is None else Li["low"] - Ri["mono"]
            mono = None if Li["mono"] is None else Li["mono"] - Ri["mono"]
        else:                                # long division: series length
            low, mono = None, None
        return (c1 / c2, cls_, (lambda x: L(x) / Rr(x)),
                _info(low, mono, Li["deep"] or Ri["deep"]))
    raise ValueError(k)


if __name__ == "__main__":
    import argparse
    from composite.backends import config
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", type=int, nargs="?", default=100,
                    help="scored cases per class (default 100)")
    ap.add_argument("--seed", type=int, default=1,
                    help="base seed; the zero-injected class uses seed + 1")
    ap.add_argument("--backend", default="sparse_dense",
                    choices=["sparse_dense", "dict", "dense_series", "fractional_dict"])
    ap.add_argument("--max-order", type=int, default=None,
                    help="set_max_order(n) for the run (default: no cap)")
    args = ap.parse_args()
    getattr(config, "use_" + args.backend)()
    cl._refresh_constants()
    cl.set_max_order(args.max_order)
    cl.CANCELLATION = "convert_and_keep"
    bd.ev = ev                               # the battery's runner, this reference
    print(f"rule {cl.CANCELLATION}, backend {type(config.get_backend()).__name__}, "
          f"orders 0..{bd.NMAX}, tol {bd.TOL} rel, mpmath taylor at 40 and 60 "
          f"digits (agree {bd.AGREE}), seed {args.seed}/+1, max order {args.max_order}",
          flush=True)
    for label, zeros, rs in (("classical (no injected zeros)", False, args.seed),
                             ("zero-injected", True, args.seed + 1)):
        (st, fails, errs, missed, unrel, tos, cut, passes,
         skip_reasons, skip_examples) = bd.run(args.cases, zeros, rs)
        print(f"\n=== {label}: {st}", flush=True)
        for k_, n_ in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  SKIP {n_:4d}  {k_}   e.g. {skip_examples[k_]}")
        for tag, rows in (("FAIL", fails), ("ERR ", errs), ("UNWARNED", missed),
                          ("UNRL", unrel), ("RTO ", tos), ("CUT ", cut)):
            for r_ in rows:
                print(f"  {tag}", r_)
        print("-- sample passes (expr | a | n | got | ref | denotation_order | events)")
        for p in [p for p in passes if not p[6]][:4] + [p for p in passes if p[6]][:6]:
            print("  ok  ", p)
