#!/usr/bin/env python3
"""Brute-force derivative battery: random expressions against an mpmath reference.

Not a pass/fail suite (pytest does not collect it).  It reports, by category,
where the library's derivatives disagree with an independent reference, so a
change can be checked against thousands of compositions nobody wrote by hand.

    python tests/brute_derivatives.py 1000 --seed 11 --backend dict

Each class (no injected zeros, then zero-injected) draws its own expressions:
the first from seed S, the second from S + 1.  The same seed and backend always
give the same expressions.

HOW AN EXPRESSION IS SCORED
  Random trees over x, constants and +, -, *, /, integer and real powers,
  power(a, b), sin cos tan exp ln sqrt atan asin acos sinh cosh tanh.  The
  zero-injected class adds expressed zeros and cancellations: R(0), e - e, e / e,
  ln(e / e), e * 0, c - c.

  Each node is built three ways in lockstep:
    comp  the composite, seeded x = R(a) + ZERO, composite functions only,
          float coercion refused
    cls   an mpmath function, the classical reading (a zero is the additive
          identity); used only to tell which orders are non-conventional
    den   an mpmath function, the composite-semantics reading, built from Zero
          Rules v2 rather than from the code:
            R1      an operand identically zero converts to (x - a)
            cancel  a +- b identically zero, operands nonzero -> L * (x - a)
            R3      x 1 returns the other operand untouched (a zero stays latent)
            power   b**s is exp(s * ln b); an ln that is identically zero converts

  Orders 0..NMAX are scored against den with a FIXED tolerance,
  |got - ref| <= TOL * max(1, |ref|).  The reference is mp.taylor at 40 and at
  60 digits, evaluated at the same binary doubles the library seeds; if the two
  disagree beyond AGREE it is UNRELIABLE and nothing is scored.

  denotation_order is a warning for the user, not an oracle, and plays no part
  in scoring.  It is checked separately: an order where den differs from cls
  but d(n) would not warn is reported as UNWARNED.

OUTPUT
  FAIL      value disagrees with the reference
  ERR       the library raised while reading, or refused to build (InexactGrade)
  UNWARNED  non-conventional order with no NotConventionalWarning
  UNRL      reference unusable at that order (branch point, singular, unreliable)
  RTO       reference took longer than REF_SECONDS
  CUT       order beyond the result's complete_order, not scored
  SKIP      the library refused the domain, grouped by reason

Known residue on the 2026-10-09 working tree (OPEN_ITEMS.md section 13):
inverse trig next to a branch point (ill-conditioned in float64), fl(pi/2),
the power/sqrt default-depth mismatch, and the fractional-power flag gap.
"""
import math, random, signal, sys, warnings
import mpmath as mp
warnings.simplefilter("ignore")
import composite.composite_lib as cl
from composite.composite_lib import (R, ZERO, sin, cos, exp, ln, sqrt, tan, atan,
                                     asin, acos, sinh, cosh, tanh, power)
from composite.backends.config import get_backend
from composite.backends.base_backend import InexactGradeError

NMAX, TOL, AGREE = 6, 1e-9, 1e-15
REF_SECONDS = 3


class RefTimeout(Exception):
    pass


def _alarm(*_):
    raise RefTimeout()


signal.signal(signal.SIGALRM, _alarm)

UN = {"sin": (sin, mp.sin), "cos": (cos, mp.cos), "exp": (exp, mp.exp),
      "ln": (ln, mp.log), "sqrt": (sqrt, mp.sqrt), "tan": (tan, mp.tan),
      "atan": (atan, mp.atan), "asin": (asin, mp.asin), "acos": (acos, mp.acos),
      "sinh": (sinh, mp.sinh), "cosh": (cosh, mp.cosh), "tanh": (tanh, mp.tanh)}
CONSTS = [0.5, 2.0, 3.0, -1.5, 1.25]


def gen(depth, zeros):
    """Random expression tree.  zeros=True injects expressed zeros and cancellations."""
    if depth == 0 or random.random() < 0.25:
        if zeros and random.random() < 0.25:
            return ("c", 0.0)
        return ("x",) if random.random() < 0.6 else ("c", random.choice(CONSTS))
    r = random.random()
    if zeros and r < 0.25:
        e = gen(depth - 1, zeros)
        kind = random.choice(["selfsub", "selfdiv", "lnself", "mulzero", "constsub"])
        if kind == "selfsub":
            return ("-", e, e)
        if kind == "selfdiv":
            return ("/", e, e)
        if kind == "lnself":
            return ("u", "ln", ("/", e, e))
        if kind == "mulzero":
            return ("*", e, ("c", 0.0))
        c = random.choice(CONSTS)
        return ("+", e, ("-", ("c", c), ("c", c)))
    if r < 0.55:
        return ("u", random.choice(list(UN)), gen(depth - 1, zeros))
    op = random.choice(["+", "-", "*", "/", "ipow", "rpow", "cpow"])
    if op == "ipow":
        return ("ipow", gen(depth - 1, zeros), random.choice([2, 3]))
    if op == "rpow":
        return ("rpow", gen(depth - 1, zeros), random.choice([0.5, 1.5, -0.5]))
    return (op, gen(depth - 1, zeros), gen(depth - 1, zeros))


def show(t):
    k = t[0]
    if k == "x": return "x"
    if k == "c": return repr(t[1])
    if k == "u": return f"{t[1]}({show(t[2])})"
    if k == "ipow": return f"({show(t[1])})**{t[2]}"
    if k == "rpow": return f"power({show(t[1])}, {t[2]})"
    if k == "cpow": return f"power({show(t[1])}, {show(t[2])})"
    return f"({show(t[1])} {k} {show(t[2])})"


def safe(f, x):
    try:
        return f(x)
    except (ZeroDivisionError, ValueError, OverflowError):
        return mp.nan


def ident_zero(f, a):
    """f identically zero near a: zero to 1e-40 at three nearby points (50 digits)."""
    with mp.workdps(50):
        for p in (a, a + mp.mpf("0.0137"), a - mp.mpf("0.0211")):
            v = safe(f, p)
            if not mp.isfinite(v) or abs(v) > mp.mpf("1e-40"):
                return False
    return True


def ev(t, seed, a, events):
    """(comp, cls, den); cls is None where the classical reading is undefined."""
    k = t[0]
    if k == "x":
        return seed, (lambda x: x), (lambda x: x)
    if k == "c":
        c = mp.mpf(t[1])
        return R(t[1]), (lambda x, c=c: c), (lambda x, c=c: c)

    def conv(f):
        if ident_zero(f, a):
            events.add("R1")
            return lambda x: x - a
        return f

    def unit(f):
        return ident_zero(lambda x: f(x) - 1, a)

    if k == "u":
        fc, fm = UN[t[1]]
        c, s, d = ev(t[2], seed, a, events)
        dd = conv(d)
        return (fc(c), (lambda x: fm(s(x))) if s else None, lambda x: fm(dd(x)))
    if k in ("ipow", "rpow"):
        c, s, d = ev(t[1], seed, a, events)
        p = t[2]
        if k == "ipow":
            dd = conv(d)
            return c ** p, (lambda x: s(x) ** p) if s else None, lambda x: dd(x) ** p
        q = mp.mpf(p)
        dd = conv(d)
        lg = conv(lambda x: mp.log(dd(x)))
        return (power(c, p), (lambda x: s(x) ** q) if s else None,
                lambda x: mp.exp(q * lg(x)))
    c1, s1, d1 = ev(t[1], seed, a, events)
    c2, s2, d2 = ev(t[2], seed, a, events)
    both = s1 is not None and s2 is not None
    if k == "cpow":
        L = conv(d1); Rr = conv(d2)
        lg = conv(lambda x: mp.log(L(x)))
        return (power(c1, c2), (lambda x: s1(x) ** s2(x)) if both else None,
                lambda x: mp.exp(Rr(x) * lg(x)))
    L, Rr = conv(d1), conv(d2)
    if k in "+-":
        comp = c1 + c2 if k == "+" else c1 - c2
        sg = 1 if k == "+" else -1
        cls_ = (lambda x: s1(x) + sg * s2(x)) if both else None
        den = lambda x: L(x) + sg * Rr(x)
        if ident_zero(den, a):
            events.add("cancel")
            den = lambda x: L(x) * (x - a)
        return comp, cls_, den
    if k == "*":
        den = d2 if unit(d1) else d1 if unit(d2) else (lambda x: L(x) * Rr(x))
        return c1 * c2, (lambda x: s1(x) * s2(x)) if both else None, den
    if k == "/":
        cls_ = None
        if both and not ident_zero(s2, a):
            cls_ = lambda x: s1(x) / s2(x)
        den = d1 if unit(d2) else (lambda x: L(x) / Rr(x))
        return c1 / c2, cls_, den
    raise ValueError(k)


def ref_derivs(f, a):
    """Derivatives 0..NMAX, or 'domain' / 'unreliable'."""
    runs = []
    for dps in (40, 60):
        with mp.workdps(dps):
            try:
                cs = mp.taylor(f, a, NMAX)
            except (ZeroDivisionError, ValueError, OverflowError):
                return "domain"
            out = []
            for n, c in enumerate(cs):
                if not mp.isfinite(c):
                    return "unreliable"
                if abs(mp.im(c)) > mp.mpf("1e-30") * max(1, abs(mp.re(c))):
                    return "domain"
                out.append(mp.re(c) * mp.factorial(n))
            runs.append(out)
    for u, v in zip(*runs):
        if abs(u - v) > AGREE * max(1, abs(v)):
            return "unreliable"
    return [float(v) for v in runs[1]]


def run(ncases, zeros, rng_seed):
    random.seed(rng_seed)
    st = dict(cases=0, skipped=0, errors=0, scored=0, passed=0, failed=0,
              ref_unreliable=0,
              ref_timeouts=0, cut_by_complete_order=0)
    fails, errs, missed, unrel, tos, cut, passes = [], [], [], [], [], [], []
    skip_reasons, skip_examples = {}, {}
    while st["cases"] < ncases:
        t = gen(random.choice([2, 3, 4]), zeros)
        af = random.choice([0.35, 0.7, 1.1, 1.6, 2.3])
        a = mp.mpf(af)                      # the same binary double the library seeds
        events = set()
        expr = show(t)
        signal.alarm(REF_SECONDS)
        try:
            with mp.workdps(50):
                with cl._refusing_float():
                    comp, s, d = ev(t, R(af) + ZERO, a, events)
            rc = ref_derivs(s, a) if s is not None else "undefined"
            rd = ref_derivs(d, a)
        except RefTimeout:
            st["ref_timeouts"] += 1; tos.append((expr, af)); continue
        except InexactGradeError as e:     # library refuses to BUILD it on this backend
            st["build_refused"] = st.get("build_refused", 0) + 1
            errs.append((expr, af, "build: InexactGradeError", str(e)[:70])); continue
        except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e:
            if isinstance(e, cl.FloatCoercionError):
                raise
            st["skipped"] += 1
            key = type(e).__name__ + (": " + str(e).splitlines()[0][:70] if str(e) else "")
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
            skip_examples.setdefault(key, (expr, af))
            continue
        finally:
            signal.alarm(0)
        if not isinstance(rc, list) and not isinstance(rd, list):
            if "unreliable" in (rc, rd):
                st["ref_unreliable"] += 1; unrel.append((expr, af, rc, rd))
            else:
                st["skipped"] += 1
            continue
        st["cases"] += 1
        try:
            kden = comp.denotation_order
            cord = comp.complete_order
            got = []
            for n in range(NMAX + 1):
                try:
                    got.append(comp.st() if n == 0 else comp.d(n))
                except cl.StandardPartUndefinedError:
                    got.append(float("inf"))
                except NotImplementedError:
                    got.append("refused")            # per order, not per case
        except Exception as e:
            st["errors"] += 1; errs.append((expr, af, type(e).__name__, str(e)[:100]))
            continue
        for n in range(NMAX + 1):
            # CORRECTNESS: always against the composite-semantics reference.
            # denotation_order is a warning for the user, not an oracle, so it
            # plays no part in deciding what the right value is.
            ref = rd
            if cord is not None and n > cord:
                st["cut_by_complete_order"] += 1; cut.append((expr, af, n, cord)); continue
            if got[n] == "refused":
                if isinstance(ref, list) and math.isfinite(ref[n]):
                    st["refused_but_ref_finite"] = st.get("refused_but_ref_finite", 0) + 1
                    fails.append((expr, af, n, "refused", "ref finite", ref[n], kden, sorted(events)))
                else:
                    st["refused_ref_also_singular"] = st.get("refused_ref_also_singular", 0) + 1
                continue
            if not isinstance(ref, list):
                unrel.append((expr, af, n, ref, got[n]))
                st["orders_ref_" + ref] = st.get("orders_ref_" + ref, 0) + 1
                continue
            st["scored"] += 1
            g, r = got[n], ref[n]
            ok = (not math.isinf(g)) and abs(g - r) <= TOL * max(1.0, abs(r))
            line = (expr, af, n, g, r, kden, sorted(events))
            (passes if ok else fails).append(line)
            st["passed" if ok else "failed"] += 1
            # WARNING COVERAGE, reported apart from correctness: where the
            # composite value is not the conventional one, does d(n) warn?
            if isinstance(rc, list) and math.isfinite(rc[n]) and math.isfinite(r):
                if abs(r - rc[n]) > TOL * max(1.0, abs(rc[n])):
                    st["non_conventional_orders"] = st.get("non_conventional_orders", 0) + 1
                    warns = n >= 1 and kden is not None and n >= kden
                    if not warns:
                        st["unwarned"] = st.get("unwarned", 0) + 1
                        missed.append((expr, af, n, g, rc[n], kden))
    return st, fails, errs, missed, unrel, tos, cut, passes, skip_reasons, skip_examples


if __name__ == "__main__":
    import argparse
    from composite.backends import config
    ap = argparse.ArgumentParser(description="Brute-force derivative battery.")
    ap.add_argument("cases", type=int, nargs="?", default=100,
                    help="scored cases per class (default 100)")
    ap.add_argument("--seed", type=int, default=1,
                    help="base seed; the zero-injected class uses seed + 1")
    ap.add_argument("--backend", default="sparse_dense",
                    choices=["sparse_dense", "dict", "dense_series", "fractional_dict"])
    args = ap.parse_args()
    getattr(config, "use_" + args.backend)()
    cl._refresh_constants()
    # the module-level names were bound at import, before the switch
    ZERO = cl.ZERO
    print(f"backend {type(get_backend()).__name__}, orders 0..{NMAX}, tol {TOL} rel, "
          f"mpmath taylor at 40 and 60 digits (agree {AGREE}), seed {args.seed}/+1",
          flush=True)
    for label, zeros, rs in (("classical (no injected zeros)", False, args.seed),
                             ("zero-injected", True, args.seed + 1)):
        (st, fails, errs, missed, unrel, tos, cut, passes,
         skip_reasons, skip_examples) = run(args.cases, zeros, rs)
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
