#!/usr/bin/env python3
"""Brute-force composite arithmetic against an executable model of the rules.

Not a pass/fail suite (pytest does not collect it).  It evaluates thousands of
random expressions twice, with the library and with a reference model written
from the specification -- Zero Rules v2 and the Revised Notation System --
in exact rational arithmetic, and reports where the two disagree.

    python tests/brute_composite.py 2000 --seed 1 --backend sparse_dense

THE MODEL (from the specification, not from the code)
  A composite is {grade: coefficient}, Fractions both.  {} is NOTHING.
  R1   a WHOLLY zero operand converts at its lowest grade: 0_d -> 1_(d-1);
       other zero terms stay (R2).
  cancellation at the site: if a + b or a - b is wholly zero while the operands
       were not, the result is the whole quantity of the LEFT operand one grade
       down (a - a = a*h; 2 + (-2) = |2|_-1, (-2) + 2 = |-2|_-1).
  R2   a zero TERM is inert and kept.
  R3   x * 1, 1 * x and x / 1 return x untouched (a zero stays latent).
  R4   + and - add coefficients per grade, shift nothing, keep zero results.
  R5   a product holds every grade its operands construct (Minkowski sum),
       zero-valued ones included.
  R6   no additive identity: R(5) + 0 = <5_0 1_-1>.  NOTHING added or
       subtracted is a no-op, checked before R1: <0_0> + NOTHING stays <0_0>
       (decided 2026-10-12).
  /    single-term divisor: grades subtract, coefficients divide.  Several
       terms: long division from the divisor's highest NONZERO term down,
       zero terms of the dividend inert, at most max(len a + len b, 50) steps.
  st   NOTHING -> 0; a nonzero positive grade -> undefined; else grade 0.
  ==   identity after R1 on both sides, zero terms included.
  <    after R1, lexicographic from the highest grade, absent read as 0.

  Not specified, so not generated: NOTHING - b, NOTHING in * or /, unary
  minus of a wholly zero value.  Counted as 'unspecified', not tested.

HOW A CASE IS SCORED
  Each node is evaluated by both, bottom up, and compared; the FIRST node that
  disagrees is reported with its operation and operands, so a failure points
  at one rule.  A coefficient is a real number at its own grade and is judged
  by its own relative error, |lib - model| <= TOL * |model|.

  VALUE        structural: a grade holds a nonzero on one side only, and not
               as a rounding-sized remainder of the operands
  PRECISION    the same terms on both sides, a coefficient off by more than
               TOL relative: float64 lost digits (typically subtracting nearly
               equal values, or dividing by the result of one)
  ZEROTERM     the values agree; only which inert zero terms exist differs
  FLOATCANCEL  float rounding decided whether a cancellation happened: one
               side's result is wholly zero, or at + and - one side recorded
               the a - a residue and the other did not, or a term exists on one
               side only as a remainder within 1e-12 of the operands' size
  RAISE        the library raised where the model has a value
  *_announced  of those, how many the library announced with a
               PrecisionLossWarning (an amplifying long division)

  Grades past the tighter of the two results' completeness bounds are not
  compared: a cut division and every product of one hold partial sums there,
  in the model as in the library.
  ST / EQ / LT st(), == or < disagree on a result (or a pair of results)
"""
import argparse
import math
import random
import sys
import warnings
from fractions import Fraction as F

TOL = 1e-12
MAX_DIV_STEPS = 50
COEFFS = [F(1), F(2), F(3), F(-1), F(-2), F(1, 2), F(-3, 2), F(1, 4), F(5)]
DECIMALS = ["0.1", "0.2", "0.3", "-0.3", "0.7", "1.1", "-2.2", "3.3", "0.05", "-0.6"]

# Set from the command line.  coeffs: dyadic (exact in float64), binary (decimals;
# the model holds the exact value of the float the library receives, so only the
# library's own rounding can differ) or decimal (the model holds the decimal
# itself: how often float64 changes what 0.1 + 0.2 - 0.3 means).  grades:
# integer, dyadic (halves) or rational (halves and thirds; exact-grade backends).
MODE = {"coeffs": "dyadic", "grades": "integer", "exact_grades": False}


def coef(rng, allow_zero=False):
    """(model value, the float the library receives).  Dyadic draws exactly as
    the original battery did, so a seed reproduces its earlier runs."""
    if MODE["coeffs"] == "dyadic":
        c = rng.choice(COEFFS + [F(0)] if allow_zero else COEFFS)
        return c, float(c)
    if allow_zero and rng.random() < 1 / 10:
        return F(0), 0.0
    sv = rng.choice(DECIMALS)
    return (F(sv) if MODE["coeffs"] == "decimal" else F(float(sv))), float(sv)


def grade_pool():
    if MODE["grades"] == "integer":
        return [F(k) for k in range(-3, 4)]
    halves = [F(k, 2) for k in range(-6, 7)]
    if MODE["grades"] == "dyadic":
        return halves
    return sorted(set(halves) | {F(k, 3) for k in range(-9, 10)})


def lit(terms):
    """A literal: model terms and library terms, keyed by the same exact grade."""
    return ("lit", {g: m for g, (m, _) in terms.items()}, {g: l for g, (_, l) in terms.items()})


# =============================================================================
# THE MODEL
# =============================================================================

def wholly_zero(c):
    return bool(c) and all(v == 0 for v in c.values())


def r1(c):
    if not wholly_zero(c):
        return dict(c)
    d = min(c)
    out = {k: v for k, v in c.items() if k != d}
    out[d - 1] = out.get(d - 1, F(0)) + 1
    return out


def is_unit(c):
    return c == {0: F(1)}


def _cancelled(src):
    return {k - 1: v for k, v in src.items()}


def m_add(a, b, sign=1):
    """a + b (sign 1) or a - b (sign -1)."""
    if not b:
        return dict(a)                        # adding NOTHING is a no-op
    if not a:
        return dict(b) if sign == 1 else None # NOTHING - b: not specified
    a, b = r1(a), r1(b)
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, F(0)) + sign * v
    if wholly_zero(out):
        m_add.cancelled = True
        return _cancelled(a)                  # a - a = a*h, read off the left operand
    return out


def m_mul(a, b):
    if not a or not b:
        return None                           # not specified
    if is_unit(a):
        return dict(b)
    if is_unit(b):
        return dict(a)
    a, b = r1(a), r1(b)
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] = out.get(ka + kb, F(0)) + va * vb
    return out


def m_div(a, b):
    if not a or not b:
        return None                           # not specified
    if is_unit(b):
        return dict(a)
    a, b = r1(a), r1(b)
    if len(b) == 1:
        (kb, vb), = b.items()
        return {k - kb: v / vb for k, v in a.items()}
    bnz = {k: v for k, v in b.items() if v != 0}
    lead = max(bnz)
    lv = bnz[lead]
    rem = {k: v for k, v in a.items() if v != 0}
    q = {}
    for _ in range(max(len(a) + len(b), MAX_DIV_STEPS)):
        if not rem:
            break
        rk = max(rem)
        qk, qv = rk - lead, rem[rk] / lv
        q[qk] = q.get(qk, F(0)) + qv
        for k, v in b.items():
            rem[k + qk] = rem.get(k + qk, F(0)) - qv * v
        rem.pop(rk, None)
        rem = {k: v for k, v in rem.items() if v != 0}
    m_div.cut = bool(rem)
    return q


def lead_order(c):
    nz = [k for k, v in c.items() if v != 0]
    return -max(nz) if nz else None


def m_complete(op, ma, ca, mb, cb, mr, cut):
    """Order through which the model's result is exact; None = exact.

    A cut division is exact through its deepest quotient order; a sum through
    the lower of its operands' bounds; a product through each factor's bound
    plus the other factor's leading order; dividing by one term shifts the
    bound by that term's order; a cut or inexact divisor b with leading order p
    leaves the quotient exact through order(q) + K_b - p."""
    INF = float("inf")
    A = INF if ca is None else ca
    B = INF if cb is None else cb
    if op == "neg":
        k = A
    elif op in "+-":
        k = min(A, B)
    elif op == "*":
        la, lb = lead_order(ma) or 0, lead_order(mb) or 0
        k = min(A + lb, B + la)
    else:                                     # "/"
        p = lead_order(mb) or 0
        lq = lead_order(mr)
        k = A - p
        if B != INF and lq is not None:
            k = min(k, lq + B - p)
        if cut and mr:
            k = min(k, -min(mr))
    return None if k == INF else k


def m_neg(a):
    if wholly_zero(a):
        return None                           # not specified
    return {k: -v for k, v in a.items()}


def m_st(c):
    if not c:
        return F(0)
    if any(k > 0 and v != 0 for k, v in c.items()):
        return "undefined"
    return c.get(0, F(0))


def m_cmp(a, b):
    a, b = r1(a), r1(b)
    for k in sorted(set(a) | set(b), reverse=True):
        x, y = a.get(k, F(0)), b.get(k, F(0))
        if x != y:
            return -1 if x < y else 1
    return 0


def m_eq(a, b):
    return r1(a) == r1(b)


def fmt(v):
    """A model coefficient as text; exact rationals can outgrow float64."""
    try:
        return f"{float(v):g}"
    except OverflowError:
        v = F(v)
        e = round((abs(v.numerator).bit_length() - v.denominator.bit_length()) * 0.30103)
        return f"{'-' if v < 0 else ''}~1e{e}"


def show(c):
    if c is None:
        return "unspecified"
    if not c:
        return "NOTHING"
    return "<" + " ".join(f"{fmt(v)}_{k}" for k, v in sorted(c.items(), reverse=True)) + ">"


# =============================================================================
# EXPRESSIONS
# =============================================================================

def rand_leaf(rng):
    r = rng.random()
    if r < 0.12:
        return ("zero",)                      # a written zero, R(0)
    if r < 0.22:
        return ("ZERO",)
    if r < 0.30:
        return ("INF",)
    if r < 0.36:
        return ("nothing",)
    n = 1 if r < 0.62 else rng.choice([2, 3])
    grades = rng.sample(grade_pool(), n)
    return lit({g: coef(rng, allow_zero=n > 1) for g in grades})


def rand_expr(rng, depth):
    if depth == 0 or rng.random() < 0.2:
        return rand_leaf(rng)
    r = rng.random()
    if MODE["coeffs"] != "dyadic" and r < 0.06:
        # the 0.1 + 0.2 - 0.3 probe: a + b - c with c the DECIMAL sum, one grade
        from decimal import Decimal
        g = rng.choice(grade_pool())
        sa, sb = rng.choice(DECIMALS), rng.choice(DECIMALS)
        sc = str(Decimal(sa) + Decimal(sb))
        val = (lambda sv: (F(sv) if MODE["coeffs"] == "decimal" else F(float(sv)), float(sv)))
        return ("-", ("+", lit({g: val(sa)}), lit({g: val(sb)})), lit({g: val(sc)}))
    e = rand_expr(rng, depth - 1)
    if r < 0.12:
        return ("-", e, e)                    # a cancellation, same value both sides
    if r < 0.20:
        return ("+", e, ("neg", e))
    if r < 0.26:
        return ("/", e, e)
    if r < 0.32:
        return ("neg", e)
    op = rng.choice(["+", "+", "-", "-", "*", "*", "/"])
    return (op, e, rand_expr(rng, depth - 1))


def text(t):
    k = t[0]
    if k == "zero": return "0"
    if k == "ZERO": return "ZERO"
    if k == "INF": return "INF"
    if k == "nothing": return "NOTHING"
    if k == "lit": return show(t[1])
    if k == "neg": return f"-({text(t[1])})"
    return f"({text(t[1])} {k} {text(t[2])})"


# =============================================================================
# THE LIBRARY SIDE
# =============================================================================

class Unspecified(Exception):
    pass


def lib_dict(c):
    return {float(k): float(v) for k, v in c.coeffs_dict().items()}


def _ff(v):
    try:
        return float(v)
    except OverflowError:
        return math.copysign(math.inf, v)


def classify(model, lib, bound=None, opscale=0.0):
    """None when they agree, else the category.

    Every coefficient is a real number at its own grade, judged by its own
    relative error -- a small real is not "almost zero".  Only grades of order
    <= bound are compared: past it one side or both hold partial sums of a cut
    series.  opscale is the largest coefficient of the operands, used only to
    say whether a term present on one side alone is what rounding of those
    operands could leave (FLOATCANCEL) or a real structural difference (VALUE)."""
    mz, lz = wholly_zero(model), bool(lib) and all(v == 0.0 for v in lib.values())
    if bound is not None:
        # a float bound against an exact grade (173/6 against 28.833333333333332):
        # compare as floats, with a margin far below any grade spacing
        model = {k: v for k, v in model.items() if -float(k) <= float(bound) + 1e-9}
        lib = {k: v for k, v in lib.items() if -float(k) <= float(bound) + 1e-9}
    mf = {float(k): _ff(v) for k, v in model.items()}
    nz_m = {k: v for k, v in mf.items() if v != 0.0}
    nz_l = {k: v for k, v in lib.items() if v != 0.0}
    if set(nz_m) == set(nz_l):
        worst = max((abs(nz_l[k] - nz_m[k]) / abs(nz_m[k]) for k in nz_m), default=0.0)
        if worst > TOL:
            return "PRECISION"
        return None if set(mf) == set(lib) else "ZEROTERM"
    if mz != lz:
        return "FLOATCANCEL"
    odd = [nz_m.get(k, 0.0) or nz_l.get(k, 0.0) for k in set(nz_m) ^ set(nz_l)]
    if opscale and all(abs(v) <= 1e-12 * opscale for v in odd):
        return "FLOATCANCEL"
    return "VALUE"


def diff(model, lib, bound=None):
    """What differs, compactly: grades on one side only, and the worst coefficient."""
    if bound is not None:
        model = {k: v for k, v in model.items() if -float(k) <= float(bound) + 1e-9}
        lib = {k: v for k, v in lib.items() if -float(k) <= float(bound) + 1e-9}
    def _f(v):
        try:
            return float(v)
        except OverflowError:
            return math.copysign(math.inf, v)
    mf = {float(k): _f(v) for k, v in model.items()}
    only_m = sorted((k for k in mf if k not in lib), reverse=True)
    only_l = sorted((k for k in lib if k not in mf), reverse=True)
    worst, at = 0.0, None
    for k in set(mf) & set(lib):
        e = abs(lib[k] - mf[k]) / max(1.0, abs(mf[k]))
        if e > worst:
            worst, at = e, k
    part = [f"{len(mf)} model terms, {len(lib)} lib terms"]
    if only_m:
        part.append(f"only model: {[(k, mf[k]) for k in only_m[:4]]}{'...' if len(only_m) > 4 else ''}")
    if only_l:
        part.append(f"only lib: {[(k, lib[k]) for k in only_l[:4]]}{'...' if len(only_l) > 4 else ''}")
    if at is not None and worst > 0:
        part.append(f"worst shared coeff rel err {worst:.1e} at grade {at:g} "
                    f"(model {mf[at]:.6g}, lib {lib[at]:.6g})")
    return "; ".join(part)


def brief(c, n=6):
    t = show(c)
    return t if len(t) <= 110 else t[:110] + f"... ({len(c)} terms)"


def run(ncases, seed, max_depth):
    from composite.composite_lib import Composite, R, StandardPartUndefinedError
    import composite.composite_lib as cl
    rng = random.Random(seed)
    stats = dict(cases=0, nodes=0, agree=0, unspecified=0, VALUE=0, PRECISION=0, ZEROTERM=0,
                 FLOATCANCEL=0, RAISE=0, ST=0, EQ=0, LT=0, precision_warned=0)
    first = {k: [] for k in ("VALUE", "PRECISION", "ZEROTERM", "FLOATCANCEL", "RAISE", "ST", "EQ", "LT")}
    by_op = {}
    results = []

    def leaf(t):
        k = t[0]
        if k == "zero": return {0: F(0)}, Composite({0: 0.0}), None
        if k == "ZERO": return {-1: F(1)}, cl.ZERO, None
        if k == "INF": return {1: F(1)}, cl.INF, None
        if k == "nothing": return {}, Composite({}), None
        key = (lambda g: g) if MODE["exact_grades"] else (lambda g: float(g))
        return dict(t[1]), Composite({key(g): v for g, v in t[2].items()}), None

    while stats["cases"] < ncases:
        t = rand_expr(rng, rng.choice(range(1, max_depth + 1)))
        failure = [None]

        def ev(t):
            if t[0] in ("zero", "ZERO", "INF", "nothing", "lit"):
                try:
                    return leaf(t)
                except Exception as e:            # the library refuses to BUILD the operand
                    if failure[0] is None:
                        failure[0] = ("RAISE", "build", t, dict(t[1]) if t[0] == "lit" else {},
                                      None, dict(t[1]) if t[0] == "lit" else {},
                                      f"{type(e).__name__}: {str(e)[:90]}")
                    raise Unspecified("library refused an operand")
            m_div.cut = False
            m_add.cancelled = False
            if t[0] == "neg":
                ma, la, ca = ev(t[1])
                mb, cb = None, None
                mr = m_neg(ma)
                if mr is None:
                    raise Unspecified("unary minus of a wholly zero value")
                op, lr = "neg", -la
            else:
                (ma, la, ca), (mb, lb, cb) = ev(t[1]), ev(t[2])
                op = t[0]
                mr = {"+": lambda: m_add(ma, mb), "-": lambda: m_add(ma, mb, -1),
                      "*": lambda: m_mul(ma, mb), "/": lambda: m_div(ma, mb)}[op]()
                if mr is None:
                    raise Unspecified(f"{op} with NOTHING")
                try:
                    lr = {"+": lambda: la + lb, "-": lambda: la - lb,
                          "*": lambda: la * lb, "/": lambda: la / lb}[op]()
                except Exception as e:
                    if failure[0] is None:
                        failure[0] = ("RAISE", op, t, ma, mb, mr, f"{type(e).__name__}: {str(e)[:70]}")
                    raise Unspecified("library raised")
            stats["nodes"] += 1
            cr = m_complete(op, ma, ca, mb, cb, mr, m_div.cut)
            lc = lr.complete_order
            bound = cr if lc is None else (lc if cr is None else min(cr, lc))
            if failure[0] is None:
                ops = [la] + ([] if mb is None else [lb])
                opscale = max((abs(v) for o in ops for v in lib_dict(o).values()), default=0.0)
                cat = classify(mr, lib_dict(lr), bound, opscale)
                if cat and op in "+-":
                    # Did each side record a cancellation?  Where rounding decided
                    # it (0.05 + 0.3 rounding exactly onto 0.35) one side carries
                    # the residue and the other a tiny difference: FLOATCANCEL.
                    ld, la_ = lib_dict(lr), lib_dict(la)
                    shifted = {k - 1: v for k, v in la_.items() if v != 0.0}
                    lib_cancelled = bool(shifted) and set(shifted) == {k for k, v in ld.items() if v != 0.0} \
                        and all(abs(ld[k] - v) <= TOL * max(1.0, abs(v)) for k, v in shifted.items())
                    if lib_cancelled != m_add.cancelled:
                        cat = "FLOATCANCEL"
                if cat:
                    failure[0] = (cat, op, t, ma, mb, mr,
                                  f"bounds: model {cr}, lib {lc}, operands model {ca}/{cb} "
                                  f"lib {la.complete_order}/{None if mb is None else lb.complete_order}; "
                                  + diff(mr, lib_dict(lr), bound))
            return mr, lr, cr

        def show_lib(l):
            d = lib_dict(l)
            return "<" + " ".join(f"{v:g}_{k:g}" for k, v in sorted(d.items(), reverse=True)) + ">" if d else "NOTHING"

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m, l, _ = ev(t)
            warned = any(issubclass(w.category, cl.PrecisionLossWarning) for w in caught)
        except Unspecified as u:
            if failure[0] and failure[0][0] == "RAISE":
                warned = False
            else:
                stats["unspecified"] += 1
                continue
        stats["cases"] += 1
        stats["precision_warned"] += warned
        if failure[0]:
            cat = failure[0][0]
            stats[cat] += 1
            if warned:
                stats[cat + "_announced"] = stats.get(cat + "_announced", 0) + 1
                failure[0] = failure[0][:6] + ("[PrecisionLossWarning raised] " + failure[0][6],)
            by_op.setdefault(cat, {}).setdefault(failure[0][1], 0)
            by_op[cat][failure[0][1]] += 1
            if len(first[cat]) < 20:
                first[cat].append(failure[0])
            continue
        stats["agree"] += 1
        # readings: st, and == / < against the previous agreeing result
        try:
            sl = l.st()
        except StandardPartUndefinedError:
            sl = "undefined"
        sm = m_st(m)
        if not ((sl == "undefined" and sm == "undefined") or
                (sl != "undefined" and sm != "undefined" and abs(sl - float(sm)) <= TOL * max(1, abs(float(sm))))):
            stats["ST"] += 1
            if len(first["ST"]) < 8:
                first["ST"].append(("ST", "st", t, m, None, sm, sl))
        if results:
            pm, pl, pt = results[-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq_l, lt_l = (l == pl), (l < pl)
            eq_m, lt_m = m_eq(m, pm), m_cmp(m, pm) < 0
            if eq_l != eq_m:
                stats["EQ"] += 1
                if len(first["EQ"]) < 8:
                    first["EQ"].append(("EQ", "==", t, m, pm, eq_m, eq_l))
            if lt_l != lt_m:
                stats["LT"] += 1
                if len(first["LT"]) < 8:
                    first["LT"].append(("LT", "<", t, m, pm, lt_m, lt_l))
        results.append((m, l, t))
    return stats, first, by_op


def main():
    ap = argparse.ArgumentParser(description="Brute-force composite arithmetic against the rules.")
    ap.add_argument("cases", type=int, nargs="?", default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--backend", default="sparse_dense",
                    choices=["sparse_dense", "dict", "dense_series", "fractional_dict", "fractional_numpy"])
    ap.add_argument("--coeffs", default="dyadic", choices=["dyadic", "binary", "decimal"])
    ap.add_argument("--grades", default="integer", choices=["integer", "dyadic", "rational"])
    ap.add_argument("--warn-cancellation", action="store_true",
                    help="turn on the library's opt-in cancellation warnings and count "
                         "how many FLOATCANCEL / PRECISION cases they announce")
    args = ap.parse_args()
    from composite.backends import config
    import composite.composite_lib as cl
    getattr(config, "use_" + args.backend)()
    cl._refresh_constants()
    if args.warn_cancellation:
        cl.set_cancellation_warnings(True)
    MODE.update(coeffs=args.coeffs, grades=args.grades,
                exact_grades=bool(getattr(config.get_backend(), "EXACT_DIMS", False)))
    if args.grades == "rational" and not MODE["exact_grades"]:
        sys.exit("--grades rational needs an exact-grade backend (fractional_dict or fractional_numpy)")
    print(f"coeffs {args.coeffs}, grades {args.grades}")
    print(f"backend {type(config.get_backend()).__name__}, {args.cases} cases, depth <= {args.depth}, "
          f"seed {args.seed}, tol {TOL} rel", flush=True)
    stats, first, by_op = run(args.cases, args.seed, args.depth)
    print(f"\n=== {stats}")
    for cat, ops in by_op.items():
        print(f"  {cat:12s} first divergence by operation: {ops}")
    for cat, rows in first.items():
        for r in rows:
            if cat in ("ST", "EQ", "LT"):
                _, op, t, m, pm, want, got = r
                print(f"  {cat:12s} {text(t)[:90]}\n               model {show(m)}  "
                      f"{'vs ' + show(pm) if pm is not None else ''}  want {want}  got {got}")
            else:
                _, op, t, ma, mb, mr, got = r
                print(f"  {cat:12s} at '{op}':\n               a     {brief(ma)}\n"
                      f"               b     {brief(mb) if mb is not None else '-'}\n"
                      f"               model {brief(mr)}\n               diff  {got}")


if __name__ == "__main__":
    main()
