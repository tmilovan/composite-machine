#!/usr/bin/env python3
"""Brute-force multivariable battery: random expressions against an mpmath reference.

Not a pass/fail suite (pytest does not collect it).  It reports, by reading and
by category, where composite_multivar disagrees with an independent reference.

    python tests/brute_multivar.py 200 --seed 1 --nvars 2 --order 4

Each class draws its own expressions: the first from seed S, the second from
S + 1.  The same seed, nvars and backend always give the same expressions.

THE EXPRESSIONS
  Random trees over the variables (x, y, z), constants and +, -, *, /, integer
  and real powers, power(a, b), sin cos tan exp ln sqrt atan asin acos sinh
  cosh tanh.  The point is drawn per case; a coordinate is 0 one time in ten.

  classical      no injected zeros.  The reference is the classical function.
  zero-injected  adds e - e, e * 0, e / e, ln(e / e) and c - c.  The module
                 refuses an R1 residue (ResidueError), so here the expected
                 outcome is a refusal; anything returned is still scored
                 against the classical reading and listed apart.

THE READINGS, each against mpmath.diff on the same binary doubles
  grad     gradient_at
  hess     hessian_at, and its symmetry H[i][j] == H[j][i]
  jets     taylor_jets to --order: every T_a * a!, |a| <= order
  partial  partial_derivative on one random multi-index, total order 1..6
  lap      laplacian_at against the sum of pure second partials
  dir      directional_derivative along a random direction, against grad . v

  A value is scored with a FIXED tolerance, |got - ref| <= TOL * max(1, |ref|).
  The reference is mpmath.diff at 40 and at 60 digits; if the two disagree
  beyond AGREE it is UNRELIABLE and nothing is scored.  The worst error seen
  per reading is printed, passes included, so the tolerance hides nothing.

OUTPUT
  FAIL        value disagrees with the reference
  REFUSED     the module raised one of its own refusals (pole, branch point,
              non-smooth, residue, incomplete jet), grouped by type
  SUSPECT     one line per case: refusals where the reference has a finite
              value for the reading -- to read, the refusal may be wrong
  RETURNED    zero-injected class only, one line per case: readings came back
              where a refusal was expected, and how they compare with the
              classical reading
  SKIP        the library refused the domain (ln of a negative, ...), grouped
  UNRL        reference unusable (complex, singular, the two precisions disagree)
"""
import math, random, signal, sys, warnings
import mpmath as mp
warnings.simplefilter("ignore")
import composite.composite_lib as cl
from composite.composite_lib import (R, sin, cos, exp, ln, sqrt, tan, atan, asin,
                                     acos, sinh, cosh, tanh, power)
import composite.composite_multivar as mv
from composite.backends.config import get_backend
from composite.backends.base_backend import InexactGradeError

TOL, AGREE = 1e-9, 1e-15
REF_SECONDS = 20
MAX_PARTIAL = 6
VARS = "xyz"

UN = {"sin": (sin, mp.sin), "cos": (cos, mp.cos), "exp": (exp, mp.exp),
      "ln": (ln, mp.log), "sqrt": (sqrt, mp.sqrt), "tan": (tan, mp.tan),
      "atan": (atan, mp.atan), "asin": (asin, mp.asin), "acos": (acos, mp.acos),
      "sinh": (sinh, mp.sinh), "cosh": (cosh, mp.cosh), "tanh": (tanh, mp.tanh)}
CONSTS = [0.5, 2.0, 3.0, -1.5, 1.25]
COORDS = [0.35, 0.7, 1.1, -0.6, 1.6, -1.3]
DIRS = [1.0, -2.0, 0.5, 3.0, -0.75]

#: The module's own refusals: a raise of one of these is a statement, not a crash.
REFUSALS = (mv.PoleError, mv.BranchPointError, mv.NonSmoothError,
            mv.IncompleteJetError, cl.ResidueError)


class RefTimeout(Exception):
    pass


def _alarm(*_):
    raise RefTimeout()


signal.signal(signal.SIGALRM, _alarm)


def gen(depth, nvars, zeros):
    """Random expression tree over nvars variables."""
    if depth == 0 or random.random() < 0.25:
        if zeros and random.random() < 0.15:
            return ("z", ("c", 0.0))
        if random.random() < 0.65:
            return ("v", random.randrange(nvars))
        return ("c", random.choice(CONSTS))
    r = random.random()
    if zeros and r < 0.25:
        e = gen(depth - 1, nvars, zeros)
        kind = random.choice(["selfsub", "selfdiv", "lnself", "mulzero", "constsub"])
        # ("z", tree) marks an injection; it evaluates as the tree itself.
        if kind == "selfsub":
            return ("z", ("-", e, e))
        if kind == "selfdiv":
            return ("z", ("/", e, e))
        if kind == "lnself":
            return ("z", ("u", "ln", ("/", e, e)))
        if kind == "mulzero":
            return ("z", ("*", e, ("c", 0.0)))
        c = random.choice(CONSTS)
        return ("+", e, ("z", ("-", ("c", c), ("c", c))))
    if r < 0.55:
        return ("u", random.choice(list(UN)), gen(depth - 1, nvars, zeros))
    op = random.choice(["+", "-", "*", "/", "ipow", "rpow", "cpow"])
    if op == "ipow":
        return ("ipow", gen(depth - 1, nvars, zeros), random.choice([2, 3]))
    if op == "rpow":
        return ("rpow", gen(depth - 1, nvars, zeros), random.choice([0.5, 1.5, -0.5]))
    return (op, gen(depth - 1, nvars, zeros), gen(depth - 1, nvars, zeros))


def injected(t):
    """Does the tree carry an injected zero or cancellation?"""
    return t[0] == "z" or any(isinstance(c, tuple) and injected(c) for c in t[1:])


def show(t):
    k = t[0]
    if k == "z": return show(t[1])
    if k == "v": return VARS[t[1]]
    if k == "c": return repr(t[1])
    if k == "u": return f"{t[1]}({show(t[2])})"
    if k == "ipow": return f"({show(t[1])})**{t[2]}"
    if k == "rpow": return f"power({show(t[1])}, {t[2]})"
    if k == "cpow": return f"power({show(t[1])}, {show(t[2])})"
    return f"({show(t[1])} {k} {show(t[2])})"


def comp_fn(t):
    """The expression as a function of composites, composite_lib functions only."""
    def ev(t, v):
        k = t[0]
        if k == "z": return ev(t[1], v)
        if k == "v": return v[t[1]]
        if k == "c": return R(t[1])
        if k == "u": return UN[t[1]][0](ev(t[2], v))
        if k == "ipow": return ev(t[1], v) ** t[2]
        if k == "rpow": return power(ev(t[1], v), t[2])
        if k == "cpow": return power(ev(t[1], v), ev(t[2], v))
        a, b = ev(t[1], v), ev(t[2], v)
        return a + b if k == "+" else a - b if k == "-" else a * b if k == "*" else a / b
    return lambda *v: ev(t, v)


def ref_fn(t):
    """The classical function, in mpmath, REAL at every node.

    A complex intermediate raises: sinh(1.25 + ln(x)) at x = -0.6 is real in
    mpmath only because ln(-0.6) carries i*pi and sinh cancels it, and a real
    library refuses ln of a negative -- correctly.  Without this the battery
    reported that refusal as suspect."""
    def real(x):
        if isinstance(x, mp.mpc):
            raise ValueError("complex intermediate")
        return x

    def ev(t, v):
        k = t[0]
        if k == "z": return ev(t[1], v)
        if k == "v": return v[t[1]]
        if k == "c": return mp.mpf(t[1])
        if k == "u": return real(UN[t[1]][1](ev(t[2], v)))
        if k == "ipow": return real(ev(t[1], v) ** t[2])
        if k == "rpow": return real(ev(t[1], v) ** mp.mpf(t[2]))
        if k == "cpow": return real(ev(t[1], v) ** ev(t[2], v))
        a, b = ev(t[1], v), ev(t[2], v)
        return a + b if k == "+" else a - b if k == "-" else a * b if k == "*" else a / b
    return lambda *v: ev(t, v)


def ref_partial(f, at, alpha):
    """d^alpha f at `at`, or 'domain' / 'unreliable'."""
    runs = []
    for dps in (40, 60):
        with mp.workdps(dps):
            try:
                v = mp.diff(f, tuple(mp.mpf(a) for a in at), tuple(alpha))
            except (ZeroDivisionError, ValueError, OverflowError):
                return "domain"
            if not mp.isfinite(v):
                return "unreliable"
            if abs(mp.im(v)) > mp.mpf("1e-30") * max(1, abs(mp.re(v))):
                return "domain"
            runs.append(mp.re(v))
    if abs(runs[0] - runs[1]) > AGREE * max(1, abs(runs[1])):
        return "unreliable"
    return float(runs[1])


def unit(n, i):
    return tuple(int(j == i) for j in range(n))


def multi_indices(n, K):
    """Every a with |a| <= K, in n variables."""
    if n == 1:
        return [(k,) for k in range(K + 1)]
    out = []
    for k in range(K + 1):
        for rest in multi_indices(n - 1, K - k):
            out.append((k,) + rest)
    return out


def readings(n, K, rng):
    """[(name, call(f, at) -> value, alphas needed, combine(refs) -> reference)]."""
    a_part = [0] * n
    for _ in range(rng.randint(1, MAX_PARTIAL)):
        a_part[rng.randrange(n)] += 1
    a_part = tuple(a_part)
    v = [rng.choice(DIRS) for _ in range(n)]
    norm = math.sqrt(sum(c * c for c in v))
    vh = [c / norm for c in v]
    out = []
    for i in range(n):
        out.append((f"grad[{VARS[i]}]", lambda f, at, i=i: mv.gradient_at(f, at)[i],
                    [unit(n, i)], lambda r: r[0]))
    for i in range(n):
        for j in range(i, n):
            a = [0] * n; a[i] += 1; a[j] += 1
            out.append((f"hess[{VARS[i]}{VARS[j]}]",
                        lambda f, at, i=i, j=j: mv.hessian_at(f, at)[i][j],
                        [tuple(a)], lambda r: r[0]))
    for a in multi_indices(n, K):
        if sum(a) == 0:
            continue
        out.append((f"jets{a}", lambda f, at, a=a: mv.taylor_jets(f, at, K)[0][a] * math.prod(
                    math.factorial(e) for e in a), [a], lambda r: r[0]))
    out.append((f"partial{a_part}", lambda f, at: mv.partial_derivative(f, at, list(a_part)),
                [a_part], lambda r: r[0]))
    out.append(("lap", lambda f, at: mv.laplacian_at(f, at),
                [tuple(2 * int(j == i) for j in range(n)) for i in range(n)], lambda r: sum(r)))
    out.append((f"dir{tuple(v)}", lambda f, at: mv.directional_derivative(f, at, v),
                [unit(n, i) for i in range(n)], lambda r: sum(g * q for g, q in zip(r, vh))))
    return out


def kind_of(name):
    return name.split("[")[0].split("(")[0]


def run(ncases, n, K, zeros, rng_seed):
    random.seed(rng_seed)
    rng = random.Random(rng_seed * 7919 + 1)
    st = dict(cases=0, skipped=0, scored=0, passed=0, failed=0, refused=0, suspect=0,
              returned=0, ref_unreliable=0, ref_domain=0, ref_timeouts=0, errors=0,
              hess_asymmetric=0)
    worst = {}
    fails, suspects, returned, errs, unrel = [], [], [], [], []
    refused_by, skip_reasons, skip_examples = {}, {}, {}
    while st["cases"] < ncases:
        t = gen(random.choice([2, 3, 4]), n, zeros)
        if zeros and not injected(t):
            continue
        at = [0.0 if rng.random() < 0.1 else rng.choice(COORDS) for _ in range(n)]
        expr = show(t)
        fc, fr = comp_fn(t), ref_fn(t)
        # Domain first, decided by the REFERENCE, not the library: a function that
        # is not real and finite at the point classically is a SKIP.
        try:
            with mp.workdps(30):
                v0 = fr(*[mp.mpf(a) for a in at])
            real = mp.isfinite(v0) and abs(mp.im(v0)) <= mp.mpf("1e-25") * max(1, abs(mp.re(v0)))
        except (ZeroDivisionError, ValueError, OverflowError):
            real = False
        if not real:
            st["skipped"] += 1
            key = "reference not real and finite at the point"
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
            skip_examples.setdefault(key, (expr, at))
            continue
        st["cases"] += 1
        case_ok, case_bad, case_worst = 0, 0, 0.0
        case_suspect = []
        for name, call, alphas, combine in readings(n, K, rng):
            k = kind_of(name)
            # reference
            refs = []
            signal.alarm(REF_SECONDS)
            try:
                for a in alphas:
                    refs.append(ref_partial(fr, at, a))
            except RefTimeout:
                st["ref_timeouts"] += 1; unrel.append((expr, at, name, "timeout")); continue
            finally:
                signal.alarm(0)
            bad = [r for r in refs if isinstance(r, str)]
            ref = None if bad else combine(refs)
            # library
            try:
                got = call(fc, at)
            except REFUSALS as e:
                st["refused"] += 1
                key = f"{k}: {type(e).__name__}"
                refused_by[key] = refused_by.get(key, 0) + 1
                if ref is not None and math.isfinite(ref) and not zeros:
                    st["suspect"] += 1
                    case_suspect.append((name, type(e).__name__, str(e).splitlines()[0][:60]))
                continue
            except InexactGradeError as e:
                st["errors"] += 1; errs.append((expr, at, name, "InexactGradeError", str(e)[:70])); continue
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                # The library refused a domain the reference accepts at the point
                # (the function is real there): its own reason, grouped; SUSPECT
                # when the reference has a finite value for this reading.
                key = f"{k}: {type(e).__name__}: " + (str(e).splitlines()[0][:60] if str(e) else "")
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
                skip_examples.setdefault(key, (expr, at))
                st["refused"] += 1
                if ref is not None and math.isfinite(ref) and not zeros:
                    st["suspect"] += 1
                    case_suspect.append((name, type(e).__name__, str(e).splitlines()[0][:60]))
                continue
            except Exception as e:
                st["errors"] += 1; errs.append((expr, at, name, type(e).__name__, str(e).splitlines()[0][:90]))
                continue
            if ref is None:
                st["ref_" + ("unreliable" if "unreliable" in bad else "domain")] += 1
                unrel.append((expr, at, name, bad[0], got)); continue
            st["scored"] += 1
            err = abs(got - ref) / max(1.0, abs(ref)) if math.isfinite(got) else float("inf")
            worst[k] = max(worst.get(k, 0.0), err)
            ok = err <= TOL
            st["passed" if ok else "failed"] += 1
            line = (expr, at, name, got, ref, f"{err:.1e}")
            if not ok:
                fails.append(line)
            case_ok, case_bad = case_ok + ok, case_bad + (not ok)
            case_worst = max(case_worst, err)
        if case_suspect:
            kinds = sorted({f"{n_}: {e_}: {m_}" for n_, e_, m_ in case_suspect})
            suspects.append((expr, at, f"{len(case_suspect)} readings refused", kinds[:3]))
        if zeros and case_ok + case_bad:
            st["returned"] += 1
            returned.append((expr, at, f"{case_ok} readings agree, {case_bad} wrong, worst {case_worst:.1e}"))
        # Hessian symmetry, read once per case
        try:
            H = mv.hessian_at(fc, at)
            if any(abs(H[i][j] - H[j][i]) > TOL * max(1.0, abs(H[i][j])) for i in range(n) for j in range(n)):
                st["hess_asymmetric"] += 1
                fails.append((expr, at, "hess symmetry", H, "", ""))
        except Exception:
            pass
    return st, worst, fails, suspects, returned, errs, unrel, refused_by, skip_reasons, skip_examples


if __name__ == "__main__":
    import argparse
    from composite.backends import config
    ap = argparse.ArgumentParser(description="Brute-force multivariable battery.")
    ap.add_argument("cases", type=int, nargs="?", default=50, help="cases per class (default 50)")
    ap.add_argument("--seed", type=int, default=1, help="base seed; zero-injected uses seed + 1")
    ap.add_argument("--nvars", type=int, default=2, choices=[2, 3])
    ap.add_argument("--order", type=int, default=4, help="taylor_jets order (default 4)")
    ap.add_argument("--backend", default="sparse_dense",
                    choices=["sparse_dense", "dict", "dense_series", "fractional_dict"])
    ap.add_argument("--classes", default="both", choices=["both", "classical", "zero-injected"])
    args = ap.parse_args()
    getattr(config, "use_" + args.backend)()
    cl._refresh_constants()
    print(f"backend {type(get_backend()).__name__}, {args.nvars} variables, jets to order "
          f"{args.order}, partials to {MAX_PARTIAL}, tol {TOL} rel, mpmath.diff at 40 and 60 "
          f"digits (agree {AGREE}), seed {args.seed}/+1", flush=True)
    classes = [("classical", False, args.seed), ("zero-injected", True, args.seed + 1)]
    if args.classes != "both":
        classes = [c for c in classes if c[0] == args.classes]
    for label, zeros, rs in classes:
        (st, worst, fails, suspects, returned, errs, unrel, refused_by,
         skip_reasons, skip_examples) = run(args.cases, args.nvars, args.order, zeros, rs)
        print(f"\n=== {label}: {st}", flush=True)
        print("  worst relative error per reading (passes included): "
              + ", ".join(f"{k} {v:.1e}" for k, v in sorted(worst.items())))
        for k_, n_ in sorted(refused_by.items(), key=lambda kv: -kv[1]):
            print(f"  REFUSED {n_:4d}  {k_}")
        for k_, n_ in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  SKIP    {n_:4d}  {k_}   e.g. {skip_examples[k_]}")
        for tag, rows in (("FAIL", fails), ("SUSPECT", suspects), ("RETURNED", returned),
                          ("ERR ", errs), ("UNRL", unrel)):
            for r_ in rows:
                print(f"  {tag}", r_)
