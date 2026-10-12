#!/usr/bin/env python3
"""Brute-force composite arithmetic against classical arithmetic (mpmath).

Not a pass/fail suite (pytest does not collect it).  Where the composite model
of tests/brute_composite.py asks "are the rules implemented as specified?",
this asks "where does composite arithmetic depart from classical arithmetic,
and is every departure one the zero rules account for?"  Mismatches are
expected; the point is where they fall.

    python tests/brute_classical.py 5000 --seed 1

Random real-valued expressions over + - * / and exp ln sqrt sin cos, leaves
including written zeros and decimals, with the provenance patterns built on
purpose (e - e, e / e, e * 0, (e - e)/(f - f), a + b - c with c the decimal sum).
Each is evaluated three ways:

  composite   R(leaf), composite operations and functions, read with st()
  mpmath      50 digits on the exact binary value of each leaf: the classical truth
  float       plain Python floats and math: classical float64, which separates
              what float64 does from what composite does

  AGREE           composite matches mpmath within TOL relative
  ROUNDING        composite differs from mpmath but equals plain float64
  EXTENSION       classical is undefined (a zero divisor, 0/0) and composite
                  gives a value -- finite, or infinite (no standard part)
  BOTH_UNDEFINED  classical is undefined and composite refuses or has no st()
  FLOAT_UNDEFINED the exact value is defined but float64 rounded a divisor to
                  exactly 0, so classical float64 is undefined too; composite
                  sees the same zero and extends from it
  PROVENANCE      classical is defined, default composite differs or refuses,
                  and composite under conventional() -- zeros as ordinary
                  zeros -- agrees: the zero rules made the difference, as they
                  are meant to (a cancelled -0.1 is -0.1*h, and its sqrt is
                  not real)
  UNEXPECTED      classical is defined and composite differs even under
                  conventional() -- the cases to read
"""
import argparse
import math
import random
import warnings

import mpmath as mp

TOL = 1e-12
LEAVES = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 0.1, 0.2, 0.3, -0.3, 0.7, 1.1, 2.5, 0.05, 0.35]
FUNCS = ["exp", "ln", "sqrt", "sin", "cos"]


def rand_expr(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return ("c", rng.choice(LEAVES))
    r = rng.random()
    if r < 0.07:
        a, b = rng.choice([0.1, 0.05, 0.2, 0.3]), rng.choice([0.2, 0.3, 0.6])
        return ("-", ("+", ("c", a), ("c", b)), ("c", float(f"{a + b:.10g}")))
    e = rand_expr(rng, depth - 1)
    if r < 0.15:
        return ("-", e, e)
    if r < 0.21:
        return ("/", e, e)
    if r < 0.26:
        return ("*", e, ("c", 0.0))
    if r < 0.31:
        f = rand_expr(rng, depth - 1)
        return ("/", ("-", e, e), ("-", f, f))
    if r < 0.45:
        return ("u", rng.choice(FUNCS), e)
    return (rng.choice(["+", "-", "*", "/"]), e, rand_expr(rng, depth - 1))


def text(t):
    if t[0] == "c":
        return repr(t[1])
    if t[0] == "u":
        return f"{t[1]}({text(t[2])})"
    return f"({text(t[1])} {t[0]} {text(t[2])})"


class Undefined(Exception):
    pass


def ev_mp(t):
    if t[0] == "c":
        return mp.mpf(t[1])
    if t[0] == "u":
        x = ev_mp(t[2])
        f = {"exp": mp.exp, "ln": mp.log, "sqrt": mp.sqrt, "sin": mp.sin, "cos": mp.cos}[t[1]]
        if t[1] in ("ln",) and x <= 0:
            raise Undefined("ln of a non-positive value")
        if t[1] == "sqrt" and x < 0:
            raise Undefined("sqrt of a negative value")
        if t[1] == "exp" and x > 700:
            raise Undefined("exp overflow")
        return f(x)
    a, b = ev_mp(t[1]), ev_mp(t[2])
    if t[0] == "+": return a + b
    if t[0] == "-": return a - b
    if t[0] == "*": return a * b
    if b == 0:
        raise Undefined("division by zero")
    return a / b


def ev_float(t):
    if t[0] == "c":
        return t[1]
    if t[0] == "u":
        x = ev_float(t[2])
        try:
            return {"exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}[t[1]](x)
        except (ValueError, OverflowError):
            raise Undefined(t[1])
    a, b = ev_float(t[1]), ev_float(t[2])
    if t[0] == "+": return a + b
    if t[0] == "-": return a - b
    if t[0] == "*": return a * b
    if b == 0.0:
        raise Undefined("division by zero")
    return a / b


def ev_comp(t, cl):
    if t[0] == "c":
        return cl.R(t[1])
    if t[0] == "u":
        f = {"exp": cl.exp, "ln": cl.ln, "sqrt": cl.sqrt, "sin": cl.sin, "cos": cl.cos}[t[1]]
        return f(ev_comp(t[2], cl))
    a, b = ev_comp(t[1], cl), ev_comp(t[2], cl)
    return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[t[0]]()


def close(x, y):
    if y == 0:
        return x == 0
    return abs(x - y) <= TOL * abs(y)


def run(ncases, seed, depth):
    import composite.composite_lib as cl
    rng = random.Random(seed)
    mp.mp.dps = 50
    stats = {k: 0 for k in ("AGREE", "ROUNDING", "EXTENSION_finite", "EXTENSION_infinite",
                             "BOTH_UNDEFINED", "FLOAT_UNDEFINED", "PROVENANCE", "UNEXPECTED")}
    examples = {k: [] for k in stats}
    for _ in range(ncases):
        t = rand_expr(rng, rng.choice(range(1, depth + 1)))
        try:
            vm = ev_mp(t)
            vm = float(vm) if mp.isfinite(vm) else None
            if vm is not None and not math.isfinite(vm):
                vm = None
        except (Undefined, ZeroDivisionError, OverflowError):
            vm = None
        try:
            vf = ev_float(t)
        except (Undefined, ZeroDivisionError, OverflowError):
            vf = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                c = ev_comp(t, cl)
                try:
                    vc, why = c.st(), str(c)[:60]
                except cl.StandardPartUndefinedError:
                    vc, why = "infinite", str(c)[:60]
            except Exception as e:
                vc, why = "refused", f"{type(e).__name__}: {str(e)[:60]}"
        if vm is None:
            if vc in ("refused", "infinite"):
                cat = "EXTENSION_infinite" if vc == "infinite" else "BOTH_UNDEFINED"
            else:
                cat = "EXTENSION_finite"
        elif isinstance(vc, float) and close(vc, vm):
            cat = "AGREE"
        elif isinstance(vc, float) and vf is not None and close(vc, vf):
            cat = "ROUNDING"
        elif vf is None:
            cat = "FLOAT_UNDEFINED"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    with cl.conventional():
                        vk = ev_comp(t, cl).st()
                except Exception:
                    vk = None
            cat = "PROVENANCE" if vk is not None and (close(vk, vm) or close(vk, vf)) else "UNEXPECTED"
            why += f"  | conventional() st {vk!r}"
        stats[cat] += 1
        if len(examples[cat]) < 12:
            examples[cat].append((text(t), vm, vf, vc, why))
    return stats, examples


def main():
    ap = argparse.ArgumentParser(description="Composite arithmetic against classical arithmetic.")
    ap.add_argument("cases", type=int, nargs="?", default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    print(f"{args.cases} cases, depth <= {args.depth}, seed {args.seed}, mpmath 50 digits, tol {TOL} rel")
    stats, examples = run(args.cases, args.seed, args.depth)
    print(f"\n=== {stats}")
    for cat in ("UNEXPECTED", "PROVENANCE", "FLOAT_UNDEFINED", "EXTENSION_finite",
                "EXTENSION_infinite", "ROUNDING", "BOTH_UNDEFINED"):
        for expr, vm, vf, vc, why in examples[cat][:8]:
            print(f"  {cat:18s} {expr[:90]}\n{'':21s}mpmath {vm!r}  float {vf!r}  composite {vc!r}  [{why}]")


if __name__ == "__main__":
    main()
