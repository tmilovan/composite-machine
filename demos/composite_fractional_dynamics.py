"""How fractional (lattice) dimensions BEHAVE while operations run.

Correctness of the fractional backends is covered by tests/test_fractional_backend.py.
This measures the dynamics instead: what the canonical lattice L does along a chain,
how term count grows, what L costs, which aggregation kernel fires, and what picking
a fractional backend costs when no fractional order is present.

Every arm is capped.  Term count under mixed denominators grows MULTIPLICATIVELY
and will run away if it is not bounded, which is the whole point of arm 2.
"""
import math
import time
from fractions import Fraction as F

import numpy as np

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, exp, sin, sqrt, ln
from composite.backends import config
from composite.backends.base_backend import InexactGradeError
from composite.backends.fractional_backend import (FractionalDictBackend,
                                                   FractionalNumpyBackend)


def timed(fn, reps=5):
    """Best-of-N wall clock, in ms.  Best, not mean: the mean is dominated by
    whatever else the machine did during the run."""
    fn()
    best = math.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def pub(be, data):
    dims, vals = be.to_arrays(data)
    return {d: v for d, v in zip(list(dims), list(vals))}


def head(n, s):
    print("\n%s. %s" % (n, s))
    print("   " + "-" * (len(s) + 2))


BD, BN = FractionalDictBackend(), FractionalNumpyBackend()

print("=" * 74)
print("FRACTIONAL DIMENSION DYNAMICS DURING OPERATIONS")
print("=" * 74)

# ---------------------------------------------------------------- arm 1
head(1, "L evolution along a chain of new denominators")
print("   multiply in (1 + h^(1/q)) for successive primes q; L must be the lcm")
print("   %-6s %-14s %6s %8s %11s" % ("q", "L after", "terms", "lcm ok", "convolve ms"))
acc = BN.create_from_terms([0, F(1, 2)], [1.0, 1.0])
want_L = 2
for q in (3, 5, 7, 11, 13, 17):
    b = BN.create_from_terms([0, F(1, q)], [1.0, 1.0])
    ms = timed(lambda a=acc, b=b: BN.convolve(a, b), 3)
    acc = BN.convolve(acc, b)
    want_L = want_L * q // math.gcd(want_L, q)
    print("   %-6d %-14d %6d %8s %11.4f"
          % (q, acc.L, BN.term_count(acc), "yes" if acc.L == want_L else "NO",  ms))

# ---------------------------------------------------------------- arm 2
head(2, "term growth: one denominator (terms collide) vs many (they cannot)")
print("   %-4s %14s %14s %12s" % ("k", "pure 1/2", "mixed denoms", "ratio"))
primes = [2, 3, 5, 7, 11, 13, 17, 19]
for k in range(1, 9):
    p = BN.create_from_terms([0, F(1, 2)], [1.0, 1.0])
    for _ in range(k - 1):
        p = BN.convolve(p, BN.create_from_terms([0, F(1, 2)], [1.0, 1.0]))
    m = BN.create_from_terms([0, F(1, primes[0])], [1.0, 1.0])
    for j in range(1, k):
        m = BN.convolve(m, BN.create_from_terms([0, F(1, primes[j])], [1.0, 1.0]))
    tp, tm = BN.term_count(p), BN.term_count(m)
    print("   %-4d %14d %14d %11.1fx" % (k, tp, tm, tm / tp))
print("   pure grows k+1 (every product lands back on the 1/2 lattice)")
print("   mixed grows 2^k (h^(1/2) and h^(1/3) are different orders, never merge)")

# ---------------------------------------------------------------- arm 3
head(3, "does a large L cost anything?  fixed 200 terms, 200x200 product")
print("   %-16s %-14s %10s %12s" % ("lattice L", "bits in L", "terms", "convolve ms"))
for L in (1, 30, 30030, 6469693230):
    if L == 1:
        dims = list(range(1, 201))
    else:
        dims = [F(i, L) for i in range(1, 201)]
    a = BN.create_from_terms(dims, [1.0 / (i + 1) for i in range(200)])
    ms = timed(lambda a=a: BN.convolve(a, a), 3)
    print("   %-16d %-14d %10d %12.3f" % (L, L.bit_length(), BN.term_count(a), ms))
print("   L is an integer key scale; Python int arithmetic does not care about magnitude")

# ---------------------------------------------------------------- arm 4
head(4, "which aggregation kernel fires, and what it costs")
print("   %-34s %9s %9s %11s %10s" % ("operands", "span", "pairs", "kernel", "ms"))
cases = [("integer series, 200 terms", list(range(200)), list(range(200))),
         ("1/3 and a distant integer", [F(1, 3), 1000], [F(1, 3)]),
         ("two large prime denominators", [F(1, 10007), F(9, 10009)], [F(1, 10007)]),
         ("mixed 1/2 1/3 1/5, 60 terms",
          [F(i, 30) for i in range(1, 61)], [F(i, 30) for i in range(1, 61)])]
for label, da, db in cases:
    a = BN.create_from_terms(da, [1.0] * len(da))
    b = BN.create_from_terms(db, [1.0] * len(db))
    L, ta, tb = BN._align(a, b)
    ka = np.fromiter(ta.keys(), np.int64, len(ta))
    kb = np.fromiter(tb.keys(), np.int64, len(tb))
    span = int(ka.max() + kb.max()) - int(ka.min() + kb.min()) + 1
    npair = ka.size * kb.size
    if span <= BN.DENSE_SPAN_FACTOR * (ka.size + kb.size):
        kern = "convolve"
    elif span <= max(npair * 16, 1 << 16):
        kern = "bincount"
    else:
        kern = "unique"
    ms = timed(lambda a=a, b=b: BN.convolve(a, b), 3)
    print("   %-34s %9d %9d %11s %10.4f" % (label, span, npair, kern, ms))

# ---------------------------------------------------------------- arm 5
head(5, "what a fractional backend costs when NO fractional order is present")
print("   ordinary integer-grade work, 200x200-term product")
series = [1.0 / math.factorial(i) if i < 20 else 1.0 / (i * i) for i in range(200)]
rows = []
for name, setter in (("sparse_dense", config.use_sparse_dense),
                     ("dict", config.use_dict),
                     ("fractional_dict", config.use_fractional_dict),
                     ("fractional_numpy", config.use_fractional_numpy)):
    setter()
    be = cl.get_backend()
    a = be.create_from_terms(list(range(200)), series)
    ms = timed(lambda be=be, a=a: be.convolve(a, a), 3)
    rows.append((name, ms, getattr(a, "L", "n/a")))
base = [m for n, m, _ in rows if n == "sparse_dense"][0]
print("   %-20s %12s %8s %12s" % ("backend", "convolve ms", "L", "vs sparse_dense"))
for name, ms, L in rows:
    print("   %-20s %12.3f %8s %11.2fx" % (name, ms, L, ms / base))
config.use_sparse_dense()

# ---------------------------------------------------------------- arm 6
head(6, "exactness through a chain: (h^(1/q))^q must come back to exactly h")
print("   %-6s %-12s %-22s %-10s %s" % ("q", "dyadic?", "fractional_numpy", "L", "sparse_dense"))
for q in (2, 4, 8, 3, 5, 7, 10, 23, 31):
    root = BN.create(F(1, q), 1.0)
    a = root
    for _ in range(q - 1):
        a = BN.convolve(a, root)
    got = pub(BN, a)
    ok = got == {1: 1.0}
    dyadic = (q & (q - 1)) == 0
    config.use_sparse_dense()
    try:
        c = cl.ZERO ** F(1, q)
        acc2 = c
        for _ in range(q - 1):
            acc2 = acc2 * c
        sd = "exact" if list(acc2.coeffs_dict()) == [-1] else "WRONG %s" % acc2
    except InexactGradeError:
        sd = "InexactGradeError"
    except Exception as e:
        sd = type(e).__name__
    config.use_fractional_numpy()
    print("   %-6d %-12s %-22s %-10d %s"
          % (q, "yes" if dyadic else "no",
             ("h exactly" if ok else "WRONG %s" % got), a.L, sd))

# ---------------------------------------------------------------- arm 7
head(7, "behaviour through the public transcendental ops")
config.use_fractional_numpy()
c = Composite({F(1, 3): 1.0})
print("   Composite({1/3: 1.0}).lead_order()      = %s" % (c.lead_order(),))
print("   (c*c*c).lead_order()                    = %s   (three thirds make a whole)"
      % ((c * c * c).lead_order(),))
print("   (c * Composite({1/5:1})).lead_order()   = %s" % ((c * Composite({F(1, 5): 1.0})).lead_order(),))
print()
print("   %-26s %10s %-26s %s" % ("op on a 1/3-grade value", "terms", "lead order", "ms"))
for label, fn in (("c * c", lambda: c * c),
                  ("c ** 2", lambda: c ** 2),
                  ("exp(c)", lambda: exp(c)),
                  ("sin(c)", lambda: sin(c)),
                  ("sqrt(R(1) + c)", lambda: sqrt(R(1) + c)),
                  ("R(1) / (R(1) + c)", lambda: R(1) / (R(1) + c))):
    try:
        r = fn()
        ms = timed(fn, 3)
        print("   %-26s %10d %-26s %.4f"
              % (label, len(r.coeffs_dict()), str(r.lead_order()), ms))
    except Exception as e:
        print("   %-26s %10s %-26s %s" % (label, "-", type(e).__name__, str(e)[:30]))
config.use_sparse_dense()
print("\n" + "=" * 74)
