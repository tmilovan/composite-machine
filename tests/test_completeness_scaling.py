"""complete_order through products and quotients that shift orders.

A bound is relative to the operand's leading order.  Taking the smaller ABSOLUTE
bound (the rule until 2026-10-07) over-claimed by one whenever the arithmetic
moved coefficients up an order -- X / h and X * (1/h) -- and under-claimed when
it moved them down.  Found reading the transmission T(E) of a rectangular
barrier at E = V0 - h: it claimed complete to order 7, and order 7 was off by
9.5e-7.  Every check prints the claim against the first wrong order.
"""
import math

import pytest

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 50

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, exp, sinh, sqrt


BACKENDS = ["dict", "sparse_dense", "dense_series", "fractional_dict", "fractional_numpy"]


@pytest.fixture(autouse=True, params=BACKENDS)
def backend(request):
    import composite.backends.config as C
    before = C.get_backend()
    getattr(C, "use_" + request.param)(); cl._refresh_constants()
    yield request.param
    C.set_backend(before); cl._refresh_constants()


def first_wrong(c, coeff, orders):
    d = c.coeffs_dict()
    for k in orders:
        want, got = coeff(k), d.get(-k, 0.0)
        if abs(got - want) > 1e-12 * max(1.0, abs(want)):
            return k
    return None


def e(k):                                    # Taylor coefficient of exp
    return 1.0 / math.factorial(k) if k >= 0 else 0.0


def H(): return Composite({-1: 1.0})
def INF(): return Composite({1: 1.0})


CASES = [
    # label, build, true coefficient of order k, the bound the rule gives
    ("X * h", lambda X: X * H(), lambda k: e(k - 1), 6),
    ("X * (1/h)", lambda X: X * INF(), lambda k: e(k + 1), 4),
    ("X / h", lambda X: X / H(), lambda k: e(k + 1), 4),
    ("X / (1/h)", lambda X: X / INF(), lambda k: e(k - 1), 6),
    ("(X h) / (h + h^2)", lambda X: (X * H()) / (H() + H() * H()),
     lambda k: float(mpmath.taylor(lambda r: mpmath.exp(r) / (1 + r), 0, 14)[k]) if 0 <= k <= 14 else 0.0, 5),
]


@pytest.mark.parametrize("label,build,coeff,want_bound", CASES, ids=[c[0] for c in CASES])
def test_bound_sits_just_below_the_first_wrong_order(backend, label, build, coeff, want_bound):
    X = exp(cl.ZERO, terms=6)                 # complete to order 5
    c = build(X)
    bad = first_wrong(c, coeff, range(-2, 14))
    print(f"\n  [{backend}] {label:20s} claims {c.complete_order}   first wrong order {bad}")
    assert c.complete_order == want_bound
    assert bad is not None and bad == want_bound + 1


def test_barrier_transmission_at_the_top(backend):
    E = Composite({0: 1.0, -1: -1.0})          # V0 - h, V0 = 1, a = 2
    T = R(1) / (R(1) + sinh(sqrt(2 * (R(1) - E)) * 2.0) ** 2 / (4 * E * (R(1) - E)))

    def T_ref(x):
        w = 2 * (1 - x) * 4
        s = mpmath.nsum(lambda n: w ** n / mpmath.factorial(2 * n + 1), [0, mpmath.inf])
        return 1 / (1 + 4 * s ** 2 / (2 * x))
    tay = mpmath.taylor(T_ref, 1, 10)
    bad = first_wrong(T, lambda k: float(tay[k]) * (-1) ** k if 0 <= k <= 10 else 0.0, range(0, 10))
    print(f"\n  [{backend}] T(V0 - h) claims {T.complete_order}   first wrong order {bad}")
    assert bad is not None and T.complete_order < bad
