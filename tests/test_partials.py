"""partials.py: pure and mixed partial derivatives read from seeded runs.

Every function is written ONCE against a small namespace, so the composite
version and the mpmath reference are the same expression.  References are
exact formulas (the polynomial) or mpmath.diff at 50 digits.  Tolerances are
relative, fixed per test, with no floor; every check prints got against exact.
"""
import math
from itertools import combinations

import mpmath as mp
import pytest

import composite.composite_lib as cl
from composite.backends import config
from composite.partials import partial, partials, Runs, PartialError

mp.mp.dps = 50

BACKENDS = {"sparse_dense": config.use_sparse_dense, "dict": config.use_dict,
            "dense_series": config.use_dense_series}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    BACKENDS[request.param]()
    cl._refresh_constants()
    cl.set_max_order(None)
    yield request.param
    config.use_sparse_dense()
    cl._refresh_constants()
    cl.set_max_order(None)


class C:            # the composite namespace
    exp, sin, cos, ln, ncdf = cl.exp, cl.sin, cl.cos, cl.ln, cl.normal_cdf


class M:            # the mpmath namespace
    exp, sin, cos, ln, ncdf = mp.exp, mp.sin, mp.cos, mp.log, mp.ncdf


def check(label, got, exact, tol):
    exact = mp.mpf(exact)
    # A relative error does not exist at an exact zero, and a floor would turn
    # it into an absolute one without saying so.  Pick points where it is not 0.
    assert exact != 0, f"{label}: the reference is exactly 0; choose another point"
    rel = float(abs(mp.mpf(got) - exact) / abs(exact))
    print(f"  {label:<34} got {got:>24.16g}  exact {float(exact):>24.16g}  rel {rel:.1e}  (tol {tol:g})")
    assert rel <= tol, f"{label}: rel {rel:.2e} > {tol:g}"


def against_mpmath(expr, at, orders_list, tol, backend):
    """partial() for every orders in the list, against mp.diff of the same expr."""
    names = list(at)
    f = lambda **a: expr(C, **a)
    g = lambda *v: expr(M, **dict(zip(names, v)))
    point = [mp.mpf(at[n]) for n in names]
    runs = Runs()
    print(f"\n[{backend}] at {at}")
    for orders in orders_list:
        got = partial(f, at, orders, runs)
        exact = mp.diff(g, point, tuple(orders.get(n, 0) for n in names))
        check(str(orders), got, exact, tol)


def mixed_orders(names, top=3):
    out = [{a: 1, b: 1} for a, b in combinations(names, 2)]
    if top >= 3:
        for a, b in combinations(names, 2):
            out += [{a: 2, b: 1}, {a: 1, b: 2}]
        out += [{a: 1, b: 1, c: 1} for a, b, c in combinations(names, 3)]
    return out


# --- against exact values -------------------------------------------------------

def test_polynomial_against_exact_formulas(backend):
    """f = x^3 y^2 + x y z: every partial is a polynomial written out by hand."""
    f = lambda x, y, z: x ** 3 * y ** 2 + x * y * z
    x, y, z = 1.3, -0.7, 2.1
    exact = {
        (("x", 1),): 3 * x ** 2 * y ** 2 + y * z,
        (("x", 2),): 6 * x * y ** 2,
        (("x", 3),): 6 * y ** 2,
        (("x", 1), ("y", 1)): 6 * x ** 2 * y + z,
        (("x", 2), ("y", 1)): 12 * x * y,
        (("x", 1), ("y", 2)): 6 * x ** 2,
        (("x", 1), ("y", 1), ("z", 1)): 1.0,
        (("y", 1), ("z", 1)): x,
    }
    runs = Runs()
    print(f"\n[{backend}]")
    for key, val in exact.items():
        check(str(dict(key)), partial(f, dict(x=x, y=y, z=z), dict(key), runs), val, 1e-13)


# --- against mpmath ---------------------------------------------------------------

def test_transcendental_mixed_to_third_order(backend):
    expr = lambda m, x, y, z: m.exp(x * y) * m.sin(z) + x * y ** 2 * z ** 3
    against_mpmath(expr, dict(x=0.5, y=1.2, z=0.7), mixed_orders("xyz"), 1e-12, backend)


def test_normal_cdf_and_log(backend):
    expr = lambda m, x, y, z: m.ncdf(x * y) * m.exp(-z) + m.ln(x + z)
    against_mpmath(expr, dict(x=0.8, y=1.1, z=0.4), mixed_orders("xyz"), 1e-12, backend)


def test_an_input_at_zero(backend):
    """x = 0: the input is seeded as the zero itself; a zero meeting a seed in
    x*y or x + z stays an ordinary zero (TAG), so the others are held fixed."""
    expr = lambda m, x, y, z: (m.exp(x * y) + x * m.cos(y) + m.exp(y * z)
                               + (x + z) ** 3 + x * y * z)
    orders = mixed_orders("xyz") + [{"x": 1}, {"x": 2}, {"x": 3}]
    against_mpmath(expr, dict(x=0.0, y=0.5, z=0.7), orders, 1e-12, backend)


def test_every_input_at_zero(backend):
    expr = lambda m, x, y: m.exp(x * y) + x * m.cos(y) + (x + y + 1) ** 3
    orders = mixed_orders("xy") + [{"x": 1}, {"y": 2}]
    against_mpmath(expr, dict(x=0.0, y=0.0), orders, 1e-12, backend)


def test_equal_inputs_do_not_cancel(backend):
    """x = y: identical seeds would make x - y cancel exactly (an identity,
    which converts).  partial() gives them different quantities instead."""
    expr = lambda m, x, y: m.exp(x - y) * x + y ** 2 * x
    against_mpmath(expr, dict(x=0.3, y=0.3), mixed_orders("xy"), 1e-12, backend)


def test_unbalanced_inputs_are_scaled(backend):
    """The inputs' own third-order terms are 1e12 apart; the mixed terms are
    order 1.  Relative quantities alone would leave them at 1e-12 of the
    coefficient; scaled, they arrive balanced."""
    f = lambda x, y: 1e6 * x ** 3 + 1e-6 * y ** 3 + x * y + x ** 2 * y
    at = dict(x=1.0, y=1.0)
    print(f"\n[{backend}]")
    check("{'x': 1, 'y': 1}", partial(f, at, {"x": 1, "y": 1}), 1 + 2 * at["x"], 1e-12)
    check("{'x': 2, 'y': 1}", partial(f, at, {"x": 2, "y": 1}), 2.0, 1e-10)


# --- sharing, scope, refusals ---------------------------------------------------

def test_partials_share_runs_and_agree_with_partial(backend):
    f = lambda x, y, z: cl.exp(x * y) * cl.sin(z) + x * y ** 2 * z ** 3
    at = dict(x=0.5, y=1.2, z=0.7)
    req = {"fx": {"x": 1}, "fxx": {"x": 2}, "fy": {"y": 1}, "fz": {"z": 1},
           "fxy": {"x": 1, "y": 1}, "fxxy": {"x": 2, "y": 1}, "fyz": {"y": 1, "z": 1}}
    runs = Runs()
    together = partials(f, at, req, runs)
    print(f"\n[{backend}] runs {len(runs)}: {sorted(runs.keys())}")
    # x, y, z alone; x and y; x (two zeros); x (two zeros) and y; y and z
    assert len(runs) == 7
    # partials() runs every run to the deepest order requested, a lone partial()
    # only to its own: the same numbers, rounded along shorter series.
    for label, orders in req.items():
        alone = partial(f, at, orders)
        rel = abs(together[label] - alone) / abs(alone)
        print(f"  {label:<5} partials {together[label]:.16g}   partial {alone:.16g}   rel {rel:.1e}")
        assert rel <= 1e-14


def test_order_cap_is_restored():
    f = lambda x, y: cl.exp(x * y)
    cl.set_max_order(None)
    partial(f, dict(x=0.3, y=0.4), {"x": 1, "y": 1})
    assert cl.get_max_order() is None
    cl.set_max_order(7)
    try:
        partial(f, dict(x=0.3, y=0.4), {"x": 2, "y": 1})
        assert cl.get_max_order() == 7
    finally:
        cl.set_max_order(None)


def test_a_shallow_run_is_not_reused_for_a_deeper_request():
    f = lambda x: cl.exp(x)
    runs = Runs()
    partial(f, dict(x=0.5), {"x": 1}, runs)          # made under cap 1
    got = partial(f, dict(x=0.5), {"x": 3}, runs)     # needs order 3
    check("{'x': 3} after {'x': 1}", got, mp.exp(mp.mpf("0.5")), 1e-13)


def test_positional_inputs():
    f = lambda x, y: cl.exp(x * y)
    got = partial(f, [0.3, 0.4], {"x": 1, "y": 1}, names=["x", "y"])
    check("positional {'x': 1, 'y': 1}", got, mp.exp(mp.mpf("0.12")) * (1 + mp.mpf("0.12")), 1e-13)


def test_refuses_a_function_that_drops_the_seed():
    f = lambda x, y: math.exp(x) * y          # math.exp coerces the seed to a float
    with pytest.raises(TypeError):
        partial(f, dict(x=0.3, y=0.4), {"x": 1, "y": 1})


def test_refuses_inside_tag_mode():
    x = cl.TAG(1.0)
    try:
        with pytest.raises(RuntimeError):
            partial(lambda x, y: x * y, dict(x=0.3, y=0.4), {"x": 1})
    finally:
        cl.UNTAG()


def test_refuses_a_denoted_run():
    """f cancels a seeded input exactly against itself: the identity converts
    even under TAG, so that run is not the jet of f's classical reading."""
    f = lambda x, y: (x - x) * y + x * y
    with pytest.raises(PartialError):
        partial(f, dict(x=0.3, y=0.4), {"x": 1, "y": 1})
