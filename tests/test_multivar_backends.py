"""composite_multivar on every storage backend.

The directional functions evaluate ordinary one-infinitesimal composites, so they
need nothing from a backend beyond ordinary arithmetic.  This runs a
representative set on each backend and checks it against known values, printing
got against known; tolerances are fixed and match test_multivar_jets.

Measured 2026-10-07: test_multivar_jets, test_multivar_disprove and
test_multivar_extended pass unchanged on all five backends below.
test_composite_vector does too, except on dense-series, where the sphere flux
integral raises in the 2D integrator (_parts_sum: "dimensions are not on a
single lattice") -- the dense-series lattice item in OPEN_ITEMS, an integration
issue, not a multivariable one.
"""
import math
import time

import pytest

import composite.backends.config as C
from composite.composite_lib import LimitDoesNotExistError, R, ResidueError, exp, sin, sqrt
from composite.composite_multivar import (PoleError, curl_at, directional_derivative, gradient_at,
                                          hessian_at, laplacian_at, multivar_limit,
                                          partial_derivative)

TOL = 1e-13
BACKENDS = ["sparse_dense", "dict", "dense_series", "fractional_dict", "fractional_numpy"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    before = C.get_backend()
    getattr(C, "use_" + request.param)()
    try:
        yield request.param
    finally:
        C.set_backend(before)


def check(label, got, want):
    err = abs(got - want) / max(1.0, abs(want))
    print(f"\n  {label:52s} got {got!r:24} known {want!r:24} rel err {err:.1e}")
    assert err <= TOL


CASES = [
    ("d/dx x^2 y at (3,2)", lambda: gradient_at(lambda x, y: x * x * y, [3, 2])[0], 12.0),
    ("d2/dxdy exp(xy) at (1,1)", lambda: hessian_at(lambda x, y: exp(x * y), [1, 1])[0][1], 2 * math.e),
    ("d/dx x/y at (2,3)", lambda: gradient_at(lambda x, y: x / y, [2, 3])[0], 1 / 3),
    ("d3/dxdydz xyz at (2,3,4)",
     lambda: partial_derivative(lambda x, y, z: x * y * z, [2, 3, 4], [1, 1, 1]), 1.0),
    ("laplacian 1/r at (1,2,2)",
     lambda: laplacian_at(lambda x, y, z: R(1) / sqrt(x * x + y * y + z * z), [1, 2, 2]), 0.0),
    ("curl [y-z, z-x, x-y] at (1,1,1) z",
     lambda: curl_at([lambda x, y, z: y - z, lambda x, y, z: z - x, lambda x, y, z: x - y], [1, 1, 1])[2], -2.0),
    ("d/dv exp(xy) at (1,2) along (3,4)",
     lambda: directional_derivative(lambda x, y: exp(x * y), [1, 2], [3, 4]), (2 * 0.6 + 0.8) * math.exp(2)),
    ("lim sin(x^2+y^2)/(x^2+y^2) at (0,0)",
     lambda: multivar_limit(lambda x, y: sin(x * x + y * y) / (x * x + y * y), [0, 0]), 1.0),
    ("d2/dxdy (e^(x+2y)-1)/(x+2y) at (0,0)",
     lambda: hessian_at(lambda x, y: (exp(x + 2 * y) - R(1)) / (x + 2 * y), [0.0, 0.0])[0][1], 2 / 3),
]


@pytest.mark.parametrize("label,run,want", CASES, ids=[c[0] for c in CASES])
def test_values(backend, label, run, want):
    t = time.perf_counter()
    got = run()
    check(f"[{backend}] {label} ({(time.perf_counter() - t) * 1e3:.1f} ms)", got, want)


REFUSALS = [
    ("path-dependent limit", lambda: multivar_limit(lambda x, y: x * y / (x * x + y * y), [0, 0]),
     LimitDoesNotExistError),
    ("cancellation residue", lambda: gradient_at(lambda x, y: exp(x * y) - exp(x * y), [1, 1]), ResidueError),
    ("pole", lambda: gradient_at(lambda x, y: x * y / (x + y), [1, -1]), PoleError),
    ("float function", lambda: gradient_at(lambda x, y: math.exp(x * y), [0.5, 1.0]), ValueError),
]


@pytest.mark.parametrize("label,run,exc", REFUSALS, ids=[c[0] for c in REFUSALS])
def test_refusals(backend, label, run, exc):
    with pytest.raises(exc) as e:
        run()
    print(f"\n  [{backend}] {label}: {type(e.value).__name__}")
