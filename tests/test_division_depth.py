"""Series division at the depth a scope asks for.

Every backend's long division ran max(len(a) + len(b), 50) steps, so under
_derivative_scope(50, 52) 1/(1 + x^2) stopped at order 49 while everything
around it reached 51.  The floor is now the requested depth; with nothing
requested it is still 50.  Tolerance relative, fixed before running: 1e-13.
"""
import mpmath as mp

import composite.composite_lib as cl
from tests.series_check import check_series
from composite.composite_lib import R, ZERO

mp.mp.dps = 60


def _q(x):
    return R(1) / (R(1) + x * x)


def test_default_depth_unchanged():
    r = _q(R(0.5) + ZERO)
    print(f"\n  1/(1 + x^2) at default depth: complete_order {r.complete_order} (want 49)")
    assert r.complete_order == 49


def test_scope_50_reaches_50():
    with cl._derivative_scope(50, 52):
        r = _q(R(0.5) + ZERO)
    print(f"\n  1/(1 + x^2) under scope 50: complete_order {r.complete_order}")
    assert r.complete_order >= 50
    ref = [float(c) for c in mp.taylor(lambda t: 1 / (1 + t * t), mp.mpf(0.5), 50)]
    check_series("1/(1 + x^2) at 0.5, orders 0..50", r.coeffs_dict(), {-n: ref[n] for n in range(51)}, 1e-13)
