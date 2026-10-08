"""composite_extended must not replace composite_lib.exp.

Until 2026-10-07 importing composite_extended (composite_vector does) installed
its own _smart_exp over composite_lib.exp for the whole process.  That copy had
no completeness tracking -- exp, sinh and cosh then claimed to be EXACT however
few terms they summed (a rectangular-barrier T(E) claimed 49 orders with order 7
already wrong) -- and could not take float powers of h (ZERO ** 1.5 raised).
These checks run after the import, which is the state every user of
composite_vector is in.
"""
from fractions import Fraction

import pytest

import composite.composite_lib as cl
import composite.composite_extended as ce            # the import that used to patch
from composite.composite_lib import Composite, ZERO, sinh, sqrt
from composite.backends.config import get_backend, set_backend, use_dict


@pytest.fixture(autouse=True)
def dict_backend():
    before = get_backend()
    use_dict(); cl._refresh_constants()
    yield
    set_backend(before); cl._refresh_constants()


def test_library_exp_is_still_the_library_s():
    print(f"\n  composite_lib.exp: {cl.exp.__module__}.{cl.exp.__name__}   composite_extended.exp is it: {ce.exp is cl.exp}")
    assert cl.exp.__module__ == "composite.composite_lib" and cl.exp.__name__ == "exp"
    assert ce.exp is cl.exp


def test_exp_and_sinh_report_their_completeness():
    e = cl.exp(ZERO)
    s = sinh(sqrt(2 * Composite({-1: 1.0})) * 2.0)
    print(f"\n  exp(h) complete_order {e.complete_order} (want 14)   sinh(2 sqrt(2h)) complete_order {s.complete_order} (want 7.0)")
    assert e.complete_order == 14
    assert s.complete_order == 7.0


def test_float_power_of_h():
    a, b = ZERO ** 1.5, ZERO ** Fraction(3, 2)
    print(f"\n  ZERO ** 1.5 = {a}   ZERO ** Fraction(3,2) = {b}")
    assert a == b
