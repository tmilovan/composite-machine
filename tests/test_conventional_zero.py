"""conventional(): an expressed zero behaves as the ordinary zero, in scope.

The default answer for `x*x + R(0)` is the derivative of x**2 + h, and h is
x - a, so the function is x**2 + x - 2 and d1 at 2 is 5.  That is right and it
is the point of the system.  A caller who wrote the zero without meaning it
wants 4, and conventional() gives them 4 without changing the arithmetic
anyone outside the block gets.
"""
import math
import pytest

import composite.composite_lib as cl
from composite.composite_lib import (
    R, ZERO, Composite, conventional, all_derivatives, derivative, sin, ln)
from composite.backends import config

BACKENDS = ["dict", "sparse_dense", "dense_series"]
SQ = [4.0, 4.0, 2.0, 0.0]          # x**2 at x = 2


@pytest.fixture(params=BACKENDS)
def backend(request):
    getattr(config, "use_" + request.param)()
    yield request.param
    config.use_sparse_dense()


ZERO_SPELLINGS = [
    ("written zero, added",     lambda x: x*x + R(0),          2.0, SQ),
    ("written zero, multiplied", lambda x: R(0)*x**3 + x*x,    2.0, SQ),
    ("cancellation, added",     lambda x: x*x + (x - x),       2.0, SQ),
    ("cancellation, subtracted", lambda x: x*x - (x - x),      2.0, SQ),
    ("zero times a transcendental", lambda x: x*x + R(0)*sin(x), 2.0, SQ),
    ("zero in a denominator",   lambda x: x*x/(R(1) + R(0)),   2.0, SQ),
    ("zero on the log axis",    lambda x: ln(x)*R(0) + x,      3.0, [3.0, 1.0, 0.0, 0.0]),
    ("raw python float zero",   lambda x: x*x + 0.0,           2.0, SQ),
    ("accumulator started at zero",
     lambda x: (lambda t: [t := t + x*x/R(3) for _ in range(3)][-1])(R(0)), 2.0, SQ),
]


@pytest.mark.parametrize("label,f,at,want", ZERO_SPELLINGS,
                         ids=[c[0] for c in ZERO_SPELLINGS])
def test_conventional_gives_the_textbook_derivatives(backend, label, f, at, want):
    with conventional():
        got = all_derivatives(f, at, up_to=3)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), \
        f"{label}: got {got}, want {want}"


def test_default_semantics_are_unchanged_outside_the_block(backend):
    # x*x + R(0) is x**2 + h, and h = x - 2, so the function is x**2 + x - 2
    # and its derivative at 2 is 5.  Verified against sympy elsewhere.
    got = all_derivatives(lambda x: x*x + R(0), 2.0, up_to=3)
    assert got == [4.0, 5.0, 2.0, 0.0], f"got {got}, want [4, 5, 2, 0]"


def test_zero_free_formulas_are_identical_in_both_modes(backend):
    f = lambda x: sin(x)*x/(R(1) + x*x)
    plain = all_derivatives(f, 2.0, up_to=4)
    with conventional():
        conv = all_derivatives(f, 2.0, up_to=4)
    assert all(a == b for a, b in zip(plain, conv)), \
        f"conventional() moved a zero-free formula: {plain} vs {conv}"


def test_removable_singularity_still_resolves(backend):
    # (x*x - 4)/(x - 2) at 2.  R2 keeps that zero inert, so no conversion was
    # needed for it and the mode changes nothing: 4 with derivative 1.
    with conventional():
        got = all_derivatives(lambda x: (x*x - R(4))/(x - R(2)), 2.0, up_to=2)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, [4.0, 1.0, 0.0])), \
        f"got {got}, want [4, 1, 0]"


def test_a_zero_divisor_raises_instead_of_returning_inf(backend):
    # Without the third switch the single-term-divisor path divides by 0.0 and
    # hands back |inf|_0 with no error, which is worse than either semantics.
    x = cl._seeded(2.0)
    with conventional():
        for spelling in (lambda: x/R(0), lambda: R(0)/R(0), lambda: x/(x - x)):
            with pytest.raises(ZeroDivisionError):
                cl._ensure_composite(spelling())


def test_division_by_zero_stays_total_by_default(backend):
    # Outside the block it is not an error: the divisor converts and the
    # grades shift up.  x/h at x=2 is 2/h + 1.
    x = cl._seeded(2.0)
    y = cl._ensure_composite(x/R(0))
    assert {g: v for g, v in y.coeffs_dict().items() if v != 0.0} == {1: 2.0, 0: 1.0}, \
        f"got {y}, want |2|_1 + |1|_0"


def test_seeding_at_the_origin_works_in_the_mode(backend):
    # R(0) is |0|_0 in the mode, so _seeded builds the seed directly at both
    # ends and the at == 0 special case disappears.
    with conventional():
        assert cl._seeded(0.0).coeffs_dict() == {0: 0.0, -1: 1.0}
        for f, want in ((lambda x: x**3, [0.0, 0.0, 0.0, 6.0]),
                        (lambda x: sin(x), [0.0, 1.0, 0.0, -1.0])):
            got = all_derivatives(f, 0.0, up_to=3)
            assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), \
                f"got {got}, want {want}"


def test_the_mode_does_not_leak(backend):
    with conventional():
        assert derivative(lambda x: x*x + R(0), 2.0) == 4.0
    assert derivative(lambda x: x*x + R(0), 2.0) == 5.0
    assert Composite.zero().coeffs_dict() == {-1: 1.0}


def test_nesting_restores_the_outer_mode(backend):
    assert derivative(lambda x: x*x + R(0), 2.0) == 5.0
    with conventional():
        with conventional():
            assert derivative(lambda x: x*x + R(0), 2.0) == 4.0
        assert derivative(lambda x: x*x + R(0), 2.0) == 4.0
    assert derivative(lambda x: x*x + R(0), 2.0) == 5.0
