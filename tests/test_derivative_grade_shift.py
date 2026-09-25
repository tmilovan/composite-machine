"""Differentiation as a grade shift: |c|_g -> |-g*c|_(g+1).

`d(n)` reads grade -n and multiplies by n!, which is the n-th derivative only
when the composite is a Taylor series in h.  When it is not -- a pole puts the
grades above 0, a branch point puts them on halves -- grade -n is ABSENT, and
an absent dimension reads as 0.0, so d(n) used to report 0.0 for derivatives
that are unbounded.  `D(n)` differentiates instead of reading, and `d(n)` now
falls back to it and takes the standard part, so those cases raise.
"""
import math
import pytest

import composite.composite_lib as cl
from composite.composite_lib import (
    R, ZERO, INF, Composite, StandardPartUndefinedError, sqrt, ln, sin)
from composite.backends import config

BACKENDS = ["dict", "sparse_dense", "dense_series"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    getattr(config, "use_" + request.param)()
    yield request.param
    config.use_sparse_dense()


def grades(c):
    return {g: v for g, v in cl._ensure_composite(c).coeffs_dict().items() if v != 0.0}


def test_pole_derivative_is_unbounded_not_zero(backend):
    # (2 + h)/h = 2/h + 1, a simple pole.  d/dh = -2/h**2, d2/dh2 = 4/h**3.
    x = R(2.0)
    y = (x + ZERO) / (x * 2 - (x * 2))
    assert grades(y) == {1: 2.0, 0: 1.0}, f"got {y}, want |2|_1 + |1|_0"
    assert grades(y.D(1)) == {2: -2.0}, f"got {y.D(1)}, want |-2|_2"
    assert grades(y.D(2)) == {3: 4.0}, f"got {y.D(2)}, want |4|_3"
    assert grades(y.D(3)) == {4: -12.0}, f"got {y.D(3)}, want |-12|_4"


def test_d_refuses_where_it_used_to_return_zero(backend):
    x = R(2.0)
    y = (x + ZERO) / (x * 2 - (x * 2))
    for n in (1, 2, 3, 4, 5):
        with pytest.raises(StandardPartUndefinedError):
            y.d(n)


def test_multiplication_by_a_converted_zero_stays_a_taylor_series(backend):
    # (2 + h)*h = 2h + h**2.  Bounded, so d(n) keeps the fast path.
    x = R(2.0)
    y = (x + ZERO) * (x * 2 - (x * 2))
    assert grades(y) == {-1: 2.0, -2: 1.0}, f"got {y}, want |2|_-1 + |1|_-2"
    got = [y.d(k) for k in range(4)]
    assert got == [0.0, 2.0, 2.0, 0.0], f"got {got}, want [0, 2, 2, 0] for 2h + h**2"


def test_shift_reproduces_the_ordinary_derivative(backend):
    # x*x seeded at 2 is |4|_0 + |4|_-1 + |1|_-2.
    # D(1) is |4|_0 + |2|_-1: standard part f'(2) = 4, its own d(1) f''(2) = 2.
    y = cl._ensure_composite(cl._seeded(2.0) ** 2)
    assert grades(y) == {0: 4.0, -1: 4.0, -2: 1.0}, f"got {y}"
    d1 = y.D(1)
    assert grades(d1) == {0: 4.0, -1: 2.0}, f"got {d1}, want |4|_0 + |2|_-1"
    assert d1.st() == 4.0, f"got {d1.st()}, want f'(2) = 4.0"
    assert d1.d(1) == 2.0, f"got {d1.d(1)}, want f''(2) = 2.0"


def test_factorial_falls_out_of_the_shift(backend):
    # d(n) carries an explicit n!.  Applying the shift n times produces it,
    # so the two agree on every order for a genuine Taylor series.
    f = sin(cl._seeded(2.0)) * cl._seeded(2.0)
    for n in range(1, 5):
        shifted = f.D(n).st()
        read = f.d(n)
        assert abs(shifted - read) < 1e-12, \
            f"order {n}: shift {shifted!r} vs read {read!r}"


def test_half_order_is_unbounded_not_zero(backend):
    # sqrt(h) = |1|_-0.5.  Every integer grade is absent; the derivative
    # 0.5*h**-0.5 is unbounded, not 0.
    s = sqrt(ZERO)
    assert grades(s) == {-0.5: 1.0}, f"got {s}, want |1|_-0.5"
    assert grades(s.D(1)) == {0.5: 0.5}, f"got {s.D(1)}, want |0.5|_0.5"
    with pytest.raises(StandardPartUndefinedError):
        s.d(1)


def test_infinity_differentiates(backend):
    # INF = 1/h, so d/dh is -1/h**2.
    assert grades(INF.D(1)) == {2: -1.0}, f"got {INF.D(1)}, want |-1|_2"
    assert grades(INF.D(2)) == {3: 2.0}, f"got {INF.D(2)}, want |2|_3"


def test_derivative_of_a_constant_is_an_expressed_zero(backend):
    # Not NOTHING.  The derivative of a constant IS zero, and a zero here is
    # a value; returning Composite({}) would say "no derivative" instead.
    d = R(5.0).D(1)
    assert d.coeffs_dict() == {0: 0.0}, f"got {d.coeffs_dict()}, want {{0: 0.0}}"
    assert R(5.0).d(1) == 0.0


def test_log_axis_is_refused_not_dropped(backend):
    # d/dh of ln(1/h) is -1/h, which crosses from the log axis to the power
    # axis, so it is not a shift.  Dropping the term would return a
    # well-formed wrong answer.
    y = ln(R(1) / ZERO)
    assert any(isinstance(g, tuple) for g in y.coeffs_dict()), f"got {y}"
    with pytest.raises(NotImplementedError):
        y.D(1)


def test_ordinary_taylor_reads_are_unchanged(backend):
    # Regression: the fast path must give exactly what it gave before.
    x = cl._seeded(2.0)
    cases = [
        (x * x, [4.0, 4.0, 2.0, 0.0]),
        (R(1) / x, [0.5, -0.25, 0.25, -0.375]),
        (x ** 3, [8.0, 12.0, 12.0, 6.0]),
    ]
    for y, want in cases:
        y = cl._ensure_composite(y)
        got = [y.d(k) for k in range(4)]
        assert all(abs(a - b) < 1e-12 for a, b in zip(got, want)), \
            f"got {got}, want {want} for {y}"
