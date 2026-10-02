"""Integrating an integrand that divides by a composite.

`integrate` puts the variable of integration on a LANE, which makes its dimensions
VECTORS -- `(0,-1)` rather than `-1`. A composite then holds a scalar dimension
beside a vector one, and `dict_backend.deconvolve` used bare `sorted()`, `max()`,
`r_dim - lead_dim` and `q_dim + d_b`, all of which assume scalars.

Two failure modes came out of that, and the second was the dangerous one:

  integrate(1/(1+x**2), 0, 1)  ->  TypeError: '<' not supported between
                                   instances of 'tuple' and 'int'
  integrate(x/(1+x), 0, 1)     ->  1.0, where the answer is 1 - ln 2 = 0.3069

The second returned the interval WIDTH with no error at all, while the integrand
evaluated correctly at every point. `improper_integral` shared the fault on any
power-law tail, since it divides in the same way.

Fixed by giving deconvolve the dimension-kind-aware helpers `convolve` already
used: `_dim_key` for ordering and `_dim_add`/`_dim_sub` for the arithmetic.
"""
import math

import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, exp, ln, sqrt
from composite.backends import config
from composite.backends.dict_backend import _dim_add, _dim_sub, _dim_key

BACKENDS = ["dict", "sparse_dense", "dense_series"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    getattr(config, "use_" + request.param)()
    cl._refresh_constants()
    yield request.param
    config.use_sparse_dense()
    cl._refresh_constants()


QUOTIENTS = [
    ("x/(1+x)        [0,1]", lambda x: x / (R(1) + x),            0, 1, 1 - math.log(2)),
    ("1/(1+x**2)     [0,1]", lambda x: R(1) / (R(1) + x * x),     0, 1, math.pi / 4),
    ("1/(2+x)        [0,1]", lambda x: R(1) / (R(2) + x),         0, 1, math.log(1.5)),
    ("1/(1+x)        [0,3]", lambda x: R(1) / (R(1) + x),         0, 3, math.log(4)),
    ("x**2/(1+x)     [0,1]", lambda x: x * x / (R(1) + x),        0, 1, 0.5 - 1 + math.log(2)),
]


@pytest.mark.parametrize("label,f,a,b,want", QUOTIENTS, ids=[q[0].split()[0] for q in QUOTIENTS])
def test_a_quotient_integrand_integrates(backend, label, f, a, b, want):
    got = cl.integrate(f, a, b)
    assert abs(got - want) / abs(want) < 1e-9, \
        "%s on %s: got %r, want %r" % (label, backend, got, want)


def test_the_silent_one_specifically(backend):
    """This is the case that returned the interval WIDTH and raised nothing."""
    got = cl.integrate(lambda x: x / (R(1) + x), 0, 1)
    assert abs(got - 1.0) > 0.5, "got exactly the interval width again: %r" % got
    assert abs(got - (1 - math.log(2))) < 1e-12, "got %r, want 1 - ln 2" % got


def test_a_zero_free_integrand_is_unchanged(backend):
    for label, f, a, b, want in (("x**2", lambda x: x * x, 0, 1, 1 / 3.0),
                                 ("sin", lambda x: cl.sin(x), 0, math.pi, 2.0),
                                 ("exp", lambda x: exp(x), 0, 1, math.e - 1),
                                 ("sqrt", lambda x: sqrt(x), 0, 1, 2 / 3.0),
                                 ("ln(1+x)", lambda x: ln(R(1) + x), 0, 1, 2 * math.log(2) - 1)):
        got = cl.integrate(f, a, b)
        assert abs(got - want) / abs(want) < 1e-12, \
            "%s regressed on %s: got %r, want %r" % (label, backend, got, want)


def test_improper_integral_handles_a_power_law_tail():
    config.use_dict(); cl._refresh_constants()
    for label, f, want in (("1/(1+x**2)", lambda x: R(1) / (R(1) + x * x), math.pi / 2),
                           ("1/(1+x)**2", lambda x: R(1) / ((R(1) + x) * (R(1) + x)), 1.0)):
        value, _ = cl.improper_integral(f, 0)
        got = value.st()
        assert abs(got - want) / abs(want) < 1e-7, "%s: got %r, want %r" % (label, got, want)
    config.use_sparse_dense(); cl._refresh_constants()


def test_division_survives_a_vector_dimension():
    """The direct cause: a lane seed is a VECTOR dimension, and dividing by a
    composite that carries one used to raise."""
    config.use_dict(); cl._refresh_constants()
    seed = cl._perturbation_seed(1)
    assert isinstance(next(iter(seed.coeffs_dict())), tuple), \
        "a lane seed should carry a vector dimension: %r" % seed.coeffs_dict()
    x = R(0.5) + seed
    quotient = x / (R(1) + x)                  # raised TypeError before the fix
    # NOT st(): under _dim_key a lane dimension (0,-1) OUTRANKS the scalar 0, so
    # the lane term leads and the quotient is a series along the lane rather than
    # a number with a standard part. That is the representation integrate() wants
    # -- _antiderivative_axis reads exactly that axis -- and the quotient
    # integrals above are what check it is right.
    assert quotient.coeffs_dict(), "the quotient came back empty"
    assert any(isinstance(d, tuple) for d in quotient.coeffs_dict()), \
        "the lane structure was dropped: %r" % quotient.coeffs_dict()
    config.use_sparse_dense(); cl._refresh_constants()


def test_dim_sub_mirrors_dim_add():
    cases = [(2, 1), (-1, -3), ((0, -1), (0, 1)), ((0, -1), 1), (2, (0, -1)),
             ((1, -2, 3), (0, 1))]
    for left, right in cases:
        assert _dim_sub(_dim_add(left, right), right) == left, \
            "sub(add(%r, %r), %r) did not return %r" % (left, right, right, left)
    # and it trims back to a scalar exactly as _dim_add does
    assert _dim_sub((2, 1), (0, 1)) == 2, _dim_sub((2, 1), (0, 1))


def test_dim_key_gives_a_total_order_over_mixed_kinds():
    """The property that matters is only that it is TOTAL. Bare sorted() raises
    on a scalar beside a tuple, and that raise was the bug."""
    mixed = [0, (0, -1), -1, (0, 1), 2, (1, -2)]
    with pytest.raises(TypeError):
        sorted(mixed)                              # what deconvolve used to do
    ordered = sorted(mixed, key=_dim_key)          # and what it does now
    assert len(ordered) == len(mixed)
    assert ordered[-1] == 2, "the largest power component should lead: %r" % ordered
    # the order agrees with the vector backend's, where a longer tuple sorts
    # BELOW a shorter one sharing its prefix: (0,) < (0,-1) is False...
    assert _dim_key((0, -1)) > _dim_key(0), \
        "a lane dimension sorts above the bare scalar 0, by tuple comparison"
