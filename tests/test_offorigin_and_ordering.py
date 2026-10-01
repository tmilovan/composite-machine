"""Three things that worked on one backend, or at one point, and nowhere else.

Each was found by writing a demo, not by a failing test, because nothing here
exercised them:

  residue(f, at)      built Composite({0: complex(at), ...}) for a non-zero
  pole_order(f, at)   expansion point.  Coefficients are float64, so both
                      raised TypeError everywhere except the origin -- and the
                      README and API Reference illustrate only residue(1/z, 0).

  _compare            promotes through _operands, which keys off the DATA TYPE.
                      VectorDimBackend subclasses DictBackend, so on the dict
                      backend both sides hold DictData, the promotion is
                      skipped, and np.union1d sorted a tuple dim against a
                      float one and raised.  `1/h > ln(1/h)` therefore answered
                      on sparse-dense and was a TypeError on dict.  T8 in
                      test_standalone asserts a total order and passes, because
                      it runs on the default backend where the type test fires.

The ordering checks run on every backend for that reason: a single-backend
ordering test is what let this through.
"""
import math

import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, exp, ln, sqrt
from composite.composite_extended import pole_order, residue
from composite.backends import config

BACKENDS = ["dict", "sparse_dense", "dense_series"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    getattr(config, "use_" + request.param)()
    cl._refresh_constants()
    cl.MAX_ACTIVE_DIMS = 10 ** 9
    yield request.param
    config.use_sparse_dense()
    cl._refresh_constants()


# --- residue away from the origin -------------------------------------------

RESIDUES = [
    ("1/(z-1) at 1", lambda z: R(1) / (z - R(1)), 1.0, 1.0),
    ("5/(z-4) at 4", lambda z: R(5) / (z - R(4)), 4.0, 5.0),
    ("1/(z+2) at -2", lambda z: R(1) / (z + R(2)), -2.0, 1.0),
    ("1/((z-1)(z-3)) at 1", lambda z: R(1) / ((z - R(1)) * (z - R(3))), 1.0, -0.5),
    ("1/((z-1)(z-3)) at 3", lambda z: R(1) / ((z - R(1)) * (z - R(3))), 3.0, 0.5),
    ("z/((z-1)(z-2)) at 1", lambda z: z / ((z - R(1)) * (z - R(2))), 1.0, -1.0),
    ("z/((z-1)(z-2)) at 2", lambda z: z / ((z - R(1)) * (z - R(2))), 2.0, 2.0),
    ("exp(z)/(z-2) at 2", lambda z: exp(z) / (z - R(2)), 2.0, math.exp(2)),
    # not a SIMPLE pole, so there is no 1/(z-a) term to report
    ("1/(z-1)^2 at 1", lambda z: R(1) / ((z - R(1)) ** 2), 1.0, 0.0),
    # not a pole at all
    ("1/(z-1) at 5", lambda z: R(1) / (z - R(1)), 5.0, 0.0),
]


@pytest.mark.parametrize("label,f,at,want", RESIDUES, ids=[r[0] for r in RESIDUES])
def test_residue_away_from_the_origin(label, f, at, want):
    got = residue(f, at=at)
    assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), \
        "%s: got %r, want %r" % (label, got, want)


ORDERS = [
    ("1/(z-2)^1 at 2", 1), ("1/(z-2)^2 at 2", 2),
    ("1/(z-2)^3 at 2", 3), ("1/(z-2)^5 at 2", 5),
]


@pytest.mark.parametrize("label,n", ORDERS, ids=[o[0] for o in ORDERS])
def test_pole_order_away_from_the_origin(label, n):
    got = pole_order(lambda z: R(1) / ((z - R(2)) ** n), at=2.0)
    assert got == n, "%s: got %r, want %r" % (label, got, n)


def test_pole_order_sees_a_transcendental_numerator_and_a_clean_point():
    assert pole_order(lambda z: exp(z) / ((z - R(3)) ** 2), at=3.0) == 2
    assert pole_order(lambda z: exp(z), at=3.0) == 0


def test_the_origin_still_works():
    assert abs(residue(lambda z: R(1) / z, at=0) - 1.0) < 1e-12
    assert pole_order(lambda z: R(1) / z, at=0) == 1


# --- ordering across mixed dimension kinds ----------------------------------

def _h():
    return Composite({-1: 1.0})


def ORDERINGS():
    """(label, bigger, smaller).  Spans power, half and log axes deliberately."""
    h = _h()
    inf = R(1) / h
    lg = ln(R(1) / h)
    return [
        ("1/h > ln(1/h)            power beats log", inf, lg),
        ("ln(1/h) > 1              log beats finite", lg, R(1)),
        ("1 > sqrt(h)              finite beats half", R(1), sqrt(h)),
        ("sqrt(h) > h              half beats first order", sqrt(h), h),
        ("ln(1/h) > h", lg, h),
        ("1 > h*ln(1/h)            a log cannot rescue a power", R(1), h * lg),
        ("h*ln(1/h) > h", h * lg, h),
        ("ln(1/h) > lnln(1/h)      one axis deeper", lg, ln(lg)),
        ("|5|_1 > |100|_0          grade decides, not size", Composite({1: 5.0}),
         Composite({0: 100.0})),
    ]


def test_ordering_holds_on_every_backend(backend):
    for label, big, small in ORDERINGS():
        assert big > small, "%s: %r > %r came back False on %s" % (
            label, big, small, backend)
        assert small < big, "%s: the converse failed on %s" % (label, backend)
        assert not (big < small), "%s: both directions true on %s" % (label, backend)


def test_ordering_is_trichotomous_across_dim_kinds(backend):
    vals = [v for _, a, b in ORDERINGS() for v in (a, b)]
    for x in vals:
        for y in vals:
            lt, eq, gt = x < y, x == y, x > y
            assert (lt + eq + gt) == 1, \
                "%r vs %r gave lt=%s eq=%s gt=%s on %s" % (x, y, lt, eq, gt, backend)


def test_equal_composites_compare_equal_on_every_backend(backend):
    for v in (R(3) + ZERO, ln(R(1) / _h()), sqrt(_h()), Composite({1: 5.0})):
        assert v == v and not (v > v) and not (v < v)


def test_sorting_a_mixed_list_does_not_raise(backend):
    h = _h()
    items = [R(1) / h, ln(R(1) / h), sqrt(h), h, R(1)]
    ordered = sorted(items)
    assert ordered[0] is not None                     # it sorted at all
    for lo, hi in zip(ordered, ordered[1:]):
        assert not (lo > hi), "sorted() produced %r before %r on %s" % (lo, hi, backend)
