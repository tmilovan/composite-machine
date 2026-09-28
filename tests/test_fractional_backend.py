"""Fractional backends: dimensions on an exact rational lattice.

Elsewhere a dimension is float64, and a dimension is an IDENTIFIER rather than a
magnitude, so 1/3 being unrepresentable means h**(1/3) cubed lands one ulp from
h and never merges with it.  `_add_exact` catches that and raises
InexactGradeError, correctly, which leaves every non-dyadic exponent
unreachable.  These backends store the power axis in units of 1/L so every
dimension is an integer key and the refusal goes away.

L is kept canonical, the lcm of the denominators actually present, so it is a
function of the value and not a free choice: the same orders stored on a coarser
lattice reduce to the same pair, and a result cannot depend on which lattice its
inputs used.  When no fractional order is present canonical L is 1 and storage
is exactly what the other backends hold.
"""
import random
from fractions import Fraction as F

import numpy as np
import pytest

import composite.composite_lib as cl
from composite.composite_lib import R, Composite, sqrt, exp, sin, cos, ln
from composite.backends import config
from composite.backends.base_backend import InexactGradeError, dim_cast
from composite.backends.fractional_backend import (
    FractionalDictBackend, FractionalNumpyBackend, as_fraction)

FLAVOURS = ["fractional_dict", "fractional_numpy"]


@pytest.fixture(params=FLAVOURS)
def flavour(request):
    getattr(config, "use_" + request.param)()
    yield request.param
    config.use_sparse_dense()


def be_of(name):
    return FractionalDictBackend() if name.endswith("dict") else FractionalNumpyBackend()


def pub(be, data):
    dims, vals = be.to_arrays(data)
    return {d: v for d, v in zip(list(dims), list(vals))}


# --- the capability the other backends do not have -----------------------

@pytest.mark.parametrize("q", [2, 3, 5, 7, 10, 16, 23, 31])
def test_qth_root_raised_back_recovers_h(flavour, q):
    be = be_of(flavour)
    root = be.create(F(1, q), 1.0)
    acc = root
    for _ in range(q - 1):
        acc = be.convolve(acc, root)
    assert pub(be, acc) == {1: 1.0}, f"(h^(1/{q}))^{q} gave {pub(be, acc)}, want {{1: 1.0}}"
    assert acc.L == 1, f"lattice should canonicalise to 1, got {acc.L}"


def test_non_dyadic_denominators_are_refused_elsewhere():
    """The premise of this backend: float64 cannot do it and says so."""
    config.use_dict()
    try:
        third = Composite({-1 / 3: 1.0})
        with pytest.raises(InexactGradeError):
            third * third * third
    finally:
        config.use_sparse_dense()


def test_mixed_denominators(flavour):
    be = be_of(flavour)
    got = be.convolve(be.create(F(1, 3), 1.0), be.create(F(1, 5), 1.0))
    assert pub(be, got) == {F(8, 15): 1.0}, pub(be, got)
    assert got.L == 15
    got = be.convolve(be.create(F(1, 3), 1.0), be.create(F(1, 23), 1.0))
    assert pub(be, got) == {F(26, 69): 1.0}, pub(be, got)


def test_division_subtracts_fractional_dimensions(flavour):
    be = be_of(flavour)
    got = be.deconvolve(be.create(F(1, 3), 1.0), be.create(F(1, 5), 1.0))
    assert pub(be, got) == {F(2, 15): 1.0}, pub(be, got)


# --- the lattice is derived, not chosen ----------------------------------

def test_lattice_is_the_lcm_of_the_denominators(flavour):
    be = be_of(flavour)
    assert be.create_from_terms([F(1, 3), F(2, 3)], [5.0, 7.0]).L == 3
    assert be.create_from_terms([F(1, 6), F(1, 10)], [1.0, 1.0]).L == 30
    assert be.create_from_terms([F(1, 2), F(1, 3), F(1, 5)], [1.0] * 3).L == 30


def test_lattice_is_one_when_no_fractional_order_is_present(flavour):
    be = be_of(flavour)
    d = be.create_from_terms([0, 1, 2], [1.0, 2.0, 3.0])
    assert d.L == 1, f"integer dimensions must store on L=1, got {d.L}"
    assert be.convolve(d, d).L == 1
    # (1 + 2h + 3h^2)^2 = 1 + 4h + 10h^2 + 12h^3 + 9h^4
    assert be.read_dim(be.convolve(d, d), 2) == 10.0
    assert be.read_dim(be.convolve(d, d), 4) == 9.0


def test_result_is_independent_of_the_input_lattice(flavour):
    """The test for whether L is metadata on the number.  It must not be."""
    be = be_of(flavour)
    fine = be.create_from_terms([F(1, 3), F(2, 3)], [2.0, 3.0])
    from composite.backends.fractional_backend import FracData
    coarse = FracData({10: 2.0, 20: 3.0}, 30, canon=False)   # same orders, coarser
    y = be.create(F(1, 5), 7.0)
    assert pub(be, be.convolve(fine, y)) == pub(be, be.convolve(coarse, y))
    assert be.convolve(fine, y).L == be.convolve(coarse, y).L


def test_dim_cast_passes_a_fraction_through_but_reduces_an_integral_one():
    assert dim_cast(F(1, 3)) == F(1, 3)
    assert dim_cast(F(4, 2)) == 2 and isinstance(dim_cast(F(4, 2)), int)


def test_as_fraction_prefers_the_simplest_fraction_for_a_float():
    # 1/3 as a float IS an exact dyadic rational with a huge denominator; taking
    # that literally would put L in the quadrillions for a dimension meant as a
    # third.  The simplest fraction that maps back to the same float64 wins.
    assert as_fraction(1 / 3) == F(1, 3)
    assert as_fraction(0.5) == F(1, 2)
    assert as_fraction(3.0) == F(3) and as_fraction(-2) == F(-2)


# --- zero handling -------------------------------------------------------

def test_a_cancelled_fractional_dimension_is_retained(flavour):
    be = be_of(flavour)
    x = be.create(F(1, 3), 1.0)
    z = be.add(x, be.negate(x))
    assert be.read_dim(z, F(1, 3)) == 0.0
    assert be.is_wholly_zero(z), "a cancelled dimension must stay expressed (R2)"


def test_a_product_never_invents_a_dimension(flavour):
    """The Minkowski-sum contract.  A dense kernel that pads the key range and
    returns every position invents dimensions, and an invented zero here is an
    EXPRESSED zero, so an infinitesimal.  A fuzz caught this in 215 of 400
    random products before the presence filter went in."""
    be = be_of(flavour)
    a = be.create_from_terms([0, 1, 5], [1.0, 2.0, 3.0])       # gap at 2,3,4
    b = be.create_from_terms([0, 1], [1.0, 1.0])
    want = sorted({0, 1, 5, 1, 2, 6})                          # Minkowski sum
    assert sorted(pub(be, be.convolve(a, b))) == want, pub(be, be.convolve(a, b))


def test_wholly_zero_product_keeps_every_constructed_dimension(flavour):
    be = be_of(flavour)
    a = be.create_from_terms([F(1, 3), F(2, 3)], [0.0, 0.0])
    b = be.create_from_terms([F(1, 5)], [2.0])
    assert pub(be, be.convolve(a, b)) == {F(8, 15): 0.0, F(13, 15): 0.0}


# --- agreement with the reference backend and between flavours -----------

EXPR = [
    ("h",              lambda: cl.ZERO),
    ("R(3)+h",         lambda: R(3.0) + cl.ZERO),
    ("h*h*h",          lambda: cl.ZERO * cl.ZERO * cl.ZERO),
    ("h/h",            lambda: cl.ZERO / cl.ZERO),
    ("1-1",            lambda: R(1.0) - R(1.0)),
    ("(2+h)**5",       lambda: (R(2.0) + cl.ZERO) ** 5),
    ("1/(1+h)",        lambda: R(1.0) / (R(1.0) + cl.ZERO)),
    ("sqrt(4+h)",      lambda: sqrt(R(4.0) + cl.ZERO)),
    ("exp(h)",         lambda: exp(cl.ZERO)),
    ("sin(2+h)",       lambda: sin(R(2.0) + cl.ZERO)),
    ("cos(2+h)",       lambda: cos(R(2.0) + cl.ZERO)),
    ("ln(1+h)",        lambda: ln(R(1.0) + cl.ZERO)),
    ("x*x at 2",       lambda: cl._seeded(2.0) ** 2),
    ("sin(x)*x at 2",  lambda: sin(cl._seeded(2.0)) * cl._seeded(2.0)),
    ("(x*x-4)/(x-2)",  lambda: (cl._seeded(2.0) ** 2 - R(4.0)) / (cl._seeded(2.0) - R(2.0))),
    ("5+0",            lambda: R(5.0) + R(0.0)),
]


def _snapshot(fn):
    with cl._derivative_scope(6, 30):
        c = cl._ensure_composite(fn())
    return {(k if isinstance(k, tuple) else float(k)): round(v, 10)
            for k, v in c.coeffs_dict().items()}


@pytest.mark.parametrize("label,fn", EXPR, ids=[e[0] for e in EXPR])
def test_matches_sparse_dense_on_ordinary_work(label, fn):
    config.use_sparse_dense()
    want = _snapshot(fn)
    try:
        for name in FLAVOURS:
            getattr(config, "use_" + name)()
            got = _snapshot(fn)
            assert got == want, f"{name} on {label}: got {got}, want {want}"
    finally:
        config.use_sparse_dense()


def test_numpy_flavour_matches_dict_flavour_on_random_products():
    """Covers all three aggregation kernels; the denominator sets are chosen so
    that the span routes some products to each."""
    bd, bn = FractionalDictBackend(), FractionalNumpyBackend()
    rng = random.Random(4)
    sets = [[1], [2], [3], [2, 3], [3, 5], [2, 3, 5], [7], [3, 7, 11],
            [1009, 1013], [10007, 10009]]
    for _ in range(300):
        dens = rng.choice(sets)
        def mk(n):
            return ([F(rng.randint(-9, 9), rng.choice(dens)) for _ in range(n)],
                    [round(rng.uniform(-3, 3), 3) for _ in range(n)])
        da, va = mk(rng.randint(1, 8))
        db, vb = mk(rng.randint(1, 8))
        r1 = bd.convolve(bd.create_from_terms(da, va), bd.create_from_terms(db, vb))
        r2 = bn.convolve(bn.create_from_terms(da, va), bn.create_from_terms(db, vb))
        k1 = {k: round(v, 9) for k, v in pub(bd, r1).items()}
        k2 = {k: round(v, 9) for k, v in pub(bn, r2).items()}
        assert k1 == k2 and r1.L == r2.L, f"dims {da} x {db}: {k1} vs {k2}"


def test_all_three_kernels_are_reachable():
    """Guards the routing rule itself: if a threshold drifts so that one kernel
    is never selected, the fuzz above stops covering it silently."""
    bn = FractionalNumpyBackend()
    seen = set()
    # spans verified by measurement, not by arithmetic in the head: 3, 3000 and
    # 80055 against a bincount threshold of 65536.
    cases = [([0, 1, 2], [1]),                                  # span 3     -> convolve
             ([F(1, 3), 1000], [F(1, 3)]),                       # span 3000  -> bincount
             ([F(1, 10007), F(9, 10009)], [F(1, 10007)])]        # span 80055 -> unique
    for da, db in cases:
        a = bn.create_from_terms(da, [1.0] * len(da))
        b = bn.create_from_terms(db, [1.0] * len(db))
        L, ta, tb = bn._align(a, b)
        ka = np.fromiter(ta.keys(), np.int64, len(ta))
        kb = np.fromiter(tb.keys(), np.int64, len(tb))
        span = int(ka.max() + kb.max()) - int(ka.min() + kb.min()) + 1
        npair = ka.size * kb.size
        if span <= bn.DENSE_SPAN_FACTOR * (ka.size + kb.size):
            seen.add("convolve")
        elif span <= max(npair * 16, 1 << 16):
            seen.add("bincount")
        else:
            seen.add("unique")
    assert seen == {"convolve", "bincount", "unique"}, f"only reached {seen}"


def test_fractional_grade_survives_through_the_public_api(flavour):
    c = Composite({F(1, 3): 1.0})
    assert c.lead_order() == F(-1, 3), c.lead_order()
    assert (c * c * c).lead_order() == -1
    assert (c * Composite({F(1, 5): 1.0})).lead_order() == F(-8, 15)


# --- the torch flavour ---------------------------------------------------

torch = pytest.importorskip("torch", reason="torch flavour needs torch")
from composite.backends.fractional_backend import FractionalTorchBackend  # noqa: E402


def test_torch_flavour_matches_the_reference_on_random_products():
    """min_pairs=1 forces the torch path; the default threshold would delegate
    almost everything to numpy, which would test nothing."""
    bd = FractionalDictBackend()
    bt = FractionalTorchBackend(device="cpu", min_pairs=1)
    rng = random.Random(11)
    sets = [[1], [3], [2, 3], [2, 3, 5], [7], [10007, 10009]]
    for _ in range(150):
        dens = rng.choice(sets)
        def mk(n):
            return ([F(rng.randint(-9, 9), rng.choice(dens)) for _ in range(n)],
                    [round(rng.uniform(-3, 3), 3) for _ in range(n)])
        da, va = mk(rng.randint(1, 9))
        db, vb = mk(rng.randint(1, 9))
        r1 = bd.convolve(bd.create_from_terms(da, va), bd.create_from_terms(db, vb))
        r2 = bt.convolve(bt.create_from_terms(da, va), bt.create_from_terms(db, vb))
        k1 = {k: round(v, 6) for k, v in pub(bd, r1).items()}
        k2 = {k: round(v, 6) for k, v in pub(bt, r2).items()}
        assert k1 == k2 and r1.L == r2.L, f"{da} x {db}: {k1} vs {k2}"


def test_torch_delegates_a_dense_span_rather_than_building_a_pair_table():
    """Contiguous keys are a convolution.  Building an n*m table instead was
    40.7 ms against numpy's 5.0 ms at n=4000, and 256 MB of it."""
    bt = FractionalTorchBackend(device="cpu", min_pairs=1)
    bn = FractionalNumpyBackend()
    dims = list(range(300))
    vals = [1.0 / (k + 1) for k in range(300)]
    a_t = bt.create_from_terms(dims, vals)
    a_n = bn.create_from_terms(dims, vals)
    got = pub(bt, bt.convolve(a_t, a_t))
    want = pub(bn, bn.convolve(a_n, a_n))
    assert got.keys() == want.keys()
    assert all(abs(got[k] - want[k]) < 1e-12 for k in want)


def test_mps_needs_an_explicit_opt_in_because_it_has_no_float64():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("no MPS device")
    with pytest.raises(ValueError, match="float32"):
        FractionalTorchBackend(device="mps")
    be = FractionalTorchBackend(device="mps", allow_float32=True)
    assert be.vdtype == torch.float32
    # The point of the guard: coefficients drop, the LATTICE does not.  A grade
    # losing digits changes which terms merge, and that is never traded.
    a = be.create(F(1, 3), 1.0)
    assert a.L == 3 and pub(be, be.convolve(be.convolve(a, a), a)) == {1: 1.0}


# --- how the capability is reached ---------------------------------------

def test_a_fraction_exponent_is_accepted_and_stays_exact(flavour):
    """A float exponent survives only because as_fraction round-trips it, which
    is luck rather than a contract: Fraction(1, 10**7) as a float comes back as
    944473296573929/9444732965739290427392."""
    for q in (2, 3, 7, 23, 10 ** 3, 10 ** 7, 10 ** 12):
        got = list((cl.ZERO ** F(1, q)).coeffs_dict())[0]
        assert got == F(-1, q), f"ZERO ** 1/{q} gave {got!r}, want {F(-1, q)!r}"


def test_a_fraction_exponent_with_denominator_one_is_the_integer_case(flavour):
    assert list((cl.ZERO ** F(6, 3)).coeffs_dict()) == [-2]     # h**2
    assert list((cl.ZERO ** F(3, 1)).coeffs_dict()) == [-3]


def test_a_fraction_exponent_round_trips_through_its_own_power(flavour):
    for q in (2, 3, 5, 7, 10, 23):
        c = cl.ZERO ** F(1, q)
        acc = c
        for _ in range(q - 1):
            acc = acc * c
        assert list(acc.coeffs_dict()) == [-1], f"q={q} gave {acc}"


def test_an_exact_dimension_prints_exactly(flavour):
    # TWO NOTATIONS, NEVER MIXED.  |4|₀ puts the dimension in real subscript
    # glyphs, so the bars delimit the coefficient.  4_1/4 uses the underscore to
    # do the subscripting, so the bars are redundant and are dropped.  `|2|_5/6`
    # was both at once, which is neither.
    assert str(cl.ZERO ** F(1, 3)) == "1_-1/3"
    assert str(cl.ZERO ** F(2, 3)) == "1_-2/3"
    assert str(Composite({F(1, 3): 2.5})) == "2.5_1/3"
    # an integral dimension keeps its subscript glyphs, and the bars with them
    assert str(cl.ZERO) == "|1|₋₁"
    assert str(R(3.0) + cl.ZERO) == "|3|₀ + |1|₋₁"
    # and the choice is made for the WHOLE number, never per term
    assert str(R(3.0) + (cl.ZERO ** F(1, 3))) == "3_0 + 1_-1/3"
    for c in (cl.ZERO ** F(1, 3), R(3.0) + (cl.ZERO ** F(1, 3)),
              Composite({F(5, 6): 2.0, 0: 4.0, -3: 0.5})):
        s = str(c)
        assert not ("|" in s and "_" in s), f"notations mixed: {s}"


def test_the_constructor_keeps_a_fraction_only_when_the_backend_can_hold_it(flavour):
    """Handing a Fraction to a float64 backend would give it Fraction and float
    keys for the same grade, which never merge, so EXACT_DIMS gates it."""
    exact = list(Composite({F(1, 10 ** 9): 2.0}).coeffs_dict())[0]
    assert exact == F(1, 10 ** 9), exact
    config.use_sparse_dense()
    floated = list(Composite({F(1, 3): 1.0}).coeffs_dict())[0]
    assert isinstance(floated, float) and floated == 1 / 3


@pytest.mark.parametrize("backend", ["sparse_dense", "dict"])
def test_the_refusal_says_where_to_go(backend):
    """A capability nobody can find from the error that blocks them is not a
    capability."""
    getattr(config, "use_" + backend)()
    try:
        third = Composite({-1 / 3: 1.0})
        with pytest.raises(InexactGradeError) as exc:
            third * third * third
        msg = str(exc.value)
        assert "use_fractional_numpy" in msg, msg
        assert "rational lattice" in msg, msg
    finally:
        config.use_sparse_dense()


# --- zero in a dimension -------------------------------------------------

def test_a_zero_numerator_is_just_dimension_zero(flavour):
    assert list(Composite({F(0, 5): 7.0}).coeffs_dict()) == [0]
    assert list(Composite({F(0, 1): 7.0}).coeffs_dict()) == [0]


def test_a_zero_denominator_is_refused_by_fraction_itself(flavour):
    """Nothing in the backend has to guard this: Fraction(5, 0) raises before
    any dimension exists."""
    with pytest.raises(ZeroDivisionError):
        Composite({F(5, 0): 7.0})
    with pytest.raises(ZeroDivisionError):
        Composite({F(0, 0): 7.0})
    with pytest.raises(ZeroDivisionError):
        cl.ZERO ** F(1, 0)


def test_a_zeroth_power_is_the_identity_not_a_root(flavour):
    assert list((cl.ZERO ** F(0, 5)).coeffs_dict()) == [0]
    assert (cl.ZERO ** F(0, 5)).st() == 1.0
    assert list((cl.ZERO ** 0).coeffs_dict()) == [0]


def test_division_of_fractional_zeros_follows_the_zero_rules(flavour):
    # ZERO ** F(1,3) has dimension -1/3: an infinitesimal of order one third.
    t = cl.ZERO ** F(1, 3)
    zf = Composite({F(1, 3): 0.0})        # a zero at the INFINITE dim +1/3
    assert cl._is_wholly_zero(zf) and not cl._is_wholly_zero(t)
    assert list(cl._r1(zf).coeffs_dict()) == [F(-2, 3)]   # R1 shifts one order
    assert list((t / t).coeffs_dict()) == [0]                       # 0/0 = 1
    assert (t / t).st() == 1.0
    assert list((t / zf).coeffs_dict()) == [F(1, 3)]      # -1/3 - (-2/3)
    assert list((zf / t).coeffs_dict()) == [F(-1, 3)]     # -2/3 - (-1/3)
    assert list((zf / zf).coeffs_dict()) == [0]


def test_a_lattice_of_zero_is_refused_at_construction():
    """It was accepted and then died in __repr__ with a bare
    ZeroDivisionError: Fraction(0, 0), a page away from the cause."""
    from composite.backends.fractional_backend import FracData
    for bad in (0, -3):
        with pytest.raises(ValueError, match="positive integer"):
            FracData({0: 1.0}, bad)
    assert FracData({0: 1.0}, 1).L == 1


def test_the_lattice_can_never_reach_zero_through_the_api(flavour):
    be = be_of(flavour)
    from composite.backends.fractional_backend import _canon
    assert _canon({0: 5.0}, 7) == ({0: 5.0}, 1)     # a lone dim-0 term reduces
    assert _canon({}, 9) == ({}, 1)                 # and so does nothing
    for dims in ([F(0, 5)], [0], [F(1, 3), F(0, 7)], []):
        d = be.create_from_terms(dims, [1.0] * len(dims))
        assert d.L >= 1, f"{dims} gave L={d.L}"


def test_division_keeps_a_fractional_dimension_exact(flavour):
    """Mixing a Fraction dimension with a float one subtracts in FLOAT and lands
    an ulp out: Fraction(-1,3) - (-1.0) is 0.6666666666666667 where
    float(Fraction(2,3)) is 0.6666666666666666.  The grade was then wrong rather
    than imprecise, and h^(1/3) / R(0) came back at dimension
    3002399751580331/4503599627370496."""
    t = cl.ZERO ** F(1, 3)
    cases = [(t / R(0.0), F(2, 3)),          # R(0) is |1|_-1, so -1/3 - (-1) = 2/3
             (t / cl.ZERO, F(2, 3)),
             (t / R(2.0), F(-1, 3)),         # a scalar divisor shifts nothing
             (R(2.0) / t, F(1, 3)),
             ((cl.ZERO ** F(1, 5)) / t, F(2, 15)),
             (t / t, 0)]
    for got, want in cases:
        dim = list(got.coeffs_dict())[0]
        assert dim == want, f"got {dim!r}, want {want!r}"
        if want != 0:
            assert isinstance(dim, F), f"{dim!r} lost its exactness"


def test_dim_fraction_is_one_implementation():
    from composite.backends.base_backend import dim_fraction
    from composite.backends.fractional_backend import as_fraction
    for d in (F(1, 3), 2, -5, 0.5, 1 / 3, 3.0, np.float64(0.25)):
        assert as_fraction(d) == dim_fraction(d)


def test_r1_on_a_fractional_grade_shifts_one_whole_order_not_one_lattice_step(flavour):
    """R1 says |0|_d becomes |1|_(d-1).  On an integer lattice d-1 IS the
    dimension below; once dimensions are rational it is not, and subtracting one
    LATTICE step instead would make the answer depend on L -- 1/3 - 1/3 = 0 on
    L=3, 1/3 - 1/30 = 3/10 on L=30 -- which the canonical-lattice invariant
    forbids.  One whole order is the only storage-independent reading."""
    from composite.backends.fractional_backend import FracData
    for L, key in ((3, 1), (30, 10), (300, 100)):          # all the same number
        c = Composite(_data=FracData({key: 0.0}, L, canon=False))
        assert cl._is_wholly_zero(c)
        assert list(cl._r1(c).coeffs_dict()) == [F(-2, 3)], \
            f"stored on L={L}: got {list(cl._r1(c).coeffs_dict())}"


ZERO_RULES = [
    ("1 - 1",       lambda: R(1.0) - R(1.0)),
    ("0 / 0",       lambda: R(0.0) / R(0.0)),
    ("0 * 5",       lambda: R(0.0) * R(5.0)),
    ("7*0*0/0/0",   lambda: R(7.0) * R(0.0) * R(0.0) / R(0.0) / R(0.0)),
    ("5 + 0",       lambda: R(5.0) + R(0.0)),
    ("0 - 0",       lambda: R(0.0) - R(0.0)),
    ("5 + nothing", lambda: R(5.0) + Composite({})),
    ("1 / 0",       lambda: R(1.0) / R(0.0)),
]


@pytest.mark.parametrize("label,fn", ZERO_RULES, ids=[c[0] for c in ZERO_RULES])
def test_the_zero_rules_are_unchanged_by_the_lattice(label, fn):
    config.use_sparse_dense()
    want = str(cl._ensure_composite(fn()))
    try:
        for name in FLAVOURS:
            getattr(config, "use_" + name)()
            assert str(cl._ensure_composite(fn())) == want, \
                f"{name} on {label}: got {cl._ensure_composite(fn())}, want {want}"
    finally:
        config.use_sparse_dense()


# --- rational powers of a multi-term composite ---------------------------

def _leading(v, keep):
    """The `keep` most DOMINANT terms.  _lead_order is POSITIVE for an
    infinitesimal, so the smallest order is the most dominant and the sort is
    ascending.  Sorting the other way picks the truncation tail, which is how I
    misread this three times."""
    v = cl._ensure_composite(v)
    d = v.coeffs_dict()
    ks = sorted(d, key=lambda x: cl._lead_order(x))[:keep]
    return {k: round(d[k], 9) for k in ks}


ROOT_CASES = [
    ("h", lambda h: h, 3), ("h", lambda h: h, 5),
    ("h^3 + h^4", lambda h: (h ** 3) + (h ** 4), 2),
    ("h^3 + h^4", lambda h: (h ** 3) + (h ** 4), 5),
    ("h^3 + h^7", lambda h: (h ** 3) + (h ** 7), 5),
    ("1 + h", lambda h: R(1.0) + h, 3),
    ("1 + h", lambda h: R(1.0) + h, 7),
    ("8*h^3", lambda h: R(8.0) * (h ** 3), 3),
    ("1+2h+3h^2", lambda h: R(1.0) + R(2.0) * h + R(3.0) * (h ** 2), 3),
]


@pytest.mark.parametrize("label,build,q", ROOT_CASES,
                         ids=[f"{c[0]}^(1/{c[2]})" for c in ROOT_CASES])
def test_a_rational_root_round_trips(flavour, label, build, q):
    base = build(cl.ZERO)
    root = base ** F(1, q)
    back = root
    for _ in range(q - 1):
        back = back * root
    want, got = _leading(base, 2), _leading(back, 2)
    assert set(want) == set(got), f"{label}: dims {sorted(got)} vs {sorted(want)}"
    assert all(abs(got[k] - want[k]) < 1e-9 for k in want), f"{label}: {got} vs {want}"


def test_a_non_dyadic_root_of_a_multi_term_composite_is_exact(flavour):
    """The old path fell through to exp(float(r) * ln(x)), and float(1/5) threw
    the grade away: (h**3 + h**4) ** Fraction(1,5) came back at
    1351079888211149/2251799813685248 instead of 3/5.  Dyadic exponents survived
    by luck, which is why 1/2 always looked fine."""
    h = cl.ZERO
    assert (((h ** 3) + (h ** 4)) ** F(1, 5)).lead_order() == F(3, 5)
    assert (((h ** 3) + (h ** 7)) ** F(1, 5)).lead_order() == F(3, 5)
    assert (((h ** 2)) ** F(1, 3)).lead_order() == F(2, 3)
    assert (((h ** 3) + (h ** 4)) ** F(1, 2)).lead_order() == F(3, 2)


def test_an_inert_zero_does_not_change_a_rational_power(flavour):
    """All three are the number h; they differ only in which zero is expressed.
    The old path raised for one of them, because ln needs a positive standard
    part and an expressed zero below the leading term is enough to reach it."""
    want = F(1, 3)
    for terms in ({-1: 1.0},
                  {0: 0.0, -1: 1.0},
                  {-1: 1.0, -2: 0.0},
                  {0: 0.0, -1: 1.0, -2: 0.0}):
        assert (Composite(terms) ** F(1, 3)).lead_order() == want, terms


def test_rational_power_coefficients_match_the_binomial_series(flavour):
    got = (R(1.0) + cl.ZERO) ** F(1, 3)
    d = got.coeffs_dict()
    coef = 1.0
    for k in range(5):
        assert abs(d.get(-k, 0.0) - coef) < 1e-12, f"h^{k}: {d.get(-k)} vs {coef}"
        coef *= (1 / 3 - k) / (k + 1)
    # and the same series carried on a fractional leading grade
    d = (((cl.ZERO ** 3) + (cl.ZERO ** 4)) ** F(1, 2)).coeffs_dict()
    coef = 1.0
    for k in range(5):
        assert abs(d.get(F(-3, 2) - k, 0.0) - coef) < 1e-12
        coef *= (0.5 - k) / (k + 1)


def test_rational_power_edge_cases(flavour):
    h = cl.ZERO
    # a negative coefficient has a real q-th root only for odd q
    assert str((R(-8.0) * (h ** 3)) ** F(1, 3)) == "|-2|₋₁"
    with pytest.raises(ValueError, match="not real"):
        (-h) ** F(1, 2)
    # nothing, and a wholly zero value, have no leading term to factor out
    for c in (Composite({}), Composite({0: 0.0}), Composite({F(1, 3): 0.0})):
        with pytest.raises(ZeroDivisionError, match="leading term"):
            c ** F(1, 3)


def test_truncation_handles_a_fractional_dimension(flavour):
    """_truncate_dims ranked object-array dimensions with
    `tuple(abs(c) for c in d)`, which raises "'Fraction' object is not iterable".

    THE CAP IS SET HERE, not assumed.  Nine test modules assign
    cl.MAX_ACTIVE_DIMS = 10**9 at module import and never restore it, so once any
    of them has loaded the truncation path never runs and a test that relies on
    the shipped cap silently stops testing anything.  _truncate_dims' own
    docstring records the same hazard: its earlier TypeError on log-axis
    dimensions "went unseen while every standalone check ran with
    MAX_ACTIVE_DIMS raised".  This Fraction crash was the second instance.
    """
    h = cl.ZERO
    saved = cl.MAX_ACTIVE_DIMS
    cl.MAX_ACTIVE_DIMS = 60
    try:
        root = ((h ** 3) + (h ** 4)) ** F(1, 2)
        assert root.lead_order() == F(3, 2)
        sq = root * root                  # past the cap, so truncation runs
        assert sq.lead_order() == 3
        assert sq.coeffs_dict(), "truncation dropped everything"
        # and the cap really did bite, which is what makes this a regression test
        assert len(sq.coeffs_dict()) <= 60
    finally:
        cl.MAX_ACTIVE_DIMS = saved
