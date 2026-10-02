"""TAG() and single_infinitesimal(): one blessed infinitesimal, conventional rest.

PROTOTYPE under test. The specification:

  `TAG(v)` blesses ONE value as the infinitesimal source. Inside
  `single_infinitesimal()` every zero that does NOT descend from the tag behaves
  conventionally -- `0`, `R(0)`, `ZERO` and a cancellation all absorb,
  multiplication by zero annihilates, and division by a zero raises.

  The payback is that no second source can enter, so every order of the jet is
  the CLASSICAL derivative rather than a denoted reading.

The blessing must be DECLARED, not inferred. Three ways of inferring it were
tried and each failed for its own reason, recorded at `_TAGGED`: counting mints
never fires because `R(3) + ZERO` mints nothing; a per-number source cannot tell
seed one from seed two because ZERO is minted once at import and every use shares
the id; and identity with ZERO fails because `from composite import ZERO` binds
at import while `_refresh_constants` rebinds the module global.
"""
import math

import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO, exp, ln, sin, sqrt
from composite.backends import config

TAG = cl.TAG


@pytest.fixture(autouse=True)
def dict_backend():
    config.use_dict()
    cl._refresh_constants()
    cl.MAX_ACTIVE_DIMS = 10 ** 9
    yield
    cl.UNTAG()                 # TAG arms the mode globally, so tests must disarm
    config.use_sparse_dense()
    cl._refresh_constants()


def grades(value):
    return dict(cl._ensure_composite(value).coeffs_dict())


# --- the tagged lineage is untouched -----------------------------------------

TAGGED_OPERATIONS = [
    ("x",          lambda x: x),
    ("x*x",        lambda x: x * x),
    ("x**5",       lambda x: x ** 5),
    ("R(1)/x",     lambda x: R(1) / x),
    ("sin(x)",     lambda x: sin(x)),
    ("exp(x)",     lambda x: exp(x)),
    ("sqrt(x)",    lambda x: sqrt(x)),
    ("ln(x)",      lambda x: ln(x)),
    ("x - x",      lambda x: x - x),
]


@pytest.mark.parametrize("label,build", TAGGED_OPERATIONS, ids=[n for n, _ in TAGGED_OPERATIONS])
def test_the_tagged_lineage_matches_default_mode(label, build):
    """The invariant the mode must not break: anything descending from the tag
    behaves exactly as it does by default. `_join` carries `_src` through every
    operation, so every term of a series expansion stays blessed."""
    default = grades(build(R(3) + ZERO))
    with cl.single_infinitesimal():
        inside = grades(build(TAG(3)))
    assert inside == default, "%s: default %r, tagged %r" % (label, default, inside)


def test_the_classical_jet_survives_to_every_order():
    with cl.single_infinitesimal():
        x = TAG(3)
        cubic = x * x * x + R(0) * x + (R(2) - R(2))
        got = [cubic.d(k) for k in range(4)]
    assert got == [27.0, 27.0, 18.0, 6.0], \
        "got %r, want the classical jet of x**3 at 3" % got


def test_a_transcendental_derivative_is_exact():
    with cl.single_infinitesimal():
        got = sin(TAG(3)).d(1)
    assert abs(got - math.cos(3.0)) < 1e-15, "got %r, want cos(3) = %r" % (got, math.cos(3.0))


# --- every untagged zero is conventional --------------------------------------

@pytest.mark.parametrize("label,build,want", [
    ("x*x + R(0)",        lambda x: x * x + R(0),            6.0),
    ("x*x + 0",           lambda x: x * x + 0,               6.0),
    ("x*x + ZERO",        lambda x: x * x + ZERO,            6.0),
    ("x*x + (R(3)-R(3))", lambda x: x * x + (R(3) - R(3)),   6.0),
    ("x*x*R(0) + x*x",    lambda x: x * x * R(0) + x * x,    6.0),
], ids=["written zero", "bare int", "bare ZERO", "cancellation", "annihilation"])
def test_an_untagged_zero_leaves_the_derivative_classical(label, build, want):
    with cl.single_infinitesimal():
        got = cl._ensure_composite(build(TAG(3))).d(1)
    assert got == want, "%s: got %r, want %r" % (label, got, want)


def test_an_untagged_seed_is_absorbed():
    # Not raised, not silently infinitesimal: absorbed, because ZERO is a zero.
    with cl.single_infinitesimal():
        x = TAG(3)
        y = R(2) + ZERO
        assert grades(y) == {0: 2.0}, "untagged seed gave %r, want {0: 2.0}" % grades(y)
        assert (x * y).d(1) == 2.0, "d(1) of x*y = %r, want 2.0" % (x * y).d(1)


def test_annihilation_and_division_errors_return():
    with cl.single_infinitesimal():
        x = TAG(3)
        annihilated = x * x * R(0)
        assert all(v == 0.0 for v in grades(annihilated).values()), \
            "multiplication by zero left %r" % grades(annihilated)
        with pytest.raises(ZeroDivisionError, match="single_infinitesimal"):
            x / R(0)


def test_a_second_tag_replaces_the_blessing():
    """Re-running a cell that tags must not be an error, and only one value is
    ever blessed -- the newest."""
    with cl.single_infinitesimal():
        first = TAG(3)
        second = TAG(5)
        assert grades(second) == {0: 5.0, -1: 1.0}, grades(second)
        assert (second * second + R(0)).d(1) == 10.0, "the new tag is not live"
        assert cl._sealed(first) is True, "the old tag should no longer be blessed"


# --- the two bugs the suite-wide run exposed ----------------------------------

@pytest.mark.parametrize("label,value,want", [
    ("TAG(3)",          3,           {0: 3.0, -1: 1.0}),
    ("TAG(0)",          0,           {0: 0.0, -1: 1.0}),
    ("TAG(R(3))",       None,        {0: 3.0, -1: 1.0}),
    # R(0) is the LATENT |0|_0, so this now matches TAG(0) exactly -- the two
    # spellings of a written zero finally agree.
    ("TAG(R(0))",       None,        {0: 0.0, -1: 1.0}),
    ("TAG(ZERO)",       None,        {-1: 1.0}),
    ("TAG(R(2)+ZERO)",  None,        {0: 2.0, -1: 1.0}),
], ids=["int", "int zero", "R(3)", "R(0)", "ZERO", "already seeded"])
def test_tag_never_doubles_the_infinitesimal(label, value, want):
    """Three ways in: R(0) has already converted to |1|_-1 before TAG sees it,
    ZERO is one by construction, and an already-seeded value carries one. Adding
    another gave |2|_-1, and the jet of ln(1+x)/x at 0 then read [1, -1, 8/3, -12]
    -- right for a seed of 2h, and not what was asked for."""
    built = {"TAG(R(3))": lambda: R(3), "TAG(R(0))": lambda: R(0),
             "TAG(ZERO)": lambda: ZERO, "TAG(R(2)+ZERO)": lambda: R(2) + ZERO}
    argument = built[label]() if value is None else value
    assert grades(TAG(argument)) == want, "%s gave %r" % (label, grades(TAG(argument)))


def test_tag_is_idempotent():
    x = TAG(3)
    assert grades(TAG(x)) == grades(x), "TAG(TAG(3)) doubled something"


def test_the_jet_at_zero_is_right():
    # The bug this guards was only visible at the origin, where R(0) converts.
    x = TAG(0)
    y = ln(R(1) + x) / x
    got = [y.d(k) for k in range(4)]
    want = [1.0, -0.5, 2.0 / 3, -1.5]       # ln(1+x)/x = 1 - x/2 + x^2/3 - x^3/4
    assert all(abs(g - w) < 1e-12 for g, w in zip(got, want)), \
        "got %r, want %r" % (got, want)


def test_asking_whether_it_is_sealed_does_not_seal_it():
    """The predicate must be pure. It was consulted from three places and the
    first consultation took the slot, so the second saw a sealed state and
    demoted ZERO on its own first use."""
    with cl.single_infinitesimal():
        x = TAG(3)
        assert cl._sealed(ZERO) == cl._sealed(ZERO), "asking twice changed the answer"
        assert cl._sealed(x) is False, "the tagged value must never read as sealed"


def test_each_operand_is_judged_on_its_own_blessing():
    """Asking whether the PAIR held a blessed value let `x*x + ZERO` through:
    x*x is blessed, so the pair was, and the bare ZERO beside it survived."""
    with cl.single_infinitesimal():
        x = TAG(3)
        assert grades(x * x + ZERO) == {0: 9.0, -1: 6.0, -2: 1.0}, grades(x * x + ZERO)


@pytest.mark.parametrize("backend", ["dict", "sparse_dense", "dense_series"])
def test_a_demoted_operand_keeps_its_own_backend(backend):
    """Composite({...}) binds the globally ACTIVE backend, so demoting an operand
    living on another one raised "'DictData' object has no attribute 'runs'"."""
    getattr(config, "use_" + backend)()
    cl._refresh_constants()
    with cl.single_infinitesimal():
        got = grades(TAG(3) ** 2 + R(0))
    assert got == {0: 9.0, -1: 6.0, -2: 1.0}, "backend %s gave %r" % (backend, got)


# --- the mode leaves nothing behind -------------------------------------------

def test_default_behaviour_outside_the_block_is_untouched():
    x = R(3) + ZERO
    assert (x * x + R(0)).d(1) == 7.0, "the denoted reading changed outside the mode"
    assert grades(R(6) - R(6)) == {-1: 6.0}
    assert grades(R(1) / R(0)) == {1: 1.0}


def test_tag_arms_the_mode_by_itself():
    """TAG is the switch. There is no block to remember."""
    x = TAG(3)
    assert grades(x) == {0: 3.0, -1: 1.0}, grades(x)
    assert (x * x + R(0)).d(1) == 6.0, "TAG did not arm the mode"
    assert (x * x + (R(2) - R(2))).d(1) == 6.0


def test_untag_restores_the_default():
    x = TAG(3)
    assert (x * x + R(0)).d(1) == 6.0
    assert cl.UNTAG() is True, "UNTAG reported the mode was not on"
    y = R(3) + ZERO
    assert (y * y + R(0)).d(1) == 7.0, "the denoted reading did not come back"
    assert grades(R(6) - R(6)) == {-1: 6.0}
    assert cl.UNTAG() is False, "UNTAG should be safe when the mode is off"


def test_degeneracy_tracking_is_restored_on_exit():
    before = cl._TRACKING[0]
    with cl.single_infinitesimal():
        assert cl._TRACKING[0] is True, "the mode needs provenance, so tracking must be on"
    assert cl._TRACKING[0] == before, "tracking left at %r, was %r" % (cl._TRACKING[0], before)


def test_the_mode_nests_and_unwinds():
    with cl.single_infinitesimal():
        x = TAG(3)
        with cl.single_infinitesimal():
            inner = TAG(5)
            assert grades(inner) == {0: 5.0, -1: 1.0}, \
                "the inner block gets its own tag: %r" % grades(inner)
        assert (x * x + R(0)).d(1) == 6.0, "the outer block lost its tag"
