"""Three things the composite arithmetic battery (tests/brute_composite.py) found.

1. The dimension cap (MAX_ACTIVE_DIMS) dropped grades without tightening the
   completeness bound: (1 + h)/(1 + h^4) kept terms to order 117 and reported
   complete_order 196, reading 0 at orders 120-196.
2. A long division whose divisor has a lower coefficient larger than its
   leading one can amplify its rounding without bound while its completeness
   bound still covers the result.  Now announced with PrecisionLossWarning;
   values are unchanged.
3. Adding or subtracting NOTHING is a no-op, checked before R1 (decided
   2026-10-12): <0_0> + NOTHING stays <0_0>, it does not convert.
"""
import warnings

import pytest

from composite.composite_lib import Composite, PrecisionLossWarning, R, ZERO


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


@pytest.fixture
def cap60():
    # Several suites set MAX_ACTIVE_DIMS = 10**9 at import and never restore it,
    # and pytest imports them before running anything: pin the default here.
    import composite.composite_lib as cl
    saved = cl.MAX_ACTIVE_DIMS
    cl.MAX_ACTIVE_DIMS = 60
    yield
    cl.MAX_ACTIVE_DIMS = saved


def test_cap_tightens_completeness(cap60):
    q = _quiet(lambda: Composite({0: 1.0}) / Composite({0: 1.0, -4: 1.0}))
    p = _quiet(lambda: q * Composite({0: 1.0, -1: 1.0}))
    d = p.coeffs_dict()
    deepest = -min(d)
    print(f"\n  (1 + h)/(1 + h^4): {len(d)} terms, deepest order {deepest:.0f}, complete_order {p.complete_order}")
    assert len(d) == 60, "the cap must have fired for this test to mean anything"
    assert p.complete_order <= deepest
    # every order the bound claims is right: 1/(1 + h^4) = sum (-1)^k h^(4k)
    for n in range(int(p.complete_order) + 1):
        k, r = divmod(n, 4)
        want = (-1) ** k if r in (0, 1) else 0
        assert p.coeff(-n) == want, (n, p.coeff(-n), want)


def test_unstable_division_is_announced():
    # (1 - 10h)/((1 - 10h)(1 - 0.1h)) = 1/(1 - 0.1h), a decaying series; the long
    # division's recurrence also has a growing mode near 10^n that rounding excites
    a = Composite({0: 1.0, -1: -10.0})
    b = Composite({0: 1.0, -1: -10.1, -2: 1.0})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q = a / b
    msgs = [str(x.message) for x in w if issubclass(x.category, PrecisionLossWarning)]
    print(f"\n  order 20: got {q.coeff(-20):.3e}, true {0.1 ** 20:.3e};  warning: {msgs[0][:100] if msgs else None}")
    assert msgs


def test_stable_division_is_not_announced():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        R(1) / (R(1) + (R(0.5) + ZERO) * (R(0.5) + ZERO))
        Composite({0: 1.0}) / Composite({0: 1.0, -1: -0.5})
    n = sum(issubclass(x.category, PrecisionLossWarning) for x in w)
    print(f"\n  stable divisions: {n} PrecisionLossWarning")
    assert n == 0


@pytest.mark.parametrize("label, build, want", [
    ("<0_0> + NOTHING", lambda: Composite({0: 0.0}) + Composite({}), {0: 0.0}),
    ("NOTHING + <0_0>", lambda: Composite({}) + Composite({0: 0.0}), {0: 0.0}),
    ("<0_0> - NOTHING", lambda: Composite({0: 0.0}) - Composite({}), {0: 0.0}),
    ("<0_-2 0_-3> + NOTHING", lambda: Composite({-2: 0.0, -3: 0.0}) + Composite({}), {-2: 0.0, -3: 0.0}),
    ("<5_0> + NOTHING", lambda: R(5) + Composite({}), {0: 5.0}),
    ("<0_0> + <0_0> (R1 still applies)", lambda: Composite({0: 0.0}) + Composite({0: 0.0}), {-1: 2.0}),
])
def test_nothing_is_a_no_op(label, build, want):
    got = _quiet(build).coeffs_dict()
    print(f"\n  {label} = {got}  want {want}")
    assert got == want


def test_nothing_no_op_keeps_a_truncation_bound():
    # Past set_max_order(6), h**7 is empty but complete to 6: what truncation left
    # of a term.  Adding it changes no value and must still carry the bound.
    import composite.composite_lib as cl
    cl.set_max_order(6)
    try:
        h = cl.ZERO
        empty = _quiet(lambda: h ** 7)
        s = _quiet(lambda: (R(1) + h) + empty)
        v = _quiet(lambda: cl.sin(R(0.5) + h))
        deep = max(-g for g in v.coeffs_dict())
        print(f"\n  h**7 capped: {empty.coeffs_dict()} complete {empty.complete_order};  "
              f"(1 + h) + it: {s.coeffs_dict()} complete {s.complete_order};  sin complete {v.complete_order} deepest {deep}")
        assert empty.coeffs_dict() == {} and empty.complete_order == 6
        assert s.coeffs_dict() == {0: 1.0, -1: 1.0} and s.complete_order == 6
        assert v.complete_order <= deep
    finally:
        cl.set_max_order(None)


def test_cap_never_drops_an_infinite_grade(cap60):
    # A product with many positive grades on a thirds lattice passes the cap.
    # Ranking by |grade| dropped its highest infinities (grade 14 here); the
    # cap now trims only infinitesimal grades, deepest first.
    from fractions import Fraction as Fr
    import composite.composite_lib as cl
    from composite.backends import config
    before = config.get_backend()
    config.use_fractional_numpy(); cl._refresh_constants()
    try:
        a = Composite({Fr(k, 3): 1.0 for k in range(-12, 25)})
        b = Composite({Fr(k, 2): 1.0 for k in range(-6, 13)})
        p = _quiet(lambda: a * b)
        d = p.coeffs_dict()
        want = {x + y for x in a.coeffs_dict() for y in b.coeffs_dict() if x + y >= 0}
        print(f"\n  kept {sum(1 for g in d if g >= 0)} of {len(want)} non-negative grades, "
              f"highest {max(d)}, complete_order {p.complete_order}")
        assert want <= set(d) and max(d) == Fr(14)
        assert p.complete_order == 0
    finally:
        config.set_backend(before); cl._refresh_constants()


def test_cancellation_warnings_are_opt_in():
    from composite.composite_lib import cancellation_warnings

    def caught(f):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f()
        return [str(x.message) for x in w if issubclass(x.category, PrecisionLossWarning)]

    near = lambda: (R(0.1) + R(0.2)) - R(0.3)
    exact = lambda: (R(0.05) + R(0.3)) - R(0.35)
    x = R(3) + ZERO
    off = caught(near) + caught(exact)
    with cancellation_warnings():
        on_near, on_exact = caught(near), caught(exact)
        same, clean = caught(lambda: x - x), caught(lambda: R(2.5) - R(1.25))
    print(f"\n  off: {len(off)}  near: {on_near[0][:60] if on_near else None}  "
          f"exact: {on_exact[0][:60] if on_exact else None}  x - x: {len(same)}  2.5 - 1.25: {len(clean)}")
    assert not off
    assert on_near and "lost precision" in on_near[0]
    assert on_exact and "exact cancellation" in on_exact[0]
    assert not same and not clean
