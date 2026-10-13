"""Python-number operands follow the composite rules, completeness is honoured,
and abs() and the reflected operators stay composite.

Three groups, one per fix.  The third is at the end of the file:

ABS AND REFLECTED OPERATORS.  abs() returned abs(st()), a float, so |x| entered
an expression as a constant: derivative(x*abs(x), 0.5) read 0.5 where it is
1.0.  __rsub__ and __rtruediv__ ran a COMPOSITE left operand through float(),
which Python reaches whenever the right operand is a subclass (forensics'
_Audited): B - A with B = 3+5h, A = 2+h read d(1) = -1 instead of 4, and
audit(-x - x, 3) reported a derivative of -1.

SCALAR OPERANDS.  `x * 2`, `x / 2`, `-x` and `2 + x` took shortcuts that skipped
the preparation every composite operand gets (TAG demotion, then R1), and
`__radd__` swapped its operands.  So a written zero stayed latent where the
composite spelling converted it:

    R(0) * 2     <0_0>   (read later as h)     R(0) * R(2)    2h
    x*x + R(0)*2 d(1) = 7 at 3                 x*x + R(2)*0   d(1) = 8
    x*x + (-R(0))  d(1) = 7                    x*x - R(0)     d(1) = 5
    2 + R(-2)    -2h                           R(2) + R(-2)   2h

Every check here compares the Python-number spelling with the composite one.

COMPLETENESS.  The library records how far each value is vouched for, and three
places did not act on it:

    sin(0.5 + h).d(12)        0.0, silently; the derivative is 0.479
    atan(1/h)                 stopped at h**23 and claimed exact
    INF**51 / (INF + 1)       cut, read 0.0 at grade 0 where the value is 1,
                              and claimed exact

d(n) past the bound now warns (IncompleteOrderWarning) and returns the value;
atan at an infinite argument and a cut quotient record their bound.

Every check prints got against want, so `pytest -rP` shows the numbers on a pass.
"""
import math
import warnings

import mpmath
import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, IncompleteOrderWarning
from composite.backends import config
from composite import forensics


BACKENDS = {"sparse_dense": config.use_sparse_dense, "dict": config.use_dict}


@pytest.fixture(params=sorted(BACKENDS), autouse=True)
def backend(request):
    """Run every check on the default backend and on dict, then restore."""
    BACKENDS[request.param]()
    cl.set_max_order(None)
    yield request.param
    cl.UNTAG()
    config.use_sparse_dense()


def show(label, got, want, tol=0.0):
    """Print got against want, then assert.  tol is absolute, fixed per check."""
    print(f"{label:46s} got {got!r:>28}  want {want!r}")
    if isinstance(want, float) and isinstance(got, float):
        assert abs(got - want) <= tol, f"{label}: got {got!r}, want {want!r}"
    else:
        assert got == want, f"{label}: got {got!r}, want {want!r}"


def d1(c):
    """d(1), quiet about the denotation warning these values carry by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return c.d(1)


def quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def seeded(a):
    return R(a) + cl.ZERO


# --- scalar operands: the Python-number spelling equals the composite one ----

@pytest.mark.parametrize("k", [2, -1, 0.5, 3.0])
def test_written_zero_times_a_number_converts(k):
    got = quiet(lambda: R(0) * k)
    want = quiet(lambda: R(0) * R(k))
    show(f"R(0) * {k}", got.coeffs_dict(), want.coeffs_dict())
    show(f"  as a value: {{-1: {float(k)}}}", got.coeffs_dict(), {-1: float(k)})


@pytest.mark.parametrize("k", [2, -4, 0.25])
def test_written_zero_divided_by_a_number_converts(k):
    got = quiet(lambda: R(0) / k)
    want = quiet(lambda: R(0) / R(k))
    show(f"R(0) / {k}", got.coeffs_dict(), want.coeffs_dict())


def test_negating_a_written_zero_converts():
    got = quiet(lambda: -R(0))
    want = quiet(lambda: R(0) * R(-1))
    show("-R(0)", got.coeffs_dict(), want.coeffs_dict())
    show("  as a value", got.coeffs_dict(), {-1: -1.0})


@pytest.mark.parametrize("spelled, composite_spelling, want", [
    (lambda x: x * x + R(0) * 2,  lambda x: x * x + R(0) * R(2),  8.0),
    (lambda x: x * x + 2 * R(0),  lambda x: x * x + R(2) * 0,     8.0),
    (lambda x: x * x + R(0) / 2,  lambda x: x * x + R(0) / R(2),  6.5),
    (lambda x: x * x + (-R(0)),   lambda x: x * x - R(0),         5.0),
    (lambda x: x * x + R(0) * -1, lambda x: x * x + R(0) * R(-1), 5.0),
], ids=["R(0)*2", "2*R(0)", "R(0)/2", "-R(0)", "R(0)*-1"])
def test_derivative_does_not_depend_on_the_spelling(spelled, composite_spelling, want):
    x = seeded(3.0)
    got, ref = d1(quiet(lambda: spelled(x))), d1(quiet(lambda: composite_spelling(x)))
    show("python-number spelling d(1) at 3", got, want)
    show("composite spelling    d(1) at 3", ref, want)


@pytest.mark.parametrize("op", ["*", "/"])
def test_times_one_is_still_the_identity(op):
    """R3: x * 1 and x / 1 return x as written, so a written zero stays latent."""
    got = quiet(lambda: R(0) * 1 if op == "*" else R(0) / 1)
    show(f"R(0) {op} 1", got.coeffs_dict(), {0: 0.0})


def test_a_nonzero_value_times_a_number_is_unchanged():
    x = seeded(3.0)
    show("(3+h) * 2", (x * 2).coeffs_dict(), {0: 6.0, -1: 2.0})
    show("(3+h) / 2", (x / 2).coeffs_dict(), {0: 1.5, -1: 0.5})
    show("-(3+h)", (-x).coeffs_dict(), {0: -3.0, -1: -1.0})


@pytest.mark.parametrize("left, right", [(2, -2), (-2, 2), (3.5, -3.5)])
def test_number_on_the_left_keeps_the_left_residue(left, right):
    """A cancellation reads its residue off the LEFT operand; 2 + R(-2) is 2h."""
    got = quiet(lambda: left + R(right))
    want = quiet(lambda: R(left) + R(right))
    show(f"{left} + R({right})", got.coeffs_dict(), want.coeffs_dict())
    show("  as a value", got.coeffs_dict(), {-1: float(left)})


def test_zero_plus_nothing_is_a_no_op():
    got = quiet(lambda: 0 + Composite({}))
    show("0 + NOTHING", got.coeffs_dict(), {0: 0.0})


def test_sum_starts_from_the_written_zero_on_the_left():
    """sum() starts at the int 0, so it is 0 + x: the 0 is expressed and deposits h."""
    x = seeded(3.0)
    got = quiet(lambda: sum([x, x]))
    want = quiet(lambda: R(0) + x + x)
    show("sum([x, x])", got.coeffs_dict(), want.coeffs_dict())


def test_tag_mode_treats_a_number_like_a_composite():
    """Under TAG an unblessed infinitesimal is a conventional zero on both spellings."""
    cl.TAG(3.0)
    try:
        for label, got, want in (
                ("ZERO * 2", cl.ZERO * 2, R(2) * cl.ZERO),
                ("ZERO / 2", cl.ZERO / 2, cl.ZERO / R(2)),
                ("-ZERO", -cl.ZERO, cl.ZERO * R(-1))):
            show(f"TAG: {label}", got.coeffs_dict(), want.coeffs_dict())
            show(f"TAG: {label} is the zero", got.coeffs_dict(), {0: 0.0})
    finally:
        cl.UNTAG()


# --- completeness: d(n) past the bound warns ---------------------------------

def incomplete_warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = fn()
    return value, sum(issubclass(x.category, IncompleteOrderWarning) for x in w)


def test_d_past_the_bound_warns_and_returns():
    s = cl.sin(seeded(0.5))
    show("sin(0.5+h).complete_order", s.complete_order, 11)
    value, n = incomplete_warnings(lambda: s.d(12))
    show("sin(0.5+h).d(12) IncompleteOrderWarnings", n, 1)
    show("sin(0.5+h).d(12) value returned (term absent)", value, 0.0)


def test_d_within_the_bound_is_silent_and_right():
    s = cl.sin(seeded(0.5))
    value, n = incomplete_warnings(lambda: s.d(11))
    show("sin(0.5+h).d(11) IncompleteOrderWarnings", n, 0)
    show("sin(0.5+h).d(11) = -cos(0.5)", value, -math.cos(0.5), tol=1e-13)


def test_a_leaked_term_past_the_bound_warns():
    """exp*exp holds terms past its bound that are wrong; reading one warns."""
    e = cl.exp(seeded(1.0)) * cl.exp(seeded(1.0))
    show("exp(x)*exp(x).complete_order", e.complete_order, 14)
    value, n = incomplete_warnings(lambda: e.d(15))
    true = math.exp(2.0) * 2 ** 15
    show("exp*exp d(15) IncompleteOrderWarnings", n, 1)
    print(f"  d(15) read {value:.6g} against the true {true:.6g}: the reason for the warning")


def test_an_exact_value_never_warns():
    x = seeded(3.0)
    p = x * x * x
    show("(3+h)**3 complete_order", p.complete_order, None)
    value, n = incomplete_warnings(lambda: p.d(5))
    show("(3+h)**3 d(5) IncompleteOrderWarnings", n, 0)
    show("(3+h)**3 d(5)", value, 0.0)


def test_nth_derivative_raises_its_own_depth_and_does_not_warn():
    value, n = incomplete_warnings(lambda: cl.nth_derivative(cl.sin, 14, 0.5))
    show("nth_derivative(sin, 14, 0.5) warnings", n, 0)
    show("nth_derivative(sin, 14, 0.5) = -sin(0.5)", value, -math.sin(0.5), tol=1e-13)


# --- completeness: atan at an infinite argument -------------------------------

INF = Composite({1: 1.0})


def atan_reference(f, order):
    mpmath.mp.dps = 40
    return [float(c) for c in mpmath.taylor(f, 0, order)]


@pytest.mark.parametrize("terms, bound", [(15, 30), (40, 80)])
def test_atan_of_infinity_records_where_its_series_stops(terms, bound):
    a = quiet(lambda: cl.atan(INF, terms=terms))
    show(f"atan(1/h, terms={terms}).complete_order", a.complete_order, float(bound))
    ref = atan_reference(lambda t: -mpmath.atan(t), bound + 1)
    worst = max(abs(a.coeff(-k) - ref[k]) for k in range(1, bound + 1))
    show(f"  worst error, orders 1..{bound}", worst, 0.0, tol=1e-15)
    show(f"  order {bound + 1} is past the bound, true value", ref[bound + 1],
         (-1.0) ** ((bound + 1) // 2 + 1) / (bound + 1), tol=1e-15)


def test_atan_of_infinity_plus_one_is_right_to_its_bound():
    """Was complete 50 by inheritance, and wrong from order 25."""
    a = quiet(lambda: cl.atan(INF + R(1)))
    show("atan(1/h + 1).complete_order", a.complete_order, 30.0)
    ref = atan_reference(lambda t: -mpmath.atan(t / (1 + t)), 30)
    worst = max(abs(a.coeff(-k) - ref[k]) / max(1.0, abs(ref[k])) for k in range(1, 31))
    # The coefficients reach ~1e3 and are alternating binomial sums; 1e-10
    # relative is float64 conditioning there, measured at 1.5e-11.
    show("  worst relative error, orders 1..30", worst, 0.0, tol=1e-10)


# --- completeness: a cut quotient with only positive grades -------------------

@pytest.mark.parametrize("n, bound", [(51, -1.0), (60, -10.0), (49, 1.0)])
def test_cut_quotient_records_its_bound(n, bound):
    """The quotient keeps 50 terms, grades n-1 down to n-50, so it is complete
    to order n-50 -- negative once every grade it holds is positive."""
    q = quiet(lambda: Composite({n: 1.0}) / (INF + R(1)))
    show(f"INF**{n} / (INF + 1) complete_order", q.complete_order, bound)


def test_an_exact_quotient_stays_exact():
    x = seeded(3.0)
    q = quiet(lambda: (x * x) / x)
    show("(3+h)**2 / (3+h) complete_order", q.complete_order, None)
    show("(3+h)**2 / (3+h)", q.coeffs_dict(), {0: 3.0, -1: 1.0})


def test_a_cut_series_quotient_is_unchanged():
    q = quiet(lambda: R(1) / (R(1) + cl.ZERO))
    show("1/(1+h) complete_order", q.complete_order, 49.0)


# --- abs() is a composite: the value, negated when its dominant term is negative

@pytest.mark.parametrize("label, build, want", [
    ("abs(3+h)",          lambda: R(3) + cl.ZERO,              {0: 3.0, -1: 1.0}),
    ("abs(-3+h)",         lambda: R(-3) + cl.ZERO,             {0: 3.0, -1: -1.0}),
    ("abs(-h)",           lambda: -cl.ZERO,                    {-1: 1.0}),
    ("abs(-1/h)",         lambda: R(-1) / cl.ZERO,             {1: 1.0}),
    ("abs(<0_1 -2_0>)",   lambda: Composite({1: 0.0, 0: -2.0}), {1: 0.0, 0: 2.0}),
], ids=["3+h", "-3+h", "-h", "-1/h", "inert-zero-on-top"])
def test_abs_is_a_composite(label, build, want):
    got = quiet(lambda: abs(build()))
    show(f"{label} is a Composite", isinstance(got, Composite), True)
    show(label, got.coeffs_dict(), want)


def test_abs_of_a_written_zero_converts_first():
    """A zero operand converts (R1), so abs(R(0)) is h."""
    show("abs(R(0))", quiet(lambda: abs(R(0))).coeffs_dict(), {-1: 1.0})


def test_abs_of_a_written_zero_under_conventional_stays_zero():
    with cl.conventional():
        got = abs(R(0))
    show("conventional(): abs(R(0))", got.coeffs_dict(), {0: 0.0})


def test_abs_of_nothing_is_nothing():
    show("abs(NOTHING)", abs(Composite({})).coeffs_dict(), {})


def test_abs_of_a_log_infinity_takes_the_dominant_sign():
    L = quiet(lambda: cl.ln(R(1) / cl.ZERO))
    show("abs(-ln(1/h)) == ln(1/h)", quiet(lambda: abs(-L)).coeffs_dict(), L.coeffs_dict())


@pytest.mark.parametrize("label, f, at, want", [
    ("d/dx x*|x| at 0.5",         lambda x: x * abs(x),           0.5, 1.0),
    ("d/dx x*|x| at -0.5",        lambda x: x * abs(x),           -0.5, 1.0),
    ("d/dx e^x*|x-2| at 0.5",     lambda x: cl.exp(x) * abs(x - 2), 0.5, math.exp(0.5) * 0.5),
], ids=["x|x|@0.5", "x|x|@-0.5", "exp|x-2|"])
def test_derivatives_through_abs(label, f, at, want):
    show(label, quiet(lambda: cl.derivative(f, at)), want, tol=1e-15)


def test_second_derivative_through_abs():
    got = quiet(lambda: cl.nth_derivative(lambda x: abs(x) ** 3, 2, -1.5))
    show("d2/dx2 |x|**3 at -1.5", got, 9.0, tol=1e-13)


@pytest.mark.parametrize("a, b, want", [(0.0, 2.0, 2.0), (-2.0, -0.5, 1.875)])
def test_integral_of_abs_away_from_the_kink(a, b, want):
    got = quiet(lambda: cl.integrate(lambda x: abs(x), a, b))
    show(f"integral of |x| on [{a}, {b}]", got, want, tol=1e-12)


# --- reflected operators keep a composite left operand ------------------------

def test_rsub_keeps_a_composite_left_operand():
    A, B = Composite({0: 2.0, -1: 1.0}), Composite({0: 3.0, -1: 5.0})
    got = A.__rsub__(B)
    show("A.__rsub__(B) is B - A", got.coeffs_dict(), (B - A).coeffs_dict())
    show("  d(1)", got.d(1), 4.0)


def test_rtruediv_keeps_a_composite_left_operand():
    A, B = Composite({0: 2.0, -1: 1.0}), Composite({0: 3.0, -1: 5.0})
    got = A.__rtruediv__(B)
    show("A.__rtruediv__(B) is B / A", got.coeffs_dict(), (B / A).coeffs_dict())
    show("  d(1)", got.d(1), 1.75, tol=1e-15)


def test_audit_through_a_negation_keeps_the_derivative():
    """-x makes the next operand a subclass on the right, which is the path."""
    a = quiet(lambda: forensics.audit(lambda x: -x - x, 3.0))
    show("audit(-x - x, 3) derivative", a.derivative, -2.0, tol=1e-15)


def test_audit_through_abs_sees_the_conditioning():
    """kappa = |x f'/f| at x = 1 - 1e-9 is about 1e9 for |x - 1|."""
    a = quiet(lambda: forensics.audit(lambda x: abs(x - 1.0), 1 - 1e-9))
    show("audit(|x-1|) derivative", a.derivative, -1.0, tol=1e-15)
    show("audit(|x-1|) kappa", a.kappa, (1 - 1e-9) / 1e-9, tol=1e3)
    show("audit(|x-1|) verdict", a.verdict, forensics.ILL_CONDITIONED)
