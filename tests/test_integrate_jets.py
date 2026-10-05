"""integrate_jets: the integral by meeting composites.

One composite per node.  Each node's antiderivative is carried to the composite
midpoint (m - x) + h and the two sides are subtracted: grade 0 is the integral.
A node is added only where the two sides disagree on value or slope there.
Singular endpoints are read by grade, from the endpoint as the composite zero.
h is never given a real value: no panel width is substituted, nothing is
sampled, no finite difference is taken.

Every check prints got against known and the evaluations spent, pass or fail.
Tolerances are fixed.  The evaluation ceilings are the counts measured when this
was written (2026-10-05, sparse-dense backend, jets to order 50): they guard the
point of the construction, which is few evaluations.  The old integrate() spent
10,834 evaluations on the 21 regular integrals below; these spend 52.
"""
import math

import pytest

import composite.composite_lib as cl
from composite.backends import config
from composite.composite_lib import (Composite, R, ZERO, antiderivative, cos, cosh,
                                     exp, integrate_jets, ln, sin, sqrt)

E, PI = math.e, math.pi


class Count:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, x):
        self.n += 1
        return self.f(x)


def _terms(c):
    out = {}
    for d, v in c.coeffs_dict().items():
        if v != 0.0:
            out[tuple(float(e) for e in d) if isinstance(d, tuple) else float(d)] = v
    return out


# --- regular integrands --------------------------------------------------------

REGULAR = [
    # name, f, a, b, known, evaluation ceiling
    ("x^2 [0,1]", lambda x: x ** 2, 0, 1, 1 / 3, 2),
    ("x^3 [0,1]", lambda x: x ** 3, 0, 1, 0.25, 2),
    ("exp [0,1]", exp, 0, 1, E - 1, 2),
    ("sin [0,pi]", sin, 0, PI, 2.0, 2),
    ("sin [-1,1]", sin, -1, 1, 0.0, 2),
    ("cosh [0,1]", cosh, 0, 1, math.sinh(1), 2),
    ("exp(-x^2) [1,2]", lambda x: exp(-(x * x)), 1, 2,
     math.sqrt(PI) / 2 * (math.erf(2) - math.erf(1)), 2),
    ("exp(-x^2) [0,3]", lambda x: exp(-(x * x)), 0, 3, math.sqrt(PI) / 2 * math.erf(3), 2),
    ("x sin x [0,1]", lambda x: x * sin(x), 0, 1, math.sin(1) - math.cos(1), 2),
    ("x^2 e^x [0,1]", lambda x: x ** 2 * exp(x), 0, 1, E - 2, 2),
    ("sin^2 [0,pi/2]", lambda x: sin(x) * sin(x), 0, PI / 2, PI / 4, 2),
    ("sin^3 [0,pi/2]", lambda x: sin(x) ** 3, 0, PI / 2, 2 / 3, 2),
    ("e^-x cos x [0,1]", lambda x: exp(-x) * cos(x), 0, 1,
     0.5 * (1 + math.exp(-1) * (math.sin(1) - math.cos(1))), 2),
    ("cos^2 [0,pi]", lambda x: cos(x) * cos(x), 0, PI, PI / 2, 2),
    ("sin cos [0,pi/2]", lambda x: sin(x) * cos(x), 0, PI / 2, 0.5, 2),
    ("1/(1+x) [0,1]", lambda x: R(1) / (R(1) + x), 0, 1, math.log(2), 2),
    # beyond the reach of a single jet: the pole at -1 is 1 from the left end
    ("1/(1+x) [0,2]", lambda x: R(1) / (R(1) + x), 0, 2, math.log(3), 3),
    ("1/(1+x^2) [0,2]", lambda x: R(1) / (R(1) + x * x), 0, 2, math.atan(2), 3),
    # poles at +-i/5, close to the interval
    ("1/(1+25x^2) [-1,1]", lambda x: R(1) / (R(1) + 25 * x * x), -1, 1, 0.4 * math.atan(5), 7),
    ("ln(x)/x [1,e]", lambda x: ln(x) / x, 1, E, 0.5, 3),
    ("sqrt [0.25,1]", sqrt, 0.25, 1, 2 / 3 * (1 - 0.125), 4),
]

# --- singular endpoints, read by grade -------------------------------------------

SINGULAR = [
    ("sqrt [0,1]", sqrt, 0, 1, 2 / 3, 2),
    ("1/sqrt [0,1]", lambda x: 1 / sqrt(x), 0, 1, 2.0, 2),
    ("cos/sqrt [0,1]", lambda x: cos(x) / sqrt(x), 0, 1, 1.8090484758005438, 2),
    ("ln(x) [0,1]", ln, 0, 1, -1.0, 2),
    ("x ln(x) [0,1]", lambda x: x * ln(x), 0, 1, -0.25, 2),
    ("1/sqrt(1-x) [0,1]", lambda x: 1 / sqrt(1 - x), 0, 1, 2.0, 2),
    ("1/sqrt(x(1-x)) [0,1]", lambda x: 1 / sqrt(x * (1 - x)), 0, 1, PI, 2),
    ("1/sqrt|x-1/2| [0,1]", lambda x: 1 / sqrt(sqrt((x - R(0.5)) * (x - R(0.5)))),
     0, 1, 2 * math.sqrt(2), 4),
    ("1/(x-1/2) [0,1], principal value", lambda x: R(1) / (x - R(0.5)), 0, 1, 0.0, 4),
]


@pytest.mark.parametrize("name,f,a,b,known,ceiling", REGULAR + SINGULAR,
                         ids=[c[0] for c in REGULAR + SINGULAR])
def test_integral(name, f, a, b, known, ceiling):
    g = Count(f)
    got = integrate_jets(g, a, b).to_ieee754()
    err = abs(got - known)
    print(f"\n  {name:34s} got {got!r:22} known {known!r:22} err {err:.1e}"
          f"  evaluations {g.n} (ceiling {ceiling})")
    assert err <= 1e-10 * max(1.0, abs(known))
    assert g.n <= ceiling


# --- what a singular node keeps ---------------------------------------------------

KEPT = [
    # name, f, a, b, expected terms -- the limit at a singular node is x +- h
    ("1/x^2 [0,1] = 1/h - 1", lambda x: 1 / (x * x), 0, 1, {1.0: 1.0, 0.0: -1.0}),
    ("1/x [0,1] = ln(1/h)", lambda x: 1 / x, 0, 1, {(0.0, 1.0): 1.0}),
    ("1/(x-1/2)^2 [0,1] = 2/h - 4", lambda x: R(1) / ((x - R(0.5)) * (x - R(0.5))),
     0, 1, {1.0: 2.0, 0.0: -4.0}),
    ("1/sqrt [0,1] = 2 - 2 sqrt(h)", lambda x: 1 / sqrt(x), 0, 1, {0.0: 2.0, -0.5: -2.0}),
    ("ln(x) [0,1] = -1 + h ln(1/h) + h", ln, 0, 1,
     {(0.0, 0.0): -1.0, (-1.0, 1.0): 1.0, (-1.0, 0.0): 1.0}),
    ("1/(x-1/2) [0,1]: the two logs cancel", lambda x: R(1) / (x - R(0.5)), 0, 1, {}),
]


@pytest.mark.parametrize("name,f,a,b,expected", KEPT, ids=[c[0] for c in KEPT])
def test_kept_terms(name, f, a, b, expected):
    got = _terms(integrate_jets(f, a, b))
    print(f"\n  {name}: got {got}  known {expected}")
    assert set(got) == set(expected)
    for k, v in expected.items():
        assert abs(got[k] - v) <= 1e-12 * max(1.0, abs(v))


# --- antiderivative, every grade --------------------------------------------------

ANTIDERIVATIVES = [
    ("1/sqrt(h) -> 2 sqrt(h)", lambda: 1 / sqrt(ZERO), {-0.5: 2.0}),
    ("1/h**2 -> -1/h", lambda: 1 / (ZERO * ZERO), {1.0: -1.0}),
    ("1/h -> -ln(1/h)", lambda: 1 / ZERO, {(0.0, 1.0): -1.0}),
    ("ln(h) -> h ln(h) - h", lambda: ln(ZERO), {(-1.0, 1.0): -1.0, (-1.0, 0.0): -1.0}),
    ("h ln(1/h) -> h^2 L/2 + h^2/4", lambda: ZERO * ln(1 / ZERO),
     {(-2.0, 1.0): 0.5, (-2.0, 0.0): 0.25}),
    ("(3+h)^2 unchanged", lambda: (3 + ZERO) ** 2, {-1.0: 9.0, -2.0: 3.0, -3.0: 1 / 3}),
]


@pytest.mark.parametrize("name,jet,expected", ANTIDERIVATIVES,
                         ids=[c[0] for c in ANTIDERIVATIVES])
def test_antiderivative_terms(name, jet, expected):
    got = _terms(antiderivative(jet()))
    print(f"\n  {name}: got {got}  known {expected}")
    assert set(got) == set(expected)
    for k, v in expected.items():
        assert abs(got[k] - v) <= 1e-12 * max(1.0, abs(v))


def test_fractional_grades_stay_exact():
    """On the exact backend 1/2 - 1 is -1/2, kept as a Fraction."""
    from fractions import Fraction
    config.use_fractional_dict()
    cl._refresh_constants()
    try:
        got = {d: v for d, v in antiderivative(1 / sqrt(cl.ZERO)).c.items() if v != 0.0}
        print(f"\n  1/sqrt(h) on the fractional backend: got {got}  known {{-1/2: 2.0}}")
        assert list(got) == [Fraction(-1, 2)] and isinstance(list(got)[0], Fraction)
        assert abs(got[Fraction(-1, 2)] - 2.0) <= 1e-12
    finally:
        config.use_sparse_dense()
        cl._refresh_constants()


# --- other backends ---------------------------------------------------------------

@pytest.mark.parametrize("backend", ["dict", "dense_series"])
def test_backends_agree(backend):
    """The dense-series case needed a fix: a single term at grade 1.5 against a
    step-1.5 series was refused as incommensurable."""
    getattr(config, "use_" + backend)()
    cl._refresh_constants()
    try:
        for name, f, a, b, known in (("exp [0,1]", exp, 0, 1, E - 1),
                                     ("sqrt [0,1]", sqrt, 0, 1, 2 / 3),
                                     ("1/(1+x) [0,2]", lambda x: R(1) / (R(1) + x), 0, 2,
                                      math.log(3))):
            got = integrate_jets(f, a, b).to_ieee754()
            print(f"\n  {backend:12s} {name:14s} got {got!r:22} known {known!r}")
            assert abs(got - known) <= 1e-10 * max(1.0, abs(known))
    finally:
        config.use_sparse_dense()
        cl._refresh_constants()


# --- refusals -----------------------------------------------------------------------

def test_a_float_integrand_is_refused():
    with pytest.raises(ValueError):
        integrate_jets(lambda x: 1.0, 0, 1)


def test_an_integrand_with_its_own_infinitesimal_is_refused():
    """Not handled yet.  Each node's composite denotes a different function, the
    two sides never agree, and the node cap stops it instead of a hang or a
    silently wrong sum."""
    a = R(1) + ZERO
    with pytest.raises(ValueError):
        integrate_jets(lambda x: exp(a * x), 0, 1, max_nodes=20)


# --- integrate() routes 1D finite integrals here -----------------------------------

from composite.composite_lib import integrate


@pytest.mark.parametrize("name,a,b,known", [
    ("x^2 [0,1]", 0, 1, 1 / 3),
    ("x^2 [1,0], reversed", 1, 0, -1 / 3),
    ("x^2 [1,1], empty", 1, 1, 0.0),
], ids=["forward", "reversed", "empty"])
def test_integrate_limits(name, a, b, known):
    got = integrate(lambda x: x * x, a, b)
    print(f"\n  integrate {name}: got {got!r}  known {known!r}")
    assert isinstance(got, float)
    assert abs(got - known) <= 1e-14


def test_integrate_divergent_is_infinite_not_nan():
    got = integrate(lambda x: 1 / (x * x), 0, 1)
    print(f"\n  integrate 1/x^2 [0,1]: got {got!r}  known inf (the old path returned nan)")
    assert got == math.inf


def test_integrate_refuses_its_own_infinitesimal():
    a = R(1) + ZERO
    with pytest.raises(ValueError, match="infinitesimal of its own"):
        integrate(lambda x: exp(a * x), 0, 1)


# --- line integrals --------------------------------------------------------------

def test_a_float_curve_is_refused():
    """math.cos turns the seeded t into a float and loses the tangent.  It is
    refused, not differenced -- and caught at the coercion itself, because a
    value comparison cannot see it: cos and sin take the same values at 0 and
    2*pi, so a coerced circle would look constant and integrate to 0."""
    with pytest.raises(ValueError, match="not composite"):
        integrate(lambda x, y: 1, (0, 2 * PI), curve=lambda t: [math.cos(t), math.sin(t)])


def test_composite_circle():
    got = integrate(lambda x, y: 1, (0, 2 * PI), curve=lambda t: [cos(t), sin(t)])
    print(f"\n  circumference with composite cos/sin: got {got!r}  known {2 * PI!r}")
    assert abs(got - 2 * PI) <= 1e-12


# --- the wrappers now on the same path ----------------------------------------------

from composite.composite_lib import definite_integral, improper_integral_to, integrate_stepped


def test_definite_integral():
    got = definite_integral(exp, 0, 1)
    print(f"\n  definite_integral exp [0,1]: got {got!r}  known {E - 1!r}")
    assert abs(got - (E - 1)) <= 1e-14


@pytest.mark.parametrize("name,f,step,known", [
    ("x^2 [0,1], step 0.5", lambda x: x * x, 0.5, 1 / 3),
    ("1/sqrt [0,1], step 0.3", lambda x: 1 / sqrt(x), 0.3, 2.0),
    ("cos [0,1], step 0.25", cos, 0.25, math.sin(1)),
], ids=["x2", "1/sqrt", "cos"])
def test_integrate_stepped(name, f, step, known):
    val, err = integrate_stepped(f, 0, 1, step=step)
    print(f"\n  integrate_stepped {name}: got {val.st()!r}  known {known!r}  err slot {err}")
    assert abs(val.st() - known) <= 1e-12
    assert math.isnan(err)


def test_improper_integral_to():
    val, err = improper_integral_to(lambda x: 1 / sqrt(x), 0, 1)
    print(f"\n  improper_integral_to 1/sqrt [0,1]: got {val!r}  known <2_0 -2_-0.5>")
    assert abs(val.st() - 2.0) <= 1e-12 and abs(val.coeff(-0.5) + 2.0) <= 1e-12
    assert math.isnan(err)


# --- improper integrals: the node at infinity is x = 1/h -------------------------------

INF_ = math.inf

IMPROPER = [
    # name, f, a, b, known, evaluation ceiling -- composite path.  Each half
    # line spends one evaluation reading its jet at infinity, whose radius
    # decides where the tail starts (_tail_start).
    ("1/x^2 [1,inf)", lambda x: 1 / (x * x), 1, INF_, 1.0, 3),
    ("1/(1+x^2) (-inf,inf)", lambda x: R(1) / (R(1) + x * x), -INF_, INF_, PI, 10),
    ("exp(-x) [0,inf), transseries tail", lambda x: exp(-x), 0, INF_, 1.0, 5),
    ("x^2 exp(-x) [0,inf)", lambda x: x * x * exp(-x), 0, INF_, 2.0, 5),
    ("exp(x) (-inf,0]", lambda x: exp(x), -INF_, 0, 1.0, 5),
    ("e^-u/(1+u) [0,inf), resummation's Laplace integral",
     lambda u: exp(-u) / (R(1) + u), 0, INF_, 0.5963473623231941, 9),
]


@pytest.mark.parametrize("name,f,a,b,known,ceiling", IMPROPER, ids=[c[0] for c in IMPROPER])
def test_improper_composite(name, f, a, b, known, ceiling):
    g = Count(f)
    got = integrate(g, a, b)
    err = abs(got - known)
    print(f"\n  {name:50s} got {got!r:22} known {known!r:22} err {err:.1e}"
          f"  evaluations {g.n} (ceiling {ceiling})")
    assert err <= 1e-14 * max(1.0, abs(known))
    assert g.n <= ceiling


def test_improper_divergent():
    got = integrate(lambda x: 1 / x, 1, INF_)
    print(f"\n  1/x [1,inf): got {got!r}  known inf")
    assert got == INF_


@pytest.mark.parametrize("name,f,a,b,known", [
    ("exp(-x^2) (-inf,inf): level two", lambda x: exp(-(x * x)), -INF_, INF_, math.sqrt(PI)),
    ("exp(-x/2) [0,inf): non-integer sector", lambda x: exp(-x / 2), 0, INF_, 2.0),
    ("exp(-x) sin x [0,inf): sin(1/h) is a range", lambda x: exp(-x) * sin(x), 0, INF_, 0.5),
], ids=["gaussian", "rate-1/2", "oscillating"])
def test_improper_falls_back(name, f, a, b, known):
    """Not representable at the transseries level: the library refuses, and
    integrate falls back to the old improper_integral."""
    got = integrate(f, a, b)
    print(f"\n  {name}: got {got!r}  known {known!r}  err {abs(got - known):.1e}")
    assert abs(got - known) <= 1e-10


def test_transseries_tail_keeps_the_flat_term():
    """int_1^X x e^-x = 2/e - (X + 1) e^-X, and at X = 1/h the second term is
    (1/h + 1) exp(-1/h): below every power, kept as a sector."""
    from composite.composite_lib import integrate_jets_tail
    r = integrate_jets_tail(lambda x: x * exp(-x), 1.0)
    print(f"\n  x e^-x [1,inf): got {r!r}")
    assert abs(r.sectors[0].st() - 2 / E) <= 1e-15
    assert _terms(r.sectors[1]) == {1.0: -1.0, 0.0: -1.0}


def test_tail_starts_where_the_jet_at_infinity_reaches():
    """A Pade-type tail: poles at |x| = 36 leave the jet at infinity a radius of
    1/36 in u = 1/x.  Started at x = 1 the meeting point lay outside it and the
    two sides never agreed; started where the radius says, it resolves."""
    from composite.composite_lib import _tail_start
    f = lambda x: exp(-x) / (R(1) + x * x / R(1296))      # poles at x = +-36i
    start = _tail_start(f, 0.0)
    got = integrate(f, 0, INF_)
    print(f"\n  tail start {start} (pole modulus 36);  integral got {got!r}")
    assert start >= 36
    assert math.isfinite(got)
