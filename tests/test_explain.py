"""explain() and lead_order(): what a function does at a point, in one call.

Every case here is one a caller actually hits: a removable hole, a pole, a
corner, a log divergence, a formula that loses digits, and the two escapes
(a math.* call that never saw the composite, and nothing at all).
"""
import math

from composite.composite_lib import Composite, R, ZERO, cos, ln, sin, sqrt
from composite.explain import explain

H = ZERO


# --- lead_order / lead_dim ---------------------------------------------------

def test_lead_order_reads_the_dominant_term():
    assert R(3).lead_order() == 0            # an ordinary number
    assert (R(3) + H).lead_order() == 0      # the infinitesimal does not lead
    assert H.lead_order() == 1               # infinitesimal, first order
    assert (H * H).lead_order() == 2
    assert (R(1) / H).lead_order() == -1     # unbounded, first order
    assert (R(1) / (H * H)).lead_order() == -2


def test_lead_order_handles_fractional_and_log_axes():
    assert sqrt(H).lead_order() == 0.5       # a branch point, not an integer order
    assert ln(H).lead_dim() == (0, 1)        # the log axis, where max(coeffs) would raise
    assert ln(H).lead_order() == -1
    assert (H * ln(H)).lead_order() == 1     # infinitesimal, despite the log factor


def test_lead_order_skips_zero_coefficients_and_nothing():
    assert Composite({}).lead_order() is None
    assert (R(1) - R(1)).lead_order() is None        # all coefficients zero
    assert ((R(1) + H) - H).lead_order() == 0        # the cancelled term does not lead


# --- explain -----------------------------------------------------------------

def test_removable_hole_gives_the_value_and_the_slope():
    e = explain(lambda x: sin(x) / x, 0)
    assert e.kind == "value"
    assert abs(e.value - 1.0) < 1e-12
    assert abs(e.slope) < 1e-12


def test_pole_reports_its_order_and_coefficient():
    e = explain(lambda x: R(1) / x, 0)
    assert e.kind == "unbounded" and e.order == 1 and e.coefficient == 1.0
    e2 = explain(lambda x: R(1) / (x * x), 0)
    assert e2.kind == "unbounded" and e2.order == 2
    e3 = explain(lambda x: R(1) / (x - R(2)), 2)
    assert e3.kind == "unbounded" and e3.order == 1


def test_corner_reports_the_fractional_order():
    e = explain(lambda x: sqrt(x), 0)
    assert e.kind == "corner" and e.order == 0.5


def test_log_divergence_is_not_a_pole():
    e = explain(lambda x: ln(x), 0)
    assert e.kind == "log-unbounded"          # unbounded, but slower than any power
    e2 = explain(lambda x: x * ln(x), 0)
    assert e2.kind == "steep" and abs(e2.value) < 1e-12   # value 0, slope unbounded
    e3 = explain(lambda x: R(1) / ln(x), 0)
    assert e3.kind == "log-value"             # reaches 0, with no ordinary slope


def test_ordinary_point():
    e = explain(lambda x: x * x + R(3) * x, 2)
    assert e.kind == "value" and e.value == 10.0 and e.slope == 7.0
    assert e.stability == "stable"


def test_stability_is_reported_for_a_formula_that_loses_digits():
    bad = explain(lambda x: (R(1) - cos(x)) / (x * x), 0)
    good = explain(lambda x: R(2) * sin(x / R(2)) ** 2 / (x * x), 0)
    assert abs(bad.value - 0.5) < 1e-9 and abs(good.value - 0.5) < 1e-9
    assert bad.stability != "stable" and "rewrite" in bad.advice
    assert good.stability == "stable"


def test_escapes_are_named_rather_than_guessed():
    e = explain(lambda x: math.sin(float(x)), 0)     # left the composite at float()
    assert e.kind == "constant"
    e2 = explain(lambda x: Composite({}), 1)
    assert e2.kind == "nothing"


def test_str_is_a_sentence():
    assert "blows up" in str(explain(lambda x: R(1) / x, 0))
    assert "corner" in str(explain(lambda x: sqrt(x), 0))
