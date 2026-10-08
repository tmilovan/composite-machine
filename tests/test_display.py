"""Rich display: the grade table, and the three states a reader must not miss.

`str(composite)` is a flat sum, which reads correctly only while the grades are
a descending run of integers on one axis.  The table exists for the cases where
it is not, so the tests are those cases: a denoted jet whose coefficients are
not the classical derivatives, terms past the order the series is vouched for,
a log-axis grade beside a power-axis one, a half grade, and an expressed zero
sitting where the value would be.

Two tests are regressions for bugs in this module rather than in the library.
Both came from treating a grade as something it was not.
"""
import math

import pytest

import composite.composite_lib as cl
import composite.display as display
from composite.composite_lib import Composite, R, ZERO, ln, sin, sqrt
from composite.backends import config


@pytest.fixture(autouse=True)
def dict_backend():
    """The log axis needs vector dimensions, which the dict backend carries."""
    config.use_dict()
    cl._refresh_constants()
    yield
    config.use_sparse_dense()
    cl._refresh_constants()


def meanings(value):
    return {grade: meaning for grade, _, meaning in display.grade_rows(value)}


def coefficients(value):
    return {grade: coefficient for grade, coefficient, _ in display.grade_rows(value)}


# --- the ordinary case -------------------------------------------------------

def test_grade_rows_reads_a_plain_jet():
    # x*x at 3 is |9|_0 + |6|_-1 + |1|_-2, so f(3)=9, f'(3)=6, f''(3)=2.
    rows = display.grade_rows((R(3) + ZERO) * (R(3) + ZERO))
    assert [g for g, _, _ in rows] == [0, -1, -2], \
        "got grades %r, want [0, -1, -2] leading first" % [g for g, _, _ in rows]
    assert [c for _, c, _ in rows] == [9.0, 6.0, 1.0], \
        "got coefficients %r, want [9, 6, 1]" % [c for _, c, _ in rows]
    said = [m for _, _, m in rows]
    assert said[0] == "the value", "got %r" % said[0]
    assert "d(1) = 6" in said[1], "got %r, want the derivative 6 not the coefficient" % said[1]
    assert "d(2) = 2" in said[2], "got %r, want d(2) = 1 * 2! = 2" % said[2]


@pytest.mark.parametrize("backend", ["dict", "sparse_dense", "dense_series"])
def test_a_plain_jet_reads_the_same_on_every_backend(backend):
    getattr(config, "use_" + backend)()
    cl._refresh_constants()
    assert coefficients((R(3) + ZERO) * (R(3) + ZERO)) == {0: 9.0, -1: 6.0, -2: 1.0}, \
        "backend %s gave %r, want {0: 9, -1: 6, -2: 1}" % (
            backend, coefficients((R(3) + ZERO) * (R(3) + ZERO)))


# --- the three states the string does not show -------------------------------

def test_a_denoted_jet_is_flagged_on_every_affected_row():
    # One operand went through a cancellation, so the coefficients are the
    # derivatives of the function the expression DENOTES, not of x*x.  The
    # string |9|_0 + |24|_-1 + ... gives no sign of that.
    x = R(3) + ZERO
    value = x * x * (R(2) - R(2)) + x * x
    assert value.denotation_order == 1.0, "got %r, want 1.0" % value.denotation_order
    said = meanings(value)
    assert "DENOTED" not in said[0], "grade 0 is order 0, below the marker: got %r" % said[0]
    for grade in (-1, -2, -3):
        assert "DENOTED" in said[grade], "grade %s: got %r, want the marker" % (grade, said[grade])
    assert "d(1) = 24" in said[-1], "got %r, want 24 (the denoted first derivative)" % said[-1]


def test_terms_past_complete_order_are_flagged():
    # sin to 6 terms squared reaches grade -10 while only 6 orders are vouched
    # for, and the library counts 2 leaked terms.  Each factor is complete to
    # 5 from a leading order of 1, so 4 orders past its lead; the product leads
    # at order 2 and is complete to 2 + 4 = 6.  (Until 2026-10-07 the bound was
    # the smaller absolute one, 5, which under-claimed here and over-claimed
    # for X / h and X * (1/h); see _scaled_complete.)
    value = sin(ZERO, terms=6) * sin(ZERO, terms=6)
    assert value.complete_order == 6, "got %r, want 6" % value.complete_order
    assert len(value.leaked_coeffs()) == 2, \
        "got %d leaked, want 2" % len(value.leaked_coeffs())
    said = meanings(value)
    for grade in (-2, -4, -6):
        assert "not vouched" not in said[grade], "grade %s: got %r" % (grade, said[grade])
    for grade in (-8, -10):
        assert "not vouched" in said[grade], \
            "grade %s: got %r, want the warning" % (grade, said[grade])


def test_an_expressed_zero_is_not_the_value():
    # x*x - 9 at 3 keeps |0|_0: a zero that is a term, inert under R2.  The
    # table must not print "the value" beside it.
    x = R(3) + ZERO
    said = meanings(x * x - R(9))
    assert said[0] == "expressed zero, inert", "got %r" % said[0]
    assert coefficients(x * x - R(9))[0] == 0.0


# --- the shapes where a flat string stops working ----------------------------

def test_the_log_axis_sorts_and_is_named():
    # Regression: grades here are 1 and (0, 1).  `sorted(coeffs_dict())` raises
    # TypeError on that mix, so the rows have to be keyed on an axis tuple.
    value = R(1) / ZERO + ln(R(1) / ZERO)
    grades = [g for g, _, _ in display.grade_rows(value)]
    assert grades == [1, (0, 1)], "got %r, want [1, (0, 1)]" % grades
    said = meanings(value)
    assert "unbounded, order 1" in said[1], "got %r" % said[1]
    assert "log axis, depth 1" in said[(0, 1)], "got %r" % said[(0, 1)]
    # A power-axis reading of a log grade would print the string "order -0 term".
    assert "-0" not in said[(0, 1)], "got %r, want no bogus power reading" % said[(0, 1)]


def test_a_branch_point_suppresses_the_derivative_reading():
    # With a half grade present, coefficient * n! is not a derivative, so the
    # integer row must NOT claim one.
    value = sqrt(ZERO) + ZERO
    said = meanings(value)
    assert "branch point, order 0.5" in said[-0.5], "got %r" % said[-0.5]
    assert "d(1)" not in said[-1], \
        "got %r, want no derivative claim beside a branch point" % said[-1]


def test_nothing_renders_as_absence():
    assert display.grade_rows(Composite({})) == []
    rendered = Composite({})._repr_html_()
    assert "NOTHING" in rendered and "absence, not zero" in rendered, rendered


# --- every renderer, and the guard -------------------------------------------

def built_objects():
    from composite import audit, explain, analyse, budget, Quantity, forensics
    from composite.transseries import from_series
    x = R(3) + ZERO
    series, _ = from_series([((-1) ** n) * math.factorial(n) for n in range(12)])
    return [
        ("Composite", x * x),
        ("Explanation", explain(lambda t: R(1) / t, 0)),
        ("Audit", audit(lambda t: (R(1) - forensics.F.cos(t)) / (t * t), 1e-5)),
        ("Singularity", analyse([math.comb(2 * n, n) for n in range(24)])),
        ("Budget", budget(lambda length, width: length * width,
                          {"length": Quantity(2.0, 0.1), "width": Quantity(3.0, 0.2)})),
        ("Transseries", series),
    ]


@pytest.mark.parametrize("name,value", built_objects(), ids=[n for n, _ in built_objects()])
def test_every_renderer_produces_html_without_falling_back(name, value):
    rendered = value._repr_html_()
    assert "rich display failed" not in rendered, \
        "%s fell through to the guard: %s" % (name, rendered[-300:])
    assert rendered.startswith("<div") and rendered.endswith("</div>"), \
        "%s produced %r..." % (name, rendered[:60])


def test_transseries_rows_do_not_coerce_the_series():
    # Regression: sorting sectors.items() with the grade-axis key fed the
    # Composite to float(), and float() of an unbounded composite raises.
    # Sector 1 here holds the Stokes constant at grade +1, so it is unbounded.
    from composite.transseries import from_series
    series, detail = from_series([((-1) ** n) * math.factorial(n) for n in range(12)])
    assert abs(detail["stokes"] - math.pi) < 1e-12, \
        "got stokes %r, want pi" % detail["stokes"]
    rendered = series._repr_html_()
    assert "rich display failed" not in rendered, rendered[-300:]
    assert "exp(-1/h)" in rendered and "perturbative" in rendered, rendered


def test_an_uninstrumented_audit_says_so_loudly():
    # math.sin(float(t)) returns a plain float, so nothing was measured.  A
    # quiet "stable" here is the worst thing the panel could show.
    from composite import audit
    result = audit(lambda t: math.sin(float(t)), 0.5)
    assert result.instrumented is False
    rendered = result._repr_html_()
    assert "NOT INSTRUMENTED" in rendered, rendered
    assert "font-weight:bold" in rendered, "the badge must be the loud one"


# --- installing and removing --------------------------------------------------

def test_enable_and_disable_round_trip():
    assert hasattr(Composite, "_repr_html_")
    display.disable()
    assert not hasattr(Composite, "_repr_html_"), "disable left the hook behind"
    touched = display.enable()
    assert "Composite" in touched, "got %r" % touched
    assert hasattr(Composite, "_repr_html_")


def test_text_from_the_caller_is_escaped():
    from composite.explain import Explanation
    rendered = Explanation("<script>x</script>", 0, "value", value=1.0, slope=2.0)._repr_html_()
    assert "&lt;script&gt;" in rendered, rendered[:200]
    assert "<script>" not in rendered, "an unescaped tag reached the page"


# --- notation: `+` is an operation between composites, not a separator -------
# A negative coefficient in the PLAIN form sits against the space (`<3_0 -1_-0.5>`),
# which was accepted deliberately: it needs a fractional or vector dimension and
# a negative coefficient together, and the glyph form's bars prevent it.

def test_terms_are_separated_by_spaces_not_plus():
    # Zero Rules v2 section 0: `<9_0 6_-1 1_-2>` is ONE number; `<9_0> + <6_-1>`
    # is an operation between two.  `__repr__` follows that rule now, so this
    # pins the form the display layer renders and a front end reads.
    value = (R(3) + ZERO) * (R(3) + ZERO)
    assert display.notation(value) == "<|9|₀ |6|₋₁ |1|₋₂>", \
        "got %r" % display.notation(value)
    assert " + " not in display.notation(value), display.notation(value)


def test_no_plus_joiner_survives_into_the_rendered_table():
    for value in ((R(3) + ZERO) * (R(3) + ZERO),
                  (R(3) + ZERO) * (R(3) + ZERO) - R(9),
                  R(1) / ZERO + ln(R(1) / ZERO),
                  sqrt(ZERO) + ZERO):
        rendered = display.composite_html(value)
        assert " + " not in rendered, \
            "a `+` joiner reached the page for %s: %r" % (value, rendered[:200])


def test_notation_keeps_the_libraries_own_term_forms():
    # Only the separator changes.  The glyph/underscore choice and the exact
    # fraction printing stay whatever __repr__ decided, so there is one place
    # that owns them.
    fractional = R(3) + sqrt(ZERO)
    assert display.notation(fractional) == "<3_0 1_-0.5>", \
        "got %r" % display.notation(fractional)
    logs = R(1) / ZERO + ln(R(1) / ZERO)
    assert display.notation(logs) == "<1_1 1_(0,1)>", "got %r" % display.notation(logs)


def test_nothing_is_not_bracketed_as_a_term_list():
    # There are no terms, so there is no list.  NOTHING is absence, not <>.
    assert display.notation(Composite({})) == "∅", \
        "got %r" % display.notation(Composite({}))
