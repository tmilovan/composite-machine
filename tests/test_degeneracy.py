"""Flagging a conventionally degenerate Taylor reading.

THE RULE: every composite carries the FIRST infinitesimal source it descends
from, as one id, and a flag that goes up when a second, different one reaches
it.  The first source is the one the derivatives are read against, whether or
not anything called it a seed, so the same rule covers a seeded extraction and
a bare calculation with no seed in it.

A FLAG, NOT A COUNT.  Knowing a third source arrived says nothing the second did
not: the conventional reading is off either way.  One id is enough alongside it,
because a number that is not yet flagged descends from at most one source, so
its first id names its whole set.  A flag ALONE is not enough: in `x*x + x` both
operands carry an infinitesimal and it is the same one.

CARRIED ON THE RESULT, NOT READ OFF THE OPERANDS.  An infinitesimal arrives as
the RESULT of an operation as often as it arrives as an operand: `x*x + (x - x)`
has no infinitesimal operand anywhere and two sources by the end.

The composite is never wrong.  What can be wrong is reading grade -n as the nth
derivative of the function the caller had in mind, and that reading holds only
when the seed is the ONLY thing that wrote to those grades.

`x*x + R(0)` is not x squared.  The written zero is an infinitesimal and it is
one unit of the seed, so the expression is x**2 + h, and h is x - a.  The
function is x**2 + x - 2 and its derivative at 2 is 5.  The composite returning
5 is correct.  It is wrong only against x**2, which nobody wrote.

So this does not refuse and does not correct.  It reports that a second source
entered, which is the one fact a caller expecting textbook derivatives cannot
otherwise recover: dimension 0 stays right while every derivative moves.
"""
import pytest

import composite.composite_lib as cl
from composite.composite_lib import (R, ZERO, INF, Composite, sin, cos, exp, sqrt,
                                     ln, taylor_degeneracy, degeneracy_watch,
                                     Degeneracy, conventional, all_derivatives,
                                     track_degeneracy, set_degeneracy_tracking,
                                     degeneracy_tracking)
from composite.backends import config
from composite import explain


SQ = [4.0, 4.0, 2.0, 0.0]          # x**2 at x = 2

CLEAN = [
    ("x*x",               lambda x: x * x),
    ("sin(x)*x",          lambda x: sin(x) * x),
    ("exp(x)/(1+x*x)",    lambda x: exp(x) / (R(1.0) + x * x)),
    ("sin(exp(sqrt(x)))", lambda x: sin(exp(sqrt(x)))),
    ("1/x",               lambda x: R(1.0) / x),
    ("x**5",              lambda x: x ** 5),
    ("ln(x)/x",           lambda x: ln(x) / x),
    ("cos(x)*sin(x)",     lambda x: cos(x) * sin(x)),
]

DEGENERATE = [
    ("written zero, added",        lambda x: x * x + R(0)),
    ("raw python 0.0",             lambda x: x * x + 0.0),
    ("bare int 0",                 lambda x: x * x + 0),
    ("Composite(0)",               lambda x: x * x + Composite(0)),
    ("cancellation, added",        lambda x: x * x + (x - x)),
    ("cancellation, subtracted",   lambda x: x * x - (x - x)),
    ("zero multiplied in",         lambda x: R(0) * x ** 3 + x * x),
    ("zero times a transcendental", lambda x: x * x + R(0) * sin(x)),
    ("zero in a denominator",      lambda x: x * x / (R(1) + R(0))),
    ("accumulator from R(0)",
     lambda x: (lambda t: [t := t + x * x / R(3) for _ in range(3)][-1])(R(0))),
    ("data value that is zero",    lambda x: R(1.0) * x * x + R(0.0)),
]

# Degenerate, and invisible to the COUNT, because ZERO is built once at import
# so using it inside f is not a construction and nothing is counted there.
# Carrying sees them anyway: the constant was minted at import and kept its id,
# so it is a different source from the seed however often it is reused.
CARRIED_ONLY = [
    ("the ZERO constant",      lambda x: x * x + ZERO),
    ("tagging again inside f", lambda x: (x + ZERO) ** 2),
]

# THE KNOWN GAP.  A source is recognised where it is BORN, and these arrive
# already graded, having passed none of the three births.  Recognising them
# would mean scanning the coefficients of every composite ever constructed, on
# every construction, to catch a spelling nobody reaches for: a written zero is
# how a second source is spelled in practice, and that is caught.
KNOWN_GAP = [
    ("written out directly",   lambda x: x * x + Composite({-1: 1.0})),
    ("1/INF",                  lambda x: x * x + R(1.0) / INF),
]


@pytest.mark.parametrize("label,f", CLEAN, ids=[c[0] for c in CLEAN])
def test_ordinary_work_is_not_flagged(label, f):
    """No false positives.  Nested transcendentals build long infinitesimal
    towers, but they are all powers of ONE seed, so no source was added."""
    d = taylor_degeneracy(f, 2.0)
    assert not d.degenerate, f"{label} flagged with {d.sources} sources"
    assert d.sources == 1, "the seed is the one infinitesimal that entered"


@pytest.mark.parametrize("label,f", DEGENERATE, ids=[c[0] for c in DEGENERATE])
def test_every_spelling_of_a_second_zero_is_flagged(label, f):
    d = taylor_degeneracy(f, 2.0)
    assert d.degenerate, f"{label} not flagged"
    assert d.sources == 2, f"{label}: expected seed plus one, got {d.sources}"


@pytest.mark.parametrize("label,f", CARRIED_ONLY, ids=[c[0] for c in CARRIED_ONLY])
def test_carrying_catches_what_the_count_cannot(label, f):
    """The count is blind to a source that was not born inside the watch.
    Measured: `x*x + ZERO` gives slope 5 where the caller expects 4, and the
    count calls it clean.  The id came in with the constant and settles it."""
    d = taylor_degeneracy(f, 2.0)
    assert d.sources == 1, f"{label}: the count should see only the seed"
    assert d.degenerate, f"{label}: carrying is the verdict"


@pytest.mark.parametrize("label,f", KNOWN_GAP, ids=[c[0] for c in KNOWN_GAP])
def test_the_known_gap_is_where_it_is_documented(label, f):
    """Pinned, not tolerated silently.  These ARE degenerate and read clean,
    because they arrive already graded and no birth was observed.  If a later
    change starts catching them this test fails, and that is the good outcome:
    delete the case rather than work around it."""
    assert not taylor_degeneracy(f, 2.0).degenerate, \
        f"{label} is now caught -- move it out of KNOWN_GAP"


@pytest.mark.parametrize("label,f", CARRIED_ONLY + KNOWN_GAP,
                         ids=[c[0] for c in CARRIED_ONLY + KNOWN_GAP])
def test_those_cases_really_are_degenerate(label, f):
    """Not taken on trust, including the gap: the conventional reading of these
    is wrong.  (x + ZERO)**2 is (2x-2)**2, slope 8 at x=2, not 4."""
    assert abs(all_derivatives(f, 2.0, up_to=1)[1] - 4.0) > 1e-9


@pytest.mark.parametrize("label,f", DEGENERATE, ids=[c[0] for c in DEGENERATE])
def test_the_flag_agrees_with_what_conventional_changes(label, f):
    """The flag's claim is testable: if it says the conventional reading does not
    hold, then re-evaluating under conventional() must change the answer to the
    textbook one.  If it says nothing entered, the two must agree already."""
    plain = all_derivatives(f, 2.0, up_to=3)
    with conventional():
        conv = all_derivatives(f, 2.0, up_to=3)
    assert all(abs(a - b) < 1e-9 for a, b in zip(conv, SQ)), \
        f"{label}: conventional() gave {conv}, want {SQ}"
    assert plain != conv or label == "cancellation, subtracted", \
        f"{label}: flagged but the readings agree"


@pytest.mark.parametrize("label,f", CLEAN, ids=[c[0] for c in CLEAN])
def test_an_unflagged_formula_reads_the_same_either_way(label, f):
    plain = all_derivatives(f, 2.0, up_to=3)
    with conventional():
        conv = all_derivatives(f, 2.0, up_to=3)
    assert all(abs(a - b) < 1e-12 for a, b in zip(plain, conv)), \
        f"{label}: {plain} vs {conv}"


def test_conventional_mode_does_not_flag_its_own_inert_zeros():
    """Under conventional() a written zero is a zero TERM and the derivatives are
    the textbook ones, so counting it would report degeneracy in the one mode
    that has none.  The increment sits after the mode check for this reason."""
    for _, f in DEGENERATE:
        with conventional():
            d = taylor_degeneracy(f, 2.0)
            assert d.sources == 1 and not d.degenerate, d.sources


def test_it_catches_cases_forensics_reports_as_stable():
    """The justification for a separate flag.  forensics audits float behaviour
    and says `conventional derivative corrupted` for some of these; for the rest it says
    STABLE, correctly, because nothing unstable happened -- the arithmetic is
    exact and the reading is what moved."""
    from composite.forensics import audit, STABLE
    missed = []
    for label, f in DEGENERATE:
        flagged = taylor_degeneracy(f, 2.0).degenerate
        try:
            stable = audit(f, 2.0).verdict is STABLE
        except Exception:
            stable = False
        if flagged and stable:
            missed.append(label)
    assert missed, "expected some degenerate cases that forensics calls stable"
    assert len(missed) >= 4, f"only {len(missed)}: {missed}"


def test_the_watch_counts_sources_not_terms():
    """A composite can carry a hundred infinitesimal terms and be ordinary,
    because they are powers of one seed.  What breaks the reading is a second
    ORIGIN."""
    x = cl._seeded(2.0)
    with degeneracy_watch() as count:
        y = exp(x) * sin(x) / (R(1.0) + x * x)
    assert len(y.coeffs_dict()) > 5, "expected a long infinitesimal tower"
    assert count() == 0, "a tower of one seed is not a second source"
    # and the same with the seed inside the watch: one, not many
    with degeneracy_watch() as c2:
        s = cl._seeded(2.0)
        _ = exp(s) * sin(s) / (R(1.0) + s * s)
    assert c2() == 1


def test_the_watch_nests():
    x = cl._seeded(2.0)
    with degeneracy_watch() as outer:
        _ = x * x + R(0)
        with degeneracy_watch() as inner:
            _ = x * x + R(0)
            assert inner() == 1
        assert inner() == 1
        assert outer() == 2, "an inner watch must not consume the outer's count"


def test_degeneracy_reports_itself_in_words():
    clean = taylor_degeneracy(lambda x: x * x, 2.0)
    dirty = taylor_degeneracy(lambda x: x * x + R(0), 2.0)
    assert not clean and bool(dirty)
    assert "one infinitesimal entered" in str(clean)
    assert "AS WRITTEN" in str(dirty) and "conventional()" in str(dirty)
    assert "degenerate=True" in repr(dirty)


def test_explain_surfaces_it_without_being_asked():
    """The point of putting it there: a caller reading explain() output finds out
    that the slope belongs to a different function, which is the one thing they
    could not otherwise recover."""
    e = explain(lambda x: R(0) * x ** 3 + x * x, 2.0)
    assert e.degeneracy is not None and e.degeneracy.degenerate
    assert "as written" in str(e)
    assert "slope 12" in str(e)               # correct for (x-2)*x**3 + x**2
    clean = explain(lambda x: x * x, 2.0)
    assert not clean.degeneracy.degenerate
    assert "as written" not in str(clean)


def test_a_refused_evaluation_still_returns_an_explanation():
    e = explain(lambda x: R(1.0) / (x - x - x + x), 2.0)
    assert e.kind in ("refused", "value", "unbounded", "nothing")


def test_the_seed_counts_once_at_either_end():
    assert taylor_degeneracy(lambda x: x, 2.0).sources == 1
    assert taylor_degeneracy(lambda x: x, 0.0).sources == 1     # seeded at the origin
    assert not taylor_degeneracy(lambda x: x, 0.0).degenerate


# --- tagging is not injecting --------------------------------------------

# Self-contained nested work only.  Tagging again INSIDE f is not here: it adds
# a second infinitesimal and is in PROBE_ONLY.  A seed the caller keeps is in
# test_a_seed_the_caller_keeps_does_count.
TAGGING = [
    ("f differentiates inside at 3",
     lambda x: x * R(cl.derivative(sin, 3.0))),
    ("f differentiates inside at 0",
     lambda x: x * R(cl.derivative(sin, 0.0))),
    ("f integrates inside",        lambda x: x * R(cl.integrate(sin, 0.0, 1.0))),
    ("nested all_derivatives at 0",
     lambda x: x * R(all_derivatives(sin, 0.0, up_to=3)[3])),
]


@pytest.mark.parametrize("label,f", TAGGING, ids=[c[0] for c in TAGGING])
def test_a_self_contained_nested_extraction_is_not_flagged(label, f):
    """Seeding is the extraction's own infinitesimal, not one the formula
    introduced, so it must not count.

    This failed.  `_seeded(0)` returned `R(0)`, which is `Composite.zero()`, so
    seeding AT THE ORIGIN was indistinguishable from injecting a zero and every
    nested differentiation at 0 came back degenerate with nothing injected.
    `_seeded` now builds the unit directly at that end.  The non-zero end was
    never affected, because ZERO is a module constant and costs nothing.
    """
    d = taylor_degeneracy(f, 2.0)
    assert not d.degenerate, f"{label} flagged with {d.sources} sources"


def test_the_seed_is_the_first_source_at_either_end():
    """It counts, and being first it is free.  Both ends must agree, or seeding
    at the origin would differ from seeding anywhere else."""
    for at in (2.0, 0.0, -3.5):
        with degeneracy_watch() as count:
            s = cl._seeded(at)
        assert count() == 1, f"_seeded({at}) counted {count()}"
        assert abs((s * s).d(1) - 2 * at) < 1e-12, f"_seeded({at}) is not a unit seed"


def test_a_nested_extraction_is_transparent():
    """An extraction builds a seed, reads a float off it and throws the composite
    away, so that seed never enters the caller's expression.  Counting it made
    every nested differentiation look like a second infinitesimal."""
    for label, f in (("derivative at 0",  lambda x: x * R(cl.derivative(sin, 0.0))),
                     ("derivative at 3",  lambda x: x * R(cl.derivative(sin, 3.0))),
                     ("all_derivatives",  lambda x: x * R(all_derivatives(sin, 0.0, up_to=3)[3])),
                     ("integrate",        lambda x: x * R(cl.integrate(sin, 0.0, 1.0)))):
        d = taylor_degeneracy(f, 2.0)
        assert not d.degenerate, f"{label}: {d.sources} sources"
        assert d.sources == 1, f"{label}: {d.sources}"


def test_a_seed_the_caller_keeps_does_count():
    """The other side of the same line.  _seeded() whose result goes INTO the
    expression is a second infinitesimal, and `x * _seeded(3.0)` is
    (x)(x+1) rather than 3x."""
    d = taylor_degeneracy(lambda x: x * cl._seeded(3.0), 2.0)
    assert d.degenerate and d.sources == 2


def test_the_rule_needs_no_seed_at_all():
    """Counted from zero, so the first zero written is the one read against."""
    from composite.composite_lib import Degeneracy
    with degeneracy_watch() as c:
        _ = R(5.0) + R(0)
    assert not Degeneracy(c()).degenerate and c() == 1
    with degeneracy_watch() as c:
        _ = R(5.0) + R(0) + R(0)
    assert Degeneracy(c()).degenerate and c() == 2


def test_you_cannot_tag_inside_f_by_either_spelling():
    """`R(2) + ZERO` and `R(1) + R(0)` are the same shape, so nothing separates
    seeding from injecting, and adding EITHER inside f is a second
    infinitesimal.  Seeding is the extraction's job, outside f."""
    for f in (lambda x: (x + R(0)) ** 2, lambda x: (x + ZERO) ** 2):
        assert taylor_degeneracy(f, 2.0).degenerate


# --- the flag rides on the number ----------------------------------------

def test_the_flag_is_readable_on_the_number_itself():
    """The reason for carrying it rather than probing: it answers for a value
    you were handed, with no function left to re-evaluate."""
    with track_degeneracy():
        x = cl._seeded(2.0)
        assert (x * x).conventionally_degenerate is False
        assert (x * x + R(0)).conventionally_degenerate is True
        # and it survives being passed on, because every operation carries it
        y = x * x + R(0)
        assert (sin(y) / R(3.0) - R(1.0)).conventionally_degenerate is True


def test_a_raw_int_operand_is_not_a_source():
    """Regression.  The first version folded minted ids in with the operands and
    told them apart by type, so `x ** 5` read the exponent as source number 5
    and every clean formula with an int in it flagged."""
    with track_degeneracy():
        x = cl._seeded(2.0)
        for y in (x ** 5, x * 3, 3 * x, x + 7, x - 2, x / 4, x ** 2 - 3 * x + 1):
            assert y.conventionally_degenerate is False, repr(y)


def test_tracking_is_off_by_default_and_says_so_rather_than_saying_clean():
    """Off costs nothing, and the cost of that is that it knows nothing.  A
    number built while it was off must read None, NEVER False: reporting a
    silent "clean" for a number nobody was watching is the exact failure this
    flag exists to prevent."""
    assert degeneracy_tracking() is False, "tracking should default to off"
    x = cl._seeded(2.0)
    assert (x * x + R(0)).conventionally_degenerate is None
    with track_degeneracy():
        y = cl._seeded(2.0)
        assert (y * y + R(0)).conventionally_degenerate is True
    assert degeneracy_tracking() is False, "the block must restore it"


def test_switching_off_removes_the_wrapper_rather_than_branching_inside_it():
    """The switch is what makes off FREE rather than cheap.  If this ever starts
    failing, off is paying for a wrapper frame on every operation again."""
    set_degeneracy_tracking(False)
    assert not hasattr(Composite.__mul__, "__wrapped__")
    with track_degeneracy():
        assert hasattr(Composite.__mul__, "__wrapped__")
    assert not hasattr(Composite.__mul__, "__wrapped__")


def test_the_reporting_entry_points_do_not_depend_on_the_switch():
    """taylor_degeneracy and explain exist to answer this question, so they turn
    tracking on themselves.  Switching it off globally must not turn their
    answer into a quiet no."""
    set_degeneracy_tracking(False)
    assert taylor_degeneracy(lambda x: x * x + R(0), 2.0).degenerate
    assert explain(lambda x: R(0) * x ** 3 + x * x, 2.0).degeneracy.degenerate
    assert degeneracy_tracking() is False, "and they must put it back"


def test_the_switch_restores_arithmetic_exactly():
    """Turning it on and off must not perturb a single coefficient."""
    before = (cl._seeded(2.0) ** 3 + R(0.5)).coeffs_dict()
    with track_degeneracy():
        during = (cl._seeded(2.0) ** 3 + R(0.5)).coeffs_dict()
    after = (cl._seeded(2.0) ** 3 + R(0.5)).coeffs_dict()
    assert before == during == after, (before, during, after)


def test_one_source_reused_is_still_one_source():
    """The case a bare boolean cannot do: both operands carry an infinitesimal
    and it is the SAME one, so nothing degenerated."""
    with track_degeneracy():
        x = cl._seeded(2.0)
        for y in (x * x + x, x / x, x - x * x, (x + x) * (x * x)):
            assert y.conventionally_degenerate is False, repr(y)
