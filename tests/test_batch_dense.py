"""batch_dense backend: one composite holding many numbers, checked row by row.

The reference for every row is the same computation on sparse_dense, one number
at a time: the batch must give each row exactly what that row gives alone, or
refuse.  Relative errors are |got - ref| / max(1, |ref|) with fixed tolerances;
every check prints got against reference.
"""
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import composite.composite_lib as cl
from composite.composite_lib import (Composite, R, exp, ln, sqrt, sin, cos, tan, atan,
                                     asin, acos, sinh, cosh, tanh, power, normal_cdf)
from composite.backends import config
from composite.backends.batch_dense_backend import Lanes, BatchDecisionError
from composite.partials import partials

sys.path.insert(0, str(Path(__file__).parent))
import brute_derivatives as bd                     # noqa: E402  (the expression generator)

warnings.simplefilter("ignore")


@pytest.fixture(autouse=True)
def restore_backend():
    yield
    config.use_sparse_dense()
    cl._refresh_constants()
    cl.set_max_order(None)


def use_batch(n):
    config.use_batch_dense(n)
    cl._refresh_constants()


def use_sparse():
    config.use_sparse_dense()
    cl._refresh_constants()


def batch_seed(pts):
    """R(a) + ZERO over a batch: R() is how batch data enters."""
    return R(Lanes(np.asarray(pts, dtype=float))) + cl.ZERO


def read(c, order):
    """Orders 0..order of a composite: floats, or one array per order for a batch."""
    out = []
    for n in range(order + 1):
        v = c.st() if n == 0 else c.d(n)
        out.append(np.asarray(v.v if isinstance(v, Lanes) else v, dtype=float))
    return out


def rows_of(c, order, n_rows):
    return np.array([np.broadcast_to(v, (n_rows,)) for v in read(c, order)]).T


def per_row(f, pts, order):
    use_sparse()
    return np.array([[float(v) for v in read(f(R(float(a)) + cl.ZERO), order)] for a in pts])


def rel(got, ref):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    return np.where((got == ref) | (np.isnan(got) & np.isnan(ref)), 0.0,
                    np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))


def check(label, got, ref, tol):
    r = rel(got, ref)
    w = np.unravel_index(int(np.argmax(r)), r.shape) if r.ndim else ()
    print(f"  {label:<34} worst rel {float(np.max(r)):.1e} at {w}: got {np.asarray(got)[w]!r} "
          f"ref {np.asarray(ref)[w]!r}  (tol {tol:g})")
    assert float(np.max(r)) <= tol, f"{label}: rel {float(np.max(r)):.2e} > {tol:g}"


# --- every elementary function, row by row ------------------------------------

PTS = [0.3, 0.7, 1.1, 1.6]
FUNCS = {
    "sin": sin, "cos": cos, "tan": tan, "atan": atan,
    "asin(x/2)": lambda x: asin(x / 2.0), "acos(x/2)": lambda x: acos(x / 2.0),
    "sinh": sinh, "cosh": cosh, "tanh": tanh, "exp": exp, "ln": ln, "sqrt": sqrt,
    "power(x, 1.5)": lambda x: power(x, 1.5), "x ** -0.5": lambda x: x ** -0.5,
    "power(2, x)": lambda x: power(R(2.0), x), "power(x, x)": lambda x: power(x, x),
    "normal_cdf": normal_cdf, "x / (1 + x*x)": lambda x: x / (R(1.0) + x * x),
}


@pytest.mark.parametrize("name", list(FUNCS))
def test_each_function_matches_sparse_per_row(name):
    f = FUNCS[name]
    ref = per_row(f, PTS, 6)
    use_batch(len(PTS))
    got = rows_of(f(batch_seed(PTS)), 6, len(PTS))
    print(f"\n{name} at {PTS}, orders 0..6")
    check(name, got, ref, 1e-12)


# --- random expressions: match every row, or refuse ---------------------------

ROW_PTS = [0.35, 0.7, 1.1, 1.6, 2.3, 2.9]
#: The kinds of refusal the batch may give where a row alone succeeds.
#: Log-axis grades (ln of a cancellation) have no batch form; a row-dependent
#: decision is refused by design.
ALLOWED_REFUSALS = ("float() of a batch coefficient", "different answers in different batch rows")


def build(t, x):
    """The battery's expression tree, composite side only."""
    k = t[0]
    if k == "x":
        return x
    if k == "c":
        return R(t[1])
    if k == "u":
        return bd.UN[t[1]][0](build(t[2], x))
    if k == "ipow":
        return build(t[1], x) ** t[2]
    if k == "rpow":
        return power(build(t[1], x), t[2])
    a, b = build(t[1], x), build(t[2], x)
    if k == "cpow":
        return power(a, b)
    return {"+": a.__add__, "-": a.__sub__, "*": a.__mul__, "/": a.__truediv__}[k](b)


def _complete(c):
    o = c.complete_order
    return 6 if o is None else min(6, int(o))


@pytest.mark.parametrize("zeros", [False, True], ids=["classical", "zero-injected"])
def test_random_expressions_match_sparse_per_row_or_refuse(zeros):
    random.seed(101 if not zeros else 102)
    cl.set_max_order(6)
    stats = dict(compared=0, orders=0, both_refuse=0, batch_refused=0)
    refusals, mismatches, worst = {}, [], 0.0
    while stats["compared"] < 150:
        t = bd.gen(random.choice([2, 3, 4]), zeros)
        expr = bd.show(t)
        use_sparse()
        alone, ok = [], True
        for a in ROW_PTS:
            try:
                with cl._refusing_float():
                    c = build(t, R(a) + cl.ZERO)
                alone.append((read(c, 6), _complete(c)))
            except Exception:
                ok = False
                break
        use_batch(len(ROW_PTS))
        try:
            with cl._refusing_float():
                cb = build(t, batch_seed(ROW_PTS))
            got = rows_of(cb, 6, len(ROW_PTS))
            batch_ok = True
        except Exception as e:
            batch_ok, msg = False, f"{type(e).__name__}: {e}"
        if not ok:
            if batch_ok:
                mismatches.append((expr, "a row alone raised, the batch did not"))
            else:
                stats["both_refuse"] += 1
            continue
        if not batch_ok:
            assert any(s in msg for s in ALLOWED_REFUSALS), f"{expr}: unexpected refusal {msg}"
            stats["batch_refused"] += 1
            key = next(s for s in ALLOWED_REFUSALS if s in msg)
            refusals[key] = refusals.get(key, 0) + 1
            continue
        stats["compared"] += 1
        top = min([cb_ for _, cb_ in alone] + [_complete(cb)])
        for i, (vals, _) in enumerate(alone):
            for n in range(top + 1):
                r = float(rel(got[i, n], vals[n]))
                stats["orders"] += 1
                worst = max(worst, r) if math.isfinite(r) else worst
                if not r <= 1e-9:
                    mismatches.append((expr, ROW_PTS[i], n, got[i, n], float(vals[n])))
    print(f"\n{'zero-injected' if zeros else 'classical'}: {stats}, refusals {refusals}, "
          f"worst rel {worst:.1e} (tol 1e-9)")
    for m in mismatches[:10]:
        print("  MISMATCH", m)
    assert not mismatches


# --- batch data is a value: its zeros behave as under TAG ----------------------

def tag_per_row(f, pts, order):
    """Each row alone, seeded the way TAG seeds it: a + h, the zero of a row at
    0 kept as a value rather than converted."""
    use_sparse()
    out = []
    for a in pts:
        x = cl.TAG(Composite({0: float(a)}))
        try:
            out.append([float(v) for v in read(f(x), order)])
        finally:
            cl.UNTAG()
    return np.array(out)


@pytest.mark.parametrize("pts", [[0.0, 0.5, 0.0, 1.2], [0.0, 0.0, 0.0, 0.0]],
                         ids=["zero in some rows", "zero in every row"])
def test_batch_data_at_zero_seeds_as_tag_does(pts):
    """R(a) + ZERO over a batch is TAG's seed <a_0 1_-1> in every row: a zero
    handed in as data is a value, whether some rows hold it or all of them."""
    f = lambda x: exp(x) * sin(x) + x * x
    ref = tag_per_row(f, pts, 3)
    use_batch(len(pts))
    x = batch_seed(pts)
    print(f"\nbatch seed {x}")
    got = rows_of(f(x), 3, len(pts))
    print("row 0 under TAG:", ref[0].tolist(), " in the batch:", got[0].tolist())
    check("batch data at 0 vs TAG per row", got, ref, 1e-12)


# --- partials on a batch: inputs zero, or equal, in some rows -------------------

def test_partials_with_an_input_zero_and_inputs_equal_in_some_rows():
    f = lambda x, y, z: exp(x * y) * sin(z + 0.5) + x * y ** 2 * z ** 3 + ln(R(1.0) + z) * x
    n = 40
    rng = np.random.default_rng(5)
    P = {"x": rng.uniform(0.2, 1.5, n), "y": rng.uniform(0.2, 1.5, n), "z": rng.uniform(0.2, 1.5, n)}
    P["z"][::4] = 0.0                     # z is zero in a quarter of the rows
    P["y"][1::5] = P["x"][1::5]           # x and y are equal in a fifth
    req = {"fx": {"x": 1}, "fzz": {"z": 2}, "fz3": {"z": 3}, "fxy": {"x": 1, "y": 1},
           "fxz": {"x": 1, "z": 1}, "fxxy": {"x": 2, "y": 1}, "fyzz": {"y": 1, "z": 2},
           "fxyz": {"x": 1, "y": 1, "z": 1}}
    use_sparse()
    ref = [partials(f, {k: float(v[i]) for k, v in P.items()}, req) for i in range(n)]
    use_batch(n)
    got = partials(f, {k: Lanes(v) for k, v in P.items()}, req)
    print(f"\n{n} rows, z = 0 in {int((P['z'] == 0).sum())}, x = y in {int((P['x'] == P['y']).sum())}")
    for k in req:
        check(k, got[k].v, [r[k] for r in ref], 1e-11)


# --- refusals ---------------------------------------------------------------------

def test_a_batch_never_collapses_to_one_float():
    with pytest.raises(TypeError):
        float(Lanes(np.array([1.0, 2.0])))


def test_an_ordering_that_differs_between_rows_is_refused():
    with pytest.raises(BatchDecisionError):
        Lanes(np.array([1.0, -1.0])) < 0
    assert (Lanes(np.array([1.0, 2.0])) > 0) is True
    assert (Lanes(np.array([-1.0, -2.0])) > 0) is False


def test_an_equality_between_values_that_differs_between_rows_is_refused():
    with pytest.raises(BatchDecisionError):
        Lanes(np.array([1.0, 2.0])) == Lanes(np.array([1.0, 3.0]))
    assert (Lanes(np.array([1.0, 2.0])) == Lanes(np.array([1.0, 2.0]))) is True


def test_a_domain_decision_that_differs_between_rows_is_refused():
    use_batch(3)
    with pytest.raises(BatchDecisionError):
        sqrt(batch_seed([1.0, -1.0, 2.0]))
    with pytest.raises(BatchDecisionError):
        asin(batch_seed([0.5, 1.5, 0.2]))


def test_overflow_in_one_row_raises_as_math_does():
    use_batch(3)
    with pytest.raises(OverflowError):
        exp(batch_seed([1.0, 800.0, 2.0]))


# --- conventions -------------------------------------------------------------------

def test_a_term_exists_if_any_row_is_nonzero():
    """== 0 is true only when every row is zero; bool() is true when any row is
    nonzero.  -0.0 is zero and NaN is nonzero, as for a single float."""
    assert (Lanes(np.array([0.0, -0.0])) == 0) is True
    assert (Lanes(np.array([0.0, 1.0])) == 0) is False
    assert bool(Lanes(np.array([0.0, 0.0]))) is False
    assert bool(Lanes(np.array([0.0, 2.0]))) is True
    assert bool(Lanes(np.array([0.0, np.nan]))) is True


def test_a_repeated_grade_is_summed():
    use_batch(3)
    be = config.get_backend()
    a, b, c = Lanes(np.array([1.0, 2.0, 3.0])), Lanes(np.array([10.0, 20.0, 30.0])), Lanes(np.array([5.0, 5.0, 5.0]))
    data = be.create_from_terms(np.array([-1.0, -1.0, 0.0]), [a, b, c])
    got = be.read_dim(data, -1).v
    print("\nrepeated grade -1:", got.tolist(), " expected", [11.0, 22.0, 33.0])
    assert got.tolist() == [11.0, 22.0, 33.0]
    assert be.read_dim(data, 0).v.tolist() == [5.0, 5.0, 5.0]


# --- run_batch: set aside the rows that fail, run the rest again ----------------

from composite.backends.batch_dense_backend import run_batch    # noqa: E402


def jet_fn(f, order=3):
    """A batch function: orders 0..order of f at the batch of points x."""
    def fn(x):
        c = f(R(x) + cl.ZERO)
        return {f"d{n}": (c.st() if n == 0 else c.d(n)) for n in range(order + 1)}
    return fn


def alone(f, a, order=3):
    """The same row computed alone on sparse_dense, or the exception it raises."""
    use_sparse()
    try:
        c = f(R(float(a)) + cl.ZERO)
        return [float(c.st())] + [float(c.d(n)) for n in range(1, order + 1)]
    except (ValueError, ArithmeticError) as e:
        return e


def check_rows(res, f, xs, order=3):
    """Every row: the value it has alone, or an error where alone it raises."""
    ok = bad = 0
    for i, a in enumerate(xs):
        want = alone(f, a, order)
        if isinstance(want, Exception):
            assert i in res.errors, f"row {i} (x={a}) raises alone but has no error recorded"
            assert all(np.isnan(res.values[f'd{n}'][i]) for n in range(order + 1))
            bad += 1
        else:
            assert i not in res.errors, f"row {i} (x={a}) computes alone but failed: {res.errors.get(i)}"
            got = [res.values[f"d{n}"][i] for n in range(order + 1)]
            r = float(np.max(rel(got, want)))
            assert r <= 1e-12, f"row {i} (x={a}): rel {r:.1e}"
            ok += 1
    return ok, bad


def test_run_batch_on_a_clean_batch_is_one_run():
    f = lambda x: exp(x) * sin(x)
    xs = np.linspace(0.1, 2.0, 50)
    res = run_batch(jet_fn(f), {"x": xs})
    ok, bad = check_rows(res, f, xs)
    print(f"\nclean: {res}, rows matching alone {ok}, failed {bad}")
    assert res.runs == 1 and not res.errors and ok == 50


def test_run_batch_sets_aside_rows_that_overflow():
    f = lambda x: exp(x * x)
    xs = np.linspace(0.1, 2.0, 50)
    xs[[3, 17, 40]] = 40.0                               # exp(1600) overflows
    res = run_batch(jet_fn(f), {"x": xs})
    ok, bad = check_rows(res, f, xs)
    print(f"\noverflow in 3 rows: {res}, rows matching alone {ok}, failed {bad}")
    for i in sorted(res.errors):
        print(f"  row {i}: {res.errors[i]}")
    assert sorted(res.errors) == [3, 17, 40] and res.runs == 2
    assert all(m.startswith("OverflowError") for m in res.errors.values())


def test_run_batch_runs_the_rows_a_decision_set_aside_as_their_own_batch():
    """fn branches on its input, so the rows take two paths.  The rows set aside
    are not errors: they are run as their own batch and all of them compute."""
    def f_row(x0):
        return (lambda x: exp(x)) if x0 > 1.0 else (lambda x: sin(x) * x)

    def fn(x):
        branch = f_row(x)                                # a Lanes comparison
        c = branch(R(x) + cl.ZERO)
        return {f"d{n}": (c.st() if n == 0 else c.d(n)) for n in range(4)}

    xs = np.linspace(0.1, 1.4, 40)                       # 12 rows above 1, 28 below
    res = run_batch(fn, {"x": xs})
    ok = 0
    for i, a in enumerate(xs):
        want = alone(f_row(a), a)
        got = [res.values[f"d{n}"][i] for n in range(4)]
        assert float(np.max(rel(got, want))) <= 1e-12
        ok += 1
    print(f"\ntwo paths: {res}, rows matching alone {ok}")
    assert not res.errors and res.runs == 3              # mixed, the majority, the minority


@pytest.mark.parametrize("name,f,bad_x", [
    ("sqrt, negative in some rows", lambda x: sqrt(x) * x, -0.5),
    ("asin, |x| > 1 in some rows", lambda x: asin(x) + x, 1.5),
])
def test_run_batch_reports_rows_outside_the_domain(name, f, bad_x):
    xs = np.linspace(0.1, 0.9, 30)
    xs[[2, 11, 25]] = bad_x
    res = run_batch(jet_fn(f), {"x": xs})
    ok, bad = check_rows(res, f, xs)
    print(f"\n{name}: {res}, rows matching alone {ok}, failed {bad}")
    for i in sorted(res.errors):
        print(f"  row {i}: {res.errors[i]}")
    assert sorted(res.errors) == [2, 11, 25] and ok == 27


def test_run_batch_strict_raises_the_first_error():
    xs = np.linspace(0.1, 2.0, 10)
    xs[4] = 40.0
    with pytest.raises(OverflowError):
        run_batch(jet_fn(lambda x: exp(x * x)), {"x": xs}, strict=True)


def test_run_batch_stops_at_max_runs_and_says_so():
    xs = np.linspace(0.1, 2.0, 10)
    xs[4] = 40.0
    res = run_batch(jet_fn(lambda x: exp(x * x)), {"x": xs}, max_runs=1)
    print(f"\nmax_runs=1: {res}: {res.errors}")
    assert res.runs == 1
    assert res.errors[4].startswith("OverflowError")
    assert all("max_runs" in res.errors[i] for i in range(10) if i != 4)


def test_run_batch_restores_the_backend():
    use_sparse()
    before = type(config.get_backend())
    run_batch(jet_fn(lambda x: exp(x)), {"x": np.linspace(0.1, 1.0, 5)})
    assert type(config.get_backend()) is before


def test_batch_errors_name_their_rows():
    use_batch(5)
    with pytest.raises(BatchDecisionError) as e:
        Lanes(np.array([1.0, 1.0, -1.0, 1.0, 1.0])) > 0
    print("\nBatchDecisionError rows:", e.value.batch_rows.tolist())
    assert e.value.batch_rows.tolist() == [False, False, True, False, False]   # the minority
    with pytest.raises(OverflowError) as e:
        exp(R(Lanes(np.array([1.0, 900.0, 2.0, 3.0, 4.0]))))
    print("OverflowError rows:", e.value.batch_rows.tolist())
    assert e.value.batch_rows.tolist() == [False, True, False, False, False]


def test_run_batch_around_partials_with_bad_rows():
    """The production shape: partials over many rows, a few of them outside the
    domain.  Every good row equals partials on that row alone."""
    f = lambda x, y: ln(x) * exp(x * y) + sqrt(x * y) * y
    req = {"fx": {"x": 1}, "fyy": {"y": 2}, "fxy": {"x": 1, "y": 1}, "fxxy": {"x": 2, "y": 1}}
    n = 200
    rng = np.random.default_rng(9)
    P = {"x": rng.uniform(0.2, 2.0, n), "y": rng.uniform(0.2, 2.0, n)}
    P["x"][[5, 60, 61, 150]] = -0.7                  # ln and sqrt of a negative
    res = run_batch(lambda **a: partials(f, a, req), P)
    use_sparse()
    worst, ok = 0.0, 0
    for i in range(n):
        try:
            want = partials(f, {k: float(v[i]) for k, v in P.items()}, req)
        except (ValueError, ArithmeticError):
            assert i in res.errors, f"row {i} raises alone but has no error recorded"
            continue
        assert i not in res.errors, f"row {i} computes alone but failed: {res.errors.get(i)}"
        worst = max(worst, max(float(rel(res.values[k][i], want[k])) for k in req))
        ok += 1
    print(f"\npartials, 4 bad rows: {res}, rows matching alone {ok}, worst rel {worst:.1e}")
    for i in sorted(res.errors):
        print(f"  row {i}: {res.errors[i]}")
    assert sorted(res.errors) == [5, 60, 61, 150] and ok == n - 4 and worst <= 1e-12
