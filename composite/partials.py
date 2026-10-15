"""partials.py -- partial derivatives, pure and mixed, read from seeded runs.

    partial(f, at, {"x": 2, "y": 1})          d^3 f / dx^2 dy at `at`
    partials(f, at, {"name": orders, ...})    several, sharing their runs

`f` is any function written in composite operations, taking its inputs as
keywords (or positionally, with `names=`).  `at` holds the point as plain
numbers.  The result is the derivative as a float, in the units of f and its
inputs; how the runs were seeded never shows in it.

HOW.  A RUN is one evaluation of f with some inputs carrying a zero.  One run
holds every order of what it seeds, so a pure partial is one run read at a
grade.  A mixed partial is one JOINT run of its inputs minus the runs of every
smaller combination of them (inclusion-exclusion over records): what remains
starts at the target grade with only terms that move every input.  When several
of those share the grade (d3/dx2dy and d3/dxdy2 both reach grade -3), joint runs
with a different quantity of zero on an input separate them; the weights are
solved exactly in rationals and applied with composite arithmetic.  The only
finite-difference step is that last subtraction of records.

This is NOT composite_multivar.partial_derivative, which combines directional
evaluations with interpolation weights.  Nothing here interpolates.

SEEDS.  Relative: x = x0 * (1 + c h) = <x0_0 c*x0_-1>, so each input's zero is
measured in its own units and the subtractions cancel terms of comparable size.
An input at exactly 0 is seeded as the zero itself, <c_-1>.  Two inputs with the
same value never get the same quantity: their seeds would be the same number and
a difference like x - y would cancel exactly (an identity, which converts).
Where the inputs' own terms are more than BALANCE apart at the target grade, the
quantities are scaled so they arrive at the same size, which leaves a mixed term
at |f_xy| / sqrt(|f_xx f_yy|) of its coefficient -- a property of f.  Below
about 1e-13 of that, no seeding recovers it in float64.

ZEROS.  Every run blesses ONE infinitesimal (TAG) and builds every seed from it,
so a data zero meeting a seed (r - q at q = 0), ln(1.0) at an equal ratio, or a
cancellation, behaves classically: the other inputs are held fixed, which is
what a partial derivative means.  A run that still comes back denoted (f itself
cancels a seeded input against itself) is refused with PartialError.

DEPTH.  The order cap is set to the highest order requested for the call and
restored afterwards.  Runs are cached in a Runs object with the order they were
computed to; a run made shallower than a later request needs is recomputed.
"""
from fractions import Fraction
from itertools import combinations, product
from math import factorial, prod

import composite.composite_lib as _cl
from composite.composite_lib import Composite, R

#: Inputs' own terms within this factor of each other at the target grade:
#: relative quantities as they are.  Further apart: scale them to balance.
BALANCE = 1e3


class PartialError(ValueError):
    """A run came back as the jet of a denoted function, not of f."""


class Runs:
    """Seeded runs shared between calls: {(input, quantity), ...} -> composite.

    Pass the same Runs to several partial() calls at the same point and each
    run is made once.  A run remembers the order cap it was made under.
    """

    def __init__(self):
        self._runs = {}

    def get(self, key, order):
        hit = self._runs.get(key)
        if hit is None or hit[0] < order:
            return None
        return hit[1]

    def put(self, key, order, value):
        self._runs[key] = (order, value)

    def __len__(self):
        return len(self._runs)

    def keys(self):
        return self._runs.keys()


def _number(v):
    """A point value as f receives it: a plain number as it is; a batch of
    values (batch-dense backend) as the composite constant holding them."""
    if isinstance(v, (int, float, Composite)):
        return v
    return Composite({0: v})


def _is_batch(v):
    return v.__class__.__name__ == "Lanes"


def _unit(x0):
    """The unit a seed measures its input in: x0, or 1 at x0 = 0.  For a batch
    (batch-dense backend) decided per row: x0 where it is nonzero, 1 where 0."""
    if _is_batch(x0):
        from composite.backends.batch_dense_backend import _xp
        xp = _xp(x0.v)
        return x0.__class__(xp.where(x0.v == 0.0, 1.0, x0.v))
    return x0 if x0 != 0 else 1.0


def _same_seed(xi, xj, qi, qj):
    """Would two inputs get the identical seed (same value, same quantity)?
    For a batch: in ANY row -- one such row is enough for x - y to cancel."""
    if not (_is_batch(xi) or _is_batch(xj) or _is_batch(qi) or _is_batch(qj)):
        return xi == xj and qi == qj
    import numpy as np
    v = lambda z: np.asarray(getattr(getattr(z, "v", z), "val", getattr(z, "v", z)))
    return bool(np.any((v(xi) == v(xj)) & (v(qi) == v(qj))))


def _run(f, at, seeds, runs, order):
    """f with each (input, quantity) in `seeds` carrying that much zero."""
    key = tuple(sorted(seeds))
    if runs is not None:
        hit = runs.get(key, order)
        if hit is not None:
            return hit
    h = _cl.TAG(Composite({0: 0.0}))       # a composite zero: TAG(0) routes a plain 0
                                           # through the integrator's float warning
    try:
        args = {name: _number(v) for name, v in at.items()}
        for name, c in key:
            hc = h * _number(c)
            if _is_batch(at[name]):
                # per row: x0 + c*unit*h, the zero seed <c_-1> where x0 = 0
                # and the relative x0*(1 + c h) elsewhere, in one composite
                args[name] = args[name] + hc * _number(_unit(at[name]))
            else:
                args[name] = hc if at[name] == 0 else (R(1.0) + hc) * args[name]
        v = f(**args)
    finally:
        _cl.UNTAG()
    if not isinstance(v, Composite):
        raise TypeError(
            f"partial(): f returned {type(v).__name__}, not a composite -- it did "
            f"not compute with the seeded inputs (a math.* call, or a float cast, "
            f"drops them).")
    if v.denotation_order is not None:
        raise PartialError(
            f"partial(): the run seeding {dict(key)} came back denoted (order "
            f"{v.denotation_order}): f cancels a seeded input exactly against "
            f"itself, and that identity converts.  Its derivative there is not "
            f"the classical one.")
    if runs is not None:
        runs.put(key, order, v)
    return v


def _record(f, at, quantities, runs, order):
    """The joint run of every input in `quantities`, minus every smaller
    combination's run: only the terms that move every input remain."""
    names = list(quantities)
    k = len(names)
    total = Composite({})
    for size in range(1, k + 1):
        for subset in combinations(names, size):
            run = _run(f, at, [(x, quantities[x]) for x in subset], runs, order)
            total = total + run if (k - size) % 2 == 0 else total - run
    return total


def _batch_scales(sizes, n):
    """The same scaling for a batch (batch-dense backend), decided per row:
    if any row is unbalanced every row gets its own scale -- scaling is valid in
    a balanced row too -- else all stay 1, as for a single number.  Written in
    the batch's own array module, so a recorded run (batch_replay) keeps it."""
    import numpy as np
    from composite.backends.batch_dense_backend import Lanes, _xp
    ref = next(z.v for z in sizes if isinstance(z, Lanes))
    xp = _xp(ref)
    z = [z.v if isinstance(z, Lanes) else xp.full(ref.shape, float(z)) for z in sizes]
    live = xp.stack([xp.where(v > 0, v, xp.nan) for v in z])
    with np.errstate(all="ignore"):
        ratio = xp.nanmax(live, axis=0) / xp.nanmin(live, axis=0)
        if not Lanes((z[0] > 0) & (ratio > BALANCE)):
            return [1.0] * len(sizes)
        out = [1.0]
        for j in range(1, len(sizes)):
            ok = (z[0] > 0) & (z[j] > 0)
            safe = xp.where(ok, z[j], 1.0)
            out.append(Lanes(xp.where(ok, (z[0] / safe) ** (1.0 / n), 1.0)))
    return out


def _rank(rows):
    """Rank of a small rational matrix."""
    m = [list(r) for r in rows]
    rank = 0
    for c in range(len(m[0]) if m else 0):
        piv = next((i for i in range(rank, len(m)) if m[i][c] != 0), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][c] != 0:
                fct = m[i][c] / m[rank][c]
                m[i] = [a - fct * b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def _solve(A, b):
    """The square rational system A w = b, exactly."""
    n = len(A)
    m = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = next(i for i in range(c, n) if m[i][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        for i in range(n):
            if i != c and m[i][c] != 0:
                fct = m[i][c] / m[c][c]
                m[i] = [a - fct * bb for a, bb in zip(m[i], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _partial(f, at, orders, runs, order):
    names = [x for x, i in orders.items() if i > 0]
    if not names:
        raise ValueError("partial(): no order requested")
    for x in names:
        if x not in at:
            raise KeyError(f"partial(): {x!r} is not an input in `at`")
    target = tuple(orders[x] for x in names)
    n = sum(target)
    units = prod(_unit(at[x]) ** i for x, i in zip(names, target))

    if len(names) == 1:
        v = _run(f, at, [(names[0], 1)], runs, order)
        return v.coeff(-n) * factorial(n) / units

    k = len(names)
    # Each input's own term at the target grade, from its unit single run
    # (the run its pure partials use).  Balanced: quantities stay 1 and those
    # runs are shared.  Unbalanced: scale so they arrive at the same size; the
    # first input keeps 1.
    sizes = [abs(_run(f, at, [(x, 1)], runs, order).coeff(-n)) for x in names]
    s = [1.0] * k
    if any(hasattr(z, "v") for z in sizes):
        s = _batch_scales(sizes, n)
    else:
        live = [z for z in sizes if z > 0]
        if sizes[0] > 0 and live and max(live) / min(live) > BALANCE:
            for j in range(1, k):
                if sizes[j] > 0:
                    s[j] = (sizes[0] / sizes[j]) ** (1.0 / n)

    # every multi-index reaching grade -n with every input moving
    unknowns = [m for m in product(range(1, n - k + 2), repeat=k) if sum(m) == n]
    # quantity vectors: all ones, then the inputs of highest order raised first,
    # so derivatives that need the same extra runs share them
    by_order = sorted(range(k), key=lambda j: -target[j])
    candidates = [tuple(1 for _ in range(k))]
    for top in range(2, n + 2):
        for j in by_order:
            for c in product(range(1, top + 1), repeat=k):
                if max(c) == top and c[j] == top and c not in candidates:
                    candidates.append(c)
    chosen, rows = [], []
    for c in candidates:
        if any(_same_seed(at[names[i]], at[names[j]], c[i] * s[i], c[j] * s[j])
               for i in range(k) for j in range(i + 1, k)):
            continue                                   # identical seeds
        row = [prod(Fraction(cj) ** mj for cj, mj in zip(c, m)) for m in unknowns]
        if _rank(rows + [row]) > len(rows):
            chosen.append(c)
            rows.append(row)
        if len(rows) == len(unknowns):
            break
    A = [[rows[ci][mi] for ci in range(len(rows))] for mi in range(len(unknowns))]
    b = [Fraction(1) if m == target else Fraction(0) for m in unknowns]
    weights = _solve(A, b)

    total = Composite({})
    for c, w in zip(chosen, weights):
        if w == 0:
            continue
        rec = _record(f, at, {x: cj * sj for x, cj, sj in zip(names, c, s)}, runs, order)
        total = total + rec * float(w)
    return (total.coeff(-n) * prod(factorial(i) for i in target)
            / (units * prod(sj ** i for sj, i in zip(s, target))))


def _point(at, names):
    """The point as {input: value}: a dict, or a sequence in `names` order."""
    if isinstance(at, dict):
        return dict(at)
    if names is None:
        raise TypeError("partial(): `at` must be a dict unless names= is given")
    return dict(zip(names, at))


def _call(f, names):
    if names is None:
        return f
    return lambda **a: f(*(a[x] for x in names))


def _scoped(order, body):
    """Run body() with the order cap at `order` and no TAG mode inherited."""
    if _cl._TAGGED.get() is not None:
        raise RuntimeError(
            "partial(): already inside TAG() / single_infinitesimal(); partial() "
            "blesses its own infinitesimal per run and would replace yours.  Call "
            "it outside that mode.")
    old = _cl.get_max_order()
    _cl.set_max_order(order)
    try:
        return body()
    finally:
        _cl.set_max_order(old)


def partial(f, at, orders, runs=None, names=None):
    """One partial derivative of f at `at`.

    orders: {input: order}, e.g. {"x": 1} (df/dx), {"x": 2, "y": 1}.
    runs:   a Runs to share seeded runs with other calls at the same point.
    names:  the input order, when f takes its inputs positionally.
    """
    order = sum(i for i in orders.values() if i > 0)
    g = _call(f, names)
    at = _point(at, names)
    return _scoped(order, lambda: _partial(g, at, orders, runs, order))


def partials(f, at, requests, runs=None, names=None):
    """Several partial derivatives at one point, sharing their runs.

    requests: {label: orders}, e.g. {"fx": {"x": 1}, "fxy": {"x": 1, "y": 1}}.
    Returns {label: value}.
    """
    order = max(sum(i for i in o.values() if i > 0) for o in requests.values())
    runs = Runs() if runs is None else runs
    g = _call(f, names)
    at = _point(at, names)
    return _scoped(order, lambda: {label: _partial(g, at, o, runs, order)
                                   for label, o in requests.items()})
