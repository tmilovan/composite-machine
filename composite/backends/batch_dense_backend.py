# composite/backends/batch_dense_backend.py
# Composite Machine -- Batch Dense Backend (PROTOTYPE)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""One Composite holding a BATCH of numbers that share their grades.

Opt-in, for dense work over many inputs at once: every Greek of a portfolio, a
derivative over a grid of points.  Each coefficient is a vector over the batch
(a `Lanes`), so one composite operation does the whole batch in numpy.

    from composite.backends import config
    config.use_batch_dense(10_000)

STORAGE.  An offset, a step and a float64 array of shape [grades, batch]: dense
along the grades, like the dense-series backend.  Dense is right here because a
Taylor series has no gaps; the sparse backends stay right where the data is
sparse.

DECISIONS.  The library decides things per number -- is this coefficient zero,
is this operand wholly zero (R1), which grade leads.  A batch decides them once,
for every row together, through the comparisons of `Lanes`:

    v != 0      True if ANY row is nonzero      (a term exists)
    v == 0      True only if EVERY row is zero
    v < 0 ...   the same answer in every row, else BatchDecisionError

so the library's existing tests read as whole-batch decisions, and a decision
that would differ between rows is refused rather than taken for some of them.
`float()` of a Lanes raises: a batch never silently collapses to one number.

PROTOTYPE.  Covers the arithmetic and the transcendentals a Black-Scholes price
uses.  Not covered: per-row decisions (an input exactly 0 in some rows only),
vector (log-axis) and Fraction grades.
"""
import math
from typing import Tuple

import numpy as np

from .base_backend import CompositeBackend, DIM_DTYPE, dim_cast
from .dense_series_backend import _lattice, _common_step


def _jnp():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jnp


def _is_jax(a):
    return type(a).__module__.startswith("jax")


def _is_rec(a):
    return getattr(a, "_batch_array", False)


def _is_batch_array(a):
    """A batch of values held as an array: numpy, jax, or recorded (batch_replay)."""
    return isinstance(a, np.ndarray) or _is_jax(a) or _is_rec(a)


def _xp(a):
    """The array module an array belongs to: numpy, jax.numpy, or a recording."""
    if _is_rec(a):
        return a.xp
    return _jnp() if _is_jax(a) else np


def _concrete(a):
    """A row of decisions as a numpy array; recorded as a guard when recording."""
    return a.concrete() if _is_rec(a) else np.asarray(a)


def _any(a):
    return bool(_xp(a).any(a))


def _all(a):
    return bool(_xp(a).all(a))


class BatchDecisionError(ValueError):
    """A decision the library makes per number differs between batch rows.

    batch_rows: a boolean mask of the rows to set aside -- the minority side of
    the decision -- or None when it is not known."""

    def __init__(self, message, batch_rows=None):
        super().__init__(message)
        self.batch_rows = batch_rows


def _minority(r):
    """The smaller side of a per-row decision, as a boolean mask."""
    r = np.asarray(getattr(r, "val", r), dtype=bool)
    return r if r.sum() <= r.size - r.sum() else ~r


class Lanes:
    """One coefficient across the batch: a float64 vector with batch-wide
    comparisons.  Arithmetic is elementwise and returns Lanes."""
    __slots__ = ("v",)
    __array_ufunc__ = None           # numpy defers to these operators

    def __init__(self, v):
        self.v = v if (_is_jax(v) or _is_rec(v)) else np.asarray(v, dtype=np.float64)

    @staticmethod
    def _o(o):
        return o.v if isinstance(o, Lanes) else o

    def _bin(self, o, fn):
        if not (isinstance(o, (Lanes, np.ndarray, int, float, np.floating, np.integer))
                or _is_jax(o) or _is_rec(o)):
            return NotImplemented     # a Composite: wrap the batch as Composite({0: lanes})
        # an object array of coefficients (the library's per-term arrays):
        # apply term by term, keeping each a Lanes
        if isinstance(o, np.ndarray) and o.dtype == object:
            out = np.empty(o.shape, dtype=object)
            for i, x in enumerate(o.flat):
                out.flat[i] = Lanes(fn(Lanes._o(x)))
            return out
        return Lanes(fn(Lanes._o(o)))

    # arithmetic
    def __add__(self, o): return self._bin(o, lambda x: self.v + x)
    def __radd__(self, o): return self._bin(o, lambda x: x + self.v)
    def __sub__(self, o): return self._bin(o, lambda x: self.v - x)
    def __rsub__(self, o): return self._bin(o, lambda x: x - self.v)
    def __mul__(self, o): return self._bin(o, lambda x: self.v * x)
    def __rmul__(self, o): return self._bin(o, lambda x: x * self.v)
    def __truediv__(self, o): return self._bin(o, lambda x: self.v / x)
    def __rtruediv__(self, o): return self._bin(o, lambda x: x / self.v)
    def __pow__(self, o): return self._bin(o, lambda x: self.v ** x)
    def __neg__(self): return Lanes(-self.v)
    def __pos__(self): return self
    def __abs__(self): return Lanes(abs(self.v))

    # batch-wide decisions
    # == 0 and != 0 ask whether a TERM exists, and in dense storage a term
    # exists if it is nonzero in any row -- as bool() and is_wholly_zero read
    # it: "== 0" is True only when every row is zero.  Any other equality is a
    # decision like < and >: the same answer in every row, or
    # BatchDecisionError.  Answering "equal in every row" for a mixed batch
    # took the wrong branch silently in the rows that were equal.
    def __eq__(self, o):
        o = Lanes._o(o)
        if isinstance(o, (int, float)) and o == 0:
            # one pass, no temporary: any() reads -0.0 as zero and NaN as
            # nonzero, exactly as v == 0 does
            return not _any(self.v)
        return self._order(self.v == o)

    def __ne__(self, o): return not self.__eq__(o)
    def __bool__(self): return _any(self.v)      # nonzero in any row
    def __hash__(self):          # consistent with ==: equal in every row
        # (equal hashes compare with ==, which raises only for rows that differ
        # in part; a key with equal bytes is equal in every row)
        return hash(np.asarray(getattr(self.v, "val", self.v)).tobytes())

    def _order(self, r):
        if _all(r):
            return True
        if not _any(r):
            return False
        raise BatchDecisionError(
            "a comparison the library makes per number has different answers "
            "in different batch rows; split the batch or use a per-number backend",
            batch_rows=_minority(r))

    def __lt__(self, o): return self._order(self.v < Lanes._o(o))
    def __le__(self, o): return self._order(self.v <= Lanes._o(o))
    def __gt__(self, o): return self._order(self.v > Lanes._o(o))
    def __ge__(self, o): return self._order(self.v >= Lanes._o(o))

    def __float__(self):
        raise TypeError("float() of a batch coefficient: read .v for the values")

    def is_integer(self):
        return _all(self.v == _xp(self.v).round(self.v))

    def __repr__(self):
        v = np.asarray(getattr(self.v, "val", self.v))     # display only: no guard
        if v.size <= 4:
            return "Lanes(" + ", ".join(f"{x:.6g}" for x in v) + ")"
        return f"Lanes({v[0]:.6g}, {v[1]:.6g}, ..., {v[-1]:.6g}; n={v.size})"

    def __format__(self, spec):
        return repr(self)


def is_lanes(x):
    return isinstance(x, Lanes)


class BatchData:
    """dim of vals[k] is offset + k*step; vals is [grades, batch]."""
    __slots__ = ("offset", "step", "vals")

    def __init__(self, offset, step, vals):
        self.offset = float(offset)
        self.step = float(step)
        self.vals = vals

    def dims(self):
        return self.offset + self.step * np.arange(self.vals.shape[0], dtype=DIM_DTYPE)


class BatchDenseBackend(CompositeBackend):
    """[grades, batch] arrays; see the module docstring."""

    VECTOR_DIMS = False
    BATCH = True

    def __init__(self, size: int, max_span: int = 1 << 16, xp="numpy"):
        """xp: "numpy", "jax", or a recording module (batch_replay.RecNP)."""
        self.size = int(size)
        self.max_span = int(max_span)
        self.jax = xp != "numpy"            # functional (.at[]) updates
        self.xp = np if xp == "numpy" else _jnp() if xp == "jax" else xp

    # numpy updates in place; jax arrays are immutable and update with .at[]
    def _zeros(self, n):
        return self.xp.zeros((n, self.size))

    def _add_at(self, arr, sl, val):
        if self.jax:
            return arr.at[sl].add(val)
        arr[sl] += val
        return arr

    def _set_at(self, arr, sl, val):
        if self.jax:
            return arr.at[sl].set(val)
        arr[sl] = val
        return arr

    # ---- construction ------------------------------------------------------
    def _row(self, value):
        if isinstance(value, Lanes):
            if value.v.shape != (self.size,):
                raise ValueError(f"batch of {value.v.shape} in a backend of size {self.size}")
            return value.v
        if not _is_batch_array(value):
            c = np.asarray(value, dtype=np.float64)
            if c.ndim == 0:                  # a plain number: the same in every row
                return self.xp.full(self.size, float(c))
            value = c
        arr = value if (_is_jax(value) or _is_rec(value)) else self.xp.asarray(value)
        if arr.ndim == 0:
            return self.xp.full(self.size, float(arr))
        if arr.shape != (self.size,):
            raise ValueError(f"batch of {arr.shape} in a backend of size {self.size}")
        return arr

    def _make(self, offset, step, vals):
        return BatchData(offset, step, vals)

    def _empty(self):
        return BatchData(0.0, 1.0, self._zeros(0))

    def create(self, dim, value) -> BatchData:
        return BatchData(dim_cast(dim), 1.0, self.xp.array(self._row(value)[None, :]))

    def create_from_terms(self, dims, vals) -> BatchData:
        dims = np.asarray(dims, dtype=DIM_DTYPE)
        if dims.size == 0:
            return self._empty()
        if _is_batch_array(vals) and vals.dtype != object and vals.ndim == 2:
            rows = self.xp.asarray(vals)
        else:
            rows = self.xp.stack([self._row(v) for v in vals])
        lat = _lattice(dims)
        if lat is None:
            raise ValueError("BatchDenseBackend: grades are not on a single lattice")
        offset, step, idx = lat
        span = int(idx.max()) + 1
        if span > self.max_span:
            raise ValueError(f"BatchDenseBackend: span {span} exceeds max_span")
        if self.jax:
            out = self._zeros(span).at[idx].add(rows)
        else:
            out = self._zeros(span)
            if np.unique(idx).size == idx.size:
                out[idx] = rows          # distinct grades: a plain write
            else:
                np.add.at(out, idx, rows)   # a grade repeats: sum its rows
        return BatchData(offset, step, out)

    # ---- access ------------------------------------------------------------
    def read_dim(self, data: BatchData, dim):
        n = data.vals.shape[0]
        if n == 0:
            return 0.0
        k = (dim_cast(dim) - data.offset) / data.step
        kr = round(k)
        if abs(k - kr) > 1e-9 or kr < 0 or kr >= n:
            return 0.0
        return Lanes(data.vals[kr])

    def write_dim(self, data: BatchData, dim, value) -> BatchData:
        dims = np.append(data.dims(), dim_cast(dim))
        vals = list(Lanes(r) for r in data.vals) + [value]
        return self.create_from_terms(dims, vals)

    def to_arrays(self, data: BatchData) -> Tuple[np.ndarray, np.ndarray]:
        out = np.empty(data.vals.shape[0], dtype=object)
        for i, r in enumerate(data.vals):
            out[i] = Lanes(r)
        return data.dims(), out

    def active_dims(self, data: BatchData) -> np.ndarray:
        return data.dims()

    def term_count(self, data: BatchData) -> int:
        return int(data.vals.shape[0])

    def is_wholly_zero(self, data: BatchData) -> bool:
        return data.vals.shape[0] > 0 and not _any(data.vals)

    def is_unit(self, data: BatchData) -> bool:
        return (data.vals.shape[0] == 1 and data.offset == 0.0
                and _all(data.vals[0] == 1.0))

    # ---- arithmetic --------------------------------------------------------
    def _align(self, a: BatchData, b: BatchData):
        na, nb = a.vals.shape[0], b.vals.shape[0]
        if na == 1 and nb > 1:
            a = BatchData(a.offset, b.step, a.vals)
        elif nb == 1 and na > 1:
            b = BatchData(b.offset, a.step, b.vals)
        step = a.step if a.step == b.step else _common_step(a.step, b.step)
        if step is None:
            return None
        delta = abs(a.offset - b.offset)
        k = delta / step
        if abs(k - round(k)) > 1e-9:
            rem = delta % step
            if rem > step / 2.0:
                rem = step - rem
            step = _common_step(step, rem)
            if step is None:
                return None
        return self._resample(a, step), self._resample(b, step)

    def _resample(self, d: BatchData, step: float) -> BatchData:
        if d.step == step:
            return d
        f = int(round(d.step / step))
        n = d.vals.shape[0]
        out = self._set_at(self._zeros((n - 1) * f + 1 if n else 0), slice(None, None, f), d.vals)
        return BatchData(d.offset, step, out)

    def add(self, a: BatchData, b: BatchData) -> BatchData:
        if a.vals.shape[0] == 0:
            return b
        if b.vals.shape[0] == 0:
            return a
        al = self._align(a, b)
        if al is None:
            dims = np.concatenate([a.dims(), b.dims()])
            return self.create_from_terms(dims, self.xp.concatenate([a.vals, b.vals]))
        a, b = al
        step = a.step
        lo = min(a.offset, b.offset)
        hi = max(a.offset + step * (a.vals.shape[0] - 1), b.offset + step * (b.vals.shape[0] - 1))
        n = int(round((hi - lo) / step)) + 1
        ia = int(round((a.offset - lo) / step))
        ib = int(round((b.offset - lo) / step))
        out = self._add_at(self._zeros(n), slice(ia, ia + a.vals.shape[0]), a.vals)
        out = self._add_at(out, slice(ib, ib + b.vals.shape[0]), b.vals)
        return BatchData(lo, step, out)

    def convolve(self, a: BatchData, b: BatchData) -> BatchData:
        na, nb = a.vals.shape[0], b.vals.shape[0]
        if na == 0 or nb == 0:
            return self._empty()
        al = self._align(a, b)
        if al is None:
            raise ValueError("BatchDenseBackend: incommensurable lattices in convolve")
        a, b = al
        na, nb = a.vals.shape[0], b.vals.shape[0]
        out = self._zeros(na + nb - 1)
        for i in range(na):                 # grades are few; the batch is long
            out = self._add_at(out, slice(i, i + nb), a.vals[i] * b.vals)
        return BatchData(a.offset + b.offset, a.step, out)

    def deconvolve_cut(self, a: BatchData, b: BatchData, min_terms: int = 50) -> tuple:
        if b.vals.shape[0] == 0 or not _any(b.vals):
            raise ZeroDivisionError("BatchDenseBackend: division by a zero composite")
        al = self._align(a, b)
        if al is None:
            raise ValueError("BatchDenseBackend: incommensurable lattices in deconvolve")
        a, b = al
        step = a.step
        nzrow = _concrete(self.xp.any(b.vals != 0.0, axis=1))
        nz = [k for k in range(b.vals.shape[0]) if nzrow[k]]
        lead_k = nz[-1]
        lead = b.vals[lead_k]
        if not _all(lead != 0.0):
            raise BatchDecisionError(
                "the divisor's leading grade is zero in some batch rows and not others",
                batch_rows=_minority(lead != 0.0))
        bv = b.vals[:lead_k + 1][::-1]                 # descending
        terms = max(a.vals.shape[0] + b.vals.shape[0], min_terms)
        rem = self._set_at(self._zeros(max(terms, a.vals.shape[0])),
                           slice(0, a.vals.shape[0]), a.vals[::-1])
        q = self._zeros(terms)
        spilled = False
        for i in range(terms):
            if i >= rem.shape[0]:
                break
            c = rem[i] / lead
            q = self._set_at(q, i, c)
            m = min(bv.shape[0], rem.shape[0] - i)
            spilled = spilled or m < bv.shape[0]
            rem = self._add_at(rem, slice(i, i + m), -(c * bv[:m]))
            rem = self._set_at(rem, i, 0.0)
        cut = spilled or _any(rem)
        a_top = a.offset + step * (a.vals.shape[0] - 1)
        b_top = b.offset + step * lead_k
        q_top = a_top - b_top
        qrow = _concrete(self.xp.any(q != 0.0, axis=1))
        used = (int(np.nonzero(qrow)[0].max()) + 1) if qrow.any() else 1
        vals = q[:used][::-1]
        return BatchData(q_top - step * (used - 1), step, vals), cut

    def scalar_multiply(self, data: BatchData, scalar) -> BatchData:
        s = scalar.v if isinstance(scalar, Lanes) else float(scalar)
        if not isinstance(scalar, Lanes) and s == 0.0:
            return self._empty()
        return BatchData(data.offset, data.step, data.vals * s)

    def scalar_divide(self, data: BatchData, scalar) -> BatchData:
        s = scalar.v if isinstance(scalar, Lanes) else float(scalar)
        return BatchData(data.offset, data.step, data.vals / s)

    def negate(self, data: BatchData) -> BatchData:
        return BatchData(data.offset, data.step, -data.vals)


# --- running a batch that may fail in some rows ---------------------------------

class BatchResult:
    """What run_batch returns.

    values  {name: float64 array over every input row}; NaN in a row that failed
            (empty if no run succeeded, so no output name is known)
    errors  {row index: "ExceptionType: message"} for every row that failed --
            NaN alone is not the signal, a computed value can be NaN
    runs    how many batch runs it took, the first included
    rows    how many input rows there were
    """

    def __init__(self, values, errors, runs, rows):
        self.values, self.errors, self.runs, self.rows = values, errors, runs, rows

    def __repr__(self):
        return f"BatchResult({self.rows} rows, {len(self.errors)} failed, {self.runs} runs)"


def _as_rows(v, n):
    v = getattr(v, "v", v)
    return np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)).copy()


def run_batch(fn, inputs, strict=False, max_runs=64):
    """Run fn over a batch; when it fails, set aside the rows that caused it and
    run the rest again.

        fn       fn(**{name: Lanes}) -> {name: value} or one value; each value a
                 Lanes, an array over the rows, or a number
        inputs   {name: array}, every array the same length
        strict   raise the first error, as a plain batch call does

    A row is never computed it can not be vouched for: an error that names its
    rows (a decision that differs between rows, an overflow or domain error in
    some rows) has those rows taken out and the rest run again.  Rows taken out
    by a DECISION are not wrong, they take another path: they are run again as
    their own batch, through this same function.  Rows taken out by a numeric
    error are reported in BatchResult.errors.

    An error that names no rows, raised while every row decided alike, is the
    same error for every row of that batch and is reported for all of them.
    Errors other than ValueError and ArithmeticError -- a fault in the code
    rather than in the data -- are raised.

    The backend in force before the call is restored afterwards.
    """
    import composite.composite_lib as cl
    from composite.backends import config
    names = list(inputs)
    arrays = {k: np.asarray(inputs[k], dtype=np.float64) for k in names}
    n = len(arrays[names[0]])
    if any(a.shape != (n,) for a in arrays.values()):
        raise ValueError("run_batch: inputs of different lengths")
    old = config.get_backend()
    values, errors, runs = {}, {}, [0]

    def attempt(rows):
        config.use_batch_dense(len(rows))
        cl._refresh_constants()
        runs[0] += 1
        out = fn(**{k: Lanes(arrays[k][rows]) for k in names})
        single = not isinstance(out, dict)
        out = {None: out} if single else out
        for key, v in out.items():
            col = values.setdefault(key, np.full(n, np.nan))
            col[rows] = _as_rows(v, len(rows))

    def solve(rows):
        pending = [rows]
        while pending:
            rows = pending.pop()
            while len(rows):
                if runs[0] >= max_runs:
                    for i in rows:
                        errors[int(i)] = "RuntimeError: run_batch stopped after max_runs runs"
                    return
                try:
                    attempt(rows)
                    break
                except (ValueError, ArithmeticError) as e:
                    if strict:
                        raise
                    mask = getattr(e, "batch_rows", None)
                    if mask is None or np.shape(mask) != (len(rows),) or not np.any(mask):
                        for i in rows:
                            errors[int(i)] = f"{type(e).__name__}: {e}"
                        break
                    out_rows, rows = rows[mask], rows[~mask]
                    if isinstance(e, BatchDecisionError):
                        pending.append(out_rows)            # another path, not an error
                    else:
                        for i in out_rows:
                            errors[int(i)] = f"{type(e).__name__}: {e}"

    try:
        solve(np.arange(n))
    finally:
        config.set_backend(old)
        cl._refresh_constants()
    if None in values and len(values) == 1:
        values = {"value": values[None]}
    return BatchResult(values, errors, runs[0], n)
