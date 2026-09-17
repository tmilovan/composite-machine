# composite/backends/dense_series_backend.py
# Composite Machine — Dense Series Backend
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Contiguous coefficient array.  For CALCULUS, where the data is never sparse.

A Taylor series occupies orders 0..n with no gaps.  Run-length bookkeeping,
lattice-membership tests and gap analysis are all managing a sparsity that is
not there, and they cost more than the arithmetic they manage: profiling one
exp(-(x*x)) evaluation, _merge_one_lattice + _merge_runs + is_wholly_zero +
dim_cast together outweighed numpy.convolve several times over.  Measured, the
same evaluation costs 285.8 us under SparseDenseBackend, 174.6 us under
DictBackend, and 43.7 us as plain dense numpy -- identical coefficients, max
difference 6.94e-18.  The other 242 us was representation, not calculus.

So this backend stores exactly what a series is: an offset, a step, and a
contiguous float64 array.  Multiply is np.convolve, add is an aligned add.

WHEN NOT TO USE IT.  It is the wrong choice for anything genuinely sparse --
a PDE grid holding 22k occupied cells out of a billion would try to allocate
the billion.  `max_span` refuses that rather than exhausting memory, and the
sparse-dense backend remains the right one for that work.  This is opt-in for
exactly the reason that the two regimes want opposite representations.

    from composite.backends import config
    config.use_dense_series()
"""
from typing import Tuple

import numpy as np

from .base_backend import CompositeBackend, DIM_DTYPE, dim_cast


class DenseData:
    """dim of vals[k] is offset + k*step.  step > 0, vals contiguous float64."""
    __slots__ = ('offset', 'step', 'vals')

    def __init__(self, offset: float, step: float, vals: np.ndarray):
        self.offset = float(offset)
        self.step = float(step)
        self.vals = vals

    def dims(self) -> np.ndarray:
        return (self.offset
                + self.step * np.arange(len(self.vals), dtype=DIM_DTYPE))

    def __repr__(self):
        return "DenseData(" + " + ".join(
            f"|{v}|_{d}" for d, v in zip(self.dims(), self.vals)) + ")"


def _lattice(dims):
    """(offset, step, index array) for dims, or None if they are not on a lattice."""
    d = np.asarray(dims, dtype=DIM_DTYPE)
    if d.size == 0:
        return 0.0, 1.0, np.array([], dtype=np.int64)
    if d.size == 1:
        return float(d[0]), 1.0, np.array([0], dtype=np.int64)
    lo = float(d.min())
    spread = d - lo
    pos = spread[spread > 0]
    step = float(pos.min()) if pos.size else 1.0
    k = spread / step
    kr = np.rint(k)
    # A tolerance is needed because the dims are float64 dyadics; 1e-9 is far
    # below any real spacing and far above the representation error.
    if not np.all(np.abs(k - kr) < 1e-9):
        return None
    return lo, step, kr.astype(np.int64)


def _common_step(s1: float, s2: float):
    """Finest step both lattices sit on, or None if they are incommensurable."""
    lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
    if lo <= 0:
        return None
    r = hi / lo
    return lo if abs(r - round(r)) < 1e-9 else None


class DenseSeriesBackend(CompositeBackend):
    """Contiguous-array backend.  See module docstring for when NOT to use it."""

    VECTOR_DIMS = False

    def __init__(self, max_span: int = 1 << 20):
        # Guard against being handed genuinely sparse data: a PDE front spanning
        # a billion indices must fail loudly here, not allocate a billion floats.
        self.max_span = int(max_span)

    # ---- construction -----------------------------------------------------

    def _make(self, offset, step, vals):
        return DenseData(offset, step, np.asarray(vals, dtype=np.float64))

    def create(self, dim, value) -> DenseData:
        return self._make(dim_cast(dim), 1.0, np.array([float(value)]))

    def create_from_terms(self, dims, vals) -> DenseData:
        dims = np.asarray(dims, dtype=DIM_DTYPE)
        vals = np.asarray(vals, dtype=np.float64)
        if dims.size == 0:
            return self._make(0.0, 1.0, np.array([]))
        lat = _lattice(dims)
        if lat is None:
            raise ValueError(
                "DenseSeriesBackend: dimensions are not on a single lattice; "
                "use the sparse-dense backend for this data")
        offset, step, idx = lat
        span = int(idx.max()) + 1
        if span > self.max_span:
            raise ValueError(
                f"DenseSeriesBackend: span {span} exceeds max_span "
                f"{self.max_span} -- this data is sparse, use SparseDenseBackend")
        out = np.zeros(span, dtype=np.float64)
        # duplicates accumulate, matching convolve/add semantics elsewhere
        np.add.at(out, idx, vals)
        return self._make(offset, step, out)

    # ---- access -----------------------------------------------------------

    def read_dim(self, data: DenseData, dim) -> float:
        if data.vals.size == 0:
            return 0.0
        k = (dim_cast(dim) - data.offset) / data.step
        kr = round(k)
        if abs(k - kr) > 1e-9 or kr < 0 or kr >= data.vals.size:
            return 0.0
        return float(data.vals[kr])

    def write_dim(self, data: DenseData, dim, value) -> DenseData:
        d = dim_cast(dim)
        if data.vals.size == 0:
            return self.create(d, value)
        k = (d - data.offset) / data.step
        kr = round(k)
        if abs(k - kr) < 1e-9 and 0 <= kr < data.vals.size:
            out = data.vals.copy()
            out[kr] = float(value)
            return self._make(data.offset, data.step, out)
        # outside the current span (or off-lattice): rebuild through the general path
        dims = np.append(data.dims(), d)
        vals = np.append(data.vals, float(value))
        return self.create_from_terms(dims, vals)

    def to_arrays(self, data: DenseData) -> Tuple[np.ndarray, np.ndarray]:
        return data.dims(), data.vals.copy()

    def active_dims(self, data: DenseData) -> np.ndarray:
        return data.dims()

    def term_count(self, data: DenseData) -> int:
        return int(data.vals.size)

    def is_wholly_zero(self, data: DenseData) -> bool:
        return data.vals.size > 0 and not data.vals.any()

    def is_unit(self, data: DenseData) -> bool:
        return (data.vals.size == 1 and data.offset == 0.0
                and data.vals[0] == 1.0)

    # ---- arithmetic -------------------------------------------------------

    def _align(self, a: DenseData, b: DenseData):
        """Both operands on one step, or None if the lattices do not share one."""
        if a.step == b.step:
            return a, b
        step = _common_step(a.step, b.step)
        if step is None:
            return None
        return self._resample(a, step), self._resample(b, step)

    def _resample(self, d: DenseData, step: float) -> DenseData:
        if d.step == step:
            return d
        f = int(round(d.step / step))
        out = np.zeros((len(d.vals) - 1) * f + 1 if len(d.vals) else 0)
        out[::f] = d.vals
        return self._make(d.offset, step, out)

    def add(self, a: DenseData, b: DenseData) -> DenseData:
        if a.vals.size == 0:
            return b
        if b.vals.size == 0:
            return a
        al = self._align(a, b)
        if al is None:
            return self._via_terms_add(a, b)
        a, b = al
        step = a.step
        lo = min(a.offset, b.offset)
        hi = max(a.offset + step * (len(a.vals) - 1),
                 b.offset + step * (len(b.vals) - 1))
        n = int(round((hi - lo) / step)) + 1
        if n > self.max_span:
            raise ValueError(
                f"DenseSeriesBackend: add would span {n}; this data is sparse")
        out = np.zeros(n, dtype=np.float64)
        ia = int(round((a.offset - lo) / step))
        ib = int(round((b.offset - lo) / step))
        out[ia:ia + len(a.vals)] += a.vals
        out[ib:ib + len(b.vals)] += b.vals
        return self._make(lo, step, out)

    def _via_terms_add(self, a, b):
        dims = np.concatenate([a.dims(), b.dims()])
        vals = np.concatenate([a.vals, b.vals])
        return self.create_from_terms(dims, vals)

    def convolve(self, a: DenseData, b: DenseData) -> DenseData:
        if a.vals.size == 0 or b.vals.size == 0:
            return self._make(0.0, 1.0, np.array([]))
        al = self._align(a, b)
        if al is None:
            raise ValueError(
                "DenseSeriesBackend: incommensurable lattices in convolve")
        a, b = al
        n = len(a.vals) + len(b.vals) - 1
        if n > self.max_span:
            raise ValueError(
                f"DenseSeriesBackend: convolve would span {n}; data is sparse")
        return self._make(a.offset + b.offset, a.step,
                          np.convolve(a.vals, b.vals))

    def deconvolve(self, a: DenseData, b: DenseData) -> DenseData:
        """Long division, highest dimension first.

        The quotient of a non-terminating division is cut at `terms`; that is
        why Composite.__truediv__ records the result's completeness rather than
        calling it exact.
        """
        if b.vals.size == 0 or not b.vals.any():
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")
        al = self._align(a, b)
        if al is None:
            raise ValueError(
                "DenseSeriesBackend: incommensurable lattices in deconvolve")
        a, b = al
        step = a.step
        nz = np.nonzero(b.vals)[0]
        lead_k = int(nz[-1])                       # highest expressed dim of b
        lead_v = float(b.vals[lead_k])
        bv = b.vals[:lead_k + 1][::-1]             # descending powers
        terms = max(len(a.vals) + len(b.vals), 50)
        rem = a.vals[::-1].astype(np.float64).copy()   # descending
        q = np.zeros(terms, dtype=np.float64)
        for i in range(terms):
            if i >= len(rem):
                break
            c = rem[i] / lead_v
            q[i] = c
            if c != 0.0:
                m = min(len(bv), len(rem) - i)
                rem[i:i + m] -= c * bv[:m]
            rem[i] = 0.0
        # q[i] is the coefficient of descending index i
        a_top = a.offset + step * (len(a.vals) - 1)
        b_top = b.offset + step * lead_k
        q_top = a_top - b_top
        used = int(np.max(np.nonzero(q)[0]) + 1) if q.any() else 1
        vals = q[:used][::-1]                      # back to ascending
        return self._make(q_top - step * (used - 1), step, vals)

    def scalar_multiply(self, data: DenseData, scalar: float) -> DenseData:
        if scalar == 0.0:
            return self._make(0.0, 1.0, np.array([]))
        return self._make(data.offset, data.step, data.vals * float(scalar))

    def negate(self, data: DenseData) -> DenseData:
        return self._make(data.offset, data.step, -data.vals)

    # ---- NO fused series evaluation, deliberately ------------------------
    #
    # A poly_series() computing sum c_k h**k directly in arrays was tried and
    # REMOVED.  It was 2.2x faster and it changed the algebra:
    #
    #   * The dense frame expresses EVERY dimension in its span, zeros
    #     included, while the Composite-level loop expresses only dimensions
    #     that actually arose.  Under the Zero Rules an expressed zero TERM is
    #     not the same object as an absent one (R2), so this is a semantic
    #     change, not a representation detail.  Most divergences showed
    #     diff = 0.0 with differing dimension SETS -- right numbers, wrong
    #     structure, which is the failure mode that would never show up in a
    #     value comparison.
    #   * Powers h**k share h's lattice only when h.offset is a multiple of
    #     h.step.  {-3:..,-1:..} puts h**1 on odd dims and h**2 on even ones;
    #     fractional dims need a finer step than a single-term h reports.  Two
    #     guards later it still diverged numerically on {0:1, -0.5:0.4}.
    #
    # It was a second implementation of the core arithmetic, and it diverged
    # the way second implementations do.  The speed is not worth it: this
    # backend exists to make the SAME operations cheaper, never to replace
    # them.  If the series loop is to be fused, it must be fused in terms the
    # Zero Rules define, not in raw arrays.
