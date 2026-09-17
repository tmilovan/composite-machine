# composite/backends/sparse_dense_backend.py
# Composite Machine — Clustered Sparse-Dense Backend (NumPy)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import math
import numpy as np
from typing import Tuple
from .base_backend import CompositeBackend, DIM_DTYPE, dim_cast

# Direct convolution costs len(a)*len(b) multiply-adds; FFT costs O(M log M) on
# the padded transform length.  So the choice must be made on the PRODUCT of the
# lengths, not their sum.  Measured crossover is near 100k multiply-adds.
#
# The old rule was `len(a) + len(b) > 128`, which is wrong in both directions:
#   kernel 5 x grid 1000  -> sum 1005, took FFT at 29.0us; direct is 2.2us (13x)
#   kernel 5 x grid 5000  -> sum 5005, took FFT at 97.4us; direct is 6.3us (15x)
# A short kernel against a long grid -- every PDE stencil, every filter -- has a
# large sum and a tiny product, and direct wins by an order of magnitude.
# When allow_fft is on, choose by comparing the two actual costs:
#   direct  ~ la * lb              multiply-adds
#   FFT     ~ K * M * log2(M)      three transforms of length M ~ la + lb
# Calibrated K = 20 from measurement; it puts the crossover at la=lb=384, which
# is where direct and FFT measured equal (29.7us vs 25.2us).
#
# Comparing la+lb (the original rule) or la*lb alone (my first fix) both get
# asymmetric shapes badly wrong.  A 1-term kernel against 263k terms is a SCALED
# COPY -- 67us direct -- but has product 263k, so a product threshold sends it to
# FFT at 5548us, 83x worse.  Every PDE stencil and every filter has exactly that
# shape.
_FFT_COST_FACTOR = 20.0

def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth integer >= n.

    numpy's FFT is fast on lengths factoring into 2, 3 and 5 and falls back to
    Bluestein's algorithm otherwise, which is dramatically slower -- and the
    natural length here, len(a)+len(b)-1, is arbitrary.  Padding to a smooth
    length and slicing back was worth up to 8.8x:
        n=1000 each   raw FFT 333.1us   padded 47.6us   direct 132.6us
        n=3000 each   raw FFT 1086us    padded 119.5us  direct 1056us
    """
    if n <= 16:
        return n
    best = 1 << (n - 1).bit_length()          # next power of two, an upper bound
    p5 = 1
    while p5 < best:
        p3 = p5
        while p3 < best:
            k = p3
            while k < n:
                k *= 2
            if k < best:
                best = k
            p3 *= 3
        p5 *= 5
    return best


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FFT convolution, padded to a fast transform length."""
    n = len(a) + len(b) - 1
    m = _next_fast_len(n)
    return np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)[:n]


def _use_fft(la: int, lb: int) -> bool:
    """Only ever consulted when a backend was explicitly built with allow_fft."""
    m = la + lb
    if m < 64:
        return False
    return la * lb > _FFT_COST_FACTOR * m * math.log2(m)


class SparseData:
    """Internal storage: a list of TRUE CONTIGUOUS RUNS.

    runs: list of (offset:int, vals:np.ndarray[float64]), sorted by offset,
          pairwise non-overlapping AND non-adjacent (there is always a real gap
          of at least one dimension between consecutive runs).  A run spans
          dims offset .. offset+len(vals)-1 with every one of them present.

    One run == a dense composite.  Many runs == a sparse one.  Nothing is ever
    gap-filled in storage, so every dimension in a run is a dimension the
    computation actually built.

    WHY RUNS AND NOT FLAT (dims, vals):  the flat form made every operation
    re-derive the run structure of its operands -- _cluster_terms ran twice per
    multiply and, with the merge and mask that followed it, accounted for ~90%
    of arithmetic time while the convolution itself was 3.5%.  The structure of
    a result is derivable from its operands (the Minkowski sum of two contiguous
    ranges is contiguous), so it never needed rediscovering.

    WHY *TRUE* RUNS:  densifying across a gap invents dimensions that the product
    never built, which then had to be removed by masking against the Minkowski
    sum.  Convolving true runs pairwise produces exactly the Minkowski sum by
    construction -- the mask becomes unnecessary rather than merely cheaper.

    .dims / .vals remain available as materialising properties so that callers
    written against the flat form (deconvolve, and anything outside this file)
    keep working unchanged.
    """
    __slots__ = ('_runs', '_dims', '_vals', '_wz')

    def __init__(self, dims=None, vals=None, runs=None):
        """Either form may be supplied; the other is derived on demand.

        Construction from flat arrays does NOT build runs eagerly.  Splitting
        into runs costs a diff and a nonzero, and composites are built far more
        often than they are multiplied -- building eagerly made construction 48%
        slower than the flat backend and cost the TSP solver 20% overall, even
        though every arithmetic operation had got faster.  Arithmetic results
        arrive as runs and never pay for the flat form at all.
        """
        self._wz = None
        if runs is not None:
            self._runs = runs
            self._dims = None
            self._vals = None
        else:
            self._runs = None
            self._dims = np.asarray(dims, dtype=DIM_DTYPE)
            self._vals = np.asarray(vals, dtype=np.float64)

    @property
    def runs(self):
        if self._runs is None:
            self._runs = _runs_from_flat(self._dims, self._vals)
        return self._runs

    def _materialise(self):
        if not self._runs:
            self._dims = np.array([], dtype=DIM_DTYPE)
            self._vals = np.array([], dtype=np.float64)
            return
        if len(self.runs) == 1:
            off, v = self.runs[0]
            self._dims = np.arange(off, off + len(v), dtype=DIM_DTYPE)
            self._vals = v
            return
        self._dims = np.concatenate([np.arange(o, o + len(v), dtype=DIM_DTYPE)
                                     for o, v in self.runs])
        self._vals = np.concatenate([v for _, v in self.runs])

    @property
    def dims(self):
        if self._dims is None:
            self._materialise()
        return self._dims

    @property
    def vals(self):
        if self._vals is None:
            self._materialise()
        return self._vals

    def __repr__(self):
        terms = [f"|{v}|_{d}" for d, v in zip(self.dims, self.vals)]
        return "Composite(" + " + ".join(terms) + ")"


def _runs_from_flat(dims: np.ndarray, vals: np.ndarray):
    """Split sorted unique (dims, vals) into true contiguous runs."""
    if len(dims) == 0:
        return []
    if len(dims) == 1:
        return [(dim_cast(dims[0]), vals)]
    gaps = np.diff(dims)
    if gaps.min() <= 0:                      # unsorted and/or duplicated
        order = np.argsort(dims, kind='mergesort')
        dims = dims[order]
        vals = vals[order]
        gaps = np.diff(dims)
        if (gaps == 0).any():
            # Duplicates break the run invariant (length must equal span), so
            # fold them by summing.  The flat form tolerated them silently.
            uniq, inv = np.unique(dims, return_inverse=True)
            acc = np.zeros(len(uniq), dtype=np.float64)
            np.add.at(acc, inv, vals)
            dims, vals = uniq, acc
            if len(dims) == 1:
                return [(dim_cast(dims[0]), vals)]
            gaps = np.diff(dims)
    cuts = np.nonzero(gaps != 1)[0]
    if len(cuts) == 0:
        return [(dim_cast(dims[0]), vals)]
    # Direct slicing, not np.split: slices are views and cost nothing, while
    # np.split routes through array_split and its validation -- which showed up
    # as 3.7% of a whole TSP run purely in dispatch.
    out = []
    prev = 0
    for c in cuts:
        c = int(c) + 1
        out.append((dim_cast(dims[prev]), vals[prev:c]))
        prev = c
    out.append((dim_cast(dims[prev]), vals[prev:]))
    return out


def _merge_runs(parts):
    """Combine (offset, vals) pieces into canonical runs, summing overlaps.

    A run is UNIT-SPACED, so two runs can only be combined when they sit on the
    same unit lattice -- that is, when their offsets differ by a whole number.
    A run at 0,1,2 and a run at 0.5,1.5 overlap in span but interleave; merging
    them into one dense array would put every member at the wrong dimension and
    ask for an array of length 1.5.  Half-integer offsets arise from sqrt of an
    odd dimension, so partition by the fractional part of the offset first and
    merge only within each lattice.
    """
    if not parts:
        return []
    if len(parts) == 1:
        return parts
    lattices = {}
    for off, v in parts:
        lattices.setdefault(float(off) % 1.0, []).append((off, v))
    if len(lattices) > 1:
        out = []
        for key in sorted(lattices):
            out.extend(_merge_one_lattice(lattices[key]))
        return sorted(out, key=lambda p: p[0])
    return _merge_one_lattice(parts)


def _merge_one_lattice(parts):
    """Merge runs known to share a unit lattice; spans here are whole numbers."""
    if len(parts) == 1:
        return parts
    parts = sorted(parts, key=lambda p: p[0])
    groups = []
    lo, hi, cur = parts[0][0], parts[0][0] + len(parts[0][1]), [parts[0]]
    for off, v in parts[1:]:
        end = off + len(v)
        if off <= hi:                      # overlapping or adjacent
            cur.append((off, v))
            hi = max(hi, end)
        else:
            groups.append((lo, hi, cur))
            lo, hi, cur = off, end, [(off, v)]
    groups.append((lo, hi, cur))
    out = []
    for glo, ghi, members in groups:
        if len(members) == 1:
            out.append(members[0])
        else:
            acc = np.zeros(int(round(ghi - glo)), dtype=np.float64)
            for off, v in members:
                i = int(round(off - glo))
                acc[i: i + len(v)] += v
            out.append((glo, acc))
    return out


# ── Clustering ────────────────────────────────────────────────

# ── Backend ───────────────────────────────────────────────────

class SparseDenseBackend(CompositeBackend):
    """Clustered Sparse-Dense backend.

    Storage: dual arrays (dims[], vals[]) — only active terms.
    Computation: cluster nearby terms → local np.convolve → merge.

    Scales with number of active terms, NOT dimension span.
    A Composite with terms at dim -10,000,000 and dim 0 uses
    exactly 2 elements of storage, not 10,000,001.
    """

    def __init__(self, gap_threshold: int = 64, zero_tol: float = 0.0,
                 max_order: int = None, allow_fft: bool = False):
        """allow_fft defaults to FALSE: composite arithmetic is EXACT.

        FFT convolution is wrong by ~1e-13 relative, by construction -- it is a
        different algorithm that happens to approximate the same answer, not a
        faster way of computing it.  Direct convolution is exact to the last bit.
        An arithmetic that is only approximately right is not this arithmetic,
        so exactness is the default and speed is the thing you opt into.

        The price is small wherever the work actually is.  Measured:
            150x150     exact 6.9us   FFT 15.3us   exact is 2x FASTER
            300x300     exact 17.9us  FFT 21.1us   exact is 1.2x FASTER
            700x700     exact 73us    FFT 36us     exact costs 2x
            1000x1000   exact 137us   FFT 46us     exact costs 3x
        Direct also wins outright below ~100k multiply-adds, which covers every
        stencil, every derivative tower and every crossword row.

        Pass allow_fft=True only when you have decided that 1e-13 does not
        matter for what you are computing, and say so where you decide it.
        """
        self.gap_threshold = gap_threshold
        self.zero_tol = zero_tol
        self.max_order = max_order
        self.allow_fft = allow_fft

    # --- lifecycle ---

    def create(self, dim: int, value: float) -> SparseData:
        return SparseData(runs=[(dim_cast(dim),
                                 np.array([value], dtype=np.float64))])

    def _truncate(self, data: SparseData) -> SparseData:
        """Drop dimensions below -max_order (higher-order derivatives)."""
        if self.max_order is None or not data.runs:
            return data
        lo = -self.max_order
        out = []
        for off, v in data.runs:
            end = off + len(v)
            if end <= lo:
                continue                       # entirely below the cut
            if off >= lo:
                out.append((off, v))           # entirely above
            else:
                out.append((lo, v[lo - off:]))  # trim the low end
        if len(out) == len(data.runs) and all(
                a is b for (_, a), (_, b) in zip(out, data.runs)):
            return data
        return SparseData(runs=out)

    def create_from_terms(self, dims: np.ndarray, vals: np.ndarray) -> SparseData:
        """Store the flat form as given; runs are derived lazily on first use.

        Callers in composite_lib hand over sorted, unique dimensions (they come
        from sorted(dict.keys()) or from arithmetic), so validating here is work
        on the hottest path in the library for a condition that essentially never
        fails.  _runs_from_flat asserts the ordering when it eventually runs.
        """
        return SparseData(np.asarray(dims, dtype=DIM_DTYPE),
                          np.asarray(vals, dtype=np.float64))

    # --- access ---

    def read_dim(self, data: SparseData, dim: int) -> float:
        """Serve from whichever representation is already built.

        Runs give O(1) for a dense composite and O(k) over k runs, about 4x
        faster than a binary search.  But a composite built from flat arrays and
        read once or twice -- which is most of what the TSP solver does -- should
        not pay for splitting into runs just to answer a read.  So fall back to
        searchsorted on the flat form when runs have not been needed yet.
        """
        # This backend stores dimensions in a float64 array and cannot hold a
        # VECTOR dimension at all, so it definitionally does not have one --
        # the answer is 0.0.  Without this, searchsorted compares a float
        # element against a tuple and raises "truth value of an array is
        # ambiguous", so reading a log-axis coefficient off a scalar composite
        # crashed instead of saying "not present".
        if isinstance(dim, tuple):
            return 0.0
        runs = data._runs
        if runs is None:
            d = data._dims
            idx = np.searchsorted(d, dim)
            if idx < len(d) and d[idx] == dim:
                return float(data._vals[idx])
            return 0.0
        for off, v in runs:
            if dim < off:
                return 0.0
            j = dim - off
            if j < len(v):
                # A run is unit-spaced, so the position within it is an integer.
                # A fractional j means dim is not on THIS run's lattice -- but
                # runs on different lattices overlap in span (a run at -0.5 and
                # one at 0 both cover dimension 0's neighbourhood), so keep
                # scanning instead of concluding the dimension is absent.
                ji = int(j)
                if ji == j:
                    return float(v[ji])
        return 0.0

    # FIXED: write_dim — always write the value, even if zero.
    # Previously: deleted existing dim if value==0, skipped insert if value==0.
    # Now: expressed zeros are preserved (canon rule: if zero is expressed, retain it).
    def write_dim(self, data: SparseData, dim: int, value: float) -> SparseData:
        dim = float(dim)
        out = []
        placed = False
        for off, v in data.runs:
            end = off + len(v)
            if not placed and off <= dim < end and float(dim - off).is_integer():
                nv = v.copy()                                # on this run's lattice
                nv[int(dim - off)] = value
                out.append((off, nv))
                placed = True
            else:
                out.append((off, v))
        if not placed:
            out.append((dim, np.array([value], dtype=np.float64)))
            out = _merge_runs(out)     # may now touch a neighbouring run
        return SparseData(runs=out)

    def to_arrays(self, data: SparseData) -> Tuple[np.ndarray, np.ndarray]:
        return data.dims.copy(), data.vals.copy()

    def active_dims(self, data: SparseData) -> np.ndarray:
        return data.dims.copy()

    # NOTE: to_arrays/active_dims materialise the flat form and are BOUNDARY
    # operations for callers that need it.  They must not be used inside add,
    # convolve, scalar_multiply or negate -- that round trip is what this
    # representation exists to remove.

    def term_count(self, data: SparseData) -> int:
        if data._runs is None:
            return len(data._dims)
        return sum(len(v) for _, v in data._runs)

    def is_wholly_zero(self, data: SparseData) -> bool:
        """Answer from whichever representation already exists.

        Called twice per multiply and division (Zero Rule R1 converts a wholly-
        zero operand).  Reading .runs here FORCED the split for every composite
        the predicate touched -- 20,823 run-builds in a two-run TSP solve, whose
        cost exceeded every saving the new representation made.  A predicate must
        never be the thing that decides a representation gets built.

        Cached: SparseData is immutable in practice.  And v.any() rather than
        np.any(v != 0.0), which allocated a boolean temporary per run.
        """
        if data._wz is None:
            if data._runs is None:
                data._wz = len(data._dims) > 0 and not data._vals.any()
            elif not data._runs:
                data._wz = False
            else:
                data._wz = not any(v.any() for _, v in data._runs)
        return data._wz

    def is_unit(self, data: SparseData) -> bool:
        if data._runs is None:
            d = data._dims
            return len(d) == 1 and d[0] == 0 and data._vals[0] == 1.0
        if len(data._runs) != 1:
            return False
        off, v = data._runs[0]
        return off == 0 and len(v) == 1 and v[0] == 1.0

    # --- arithmetic ---

    # FIXED: add — keep all union dimensions, do NOT strip zeros.
    # Canon rule: 1-1 = |0|₀ (zero at dimension 0, dimension retained).
    def add(self, a: SparseData, b: SparseData) -> SparseData:
        """Union of dimensions, values summed.  Zeros are RETAINED (1-1 = |0|_0).

        With runs this is exactly _merge_runs of the two lists: pieces that
        overlap or touch densify into one and sum on the overlap, pieces with a
        real gap between them stay apart.  No union1d, no searchsorted, no
        argsort -- and no span between distant runs is ever allocated.
        """
        if not a.runs:
            return b
        if not b.runs:
            return a
        return SparseData(runs=_merge_runs(list(a.runs) + list(b.runs)))

    def convolve(self, a: SparseData, b: SparseData) -> SparseData:
        """Run x run convolution.

        The Minkowski sum of two TRUE contiguous runs is contiguous, so the
        pairwise products already carry exactly the dimensions the product
        builds -- including those whose coefficient came out zero, and excluding
        anything a gap-fill would have invented.  _mask_to_minkowski is
        therefore unnecessary here rather than merely cheaper, and the whole
        cluster/merge/mask round trip disappears.
        """
        if not a.runs or not b.runs:
            return SparseData(runs=[])

        results = []
        for offset_a, dense_a in a.runs:
            for offset_b, dense_b in b.runs:
                if self.allow_fft and _use_fft(len(dense_a), len(dense_b)):
                    conv = _fft_convolve(dense_a, dense_b)
                    if not np.all(np.isfinite(conv)):
                        # FFT convolution mixes every input coefficient into
                        # every output bin, so one overflow destroys all of
                        # them -- including bins whose direct product is
                        # perfectly finite.  Deep Taylor towers reach 1e165
                        # routinely, and squaring one silently zeroed the
                        # standard part.  Redo the product exactly: direct
                        # convolution overflows only where the value really
                        # does, leaving every other coefficient intact.
                        conv = np.convolve(dense_a, dense_b)
                else:
                    conv = np.convolve(dense_a, dense_b)
                results.append((offset_a + offset_b, conv))

        return self._truncate(SparseData(runs=_merge_runs(results)))

    # FIXED: deconvolve — use highest dim with non-zero coeff as leading term.
    # With expressed zero preservation, the highest dim may have coeff 0.0,
    # which would cause division by zero / NaN in the quotient step.
    # Also clean near-zero remainder artifacts after each step.
    def deconvolve(self, a: SparseData, b: SparseData) -> SparseData:
        """Polynomial long division in sparse form.

        Computes Q such that A = Q * B (approximately).
        Works term-by-term from highest dimension down.
        """
        if len(b.dims) == 0:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")

        # FIXED: Leading term — highest dim with non-zero coeff
        # 7.1: the leading term is the highest dimension whose coefficient is
        # exactly nonzero.  A tolerance would pick the wrong leading term when
        # a genuine coefficient happens to be tiny.
        nonzero_mask = b.vals != 0.0
        if not np.any(nonzero_mask):
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        lead_dim = b.dims[nonzero_mask][-1]
        lead_val = b.vals[nonzero_mask][-1]

        # Strip near-zero terms from dividend for clean division
        a_mask = a.vals != 0.0
        remainder_dims = a.dims[a_mask].copy()
        remainder_vals = a.vals[a_mask].copy()

        q_dims = []
        q_vals = []

        max_iter = max(len(a.dims) + len(b.dims), 50)
        for _ in range(max_iter):
            if len(remainder_dims) == 0:
                break

            r_dim = remainder_dims[-1]
            r_val = remainder_vals[-1]

            q_dim = r_dim - lead_dim
            q_val = r_val / lead_val
            q_dims.append(q_dim)
            q_vals.append(q_val)

            sub_dims = b.dims + q_dim
            sub_vals = b.vals * q_val

            remainder = self.add(
                SparseData(remainder_dims, remainder_vals),
                SparseData(sub_dims, -sub_vals)
            )
            # FIXED: Clean near-zero remainder terms (division artifacts,
            # not user-expressed zeros — safe to strip here).
            #
            # r_dim is dropped as well: q_val was chosen so that term cancels
            # exactly, so the remainder there is zero by construction.  In
            # floating point the subtraction can leave dust instead, which
            # would keep r_dim as the highest remaining dimension and make the
            # loop emit the same q_dim again — appending a duplicate quotient
            # entry built from the residue.
            mask = (remainder.vals != 0.0) & (remainder.dims != r_dim)
            remainder_dims = remainder.dims[mask]
            remainder_vals = remainder.vals[mask]

        if len(q_dims) == 0:
            return SparseData(np.array([], dtype=DIM_DTYPE),
                              np.array([], dtype=np.float64))

        q_dims = np.array(q_dims, dtype=DIM_DTYPE)
        q_vals = np.array(q_vals, dtype=np.float64)
        order = np.argsort(q_dims)
        return SparseData(q_dims[order], q_vals[order])

    def _map_vals(self, data: SparseData, fn) -> SparseData:
        return SparseData(runs=[(o, fn(v)) for o, v in data.runs])

    def scalar_multiply(self, data: SparseData, scalar: float) -> SparseData:
        if scalar == 0.0:
            return SparseData(runs=[])
        return self._map_vals(data, lambda v: v * scalar)

    def negate(self, data: SparseData) -> SparseData:
        return self._map_vals(data, lambda v: -v)
