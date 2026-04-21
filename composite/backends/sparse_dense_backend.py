# composite/backends/sparse_dense_backend.py
# Composite Machine — Clustered Sparse-Dense Backend (NumPy)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import numpy as np
from typing import Tuple, List
from .base_backend import CompositeBackend

# Segment size threshold for FFT convolution.
# Below this: np.convolve (direct, exact).
# Above this: FFT (O(N log N), tiny rounding artifacts ~1e-16).
# Set at 128 to avoid rounding near the zero_tol pruning boundary.
_FFT_THRESHOLD = 128


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FFT-based convolution. O(N log N) vs O(N^2) direct."""
    n = len(a) + len(b) - 1
    return np.fft.irfft(np.fft.rfft(a, n) * np.fft.rfft(b, n), n)


class SparseData:
    """Internal storage: parallel sorted arrays of dims and vals.

    dims: int64 array of dimension indices (sorted, unique)
    vals: float64 array of corresponding coefficients
    len(dims) == len(vals) == number of active terms

    INVARIANT: No gaps are filled. Only explicitly computed
    dimensions exist. dim -10,000,000 and dim 0 coexist
    without allocating anything in between.
    """
    __slots__ = ('dims', 'vals')

    def __init__(self, dims: np.ndarray, vals: np.ndarray):
        self.dims = dims
        self.vals = vals

    def __repr__(self):
        terms = [f"|{v}|_{d}" for d, v in zip(self.dims, self.vals)]
        return "Composite(" + " + ".join(terms) + ")"


# ── Clustering ────────────────────────────────────────────────

def _cluster_terms(
    dims: np.ndarray, vals: np.ndarray, gap_threshold: int = 64
) -> List[Tuple[int, np.ndarray]]:
    """Split terms into clusters of nearby dimensions.

    Returns list of (offset, dense_array) tuples.
    offset = the lowest dimension in the cluster.
    dense_array = local dense expansion of that cluster.

    Terms within gap_threshold of each other are grouped.
    Gaps within a cluster ARE filled with zeros (these are
    temporary computation artifacts, never stored back).
    """
    if len(dims) == 0:
        return []

    clusters = []
    gaps = np.diff(dims)
    split_mask = gaps > gap_threshold
    split_indices = np.nonzero(split_mask)[0] + 1

    dim_groups = np.split(dims, split_indices)
    val_groups = np.split(vals, split_indices)

    for dg, vg in zip(dim_groups, val_groups):
        offset = int(dg[0])
        span = int(dg[-1] - dg[0]) + 1
        dense = np.zeros(span, dtype=np.float64)
        dense[dg - offset] = vg
        clusters.append((offset, dense))

    return clusters


def _merge_cluster_outputs(
    results: List[Tuple[int, np.ndarray]], zero_tol: float = 0.0
) -> SparseData:
    """Merge cluster convolution outputs back into SparseData.

    Handles overlapping output ranges via scatter-add.
    Strips exact zeros (or below zero_tol) from final result.
    """
    if not results:
        return SparseData(np.array([], dtype=np.int64),
                          np.array([], dtype=np.float64))

    all_dims = []
    all_vals = []
    for offset, dense in results:
        local_dims = np.arange(len(dense), dtype=np.int64) + offset
        all_dims.append(local_dims)
        all_vals.append(dense)

    all_dims = np.concatenate(all_dims)
    all_vals = np.concatenate(all_vals)

    order = np.argsort(all_dims, kind='mergesort')
    all_dims = all_dims[order]
    all_vals = all_vals[order]

    unique_dims, inverse = np.unique(all_dims, return_inverse=True)
    summed_vals = np.zeros(len(unique_dims), dtype=np.float64)
    np.add.at(summed_vals, inverse, all_vals)

    nonzero = np.abs(summed_vals) > zero_tol
    return SparseData(unique_dims[nonzero], summed_vals[nonzero])


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
                 max_order: int = None):
        self.gap_threshold = gap_threshold
        self.zero_tol = zero_tol
        self.max_order = max_order

    # --- lifecycle ---

    def create(self, dim: int, value: float) -> SparseData:
        return SparseData(
            np.array([dim], dtype=np.int64),
            np.array([value], dtype=np.float64)
        )

    def _truncate(self, data: SparseData) -> SparseData:
        """Drop dimensions below -max_order (higher-order derivatives)."""
        if self.max_order is None:
            return data
        if len(data.dims) == 0:
            return data
        mask = data.dims >= -self.max_order
        if mask.all():
            return data
        return SparseData(data.dims[mask], data.vals[mask])

    def create_from_terms(self, dims: np.ndarray, vals: np.ndarray) -> SparseData:
        dims = np.asarray(dims, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        order = np.argsort(dims)
        return SparseData(dims[order], vals[order])

    # --- access ---

    def read_dim(self, data: SparseData, dim: int) -> float:
        idx = np.searchsorted(data.dims, dim)
        if idx < len(data.dims) and data.dims[idx] == dim:
            return float(data.vals[idx])
        return 0.0

    # FIXED: write_dim — always write the value, even if zero.
    # Previously: deleted existing dim if value==0, skipped insert if value==0.
    # Now: expressed zeros are preserved (canon rule: if zero is expressed, retain it).
    def write_dim(self, data: SparseData, dim: int, value: float) -> SparseData:
        idx = np.searchsorted(data.dims, dim)
        if idx < len(data.dims) and data.dims[idx] == dim:
            new_vals = data.vals.copy()
            new_vals[idx] = value
            return SparseData(data.dims.copy(), new_vals)
        else:
            new_dims = np.insert(data.dims, idx, dim)
            new_vals = np.insert(data.vals, idx, value)
            return SparseData(new_dims, new_vals)

    def to_arrays(self, data: SparseData) -> Tuple[np.ndarray, np.ndarray]:
        return data.dims.copy(), data.vals.copy()

    def active_dims(self, data: SparseData) -> np.ndarray:
        return data.dims.copy()

    # --- arithmetic ---

    # FIXED: add — keep all union dimensions, do NOT strip zeros.
    # Canon rule: 1-1 = |0|₀ (zero at dimension 0, dimension retained).
    def add(self, a: SparseData, b: SparseData) -> SparseData:
        """Merge-add: like merge step of merge sort, O(n+m)."""
        if len(a.dims) == 0:
            return b
        if len(b.dims) == 0:
            return a

        all_dims = np.union1d(a.dims, b.dims)

        a_idx = np.searchsorted(a.dims, all_dims)
        b_idx = np.searchsorted(b.dims, all_dims)

        a_vals = np.where(
            (a_idx < len(a.dims)) & (a.dims[np.minimum(a_idx, len(a.dims)-1)] == all_dims),
            a.vals[np.minimum(a_idx, len(a.dims)-1)],
            0.0
        )
        b_vals = np.where(
            (b_idx < len(b.dims)) & (b.dims[np.minimum(b_idx, len(b.dims)-1)] == all_dims),
            b.vals[np.minimum(b_idx, len(b.dims)-1)],
            0.0
        )

        result_vals = a_vals + b_vals
        return SparseData(all_dims, result_vals)

    def convolve(self, a: SparseData, b: SparseData) -> SparseData:
        """Clustered convolution: cluster × cluster → merge."""
        if len(a.dims) == 0 or len(b.dims) == 0:
            return SparseData(np.array([], dtype=np.int64),
                              np.array([], dtype=np.float64))

        clusters_a = _cluster_terms(a.dims, a.vals, self.gap_threshold)
        clusters_b = _cluster_terms(b.dims, b.vals, self.gap_threshold)

        results = []
        for offset_a, dense_a in clusters_a:
            for offset_b, dense_b in clusters_b:
                if len(dense_a) + len(dense_b) > _FFT_THRESHOLD:
                    conv = _fft_convolve(dense_a, dense_b)
                else:
                    conv = np.convolve(dense_a, dense_b)
                out_offset = offset_a + offset_b
                results.append((out_offset, conv))

        result = _merge_cluster_outputs(results, self.zero_tol)
        return self._truncate(result)

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
        nonzero_mask = np.abs(b.vals) > 1e-15
        if not np.any(nonzero_mask):
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        lead_dim = b.dims[nonzero_mask][-1]
        lead_val = b.vals[nonzero_mask][-1]

        # Strip near-zero terms from dividend for clean division
        a_mask = np.abs(a.vals) > 1e-15
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
            mask = np.abs(remainder.vals) > 1e-15
            remainder_dims = remainder.dims[mask]
            remainder_vals = remainder.vals[mask]

        if len(q_dims) == 0:
            return SparseData(np.array([], dtype=np.int64),
                              np.array([], dtype=np.float64))

        q_dims = np.array(q_dims, dtype=np.int64)
        q_vals = np.array(q_vals, dtype=np.float64)
        order = np.argsort(q_dims)
        return SparseData(q_dims[order], q_vals[order])

    def scalar_multiply(self, data: SparseData, scalar: float) -> SparseData:
        if scalar == 0.0:
            return SparseData(np.array([], dtype=np.int64),
                              np.array([], dtype=np.float64))
        return SparseData(data.dims.copy(), data.vals * scalar)

    def negate(self, data: SparseData) -> SparseData:
        return SparseData(data.dims.copy(), -data.vals)
