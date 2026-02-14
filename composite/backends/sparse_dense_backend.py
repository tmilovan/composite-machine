# composite/backends/sparse_dense_backend.py
# Composite Machine — Clustered Sparse-Dense Backend (NumPy)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import numpy as np
from typing import Tuple, List
from .base_backend import CompositeBackend


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
    # Find split points: where gap between consecutive dims > threshold
    gaps = np.diff(dims)
    split_mask = gaps > gap_threshold
    split_indices = np.nonzero(split_mask)[0] + 1  # +1 for right edge

    # Split dims and vals at those points
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

    # Collect all (dim, val) pairs
    all_dims = []
    all_vals = []
    for offset, dense in results:
        local_dims = np.arange(len(dense), dtype=np.int64) + offset
        all_dims.append(local_dims)
        all_vals.append(dense)

    all_dims = np.concatenate(all_dims)
    all_vals = np.concatenate(all_vals)

    # Sort by dimension
    order = np.argsort(all_dims, kind='mergesort')
    all_dims = all_dims[order]
    all_vals = all_vals[order]

    # Sum duplicates (overlapping cluster outputs)
    unique_dims, inverse = np.unique(all_dims, return_inverse=True)
    summed_vals = np.zeros(len(unique_dims), dtype=np.float64)
    np.add.at(summed_vals, inverse, all_vals)

    # Strip zeros
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

    def __init__(self, gap_threshold: int = 64, zero_tol: float = 0.0):
        self.gap_threshold = gap_threshold
        self.zero_tol = zero_tol

    # --- lifecycle ---

    def create(self, dim: int, value: float) -> SparseData:
        return SparseData(
            np.array([dim], dtype=np.int64),
            np.array([value], dtype=np.float64)
        )

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

    def write_dim(self, data: SparseData, dim: int, value: float) -> SparseData:
        idx = np.searchsorted(data.dims, dim)
        if idx < len(data.dims) and data.dims[idx] == dim:
            # Overwrite existing
            new_vals = data.vals.copy()
            new_vals[idx] = value
            if value == 0.0:
                # Remove zero term
                return SparseData(
                    np.delete(data.dims, idx),
                    np.delete(new_vals, idx)
                )
            return SparseData(data.dims.copy(), new_vals)
        else:
            # Insert new term
            if value == 0.0:
                return data  # don't insert zeros
            new_dims = np.insert(data.dims, idx, dim)
            new_vals = np.insert(data.vals, idx, value)
            return SparseData(new_dims, new_vals)

    def to_arrays(self, data: SparseData) -> Tuple[np.ndarray, np.ndarray]:
        return data.dims.copy(), data.vals.copy()

    def active_dims(self, data: SparseData) -> np.ndarray:
        return data.dims.copy()

    # --- arithmetic ---

    def add(self, a: SparseData, b: SparseData) -> SparseData:
        """Merge-add: like merge step of merge sort, O(n+m)."""
        if len(a.dims) == 0:
            return b
        if len(b.dims) == 0:
            return a

        # Union of dimensions
        all_dims = np.union1d(a.dims, b.dims)

        # Lookup values from each
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

        # Strip zeros
        nonzero = np.abs(result_vals) > self.zero_tol
        return SparseData(all_dims[nonzero], result_vals[nonzero])

    def convolve(self, a: SparseData, b: SparseData) -> SparseData:
        """Clustered convolution: cluster × cluster → merge.

        1. Cluster both operands by proximity
        2. For each (cluster_a, cluster_b) pair:
           - np.convolve their local dense arrays
           - output offset = offset_a + offset_b
        3. Merge all outputs back into sparse form

        Mathematically exact — distributive property guarantees
        sum of partial convolutions equals full convolution.
        """
        if len(a.dims) == 0 or len(b.dims) == 0:
            return SparseData(np.array([], dtype=np.int64),
                              np.array([], dtype=np.float64))

        clusters_a = _cluster_terms(a.dims, a.vals, self.gap_threshold)
        clusters_b = _cluster_terms(b.dims, b.vals, self.gap_threshold)

        results = []
        for offset_a, dense_a in clusters_a:
            for offset_b, dense_b in clusters_b:
                conv = np.convolve(dense_a, dense_b)
                out_offset = offset_a + offset_b
                results.append((out_offset, conv))

        return _merge_cluster_outputs(results, self.zero_tol)

    def deconvolve(self, a: SparseData, b: SparseData) -> SparseData:
        """Polynomial long division in sparse form.

        Computes Q such that A = Q * B (approximately).
        Works term-by-term from highest dimension down.
        """
        if len(b.dims) == 0:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")

        # Leading term of divisor
        lead_dim = b.dims[-1]
        lead_val = b.vals[-1]

        # Work on a copy of the dividend
        remainder_dims = a.dims.copy()
        remainder_vals = a.vals.copy()

        q_dims = []
        q_vals = []

        max_iter = max(len(a.dims) + len(b.dims), 50)  # safety
        for _ in range(max_iter):
            if len(remainder_dims) == 0:
                break

            # Highest remaining term
            r_dim = remainder_dims[-1]
            r_val = remainder_vals[-1]

            # Quotient term
            q_dim = r_dim - lead_dim
            q_val = r_val / lead_val
            q_dims.append(q_dim)
            q_vals.append(q_val)

            # Subtract q_term * b from remainder
            sub_dims = b.dims + q_dim
            sub_vals = b.vals * q_val

            # Merge-subtract
            remainder = self.add(
                SparseData(remainder_dims, remainder_vals),
                SparseData(sub_dims, -sub_vals)
            )
            remainder_dims = remainder.dims
            remainder_vals = remainder.vals

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
