# composite/backends/dict_backend.py
# Composite Machine — Dict Backend (Pure Python, Reference)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import numpy as np
from typing import Tuple
from .base_backend import CompositeBackend


class DictData:
    """Internal storage: Python dict {dim: value}.
    Reference implementation — readable, slow."""
    __slots__ = ('terms',)

    def __init__(self, terms: dict):
        self.terms = terms

    def __repr__(self):
        parts = [f"|{v}|_{d}" for d, v in sorted(self.terms.items())]
        return "Composite(" + " + ".join(parts) + ")"


class DictBackend(CompositeBackend):
    """Pure-Python dict backend.

    Every Composite is a dict {dim: coefficient}.
    No NumPy dependency. Useful for:
    - Research and debugging
    - Validating SparseDenseBackend results
    - Environments without NumPy
    """

    def create(self, dim: int, value: float) -> DictData:
        return DictData({dim: value})

    def create_from_terms(self, dims: np.ndarray, vals: np.ndarray) -> DictData:
        return DictData({int(d): float(v) for d, v in zip(dims, vals)})

    def read_dim(self, data: DictData, dim: int) -> float:
        return data.terms.get(dim, 0.0)

    def write_dim(self, data: DictData, dim: int, value: float) -> DictData:
        new_terms = dict(data.terms)
        if value == 0.0:
            new_terms.pop(dim, None)
        else:
            new_terms[dim] = value
        return DictData(new_terms)

    def to_arrays(self, data: DictData) -> Tuple[np.ndarray, np.ndarray]:
        if not data.terms:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64))
        sorted_items = sorted(data.terms.items())
        dims = np.array([d for d, _ in sorted_items], dtype=np.int64)
        vals = np.array([v for _, v in sorted_items], dtype=np.float64)
        return dims, vals

    def active_dims(self, data: DictData) -> np.ndarray:
        return np.array(sorted(data.terms.keys()), dtype=np.int64)

    def add(self, a: DictData, b: DictData) -> DictData:
        result = dict(a.terms)
        for dim, val in b.terms.items():
            result[dim] = result.get(dim, 0.0) + val
            if result[dim] == 0.0:
                del result[dim]
        return DictData(result)

    def convolve(self, a: DictData, b: DictData) -> DictData:
        result = {}
        for d_a, v_a in a.terms.items():
            for d_b, v_b in b.terms.items():
                d_out = d_a + d_b
                result[d_out] = result.get(d_out, 0.0) + v_a * v_b
        # Strip zeros
        return DictData({d: v for d, v in result.items() if v != 0.0})

    def deconvolve(self, a: DictData, b: DictData) -> DictData:
        if not b.terms:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")

        remainder = dict(a.terms)
        b_sorted = sorted(b.terms.items())
        lead_dim, lead_val = b_sorted[-1]
        quotient = {}

        max_iter = max(len(a.terms) + len(b.terms), 50)
        for _ in range(max_iter):
            if not remainder:
                break
            r_dim = max(remainder.keys())
            r_val = remainder[r_dim]

            q_dim = r_dim - lead_dim
            q_val = r_val / lead_val
            quotient[q_dim] = q_val

            for d_b, v_b in b.terms.items():
                out_d = q_dim + d_b
                remainder[out_d] = remainder.get(out_d, 0.0) - q_val * v_b
                if abs(remainder[out_d]) < 1e-15:
                    del remainder[out_d]

        return DictData({d: v for d, v in quotient.items() if v != 0.0})

    def scalar_multiply(self, data: DictData, scalar: float) -> DictData:
        if scalar == 0.0:
            return DictData({})
        return DictData({d: v * scalar for d, v in data.terms.items()})

    def negate(self, data: DictData) -> DictData:
        return DictData({d: -v for d, v in data.terms.items()})
