# composite/backends/dict_backend.py
# Composite Machine — Dict Backend (Pure Python, Reference)
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import numpy as np
from typing import Tuple
from .base_backend import (CompositeBackend, DIM_DTYPE, dim_cast,
                            InexactGradeError, _add_exact)


class DictData:
    """Internal storage: Python dict {dim: value}.
    Reference implementation — readable, slow."""
    __slots__ = ('terms',)

    def __init__(self, terms: dict):
        self.terms = terms

    def __repr__(self):
        parts = [f"|{v}|_{d}" for d, v in sorted(self.terms.items())]
        return "Composite(" + " + ".join(parts) + ")"


def _dim_key(d):
    """Sort key that orders scalars and vectors together, lexicographically.

    sorted() on a mixed list raises -- a tuple and a float are not comparable
    -- and a scalar d IS the vector (d, 0, ...), so padding makes the order
    total and agrees with the dominance order the vector backend uses.
    """
    return d if isinstance(d, tuple) else (d,)


def _dim_add(da, db):
    """Add two dimensions, scalar or vector, componentwise.

    Plain `+` is right for two scalars and WRONG the moment either side is a
    tuple: `1 + (0, -1)` raises, and `(0,-1) + (0,1)` CONCATENATES to
    (0,-1,0,1) rather than adding.  Both happen here in ordinary use -- a
    scalar-dimension composite meeting one that has been through ln(), which
    puts a term on the log axis -- and the crash was reached by
    (h**0.1)*(h**0.2) on this backend.

    A scalar d is the vector (d, 0, 0, ...), so the shorter side is padded
    with zeros and the result is trimmed back to a scalar when nothing but
    the power component survives, keeping scalar-only arithmetic on exactly
    the keys it had before.
    """
    ta, tb = isinstance(da, tuple), isinstance(db, tuple)
    if not ta and not tb:
        out, ok = _add_exact(float(da), float(db))
        if not ok:
            raise InexactGradeError(
                f"grade {da!r} + {db!r} is not exact in float64 (got {out!r}); "
                f"the product would carry an identifier one ulp from the one "
                f"it should share")
        return out
    va = da if ta else (da,)
    vb = db if tb else (db,)
    n = max(len(va), len(vb))
    out = tuple((va[i] if i < len(va) else 0) + (vb[i] if i < len(vb) else 0)
                for i in range(n))
    while len(out) > 1 and out[-1] == 0:
        out = out[:-1]
    return out[0] if len(out) == 1 else out


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
        return DictData({dim_cast(d): float(v) for d, v in zip(dims, vals)})

    def read_dim(self, data: DictData, dim: int) -> float:
        return data.terms.get(dim, 0.0)

    # FIXED: write_dim — always write the value, even if zero.
    # Previously: popped the dim if value==0.0.
    # Now: expressed zeros are preserved (canon rule: if zero is expressed, retain it).
    def write_dim(self, data: DictData, dim: int, value: float) -> DictData:
        new_terms = dict(data.terms)
        new_terms[dim] = value
        return DictData(new_terms)

    # --- structural predicates ---
    # The base class serves these from to_arrays(), which builds two numpy
    # arrays.  They are called on EVERY multiply and division (Zero Rule R1
    # inspects both operands), so the dict was being flattened purely to answer
    # "are you zero?".  Read it directly instead.  Semantics are the base
    # class's, unchanged: is_wholly_zero requires at least one expressed term,
    # and is_unit requires EXACTLY one term -- an expressed zero alongside it
    # still means not-unit.

    def term_count(self, data: DictData) -> int:
        return len(data.terms)

    def is_wholly_zero(self, data: DictData) -> bool:
        t = data.terms
        return len(t) > 0 and all(v == 0.0 for v in t.values())

    def is_unit(self, data: DictData) -> bool:
        t = data.terms
        if len(t) != 1:
            return False
        (dim, val), = t.items()
        return dim == 0 and val == 1.0

    def to_arrays(self, data: DictData) -> Tuple[np.ndarray, np.ndarray]:
        if not data.terms:
            return (np.array([], dtype=DIM_DTYPE),
                    np.array([], dtype=np.float64))
        sorted_items = sorted(data.terms.items(), key=lambda kv: _dim_key(kv[0]))
        keys = [d for d, _ in sorted_items]
        vals = np.array([v for _, v in sorted_items], dtype=np.float64)
        # A VECTOR dimension must stay a tuple.  np.array on tuple keys builds
        # a 2-D float array, so each dimension came back as a length-n ARRAY
        # and dim_cast's float() raised "only length-1 arrays can be converted
        # to Python scalars" -- reached by (h**0.1)*(h**0.2) on this backend,
        # one layer behind the convolve crash.  An object array keeps the
        # tuples whole; scalar-only data takes the float path exactly as before.
        if any(isinstance(d, tuple) for d in keys):
            dims = np.empty(len(keys), dtype=object)
            for i, d in enumerate(keys):
                dims[i] = d
            return dims, vals
        return np.array(keys, dtype=DIM_DTYPE), vals

    def active_dims(self, data: DictData) -> np.ndarray:
        keys = sorted(data.terms.keys(), key=_dim_key)
        if any(isinstance(d, tuple) for d in keys):
            out = np.empty(len(keys), dtype=object)
            for i, d in enumerate(keys):
                out[i] = d
            return out
        return np.array(keys, dtype=DIM_DTYPE)

    # FIXED: add — do NOT delete zero-sum dimensions.
    # Canon rule: 1-1 = |0|₀ (zero at dimension 0, dimension retained).
    def add(self, a: DictData, b: DictData) -> DictData:
        result = dict(a.terms)
        for dim, val in b.terms.items():
            result[dim] = result.get(dim, 0.0) + val
        return DictData(result)

    def convolve(self, a: DictData, b: DictData) -> DictData:
        """Composite multiplication.

        The dimensions a product constructs are exactly the Minkowski sum
        {da + db}, which the double loop below produces directly.  Every one of
        them is retained, including those whose coefficient came out zero: a
        dimension exists because the computation built it, whatever it holds.
        """
        result = {}
        for d_a, v_a in a.terms.items():
            for d_b, v_b in b.terms.items():
                d_out = _dim_add(d_a, d_b)
                result[d_out] = result.get(d_out, 0.0) + v_a * v_b
        return DictData(result)

    # FIXED: deconvolve — use highest dim with non-zero coeff as leading term.
    # With expressed zero preservation, the highest dim may have coeff 0.0,
    # which would cause division by zero / NaN in the quotient step.
    # Also clean near-zero terms from inputs before division.
    def deconvolve(self, a: DictData, b: DictData) -> DictData:
        if not b.terms:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")

        # FIXED: clean near-zero terms from dividend
        remainder = {d: v for d, v in a.terms.items() if v != 0.0}

        # FIXED: Leading term — highest dim with non-zero coeff
        b_nonzero = {d: v for d, v in b.terms.items() if v != 0.0}
        if not b_nonzero:
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        b_sorted = sorted(b_nonzero.items())
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

            for d_b, v_b in b_nonzero.items():
                out_d = q_dim + d_b
                remainder[out_d] = remainder.get(out_d, 0.0) - q_val * v_b
                if remainder[out_d] == 0.0:
                    del remainder[out_d]

            # q_val was chosen so that this term cancels exactly, so the
            # remainder at r_dim is zero by construction.  In floating point
            # the subtraction above can leave dust there instead; dropping it
            # keeps max(remainder) strictly decreasing.  Without this the loop
            # can re-select r_dim, recompute the same q_dim, and overwrite a
            # correct quotient coefficient with the shrinking residue.
            remainder.pop(r_dim, None)

        # zero-valued quotient terms are retained, as in convolve
        return DictData(quotient)

    def scalar_multiply(self, data: DictData, scalar: float) -> DictData:
        if scalar == 0.0:
            return DictData({})
        return DictData({d: v * scalar for d, v in data.terms.items()})

    def negate(self, data: DictData) -> DictData:
        return DictData({d: -v for d, v in data.terms.items()})
