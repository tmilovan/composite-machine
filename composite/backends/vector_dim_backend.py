# composite/backends/vector_dim_backend.py
# Composite Machine — vector-valued dimension indices
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Dimensions as VECTORS over a declared basis, so scales can be named.

A scalar dimension can only say "power".  The log scale needs a dimension that
is positive but smaller than EVERY power -- log x outgrows any constant and is
outgrown by x^e for every e > 0 -- and no float sits there.  A vector does:
the power is the major component, the log the minor one, and lexicographic
comparison then gives the dominance order for free.

    dim = (power, log)        x^p (log x)^l
    scalar d is (d, 0)

Dimensions still ADD on multiplication, componentwise.  Plain `+` on Python
tuples CONCATENATES -- (-1,0)+(0,1) gives (-1,0,0,1) -- which is why feeding
tuples to DictBackend silently produced garbage rather than an error.

This backend is deliberately dict-based.  Vector dimensions are rare and the
composites carrying them are small, so they do not need the run representation;
keeping them out of SparseDenseBackend means the fast scalar path cannot
regress at all.  See BASIS for extending to further scales.
"""
import numpy as np
from typing import Tuple

from .dict_backend import DictBackend, DictData
from .base_backend import DIM_DTYPE

BASIS = ("power", "log")          # extend here; order IS the dominance order
WIDTH = len(BASIS)
MIN_QUOTIENT_TERMS = 50   # floor; matches the scalar backend's bound


def as_vec(d, width: int = WIDTH) -> tuple:
    """Any dimension as a vector of `width` components."""
    if isinstance(d, tuple):
        return d if len(d) == width else d + (0,) * (width - len(d))
    return (d,) + (0,) * (width - 1)


def is_scalarish(dims) -> bool:
    """True when every non-power component is zero -- the composite demotes."""
    return all(all(c == 0 for c in as_vec(d)[1:]) for d in dims)


def demote(d):
    """Vector dimension back to a scalar, when nothing but the power is set."""
    v = as_vec(d)
    return v[0] if all(c == 0 for c in v[1:]) else v


class VectorDimBackend(DictBackend):
    """DictBackend with componentwise dimension arithmetic."""

    VECTOR_DIMS = True

    def create(self, dim, value: float) -> DictData:
        return DictData({as_vec(dim): float(value)})

    def create_from_terms(self, dims, vals) -> DictData:
        return DictData({as_vec(d): float(v) for d, v in zip(dims, vals)})

    def read_dim(self, data: DictData, dim) -> float:
        return data.terms.get(as_vec(dim), 0.0)

    def write_dim(self, data: DictData, dim, value: float) -> DictData:
        t = dict(data.terms)
        t[as_vec(dim)] = value
        return DictData(t)

    def to_arrays(self, data: DictData) -> Tuple[np.ndarray, np.ndarray]:
        """dims is an OBJECT array of tuples, so it stays 1-D and every
        consumer that only indexes, iterates or sorts it keeps working."""
        if not data.terms:
            return (np.empty(0, dtype=object), np.array([], dtype=np.float64))
        items = sorted(data.terms.items())
        dims = np.empty(len(items), dtype=object)
        for i, (d, _) in enumerate(items):
            dims[i] = d
        return dims, np.array([v for _, v in items], dtype=np.float64)

    def active_dims(self, data: DictData) -> np.ndarray:
        out = np.empty(len(data.terms), dtype=object)
        for i, d in enumerate(sorted(data.terms)):
            out[i] = d
        return out

    def add(self, a: DictData, b: DictData) -> DictData:
        out = {as_vec(d): v for d, v in a.terms.items()}
        for d, v in b.terms.items():
            d = as_vec(d)
            out[d] = out.get(d, 0.0) + v
        return DictData(out)

    def convolve(self, a: DictData, b: DictData) -> DictData:
        out = {}
        for da, va in a.terms.items():
            da = as_vec(da)
            for db, vb in b.terms.items():
                db = as_vec(db)
                k = tuple(x + y for x, y in zip(da, db))
                out[k] = out.get(k, 0.0) + va * vb
        return DictData(out)

    def deconvolve(self, a: DictData, b: DictData) -> DictData:
        """Long division; dimensions SUBTRACT componentwise."""
        if not b.terms:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")
        rem = {as_vec(d): v for d, v in a.terms.items() if v != 0.0}
        bnz = {as_vec(d): v for d, v in b.terms.items() if v != 0.0}
        if not bnz:
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        b_lead = max(bnz)
        b_coef = bnz[b_lead]
        quot = {}
        # Long division of a non-exact quotient descends forever (1/(1+x) is an
        # infinite series), so it needs a bound.  The scalar path is capped by
        # its own arithmetic; here it is explicit.
        # Same bound the scalar backend uses: enough iterations for an exact
        # division, with a floor for the non-exact case.  A quotient whose
        # divisor has a leading coefficient smaller than its lower terms
        # DIVERGES -- each step multiplies the remainder by their ratio -- so
        # without this the loop runs until the coefficients overflow to inf.
        limit = max(len(rem) + len(bnz), MIN_QUOTIENT_TERMS)
        while rem and limit > 0:
            limit -= 1
            r_dim = max(rem)
            q_dim = tuple(x - y for x, y in zip(r_dim, b_lead))
            q_val = rem[r_dim] / b_coef
            quot[q_dim] = quot.get(q_dim, 0.0) + q_val
            for d_b, v_b in bnz.items():
                o = tuple(x + y for x, y in zip(q_dim, d_b))
                rem[o] = rem.get(o, 0.0) - q_val * v_b
                if rem[o] == 0.0:
                    del rem[o]
            rem.pop(r_dim, None)
        return DictData(quot)
