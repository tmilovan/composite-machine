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

# Index k is the k-times-iterated logarithm: 0 = power, 1 = log, 2 = log log.
# The list GROWS on demand -- ln() pushes a term one index up, so the depth a
# program needs is not knowable in advance.  Order IS the dominance order, and
# it stays so at any depth: each further log outgrows nothing and is outgrown
# by everything before it.
BASIS = ["power", "log"]
WIDTH = len(BASIS)
MIN_QUOTIENT_TERMS = 50   # floor; matches the scalar backend's bound


def basis_name(k: int) -> str:
    return ("power", "log")[k] if k < 2 else f"log^{k}"


def ensure_depth(n: int) -> int:
    """Widen the basis to at least n components.  Returns the new WIDTH.

    Nothing already built becomes invalid: as_vec pads on read, so a dimension
    created at width 2 compares and adds correctly against one created at 4.
    """
    global WIDTH
    while len(BASIS) < n:
        BASIS.append(basis_name(len(BASIS)))
    WIDTH = len(BASIS)
    return WIDTH


def canon(d):
    """Canonical dimension: trailing zeros stripped, minimum length 2.

    A dimension created at width 2 and one created at width 4 must be the SAME
    KEY, or a lookup after the basis grows pads (0,1) to (0,1,0), misses the
    stored (0,1), and silently returns 0 -- which made log(1/h) compare as not
    greater than loglog(1/h), because every cross-depth read was zero.

    Stripping also makes Python's own tuple ordering the dominance order: for
    canonical forms, (0,1) > (0,0,1) and (0,1) < (0,1,1) both come out right
    without padding either side.  And it stops the basis from ratcheting: the
    stored length stays at what a term actually needs.
    """
    if not isinstance(d, tuple):
        return d
    e = len(d)
    while e > 2 and d[e - 1] == 0:
        e -= 1
    return d[:e] if e != len(d) else d


def dom_key(d, width=None):
    """Sort key giving the DOMINANCE order for a canonical dimension.

    canon() strips trailing zeros so a dimension built at width 2 is the same
    dict KEY as one built at width 4.  That is required for lookups, but it
    breaks raw tuple comparison: Python ranks a strict prefix as LESS, which is
    right when the extension is positive -- (0,0,1) is loglog(1/h), infinite --
    and backwards when it is negative, because (0,0,-1) is 1/loglog(1/h), an
    infinitesimal.  Measured: max([(0,0), (0,0,-3)]) returned (0,0,-3), so sqrt
    took a -1.2e-32 cancellation term for its leading coefficient and refused
    the whole expression.

    Comparing at equal width restores the direction automatically: extending
    with a positive component raises the dimension, with a negative one lowers
    it.  No separate pass for negatives is needed.
    """
    if width is None:
        width = WIDTH
    return as_vec(d, width)


def _common_width(dims):
    w = WIDTH
    for d in dims:
        if isinstance(d, tuple) and len(d) > w:
            w = len(d)
    return w


def dom_max(dims):
    """max by dominance order.  All keys padded to ONE common width."""
    dims = list(dims)
    w = _common_width(dims)
    return max(dims, key=lambda d: as_vec(d, w))


def dom_sorted(dims, reverse=False):
    """sorted by dominance order.  All keys padded to ONE common width."""
    dims = list(dims)
    w = _common_width(dims)
    return sorted(dims, key=lambda d: as_vec(d, w), reverse=reverse)


def as_vec(d, width: int = None) -> tuple:
    """Any dimension as a vector of `width` components (default: current WIDTH).

    `width` must be read at CALL time, not bound as a default at import -- a
    default captures the width the module happened to have when it loaded, so
    every dimension built after a widening would be silently truncated.
    """
    if width is None:
        width = WIDTH
    if isinstance(d, tuple):
        return d if len(d) == width else (d + (0,) * (width - len(d)))[:max(width, len(d))]
    return (d,) + (0,) * (width - 1)


def pair(da, db):
    """Two dimensions padded to a COMMON width, for componentwise arithmetic.

    zip() over tuples of different lengths truncates to the shorter and drops
    the trailing components in silence -- so a width-2 dim convolved with a
    width-3 one lost the deepest axis entirely, with no error.
    """
    va, vb = as_vec(da), as_vec(db)
    w = max(len(va), len(vb))
    return as_vec(da, w), as_vec(db, w)


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
        return DictData({canon(as_vec(dim)): float(value)})

    def create_from_terms(self, dims, vals) -> DictData:
        return DictData({canon(as_vec(d)): float(v) for d, v in zip(dims, vals)})

    def read_dim(self, data: DictData, dim) -> float:
        return data.terms.get(canon(as_vec(dim)), 0.0)

    def write_dim(self, data: DictData, dim, value: float) -> DictData:
        t = dict(data.terms)
        t[canon(as_vec(dim))] = value
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
        out = {canon(as_vec(d)): v for d, v in a.terms.items()}
        for d, v in b.terms.items():
            d = canon(as_vec(d))
            out[d] = out.get(d, 0.0) + v
        return DictData(out)

    def convolve(self, a: DictData, b: DictData) -> DictData:
        out = {}
        for da, va in a.terms.items():
            for db, vb in b.terms.items():
                pa, pb = pair(da, db)
                k = canon(tuple(x + y for x, y in zip(pa, pb)))
                out[k] = out.get(k, 0.0) + va * vb
        return DictData(out)

    def deconvolve(self, a: DictData, b: DictData) -> DictData:
        """Long division; dimensions SUBTRACT componentwise."""
        if not b.terms:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")
        rem = {canon(as_vec(d)): v for d, v in a.terms.items() if v != 0.0}
        bnz = {canon(as_vec(d)): v for d, v in b.terms.items() if v != 0.0}
        if not bnz:
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        b_lead = dom_max(bnz)
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
            r_dim = dom_max(rem)
            _r, _b = pair(r_dim, b_lead)
            q_dim = canon(tuple(x - y for x, y in zip(_r, _b)))
            q_val = rem[r_dim] / b_coef
            quot[q_dim] = quot.get(q_dim, 0.0) + q_val
            for d_b, v_b in bnz.items():
                _q, _d = pair(q_dim, d_b)
                o = canon(tuple(x + y for x, y in zip(_q, _d)))
                rem[o] = rem.get(o, 0.0) - q_val * v_b
                if rem[o] == 0.0:
                    del rem[o]
            rem.pop(r_dim, None)
        return DictData(quot)
