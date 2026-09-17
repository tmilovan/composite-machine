# Composite Machine — Automatic Calculus via Dimensional Arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing available. Contact: tmilovan@fwd.hr
"""
composite_lib.py — Unified Calculus Library (Fixed v3: Expressed Zero Preservation)
====================================================================================
All operations use composite arithmetic. No plain-number fast paths
in transcendental functions. Integration accumulates as Composite.

v3 changes:
  - Composite(0) now produces |0|₀ (expressed zero at dim 0), not empty
  - Dict construction preserves zero-valued coefficients
  - Empty composite repr changed from |0|₀ to ∅

Usage:
    from composite_lib import *

    # Derivatives
    derivative(lambda x: x**2, at=3)           # → 6
    derivative(lambda x: sin(x), at=0)         # → 1
    nth_derivative(lambda x: x**5, n=3, at=2)  # → 120

    # Limits
    limit(lambda x: sin(x)/x, as_x_to=0)                    # → 1
    limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)          # → 4
    limit(lambda x: (1 - cos(x))/x**2, as_x_to=0)           # → 0.5

    # All derivatives at once
    all_derivatives(lambda x: exp(x), at=0, up_to=5)  # → [1,1,1,1,1,1]

    # Direct composite computation
    h = ZERO
    x = R(3) + h
    result = x**2
    print(result)        # |9|₀ + |6|₋₁ + |1|₋₂
    print(result.st())   # 9 (function value)
    print(result.d(1))   # 6 (first derivative)
    print(result.d(2))   # 2 (second derivative)

Author: Toni Milovan
"""

import math
import contextlib as _contextlib
from typing import Callable, List, Optional, Union
import struct
import numpy as np

from composite.backends import get_backend
from composite.backends.base_backend import DIM_DTYPE, dim_cast

# =============================================================================
# EXCEPTIONS
# =============================================================================

class LimitDoesNotExistError(ValueError):
    """Raised when a limit provably does not exist."""
    pass

class LimitUndecidableError(ValueError):
    """Raised when composite arithmetic cannot determine the limit."""
    pass

class CompositionError(TypeError):
    """Raised when a function is not composable with composite arithmetic."""
    pass

# =============================================================================
# CORE: COMPOSITE NUMBER CLASS
# =============================================================================

def _is_wholly_zero(c):
    """True when every expressed coefficient is zero — the composite is a zero."""
    return c._backend.is_wholly_zero(c._data)


def _is_unit(c):
    """True for |1|_0, the multiplicative identity."""
    return c._backend.is_unit(c._data)


def _operands(a, b):
    """Prepare two composites for an operation.

    Brings them onto a single backend first -- set_backend() may have been
    called after one of them was built, and the module constants ZERO / INF
    are rebuilt on switch but references captured by `from ... import ZERO`
    are not -- then applies R1 to each.
    """
    if type(a._data) is not type(b._data):
        # Convert toward the RICHER representation.  A vector-dimension backend
        # can hold a scalar dimension d as (d, 0, ...); the reverse loses the
        # scale components, and numpy cannot even hold the tuples.  So when the
        # two differ, the vector side wins regardless of which operand it is.
        if getattr(b._backend, "VECTOR_DIMS", False) and not getattr(
                a._backend, "VECTOR_DIMS", False):
            dims, vals = a._backend.to_arrays(a._data)
            a = Composite._wrap(b._backend.create_from_terms(dims, vals),
                                b._backend, demote=False)
        else:
            dims, vals = b._backend.to_arrays(b._data)
            b = Composite._wrap(a._backend.create_from_terms(dims, vals),
                                a._backend, demote=False)
    return _r1(a), _r1(b)


def _r1(c):
    """R1 — a zero used as an OPERAND converts: |0|_d becomes |1|_(d-1).

    Only a composite that is wholly zero is a zero.  A zero sitting among
    nonzero terms is a term of the number, not an operand, so nothing happens
    to it (R2).  When several dimensions are zero, only the lowest converts;
    the composite then holds a nonzero and the rest are terms.
    """
    if not _is_wholly_zero(c):
        return c
    dims, vals = c._backend.to_arrays(c._data)
    dims = dims.copy()
    vals = vals.copy()
    dims[0] = dims[0] - 1          # to_arrays is sorted ascending: [0] is lowest
    vals[0] = 1.0
    return Composite._wrap(c._backend.create_from_terms(dims, vals),
                           c._backend, demote=False)


_ZERO_SEED_MSG = (
    "bare Python 0 used with `{op}` on a Composite. A Python zero meeting a "
    "Composite ALWAYS converts to the composite zero |1|_-1 (R1) -- that is the "
    "rule and it is what just happened. If you meant an accumulator that has not "
    "added a term yet, that is NOTHING, not zero (R6): use Composite({{}}) or seed "
    "with the first term. `acc = 0` leaves the dimension-0 value correct while "
    "injecting |1|_-1 into every derivative -- 321 instead of 257 for a "
    "5th-degree polynomial. Write R(0) or ZERO to silence this when the "
    "composite zero is what you want."
)


def _warn_zero_seed(other, op):
    """R6 made audible.  The SEMANTICS are not in question here.

    A Python 0 meeting a Composite converts to |1|_-1.  That is R1, it is
    deliberate, and it is why `0 * t` is a perfectly good way to write the zero
    component of a path -- this repo's own curve tests do exactly that.

    What is missing is visibility.  `acc = 0` is the most natural seed anyone
    writes, and it silently asks for the composite zero instead of an empty
    accumulator.  The dimension-0 value stays correct, so the result looks
    right; only the derivatives are wrong.  That is the worst failure shape
    there is, so it gets a warning.

    A WARNING, not an error: `0 * t` is legitimate, and raising also broke the
    exp(-1/x^2) limits by firing inside the limit machinery where the exception
    became a nan.  Nothing about the arithmetic changes.
    """
    if type(other) in (int, float) and other == 0:
        import warnings
        warnings.warn(_ZERO_SEED_MSG.format(op=op), stacklevel=3)


class Composite:
    """
    Composite number: |coefficient|_dimension

    Represents numbers with dimensional structure where:
        dimension 0  = real numbers
        dimension -1 = infinitesimals (structural zero)
        dimension -2 = second-order infinitesimals
        dimension +1 = infinities (structural infinity)

    Examples:
        |5|₀      = real number 5
        |1|₋₁     = structural zero (infinitesimal h)
        |1|₁      = structural infinity
        |3|₀+|2|₋₁ = 3 + 2h (3 plus 2 infinitesimals)
    """

    # _complete: highest Taylor order this value is COMPLETE to, or None for
    # "exact" -- a literal, a seeded variable, a polynomial, a deconvolution.
    # It cannot be inferred from the coefficients: _seeded(t) is exact with max
    # order 1, sin(x*x) is truncated with max order 11, and both merely look
    # like "max order K".  So it is carried.
    __slots__ = ['_data', '_backend', '_complete']

    def __init__(self, coefficients=None, _data=None):
        self._backend = get_backend()
        self._complete = None          # exact unless a series says otherwise

        if _data is not None:
            # Internal fast path: created by arithmetic ops
            self._data = _data
            return

        if coefficients is None:
            self._data = self._backend.create_from_terms(
                np.array([], dtype=DIM_DTYPE),
                np.array([], dtype=np.float64))
        elif isinstance(coefficients, (int, float)):
            # FIXED: Composite(0) → |0|₀ (expressed zero at dim 0)
            # Previously: Composite(0) → {} (empty, indistinguishable from Composite())
            # Now: any numeric input creates an expressed dimension 0
            self._data = self._backend.create(0, float(coefficients))
        elif isinstance(coefficients, dict):
            # FIXED: keep expressed zeros — if a dimension is in the dict,
            # it was expressed, even if its coefficient is 0.0
            # Previously: {k: v for k, v in ... if v != 0} stripped them
            if coefficients:
                sorted_dims = sorted(coefficients.keys())
                if sorted_dims and isinstance(sorted_dims[0], tuple):
                    # VECTOR dimensions (power, log, ...).  np.array would turn
                    # these into a 2-D float grid and the rows are unhashable;
                    # an object array keeps them 1-D and keeps the tuples whole.
                    dims = np.empty(len(sorted_dims), dtype=object)
                    for _i, _d in enumerate(sorted_dims):
                        dims[_i] = _d
                else:
                    dims = np.array(sorted_dims, dtype=DIM_DTYPE)
                vals = np.array([coefficients[d] for d in sorted_dims], dtype=np.float64)
                self._data = self._backend.create_from_terms(dims, vals)
            else:
                self._data = self._backend.create_from_terms(
                    np.array([], dtype=DIM_DTYPE),
                    np.array([], dtype=np.float64))
        else:
            raise TypeError(f"Cannot create Composite from {type(coefficients)}")

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------

    @classmethod
    def _wrap(cls, data, backend=None, demote=True, complete=None):
        """Create a Composite directly from backend data. No dict parsing.

        `backend` must be given whenever the data was built by a backend that
        is not the active one -- otherwise the object carries that backend's
        data under the global backend's methods, and the next call reaches for
        an attribute the data does not have.
        """
        be = backend if backend is not None else get_backend()
        if demote and be.VECTOR_DIMS:
            data, be = _demote(data, be)
        obj = cls.__new__(cls)
        obj._backend = be
        obj._data = data
        obj._complete = complete
        return obj

    # -------------------------------------------------------------------------
    # Backward compatibility: .c property
    # -------------------------------------------------------------------------

    @property
    def c(self):
        """Backward-compatible dict view. Returns {dim: coeff} dict.

        WARNING: This reconstructs a dict from backend data on every call.
        Use read_dim / to_arrays for new code. This exists only so that
        existing transcendental functions, antiderivative(), show(), and
        TracedComposite keep working without changes.
        """
        dims, vals = self._backend.to_arrays(self._data)
        return {dim_cast(d): float(v) for d, v in zip(dims, vals)}

    # -------------------------------------------------------------------------
    # Constructors (UNCHANGED)
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls):
        """Structural zero: |1|₋₁ (infinitesimal)"""
        return cls({-1: 1.0})

    @classmethod
    def infinity(cls):
        """Structural infinity: |1|₁"""
        return cls({1: 1.0})

    @classmethod
    def real(cls, value):
        """Real number: |value|₀"""
        if isinstance(value, Composite):
            # float(Composite) is st(), so this would drop every dimension but
            # 0 and return a plausible wrong answer.  That is how
            # power(1+x, 1/x) returned 1.0: the exponent 1/x is |1|_1, whose
            # standard part is 0, so it computed x^0.
            raise TypeError(
                "Composite.real() takes a real number, not a Composite -- "
                "converting one collapses it to its standard part. If the "
                "value is already a composite, use it directly.")
        return cls({0: float(value)})

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        dims, vals = self._backend.to_arrays(self._data)

        if len(dims) == 0:
            # FIXED: was "|0|₀" — now distinguishable from expressed |0|₀
            return "∅"

        sub = "₀₁₂₃₄₅₆₇₈₉"
        def fmt_dim(n):
            if isinstance(n, tuple):
                # Vector dimension over the declared basis: no subscript glyphs,
                # so render it plainly rather than trying to coerce it to a float.
                return "_(" + ",".join(f"{c:g}" for c in n) + ")"
            f = float(n)
            if not f.is_integer():
                # A fractional dimension -- sqrt of an odd dimension produces
                # these.  There are no subscript glyphs for them, so render
                # plainly rather than silently truncating to an integer.
                return "_" + f"{f:g}"
            n = int(f)
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            else:
                return "₋" + ''.join(sub[int(d)] for d in str(-n))

        def fmt_coeff(c):
            c = float(c)
            if c == int(c):
                return str(int(c))
            return f"{c:.6g}"

        # Highest dimension first (descending)
        parts = [f"|{fmt_coeff(vals[i])}|{fmt_dim(dims[i])}"
                 for i in range(len(dims) - 1, -1, -1)]
        return " + ".join(parts)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self):
        """Serialize to JSON-safe dict."""
        return {str(k): v for k, v in self.c.items()}

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dict. Accepts string, int or float keys."""
        return cls({dim_cast(float(k)): v for k, v in d.items()})

    def to_bytes(self):
        """Serialize to compact binary format."""
        import struct
        parts = []
        for dim, coeff in self.c.items():
            parts.append(struct.pack('<id', dim, coeff))
        return b''.join(parts)

    @classmethod
    def from_bytes(cls, data):
        """Deserialize from binary. Inverse of to_bytes()."""
        import struct
        c = {}
        for i in range(0, len(data), 12):
            dim, coeff = struct.unpack('<id', data[i:i+12])
            c[dim] = coeff
        return cls(c)

    def to_array(self, dims):
        """Extract coefficients at fixed dimensions as a flat list."""
        return [self.c.get(d, 0.0) for d in dims]

    @classmethod
    def from_array(cls, values, dims):
        """Reconstruct from flat list + dimension map."""
        # zero coefficients are retained: a dimension exists because it
        # was constructed, whatever it holds
        return cls({d: v for d, v in zip(dims, values)})

    def to_json(self):
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s):
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(s))

    # -------------------------------------------------------------------------
    # Arithmetic operations
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """Addition never shifts dimensions (R4).  A zero OPERAND converts (R1);
        a zero TERM does not (R2)."""
        _warn_zero_seed(other, "+")
        if isinstance(other, (int, float)):
            other = Composite({0: 0.0}) if other == 0 else Composite(float(other))
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        return Composite._wrap(a._backend.add(a._data, b._data), a._backend,
                               complete=_min_complete(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction never shifts dimensions (R4).

        a - a leaves |0|_d: the dimension was constructed by both operands, so
        it exists and holds zero.  No special case is needed -- |0|_a - |0|_a
        giving 0**(a+1) follows from R1 plus this rule, not from a branch here.
        """
        _warn_zero_seed(other, "-")
        if isinstance(other, (int, float)):
            other = Composite({0: 0.0}) if other == 0 else Composite(float(other))
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        return Composite._wrap(a._backend.add(a._data, a._backend.negate(b._data)),
                               a._backend,
                               complete=_min_complete(self, other))

    def __rsub__(self, other):
        _warn_zero_seed(other, "-")
        left = Composite({0: 0.0}) if other == 0 else Composite(float(other))
        return left.__sub__(self)

    def __neg__(self):
        return Composite._wrap(self._backend.negate(self._data), self._backend,
                               complete=self._complete)

    def __mul__(self, other):
        """Multiplication: dimensions add, coefficients multiply.

        A zero OPERAND converts first (R1), so |0|_0 x |5|_0 = |1|_-1 x |5|_0
        = |5|_-1 -- the value is translated, not annihilated.  Multiplying by
        |1|_0 is the identity (R3).
        """
        _warn_zero_seed(other, "*")
        if isinstance(other, (int, float)):
            if other == 0:
                other = Composite({0: 0.0})
            else:
                return Composite._wrap(
                    self._backend.scalar_multiply(self._data, float(other)),
                    self._backend, complete=self._complete)
        if not isinstance(other, Composite):
            return NotImplemented
        if _is_unit(other):
            return self
        if _is_unit(self):
            return other
        a, b = _operands(self, other)
        return Composite._wrap(
            _truncate_dims(a._backend, a._backend.convolve(a._data, b._data)),
            a._backend, complete=_min_complete(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: dimensions subtract, coefficients divide.

        A zero OPERAND converts first (R1), on either side.  Dividing by
        |1|_0 is the identity (R3).
        """
        if isinstance(other, (int, float)):
            if other == 0:
                other = Composite({0: 0.0})
            else:
                return Composite._wrap(
                    self._backend.scalar_multiply(self._data, 1.0 / other),
                    complete=self._complete)
        if not isinstance(other, Composite):
            return NotImplemented
        if _is_unit(other):
            return self

        a, b = _operands(self, other)
        b_dims = b._backend.active_dims(b._data)

        if len(b_dims) == 0:
            raise LimitDoesNotExistError(
                "Division by nothing (empty composite). "
                "Denominator is indeterminate — limit does not exist.")

        # Single-term divisor -> a pure dimension shift.  Every term of the
        # dividend moves together, zeros included: they shift and stay zero.
        if len(b_dims) == 1:
            _bd = b_dims[0]
            div_dim = _bd if isinstance(_bd, tuple) else float(_bd)
            div_coeff = b._backend.read_dim(b._data, div_dim)
            my_dims, my_vals = a._backend.to_arrays(a._data)
            if isinstance(div_dim, tuple):
                # Vector dimensions subtract componentwise; object arrays do not
                # broadcast `-`, so shift each one explicitly.
                shifted = [tuple(x - y for x, y in zip(d, div_dim)) for d in my_dims]
            else:
                shifted = my_dims - div_dim
            return Composite._wrap(
                a._backend.create_from_terms(shifted, my_vals / div_coeff),
                a._backend, complete=_min_complete(self, other))

        result = a._backend.deconvolve(a._data, b._data)
        out = Composite._wrap(_truncate_dims(a._backend, result), a._backend,
                              complete=_min_complete(self, other))
        # A multi-term divisor generally yields a NON-TERMINATING quotient that
        # the backend cuts at a fixed length: 1/(1-x) comes back as 50 correct
        # coefficients, not as a closed form.  Every order produced is right,
        # but there is no order beyond them -- so the quotient is complete TO
        # what it produced, never "exact".  Reporting exact here is what made
        # (1/x)*x = 1 diverge at order 50.
        got = [_dim_order(d) for d in out.c if _dim_order(d) >= 0]
        if got:
            out._complete = _tighter(out._complete, max(got))
        return out

    def __rtruediv__(self, other):
        # other / self -- the operands are reversed, so no unit-divisor
        # short-circuit applies here.
        left = Composite({0: 0.0}) if other == 0 else Composite(float(other))
        return left.__truediv__(self)

    def __abs__(self):
        """Absolute value of the standard part."""
        return abs(self.st())

    def __float__(self):
        """Float conversion returns standard part."""
        return float(self.st())

    def __int__(self):
        """Int conversion returns int of standard part."""
        return int(self.st())

    def __pow__(self, n):
        """Power: integer via repeated multiplication, otherwise exp(n*ln(self))."""
        if isinstance(n, int):
            if n == 0:
                return Composite({0: 1})
            if n < 0:
                return Composite({0: 1}) / (self ** (-n))
            result = Composite({0: 1})
            for _ in range(n):
                result = result * self
            return result
        if isinstance(n, float):
            return exp(Composite(n) * ln(self))
        if isinstance(n, Composite):
            return exp(n * ln(self))
        raise TypeError(f"Power exponent must be int, float, or Composite, got {type(n)}")

    # -------------------------------------------------------------------------
    # Extraction methods
    # -------------------------------------------------------------------------

    def st(self):
        """Standard part: coefficient at dimension 0"""
        return self._backend.read_dim(self._data, 0)

    def coeff(self, dim):
        """Get coefficient at specific dimension"""
        return self._backend.read_dim(self._data, dim)

    def d(self, n=1):
        """Extract nth derivative."""
        return self._backend.read_dim(self._data, -n) * math.factorial(n)

    def __format__(self, fmt):
        """Support format strings by formatting the standard part."""
        if fmt:
            return format(self.st(), fmt)
        return repr(self)

    def max_positive_dim(self):
        """Return the highest positive dimension, or None if none exist."""
        dims, vals = self._backend.to_arrays(self._data)
        # A vector dimension is positive when its POWER component is.
        _p = lambda d: d[0] if isinstance(d, tuple) else d
        pos = [dim_cast(d) for d, v in zip(dims, vals) if _p(d) > 0 and v != 0]
        return max(pos) if pos else None

    def coeffs_dict(self):
        """Return {dim: coeff} dict for all non-zero dimensions."""
        dims, vals = self._backend.to_arrays(self._data)
        return {dim_cast(d): float(v) for d, v in zip(dims, vals)}

    # -------------------------------------------------------------------------
    # Simplified integration operators (dimensional shifts)
    # -------------------------------------------------------------------------

    def eval_taylor(self, h_value):
        """Evaluate Taylor polynomial by substituting h → h_value."""
        dims, vals = self._backend.to_arrays(self._data)
        _p = lambda d: d[0] if isinstance(d, tuple) else d
        return sum(float(v) * h_value ** (-float(_p(d)))
                   for d, v in zip(dims, vals) if _p(d) < 0)

    def integrate_step(self, dx):
        """Integrate over interval [x, x+dx]."""
        Fx = antiderivative(self)
        return Fx.eval_taylor(dx)

    # -------------------------------------------------------------------------
    # Comparison (lexicographic by dimension)
    # -------------------------------------------------------------------------

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = _compare(self, other)
        if isinstance(result, float) and math.isnan(result):
            return False
        return result == 0

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) < 0

    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) <= 0

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) > 0

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) >= 0

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = _compare(self, other)
        if isinstance(result, float) and math.isnan(result):
            return True
        return result != 0

def _compare(a, b):
    """Lexicographic comparison by dimension (highest first).

    Operands are read through R1 first, so the two spellings of a zero compare
    equal: |0|_d and |1|_(d-1) are one number, and <0|_-1> == ZERO**2.
    """
    # _operands, not _r1 alone: comparing across representations needs the
    # same promotion arithmetic gets.  Without it np.union1d below sorts a
    # float dim array against a tuple one and raises -- so log(1/r) could not
    # be ordered against 1/sqrt(r), which is exactly the asymptotic question
    # the ordering exists to answer.
    a, b = _operands(a, b)
    a_dims, a_vals = a._backend.to_arrays(a._data)
    b_dims, b_vals = b._backend.to_arrays(b._data)

    all_dims = np.union1d(a_dims, b_dims)
    if len(all_dims) == 0:
        return 0

    for dim in reversed(all_dims):
        _d = dim if isinstance(dim, tuple) else float(dim)
        ca = a._backend.read_dim(a._data, _d)
        cb = b._backend.read_dim(b._data, _d)
        if math.isnan(ca) or math.isnan(cb):
            return float('nan')
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0


# =============================================================================
# CONVENIENCE SHORTCUTS
# =============================================================================

def R(x):
    """Create real number |x|₀, or ZERO (|1|₋₁) if x == 0.

    |0|₀ and |1|₋₁ are the same number (zero) at different dimensions.
    R(0) returns the canonical form |1|₋₁.
    A raw Python scalar 0 (via Composite(0)) keeps the |0|₀ form.
    """
    if isinstance(x, Composite):
        raise TypeError(
            "R() takes a real number. Passing a Composite silently collapses it "
            "to its standard part -- float(|2|_0 + |1|_-1) is 2.0 -- which is how "
            "power(1+x, 1/x) returned 1.0. If the value is already a composite, "
            "use it directly.")
    if x == 0:
        return Composite.zero()
    return Composite.real(x)

ZERO = Composite.zero()       # |1|₋₁ (infinitesimal)
INF = Composite.infinity()    # |1|₁ (infinity)
h = ZERO                      # Alias: h is the infinitesimal

def _refresh_constants():
    """Rebuild the module constants under the active backend.

    Called by backends.config.set_backend().  ZERO / INF / h are built at
    import time, so without this a later set_backend() would leave them on the
    old backend and every expression touching them would mix representations.
    """
    global ZERO, INF, h
    ZERO = Composite.zero()
    INF = Composite.infinity()
    h = ZERO


MAX_ACTIVE_DIMS = 60


def _truncate_dims(backend, data):
    """Keep the MAX_ACTIVE_DIMS dimensions closest to dim 0.

    Prevents dimension explosion in deep composition chains.
    Without this, sin(atan(sin(atan(x)))) produces 100k+ dims
    with overflow that corrupts the derivative tower.
    """
    # Check the count BEFORE materialising: this runs on every multiply, and
    # the overwhelming majority of products are under the cap, so flattening
    # first meant paying for the flat form purely to discover it was not needed.
    if backend.term_count(data) <= MAX_ACTIVE_DIMS:
        return data
    dims, vals = backend.to_arrays(data)
    # Sort by distance from dim 0, keep closest.
    if dims.dtype == object:
        # VECTOR dimensions: np.abs cannot take a tuple.  Rank by magnitude in
        # dominance order -- the power component first, then the log -- so the
        # terms kept are the ones nearest dimension zero in the same sense the
        # comparison uses.  Without this the whole truncation path raised
        # TypeError the moment a log-scale composite grew past the cap, which
        # is only reachable at the DEFAULT cap and so went unseen while every
        # standalone check ran with MAX_ACTIVE_DIMS raised.
        order = np.array(sorted(range(len(dims)),
                                key=lambda i: tuple(abs(c) for c in dims[i])))
    else:
        order = np.argsort(np.abs(dims))
    keep = order[:MAX_ACTIVE_DIMS]
    keep.sort()  # restore dimension order
    return backend.create_from_terms(dims[keep], vals[keep])


def _seeded(at):
    """Evaluation point seeded with infinitesimal for derivative extraction.

    R(0) = ZERO already carries the infinitesimal, so R(0) + ZERO
    would double-seed to |2|₋₁. This helper avoids that.
    """
    x = R(at)
    if at == 0:
        return x
    return x + ZERO

def set_max_order(n: int = None):
    """Set global MAX truncation order."""
    backend = get_backend()
    backend.max_order = n


def get_max_order() -> int:
    """Get current MAX truncation order. None = unlimited."""
    backend = get_backend()
    return getattr(backend, 'max_order', None)


# Global minimum terms for transcendentals. Set by nth_derivative/all_derivatives
# so that Taylor expansions inside black-box functions use enough terms.
_min_terms = [0]

def _effective_terms(default):
    """Return max(default, global minimum) for Taylor series length."""
    return max(default, _min_terms[0])


@_contextlib.contextmanager
def _derivative_scope(order, terms):
    """Raise expansion depth AND the dimension cap for one extraction.

    _truncate_dims keeps the MAX_ACTIVE_DIMS dimensions CLOSEST TO ZERO, which
    is the right policy for a derivative jet -- but it also caps the reachable
    order at MAX_ACTIVE_DIMS - 1, and above that the extraction returned 0.0
    with no error at all.  Measured: order 59 correct, order 60 silently wrong.

    So raise the cap to cover what was actually asked for and restore it after.
    The guard stays in force for everything outside a derivative call, which is
    what it was written for (deep composition chains like sin(atan(sin(atan(x))))
    that otherwise reach 100k+ dims).
    """
    global MAX_ACTIVE_DIMS
    old_min, old_cap = _min_terms[0], MAX_ACTIVE_DIMS
    _min_terms[0] = max(terms, order + 2)
    MAX_ACTIVE_DIMS = max(MAX_ACTIVE_DIMS, order + 8)
    try:
        yield
    finally:
        _min_terms[0] = old_min
        MAX_ACTIVE_DIMS = old_cap

# =============================================================================
# TAYLOR SERIES FOR TRANSCENDENTAL FUNCTIONS
# =============================================================================

def _has_positive_dims(x):
    """Check if a composite has any positive-dimension components."""
    return x.max_positive_dim() is not None


def _dim_order(d):
    """Taylor order carried by a dimension: -power, scalar or vector dim alike.

    int(d) is wrong here -- dimensions are float64 and int(-0.5) == 0, which is
    how a fractional dimension once read as 'no infinitesimal part' and sent ln
    down the wrong branch.
    """
    p = d[0] if isinstance(d, tuple) else d
    return -p


def _complete_order(h_terms, terms):
    """Highest Taylor order the h-power loop actually finishes.

    h**n starts at order n*m, where m is the LOWEST order present in h.  After
    forming powers 1..terms-1, every order at or below m*(terms-1) has received
    all of its contributions; every order above it is missing the contributions
    of the powers never formed.

    A term counter cannot see this distinction -- only the dimension can.  When
    h spans a single order (h = e, the common case: exp(_seeded(t))) the bound
    is terms-1 and nothing is discarded.  When h spans several -- which is what
    any composed argument gives you, exp(-(x*x)) having h = -2*mid*e - e**2 --
    the loop reaches orders it cannot complete, and returning them handed back
    partial sums wearing the shape of finished coefficients: measured against
    exact Hermite values, wrong by a factor of ~1e3.  They were invisible
    because taylor_coefficients raises the depth through _derivative_scope and
    so never reads past the boundary; the consumers that sweep the whole
    coefficient dict -- antiderivative, and integrate through it -- did.
    """
    m = min(_dim_order(d) for d in h_terms)
    if m <= 0:
        return None
    return m * (terms - 1)


def _infinitesimal_terms(x):
    """The strictly-infinitesimal part of x, as a {dim: coeff} dict."""
    return {d: c for d, c in x.c.items() if c != 0.0 and _dim_order(d) > 0}


def _min_complete(*xs):
    """Lowest CARRIED completeness among the operands; None if all are exact.

    This is the propagation rule for every arithmetic op: a sum, product or
    quotient is complete only as far as its least complete operand.
    """
    best = None
    for x in xs:
        v = getattr(x, "_complete", None) if isinstance(x, Composite) else None
        if v is None:
            continue
        best = v if best is None else min(best, v)
    return best


def _tighter(*bounds):
    """The strictest of several bounds, ignoring None (= no bound)."""
    best = None
    for b in bounds:
        if b is None:
            continue
        best = b if best is None else min(best, b)
    return best


def _shared_order(*xs):
    """Highest order a COMBINATION of truncated series is valid to.

    A product or quotient is only as complete as its least complete operand.
    sin and cos are each sound to order 11 at the default depth; their quotient
    comes back from the deconvolution carrying orders up to 49, and every one
    above 11 is built from coefficients neither operand ever had.  That is how
    tan, tanh, asin and acos each returned dozens of finished-looking orders
    they could not support -- asin reached 785, with a backend overflow warning
    on the way there.
    """
    best = None
    for x in xs:
        if not isinstance(x, Composite):
            continue
        got = [_dim_order(d) for d in x.c if _dim_order(d) >= 0]
        if not got:
            continue
        m = max(got)
        best = m if best is None else min(best, m)
    return best


def _truncate_order(result, order):
    """Drop the orders above `order`, keeping every dimension at or below it.

    Reads the backend arrays rather than the .c dict.  Building that dict was
    ~20% of an exp evaluation once the series loop was fused -- it allocates a
    Python dict and calls dim_cast per term, to answer a question numpy can
    answer with one comparison.
    """
    if order is None:
        return result
    dims = result._backend.active_dims(result._data)
    if getattr(dims, "dtype", None) is not None and dims.dtype != object:
        over = (-dims) > order
        if not over.any():
            result._complete = _tighter(result._complete, order)
            return result
        keep = ~over
        d2 = dims[keep]
        _, v = result._backend.to_arrays(result._data)
        out = Composite._wrap(
            result._backend.create_from_terms(d2, v[keep]), result._backend,
            complete=_tighter(getattr(result, "_complete", None), order))
        return out
    extra = [d for d in result.c if _dim_order(d) > order]
    if not extra:
        # Nothing to drop, but the bound still HOLDS and must be recorded --
        # otherwise a caller downstream reads "exact" off a truncated series.
        result._complete = _tighter(result._complete, order)
        return result
    out = _like(result, {d: v for d, v in result.c.items()
                         if _dim_order(d) <= order})
    out._complete = _tighter(getattr(result, "_complete", None), order)
    return out


def _bounded_at_inf(func, x):
    """Evaluate a bounded transcendental at an infinite composite argument.

    For monotonic bounded functions (atan, tanh): math.func(±inf) returns
    the correct asymptotic value (e.g. atan(inf) = π/2).  Result via R().

    For oscillatory functions (sin, cos): math.func(±inf) raises ValueError.
    The result is genuinely indeterminate — return ∅ (empty composite,
    nothing).  ∅ absorbs under multiplication (a × ∅ = ∅) and is
    transparent under addition (a + ∅ = a), so downstream arithmetic
    propagates correctly: ZERO × ∅ = ∅ with st=0.
    """
    max_d = x.max_positive_dim()
    sign = 1.0 if x.coeff(max_d) > 0 else -1.0
    try:
        return R(func(sign * float('inf')))
    except (ValueError, OverflowError):
        return Composite({})


# =============================================================================
# LOG SCALE  (opt-in)
# =============================================================================
#
# ln of an infinitesimal is  ln(c) + d*ln(h), and ln(h) needs a dimension that
# is positive but smaller than EVERY power -- log x outgrows any constant and is
# outgrown by x^e for every e > 0.  No float sits there, so with scalar
# dimensions the d*ln(h) term has nowhere to go and ln() raises.
#
# A VECTOR dimension (power, log) does have room: the log component is the minor
# one, so lexicographic comparison puts any log term below any power term, which
# is exactly the dominance order.  Then
#
#     ln(|c|_d)      =  |ln c|_(0,0) + |d|_(0,1)
#     exp(|k|_(0,1)) =  |1|_(k,0)
#
# It is ON by default.  ln of an infinitesimal is a question with an answer,
# and refusing it -- or worse, approximating it through the numeric fallback,
# which left lim(x->0+) x^x at 1.6e-6 short of 1 -- is the wrong default when
# the exact answer is available.  Escalation costs nothing until it happens:
# vector dimensions live on their own dict backend, a composite only moves
# there when a log term actually appears, and it demotes back the moment the
# log components cancel.  The scalar fast path is never touched.
#
# Set LOG_SCALE = False to disable it, in which case ln of an infinitesimal
# raises rather than silently dropping the scale (which is what it used to do,
# and which produced wrong answers: ln(h), ln(h^2) and ln(sqrt(h)) all returned
# the same object).
#
# Known bound: ONE level of log.  ln(ln(x)) needs a third basis component,
# because ln(d*ln(h)) = ln(d) + ln(ln(h)) is a new scale.
LOG_SCALE = True

_VEC_BACKEND = [None]


def _vector_backend():
    if _VEC_BACKEND[0] is None:
        from composite.backends.vector_dim_backend import VectorDimBackend
        _VEC_BACKEND[0] = VectorDimBackend()
    return _VEC_BACKEND[0]


def _unit_dim(x):
    """The dimension meaning 1 on x's backend: 0, or (0, 0, ...) for vectors."""
    if getattr(x._backend, "VECTOR_DIMS", False):
        from composite.backends.vector_dim_backend import WIDTH
        return (0,) * WIDTH
    return 0


def _like(x, terms):
    """Build a composite on the SAME backend as `x`, not the active one.

    The transcendental series construct intermediates with Composite({...}),
    which binds whatever backend is globally active.  When the argument carries
    vector dimensions that backend cannot hold the tuples, and numpy raises
    "setting an array element with a sequence".
    """
    be = x._backend
    return Composite._wrap(
        be.create_from_terms(list(terms.keys()), list(terms.values())),
        be, demote=False)


def _vec_composite(terms):
    """Build a vector-dimension composite whatever the active backend is.

    Goes through _wrap so the result demotes when it turns out to carry nothing
    but powers -- exp(ln(h)) is |1|_(-1,0), which is just h and belongs back on
    the scalar path.  A term with a real log component will not demote.
    """
    be = _vector_backend()
    return Composite._wrap(
        be.create_from_terms(list(terms.keys()), list(terms.values())), be)


def _demote(data, be):
    """Return a vector-dimension composite to the scalar path when it can.

    A composite keeps vector dimensions only while something other than the
    power component is non-zero.  Without this, a single ln() anywhere would
    leave every value downstream of it on the dict backend for good, and the
    whole point of the vector representation is that you pay for it only while
    you are using it.  Returns (data, backend), unchanged when it cannot demote.
    """
    dims, vals = be.to_arrays(data)
    for d in dims:
        if isinstance(d, tuple) and any(c != 0 for c in d[1:]):
            return data, be
    active = get_backend()
    if active.VECTOR_DIMS:
        return data, be                      # nowhere scalar to go
    flat = [d[0] if isinstance(d, tuple) else d for d in dims]
    return active.create_from_terms(flat, vals), active


def _log_part(x):
    """Split the INFINITE log terms off an exponent.  Returns (logs, rest).

    exp needs the split because an infinite argument changes scale while a
    vanishing one is just a Taylor series.  A vector dimension (p, l) is
    infinite exactly when it is lexicographically above zero -- p > 0, or
    p == 0 and l > 0 -- which is the same dominance order the comparison uses.

    So a MIXED term like (-1, 1) -- that is h*log(h) -- is infinitesimal, not
    infinite: the power dominates the log.  It belongs in `rest` and expands
    ordinarily.  Treating it as unrepresentable is what left x^x falling back
    to numeric sampling and landing 1.6e-6 short of 1.
    """
    dims, vals = x._backend.to_arrays(x._data)
    logs, rest = {}, {}
    for d, v in zip(dims, vals):
        if isinstance(d, tuple) and len(d) > 1:
            p, l = d[0], d[1]
            infinite = p > 0 or (p == 0 and l > 0)
            if infinite and p == 0:
                logs[d] = v                     # k * ln(1/h): changes scale
                continue
        rest[d] = v
    return logs, rest


def _has_infinitesimal_part(x):
    """True when x carries any NONZERO coefficient away from dimension 0.

    A composite whose only content is at dimension 0 -- or whose other
    dimensions are all zero -- is just its standard part, and a Taylor series
    around it must not be built: x - R(a) would be a zero, and R1 would turn
    that zero into |1|_-1, manufacturing an infinitesimal that is not there.
    """
    dims, vals = x._backend.to_arrays(x._data)
    # NB: compare the dimension itself, not int(d) -- int(-0.5) is 0, which
    # made a purely fractional infinitesimal (sqrt of an odd dimension) look
    # like a bare standard part, so sin/ln returned f(st(x)) and dropped it.
    return any(d != 0 and v != 0.0 for d, v in zip(dims, vals))


def _is_nothing(x):
    """Check if x is the empty composite (nothing)."""
    return isinstance(x, Composite) and not x.c


def sin(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.sin, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.sin(a)})
    _nz = {d: c for d, c in x.c.items() if d != 0 and c != 0.0}
    h = Composite({d: c for d, c in x.c.items() if d != 0})
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_h = Composite({})
    cos_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    # Skip zero-coefficient terms to avoid 0 * Composite uplift
    result = Composite({})
    if sin_a != 0:
        result = result + sin_a * cos_h
    if cos_a != 0:
        result = result + cos_a * sin_h
    return _truncate_order(result, _tighter(_complete_order(_nz, terms),
                                            _min_complete(x)))


def cos(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.cos, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.cos(a)})
    _nz = {d: c for d, c in x.c.items() if d != 0 and c != 0.0}
    h = Composite({d: c for d, c in x.c.items() if d != 0})
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_h = Composite({})
    cos_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    result = Composite({})
    if cos_a != 0:
        result = result + cos_a * cos_h
    if sin_a != 0:
        result = result - sin_a * sin_h
    return _truncate_order(result, _tighter(_complete_order(_nz, terms),
                                            _min_complete(x)))


def exp(x, terms=15):
    """Exponential function for Composite numbers."""
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return Composite({0: math.exp(float(x))})

    if not isinstance(x, Composite):
        return Composite({0: math.exp(float(x))})

    if _is_nothing(x):
        return Composite({})

    if LOG_SCALE and getattr(x._backend, "VECTOR_DIMS", False):
        _logs, _rest = _log_part(x)
        if _logs:
            # exp(k * ln(1/h)) = (1/h)^k = h^(-k), which sits at power +k.
            out = None
            for (_, k), v in _logs.items():
                t = _vec_composite({(v, 0): 1.0}) if k == 1 else None
                if t is None:
                    raise ValueError(
                        f"exp of a log term at level {k} needs a deeper basis "
                        f"than {('power', 'log')}.")
                out = t if out is None else out * t
            if _rest:
                # A remainder that is only the standard part needs no series --
                # and recursing would rebuild it through the ACTIVE backend,
                # which cannot hold vector dimensions.
                if set(_rest) <= {(0, 0)}:
                    out = out * _vec_composite(
                        {(0, 0): math.exp(_rest.get((0, 0), 0.0))})
                else:
                    out = out * exp(_vec_composite(_rest), terms)
            return out

    a = x.st()
    # 7.1: a term exists iff its coefficient is nonzero -- exactly zero,
    # not 'small'.  A tolerance here silently discards real content.
    non_zero = {d: c for d, c in x.c.items() if d != 0 and c != 0.0}

    if not non_zero:
        return Composite({0: math.exp(a)})

    base = math.exp(a)
    h = _like(x, non_zero)
    one = _like(x, {_unit_dim(x): 1.0})

    exp_h = one
    h_power = one
    for n in range(1, terms):
        h_power = h_power * h
        exp_h = exp_h + (1.0 / math.factorial(n)) * h_power

    # f(g(x)) is complete only as far as g is: the outer series reaches for
    # orders of its argument that a truncated inner function never produced.
    return _truncate_order(base * exp_h,
                           _tighter(_complete_order(non_zero, terms),
                                    _min_complete(x)))


def ln(x, terms=15):
    """Natural logarithm for composite numbers.

    For positive infinitesimals (st=0 but positive coefficient at a
    negative dim), evaluates ln at the coefficient.  This is the
    composite-native handling: ZERO = |1|_{-1} is a positive
    infinitesimal with coefficient 1, so ln(ZERO) = R(ln(1)) = R(0)
    = ZERO.  Then x·ln(x) = ZERO² with st=0, and x^x = exp(ZERO²)
    = 1.0 exactly.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    if _is_nothing(x):
        return Composite({})

    a = x.st()

    if a <= 0:
        # Check for positive infinitesimal: st=0 but positive coeff
        # at a negative dimension (e.g. ZERO = |1|_{-1})
        coeffs = x.c
        if LOG_SCALE:
            # ln(|c|_d) = ln(c) + d*ln(h) holds for ANY non-zero d, so an
            # INFINITY is the same rule with the sign the other way round:
            # ln(1/h) = -ln(h) = +L.  Handled here because the branch below
            # only looks at negative dimensions.
            _pos = {d: c for d, c in coeffs.items() if d > 0 and c != 0.0}
            if _pos:
                _pd = max(_pos)
                _pc = _pos[_pd]
                if _pc > 0:
                    _t = {(0, 1): float(_pd)}
                    _lc = math.log(_pc)
                    if _lc != 0.0:
                        _t[(0, 0)] = _lc
                    _lead = _vec_composite(_t)
                    _rest = x / Composite({_pd: _pc})
                    if _is_unit(_rest):
                        return _lead
                    return _lead + ln(_rest, terms)
        neg_dims = {d: c for d, c in coeffs.items() if d < 0}
        if neg_dims:
            min_dim = min(neg_dims.keys())
            coeff = neg_dims[min_dim]
            if coeff > 0:
                # ln(|c|_d) = ln(c) + d*ln(h), and ln(h) needs a dimension that
                # is positive but smaller than EVERY power -- log x outgrows any
                # constant and is outgrown by x^e for every e > 0.  No float can
                # sit there, so the d*ln(h) term has nowhere to go.
                #
                # This used to drop it and return ln(c) alone, which makes
                # ln(h), ln(h^2), ln(h^3) and ln(sqrt(h)) all the SAME object
                # and produces wrong answers, not merely lost structure:
                #   lim(x->0+) ln(x)/ln(x*x)      gave 1.0   (is 0.5)
                #   lim(x->0+) ln(x)/ln(sqrt(x))  gave 1.0   (is 2.0)
                #   lim(x->0+) 1/ln(x)            gave |1|_1 (is 0.0, inverted)
                # It is right only when the log is multiplied by something that
                # vanishes, which is why x*ln(x) and x^x survived it.
                if LOG_SCALE:
                    # ln(x) = ln(lead) + ln(x/lead), and x/lead has standard
                    # part 1 so the second term takes the ordinary Taylor path.
                    _lead_terms = {(0, 1): float(min_dim)}
                    _lc = math.log(coeff)
                    if _lc != 0.0:
                        _lead_terms[(0, 0)] = _lc
                    _lead = _vec_composite(_lead_terms)
                    _rest = x / Composite({min_dim: coeff})
                    if _is_unit(_rest):
                        return _lead
                    return _lead + ln(_rest, terms)
                raise ValueError(
                    f"ln(|{coeff}|_{min_dim}): the log SCALE cannot be "
                    f"represented. ln of an infinitesimal is ln({coeff}) + "
                    f"({min_dim})*ln(h), and ln(h) needs a dimension between 0 "
                    f"and every positive power -- not expressible as a float. "
                    f"Dropping it silently returned ln({coeff}) and made "
                    f"ln(h), ln(h^2) and ln(sqrt(h)) indistinguishable. "
                    f"Rewrite so the log is multiplied by a vanishing factor "
                    f"(x*ln(x) and x^x work), or expand about a point with a "
                    f"nonzero standard part.")
        raise ValueError("ln requires positive standard part")

    if not _has_infinitesimal_part(x):
        return Composite({0: math.log(a)})

    h_part = x - R(a)
    ratio = h_part / R(a)

    # R6: there is no additive identity.  A summation that has not yet added a
    # term holds NOTHING, not zero -- seeding with Composite({0: 0.0}) would
    # assert a zero, which R1 converts to |1|_-1 and adds to the series.
    lead = math.log(a)
    result = Composite({}) if lead == 0.0 else Composite({0: lead})
    power = Composite({0: 1})

    for n in range(1, terms):
        power = power * ratio
        sign = (-1) ** (n + 1)
        result = result + sign * power / n

    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))


def sqrt(x, terms=12):
    """Square root for composite numbers via binomial series.

    For positive infinitesimals (st=0, positive coeff at negative dim),
    evaluates sqrt at the coefficient.  ZERO = |1|_{-1} has coeff 1,
    so sqrt(ZERO) = R(sqrt(1)) = R(1).  Then x·sqrt(x) at ZERO gives
    ZERO × R(1) = ZERO, st=0 exactly.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    if _is_nothing(x):
        return Composite({})

    # Dimensions HALVE under a square root:  sqrt(|c|_d) = |sqrt(c)|_(d/2).
    #
    # This used to keep the dimension instead of halving it, returning ZERO for
    # sqrt(ZERO) and silently giving wrong LIMITS, not just wrong intermediates:
    #   lim(x->0+) sqrt(x)/x   gave 1.0   (is infinity)
    #   lim(x->0+) x/sqrt(x)   gave 1.0   (is 0)
    #   lim(x->0+) sqrt(x*x)/x gave 0.0   (is 1 -- |x|/x for x > 0)
    #
    # Halving an ODD dimension lands on a half-integer.  That used to be
    # unrepresentable and raised, naming the x = s*s substitution as the way
    # round it.  Dimensions are float64 now, so sqrt(|1|_-1) = |1|_-0.5 is an
    # ordinary value and the substitution is no longer required.  Note the
    # result stays UNIT-SPACED -- sqrt shifts the lattice by 1/2, it does not
    # refine it -- so the run representation is unaffected.
    _nz = {d: c for d, c in x.coeffs_dict().items() if c != 0.0}
    if _nz:
        _lead = max(_nz)
        if _lead != 0:
            _c = _nz[_lead]
            if _c < 0:
                raise ValueError(
                    f"sqrt of a negative leading coefficient |{_c}|_{_lead}")
            _root = Composite({dim_cast(_lead / 2): math.sqrt(_c)})
            _rest = x / Composite({_lead: _c})        # leading dim 0, st() == 1
            return _root * sqrt(_rest, terms)

    a = x.st()
    if a < 0:
        raise ValueError("sqrt requires non-negative standard part")

    if not _has_infinitesimal_part(x):
        return Composite({0: math.sqrt(a)})

    sqrt_a = math.sqrt(a)
    h_part = x - R(a)
    ratio = h_part / R(a)

    def binom(n):
        if n == 0:
            return 1
        result = 1
        for k in range(n):
            result *= (0.5 - k)
        return result / math.factorial(n)

    result = Composite({0: sqrt_a})
    power = Composite({0: 1})

    for n in range(1, terms):
        power = power * ratio
        result = result + binom(n) * sqrt_a * power

    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))


def tan(x, terms=12):
    """Tangent function via sin/cos"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    _s, _c = sin(x, terms), cos(x, terms)
    return _truncate_order(_s / _c, _tighter(_shared_order(_s, _c),
                                             _min_complete(_s, _c, x)))

# =============================================================================
# INVERSE TRIGONOMETRIC FUNCTIONS
# =============================================================================

def _reciprocal(x, terms=15):
    """Compute 1/x via geometric series. Internal helper."""
    a = x.st()
    if a == 0.0:
        # 7.1: exactly zero, not "small".  A tolerance here refused to invert
        # perfectly good composites whose standard part happened to be tiny.
        raise ZeroDivisionError("Cannot compute 1/x at x=0")
    h_part = x - R(a)
    ratio = h_part / R(-a)
    result = Composite({0: 1/a})
    power = Composite({0: 1})
    for n in range(1, terms):
        power = power * ratio
        result = result + power / a
    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))

def atan(x, terms=15):
    """Arctangent for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.atan, x)
    a = x.st()
    one_plus_x2 = R(1) + x * x
    deriv = _reciprocal(one_plus_x2, terms)
    if not _has_infinitesimal_part(x):
        return Composite({0: math.atan(a)})
    _lead = math.atan(a)
    result = {} if _lead == 0.0 else {0: _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = dim - 1
        if new_dim != 0:
            result[new_dim] = coeff / abs(new_dim)
    out = Composite(result)
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
    return out

def asin(x, terms=15):
    """Arcsine for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.asin, x)
    a = x.st()
    if abs(a) >= 1:
        raise ValueError("asin requires |standard part| < 1")
    inner = R(1) - x * x
    deriv = _reciprocal(sqrt(inner, terms), terms)
    if not _has_infinitesimal_part(x):
        return Composite({0: math.asin(a)})
    _lead = math.asin(a)
    result = {} if _lead == 0.0 else {0: _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = dim - 1
        if new_dim != 0:
            result[new_dim] = coeff / abs(new_dim)
    out = Composite(result)
    # Same order shift as antiderivative, and the same reason to record it:
    # this dict is built directly, so nothing else would.  Read the bound from
    # deriv AFTER the chain-rule multiply, which has already taken the min of
    # 1/sqrt(1-u^2) and u'.
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
    return out

def acos(x, terms=15):
    """Arccosine for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    return R(math.pi / 2) - asin(x, terms)

# =============================================================================
# HYPERBOLIC FUNCTIONS
# =============================================================================

def sinh(x, terms=15):
    """Hyperbolic sine: (exp(x) - exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    return (exp(x, terms) - exp(-x, terms)) / 2

def cosh(x, terms=15):
    """Hyperbolic cosine: (exp(x) + exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    return (exp(x, terms) + exp(-x, terms)) / 2

def tanh(x, terms=15):
    """Hyperbolic tangent: sinh(x) / cosh(x)"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.tanh, x)
    _s, _c = sinh(x, terms), cosh(x, terms)
    return _truncate_order(_s / _c, _tighter(_shared_order(_s, _c),
                                             _min_complete(_s, _c, x)))


# =============================================================================
# ERROR FUNCTION AND THE NORMAL CDF
# =============================================================================
#
# Each of these is an integral of a Gaussian, and both pieces already exist:
# exp() works on a composite, and integrating in the infinitesimal is a
# DIMENSION SHIFT that antiderivative() performs.  So
#
#     f(a + h) = f(a) + integral_0^h f'(a + s) ds
#              = antiderivative( f'(x), f(a) )
#
# is the whole implementation -- no series coefficients to derive, no table.
# The standard part comes from math so it keeps full precision; the
# infinitesimal part comes from the algebra so derivatives and limits work.
# erfc and Phi use math.erfc rather than 1 - erf, which loses its significant
# digits once erf(a) approaches 1.

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def erf(x, terms=15):
    """Error function.  d/dx erf = (2/sqrt(pi)) exp(-x^2)."""
    if isinstance(x, (int, float)):
        return Composite({0: math.erf(float(x))})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.erf, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.erf(a)})
    return antiderivative(_TWO_OVER_SQRT_PI * exp(-(x * x), terms), math.erf(a))


def erfc(x, terms=15):
    """Complementary error function.  d/dx erfc = -(2/sqrt(pi)) exp(-x^2).

    Not 1 - erf(x): that cancels away the answer for x beyond about 2, where
    erf is 0.995 and the difference is the part that matters.
    """
    if isinstance(x, (int, float)):
        return Composite({0: math.erfc(float(x))})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.erfc, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.erfc(a)})
    return antiderivative(-_TWO_OVER_SQRT_PI * exp(-(x * x), terms), math.erfc(a))


def normal_cdf(x, terms=15):
    """Standard normal CDF.  Phi'(x) = exp(-x^2/2)/sqrt(2 pi).

    Evaluated as 0.5*erfc(-x/sqrt(2)) at the standard part, which stays
    accurate in the left tail where 0.5*(1 + erf) does not.
    """
    if isinstance(x, (int, float)):
        return Composite({0: 0.5 * math.erfc(-float(x) / math.sqrt(2.0))})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(
            lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)), x)
    a = x.st()
    base = 0.5 * math.erfc(-a / math.sqrt(2.0))
    if not _has_infinitesimal_part(x):
        return Composite({0: base})
    return antiderivative(
        _ONE_OVER_SQRT_2PI * exp(-(x * x) * 0.5, terms), base)


Phi = normal_cdf          # the name the finance literature uses


# =============================================================================
# REAL-VALUED POWERS
# =============================================================================

def power(x, s, terms=15):
    """x^s via exp(s * ln(x)).  The exponent may be real OR a Composite.

    A composite exponent used to be routed through R(s), which calls float(s)
    -- that is st(s), and the standard part of an infinity is 0.  So the
    exponent was silently replaced by an expressed zero and the function
    computed x^0:

        power(1+x, 1/x)  returned 1.0   (is e)

    because 1/x is |1|_1, float(|1|_1) is 0.0, and R(0.0) is |0|_0.  The
    exponent has to stay a composite, which is what __pow__ already did --
    hence a ** b and exp(ln(a)*b) both gave e while power() did not.
    """
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if isinstance(s, int):
        return x ** s
    if isinstance(s, Composite):
        return exp(s * ln(x, terms), terms)
    return exp(R(s) * ln(x, terms), terms)


# =============================================================================
# HIGH-LEVEL API: AUTOMATIC TRANSLATION
# =============================================================================

def _d_deps(x):
    """d(x)/d(eps): the derivative of a composite w.r.t. its own infinitesimal.

    A term c*eps**k sits at dim -k, and differentiating gives k*c*eps**(k-1) at
    dim -(k-1) = d+1.  This is the CHAIN RULE FACTOR that asin and atan need:
    they form deriv = 1/sqrt(1-u**2) (resp. 1/(1+u**2)) and then antidifferentiate
    it with respect to eps, but d/deps asin(u(eps)) = u'(eps)/sqrt(1-u**2).
    Without the u' factor they are correct only when u' == 1 -- a bare seeded
    variable, which is what every test used.  asin(2x) came back exactly half
    its true first derivative; asin(x*x) looked correct only because 2a == 1 at
    the probe point a = 0.5.
    """
    out = Composite({d + 1: c * abs(d) for d, c in x.c.items()
                     if not isinstance(d, tuple) and d < 0 and c != 0.0})
    # Differentiating LOSES an order: the top coefficient of x produces the top
    # of x', and there is nothing above it to produce the next.  Failing to
    # record that made asin(sin x) claim order 12 on 11 sound ones.
    _c = getattr(x, "_complete", None)
    if _c is not None:
        out._complete = _c - 1
    return out


def _reject_pole(result, what: str, at: float):
    """Raise if the seeded result carries a POLE, instead of quietly dropping it.

    A seeded evaluation puts the regular part on dimensions <= 0 and any
    singular part on dimensions > 0.  Every extractor here reads dimensions
    <= 0 only, so f(x) = 1/(1-cos x) at 0 -- whose composite is correctly
    |2|_+2 + |1/6|_0 + |1/120|_-2, the Laurent series 2/x^2 + 1/6 + x^2/120 --
    came back as a clean-looking [1/6, 0, 1/120] with the double pole silently
    discarded.  There is no Taylor series at a pole; returning the regular part
    as if there were is the wrong answer, not a partial one.

    Only NONZERO positive coefficients count.  An expressed zero up there is
    legitimate and must not trip this: INF - INF = |0|_+1 by the canon rule that
    a constructed dimension is retained.
    """
    poles = {d: c for d, c in result.coeffs_dict().items() if d > 0 and c != 0.0}
    if poles:
        order = max(poles)
        raise ValueError(
            f"{what} at {at}: f has a pole of order {order} here, so no Taylor "
            f"series exists.  The singular part IS present in the composite, on "
            f"dimensions {sorted(poles)} (coefficients "
            f"{[poles[d] for d in sorted(poles)]}); these extractors read "
            f"dimensions <= 0 only.  Read the full Laurent series off the "
            f"seeded result directly, or expand about a regular point."
        )


def derivative(f: Callable, at: float, terms: int = 12) -> float:
    """Compute f'(at) automatically."""
    x = _seeded(at)
    result = f(x)
    _reject_pole(result, "derivative", at)
    return result.d(1)


def nth_derivative(f: Callable, n: int, at: float, terms: int = 12) -> float:
    """Compute f^(n)(at) - the nth derivative at a point."""
    with _derivative_scope(n, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "nth_derivative", at)
        return result.d(n)


def all_derivatives(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Compute [f(at), f'(at), f''(at), ...] up to nth derivative."""
    with _derivative_scope(up_to, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "all_derivatives", at)
        return [result.st()] + [result.d(n) for n in range(1, up_to + 1)]


def limit(f: Callable, as_x_to: float, terms: int = 12,
          dir: str = "both", fallback: bool = False) -> float:
    """Compute lim(x→as_x_to) f(x) via composite evaluation.

    Evaluates f at a composite infinitesimal.  Bounded transcendentals
    (sin, cos, atan, asin, acos, tanh) evaluate at st(x) for infinite
    arguments, so oscillatory limits like x·sin(1/x) resolve algebraically.

    Positive dims in the result indicate unbounded divergence (e.g. exp).
    Domain errors (ln(0), sqrt(-x)) raise LimitUndecidableError, or fall
    back to integral averaging with ``fallback=True``.

    Args:
        f:         Function to evaluate.
        as_x_to:   Point to approach (float, float('inf'), or Composite INF).
        terms:     Truncation order for transcendentals.
        dir:       Direction — "both" (default), "+" (right), "-" (left).
        fallback:  If True, use integral averaging when algebraic fails.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _limit_impl(f, as_x_to, terms, dir, fallback)


def _limit_impl(f, as_x_to, terms, dir, fallback):
    """Internal implementation of limit(), wrapped to suppress numpy warnings."""
    # Normalize: accept Composite INF/-INF as well as float('inf')
    _is_inf = False
    if isinstance(as_x_to, Composite):
        max_d = as_x_to.max_positive_dim()
        if max_d is not None:
            coeffs = as_x_to.coeffs_dict()
            _is_inf = True
            as_x_to = float('inf') if coeffs.get(max_d, 0) > 0 else float('-inf')
        elif _has_infinitesimal_part(as_x_to):
            # No positive dimension, so not an infinity -- and it carries an
            # infinitesimal part, so st() would silently throw that away and
            # answer a DIFFERENT question.  "Approach 2 + eps" is not a limit
            # point; "approach 2" is.  A composite that is purely dimension 0
            # falls through and collapses losslessly, which is fine.
            raise TypeError(
                f"limit(as_x_to={as_x_to}): the point to approach must be a "
                f"real, float('inf'), or a composite INFINITY. This carries an "
                f"infinitesimal part, and taking its standard part would "
                f"silently change which limit is computed. Pass st() explicitly "
                f"if that is what you meant.")
        else:
            as_x_to = as_x_to.st()

    # Build the evaluation point
    if as_x_to == float('inf'):
        x = INF
        _is_inf = True
    elif as_x_to == float('-inf'):
        x = -INF
        _is_inf = True
    elif dir == "-":
        x = -ZERO if as_x_to == 0 else R(as_x_to) - ZERO
    else:
        x = _seeded(as_x_to)

    try:
        result = f(x)
    except TypeError:
        raise CompositionError(
            "Function not composable with composite arithmetic")
    except LimitDoesNotExistError:
        # Division by nothing (∅) — denominator is indeterminate.
        # The limit provably does not exist. Don't try to recover.
        raise
    except (ValueError, ZeroDivisionError):
        # Domain error (ln(0), sqrt(0), etc.) — try composite extrapolation
        # from a nearby point where the function is well-defined.
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Algebraic evaluation failed (domain error at limit point). "
            "Use fallback=True for integral averaging.")

    # Check for NaN/Inf contamination
    st_val = result.st()
    if not math.isfinite(st_val):
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Algebraic evaluation produced NaN/Inf.")

    # Positive dims → unbounded divergence (from exp, ln, etc.)
    max_pos = result.max_positive_dim()
    if max_pos is not None:
        pos_coeffs = {d: c for d, c in result.coeffs_dict().items() if d > 0}
        signs = [c > 0 for c in pos_coeffs.values()]
        if all(signs):
            return INF
        elif not any(signs):
            return -INF
        # Mixed signs — try composite extrapolation before giving up
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Result has mixed-sign positive dimensions.")

    # Nothing (∅) means an indeterminate value was involved. Check if the
    # overall expression still converges by probing at real points.
    if _is_nothing(result):
        if _is_inf:
            # Evaluate at two large windows to check convergence
            v1 = _limit_at_inf_fallback(f, as_x_to, n=500, width=50.0)
            v2 = _limit_at_inf_fallback(f, as_x_to, n=500, width=100.0)
            if math.isfinite(v1) and math.isfinite(v2):
                diff = abs(v1 - v2)
                # Both small → converging to 0
                if abs(v1) < 1e-4 and abs(v2) < 1e-4:
                    return 0.0
                # Close relative to magnitude → converged
                if diff < 1e-3 * (abs(v1) + abs(v2)):
                    return v2
        else:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        raise LimitDoesNotExistError(
            "Result is indeterminate (nothing). Limit does not exist.")

    return st_val


def _limit_extrapolate(f, as_x_to, dir, terms, n_probes=6):
    """Extrapolate limit from nearby composite evaluations.

    Two strategies, tried in order:
      1. Taylor extrapolation: evaluate at probe, use eval_taylor(-eps) to
         extrapolate back to the limit point. Exact when Taylor converges.
      2. Value convergence: if Taylor overflows, use the st() values at
         decreasing probe distances. The values themselves converge.

    Returns the extrapolated value, or None if neither converged.
    """
    if dir == "-":
        signs = [-1]
    elif dir == "+":
        signs = [1]
    else:
        signs = [1, -1]

    for sign in signs:
        taylor_candidates = []
        value_candidates = []

        for k in range(n_probes):
            eps = 10 ** (-(k + 2))  # 1e-2, 1e-3, ..., 1e-7

            try:
                x_comp = _seeded(as_x_to + sign * eps)
                result = f(x_comp)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

            if not isinstance(result, Composite):
                result = R(float(result))

            st = result.st()
            if not math.isfinite(st):
                continue

            value_candidates.append(st)

            # Try Taylor extrapolation
            try:
                extrap = st + result.eval_taylor(-sign * eps)
                if math.isfinite(extrap):
                    taylor_candidates.append(extrap)
            except (OverflowError, ValueError):
                pass

        # Strategy 1: Taylor extrapolation converged
        if len(taylor_candidates) >= 2:
            if abs(taylor_candidates[-1] - taylor_candidates[-2]) < 1e-6 * (abs(taylor_candidates[-1]) + 1e-100):
                return taylor_candidates[-1]

        # Strategy 2: raw values converging (for cases like x^x where
        # Taylor overflows, or sqrt(x) where Taylor radius is too small).
        if len(value_candidates) >= 3:
            v = value_candidates
            d1 = abs(v[-1] - v[-2])
            d2 = abs(v[-2] - v[-3])
            # Values converging: differences shrinking
            if d1 < d2:
                # Tight convergence — last two very close
                if d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                    return v[-1]
            # Converging to 0: last few values all small (even if oscillating)
            if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                return 0.0

    return None


def _limit_integral_fallback(f, as_x_to, dir, a=1e-4, n=1000):
    """Compute limit via integral average: ∫f(x)dx / a over a small interval.

    Integration smooths oscillation — the average value over [as_x_to, as_x_to+a]
    converges to the limit even when point evaluation oscillates.
    """
    if dir == "-":
        lo, hi = as_x_to - a, as_x_to
    else:
        lo, hi = as_x_to, as_x_to + a

    # Avoid evaluating exactly at the singularity
    eps = a * 1e-10
    lo = lo + eps

    h = (hi - lo) / n
    total = 0.0
    for i in range(n):
        xi = lo + (i + 0.5) * h
        fi = f(R(xi) + ZERO)
        total += fi.st()
    total *= h

    return total / a


def _limit_at_inf_fallback(f, as_x_to, n=1000, width=100.0):
    """Compute limit at ±∞ via integral average over a large-x window.

    Evaluates ∫_M^{M+w} f(x)dx / w for large M.
    """
    sign = 1 if as_x_to == float('inf') else -1
    M = sign * 1e4
    lo = M
    hi = M + sign * width

    if lo > hi:
        lo, hi = hi, lo

    h = (hi - lo) / n
    total = 0.0
    for i in range(n):
        xi = lo + (i + 0.5) * h
        fi = f(R(xi) + ZERO)
        total += fi.st()
    total *= h

    return total / width


# Backward compatibility aliases
def limit_right(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute right-hand limit: lim(x→a⁺) f(x)"""
    return limit(f, as_x_to, terms=terms, dir="+")


def limit_left(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute left-hand limit: lim(x→a⁻) f(x)"""
    return limit(f, as_x_to, terms=terms, dir="-")


def taylor_coefficients(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Get Taylor series coefficients c_n = f^(n)(at)/n! of f around 'at'.

    Was missing the depth scope the other extractors have, so a high `up_to`
    quietly read zeros off dimensions the transcendentals never expanded to.
    """
    with _derivative_scope(up_to, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "taylor_coefficients", at)
        return [result.coeff(-n) for n in range(up_to + 1)]


# =============================================================================
# FIXED: antiderivative, _ensure_composite, _detect_singularity
# =============================================================================

def antiderivative(f_composite: Composite, constant: float = 0) -> Composite:
    """Compute antiderivative via dimensional shift.
    Each |c|_{-n} -> |c/(n+1)|_{-(n+1)}

    FIXED: Only processes dim <= 0. The old code let positive dims through:
      dim=2 -> new_dim=1 -> divisor=1 -> silently created a dim+1 term.
    Now skips all positive dims (INF components, etc.).
    """
    result = {0: constant}
    for dim, coeff in f_composite.c.items():
        if dim <= 0:
            new_dim = dim - 1
            divisor = abs(new_dim)
            result[new_dim] = coeff / divisor
    # Every order moves up by one, so a f sound to K integrates to one sound to
    # K+1.  Building the dict directly skips _truncate_order, which is where
    # the bound would otherwise be recorded -- dropping it here made asin, atan
    # and the derivative round trip all claim to be exact.
    _c = getattr(f_composite, "_complete", None)
    out = Composite(result)
    if _c is not None:
        out._complete = _c + 1
    return out


def _ensure_composite(val):
    """Wrap plain float/int as Composite. Warns on silent degradation.

    IMPORTANT: Uses Composite(float(val)), NOT R(float(val)).
    In v3, R(0) = ZERO = |1|_{-1} which carries d(1)=1.
    A constant return value should be |val|_0 with d(1)=0.
    """
    if isinstance(val, Composite):
        return val
    import warnings
    warnings.warn(
        "integrate: f returned plain float — Taylor convergence disabled "
        "for this panel. Wrap your function to return Composite.",
        stacklevel=3)
    return Composite(float(val))


def _detect_singularity(f, x_near, x_away, panel_dx):
    """Detect power-law singularity at x_near using composite.

    ONE composite eval near x_near. Extracts exponent
    alpha = dist * f'(x) / f(x) where dist = distance from boundary.
    If -1 < alpha < 0: integrable singularity, compute analytically.

    Returns (handled, value, C, alpha) — 4-tuple.
    If not handled, C and alpha are 0.0.
    """
    _not_found = (False, 0.0, 0.0, 0.0)

    eps = abs(panel_dx) * 0.01
    x_test = x_near + eps if x_near < x_away else x_near - eps
    dist_from_boundary = abs(x_test - x_near)

    if dist_from_boundary < 1e-30:
        return _not_found

    try:
        fx = _ensure_composite(f(_seeded(x_test)))
        f_val = fx.st()
        f_d1 = fx.d(1)

        if abs(f_val) < 1e-100:
            return _not_found

        # Exponent: f(x) ~ C * |x - boundary|^alpha
        #   f'/f = alpha / dist  =>  alpha = dist * f'/f
        sign = 1.0 if x_near < x_away else -1.0
        alpha = dist_from_boundary * (sign * f_d1) / f_val

        if not (-1.0 < alpha < -0.01):
            return _not_found

        # Sanity: derivative ratio must indicate true singularity
        deriv_ratio = abs(f_d1 / f_val) * abs(x_away - x_near)
        if deriv_ratio < 10:
            return _not_found

        # Coefficient: f(x) = C * dist^alpha  =>  C = f(x) / dist^alpha
        C = f_val / (dist_from_boundary ** alpha)
        ap1 = alpha + 1

        # Return (True, C, alpha) so the caller can integrate over any sub-range.
        # Default: integrate over [0, full_dist]
        full_dist = abs(x_away - x_near)
        singular_val = C * (full_dist ** ap1) / ap1

        return True, singular_val, C, alpha

    except (ZeroDivisionError, OverflowError, ValueError):
        return False, 0.0, 0.0, 0.0


def definite_integral(f: Callable, a: float, b: float, terms: int = 12) -> float:
    """Compute ∫ₐᵇ f(x) dx."""
    result, _ = integrate_adaptive(f, a, b, tol=1e-10, terms=terms)
    return result.st()

# =============================================================================
# MULTI-POINT STEPPED INTEGRATION
# =============================================================================


def integrate(f, *args, curve=None, surface=None, tol=1e-10, terms=15):
    """One integral to rule them all."""

    def _st(v):
        return v.st() if isinstance(v, Composite) else float(v)

    # --- LINE INTEGRAL ---
    if curve is not None:
        t_range = args[0] if args else (0, 1)
        a_t, b_t = t_range
        is_vector = isinstance(f, list)

        t_mid = (a_t + b_t) / 2
        try:
            probe = curve(_seeded(t_mid))
            if isinstance(probe, (list, tuple)):
                # A non-composite component is safe to freeze as a constant
                # ONLY if it is genuinely constant.  One probe cannot tell
                # "0" from "math.cos(t)" -- both come back as plain floats --
                # so probe a second t and see which ones move.
                #
                # Judging by probe[0] alone sent [0, t] down the 2000-step
                # finite-difference fallback because its FIRST component was
                # constant.  Accepting ANY composite component instead froze
                # math.cos(t)/math.sin(t) at their midpoint values and made the
                # helix arc length come out 2*pi instead of 2*pi*sqrt(2): the
                # z-component was the only one left with a tangent.
                other = curve(_seeded(t_mid + 0.25 * (b_t - a_t) + 1e-3))
                composite_curve = True
                for pa, pb in zip(probe, other):
                    if isinstance(pa, Composite):
                        continue
                    if float(pa) != float(pb):
                        composite_curve = False      # varies, but opaque
                        break
            else:
                composite_curve = isinstance(probe, Composite)
        except (TypeError, AttributeError):
            composite_curve = False

        if composite_curve:
            def _line_integrand(t_comp):
                pos_comp = curve(t_comp)
                pos_comp = [p if isinstance(p, Composite) else Composite({0: float(p)})
                            for p in pos_comp]
                tangent = [p.d(1) for p in pos_comp]

                if is_vector:
                    F_comp = [comp(*pos_comp) for comp in f]
                    F_comp = [Composite({0: float(fc)}) if isinstance(fc, (int, float)) else fc
                              for fc in F_comp]
                    # A component of the curve that does not move contributes
                    # nothing to F.dr, so skip it rather than multiply by its
                    # zero tangent.  Both operands are then zeros, R1 converts
                    # each, and |0|_0 * 0.0 comes back as |1|_-2 -- an
                    # infinitesimal injected into the integrand.  The standard
                    # part survives that, but integrate_adaptive builds its
                    # antiderivative from the Taylor coefficients, so it
                    # integrates a series whose derivatives are fabricated:
                    # work of F=[1,0] along [t,0] returned 3.140625, not 3.
                    # Python's sum() is avoided for the same reason -- it
                    # starts from the int 0, which is another zero operand.
                    acc = None
                    for fc, tv in zip(F_comp, tangent):
                        if tv == 0.0:
                            continue
                        term = fc * tv
                        acc = term if acc is None else acc + term
                    return acc if acc is not None else Composite({0: 0.0})
                else:
                    f_comp = f(*pos_comp)
                    if isinstance(f_comp, (int, float)):
                        f_comp = Composite({0: float(f_comp)})
                    speed = math.sqrt(sum(tv**2 for tv in tangent))
                    return f_comp * speed
        else:
            N = 2000
            dt = (b_t - a_t) / N
            total = 0.0
            for i in range(N):
                t_mid = a_t + (i + 0.5) * dt
                pt = curve(t_mid)
                eps_fd = 1e-7
                pt_fwd = curve(t_mid + eps_fd)
                tangent = [(float(pt_fwd[j]) - float(pt[j])) / eps_fd
                            for j in range(len(pt))]
                if is_vector:
                    F_vals = [_st(comp(*[float(p) for p in pt])) for comp in f]
                    total += sum(fv * tv for fv, tv in zip(F_vals, tangent)) * dt
                else:
                    speed = math.sqrt(sum(tv**2 for tv in tangent))
                    total += _st(f(*[float(p) for p in pt])) * speed * dt
            return total
        result, err = integrate_adaptive(_line_integrand, a_t, b_t, tol=tol, terms=terms)
        return result.st()

    # --- SURFACE INTEGRAL ---
    if surface is not None:
        uv = args[0] if args else ((0, 1), (0, 1))
        (a_u, b_u), (a_v, b_v) = uv
        is_vector = isinstance(f, list)

        from composite.composite_multivar import MC
        composite_surface = True
        try:
            u_mid = (a_u + b_u) / 2.0
            v_mid = (a_v + b_v) / 2.0
            u_test = MC.var(0, 2, val=u_mid)
            v_test = MC.var(1, 2, val=v_mid)
            test_result = surface(u_test, v_test)
            if not isinstance(test_result, (list, tuple)) or len(test_result) < 3:
                composite_surface = False
            else:
                for comp in test_result:
                    if not isinstance(comp, MC):
                        composite_surface = False
                        break
        except Exception:
            composite_surface = False

        if composite_surface:
            def _surface_integrand(u_val, v_val):
                u_mc = MC.var(0, 2, val=u_val)
                v_mc = MC.var(1, 2, val=v_val)
                S = surface(u_mc, v_mc)
                dSdu = [S[i].d(1, 0) for i in range(3)]
                dSdv = [S[i].d(1, 1) for i in range(3)]
                nx = dSdu[1] * dSdv[2] - dSdu[2] * dSdv[1]
                ny = dSdu[2] * dSdv[0] - dSdu[0] * dSdv[2]
                nz = dSdu[0] * dSdv[1] - dSdu[1] * dSdv[0]
                norm = math.sqrt(float(nx)**2 + float(ny)**2 + float(nz)**2)
                if norm < 1e-30:
                    return 0.0
                pos = [float(S[i].st()) for i in range(3)]
                if is_vector:
                    F_vals = [_st(comp(*pos)) for comp in f]
                    return F_vals[0]*float(nx) + F_vals[1]*float(ny) + F_vals[2]*float(nz)
                else:
                    return _st(f(*pos)) * norm

            def _inner_v(u_val):
                def g(v_val):
                    return _surface_integrand(u_val, v_val)
                result, _ = integrate_adaptive(
                    lambda v_comp: R(g(v_comp.st())),
                    a_v, b_v, tol=tol, terms=terms
                )
                return result.st()

            outer, _ = integrate_adaptive(
                lambda u_comp: R(_inner_v(u_comp.st())),
                a_u, b_u, tol=tol, terms=terms
            )
            return outer.st()

        else:
            Nu, Nv = 300, 300
            du = (b_u - a_u) / Nu
            dv = (b_v - a_v) / Nv
            total = 0.0
            eps = 1e-7
            for i in range(Nu):
                u = a_u + (i + 0.5) * du
                for j in range(Nv):
                    v = a_v + (j + 0.5) * dv
                    p0 = [float(x) for x in surface(u, v)]
                    pu = [float(x) for x in surface(u + eps, v)]
                    pv = [float(x) for x in surface(u, v + eps)]
                    du_vec = [(pu[k] - p0[k]) / eps for k in range(3)]
                    dv_vec = [(pv[k] - p0[k]) / eps for k in range(3)]
                    nx = du_vec[1]*dv_vec[2] - du_vec[2]*dv_vec[1]
                    ny = du_vec[2]*dv_vec[0] - du_vec[0]*dv_vec[2]
                    nz = du_vec[0]*dv_vec[1] - du_vec[1]*dv_vec[0]
                    if is_vector:
                        F_vals = [_st(comp(*p0)) for comp in f]
                        total += (F_vals[0]*nx + F_vals[1]*ny + F_vals[2]*nz) * du * dv
                    else:
                        dS = math.sqrt(nx**2 + ny**2 + nz**2)
                        total += _st(f(*p0)) * dS * du * dv
            return total

    # --- 1D DEFINITE / IMPROPER ---
    if len(args) == 2 and isinstance(args[0], (int, float)):
        a_val, b_val = args
        a_inf = math.isinf(a_val) and a_val < 0
        b_inf = math.isinf(b_val) and b_val > 0
        if a_inf and b_inf:
            val, _ = improper_integral_both(f, tol=tol)
            return val.st()
        if b_inf:
            val, _ = improper_integral(f, a_val, tol=tol)
            return val.st()
        if a_inf:
            val, _ = improper_integral(lambda x: f(-x), -b_val, tol=tol)
            return val.st()
        result, _ = integrate_adaptive(f, a_val, b_val, tol=tol, terms=terms)
        return result.st()

    # --- 2D BOX ---
    if len(args) == 2 and isinstance(args[0], tuple):
        (a_x, b_x), (a_y, b_y) = args
        N = 200
        dx = (b_x - a_x) / N
        dy = (b_y - a_y) / N
        total = 0.0
        for i in range(N):
            x = a_x + (i + 0.5) * dx
            for j in range(N):
                y = a_y + (j + 0.5) * dy
                total += _st(f(x, y)) * dx * dy
        return total

    # --- 3D BOX ---
    if len(args) == 3 and isinstance(args[0], tuple):
        (a_x, b_x), (a_y, b_y), (a_z, b_z) = args
        N = 50
        dx = (b_x - a_x) / N
        dy = (b_y - a_y) / N
        dz = (b_z - a_z) / N
        total = 0.0
        for i in range(N):
            x = a_x + (i + 0.5) * dx
            for j in range(N):
                y = a_y + (j + 0.5) * dy
                for k in range(N):
                    z = a_z + (k + 0.5) * dz
                    total += _st(f(x, y, z)) * dx * dy * dz
        return total

    raise ValueError(f"Could not determine integral type from arguments: {args}")

def integrate_stepped(f: Callable, a: float, b: float, step: float = 0.5, terms: int = 15):
    """Multi-point stepped integration with error estimate."""
    total = Composite({})
    total_error = 0.0
    x0 = a
    while x0 < b:
        dx = min(step, b - x0)
        fx = f(_seeded(x0))
        Fx = antiderivative(fx)
        neg_terms = {d: c for d, c in Fx.c.items() if d < 0}
        contribution = sum(
            coeff * dx ** abs(dim)
            for dim, coeff in neg_terms.items()
        )
        if neg_terms:
            min_dim = min(neg_terms.keys())
            step_error = abs(neg_terms[min_dim] * dx ** abs(min_dim))
        else:
            step_error = 0.0
        total = total + Composite({0: contribution})
        total_error += step_error
        x0 += dx
    return total, total_error


# =============================================================================
# FIXED: integrate_adaptive — recursive bisection with Taylor error
# =============================================================================

def integrate_adaptive(f, a, b, tol=1e-10, terms=15, max_depth=20, min_panels=4):
    """Adaptive integration via composite Taylor convergence.

    Primary path (ONE evaluation per panel):
      1. Evaluate f at panel midpoint via _seeded(mid)
      2. Integrate via antiderivative evaluated at +/- dx/2
      3. Estimate error from Taylor tail
      4. If converged -> accept. If not -> bisect recursively.

    Singularity handling:
      If Taylor tail overflows, tries power-law detection at boundaries.
      If detected, integrates analytically. Otherwise 3-eval fallback.

    Lift-at-the-gate:
      If f doesn't propagate composite structure, builds 4th-order
      approximation via 5-point stencil. Emits warning.

    Returns (Composite, float) — integral value, error estimate.
    """
    _fallback_count = [0]

    # --- LIFT AT THE GATE ---
    probe = _ensure_composite(f(_seeded((a + b) / 2)))
    if not any(dim < 0 for dim in probe.c):
        import warnings
        warnings.warn(
            "integrate_adaptive: f does not propagate composite structure to output. "
            "Falling back to midpoint rule (no adaptive refinement for this integrand).",
            stacklevel=2)

    def _panel_with_error(x, dx):
        """One composite eval at midpoint -> integral value + error estimate."""
        mid = x + dx / 2
        fx = _ensure_composite(f(_seeded(mid)))

        # Integral via antiderivative
        Fx = antiderivative(fx)
        right = Fx.eval_taylor(dx / 2)
        left = -Fx.eval_taylor(-dx / 2)
        value = left + right

        # The remainder is not something to model -- the top two orders of the
        # expansion ARE it.  Reading it off the last few terms instead assumed
        # the coefficients decay monotonically, and they do not: exp(-t^2) about
        # a panel midpoint climbs two orders of magnitude before it falls, so the
        # trailing terms sit far below the hump that dominates the remainder.
        # The estimate came back ~1e-8 of the true error, panels were accepted
        # unrefined, and integral_0^8 exp(-t^2) dt was wrong by 1.7e-05 while
        # claiming 5.9e-11 -- returning erf > 1.  Every order is now complete
        # (see _complete_order), so the top two ARE the leading remainder and
        # there is nothing left to estimate.
        half_dx = abs(dx) / 2
        orders = [_dim_order(d) for d in fx.c if _dim_order(d) >= 0]
        if not orders or max(orders) <= 2:
            # Nothing above a quadratic -- the antiderivative is EXACT.
            return value, 0.0

        cut = max(max(orders) - 2, 2)
        err_est = 0.0
        for dim, coeff in fx.c.items():
            k = _dim_order(dim)
            if k <= cut:
                continue
            # |h**(k+1) - (-h)**(k+1)| <= 2*h**(k+1), so the orders that cancel
            # on a symmetric panel cost at most a factor 2.  Erring high here
            # spends evaluations; erring low returns wrong answers.
            term = 2.0 * abs(coeff) / (k + 1) * half_dx ** (k + 1)
            if not math.isfinite(term):
                return value, -1.0  # signal: needs fallback
            err_est += term

        if not math.isfinite(err_est):
            return value, -1.0  # signal: needs fallback
        return value, err_est

    def _panel_classic(x, dx):
        """Classic single-panel integral for fallback comparison."""
        mid = x + dx / 2
        fx = _ensure_composite(f(_seeded(mid)))
        Fx = antiderivative(fx)
        right = Fx.eval_taylor(dx / 2)
        left = -Fx.eval_taylor(-dx / 2)
        return left + right

    def _adaptive(a, b, depth):
        dx = b - a
        mid = (a + b) / 2

        value, err_est = _panel_with_error(a, dx)

        if err_est >= 0:
            # Primary path: Taylor convergence check
            if err_est < tol * (abs(value) + 1e-100) or depth >= max_depth:
                return value, err_est
            if abs(value) < tol * 0.01 and err_est < tol:
                return value, err_est
        else:
            # Taylor tail overflowed — try singularity detection
            handled_left, val_left, _, _ = _detect_singularity(f, a, b, dx)
            if handled_left:
                return val_left, abs(val_left) * 1e-8

            handled_right, val_right, _, _ = _detect_singularity(f, b, a, dx)
            if handled_right:
                return val_right, abs(val_right) * 1e-8

            # Classical 3-eval fallback
            _fallback_count[0] += 1
            left_p = _panel_classic(a, dx / 2)
            right_p = _panel_classic(mid, dx / 2)
            half = left_p + right_p
            error = abs(half - value)
            if error < tol * (abs(half) + 1e-100) or depth >= max_depth:
                return half, error
            if abs(half) < tol * 0.01:
                return half, error

        # Bisect
        left_val, left_err = _adaptive(a, mid, depth + 1)
        right_val, right_err = _adaptive(mid, b, depth + 1)
        return left_val + right_val, left_err + right_err

    # Pre-subdivide into min_panels equal panels
    total_val = 0.0
    total_err = 0.0
    panel_dx = (b - a) / min_panels
    for i in range(min_panels):
        x0 = a + i * panel_dx
        x1 = x0 + panel_dx
        val, err = _adaptive(x0, x1, 0)
        total_val += val
        total_err += err

    if _fallback_count[0] > 0:
        import warnings
        warnings.warn(
            f"integrate_adaptive: Taylor convergence inconclusive on "
            f"{_fallback_count[0]} panel(s), used classical 3-eval fallback.",
            stacklevel=2)

    return Composite({0: total_val}), total_err


# =============================================================================
# IMPROPER INTEGRALS
# =============================================================================

# =============================================================================
# FIXED: Improper integrals with composite tail analysis
# =============================================================================

def improper_integral(f, a, tol=1e-8, cutoff=20):
    """Compute integral from a to +infinity. Returns (Composite, float).

    Uses composite tail analysis with power-law VERIFICATION:
    two probes at M and M/2 — true power laws give the same exponent,
    exponential/Gaussian decay gives wildly different exponents.
    """
    # Start with a smaller M — grow from here if needed
    M = max(abs(a) + 1, 5.0)

    # Detect tail behavior from composite evaluation at M
    fx = _ensure_composite(f(_seeded(M)))
    f_val = fx.st()
    f_d1 = fx.d(1)

    if (abs(f_val) > 1e-100
            and math.isfinite(f_d1)
            and abs(f_d1) > 1e-100):
        alpha = M * f_d1 / f_val

        if alpha < -1.01:
            # VERIFY: true power law gives same alpha at M/2
            # For f = C*x^alpha:  alpha(M) = alpha(M/2) = alpha  (constant)
            # For f = exp(-x^2):  alpha(M) = -2M^2, alpha(M/2) = -M^2/2  (wildly different)
            M2 = M * 0.5
            fx2 = _ensure_composite(f(_seeded(M2)))
            f_val2 = fx2.st()
            f_d12 = fx2.d(1)

            is_power_law = False
            if (abs(f_val2) > 1e-100
                    and math.isfinite(f_d12)
                    and abs(f_d12) > 1e-100):
                alpha2 = M2 * f_d12 / f_val2
                # True power law: alpha and alpha2 within 30%
                if abs(alpha - alpha2) < 0.3 * max(abs(alpha), abs(alpha2)):
                    is_power_law = True

            if is_power_law:
                # A LOCAL exponent is not the ASYMPTOTIC one.  For 1/(1+x^2)
                # alpha(5) = -1.923 while the true tail exponent is -2, and the
                # 30% agreement test above happily accepts it -- which put
                # integral 1/(1+x^2) over the whole line 0.7% off pi.
                #
                # So do not trust alpha at the first M.  Push the cutoff out,
                # accumulating the bulk rather than recomputing it, and stop
                # when the ANSWER stops moving.  That tests what is actually
                # wanted instead of a proxy for it.
                bulk, bulk_err = integrate_adaptive(f, a, M, tol=tol)
                acc = bulk.st()
                prev_total = None
                for _ in range(60):
                    fxM = _ensure_composite(f(_seeded(M)))
                    vM, dM = fxM.st(), fxM.d(1)
                    if abs(vM) <= 1e-300 or not math.isfinite(dM):
                        return Composite({0: acc}), bulk_err
                    aM = M * dM / vM
                    if aM >= -1.0:
                        # The power-law reading has broken down.  For an
                        # OSCILLATING decay like e^-x sin(x), alpha = M(cos-sin)/sin
                        # swings with the phase: -6.48 at M=5 (which passes the
                        # 30% test at M/2) and +5.42 at M=10.  Returning the
                        # earlier estimate would keep a tail computed from a
                        # classification now known to be wrong, so DISCARD it and
                        # finish by integrating outward instead.
                        prev_total = None
                        break
                    C = vM / (M ** aM)
                    total = acc - C * (M ** (aM + 1)) / (aM + 1)
                    if (prev_total is not None
                            and abs(total - prev_total)
                                <= tol * max(1.0, abs(total))):
                        return Composite({0: total}), abs(total - prev_total)
                    prev_total = total
                    nxt = M * 2.0
                    seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
                    acc += seg.st()
                    bulk_err += seg_err
                    M = nxt
                    if M > 1e12:
                        break
                if prev_total is not None:
                    return Composite({0: prev_total}), bulk_err
                # Fall through: integrate outward and stop on what the tail
                # actually CONTRIBUTES, not on how big f looks at probe points.
                # Sampling can never be phase-immune -- a window tuned to catch
                # e^-x sin(x) still lands wrong for e^-x cos(x).  Integrating
                # the next octave answers the real question and costs one more
                # panel set.  Two consecutive negligible octaves, so a single
                # near-cancelling octave cannot end it early.
                quiet = 0
                for _ in range(80):
                    nxt = M * 2.0
                    seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
                    acc += seg.st()
                    bulk_err += seg_err
                    M = nxt
                    quiet = quiet + 1 if abs(seg.st()) <= tol * max(1.0, abs(acc)) else 0
                    if quiet >= 2 or M > 1e12:
                        break
                return Composite({0: acc}), bulk_err

    # Not power-law — integrate outward until successive octaves stop
    # contributing.  A pointwise |f(M)| < tol test fires at every zero of an
    # oscillating decay, truncating a tail that is still alive; what matters is
    # the CONTRIBUTION of the next stretch, so measure that.
    bulk, bulk_err = integrate_adaptive(f, a, M, tol=tol)
    acc = bulk.st()
    quiet = 0
    for _ in range(80):
        nxt = M * 2.0
        seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
        acc += seg.st()
        bulk_err += seg_err
        M = nxt
        quiet = quiet + 1 if abs(seg.st()) <= tol * max(1.0, abs(acc)) else 0
        if quiet >= 2 or M > 1e12:
            break
    return Composite({0: acc}), bulk_err


def improper_integral_both(f, tol=1e-8):
    """Compute integral from -inf to +inf. Splits at 0.
    Returns (Composite, float)."""
    left, left_err = improper_integral(lambda x: f(-x), 0, tol=tol)
    right, right_err = improper_integral(f, 0, tol=tol)
    return left + right, left_err + right_err


def improper_integral_to(f, a, b, tol=1e-8):
    """Compute integral from a to b where f may have singularities
    at boundaries. Uses singularity detection at both endpoints.

    Splits at a fraction of the interval: analytical C*t^alpha integral
    covers the singular sub-interval, bulk integration covers the rest.
    No overlap.

    Returns (Composite, float)."""
    dx = b - a
    singular_val = 0.0
    int_a, int_b = a, b
    split_frac = 0.1

    # Check left boundary
    handled_left, _, C_left, alpha_left = _detect_singularity(f, a, b, dx)
    if handled_left:
        split_dist = abs(dx) * split_frac
        ap1 = alpha_left + 1
        # Analytical integral over [0, split_dist] only
        singular_val += C_left * (split_dist ** ap1) / ap1
        int_a = a + split_dist

    # Check right boundary
    handled_right, _, C_right, alpha_right = _detect_singularity(f, b, a, dx)
    if handled_right:
        split_dist = abs(dx) * split_frac
        ap1 = alpha_right + 1
        singular_val += C_right * (split_dist ** ap1) / ap1
        int_b = b - split_dist

    if int_a >= int_b:
        return Composite({0: singular_val}), 0.0

    bulk, bulk_err = integrate_adaptive(f, int_a, int_b, tol=tol)
    return Composite({0: bulk.st() + singular_val}), bulk_err

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def show(composite: Composite, name: str = "result"):
    """Pretty print a composite number with extracted values"""
    print(f"{name} = {composite}")
    print(f"  st() = {composite.st()}")
    if -1 in composite.c:
        print(f"  f'   = {composite.d(1)}")
    if -2 in composite.c:
        print(f"  f''  = {composite.d(2)}")
    if -3 in composite.c:
        print(f"  f'''  = {composite.d(3)}")


class TracedComposite(Composite):
    """A Composite that prints each operation as it happens."""
    def _wrap(self, result):
        if isinstance(result, Composite):
            tc = TracedComposite.__new__(TracedComposite)
            tc.c = result.c
            return tc
        return result
    def __add__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__add__(other)
        print(f"    {self}  +  {other_disp}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __radd__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__radd__(other)
        print(f"    {other_disp}  +  {self}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __sub__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__sub__(other)
        print(f"    {self}  -  {other_disp}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __mul__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__mul__(other)
        print(f"    {self}  ×  {other_disp}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __rmul__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__rmul__(other)
        print(f"    {other_disp}  ×  {self}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __truediv__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__truediv__(other)
        print(f"    {self}  ÷  {other_disp}")
        print(f"  = {result}")
        print()
        return self._wrap(result)
    def __pow__(self, n):
        result = super().__pow__(n)
        print(f"    ({self})^{n}")
        print(f"  = {result}")
        print()
        return self._wrap(result)


def trace(f: Callable, at: float = None, to: float = None) -> Composite:
    """Trace composite computation showing ALL intermediate steps."""
    if to is not None:
        if to == float('inf'):
            x = TracedComposite({1: 1.0})
            print(f"\n=== TRACE: lim(x→∞) ===")
            print(f"Let x = |1|₁  (INF)\n")
        elif to == float('-inf'):
            x = TracedComposite({1: -1.0})
            print(f"\n=== TRACE: lim(x→-∞) ===")
            print(f"Let x = |-1|₁  (-INF)\n")
        elif to == 0:
            x = TracedComposite({-1: 1.0})
            print(f"\n=== TRACE: lim(x→0) ===")
            print(f"Let x = |1|₋₁  (ZERO)\n")
        else:
            x = TracedComposite({0: float(to), -1: 1.0})
            print(f"\n=== TRACE: lim(x→{to}) ===")
            print(f"Let x = |{to}|₀ + |1|₋₁\n")
    elif at is not None:
        x = TracedComposite({0: float(at), -1: 1.0})
        print(f"\n=== TRACE: f'({at}) ===")
        print(f"Let x = |{at}|₀ + |1|₋₁  (i.e., {at} + h)\n")
    else:
        x = TracedComposite({-1: 1.0})
        print(f"\n=== TRACE ===")
        print(f"Let x = |1|₋₁  (ZERO)\n")
    result = f(x)
    if isinstance(result, (int, float)):
        result = Composite(result)
    print(f"RESULT: {result}")
    if to is not None:
        print(f"Limit = {result.st()}")
    else:
        print(f"f({at if at else 0}) = {result.st()}")
        if -1 in result.c:
            print(f"f'({at if at else 0}) = {result.d(1)}")
    return Composite(result.c) if isinstance(result, TracedComposite) else result


def translate(f: Callable, at: float = None, to: float = None) -> Composite:
    """Show the composite translation WITHOUT resolving."""
    if to is not None:
        if to == float('inf'):
            x = INF
            sub_str = "x = INF"
        elif to == float('-inf'):
            x = -INF
            sub_str = "x = -INF"
        elif to == 0:
            x = ZERO
            sub_str = "x = ZERO"
        else:
            x = _seeded(to)
            sub_str = f"x = _seeded({to})"
    elif at is not None:
        x = _seeded(at)
        sub_str = f"x = _seeded({at})"
    else:
        x = ZERO
        sub_str = "x = ZERO"
    result = f(x)
    print(f"Substitution: {sub_str}")
    print(f"Translation:  {result}")
    print(f"")
    if to is not None:
        print(f"Limit = {result.st()}")
    else:
        print(f"f({at if at else 0}) = {result.st()}")
        if -1 in result.c:
            print(f"f'({at if at else 0}) = {result.d(1)}")
        if -2 in result.c:
            print(f"f''({at if at else 0}) = {result.d(2)}")
    return result


def verify_derivative(f: Callable, f_prime: Callable, at: float, tol: float = 1e-6) -> bool:
    """Verify that f_prime is indeed the derivative of f at a point."""
    computed = derivative(f, at)
    expected = f_prime(at) if callable(f_prime) else f_prime
    return abs(computed - expected) < tol


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    """Run basic tests to verify the library works"""
    print("=" * 60)
    print("COMPOSITE LIBRARY TEST SUITE (FIXED v3: EXPRESSED ZERO)")
    print("=" * 60)

    tests = []

    # Derivative tests
    print("\n--- Derivatives ---")

    d1 = derivative(lambda x: x**2, at=3)
    tests.append(("d/dx[x²] at x=3", d1, 6))
    print(f"d/dx[x²] at x=3 = {d1}, expected 6 {'✓' if abs(d1-6)<1e-6 else '✗'}")

    d2 = derivative(lambda x: x**3, at=2)
    tests.append(("d/dx[x³] at x=2", d2, 12))
    print(f"d/dx[x³] at x=2 = {d2}, expected 12 {'✓' if abs(d2-12)<1e-6 else '✗'}")

    d3 = derivative(lambda x: sin(x), at=0)
    tests.append(("d/dx[sin(x)] at x=0", d3, 1))
    print(f"d/dx[sin(x)] at x=0 = {d3}, expected 1 {'✓' if abs(d3-1)<1e-6 else '✗'}")

    d4 = nth_derivative(lambda x: x**5, n=3, at=2)
    tests.append(("d³/dx³[x⁵] at x=2", d4, 240))
    print(f"d³/dx³[x⁵] at x=2 = {d4}, expected 240 {'✓' if abs(d4-240)<1e-6 else '✗'}")

    # Limit tests
    print("\n--- Limits ---")

    l1 = limit(lambda x: sin(x)/x, as_x_to=0)
    tests.append(("lim sin(x)/x as x→0", l1, 1))
    print(f"lim sin(x)/x as x→0 = {l1}, expected 1 {'✓' if abs(l1-1)<1e-6 else '✗'}")

    l2 = limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)
    tests.append(("lim (x²-4)/(x-2) as x→2", l2, 4))
    print(f"lim (x²-4)/(x-2) as x→2 = {l2}, expected 4 {'✓' if abs(l2-4)<1e-6 else '✗'}")

    l3 = limit(lambda x: (1 - cos(x))/(x**2), as_x_to=0)
    tests.append(("lim (1-cos(x))/x² as x→0", l3, 0.5))
    print(f"lim (1-cos(x))/x² as x→0 = {l3}, expected 0.5 {'✓' if abs(l3-0.5)<1e-6 else '✗'}")

    l4 = limit(lambda x: (exp(x) - 1)/x, as_x_to=0)
    tests.append(("lim (eˣ-1)/x as x→0", l4, 1))
    print(f"lim (eˣ-1)/x as x→0 = {l4}, expected 1 {'✓' if abs(l4-1)<1e-6 else '✗'}")

    # Special values
    print("\n--- Special Values ---")

    s1 = (ZERO / ZERO).st()
    tests.append(("0/0", s1, 1))
    print(f"ZERO / ZERO = {s1}, expected 1 {'✓' if abs(s1-1)<1e-6 else '✗'}")

    s2 = (INF * ZERO).st()
    tests.append(("∞ × 0", s2, 1))
    print(f"INF * ZERO = {s2}, expected 1 {'✓' if abs(s2-1)<1e-6 else '✗'}")

    s3 = ((R(5) * ZERO) / ZERO).st()
    tests.append(("(5×0)/0", s3, 5))
    print(f"(R(5) * ZERO) / ZERO = {s3}, expected 5 {'✓' if abs(s3-5)<1e-6 else '✗'}")

    # Fix 1 verification: transcendentals on plain floats return Composite
    print("\n--- Fix 1: Transcendentals return Composite ---")

    sin_plain = sin(0.5)
    is_composite = isinstance(sin_plain, Composite)
    tests.append(("sin(0.5) returns Composite", 1 if is_composite else 0, 1))
    print(f"sin(0.5) returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  sin(0.5).st() = {sin_plain.st():.6f}, math.sin(0.5) = {math.sin(0.5):.6f}")

    exp_plain = exp(1.0)
    is_composite = isinstance(exp_plain, Composite)
    tests.append(("exp(1.0) returns Composite", 1 if is_composite else 0, 1))
    print(f"exp(1.0) returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  exp(1.0).st() = {exp_plain.st():.6f}, math.exp(1.0) = {math.exp(1.0):.6f}")

    # Fix 2 verification: integration returns Composite
    print("\n--- Fix 2: Integration returns Composite ---")

    int_result, int_err = integrate_adaptive(lambda x: x**2, 0, 1, tol=1e-8)
    is_composite = isinstance(int_result, Composite)
    tests.append(("integrate_adaptive returns Composite", 1 if is_composite else 0, 1))
    print(f"integrate_adaptive returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  ∫x² dx from 0 to 1 = {int_result.st():.6f}, expected 0.333333")

    # v2 verification: line integral through integrate_adaptive
    print("\n--- v2: Line Integral (Composite-First) ---")

    line_result = integrate(
        [lambda x, y: y, lambda x, y: -x],
        (0, 2 * math.pi),
        curve=lambda t: [cos(t), sin(t)]
    )
    expected_line = -2 * math.pi
    line_ok = abs(line_result - expected_line) < 0.1
    tests.append(("∫_C F·dr (unit circle)", line_result, expected_line))
    print(f"∫_C [y,-x]·dr (unit circle) = {line_result:.6f}, expected {expected_line:.6f} {'✓' if line_ok else '✗'}")

    # v3 verification: expressed zero preservation
    print("\n--- Fix 3: Expressed Zero Preservation ---")

    zero_sub = R(1) - R(1)
    has_dim0 = 0 in zero_sub.c
    tests.append(("R(1)-R(1) retains dim 0", 1 if has_dim0 else 0, 1))
    print(f"R(1) - R(1) = {zero_sub}")
    print(f"  dim 0 retained: {has_dim0} {'✓' if has_dim0 else '✗'}")
    print(f"  .c = {zero_sub.c}")

    comp_zero = Composite(0)
    has_dim0 = 0 in comp_zero.c
    tests.append(("Composite(0) retains dim 0", 1 if has_dim0 else 0, 1))
    print(f"Composite(0) = {comp_zero}")
    print(f"  dim 0 retained: {has_dim0} {'✓' if has_dim0 else '✗'}")

    empty = Composite()
    is_empty = len(empty.c) == 0
    tests.append(("Composite() is truly empty", 1 if is_empty else 0, 1))
    print(f"Composite() = {empty}")
    print(f"  truly empty: {is_empty} {'✓' if is_empty else '✗'}")

    # All derivatives at once
    print("\n--- All Derivatives ---")

    derivs = all_derivatives(lambda x: exp(x), at=0, up_to=5)
    print(f"All derivatives of eˣ at x=0: {[round(d,2) for d in derivs]}")
    print(f"Expected: [1, 1, 1, 1, 1, 1] {'✓' if all(abs(d-1)<1e-6 for d in derivs) else '✗'}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, actual, expected in tests if abs(actual - expected) < 0.1)
    print(f"PASSED: {passed}/{len(tests)}")
    print("=" * 60)

    return passed == len(tests)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_tests()

    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)

    print("\n# Direct composite computation:")
    print("h = ZERO")
    print("x = R(3) + h")
    x = R(3) + h
    result = x**2
    print(f"(R(3) + h)**2 = {result}")
    print(f"  Value at x=3: {result.st()}")
    print(f"  Derivative:   {result.d(1)}")

    print("\n# High-level API:")
    print(f"derivative(lambda x: x**2, at=3) = {derivative(lambda x: x**2, at=3)}")
    print(f"limit(lambda x: sin(x)/x, as_x_to=0) = {limit(lambda x: sin(x)/x, as_x_to=0)}")

    print("\n# Line integral (composite-first):")
    print("∫_C [y,-x]·dr around unit circle:")
    result = integrate(
        [lambda x, y: y, lambda x, y: -x],
        (0, 2 * math.pi),
        curve=lambda t: [cos(t), sin(t)]
    )
    print(f"  = {result:.6f}, expected {-2*math.pi:.6f}")
