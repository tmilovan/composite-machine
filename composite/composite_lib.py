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
    from composite.composite_lib import *

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

class NotRepresentableError(ValueError):
    """The quantity exists but has no composite for it.

    Distinct from LimitDoesNotExistError, which asserts something about a
    LIMIT.  sin(1/h) has no representation here -- bounded by 1, no limit, not
    eventually monotone, so outside a Hardy field at any basis extension --
    but that says nothing about an expression CONTAINING it: x*sin(1/x) tends
    to 0 perfectly well.  Raising the stronger error stopped limit() from
    recovering those, because it re-raises "provably does not exist" without
    trying.  A ValueError subclass, so the existing domain-error path picks it
    up and extrapolates.
    """


class StandardPartUndefinedError(ValueError):
    """Asked for the standard part of something that has none.

    An infinitesimal IS infinitely close to zero, so st(|1|_-1) = 0.0 is
    correct and is not this case.  A quantity with a positive grade and
    nothing below it is unbounded -- there is no real number it approaches,
    and reporting the grade-0 coefficient (0.0, because the grade is absent)
    would say "vanishing" about something infinite.
    """


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
    if len(dims) == 0:
        return c                       # NOTHING has no dimension to convert
    _warn_zero_operand()
    dims = dims.copy()
    vals = vals.copy()
    # to_arrays is sorted ascending, so [0] is the lowest dimension.
    #
    # _dim_shift, not `- 1`: on a vector dimension the scalar spelling raises
    # "unsupported operand for -: 'tuple' and 'int'", and it raised from inside
    # R1 -- so `0.0 + ln(1/h)`, the most ordinary line anyone would write,
    # could not be evaluated at all.  Shifting the POWER axis is the right
    # move rather than the leading axis: it lowers the dimension in every case
    # ((-1, 1) < (0, 1) lexicographically), whereas decrementing the leading
    # axis would turn the zero at (0, 1) into (0, 0) -- finite, not
    # infinitesimal, which is not what R1 means.  R(0) already produced
    # (-1, 0) by this convention, so this only makes the two paths agree.
    dims[0] = _dim_shift(dims[0], -1)
    vals[0] = 1.0
    return Composite._wrap(c._backend.create_from_terms(dims, vals),
                           c._backend, demote=False)


def _scalar_operand(other):
    """A Python or numpy scalar entering an operation as the other operand.

    A WRITTEN ZERO IS AN EXPRESSED ZERO: |0|_0, which R1 converts.  That is
    not a cost the system imposes on the unwary -- it IS the system.  An
    expressed zero is an infinitesimal, and `1 - 1 != 0` is the same statement.

    This was briefly changed so that a bare scalar zero produced NOTHING, on
    the argument that a zero arriving from data has no event behind it.  The
    argument was wrong on both counts.  §0 already draws the line: NOTHING is
    the ABSENCE of a term, and a zero that was written is not absent -- the
    event R1 records is the EXPRESSION of the zero, and writing it is that
    event.  Worse, making a keyboard-reachable zero into an additive identity
    turns this back into a conventional ring with an extra symbol attached,
    and two zeros obeying different laws is worse than one obeying one.

    The bug that motivated the change was never here.  `d.get(k, 0.0)`
    MANUFACTURED a zero for a Pade coefficient that did not exist -- it
    expressed a zero the mathematics never expressed -- and the lateral Borel
    integral returned 3224 where the answer is 0.697.  The fix is `.get(k)`
    and skip the absent term, which is what the warning below has always said.
    """
    if other == 0:
        return Composite({0: 0.0})
    return Composite(float(other))


_ZERO_OPERAND_MSG = (
    "a zero operand converted: |0|_d -> |1|_(d-1) (R1). That is the rule and "
    "it is correct -- an expressed zero IS an infinitesimal, which is the same "
    "statement as 1 - 1 != 0. It is reported because the dimension-0 value "
    "stays right while every derivative moves, which is the worst failure "
    "shape there is. Three ways to arrive here: a written zero (`c + 0`, "
    "`acc = 0`) -- correct, and Composite(0) or ZERO silences it; a zero that "
    "was MANUFACTURED for something absent (`d.get(k, 0.0)` for a coefficient "
    "that does not exist) -- a bug, use `d.get(k)` and skip, or Composite({}) "
    "for an accumulator, and it returned 3224 for a value of 0.697 once; or a "
    "computation that cancelled to all-zero -- correct, and nothing announced "
    "it before this warning existed."
)


def _warn_zero_operand():
    """R1 made audible, once, at the only place a zero actually converts.

    This used to be two warnings.  A bare Python 0 became |0|_0 and then
    converted here anyway, so `c + 0` fired both the seed warning and this
    one, while `c / 0` fired neither -- the seed warning was never wired into
    __truediv__.  _r1 is the single point every route passes through, so the
    message lives here and covers all three: written, manufactured, cancelled.
    """
    import warnings
    warnings.warn(_ZERO_OPERAND_MSG, stacklevel=5)


def is_nothing(x):
    """True for NOTHING -- no term at any dimension.  The classical zero.

    This is what a bare Python 0 becomes when it enters an operation, and it
    is an additive identity and a multiplicative annihilator.  It is NOT a
    zero at a dimension and does not convert under R1.
    """
    return isinstance(x, Composite) and not x.coeffs_dict()


def is_vanishing(x):
    """True for a zero that HAS a dimension -- one that converts under R1.

    `c - c`, `Composite({0: 0.0})`, `L - L`.  These are the values for which
    `x == 0` is False, because a dimension existed and was annihilated, and
    the trace of that event is what `1 - 1 != 0` is a claim about.
    """
    return (isinstance(x, Composite) and bool(x.coeffs_dict())
            and _is_wholly_zero(x))


def is_zero(x):
    """True when x carries no value: NOTHING, or a zero at some dimension.

    USE THIS RATHER THAN `x == 0`.  `x == 0` compares against NOTHING, so it
    is True for NOTHING and False for a dimensioned zero -- which is R1 and R6
    working exactly as specified, and which surprises every reader once.
    is_zero() asks the question people actually mean; is_vanishing()
    distinguishes the case that still carries a dimension.
    """
    if isinstance(x, (int, float)):
        return x == 0
    return is_nothing(x) or is_vanishing(x)


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
                from composite.backends.vector_dim_backend import dom_sorted as _dom_sorted
                sorted_dims = _dom_sorted(coefficients.keys())
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
        if isinstance(other, (int, float)):
            other = _scalar_operand(other)
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
        if isinstance(other, (int, float)):
            other = _scalar_operand(other)
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        return Composite._wrap(a._backend.add(a._data, a._backend.negate(b._data)),
                               a._backend,
                               complete=_min_complete(self, other))

    def __rsub__(self, other):
        left = _scalar_operand(other)
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
                    # self._backend, not the global one: __mul__ passes it and
                    # __truediv__ did not, so `x / 2` on a vector composite
                    # wrapped DictData in sparse-dense methods.  sinh, cosh,
                    # tanh and atan all divide by a scalar at the end.
                    self._backend, complete=self._complete)
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
                # broadcast `-`, so shift each one explicitly.  PAD FIRST: zip
                # over different-length tuples truncates to the shorter and
                # drops the deepest axes without a word, which turned
                # 1/ln(ln(1/h)) -- dim (0,0) over (0,0,1) -- into a plain 1.
                from composite.backends.vector_dim_backend import pair, canon
                shifted = []
                for d in my_dims:
                    _a, _b = pair(d, div_dim)
                    shifted.append(canon(tuple(x - y for x, y in zip(_a, _b))))
            else:
                shifted = my_dims - div_dim
            return Composite._wrap(
                a._backend.create_from_terms(shifted, my_vals / div_coeff),
                a._backend, complete=_min_complete(self, other))

        # deconvolve is lexicographic long division: it repeatedly takes
        # max(rem).  With infinitesimals on TWO independent axes that starves
        # one of them -- (0,-k) outranks (-1,anything), so the y-axis geometric
        # series (which never terminates) eats every iteration and the x-axis
        # terms sit in the remainder until the cap runs out.  1/(1+x+y) came
        # back with NO x-dependence at all.  The geometric series treats the
        # whole infinitesimal part as one object and has no such ordering, so
        # use it when the divisor genuinely spans more than one axis.
        if _spans_multiple_axes(b) and b.st() != 0.0:
            return a * _reciprocal(b, terms=_effective_terms(15))

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
        left = _scalar_operand(other)
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
        """Standard part: the real number this is infinitely close to.

        THE DOMINANT GRADE DECIDES, not the presence of any particular one:

          any positive grade   ->  UNDEFINED
              Unbounded, so no real number is approached.  0.0 there is not
              imprecise, it is inverted -- sin(1)/sin(h) is infinite and
              `abs(st(x)) < 1e-9` was True for it.  Testing for NEGATIVE
              grades instead was tried and misses exactly that case, because
              |0.841471|_+1 + |0.140245|_-1 has negatives too; what makes it
              unbounded is the +1 on top.  For the same reason 1 + 1/h has no
              standard part despite carrying a grade-0 term.

          otherwise            ->  coefficient at grade 0, 0.0 if absent
              An infinitesimal IS infinitely close to zero, so st(|1|_-1) and
              st(h*h) are 0.0 -- correct, and not the case above.

          empty                ->  0.0
              NOTHING, and reading it expresses the zero (R6).  Provisional:
              this is the one case that may become undefined later.
        """
        dims, vals = self._backend.to_arrays(self._data)
        nz = [d for d, v in zip(dims, vals) if v != 0.0]
        if not nz:
            return 0.0
        if any(_dim_positive(d) for d in nz):
            raise StandardPartUndefinedError(
                f"no standard part: {str(self)[:60]} is unbounded -- its "
                f"dominant grade is positive")
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

        pos = [d for d, v in zip(dims, vals) if _dim_positive(d) and v != 0]
        if not pos:
            return None
        if any(isinstance(d, tuple) for d in pos):
            from composite.backends.vector_dim_backend import as_vec
            return max(pos, key=lambda d: as_vec(d))
        return max(dim_cast(d) for d in pos)

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

    def eval_taylor_axes(self, h_value):
        """eval_taylor, but the NON-power axes survive as dimensions.

        eval_taylor substitutes h and returns a FLOAT, which adds terms that
        differ only on a later axis into the same number: (-1, 0) and (-1, -1)
        both become coefficient * h**1.  That is correct when the power axis is
        the only one in play, and it is what collapses a box integral -- the
        second variable's structure is destroyed at exactly this step, not at
        .st() as it appears.

        Returns {dim: coeff} with the power component consumed and every other
        component kept, so the caller still has a series in the remaining
        variables.  eval_taylor is left alone: its callers want the float.
        """
        dims, vals = self._backend.to_arrays(self._data)
        out = {}
        for d, v in zip(dims, vals):
            p = d[0] if isinstance(d, tuple) else d
            if p >= 0:
                continue
            if isinstance(d, tuple):
                from composite.backends.vector_dim_backend import canon
                key = canon((0,) + tuple(d[1:]))
            else:
                key = 0
            out[key] = out.get(key, 0.0) + float(v) * h_value ** (-float(p))
        return out

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


def _dim_nonzero(d):
    """Is this dimension anything other than THE zero dimension?

    `d != 0` is the scalar spelling, and a tuple is never equal to 0 -- so the
    vector zero (0, 0) tested as non-zero, sqrt took its leading-term branch,
    divided x by 1, and recursed on itself until the stack ran out.
    """
    if isinstance(d, tuple):
        return any(c != 0 for c in d)
    return d != 0


def _zero_dim_like(x):
    """The zero dimension in x's own KIND -- 0, or (0, 0, ...) for vectors.

    sorted() cannot order a tuple against an int, so a scalar 0 mixed into a
    dict of vector dims raises on construction.
    """
    for d in x.c:
        if isinstance(d, tuple):
            return tuple(0 for _ in range(len(d)))
    return 0


def _dim_shift(d, k):
    """Move a dimension by k on the POWER axis, leaving the log axes alone.

    Integration and differentiation change the order in x, which is the power
    component; the log components ride along unchanged.  `d - 1` is the scalar
    spelling and raises "unsupported operand for -: 'tuple' and 'int'" as soon
    as a log axis is present -- which is what stopped erf on a log-axis
    argument.
    """
    if isinstance(d, tuple):
        return (d[0] + k,) + tuple(d[1:])
    return d + k


def _dim_scaled(d, factor):
    """Scale a dimension by `factor`, componentwise for a vector dim.

    sqrt halves the index; for (p, l) that is (p/2, l/2), since
    sqrt(h**p * L**l) = h**(p/2) * L**(l/2).  Writing `d / 2` works only for a
    scalar and raises "unsupported operand for /: 'tuple'" the moment a log
    axis is involved.
    """
    if isinstance(d, tuple):
        return tuple(c * factor for c in d)
    return dim_cast(d * factor)


def _dim_positive(d):
    """Dimension lexicographically ABOVE zero -- the infinite side, any depth.

    `d > 0` is the scalar spelling of this and it raises TypeError the moment a
    dimension is a tuple.  The first NON-ZERO component decides: (0,1) is
    log(1/h), infinite; (0,0,-1) is 1/loglog(1/h), infinitesimal.  Reading only
    the power component -- the previous shortcut -- calls every pure log term
    finite, which routes infinite arguments down the Taylor path.
    """
    if not isinstance(d, tuple):
        return d > 0
    for c in d:
        if c != 0:
            return c > 0
    return False


def _dim_negative(d):
    """Dimension lexicographically BELOW zero -- the infinitesimal side."""
    if not isinstance(d, tuple):
        return d < 0
    for c in d:
        if c != 0:
            return c < 0
    return False


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
    if not h_terms:
        return None          # nothing infinitesimal: no order bound to state
    m = min(_lead_order(d) for d in h_terms)
    if m <= 0:
        return None
    return m * (terms - 1)


def _lead_order(d):
    """Order along the DOMINANT axis -- positive for an infinitesimal.

    _dim_order reads only the POWER component and returns 0 for anything living
    purely on a log axis, so (0, -1) -- which IS an infinitesimal, 1/ln(1/h) --
    read as order zero and was excluded from every "infinitesimal terms" test.
    That is how a truncated log-axis series kept reporting itself EXACT.

    This has now been the same mistake four times in one day: max_positive_dim,
    the non-dyadic guard, atan's antiderivative filter, and here.  _dim_order is
    correct for what it means (Taylor order on the power axis) and dangerous for
    what it reads like, so anything asking "how small is this dimension" across
    axes must use THIS instead.
    """
    if not isinstance(d, tuple):
        return -d
    for c in d:
        if c != 0:
            return -c
    return 0


def _infinitesimal_terms(x):
    """The strictly-infinitesimal part of x, as a {dim: coeff} dict.

    Infinitesimal means lexicographically BELOW zero, on whichever axis leads --
    not merely "negative power component".
    """
    return {d: c for d, c in x.c.items() if c != 0.0 and _dim_negative(d)}


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


def _bounded_at_inf(func, x, terms=12):
    """Evaluate a bounded transcendental at an infinite composite argument.

    For monotonic bounded functions (atan, tanh): math.func(±inf) returns the
    correct asymptotic value (e.g. atan(inf) = pi/2).  Result via R().

    For oscillatory functions (sin, cos): math.func(±inf) raises ValueError.
    The value cannot be pinpointed at grade 0 -- sin(1/h) is bounded by 1 and
    has no limit -- but that does not mean there is nothing to return.  It is
    handled the way sqrt handles an odd dimension: the DIMENSION DEGRADES.

        sin(|c|_d) = |sin(c)|_(d/2)        for d > 0

    A RANGE IS NOT A POINT.  sin(1/h) takes every value in [-1, 1] -- it is
    0 and 1 infinitely often at arbitrarily small eps -- and a composite holds
    ONE value at ONE grade.  That is the whole obstruction, and it is why no
    rule on the dimension repairs it.  Every candidate was measured:

      grade d   (dimension kept)  revertible, st undefined, but the damping
                cancels: x*sin(1/x) came back |sin 1|_0 = 0.841471, not 0,
                and 1/sin(1/h) came back INFINITESIMAL for something that is
                genuinely unbounded.              test_limits 87/109
      grade d/2 squeeze fixed, magnitude still wrong: limit() called
                sin(sin(1/x)) divergent, and it never exceeds 1.   90/105
      grade 0   squeeze and magnitude both right, but st becomes DEFINED as
                sin(1), and asin no longer reverts the skip.       94/109
      d**(1/d)  fails the squeeze at d=1 exactly, and is not monotone.
      asin(d)   fails the squeeze at d=1, and is undefined for d > 1.
      sin(d)    works on (0, pi) only; sign flips past pi, and it inherits
                sin's non-monotonicity, so the dominance order reverses.

    The two requirements that decide it contradict: st-undefined needs a
    positive grade, an honest magnitude needs a non-positive one.  So the
    function refuses instead of choosing which to break.   103/105

    The coefficient reading that survived all this is worth keeping in mind:
    |sin(1)|_1 is ONE SAMPLE of the oscillation -- a y-value at a known
    argument, which is why asin could inverted it.  A sample cannot answer
    anything that needs the whole range, and the limit, the supremum and
    whether the reciprocal blows up are all of that kind.
    """
    max_d = x.max_positive_dim()
    sign = 1.0 if x.coeff(max_d) > 0 else -1.0

    # atan has an ASYMPTOTIC SERIES at an unbounded argument and it is exactly
    # representable here, because 1/x is infinitesimal when x has a positive
    # grade:
    #     atan(x) = +-pi/2 - 1/x + 1/(3x^3) - 1/(5x^5) + ...
    # Returning pi/2 alone is the LIMIT, not the value: atan(1/h) is
    # pi/2 - h + h^3/3 - ..., so the leading term was right and every order
    # below it was silently dropped.  The same identity covers both signs --
    # atan(x) + atan(1/x) is +pi/2 for x > 0 and -pi/2 for x < 0, and the
    # series for atan(1/x) is the same either way.
    #
    # tanh is NOT done this way: tanh(1/h) = 1 - 2exp(-2/h) + ..., and that
    # correction is exponentially flat, so it lies outside the value group
    # entirely.  1 is right to every representable order.
    if func is math.atan:
        u = R(1) / x
        u2 = u * u
        acc = R(sign * math.pi / 2)
        term = u
        for k in range(_effective_terms(terms)):
            acc = acc + (term / float(2 * k + 1)) * (1.0 if k % 2 else -1.0)
            term = term * u2
        return acc

    try:
        return R(func(sign * float('inf')))
    except (ValueError, OverflowError):
        raise NotRepresentableError(
            f"{func.__name__}(x) at an unbounded argument is a RANGE, not a "
            f"point: sin(1/h) takes every value in [-1, 1] and a composite "
            f"holds one value at one grade.  No rule on the grade repairs "
            f"that -- see the analysis above."
        ) from None


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


def _carries_vector(x):
    """Does this composite actually HOLD a vector dimension?

    Not the same question as whether its backend advertises VECTOR_DIMS.  ln()
    puts a term on the log axis whatever backend it was called on, so a plain
    scalar backend routinely ends up holding tuple dimensions -- and gating the
    log handling on the backend flag meant exp() never recognised them there.
    On DictBackend that made h**0.25 refuse: ** routes through
    exp(n*ln(h)), ln gave the log-axis term, exp did not see it, and the
    positive-grade check rejected a perfectly representable object.
    """
    try:
        dims, _ = x._backend.to_arrays(x._data)
    except Exception:
        return False
    return any(isinstance(d, tuple) for d in dims)


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
    if not terms:
        return Composite._wrap(
            be.create_from_terms(np.array([], dtype=DIM_DTYPE),
                                 np.array([], dtype=np.float64)),
            be, demote=False)
    # SORTED, and as the same array kinds Composite.__init__ builds.  Passing a
    # dict's insertion order straight through is what broke asin: its result is
    # assembled with key 0 first and the negative dims after, and the
    # sparse-dense backend reads runs off the order it is given -- so the
    # standard part was dropped and asin(x) came back with st() == 0.  _like
    # only ever worked because every earlier caller happened to pass a sorted
    # dict.
    # dom_sorted, not sorted: _r1 shifts dims[0] on the stated contract that
    # to_arrays comes back ascending and [0] is the LOWEST dimension.  Raw
    # tuple order puts a canonical (0,0) BELOW (0,0,-3), so "lowest" could be
    # a term that dominates it, and R1 would uplift the wrong one.
    from composite.backends.vector_dim_backend import dom_sorted as _dom_sorted
    sorted_dims = _dom_sorted(terms.keys())
    if isinstance(sorted_dims[0], tuple):
        dims = np.empty(len(sorted_dims), dtype=object)
        for _i, _d in enumerate(sorted_dims):
            dims[_i] = _d
    else:
        dims = np.array(sorted_dims, dtype=DIM_DTYPE)
    vals = np.array([terms[d] for d in sorted_dims], dtype=np.float64)
    return Composite._wrap(be.create_from_terms(dims, vals), be, demote=False)


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


def _vec_unit(k, width=None):
    """The dimension that IS the k-th basis element: 1 at index k, 0 elsewhere."""
    from composite.backends.vector_dim_backend import ensure_depth
    w = ensure_depth(max(k + 1, width or 0))
    return tuple(1 if i == k else 0 for i in range(w))


def _sole_log_index(d):
    """(k, e) when dim d is e copies of ONE basis element at index k >= 1.

    exp can only lower a term that is a single basis element: exp(v*B_k) is
    B_{k-1}**v.  A term like (0, 2) -- that is (log x)**2 -- has no image at
    any depth, because exponentiating a SQUARED log produces a scale outside
    this family entirely.  That is a different refusal from "the basis is too
    shallow", and conflating the two is what made the old k == 1 test look like
    a depth limit when it is really an exponent limit.
    """
    from composite.backends.vector_dim_backend import as_vec
    v = as_vec(d)
    nz = [(i, c) for i, c in enumerate(v) if c != 0]
    if len(nz) != 1:
        return None
    k, e = nz[0]
    return (k, e) if k >= 1 else None


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
            # A term changes SCALE under exp when its power component is zero
            # and it is positive on some log axis -- at ANY depth, not only the
            # first.  Lexicographic order decides the sign: the first non-zero
            # component after the power is what dominates.
            rest_comps = d[1:]
            lead = next((c for c in rest_comps if c != 0), 0)
            if d[0] == 0 and lead > 0:
                logs[d] = v
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
    #
    # _dim_nonzero, not `d != 0`: the VECTOR zero is (0, 0), and a tuple is
    # never equal to an int, so a vector-keyed standard part tested as an
    # infinitesimal.  sin then built its series with a = st(x) AND h still
    # holding that same standard part, so sin(5) came back as sin(10).
    return any(_dim_nonzero(d) and v != 0.0 for d, v in zip(dims, vals))


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
    _nz = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}
    # _like, not Composite({...}): the bare constructor binds whatever backend
    # is globally ACTIVE, so a vector-dimension argument had its tuples handed
    # to numpy -- "setting an array element with a sequence" -- or came back as
    # DictData wearing sparse-dense methods.
    #
    # _dim_nonzero, not `d != 0`: h must hold the part AWAY from dimension
    # zero.  The scalar spelling let the vector zero (0, 0) through, so the
    # standard part was counted twice -- once as a, once inside h.
    h = _like(x, {d: c for d, c in x.c.items() if _dim_nonzero(d)})
    sin_a, cos_a = math.sin(a), math.cos(a)
    _one = _like(x, {_unit_dim(x): 1.0})
    sin_h = _like(x, {})
    cos_h = _one
    h_power = _one
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    # Skip zero-coefficient terms to avoid 0 * Composite uplift
    result = _like(x, {})
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
    _nz = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}
    # _like, not Composite({...}): the bare constructor binds whatever backend
    # is globally ACTIVE, so a vector-dimension argument had its tuples handed
    # to numpy -- "setting an array element with a sequence" -- or came back as
    # DictData wearing sparse-dense methods.
    #
    # _dim_nonzero, not `d != 0`: h must hold the part AWAY from dimension
    # zero.  The scalar spelling let the vector zero (0, 0) through, so the
    # standard part was counted twice -- once as a, once inside h.
    h = _like(x, {d: c for d, c in x.c.items() if _dim_nonzero(d)})
    sin_a, cos_a = math.sin(a), math.cos(a)
    _one = _like(x, {_unit_dim(x): 1.0})
    sin_h = _like(x, {})
    cos_h = _one
    h_power = _one
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    result = _like(x, {})
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

    if LOG_SCALE and (getattr(x._backend, "VECTOR_DIMS", False)
                      or _carries_vector(x)):
        _logs, _rest = _log_part(x)
        if _logs:
            # exp(k * ln(1/h)) = (1/h)^k = h^(-k), which sits at power +k.
            out = None
            for _d, v in _logs.items():
                # exp(v * B_k) = B_{k-1} ** v: one step DOWN the basis, at any
                # depth.  B_1 = ln(1/h) so exp lands on the power axis; B_2 =
                # ln(ln(1/h)) so exp lands on the log axis; and so on.
                _ke = _sole_log_index(_d)
                if _ke is None or _ke[1] != 1:
                    from composite.backends.vector_dim_backend import BASIS
                    raise ValueError(
                        f"exp of {_d}: a log term can only be exponentiated when "
                        f"it is ONE basis element to the first power. "
                        f"{_d} is not, and no depth fixes that -- exp of a "
                        f"squared or mixed log leaves this family of scales. "
                        f"(basis {tuple(BASIS)})")
                _k = _ke[0]
                _tgt = list(_vec_unit(_k - 1))
                _tgt[_k - 1] = v
                t = _vec_composite({tuple(_tgt): 1.0})
                out = t if out is None else out * t
            if _rest:
                # A remainder that is only the standard part needs no series --
                # and recursing would rebuild it through the ACTIVE backend,
                # which cannot hold vector dimensions.
                _zero = tuple(0 for _ in range(len(next(iter(_rest)))))
                if set(_rest) <= {_zero}:
                    out = out * _vec_composite(
                        {_zero: math.exp(_rest.get(_zero, 0.0))})
                else:
                    out = out * exp(_vec_composite(_rest), terms)
            return out

    # OUTSIDE THE VALUE GROUP.  exp of a positive grade is not a large number,
    # it is a different LEVEL: exp(1/h) is above every power of 1/h, and
    # exp(-1/h) is nonzero and below every power of h -- the flat object.  No
    # finite-rank dimension names either, because the transmonomial ordering
    # stops being finite-rank lex once exponentials appear; grades would have
    # to become recursive expressions rather than coordinates.  That is the
    # step from Hahn/Hardy series to transseries, and this library stops at
    # powers and iterated logs.
    #
    # It used to apply the Maclaurin series regardless, so exp(1/h) and
    # exp(-1/h) BOTH came back as 1 - 1 + 1/2 - ... truncated at 15 terms,
    # with standard part 1.0 for each -- two objects at opposite ends of the
    # scale reported as the same finite number, and any comparison between
    # them decided by round-off.  A series evaluation is sound exactly when
    # the argument is (finite standard part) + (strictly negative grade), and
    # that is what is checked here.
    if _has_positive_dims(x):
        raise NotRepresentableError(
            f"exp of a positive grade is outside this value group: "
            f"exp(1/h) is above every power and exp(-1/h) is below every "
            f"power (nonzero, flat).  Naming either needs exponential "
            f"levels -- transseries -- not a power-and-log dimension.")

    a = x.st()
    # 7.1: a term exists iff its coefficient is nonzero -- exactly zero,
    # not 'small'.  A tolerance here silently discards real content.
    # _dim_nonzero, not `d != 0`: non_zero must hold the part AWAY from
    # dimension zero, because the result is exp(a) * exp(h) and a is already
    # the standard part.  The scalar spelling let the vector zero (0, 0)
    # through, so exp(0.4 + h*ln(1/h)) came back as e**0.8 -- the standard
    # part multiplied in twice, the same double-count that made sin(5)
    # return sin(10).
    non_zero = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}

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


def _ln_vector(x, terms):
    """ln of a vector-dimension composite, at any depth.  None = not ours.

    A dimension (e0, e1, ...) means c * (1/h)**e0 * B1**e1 * B2**e2 * ..., so

        ln(c * prod B_k**e_k) = ln(c) + sum_k e_k * ln(B_k)
                              = ln(c) + sum_k e_k * B_{k+1}

    because ln(1/h) IS B1 and ln(B_k) IS B_{k+1}.  Every exponent moves ONE
    index up and becomes a coefficient.  That single rule covers every depth:
    ln(h) lands on the log axis, ln(ln(1/h)) on the loglog axis, and the basis
    grows to hold it.  Without this branch ln simply compared a tuple against
    0 and raised TypeError, which then surfaced through limit() as
    "not composable with composite arithmetic" -- pointing at the wrong thing
    entirely, since the function composed fine and the basis was the problem.
    """
    from composite.backends.vector_dim_backend import as_vec, canon, ensure_depth
    dims, vals = x._backend.to_arrays(x._data)
    # CANONICAL length, not the padded one: as_vec pads to the current WIDTH,
    # so measuring it and asking for one more grew the basis on EVERY ln call,
    # ratcheting to 19 components over a handful of limits.  The canonical form
    # is what the term actually needs.
    items = [(canon(as_vec(d)), float(v)) for d, v in zip(dims, vals) if v != 0.0]
    if not items:
        return None
    lead_d, lead_c = max(items, key=lambda t: t[0])
    if all(e == 0 for e in lead_d):
        return None                      # a plain real: the scalar path owns it
    if lead_c <= 0.0:
        raise ValueError(
            f"ln of a composite whose leading coefficient is {lead_c}: "
            "the logarithm needs a positive leading term.")
    w = ensure_depth(len(lead_d) + 1)
    out = {}
    lc = math.log(lead_c)
    if lc != 0.0:
        out[tuple(0 for _ in range(w))] = lc
    for k, e in enumerate(lead_d):
        if e == 0:
            continue
        u = _vec_unit(k + 1, w)
        out[u] = out.get(u, 0.0) + e
    lead = _vec_composite(out)
    if len(items) == 1:
        return lead
    # x = lead_term * (1 + r); ln(x) = ln(lead_term) + ln(1 + r)
    ratio = x / _vec_composite({lead_d: lead_c})
    if _is_unit(ratio):
        return lead
    return lead + ln(ratio, terms)


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

    if LOG_SCALE and (getattr(x._backend, "VECTOR_DIMS", False)
                      or _carries_vector(x)):
        _v = _ln_vector(x, terms)
        if _v is not None:
            return _v

    coeffs = x.c
    if LOG_SCALE:
        # ln(|c|_d) = ln(c) + d*ln(h) holds for ANY non-zero d, so an
        # INFINITY is the same rule with the sign the other way round:
        # ln(1/h) = -ln(h) = +L.
        #
        # BEFORE x.st().  This used to sit inside the `a <= 0` branch,
        # which meant reading a standard part first -- and an unbounded
        # argument has none, so ln(1/h) raised on the way to the code that
        # already knew the answer.  The grade decides here; nothing about
        # this case needs a standard part.
        _pos = {d: c for d, c in coeffs.items()
                if _dim_positive(d) and c != 0.0}
        if _pos:
            from composite.backends.vector_dim_backend import dom_max as _dom_max
            _pd = _dom_max(_pos)
            _pc = _pos[_pd]
            if _pc > 0:
                _t = {(0, 1): float(_pd)}
                _lc = math.log(_pc)
                if _lc != 0.0:
                    _t[(0, 0)] = _lc
                _lead = _vec_composite(_t)
                _rest = x / _like(x, {_pd: _pc})
                if _is_unit(_rest):
                    return _lead
                return _lead + ln(_rest, terms)

    a = x.st()

    if a <= 0:
        # Check for positive infinitesimal: st=0 but positive coeff
        # at a negative dimension (e.g. ZERO = |1|_{-1})
        coeffs = x.c
        neg_dims = {d: c for d, c in coeffs.items() if _dim_negative(d)}
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
                    _rest = x / _like(x, {min_dim: coeff})
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
    result = _like(x, {}) if lead == 0.0 else _like(x, {_unit_dim(x): lead})
    power = _like(x, {_unit_dim(x): 1.0})

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
        # dom_max, not max: raw tuple order ranks a strict prefix lower, so a
        # canonical (0,0) lost to (0,0,-3) -- an INFINITESIMAL taken as the
        # leading term.  sqrt then read a 1e-32 cancellation coefficient and
        # refused the expression.
        from composite.backends.vector_dim_backend import dom_max as _dom_max
        _lead = _dom_max(_nz)
        if _dim_nonzero(_lead):
            _c = _nz[_lead]
            if _c < 0:
                raise ValueError(
                    f"sqrt of a negative leading coefficient |{_c}|_{_lead}")
            _root = _like(x, {_dim_scaled(_lead, 0.5): math.sqrt(_c)})
            _rest = x / _like(x, {_lead: _c})         # leading dim 0, st() == 1
            return _root * sqrt(_rest, terms)

    a = x.st()
    if a < 0:
        raise ValueError("sqrt requires non-negative standard part")

    if not _has_infinitesimal_part(x):
        return Composite({0: math.sqrt(a)})

    # SOLVE y**2 = x ORDER BY ORDER, where the dimensions allow it.
    #
    #     y_0 = sqrt(x_0),   2 y_0 y_n = x_n - sum_{k=1..n-1} y_k y_{n-k}
    #
    # Each order is fixed once from the orders below it.  The binomial series
    # kept below forms ratio**n instead, and for an argument with many terms
    # that is a far longer chain of roundings to reach the same coefficient.
    # Against sqrt(1+sin x) at x=0.4, relative error by order:
    #
    #     order         8         12        16        20
    #     binomial      2.4e-12   1.3e-07   2.3e-02   5.0e+03
    #     recurrence    5.6e-13   6.8e-09   3.1e-04   3.9e+01
    #     Newton        2.3e-12   2.9e-08   1.3e-03   1.7e+02
    #
    # None of them is exact past order 16, and that is not the algorithm: the
    # SAME recurrence at 60 decimal digits is exact to the last digit at every
    # order.  The coefficients fall eighteen orders of magnitude between n=8
    # and n=20 while the input is O(1), so each order costs about a digit and
    # float64's sixteen run out near order 16.  The recurrence only spends
    # them more slowly.
    _scalar_orders = None
    if all(not isinstance(d, tuple) for d in x.coeffs_dict()):
        _ks = [-float(d) for d, c in x.coeffs_dict().items() if c != 0.0]
        if all(k >= 0 and k == int(k) for k in _ks):
            _scalar_orders = {int(-float(d)): c
                              for d, c in x.coeffs_dict().items()}
    if _scalar_orders is not None:
        y = {0: math.sqrt(a)}
        for n in range(1, terms):
            acc = 0.0
            for k in range(1, n):
                if k in y and (n - k) in y:
                    acc += y[k] * y[n - k]
            v = (_scalar_orders.get(n, 0.0) - acc) / (2.0 * y[0])
            if v != 0.0 or n in _scalar_orders:
                y[n] = v
        out = _like(x, {-float(n): c for n, c in y.items()})
        return _truncate_order(out, _tighter(terms - 1, _min_complete(x)))

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

    result = _like(x, {_unit_dim(x): sqrt_a})
    power = _like(x, {_unit_dim(x): 1.0})

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

def _spans_multiple_axes(x):
    """True when x's non-standard terms live on more than one basis axis.

    The axis a dimension belongs to is the index of its first non-zero
    component: (-1, 0) is the power axis, (0, -1) the first log axis.  A value
    touching both is the case lexicographic long division cannot order.
    """
    axes = set()
    for d in x.c:
        if not isinstance(d, tuple):
            if d != 0:
                axes.add(0)
            continue
        # EVERY non-zero component, not just the leading one.  Breaking at the
        # first one filed a dimension like (-1, -1) -- which touches the power
        # axis AND a log axis -- under the power axis alone, so a value that
        # genuinely spans both reported False and division fell through to
        # lexicographic long division, the one case the docstring above says it
        # cannot order.
        #
        # Measured on exp(-t)/(1 + eps*t) with t carrying a quadrature seed:
        # the long-division route returned 4 power grades (complete to order 3)
        # where the reciprocal route returns 10 (order 9), for the same
        # expression written as exp(-t)*(1/(1+eps*t)).  That is what capped the
        # Euler-Stieltjes derivation at four coefficients.
        for i, e in enumerate(d):
            if e != 0:
                axes.add(i)
        if len(axes) > 1:
            return True
    return len(axes) > 1


def _lane_d1(x, axis=1):
    """First derivative with respect to the variable on `axis`.

    Composite.d(1) reads the POWER axis, which is correct when the variable is
    seeded there.  Once the integrator moved its seed to a lane, every reader
    of that derivative has to follow -- the curve tangent did not, so every
    line integral came back exactly 0.0: r'(t) read off an axis the parameter
    no longer occupies.
    """
    from composite.backends.vector_dim_backend import as_vec
    if not isinstance(x, Composite):
        return 0.0
    total = 0.0
    for d, c in x.c.items():
        v = as_vec(d)
        if len(v) > axis and v[axis] == -1 and v[0] == 0 and \
                all(e == 0 for i, e in enumerate(v) if i not in (0, axis)):
            total += c
    return total


def _reciprocal(x, terms=15):
    """Compute 1/x via geometric series. Internal helper."""
    a = x.st()
    if a == 0.0:
        # 7.1: exactly zero, not "small".  A tolerance here refused to invert
        # perfectly good composites whose standard part happened to be tiny.
        raise ZeroDivisionError("Cannot compute 1/x at x=0")
    h_part = x - R(a)
    ratio = h_part / R(-a)
    result = _like(x, {_unit_dim(x): 1/a})
    power = _like(x, {_unit_dim(x): 1.0})
    for n in range(1, terms):
        power = power * ratio
        result = result + power / a
    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))

def _maclaurin_odd(x, coeffs, terms):
    """sum coeffs[n] * x**(2n+1) for a composite x with ZERO standard part.

    asin and atan normally go the derivative route: form 1/sqrt(1-u**2) or
    1/(1+u**2), then antidifferentiate w.r.t. eps.  That works on the power
    axis, where the shift is one dimension and the divisor is the order.  It
    does NOT generalise across log axes -- integral of h**k * B1**m dh is not
    a single term unless k == -1, so there is no shift to apply.

    When the standard part is zero the Maclaurin series can just be composed
    instead, which needs no derivative, no antiderivative and no chain rule,
    and works on any axis at any depth.  Powers of x are formed once and
    reused.
    """
    x2 = x * x
    term = x                       # x**1
    out = None
    for n, c in enumerate(coeffs):
        if c != 0.0:
            piece = term * c
            out = piece if out is None else out + piece
        term = term * x2
    if out is None:
        return _like(x, {})
    # RECORD THE TRUNCATION.  Summing T odd powers omits c_T * x**(2T+1), whose
    # lowest order is m*(2T+1) with m the lowest order in x -- so every order
    # below that is complete and nothing above it is.  Leaving this out let
    # asin(h) and atan(h) report _complete = None, i.e. EXACT, while being a
    # 15-term truncation reaching order 29: the same claim-more-than-you-have
    # failure the completeness bookkeeping exists to stop, reintroduced by a
    # second code path that skipped it.
    nz = _infinitesimal_terms(x)
    bound = None
    if nz:
        m = min(_lead_order(d) for d in nz)
        if m > 0:
            bound = m * (2 * len(coeffs) + 1) - 1
    return _truncate_order(out, _tighter(bound, _min_complete(x)))


def atan(x, terms=15):
    """Arctangent for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.atan, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.atan(a)})
    if a == 0.0:
        # atan(z) = z - z**3/3 + z**5/5 - ...
        return _maclaurin_odd(
            x, [(-1.0) ** n / (2 * n + 1) for n in range(terms)], terms)
    # Built AFTER both early returns.  Standing above them, this was computed
    # and then discarded on every bare standard part -- and for a vector-keyed
    # one it did not merely waste the work: R(1) + x*x is wholly zero at no
    # point, but _reciprocal divides by R(-a), and dividing sends both operands
    # through R1, whose `dims[0] - 1` is a scalar spelling.  atan was the only
    # transcendental still raising TypeError on a log-axis argument, and this
    # ordering was the whole reason.
    one_plus_x2 = R(1) + x * x
    deriv = _reciprocal(one_plus_x2, terms)
    _lead = math.atan(a)
    result = {} if _lead == 0.0 else {_zero_dim_like(x): _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = _dim_shift(dim, -1)
        if _dim_order(new_dim) != 0:
            result[new_dim] = coeff / abs(_dim_order(new_dim))
    out = _like(x, result)
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
        # TRUNCATE to what is complete, as sqrt does.  Recording the bound and
        # then handing back the orders past it means the caller reads a
        # coefficient that is a partial sum wearing the shape of a finished
        # one: at the default depth atan's order 16 moved by 5.0 once the
        # series could actually be deepened.
        out = _truncate_order(out, out._complete)
    return out

def asin(x, terms=15):
    """Arcsine for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
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
    if a == 0.0:
        # asin(z) = sum C(2n,n) / (4**n (2n+1)) * z**(2n+1)
        _co = []
        for n in range(terms):
            _co.append(math.comb(2 * n, n) / (4.0 ** n * (2 * n + 1)))
        return _maclaurin_odd(x, _co, terms)
    _lead = math.asin(a)
    result = {} if _lead == 0.0 else {_zero_dim_like(x): _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = _dim_shift(dim, -1)
        if _dim_order(new_dim) != 0:
            result[new_dim] = coeff / abs(_dim_order(new_dim))
    out = _like(x, result)
    # Same order shift as antiderivative, and the same reason to record it:
    # this dict is built directly, so nothing else would.  Read the bound from
    # deriv AFTER the chain-rule multiply, which has already taken the min of
    # 1/sqrt(1-u^2) and u'.
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
        # As in atan: truncate to what is complete rather than hand back
        # partial sums past the bound.  acos is asin, so it follows.
        out = _truncate_order(out, out._complete)
    return out

def acos(x, terms=15):
    """Arccosine for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
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

def _is_clean_dyadic(s, max_k=40):
    """True when s is m/2**k for a small k -- a number the index set can hold.

    Every float64 IS a dyadic rational, so "is it dyadic" is trivially yes and
    useless as a test.  What matters is whether the value the CALLER meant is
    dyadic: 0.5, 0.25, 3/8 are; 1/3 and 1/7 are not, and float64 merely holds
    their nearest neighbour.  Scaling by 2**k and asking for an integer
    separates the two.
    """
    v = float(s)
    if v != v or v in (float("inf"), float("-inf")):
        return False
    for k in range(max_k + 1):
        if (v * (1 << k)).is_integer():
            return True
    return False


def _warn_non_dyadic_exponent(x, s):
    """Audible when a root asks for an index the set cannot hold exactly.

    WHEN THIS DOES NOT APPLY, which is most of the time.  If x has a non-zero
    standard part then ln(x) is an ordinary series with INTEGER dimensions, so
    the exponent only scales coefficients and never reaches the index at all:
    power(_seeded(8), 1/3) comes back with dims -14..0 and values correct to
    2.2e-16.  Nothing is approximate about its structure.  An earlier version
    of this guard refused that case, which was simply wrong.

    WHEN IT DOES.  If x is a pure infinitesimal or infinity, ln(x) sits on the
    log axis and exp moves the exponent onto the POWER axis, where it becomes
    the index.  Dimensions are exact under {+, -, /2} -- the dyadics -- so 1/2,
    1/4, 3/8 land exactly and 1/3, 1/7 do not.

    WHY A WARNING AND NOT A REFUSAL.  The index is still right to sixteen
    digits, which is no worse than any other floating-point result, and asking
    for h**(1/3) is a reasonable thing to do.  What is NOT ordinary rounding is
    the shape of the failure: h**(1/7) raised to the seventh lands at
    -0.9999999999999998, a SEPARATE term from the -1 it should have merged
    with, so coeff(-1) returns the wrong number rather than a slightly wrong
    one.  That deserves to be heard, not blocked.  (h**(1/3) closes only
    because rounding happens to land favourably -- luck, not a guarantee.)
    """
    if _is_clean_dyadic(s):
        return
    if x.st() != 0.0:
        return                      # exponent stays in the coefficients
    if not any(_dim_nonzero(d) for d in x.c):
        return                      # no dimension at all
    # _dim_order reads only the POWER component, so a pure log-axis term like
    # (0, 1) looked dimensionless and the warning never fired for
    # power(ln(1/h), 1/3) -- which lands a fractional index on the LOG axis,
    # exactly the same failure one axis over.
    import warnings
    warnings.warn(
        f"power(x, {s!r}) on a composite with no standard part: the exponent "
        f"becomes the DIMENSION, and dimensions are exact only under "
        f"{{+, -, /2}} (the dyadics). {s!r} is not m/2**k, so the index is a "
        f"float approximation -- accurate to ~1e-16, but it will not always "
        f"recombine: h**(1/7) to the 7th lands at -0.9999999999999998, a "
        f"separate term from -1, so coeff(-1) then reads the wrong value. "
        f"sqrt and repeated halving are exact and always will be.",
        stacklevel=3)


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
    _warn_non_dyadic_exponent(x, s)
    return exp(R(s) * ln(x, terms), terms)


# =============================================================================
# HIGH-LEVEL API: AUTOMATIC TRANSLATION
# =============================================================================

def _d_deps(x):
    """d(x)/d(eps): the derivative of a composite w.r.t. its own infinitesimal.

    THE POWER AXIS.  A term c*eps**k sits at dim -k and differentiates to
    k*c*eps**(k-1) at dim -(k-1).  In (e0, e1, ...) form, where a dim means
    c * (1/h)**e0 * B1**e1 * B2**e2 ... , that is e0 += 1 with the coefficient
    scaled by -e0.

    THE LOG AXES.  These are not constants -- they are functions of h, so the
    chain rule reaches across scales:

        B1 = ln(1/h)        dB1/dh = -1/h
        B2 = ln(B1)         dB2/dh = -(1/h) * B1**-1
        B_k                 dB_k/dh = -(1/h) * B1**-1 ... B_{k-1}**-1

    so differentiating B_k**e_k gives  -e_k * (1/h) * B1**-1 ... B_{k-1}**-1
    * B_k**(e_k - 1): e0 += 1, every axis from 1 to k drops by one, and the
    coefficient is scaled by -e_k.  ONE TERM PER AXIS THE DIMENSION TOUCHES,
    summed.

    Returning only the power-axis part -- which is what this did before -- made
    atan(1/ln(1/h)) come back as NOTHING, since a pure log-axis term has no
    power component at all and the whole factor vanished.

    THE CHAIN RULE FACTOR is what asin and atan need: they form 1/sqrt(1-u**2)
    (resp. 1/(1+u**2)) and antidifferentiate w.r.t. eps, but
    d/deps asin(u(eps)) = u' / sqrt(1-u**2).
    """
    # Differentiating LOSES an order: the top coefficient of x produces the top
    # of x', and there is nothing above it to produce the next.  Not recording
    # that made asin(sin x) claim order 12 on 11 sound ones.
    _c = getattr(x, "_complete", None)

    def _tag(out):
        if _c is not None:
            out._complete = _c - 1
        return out

    if not any(isinstance(d, tuple) for d in x.c):
        # Scalar fast path: keeps scalar work off the vector backend entirely.
        return _tag(Composite({_dim_shift(d, 1): c * abs(_dim_order(d))
                               for d, c in x.c.items()
                               if _dim_negative(d) and c != 0.0}))

    from composite.backends.vector_dim_backend import as_vec, canon, ensure_depth
    out = {}
    for d, c in x.c.items():
        if c == 0.0:
            continue
        v = list(as_vec(d))
        for k, ek in enumerate(v):
            if ek == 0:
                continue
            nd = list(v)
            nd[0] += 1                      # every axis contributes a 1/h
            for j in range(1, k + 1):       # ...and drops axes 1..k by one
                nd[j] -= 1
            key = canon(tuple(nd))
            out[key] = out.get(key, 0.0) + c * (-ek)
    out = {k: val for k, val in out.items() if val != 0.0}
    if not out:
        return _tag(Composite({}))
    ensure_depth(max(len(k) if isinstance(k, tuple) else 2 for k in out))
    return _tag(_vec_composite(out))


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
    poles = {d: c for d, c in result.coeffs_dict().items()
             if _dim_positive(d) and c != 0.0}
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


def _limit_probe(f, as_x_to, dir, terms, _is_inf, why):
    """Probe at real points when the algebra could not answer.

    Used for two cases that look different and behave the same: a result of
    NOTHING, and a sub-expression with no composite at all (sin(1/h)).  In
    both the expression as a WHOLE may still converge -- x*sin(1/x) does --
    so the limit is not refused until probing fails.

    It never claims a limit on one sample.  At infinity two windows must
    AGREE; at a finite point the extrapolation has to succeed.  That is what
    keeps sin(x) at infinity from being reported as 0 by an integral average
    that happens to cancel.
    """
    if _is_inf:
        v1 = _limit_at_inf_fallback(f, as_x_to, n=500, width=50.0)
        v2 = _limit_at_inf_fallback(f, as_x_to, n=500, width=100.0)
        if math.isfinite(v1) and math.isfinite(v2):
            if abs(v1) < 1e-4 and abs(v2) < 1e-4:
                return 0.0
            if abs(v1 - v2) < 1e-3 * (abs(v1) + abs(v2)):
                return v2
    else:
        extrap = _limit_extrapolate(f, as_x_to, dir, terms)
        if extrap is not None:
            return extrap
    raise LimitDoesNotExistError(why)


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
    except NotRepresentableError:
        # DO NOT RECOVER.  A sub-expression has no composite -- sin(1/h) takes
        # every value in [-1, 1] and a composite holds one value at one grade.
        #
        # Probing was tried and is the wrong answer even when it looks right.
        # It got x*sin(1/x) -> 0, which is the true limit, but only by
        # sampling: the bound |sin| <= 1 is exactly the information the raise
        # declined to carry, so the algebra cannot reach that limit and the
        # probe is guessing from points.  On x/sin(1/x) the same probe
        # returned 0.0 for a function that is UNBOUNDED -- sin(1/x) passes
        # through zero at x = 1/(k*pi), where the quotient blows up -- and a
        # single-resolution probe cannot see it: its maximum reads 1.006 at
        # 2,000 samples and 20.97 at 32,000.  A method that answers 0.0 for
        # both a convergent and a divergent case is not a fallback, it is a
        # coin toss with a confident face.
        #
        # So the refusal propagates.  The limit is not computed rather than
        # computed by other means.
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

    # Positive dims → unbounded divergence (from exp, ln, etc.)
    max_pos = result.max_positive_dim()
    if max_pos is not None:
        pos_coeffs = {d: c for d, c in result.coeffs_dict().items()
                      if _dim_positive(d)}
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

    # Nothing (∅) means an indeterminate value was involved. Check if the
    # overall expression still converges by probing at real points.
    if _is_nothing(result):
        return _limit_probe(
            f, as_x_to, dir, terms, _is_inf,
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
    # The constant's key has to be the same KIND as the dimensions it will sit
    # beside: sorted() cannot order a tuple against an int, so a scalar 0 mixed
    # with vector dims raised "'<' not supported between tuple and int" the
    # moment erf integrated a log-axis argument.
    _dims = list(f_composite.c)
    _zero = 0
    for _d in _dims:
        if isinstance(_d, tuple):
            _zero = tuple(0 for _ in range(len(_d)))
            break
    result = {_zero: constant}
    for dim, coeff in f_composite.c.items():
        if not _dim_positive(dim):
            new_dim = _dim_shift(dim, -1)
            divisor = abs(_dim_order(new_dim))
            result[new_dim] = coeff / divisor
    # Every order moves up by one, so a f sound to K integrates to one sound to
    # K+1.  Building the dict directly skips _truncate_order, which is where
    # the bound would otherwise be recorded -- dropping it here made asin, atan
    # and the derivative round trip all claim to be exact.
    _c = getattr(f_composite, "_complete", None)
    out = _like(f_composite, result)
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


def _perturbation_seed(axis):
    """Unit infinitesimal on basis axis `axis` (1 = first non-power axis).

    The basis axes past 0 mean iterated logarithms, and this borrows them as
    independent perturbation directions.  That is sound HERE because the only
    way a log-axis dimension enters is ln/exp of an infinitesimal or infinite
    value, and a panel midpoint is neither -- ln(_seeded(t)) expands purely on
    the power axis for every finite non-zero t.  _box_exact probes for that
    and declines the exact path if the integrand brings its own log axes.
    """
    from composite.backends.vector_dim_backend import canon, ensure_depth
    ensure_depth(axis + 1)
    return _vec_composite({canon((0,) * axis + (-1,)): 1.0})


def _derivative_axis(x, axis):
    """d/d(variable on `axis`), staying a composite in every other lane.

    The inverse shift to _antiderivative_axis.  A term at lane order k becomes
    k times the term at order k-1, so d(sigma)/du comes back as a SERIES in
    (u, v) rather than a number at one point -- which is what a surface
    integral needs: the area element varies across the patch, and reading the
    tangent off as a float freezes it at the midpoint.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in x.c.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        k = v[axis]
        if k >= 0:
            continue                      # no content on this lane: d/d(var)=0
        v[axis] = k + 1
        key = canon(tuple(v))
        out[key] = out.get(key, 0.0) + coeff * abs(k)
    return _like(x, out)


def _integrand_needs_lane(f, points):
    """Does the integrand put its OWN content on the power axis?

    The variable of integration needs a lane of its own only when the power
    axis is already occupied.  Seeded on lane 1 the two are separable: the
    variable's contribution sits on axis 1, so any non-zero POWER component
    belongs to the integrand -- an R1 zero, or anything the caller's
    expression genuinely made infinitesimal.  `x*0 + 1` shows
    {(-1,-1): 1.0, (-1,0): 0.5}; x*x, exp(-x), sin(x), 1/(1+x) show nothing.

    When nothing is there the variable rides the power axis and the whole
    integral stays on the SCALAR backend.  That matters: seeded on a lane,
    every panel is a vector-dimension composite, which is excluded from
    SparseDenseBackend by design -- 83% of the backend operations in
    integral of exp(-x) over [0,inf), and 61% in a lateral Borel sum.

    Sampled, not proved.  An integrand that develops power-axis content only
    away from these points would be misclassified; the points are spread
    across the interval to make that unlikely, and the lane is taken whenever
    the probe cannot evaluate at all.
    """
    if not LANE_AUTO:
        return True                          # see LANE_AUTO
    for pt in points:
        try:
            fx = _ensure_composite(f(R(pt) + _perturbation_seed(1)))
        except Exception:
            return True                      # cannot tell: take the safe path
        for d, v in fx.c.items():
            if v == 0.0:
                continue
            power = d[0] if isinstance(d, tuple) else d
            if power != 0:
                return True
    return False


# OFF BY DEFAULT.  Putting the variable on the power axis when the integrand
# leaves it free is sound for R1 zeros and measurably faster -- but the lane
# turned out to do a second job nobody had written down: it separates the
# variable's derivatives from the integrand's, and the adaptive error estimate
# reads them separately.  With the variable on the power axis,
#
#     integral of sqrt(x) over [0,1]   lane 0: 2.2e-04     lane 1: 2.2e-13
#
# -- nine orders, on an integrand whose power axis the probe correctly reports
# as free.  The probe detects the R1 condition and cannot see this one, and
# three suites fail with it on.  So it is opt-in, for a caller that knows its
# integrand is plain.  Measured gain when it applies: improper integral 0.68x,
# a lateral Borel sum 1.11s -> 0.77s.
LANE_AUTO = True

LANE_PROBE_POINTS = 1      # raise it for an integrand whose structure varies


def _lane_probe_points(a, b, n=None):
    """Points inside [a, b] to ask _integrand_needs_lane about.

    ONE by default, and the count is not free: each probe evaluates the whole
    integrand with a lane seed, which is exactly the slow path the probe
    exists to avoid.  Measured on integral of exp(-x) over [0,inf) and on a
    lateral Borel sum:

        probes    improper integral    resum_median
          1          0.0130s              0.768s
          2          0.0144s              0.985s
          3          0.0157s              1.225s
          5          0.0198s              1.692s

    One is enough for the case the lane exists for, because an R1 zero is a
    property of the EXPRESSION rather than of the point: x*0+1, x-x+1,
    (x-0.5)*0+1 and sin(x)-sin(x)+1 are all caught by a single probe, and all
    integrate to 1.000000000000000 exactly.  Raise LANE_PROBE_POINTS for an
    integrand whose structure genuinely varies across the interval.
    """
    if n is None:
        n = LANE_PROBE_POINTS
    lo = a if math.isfinite(a) else (b - 10.0 if math.isfinite(b) else -1.0)
    hi = b if math.isfinite(b) else (lo + 10.0)
    if hi == lo:
        return [lo]
    return [lo + (hi - lo) * t for t in
            [(i + 0.5) / n for i in range(n)]]


def _antiderivative_axis(x, axis):
    """Antiderivative with respect to the variable living on `axis`.

    antiderivative() shifts the POWER axis, which is right when the integration
    variable is seeded there.  It is not right once the seed has its own lane:
    the power axis then carries the INTEGRAND's own infinitesimal content --
    R1 zeros, anything the caller's expression genuinely made small -- and
    shifting that is integrating something that is not the variable.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in x.c.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        if v[axis] > 0:
            continue                       # positive: an INF component, skip
        v[axis] -= 1
        divisor = abs(v[axis])
        out[canon(tuple(v))] = coeff / divisor
    return out


def _eval_axis(terms, h_value, axis):
    """Substitute a real h for the axis-`axis` infinitesimal, keep every other.

    This is the only place a real number may replace an infinitesimal, and it
    may do so ONLY for the lane the integrator seeded.  Doing it for every
    dimension at once is what turned a structural zero into a real 4.967e-09:
    x*0 leaves h*z and z*z both at dim -2 when the seed shares the power axis,
    and no substitution can separate them afterwards.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in terms.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        k = v[axis]
        if k >= 0:
            continue
        v[axis] = 0
        out_key = canon(tuple(v))
        out[out_key] = out.get(out_key, 0.0) + coeff * h_value ** (-k)
    return out


@_contextlib.contextmanager
def _box_scope(nvars):
    """Raise the dimension cap for one box integral.

    The perturbation seeds put a series on each extra axis, so the integrand's
    expansion is a PRODUCT across nvars axes and easily passes MAX_ACTIVE_DIMS
    = 60: 1/(1+x+y+z) holds 680 terms at a single panel.  _truncate_dims then
    keeps the 60 nearest zero, which stops the panel refinement converging --
    so the loop doubles all the way to the cap and still lands on a worse
    answer than the Riemann sum it replaced.  Measured on that integrand:
    cap 60 -> err 9.5e-06 in 34.2s;  cap lifted -> err 7.9e-09 in 0.7s.
    Slower AND wronger, from a guard written for a different purpose.
    """
    global MAX_ACTIVE_DIMS
    old = MAX_ACTIVE_DIMS
    MAX_ACTIVE_DIMS = max(MAX_ACTIVE_DIMS, 400 * max(1, nvars - 1))
    try:
        yield
    finally:
        MAX_ACTIVE_DIMS = old


def _surface_exact(f, uv, surface, is_vector, tol=1e-10):
    """Surface integral with u and v on their own lanes.

    Builds the integrand as a COMPOSITE in (u, v) -- position, both tangents,
    the cross product and its norm -- and hands it to the box integrator.  The
    sampling path froze the tangents at each sample point with float(), so the
    area element was piecewise constant; here it varies across the patch
    because d(sigma)/du is still a series.
    """
    (a_u, b_u), (a_v, b_v) = uv

    def integrand(u_c, v_c):
        S = surface(u_c, v_c)
        if not isinstance(S, (list, tuple)) or len(S) < 3:
            raise TypeError("surface must return three components")
        S = [_ensure_composite(c) for c in S]
        Su = [_derivative_axis(c, 1) for c in S]
        Sv = [_derivative_axis(c, 2) for c in S]
        nx = Su[1] * Sv[2] - Su[2] * Sv[1]
        ny = Su[2] * Sv[0] - Su[0] * Sv[2]
        nz = Su[0] * Sv[1] - Su[1] * Sv[0]
        if is_vector:
            F = [_ensure_composite(comp(*S)) for comp in f]
            return F[0] * nx + F[1] * ny + F[2] * nz
        val = _ensure_composite(f(*S))
        return val * sqrt(nx * nx + ny * ny + nz * nz)

    # DOES THE SURFACE PROPAGATE COMPOSITE STRUCTURE?
    #
    # A surface written with math.cos rather than the composite cos does NOT
    # raise: Composite defines __float__, so math.cos silently takes the
    # standard part and hands back a plain float.  Every component then has no
    # lane content, both tangents come out empty, the cross product is zero and
    # the integral is 0.0 -- with no exception anywhere to trigger a fallback.
    # That is how five passing tests turned into exact zeros.  Check for it.
    try:
        u_p = 0.5 * (a_u + b_u) + _perturbation_seed(1)
        v_p = 0.5 * (a_v + b_v) + _perturbation_seed(2)
        S_p = surface(u_p, v_p)
        if not isinstance(S_p, (list, tuple)) or len(S_p) < 3:
            return None
        # A CONSTANT component is legitimate -- [u, v, 0] is a flat patch, and
        # requiring every component to be a Composite sent it to the fallback
        # (5.0e-10 in 182ms instead of 0.0e+00 in 5ms).  What matters is not
        # that each part is composite but that the patch MOVES on both
        # parameters, which is what the two checks below test.
        S_p = [_ensure_composite(c) for c in S_p]
        moves_u = any(_derivative_axis(c, 1).c for c in S_p)
        moves_v = any(_derivative_axis(c, 2).c for c in S_p)
        if not (moves_u and moves_v):
            return None
    except Exception:
        return None

    try:
        return _box_exact(integrand, [(a_u, b_u), (a_v, b_v)], tol=tol,
                          probe_floats=False)
    except Exception:
        return None


def _box_exact(f, ranges, tol=1e-10, max_panels=4096, probe_floats=True):
    """Exact box integral.  The INTEGRAND keeps the power axis; every
    integration variable gets its own lane.

    Variable i is seeded on basis axis i+1.  Nothing the caller's expression
    produces on the power axis is ever touched: it is not the variable, so it
    is not integrated, and it is never handed to a real panel width.

    Returns None when the exact path does not apply, so the caller falls back
    rather than returning something wrong.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    nvars = len(ranges)
    if nvars < 2:
        return None

    # The integrand must not bring content on the lanes the seeds will use.
    probe_pt = [0.5 * (lo + hi) for lo, hi in ranges]
    if not probe_floats:
        probe = None            # caller builds its integrand from composites
    else:
      try:
        probe = f(*probe_pt)
      except Exception:
        return None
    if isinstance(probe, Composite):
        if any(isinstance(d, tuple) and any(e != 0 for e in d[1:nvars + 1])
               for d in probe.c):
            return None

    centres = [0.5 * (lo + hi) for lo, hi in ranges]
    a, b = ranges[0]
    prev = None
    last_delta = None
    stalled = 0
    panels = 8
    with _box_scope(nvars):
        # seeds[i] rides on lane i+1; lane 0 (power) stays the integrand's
        seeds_tail = [centres[i] + _perturbation_seed(i + 1)
                      for i in range(1, nvars)]
        while panels <= max_panels:
            acc = {}
            dx = (b - a) / panels
            try:
                for i in range(panels):
                    x_seed = (a + i * dx + dx / 2) + _perturbation_seed(1)
                    fx = _ensure_composite(f(x_seed, *seeds_tail))
                    Fx = _antiderivative_axis(fx, 1)
                    for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
                        for k, v in _eval_axis(Fx, hv, 1).items():
                            acc[k] = acc.get(k, 0.0) + sign * v
            except Exception:
                return None

            # integrate the remaining lanes term by term, exactly
            total = 0.0
            for key, coeff in acc.items():
                v = list(as_vec(key))
                while len(v) <= nvars:
                    v.append(0)
                if v[0] != 0:
                    continue          # the integrand's OWN infinitesimal part:
                                      # metadata, never fused into the float
                factor = coeff
                for var_i in range(1, nvars):
                    order = -int(v[var_i + 1])
                    lo, hi = ranges[var_i]
                    c = centres[var_i]
                    factor *= ((hi - c) ** (order + 1)
                               - (lo - c) ** (order + 1)) / (order + 1)
                total += factor

            if not math.isfinite(total):
                return None
            if prev is not None:
                delta = abs(total - prev)
                if delta <= tol * max(1.0, abs(total)):
                    return total
                if last_delta is not None and delta > 0.5 * last_delta:
                    stalled += 1
                    if stalled >= 2:
                        return total
                else:
                    stalled = 0
                last_delta = delta
            prev = total
            panels *= 2
        return prev


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
                tangent = [_lane_d1(p) for p in pos_comp]

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
        result, err = integrate_adaptive(_line_integrand, a_t, b_t, tol=tol,
                                         terms=terms, lane=1)
        return result.st()

    # --- SURFACE INTEGRAL ---
    if surface is not None:
        uv = args[0] if args else ((0, 1), (0, 1))
        (a_u, b_u), (a_v, b_v) = uv
        is_vector = isinstance(f, list)

        _exact = _surface_exact(f, uv, surface, is_vector, tol=tol)
        if _exact is not None:
            return _exact

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
        exact = _box_exact(f, list(args), tol=tol)
        if exact is not None:
            return exact
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
        exact = _box_exact(f, list(args), tol=tol)
        if exact is not None:
            return exact
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
        neg_terms = {d: c for d, c in Fx.c.items() if _dim_negative(d)}
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

def integrate_adaptive(f, a, b, tol=1e-10, terms=15, max_depth=20, min_panels=4,
                       lane=None):
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
    # THE VARIABLE ONLY NEEDS A LANE IF THE POWER AXIS IS TAKEN.  See
    # _integrand_needs_lane: when it is free the variable rides it and every
    # panel stays a scalar-dimension composite on the fast backend.
    # `lane` PINS the axis for a caller whose integrand is coupled to one.
    # The curve and surface paths are: _line_integrand reads the tangent with
    # _lane_d1, which looks at lane 1, so seeding anywhere else makes every
    # tangent zero and every line integral come back exactly 0.0.  A probe
    # cannot see that -- an integrand that READS a lane looks identical to one
    # that ignores it.
    _lane = lane if lane is not None else (
        1 if _integrand_needs_lane(f, _lane_probe_points(a, b)) else 0)

    probe = _ensure_composite(f(_seeded((a + b) / 2)))
    if not any(_dim_negative(dim) for dim in probe.c):
        import warnings
        warnings.warn(
            "integrate_adaptive: f does not propagate composite structure to output. "
            "Falling back to midpoint rule (no adaptive refinement for this integrand).",
            stacklevel=2)

    def _panel_with_error(x, dx):
        """One composite eval at midpoint -> integral value + error estimate.

        The variable of integration rides on LANE 1, not the power axis.  The
        power axis belongs to the integrand: R1 zeros and anything else the
        caller's expression genuinely made infinitesimal live there, and they
        are not the variable, so they are neither integrated nor handed to a
        real panel width.  With the seed on the power axis they were, and
        integral of (x*0 + 1) over [0,1] came back 1.0052083333333333 -- a real
        number manufactured from a structural zero, off by 5e-3.
        """
        mid = x + dx / 2
        seed = _perturbation_seed(_lane) if _lane else ZERO
        fx = _ensure_composite(f(mid + seed))

        # Integral via antiderivative along the variable's own lane
        Fx_terms = _antiderivative_axis(fx, _lane)
        acc = {}
        for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
            for k, v in _eval_axis(Fx_terms, hv, _lane).items():
                acc[k] = acc.get(k, 0.0) + sign * v
        # KEEP EVERY POWER-AXIS GRADE, each at its own grade.  Terms carrying a
        # power-axis component are the integrand's own infinitesimal content --
        # they must never be summed INTO dimension 0, which is what the comment
        # here used to say, but they were then dropped entirely, which is a
        # different thing and loses the answer.
        #
        # It matters exactly where it is hardest to get otherwise.  With a
        # seeded parameter eps = h, the integrand of the Euler-Stieltjes
        # integral carries e^-t * (1, -t, t^2, -t^3, ...) across grades, and
        # integrating grade -n gives (-1)^n n! -- the asymptotic series, term
        # by term, without ever summing it (it diverges factorially for every
        # nonzero eps).  Dropping the grades returned 1.0: the correct LIMIT,
        # and a much smaller answer than the one asked for, with nothing to
        # say it had been narrowed.
        #
        # Grade 0 alone is what it always was, so an ordinary integral is
        # unchanged and still reads as a float through st().
        vals = {}
        for k, v in acc.items():
            kk = k if isinstance(k, tuple) else (k,)
            pk = kk[0] if kk else 0
            if len(kk) > 1 and any(c != 0 for c in kk[1:]):
                continue          # still carries the variable's lane: not integrated
            vals[pk] = vals.get(pk, 0.0) + v
        value = Composite(vals) if vals else Composite({0: 0.0})

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
        # Orders along the VARIABLE's lane.  _dim_order reads the power axis,
        # which now holds the integrand's own content -- reading it here made
        # every panel look like a constant.
        def _lane1(d):
            """Order along the VARIABLE's axis -- whichever one it is.

            This read index 1 unconditionally.  With the variable on the power
            axis every dim is a scalar, as_vec gives (d, 0), and the order came
            back 0 for every term -- so `max(orders) <= 2` fired on every panel
            and the estimate was 0.0.  The adaptive path then refined nothing:
            integral of exp(-x^2) over [0,20] returned 0.59 against 0.886, and
            sqrt(x) lost four digits.  Both looked like accuracy problems; they
            were the error estimate reading an axis the variable was not on.
            """
            from composite.backends.vector_dim_backend import as_vec
            v = as_vec(d) if isinstance(d, tuple) else (d,)
            return -(v[_lane] if len(v) > _lane else 0)

        orders = [_lane1(d) for d in fx.c if _lane1(d) >= 0]
        if not orders or max(orders) <= 2:
            # Nothing above a quadratic -- the antiderivative is EXACT.
            return value, 0.0

        cut = max(max(orders) - 2, 2)
        err_est = 0.0
        for dim, coeff in fx.c.items():
            k = _lane1(dim)
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

    def _scale(v):
        """Magnitude for the relative error test, ACROSS EVERY GRADE.

        abs(Composite) is the standard part, i.e. dimension 0 alone.  With a
        seeded parameter the integrand carries real content at other grades,
        and sizing panels by dimension 0 accepted them while those grades were
        still garbage -- the Euler-Stieltjes coefficient at grade -1 came back
        -0.0 where uniform panels gave -1 exactly.  The error estimate already
        sums over every dimension; only the scale it was compared against was
        one-dimensional.
        """
        if isinstance(v, Composite):
            cs = [abs(c) for c in v.coeffs_dict().values() if c != 0.0]
            return max(cs) if cs else 0.0
        return abs(v)

    def _panel_classic(x, dx):
        """Classic single-panel integral for fallback comparison."""
        mid = x + dx / 2
        seed = _perturbation_seed(_lane) if _lane else ZERO
        fx = _ensure_composite(f(mid + seed))
        Fx_terms = _antiderivative_axis(fx, _lane)
        acc = {}
        for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
            for k, v in _eval_axis(Fx_terms, hv, _lane).items():
                acc[k] = acc.get(k, 0.0) + sign * v
        # Same rule as _panel_with_error: every power-axis grade is kept at
        # its own grade.  This path dropped them, so any panel that fell back
        # here silently contributed only its dimension-0 part -- which is why
        # the Euler-Stieltjes grade -1 coefficient came back 5e-15 instead of
        # -1 while uniform panels gave -1 exactly.
        out = {}
        for k, v in acc.items():
            kk = k if isinstance(k, tuple) else (k,)
            if len(kk) > 1 and any(c != 0 for c in kk[1:]):
                continue
            pk = kk[0] if kk else 0
            out[pk] = out.get(pk, 0.0) + v
        return Composite(out) if out else Composite({0: 0.0})

    def _adaptive(a, b, depth):
        dx = b - a
        mid = (a + b) / 2

        value, err_est = _panel_with_error(a, dx)

        if err_est >= 0:
            # Primary path: Taylor convergence check
            if err_est < tol * (_scale(value) + 1e-100) or depth >= max_depth:
                return value, err_est
            if _scale(value) < tol * 0.01 and err_est < tol:
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
            error = _scale(half - value)
            if error < tol * (_scale(half) + 1e-100) or depth >= max_depth:
                return half, error
            if _scale(half) < tol * 0.01:
                return half, error

        # Bisect
        left_val, left_err = _adaptive(a, mid, depth + 1)
        right_val, right_err = _adaptive(mid, b, depth + 1)
        return left_val + right_val, left_err + right_err

    # Pre-subdivide into min_panels equal panels.
    #
    # NOTHING, not zero (R6).  `total_val = 0.0` looks harmless and is not: a
    # bare Python zero meeting a Composite converts to the composite zero
    # |1|_-1 by R1, so the accumulator injected +1 at grade -1 on its first
    # addition.  Dimension 0 stayed right, which is why it went unseen -- but
    # a panel value of -1 at grade -1 came out 0, and the Euler-Stieltjes
    # coefficient of eps^1 read 1.8e-19 instead of -1 while every neighbouring
    # grade was exact.  It was bit-identical at every refinement depth because
    # the injection is one constant, not an error that refines away.
    #
    # The library's own R1 warning names this case exactly.  It was firing.
    total_val = Composite({})
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

    # total_val already IS a composite when the integrand carried grades of its
    # own -- re-wrapping it as Composite({0: total_val}) collapsed every one of
    # them into dimension 0 after the panels had computed them correctly.
    if isinstance(total_val, Composite):
        return total_val, total_err
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
    def _as_traced(self, result):
        """Re-wrap an operation's result so tracing survives the next step.

        Was `_wrap`, which SHADOWED Composite._wrap -- a classmethod taking
        (data, backend, demote, complete) -- with an instance method taking
        (result).  Nothing outside this class broke only because every call
        site spells it `Composite._wrap(...)` on the class rather than on an
        instance.

        And it assigned `tc.c = result.c`.  `.c` became a read-only property
        computed from ._data when the backends landed, so this raised
        AttributeError on the first traced operation -- trace() printed one
        line and died.  Copy the three slots instead.
        """
        if isinstance(result, Composite):
            tc = TracedComposite.__new__(TracedComposite)
            tc._backend = result._backend
            tc._data = result._data
            tc._complete = result._complete
            return tc
        return result
    def __add__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__add__(other)
        print(f"    {self}  +  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __radd__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__radd__(other)
        print(f"    {other_disp}  +  {self}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __sub__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__sub__(other)
        print(f"    {self}  -  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __mul__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__mul__(other)
        print(f"    {self}  ×  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __rmul__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__rmul__(other)
        print(f"    {other_disp}  ×  {self}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __truediv__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__truediv__(other)
        print(f"    {self}  ÷  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __pow__(self, n):
        result = super().__pow__(n)
        print(f"    ({self})^{n}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)


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
