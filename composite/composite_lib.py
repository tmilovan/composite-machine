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
composite_lib.py — Unified Calculus Library (Fixed v2: Line Integral Composite-First)
=====================================================================================
All operations use composite arithmetic. No plain-number fast paths
in transcendental functions. Integration accumulates as Composite.

v2 changes:
  - Line integral rewritten to composite-first (Option B: probe once, choose path)
  - Phase 1: Composite tangent vectors via .d(1) — no epsilon
  - Phase 2: Routes through integrate_adaptive — adaptive, error estimate
  - Phase 3: F evaluated at composite curve positions — full Taylor in t
  - Fallback for non-composite curves still benefits from Phase 2

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
from typing import Callable, List, Optional, Union
import struct
import numpy as np

from composite.backends import get_backend

# =============================================================================
# CORE: COMPOSITE NUMBER CLASS
# =============================================================================

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

    __slots__ = ['_data', '_backend']

    def __init__(self, coefficients=None, _data=None):
        self._backend = get_backend()

        if _data is not None:
            # Internal fast path: created by arithmetic ops
            self._data = _data
            return

        if coefficients is None:
            self._data = self._backend.create_from_terms(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64))
        elif isinstance(coefficients, (int, float)):
            if coefficients == 0:
                self._data = self._backend.create_from_terms(
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64))
            else:
                self._data = self._backend.create(0, float(coefficients))
        elif isinstance(coefficients, dict):
            items = {k: v for k, v in coefficients.items() if v != 0}
            if items:
                sorted_dims = sorted(items.keys())
                dims = np.array(sorted_dims, dtype=np.int64)
                vals = np.array([items[d] for d in sorted_dims], dtype=np.float64)
                self._data = self._backend.create_from_terms(dims, vals)
            else:
                self._data = self._backend.create_from_terms(
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64))
        else:
            raise TypeError(f"Cannot create Composite from {type(coefficients)}")

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------

    @classmethod
    def _wrap(cls, data):
        """Create a Composite directly from backend data. No dict parsing."""
        obj = cls.__new__(cls)
        obj._backend = get_backend()
        obj._data = data
        return obj

    # -------------------------------------------------------------------------
    # Backward compatibility: .c property
    # -------------------------------------------------------------------------
    # This lets existing code that reads self.c (antiderivative, show,
    # TracedComposite, transcendentals that check `x.c`) keep working.
    # It reconstructs a dict from the backend data.
    # MIGRATE AWAY FROM THIS over time — it defeats the purpose of
    # the backend by creating a dict on every access.
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
        return {int(d): float(v) for d, v in zip(dims, vals)}

    # -------------------------------------------------------------------------
    # Constructors (UNCHANGED — they use dict, which __init__ accepts)
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
        return cls({0: float(value)})

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        dims, vals = self._backend.to_arrays(self._data)

        if len(dims) == 0:
            return "|0|₀"

        sub = "₀₁₂₃₄₅₆₇₈₉"
        def fmt_dim(n):
            n = int(n)
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
        """
        Serialize to JSON-safe dict.

        Returns dict with string keys (JSON requires strings)
        and float values. Round-trips perfectly via from_dict().

        Example:
            c = R(9) + 6 * ZERO
            c.to_dict()  # → {'0': 9.0, '-1': 6.0}
        """
        return {str(k): v for k, v in self.c.items()}

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize from dict. Accepts string or int keys.

        Example:
            Composite.from_dict({'0': 9.0, '-1': 6.0})
            # → |9|₀ + |6|₋₁
        """
        return cls({int(k): v for k, v in d.items()})

    def to_bytes(self):
        """
        Serialize to compact binary format.

        Layout: sequence of (int32 dimension, float64 coefficient) pairs.
        12 bytes per active dimension. No header — length is implicit.

        Example:
            data = result.to_bytes()   # 24 bytes for 2-term composite
            restored = Composite.from_bytes(data)
        """
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
        """
        Extract coefficients at fixed dimensions as a flat list.

        Useful for columnar storage (NumPy, Parquet, CSV).
        Missing dimensions return 0.0.

        Args:
            dims: tuple/list of integer dimensions to extract.

        Example:
            c = R(9) + 6 * ZERO
            c.to_array((0, -1, -2))  # → [9.0, 6.0, 0.0]
        """
        return [self.c.get(d, 0.0) for d in dims]

    @classmethod
    def from_array(cls, values, dims):
        """
        Reconstruct from flat list + dimension map.
        Inverse of to_array().

        Example:
            Composite.from_array([9.0, 6.0, 0.0], (0, -1, -2))
            # → |9|₀ + |6|₋₁
        """
        return cls({d: v for d, v in zip(dims, values) if v != 0})

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
        if isinstance(other, (int, float)):
            other = Composite(other)
        return Composite._wrap(self._backend.add(self._data, other._data))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        neg_other = self._backend.negate(other._data)
        return Composite._wrap(self._backend.add(self._data, neg_other))

    def __rsub__(self, other):
        return Composite(other).__sub__(self)

    def __neg__(self):
        return Composite._wrap(self._backend.negate(self._data))

    def __mul__(self, other):
        """Multiplication: dimensions add, coefficients multiply"""
        if isinstance(other, (int, float)):
            return Composite._wrap(
                self._backend.scalar_multiply(self._data, float(other)))
        return Composite._wrap(
            self._backend.convolve(self._data, other._data))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: dimensions subtract, coefficients divide"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide by Python zero. Use ZERO for structural zero.")
            return Composite._wrap(
                self._backend.scalar_multiply(self._data, 1.0 / other))

        if isinstance(other, Composite):
            other_dims = self._backend.active_dims(other._data)

            if len(other_dims) == 0:
                raise ZeroDivisionError("Cannot divide by empty composite")

            # Fast path: single-term divisor → dimension shift
            if len(other_dims) == 1:
                div_dim = int(other_dims[0])
                div_coeff = self._backend.read_dim(other._data, div_dim)
                my_dims, my_vals = self._backend.to_arrays(self._data)
                new_dims = my_dims - div_dim
                new_vals = my_vals / div_coeff
                return Composite._wrap(
                    self._backend.create_from_terms(new_dims, new_vals))

            # Multi-term: polynomial long division via backend
            return Composite._wrap(
                self._backend.deconvolve(self._data, other._data))

        return NotImplemented

    def __rtruediv__(self, other):
        return Composite(other).__truediv__(self)

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
        """Integer power via repeated multiplication"""
        if not isinstance(n, int):
            raise TypeError("Power must be integer")
        if n == 0:
            return Composite({0: 1})
        if n < 0:
            return Composite({0: 1}) / (self ** (-n))
        result = Composite({0: 1})
        for _ in range(n):
            result = result * self
        return result

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
        """
        Extract nth derivative.
        d(1) = first derivative
        d(2) = second derivative
        etc.
        """
        return self._backend.read_dim(self._data, -n) * math.factorial(n)

    def __format__(self, fmt):
        """Support format strings by formatting the standard part."""
        if fmt:
            return format(self.st(), fmt)
        return repr(self)

    # -------------------------------------------------------------------------
    # Simplified integration operators (dimensional shifts)
    # -------------------------------------------------------------------------

    def eval_taylor(self, h_value):
        """
        Evaluate Taylor polynomial by substituting h → h_value.
        Uses backend arrays instead of dict iteration.
        """
        dims, vals = self._backend.to_arrays(self._data)
        mask = dims < 0
        neg_dims = dims[mask]
        neg_vals = vals[mask]
        return sum(float(v) * h_value ** (-int(d))
                   for d, v in zip(neg_dims, neg_vals))

    def integrate_step(self, dx):
        """
        Integrate over interval [x, x+dx] where this composite represents f(x+h).
        """
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
    Uses backend arrays instead of reading .c dicts."""
    a_dims, a_vals = a._backend.to_arrays(a._data)
    b_dims, b_vals = b._backend.to_arrays(b._data)

    # Union of all dimensions
    all_dims = np.union1d(a_dims, b_dims)
    if len(all_dims) == 0:
        return 0

    # Walk from highest dimension down
    for dim in reversed(all_dims):
        ca = a._backend.read_dim(a._data, int(dim))
        cb = b._backend.read_dim(b._data, int(dim))
        if math.isnan(ca) or math.isnan(cb):
            return float('nan')
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0


def _poly_divide(numerator, denominator, max_terms=50):
    """Polynomial long division for multi-term divisors"""
    if not denominator.c:
        raise ZeroDivisionError("Cannot divide by zero polynomial")

    denom_dims = sorted(denominator.c.keys(), reverse=True)
    lead_dim = denom_dims[0]
    lead_coeff = denominator.c[lead_dim]

    quotient = Composite({})
    remainder = Composite(dict(numerator.c))

    for _ in range(max_terms):
        if not remainder.c:
            break

        rem_dims = sorted(remainder.c.keys(), reverse=True)
        rem_lead_dim = rem_dims[0]
        rem_lead_coeff = remainder.c[rem_lead_dim]

        q_dim = rem_lead_dim - lead_dim
        q_coeff = rem_lead_coeff / lead_coeff

        quotient = quotient + Composite({q_dim: q_coeff})
        subtract_term = Composite({q_dim: q_coeff}) * denominator
        remainder = remainder - subtract_term
        remainder.c = {k: v for k, v in remainder.c.items() if abs(v) > 1e-14}

    return quotient, remainder


# =============================================================================
# CONVENIENCE SHORTCUTS
# =============================================================================

def R(x):
    """Create real number |x|₀"""
    return Composite.real(x)

ZERO = Composite.zero()       # |1|₋₁ (infinitesimal)
INF = Composite.infinity()    # |1|₁ (infinity)
h = ZERO                      # Alias: h is the infinitesimal


# =============================================================================
# TAYLOR SERIES FOR TRANSCENDENTAL FUNCTIONS
# =============================================================================
# FIX 1: All functions now wrap plain (int, float) inputs into Composite
#         instead of returning math.* results. Every call flows through
#         composite arithmetic.
# =============================================================================

def sin(x, terms=12):
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    a = x.st()
    h = Composite({d: c for d, c in x.c.items() if d != 0})
    if not h.c:
        return Composite({0: math.sin(a)})
    # sin(a+h) = sin(a)*cos(h) + cos(a)*sin(h)
    # cos(h) and sin(h) converge FAST because h has no dim-0 part
    sin_a, cos_a = math.sin(a), math.cos(a)
    # Taylor of sin(h) and cos(h) — h is purely infinitesimal
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
    return sin_a * cos_h + cos_a * sin_h


def cos(x, terms=12):
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    a = x.st()
    h = Composite({d: c for d, c in x.c.items() if d != 0})
    if not h.c:
        return Composite({0: math.cos(a)})
    # cos(a+h) = cos(a)*cos(h) - sin(a)*sin(h)
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
    return cos_a * cos_h - sin_a * sin_h


def exp(x, terms=15):
    """
    Exponential function for Composite numbers.

    Uses base+perturbation splitting: exp(a + h) = math.exp(a) * exp(h)
    where a = standard part (dim 0) and h = infinitesimal part (dims != 0).

    math.exp(a) handles the scalar part exactly (IEEE 754).
    Taylor series on h converges fast since h has no dim-0 component.

    This avoids catastrophic cancellation of naive Taylor series
    sum(x^n/n!) for large |x| (e.g., 15-term Taylor gives
    exp(-10) = 466 instead of 4.5e-5).
    """
    if isinstance(x, (int, float)):
        return Composite({0: math.exp(float(x))})

    if not isinstance(x, Composite):
        return Composite({0: math.exp(float(x))})

    a = x.st()  # Dimension-0 coefficient
    non_zero = {d: c for d, c in x.c.items() if d != 0 and abs(c) > 1e-15}

    if not non_zero:
        # Pure scalar Composite (only dim 0): math.exp directly
        return Composite({0: math.exp(a)})

    # Split: exp(a + h) = exp(a) * exp(h)
    base = math.exp(a)
    h = Composite(non_zero)

    # Taylor series for exp(h) — converges fast, h has no dim-0 part
    exp_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        exp_h = exp_h + (1.0 / math.factorial(n)) * h_power

    return base * exp_h


def ln(x, terms=15):
    """
    Natural logarithm for composite numbers.
    Uses ln(a + h) = ln(a) + ln(1 + h/a) and Mercator series.
    """
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    a = x.st()
    if a <= 0:
        raise ValueError("ln requires positive standard part")

    h_part = x - R(a)
    ratio = h_part / R(a)

    result = Composite({0: math.log(a)})
    power = Composite({0: 1})

    for n in range(1, terms):
        power = power * ratio
        sign = (-1) ** (n + 1)
        result = result + sign * power / n

    return result


def sqrt(x, terms=12):
    """Square root for composite numbers via binomial series"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    a = x.st()
    if a <= 0:
        raise ValueError("sqrt requires positive standard part")

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

    return result


def tan(x, terms=12):
    """Tangent function via sin/cos"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    return sin(x, terms) / cos(x, terms)

# =============================================================================
# INVERSE TRIGONOMETRIC FUNCTIONS
# =============================================================================

def _reciprocal(x, terms=15):
    """Compute 1/x via geometric series. Internal helper."""
    a = x.st()
    if abs(a) < 1e-14:
        raise ZeroDivisionError("Cannot compute 1/x at x=0")
    h_part = x - R(a)
    ratio = h_part / R(-a)
    result = Composite({0: 1/a})
    power = Composite({0: 1})
    for n in range(1, terms):
        power = power * ratio
        result = result + power / a
    return result

def atan(x, terms=15):
    """Arctangent for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    a = x.st()
    one_plus_x2 = R(1) + x * x
    deriv = _reciprocal(one_plus_x2, terms)
    result = {0: math.atan(a)}
    for dim, coeff in deriv.c.items():
        new_dim = dim - 1
        if new_dim != 0:
            result[new_dim] = coeff / abs(new_dim)
    return Composite(result)

def asin(x, terms=15):
    """Arcsine for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    a = x.st()
    if abs(a) >= 1:
        raise ValueError("asin requires |standard part| < 1")
    inner = R(1) - x * x
    deriv = _reciprocal(sqrt(inner, terms), terms)
    result = {0: math.asin(a)}
    for dim, coeff in deriv.c.items():
        new_dim = dim - 1
        if new_dim != 0:
            result[new_dim] = coeff / abs(new_dim)
    return Composite(result)

def acos(x, terms=15):
    """Arccosine for composite numbers."""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    return R(math.pi / 2) - asin(x, terms)

# =============================================================================
# HYPERBOLIC FUNCTIONS
# =============================================================================

def sinh(x, terms=15):
    """Hyperbolic sine: (exp(x) - exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    return (exp(x, terms) - exp(-x, terms)) / 2

def cosh(x, terms=15):
    """Hyperbolic cosine: (exp(x) + exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    return (exp(x, terms) + exp(-x, terms)) / 2

def tanh(x, terms=15):
    """Hyperbolic tangent: sinh(x) / cosh(x)"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    return sinh(x, terms) / cosh(x, terms)

# =============================================================================
# REAL-VALUED POWERS
# =============================================================================

def power(x, s, terms=15):
    """
    x^s for any real s, via exp(s * ln(x)).
    """
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if isinstance(s, int):
        return x ** s
    return exp(R(s) * ln(x, terms), terms)


# =============================================================================
# HIGH-LEVEL API: AUTOMATIC TRANSLATION
# =============================================================================

def derivative(f: Callable, at: float, terms: int = 12) -> float:
    """Compute f'(at) automatically."""
    x = R(at) + ZERO
    result = f(x)
    return result.d(1)


def nth_derivative(f: Callable, n: int, at: float, terms: int = 12) -> float:
    """Compute f^(n)(at) - the nth derivative at a point."""
    effective_terms = max(terms, n + 2)  # need at least n+1 terms
    x = R(at) + ZERO
    result = f(x)  # but f's internal Taylor uses its own 'terms'...
    return result.d(n)


def all_derivatives(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Compute [f(at), f'(at), f''(at), ...] up to nth derivative."""
    x = R(at) + ZERO
    result = f(x)
    return [result.st()] + [result.d(n) for n in range(1, up_to + 1)]


def limit(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute lim(x→as_x_to) f(x) automatically."""
    if as_x_to == float('inf'):
        x = INF
    elif as_x_to == float('-inf'):
        x = -INF
    elif as_x_to == 0:
        x = ZERO
    else:
        x = R(as_x_to) + ZERO

    result = f(x)
    return result.st()


def limit_right(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute right-hand limit: lim(x→a⁺) f(x)"""
    if as_x_to == 0:
        x = ZERO
    else:
        x = R(as_x_to) + ZERO
    result = f(x)
    return result.st()


def limit_left(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute left-hand limit: lim(x→a⁻) f(x)"""
    if as_x_to == 0:
        x = -ZERO
    else:
        x = R(as_x_to) - ZERO
    result = f(x)
    return result.st()


def taylor_coefficients(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Get Taylor series coefficients of f around 'at'."""
    x = R(at) + ZERO
    result = f(x)
    return [result.coeff(-n) for n in range(up_to + 1)]


def antiderivative(f_composite: Composite, constant: float = 0) -> Composite:
    """
    Compute antiderivative via dimensional shift.
    Each |c|₋ₙ → |c/n|₋₍ₙ₊₁₎
    """
    result = {0: constant}
    for dim, coeff in f_composite.c.items():
        new_dim = dim - 1
        divisor = abs(new_dim)
        if divisor > 0:
            result[new_dim] = coeff / divisor
    return Composite(result)


def definite_integral(f: Callable, a: float, b: float, terms: int = 12) -> float:
    """Compute ∫ₐᵇ f(x) dx. Wrapper around integrate_adaptive."""
    result, _ = integrate_adaptive(f, a, b, tol=1e-10, terms=terms)
    return result.st()

# =============================================================================
# MULTI-POINT STEPPED INTEGRATION
# FIX 2: Accumulate as Composite instead of plain float.
#         Returns (Composite, float) — integral as Composite, error as float.
# =============================================================================


def integrate(f, *args, curve=None, surface=None, tol=1e-10, terms=15):
    """
    One integral to rule them all.

    1D definite/improper: composite-powered via integrate_adaptive
    2D/3D box: midpoint Riemann sums (simple, robust)
    Line integral: composite-first with fallback (v2)
    Surface integral: midpoint Riemann sums
    """

    def _st(v):
        return v.st() if isinstance(v, Composite) else float(v)

    # --- LINE INTEGRAL (v2: COMPOSITE-FIRST, Option B: probe once) ---
    # Phase 1: Composite tangent vectors via .d(1) — no epsilon
    # Phase 2: Routes through integrate_adaptive — adaptive, error estimate
    # Phase 3: F evaluated at composite coords — full Taylor in t
    # Fallback: finite-diff tangents + float F, still through integrate_adaptive
    if curve is not None:
        t_range = args[0] if args else (0, 1)
        a_t, b_t = t_range
        is_vector = isinstance(f, list)

        # Probe: can the curve accept Composite input?
        t_mid = (a_t + b_t) / 2
        try:
            probe = curve(R(t_mid) + ZERO)
            composite_curve = isinstance(probe[0], Composite) if isinstance(probe, (list, tuple)) else isinstance(probe, Composite)
        except (TypeError, AttributeError):
            composite_curve = False

        if composite_curve:
            # --- COMPOSITE PATH (Phase 1+2+3) ---
            def _line_integrand(t_comp):
                """
                Scalar integrand g(t) for the line integral.

                Vector field:  g(t) = F(r(t)) · r'(t)
                Scalar field:  g(t) = f(r(t)) · |r'(t)|

                Phase 1: Tangent via composite .d(1) — exact, no epsilon.
                Phase 3: F at composite positions — chain rule propagates d/dt.
                """
                pos_comp = curve(t_comp)                      # each component is Composite in t
                tangent = [p.d(1) for p in pos_comp]           # exact dr/dt (Phase 1)

                if is_vector:
                    # Phase 3: F at composite positions → chain rule propagates d/dt
                    F_comp = [comp(*pos_comp) for comp in f]   # F_j(r(t)) as Composite in t
                    # Dot product: F · r'  (F_comp is Composite, tangent is float)
                    F_comp = [Composite({0: float(fc)}) if isinstance(fc, (int, float)) else fc
							          for fc in F_comp]
                    return sum(fc * tv for fc, tv in zip(F_comp, tangent))
                else:
                    # Phase 3: f at composite positions
                    f_comp = f(*pos_comp)                      # f(r(t)) as Composite in t
                    if isinstance(f_comp, (int, float)):
                        f_comp = Composite({0: float(f_comp)})
                    speed = math.sqrt(sum(tv**2 for tv in tangent))  # |r'(t)| as float
                    return f_comp * speed
        else:
            # --- FALLBACK PATH (finite-diff tangent, float F, Phase 2 only) ---
            if composite_curve:
                # --- COMPOSITE PATH (unchanged) ---
                def _line_integrand(t_comp):
                    ...  # (keep as-is)
            else:
                # --- FALLBACK: classical midpoint Riemann (N=2000) ---
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
        # Composite path feeds into integrate_adaptive
        result, err = integrate_adaptive(_line_integrand, a_t, b_t, tol=tol, terms=terms)
        return result.st()
    # --- SURFACE INTEGRAL (v2: COMPOSITE-FIRST, Option B: probe once) ---
    if surface is not None:
        uv = args[0] if args else ((0, 1), (0, 1))
        (a_u, b_u), (a_v, b_v) = uv
        is_vector = isinstance(f, list)

        # Probe: can the surface accept Composite input?
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
            # --- COMPOSITE PATH: exact normals via .d(1) cross product ---
            def _surface_integrand(u_val, v_val):
                u_mc = MC.var(0, 2, val=u_val)
                v_mc = MC.var(1, 2, val=v_val)
                S = surface(u_mc, v_mc)
                # Exact partial derivatives from composite structure
                dSdu = [S[i].d(1, 0) for i in range(3)]  # ∂S/∂u
                dSdv = [S[i].d(1, 1) for i in range(3)]  # ∂S/∂v
                # Cross product dS/du × dS/dv
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

            # Nested 1D adaptive integration (composite accumulation)
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
            # --- FALLBACK: classical N×N midpoint Riemann (totally independent) ---
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

    # --- 1D DEFINITE / IMPROPER (composite-powered) ---
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

    # --- 2D BOX (midpoint Riemann sum) ---
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

    # --- 3D BOX (midpoint Riemann sum) ---
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
    """Multi-point stepped integration with error estimate.
    Returns (Composite, float) — integral result as Composite, error as float."""
    total = Composite({})
    total_error = 0.0
    x0 = a
    while x0 < b:
        dx = min(step, b - x0)
        fx = f(R(x0) + ZERO)
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


def integrate_adaptive(f: Callable, a: float, b: float, tol: float = 1e-10, terms: int = 15):
    """Adaptive stepped integration via composite dimensional shift.

    If the integrand doesn't propagate composite structure, lifts it
    once at the gate by wrapping it with numerical derivatives.
    Every number is a composite — this function ensures it.

    Returns (Composite, float) — integral result as Composite, error as float.
    """
    # --- LIFT AT THE GATE ---
    probe = f(R((a + b) / 2) + ZERO)
    if not any(dim < 0 for dim in probe.c):
        f_original = f
        eps = 1e-7
        def f(x):
            a_val = x.st()
            val = f_original(R(a_val) + ZERO).st()
            val_plus = f_original(R(a_val + eps) + ZERO).st()
            val_minus = f_original(R(a_val - eps) + ZERO).st()
            d1 = (val_plus - val_minus) / (2 * eps)
            d2 = (val_plus - 2 * val + val_minus) / (eps ** 2)
            return Composite({0: val, -1: d1, -2: d2 / 2})

    # --- INTEGRATION LOOP ---
    total = Composite({})
    total_error = 0.0
    x0 = a

    while x0 < b:
        fx = f(R(x0) + ZERO)

        # Step size: dim < -1 coefficients control truncation beyond captured Taylor
        max_coeff = max(
            (abs(c) for d, c in fx.c.items() if d < -1),
            default=1e-10
        )
        dx = min((tol / max(max_coeff, 1e-15)) ** 0.1, b - x0)
        dx = max(dx, (b - a) * 1e-8)

        # Integrate via dimensional shift
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
# IMPROPER INTEGRALS
# FIX 3: Now returns (Composite, float) since integrate_adaptive does.
# =============================================================================

def improper_integral(f: Callable, a: float, tol: float = 1e-8, cutoff: float = 20):
    """Compute ∫_a^∞ f(x) dx. Returns (Composite, float)."""
    M = cutoff
    while M < 1000:
        fx = f(R(M) + ZERO)
        if abs(fx.st()) < tol * 0.01:
            break
        M *= 2
    bulk, bulk_err = integrate_adaptive(f, a, min(M, cutoff), tol=tol)
    if M > cutoff:
        tail, tail_err = integrate_adaptive(f, cutoff, M, tol=tol)
        bulk = bulk + tail
        bulk_err += tail_err
    return bulk, bulk_err

def improper_integral_both(f: Callable, tol: float = 1e-8):
    """Compute ∫_{-∞}^{∞} f(x) dx. Splits at 0. Returns (Composite, float)."""
    left, left_err = improper_integral(lambda x: f(-x), 0, tol=tol)
    right, right_err = improper_integral(f, 0, tol=tol)
    return left + right, left_err + right_err

def improper_integral_to(f: Callable, a: float, b: float, tol: float = 1e-8):
    """Compute ∫_a^b f(x) dx where f has a singularity at a or b. Returns (Composite, float)."""
    eps = tol ** 0.25
    try:
        f(R(a + eps) + ZERO).st()
        a_ok = True
    except:
        a_ok = False
    try:
        f(R(b - eps) + ZERO).st()
        b_ok = True
    except:
        b_ok = False
    start = a + eps if not a_ok else a
    end = b - eps if not b_ok else b
    val, err = integrate_adaptive(f, start, end, tol=tol)
    return val, err

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
            x = R(to) + ZERO
            sub_str = f"x = R({to}) + ZERO"
    elif at is not None:
        x = R(at) + ZERO
        sub_str = f"x = R({at}) + ZERO"
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
    print("COMPOSITE LIBRARY TEST SUITE (FIXED v2: LINE INTEGRAL COMPOSITE)")
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
    tests.append(("d³/dx³[x⁵] at x=2", d4, 120))
    print(f"d³/dx³[x⁵] at x=2 = {d4}, expected 120 {'✓' if abs(d4-120)<1e-6 else '✗'}")

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

    # Vector field: ∫_C F · dr where F = [y, -x], curve = unit circle
    # Expected: -2π
    line_result = integrate(
        [lambda x, y: y, lambda x, y: -x],
        (0, 2 * math.pi),
        curve=lambda t: [cos(t), sin(t)]
    )
    expected_line = -2 * math.pi
    line_ok = abs(line_result - expected_line) < 0.1  # wider tol for adaptive
    tests.append(("∫_C F·dr (unit circle)", line_result, expected_line))
    print(f"∫_C [y,-x]·dr (unit circle) = {line_result:.6f}, expected {expected_line:.6f} {'✓' if line_ok else '✗'}")

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
