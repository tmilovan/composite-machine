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
composite_lib.py — Unified Calculus Library (Fixed: Fully Composite)
====================================================================
All operations use composite arithmetic. No plain-number fast paths
in transcendental functions. Integration accumulates as Composite.

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

    __slots__ = ['c']

    def __init__(self, coefficients=None):
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            self.c = {0: float(coefficients)} if coefficients != 0 else {}
        elif isinstance(coefficients, dict):
            self.c = {k: v for k, v in coefficients.items() if v != 0}
        else:
            raise TypeError(f"Cannot create Composite from {type(coefficients)}")

    # -------------------------------------------------------------------------
    # Constructors
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
        if not self.c:
            return "|0|₀"

        sub = "₀₁₂₃₄₅₆₇₈₉"
        def fmt_dim(n):
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            else:
                return "₋" + ''.join(sub[int(d)] for d in str(-n))

        def fmt_coeff(c):
            if isinstance(c, complex) and c.imag == 0:
                c = c.real
            if isinstance(c, float) and c == int(c):
                return str(int(c))
            elif isinstance(c, float):
                return f"{c:.6g}"
            return str(c)

        terms = sorted(self.c.items(), key=lambda x: -x[0])
        parts = [f"|{fmt_coeff(coeff)}|{fmt_dim(dim)}" for dim, coeff in terms]
        return " + ".join(parts)

    # -------------------------------------------------------------------------
    # Arithmetic operations
    # -------------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return Composite(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) - coeff
        return Composite(result)

    def __rsub__(self, other):
        return Composite(other).__sub__(self)

    def __neg__(self):
        return Composite({k: -v for k, v in self.c.items()})

    def __mul__(self, other):
        """Multiplication: dimensions add, coefficients multiply"""
        if isinstance(other, (int, float)):
            return Composite({k: v * other for k, v in self.c.items()})
        result = {}
        for d1, c1 in self.c.items():
            for d2, c2 in other.c.items():
                dim = d1 + d2
                result[dim] = result.get(dim, 0) + c1 * c2
        return Composite(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: dimensions subtract, coefficients divide"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by Python zero. Use ZERO for structural zero.")
            return Composite({k: v / other for k, v in self.c.items()})

        if len(other.c) == 0:
            raise ZeroDivisionError("Cannot divide by empty composite")

        if len(other.c) == 1:
            div_dim, div_coeff = list(other.c.items())[0]
            result = {}
            for dim, coeff in self.c.items():
                result[dim - div_dim] = coeff / div_coeff
            return Composite(result)

        return _poly_divide(self, other)[0]

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
        return self.c.get(0, 0.0)

    def coeff(self, dim):
        """Get coefficient at specific dimension"""
        return self.c.get(dim, 0.0)

    def d(self, n=1):
        """
        Extract nth derivative.
        d(1) = first derivative
        d(2) = second derivative
        etc.
        """
        return self.c.get(-n, 0.0) * math.factorial(n)

    # -------------------------------------------------------------------------
    # Simplified integration operators (dimensional shifts)
    # -------------------------------------------------------------------------

    def eval_taylor(self, h_value):
        """
        Evaluate Taylor polynomial by substituting h → h_value.
        """
        return sum(coeff * h_value ** (-dim)
                   for dim, coeff in self.c.items() if dim < 0)

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
        return _compare(self, other) == 0

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


def _compare(a, b):
    """Lexicographic comparison by dimension (highest first)"""
    all_dims = set(a.c.keys()) | set(b.c.keys())
    if not all_dims:
        return 0
    for dim in sorted(all_dims, reverse=True):
        ca = a.c.get(dim, 0)
        cb = b.c.get(dim, 0)
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0


def _poly_divide(numerator, denominator, max_terms=20):
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
    """Sine function for composite numbers via Taylor series"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n + 1)
        result = result + coeff * (x ** (2*n + 1))
    return result


def cos(x, terms=12):
    """Cosine function for composite numbers via Taylor series"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n)
        result = result + coeff * (x ** (2*n))
    return result


def exp(x, terms=15):
    """Exponential function for composite numbers via Taylor series"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    # Check for infinite arguments (positive dimensions)
    max_dim = max(x.c.keys()) if x.c else 0
    if max_dim > 0:
        lead_coeff = x.c[max_dim]
        if lead_coeff < 0:
            return Composite({})
        else:
            return Composite({1: float('inf')})

    result = Composite({})
    for n in range(terms):
        coeff = 1 / math.factorial(n)
        result = result + coeff * (x ** n)
    return result


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


def tan(x, terms=10):
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
    x = R(at) + ZERO
    result = f(x)
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
    """Compute ∫ₐᵇ f(x) dx via antiderivative."""
    x_a = R(a) + ZERO
    x_b = R(b) + ZERO
    F_a = antiderivative(f(x_a))
    F_b = antiderivative(f(x_b))
    return F_b.st() - F_a.st()

# =============================================================================
# MULTI-POINT STEPPED INTEGRATION
# FIX 2: Accumulate as Composite instead of plain float.
#         Returns (Composite, float) — integral as Composite, error as float.
# =============================================================================

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
    """Adaptive stepped integration with error estimate.
    Returns (Composite, float) — integral result as Composite, error as float."""
    total = Composite({})
    total_error = 0.0
    x0 = a
    while x0 < b:
        fx = f(R(x0) + ZERO)
        max_coeff = max(
            (abs(coeff) for dim, coeff in fx.c.items() if dim < -1),
            default=1e-10
        )
        dx = min((tol / max(max_coeff, 1e-15)) ** 0.1, b - x0)
        dx = max(dx, 1e-6)
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
    print("COMPOSITE LIBRARY TEST SUITE (FIXED: FULLY COMPOSITE)")
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

    # All derivatives at once
    print("\n--- All Derivatives ---")

    derivs = all_derivatives(lambda x: exp(x), at=0, up_to=5)
    print(f"All derivatives of eˣ at x=0: {[round(d,2) for d in derivs]}")
    print(f"Expected: [1, 1, 1, 1, 1, 1] {'✓' if all(abs(d-1)<1e-6 for d in derivs) else '✗'}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, actual, expected in tests if abs(actual - expected) < 1e-6)
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
