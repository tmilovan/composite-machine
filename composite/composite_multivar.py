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
composite_multivar.py — Multi-Variable Calculus Extension
=================================================================
Extends composite arithmetic from scalar dimensions (int) to
tuple dimensions (tuple of ints) for multi-variable calculus.

Same algebra: dimensions add component-wise, coefficients multiply.
Same mechanism: evaluate f(x, y, ...) on composite inputs, read off
partial derivatives, gradients, Hessians from tuple positions.

FIXES applied:
  1. Constructor: v != 0 (not abs(v) > 1e-15)
  2. Transcendentals always return MC (never plain float)
  3. __abs__, __float__, __int__ added to MC
  4. divergence_of is @staticmethod; curl_at is standalone function

Usage:
    from composite_multivar import *

    # Two variables
    x = RR(3, var=0, nvars=2)   # x = 3 + hx
    y = RR(2, var=1, nvars=2)   # y = 2 + hy
    result = x**2 * y + mc_sin(y)

    print(result.st())           # f(3,2) = 18 + sin(2)
    print(result.partial(1,0))   # ∂f/∂x = 2*3*2 = 12
    print(result.partial(0,1))   # ∂f/∂y = 9 + cos(2)
    print(result.partial(1,1))   # ∂²f/∂x∂y = 6
    print(result.gradient())     # [12, 9+cos(2)]
    print(result.hessian())      # [[4, 6], [6, -sin(2)]]

Author: Toni Milovan
"""
import math
from typing import Callable, List, Tuple, Dict, Optional
class MC:
    """
    MultiComposite: composite number with tuple dimensions.

    Each term is |coefficient|_{(d1, d2, ..., dn)} where
    the tuple encodes partial derivative orders for each variable.

    Arithmetic rules (identical to single-variable):
      - Multiply: coefficients multiply, dimensions add component-wise
      - Divide: coefficients divide, dimensions subtract component-wise
      - Add: same-dimension terms combine, cross-dimension terms coexist

    Examples (2 variables):
      |5|_{(0,0)}      = real number 5
      |1|_{(-1,0)}     = infinitesimal in x direction
      |1|_{(0,-1)}     = infinitesimal in y direction
      |3|_{(-1,-1)}    = mixed second-order infinitesimal
    """

    __slots__ = ['c', 'nvars']

    def __init__(self, coefficients=None, nvars=1):
        self.nvars = nvars
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            key = tuple([0] * nvars)
            # FIX 1: Use v != 0 instead of abs(v) > 1e-15
            self.c = {key: float(coefficients)} if coefficients != 0 else {}
        elif isinstance(coefficients, dict):
            # FIX 1: Use v != 0 instead of abs(v) > 1e-15
            self.c = {k: v for k, v in coefficients.items() if v != 0}
            # Infer nvars from keys if not specified
            if self.c and nvars == 1:
                first_key = next(iter(self.c))
                if isinstance(first_key, tuple):
                    self.nvars = len(first_key)
        else:
            raise TypeError(f"Cannot create MC from {type(coefficients)}")

    def _zero_key(self):
        return tuple([0] * self.nvars)

    def _ensure_compatible(self, other):
        if isinstance(other, MC):
            return max(self.nvars, other.nvars)
        return self.nvars

    def _promote_key(self, key, target_nvars):
        """Extend a dimension tuple to target length by padding with zeros."""
        if len(key) < target_nvars:
            return key + tuple([0] * (target_nvars - len(key)))
        return key

    def _promote(self, target_nvars):
        """Promote all keys to target_nvars dimensions."""
        if self.nvars == target_nvars:
            return self
        new_c = {}
        for k, v in self.c.items():
            new_k = self._promote_key(k, target_nvars)
            new_c[new_k] = v
        result = MC(new_c, target_nvars)
        return result

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def real(cls, value, nvars=1):
        """Real number at dimension (0,0,...,0)"""
        key = tuple([0] * nvars)
        return cls({key: float(value)}, nvars)

    @classmethod
    def zero_var(cls, var, nvars):
        """
        Structural zero (infinitesimal) in variable direction.
        var=0 → hx = |1|_{(-1,0,...)}
        var=1 → hy = |1|_{(0,-1,...)}
        """
        key = tuple(-1 if i == var else 0 for i in range(nvars))
        return cls({key: 1.0}, nvars)

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        if not self.c:
            return f"|0|_{tuple([0]*self.nvars)}"

        def fmt_coeff(c):
            if isinstance(c, float) and c == int(c):
                return str(int(c))
            elif isinstance(c, float):
                return f"{c:.6g}"
            return str(c)

        # Sort by sum of dimensions (descending), then lexicographic
        terms = sorted(self.c.items(), key=lambda x: (-sum(x[0]), x[0]))
        parts = [f"|{fmt_coeff(coeff)}|_{dim}" for dim, coeff in terms]
        return " + ".join(parts)

    # -------------------------------------------------------------------------
    # FIX 3: Add __abs__, __float__, __int__ dunder methods
    # -------------------------------------------------------------------------

    def __abs__(self):
        """Return abs of standard part (for use with built-in abs())."""
        return abs(self.st())

    def __float__(self):
        """Return standard part as float (for use with float())."""
        return float(self.st())

    def __int__(self):
        """Return standard part as int (for use with int())."""
        return int(self.st())

    # -------------------------------------------------------------------------
    # Arithmetic (identical rules, tuple dimensions)
    # -------------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = MC(other, self.nvars)
        nv = self._ensure_compatible(other)
        a = self._promote(nv)
        b = other._promote(nv)
        result = dict(a.c)
        for dim, coeff in b.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return MC(result, nv)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = MC(other, self.nvars)
        nv = self._ensure_compatible(other)
        a = self._promote(nv)
        b = other._promote(nv)
        result = dict(a.c)
        for dim, coeff in b.c.items():
            result[dim] = result.get(dim, 0) - coeff
        return MC(result, nv)

    def __rsub__(self, other):
        return MC(other, self.nvars).__sub__(self)

    def __neg__(self):
        return MC({k: -v for k, v in self.c.items()}, self.nvars)

    def __mul__(self, other):
        """Multiply: coefficients multiply, dimensions add component-wise."""
        if isinstance(other, (int, float)):
            return MC({k: v * other for k, v in self.c.items()}, self.nvars)
        nv = self._ensure_compatible(other)
        a = self._promote(nv)
        b = other._promote(nv)
        result = {}
        for d1, c1 in a.c.items():
            for d2, c2 in b.c.items():
                dim = tuple(d1[i] + d2[i] for i in range(nv))
                result[dim] = result.get(dim, 0) + c1 * c2
        return MC(result, nv)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide: coefficients divide, dimensions subtract component-wise."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Use MC.zero_var() for structural zero.")
            return MC({k: v / other for k, v in self.c.items()}, self.nvars)

        if len(other.c) == 0:
            raise ZeroDivisionError("Cannot divide by empty composite")

        nv = self._ensure_compatible(other)
        a = self._promote(nv)
        b = other._promote(nv)

        if len(b.c) == 1:
            # Single-term divisor: fast path
            div_dim, div_coeff = list(b.c.items())[0]
            result = {}
            for dim, coeff in a.c.items():
                new_dim = tuple(dim[i] - div_dim[i] for i in range(nv))
                result[new_dim] = coeff / div_coeff
            return MC(result, nv)

        # Multi-term: polynomial long division
        return _mc_poly_divide(a, b, nv)[0]

    def __rtruediv__(self, other):
        return MC(other, self.nvars).__truediv__(self)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Power must be integer. Use mc_power() for real exponents.")
        if n == 0:
            return MC.real(1, self.nvars)
        if n < 0:
            return MC.real(1, self.nvars) / (self ** (-n))
        result = MC.real(1, self.nvars)
        for _ in range(n):
            result = result * self
        return result

    # -------------------------------------------------------------------------
    # Extraction methods
    # -------------------------------------------------------------------------

    def st(self):
        """Standard part: coefficient at (0,0,...,0)"""
        return self.c.get(self._zero_key(), 0.0)

    def coeff(self, *dims):
        """Get coefficient at specific tuple dimension."""
        if len(dims) == 1 and isinstance(dims[0], tuple):
            key = dims[0]
        else:
            key = tuple(dims)
        return self.c.get(key, 0.0)

    def partial(self, *orders):
        """
        Extract partial derivative value.
        partial(n1, n2, ...) = ∂^(n1+n2+...)f / ∂x1^n1 ∂x2^n2 ...

        Example:
            result.partial(1, 0)   # ∂f/∂x
            result.partial(0, 1)   # ∂f/∂y
            result.partial(2, 0)   # ∂²f/∂x²
            result.partial(1, 1)   # ∂²f/∂x∂y
        """
        key = tuple(-o for o in orders)
        raw = self.c.get(key, 0.0)
        # Multiply by product of factorials: n1! * n2! * ...
        scale = 1
        for o in orders:
            scale *= math.factorial(o)
        return raw * scale

    def gradient(self):
        """
        Extract gradient vector [∂f/∂x1, ∂f/∂x2, ...].
        """
        result = []
        for var in range(self.nvars):
            orders = tuple(-1 if i == var else 0 for i in range(self.nvars))
            result.append(self.c.get(orders, 0.0))
        return result

    def hessian(self):
        """
        Extract Hessian matrix [[∂²f/∂xi∂xj]].
        """
        n = self.nvars
        H = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                orders = [0] * n
                orders[i] -= 1
                orders[j] -= 1
                key = tuple(orders)
                raw = self.c.get(key, 0.0)
                # Scale: if i == j, it's f''/2!, so multiply by 2
                # if i != j, it's the mixed partial coefficient directly
                if i == j:
                    H[i][j] = raw * 2  # ∂²f/∂x² = 2 * coeff at (-2,0)
                else:
                    H[i][j] = raw      # ∂²f/∂x∂y = coeff at (-1,-1)
        return H

    def laplacian(self):
        """
        Compute Laplacian: ∇²f = Σ ∂²f/∂xi²
        """
        total = 0.0
        for i in range(self.nvars):
            orders = [0] * self.nvars
            orders[i] = -2
            key = tuple(orders)
            total += self.c.get(key, 0.0) * 2  # times 2!/1 = 2
        return total

    # FIX 4: divergence_of as @staticmethod (was wrongly indented without self)
    @staticmethod
    def divergence_of(components):
        """
        Compute divergence of a vector field F = [f1, f2, ...].
        div(F) = ∂f1/∂x1 + ∂f2/∂x2 + ...
        Takes a list of MC objects (one per component).
        """
        total = 0.0
        for i, comp in enumerate(components):
            key = tuple(-1 if j == i else 0 for j in range(comp.nvars))
            total += comp.c.get(key, 0.0)
        return total


# FIX 4: curl_at as standalone function (was tangled with MC class definition)
def curl_at(F: List[Callable], at: List[float]):
    """
    Compute curl of a 3D vector field F = [Fx, Fy, Fz] at a point.
    ∇ × F = (∂Fz/∂y - ∂Fy/∂z,
             ∂Fx/∂z - ∂Fz/∂x,
             ∂Fy/∂x - ∂Fx/∂y)
    Returns [curl_x, curl_y, curl_z]

    Example:
        # Curl of F = [y, -x, 0] (rotation field) at (1, 1, 0)
        curl_at([lambda x,y,z: y, lambda x,y,z: -x, lambda x,y,z: 0*x],
                [1, 1, 0])
        # → [0, 0, -2]  (rotation around z-axis)
    """
    if len(at) != 3:
        raise ValueError("Curl requires 3D vector field")
    nvars = 3
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    # Evaluate each component as MC
    Fx = F[0](*args)
    Fy = F[1](*args)
    Fz = F[2](*args)
    # Extract partials
    curl_x = Fz.partial(0, 1, 0) - Fy.partial(0, 0, 1)
    curl_y = Fx.partial(0, 0, 1) - Fz.partial(1, 0, 0)
    curl_z = Fy.partial(1, 0, 0) - Fx.partial(0, 1, 0)
    return [curl_x, curl_y, curl_z]


def _mc_poly_divide(num, den, nvars, max_terms=20):
    """Polynomial long division for multi-term tuple-dimension divisors."""
    if not den.c:
        raise ZeroDivisionError("Cannot divide by zero polynomial")

    # Find leading term of denominator (highest total dimension)
    denom_sorted = sorted(den.c.items(), key=lambda x: (-sum(x[0]), x[0]))
    lead_dim, lead_coeff = denom_sorted[0]

    quotient = MC({}, nvars)
    remainder = MC(dict(num.c), nvars)

    for _ in range(max_terms):
        if not remainder.c:
            break
        rem_sorted = sorted(remainder.c.items(), key=lambda x: (-sum(x[0]), x[0]))
        rem_dim, rem_coeff = rem_sorted[0]

        # Check if remainder leading dim >= denominator leading dim
        if sum(rem_dim) < sum(lead_dim):
            break

        q_dim = tuple(rem_dim[i] - lead_dim[i] for i in range(nvars))
        q_coeff = rem_coeff / lead_coeff

        quotient = quotient + MC({q_dim: q_coeff}, nvars)
        subtract = MC({q_dim: q_coeff}, nvars) * den
        remainder = remainder - subtract
        remainder.c = {k: v for k, v in remainder.c.items() if abs(v) > 1e-14}

    return quotient, remainder


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def RR(value, var=0, nvars=2):
    """
    Create a composite variable: real value + infinitesimal in var direction.

    RR(3, var=0, nvars=2) → |3|_{(0,0)} + |1|_{(-1,0)}  (x = 3 + hx)
    RR(2, var=1, nvars=2) → |2|_{(0,0)} + |1|_{(0,-1)}  (y = 2 + hy)
    """
    real_key = tuple([0] * nvars)
    inf_key = tuple(-1 if i == var else 0 for i in range(nvars))
    return MC({real_key: float(value), inf_key: 1.0}, nvars)


def RR_const(value, nvars=2):
    """
    Create a constant (no infinitesimal seed) in multi-var space.
    Useful for parameters that aren't differentiated.

    RR_const(5, nvars=2) → |5|_{(0,0)}
    """
    return MC.real(value, nvars)


# =============================================================================
# TRANSCENDENTAL FUNCTIONS
# FIX 2: Always return MC, never plain float
# =============================================================================

def mc_sin(x, terms=12):
    """Sine via angle-addition splitting. Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)

    a = x.st()  # scalar part
    zero_key = tuple([0] * x.nvars)
    non_zero = {d: c for d, c in x.c.items()
                if d != zero_key and abs(c) > 1e-15}

    if not non_zero:
        return MC({zero_key: math.sin(a)}, x.nvars)

    # sin(a + h) = sin(a)*cos(h) + cos(a)*sin(h)
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    h = MC(non_zero, x.nvars)

    # Taylor for sin(h) and cos(h) — h has no dim-0, converges fast
    sin_h = MC({}, x.nvars)
    cos_h = MC({zero_key: 1.0}, x.nvars)
    h_power = MC({zero_key: 1.0}, x.nvars)

    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:  # odd terms → sin
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:            # even terms → cos
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power

    return sin_a * cos_h + cos_a * sin_h


def mc_cos(x, terms=12):
    """Cosine via angle-addition splitting. Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)

    a = x.st()  # scalar part
    zero_key = tuple([0] * x.nvars)
    non_zero = {d: c for d, c in x.c.items()
                if d != zero_key and abs(c) > 1e-15}

    if not non_zero:
        return MC({zero_key: math.cos(a)}, x.nvars)

    # cos(a + h) = cos(a)*cos(h) - sin(a)*sin(h)
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    h = MC(non_zero, x.nvars)

    # Taylor for sin(h) and cos(h) — h has no dim-0, converges fast
    sin_h = MC({}, x.nvars)
    cos_h = MC({zero_key: 1.0}, x.nvars)
    h_power = MC({zero_key: 1.0}, x.nvars)

    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power

    return cos_a * cos_h - sin_a * sin_h


def mc_exp(x, terms=15):
    """Exponential via base+perturbation splitting. Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)

    a = x.st()
    zero_key = tuple([0] * x.nvars)
    non_zero = {d: c for d, c in x.c.items()
                if d != zero_key and abs(c) > 1e-15}

    if not non_zero:
        return MC({zero_key: math.exp(a)}, x.nvars)

    base = math.exp(a)
    h = MC(non_zero, x.nvars)

    exp_h = MC({zero_key: 1.0}, x.nvars)
    h_power = MC({zero_key: 1.0}, x.nvars)
    for n in range(1, terms):
        h_power = h_power * h
        exp_h = exp_h + (1.0 / math.factorial(n)) * h_power

    return base * exp_h


def mc_ln(x, terms=15):
    """Natural log via Mercator series. Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)
    a = x.st()
    if a <= 0:
        raise ValueError("ln requires positive standard part")
    h_part = x - MC.real(a, x.nvars)
    ratio = h_part / MC.real(a, x.nvars)
    result = MC.real(math.log(a), x.nvars)
    power = MC.real(1, x.nvars)
    for n in range(1, terms):
        power = power * ratio
        sign = (-1) ** (n + 1)
        result = result + sign * power / n
    return result


def mc_sqrt(x, terms=12):
    """Square root via binomial series. Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)
    a = x.st()
    if a <= 0:
        raise ValueError("sqrt requires positive standard part")
    sqrt_a = math.sqrt(a)
    h_part = x - MC.real(a, x.nvars)
    ratio = h_part / MC.real(a, x.nvars)
    def binom(n):
        if n == 0: return 1
        r = 1
        for k in range(n):
            r *= (0.5 - k)
        return r / math.factorial(n)
    result = MC.real(sqrt_a, x.nvars)
    power = MC.real(1, x.nvars)
    for n in range(1, terms):
        power = power * ratio
        result = result + binom(n) * sqrt_a * power
    return result


def mc_tan(x, terms=10):
    """Tangent via sin/cos. Always returns MC."""
    return mc_sin(x, terms) / mc_cos(x, terms)


def mc_power(x, s, terms=15):
    """x^s for any real s, via exp(s * ln(x)). Always returns MC."""
    if isinstance(x, (int, float)):
        x = MC.real(x, nvars=1)
    if isinstance(s, int):
        return x ** s
    return mc_exp(MC.real(s, x.nvars) * mc_ln(x, terms), terms)


# =============================================================================
# HIGH-LEVEL API: MULTI-VARIABLE CALCULUS
# =============================================================================

def partial_derivative(f, at: List[float], wrt: List[int]):
    """
    Compute partial derivative of f at a point.

    f: function of multiple MC arguments
    at: list of float values [x0, y0, ...]
    wrt: list of derivative orders [nx, ny, ...]

    Example:
        # ∂²f/∂x∂y of f(x,y) = x²y at (3, 2)
        partial_derivative(lambda x, y: x**2 * y, at=[3, 2], wrt=[1, 1])
        # → 6.0
    """
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    result = f(*args)
    return result.partial(*wrt)


def gradient_at(f, at: List[float]):
    """
    Compute gradient of f at a point.
    Returns [∂f/∂x1, ∂f/∂x2, ...].
    """
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    result = f(*args)
    return result.gradient()


def hessian_at(f, at: List[float]):
    """
    Compute Hessian matrix of f at a point.
    Returns [[∂²f/∂xi∂xj]].
    """
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    result = f(*args)
    return result.hessian()


def jacobian_at(fs: List[Callable], at: List[float]):
    """
    Compute Jacobian matrix of vector function F = [f1, f2, ...] at a point.
    Returns [[∂fi/∂xj]].
    """
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    J = []
    for f in fs:
        result = f(*args)
        J.append(result.gradient())
    return J


def laplacian_at(f, at: List[float]):
    """
    Compute Laplacian ∇²f at a point.
    Returns scalar: ∂²f/∂x² + ∂²f/∂y² + ...
    """
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    result = f(*args)
    return result.laplacian()


def directional_derivative(f, at: List[float], direction: List[float]):
    """
    Compute directional derivative ∇f · v̂ at a point.
    The direction vector is automatically normalized.
    """
    grad = gradient_at(f, at)
    # Normalize direction
    norm = math.sqrt(sum(d**2 for d in direction))
    v_hat = [d / norm for d in direction]
    return sum(g * v for g, v in zip(grad, v_hat))


def multivar_limit(f, as_vars_to: List[float]):
    """
    Compute lim_{(x,y,...) → (a,b,...)} f(x,y,...).
    """
    nvars = len(as_vars_to)
    args = []
    for i, val in enumerate(as_vars_to):
        if val == 0:
            args.append(MC.zero_var(i, nvars))
        else:
            args.append(RR(val, var=i, nvars=nvars))
    result = f(*args)
    return result.st()


def double_integral(f, x_range, y_range, tol=1e-8):
    """
    Compute ∫∫ f(x,y) dy dx by iterated single-variable integration.
    x_range = (a, b), y_range = (c, d)

    Uses composite adaptive integration in each variable.
    """
    from composite_lib import integrate_adaptive, R, ZERO, Composite

    a, b = x_range
    c, d = y_range

    def inner(x_val):
        """For fixed x, integrate over y."""
        def g(y_comp):
            x_mc = RR_const(x_val, nvars=2)
            y_mc = MC({(0,0): y_comp.st(), (0,-1): y_comp.coeff(-1)}, nvars=2)
            result_mc = f(x_mc, y_mc)
            out = Composite({})
            for dim, coeff in result_mc.c.items():
                if dim[0] == 0:
                    out.c[dim[1]] = out.c.get(dim[1], 0) + coeff
            return out

        val, err = integrate_adaptive(g, c, d, tol=tol)
        return val

    total = 0.0
    n_steps = max(20, int((b - a) / 0.05))
    dx = (b - a) / n_steps
    for i in range(n_steps):
        xi = a + (i + 0.5) * dx
        total += inner(xi) * dx

    return total
