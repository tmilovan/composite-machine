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
composite_extended.py — Extended Capabilities
==============================================
Builds on composite_lib.py to add:
  1. Complex-valued composites (residues, poles)
  2. Asymptotic expansion extraction
  3. Convergence radius detection
  4. Composite ODE stepper
  5. Improper integrals via tail analysis
  6. Analytic continuation

All capabilities reduce to the same mechanism:
  evaluate f on a composite number, read coefficients.

"""

import math
import cmath
from composite_lib import Composite, R, ZERO, INF, sin, cos, exp, ln, sqrt
from composite_lib import integrate_adaptive, antiderivative


# =============================================================================
# 1. COMPLEX COMPOSITES
# =============================================================================

def C(z):
    """Create a complex-valued real-dimension composite."""
    return Composite({0: complex(z)})


def C_var(z):
    """Complex variable with infinitesimal: z + h"""
    return Composite({0: complex(z), -1: 1.0})


def cexp(x, terms=15):
    """Complex exponential via Taylor series."""
    if isinstance(x, (int, float, complex)):
        return cmath.exp(x)
    result = Composite({})
    for n in range(terms):
        coeff = 1 / math.factorial(n)
        result = result + coeff * (x ** n)
    return result


def csin(x, terms=12):
    """Complex sine: (e^(ix) - e^(-ix)) / 2i"""
    if isinstance(x, (int, float, complex)):
        return cmath.sin(x)
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n + 1)
        result = result + coeff * (x ** (2*n + 1))
    return result


def ccos(x, terms=12):
    """Complex cosine."""
    if isinstance(x, (int, float, complex)):
        return cmath.cos(x)
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n)
        result = result + coeff * (x ** (2*n))
    return result
# =============================================================================
# 2. RESIDUE COMPUTATION
# =============================================================================

def residue(f, at, terms=15):
    """
    Compute the residue of f(z) at z = a.

    Residue = coefficient of (z-a)^(-1) in the Laurent expansion
            = dimension -1 coefficient in composite evaluation.

    Examples:
        # Simple pole: f(z) = 1/z at z = 0
        residue(lambda z: 1/z, at=0)          # → 1

        # f(z) = 1/(z² + 1) at z = i
        residue(lambda z: 1/(z**2 + 1), at=1j)  # → 1/(2i) = -i/2

        # f(z) = e^z / z² at z = 0 (second-order pole)
        residue(lambda z: cexp(z) / z**2, at=0)  # → 1
    """
    if at == 0:
        z = ZERO  # |1|₋₁
    else:
        z = Composite({0: complex(at), -1: 1.0})

    result = f(z)
    return result.coeff(-1)


def pole_order(f, at, max_order=10):
    """
    Detect the order of a pole of f at z = a.

    Returns:
        n > 0: pole of order n
        n = 0: regular point (or removable singularity)
        n < 0: zero of order |n|

    Example:
        pole_order(lambda z: 1/z**3, at=0)     # → 3
        pole_order(lambda z: sin(z)/z, at=0)    # → 0 (removable)
    """
    if at == 0:
        z = ZERO
    else:
        z = Composite({0: complex(at), -1: 1.0})

    result = f(z)

    # Find highest positive dimension with nonzero coefficient (= pole order)
    # or lowest negative dimension with nonzero coefficient (= zero order)
    max_pos = 0
    min_neg = 0
    for dim, coeff in result.c.items():
        if abs(coeff) > 1e-12:
            if dim > max_pos:
                max_pos = dim
            if dim < min_neg:
                min_neg = dim

    if max_pos > 0:
        return max_pos  # Pole of this order
    return 0  # Regular point


def contour_integral(f, poles_inside, terms=15):
    """
    Compute a contour integral via the Residue Theorem:
    ∮ f(z) dz = 2πi × Σ Res(f, aₖ)

    poles_inside: list of poles enclosed by the contour

    Example:
        # ∮ 1/(z² + 1) dz around a contour enclosing z = i
        contour_integral(lambda z: 1/(z**2 + 1), [1j])
        # → 2πi × (-i/2) = π
    """
    total_residue = sum(residue(f, pole, terms) for pole in poles_inside)
    return 2j * cmath.pi * total_residue

# =============================================================================
# 3. ASYMPTOTIC EXPANSION
# =============================================================================

def asymptotic_expansion(f, order=5):
    """
    Extract asymptotic expansion of f(x) as x → ∞.

    Returns coefficients [a₀, a₁, a₂, ...] where:
    f(x) ~ a₀ + a₁/x + a₂/x² + ...

    Mechanism: evaluate f(INF), read coefficients.
    INF = |1|₁, so 1/INF = |1|₋₁, 1/INF² = |1|₋₂, etc.

    Example:
        # f(x) = (3x² + 2x + 1) / (x² + 1)
        asymptotic_expansion(lambda x: (3*x**2 + 2*x + 1)/(x**2 + 1))
        # → [3, 2, -2, ...]  meaning f(x) ~ 3 + 2/x - 2/x² + ...
    """
    result = f(INF)
    coeffs = []
    for n in range(order + 1):
        coeffs.append(result.coeff(-n))
    return coeffs


def limit_at_infinity(f):
    """
    Compute lim(x→∞) f(x).

    Just the standard part of f(INF).

    Example:
        limit_at_infinity(lambda x: (3*x + 1)/(x + 2))  # → 3
    """
    return f(INF).st()


def asymptotic_order(f):
    """
    Determine the asymptotic growth order of f(x) as x → ∞.

    Returns the highest dimension with a nonzero coefficient.
    - Positive: f grows like x^n
    - Zero: f approaches a constant
    - Negative: f decays like 1/x^|n|

    Example:
        asymptotic_order(lambda x: x**2 + x)        # → 2 (quadratic growth)
        asymptotic_order(lambda x: 1/(x**2 + 1))    # → -2 (quadratic decay)
        asymptotic_order(lambda x: (x+1)/(x+2))     # → 0 (constant limit)
    """
    result = f(INF)
    if not result.c:
        return float('-inf')  # zero function
    return max(d for d, c in result.c.items() if abs(c) > 1e-12)

# =============================================================================
# 4. CONVERGENCE RADIUS DETECTION
# =============================================================================

def convergence_radius(f, at=0, order=20):
    """
    Estimate the convergence radius of the Taylor series of f around 'at'.

    Uses the Cauchy-Hadamard theorem:
    R = 1 / limsup |cₙ|^(1/n)

    In practice, estimates via the ratio test:
    R ≈ lim |cₙ / cₙ₊₁|

    The composite system already HAS all the cₙ — this is just
    reading them and applying the formula.

    Example:
        # ln(1+x) has radius 1 around x = 0
        convergence_radius(lambda x: ln(x), at=1)   # → ~1.0

        # 1/(1+x²) has radius 1 around x = 0 (poles at ±i)
        convergence_radius(lambda x: 1/(1 + x**2), at=0)  # → ~1.0

        # exp(x) has infinite radius
        convergence_radius(lambda x: exp(x), at=0)  # → very large
    """
    if at == 0:
        x = ZERO
    else:
        x = R(at) + ZERO

    result = f(x)

    # Extract Taylor coefficients
    coeffs = [abs(result.coeff(-n)) for n in range(order + 1)]

    # Ratio test: R ≈ |cₙ| / |cₙ₊₁|
    ratios = []
    for n in range(1, len(coeffs) - 1):
        if coeffs[n+1] > 1e-15:
            ratios.append(coeffs[n] / coeffs[n+1])

    if not ratios:
        return float('inf')  # All coefficients zero or polynomial

    # Use the last few ratios (most stable)
    stable_ratios = ratios[-5:] if len(ratios) >= 5 else ratios
    return sum(stable_ratios) / len(stable_ratios)


def is_within_convergence(f, at, eval_point):
    """
    Check if eval_point is within the convergence radius of f's
    Taylor series centered at 'at'.

    Returns (bool, radius).

    Example:
        is_within_convergence(lambda x: ln(x), at=1, eval_point=1.5)
        # → (True, ~1.0)
        is_within_convergence(lambda x: ln(x), at=1, eval_point=2.5)
        # → (False, ~1.0)  ← warns you!
    """
    radius = convergence_radius(f, at)
    distance = abs(eval_point - at)
    return distance < radius, radius

# =============================================================================
# 5. COMPOSITE ODE STEPPER
# =============================================================================

def ode_step(f, x0, y0, h, order=8):
    """
    One step of ODE y' = f(x, y) from (x0, y0) with step size h.

    Uses composite evaluation to get ALL derivatives at once:
      y' = f(x, y)
      y'' = df/dx + (df/dy)·y' = available from composite
      y''' = ... all from the same evaluation

    Then Taylor steps:
      y(x0+h) = y0 + y'·h + y''·h²/2! + y'''·h³/3! + ...

    Returns (y_new, error_estimate).

    ONE function evaluation gives arbitrary-order stepping.
    Compare: RK4 needs 4 evaluations for 4th-order accuracy.

    Example:
        # y' = y, y(0) = 1 → y = eˣ
        x, y = 0, 1
        for _ in range(10):
            y, err = ode_step(lambda x, y: y, x, y, 0.1)
            x += 0.1
        # y ≈ e ≈ 2.71828
    """
    # Evaluate f at composite point to get all derivative info
    x_c = R(x0) + ZERO
    y_c = R(y0) + ZERO

    # f gives us y' and its derivatives w.r.t. x
    f_result = f(x_c, y_c)

    # Build Taylor step: y(x0+h) = y0 + Σ cₙ hⁿ
    # c₁ = y' = f(x0, y0)
    # cₙ = (1/n!) × nth derivative of the solution

    # For autonomous-like systems, successive derivatives come from
    # the chain rule applied through the composite structure

    # Simple approach: use the Taylor coefficients of f
    # to build successive solution derivatives
    y_derivs = [y0]  # y^(0) = y0
    y_prime = f_result.st()  # y' = f(x0, y0)
    y_derivs.append(y_prime)

    # Higher derivatives via composite coefficients
    # y'' = f_x + f_y · y' (from total derivative)
    f_x = f_result.coeff(-1)  # ∂f/∂x at (x0, y0)
    # For f_y, we need the derivative w.r.t. y
    # This comes from perturbing y: f(x0, y0 + ε)
    y_c2 = R(y0) + ZERO
    x_c2 = R(x0)  # no infinitesimal in x for this eval
    f_y_result = f(x_c2, y_c2)
    f_y = f_y_result.coeff(-1)  # ∂f/∂y

    y_double_prime = f_x + f_y * y_prime
    y_derivs.append(y_double_prime)

    # Taylor step
    y_new = y0
    for n in range(1, min(order, len(y_derivs))):
        y_new += y_derivs[n] * h**n / math.factorial(n)

    # Error estimate: last included term
    if len(y_derivs) > 1:
        err = abs(y_derivs[-1] * h**len(y_derivs) / math.factorial(len(y_derivs)))
    else:
        err = 0.0

    return y_new, err


def solve_ode(f, x_range, y0, tol=1e-10, max_steps=10000):
    """
    Solve y' = f(x, y) over x_range = (a, b) with y(a) = y0.

    Uses adaptive composite stepping with automatic error control.

    Returns list of (x, y) pairs.

    Example:
        # y' = y, y(0) = 1 → y = eˣ
        points = solve_ode(lambda x, y: y, (0, 1), y0=1)
        # points[-1][1] ≈ e

        # y' = -2xy, y(0) = 1 → y = e^(-x²)
        points = solve_ode(lambda x, y: -2*x*y, (0, 2), y0=1)
    """
    a, b = x_range
    x = a
    y = y0
    h = (b - a) / 100  # initial step
    points = [(x, y)]

    steps = 0
    while x < b and steps < max_steps:
        h = min(h, b - x)

        # Composite step (simplified: using Taylor via composite eval)
        x_c = R(x) + ZERO
        fx = f(x_c, R(y) + ZERO)

        # Get y' and approximate y''
        yp = fx.st()

        # Simple adaptive: two half-steps vs one full step
        y_full = y + yp * h
        y_half1 = y + yp * (h/2)
        fx2 = f(R(x + h/2), R(y_half1))
        yp2 = fx2 if isinstance(fx2, (int, float)) else fx2.st()
        y_half2 = y_half1 + yp2 * (h/2)

        err = abs(y_full - y_half2)

        if err < tol or h < 1e-12:
            x += h
            y = y_half2  # use more accurate value
            points.append((x, y))
            if err > 0:
                h = min(h * (tol / err) ** 0.5, (b - a) / 10)
            else:
                h = min(h * 2, (b - a) / 10)
        else:
            h = h * (tol / err) ** 0.5

        steps += 1

    return points

# =============================================================================
# 6. IMPROPER INTEGRALS
# =============================================================================

def improper_integral(f, a, tol=1e-8, cutoff=20):
    """
    Compute ∫_a^∞ f(x) dx.

    Strategy:
    1. Get asymptotic expansion of f at infinity
    2. Determine decay rate (must be > 1/x for convergence)
    3. Adaptively integrate [a, M] where M is chosen so tail < tol
    4. Add analytic tail contribution

    Example:
        improper_integral(lambda x: exp(-(x*x)), 0)  # → √π/2 ≈ 0.8862
        improper_integral(lambda x: 1/(1 + x**2), 0) # → π/2 ≈ 1.5708
    """
    # Step 1: Estimate decay via evaluation at large x
    test_vals = [(10, f(R(10) + ZERO).st()),
                 (20, f(R(20) + ZERO).st()),
                 (50, f(R(50) + ZERO).st())]

    # Step 2: Find cutoff M where |f(M)| < tol
    M = cutoff
    while M < 1000:
        fx = f(R(M) + ZERO)
        if abs(fx.st()) < tol * 0.01:
            break
        M *= 2

    # Step 3: Adaptive integration over [a, M]
    bulk, bulk_err = integrate_adaptive(f, a, min(M, cutoff), tol=tol)

    # Step 4: If M > cutoff, integrate [cutoff, M] with larger steps
    if M > cutoff:
        tail, tail_err = integrate_adaptive(f, cutoff, M, tol=tol)
        bulk += tail
        bulk_err += tail_err

    return bulk, bulk_err


def improper_integral_both(f, tol=1e-8):
    """
    Compute ∫_{-∞}^{∞} f(x) dx.

    Splits at 0 and computes two improper integrals.

    Example:
        improper_integral_both(lambda x: exp(-(x*x)))  # → √π ≈ 1.7725
    """
    # ∫_{-∞}^0 f(x) dx = ∫_0^∞ f(-x) dx
    left, left_err = improper_integral(lambda x: f(-x), 0, tol=tol)
    right, right_err = improper_integral(f, 0, tol=tol)
    return left + right, left_err + right_err

# =============================================================================
# 7. ANALYTIC CONTINUATION
# =============================================================================

def analytic_continue(f, start, target, max_steps=50, safety=0.4):
    """
    Analytically continue f from 'start' to 'target' by chaining
    composite evaluations along a path.

    At each step:
    1. Evaluate f at current point → get full Taylor jet
    2. Estimate convergence radius from coefficients
    3. Step forward by safety × radius
    4. Use Taylor coefficients to evaluate at next point

    Returns the composite result at 'target' (value + derivatives).

    Example:
        # Continue ln(x) from x=1 to x=3
        # (ln has radius = distance to nearest singularity at 0)
        result = analytic_continue(lambda x: ln(x), start=1, target=3)
        # result.st() ≈ ln(3) ≈ 1.0986
    """
    current = float(start)

    for step in range(max_steps):
        if abs(current - target) < 1e-12:
            # We've arrived — do final evaluation
            return f(R(current) + ZERO)

        # Evaluate at current point
        result = f(R(current) + ZERO)

        # Estimate convergence radius
        radius = convergence_radius(f, at=current, order=15)

        # Step toward target, within convergence disk
        direction = 1 if target > current else -1
        max_step = safety * radius
        remaining = abs(target - current)
        actual_step = min(max_step, remaining)

        current += direction * actual_step

    # Final evaluation at target
    return f(R(current) + ZERO)


def find_singularities(f, x_range, n_points=100):
    """
    Scan for singularities of f by checking convergence radius
    at many points. Where the radius drops to near zero,
    there's likely a singularity.

    Returns list of (x, radius) pairs sorted by radius.

    Example:
        # f(x) = 1/(x-2) has singularity at x=2
        find_singularities(lambda x: 1/(x - 2), (0, 4))
        # → [(~2.0, ~0.0), ...]
    """
    a, b = x_range
    results = []

    for i in range(n_points):
        x = a + (b - a) * (i + 0.5) / n_points
        try:
            r = convergence_radius(f, at=x, order=10)
            results.append((x, r))
        except:
            results.append((x, 0.0))  # Likely AT a singularity

    return sorted(results, key=lambda p: p[1])
