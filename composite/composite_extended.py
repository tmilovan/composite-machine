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
composite_extended.py -- Extended Capabilities (FIXED v4)
=========================================================
Builds on composite_lib.py to add:
  1. Complex-valued composites (residues, poles)
  2. Asymptotic expansion extraction
  3. Convergence radius detection
  4. Composite ODE stepper
  5. Improper integrals via RK4 integration
  6. Analytic continuation

All capabilities reduce to the same mechanism:
  evaluate f on a composite number, read coefficients.

FIXES applied:
  1. cexp, csin, ccos always return Composite (never plain complex)
  2. residue() reads dimension +1 (not -1) -- h^{-1} = |1|_{+1}
  3. convergence_radius() uses generalized ratio + root test
  4. solve_ode() replaced with RK4 using Composite evaluation
  5. convergence_radius() combiner: monotonic detection + adaptive
     ratio-only for finite-radius series (fixes EX11 + EX12)
  6. improper_integral() uses solve_ode (RK4) instead of
     integrate_adaptive -- avoids ZERO overhead (fixes EX17/18 hang)
  7. exp() monkey-patched: split exp(a+h) = math.exp(a) * Taylor(h)
     Fixes catastrophic Taylor truncation for |x| > 6 (fixes EX17/18
     accuracy -- 15-term Taylor gives exp(-10) = 466, not 4.5e-5)

Author: Toni Milovan
"""

import math
import cmath
import composite.composite_lib as _clib
from composite.composite_lib import Composite, R, ZERO, INF, sin, cos, ln, sqrt
from composite.composite_lib import integrate_adaptive, antiderivative

# NOTE: we deliberately do NOT import exp from composite_lib here.
# Instead, we define _smart_exp below and monkey-patch composite_lib.exp.


# =============================================================================
# FIX 7: SMART EXP (monkey-patch composite_lib.exp)
# =============================================================================
#
# Problem: composite_lib.exp(x) uses Taylor series sum(x^n/n!, n=0..14).
#   For |x| > ~6, the alternating series hasn't converged with 15 terms:
#   exp(-10) via 15-term Taylor = 466 (true: 4.5e-5). Factor of 10^7 wrong.
#
# Fix: split exp(a + h) = math.exp(a) * exp(h)
#   - a = standard part (dim 0) -> use math.exp (IEEE 754, always accurate)
#   - h = infinitesimal part (dims != 0) -> Taylor series (converges fast
#     because h has no dim-0 component, so h^n shrinks by dimension)
#
# This is the standard "base + perturbation" splitting used in automatic
# differentiation (dual numbers, jets, etc.).

def _smart_exp(x, terms=15):
    """
    Composite-aware exp that uses math.exp for the scalar part.

    For Composite x = a + h where a = x.st() and h = infinitesimal part:
      exp(a + h) = math.exp(a) * sum(h^n / n!, n=0..terms-1)

    This avoids the catastrophic cancellation of the naive Taylor series
    sum(x^n / n!) for large |x|.
    """
    if isinstance(x, (int, float)):
        # Pure scalar: just use math.exp
        return Composite({0: math.exp(float(x))})

    if isinstance(x, Composite):
        a = x.st()  # Dimension-0 coefficient
        non_zero = {d: c for d, c in x.c.items() if d != 0 and abs(c) > 1e-15}

        if not non_zero:
            # Pure scalar Composite (only dim 0): use math.exp directly
            return Composite({0: math.exp(a)})

        # Split: exp(a + h) = exp(a) * exp(h)
        # exp(a) is exact via math.exp
        # exp(h) is Taylor on the SMALL infinitesimal part (converges fast)
        base = math.exp(a)
        h = Composite(non_zero)

        # Taylor series for exp(h): sum of h^n / n!
        exp_h = Composite({0: 1.0})  # n=0 term
        h_power = Composite({0: 1.0})
        for n in range(1, terms):
            h_power = h_power * h
            exp_h = exp_h + (1.0 / math.factorial(n)) * h_power

        return base * exp_h

    # Fallback for unexpected types
    return Composite({0: math.exp(float(x))})


# Monkey-patch composite_lib so ALL downstream code gets the fix.
# This includes test lambdas like `lambda x: exp(-x)` that capture
# composite_lib.exp at import time.
_clib.exp = _smart_exp

# Local reference for use within this module
exp = _smart_exp


# =============================================================================
# 1. COMPLEX COMPOSITES
# =============================================================================

def C(z):
    """Create a complex-valued real-dimension composite."""
    return Composite({0: complex(z)})


def C_var(z):
    """Complex variable with infinitesimal: z + h"""
    return Composite({0: complex(z), -1: 1.0})


# FIX 1: Always return Composite, never plain complex
def cexp(x, terms=15):
    """Complex exponential via Taylor series. Always returns Composite."""
    if isinstance(x, (int, float, complex)):
        x = Composite({0: complex(x)})
    result = Composite({})
    for n in range(terms):
        coeff = 1 / math.factorial(n)
        result = result + coeff * (x ** n)
    return result


def csin(x, terms=12):
    """Complex sine via Taylor series. Always returns Composite."""
    if isinstance(x, (int, float, complex)):
        x = Composite({0: complex(x)})
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n + 1)
        result = result + coeff * (x ** (2*n + 1))
    return result


def ccos(x, terms=12):
    """Complex cosine via Taylor series. Always returns Composite."""
    if isinstance(x, (int, float, complex)):
        x = Composite({0: complex(x)})
    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n)
        result = result + coeff * (x ** (2*n))
    return result


# =============================================================================
# 2. RESIDUE COMPUTATION
# =============================================================================

# FIX 2: Read dimension +1, not -1
def residue(f, at, terms=15):
    """
    Compute the residue of f(z) at z = a.

    Residue = coefficient of (z-a)^(-1) in the Laurent expansion
            = dimension +1 coefficient in composite evaluation.
    Why +1? Because h = |1|_{-1}, so h^{-1} = |1|_{+1}.
    The Laurent term b_1/(z-a) = b_1 * h^{-1} = |b_1|_{+1}.
    """
    if at == 0:
        z = ZERO
    else:
        z = Composite({0: complex(at), -1: 1.0})

    result = f(z)
    return result.coeff(1)


def pole_order(f, at, max_order=10):
    """
    Detect the order of a pole of f at z = a.

    Returns:
        n > 0: pole of order n
        n = 0: regular point (or removable singularity)
    """
    if at == 0:
        z = ZERO
    else:
        z = Composite({0: complex(at), -1: 1.0})

    result = f(z)

    max_pos = 0
    for dim, coeff in result.c.items():
        if abs(coeff) > 1e-12:
            if dim > max_pos:
                max_pos = dim

    if max_pos > 0:
        return max_pos
    return 0


def contour_integral(f, poles_inside, terms=15):
    """
    Compute a contour integral via the Residue Theorem:
    integral f(z) dz = 2*pi*i * sum Res(f, a_k)
    """
    total_residue = sum(residue(f, pole, terms) for pole in poles_inside)
    return 2j * cmath.pi * total_residue


# =============================================================================
# 3. ASYMPTOTIC EXPANSION
# =============================================================================

def asymptotic_expansion(f, order=5):
    """
    Extract asymptotic expansion of f(x) as x -> inf.
    Returns coefficients [a0, a1, a2, ...] where:
    f(x) ~ a0 + a1/x + a2/x^2 + ...
    """
    result = f(INF)
    coeffs = []
    for n in range(order + 1):
        coeffs.append(result.coeff(-n))
    return coeffs


def limit_at_infinity(f):
    """Compute lim(x->inf) f(x). Just the standard part of f(INF)."""
    return f(INF).st()


def asymptotic_order(f):
    """
    Determine the asymptotic growth order of f(x) as x -> inf.
    Returns the highest dimension with a nonzero coefficient.
    """
    result = f(INF)
    if not result.c:
        return float('-inf')
    return max(d for d, c in result.c.items() if abs(c) > 1e-12)


# =============================================================================
# 4. CONVERGENCE RADIUS DETECTION
# =============================================================================

# FIX 3 + FIX 5: Generalized ratio test + root test + adaptive combiner
def convergence_radius(f, at=0, order=20):
    """
    Estimate the convergence radius of the Taylor series of f around 'at'.

    Uses three complementary strategies:
    1. Generalized ratio test (handles gapped series like 1/(1+x^2))
    2. Root test / Cauchy-Hadamard (secondary estimate)
    3. Adaptive combiner:
       - Monotonically increasing ratios -> infinite radius (exp, sin, cos)
       - Decreasing ratios -> ratio estimate only (avoids root test bias)
       - Otherwise -> max(ratio, root) as lower bound
    """
    if at == 0:
        x = ZERO
    else:
        x = R(at) + ZERO

    result = f(x)

    # Extract Taylor coefficients: c_n = f^(n)(a)/n!
    coeffs = [abs(result.coeff(-n)) for n in range(order + 1)]

    # --- Generalized ratio test for gapped series ---
    nonzero = [(n, coeffs[n]) for n in range(order + 1) if coeffs[n] > 1e-15]

    ratios = []
    for i in range(len(nonzero) - 1):
        n1, c1 = nonzero[i]
        n2, c2 = nonzero[i + 1]
        gap = n2 - n1
        if gap > 0 and c2 > 1e-15:
            ratios.append((c1 / c2) ** (1.0 / gap))

    if not ratios:
        return float('inf')  # All coefficients zero -> polynomial

    # --- FIX 5a: Detect monotonically increasing ratios (entire function) ---
    if len(ratios) >= 3:
        is_monotonic = all(
            ratios[i + 1] > ratios[i] for i in range(len(ratios) - 1)
        )
        if is_monotonic:
            return float('inf')

    # Use the last few ratios (highest-order, most stable)
    stable_ratios = ratios[-5:] if len(ratios) >= 5 else ratios
    ratio_estimate = sum(stable_ratios) / len(stable_ratios)

    # --- Root test (Cauchy-Hadamard) as secondary estimate ---
    root_estimates = []
    for n, cn in nonzero:
        if n >= 2 and cn > 1e-15:
            root_estimates.append(1.0 / (cn ** (1.0 / n)))

    # --- FIX 5b: Adaptive combiner ---
    if root_estimates:
        stable_roots = root_estimates[-5:] if len(root_estimates) >= 5 else root_estimates
        root_estimate = sum(stable_roots) / len(stable_roots)
        if ratios[-1] < ratios[0]:
            return ratio_estimate   # Finite radius: ratio test is tighter
        return max(ratio_estimate, root_estimate)

    return ratio_estimate


def is_within_convergence(f, at, eval_point):
    """
    Check if eval_point is within the convergence radius of f's
    Taylor series centered at 'at'.
    Returns (bool, radius).
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
    Uses composite evaluation to get derivatives at once.
    Returns (y_new, error_estimate).
    """
    x_c = R(x0) + ZERO
    y_c = R(y0) + ZERO

    f_result = f(x_c, y_c)

    y_derivs = [y0]
    y_prime = f_result.st()
    y_derivs.append(y_prime)

    f_x = f_result.coeff(-1)
    y_c2 = R(y0) + ZERO
    x_c2 = R(x0)
    f_y_result = f(x_c2, y_c2)
    f_y = f_y_result.coeff(-1)

    y_double_prime = f_x + f_y * y_prime
    y_derivs.append(y_double_prime)

    y_new = y0
    for n in range(1, min(order, len(y_derivs))):
        y_new += y_derivs[n] * h**n / math.factorial(n)

    if len(y_derivs) > 1:
        err = abs(y_derivs[-1] * h**len(y_derivs) / math.factorial(len(y_derivs)))
    else:
        err = 0.0

    return y_new, err


# FIX 4: RK4 with Composite evaluation (replaces broken forward Euler)
def _eval_ode_composite(f, x_val, y_val):
    """
    Evaluate ODE right-hand side f(x, y) using Composite numbers.
    Returns the standard part (float).
    """
    result = f(R(x_val), R(y_val))
    if isinstance(result, Composite):
        return result.st()
    return float(result)


def solve_ode(f, x_range, y0, steps=1000):
    """
    Solve y' = f(x, y) over x_range = (a, b) with y(a) = y0.

    Uses classical RK4 with Composite evaluation at each stage.
    Every f evaluation goes through composite arithmetic via R().

    Returns list of (x, y) pairs.
    """
    a, b = x_range
    h = (b - a) / steps
    x = a
    y = float(y0)
    points = [(x, y)]

    for _ in range(steps):
        k1 = _eval_ode_composite(f, x, y)
        k2 = _eval_ode_composite(f, x + h/2, y + h*k1/2)
        k3 = _eval_ode_composite(f, x + h/2, y + h*k2/2)
        k4 = _eval_ode_composite(f, x + h, y + h*k3)

        y = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        points.append((x, y))

    return points


# =============================================================================
# 6. IMPROPER INTEGRALS
# =============================================================================

# FIX 6: Use solve_ode (RK4) instead of integrate_adaptive
# FIX 7: exp monkey-patch makes this accurate for all arguments

def improper_integral(f, a, tol=1e-8, cutoff=20):
    """
    Compute integral from a to inf of f(x) dx.

    Strategy: integrate [a, cutoff] via RK4 (solve_ode).
    Uses Composite evaluation through R() only (no ZERO), avoiding
    the performance trap of integrate_adaptive.

    Accuracy depends on FIX 7 (smart exp): without the math.exp
    splitting, Taylor-series exp gives garbage for |x| > ~6.

    For well-behaved integrands (exponential decay, algebraic decay
    faster than 1/x), the tail beyond cutoff is negligible.
    """
    points = solve_ode(
        lambda x, y: f(x),
        (a, cutoff),
        y0=0,
        steps=2000
    )
    result = points[-1][1]

    # Estimate tail contribution for error bound
    try:
        tail_val = abs(f(R(float(cutoff))).st())
    except:
        tail_val = 0.0

    return result, tail_val


def improper_integral_both(f, tol=1e-8):
    """
    Compute integral from -inf to inf of f(x) dx.
    Splits at 0 and computes two improper integrals.
    """
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
    Returns the composite result at 'target' (value + derivatives).
    """
    current = float(start)

    for step in range(max_steps):
        if abs(current - target) < 1e-12:
            return f(R(current) + ZERO)

        result = f(R(current) + ZERO)
        radius = convergence_radius(f, at=current, order=15)

        direction = 1 if target > current else -1
        max_step = safety * radius
        remaining = abs(target - current)
        actual_step = min(max_step, remaining)

        current += direction * actual_step

    return f(R(current) + ZERO)


def find_singularities(f, x_range, n_points=100):
    """
    Scan for singularities of f by checking convergence radius
    at many points. Where the radius drops to near zero,
    there's likely a singularity.
    Returns list of (x, radius) pairs sorted by radius.
    """
    a, b = x_range
    results = []

    for i in range(n_points):
        x = a + (b - a) * (i + 0.5) / n_points
        try:
            r = convergence_radius(f, at=x, order=10)
            results.append((x, r))
        except:
            results.append((x, 0.0))

    return sorted(results, key=lambda p: p[1])
