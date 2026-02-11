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
composite_vector.py — Vector Calculus Extensions (FIXED v2)
===========================================================
Extends composite_multivar.py with complete vector calculus operations:
  - Triple integrals over 3D regions
  - Line integrals (scalar and vector fields)
  - Surface integrals (scalar and vector fields / flux)

Integration strategy:
  - Inner z-axis of triple integrals: composite adaptive integration
    (real MC infinitesimal seeds → dimensional-shift antiderivative)
  - Line integrals: midpoint quadrature with composite curve derivatives
    (exact via .d(1) when curves return Composites, numerical fallback)
  - Surface integrals: midpoint quadrature for both u and v axes

Why not integrate_adaptive everywhere?
  integrate_adaptive determines step size from dim < -1 coefficients.
  When the integrand is a numerically-evaluated wrapper returning only
  Composite({0: value}), there are no higher-order terms → the step size
  blows up to the full interval → evaluates only at one endpoint → wrong.
  Only the inner z-loop of triple_integral has real composite structure
  (MC infinitesimal seeds) that integrate_adaptive can exploit.

Parametric curves accept both composite-compatible functions
(sin, cos from composite_lib) and standard math.* functions.

Requires: composite_lib.py, composite_multivar.py, composite_extended.py

Author: Toni Milovan
License: AGPL
"""

import math
from typing import Callable, List
from composite.composite_multivar import MC, RR, RR_const
from composite.composite_lib import Composite, R, ZERO, integrate_adaptive

# Import composite_extended to activate the smart exp monkey-patch.
# This ensures exp() works correctly for large arguments in any
# integrand that uses exponential functions.
import composite.composite_extended as _cext


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _to_float(val):
    """Convert a value to float. Handles Composite, MC, and plain numbers."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, Composite):
        return float(val.st())
    if isinstance(val, MC):
        return float(val.st())
    return float(val)


def _try_composite_curve(curve, t_comp):
    """
    Try to evaluate curve with a Composite t.

    If the curve uses composite-compatible functions, this returns
    a list of Composite components with derivative information.
    If it fails (e.g. math.cos silently converts via __float__),
    returns None.

    Safety: ALL components must be Composite. This prevents the
    silent-float-conversion pitfall where math.cos(Composite)
    succeeds via __float__ but loses derivative information.
    """
    try:
        point = curve(t_comp)
        if all(isinstance(p, Composite) for p in point):
            return point
    except (TypeError, AttributeError):
        pass
    return None


def _curve_eval(curve, t_val):
    """
    Evaluate curve at t, returning (positions, velocities).

    Strategy:
    1. Try composite evaluation (exact derivatives via .d(1))
       by passing R(t_val) + ZERO to the curve function.
    2. Fall back to numerical differentiation if curve uses
       math.* functions that silently lose derivative info.

    Composite path works for curves like:
        lambda t: [3*t, 4*t]           # int * Composite = Composite
        lambda t: [sin(t), cos(t)]     # composite_lib trig → Composite

    Numerical fallback for curves like:
        lambda t: [math.cos(t), math.sin(t)]  # __float__ loses derivatives

    Args:
        curve: parametric curve function
        t_val: float parameter value

    Returns:
        (positions, velocities) as lists of floats
    """
    # Try composite evaluation for exact derivatives
    t_comp = R(t_val) + ZERO
    point = _try_composite_curve(curve, t_comp)
    if point is not None:
        positions = [p.st() for p in point]
        velocities = [p.d(1) for p in point]
        return positions, velocities

    # Fallback: numerical differentiation
    point = curve(t_val)
    positions = [float(p) for p in point]
    eps = 1e-8
    point_plus = curve(t_val + eps)
    velocities = [(float(point_plus[i]) - positions[i]) / eps
                  for i in range(len(positions))]
    return positions, velocities


def _surface_normal(surface, u_val, v_val):
    """
    Compute the (unnormalized) normal vector to a parametric surface
    via numerical partial derivatives: normal = r_u × r_v.

    Returns (point, normal_vector) as lists of floats.
    """
    point = surface(u_val, v_val)
    eps = 1e-6
    point_u = surface(u_val + eps, v_val)
    point_v = surface(u_val, v_val + eps)

    ru = [(float(point_u[i]) - float(point[i])) / eps for i in range(3)]
    rv = [(float(point_v[i]) - float(point[i])) / eps for i in range(3)]

    normal = [
        ru[1]*rv[2] - ru[2]*rv[1],
        ru[2]*rv[0] - ru[0]*rv[2],
        ru[0]*rv[1] - ru[1]*rv[0]
    ]

    return [float(p) for p in point], normal


# =============================================================================
# TRIPLE INTEGRALS
# =============================================================================

def triple_integral(f, x_range, y_range, z_range, tol=1e-8):
    """
    Compute ∫∫∫ f(x,y,z) dz dy dx by iterated integration.

    Innermost z-axis uses composite adaptive integration (with MC
    infinitesimal seed → dimensional-shift antiderivative).
    Outer y and x axes use midpoint quadrature.

    Args:
        f: function f(x, y, z) accepting MC or float arguments
        x_range, y_range, z_range: (lower, upper) bounds
        tol: integration tolerance for inner z-axis

    Example:
        triple_integral(lambda x,y,z: 1, (0,1), (0,1), (0,1))  # → 1.0
        triple_integral(lambda x,y,z: x*y*z, (0,1), (0,1), (0,1))  # → 0.125
    """
    xa, xb = x_range
    ya, yb = y_range
    za, zb = z_range

    def inner_z(x_val, y_val):
        """Integrate over z for fixed (x, y) via composite adaptive."""
        def g(z_comp):
            x_mc = RR_const(x_val, nvars=3)
            y_mc = RR_const(y_val, nvars=3)
            # Build z as MC with infinitesimal seed in z-direction
            z_mc = MC(
                {(0,0,0): z_comp.st(), (0,0,-1): z_comp.coeff(-1)},
                nvars=3
            )
            result_mc = f(x_mc, y_mc, z_mc)
            # Handle plain numbers
            if isinstance(result_mc, (int, float)):
                return Composite({0: float(result_mc)})
            elif isinstance(result_mc, MC):
                # Project onto z-axis: keep only (0, 0, dz) terms
                out = Composite({})
                for dim, coeff in result_mc.c.items():
                    if dim[0] == 0 and dim[1] == 0:
                        out.c[dim[2]] = out.c.get(dim[2], 0) + coeff
                return out if out.c else Composite({0: 0.0})
            return result_mc

        val, _ = integrate_adaptive(g, za, zb, tol=tol)
        return _to_float(val)

    def inner_y(x_val):
        """Integrate over y for fixed x via midpoint quadrature."""
        total = 0.0
        n_steps = max(20, int((yb - ya) / 0.05))
        dy = (yb - ya) / n_steps
        for i in range(n_steps):
            yi = ya + (i + 0.5) * dy
            total += inner_z(x_val, yi) * dy
        return total

    # Outer integration over x via midpoint quadrature
    total = 0.0
    n_steps = max(20, int((xb - xa) / 0.05))
    dx = (xb - xa) / n_steps
    for i in range(n_steps):
        xi = xa + (i + 0.5) * dx
        total += inner_y(xi) * dx

    return total


# =============================================================================
# LINE INTEGRALS
# =============================================================================

def line_integral_scalar(f, curve, t_range, tol=1e-8):
    """
    Compute line integral of scalar field f along a parametric curve.

    ∫_C f(x,y,...) ds = ∫_a^b f(r(t)) |r'(t)| dt

    Uses midpoint quadrature over t.
    Curve derivatives use composite evaluation when possible
    (exact via .d(1)), with automatic fallback to numerical
    differentiation for math.* parametrizations.

    Args:
        f: scalar function f(x, y) or f(x, y, z)
        curve: parametric curve returning [x(t), y(t), ...]
               For exact derivatives, use composite_lib trig functions.
        t_range: (t_start, t_end)

    Example:
        # Arc length of (0,0) → (3,4)
        line_integral_scalar(lambda x,y: 1, lambda t: [3*t, 4*t], (0,1))
        # → 5.0

        # Circumference of unit circle
        line_integral_scalar(
            lambda x,y: 1,
            lambda t: [math.cos(t), math.sin(t)],
            (0, 2*math.pi)
        )
        # → 2π  (uses numerical fallback for math.cos/sin)
    """
    t_start, t_end = t_range
    n_steps = max(100, int((t_end - t_start) / 0.01))
    dt = (t_end - t_start) / n_steps
    total = 0.0

    for i in range(n_steps):
        t_mid = t_start + (i + 0.5) * dt
        positions, velocities = _curve_eval(curve, t_mid)
        speed = math.sqrt(sum(v**2 for v in velocities))
        f_val = _to_float(f(*positions))
        total += f_val * speed * dt

    return total


def line_integral_vector(F, curve, t_range, tol=1e-8):
    """
    Compute line integral of vector field F along a parametric curve.

    ∫_C F · dr = ∫_a^b F(r(t)) · r'(t) dt

    Uses midpoint quadrature over t.

    Note: this computes the dot product F · dr/dt (not F · ds).
    For conservative fields around closed loops, the result is zero.

    Args:
        F: vector field [Fx, Fy, ...] as list of callables
        curve: parametric curve returning [x(t), y(t), ...]
        t_range: (t_start, t_end)

    Example:
        # Circulation of rotation field around unit circle
        line_integral_vector(
            [lambda x,y: -y, lambda x,y: x],
            lambda t: [math.cos(t), math.sin(t)],
            (0, 2*math.pi)
        )
        # → 2π
    """
    t_start, t_end = t_range
    n_steps = max(100, int((t_end - t_start) / 0.01))
    dt = (t_end - t_start) / n_steps
    total = 0.0

    for i in range(n_steps):
        t_mid = t_start + (i + 0.5) * dt
        positions, velocities = _curve_eval(curve, t_mid)
        F_vals = [_to_float(Fi(*positions)) for Fi in F]
        dot = sum(F_vals[j] * velocities[j] for j in range(len(F_vals)))
        total += dot * dt

    return total


# =============================================================================
# SURFACE INTEGRALS
# =============================================================================

def surface_integral_scalar(f, surface, u_range, v_range, tol=1e-6):
    """
    Compute surface integral of scalar field over parametric surface.

    ∫∫_S f(x,y,z) dS where surface(u,v) = [x(u,v), y(u,v), z(u,v)]

    Uses midpoint quadrature for both u and v axes.
    Surface normals computed via numerical partials (cross product).

    Args:
        f: scalar function f(x, y, z)
        surface: parametric surface returning [x, y, z]
        u_range, v_range: parameter bounds

    Example:
        # Surface area of unit sphere
        surface_integral_scalar(
            lambda x,y,z: 1,
            lambda u,v: [math.sin(u)*math.cos(v),
                         math.sin(u)*math.sin(v),
                         math.cos(u)],
            (0, math.pi), (0, 2*math.pi)
        )
        # → 4π ≈ 12.566
    """
    ua, ub = u_range
    va, vb = v_range
    n_u = max(30, int((ub - ua) / 0.1))
    n_v = max(30, int((vb - va) / 0.15))
    du = (ub - ua) / n_u
    dv = (vb - va) / n_v

    total = 0.0
    for i in range(n_u):
        ui = ua + (i + 0.5) * du
        for j in range(n_v):
            vj = va + (j + 0.5) * dv
            point, normal = _surface_normal(surface, ui, vj)
            dS = math.sqrt(sum(c**2 for c in normal))
            f_val = _to_float(f(*point))
            total += f_val * dS * du * dv

    return total


def surface_integral_vector(F, surface, u_range, v_range, tol=1e-6):
    """
    Compute flux of vector field F through parametric surface.

    ∫∫_S F · dS = ∫∫ F · (r_u × r_v) du dv

    Uses midpoint quadrature for both u and v axes.

    Args:
        F: vector field [Fx, Fy, Fz] as list of callables
        surface: parametric surface returning [x, y, z]
        u_range, v_range: parameter bounds

    Example:
        # Flux of F = [x,y,z] through unit sphere (divergence theorem: 4π)
        surface_integral_vector(
            [lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z],
            lambda u,v: [math.sin(u)*math.cos(v),
                         math.sin(u)*math.sin(v),
                         math.cos(u)],
            (0, math.pi), (0, 2*math.pi)
        )
        # → 4π ≈ 12.566
    """
    ua, ub = u_range
    va, vb = v_range
    n_u = max(30, int((ub - ua) / 0.1))
    n_v = max(30, int((vb - va) / 0.15))
    du = (ub - ua) / n_u
    dv = (vb - va) / n_v

    total = 0.0
    for i in range(n_u):
        ui = ua + (i + 0.5) * du
        for j in range(n_v):
            vj = va + (j + 0.5) * dv
            point, normal = _surface_normal(surface, ui, vj)
            F_vals = [_to_float(Fi(*point)) for Fi in F]
            flux = sum(F_vals[k] * normal[k] for k in range(3))
            total += flux * du * dv

    return total
