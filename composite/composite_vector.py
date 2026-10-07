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
composite_vector.py -- Vector Calculus Extensions (v4, directional composites)
==============================================================================
Vector calculus on the library's composite integrators:
  - Triple integrals over 3D boxes
  - Line integrals (scalar and vector fields)
  - Surface integrals (scalar and vector fields / flux)

v4 change (2026-10-06): off MC.  The integrals are the library's
(composite_lib.integrate): line integrals meet composites along the curve,
surface integrals use the 2D meet on merged composites separated from parts
(integrate_surface), and triple integrals go through integrate()'s 3D box path.
Surface partials (_surface_eval) come from directional composites
(composite_multivar.jacobian_at) instead of MC's RR seeds
(MC is parked in composite_multivar_mc.py).

The midpoint quadrature and the finite-difference fallbacks are gone: a curve or
surface written with math.* turns the seed into a float, and that is refused
(ValueError) rather than differentiated numerically.  Write curves and surfaces
with the composite_lib functions (sin, cos, exp, ...).

The tol defaults are the library's (1e-10).  Before v4 the functions took
tol=1e-8 / 1e-6 and never read it.

Requires: composite_lib.py, composite_multivar.py, composite_extended.py

Author: Toni Milovan
License: AGPL
"""

import math
from typing import Callable, List
from composite.composite_lib import Composite, integrate, integrate_surface
from composite.composite_multivar import jacobian_at

# Import composite_extended to activate the smart exp monkey-patch.
# This ensures exp() works correctly for large arguments in any
# integrand that uses exponential functions.
import composite.composite_extended as _cext


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _to_float(val):
    """Convert a value to float. Handles Composite and plain numbers."""
    if isinstance(val, Composite):
        return float(val.st())
    return float(val)


def _surface_eval(surface, u_val, v_val):
    """
    Evaluate surface at (u, v), returning (positions, r_u, r_v) as floats.

    r_u and r_v are read from directional composites: each component's
    gradient in (u, v), two composites per component.  A math.* surface is
    refused (ValueError), not differentiated numerically.
    """
    J = jacobian_at([lambda u, v, k=k: surface(u, v)[k] for k in range(3)],
                         [u_val, v_val])
    positions = [_to_float(p) for p in surface(u_val, v_val)]
    return positions, [row[0] for row in J], [row[1] for row in J]


def _surface_normal(surface, u_val, v_val):
    """
    The (unnormalized) normal r_u x r_v of a parametric surface at (u, v).

    Returns (point, normal_vector) as lists of floats.
    """
    positions, r_u, r_v = _surface_eval(surface, u_val, v_val)
    normal = [
        r_u[1]*r_v[2] - r_u[2]*r_v[1],
        r_u[2]*r_v[0] - r_u[0]*r_v[2],
        r_u[0]*r_v[1] - r_u[1]*r_v[0]
    ]
    return positions, normal


# =============================================================================
# TRIPLE INTEGRALS
# =============================================================================

def triple_integral(f, x_range, y_range, z_range, tol=1e-10):
    """
    Integral of f(x, y, z) over a box, via integrate()'s 3D path.

    Example:
        triple_integral(lambda x,y,z: x*y*z, (0,1), (0,1), (0,1))  # -> 0.125
    """
    return _to_float(integrate(f, x_range, y_range, z_range, tol=tol))


# =============================================================================
# LINE INTEGRALS
# =============================================================================

def line_integral_scalar(f, curve, t_range, tol=1e-10):
    """
    Integral of a scalar field along a parametric curve, f(r(t)) |r'(t)| dt.

    The curve must be composite (composite_lib sin, cos, ...).

    Example:
        from composite.composite_lib import sin, cos
        line_integral_scalar(lambda x,y: 1, lambda t: [cos(t), sin(t)],
                             (0, 2*math.pi))   # -> 2 pi
    """
    return _to_float(integrate(f, t_range, curve=curve, tol=tol))


def line_integral_vector(F, curve, t_range, tol=1e-10):
    """
    Integral of a vector field along a parametric curve, F(r(t)) . r'(t) dt.

    Example:
        from composite.composite_lib import sin, cos
        line_integral_vector([lambda x,y: -y, lambda x,y: x],
                             lambda t: [cos(t), sin(t)], (0, 2*math.pi))  # -> 2 pi
    """
    return _to_float(integrate(F, t_range, curve=curve, tol=tol))


# =============================================================================
# SURFACE INTEGRALS
# =============================================================================

def surface_integral_scalar(f, surface, u_range, v_range, tol=1e-10):
    """
    Integral of a scalar field over a parametric surface, f |r_u x r_v| du dv,
    on the 2D meet (integrate_surface).  The surface must be composite.

    Example:
        from composite.composite_lib import sin, cos
        surface_integral_scalar(lambda x,y,z: 1,
                                lambda u,v: [cos(v), sin(v), u],
                                (0, 1), (0, 2*math.pi))   # cylinder: 2 pi
    """
    return _to_float(integrate_surface(f, (u_range, v_range), surface, tol=tol))


def surface_integral_vector(F, surface, u_range, v_range, tol=1e-10):
    """
    Flux of a vector field through a parametric surface, F . (r_u x r_v) du dv,
    on the 2D meet (integrate_surface).

    Example:
        from composite.composite_lib import sin, cos
        surface_integral_vector(
            [lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z],
            lambda u,v: [sin(u)*cos(v), sin(u)*sin(v), cos(u)],
            (0, math.pi), (0, 2*math.pi))   # divergence theorem: 4 pi
    """
    return _to_float(integrate_surface(F, (u_range, v_range), surface, tol=tol))
