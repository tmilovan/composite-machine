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
test_multivar_extended.py — Combined Import-Based Test Suite
=============================================================
Comprehensive tests for both:
  - composite_multivar.py (30 tests: MV01-MV30)
  - composite_extended.py (20 tests: EX01-EX20)

Libraries are IMPORTED, not included.
Uses only MC class methods (RR, .partial(), .gradient(), .hessian(),
.laplacian(), .divergence_of()) — no high-level API wrappers required.

Requires on Python path:
  - composite_lib.py      (fixed version)
  - composite_multivar.py  (fixed version)
  - composite_extended.py  (fixed version)

Run:
  python test_multivar_extended.py

Author: Toni Milovan
License: AGPL
"""

import math
import cmath
import sys
import time

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from composite.composite_multivar import (
        MC, RR, RR_const,
        mc_sin, mc_cos, mc_exp, mc_ln, mc_sqrt, mc_tan, mc_power,
    )
    MULTIVAR_OK = True
    print("✅ composite_multivar imported successfully")
except ImportError as e:
    MULTIVAR_OK = False
    print(f"❌ Could not import composite_multivar: {e}")

try:
    from composite.composite_extended import (
        C, C_var,
        cexp, csin, ccos,
        residue, pole_order, contour_integral,
        asymptotic_expansion, limit_at_infinity, asymptotic_order,
        convergence_radius, is_within_convergence,
        ode_step, solve_ode,
        improper_integral, improper_integral_both,
        analytic_continue, find_singularities,
    )
    from composite.composite_lib import R, ZERO, INF, exp, ln, sin, cos, Composite
    EXTENDED_OK = True
    print("✅ composite_extended imported successfully")
except ImportError as e:
    EXTENDED_OK = False
    print(f"❌ Could not import composite_extended: {e}")


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def check(self, name, got, expected, tol=1e-8):
        """Check scalar value."""
        if isinstance(expected, complex):
            ok = abs(got - expected) < tol
        else:
            ok = abs(float(got) - float(expected)) < tol
        status = "✅" if ok else "❌"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((name, ok))
        got_s = f"{got:.10g}" if not isinstance(got, complex) else f"{got}"
        exp_s = f"{expected:.10g}" if not isinstance(expected, complex) else f"{expected}"
        print(f"  {status} {name}: got {got_s}, expected {exp_s}")
        return ok

    def check_list(self, name, got, expected, tol=1e-8):
        """Check list of scalars."""
        ok = len(got) == len(expected) and all(
            abs(float(g) - float(e)) < tol for g, e in zip(got, expected)
        )
        status = "✅" if ok else "❌"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((name, ok))
        got_s = [f"{g:.6g}" for g in got]
        exp_s = [f"{e:.6g}" for e in expected]
        print(f"  {status} {name}: got {got_s}, expected {exp_s}")
        return ok

    def check_matrix(self, name, got, expected, tol=1e-8):
        """Check matrix (list of lists)."""
        flat_got = [v for row in got for v in row]
        flat_exp = [v for row in expected for v in row]
        ok = len(flat_got) == len(flat_exp) and all(
            abs(float(g) - float(e)) < tol for g, e in zip(flat_got, flat_exp)
        )
        status = "✅" if ok else "❌"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((name, ok))
        print(f"  {status} {name}")
        if not ok:
            for row in got:
                print(f"       got: [{', '.join(f'{v:.6g}' for v in row)}]")
            for row in expected:
                print(f"       exp: [{', '.join(f'{v:.6g}' for v in row)}]")
        return ok

    def check_true(self, name, condition, detail=""):
        """Check boolean condition."""
        status = "✅" if condition else "❌"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((name, condition))
        extra = f" ({detail})" if detail and not condition else ""
        print(f"  {status} {name}{extra}")
        return condition

    def skip(self, name, reason):
        self.skipped += 1
        self.results.append((name, None))
        print(f"  ⏭️  {name}: SKIPPED ({reason})")

    def summary(self, label, total_expected):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"{label}: {self.passed}/{total} passed", end="")
        if self.skipped:
            print(f", {self.skipped} skipped", end="")
        if self.failed:
            print(f", {self.failed} FAILED", end="")
        print(f"\n{'='*60}")
        for name, ok in self.results:
            if ok is None:
                print(f"  ⏭️  {name}")
            else:
                print(f"  {'✅' if ok else '❌'} {name}")
        return self.failed == 0


# =============================================================================
# HELPERS: inline equivalents of high-level API (uses only MC class methods)
# =============================================================================

def _eval_f(f, at):
    """Evaluate f on RR composite args and return the MC result."""
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    return f(*args)


def _jacobian_at(fs, at):
    """Compute Jacobian matrix using MC.gradient() on each component."""
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    return [f(*args).gradient() for f in fs]


def _curl_at(F, at):
    """Compute curl of 3D vector field using RR + .partial()."""
    nvars = 3
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    Fx = F[0](*args)
    Fy = F[1](*args)
    Fz = F[2](*args)
    curl_x = Fz.partial(0, 1, 0) - Fy.partial(0, 0, 1)
    curl_y = Fx.partial(0, 0, 1) - Fz.partial(1, 0, 0)
    curl_z = Fy.partial(1, 0, 0) - Fx.partial(0, 1, 0)
    return [curl_x, curl_y, curl_z]


def _multivar_limit(f, as_vars_to):
    """Compute multivar limit using MC.zero_var + RR + .st()."""
    nvars = len(as_vars_to)
    args = []
    for i, val in enumerate(as_vars_to):
        if val == 0:
            args.append(MC.zero_var(i, nvars))
        else:
            args.append(RR(val, var=i, nvars=nvars))
    return f(*args).st()


# =============================================================================
# PART 1: MULTI-VARIABLE TESTS (MV01-MV30)
# =============================================================================

def run_multivar_tests(t: TestRunner):
    if not MULTIVAR_OK:
        for i in range(1, 31):
            t.skip(f"MV{i:02d}", "composite_multivar not available")
        return

    print("\n" + "="*60)
    print("PART 1: MULTI-VARIABLE CALCULUS — 30 Tests")
    print("="*60)

    # -------------------------------------------------------------------------
    print("\n--- Partial Derivatives (MV01-MV05) ---")
    # -------------------------------------------------------------------------

    # MV01: f(x,y) = x²y at (3,2) → ∂f/∂x = 2xy = 12
    r = _eval_f(lambda x, y: x**2 * y, [3, 2])
    t.check("MV01 ∂/∂x[x²y] at (3,2)", r.partial(1, 0), 12.0)

    # MV02: f(x,y) = x²y at (3,2) → ∂f/∂y = x² = 9
    t.check("MV02 ∂/∂y[x²y] at (3,2)", r.partial(0, 1), 9.0)

    # MV03: f(x,y) = x²y at (3,2) → ∂²f/∂x∂y = 2x = 6
    t.check("MV03 ∂²/∂x∂y[x²y] at (3,2)", r.partial(1, 1), 6.0)

    # MV04: f(x,y) = x³y² at (2,1) → ∂²f/∂x² = 6xy² = 12
    r4 = _eval_f(lambda x, y: x**3 * y**2, [2, 1])
    t.check("MV04 ∂²/∂x²[x³y²] at (2,1)", r4.partial(2, 0), 12.0)

    # MV05: f(x,y) = x⁴ + y⁴ at (1,1) → ∂⁴f/∂x²∂y² = 0
    r5 = _eval_f(lambda x, y: x**4 + y**4, [1, 1])
    t.check("MV05 ∂⁴/∂x²∂y²[x⁴+y⁴] at (1,1)", r5.partial(2, 2), 0.0)

    # -------------------------------------------------------------------------
    print("\n--- Gradients (MV06-MV08) ---")
    # -------------------------------------------------------------------------

    # MV06: f(x,y) = x² + y² at (3,4) → ∇f = [6, 8]
    r6 = _eval_f(lambda x, y: x**2 + y**2, [3, 4])
    t.check_list("MV06 ∇[x²+y²] at (3,4)", r6.gradient(), [6.0, 8.0])

    # MV07: f(x,y) = x·y at (5,7) → ∇f = [7, 5]
    r7 = _eval_f(lambda x, y: x * y, [5, 7])
    t.check_list("MV07 ∇[xy] at (5,7)", r7.gradient(), [7.0, 5.0])

    # MV08: f(x,y,z) = xyz at (2,3,4) → ∇f = [12, 8, 6]
    r8 = _eval_f(lambda x, y, z: x * y * z, [2, 3, 4])
    t.check_list("MV08 ∇[xyz] at (2,3,4)", r8.gradient(), [12.0, 8.0, 6.0])

    # -------------------------------------------------------------------------
    print("\n--- Hessians (MV09-MV11) ---")
    # -------------------------------------------------------------------------

    # MV09: f(x,y) = x²+y² at (1,1) → H = [[2,0],[0,2]]
    r9 = _eval_f(lambda x, y: x**2 + y**2, [1, 1])
    t.check_matrix("MV09 H[x²+y²] at (1,1)", r9.hessian(), [[2, 0], [0, 2]])

    # MV10: f(x,y) = x²y+y³ at (1,2) → H = [[4,2],[2,12]]
    r10 = _eval_f(lambda x, y: x**2 * y + y**3, [1, 2])
    t.check_matrix("MV10 H[x²y+y³] at (1,2)", r10.hessian(), [[4, 2], [2, 12]])

    # MV11: f(x,y) = x³y³ at (1,1) → H = [[6,9],[9,6]]
    r11 = _eval_f(lambda x, y: x**3 * y**3, [1, 1])
    t.check_matrix("MV11 H[x³y³] at (1,1)", r11.hessian(), [[6, 9], [9, 6]])

    # -------------------------------------------------------------------------
    print("\n--- Jacobians (MV12-MV14) ---")
    # -------------------------------------------------------------------------

    # MV12: F(x,y) = [x²+y, xy²] at (1,2) → J = [[2,1],[4,4]]
    t.check_matrix("MV12 J[x²+y, xy²] at (1,2)",
                   _jacobian_at(
                       [lambda x, y: x**2 + y, lambda x, y: x * y**2],
                       [1, 2]),
                   [[2, 1], [4, 4]])

    # MV13: F(x,y) = [x+y, x-y] at (3,5) → J = [[1,1],[1,-1]]
    t.check_matrix("MV13 J[x+y, x-y] at (3,5)",
                   _jacobian_at(
                       [lambda x, y: x + y, lambda x, y: x - y],
                       [3, 5]),
                   [[1, 1], [1, -1]])

    # MV14: Polar transform F(r,θ) = [r·cos(θ), r·sin(θ)] at (2, π/4)
    theta = math.pi / 4
    s2 = math.sqrt(2) / 2
    t.check_matrix("MV14 J[r·cosθ, r·sinθ] at (2,π/4)",
                   _jacobian_at(
                       [lambda r, th: r * mc_cos(th),
                        lambda r, th: r * mc_sin(th)],
                       [2, theta]),
                   [[s2, -2 * s2], [s2, 2 * s2]],
                   tol=1e-6)

    # -------------------------------------------------------------------------
    print("\n--- Laplacians (MV15-MV16) ---")
    # -------------------------------------------------------------------------

    # MV15: f(x,y) = x² + y² → ∇²f = 4
    r15 = _eval_f(lambda x, y: x**2 + y**2, [1, 1])
    t.check("MV15 ∇²[x²+y²] at (1,1)", r15.laplacian(), 4.0)

    # MV16: f(x,y) = x³y + xy³ at (2,3) → ∇²f = 6xy + 6xy = 72
    r16 = _eval_f(lambda x, y: x**3 * y + x * y**3, [2, 3])
    t.check("MV16 ∇²[x³y+xy³] at (2,3)", r16.laplacian(), 72.0)

    # -------------------------------------------------------------------------
    print("\n--- Multi-Variable Limits (MV17-MV18) ---")
    # -------------------------------------------------------------------------

    # MV17: lim (x²+y²)/(x²+y²) → (0,0) = 1
    t.check("MV17 lim (x²+y²)/(x²+y²) → (0,0)",
            _multivar_limit(lambda x, y: (x**2 + y**2) / (x**2 + y**2), [0, 0]),
            1.0)

    # MV18: lim (x²y-2x²)/(y-2) → (1,2) = x² = 1
    t.check("MV18 lim (x²y-2x²)/(y-2) → (1,2)",
            _multivar_limit(
                lambda x, y: (x**2 * y - 2 * x**2) / (y - 2),
                [1, 2]),
            1.0)

    # -------------------------------------------------------------------------
    print("\n--- Transcendental Compositions (MV19-MV20) ---")
    # -------------------------------------------------------------------------

    # MV19: ∂/∂x[sin(x)cos(y)] at (π/6,π/3) = cos(π/6)cos(π/3) = √3/4
    r19 = _eval_f(lambda x, y: mc_sin(x) * mc_cos(y),
                  [math.pi / 6, math.pi / 3])
    t.check("MV19 ∂/∂x[sin(x)cos(y)] at (π/6,π/3)",
            r19.partial(1, 0), math.sqrt(3) / 4, tol=1e-6)

    # MV20: ∂²/∂x∂y[exp(xy)] at (1,1) = (1+xy)e^(xy) = 2e
    r20 = _eval_f(lambda x, y: mc_exp(x * y), [1, 1])
    t.check("MV20 ∂²/∂x∂y[exp(xy)] at (1,1)",
            r20.partial(1, 1), 2 * math.e, tol=1e-5)

    # -------------------------------------------------------------------------
    print("\n--- Divergence (MV21-MV23) ---")
    # -------------------------------------------------------------------------

    # MV21: div(F) where F = [x², y²] at (3,4) → 2x + 2y = 14
    hx = MC.zero_var(0, nvars=2)
    hy = MC.zero_var(1, nvars=2)
    x21 = MC.real(3, nvars=2) + hx
    y21 = MC.real(4, nvars=2) + hy
    t.check("MV21 div[x², y²] at (3,4)",
            MC.divergence_of([x21**2, y21**2]), 14.0)

    # MV22: div(F) where F = [xy, x+y] at (2,3) → y + 1 = 4
    x22 = MC.real(2, nvars=2) + hx
    y22 = MC.real(3, nvars=2) + hy
    t.check("MV22 div[xy, x+y] at (2,3)",
            MC.divergence_of([x22 * y22, x22 + y22]), 4.0)

    # MV23: div(F) where F = [x, y, z] at any point → 3
    hx3 = MC.zero_var(0, nvars=3)
    hy3 = MC.zero_var(1, nvars=3)
    hz3 = MC.zero_var(2, nvars=3)
    x23 = MC.real(5, nvars=3) + hx3
    y23 = MC.real(7, nvars=3) + hy3
    z23 = MC.real(2, nvars=3) + hz3
    t.check("MV23 div[x, y, z] = 3",
            MC.divergence_of([x23, y23, z23]), 3.0)

    # -------------------------------------------------------------------------
    print("\n--- Curl (MV24-MV26) ---")
    # -------------------------------------------------------------------------

    # MV24: curl of F = [y, -x, 0] at (1,1,0) → [0, 0, -2]
    t.check_list("MV24 curl[y, -x, 0] at (1,1,0)",
                 _curl_at(
                     [lambda x, y, z: y,
                      lambda x, y, z: -1 * x,
                      lambda x, y, z: 0 * x],
                     [1, 1, 0]),
                 [0.0, 0.0, -2.0])

    # MV25: curl of F = [0, 0, x] at (1,1,1) → [0, -1, 0]
    t.check_list("MV25 curl[0, 0, x] at (1,1,1)",
                 _curl_at(
                     [lambda x, y, z: 0 * x,
                      lambda x, y, z: 0 * x,
                      lambda x, y, z: x],
                     [1, 1, 1]),
                 [0.0, -1.0, 0.0])

    # MV26: curl(∇f) = 0 for f = x²y + yz²
    t.check_list("MV26 curl(∇f) = 0 for f=x²y+yz²",
                 _curl_at(
                     [lambda x, y, z: 2 * x * y,
                      lambda x, y, z: x**2 + z**2,
                      lambda x, y, z: 2 * y * z],
                     [1, 2, 3]),
                 [0.0, 0.0, 0.0], tol=1e-6)

    # -------------------------------------------------------------------------
    print("\n--- Type Safety (MV27-MV30) ---")
    # -------------------------------------------------------------------------

    # MV27: mc_sin of plain float returns MC
    result27 = mc_sin(1.0)
    t.check_true("MV27 mc_sin(1.0) returns MC",
                 isinstance(result27, MC),
                 f"got {type(result27).__name__}")

    # MV28: mc_exp of plain int returns MC
    result28 = mc_exp(0)
    t.check_true("MV28 mc_exp(0) returns MC",
                 isinstance(result28, MC),
                 f"got {type(result28).__name__}")

    # MV29: abs() works on MC (tests __abs__)
    val29 = MC.real(-5, nvars=2)
    t.check("MV29 abs(MC.real(-5)) = 5", abs(val29), 5.0)

    # MV30: float() works on MC (tests __float__)
    val30 = MC.real(3.14, nvars=2)
    t.check("MV30 float(MC.real(3.14)) = 3.14", float(val30), 3.14)


# =============================================================================
# PART 2: EXTENDED CAPABILITY TESTS (EX01-EX20)
# =============================================================================

def run_extended_tests(t: TestRunner):
    if not EXTENDED_OK:
        for i in range(1, 21):
            t.skip(f"EX{i:02d}", "composite_extended not available")
        return

    print("\n" + "="*60)
    print("PART 2: EXTENDED CAPABILITIES — 20 Tests")
    print("="*60)

    # -------------------------------------------------------------------------
    print("\n--- Residues (EX01-EX04) ---")
    # -------------------------------------------------------------------------

    # EX01: Res(1/z, z=0) = 1
    t.check("EX01 Res[1/z] at z=0",
            residue(lambda z: 1 / z, at=0), 1.0)

    # EX02: Res(1/z², z=0) = 0
    t.check("EX02 Res[1/z²] at z=0",
            residue(lambda z: 1 / (z**2), at=0), 0.0)

    # EX03: Res(e^z/z, z=0) = 1
    t.check("EX03 Res[eᶻ/z] at z=0",
            residue(lambda z: cexp(z) / z, at=0), 1.0)

    # EX04: Res(e^z/z², z=0) = 1
    t.check("EX04 Res[eᶻ/z²] at z=0",
            residue(lambda z: cexp(z) / (z**2), at=0), 1.0)

    # -------------------------------------------------------------------------
    print("\n--- Asymptotic Expansions (EX05-EX08) ---")
    # -------------------------------------------------------------------------

    # EX05: (3x+1)/(x+2) → 3 as x→∞
    t.check("EX05 lim(3x+1)/(x+2) as x→∞",
            limit_at_infinity(lambda x: (3 * x + 1) / (x + 2)), 3.0)

    # EX06: (x²+1)/(x²-1) → 1 as x→∞
    t.check("EX06 lim(x²+1)/(x²-1) as x→∞",
            limit_at_infinity(lambda x: (x**2 + 1) / (x**2 - 1)), 1.0)

    # EX07: Leading coefficient of (3x²+2x+1)/(x²+1) → a₀ = 3
    coeffs = asymptotic_expansion(
        lambda x: (3 * x**2 + 2 * x + 1) / (x**2 + 1), order=2)
    t.check("EX07 asymptotic a₀ of (3x²+2x+1)/(x²+1)", coeffs[0], 3.0)

    # EX08: a₁ coefficient (1/x term) = 2
    t.check("EX08 asymptotic a₁ of (3x²+2x+1)/(x²+1)", coeffs[1], 2.0)

    # -------------------------------------------------------------------------
    print("\n--- Convergence Radius (EX09-EX12) ---")
    # -------------------------------------------------------------------------

    # EX09: 1/(1+x²) around x=0 has radius 1
    t.check("EX09 R[1/(1+x²)] at x=0",
            convergence_radius(lambda x: 1 / (1 + x**2), at=0),
            1.0, tol=0.15)

    # EX10: 1/(1-x) around x=0 has radius 1
    t.check("EX10 R[1/(1-x)] at x=0",
            convergence_radius(lambda x: 1 / (1 - x + R(1e-15)), at=0),
            1.0, tol=0.15)

    # EX11: exp(x) has infinite radius
    r11 = convergence_radius(lambda x: exp(x), at=0)
    t.check("EX11 R[eˣ] at x=0 (should be large)",
            min(r11, 100), 100, tol=1)

    # EX12: ln(x) around x=1 has radius 1
    t.check("EX12 R[ln(x)] at x=1",
            convergence_radius(lambda x: ln(x), at=1),
            1.0, tol=0.15)

    # -------------------------------------------------------------------------
    print("\n--- ODE Solving (EX13-EX16) ---")
    # -------------------------------------------------------------------------

    # EX13: y' = y, y(0) = 1 → y(1) = e
    pts13 = solve_ode(lambda x, y: y, (0, 1), y0=1)
    t.check("EX13 y'=y, y(0)=1 → y(1)", pts13[-1][1], math.e, tol=1e-4)

    # EX14: y' = -y, y(0) = 1 → y(1) = 1/e
    pts14 = solve_ode(lambda x, y: -y, (0, 1), y0=1)
    t.check("EX14 y'=-y, y(0)=1 → y(1)", pts14[-1][1], 1/math.e, tol=1e-4)

    # EX15: y' = 2x, y(0) = 0 → y(1) = 1
    pts15 = solve_ode(lambda x, y: 2 * x, (0, 1), y0=0)
    t.check("EX15 y'=2x, y(0)=0 → y(1)", pts15[-1][1], 1.0, tol=1e-4)

    # EX16: y' = cos(x), y(0) = 0 → y(π) = 0
    pts16 = solve_ode(lambda x, y: cos(x), (0, math.pi), y0=0)
    t.check("EX16 y'=cos(x), y(0)=0 → y(π)", pts16[-1][1], 0.0, tol=1e-3)

    # -------------------------------------------------------------------------
    print("\n--- Improper Integrals (EX17-EX18) ---")
    # -------------------------------------------------------------------------

    # EX17: ∫₀^∞ e^(-x) dx = 1
    val17, err17 = improper_integral(lambda x: exp(-x), 0)
    t.check("EX17 ∫₀^∞ e⁻ˣ dx", val17, 1.0, tol=1e-3)

    # EX18: ∫₀^∞ e^(-x²) dx = √π/2
    val18, err18 = improper_integral(lambda x: exp(-(x * x)), 0)
    t.check("EX18 ∫₀^∞ e⁻ˣ² dx", val18, math.sqrt(math.pi)/2, tol=1e-3)

    # -------------------------------------------------------------------------
    print("\n--- Analytic Continuation (EX19-EX20) ---")
    # -------------------------------------------------------------------------

    # EX19: Continue ln(x) from x=1 to x=3 → ln(3)
    res19 = analytic_continue(lambda x: ln(x), start=1, target=3)
    t.check("EX19 ln(x) continued 1→3", res19.st(), math.log(3), tol=1e-4)

    # EX20: Continue ln(x) from x=1 to x=0.5 → ln(0.5)
    res20 = analytic_continue(lambda x: ln(x), start=1, target=0.5)
    t.check("EX20 ln(x) continued 1→0.5", res20.st(), math.log(0.5), tol=1e-4)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# COMBINED TEST SUITE: composite_multivar + composite_extended")
    print("#"*60)

    t = TestRunner()
    t0 = time.time()

    run_multivar_tests(t)
    run_extended_tests(t)

    elapsed = time.time() - t0

    print(f"\n\n{'#'*60}")
    print(f"# FINAL RESULTS")
    print(f"{'#'*60}")
    total = t.passed + t.failed
    print(f"\n  Total:   {total} tests")
    print(f"  Passed:  {t.passed}")
    print(f"  Failed:  {t.failed}")
    if t.skipped:
        print(f"  Skipped: {t.skipped}")
    print(f"  Time:    {elapsed:.3f}s")
    pct = 100 * t.passed / total if total else 0
    print(f"  Rate:    {pct:.1f}%")

    if t.failed:
        print("\n  ❌ FAILED TESTS:")
        for name, ok in t.results:
            if ok is False:
                print(f"     - {name}")

    print()
    if t.failed == 0 and t.skipped == 0:
        print("  🎯 ALL 50 TESTS PASSED!")
    elif t.failed == 0:
        print("  ✅ All run tests passed (some skipped due to missing imports).")
    else:
        print(f"  ⚠️  {t.failed} test(s) need attention.")

    sys.exit(0 if t.failed == 0 else 1)
