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
test_integration_comprehensive.py — Unified Integration Test Suite (v2)
========================================================================
ALL tests route through the single integrate() wrapper.
"One integral to rule them all."

Calling conventions tested:
  integrate(f, a, b)                            — 1D definite
  integrate(f, a, float('inf'))                  — semi-infinite improper
  integrate(f, float('-inf'), float('inf'))       — doubly-infinite improper
  integrate(f, (a,b), (c,d), (e,g))              — 3D triple
  integrate(f, (t0,t1), curve=gamma)             — scalar line integral
  integrate(F, (t0,t1), curve=gamma)             — vector line integral
  integrate(f, ((u0,u1),(v0,v1)), surface=sigma) — scalar surface integral
  integrate(F, ((u0,u1),(v0,v1)), surface=sigma) — vector surface integral (flux)

Only import needed: integrate (+ math helpers) from composite_lib.

Run:
  python test_integration_comprehensive.py

Author: Toni Milovan
License: AGPL
"""

import math
import sys
import time

from composite.composite_lib import (
    Composite, R, ZERO, INF,
    sin, cos, exp, ln, sqrt, tan,
    sinh, cosh, tanh, atan,
    integrate, antiderivative,
)


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.results = []

    def check(self, tag, got, want, tol=1e-4):
        """Check a scalar value against expected."""
        try:
            got_f = float(got.st()) if isinstance(got, Composite) else float(got)
            ok = abs(got_f - want) < tol
            self.passed += ok
            self.failed += (not ok)
            self.results.append((tag, ok))
            status = "✅" if ok else "❌"
            print(f"  {status} {tag}")
            if not ok:
                print(f"      got={got_f:.10g}  want={want:.10g}  diff={abs(got_f-want):.2e}")
        except Exception as e:
            self.errors += 1
            self.results.append((tag, False))
            print(f"  ⚠️  {tag}  ERROR: {e}")

    def summary(self):
        total = self.passed + self.failed + self.errors
        print(f"\n{'='*65}")
        print(f"RESULTS: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f", {self.failed} failed", end="")
        if self.errors:
            print(f", {self.errors} errors", end="")
        print()
        print(f"{'='*65}")
        if self.passed == total:
            print("🎯 ALL INTEGRATION TESTS PASSED!")
        else:
            print(f"⚠️  {total - self.passed} test(s) need attention.")
            print("\nFailed tests:")
            for name, ok in self.results:
                if not ok:
                    print(f"  ❌ {name}")
        return self.passed == total


pi = math.pi
e  = math.e


# =============================================================================
# CATEGORY 1: BASIC DEFINITE INTEGRALS  —  integrate(f, a, b)
# =============================================================================

def test_basic_definite(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 1: Basic Definite Integrals — integrate(f, a, b)")
    print("="*65)

    t.check("I01 ∫₀¹ x² dx = 1/3",
            integrate(lambda x: x**2, 0, 1), 1/3)

    t.check("I02 ∫₀¹ eˣ dx = e−1",
            integrate(lambda x: exp(x), 0, 1), e - 1)

    t.check("I03 ∫₀π sin x dx = 2",
            integrate(lambda x: sin(x), 0, pi), 2.0)

    t.check("I04 ∫₁² e^(−x²) dx ≈ 0.13526",
            integrate(lambda x: exp(-(x*x)), 1, 2), 0.13525725794)

    t.check("I05 ∫₀¹ x·sin x dx = sin1−cos1",
            integrate(lambda x: x*sin(x), 0, 1),
            math.sin(1) - math.cos(1))

    t.check("I06 ∫₀¹ cosh x dx = sinh 1",
            integrate(lambda x: cosh(x), 0, 1), math.sinh(1))


# =============================================================================
# CATEGORY 2: HARD DEFINITE INTEGRALS  —  integrate(f, a, b)
# =============================================================================

def test_hard_definite(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 2: Hard Definite Integrals — integrate(f, a, b)")
    print("="*65)

    t.check("I07 ∫₀¹ x²·eˣ dx = e−2",
            integrate(lambda x: x**2 * exp(x), 0, 1), e - 2)

    t.check("I08 ∫₀^(π/2) sin²x dx = π/4",
            integrate(lambda x: sin(x) * sin(x), 0, pi/2), pi/4)

    t.check("I09 ∫₀^(π/2) sin³x dx = 2/3",
            integrate(lambda x: sin(x)**3, 0, pi/2), 2/3)

    t.check("I10 ∫₀¹ e^(−x)·cos(x) dx",
            integrate(lambda x: exp(-x) * cos(x), 0, 1),
            (1.0 + math.exp(-1)*(math.sin(1) - math.cos(1))) / 2)

    t.check("I11 ∫₀¹ x·e^(−x²) dx = (1−e⁻¹)/2",
            integrate(lambda x: x * exp(-(x*x)), 0, 1),
            (1 - math.exp(-1)) / 2)

    t.check("I12 ∫₀¹ x²·cos(x) dx",
            integrate(lambda x: x**2 * cos(x), 0, 1),
            -math.sin(1) + 2*math.cos(1))

    t.check("I13 ∫₀² 1/(1+x²) dx = atan(2)",
            integrate(lambda x: R(1) / (R(1) + x*x), 0, 2),
            math.atan(2))


# =============================================================================
# CATEGORY 3: ADDITIONAL DEFINITE INTEGRALS  —  integrate(f, a, b)
# =============================================================================

def test_additional_definite(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 3: Additional Definite Integrals — integrate(f, a, b)")
    print("="*65)

    t.check("I14 ∫₀¹ √x dx = 2/3",
            integrate(lambda x: sqrt(x), 0.001, 1), 2/3, tol=1e-3)

    t.check("I15 ∫₀¹ x³ dx = 1/4",
            integrate(lambda x: x**3, 0, 1), 0.25)

    t.check("I16 ∫₀π cos²x dx = π/2",
            integrate(lambda x: cos(x) * cos(x), 0, pi), pi/2)

    t.check("I17 ∫₀¹ 1/(1+x) dx = ln 2",
            integrate(lambda x: R(1) / (R(1) + x), 0, 1),
            math.log(2))

    t.check("I18 ∫₁ᵉ ln(x)/x dx = 1/2",
            integrate(lambda x: ln(x) / x, 1, e), 0.5)

    t.check("I19 ∫₀¹ x·eˣ dx = 1",
            integrate(lambda x: x * exp(x), 0, 1), 1.0)

    t.check("I20 ∫₀^(π/2) sin(x)·cos(x) dx = 1/2",
            integrate(lambda x: sin(x) * cos(x), 0, pi/2), 0.5)


# =============================================================================
# CATEGORY 4: ANTIDERIVATIVE ROUND-TRIP  (dimensional shift — no routing)
# =============================================================================

def _derivative_of(F_composite):
    """Inverse of antiderivative: shift up, multiply by position."""
    result = {}
    for dim, coeff in F_composite.c.items():
        if dim < 0:
            new_dim = dim + 1
            result[new_dim] = coeff * abs(dim)
    return Composite(result)


def test_antiderivative_roundtrip(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 4: Antiderivative Round-Trip (dimensional shift)")
    print("="*65)

    h = ZERO

    f1 = R(3) + h
    F1 = antiderivative(f1)
    f1_back = _derivative_of(F1)
    t.check("A01 ∫x dx round-trip at x=3", f1_back.st(), f1.st())

    x = R(2) + h
    f2 = x ** 2
    F2 = antiderivative(f2)
    f2_back = _derivative_of(F2)
    t.check("A02 ∫x² dx round-trip at x=2", f2_back.st(), f2.st())

    f3 = x ** 3
    F3 = antiderivative(f3)
    f3_back = _derivative_of(F3)
    t.check("A03 ∫x³ dx round-trip at x=2", f3_back.st(), f3.st())

    t.check("A04 ∞ × 0 = 1 (Riemann sum foundation)", (INF * ZERO).st(), 1.0)


# =============================================================================
# CATEGORY 5: IMPROPER INTEGRALS  —  integrate(f, a, ∞) / integrate(f, −∞, ∞)
# =============================================================================

def test_improper(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 5: Improper Integrals — integrate(f, a, ∞)")
    print("="*65)

    t.check("IP01 ∫₀^∞ e⁻ˣ dx = 1",
            integrate(lambda x: exp(-x), 0, float('inf')),
            1.0, tol=1e-3)

    t.check("IP02 ∫₀^∞ e⁻ˣ² dx = √π/2",
            integrate(lambda x: exp(-(x * x)), 0, float('inf')),
            math.sqrt(pi)/2, tol=1e-3)

    t.check("IP03 ∫₀^∞ x·e⁻ˣ dx = 1 (Γ(2))",
            integrate(lambda x: x * exp(-x), 0, float('inf')),
            1.0, tol=1e-3)

    t.check("IP04 ∫₀^∞ x²·e⁻ˣ dx = 2 (Γ(3))",
            integrate(lambda x: x**2 * exp(-x), 0, float('inf')),
            2.0, tol=1e-2)

    t.check("IP05 ∫₋∞^∞ e⁻ˣ² dx = √π (Gaussian)",
            integrate(lambda x: exp(-(x * x)), float('-inf'), float('inf')),
            math.sqrt(pi), tol=1e-2)


# =============================================================================
# CATEGORY 6: TRIPLE INTEGRALS  —  integrate(f, (a,b), (c,d), (e,g))
# =============================================================================

def test_triple(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 6: Triple Integrals — integrate(f, (a,b), (c,d), (e,g))")
    print("="*65)

    t.check("T01 Unit cube volume = 1",
            integrate(lambda x, y, z: x*0 + 1, (0,1), (0,1), (0,1)),
            1.0, tol=1e-3)

    t.check("T02 ∭ x dV over [0,1]³ = 1/2",
            integrate(lambda x, y, z: x, (0,1), (0,1), (0,1)),
            0.5, tol=1e-3)

    t.check("T03 ∭ xyz dV = 1/8",
            integrate(lambda x, y, z: x * y * z, (0,1), (0,1), (0,1)),
            0.125, tol=1e-3)

    t.check("T04 Volume of 2×3×4 box = 24",
            integrate(lambda x, y, z: x*0 + 1, (0,2), (0,3), (0,4)),
            24.0, tol=0.1)

    t.check("T05 ∭ (x²+y²) dV = 2/3",
            integrate(lambda x, y, z: x**2 + y**2, (0,1), (0,1), (0,1)),
            2/3, tol=1e-2)


# =============================================================================
# CATEGORY 7: LINE INTEGRALS  —  integrate(f/F, (t0,t1), curve=γ)
# =============================================================================

def test_line(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 7: Line Integrals — integrate(f/F, (t0,t1), curve=γ)")
    print("="*65)

    # --- Scalar line integrals: f is a callable ---

    t.check("L01 Arc length (0,0)→(3,4) = 5",
            integrate(lambda x, y: 1,
                      (0, 1), curve=lambda t: [3*t, 4*t]),
            5.0, tol=1e-3)

    t.check("L02 ∫ x ds along x-axis = 1/2",
            integrate(lambda x, y: x,
                      (0, 1), curve=lambda t: [t, 0*t]),
            0.5, tol=1e-3)

    t.check("L03 ∫ (x+y) ds along y=x = √2",
            integrate(lambda x, y: x + y,
                      (0, 1), curve=lambda t: [t, t]),
            math.sqrt(2), tol=1e-3)

    t.check("L04 Circumference of unit circle = 2π",
            integrate(lambda x, y: 1,
                      (0, 2*pi),
                      curve=lambda t: [math.cos(t), math.sin(t)]),
            2*pi, tol=1e-2)

    t.check("L05 Helix arc length = 2π√2",
            integrate(lambda x, y, z: 1,
                      (0, 2*pi),
                      curve=lambda t: [math.cos(t), math.sin(t), t]),
            2*pi*math.sqrt(2), tol=0.1)

    # --- Vector line integrals: f is a list of component callables ---

    t.check("L06 Constant force work = 3",
            integrate([lambda x, y: 3, lambda x, y: 0],
                      (0, 1), curve=lambda t: [t, 0*t]),
            3.0, tol=1e-3)

    t.check("L07 Conservative field work = 1",
            integrate([lambda x, y: y, lambda x, y: x],
                      (0, 1), curve=lambda t: [t, t]),
            1.0, tol=1e-3)

    t.check("L08 Rotation field circulation = 2π",
            integrate([lambda x, y: -y, lambda x, y: x],
                      (0, 2*pi),
                      curve=lambda t: [math.cos(t), math.sin(t)]),
            2*pi, tol=0.1)

    t.check("L09 Conservative field (closed loop) = 0",
            integrate([lambda x, y: 2*x, lambda x, y: 2*y],
                      (0, 2*pi),
                      curve=lambda t: [math.cos(t), math.sin(t)]),
            0.0, tol=0.1)

    t.check("L10 3D constant field work = 6",
            integrate([lambda x, y, z: 1, lambda x, y, z: 2, lambda x, y, z: 3],
                      (0, 1), curve=lambda t: [t, t, t]),
            6.0, tol=1e-3)


# =============================================================================
# CATEGORY 8: SURFACE INTEGRALS  —  integrate(f/F, uv, surface=σ)
# =============================================================================

def test_surface(t: TestRunner):
    print("\n" + "="*65)
    print("CATEGORY 8: Surface Integrals — integrate(f/F, uv, surface=σ)")
    print("="*65)

    # --- Scalar surface integrals ---

    t.check("S01 Unit square area = 1",
            integrate(lambda x, y, z: 1,
                      ((0,1), (0,1)),
                      surface=lambda u, v: [u, v, 0]),
            1.0, tol=1e-2)

    t.check("S02 3×4 rectangle area = 12",
            integrate(lambda x, y, z: 1,
                      ((0,3), (0,4)),
                      surface=lambda u, v: [u, v, 0]),
            12.0, tol=0.1)

    t.check("S03 Unit sphere area = 4π",
            integrate(lambda x, y, z: 1,
                      ((0, pi), (0, 2*pi)),
                      surface=lambda u, v: [math.sin(u)*math.cos(v),
                                            math.sin(u)*math.sin(v),
                                            math.cos(u)]),
            4*pi, tol=0.5)

    t.check("S04 Cylinder lateral area = 4π",
            integrate(lambda x, y, z: 1,
                      ((0, 2*pi), (0, 2)),
                      surface=lambda u, v: [math.cos(u), math.sin(u), v]),
            4*pi, tol=0.5)

    t.check("S05 ∬ z dS over hemisphere = π",
            integrate(lambda x, y, z: z,
                      ((0, pi/2), (0, 2*pi)),
                      surface=lambda u, v: [math.sin(u)*math.cos(v),
                                            math.sin(u)*math.sin(v),
                                            math.cos(u)]),
            pi, tol=0.5)

    # --- Vector surface integrals (flux): f is a list of component callables ---

    t.check("S06 Constant flux through square = 1",
            integrate([lambda x, y, z: 0, lambda x, y, z: 0, lambda x, y, z: 1],
                      ((0,1), (0,1)),
                      surface=lambda u, v: [u, v, 0]),
            1.0, tol=1e-2)

    t.check("S07 Radial flux through sphere = 4π",
            integrate([lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: z],
                      ((0, pi), (0, 2*pi)),
                      surface=lambda u, v: [math.sin(u)*math.cos(v),
                                            math.sin(u)*math.sin(v),
                                            math.cos(u)]),
            4*pi, tol=0.5)

    t.check("S08 Tangent field (zero flux) ≈ 0",
            integrate([lambda x, y, z: -y, lambda x, y, z: x, lambda x, y, z: 0],
                      ((0, pi), (0, 2*pi)),
                      surface=lambda u, v: [math.sin(u)*math.cos(v),
                                            math.sin(u)*math.sin(v),
                                            math.cos(u)]),
            0.0, tol=0.5)

    t.check("S09 Flux through cylinder = 4π",
            integrate([lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: 0],
                      ((0, 2*pi), (0, 2)),
                      surface=lambda u, v: [math.cos(u), math.sin(u), v]),
            4*pi, tol=0.5)

    t.check("S10 Flux through tilted plane = 1",
            integrate([lambda x, y, z: 0, lambda x, y, z: 0, lambda x, y, z: 1],
                      ((0,1), (0,1)),
                      surface=lambda u, v: [u, v, u + v]),
            1.0, tol=0.1)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all():
    print("\n" + "#"*65)
    print("# COMPREHENSIVE INTEGRATION TEST SUITE (v2)")
    print("# All tests route through integrate()")
    print("# One integral to rule them all")
    print("#"*65)

    t = TestRunner()
    t0 = time.perf_counter()

    test_basic_definite(t)
    test_hard_definite(t)
    test_additional_definite(t)
    test_antiderivative_roundtrip(t)
    test_improper(t)
    test_triple(t)
    test_line(t)
    test_surface(t)

    elapsed = time.perf_counter() - t0
    print(f"\nTime: {elapsed:.2f}s")
    return t.summary()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
