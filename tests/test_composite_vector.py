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
test_composite_vector.py — Vector Calculus Tests
================================================
Validation suite for triple integrals, line integrals, and surface integrals.

Each test compares numerical results against known analytical solutions.

Author: Toni Milovan
"""

import math
from composite.composite_vector import (
    triple_integral,
    line_integral_scalar,
    line_integral_vector,
    surface_integral_scalar,
    surface_integral_vector
)

# Test tolerance
TOL = 1e-2

def assert_close(computed, expected, name, tol=TOL):
    """Helper to check numerical results."""
    error = abs(computed - expected)
    rel_error = error / (abs(expected) + 1e-10)
    if error > tol and rel_error > tol:
        print(f"❌ {name}: got {computed:.6f}, expected {expected:.6f}, error={error:.6f}")
        return False
    else:
        print(f"✅ {name}: {computed:.6f} ≈ {expected:.6f}")
        return True


# =============================================================================
# TRIPLE INTEGRAL TESTS
# =============================================================================

def test_triple_integrals():
    print("\n=== Triple Integrals ===")
    passed = 0
    total = 0

    # Test 1: Volume of unit cube
    total += 1
    result = triple_integral(
        lambda x, y, z: 1,
        (0, 1), (0, 1), (0, 1)
    )
    if assert_close(result, 1.0, "Unit cube volume"):
        passed += 1

    # Test 2: Integral of x over unit cube
    total += 1
    result = triple_integral(
        lambda x, y, z: x.st() if hasattr(x, 'st') else x,
        (0, 1), (0, 1), (0, 1)
    )
    if assert_close(result, 0.5, "∭ x dx dy dz over [0,1]³"):
        passed += 1

    # Test 3: Integral of x*y*z over unit cube
    total += 1
    result = triple_integral(
        lambda x, y, z: x * y * z,
        (0, 1), (0, 1), (0, 1)
    )
    if assert_close(result, 0.125, "∭ xyz dx dy dz = 1/8"):
        passed += 1

    # Test 4: Volume of rectangular box
    total += 1
    result = triple_integral(
        lambda x, y, z: 1,
        (0, 2), (0, 3), (0, 4)
    )
    if assert_close(result, 24.0, "Volume of 2×3×4 box"):
        passed += 1

    # Test 5: Moment integral (x² + y²) over unit cube
    total += 1
    result = triple_integral(
        lambda x, y, z: x**2 + y**2,
        (0, 1), (0, 1), (0, 1)
    )
    expected = 2.0/3.0  # ∫₀¹ x² dx = 1/3, times 2 for x² and y²
    if assert_close(result, expected, "∭ (x²+y²) dx dy dz", tol=0.05):
        passed += 1

    return passed, total


# =============================================================================
# LINE INTEGRAL TESTS (Scalar Fields)
# =============================================================================

def test_line_integrals_scalar():
    print("\n=== Line Integrals (Scalar Fields) ===")
    passed = 0
    total = 0

    # Test 1: Arc length of line segment from (0,0) to (3,4)
    total += 1
    result = line_integral_scalar(
        lambda x, y: 1,
        lambda t: [3*t, 4*t],
        (0, 1)
    )
    expected = 5.0  # √(3² + 4²) = 5
    if assert_close(result, expected, "Arc length (0,0)→(3,4)"):
        passed += 1

    # Test 2: Integral of f(x,y) = x along x-axis from 0 to 1
    total += 1
    result = line_integral_scalar(
        lambda x, y: x,
        lambda t: [t, 0],
        (0, 1)
    )
    expected = 0.5  # ∫₀¹ x dx = 1/2
    if assert_close(result, expected, "∫ x ds along x-axis"):
        passed += 1

    # Test 3: Integral of f(x,y) = x+y along diagonal
    total += 1
    result = line_integral_scalar(
        lambda x, y: x + y,
        lambda t: [t, t],
        (0, 1)
    )
    expected = math.sqrt(2)  # ∫₀¹ 2t √2 dt = √2
    if assert_close(result, expected, "∫ (x+y) ds along y=x"):
        passed += 1

    # Test 4: Circumference of unit circle
    total += 1
    result = line_integral_scalar(
        lambda x, y: 1,
        lambda t: [math.cos(t), math.sin(t)],
        (0, 2*math.pi)
    )
    expected = 2*math.pi
    if assert_close(result, expected, "Circumference of unit circle"):
        passed += 1

    # Test 5: 3D helix arc length (one turn)
    total += 1
    result = line_integral_scalar(
        lambda x, y, z: 1,
        lambda t: [math.cos(t), math.sin(t), t],
        (0, 2*math.pi)
    )
    expected = 2*math.pi * math.sqrt(2)  # √(1² + 1²) × 2π
    if assert_close(result, expected, "Helix arc length", tol=0.1):
        passed += 1

    return passed, total


# =============================================================================
# LINE INTEGRAL TESTS (Vector Fields)
# =============================================================================

def test_line_integrals_vector():
    print("\n=== Line Integrals (Vector Fields) ===")
    passed = 0
    total = 0

    # Test 1: Work by constant force F = [1, 0] along x-axis
    total += 1
    result = line_integral_vector(
        [lambda x, y: 1, lambda x, y: 0],
        lambda t: [t, 0],
        (0, 3)
    )
    expected = 3.0  # Force · displacement = 1 × 3
    if assert_close(result, expected, "Constant force along x-axis"):
        passed += 1

    # Test 2: Conservative field F = [x, y] from origin to (1,1)
    total += 1
    result = line_integral_vector(
        [lambda x, y: x,
         lambda x, y: y],
        lambda t: [t, t],
        (0, 1)
    )
    expected = 1.0  # Potential ϕ = (x²+y²)/2, ϕ(1,1) - ϕ(0,0) = 1
    if assert_close(result, expected, "Conservative field work"):
        passed += 1

    # Test 3: Circulation of rotation field F = [-y, x] around unit circle
    total += 1
    result = line_integral_vector(
        [lambda x, y: -y,
         lambda x, y: x],
        lambda t: [math.cos(t), math.sin(t)],
        (0, 2*math.pi)
    )
    expected = 2*math.pi  # Circulation = ∫ r² dθ = 2π for r=1
    if assert_close(result, expected, "Rotation field circulation"):
        passed += 1

    # Test 4: Gradient field around closed loop (should be zero)
    total += 1
    result = line_integral_vector(
        [lambda x, y: x,
         lambda x, y: y],
        lambda t: [math.cos(t), math.sin(t)],
        (0, 2*math.pi)
    )
    expected = 0.0  # Conservative field around closed loop
    if assert_close(result, expected, "Conservative field (closed loop)", tol=0.05):
        passed += 1

    # Test 5: 3D vector field work along straight path
    total += 1
    result = line_integral_vector(
        [lambda x, y, z: 1,
         lambda x, y, z: 2,
         lambda x, y, z: 3],
        lambda t: [t, t, t],
        (0, 1)
    )
    expected = 6.0  # ∫ F·dr = ∫₀¹ (1+2+3) dt = 6 (NOT multiplied by |r'|)
    if assert_close(result, expected, "3D constant field work"):
        passed += 1

    return passed, total


# =============================================================================
# SURFACE INTEGRAL TESTS (Scalar Fields)
# =============================================================================

def test_surface_integrals_scalar():
    print("\n=== Surface Integrals (Scalar Fields) ===")
    passed = 0
    total = 0

    # Test 1: Surface area of unit square in xy-plane
    total += 1
    result = surface_integral_scalar(
        lambda x, y, z: 1,
        lambda u, v: [u, v, 0],
        (0, 1), (0, 1)
    )
    expected = 1.0
    if assert_close(result, expected, "Unit square area"):
        passed += 1

    # Test 2: Surface area of rectangular region
    total += 1
    result = surface_integral_scalar(
        lambda x, y, z: 1,
        lambda u, v: [u, v, 0],
        (0, 3), (0, 4)
    )
    expected = 12.0
    if assert_close(result, expected, "3×4 rectangle area"):
        passed += 1

    # Test 3: Surface area of unit sphere
    total += 1
    result = surface_integral_scalar(
        lambda x, y, z: 1,
        lambda u, v: [math.sin(u)*math.cos(v),
                     math.sin(u)*math.sin(v),
                     math.cos(u)],
        (0, math.pi), (0, 2*math.pi)
    )
    expected = 4*math.pi
    if assert_close(result, expected, "Unit sphere surface area", tol=0.2):
        passed += 1

    # Test 4: Surface area of cylinder (lateral surface, radius=1, height=2)
    total += 1
    result = surface_integral_scalar(
        lambda x, y, z: 1,
        lambda u, v: [math.cos(u), math.sin(u), v],
        (0, 2*math.pi), (0, 2)
    )
    expected = 4*math.pi  # 2πrh = 2π(1)(2)
    if assert_close(result, expected, "Cylinder lateral surface", tol=0.15):
        passed += 1

    # Test 5: Integral of z over hemisphere z = √(1-x²-y²)
    total += 1
    result = surface_integral_scalar(
        lambda x, y, z: z,
        lambda u, v: [math.sin(u)*math.cos(v),
                     math.sin(u)*math.sin(v),
                     math.cos(u)],
        (0, math.pi/2), (0, 2*math.pi)
    )
    expected = math.pi  # ∬ z dS over hemisphere = π (not 2π)
    if assert_close(result, expected, "∬ z dS over hemisphere", tol=0.1):
        passed += 1

    return passed, total


# =============================================================================
# SURFACE INTEGRAL TESTS (Vector Fields / Flux)
# =============================================================================

def test_surface_integrals_vector():
    print("\n=== Surface Integrals (Vector Fields) ===")
    passed = 0
    total = 0

    # Test 1: Flux of F = [0, 0, 1] through unit square in xy-plane
    total += 1
    result = surface_integral_vector(
        [lambda x, y, z: 0,
         lambda x, y, z: 0,
         lambda x, y, z: 1],
        lambda u, v: [u, v, 0],
        (0, 1), (0, 1)
    )
    expected = 1.0  # Constant field, area = 1
    if assert_close(result, expected, "Constant flux through square"):
        passed += 1

    # Test 2: Flux of F = [x, y, z] through unit sphere (div F = 3)
    total += 1
    result = surface_integral_vector(
        [lambda x, y, z: x,
         lambda x, y, z: y,
         lambda x, y, z: z],
        lambda u, v: [math.sin(u)*math.cos(v),
                     math.sin(u)*math.sin(v),
                     math.cos(u)],
        (0, math.pi), (0, 2*math.pi)
    )
    expected = 4*math.pi  # Divergence theorem: div F = 3, V = 4π/3
    if assert_close(result, expected, "Radial flux through sphere", tol=0.3):
        passed += 1

    # Test 3: Zero flux for tangent field
    total += 1
    result = surface_integral_vector(
        [lambda x, y, z: -y,
         lambda x, y, z: x,
         lambda x, y, z: 0],
        lambda u, v: [math.sin(u)*math.cos(v),
                     math.sin(u)*math.sin(v),
                     math.cos(u)],
        (0, math.pi), (0, 2*math.pi)
    )
    expected = 0.0  # Tangent field has zero flux
    if assert_close(result, expected, "Tangent field (zero flux)", tol=0.1):
        passed += 1

    # Test 4: Flux through cylinder (F = [x, y, 0], outward)
    total += 1
    result = surface_integral_vector(
        [lambda x, y, z: x,
         lambda x, y, z: y,
         lambda x, y, z: 0],
        lambda u, v: [math.cos(u), math.sin(u), v],
        (0, 2*math.pi), (0, 2)
    )
    expected = 4*math.pi  # div F = 2, V = πr²h = 2π
    if assert_close(result, expected, "Flux through cylinder", tol=0.3):
        passed += 1

    # Test 5: Constant field through tilted plane
    total += 1
    result = surface_integral_vector(
        [lambda x, y, z: 0,
         lambda x, y, z: 0,
         lambda x, y, z: 1],
        lambda u, v: [u, v, u + v],  # Plane z = x + y
        (0, 1), (0, 1)
    )
    # Normal = ru × rv = [1,0,1] × [0,1,1] = [-1, -1, 1]
    # F · n = [0,0,1] · [-1,-1,1] = 1, integrated over [0,1]² = 1.0
    expected = 1.0  # Flux = 1.0 (not √3)
    if assert_close(result, expected, "Flux through tilted plane"):
        passed += 1

    return passed, total


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("VECTOR CALCULUS TEST SUITE")
    print("="*60)

    results = []

    results.append(test_triple_integrals())
    results.append(test_line_integrals_scalar())
    results.append(test_line_integrals_vector())
    results.append(test_surface_integrals_scalar())
    results.append(test_surface_integrals_vector())

    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)

    print("\n" + "="*60)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    print(f"Success rate: {100*total_passed/total_tests:.1f}%")
    print("="*60)

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
