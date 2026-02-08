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
composite_stress_test.py — 20 Hard Problems
Run: python composite_stress_test.py
"""
import math, sys, time
from composite_lib import (
    R, ZERO, INF, sin, cos, tan, exp, ln, sqrt,
    atan, sinh, cosh, tanh,
    derivative, nth_derivative, limit, integrate_adaptive,
)

passed = failed = errors = 0

def check(tag, got, want, tol=1e-6):
    global passed, failed, errors
    try:
        ok = abs(got - want) < tol
        passed += ok; failed += (not ok)
        print(f"  {'✓' if ok else '✗'} {tag}")
        if not ok: print(f"      got={got:.12g}  want={want:.12g}  diff={abs(got-want):.2e}")
    except Exception as e:
        errors += 1; print(f"  ⚠ {tag}  ERROR: {e}")

def check_int(tag, f, a, b, want, tol=1e-4):
    global passed, failed, errors
    try:
        val, err = integrate_adaptive(f, a, b)
        ok = abs(val - want) < tol
        passed += ok; failed += (not ok)
        print(f"  {'✓' if ok else '✗'} {tag}")
        if not ok: print(f"      got={val:.12g}  want={want:.12g}  diff={abs(val-want):.2e}")
    except Exception as e:
        errors += 1; print(f"  ⚠ {tag}  ERROR: {e}")

pi, e = math.pi, math.e
t0 = time.perf_counter()

# ============ LIMITS (7) ============
print("=" * 55)
print("LIMITS")
print("=" * 55)

check("L01 lim x→0 sin(x)/x = 1",
      limit(lambda x: sin(x)/x, as_x_to=0), 1.0)

check("L02 lim x→0 (1−cos x)/x² = 1/2",
      limit(lambda x: (R(1) - cos(x))/(x**2), as_x_to=0), 0.5)

check("L03 lim x→0 (eˣ−1−x)/x² = 1/2",
      limit(lambda x: (exp(x) - R(1) - x)/(x**2), as_x_to=0), 0.5)

check("L04 lim x→0 (sin x − x)/x³ = −1/6",
      limit(lambda x: (sin(x) - x)/(x**3), as_x_to=0), -1.0/6)

check("L05 lim x→0 (tan x − sin x)/x³ = 1/2",
      limit(lambda x: (tan(x) - sin(x))/(x**3), as_x_to=0), 0.5)

check("L06 lim x→0 (√(1+x)−√(1−x))/x = 1",
      limit(lambda x: (sqrt(R(1)+x) - sqrt(R(1)-x))/x, as_x_to=0), 1.0)

check("L07 lim x→∞ (5x²+3x)/(2x²+1) = 5/2",
      limit(lambda x: (R(5)*x**2+R(3)*x)/(R(2)*x**2+R(1)),
            as_x_to=float('inf')), 2.5)

# ============ DERIVATIVES (7) ============
print("\n" + "=" * 55)
print("DERIVATIVES")
print("=" * 55)

check("D01 d/dx[sin x] at π/4 = cos(π/4)",
      derivative(lambda x: sin(x), at=pi/4), math.cos(pi/4))

check("D02 d/dx[e^(−x²)] at 1 = −2e⁻¹",
      derivative(lambda x: exp(-(x*x)), at=1), -2*math.exp(-1))

check("D03 d/dx[atan x] at 1 = 1/2",
      derivative(lambda x: atan(x), at=1), 0.5)

check("D04 d/dx[sin(x²)] at 1 = 2·cos 1",
      derivative(lambda x: sin(x*x), at=1), 2*math.cos(1))

check("D05 d²/dx²[eˣ·sin x] at 0 = 2",
      nth_derivative(lambda x: exp(x)*sin(x), n=2, at=0), 2.0)

check("D06 d³/dx³[sin x] at 0 = −1",
      nth_derivative(lambda x: sin(x), n=3, at=0), -1.0)

check("D07 d⁵/dx⁵[eˣ] at 1 = e",
      nth_derivative(lambda x: exp(x), n=5, at=1), e)

# ============ INTEGRALS (6) ============
print("\n" + "=" * 55)
print("INTEGRALS")
print("=" * 55)

check_int("I01 ∫₀¹ x² dx = 1/3",
          lambda x: x**2, 0, 1, 1.0/3)

check_int("I02 ∫₀¹ eˣ dx = e−1",
          lambda x: exp(x), 0, 1, e - 1)

check_int("I03 ∫₀π sin x dx = 2",
          lambda x: sin(x), 0, pi, 2.0)

check_int("I04 ∫₁² e^(−x²) dx ≈ 0.13526 (Gaussian)",
          lambda x: exp(-(x*x)), 1, 2, 0.13525725794)

check_int("I05 ∫₀¹ x·sin x dx = sin1−cos1",
          lambda x: x*sin(x), 0, 1, math.sin(1)-math.cos(1))

check_int("I06 ∫₀¹ cosh x dx = sinh 1",
          lambda x: cosh(x), 0, 1, math.sinh(1))

# ============ SUMMARY ============
total = passed + failed + errors
elapsed = time.perf_counter() - t0
print("\n" + "=" * 55)
print(f"RESULTS: {passed}/{total} passed, {failed} failed, {errors} errors")
print(f"Time: {elapsed:.2f}s")
print("=" * 55)
if passed == total:
    print("🎉 ALL 20 TESTS PASSED")
sys.exit(0 if passed == total else 1)
