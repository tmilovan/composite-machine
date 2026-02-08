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
composite_hard_edges.py — 20 Hard Edge Cases
Run: python composite_hard_edges.py
Requires composite_lib.py (with _poly_divide fix)
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
        print(f"  {chr(10003) if ok else chr(10007)} {tag}")
        if not ok:
            print(f"      got={got:.12g}  want={want:.12g}  diff={abs(got-want):.2e}")
    except Exception as e:
        errors += 1; print(f"  ! {tag}  ERROR: {e}")
def check_int(tag, f, a, b, want, tol=1e-4):
    global passed, failed, errors
    try:
        val, err = integrate_adaptive(f, a, b)
        ok = abs(val - want) < tol
        passed += ok; failed += (not ok)
        print(f"  {chr(10003) if ok else chr(10007)} {tag}")
        if not ok:
            print(f"      got={val:.12g}  want={want:.12g}  diff={abs(val-want):.2e}")
    except Exception as e:
        errors += 1; print(f"  ! {tag}  ERROR: {e}")
pi = math.pi
e  = math.e
t0 = time.perf_counter()
# =============================================================
# HARD LIMITS (7) — 3rd/4th order cancellations, compositions
# =============================================================
print("=" * 60)
print("HARD LIMITS")
print("=" * 60)
# Requires correct 3rd-order Taylor of atan
check("L08  lim x->0 (atan(x) - x) / x^3 = -1/3",
      limit(lambda x: (atan(x) - x) / (x**3), as_x_to=0),
      -1.0/3)
# Hyperbolic 3rd order: x*cosh - sinh = x^3/3 + ...
check("L09  lim x->0 (x*cosh(x) - sinh(x)) / x^3 = 1/3",
      limit(lambda x: (x*cosh(x) - sinh(x)) / (x**3), as_x_to=0),
      1.0/3)
# 4th order exponential remainder
check("L10  lim x->0 (e^x - 1 - x - x^2/2) / x^3 = 1/6",
      limit(lambda x: (exp(x) - R(1) - x - x**2/R(2)) / (x**3),
            as_x_to=0),
      1.0/6)
# Ratio of two 3rd-order quantities
# 2sin(x)-sin(2x) = x^3 + ..., x-sin(x) = x^3/6 + ...
check("L11  lim x->0 (2sin(x)-sin(2x))/(x-sin(x)) = 6",
      limit(lambda x: (R(2)*sin(x) - sin(R(2)*x)) / (x - sin(x)),
            as_x_to=0),
      6.0)
# Both numerator and denominator vanish to 3rd order
# x-sin = x^3/6, x-tan = -x^3/3  =>  ratio = -1/2
check("L12  lim x->0 (x-sin(x))/(x-tan(x)) = -1/2",
      limit(lambda x: (x - sin(x)) / (x - tan(x)), as_x_to=0),
      -0.5)
# Hyperbolic difference: sinh - tanh = x^3/2 + ...
check("L13  lim x->0 (sinh(x)-tanh(x)) / x^3 = 1/2",
      limit(lambda x: (sinh(x) - tanh(x)) / (x**3), as_x_to=0),
      0.5)
# Nested composition: cos(sin(x)) - cos(x) = x^4/6 + ...
check("L14  lim x->0 (cos(sin(x))-cos(x)) / x^4 = 1/6",
      limit(lambda x: (cos(sin(x)) - cos(x)) / (x**4), as_x_to=0),
      1.0/6)
# =============================================================
# HARD DERIVATIVES (6) — deep chains, high-order extraction
# =============================================================
print("\n" + "=" * 60)
print("HARD DERIVATIVES")
print("=" * 60)
# Chain: d/dx[ln(sin x)] = cot(x).  At pi/4: cot = 1
check("D08  d/dx[ln(sin x)] at pi/4 = cot(pi/4) = 1",
      derivative(lambda x: ln(sin(x)), at=pi/4),
      1.0)
# Double chain: d/dx[atan(e^x)] = e^x/(1+e^2x).  At 0: 1/2
check("D09  d/dx[atan(e^x)] at 0 = 1/2",
      derivative(lambda x: atan(exp(x)), at=0),
      0.5)
# Second deriv of composition:
# f=e^(cos x), f''= e^(cos x)(sin^2 x - cos x). At 0: e*(0-1)=-e
check("D10  d^2/dx^2[e^(cos x)] at 0 = -e",
      nth_derivative(lambda x: exp(cos(x)), n=2, at=0),
      -e)
# Leibniz 4th order: d^4/dx^4[x^4 * e^x] at 0 = 24
check("D11  d^4/dx^4[x^4 * e^x] at 0 = 24",
      nth_derivative(lambda x: x**4 * exp(x), n=4, at=0),
      24.0)
# 6th derivative extraction (cycle of sin): d^6[sin] = -sin
check("D12  d^6/dx^6[sin x] at pi/6 = -sin(pi/6) = -1/2",
      nth_derivative(lambda x: sin(x), n=6, at=pi/6),
      -0.5)
# Triple chain: d/dx[exp(x^2 * ln(x))] at 1 = 1
# f(x)=x^(x^2), f'= f*(2x*ln(x)+x). At x=1: 1*(0+1)=1
check("D13  d/dx[exp(x^2 * ln x)] at 1 = 1",
      derivative(lambda x: exp(x*x * ln(x)), at=1),
      1.0)
# =============================================================
# HARD INTEGRALS (7) — products, powers, compositions
# =============================================================
print("\n" + "=" * 60)
print("HARD INTEGRALS")
print("=" * 60)
# Double integration by parts: x^2*e^x
check_int("I07  int_0^1 x^2 * e^x dx = e - 2",
          lambda x: x**2 * exp(x), 0, 1,
          e - 2)
# Trig power: sin^2 x = (1-cos2x)/2
check_int("I08  int_0^(pi/2) sin^2(x) dx = pi/4",
          lambda x: sin(x) * sin(x), 0, pi/2,
          pi / 4)
# Odd trig power: sin^3 x
check_int("I09  int_0^(pi/2) sin^3(x) dx = 2/3",
          lambda x: sin(x)**3, 0, pi/2,
          2.0/3)
# Mixed exp-trig: e^(-x)*cos(x)
check_int("I10  int_0^1 e^(-x)*cos(x) dx = (1+e^-1*(sin1-cos1))/2",
          lambda x: exp(-x) * cos(x), 0, 1,
          (1.0 + math.exp(-1)*(math.sin(1)-math.cos(1))) / 2)
# Gaussian-style substitution: x*e^(-x^2)
check_int("I11  int_0^1 x*e^(-x^2) dx = (1-e^-1)/2",
          lambda x: x * exp(-(x*x)), 0, 1,
          (1.0 - math.exp(-1)) / 2)
# Double IBP: x^2*cos(x)
# = -sin(1) + 2cos(1)
check_int("I12  int_0^1 x^2*cos(x) dx = -sin1 + 2cos1",
          lambda x: x**2 * cos(x), 0, 1,
          -math.sin(1) + 2*math.cos(1))
# Wider rational integral: atan(2)
check_int("I13  int_0^2 1/(1+x^2) dx = atan(2)",
          lambda x: R(1) / (R(1) + x*x), 0, 2,
          math.atan(2))
# =============================================================
# SUMMARY
# =============================================================
total = passed + failed + errors
elapsed = time.perf_counter() - t0
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed} failed, {errors} errors")
print(f"Time: {elapsed:.2f}s")
print("=" * 60)
if passed == total:
    print("ALL 20 HARD EDGE CASES PASSED")
else:
    print(f"{total - passed} problem(s) need attention.")
sys.exit(0 if passed == total else 1)
