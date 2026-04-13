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
Comprehensive Limits Test Suite
================================
Tests limit computation via composite algebraic evaluation.

Covers:
  1. Algebraic 0/0 — L'Hopital class (sin(x)/x, (e^x-1)/x, ...)
  2. Higher-order cancellation — 3rd/4th order Taylor matching
  3. Polynomial/algebraic — factoring singularities
  4. General powers — (1+x)^(1/x) class
  5. Oscillatory — x*sin(1/x), x*cos(1/x), products, sums
  6. Monotonic bounded — atan(1/x), tanh(1/x) at boundaries
  7. Limits at infinity — rational, trig/x, bounded/x
  8. Directional limits — left/right, sign detection
  9. Hyperbolic — sinh, cosh, tanh cancellation forms
  10. Domain errors — ln(0), sqrt(0) class, detection
  11. Nothing (empty) propagation — nested transcendentals of infinity
  12. Division by nothing — 1/sin(1/x) class
  13. Compositions — mixed oscillatory/algebraic/exponential
  14. Edge cases — constant functions, identity, already defined

Run: python tests/test_limits.py
"""

import math
import sys
from typing import List, Tuple

from composite.composite_lib import (
    Composite, R, ZERO, INF,
    sin, cos, exp, ln, sqrt, tan,
    atan, asin, acos, sinh, cosh, tanh,
    derivative, limit,
    LimitDoesNotExistError, LimitUndecidableError, CompositionError,
)


# =============================================================================
# TEST FRAMEWORK (same pattern as test_standalone.py)
# =============================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details

class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []

    def add(self, name: str, passed: bool, details: str = ""):
        self.results.append(TestResult(name, passed, details))

    def assert_eq(self, name: str, actual, expected, tol=1e-10):
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            passed = abs(actual - expected) < tol
        else:
            passed = actual == expected
        self.add(name, passed, f"actual={actual}, expected={expected}")

    def assert_true(self, name: str, condition: bool, details: str = ""):
        self.add(name, condition, details)

    def assert_raises(self, name: str, exc_type, func, *args, **kwargs):
        exc_name = exc_type.__name__ if hasattr(exc_type, '__name__') else str(exc_type)
        try:
            func(*args, **kwargs)
            self.add(name, False, f"Expected {exc_name}, got no error")
        except exc_type:
            self.add(name, True)
        except Exception as e:
            self.add(name, False, f"Expected {exc_name}, got {type(e).__name__}: {e}")

    def assert_nothing(self, name: str, value):
        """Assert the result is the empty composite (nothing)."""
        is_empty = isinstance(value, Composite) and not value.c
        self.add(name, is_empty, f"actual={value}, expected=nothing")

    def report(self) -> Tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*70}")
        print(f"{self.name}: {passed}/{total} passed ({100*passed/total:.1f}%)")
        print(f"{'='*70}")
        for r in self.results:
            status = "\u2713" if r.passed else "\u2717"
            print(f"  {status} {r.name}")
            if not r.passed and r.details:
                print(f"      {r.details}")
        return passed, total


# =============================================================================
# 1. ALGEBRAIC 0/0 LIMITS (L'Hopital class)
# =============================================================================

def test_algebraic_0_over_0():
    suite = TestSuite("Algebraic 0/0 Limits")

    suite.assert_eq(
        "lim(x\u21920) sin(x)/x = 1",
        limit(lambda x: sin(x)/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) (e^x-1)/x = 1",
        limit(lambda x: (exp(x)-R(1))/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) (1-cos x)/x\u00b2 = 1/2",
        limit(lambda x: (R(1)-cos(x))/(x**2), 0), 0.5)

    suite.assert_eq(
        "lim(x\u21920) (e^x-1-x)/x\u00b2 = 1/2",
        limit(lambda x: (exp(x)-R(1)-x)/(x**2), 0), 0.5)

    suite.assert_eq(
        "lim(x\u21920) (e^x-1-x-x\u00b2/2)/x\u00b3 = 1/6",
        limit(lambda x: (exp(x)-R(1)-x-x**2/R(2))/(x**3), 0), 1.0/6)

    suite.assert_eq(
        "lim(x\u21920) ln(1+x)/x = 1",
        limit(lambda x: ln(R(1)+x)/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) sinh(x)/x = 1",
        limit(lambda x: sinh(x)/x, 0), 1.0)

    return suite.report()


# =============================================================================
# 2. HIGHER-ORDER CANCELLATION
# =============================================================================

def test_higher_order_cancellation():
    suite = TestSuite("Higher-Order Cancellation")

    suite.assert_eq(
        "lim(x\u21920) (sin x - x)/x\u00b3 = -1/6",
        limit(lambda x: (sin(x)-x)/(x**3), 0), -1.0/6)

    suite.assert_eq(
        "lim(x\u21920) (tan x - sin x)/x\u00b3 = 1/2",
        limit(lambda x: (tan(x)-sin(x))/(x**3), 0), 0.5)

    suite.assert_eq(
        "lim(x\u21920) (atan x - x)/x\u00b3 = -1/3",
        limit(lambda x: (atan(x)-x)/(x**3), 0), -1.0/3)

    suite.assert_eq(
        "lim(x\u21920) (cos x - 1 + x\u00b2/2)/x\u2074 = 1/24",
        limit(lambda x: (cos(x)-R(1)+x**2/R(2))/(x**4), 0), 1.0/24)

    suite.assert_eq(
        "lim(x\u21920) (2sin x - sin 2x)/(x - sin x) = 6",
        limit(lambda x: (R(2)*sin(x)-sin(R(2)*x))/(x-sin(x)), 0), 6.0)

    suite.assert_eq(
        "lim(x\u21920) (x - sin x)/(x - tan x) = -1/2",
        limit(lambda x: (x-sin(x))/(x-tan(x)), 0), -0.5, tol=1e-8)

    suite.assert_eq(
        "lim(x\u21920) (cos(sin x) - cos x)/x\u2074 = 1/6",
        limit(lambda x: (cos(sin(x))-cos(x))/(x**4), 0), 1.0/6)

    return suite.report()


# =============================================================================
# 3. POLYNOMIAL / ALGEBRAIC
# =============================================================================

def test_polynomial_algebraic():
    suite = TestSuite("Polynomial / Algebraic Limits")

    suite.assert_eq(
        "lim(x\u21922) (x\u00b2-4)/(x-2) = 4",
        limit(lambda x: (x**2-R(4))/(x-R(2)), 2), 4.0)

    suite.assert_eq(
        "lim(x\u21921) (x\u00b3-1)/(x-1) = 3",
        limit(lambda x: (x**3-R(1))/(x-R(1)), 1), 3.0)

    suite.assert_eq(
        "lim(x\u21920) (\u221a(1+x)-\u221a(1-x))/x = 1",
        limit(lambda x: (sqrt(R(1)+x)-sqrt(R(1)-x))/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) (x\u00b3+2x\u00b2)/(x\u00b2+x) = 0",
        limit(lambda x: (x**3+R(2)*x**2)/(x**2+x), 0), 0.0, tol=1e-8)

    return suite.report()


# =============================================================================
# 4. GENERAL POWER LIMITS
# =============================================================================

def test_general_powers():
    suite = TestSuite("General Power Limits")

    suite.assert_eq(
        "lim(x\u21920) (1+x)^(1/x) = e",
        limit(lambda x: (R(1)+x)**(R(1)/x), 0), math.e)

    suite.assert_eq(
        "lim(x\u21920) (1+2x)^(1/x) = e\u00b2",
        limit(lambda x: (R(1)+R(2)*x)**(R(1)/x), 0), math.e**2, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920) (1+x/3)^(1/x) = e^(1/3)",
        limit(lambda x: (R(1)+x/R(3))**(R(1)/x), 0), math.e**(1.0/3), tol=1e-6)

    return suite.report()


# =============================================================================
# 5. OSCILLATORY LIMITS
# =============================================================================

def test_oscillatory():
    suite = TestSuite("Oscillatory Limits")

    # --- x^n * bounded(1/x) -> 0 ---
    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x) = 0",
        limit(lambda x: x*sin(1/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7cos(1/x) = 0",
        limit(lambda x: x*cos(1/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b2\u00b7sin(1/x) = 0",
        limit(lambda x: x**2*sin(1/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b2\u00b7cos(1/x) = 0",
        limit(lambda x: x**2*cos(1/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x\u00b2) = 0",
        limit(lambda x: x*sin(R(1)/(x**2)), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b3\u00b7sin(1/x) = 0",
        limit(lambda x: x**3*sin(1/x), 0), 0.0)

    # --- Products of oscillatory ---
    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x)\u00b7cos(1/x) = 0",
        limit(lambda x: x*sin(1/x)*cos(1/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7(sin(1/x))\u00b2 = 0",
        limit(lambda x: x*sin(1/x)**2, 0), 0.0)

    # --- Oscillatory + constant ---
    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x) + 5 = 5",
        limit(lambda x: x*sin(1/x)+R(5), 0), 5.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x) + x = 0",
        limit(lambda x: x*sin(1/x)+x, 0), 0.0)

    # --- Oscillatory with addition inside ---
    suite.assert_eq(
        "lim(x\u21920) x\u00b7(sin(1/x)+1) = 0",
        limit(lambda x: x*(sin(1/x)+R(1)), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7(cos(1/x)+3) = 0",
        limit(lambda x: x*(cos(1/x)+R(3)), 0), 0.0)

    return suite.report()


# =============================================================================
# 6. MONOTONIC BOUNDED AT INFINITY
# =============================================================================

def test_monotonic_bounded():
    suite = TestSuite("Monotonic Bounded (atan, tanh)")

    suite.assert_eq(
        "lim(x\u21920+) atan(1/x) = \u03c0/2",
        limit(lambda x: atan(R(1)/x), 0, dir='+'), math.pi/2)

    suite.assert_eq(
        "lim(x\u21920-) atan(1/x) = -\u03c0/2",
        limit(lambda x: atan(R(1)/x), 0, dir='-'), -math.pi/2)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7atan(1/x) = 0",
        limit(lambda x: x*atan(R(1)/x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920+) tanh(1/x) = 1",
        limit(lambda x: tanh(R(1)/x), 0, dir='+'), 1.0)

    suite.assert_eq(
        "lim(x\u21920-) tanh(1/x) = -1",
        limit(lambda x: tanh(R(1)/x), 0, dir='-'), -1.0)

    return suite.report()


# =============================================================================
# 7. LIMITS AT INFINITY
# =============================================================================

def test_limits_at_infinity():
    suite = TestSuite("Limits at Infinity")

    suite.assert_eq(
        "lim(x\u2192\u221e) sin(x)/x = 0",
        limit(lambda x: sin(x)/x, INF), 0.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) cos(x)/x = 0",
        limit(lambda x: cos(x)/x, INF), 0.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) 1/x = 0",
        limit(lambda x: R(1)/x, INF), 0.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) (5x\u00b2+3x)/(2x\u00b2+1) = 5/2",
        limit(lambda x: (R(5)*x**2+R(3)*x)/(R(2)*x**2+R(1)), INF), 2.5)

    suite.assert_eq(
        "lim(x\u2192\u221e) x/(x+1) = 1",
        limit(lambda x: x/(x+R(1)), INF), 1.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) (3x+1)/(2x-1) = 3/2",
        limit(lambda x: (R(3)*x+R(1))/(R(2)*x-R(1)), INF), 1.5)

    suite.assert_eq(
        "lim(x\u2192\u221e) sin(x)\u00b7cos(x)/x = 0",
        limit(lambda x: sin(x)*cos(x)/x, INF), 0.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) atan(x)/x = 0",
        limit(lambda x: atan(x)/x, INF), 0.0)

    # sin(1/x) as x->inf: 1/x -> 0, sin(small) ~ small, sin(1/x)/1/x -> 1
    suite.assert_eq(
        "lim(x\u2192\u221e) x\u00b7sin(1/x) = 1",
        limit(lambda x: x*sin(R(1)/x), INF), 1.0, tol=1e-6)

    return suite.report()


# =============================================================================
# 8. DIRECTIONAL LIMITS
# =============================================================================

def test_directional():
    suite = TestSuite("Directional Limits")

    r_pos = limit(lambda x: R(1)/x, 0, dir='+')
    suite.assert_true(
        "lim(x\u21920+) 1/x = +\u221e",
        isinstance(r_pos, Composite) and r_pos.max_positive_dim() is not None)

    r_neg = limit(lambda x: R(1)/x, 0, dir='-')
    suite.assert_true(
        "lim(x\u21920-) 1/x = -\u221e",
        isinstance(r_neg, Composite) and r_neg.max_positive_dim() is not None)

    suite.assert_eq(
        "lim(x\u21920+) x/x = 1",
        limit(lambda x: x/x, 0, dir='+'), 1.0)

    suite.assert_eq(
        "lim(x\u21920-) x/x = 1",
        limit(lambda x: x/x, 0, dir='-'), 1.0)

    return suite.report()


# =============================================================================
# 9. HYPERBOLIC LIMITS
# =============================================================================

def test_hyperbolic():
    suite = TestSuite("Hyperbolic Limits")

    suite.assert_eq(
        "lim(x\u21920) (x\u00b7cosh x - sinh x)/x\u00b3 = 1/3",
        limit(lambda x: (x*cosh(x)-sinh(x))/(x**3), 0), 1.0/3, tol=1e-8)

    suite.assert_eq(
        "lim(x\u21920) (sinh x - tanh x)/x\u00b3 = 1/2",
        limit(lambda x: (sinh(x)-tanh(x))/(x**3), 0), 0.5)

    suite.assert_eq(
        "lim(x\u21920) sinh(x)/x = 1",
        limit(lambda x: sinh(x)/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) (cosh x - 1)/x\u00b2 = 1/2",
        limit(lambda x: (cosh(x)-R(1))/(x**2), 0), 0.5, tol=1e-8)

    return suite.report()


# =============================================================================
# 10. DOMAIN ERRORS — these limits exist but composite can't compute them yet
# =============================================================================

def test_domain_errors():
    suite = TestSuite("Domain Errors (ln, sqrt at 0) — MUST FIX")

    # These limits are well-defined. A production tool must compute them,
    # not raise errors. Tests are written with correct expected values.
    # Failures here show exactly what needs fixing.

    suite.assert_eq(
        "lim(x\u21920+) x\u00b7ln(x) = 0",
        _safe_limit(lambda x: x*ln(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) x^x = 1",
        _safe_limit(lambda x: x**x, 0, dir='+'), 1.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) x^(sin x) = 1",
        _safe_limit(lambda x: x**(sin(x)), 0, dir='+'), 1.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) sqrt(x) = 0",
        _safe_limit(lambda x: sqrt(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) sqrt(x)\u00b7sin(x) = 0",
        _safe_limit(lambda x: sqrt(x)*sin(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) x\u00b7sqrt(x) = 0",
        _safe_limit(lambda x: x*sqrt(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) sqrt(x)\u00b7ln(x) = 0",
        _safe_limit(lambda x: sqrt(x)*ln(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) x\u00b2\u00b7ln(x) = 0",
        _safe_limit(lambda x: x**2*ln(x), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) exp(-1/x\u00b2) = 0",
        _safe_limit(lambda x: exp(R(-1)/(x**2)), 0, dir='+'), 0.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920+) x\u00b7exp(-1/x) = 0",
        _safe_limit(lambda x: x*exp(R(-1)/x), 0, dir='+'), 0.0, tol=1e-6)

    return suite.report()


def _safe_limit(f, at, dir='both', tol=1e-6):
    """Try algebraic limit; return NaN on error so the test fails visibly."""
    try:
        return limit(f, at, dir=dir)
    except (LimitUndecidableError, CompositionError, ValueError,
            ZeroDivisionError, OverflowError):
        return float('nan')


# =============================================================================
# 11. NOTHING (EMPTY COMPOSITE) PROPAGATION
# =============================================================================

def test_nothing_propagation():
    suite = TestSuite("Nothing (\u2205) Propagation")

    # sin(INF) and cos(INF) should return nothing
    suite.assert_nothing("sin(INF) = \u2205", sin(INF))
    suite.assert_nothing("cos(INF) = \u2205", cos(INF))
    suite.assert_nothing("sin(-INF) = \u2205", sin(-INF))
    suite.assert_nothing("cos(-INF) = \u2205", cos(-INF))

    # Monotonic bounded should NOT return nothing
    suite.assert_true(
        "atan(INF) = \u03c0/2 (not nothing)",
        abs(atan(INF).st() - math.pi/2) < 1e-10)
    suite.assert_true(
        "tanh(INF) = 1 (not nothing)",
        abs(tanh(INF).st() - 1.0) < 1e-10)

    # Multiplication: a * nothing = nothing
    nothing = Composite({})
    suite.assert_nothing("ZERO \u00d7 \u2205 = \u2205", ZERO * nothing)
    suite.assert_nothing("R(5) \u00d7 \u2205 = \u2205", R(5) * nothing)
    suite.assert_nothing("INF \u00d7 \u2205 = \u2205", INF * nothing)
    suite.assert_nothing("\u2205 \u00d7 \u2205 = \u2205", nothing * nothing)

    # Addition: a + nothing = a
    suite.assert_eq(
        "R(5) + \u2205 = R(5)",
        (R(5) + nothing).st(), 5.0)
    suite.assert_eq(
        "ZERO + \u2205 = ZERO",
        (ZERO + nothing).coeff(-1), 1.0)

    # Nested: sin(nothing) — nothing has no positive dims, treated as sin(0)
    # Nothing in, nothing out — all transcendentals propagate ∅
    suite.assert_nothing("sin(\u2205) = \u2205", sin(nothing))
    suite.assert_nothing("cos(\u2205) = \u2205", cos(nothing))
    suite.assert_nothing("exp(\u2205) = \u2205", exp(nothing))
    suite.assert_nothing("ln(\u2205) = \u2205", ln(nothing))
    suite.assert_nothing("tan(\u2205) = \u2205", tan(nothing))
    suite.assert_nothing("atan(\u2205) = \u2205", atan(nothing))
    suite.assert_nothing("sqrt(\u2205) = \u2205", sqrt(nothing))

    return suite.report()


# =============================================================================
# 12. DIVISION BY NOTHING
# =============================================================================

def test_division_by_nothing():
    suite = TestSuite("Division by Nothing")

    # 1/sin(1/x): sin(INF)=∅, 1/∅=∅. Probes at real points see
    # oscillating values → extrapolation doesn't converge → DNE.
    suite.assert_raises(
        "1/sin(1/x) at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: R(1)/sin(1/x), 0)

    # x/sin(1/x): ZERO / sin(INF) = ZERO / ∅ → division by nothing
    # raises LimitDoesNotExistError directly. No extrapolation needed.
    suite.assert_raises(
        "x/sin(1/x) at 0 → DNE (division by nothing)",
        LimitDoesNotExistError,
        limit, lambda x: x/sin(1/x), 0)

    return suite.report()


# =============================================================================
# 13. COMPOSITIONS
# =============================================================================

def test_compositions():
    suite = TestSuite("Compositions")

    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x)\u00b7exp(x) = 0",
        limit(lambda x: x*sin(1/x)*exp(x), 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) (x\u00b7sin(1/x))\u00b2 = 0",
        limit(lambda x: (x*sin(1/x))**2, 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) sin(sin(x))/x = 1",
        limit(lambda x: sin(sin(x))/x, 0), 1.0, tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920) sin(cos(x)) = sin(1)",
        limit(lambda x: sin(cos(x)), 0), math.sin(1.0), tol=1e-6)

    suite.assert_eq(
        "lim(x\u21920) x\u00b2\u00b7sin(1/x) + x = 0",
        limit(lambda x: x**2*sin(1/x)+x, 0), 0.0)

    suite.assert_eq(
        "lim(x\u21920) x\u00b7sin(1/x) + x\u00b7cos(1/x) = 0",
        limit(lambda x: x*sin(1/x)+x*cos(1/x), 0), 0.0)

    return suite.report()


# =============================================================================
# 14. EDGE CASES
# =============================================================================

def test_edge_cases():
    suite = TestSuite("Edge Cases")

    # Constant function
    suite.assert_eq(
        "lim(x\u21920) 5 = 5",
        limit(lambda x: R(5), 0), 5.0)

    suite.assert_eq(
        "lim(x\u2192\u221e) 5 = 5",
        limit(lambda x: R(5), INF), 5.0)

    # Identity
    suite.assert_eq(
        "lim(x\u21923) x = 3",
        limit(lambda x: x, 3), 3.0)

    # Already defined at the point
    suite.assert_eq(
        "lim(x\u21922) x\u00b2 = 4",
        limit(lambda x: x**2, 2), 4.0)

    suite.assert_eq(
        "lim(x\u2192\u03c0/2) sin(x) = 1",
        limit(lambda x: sin(x), math.pi/2), 1.0, tol=1e-6)

    # Nested compositions without singularity
    suite.assert_eq(
        "lim(x\u21920) sin(sin(x))/sin(x) = 1",
        limit(lambda x: sin(sin(x))/sin(x), 0), 1.0, tol=1e-6)

    # asin/acos at regular points
    suite.assert_eq(
        "lim(x\u21920) asin(x)/x = 1",
        limit(lambda x: asin(x)/x, 0), 1.0)

    suite.assert_eq(
        "lim(x\u21920) acos(x) = \u03c0/2",
        limit(lambda x: acos(x), 0), math.pi/2)

    return suite.report()


# =============================================================================
# 15. NESTED OSCILLATORY — nothing must propagate through all functions
# =============================================================================

def test_nested_oscillatory():
    """Tests for nested transcendentals of infinity.

    All these limits do not exist. Nothing propagates through
    transcendentals, the limit function detects ∅, probes at real
    points, finds oscillation → LimitDoesNotExistError.
    """
    suite = TestSuite("Nested Oscillatory (DNE)")

    suite.assert_raises(
        "sin(sin(1/x)) at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: sin(sin(1/x)), 0)

    suite.assert_raises(
        "cos(sin(1/x)) at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: cos(sin(1/x)), 0)

    suite.assert_raises(
        "exp(sin(1/x)) at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: exp(sin(1/x)), 0)

    suite.assert_raises(
        "sin(1/x)\u00b2 at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: sin(1/x)**2, 0)

    suite.assert_raises(
        "atan(sin(1/x)) at 0 → DNE",
        LimitDoesNotExistError,
        limit, lambda x: atan(sin(1/x)), 0)

    return suite.report()


def _safe_eval(f):
    """Evaluate, return result or NaN-composite on error."""
    try:
        return f()
    except Exception:
        return Composite({0: float('nan')})


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_tests():
    print("\n" + "#" * 70)
    print("# COMPREHENSIVE LIMITS TEST SUITE")
    print("# Composite Machine — Algebraic Limit Computation")
    print("#" * 70)

    all_results = []

    all_results.append(("Algebraic 0/0", test_algebraic_0_over_0()))
    all_results.append(("Higher-Order Cancellation", test_higher_order_cancellation()))
    all_results.append(("Polynomial / Algebraic", test_polynomial_algebraic()))
    all_results.append(("General Powers", test_general_powers()))
    all_results.append(("Oscillatory", test_oscillatory()))
    all_results.append(("Monotonic Bounded", test_monotonic_bounded()))
    all_results.append(("Limits at Infinity", test_limits_at_infinity()))
    all_results.append(("Directional", test_directional()))
    all_results.append(("Hyperbolic", test_hyperbolic()))
    all_results.append(("Domain Errors", test_domain_errors()))
    all_results.append(("Nothing Propagation", test_nothing_propagation()))
    all_results.append(("Division by Nothing", test_division_by_nothing()))
    all_results.append(("Compositions", test_compositions()))
    all_results.append(("Edge Cases", test_edge_cases()))
    all_results.append(("Nested Oscillatory (known)", test_nested_oscillatory()))

    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)

    total_passed = sum(r[0] for _, r in all_results)
    total_tests = sum(r[1] for _, r in all_results)

    print(f"\n{'Suite':<40} {'Passed':<10} {'Total':<10} {'Rate'}")
    print("-" * 70)
    for name, (passed, total) in all_results:
        rate = f"{100*passed/total:.0f}%" if total > 0 else "N/A"
        status = "\u2713" if passed == total else "\u2717"
        print(f"{status} {name:<38} {passed:<10} {total:<10} {rate}")

    print("-" * 70)
    print(f"{'TOTAL':<40} {total_passed:<10} {total_tests:<10} {100*total_passed/total_tests:.1f}%")
    print()

    if total_passed == total_tests:
        print(" ALL TESTS PASSED!")
    else:
        print(f"\u26a0\ufe0f  {total_tests - total_passed} test(s) failed.")
        print("Review failed tests above for details.")

    print("\n" + "#" * 70)

    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
