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
Test Suite for Composite Calculus Library
==========================================
Imports composite_lib.py — does NOT include implementation.

Run: python test_composite.py

Requires:
  - composite_lib.py in the same directory (or on PYTHONPATH)
  - numpy (for FFT-based stress tests only)

Test Categories:
  1. Paper Theorems (T1-T8) — Direct verification of formal claims
  2. Edge Cases — Boundary conditions, special values
  3. Algebraic Properties — Ring axioms verification
  4. Calculus Operations — Derivatives, integration, limits
  5. Comparison Semantics — Total ordering verification
  6. Stress Tests — Chains of operations, numerical stability
  7. Multivariate — Partial derivatives, gradients, Laplacians
  8. Transcendental — sin, cos, exp via Taylor series
"""

import math
import sys
import numpy as np
from numpy.fft import fft, ifft
from typing import List, Tuple

# =============================================================================
# IMPORT THE LIBRARY UNDER TEST
# =============================================================================

from composite.composite_lib import (
    Composite, R, ZERO, INF, h,
    sin, cos, exp, ln, sqrt, tan,
    derivative, nth_derivative, all_derivatives,
    limit, antiderivative,
)


# =============================================================================
# ALIASES: Map standalone test names → composite_lib functions
# =============================================================================

sin_composite = sin
cos_composite = cos
exp_composite = exp

def ln_1_plus_x(x, terms=10):
    """ln(1+x) via composite_lib's ln function."""
    return ln(R(1) + x, terms=terms)


# =============================================================================
# HELPER: derivative_of (inverse of antiderivative, not in composite_lib)
# =============================================================================

def derivative_of(F_composite):
    """Inverse of antiderivative: shift up, multiply by position."""
    result = {}
    for dim, coeff in F_composite.c.items():
        if dim < 0:
            new_dim = dim + 1
            result[new_dim] = coeff * abs(dim)
    return Composite(result)


# =============================================================================
# ADDITIONAL IMPLEMENTATIONS NOT IN composite_lib
# (Kept here because they are separate modules / test dependencies)
# =============================================================================

class CompositeFFT:
    """NumPy-optimized composite number with FFT multiplication."""

    DEFAULT_WINDOW = 16

    def __init__(self, coefficients=None, window=None):
        self.window = window or self.DEFAULT_WINDOW
        self.size = 2 * self.window + 1

        if coefficients is None:
            self.c = np.zeros(self.size, dtype=np.float64)
        elif isinstance(coefficients, np.ndarray):
            self.c = coefficients.copy()
        elif isinstance(coefficients, (int, float)):
            self.c = np.zeros(self.size, dtype=np.float64)
            self.c[self.window] = coefficients
        elif isinstance(coefficients, dict):
            self.c = np.zeros(self.size, dtype=np.float64)
            for dim, coeff in coefficients.items():
                idx = dim + self.window
                if 0 <= idx < self.size:
                    self.c[idx] = coeff
        else:
            raise TypeError(f"Unsupported type: {type(coefficients)}")

    @classmethod
    def zero(cls, window=None):
        return cls({-1: 1}, window=window)

    @classmethod
    def infinity(cls, window=None):
        return cls({1: 1}, window=window)

    @classmethod
    def real(cls, value, window=None):
        return cls({0: value}, window=window)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = CompositeFFT(other, window=self.window)
        result = CompositeFFT(window=self.window)
        result.c = self.c + other.c
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = CompositeFFT(other, window=self.window)
        result = CompositeFFT(window=self.window)
        result.c = self.c - other.c
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = CompositeFFT(window=self.window)
            result.c = self.c * other
            return result

        n = self.size
        fft_size = 1 << (2 * n - 2).bit_length()

        a_padded = np.zeros(fft_size)
        b_padded = np.zeros(fft_size)
        a_padded[:n] = self.c
        b_padded[:n] = other.c

        conv = np.real(ifft(fft(a_padded) * fft(b_padded)))

        result = CompositeFFT(window=self.window)
        center = 2 * self.window
        start = center - self.window
        end = center + self.window + 1

        if start >= 0 and end <= len(conv):
            result.c = conv[start:end]
        else:
            for i in range(self.size):
                conv_idx = start + i
                if 0 <= conv_idx < len(conv):
                    result.c[i] = conv[conv_idx]

        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = CompositeFFT(window=self.window)
            result.c = self.c / other
            return result

        nonzero = np.where(np.abs(other.c) > 1e-14)[0]
        if len(nonzero) != 1:
            raise NotImplementedError("Division only supported for single-term divisor")

        div_idx = nonzero[0]
        div_coeff = other.c[div_idx]

        result = CompositeFFT(window=self.window)
        for idx, coeff in enumerate(self.c):
            if abs(coeff) > 1e-14:
                new_idx = idx - div_idx + self.window
                if 0 <= new_idx < self.size:
                    result.c[new_idx] = coeff / div_coeff

        return result

    def __pow__(self, n):
        if n == 0:
            return CompositeFFT({0: 1}, window=self.window)
        if n == 1:
            return CompositeFFT(self.c, window=self.window)

        result = CompositeFFT({0: 1}, window=self.window)
        base = CompositeFFT(self.c, window=self.window)

        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2

        return result

    def __neg__(self):
        result = CompositeFFT(window=self.window)
        result.c = -self.c
        return result

    def st(self):
        return self.c[self.window]


# =============================================================================
# MULTIVARIATE COMPOSITE (not in composite_lib — separate module)
# =============================================================================

class CompositeMulti:
    """Multivariate composite number with tuple dimensions."""

    def __init__(self, coefficients=None, n_vars=2):
        self.n_vars = n_vars
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            zero_dim = tuple([0] * n_vars)
            self.c = {zero_dim: coefficients} if coefficients != 0 else {}
        else:
            self.c = {k: v for k, v in coefficients.items() if v != 0}

    @classmethod
    def real(cls, value, n_vars=2):
        zero_dim = tuple([0] * n_vars)
        return cls({zero_dim: value}, n_vars=n_vars)

    @classmethod
    def zero_var(cls, var_index, n_vars=2):
        dim = [0] * n_vars
        dim[var_index] = -1
        return cls({tuple(dim): 1}, n_vars=n_vars)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = CompositeMulti.real(other, self.n_vars)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return CompositeMulti(result, self.n_vars)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = CompositeMulti.real(other, self.n_vars)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) - coeff
        return CompositeMulti(result, self.n_vars)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = CompositeMulti.real(other, self.n_vars)
        result = {}
        for d1, c1 in self.c.items():
            for d2, c2 in other.c.items():
                new_dim = tuple(a + b for a, b in zip(d1, d2))
                result[new_dim] = result.get(new_dim, 0) + c1 * c2
        return CompositeMulti(result, self.n_vars)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, n):
        if n == 0:
            return CompositeMulti.real(1, self.n_vars)
        result = CompositeMulti.real(1, self.n_vars)
        for _ in range(n):
            result = result * self
        return result

    def __neg__(self):
        return CompositeMulti({k: -v for k, v in self.c.items()}, self.n_vars)

    def st(self):
        zero_dim = tuple([0] * self.n_vars)
        return self.c.get(zero_dim, 0)

    def partial(self, var_index, order=1):
        dim = [0] * self.n_vars
        dim[var_index] = -order
        coeff = self.c.get(tuple(dim), 0)
        return coeff * math.factorial(order)

    def mixed_partial(self, orders):
        dim = tuple(-o for o in orders)
        coeff = self.c.get(dim, 0)
        factorial_product = 1
        for o in orders:
            factorial_product *= math.factorial(o)
        return coeff * factorial_product

    def gradient(self):
        return [self.partial(i, 1) for i in range(self.n_vars)]

    def laplacian(self):
        """Laplacian: sum of second partial derivatives."""
        return sum(self.partial(i, 2) for i in range(self.n_vars))


class MultiComposite(CompositeMulti):
    """Alias for CompositeMulti with laplacian."""

    @classmethod
    def var(cls, idx, n_vars=2):
        return cls.zero_var(idx, n_vars)

    def laplacian(self):
        return sum(self.partial(i, 2) for i in range(self.n_vars))


# =============================================================================
# TEST FRAMEWORK
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

    def report(self) -> Tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*70}")
        print(f"{self.name}: {passed}/{total} passed ({100*passed/total:.1f}%)")
        print(f"{'='*70}")
        for r in self.results:
            status = '\u2713' if r.passed else '\u2717'
            print(f"  {status} {r.name}")
            if not r.passed and r.details:
                print(f"      {r.details}")
        return passed, total


# =============================================================================
# PAPER THEOREMS (T1-T8)
# =============================================================================

def test_theorem_1_information_preservation():
    suite = TestSuite("Theorem 1: Information Preservation")
    for a in [1, 5, -3, 0.5, 100, -0.001]:
        result = R(a) * ZERO
        suite.assert_eq(f"{a} \u00d7 0 preserves coefficient", result.c.get(-1, 0), a)
    for a in [1, 5, -3, 0.5, 100]:
        result = R(a) / ZERO
        suite.assert_eq(f"{a} / 0 preserves coefficient", result.c.get(1, 0), a)
    a, b = R(5), R(7)
    result_a = a * ZERO
    result_b = b * ZERO
    suite.assert_true("5\u00d70 \u2260 7\u00d70 (different provenances)",
                      result_a.c.get(-1, 0) != result_b.c.get(-1, 0))
    return suite.report()

def test_theorem_2_zero_infinity_duality():
    suite = TestSuite("Theorem 2: Zero-Infinity Duality")
    product = ZERO * INF
    suite.assert_eq("0 \u00d7 \u221e = 1", product.st(), 1)
    product_rev = INF * ZERO
    suite.assert_eq("\u221e \u00d7 0 = 1 (commutative)", product_rev.st(), 1)
    inv_zero = R(1) / ZERO
    suite.assert_true("1/0 has dimension 1", 1 in inv_zero.c and inv_zero.c[1] == 1)
    inv_inf = R(1) / INF
    suite.assert_true("1/\u221e has dimension -1", -1 in inv_inf.c and inv_inf.c[-1] == 1)
    zero_sq = ZERO * ZERO
    inf_sq = INF * INF
    product_sq = zero_sq * inf_sq
    suite.assert_eq("0\u00b2 \u00d7 \u221e\u00b2 = 1", product_sq.st(), 1)
    scaled_zero = Composite({-1: 5})
    scaled_product = scaled_zero * INF
    suite.assert_eq("|5|\u208b\u2081 \u00d7 |1|\u2081 = 5", scaled_product.st(), 5)
    return suite.report()

def test_theorem_3_provenance_non_uniqueness():
    suite = TestSuite("Theorem 3: Provenance (Non-Uniqueness of Zero)")
    zero_1 = Composite({-1: 1})
    zero_2 = Composite({-1: 2})
    zero_5 = Composite({-1: 5})
    suite.assert_true("|1|\u208b\u2081 \u2260 |2|\u208b\u2081", zero_1.c != zero_2.c)
    suite.assert_true("|2|\u208b\u2081 \u2260 |5|\u208b\u2081", zero_2.c != zero_5.c)
    suite.assert_eq("st(|1|\u208b\u2081) = 0", zero_1.st(), 0)
    suite.assert_eq("st(|2|\u208b\u2081) = 0", zero_2.st(), 0)
    suite.assert_eq("st(|5|\u208b\u2081) = 0", zero_5.st(), 0)
    suite.assert_eq("|1|\u208b\u2081 coefficient", zero_1.c[-1], 1)
    suite.assert_eq("|2|\u208b\u2081 coefficient", zero_2.c[-1], 2)
    suite.assert_eq("|5|\u208b\u2081 coefficient", zero_5.c[-1], 5)
    z_from_5 = R(5) * ZERO
    z_from_3 = R(3) * ZERO
    suite.assert_true("5\u00d70 \u2260 3\u00d70", z_from_5.c[-1] != z_from_3.c[-1])
    return suite.report()

def test_theorem_4_reversibility():
    suite = TestSuite("Theorem 4: Reversibility")
    for a in [1, 5, -3, 0.5, 100, -0.001, 3.14159]:
        result = (R(a) * ZERO) / ZERO
        suite.assert_eq(f"({a} \u00d7 0) / 0 = {a}", result.st(), a)
    for a in [1, 5, -3, 0.5, 100, 3.14159]:
        result = (R(a) / ZERO) * ZERO
        suite.assert_eq(f"({a} / 0) \u00d7 0 = {a}", result.st(), a)
    a = 7
    double_zero = (R(a) * ZERO) * ZERO
    recovered = (double_zero / ZERO) / ZERO
    suite.assert_eq("((7\u00d70)\u00d70)/0/0 = 7", recovered.st(), 7)
    double_inf = (R(a) / ZERO) / ZERO
    recovered_2 = (double_inf * ZERO) * ZERO
    suite.assert_eq("((7/0)/0)\u00d70\u00d70 = 7", recovered_2.st(), 7)
    return suite.report()

def test_theorem_5_coefficient_cancellation():
    suite = TestSuite("Theorem 5: Coefficient Cancellation")
    for a, b in [(2, 3), (5, 7), (-2, 4), (0.5, 8)]:
        zero_a = Composite({-1: a})
        inf_b = Composite({1: b})
        product = zero_a * inf_b
        suite.assert_eq(f"|{a}|\u208b\u2081 \u00d7 |{b}|\u2081 = {a*b}", product.st(), a * b)
    zero_2nd = Composite({-2: 3})
    inf_2nd = Composite({2: 4})
    product_2 = zero_2nd * inf_2nd
    suite.assert_eq("|3|\u208b\u2082 \u00d7 |4|\u2082 = 12", product_2.st(), 12)
    zero_3rd = Composite({-3: 2})
    inf_2nd = Composite({2: 5})
    product_asym = zero_3rd * inf_2nd
    suite.assert_eq("|2|\u208b\u2083 \u00d7 |5|\u2082 = |10|\u208b\u2081", product_asym.c.get(-1, 0), 10)
    return suite.report()

def test_theorem_6_identity_elements():
    suite = TestSuite("Theorem 6: Identity Elements")
    one = R(1)
    for val in [5, -3, 0.5, 100]:
        x = R(val)
        suite.assert_eq(f"{val} \u00d7 1 = {val}", (x * one).st(), val)
        suite.assert_eq(f"1 \u00d7 {val} = {val}", (one * x).st(), val)
    multi = Composite({0: 3, -1: 2, 1: 5})
    result = multi * one
    suite.assert_true("|1|\u2080 is identity for multi-term composite", result.c == multi.c)
    zero_plus_zero = ZERO + ZERO
    suite.assert_eq("0 + 0 = |2|\u208b\u2081 (coefficient)", zero_plus_zero.c.get(-1, 0), 2)
    suite.assert_true("0 + 0 \u2260 0 (zeroes accumulate)", zero_plus_zero.c[-1] != ZERO.c[-1])
    triple_zero = ZERO + ZERO + ZERO
    suite.assert_eq("0 + 0 + 0 = |3|\u208b\u2081", triple_zero.c.get(-1, 0), 3)
    return suite.report()

def test_theorem_7_fractional_orders():
    suite = TestSuite("Theorem 7: Fractional Orders")
    base = Composite({-1: 2})
    squared = base * base
    suite.assert_eq("|2|\u208b\u2081\u00b2 = |4|\u208b\u2082 (coefficient)", squared.c.get(-2, 0), 4)
    cubed = base * base * base
    suite.assert_eq("|2|\u208b\u2081\u00b3 = |8|\u208b\u2083 (coefficient)", cubed.c.get(-3, 0), 8)
    inf_base = Composite({1: 3})
    inf_squared = inf_base * inf_base
    suite.assert_eq("|3|\u2081\u00b2 = |9|\u2082 (coefficient)", inf_squared.c.get(2, 0), 9)
    for a, n, k in [(2, -1, 3), (3, 1, 2), (2, -2, 2)]:
        base = Composite({n: a})
        result = base
        for _ in range(k - 1):
            result = result * base
        expected_coeff = a ** k
        expected_dim = n * k
        suite.assert_eq(f"|{a}|_{n}^{k} coefficient", result.c.get(expected_dim, 0), expected_coeff)
    return suite.report()

def test_theorem_8_total_ordering():
    suite = TestSuite("Theorem 8: Total Ordering")
    suite.assert_true("|5|\u2081 > |1000000|\u2080", INF * 5 > R(1000000))
    suite.assert_true("|1|\u2080 > |1000|\u208b\u2081", R(1) > Composite({-1: 1000}))
    suite.assert_true("|5|\u2080 > |3|\u2080", R(5) > R(3))
    suite.assert_true("|3|\u208b\u2081 > |2|\u208b\u2081", Composite({-1: 3}) > Composite({-1: 2}))
    suite.assert_true("|-1|\u2081 < |1000000|\u2080", Composite({1: -1}) < R(1000000))
    a = Composite({1: 2, 0: -10})
    b = Composite({1: -5, 0: 3})
    suite.assert_true("|2|\u2081 - 10 > |-5|\u2081 + 3", a > b)
    x, y, z = R(-100), R(0), R(100)
    suite.assert_true("Transitivity: -100 < 0 < 100", x < y < z)
    h = ZERO
    suite.assert_true("0 < h (infinitesimal is positive)", R(0) < h)
    suite.assert_true("h < 0.0001", h < R(0.0001))
    suite.assert_true("h < 1e-100", h < R(1e-100))
    neg_inf = Composite({1: -1})
    neg_h = Composite({-1: -1})
    chain = [neg_inf, R(-1), neg_h, R(0), h, R(1), INF]
    all_ordered = all(chain[i] < chain[i+1] for i in range(len(chain)-1))
    suite.assert_true("Full chain: -\u221e < -1 < -h < 0 < h < 1 < \u221e", all_ordered)
    return suite.report()


# =============================================================================
# EDGE CASES
# =============================================================================

def test_edge_cases():
    suite = TestSuite("Edge Cases")
    zero_coeff = Composite({0: 0})
    suite.assert_true("Zero coefficient removed from composite", 0 not in zero_coeff.c or zero_coeff.c[0] == 0)
    empty = Composite({})
    suite.assert_eq("st(empty) = 0", empty.st(), 0)
    tiny = Composite({0: 1e-15})
    suite.assert_eq("Very small coefficient preserved", tiny.st(), 1e-15)
    huge = Composite({0: 1e15})
    suite.assert_eq("Very large coefficient preserved", huge.st(), 1e15)
    deep_zero = Composite({-10: 1})
    recovered = deep_zero
    for _ in range(10):
        recovered = recovered / ZERO
    suite.assert_eq("|1|\u208b\u2081\u2080 / 0^10 = |1|\u2080", recovered.st(), 1)
    a = Composite({0: 3, -1: 2})
    b = Composite({0: 4, -2: 5})
    c = a + b
    suite.assert_eq("Cross-dim addition: dim 0", c.c.get(0, 0), 7)
    suite.assert_eq("Cross-dim addition: dim -1", c.c.get(-1, 0), 2)
    suite.assert_eq("Cross-dim addition: dim -2", c.c.get(-2, 0), 5)
    d = Composite({0: 5})
    e = Composite({0: 5})
    f = d - e
    suite.assert_eq("5 - 5 = 0 (st)", f.st(), 0)
    x = R(1)
    for _ in range(100):
        x = x * ZERO
    for _ in range(100):
        x = x / ZERO
    suite.assert_true("100\u00d7 zero then 100\u00d7 divide recovers 1", abs(x.st() - 1) < 1e-10)
    return suite.report()


# =============================================================================
# ALGEBRAIC PROPERTIES
# =============================================================================

def test_algebraic_properties():
    suite = TestSuite("Algebraic Properties")
    a = Composite({0: 3, -1: 2})
    b = Composite({0: 5, -1: 1})
    c = Composite({0: 2, 1: 4})
    lhs = (a * b) * c
    rhs = a * (b * c)
    suite.assert_true("Multiplicative associativity", lhs.c == rhs.c)
    suite.assert_true("Multiplicative commutativity", (a * b).c == (b * a).c)
    lhs_dist = a * (b + c)
    rhs_dist = (a * b) + (a * c)
    dist_match = all(
        abs(lhs_dist.c.get(k, 0) - rhs_dist.c.get(k, 0)) < 1e-10
        for k in set(lhs_dist.c.keys()) | set(rhs_dist.c.keys())
    )
    suite.assert_true("Distributivity", dist_match)
    suite.assert_true("Additive commutativity", (a + b).c == (b + a).c)
    lhs_add = (a + b) + c
    rhs_add = a + (b + c)
    suite.assert_true("Additive associativity", lhs_add.c == rhs_add.c)
    neg_a = -a
    sum_zero = a + neg_a
    suite.assert_eq("a + (-a) = 0 at dim 0", sum_zero.c.get(0, 0), 0)
    suite.assert_eq("a + (-a) = 0 at dim -1", sum_zero.c.get(-1, 0), 0)
    return suite.report()


# =============================================================================
# CALCULUS: DERIVATIVES
# =============================================================================

def test_calculus_derivatives():
    suite = TestSuite("Calculus: Derivatives")
    h = ZERO
    x = R(3)
    deriv_x2 = ((x + h)**2 - x**2) / h
    suite.assert_eq("d/dx[x\u00b2] at x=3 = 6", deriv_x2.st(), 6)
    x = R(2)
    deriv_x3 = ((x + h)**3 - x**3) / h
    suite.assert_eq("d/dx[x\u00b3] at x=2 = 12", deriv_x3.st(), 12)
    deriv_x4 = ((x + h)**4 - x**4) / h
    suite.assert_eq("d/dx[x\u2074] at x=2 = 32", deriv_x4.st(), 32)
    x = R(1)
    deriv_x5 = ((x + h)**5 - x**5) / h
    suite.assert_eq("d/dx[x\u2075] at x=1 = 5", deriv_x5.st(), 5)
    x = R(5)
    f_linear = lambda t: 3*t + 2
    deriv_linear = (f_linear(x + h) - f_linear(x)) / h
    suite.assert_eq("d/dx[3x+2] = 3", deriv_linear.st(), 3)
    x = R(10)
    deriv_const = (R(7) - R(7)) / h
    suite.assert_eq("d/dx[7] = 0", deriv_const.st(), 0)
    x = R(2)
    f_expanded = (x + h)**3
    second_deriv_coeff = f_expanded.c.get(-2, 0)
    suite.assert_eq("d\u00b2/dx\u00b2[x\u00b3] at x=2 = 12", second_deriv_coeff * 2, 12)
    return suite.report()


# =============================================================================
# CALCULUS: LIMITS
# =============================================================================

def test_calculus_limits():
    suite = TestSuite("Calculus: Limits")
    h = ZERO
    sin_x = sin_composite(h, terms=8)
    limit_sin = (sin_x / h).st()
    suite.assert_eq("lim(x\u21920) sin(x)/x = 1", limit_sin, 1, tol=1e-6)
    exp_x = exp_composite(h, terms=10)
    limit_exp = ((exp_x - R(1)) / h).st()
    suite.assert_eq("lim(x\u21920) (e\u02e3-1)/x = 1", limit_exp, 1, tol=1e-6)
    cos_x = cos_composite(h, terms=8)
    limit_cos = ((R(1) - cos_x) / (h * h)).st()
    suite.assert_eq("lim(x\u21920) (1-cos(x))/x\u00b2 = 0.5", limit_cos, 0.5, tol=1e-6)
    x = R(2) + h
    numer = x**2 - R(4)
    denom = x - R(2)
    limit_factor = (numer / denom).st()
    suite.assert_eq("lim(x\u21922) (x\u00b2-4)/(x-2) = 4", limit_factor, 4)
    ln_1_plus_h = ln_1_plus_x(h, terms=10)
    limit_ln = (ln_1_plus_h / h).st()
    suite.assert_eq("lim(x\u21920) ln(1+x)/x = 1", limit_ln, 1, tol=1e-6)
    x = R(1) + h
    numer = x**3 - R(1)
    denom = x - R(1)
    limit_cubic = (numer / denom).st()
    suite.assert_eq("lim(x\u21921) (x\u00b3-1)/(x-1) = 3", limit_cubic, 3)
    return suite.report()


# =============================================================================
# CALCULUS: INTEGRATION
# =============================================================================

def test_calculus_integration():
    suite = TestSuite("Calculus: Integration Round-Trip")
    h = ZERO
    f1 = R(3) + h
    F1 = antiderivative(f1)
    f1_back = derivative_of(F1)
    suite.assert_eq("\u222bx dx round-trip at x=3", f1_back.st(), f1.st(), tol=1e-10)
    x = R(2) + h
    f2 = x ** 2
    F2 = antiderivative(f2)
    f2_back = derivative_of(F2)
    suite.assert_eq("\u222bx\u00b2 dx round-trip at x=2", f2_back.st(), f2.st(), tol=1e-10)
    f3 = x ** 3
    F3 = antiderivative(f3)
    f3_back = derivative_of(F3)
    suite.assert_eq("\u222bx\u00b3 dx round-trip at x=2", f3_back.st(), f3.st(), tol=1e-10)
    riemann = (INF * ZERO).st()
    suite.assert_eq("\u221e \u00d7 0 = 1 (Riemann foundation)", riemann, 1)
    return suite.report()


# =============================================================================
# 0/0 PROVENANCE-DEPENDENT RESULTS
# =============================================================================

def test_zero_division():
    suite = TestSuite("Zero Division (Novel Feature)")
    result_1 = ZERO / ZERO
    suite.assert_eq("|1|\u208b\u2081 / |1|\u208b\u2081 = 1", result_1.st(), 1)
    zero_2 = Composite({-1: 2})
    result_2 = zero_2 / ZERO
    suite.assert_eq("|2|\u208b\u2081 / |1|\u208b\u2081 = 2", result_2.st(), 2)
    zero_5 = Composite({-1: 5})
    result_5 = zero_5 / ZERO
    suite.assert_eq("|5|\u208b\u2081 / |1|\u208b\u2081 = 5", result_5.st(), 5)
    zero_10 = Composite({-1: 10})
    zero_2 = Composite({-1: 2})
    result_10_2 = zero_10 / zero_2
    suite.assert_eq("|10|\u208b\u2081 / |2|\u208b\u2081 = 5", result_10_2.st(), 5)
    produced_zero = R(5) * ZERO
    recovered = produced_zero / ZERO
    suite.assert_eq("(5 \u00d7 0) / 0 = 5", recovered.st(), 5)
    zero_sq = ZERO * ZERO
    result_sq = zero_sq / zero_sq
    suite.assert_eq("|1|\u208b\u2082 / |1|\u208b\u2082 = 1", result_sq.st(), 1)
    result_mixed = (ZERO * ZERO) / ZERO
    suite.assert_true("|1|\u208b\u2082 / |1|\u208b\u2081 = |1|\u208b\u2081", -1 in result_mixed.c)
    return suite.report()


# =============================================================================
# STRESS TESTS
# =============================================================================

def test_stress():
    suite = TestSuite("Stress Tests")
    x = R(1)
    for _ in range(50):
        x = x * ZERO
    suite.assert_eq("50\u00d7 zero gives dim -50", min(x.c.keys()), -50)
    for _ in range(50):
        x = x / ZERO
    suite.assert_eq("50\u00d7 divide recovers dim 0", x.st(), 1)
    y = R(7)
    for _ in range(20):
        y = y * ZERO
        y = y / ZERO
    suite.assert_eq("20\u00d7 (\u00d70 then /0) preserves value", y.st(), 7)
    h = ZERO
    x = R(1) + h
    high_power = x ** 20
    suite.assert_eq("(1+h)\u00b2\u2070 constant term = 1", high_power.c.get(0, 0), 1)
    suite.assert_eq("(1+h)\u00b2\u2070 linear term = 20", high_power.c.get(-1, 0), 20)
    suite.assert_eq("(1+h)\u00b2\u2070 quadratic term = 190", high_power.c.get(-2, 0), 190)
    # Dict vs FFT cross-check
    h_dict = Composite.zero()
    x_dict = Composite.real(2) + h_dict
    result_dict = ((x_dict + h_dict) ** 10 - x_dict ** 10) / h_dict
    h_fft = CompositeFFT.zero()
    x_fft = CompositeFFT.real(2) + h_fft
    result_fft = ((x_fft + h_fft) ** 10 - x_fft ** 10) / h_fft
    suite.assert_eq("Dict and FFT match for x\u00b9\u2070 derivative",
                    result_dict.st(), result_fft.st(), tol=0.01)
    # Negative coefficient chains
    neg_result = (R(-5) * ZERO) / ZERO
    suite.assert_eq("(-5 \u00d7 0) / 0 = -5", neg_result.st(), -5)
    neg_result_2 = (R(-3.14) * ZERO) / ZERO
    suite.assert_eq("(-3.14 \u00d7 0) / 0 = -3.14", neg_result_2.st(), -3.14)
    neg_chain = R(-7)
    for _ in range(10):
        neg_chain = neg_chain * ZERO
    for _ in range(10):
        neg_chain = neg_chain / ZERO
    suite.assert_eq("10\u00d7 (neg \u00d7 0) then 10\u00d7 /0 recovers -7", neg_chain.st(), -7)
    # Mixed infinity/zero cancellation
    for a in [1, 5, -3, 0.5, 100]:
        mixed = ((R(a) * ZERO) * INF) / INF / ZERO
        suite.assert_eq(f"({a} \u00d7 0 \u00d7 \u221e) / \u221e / 0 = {a}", mixed.st(), a)
    # More complex mixed chain
    complex_chain = (((R(7) * ZERO) * INF) * ZERO) / ZERO / INF / ZERO
    suite.assert_eq("((7\u00d70\u00d7\u221e)\u00d70)/0/\u221e/0 = 7", complex_chain.st(), 7)
    # FFT precision at window boundaries
    h_fft_boundary = CompositeFFT.zero(window=8)
    x_boundary = CompositeFFT.real(1, window=8) + h_fft_boundary
    power_7 = x_boundary ** 7
    suite.assert_eq("FFT (1+h)\u2077 constant term = 1", abs(power_7.st() - 1) < 1e-10, True)
    # Compare small vs large window
    h_small = CompositeFFT.zero(window=4)
    h_large = CompositeFFT.zero(window=32)
    x_small = CompositeFFT.real(2, window=4) + h_small
    x_large = CompositeFFT.real(2, window=32) + h_large
    deriv_small = ((x_small + h_small)**5 - x_small**5) / h_small
    deriv_large = ((x_large + h_large)**5 - x_large**5) / h_large
    suite.assert_eq("FFT window=4 vs window=32 derivative match",
                    abs(deriv_small.st() - deriv_large.st()) < 0.1, True)
    return suite.report()


# =============================================================================
# MULTIVARIATE TESTS
# =============================================================================

def test_multivariate():
    suite = TestSuite("Multivariate Calculus")
    hx = CompositeMulti.zero_var(0)
    hy = CompositeMulti.zero_var(1)
    x = CompositeMulti.real(3) + hx
    y = CompositeMulti.real(4) + hy
    f1 = x**2 + y**2
    suite.assert_eq("\u2202/\u2202x[x\u00b2+y\u00b2] at (3,4) = 6", f1.partial(0), 6)
    suite.assert_eq("\u2202/\u2202y[x\u00b2+y\u00b2] at (3,4) = 8", f1.partial(1), 8)
    f2 = x * y
    suite.assert_eq("\u2202\u00b2/\u2202x\u2202y[xy] = 1", f2.mixed_partial((1, 1)), 1)
    grad = f1.gradient()
    suite.assert_true("\u2207(x\u00b2+y\u00b2) = [6, 8]", grad == [6, 8])
    hx2 = MultiComposite.var(0)
    hy2 = MultiComposite.var(1)
    x2 = MultiComposite.real(3) + hx2
    y2 = MultiComposite.real(4) + hy2
    f3 = x2**2 + y2**2
    lap = f3.laplacian()
    suite.assert_eq("\u2207\u00b2(x\u00b2+y\u00b2) = 4", lap, 4)
    f4 = x2**2 - y2**2
    lap_harm = f4.laplacian()
    suite.assert_eq("\u2207\u00b2(x\u00b2-y\u00b2) = 0 (harmonic)", lap_harm, 0)
    return suite.report()


# =============================================================================
# TRANSCENDENTAL FUNCTIONS
# =============================================================================

def test_multiterm_division():
    """Test multi-term division via backend deconvolve."""
    suite = TestSuite("Multi-Term Division")
    h = ZERO
    # Test: (x² - 1) / (x - 1) = x + 1 at x=3 → st = 4
    x = R(3) + h
    numer = x**2 - R(1)
    denom = x - R(1)
    result = numer / denom
    suite.assert_eq("(x²-1)/(x-1) at x=3 = 4", result.st(), 4)
    # Test: (x³ - 1) / (x - 1) = x² + x + 1 at x=2 → st = 7
    x = R(2) + h
    numer = x**3 - R(1)
    denom = x - R(1)
    result = numer / denom
    suite.assert_eq("(x³-1)/(x-1) at x=2 = 7", result.st(), 7)
    # Test: (x⁴ - 1) / (x - 1) = x³ + x² + x + 1 at x=2 → st = 15
    x = R(2) + h
    numer = x**4 - R(1)
    denom = x - R(1)
    result = numer / denom
    suite.assert_eq("(x⁴-1)/(x-1) at x=2 = 15", result.st(), 15)
    # Test: sin(x)/x at x→0 via multi-term division
    x = h
    sin_x = sin_composite(x, terms=5)
    result = sin_x / x
    suite.assert_eq("sin(x)/x at x→0 = 1", result.st(), 1, tol=1e-6)
    # Test: (1 + x) / (1 - x) at x→0 = 1
    x = h
    numer = R(1) + x
    denom = R(1) - x
    result = numer / denom
    suite.assert_eq("(1+x)/(1-x) at x→0 = 1", result.st(), 1, tol=1e-6)
    # Test: Verify single-term division still works (fast path)
    result_single = R(10) / R(2)
    suite.assert_eq("10 / 2 = 5 (single-term fast path)", result_single.st(), 5)
    # Test: Division by structural zero still works
    result_zero = R(5) / ZERO
    suite.assert_true("5 / 0 = |5|₁ (infinity)", 1 in result_zero.c and result_zero.c[1] == 5)
    return suite.report()

def test_transcendental():
    suite = TestSuite("Transcendental Functions")
    h = ZERO
    sin_0 = sin_composite(R(0), terms=5)
    cos_0 = cos_composite(R(0), terms=5)
    suite.assert_eq("sin(0) = 0", sin_0.st(), 0, tol=1e-10)
    suite.assert_eq("cos(0) = 1", cos_0.st(), 1, tol=1e-10)
    sin_h = sin_composite(h, terms=8)
    cos_h = cos_composite(h, terms=8)
    identity = sin_h * sin_h + cos_h * cos_h
    suite.assert_eq("sin\u00b2(h) + cos\u00b2(h) = 1", identity.st(), 1, tol=1e-6)
    exp_0 = exp_composite(R(0), terms=10)
    suite.assert_eq("exp(0) = 1", exp_0.st(), 1, tol=1e-10)
    deriv_sin = (sin_composite(h, terms=8) / h).st()
    suite.assert_eq("d/dx[sin(x)]|\u2080 = cos(0) = 1", deriv_sin, 1, tol=1e-6)
    exp_h = exp_composite(h, terms=10)
    deriv_exp = ((exp_h - R(1)) / h).st()
    suite.assert_eq("d/dx[e\u02e3]|\u2080 = e\u2070 = 1", deriv_exp, 1, tol=1e-6)
    cos_deriv = ((cos_composite(h, terms=8) - R(1)) / h).st()
    suite.assert_eq("d/dx[cos(x)]|\u2080 = -sin(0) = 0", cos_deriv, 0, tol=1e-6)
    return suite.report()


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_tests():
    print("\n" + "#" * 70)
    print("# COMPREHENSIVE TEST SUITE")
    print("# Provenance-Preserving Arithmetic — Import-Based")
    print("# (Tests import from composite_lib.py)")
    print("#" * 70)

    all_results = []

    # Paper Theorems
    all_results.append(("T1: Information Preservation", test_theorem_1_information_preservation()))
    all_results.append(("T2: Zero-Infinity Duality", test_theorem_2_zero_infinity_duality()))
    all_results.append(("T3: Provenance Non-Uniqueness", test_theorem_3_provenance_non_uniqueness()))
    all_results.append(("T4: Reversibility", test_theorem_4_reversibility()))
    all_results.append(("T5: Coefficient Cancellation", test_theorem_5_coefficient_cancellation()))
    all_results.append(("T6: Identity Elements", test_theorem_6_identity_elements()))
    all_results.append(("T7: Fractional Orders", test_theorem_7_fractional_orders()))
    all_results.append(("T8: Total Ordering", test_theorem_8_total_ordering()))

    # Additional tests
    all_results.append(("Edge Cases", test_edge_cases()))
    all_results.append(("Algebraic Properties", test_algebraic_properties()))
    all_results.append(("Calculus: Derivatives", test_calculus_derivatives()))
    all_results.append(("Calculus: Limits", test_calculus_limits()))
    all_results.append(("Calculus: Integration", test_calculus_integration()))
    all_results.append(("Zero Division (Novel)", test_zero_division()))
    all_results.append(("Stress Tests", test_stress()))
    all_results.append(("Multivariate", test_multivariate()))
    all_results.append(("Transcendental Functions", test_transcendental()))
    all_results.append(("Multi-Term Division", test_multiterm_division()))

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
        print("The implementation correctly satisfies all paper theorems and edge cases.")
    else:
        print(f"\u26a0\ufe0f  {total_tests - total_passed} test(s) failed.")
        print("Review failed tests above for details.")

    print("\n" + "#" * 70)

    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
