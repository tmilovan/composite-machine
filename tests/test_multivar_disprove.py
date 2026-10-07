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
test_multivar_disprove.py -- Adversarial Tests for Multivariate Composite
=========================================================================

Purpose: find where multivariate composite gives WRONG results.  Each test
compares a multivariate result against:
  - analytic derivatives (ground truth)
  - mpmath.diff at 50 digits (numerical ground truth)
  - single-variable composite (the gold standard)

Since 2026-10-06 the multivariate results come from directional composites
(composite_multivar): every partial is read off ordinary
one-variable composites evaluated along several directions, and integrands use
the library functions (sin, exp), not mc_ ones.  Before that the file tested the
MC class (tuple dimensions, now parked in composite_multivar_mc.py); its division, term-explosion and zero-handling
failures are what the categories are named after.  See
docs/MC Replacement - Directional Composites (DRAFT).md.

Every check prints got against known, pass or fail.

Run:
  pytest tests/test_multivar_disprove.py -v -s
"""

import math
import time
import sys
import pytest

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 50

from composite.composite_multivar import (
    partial_derivative, gradient_at, hessian_at,
    laplacian_at, taylor_jets,
)
from composite.composite_lib import (
    R, Composite, ResidueError,
    sin, cos, exp, ln, sqrt,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def mv_partial(f, at, wrt):
    """Partial derivative of f at `at` from directional composites."""
    return partial_derivative(f, at, list(wrt))


def mv_value(f, at):
    """Value of f at `at`: the order-0 coefficient, one composite."""
    T, _ = taylor_jets(f, at, 0)
    return T[(0,) * len(at)]


def mv_cost(f, at, order):
    """(Taylor coefficients to `order`, composites evaluated, seconds)."""
    t0 = time.perf_counter()
    T, n = taylor_jets(f, at, order)
    return T, n, time.perf_counter() - t0


def sv_deriv(f_single, x0, order=1):
    """Single-variable composite derivative (the gold standard)."""
    x = Composite({0: x0, -1: 1.0})
    r = f_single(x)
    raw = r.c.get(-order, 0.0)
    return raw * math.factorial(order)


def ref(fm, at, orders):
    """mpmath.diff at 50 digits."""
    return float(mpmath.diff(fm, [mpmath.mpf(a) for a in at], tuple(orders)))


def check(label, got, want, tol):
    print(f"\n  {label:52s} got {got!r:24} known {want!r:24} err {abs(got - want):.1e}")
    assert got == pytest.approx(want, abs=tol), f"{label}: got {got}, expected {want}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 1: DIVISION BY MULTIVARIATE EXPRESSIONS
# ═══════════════════════════════════════════════════════════════
# MC lost every derivative through division: _mc_poly_divide stopped when the
# remainder's leading total dimension dropped below the divisor's.  Each
# directional composite divides as an ordinary composite.

class TestDivisionFailures:
    """Division by an expression in the variables keeps its derivatives."""

    def test_D01_x_over_y_df_dx(self):
        """x/y at (2,3): df/dx = 1/y = 1/3."""
        check("D01 d/dx x/y at (2,3)", mv_partial(lambda x, y: x / y, [2, 3], (1, 0)), 1 / 3, 1e-6)

    def test_D02_x_over_y_df_dy(self):
        """x/y at (2,3): df/dy = -x/y^2 = -2/9."""
        check("D02 d/dy x/y at (2,3)", mv_partial(lambda x, y: x / y, [2, 3], (0, 1)), -2 / 9, 1e-6)

    def test_D03_one_over_y_df_dy(self):
        """1/y at (2,3): df/dy = -1/y^2 = -1/9."""
        check("D03 d/dy 1/y at (2,3)", mv_partial(lambda x, y: R(1) / y, [2, 3], (0, 1)), -1 / 9, 1e-6)

    def test_D04_xy_over_sum_df_dx(self):
        """xy/(x+y) at (2,3): df/dx = y^2/(x+y)^2 = 9/25."""
        check("D04 d/dx xy/(x+y) at (2,3)", mv_partial(lambda x, y: x * y / (x + y), [2, 3], (1, 0)), 9 / 25, 1e-6)

    def test_D05_difference_over_sum(self):
        """(x-y)/(x+y) at (3,1): df/dx = 2y/(x+y)^2 = 1/8."""
        check("D05 d/dx (x-y)/(x+y) at (3,1)", mv_partial(lambda x, y: (x - y) / (x + y), [3, 1], (1, 0)), 1 / 8, 1e-6)

    def test_D06_sin_x_over_y(self):
        """sin(x)/y at (1,2): df/dx = cos(x)/y = cos(1)/2."""
        check("D06 d/dx sin(x)/y at (1,2)", mv_partial(lambda x, y: sin(x) / y, [1, 2], (1, 0)), math.cos(1) / 2, 1e-6)

    def test_D07_exp_xy_over_sum(self):
        """exp(xy)/(x+y) at (1,2): df/dx = [y(x+y)-1]*exp(xy)/(x+y)^2."""
        check("D07 d/dx exp(xy)/(x+y) at (1,2)", mv_partial(lambda x, y: exp(x * y) / (x + y), [1, 2], (1, 0)),
              5 * math.exp(2) / 9, 1e-4)

    def test_D08_x_squared_over_y(self):
        """x^2/y at (2,3): df/dx = 2x/y = 4/3."""
        check("D08 d/dx x^2/y at (2,3)", mv_partial(lambda x, y: x ** 2 / y, [2, 3], (1, 0)), 4 / 3, 1e-6)

    def test_D09_exp_x_over_y(self):
        """exp(x)/y at (1,2): df/dx = exp(x)/y = e/2."""
        check("D09 d/dx exp(x)/y at (1,2)", mv_partial(lambda x, y: exp(x) / y, [1, 2], (1, 0)), math.e / 2, 1e-6)

    def test_D10_division_value_and_derivatives(self):
        """x/y at (2,3): the value is 2/3 and the derivatives are 1/3 and -2/9."""
        f = lambda x, y: x / y
        check("D10 x/y at (2,3)", mv_value(f, [2, 3]), 2 / 3, 1e-10)
        g = gradient_at(f, [2, 3])
        check("D10 d/dx x/y", g[0], 1 / 3, 1e-10)
        check("D10 d/dy x/y", g[1], -2 / 9, 1e-10)

    def test_D11_three_var_division(self):
        """xy/z at (2,3,4): df/dx = y/z = 3/4."""
        check("D11 d/dx xy/z at (2,3,4)", mv_partial(lambda x, y, z: x * y / z, [2, 3, 4], (1, 0, 0)), 3 / 4, 1e-6)

    def test_D12_reciprocal_of_variable(self):
        """1/x at (3, unused): df/dx = -1/x^2 = -1/9."""
        check("D12 d/dx 1/x at (3,5)", mv_partial(lambda x, y: R(1) / x, [3, 5], (1, 0)), -1 / 9, 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 2: SINGLE-VAR vs MULTIVAR COMPARISON
# ═══════════════════════════════════════════════════════════════
# Single-variable composite gives correct derivatives.
# If multivar disagrees, multivar is wrong.

class TestSingleVarComparison:
    """Compare multivariate against single-variable composite (gold standard)."""

    def test_SV01_sin_2x_derivative(self):
        """sin(2x) at x=1: single-var and multivar must agree on df/dx."""
        check("SV01 d/dx sin(2x)", mv_partial(lambda x, y: sin(2 * x), [1, 2], (1, 0)),
              sv_deriv(lambda x: sin(2 * x), 1.0), 1e-8)

    def test_SV02_exp_x_sin_y_df_dx(self):
        """exp(x)*sin(y) at (1,2): df/dx = exp(x)*sin(y)."""
        check("SV02 d/dx exp(x) sin(y)", mv_partial(lambda x, y: exp(x) * sin(y), [1, 2], (1, 0)),
              sv_deriv(lambda x: exp(x) * Composite({0: math.sin(2)}), 1.0), 1e-8)

    def test_SV03_sin_xy_df_dx(self):
        """sin(xy) at (1,2): df/dx = y*cos(xy) = 2*cos(2)."""
        check("SV03 d/dx sin(xy)", mv_partial(lambda x, y: sin(x * y), [1, 2], (1, 0)),
              sv_deriv(lambda x: sin(x * Composite({0: 2.0})), 1.0), 1e-6)

    def test_SV04_exp_sin_x_df_dx(self):
        """exp(sin(x)) at x=1: deep composition."""
        check("SV04 d/dx exp(sin(x))", mv_partial(lambda x, y: exp(sin(x)), [1, 2], (1, 0)),
              sv_deriv(lambda x: exp(sin(x)), 1.0), 1e-6)

    def test_SV05_ln_x_plus_const_df_dx(self):
        """ln(x+3) at x=2: df/dx = 1/5."""
        check("SV05 d/dx ln(x+3)", mv_partial(lambda x, y: ln(x + 3), [2, 1], (1, 0)),
              sv_deriv(lambda x: ln(x + Composite({0: 3.0})), 2.0), 1e-8)

    def test_SV06_sqrt_x_df_dx(self):
        """sqrt(x) at x=4: df/dx = 1/(2*sqrt(x)) = 1/4."""
        check("SV06 d/dx sqrt(x)", mv_partial(lambda x, y: sqrt(x), [4, 1], (1, 0)),
              sv_deriv(lambda x: sqrt(x), 4.0), 1e-8)

    def test_SV07_x_over_const_division_ok(self):
        """x/3 at x=2: df/dx = 1/3."""
        check("SV07 d/dx x/3", mv_partial(lambda x, y: x / 3, [2, 1], (1, 0)),
              sv_deriv(lambda x: x / Composite({0: 3.0}), 2.0), 1e-8)

    def test_SV08_x_over_y_divergence(self):
        """x/y at (2,3): multivar vs single-var df/dx = 1/3 (fix y=3)."""
        check("SV08 d/dx x/y", mv_partial(lambda x, y: x / y, [2, 3], (1, 0)),
              sv_deriv(lambda x: x / Composite({0: 3.0}), 2.0), 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 3: TERM EXPLOSION AND PERFORMANCE
# ═══════════════════════════════════════════════════════════════
# MC created combinatorially many cross terms (exp(sin(xy)) ~24k terms, the
# triple composition ~2 minutes).  A directional composite holds one
# infinitesimal, so the cost is the number of composites, C(K+n-1, n-1) for
# order K plus one smoothness check, and does not depend on how deep the
# expression is.

class TestTermExplosion:
    """Cost of deep and many-variable expressions: composites and time."""

    def _hessian_case(self, label, f, fm, at, budget):
        T, n, secs = mv_cost(f, at, 2)
        nv = len(at)
        print(f"\n  {label}: {n} composites, {secs * 1e3:.1f} ms")
        assert n == math.comb(2 + nv - 1, nv - 1) + 1      # + 1 smoothness check direction
        assert secs < budget, f"{label} took {secs:.1f}s"
        for a, t in T.items():
            want = ref(fm, at, a) / math.prod(math.factorial(e) for e in a)
            check(f"{label} T{a}", t, want, 1e-10)

    def test_TE01_exp_xy(self):
        self._hessian_case("TE01 exp(xy)", lambda x, y: exp(x * y),
                           lambda x, y: mpmath.exp(x * y), [1, 1], 30)

    def test_TE02_sin_xy(self):
        self._hessian_case("TE02 sin(xy)", lambda x, y: sin(x * y),
                           lambda x, y: mpmath.sin(x * y), [1, 1], 30)

    def test_TE03_double_composition(self):
        """exp(sin(xy)): MC produced ~24k terms."""
        self._hessian_case("TE03 exp(sin(xy))", lambda x, y: exp(sin(x * y)),
                           lambda x, y: mpmath.exp(mpmath.sin(x * y)), [0.5, 0.5], 30)

    def test_TE04_three_var_exp(self):
        self._hessian_case("TE04 exp(xyz)", lambda x, y, z: exp(x * y * z),
                           lambda x, y, z: mpmath.exp(x * y * z), [0.5, 0.5, 0.5], 30)

    def test_TE05_four_var_feasibility(self):
        """exp(x1 x2 x3 x4): MC questioned whether 4 variables were feasible."""
        self._hessian_case("TE05 exp(x1x2x3x4)", lambda a, b, c, d: exp(a * b * c * d),
                           lambda a, b, c, d: mpmath.exp(a * b * c * d), [0.5] * 4, 60)

    def test_TE06_triple_composition(self):
        """exp(sin(cos(xy))): ~2 minutes on MC, skipped there."""
        self._hessian_case("TE06 exp(sin(cos(xy)))", lambda x, y: exp(sin(cos(x * y))),
                           lambda x, y: mpmath.exp(mpmath.sin(mpmath.cos(x * y))), [0.5, 0.5], 30)

    def test_TE07_cost_does_not_grow_with_depth(self):
        """The composite count for order K is fixed by K and n, whatever the
        expression; MC's term count grew with every composition layer."""
        counts = {}
        for label, f in (("xy", lambda x, y: x * y),
                         ("exp(sin(cos(xy)))", lambda x, y: exp(sin(cos(x * y))))):
            _, n, secs = mv_cost(f, [0.5, 0.5], 4)
            counts[label] = n
            print(f"\n  TE07 order 4, {label}: {n} composites, {secs * 1e3:.1f} ms")
        assert counts["xy"] == counts["exp(sin(cos(xy)))"] == 5 + 1      # + 1 check direction


# ═══════════════════════════════════════════════════════════════
# CATEGORY 4: BLACK-SCHOLES AND FINANCIAL APPLICATIONS
# ═══════════════════════════════════════════════════════════════
# Greeks with several risk factors.  The d1 formula divides by sigma.

class TestBlackScholes:
    """Black-Scholes d1 and its sensitivities in S and sigma."""

    S, K, r_rate, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    def _d1(self, S, sig):
        return (ln(S / self.K) + (self.r_rate + sig ** 2 / 2) * self.T) / (sig * math.sqrt(self.T))

    def _d1_analytic(self):
        S, K, r, s, T = self.S, self.K, self.r_rate, self.sigma, self.T
        return (math.log(S / K) + (r + s ** 2 / 2) * T) / (s * math.sqrt(T))

    def test_BS01_d1_value(self):
        check("BS01 d1", mv_value(self._d1, [self.S, self.sigma]), self._d1_analytic(), 1e-6)

    def test_BS02_dd1_dS(self):
        """dd1/dS = 1/(S*sigma*sqrt(T))."""
        check("BS02 dd1/dS", mv_partial(self._d1, [self.S, self.sigma], (1, 0)),
              1 / (self.S * self.sigma * math.sqrt(self.T)), 1e-6)

    def test_BS03_dd1_dsigma(self):
        """dd1/dsigma = -(ln(S/K) + rT)/(sigma^2 sqrt(T)) + sqrt(T)/2."""
        S, K, r, s, T = self.S, self.K, self.r_rate, self.sigma, self.T
        want = -(math.log(S / K) + r * T) / (s ** 2 * math.sqrt(T)) + math.sqrt(T) / 2
        check("BS03 dd1/dsigma", mv_partial(self._d1, [S, s], (0, 1)), want, 1e-4)

    def test_BS04_d1_keeps_derivative_information(self):
        """Under MC, division left d1 a single term.  Both sensitivities are nonzero."""
        g = gradient_at(self._d1, [self.S, self.sigma])
        print(f"\n  BS04 grad d1 = {g}")
        assert abs(g[0]) > 1e-12 and abs(g[1]) > 1e-12

    def test_BS05_single_var_delta_works(self):
        """Single-var composite Delta = dd1/dS, the control for BS02."""
        S_sv = Composite({0: self.S, -1: 1.0})
        K_sv = Composite({0: self.K})
        r_sv = Composite({0: self.r_rate})
        sigma_sv = Composite({0: self.sigma})
        T_sv = Composite({0: self.T})

        d1_sv = (ln(S_sv / K_sv) + (r_sv + sigma_sv ** 2 / 2) * T_sv) / (
            sigma_sv * sqrt(T_sv)
        )
        check("BS05 single-var dd1/dS", d1_sv.c.get(-1, 0.0),
              1 / (self.S * self.sigma * math.sqrt(self.T)), 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 5: DEEP TRANSCENDENTAL CHAINS
# ═══════════════════════════════════════════════════════════════

class TestDeepTranscendentals:
    """Compositions of transcendentals with mixed variables."""

    def test_DT01_sin_cos_xy_df_dx(self):
        """sin(cos(xy)) at (1,1): df/dx = -y*sin(xy)*cos(cos(xy))."""
        check("DT01 d/dx sin(cos(xy))", mv_partial(lambda x, y: sin(cos(x * y)), [1, 1], (1, 0)),
              -math.sin(1) * math.cos(math.cos(1)), 1e-4)

    def test_DT02_sin_cos_xy_mixed_partial(self):
        """d2f/dxdy of sin(cos(xy)) at (1,1)."""
        check("DT02 d2/dxdy sin(cos(xy))", mv_partial(lambda x, y: sin(cos(x * y)), [1, 1], (1, 1)),
              ref(lambda x, y: mpmath.sin(mpmath.cos(x * y)), [1, 1], (1, 1)), 1e-3)

    def test_DT03_exp_sin_xy_df_dx(self):
        """exp(sin(xy)) at (0.5, 0.5): df/dx = y*cos(xy)*exp(sin(xy))."""
        v = 0.25
        check("DT03 d/dx exp(sin(xy))", mv_partial(lambda x, y: exp(sin(x * y)), [0.5, 0.5], (1, 0)),
              0.5 * math.cos(v) * math.exp(math.sin(v)), 1e-4)

    def test_DT04_ln_exp_x_plus_exp_y(self):
        """ln(exp(x)+exp(y)) at (1,1): df/dx = 1/2."""
        check("DT04 d/dx ln(e^x+e^y)", mv_partial(lambda x, y: ln(exp(x) + exp(y)), [1, 1], (1, 0)), 0.5, 1e-6)

    def test_DT05_sqrt_sin_x_sq_plus_cos_y_sq(self):
        """sqrt(sin(x)^2 + cos(y)^2) at (pi/4, pi/4):
        df/dx = sin(x)*cos(x) / sqrt(sin(x)^2 + cos(y)^2)."""
        pt = [math.pi / 4, math.pi / 4]
        s, c, c2 = math.sin(pt[0]), math.cos(pt[0]), math.cos(pt[1])
        check("DT05 d/dx sqrt(sin^2 x + cos^2 y)",
              mv_partial(lambda x, y: sqrt(sin(x) ** 2 + cos(y) ** 2), pt, (1, 0)),
              s * c / math.sqrt(s ** 2 + c2 ** 2), 1e-4)

    def test_DT06_exp_x_times_sin_y_second_order(self):
        """d2f/dx2 of exp(x)*sin(y) at (1,2) = exp(1)*sin(2)."""
        check("DT06 d2/dx2 exp(x) sin(y)", mv_partial(lambda x, y: exp(x) * sin(y), [1, 2], (2, 0)),
              math.e * math.sin(2), 1e-4)

    def test_DT07_separate_variables_no_interaction(self):
        """f(x,y) = sin(x) + cos(y): d2f/dxdy = 0."""
        check("DT07 d2/dxdy sin(x)+cos(y)", mv_partial(lambda x, y: sin(x) + cos(y), [1, 1], (1, 1)), 0.0, 1e-8)

    def test_DT08_composition_precision(self):
        """sin(exp(x)*cos(y)) at (0.5, 0.5): gradient against mpmath."""
        fm = lambda x, y: mpmath.sin(mpmath.exp(x) * mpmath.cos(y))
        g = gradient_at(lambda x, y: sin(exp(x) * cos(y)), [0.5, 0.5])
        check("DT08 d/dx sin(e^x cos y)", g[0], ref(fm, [0.5, 0.5], (1, 0)), 1e-4)
        check("DT08 d/dy sin(e^x cos y)", g[1], ref(fm, [0.5, 0.5], (0, 1)), 1e-4)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 6: HIGH-ORDER DERIVATIVES
# ═══════════════════════════════════════════════════════════════

class TestHighOrderDerivatives:
    """High-order and mixed partials."""

    def test_HO01_fourth_order_polynomial(self):
        """d4f/dx2dy2 of x^3*y^3 at (1,1) = 36."""
        check("HO01 d4/dx2dy2 x^3y^3", mv_partial(lambda x, y: x ** 3 * y ** 3, [1, 1], (2, 2)), 36.0, 1e-6)

    def test_HO02_fifth_order_single_dir(self):
        """d5f/dx5 of x^6*y at (1,2) = 720*y = 1440."""
        check("HO02 d5/dx5 x^6 y", mv_partial(lambda x, y: x ** 6 * y, [1, 2], (5, 0)), 1440.0, 1e-4)

    def test_HO03_third_order_mixed_poly(self):
        """d3f/dx2dy of x^3*y^2 = 12xy; at (2,3): 72."""
        check("HO03 d3/dx2dy x^3y^2", mv_partial(lambda x, y: x ** 3 * y ** 2, [2, 3], (2, 1)), 72.0, 1e-6)

    def test_HO04_third_order_three_vars(self):
        """d3f/dxdydz of xyz at (2,3,4) = 1."""
        check("HO04 d3/dxdydz xyz", mv_partial(lambda x, y, z: x * y * z, [2, 3, 4], (1, 1, 1)), 1.0, 1e-8)

    def test_HO05_high_order_transcendental(self):
        """d3f/dx3 of exp(x)*y at (1,2) = 2*e."""
        check("HO05 d3/dx3 exp(x) y", mv_partial(lambda x, y: exp(x) * y, [1, 2], (3, 0)), 2 * math.e, 1e-3)

    def test_HO06_second_order_sin_xy(self):
        """d2f/dx2 of sin(xy) = -y^2*sin(xy); at (1,2): -4*sin(2)."""
        check("HO06 d2/dx2 sin(xy)", mv_partial(lambda x, y: sin(x * y), [1, 2], (2, 0)), -4 * math.sin(2), 1e-3)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 7: HESSIAN AND GRADIENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════

class TestGradientHessian:
    """Gradient and Hessian extraction vs known values."""

    def test_GH01_gradient_components(self):
        """Gradient of x^2+y^2 at (3,4) is [6, 8]."""
        g = gradient_at(lambda x, y: x ** 2 + y ** 2, [3, 4])
        check("GH01 d/dx x^2+y^2", g[0], 6.0, 1e-8)
        check("GH01 d/dy x^2+y^2", g[1], 8.0, 1e-8)

    def test_GH02_hessian_symmetric(self):
        """Hessian of exp(xy) at (1,1) is symmetric."""
        H = hessian_at(lambda x, y: exp(x * y), [1, 1])
        check("GH02 H[0][1] vs H[1][0]", H[0][1], H[1][0], 1e-6)

    def test_GH03_hessian_exp_xy_values(self):
        """Hessian of exp(xy) at (1,1): e, e, 2e."""
        H = hessian_at(lambda x, y: exp(x * y), [1, 1])
        check("GH03 d2/dx2 exp(xy)", H[0][0], math.e, 1e-4)
        check("GH03 d2/dy2 exp(xy)", H[1][1], math.e, 1e-4)
        check("GH03 d2/dxdy exp(xy)", H[0][1], 2 * math.e, 1e-4)

    def test_GH04_laplacian_harmonic_function(self):
        """x^2-y^2 is harmonic: Laplacian = 0."""
        check("GH04 laplacian x^2-y^2", laplacian_at(lambda x, y: x ** 2 - y ** 2, [3, 4]), 0.0, 1e-10)

    def test_GH05_hessian_with_division(self):
        """Hessian of x/y at (2,3): 0, 4/27, -1/9.  MC lost all three."""
        H = hessian_at(lambda x, y: x / y, [2, 3])
        check("GH05 d2/dx2 x/y", H[0][0], 0.0, 1e-8)
        check("GH05 d2/dy2 x/y", H[1][1], 4 / 27, 1e-6)
        check("GH05 d2/dxdy x/y", H[0][1], -1 / 9, 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 8: EDGE CASES AND ZERO HANDLING
# ═══════════════════════════════════════════════════════════════
# A cancellation in composite arithmetic is a times zero: a - a = a*h, and a
# written 0 times f is h*f.  In one variable h denotes x - x0.  Along a
# direction (1, c2, ..., cn) the one h is the direction's own parameter, and
# which variable the residue belongs to is a convention of the direction set:
# with the first component fixed at 1 it read as f*(x - x0), measured exactly to
# order 2.  Since 2026-10-06 the directional functions refuse it (ResidueError,
# composite_lib._refusing_residue) rather than choose.  MC stripped zeros and
# these tests used to expect every derivative to be 0.

class TestEdgeCases:
    """Edge cases: zero handling, cancellation, identities."""

    def test_EC01_pythagorean_identity(self):
        """sin(x)^2 + cos(x)^2 = 1 with zero derivative."""
        f = lambda x, y: sin(x) ** 2 + cos(x) ** 2
        check("EC01 sin^2+cos^2", mv_value(f, [1, 1]), 1.0, 1e-10)
        check("EC01 d/dx sin^2+cos^2", mv_partial(f, [1, 1], (1, 0)), 0.0, 1e-8)

    def _refused(self, label, f):
        with pytest.raises(ResidueError) as e:
            gradient_at(f, [1, 1])
        print(f"\n  {label}: refused, {type(e.value).__name__}: {str(e.value)[:60]}")

    def test_EC02_subtraction_cancellation(self):
        """exp(xy) - exp(xy): a cancellation, refused."""
        self._refused("EC02 f - f", lambda x, y: exp(x * y) - exp(x * y))

    def test_EC03_multiply_by_zero(self):
        """0 * exp(xy): a written zero as an operand, refused."""
        self._refused("EC03 0*f", lambda x, y: 0 * exp(x * y))

    def test_EC04_division_by_constant_preserves_derivs(self):
        """x^2/5 at (3,1): df/dx = 2x/5 = 6/5."""
        check("EC04 d/dx x^2/5", mv_partial(lambda x, y: x ** 2 / 5, [3, 1], (1, 0)), 6 / 5, 1e-8)

    def test_EC05_cancellation_inside_an_expression(self):
        """sin(x) - sin(x) inside a larger expression is refused too."""
        self._refused("EC05 (sin(x) - sin(x)) + y", lambda x, y: (sin(x) - sin(x)) + y)

    def test_EC05b_written_zero_added_is_refused(self):
        """x*y + 0: a written 0 is an expressed zero and converts under R1, so it
        is refused like 0*f (in one variable it would denote x - x0)."""
        self._refused("EC05b x*y + 0", lambda x, y: x * y + 0)

    def test_EC05c_plain_zero_function_and_unequal_infinitesimals(self):
        """A function returning the plain number 0 never operates on a zero, and
        x - y at (1,1) is not a cancellation: its infinitesimal parts differ."""
        g0 = gradient_at(lambda x, y: 0, [1, 1])
        check("EC05c d/dx 0", g0[0], 0.0, 0.0)
        check("EC05c d/dy 0", g0[1], 0.0, 0.0)
        g = gradient_at(lambda x, y: x - y, [1, 1])
        check("EC05c d/dx x-y at (1,1)", g[0], 1.0, 1e-15)
        check("EC05c d/dy x-y at (1,1)", g[1], -1.0, 1e-15)

    def test_EC06_value_at_zero_point(self):
        """x + y at (0, 0): value 0, gradient [1, 1]."""
        f = lambda x, y: x + y
        check("EC06 x+y at origin", mv_value(f, [0, 0]), 0.0, 1e-10)
        g = gradient_at(f, [0, 0])
        check("EC06 d/dx x+y", g[0], 1.0, 1e-10)
        check("EC06 d/dy x+y", g[1], 1.0, 1e-10)

    def test_EC07_large_coefficients(self):
        """x^10*y^10 at (2, 2): value = 2^20 = 1048576."""
        check("EC07 x^10 y^10", mv_value(lambda x, y: x ** 10 * y ** 10, [2, 2]), 2.0 ** 20, 1e-6 * 2 ** 20)

    def test_EC08_negative_evaluation_point(self):
        """x^3*y at (-2, 3): value = -24, df/dx = 36."""
        f = lambda x, y: x ** 3 * y
        check("EC08 x^3 y at (-2,3)", mv_value(f, [-2, 3]), -24.0, 1e-8)
        check("EC08 d/dx x^3 y", mv_partial(f, [-2, 3], (1, 0)), 36.0, 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 9: FUNCTIONS THAT WORKED UNDER MC (BOUNDARY DOCUMENTATION)
# ═══════════════════════════════════════════════════════════════

class TestWorkingBoundary:
    """Functions where MC was already correct."""

    def test_WB01_polynomial_all_orders(self):
        """d5f/dx3dy2 of x^4*y^3 = 144xy; at (2,1): 288."""
        check("WB01 d5/dx3dy2 x^4y^3", mv_partial(lambda x, y: x ** 4 * y ** 3, [2, 1], (3, 2)), 288.0, 1e-4)

    def test_WB02_separable_transcendentals(self):
        """sin(x)*cos(y): df/dx = cos(x)*cos(y)."""
        check("WB02 d/dx sin(x)cos(y)", mv_partial(lambda x, y: sin(x) * cos(y), [1, 2], (1, 0)),
              math.cos(1) * math.cos(2), 1e-6)

    def test_WB03_additive_functions(self):
        """exp(x) + sin(y): df/dx = e."""
        check("WB03 d/dx exp(x)+sin(y)", mv_partial(lambda x, y: exp(x) + sin(y), [1, 2], (1, 0)), math.e, 1e-6)

    def test_WB04_composition_without_division(self):
        """sin(x+y): df/dx = cos(3)."""
        check("WB04 d/dx sin(x+y)", mv_partial(lambda x, y: sin(x + y), [1, 2], (1, 0)), math.cos(3), 1e-6)

    def test_WB05_exp_of_product(self):
        """exp(xy) at (1,2): df/dx = 2*exp(2)."""
        check("WB05 d/dx exp(xy)", mv_partial(lambda x, y: exp(x * y), [1, 2], (1, 0)), 2 * math.exp(2), 1e-4)

    def test_WB06_polynomial_with_constants(self):
        """(x+2)^3*(y-1)^2 at (1,3): df/dx = 3*9*4 = 108."""
        check("WB06 d/dx (x+2)^3 (y-1)^2",
              mv_partial(lambda x, y: (x + 2) ** 3 * (y - 1) ** 2, [1, 3], (1, 0)), 108.0, 1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 10: SINGLE-VAR CORRECTNESS (CONTROL GROUP)
# ═══════════════════════════════════════════════════════════════
# Prove that single-variable composite gives correct answers
# for the SAME functions where multivar fails.

class TestSingleVarControl:
    """Single-variable composite correctness for functions multivar fails on."""

    def test_SC01_x_over_y_fixed_y(self):
        """x/y with y=3 fixed: df/dx = 1/3."""
        sv = sv_deriv(lambda x: x / Composite({0: 3.0}), 2.0)
        assert sv == pytest.approx(1 / 3, abs=1e-8)

    def test_SC02_exp_xy_over_sum_fixed_y(self):
        """exp(2x)/(x+2) at x=1: using single-var composite."""
        sv = sv_deriv(
            lambda x: exp(2 * x) / (x + Composite({0: 2.0})), 1.0
        )
        expected = (2 * 3 - 1) * math.exp(2) / 9  # [2(x+2)-1]*exp(2x)/(x+2)^2
        assert sv == pytest.approx(expected, abs=1e-4)

    def test_SC03_sin_x_over_y_fixed_y(self):
        """sin(x)/2 at x=1: df/dx = cos(1)/2."""
        sv = sv_deriv(lambda x: sin(x) / Composite({0: 2.0}), 1.0)
        assert sv == pytest.approx(math.cos(1) / 2, abs=1e-8)

    def test_SC04_black_scholes_d1_delta(self):
        """BS Delta via single-var: seed S, fix sigma."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        S_sv = Composite({0: S, -1: 1.0})
        d1 = (ln(S_sv / Composite({0: K}))
              + Composite({0: (r + sigma ** 2 / 2) * T})) / Composite(
            {0: sigma * math.sqrt(T)}
        )
        dd1_dS = d1.c.get(-1, 0.0)
        expected = 1 / (S * sigma * math.sqrt(T))
        assert dd1_dS == pytest.approx(expected, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# SUMMARY RUNNER
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
