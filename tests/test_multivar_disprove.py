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
test_multivar_disprove.py — Adversarial Tests for Multivariate Composite
=========================================================================

Purpose: Find where multivariate composite (tuple dimensions) gives
WRONG results. Each test compares multivariate output against:
  - Analytic derivatives (ground truth)
  - Finite differences (numerical ground truth)
  - Single-variable composite (the gold standard that IS correct)

Tests are organized by FAILURE CATEGORY, not by what works.
A failing test here means multivariate composite is INCORRECT for that case.

Run:
  pytest tests/test_multivar_disprove.py -v
  python tests/test_multivar_disprove.py

Author: Toni Milovan
License: AGPL-3.0
"""

import math
import time
import sys
import pytest

from composite.composite_multivar import (
    MC, RR, RR_const,
    mc_sin, mc_cos, mc_exp, mc_ln, mc_sqrt, mc_tan, mc_power,
)
from composite.composite_lib import (
    R, ZERO, INF, Composite,
    sin, cos, exp, ln, sqrt,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def mv_partial(f, at, wrt):
    """Evaluate f on multivariate composite and extract partial derivative."""
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    r = f(*args)
    return r.partial(*wrt)


def mv_eval(f, at):
    """Evaluate f on multivariate composite and return the MC result."""
    nvars = len(at)
    args = [RR(at[i], var=i, nvars=nvars) for i in range(nvars)]
    return f(*args)


def mv_value(f, at):
    """Evaluate f on multivariate composite and return the standard part."""
    return mv_eval(f, at).st()


def sv_deriv(f_single, x0, order=1):
    """Single-variable composite derivative (the gold standard)."""
    x = Composite({0: x0, -1: 1.0})
    r = f_single(x)
    raw = r.c.get(-order, 0.0)
    return raw * math.factorial(order)


def fd_partial(f_scalar, at, var_idx, h=1e-7):
    """Finite-difference partial derivative (numerical ground truth)."""
    at_plus = list(at)
    at_minus = list(at)
    at_plus[var_idx] += h
    at_minus[var_idx] -= h
    return (f_scalar(*at_plus) - f_scalar(*at_minus)) / (2 * h)


def fd_mixed(f_scalar, at, i, j, h=1e-5):
    """Finite-difference mixed second partial d2f/dxi dxj."""
    at_pp = list(at); at_pp[i] += h; at_pp[j] += h
    at_pm = list(at); at_pm[i] += h; at_pm[j] -= h
    at_mp = list(at); at_mp[i] -= h; at_mp[j] += h
    at_mm = list(at); at_mm[i] -= h; at_mm[j] -= h
    return (f_scalar(*at_pp) - f_scalar(*at_pm)
            - f_scalar(*at_mp) + f_scalar(*at_mm)) / (4 * h * h)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 1: DIVISION BY MULTIVARIATE EXPRESSIONS
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: _mc_poly_divide stops when remainder's leading total
# dimension sum drops below divisor's leading total dimension sum.
# Infinitesimal terms at (-1,0), (0,-1) etc. have sum < 0, so
# the division NEVER produces quotient terms carrying derivatives.
#
# EVERY function involving division by a variable fails.

class TestDivisionFailures:
    """Division by any MC containing infinitesimals destroys derivatives."""

    def test_D01_x_over_y_df_dx(self):
        """x/y at (2,3): df/dx = 1/y = 1/3."""
        mv = mv_partial(lambda x, y: x / y, [2, 3], (1, 0))
        assert mv == pytest.approx(1 / 3, abs=1e-6), \
            f"multivar gives {mv}, expected {1/3}"

    def test_D02_x_over_y_df_dy(self):
        """x/y at (2,3): df/dy = -x/y^2 = -2/9."""
        mv = mv_partial(lambda x, y: x / y, [2, 3], (0, 1))
        assert mv == pytest.approx(-2 / 9, abs=1e-6), \
            f"multivar gives {mv}, expected {-2/9}"

    def test_D03_one_over_y_df_dy(self):
        """1/y at (2,3): df/dy = -1/y^2 = -1/9."""
        mv = mv_partial(lambda x, y: MC.real(1, 2) / y, [2, 3], (0, 1))
        assert mv == pytest.approx(-1 / 9, abs=1e-6), \
            f"multivar gives {mv}, expected {-1/9}"

    def test_D04_xy_over_sum_df_dx(self):
        """xy/(x+y) at (2,3): df/dx = y^2/(x+y)^2 = 9/25."""
        mv = mv_partial(lambda x, y: x * y / (x + y), [2, 3], (1, 0))
        assert mv == pytest.approx(9 / 25, abs=1e-6), \
            f"multivar gives {mv}, expected {9/25}"

    def test_D05_difference_over_sum(self):
        """(x-y)/(x+y) at (3,1): df/dx = 2y/(x+y)^2 = 1/8."""
        mv = mv_partial(lambda x, y: (x - y) / (x + y), [3, 1], (1, 0))
        assert mv == pytest.approx(1 / 8, abs=1e-6), \
            f"multivar gives {mv}, expected {1/8}"

    def test_D06_sin_x_over_y(self):
        """sin(x)/y at (1,2): df/dx = cos(x)/y = cos(1)/2."""
        mv = mv_partial(lambda x, y: mc_sin(x) / y, [1, 2], (1, 0))
        expected = math.cos(1) / 2
        assert mv == pytest.approx(expected, abs=1e-6), \
            f"multivar gives {mv}, expected {expected}"

    def test_D07_exp_xy_over_sum(self):
        """exp(xy)/(x+y) at (1,2): df/dx = [y(x+y)-1]*exp(xy)/(x+y)^2."""
        mv = mv_partial(lambda x, y: mc_exp(x * y) / (x + y), [1, 2], (1, 0))
        expected = 5 * math.exp(2) / 9
        assert mv == pytest.approx(expected, abs=1e-4), \
            f"multivar gives {mv}, expected {expected}"

    def test_D08_x_squared_over_y(self):
        """x^2/y at (2,3): df/dx = 2x/y = 4/3."""
        mv = mv_partial(lambda x, y: x ** 2 / y, [2, 3], (1, 0))
        assert mv == pytest.approx(4 / 3, abs=1e-6), \
            f"multivar gives {mv}, expected {4/3}"

    def test_D09_exp_x_over_y(self):
        """exp(x)/y at (1,2): df/dx = exp(x)/y = e/2."""
        mv = mv_partial(lambda x, y: mc_exp(x) / y, [1, 2], (1, 0))
        expected = math.e / 2
        assert mv == pytest.approx(expected, abs=1e-6), \
            f"multivar gives {mv}, expected {expected}"

    def test_D10_division_value_correct_but_derivatives_lost(self):
        """x/y at (2,3): VALUE is correct (2/3), but derivatives are 0."""
        r = mv_eval(lambda x, y: x / y, [2, 3])
        assert r.st() == pytest.approx(2 / 3, abs=1e-10), \
            "value should be correct"
        dx = r.partial(1, 0)
        dy = r.partial(0, 1)
        has_any_deriv = abs(dx) > 1e-12 or abs(dy) > 1e-12
        assert has_any_deriv, \
            f"all derivatives are zero: dx={dx}, dy={dy}"

    def test_D11_three_var_division(self):
        """xy/z at (2,3,4): df/dx = y/z = 3/4."""
        mv = mv_partial(lambda x, y, z: x * y / z, [2, 3, 4], (1, 0, 0))
        assert mv == pytest.approx(3 / 4, abs=1e-6), \
            f"multivar gives {mv}, expected {3/4}"

    def test_D12_reciprocal_of_variable(self):
        """1/x at (3, unused): df/dx = -1/x^2 = -1/9."""
        mv = mv_partial(lambda x, y: MC.real(1, 2) / x, [3, 5], (1, 0))
        assert mv == pytest.approx(-1 / 9, abs=1e-6), \
            f"multivar gives {mv}, expected {-1/9}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 2: SINGLE-VAR vs MULTIVAR COMPARISON
# ═══════════════════════════════════════════════════════════════
# Single-variable composite gives correct derivatives.
# If multivar disagrees, multivar is wrong.

class TestSingleVarComparison:
    """Compare multivariate against single-variable composite (gold standard)."""

    def test_SV01_sin_2x_derivative(self):
        """sin(2x) at x=1: single-var and multivar must agree on df/dx."""
        mv = mv_partial(lambda x, y: mc_sin(2 * x), [1, 2], (1, 0))
        sv = sv_deriv(lambda x: sin(2 * x), 1.0)
        assert mv == pytest.approx(sv, abs=1e-8), \
            f"mv={mv}, sv={sv}"

    def test_SV02_exp_x_sin_y_df_dx(self):
        """exp(x)*sin(y) at (1,2): df/dx = exp(x)*sin(y)."""
        mv = mv_partial(lambda x, y: mc_exp(x) * mc_sin(y), [1, 2], (1, 0))
        sv = sv_deriv(
            lambda x: exp(x) * Composite({0: math.sin(2)}), 1.0
        )
        assert mv == pytest.approx(sv, abs=1e-8), \
            f"mv={mv}, sv={sv}"

    def test_SV03_sin_xy_df_dx(self):
        """sin(xy) at (1,2): df/dx = y*cos(xy) = 2*cos(2)."""
        mv = mv_partial(lambda x, y: mc_sin(x * y), [1, 2], (1, 0))
        sv = sv_deriv(lambda x: sin(x * Composite({0: 2.0})), 1.0)
        assert mv == pytest.approx(sv, abs=1e-6), \
            f"mv={mv}, sv={sv}"

    def test_SV04_exp_sin_x_df_dx(self):
        """exp(sin(x)) at x=1: deep composition."""
        mv = mv_partial(lambda x, y: mc_exp(mc_sin(x)), [1, 2], (1, 0))
        sv = sv_deriv(lambda x: exp(sin(x)), 1.0)
        assert mv == pytest.approx(sv, abs=1e-6), \
            f"mv={mv}, sv={sv}"

    def test_SV05_ln_x_plus_const_df_dx(self):
        """ln(x+3) at x=2: df/dx = 1/5."""
        mv = mv_partial(lambda x, y: mc_ln(x + 3), [2, 1], (1, 0))
        sv = sv_deriv(lambda x: ln(x + Composite({0: 3.0})), 2.0)
        assert mv == pytest.approx(sv, abs=1e-8), \
            f"mv={mv}, sv={sv}"

    def test_SV06_sqrt_x_df_dx(self):
        """sqrt(x) at x=4: df/dx = 1/(2*sqrt(x)) = 1/4."""
        mv = mv_partial(lambda x, y: mc_sqrt(x), [4, 1], (1, 0))
        sv = sv_deriv(lambda x: sqrt(x), 4.0)
        assert mv == pytest.approx(sv, abs=1e-8), \
            f"mv={mv}, sv={sv}"

    def test_SV07_x_over_const_division_ok(self):
        """x/3 at x=2: df/dx = 1/3 (scalar divisor works in both)."""
        mv = mv_partial(lambda x, y: x / 3, [2, 1], (1, 0))
        sv = sv_deriv(lambda x: x / Composite({0: 3.0}), 2.0)
        assert mv == pytest.approx(sv, abs=1e-8), \
            f"mv={mv}, sv={sv}"

    def test_SV08_x_over_y_divergence(self):
        """x/y at (2,3): multivar vs single-var df/dx = 1/3.
        Single-var (fix y=3, differentiate x) gives 1/3.
        Multivar SHOULD give the same."""
        mv = mv_partial(lambda x, y: x / y, [2, 3], (1, 0))
        sv = sv_deriv(lambda x: x / Composite({0: 3.0}), 2.0)
        assert mv == pytest.approx(sv, abs=1e-6), \
            f"multivar={mv}, single-var={sv}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 3: TERM EXPLOSION AND PERFORMANCE
# ═══════════════════════════════════════════════════════════════
# Multivariate composite creates combinatorially many cross-terms.
# Deep compositions become infeasible.

class TestTermExplosion:
    """Document the combinatorial explosion of multivariate terms."""

    def test_TE01_exp_xy_term_count(self):
        """exp(xy) with 2 vars creates O(n^2) terms from n Taylor terms."""
        r = mv_eval(lambda x, y: mc_exp(x * y), [1, 1])
        assert len(r.c) <= 500, \
            f"exp(xy) produced {len(r.c)} terms (expected ~225)"

    def test_TE02_sin_xy_term_count(self):
        """sin(xy) with 2 vars — how many terms?"""
        r = mv_eval(lambda x, y: mc_sin(x * y), [1, 1])
        assert len(r.c) <= 500, \
            f"sin(xy) produced {len(r.c)} terms"

    def test_TE03_double_composition_explosion(self):
        """exp(sin(xy)) — ~24k terms in 2 vars. Still computable?"""
        t0 = time.time()
        r = mv_eval(lambda x, y: mc_exp(mc_sin(x * y)), [0.5, 0.5])
        elapsed = time.time() - t0
        term_count = len(r.c)
        assert term_count < 100000, \
            f"exp(sin(xy)) produced {term_count} terms"
        assert elapsed < 30, \
            f"exp(sin(xy)) took {elapsed:.1f}s"

    def test_TE04_three_var_exp_term_count(self):
        """exp(xyz) with 3 vars — cubic term growth."""
        r = mv_eval(lambda x, y, z: mc_exp(x * y * z), [0.5, 0.5, 0.5])
        assert len(r.c) <= 10000, \
            f"exp(xyz) 3-var produced {len(r.c)} terms"

    def test_TE05_four_var_feasibility(self):
        """exp(x1*x2*x3*x4) — is 4-var even feasible?"""
        t0 = time.time()
        args = [RR(0.5, var=i, nvars=4) for i in range(4)]
        product = args[0] * args[1] * args[2] * args[3]
        r = mc_exp(product)
        elapsed = time.time() - t0
        assert elapsed < 60, \
            f"4-var exp took {elapsed:.1f}s, {len(r.c)} terms"

    @pytest.mark.skip(reason="Takes ~2min; triple composition IS feasible but very slow")
    def test_TE06_triple_composition_very_slow(self):
        """exp(sin(cos(xy))) — feasible but ~2 minutes for 2 variables.
        Demonstrates the O(N^2) term explosion per composition layer.
        Double composition produces ~24k terms; third layer multiplies again."""
        t0 = time.time()
        r = mv_eval(
            lambda x, y: mc_exp(mc_sin(mc_cos(x * y))), [0.5, 0.5]
        )
        elapsed = time.time() - t0
        assert elapsed < 300, f"triple composition took {elapsed:.1f}s"
        fd = (-0.1597054666)  # from finite-difference ground truth
        assert r.partial(1, 0) == pytest.approx(fd, abs=1e-4)

    def test_TE07_no_truncation_mechanism(self):
        """MC has no equivalent of MAX_ACTIVE_DIMS truncation.
        Terms accumulate without bound through compositions."""
        x = RR(1.0, var=0, nvars=2)
        y = RR(1.0, var=1, nvars=2)
        r = mc_exp(x * y)
        min_dim_sum = min(sum(k) for k in r.c.keys())
        assert min_dim_sum < -10, \
            f"exp(xy) has terms down to total dim {min_dim_sum}, confirming no truncation"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 4: BLACK-SCHOLES AND FINANCIAL APPLICATIONS
# ═══════════════════════════════════════════════════════════════
# The whole point of multivariate composite for XVA is computing
# Greeks with multiple risk factors. This requires division (d1 formula).

class TestBlackScholes:
    """Black-Scholes Greeks via multivariate composite."""

    S, K, r_rate, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    def _d1_analytic(self):
        S, K, r, s, T = self.S, self.K, self.r_rate, self.sigma, self.T
        return (math.log(S / K) + (r + s ** 2 / 2) * T) / (s * math.sqrt(T))

    def test_BS01_d1_value(self):
        """d1 formula value should be correct."""
        S_mv = RR(self.S, var=0, nvars=2)
        sigma_mv = RR(self.sigma, var=1, nvars=2)
        K_mv = RR_const(self.K, nvars=2)
        r_mv = RR_const(self.r_rate, nvars=2)
        T_mv = RR_const(self.T, nvars=2)

        d1 = (mc_ln(S_mv / K_mv) + (r_mv + sigma_mv ** 2 / 2) * T_mv) / (
            sigma_mv * mc_sqrt(T_mv)
        )
        assert d1.st() == pytest.approx(self._d1_analytic(), abs=1e-6), \
            f"d1 value wrong: got {d1.st()}, expected {self._d1_analytic()}"

    def test_BS02_dd1_dS(self):
        """dd1/dS = 1/(S*sigma*sqrt(T))."""
        S_mv = RR(self.S, var=0, nvars=2)
        sigma_mv = RR(self.sigma, var=1, nvars=2)
        K_mv = RR_const(self.K, nvars=2)
        r_mv = RR_const(self.r_rate, nvars=2)
        T_mv = RR_const(self.T, nvars=2)

        d1 = (mc_ln(S_mv / K_mv) + (r_mv + sigma_mv ** 2 / 2) * T_mv) / (
            sigma_mv * mc_sqrt(T_mv)
        )

        dd1_dS = d1.partial(1, 0)
        expected = 1 / (self.S * self.sigma * math.sqrt(self.T))
        assert dd1_dS == pytest.approx(expected, abs=1e-6), \
            f"dd1/dS: got {dd1_dS}, expected {expected}"

    def test_BS03_dd1_dsigma(self):
        """dd1/dsigma via finite differences."""
        S_mv = RR(self.S, var=0, nvars=2)
        sigma_mv = RR(self.sigma, var=1, nvars=2)
        K_mv = RR_const(self.K, nvars=2)
        r_mv = RR_const(self.r_rate, nvars=2)
        T_mv = RR_const(self.T, nvars=2)

        d1 = (mc_ln(S_mv / K_mv) + (r_mv + sigma_mv ** 2 / 2) * T_mv) / (
            sigma_mv * mc_sqrt(T_mv)
        )

        dd1_dsigma = d1.partial(0, 1)

        h = 1e-7
        S, K, r, T = self.S, self.K, self.r_rate, self.T
        d1_fn = lambda sig: (math.log(S / K) + (r + sig ** 2 / 2) * T) / (
            sig * math.sqrt(T)
        )
        fd = (d1_fn(self.sigma + h) - d1_fn(self.sigma - h)) / (2 * h)

        assert dd1_dsigma == pytest.approx(fd, abs=1e-4), \
            f"dd1/dsigma: got {dd1_dsigma}, expected {fd}"

    def test_BS04_d1_term_count(self):
        """d1 should retain derivative information (more than 1 term)."""
        S_mv = RR(self.S, var=0, nvars=2)
        sigma_mv = RR(self.sigma, var=1, nvars=2)
        K_mv = RR_const(self.K, nvars=2)
        r_mv = RR_const(self.r_rate, nvars=2)
        T_mv = RR_const(self.T, nvars=2)

        d1 = (mc_ln(S_mv / K_mv) + (r_mv + sigma_mv ** 2 / 2) * T_mv) / (
            sigma_mv * mc_sqrt(T_mv)
        )
        assert len(d1.c) > 1, \
            f"d1 has only {len(d1.c)} term(s) — division destroyed derivatives"

    def test_BS05_single_var_delta_works(self):
        """Single-var composite correctly computes Delta = dd1/dS.
        This proves the MATH is correct; multivar is the broken path."""
        S_sv = Composite({0: self.S, -1: 1.0})
        K_sv = Composite({0: self.K})
        r_sv = Composite({0: self.r_rate})
        sigma_sv = Composite({0: self.sigma})
        T_sv = Composite({0: self.T})

        d1_sv = (ln(S_sv / K_sv) + (r_sv + sigma_sv ** 2 / 2) * T_sv) / (
            sigma_sv * sqrt(T_sv)
        )
        dd1_dS_sv = d1_sv.c.get(-1, 0.0)
        expected = 1 / (self.S * self.sigma * math.sqrt(self.T))
        assert dd1_dS_sv == pytest.approx(expected, abs=1e-6), \
            f"single-var dd1/dS={dd1_dS_sv}, expected {expected}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 5: DEEP TRANSCENDENTAL CHAINS
# ═══════════════════════════════════════════════════════════════
# Where interaction between variables goes through transcendentals.

class TestDeepTranscendentals:
    """Compositions of transcendentals with mixed variables."""

    def test_DT01_sin_cos_xy_df_dx(self):
        """sin(cos(xy)) at (1,1): df/dx = -y*sin(xy)*cos(cos(xy))."""
        mv = mv_partial(lambda x, y: mc_sin(mc_cos(x * y)), [1, 1], (1, 0))
        expected = -math.sin(1) * math.cos(math.cos(1))
        assert mv == pytest.approx(expected, abs=1e-4), \
            f"mv={mv}, expected={expected}"

    def test_DT02_sin_cos_xy_mixed_partial(self):
        """d2f/dxdy of sin(cos(xy)) at (1,1) vs finite differences."""
        mv = mv_partial(
            lambda x, y: mc_sin(mc_cos(x * y)), [1, 1], (1, 1)
        )
        fd = fd_mixed(
            lambda x, y: math.sin(math.cos(x * y)), [1, 1], 0, 1
        )
        assert mv == pytest.approx(fd, abs=1e-3), \
            f"mv={mv}, fd={fd}"

    def test_DT03_exp_sin_xy_df_dx(self):
        """exp(sin(xy)) at (0.5, 0.5): df/dx = y*cos(xy)*exp(sin(xy))."""
        mv = mv_partial(
            lambda x, y: mc_exp(mc_sin(x * y)), [0.5, 0.5], (1, 0)
        )
        val = 0.5 * 0.5
        expected = 0.5 * math.cos(val) * math.exp(math.sin(val))
        assert mv == pytest.approx(expected, abs=1e-4), \
            f"mv={mv}, expected={expected}"

    def test_DT04_ln_exp_x_plus_exp_y(self):
        """ln(exp(x)+exp(y)) at (1,1): df/dx = exp(x)/(exp(x)+exp(y)) = 1/2."""
        mv = mv_partial(
            lambda x, y: mc_ln(mc_exp(x) + mc_exp(y)), [1, 1], (1, 0)
        )
        assert mv == pytest.approx(0.5, abs=1e-6), \
            f"mv={mv}, expected=0.5"

    def test_DT05_sqrt_sin_x_sq_plus_cos_y_sq(self):
        """sqrt(sin(x)^2 + cos(y)^2) at (pi/4, pi/4).
        df/dx = sin(x)*cos(x) / sqrt(sin(x)^2 + cos(y)^2)."""
        pt = [math.pi / 4, math.pi / 4]
        mv = mv_partial(
            lambda x, y: mc_sqrt(mc_sin(x) ** 2 + mc_cos(y) ** 2),
            pt, (1, 0),
        )
        s = math.sin(pt[0])
        c = math.cos(pt[0])
        c2 = math.cos(pt[1])
        expected = s * c / math.sqrt(s ** 2 + c2 ** 2)
        assert mv == pytest.approx(expected, abs=1e-4), \
            f"mv={mv}, expected={expected}"

    def test_DT06_exp_x_times_sin_y_second_order(self):
        """d2f/dx2 of exp(x)*sin(y) at (1,2) = exp(1)*sin(2)."""
        mv = mv_partial(
            lambda x, y: mc_exp(x) * mc_sin(y), [1, 2], (2, 0)
        )
        expected = math.e * math.sin(2)
        assert mv == pytest.approx(expected, abs=1e-4), \
            f"mv={mv}, expected={expected}"

    def test_DT07_separate_variables_no_interaction(self):
        """f(x,y) = sin(x) + cos(y): d2f/dxdy = 0 (no interaction)."""
        mv = mv_partial(
            lambda x, y: mc_sin(x) + mc_cos(y), [1, 1], (1, 1)
        )
        assert abs(mv) < 1e-8, \
            f"separate-variable function has nonzero mixed partial: {mv}"

    def test_DT08_composition_precision_vs_fd(self):
        """sin(exp(x)*cos(y)) at (0.5, 0.5): compare gradient to FD."""
        f_scalar = lambda x, y: math.sin(math.exp(x) * math.cos(y))
        f_mv = lambda x, y: mc_sin(mc_exp(x) * mc_cos(y))
        pt = [0.5, 0.5]

        mv_dx = mv_partial(f_mv, pt, (1, 0))
        fd_dx = fd_partial(f_scalar, pt, 0)
        assert mv_dx == pytest.approx(fd_dx, abs=1e-4), \
            f"df/dx: mv={mv_dx}, fd={fd_dx}"

        mv_dy = mv_partial(f_mv, pt, (0, 1))
        fd_dy = fd_partial(f_scalar, pt, 1)
        assert mv_dy == pytest.approx(fd_dy, abs=1e-4), \
            f"df/dy: mv={mv_dy}, fd={fd_dy}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 6: HIGH-ORDER DERIVATIVES
# ═══════════════════════════════════════════════════════════════

class TestHighOrderDerivatives:
    """High-order and mixed partials — precision boundary."""

    def test_HO01_fourth_order_polynomial(self):
        """d4f/dx2dy2 of x^3*y^3 at (1,1) = 36."""
        mv = mv_partial(lambda x, y: x ** 3 * y ** 3, [1, 1], (2, 2))
        assert mv == pytest.approx(36.0, abs=1e-6)

    def test_HO02_fifth_order_single_dir(self):
        """d5f/dx5 of x^6*y at (1,2) = 720*y = 1440."""
        mv = mv_partial(lambda x, y: x ** 6 * y, [1, 2], (5, 0))
        assert mv == pytest.approx(1440.0, abs=1e-4)

    def test_HO03_third_order_mixed_poly(self):
        """d3f/dx2dy of x^3*y^2 at (2,3) = 12*y = 36... wait.
        f = x^3*y^2, d/dx = 3x^2*y^2, d2/dx2 = 6x*y^2, d3/dx2dy = 12xy.
        At (2,3): 72."""
        mv = mv_partial(lambda x, y: x ** 3 * y ** 2, [2, 3], (2, 1))
        assert mv == pytest.approx(72.0, abs=1e-6)

    def test_HO04_third_order_three_vars(self):
        """d3f/dxdydz of xyz at (2,3,4) = 1."""
        mv = mv_partial(
            lambda x, y, z: x * y * z, [2, 3, 4], (1, 1, 1)
        )
        assert mv == pytest.approx(1.0, abs=1e-8)

    def test_HO05_high_order_transcendental(self):
        """d3f/dx3 of exp(x)*y at (1,2) = 2*e (polynomial in y, exp in x)."""
        mv = mv_partial(lambda x, y: mc_exp(x) * y, [1, 2], (3, 0))
        expected = 2 * math.e
        assert mv == pytest.approx(expected, abs=1e-3), \
            f"mv={mv}, expected={expected}"

    def test_HO06_second_order_sin_xy(self):
        """d2f/dx2 of sin(xy) at (1,2).
        d/dx = y*cos(xy), d2/dx2 = -y^2*sin(xy).
        At (1,2): -4*sin(2)."""
        mv = mv_partial(lambda x, y: mc_sin(x * y), [1, 2], (2, 0))
        expected = -4 * math.sin(2)
        assert mv == pytest.approx(expected, abs=1e-3), \
            f"mv={mv}, expected={expected}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 7: HESSIAN AND GRADIENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════

class TestGradientHessian:
    """Gradient and Hessian extraction vs known values."""

    def test_GH01_gradient_norm_equals_components(self):
        """Gradient of x^2+y^2 at (3,4) should be [6, 8]."""
        r = mv_eval(lambda x, y: x ** 2 + y ** 2, [3, 4])
        grad = r.gradient()
        assert grad[0] == pytest.approx(6.0, abs=1e-8)
        assert grad[1] == pytest.approx(8.0, abs=1e-8)

    def test_GH02_hessian_symmetric(self):
        """Hessian of exp(xy) at (1,1) must be symmetric."""
        r = mv_eval(lambda x, y: mc_exp(x * y), [1, 1])
        H = r.hessian()
        assert H[0][1] == pytest.approx(H[1][0], abs=1e-6), \
            f"Hessian not symmetric: H[0][1]={H[0][1]}, H[1][0]={H[1][0]}"

    def test_GH03_hessian_exp_xy_values(self):
        """Hessian of exp(xy) at (1,1).
        d2f/dx2 = y^2*exp(xy) = e
        d2f/dy2 = x^2*exp(xy) = e
        d2f/dxdy = (1+xy)*exp(xy) = 2e."""
        r = mv_eval(lambda x, y: mc_exp(x * y), [1, 1])
        H = r.hessian()
        assert H[0][0] == pytest.approx(math.e, abs=1e-4), \
            f"d2f/dx2={H[0][0]}, expected {math.e}"
        assert H[1][1] == pytest.approx(math.e, abs=1e-4), \
            f"d2f/dy2={H[1][1]}, expected {math.e}"
        assert H[0][1] == pytest.approx(2 * math.e, abs=1e-4), \
            f"d2f/dxdy={H[0][1]}, expected {2*math.e}"

    def test_GH04_laplacian_harmonic_function(self):
        """x^2-y^2 is harmonic: Laplacian = 0."""
        r = mv_eval(lambda x, y: x ** 2 - y ** 2, [3, 4])
        assert r.laplacian() == pytest.approx(0.0, abs=1e-10), \
            f"laplacian={r.laplacian()}, should be 0 for harmonic function"

    def test_GH05_hessian_with_division_fails(self):
        """Hessian of x/y at (2,3).
        d2f/dx2 = 0
        d2f/dy2 = 2x/y^3 = 4/27
        d2f/dxdy = -1/y^2 = -1/9.
        Division likely breaks all of these."""
        r = mv_eval(lambda x, y: x / y, [2, 3])
        H = r.hessian()
        assert H[0][0] == pytest.approx(0.0, abs=1e-8), \
            f"d2f/dx2={H[0][0]}, expected 0"
        assert H[1][1] == pytest.approx(4 / 27, abs=1e-6), \
            f"d2f/dy2={H[1][1]}, expected {4/27}"
        assert H[0][1] == pytest.approx(-1 / 9, abs=1e-6), \
            f"d2f/dxdy={H[0][1]}, expected {-1/9}"


# ═══════════════════════════════════════════════════════════════
# CATEGORY 8: EDGE CASES AND ZERO HANDLING
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: zero handling, cancellation, identity violations."""

    def test_EC01_pythagorean_identity(self):
        """sin(x)^2 + cos(x)^2 = 1 in multivariate context."""
        r = mv_eval(
            lambda x, y: mc_sin(x) ** 2 + mc_cos(x) ** 2, [1, 1]
        )
        assert r.st() == pytest.approx(1.0, abs=1e-10)
        assert abs(r.partial(1, 0)) < 1e-8, "derivative of constant should be 0"

    def test_EC02_subtraction_cancellation(self):
        """f - f = 0 with all zero derivatives."""
        r = mv_eval(
            lambda x, y: mc_exp(x * y) - mc_exp(x * y), [1, 1]
        )
        assert abs(r.st()) < 1e-10
        assert abs(r.partial(1, 0)) < 1e-8
        assert abs(r.partial(0, 1)) < 1e-8

    def test_EC03_multiply_by_zero(self):
        """MC constructor strips zeros: 0 * f annihilates entirely."""
        r = mv_eval(lambda x, y: 0 * mc_exp(x * y), [1, 1])
        assert len(r.c) == 0 or abs(r.st()) < 1e-10

    def test_EC04_division_by_constant_preserves_derivs(self):
        """x^2/5 at (3,1): df/dx = 2x/5 = 6/5."""
        mv = mv_partial(lambda x, y: x ** 2 / 5, [3, 1], (1, 0))
        assert mv == pytest.approx(6 / 5, abs=1e-8)

    def test_EC05_near_zero_coefficients_accumulate(self):
        """After many operations, do near-zero coefficients accumulate?
        sin(x) - sin(x) should have truly empty terms."""
        r = mv_eval(
            lambda x, y: mc_sin(x) - mc_sin(x), [1, 1]
        )
        nonzero_count = sum(1 for v in r.c.values() if abs(v) > 1e-15)
        assert nonzero_count == 0, \
            f"{nonzero_count} near-zero terms survived cancellation"

    def test_EC06_value_at_zero_point(self):
        """Evaluation at (0, 0) where st=0 but derivatives should still work."""
        r = mv_eval(lambda x, y: x + y, [0, 0])
        assert r.st() == pytest.approx(0.0, abs=1e-10)
        grad = r.gradient()
        assert grad[0] == pytest.approx(1.0, abs=1e-10)
        assert grad[1] == pytest.approx(1.0, abs=1e-10)

    def test_EC07_large_coefficients(self):
        """x^10*y^10 at (2, 2): value = 2^20 = 1048576."""
        r = mv_eval(lambda x, y: x ** 10 * y ** 10, [2, 2])
        assert r.st() == pytest.approx(2 ** 20, rel=1e-6)

    def test_EC08_negative_evaluation_point(self):
        """x^3*y at (-2, 3): value = -24, df/dx = 3*(-2)^2*3 = 36."""
        r = mv_eval(lambda x, y: x ** 3 * y, [-2, 3])
        assert r.st() == pytest.approx(-24, abs=1e-8)
        assert r.partial(1, 0) == pytest.approx(36, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# CATEGORY 9: FUNCTIONS THAT SHOULD WORK (BOUNDARY DOCUMENTATION)
# ═══════════════════════════════════════════════════════════════
# These test the WORKING boundary of multivariate composite.
# They document where it IS correct to use multivar.

class TestWorkingBoundary:
    """Functions where multivariate composite IS correct."""

    def test_WB01_polynomial_all_orders(self):
        """Pure polynomials always give exact derivatives."""
        mv = mv_partial(lambda x, y: x ** 4 * y ** 3, [2, 1], (3, 2))
        # d5f/dx3dy2 = 4*3*2 * 3*2 * x * y = 144
        # Actually: f=x^4*y^3, d/dx=4x^3*y^3, d2/dx2=12x^2*y^3,
        # d3/dx3=24x*y^3, d4/dx3dy=72x*y^2, d5/dx3dy2=144x*y
        # at (2,1): 288
        assert mv == pytest.approx(288.0, abs=1e-4)

    def test_WB02_separable_transcendentals(self):
        """f(x,y) = g(x) * h(y) — separable products always work."""
        mv_dx = mv_partial(
            lambda x, y: mc_sin(x) * mc_cos(y), [1, 2], (1, 0)
        )
        expected = math.cos(1) * math.cos(2)
        assert mv_dx == pytest.approx(expected, abs=1e-6)

    def test_WB03_additive_functions(self):
        """f(x,y) = g(x) + h(y) — additive always works."""
        mv = mv_partial(
            lambda x, y: mc_exp(x) + mc_sin(y), [1, 2], (1, 0)
        )
        assert mv == pytest.approx(math.e, abs=1e-6)

    def test_WB04_composition_without_division(self):
        """sin(x+y) — composition without division works."""
        mv = mv_partial(lambda x, y: mc_sin(x + y), [1, 2], (1, 0))
        assert mv == pytest.approx(math.cos(3), abs=1e-6)

    def test_WB05_exp_of_product(self):
        """exp(xy) — products inside transcendentals work."""
        mv = mv_partial(lambda x, y: mc_exp(x * y), [1, 2], (1, 0))
        expected = 2 * math.exp(2)
        assert mv == pytest.approx(expected, abs=1e-4)

    def test_WB06_polynomial_with_constants(self):
        """(x+2)^3*(y-1)^2 at (1,3) — polynomial with shifts."""
        mv = mv_partial(
            lambda x, y: (x + 2) ** 3 * (y - 1) ** 2, [1, 3], (1, 0)
        )
        # d/dx = 3(x+2)^2*(y-1)^2 at (1,3) = 3*9*4 = 108
        assert mv == pytest.approx(108.0, abs=1e-6)


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
