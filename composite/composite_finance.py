# Composite Machine — Financial Analysis Layer (v2 — Composite-Native)
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
composite_finance.py — Financial Domain Layer (v2: Composite-Native)
=====================================================================
All operations stay in the composite ring until the serialization
boundary. This means every financial computation can be composed
with further composite operations and still propagate derivatives.

Design principle:
    Composite in → Composite out → extract at display/serialization only.

Usage:
    from composite.composite_finance import FinancialComposite, ScenarioEngine

    fc = FinancialComposite.from_composite(option_price_composite)
    print(fc.price, fc.delta, fc.gamma)  # these are scalar reads (boundary)

    # But scenario_pnl stays composite:
    pnl = fc.scenario_pnl(R(0.01) + ZERO)  # composite shock!
    pnl.st()    # PnL estimate
    pnl.d(1)    # d(PnL)/d(shock_size) — impossible in v1

Author: Toni Milovan
License: AGPL-3.0 (commercial licensing: tmilovan@fwd.hr)
"""

import math
from composite.composite_lib import Composite, R, ZERO


def _ensure_composite(x):
    """Convert scalar to Composite if needed. Passthrough if already Composite."""
    if isinstance(x, Composite):
        return x
    return R(x)


# =============================================================================
# 1. FINANCIAL COMPOSITE — NAMED ACCESSORS + COMPOSITE-NATIVE OPERATIONS
# =============================================================================

class FinancialComposite(Composite):
    """
    A Composite with named financial accessors and composite-native operations.

    KEY DESIGN RULE: All methods that perform computation return Composite
    objects (or FinancialComposite). Scalar extraction happens ONLY in
    explicitly-named boundary methods (.price, .delta, .gamma, etc.)
    and serialization (.to_greeks_dict).

    Dimension mapping (single underlying):
        dim  0  -> price / present value
        dim -1  -> delta (first derivative) / 1!
        dim -2  -> gamma / 2!
        dim -3  -> speed / 3!

    Named properties (.price, .delta, .gamma, .speed) are SCALAR READS
    at the boundary — use them for display, not for further computation.
    For computation, use the FinancialComposite itself as a Composite value.
    """

    @classmethod
    def from_composite(cls, c):
        """Wrap an existing Composite as FinancialComposite.
        Backend-agnostic: .c always returns a dict view regardless
        of backend, and Composite.__init__ routes to active backend."""
        fc = cls.__new__(cls)
        Composite.__init__(fc, dict(c.c))
        return fc

    # --- Scalar boundary reads (for display/serialization) ---

    @property
    def price(self):
        """BOUNDARY: scalar price for display. Use self for computation."""
        return self.st()

    @property
    def pv(self):
        """Alias for price."""
        return self.st()

    @property
    def delta(self):
        """BOUNDARY: scalar delta for display."""
        return self.d(1)

    @property
    def gamma(self):
        """BOUNDARY: scalar gamma for display."""
        return self.d(2)

    @property
    def speed(self):
        """BOUNDARY: scalar speed for display."""
        return self.d(3)

    # --- Composite-native financial operations ---
    # These return Composite, preserving the derivative chain.

    def duration_composite(self):
        """
        Modified duration as a COMPOSITE value: -dP/dy / P.

        Uses composite deconvolution (division), so if this bond
        was priced with composite yield, the result carries
        d(duration)/d(yield) and all higher-order sensitivities.

        Returns: Composite (not a float)
        """
        d1_coeff = self.coeff(-1)
        if d1_coeff == 0:
            return R(0)
        dP_dy = Composite({0: self.d(1)})
        return -(dP_dy / self)

    def convexity_composite(self):
        """
        Convexity as a COMPOSITE value: d2P/dy2 / P.

        Returns: Composite (not a float)
        """
        d2P_dy2 = Composite({0: self.d(2)})
        return d2P_dy2 / self

    @property
    def duration(self):
        """BOUNDARY: scalar duration for display."""
        p = self.st()
        if abs(p) < 1e-15:
            return float('inf')
        return -self.d(1) / p

    @property
    def convexity(self):
        """BOUNDARY: scalar convexity for display."""
        p = self.st()
        if abs(p) < 1e-15:
            return float('inf')
        return self.d(2) / p

    # --- Composite-native scenario analysis ---

    def scenario_pnl(self, shock):
        h = _ensure_composite(shock)

        # Fast path: if shock is a plain scalar (no derivative dims),
        # do simple float polynomial evaluation — no convolutions needed
        if len(h.c) <= 1 and 0 in h.c or len(h.c) == 0:
            h_val = h.st()
            pnl_val = 0.0
            h_power = h_val
            for dim in self.c:
                if dim < 0 and -dim > 0:
                    pass  # just need max_order
            max_order = 0
            for dim in self.c:
                if dim < 0 and -dim > max_order:
                    max_order = -dim
            h_power = h_val
            for n in range(1, max_order + 1):
                c_n = self.coeff(-n)
                if c_n != 0:
                    pnl_val += c_n * h_power
                h_power *= h_val
            return R(pnl_val)

        # Composite path: full convolution (for composite shocks)
        pnl = R(0)
        h_power = h
        max_order = 0
        for dim in self.c:
            if dim < 0 and -dim > max_order:
                max_order = -dim
        for n in range(1, max_order + 1):
            c_n = self.coeff(-n)
            if c_n != 0:
                pnl = pnl + R(c_n) * h_power
            h_power = h_power * h
        return pnl

    def scenario_price(self, shock):
        """
        Estimated new price after shock — COMPOSITE-NATIVE.

        Returns Composite: the full price composite at the shifted point.
        """
        return R(self.st()) + self.scenario_pnl(shock)

    # --- Scalar boundary versions (convenience for display) ---

    def scenario_pnl_scalar(self, shock_float):
        """BOUNDARY: scalar PnL for a scalar shock. For display."""
        return self.scenario_pnl(shock_float).st()

    def scenario_price_scalar(self, shock_float):
        """BOUNDARY: scalar price for a scalar shock. For display."""
        return self.scenario_price(shock_float).st()

    # --- Serialization (boundary) ---

    def to_greeks_dict(self, max_order=4):
        """BOUNDARY: serialize to a dict with financial names."""
        names = {0: 'price', 1: 'delta', 2: 'gamma', 3: 'speed'}
        result = {}
        for n in range(max_order + 1):
            key = names.get(n, f'd{n}')
            if n == 0:
                result[key] = self.st()
            else:
                result[key] = self.d(n)
        return result

    def __repr__(self):
        base = super().__repr__()
        return (f"{base}  "
                f"[price={self.price:.6g}, "
                f"delta={self.delta:.6g}, "
                f"gamma={self.gamma:.6g}]")


# =============================================================================
# 2. SCENARIO ENGINE — COMPOSITE-NATIVE PORTFOLIO ANALYSIS
# =============================================================================

class ScenarioEngine:
    """
    Portfolio-level scenario analysis — COMPOSITE-NATIVE.

    All aggregation methods return Composite objects.
    Weights CAN be Composite (for position-sizing sensitivity).

    Usage:
        positions = [fc1, fc2, fc3]
        weights = [R(100), R(-50), R(200)]  # can be Composite!
        engine = ScenarioEngine(positions, weights)

        # All return Composite:
        engine.total_value()                # portfolio value (Composite)
        engine.total_delta_composite()      # aggregate delta (Composite)
        engine.scenario_pnl(R(0.01) + ZERO) # PnL with shock sensitivity

        # Scalar reads at the boundary:
        engine.total_value().st()            # scalar portfolio value
    """

    def __init__(self, composites, weights=None):
        """
        Args:
            composites: list of FinancialComposite (or Composite).
            weights:    list of floats or Composites.
                        If Composite, enables d(portfolio)/d(weight_i).
                        If None, all weights = R(1).
        """
        self.composites = [
            c if isinstance(c, FinancialComposite)
            else FinancialComposite.from_composite(c)
            for c in composites
        ]
        if weights is None:
            self.weights = [R(1.0)] * len(composites)
        else:
            self.weights = [_ensure_composite(w) for w in weights]

    def _weighted_composite_sum(self, extractor):
        """
        Composite-native weighted sum.

        extractor: function that takes a FinancialComposite and returns
                   a Composite value (NOT a scalar).

        Returns: Composite
        """
        total = R(0)
        for c, w in zip(self.composites, self.weights):
            total = total + w * extractor(c)
        return total

    def total_value(self):
        """
        Aggregate portfolio value as Composite.

        If weights are composite, result carries d(value)/d(weight_i).
        """
        return self._weighted_composite_sum(
            lambda c: R(c.st())
        )

    def total_delta_composite(self):
        """Aggregate portfolio delta as Composite."""
        return self._weighted_composite_sum(
            lambda c: R(c.d(1))
        )

    def total_gamma_composite(self):
        """Aggregate portfolio gamma as Composite."""
        return self._weighted_composite_sum(
            lambda c: R(c.d(2))
        )

    def total_nth_composite(self, n):
        """Aggregate nth derivative as Composite."""
        return self._weighted_composite_sum(
            lambda c: R(c.d(n))
        )

    # --- Scalar boundary reads ---

    def total_price(self):
        """BOUNDARY: scalar portfolio value."""
        return self.total_value().st()

    def total_delta(self):
        """BOUNDARY: scalar portfolio delta."""
        return self.total_delta_composite().st()

    def total_gamma(self):
        """BOUNDARY: scalar portfolio gamma."""
        return self.total_gamma_composite().st()

    # --- Composite-native scenario analysis ---

    def scenario_pnl(self, shock):
        """
        Portfolio P&L for a shock — COMPOSITE-NATIVE.

        shock can be float or Composite.
        Returns Composite (carries dPnL/dshock if shock is composite).
        """
        h = _ensure_composite(shock)
        total = R(0)
        for c, w in zip(self.composites, self.weights):
            total = total + w * c.scenario_pnl(h)
        return total

    def scenario_ladder(self, shocks):
        """
        P&L at multiple shock levels.

        Returns list of (shock, Composite_pnl) pairs.
        Call .st() on each pnl for the scalar value.
        """
        return [(s, self.scenario_pnl(s)) for s in shocks]

    def taylor_var(self, vol, confidence=0.99, horizon_days=1):
        """
        Parametric VaR — COMPOSITE-NATIVE.

        If vol is Composite, result is Composite and carries dVaR/dvol.
        If confidence is Composite, result carries dVaR/dconfidence.

        Uses math.erf-based quantile (NOT the unreliable A&S 26.2.17).

        Args:
            vol:          float or Composite — annualized volatility
            confidence:   float — confidence level (0.99 = 99%)
            horizon_days: float or Composite — holding period

        Returns:
            Composite — estimated VaR
        """
        vol_c = _ensure_composite(vol)
        horizon_c = _ensure_composite(horizon_days)

        # Normal quantile — scalar computation
        p = 1.0 - confidence
        if p <= 0 or p >= 1:
            raise ValueError("confidence must be between 0 and 1")
        t = math.sqrt(-2 * math.log(p))
        z = (t
             - (2.515517 + 0.802853 * t + 0.010328 * t**2)
             / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3))

        # Scale vol to horizon — COMPOSITE arithmetic
        daily_vol = vol_c * R(math.sqrt(horizon_c.st() / 252))
        shock = R(-1) * daily_vol * R(z)   # loss scenario

        # Full Taylor P&L at that shock — COMPOSITE
        pnl = self.scenario_pnl(shock)

        # VaR = -pnl (assumes loss scenario)
        return R(-1) * pnl


# =============================================================================
# 3. COMPOSITE-NATIVE HELPERS
# =============================================================================

def portfolio_weighted_composite(composites, weights):
    """
    Build a single weighted-sum Composite from positions.

    This is the composite-native equivalent of:
        sum(price_i * weight_i for i in positions)

    But the result is Composite, so if any weight is Composite,
    you get d(portfolio_value)/d(weight_i) for free.

    Args:
        composites: list of Composite or FinancialComposite
        weights:    list of float or Composite

    Returns:
        Composite — the weighted portfolio value
    """
    total = R(0)
    for c, w in zip(composites, weights):
        w_c = _ensure_composite(w)
        total = total + w_c * c
    return total


def composite_polynomial_eval(coefficients, x):
    """
    Evaluate a polynomial with given coefficients at point x.

    COMPOSITE-NATIVE: both coefficients and x can be Composite.

    coefficients: list [a0, a1, a2, ...] where p(x) = a0 + a1*x + a2*x^2 + ...
    x:            float or Composite

    Returns: Composite
    """
    x_c = _ensure_composite(x)
    result = R(0)
    x_power = R(1)   # x^0
    for coeff in coefficients:
        c = _ensure_composite(coeff)
        result = result + c * x_power
        x_power = x_power * x_c
    return result


# =============================================================================
# 4. PANDAS INTEGRATION (BOUNDARY LAYER)
# =============================================================================

def composites_to_dataframe(composites, labels=None, max_order=4):
    """
    BOUNDARY: Convert composites to pandas DataFrame.
    This is a serialization boundary — extracts scalars for display.
    """
    import pandas as pd

    names = {0: 'price', 1: 'delta', 2: 'gamma', 3: 'speed'}
    columns = [names.get(n, f'd{n}') for n in range(max_order + 1)]

    rows = []
    for c in composites:
        row = [c.st()] + [c.d(n) for n in range(1, max_order + 1)]
        rows.append(row)

    return pd.DataFrame(rows, columns=columns, index=labels)


def dataframe_to_composites(df, column_map=None):
    """
    BOUNDARY: Convert DataFrame back to FinancialComposites.
    This is a deserialization boundary — constructs composites from scalars.
    """
    if column_map is None:
        column_map = {}
        standard = {
            'price': 0, 'pv': 0,
            'delta': 1, 'gamma': 2,
            'speed': 3, 'duration': 1, 'convexity': 2
        }
        for col in df.columns:
            if col.lower() in standard:
                column_map[col] = standard[col.lower()]
            elif col.startswith('d') and col[1:].isdigit():
                column_map[col] = int(col[1:])

    result = []
    for _, row in df.iterrows():
        coeffs = {}
        for col, order in column_map.items():
            val = row[col]
            if val != 0:
                if order == 0:
                    coeffs[0] = float(val)
                else:
                    coeffs[-order] = float(val) / math.factorial(order)
        result.append(FinancialComposite(coeffs))

    return result
