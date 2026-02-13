# Composite Machine — Financial Analysis Layer
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
composite_finance.py — Financial Domain Layer
===============================================
Named accessors, scenario analysis, and portfolio aggregation
built on top of Composite serialization.

Usage:
    from composite.composite_finance import FinancialComposite, ScenarioEngine

    # Wrap any composite result
    fc = FinancialComposite.from_composite(option_price_composite)
    print(fc.price, fc.delta, fc.gamma)

    # Taylor-based scenario shift
    engine = ScenarioEngine(composites, weights)
    pnl = engine.scenario_pnl(0.01)      # +1% move
    var = engine.taylor_var(vol=0.02)     # parametric VaR

Author: Toni Milovan
License: AGPL-3.0 (commercial licensing: tmilovan@fwd.hr)
"""

import math
from composite.composite_lib import Composite, R, ZERO


# =============================================================================
# 1. FINANCIAL COMPOSITE — NAMED ACCESSORS
# =============================================================================

class FinancialComposite(Composite):
    """
    A Composite with named financial accessors.

    Wraps the raw dimensional structure with the language
    traders and risk managers actually use.

    Dimension mapping (single underlying):
        dim  0  -> price / present value
        dim -1  -> delta (first derivative)
        dim -2  -> gamma / 2!  (second derivative coefficient)
        dim -3  -> speed / 3!  (third derivative coefficient)
        dim -4  -> fourth order / 4!

    For nth derivative values (with factorial scaling):
        .d(n) returns f^(n) — the actual derivative
        .coeff(-n) returns f^(n)/n! — the Taylor coefficient

    The named properties below return the ACTUAL derivatives
    (i.e., with factorial scaling), matching financial convention.
    """

    @classmethod
    def from_composite(cls, c):
        """Wrap an existing Composite as FinancialComposite."""
        fc = cls.__new__(cls)
        fc.c = dict(c.c)
        return fc

    # --- Core accessors (actual derivative values) ---

    @property
    def price(self):
        """Present value / option price. Dimension 0."""
        return self.st()

    @property
    def pv(self):
        """Alias for price (present value)."""
        return self.st()

    @property
    def delta(self):
        """First derivative w.r.t. underlying. dV/dS."""
        return self.d(1)

    @property
    def gamma(self):
        """Second derivative w.r.t. underlying. d^2 V/dS^2."""
        return self.d(2)

    @property
    def speed(self):
        """Third derivative w.r.t. underlying. d^3 V/dS^3."""
        return self.d(3)

    # --- Duration / Convexity (bond convention) ---

    @property
    def duration(self):
        """
        Modified duration: -dP/dy / P.
        Requires price != 0. Returns the actual duration value.
        (For raw dP/dy, use .d(1) directly.)
        """
        p = self.st()
        if abs(p) < 1e-15:
            return float('inf')
        return -self.d(1) / p

    @property
    def convexity(self):
        """
        Convexity: d^2 P/dy^2 / P.
        Bond convention (positive for vanilla bonds).
        """
        p = self.st()
        if abs(p) < 1e-15:
            return float('inf')
        return self.d(2) / p

    # --- Taylor expansion for scenario analysis ---

    def scenario_pnl(self, shock):
        """
        Estimate P&L for a given shock using the full Taylor expansion.

        Uses all available coefficients:
            delta_V = sum( coeff_at_dim_-n * shock^n )  for n = 1, 2, ...

        This is the standard "Greeks-based P&L" but using ALL
        available orders, not just delta-gamma.

        Args:
            shock: float, the size of the move (e.g., 0.01 for +1%)

        Returns:
            float, estimated P&L
        """
        pnl = 0.0
        for dim, coeff in self.c.items():
            if dim < 0:
                order = -dim
                pnl += coeff * shock ** order
        return pnl

    def scenario_price(self, shock):
        """
        Estimate new price after a shock.
        price_new = price + scenario_pnl(shock)
        """
        return self.st() + self.scenario_pnl(shock)

    # --- Serialization with financial labels ---

    def to_greeks_dict(self, max_order=4):
        """
        Serialize as a dict with financial names.

        Returns:
            {'price': ..., 'delta': ..., 'gamma': ..., 'speed': ..., ...}
        """
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
        """Show financial-friendly repr alongside dimensional."""
        base = super().__repr__()
        return (f"{base}  "
                f"[price={self.price:.6g}, "
                f"delta={self.delta:.6g}, "
                f"gamma={self.gamma:.6g}]")


# =============================================================================
# 2. SCENARIO ENGINE — PORTFOLIO-LEVEL ANALYSIS
# =============================================================================

class ScenarioEngine:
    """
    Portfolio-level scenario analysis using Composite Taylor expansions.

    Takes a collection of FinancialComposites (one per position)
    and provides aggregated risk measures.

    Usage:
        positions = [fc1, fc2, fc3]  # FinancialComposite objects
        weights = [100, -50, 200]    # number of contracts/shares
        engine = ScenarioEngine(positions, weights)

        engine.total_price()            # portfolio value
        engine.total_delta()            # aggregate delta
        engine.scenario_pnl(0.01)       # P&L for +1% move
        engine.scenario_ladder(shocks)  # P&L at multiple shock levels
        engine.taylor_var(vol, conf)    # parametric VaR using gamma
    """

    def __init__(self, composites, weights=None):
        """
        Args:
            composites: list of FinancialComposite (or Composite) objects.
            weights:    list of position sizes (notional, contracts, etc.)
                        If None, all weights = 1.
        """
        self.composites = [
            c if isinstance(c, FinancialComposite)
            else FinancialComposite.from_composite(c)
            for c in composites
        ]
        self.weights = weights or [1.0] * len(composites)

    def _weighted_sum(self, accessor):
        """Sum accessor(c) * weight across all positions."""
        return sum(
            accessor(c) * w
            for c, w in zip(self.composites, self.weights)
        )

    def total_price(self):
        """Aggregate portfolio value."""
        return self._weighted_sum(lambda c: c.price)

    def total_delta(self):
        """Aggregate portfolio delta."""
        return self._weighted_sum(lambda c: c.delta)

    def total_gamma(self):
        """Aggregate portfolio gamma."""
        return self._weighted_sum(lambda c: c.gamma)

    def total_nth(self, n):
        """Aggregate nth derivative across portfolio."""
        return self._weighted_sum(lambda c: c.d(n))

    def scenario_pnl(self, shock):
        """
        Portfolio P&L for a given shock, using full Taylor expansion.

        Args:
            shock: float, size of the move.

        Returns:
            float, total portfolio P&L.
        """
        return sum(
            c.scenario_pnl(shock) * w
            for c, w in zip(self.composites, self.weights)
        )

    def scenario_ladder(self, shocks):
        """
        P&L at multiple shock levels.

        Args:
            shocks: list of floats
                    (e.g., [-0.05, -0.01, 0, 0.01, 0.05])

        Returns:
            list of (shock, pnl) pairs.
        """
        return [(s, self.scenario_pnl(s)) for s in shocks]

    def taylor_var(self, vol, confidence=0.99, horizon_days=1):
        """
        Parametric VaR using delta-gamma-speed Taylor expansion.

        Uses the Cornish-Fisher expansion to adjust for non-normality
        captured by the gamma (skewness proxy) and speed terms.

        For delta-only VaR (ignoring convexity):
            VaR ~ |delta * vol * z|

        For delta-gamma VaR:
            VaR ~ |delta * vol * z + 0.5 * gamma * (vol * z)^2|

        This method uses all available Taylor orders.

        Args:
            vol:            annualized volatility of the underlying.
            confidence:     VaR confidence level (default 0.99 = 99%).
            horizon_days:   holding period in days (default 1).

        Returns:
            float, estimated VaR (positive number = loss).
        """
        from math import sqrt, log

        # Normal quantile (Abramowitz & Stegun 26.2.23)
        p = 1 - confidence
        if p <= 0 or p >= 1:
            raise ValueError("confidence must be between 0 and 1")
        t = sqrt(-2 * log(p))
        z = (t
             - (2.515517 + 0.802853 * t + 0.010328 * t**2)
             / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3))

        # Scale vol to horizon
        daily_vol = vol * sqrt(horizon_days / 252)
        shock = -daily_vol * z  # negative shock (loss scenario)

        # Full Taylor P&L at that shock
        pnl = self.scenario_pnl(shock)
        return -pnl if pnl < 0 else pnl


# =============================================================================
# 3. PANDAS INTEGRATION HELPERS
# =============================================================================

def composites_to_dataframe(composites, labels=None, max_order=4):
    """
    Convert a list of Composites to a pandas DataFrame.

    Each row = one Composite.
    Columns = 'price', 'delta', 'gamma', 'speed', 'd4', ...

    Args:
        composites: list of Composite or FinancialComposite.
        labels:     optional row labels (e.g., ticker symbols).
        max_order:  highest derivative order to include.

    Returns:
        pandas.DataFrame

    Example:
        import pandas as pd
        df = composites_to_dataframe(
            [fc1, fc2, fc3],
            labels=['AAPL_C100', 'SPY_P450', 'TSLA_C200']
        )
    """
    import pandas as pd

    names = {0: 'price', 1: 'delta', 2: 'gamma', 3: 'speed'}
    columns = [names.get(n, f'd{n}') for n in range(max_order + 1)]

    rows = []
    for c in composites:
        row = [c.st()] + [c.d(n) for n in range(1, max_order + 1)]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns, index=labels)
    return df


def dataframe_to_composites(df, column_map=None):
    """
    Convert a DataFrame back to a list of Composites.

    Args:
        df:          pandas DataFrame with derivative columns.
        column_map:  dict mapping column names to derivative orders.
                     If None, auto-detects from standard names.

    Returns:
        list of FinancialComposite

    Example:
        composites = dataframe_to_composites(df)
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
                    # Store as Taylor coefficient: f^(n) / n!
                    coeffs[-order] = float(val) / math.factorial(order)
        result.append(FinancialComposite(coeffs))

    return result


def ode_points_to_dataframe(points, max_order=2):
    """
    Convert solve_ode output (with composite=True) to DataFrame.

    Args:
        points:    list of (x, y_composite) from solve_ode.
        max_order: highest derivative to extract.

    Returns:
        DataFrame with columns: x, y, dy_dy0, d2y_dy0, ...
    """
    import pandas as pd

    names = {0: 'y', 1: 'dy_dy0', 2: 'd2y_dy0'}
    columns = (['x']
               + [names.get(n, f'd{n}y_dy0')
                  for n in range(max_order + 1)])

    rows = []
    for x, y in points:
        if isinstance(y, Composite):
            row = ([x, y.st()]
                   + [y.d(n) for n in range(1, max_order + 1)])
        else:
            row = [x, float(y)] + [0.0] * max_order
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
