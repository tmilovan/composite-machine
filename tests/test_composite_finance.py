# Composite Machine — Financial Layer Test Suite
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
test_composite_finance.py — Financial Layer Test Suite
======================================================
Validates all functionality in composite_finance.py:
  1. FinancialComposite named accessors
  2. Serialization round-trips (dict, bytes, array, JSON)
  3. Scenario P&L and scenario pricing
  4. Bond duration and convexity
  5. ScenarioEngine portfolio aggregation
  6. Scenario ladder
  7. Taylor VaR
  8. Greeks dict serialization
  9. Pandas integration (composites_to_dataframe, dataframe_to_composites)
 10. ODE points to DataFrame
 11. Edge cases (zero price, empty composite, single-term)

Usage:
    python test_composite_finance.py

Author: Toni Milovan
License: AGPL-3.0
"""

import math
import sys

from composite.composite_lib import Composite, R, ZERO, INF
from composite.composite_finance import (
    FinancialComposite,
    ScenarioEngine,
    composites_to_dataframe,
    dataframe_to_composites,
    ode_points_to_dataframe,
)


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

PASSED = 0
FAILED = 0
TOTAL = 0


def check(name, actual, expected, tol=1e-6):
    """Check a single value against expected."""
    global PASSED, FAILED, TOTAL
    TOTAL += 1
    if isinstance(expected, float) and math.isinf(expected):
        ok = math.isinf(actual) and (actual > 0) == (expected > 0)
    elif isinstance(expected, float):
        ok = abs(actual - expected) < tol
    elif isinstance(expected, bool):
        ok = actual == expected
    elif isinstance(expected, str):
        ok = actual == expected
    elif isinstance(expected, dict):
        ok = all(
            abs(actual.get(k, 0) - v) < tol
            for k, v in expected.items()
        )
    else:
        ok = actual == expected
    if ok:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}: got {actual}, expected {expected}")
    return ok


def check_type(name, obj, expected_type):
    """Check that obj is an instance of expected_type."""
    global PASSED, FAILED, TOTAL
    TOTAL += 1
    ok = isinstance(obj, expected_type)
    if ok:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}: got {type(obj).__name__}, expected {expected_type.__name__}")
    return ok


# =============================================================================
# TEST DATA: BUILD KNOWN COMPOSITES
# =============================================================================
# We build composites by hand so tests don't depend on the calculus engine.
# This tests the financial layer in isolation.
#
# Example: an option-like composite where:
#   price = 10.0, delta = 0.6, gamma = 0.04, speed = -0.003
#
# Stored as Taylor coefficients: coeff(-n) = f^(n)/n!
#   dim  0 = 10.0          (price)
#   dim -1 = 0.6 / 1! = 0.6  (delta / 1!)
#   dim -2 = 0.04 / 2! = 0.02  (gamma / 2!)
#   dim -3 = -0.003 / 3! = -0.0005  (speed / 3!)
# =============================================================================

def make_option_composite(price, delta, gamma, speed=0.0):
    """Build a Composite with known Greek values."""
    coeffs = {0: price}
    if delta != 0:
        coeffs[-1] = delta / math.factorial(1)
    if gamma != 0:
        coeffs[-2] = gamma / math.factorial(2)
    if speed != 0:
        coeffs[-3] = speed / math.factorial(3)
    return Composite(coeffs)


def make_bond_composite(price, dP_dy, d2P_dy2):
    """Build a Composite for a bond (price as function of yield)."""
    coeffs = {0: price}
    if dP_dy != 0:
        coeffs[-1] = dP_dy / math.factorial(1)
    if d2P_dy2 != 0:
        coeffs[-2] = d2P_dy2 / math.factorial(2)
    return Composite(coeffs)


# =============================================================================
# TESTS
# =============================================================================

def test_FC01_named_accessors():
    """FC01: FinancialComposite named accessors return correct values."""
    print("\n--- FC01: Named Accessors ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04, speed=-0.003)
    fc = FinancialComposite.from_composite(c)

    check("price", fc.price, 10.0)
    check("pv (alias)", fc.pv, 10.0)
    check("delta", fc.delta, 0.6)
    check("gamma", fc.gamma, 0.04)
    check("speed", fc.speed, -0.003)


def test_FC02_from_composite_type():
    """FC02: from_composite returns a FinancialComposite instance."""
    print("\n--- FC02: Type Check ---")
    c = make_option_composite(5.0, 0.5, 0.02)
    fc = FinancialComposite.from_composite(c)

    check_type("is FinancialComposite", fc, FinancialComposite)
    check_type("is also Composite", fc, Composite)


def test_FC03_bond_duration_convexity():
    """FC03: Duration and convexity match bond conventions."""
    print("\n--- FC03: Duration & Convexity ---")
    # Bond: P=100, dP/dy = -500, d2P/dy2 = 30000
    # Duration = -dP/dy / P = 500/100 = 5.0
    # Convexity = d2P/dy2 / P = 30000/100 = 300.0
    c = make_bond_composite(price=100.0, dP_dy=-500.0, d2P_dy2=30000.0)
    fc = FinancialComposite.from_composite(c)

    check("duration", fc.duration, 5.0)
    check("convexity", fc.convexity, 300.0)


def test_FC04_duration_zero_price():
    """FC04: Duration returns inf when price is zero."""
    print("\n--- FC04: Duration Zero Price ---")
    c = make_bond_composite(price=0.0, dP_dy=-10.0, d2P_dy2=100.0)
    fc = FinancialComposite.from_composite(c)

    check("duration at zero price", fc.duration, float('inf'))
    check("convexity at zero price", fc.convexity, float('inf'))


def test_FC05_scenario_pnl():
    """FC05: Scenario P&L uses full Taylor expansion."""
    print("\n--- FC05: Scenario P&L ---")
    # price=10, delta=0.6, gamma=0.04
    # For shock=1.0:
    #   P&L = delta/1! * 1^1 + gamma/2! * 1^2
    #       = 0.6 * 1 + 0.02 * 1 = 0.62
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    fc = FinancialComposite.from_composite(c)

    # shock = 1.0 for easy manual calculation
    pnl = fc.scenario_pnl(1.0)
    # coeff(-1) = 0.6, coeff(-2) = 0.02
    # P&L = 0.6 * 1^1 + 0.02 * 1^2 = 0.62
    check("pnl at shock=1.0", pnl, 0.62)

    # shock = 0.0 should give 0
    check("pnl at shock=0.0", fc.scenario_pnl(0.0), 0.0)


def test_FC06_scenario_price():
    """FC06: Scenario price = price + P&L."""
    print("\n--- FC06: Scenario Price ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    fc = FinancialComposite.from_composite(c)

    new_price = fc.scenario_price(1.0)
    check("scenario price at shock=1.0", new_price, 10.0 + 0.62)

    # No shock = same price
    check("scenario price at shock=0.0", fc.scenario_price(0.0), 10.0)


def test_FC07_greeks_dict():
    """FC07: to_greeks_dict returns correctly named and scaled values."""
    print("\n--- FC07: Greeks Dict ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04, speed=-0.003)
    fc = FinancialComposite.from_composite(c)

    gd = fc.to_greeks_dict(max_order=3)
    check("greeks_dict price", gd['price'], 10.0)
    check("greeks_dict delta", gd['delta'], 0.6)
    check("greeks_dict gamma", gd['gamma'], 0.04)
    check("greeks_dict speed", gd['speed'], -0.003)


def test_FC08_repr():
    """FC08: repr includes financial summary."""
    print("\n--- FC08: Repr ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    fc = FinancialComposite.from_composite(c)
    r = repr(fc)

    # Should contain dimensional notation AND financial summary
    check("repr contains 'price='", 'price=' in r, True)
    check("repr contains 'delta='", 'delta=' in r, True)
    check("repr contains 'gamma='", 'gamma=' in r, True)


def test_FC09_serialization_dict_roundtrip():
    """FC09: to_dict / from_dict round-trip preserves all coefficients."""
    print("\n--- FC09: Dict Serialization Round-Trip ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04, speed=-0.003)

    d = c.to_dict()
    restored = Composite.from_dict(d)

    check("dict rt: price", restored.st(), 10.0)
    check("dict rt: delta", restored.d(1), 0.6)
    check("dict rt: gamma", restored.d(2), 0.04)
    check("dict rt: speed", restored.d(3), -0.003)


def test_FC10_serialization_bytes_roundtrip():
    """FC10: to_bytes / from_bytes round-trip preserves all coefficients."""
    print("\n--- FC10: Bytes Serialization Round-Trip ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)

    data = c.to_bytes()
    restored = Composite.from_bytes(data)

    check("bytes rt: price", restored.st(), 10.0)
    check("bytes rt: delta", restored.d(1), 0.6)
    check("bytes rt: gamma", restored.d(2), 0.04)
    # Check byte count: 3 terms * 12 bytes = 36
    check("bytes length", len(data), 36)


def test_FC11_serialization_array_roundtrip():
    """FC11: to_array / from_array round-trip with fixed dimensions."""
    print("\n--- FC11: Array Serialization Round-Trip ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    dims = (0, -1, -2, -3, -4)

    arr = c.to_array(dims)
    check("array length", len(arr), 5)
    check("array[0] = price coeff", arr[0], 10.0)
    check("array[3] = 0 (no speed)", arr[3], 0.0)
    check("array[4] = 0 (no d4)", arr[4], 0.0)

    restored = Composite.from_array(arr, dims)
    check("array rt: price", restored.st(), 10.0)
    check("array rt: delta", restored.d(1), 0.6)


def test_FC12_serialization_json_roundtrip():
    """FC12: to_json / from_json round-trip."""
    print("\n--- FC12: JSON Serialization Round-Trip ---")
    c = make_option_composite(price=7.5, delta=0.55, gamma=0.03)

    json_str = c.to_json()
    check_type("json is string", json_str, str)

    restored = Composite.from_json(json_str)
    check("json rt: price", restored.st(), 7.5)
    check("json rt: delta", restored.d(1), 0.55)
    check("json rt: gamma", restored.d(2), 0.03)


def test_FC13_scenario_engine_aggregation():
    """FC13: ScenarioEngine aggregates price, delta, gamma with weights."""
    print("\n--- FC13: ScenarioEngine Aggregation ---")
    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02)
    c3 = make_option_composite(price=8.0, delta=0.7, gamma=0.05)

    weights = [100, 50, 200]
    engine = ScenarioEngine([c1, c2, c3], weights)

    # total_price = 10*100 + 5*50 + 8*200 = 1000 + 250 + 1600 = 2850
    check("total price", engine.total_price(), 2850.0)

    # total_delta = 0.6*100 + (-0.3)*50 + 0.7*200 = 60 - 15 + 140 = 185
    check("total delta", engine.total_delta(), 185.0)

    # total_gamma = 0.04*100 + 0.02*50 + 0.05*200 = 4 + 1 + 10 = 15
    check("total gamma", engine.total_gamma(), 15.0)


def test_FC14_scenario_engine_default_weights():
    """FC14: ScenarioEngine with no weights uses 1.0 for all."""
    print("\n--- FC14: ScenarioEngine Default Weights ---")
    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02)

    engine = ScenarioEngine([c1, c2])

    check("default weight price", engine.total_price(), 15.0)
    check("default weight delta", engine.total_delta(), 0.3)


def test_FC15_scenario_engine_pnl():
    """FC15: ScenarioEngine portfolio P&L at a given shock."""
    print("\n--- FC15: ScenarioEngine P&L ---")
    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02)

    engine = ScenarioEngine([c1, c2], weights=[100, 100])

    # At shock=0, P&L = 0
    check("pnl at shock=0", engine.scenario_pnl(0.0), 0.0)

    # At shock=1.0:
    #   c1 pnl = 0.6*1 + 0.02*1 = 0.62, weighted: 62.0
    #   c2 pnl = -0.3*1 + 0.01*1 = -0.29, weighted: -29.0
    #   total = 33.0
    check("pnl at shock=1.0", engine.scenario_pnl(1.0), 33.0)


def test_FC16_scenario_ladder():
    """FC16: Scenario ladder returns correct (shock, pnl) pairs."""
    print("\n--- FC16: Scenario Ladder ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    engine = ScenarioEngine([c])

    shocks = [-1.0, 0.0, 1.0]
    ladder = engine.scenario_ladder(shocks)

    check("ladder length", len(ladder), 3)
    check("ladder[0] shock", ladder[0][0], -1.0)
    check("ladder[1] pnl (zero shock)", ladder[1][1], 0.0)
    # At shock=1.0: pnl = 0.6 + 0.02 = 0.62
    check("ladder[2] pnl", ladder[2][1], 0.62)


def test_FC17_taylor_var():
    """FC17: Taylor VaR returns a positive number."""
    print("\n--- FC17: Taylor VaR ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    engine = ScenarioEngine([c], weights=[1000])

    var_99 = engine.taylor_var(vol=0.20, confidence=0.99)
    check("VaR is positive", var_99 > 0, True)

    # Higher confidence should give higher VaR
    var_95 = engine.taylor_var(vol=0.20, confidence=0.95)
    check("VaR99 > VaR95", var_99 > var_95, True)

    # Higher vol should give higher VaR
    var_high_vol = engine.taylor_var(vol=0.40, confidence=0.99)
    check("higher vol -> higher VaR", var_high_vol > var_99, True)


def test_FC18_taylor_var_horizon():
    """FC18: Taylor VaR scales with holding horizon."""
    print("\n--- FC18: Taylor VaR Horizon ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    engine = ScenarioEngine([c], weights=[1000])

    var_1d = engine.taylor_var(vol=0.20, confidence=0.99, horizon_days=1)
    var_10d = engine.taylor_var(vol=0.20, confidence=0.99, horizon_days=10)

    check("10d VaR > 1d VaR", var_10d > var_1d, True)


def test_FC19_scenario_engine_accepts_raw_composites():
    """FC19: ScenarioEngine wraps raw Composites into FinancialComposite."""
    print("\n--- FC19: Engine Accepts Raw Composites ---")
    # Pass plain Composite, not FinancialComposite
    c = Composite({0: 10.0, -1: 0.6, -2: 0.02})
    engine = ScenarioEngine([c])

    check("engine total price from raw", engine.total_price(), 10.0)
    check("engine total delta from raw", engine.total_delta(), 0.6)


def test_FC20_negative_shock():
    """FC20: Negative shock gives correct P&L direction."""
    print("\n--- FC20: Negative Shock ---")
    # Long delta position: negative shock = loss
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    fc = FinancialComposite.from_composite(c)

    pnl_down = fc.scenario_pnl(-1.0)
    pnl_up = fc.scenario_pnl(1.0)

    # With positive delta, down shock should give negative P&L
    check("negative shock -> negative pnl", pnl_down < 0, True)
    check("positive shock -> positive pnl", pnl_up > 0, True)


def test_FC21_empty_composite():
    """FC21: Empty composite gives zeros for all Greeks."""
    print("\n--- FC21: Empty Composite ---")
    c = Composite({})
    fc = FinancialComposite.from_composite(c)

    check("empty price", fc.price, 0.0)
    check("empty delta", fc.delta, 0.0)
    check("empty gamma", fc.gamma, 0.0)
    check("empty pnl", fc.scenario_pnl(1.0), 0.0)


def test_FC22_single_dim_composite():
    """FC22: Composite with only price (dim 0) has zero Greeks."""
    print("\n--- FC22: Price-Only Composite ---")
    c = Composite({0: 42.0})
    fc = FinancialComposite.from_composite(c)

    check("price only: price", fc.price, 42.0)
    check("price only: delta", fc.delta, 0.0)
    check("price only: gamma", fc.gamma, 0.0)
    check("price only: pnl", fc.scenario_pnl(5.0), 0.0)


def test_FC23_pandas_composites_to_dataframe():
    """FC23: composites_to_dataframe produces correct DataFrame."""
    print("\n--- FC23: Pandas composites_to_dataframe ---")
    try:
        import pandas as pd
    except ImportError:
        print("  (skipped — pandas not installed)")
        return

    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02)

    df = composites_to_dataframe(
        [c1, c2],
        labels=['OPT_A', 'OPT_B'],
        max_order=3
    )

    check("df shape rows", df.shape[0], 2)
    check("df shape cols", df.shape[1], 4)  # price, delta, gamma, speed
    check("df OPT_A price", df.loc['OPT_A', 'price'], 10.0)
    check("df OPT_A delta", df.loc['OPT_A', 'delta'], 0.6)
    check("df OPT_B delta", df.loc['OPT_B', 'delta'], -0.3)
    check("df OPT_B gamma", df.loc['OPT_B', 'gamma'], 0.02)


def test_FC24_pandas_dataframe_to_composites():
    """FC24: dataframe_to_composites round-trips through DataFrame."""
    print("\n--- FC24: Pandas dataframe_to_composites ---")
    try:
        import pandas as pd
    except ImportError:
        print("  (skipped — pandas not installed)")
        return

    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02)

    df = composites_to_dataframe([c1, c2], max_order=2)
    restored = dataframe_to_composites(df)

    check("rt[0] price", restored[0].st(), 10.0)
    check("rt[0] delta", restored[0].d(1), 0.6)
    check("rt[0] gamma", restored[0].d(2), 0.04)
    check("rt[1] price", restored[1].st(), 5.0)
    check("rt[1] delta", restored[1].d(1), -0.3)
    check_type("rt[0] is FinancialComposite", restored[0], FinancialComposite)


def test_FC25_pandas_ode_points():
    """FC25: ode_points_to_dataframe handles composite and float y."""
    print("\n--- FC25: Pandas ODE Points ---")
    try:
        import pandas as pd
    except ImportError:
        print("  (skipped — pandas not installed)")
        return

    # Simulate solve_ode output with composite=True
    y1 = Composite({0: 1.0, -1: 1.0})   # y=1, dy/dy0=1
    y2 = Composite({0: 0.5, -1: 0.5})   # y=0.5, dy/dy0=0.5
    points = [(0.0, y1), (1.0, y2)]

    df = ode_points_to_dataframe(points, max_order=1)

    check("ode df shape", df.shape, (2, 3))  # x, y, dy_dy0
    check("ode df x[0]", df.iloc[0]['x'], 0.0)
    check("ode df y[0]", df.iloc[0]['y'], 1.0)
    check("ode df dy_dy0[0]", df.iloc[0]['dy_dy0'], 1.0)
    check("ode df y[1]", df.iloc[1]['y'], 0.5)

    # Also test with plain float y (composite=False output)
    points_scalar = [(0.0, 1.0), (1.0, 0.5)]
    df2 = ode_points_to_dataframe(points_scalar, max_order=1)
    check("scalar ode df y[0]", df2.iloc[0]['y'], 1.0)
    check("scalar ode df dy_dy0[0]", df2.iloc[0]['dy_dy0'], 0.0)


def test_FC26_symmetry_positive_negative_shock():
    """FC26: For pure-delta position, P&L is antisymmetric."""
    print("\n--- FC26: Shock Symmetry ---")
    # Delta only, no gamma — P&L should be exactly antisymmetric
    c = Composite({0: 10.0, -1: 0.5})
    fc = FinancialComposite.from_composite(c)

    pnl_up = fc.scenario_pnl(0.5)
    pnl_down = fc.scenario_pnl(-0.5)

    check("pure delta: pnl_up = -pnl_down", pnl_up, -pnl_down)


def test_FC27_gamma_convexity_effect():
    """FC27: With gamma, both up and down shocks gain value (long gamma)."""
    print("\n--- FC27: Gamma Convexity Effect ---")
    # No delta, only gamma — should gain for any shock direction
    c = Composite({0: 10.0, -2: 0.05})  # gamma/2! = 0.05, gamma = 0.1
    fc = FinancialComposite.from_composite(c)

    pnl_up = fc.scenario_pnl(1.0)
    pnl_down = fc.scenario_pnl(-1.0)

    # Both should be positive (long gamma benefits from any move)
    check("long gamma: up shock gains", pnl_up > 0, True)
    check("long gamma: down shock gains", pnl_down > 0, True)
    # And symmetric (even function)
    check("long gamma: symmetric", pnl_up, pnl_down)


def test_FC28_total_nth_derivative():
    """FC28: ScenarioEngine total_nth works for arbitrary order."""
    print("\n--- FC28: total_nth ---")
    c1 = make_option_composite(price=10.0, delta=0.6, gamma=0.04, speed=-0.003)
    c2 = make_option_composite(price=5.0, delta=-0.3, gamma=0.02, speed=0.001)

    engine = ScenarioEngine([c1, c2], weights=[10, 20])

    # total speed = -0.003*10 + 0.001*20 = -0.03 + 0.02 = -0.01
    check("total 3rd derivative", engine.total_nth(3), -0.01)


def test_FC29_large_portfolio():
    """FC29: ScenarioEngine handles 100-position portfolio."""
    print("\n--- FC29: Large Portfolio ---")
    composites = [
        make_option_composite(price=10.0 + i * 0.1, delta=0.5, gamma=0.03)
        for i in range(100)
    ]
    weights = [1.0] * 100
    engine = ScenarioEngine(composites, weights)

    # total price = sum(10.0 + i*0.1 for i in range(100))
    expected_price = sum(10.0 + i * 0.1 for i in range(100))
    check("100-position total price", engine.total_price(), expected_price)
    check("100-position total delta", engine.total_delta(), 50.0)  # 0.5 * 100


def test_FC30_greeks_dict_max_order():
    """FC30: to_greeks_dict respects max_order parameter."""
    print("\n--- FC30: Greeks Dict max_order ---")
    c = make_option_composite(price=10.0, delta=0.6, gamma=0.04, speed=-0.003)
    fc = FinancialComposite.from_composite(c)

    gd2 = fc.to_greeks_dict(max_order=2)
    check("max_order=2: has price", 'price' in gd2, True)
    check("max_order=2: has delta", 'delta' in gd2, True)
    check("max_order=2: has gamma", 'gamma' in gd2, True)
    check("max_order=2: no speed", 'speed' not in gd2, True)

    gd1 = fc.to_greeks_dict(max_order=1)
    check("max_order=1: no gamma", 'gamma' not in gd1, True)


# =============================================================================
# RUNNER
# =============================================================================

def run_all():
    print("=" * 60)
    print("COMPOSITE FINANCE TEST SUITE")
    print("=" * 60)

    test_FC01_named_accessors()
    test_FC02_from_composite_type()
    test_FC03_bond_duration_convexity()
    test_FC04_duration_zero_price()
    test_FC05_scenario_pnl()
    test_FC06_scenario_price()
    test_FC07_greeks_dict()
    test_FC08_repr()
    test_FC09_serialization_dict_roundtrip()
    test_FC10_serialization_bytes_roundtrip()
    test_FC11_serialization_array_roundtrip()
    test_FC12_serialization_json_roundtrip()
    test_FC13_scenario_engine_aggregation()
    test_FC14_scenario_engine_default_weights()
    test_FC15_scenario_engine_pnl()
    test_FC16_scenario_ladder()
    test_FC17_taylor_var()
    test_FC18_taylor_var_horizon()
    test_FC19_scenario_engine_accepts_raw_composites()
    test_FC20_negative_shock()
    test_FC21_empty_composite()
    test_FC22_single_dim_composite()
    test_FC23_pandas_composites_to_dataframe()
    test_FC24_pandas_dataframe_to_composites()
    test_FC25_pandas_ode_points()
    test_FC26_symmetry_positive_negative_shock()
    test_FC27_gamma_convexity_effect()
    test_FC28_total_nth_derivative()
    test_FC29_large_portfolio()
    test_FC30_greeks_dict_max_order()

    print("\n" + "=" * 60)
    print(f"PASSED: {PASSED}/{TOTAL}")
    if FAILED:
        print(f"FAILED: {FAILED}/{TOTAL}")
    print("=" * 60)

    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
