"""ts_inverse and division by a transseries.

1/t = (exp(n0/h) / c0) (1 - u + u^2 - ...), cut at SECTOR_DEPTH sectors past the
leading one.  Every check prints got against known.  Also: a sector whose
contributions cancel stays an inert zero term (R2) instead of depositing an R1
residue -- found when T * ts_inverse(T) left |-0.667|_-1 at sector 1.
"""
import pytest

import composite.composite_lib as cl
from composite.composite_lib import Composite, R
from composite.transseries import SECTOR_DEPTH, Transseries, sector, ts_exp, ts_inverse

INF = Composite({1: 1.0})


def coeffs(t, upto):
    """{sector: coefficient at grade g} for one-grade sectors, as plain floats."""
    out = {}
    for n, c in t.sectors.items():
        if n <= upto:
            d = {g: v for g, v in c.coeffs_dict().items() if v != 0.0}
            out[n] = d
    return out


def check(label, t, want, tol=1e-14):
    got = coeffs(t, max(want))
    print(f"\n  {label}\n    got  {got}\n    want {want}")
    for n, w in want.items():
        g = got.get(n, {})
        for grade, v in w.items():
            assert abs(g.get(grade, 0.0) - v) <= tol * max(1.0, abs(v)), (n, grade, g)
        assert set(k for k, v in g.items() if abs(v) > tol) <= set(w), (n, g)


def test_tanh_of_an_infinity():
    g = ts_exp(2 * INF)
    check("tanh(1/h) = (e^{2/h} - 1)/(e^{2/h} + 1)", (g - 1) / (g + 1),
          {0: {0: 1.0}, 2: {0: -2.0}, 4: {0: 2.0}, 6: {0: -2.0}, 8: {0: 2.0}})


def test_geometric_series():
    check("1 / (1 + e^{-1/h})", R(1) / (Transseries.lift(R(1)) + sector(1)),
          {k: {0: (-1.0) ** k} for k in range(SECTOR_DEPTH + 1)})


def test_thermal_oscillator_at_low_temperature():
    s = (ts_exp(INF) - ts_exp(-INF)) / 2                     # sinh(1/h)
    check("Z = 1 / (2 sinh(1/h))", R(1) / (2 * s), {1: {0: 1.0}, 3: {0: 1.0}, 5: {0: 1.0}, 7: {0: 1.0}, 9: {0: 1.0}})
    check("C = x^2 / sinh^2 x, x = 1/h", (INF * INF) / (s * s),
          {2: {2: 4.0}, 4: {2: 8.0}, 6: {2: 12.0}, 8: {2: 16.0}, 10: {2: 20.0}})


def test_round_trip():
    t = Transseries({-1: R(3), 0: R(2), 2: Composite({0: 1.0, -1: 0.5})})
    check("t * ts_inverse(t) = 1, t = 3 e^{1/h} + 2 + (1 + h/2) e^{-2/h}", t * ts_inverse(t),
          {0: {0: 1.0}, **{k: {} for k in range(1, SECTOR_DEPTH)}}, tol=1e-15)


def test_nothing_past_the_cut():
    q = R(1) / (Transseries.lift(R(1)) + sector(1))
    print(f"\n  sectors returned: {sorted(q.sectors)}   cut at {SECTOR_DEPTH}")
    assert max(q.sectors) == SECTOR_DEPTH


def test_a_cancelled_sector_stays_an_inert_zero():
    a = Transseries({0: R(1), 1: R(2)})
    b = Transseries({1: R(2)})
    d = (a - b).sectors[1]
    print(f"\n  ({{0: 1, 1: 2}} - {{1: 2}}).sectors[1] = {d}   want an inert zero, no infinitesimal")
    assert all(v == 0.0 for v in d.coeffs_dict().values())


def test_division_by_an_ordinary_value_is_unchanged():
    check("(e^{-1/h} + 4) / 2", (sector(1) + 4) / 2, {0: {0: 2.0}, 1: {0: 0.5}})


def test_zero_has_no_inverse():
    with pytest.raises(ZeroDivisionError):
        ts_inverse(Transseries())
