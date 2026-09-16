#!/usr/bin/env python3
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
test_composite_metadata.py — one number carrying data AND metadata
==================================================================

A composite record holds the raw value in its NEGATIVE dimensions and
constructed metadata in its POSITIVE ones:

    dim -(c+1)  data      (e.g. distance to c)
    dim   0     EMPTY
    dim +(c+1)  metadata  (e.g. a learned or derived score for c)

These tests pin the four layout rules that make such a schema safe.  Each rule
exists because breaking it produced a SILENT failure -- no exception, just a
wrong number or a hang -- so each test asserts the trap as well as the fix.

Verbose:  PYTHONPATH=. python tests/test_composite_metadata.py
Quiet:    python -m pytest tests/test_composite_metadata.py -q
Tracing:  python -m pytest tests/test_composite_metadata.py -s
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import composite.backends.config as cfg
cfg.use_dict()
from composite.composite_lib import Composite          # noqa: E402
import composite.composite_lib as cl                   # noqa: E402

EMPTY = Composite({})


def dims(c):
    """{dimension: coefficient}, highest dimension first."""
    return {int(k): v for k, v in sorted(c.to_dict().items(),
                                         key=lambda kv: -int(kv[0]))}


def say(*a):
    print("   ", *a)


# ---------------------------------------------------------------- schema
FIELDS = 2          # data + metadata

def record(data, meta):
    """One city as one composite.  Both halves offset past dimension 0 (R3)."""
    n = len(data)
    return Composite({**{-(c + 1): float(data[c]) for c in range(n)},
                      **{(c + 1): float(meta[c]) for c in range(n)}})

def data_of(rec, c):   return rec.coeff(-(c + 1))
def meta_of(rec, c):   return rec.coeff(c + 1)


# ================================================================== R1
def test_r1_accumulator_must_start_empty():
    """An all-zero accumulator is WHOLLY ZERO, so R1 converts it as an operand
    and injects a phantom one dimension below the lowest schema field."""
    print("\nR1  accumulator must start as Composite({}), not all-zeros")
    schema_zeros = Composite({2: 0.0, 1: 0.0, 0: 0.0, -1: 0.0})
    rec = Composite({2: 1.0, 1: 5.0, 0: 1.0, -1: 3.0})

    say("all-zero acc is wholly zero :", cl._is_wholly_zero(schema_zeros))
    bad = schema_zeros + rec
    say("all-zero acc + record       :", dims(bad))
    assert cl._is_wholly_zero(schema_zeros)
    assert -2 in dims(bad), "expected the phantom R1 injects below the schema"
    say("-> phantom at dim -2, one BELOW the lowest field (-1). Silent.")

    say("EMPTY is wholly zero        :", cl._is_wholly_zero(EMPTY))
    good = EMPTY + rec
    say("EMPTY + record              :", dims(good))
    assert not cl._is_wholly_zero(EMPTY)
    assert dims(good) == {2: 1.0, 1: 5.0, 0: 1.0, -1: 3.0}
    say("-> clean.")


# ================================================================== R2
def test_r2_fields_collide_under_multiplication():
    """Addition keeps fields apart; multiplication lands dims i and j on i+j."""
    print("\nR2  fields collide under multiplication at i+j")
    a = Composite({1: 3.0, 0: 1.0, -1: 10.0})     # meta 3, count 1, data 10
    b = Composite({1: 5.0, 0: 1.0, -1: 20.0})

    s = a + b
    say("a + b :", dims(s))
    assert s.coeff(1) == 8.0 and s.coeff(-1) == 30.0 and s.coeff(0) == 2.0
    say("-> addition is field-wise: safe.")

    p = a * b
    say("a * b :", dims(p))
    say("dim 0 =", p.coeff(0), "-- 1*1 plus meta(+1) x data(-1) cross terms")
    assert p.coeff(0) != 2.0, "meta x data must land on dim 0 and corrupt it"
    assert p.coeff(0) == 1.0 + 3.0 * 20.0 + 5.0 * 10.0
    say("-> a layout safe for + is NOT automatically safe for *.")


# ================================================================== R3
def test_r3_zero_is_its_own_negation():
    """-0 == 0, so an unoffset signed layout collides for c == 0."""
    print("\nR3  -0 == 0, so both halves must be offset past dimension 0")
    n = 3
    naive = Composite({**{-c: float(10 + c) for c in range(n)},
                       **{c: 0.0 for c in range(1, n)}})
    say("naive {-c} / {+c} :", dims(naive))
    assert naive.coeff(0) == 10.0, "city 0's DATA landed on dim 0"
    say("-> dim 0 holds data for city 0, and the metadata half skipped it:")
    say("   one city's two fields collide, silently.")

    ok = record([10.0, 11.0, 12.0], [0.0, 0.0, 0.0])
    say("offset by 1        :", dims(ok))
    assert ok.coeff(0) == 0.0
    for c in range(n):
        assert data_of(ok, c) == 10.0 + c
    say("-> dimension 0 empty, halves disjoint.")


# ================================================================== schema
def test_record_round_trip():
    """Both halves readable, independent, and unchanged by writing the other."""
    print("\nrecord  data and metadata coexist and stay separate")
    dist = [0.0, 5.0, 9.0, 4.0]
    alpha = [0.0, 0.10, 0.90, 0.20]
    rec = record(dist, alpha)
    say("record :", {k: round(v, 2) for k, v in dims(rec).items()})
    for c in range(len(dist)):
        assert data_of(rec, c) == dist[c]
        assert meta_of(rec, c) == alpha[c]
    say("data  reads back :", [data_of(rec, c) for c in range(4)])
    say("meta  reads back :", [round(meta_of(rec, c), 2) for c in range(4)])

    # updating the metadata half must leave the data half untouched
    bumped = rec + Composite({(2 + 1): 0.5})
    assert meta_of(bumped, 2) == 0.90 + 0.5
    assert [data_of(bumped, c) for c in range(4)] == dist
    say("after +0.5 on meta[2]: meta", round(meta_of(bumped, 2), 2),
        "data unchanged", [data_of(bumped, c) for c in range(4)])


def test_cross_half_quantity():
    """A quantity that exists only because both halves live in one number."""
    print("\ncross-half  alignment = sum_c data(c) * meta(c)")
    rec = record([0.0, 5.0, 9.0, 4.0], [0.0, 0.10, 0.90, 0.20])
    align = sum(data_of(rec, c) * meta_of(rec, c) for c in range(4))
    expect = 5.0 * 0.10 + 9.0 * 0.90 + 4.0 * 0.20
    say("alignment :", round(align, 3), " expected", round(expect, 3))
    assert abs(align - expect) < 1e-9
    say("-> asks how far the metadata departs from the data, per record.")


# ================================================================== helper
def top_k_excluding(row, self_idx, k):
    """The k smallest entries of `row`, EXCLUDING self_idx."""
    order = sorted(range(len(row)), key=lambda j: (row[j], j))
    return [j for j in order if j != self_idx][:k]


def test_self_exclusion_with_ties():
    """argsort[1:k+1] assumes position 0 is self.  True for distances, FALSE
    for metadata with exact ties -- the bug that hung a 2-opt loop."""
    print("\nhelper  never assume argsort position 0 is the self index")
    row = [5.0, 0.0, 0.0, 0.0, 9.0]        # ties at 0.0; self is index 3
    naive = sorted(range(len(row)), key=lambda j: (row[j], j))[1:4]
    say("row", row, " self=3")
    say("argsort[1:k+1] gives", naive, "-> self present:", 3 in naive)
    assert 3 in naive, "the naive form is expected to admit self here"

    good = top_k_excluding(row, 3, 3)
    say("top_k_excluding gives", good, "-> self present:", 3 in good)
    assert 3 not in good and len(good) == 3

    uniq = [0.0, 7.0, 2.0]                  # self IS the unique minimum
    assert 0 not in top_k_excluding(uniq, 0, 2)
    say("also correct when self is the unique minimum:", top_k_excluding(uniq, 0, 2))


# ================================================================== end to end
def test_accumulate_a_route_through_records():
    """A whole route summarised by ONE running composite (R1 start, R2 addition
    only).  Every field accumulates in a single pass."""
    print("\nend-to-end  accumulate a route through the records")
    D = [[0, 5, 9, 4], [5, 0, 3, 8], [9, 3, 0, 6], [4, 8, 6, 0]]
    A = [[0, .1, .9, .2], [.1, 0, .3, .7], [.9, .3, 0, .4], [.2, .7, .4, 0]]
    recs = [record(D[i], A[i]) for i in range(4)]
    tour = [0, 1, 2, 3]

    COUNT, DIST, ALPHA = 0, -1, 1
    acc = EMPTY                                        # R1
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        acc = acc + Composite({COUNT: 1.0,
                               DIST: data_of(recs[a], b),
                               ALPHA: meta_of(recs[a], b)})   # R2: addition only
    say("accumulator :", {k: round(v, 2) for k, v in dims(acc).items()})
    say("edges", int(acc.coeff(COUNT)), " length", acc.coeff(DIST),
        " alpha sum", round(acc.coeff(ALPHA), 2))
    assert acc.coeff(COUNT) == 4
    assert acc.coeff(DIST) == 5 + 3 + 6 + 4
    assert abs(acc.coeff(ALPHA) - (.1 + .3 + .4 + .2)) < 1e-9
    say("-> count, length and metadata totals from ONE pass, one object.")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("=" * 70)
    print("composite records: data in the negative half, metadata in the positive")
    print("=" * 70)
    failed = 0
    for t in tests:
        try:
            t()
            print("    PASS")
        except AssertionError as e:
            failed += 1
            print(f"    FAIL: {e}")
    print("\n" + "=" * 70)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
