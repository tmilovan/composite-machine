#!/usr/bin/env python3
# Composite Machine — stability radius of a routing optimum
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""How much can one road get slower before the route changes?

A solver returns a tour.  The question a dispatcher asks next is not "what
does it cost" but "how much would this road have to change before the answer
changes" -- and the usual way to find out is to re-solve, once per edge, once
per perturbation.

Make that edge cost  d_e + h  instead.  Every tour's cost comes back as

    C_T  +  (dC_T/dd_e) h  +  (1/2 d2C_T/dd_e2) h^2

so ONE pass gives the value, the exact sensitivity and the curvature together.
The sensitivity is not a formality: with time windows, making an edge slower
pushes every downstream stop later, so dC/dd_e is a real number that depends
on WHERE in the tour the edge sits.

TWO PLACES st() IS THE WRONG READ, both of which were wrong when this was
first written:

  * Choosing the optimum with float(cost) picks arbitrarily among ties -- and
    a tie is exactly the case where the radius is ZERO.  Section 3 shows the
    composite and the float picking DIFFERENT tours at equal cost.
  * `late > 0` does not ask what it looks like.  R(0) is ZERO = |1|_-1, so
    `x > 0` asks `x > h`.  The classical test is against NOTHING.  With a
    depot due-time of 0.0 the resulting R(0) deposited an infinitesimal that
    cancelled the depot's whole contribution to the sensitivity: dimension 0
    stayed right and every derivative was wrong.

Run:  python demos/composite_stability_radius.py
"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO

cl.MAX_ACTIVE_DIMS = 10 ** 9
NOTHING = Composite({})          # the classical zero: an absence, not |1|_-1


# =============================================================================
# 1.  A routing instance with time windows
# =============================================================================
N = 8
W = 0.05                          # weight on squared lateness
_rng = random.Random(5)
POS = [(round(_rng.uniform(0, 60), 2), round(_rng.uniform(0, 60), 2))
       for _ in range(N)]
DUE = [185.0] + [round(_rng.uniform(60, 190), 1) for _ in range(N - 1)]


def dist(i, j):
    return math.dist(POS[i], POS[j])


def tours():
    """Every distinct tour on N stops, depot fixed, direction fixed."""
    for p in itertools.permutations(range(1, N)):
        if p[0] < p[-1]:
            yield (0,) + p


def evaluate(tour, edge=None, edge_cost=None):
    """distance + W * sum(lateness^2), as a Composite.

    `edge_cost` replaces the cost of `edge` -- pass R(d) + ZERO to perturb it.
    Accumulators start at NOTHING, never 0: a Python zero meeting a Composite
    is an EXPRESSED zero and converts (R1), which would add |1|_-1 to the
    total and to every arrival time.
    """
    total = NOTHING
    clock = NOTHING
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        c = (edge_cost if edge is not None and {u, v} == {edge[0], edge[1]}
             else R(dist(u, v)))
        total = total + c
        clock = clock + c
        late = clock - R(DUE[v])
        if late > NOTHING:                    # NOT `> 0`, which means `> h`
            total = total + R(W) * late * late
    return total


# =============================================================================
# 2.  Sensitivity, curvature and radius from one pass
# =============================================================================
def analyse(edge):
    """(sensitivity, curvature, degenerate, 1st-order radius, 2nd-order radius)."""
    ec = R(dist(*edge)) + ZERO
    costs = [(t, evaluate(t, edge, ec)) for t in tours()]

    # COMPOSITE comparison, not float: grade 0 first, then grade -1 as the
    # tie-break, which is the tour that survives d_e -> d_e+.
    opt_t, opt_c = min(costs, key=lambda kv: kv[1])
    k = opt_c.coeffs_dict().get(-1, 0.0)
    q = opt_c.coeffs_dict().get(-2, 0.0)

    # A grade-0 tie the perturbation SEPARATES means the radius is zero.
    degenerate = any(t is not opt_t
                     and abs(float(c) - float(opt_c)) < 1e-12
                     and c.coeffs_dict().get(-1, 0.0) != k
                     for t, c in costs)

    linear, quadratic = [], []
    for t, c in costs:
        if t is opt_t:
            continue                          # the difference is wholly zero
        gap = c - opt_c                       # the gap polynomial in delta
        c0 = gap.coeffs_dict().get(0, 0.0)
        c1 = gap.coeffs_dict().get(-1, 0.0)
        c2 = gap.coeffs_dict().get(-2, 0.0)
        if c1 != 0.0 and -c0 / c1 > 1e-12:
            linear.append(-c0 / c1)
        if abs(c2) > 1e-14:
            disc = c1 * c1 - 4 * c2 * c0
            if disc >= 0:
                for r in ((-c1 + math.sqrt(disc)) / (2 * c2),
                          (-c1 - math.sqrt(disc)) / (2 * c2)):
                    if r > 1e-12:
                        quadratic.append(r)
        elif c1 != 0.0 and -c0 / c1 > 1e-12:
            quadratic.append(-c0 / c1)

    return (opt_t, k, 2 * q, degenerate,
            min(linear, default=float('inf')),
            min(quadratic, default=float('inf')))


def true_radius(edge, opt_t):
    """Re-solve repeatedly to find where the optimum actually changes."""
    def flips(delta):
        return min(tours(),
                   key=lambda t: evaluate(t, edge, R(dist(*edge) + delta))) != opt_t
    lo, hi = 0.0, 1.0
    while not flips(hi) and hi < 500:
        hi *= 2
    if not flips(hi):
        return float('inf')
    for _ in range(45):
        mid = (lo + hi) / 2
        if flips(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _fmt(x):
    return "inf" if math.isinf(x) else "%.4f" % x


def main():
    best = min(tours(), key=evaluate)          # composite comparison
    print("=" * 74)
    print("1. ONE PASS GIVES VALUE, SENSITIVITY AND CURVATURE")
    print("=" * 74)
    print("   %d stops, objective = distance + %.2f * sum(lateness^2)" % (N, W))
    print("   optimum %s   cost %.4f" % (list(best), evaluate(best).st()))
    clock, late = 0.0, []
    for i in range(N):
        u, v = best[i], best[(i + 1) % N]
        clock += dist(u, v)
        if clock > DUE[v]:
            late.append((i, v, clock - DUE[v]))
    print("   late stops:", ", ".join("leg %d -> stop %d by %.1f" % x for x in late)
          or "none")
    print()
    print("   edge    dC/dd_e   d2C/dd_e2   1st-order   2nd-order   true (re-solved)")
    for pos in (0, 4, 5, 6):
        e = (best[pos], best[(pos + 1) % N])
        opt_t, k, qq, degen, r1, r2 = analyse(e)
        print("   (%d,%d)   %7.4f %11.4f %11s %11s %13s"
              % (e[0], e[1], k, qq, _fmt(r1), _fmt(r2),
                 "0 (degenerate)" if degen else _fmt(true_radius(e, opt_t))))
    print()
    print("   The sensitivity is position-dependent: an edge BEFORE the late stop")
    print("   carries 1 + 2*W*lateness; one after it carries exactly 1.")
    print("   Where the set of late stops changes with delta the radius is a")
    print("   bracket, not an equality -- the quadratic model does not see that kink.")

    # =========================================================================
    print()
    print("=" * 74)
    print("2. WHY THE COMPARISON MUST BE ON THE WHOLE COMPOSITE")
    print("=" * 74)
    D = {(0, 1): 2, (0, 2): 3, (0, 3): 4, (0, 4): 3, (1, 2): 4,
         (1, 3): 3, (1, 4): 5, (2, 3): 2, (2, 4): 4, (3, 4): 3}

    def d5(i, j):
        return float(D[(min(i, j), max(i, j))])

    def tours5():
        for p in itertools.permutations(range(1, 5)):
            if p[0] < p[-1]:
                yield (0,) + p

    def eval5(t, e=None, ec=None):
        acc = NOTHING
        for i in range(len(t)):
            u, v = t[i], t[(i + 1) % len(t)]
            acc = acc + (ec if e and {u, v} == {e[0], e[1]} else R(d5(u, v)))
        return acc

    tied = [t for t in tours5() if eval5(t).st() == 14.0]
    print("   a DEGENERATE instance: %d distinct tours at cost 14" % len(tied))
    for t in tied:
        print("      %s" % list(t))
    print()
    print("   edge   optimum by float   optimum by composite   degenerate?  radius")
    for e in [(1, 2), (1, 3), (2, 3), (0, 1)]:
        ec = R(d5(*e)) + ZERO
        cs = [(t, eval5(t, e, ec)) for t in tours5()]
        by_float = min(cs, key=lambda kv: float(kv[1]))[0]
        opt_t, opt_c = min(cs, key=lambda kv: kv[1])
        k = opt_c.coeffs_dict().get(-1, 0.0)
        degen = any(t is not opt_t and abs(float(c) - float(opt_c)) < 1e-12
                    and c.coeffs_dict().get(-1, 0.0) != k for t, c in cs)
        gaps = []
        for t, c in cs:
            if t is opt_t:
                continue
            gap = c - opt_c
            c0 = gap.coeffs_dict().get(0, 0.0)
            c1 = gap.coeffs_dict().get(-1, 0.0)
            if c1 != 0.0 and -c0 / c1 > 1e-12:
                gaps.append(-c0 / c1)
        print("   (%d,%d)  %-18s %-22s %-12s %s"
              % (e[0], e[1], list(by_float), list(opt_t),
                 "YES" if degen else "no",
                 "0 (any change moves it)" if degen
                 else _fmt(min(gaps, default=float('inf')))))
    print()
    print("   Row 1: float and composite pick DIFFERENT tours.  Both cost 14;")
    print("   only |14|_0 vs |14|_0 + |1|_-1 says which one survives the edge")
    print("   getting more expensive.  A float comparison reports the same tour")
    print("   for every row and cannot tell radius 2.0 from radius 0.")


if __name__ == "__main__":
    main()
