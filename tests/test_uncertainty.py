#!/usr/bin/env python3
# Composite Machine — GUM uncertainty budgets and the linearity verdict
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Known answers for composite/uncertainty.py.

Every case below has a closed form, so the tests compare a computed number
against a derived one rather than against a previous run of this code:

    U1  linear model            bias and skewness are identically zero
    U2  y = x^2, normal         bias u^2, var 4mu^2u^2 + 2u^4, mu3 24mu^2u^4
    U3  y = x^2, rectangular    var (4/3)mu^2a^2 + 4a^4/45, exactly
    U4  y = x1 x2               var u1^2mu2^2 + u2^2mu1^2 + u1^2u2^2,
                                mu3 = 6 mu1 mu2 u1^2 u2^2  -- the cross terms,
                                which come from DIRECTIONAL derivatives and not
                                from any multivariate number
    U5  y = exp(x)              lognormal: mean, variance and skewness exact
    U6  verdict                 exp(x) at u = 0.5 must refuse the linear budget
    U7  Monte Carlo             the analytic moments checked against GUM-S1
                                sampling rather than trusted
    U8  escape to float         a model that drops out of composite arithmetic
                                is refused, not silently differentiated wrong
    U9  the expressed zero      a vanishing scalar prefactor multiplying a
                                composite corrupts the derivative and leaves
                                the value correct; the cross-check must catch it
    U10 orifice meter           ISO 5167-2, the real case, against sampling
    U11 flat response           GUM-S1 clause 9.4's model, where every
                                sensitivity coefficient vanishes and the linear
                                budget therefore reports ZERO uncertainty:
                                bias -2u^2, u_c 2u^2, skewness exactly -2
    U12 zero estimates          an estimate of exactly zero is an ABSENT term:
                                "x = 0 +- u" applies no term and carries the
                                uncertainty separately, so Composite({}) is
                                what the model is given.  A written zero would
                                put R1's residue on the same axis the seeded
                                variable's jet is read from -- a collision
                                between two different inputs' zeros

U2's third moment carries an exact 8u^6 that the second-order construction
does not reach, and U5 is a truncated Taylor series of a lognormal.  Those
tolerances are set from the size of the omitted term, not widened until green,
and both the computed and the exact value are printed either way.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import composite.composite_lib as cl
from composite.composite_lib import Composite, R, ZERO
from composite.uncertainty import (Quantity, budget, montecarlo,
                                   NORMAL, RECTANGULAR, SENSITIVITY_ALARM)
from test_dimension_scales import Suite, head

cl.MAX_ACTIVE_DIMS = 10 ** 9


def _isf(z):
    return isinstance(z, float)


def _pw(b, e):
    if _isf(b) and _isf(e):
        return b ** e
    return cl.exp(e * cl.ln(b)) if not _isf(e) else b ** e


def _sqrt(z):
    return math.sqrt(z) if _isf(z) else z ** 0.5


# =============================================================================
def u1_linear_is_exact(t):
    head("U1  a linear model has no bias, no skew, and no need of anything more")
    m = lambda x, y: 3.0 * x - 2.0 * y
    b = budget(m, {"x": Quantity(10.0, 0.5), "y": Quantity(4.0, 0.25)})
    want = math.sqrt((3 * 0.5) ** 2 + (2 * 0.25) ** 2)
    t.close(f"U1.01 u_c = {b.u_linear:.12g}, derived {want:.12g}", b.u_linear, want, tol=1e-15)
    t.close(f"U1.02 higher-order u_c is the same: {b.u_higher:.12g}", b.u_higher, want, tol=1e-15)
    t.true(f"U1.03 bias is exactly zero: {b.bias!r}", b.bias == 0.0, f"{b.bias!r}")
    t.true(f"U1.04 skewness is exactly zero: {b.skewness!r}", b.skewness == 0.0, f"{b.skewness!r}")
    t.true(f"U1.05 verdict {b.verdict!r}", b.linear_is_adequate, b.verdict)
    t.close(f"U1.06 sensitivity to x is 3: {b.contributions[0].sensitivity}",
            [c for c in b.contributions if c.name == "x"][0].sensitivity, 3.0, tol=1e-15)


# =============================================================================
def u2_square_normal(t):
    head("U2  y = x^2 with a normal input -- bias, variance and third moment")
    mu, u = 2.0, 0.1
    b = budget(lambda x: x * x, {"x": Quantity(mu, u, NORMAL)})
    bias_x = u * u
    var_x = 4 * mu ** 2 * u ** 2 + 2 * u ** 4
    mu3_x = 24 * mu ** 2 * u ** 4 + 8 * u ** 6
    t.close(f"U2.01 bias {b.bias:.12g}, exact {bias_x:.12g}", b.bias, bias_x, tol=1e-14)
    t.close(f"U2.02 u_c(higher) {b.u_higher:.12g}, exact sqrt(var) {math.sqrt(var_x):.12g}",
            b.u_higher, math.sqrt(var_x), tol=1e-14)
    t.true(f"U2.03 the LINEAR u_c {b.u_linear:.12g} misses the 2u^4 term "
           f"(exact {math.sqrt(var_x):.12g}, short by {math.sqrt(var_x)-b.u_linear:.3e})",
           b.u_linear < math.sqrt(var_x), f"{b.u_linear} !< {math.sqrt(var_x)}")
    got_mu3 = b.skewness * b.u_higher ** 3
    t.close(f"U2.04 mu3 {got_mu3:.15g}, exact 24mu^2u^4 + 8u^6 = {mu3_x:.15g} "
            f"(the 8u^6 comes from the purely quadratic term)",
            got_mu3, mu3_x, tol=1.0e-15)
    t.close(f"U2.05 second derivative is 2: {b.contributions[0].second:.12g}",
            b.contributions[0].second, 2.0, tol=1e-12)


# =============================================================================
def u3_square_rectangular(t):
    head("U3  the same model on a rectangular input -- a different fourth moment")
    mu, a = 2.0, 0.3
    q = Quantity.from_halfwidth(mu, a, RECTANGULAR)
    t.close(f"U3.01 half-width {a} becomes u = {q.u:.12g}", q.u, a / math.sqrt(3.0), tol=1e-15)
    b = budget(lambda x: x * x, {"x": q})
    var_x = (4.0 / 3.0) * mu ** 2 * a ** 2 + 4.0 * a ** 4 / 45.0
    t.close(f"U3.02 u_c(higher) {b.u_higher:.12g}, exact {math.sqrt(var_x):.12g}",
            b.u_higher, math.sqrt(var_x), tol=1e-14)
    bn = budget(lambda x: x * x, {"x": Quantity(mu, q.u, NORMAL)})
    t.true(f"U3.03 shape changes the answer: rectangular {b.u_higher:.10g} vs "
           f"normal {bn.u_higher:.10g} at identical u",
           abs(b.u_higher - bn.u_higher) > 1e-6, f"{b.u_higher} vs {bn.u_higher}")


# =============================================================================
def u4_product_cross_terms(t):
    head("U4  y = x1*x2 -- the cross terms, from directional derivatives only")
    m1, s1, m2, s2 = 5.0, 0.2, 3.0, 0.15
    b = budget(lambda a, c: a * c,
               {"a": Quantity(m1, s1), "c": Quantity(m2, s2)})
    var_x = s1 ** 2 * m2 ** 2 + s2 ** 2 * m1 ** 2 + s1 ** 2 * s2 ** 2
    mu3_x = 6.0 * m1 * m2 * s1 ** 2 * s2 ** 2
    t.close(f"U4.01 u_c(higher) {b.u_higher:.12g}, exact {math.sqrt(var_x):.12g}",
            b.u_higher, math.sqrt(var_x), tol=1e-14)
    t.true(f"U4.02 the u1^2u2^2 term is the cross term: linear {b.u_linear:.12g}, "
           f"higher {b.u_higher:.12g}, difference {b.u_higher-b.u_linear:.3e}",
           b.u_higher > b.u_linear, f"{b.u_higher} !> {b.u_linear}")
    got = b.skewness * b.u_higher ** 3
    t.close(f"U4.03 mu3 {got:.12g}, exact {mu3_x:.12g}", got, mu3_x, tol=1e-12)
    t.true(f"U4.04 bias is zero for a bilinear model: {b.bias:.3e}",
           abs(b.bias) < 1e-15, f"{b.bias!r}")
    # without cross terms the u1^2u2^2 is simply absent
    nb = budget(lambda a, c: a * c,
                {"a": Quantity(m1, s1), "c": Quantity(m2, s2)}, cross_terms=False)
    t.close(f"U4.05 cross_terms=False drops it: {nb.u_higher:.12g} = linear "
            f"{nb.u_linear:.12g}", nb.u_higher, nb.u_linear, tol=1e-15)


# =============================================================================
def u5_lognormal(t):
    head("U5  y = exp(x) -- against the exact lognormal moments")
    mu, u = 0.0, 0.1
    # mu is exactly 0.  The estimate is an absent term; see U12.
    b = budget(lambda x: cl.exp(x), {"x": Quantity(mu, u)})
    e_x = math.exp(mu + u * u / 2.0)
    var_x = (math.exp(u * u) - 1.0) * math.exp(2 * mu + u * u)
    skew_x = (math.exp(u * u) + 2.0) * math.sqrt(math.exp(u * u) - 1.0)
    t.close(f"U5.01 mean {b.mean:.12g}, exact {e_x:.12g} "
            f"(bias carries both the f'' and the f term)", b.mean, e_x, tol=1e-7)
    t.close(f"U5.02 u_c(higher) {b.u_higher:.12g}, exact {math.sqrt(var_x):.12g}",
            b.u_higher, math.sqrt(var_x), tol=1e-4)
    t.close(f"U5.03 skewness {b.skewness:.12g}, exact {skew_x:.12g} "
            f"(second-order truncation of a lognormal)",
            b.skewness, skew_x, tol=5e-2)
    t.true(f"U5.04 the linear budget ignores the mean shift entirely: "
           f"value {b.value:.12g} vs true mean {e_x:.12g}",
           abs(b.value - e_x) > 1e-4, f"{b.value} {e_x}")


# =============================================================================
def u6_verdict(t):
    head("U6  the verdict itself -- when the linear budget must be refused")
    small = budget(lambda x: cl.exp(x), {"x": Quantity(0.0, 0.002)})
    t.true(f"U6.01 exp(x) at u=0.002 -> {small.verdict!r} (skew {small.skewness:.3e})",
           small.linear_is_adequate, small.verdict + " " + str(small.reasons))
    big = budget(lambda x: cl.exp(x), {"x": Quantity(0.0, 0.5)})
    t.true(f"U6.02 exp(x) at u=0.5 -> {big.verdict!r} (skew {big.skewness:.4g})",
           big.verdict.startswith("MONTE CARLO"), big.verdict + " " + str(big.reasons))
    t.true(f"U6.03 and it says why: {big.reasons[0] if big.reasons else '(none)'}",
           len(big.reasons) > 0, "no reason recorded")
    mid = budget(lambda x: x * x, {"x": Quantity(2.0, 0.1)})
    t.true(f"U6.04 y=x^2 at mu=2,u=0.1 -> {mid.verdict!r}; bias {mid.bias:.3e} "
           f"vs tolerance {mid.tolerance:.3e}",
           not mid.linear_is_adequate, mid.verdict)
    t.true(f"U6.05 the corrected interval is asymmetric about the mean: "
           f"[{mid.interval_corrected[0]:.6g}, {mid.interval_corrected[1]:.6g}] "
           f"about {mid.mean:.6g}",
           abs((mid.interval_corrected[1] - mid.mean)
               - (mid.mean - mid.interval_corrected[0])) > 1e-9,
           "interval came out symmetric")


# =============================================================================
def u7_monte_carlo(t):
    head("U7  the analytic moments checked against GUM-S1 sampling, not trusted")
    mu, u = 2.0, 0.1
    b = budget(lambda x: x * x, {"x": Quantity(mu, u)})
    mc = montecarlo(lambda x: x * x, {"x": Quantity(mu, u)}, trials=200000)
    t.close(f"U7.01 u_c analytic {b.u_higher:.8g} vs Monte Carlo {mc.u:.8g} "
            f"({mc.trials} trials)", b.u_higher, mc.u, tol=1e-2)
    t.true(f"U7.02 mean analytic {b.mean:.10g} vs Monte Carlo {mc.mean:.10g} "
           f"+- {mc.stderr:.2e} (antithetic); exact mu^2+u^2 = {mu*mu+u*u:.10g}",
           mc.mean_agrees(b.mean, 3.0),
           f"analytic {b.mean} mc {mc.mean} stderr {mc.stderr}")
    t.close(f"U7.02b and the analytic mean is exactly mu^2+u^2", b.mean,
            mu * mu + u * u, tol=1e-14)
    t.close(f"U7.03 skew analytic {b.skewness:.6g} vs Monte Carlo {mc.skewness:.6g}",
            b.skewness, mc.skewness, tol=5e-2)
    m1, s1, m2, s2 = 5.0, 0.2, 3.0, 0.15
    ins = {"a": Quantity(m1, s1), "c": Quantity(m2, s2)}
    bp = budget(lambda a, c: a * c, ins)
    mcp = montecarlo(lambda a, c: a * c, ins, trials=200000)
    t.close(f"U7.04 product u_c analytic {bp.u_higher:.8g} vs MC {mcp.u:.8g}",
            bp.u_higher, mcp.u, tol=1e-2)
    t.close(f"U7.05 product skew analytic {bp.skewness:.6g} vs MC {mcp.skewness:.6g}",
            bp.skewness, mcp.skewness, tol=5e-2)


# =============================================================================
def u8_escape_to_float(t):
    head("U8  a model that leaves composite arithmetic is refused, not guessed at")
    def leaky(x):
        return math.exp(float(x))          # float() drops the infinitesimal
    try:
        budget(leaky, {"x": Quantity(1.0, 0.1)})
        t.true("U8.01 a leaking model raises", False, "no exception raised")
    except TypeError as e:
        t.true(f"U8.01 refused with: {str(e)[:60]}...", "not a Composite" in str(e), str(e))
    ok = budget(lambda x: cl.exp(x), {"x": Quantity(1.0, 0.1)})
    t.close(f"U8.02 the composite spelling works: dy/dx = {ok.contributions[0].sensitivity:.12g}, "
            f"exact e = {math.e:.12g}", ok.contributions[0].sensitivity, math.e, tol=1e-12)


# =============================================================================
def u9_expressed_zero_trap(t):
    head("U9  a vanishing scalar prefactor corrupts the derivative, not the value")
    # This is ISO 5167's tapping bracket: exactly 0.0 for corner taps.
    prefactor = 0.043 + 0.080 * math.exp(-0.0) - 0.123 * math.exp(-0.0)
    t.true(f"U9.01 the prefactor is exactly zero: {prefactor!r}", prefactor == 0.0,
           f"{prefactor!r}")

    def trap(x):
        return x + prefactor * (x * x)     # mathematically just x

    x0 = 2.0
    b = budget(trap, {"x": Quantity(x0, 0.1)})
    t.close(f"U9.02 the VALUE is still exactly right: {b.value:.12g}", b.value, x0, tol=1e-15)
    got = b.contributions[0].sensitivity
    t.true(f"U9.03 but dy/dx reads {got:.12g} where the truth is 1.0 -- R1 converted "
           f"the expressed zero and x^2 = {x0*x0:.6g} landed in the derivative",
           abs(got - 1.0) > 1.0, f"got {got}")
    t.true(f"U9.04 the cross-check caught it: {b.verdict!r}, "
           f"disagreement {b.contributions[0].fd_check:.3e} > alarm {SENSITIVITY_ALARM:.0e}",
           b.verdict == "SENSITIVITY SUSPECT", b.verdict)

    def guarded(x):
        return x + prefactor * (x * x) if prefactor else x

    g = budget(guarded, {"x": Quantity(x0, 0.1)})
    t.close(f"U9.05 guarding the multiplication fixes it: dy/dx = "
            f"{g.contributions[0].sensitivity:.12g}",
            g.contributions[0].sensitivity, 1.0, tol=1e-15)
    t.true(f"U9.06 and the verdict clears: {g.verdict!r}", g.linear_is_adequate, g.verdict)


# =============================================================================
def _orifice(d, D, dp, rho, mu, p1, kappa, iters=10):
    """ISO 5167-2 orifice plate, corner tappings, Reader-Harris/Gallagher C."""
    beta = d / D
    b2 = beta * beta
    b4 = b2 * b2
    eps = 1.0 - (0.351 + 0.256 * b4 + 0.93 * b4 * b4) * (1.0 - _pw((p1 - dp) / p1, 1.0 / kappa))
    area = (math.pi / 4.0) * d * d
    Re = R(1.0e6) if not _isf(d) else 1.0e6
    q = None
    for _ in range(iters):
        A = _pw(19000.0 * beta / Re, 0.8)
        C = (0.5961 + 0.0261 * b2 - 0.216 * b4 * b4
             + 0.000521 * _pw(1.0e6 * beta / Re, 0.7)
             + (0.0188 + 0.0063 * A) * _pw(beta, 3.5) * _pw(1.0e6 / Re, 0.3))
        # corner taps: the tapping bracket is identically zero, so it is not
        # written at all.  See U9 for what writing it would cost.
        q = C / _sqrt(1.0 - b4) * eps * area * _sqrt(2.0 * dp * rho)
        Re = 4.0 * q / (math.pi * mu * D)
    return q


def u10_orifice(t):
    head("U10 ISO 5167-2 orifice meter -- the real budget, against sampling")
    ins = {
        "d":     Quantity.relative(0.050, 5e-4),
        "D":     Quantity.relative(0.100, 1e-3),
        "dp":    Quantity.relative(25000.0, 1e-3),
        "rho":   Quantity.relative(998.2, 1e-3),
        "mu":    Quantity.relative(1.002e-3, 2e-2),
        "p1":    Quantity.relative(5.0e5, 5e-3),
        "kappa": Quantity.relative(1.3, 1e-2),
    }
    b = budget(_orifice, ins)
    print(b.table())
    t.close(f"U10.01 Q_m = {b.value:.8g} kg/s, scalar model {_orifice(**{k: v.value for k, v in ins.items()}):.8g}",
            b.value, _orifice(**{k: v.value for k, v in ins.items()}), tol=1e-14)
    worst = max(c.fd_check for c in b.contributions if c.fd_check is not None)
    t.true(f"U10.02 every sensitivity agrees with a difference quotient, worst {worst:.2e}",
           worst < SENSITIVITY_ALARM, f"worst {worst}")
    dom = b.dominant()
    t.true(f"U10.03 the bore dominates: {dom.name!r} at {dom.index:.2f}% of u_c^2",
           dom.name == "d", f"{dom.name} {dom.index}")
    t.true(f"U10.04 u_c = {b.u_linear:.6e} kg/s = {100*b.u_linear/b.value:.4f}% of reading",
           0.0005 < b.u_linear / b.value < 0.01, f"{b.u_linear/b.value}")
    mc = montecarlo(_orifice, ins, trials=20000)
    t.close(f"U10.05 u_c analytic {b.u_higher:.6e} vs Monte Carlo {mc.u:.6e} "
            f"({mc.trials} trials)", b.u_higher, mc.u, tol=3e-2)
    t.true(f"U10.06 mean analytic {b.mean:.10g} vs Monte Carlo {mc.mean:.10g} "
           f"+- {mc.stderr:.2e} (antithetic, {mc.trials} trials); "
           f"bias {b.bias:+.3e}",
           mc.mean_agrees(b.mean, 3.0),
           f"analytic {b.mean} mc {mc.mean} stderr {mc.stderr}")
    t.true(f"U10.07 verdict {b.verdict!r} -- a well-conditioned meter needs no "
           f"Monte Carlo (skew {b.skewness:.3e})",
           not b.verdict.startswith("MONTE CARLO"), b.verdict)


# =============================================================================
def _comparison_loss(a, b):
    """GUM-S1 clause 9.4: comparison loss in microwave power meter calibration.

    Y = 1 - |Gamma|^2 with Gamma = a + i b.  Both components are estimated at
    zero, so every dY/dx vanishes there and the GUM's linearised budget returns
    zero uncertainty for a quantity whose uncertainty is plainly not zero.  It
    is the standard's own illustration of when its main procedure fails.
    """
    return 1.0 - a * a - b * b


def u11_flat_response(t):
    head("U11 GUM-S1 9.4 -- a response that is flat at the estimate")
    u = 0.005
    ins = {"a": Quantity(0.0, u), "b": Quantity(0.0, u)}
    r = budget(_comparison_loss, ins)
    t.true(f"U11.01 the LINEAR budget returns zero uncertainty: u_c = {r.u_linear!r}",
           r.u_linear == 0.0, f"{r.u_linear!r}")
    t.close(f"U11.02 u_c from curvature {r.u_higher:.12g}, exact 2u^2 = {2*u*u:.12g}",
            r.u_higher, 2 * u * u, tol=1e-17)
    t.close(f"U11.03 bias {r.bias:.12g}, exact -2u^2 = {-2*u*u:.12g}",
            r.bias, -2 * u * u, tol=1e-17)
    t.close(f"U11.04 skewness {r.skewness:.12g}, exact -2 (chi-square with 2 df)",
            r.skewness, -2.0, tol=1e-12)
    t.true(f"U11.05 verdict {r.verdict!r}", r.verdict.startswith("MONTE CARLO"), r.verdict)
    t.true(f"U11.06 and it names the cause: {r.reasons[0][:58]}...",
           "vanish" in r.reasons[0], r.reasons[0])
    mc = montecarlo(_comparison_loss, ins, trials=200000)
    t.close(f"U11.07 u_c {r.u_higher:.8e} vs Monte Carlo {mc.u:.8e}",
            r.u_higher, mc.u, tol=1e-6)
    t.true(f"U11.08 mean {r.mean:.12g} vs Monte Carlo {mc.mean:.12g} "
           f"+- {mc.stderr:.2e}", mc.mean_agrees(r.mean, 3.0),
           f"{r.mean} vs {mc.mean} +- {mc.stderr}")
    t.close(f"U11.09 skew {r.skewness:.6g} vs Monte Carlo {mc.skewness:.6g}",
            r.skewness, mc.skewness, tol=5e-2)
    # away from the degenerate point the ordinary machinery must still agree
    off = {"a": Quantity(0.005, u), "b": Quantity(0.005, u)}
    ro = budget(_comparison_loss, off)
    mo = montecarlo(_comparison_loss, off, trials=200000)
    t.close(f"U11.10 at a non-zero estimate: u_c {ro.u_higher:.8e} vs MC {mo.u:.8e}",
            ro.u_higher, mo.u, tol=1e-6)
    t.close(f"U11.11 and its skew {ro.skewness:.6g} vs MC {mo.skewness:.6g}",
            ro.skewness, mo.skewness, tol=5e-2)


# =============================================================================
def u12_zero_estimate_is_absent(t):
    head("U12 an estimate of exactly zero is an absent term")
    ins = {"a": Quantity(0.0, 0.005), "b": Quantity(0.0, 0.005)}
    ok = budget(_comparison_loss, ins)
    t.close(f"U12.01 it just works, by default: u_c = {ok.u_higher:.6e}, "
            f"exact 2u^2 = {2*0.005**2:.6e}", ok.u_higher, 2 * 0.005 ** 2, tol=1e-17)
    t.true("U12.02 an unknown mode is still rejected",
           _raises(lambda: budget(_comparison_loss, ins, zero_estimate="sure")),
           "no exception for a bad mode")
    t.true("U12.03 'refuse' is available for a caller who wants telling",
           _raises(lambda: budget(_comparison_loss, ins, zero_estimate="refuse")),
           "refuse did not raise")
    # a bare 0.0 reaching the model is what the refusal is protecting against
    from composite.uncertainty import _seed
    bare = _seed({"a": 0.0, "b": 0.0}, {"a": 1.0}, zero_absent=False)
    absent = _seed({"a": 0.0, "b": 0.0}, {"a": 1.0}, zero_absent=True)
    t.close(f"U12.05 bare 0.0 for the un-seeded input gives df/da = "
            f"{_comparison_loss(**bare).d(1)}, which is wrong",
            _comparison_loss(**bare).d(1), -1.0, tol=1e-15)
    t.close(f"U12.06 absent gives df/da = {_comparison_loss(**absent).d(1)}, "
            f"which is right", _comparison_loss(**absent).d(1), 0.0, tol=1e-15)
    t.close(f"U12.07 and d2f/da2 = {_comparison_loss(**absent).d(2)}, exact -2",
            _comparison_loss(**absent).d(2), -2.0, tol=1e-15)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def run_all():
    t = Suite()
    for fn in (u1_linear_is_exact, u2_square_normal, u3_square_rectangular,
               u4_product_cross_terms, u5_lognormal, u6_verdict,
               u7_monte_carlo, u8_escape_to_float, u9_expressed_zero_trap,
               u10_orifice, u11_flat_response, u12_zero_estimate_is_absent):
        try:
            fn(t)
        except Exception as e:
            t._note(f"{fn.__name__} ABORTED", False, f"{type(e).__name__}: {e}")
    total = t.passed + t.failed
    print(f"\n{'=' * 66}")
    print(f"RESULTS: {t.passed}/{total} passed")
    if t.fails:
        print("\nFailed:")
        for f in t.fails:
            print(f"  - {f}")
    print("=" * 66)
    return 0 if t.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
