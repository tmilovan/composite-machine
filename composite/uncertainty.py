"""Measurement uncertainty budgets, and the verdict on whether the linear one is admissible.

The GUM (JCGM 100:2008) propagates uncertainty through a measurement model by
linearising it:  u_c^2 = sum_i c_i^2 u_i^2,  with the sensitivity coefficients
c_i = df/dx_i.  That is exact only for a linear model, and the standard says so:
clause 5.1.2 Note carries a higher-order correction, and Supplement 1
(JCGM 101:2008) replaces the whole procedure with a Monte Carlo propagation of
distributions when the linearisation is not good enough.

Deciding WHICH regime a given measurement is in is, in practice, a judgement an
engineer makes per application -- usually from experience, sometimes not at all.
This module computes the decision instead of judging it.

The sensitivity coefficients, the second derivatives that carry the bias, and
the third derivatives that carry the GUM higher-order variance term all come
out of a composite evaluation as coefficients of the infinitesimal.  There is no
perturbation size to choose, which matters here because the quantity the
decision turns on is a SECOND derivative, and a hand-picked step that is merely
adequate for a first derivative is routinely useless for a second one.

Scope and honesty about what is new
-----------------------------------
First-order GUM propagation by automatic differentiation is not new; the Python
``uncertainties`` package has done it for years and does it well.  What is here
that is not there: the bias, the higher-order variance, the third and fourth
moments of the output, and the GUM-S1 validation check run against those moments
rather than against a million-sample Monte Carlo.

Everything below is a Taylor construction, so it degrades the way Taylor
constructions degrade: it is trustworthy when the model is smooth over a few
standard uncertainties and says nothing useful when it is not.  ``montecarlo()``
is provided for exactly that case, and ``validate=True`` on :func:`budget`
cross-checks the analytic moments against it.

Derivatives without a multivariate number
-----------------------------------------
Mixed partials are needed for the cross terms, but they are obtained from
DIRECTIONAL derivatives along a line -- one infinitesimal, seeded into two
inputs at once -- not from a multivariate composite:

    D2 along (e_i + e_j)  =  f_ii + 2 f_ij + f_jj        ->  f_ij
    D3 along (e_i +- e_j) =  f_iii +- 3 f_iij + 3 f_ijj +- f_jjj

so f_ij, f_ijj and f_iij fall out of ordinary single-axis seeding.

References
----------
JCGM 100:2008 (GUM), clauses 5.1.2 and G.  JCGM 101:2008 (GUM Supplement 1),
clauses 7 and 8.  Cornish & Fisher (1938) for the quantile expansion.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import composite_lib as _cl
from .composite_lib import Composite, R, ZERO

__all__ = [
    "Quantity", "Contribution", "Budget",
    "budget", "montecarlo", "NORMAL", "RECTANGULAR", "TRIANGULAR", "ARCSINE",
]

# Excess kurtosis of the input distribution.  It is the only shape parameter the
# second-order moment formulas need, because every distribution the GUM admits
# for a Type B evaluation is symmetric about its estimate.
NORMAL = 0.0
RECTANGULAR = -1.2
TRIANGULAR = -0.6
ARCSINE = -1.5

# A first-order sensitivity that disagrees with a plain difference quotient by
# more than this is not a precision matter -- a central difference is good to
# roughly 1e-10 relative on a well-scaled model, so anything past 1e-4 means the
# two are computing different things.
SENSITIVITY_ALARM = 1.0e-4

_SHAPE_NAME = {
    NORMAL: "normal", RECTANGULAR: "rectangular",
    TRIANGULAR: "triangular", ARCSINE: "arcsine (U-shaped)",
}

# Standardised sixth central moment, m6 / u^6.  Needed for the third moment of
# a purely quadratic response, where the linear term vanishes and every lower
# moment of the output comes from the curvature alone.
_M6 = {
    NORMAL: 15.0,
    RECTANGULAR: 27.0 / 7.0,
    TRIANGULAR: 216.0 / 28.0,
    ARCSINE: 2.5,
}


@dataclass(frozen=True)
class Quantity:
    """One input: its estimate, its standard uncertainty, and its distribution.

    ``u`` is always a STANDARD uncertainty.  A Type B half-width ``a`` from a
    rectangular distribution becomes ``u = a / sqrt(3)``; use
    :meth:`from_halfwidth` rather than converting by hand.
    """

    value: float
    u: float
    shape: float = NORMAL
    unit: str = ""

    @classmethod
    def from_halfwidth(cls, value: float, halfwidth: float,
                       shape: float = RECTANGULAR, unit: str = "") -> "Quantity":
        divisor = {RECTANGULAR: math.sqrt(3.0), TRIANGULAR: math.sqrt(6.0),
                   ARCSINE: math.sqrt(2.0)}.get(shape)
        if divisor is None:
            raise ValueError("no standard half-width divisor for shape %r" % (shape,))
        return cls(value, halfwidth / divisor, shape, unit)

    @classmethod
    def relative(cls, value: float, rel: float,
                 shape: float = NORMAL, unit: str = "") -> "Quantity":
        """Standard uncertainty given as a fraction of the estimate."""
        return cls(value, abs(value) * rel, shape, unit)

    @property
    def shape_name(self) -> str:
        return _SHAPE_NAME.get(self.shape, "excess kurtosis %+.3g" % self.shape)


@dataclass
class Contribution:
    """One row of the uncertainty budget."""

    name: str
    value: float
    u: float
    shape: float
    sensitivity: float          # c_i = df/dx_i
    second: float               # d2f/dx_i2
    third: float                # d3f/dx_i3
    contribution: float         # c_i * u_i, signed
    index: float                # share of u_c^2, per cent
    bias: float                 # this input's share of the second-order bias
    fd_check: Optional[float] = None   # relative disagreement with a difference quotient

    @property
    def variance(self) -> float:
        return (self.sensitivity * self.u) ** 2

    @property
    def shape_name(self) -> str:
        """Same spelling as Quantity.shape_name, so a row reads like its input."""
        return _SHAPE_NAME.get(self.shape, "excess kurtosis %+.3g" % self.shape)


@dataclass
class Budget:
    """A complete budget, plus the verdict on whether its linear part suffices."""

    value: float                # f(x), the model at the input estimates
    contributions: List[Contribution]
    u_linear: float             # the ordinary GUM combined standard uncertainty
    u_higher: float             # with the clause 5.1.2 higher-order terms
    bias: float                 # E[y] - f(x); the GUM assumes this is zero
    skewness: float
    excess_kurtosis: float
    coverage: float
    k: float                    # coverage factor actually used by the GUM interval
    interval_linear: Tuple[float, float]
    interval_corrected: Tuple[float, float]
    verdict: str
    reasons: List[str] = field(default_factory=list)
    digits: int = 2
    tolerance: float = 0.0      # GUM-S1 clause 8 numerical tolerance
    mc: Optional["MonteCarloResult"] = None

    @property
    def mean(self) -> float:
        """The expectation of the output, which is the estimate PLUS the bias."""
        return self.value + self.bias

    @property
    def expanded_linear(self) -> float:
        return self.k * self.u_linear

    @property
    def linear_is_adequate(self) -> bool:
        return self.verdict == "LINEAR GUM ADEQUATE"

    def dominant(self) -> Optional[Contribution]:
        return max(self.contributions, key=lambda c: c.variance, default=None)

    # -- presentation ----------------------------------------------------
    def table(self) -> str:
        w = max([len(c.name) for c in self.contributions] + [5])
        out = []
        head = ("%-*s %-13s %-12s %-14s %-13s %-8s %s"
                % (w, "input", "estimate", "u(x)", "c = df/dx", "c*u(x)", "index%", "dist"))
        out.append(head)
        out.append("-" * len(head))
        for c in self.contributions:
            out.append("%-*s %-13.6g %-12.6g %-14.6g %-13.4e %-8.2f %s"
                       % (w, c.name, c.value, c.u, c.sensitivity,
                          c.contribution, c.index, _SHAPE_NAME.get(c.shape, "?")))
        out.append("-" * len(head))
        out.append("%-*s %s" % (w, "value", "%.10g" % self.value))
        out.append("%-*s %s" % (w, "u_c (linear)", "%.6e" % self.u_linear))
        out.append("%-*s %s" % (w, "u_c (higher)", "%.6e" % self.u_higher))
        out.append("%-*s %s" % (w, "bias", "%+.6e" % self.bias))
        out.append("%-*s %s" % (w, "skewness", "%+.6g" % self.skewness))
        out.append("%-*s %s" % (w, "ex. kurtosis", "%+.6g" % self.excess_kurtosis))
        out.append("%-*s %s" % (w, "%.0f%% (k=%.3f)" % (100 * self.coverage, self.k),
                                "[%.10g, %.10g]" % self.interval_linear))
        out.append("%-*s %s" % (w, "%.0f%% corrected" % (100 * self.coverage,),
                                "[%.10g, %.10g]" % self.interval_corrected))
        out.append("%-*s %s" % (w, "verdict", self.verdict))
        for r in self.reasons:
            out.append("%-*s   - %s" % (w, "", r))
        if any(c.fd_check is not None for c in self.contributions):
            worst = max((c for c in self.contributions if c.fd_check is not None),
                        key=lambda c: c.fd_check)
            out.append("%-*s %s" % (w, "fd cross-check",
                                    "worst %.2e on %r" % (worst.fd_check, worst.name)))
        return "\n".join(out)

    def __str__(self) -> str:          # pragma: no cover - convenience
        return self.table()


# ----------------------------------------------------------------------
# derivatives
# ----------------------------------------------------------------------

@contextlib.contextmanager
def _jet_scope(order: int):
    """Hold the dimension cap at what a jet of this order actually needs.

    A budget reads four derivatives and nothing else, but a measurement model
    is often a fixed point -- ISO 5167's discharge coefficient is solved
    iteratively -- and every iteration convolves the series with itself.  Left
    uncapped the term count grows without bound while the extra grades are
    read by nobody: the same eight-point turndown sweep does not finish in ten
    minutes uncapped and takes seconds capped, for identical numbers.

    The library's own truncation keeps the dimensions CLOSEST TO ZERO, so
    capping discards exactly the high-order tail that is not being read.  The
    margin above ``order`` is there for the transcendental expansions, whose
    intermediate terms feed back down into the low grades.
    """
    old = _cl.MAX_ACTIVE_DIMS
    _cl.MAX_ACTIVE_DIMS = order + 8
    try:
        yield
    finally:
        _cl.MAX_ACTIVE_DIMS = old


def _seed(values: Dict[str, float], directions: Dict[str, float],
          zero_absent: bool = False) -> Dict[str, object]:
    """Inputs with a single infinitesimal seeded along ``directions``.

    An estimate of exactly zero is built as ``Composite({-1: w})`` -- the
    grade-0 term ABSENT -- and not as ``R(0.0) + w*ZERO``.  Writing the zero
    would express it, R1 would convert it to a unit one grade down, and the
    seed would arrive carrying an extra infinitesimal that lands straight in
    the first derivative.  Nothing is being coerced here: there is no term to
    write, so none is written.
    """
    out: Dict[str, object] = {}
    for name, v in values.items():
        if name in directions:
            terms = {-1: float(directions[name])}
            if v != 0.0:
                terms[0] = float(v)
            out[name] = Composite(terms)
        elif v == 0.0 and zero_absent:
            out[name] = Composite({})
        else:
            out[name] = v
    return out


def _derivs(model: Callable[..., object], values: Dict[str, float],
            directions: Dict[str, float], order: int,
            zero_absent: bool = False) -> List[float]:
    """Directional derivatives 1..order of ``model`` along ``directions``."""
    with _jet_scope(order):
        y = model(**_seed(values, directions, zero_absent))
    if not isinstance(y, Composite):
        raise TypeError(
            "the model returned %s, not a Composite, when an input carried an "
            "infinitesimal.  Some branch has dropped to plain floats -- a "
            "math.* call, an int() or float(), or a comparison that took the "
            "scalar path -- and the derivative is lost.  Every operation on a "
            "seeded input must stay in composite arithmetic."
            % type(y).__name__)
    return [y.d(n) for n in range(1, order + 1)]


def _model_value(model: Callable[..., object], values: Dict[str, float]) -> float:
    y = model(**values)
    return y.st() if isinstance(y, Composite) else float(y)


def _fd_sensitivity(model: Callable[..., object], values: Dict[str, float],
                    name: str) -> Optional[float]:
    """A central difference, used only to cross-check the composite result.

    The step is the usual cube-root-of-epsilon compromise.  It is nowhere near
    as accurate as the composite derivative -- that is the point of the
    composite -- but it is accurate enough to catch a sensitivity that is
    wrong by orders of magnitude, which is the failure this guards against.
    """
    x = values[name]
    h = (abs(x) if x else 1.0) * 6.055454452393343e-06     # eps ** (1/3)
    if h == 0.0:
        return None
    lo, hi = dict(values), dict(values)
    lo[name], hi[name] = x - h, x + h
    try:
        return (_model_value(model, hi) - _model_value(model, lo)) / (2.0 * h)
    except Exception:
        return None


# ----------------------------------------------------------------------
# the budget
# ----------------------------------------------------------------------

def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, refined)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    # one Halley refinement
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    v = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    return x - e / v / (1 + x * e / v / 2)


def _cornish_fisher(z: float, skew: float, exkurt: float) -> float:
    """Quantile of a standardised distribution with the given skewness/kurtosis."""
    return (z
            + (z * z - 1.0) * skew / 6.0
            + (z ** 3 - 3.0 * z) * exkurt / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * skew * skew / 36.0)


def _gum_s1_tolerance(u: float, digits: int) -> float:
    """GUM Supplement 1 clause 8.2: half a unit in the last reported digit of u_c."""
    if u <= 0.0 or not math.isfinite(u):
        return 0.0
    exponent = math.floor(math.log10(abs(u))) - (digits - 1)
    return 0.5 * 10.0 ** exponent


def budget(model: Callable[..., object],
           inputs: Dict[str, Quantity],
           coverage: float = 0.95,
           digits: int = 2,
           cross_terms: bool = True,
           zero_estimate: str = "absent",
           check_sensitivities: bool = True,
           validate: bool = False,
           mc_trials: int = 200000,
           seed: int = 20260922) -> Budget:
    """Build the budget and decide whether its linear part is admissible.

    ``model`` takes the input names as keyword arguments and returns the
    measurand.  It must be written so that composite arithmetic flows through
    it: use ``composite.composite_lib`` transcendentals rather than ``math.*``
    on any quantity that can carry an infinitesimal.

    ``validate=True`` runs a Monte Carlo propagation as well and attaches it, so
    the analytic moments can be checked rather than trusted.

    An input estimated at exactly zero is presented to the model as
    ``Composite({})``, an absent term -- which is what ``x = 0 +- u`` states:
    no term applied, the uncertainty carried separately.  ``zero_estimate=
    'refuse'`` raises instead, for a caller who would rather be told than have
    the model see an absence.

    ``check_sensitivities`` (on by default) cross-checks every first-order
    coefficient against a difference quotient and records the disagreement on
    the row.  It costs two extra model evaluations per input and exists because
    of a specific, silent failure: a term whose scalar prefactor evaluates to
    exactly ``0.0`` -- common in transcribed standard formulas, where a bracket
    such as ISO 5167's ``0.043 + 0.080 e^-10L1 - 0.123 e^-7L1`` vanishes for one
    tapping arrangement -- multiplies a composite as an EXPRESSED zero, R1
    converts it, and the term's value reappears as a derivative.  The value of
    the model stays perfectly correct while the sensitivity is wrong by orders
    of magnitude.  Guard the multiplication (``if prefactor:``) rather than the
    algebra; this check tells you that you have not.
    """
    if not inputs:
        raise ValueError("a budget needs at least one input")
    names = list(inputs)
    values = {n: float(inputs[n].value) for n in names}

    # An estimate of exactly zero is an ABSENT term, which is Composite({}).
    # In a budget, "x = 0 +- u" says that no term is applied and that the
    # uncertainty about it is carried separately -- the estimate contributes
    # nothing and u carries the infinitesimal.  There is nothing to write.
    #
    # Writing one instead expresses a zero, R1 converts it, and the residue
    # lands on the same single axis the seeded variable's jet is being read
    # from.  That collision is the mechanical reason this matters here: the
    # residue belongs to a DIFFERENT input than the one being differentiated,
    # and nothing downstream can tell the two apart.  For 1 - a^2 - b^2 at the
    # origin it puts the linear u_c at 100x its true value.
    zeros = [n for n in names if values[n] == 0.0]
    zero_absent = zero_estimate == "absent"
    if zero_estimate not in ("absent", "refuse"):
        raise ValueError("zero_estimate must be 'absent' or 'refuse', not %r"
                         % (zero_estimate,))
    if zeros and not zero_absent:
        raise ValueError(
            "input(s) %s have an estimate of exactly 0.0, and zero_estimate="
            "'refuse' was asked for.  The default, 'absent', presents them as "
            "Composite({}) -- which is what an unapplied term is." % (
                ", ".join(repr(z) for z in zeros),))
    y0 = _model_value(model, values)

    # -- diagonal derivatives, one seeded evaluation per input ------------
    f1: Dict[str, float] = {}
    f2: Dict[str, float] = {}
    f3: Dict[str, float] = {}
    f4: Dict[str, float] = {}
    for n in names:
        d1, d2, d3, d4 = _derivs(model, values, {n: 1.0}, 4, zero_absent)
        f1[n], f2[n], f3[n], f4[n] = d1, d2, d3, d4

    # -- mixed partials, from directional derivatives --------------------
    fij: Dict[Tuple[str, str], float] = {}
    fijj: Dict[Tuple[str, str], float] = {}
    if cross_terms and len(names) > 1:
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                i, j = names[a], names[b]
                dp = _derivs(model, values, {i: 1.0, j: 1.0}, 3, zero_absent)
                dm = _derivs(model, values, {i: 1.0, j: -1.0}, 3, zero_absent)
                fij[(i, j)] = 0.5 * (dp[1] - f2[i] - f2[j])
                # D3(+) = f_iii + 3f_iij + 3f_ijj + f_jjj
                # D3(-) = f_iii - 3f_iij + 3f_ijj - f_jjj
                s, d = dp[2] + dm[2], dp[2] - dm[2]
                fijj[(i, j)] = (s - 2.0 * f3[i]) / 6.0
                fijj[(j, i)] = (d - 2.0 * f3[j]) / 6.0

    def mixed(i: str, j: str) -> float:
        if i == j:
            return f2[i]
        return fij.get((i, j), fij.get((j, i), 0.0))

    def mixed3(i: str, j: str) -> float:
        """d3f / dx_i dx_j dx_j."""
        if i == j:
            return f3[i]
        return fijj.get((i, j), 0.0)

    # -- moments ---------------------------------------------------------
    var_linear = sum((f1[n] * inputs[n].u) ** 2 for n in names)
    # E[y] = f + (f_ii/2) m2 + (f_iiii/24) m4 + ...   The GUM takes E[y] = f.
    bias = 0.0
    for n in names:
        ui2 = inputs[n].u ** 2
        m4 = (3.0 + inputs[n].shape) * ui2 * ui2
        bias += 0.5 * f2[n] * ui2 + f4[n] * m4 / 24.0

    var_higher = var_linear
    for i in names:
        ui2 = inputs[i].u ** 2
        # diagonal: (f_ii^2 / 4)(m4 - m2^2) + (f_i f_iii / 3) m4
        m4 = (3.0 + inputs[i].shape) * ui2 * ui2
        var_higher += 0.25 * f2[i] ** 2 * (m4 - ui2 * ui2)
        var_higher += f1[i] * f3[i] * m4 / 3.0
    for a in range(len(names)):
        for b in range(len(names)):
            if a == b:
                continue
            i, j = names[a], names[b]
            ui2, uj2 = inputs[i].u ** 2, inputs[j].u ** 2
            var_higher += 0.5 * mixed(i, j) ** 2 * ui2 * uj2
            var_higher += f1[i] * mixed3(i, j) * ui2 * uj2
    var_higher = max(var_higher, 0.0)

    # third central moment: 3 sum_ij f_i f_j f_ij u_i^2 u_j^2, with the
    # diagonal carrying the input's own fourth moment.
    mu3 = 0.0
    for i in names:
        ui2 = inputs[i].u ** 2
        m4 = (3.0 + inputs[i].shape) * ui2 * ui2
        m6 = _M6.get(inputs[i].shape, 15.0) * ui2 ** 3
        mu3 += 1.5 * f1[i] ** 2 * f2[i] * (m4 - ui2 * ui2)
        # The purely quadratic part: E[(d^2 - m2)^3] = m6 - 3 m2 m4 + 2 m2^3.
        # It is the ONLY surviving third moment when the response is flat at
        # the estimate (every c_i = 0), which is exactly the case the linear
        # budget reports as zero uncertainty.
        mu3 += (f2[i] ** 3 / 8.0) * (m6 - 3.0 * ui2 * m4 + 2.0 * ui2 ** 3)
    for a in range(len(names)):
        for b in range(len(names)):
            if a == b:
                continue
            i, j = names[a], names[b]
            mu3 += 3.0 * f1[i] * f1[j] * mixed(i, j) * inputs[i].u ** 2 * inputs[j].u ** 2

    # fourth cumulant, leading (diagonal) order: 12 f_i^2 f_ii^2 u_i^6
    kappa4 = sum(12.0 * f1[n] ** 2 * f2[n] ** 2 * inputs[n].u ** 6 for n in names)

    u_linear = math.sqrt(max(var_linear, 0.0))
    u_higher = math.sqrt(var_higher)
    sigma = u_higher if u_higher > 0.0 else u_linear
    skew = mu3 / sigma ** 3 if sigma > 0.0 else 0.0
    exkurt = kappa4 / sigma ** 4 if sigma > 0.0 else 0.0

    # -- intervals -------------------------------------------------------
    p_hi = 1.0 - 0.5 * (1.0 - coverage)
    z = _normal_quantile(p_hi)
    interval_linear = (y0 - z * u_linear, y0 + z * u_linear)
    lo = _cornish_fisher(-z, skew, exkurt)
    hi = _cornish_fisher(z, skew, exkurt)
    centre = y0 + bias
    interval_corrected = (centre + lo * sigma, centre + hi * sigma)

    # -- verdict ---------------------------------------------------------
    tol = _gum_s1_tolerance(max(u_linear, u_higher), digits)
    reasons: List[str] = []
    d_lo = abs(interval_linear[0] - interval_corrected[0])
    d_hi = abs(interval_linear[1] - interval_corrected[1])
    if d_lo > tol or d_hi > tol:
        reasons.append("coverage endpoints move by %.3g / %.3g, tolerance %.3g"
                       % (d_lo, d_hi, tol))
    if abs(bias) > tol:
        reasons.append("second-order bias %+.3g exceeds tolerance %.3g" % (bias, tol))
    if abs(u_higher - u_linear) > tol:
        reasons.append("higher-order terms move u_c by %+.3g (%.2f%%)"
                       % (u_higher - u_linear,
                          100.0 * (u_higher - u_linear) / u_linear if u_linear else 0.0))
    if abs(skew) > 0.1:
        reasons.append("output skewness %+.3g; the interval is asymmetric" % skew)

    flat = u_higher > 0.0 and u_linear <= 1e-12 * u_higher
    if flat:
        reasons.insert(0, "every sensitivity coefficient vanishes at the estimate: "
                          "the linear budget reports zero uncertainty and the whole "
                          "of u_c comes from curvature")

    if not reasons:
        verdict = "LINEAR GUM ADEQUATE"
    elif flat or abs(skew) > 0.5 or (u_linear > 0.0 and abs(u_higher - u_linear) > 0.1 * u_linear):
        verdict = "MONTE CARLO REQUIRED (GUM-S1)"
    else:
        verdict = "HIGHER-ORDER TERMS NEEDED"

    # -- sensitivity cross-check ------------------------------------------
    fd_rel: Dict[str, float] = {}
    if check_sensitivities:
        for n in names:
            fd = _fd_sensitivity(model, values, n)
            if fd is None:
                continue
            scale = max(abs(f1[n]), abs(fd))
            rel = abs(fd - f1[n]) / scale if scale > 0.0 else 0.0
            fd_rel[n] = rel
            if rel > SENSITIVITY_ALARM:
                reasons.append(
                    "sensitivity for %r disagrees with a difference quotient by "
                    "%.2e (exact %.6g, difference %.6g) -- suspect an expressed "
                    "zero multiplying a composite, not the arithmetic" % (n, rel, f1[n], fd))
                verdict = "SENSITIVITY SUSPECT"

    # -- rows ------------------------------------------------------------
    rows: List[Contribution] = []
    for n in names:
        q = inputs[n]
        c = f1[n]
        rows.append(Contribution(
            name=n, value=q.value, u=q.u, shape=q.shape,
            sensitivity=c, second=f2[n], third=f3[n],
            contribution=c * q.u,
            index=100.0 * (c * q.u) ** 2 / var_linear if var_linear > 0 else 0.0,
            bias=0.5 * f2[n] * q.u ** 2
                 + f4[n] * (3.0 + q.shape) * q.u ** 4 / 24.0,
            fd_check=fd_rel.get(n),
        ))
    rows.sort(key=lambda r: r.variance, reverse=True)

    out = Budget(
        value=y0, contributions=rows, u_linear=u_linear, u_higher=u_higher,
        bias=bias, skewness=skew, excess_kurtosis=exkurt, coverage=coverage,
        k=z, interval_linear=interval_linear, interval_corrected=interval_corrected,
        verdict=verdict, reasons=reasons, digits=digits, tolerance=tol,
    )

    if validate:
        out.mc = montecarlo(model, inputs, coverage=coverage,
                            trials=mc_trials, seed=seed)
    return out


# ----------------------------------------------------------------------
# Monte Carlo, for validating the analytic moments (GUM Supplement 1)
# ----------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    trials: int
    mean: float
    u: float
    skewness: float
    excess_kurtosis: float
    interval: Tuple[float, float]          # SHORTEST coverage interval
    coverage: float
    stderr: float = 0.0        # standard error OF THE MEAN estimate
    antithetic: bool = True
    interval_symmetric: Tuple[float, float] = (0.0, 0.0)
    u_stderr: float = 0.0      # standard error of the u estimate itself

    @property
    def shortest(self) -> Tuple[float, float]:
        return self.interval

    def mean_agrees(self, other: float, sigmas: float = 3.0) -> bool:
        """Is ``other`` inside the sampling resolution of this mean?"""
        return abs(other - self.mean) <= sigmas * self.stderr


def _draw_deviation(rng: random.Random, q: Quantity) -> float:
    """A deviation about the estimate.  Every GUM input shape is symmetric, so
    negating a deviation is again a valid draw -- which is what makes the
    antithetic pairing below exact rather than approximate."""
    if q.shape == NORMAL:
        return rng.gauss(0.0, q.u)
    if q.shape == RECTANGULAR:
        a = q.u * math.sqrt(3.0)
        return rng.uniform(-a, a)
    if q.shape == TRIANGULAR:
        a = q.u * math.sqrt(6.0)
        return rng.triangular(-a, a, 0.0)
    if q.shape == ARCSINE:
        a = q.u * math.sqrt(2.0)
        return a * math.cos(math.pi * rng.random())
    raise ValueError("no sampler for shape %r" % (q.shape,))


def montecarlo(model: Callable[..., object], inputs: Dict[str, Quantity],
               coverage: float = 0.95, trials: int = 200000,
               seed: int = 20260922, antithetic: bool = True) -> MonteCarloResult:
    """Propagate distributions by sampling, per GUM Supplement 1 clause 7.

    This is the reference the analytic moments are checked against, and the
    fallback when :func:`budget` returns ``MONTE CARLO REQUIRED``.

    ``antithetic`` pairs every draw with its mirror image about the estimates.
    For a model that is close to linear the linear part then cancels exactly
    between the two members of a pair, so what is left in the pair mean is the
    curvature -- which is precisely the bias term being checked.  It costs
    nothing and it is what makes the mean sharp enough to test against.
    """
    rng = random.Random(seed)
    names = list(inputs)
    ys: List[float] = []
    pair_means: List[float] = []

    def evaluate(draw):
        y = model(**draw)
        return y.st() if isinstance(y, Composite) else float(y)

    if antithetic:
        for _ in range((trials + 1) // 2):
            dev = {n: _draw_deviation(rng, inputs[n]) for n in names}
            a = evaluate({n: inputs[n].value + dev[n] for n in names})
            b = evaluate({n: inputs[n].value - dev[n] for n in names})
            ys.append(a)
            ys.append(b)
            pair_means.append(0.5 * (a + b))
    else:
        for _ in range(trials):
            dev = {n: _draw_deviation(rng, inputs[n]) for n in names}
            ys.append(evaluate({n: inputs[n].value + dev[n] for n in names}))
    n = float(len(ys))
    mean = math.fsum(ys) / n
    m2 = math.fsum((v - mean) ** 2 for v in ys) / n
    m3 = math.fsum((v - mean) ** 3 for v in ys) / n
    m4 = math.fsum((v - mean) ** 4 for v in ys) / n
    sd = math.sqrt(m2)
    ys.sort()
    # GUM-S1 clause 7.7 defines TWO coverage intervals, and they differ once the
    # output is skewed: the shortest one, and the probabilistically symmetric
    # one that cuts equal tails.  budget() produces the symmetric kind, so a
    # comparison of endpoints has to be made against the symmetric kind here.
    alpha = 0.5 * (1.0 - coverage)
    def _pct(f):
        i = min(max(int(round(f * (len(ys) - 1))), 0), len(ys) - 1)
        return ys[i]
    interval_symmetric = (_pct(alpha), _pct(1.0 - alpha))
    q = int(round(coverage * n))
    q = min(max(q, 1), len(ys))
    best = None
    for lo in range(0, len(ys) - q + 1):
        width = ys[lo + q - 1] - ys[lo]
        if best is None or width < best[0]:
            best = (width, ys[lo], ys[lo + q - 1])
    if pair_means:
        npair = float(len(pair_means))
        pm = math.fsum(pair_means) / npair
        pv = math.fsum((v - pm) ** 2 for v in pair_means) / npair
        stderr = math.sqrt(pv / npair) if npair > 1 else 0.0
    else:
        stderr = math.sqrt(m2 / n)
    # Var(s^2) ~ (m4 - m2^2)/n, so the sampled u carries this much resolution.
    # Without it a gap between analytic and sampled u cannot be told apart from
    # the sampling noise, and the reader is invited to over-read either one.
    u_stderr = (math.sqrt(max(m4 - m2 * m2, 0.0) / n) / (2.0 * sd)) if sd > 0 else 0.0
    return MonteCarloResult(
        trials=len(ys), mean=mean, u=sd,
        skewness=m3 / sd ** 3 if sd > 0 else 0.0,
        excess_kurtosis=m4 / sd ** 4 - 3.0 if sd > 0 else 0.0,
        interval=(best[1], best[2]), coverage=coverage,
        stderr=stderr, antithetic=bool(pair_means),
        interval_symmetric=interval_symmetric, u_stderr=u_stderr,
    )
