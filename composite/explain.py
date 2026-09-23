"""explain(f, at) -- what does this function actually do at that point?

The question behind every nan, every inf, and every "it works except near zero".
float64 answers it with nan and no explanation; a plot answers it slowly and
approximately.  One evaluation on a seeded input answers it exactly:

    >>> from composite.composite_lib import sin
    >>> from composite.explain import explain
    >>> print(explain(lambda x: sin(x)/x, 0))
    f at x = 0: value 1; slope 0; numerically stable

What it reports, in the order it decides them:

    nothing here        f is not defined there in any sense -- the value is
                        absent rather than zero.
    blows up            the dominant term is unbounded.  Its ORDER is the order
                        of the pole (1/x, 1/x^2, ...), read off the grade.
    grows logarithmically
                        unbounded, but on the log axis: ln(x) at 0 beats no
                        power of 1/x, which is why it needs its own answer.
    a corner            the leading order is fractional: sqrt(x) leaves the
                        point like x^(1/2), so the slope is infinite and every
                        difference quotient near it is wrong.
    a value and a slope the ordinary case, with the derivative exact.  Where
                        float64 would divide 0 by 0, this is the value to return
                        from the special case, and the slope beside it.
    slope unbounded     the value is finite but the derivative is not, as in
                        x*ln(x) at 0: the slope diverges on the log axis.

and then, for finite cases, what float64 does with the same formula: whether
the spelling loses digits it did not have to (`forensics.audit`).

`f` is written against the library's own functions (`composite_lib.sin`, `cos`,
`exp`, ...).  Anything built from `math.*` silently coerces its argument to a
float and comes back as a plain number, which this reports as `constant, so it
carries no information about x` rather than pretending to have analysed it.
"""

import math

from composite.composite_lib import Composite, R, ZERO

__all__ = ["explain", "Explanation"]


class Explanation:
    """What `explain` found.  `str()` is the sentence; the fields are the facts."""

    __slots__ = ("name", "at", "kind", "value", "slope", "order", "coefficient",
                 "stability", "advice")

    def __init__(self, name, at, kind, value=None, slope=None, order=None,
                 coefficient=None, stability=None, advice=None):
        self.name, self.at, self.kind = name, at, kind
        self.value, self.slope = value, slope
        self.order, self.coefficient = order, coefficient
        self.stability, self.advice = stability, advice

    def __repr__(self):
        return "<Explanation %s at %s: %s>" % (self.name, self.at, self.kind)

    def __str__(self):
        where = "(x-%g)" % self.at if self.at else "x"
        if self.kind == "nothing":
            body = "nothing here -- no value, not even zero"
        elif self.kind == "unbounded":
            n = self.order
            power = where if n == 1 else "%s**%g" % (where, n)
            body = "blows up like %+.6g / %s" % (self.coefficient, power)
        elif self.kind == "corner":
            body = ("value %.12g, but with a corner: it leaves the point like %s**%g, "
                    "so the slope is infinite" % (self.value, where, self.order))
        elif self.kind == "log-value":
            body = ("value %.12g, approached only logarithmically -- it has no ordinary slope here"
                    % self.value)
        elif self.kind == "log-unbounded":
            body = ("grows without bound as x approaches %g, but only logarithmically "
                    "(slower than any power of 1/%s)" % (self.at, where))
        elif self.kind == "steep":
            body = "value %.12g, but the slope is unbounded (it diverges logarithmically)" % self.value
        elif self.kind == "constant":
            body = "constant %.12g, so it carries no information about x (a math.* call?)" % self.value
        elif self.kind == "refused":
            body = "refused: %s" % self.advice
        else:
            body = "value %.12g; slope %.12g" % (self.value, self.slope)
        tail = ""
        if self.stability == "stable":
            tail = "; numerically stable"
        elif self.stability:
            tail = "; numerically %s -- %s" % (self.stability, self.advice)
        return "%s at x = %g: %s%s" % (self.name, self.at, body, tail)


def _log_axis(dim):
    return isinstance(dim, tuple) and dim[0] == 0 and any(dim[1:])


def _stability(f, at):
    """What float64 makes of the same formula here.  Never raises."""
    try:
        from composite.forensics import audit, STABLE
    except Exception:
        return None, None
    probe = at if at else 1e-8
    try:
        a = audit(f, probe)
    except Exception:
        return None, None
    if a.verdict is STABLE:
        return "stable", None
    return str(a.verdict), a.advice


def explain(f, at, name="f"):
    """Describe `f` at `at`: the value or the pole, the slope, and the stability.

    `f` takes one number and returns one number, written against the library's
    functions.  `at` is an ordinary float.
    """
    seed = (R(at) if at else Composite({})) + ZERO
    try:
        y = f(seed)
    except Exception as e:
        return Explanation(name, at, "refused", advice="%s: %s" % (type(e).__name__, e))

    if not isinstance(y, Composite):
        return Explanation(name, at, "constant", value=float(y))

    order = y.lead_order()
    if order is None:
        return Explanation(name, at, "nothing")

    dim = y.lead_dim()
    coeffs = y.coeffs_dict()

    if order < 0:
        if _log_axis(dim):                               # ln(x): unbounded, but slowly
            return Explanation(name, at, "log-unbounded", coefficient=coeffs[dim])
        return Explanation(name, at, "unbounded", order=-order,     # a pole
                           coefficient=coeffs[dim])

    if order != int(order):                              # sqrt-like: a corner
        value = coeffs.get(0, 0.0)
        return Explanation(name, at, "corner", value=value, order=order)

    value = coeffs.get(0, coeffs.get((0, 0), 0.0))
    slope = coeffs.get(-1, coeffs.get((-1, 0), 0.0))
    stability, advice = _stability(f, at)

    # 1/ln(x) reaches 0, but on the log axis and with no ordinary slope: there
    # is no grade -1 term to read, and reporting the missing coefficient as
    # "slope 0" would be a number the function does not have.
    if _log_axis(dim) and -1 not in coeffs and (-1, 0) not in coeffs:
        return Explanation(name, at, "log-value", value=value,
                           stability=stability, advice=advice)

    # A slope that lives on the log axis is not a number: x*ln(x) has value 0
    # at 0 and a derivative that diverges, and reading grade -1 alone reports
    # the slope as 0.0 -- right coefficient, wrong axis.
    if any(isinstance(k, tuple) and k[0] == -1 and any(k[1:]) and v != 0.0
           for k, v in coeffs.items()):
        return Explanation(name, at, "steep", value=value,
                           stability=stability, advice=advice)

    return Explanation(name, at, "value", value=value, slope=slope,
                       stability=stability, advice=advice)
