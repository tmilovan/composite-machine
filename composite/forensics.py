"""Cancellation forensics: where a formula loses significance, and whose fault it is.

Every numerical codebase has expressions that quietly destroy precision, and
float64 gives no indication when it happens -- `1 - cos(x)` at x = 1e-5 returns
a number with eight correct digits and no complaint.

The hard part is not spotting the loss.  It is answering the question that
follows: *is the formula bad, or is the problem hard?*  Those need opposite
responses -- rewrite the expression, or accept the error and stop trying -- and
telling them apart needs two numbers.

  kappa      how much the *problem* amplifies error in its input.
             |x f'(x) / f(x)|, the relative condition number.  No formula,
             however careful, does better than kappa * eps.

  predicted  how much error the *formula as written* accumulates on its own.
             A first-order forward bound, propagated operation by operation
             from an exact input, so it measures what the code adds rather
             than what it inherits.

predicted >> kappa * eps  ->  the formula is adding loss the problem does not
                              require.  Rewrite it; this module names the line.
kappa large               ->  the problem is hard.  Rewriting will not help.

The composite supplies both.  Seeding the input with an infinitesimal gives
f'(x) from the same pass that computes f(x) -- no step size, no difference
quotient -- and that is what kappa needs.  It matters more than it looks: a
finite difference *is* a subtraction of nearly equal numbers, so the usual way
to estimate conditioning fails hardest on exactly the formulas worth
diagnosing.  On (1 - cos x)/x^2 at x = 1e-5, where the true kappa is 1.67e-11,
central differences report 1.8e-06 to 1.6e-03.  The same seeding supplies the
elementary-function derivatives the error bound needs at each node.

Three faults are recorded, because they have different fixes:

  cancellation  a +- b where the operands nearly agree; the leading digits
                agree and are lost, leaving the result carrying the noise.
  absorption    a +- b where one operand is negligible beside the other; the
                small one's low bits are discarded.  Its amplification is 1,
                so cancellation metrics miss it -- this is why `ln(1 + x)` at
                small x looks innocent and is not.
  data-zero     an expressed zero entering an additive position.  R1 converts
                it to an infinitesimal, so the value comes through untouched
                while the derivative moves.  Invisible in float64, where the
                number is the same either way, and it silently invalidates
                any sensitivity read off the result.

The bound covers *rounding*, not approximation.  A formula that truncates a
series -- three terms of exp(x) - 1, say -- is bounded correctly for the
arithmetic it performs and says nothing about the terms it left out, so its
observed error against the true function can exceed `predicted` by any amount.
Audit exact rewrites (Kahan's expm1, 2 sin^2(x/2)/x^2) rather than truncations,
or read the bound as applying to the series actually written.

Two further caveats, stated plainly.  kappa is the exact derivative of the *program as
written*, float64 rounding included, not of the mathematical function; on a
badly unstable formula it is computed from polluted intermediates and loses
digits itself.  It stays orders of magnitude closer than a difference quotient
and the verdict survives, but the digits of kappa do not.  And `predicted` is a
first-order bound: it is expected to sit above the observed error, often by one
to two orders.  The verdict is the claim, not the digits.
"""

import linecache
import math
import os
import sys

from .composite_lib import (
    Composite, R, NotRepresentableError, StandardPartUndefinedError,
)
from . import composite_lib as _cl

EPS = 2.220446049250313e-16

__all__ = [
    "audit", "compare", "table", "report", "Audit", "Finding", "F", "EPS",
    "STABLE", "ILL_CONDITIONED", "UNSTABLE", "DERIVATIVE_LOST", "REFUSED",
]

STABLE = "stable"
ILL_CONDITIONED = "ill-conditioned problem"
UNSTABLE = "unstable formula"
DERIVATIVE_LOST = "derivative corrupted"
REFUSED = "refused"

_CANCEL = 100.0     # (|a|+|b|)/|a+-b| worth reporting
_ABSORB = 1e-8      # min/max operand ratio at which the small one is being eaten
_BLAME = 10.0       # predicted must exceed _BLAME * floor to blame the formula
_HARD = 1e8         # kappa above this is a hard problem in its own right


class Finding:
    """One operation that lost information, and the line it was written on."""

    __slots__ = ("op", "a", "b", "result", "amp", "file", "line", "src", "kind")

    def __init__(self, op, a, b, result, amp, file, line, src, kind):
        self.op, self.a, self.b, self.result = op, a, b, result
        self.amp, self.file, self.line, self.src, self.kind = amp, file, line, src, kind

    @property
    def digits(self):
        return math.log10(self.amp) if self.amp > 1 else 0.0

    @property
    def where(self):
        return "%s:%d" % (os.path.basename(self.file), self.line)

    def __repr__(self):
        return "<Finding %s %s amp=%.3g at %s>" % (self.kind, self.op, self.amp, self.where)


class Audit:
    """The result of auditing one formula at one input."""

    def __init__(self, name, at, value, derivative, kappa, error_bound,
                 findings, exception=None):
        self.name = name
        self.at = at
        self.value = value              # standard part
        self.derivative = derivative    # f'(at), exact for the program as written
        self.kappa = kappa
        self.error_bound = error_bound  # absolute, first-order, from an exact input
        self.findings = sorted(findings, key=lambda f: -f.amp)
        self.exception = exception

    # -- the two numbers ----------------------------------------------------

    @property
    def floor(self):
        """Best relative error achievable on this problem, by any formula."""
        k = self.kappa
        return EPS if math.isnan(k) else max(k, 1.0) * EPS

    @property
    def predicted(self):
        """Relative error this formula is expected to stay under."""
        if math.isnan(self.value):
            return float("nan")
        if self.value == 0.0:
            # Everything cancelled.  If the bound admits any error at all the
            # true value need not be zero, so the relative error is unbounded.
            # Reporting nan here once let total annihilation fall through to
            # the mildest verdict instead of the harshest.
            return float("inf") if self.error_bound > 0.0 else EPS
        return max(self.error_bound / abs(self.value), EPS)

    @property
    def digits_lost(self):
        """Decimal digits this formula gives up beyond machine precision."""
        p = self.predicted
        return 0.0 if math.isnan(p) else math.log10(max(p / EPS, 1.0))

    # -- attribution --------------------------------------------------------

    def of_kind(self, kind):
        return [f for f in self.findings if f.kind == kind]

    @property
    def zeros(self):
        return self.of_kind("data-zero")

    @property
    def growth(self):
        """Worst single cancellation amplification seen."""
        return max([f.amp for f in self.findings if f.kind == "cancellation"],
                   default=1.0)

    @property
    def culprit(self):
        """The finding most likely responsible, or None."""
        for kind in ("cancellation", "absorption"):
            got = self.of_kind(kind)
            if got:
                return got[0]
        return None

    # -- the verdict --------------------------------------------------------

    @property
    def verdict(self):
        if self.exception is not None:
            return REFUSED
        if self.value == 0.0 and self.error_bound > 0.0:
            # Defensive and, as it stands, redundant: `predicted` already
            # returns inf here and kappa is nan, so `floor` falls back to EPS
            # and the generic test below fires anyway.  Mutation testing
            # confirms removing this branch changes no result.  Kept because
            # it states the intent at the point where the intent applies.
            return UNSTABLE          # everything cancelled; no digits survive
        p = self.predicted
        if not math.isnan(p) and p > _BLAME * max(self.floor, EPS):
            return UNSTABLE
        if not math.isnan(self.kappa) and self.kappa > _HARD:
            return ILL_CONDITIONED
        if self.zeros:
            return DERIVATIVE_LOST
        return STABLE

    def _cost(self):
        """Short form, for a report line."""
        d = self.digits_lost
        return "no correct digits" if math.isinf(d) else "costs %.1f digits" % d

    def _cost_clause(self):
        """Full clause, for a sentence of advice."""
        d = self.digits_lost
        return ("leaves no correct digits" if math.isinf(d)
                else "costs %.1f digits the problem does not require" % d)

    @property
    def advice(self):
        v = self.verdict
        if v == UNSTABLE:
            c = self.culprit
            if c is None:
                return "rewrite -- the formula %s" % self._cost_clause()
            what = {"cancellation": "cancellation in `%s`" % c.op,
                    "absorption": "absorption in `%s`" % c.op}[c.kind]
            return "rewrite -- %s at %s %s" % (what, c.where, self._cost_clause())
        if v == DERIVATIVE_LOST:
            w = self.zeros[0]
            return ("value is right, sensitivity is not -- an expressed zero at %s "
                    "converts to an infinitesimal and shifts the derivative; drop "
                    "the term rather than adding a literal 0.0" % w.where)
        if v == ILL_CONDITIONED:
            return ("accept -- the problem amplifies input error %.3g x; no formula "
                    "does better than %.1e" % (self.kappa, self.floor))
        if v == REFUSED:
            return "refused: %s: %s" % (type(self.exception).__name__, self.exception)
        return "no action -- loses no more than the problem requires"

    def __repr__(self):
        return "<Audit %s at %g: %s>" % (self.name, self.at, self.verdict)


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

_ledger = None
_depth = 0
_symbols = [0]

#  Error is carried as an affine form: a map from an independent noise symbol
#  to its coefficient, rather than a single magnitude.  Every rounding
#  introduces a fresh symbol and every operation propagates the existing ones
#  linearly, with their signs.
#
#  Signs are the whole point.  A scalar magnitude has to assume the worst of
#  each operation independently, which double-counts any error that appears
#  twice and then cancels.  That is exactly what a compensated algorithm
#  arranges: in Kahan's `x (u-1) / ln(u)`, the rounding of u enters the
#  numerator and the denominator with the same coefficient and divides out.
#  A magnitude bound calls that formula unstable; the affine form watches the
#  coefficient go to zero and does not.

_MAX_SYMBOLS = 256


def _fresh(mag):
    """One new independent rounding, of the given magnitude."""
    if not mag or math.isnan(mag) or math.isinf(mag):
        return {} if not mag else {"overflow": float("inf")}
    _symbols[0] += 1
    return {_symbols[0]: mag}


def _lin(*terms):
    """Linear combination of affine forms: sum of coefficient * form."""
    out = {}
    for coef, vec in terms:
        if not vec or coef == 0.0:
            continue
        if math.isnan(coef) or math.isinf(coef):
            return {"overflow": float("inf")}
        for k, v in vec.items():
            nv = out.get(k, 0.0) + coef * v
            if nv == 0.0:
                out.pop(k, None)
            else:
                out[k] = nv
    if len(out) > _MAX_SYMBOLS:
        # Keep the dominant terms exactly and pool the rest conservatively;
        # pooled symbols can no longer cancel, which only loosens the bound.
        items = sorted(out.items(), key=lambda kv: -abs(kv[1]))
        head, tail = items[:_MAX_SYMBOLS - 1], items[_MAX_SYMBOLS - 1:]
        out = dict(head)
        out.update(_fresh(sum(abs(v) for _, v in tail)))
    return out


def _bound(vec):
    """Worst case magnitude of an affine form."""
    return sum(abs(v) for v in vec.values())


def _mag(v):
    """Magnitude of the standard part, or None when there isn't one."""
    sv = _sval(v)
    return None if sv is None or math.isnan(sv) else abs(sv)


def _sval(v):
    """Signed standard part, or None when there isn't one."""
    try:
        if isinstance(v, Composite):
            return float(v.st())
        return float(v)
    except (StandardPartUndefinedError, TypeError, ValueError, OverflowError):
        return None


def _err(v):
    """Affine error form carried by a value; literals carry none."""
    return getattr(v, "_err", None) or {}


def _site():
    """The first frame outside this module and the library: the caller's line."""
    here = os.path.dirname(os.path.abspath(__file__))
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if os.path.dirname(os.path.abspath(fn)) != here:
            return fn, f.f_lineno, linecache.getline(fn, f.f_lineno).strip()
        f = f.f_back
    return "<unknown>", 0, ""


def _note(op, ma, mb, mr, amp, kind):
    if _ledger is None:
        return
    fn, line, src = _site()
    _ledger.append(Finding(op, ma, mb, mr, amp, fn, line, src, kind))


def _classify_additive(op, a, b, result):
    """Score one additive step, and name the fault if there is one."""
    if _ledger is None:
        return
    ma, mb, mr = _mag(a), _mag(b), _mag(result)
    if ma is None or mb is None or mr is None:
        return
    if (ma == 0.0) != (mb == 0.0):
        # Only a zero that entered from OUTSIDE the computation is a data zero.
        # An intermediate may pass through zero legitimately and still carry a
        # perfectly good infinitesimal -- ln(S/K) at S = K is 0 with derivative
        # 1/S -- and blaming that was a false positive on every formula with a
        # zero crossing.  A literal, or a Composite the audit never touched,
        # has no dependence on the variable and is the real case.
        zero_operand = b if mb == 0.0 else a
        if not isinstance(zero_operand, _Audited):
            _note(op, ma, mb, mr, 1.0, "data-zero")
        return
    if ma == 0.0 and mb == 0.0:
        return
    if mr == 0.0:
        _note(op, ma, mb, mr, float("inf"), "cancellation")
        return
    amp = (ma + mb) / mr
    if amp >= _CANCEL:
        _note(op, ma, mb, mr, amp, "cancellation")
        return
    lo, hi = (ma, mb) if ma < mb else (mb, ma)
    if hi > 0.0 and lo / hi <= _ABSORB:
        # The small operand's low bits are discarded.  Amplification is ~1, so
        # a cancellation metric sees nothing -- this is the ln(1+x) failure.
        _note(op, ma, mb, mr, hi / lo, "absorption")


class _Audited(Composite):
    """A Composite that carries an affine error form and records its faults.

    The re-wrap follows `TracedComposite._as_traced` in composite_lib: copy the
    three slots rather than assigning `.c`, which is a read-only property, and
    do not name the method `_wrap`, which would shadow the classmethod.
    """

    __slots__ = ("_err",)

    def _rewrap(self, r, err=None):
        if isinstance(r, Composite):
            t = _Audited.__new__(_Audited)
            t._backend = r._backend
            t._data = r._data
            t._complete = r._complete
            t._err = err or {}
            return t
        return r


def _binary(name, symbol, err_rule, additive, reverse=False):
    """Build one operator: run it, propagate its error form, record its fault."""

    def method(self, other):
        global _depth
        parent = getattr(Composite, name)
        if _depth:                      # library internals, not the caller's line
            return self._rewrap(parent(self, other), _err(self))
        _depth += 1
        try:
            r = parent(self, other)
        finally:
            _depth -= 1
        if r is NotImplemented:
            return r
        a, b = (other, self) if reverse else (self, other)
        if additive:
            _classify_additive(symbol, a, b, r)
        sa, sb, sr = _sval(a), _sval(b), _sval(r)
        if None in (sa, sb, sr):
            e = {"overflow": float("inf")}
        else:
            e = _lin(*err_rule(sa, sb, sr, _err(a), _err(b)))
            e.update(_fresh(EPS * abs(sr)))
        return self._rewrap(r, e)

    method.__name__ = name
    return method


def _pow_method(self, n):
    global _depth
    _depth += 1
    try:
        r = Composite.__pow__(self, n)
    finally:
        _depth -= 1
    sa, sr = _sval(self), _sval(r)
    if sa is None or sr is None or sa == 0.0:
        e = {} if sa == 0.0 else {"overflow": float("inf")}
    else:
        e = _lin((float(n) * sr / sa, _err(self)))      # n a^(n-1) = n r / a
    if sr is not None and not math.isnan(sr):
        e.update(_fresh(EPS * abs(sr)))
    return self._rewrap(r, e)


#  d(a +- b) = da +- db          d(ab) = a db + b da
#  d(a/b)    = da/b - (a/b)(db/b)
_ADD = lambda a, b, r, ea, eb: ((1.0, ea), (1.0, eb))
_SUB = lambda a, b, r, ea, eb: ((1.0, ea), (-1.0, eb))
_MUL = lambda a, b, r, ea, eb: ((b, ea), (a, eb))
_DIV = lambda a, b, r, ea, eb: (((1.0 / b) if b else float("inf"), ea),
                                ((-r / b) if b else float("inf"), eb))

for _n, _s, _rule, _add, _rev in (
        ("__add__", "+", _ADD, True, False),
        ("__radd__", "+", _ADD, True, True),
        ("__sub__", "-", _SUB, True, False),
        ("__rsub__", "-", _SUB, True, True),
        ("__mul__", "*", _MUL, False, False),
        ("__rmul__", "*", _MUL, False, True),
        ("__truediv__", "/", _DIV, False, False),
        # reverse=True hands the rule (numerator, denominator) already in
        # order, so reverse division takes the same rule as forward division.
        ("__rtruediv__", "/", _DIV, False, True)):
    setattr(_Audited, _n, _binary(_n, _s, _rule, _add, _rev))
_Audited.__pow__ = _pow_method
del _n, _s, _rule, _add, _rev


# --------------------------------------------------------------------------
# the function namespace the formula is written against
# --------------------------------------------------------------------------

class _Namespace:
    """Elementary functions that carry the audit through.

    The library's `cos` and friends return a plain Composite, so the wrapper
    would be dropped at the first call and every later step would go
    unrecorded.  Re-wrapping here restores it.  The same boundary suppresses
    the library's internal arithmetic, which is what keeps findings pointing at
    the caller's line rather than at a Taylor loop.

    The error rule needs |f'(a)| at the node.  That is one extra evaluation on
    a scalar seeded with an infinitesimal -- exact, and no step size.
    """

    _NAMES = ("sin", "cos", "tan", "exp", "ln", "sqrt", "sinh", "cosh", "tanh",
              "atan", "asin", "acos", "erf", "erfc")

    def __getattr__(self, name):
        if name not in self._NAMES:
            raise AttributeError(name)
        fn = getattr(_cl, name)

        def wrapped(v):
            global _depth
            _depth += 1
            try:
                arg = v if isinstance(v, Composite) else R(v)
                r = fn(arg)
                slope = _slope(fn, _sval(arg))
            finally:
                _depth -= 1
            sr = _sval(r)
            if sr is None or slope is None or math.isnan(slope):
                e = {"overflow": float("inf")}
            else:
                e = _lin((slope, _err(v)))
                e.update(_fresh(EPS * abs(sr)))
            if isinstance(r, Composite):
                t = _Audited.__new__(_Audited)
                t._backend, t._data, t._complete, t._err = \
                    r._backend, r._data, r._complete, e
                return t
            return r

        wrapped.__name__ = name
        return wrapped

    def log(self, v):
        return self.ln(v)


def _slope(fn, a):
    """f'(a), from one pass over a seeded scalar."""
    if a is None or math.isnan(a):
        return float("nan")
    try:
        return float(fn(Composite({0: a, -1: 1.0})).d(1))
    except Exception:
        return float("nan")


F = _Namespace()


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def audit(f, at, name=None):
    """Audit `f` at the point `at`.

    `f` takes one argument and returns a number.  Write it against `F` for
    elementary functions (`F.cos`, `F.exp`, ...) so the audit survives them.

    Returns an `Audit`: the value, the exact derivative, the problem's
    condition number, a forward error bound for the formula, every operation
    that lost information with the line it was written on, and a verdict
    separating a bad formula from a hard problem.
    """
    global _ledger, _depth
    name = name or getattr(f, "__name__", "formula")
    at = float(at)

    seeded = _Audited.__new__(_Audited)
    _base = Composite({0: at, -1: 1.0})     # at + h, one infinitesimal
    seeded._backend, seeded._data, seeded._complete = \
        _base._backend, _base._data, _base._complete
    seeded._err = {}                         # the input is taken as exact

    _ledger, _depth = [], 0
    _symbols[0] = 0
    exc = None
    try:
        out = f(seeded)
    except (NotRepresentableError, StandardPartUndefinedError, ZeroDivisionError,
            ArithmeticError, ValueError) as e:
        out, exc = None, e
    findings, _ledger = _ledger, None

    value = derivative = kappa = float("nan")
    bound = float("nan")
    if out is not None:
        if not isinstance(out, Composite):
            out = R(float(out))
        bound = _bound(_err(out))
        try:
            value = float(out.st())
        except StandardPartUndefinedError as e:
            exc = exc or e
        try:
            derivative = float(out.d(1))
        except Exception:
            derivative = 0.0
        if value == 0.0:
            kappa = float("nan")     # undefined: there is no relative scale left
        elif not math.isnan(value):
            kappa = abs(at * derivative / value)
    return Audit(name, at, value, derivative, kappa, bound, findings, exc)


def compare(variants, at, reference=None):
    """Audit several spellings of the same function at one point.

    `variants` maps a label to a callable.  When `reference` is given -- any
    callable returning the true value, mpmath say -- the observed relative
    error is reported beside the predicted bound, which is what makes the
    prediction a check rather than a claim.

    A reference must be evaluated at the *float* `at`, not at a decimal
    re-reading of it: `mp.mpf(x)` is exact, `mp.mpf(repr(x))` is a different
    point and silently reports the gap between them as formula error.
    """
    rows = []
    for label, fn in variants.items():
        a = audit(fn, at, name=label)
        obs = None
        if reference is not None:
            try:
                t = float(reference(at))
                obs = abs(a.value - t) / abs(t) if t != 0.0 else abs(a.value)
            except Exception:
                obs = None
        rows.append((a, obs))
    return rows


def _fmt(x, w=9):
    if x is None:
        return "-".rjust(w)
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return ("inf" if math.isinf(x) else "nan").rjust(w)
    return ("%.2e" % x).rjust(w)


def table(rows, title=None):
    """Render `compare` output."""
    out = ([title] if title else []) + [
        "  %-26s %9s %9s %9s %9s  %s"
        % ("formula", "kappa", "floor", "predicted", "observed", "verdict"),
        "  " + "-" * 92]
    for a, obs in rows:
        out.append("  %-26s %s %s %s %s  %s"
                   % (a.name[:26], _fmt(a.kappa), _fmt(a.floor),
                      _fmt(a.predicted), _fmt(obs), a.verdict))
    return "\n".join(out)


def report(a, top=3):
    """A full account of one audit, including the offending lines."""
    out = ["  %s at x = %g" % (a.name, a.at),
           "    value          %.17g" % a.value,
           "    f'(x)          %.17g   (exact, one pass, no step size)" % a.derivative,
           "    kappa          %.3g   problem conditioning" % a.kappa,
           "    floor          %.2e   best any formula can do here" % a.floor,
           "    predicted      %.2e   bound for this formula (%s)"
           % (a.predicted, a._cost()),
           "    VERDICT        %s" % a.verdict,
           "    %s" % a.advice]
    shown = [f for f in a.findings if f.kind in ("cancellation", "data-zero")] \
        or a.findings
    if shown:
        out.append("    information lost at:")
        for f in shown[:top]:
            if f.kind == "data-zero":
                out.append("      %s  %s  expressed zero in an additive position"
                           % (f.where, f.op))
            elif f.kind == "absorption":
                out.append("      %s  %s  absorption, operands differ by %.3g x"
                           % (f.where, f.op, f.amp))
            else:
                out.append("      %s  %s  cancellation, amp %.3g (%.1f digits)"
                           % (f.where, f.op, f.amp, f.digits))
            if f.src:
                out.append("          %s" % f.src)
            if f.kind != "data-zero":
                out.append("          |a|=%.6g  |b|=%.6g  ->  |result|=%.6g"
                           % (f.a, f.b, f.result))
    return "\n".join(out)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def _namespace():
    """What an expression on the command line may refer to."""
    ns = {n: getattr(F, n) for n in _Namespace._NAMES}
    ns["log"] = F.ln
    ns.update(pi=math.pi, e=math.e, abs=abs, __builtins__={})
    return ns


def evaluate(expr, var="x"):
    """Turn an expression string into a callable of one variable."""
    code = compile(expr, "<expr>", "eval")
    # Register the text so findings can quote the line they came from; eval'd
    # code has no file for linecache to read.
    linecache.cache["<expr>"] = (len(expr), None, [expr], "<expr>")
    ns = _namespace()

    def f(value):
        return eval(code, dict(ns), {var: value})

    f.__name__ = expr
    return f


def sweep(f, lo, hi, points=25, name=None):
    """Audit across a logarithmic range and find where the formula gives way."""
    if lo <= 0 or hi <= 0:
        xs = [lo + (hi - lo) * i / (points - 1.0) for i in range(points)]
    else:
        a, b = math.log10(lo), math.log10(hi)
        xs = [10 ** (a + (b - a) * i / (points - 1.0)) for i in range(points)]
    return [(x, audit(f, x, name=name)) for x in xs]


def _breakdown(rows):
    """Describe where the formula gives way, in whichever direction it does.

    A sweep runs low to high, but instability sits at the low end as often as
    the high one, so reporting only "the first non-stable point" describes the
    x -> 0 cases backwards.  Return the transitions instead and let the caller
    phrase it.
    """
    verdicts = [(x, a.verdict == STABLE) for x, a in rows]
    if all(ok for _, ok in verdicts):
        return "all-stable", None, None
    if not any(ok for _, ok in verdicts):
        return "none-stable", None, None
    crossings = [(verdicts[i][0], verdicts[i + 1][0])
                 for i in range(len(verdicts) - 1)
                 if verdicts[i][1] != verdicts[i + 1][1]]
    first = crossings[0]
    stable_high = verdicts[-1][1]
    return ("gives-way-below" if stable_high else "gives-way-above",
            first, len(crossings))


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m composite.forensics",
        description="Where a formula loses significance, and whether it had to.",
        epilog="example:  python -m composite.forensics '(1-cos(x))/x**2' "
               "--against '2*sin(x/2)**2/x**2' --sweep 1e-8 1")
    p.add_argument("expr", help="expression in one variable, e.g. '(1-cos(x))/x**2'")
    p.add_argument("--at", type=float, help="audit at this single point")
    p.add_argument("--against", help="a second spelling of the same function")
    p.add_argument("--sweep", nargs=2, type=float, metavar=("LO", "HI"),
                   help="audit across a range and report where it gives way")
    p.add_argument("--points", type=int, default=25, help="points in the sweep")
    p.add_argument("--var", default="x", help="name of the variable (default x)")
    args = p.parse_args(argv)

    try:
        f = evaluate(args.expr, args.var)
    except SyntaxError as e:
        print("cannot parse %r: %s" % (args.expr, e))
        return 2
    g = evaluate(args.against, args.var) if args.against else None

    if args.sweep:
        lo, hi = args.sweep
        rows = sweep(f, lo, hi, args.points, name=args.expr)
        print("  %-12s %9s %9s %9s  %s"
              % (args.var, "kappa", "floor", "predicted", "verdict"))
        print("  " + "-" * 74)
        for x, a in rows:
            print("  %-12.4g %s %s %s  %s"
                  % (x, _fmt(a.kappa), _fmt(a.floor), _fmt(a.predicted), a.verdict))
        kind, cross, n = _breakdown(rows)
        print()
        if kind == "all-stable":
            print("  stable across the whole range.")
        elif kind == "none-stable":
            print("  gives way across the whole range; widen it to find the boundary.")
            print("  %s" % rows[-1][1].advice)
        else:
            lo, hi = cross
            side = "below" if kind == "gives-way-below" else "above"
            print("  gives way %s %s = %.4g  (still sound at %.4g)"
                  % (side, args.var, hi if side == "above" else lo,
                     lo if side == "above" else hi))
            if n > 1:
                print("  %d transitions in this range; the first is shown." % n)
            bad = next(a for x, a in rows if a.verdict != STABLE)
            print("  %s" % bad.advice)
        return 0

    at = args.at if args.at is not None else 1.0
    if g is not None:
        rows = compare({args.expr: f, args.against: g}, at)
        print(table(rows, "  at %s = %g" % (args.var, at)))
        best = min((r for r in rows if not math.isnan(r[0].predicted)),
                   key=lambda r: r[0].predicted, default=None)
        if best is not None:
            print()
            print("  prefer:  %s   (%.1f digits better)"
                  % (best[0].name,
                     max(0.0, max(r[0].digits_lost for r in rows) - best[0].digits_lost)))
        return 0

    a = audit(f, at, name=args.expr)
    print()
    print(report(a))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
