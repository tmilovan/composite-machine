"""Rich display for notebooks: the grades as a table, not a one-line string.

A composite is a sparse map from grade to coefficient, and `str()` flattens it
to `|9|_0 + |24|_-1 + ...`.  That reading works until the grades stop being a
plain descending run of integers, and then it stops working in three ways at
once: a vector grade on the log axis prints as `1_(0,1)` beside a power-axis
`1_1` with nothing to say they are different axes; a half grade reads as a
typo; and a `denotation_order` marker, which says the jet is NOT the classical
one, does not appear in the string at all.

So this module adds `_repr_html_` to the result types.  Nothing here computes:
every number and every verdict comes from the property that already owns it
(`Audit.verdict`, `Singularity.kind`, `Composite.complete_order`), because a
second implementation of a verdict is a second answer to maintain.

Importing the module installs the hooks.  They are inert outside a rich
display: a terminal and `repr()` are untouched, and `disable()` removes them.

`grade_rows` is the structured form the HTML is built from, exposed because it
is what a GUI binds to and what the tests assert on.
"""
import html
import math

from composite.composite_lib import Composite

__all__ = ["enable", "disable", "grade_rows", "composite_html", "notation"]

_INSTALLED = {}

_TABLE = ("border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,"
          "Menlo,monospace;font-size:12px;margin-top:4px")
_TH = "text-align:left;padding:2px 10px 2px 0;border-bottom:1px solid #999"
_TD = "padding:1px 10px 1px 0;white-space:nowrap"
_NUM = _TD + ";text-align:right"
_BADGE = ("display:inline-block;padding:1px 6px;margin-right:4px;border-radius:3px;"
          "font-size:11px;background:#eee")
_LOUD = _BADGE.replace("background:#eee", "background:#fdd;font-weight:bold")


def _esc(value):
    return html.escape(str(value), quote=True)


def _fmt(value):
    """Full precision, because a rounded coefficient is a different number."""
    try:
        if isinstance(value, complex):
            return "%.17g%+.17gj" % (value.real, value.imag)
        if value == int(value) and abs(value) < 1e16:
            return "%d" % int(value)
        return "%.17g" % value
    except (TypeError, ValueError, OverflowError):
        return _esc(value)


def notation(composite):
    """`<9_0 6_-1 1_-2>`: one number, three terms.

    `__repr__` now follows Zero Rules v2 section 0 itself, so this is a thin
    alias kept for the name: it says at the call site that the braces-and-spaces
    form is the deliberate one, and gives a front end something stable to call
    if the repr is ever specialised per audience.
    """
    return str(composite)


def _axis(grade):
    """(power, log exponents).  Sorting grades needs this: a tuple grade and a
    scalar grade do not compare in Python 3, so `sorted(coeffs)` raises on any
    composite that reached the log axis."""
    if isinstance(grade, tuple):
        return (float(grade[0]), tuple(float(e) for e in grade[1:]))
    return (float(grade), ())


def _is_taylor(grades):
    """True when every grade is a non-positive integer on the power axis, which
    is the only shape where coefficient * n! is the n-th derivative."""
    for grade in grades:
        power, logs = _axis(grade)
        if logs and any(logs):
            return False
        if power > 0 or not float(power).is_integer():
            return False
    return True


def grade_rows(value):
    """[(grade, coefficient, meaning)], leading grade first.

    The meaning column is the point of the table: it says which of the four
    things a grade can be this one is, and flags the two states that change how
    the number must be read -- a denoted order, and an order past the one the
    series is vouched for.
    """
    composite = value if isinstance(value, Composite) else Composite(value)
    coefficients = composite.coeffs_dict()
    complete = composite.complete_order
    denoted = composite.denotation_order
    taylor = _is_taylor(coefficients)
    rows = []
    for grade in sorted(coefficients, key=_axis, reverse=True):
        coefficient = coefficients[grade]
        power, logs = _axis(grade)
        order = -power
        notes = []
        depth = sum(1 for e in logs if e)
        if depth:
            notes.append("log axis, depth %d" % depth)
        if power > 0:
            notes.append("unbounded, order %g" % power)
        elif power == 0:
            # On the log axis at power 0 the log note is the whole story, and
            # the power-axis reading would be the string "order -0 term".
            if not depth:
                notes.append("expressed zero, inert" if coefficient == 0.0 else "the value")
        elif not float(order).is_integer():
            notes.append("branch point, order %g" % order)
        elif taylor:
            n = int(order)
            notes.append("d(%d) = %s" % (n, _fmt(coefficient * math.factorial(n))))
        else:
            notes.append("order %g term" % order)
        if denoted is not None and order >= denoted:
            notes.append("DENOTED, not the classical derivative")
        if complete is not None and order > complete:
            notes.append("past complete_order, not vouched for")
        rows.append((grade, coefficient, "; ".join(notes)))
    return rows


def _badges(composite):
    out = []
    try:
        from composite.backends.config import get_backend
        out.append(type(get_backend()).__name__)
    except Exception:
        pass
    lead = composite.lead_order()
    out.append("lead_order %s" % ("none" if lead is None else _fmt(lead)))
    if composite.complete_order is not None:
        out.append("complete to %s" % _fmt(composite.complete_order))
    leaked = composite.leaked_coeffs()
    if leaked:
        out.append("%d leaked term%s" % (len(leaked), "" if len(leaked) == 1 else "s"))
    loud = []
    if composite.denotation_order is not None:
        loud.append("DENOTED from order %s" % _fmt(composite.denotation_order))
    return out, loud


def _table(headers, rows):
    head = "".join("<th style='%s'>%s</th>" % (_TH, _esc(h)) for h in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            style = _NUM if index and isinstance(cell, (int, float, complex)) else _TD
            text = _fmt(cell) if isinstance(cell, (int, float, complex)) else _esc(cell)
            cells.append("<td style='%s'>%s</td>" % (style, text))
        body.append("<tr>%s</tr>" % "".join(cells))
    return ("<table style='%s'><tr>%s</tr>%s</table>"
            % (_TABLE, head, "".join(body)))


def _header(title, plain, badges, loud=()):
    parts = ["<div style='font-family:ui-monospace,Menlo,monospace;font-size:12px'>"]
    if title:
        parts.append("<div style='color:#666'>%s</div>" % _esc(title))
    if plain:
        parts.append("<div style='font-size:13px;margin:2px 0'><b>%s</b></div>" % _esc(plain))
    chips = ["<span style='%s'>%s</span>" % (_LOUD, _esc(b)) for b in loud]
    chips += ["<span style='%s'>%s</span>" % (_BADGE, _esc(b)) for b in badges]
    if chips:
        parts.append("<div style='margin:3px 0'>%s</div>" % "".join(chips))
    return "".join(parts)


def composite_html(composite):
    rows = grade_rows(composite)
    if not rows:
        return (_header(None, "NOTHING", ["no terms"])
                + "<div style='color:#666'>absence, not zero: there is no term to read</div></div>")
    badges, loud = _badges(composite)
    return (_header(None, notation(composite), badges, loud)
            + _table(["grade", "coefficient", "meaning"], rows) + "</div>")


def explanation_html(explanation):
    facts = [(name, getattr(explanation, name))
             for name in ("kind", "value", "slope", "order", "coefficient",
                          "stability", "advice", "degeneracy")
             if getattr(explanation, name) is not None]
    loud = []
    if explanation.kind == "constant":
        loud.append("NO COMPOSITE REACHED THE RESULT (a math.* call?)")
    if explanation.kind == "refused":
        loud.append("REFUSED")
    if explanation.stability not in (None, "stable"):
        loud.append(str(explanation.stability))
    return (_header("explain %s at %s" % (explanation.name, _fmt(explanation.at)),
                    str(explanation), [], loud)
            + _table(["field", "value"], facts) + "</div>")


def audit_html(audit):
    loud = []
    if not audit.instrumented:
        loud.append("NOT INSTRUMENTED: no composite reached the result, so nothing was measured")
    if audit.exception is not None:
        loud.append("raised %s" % type(audit.exception).__name__)
    verdict = audit.verdict
    if verdict is not None and getattr(verdict, "name", str(verdict)) != "STABLE":
        loud.append(str(getattr(verdict, "name", verdict)))
    facts = [("value", audit.value), ("f'(at)", audit.derivative),
             ("kappa", audit.kappa), ("floor", audit.floor),
             ("predicted", audit.predicted), ("digits lost", audit.digits_lost),
             ("verdict", getattr(verdict, "name", verdict)), ("advice", audit.advice)]
    parts = [_header("audit %s at %s" % (audit.name, _fmt(audit.at)), None, [], loud),
             _table(["field", "value"], facts)]
    if audit.findings:
        parts.append("<div style='color:#666;margin-top:6px'>%d finding%s, worst first</div>"
                     % (len(audit.findings), "" if len(audit.findings) == 1 else "s"))
        parts.append(_table(["kind", "op", "amp", "digits", "where", "source"],
                            [(f.kind, f.op, f.amp, f.digits, f.where, f.src)
                             for f in audit.findings]))
    return "".join(parts) + "</div>"


def singularity_html(singularity):
    facts = [("location z0", singularity.location),
             ("exponent beta", singularity.exponent),
             ("kind", singularity.kind),
             ("critical exponent gamma", singularity.critical_exponent),
             ("blowup rate alpha", singularity.blowup_rate),
             ("pole order", singularity.pole_order),
             ("spread", singularity.spread),
             ("orders agreeing", singularity.orders),
             ("terms used", singularity.terms_used),
             ("Pade cross-check", singularity.cross_check)]
    badges = ["%d orders agreed" % singularity.orders]
    loud = [] if singularity.cross_check is not None else ["no Pade cross-check"]
    return (_header("nearest singularity", singularity.kind, badges, loud)
            + _table(["field", "value"], facts) + "</div>")


def budget_html(budget):
    facts = [("value", budget.value), ("u linear", budget.u_linear),
             ("u higher order", budget.u_higher), ("bias", budget.bias),
             ("skewness", budget.skewness), ("excess kurtosis", budget.excess_kurtosis),
             ("coverage", budget.coverage), ("k", budget.k),
             ("interval (linear)", budget.interval_linear),
             ("interval (corrected)", budget.interval_corrected)]
    loud = [] if budget.verdict == "admissible" else [str(budget.verdict)]
    parts = [_header("uncertainty budget", budget.verdict, [], loud),
             _table(["field", "value"], facts)]
    if budget.reasons:
        parts.append("<ul style='margin:4px 0;font-size:12px'>%s</ul>"
                     % "".join("<li>%s</li>" % _esc(r) for r in budget.reasons))
    if budget.contributions:
        parts.append(_table(["input", "value", "u", "sensitivity", "contribution",
                             "index %", "bias"],
                            [(c.name, c.value, c.u, c.sensitivity, c.contribution,
                              c.index, c.bias) for c in budget.contributions]))
    return "".join(parts) + "</div>"


def transseries_html(transseries):
    # Key on the sector alone.  Keying on the whole item put the Composite
    # through _axis, which read it as a log exponent and called float() on it --
    # and float() of an unbounded composite raises.
    rows = [(sector, notation(series), "exp(-%s/h)" % sector if sector else "perturbative")
            for sector, series in sorted(transseries.sectors.items(),
                                         key=lambda item: _axis(item[0]))]
    return (_header("transseries", None, ["%d sector%s" % (len(rows), "" if len(rows) == 1 else "s")])
            + _table(["sector", "series", "weight"], rows) + "</div>")


def _guard(render):
    """A repr that raises is worse than no repr: Jupyter shows the traceback
    instead of the object.  Fall back to the plain string."""
    def hook(self):
        try:
            return render(self)
        except Exception as why:                       # pragma: no cover
            return ("<pre style='font-size:12px'>%s</pre>"
                    "<div style='color:#a00;font-size:11px'>rich display failed: "
                    "%s: %s</div>" % (_esc(self), type(why).__name__, _esc(why)))
    return hook


def _targets():
    """(class, renderer) for every type that has one, skipping the optional ones
    that are not importable in this install."""
    pairs = [(Composite, composite_html)]
    for module, name, render in (
            ("composite.explain", "Explanation", explanation_html),
            ("composite.forensics", "Audit", audit_html),
            ("composite.singularity", "Singularity", singularity_html),
            ("composite.uncertainty", "Budget", budget_html),
            ("composite.transseries", "Transseries", transseries_html)):
        try:
            pairs.append((getattr(__import__(module, fromlist=[name]), name), render))
        except Exception:                              # pragma: no cover
            pass
    return pairs


def enable():
    """Install `_repr_html_` on the result types.  Returns the names touched."""
    touched = []
    for cls, render in _targets():
        if cls not in _INSTALLED:
            _INSTALLED[cls] = cls.__dict__.get("_repr_html_")
            cls._repr_html_ = _guard(render)
            touched.append(cls.__name__)
    return touched


def disable():
    """Remove them, restoring anything that was there before."""
    for cls, previous in list(_INSTALLED.items()):
        if previous is None:
            try:
                del cls._repr_html_
            except AttributeError:                     # pragma: no cover
                pass
        else:
            cls._repr_html_ = previous
        del _INSTALLED[cls]


enable()
