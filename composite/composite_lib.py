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
composite_lib.py — Unified Calculus Library (Fixed v3: Expressed Zero Preservation)
====================================================================================
All operations use composite arithmetic. No plain-number fast paths
in transcendental functions. Integration accumulates as Composite.

v3 changes:
  - Composite(0) now produces |0|₀ (expressed zero at dim 0), not empty
  - Dict construction preserves zero-valued coefficients
  - Empty composite repr changed from |0|₀ to ∅

Usage:
    from composite.composite_lib import *

    # Derivatives
    derivative(lambda x: x**2, at=3)           # → 6
    derivative(lambda x: sin(x), at=0)         # → 1
    nth_derivative(lambda x: x**5, n=3, at=2)  # → 120

    # Limits
    limit(lambda x: sin(x)/x, as_x_to=0)                    # → 1
    limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)          # → 4
    limit(lambda x: (1 - cos(x))/x**2, as_x_to=0)           # → 0.5

    # All derivatives at once
    all_derivatives(lambda x: exp(x), at=0, up_to=5)  # → [1,1,1,1,1,1]

    # Direct composite computation
    h = ZERO
    x = R(3) + h
    result = x**2
    print(result)        # |9|₀ + |6|₋₁ + |1|₋₂
    print(result.st())   # 9 (function value)
    print(result.d(1))   # 6 (first derivative)
    print(result.d(2))   # 2 (second derivative)

Author: Toni Milovan
"""

import math
import contextlib as _contextlib
import contextvars as _contextvars
import warnings as _warnings
import functools as _functools
from fractions import Fraction as _Fraction
from typing import Callable, List, Optional, Union
import struct
import numpy as np

from composite.backends import get_backend
from composite.backends.base_backend import (DIM_DTYPE, dim_cast,
                                             dim_fraction)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class NotRepresentableError(ValueError):
    """The quantity exists but has no composite for it.

    Distinct from LimitDoesNotExistError, which asserts something about a
    LIMIT.  sin(1/h) has no representation here -- bounded by 1, no limit, not
    eventually monotone, so outside a Hardy field at any basis extension --
    but that says nothing about an expression CONTAINING it: x*sin(1/x) tends
    to 0 perfectly well.  Raising the stronger error stopped limit() from
    recovering those, because it re-raises "provably does not exist" without
    trying.  A ValueError subclass, so the existing domain-error path picks it
    up and extrapolates.
    """


class StandardPartUndefinedError(ValueError):
    """Asked for the standard part of something that has none.

    An infinitesimal IS infinitely close to zero, so st(|1|_-1) = 0.0 is
    correct and is not this case.  A quantity with a positive grade and
    nothing below it is unbounded -- there is no real number it approaches,
    and reporting the grade-0 coefficient (0.0, because the grade is absent)
    would say "vanishing" about something infinite.
    """


class LimitDoesNotExistError(ValueError):
    """Raised when a limit provably does not exist."""
    pass

class RoundingResidueWarning(UserWarning):
    """limit() dropped terms above grade 0 as float rounding residue.

    Float coefficients cannot cancel exactly along two different routes:
    sin(tan x) and tan(sin x) agree through x^6, but their x^3 and x^5
    coefficients differ in the last bit, and dividing by a denominator that
    starts at x^7 turns 2.8e-17 into a pole term of 8e-16.  The arithmetic keeps
    it -- it is exact by design and cannot know -- and limit() then read the
    sign of that residue as divergence and returned -INF for a limit of 1.

    limit() now reads terms above grade 0 as residue when every one of them is
    at most LIMIT_ROUNDING_RESIDUE of the largest finite term, and says so with
    this warning.  The cost is stated: a GENUINE infinite part that small would
    be read the same way, which is why this is a warning and never silent.
    """


#: Relative size, against the largest term at grade 0 or below, under which
#: limit() reads every term above grade 0 as rounding residue.  The measured
#: residue in the case that motivated it was 8.5e-16; the bound leaves three
#: orders of headroom for residues amplified by small denominators.
LIMIT_ROUNDING_RESIDUE = 1e-12


class LimitUndecidableError(ValueError):
    """Raised when composite arithmetic cannot determine the limit."""
    pass

class CompositionError(TypeError):
    """Raised when a function is not composable with composite arithmetic."""
    pass

# =============================================================================
# CORE: COMPOSITE NUMBER CLASS
# =============================================================================

def _is_wholly_zero(c):
    """True when every expressed coefficient is zero — the composite is a zero."""
    return c._backend.is_wholly_zero(c._data)


def _is_unit(c):
    """True for |1|_0, the multiplicative identity."""
    return c._backend.is_unit(c._data)


def _operands(a, b):
    """Prepare two composites for an operation.

    Brings them onto a single backend first -- set_backend() may have been
    called after one of them was built, and the module constants ZERO / INF
    are rebuilt on switch but references captured by `from ... import ZERO`
    are not -- then applies R1 to each.
    """
    if type(a._data) is not type(b._data):
        # Convert toward the RICHER representation.  A vector-dimension backend
        # can hold a scalar dimension d as (d, 0, ...); the reverse loses the
        # scale components, and numpy cannot even hold the tuples.  So when the
        # two differ, the vector side wins regardless of which operand it is.
        #
        # A change of storage is not a change of value: the completeness bound
        # and the denotation marker travel with it.  Rebuilt without them,
        # R(0)**3 / (R(1) + h*ln(h)) lost denotation_order 3 on the sparse-dense
        # backend -- the dividend moved to the vector backend unmarked -- while
        # the dict backend, which needs no move, kept it.
        if getattr(b._backend, "VECTOR_DIMS", False) and not getattr(
                a._backend, "VECTOR_DIMS", False):
            dims, vals = a._backend.to_arrays(a._data)
            a = Composite._wrap(b._backend.create_from_terms(dims, vals),
                                b._backend, demote=False,
                                complete=a._complete, denot=_denot_of(a))
        else:
            dims, vals = b._backend.to_arrays(b._data)
            b = Composite._wrap(a._backend.create_from_terms(dims, vals),
                                a._backend, demote=False,
                                complete=b._complete, denot=_denot_of(b))
    # Judge each operand on ITS OWN blessing. Asking whether the PAIR contained
    # a blessed value let `x*x + ZERO` through: x*x is blessed, so the pair was,
    # and the bare ZERO beside it stayed infinitesimal.
    #
    # _like, not the bare constructor: Composite({...}) binds whatever backend is
    # globally ACTIVE, so demoting an operand living on another one handed its
    # data to the wrong backend -- "'DictData' object has no attribute 'runs'".
    # ORDER MATTERS, measured: _is_infinitesimal_operand calls lead_order(), so
    # testing it first ran that on both operands of EVERY operation and cost 2.07x
    # on the default path with the mode never entered (0.1078s against 0.0522s).
    # _sealed is one ContextVar read that returns False immediately when the mode
    # is off, so it goes first and short-circuits.
    if _TAGGED.get() is not None:
        if _sealed(a) and _is_infinitesimal_operand(a):
            a = _like(a, {0: 0.0})
        if _sealed(b) and _is_infinitesimal_operand(b):
            b = _like(b, {0: 0.0})
    return _r1(a), _r1(b)


_CONVENTIONAL = _contextvars.ContextVar("composite_conventional_zero", default=False)

# See Composite.__float__ and _refusing_float.
_REFUSE_FLOAT = _contextvars.ContextVar("composite_refuse_float", default=False)


class FloatCoercionError(TypeError):
    """A composite was converted to float where its infinitesimal is needed."""


# See _refusing_residue.
_REFUSE_RESIDUE = _contextvars.ContextVar("composite_refuse_residue", default=False)


class ResidueError(ValueError):
    """A zero converted under R1 where its denotation is not defined.

    In one variable a cancellation or a written zero denotes x - x0.  Evaluated
    along a direction in several variables, the one infinitesimal is the
    direction's own parameter, and which variable the residue belongs to is a
    convention, not a property of the function.  Inside _refusing_residue()
    such an evaluation raises this instead of choosing.
    """


@_contextlib.contextmanager
def _refusing_residue():
    """For the duration, a zero that would convert under R1 raises ResidueError:
    a cancellation (`f - f`) and a written or manufactured zero (`0 * f`)."""
    token = _REFUSE_RESIDUE.set(True)
    try:
        yield
    finally:
        _REFUSE_RESIDUE.reset(token)


def _refuse_residue(what):
    if _REFUSE_RESIDUE.get():
        raise ResidueError(
            "%s would leave an R1 residue, and in several variables which "
            "variable it denotes is not defined; write the zero as a plain 0 "
            "or remove the cancellation" % what)


# See exp() and _exp_to_transseries.
_EXP_TO_TS = _contextvars.ContextVar("composite_exp_to_transseries", default=False)


@_contextlib.contextmanager
def _exp_to_transseries():
    """For the duration, exp of an infinite argument returns a Transseries.

    exp(-1/h) is nonzero and below every power, and the powers-and-logs group
    has no slot for it, so exp() refuses it.  composite.transseries gives it
    one: sector n carries exp(-n/h).  Inside this scope exp() hands such an
    argument to ts_exp instead of raising -- which is what lets an integrand
    like exp(-x) be evaluated at the node x = 1/h.  Outside it nothing changes.
    """
    token = _EXP_TO_TS.set(True)
    try:
        yield
    finally:
        _EXP_TO_TS.reset(token)


def _exp_level_one(x):
    """ts_exp, imported late: composite.transseries imports this module."""
    from composite.transseries import ts_exp
    return ts_exp(x)


@_contextlib.contextmanager
def _refusing_float():
    """For the duration, converting a composite to float raises."""
    token = _REFUSE_FLOAT.set(True)
    try:
        yield
    finally:
        _REFUSE_FLOAT.reset(token)

#: PROTOTYPE. The trigger: once a number carries an infinitesimal source, every
#: further zero it meets behaves conventionally. Per NUMBER, not per scope, so a
#: value cannot escape the regime by leaving a block.
_ARMED = object()          #: mode on, nothing tagged yet; no _src equals it
_TAG_PREVIOUS_TRACKING = None   #: what tracking was before TAG armed the mode
_TAGGED = _contextvars.ContextVar("composite_tagged_source", default=None)


def _is_infinitesimal_operand(o):
    """A purely infinitesimal operand: every term below grade 0.

    ZERO, h, h**2 and `3*h` qualify; a seeded `3 + h` does not, because it has a
    standard part and is a number being differentiated rather than a zero.
    """
    if not isinstance(o, Composite):
        return False
    order = o.lead_order()
    return order is not None and order > 0


def _is_blessed(o):
    """Does this operand descend from the tagged value?

    `_join` propagates `_src` through every operation, so every term of
    `sin(TAG(3))` carries the tag and keeps composite behaviour, while a bare
    ZERO does not and is conventional.
    """
    blessed = _TAGGED.get()
    if blessed is None or blessed is _ARMED:
        return False
    if getattr(o, "_src", None) != blessed:
        return False
    # Lineage is not enough. `R(0) * x` inherits x's source through _join, so the
    # all-zero product read as blessed and R1 converted it, depositing a stray
    # unit one grade below -- the jet of x**3 came back [27, 27, 20, 6]. A wholly
    # zero value has no infinitesimal CONTENT to have descended from the tag, so
    # it is never blessed. `x - x` is unaffected: the cancellation site sees both
    # operands blessed and converts there, before this is ever consulted.
    return not (isinstance(o, Composite) and _is_wholly_zero(o))


def _sealed(*operands):
    """True when the zeros among these operands must behave conventionally.

    PURE: asking never changes the answer. The version this replaced counted
    operations and mutated, so the first of its three call sites took the slot
    and the second saw a sealed state -- `R(1)/ZERO` came back <|inf|_0> instead
    of |1|_1 and `sqrt(ZERO)` came back <|nan|_0> instead of |1|_-0.5.

    Only TAG() blesses a source. Inferring which infinitesimal is the blessed
    one does not work, and the three attempts are recorded at _TAGGED.
    """
    if _TAGGED.get() is None:
        return False
    return not any(_is_blessed(o) for o in operands)


# Every infinitesimal SOURCE created, counted at the places one can be born.
# See degeneracy_watch() for what it is for and why the count is of sources
# rather than of terms.  This is observability; the verdict is carried on the
# numbers themselves, see _mint.
_inf_sources = [0]

# Provenance carried on every composite, as ONE id and ONE flag:
#
#   _src   the FIRST infinitesimal source this number descends from, or None
#   _deg   True once a SECOND, different source has reached it
#
# A flag alone cannot do it: in `x*x + x` both operands carry an infinitesimal
# and it is the same one, so "both are graded" is not the question.  The id
# answers it, and one id is enough, because a number that is not yet flagged
# descends from at most one source -- so its first id names its whole set.
# That makes this exactly equivalent to carrying the set, for the only question
# asked of it, at two fields instead of an allocation per number.
#
# Counting is not carried.  Knowing a third source arrived tells a caller
# nothing the second did not already tell them: the reading is off the
# conventional function either way.
_next_src = [0]

# Sources minted DURING an operation, drained by _carries.  R1 converts a
# wholly-zero operand inside the op, and that conversion makes infinitesimal
# content no operand had, so the operand list cannot show it.  Measured: with
# the union taken over the operands alone, `x*x + 0.0` and `x*x + (x - x)` both
# came back clean.  Global like _inf_sources, and no more thread-isolated.
_pending_mints = []
_in_op = [0]

# OFF by default.  Carrying only matters to a caller who is going to read grade
# -n as the nth derivative of a function they have in mind, and it costs 11-21%
# on ordinary arithmetic (measured: +11.1% on 200 derivative(), +21% on a
# 200-term multiply).  Switched off it costs nothing at all rather than less:
# set_degeneracy_tracking puts the UNDECORATED operators back on the class, so
# there is no wrapper frame to enter.
#
# The entry points that report on degeneracy -- taylor_degeneracy, explain --
# turn it on around their own evaluation, so the supported path works whatever
# this is set to.  MINTING IS NOT SWITCHED: it happens at three cold sites and
# ZERO is built once at import, so if minting were skipped while off, a later
# `x*x + ZERO` would not flag once it was switched on.
_TRACKING = [False]

#: The operators that carry.  Named here rather than discovered, so switching
#: cannot half-apply.
_CARRIED_OPS = ("__add__", "__radd__", "__sub__", "__rsub__", "__neg__",
                "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__pow__")


def _mint(c):
    """Mark c as a fresh infinitesimal source.

    The three callers are the three ways a source comes into existence, and
    they are the three places _inf_sources was already counted: Composite.zero
    (a written zero), _r1 (a cancellation converting), and _seeded (an
    extraction's own seed).  A composite written by hand as Composite({-1: 1.0})
    passes none of them and carries no source, so it does not flag.  That is a
    known gap, not an oversight: see taylor_degeneracy.
    """
    _inf_sources[0] += 1
    _next_src[0] += 1
    c._src = _next_src[0]
    c._deg = False
    if _in_op[0]:
        _pending_mints.append(c._src)
    return c


def _join(result, *operands, minted=()):
    """Carry provenance from operands, and any source minted inside the op.

    `minted` is a separate channel on purpose.  Folding ids in with the
    operands and telling them apart by type read the exponent of `x ** 5` as
    source number 5, and every clean expression with a raw int operand flagged.
    """
    src, deg = None, False
    # getattr, not attribute access: a Composite SUBCLASS may build its
    # instances through __new__ without going near __init__ or _wrap, and
    # forensics._Audited does exactly that.  An unset slot then raises, and it
    # raised from inside every arithmetic operation the audit performed.
    ids = [s for s in (getattr(o, "_src", None) for o in operands
                       if isinstance(o, Composite)) if s is not None]
    if any(getattr(o, "_deg", False) for o in operands
           if isinstance(o, Composite)):
        deg = True
    for o_src in [*ids, *minted]:
        if src is None:
            src = o_src
        elif src != o_src:
            deg = True                 # a second, different source
    result._src = src
    result._deg = deg
    return result


def _carries(op):
    """Give an arithmetic dunder provenance carrying.

    Wrapping the ten dunders keeps the rule in one place: every other composite
    in the library is built by them or by _like, so nothing else has to know.
    """
    @_functools.wraps(op)
    def inner(self, *args):
        depth = len(_pending_mints)
        _in_op[0] += 1
        try:
            result = op(self, *args)
        finally:
            _in_op[0] -= 1
        # FAST PATH: one argument and nothing minted inside, which is every
        # operation that is not a conversion.  Inlined rather than delegated to
        # _join because this runs on every arithmetic operation in the library
        # and the general form costs two comprehensions and a call: measured at
        # +32% on `40x sin(exp(x))`, +12% with this branch.
        # `type(...) is Composite`, not isinstance: a SUBCLASS may never have
        # set the slots (forensics._Audited builds through __new__), and only
        # the general form is tolerant of that.
        if (len(_pending_mints) == depth and type(self) is Composite
                and len(args) < 2):
            if type(result) is not Composite:
                return result
            src, deg = self._src, self._deg
            if args and type(args[0]) is Composite:
                other = args[0]
                if other._deg:
                    deg = True
                o_src = other._src
                if o_src is not None:
                    if src is None:
                        src = o_src
                    elif src != o_src:
                        deg = True         # a second, different source
            elif args and isinstance(args[0], Composite):
                return _join(result, self, args[0])     # subclass: general form
            result._src = src
            result._deg = deg
            return result
        if isinstance(result, Composite):
            _join(result, self, *args, minted=_pending_mints[depth:])
        del _pending_mints[depth:]
        return result
    return inner


@_contextlib.contextmanager
def conventional():
    """Treat an expressed zero as the ORDINARY zero, for the duration.

    The system's answer to `x*x + 0` is the derivative of x**2 + h, and since
    h is x - a that is x**2 + x - a, whose derivative at 2 is 5.  That is
    correct and it is the whole point.  But a caller who wrote the zero
    without meaning it, or who is porting a formula from somewhere that has an
    additive identity, wants 4.  This gives them 4 without touching the
    arithmetic everyone else gets.

    Three switches, at the only three places the difference shows:

      - Composite.zero() yields |0|_0 instead of |1|_-1, so a written zero is
        a zero TERM and gets absorbed rather than converting
      - R1 does not convert a wholly-zero operand, so a cancellation stays
        |0|_d instead of lifting one grade down
      - a wholly-zero divisor raises ZeroDivisionError.  Without this one the
        single-term-divisor path divides coefficients by 0.0 and hands back
        |inf|_0 or |nan|_0 with no error at all, which is worse than either
        semantics.

    What it neutralises is a zero that arrives as a VALUE: `R(0)`, a bare `0`
    or `0.0`, a data field that happens to be zero, and a cancellation.

    ZERO and h are NOT neutralised.  They name the infinitesimal, in both
    modes.  Rebinding them was tried and does not work: `from composite import
    ZERO` binds the object at import, so rebinding this module's global cannot
    reach the name the caller is actually using, and that spelling is the one
    everybody writes.  Walking every loaded composite module and rebinding
    there fixed the library's own uses and still left the caller's, so
    `(x + ZERO)**2` kept its infinitesimal while `(x + cl.ZERO)**2` lost it --
    a split decided by import style, which is worse than either answer.  There
    is no arithmetic route either: the operand in `x + ZERO` is |1|_-1, the
    same value as the seed's own unit, so nothing distinguishes a zero spelled
    ZERO from an infinitesimal meant as one.

    `(x*x - R(4))/(x - R(2))` still resolves to 4 with derivative 1: R2 holds
    that zero inert, so no conversion was ever needed for it.

    Seeding does not go through ZERO in the block.  `_seeded` builds the seed
    directly at both ends, which also retires its `if at == 0` special case.

        with conventional():
            all_derivatives(lambda x: x*x + R(0), 2.0, up_to=3)   # 4, 4, 2, 0
    """
    token = _CONVENTIONAL.set(True)
    try:
        yield
    finally:
        _CONVENTIONAL.reset(token)


@_contextlib.contextmanager
def single_infinitesimal():
    """PROTOTYPE. The FIRST infinitesimal is composite; every later zero is not.

    Once a number carries an infinitesimal source, further zeros it meets stop
    converting: multiplication by zero annihilates, `a - a` is an inert |0|_0,
    `c + 0` is c, and division by a zero raises again. The payback is that the
    jet stays the CLASSICAL one -- no second source can enter, so no order is
    ever a denoted reading.

    This is a trigger on the NUMBER, not a scope: the state travels with the
    value through `_join`, so it cannot be defeated by the value leaving a block,
    which is the one thing conventional() cannot promise.
    """
    # The per-number state rides on _src, which only propagates while the
    # degeneracy wrappers are installed -- set_degeneracy_tracking is a REBIND
    # of the operators, so with it off there is no wrapper and no provenance.
    # The trigger therefore cannot be cheaper than tracking: 11-21% by its own
    # docstring. That is the price of per-number rather than per-scope.
    previous = set_degeneracy_tracking(True)
    token = _TAGGED.set(_ARMED)            # armed; TAG() installs the real id
    try:
        yield
    finally:
        _TAGGED.reset(token)
        set_degeneracy_tracking(previous)


def TAG(value):
    """PROTOTYPE. Bless ONE value as the infinitesimal source, and arm the mode.

    TAG is the switch. There is nothing else to turn on::

        x = TAG(3)
        (x*x).d(1)              # 6.0
        (x*x + R(0)).d(1)       # 6.0   the written zero absorbed

    From the call onward every zero that does NOT descend from the tag behaves
    conventionally -- `0`, `R(0)`, `ZERO` and a cancellation all absorb,
    multiplication by zero annihilates, division by one raises -- so no second
    source can enter and the jet stays the CLASSICAL derivative at every order.
    Every term of `sin(TAG(3))` descends from the tag, so series still work.

    A second TAG REPLACES the blessing rather than raising: re-running a cell
    that tags should not be an error, and the newest tag is the live one. Only
    one value is ever blessed, which is the whole point.

    `UNTAG()` turns it off. `single_infinitesimal()` is the scoped form, for code
    that wants the mode to end with a block rather than with a call.

    The mode needs `set_degeneracy_tracking`, since the blessing travels on
    `_src`, so arming costs the 11-21% that switch documents. Arming is therefore
    not free, but it is paid once and only while the mode is on.

    The seed itself is built with the regime switched OFF. Built inside it,
    `_ensure_composite(value) + ZERO` has the mode demote its own ZERO -- nothing
    is blessed yet, so that ZERO is not blessed either -- and TAG would return a
    value carrying no infinitesimal at all.
    """
    global _TAG_PREVIOUS_TRACKING
    token = _TAGGED.set(None)
    try:
        base = _ensure_composite(value)
        terms = dict(base.coeffs_dict())

        def _power(dim):
            return dim[0] if isinstance(dim, tuple) else dim

        if any(v != 0.0 and _power(d) < 0 for d, v in terms.items()):
            # It ALREADY carries an infinitesimal, so adding another doubles it.
            # Three ways in: TAG(R(0)), because R(0) converted to |1|_-1 before
            # TAG ever saw it; TAG(ZERO); and TAG(x) where x was seeded already.
            # TAG(R(0)) came out |2|_-1 and the jet of ln(1+x)/x at 0 then read
            # [1, -1, 8/3, -12] -- correct for a seed of 2h, and not what was
            # asked for. Bless what is there instead, which also makes TAG
            # idempotent: TAG(TAG(3)) is TAG(3).
            seeded = _mint(_like(base, terms))
        elif _is_wholly_zero(base):
            # No infinitesimal and nothing but zeros: `base + ZERO` would convert
            # base under R1 and again hand back two units. _seeded carries the
            # same special case for the same reason.
            terms[-1] = terms.get(-1, 0.0) + 1.0
            seeded = _mint(_like(base, terms))
        else:
            seeded = _mint(base + ZERO)
    finally:
        _TAGGED.reset(token)
    if _TAGGED.get() in (None, _ARMED):
        # Arming for the first time: remember what tracking was, so UNTAG can
        # put it back rather than guessing that it was off.
        _TAG_PREVIOUS_TRACKING = set_degeneracy_tracking(True)
    _TAGGED.set(seeded._src)
    return seeded


def UNTAG():
    """Turn the single-infinitesimal mode off and restore degeneracy tracking.

    Returns True if the mode had been on. Safe to call when it was not.
    """
    global _TAG_PREVIOUS_TRACKING
    was_on = _TAGGED.get() is not None
    _TAGGED.set(None)
    if was_on and _TAG_PREVIOUS_TRACKING is not None:
        set_degeneracy_tracking(_TAG_PREVIOUS_TRACKING)
        _TAG_PREVIOUS_TRACKING = None
    return was_on



def _r1(c, sibling=None):
    """R1 — a zero used as an OPERAND converts: |0|_d becomes |1|_(d-1).

    Only a composite that is wholly zero is a zero.  A zero sitting among
    nonzero terms is a term of the number, not an operand, so nothing happens
    to it (R2).  When several dimensions are zero, only the lowest converts;
    the composite then holds a nonzero and the rest are terms.
    """
    if _CONVENTIONAL.get():
        return c                       # see conventional()
    if _sealed(c):
        return c                       # not the tagged lineage: conventional
    if not _is_wholly_zero(c):
        return c
    dims, vals = c._backend.to_arrays(c._data)
    if len(dims) == 0:
        return c                       # NOTHING has no dimension to convert
    _refuse_residue("a zero used as an operand")
    _warn_zero_operand()
    dims = dims.copy()
    vals = vals.copy()
    # to_arrays is sorted ascending, so [0] is the lowest dimension.
    #
    # _dim_shift, not `- 1`: on a vector dimension the scalar spelling raises
    # "unsupported operand for -: 'tuple' and 'int'", and it raised from inside
    # R1 -- so `0.0 + ln(1/h)`, the most ordinary line anyone would write,
    # could not be evaluated at all.  Shifting the POWER axis is the right
    # move rather than the leading axis: it lowers the dimension in every case
    # ((-1, 1) < (0, 1) lexicographically), whereas decrementing the leading
    # axis would turn the zero at (0, 1) into (0, 0) -- finite, not
    # infinitesimal, which is not what R1 means.  R(0) already produced
    # (-1, 0) by this convention, so this only makes the two paths agree.
    dims[0] = _dim_shift(dims[0], -1)
    vals[0] = 1.0
    # _mint: this conversion MAKES infinitesimal content no operand had, which
    # is the second of the three ways a source comes into existence.
    return _mint(Composite._wrap(
        c._backend.create_from_terms(dims, vals),
        c._backend, demote=False,
        denot=_merge_denot(_denot_of(c), _lead_order(dims[0]))))


CANCELLATION_CARRIES = "quantity"
"""What a cancellation deposits.  "quantity" or "magnitude".

    quantity    a - a  =  a * h.  The ANNIHILATED QUANTITY, one grade down.
                6 - 6 is |6|_-1 and (3+h) - (3+h) is |3|_-1 + |1|_-2.
    magnitude   only the deepest coefficient, shifted.  (3+h) - (3+h) is
                |0|_0 + |1|_-2.  Kept reachable for comparison.

"quantity" is the rule because "magnitude" only half-keeps Euler.  Measured:

    (x-x)/(y-y), x=3+h y=2+h    magnitude 1.0        quantity 1.5 = x/y
    a*(b-b) == a*b - a*b        magnitude 32/64      quantity 64/64

Under "magnitude" different SCALAR zeros get different characters and different
COMPOSITE zeros all come back as 1, because the deepest coefficient of a seeded
quantity is always 1.  Carrying the whole quantity gives the ratio at every
order, which is what §85 asks for.

It also restores distributivity across a cancellation with no sign sacrifice:
a*(b*h) and (a*b)*h are the same term by associativity of multiplication, sign
included, so the trilemma does not apply -- it was about a residue read off a
cancelling PAIR of coefficients, and `a` in `a - a` is one quantity.

THE COST.  `a - a` has `a` on both sides, so the annihilated quantity is
unambiguous.  `a + (-a)` does not, and taking it from the left operand makes
`a + (-a)` and `(-a) + a` differ -- the same law CANCELLATION_SIGNED = True
broke, for the same reason.
"""

CANCELLATION_SIGNED = False
"""Residue of a cancellation keeps the SIGN of what was annihilated.

True  -> (-6) - (-6) is -6_-1.  Multiplication distributes across cancellation
         (81/81 on scalar pairs) and `a + b == b + a` fails on exactly the
         cancelling pair b == -a (73/81).
False -> (-6) - (-6) is 6_-1.  Addition stays commutative (81/81) and
         distributivity holds only where the other factor is positive (45/81).

No residue function can do both: commutativity needs m(v,-v) = m(-v,v), and
distributivity at factor -1 needs m(-v,v) = -m(v,-v), so m = 0 -- which is the
conventional ring, and the corner this system exists to leave.
"""


_CANCELLED_MSG = (
    "a cancellation at grade %s added an order-%g infinitesimal of magnitude "
    "%.6g. This is a DENOTATION, not a corruption: an expressed zero of "
    "magnitude m at grade -k denotes m*(x-a)**k, so from here the jet is the "
    "exact jet of the function the expression denotes rather than of its "
    "classical reading. Measured: x*x*(R(2)-R(2)) + x*x gives 9, 24, 26, 12, "
    "and so does 2x**3 - 5x**2 computed independently -- a real slope and a "
    "real acceleration, of that function. The standard part is unchanged; "
    "orders from %g are the denoted reading. Composite({}) is NOTHING and adds "
    "no order, if the classical reading was intended.")


CONVENTIONAL_STRICT = False
"""What d(n) does when the jet is not the CONVENTIONAL one.

False (default) warns and returns the value.  True raises
NotConventionalError.  Warning is the default because the value is not wrong:
an expressed zero of magnitude m at grade -k denotes m*(x-a)**k, and read that
way the jet is the exact jet of the function the expression denotes.  Measured:

    x*x*(R(2)-R(2)) + x*x   jet 9, 24, 26, 12
    2x**3 - 5x**2           jet 9, 24, 26, 12    the same function, independently

So d(n) returns a true derivative -- a real slope, a real acceleration -- of
that function rather than of the classical reading of the formula.  Refusing it
would be refusing a correct answer, which is why strictness is opt-in: it is
for a caller who wants classical semantics and would rather fail than receive
the other reading.
"""


class NotConventionalError(ValueError):
    """Raised by d(n) under CONVENTIONAL_STRICT when the jet is not classical."""


class NotConventionalWarning(UserWarning):
    """d(n) returned a jet of the DENOTED function rather than the classical one.

    Its own category, and registered `always` below, because the default filter
    shows a given message once per (text, location) and that is the wrong
    mechanism here.  This warning is a fact about the VALUE being read, not
    about the line reading it: in a REPL every read is `<stdin>:1`, so after
    one `d(1)` the rest are silent and non-classical numbers are collected with
    nothing said.  Observed exactly that way -- d(1) and d(2) went quiet while
    d(3), whose text had not been seen, still appeared.

    Quiet it per caller in the ordinary way, which the `always` registration
    does not prevent:

        warnings.filterwarnings("once", category=NotConventionalWarning)
    """


class CancellationWarning(UserWarning):
    """A cancellation deposited a residue at this point in the code.

    Left on the DEFAULT filter, once per location, because it reports a
    property of the line rather than of a value -- and a loop that cancels a
    million times should say so once, not a million times.
    """


_warnings.filterwarnings("always", category=NotConventionalWarning)


_DENOT_MSG = (
    "d(%s) is not the conventional derivative. An expressed zero entered this "
    "value at order %g, and an expressed zero of magnitude m at grade -k "
    "denotes m*(x-a)**k -- so this jet is the exact jet of the function the "
    "expression DENOTES, not of its classical reading. It is a true derivative "
    "of that function, not a corrupted one: orders below %g are classical, "
    "orders from %g are not. To get the classical reading, spell the annihilation "
    "as Composite({}) (NOTHING adds no order) or re-evaluate the function with "
    "the conversions off. Set CONVENTIONAL_STRICT = True to raise here instead.")


def _report_denotation(n, order):
    """d(n) reached an order an expressed zero contributed to."""
    msg = _DENOT_MSG % (n, order, order, order)
    if CONVENTIONAL_STRICT:
        raise NotConventionalError(msg)
    import warnings
    warnings.warn(msg, NotConventionalWarning, stacklevel=3)


def _denot_of(x):
    """The marker, tolerating an object that never assigned the slot.

    Four places build a Composite through __new__ and assign only the fields
    they know about -- TracedComposite, and forensics' _Audited in three spots.
    A __slots__ attribute that was never set raises AttributeError on READ, so
    every read goes through here.  Measured when it did not: explain() came
    back with stability None and test_forensics aborted.
    """
    return getattr(x, "_denot", None)


def _lead_or_zero(x):
    """lead_order, with NOTHING and a bare scalar reading as 0."""
    o = x.lead_order() if isinstance(x, Composite) else None
    return 0 if o is None else o


def _merge_denot(*cands):
    """The SHALLOWEST denotation order among the candidates, or None."""
    vals = [c for c in cands if c is not None]
    return min(vals) if vals else None


def _denot_additive(a, b):
    """+ and - keep each operand's order: neither shifts a grade (R4)."""
    da, db = _denot_of(a), _denot_of(b)
    if da is None and db is None:
        return None                      # the common path costs two compares
    return _merge_denot(da, db)


def _denot_product(a, b):
    """Grades ADD, so a denoted operand moves by the other's leading order.

    Measured: a residue at order 1 times an ordinary number stays at 1, times
    ZERO goes to 2, divided by ZERO comes back to 0 -- exactly lead_order
    arithmetic, which is why the shift is read off lead_order rather than
    guessed.  lead_order() is only called when a marker exists, so the
    unmarked path pays nothing.
    """
    da, db = _denot_of(a), _denot_of(b)
    if da is None and db is None:
        return None
    return _merge_denot(None if da is None else da + _lead_or_zero(b),
                        None if db is None else db + _lead_or_zero(a))


def _denot_quotient(a, b):
    """Division subtracts the divisor's leading order, so a marker moves UP.

    That is the case that makes the marker necessary rather than inferable:
    residue / ZERO reaches order 0, where even the standard part carries it.
    """
    da, db = _denot_of(a), _denot_of(b)
    if da is None and db is None:
        return None
    shift = -_lead_or_zero(b)
    return _merge_denot(None if da is None else da + shift,
                        None if db is None else db + shift)


def _warn_cancelled(dim, order, mag):
    """The injection made audible, at the one place a cancellation converts.

    R1's warning covered this case while the conversion happened on next use.
    Converting at the site took it out of _r1's path, so it is re-raised here
    and says more: _r1 could only report that a zero converted, this knows
    which order entered and how big it is.
    """
    import warnings
    # to_arrays hands back numpy floats, so an integer grade prints as "-1.0"
    # and a reader has to wonder whether it is a fractional grade.  It matters:
    # a half grade is a branch point, and the two must not look alike.
    if not isinstance(dim, tuple) and float(dim).is_integer():
        dim = int(dim)
    warnings.warn(_CANCELLED_MSG % (dim, order, mag, order),
                  CancellationWarning, stacklevel=4)


def _cancellation_residue(a, b, result):
    """R1 AT THE CANCELLATION SITE: the residue remembers what was cancelled.

    `6 - 6` leaves 6_-1, not 1_-1.

    The magnitude that was annihilated exists only here, in the operands.  _r1
    runs later, once the coefficients are gone, so all it can deposit is a
    unit -- which makes it inhomogeneous, and that is precisely why
    multiplication does not distribute across it: R1(c*x) = 1_(d-1) for every
    c, so the c cannot come back out.  Reading the magnitude at the site makes
    the conversion homogeneous of degree 1 and the law returns.

    Converts the LOWEST grade only, exactly as R1 does, so the composite then
    holds a nonzero and every remaining zero is inert by R2 and retained.
    Falls through untouched when nothing was annihilated at that grade, which
    leaves _r1 to handle a written or manufactured zero with its unit residue:
    a written zero has no magnitude to read, which is not the same as a
    magnitude of zero.

    Honours conventional() exactly as _r1 does.  Converting at the site takes
    this route out of _r1's path, so the context's own check there does not
    cover it, and a cancellation kept converting inside the very block that
    exists to stop it -- which is what conventional() is for.
    """
    if _CONVENTIONAL.get():
        return result                  # see conventional()
    if _sealed(a, b):
        return result                  # the trigger: a - a stays |0|_0
    if not _is_wholly_zero(result):
        return result                       # a tail survives: R2 governs, not R1
    dims, vals = result._backend.to_arrays(result._data)
    if len(dims) == 0:
        return result                       # NOTHING has no grade to convert
    _refuse_residue("a cancellation")
    if CANCELLATION_CARRIES == "quantity":
        # a - a = a * h.  Every grade of the annihilated quantity moves down
        # one, so the residue IS that quantity and the ratio of two zeros is
        # the ratio of what they destroyed.  Built by shifting rather than by
        # `a * ZERO` so it cannot re-enter this function or meet the order cap.
        src = a if not _is_wholly_zero(a) else b
        sdims, svals = src._backend.to_arrays(src._data)
        if len(sdims) == 0:
            return result                   # nothing was annihilated
        shifted = {_dim_shift(sd, -1): float(sv) for sd, sv in zip(sdims, svals)}
        out = _like(src, shifted)
        if _is_wholly_zero(out):
            return result                   # would recurse: leave it to _r1
        new_dim = out.lead_dim()
        order = _lead_order(new_dim)
        out._denot = _merge_denot(_denot_of(result), order)
        out._complete = result._complete
        _mint(out)
        _warn_cancelled(new_dim, order, out.coeff(new_dim))
        return out

    d = dims[0]                             # to_arrays is sorted ascending
    mag = a.coeff(d)
    if mag == 0.0:
        mag = b.coeff(d)                    # whichever operand carried the value
    if mag == 0.0:
        return result                       # nothing annihilated here: leave to _r1
    dims, vals = dims.copy(), vals.copy()
    new_dim = _dim_shift(d, -1)
    dims[0] = new_dim
    vals[0] = mag if CANCELLATION_SIGNED else abs(mag)
    order = _lead_order(new_dim)
    out = Composite._wrap(result._backend.create_from_terms(dims, vals),
                          result._backend, demote=False,
                          denot=_merge_denot(_denot_of(result), order))
    out._complete = result._complete
    # A cancellation is a source, and it used to be minted in _r1 -- converting
    # at the site moved this case out of _r1's path, so the mint moves with it
    # or a cancelled zero stops being a second source and stops flagging.
    _mint(out)
    _warn_cancelled(new_dim, order, vals[0])
    return out


def _scalar_operand(other):
    """A Python or numpy scalar entering an operation as the other operand.

    A WRITTEN ZERO IS AN EXPRESSED ZERO: |0|_0, which R1 converts.  That is
    not a cost the system imposes on the unwary -- it IS the system.  An
    expressed zero is an infinitesimal, and `1 - 1 != 0` is the same statement.

    This was briefly changed so that a bare scalar zero produced NOTHING, on
    the argument that a zero arriving from data has no event behind it.  The
    argument was wrong on both counts.  §0 already draws the line: NOTHING is
    the ABSENCE of a term, and a zero that was written is not absent -- the
    event R1 records is the EXPRESSION of the zero, and writing it is that
    event.  Worse, making a keyboard-reachable zero into an additive identity
    turns this back into a conventional ring with an extra symbol attached,
    and two zeros obeying different laws is worse than one obeying one.

    The bug that motivated the change was never here.  `d.get(k, 0.0)`
    MANUFACTURED a zero for a Pade coefficient that did not exist -- it
    expressed a zero the mathematics never expressed -- and the lateral Borel
    integral returned 3224 where the answer is 0.697.  The fix is `.get(k)`
    and skip the absent term, which is what the warning below has always said.
    """
    if other == 0:
        return Composite({0: 0.0})
    return Composite(float(other))


_ZERO_OPERAND_MSG = (
    "a zero operand converted: |0|_d -> |1|_(d-1) (R1). That is the rule and "
    "it is correct -- an expressed zero IS an infinitesimal, which is the same "
    "statement as 1 - 1 != 0. It is reported because the dimension-0 value "
    "stays right while every derivative moves, which is the worst failure "
    "shape there is. Three ways to arrive here: a written zero (`c + 0`, "
    "`acc = 0`) -- correct, and Composite(0) or ZERO silences it; a zero that "
    "was MANUFACTURED for something absent (`d.get(k, 0.0)` for a coefficient "
    "that does not exist) -- a bug, use `d.get(k)` and skip, or Composite({}) "
    "for an accumulator, and it returned 3224 for a value of 0.697 once; or a "
    "computation that cancelled to all-zero -- correct, and nothing announced "
    "it before this warning existed."
)


_TRUNCATED_MSG = (
    "%d term%s dropped: %s. The result is complete to order %s and the terms "
    "past it are still present -- arithmetic keeps producing them, and they are "
    "not wrong so much as unvouched for. That is the dangerous shape: a "
    "coefficient that looks right and is not. Read complete_coeffs() rather than "
    "coeffs_dict() wherever a wrong coefficient would be worse than a missing "
    "one, and leaked_coeffs() to see exactly what is beyond the bound. To move "
    "the bound instead: raise `terms=` on the transcendental that set it, or "
    "set_max_order(None) / a larger MAX_ACTIVE_DIMS for the caps.")


def _warn_truncated(dropped, reason, bound):
    """Truncation made audible, at the two places a term is actually discarded.

    Only where something is REMOVED, never where a bound is merely recorded.
    _truncate_order is called on the way out of every transcendental and
    usually has nothing to drop; warning there too would fire on every sin()
    in the library and the message would be worth nothing.
    """
    import warnings
    warnings.warn(
        _TRUNCATED_MSG % (dropped, "" if dropped == 1 else "s", reason,
                          "unbounded" if bound is None else bound),
        stacklevel=4)


def _warn_zero_operand():
    """R1 made audible, once, at the only place a zero actually converts.

    This used to be two warnings.  A bare Python 0 became |0|_0 and then
    converted here anyway, so `c + 0` fired both the seed warning and this
    one, while `c / 0` fired neither -- the seed warning was never wired into
    __truediv__.  _r1 is the single point every route passes through, so the
    message lives here and covers all three: written, manufactured, cancelled.
    """
    import warnings
    warnings.warn(_ZERO_OPERAND_MSG, stacklevel=5)


def is_nothing(x):
    """True for NOTHING -- no term at any dimension.  The classical zero.

    This is what a bare Python 0 becomes when it enters an operation, and it
    is an additive identity and a multiplicative annihilator.  It is NOT a
    zero at a dimension and does not convert under R1.
    """
    return isinstance(x, Composite) and not x.coeffs_dict()


def is_vanishing(x):
    """True for a zero that HAS a dimension -- one that converts under R1.

    `c - c`, `Composite({0: 0.0})`, `L - L`.  These are the values for which
    `x == 0` is False, because a dimension existed and was annihilated, and
    the trace of that event is what `1 - 1 != 0` is a claim about.
    """
    return (isinstance(x, Composite) and bool(x.coeffs_dict())
            and _is_wholly_zero(x))


def is_zero(x):
    """True when x carries no value: NOTHING, or a zero at some dimension.

    USE THIS RATHER THAN `x == 0`.  `x == 0` compares against NOTHING, so it
    is True for NOTHING and False for a dimensioned zero -- which is R1 and R6
    working exactly as specified, and which surprises every reader once.
    is_zero() asks the question people actually mean; is_vanishing()
    distinguishes the case that still carries a dimension.
    """
    if isinstance(x, (int, float)):
        return x == 0
    return is_nothing(x) or is_vanishing(x)


class Composite:
    """
    Composite number: |coefficient|_dimension

    Represents numbers with dimensional structure where:
        dimension 0  = real numbers
        dimension -1 = infinitesimals (structural zero)
        dimension -2 = second-order infinitesimals
        dimension +1 = infinities (structural infinity)

    Examples:
        |5|₀      = real number 5
        |1|₋₁     = structural zero (infinitesimal h)
        |1|₁      = structural infinity
        |3|₀+|2|₋₁ = 3 + 2h (3 plus 2 infinitesimals)
    """

    # _complete: highest Taylor order this value is COMPLETE to, or None for
    # "exact" -- a literal, a seeded variable, a polynomial, a deconvolution.
    # It cannot be inferred from the coefficients: _seeded(t) is exact with max
    # order 1, sin(x*x) is truncated with max order 11, and both merely look
    # like "max order K".  So it is carried.
    __slots__ = ['_data', '_backend', '_complete', '_denot', '_src', '_deg']

    def __init__(self, coefficients=None, _data=None):
        self._backend = get_backend()
        self._complete = None          # exact unless a series says otherwise
        self._denot = None             # no expressed zero has entered
        # Assigned here and in _wrap, never left unset: the carry fast path
        # reads self._src directly rather than through getattr, deliberately,
        # and a __slots__ attribute that was never assigned raises on READ.
        self._src = None               # descends from no infinitesimal source
        self._deg = False              # no second source has reached it

        if _data is not None:
            # Internal fast path: created by arithmetic ops
            self._data = _data
            return

        if coefficients is None:
            self._data = self._backend.create_from_terms(
                np.array([], dtype=DIM_DTYPE),
                np.array([], dtype=np.float64))
        elif isinstance(coefficients, (int, float)):
            # FIXED: Composite(0) → |0|₀ (expressed zero at dim 0)
            # Previously: Composite(0) → {} (empty, indistinguishable from Composite())
            # Now: any numeric input creates an expressed dimension 0
            self._data = self._backend.create(0, float(coefficients))
        elif isinstance(coefficients, dict):
            # FIXED: keep expressed zeros — if a dimension is in the dict,
            # it was expressed, even if its coefficient is 0.0
            # Previously: {k: v for k, v in ... if v != 0} stripped them
            if coefficients:
                from composite.backends.vector_dim_backend import dom_sorted as _dom_sorted
                sorted_dims = _dom_sorted(coefficients.keys())
                _exact_frac = (
                    getattr(self._backend, "EXACT_DIMS", False)
                    and any(isinstance(_d, _Fraction) and _d.denominator != 1
                            for _d in sorted_dims))
                if (sorted_dims and isinstance(sorted_dims[0], tuple)) or _exact_frac:
                    # VECTOR dimensions (power, log, ...).  np.array would turn
                    # these into a 2-D float grid and the rows are unhashable;
                    # an object array keeps them 1-D and keeps the tuples whole.
                    #
                    # A FRACTION needs the same treatment for the same reason one
                    # step on: DIM_DTYPE is float64, so Fraction(1, 10**7) became
                    # the binary value of 1e-7 and came back out as
                    # Fraction(944473296573929, 9444732965739290427392).  Simple
                    # fractions survived only because as_fraction round-trips
                    # them, which is luck rather than a contract.  Guarded by
                    # EXACT_DIMS so a float64 backend never receives one.
                    dims = np.empty(len(sorted_dims), dtype=object)
                    for _i, _d in enumerate(sorted_dims):
                        dims[_i] = _d
                else:
                    dims = np.array(sorted_dims, dtype=DIM_DTYPE)
                vals = np.array([coefficients[d] for d in sorted_dims], dtype=np.float64)
                self._data = self._backend.create_from_terms(dims, vals)
            else:
                self._data = self._backend.create_from_terms(
                    np.array([], dtype=DIM_DTYPE),
                    np.array([], dtype=np.float64))
        else:
            raise TypeError(f"Cannot create Composite from {type(coefficients)}")

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------

    @classmethod
    def _wrap(cls, data, backend=None, demote=True, complete=None, denot=None):
        """Create a Composite directly from backend data. No dict parsing.

        `backend` must be given whenever the data was built by a backend that
        is not the active one -- otherwise the object carries that backend's
        data under the global backend's methods, and the next call reaches for
        an attribute the data does not have.
        """
        be = backend if backend is not None else get_backend()
        if demote and be.VECTOR_DIMS:
            data, be = _demote(data, be)
        obj = cls.__new__(cls)
        obj._backend = be
        obj._data = data
        obj._complete = complete
        obj._denot = denot
        obj._src = None
        obj._deg = False
        return obj

    # -------------------------------------------------------------------------
    # Backward compatibility: .c property
    # -------------------------------------------------------------------------

    @property
    def c(self):
        """Backward-compatible dict view. Returns {dim: coeff} dict.

        WARNING: This reconstructs a dict from backend data on every call.
        Use read_dim / to_arrays for new code. This exists only so that
        existing transcendental functions, antiderivative(), show(), and
        TracedComposite keep working without changes.
        """
        dims, vals = self._backend.to_arrays(self._data)
        return {dim_cast(d): float(v) for d, v in zip(dims, vals)}

    # -------------------------------------------------------------------------
    # Constructors (UNCHANGED)
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls):
        """Structural zero: |1|₋₁ (infinitesimal), or |0|₀ under conventional()

        Counted: this is one of the two ways an infinitesimal comes into
        existence, and the one that is otherwise invisible.  R(0), ZERO, a raw
        Python 0 coerced by an operator, Composite(0), and a data value that
        happens to be zero all arrive here, already converted, so none of them
        ever reaches _r1 and none of them warns.  A cancellation is audible and
        a written zero is not, which is the asymmetry degeneracy_watch closes.
        """
        if _CONVENTIONAL.get() or _TAGGED.get() is not None:
            # Inert here, so no source: under conventional() a written zero is
            # a zero TERM and the derivatives are the textbook ones.  Counting
            # it would report degeneracy in the one mode that has none.
            # Under the trigger it is LATENT rather than inert -- _r1 at the
            # point of use decides, where the sibling's source is visible.
            return cls({0: 0.0})
        return _mint(cls({-1: 1.0}))

    @classmethod
    def infinity(cls):
        """Structural infinity: |1|₁"""
        return cls({1: 1.0})

    @classmethod
    def real(cls, value):
        """Real number: |value|₀"""
        if isinstance(value, Composite):
            # float(Composite) is st(), so this would drop every dimension but
            # 0 and return a plausible wrong answer.  That is how
            # power(1+x, 1/x) returned 1.0: the exponent 1/x is |1|_1, whose
            # standard part is 0, so it computed x^0.
            raise TypeError(
                "Composite.real() takes a real number, not a Composite -- "
                "converting one collapses it to its standard part. If the "
                "value is already a composite, use it directly.")
        return cls({0: float(value)})

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        dims, vals = self._backend.to_arrays(self._data)

        if len(dims) == 0:
            # FIXED: was "|0|₀" — now distinguishable from expressed |0|₀
            return "∅"

        # TWO NOTATIONS, NEVER MIXED.
        #
        #   |4|₀        the dimension is a real subscript glyph, so the bars
        #               delimit the coefficient
        #   4_1/4       the underscore does the subscripting, so the bars are
        #               redundant and are dropped
        #
        # `|2|_5/6` was both at once, which is neither.  Whether the plain form
        # is needed is a property of the WHOLE number, not of one term: a
        # dimension with no glyph anywhere in it puts every term in the
        # underscore form, so `<4_0 2_5/6>` rather than `<|4|₀ 2_5/6>`.
        sub = "₀₁₂₃₄₅₆₇₈₉"

        def _glyphless(n):
            """No subscript glyphs exist for a fraction or a vector dimension."""
            if isinstance(n, tuple):
                return True
            if isinstance(n, _Fraction):
                return n.denominator != 1
            return not float(n).is_integer()

        plain = any(_glyphless(d) for d in dims)

        def fmt_dim(n):
            if isinstance(n, tuple):
                return "(" + ",".join(f"{c:g}" for c in n) + ")"
            if isinstance(n, _Fraction):
                # EXACT, so print it exactly.  "%g" showed -0.333333 for a
                # dimension that is precisely -1/3, which reads as a rounded
                # float in the one backend whose point is that it is not one.
                return (str(n.numerator) if n.denominator == 1
                        else f"{n.numerator}/{n.denominator}")
            f = float(n)
            if not f.is_integer():
                return f"{f:g}"
            n = int(f)
            if plain:
                return str(n)
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            return "₋" + ''.join(sub[int(d)] for d in str(-n))

        def fmt_coeff(c):
            c = float(c)
            # int(nan) raises ValueError and int(inf) raises OverflowError, and
            # `c == int(c)` evaluates the int FIRST -- so printing a composite
            # that had picked up a NaN crashed instead of showing it, which is
            # the one moment you most want to look at the number.
            if not math.isfinite(c):
                return "nan" if math.isnan(c) else ("inf" if c > 0 else "-inf")
            if c == int(c):
                return str(int(c))
            return f"{c:.6g}"

        # Highest dimension first (descending)
        if plain:
            parts = [f"{fmt_coeff(vals[i])}_{fmt_dim(dims[i])}"
                     for i in range(len(dims) - 1, -1, -1)]
        else:
            parts = [f"|{fmt_coeff(vals[i])}|{fmt_dim(dims[i])}"
                     for i in range(len(dims) - 1, -1, -1)]
        # Braces and spaces, never `+`.  Zero Rules v2 section 0: `+` is an
        # operation BETWEEN composites, so using it for the terms WITHIN one
        # gives `0_3 + -4_1` two readings -- one number whose dimension-3 term
        # is empty, which stays as written, and two numbers being added, which
        # is `<-4_1 1_2>`.  Every apparent inconsistency in those rules traced
        # back to reading one as the other.
        #
        # In the glyph form the bars delimit each coefficient, so a space is
        # enough.  In the plain form a negative coefficient does sit straight
        # against the space (`<3_0 -1_-0.5>`), which is accepted: it needs a
        # fractional or vector dimension AND a negative coefficient together.
        return "<" + " ".join(parts) + ">"

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self):
        """Serialize to JSON-safe dict."""
        return {str(k): v for k, v in self.c.items()}

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dict. Accepts string, int or float keys."""
        return cls({dim_cast(float(k)): v for k, v in d.items()})

    def to_bytes(self):
        """Serialize to compact binary format."""
        import struct
        parts = []
        for dim, coeff in self.c.items():
            parts.append(struct.pack('<id', dim, coeff))
        return b''.join(parts)

    @classmethod
    def from_bytes(cls, data):
        """Deserialize from binary. Inverse of to_bytes()."""
        import struct
        c = {}
        for i in range(0, len(data), 12):
            dim, coeff = struct.unpack('<id', data[i:i+12])
            c[dim] = coeff
        return cls(c)

    def to_array(self, dims):
        """Extract coefficients at fixed dimensions as a flat list."""
        return [self.c.get(d, 0.0) for d in dims]

    @classmethod
    def from_array(cls, values, dims):
        """Reconstruct from flat list + dimension map."""
        # zero coefficients are retained: a dimension exists because it
        # was constructed, whatever it holds
        return cls({d: v for d, v in zip(dims, values)})

    def to_json(self):
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s):
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(s))

    # -------------------------------------------------------------------------
    # Arithmetic operations
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """Addition never shifts dimensions (R4).  A zero OPERAND converts (R1);
        a zero TERM does not (R2)."""
        if isinstance(other, (int, float)):
            other = _scalar_operand(other)
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        out = Composite._wrap(a._backend.add(a._data, b._data), a._backend,
                              complete=_min_complete(self, other),
                              denot=_denot_additive(a, b))
        return _cancellation_residue(a, b, out)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction never shifts dimensions (R4).

        a - a leaves |0|_d: the dimension was constructed by both operands, so
        it exists and holds zero.  No special case is needed -- |0|_a - |0|_a
        giving 0**(a+1) follows from R1 plus this rule, not from a branch here.
        """
        if isinstance(other, (int, float)):
            other = _scalar_operand(other)
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        out = Composite._wrap(a._backend.add(a._data, a._backend.negate(b._data)),
                              a._backend,
                              complete=_min_complete(self, other),
                              denot=_denot_additive(a, b))
        return _cancellation_residue(a, b, out)

    def __rsub__(self, other):
        left = _scalar_operand(other)
        return left.__sub__(self)

    def __neg__(self):
        return Composite._wrap(self._backend.negate(self._data), self._backend,
                               complete=self._complete, denot=_denot_of(self))

    def __mul__(self, other):
        """Multiplication: dimensions add, coefficients multiply.

        A zero OPERAND converts first (R1), so |0|_0 x |5|_0 = |1|_-1 x |5|_0
        = |5|_-1 -- the value is translated, not annihilated.  Multiplying by
        |1|_0 is the identity (R3).
        """
        if isinstance(other, (int, float)):
            if other == 0:
                other = Composite({0: 0.0})
            else:
                return Composite._wrap(
                    self._backend.scalar_multiply(self._data, float(other)),
                    self._backend, complete=self._complete,
                    denot=_denot_of(self))
        if not isinstance(other, Composite):
            return NotImplemented
        if _is_unit(other):
            return self
        if _is_unit(self):
            return other
        a, b = _operands(self, other)
        data, complete = _capped(a._backend,
                                 a._backend.convolve(a._data, b._data),
                                 _scaled_complete(self, other, a, b, +1))
        return Composite._wrap(data, a._backend, complete=complete,
                               denot=_denot_product(a, b))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: dimensions subtract, coefficients divide.

        A zero OPERAND converts first (R1), on either side.  Dividing by
        |1|_0 is the identity (R3).
        """
        if isinstance(other, (int, float)):
            if other == 0:
                other = Composite({0: 0.0})
            else:
                return Composite._wrap(
                    self._backend.scalar_multiply(self._data, 1.0 / other),
                    # self._backend, not the global one: __mul__ passes it and
                    # __truediv__ did not, so `x / 2` on a vector composite
                    # wrapped DictData in sparse-dense methods.  sinh, cosh,
                    # tanh and atan all divide by a scalar at the end.
                    self._backend, complete=self._complete,
                    denot=_denot_of(self))
        if not isinstance(other, Composite):
            return NotImplemented
        if _is_unit(other):
            return self

        # Under conventional() a zero divisor is an error, as it is anywhere
        # that has an additive identity.  Checked BEFORE _operands, which is
        # where R1 would have lifted it, and before the single-term path,
        # which divides coefficients by 0.0 and returns |inf|_0 silently.
        if (_CONVENTIONAL.get() or _sealed(other)) and _is_wholly_zero(other):
            raise ZeroDivisionError(
                "division by zero: a zero divisor is conventional here, under "
                "conventional() or after the first infinitesimal under "
                "single_infinitesimal(). Outside both, a zero divisor converts "
                "(R1) and division by zero is total")

        a, b = _operands(self, other)
        b_dims = b._backend.active_dims(b._data)

        if len(b_dims) == 0:
            raise LimitDoesNotExistError(
                "Division by nothing (empty composite). "
                "Denominator is indeterminate — limit does not exist.")

        # Single-term divisor -> a pure dimension shift.  Every term of the
        # dividend moves together, zeros included: they shift and stay zero.
        if len(b_dims) == 1:
            _bd = b_dims[0]
            div_dim = _bd if isinstance(_bd, (tuple, _Fraction)) else float(_bd)
            div_coeff = b._backend.read_dim(b._data, div_dim)
            my_dims, my_vals = a._backend.to_arrays(a._data)
            # Either side may carry vector dimensions.  Testing only the DIVISOR
            # missed the common half of it: ln(h) / R(2) has tuple dims above a
            # scalar below, and `my_dims - div_dim` is then tuple minus float,
            # which raises.  It surfaced on the dict backend, whose to_arrays
            # hands back the tuples as written; the other two normalise first
            # and so hid it.  Ask about the dimensions actually being shifted.
            _vector = isinstance(div_dim, tuple) or (
                getattr(my_dims, "dtype", None) == object
                and any(isinstance(d, tuple) for d in my_dims))
            if _vector:
                # Vector dimensions subtract componentwise; object arrays do not
                # broadcast `-`, so shift each one explicitly.  PAD FIRST: zip
                # over different-length tuples truncates to the shorter and
                # drops the deepest axes without a word, which turned
                # 1/ln(ln(1/h)) -- dim (0,0) over (0,0,1) -- into a plain 1.
                from composite.backends.vector_dim_backend import pair, canon
                shifted = []
                for d in my_dims:
                    _a, _b = pair(d, div_dim)
                    shifted.append(canon(tuple(x - y for x, y in zip(_a, _b))))
            elif _exact_dims(my_dims) or isinstance(div_dim, _Fraction):
                # EXACT SUBTRACTION when either side carries a Fraction.  Mixing
                # the two types does it in float and lands one ulp out:
                # Fraction(-1,3) - (-1.0) is 0.6666666666666667 where
                # float(Fraction(2,3)) is 0.6666666666666666, so the grade was
                # off by an ulp before the backend ever saw it and no round-trip
                # could recover the third.  h^(1/3) / R(0) came back at
                # dimension 3002399751580331/4503599627370496.
                shifted = [dim_fraction(d) - dim_fraction(div_dim)
                           for d in my_dims]
            else:
                shifted = my_dims - div_dim
            return Composite._wrap(
                a._backend.create_from_terms(shifted, my_vals / div_coeff),
                a._backend, complete=_scaled_complete(self, other, a, b, -1),
                denot=_denot_quotient(a, b))

        # deconvolve is lexicographic long division: it repeatedly takes
        # max(rem).  With infinitesimals on TWO independent axes that starves
        # one of them -- (0,-k) outranks (-1,anything), so the y-axis geometric
        # series (which never terminates) eats every iteration and the x-axis
        # terms sit in the remainder until the cap runs out.  1/(1+x+y) came
        # back with NO x-dependence at all.  The geometric series treats the
        # whole infinitesimal part as one object and has no such ordering, so
        # use it when the divisor genuinely spans more than one axis.
        if _spans_multiple_axes(b) and _expandable_about_st(b):
            return a * _reciprocal(b, terms=_effective_terms(15))

        result, cut = a._backend.deconvolve_cut(a._data, b._data)
        out = Composite._wrap(_truncate_dims(a._backend, result), a._backend,
                              complete=_scaled_complete(self, other, a, b, -1),
                              denot=_denot_quotient(a, b))
        # A multi-term divisor generally yields a NON-TERMINATING quotient that
        # the backend cuts at a fixed length: 1/(1-x) comes back as 50 correct
        # coefficients, not as a closed form.  Every order produced is right,
        # but there is no order beyond them -- so the quotient is complete TO
        # what it produced, never "exact".  Reporting exact here is what made
        # (1/x)*x = 1 diverge at order 50.
        # Only a CUT quotient is bounded.  One whose remainder emptied is exact,
        # and bounding it is the same mistake the other way round: x*x/x is x,
        # and a bound of 1 made exp((x*x/x)*x) drop every order from 2.
        got = [_dim_order(d) for d in out.c if _dim_order(d) >= 0]
        if got and cut:
            out._complete = _tighter(out._complete, max(got))
        return out

    def __rtruediv__(self, other):
        # other / self -- the operands are reversed, so no unit-divisor
        # short-circuit applies here.
        left = _scalar_operand(other)
        return left.__truediv__(self)

    def __abs__(self):
        """Absolute value of the standard part."""
        return abs(self.st())

    def __float__(self):
        """Float conversion returns standard part.

        Deliberately NOT the IEEE754 projection: this keeps raising on an
        unbounded composite, because that exception is what catches an
        accidental coercion through `math.*`.  See `to_ieee754`.

        Inside `_refusing_float()` it raises instead, for every composite.
        `math.cos(t)` on a seeded t calls this and silently hands back a float
        with the infinitesimal gone; a routine that reads the infinitesimal
        (a line integral's tangent) must know that happened, and a value
        comparison cannot tell it -- cos and sin take the same values at 0
        and 2*pi, so a coerced circle looks constant.
        """
        if _REFUSE_FLOAT.get():
            raise FloatCoercionError(
                "a composite was converted to float -- math.* or float() on a "
                "value carrying an infinitesimal.  Use the composite functions "
                "(composite.cos, sin, exp, ...) so the infinitesimal survives.")
        return float(self.st())

    def __int__(self):
        """Int conversion returns int of standard part."""
        return int(self.st())

    def to_ieee754(self):
        """The image of this composite in float arithmetic.

        The one well-defined way back into a system that HAS an additive
        identity.  No single substitution `h = value` does it, because the two
        halves of the dimension axis want opposite limits:

        Negative grades want `h = 0`, so that an infinitesimal becomes a true
        zero.  At the smallest representable float instead, `R(6) - R(6)` comes
        back as 2.96e-323 -- a subnormal crumb, representable because
        `6 * 5e-324` is -- and the additive identity is NOT restored.

        Positive grades want `h -> 0` from above, where a pole becomes an
        infinity, which is IEEE754's own answer for `1/0`.  At `h = 0` exactly
        they divide by zero and raise instead.

        So the projection is piecewise, keyed on `lead_order`:

            lead_order > 0    infinitesimal      ->  0.0
            lead_order == 0   bounded            ->  st()
            lead_order < 0    unbounded          ->  +-inf, by the leading sign
            no nonzero term   an expressed zero  ->  0.0
                              NOTHING            ->  nan

        NOTHING is absence rather than a value and float has no absence, so nan
        is the nearest honest answer; 0.0 would claim it was a zero.

        The sign comes from the DOMINANT term, so `ln(h)` projects to -inf and
        `ln(1/h)` to +inf, and in `1/h + ln(1/h)` the pole dominates the log.
        """
        order = self.lead_order()
        if order is None:
            # Every coefficient is zero, or there are none at all.  An
            # expressed zero IS a value and converts; NOTHING is not.
            return 0.0 if self.coeffs_dict() else float("nan")
        if order > 0:
            return 0.0
        if order == 0:
            return float(self.st())

        # Unbounded.  Read the sign off the dominant term rather than
        # `lead_dim()`, whose key may be a tuple carrying numpy scalars that
        # will not index coeffs_dict reliably.
        def _axis(dim):
            if isinstance(dim, tuple):
                return (float(dim[0]), tuple(float(e) for e in dim[1:]))
            return (float(dim), ())

        terms = [(d, v) for d, v in self.coeffs_dict().items() if v != 0.0]
        _, coefficient = max(terms, key=lambda kv: _axis(kv[0]))
        return math.copysign(float("inf"), coefficient)

    def __pow__(self, n):
        """Power: integer via repeated multiplication, otherwise exp(n*ln(self))."""
        if isinstance(n, int):
            if n == 0:
                return Composite({0: 1})
            if n < 0:
                return Composite({0: 1}) / (self ** (-n))
            result = Composite({0: 1})
            for _ in range(n):
                result = result * self
            return result
        if isinstance(n, _Fraction):
            # An exact exponent, which is what the fractional backends exist for.
            # An integral one is the int case, not a root.
            if n.denominator == 1:
                return self ** int(n)
            return _rational_power(self, n)
        if isinstance(n, float):
            return exp(Composite(n) * ln(self))
        if isinstance(n, Composite):
            return exp(n * ln(self))
        raise TypeError(
            f"Power exponent must be int, float, Fraction or Composite, "
            f"got {type(n)}")

    # -------------------------------------------------------------------------
    # Extraction methods
    # -------------------------------------------------------------------------

    def st(self):
        """Standard part: the real number this is infinitely close to.

        THE DOMINANT GRADE DECIDES, not the presence of any particular one:

          any positive grade   ->  UNDEFINED
              Unbounded, so no real number is approached.  0.0 there is not
              imprecise, it is inverted -- sin(1)/sin(h) is infinite and
              `abs(st(x)) < 1e-9` was True for it.  Testing for NEGATIVE
              grades instead was tried and misses exactly that case, because
              |0.841471|_+1 + |0.140245|_-1 has negatives too; what makes it
              unbounded is the +1 on top.  For the same reason 1 + 1/h has no
              standard part despite carrying a grade-0 term.

          otherwise            ->  coefficient at grade 0, 0.0 if absent
              An infinitesimal IS infinitely close to zero, so st(|1|_-1) and
              st(h*h) are 0.0 -- correct, and not the case above.

          empty                ->  0.0
              NOTHING, and reading it expresses the zero (R6).  Provisional:
              this is the one case that may become undefined later.
        """
        dims, vals = self._backend.to_arrays(self._data)
        nz = [d for d, v in zip(dims, vals) if v != 0.0]
        if not nz:
            return 0.0
        if any(_dim_positive(d) for d in nz):
            raise StandardPartUndefinedError(
                f"no standard part: {str(self)[:60]} is unbounded -- its "
                f"dominant grade is positive")
        return self._backend.read_dim(self._data, 0)

    def coeff(self, dim):
        """Get coefficient at specific dimension"""
        return self._backend.read_dim(self._data, dim)

    @property
    def denotation_order(self):
        """Shallowest order at which an expressed zero entered, or None.

        None means every order is the classical one.  A number k means orders
        below k are classical and orders from k are the jet of the function the
        expression DENOTES, reading an expressed zero of magnitude m at grade
        -j as m*(x-a)**j.  Carried rather than inferred because division moves
        it: residue / ZERO reaches order 0.
        """
        return _denot_of(self)

    def d(self, n=1):
        """nth derivative with respect to h, as a real number.

        The fast path is a single coefficient read: for a value whose grades
        are all non-positive integers the composite IS a Taylor series in h,
        grade -n holds the coefficient of h**n, and n! converts it.

        Everything else goes through `D`, which differentiates properly and
        then has its standard part taken -- so a derivative that is unbounded
        says so instead of coming back 0.0.  That was the one genuinely
        misleading read: grade -n is ABSENT for (2+h)/h, whose grades are 0
        and +1, and read_dim reports an absent dimension as 0.0, so d(1)
        through d(5) all returned 0.0 for a simple pole whose derivatives are
        -2/h**2, 4/h**3, ... .  Absent read as zero is the same mistake the
        R1 warning text calls a bug; it is harmless when a Taylor series
        exists, because an absent grade genuinely has coefficient zero there,
        and wrong exactly when one does not.

        A half order is the same trap without the pole: sqrt(h) is |1|_-0.5,
        every integer grade is absent, and its derivative 0.5*h**-0.5 is
        unbounded rather than zero.

        REPORTS when the order asked for is one an expressed zero contributed
        to.  The value returned is still a true derivative -- of the function
        the expression denotes -- so this warns by default rather than
        refusing; CONVENTIONAL_STRICT = True raises instead.
        """
        _dn = _denot_of(self)
        if _dn is not None and n >= _dn:
            _report_denotation(n, _dn)
        dims, vals = self._backend.to_arrays(self._data)
        for _g, _v in zip(dims, vals):
            if _v == 0.0:
                continue
            # A term that is not an integer power of h -- a pole, a fractional
            # power, or a LOG-AXIS term h**p * ln(1/h)**k -- is absent from
            # grade -n, so read_dim skips it.  Its power order p decides
            # whether that matters: for p > n its nth derivative vanishes at
            # the point and the plain read is exact; for p <= n it is part of
            # the answer, usually an unbounded one, and D is the one place that
            # knows how to take it.  Before the p test, a log term at p <= n
            # was skipped -- (h + h*ln h).d(1) returned 1, exp(2.25h*ln(4h))
            # returned 3.119 for a derivative that diverges -- and a fractional
            # term at p > n sent the read to D, which refuses log terms, so
            # h**3.35 * exp(h*ln h) could not report d(1) = 0.
            _log = isinstance(_g, tuple) and any(c != 0 for c in _g[1:])
            if _dim_positive(_g) or _fractional_power(_g) or _log:
                if -(_g[0] if isinstance(_g, tuple) else _g) <= n:
                    return self.D(n).st()
        return self._backend.read_dim(self._data, -n) * math.factorial(n)

    def D(self, n=1):
        """nth derivative with respect to h, as a Composite.

        Differentiation is a GRADE SHIFT.  A term |c|_g is c * h**(-g), so
        differentiating gives |-g*c|_(g+1): the coefficient picks up -g and
        the grade moves up one.  That single rule is total over the power
        axis -- infinitesimals, values, infinities and half orders alike:

            2/h + 1   = |2|_1 + |1|_0     ->  |-2|_2      = -2/h**2
            h**2 + 2h = |2|_-1 + |1|_-2   ->  |2|_-1 + |2|_0
            sqrt(h)   = |1|_-0.5          ->  |0.5|_0.5   = 0.5*h**-0.5
            INF       = |1|_1             ->  |-1|_2

        and it composes, so D(n) is the shift applied n times.  The factorial
        in `d` is not a separate convention bolted on: applying the shift n
        times produces n! on its own.  x*x seeded at 2 is |4|_0 + |4|_-1 +
        |1|_-2, whose D(1) is |4|_0 + |2|_-1 -- standard part 4 = f'(2), and
        its own next shift gives |2|_0 = f''(2).

        Grade 0 drops out, because a constant differentiates away.  When
        nothing survives, the answer is the EXPRESSED zero |0|_0 rather than
        nothing: the derivative of a constant is zero, and a zero here is a
        value, not an absence.

        The log axis is not a shift and is refused rather than dropped.
        d/dh of ln(1/h), which is |1|_(0,1), is -1/h -- it crosses from the
        log axis to the power axis, so it needs its own case, and until that
        case is written returning something is worse than saying so.
        `_derivative_axis` drops those terms silently, which is the mistake
        this avoids.
        """
        out = self
        for _ in range(n):
            out = _shift_derivative(out)
        return out

    def lead_dim(self):
        """The dominant dimension: the one that decides how big this number is.

        Zero coefficients are skipped -- a term that cancelled does not decide
        the size of the number -- and the comparison is the backend's dominance
        order, so a power beats a log at the same nominal order.  None when the
        composite is nothing, or holds only zeros.
        """
        dims, vals = self._backend.to_arrays(self._data)
        nz = [d for d, v in zip(dims, vals) if v != 0.0]
        if not nz:
            return None
        from composite.backends.vector_dim_backend import dom_sorted as _dom_sorted
        return _dom_sorted(nz, reverse=True)[0]     # biggest dimension = dominant term

    @property
    def conventionally_degenerate(self):
        """True once a SECOND infinitesimal source has reached this number.

        None when nothing is known: tracking was off while this number was
        built, or it descends from no infinitesimal at all and has no Taylor
        reading to corrupt.  NOT False -- reporting a silent "clean" for a
        number nobody was watching is the failure shape this exists to prevent.
        See set_degeneracy_tracking, which is off by default.

        The composite is not wrong when this is set, and nothing here refuses
        or corrects.  What stops holding is the reading of grade -n as the nth
        derivative of the function you had in mind: `x*x + R(0)` is not x
        squared, because the written zero is one unit of the seed and h is
        x - a, so the function is x**2 + x - a and the slope 5 is right.
        Dimension 0 stays correct while every derivative moves, which is the
        worst failure shape there is, so it is worth a flag.

        Carried, not computed: reading it costs nothing and it answers for a
        number you were handed, with no function left to re-evaluate.  See
        taylor_degeneracy for the one case it does not catch.
        """
        if self._deg:
            return True
        return False if self._src is not None else None

    def lead_order(self):
        """Order of the dominant term: positive infinitesimal, negative unbounded.

        The public way to ask "how big is this number", across every axis.
        Reading `max(coeffs_dict())` instead is the trap: a log-axis grade is a
        tuple, comparing it with an integer raises, and `_dim_order` reads only
        the power component, so `x*ln(x)` (grade (-1, 1)) looks like order 0.

            R(3).lead_order()          ->  0     an ordinary number
            ZERO.lead_order()          ->  1     infinitesimal, first order
            (ZERO*ZERO).lead_order()   ->  2     infinitesimal, second order
            (R(1)/ZERO).lead_order()   -> -1     unbounded, first order
            sqrt(ZERO).lead_order()    ->  0.5   a half order: a branch point
            Composite({}).lead_order() ->  None  nothing

        `lead_dim` says WHICH axis that order is on: ln(ZERO) leads on the log
        axis, so its order is read there, not on the power axis.
        """
        d = self.lead_dim()
        if d is None:
            return None
        order = _lead_order(d)
        return order + 0.0 if isinstance(order, float) else order   # never -0.0

    def _fn_result(self, result, fn):
        """Hook: `result` is what a library function made from this operand.

        A plain composite keeps it as it is, so this costs nothing.  A subclass
        that carries state beside the coefficients -- the forensics audit and
        its error form -- overrides this to re-attach that state, which is why
        every transcendental hands its result back through its argument.

        The argument is handed in PLAIN (see `_preserves_type`), so the
        function's own Taylor loops run on ordinary composites and a subclass
        never sees the library's internal arithmetic.
        """
        return result

    def _as_plain(self):
        """This composite as a plain Composite, sharing the same data."""
        if type(self) is Composite:
            return self
        plain = Composite(_data=self._data)
        plain._complete = self._complete
        plain._denot = _denot_of(self)
        return plain

    def __format__(self, fmt):
        """Support format strings by formatting the standard part."""
        if fmt:
            return format(self.st(), fmt)
        return repr(self)

    def max_positive_dim(self):
        """Return the highest positive dimension, or None if none exist."""
        dims, vals = self._backend.to_arrays(self._data)

        pos = [d for d, v in zip(dims, vals) if _dim_positive(d) and v != 0]
        if not pos:
            return None
        if any(isinstance(d, tuple) for d in pos):
            from composite.backends.vector_dim_backend import as_vec
            return max(pos, key=lambda d: as_vec(d))
        return max(dim_cast(d) for d in pos)

    def coeffs_dict(self):
        """Every term the arithmetic produced -- NOT every term vouched for.

        A product of two truncated series reaches deeper than either operand is
        known to, because the terms below that depth are the convolution
        missing everything the operands cut away.  Measured:

            exp(h, terms=5) * exp(h, terms=5)
                complete_order 4, but nine terms down to grade -8
                grade 4  0.666666666667   exp(2h) 0.666666666667   ok
                grade 5  0.25             exp(2h) 0.266666666667   WRONG

        `complete_order` says where the guarantee stops and `complete_coeffs`
        returns only that part.  The terms past it are kept deliberately rather
        than dropped in __mul__: _truncate_order reads `active_dims`, which the
        sparse-dense backend documents as a BOUNDARY operation that "must not
        be used inside add, convolve, scalar_multiply or negate -- that round
        trip is what this representation exists to remove".  Calling it on
        every product cost 2x on the suite (101s against 55s) and broke two
        resummation checks.  So the leak is reported, not hidden.
        """
        dims, vals = self._backend.to_arrays(self._data)
        return {dim_cast(d): float(v) for d, v in zip(dims, vals)}

    @property
    def complete_order(self):
        """Highest order this composite vouches for, or None if unbounded.

        None means no series truncation is recorded -- a literal, a polynomial
        built term by term, or an exact result.  It does NOT mean "complete to
        every order"; it means nothing here has claimed otherwise.
        """
        return self._complete

    def complete_coeffs(self):
        """Only the terms within `complete_order`.

        Use this wherever a wrong coefficient would be worse than a missing
        one.  With complete_order None every term is returned, because nothing
        has claimed a limit.
        """
        d = self.coeffs_dict()
        if self._complete is None:
            return d
        return {k: v for k, v in d.items() if _dim_order(k) <= self._complete}

    def leaked_coeffs(self):
        """The terms BEYOND `complete_order` -- present, and not vouched for.

        Empty whenever the composite is within its bound.  Non-empty is not an
        error: it is the arithmetic reporting that it produced more than it can
        stand behind.
        """
        if self._complete is None:
            return {}
        return {k: v for k, v in self.coeffs_dict().items()
                if _dim_order(k) > self._complete}

    # -------------------------------------------------------------------------
    # Simplified integration operators (dimensional shifts)
    # -------------------------------------------------------------------------

    def eval_taylor(self, h_value):
        """The INCREMENT f(x0 + h_value) - f(x0), by substituting h -> h_value.

        Grade 0 is deliberately excluded, so this is the change and NOT the
        value.  Every caller wants it that way: `integrate_step` applies it to an
        antiderivative, whose grade-0 term is an arbitrary constant; the lane
        extrapolation at _probe writes `st + result.eval_taylor(-eps)`, adding the
        standard part back itself; and the tutor takes a difference of two calls,
        where a constant would cancel regardless.

        For the VALUE, add the standard part: `c.st() + c.eval_taylor(d)`.  The
        old one-line docstring said "evaluate Taylor polynomial", which promises
        the value and silently delivers a result short by exactly f(x0).
        """
        dims, vals = self._backend.to_arrays(self._data)
        _p = lambda d: d[0] if isinstance(d, tuple) else d
        return sum(float(v) * h_value ** (-float(_p(d)))
                   for d, v in zip(dims, vals) if _p(d) < 0)

    def eval_taylor_axes(self, h_value):
        """eval_taylor, but the NON-power axes survive as dimensions.

        eval_taylor substitutes h and returns a FLOAT, which adds terms that
        differ only on a later axis into the same number: (-1, 0) and (-1, -1)
        both become coefficient * h**1.  That is correct when the power axis is
        the only one in play, and it is what collapses a box integral -- the
        second variable's structure is destroyed at exactly this step, not at
        .st() as it appears.

        Returns {dim: coeff} with the power component consumed and every other
        component kept, so the caller still has a series in the remaining
        variables.  eval_taylor is left alone: its callers want the float.
        """
        dims, vals = self._backend.to_arrays(self._data)
        out = {}
        for d, v in zip(dims, vals):
            p = d[0] if isinstance(d, tuple) else d
            if p >= 0:
                continue
            if isinstance(d, tuple):
                from composite.backends.vector_dim_backend import canon
                key = canon((0,) + tuple(d[1:]))
            else:
                key = 0
            out[key] = out.get(key, 0.0) + float(v) * h_value ** (-float(p))
        return out

    def integrate_step(self, dx):
        """Integrate over interval [x, x+dx]."""
        Fx = antiderivative(self)
        return Fx.eval_taylor(dx)

    # -------------------------------------------------------------------------
    # Comparison (lexicographic by dimension)
    # -------------------------------------------------------------------------

    def __eq__(self, other):
        """Identity, not magnitude.  An expressed zero is part of the number.

        This used to return `_compare(self, other) == 0`, which is the tie case
        of the dominance ORDER.  That order reads an absent dimension and a
        zero coefficient alike -- `read_dim` returns 0.0 for both -- so

            (1 + ZERO) - ZERO   ==   R(1)        was True

        even though the left side carries a cancelled infinitesimal at grade -1
        and the right side never had one.  R2 retains that term deliberately;
        equality then threw it away, which is the one thing a provenance-
        preserving arithmetic must not do.

        R1 is still applied first, so the two spellings of a zero remain ONE
        number: |0|_d and |1|_(d-1) compare equal, as they always did.

        The cost, stated because it is real: `==` is no longer the tie case of
        `<` and `>`.  Two composites can have identical magnitude at every
        scale -- neither dominates -- and still be unequal.  Code shaped
        `if x == y: ... elif x < y: ...` takes a different branch than before.
        Z7 pins that as a property rather than leaving it to be discovered.
        """
        if isinstance(other, (int, float)):
            other = Composite(other)
        if not isinstance(other, Composite):
            return NotImplemented
        a, b = _operands(self, other)
        return a.coeffs_dict() == b.coeffs_dict()

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) < 0

    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) <= 0

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) > 0

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) >= 0

    def __ne__(self, other):
        """Strictly the negation of __eq__.

        This had its own _compare call, so when __eq__ moved from magnitude to
        identity this kept answering the old question: `a == b` was False and
        `a != b` was False at the same time, for the very case the change
        exists to catch.  Python does not derive __ne__ when __eq__ is
        overridden on a class that already defines it, so it has to delegate
        explicitly.
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

def _compare(a, b):
    """Lexicographic comparison by dimension (highest first).

    Operands are read through R1 first, so the two spellings of a zero compare
    equal: |0|_d and |1|_(d-1) are one number, and <0|_-1> == ZERO**2.
    """
    # _operands, not _r1 alone: comparing across representations needs the
    # same promotion arithmetic gets.  Without it np.union1d below sorts a
    # float dim array against a tuple one and raises -- so log(1/r) could not
    # be ordered against 1/sqrt(r), which is exactly the asymptotic question
    # the ordering exists to answer.
    a, b = _operands(a, b)
    a_dims, a_vals = a._backend.to_arrays(a._data)
    b_dims, b_vals = b._backend.to_arrays(b._data)

    # Key every dimension as a COMMON-WIDTH vector and compare those.
    #
    # _operands promotes on the DATA TYPE, and VectorDimBackend subclasses
    # DictBackend, so on the dict backend both sides hold DictData, the
    # promotion is skipped, and the dims stay float on one side and tuple on the
    # other.  np.union1d then sorted a tuple against a float and raised, so
    # `1/h > ln(1/h)` answered on sparse-dense and was a TypeError on dict --
    # the asymptotic comparison the ordering exists to answer, failing on one
    # backend only.
    #
    # canon() is NOT the tool here: it strips trailing zeros, so canon((1,0))
    # collapses back to the scalar and the mixed kinds return.  At equal width
    # Python's own tuple order IS the dominance order -- that is what dom_key
    # documents -- so pad instead of canonicalising, and read the coefficients
    # out of the arrays already in hand rather than through read_dim, which
    # would need the stored key form back again.
    from composite.backends.vector_dim_backend import as_vec
    _w = max([len(d) if isinstance(d, tuple) else 1
              for d in list(a_dims) + list(b_dims)] or [1])
    ka = {as_vec(dim_cast(d), _w): float(v) for d, v in zip(a_dims, a_vals)}
    kb = {as_vec(dim_cast(d), _w): float(v) for d, v in zip(b_dims, b_vals)}

    all_keys = set(ka) | set(kb)
    if not all_keys:
        return 0

    for key in sorted(all_keys, reverse=True):
        ca = ka.get(key, 0.0)
        cb = kb.get(key, 0.0)
        if math.isnan(ca) or math.isnan(cb):
            return float('nan')
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0


# =============================================================================
# CONVENIENCE SHORTCUTS
# =============================================================================

def R(x):
    """Create real number |x|₀, or the EXPRESSED zero |0|₀ if x == 0.

    |0|₀ and |1|₋₁ are the same number (zero) at different dimensions, and R(0)
    hands back the un-converted one so that R1 converts it at the point of USE,
    exactly as a raw Python `0` is converted.

    R(0) used to return |1|₋₁ already converted, which skipped _r1 entirely --
    and _r1 is where the R1 warning fires and where `_denot` is recorded. The
    consequence was that `x*x + R(0)` silently read d(1) = 7 instead of 6 with no
    warning and `denotation_order is None`, while `x*x + 0`, with identical
    semantics, warned twice and recorded the order. The explicit spelling was the
    undisclosed one.

    ZERO is NOT built through here. It names the infinitesimal rather than being a
    written zero, so it stays |1|₋₁ and carries no denotation.
    """
    if isinstance(x, Composite):
        raise TypeError(
            "R() takes a real number. Passing a Composite silently collapses it "
            "to its standard part -- float(|2|_0 + |1|_-1) is 2.0 -- which is how "
            "power(1+x, 1/x) returned 1.0. If the value is already a composite, "
            "use it directly.")
    if x == 0:
        return Composite({0: 0.0})     # latent: _r1 converts it at the point of use
    return Composite.real(x)

ZERO = Composite.zero()       # |1|₋₁ (infinitesimal)
INF = Composite.infinity()    # |1|₁ (infinity)
h = ZERO                      # Alias: h is the infinitesimal


# -----------------------------------------------------------------------------
# Degeneracy tracking: the switch
# -----------------------------------------------------------------------------

def set_degeneracy_tracking(on):
    """Turn conventional-derivative degeneracy tracking on or off.  Returns the
    PREVIOUS setting, so a caller can restore it.

    Off by default.  It matters only when you are going to read grade -n as the
    nth derivative of a function you have in mind; for arithmetic, limits, root
    finding or anything that reads dimension 0, it answers a question nobody
    asked and costs 11-21%.

    Switching is a rebind of the operators on the class, not a branch inside
    them, so OFF costs nothing rather than a little: there is no wrapper to
    enter.  That also means it is global and not thread-isolated, like
    set_backend.  Prefer track_degeneracy() to setting it by hand.
    """
    on = bool(on)
    previous = _TRACKING[0]
    if on != previous:
        for _name in _CARRIED_OPS:
            _m = getattr(Composite, _name)
            setattr(Composite, _name,
                    _carries(_m) if on else _m.__wrapped__)
        _TRACKING[0] = on
    return previous


def degeneracy_tracking():
    """Is tracking on?  A composite built while it was off carries no verdict."""
    return _TRACKING[0]


@_contextlib.contextmanager
def track_degeneracy(on=True):
    """Track degeneracy for the duration.

        with track_degeneracy():
            y = f(_seeded(2.0))
        y.conventionally_degenerate

    Wrap the WHOLE calculation, seed included.  A number built outside the
    block carries nothing, and reads back as None rather than as clean.
    """
    previous = _TRACKING[0]
    set_degeneracy_tracking(on)
    try:
        yield
    finally:
        set_degeneracy_tracking(previous)

def _refresh_constants():
    """Rebuild the module constants under the active backend.

    Called by backends.config.set_backend().  ZERO / INF / h are built at
    import time, so without this a later set_backend() would leave them on the
    old backend and every expression touching them would mix representations.
    """
    global ZERO, INF, h
    ZERO = Composite.zero()
    INF = Composite.infinity()
    h = ZERO


MAX_ACTIVE_DIMS = 60


def _order_cap(backend, data):
    """Drop terms below the order the caller asked for with `set_max_order`.

    The knob used to reach one backend's convolve and nothing else, so on dict
    and dense-series it set an attribute nobody read, and even on sparse-dense
    it left the series a transcendental had just built untouched.  Asking for
    order 1 and paying for order 50 is not a constant tax: every operand keeps
    the terms, so each multiply is a 50x50 convolution instead of 2x2, and in an
    iterated computation the count grows at every step.  Measured on a fixed
    point: 45x, and rising with the iteration count.

    Applied here because this is the choke point every product already passes
    through.  Only terms BELOW dimension -max_order go; infinities and the
    standard part are never touched.
    """
    cap = getattr(backend, "max_order", None)
    if cap is None:
        return data
    dims, vals = backend.to_arrays(data)
    keep = [i for i, d in enumerate(dims) if _dim_order(d) <= cap]
    if len(keep) == len(dims):
        return data
    idx = np.array(keep, dtype=int)
    _order_cap.bit = True          # something was dropped; the caller tightens
    _warn_truncated(len(dims) - len(keep), "set_max_order(%s)" % cap, cap)
    return backend.create_from_terms(dims[idx], vals[idx])


_order_cap.bit = False


def _capped(backend, data, complete):
    """Truncate, and report the cap in the completeness bound.

    A dropped term is exactly what `complete_order` exists to record: the
    result is right only to the order the cap left standing, and saying so is
    the difference between a cheaper answer and a quietly wrong one.
    """
    _order_cap.bit = False
    out = _truncate_dims(backend, data)
    if _order_cap.bit:
        complete = _tighter(complete, getattr(backend, "max_order", None))
        _order_cap.bit = False
    return out, complete


def _truncate_dims(backend, data):
    """Keep the MAX_ACTIVE_DIMS dimensions closest to dim 0.

    Prevents dimension explosion in deep composition chains.
    Without this, sin(atan(sin(atan(x)))) produces 100k+ dims
    with overflow that corrupts the derivative tower.
    """
    data = _order_cap(backend, data)
    # Check the count BEFORE materialising: this runs on every multiply, and
    # the overwhelming majority of products are under the cap, so flattening
    # first meant paying for the flat form purely to discover it was not needed.
    if backend.term_count(data) <= MAX_ACTIVE_DIMS:
        return data
    _warn_truncated(backend.term_count(data) - MAX_ACTIVE_DIMS,
                    "MAX_ACTIVE_DIMS = %d" % MAX_ACTIVE_DIMS, None)
    dims, vals = backend.to_arrays(data)
    # Sort by distance from dim 0, keep closest.
    if dims.dtype == object:
        # VECTOR dimensions: np.abs cannot take a tuple.  Rank by magnitude in
        # dominance order -- the power component first, then the log -- so the
        # terms kept are the ones nearest dimension zero in the same sense the
        # comparison uses.  Without this the whole truncation path raised
        # TypeError the moment a log-scale composite grew past the cap, which
        # is only reachable at the DEFAULT cap and so went unseen while every
        # standalone check ran with MAX_ACTIVE_DIMS raised.
        # A scalar dimension IS the vector (d, 0, ...), so pad it rather than
        # assuming every object-array entry is iterable.  An object array holds
        # tuples for the log axes AND Fractions for an exact fractional grade,
        # and `tuple(abs(c) for c in Fraction(1,3))` raises "'Fraction' object
        # is not iterable" -- which crashed every product that grew past the cap
        # on a fractional backend, reached by (h**3 + h**4) ** Fraction(1,2)
        # squared back.
        def _mag(d):
            return tuple(abs(c) for c in d) if isinstance(d, tuple) else (abs(d),)
        order = np.array(sorted(range(len(dims)), key=lambda i: _mag(dims[i])))
    else:
        order = np.argsort(np.abs(dims))
    keep = order[:MAX_ACTIVE_DIMS]
    keep.sort()  # restore dimension order
    return backend.create_from_terms(dims[keep], vals[keep])


def _exact_dims(dims):
    """Does this dimension array carry a Fraction, i.e. an exact fractional grade?"""
    return (getattr(dims, "dtype", None) == object
            and any(isinstance(d, _Fraction) for d in dims))


def _seeded(at):
    """Evaluation point seeded with infinitesimal for derivative extraction.

    R(0) = ZERO already carries the infinitesimal, so R(0) + ZERO
    would double-seed to |2|₋₁. This helper avoids that.

    Under conventional() R(0) is |0|₀, a zero TERM, so there is nothing to
    double and nothing to avoid: the seed is built directly at both ends and
    the special case disappears.

    THE SEED IS BUILT DIRECTLY, never through R(0).  R(0) is Composite.zero(),
    which counts as an infinitesimal SOURCE, and the seed is the extraction's
    own infinitesimal rather than one the caller's formula introduced.  Going
    through R(0) made seeding AT THE ORIGIN indistinguishable from injecting a
    zero, so `x * _seeded(0.0)` and any nested differentiation at 0 came back
    flagged as conventionally degenerate when nothing had been injected at all.
    ZERO is a module constant and costs nothing, so only the at == 0 end was
    affected.
    """
    # The seed is the first infinitesimal to enter, and it is a FRESH source
    # every time.  Minting last, over the result, is deliberate: `R(at) + ZERO`
    # would otherwise carry ZERO's id, and ZERO is a module constant built once
    # at import, so two separate seeds would share one id and `x * _seeded(3.0)`
    # would not flag.
    if _CONVENTIONAL.get():
        return _mint(Composite({0: float(at), -1: 1.0}))
    if at == 0:
        return _mint(Composite({-1: 1.0}))
    return _mint(R(at) + ZERO)

@_contextlib.contextmanager
def _extraction_scope():
    """An extraction is transparent to any degeneracy count around it.

    derivative(), the integrators and the limit routines all build a seed, read
    a float off the result and throw the composite away.  That seed never
    reaches the caller's expression, so counting it made every nested extraction
    look like a second infinitesimal: `x * R(derivative(sin, 0.0))` came back
    flagged when nothing had entered the outer calculation at all.

    Restoring the count on exit is the difference between an infinitesimal that
    ENTERS an expression and one that is consumed inside a self-contained
    calculation.  A bare _seeded() whose result the caller goes on to use is the
    first kind and still counts.
    """
    saved = _inf_sources[0]
    try:
        yield
    finally:
        _inf_sources[0] = saved


@_contextlib.contextmanager
def _full_order():
    """Suspend the caller's order cap for one library routine.

    `set_max_order` is an economy for the caller's own arithmetic: it says "I
    only want this many orders back".  It is not a statement about how much
    depth the library may use internally to produce them.  Integration panels,
    limits and series solvers all build deep intermediate expansions and then
    read one number off the end, so a cap applied to them does not return a
    coarser answer -- it returns a quietly wrong one (integral of e^-x over
    [0,1] came back 1.3e-06 low, with nothing raised).

    Derivative extraction handles this in `_derivative_scope`, which lifts the
    cap to the order requested.  Here there is no order to ask for, so the cap
    is lifted entirely and restored afterwards.
    """
    backend = get_backend()
    old = getattr(backend, "max_order", None)
    if old is not None:
        backend.max_order = None
    _saved_sources = _inf_sources[0]   # see _extraction_scope
    try:
        yield
    finally:
        _inf_sources[0] = _saved_sources
        if old is not None:
            backend.max_order = old


@_contextlib.contextmanager
def _dim_headroom(n):
    """Raise the dimension cap to hold `n` terms, then put it back.

    Same trade as _derivative_scope: the cap exists for deep composition chains
    that would otherwise reach 100k+ dimensions, and it stays in force for
    everything that did not ask for a specific depth.
    """
    global MAX_ACTIVE_DIMS
    old = MAX_ACTIVE_DIMS
    MAX_ACTIVE_DIMS = max(old, n + 8)      # + 8 for intermediate products, as
    try:                                   # _derivative_scope uses order + 8
        yield
    finally:
        MAX_ACTIVE_DIMS = old


def _honours_terms(fn):
    """An explicit `terms=` is a REQUEST, not a suggestion.

    MAX_ACTIVE_DIMS is a safety default against dimension explosion, and as a
    default it is right: a caller who said nothing about depth gets the guard.
    A caller who wrote `terms=300` has said something about depth, and silently
    handing back 60 answers a question they did not ask.  Measured before this
    existed: sqrt(h - h*h, terms=300) and terms=1000 returned byte-identical
    60-term series, so a convergence study flat-lined at 3.24e-03 and looked
    like a property of the mathematics.

    _derivative_scope already took this position for extraction -- "a global
    order cap set by the caller is an economy, not a ceiling on what an
    extraction may ask for" -- and this is the same rule for a direct call.

    No-op unless the request exceeds the cap, so the default path is untouched:
    every terms= default in this module is 12 or 15 against a cap of 60.
    """
    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        asked = kwargs.get("terms")
        if asked is None and len(args) > 1:
            asked = args[1]
        asked = max(asked or 0, _min_terms[0])
        if asked <= MAX_ACTIVE_DIMS:
            return fn(*args, **kwargs)
        with _dim_headroom(asked):
            return fn(*args, **kwargs)
    return wrapper


def _needs_full_order(fn):
    """Run `fn` with the caller's order cap suspended."""

    def wrapper(*args, **kwargs):
        with _full_order():
            return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


def set_max_order(n: int = None):
    """Set global MAX truncation order."""
    backend = get_backend()
    backend.max_order = n


def get_max_order() -> int:
    """Get current MAX truncation order. None = unlimited."""
    backend = get_backend()
    return getattr(backend, 'max_order', None)


# Global minimum terms for transcendentals. Set by nth_derivative/all_derivatives
# so that Taylor expansions inside black-box functions use enough terms.
_min_terms = [0]

def _rational_power(x, r):
    """x ** r for an exact rational r, by factoring out the leading term.

    Write x = c0 * h**(-m) * (1 + u), where m is the DOMINANT dimension and u is
    what is left after normalising.  Then

        x**r = c0**r * h**(-m*r) * (1 + u)**r

    with (1+u)**r the binomial series.  m*r is Fraction arithmetic, so the grade
    is exact for any denominator, and `ln` is never called.

    THIS REPLACES TWO BROKEN PATHS.  The old code was exact only for a composite
    with exactly ONE expressed term, and fell through to exp(float(r) * ln(x))
    otherwise, which failed two ways:

      - ln needs a positive standard part, so an expressed zero BELOW the leading
        term raised.  |1|_-1 ** 1/3 worked and |1|_-1 + |0|_-2 ** 1/3 raised,
        for the same number h differing only in which inert zero was written.
      - float(r) threw the exactness away, so (h**3 + h**4) ** Fraction(1,5)
        came back at grade 1351079888211149/2251799813685248 instead of 3/5.
        Dyadic exponents survived by luck, which is why 1/2 always looked fine.

    A monomial makes u NOTHING and the loop exits at once.  That is tested for
    rather than fallen into: it used to arrive by `q - 1` cancelling to an
    inert zero, and a cancellation is no longer inert.
    """
    lead = x.lead_dim()
    if lead is None:
        raise ZeroDivisionError(
            f"{x} ** {r}: a fractional power of nothing, or of a value that is "
            f"wholly zero, has no leading term to factor out")
    c0 = x._backend.read_dim(x._data, lead)

    # c0 ** r must be real.  A negative coefficient has a real q-th root only
    # for odd q; for even q the answer is complex and saying so beats returning
    # a nan that propagates silently.
    if c0 < 0:
        if r.denominator % 2 == 0:
            raise ValueError(
                f"({c0}) ** {r} is not real: an even denominator has no real "
                f"root of a negative coefficient")
        scale = -((-c0) ** float(r)) if r.numerator % 2 else (-c0) ** float(r)
    else:
        scale = c0 ** float(r)

    head = lead[0] if isinstance(lead, tuple) else lead
    new_head = dim_fraction(head) * r
    new_dim = ((new_head,) + tuple(lead[1:])) if isinstance(lead, tuple) else new_head

    # u = x / (c0 * h**(-lead)) - 1.  Dividing by a single-term composite is a
    # pure dimension shift, so this stays exact.
    #
    # ASK BEFORE SUBTRACTING.  An exact monomial divides out to |1|_0, and
    # `q - 1` is then a cancellation -- which converts AT THE SITE and deposits
    # |1|_-1, so "nothing left to expand" became "expand h" and the binomial
    # series ran its full 15 terms.  Measured: ZERO ** 1/2 came back at grade
    # -31/2 where it is -1/2, and -1/2 - 15 is exactly that.  NOTHING is what
    # the loop below needs in order to exit at once, and it is also what an
    # exact monomial genuinely has left over, so it is built rather than
    # arrived at by cancelling.
    monomial = Composite({lead: c0})
    q = x / monomial
    u = Composite({}) if _is_unit(q) else q - Composite({0: 1.0})

    acc = Composite({0: 1.0})
    term = Composite({0: 1.0})
    coef = _Fraction(1)
    for k in range(1, _effective_terms(15) + 1):
        coef = coef * (r - (k - 1)) / k
        term = term * u
        if not term.c or all(v == 0.0 for v in term.c.values()):
            break
        acc = acc + term * float(coef)
    return acc * Composite({new_dim: scale})


class Degeneracy:
    """Whether a Taylor reading of this result is conventionally degenerate.

    The composite is never wrong.  What can be wrong is reading grade -n as the
    nth derivative of the function you had in mind, and that reading holds only
    when the seed is the ONLY thing that wrote to those grades.

    `x*x + R(0)` is not x squared.  The written zero is an infinitesimal, and it
    is one unit of the seed, so the expression is x**2 + h, and h is x - a.  The
    function is x**2 + x - a and its derivative at 2 is 5.  The composite
    returning 5 is correct.  It is only wrong against x**2, which nobody wrote.

    So this does not refuse and does not correct.  It reports that a second
    source entered, which is the one fact a caller expecting textbook
    derivatives cannot otherwise recover: dimension 0 stays right while every
    derivative moves, and a well-formed wrong number is the worst failure shape
    there is.
    """

    __slots__ = ("sources", "value", "at", "flag")

    def __init__(self, sources, value=None, at=None, flag=None):
        #: How many sources were born inside the watch.  Observability only:
        #: the verdict does not consult it.  See degeneracy_watch.
        self.sources = sources
        self.value, self.at = value, at
        #: The verdict, carried by the result.  None when there was no result
        #: to read it off, and the count then stands in.
        self.flag = flag

    #: The first infinitesimal to enter a calculation is the one the
    #: derivatives are read against, whether or not anything called it a seed.
    #: From the second onward the grades are still exact and no longer answer
    #: the conventional question.
    FIRST_IS_FREE = 1

    @property
    def degenerate(self):
        # A FLAG, NOT A COUNT.  Knowing a third source arrived says nothing the
        # second did not already say: the conventional reading is off either
        # way.  The count survives one step behind it, for degeneracy_watch
        # around code that produces no composite to read.
        if self.flag is not None:
            return self.flag
        return self.sources > self.FIRST_IS_FREE

    def __bool__(self):
        return self.degenerate

    def __repr__(self):
        return "<Degeneracy degenerate=%s sources=%d>" % (self.degenerate, self.sources)

    def __str__(self):
        if not self.degenerate:
            return ("one infinitesimal entered, so grade -n is the nth derivative "
                    "in the ordinary sense")
        return (
            "a second infinitesimal entered. The first is the one the derivatives "
            "are read against; the other puts the expression AS WRITTEN at a "
            "distance from the function it resembles. The grades are correct; "
            "reading them as conventional derivatives is not. Re-evaluate under "
            "conventional() for the textbook reading.")


@_contextlib.contextmanager
def degeneracy_watch():
    """Count infinitesimal sources created inside the block.

    Yields a callable returning the count so far, so nesting works: each level
    reads its own delta and an inner watch does not consume the outer one's.

    COUNTED FROM ZERO, AND THE FIRST IS FREE.  Whichever infinitesimal enters a
    calculation first is the one the derivatives are read against, whether or
    not anything called it a seed.  From the second onward the grades stay exact
    and stop answering the conventional question.  So this works with an
    explicit seed and without one: seed it and the seed is number one, or do not
    and the first zero you write is.

    SOURCES, NOT TERMS.  A composite can carry a hundred infinitesimal terms and
    be perfectly ordinary, because they are all powers of the first one.  There
    are three ways an infinitesimal enters: _seeded(), Composite.zero() and an
    R1 conversion.  All three are counted, so nothing that merely propagates an
    existing infinitesimal is counted again.

    Measured on the committed library: ten ordinary expressions, including
    sin(exp(sqrt(x))) and exp(x)/(1+x*x), create zero sources; x*x + R(0),
    x*x + 0.0, x*x + (x-x) and R(0)*x**3 + x*x create exactly one each.

    Not counted, and worth knowing: a composite written directly with a non-zero
    grade, and 1/INF, both introduce grade content without passing either point.
    Neither arises in ordinary use and both would need their own instrumentation.

    The count is a module global, so it is not thread safe and it belongs around
    one evaluation rather than around a long-lived section.
    """
    start = _inf_sources[0]
    yield lambda: _inf_sources[0] - start


def _makes_own_infinitesimal(f, at):
    """Does f produce infinitesimal content when given NO seed?

    Evaluated at a plain real, so anything infinitesimal in the result is the
    formula's own, whatever route it took in.  That is the whole advantage over
    counting origin events, which can only see the routes it was told about.

    Sampled at a few points rather than proved, because a formula could make its
    zero only somewhere else; the same limitation `_integrand_needs_lane`
    carries.  Returns None when no probe point could be evaluated at all.
    """
    seen_any = False
    for pt in ([at] if at else []) + [1.0, 2.0, 0.5]:
        try:
            y = _ensure_composite(f(R(float(pt))))
        except Exception:
            continue
        seen_any = True
        if any(_lead_order(d) != 0 and v != 0.0 for d, v in y.coeffs_dict().items()):
            return True
    return False if seen_any else None


def taylor_degeneracy(f, at, name="f"):
    """Evaluate f and report whether its Taylor reading is conventionally valid.

    The verdict is carried by the result itself, so this is one evaluation and
    the reading is off the number: see _mint and _join.  Every earlier scheme
    here inspected the OPERANDS of a calculation, and an infinitesimal arrives
    as the RESULT of one just as often -- `x*x + (x - x)` has no infinitesimal
    operand anywhere and two sources by the end.

    KNOWN GAP.  A source is recognised where it is born, and a composite
    written by hand as Composite({-1: 1.0}) is not born at any of the three
    places: it arrives already graded, with nothing to observe.  Recognising it
    would mean scanning the coefficients of every composite ever constructed.
    So `x*x + Composite({-1: 1.0})` and `x*x + R(1)/INF` read clean and are not.
    A written zero -- R(0), ZERO, a raw 0, a cancellation -- is caught, and that
    is how a second source is spelled in practice.
    """
    # Tracking on for the duration whatever the global setting is: this
    # function exists to answer the question tracking answers, so switching it
    # off globally must not silently turn the answer into "clean".
    with track_degeneracy(), degeneracy_watch() as count:
        x = _seeded(at)              # inside the watch: the seed is source one
        try:
            value = _ensure_composite(f(x))
        except Exception:
            value = None
        n = count()
    flag = bool(value is not None and value._deg)
    return Degeneracy(n, value, at, flag=flag)


def _effective_terms(default):
    """Return max(default, global minimum) for Taylor series length."""
    return max(default, _min_terms[0])


@_contextlib.contextmanager
def _derivative_scope(order, terms):
    """Raise expansion depth AND the dimension cap for one extraction.

    _truncate_dims keeps the MAX_ACTIVE_DIMS dimensions CLOSEST TO ZERO, which
    is the right policy for a derivative jet -- but it also caps the reachable
    order at MAX_ACTIVE_DIMS - 1, and above that the extraction returned 0.0
    with no error at all.  Measured: order 59 correct, order 60 silently wrong.

    So raise the cap to cover what was actually asked for and restore it after.
    The guard stays in force for everything outside a derivative call, which is
    what it was written for (deep composition chains like sin(atan(sin(atan(x))))
    that otherwise reach 100k+ dims).
    """
    global MAX_ACTIVE_DIMS
    backend = get_backend()
    old_min, old_cap = _min_terms[0], MAX_ACTIVE_DIMS
    old_order = getattr(backend, "max_order", None)
    _min_terms[0] = max(terms, order + 2)
    MAX_ACTIVE_DIMS = max(MAX_ACTIVE_DIMS, order + 8)
    # A global order cap set by the caller is an economy, not a ceiling on what
    # an extraction may ask for: with cap 2, nth_derivative(..., n=4) returned
    # 0.0 and said nothing.  Lift it to cover the order requested and restore it
    # after, exactly as the dimension cap above is handled.
    if old_order is not None:
        backend.max_order = max(old_order, order + 2)
    _saved_sources = _inf_sources[0]   # see _extraction_scope
    try:
        yield
    finally:
        _inf_sources[0] = _saved_sources
        _min_terms[0] = old_min
        MAX_ACTIVE_DIMS = old_cap
        if old_order is not None:
            backend.max_order = old_order

# =============================================================================
# TAYLOR SERIES FOR TRANSCENDENTAL FUNCTIONS
# =============================================================================

def _has_positive_dims(x):
    """Check if a composite has any positive-dimension components."""
    return x.max_positive_dim() is not None


def _infinite_part_is_logarithmic(x):
    """Every infinite term of x sits on a log axis, none on the power axis.

    Such a value grows like ln(1/h), not like 1/h, so exp of it is a power of h
    rather than a transmonomial outside the value group.
    """
    dims, vals = x._backend.to_arrays(x._data)
    inf = [d for d, v in zip(dims, vals) if v != 0.0 and _dim_positive(d)]
    return bool(inf) and all(isinstance(d, tuple) and d[0] == 0 for d in inf)


def _dim_nonzero(d):
    """Is this dimension anything other than THE zero dimension?

    `d != 0` is the scalar spelling, and a tuple is never equal to 0 -- so the
    vector zero (0, 0) tested as non-zero, sqrt took its leading-term branch,
    divided x by 1, and recursed on itself until the stack ran out.
    """
    if isinstance(d, tuple):
        return any(c != 0 for c in d)
    return d != 0


def _zero_dim_like(x):
    """The zero dimension in x's own KIND -- 0, or (0, 0, ...) for vectors.

    sorted() cannot order a tuple against an int, so a scalar 0 mixed into a
    dict of vector dims raises on construction.
    """
    for d in x.c:
        if isinstance(d, tuple):
            return tuple(0 for _ in range(len(d)))
    return 0


def _dim_shift(d, k):
    """Move a dimension by k on the POWER axis, leaving the log axes alone.

    Integration and differentiation change the order in x, which is the power
    component; the log components ride along unchanged.  `d - 1` is the scalar
    spelling and raises "unsupported operand for -: 'tuple' and 'int'" as soon
    as a log axis is present -- which is what stopped erf on a log-axis
    argument.
    """
    if isinstance(d, tuple):
        return (d[0] + k,) + tuple(d[1:])
    return d + k


def _dim_scaled(d, factor):
    """Scale a dimension by `factor`, componentwise for a vector dim.

    sqrt halves the index; for (p, l) that is (p/2, l/2), since
    sqrt(h**p * L**l) = h**(p/2) * L**(l/2).  Writing `d / 2` works only for a
    scalar and raises "unsupported operand for /: 'tuple'" the moment a log
    axis is involved.
    """
    if isinstance(d, tuple):
        return tuple(c * factor for c in d)
    return dim_cast(d * factor)


def _dim_positive(d):
    """Dimension lexicographically ABOVE zero -- the infinite side, any depth.

    `d > 0` is the scalar spelling of this and it raises TypeError the moment a
    dimension is a tuple.  The first NON-ZERO component decides: (0,1) is
    log(1/h), infinite; (0,0,-1) is 1/loglog(1/h), infinitesimal.  Reading only
    the power component -- the previous shortcut -- calls every pure log term
    finite, which routes infinite arguments down the Taylor path.
    """
    if not isinstance(d, tuple):
        return d > 0
    for c in d:
        if c != 0:
            return c > 0
    return False


def _dim_negative(d):
    """Dimension lexicographically BELOW zero -- the infinitesimal side."""
    if not isinstance(d, tuple):
        return d < 0
    for c in d:
        if c != 0:
            return c < 0
    return False


def _dim_order(d):
    """Taylor order carried by a dimension: -power, scalar or vector dim alike.

    int(d) is wrong here -- dimensions are float64 and int(-0.5) == 0, which is
    how a fractional dimension once read as 'no infinitesimal part' and sent ln
    down the wrong branch.
    """
    p = d[0] if isinstance(d, tuple) else d
    return -p


def _complete_order(h_terms, terms):
    """Highest Taylor order the h-power loop actually finishes.

    h**n starts at order n*m, where m is the LOWEST order present in h.  After
    forming powers 1..terms-1, every order at or below m*(terms-1) has received
    all of its contributions; every order above it is missing the contributions
    of the powers never formed.

    A term counter cannot see this distinction -- only the dimension can.  When
    h spans a single order (h = e, the common case: exp(_seeded(t))) the bound
    is terms-1 and nothing is discarded.  When h spans several -- which is what
    any composed argument gives you, exp(-(x*x)) having h = -2*mid*e - e**2 --
    the loop reaches orders it cannot complete, and returning them handed back
    partial sums wearing the shape of finished coefficients: measured against
    exact Hermite values, wrong by a factor of ~1e3.  They were invisible
    because taylor_coefficients raises the depth through _derivative_scope and
    so never reads past the boundary; the consumers that sweep the whole
    coefficient dict -- antiderivative, and integrate through it -- did.
    """
    if not h_terms:
        return None          # nothing infinitesimal: no order bound to state
    m = min(_lead_order(d) for d in h_terms)
    if m <= 0:
        return None
    return m * (terms - 1)


def _lead_order(d):
    """Order along the DOMINANT axis -- positive for an infinitesimal.

    _dim_order reads only the POWER component and returns 0 for anything living
    purely on a log axis, so (0, -1) -- which IS an infinitesimal, 1/ln(1/h) --
    read as order zero and was excluded from every "infinitesimal terms" test.
    That is how a truncated log-axis series kept reporting itself EXACT.

    This has now been the same mistake four times in one day: max_positive_dim,
    the non-dyadic guard, atan's antiderivative filter, and here.  _dim_order is
    correct for what it means (Taylor order on the power axis) and dangerous for
    what it reads like, so anything asking "how small is this dimension" across
    axes must use THIS instead.
    """
    if not isinstance(d, tuple):
        return -d
    for c in d:
        if c != 0:
            return -c
    return 0


def _infinitesimal_terms(x):
    """The strictly-infinitesimal part of x, as a {dim: coeff} dict.

    Infinitesimal means lexicographically BELOW zero, on whichever axis leads --
    not merely "negative power component".
    """
    return {d: c for d, c in x.c.items() if c != 0.0 and _dim_negative(d)}


def _min_complete(*xs):
    """Lowest CARRIED completeness among the operands; None if all are exact.

    This is the propagation rule for every arithmetic op: a sum, product or
    quotient is complete only as far as its least complete operand.
    """
    best = None
    for x in xs:
        v = getattr(x, "_complete", None) if isinstance(x, Composite) else None
        if v is None:
            continue
        best = v if best is None else min(best, v)
    return best


def _scaled_complete(x, y, ax, ay, sign):
    """Completeness of a product (sign = +1) or quotient (sign = -1).

    A bound is RELATIVE to the operand's leading order: X complete to C with
    leading order L is right for C - L orders past its leading term.  A product
    or quotient keeps the tighter of those relative bounds, starting from its
    own leading order L_x + sign * L_y.  _min_complete took the smaller ABSOLUTE
    bound, which ignores the shift: X / h, X complete to 5, moves X's order 5
    to order 4 and its first wrong coefficient (order 6) to order 5 -- yet
    claimed 5.  The same for X * (1/h).  Found reading T(E) of a rectangular
    barrier at E = V0 - h: claimed complete to 7, order 7 off by 9.5e-7.  In the
    other direction it under-claimed: X * h is complete to 6 and said 5.

    x, y are the operands as given (they carry the bounds); ax, ay the operands
    after R1 (their leading orders are what the arithmetic used).  With no
    leading term on either side (nothing, or only zeros) the old rule stands.
    """
    lx = ax.lead_order() if isinstance(ax, Composite) else 0
    ly = ay.lead_order() if isinstance(ay, Composite) else 0
    if lx is None or ly is None:
        return _min_complete(x, y)
    rel = None
    for v, lead in ((x, lx), (y, ly)):
        c = getattr(v, "_complete", None) if isinstance(v, Composite) else None
        if c is None:
            continue
        r = c - lead
        rel = r if rel is None else min(rel, r)
    if rel is None:
        return None
    return lx + sign * ly + rel


def _tighter(*bounds):
    """The strictest of several bounds, ignoring None (= no bound)."""
    best = None
    for b in bounds:
        if b is None:
            continue
        best = b if best is None else min(best, b)
    return best


def _truncate_order(result, order):
    """Drop the orders above `order`, keeping every dimension at or below it.

    Reads the backend arrays rather than the .c dict.  Building that dict was
    ~20% of an exp evaluation once the series loop was fused -- it allocates a
    Python dict and calls dim_cast per term, to answer a question numpy can
    answer with one comparison.
    """
    if order is None:
        return result
    dims = result._backend.active_dims(result._data)
    if getattr(dims, "dtype", None) is not None and dims.dtype != object:
        over = (-dims) > order
        if not over.any():
            result._complete = _tighter(result._complete, order)
            return result
        keep = ~over
        _warn_truncated(int(over.sum()), "order cap %s" % order, order)
        d2 = dims[keep]
        _, v = result._backend.to_arrays(result._data)
        # _join: dropping orders off a number does not change WHERE it came
        # from.  Without this every transcendental lost the flag at its last
        # step, because the truncation is the last thing sin() does.
        return _join(Composite._wrap(
            result._backend.create_from_terms(d2, v[keep]), result._backend,
            complete=_tighter(getattr(result, "_complete", None), order)),
            result)
    extra = [d for d in result.c if _dim_order(d) > order]
    if not extra:
        # Nothing to drop, but the bound still HOLDS and must be recorded --
        # otherwise a caller downstream reads "exact" off a truncated series.
        result._complete = _tighter(result._complete, order)
        return result
    _warn_truncated(len(extra), "order cap %s" % order, order)
    out = _like(result, {d: v for d, v in result.c.items()
                         if _dim_order(d) <= order})
    out._complete = _tighter(getattr(result, "_complete", None), order)
    return out


def _bounded_at_inf(func, x, terms=12):
    """Evaluate a bounded transcendental at an infinite composite argument.

    For monotonic bounded functions (atan, tanh): math.func(±inf) returns the
    correct asymptotic value (e.g. atan(inf) = pi/2).  Result via R().

    For oscillatory functions (sin, cos): math.func(±inf) raises ValueError.
    The value cannot be pinpointed at grade 0 -- sin(1/h) is bounded by 1 and
    has no limit -- but that does not mean there is nothing to return.  It is
    handled the way sqrt handles an odd dimension: the DIMENSION DEGRADES.

        sin(|c|_d) = |sin(c)|_(d/2)        for d > 0

    A RANGE IS NOT A POINT.  sin(1/h) takes every value in [-1, 1] -- it is
    0 and 1 infinitely often at arbitrarily small eps -- and a composite holds
    ONE value at ONE grade.  That is the whole obstruction, and it is why no
    rule on the dimension repairs it.  Every candidate was measured:

      grade d   (dimension kept)  revertible, st undefined, but the damping
                cancels: x*sin(1/x) came back |sin 1|_0 = 0.841471, not 0,
                and 1/sin(1/h) came back INFINITESIMAL for something that is
                genuinely unbounded.              test_limits 87/109
      grade d/2 squeeze fixed, magnitude still wrong: limit() called
                sin(sin(1/x)) divergent, and it never exceeds 1.   90/105
      grade 0   squeeze and magnitude both right, but st becomes DEFINED as
                sin(1), and asin no longer reverts the skip.       94/109
      d**(1/d)  fails the squeeze at d=1 exactly, and is not monotone.
      asin(d)   fails the squeeze at d=1, and is undefined for d > 1.
      sin(d)    works on (0, pi) only; sign flips past pi, and it inherits
                sin's non-monotonicity, so the dominance order reverses.

    The two requirements that decide it contradict: st-undefined needs a
    positive grade, an honest magnitude needs a non-positive one.  So the
    function refuses instead of choosing which to break.   103/105

    The coefficient reading that survived all this is worth keeping in mind:
    |sin(1)|_1 is ONE SAMPLE of the oscillation -- a y-value at a known
    argument, which is why asin could inverted it.  A sample cannot answer
    anything that needs the whole range, and the limit, the supremum and
    whether the reciprocal blows up are all of that kind.
    """
    max_d = x.max_positive_dim()
    sign = 1.0 if x.coeff(max_d) > 0 else -1.0

    # atan has an ASYMPTOTIC SERIES at an unbounded argument and it is exactly
    # representable here, because 1/x is infinitesimal when x has a positive
    # grade:
    #     atan(x) = +-pi/2 - 1/x + 1/(3x^3) - 1/(5x^5) + ...
    # Returning pi/2 alone is the LIMIT, not the value: atan(1/h) is
    # pi/2 - h + h^3/3 - ..., so the leading term was right and every order
    # below it was silently dropped.  The same identity covers both signs --
    # atan(x) + atan(1/x) is +pi/2 for x > 0 and -pi/2 for x < 0, and the
    # series for atan(1/x) is the same either way.
    #
    # tanh is NOT done this way: tanh(1/h) = 1 - 2exp(-2/h) + ..., and that
    # correction is exponentially flat, so it lies outside the value group
    # entirely.  1 is right to every representable order.
    if func is math.atan:
        u = R(1) / x
        u2 = u * u
        acc = R(sign * math.pi / 2)
        term = u
        for k in range(_effective_terms(terms)):
            acc = acc + (term / float(2 * k + 1)) * (1.0 if k % 2 else -1.0)
            term = term * u2
        return acc

    try:
        return R(func(sign * float('inf')))
    except (ValueError, OverflowError):
        raise NotRepresentableError(
            f"{func.__name__}(x) at an unbounded argument is a RANGE, not a "
            f"point: sin(1/h) takes every value in [-1, 1] and a composite "
            f"holds one value at one grade.  No rule on the grade repairs "
            f"that -- see the analysis above."
        ) from None


# =============================================================================
# LOG SCALE  (opt-in)
# =============================================================================
#
# ln of an infinitesimal is  ln(c) + d*ln(h), and ln(h) needs a dimension that
# is positive but smaller than EVERY power -- log x outgrows any constant and is
# outgrown by x^e for every e > 0.  No float sits there, so with scalar
# dimensions the d*ln(h) term has nowhere to go and ln() raises.
#
# A VECTOR dimension (power, log) does have room: the log component is the minor
# one, so lexicographic comparison puts any log term below any power term, which
# is exactly the dominance order.  Then
#
#     ln(|c|_d)      =  |ln c|_(0,0) + |d|_(0,1)
#     exp(|k|_(0,1)) =  |1|_(k,0)
#
# It is ON by default.  ln of an infinitesimal is a question with an answer,
# and refusing it -- or worse, approximating it through the numeric fallback,
# which left lim(x->0+) x^x at 1.6e-6 short of 1 -- is the wrong default when
# the exact answer is available.  Escalation costs nothing until it happens:
# vector dimensions live on their own dict backend, a composite only moves
# there when a log term actually appears, and it demotes back the moment the
# log components cancel.  The scalar fast path is never touched.
#
# Set LOG_SCALE = False to disable it, in which case ln of an infinitesimal
# raises rather than silently dropping the scale (which is what it used to do,
# and which produced wrong answers: ln(h), ln(h^2) and ln(sqrt(h)) all returned
# the same object).
#
# Known bound: ONE level of log.  ln(ln(x)) needs a third basis component,
# because ln(d*ln(h)) = ln(d) + ln(ln(h)) is a new scale.
LOG_SCALE = True

_VEC_BACKEND = [None]


def _vector_backend():
    if _VEC_BACKEND[0] is None:
        from composite.backends.vector_dim_backend import VectorDimBackend
        _VEC_BACKEND[0] = VectorDimBackend()
    return _VEC_BACKEND[0]


def _carries_vector(x):
    """Does this composite actually HOLD a vector dimension?

    Not the same question as whether its backend advertises VECTOR_DIMS.  ln()
    puts a term on the log axis whatever backend it was called on, so a plain
    scalar backend routinely ends up holding tuple dimensions -- and gating the
    log handling on the backend flag meant exp() never recognised them there.
    On DictBackend that made h**0.25 refuse: ** routes through
    exp(n*ln(h)), ln gave the log-axis term, exp did not see it, and the
    positive-grade check rejected a perfectly representable object.
    """
    try:
        dims, _ = x._backend.to_arrays(x._data)
    except Exception:
        return False
    return any(isinstance(d, tuple) for d in dims)


def _unit_dim(x):
    """The dimension meaning 1 on x's backend: 0, or (0, 0, ...) for vectors."""
    if getattr(x._backend, "VECTOR_DIMS", False):
        from composite.backends.vector_dim_backend import WIDTH
        return (0,) * WIDTH
    return 0


def _fractional_power(d):
    """Power-axis component is not a whole number: a branch point, not a series.

    sqrt(h) is |1|_-0.5 and every INTEGER grade in it is absent, so reading
    grade -1 off it reports 0.0 for a derivative that is actually unbounded.
    Asked of the power component only, because a log-axis grade is always a
    whole number and is refused elsewhere on its own grounds.
    """
    p = d[0] if isinstance(d, tuple) else d
    p = float(p)
    return p != int(p)


def _shift_derivative(x):
    """d/dh by the grade shift: |c|_g -> |-g*c|_(g+1).  See Composite.D."""
    from composite.backends.vector_dim_backend import canon
    terms = {}
    for g, c in x.coeffs_dict().items():
        if isinstance(g, tuple):
            if any(e != 0 for e in g[1:]):
                raise NotImplementedError(
                    f"d/dh of a log-axis term is not a grade shift: {g} is an "
                    f"iterated logarithm, and d/dh of ln(1/h) is -1/h, which "
                    f"lands on the POWER axis.  Refusing rather than dropping "
                    f"the term, which would return a well-formed wrong answer.")
            power = g[0]
        else:
            power = g
        if power == 0:
            continue                    # a constant differentiates away
        key = canon((power + 1,) + g[1:]) if isinstance(g, tuple) else power + 1
        terms[key] = terms.get(key, 0.0) + (-power) * c
    if not terms:
        # The derivative of a constant is ZERO, and a zero here is a value.
        # Returning nothing would say "no derivative", which is a different
        # statement and the one R6 exists to keep apart.
        terms = {canon((0,) + (0,) * (WIDTH - 1))
                 if x.coeffs_dict() and isinstance(next(iter(x.coeffs_dict())), tuple)
                 else 0: 0.0}
    return _like(x, terms)


def _like(x, terms):
    """Build a composite on the SAME backend as `x`, not the active one.

    The transcendental series construct intermediates with Composite({...}),
    which binds whatever backend is globally active.  When the argument carries
    vector dimensions that backend cannot hold the tuples, and numpy raises
    "setting an array element with a sequence".
    """
    be = x._backend
    if not terms:
        return Composite._wrap(
            be.create_from_terms(np.array([], dtype=DIM_DTYPE),
                                 np.array([], dtype=np.float64)),
            be, demote=False)
    # SORTED, and as the same array kinds Composite.__init__ builds.  Passing a
    # dict's insertion order straight through is what broke asin: its result is
    # assembled with key 0 first and the negative dims after, and the
    # sparse-dense backend reads runs off the order it is given -- so the
    # standard part was dropped and asin(x) came back with st() == 0.  _like
    # only ever worked because every earlier caller happened to pass a sorted
    # dict.
    # dom_sorted, not sorted: _r1 shifts dims[0] on the stated contract that
    # to_arrays comes back ascending and [0] is the LOWEST dimension.  Raw
    # tuple order puts a canonical (0,0) BELOW (0,0,-3), so "lowest" could be
    # a term that dominates it, and R1 would uplift the wrong one.
    from composite.backends.vector_dim_backend import dom_sorted as _dom_sorted
    sorted_dims = _dom_sorted(terms.keys())
    if isinstance(sorted_dims[0], tuple):
        dims = np.empty(len(sorted_dims), dtype=object)
        for _i, _d in enumerate(sorted_dims):
            dims[_i] = _d
    else:
        dims = np.array(sorted_dims, dtype=DIM_DTYPE)
    vals = np.array([terms[d] for d in sorted_dims], dtype=np.float64)
    # Provenance follows x: a series built from x is a value DERIVED from it,
    # not a new source.  Without this every transcendental would look like a
    # leaf and sin(exp(sqrt(x))) would flag.
    return _join(Composite._wrap(be.create_from_terms(dims, vals), be,
                                 demote=False), x)


def _vec_composite(terms):
    """Build a vector-dimension composite whatever the active backend is.

    Goes through _wrap so the result demotes when it turns out to carry nothing
    but powers -- exp(ln(h)) is |1|_(-1,0), which is just h and belongs back on
    the scalar path.  A term with a real log component will not demote.
    """
    be = _vector_backend()
    return Composite._wrap(
        be.create_from_terms(list(terms.keys()), list(terms.values())), be)


def _demote(data, be):
    """Return a vector-dimension composite to the scalar path when it can.

    A composite keeps vector dimensions only while something other than the
    power component is non-zero.  Without this, a single ln() anywhere would
    leave every value downstream of it on the dict backend for good, and the
    whole point of the vector representation is that you pay for it only while
    you are using it.  Returns (data, backend), unchanged when it cannot demote.
    """
    dims, vals = be.to_arrays(data)
    for d in dims:
        if isinstance(d, tuple) and any(c != 0 for c in d[1:]):
            return data, be
    active = get_backend()
    if active.VECTOR_DIMS:
        return data, be                      # nowhere scalar to go
    flat = [d[0] if isinstance(d, tuple) else d for d in dims]
    return active.create_from_terms(flat, vals), active


def _vec_unit(k, width=None):
    """The dimension that IS the k-th basis element: 1 at index k, 0 elsewhere."""
    from composite.backends.vector_dim_backend import ensure_depth
    w = ensure_depth(max(k + 1, width or 0))
    return tuple(1 if i == k else 0 for i in range(w))


def _sole_log_index(d):
    """(k, e) when dim d is e copies of ONE basis element at index k >= 1.

    exp can only lower a term that is a single basis element: exp(v*B_k) is
    B_{k-1}**v.  A term like (0, 2) -- that is (log x)**2 -- has no image at
    any depth, because exponentiating a SQUARED log produces a scale outside
    this family entirely.  That is a different refusal from "the basis is too
    shallow", and conflating the two is what made the old k == 1 test look like
    a depth limit when it is really an exponent limit.
    """
    from composite.backends.vector_dim_backend import as_vec
    v = as_vec(d)
    nz = [(i, c) for i, c in enumerate(v) if c != 0]
    if len(nz) != 1:
        return None
    k, e = nz[0]
    return (k, e) if k >= 1 else None


def _log_part(x):
    """Split the INFINITE log terms off an exponent.  Returns (logs, rest).

    exp needs the split because an infinite argument changes scale while a
    vanishing one is just a Taylor series.  A vector dimension (p, l) is
    infinite exactly when it is lexicographically above zero -- p > 0, or
    p == 0 and l > 0 -- which is the same dominance order the comparison uses.

    So a MIXED term like (-1, 1) -- that is h*log(h) -- is infinitesimal, not
    infinite: the power dominates the log.  It belongs in `rest` and expands
    ordinarily.  Treating it as unrepresentable is what left x^x falling back
    to numeric sampling and landing 1.6e-6 short of 1.
    """
    dims, vals = x._backend.to_arrays(x._data)
    logs, rest = {}, {}
    for d, v in zip(dims, vals):
        if isinstance(d, tuple) and len(d) > 1:
            # A term changes SCALE under exp when its power component is zero
            # and it is positive on some log axis -- at ANY depth, not only the
            # first.  Lexicographic order decides the sign: the first non-zero
            # component after the power is what dominates.
            rest_comps = d[1:]
            lead = next((c for c in rest_comps if c != 0), 0)
            if d[0] == 0 and lead > 0:
                logs[d] = v
                continue
        rest[d] = v
    return logs, rest


def _has_infinitesimal_part(x):
    """True when x carries any NONZERO coefficient away from dimension 0.

    A composite whose only content is at dimension 0 -- or whose other
    dimensions are all zero -- is just its standard part, and a Taylor series
    around it must not be built: x - R(a) would be a zero, and R1 would turn
    that zero into |1|_-1, manufacturing an infinitesimal that is not there.
    """
    dims, vals = x._backend.to_arrays(x._data)
    # NB: compare the dimension itself, not int(d) -- int(-0.5) is 0, which
    # made a purely fractional infinitesimal (sqrt of an odd dimension) look
    # like a bare standard part, so sin/ln returned f(st(x)) and dropped it.
    #
    # _dim_nonzero, not `d != 0`: the VECTOR zero is (0, 0), and a tuple is
    # never equal to an int, so a vector-keyed standard part tested as an
    # infinitesimal.  sin then built its series with a = st(x) AND h still
    # holding that same standard part, so sin(5) came back as sin(10).
    return any(_dim_nonzero(d) and v != 0.0 for d, v in zip(dims, vals))


def _is_nothing(x):
    """Check if x is the empty composite (nothing)."""
    return isinstance(x, Composite) and not x.c


@_honours_terms
def sin(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.sin, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.sin(a)})
    _nz = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}
    # _like, not Composite({...}): the bare constructor binds whatever backend
    # is globally ACTIVE, so a vector-dimension argument had its tuples handed
    # to numpy -- "setting an array element with a sequence" -- or came back as
    # DictData wearing sparse-dense methods.
    #
    # _dim_nonzero, not `d != 0`: h must hold the part AWAY from dimension
    # zero.  The scalar spelling let the vector zero (0, 0) through, so the
    # standard part was counted twice -- once as a, once inside h.
    h = _like(x, {d: c for d, c in x.c.items() if _dim_nonzero(d)})
    sin_a, cos_a = math.sin(a), math.cos(a)
    _one = _like(x, {_unit_dim(x): 1.0})
    sin_h = _like(x, {})
    cos_h = _one
    h_power = _one
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    # Skip zero-coefficient terms to avoid 0 * Composite uplift
    result = _like(x, {})
    if sin_a != 0:
        result = result + sin_a * cos_h
    if cos_a != 0:
        result = result + cos_a * sin_h
    return _truncate_order(result, _tighter(_complete_order(_nz, terms),
                                            _min_complete(x)))


@_honours_terms
def cos(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({0: 1.0})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.cos, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.cos(a)})
    _nz = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}
    # _like, not Composite({...}): the bare constructor binds whatever backend
    # is globally ACTIVE, so a vector-dimension argument had its tuples handed
    # to numpy -- "setting an array element with a sequence" -- or came back as
    # DictData wearing sparse-dense methods.
    #
    # _dim_nonzero, not `d != 0`: h must hold the part AWAY from dimension
    # zero.  The scalar spelling let the vector zero (0, 0) through, so the
    # standard part was counted twice -- once as a, once inside h.
    h = _like(x, {d: c for d, c in x.c.items() if _dim_nonzero(d)})
    sin_a, cos_a = math.sin(a), math.cos(a)
    _one = _like(x, {_unit_dim(x): 1.0})
    sin_h = _like(x, {})
    cos_h = _one
    h_power = _one
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    result = _like(x, {})
    if cos_a != 0:
        result = result + cos_a * cos_h
    if sin_a != 0:
        result = result - sin_a * sin_h
    return _truncate_order(result, _tighter(_complete_order(_nz, terms),
                                            _min_complete(x)))


@_honours_terms
def _exp_float(a):
    """math.exp, refusing an underflow.

    e**a for a below about -745 is a positive real that float64 cannot hold,
    and math.exp returns 0.0 for it.  Built into a composite that 0.0 is a
    WRITTEN zero, R1 converts it, and exp(-(800 + h)) came back as
    h - h**2 + h**3/2 - ...: an infinitesimal of coefficient one where the
    value is e**-800.  So it is refused, as exp(-1/h) is: the value exists and
    this representation cannot carry it.
    """
    v = math.exp(a)
    if v == 0.0:
        raise NotRepresentableError(
            "exp(%r) underflows float64: the value is positive but below the "
            "smallest representable number, and a 0.0 standing in for it would "
            "be a written zero that converts to an infinitesimal." % (a,))
    return v


def exp(x, terms=15):
    """Exponential function for Composite numbers."""
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return Composite({0: _exp_float(float(x))})

    if not isinstance(x, Composite):
        return Composite({0: _exp_float(float(x))})

    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({0: 1.0})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)

    if LOG_SCALE and (getattr(x._backend, "VECTOR_DIMS", False)
                      or _carries_vector(x)):
        _logs, _rest = _log_part(x)
        if _logs:
            # exp(k * ln(1/h)) = (1/h)^k = h^(-k), which sits at power +k.
            out = None
            for _d, v in _logs.items():
                # exp(v * B_k) = B_{k-1} ** v: one step DOWN the basis, at any
                # depth.  B_1 = ln(1/h) so exp lands on the power axis; B_2 =
                # ln(ln(1/h)) so exp lands on the log axis; and so on.
                _ke = _sole_log_index(_d)
                if _ke is None or _ke[1] != 1:
                    from composite.backends.vector_dim_backend import BASIS
                    raise ValueError(
                        f"exp of {_d}: a log term can only be exponentiated when "
                        f"it is ONE basis element to the first power. "
                        f"{_d} is not, and no depth fixes that -- exp of a "
                        f"squared or mixed log leaves this family of scales. "
                        f"(basis {tuple(BASIS)})")
                _k = _ke[0]
                _tgt = list(_vec_unit(_k - 1))
                _tgt[_k - 1] = v
                t = _vec_composite({tuple(_tgt): 1.0})
                out = t if out is None else out * t
            if _rest:
                # _rest CAN HOLD SCALAR DIMENSIONS next to vector ones.  ln()
                # puts a term on the log axis whatever backend it was called
                # on (see _carries_vector), so a scalar backend routinely ends
                # up mixing the two, and everything below indexes dimensions
                # positionally.  Taking len() of a scalar key is what crashed
                # (ZERO/2)**0.5 on DictBackend with "object of type 'int' has
                # no len()" -- while the same expression was fine on the two
                # array backends, whose to_arrays hands back uniform tuples.
                # A bare ZERO**0.5 missed it because its remainder is empty.
                #
                # Promote every key to one width first.  A scalar dimension d
                # IS the vector dimension (d, 0, ...): d on the power axis,
                # nothing on the log axes.
                _w = max([len(_d) for _d in _rest if isinstance(_d, tuple)]
                         + [len(_vec_unit(0))])

                def _as_vec(_d):
                    if isinstance(_d, tuple):
                        return _d + (0,) * (_w - len(_d))
                    return (_d,) + (0,) * (_w - 1)

                _wide = {}
                for _d, _v in _rest.items():
                    _k = _as_vec(_d)
                    # Two spellings of one dimension must ADD, not overwrite:
                    # a scalar 0 and a tuple (0, 0) are the same term.
                    _wide[_k] = _wide.get(_k, 0.0) + _v
                _rest = _wide

                # A remainder that is only the standard part needs no series --
                # and recursing would rebuild it through the ACTIVE backend,
                # which cannot hold vector dimensions.
                _zero = tuple(0 for _ in range(_w))
                if set(_rest) <= {_zero}:
                    out = out * _vec_composite(
                        {_zero: _exp_float(_rest.get(_zero, 0.0))})
                else:
                    out = out * exp(_vec_composite(_rest), terms)
            return out

    # OUTSIDE THE VALUE GROUP.  exp of a positive grade is not a large number,
    # it is a different LEVEL: exp(1/h) is above every power of 1/h, and
    # exp(-1/h) is nonzero and below every power of h -- the flat object.  No
    # finite-rank dimension names either, because the transmonomial ordering
    # stops being finite-rank lex once exponentials appear; grades would have
    # to become recursive expressions rather than coordinates.  That is the
    # step from Hahn/Hardy series to transseries, and this library stops at
    # powers and iterated logs.
    #
    # It used to apply the Maclaurin series regardless, so exp(1/h) and
    # exp(-1/h) BOTH came back as 1 - 1 + 1/2 - ... truncated at 15 terms,
    # with standard part 1.0 for each -- two objects at opposite ends of the
    # scale reported as the same finite number, and any comparison between
    # them decided by round-off.  A series evaluation is sound exactly when
    # the argument is (finite standard part) + (strictly negative grade), and
    # that is what is checked here.
    if _has_positive_dims(x):
        if _EXP_TO_TS.get():
            return _exp_level_one(x)
        raise NotRepresentableError(
            f"exp of a positive grade is outside this value group: "
            f"exp(1/h) is above every power and exp(-1/h) is below every "
            f"power (nonzero, flat).  Naming either needs exponential "
            f"levels -- transseries -- not a power-and-log dimension.")

    a = x.st()
    # 7.1: a term exists iff its coefficient is nonzero -- exactly zero,
    # not 'small'.  A tolerance here silently discards real content.
    # _dim_nonzero, not `d != 0`: non_zero must hold the part AWAY from
    # dimension zero, because the result is exp(a) * exp(h) and a is already
    # the standard part.  The scalar spelling let the vector zero (0, 0)
    # through, so exp(0.4 + h*ln(1/h)) came back as e**0.8 -- the standard
    # part multiplied in twice, the same double-count that made sin(5)
    # return sin(10).
    non_zero = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}

    if not non_zero:
        return Composite({0: _exp_float(a)})

    base = _exp_float(a)
    h = _like(x, non_zero)
    one = _like(x, {_unit_dim(x): 1.0})

    exp_h = one
    h_power = one
    for n in range(1, terms):
        h_power = h_power * h
        exp_h = exp_h + (1.0 / math.factorial(n)) * h_power

    # f(g(x)) is complete only as far as g is: the outer series reaches for
    # orders of its argument that a truncated inner function never produced.
    return _truncate_order(base * exp_h,
                           _tighter(_complete_order(non_zero, terms),
                                    _min_complete(x)))


def _ln_vector(x, terms):
    """ln of a vector-dimension composite, at any depth.  None = not ours.

    A dimension (e0, e1, ...) means c * (1/h)**e0 * B1**e1 * B2**e2 * ..., so

        ln(c * prod B_k**e_k) = ln(c) + sum_k e_k * ln(B_k)
                              = ln(c) + sum_k e_k * B_{k+1}

    because ln(1/h) IS B1 and ln(B_k) IS B_{k+1}.  Every exponent moves ONE
    index up and becomes a coefficient.  That single rule covers every depth:
    ln(h) lands on the log axis, ln(ln(1/h)) on the loglog axis, and the basis
    grows to hold it.  Without this branch ln simply compared a tuple against
    0 and raised TypeError, which then surfaced through limit() as
    "not composable with composite arithmetic" -- pointing at the wrong thing
    entirely, since the function composed fine and the basis was the problem.
    """
    from composite.backends.vector_dim_backend import as_vec, canon, ensure_depth
    dims, vals = x._backend.to_arrays(x._data)
    # CANONICAL length, not the padded one: as_vec pads to the current WIDTH,
    # so measuring it and asking for one more grew the basis on EVERY ln call,
    # ratcheting to 19 components over a handful of limits.  The canonical form
    # is what the term actually needs.
    items = [(canon(as_vec(d)), float(v)) for d, v in zip(dims, vals) if v != 0.0]
    if not items:
        return None
    lead_d, lead_c = max(items, key=lambda t: t[0])
    if all(e == 0 for e in lead_d):
        return None                      # a plain real: the scalar path owns it
    if lead_c <= 0.0:
        raise ValueError(
            f"ln of a composite whose leading coefficient is {lead_c}: "
            "the logarithm needs a positive leading term.")
    w = ensure_depth(len(lead_d) + 1)
    out = {}
    lc = math.log(lead_c)
    if lc != 0.0:
        out[tuple(0 for _ in range(w))] = lc
    for k, e in enumerate(lead_d):
        if e == 0:
            continue
        u = _vec_unit(k + 1, w)
        out[u] = out.get(u, 0.0) + e
    lead = _vec_composite(out)
    if len(items) == 1:
        return lead
    # x = lead_term * (1 + r); ln(x) = ln(lead_term) + ln(1 + r)
    ratio = x / _vec_composite({lead_d: lead_c})
    if _is_unit(ratio):
        return lead
    return lead + ln(ratio, terms)


@_honours_terms
def ln(x, terms=15):
    """Natural logarithm for composite numbers.

    For positive infinitesimals (st=0 but positive coefficient at a
    negative dim), evaluates ln at the coefficient.  This is the
    composite-native handling: ZERO = |1|_{-1} is a positive
    infinitesimal with coefficient 1, so ln(ZERO) = R(ln(1)) = R(0)
    = ZERO.  Then x·ln(x) = ZERO² with st=0, and x^x = exp(ZERO²)
    = 1.0 exactly.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)

    if LOG_SCALE and (getattr(x._backend, "VECTOR_DIMS", False)
                      or _carries_vector(x)):
        _v = _ln_vector(x, terms)
        if _v is not None:
            return _v

    coeffs = x.c
    if LOG_SCALE:
        # ln(|c|_d) = ln(c) + d*ln(h) holds for ANY non-zero d, so an
        # INFINITY is the same rule with the sign the other way round:
        # ln(1/h) = -ln(h) = +L.
        #
        # BEFORE x.st().  This used to sit inside the `a <= 0` branch,
        # which meant reading a standard part first -- and an unbounded
        # argument has none, so ln(1/h) raised on the way to the code that
        # already knew the answer.  The grade decides here; nothing about
        # this case needs a standard part.
        _pos = {d: c for d, c in coeffs.items()
                if _dim_positive(d) and c != 0.0}
        if _pos:
            from composite.backends.vector_dim_backend import dom_max as _dom_max
            _pd = _dom_max(_pos)
            _pc = _pos[_pd]
            if _pc > 0:
                _t = {(0, 1): float(_pd)}
                _lc = math.log(_pc)
                if _lc != 0.0:
                    _t[(0, 0)] = _lc
                _lead = _vec_composite(_t)
                _rest = x / _like(x, {_pd: _pc})
                if _is_unit(_rest):
                    return _lead
                return _lead + ln(_rest, terms)

    a = x.st()

    if a <= 0:
        # Check for positive infinitesimal: st=0 but positive coeff
        # at a negative dimension (e.g. ZERO = |1|_{-1})
        coeffs = x.c
        neg_dims = {d: c for d, c in coeffs.items() if _dim_negative(d)}
        if neg_dims:
            min_dim = min(neg_dims.keys())
            coeff = neg_dims[min_dim]
            if coeff > 0:
                # ln(|c|_d) = ln(c) + d*ln(h), and ln(h) needs a dimension that
                # is positive but smaller than EVERY power -- log x outgrows any
                # constant and is outgrown by x^e for every e > 0.  No float can
                # sit there, so the d*ln(h) term has nowhere to go.
                #
                # This used to drop it and return ln(c) alone, which makes
                # ln(h), ln(h^2), ln(h^3) and ln(sqrt(h)) all the SAME object
                # and produces wrong answers, not merely lost structure:
                #   lim(x->0+) ln(x)/ln(x*x)      gave 1.0   (is 0.5)
                #   lim(x->0+) ln(x)/ln(sqrt(x))  gave 1.0   (is 2.0)
                #   lim(x->0+) 1/ln(x)            gave |1|_1 (is 0.0, inverted)
                # It is right only when the log is multiplied by something that
                # vanishes, which is why x*ln(x) and x^x survived it.
                if LOG_SCALE:
                    # ln(x) = ln(lead) + ln(x/lead), and x/lead has standard
                    # part 1 so the second term takes the ordinary Taylor path.
                    _lead_terms = {(0, 1): float(min_dim)}
                    _lc = math.log(coeff)
                    if _lc != 0.0:
                        _lead_terms[(0, 0)] = _lc
                    _lead = _vec_composite(_lead_terms)
                    _rest = x / _like(x, {min_dim: coeff})
                    if _is_unit(_rest):
                        return _lead
                    return _lead + ln(_rest, terms)
                raise ValueError(
                    f"ln(|{coeff}|_{min_dim}): the log SCALE cannot be "
                    f"represented. ln of an infinitesimal is ln({coeff}) + "
                    f"({min_dim})*ln(h), and ln(h) needs a dimension between 0 "
                    f"and every positive power -- not expressible as a float. "
                    f"Dropping it silently returned ln({coeff}) and made "
                    f"ln(h), ln(h^2) and ln(sqrt(h)) indistinguishable. "
                    f"Rewrite so the log is multiplied by a vanishing factor "
                    f"(x*ln(x) and x^x work), or expand about a point with a "
                    f"nonzero standard part.")
        raise ValueError("ln requires positive standard part")

    if not _has_infinitesimal_part(x):
        return Composite({0: math.log(a)})

    h_part = x - R(a)
    ratio = h_part / R(a)

    # R6: there is no additive identity.  A summation that has not yet added a
    # term holds NOTHING, not zero -- seeding with Composite({0: 0.0}) would
    # assert a zero, which R1 converts to |1|_-1 and adds to the series.
    lead = math.log(a)
    result = _like(x, {}) if lead == 0.0 else _like(x, {_unit_dim(x): lead})
    power = _like(x, {_unit_dim(x): 1.0})

    for n in range(1, terms):
        power = power * ratio
        sign = (-1) ** (n + 1)
        result = result + sign * power / n

    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))


@_honours_terms
def sqrt(x, terms=12):
    """Square root for composite numbers via binomial series.

    For positive infinitesimals (st=0, positive coeff at negative dim),
    evaluates sqrt at the coefficient.  ZERO = |1|_{-1} has coeff 1,
    so sqrt(ZERO) = R(sqrt(1)) = R(1).  Then x·sqrt(x) at ZERO gives
    ZERO × R(1) = ZERO, st=0 exactly.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})

    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)

    # Dimensions HALVE under a square root:  sqrt(|c|_d) = |sqrt(c)|_(d/2).
    #
    # This used to keep the dimension instead of halving it, returning ZERO for
    # sqrt(ZERO) and silently giving wrong LIMITS, not just wrong intermediates:
    #   lim(x->0+) sqrt(x)/x   gave 1.0   (is infinity)
    #   lim(x->0+) x/sqrt(x)   gave 1.0   (is 0)
    #   lim(x->0+) sqrt(x*x)/x gave 0.0   (is 1 -- |x|/x for x > 0)
    #
    # Halving an ODD dimension lands on a half-integer.  That used to be
    # unrepresentable and raised, naming the x = s*s substitution as the way
    # round it.  Dimensions are float64 now, so sqrt(|1|_-1) = |1|_-0.5 is an
    # ordinary value and the substitution is no longer required.  Note the
    # result stays UNIT-SPACED -- sqrt shifts the lattice by 1/2, it does not
    # refine it -- so the run representation is unaffected.
    _nz = {d: c for d, c in x.coeffs_dict().items() if c != 0.0}
    if _nz:
        # dom_max, not max: raw tuple order ranks a strict prefix lower, so a
        # canonical (0,0) lost to (0,0,-3) -- an INFINITESIMAL taken as the
        # leading term.  sqrt then read a 1e-32 cancellation coefficient and
        # refused the expression.
        from composite.backends.vector_dim_backend import dom_max as _dom_max
        _lead = _dom_max(_nz)
        if _dim_nonzero(_lead):
            _c = _nz[_lead]
            if _c < 0:
                raise ValueError(
                    f"sqrt of a negative leading coefficient |{_c}|_{_lead}")
            _root = _like(x, {_dim_scaled(_lead, 0.5): math.sqrt(_c)})
            _rest = x / _like(x, {_lead: _c})         # leading dim 0, st() == 1
            return _root * sqrt(_rest, terms)

    a = x.st()
    if a < 0:
        raise ValueError("sqrt requires non-negative standard part")

    if not _has_infinitesimal_part(x):
        return Composite({0: math.sqrt(a)})

    # SOLVE y**2 = x ORDER BY ORDER, where the dimensions allow it.
    #
    #     y_0 = sqrt(x_0),   2 y_0 y_n = x_n - sum_{k=1..n-1} y_k y_{n-k}
    #
    # Each order is fixed once from the orders below it.  The binomial series
    # kept below forms ratio**n instead, and for an argument with many terms
    # that is a far longer chain of roundings to reach the same coefficient.
    # Against sqrt(1+sin x) at x=0.4, relative error by order:
    #
    #     order         8         12        16        20
    #     binomial      2.4e-12   1.3e-07   2.3e-02   5.0e+03
    #     recurrence    5.6e-13   6.8e-09   3.1e-04   3.9e+01
    #     Newton        2.3e-12   2.9e-08   1.3e-03   1.7e+02
    #
    # None of them is exact past order 16, and that is not the algorithm: the
    # SAME recurrence at 60 decimal digits is exact to the last digit at every
    # order.  The coefficients fall eighteen orders of magnitude between n=8
    # and n=20 while the input is O(1), so each order costs about a digit and
    # float64's sixteen run out near order 16.  The recurrence only spends
    # them more slowly.
    _scalar_orders = None
    if all(not isinstance(d, tuple) for d in x.coeffs_dict()):
        _ks = [-float(d) for d, c in x.coeffs_dict().items() if c != 0.0]
        if all(k >= 0 and k == int(k) for k in _ks):
            _scalar_orders = {int(-float(d)): c
                              for d, c in x.coeffs_dict().items()}
    if _scalar_orders is not None:
        y = {0: math.sqrt(a)}
        for n in range(1, terms):
            acc = 0.0
            for k in range(1, n):
                if k in y and (n - k) in y:
                    acc += y[k] * y[n - k]
            v = (_scalar_orders.get(n, 0.0) - acc) / (2.0 * y[0])
            if v != 0.0 or n in _scalar_orders:
                y[n] = v
        out = _like(x, {-float(n): c for n, c in y.items()})
        return _truncate_order(out, _tighter(terms - 1, _min_complete(x)))

    sqrt_a = math.sqrt(a)
    h_part = x - R(a)
    ratio = h_part / R(a)

    def binom(n):
        if n == 0:
            return 1
        result = 1
        for k in range(n):
            result *= (0.5 - k)
        return result / math.factorial(n)

    result = _like(x, {_unit_dim(x): sqrt_a})
    power = _like(x, {_unit_dim(x): 1.0})

    for n in range(1, terms):
        power = power * ratio
        result = result + binom(n) * sqrt_a * power

    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))


@_honours_terms
def tan(x, terms=12):
    """Tangent function via sin/cos"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    _s, _c = sin(x, terms), cos(x, terms)
    # The quotient CARRIES its completeness: the operands' bounds through the
    # division, and the division's own cut.  Nothing is inferred from which
    # orders happen to be present.
    _q = _s / _c
    return _truncate_order(_q, _tighter(_q._complete, _min_complete(_s, _c, x)))

# =============================================================================
# INVERSE TRIGONOMETRIC FUNCTIONS
# =============================================================================

def _expandable_about_st(c):
    """True when c has a standard part and it is not zero.

    _reciprocal expands 1/(s + u) about the standard part s, so it needs one.
    This test was written inline as `b.st() != 0.0`, which does not ANSWER the
    question when the divisor is unbounded -- it raises.  Any divisor spanning
    two axes with no standard part therefore crashed the division itself:
    ln(h) + h is dominated by ln(h) and has no standard part, yet
    h / (ln(h) + h) is an ordinary infinitesimal and perfectly representable.
    Measured before this: all twelve divisions by ln(h) + h raised
    StandardPartUndefinedError from inside __truediv__.

    Falling through to deconvolve is the right answer here rather than a
    fallback: the reciprocal series is what needs the expansion point, and
    without one there is nothing for it to do.
    """
    try:
        return c.st() != 0.0
    except StandardPartUndefinedError:
        return False


def _spans_multiple_axes(x):
    """True when x's non-standard terms live on more than one basis axis.

    The axis a dimension belongs to is the index of its first non-zero
    component: (-1, 0) is the power axis, (0, -1) the first log axis.  A value
    touching both is the case lexicographic long division cannot order.
    """
    axes = set()
    for d in x.c:
        if not isinstance(d, tuple):
            if d != 0:
                axes.add(0)
            continue
        # EVERY non-zero component, not just the leading one.  Breaking at the
        # first one filed a dimension like (-1, -1) -- which touches the power
        # axis AND a log axis -- under the power axis alone, so a value that
        # genuinely spans both reported False and division fell through to
        # lexicographic long division, the one case the docstring above says it
        # cannot order.
        #
        # Measured on exp(-t)/(1 + eps*t) with t carrying a quadrature seed:
        # the long-division route returned 4 power grades (complete to order 3)
        # where the reciprocal route returns 10 (order 9), for the same
        # expression written as exp(-t)*(1/(1+eps*t)).  That is what capped the
        # Euler-Stieltjes derivation at four coefficients.
        for i, e in enumerate(d):
            if e != 0:
                axes.add(i)
        if len(axes) > 1:
            return True
    return len(axes) > 1


def _lane_d1(x, axis=1):
    """First derivative with respect to the variable on `axis`.

    Composite.d(1) reads the POWER axis, which is correct when the variable is
    seeded there.  Once the integrator moved its seed to a lane, every reader
    of that derivative has to follow -- the curve tangent did not, so every
    line integral came back exactly 0.0: r'(t) read off an axis the parameter
    no longer occupies.
    """
    from composite.backends.vector_dim_backend import as_vec
    if not isinstance(x, Composite):
        return 0.0
    total = 0.0
    for d, c in x.c.items():
        v = as_vec(d)
        if len(v) > axis and v[axis] == -1 and v[0] == 0 and \
                all(e == 0 for i, e in enumerate(v) if i not in (0, axis)):
            total += c
    return total


@_honours_terms
def _reciprocal(x, terms=15):
    """Compute 1/x via geometric series. Internal helper."""
    a = x.st()
    if a == 0.0:
        # 7.1: exactly zero, not "small".  A tolerance here refused to invert
        # perfectly good composites whose standard part happened to be tiny.
        raise ZeroDivisionError("Cannot compute 1/x at x=0")
    h_part = x - R(a)
    ratio = h_part / R(-a)
    result = _like(x, {_unit_dim(x): 1/a})
    power = _like(x, {_unit_dim(x): 1.0})
    for n in range(1, terms):
        power = power * ratio
        result = result + power / a
    return _truncate_order(result,
                           _tighter(_complete_order(_infinitesimal_terms(ratio),
                                                    terms),
                                    _min_complete(x)))

def _maclaurin_odd(x, coeffs, terms):
    """sum coeffs[n] * x**(2n+1) for a composite x with ZERO standard part.

    asin and atan normally go the derivative route: form 1/sqrt(1-u**2) or
    1/(1+u**2), then antidifferentiate w.r.t. eps.  That works on the power
    axis, where the shift is one dimension and the divisor is the order.  It
    does NOT generalise across log axes -- integral of h**k * B1**m dh is not
    a single term unless k == -1, so there is no shift to apply.

    When the standard part is zero the Maclaurin series can just be composed
    instead, which needs no derivative, no antiderivative and no chain rule,
    and works on any axis at any depth.  Powers of x are formed once and
    reused.
    """
    x2 = x * x
    term = x                       # x**1
    out = None
    for n, c in enumerate(coeffs):
        if c != 0.0:
            piece = term * c
            out = piece if out is None else out + piece
        term = term * x2
    if out is None:
        return _like(x, {})
    # RECORD THE TRUNCATION.  Summing T odd powers omits c_T * x**(2T+1), whose
    # lowest order is m*(2T+1) with m the lowest order in x -- so every order
    # below that is complete and nothing above it is.  Leaving this out let
    # asin(h) and atan(h) report _complete = None, i.e. EXACT, while being a
    # 15-term truncation reaching order 29: the same claim-more-than-you-have
    # failure the completeness bookkeeping exists to stop, reintroduced by a
    # second code path that skipped it.
    nz = _infinitesimal_terms(x)
    bound = None
    if nz:
        m = min(_lead_order(d) for d in nz)
        if m > 0:
            bound = m * (2 * len(coeffs) + 1) - 1
    return _truncate_order(out, _tighter(bound, _min_complete(x)))


@_honours_terms
def atan(x, terms=15):
    """Arctangent for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.atan, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.atan(a)})
    if a == 0.0:
        # atan(z) = z - z**3/3 + z**5/5 - ...
        return _maclaurin_odd(
            x, [(-1.0) ** n / (2 * n + 1) for n in range(terms)], terms)
    # Built AFTER both early returns.  Standing above them, this was computed
    # and then discarded on every bare standard part -- and for a vector-keyed
    # one it did not merely waste the work: R(1) + x*x is wholly zero at no
    # point, but _reciprocal divides by R(-a), and dividing sends both operands
    # through R1, whose `dims[0] - 1` is a scalar spelling.  atan was the only
    # transcendental still raising TypeError on a log-axis argument, and this
    # ordering was the whole reason.
    one_plus_x2 = R(1) + x * x
    deriv = _reciprocal(one_plus_x2, terms)
    _lead = math.atan(a)
    result = {} if _lead == 0.0 else {_zero_dim_like(x): _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = _dim_shift(dim, -1)
        if _dim_order(new_dim) != 0:
            result[new_dim] = coeff / abs(_dim_order(new_dim))
    out = _like(x, result)
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
        # TRUNCATE to what is complete, as sqrt does.  Recording the bound and
        # then handing back the orders past it means the caller reads a
        # coefficient that is a partial sum wearing the shape of a finished
        # one: at the default depth atan's order 16 moved by 5.0 once the
        # series could actually be deepened.
        out = _truncate_order(out, out._complete)
    return out

@_honours_terms
def asin(x, terms=15):
    """Arcsine for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.asin, x)
    a = x.st()
    if abs(a) >= 1:
        raise ValueError("asin requires |standard part| < 1")
    inner = R(1) - x * x
    deriv = _reciprocal(sqrt(inner, terms), terms)
    if not _has_infinitesimal_part(x):
        return Composite({0: math.asin(a)})
    if a == 0.0:
        # asin(z) = sum C(2n,n) / (4**n (2n+1)) * z**(2n+1)
        _co = []
        for n in range(terms):
            _co.append(math.comb(2 * n, n) / (4.0 ** n * (2 * n + 1)))
        return _maclaurin_odd(x, _co, terms)
    _lead = math.asin(a)
    result = {} if _lead == 0.0 else {_zero_dim_like(x): _lead}   # R6, see ln
    deriv = deriv * _d_deps(x)          # chain rule; see _d_deps
    for dim, coeff in deriv.c.items():
        new_dim = _dim_shift(dim, -1)
        if _dim_order(new_dim) != 0:
            result[new_dim] = coeff / abs(_dim_order(new_dim))
    out = _like(x, result)
    # Same order shift as antiderivative, and the same reason to record it:
    # this dict is built directly, so nothing else would.  Read the bound from
    # deriv AFTER the chain-rule multiply, which has already taken the min of
    # 1/sqrt(1-u^2) and u'.
    _c = _min_complete(deriv)
    if _c is not None:
        out._complete = _c + 1
        # As in atan: truncate to what is complete rather than hand back
        # partial sums past the bound.  acos is asin, so it follows.
        out = _truncate_order(out, out._complete)
    return out

@_honours_terms
def acos(x, terms=15):
    """Arccosine for composite numbers."""
    # WITHOUT THIS the series stops at the function's own default,
    # whatever depth the caller asked for.  _derivative_scope raises
    # _min_terms so a 24th derivative can be taken; atan/asin/acos read
    # `terms` raw, capped at grade -15 for terms=15, 25 and 40 alike,
    # and returned EXACTLY 0.0 for every order past it -- a silent zero
    # where atan^(16)(0.4) is -7.7e+10.  sin, cos, exp, ln, sqrt all
    # call this on their first line.
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({0: math.pi / 2})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    return R(math.pi / 2) - asin(x, terms)

# =============================================================================
# HYPERBOLIC FUNCTIONS
# =============================================================================

@_honours_terms
def sinh(x, terms=15):
    """Hyperbolic sine: (exp(x) - exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    return (exp(x, terms) - exp(-x, terms)) / 2

@_honours_terms
def cosh(x, terms=15):
    """Hyperbolic cosine: (exp(x) + exp(-x)) / 2"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({0: 1.0})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    return (exp(x, terms) + exp(-x, terms)) / 2

@_honours_terms
def tanh(x, terms=15):
    """Hyperbolic tangent: sinh(x) / cosh(x)"""
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        # The APPROACH to +-1 is e^(-2|u|).  For a power infinity, u ~ c/h, that
        # is exp(-2c/h): below every power of h, so the limit alone is the whole
        # answer at every order.  For a LOG infinity, u ~ c*ln(1/h), it is
        # h**(2|c|) -- an ordinary power, and dropping it was wrong:
        # tanh(ln(h) + h) came back |-1|_0, claimed exact, where the function is
        # -1 + 2h**2 e**(2h) - ...  exp already carries log terms, so say tanh
        # in it.
        if _infinite_part_is_logarithmic(x):
            e2 = exp(x * 2.0, terms)
            return (e2 - R(1)) / (e2 + R(1))
        return _bounded_at_inf(math.tanh, x)
    # ADDITION FORMULA about the standard part a, with u the infinitesimal rest:
    #
    #     tanh(a + u)  =  t + sech(a)**2 * T / (1 + t*T),   t = tanh a, T = tanh u
    #
    # sinh(x)/cosh(x) builds every coefficient of order >= 1 as a difference of
    # numbers of size cosh(a) whose answer is of size sech(a)**2: an absolute
    # error of eps, so a RELATIVE error of eps*e**(2a) -- 1e-6 at a = 12, and
    # tanh(x**3) at 2.3 missed d5 by 1e-9.  Here T is the series of a value with
    # no standard part, nothing in it is large, and sech(a)**2 is taken as
    # 4e/(1+e)**2 with e = exp(-2|a|): no cancellation and no overflow, where
    # cosh(a) overflowed past a = 710.
    _inf = {d: c for d, c in x.c.items() if _dim_nonzero(d) and c != 0.0}
    a = x.st()
    if not _inf:
        return Composite({0: math.tanh(a)})
    u = _like(x, _inf)
    _s, _c = sinh(u, terms), cosh(u, terms)
    # Carried completeness, as in tan.
    _q = _s / _c
    T = _truncate_order(_q, _tighter(_q._complete, _min_complete(_s, _c, x)))
    if a == 0.0:
        return T                       # t = 0 and sech**2 = 1: the formula is T
    t = math.tanh(a)
    e = math.exp(-2.0 * abs(a))
    sech2 = 4.0 * e / (1.0 + e) ** 2
    core = T / (R(1) + T * t)
    # Scaled through the backend: sech2 is a positive number that may underflow
    # to 0.0 for |a| > ~370, and a Python 0.0 entering `core * sech2` would be
    # an expressed zero, which R1 converts.  An underflowed scale is not that.
    scaled = Composite._wrap(core._backend.scalar_multiply(core._data, sech2),
                             core._backend, complete=core._complete,
                             denot=_denot_of(core))
    # The division above runs past T's bound; return only what is complete,
    # as every transcendental does (test_series_completeness audits it).
    out = R(t) + scaled
    return _truncate_order(out, out._complete)


# =============================================================================
# ERROR FUNCTION AND THE NORMAL CDF
# =============================================================================
#
# Each of these is an integral of a Gaussian, and both pieces already exist:
# exp() works on a composite, and integrating in the infinitesimal is a
# DIMENSION SHIFT that antiderivative() performs.  So
#
#     f(a + h) = f(a) + integral_0^h f'(a + s) ds
#              = antiderivative( f'(x) * dx/dh, f(a) )
#
# is the whole implementation -- no series coefficients to derive, no table.
#
# THE dx/dh FACTOR IS NOT OPTIONAL.  antiderivative() integrates with respect to
# the composite's own infinitesimal, so the chain rule has to be supplied the way
# atan and asin already supply it, with _d_deps(x).  Without it these three were
# correct for a plain seed -- where dx/dh is 1 -- and silently wrong for every
# composed argument: d/dx erf(2x) came back 0.4151 where it is 0.8302, exactly
# the missing factor of 2, and likewise for erfc and Phi.  A plain seed is what
# a direct test uses, which is why it survived.
# The standard part comes from math so it keeps full precision; the
# infinitesimal part comes from the algebra so derivatives and limits work.
# erfc and Phi use math.erfc rather than 1 - erf, which loses its significant
# digits once erf(a) approaches 1.

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _chained(deriv, x):
    """f'(x) * dx/dh, truncated back to what f'(x) actually knew.

    antiderivative() integrates in h, so the chain factor is required -- without
    it erf, erfc and Phi were right only for a plain seed, where dx/dh is 1, and
    silently wrong for every composed argument (d/dx erf(2x) came back at half
    its value).

    The truncation is the other half.  dx/dh has more than one term whenever x
    does, so the product CONVOLVES the series one order further than f'(x) was
    computed to -- and that new top order is missing the contribution of a term
    nobody evaluated.  Emitting it means the answer moves when `terms` is raised,
    which is exactly what completeness promises it will not do.  So the product
    is cut back to f'(x)'s own bound: one fewer order than before, and every
    order that remains is one the series actually knows.
    """
    order = getattr(deriv, "_complete", None)
    return _truncate_order(deriv * _d_deps(x), order)


@_honours_terms
def erf(x, terms=15):
    """Error function.  d/dx erf = (2/sqrt(pi)) exp(-x^2)."""
    if isinstance(x, (int, float)):
        return Composite({0: math.erf(float(x))})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.erf, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.erf(a)})
    return antiderivative(_chained(_TWO_OVER_SQRT_PI * exp(-(x * x), terms), x),
                          math.erf(a))


@_honours_terms
def erfc(x, terms=15):
    """Complementary error function.  d/dx erfc = -(2/sqrt(pi)) exp(-x^2).

    Not 1 - erf(x): that cancels away the answer for x beyond about 2, where
    erf is 0.995 and the difference is the part that matters.
    """
    if isinstance(x, (int, float)):
        return Composite({0: math.erfc(float(x))})
    if _is_nothing(x):
        # R6: adding nothing is a no-op, so f(nothing) is whatever term
        # of the series does NOT carry x -- the constant, if there is
        # one.  Returning Composite({}) unconditionally made exp(nothing)
        # nothing, when 1 + nothing + nothing + ... is 1.  Going the other
        # way and dropping this branch is worse: the scalar fast path
        # below reads st(nothing) as 0 and returns Composite({0: f(0)}),
        # an EXPRESSED zero, which R1 then converts -- that is where
        # acos(nothing) picked up a spurious |-1|_-1.
        return Composite({0: 1.0})
    # R1 on the ARGUMENT.  Arithmetic runs its operands through
    # _operands, which ends in _r1; these did not, so a cancellation
    # arrived here as |0|_0 instead of |1|_-1 and the infinitesimal was
    # discarded.  sqrt(1 - (Z*alpha)**2) at Z*alpha = 1 returned 0
    # rather than |1|_-0.5 -- the half grade IS the branch point.
    x = _r1(x)
    if _has_positive_dims(x):
        return _bounded_at_inf(math.erfc, x)
    a = x.st()
    if not _has_infinitesimal_part(x):
        return Composite({0: math.erfc(a)})
    return antiderivative(_chained(-_TWO_OVER_SQRT_PI * exp(-(x * x), terms), x),
                          math.erfc(a))


@_honours_terms
def normal_cdf(x, terms=15):
    """Standard normal CDF.  Phi'(x) = exp(-x^2/2)/sqrt(2 pi).

    Evaluated as 0.5*erfc(-x/sqrt(2)) at the standard part, which stays
    accurate in the left tail where 0.5*(1 + erf) does not.
    """
    if isinstance(x, (int, float)):
        return Composite({0: 0.5 * math.erfc(-float(x) / math.sqrt(2.0))})
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(
            lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)), x)
    a = x.st()
    base = 0.5 * math.erfc(-a / math.sqrt(2.0))
    if not _has_infinitesimal_part(x):
        return Composite({0: base})
    return antiderivative(
        _chained(_ONE_OVER_SQRT_2PI * exp(-(x * x) * 0.5, terms), x), base)


Phi = normal_cdf          # the name the finance literature uses



# =============================================================================
# REAL-VALUED POWERS
# =============================================================================

def _is_clean_dyadic(s, max_k=40):
    """True when s is m/2**k for a small k -- a number the index set can hold.

    Every float64 IS a dyadic rational, so "is it dyadic" is trivially yes and
    useless as a test.  What matters is whether the value the CALLER meant is
    dyadic: 0.5, 0.25, 3/8 are; 1/3 and 1/7 are not, and float64 merely holds
    their nearest neighbour.  Scaling by 2**k and asking for an integer
    separates the two.
    """
    v = float(s)
    if v != v or v in (float("inf"), float("-inf")):
        return False
    for k in range(max_k + 1):
        if (v * (1 << k)).is_integer():
            return True
    return False


def _warn_non_dyadic_exponent(x, s):
    """Audible when a root asks for an index the set cannot hold exactly.

    WHEN THIS DOES NOT APPLY, which is most of the time.  If x has a non-zero
    standard part then ln(x) is an ordinary series with INTEGER dimensions, so
    the exponent only scales coefficients and never reaches the index at all:
    power(_seeded(8), 1/3) comes back with dims -14..0 and values correct to
    2.2e-16.  Nothing is approximate about its structure.  An earlier version
    of this guard refused that case, which was simply wrong.

    WHEN IT DOES.  If x is a pure infinitesimal or infinity, ln(x) sits on the
    log axis and exp moves the exponent onto the POWER axis, where it becomes
    the index.  Dimensions are exact under {+, -, /2} -- the dyadics -- so 1/2,
    1/4, 3/8 land exactly and 1/3, 1/7 do not.

    WHY A WARNING AND NOT A REFUSAL.  The index is still right to sixteen
    digits, which is no worse than any other floating-point result, and asking
    for h**(1/3) is a reasonable thing to do.  What is NOT ordinary rounding is
    the shape of the failure: h**(1/7) raised to the seventh lands at
    -0.9999999999999998, a SEPARATE term from the -1 it should have merged
    with, so coeff(-1) returns the wrong number rather than a slightly wrong
    one.  That deserves to be heard, not blocked.  (h**(1/3) closes only
    because rounding happens to land favourably -- luck, not a guarantee.)
    """
    if _is_clean_dyadic(s):
        return
    if (getattr(get_backend(), "EXACT_DIMS", False)
            or getattr(x._backend, "EXACT_DIMS", False)):
        # The whole warning is about float64 dimensions, so it has nothing to
        # say when the grade is held exactly.  It fired on fractional_numpy for
        # power(h, 1/3) and told the caller the index "will not always
        # recombine" immediately after it recombined exactly: h**(1/3) cubed
        # came back {-1: 1.0}, and so did 1/7 and 1/23.
        #
        # BOTH backends are asked, and the ACTIVE one first.  Asking only
        # x._backend missed every case that matters: ZERO and the other module
        # constants are built once at import, so they carry whichever backend
        # was installed then, and power(ZERO, 1/3) under use_fractional_numpy()
        # still warned.  The result is built on the active backend, so that is
        # what decides exactness; x._backend is kept in the test because
        # _operands converts toward the richer representation, so a value
        # carrying an exact backend stays exact regardless of the global.
        return
    try:
        if x.st() != 0.0:
            return                  # exponent stays in the coefficients
    except StandardPartUndefinedError:
        # UNBOUNDED IS NOT "HAS A STANDARD PART".  st() raises when the dominant
        # grade is positive, and this line called it unguarded, so the helper
        # threw instead of warning -- power(ln(1/h), 1/3) died with
        # StandardPartUndefinedError from inside a diagnostic.  That is the very
        # case the docstring above says this was extended to cover, so it had
        # never once fired there; it had crashed.  No standard part means
        # nothing absorbs the exponent and it DOES land on the index, which is
        # the condition to warn about, so fall through rather than return.
        pass
    if not any(_dim_nonzero(d) for d in x.c):
        return                      # no dimension at all
    # _dim_order reads only the POWER component, so a pure log-axis term like
    # (0, 1) looked dimensionless and the warning never fired for
    # power(ln(1/h), 1/3) -- which lands a fractional index on the LOG axis,
    # exactly the same failure one axis over.
    import warnings
    warnings.warn(
        f"power(x, {s!r}) on a composite with no standard part: the exponent "
        f"becomes the DIMENSION, and dimensions are exact only under "
        f"{{+, -, /2}} (the dyadics). {s!r} is not m/2**k, so the index is a "
        f"float approximation -- accurate to ~1e-16, but it will not always "
        f"recombine: h**(1/7) to the 7th lands at -0.9999999999999998, a "
        f"separate term from -1, so coeff(-1) then reads the wrong value. "
        f"sqrt and repeated halving are exact and always will be, and "
        f"config.use_fractional_numpy() makes every denominator exact.",
        stacklevel=3)


def power(x, s, terms=15):
    """x^s via exp(s * ln(x)).  The exponent may be real OR a Composite.

    A composite exponent used to be routed through R(s), which calls float(s)
    -- that is st(s), and the standard part of an infinity is 0.  So the
    exponent was silently replaced by an expressed zero and the function
    computed x^0:

        power(1+x, 1/x)  returned 1.0   (is e)

    because 1/x is |1|_1, float(|1|_1) is 0.0, and R(0.0) is |0|_0.  The
    exponent has to stay a composite, which is what __pow__ already did --
    hence a ** b and exp(ln(a)*b) both gave e while power() did not.
    """
    if isinstance(x, (int, float)):
        x = Composite({0: float(x)})
    if isinstance(s, int):
        return x ** s
    if isinstance(s, Composite):
        return exp(s * ln(x, terms), terms)
    _warn_non_dyadic_exponent(x, s)
    return exp(R(s) * ln(x, terms), terms)


# =============================================================================
# HIGH-LEVEL API: AUTOMATIC TRANSLATION
# =============================================================================

def _d_deps(x):
    """d(x)/d(eps): the derivative of a composite w.r.t. its own infinitesimal.

    THE POWER AXIS.  A term c*eps**k sits at dim -k and differentiates to
    k*c*eps**(k-1) at dim -(k-1).  In (e0, e1, ...) form, where a dim means
    c * (1/h)**e0 * B1**e1 * B2**e2 ... , that is e0 += 1 with the coefficient
    scaled by -e0.

    THE LOG AXES.  These are not constants -- they are functions of h, so the
    chain rule reaches across scales:

        B1 = ln(1/h)        dB1/dh = -1/h
        B2 = ln(B1)         dB2/dh = -(1/h) * B1**-1
        B_k                 dB_k/dh = -(1/h) * B1**-1 ... B_{k-1}**-1

    so differentiating B_k**e_k gives  -e_k * (1/h) * B1**-1 ... B_{k-1}**-1
    * B_k**(e_k - 1): e0 += 1, every axis from 1 to k drops by one, and the
    coefficient is scaled by -e_k.  ONE TERM PER AXIS THE DIMENSION TOUCHES,
    summed.

    Returning only the power-axis part -- which is what this did before -- made
    atan(1/ln(1/h)) come back as NOTHING, since a pure log-axis term has no
    power component at all and the whole factor vanished.

    THE CHAIN RULE FACTOR is what asin and atan need: they form 1/sqrt(1-u**2)
    (resp. 1/(1+u**2)) and antidifferentiate w.r.t. eps, but
    d/deps asin(u(eps)) = u' / sqrt(1-u**2).
    """
    # Differentiating LOSES an order: the top coefficient of x produces the top
    # of x', and there is nothing above it to produce the next.  Not recording
    # that made asin(sin x) claim order 12 on 11 sound ones.
    _c = getattr(x, "_complete", None)

    def _tag(out):
        if _c is not None:
            out._complete = _c - 1
        return out

    if not any(isinstance(d, tuple) for d in x.c):
        # Scalar fast path: keeps scalar work off the vector backend entirely.
        return _tag(Composite({_dim_shift(d, 1): c * abs(_dim_order(d))
                               for d, c in x.c.items()
                               if _dim_negative(d) and c != 0.0}))

    from composite.backends.vector_dim_backend import as_vec, canon, ensure_depth
    out = {}
    for d, c in x.c.items():
        if c == 0.0:
            continue
        v = list(as_vec(d))
        for k, ek in enumerate(v):
            if ek == 0:
                continue
            nd = list(v)
            nd[0] += 1                      # every axis contributes a 1/h
            for j in range(1, k + 1):       # ...and drops axes 1..k by one
                nd[j] -= 1
            key = canon(tuple(nd))
            out[key] = out.get(key, 0.0) + c * (-ek)
    out = {k: val for k, val in out.items() if val != 0.0}
    if not out:
        return _tag(Composite({}))
    ensure_depth(max(len(k) if isinstance(k, tuple) else 2 for k in out))
    return _tag(_vec_composite(out))


def _reject_pole(result, what: str, at: float):
    """Raise if the seeded result carries a POLE, instead of quietly dropping it.

    A seeded evaluation puts the regular part on dimensions <= 0 and any
    singular part on dimensions > 0.  Every extractor here reads dimensions
    <= 0 only, so f(x) = 1/(1-cos x) at 0 -- whose composite is correctly
    |2|_+2 + |1/6|_0 + |1/120|_-2, the Laurent series 2/x^2 + 1/6 + x^2/120 --
    came back as a clean-looking [1/6, 0, 1/120] with the double pole silently
    discarded.  There is no Taylor series at a pole; returning the regular part
    as if there were is the wrong answer, not a partial one.

    Only NONZERO positive coefficients count.  An expressed zero up there is
    legitimate and must not trip this: INF - INF = |0|_+1 by the canon rule that
    a constructed dimension is retained.
    """
    poles = {d: c for d, c in result.coeffs_dict().items()
             if _dim_positive(d) and c != 0.0}
    if poles:
        order = max(poles)
        raise ValueError(
            f"{what} at {at}: f has a pole of order {order} here, so no Taylor "
            f"series exists.  The singular part IS present in the composite, on "
            f"dimensions {sorted(poles)} (coefficients "
            f"{[poles[d] for d in sorted(poles)]}); these extractors read "
            f"dimensions <= 0 only.  Read the full Laurent series off the "
            f"seeded result directly, or expand about a regular point."
        )


def derivative(f: Callable, at: float, terms: int = 12) -> float:
    """Compute f'(at) automatically."""
    with _extraction_scope():
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "derivative", at)
        return result.d(1)


def nth_derivative(f: Callable, n: int, at: float, terms: int = 12) -> float:
    """Compute f^(n)(at) - the nth derivative at a point."""
    with _derivative_scope(n, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "nth_derivative", at)
        return result.d(n)


def all_derivatives(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Compute [f(at), f'(at), f''(at), ...] up to nth derivative."""
    with _derivative_scope(up_to, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "all_derivatives", at)
        return [result.st()] + [result.d(n) for n in range(1, up_to + 1)]


def limit(f: Callable, as_x_to: float, terms: int = 12,
          dir: str = "both", fallback: bool = False) -> float:
    """Compute lim(x→as_x_to) f(x) via composite evaluation.

    Evaluates f at a composite infinitesimal.  Bounded transcendentals
    (sin, cos, atan, asin, acos, tanh) evaluate at st(x) for infinite
    arguments, so oscillatory limits like x·sin(1/x) resolve algebraically.

    Positive dims in the result indicate unbounded divergence (e.g. exp) --
    unless every one is at float rounding size against the finite part, the
    residue of a cancellation floats could not make exact; those are dropped
    with a RoundingResidueWarning (see LIMIT_ROUNDING_RESIDUE).
    Domain errors (ln(0), sqrt(-x)) raise LimitUndecidableError, or fall
    back to integral averaging with ``fallback=True``.

    Args:
        f:         Function to evaluate.
        as_x_to:   Point to approach (float, float('inf'), or Composite INF).
        terms:     Truncation order for transcendentals.
        dir:       Direction — "both" (default), "+" (right), "-" (left).
        fallback:  If True, use integral averaging when algebraic fails.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _limit_impl(f, as_x_to, terms, dir, fallback)


def _limit_probe(f, as_x_to, dir, terms, _is_inf, why):
    """Probe at real points when the algebra could not answer.

    Used for two cases that look different and behave the same: a result of
    NOTHING, and a sub-expression with no composite at all (sin(1/h)).  In
    both the expression as a WHOLE may still converge -- x*sin(1/x) does --
    so the limit is not refused until probing fails.

    It never claims a limit on one sample.  At infinity two windows must
    AGREE; at a finite point the extrapolation has to succeed.  That is what
    keeps sin(x) at infinity from being reported as 0 by an integral average
    that happens to cancel.
    """
    if _is_inf:
        v1 = _limit_at_inf_fallback(f, as_x_to, n=500, width=50.0)
        v2 = _limit_at_inf_fallback(f, as_x_to, n=500, width=100.0)
        if math.isfinite(v1) and math.isfinite(v2):
            if abs(v1) < 1e-4 and abs(v2) < 1e-4:
                return 0.0
            if abs(v1 - v2) < 1e-3 * (abs(v1) + abs(v2)):
                return v2
    else:
        extrap = _limit_extrapolate(f, as_x_to, dir, terms)
        if extrap is not None:
            return extrap
    raise LimitDoesNotExistError(why)


def _limit_impl(f, as_x_to, terms, dir, fallback):
    """Internal implementation of limit(), wrapped to suppress numpy warnings."""
    # Normalize: accept Composite INF/-INF as well as float('inf')
    _is_inf = False
    if isinstance(as_x_to, Composite):
        max_d = as_x_to.max_positive_dim()
        if max_d is not None:
            coeffs = as_x_to.coeffs_dict()
            _is_inf = True
            as_x_to = float('inf') if coeffs.get(max_d, 0) > 0 else float('-inf')
        elif _has_infinitesimal_part(as_x_to):
            # No positive dimension, so not an infinity -- and it carries an
            # infinitesimal part, so st() would silently throw that away and
            # answer a DIFFERENT question.  "Approach 2 + eps" is not a limit
            # point; "approach 2" is.  A composite that is purely dimension 0
            # falls through and collapses losslessly, which is fine.
            raise TypeError(
                f"limit(as_x_to={as_x_to}): the point to approach must be a "
                f"real, float('inf'), or a composite INFINITY. This carries an "
                f"infinitesimal part, and taking its standard part would "
                f"silently change which limit is computed. Pass st() explicitly "
                f"if that is what you meant.")
        else:
            as_x_to = as_x_to.st()

    # Build the evaluation point
    if as_x_to == float('inf'):
        x = INF
        _is_inf = True
    elif as_x_to == float('-inf'):
        x = -INF
        _is_inf = True
    elif dir == "-":
        x = -ZERO if as_x_to == 0 else R(as_x_to) - ZERO
    else:
        x = _seeded(as_x_to)

    try:
        result = f(x)
    except TypeError:
        raise CompositionError(
            "Function not composable with composite arithmetic")
    except LimitDoesNotExistError:
        # Division by nothing (∅) — denominator is indeterminate.
        # The limit provably does not exist. Don't try to recover.
        raise
    except NotRepresentableError:
        # DO NOT RECOVER.  A sub-expression has no composite -- sin(1/h) takes
        # every value in [-1, 1] and a composite holds one value at one grade.
        #
        # Probing was tried and is the wrong answer even when it looks right.
        # It got x*sin(1/x) -> 0, which is the true limit, but only by
        # sampling: the bound |sin| <= 1 is exactly the information the raise
        # declined to carry, so the algebra cannot reach that limit and the
        # probe is guessing from points.  On x/sin(1/x) the same probe
        # returned 0.0 for a function that is UNBOUNDED -- sin(1/x) passes
        # through zero at x = 1/(k*pi), where the quotient blows up -- and a
        # single-resolution probe cannot see it: its maximum reads 1.006 at
        # 2,000 samples and 20.97 at 32,000.  A method that answers 0.0 for
        # both a convergent and a divergent case is not a fallback, it is a
        # coin toss with a confident face.
        #
        # So the refusal propagates.  The limit is not computed rather than
        # computed by other means.
        raise
    except (ValueError, ZeroDivisionError):
        # Domain error (ln(0), sqrt(0), etc.) — try composite extrapolation
        # from a nearby point where the function is well-defined.
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Algebraic evaluation failed (domain error at limit point). "
            "Use fallback=True for integral averaging.")

    # Terms above grade 0 at rounding size against the finite part: a
    # cancellation that float coefficients could not make exact.  See
    # RoundingResidueWarning.
    if isinstance(result, Composite):
        _all = result.coeffs_dict()
        _pos = [abs(c) for d, c in _all.items() if _dim_positive(d) and c != 0.0]
        _fin = [abs(c) for d, c in _all.items() if not _dim_positive(d)]
        _scale = max(_fin) if _fin else 0.0
        if _pos and _scale > 0.0 and max(_pos) <= LIMIT_ROUNDING_RESIDUE * _scale:
            _warnings.warn(
                "limit(): %d term(s) above grade 0, the largest %.1e of the "
                "finite part, read as float rounding residue of a cancellation "
                "and dropped; the limit is the grade-0 value. A genuine infinite "
                "part that small would be read the same way (bound: "
                "LIMIT_ROUNDING_RESIDUE = %.0e)." % (len(_pos), max(_pos) / _scale,
                                                    LIMIT_ROUNDING_RESIDUE),
                RoundingResidueWarning, stacklevel=3)
            return float(result.coeff(0))

    # Positive dims → unbounded divergence (from exp, ln, etc.)
    max_pos = result.max_positive_dim()
    if max_pos is not None:
        pos_coeffs = {d: c for d, c in result.coeffs_dict().items()
                      if _dim_positive(d)}
        signs = [c > 0 for c in pos_coeffs.values()]
        if all(signs):
            return INF
        elif not any(signs):
            return -INF
        # Mixed signs — try composite extrapolation before giving up
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Result has mixed-sign positive dimensions.")

    # Check for NaN/Inf contamination
    st_val = result.st()
    if not math.isfinite(st_val):
        if not _is_inf:
            extrap = _limit_extrapolate(f, as_x_to, dir, terms)
            if extrap is not None:
                return extrap
        if fallback:
            if _is_inf:
                return _limit_at_inf_fallback(f, as_x_to)
            return _limit_integral_fallback(f, as_x_to, dir)
        raise LimitUndecidableError(
            "Algebraic evaluation produced NaN/Inf.")

    # Nothing (∅) means an indeterminate value was involved. Check if the
    # overall expression still converges by probing at real points.
    if _is_nothing(result):
        return _limit_probe(
            f, as_x_to, dir, terms, _is_inf,
            "Result is indeterminate (nothing). Limit does not exist.")

    return st_val


def _limit_extrapolate(f, as_x_to, dir, terms, n_probes=6):
    """Extrapolate limit from nearby composite evaluations.

    Two strategies, tried in order:
      1. Taylor extrapolation: evaluate at probe, use eval_taylor(-eps) to
         extrapolate back to the limit point. Exact when Taylor converges.
      2. Value convergence: if Taylor overflows, use the st() values at
         decreasing probe distances. The values themselves converge.

    Returns the extrapolated value, or None if neither converged.
    """
    if dir == "-":
        signs = [-1]
    elif dir == "+":
        signs = [1]
    else:
        signs = [1, -1]

    for sign in signs:
        taylor_candidates = []
        value_candidates = []

        for k in range(n_probes):
            eps = 10 ** (-(k + 2))  # 1e-2, 1e-3, ..., 1e-7

            try:
                x_comp = _seeded(as_x_to + sign * eps)
                result = f(x_comp)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

            if not isinstance(result, Composite):
                result = R(float(result))

            st = result.st()
            if not math.isfinite(st):
                continue

            value_candidates.append(st)

            # Try Taylor extrapolation
            try:
                extrap = st + result.eval_taylor(-sign * eps)
                if math.isfinite(extrap):
                    taylor_candidates.append(extrap)
            except (OverflowError, ValueError):
                pass

        # Strategy 1: Taylor extrapolation converged
        if len(taylor_candidates) >= 2:
            if abs(taylor_candidates[-1] - taylor_candidates[-2]) < 1e-6 * (abs(taylor_candidates[-1]) + 1e-100):
                return taylor_candidates[-1]

        # Strategy 2: raw values converging (for cases like x^x where
        # Taylor overflows, or sqrt(x) where Taylor radius is too small).
        if len(value_candidates) >= 3:
            v = value_candidates
            d1 = abs(v[-1] - v[-2])
            d2 = abs(v[-2] - v[-3])
            # Values converging: differences shrinking
            if d1 < d2:
                # Tight convergence — last two very close
                if d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                    return v[-1]
            # Converging to 0: last few values all small (even if oscillating)
            if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                return 0.0

    return None


def _limit_integral_fallback(f, as_x_to, dir, a=1e-4, n=1000):
    """Compute limit via integral average: ∫f(x)dx / a over a small interval.

    Integration smooths oscillation — the average value over [as_x_to, as_x_to+a]
    converges to the limit even when point evaluation oscillates.
    """
    if dir == "-":
        lo, hi = as_x_to - a, as_x_to
    else:
        lo, hi = as_x_to, as_x_to + a

    # Avoid evaluating exactly at the singularity
    eps = a * 1e-10
    lo = lo + eps

    h = (hi - lo) / n
    total = 0.0
    for i in range(n):
        xi = lo + (i + 0.5) * h
        fi = f(R(xi) + ZERO)
        total += fi.st()
    total *= h

    return total / a


def _limit_at_inf_fallback(f, as_x_to, n=1000, width=100.0):
    """Compute limit at ±∞ via integral average over a large-x window.

    Evaluates ∫_M^{M+w} f(x)dx / w for large M.
    """
    sign = 1 if as_x_to == float('inf') else -1
    M = sign * 1e4
    lo = M
    hi = M + sign * width

    if lo > hi:
        lo, hi = hi, lo

    h = (hi - lo) / n
    total = 0.0
    for i in range(n):
        xi = lo + (i + 0.5) * h
        fi = f(R(xi) + ZERO)
        total += fi.st()
    total *= h

    return total / width


# Backward compatibility aliases
def limit_right(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute right-hand limit: lim(x→a⁺) f(x)"""
    return limit(f, as_x_to, terms=terms, dir="+")


def limit_left(f: Callable, as_x_to: float, terms: int = 12) -> float:
    """Compute left-hand limit: lim(x→a⁻) f(x)"""
    return limit(f, as_x_to, terms=terms, dir="-")


def taylor_coefficients(f: Callable, at: float, up_to: int = 5, terms: int = 12) -> List[float]:
    """Get Taylor series coefficients c_n = f^(n)(at)/n! of f around 'at'.

    Was missing the depth scope the other extractors have, so a high `up_to`
    quietly read zeros off dimensions the transcendentals never expanded to.
    """
    with _derivative_scope(up_to, terms):
        x = _seeded(at)
        result = f(x)
        _reject_pole(result, "taylor_coefficients", at)
        return [result.coeff(-n) for n in range(up_to + 1)]


# =============================================================================
# FIXED: antiderivative, _ensure_composite, _detect_singularity
# =============================================================================

def _antiderivative_term(dim, coeff):
    """Integrate ONE term with respect to h.  Returns {dim: coeff}, or None.

    A term is c * h**p * L**l with L = ln(1/h), p = -(power component) and l
    the first log component -- the reading `ln(h) = <-1_(0,1)>` fixes.  With
    q = p + 1, integration by parts gives a FINITE sum, since dL/dh = -1/h:

        q != 0:  int h**p L**l dh = h**q * sum_{j=0..l} l!/(l-j)! L**(l-j) / q**(j+1)
        q == 0:  int h**-1 L**l dh = -L**(l+1) / (l+1)

    With l = 0 and p a non-negative integer this is the old rule
    |c|_-n -> |c/(n+1)|_-(n+1).  It also covers what the old rule skipped: a
    FRACTIONAL grade (1/sqrt(h) -> 2 sqrt(h)), a POLE (1/h**2 -> -1/h), the
    simple pole 1/h (-> -ln(1/h)), and log-carrying terms (ln h -> h ln h - h).

    Grades keep their own type: a Fraction on the exact backend must not pass
    through float(), or 7/10 - 1 stops being -3/10.

    None for a negative or fractional log power (int h**p / L is the
    logarithmic integral, not a finite sum) or content on a deeper axis; the
    caller keeps the old behaviour there.
    """
    if isinstance(dim, tuple):
        d0, rest = dim[0], dim[1:]
        l = rest[0] if rest else 0
        if any(c != 0 for c in rest[1:]) or l < 0 or l != int(l):
            return None
        l = int(l)
    else:
        d0, l = dim, 0
    q = 1 - d0                                # p + 1, with p = -d0

    def key(power, logp):
        if logp == 0 and not isinstance(dim, tuple):
            return power
        from composite.backends.vector_dim_backend import canon
        return canon((power, logp))

    if q == 0:
        return {key(0, l + 1): -coeff / (l + 1)}
    out = {}
    fall = 1.0                                # l!/(l-j)!
    for j in range(l + 1):
        out[key(_dim_shift(d0, -1), l - j)] = coeff * fall / float(q) ** (j + 1)
        fall *= (l - j)
    return out


def antiderivative(f_composite: Composite, constant: float = 0) -> Composite:
    """Compute antiderivative via dimensional shift.

    Each |c|_{-n} -> |c/(n+1)|_{-(n+1)}, and in general _antiderivative_term:
    every grade, fractional or positive, and the first log axis.  Positive
    grades used to be skipped because the old divisor abs(_dim_order(new_dim))
    was wrong for them (dim 2 -> divisor 1); the right divisor is q = 1 - dim,
    signed, so 1/h**2 integrates to -1/h, |1|_2 -> |-1|_1.
    """
    terms = {}
    for dim, coeff in f_composite.c.items():
        piece = _antiderivative_term(dim, coeff)
        if piece is None:
            # Outside the closed form: the old power-axis shift, positive
            # grades still skipped as they were.
            if _dim_positive(dim):
                continue
            new_dim = _dim_shift(dim, -1)
            piece = {new_dim: coeff / abs(_dim_order(new_dim))}
        for k, v in piece.items():
            terms[k] = terms.get(k, 0.0) + v
    # The constant's key has to be the same KIND as the dimensions it sits
    # beside: sorted() cannot order a tuple against an int.
    vector = any(isinstance(k, tuple) for k in terms) or any(
        isinstance(d, tuple) for d in f_composite.c)
    if vector:
        from composite.backends.vector_dim_backend import canon, as_vec
        terms = {canon(as_vec(k) if isinstance(k, tuple) else (k, 0)): v
                 for k, v in terms.items()}
        zero = canon((0, 0))
    else:
        terms = {(int(k) if not isinstance(k, _Fraction)
                  and float(k).is_integer() else k): v
                 for k, v in terms.items()}
        zero = 0
    result = {zero: constant}
    result.update({k: v for k, v in terms.items() if k != zero})
    # Every order moves up by one, so a f sound to K integrates to one sound to
    # K+1.  Building the dict directly skips _truncate_order, which is where
    # the bound would otherwise be recorded -- dropping it here made asin, atan
    # and the derivative round trip all claim to be exact.
    _c = getattr(f_composite, "_complete", None)
    if vector and not getattr(f_composite._backend, "VECTOR_DIMS", False):
        out = _join(_vec_composite(result), f_composite)   # a log out of a scalar input
    else:
        out = _like(f_composite, result)
    if _c is not None:
        out._complete = _c + 1
    return out


def _ensure_composite(val):
    """Wrap plain float/int as Composite. Warns on silent degradation.

    IMPORTANT: Uses Composite(float(val)), NOT R(float(val)).
    In v3, R(0) = ZERO = |1|_{-1} which carries d(1)=1.
    A constant return value should be |val|_0 with d(1)=0.
    """
    if isinstance(val, Composite):
        return val
    import warnings
    warnings.warn(
        "integrate: f returned plain float — Taylor convergence disabled "
        "for this panel. Wrap your function to return Composite.",
        stacklevel=3)
    return Composite(float(val))


def _detect_singularity(f, x_near, x_away, panel_dx):
    """Detect power-law singularity at x_near using composite.

    ONE composite eval near x_near. Extracts exponent
    alpha = dist * f'(x) / f(x) where dist = distance from boundary.
    If -1 < alpha < 0: integrable singularity, compute analytically.

    Returns (handled, value, C, alpha) — 4-tuple.
    If not handled, C and alpha are 0.0.
    """
    _not_found = (False, 0.0, 0.0, 0.0)

    eps = abs(panel_dx) * 0.01
    x_test = x_near + eps if x_near < x_away else x_near - eps
    dist_from_boundary = abs(x_test - x_near)

    if dist_from_boundary < 1e-30:
        return _not_found

    try:
        fx = _ensure_composite(f(_seeded(x_test)))
        f_val = fx.st()
        f_d1 = fx.d(1)

        if abs(f_val) < 1e-100:
            return _not_found

        # Exponent: f(x) ~ C * |x - boundary|^alpha
        #   f'/f = alpha / dist  =>  alpha = dist * f'/f
        sign = 1.0 if x_near < x_away else -1.0
        alpha = dist_from_boundary * (sign * f_d1) / f_val

        if not (-1.0 < alpha < -0.01):
            return _not_found

        # Sanity: derivative ratio must indicate true singularity
        deriv_ratio = abs(f_d1 / f_val) * abs(x_away - x_near)
        if deriv_ratio < 10:
            return _not_found

        # Coefficient: f(x) = C * dist^alpha  =>  C = f(x) / dist^alpha
        C = f_val / (dist_from_boundary ** alpha)
        ap1 = alpha + 1

        # Return (True, C, alpha) so the caller can integrate over any sub-range.
        # Default: integrate over [0, full_dist]
        full_dist = abs(x_away - x_near)
        singular_val = C * (full_dist ** ap1) / ap1

        return True, singular_val, C, alpha

    except (ZeroDivisionError, OverflowError, ValueError):
        return False, 0.0, 0.0, 0.0


def definite_integral(f: Callable, a: float, b: float, terms: int = 12) -> float:
    """Compute ∫ₐᵇ f(x) dx, by meeting composites (integrate_jets).

    `terms` is kept for the signature and no longer used: jet depth is
    integrate_jets' own.
    """
    return integrate(f, a, b)

# =============================================================================
# MULTI-POINT STEPPED INTEGRATION
# =============================================================================


def _perturbation_seed(axis):
    """Unit infinitesimal on basis axis `axis` (1 = first non-power axis).

    The basis axes past 0 mean iterated logarithms, and this borrows them as
    independent perturbation directions.  That is sound HERE because the only
    way a log-axis dimension enters is ln/exp of an infinitesimal or infinite
    value, and a panel midpoint is neither -- ln(_seeded(t)) expands purely on
    the power axis for every finite non-zero t.  _box_exact probes for that
    and declines the exact path if the integrand brings its own log axes.
    """
    from composite.backends.vector_dim_backend import canon, ensure_depth
    ensure_depth(axis + 1)
    return _vec_composite({canon((0,) * axis + (-1,)): 1.0})


def _derivative_axis(x, axis):
    """d/d(variable on `axis`), staying a composite in every other lane.

    The inverse shift to _antiderivative_axis.  A term at lane order k becomes
    k times the term at order k-1, so d(sigma)/du comes back as a SERIES in
    (u, v) rather than a number at one point -- which is what a surface
    integral needs: the area element varies across the patch, and reading the
    tangent off as a float freezes it at the midpoint.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in x.c.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        k = v[axis]
        if k >= 0:
            continue                      # no content on this lane: d/d(var)=0
        v[axis] = k + 1
        key = canon(tuple(v))
        out[key] = out.get(key, 0.0) + coeff * abs(k)
    return _like(x, out)


def _integrand_needs_lane(f, points):
    """Does the integrand put its OWN content on the power axis?

    The variable of integration needs a lane of its own only when the power
    axis is already occupied.  Seeded on lane 1 the two are separable: the
    variable's contribution sits on axis 1, so any non-zero POWER component
    belongs to the integrand -- an R1 zero, or anything the caller's
    expression genuinely made infinitesimal.  `x*0 + 1` shows
    {(-1,-1): 1.0, (-1,0): 0.5}; x*x, exp(-x), sin(x), 1/(1+x) show nothing.

    When nothing is there the variable rides the power axis and the whole
    integral stays on the SCALAR backend.  That matters: seeded on a lane,
    every panel is a vector-dimension composite, which is excluded from
    SparseDenseBackend by design -- 83% of the backend operations in
    integral of exp(-x) over [0,inf), and 61% in a lateral Borel sum.

    Sampled, not proved.  An integrand that develops power-axis content only
    away from these points would be misclassified; the points are spread
    across the interval to make that unlikely, and the lane is taken whenever
    the probe cannot evaluate at all.
    """
    if not LANE_AUTO:
        return True                          # see LANE_AUTO
    for pt in points:
        try:
            fx = _ensure_composite(f(R(pt) + _perturbation_seed(1)))
        except Exception:
            return True                      # cannot tell: take the safe path
        for d, v in fx.c.items():
            if v == 0.0:
                continue
            power = d[0] if isinstance(d, tuple) else d
            if power != 0:
                return True
    return False


# OFF BY DEFAULT.  Putting the variable on the power axis when the integrand
# leaves it free is sound for R1 zeros and measurably faster -- but the lane
# turned out to do a second job nobody had written down: it separates the
# variable's derivatives from the integrand's, and the adaptive error estimate
# reads them separately.  With the variable on the power axis,
#
#     integral of sqrt(x) over [0,1]   lane 0: 2.2e-04     lane 1: 2.2e-13
#
# -- nine orders, on an integrand whose power axis the probe correctly reports
# as free.  The probe detects the R1 condition and cannot see this one, and
# three suites fail with it on.  So it is opt-in, for a caller that knows its
# integrand is plain.  Measured gain when it applies: improper integral 0.68x,
# a lateral Borel sum 1.11s -> 0.77s.
LANE_AUTO = True

LANE_PROBE_POINTS = 1      # raise it for an integrand whose structure varies


def _lane_probe_points(a, b, n=None):
    """Points inside [a, b] to ask _integrand_needs_lane about.

    ONE by default, and the count is not free: each probe evaluates the whole
    integrand with a lane seed, which is exactly the slow path the probe
    exists to avoid.  Measured on integral of exp(-x) over [0,inf) and on a
    lateral Borel sum:

        probes    improper integral    resum_median
          1          0.0130s              0.768s
          2          0.0144s              0.985s
          3          0.0157s              1.225s
          5          0.0198s              1.692s

    One is enough for the case the lane exists for, because an R1 zero is a
    property of the EXPRESSION rather than of the point: x*0+1, x-x+1,
    (x-0.5)*0+1 and sin(x)-sin(x)+1 are all caught by a single probe, and all
    integrate to 1.000000000000000 exactly.  Raise LANE_PROBE_POINTS for an
    integrand whose structure genuinely varies across the interval.
    """
    if n is None:
        n = LANE_PROBE_POINTS
    lo = a if math.isfinite(a) else (b - 10.0 if math.isfinite(b) else -1.0)
    hi = b if math.isfinite(b) else (lo + 10.0)
    if hi == lo:
        return [lo]
    return [lo + (hi - lo) * t for t in
            [(i + 0.5) / n for i in range(n)]]


def _antiderivative_axis(x, axis):
    """Antiderivative with respect to the variable living on `axis`.

    antiderivative() shifts the POWER axis, which is right when the integration
    variable is seeded there.  It is not right once the seed has its own lane:
    the power axis then carries the INTEGRAND's own infinitesimal content --
    R1 zeros, anything the caller's expression genuinely made small -- and
    shifting that is integrating something that is not the variable.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in x.c.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        if v[axis] > 0:
            continue                       # positive: an INF component, skip
        v[axis] -= 1
        divisor = abs(v[axis])
        out[canon(tuple(v))] = coeff / divisor
    return out


def _eval_axis(terms, h_value, axis):
    """Substitute a real h for the axis-`axis` infinitesimal, keep every other.

    This is the only place a real number may replace an infinitesimal, and it
    may do so ONLY for the lane the integrator seeded.  Doing it for every
    dimension at once is what turned a structural zero into a real 4.967e-09:
    x*0 leaves h*z and z*z both at dim -2 when the seed shares the power axis,
    and no substitution can separate them afterwards.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    out = {}
    for dim, coeff in terms.items():
        v = list(as_vec(dim))
        while len(v) <= axis:
            v.append(0)
        k = v[axis]
        if k >= 0:
            continue
        v[axis] = 0
        out_key = canon(tuple(v))
        out[out_key] = out.get(out_key, 0.0) + coeff * h_value ** (-k)
    return out


@_contextlib.contextmanager
def _box_scope(nvars):
    """Raise the dimension cap for one box integral.

    The perturbation seeds put a series on each extra axis, so the integrand's
    expansion is a PRODUCT across nvars axes and easily passes MAX_ACTIVE_DIMS
    = 60: 1/(1+x+y+z) holds 680 terms at a single panel.  _truncate_dims then
    keeps the 60 nearest zero, which stops the panel refinement converging --
    so the loop doubles all the way to the cap and still lands on a worse
    answer than the Riemann sum it replaced.  Measured on that integrand:
    cap 60 -> err 9.5e-06 in 34.2s;  cap lifted -> err 7.9e-09 in 0.7s.
    Slower AND wronger, from a guard written for a different purpose.
    """
    global MAX_ACTIVE_DIMS
    old = MAX_ACTIVE_DIMS
    MAX_ACTIVE_DIMS = max(MAX_ACTIVE_DIMS, 400 * max(1, nvars - 1))
    try:
        yield
    finally:
        MAX_ACTIVE_DIMS = old


def _surface_exact(f, uv, surface, is_vector, tol=1e-10):
    """Surface integral with u and v on their own lanes.

    Builds the integrand as a COMPOSITE in (u, v) -- position, both tangents,
    the cross product and its norm -- and hands it to the box integrator.  The
    sampling path froze the tangents at each sample point with float(), so the
    area element was piecewise constant; here it varies across the patch
    because d(sigma)/du is still a series.
    """
    (a_u, b_u), (a_v, b_v) = uv

    def integrand(u_c, v_c):
        S = surface(u_c, v_c)
        if not isinstance(S, (list, tuple)) or len(S) < 3:
            raise TypeError("surface must return three components")
        S = [_ensure_composite(c) for c in S]
        Su = [_derivative_axis(c, 1) for c in S]
        Sv = [_derivative_axis(c, 2) for c in S]
        nx = Su[1] * Sv[2] - Su[2] * Sv[1]
        ny = Su[2] * Sv[0] - Su[0] * Sv[2]
        nz = Su[0] * Sv[1] - Su[1] * Sv[0]
        if is_vector:
            F = [_ensure_composite(comp(*S)) for comp in f]
            return F[0] * nx + F[1] * ny + F[2] * nz
        val = _ensure_composite(f(*S))
        return val * sqrt(nx * nx + ny * ny + nz * nz)

    # DOES THE SURFACE PROPAGATE COMPOSITE STRUCTURE?
    #
    # A surface written with math.cos rather than the composite cos does NOT
    # raise: Composite defines __float__, so math.cos silently takes the
    # standard part and hands back a plain float.  Every component then has no
    # lane content, both tangents come out empty, the cross product is zero and
    # the integral is 0.0 -- with no exception anywhere to trigger a fallback.
    # That is how five passing tests turned into exact zeros.  Check for it.
    try:
        u_p = 0.5 * (a_u + b_u) + _perturbation_seed(1)
        v_p = 0.5 * (a_v + b_v) + _perturbation_seed(2)
        S_p = surface(u_p, v_p)
        if not isinstance(S_p, (list, tuple)) or len(S_p) < 3:
            return None
        # A CONSTANT component is legitimate -- [u, v, 0] is a flat patch, and
        # requiring every component to be a Composite sent it to the fallback
        # (5.0e-10 in 182ms instead of 0.0e+00 in 5ms).  What matters is not
        # that each part is composite but that the patch MOVES on both
        # parameters, which is what the two checks below test.
        S_p = [_ensure_composite(c) for c in S_p]
        moves_u = any(_derivative_axis(c, 1).c for c in S_p)
        moves_v = any(_derivative_axis(c, 2).c for c in S_p)
        if not (moves_u and moves_v):
            return None
    except Exception:
        return None

    try:
        return _box_exact(integrand, [(a_u, b_u), (a_v, b_v)], tol=tol,
                          probe_floats=False)
    except Exception:
        return None


def _box_exact(f, ranges, tol=1e-10, max_panels=4096, probe_floats=True):
    """Exact box integral.  The INTEGRAND keeps the power axis; every
    integration variable gets its own lane.

    Variable i is seeded on basis axis i+1.  Nothing the caller's expression
    produces on the power axis is ever touched: it is not the variable, so it
    is not integrated, and it is never handed to a real panel width.

    Returns None when the exact path does not apply, so the caller falls back
    rather than returning something wrong.
    """
    from composite.backends.vector_dim_backend import canon, as_vec
    nvars = len(ranges)
    if nvars < 2:
        return None

    # The integrand must not bring content on the lanes the seeds will use.
    probe_pt = [0.5 * (lo + hi) for lo, hi in ranges]
    if not probe_floats:
        probe = None            # caller builds its integrand from composites
    else:
      try:
        probe = f(*probe_pt)
      except Exception:
        return None
    if isinstance(probe, Composite):
        if any(isinstance(d, tuple) and any(e != 0 for e in d[1:nvars + 1])
               for d in probe.c):
            return None

    centres = [0.5 * (lo + hi) for lo, hi in ranges]
    a, b = ranges[0]
    prev = None
    last_delta = None
    stalled = 0
    panels = 8
    with _box_scope(nvars):
        # seeds[i] rides on lane i+1; lane 0 (power) stays the integrand's
        seeds_tail = [centres[i] + _perturbation_seed(i + 1)
                      for i in range(1, nvars)]
        while panels <= max_panels:
            acc = {}
            dx = (b - a) / panels
            try:
                for i in range(panels):
                    x_seed = (a + i * dx + dx / 2) + _perturbation_seed(1)
                    fx = _ensure_composite(f(x_seed, *seeds_tail))
                    Fx = _antiderivative_axis(fx, 1)
                    for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
                        for k, v in _eval_axis(Fx, hv, 1).items():
                            acc[k] = acc.get(k, 0.0) + sign * v
            except Exception:
                return None

            # integrate the remaining lanes term by term, exactly
            total = 0.0
            for key, coeff in acc.items():
                v = list(as_vec(key))
                while len(v) <= nvars:
                    v.append(0)
                if v[0] != 0:
                    continue          # the integrand's OWN infinitesimal part:
                                      # metadata, never fused into the float
                factor = coeff
                for var_i in range(1, nvars):
                    order = -int(v[var_i + 1])
                    lo, hi = ranges[var_i]
                    c = centres[var_i]
                    factor *= ((hi - c) ** (order + 1)
                               - (lo - c) ** (order + 1)) / (order + 1)
                total += factor

            if not math.isfinite(total):
                return None
            if prev is not None:
                delta = abs(total - prev)
                if delta <= tol * max(1.0, abs(total)):
                    return total
                if last_delta is not None and delta > 0.5 * last_delta:
                    stalled += 1
                    if stalled >= 2:
                        return total
                else:
                    stalled = 0
                last_delta = delta
            prev = total
            panels *= 2
        return prev


def integrate(f, *args, curve=None, surface=None, tol=1e-10, terms=15):
    """One integral to rule them all."""

    def _st(v):
        return v.st() if isinstance(v, Composite) else float(v)

    # --- LINE INTEGRAL ---
    # The integrand along the curve is a 1D integral in t, read by meeting
    # composites.  The tangent is Composite.D() -- the derivative as a grade
    # shift -- divided by t's own D(): the right end is seeded inward, t = b - h,
    # so d/dh there is -d/dt.  The speed is a composite sqrt.  A component whose
    # D() is all zeros does not move and is skipped BEFORE any arithmetic: a
    # wholly zero operand would be converted by R1 into an infinitesimal the
    # curve never had.
    #
    # The curve must be composite.  math.cos(t) turns the seed into a float and
    # loses the tangent, so it is refused, not differenced.
    if curve is not None:
        t_range = args[0] if args else (0, 1)
        a_t, b_t = t_range
        is_vector = isinstance(f, list)

        def _line_integrand(t_comp):
            pos = [p if isinstance(p, Composite) else Composite({0: float(p)})
                   for p in curve(t_comp)]
            dt = t_comp.D()
            tangent = []
            for p in pos:
                d = p.D()
                tangent.append(d / dt if any(v != 0.0 for v in d.c.values())
                               else None)
            acc = None
            if is_vector:
                for comp, tv in zip(f, tangent):
                    if tv is None:
                        continue
                    term = _ensure_composite(comp(*pos)) * tv
                    acc = term if acc is None else acc + term
                return acc if acc is not None else Composite({0: 0.0})
            for tv in tangent:
                if tv is None:
                    continue
                acc = tv * tv if acc is None else acc + tv * tv
            if acc is None:
                return Composite({0: 0.0})
            return _ensure_composite(f(*pos)) * sqrt(acc)

        try:
            with _refusing_float():
                return integrate_jets(_line_integrand, a_t, b_t, tol=tol).to_ieee754()
        except FloatCoercionError as e:
            raise ValueError(
                "integrate: the curve or the integrand is not composite -- %s"
                % e) from None

    # --- SURFACE INTEGRAL ---
    # Read by meeting composites over the (u, v) rectangle: each node is
    # order+2 merged composites of the surface, separated from parts into its
    # 2D jet; see integrate_surface.  A float surface (math.sin) is refused.
    if surface is not None:
        uv = args[0] if args else ((0, 1), (0, 1))
        return integrate_surface(f, uv, surface, tol=tol)

    # --- 1D DEFINITE / IMPROPER ---
    if len(args) == 2 and isinstance(args[0], (int, float)):
        a_val, b_val = args
        a_inf = math.isinf(a_val) and a_val < 0
        b_inf = math.isinf(b_val) and b_val > 0
        if a_inf or b_inf:
            # Composite first: the node at infinity is x = 1/h, an algebraic
            # tail an ordinary composite there and an exponential one a
            # transseries sector.  Where the library says the tail is not
            # representable -- a Gaussian (exp(-1/h**2), level two), a rate
            # that is not an integer sector, an oscillation (sin(1/h) is a
            # range) -- it falls back to the panel path.  The
            # trigger is the library's own refusal, not a probe.
            try:
                return _improper_jets(f, a_val, b_val, tol)
            except (NotRepresentableError, NotImplementedError):
                pass
            if a_inf and b_inf:
                val, _ = _improper_integral_panels_both(f, tol=tol)
                return val.st()
            if b_inf:
                val, _ = _improper_integral_panels(f, a_val, tol=tol)
                return val.st()
            val, _ = _improper_integral_panels(lambda x: f(-x), -b_val, tol=tol)
            return val.st()
        # Integrated by meeting composites (integrate_jets).  The standard
        # part, or +-inf for a divergent integral -- the IEEE754 projection,
        # where .st() of the old path returned nan.
        return integrate_jets(f, a_val, b_val, tol=tol).to_ieee754()

    # --- 2D BOX ---
    # Read by meeting composites: each node is order+1 merged composites of
    # f(x + h, y + c h), separated from parts into the 2D jet; see _meet2d.
    if len(args) == 2 and isinstance(args[0], tuple):
        return integrate_box2d(f, args[0], args[1], tol=tol)

    # --- 3D BOX ---
    if len(args) == 3 and isinstance(args[0], tuple):
        exact = _box_exact(f, list(args), tol=tol)
        if exact is not None:
            return exact
        (a_x, b_x), (a_y, b_y), (a_z, b_z) = args
        N = 50
        dx = (b_x - a_x) / N
        dy = (b_y - a_y) / N
        dz = (b_z - a_z) / N
        total = 0.0
        for i in range(N):
            x = a_x + (i + 0.5) * dx
            for j in range(N):
                y = a_y + (j + 0.5) * dy
                for k in range(N):
                    z = a_z + (k + 0.5) * dz
                    total += _st(f(x, y, z)) * dx * dy * dz
        return total

    raise ValueError(f"Could not determine integral type from arguments: {args}")

def integrate_stepped(f: Callable, a: float, b: float, step: float = 0.5, terms: int = 15):
    """Integral over [a, b] with nodes at every `step`, each step read by
    integrate_jets.  Returns (Composite, error).

    Each step used to be one jet evaluated against real powers of its width --
    h given the value dx -- with the deepest term as the error.  Now each step
    is read by meeting composites, and integrate_jets adds nodes inside a step
    where its two ends disagree.  There is no error estimate any more, so the
    error slot is nan rather than a number that would look like a bound.
    `terms` is kept for the signature and no longer used.

    The steps are joined as terms of ONE number, not as a sum of operands:
    adding step results to a running total could cancel to a written zero,
    which R1 would convert.
    """
    from composite.backends.vector_dim_backend import as_vec, canon
    a, b = float(a), float(b)
    edges = [a]
    while edges[-1] < b:
        edges.append(min(edges[-1] + step, b))
    terms_ = {}
    for x0, x1 in zip(edges, edges[1:]):
        for d, v in integrate_jets(f, x0, x1).coeffs_dict().items():
            key = canon(as_vec(d)) if isinstance(d, tuple) else d
            terms_[key] = terms_.get(key, 0.0) + v
    if not terms_:
        return Composite({0: 0.0}), float("nan")
    if any(isinstance(d, tuple) for d in terms_):
        terms_ = {(d if isinstance(d, tuple) else canon((d, 0))): v
                  for d, v in terms_.items()}
        return _vec_composite(terms_), float("nan")
    return Composite(terms_), float("nan")


# =============================================================================
# FIXED: integrate_adaptive — recursive bisection with Taylor error
# =============================================================================


def _integrate_by_sampling(f, a, b, tol=1e-10, max_depth=40):
    """Adaptive Simpson on real evaluations: the fallback for branchy integrands.

    Used when the integrand does not carry composite structure through -- a step,
    a piecewise definition, a lookup -- so there is no expansion to read an error
    off.  Simpson on [a, b] against Simpson on its two halves is the classical
    estimate, and bisecting where they disagree corners a discontinuity instead
    of straddling it.

    Returns (Composite, float) to match `integrate_adaptive`.
    """
    def val(x):
        y = f(R(x))
        return float(y.st()) if isinstance(y, Composite) else float(y)

    def simpson(x0, x2, f0, f1, f2):
        return (x2 - x0) / 6.0 * (f0 + 4.0 * f1 + f2)

    def rec(x0, x2, f0, f1, f2, whole, depth):
        xm1, xm2 = (x0 + (x0 + x2) / 2) / 2, ((x0 + x2) / 2 + x2) / 2
        fm1, fm2 = val(xm1), val(xm2)
        left = simpson(x0, (x0 + x2) / 2, f0, fm1, f1)
        right = simpson((x0 + x2) / 2, x2, f1, fm2, f2)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 15.0 * tol * max(1.0, abs(left + right)):
            return left + right + delta / 15.0, abs(delta) / 15.0
        lv, le = rec(x0, (x0 + x2) / 2, f0, fm1, f1, left, depth + 1)
        rv, re = rec((x0 + x2) / 2, x2, f1, fm2, f2, right, depth + 1)
        return lv + rv, le + re

    f0, f1, f2 = val(a), val((a + b) / 2.0), val(b)
    whole = simpson(a, b, f0, f1, f2)
    value, err = rec(a, b, f0, f1, f2, whole, 0)
    return Composite({0: value}), err


def integrate_adaptive(f, a, b, tol=1e-10, terms=15, max_depth=20, min_panels=4,
                       lane=None):
    """Adaptive integration via composite Taylor convergence.

    Primary path (ONE evaluation per panel):
      1. Evaluate f at panel midpoint via _seeded(mid)
      2. Integrate via antiderivative evaluated at +/- dx/2
      3. Estimate error from Taylor tail
      4. If converged -> accept. If not -> bisect recursively.

    Singularity handling:
      If Taylor tail overflows, tries power-law detection at boundaries.
      If detected, integrates analytically. Otherwise 3-eval fallback.

    Lift-at-the-gate:
      If f doesn't propagate composite structure, builds 4th-order
      approximation via 5-point stencil. Emits warning.

    Returns (Composite, float) — integral value, error estimate.
    """
    _fallback_count = [0]

    # --- LIFT AT THE GATE ---
    # THE VARIABLE ONLY NEEDS A LANE IF THE POWER AXIS IS TAKEN.  See
    # _integrand_needs_lane: when it is free the variable rides it and every
    # panel stays a scalar-dimension composite on the fast backend.
    # `lane` PINS the axis for a caller whose integrand is coupled to one.
    # The curve and surface paths are: _line_integrand reads the tangent with
    # _lane_d1, which looks at lane 1, so seeding anywhere else makes every
    # tangent zero and every line integral come back exactly 0.0.  A probe
    # cannot see that -- an integrand that READS a lane looks identical to one
    # that ignores it.
    _lane = lane if lane is not None else (
        1 if _integrand_needs_lane(f, _lane_probe_points(a, b)) else 0)

    probe = _ensure_composite(f(_seeded((a + b) / 2)))
    # A PLAIN REAL, and nothing else.  Carrying any other grade means the
    # integrand does propagate structure -- the seeded parameter of a Borel
    # transform, a line integral's tangent, an integrand with its own
    # infinitesimal content -- and sampling would silently throw those grades
    # away, which is a worse failure than the one being fixed.
    _plain = [d for d, v in probe.coeffs_dict().items() if v != 0.0]
    if (_lane == 0                     # an integrand that READS a lane is not sampleable:
                                       # a line integral takes its tangent from lane 1, and
                                       # evaluating it at a plain real makes every tangent
                                       # zero -- arc length came back 0.000 instead of 5
            and not any(_dim_negative(dim) for dim in probe.c)
            and all(d == 0 or d == (0,) or (isinstance(d, tuple) and not any(d))
                    for d in _plain)):
        # NO STRUCTURE CAME BACK.  The integrand answered a seeded input with a
        # plain number, so there is no Taylor tail to read and no way to size a
        # panel from one evaluation.  This is the shape of every integrand
        # written with a branch -- a step, a piecewise rate, a table lookup,
        # anything that decides on `x.st()` -- and the old behaviour was a
        # midpoint rule with NO refinement at all: the unit step at 0.3 over
        # [0, 1] came back 0.75 against 0.7, tightening tol did nothing, and the
        # only sign was a warning most callers never see.
        #
        # There is still a way to integrate it, just not this one: sample it.
        # Adaptive Simpson on real evaluations converges on exactly these
        # functions, bisecting until the discontinuity is cornered in a panel
        # too small to matter.  It cannot produce the higher grades a composite
        # integrand carries, which is why it is the fallback and not the method.
        return _integrate_by_sampling(f, a, b, tol=tol, max_depth=max_depth)

    def _panel_with_error(x, dx):
        """One composite eval at midpoint -> integral value + error estimate.

        The variable of integration rides on LANE 1, not the power axis.  The
        power axis belongs to the integrand: R1 zeros and anything else the
        caller's expression genuinely made infinitesimal live there, and they
        are not the variable, so they are neither integrated nor handed to a
        real panel width.  With the seed on the power axis they were, and
        integral of (x*0 + 1) over [0,1] came back 1.0052083333333333 -- a real
        number manufactured from a structural zero, off by 5e-3.
        """
        mid = x + dx / 2
        seed = _perturbation_seed(_lane) if _lane else ZERO
        fx = _ensure_composite(f(mid + seed))

        # Integral via antiderivative along the variable's own lane
        Fx_terms = _antiderivative_axis(fx, _lane)
        acc = {}
        for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
            for k, v in _eval_axis(Fx_terms, hv, _lane).items():
                acc[k] = acc.get(k, 0.0) + sign * v
        # KEEP EVERY POWER-AXIS GRADE, each at its own grade.  Terms carrying a
        # power-axis component are the integrand's own infinitesimal content --
        # they must never be summed INTO dimension 0, which is what the comment
        # here used to say, but they were then dropped entirely, which is a
        # different thing and loses the answer.
        #
        # It matters exactly where it is hardest to get otherwise.  With a
        # seeded parameter eps = h, the integrand of the Euler-Stieltjes
        # integral carries e^-t * (1, -t, t^2, -t^3, ...) across grades, and
        # integrating grade -n gives (-1)^n n! -- the asymptotic series, term
        # by term, without ever summing it (it diverges factorially for every
        # nonzero eps).  Dropping the grades returned 1.0: the correct LIMIT,
        # and a much smaller answer than the one asked for, with nothing to
        # say it had been narrowed.
        #
        # Grade 0 alone is what it always was, so an ordinary integral is
        # unchanged and still reads as a float through st().
        vals = {}
        for k, v in acc.items():
            kk = k if isinstance(k, tuple) else (k,)
            pk = kk[0] if kk else 0
            if len(kk) > 1 and any(c != 0 for c in kk[1:]):
                continue          # still carries the variable's lane: not integrated
            vals[pk] = vals.get(pk, 0.0) + v
        value = Composite(vals) if vals else Composite({0: 0.0})

        # The remainder is not something to model -- the top two orders of the
        # expansion ARE it.  Reading it off the last few terms instead assumed
        # the coefficients decay monotonically, and they do not: exp(-t^2) about
        # a panel midpoint climbs two orders of magnitude before it falls, so the
        # trailing terms sit far below the hump that dominates the remainder.
        # The estimate came back ~1e-8 of the true error, panels were accepted
        # unrefined, and integral_0^8 exp(-t^2) dt was wrong by 1.7e-05 while
        # claiming 5.9e-11 -- returning erf > 1.  Every order is now complete
        # (see _complete_order), so the top two ARE the leading remainder and
        # there is nothing left to estimate.
        half_dx = abs(dx) / 2
        # Orders along the VARIABLE's lane.  _dim_order reads the power axis,
        # which now holds the integrand's own content -- reading it here made
        # every panel look like a constant.
        def _lane1(d):
            """Order along the VARIABLE's axis -- whichever one it is.

            This read index 1 unconditionally.  With the variable on the power
            axis every dim is a scalar, as_vec gives (d, 0), and the order came
            back 0 for every term -- so `max(orders) <= 2` fired on every panel
            and the estimate was 0.0.  The adaptive path then refined nothing:
            integral of exp(-x^2) over [0,20] returned 0.59 against 0.886, and
            sqrt(x) lost four digits.  Both looked like accuracy problems; they
            were the error estimate reading an axis the variable was not on.
            """
            from composite.backends.vector_dim_backend import as_vec
            v = as_vec(d) if isinstance(d, tuple) else (d,)
            return -(v[_lane] if len(v) > _lane else 0)

        orders = [_lane1(d) for d in fx.c if _lane1(d) >= 0]
        if not orders or max(orders) <= 2:
            # Nothing above a quadratic -- the antiderivative is EXACT.
            return value, 0.0

        cut = max(max(orders) - 2, 2)
        err_est = 0.0
        for dim, coeff in fx.c.items():
            k = _lane1(dim)
            if k <= cut:
                continue
            # |h**(k+1) - (-h)**(k+1)| <= 2*h**(k+1), so the orders that cancel
            # on a symmetric panel cost at most a factor 2.  Erring high here
            # spends evaluations; erring low returns wrong answers.
            term = 2.0 * abs(coeff) / (k + 1) * half_dx ** (k + 1)
            if not math.isfinite(term):
                return value, -1.0  # signal: needs fallback
            err_est += term

        if not math.isfinite(err_est):
            return value, -1.0  # signal: needs fallback
        return value, err_est

    def _scale(v):
        """Magnitude for the relative error test, ACROSS EVERY GRADE.

        abs(Composite) is the standard part, i.e. dimension 0 alone.  With a
        seeded parameter the integrand carries real content at other grades,
        and sizing panels by dimension 0 accepted them while those grades were
        still garbage -- the Euler-Stieltjes coefficient at grade -1 came back
        -0.0 where uniform panels gave -1 exactly.  The error estimate already
        sums over every dimension; only the scale it was compared against was
        one-dimensional.
        """
        if isinstance(v, Composite):
            cs = [abs(c) for c in v.coeffs_dict().values() if c != 0.0]
            return max(cs) if cs else 0.0
        return abs(v)

    def _panel_classic(x, dx):
        """Classic single-panel integral for fallback comparison."""
        mid = x + dx / 2
        seed = _perturbation_seed(_lane) if _lane else ZERO
        fx = _ensure_composite(f(mid + seed))
        Fx_terms = _antiderivative_axis(fx, _lane)
        acc = {}
        for sign, hv in ((1.0, dx / 2), (-1.0, -dx / 2)):
            for k, v in _eval_axis(Fx_terms, hv, _lane).items():
                acc[k] = acc.get(k, 0.0) + sign * v
        # Same rule as _panel_with_error: every power-axis grade is kept at
        # its own grade.  This path dropped them, so any panel that fell back
        # here silently contributed only its dimension-0 part -- which is why
        # the Euler-Stieltjes grade -1 coefficient came back 5e-15 instead of
        # -1 while uniform panels gave -1 exactly.
        out = {}
        for k, v in acc.items():
            kk = k if isinstance(k, tuple) else (k,)
            if len(kk) > 1 and any(c != 0 for c in kk[1:]):
                continue
            pk = kk[0] if kk else 0
            out[pk] = out.get(pk, 0.0) + v
        return Composite(out) if out else Composite({0: 0.0})

    def _adaptive(a, b, depth):
        dx = b - a
        mid = (a + b) / 2

        value, err_est = _panel_with_error(a, dx)

        if err_est >= 0:
            # Primary path: Taylor convergence check
            if err_est < tol * (_scale(value) + 1e-100) or depth >= max_depth:
                return value, err_est
            if _scale(value) < tol * 0.01 and err_est < tol:
                return value, err_est
        else:
            # Taylor tail overflowed — try singularity detection
            handled_left, val_left, _, _ = _detect_singularity(f, a, b, dx)
            if handled_left:
                return val_left, abs(val_left) * 1e-8

            handled_right, val_right, _, _ = _detect_singularity(f, b, a, dx)
            if handled_right:
                return val_right, abs(val_right) * 1e-8

            # Classical 3-eval fallback
            _fallback_count[0] += 1
            left_p = _panel_classic(a, dx / 2)
            right_p = _panel_classic(mid, dx / 2)
            half = left_p + right_p
            error = _scale(half - value)
            if error < tol * (_scale(half) + 1e-100) or depth >= max_depth:
                return half, error
            if _scale(half) < tol * 0.01:
                return half, error

        # Bisect
        left_val, left_err = _adaptive(a, mid, depth + 1)
        right_val, right_err = _adaptive(mid, b, depth + 1)
        return left_val + right_val, left_err + right_err

    # Pre-subdivide into min_panels equal panels.
    #
    # NOTHING, not zero (R6).  `total_val = 0.0` looks harmless and is not: a
    # bare Python zero meeting a Composite converts to the composite zero
    # |1|_-1 by R1, so the accumulator injected +1 at grade -1 on its first
    # addition.  Dimension 0 stayed right, which is why it went unseen -- but
    # a panel value of -1 at grade -1 came out 0, and the Euler-Stieltjes
    # coefficient of eps^1 read 1.8e-19 instead of -1 while every neighbouring
    # grade was exact.  It was bit-identical at every refinement depth because
    # the injection is one constant, not an error that refines away.
    #
    # The library's own R1 warning names this case exactly.  It was firing.
    total_val = Composite({})
    total_err = 0.0
    panel_dx = (b - a) / min_panels
    for i in range(min_panels):
        x0 = a + i * panel_dx
        x1 = x0 + panel_dx
        val, err = _adaptive(x0, x1, 0)
        total_val += val
        total_err += err

    if _fallback_count[0] > 0:
        import warnings
        warnings.warn(
            f"integrate_adaptive: Taylor convergence inconclusive on "
            f"{_fallback_count[0]} panel(s), used classical 3-eval fallback.",
            stacklevel=2)

    # total_val already IS a composite when the integrand carried grades of its
    # own -- re-wrapping it as Composite({0: total_val}) collapsed every one of
    # them into dimension 0 after the panels had computed them correctly.
    if isinstance(total_val, Composite):
        return total_val, total_err
    return Composite({0: total_val}), total_err


# =============================================================================
# INTEGRATION BY MEETING COMPOSITES
# =============================================================================
#
# A composite at a node x is the integrand's exact local behaviour, and
# antiderivative() of it is F - F(x) around x, exact grade by grade.  What it
# cannot hold is the constant F(x) -- and the difference of two constants is
# the integral.  Two nodes give it by SUBTRACTION:
#
#   carry both antiderivatives to the same composite point m + h, by putting
#   the composite (m - x) + h into each one's series:
#
#       G0((m - x0) + h) = F(m + h) - F(x0)
#       G1((m - x1) + h) = F(m + h) - F(x1)
#
#   subtract: grade 0 is F(x1) - F(x0), and every other grade cancels.
#
# h is never given a real value.  What goes into a series is a composite whose
# standard part is the real distance between two points and whose
# infinitesimal part is h; that is composition, exact grade by grade.
#
# The subtraction checks itself.  Both sides are the same function's jet at the
# same point, so the integrand and its slope at m -- grades -1 and -2 -- must
# agree from both sides.  Value alone is fooled by symmetry: an even integrand
# met at its centre gives the same wrong value from both mirror images, while
# the slope flips sign.  Where they disagree, one side's series did not reach
# m, and the panel's midpoint becomes a node: one more evaluation.
#
# A node's jet is made as deep as `order` by the scope derivative extraction
# uses, so every transcendental in the integrand builds its series that far.
# Depth is cheap; an evaluation is what is being saved.

def _integer_jet(c):
    """An ordinary Taylor jet: integer grades at or below zero, power axis only."""
    for d, v in c.c.items():
        if v == 0.0:
            continue
        if isinstance(d, tuple):
            return False
        d = float(d)
        if d > 0 or not d.is_integer():
            return False
    return True


def _compose(G, u):
    """G's series in its variable, with the composite u put in for it.

    A term c * u**p * ln(1/u)**l becomes c * u**p * (-ln(u))**l in composite
    arithmetic.  Non-negative integer orders go through Horner; poles,
    fractional orders and log factors term by term with composite power and
    ln.  u's standard part is a real distance, never zero.
    """
    ints, rest = {}, []
    for d, v in G.c.items():
        if v == 0.0:
            continue
        if isinstance(d, tuple):
            if any(float(e) != 0 for e in d[2:]):
                raise ValueError("integrate_jets: no composition for a deeper log axis")
            p, l = -float(d[0]), (int(float(d[1])) if len(d) > 1 else 0)
        else:
            p, l = -float(d), 0
        if l == 0 and p.is_integer() and p >= 0:
            ints[int(p)] = ints.get(int(p), 0.0) + v
        else:
            rest.append((p, l, v))
    out = None
    if ints:
        N = max(ints)
        out = Composite({0: ints.get(N, 0.0)})
        for k in range(N - 1, -1, -1):
            out = out * u
            if ints.get(k, 0.0) != 0.0:
                out = out + Composite({0: ints[k]})
    for p, l, v in rest:
        # u**0 is not computed: through exp(0 * ln u) the 0 is a WRITTEN zero,
        # R1 makes it an infinitesimal, and ln(1/u) came back as ln(1/u) + h.
        # Integer powers take the integer path, which never meets exp.
        if p == 0.0:
            t = Composite({0: v})
        elif p.is_integer():
            t = (u ** int(p)) * v
        else:
            t = (u ** p) * v
        if l:
            t = t * (-ln(u)) ** l
        out = t if out is None else out + t
    return out if out is not None else Composite({})


def _sector_antiderivative(C, n, order):
    """P with d/du[exp(-n/u) P(u)] = exp(-n/u) C(u).

    Differentiating gives exp(-n/u) (n P/u**2 + P'), so P = (u**2/n)(C - P').
    Each pass of that recursion fixes one more order of P -- u**2 raises the
    grade by two and d/du lowers it by one -- so `order + 2` passes settle
    every grade kept.  The series is often asymptotic rather than convergent
    (exp(-1/u)/u gives 1, -1!, 2!, ...); where it does not reach the meeting
    point, the two sides disagree there and the panel is split.

    C - P' is formed coefficient by coefficient, as terms of one series:
    through Composite.__sub__ an exact cancellation would deposit a residue,
    which is a cancellation event this series does not have.
    """
    shift = Composite({-2: 1.0 / n})                  # u**2 / n
    keep = lambda d: -float(d[0] if isinstance(d, tuple) else d) <= order
    cd = {d: v for d, v in C.c.items() if v != 0.0}
    P = Composite({})
    for _ in range(order + 2):
        q = dict(cd)
        if P.c:
            for d, v in P.D().c.items():
                if v != 0.0:
                    q[d] = q.get(d, 0.0) - v
        q = {d: v for d, v in q.items() if v != 0.0}
        if not q:
            nxt = Composite({})
        else:
            nxt = shift * Composite(q)
            nxt = Composite({d: v for d, v in nxt.c.items() if v != 0.0 and keep(d)})
        if nxt.coeffs_dict() == P.coeffs_dict():
            break
        P = nxt
    return P


def _ts_antiderivative(T, order):
    """antiderivative() of a Transseries, sector by sector."""
    from composite.transseries import Transseries
    out = {}
    for n, C in T.sectors.items():
        out[n] = antiderivative(C) if n == 0 else _sector_antiderivative(C, n, order)
    return Transseries(out)


def _compose_ts(G, u):
    """A transseries antiderivative carried to the finite composite point u:
    sector n contributes compose(P_n, u) * exp(-n/u), and with u finite that
    exp is an ordinary composite -- the flat term is flat only AT u = h."""
    out = None
    for n, P in G.sectors.items():
        if not P.c:
            continue
        if n != 0 and math.exp(-float(n) / u.st()) == 0.0:
            # exp(-n/u) underflows float64 to exactly 0.0 here (u = 0.001:
            # e**-1000).  Built anyway, that 0.0 is a WRITTEN zero, R1 converts
            # it, and the manufactured infinitesimal surfaced as the integrand's
            # value -- 0.027 where it is e**-1000 -- so the two sides of the meet
            # never agreed.  Below float64 the sector contributes nothing a
            # float can hold, so it is left out: absent, not zero.
            continue
        t = _compose(P, u)
        if n != 0:
            t = t * exp(R(-float(n)) / u)
        out = t if out is None else out + t
    return out if out is not None else Composite({})


def integrate_jets(f, a, b, tol=1e-10, order=50, max_nodes=200, max_depth=60,
                   scale=1.0):
    """int_a^b f(x) dx by meeting composites.  Returns a Composite whose grade 0
    is the integral.

    Starts with the two ends.  Each panel's two node antiderivatives are carried
    to the composite midpoint and subtracted; where value or slope disagree
    there, the midpoint becomes a node.  Each node is one evaluation, its jet
    built to `order`.  The right end is seeded toward the interior, f(b - h).

    A node whose jet is not an ordinary Taylor jet -- a pole, a fractional
    grade, a log, which is what a singular endpoint looks like -- is handled by
    the same construction: antiderivative() integrates those grades exactly and
    composition carries them to m.  Such a node keeps what its limit leaves, so
    int_0^1 dx/x**2 is <|1|_1 |-1|_0>, 1/h - 1, and sqrt keeps its tail.

    Raises ValueError when the integrand returns no composite, and when
    max_nodes evaluations have not resolved [a, b] -- an integrand carrying an
    infinitesimal of its own never agrees, and is refused rather than summed.
    """
    a, b = float(a), float(b)
    if b < a:
        return -integrate_jets(f, b, a, tol, order, max_nodes, max_depth, scale)
    if b == a:
        return Composite({0: 0.0})
    nodes = {}

    def G(x, s):
        """antiderivative(f(x + s h)) = s * (F(x + s u) - F(x)), a series in u."""
        if (x, s) not in nodes:
            if len(nodes) >= max_nodes:
                raise ValueError(
                    "integrate_jets: %d nodes evaluated and [%r, %r] still "
                    "unresolved.  The usual cause is an integrand carrying an "
                    "infinitesimal of its own -- a parameter built with ZERO "
                    "or R(0), a written 0 such as 0*x -- whose composite "
                    "denotes a different function at every node, so the two "
                    "sides never agree.  That case is not supported yet."
                    % (len(nodes), a, b))
            # `scale` is the seed's quantity: the jet's grade -k then carries
            # scale**k, keeping coefficients of order one on a short interval
            # beside a singularity (the seed carries the chain rule).
            seed = (_mint(Composite({-1: s * scale})) if x == 0 else
                    _mint(Composite({0: x, -1: s * scale})))
            with _derivative_scope(order, order), _exp_to_transseries():
                c = f(seed)
            from composite.transseries import Transseries
            if isinstance(c, Transseries):
                if set(c.sectors) <= {0}:
                    c = c.sectors.get(0, Composite({}))
                else:
                    # A flat term at this node -- exp(-1/h) where x = 1/h --
                    # integrated sector by sector.
                    nodes[(x, s)] = (_ts_antiderivative(c, order) * scale, False)
                    return nodes[(x, s)]
            if not isinstance(c, Composite):
                raise ValueError(
                    "integrate_jets: the integrand returned %r, not a composite"
                    % type(c).__name__)
            G_ = antiderivative(c)
            nodes[(x, s)] = (G_ * scale if scale != 1.0 else G_, _integer_jet(c))
        return nodes[(x, s)]

    def right(x):
        """The node at the RIGHT of a panel, read toward the panel.  At b, and
        at a singular interior node, that is f(x - h); a regular interior node
        reuses its one forward jet."""
        if x == b:
            return G(x, -1.0), -1.0
        g = G(x, 1.0)
        if not g[1]:
            return G(x, -1.0), -1.0             # one more evaluation, singular only
        return g, 1.0

    def panel(x0, x1, depth=0):
        (G0, plain0), ((G1, plain1), s1) = G(x0, 1.0), right(x1)
        m = 0.5 * (x0 + x1)
        carry = lambda Gx, u: (_compose(Gx, u) if isinstance(Gx, Composite)
                               else _compose_ts(Gx, u))
        with _derivative_scope(order, order):
            # distances in the seed's units
            A = carry(G0, R((m - x0) / scale) + ZERO)                 # F(m+h) - F(x0)
            B = carry(G1, R(s1 * (m - x1) / scale) + ZERO * s1)       # s1 (F(m+h) - F(x1))
        fa, fb = A.coeff(-1), s1 * B.coeff(-1)
        da, db = A.coeff(-2), s1 * B.coeff(-2)
        value = A.coeff(0) - s1 * B.coeff(0)
        size = max(1.0, abs(fa), abs(da))
        if (all(math.isfinite(v) for v in (value, fa, fb, da, db))
                and abs(fa - fb) <= tol * size and abs(da - db) <= tol * size):
            kept = []
            # A singular node keeps what its limit leaves: the panel runs from
            # x0 + h, or to x1 - h -- the written endpoint 0 is h, never 0.001.
            # A divergence stays a graded infinity; two cancelling ones around
            # an interior pole cancel as terms.
            if not plain0:
                kept.append(G0 * -1.0)
            if not plain1:
                kept.append(G1 * -1.0)
            return [(A - B * s1, kept)]         # grade 0 is this panel's integral
        if not x0 < m < x1 or depth >= max_depth:
            # No float strictly between the two nodes, or max_depth halvings
            # toward one point: a split would make no progress, and recursing
            # toward a point the two sides never agree at ran into Python's
            # recursion limit before max_nodes.
            raise ValueError(
                "integrate_jets: [%r, %r] unresolved after %d bisections"
                % (x0, x1, depth))
        return panel(x0, m, depth + 1) + panel(m, x1, depth + 1)

    parts = panel(a, b)
    total = parts[0][0]
    for p, _ in parts[1:]:
        total = total + p
    # ONE number assembled from its terms, not a sum of operands: adding kept
    # terms to Composite({0: value}) would make a written zero of a finite
    # part that is exactly 0, and R1 would convert it.
    from composite.backends.vector_dim_backend import as_vec, canon
    from composite.transseries import Transseries
    terms = {0: total.coeff(_zero_dim_like(total))}
    flat = {}                                   # sector n -> {grade: coeff}
    for _, kept in parts:
        for k in kept:
            pieces = (k.sectors.items() if isinstance(k, Transseries)
                      else [(0, k)])
            for n, c in pieces:
                into = terms if n == 0 else flat.setdefault(n, {})
                for d, v in c.c.items():
                    if n == 0 and ((not any(d)) if isinstance(d, tuple) else d == 0):
                        continue                # G's own constant slot
                    if v == 0.0:
                        continue
                    key = canon(as_vec(d)) if isinstance(d, tuple) else d
                    into[key] = into.get(key, 0.0) + v

    def one(t):
        if any(isinstance(d, tuple) for d in t):
            return _vec_composite({(d if isinstance(d, tuple) else canon((d, 0))): v
                                   for d, v in t.items()})
        return Composite(t)

    if flat:
        # What the node at infinity leaves: exp(-n/h) * P(h), below every
        # power -- kept, as the lower limit's tail is kept elsewhere.
        return Transseries({0: one(terms), **{n: one(t) for n, t in flat.items()}})
    return one(terms)


def integrate_jets_tail(f, a, tol=1e-10, order=50, max_nodes=200):
    """int_a^inf f(x) dx, a > 0, as int_0^(1/a) f(1/u) / u**2 du by meeting
    composites.  The u = 0 node is x = 1/h: an algebraic tail is an ordinary
    composite there, an exponential one exp(-n/h) a transseries sector.

    The seed's quantity is the interval, 1/a, so jet coefficients are in units
    of it.  With an integer a the node at infinity is x = a/h and exp(-x)
    lands on sector a -- an integer, as ts_exp requires; so a should be one.
    """
    if a <= 0:
        raise ValueError("integrate_jets_tail needs a > 0")
    return integrate_jets(lambda u: f(R(1) / u) * (R(1) / (u * u)),
                          0.0, 1.0 / a, tol=tol, order=order, max_nodes=max_nodes,
                          scale=1.0 / a)


def _tail_start(f, a, order=50):
    """Where the tail should start: far enough out that the node at infinity's
    own series reaches across it.

    In u = 1/x the tail [c, inf) is [0, 1/c], and the jet at u = 0 converges
    only out to its nearest singularity -- a pole of the integrand at x = p
    lands at u = 1/p, so a rational with poles at |x| = 36 leaves that jet a
    radius of 1/36.  Started at x = 1, the meeting point sat outside it and the
    two sides never agreed.  The radius is read off the same jet by its
    coefficient growth (composite.singularity.radius -- an estimate from the
    last coefficients, not a bound), and the tail starts at its inverse: the
    u-interval is then the radius and the meeting point sits at half of it,
    the same margin every panel has.  Everything from a to there is an
    ordinary finite piece.
    """
    from composite.singularity import radius
    from composite.transseries import Transseries
    seed = _mint(Composite({-1: 1.0}))
    with _derivative_scope(order, order), _exp_to_transseries():
        g = f(R(1) / seed) * (R(1) / (seed * seed))
    series = (list(g.sectors.values()) if isinstance(g, Transseries) else [g])
    rs = []
    for c in series:
        k = {int(-float(d)): v for d, v in c.c.items()
             if v != 0.0 and not isinstance(d, tuple) and float(d).is_integer()}
        if len(k) < 3:
            continue                        # a finite expansion reaches everywhere
        lo = min(k)
        r = radius([k.get(i, 0.0) for i in range(lo, max(k) + 1)])
        if r and math.isfinite(r) and r > 0:
            rs.append(r)
    start = max(a, 1.0)
    if rs:
        start = max(start, 1.0 / min(rs))
    # An integer, so the tail's node at infinity, x = start/h, puts exp(-x)
    # on an integer sector.
    return float(math.ceil(start))


def _half_line_jets(f, a, tol):
    """int_a^inf f by meeting composites: [a, c] + the tail from c, with c
    where the node at infinity's series reaches (_tail_start)."""
    c = _tail_start(f, a)
    if c <= a:
        return [integrate_jets_tail(f, a, tol=tol)]
    return [integrate_jets(f, a, c, tol=tol), integrate_jets_tail(f, c, tol=tol)]


def _improper_jets(f, a, b, tol):
    """An improper integral by meeting composites, as a float: the standard
    part, or +-inf when it diverges."""
    return _improper_jets_composite(f, a, b, tol).to_ieee754()


def _improper_jets_composite(f, a, b, tol):
    """An improper integral by meeting composites, as ONE composite.

    The pieces are read together as terms of ONE number before projecting, so
    a divergence keeps its sign and two infinities of one sign add.  Only the
    standard sector is read: a kept exp(-n/h) term is below every power and
    contributes nothing to a float.  Raises NotRepresentableError or
    NotImplementedError when the library cannot represent the tail.
    """
    from composite.transseries import Transseries
    neg = lambda x: f(-x)
    if math.isinf(a) and math.isinf(b):
        pieces = _half_line_jets(f, 0.0, tol) + _half_line_jets(neg, 0.0, tol)
    elif math.isinf(b):
        pieces = _half_line_jets(f, a, tol)
    else:
        pieces = _half_line_jets(neg, -b, tol)
    terms = {}
    for p in pieces:
        c = p.sectors.get(0, Composite({})) if isinstance(p, Transseries) else p
        for d, v in c.c.items():
            if v != 0.0:
                terms[d] = terms.get(d, 0.0) + v
    if not terms:
        return Composite({0: 0.0})
    vector = any(isinstance(d, tuple) for d in terms)
    if vector:
        from composite.backends.vector_dim_backend import canon
        terms = {(d if isinstance(d, tuple) else canon((d, 0))): v
                 for d, v in terms.items()}
    return _vec_composite(terms) if vector else Composite(terms)


# =============================================================================
# TWO VARIABLES: MERGED COMPOSITES, SEPARATED FROM PARTS
# =============================================================================
#
# One composite carries one infinitesimal, so at a node it holds ONE number per
# grade -- and grade k of a function of (u, v) has k+1 Taylor terms T_ij,
# i + j = k.  The node therefore evaluates K+1 MERGED composites,
#
#     f(u + h, v + c h)          grade k  =  sum_j  T_{k-j, j} c**j
#
# for K+1 seed quantities c, and reads the full 2D jet off them with fixed
# weights: the inverse Vandermonde of the c's, exact rationals, so that
# sum_m w_m c_m**j picks out one v-order.  The weights are applied to the
# composites' PARTS -- their coefficients -- never as composite sums: a partial
# sum that is wholly zero is a cancellation, R1 deposits a residue one grade
# down (a - a = a*h), and that wrote spurious terms into the jet of u*v.
#
# The seed quantities are Chebyshev points in [-0.95, 0.95] rounded to odd
# multiples of 1/1024: exact in float64, never 0 (a plain coordinate at 0 would
# be a written zero), and never a simple ratio such as -1, which made u + v
# cancel exactly at the origin along that direction and corrupted that node.
#
# A rectangle is integrated by meeting its four corners' double antiderivatives
# at the composite centre, as integrate_jets meets two nodes: each corner
# covers its quadrant, and the four quadrants with signs make the rectangle.
# The corners must agree on the integrand and its diagonal slope at the
# centre; where they do not, the rectangle splits in four.  Nodes are shared.

@_functools.lru_cache(maxsize=None)
def _seed_quantities(n):
    """n seed quantities: Chebyshev points in [-0.95, 0.95], odd multiples of 1/1024."""
    out = []
    for k in range(n):
        x = 0.95 * math.cos(math.pi * (k + 0.5) / n)
        num = round(x * 512) * 2 + (1 if x >= 0 else -1)
        out.append(_Fraction(num, 1024))
    if len(set(out)) != n:
        raise ValueError("_seed_quantities: %d points collide at 1/1024 spacing" % n)
    return tuple(out)


@_functools.lru_cache(maxsize=None)
def _separation_weights(n):
    """W[J][m]: sum_m W[J][m] c_m**j = [j == J], j = 0..n-1, exact rationals."""
    cs = _seed_quantities(n)
    W = []
    for J in range(n):
        A = [[c ** j for c in cs] + [_Fraction(int(j == J))] for j in range(n)]
        for col in range(n):
            piv = next(r for r in range(col, n) if A[r][col] != 0)
            A[col], A[piv] = A[piv], A[col]
            A[col] = [x / A[col][col] for x in A[col]]
            for r in range(n):
                if r != col and A[r][col] != 0:
                    A[r] = [x - A[r][col] * y for x, y in zip(A[r], A[col])]
        W.append(tuple(float(A[m][n]) for m in range(n)))
    return tuple(W)


def _seed_at(x0, q):
    """x0 + q h, without writing a zero at x0 = 0."""
    return _mint(Composite({-1: float(q)}) if x0 == 0 else
                 Composite({0: float(x0), -1: float(q)}))


def _parts(x):
    """A composite's coefficients, or a plain number's, as {grade: value}."""
    if isinstance(x, Composite):
        return x.coeffs_dict()
    return {0: float(x)} if x != 0 else {}


def _separate2d(values, order):
    """T[i][j], i + j <= order, from the parts of composites along the seed quantities."""
    n = len(values)
    W = _separation_weights(n)
    parts = [_parts(v) for v in values]
    T = [[0.0] * (order + 1) for _ in range(order + 1)]
    for J in range(order + 1):
        for i in range(order + 1 - J):
            T[i][J] = sum(w * p.get(-(i + J), 0.0) for w, p in zip(W[J], parts))
    return T


def _along2d(T, c, order):
    """The composite along direction (1, c), built from a 2D jet's parts; None if empty."""
    acc = {}
    for k in range(order + 1):
        v = sum(T[k - j][j] * c ** j for j in range(k + 1))
        if v != 0.0:
            acc[-k] = v
    return Composite(acc) if acc else None


def _parts_sum(terms, signs=None):
    """A sum assembled from parts: one number, no operand can cancel to zero."""
    acc = {}
    for t, s in zip(terms, signs or [1.0] * len(terms)):
        if t is None:
            continue
        for d, v in t.coeffs_dict().items():
            acc[d] = acc.get(d, 0.0) + s * v
    acc = {d: v for d, v in acc.items() if v != 0.0}
    return Composite(acc) if acc else None


def _series2d(T, P, Q, order, anti):
    """sum T_ij P**i Q**j, or the double antiderivative, with composite P, Q;
    assembled from parts, zero coefficients never written."""
    top = order + 1 if anti else order
    Pp, Qp = [None, P], [None, Q]
    for _ in range(top - 1):
        Pp.append(Pp[-1] * P)
        Qp.append(Qp[-1] * Q)
    acc = {}
    for i in range(order + 1):
        for j in range(order + 1 - i):
            c = T[i][j]
            if c == 0.0:
                continue
            if anti:
                t = Pp[i + 1] * Qp[j + 1] * (c / ((i + 1) * (j + 1)))
            elif i == 0 and j == 0:
                acc[0] = acc.get(0, 0.0) + c
                continue
            else:
                t = (Pp[i] if j == 0 else Qp[j] if i == 0 else Pp[i] * Qp[j]) * c
            for d, v in t.coeffs_dict().items():
                acc[d] = acc.get(d, 0.0) + v
    return Composite(acc) if acc else Composite({})


def _meet2d(jet_at, u0, u1, v0, v1, order, tol, max_nodes, max_depth=40):
    """int over [u0,u1] x [v0,v1] from 2D jets at the nodes, by meeting corners.

    jet_at(a, b) returns the integrand's 2D jet T[i][j] at (a, b); it is called
    once per node.  Raises ValueError when max_nodes or max_depth is reached.
    """
    nodes = {}

    def jet(a, b):
        if (a, b) not in nodes:
            if len(nodes) >= max_nodes:
                raise ValueError(
                    "2D integral: %d nodes evaluated and the region is still "
                    "unresolved" % len(nodes))
            nodes[(a, b)] = jet_at(a, b)
        return nodes[(a, b)]

    def panel(a0, a1, b0, b1, depth):
        m, n = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        value, fs = 0.0, []
        with _derivative_scope(order + 2, order + 2):
            for a, b, sgn in ((a0, b0, 1.0), (a1, b0, -1.0), (a0, b1, -1.0), (a1, b1, 1.0)):
                T = jet(a, b)
                P, Q = R(m - a) + ZERO, R(n - b) + ZERO      # real distances as standard parts
                value += sgn * _series2d(T, P, Q, order, True).coeff(0)
                fc = _series2d(T, P, Q, order, False)
                fs.append((fc.coeff(0), fc.coeff(-1)))
        f0 = [x for x, _ in fs]
        f1 = [y for _, y in fs]
        size = max(1.0, max(abs(x) for x in f0 + f1))
        if (all(math.isfinite(x) for x in f0 + f1 + [value])
                and max(f0) - min(f0) <= tol * size and max(f1) - min(f1) <= tol * size):
            return value
        if depth >= max_depth or not (a0 < m < a1 and b0 < n < b1):
            raise ValueError(
                "2D integral: [%r, %r] x [%r, %r] unresolved after %d splits"
                % (a0, a1, b0, b1, depth))
        return (panel(a0, m, b0, n, depth + 1) + panel(m, a1, b0, n, depth + 1)
                + panel(a0, m, n, b1, depth + 1) + panel(m, a1, n, b1, depth + 1))

    return panel(float(u0), float(u1), float(v0), float(v1), 0)


def _surface_jet_at(f, surface, vector, order):
    """A node provider for the surface integrand, f(sigma)|sigma_u x sigma_v| or
    F . (sigma_u x sigma_v).

    sigma is evaluated along order+2 merged directions and separated to order+1;
    sigma_u and sigma_v come from its jet by coefficient shift and are rebuilt
    along each direction from those parts.  The integrand is computed along each
    direction in composite arithmetic, its differences and sums assembled from
    parts, and separated to order.  A field component the caller wrote as the
    number 0 contributes nothing: multiplied by a normal component it would hand
    R1 a written zero the caller never used as an operand.
    """
    n = order + 2
    cs = _seed_quantities(n)

    def jet_at(a, b):
        with _derivative_scope(order + 3, order + 3):
            S = [surface(_seed_at(a, 1.0), _seed_at(b, c)) for c in cs]
        if any(not isinstance(s, (list, tuple)) or len(s) < 3 for s in S):
            raise ValueError("integrate: a surface must return three components")
        Ts = [_separate2d([S[r][comp] for r in range(n)], order + 1) for comp in range(3)]
        Tu = [[[(i + 1) * T[i + 1][j] for j in range(order + 1)] for i in range(order + 1)] for T in Ts]
        Tv = [[[(j + 1) * T[i][j + 1] for j in range(order + 1)] for i in range(order + 1)] for T in Ts]
        g = []
        for m, c in enumerate(cs):
            cf = float(c)
            Su = [_along2d(T, cf, order) for T in Tu]
            Sv = [_along2d(T, cf, order) for T in Tv]
            mul = lambda x, y: None if x is None or y is None else x * y
            nrm = [_parts_sum([mul(Su[1], Sv[2]), mul(Su[2], Sv[1])], [1.0, -1.0]),
                   _parts_sum([mul(Su[2], Sv[0]), mul(Su[0], Sv[2])], [1.0, -1.0]),
                   _parts_sum([mul(Su[0], Sv[1]), mul(Su[1], Sv[0])], [1.0, -1.0])]
            pos = S[m]
            with _derivative_scope(order + 3, order + 3):
                if vector:
                    terms = []
                    for Fi, ni in zip(f, nrm):
                        val = Fi(*pos)
                        if ni is None or (isinstance(val, (int, float)) and val == 0):
                            continue
                        terms.append(_ensure_composite(val) * ni)
                    gm = _parts_sum(terms)
                else:
                    sq = _parts_sum([mul(x, x) for x in nrm])
                    gm = None if sq is None else _ensure_composite(f(*pos)) * sqrt(sq)
            g.append(gm if gm is not None else Composite({}))
        T = _separate2d(g, order + 1)
        return [row[:order + 1] for row in T[:order + 1]]

    return jet_at


def _box_jet_at(f, order):
    """A node provider for f(x, y): order+1 merged composites, separated from parts."""
    cs = _seed_quantities(order + 1)

    def jet_at(a, b):
        with _derivative_scope(order + 2, order + 2):
            vals = [f(_seed_at(a, 1.0), _seed_at(b, c)) for c in cs]
        return _separate2d(vals, order)

    return jet_at


def integrate_box2d(f, xr, yr, tol=1e-10, order=12, max_nodes=400):
    """int over [x0,x1] x [y0,y1] of f(x, y) by meeting composites.  Returns a float.

    The integrand must be composite: a float coercion (math.*) is refused.
    Raises ValueError when the region is not resolved within max_nodes -- an
    integrand carrying an infinitesimal of its own, such as x*0 + 1, denotes a
    different function at every node and the corners never agree.
    """
    jet_at = _box_jet_at(f, order)
    try:
        with _refusing_float():
            return _meet2d(jet_at, xr[0], xr[1], yr[0], yr[1], order, tol, max_nodes)
    except FloatCoercionError as e:
        raise ValueError("integrate: the integrand is not composite -- %s" % e) from None


def integrate_surface(f, uv, surface, tol=1e-10, order=12, max_nodes=400):
    """Surface integral by meeting composites.  f is a scalar callable, or a list
    of three component callables for a flux.  Returns a float.

    The surface must be composite: math.sin(u) turns the seed into a float and
    loses the tangent, so a float coercion is refused, not worked around.
    Raises ValueError when the region is not resolved within max_nodes.
    """
    (a_u, b_u), (a_v, b_v) = uv
    jet_at = _surface_jet_at(f, surface, isinstance(f, list), order)
    try:
        with _refusing_float():
            return _meet2d(jet_at, a_u, b_u, a_v, b_v, order, tol, max_nodes)
    except FloatCoercionError as e:
        raise ValueError("integrate: the surface or the integrand is not composite -- %s"
                         % e) from None


# =============================================================================
# IMPROPER INTEGRALS
# =============================================================================

# =============================================================================
# FIXED: Improper integrals with composite tail analysis
# =============================================================================

def _improper_integral_panels(f, a, tol=1e-8, cutoff=20):
    """The panel path to +infinity, kept as the fallback for tails the library
    cannot represent.  Returns (Composite, float).

    Formerly the public improper_integral.  Compute integral from a to +infinity. Returns (Composite, float).

    Uses composite tail analysis with power-law VERIFICATION:
    two probes at M and M/2 — true power laws give the same exponent,
    exponential/Gaussian decay gives wildly different exponents.
    """
    # Start with a smaller M — grow from here if needed
    M = max(abs(a) + 1, 5.0)

    # Detect tail behavior from composite evaluation at M
    fx = _ensure_composite(f(_seeded(M)))
    f_val = fx.st()
    f_d1 = fx.d(1)

    if (abs(f_val) > 1e-100
            and math.isfinite(f_d1)
            and abs(f_d1) > 1e-100):
        alpha = M * f_d1 / f_val

        if alpha < -1.01:
            # VERIFY: true power law gives same alpha at M/2
            # For f = C*x^alpha:  alpha(M) = alpha(M/2) = alpha  (constant)
            # For f = exp(-x^2):  alpha(M) = -2M^2, alpha(M/2) = -M^2/2  (wildly different)
            M2 = M * 0.5
            fx2 = _ensure_composite(f(_seeded(M2)))
            f_val2 = fx2.st()
            f_d12 = fx2.d(1)

            is_power_law = False
            if (abs(f_val2) > 1e-100
                    and math.isfinite(f_d12)
                    and abs(f_d12) > 1e-100):
                alpha2 = M2 * f_d12 / f_val2
                # True power law: alpha and alpha2 within 30%
                if abs(alpha - alpha2) < 0.3 * max(abs(alpha), abs(alpha2)):
                    is_power_law = True

            if is_power_law:
                # A LOCAL exponent is not the ASYMPTOTIC one.  For 1/(1+x^2)
                # alpha(5) = -1.923 while the true tail exponent is -2, and the
                # 30% agreement test above happily accepts it -- which put
                # integral 1/(1+x^2) over the whole line 0.7% off pi.
                #
                # So do not trust alpha at the first M.  Push the cutoff out,
                # accumulating the bulk rather than recomputing it, and stop
                # when the ANSWER stops moving.  That tests what is actually
                # wanted instead of a proxy for it.
                bulk, bulk_err = integrate_adaptive(f, a, M, tol=tol)
                acc = bulk.st()
                prev_total = None
                for _ in range(60):
                    fxM = _ensure_composite(f(_seeded(M)))
                    vM, dM = fxM.st(), fxM.d(1)
                    if abs(vM) <= 1e-300 or not math.isfinite(dM):
                        return Composite({0: acc}), bulk_err
                    aM = M * dM / vM
                    if aM >= -1.0:
                        # The power-law reading has broken down.  For an
                        # OSCILLATING decay like e^-x sin(x), alpha = M(cos-sin)/sin
                        # swings with the phase: -6.48 at M=5 (which passes the
                        # 30% test at M/2) and +5.42 at M=10.  Returning the
                        # earlier estimate would keep a tail computed from a
                        # classification now known to be wrong, so DISCARD it and
                        # finish by integrating outward instead.
                        prev_total = None
                        break
                    C = vM / (M ** aM)
                    total = acc - C * (M ** (aM + 1)) / (aM + 1)
                    if (prev_total is not None
                            and abs(total - prev_total)
                                <= tol * max(1.0, abs(total))):
                        return Composite({0: total}), abs(total - prev_total)
                    prev_total = total
                    nxt = M * 2.0
                    seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
                    acc += seg.st()
                    bulk_err += seg_err
                    M = nxt
                    if M > 1e12:
                        break
                if prev_total is not None:
                    return Composite({0: prev_total}), bulk_err
                # Fall through: integrate outward and stop on what the tail
                # actually CONTRIBUTES, not on how big f looks at probe points.
                # Sampling can never be phase-immune -- a window tuned to catch
                # e^-x sin(x) still lands wrong for e^-x cos(x).  Integrating
                # the next octave answers the real question and costs one more
                # panel set.  Two consecutive negligible octaves, so a single
                # near-cancelling octave cannot end it early.
                quiet = 0
                for _ in range(80):
                    nxt = M * 2.0
                    seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
                    acc += seg.st()
                    bulk_err += seg_err
                    M = nxt
                    quiet = quiet + 1 if abs(seg.st()) <= tol * max(1.0, abs(acc)) else 0
                    if quiet >= 2 or M > 1e12:
                        break
                return Composite({0: acc}), bulk_err

    # Not power-law — integrate outward until successive octaves stop
    # contributing.  A pointwise |f(M)| < tol test fires at every zero of an
    # oscillating decay, truncating a tail that is still alive; what matters is
    # the CONTRIBUTION of the next stretch, so measure that.
    bulk, bulk_err = integrate_adaptive(f, a, M, tol=tol)
    acc = bulk.st()
    quiet = 0
    for _ in range(80):
        nxt = M * 2.0
        seg, seg_err = integrate_adaptive(f, M, nxt, tol=tol)
        acc += seg.st()
        bulk_err += seg_err
        M = nxt
        quiet = quiet + 1 if abs(seg.st()) <= tol * max(1.0, abs(acc)) else 0
        if quiet >= 2 or M > 1e12:
            break
    return Composite({0: acc}), bulk_err


def _improper_integral_panels_both(f, tol=1e-8):
    """The panel path over the whole line, split at 0: the fallback.
    Returns (Composite, float)."""
    left, left_err = _improper_integral_panels(lambda x: f(-x), 0, tol=tol)
    right, right_err = _improper_integral_panels(f, 0, tol=tol)
    return left + right, left_err + right_err


def improper_integral(f, a, tol=1e-8, cutoff=20):
    """int_a^inf f(x) dx.  Returns (Composite, error).

    Composite first, as integrate() does: the node at infinity is x = 1/h,
    an algebraic tail an ordinary composite there and an exponential one a
    transseries sector, and the tail starts where that node's jet reaches.
    The result is the composite with the pieces joined as terms of one number,
    and the error slot is nan -- nothing estimates one.

    Where the library says the tail is not representable (a Gaussian, a rate
    that is not an integer sector, an oscillation, float64 underflow) it falls
    back to the panel path, whose error estimate is returned as before.
    `cutoff` belongs to that path.
    """
    try:
        return _improper_jets_composite(f, a, math.inf, tol), float("nan")
    except (NotRepresentableError, NotImplementedError):
        return _improper_integral_panels(f, a, tol=tol, cutoff=cutoff)


def improper_integral_both(f, tol=1e-8):
    """int over the whole line, composite first, as improper_integral.
    Returns (Composite, error)."""
    try:
        return _improper_jets_composite(f, -math.inf, math.inf, tol), float("nan")
    except (NotRepresentableError, NotImplementedError):
        return _improper_integral_panels_both(f, tol=tol)


def improper_integral_to(f, a, b, tol=1e-8):
    """Integral from a to b where f may be singular at either end.  Returns
    (Composite, error).

    A singular end needs nothing special any more: integrate_jets reads the
    endpoint's jet by grade, so a power law, a log or a pole there is
    integrated exactly instead of having an exponent fitted from one nearby
    point.  The error slot is nan: integrate_jets accepts a panel when its two
    sides agree within tol, it does not estimate an error.
    """
    return integrate_jets(f, a, b, tol=tol), float("nan")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def show(composite: Composite, name: str = "result"):
    """Pretty print a composite number with extracted values"""
    print(f"{name} = {composite}")
    print(f"  st() = {composite.st()}")
    if -1 in composite.c:
        print(f"  f'   = {composite.d(1)}")
    if -2 in composite.c:
        print(f"  f''  = {composite.d(2)}")
    if -3 in composite.c:
        print(f"  f'''  = {composite.d(3)}")


class TracedComposite(Composite):
    """A Composite that prints each operation as it happens."""
    def _as_traced(self, result):
        """Re-wrap an operation's result so tracing survives the next step.

        Was `_wrap`, which SHADOWED Composite._wrap -- a classmethod taking
        (data, backend, demote, complete) -- with an instance method taking
        (result).  Nothing outside this class broke only because every call
        site spells it `Composite._wrap(...)` on the class rather than on an
        instance.

        And it assigned `tc.c = result.c`.  `.c` became a read-only property
        computed from ._data when the backends landed, so this raised
        AttributeError on the first traced operation -- trace() printed one
        line and died.  Copy the three slots instead.
        """
        if isinstance(result, Composite):
            tc = TracedComposite.__new__(TracedComposite)
            tc._backend = result._backend
            tc._data = result._data
            tc._complete = result._complete
            return tc
        return result
    def __add__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__add__(other)
        print(f"    {self}  +  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __radd__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__radd__(other)
        print(f"    {other_disp}  +  {self}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __sub__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__sub__(other)
        print(f"    {self}  -  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __mul__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__mul__(other)
        print(f"    {self}  ×  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __rmul__(self, other):
        other_disp = f"|{other}|₀"
        result = super().__rmul__(other)
        print(f"    {other_disp}  ×  {self}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __truediv__(self, other):
        other_disp = other if isinstance(other, Composite) else f"|{other}|₀"
        result = super().__truediv__(other)
        print(f"    {self}  ÷  {other_disp}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)
    def __pow__(self, n):
        result = super().__pow__(n)
        print(f"    ({self})^{n}")
        print(f"  = {result}")
        print()
        return self._as_traced(result)


def trace(f: Callable, at: float = None, to: float = None) -> Composite:
    """Trace composite computation showing ALL intermediate steps."""
    if to is not None:
        if to == float('inf'):
            x = TracedComposite({1: 1.0})
            print(f"\n=== TRACE: lim(x→∞) ===")
            print(f"Let x = |1|₁  (INF)\n")
        elif to == float('-inf'):
            x = TracedComposite({1: -1.0})
            print(f"\n=== TRACE: lim(x→-∞) ===")
            print(f"Let x = |-1|₁  (-INF)\n")
        elif to == 0:
            x = TracedComposite({-1: 1.0})
            print(f"\n=== TRACE: lim(x→0) ===")
            print(f"Let x = |1|₋₁  (ZERO)\n")
        else:
            x = TracedComposite({0: float(to), -1: 1.0})
            print(f"\n=== TRACE: lim(x→{to}) ===")
            print(f"Let x = |{to}|₀ + |1|₋₁\n")
    elif at is not None:
        x = TracedComposite({0: float(at), -1: 1.0})
        print(f"\n=== TRACE: f'({at}) ===")
        print(f"Let x = |{at}|₀ + |1|₋₁  (i.e., {at} + h)\n")
    else:
        x = TracedComposite({-1: 1.0})
        print(f"\n=== TRACE ===")
        print(f"Let x = |1|₋₁  (ZERO)\n")
    result = f(x)
    if isinstance(result, (int, float)):
        result = Composite(result)
    print(f"RESULT: {result}")
    if to is not None:
        print(f"Limit = {result.st()}")
    else:
        print(f"f({at if at else 0}) = {result.st()}")
        if -1 in result.c:
            print(f"f'({at if at else 0}) = {result.d(1)}")
    return Composite(result.c) if isinstance(result, TracedComposite) else result


def translate(f: Callable, at: float = None, to: float = None) -> Composite:
    """Show the composite translation WITHOUT resolving."""
    if to is not None:
        if to == float('inf'):
            x = INF
            sub_str = "x = INF"
        elif to == float('-inf'):
            x = -INF
            sub_str = "x = -INF"
        elif to == 0:
            x = ZERO
            sub_str = "x = ZERO"
        else:
            x = _seeded(to)
            sub_str = f"x = _seeded({to})"
    elif at is not None:
        x = _seeded(at)
        sub_str = f"x = _seeded({at})"
    else:
        x = ZERO
        sub_str = "x = ZERO"
    result = f(x)
    print(f"Substitution: {sub_str}")
    print(f"Translation:  {result}")
    print(f"")
    if to is not None:
        print(f"Limit = {result.st()}")
    else:
        print(f"f({at if at else 0}) = {result.st()}")
        if -1 in result.c:
            print(f"f'({at if at else 0}) = {result.d(1)}")
        if -2 in result.c:
            print(f"f''({at if at else 0}) = {result.d(2)}")
    return result


def verify_derivative(f: Callable, f_prime: Callable, at: float, tol: float = 1e-6) -> bool:
    """Verify that f_prime is indeed the derivative of f at a point."""
    computed = derivative(f, at)
    expected = f_prime(at) if callable(f_prime) else f_prime
    return abs(computed - expected) < tol


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    """Run basic tests to verify the library works"""
    print("=" * 60)
    print("COMPOSITE LIBRARY TEST SUITE (FIXED v3: EXPRESSED ZERO)")
    print("=" * 60)

    tests = []

    # Derivative tests
    print("\n--- Derivatives ---")

    d1 = derivative(lambda x: x**2, at=3)
    tests.append(("d/dx[x²] at x=3", d1, 6))
    print(f"d/dx[x²] at x=3 = {d1}, expected 6 {'✓' if abs(d1-6)<1e-6 else '✗'}")

    d2 = derivative(lambda x: x**3, at=2)
    tests.append(("d/dx[x³] at x=2", d2, 12))
    print(f"d/dx[x³] at x=2 = {d2}, expected 12 {'✓' if abs(d2-12)<1e-6 else '✗'}")

    d3 = derivative(lambda x: sin(x), at=0)
    tests.append(("d/dx[sin(x)] at x=0", d3, 1))
    print(f"d/dx[sin(x)] at x=0 = {d3}, expected 1 {'✓' if abs(d3-1)<1e-6 else '✗'}")

    d4 = nth_derivative(lambda x: x**5, n=3, at=2)
    tests.append(("d³/dx³[x⁵] at x=2", d4, 240))
    print(f"d³/dx³[x⁵] at x=2 = {d4}, expected 240 {'✓' if abs(d4-240)<1e-6 else '✗'}")

    # Limit tests
    print("\n--- Limits ---")

    l1 = limit(lambda x: sin(x)/x, as_x_to=0)
    tests.append(("lim sin(x)/x as x→0", l1, 1))
    print(f"lim sin(x)/x as x→0 = {l1}, expected 1 {'✓' if abs(l1-1)<1e-6 else '✗'}")

    l2 = limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)
    tests.append(("lim (x²-4)/(x-2) as x→2", l2, 4))
    print(f"lim (x²-4)/(x-2) as x→2 = {l2}, expected 4 {'✓' if abs(l2-4)<1e-6 else '✗'}")

    l3 = limit(lambda x: (1 - cos(x))/(x**2), as_x_to=0)
    tests.append(("lim (1-cos(x))/x² as x→0", l3, 0.5))
    print(f"lim (1-cos(x))/x² as x→0 = {l3}, expected 0.5 {'✓' if abs(l3-0.5)<1e-6 else '✗'}")

    l4 = limit(lambda x: (exp(x) - 1)/x, as_x_to=0)
    tests.append(("lim (eˣ-1)/x as x→0", l4, 1))
    print(f"lim (eˣ-1)/x as x→0 = {l4}, expected 1 {'✓' if abs(l4-1)<1e-6 else '✗'}")

    # Special values
    print("\n--- Special Values ---")

    s1 = (ZERO / ZERO).st()
    tests.append(("0/0", s1, 1))
    print(f"ZERO / ZERO = {s1}, expected 1 {'✓' if abs(s1-1)<1e-6 else '✗'}")

    s2 = (INF * ZERO).st()
    tests.append(("∞ × 0", s2, 1))
    print(f"INF * ZERO = {s2}, expected 1 {'✓' if abs(s2-1)<1e-6 else '✗'}")

    s3 = ((R(5) * ZERO) / ZERO).st()
    tests.append(("(5×0)/0", s3, 5))
    print(f"(R(5) * ZERO) / ZERO = {s3}, expected 5 {'✓' if abs(s3-5)<1e-6 else '✗'}")

    # Fix 1 verification: transcendentals on plain floats return Composite
    print("\n--- Fix 1: Transcendentals return Composite ---")

    sin_plain = sin(0.5)
    is_composite = isinstance(sin_plain, Composite)
    tests.append(("sin(0.5) returns Composite", 1 if is_composite else 0, 1))
    print(f"sin(0.5) returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  sin(0.5).st() = {sin_plain.st():.6f}, math.sin(0.5) = {math.sin(0.5):.6f}")

    exp_plain = exp(1.0)
    is_composite = isinstance(exp_plain, Composite)
    tests.append(("exp(1.0) returns Composite", 1 if is_composite else 0, 1))
    print(f"exp(1.0) returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  exp(1.0).st() = {exp_plain.st():.6f}, math.exp(1.0) = {math.exp(1.0):.6f}")

    # Fix 2 verification: integration returns Composite
    print("\n--- Fix 2: Integration returns Composite ---")

    int_result, int_err = integrate_adaptive(lambda x: x**2, 0, 1, tol=1e-8)
    is_composite = isinstance(int_result, Composite)
    tests.append(("integrate_adaptive returns Composite", 1 if is_composite else 0, 1))
    print(f"integrate_adaptive returns Composite: {is_composite} {'✓' if is_composite else '✗'}")
    print(f"  ∫x² dx from 0 to 1 = {int_result.st():.6f}, expected 0.333333")

    # v2 verification: line integral through integrate_adaptive
    print("\n--- v2: Line Integral (Composite-First) ---")

    line_result = integrate(
        [lambda x, y: y, lambda x, y: -x],
        (0, 2 * math.pi),
        curve=lambda t: [cos(t), sin(t)]
    )
    expected_line = -2 * math.pi
    line_ok = abs(line_result - expected_line) < 0.1
    tests.append(("∫_C F·dr (unit circle)", line_result, expected_line))
    print(f"∫_C [y,-x]·dr (unit circle) = {line_result:.6f}, expected {expected_line:.6f} {'✓' if line_ok else '✗'}")

    # v3 verification: expressed zero preservation
    print("\n--- Fix 3: Expressed Zero Preservation ---")

    zero_sub = R(1) - R(1)
    has_dim0 = 0 in zero_sub.c
    tests.append(("R(1)-R(1) retains dim 0", 1 if has_dim0 else 0, 1))
    print(f"R(1) - R(1) = {zero_sub}")
    print(f"  dim 0 retained: {has_dim0} {'✓' if has_dim0 else '✗'}")
    print(f"  .c = {zero_sub.c}")

    comp_zero = Composite(0)
    has_dim0 = 0 in comp_zero.c
    tests.append(("Composite(0) retains dim 0", 1 if has_dim0 else 0, 1))
    print(f"Composite(0) = {comp_zero}")
    print(f"  dim 0 retained: {has_dim0} {'✓' if has_dim0 else '✗'}")

    empty = Composite()
    is_empty = len(empty.c) == 0
    tests.append(("Composite() is truly empty", 1 if is_empty else 0, 1))
    print(f"Composite() = {empty}")
    print(f"  truly empty: {is_empty} {'✓' if is_empty else '✗'}")

    # All derivatives at once
    print("\n--- All Derivatives ---")

    derivs = all_derivatives(lambda x: exp(x), at=0, up_to=5)
    print(f"All derivatives of eˣ at x=0: {[round(d,2) for d in derivs]}")
    print(f"Expected: [1, 1, 1, 1, 1, 1] {'✓' if all(abs(d-1)<1e-6 for d in derivs) else '✗'}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, actual, expected in tests if abs(actual - expected) < 0.1)
    print(f"PASSED: {passed}/{len(tests)}")
    print("=" * 60)

    return passed == len(tests)


# =============================================================================
# TYPE PRESERVATION ACROSS THE ELEMENTARY FUNCTIONS
# =============================================================================
#
# `cos(x)` built its result with `Composite(...)`, so a subclass handed in as
# `x` was dropped at the first call and everything after it ran unwatched.
# That is why forensics needed its own namespace (`F.cos`): it re-wrapped the
# result afterwards, and a formula written against the library's own `cos`
# silently came back "stable".
#
# Here the function hands its result back through the operand's `_fn_result`
# hook instead.  Three properties matter:
#
#   plain in, plain out   `type(x) is Composite` takes the fast path and this
#                         costs one type check.
#   internals stay plain  the argument is downcast before the function runs, so
#                         the Taylor loops inside never see a subclass and a
#                         subclass never records the library's own arithmetic.
#   opt-in                a subclass that does not override `_fn_result` gets
#                         exactly the old behaviour.

def _preserves_type(fn):
    """Run `fn` on a plain composite, then hand the result back through `x`."""

    def wrapper(x, *args, **kwargs):
        plain = type(x) is Composite or not isinstance(x, Composite)
        # R1 on the argument HERE, before its marker is read below.  Every fn
        # runs _r1 on its argument itself, and does so after this wrapper has
        # read the marker off the unconverted zero: sin(R(0)) built sin(h)
        # and came back with denotation_order None.  Converted here, fn's own
        # _r1 finds no wholly-zero operand and is a no-op.
        if plain and isinstance(x, Composite):
            x = _r1(x)
        out = fn(x if plain else x._as_plain(), *args, **kwargs)
        if not plain:
            out = x._fn_result(out, fn)
        # CARRY THE DENOTATION MARKER.  A transcendental expands about st(x),
        # so f(st) + f'(st)*(rest) puts a denotation in the argument at the
        # SAME order in the result.  It is carried here because this wrapper is
        # the one point all fifteen of them pass through, and because the
        # plain-Composite branch returns fn's result directly without reaching
        # _fn_result -- which is exactly where the marker was being dropped:
        # exp(x + (R(6)-R(6))) came back with denotation_order None.
        _dx = _denot_of(x) if isinstance(x, Composite) else None
        if _dx is not None and isinstance(out, Composite):
            out._denot = _merge_denot(_denot_of(out), _dx)
        return out

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


for _fname in ("sin", "cos", "tan", "exp", "ln", "sqrt", "sinh", "cosh",
               "tanh", "atan", "asin", "acos", "erf", "erfc", "normal_cdf"):
    if _fname in globals():
        globals()[_fname] = _preserves_type(globals()[_fname])
del _fname


# Routines that build deep intermediate series and read one number off the end.
# A caller's order cap is an economy on what comes BACK, never a budget for the
# work in between: capped, these degraded silently instead of getting cheaper.
for _fname in ("limit", "limit_right", "limit_left", "integrate",
               "integrate_stepped", "integrate_adaptive", "improper_integral",
               "improper_integral_both", "improper_integral_to",
               "antiderivative"):
    if _fname in globals():
        globals()[_fname] = _needs_full_order(globals()[_fname])
del _fname


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_tests()

    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)

    print("\n# Direct composite computation:")
    print("h = ZERO")
    print("x = R(3) + h")
    x = R(3) + h
    result = x**2
    print(f"(R(3) + h)**2 = {result}")
    print(f"  Value at x=3: {result.st()}")
    print(f"  Derivative:   {result.d(1)}")

    print("\n# High-level API:")
    print(f"derivative(lambda x: x**2, at=3) = {derivative(lambda x: x**2, at=3)}")
    print(f"limit(lambda x: sin(x)/x, as_x_to=0) = {limit(lambda x: sin(x)/x, as_x_to=0)}")

    print("\n# Line integral (composite-first):")
    print("∫_C [y,-x]·dr around unit circle:")
    result = integrate(
        [lambda x, y: y, lambda x, y: -x],
        (0, 2 * math.pi),
        curve=lambda t: [cos(t), sin(t)]
    )
    print(f"  = {result:.6f}, expected {-2*math.pi:.6f}")
