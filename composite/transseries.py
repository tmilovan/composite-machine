# composite/transseries.py
# Composite Machine — the scale below every power
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""exp(-1/h): a value the powers-and-logs group cannot hold, given a place.

Composite's dimensions are powers and iterated logarithms -- the
powers-and-logs fragment of a Hardy field -- ordered lexicographically, which
IS the dominance order.  exp(-1/h) is below every power of h, and that group
has no slot there.  So the same asymptotic series belongs to a whole family:
S and S + C*exp(-1/h) have identical coefficients to all orders, and no number
of coefficients tells them apart.  `composite.resummation` measures the
difference instead -- pi/e = 1.1557273497909217 for Euler-Stieltjes at
eps = -1, by two independent routes -- so C is a number, not a definition.

WHAT THIS MODULE DOES AND DOES NOT DO.  The arithmetic HOLDS the flat term;
the Borel layer FINDS it.  The exponent comes from the position of the Borel
singularity and the size from its residue (or, at a branch point, from the gap
between the two lateral contours) -- neither is derived inside the composite
algebra, and nothing here computes exp(-1/h) from the coefficients.  What this
module contributes is the place to put it and the order relation that makes it
behave: a value group with a slot below every power, closed under
multiplication and under d/dh.  The bridge between the two layers -- sector
coefficients out, a resummed number back, the ambiguity checked against the
next sector -- is the actual claim.  Read any sentence below in that light.

HOW IT IS CARRIED, and why not as another dimension axis.

The obvious move is to prepend an exponential component to the dimension
vector, so lexicographic comparison keeps giving dominance for free.  It was
measured and it is the wrong move, twice over:

  * SPEED.  Tuple dimensions are excluded from SparseDenseBackend by design --
    that exclusion is what stops the fast scalar path regressing.  On 120
    terms, one multiply: SparseDense scalar 0.1 ms, DictBackend tuple 15.7 ms,
    VectorDimBackend tuple 28.7 ms.  150x to 300x, paid on every sector
    operation, to buy a comparison that is ten lines to write.

  * CORRECTNESS.  Dimension padding happens at the TAIL, and tail-padding is
    the identity ONLY because index 0 is the power axis.  Put the exponential
    at index 0 and a bare scalar dimension changes meaning:
    _dim_add(-3, (0,-1)) returns (-3,-1), which the new reading calls
    exponential -3, power -1, where the truth is (0,-4).  Every scalar/tuple
    interaction in the library silently means something else.

So the power axis keeps index 0 forever, and the exponential level rides
OUTSIDE the dimension as a sector index: a Transseries is a sparse map from an
integer sector to an ordinary Composite.  Sector n carries exp(-n/h), the
sectors multiply by ADDING, and each sector's series stays a scalar-dimension
Composite on the fast path.

THE SECTOR INDEX IS IN UNITS OF THE ACTION.  For Painleve I the action is
A = 1.770691071520514, and the sectors are exp(-A t^(5/4)), exp(-2A t^(5/4)),
exp(-3A t^(5/4)).  Carrying A in the index would put 3A on a grade that float
addition cannot represent, and the library refuses inexact grade addition --
correctly.  Measuring in units of A makes the indices integers and their
addition exact, which is also the standard resurgence convention.

ORDERING IS THE CONTENT.  Sector first, smaller index dominating (sector -1 is
exp(+1/h), beyond every power upward; sector +1 is exp(-1/h), beyond every
power downward), then the existing order inside the sector.  If that
comparison is wrong nothing above it is recoverable, which is why it is what
tests/test_transseries.py tests before any arithmetic.
"""
import math

from .composite_lib import (Composite, R, ZERO, exp as _c_exp, ln as _c_ln,
                             _r1 as _apply_r1, _vec_composite as _vec)
from .backends.dict_backend import _dim_key, _dim_add
from .resummation import (resum, resum_median, classify_singularity,
                          SingularityKind)

# Enough coefficients for a [2/2].  Below this there is nothing for Pade to
# approximate and the series is summed as written.
_MIN_FOR_BOREL = 5


def _is_zero(c):
    """True when a Composite carries no non-zero coefficient.

    Expressed zeros are kept by the library on purpose -- a dimension exists
    because the computation built it -- so this asks about VALUES, not keys.
    """
    return not any(v != 0.0 for v in c.coeffs_dict().values())


def _rebuild(d):
    """A Composite from a {dim: coeff} map, scalar or vector keys alike."""
    if not d:
        return Composite({})
    return _vec(d) if any(isinstance(k, tuple) for k in d) else Composite(d)


def _term_add(a, b):
    """a + b with a WHOLLY ZERO side kept inert.

    A sector's series is a TERM of the transseries, not a number in its own
    right, and R2 says a zero term is not an operand -- "there is nothing for
    R1 to do, and the term is retained".  Going through Composite.__add__
    makes it an operand, so R1 fired and converted it.  Measured before this:

        A = {0: R(1), 1: (c-c)},  B = {1: R(5)}
        (A + B).sectors[1]  ->  {-1: 1.0, 0: 5.0}      should be {0: 5.0}

    -- an infinitesimal manufactured inside a sector nobody used as an operand.
    """
    if not (_is_zero(a) or _is_zero(b)):
        return a + b
    out = dict(a.coeffs_dict())
    for d, v in b.coeffs_dict().items():
        out[d] = out.get(d, 0.0) + v
    return _rebuild(out)


def _term_sub(a, b):
    """a - b, same rule."""
    if not (_is_zero(a) or _is_zero(b)):
        return a - b
    out = dict(a.coeffs_dict())
    for d, v in b.coeffs_dict().items():
        out[d] = out.get(d, 0.0) - v
    return _rebuild(out)


def _term_mul(a, b):
    """a * b with a WHOLLY ZERO side kept inert.

    R5: a product retains the dimensions it constructs, so a zero times
    something is a ZERO at the summed dimensions.  Before this,
    (A * 2).sectors[1] came back {-1: 2.0} where R2 and R5 give {0: 0.0} --
    the zero was converted AND then scaled.
    """
    if not (_is_zero(a) or _is_zero(b)):
        return a * b
    out = {}
    for da, va in a.coeffs_dict().items():
        for db, vb in b.coeffs_dict().items():
            k = _dim_add(da, db)
            out[k] = out.get(k, 0.0) + va * vb
    return _rebuild(out)


def _lead_sign(c):
    """Sign of the DOMINANT term: +1, -1, or 0 for a composite that is zero.

    Dominant is the largest dimension, lexicographically, which _dim_key
    orders across scalar and tuple dimensions alike -- max() on a raw mix of
    the two raises TypeError.
    """
    d = {k: v for k, v in c.coeffs_dict().items() if v != 0.0}
    if not d:
        return 0
    k = max(d, key=_dim_key)
    return 1 if d[k] > 0 else -1


def _lead_dim(c):
    d = {k: v for k, v in c.coeffs_dict().items() if v != 0.0}
    if not d:
        return None
    return max(d, key=_dim_key)


def _dim_is_positive(d):
    """Lexicographically above zero: the infinite side, on whichever axis leads."""
    if not isinstance(d, tuple):
        return d > 0
    for c in d:
        if c != 0:
            return c > 0
    return False


def _eval_composite(c, hv):
    """Substitute h -> hv.  Grade -n is h**n, so the exponent is -grade.

    Composite.eval_taylor does this for the NEGATIVE grades only -- it is a
    Taylor-polynomial evaluation and drops the constant and everything
    infinite.  A sector's prefactor is routinely infinite (pi/eps is h**-1),
    so this one keeps every grade.
    """
    total = 0.0
    for d, v in c.coeffs_dict().items():
        if v == 0.0:
            continue
        if isinstance(d, tuple):
            raise NotImplementedError(
                f"evaluating a log-axis dimension {d!r} needs a value for "
                f"ln(1/h) as well as for h; only the power axis is supported "
                f"here.")
        total += float(v) * hv ** (-float(d))
    return total


def sector_coefficients(c):
    """(p, [c_0, c_1, ...]) with the composite read as h**-p * sum c_k h**k.

    Grade d is h**-d, so the most POSITIVE grade is the most infinite term and
    it comes out as the prefactor.  A sector's prefactor is routinely infinite
    -- the Euler-Stieltjes flat term is (pi/eps)exp(-1/eps), whose prefactor is
    h**-1 -- and Borel-Pade wants a series starting at h**0, so the prefactor
    is factored out here and multiplied back after the sum.
    """
    d = {}
    for k, v in c.coeffs_dict().items():
        if isinstance(k, tuple):
            raise NotImplementedError(
                f"a log-axis dimension {k!r} is not a power series in h; "
                f"Borel-Pade has no variable to sum over here.")
        d[float(k)] = v
    d = {k: v for k, v in d.items() if v != 0.0}
    if not d:
        return 0.0, []
    p = max(d)
    lo = min(d)
    n = p - lo
    if abs(n - round(n)) > 0:
        raise NotImplementedError(
            f"grades {sorted(d)} are not a whole number of steps apart, so "
            f"they are not the coefficients of a power series in h.")
    return p, [d.get(p - k, 0.0) for k in range(int(round(n)) + 1)]


def resum_sector(c, h_value, L=None, M=None, force=None):
    """(value, ambiguity) for ONE sector's series at h = h_value.

    THIS IS THE BRIDGE.  A sector's coefficients are a formal series that
    generally diverges factorially -- Euler-Stieltjes runs 1, 1, 2, 6, 24 and
    its partial sums go 1, 2, 4, 10, 34 away from the answer -- so summing it
    as written is not an evaluation at all.  composite.resummation turns it
    into a number, and which route it takes is itself information:

      * ray clear   -> resum(), ambiguity 0.  The number is the value.
      * ray BLOCKED -> resum_median(), ambiguity > 0.  The Borel singularity
        sits on the integration path, the sum exists only as a boundary value
        from one side, and the two sides differ by the flat term.

    A NON-ZERO AMBIGUITY IS THE POINT, not a failure: it is the size of the
    next sector, measured from sector 0 alone.  T4 checks the two against each
    other.
    """
    p, coeffs = sector_coefficients(c)
    scale = h_value ** (-p)
    use = (len(coeffs) >= _MIN_FOR_BOREL) if force is None else force
    if not use:
        return _eval_composite(c, h_value), 0.0
    if len(coeffs) < 3:
        raise ValueError(f"Borel-Pade needs at least 3 coefficients, got "
                         f"{len(coeffs)}")
    try:
        v, _ = resum(coeffs, h_value, L, M)
        return float(v) * scale, 0.0
    except NotImplementedError:
        med, amb = resum_median(coeffs, h_value, L, M)
        return med.real * scale, amb * scale


class Transseries:
    """A sparse map {sector: Composite}.  Sector n carries exp(-n/h)."""

    __slots__ = ("sectors",)

    def __init__(self, sectors=None):
        # A sector whose series is wholly zero is KEPT, not dropped.  R2: a
        # zero term is inert and is kept -- the sector exists because the
        # computation built it, and discarding it is the same mistake as
        # discarding a zero coefficient.  Only a sector that was never
        # supplied is absent.
        s = {}
        for n, c in (sectors or {}).items():
            s[int(n)] = c if isinstance(c, Composite) else Composite(c)
        self.sectors = s

    # -- construction --------------------------------------------------------
    @staticmethod
    def lift(x):
        """An ordinary value as a transseries: everything in sector 0.

        A WRITTEN ZERO IS EXPRESSED, so `lift(0)` is |0|_0 at sector 0 and R1
        reaches it exactly as it does at the Composite layer.  It is not
        NOTHING: nothing is the absence of a sector, which is what
        `Transseries()` gives.

        This briefly lifted 0 to NOTHING, which left the two layers obeying
        different rules one call apart -- `0` meant h in a Composite and an
        absence in a Transseries.
        """
        if isinstance(x, Transseries):
            return x
        return Transseries({0: x if isinstance(x, Composite) else Composite(x)})

    # -- shape ---------------------------------------------------------------
    def leading_sector(self):
        """The dominating sector: the smallest index that holds a NON-ZERO series.

        Smaller means larger: sector -1 is exp(+1/h) and outgrows every power;
        sector +1 is exp(-1/h) and is outgrown by every power.

        Sectors holding only a zero are skipped.  A zero sitting among other
        terms is a TERM, not an operand (R2), so it contributes nothing to the
        dominant behaviour -- `exp(-1/h) + 0` keeps a zero at sector 0 and is
        still an infinitesimal.  When EVERY sector is zero the number is a
        zero, and R1 decides it; see _r1.
        """
        ns = [n for n, c in self.sectors.items() if not _is_zero(c)]
        return min(ns) if ns else None

    def __repr__(self):
        if not self.sectors:
            return "Transseries(NOTHING)"
        parts = []
        for n in sorted(self.sectors):
            c = self.sectors[n]
            tag = "" if n == 0 else f"*exp({-n}/h)"
            parts.append(f"[{c}]{tag}")
        return "Transseries(" + " + ".join(parts) + ")"

    # -- arithmetic ----------------------------------------------------------
    # R1 ACTS ON THE OPERAND, WHICH IS THE WHOLE TRANSSERIES.  _r1() below
    # fires only when every sector is zero; after that the per-sector helpers
    # (_term_add / _term_sub / _term_mul) keep a zero SECTOR inert, because a
    # sector is a term of the number and R2 says a term is not an operand.
    # Getting either half wrong leaks: without _r1 the two layers disagreed --
    # Composite(0)+R(5) gave {-1:1.0, 0:5.0} and lift(0)+lift(5) gave {0:5.0};
    # without the helpers a zero sector converted inside a number nobody used
    # as an operand.
    def __add__(self, other):
        a, b = self._r1(), Transseries.lift(other)._r1()
        out = dict(a.sectors)
        for n, c in b.sectors.items():
            out[n] = _term_add(out[n], c) if n in out else c
        return Transseries(out)

    __radd__ = __add__

    def __neg__(self):
        return Transseries({n: -c for n, c in self.sectors.items()})

    def __sub__(self, other):
        """Per sector, through the library's own subtraction.

        NOT `self + (-other)`.  Composite.__sub__ applies R1 to the operands
        and negates AFTER, so `h - 0` is `h - h`.  Negating first hands R1 a
        wholly zero composite whose sign it discards -- R1 sets the converted
        coefficient to 1.0 -- and `h - 0` came out as `2h`.
        """
        a, b = self._r1(), Transseries.lift(other)._r1()
        out = dict(a.sectors)
        for n, c in b.sectors.items():
            out[n] = _term_sub(out[n], c) if n in out else -c
        return Transseries(out)

    def __rsub__(self, other):
        return Transseries.lift(other) + (-self)

    def __mul__(self, other):
        """Sectors ADD: exp(-m/h)*exp(-n/h) = exp(-(m+n)/h).

        Which is the same rule the dimensions already follow under convolution
        -- it is simply applied one level up, where the fast path is not paid
        for it.
        """
        sa, sb = self._r1(), Transseries.lift(other)._r1()
        out = {}
        for m, a in sa.sectors.items():
            for n, b in sb.sectors.items():
                k = m + n
                p = _term_mul(a, b)
                out[k] = _term_add(out[k], p) if k in out else p
        return Transseries(out)

    __rmul__ = __mul__

    # -- ordering: the whole point ------------------------------------------
    def _r1(self):
        """R1 for a transseries: a WHOLLY zero operand converts.

        Every sector zero means the whole number is a zero, so R1 applies and
        the LOWEST dimension converts -- which here is the most infinitesimal
        term, i.e. the deepest sector's lowest dimension.  Without this a
        written 0 stayed an inert |0|_0 at sector 0 and never became h, so
        `exp(-1/h) > 0` answered against an absence instead of against h.
        """
        if not self.sectors or any(not _is_zero(c) for c in self.sectors.values()):
            return self
        deep = max(self.sectors)
        out = dict(self.sectors)
        out[deep] = _apply_r1(out[deep])
        return Transseries(out)

    def _cmp(self, other):
        d = self._r1() - Transseries.lift(other)._r1()
        for n in sorted(d.sectors):          # dominating sector first
            s = _lead_sign(d.sectors[n])
            if s:
                return s
        return 0

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __ne__(self, other):
        return self._cmp(other) != 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __hash__(self):
        return hash(tuple(sorted(self.sectors)))

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, h_value, borel=None, L=None, M=None):
        """Substitute a value for h.  Sector n contributes exp(-n/h) times its
        own series, so a sector is exponentially suppressed however infinite
        its prefactor is -- which is the property the ordering asserts.

        Divergent sectors go through Borel-Pade (see resum_sector); `borel`
        forces that on or off, and defaults to on for any sector carrying at
        least 5 coefficients.  Borel summation agrees with ordinary summation
        wherever the latter converges, so this is not a choice between two
        answers.
        """
        return self.evaluate_with_ambiguity(h_value, borel, L, M)[0]

    def evaluate_parts(self, h_value, borel=None, L=None, M=None):
        """{sector: its contribution at h}.  THE DECOMPOSITION, not the sum.

        A single float cannot hold the answer once the flat term drops below
        the grade-0 part's last bit.  At h = 0.01 the flat term is exp(-100) =
        3.7e-44 and the leading part is O(1), so float addition discards it
        entirely -- 28 orders below the relative floor.  Summing is then not a
        lossy evaluation, it is a total one: the term is gone, and nothing in
        the result says it was ever there.  So the parts stay separable, and
        `evaluate` is the convenience, not the representation.
        """
        if h_value <= 0:
            raise ValueError(f"h must be positive to evaluate exp(-n/h); "
                             f"got {h_value!r}")
        out = {}
        for n, c in self.sectors.items():
            v, _ = resum_sector(c, h_value, L, M, force=borel)
            out[n] = math.exp(-n / h_value) * v
        return out

    def flat_part(self, h_value, borel=None, L=None, M=None):
        """Everything beyond all orders: the sectors above 0, summed.

        Readable at any h, including where it is far below what adding it to
        the grade-0 part could ever show.
        """
        return sum(v for n, v in
                   self.evaluate_parts(h_value, borel, L, M).items() if n > 0)

    def evaluate_with_ambiguity(self, h_value, borel=None, L=None, M=None):
        """(value, ambiguity).  A non-zero ambiguity means some sector's Borel
        ray was blocked, and the value is the median of the two sides."""
        if h_value <= 0:
            raise ValueError(f"h must be positive to evaluate exp(-n/h); "
                             f"got {h_value!r}")
        total, amb = 0.0, 0.0
        for n, c in self.sectors.items():
            w = math.exp(-n / h_value)
            v, a = resum_sector(c, h_value, L, M, force=borel)
            total += w * v
            amb += w * a
        return total, amb


# =============================================================================
def action_from_growth(coeffs, skip=4):
    """The action read off the COEFFICIENT RATIO, with one Richardson step.

    c_n = n! b_n and b_n ~ n^(beta-1) A^-n, so |c_{n+1}/c_n| -> (n+1)/A and
    the estimate A_n = n/ratio falls as A + const/n.  One Richardson step
    removes the 1/n.

    This is the better estimator AT A BRANCH POINT, where Pade only ever lays
    another link of a chain and its leading root creeps in from outside.
    Measured:

        series            Pade |z0|            growth + Richardson   truth
        Airy (Riccati)    1.3508  (1.3% high)  1.33397  (5e-4)       4/3
        Painleve I        1.8425  (4% high)    1.770724 (3e-5)       1.770691

    At a genuine POLE the comparison reverses -- Pade reproduces it exactly
    (Euler-Stieltjes: z0 = -1.000000000000 at every order) -- so from_series
    uses Pade for a pole and this for a cut.
    """
    nz = [n for n, v in enumerate(coeffs) if v != 0.0]
    if len(nz) < 4:
        return None
    # STRIDE.  A lacunary series -- Painleve I's is even, Stirling's odd -- has
    # a zero in every consecutive pair, and reading ratios pairwise skips the
    # whole series.  With coefficients only at n, n+s, n+2s, the ratio over one
    # stride is |c_{n+s}/c_n| ~ (n/A)^s.
    gaps = {nz[i + 1] - nz[i] for i in range(len(nz) - 1)}
    if len(gaps) != 1:
        return None
    step = gaps.pop()
    est = []
    for i in range(len(nz) - 1):
        n = nz[i]
        if n < skip:
            continue
        r = abs(coeffs[nz[i + 1]] / coeffs[n])
        if r > 0.0:
            est.append((n, n / r ** (1.0 / step)))
    if len(est) < 2:
        return None
    (n1, a1), (n0, a0) = est[-1], est[-2]
    # A_n = A + c/n sampled at n1 and n0 = n1 - step
    return a1 + (n1 - step) * (a1 - a0) / step


def from_series(coeffs, L=None, M=None, action=None):
    """Build a Transseries FROM a divergent series -- the constructive direction.

    Everywhere else the bridge runs one way: a sector goes in, a number comes
    back.  This is the other way.  Given only the coefficients of a divergent
    asymptotic series it locates the Borel singularity, reads the ACTION off
    its distance, rescales the variable into units of that action so the
    sectors are integers, and returns a transseries with sector 0 filled and
    the exponent of sector 1 fixed -- exp(-1/h) is then exp(-A/w) in the
    original variable.

    THE COEFFICIENT OF SECTOR 1 IS ONLY SET WHEN THE SINGULARITY IS A RESOLVED
    POLE, because that is the only case where a residue exists to size it.
    For a branch point the exponent is still determined -- it is the action --
    and the coefficient is left unset rather than invented.  `info["stokes"]`
    is None there, and the caller can measure it from the lateral gap instead.

    Nothing here derives exp(-1/h) from the coefficients: the algebra HOLDS
    the flat term, composite.resummation FINDS it.  What this function does is
    put the two together, so a problem -- rather than a person -- builds the
    object.

    Returns (Transseries, info) with info carrying kind, z0, action, stokes
    and the classifier's detail.
    """
    kind, z0, res, det = classify_singularity(coeffs, L, M)
    raw = Composite({-k: v for k, v in enumerate(coeffs) if v != 0.0})
    if z0 is None:
        return (Transseries({0: raw}),
                {"kind": kind, "z0": None, "action": None, "stokes": None,
                 "detail": det})

    if action is not None:
        A = float(action)
    elif kind == SingularityKind.POLE:
        A = abs(z0)                       # Pade reproduces a pole exactly
    else:
        A = action_from_growth(coeffs) or abs(z0)   # see action_from_growth
    if not (A > 0):
        raise ValueError(f"action must be positive; got {A!r} from z0={z0!r}")

    # h = w / A, so that the flat term exp(-A/w) becomes exp(-1/h) and the
    # sector indices are integers -- see the module docstring on units.
    sectors = {0: Composite({-k: v * A ** k
                             for k, v in enumerate(coeffs) if v != 0.0})}
    stokes = None
    if kind == SingularityKind.POLE and res is not None:
        # The one-instanton size the residue gives: pi*|Res|*exp(-A/w)/w,
        # and w = A*h, so the prefactor is (pi*|Res|/A) * h^-1.
        stokes = math.pi * abs(res) / A
        sectors[1] = Composite({1.0: stokes})
    return (Transseries(sectors),
            {"kind": kind, "z0": z0, "action": A, "stokes": stokes,
             "action_pade": abs(z0), "action_growth": action_from_growth(coeffs),
             "detail": det})


def sector(n, c=1.0):
    """The transseries c*exp(-n/h)."""
    return Transseries({int(n): c if isinstance(c, Composite) else Composite(c)})


def flat(n=1):
    """exp(-n/h) -- the flat term itself, coefficient 1."""
    return sector(n, R(1))


def ts_exp(x):
    """exp of a Composite, landing in whatever sector its infinite part names.

    Splits x into its infinite part, its standard part and its infinitesimal
    part.  The infinitesimal part exponentiates to an ordinary series, the
    standard part to a scalar, and the infinite part is what selects the
    sector -- which is the step the powers-and-logs group cannot take, because
    exp of something infinite is not in it.

    Only level one is supported: the infinite part must be k*h^-1 for an
    INTEGER k, giving sector -k.  A non-integer k means the action has not
    been normalised -- see the module docstring on units.
    """
    if isinstance(x, Transseries):
        if set(x.sectors) - {0}:
            raise NotImplementedError(
                "exp of a transseries with a non-zero sector is a level-two "
                "object (exp of exp); only level one is built.")
        x = x.sectors.get(0, R(0))
    x = x if isinstance(x, Composite) else Composite(x)

    inf, rest = {}, {}
    for d, v in x.coeffs_dict().items():
        if v == 0.0:
            continue
        (inf if _dim_is_positive(d) else rest)[d] = v
    if not inf:
        return Transseries({0: _c_exp(x)})
    if len(inf) > 1 or 1.0 not in {float(d) if not isinstance(d, tuple) else None
                                   for d in inf}:
        raise NotImplementedError(
            f"exp of an infinite part {inf!r}: only k*h^-1 is supported "
            f"(level one).  Anything faster than 1/h needs a further level.")
    k = list(inf.values())[0]
    if abs(k - round(k)) > 0:
        raise ValueError(
            f"exp({k}/h) would sit at sector {-k}, and a non-integer sector "
            f"cannot be added exactly -- 3*{k} is not representable, and "
            f"inexact grade addition is refused.  Measure the exponent in "
            f"units of the action so the sectors are integers.")
    body = Composite({d: v for d, v in rest.items()})
    return Transseries({-int(round(k)): _c_exp(body) if rest else R(1)})


def _d_composite(c):
    """d/dh of a Composite.  Grade d is h**-d, so d/dh sends it to grade d+1
    with the coefficient multiplied by -d; the constant term dies.

    R1 FIRST, as for every other operation.  Reading the coefficients raw made
    d/dh ill-defined on equal operands: |0|_0 and ZERO are the same number --
    each converts under R1, so they are operationally indistinguishable -- yet
    raw reading gave d/dh(|0|_0) = NOTHING and d/dh(ZERO) = |1|_0.  Equal in,
    unequal out; substitution broken.
    """
    c = _apply_r1(c)
    out = {}
    for d, v in c.coeffs_dict().items():
        if v == 0.0:
            continue
        if isinstance(d, tuple):
            raise NotImplementedError(
                f"d/dh of a log-axis dimension {d!r} produces a 1/(h ln(1/h)) "
                f"term, which needs the log axis in the derivative too.")
        dd = float(d)
        if dd == 0.0:
            continue
        out[dd + 1.0] = out.get(dd + 1.0, 0.0) + (-dd) * v
    return Composite(out)


def ts_d(x):
    """d/dh, and it CLOSES on the exponential level -- which is the point.

        d/dh [ C(h) exp(-n/h) ] = [ C'(h) + C(h)*n/h^2 ] exp(-n/h)

    The sector is unchanged; only its series moves.  A container with an extra
    symbol bolted on would leak out of the level here.  This is what makes the
    value group an H-field rather than a graded box: it is closed under
    derivation, which is the property an ODE needs.
    """
    x = Transseries.lift(x)
    out = {}
    for n, c in x.sectors.items():
        term = _d_composite(c)
        if n != 0:
            boost = c * Composite({2.0: float(n)})      # C(h) * n/h^2
            term = boost if _is_zero(term) else term + boost
        out[n] = term
    return Transseries(out)


def ts_ln(x):
    """ln, which moves a sector back onto the power axis:

        ln[ C(h) exp(-n/h) ] = ln C(h) - n/h

    the inverse of ts_exp, and the round trip is exact.
    """
    x = Transseries.lift(x)
    if len(x.sectors) != 1:
        raise NotImplementedError(
            f"ln of a {len(x.sectors)}-sector transseries needs the dominant "
            f"term factored out first -- ln(L(1+u)) = ln L + ln(1+u) -- which "
            f"is not built.")
    n, c = next(iter(x.sectors.items()))
    body = _c_ln(c)
    if n:
        lin = Composite({1.0: -float(n)})               # -n/h
        body = lin if _is_zero(body) else body + lin
    return Transseries({0: body})


def is_infinite(x):
    """Beyond every power upward: a negative sector, or an infinite sector 0.

    R1 FIRST, as for every other operation -- a wholly zero operand converts,
    and what it converts to is an infinitesimal, not nothing.  Reading the
    sectors raw reported a zero as neither infinite nor infinitesimal, which
    is the same defect the substitution test caught in d/dh.
    """
    x = Transseries.lift(x)._r1()
    n = x.leading_sector()
    if n is None:
        return False
    if n != 0:
        return n < 0
    d = _lead_dim(x.sectors[0])
    return d is not None and _dim_is_positive(d)


def is_infinitesimal(x):
    """Beyond every power downward: a positive sector, or an infinitesimal
    sector 0.  A sector outranks its own prefactor -- exp(-1/h)*h^-1000 is
    infinitesimal although h^-1000 is infinite, and that is the claim.

    R1 first, for the reason given in is_infinite.
    """
    x = Transseries.lift(x)._r1()
    n = x.leading_sector()
    if n is None:
        return False                      # NOTHING is not an infinitesimal
    if n != 0:
        return n > 0
    d = _lead_dim(x.sectors[0])
    if d is None:
        return False
    return not _dim_is_positive(d) and not (
        d == 0 or (isinstance(d, tuple) and all(c == 0 for c in d)))


def ts_st(x):
    """Standard part.  Refuses when there is none -- as it must for exp(1/h).

    Sectors above 0 are beyond every power downward and contribute nothing;
    sector 0 answers, through the library's own st().
    """
    x = Transseries.lift(x)
    x = x._r1()
    if is_infinite(x):
        raise ValueError(
            f"{x!r} has no standard part: it is beyond every power upward. "
            f"A sector below 0 is exp(+n/h), which outgrows 1/h**k for every "
            f"k, so no real number stands at its dimension.")
    c = x.sectors.get(0)
    return 0.0 if c is None else c.st()
