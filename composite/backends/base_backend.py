# composite/backends/base_backend.py
# Composite Machine — Backend ABC
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

from abc import ABC, abstractmethod
from fractions import Fraction as _Fraction
from typing import Tuple
import numpy as np


# --- dimension index type -------------------------------------------------
# Dimensions are stored as float64 so that fractional indices (e.g. -1/2, which
# sqrt of an odd dimension produces) are representable.  Integer-valued
# dimensions are UNCHANGED by this: they are exactly representable in float64,
# they keep unit gaps so the run representation is untouched, and dim_cast()
# hands them back as Python ints so every existing caller sees what it saw
# before.  Only a genuinely fractional dimension surfaces as a float.
DIM_DTYPE = np.float64


class InexactGradeError(ArithmeticError):
    """A product asked for a grade float64 cannot hold exactly.

    Multiplication ADDS grades, so a grade is an identifier, not a magnitude.
    A relative error of 1e-17 in a coefficient is invisible; the same error in
    a grade makes two terms that should merge into two that never will --
    there is no tolerance in a dict lookup.  h**0.1 * h**0.2 lands on
    -0.30000000000000004 while h**0.3 is -0.3, one ulp apart and permanently
    distinct, and the sparse-dense backend then merges such pairs while the
    dict backend keeps them, so the same two grades are one term or two
    depending on how they were built.

    Integer and dyadic grades are never affected -- float64 holds every
    integer to 2**53 and every binary fraction exactly, and so does their sum.
    Measured over 200,000 random pairs of each: zero inexact additions.  Only
    arbitrary fractional exponents reach this, at about 25% of random pairs,
    and nothing in derivatives, series, PDE stencils or quadrature goes near
    them.
    """


FRACTIONAL_HINT = (
    " A backend that CAN hold this grade exists: config.use_fractional_numpy() "
    "(or use_fractional_dict / use_fractional_torch) stores the power axis on a "
    "rational lattice, so thirds, sevenths and the rest are exact integers "
    "internally and merge as they should. It is not the default because the "
    "lattice only helps when a fractional grade is actually present, and float64 "
    "is faster when none is.")


def inexact_grade(a, b, got):
    """The InexactGradeError for a grade sum float64 could not hold.

    One constructor for both raise sites, so the pointer to the fractional
    backend cannot be present in one message and missing from the other.  A
    capability nobody can find from the error that blocks them is not a
    capability.
    """
    return InexactGradeError(
        f"grade {a!r} + {b!r} is not exact in float64 (got {got!r}); the "
        f"product would carry an identifier one ulp from the one it should "
        f"share." + FRACTIONAL_HINT)


def _add_exact(a, b):
    """a + b, and whether float64 held it exactly (TwoSum).

    Checking the RESULT is not enough: -1/3 + -1/6 lands on exactly -0.5 and
    1/3+1/6+1/2 lands on exactly -1, through inexact steps whose errors happen
    to cancel.  Only the addition itself can be interrogated.
    """
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err == 0.0


def dim_cast(d):
    """A single dimension at Python level: int when integral, else float.

    A VECTOR dimension (a tuple over a declared basis) passes through whole --
    float() on a tuple raises, and there is nothing to normalise.

    A FRACTION passes through whole for the same reason one step further on:
    float(Fraction(1,3)) is 0.3333333333333333, which is a different identifier
    from a third, and the whole point of the fractional backend is that a third
    stays a third.  Integral fractions still come back as ints, so a caller who
    never touches a fractional grade sees exactly what it saw before.
    """
    if isinstance(d, tuple):
        return d
    if isinstance(d, _Fraction):
        return int(d) if d.denominator == 1 else d
    f = float(d)
    return int(f) if f.is_integer() else f


def dim_fraction(d):
    """A dimension as an exact Fraction.

    An int or a Fraction is exact already.  A float is the awkward one: the
    binary value 0.3333333333333333 IS exactly
    6004799503160661/18014398509481984, and taking that literally would put a
    lattice denominator in the quadrillions for a dimension the caller meant as
    a third.  So prefer the SIMPLEST fraction that maps back to the same
    float64, and fall back to the exact binary value when none does.  Integer
    and dyadic floats round-trip immediately and are unaffected.

    Lives here rather than in the fractional backend because __truediv__ needs
    it too: mixing a Fraction dimension with a float one subtracts in float and
    lands an ulp out, which is a wrong grade rather than an imprecise one.
    """
    if isinstance(d, _Fraction):
        return d
    if isinstance(d, (int, np.integer)):
        return _Fraction(int(d))
    f = float(d)
    if f.is_integer():
        return _Fraction(int(f))
    simple = _Fraction(f).limit_denominator(10 ** 6)
    return simple if float(simple) == f else _Fraction(f)


class CompositeBackend(ABC):
    """Abstract base class for Composite arithmetic backends.

    All backends store a Composite as an ordered collection of
    (dimension, value) pairs. Only dimensions explicitly created
    by computation exist. No gaps are ever filled.
    """
    # A backend whose dimensions are VECTORS over a declared basis can
    # represent a scalar dimension d as (d, 0, ...), but not the reverse.
    # _operands() uses this to convert toward the richer representation.
    VECTOR_DIMS = False

    # A backend that can hold a Fraction dimension EXACTLY.  The constructor
    # asks before passing one through: handing a Fraction to a float64 backend
    # gives it a mix of Fraction and float keys for the same grade, which never
    # merge, so the default stays float and only a backend that says yes sees
    # the exact value.
    EXACT_DIMS = False


    # --- lifecycle ---
    @abstractmethod
    def create(self, dim: int, value: float) -> object:
        """Create a single-term Composite: value at dimension dim."""

    @abstractmethod
    def create_from_terms(self, dims: np.ndarray, vals: np.ndarray) -> object:
        """Create a Composite from parallel arrays of dims and vals.
        Both arrays must be the same length. dims must be sorted."""

    # --- access ---
    @abstractmethod
    def read_dim(self, data: object, dim: int) -> float:
        """Read the coefficient at a specific dimension. Returns 0.0 if absent."""

    @abstractmethod
    def write_dim(self, data: object, dim: int, value: float) -> object:
        """Set the coefficient at a specific dimension. Returns new data."""

    @abstractmethod
    def to_arrays(self, data: object) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dims, vals) sorted arrays of all active terms."""

    @abstractmethod
    def active_dims(self, data: object) -> np.ndarray:
        """Return sorted array of all active dimension indices."""

    # --- structural predicates (concrete: backends may override) ---
    def term_count(self, data: object) -> int:
        """Number of active terms.  Override to avoid materialising flat form."""
        return len(self.active_dims(data))

    def is_wholly_zero(self, data: object) -> bool:
        """True when every expressed coefficient is zero.

        Default scans the flat form.  Backends that retain structure should
        override: this is called on EVERY multiply and division (Zero Rule R1
        converts a wholly-zero operand), so materialising flat arrays here was
        ~25% of multiply time once the backend itself got fast.
        """
        dims, vals = self.to_arrays(data)
        return len(dims) > 0 and not np.any(vals != 0.0)

    def is_unit(self, data: object) -> bool:
        """True for |1|_0, the multiplicative identity.  See is_wholly_zero."""
        dims, vals = self.to_arrays(data)
        return len(dims) == 1 and dims[0] == 0 and vals[0] == 1.0

    # --- arithmetic ---
    @abstractmethod
    def add(self, a: object, b: object) -> object:
        """Composite addition: merge terms, sum where dims match.

        The result carries the union of both dimension sets.  A dimension whose
        coefficients sum to zero is RETAINED with coefficient 0 -- it was
        constructed by the operands, so it exists.  Addition never shifts a
        dimension.
        """

    @abstractmethod
    def convolve(self, a: object, b: object) -> object:
        """Composite multiplication via convolution.

        The dimensions of the result are exactly the Minkowski sum of the two
        input dimension sets, {da + db}.  Zero-valued coefficients at those
        dimensions are RETAINED; dimensions outside the set must not appear,
        even if an implementation densifies gaps internally.
        """

    @abstractmethod
    def deconvolve(self, a: object, b: object) -> object:
        """Composite division via deconvolution.

        The leading term of the divisor is its highest dimension with a
        NONZERO coefficient: retained zeros may sit above it.  Zero-valued
        quotient terms are retained.
        """

    @abstractmethod
    def scalar_multiply(self, data: object, scalar: float) -> object:
        """Multiply all coefficients by a scalar."""

    @abstractmethod
    def negate(self, data: object) -> object:
        """Negate all coefficients."""
