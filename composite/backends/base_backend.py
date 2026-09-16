# composite/backends/base_backend.py
# Composite Machine — Backend ABC
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

from abc import ABC, abstractmethod
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


def dim_cast(d):
    """A single dimension at Python level: int when integral, else float."""
    f = float(d)
    return int(f) if f.is_integer() else f


class CompositeBackend(ABC):
    """Abstract base class for Composite arithmetic backends.

    All backends store a Composite as an ordered collection of
    (dimension, value) pairs. Only dimensions explicitly created
    by computation exist. No gaps are ever filled.
    """

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
