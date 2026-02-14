# composite/backends/base_backend.py
# Composite Machine — Backend ABC
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


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

    # --- arithmetic ---
    @abstractmethod
    def add(self, a: object, b: object) -> object:
        """Composite addition: merge terms, sum where dims match."""

    @abstractmethod
    def convolve(self, a: object, b: object) -> object:
        """Composite multiplication via convolution."""

    @abstractmethod
    def deconvolve(self, a: object, b: object) -> object:
        """Composite division via deconvolution."""

    @abstractmethod
    def scalar_multiply(self, data: object, scalar: float) -> object:
        """Multiply all coefficients by a scalar."""

    @abstractmethod
    def negate(self, data: object) -> object:
        """Negate all coefficients."""
