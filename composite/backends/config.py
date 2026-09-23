# composite/backends/config.py
# Composite Machine — Backend Configuration
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import sys as _sys

from .sparse_dense_backend import SparseDenseBackend
from .dict_backend import DictBackend
from .dense_series_backend import DenseSeriesBackend

# Default: Clustered Sparse-Dense (NumPy)
_active_backend = SparseDenseBackend(gap_threshold=64)

def get_backend():
    return _active_backend

def set_backend(backend):
    global _active_backend
    # The order cap is the CALLER's setting, not the backend's: it says how many
    # orders they want back, which has nothing to do with how the coefficients
    # are stored.  It lived on the backend instance, so switching storage in the
    # middle of a computation silently dropped it and the work went back to full
    # depth -- the knob appearing to fail again, for a different reason.
    carried = getattr(_active_backend, "max_order", None)
    if carried is not None and getattr(backend, "max_order", None) is None:
        backend.max_order = carried
    _active_backend = backend
    # ZERO / INF / h are module constants built at import time; rebuild them so
    # they follow the active backend.  Imported lazily to avoid a cycle, and
    # skipped if composite_lib has not been imported yet.
    _cl = _sys.modules.get("composite.composite_lib")
    if _cl is not None:
        _cl._refresh_constants()

def use_sparse_dense(gap_threshold=64, zero_tol=0.0, allow_fft=False):
    """allow_fft=True trades exactness (~1e-13) for speed on large operands."""
    set_backend(SparseDenseBackend(gap_threshold=gap_threshold,
                                    zero_tol=zero_tol, allow_fft=allow_fft))

def use_dict():
    set_backend(DictBackend())


def use_dense_series(max_span=1 << 20):
    """Contiguous-array backend: the right one for CALCULUS, wrong for grids.

    A Taylor series has no gaps, so run/lattice bookkeeping is pure overhead --
    measured at ~85% of an exp(-(x*x)) evaluation.  Anything genuinely sparse
    (a PDE front over a large domain) must stay on the sparse-dense backend;
    max_span makes the misuse fail loudly instead of allocating the domain.
    """
    set_backend(DenseSeriesBackend(max_span=max_span))
