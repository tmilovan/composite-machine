# composite/backends/config.py
# Composite Machine — Backend Configuration
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

import sys as _sys

from .sparse_dense_backend import SparseDenseBackend
from .dict_backend import DictBackend
from .dense_series_backend import DenseSeriesBackend
from .fractional_backend import (FractionalDictBackend,
                                 FractionalNumpyBackend,
                                 FractionalTorchBackend)

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


def use_fractional_dict():
    """Dimensions on an exact rational lattice, pure Python.

    The reference flavour.  Any fractional grade works, including the
    denominators float64 has to refuse: 1/3, 1/7, 1/23.  Canonical lattice is 1
    when no fractional order is present, so integer-grade work is stored exactly
    as it is elsewhere.
    """
    set_backend(FractionalDictBackend())


def use_fractional_numpy():
    """The same lattice storage with a vectorised convolve.

    Picks one of three aggregation kernels from the data: plain convolve when
    the key span is about 2n (which is the case whenever the lattice is 1),
    bincount on shifted keys for a moderate span, and unique for a span that
    would make a bin array unreasonable.
    """
    set_backend(FractionalNumpyBackend())


def use_fractional_torch(device="cpu", min_pairs=1 << 20, allow_float32=False):
    """The lattice storage with torch doing the aggregation.

    Delegates products smaller than `min_pairs` to the numpy flavour.  Measured
    on this machine torch only breaks even near a million pairs and MPS never
    wins, so the default threshold is a million; the flavour is here for a CUDA
    device, which is untested.
    device="mps" has no float64 and so needs allow_float32=True; the lattice
    keys stay int64 and exact regardless of device.
    """
    set_backend(FractionalTorchBackend(device=device, min_pairs=min_pairs,
                                       allow_float32=allow_float32))
