# composite/backends/config.py
# Composite Machine — Backend Configuration
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0

from .sparse_dense_backend import SparseDenseBackend
from .dict_backend import DictBackend

# Default: Clustered Sparse-Dense (NumPy)
_active_backend = SparseDenseBackend(gap_threshold=64)

def get_backend():
    return _active_backend

def set_backend(backend):
    global _active_backend
    _active_backend = backend

def use_sparse_dense(gap_threshold=64, zero_tol=0.0):
    set_backend(SparseDenseBackend(gap_threshold=gap_threshold,
                                    zero_tol=zero_tol))

def use_dict():
    set_backend(DictBackend())
