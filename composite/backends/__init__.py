# composite/backends/__init__.py
from .config import get_backend, set_backend, use_sparse_dense, use_dict
from .base_backend import CompositeBackend
from .sparse_dense_backend import SparseDenseBackend, SparseData
from .dict_backend import DictBackend, DictData
