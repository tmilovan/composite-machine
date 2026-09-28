# composite/backends/fractional_backend.py
# Composite Machine - Fractional (lattice) Backends
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Dimensions on an exact rational lattice, so any fractional grade works.

WHY.  Dimensions are stored as float64 everywhere else, and a dimension is an
IDENTIFIER, not a magnitude: two terms merge only if their dimensions are equal.
1/3 is not representable in binary, so h**(1/3) * h**(1/3) * h**(1/3) lands one
ulp from h and would never merge with it.  `_add_exact` catches that and raises
InexactGradeError rather than getting it wrong, which is correct and leaves every
non-dyadic exponent unreachable: only denominators that are powers of two work.
Measured: 1/2, 1/4, 3/4, 1/8, 5/8 all fine; 1/3, 2/3, 1/5, 1/6, 1/7, 1/10 all
refused on the first multiply.

WHAT.  Store the power axis in units of 1/L for a per-composite integer L, so
every dimension is an integer key again and 1/3 is the key L/3.  Grade arithmetic
becomes integer arithmetic, which is exact by construction, and the refusal
disappears.

L IS NOT METADATA ON THE NUMBER.  It is kept canonical -- the lcm of the
denominators of the orders actually present -- so it is a function of the value
and not a free choice.  Two composites holding the same orders reduce to the
same (L, keys) pair, and a result cannot depend on which lattice its inputs
happened to be stored on.  Verified over the arithmetic: x on L=3 times y and
the same x on L=30 times y give identical output.  L therefore lives here in the
backend's data, exactly where SparseDenseBackend keeps its run offsets, and is
invisible above the backend boundary.

AND IT COSTS NOTHING WHEN UNUSED.  Canonical L is exactly 1 whenever no
fractional order is present, so an ordinary Taylor series is stored with the
keys it has today.  Measured: a 200x200-term multiply takes the same time at
L = 1 and at L = 6469693230 (33 bits), because Python integer arithmetic and
hashing do not care about the magnitude.  What mixing denominators does cost is
TERM COUNT, since h**(1/2) and h**(1/3) are genuinely different orders and their
products no longer collide.  Below a given order the ceiling is cap * L.

Three flavours share this storage:
  FractionalDictBackend   pure Python, the reference the others are checked against
  FractionalNumpyBackend  vectorised convolve, three aggregation kernels
  FractionalTorchBackend  the same in torch, for large products and for a GPU
"""
import math
from fractions import Fraction as F
from typing import Tuple

import numpy as np

from .base_backend import (CompositeBackend, DIM_DTYPE, dim_cast,
                           dim_fraction)


# --- dimensions ----------------------------------------------------------

def as_fraction(d):
    """A public dimension as an exact Fraction.  See base_backend.dim_fraction.

    One implementation, in base_backend, because __truediv__ needs the identical
    conversion: mixing a Fraction dimension with a float one subtracts in float
    and lands an ulp out, which is a wrong grade rather than an imprecise one.
    Two copies of that rule would be two chances to disagree.
    """
    return dim_fraction(d)


def _public_dim(key, L):
    """Storage key back to a public dimension: int when integral, else Fraction.

    Mirrors dim_cast's contract, which hands integral dimensions back as ints so
    every existing caller sees what it saw before.

    L == 1 short-circuits: the key IS the dimension, so building a Fraction only
    to reduce it away again is pure waste, and L is 1 for every composite that
    has no fractional grade -- which is most of them.
    """
    if isinstance(key, tuple):
        head = _public_dim(key[0], L)
        return (head,) + tuple(int(c) for c in key[1:])
    if L == 1:
        return int(key)
    fr = F(int(key), L)
    return int(fr) if fr.denominator == 1 else fr


def _integer_keys(dims):
    """The keys on lattice 1 when every dimension is integral, else None.

    One scan to avoid the Fraction machinery entirely in the common case.  It
    was costing 20050 dim_fraction calls and 30100 Fraction constructions per
    50 divisions of a 200-term composite, all to rediscover that the dimensions
    were integers.
    """
    out = []
    for d in dims:
        if isinstance(d, tuple):
            return None
        if isinstance(d, F):
            if d.denominator != 1:
                return None
            out.append(int(d))
        else:
            f = float(d)
            if not f.is_integer():
                return None
            out.append(int(f))
    return out


def _lattice_of(dims):
    """The lcm of the denominators of these public dimensions."""
    L = 1
    for d in dims:
        head = d[0] if isinstance(d, tuple) else d
        L = math.lcm(L, as_fraction(head).denominator)
    return L


def _to_key(d, L):
    """Public dimension to a storage key on lattice L, or None if it is off it."""
    if isinstance(d, tuple):
        head = _to_key(d[0], L)
        if head is None:
            return None
        return (head,) + tuple(int(c) for c in d[1:])
    fr = as_fraction(d)
    num, den = fr.numerator, fr.denominator
    if L % den:
        return None
    return num * (L // den)


def _canon(terms, L):
    """Reduce to the smallest lattice holding these keys.

    This is what makes L a function of the value rather than a choice.  The gcd
    runs over the POWER component only; log-axis components are integer-graded
    and never scaled.
    """
    if not terms:
        return {}, 1
    g = L
    for k in terms:
        g = math.gcd(g, abs(k[0] if isinstance(k, tuple) else k))
        if g == 1:
            return terms, L
    if g <= 1:
        return terms, L
    out = {}
    for k, v in terms.items():
        nk = ((k[0] // g,) + k[1:]) if isinstance(k, tuple) else k // g
        out[nk] = v
    return out, L // g


def _scale_keys(terms, s):
    if s == 1:
        return terms
    out = {}
    for k, v in terms.items():
        nk = ((k[0] * s,) + k[1:]) if isinstance(k, tuple) else k * s
        out[nk] = v
    return out


def _key_add(ka, kb):
    """Add two storage keys.  The power components are on a common lattice.

    A scalar key is the vector (key, 0, 0, ...), so the shorter side is padded
    and the result trimmed back to a scalar when only the power survives, which
    keeps scalar-only arithmetic on exactly the keys it had.
    """
    ta, tb = isinstance(ka, tuple), isinstance(kb, tuple)
    if not ta and not tb:
        return ka + kb
    va = ka if ta else (ka,)
    vb = kb if tb else (kb,)
    n = max(len(va), len(vb))
    out = tuple((va[i] if i < len(va) else 0) + (vb[i] if i < len(vb) else 0)
                for i in range(n))
    while len(out) > 1 and out[-1] == 0:
        out = out[:-1]
    return out[0] if len(out) == 1 else out


def _key_sub(ka, kb):
    return _key_add(ka, _key_neg(kb))


def _key_neg(k):
    return tuple(-c for c in k) if isinstance(k, tuple) else -k


def _key_sort(k):
    """Total order over mixed scalar and vector keys, lexicographic.

    sorted() on a mixed list raises, and a scalar key IS the vector (key, 0,
    ...), so padding makes the order total and agrees with the dominance order.
    """
    return k if isinstance(k, tuple) else (k,)


class FracData:
    """Storage: {int-or-tuple key: float} plus the lattice L.  Key k is order k/L."""
    __slots__ = ('terms', 'L')

    def __init__(self, terms, L=1, canon=True):
        # A lattice of zero or less is not a lattice.  Nothing reachable through
        # the public API produces one -- _lattice_of takes the lcm of Fraction
        # denominators, which are 1 or more -- but a direct FracData(..., L=0)
        # was accepted and then died in __repr__ with a bare
        # "ZeroDivisionError: Fraction(0, 0)", a page away from the cause.
        if not isinstance(L, int) or L < 1:
            raise ValueError(
                f"lattice must be a positive integer, got {L!r}. Key k denotes "
                f"the order k/L, so L is the denominator of the dimension axis "
                f"and there is no order k/0.")
        if canon:
            terms, L = _canon(terms, L)
        self.terms = terms
        self.L = L

    def __repr__(self):
        parts = [f"|{v}|_{_public_dim(k, self.L)}"
                 for k, v in sorted(self.terms.items(), key=lambda kv: _key_sort(kv[0]))]
        return "Composite(" + " + ".join(parts) + f")  [L={self.L}]"


# --- the reference flavour ------------------------------------------------

class FractionalDictBackend(CompositeBackend):
    """Pure-Python lattice backend.  The reference the other flavours match."""

    VECTOR_DIMS = False          # tuple keys are supported but not required
    EXACT_DIMS = True            # a Fraction dimension is held exactly

    # ---- construction
    def create(self, dim, value) -> FracData:
        L = _lattice_of([dim])
        return FracData({_to_key(dim, L): float(value)}, L)

    def create_from_terms(self, dims, vals) -> FracData:
        dims = list(dims)
        if not dims:
            return FracData({}, 1)
        # Integral dimensions mean lattice 1 and keys that ARE the dimensions.
        # canon=False is safe because gcd(1, anything) is 1, so the reduction
        # pass could not change what this builds.
        fast = _integer_keys(dims)
        if fast is not None:
            terms = {}
            for k, v in zip(fast, vals):
                terms[k] = terms.get(k, 0.0) + float(v)
            return FracData(terms, 1, canon=False)
        L = _lattice_of(dims)
        terms = {}
        for d, v in zip(dims, vals):
            k = _to_key(d, L)
            terms[k] = terms.get(k, 0.0) + float(v)
        return FracData(terms, L)

    # ---- reads
    def read_dim(self, data: FracData, dim) -> float:
        k = _to_key(dim, data.L)
        if k is None:
            return 0.0              # off this lattice, so no such term
        return data.terms.get(k, 0.0)

    def write_dim(self, data: FracData, dim, value) -> FracData:
        """Zeros are written, not popped: an expressed zero is a term (R2)."""
        fr = as_fraction(dim[0] if isinstance(dim, tuple) else dim)
        L = math.lcm(data.L, fr.denominator)
        terms = _scale_keys(data.terms, L // data.L)
        terms = dict(terms)
        terms[_to_key(dim, L)] = float(value)
        return FracData(terms, L)

    def term_count(self, data: FracData) -> int:
        return len(data.terms)

    def is_wholly_zero(self, data: FracData) -> bool:
        t = data.terms
        return len(t) > 0 and all(v == 0.0 for v in t.values())

    def is_unit(self, data: FracData) -> bool:
        t = data.terms
        if len(t) != 1:
            return False
        (k, v), = t.items()
        return k == 0 and v == 1.0

    def to_arrays(self, data: FracData) -> Tuple[np.ndarray, np.ndarray]:
        if not data.terms:
            return (np.array([], dtype=DIM_DTYPE),
                    np.array([], dtype=np.float64))
        items = sorted(data.terms.items(), key=lambda kv: _key_sort(kv[0]))
        pub = [_public_dim(k, data.L) for k, _ in items]
        vals = np.array([v for _, v in items], dtype=np.float64)
        # A Fraction or a tuple must stay whole, so an object array, exactly as
        # the dict backend does for vector dimensions.  When every dimension is
        # integral the float64 path is kept, which is what the rest of the
        # library already handles.
        if any(isinstance(d, (F, tuple)) for d in pub):
            dims = np.empty(len(pub), dtype=object)
            for i, d in enumerate(pub):
                dims[i] = d
            return dims, vals
        return np.array(pub, dtype=DIM_DTYPE), vals

    def active_dims(self, data: FracData) -> np.ndarray:
        return self.to_arrays(data)[0]

    # ---- alignment
    @staticmethod
    def _align(a: FracData, b: FracData):
        L = math.lcm(a.L, b.L)
        return (L,
                _scale_keys(a.terms, L // a.L),
                _scale_keys(b.terms, L // b.L))

    # ---- arithmetic
    def add(self, a: FracData, b: FracData) -> FracData:
        """A zero-sum dimension is RETAINED: 1 - 1 is |0|_0, not absence."""
        L, ta, tb = self._align(a, b)
        out = dict(ta)
        for k, v in tb.items():
            out[k] = out.get(k, 0.0) + v
        return FracData(out, L)

    def convolve(self, a: FracData, b: FracData) -> FracData:
        """Every constructed dimension is retained, zero coefficients included.

        No InexactGradeError is possible here: the keys are integers, so the
        Minkowski sum is exact whatever the denominators were.
        """
        L, ta, tb = self._align(a, b)
        out = {}
        for ka, va in ta.items():
            for kb, vb in tb.items():
                k = _key_add(ka, kb)
                out[k] = out.get(k, 0.0) + va * vb
        return FracData(out, L)

    def deconvolve(self, a: FracData, b: FracData) -> FracData:
        L, ta, tb = self._align(a, b)
        if not tb:
            raise ZeroDivisionError("Cannot deconvolve by empty Composite")
        rem = {k: v for k, v in ta.items() if v != 0.0}
        bnz = {k: v for k, v in tb.items() if v != 0.0}
        if not bnz:
            raise ZeroDivisionError("Cannot deconvolve by zero Composite")
        lead_k = max(bnz, key=_key_sort)
        lead_v = bnz[lead_k]
        quot = {}
        for _ in range(max(len(ta) + len(tb), 50)):
            if not rem:
                break
            rk = max(rem, key=_key_sort)
            qk = _key_sub(rk, lead_k)
            qv = rem[rk] / lead_v
            quot[qk] = qv
            for kb, vb in bnz.items():
                ok = _key_add(qk, kb)
                rem[ok] = rem.get(ok, 0.0) - qv * vb
                if rem[ok] == 0.0:
                    del rem[ok]
            # qv cancels rk exactly by construction; float dust there would let
            # the loop reselect rk and overwrite a correct quotient coefficient.
            rem.pop(rk, None)
        return FracData(quot, L)

    def scalar_multiply(self, data: FracData, scalar: float) -> FracData:
        if scalar == 0.0:
            return FracData({}, 1)
        return FracData({k: v * scalar for k, v in data.terms.items()},
                        data.L, canon=False)

    def negate(self, data: FracData) -> FracData:
        return FracData({k: -v for k, v in data.terms.items()},
                        data.L, canon=False)


# --- the vectorised flavour ----------------------------------------------

class FractionalNumpyBackend(FractionalDictBackend):
    """Same lattice storage, vectorised convolve.

    A product is an outer SUM of two integer key arrays and an outer PRODUCT of
    the coefficients, then one aggregation.  No dense layout is needed anywhere,
    so the cost does not depend on the lattice at all -- measured flat from
    L = 1 to L = 30030 where a padded array would be 12 million long.

    Three aggregation kernels, chosen from the data:

      span ~ 2n        np.convolve on the dense range.  Unbeatable when L is 1,
                       0.01 ms against 1.07 for the sort at n = 200, and L IS 1
                       whenever no fractional order is present.
      span moderate    np.bincount on shifted keys.  No sort.  0.71 ms at
                       n = 200, L = 210, where convolve collapses to 2372 ms
                       because it is quadratic in the padded length.
      span large       np.unique then bincount.  The only one indifferent to the
                       span: 1.21 ms at L = 30030 where bincount pays 73 ms
                       allocating a 12-million-element bin array.

    bincount costs O(n^2 + span) and unique O(n^2 log n^2), so bincount wins
    while span < n^2 log n^2.  At n = 200 that threshold is about 600k and the
    measured crossover sits between 83k and 12M, as predicted.
    """

    DENSE_SPAN_FACTOR = 4        # span <= factor * terms -> plain convolve

    def convolve(self, a: FracData, b: FracData) -> FracData:
        L, ta, tb = self._align(a, b)
        if not ta or not tb:
            return FracData({}, 1) if not (ta or tb) else FracData({}, L)
        # A tuple key means log-axis content, which is not a 1-D convolution.
        if any(isinstance(k, tuple) for k in ta) or \
           any(isinstance(k, tuple) for k in tb):
            return FractionalDictBackend.convolve(self, a, b)

        ka = np.fromiter(ta.keys(), dtype=np.int64, count=len(ta))
        va = np.fromiter(ta.values(), dtype=np.float64, count=len(ta))
        kb = np.fromiter(tb.keys(), dtype=np.int64, count=len(tb))
        vb = np.fromiter(tb.values(), dtype=np.float64, count=len(tb))

        lo = int(ka.min() + kb.min())
        hi = int(ka.max() + kb.max())
        span = hi - lo + 1
        n_pairs = ka.size * kb.size

        if span <= self.DENSE_SPAN_FACTOR * (ka.size + kb.size):
            keys, vals = self._by_convolve(ka, va, kb, vb)
        elif span <= max(n_pairs * 16, 1 << 16):
            keys, vals = self._by_bincount(ka, va, kb, vb, lo, span)
        else:
            keys, vals = self._by_unique(ka, va, kb, vb)
        return FracData(dict(zip(keys.tolist(), vals.tolist())), L)

    # Each kernel returns every constructed key, zero coefficients included,
    # because a dimension exists because the computation built it.
    @staticmethod
    def _by_convolve(ka, va, kb, vb):
        """Dense convolution, then filtered back to the Minkowski sum.

        The filter is not optional.  Padding the key range and returning every
        position invents dimensions the product never constructed, and an
        invented zero here is an EXPRESSED zero, which is an infinitesimal.  A
        fuzz over 400 random fractional products caught it in 215 of them.

        Presence is convolved separately from the coefficients, because a term
        can legitimately hold 0.0 and still exist: the indicator has to say
        "this key is present", never "this coefficient is non-zero".
        """
        amin, bmin = int(ka.min()), int(kb.min())
        na = int(ka.max()) - amin + 1
        nb = int(kb.max()) - bmin + 1
        A = np.zeros(na); B = np.zeros(nb)
        A[ka - amin] = va
        B[kb - bmin] = vb
        IA = np.zeros(na); IB = np.zeros(nb)
        IA[ka - amin] = 1.0
        IB[kb - bmin] = 1.0
        out = np.convolve(A, B)
        present = np.convolve(IA, IB) > 0.5
        idx = np.flatnonzero(present)
        return idx + (amin + bmin), out[idx]

    @staticmethod
    def _by_bincount(ka, va, kb, vb, lo, span):
        K = (ka[:, None] + kb[None, :]).ravel() - lo
        V = (va[:, None] * vb[None, :]).ravel()
        acc = np.bincount(K, weights=V, minlength=span)
        seen = np.bincount(K, minlength=span) > 0
        idx = np.flatnonzero(seen)
        return idx + lo, acc[idx]

    @staticmethod
    def _by_unique(ka, va, kb, vb):
        K = (ka[:, None] + kb[None, :]).ravel()
        V = (va[:, None] * vb[None, :]).ravel()
        uk, inv = np.unique(K, return_inverse=True)
        return uk, np.bincount(inv, weights=V, minlength=uk.size)


# --- the torch flavour ----------------------------------------------------

class FractionalTorchBackend(FractionalNumpyBackend):
    """The same lattice storage with the aggregation done in torch.

    MEASURED ON THIS MACHINE IT DOES NOT PAY OFF, and the default threshold says
    so.  On a scattered lattice where the pair table is the right algorithm for
    everyone, numpy against torch-cpu against torch-mps:

        n=200,   40k pairs    0.89 ms   2.72 ms   17.12 ms
        n=500,  250k pairs    3.71 ms   8.03 ms   21.69 ms
        n=1000,   1M pairs   16.73 ms  15.72 ms   25.20 ms
        n=2000,   4M pairs   32.82 ms  33.56 ms   36.38 ms

    So it breaks even near a million pairs on CPU and MPS never wins, which is
    why min_pairs defaults above that: below it, delegating to numpy is simply
    faster.  What is untested is a real CUDA device, where the pair table can
    stay resident and the arithmetic is an order of magnitude wider; there is no
    CUDA on this machine and I am not going to guess the number.

    The dense-span kernel always delegates.  When the keys are contiguous, which
    is every composite whose lattice is 1, the answer is a convolution over two
    length-n arrays, and building an n*m pair table instead costs 256 MB at
    n = 4000 and measured 40.7 ms against numpy's 5.0 ms.

    PRECISION.  MPS has no float64.  Asking for device="mps" therefore drops the
    COEFFICIENTS to float32, which is a real loss and is refused unless
    allow_float32=True is passed explicitly.  The DIMENSIONS are unaffected
    either way: they are int64 keys on the lattice, so the grade arithmetic that
    this whole backend exists to make exact stays exact on any device.  A
    coefficient losing digits is an accuracy question; a grade losing digits
    changes which terms merge, and that is the one thing we do not trade.
    """

    def __init__(self, device="cpu", min_pairs=1 << 20, allow_float32=False):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.min_pairs = min_pairs
        if self.device.type == "mps":
            if not allow_float32:
                raise ValueError(
                    "device='mps' has no float64, so coefficients would silently "
                    "drop to float32.  Pass allow_float32=True to accept that; "
                    "the lattice keys stay int64 and exact either way.")
            self.vdtype = torch.float32
        else:
            self.vdtype = torch.float64

    def convolve(self, a: FracData, b: FracData) -> FracData:
        torch = self._torch
        L, ta, tb = self._align(a, b)
        if not ta or not tb:
            return FracData({}, 1) if not (ta or tb) else FracData({}, L)
        if any(isinstance(k, tuple) for k in ta) or \
           any(isinstance(k, tuple) for k in tb):
            return FractionalDictBackend.convolve(self, a, b)
        if len(ta) * len(tb) < self.min_pairs:
            return FractionalNumpyBackend.convolve(self, a, b)

        # A DENSE SPAN IS A CONVOLUTION, NOT A PAIR TABLE.  When the keys are
        # contiguous -- which is every composite whose lattice is 1, so most of
        # them -- the answer is np.convolve over two length-n arrays.  Building
        # the n*m pair table instead costs 256 MB at n = 4000 and measured
        # 40.7 ms against numpy's 5.0 ms.  Delegate rather than lose.
        _lo = min(ta) + min(tb)
        _hi = max(ta) + max(tb)
        if (_hi - _lo + 1) <= self.DENSE_SPAN_FACTOR * (len(ta) + len(tb)):
            return FractionalNumpyBackend.convolve(self, a, b)

        ka = torch.tensor(list(ta.keys()), dtype=torch.int64, device=self.device)
        va = torch.tensor(list(ta.values()), dtype=self.vdtype, device=self.device)
        kb = torch.tensor(list(tb.keys()), dtype=torch.int64, device=self.device)
        vb = torch.tensor(list(tb.values()), dtype=self.vdtype, device=self.device)

        lo = int(ka.min() + kb.min())
        hi = int(ka.max() + kb.max())
        span = hi - lo + 1
        n_pairs = ka.numel() * kb.numel()

        K = (ka.unsqueeze(1) + kb.unsqueeze(0)).reshape(-1)
        V = (va.unsqueeze(1) * vb.unsqueeze(0)).reshape(-1)

        if span <= self.DENSE_SPAN_FACTOR * (ka.numel() + kb.numel()) or \
           span <= max(n_pairs * 16, 1 << 16):
            # bincount on shifted keys.  Presence is counted separately from the
            # weights, because a term may legitimately hold 0.0 and still exist.
            idxs = K - lo
            acc = torch.bincount(idxs, weights=V, minlength=span)
            seen = torch.bincount(idxs, minlength=span) > 0
            where = torch.nonzero(seen, as_tuple=True)[0]
            keys = (where + lo).tolist()
            vals = acc[where].tolist()
        else:
            uk, inv = torch.unique(K, return_inverse=True)
            acc = torch.zeros(uk.numel(), dtype=V.dtype, device=self.device)
            acc.scatter_add_(0, inv, V)
            keys = uk.tolist()
            vals = acc.tolist()
        return FracData(dict(zip(keys, vals)), L)
