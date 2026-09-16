# Sparse-Dense Backend Optimization — Specification

**Status:** draft, not implemented
**Target:** `composite/backends/sparse_dense_backend.py`
**Author of measurements:** session of 2026-09-14, all numbers reproducible with the harness in §8

---

## 1. Problem

Composite arithmetic spends roughly 90% of its time deciding how its own data is
laid out, and about 3.5% doing arithmetic.

Profile of 20,000 multiplies of two 20-term composites (both single contiguous
runs, dims −19..0), `cProfile`, sorted by `tottime`:

| share | function |
|---|---|
| 11.0% | `_cluster_terms` — 40k calls, twice per multiply, once per operand |
| 8.2% | `np.unique` (inside clustering) |
| 7.9% | `np.array_split` — 80k calls |
| 7.6% | `_mask_to_minkowski` |
| 7.5% | `_merge_cluster_outputs` |
| 5.0% | `np.diff` (inside clustering) |
| 4.1% | `composite_lib._is_wholly_zero` |
| **3.5%** | **`convolve` — the actual arithmetic** |

Cost breakdown of one `a*b` at n=20, measured independently:

```
current a*b                          65.35 us
  to_arrays x2   (flatten operands)   1.00 us   unnecessary
  np.convolve    (the work)           1.42 us   IRREDUCIBLE
  np.arange      (materialise dims)   1.00 us   unnecessary
  create_from_terms (re-flatten)      3.04 us   unnecessary
clusters as the representation        1.65 us   <- 40x, 86% efficient
```

Scaling of a prototype dense fast path against the current implementation,
bit-identical for n<=60 (above that the only difference is FFT rounding,
2.3e-12 relative, because the prototype used direct convolution where the
backend switches to FFT at `_FFT_THRESHOLD = 128`):

| terms | current | prototype | speedup |
|---|---|---|---|
| 20 | 65.8 us | 7.3 us | 9.0x |
| 60 | 99.7 us | 14.5 us | 6.9x |
| 200 | 704.6 us | 23.7 us | 29.7x |
| 1000 | 18805 us | 174 us | 107.9x |

The prototype only helps contiguous operands. The design below helps every
layout, because it removes the clustering round-trip rather than skipping it in
one special case.

## 2. Root cause

`SparseData` stores flat parallel arrays:

```python
class SparseData:
    __slots__ = ('dims', 'vals')      # int64[] sorted unique, float64[]
```

No cluster structure is retained. Every operation therefore performs
**flat -> cluster -> compute -> flat**, and `_cluster_terms` re-derives, on every
single operation, a property of the operands that never changed.

The backend is named *clustered* sparse-dense. The clustering is currently a
transient computation, not the representation.

Two further consequences visible in the numbers above:

- **Dimension indices are materialised even when implied.** For a contiguous run
  the `dims` array is `arange(offset, offset+len)` and carries no information.
  Building it cost 1.00 us of the 7.3 us prototype.
- **`create_from_terms` re-sorts on every result** (`np.argsort`), although
  convolution and addition both produce sorted output by construction.

## 3. Target design

### 3.1 Data structure

```python
class SparseData:
    __slots__ = ('clusters',)
    # clusters: List[Cluster], sorted by offset, non-overlapping
    # Cluster:  (offset: int, vals: np.ndarray[float64])
    #           represents dims offset .. offset+len(vals)-1, densely
```

- **one cluster** = a dense contiguous composite
- **many clusters** = a sparse composite
- `gap_threshold` (constructor, default 64) decides at *construction* whether two
  runs separated by a gap merge into one cluster (paying interior zeros to gain
  contiguity) or stay separate. This is the sparse/dense decision, made once.

### 3.2 Invariants

1. Clusters are sorted by `offset` and are **non-overlapping and non-adjacent**:
   for consecutive clusters, `offset[i] + len(vals[i]) + gap_threshold <= offset[i+1]`.
   Anything closer is merged at construction.
2. `vals` within a cluster may contain zeros (expressed zeros are meaningful —
   see Zero Rules; they must not be pruned silently).
3. A cluster is never empty (`len(vals) >= 1`).
4. The empty composite is `clusters == []`. This is distinct from a composite
   holding `|0|_0`, which is `[(0, array([0.0]))]`.
5. Leading and trailing zeros within a cluster are permitted but SHOULD be
   trimmed at construction *only when* trimming cannot change the meaning —
   i.e. never for a cluster of length 1, and never if it would empty a cluster.
   (Open question, §7.)

### 3.3 Structure propagation

The structure of a result is derivable from the operands and MUST NOT be
rediscovered by scanning:

| operation | resulting structure |
|---|---|
| dense x dense | **dense** — the Minkowski sum of two contiguous ranges is contiguous |
| dense x k clusters | k clusters, merged where they overlap after offsetting |
| j clusters x k clusters | at most j*k clusters, merged where they overlap |
| add | sorted merge of the two cluster lists; overlapping clusters sum elementwise |
| scalar_multiply, negate | structure unchanged, values scaled |
| write_dim | may split one cluster into two, or extend one, or add one |

## 4. Operation semantics

The public interface is fixed by `base_backend.CompositeBackend` and MUST NOT
change:

`create`, `create_from_terms`, `read_dim`, `write_dim`, `to_arrays`,
`active_dims`, `add`, `convolve`, `deconvolve`, `scalar_multiply`, `negate`.

### 4.1 `convolve(a, b)`

- Single cluster each: one `np.convolve` (or `_fft_convolve` above
  `_FFT_THRESHOLD`), result offset `a.offset + b.offset`. **No clustering, no
  merging, no dims array.**
- Otherwise: convolve each pair of clusters, offsetting by the sum of offsets;
  merge results whose ranges overlap. Merging is an elementwise add on the
  overlap, not a re-clustering pass.
- FFT-vs-direct selection MUST use the same rule as today
  (`len(a) + len(b) > _FFT_THRESHOLD`, currently 128) or results will differ in
  the last bits. See §6.

### 4.2 `add(a, b)`

Sorted merge of cluster lists. Clusters that overlap or come within
`gap_threshold` are combined into one, summing the overlap elementwise. No
`np.unique`, no `argsort`.

### 4.3 `to_arrays(data)`

Materialises flat `(dims, vals)` from clusters. This is now a **boundary
operation** for callers that need flat form; it MUST NOT be called inside
`add`, `convolve`, `scalar_multiply` or `negate`.

### 4.4 `read_dim(data, dim)`

Binary search over cluster offsets, then direct index. O(log k) rather than
O(log n) over a flat dims array — and O(1) for the single-cluster case.

### 4.5 `deconvolve(a, b)`

Polynomial long division, currently term-by-term from the highest dimension
down. This is the most intricate method and the one most likely to regress.
It MAY continue to operate on flat arrays internally (via `to_arrays` at entry
and `create_from_terms` at exit) in the first implementation, and be converted
later. Correctness first; it is not on the measured hot path.

## 5. Compatibility contract

The following MUST hold after the change:

1. **`DictBackend` is unchanged** and serves as the differential reference.
2. For every operation and every input, sparse-dense and dict backends agree to
   within FFT rounding (`<= 1e-10` relative), and **exactly** when both take the
   direct-convolution path.
3. The two existing truncation mechanisms keep their current behaviour and stay
   distinct:
   - `composite_lib.MAX_ACTIVE_DIMS = 60` via `_truncate_dims`, applied in
     `Composite.__mul__`, keeps the dims **closest to zero**.
   - `SparseDenseBackend._truncate` via `max_order` (default `None`), drops dims
     below `-max_order`.
   Neither is in scope to change here. (`MAX_ACTIVE_DIMS` keeping dims nearest
   zero is wrong for grid-indexed layouts and is tracked separately.)
4. `zero_tol` pruning keeps its current semantics. Expressed zeros are load-bearing
   under the Zero Rules and MUST NOT be pruned by the new code paths.
5. `gap_threshold` continues to mean what it means today; `use_sparse_dense(8)`,
   `use_sparse_dense(256)` etc. must still produce identical *results* (they
   currently do — only performance should vary).

## 6. Why results can differ, and when that is acceptable

Direct `np.convolve` and `_fft_convolve` do not agree bit-for-bit. Measured on
1000-term operands the relative difference is 2.8e-11, with direct convolution
being the more accurate of the two.

Therefore: any reordering of *which* path is taken changes results in the last
bits. The implementation MUST preserve the existing threshold rule so that
before/after comparison is exact. If the threshold is later retuned, that is a
separate, independently-justified change.

## 7. Open questions

1. **Zero trimming (§3.2 rule 5).** Trimming leading/trailing zeros from a
   cluster shortens arrays but can destroy an expressed zero that the Zero Rules
   treat as meaningful. Needs a decision, and it interacts with `_is_wholly_zero`
   in `composite_lib` (4.1% of the current profile — worth checking whether it
   can be answered from cluster metadata instead of a value scan).
2. **Cached scalar metadata.** `_is_wholly_zero`, leading dimension, and term
   count are all derivable from the cluster list in O(k). Consider caching the
   leading dimension on the object, since `_lead`-style queries are frequent.
3. **`deconvolve` conversion.** Deferred per §4.5; decide whether to convert
   after the hot path is landed and measured.
4. **Memory.** Merging runs across a gap of up to `gap_threshold` allocates
   interior zeros. For layouts with many small far-apart clusters (blocked
   addressing, `f*n + c + 1`), confirm the default 64 is still right.

## 8. Verification plan

1. **Differential test against `DictBackend`** over randomised inputs covering:
   single term; empty composite; wholly-zero composite; expressed zeros interior
   to a run; one dense run; two runs separated by less than `gap_threshold`; two
   runs separated by more; negative-only, positive-only and straddling dims;
   operands of very different lengths.
2. **Existing suite green.** Baseline at time of writing: `pytest tests/` gives
   **76 passed, 1 failed** — the one failure is `test_limits`, from the separate
   `sqrt` dimension fix (2 of 105 limit assertions), not from this work. That
   baseline must not get worse.
3. **Bit-identity check** on a fixed corpus, before vs after, asserting exact
   equality wherever both runs take the direct path.
4. **Benchmark**, reported as a table of n in {3, 8, 20, 60, 200, 1000} for
   `a*b` and `a+b`, dense and multi-cluster, against the numbers in §1.
5. **Application-level regression**: rerun the PDE heat-equation convergence
   check, the `composite_roots` suite (9 cases), and one cached TSP instance, and
   confirm identical outputs.

Benchmark harness:

```python
import time
from composite import Composite
def bench(fn, reps=3000):
    for _ in range(200): fn()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1e6     # microseconds
n = 20
a = Composite({-i: 1.0 + i for i in range(n)})
b = Composite({-i: 2.0 - 0.001 * i for i in range(n)})
print(bench(lambda: a * b), "us")                      # baseline 65.35
```

## 9. Out of scope

- Rust or C rewrite. This work first: it removes ~40x of avoidable Python-level
  work, and a native implementation measured against the *current* baseline would
  attribute that 40x to the language rather than to the representation.
- Changing `MAX_ACTIVE_DIMS` or its truncation policy (tracked separately).
- Changing the Zero Rules.
- Multi-lane / tensor representations.

## 10. Expected outcome

40x on dense-operand multiplication at n=20 (65.35 us -> ~1.65 us), with 86% of
the remaining time in `np.convolve` itself. Sparse layouts benefit by the removal
of the same clustering round-trip, by an amount not yet measured — the TSP
crossword (one dense negative run plus sparse positive metadata) is the case to
measure, since it re-derives its structure on every operation today.
