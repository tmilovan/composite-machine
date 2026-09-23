# README.md

## Announcements

### New release (September 2026)

After months of experimenting, learning (finding about Levi Civita fields etc.), building and testing different implementations, here is the new release that contains the accumulated findings. This release contains results of trying out different approaches and results of numeruos experiments. The more experimental stuff still relies to external support, oracles etc. (as it should), the more tested features are tending to become more and more self reliant with additional iterations (eg, derivations and integrals.).

What it tries to achieve:

- thinning the reliance on external libraries, trying to express as much as we can through composite tooling
- performance enhancements
- isolation end elimination of trucation errors
- add more depth, reach and precision to the toolkit by adding the transseries support for initial experimentation (can of worms)

What it adds:

- refinements to zero handling edge cases
- adds float dimensions, so we can finally take a square roots on composites with exact precision and remain composite
- adds experimental support for vector dimensions which enables taking log of an
composite number
- tons of edge case bugfixes (especially for integration)

Note of caution: this is still highly experimental and most likely (for sure) still contains some misconceptions and a lot of edge cases and other bugs. The purpose of the library is to showcase what is possible and to serve as a baseline for further exploration.

### Library release (April 2026)

The first proper pypy library based on this experimental features has been released. A standalone tool to evaluate Python functions at points where they're undefined and get exact limit values if they exists.

- **[https://github.com/FWDhr/composite-resolve](https://github.com/FWDhr/composite-resolve)**


# Composite Machine

**Automatic calculus via dimensional arithmetic... and a bit more.**

A data structure that implements a number system, letting different parts of non-standard (and standard) math work together.

For example, it gives you derivatives, integrals, and limits as a side effect of normal computation. No symbolic engine, no computation graph, no tape. Tag a number, do your math, read the results off the dimensional coefficients.

> *1 − 1 ≠ 0*
>

> *The residue is infinitesimal, structured, and it contains the derivative of every operation that produced it.*
>

Alpha stage. Research code. The math works, ~~performance doesn't (yet)~~. AGPL-licensed. A PyTorch/CUDA backend is available under commercial license."

---

## What's this

Numbers are sparse dicts mapping dimensions to coefficients. A dimension is an integer, or a vector over an iterated-logarithm basis when log-scale terms are in play. Dimension 0 is the value. Negative dimensions store derivative info. Multiply dimensions - turns out that's the same thing as the product rule and chain rule, just expressed as data structure operations.

```python
from composite.composite_lib import R, ZERO

x = R(3) + ZERO          # 3 + infinitesimal seed
result = x ** 4           # just compute normally

result.st()               # 81  - the value, f(3)
result.d(1)               # 108 - first derivative
result.d(2)               # 108 - second derivative
result.d(3)               # 72  - third derivative
result.d(4)               # 24  - fourth derivative
```

One evaluation. All derivatives fall out. No separate differentiation pass.

---

## Background

The derivative computation part builds on well-known work: Clifford's **dual numbers** (1873), Wengert's **forward-mode AD** (1964), Rall's **Taylor arithmetic** (1981), Griewank's framework (2000).

The number system has a separate and older lineage. A sparse map from exponents to coefficients, with non-integer exponents and finitely many terms below any given one, is the shape of the **Levi-Civita field** (Levi-Civita, 1892–1898) - the smallest non-Archimedean ordered field extension of the reals that is real-closed and Cauchy-complete. Letting an exponent be a *vector* ordered lexicographically instead of a single number gives **Hahn series** (Hahn, 1907), which is what the iterated-logarithm basis here amounts to: dimensions valued in an ordered group, compared componentwise. The scale those vectors index - *x*, log *x*, log log *x*, ranked by eventual dominance - is du Bois-Reymond's *Infinitärcalcül* as set out in Hardy's **Orders of Infinity** (1910), and the **Hardy fields** built on it. Expansions that mix powers, exponentials and iterated logs are **transseries** (Écalle, 1992; van der Hoeven, 2006). The infinitesimals themselves are made rigorous by Robinson's **non-standard analysis** (1966), and the surreals (Conway, 1976) contain the Levi-Civita field as a subfield.

Computing in such a field, rather than reasoning about it, also has prior art. Berz framed **automatic differentiation as non-Archimedean analysis** (1992), and Shamseddine and Berz developed numerical analysis directly on the Levi-Civita field, including derivatives of functions where classical AD breaks down. Sergeyev's **grossone** (2003 onward) is the closest in representation: a positional numeral system in powers of an infinite unit ①, with the infinitesimal ①⁻¹, used on an "Infinity Computer" for exact higher-order differentiation, ODE solvers and lexicographic optimization - the same records as the dimensions here, written in a different notation. Grossone keeps the ordinary zero (0·① = 0, ① − ① = 0); this library does not, and that is where the two part ways. The overlap is worth stating plainly: the algebra here is not new, and where this library's structures coincide with those, the credit is theirs.

What this library explores is a different algebraic context for that mechanism. Higher-order terms are preserved instead of truncated. Subtraction retains provenance instead of collapsing to zero. Multiplication by zero shifts structure instead of destroying it. The idea is that if you stop throwing away information at each step, calculus operations become extractable from the algebra.

Does this generalize to everything? Open question. The test suite covers a wide range of standard problems and the results match. Finding the boundaries is the point of this project.

For the theoretical framing, see the paper.

---

## How it compares

Breadth in one structure, at a cost that depends entirely on the shape of the problem.
The numbers under [Performance](#performance) are measured, not asserted, and they do
not all point the same way.

- **vs PyTorch/JAX** - They give first-order gradients, fast, and vectorised across a batch. This gives every order from one evaluation, plus limits and integration. Neither is a backend here, so no throughput ratio against them is quoted - the measurements below are against NumPy and SymPy, which are what this actually runs on.
- **vs SymPy** - SymPy is symbolic, this is numerical. On the cases measured this is the faster of the two: 3–70x on indeterminate limits (both exact) and ~6400x on a Taylor expansion to order 8, agreeing to 2.5e-15.
- **vs mpmath** - mpmath is arbitrary-precision and carries the special-function library this does not (gamma, zeta, Bessel). On derivatives the two agree exactly: the 4th derivative of x⁴eˣ at 1 matches to all 15 digits. The difference is method - mpmath samples and extrapolates, so a limit is only as good as the extrapolation converges. On six harder limits it returned 0.99962 for xˣ as x→0⁺ and −2.7e−8 for x²·ln x, where reading the standard part off the algebra gives both exactly.
- **vs dual numbers** - Classic dual numbers give you one derivative (epsilon squared is zero). Here epsilon squared is kept, so you get all orders.

---

## Examples

### Derivatives

```python
from composite.composite_lib import derivative, nth_derivative, all_derivatives, exp

derivative(lambda x: x ** 2, at=3)               # 6.0
nth_derivative(lambda x: x ** 5, n=3, at=2)      # 240.0
all_derivatives(lambda x: exp(x), at=0, up_to=5) # [1, 1, 1, 1, 1, 1]
```

### Limits

No L'Hôpital. Plug in the infinitesimal, read the standard part.

```python
from composite.composite_lib import limit, sin, R

limit(lambda x: sin(x) / x, as_x_to=0)                  # 1.0
limit(lambda x: (x**2 - R(4)) / (x - R(2)), as_x_to=2)  # 4.0
```

### Integration

```python
from composite.composite_lib import integrate, exp

integrate(lambda x: x ** 2, 0, 1)              # 0.333...
integrate(lambda x: exp(-x), 0, float('inf'))  # 1.0
integrate(lambda x, y: x * y, (0, 1), (0, 1))  # 0.25
```

### Division by zero

ZERO isn't Python's 0 - it's a structural infinitesimal, coefficient 1 at dimension −1. Operations on it are well-defined and reversible:

```python
from composite.composite_lib import ZERO, R

(ZERO / ZERO).st()                          # 1.0
(R(5) * ZERO / ZERO).st()                   # 5.0
(R(7) * ZERO * ZERO / ZERO / ZERO).st()     # 7.0
```

### Multivariable

```python
from composite.composite_multivar import gradient_at, laplacian_at

gradient_at(lambda x, y: x**2 + y**2, [3, 4])  # [6, 8]
laplacian_at(lambda x, y: x**2 + y**2, [3, 4]) # 4
```

### Complex analysis

```python
from composite.composite_extended import residue, convergence_radius

residue(lambda z: 1 / z, at=0)                  # 1.0
convergence_radius(lambda z: 1 / (1 - z), at=0) # 1.0
```

---

## Modules

- **[composite_lib.py](composite/composite_lib.py)** - Core engine. Composite class, all arithmetic, transcendentals, derivatives, limits, integration.
- **[composite_multivar.py](composite/composite_multivar.py)** - Multivariable calculus. MC class, partial derivatives, gradient, Hessian, Jacobian, Laplacian, divergence, curl.
- **[composite_extended.py](composite/composite_extended.py)** - Complex analysis. Complex composites, residues, poles, contour integrals, asymptotics, ODE solver.
- **[composite_vector.py](composite/composite_vector.py)** - Vector calculus. Triple integrals, line integrals, surface integrals.
- **[backends/](composite/backends/)** - Interchangeable storage for the dimension map: dict, sparse-dense, vector-dimension, dense-series.

---

## What works

**Stable:**

- Full arithmetic with dimensional convolution and deconvolution
- Integer and real-exponent powers
- Transcendentals - sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, ln, sqrt, erf, erfc, normal_cdf
- All-order derivatives from a single evaluation
- Algebraic limits including indeterminate forms and limits at infinity
- Definite, improper, and adaptive integration with error estimates
- Vector dimensions over an iterated-logarithm basis - log, loglog and deeper
  scales as ordinary arithmetic, at any depth, with the transcendentals
  accepting log-axis arguments
- Completeness tracking - every value carries the highest order it is complete
  to, propagated through each operation
- TracedComposite for step-by-step operation logging

**Experimental:**

- Multivariable calculus (MC class, partial derivatives, differential operators)
- Vector calculus (line integrals, surface integrals, triple integrals)
- Complex analysis (residues, contour integrals, analytic continuation, convergence radius)
- ODE solver via RK4 with composite evaluation

**Not yet implemented or highly experimental:**

- Inverse hyperbolics (asinh, acosh, atanh)
- Fourier, Laplace, Z transforms
- Special functions (Bessel, gamma)
- Optimization routines

---

## Performance

No single ratio - it depends on what you ask for. Measured on CPython/macOS, float64, one thread,
NumPy 1.26.4, SymPy 1.14.0.

**Derivatives.** Tag a number, evaluate `1/(1+x)` once, read all ten derivatives off the result:

| | time | accuracy |
|---|---|---|
| composite (dict backend) | **90 µs** | exact, every order |
| NumPy finite differences | 140 µs | 13 correct digits at order 1, **about 1** by order 10 |

Faster and exact. One evaluation carries every order, while a stencil needs a fresh step size per
order and has run out of digits by order 10.

A *single* first derivative goes the other way: NumPy does it in **0.2 µs** against **35 µs** here,
roughly 140x. The crossover is around the third derivative.

**Batches.** NumPy vectorises and this does not - four orders of magnitude per point. Pythorch and CUDA backends not present here support batching.

**Sparse grids.** An explicit PDE whose active front stays at 121 cells: **24x faster** than a
dense NumPy grid at 200,000 cells, **260x** at 2,000,000. Composite time is flat; the dense grid
pays for the whole domain whether anything is happening in it or not.

**vs SymPy.** Indeterminate limits: **3–70x faster**, both exact. A Taylor expansion to order 8:
0.6 ms against 3.7 s, about **6000x**, agreeing to 15 digits.

**Backend selection.** `use_dense_series()` for calculus - anything with transcendentals builds a
dense, contiguous series, and it is ~1.6x faster there than the alternatives. `use_dict()` for
pure-arithmetic jets, where there is no series to lay out (~2x faster on `1/(1+x)`). The default
`use_sparse_dense()` is the grid backend: several times slower on either kind of jet, and the one
to keep for sparse supports over a large domain.

A high-performance backend using PyTorch with CUDA and MPS acceleration is available under commercial license. Contact [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr).

---

## Installation

```bash
git clone https://github.com/tmilovan/composite-machine.git
cd composite-machine
pip install -e .
```

Python 3.7+. NumPy is optional (used for FFT-accelerated multiplication).

---

## Demos

Each one runs on its own and prints what it computed against a known answer.

```bash
python demos/composite_forensics.py        # start here
```

| demo | what it shows |
|---|---|
| [`composite_forensics.py`](demos/composite_forensics.py) | Is the formula bad, or is the problem hard? The two spellings of a quadratic root, one losing 25% of the answer, and the verdict that separates them. Also the failure float64 cannot see: a value that is right while its derivative is not. |
| [`composite_physics.py`](demos/composite_physics.py) | Dirac hydrogen to α⁸ from one evaluation, zero-point mode sums with the divergent and finite parts on separate grades, Schwarzschild at the horizon and near r = 0. 44 checks against closed forms. |
| [`composite_roots.py`](demos/composite_roots.py) | Global root finding: intervals *proved* empty by a Taylor bound rather than sampled and hoped for, then Householder polishing that costs nothing because the derivatives are already there. |
| [`composite_singularity.py`](demos/composite_singularity.py) | A power series locating its own nearest singularity and exponent, which is the blow-up time of an ODE and the critical point of a lattice model. |
| [`composite_stability_radius.py`](demos/composite_stability_radius.py) | How much can one road get slower before the best route changes? One solve instead of one re-solve per edge. |
| [`calculus_tutor.py`](demos/calculus_tutor.py) | An interactive console tutor: what the dimensions are doing while calculus happens. |

The forensics demo prints a few warnings before its first table. They are part of
the demonstration, not breakage: the R1 notice fires because the demo audits
formulas that contain written zeros, which is the fault it is there to catch.

---

## Testing

```bash
pip install -e .          # so `import composite` resolves to this checkout
python -m pytest tests/   # everything
```

Or run a single suite directly:

```bash
PYTHONPATH=. python tests/test_standalone.py               # core + paper theorems
PYTHONPATH=. python tests/test_limits.py                   # limits, all classes
PYTHONPATH=. python tests/test_integration_comprehensive.py # every integral form
```

`PYTHONPATH=.` matters: running a test as a bare script puts `tests/` on the
import path rather than the repo root, so `composite` resolves to whatever is
installed instead of the working copy.

**839 tests across thirteen suites, all passing.**

| suite | tests | covers |
|---|---|---|
| `test_standalone.py` | 167 | paper theorems T1–T8, algebra, derivatives, limits, zero division |
| `test_vector_dimensions.py` | 150 | vector dimensions, depth genericity, log-axis transcendentals |
| `test_dimension_scales.py` | 131 | dimensions that are not integers, and not scalars |
| `test_limits.py` | 105 | indeterminate forms, oscillatory, at infinity, directional, domain errors |
| `test_multivar_disprove.py` | 68 | multivariable vs single-variable, Black-Scholes Greeks |
| `test_integration_comprehensive.py` | 54 | definite, improper, triple, line, surface |
| `test_multivar_extended.py` | 50 | gradients, Hessians, Jacobians, complex analysis, ODEs |
| `test_composite_vector.py` | 25 | vector calculus |
| `test_series_completeness.py` | 22 | every transcendental returns only the orders it completes |
| `test_identities.py` | 20 | identities computed through independent paths |
| `test_stress.py` | 20 | hard limits, derivatives, integrals |
| `test_stress_hard_edge.py` | 20 | 3rd/4th order, deep composition chains |
| `test_composite_metadata.py` | 7 | one number carrying data and metadata |
| `turing_completeness/` | 3 files | Turing-completeness experiments |

---

## Paper

Milovan, T. (2026). *Provenance-Preserving Arithmetic: A Unified Framework for Automatic Calculus.* Zenodo.

[https://doi.org/10.5281/zenodo.18528788](https://doi.org/10.5281/zenodo.18528788)

---

## Docs

- [**Tutorial**](docs/Tutorial%20-%20Getting%20Started.md) - Get started quickly
- [**API Reference**](docs/API%20Reference.md) - Complete function docs
- [**Implementation Guide**](docs/Implementation%20Guide.md) - How it works internally
- [**Examples**](docs/Examples.md) - Code snippets for common tasks
- [**Roadmap (DRAFT)**](docs/Roadmap%20(DRAFT).md) - What's next
- [**Zero Rules v2**](docs/Zero%20Rules%20v2%20%E2%80%94%20Formal%20Specification%20(DRAFT).md) - What a zero coefficient means, and how it behaves

---

## Contributing

Contributions welcome. Useful areas:

- Special functions - Bessel, gamma, etc.
- Bug reports and edge cases
- Docs and examples

Process: open an issue first, fork, add tests, PR.

---

## Citation

If you use this in research:

Milovan, T. (2026). Composite Machine: Automatic Calculus via Dimensional Arithmetic. [https://github.com/tmilovan/composite-machine](https://github.com/tmilovan/composite-machine)

---

## License

**Code:** AGPL-3.0. Free for open-source, research, and personal use. Commercial licensing available - contact [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr).

**Paper:** CC BY 4.0.

---

Toni Milovan · Pula, Croatia · [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr)
