# README.md

## Announcements

### New changes (September and October 2026)

After months of experimenting, learning, building and testing different implementations, here is the new release that contains the accumulated findings. This release contains results of trying out different approaches and results of numerous experiments. The more experimental stuff still relies to external support, oracles etc. (as it should), the more tested features are tending to become more and more self reliant with additional iterations (eg, derivations and integrals.).

**What it tries to achieve:**

- thinning the reliance on external libraries, trying to express as much as we can through composite tooling
- performance enhancements
- isolation and elimination of truncation errors
- add more depth, reach and precision to the toolkit by adding the transseries support for initial experimentation (can of worms)

**What it adds:**

- refinements to zero handling edge cases
- adds float dimensions, so we can finally take a square roots on composites with exact precision and remain composite
- adds experimental support for vector dimensions which enables taking log of an
composite number
- tons of edge case bugfixes (especially for integration)

Note of caution: this is still highly experimental and most likely (for sure) still contains some misconceptions and a lot of edge cases and other bugs. The purpose of the library is to showcase what is possible and to serve as a baseline for further exploration.

### Library release (April 2026)

The first proper PyPI library based on this experimental features has been released. A standalone tool to evaluate Python functions at points where they're undefined and get exact limit values if they exists.

- **[https://github.com/FWDhr/composite-resolve](https://github.com/FWDhr/composite-resolve)**


# Composite Machine

**A computational system that does not lose data on operations with zeros, giving automatic calculus via dimensional arithmetic... and a bit more.**

## Why all this? ##

Out of a desire to create a computational system that doesn't lose data on operations with 0, where a * 0 will be invertible and where a / 0 will not crash, but that will still give accurate results. All of this builds on Euler's notion that infinitesimals are actually zeros with different ratios.

In essence this system converts zeros to infinitesimals and tries to do arithmetic with them.

## What does it give? ##
 
The implementation leads to a data structure that implements a number system, letting different parts of non-standard (and standard) math work together.

For example, it gives you derivatives, integrals, and limits as a side effect of normal computation. No symbolic engine, no computation graph, no tape. Tag a number, do your math, read the results off the dimensional coefficients.

> Here *1 − 1 ≠ 0 - it is zero with residue* 
>

> *The residue is infinitesimal, structured, and it contains the derivative of every operation that produced it.*
>

Euler got here first. In *Institutiones calculi differentialis* (1755), Part I, Chapter 3
(*De infinitis et infinite parvis*), §84-§88, he says an infinitely small quantity really is
zero, and that this is not a problem, because there are two ways two quantities can be equal:
arithmetically, when *a* − *b* = 0, and geometrically, when *a*/*b* = 1. Any two zeros are
equal in the first way but not in the second. So d*x* = 0 and *a*·d*x* = 0, and still
*a*·d*x* : d*x* = *a* : 1. In §86 he says the whole force of the differential calculus is
finding that ratio.

Then in §85 he says what to do about it:

> Since any ratio whatever can hold between zeros, various characters are deliberately used
> to indicate this diversity ... otherwise the business would slide into the greatest confusion
> and could not be sorted out in any way.
>
> *ad hanc diversitatem indicandam consulto varii characteres usurpantur ... in maximam
> confusionem illaberetur neque ullo modo expediri posset.*

Different zeros need different characters, or the ratio between them is lost. Here the
characters are dimensions: `|0|₀`, `|1|₋₁`, `|1|₋₂` are different zeros, and the arithmetic
keeps their ratios. Euler tracked the ratio by reasoning about it next to the calculation.
Here it is kept in the number, so it is still there for the next operation. In §88 he also
orders zeros by how fast they vanish, d*x*² before d*x*, which is the dimension ladder.

Euler states infinitesimal is zero, so this system follows that with proposition that zero is infinitesimal too.

The whole system is that identification, made uniformly: `0` is `|1|₋₁`, the
infinitesimal `h`, and every operation is ordinary arithmetic on series in `h`
with that one substitution. Measured on the code:

```
a - a           ==  a * 0                a cancellation is a multiplication by h
(a - a) / 0     ==  a                    and it is reversible
x = R(3) + ZERO
x**2 - R(9)      ->  <|0|₀ |6|₋₁ |1|₋₂>                       the jet of x^2 - 9 at 3
exp(x) - exp(x)  ->  <|20.0855|₋₁ |20.0855|₋₂ |10.0428|₋₃ ...>  the jet of e^x * (x - 3)
```

A composite is the jet of the function the expression denotes, and a written
constant is a germ too: a written `0` denotes `x - a`. That is what lets the
residue of `1 - 1` carry the derivative of the operation that produced it, and
it is also the one habit to unlearn. `x*x + 0` is the jet of `x^2 + (x - a)`,
derivative 5 at 2, and the library says so with `NotConventionalWarning`. Spell
an absence as `Composite({})`, or use `conventional()` or `TAG()` for the
classical reading. The price, stated plainly: there is no additive identity,
and `a + (-a)` does not commute on a cancelling pair. Zero Rules v2 has the
rules and the measured laws.

Alpha stage. Research code. The math works, ~~performance doesn't (yet)~~. AGPL-licensed. A PyTorch/CUDA backend is available under commercial license.

---

## In short?


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

The system builds on the ideas and work of Euler, Levi-Civita, Laurent, Robinson and many others. Without their work on formalizing those ideas and standardizing the methods and proofs for working with them, implementation of this system's main proposition would not be possible.

The derivative computation part builds on well-known work: Clifford's **dual numbers** (1873), Wengert's **forward-mode AD** (1964), Rall's **Taylor arithmetic** (1981), **Griewank's framework** (2000). The multivariable derivatives use univariate Taylor propagation with interpolation (Griewank, Utke and Walther, 2000).

The number system has a separate and older lineage. A sparse map from exponents to coefficients, with non-integer exponents and finitely many terms below any given one, is the shape of the **Levi-Civita field** (Levi-Civita, 1892-1898), the smallest non-Archimedean ordered field extension of the reals that is real-closed and Cauchy-complete. Letting an exponent be a vector ordered lexicographically instead of a single number gives **Hahn series** (Hahn, 1907), which is what the iterated-logarithm basis here amounts to: dimensions valued in an ordered group, compared lexicographically. The scale those vectors index, x, log x, log log x, ranked by eventual dominance, is **du Bois-Reymond's Infinitarcalcul** as set out in **Hardy's Orders of Infinity** (1910), and the **Hardy fields** built on it. Expansions that mix powers, exponentials and iterated logs are **transseries** (Ecalle, 1992; van der Hoeven, 2006). The infinitesimals themselves are made rigorous by Robinson's **non-standard analysis** (1966), and the **surreals** (Conway, 1976) contain the Levi-Civita field as a subfield.

Computing in such a field, rather than reasoning about it, also has prior art. Berz framed automatic differentiation as non-Archimedean analysis (1992), and Shamseddine and Berz developed numerical analysis directly on the Levi-Civita field, including derivatives of functions where classical AD breaks down. **Sergeyev's grossone** (2003 onward) is the closest in representation: a positional numeral system in powers of an infinite unit, with its reciprocal as the infinitesimal, used on an "Infinity Computer" for exact higher-order differentiation, ODE solvers and lexicographic optimization. Those are the same records as the dimensions here, written in a different notation. Grossone keeps the ordinary zero (0 times grossone is 0, grossone minus grossone is 0); this library does not, and that is where the two part ways. The overlap is worth stating plainly: the algebra here is not new, and where this library's structures coincide with those, the credit is theirs.

Total division has its own prior art, adjacent rather than overlapping. **Wheel theory** (Carlstrom, 2004) extends a commutative ring with one extra element, bottom, so that x/0 is always defined. **Meadows** (Bergstra and Tucker, 2007) take the convention 1/0 = 0 and keep the field equations. The projective reals add one point at infinity. IEEE 754 has signed inf and nan. All of these make division total by adding a point or by adopting a convention for 1/0. None grades the infinity or keeps what was divided: 1/0 and 2/0 are the same element in each of them, and in inf - inf the operands are gone. Here 1/0 is |1|_1 and 2/0 is |2|_1, |2|_1 - |1|_1 is |1|_1, and |2|_1 * |1|_-1 is 2 again. That is the difference, and a reader who knows those systems should expect it to be named.

The word provenance comes from **Provenance semirings** (Green, Karvounarakis and Tannen, 2007) which annotate every query result in a database with a polynomial recording which source tuples produced it and how; the annotation is a separate algebraic object carried beside the value. Here the record is numerical and lives in the same number as the value: the residue of a - a is a itself one grade down, so the provenance of a cancellation is read with the same arithmetic that produced it, and dividing by 0 recovers the operand. The aim is the same, knowing where a result came from; the mechanism is the number rather than an annotation on it.

**What this library explores is a different algebraic context for that mechanism. Higher-order terms are kept, and where they are cut off the cut is explicit and tracked. Subtraction retains provenance instead of collapsing to zero. Multiplication by zero shifts structure instead of destroying it. The idea is that if you stop throwing away information at each step, calculus operations become extractable from the algebra.**

Does this generalize to everything? Open question. The test suite covers a wide range of standard problems and the results match, apart from the failures listed under Testing. Finding the boundaries is the point of this project.

For the theoretical framing, see the paper but keep in mind paper was a starting blueprint and currently lagging behind the implementation.

---

## How it compares

Breadth in one structure, at a cost that depends entirely on the shape of the problem.
The numbers under [Performance](#performance) are measured, not asserted, and they do
not all point the same way.

- **vs PyTorch/JAX** - They give first-order gradients, fast, and vectorised across a batch. This gives every order from one evaluation, plus limits and integration. Neither is a backend here, so no throughput ratio against them is quoted - the measurements below are against NumPy and SymPy, which are what this actually runs on.
- **vs SymPy** - SymPy is symbolic, this is numerical. On the cases measured this is the faster of the two: 3-70x on indeterminate limits (both exact) and ~6400x on a Taylor expansion to order 8, agreeing to 2.5e-15.
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
- **[composite_multivar.py](composite/composite_multivar.py)** - Multivariable calculus by directional composites: ordinary composites evaluated along several directions. Partial derivatives, gradient, Hessian, Jacobian, Laplacian, divergence, curl, limits. (The former MC class is parked in `composite_multivar_mc.py`.)
- **[composite_extended.py](composite/composite_extended.py)** - Complex analysis. Complex composites, residues, poles, contour integrals, asymptotics, ODE solver.
- **[composite_vector.py](composite/composite_vector.py)** - Vector calculus. Triple integrals, line integrals, surface integrals.
- **[backends/](composite/backends/)** - Interchangeable storage for the dimension map: dict, sparse-dense, vector-dimension, dense-series, and fractional (exact rational lattice; dict, NumPy and PyTorch flavours).
- **[forensics.py](composite/forensics.py)** - Cancellation forensics. Condition number and forward error bound from one seeded evaluation: is the formula bad, or is the problem hard?
- **[uncertainty.py](composite/uncertainty.py)** - GUM uncertainty budgets with the bias and higher-order variance terms, and the GUM-S1 admissibility check, from composite derivatives.
- **[singularity.py](composite/singularity.py)** - Location and exponent of a series' nearest singularity: blow-up time of an ODE, critical point of a lattice model, coefficient growth.
- **[resummation.py](composite/resummation.py)** - Borel-Pade resummation of divergent series, with the Pade approximant computed as composite Euclidean division.
- **[transseries.py](composite/transseries.py)** - The `exp(-1/h)` scale below every power, carried as a sector index outside the dimension.
- **[explain.py](composite/explain.py)** - `explain(f, at)`: what a function does at a point (pole, corner, log growth, value and slope), from one evaluation.
- **[display.py](composite/display.py)** - Notebook display: the grades as a table, with completeness and denotation shown.

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

- Multivariable calculus (directional composites: partial derivatives, differential operators)
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

**Batches.** NumPy vectorises and this does not - four orders of magnitude per point. PyTorch and CUDA backends not present here support batching.

**Sparse grids.** An explicit PDE whose active front stays at 121 cells: **24x faster** than a
dense NumPy grid at 200,000 cells, **260x** at 2,000,000. Composite time is flat; the dense grid
pays for the whole domain whether anything is happening in it or not.

**vs SymPy.** Indeterminate limits: **3-70x faster**, both exact. A Taylor expansion to order 8:
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

Python 3.8+ (the fractional backends need 3.9+, for `math.lcm`). NumPy is required.

---

## Notebooks

Executed Jupyter notebooks, each starting from the beginning and printing what it
computed against what it should be.

| notebook                                                                                                       | what it covers                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [`0 Arithmetic`](notebooks/0%20Arithmetic.ipynb)                                                               | Ordinary `+ - * /` on composites: the same floats back, plus what arithmetic throws away at zero, what that costs, and how to switch it off.    |
| [`1 Basic concepts`](notebooks/1%20Basic%20concepts.ipynb)                                                     | A number that carries its own metadata: grades, the zero that does not annihilate, and division by zero.                                        |
| [`2 Simple usage`](notebooks/2%20Simple%20usage.ipynb)                                                         | Five one-call tasks: `explain`, every derivative from one evaluation, `0/0` without L'Hopital, `audit`, and `TAG` / `to_ieee754`.               |
| [`3 Advanced concepts`](notebooks/3%20Advanced%20concepts.ipynb)                                               | Residues, root multiplicity, limits at infinity, branch points, the log axis, divergent series, uncertainty budgets, and what this does not do. |
| [`4 Derivatives conventional and composite`](notebooks/4%20Derivatives%20conventional%20and%20composite.ipynb) | The derivative of the function you meant against the function the expression denotes, and how to ask for either.                                |
| [`5 Integration`](notebooks/5%20Integration.ipynb)                                                             | Integration as the opposite grade shift: definite integrals from two composites, singular endpoints, and infinite ranges.                       |


---

## Demos

Each one runs on its own and prints what it computed against a known answer.

```bash
python demos/composite_forensics.py        # start here
```

| demo | what it shows |
|---|---|
| [`composite_forensics.py`](demos/composite_forensics.py) | Is the formula bad, or is the problem hard? The two spellings of a quadratic root, one losing 25% of the answer, and the verdict that separates them. Also the failure float64 cannot see: a value that is right while its derivative is not. |
| [`composite_degeneracy.py`](demos/composite_degeneracy.py) | Four questions that classically need four algorithms and four tolerances, all answered by reading one grade: rank deficiency from the grade of a determinant, root multiplicity from expanded coefficients alone, order of contact between two curves, and a vertex found without ever forming the curvature. The answers are integers, so there is nothing to threshold. 60 checks. |
| [`composite_physics.py`](demos/composite_physics.py) | Dirac hydrogen to α⁸ from one evaluation, zero-point mode sums with the divergent and finite parts on separate grades, Schwarzschild at the horizon and near r = 0. 44 checks against closed forms. |
| [`composite_roots.py`](demos/composite_roots.py) | Global root finding: intervals *proved* empty by a Taylor bound rather than sampled and hoped for, then Householder polishing that costs nothing because the derivatives are already there. |
| [`composite_singularity.py`](demos/composite_singularity.py) | A power series locating its own nearest singularity and exponent, which is the blow-up time of an ODE and the critical point of a lattice model. |
| [`composite_stability_radius.py`](demos/composite_stability_radius.py) | How much can one road get slower before the best route changes? One solve instead of one re-solve per edge. |
| [`composite_fractional_dynamics.py`](demos/composite_fractional_dynamics.py) | How fractional (lattice) dimensions behave while operations run: what the canonical lattice does along a chain, term growth, and what a fractional backend costs when no fractional order is present. |
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

Measured 2026-10-07 on `composite-env-arm`: **739 pytest tests, plus 23
script suites totalling 1627 checks**, the script suites run under pytest by
`test_suites.py`. Three checks fail:
`test_resummation` R4.11 asserts that a finite Laplace cutoff is inexact and it
no longer is, and two `test_singularity` S3 exponents miss their tolerance by
4e-14 and 1.2e-12.

Script suites (each prints its own tally):

| suite | checks | covers |
|---|---|---|
| `test_resummation.py` | 177 | Borel-Pade resummation of divergent series: Euler-Stieltjes, Painleve I |
| `test_standalone.py` | 172 | paper theorems T1-T8, algebra, derivatives, limits, zero division |
| `test_zero_coercion.py` | 168 | a written zero is an expressed zero: R1-R6, the algebraic laws, cancelling pairs |
| `test_vector_dimensions.py` | 150 | vector dimensions, depth genericity, log-axis transcendentals |
| `test_dimension_scales.py` | 131 | dimensions that are not integers, and not scalars |
| `test_forensics.py` | 124 | kappa against predicted error: is the formula bad or the problem hard |
| `test_limits.py` | 105 | indeterminate forms, oscillatory, at infinity, directional, domain errors |
| `test_transseries.py` | 95 | the `exp(-1/h)` sector below every power: ordering first, then arithmetic |
| `test_singularity.py` | 91 | location and exponent of a series' nearest singularity, against known answers |
| `test_uncertainty.py` | 64 | GUM budgets with bias and higher-order terms, against closed forms |
| `test_integration_comprehensive.py` | 54 | definite, improper, triple, line, surface |
| `test_backend_agreement.py` | 52 | dict, sparse-dense and vector backends must not disagree about a number |
| `test_multivar_extended.py` | 50 | gradients, Hessians, Jacobians, complex analysis, ODEs |
| `test_singularity_handling.py` | 36 | what every operation must do when its argument is not finite |
| `test_derivatives.py` | 35 | one evaluation, every derivative, tested where it can fail |
| `test_series_completeness.py` | 32 | every transcendental returns only the orders it completes |
| `test_composite_vector.py` | 25 | vector calculus |
| `test_identities.py` | 20 | identities computed through independent paths |
| `test_stress.py` | 20 | hard limits, derivatives, integrals |
| `test_stress_hard_edge.py` | 20 | 3rd/4th order, deep composition chains |
| `turing_completeness/` | 3 files | Turing-completeness experiments (6 checks plus two staged scripts) |

Pytest modules:

| module | tests | covers |
|---|---|---|
| `test_fractional_backend.py` | 122 | fractional dimensions on an exact rational lattice |
| `test_integrate_jets.py` | 84 | the integral by meeting composites, one per node |
| `test_degeneracy.py` | 70 | flagging a Taylor reading that a second infinitesimal source has entered |
| `test_multivar_disprove.py` | 71 | multivariable vs single-variable, Black-Scholes Greeks |
| `test_multivar_jets.py` | 85 | directional composites against 50-digit mpmath |
| `test_multivar_backends.py` | 65 | the multivariable functions on every storage backend |
| `test_conventional_zero.py` | 51 | `conventional()`: an expressed zero as the ordinary zero, in scope |
| `test_single_infinitesimal.py` | 37 | `TAG()` and `single_infinitesimal()`: one blessed infinitesimal, conventional rest |
| `test_derivative_grade_shift.py` | 30 | differentiation as a grade shift |
| `test_offorigin_and_ordering.py` | 28 | residue and pole order off the origin, mixed-grade ordering on the dict backend |
| `test_integrate_quotient.py` | 25 | integrands that divide by a composite |
| `test_display.py` | 24 | the grade table, and the three states a reader must not miss |
| `test_truncation.py` | 13 | the dimension cap is a default; an explicit request overrides it |
| `test_explain.py` | 12 | `explain()` and `lead_order()`: what a function does at a point |
| `test_ieee754_projection.py` | 9 | `to_ieee754`: the one exit to a system with an additive identity |
| `test_composite_metadata.py` | 7 | one number carrying data and metadata |
| `test_written_zero_disclosure.py` | 6 | a written zero discloses itself however it is spelled |

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
