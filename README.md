# README.md


# Composite Machine

**Automatic calculus via dimensional arithmetic.**

A data structure and a set of arithmetic rules that give you derivatives, integrals, and limits as a side effect of normal computation. No symbolic engine, no computation graph, no tape. Tag a number, do your math, read the results off the dimensional coefficients.

> *1 − 1 ≠ 0*
>

> *The residue is infinitesimal, structured, and it contains the derivative of every operation that produced it.*
>

Alpha stage. Research code. The math works, performance doesn't (yet).

---

## What's this

Numbers are sparse dicts mapping integer dimensions to coefficients. Dimension 0 is the value. Negative dimensions store derivative info. Multiplication convolves dimensions — turns out that's the same thing as the product rule and chain rule, just expressed as data structure operations.

```python
from composite.composite_lib import R, ZERO

x = R(3) + ZERO          # 3 + infinitesimal seed
result = x ** 4           # just compute normally

result.st()               # 81  — the value, f(3)
result.d(1)               # 108 — first derivative
result.d(2)               # 108 — second derivative
result.d(3)               # 24  — third derivative
result.d(4)               # 24  — fourth derivative
```

One evaluation. All derivatives fall out. No separate differentiation pass.

---

## Background

The derivative computation part builds on well-known work: Clifford's **dual numbers** (1873), Wengert's **forward-mode AD** (1964), Rall's **Taylor arithmetic** (1981), Griewank's framework (2000).

What this library explores is a different algebraic context for that mechanism. Higher-order terms are preserved instead of truncated. Subtraction retains provenance instead of collapsing to zero. Multiplication by zero shifts structure instead of destroying it. The idea is that if you stop throwing away information at each step, calculus operations become extractable from the algebra.

Does this generalize to everything? Open question. The test suite covers a wide range of standard problems and the results match. Finding the boundaries is the point of this project.

For the theoretical framing, see the paper.

---

## How it compares

The trade-off is breadth vs speed. This covers a lot of operations in one structure, but it's slow.

- **vs PyTorch/JAX** — They're fast but give you first-order gradients. This gives you all orders, plus limits and integration, but is ~1000x slower.
- **vs SymPy** — SymPy does symbolic math. This is numerical. SymPy is slow for large expressions. This is slow for everything, but conceptually simpler.
- **vs dual numbers** — Classic dual numbers give you one derivative (epsilon squared is zero). Here epsilon squared is kept, so you get all orders.

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

ZERO isn't Python's 0 — it's a structural infinitesimal, coefficient 1 at dimension −1. Operations on it are well-defined and reversible:

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

- **[composite_lib.py](composite/composite_lib.py)** — Core engine. Composite class, all arithmetic, transcendentals, derivatives, limits, integration.
- **[composite_multivar.py](composite/composite_multivar.py)** — Multivariable calculus. MC class, partial derivatives, gradient, Hessian, Jacobian, Laplacian, divergence, curl.
- **[composite_extended.py](composite/composite_extended.py)** — Complex analysis. Complex composites, residues, poles, contour integrals, asymptotics, ODE solver.
- **[composite_vector.py](composite/composite_vector.py)** — Vector calculus. Triple integrals, line integrals, surface integrals.

---

## What works

**Stable:**

- Full arithmetic with dimensional convolution and deconvolution
- Integer and real-exponent powers
- Transcendentals — sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, ln, sqrt
- All-order derivatives from a single evaluation
- Algebraic limits including indeterminate forms and limits at infinity
- Definite, improper, and adaptive integration with error estimates
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

Pure Python, dict-based sparse storage. Roughly 500–1000x slower than PyTorch for simple gradients.

Fine for research, prototyping, and problems where higher-order derivatives or algebraic limits matter more than throughput.

---

## Installation

```bash
git clone https://github.com/tmilovan/composite-machine.git
cd composite-machine
pip install -e .
```

Python 3.7+. NumPy is optional (used for FFT-accelerated multiplication).

---

## Testing

```bash
python test_composite.py                # ~105 tests — core + calculus + algebra
python composite_stress_test.py         # 20 hard problems (limits, derivatives, integrals)
python composite_hard_edges.py          # 20 hard edge cases (3rd/4th order, deep chains)
python any_test_file.py                 # evergrowing test suite
```

168 tests, all passing.

Covers: paper theorems, algebraic properties, derivatives (orders 1–6, chain rule, Leibniz), limits (indeterminate forms, infinity), integration (definite, improper, adaptive), zero/infinity handling, transcendentals, polynomial division, multivariable ops, and cross-checks against numerical differentiation and Python's math module.

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

---

## Contributing

Contributions welcome. Useful areas:

- Performance — vectorization, JIT, GPU backends
- Special functions — Bessel, gamma, etc.
- Bug reports and edge cases
- Docs and examples

Process: open an issue first, fork, add tests, PR.

---

## Citation

If you use this in research:

Milovan, T. (2026). Composite Machine: Automatic Calculus via Dimensional Arithmetic. [https://github.com/tmilovan/composite-machine](https://github.com/tmilovan/composite-machine)

---

## License

**Code:** AGPL-3.0. Free for open-source, research, and personal use. Commercial licensing available — contact [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr).

**Paper:** CC BY 4.0.

---

Toni Milovan · Pula, Croatia · [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr)
