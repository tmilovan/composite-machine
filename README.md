# README.md

> [!NOTE]
> ## 📢 LONG DISCLAIMER, PLEASE READ.
>
> <details>
> <summary><strong>Let me address a few things here :)</strong>
>
> **1.** There has been a number of reactions on social media that this is some kind of AI slop project and while AI is used to do a cleanup on this repo and maintain the docs, and the code in at least usable and readable shape, **the main data structure, it's behavior, weird exceptions rules, main class, subclasses, implementation etc. are genuinely my work and based on my implementation and interpretation.** If anything AI has been more of a burden in its conception **because this system's artificial/made-up rules go against concepts LLM models have been trained with.** Click to read more.</summary>
>
> **2.** So why use AI at all? Well because **there is only one ME working on this project** part time. I can't do all the work needed to maintain public project of such a scale, which is changing so rapidly without help. **AI has been very useful for testing and rapid prototyping here.**
>
> **3.** So what about paper? Yes, what about it? **The sole purpose of the paper is to provide explanation how the system works and that I have done basic homework and I'm not claiming I have invented already known things or methods**. I have tried to do it to the best of my abilities. I'm not career academician producing scientific papers for life and career advancement, neither am I software scientist meticulously researching and mapping down information science achievements.
>
> **I'm a software engineer trying to figure out what have I stumbled upon and to explain what I have noticed the best way I can.** Of course I have used AI to help me structure, shorten and clarify the document, **I wouldn't know how to do it properly without it.** The content is what matters and content is genuine. **So READ IT please before making any complaints about it.**
>
> **4.** And finally, what is this library here? **This is the result of my work on specific problem in audio domain. While I was trying to (unsuccessfully) resolve very specific computation problem I had created (designed?) data structure and a set of rules to calculate with it for which I hoped it will help me solve my problem.**
>
> **While working with it I have noticed some peculiar behaviors that led me to conclusion I don't need specific solution because I can do integrations, derivatives and limits directly with it (structure+rules). At the end it turned out I can do a lot more with it.** I don't claim I have invented some new mathematics here, but I have developed an implementation/interpretation of already known parts in a way I have not seen being done yet. If you have any examples I'll be eager to know about them, but believe me I've did my due diligence on it.
>
> **5.** And lastly, once I figured out I have a generic tool with potential usefulness for broader public I have created this repo to share it. **I don't know what this tool can do at the end. For now I know that I have managed to implement a wide array of math operations in it in very (unrealistically?) short time.** They seems to work in test and in controlled environments. I don't know of any other system where you can do so many different math operations inside singular platform/paradigm. I don't know what this means. **I know there are some similar systems but none can do all of this in one place.** That's why I'm using this annoyingly new notation, because everything seems work only with it.
>
> **6.** My goal now is to find practical applications and what are the limits of it. If you are interested, please join, fork and play with it. **The system is real, basic computations work, we just don't know yet for how many cases is this true and that is the goal of this journey here.** If you find this interesting clone it and try it. I'm eager to get your feedback.
>
> Thank you :)
>
> </details>

# Composite Machine: Automatic Calculus (and more) via Dimensional Arithmetic

⚠️ **ALPHA VERSION** - Research code under active development. API may change. Performance optimization in progress.

[![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)]()

> **What if all derivatives, integrals, and limits were just algebraic operations on a single number?**

This is a working implementation of **composite arithmetic** — a number system where calculus operations reduce to coefficient manipulation. No symbolic engines, no computation graphs, just algebra.

---

## The Idea in 30 Seconds

```python
from composite.composite_lib import R, ZERO

# Traditional: Need N function evaluations for N derivatives
# Composite: ONE evaluation → ALL derivatives

x = R(3) + ZERO  # 3 + infinitesimal
result = x**4     # Compute once

print(result.d(1))  # 108 ← First derivative
print(result.d(2))  # 108 ← Second derivative
print(result.d(10)) # ← 10th derivative!

# All extracted from the SAME evaluation
```

**Key insight:** Represent numbers with "dimensional structure" where negative dimensions encode derivative information. Calculus becomes coefficient extraction.

---

## Library Modules

The library is organized into four files, each extending the core system:

| Module | Purpose | Key features |
|--------|---------|-------------|
| `composite_lib.py` | Core engine | Composite class, all arithmetic, transcendentals, derivatives, limits, integration, antiderivative |
| `composite_multivar.py` | Multivariable calculus | MC class (tuple dimensions), partial derivatives, gradient, Hessian, Jacobian, Laplacian, divergence, curl, double integrals |
| `composite_extended.py` | Complex analysis | Complex composites, residues, poles, contour integrals, asymptotics, convergence radius, ODE solver, analytic continuation |
| `composite_vector.py` | Vector calculus | Triple integrals, line integrals (scalar and vector), surface integrals (scalar and flux) |

---

## What Works Now ✅

### Core Arithmetic (`composite_lib.py`)
- ✅ **Full arithmetic** — +, −, ×, ÷ with dimensional convolution/deconvolution
- ✅ **Integer and real-exponent powers** — `x**n` and `power(x, r)`
- ✅ **Division by zero is defined** — reversible operations: `(5×0)/0 = 5`
- ✅ **0/0 = 1** — well-defined via dimensional cancellation
- ✅ **∞ × 0 = 1** — zero-infinity duality
- ✅ **Comparison operators** with NaN handling and total ordering across dimensions
- ✅ **TracedComposite** for step-by-step operation tracing

### Transcendental Functions (`composite_lib.py`)
- ✅ **Trigonometric** — `sin`, `cos`, `tan`
- ✅ **Inverse trigonometric** — `atan`, `asin`, `acos`
- ✅ **Hyperbolic** — `sinh`, `cosh`, `tanh`
- ✅ **Exponential and logarithmic** — `exp`, `ln`
- ✅ **Other** — `sqrt`, `power` (real exponents)

### Derivatives (`composite_lib.py`)
- ✅ **All-order derivatives** from single evaluation — `d(n)`, `derivative()`, `nth_derivative()`
- ✅ **All derivatives at once** — `all_derivatives()`, `taylor_coefficients()`
- ✅ **Derivative verification** — `verify_derivative()`

### Limits (`composite_lib.py`)
- ✅ **Algebraic limits** — no L'Hôpital needed, just substitute and read
- ✅ **One-sided limits** — `limit_left()`, `limit_right()`
- ✅ **Limits at infinity**

### Integration (`composite_lib.py` + extensions)
- ✅ **Unified `integrate()` wrapper** — single entry point for 1D, 2D, 3D, line, and surface integrals
- ✅ **Adaptive integration** with automatic error estimates — `integrate_adaptive()`
- ✅ **Improper integrals (experimental)** — handles `±∞` bounds and singularities
- ✅ **Antiderivative** via dimensional shift
- ✅ **Double integrals (experimental)** — `double_integral()` (in `composite_multivar.py`)
- ✅ **Triple integrals (experimental)** — `triple_integral()` (in `composite_vector.py`)
- ✅ **Line integrals (experimental)** — scalar and vector field: `line_integral_scalar()`, `line_integral_vector()`
- ✅ **Surface integrals (experimental)** — scalar and flux: `surface_integral_scalar()`, `surface_integral_vector()`

### Multivariable Calculus (`composite_multivar.py`)
- ✅ **MC class  (experimental)** — multi-composite with tuple dimensions, full arithmetic
- ✅ **Partial derivatives  (experimental)** — `partial_derivative()`
- ✅ **Differential operators  (experimental)** — `gradient_at()`, `hessian_at()`, `jacobian_at()`, `laplacian_at()`
- ✅ **Vector operators (experimental)** — `divergence_of()`, `curl_at()`, `directional_derivative()`
- ✅ **Multivariate limits (experimental)** — `multivar_limit()`
- ✅ **Multivariate transcendentals (experimental)** — `mc_sin`, `mc_cos`, `mc_exp`, `mc_ln`, `mc_sqrt`, `mc_tan`, `mc_power`

### Complex Analysis (`composite_extended.py`)
- ✅ **Complex composites (experimental)** — `C()`, `C_var()`, `cexp()`, `csin()`, `ccos()`
- ✅ **Residue computation (experimental)** and **pole detection** — `residue()`, `pole_order()`
- ✅ **Contour integrals  (experimental)** via residue theorem — `contour_integral()`
- ✅ **Asymptotic expansion (experimental)** — `asymptotic_expansion()`, `limit_at_infinity()`, `asymptotic_order()`
- ✅ **Convergence radius (experimental)** — generalized ratio test + root test
- ✅ **ODE solver  (experimental)** — RK4 via composite evaluation
- ✅ **Analytic continuation  (experimental)** and **singularity detection**

---

## What Doesn't Work Yet ❌

- ❌ **Inverse hyperbolics** — `asinh`, `acosh`, `atanh` not yet implemented
- ❌ **Stokes'/Divergence/Green's theorem wrappers** — differential operators exist (`curl_at`, `divergence_of`) but no theorem-level verification functions
- ❌ **Fourier/Laplace/Z transforms**
- ❌ **Optimization routines** — gradient descent, Newton's method using composite derivatives
- ❌ **Piecewise function support** — explored separately but not in the library
- ❌ **Special functions** — Bessel, gamma, etc.
- ❌ **Performance** — ~500-1000× slower than PyTorch (dict-based implementation)
- ❌ **API stability** — may change before v1.0

**But:** The math works. All 168 tests pass at 100%. Optimization is in progress.

---

## Installation

~~~bash
# From source (only option for now)
git clone https://github.com/tmilovan/composite-machine.git
cd composite-machine
pip install -e .
~~~

**Requirements:** Python 3.7+, NumPy (optional, for FFT-accelerated multiplication)

---

## Quick Examples

### Derivatives (The Headline Feature)

~~~python
from composite_lib import derivative, nth_derivative, all_derivatives, exp

# Simple API
derivative(lambda x: x**2, at=3)  # → 6

# Any order
nth_derivative(lambda x: x**5, n=3, at=2)  # → 120

# All at once
all_derivatives(lambda x: exp(x), at=0, up_to=5)
# → [1, 1, 1, 1, 1, 1]  (all derivatives of e^x)
~~~

### Limits (No L'Hôpital Needed)

~~~python
from composite_lib import limit, sin, R

limit(lambda x: sin(x)/x, as_x_to=0)  # → 1.0
limit(lambda x: (x**2 - R(4))/(x - R(2)), as_x_to=2)  # → 4.0
limit(lambda x: (R(5)*x**2+R(3)*x)/(R(2)*x**2+R(1)),
      as_x_to=float('inf'))  # → 2.5
~~~

### Integration (Unified API)

~~~python
from composite_lib import integrate, exp, sin
import math

# 1D definite integral
integrate(lambda x: x**2, 0, 1)  # → 0.333...

# 1D with error estimate
integrate(lambda x: exp(-(x*x)), 1, 2)  # → 0.1353 (error estimate is FREE)

# Improper integral (∞ bounds)
integrate(lambda x: exp(-x), 0, float('inf'))  # → 1.0

# 2D integral
integrate(lambda x, y: x*y, 0, 1, 0, 1)  # → 0.25

# Line integral along a curve
integrate(lambda x, y: x + y, curve=lambda t: [t, t], t_range=(0, 1))

# Surface integral
integrate(f, surface=parametrization, u_range=(0, math.pi), v_range=(0, 2*math.pi))
~~~

### Division by Zero (Yes, Really)

~~~python
from composite_lib import ZERO, R

ZERO / ZERO  # → 1 (well-defined!)
(R(5) * ZERO) / ZERO  # → 5 (reversible!)
(R(7) * ZERO * ZERO) / ZERO / ZERO  # → 7 (multi-depth recovery!)
~~~

### Multivariable Calculus

~~~python
from composite_multivar import MC, RR, RR_const, gradient_at, laplacian_at

# Gradient of f(x,y) = x² + y² at (3, 4)
gradient_at(lambda x, y: x**2 + y**2, [3, 4])  # → [6, 8]

# Laplacian of f(x,y) = x² + y²
laplacian_at(lambda x, y: x**2 + y**2, [3, 4])  # → 4
~~~

### Complex Analysis

~~~python
from composite_extended import residue, contour_integral, convergence_radius

# Residue of 1/z at z=0
residue(lambda z: 1/z, at=0)  # → 1.0

# Convergence radius of a series
convergence_radius(lambda z: 1/(1 - z), at=0)  # → 1.0
~~~

---

## How Is This Different?

| Feature | PyTorch/JAX | SymPy | Dual Numbers | **Composite** |
|---------|-------------|-------|--------------|---------------|
| All-order derivatives | ❌ | ✅ | ❌ (1st only) | ✅ |
| One evaluation | ✅ | ❌ | ✅ | ✅ |
| Division by zero | ❌ | ❌ | ❌ | ✅ |
| Algebraic limits | ❌ | ✅ | ❌ | ✅ |
| Integration + AD | ❌ | ✅ | ❌ | ✅ |
| Multivariable calculus | ✅ (grad only) | ✅ | ❌ | ✅ |
| Complex analysis | ❌ | ✅ | ❌ | ✅ |
| Vector calculus | ❌ | partial | ❌ | ✅ |
| Fast | ✅ | ❌ | ✅ | ❌ (yet) |

**Unique combo:** All derivatives + integration + limits + zero handling + complex analysis + vector calculus in ONE algebraic structure.

---

## The Core Idea (For the Curious)

### Traditional calculus = Algorithms
- Derivative → Build computation graph, apply chain rule
- Integral → Pattern matching, special cases
- Limit → L'Hôpital's rule, case analysis

### Composite arithmetic = Algebra
- Derivative → Read coefficient at dimension −n
- Integral → Dimensional shift + adaptive stepping
- Limit → Substitute infinitesimal, take standard part

**Example:**

~~~python
x = R(2) + ZERO  # 2 + infinitesimal h
result = x**4    # (2+h)⁴ expanded via polynomial arithmetic

# Result encodes: |16|₀ + |32|₋₁ + |24|₋₂ + |8|₋₃ + |1|₋₄
#                  ↑       ↑        ↑        ↑        ↑
#                 f(2)   f'(2)/1!  f''(2)/2! f'''(2)/3! f⁴(2)/4!

result.st()   # 16   ← Function value
result.d(1)   # 32   ← First derivative (32 × 1!)
result.d(2)   # 48   ← Second derivative (24 × 2!)
result.d(3)   # 48   ← Third derivative (8 × 3!)
~~~

**All derivatives emerge from polynomial convolution.** No separate algorithm needed!

---

## Testing

~~~bash
# Run all tests
python test_composite.py                # ~105 tests — core + calculus + algebra
python composite_stress_test.py         # 20 hard problems (limits, derivatives, integrals)
python composite_hard_edges.py          # 20 hard edge cases (3rd/4th order, deep chains)
python any_test_file.py                 # evergrowing test suite
~~~

**All 168 tests pass at 100%.**

### Test Coverage

| **Category** | **Tests** | **What's validated** |
|---|---|---|
| Paper Theorems (T1-T8) | ~45 | Information preservation, zero-infinity duality, provenance, reversibility, cancellation, identity, fractional orders, total ordering |
| Algebraic Properties | ~8 | Associativity, commutativity, distributivity, negation |
| Derivatives | ~21 | Polynomials, transcendentals, chain rule, Leibniz rule, 2nd-6th order, compositions |
| Limits | ~20 | sin(x)/x, indeterminate forms, 3rd/4th order cancellations, nested compositions, limits at ∞ |
| Integration | ~17 | Definite, improper, products, trig powers, Gaussian, adaptive error estimates |
| Zero/Infinity | ~10 | 0/0 provenance, reversibility chains, deep dimension recovery |
| Transcendentals | ~6 | sin, cos, exp identities and derivatives |
| Multi-term Division | ~7 | Polynomial long division, rational functions |
| Multivariate | ~5 | Partial derivatives, gradient, Laplacian, harmonic functions |
| Edge Cases & Stress | ~16 | Deep chains, tiny/huge coefficients, FFT vs Dict cross-check, boundary conditions |
| Standard Python Validation | ~23 | Cross-check against numerical differentiation, numerical limits, math.* functions |

---

## Project Status & Roadmap

### Current State (v0.1-alpha)
- ✅ Core single-variable calculus (derivatives, limits, integration)
- ✅ Full transcendental library (trig, inverse trig, hyperbolic, exp, ln)
- ✅ Multivariable calculus (gradient, Hessian, Jacobian, Laplacian, curl, divergence)
- ✅ Vector calculus (line integrals, surface integrals, triple integrals)
- ✅ Complex analysis (residues, poles, contour integrals, asymptotics, convergence)
- ✅ Unified `integrate()` API across all integral types
- ✅ 168 tests passing at 100%
- ⚠️ Performance is SLOW (research code)
- ⚠️ API may change

### Next Steps (v0.2)
- 🚧 Inverse hyperbolics (asinh, acosh, atanh)
- 🚧 Vectorization with NumPy (target: 10× speedup)
- 🚧 JIT compilation with Numba (target: 50× speedup)
- 🚧 Theorem-level verification (Stokes', Divergence, Green's)
- 🚧 Optimization routines (gradient descent, Newton's method)
- 🚧 API stabilization

### Future (v1.0)
- 🔮 GPU support (CuPy/JAX backend)
- 🔮 Fourier/Laplace transforms
- 🔮 Special functions (Bessel, gamma)
- 🔮 Production-ready performance
- 🔮 PyPI package

---

## Documentation

- [**10-Minute Tutorial**](docs/Tutorial%20-%20Getting%20Started.md) - Get started quickly
- [**API Reference**](docs/API%20Reference.md) - Complete function docs
- [**Implementation Guide**](docs/Implementation%20Guide.md) - How it works internally
- [**Examples**](docs/Examples.md) - Code snippets for common tasks
- [**Roadmap (DRAFT)**](docs/Roadmap%20(DRAFT).md) - What's next
---

## Testing

```bash
# Run specific test suites (all should pass)
python tests/test_filename.py
```

**Test coverage:**

- Core arithmetic (20 tests) - Addition, multiplication, division
- Zero/infinity (15 tests) - 0/0, ∞×0, reversibility
- Derivatives (20 tests) - All orders, product rule, chain rule
- Limits (15 tests) - Indeterminate forms, infinity
- Integration (15 tests) - Definite, improper, singularities
- Transcendentals (15 tests) - Trig, exponential, inverse
- Theorems (5 tests) - Formal validation of claims

---

## Theory & Papers

📄 **Preprint (coming soon):** "Provenance-Preserving Arithmetic: A Unified Framework for Automatic Calculus"

[Milovan, T. (2026). Provenance-Preserving Arithmetic. Zenodo.](https://doi.org/10.5281/zenodo.18528788)

**Core insight:** Reinterpret Laurent polynomials where z⁻¹ represents "zero with provenance" — an infinitesimal that remembers its origin. This single reinterpretation makes calculus algebraic.

**Key results:**

- **Theorem 1:** Information preservation under ×0
- **Theorem 2:** Zero-infinity duality (∞ × 0 = 1)
- **Theorem 3:** Reversible zero operations
- **Theorem 4:** Derivatives emerge from convolution (no separate rules needed)

*Formal proofs available in `papers/` directory.*

---

## Limitations & Caveats (Read This!)

### Not a Drop-In Replacement

- Standard code expects `0 + 0 = 0`, but here `ZERO + ZERO = |2|₋₁`
- Modified semantics require explicit handling
- Not suitable for general-purpose arithmetic

### Performance

- **~1000× slower than PyTorch** for simple gradients (pure Python)
- Competitive for: high-order derivatives, integration, meta-optimization
- Use PyTorch for production ML training
- Use this for: research, prototyping, second-order methods

### Function Coverage

- Common transcendentals: ✅ (sin, cos, exp, ln, etc.)
- Special functions: ❌ (Bessel, gamma, etc. - not yet)
- Custom functions: Requires Taylor series expansion

### When to Use This

✅ Research projects needing all-order derivatives

✅ Sensitivity analysis with Hessian information

✅ Numerical methods with automatic error bounds

✅ Exploring novel approaches to automatic differentiation

❌ Performance-critical code (not optimized yet)

❌ Production use (this is research code)

---

## Contributing

**We welcome contributions!** This is an early-stage research project.

**High-priority areas:**

- Performance optimization (vectorization, GPU, JIT)
- Additional special functions (Bessel, gamma, etc.)
- Improved documentation & examples
- Bug reports & edge cases
- Novel applications of composite arithmetic

**Process:**

1. Open an issue to discuss your idea
2. Fork the repo
3. Make your changes
4. Add tests
5. Submit a pull request

---

## Citation

If you use this in research, please cite:

```
@software{milovan2026composite,
  author = {Milovan, Toni},
  title = {Composite Machine: Automatic Calculus via Dimensional Arithmetic},
  year = {2026},
  url = {https://github.com/tmilovan/composite-machine}
}
```

---

## License

### Code

**AGPL-3.0** — Free for open source, research, and personal use. Modifications must be shared under the same license.

For use in proprietary or closed-source software, a **commercial license** is available.

Contact: [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr)

See [LICENSE](LICENSE) for the full AGPL-3.0 text.

### Papers

The accompanying papers ("Provenance-Preserving Arithmetic" and "Composite Calculus Machine") are licensed under **CC BY 4.0**.

---

## Author

**Toni Milovan**

Independent Researcher

Pula, Croatia

📧 [tmilovan@fwd.hr](mailto:tmilovan@fwd.hr)

---

## Acknowledgments

This work builds on:

- **Laurent polynomial algebra** - Mathematical foundation
- **Non-standard analysis** (Robinson) - Infinitesimals as rigorous objects
- **Automatic differentiation** (Wengert, Griewank) - Forward-mode AD inspiration
- **Wheel theory** (Carlström) - Division by zero approaches

**Key innovation:** Treating z⁻¹ as "zero with provenance" unifies calculus operations into a single algebraic structure.

---

## FAQ

**Q: Is this production-ready?**

A: No. It's alpha research code. Performance is ~1000× slower than PyTorch. Use for exploration, not production.

**Q: Will 0/0 = 1 break my code?**

A: ZERO is a special infinitesimal (|1|₋₁), not Python's `0`. Regular Python arithmetic is unaffected.

**Q: Can I use this with PyTorch?**

A: Not yet, but it's on the roadmap. Currently standalone.

**Q: Why is it so slow?**

A: Pure Python with dict-based sparse representation. Vectorization + GPU will bring ~500-1000× speedup.

**Q: What's the best use case TODAY?**

A: Research and prototyping where you need all-order derivatives, algebraic limits, or integration with automatic error bounds.

---

## Star ⭐ this repo if you find it interesting!

**Have questions?** Open an issue

**Found a bug?** Please report it!

---

**Built with curiosity. Shared for science. Use with caution.** 🚀

© Toni Milovan. Documentation licensed under CC BY-SA 4.0. Code licensed under AGPL-3.0.
