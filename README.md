# README.md

# Composite Machine: Automatic Calculus via Dimensional Arithmetic

⚠️ **ALPHA VERSION** - Research code under active development. API may change. Performance optimization in progress.

[![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen)]()

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)]()

> **What if all derivatives, integrals, and limits were just algebraic operations on a single number?**
>

This is a working implementation of **composite arithmetic** — a number system where calculus operations reduce to coefficient manipulation. No symbolic engines, no computation graphs, just algebra.

---

## The Idea in 30 Seconds

```python
from composite_lib import R, ZERO

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

## What Works Now ✅

- ✅ **All-order derivatives** from single evaluation (not just 1st or fixed order)
- ✅ **Division by zero is defined** - reversible operations (5×0)/0 = 5
- ✅ **Algebraic limits** - no L'Hôpital's rule, just substitute & read
- ✅ **Adaptive integration** with automatic error estimates (free!)
- ✅ **Improper integrals** - handles ∞ bounds and singularities
- ✅ **Full transcendental library** - sin, cos, exp, ln, inverse trig, hyperbolic
- ✅ **FFT-accelerated multiplication** via `CompositeFFT` (NumPy backend)
- ✅ **175 passing tests** validating all claims

---

## What Doesn't Work Yet ❌

- ❌ **Performance**: ~500-1000× slower than PyTorch (dict-based implementation; FFT version is faster but not yet fully optimized)
- ❌ **API stability**: May change before v1.0
- ❌ **Production ready**: This is research code, use at own risk

**But:** The math works. The tests pass. Optimization is in progress (vectorization, GPU, JIT).

---

## Installation

```bash
# From source (only option for now)
git clone https://github.com/tmilovan/composite-machine.git
cd composite-machine
pip install -e .
```

**Requirements:** Python 3.7+, NumPy (that's it!)

---

## Quick Examples

### Derivatives (The Headline Feature)

```python
from composite_lib import derivative, nth_derivative, all_derivatives

# Simple API
derivative(lambda x: x**2, at=3)  # → 6

# Any order
nth_derivative(lambda x: x**5, n=3, at=2)  # → 120

# All at once
all_derivatives(lambda x: exp(x), at=0, up_to=5)
# → [1, 1, 1, 1, 1, 1]  (all derivatives of e^x)
```

### Limits (No L'Hôpital Needed)

```python
from composite_lib import limit

limit(lambda x: sin(x)/x, as_x_to=0)  # → 1.0
limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)  # → 4.0
limit(lambda x: (3*x + 1)/(x + 2), as_x_to=float('inf'))  # → 3.0
```

### Integration (With Error Estimates)

```python
from composite_lib import integrate_adaptive

val, err = integrate_adaptive(lambda x: exp(-(x*x)), 1, 2)
# val ≈ 0.1353, err ≈ 1e-15 (error estimate is FREE!)
```

### Division by Zero (Yes, Really)

```python
from composite_lib import ZERO, R

ZERO / ZERO  # → 1 (well-defined!)
(R(5) * ZERO) / ZERO  # → 5 (reversible!)
```

---

## How Is This Different?

| Feature | PyTorch/JAX | SymPy | Dual Numbers | **Composite** |
| --- | --- | --- | --- | --- |
| All-order derivatives | ❌ | ✅ | ❌ (1st only) | ✅ |
| One evaluation | ✅ | ❌ | ✅ | ✅ |
| Division by zero | ❌ | ❌ | ❌ | ✅ |
| Algebraic limits | ❌ | ✅ | ❌ | ✅ |
| Integration + AD | ❌ | ✅ | ❌ | ✅ |
| Fast | ✅ | ❌ | ✅ | ❌ (yet) |

**Unique combo:** All derivatives + integration + zero handling in ONE algebraic structure.

---

## The Core Idea (For the Curious)

### Traditional calculus = Algorithms

- Derivative → Build computation graph, apply chain rule
- Integral → Pattern matching, special cases
- Limit → L'Hôpital's rule, case analysis

### Composite arithmetic = Algebra

- Derivative → Read coefficient at dimension -n
- Integral → Dimensional shift + adaptive stepping
- Limit → Substitute infinitesimal, take standard part

**Example:**

```python
x = R(2) + ZERO  # 2 + infinitesimal h
result = x**4    # (2+h)⁴ expanded via polynomial arithmetic

# Result encodes: |16|₀ + |32|₋₁ + |24|₋₂ + |8|₋₃ + |1|₋₄
#                  ↑       ↑        ↑        ↑        ↑
#                 f(2)   f'(2)/1!  f''(2)/2! f'''(2)/3! f⁴(2)/4!

result.st()   # 16   ← Function value
result.d(1)   # 32   ← First derivative (32 × 1!)
result.d(2)   # 48   ← Second derivative (24 × 2!)
result.d(3)   # 48   ← Third derivative (8 × 3!)
```

**All derivatives emerge from polynomial convolution.** No separate algorithm needed!

---

## Project Status & Roadmap

### Current State (v0.1-alpha)

- ✅ Core calculus working
- ✅ Comprehensive test suite (175 tests)
- ✅ Documentation & examples
- ⚠️ Performance is SLOW (research code)
- ⚠️ API may change

### Next Steps (v0.2)

- 🚧 Vectorization with NumPy (target: 10× speedup)
- 🚧 JIT compilation with Numba (target: 50× speedup)
- 🚧 More examples and tutorials
- 🚧 API stabilization
- 🚧 Practical application demos

### Future (v1.0)

- 🔮 GPU support (CuPy/JAX backend)
- 🔮 Production-ready performance
- 🔮 Framework integrations

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
# Run all tests (175 tests, should all pass)
pytest tests/

# Run specific test suites
pytest tests/test_core.py         # Core algebra
pytest tests/test_calculus.py     # Derivatives, limits, integrals
pytest tests/test_transcendental.py  # sin, exp, ln, etc.
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
