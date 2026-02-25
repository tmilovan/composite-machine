# docs/ROADMAP.md

# Composite Machine — What Works, What Doesn't, What's Next

An honest accounting of where the project stands.

---

## What works

These are tested, stable, and used regularly.

### Core arithmetic

Numbers are sparse dicts over integer dimensions. Multiplication is polynomial multiplication, division polynomial division. This follows from Laurent polynomial ring theory — nothing exotic about the arithmetic itself.

- Addition, subtraction, negation
- Multiplication
- Division — single-term and multi-term (polynomial long division)
- Integer and real-valued powers

### Differentiation

All derivatives from a single evaluation. Evaluate `f(a + ε)`, read the coefficients at negative dimensions. No tape, no graph, no separate differentiation pass.

- First through nth derivative
- All derivatives simultaneously
- Taylor coefficients as direct coefficient reads
- Chain rule and product rule handled automatically by dimensional convolution

### Limits

Plug in the infinitesimal or infinity, read the standard part. No L'Hôpital needed — composite division resolves indeterminate forms directly.

- Limits at a point, at zero, at ±∞
- One-sided limits
- 0/0 and other indeterminate forms

### Transcendental functions

All implemented via Taylor series on composite numbers. Derivatives come free.

- sin, cos, tan, exp, ln, sqrt
- asin, acos, atan
- sinh, cosh, tanh

### Integration

Works well for a wide range of functions. Adaptive stepping uses higher-order Taylor coefficients for error estimates.

- Definite integrals
- Improper integrals (infinite bounds)
- Adaptive integration with automatic step-size control
- Multi-dimensional integration

168 tests cover all of the above. They all pass.

### Provenance-preserving operations

This is the part that's genuinely new. Multiplying by zero doesn't destroy information — it shifts it to dimension −1. Dividing by zero shifts it to dimension +1. You can recover the original value by reversing the operation.

- `a × 0` → coefficient preserved at dim −1
- `a / 0` → coefficient preserved at dim +1
- `(a × 0) / 0` → recovers `a`
- `0 / 0` → resolves to `1` (provenance-dependent)
- Arbitrary chains of ×0 and ÷0 preserve and recover values

---

## What works but needs more testing

Implemented and functional, but edge cases and numerical stability haven't been fully explored.

### Multivariable calculus

Extends the algebra using tuple dimensions.

- Partial derivatives (first and higher order)
- Mixed partials
- Gradient, Laplacian
- Hessian, Jacobian
- Divergence, curl

### Vector calculus

- Line integrals
- Surface integrals
- Triple integrals

### Complex analysis

Same arithmetic, complex coefficients.

- Residue computation
- Pole detection
- Contour integrals via residue theorem
- Convergence radius estimation
- Analytic continuation (path stepping through convergence disks)

### ODE solving

- RK4 with composite evaluation
- Works for basic problems, not rigorously validated

### Asymptotic analysis

- Asymptotic expansion at infinity
- Growth order detection

---

## What's not implemented yet

- Inverse hyperbolics (asinh, acosh, atanh)
- Fourier, Laplace, Z transforms
- Special functions (Bessel, gamma, etc.)
- Optimization routines
- Stiff ODE solvers
- Systems of ODEs
- Branch cut handling in analytic continuation

---

## Performance

Pure Python, dict-based sparse storage. Roughly 500–1000× slower than PyTorch for simple gradients.

This is a research prototype. It's useful for problems where having all derivative orders, algebraic limits, or provenance matters more than throughput. It's not useful for production numerical computing — not yet.

---

## Roadmap

In rough priority order:

1. **Validate experimental modules** — systematic edge-case testing for multivariable, complex analysis, vector calculus, and ODE solving
2. **Missing transcendentals** — inverse hyperbolics, special functions
3. **Performance** — extending numpy support.
4. **Transforms** — Fourier, Laplace. These should map naturally onto the dimensional structure.
5. **Better ODE support** — implicit methods for stiff systems, adaptive order selection

### What's not on the roadmap

There's a theoretical question about whether this structure can encode a universal Turing machine (dimensions as tape positions, coefficients as symbols, ×0 as tape shift). It's an interesting idea but it's not something we're focusing on. If someone wants to try it, contributions are welcome. There is a basic set of tests in Turing playground that surprisingly pass, but this needs more scrutiny.

---

## The underlying idea

Every operation listed above uses the same mechanism: evaluate a function on a composite number, read coefficients at the right dimensions.

- Dimension 0 → value
- Dimension −n → nth derivative coefficient (× n!)
- Dimension +n → antiderivative / growth structure
- Dimension −1 (complex) → residue

One algebraic structure. Multiple mathematical readings. That's the whole idea.

---

© Toni Milovan. Documentation licensed under CC BY-SA 4.0. Code licensed under AGPL-3.0.