# Implementation Guide

This guide explains how composite calculus works internally and how to implement it yourself.

---

## Table of Contents

- [Core Data Structure](#core-data-structure)
- [Dimensional System](#dimensional-system)
- [Arithmetic Operations](#arithmetic-operations)
- [How Derivatives Emerge](#how-derivatives-emerge)
- [Transcendental Functions](#transcendental-functions)
- [Integration](#integration)
- [Optimization Strategies](#optimization-strategies)

---

## Core Data Structure

### Sparse Dictionary Representation

Composite numbers are represented as a **sparse dictionary** mapping dimensions to coefficients:

```python
class Composite:
    def __init__(self, coefficients=None):
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            # Scalar: store at dimension 0
            self.c = {0: float(coefficients)} if coefficients != 0 else {}
        elif isinstance(coefficients, dict):
            # Filter out near-zero coefficients
            self.c = {k: v for k, v in coefficients.items() if abs(v) > 1e-15}
```

**Why sparse?** Most composite numbers have only a few non-zero dimensions. A polynomial x³ evaluated at x = 2 + h produces only 4 dimensions: {0: 8, -1: 12, -2: 6, -3: 1}.

**Memory:** O(k) where k = number of non-zero dimensions, typically 3-10.

---

## Dimensional System

### Dimension Semantics

| Dimension | Meaning | Example |
| --- | --- | --- |
| 0 | Real value | `{0: 5}` = 5 |
| -1 | First infinitesimal | `{-1: 1}` = h |
| -2 | Second infinitesimal | `{-2: 1}` = h² |
| -3 | Third infinitesimal | `{-3: 1}` = h³ |
| +1 | Infinity | `{1: 1}` = ∞ |

### Why Negative Dimensions?

Derivatives naturally appear at **negative dimensions** because:

- f(x + h) = f(x) + f'(x)·h + f''(x)·h²/2! + ...
- h → dimension -1
- h² → dimension -2
- Coefficient at -1 is the derivative (after factorial rescaling)

### Creating the Infinitesimal

```python
ZERO = Composite({-1: 1.0})  # |1|₋₁
```

This is the **structural zero** that enables reversible multiplication:

- 5 × ZERO = `{-1: 5}` (not 0!)
- The coefficient 5 is preserved at dimension -1

---

## Arithmetic Operations

### Addition: Dimension Matching

Coefficients at **matching dimensions** are added:

```python
def __add__(self, other):
    result = dict(self.c)  # Copy self
    for dim, coeff in other.c.items():
        result[dim] = result.get(dim, 0) + coeff
    return Composite(result)
```

**Example:**

```
  {0: 3, -1: 2}  (3 + 2h)
+ {0: 1, -1: 4}  (1 + 4h)
= {0: 4, -1: 6}  (4 + 6h)
```

---

### Multiplication: Convolution

Dimensions **add**, coefficients **multiply** (convolution):

```python
def __mul__(self, other):
    result = {}
    for d1, c1 in self.c.items():
        for d2, c2 in other.c.items():
            dim = d1 + d2  # Dimensions add!
            result[dim] = result.get(dim, 0) + c1 * c2
    return Composite(result)
```

**Why this works:**

```
(a + bh)(c + dh) = ac + (ad + bc)h + bdh²

Dimension view:
  a at dim 0, b at dim -1
  c at dim 0, d at dim -1

Products:
  a×c → dim 0+0=0,   coeff a*c
  a×d → dim 0+(-1)=-1, coeff a*d
  b×c → dim (-1)+0=-1, coeff b*c
  b×d → dim (-1)+(-1)=-2, coeff b*d
```

**This IS the Leibniz product rule!** The convolution formula for polynomial multiplication is **identical** to the formula for derivatives of products.

---

### Division: Polynomial Long Division

For single-term divisors (fast path):

```python
def __truediv__(self, other):
    if len(other.c) == 1:
        div_dim, div_coeff = list(other.c.items())[0]
        result = {}
        for dim, coeff in self.c.items():
            result[dim - div_dim] = coeff / div_coeff  # Dimensions subtract
        return Composite(result)
```

For multi-term divisors, use polynomial long division.

**Why it works:**

```
(5h) / h = 5    (dimensions: -1 - (-1) = 0)
(12h²) / (2h) = 6h    (dimensions: -2 - (-1) = -1)
```

---

## How Derivatives Emerge

### The Magic of Taylor Expansion

When you evaluate f(x + h):

```python
x = R(3) + ZERO  # 3 + h
result = x**2    # (3 + h)²
```

You're computing the **Taylor series**:

```
(3 + h)² = 9 + 6h + h²
```

In composite representation:

```
{0: 9, -1: 6, -2: 1}
```

### Extracting Derivatives

The coefficient at dimension -n gives f⁽ⁿ⁾(a) / n!:

```python
def d(self, n=1):
    """Extract nth derivative"""
    return self.c.get(-n, 0.0) * math.factorial(n)
```

**Why factorial?** Taylor series has factorials:

```
f(a+h) = f(a) + f'(a)h/1! + f''(a)h²/2! + f'''(a)h³/3! + ...
```

So:

- Coefficient at -1 = f'(a) / 1! → multiply by 1! to get f'(a)
- Coefficient at -2 = f''(a) / 2! → multiply by 2! to get f''(a)

---

## Transcendental Functions

### Strategy: Taylor Series Expansion

Since we can't "multiply sin(x) by h" directly, we expand sin as a Taylor series:

```python
def sin(x, terms=12):
    if isinstance(x, (int, float)):
        return math.sin(x)  # Fast path for scalars

    result = Composite({})
    for n in range(terms):
        sign = (-1) ** n
        coeff = sign / math.factorial(2*n + 1)
        result = result + coeff * (x ** (2*n + 1))
    return result
```

**What this does:**

```
sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
```

Each power of x is computed via composite arithmetic, so:

- x = 3 + h
- x³ = (3 + h)³ = 27 + 27h + 9h² + h³
- The convolution automatically handles all derivatives!

### Why Terms Parameter?

More terms = more accuracy:

- `terms=12`: Good for most functions, derivatives up to ~10th order
- `terms=20`: High-order derivatives or difficult functions
- `terms=30`: Extreme precision needs

**Tradeoff:** More terms = slower, but more accurate derivatives.

---

### Handling ln and sqrt

For functions like ln(x), we need the **standard part** first:

```python
def ln(x, terms=15):
    a = x.st()  # Standard part
    if a <= 0:
        raise ValueError("ln requires positive standard part")

    h_part = x - R(a)  # Infinitesimal part
    ratio = h_part / R(a)  # h/a

    # Mercator series: ln(1 + u) = u - u²/2 + u³/3 - ...
    result = Composite({0: math.log(a)})
    power = Composite({0: 1})

    for n in range(1, terms):
        power = power * ratio
        sign = (-1) ** (n + 1)
        result = result + sign * power / n

    return result
```

**Key insight:** ln(a + h) = ln(a) + ln(1 + h/a), then expand ln(1 + u) as a series.

---

## Integration

### Dimensional Shift Strategy

Integration is the **inverse of differentiation**, so:

- Differentiation shifts dimensions **down** (0 → -1 → -2)
- Integration shifts dimensions **up** (-2 → -1 → 0)

```python
def antiderivative(f_composite, constant=0):
    result = {0: constant}
    for dim, coeff in f_composite.c.items():
        new_dim = dim - 1  # Shift up (more negative becomes less negative)
        divisor = abs(new_dim)
        if divisor > 0:
            result[new_dim] = coeff / divisor
    return Composite(result)
```

**Example:**

```
f = {0: 6, -1: 12, -2: 6}  (polynomial with derivatives)
F = antiderivative(f)
  = {0: constant, -1: 6/1, -2: 12/2, -3: 6/3}
  = {0: constant, -1: 6, -2: 6, -3: 2}
```

### Definite Integration

Use **stepped integration**:

1. Evaluate f at point x₀ as composite → get all derivatives
2. Shift dimensions to get antiderivative
3. Evaluate F(x₀ + Δx) - F(x₀) using Taylor coefficients
4. Repeat for next step

```python
def integrate_adaptive(f, a, b, tol=1e-10):
    total = 0.0
    x0 = a

    while x0 < b:
        # ONE evaluation → all derivatives
        fx = f(R(x0) + ZERO)

        # ONE dimensional shift → antiderivative
        Fx = antiderivative(fx)

        # Evaluate contribution from this step
        dx = min(step_size, b - x0)
        contribution = sum(
            coeff * dx ** abs(dim)
            for dim, coeff in Fx.c.items() if dim < 0
        )

        total += contribution
        x0 += dx

    return total
```

**Efficiency:** One function evaluation per step gives you the antiderivative for free!

---

## Optimization Strategies

### 1. Sparse Representation

Only store non-zero coefficients:

```python
# Good: {0: 5, -2: 3}  (2 entries)
# Bad:  {0: 5, -1: 0, -2: 3, -3: 0}  (4 entries, 2 useless)
```

**Filter on creation:**

```python
self.c = {k: v for k, v in coefficients.items() if abs(v) > 1e-15}
```

---

### 2. Fast Path for Scalars

Check for scalar operations first:

```python
def __mul__(self, other):
    if isinstance(other, (int, float)):
        # Fast scalar multiply
        return Composite({k: v * other for k, v in self.c.items()})
    # ... full convolution for composite × composite
```

---

### 3. Limit Dimension Range

For most applications, you don't need infinite dimensions:

```python
MAX_DIMENSION = 10  # Track up to 10th derivative

def prune(self):
    self.c = {k: v for k, v in self.c.items() if abs(k) <= MAX_DIMENSION}
```

---

### 4. Vectorization (Future)

Replace dictionary with dense array for fixed dimension range:

```python
# Instead of: {0: 5, -1: 3, -2: 1}
# Use array:  [5, 3, 1, 0, 0, ...] with known dimension mapping
```

**Benefit:** SIMD operations, much faster convolution.

---

### 5. FFT Convolution (Future)

For high-order derivatives (20+ dimensions), use FFT:

```python
import numpy.fft as fft

def mul_fft(a, b):
    n = len(a) + len(b) - 1
    fa = fft.fft(a, n)
    fb = fft.fft(b, n)
    return fft.ifft(fa * fb).real
```

**Speedup:** O(n log n) instead of O(n²).

---

## Memory Layout

### Current Implementation

```
Composite object:
  c: dict (Python dict, ~200 bytes overhead + 24 bytes per entry)

Typical size for x³ at x=2:
  4 dimensions × 24 bytes = 96 bytes
  + 200 bytes overhead
  = ~300 bytes per number
```

### Optimized Implementation (Future)

```python
class CompositeOptimized:
    __slots__ = ['coeffs', 'dim_offset']

    def __init__(self, coeffs, dim_offset):
        self.coeffs = np.array(coeffs, dtype=np.float64)  # Dense array
        self.dim_offset = dim_offset  # Lowest dimension index
```

**Size:** 8 bytes per coefficient + 16 bytes overhead = 88 bytes for x³.

---

## Why It's Slow (Currently)

**Bottlenecks:**

1. **Dictionary operations** - Hash lookups instead of array indexing
2. **Python loops** - No vectorization
3. **Memory allocation** - New dict for every operation
4. **No JIT compilation** - Interpreted Python

**Expected speedup with optimization:**

- Vectorized arrays: 10×
- FFT convolution: 20× (high-order)
- JIT compilation (Numba): 50×
- GPU implementation: 100×

Total potential: **500-1000× faster** → competitive with PyTorch autograd!

---

## Comparison with Other Systems

| System | How it works | Complexity |
| --- | --- | --- |
| **Dual numbers** | ε² = 0 truncation | O(1) per op |
| **Taylor AD** | Fixed-order tape | O(k) per op |
| **This system** | Sparse convolution | O(k²) per op |

**Tradeoff:** This system is slower per operation but gives **all orders** of derivatives simultaneously. Taylor AD requires specifying order upfront.

---

## Key Implementation Insights

### 1. Convolution = Leibniz Rule

The fundamental insight: polynomial multiplication **is** the Leibniz product rule. You don't need to "implement" the product rule - it emerges from convolution.

### 2. Dimensional Shift = Integration

Integration isn't a complex algorithm - it's just shifting dimensions up and dividing by the new index.

### 3. Taylor Series = Function Encoding

Transcendental functions are "encoded" as their Taylor series, then composite arithmetic handles all derivatives automatically.

### 4. One Evaluation = All Derivatives

Unlike standard calculus where you need N evaluations for N derivatives, composite arithmetic gets all N derivatives from a single evaluation.

---

## Extending the System

### Adding a New Transcendental Function

Template:

```python
def my_function(x, terms=15):
    if isinstance(x, (int, float)):
        return standard_implementation(x)

    a = x.st()  # Get standard part
    # ... compute f(a) using standard math

    # Taylor expand around a
    result = Composite({0: f_at_a})

    # Add terms: use known Taylor series for your function
    # ...

    return result
```

### Supporting Complex Numbers

Change coefficient type:

```python
self.c = {k: complex(v) for k, v in coefficients.items()}
```

All arithmetic operations work unchanged!

---

## See Also

- [**10-Minute Tutorial**](docs/Tutorial%20-%20Getting%20Started.md) - Get started quickly
- [**API Reference**](docs/API%20Reference.md) - Complete function docs
- [**Implementation Guide**](docs/Implementation%20Guide.md) - How it works internally
- [**Examples**](docs/Examples.md) - Code snippets for common tasks
- [**Roadmap (DRAFT)**](docs/Roadmap%20(DRAFT).md) - What's next
