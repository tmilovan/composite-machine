# Tutorial - Getting Started

# Tutorial: Getting Started with Composite Calculus

**Time to complete:** 10 minutes

This tutorial will teach you the basics of composite arithmetic and how to use it for automatic calculus.

---

## Installation

```bash
pip install composite-machine
```

Or from source:

```bash
git clone https://github.com/tmilovan/composite-machine.git
cd composite-machine
pip install -e .
```

---

## Core Concepts

### 1. Composite Numbers

A composite number represents a **value + all its derivatives** at once.

```python
from composite import R, ZERO

# Create a composite number: 3 + infinitesimal
x = R(3) + ZERO

print(x)  # <|3|₀ |1|₋₁>
# |3|₀ means: value = 3 at dimension 0
# |1|₋₁ means: derivative seed = 1 at dimension -1
```

**Dimensions:**

- Dimension 0: The actual value
- Dimension -1: First derivative information
- Dimension -2: Second derivative information
- And so on...

### 2. Automatic Differentiation

When you compute with composite numbers, **derivatives emerge automatically**:

```python
# f(x) = x²
x = R(3) + ZERO
result = x ** 2

print(result)  # <|9|₀ |6|₋₁ |1|₋₂>

# Extract values:
print(result.st())   # 9 (function value)
print(result.d(1))   # 6 (first derivative)
print(result.d(2))   # 2 (second derivative)
```

**Why this works:**

- x² = (3 + h)² = 9 + 6h + h²
- The coefficient of h is the derivative
- The coefficient of h² gives the second derivative

### 3. Higher-Order Derivatives

All derivatives appear simultaneously:

```python
# f(x) = x⁴ at x = 2
x = R(2) + ZERO
result = x ** 4

print(result.st())   # 16   (f(2))
print(result.d(1))   # 32   (f'(2))
print(result.d(2))   # 48   (f''(2))
print(result.d(3))   # 48   (f'''(2))
print(result.d(4))   # 24   (f⁴(2))
```

---

## Common Operations

### Basic Arithmetic

```python
from composite import R, ZERO

x = R(3) + ZERO
y = R(5) + ZERO

# Addition
z = x + y  # Composite addition

# Multiplication (uses convolution - Leibniz rule!)
z = x * y  # Product rule is automatic

# Division
z = x / y  # Quotient rule is automatic
```

### Transcendental Functions

```python
from composite import R, ZERO, sin, cos, exp, ln

x = R(1) + ZERO

# Trigonometric
result = sin(x)
print(result.st())   # sin(1) ≈ 0.841
print(result.d(1))   # cos(1) ≈ 0.540

# Exponential
result = exp(x)
print(result.st())   # e¹ ≈ 2.718
print(result.d(1))   # e¹ ≈ 2.718 (derivative of e^x is e^x)

# Logarithm
result = ln(x)
print(result.st())   # ln(1) = 0
print(result.d(1))   # 1/1 = 1
```

---

## High-Level API

For convenience, use the high-level functions:

### Derivatives

```python
from composite import derivative, nth_derivative

# First derivative
f_prime = derivative(lambda x: x**3, at=2)  # → 12

# nth derivative:  d³/dx³ of x⁵ is 60x², so at x=2 it is 60·4
f_triple_prime = nth_derivative(lambda x: x**5, n=3, at=2)  # → 240
```

### Limits

```python
from composite import limit, sin

# Classic limit
result = limit(lambda x: sin(x)/x, as_x_to=0)  # → 1.0

# Limit at infinity
result = limit(lambda x: (3*x + 1)/(x + 2), as_x_to=float('inf'))  # → 3.0
```

### All Derivatives at Once

```python
from composite import all_derivatives, exp

# Get f(x), f'(x), f''(x), ... up to nth derivative
derivs = all_derivatives(lambda x: exp(x), at=0, up_to=5)
print(derivs)  # [1, 1, 1, 1, 1, 1]  (all derivatives of e^x at 0 are 1)
```

---

## Division by Zero (Yes, Really!)

One of the unique features:

```python
from composite import ZERO

# Multiplication by zero is reversible
result = 5 * ZERO
print(result)  # |5|₋₁ (the 5 is preserved!)

# Division by zero works
result = (5 * ZERO) / ZERO
print(result)  # 5 (reversible!)

# 0/0 is well-defined
result = ZERO / ZERO
print(result)  # 1
```

**Why this matters:**

- Enables limits without L'Hôpital's rule
- Makes calculus algebraic instead of algorithmic
- No NaN or special cases

---

## Practical Example: Product Rule

The product rule emerges automatically from convolution:

```python
from composite import R, ZERO, sin, cos

# f(x) = x² · sin(x) at x = 1
x = R(1) + ZERO

f = x**2
g = sin(x)
product = f * g

# Verify product rule: (fg)' = f'g + fg'
f_val, f_prime = f.st(), f.d(1)        # x², 2x
g_val, g_prime = g.st(), g.d(1)        # sin(x), cos(x)

expected = f_prime * g_val + f_val * g_prime  # Product rule
actual = product.d(1)

print(f"Expected: {expected}")
print(f"Actual:   {actual}")
print(f"Match: {abs(expected - actual) < 1e-10}")  # True
```

---

## Next Steps

Now that you understand the basics:

[API Reference](API%20Reference.md)

[Explainer: Core Composite Class — Annotated Reference](Core%20Composite%20Class%20—%20Annotated%20Reference.md)

[Implementation Guide](Implementation%20Guide.md)

[Examples](Examples.md)

[Roadmap (DRAFT)](Roadmap%20(DRAFT).md)

[Exploration & research (Turing completeness)](Turing%20Completeness%20—%20Evidence%20and%20Open%20Questions.md)

---

## Quick Reference

### Creating Composite Numbers

```python
R(5)           # Real number 5
ZERO           # Structural zero (infinitesimal)
INF            # Structural infinity
R(3) + ZERO    # 3 with derivative seed
```

### Extraction

```python
result.st()    # Standard part (value)
result.d(1)    # First derivative
result.d(n)    # nth derivative
result.coeff(k) # Coefficient at dimension k
```

### Functions

```python
derivative(f, at=x)           # f'(x)
nth_derivative(f, n, at=x)    # f⁽ⁿ⁾(x)
limit(f, as_x_to=a)           # lim_{x→a} f(x)
all_derivatives(f, at=x, up_to=n)  # [f, f', f'', ..., f⁽ⁿ⁾]
```

---

**You're ready to start using composite calculus!** 🎉

© Toni Milovan. Documentation licensed under CC BY-SA 4.0. Code licensed under AGPL-3.0.
