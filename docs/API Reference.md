# API Reference# API Reference

# API Reference

Complete reference for the Composite Calculus library.

---

## Table of Contents

- [Core Classes](#core-classes)
- [Constructor Functions](#constructor-functions)
- [Arithmetic Operations](#arithmetic-operations)
- [Transcendental Functions](#transcendental-functions)
- [High-Level Calculus API](#high-level-calculus-api)
- [Extraction Methods](#extraction-methods)
- [Utility Functions](#utility-functions)

---

## Core Classes

### `Composite`

The fundamental class representing a composite number with dimensional structure.

**Constructor:**

```python
Composite(coefficients=None)
```

**Parameters:**

- `coefficients`: Dict mapping dimension (int) to coefficient (float), or a scalar number

**Attributes:**

- `c`: Dict[int, float] - Sparse representation of coefficients by dimension

**Dimensions:**

- `dimension 0`: Real numbers
- `dimension -1`: Infinitesimals (first-order)
- `dimension -2`: Second-order infinitesimals
- `dimension +1`: Infinities

**Example:**

```python
# Create from dict
x = Composite({0: 3, -1: 1})  # |3|₀ + |1|₋₁

# Create from scalar
x = Composite(5)  # |5|₀
```

---

## Constructor Functions

### `R(x)`

Create a real number at dimension 0.

**Parameters:**

- `x`: float - The real value

**Returns:** Composite

**Example:**

```python
x = R(3.14)  # |3.14|₀
```

---

### `ZERO`

Structural zero (infinitesimal): `|1|₋₁`

**Type:** Composite

**Example:**

```python
h = ZERO  # The infinitesimal
x = R(3) + ZERO  # 3 + h for differentiation
```

---

### `INF`

Structural infinity: `|1|₁`

**Type:** Composite

**Example:**

```python
limit(f, as_x_to=float('inf'))  # Uses INF internally
```

---

## Arithmetic Operations

All standard arithmetic operations are overloaded for Composite objects.

### Addition: `a + b`

Adds coefficients at matching dimensions.

**Example:**

```python
a = Composite({0: 3, -1: 2})
b = Composite({0: 1, -1: 4})
result = a + b  # |4|₀ + |6|₋₁
```

---

### Subtraction: `a - b`

Subtracts coefficients at matching dimensions.

---

### Multiplication: `a * b`

Uses convolution: dimensions add, coefficients multiply. This automatically implements the Leibniz product rule.

**Example:**

```python
x = R(2) + ZERO  # |2|₀ + |1|₋₁
y = x * x         # |4|₀ + |4|₋₁ + |1|₋₂
# (2 + h)² = 4 + 4h + h²
```

---

### Division: `a / b`

Dimensions subtract, uses polynomial long division for multi-term divisors.

**Example:**

```python
result = (R(10) * ZERO) / ZERO  # |10|₀ (reversible!)
```

---

### Power: `a ** n`

Integer powers via repeated multiplication.

**Parameters:**

- `n`: int - The exponent

**Example:**

```python
x = R(2) + ZERO
result = x ** 3  # (2+h)³ = 8 + 12h + 6h² + h³
```

---

## Transcendental Functions

All transcendental functions use Taylor series expansion.

### `sin(x, terms=12)`

Sine function for composite numbers.

**Parameters:**

- `x`: Composite or float
- `terms`: int - Number of Taylor series terms (default: 12)

**Returns:** Composite or float

**Example:**

```python
x = R(0) + ZERO
result = sin(x)
print(result.st())  # 0
print(result.d(1))  # 1 (derivative: cos(0) = 1)
```

---

### `cos(x, terms=12)`

Cosine function for composite numbers.

**Example:**

```python
result = cos(R(0) + ZERO)
print(result.st())  # 1
print(result.d(1))  # 0 (derivative: -sin(0) = 0)
```

---

### `tan(x, terms=10)`

Tangent function via sin/cos.

---

### `exp(x, terms=15)`

Exponential function eˣ.

**Example:**

```python
result = exp(R(0) + ZERO)
print(result.st())  # 1
print(result.d(1))  # 1 (derivative of eˣ is eˣ)
```

---

### `ln(x, terms=15)`

Natural logarithm.

**Requires:** [x.st](http://x.st)() > 0

**Example:**

```python
result = ln(R(1) + ZERO)
print(result.st())  # 0
print(result.d(1))  # 1 (derivative: 1/1 = 1)
```

---

### `sqrt(x, terms=12)`

Square root via binomial series.

**Requires:** [x.st](http://x.st)() > 0

---

### `atan(x, terms=15)`

Arctangent.

**Example:**

```python
result = atan(R(1) + ZERO)
print(result.st())  # π/4 ≈ 0.785
print(result.d(1))  # 0.5 (derivative: 1/(1+1) = 0.5)
```

---

### `asin(x, terms=15)`

Arcsine.

**Requires:** |[x.st](http://x.st)()| < 1

---

### `acos(x, terms=15)`

Arccosine via π/2 - asin(x).

---

### Hyperbolic Functions

**`sinh(x, terms=15)`** - Hyperbolic sine: (eˣ - e⁻ˣ)/2

**`cosh(x, terms=15)`** - Hyperbolic cosine: (eˣ + e⁻ˣ)/2

**`tanh(x, terms=15)`** - Hyperbolic tangent: sinh/cosh

---

### `power(x, s, terms=15)`

Real-valued power xˢ for any real exponent.

**Parameters:**

- `x`: Composite (requires [x.st](http://x.st)() > 0)
- `s`: float - Any real exponent
- `terms`: int - Taylor series terms

**Returns:** Composite

**Example:**

```python
# Cube root
result = power(R(8) + ZERO, 1/3)
print(result.st())  # 2

# Fractional power
result = power(R(4) + ZERO, 0.5)
print(result.st())  # 2 (same as sqrt)

# Irrational power
result = power(R(2) + ZERO, math.pi)
print(result.st())  # 2^π ≈ 8.825
```

---

## High-Level Calculus API

Convenience functions that automatically translate calculus problems to composite arithmetic.

### `derivative(f, at, terms=12)`

Compute f'(at) automatically.

**Parameters:**

- `f`: Callable - Function to differentiate
- `at`: float - Point at which to evaluate derivative
- `terms`: int - Taylor series terms for transcendentals

**Returns:** float

**Example:**

```python
# Simple polynomial
f_prime = derivative(lambda x: x**2, at=3)  # → 6

# Transcendental
f_prime = derivative(lambda x: sin(x), at=0)  # → 1

# Composition
f_prime = derivative(lambda x: exp(x**2), at=1)  # → 2e
```

---

### `nth_derivative(f, n, at, terms=12)`

Compute the nth derivative f⁽ⁿ⁾(at).

**Parameters:**

- `f`: Callable
- `n`: int - Order of derivative
- `at`: float - Point of evaluation
- `terms`: int - Taylor series terms

**Returns:** float

**Example:**

```python
# Third derivative of x⁵ at x=2
result = nth_derivative(lambda x: x**5, n=3, at=2)  # → 120

# Fifth derivative of eˣ at x=1
result = nth_derivative(lambda x: exp(x), n=5, at=1)  # → e
```

---

### `all_derivatives(f, at, up_to=5, terms=12)`

Get all derivatives [f(at), f'(at), f''(at), ...] up to nth derivative.

**Parameters:**

- `f`: Callable
- `at`: float
- `up_to`: int - Highest derivative order
- `terms`: int - Taylor series terms

**Returns:** List[float]

**Example:**

```python
# All derivatives of eˣ at x=0
derivs = all_derivatives(lambda x: exp(x), at=0, up_to=5)
# → [1, 1, 1, 1, 1, 1]

# All derivatives of sin(x) at x=0
derivs = all_derivatives(lambda x: sin(x), at=0, up_to=4)
# → [0, 1, 0, -1, 0]
```

---

### `limit(f, as_x_to, terms=12)`

Compute lim(x→a) f(x) automatically.

**Parameters:**

- `f`: Callable
- `as_x_to`: float or float('inf') or float('-inf')
- `terms`: int - Taylor series terms

**Returns:** float

**Example:**

```python
# Classic limits
limit(lambda x: sin(x)/x, as_x_to=0)  # → 1

# Algebraic limit
limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)  # → 4

# Limit at infinity
limit(lambda x: (3*x + 1)/(x + 2), as_x_to=float('inf'))  # → 3
```

---

### `limit_right(f, as_x_to, terms=12)`

Right-hand limit: lim(x→a⁺) f(x)

---

### `limit_left(f, as_x_to, terms=12)`

Left-hand limit: lim(x→a⁻) f(x)

---

### `taylor_coefficients(f, at, up_to=5, terms=12)`

Get Taylor series coefficients [a₀, a₁, a₂, ...] where f(x) ≈ Σ aₙ(x-at)ⁿ

**Note:** aₙ = f⁽ⁿ⁾(at) / n!

**Example:**

```python
coeffs = taylor_coefficients(lambda x: exp(x), at=0, up_to=4)
# → [1, 1, 0.5, 0.166..., 0.041...]  (all 1/n!)
```

---

### Integration Functions

#### `antiderivative(f_composite, constant=0)`

Compute antiderivative via dimensional shift.

**Parameters:**

- `f_composite`: Composite - Function represented as composite
- `constant`: float - Integration constant

**Returns:** Composite

**Example:**

```python
x = R(2) + ZERO
f = x**2  # Function
F = antiderivative(f)  # Antiderivative

# Verify: differentiate(F) should equal f
```

---

#### `integrate_stepped(f, a, b, step=0.5, terms=15)`

Multi-point stepped integration with error estimate.

**Parameters:**

- `f`: Callable
- `a`, `b`: float - Integration bounds
- `step`: float - Step size
- `terms`: int - Taylor series terms

**Returns:** Tuple[float, float] - (value, error_estimate)

**Example:**

```python
val, err = integrate_stepped(lambda x: x**2, 0, 1)
# val ≈ 0.333, err ≈ 0 (exact for polynomials)
```

---

#### `integrate_adaptive(f, a, b, tol=1e-10, terms=15)`

Adaptive stepped integration that automatically adjusts step size.

**Parameters:**

- `f`: Callable
- `a`, `b`: float - Integration bounds
- `tol`: float - Target accuracy
- `terms`: int - Taylor series terms

**Returns:** Tuple[float, float] - (value, error_estimate)

**Example:**

```python
val, err = integrate_adaptive(lambda x: exp(-(x*x)), 1, 2)
# val ≈ 0.1353, err ≈ 1e-15
```

---

#### `improper_integral(f, a, tol=1e-8, cutoff=20)`

Compute ∫ₐ^∞ f(x) dx.

**Example:**

```python
val, err = improper_integral(lambda x: exp(-x), 0)  # ≈ 1.0
```

---

#### `improper_integral_both(f, tol=1e-8)`

Compute ∫₋∞^∞ f(x) dx.

**Example:**

```python
val, err = improper_integral_both(lambda x: exp(-(x*x)))  # ≈ √π
```

---

#### `improper_integral_to(f, a, b, tol=1e-8)`

Integrate when f has a singularity at a or b.

**Example:**

```python
val, err = improper_integral_to(lambda x: 1/sqrt(x), 0, 1)  # ≈ 2.0
```

---

## Extraction Methods

Methods on `Composite` objects to extract information.

### `.st()`

Get the standard part (coefficient at dimension 0).

**Returns:** float

**Example:**

```python
x = Composite({0: 5, -1: 2, -2: 1})
print(x.st())  # 5
```

---

### `.coeff(dim)`

Get coefficient at a specific dimension.

**Parameters:**

- `dim`: int - The dimension

**Returns:** float

**Example:**

```python
x = Composite({0: 5, -1: 2, -2: 1})
print(x.coeff(-1))  # 2
print(x.coeff(-2))  # 1
```

---

### `.d(n=1)`

Extract the nth derivative, accounting for factorial scaling.

**Parameters:**

- `n`: int - Derivative order (default: 1)

**Returns:** float

**Formula:** Returns `coeff(-n) * n!`

**Example:**

```python
x = R(3) + ZERO
result = x**4

print(result.d(1))  # 108 = first derivative at x=3
print(result.d(2))  # 216 = second derivative at x=3
print(result.d(3))  # 216 = third derivative at x=3
```

---

## Utility Functions

### `show(composite, name="result")`

Pretty print a composite number with extracted values.

**Parameters:**

- `composite`: Composite
- `name`: str - Label for output

**Example:**

```python
x = R(2) + ZERO
result = x**3
show(result, "cubic")

# Output:
# cubic = |8|₀ + |12|₋₁ + |6|₋₂ + |1|₋₃
#   st() = 8
#   f'   = 12
#   f''  = 12
#   f''' = 6
```

---

### `trace(f, at=None, to=None)`

Trace composite computation showing all intermediate steps.

**Parameters:**

- `f`: Callable
- `at`: float - For derivative (x = at + h)
- `to`: float - For limit (x → to)

**Returns:** Composite

**Example:**

```python
trace(lambda x: (3*x + 1)/(x + 2), to=float('inf'))

# Output:
# === TRACE: lim(x→∞) ===
# Let x = |1|₁  (INF)
#     |3|₀  ×  |1|₁
#   = |3|₁
#     |3|₁  +  |1|₀
#   = |3|₁ + |1|₀
#     ...
# RESULT: |3|₀
# Limit = 3.0
```

---

### `translate(f, at=None, to=None)`

Show the composite translation without step-by-step trace.

**Example:**

```python
translate(lambda x: x**2, at=3)

# Output:
# Substitution: x = R(3) + ZERO
# Translation:  |9|₀ + |6|₋₁ + |1|₋₂
# f(3) = 9
# f'(3) = 6
```

---

### `verify_derivative(f, f_prime, at, tol=1e-6)`

Verify that f_prime is the derivative of f at a point.

**Parameters:**

- `f`: Callable
- `f_prime`: Callable or float - Expected derivative
- `at`: float
- `tol`: float - Tolerance

**Returns:** bool

**Example:**

```python
is_correct = verify_derivative(
    lambda x: x**2,
    lambda x: 2*x,
    at=3
)  # → True
```

---

### `run_tests()`

Run the built-in test suite to verify library functionality.

**Returns:** bool - True if all tests pass

**Example:**

```python
from composite_lib import run_tests
run_tests()
```

---

## Comparison Operations

Composite numbers support lexicographic comparison by dimension.

**Available operators:**

- `==` - Equality
- `<` - Less than
- `<=` - Less than or equal
- `>` - Greater than
- `>=` - Greater than or equal

**Comparison rule:** Compare highest dimension first, then next highest, etc.

**Example:**

```python
a = Composite({1: 5})     # |5|₁ (infinity-like)
b = Composite({0: 100})   # |100|₀ (finite)
print(a > b)  # True (dimension 1 > dimension 0)
```

---

## Type Conversions

Composite objects can interact with Python scalars (int, float):

```python
# Scalars are automatically converted
x = R(3) + ZERO
result = x + 5      # Composite + int → Composite
result = x * 2.5    # Composite * float → Composite
result = 10 / x     # int / Composite → Composite
```

---

## Error Handling

### Common Exceptions

**`ZeroDivisionError`**

- Raised when dividing by Python zero (not ZERO)
- Use `ZERO` for structural zero instead

**`ValueError`**

- Raised for ln(x) when [x.st](http://x.st)() ≤ 0
- Raised for sqrt(x) when [x.st](http://x.st)() < 0
- Raised for asin(x) when |[x.st](http://x.st)()| ≥ 1

**`TypeError`**

- Raised for non-integer powers with `**` operator
- Use `power(x, s)` for fractional/real powers

---

## Best Practices

1. **Use high-level API when possible**

    ```python
    # Good
    result = derivative(lambda x: x**2, at=3)

    # Also fine, but more verbose
    x = R(3) + ZERO
    result = (x**2).d(1)
    ```

2. **Adjust terms for precision**

    ```python
    # Default terms=12 is usually sufficient
    sin(x)

    # Increase for high-order derivatives or difficult functions
    sin(x, terms=20)
    ```

3. **Use ZERO, not Python 0**

    ```python
    # Good
    x = R(3) + ZERO

    # Bad - won't give derivatives
    x = R(3) + 0
    ```

4. **Check standard part for validity**

    ```python
    x = some_computation()
    if x.st() > 0:
        result = ln(x)  # Safe
    ```


---

## Performance Notes

- Dictionary-based sparse representation: O(k) where k = number of non-zero dimensions
- Multiplication (convolution): O(k²) for k dimensions
- Division: O(k*n) for n iterations of long division
- Transcendental functions: O(terms * operations)

**Typical performance:** ~500-1000× slower than scalar operations, but computes all derivatives simultaneously.

---

## See Also

- [Tutorial](tutorial.md) - Getting started guide
- [Implementation Guide](implementation.md) - How it works internally
- [Examples](examples/) - Code examples for common tasks
