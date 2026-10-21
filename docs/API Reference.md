# API Reference

Complete reference for the Composite Calculus library.

---

## Table of Contents

- [Core Classes](#core-classes)
- [Constructor Functions](#constructor-functions)
- [Arithmetic Operations](#arithmetic-operations)
- [Transcendental Functions](#transcendental-functions)
- [High-Level Calculus API](#high-level-calculus-api)
- [Partial Derivatives](#partial-derivatives)
- [Extraction Methods](#extraction-methods)
- [Zero Handling](#zero-handling)
- [Utility Functions](#utility-functions)
- [Comparison Operations](#comparison-operations)
- [Type Conversions](#type-conversions)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Performance Notes](#performance-notes)

---

## Core Classes

### `Composite`

The fundamental class representing a composite number with dimensional structure.

**Constructor:**

```python
Composite(coefficients=None)
```

**Parameters:**

- `coefficients`: Dict mapping dimension to coefficient (float), or a scalar number

**Attributes:**

- `c`: Sparse dict view of coefficients by dimension, rebuilt on every access.
  Prefer `coeffs_dict()`, or the backend's `to_arrays`, in new code.

**Dimensions.** A dimension is an integer, a real number, or a vector over an
iterated-logarithm basis:

- `dimension 0`: real numbers
- `dimension -1`: infinitesimals (first order)
- `dimension -2`: second-order infinitesimals
- `dimension +1`: infinities
- `dimension -0.5`: a half order, which is a branch point - `sqrt(ZERO)`
- `dimension (0, 1)`: the log axis - `ln(1/h)`. `(0, 0, 1)` is `ln(ln(1/h))`,
  and so on to any depth. These appear on their own when `ln` meets an
  infinitesimal or an infinite value; nothing has to be switched on.

Dimensions compare by dominance, which for a vector is Python's own tuple
order, so `ln(1/h) > 1 > h*ln(1/h) > h`.

**Example:**

```python
# Create from dict
x = Composite({0: 3, -1: 1})  # <|3|₀ |1|₋₁>

# Create from scalar
x = Composite(5)  # |5|₀

# NOT a zero: an empty composite is NOTHING, which is the classical zero
e = Composite({})             # ∅
z = Composite({0: 0.0})       # |0|₀, an EXPRESSED zero, which converts under R1
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

Adds coefficients at matching dimensions. What happens to a zero operand, and
to a cancellation, is set by [`CANCELLATION`](#cancellation); under the default
(`"convert_and_keep"`):

```python
R(5) + R(0)               # <|5|_0 |1|_-1>    a written zero converts (R1) and is kept
R(0) + R(0) + R(0)        # <|3|_-1>          three zeros
3*INF - 3*INF             # <|0|_0>           a cancellation is counted, no residue
(3*INF + 1) - 3*INF       # <|1|_0>
```

**Example:**

```python
a = Composite({0: 3, -1: 2})
b = Composite({0: 1, -1: 4})
result = a + b  # <|4|₀ |6|₋₁>
```

---

### Subtraction: `a - b`

Subtracts coefficients at matching dimensions. A zero operand converts first and
the sign then applies, so `x - 0` and `x + (-0)` are the same number. Under the
default rule:

```python
R(6) - R(6)               # <|0|_0>           counted: the same number as R(0)
R(5) - R(0)               # <|5|_0 |-1|_-1>
R(0) - R(0)               # <|0|_-1>          ZERO - ZERO, counted
R(0) - R(0) - R(0)        # <|-1|_-1 |1|_-2>  <|0|_-1> is a zero again and converts
```

---

### Multiplication: `a * b`

Uses convolution: dimensions add, coefficients multiply. This automatically implements the Leibniz product rule.

**Example:**

```python
x = R(2) + ZERO  # <|2|₀ |1|₋₁>
y = x * x         # <|4|₀ |4|₋₁ |1|₋₂>
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

### `tan(x, terms=12)`

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

### `erf(x, terms=15)`, `erfc(x, terms=15)`, `normal_cdf(x, terms=15)`

Error function, complementary error function, and the standard normal CDF.
Each is a series in the argument, so derivatives of every order come back with
the value.

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
f_prime = derivative(lambda x: x**2, at=3)  # -> 6

# Transcendental
f_prime = derivative(lambda x: sin(x), at=0)  # -> 1

# Composition
f_prime = derivative(lambda x: exp(x**2), at=1)  # -> 2e
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
# Third derivative of x⁵ at x=2:  60·x² = 60·4
result = nth_derivative(lambda x: x**5, n=3, at=2)  # -> 240

# Fifth derivative of eˣ at x=1
result = nth_derivative(lambda x: exp(x), n=5, at=1)  # -> e
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
# -> [1, 1, 1, 1, 1, 1]

# All derivatives of sin(x) at x=0
derivs = all_derivatives(lambda x: sin(x), at=0, up_to=4)
# -> [0, 1, 0, -1, 0]
```

---

### `limit(f, as_x_to, terms=12, dir="both", fallback=False)`

Compute lim(x->a) f(x) automatically.

**Parameters:**

- `f`: Callable
- `as_x_to`: float, float('inf'), float('-inf'), or a composite INFINITY
- `terms`: int - Taylor series terms
- `dir`: str - `"both"` (default), `"+"`, `"-"`
- `fallback`: bool - if True, fall back to integral averaging when the
  algebraic route hits a domain error

**Returns:** float, or a Composite when the result is unbounded
(`limit(lambda x: R(1)/x, 0)` returns `|1|₁`).

**The infinitesimal has a sign, so a single evaluation is one-sided.**
`dir="both"` builds the same point as `dir="+"` and does not compare the two
sides, so a function whose sides disagree returns the right-hand value without
saying so:

```python
limit(lambda x: sqrt(x*x)/x, 0)   # 1.0
limit_left(lambda x: sqrt(x*x)/x, 0)   # -1.0
```

Ask for each side explicitly where that matters.

**Oscillation is refused, not approximated.** `sin(1/x)` at 0 takes every value
in [-1, 1] and a composite holds one value at one grade, so it raises
`NotRepresentableError`. This costs the squeeze cases (`x*sin(1/x)`)
deliberately: probing them returned the right answer for a convergent case and
0.0 for a divergent one, which is not a fallback.

**Example:**

```python
# Classic limits
limit(lambda x: sin(x)/x, as_x_to=0)  # -> 1

# Algebraic limit
limit(lambda x: (x**2 - 4)/(x - 2), as_x_to=2)  # -> 4

# Limit at infinity
limit(lambda x: (3*x + 1)/(x + 2), as_x_to=float('inf'))  # -> 3
```

---

### `limit_right(f, as_x_to, terms=12)`

Right-hand limit: lim(x->a⁺) f(x)

---

### `limit_left(f, as_x_to, terms=12)`

Left-hand limit: lim(x->a⁻) f(x)

---

### `taylor_coefficients(f, at, up_to=5, terms=12)`

Get Taylor series coefficients [a₀, a₁, a₂, ...] where f(x) ≈ Σ aₙ(x-at)ⁿ

**Note:** aₙ = f⁽ⁿ⁾(at) / n!

**Example:**

```python
coeffs = taylor_coefficients(lambda x: exp(x), at=0, up_to=4)
# -> [1, 1, 0.5, 0.166..., 0.041...]  (all 1/n!)
```

---

### Integration Functions

#### `integrate(f, *args, curve=None, surface=None, tol=1e-10, terms=15)`

One entry point for every integral form. This is the function the README and
the demos use; the routines below it are the specific engines.

**Forms:**

```python
integrate(lambda x: x**2, 0, 1)                    # 0.333...   definite
integrate(lambda x: exp(-x), 0, float('inf'))      # 1.0        improper
integrate(lambda x, y: x*y, (0, 1), (0, 1))        # 0.25       double, one range per variable
integrate(f, t_range, curve=c)                     #            line integral
integrate(f, u_range, v_range, surface=s)          #            surface integral
```

**Returns:** float.

Note that it ends in `result.st()`, so any grade other than the standard part
is discarded - which is why differentiating under the integral sign is not
available through this API even though the arithmetic underneath supports it.

---

#### `definite_integral(f, a, b, terms=12)`

The plain definite case, without the form dispatch above. Read by meeting
composites, as `integrate(f, a, b)` is; `terms` is kept for the signature and
no longer used.

---

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

Integration with a node at every `step`, each step read by meeting
composites; nodes are added inside a step only where its two ends disagree.

**Parameters:**

- `f`: Callable
- `a`, `b`: float - Integration bounds
- `step`: float - Spacing of the starting nodes
- `terms`: int - kept for the signature, no longer used

**Returns:** Tuple[Composite, float] - (value, nan). There is no error
estimate: a step is accepted when its two sides agree within tol, so the slot
is nan rather than a number that would look like a bound.

**Example:**

```python
val, err = integrate_stepped(lambda x: x**2, 0, 1)
# val.st() == 1/3 exactly, err is nan
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

Compute ∫ₐ^∞ f(x) dx. Returns (Composite, error).

Composite first: the node at infinity is x = 1/h, where an algebraic tail is an
ordinary composite and an exponential one with an integer rate is a transseries
sector; the tail starts where that node's jet reaches. The error slot is then
nan, since nothing estimates one. Tails the library cannot represent (a
Gaussian, a non-integer rate, an oscillation, float64 underflow) fall back to
the panel path, which returns its error estimate as before; `cutoff` belongs
to that path.

**Example:**

```python
val, err = improper_integral(lambda x: exp(-x), 0)          # <|1|_0>, nan
val, err = improper_integral(lambda x: exp(-(x*x)), 0)      # Gaussian: panel path
```

---

#### `improper_integral_both(f, tol=1e-8)`

Compute ∫₋∞^∞ f(x) dx, split at 0, composite first with the same fallback
as `improper_integral`. Returns (Composite, error).

**Example:**

```python
val, err = improper_integral_both(lambda x: exp(-(x*x)))  # ≈ √π
```

---

#### `improper_integral_to(f, a, b, tol=1e-8)`

Integrate when f has a singularity at a or b. The endpoint's jet is read by
grade, so a power law, a log or a pole there is integrated exactly rather than
fitted. Returns (Composite, nan): no error is estimated.

**Example:**

```python
val, err = improper_integral_to(lambda x: 1/sqrt(x), 0, 1)
# val is <2_0 -2_-0.5>: 2, and the -2 sqrt(h) its lower limit h leaves
```

---

## Partial Derivatives

`composite.partials` -- pure and mixed partial derivatives of a function of
several inputs, read from seeded runs.

```python
from composite.partials import partial, partials, Runs, PartialError
```

Not exported from `composite` itself: a top-level `partial` would shadow
`functools.partial` under `from composite import *`.

### `partial(f, at, orders, runs=None, names=None)`

One partial derivative of `f` at the point `at`, as a float in the units of `f`
and its inputs.

- **`f`** - any function written in composite operations, taking its inputs as
  keywords. A function that takes them positionally works with `names=`.
- **`at`** - the point: `{input: value}` with plain numbers, or a sequence in
  `names` order.
- **`orders`** - `{input: order}`. `{"x": 1}` is df/dx, `{"x": 2}` is d2f/dx2,
  `{"x": 1, "y": 1}` is d2f/dxdy, `{"x": 2, "y": 1, "z": 1}` is d4f/dx2dydz.
- **`runs`** - a `Runs` shared between calls at the same point, so each seeded
  run is made once.

```python
from composite.composite_lib import exp, sin
from composite.partials import partial, Runs

f = lambda x, y, z: exp(x * y) * sin(z) + x * y**2 * z**3
p = dict(x=0.5, y=1.2, z=0.7)
runs = Runs()

fx   = partial(f, p, {"x": 1}, runs)
fxy  = partial(f, p, {"x": 1, "y": 1}, runs)
fxxy = partial(f, p, {"x": 2, "y": 1}, runs)
fxyz = partial(f, p, {"x": 1, "y": 1, "z": 1}, runs)
```

### `partials(f, at, requests, runs=None, names=None)`

Several derivatives at one point, sharing their runs. `requests` maps a label to
its `orders`; the result maps the same labels to values.

```python
from composite.partials import partials

g = partials(f, p, {"fx":   {"x": 1},
                    "fxy":  {"x": 1, "y": 1},
                    "fxxy": {"x": 2, "y": 1},
                    "fyz":  {"y": 1, "z": 1}})
g["fxy"]
```

### How it works

A **run** is one evaluation of `f` with some inputs carrying a zero. One run
holds every order of what it seeds, so a pure partial is one run read at a grade
-- `d(n)`, no differencing.

A **mixed** partial is one **joint** run of its inputs minus the runs of every
smaller combination of them (inclusion-exclusion over records). What remains
starts at the target grade and holds only the terms that move every input. When
several of those share the grade -- d3f/dx2dy and d3f/dxdy2 both reach grade -3
-- joint runs with a different quantity of zero on an input separate them, with
weights solved exactly in rationals and applied in composite arithmetic. The
only finite-difference step is that final subtraction of records.

This is a different method from `composite_multivar.partial_derivative`, which
combines directional evaluations with interpolation weights.

The run count follows what is asked for, not every possible combination. All
sixteen derivative Greeks of a Black-Scholes price -- eleven pure, vanna, zomma,
charm, color, veta -- take twelve runs.

### Seeds

The seeding is internal; the result never depends on it, only its precision
does.

- **Relative.** Each input carries a zero in its own units,
  `x0 * (1 + c*h)` = `<x0_0 c*x0_-1>`, so the subtractions cancel terms of
  comparable size. With a unit zero on inputs of very different sizes (100
  beside 0.25) the mixed terms lose up to six digits.
- **An input at 0** is seeded as the zero itself, `<c_-1>`.
- **Equal inputs** never get the same quantity of zero: their seeds would be the
  same number, and a difference like `x - y` at `x = y` would cancel exactly.
- **Adaptive.** Where the inputs' own terms at the target grade are more than
  `BALANCE` (1e3) apart, the quantities are scaled so they arrive at the same
  size. That leaves a mixed term at |f_xy| / sqrt(|f_xx| |f_yy|) of its
  coefficient, which is a property of `f`: below about 1e-13 of that, float64
  holds no digit of it whatever the seeding. On a Black-Scholes price this
  happens only deep in the money, for vol-time cross terms of order 1e-20 and
  below.

### Zeros and denotation

Every run blesses one infinitesimal with `TAG` and builds every seed from it.
Any other zero -- a data `0.0` meeting a seed (`r - q` at `q = 0`), `ln(1.0)`
when two inputs have equal ratio, a cancellation -- then behaves as an ordinary
zero, which is what holding the other inputs fixed means. A run that still comes
back denoted (`f` cancels a seeded input exactly against itself) raises
`PartialError` rather than returning the derivative of a different function.

Calling `partial` inside `TAG()` or `single_infinitesimal()` raises
`RuntimeError`: it blesses its own infinitesimal per run.

### Depth

The order cap is set to the order requested for the duration of the call and
restored afterwards; no `set_max_order` is needed around it. A `Runs` remembers
the order each run was made to and remakes one that is too shallow for a later
request. `partials` runs everything to its deepest request, so its values can
differ from separate `partial` calls in the last bit.

### Errors

- `TypeError` - `f` returned something that is not a composite (a `math.*` call
  or a float cast dropped the seed).
- `PartialError` - a run came back denoted.
- `RuntimeError` - called inside TAG mode.
- `KeyError` - an input in `orders` is not in `at`.

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

### `.to_ieee754()`

The image of the composite in float arithmetic: the one well-defined way back
into a system that HAS an additive identity.

**Returns:** float

No single substitution `h = value` does this, because the two halves of the
dimension axis want opposite limits. Negative grades want `h = 0`, so an
infinitesimal becomes a true zero; at the smallest representable float instead,
`R(6) - R(6)` under mechanical cancellation (`|6|_-1`) comes back as `2.96e-323`, a subnormal crumb, and the identity is
not restored. Positive grades want `h -> 0` from above, where a pole becomes an
infinity -- which is IEEE754's own answer for `1/0`; at `h = 0` exactly they
divide by zero and raise. So the projection is piecewise, keyed on `lead_order`:

| `lead_order` | meaning | projects to |
| --- | --- | --- |
| `> 0` | infinitesimal | `0.0` |
| `== 0` | bounded | `st()` |
| `< 0` | unbounded | `+-inf`, by the dominant term's sign |
| `None` | an expressed zero | `0.0` |
| `None` | NOTHING | `nan` |

```python
((R(3)+ZERO)*(R(3)+ZERO)).to_ieee754()   # 9.0     bounded
(R(6) - R(6)).to_ieee754()               # 0.0     a true zero, not a subnormal
(R(1)/ZERO).to_ieee754()                 # inf
(R(-1)/ZERO).to_ieee754()                # -inf
ln(ZERO).to_ieee754()                    # -inf    ln of a small positive is large negative
(R(1)/ln(ZERO)).to_ieee754()             # 0.0     reaches 0, despite the log axis
Composite({}).to_ieee754()               # nan     absence, and float has no absence
Composite({0: 0.0}).to_ieee754()         # 0.0     an expressed zero IS a value
```

NOTHING becomes `nan` rather than `0.0` because float has no representation of
absence, and `0.0` would claim it was a zero. See `NOTHING`.

`float(c)` is deliberately NOT this: it keeps raising
`StandardPartUndefinedError` on an unbounded composite, because that exception
is what catches an accidental coercion through `math.*`. Ask for the projection
when you want it.

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
result = x**4          # <|81|₀ |108|₋₁ |54|₋₂ |12|₋₃ |1|₋₄>

print(result.st())  # 81  = f(3)
print(result.d(1))  # 108 = f'(3)   = 4·27
print(result.d(2))  # 108 = f''(3)  = 12·9
print(result.d(3))  # 72  = f'''(3) = 24·3
print(result.d(4))  # 24  = f''''(3)
```

`d(n)` is `coeff(-n) * n!`, so it is the derivative and not the Taylor
coefficient. For the coefficients use `taylor_coefficients`.

---

---

### `.D(n=1)`

The nth derivative as a **Composite**, not a float. Differentiation is a grade
shift: `|c|_g` becomes `|-g*c|_(g+1)`. The result is still differentiable, so
`f.D(1).d(1)` equals `f.d(2)`.

Use it where `d(n)` would flatten something that is not a number: for `(2+h)/h`
the grades are 0 and +1, grade -n is absent, and `d(n)` read through `D` says
the derivative is unbounded instead of returning 0.0.

---

### `.lead_dim()` and `.lead_order()`

`lead_dim` returns the dominant dimension, skipping zero coefficients;
`lead_order` returns its order - positive for an infinitesimal, negative for an
unbounded value, `None` for nothing.

```python
R(3).lead_order()          # 0     an ordinary number
ZERO.lead_order()          # 1     infinitesimal, first order
(ZERO*ZERO).lead_order()   # 2
(R(1)/ZERO).lead_order()   # -1    unbounded
sqrt(ZERO).lead_order()    # 0.5   a half order: a branch point
```

**Use this rather than `max(coeffs_dict())`.** A log-axis grade is a tuple,
comparing it against an integer raises, and reading only the power component
reports `x*ln(x)` as order 0.

---

### `.coeffs_dict()`, `.complete_order`, `.complete_coeffs()`, `.leaked_coeffs()`

`coeffs_dict()` returns every term the arithmetic produced. `complete_order` is
the highest order the value vouches for, or `None` when nothing has claimed a
limit. `complete_coeffs()` returns only the terms within that bound and
`leaked_coeffs()` only those beyond it.

The distinction is not cosmetic: a product of two truncated series reaches
deeper than either operand is known to, so terms past the bound are present and
not vouched for. Read `complete_coeffs()` wherever a wrong coefficient would be
worse than a missing one.

---

### `.denotation_order`

The shallowest order at which an **expressed zero** entered this value, or
`None` when every order is the classical one.

An expressed zero of magnitude m at grade -k denotes `m*(x-a)**k`, so from that
order down the jet is the exact jet of the function the expression *denotes*
rather than of its classical reading. It is still a true derivative - a real
slope, a real acceleration - of that function:

```python
x = R(3) + ZERO
f = x*x*(R(2)-R(2)) + x*x
f.denotation_order      # 1.0
[f.d(n) for n in range(4)]              # [9, 24, 26, 12]
# reading R(2)-R(2) as 2(x-3) gives g(x) = 2x**3 - 5x**2, whose jet is the same
```

It is carried rather than inferred, because division moves it: a residue at
order 1 divided by `ZERO` reaches order 0. Propagation is `min` on add and
subtract, `+ other.lead_order()` on multiply, `-` on divide.

`d(n)` reports when `n` is at or below it - see `CONVENTIONAL_STRICT`.

---

## Zero Handling

### `is_zero(x)`, `is_nothing(x)`, `is_vanishing(x)`

**Use `is_zero(x)` rather than `x == 0`.** `x == 0` compares against NOTHING,
so it is True for NOTHING and False for a dimensioned zero.

- `is_nothing(x)` - `Composite({})`, printed `∅`: no term at any dimension.
  This is the classical zero, an additive identity and a multiplicative
  annihilator, and it is the right accumulator seed.
- `is_vanishing(x)` - a zero that HAS a dimension, so it converts under R1.
  A written zero, `Composite({0: 0.0})`, is one.
- `is_zero(x)` - either of the above.

### `CANCELLATION`

Module switch, `"convert_and_keep"`. Which rule `+` and `-` follow; `*`, `/`,
negation and the functions are the same under all three. Zero Rules v2,
sections 1b and 1c, has the rules and the measurements.

```python
import composite.composite_lib as cl
cl.CANCELLATION = "convert_and_keep"   # the default
cl.CANCELLATION = "natural"
cl.CANCELLATION = "mechanical"         # the rule up to 2026-10-20
```

| | `"convert_and_keep"` | `"natural"` | `"mechanical"` |
|---|---|---|---|
| `R(5) + R(0)` | `<\|5\|_0 \|1\|_-1>` | `<\|5\|_0>` | `<\|5\|_0 \|1\|_-1>` |
| `R(5) - R(0)` | `<\|5\|_0 \|-1\|_-1>` | `<\|5\|_0>` | `<\|5\|_0 \|-1\|_-1>` |
| `R(0) + R(0)` | `<\|2\|_-1>` | `<\|0\|_0>` | `<\|2\|_-1>` |
| `R(6) - R(6)` | `<\|0\|_0>` | `<\|0\|_0>` | `<\|6\|_-1>` |
| `3*INF - 3*INF` | `<\|0\|_0>` | `<\|0\|_0>` | `<\|3\|_0>` |
| `(3*INF + 1) - 3*INF` | `<\|1\|_0>` | `<\|1\|_0>` | `<\|0\|_1 \|1\|_0>` |
| `(3*INF - 3*INF) + 1` | `<\|1\|_0 \|1\|_-1>` | `<\|1\|_0>` | `<\|4\|_0>` |
| `R(3) + 0`, then `x*x` | `<\|9\|_0 \|6\|_-1 \|1\|_-2>`, denoted | `<\|9\|_0>` | as convert_and_keep |

- `"convert_and_keep"` - `+` and `-` count, grade by grade, and leave no
  residue: a cancelled real or infinitesimal grade stays as an inert `|0|_k`,
  a cancelled infinite grade drops to `|0|_0`. A wholly zero operand converts
  by R1 and is kept, whatever it meets; it converts first and the sign then
  applies.
- `"natural"` - the same counting, but a wholly zero operand is counted too and
  adds nothing, so `R(5) + R(0)` is 5 and two zeros added stay one zero.
- `"mechanical"` - R1 on every operand, and a cancellation leaves the annihilated
  quantity one grade down, `a - a = a*h`, as set by `CANCELLATION_CARRIES`.

Any other value raises `ValueError` at the next `+` or `-`. Batch-dense data
follows `"mechanical"` whatever this says. To switch for a block and restore,
the tests use `tests/zero_modes.cancellation(rule)`.

### `CANCELLATION_CARRIES`

Module switch, `"quantity"`. What a cancellation deposits under
`CANCELLATION = "mechanical"`; the other two rules leave no residue and do not
read it. The examples below are mechanical.

`"quantity"` - the whole annihilated quantity, one grade down: **`a - a = a*h`**.

```python
R(6) - R(6)                  # |6|₋₁
R(-6) - R(-6)                # |-6|₋₁      the quantity, so the sign comes too
(R(3)+ZERO) - (R(3)+ZERO)    # <|3|₋₁ |1|₋₂>
```

The same number as a multiplication by zero: `(a - a) == a * R(0) == a * ZERO`,
and `(a - a) / R(0) == a` recovers the operand. So the residue is the jet of
`a*(x - x0)`, not of `a`: its coefficients are those of `a` one grade down, and
`d(k)` reads `k * a^(k-1)(x0)`. For `a = exp(3+h)`, `a.d(k)` is `e^3` at every
order and `(a - a).d(k)` is `0, e^3, 2e^3, 3e^3`. The result is marked denoted
from order 1 and `d(k)` warns, because the function it is the jet of is not `a`.

Equivalently, grade by grade: each dimension that cancels converts where it
stands, carrying its own coefficient. Multiplying by `h` shifts every grade down
by one, which is the shift R1 already prescribes, so this is R1 with the
coefficients kept rather than a separate rule.

Two properties follow rather than being imposed. `(a*h)/(b*h) = a/b`, so the
ratio between two zeros is the ratio of what they destroyed - 
`(2-2)/(3-3)` is 2/3 and `(x-x)/(y-y)` is `x/y`. And `a*(b*h)` equals `(a*b)*h`
by associativity of multiplication, so distributivity across a cancellation
holds, sign included.

**The cost.** `a - a` has the quantity on both sides and is unambiguous;
`a + (-a)` does not, the rule reads the left operand, and the two orders differ
by a sign. So `2 + (-2)` is `|2|₋₁` and `(-2) + 2` is `|-2|₋₁`: addition does not
commute on a cancelling pair. The difference is never more than a sign.
Multiplication is unaffected - commutative and associative either way.

**The dimensional cost.** A coefficient at grade `-k` carries units `[f]/[x]^k`,
so with `[h] = [x]` every term of `f(x0 + h)` has units `[f]`. Because `a - a` is
`a*h`, the residue is homogeneous in `[f]*[x]` instead: subtracting two energies
gives energy times length. It closes only when the infinitesimal is
dimensionless -- `v/c`, `alpha`, a bare perturbation parameter -- which is how the
physics demos are seeded. No units are tracked, so nothing warns about this;
`NotConventionalWarning` is about the denoted jet, not about dimensions. Where a
cancellation is reachable and the seed carries a unit, prefer a dimensionless
seed, or annihilate with `Composite({})`, which leaves no residue. Zero Rules v2
section 1 has the derivation.

`"magnitude"` - the previous rule, keeping only the deepest coefficient, so
`(3+h) - (3+h)` is `<|0|₀ |1|₋₂>`. Kept reachable for comparison. It keeps
addition commutative, and in exchange every composite zero has ratio 1 and
distributivity fails wherever the other factor is negative.

`CANCELLATION_SIGNED` applies only under `"magnitude"`, choosing whether that
coefficient keeps its sign.

### `CONVENTIONAL_STRICT`

Module switch, `False`. What `d(n)` does when the order asked for is one an
expressed zero contributed to. `False` warns and returns the value; `True`
raises `NotConventionalError`. Warning is the default because the value is a
true derivative of the denoted function, not a corrupted one.

### `set_max_order(n)` / `get_max_order()`

Cap how many orders the arithmetic returns. `set_max_order(None)` removes the
cap. It is the caller's economy, not a property of the backend, and it is
carried across a backend switch.

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
# cubic = <|8|₀ |12|₋₁ |6|₋₂ |1|₋₃>
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
- `to`: float - For limit (x -> to)

**Returns:** Composite

**Example:**

```python
trace(lambda x: (3*x + 1)/(x + 2), to=float('inf'))

# Output:
# === TRACE: lim(x->∞) ===
# Let x = |1|₁  (INF)
#     |3|₀  ×  |1|₁
#   = |3|₁
#     |3|₁  +  |1|₀
#   = <|3|₁ |1|₀>
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
# Translation:  <|9|₀ |6|₋₁ |1|₋₂>
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
)  # -> True
```

---

### `run_tests()`

Run the built-in test suite to verify library functionality.

**Returns:** bool - True if all tests pass

**Example:**

```python
from composite import run_tests
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
result = x + 5      # Composite + int -> Composite
result = x * 2.5    # Composite * float -> Composite
result = 10 / x     # int / Composite -> Composite
```

---

## Error Handling

### Common Exceptions

**`ZeroDivisionError`**

- Not raised for `x / 0`. A written zero is an expressed zero, which R1
  converts, so `R(1) / 0` is `|1|₁` - an infinity of definite order, not an
  error. Use `Composite({})` if you mean an absent denominator.

**`ValueError`**

- Raised for `ln(x)` when `x.st() < 0`. At `x.st() == 0` it does NOT raise:
  `ln(R(0))` is `|-1|_(0,1)`, a term on the log axis, because `R(0)` is the
  infinitesimal and `ln` of an infinitesimal is representable.
- Raised for `sqrt(x)` when `x.st() < 0`
- Raised for `asin(x)` when `|x.st()| ≥ 1`

**`TypeError`**

- Raised when `**` is given an exponent that is not an int, float or Composite.
  A float exponent is supported: `(R(4) + ZERO) ** 0.5` returns a series with
  standard part 2. `power(x, s)` is the same operation with an explicit
  `terms=`.

**`LimitDoesNotExistError`**

- Raised by division when the divisor is `Composite({})` (NOTHING). An absent
  denominator is indeterminate, unlike an expressed zero.

**`NotRepresentableError`**

- Raised where a value has no composite at all: `exp` of a positive grade
  (`exp(-1/x**2)` at 0), or a bounded transcendental at an unbounded argument
  (`sin(1/x)` as x -> 0, which is a range rather than a point).

**`StandardPartUndefinedError`**

- Raised by `st()` when the dominant grade is positive, so there is no standard
  part to take.

**`NotConventionalError`**

- Raised by `d(n)` only when `CONVENTIONAL_STRICT` is True and the order asked
  for is one an expressed zero contributed to. See `denotation_order`.

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

    # Bad - a written zero, not a seed
    x = R(3) + 0
    ```

    Under the default rule the `0` converts to `ZERO`, so this gives the same jet
    but marks it denoted, and every `d(n)` warns. Under `CANCELLATION = "natural"`
    the `0` is counted away and there are no derivatives at all.

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

**Typical performance:** depends entirely on the shape of the problem and on
the backend; see the Performance section of the README for measured figures
rather than a single ratio. `use_dense_series()` is the right backend for
calculus, `use_dict()` for small scattered composites, and the default
`use_sparse_dense()` for sparse supports over a large domain.

---

## See Also

- [**10-Minute Tutorial**](Tutorial%20-%20Getting%20Started.md) - Get started quickly
- [**Implementation Guide**](Implementation%20Guide.md) - How it works internally
- [**Zero Rules v2**](Zero%20Rules%20v2%20%E2%80%94%20Formal%20Specification%20(DRAFT).md) - What a zero coefficient means, and how it behaves
- [**Examples**](Examples.md) - Code snippets for common tasks
- [**Roadmap (DRAFT)**](Roadmap%20(DRAFT).md) - What's next
