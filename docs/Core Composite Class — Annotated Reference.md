# Core Composite Class — Annotated Reference

<aside>
📐

**The essential mechanism of Composite Calculus in ~150 lines.**

This is the irreducible core: a number that carries coefficients at integer dimensions.

Everything else — derivatives, limits, integrals, series, Turing machines — is built on this.

</aside>

---

## The Core Class

```python
"""
core_composite.py — The Essential Mechanism
============================================

A Composite number is a Laurent-like object:

    |a₋₂|₋₂  +  |a₋₁|₋₁  +  |a₀|₀  +  |a₁|₁  +  ...

Each term has:
  - a COEFFICIENT (any real number)
  - a DIMENSION   (any integer)

The dimension encodes what KIND of quantity the coefficient represents:

    Dimension   Meaning                  Example
    ─────────   ───────────────────────   ──────────────────────
       0        Real number              |5|₀ = the number 5
      -1        First infinitesimal      |1|₋₁ = structural zero (ZERO)
      -2        Second infinitesimal     |1|₋₂ = zero² (finer grain)
      -n        nth-order infinitesimal   carries the nth derivative ÷ n!
      +1        First infinity           |1|₁ = structural infinity (INF)
      +n        nth-order infinity        grows without bound

The KEY INSIGHT:
    When you compute f(a + h) where h = |1|₋₁, the result is a
    composite whose negative-dimension coefficients ARE the Taylor
    coefficients of f around a. No limits needed — the algebra
    produces them automatically.

    Example: (3 + h)² = |9|₀ + |6|₋₁ + |1|₋₂
                         ↑       ↑        ↑
                        f(3)   f'(3)/1!  f''(3)/2!

Author: Toni Milovan
License: AGPL-3.0 (commercial licensing: tmilovan@fwd.hr)
"""

import math

class Composite:
    """
    A number with dimensional structure: |coefficient|_dimension.

    Internally stored as a dictionary {dimension: coefficient}.
    Zero-valued coefficients are pruned automatically (threshold: 1e-15)
    to keep the representation sparse and clean.

    The arithmetic rules are simple:
      - Addition:       combine coefficients at matching dimensions
      - Multiplication: dimensions ADD, coefficients MULTIPLY (convolution)
      - Division:       dimensions SUBTRACT, coefficients DIVIDE

    These are the SAME rules as polynomial/Laurent series arithmetic.
    That's not a coincidence — it's the whole point.
    """

    # =========================================================================
    # STORAGE
    # =========================================================================
    # Using __slots__ for memory efficiency. The only instance attribute
    # is 'c': a dict mapping integer dimensions to float coefficients.
    #
    #   self.c = {0: 5.0, -1: 3.0}   represents   |5|₀ + |3|₋₁
    #
    # Empty dict {} represents the additive identity (true zero, NOT
    # the structural ZERO which is |1|₋₁).
    # =========================================================================

    __slots__ = ['c']

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================

    def __init__(self, coefficients=None):
        """
        Create a Composite from:
          - None          → empty (additive identity)
          - int or float  → real number at dimension 0
          - dict          → explicit {dimension: coefficient} map

        Coefficients with |value| ≤ 1e-15 are pruned to keep
        the representation clean and avoid floating-point dust.

        Examples:
            Composite()           → {}           (additive identity)
            Composite(5)          → {0: 5.0}     i.e. |5|₀
            Composite({-1: 1.0})  → {-1: 1.0}    i.e. |1|₋₁ = ZERO
        """
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            # A plain number lives entirely at dimension 0.
            # Python zero maps to empty dict (not ZERO = |1|₋₁).
            self.c = {0: float(coefficients)} if coefficients != 0 else {}
        elif isinstance(coefficients, dict):
            # Prune near-zero coefficients (floating-point dust).
            self.c = {k: v for k, v in coefficients.items() if abs(v) > 1e-15}
        else:
            raise TypeError(f"Cannot create Composite from {type(coefficients)}")

    # -------------------------------------------------------------------------
    # Named constructors for the three fundamental objects
    # -------------------------------------------------------------------------

    @classmethod
    def zero(cls):
        """
        Structural zero: ZERO = |1|₋₁

        This is NOT the additive identity (which is Composite()).
        It is an infinitesimal — a number smaller than any positive
        real but not equal to nothing. It has coefficient 1 at
        dimension -1.

        Key property: multiplying by ZERO does NOT destroy information.
            R(5) * ZERO = |5|₋₁   (coefficient 5 moved to dimension -1)
            |5|₋₁ / ZERO = |5|₀   (moved back — fully reversible)

        This reversibility is what makes 0/0 and ∞×0 well-defined
        in composite arithmetic.
        """
        return cls({-1: 1.0})

    @classmethod
    def infinity(cls):
        """
        Structural infinity: INF = |1|₁

        The dimensional dual of ZERO. Lives at dimension +1.

        Key properties:
            INF * ZERO = |1|₀ = R(1)   (dimensions -1 + 1 = 0)
            R(5) * INF = |5|₁           (5 at the infinity level)

        INF is not a "limit" — it's a concrete algebraic object
        with dimension +1 and coefficient 1.
        """
        return cls({1: 1.0})

    @classmethod
    def real(cls, value):
        """
        Real number: |value|₀

        A pure real number living entirely at dimension 0.
        This is the embedding of ℝ into composite space.

        Example:
            Composite.real(3) = |3|₀
        """
        return cls({0: float(value)})

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def __repr__(self):
        """
        Human-readable notation: |coefficient|_dimension

        Uses Unicode subscript characters for dimensions.
        Terms are sorted by dimension, highest first (so real
        part appears before infinitesimal parts).

        Examples:
            |9|₀ + |6|₋₁ + |1|₋₂
            |1|₁                     (INF)
            |0|₀                     (empty composite)
        """
        if not self.c:
            return "|0|₀"

        sub = "₀₁₂₃₄₅₆₇₈₉"

        def fmt_dim(n):
            """Convert integer dimension to subscript string."""
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            else:
                return "₋" + ''.join(sub[int(d)] for d in str(-n))

        def fmt_coeff(c):
            """Format coefficient, dropping unnecessary decimals."""
            if isinstance(c, float) and c == int(c):
                return str(int(c))
            elif isinstance(c, float):
                return f"{c:.6g}"
            return str(c)

        terms = sorted(self.c.items(), key=lambda x: -x[0])
        parts = [f"|{fmt_coeff(coeff)}|{fmt_dim(dim)}" for dim, coeff in terms]
        return " + ".join(parts)

    # =========================================================================
    # ARITHMETIC: THE HEART OF THE SYSTEM
    # =========================================================================
    #
    # These four operations (+, -, ×, ÷) on composites are the ONLY
    # mechanism. Every calculus result — derivatives, limits, integrals,
    # series, even Turing machine steps — emerges from these rules.
    #
    # The rules mirror Laurent polynomial arithmetic:
    #   Addition:       pointwise by dimension
    #   Multiplication: convolution (dimensions add, coefficients multiply)
    #   Division:       deconvolution (dimensions subtract)
    #
    # =========================================================================

    def __add__(self, other):
        """
        Addition: combine coefficients at each dimension.

        |a|ₙ + |b|ₙ = |a+b|ₙ    (same dimension: add coefficients)
        |a|ₙ + |b|ₘ = |a|ₙ + |b|ₘ  (different dimensions: keep both)

        Example:
            (|3|₀ + |2|₋₁) + (|1|₀ + |5|₋₁) = |4|₀ + |7|₋₁

        This is identical to adding two polynomials term by term.
        """
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return Composite(result)

    def __radd__(self, other):
        """Allow int/float + Composite (addition is commutative)."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction: add the negation."""
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) - coeff
        return Composite(result)

    def __rsub__(self, other):
        """Allow int/float - Composite."""
        return Composite(other).__sub__(self)

    def __neg__(self):
        """Negation: negate all coefficients, dimensions unchanged."""
        return Composite({k: -v for k, v in self.c.items()})

    def __mul__(self, other):
        """
        Multiplication: dimensions ADD, coefficients MULTIPLY.

        |a|ₙ × |b|ₘ = |a·b|₍ₙ₊ₘ₎

        This is the CRUCIAL rule. It means:
          - Multiplying by ZERO (|1|₋₁) SHIFTS dimensions down by 1.
            |5|₀ × |1|₋₁ = |5|₋₁    (5 moves from real to infinitesimal)
          - Multiplying by INF (|1|₊₁) SHIFTS dimensions up by 1.
            |5|₋₁ × |1|₊₁ = |5|₀    (5 moves from infinitesimal to real)
          - Multiplying two ZEROs gives a DEEPER infinitesimal.
            |1|₋₁ × |1|₋₁ = |1|₋₂

        For multi-term composites, this is polynomial convolution:
            (|a|₀ + |b|₋₁) × (|c|₀ + |d|₋₁)
            = |ac|₀ + |ad+bc|₋₁ + |bd|₋₂

        This is EXACTLY how (a + bh)(c + dh) expands — because h = |1|₋₁
        and h² = |1|₋₂. The dimensional algebra IS Taylor expansion.
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication: scale all coefficients.
            return Composite({k: v * other for k, v in self.c.items()})
        result = {}
        for d1, c1 in self.c.items():
            for d2, c2 in other.c.items():
                dim = d1 + d2       # Dimensions ADD
                result[dim] = result.get(dim, 0) + c1 * c2  # Coefficients MULTIPLY
        return Composite(result)

    def __rmul__(self, other):
        """Allow int/float × Composite (multiplication is commutative)."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Division: dimensions SUBTRACT, coefficients DIVIDE.

        |a|ₙ ÷ |b|ₘ = |a/b|₍ₙ₋ₘ₎

        This is the inverse of multiplication:
          - Dividing by ZERO (|1|₋₁) SHIFTS dimensions UP by 1.
            |5|₋₁ ÷ |1|₋₁ = |5|₀    (restores the real number 5)
          - This makes ×ZERO fully reversible: no information is lost.

        For single-term divisors, division is exact.
        For multi-term divisors, we use polynomial long division
        (see _poly_divide), which produces a Laurent series.

        Note: dividing by Python's 0 raises ZeroDivisionError.
        Use ZERO for structural zero division.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide by Python zero. Use ZERO for structural zero."
                )
            return Composite({k: v / other for k, v in self.c.items()})

        if len(other.c) == 0:
            raise ZeroDivisionError("Cannot divide by empty composite")

        # Fast path: single-term divisor → exact dimension shift.
        if len(other.c) == 1:
            div_dim, div_coeff = list(other.c.items())[0]
            result = {}
            for dim, coeff in self.c.items():
                result[dim - div_dim] = coeff / div_coeff
            return Composite(result)

        # Multi-term divisor → polynomial long division.
        return _poly_divide(self, other)[0]

    def __rtruediv__(self, other):
        """Allow int/float ÷ Composite."""
        return Composite(other).__truediv__(self)

    def __pow__(self, n):
        """
        Integer power: repeated multiplication.

        Negative powers use: x⁻ⁿ = 1 / xⁿ

        Since each multiplication is a convolution, x**n produces
        coefficients at dimensions down to -n. This is why
        (a + h)**n automatically generates n Taylor coefficients.
        """
        if not isinstance(n, int):
            raise TypeError("Power must be integer (use power() for real exponents)")
        if n == 0:
            return Composite({0: 1})  # x⁰ = 1 for all x
        if n < 0:
            return Composite({0: 1}) / (self ** (-n))
        result = Composite({0: 1})
        for _ in range(n):
            result = result * self
        return result

    # =========================================================================
    # EXTRACTION: READING RESULTS FROM THE COMPOSITE
    # =========================================================================
    #
    # After computing f(a + h), the composite holds ALL information
    # about f at the point a. These methods extract specific pieces.
    #
    # =========================================================================

    def st(self):
        """
        Standard part: the coefficient at dimension 0.

        This extracts the "real number" part of the composite,
        discarding all infinitesimal and infinite components.

        For f(a + h), st() returns f(a) — the function value.

        For a limit computation, st() returns the limit value.

        Example:
            (|9|₀ + |6|₋₁ + |1|₋₂).st() = 9
        """
        return self.c.get(0, 0.0)

    def coeff(self, dim):
        """
        Get the coefficient at a specific dimension.

        This is the raw coefficient without any factorial scaling.
        Useful for direct inspection of the Laurent structure.

        Example:
            x = |9|₀ + |6|₋₁ + |1|₋₂
            x.coeff(0)  = 9
            x.coeff(-1) = 6
            x.coeff(-2) = 1
            x.coeff(5)  = 0   (absent dimensions return 0)
        """
        return self.c.get(dim, 0.0)

    def d(self, n=1):
        """
        Extract the nth derivative.

        The coefficient at dimension -n holds f⁽ⁿ⁾(a) / n!
        (the nth Taylor coefficient). So:

            d(n) = coeff(-n) × n!

        This recovers the actual derivative value.

        Examples (for f(x) = x² at x = 3):
            composite = |9|₀ + |6|₋₁ + |1|₋₂

            d(1) = coeff(-1) × 1! = 6 × 1 = 6    ← f'(3) = 2·3 = 6  ✓
            d(2) = coeff(-2) × 2! = 1 × 2 = 2    ← f''(3) = 2      ✓
        """
        return self.c.get(-n, 0.0) * math.factorial(n)

    # =========================================================================
    # COMPARISON
    # =========================================================================

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) == 0

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) < 0

    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) <= 0

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) > 0

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        return _compare(self, other) >= 0

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _compare(a, b):
    """
    Lexicographic comparison by dimension, highest dimension first.

    The composite with a larger coefficient at the highest dimension
    is "greater". This gives a total order consistent with the
    intuition that infinite quantities dominate finite ones, and
    finite quantities dominate infinitesimals.

    Example:
        |1|₁ > |1000000|₀  (infinity beats any real)
        |5|₀ > |3|₀         (among reals, normal ordering)
        |3|₀ > |1000|₋₁     (any real beats any infinitesimal)
    """
    all_dims = set(a.c.keys()) | set(b.c.keys())
    if not all_dims:
        return 0
    for dim in sorted(all_dims, reverse=True):
        ca = a.c.get(dim, 0)
        cb = b.c.get(dim, 0)
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0

def _poly_divide(numerator, denominator, max_terms=20):
    """
    Polynomial long division for multi-term divisors.

    When the divisor has more than one term, we can't do a simple
    dimension shift. Instead, we perform iterative long division:

    1. Find the leading term of the remainder (highest dimension).
    2. Divide it by the leading term of the denominator.
    3. Subtract the result × denominator from the remainder.
    4. Repeat until the remainder is negligible or max_terms reached.

    This naturally produces Laurent series expansions. For example,
    1 / (1 - x) with x = |1|₋₁ will produce 1 + x + x² + ...
    which is the geometric series — the mechanism behind
    "computation as division" in the self-hosted execution work.

    Returns:
        (quotient, remainder) — both Composite objects.
    """
    if not denominator.c:
        raise ZeroDivisionError("Cannot divide by zero polynomial")

    # Identify the leading term of the denominator.
    denom_dims = sorted(denominator.c.keys(), reverse=True)
    lead_dim = denom_dims[0]
    lead_coeff = denominator.c[lead_dim]

    quotient = Composite({})
    remainder = Composite(dict(numerator.c))

    for _ in range(max_terms):
        if not remainder.c:
            break

        # Leading term of current remainder.
        rem_dims = sorted(remainder.c.keys(), reverse=True)
        rem_lead_dim = rem_dims[0]
        rem_lead_coeff = remainder.c[rem_lead_dim]

        # Quotient term: shift dimension, scale coefficient.
        q_dim = rem_lead_dim - lead_dim
        q_coeff = rem_lead_coeff / lead_coeff

        # Accumulate into quotient, subtract from remainder.
        quotient = quotient + Composite({q_dim: q_coeff})
        subtract_term = Composite({q_dim: q_coeff}) * denominator
        remainder = remainder - subtract_term

        # Prune floating-point dust from remainder.
        remainder.c = {k: v for k, v in remainder.c.items() if abs(v) > 1e-14}

    return quotient, remainder

# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def R(x):
    """
    Create a real-valued composite: |x|₀

    Short for Composite.real(x). Used constantly in expressions:
        R(3) + ZERO   →  |3|₀ + |1|₋₁   (i.e., 3 + h)
    """
    return Composite.real(x)

# The two structural constants.
# These are the atoms from which all composite computation is built.

ZERO = Composite.zero()       # |1|₋₁  — the infinitesimal
INF  = Composite.infinity()   # |1|₊₁  — the infinite
h    = ZERO                   # Alias: h is the infinitesimal probe
```

---

## What This Gives You

With **only** the code above (~150 lines of mechanism, no transcendental functions, no API layer), you can already:

```python
# ── Derivatives (any polynomial) ──────────────────────────
x = R(3) + h           # x = 3 + infinitesimal
result = x**2          # |9|₀ + |6|₋₁ + |1|₋₂
result.st()            # → 9    (value)
result.d(1)            # → 6    (first derivative)
result.d(2)            # → 2    (second derivative)

# ── Limits (algebraic) ────────────────────────────────────
x = R(2) + ZERO                           # x → 2
((x**2 - R(4)) / (x - R(2))).st()         # → 4  (0/0 resolved algebraically)

# ── Reversible zero multiplication ────────────────────────
a = R(5) * ZERO        # |5|₋₁  — information preserved
a / ZERO               # |5|₀   — fully recovered

# ── 0/0, ∞×0 are well-defined ────────────────────────────
(ZERO / ZERO).st()     # → 1
(INF * ZERO).st()      # → 1
((R(5) * ZERO) / ZERO).st()  # → 5

# ── Division generates series (computation = division) ────
one = Composite({0: 1})
x = Composite({-1: 1})  # = ZERO
result = one / (one - x)  # → 1 + x + x² + ... (geometric series)
```

---

## Architecture Summary

| Layer | What it does | Lines |
| --- | --- | --- |
| **Storage** | `dict` mapping `int → float` — sparse Laurent coefficients | ~10 |
| **Arithmetic** | `+` pointwise, `×` convolution (dims add), `÷` deconvolution (dims subtract) | ~60 |
| **Extraction** | `st()` = dim 0, `d(n)` = dim −n × n!, `coeff(d)` = raw | ~15 |
| **Constants** | `ZERO` = |1|₋₁, `INF` = |1|₊₁, `R(x)` = |x|₀ | ~5 |
| **Long division** | Multi-term ÷ via iterative leading-term cancellation | ~30 |

**Total: ~120 lines of mechanism. Everything else is built on top.**

© Toni Milovan. Documentation licensed under CC BY-SA 4.0. Code licensed under AGPL-3.0.
