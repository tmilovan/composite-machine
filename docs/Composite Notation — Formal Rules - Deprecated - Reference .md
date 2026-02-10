# Composite Notation — Formal Rules - Deprecated - Reference

This document formalizes the rules for composite number notation and dimensional collapse/expansion operations developed through systematic testing.

---

## 1. Composite Number Representation

### Definition 1.1: Composite Number

A composite number is a three-component value:

```
⟨ i ; r | z ⟩
```

Where:

- `i` = infinity component (coefficient of ∞ basis)
- `r` = rational component (coefficient of 1 basis)
- `z` = zero component (coefficient of 0 basis)

**Interpretation:** The composite `⟨i; r | z⟩` represents the value `i×∞ + r×1 + z×0`

### Definition 1.2: Extended Notation for Higher Orders

Higher-order zeroes and infinities use additional separators:

```
⟨ ∞³ ;;; ∞² ;; ∞ ; r | 0 || 0² ||| 0³ ⟩
```

- `||` separates first-order from second-order zeroes
- `|||` separates second-order from third-order zeroes
- `;;` separates first-order from second-order infinities
- `;;;` separates second-order from third-order infinities

### Definition 1.3: Coefficient Notation

**The notation `a(0ⁿ)` means "coefficient a times basis 0ⁿ".**

When written without an explicit coefficient, the coefficient 1 is implied:

| Shorthand | Explicit form | Meaning |
| --- | --- | --- |
| `0` | `1(0)` | 1 × 0¹ |
| `0²` | `1(0²)` | 1 × 0² |
| `∞` | `1(∞)` | 1 × ∞¹ |
| `∞²` | `1(∞²)` | 1 × ∞² |
| `5(0)` | `5(0)` | 5 × 0¹ |

This is **not** a recursive definition — `0ⁿ` and `∞ⁿ` are primitive basis elements, and the coefficient notation simply makes the multiplier explicit.

### Definition 1.4: Structural Absence

- Omitted components indicate **structural absence** (component does not exist)
- `⟨;5|⟩` means only rational 5 exists; infinity and zero are absent
- Absent components do not participate in operations

---

## 2. Value Types and Natural Segments

### Definition 2.1: Value Types

Every value belongs to exactly one **natural type**:

| Type | Examples | Natural Segment |
| --- | --- | --- |
| Plain number | `2`, `5`, `-3.7`, `π` | Rational |
| Symbolic zero | `a(0)`, `a(0²)`, `a(0³)` | Zero dimension |
| Symbolic infinity | `a(∞)`, `a(∞²)`, `a(∞³)` | Infinity dimension |

### Definition 2.2: Natural Segment Rule

**A value is in its natural segment when its type matches the dimension it occupies.**

- Plain numbers naturally belong in the rational segment
- Symbolic zeroes naturally belong in the zero segment
- Symbolic infinities naturally belong in the infinity segment

### Definition 2.3: Mixed State

A segment is in a **mixed state** when it contains values whose natural type differs from the segment's dimension.

*Example:* `⟨_; 5(0)+2(0) | _⟩` has symbolic zeroes in the rational segment → mixed state

---

## 3. Collapse and Expansion Operations

### Definition 3.1: Trigger Condition

Dimensional collapse or expansion is triggered **only** by multiplying or dividing the **entire composite number** by `0` or `∞`.

| Operation | Effect | Direction |
| --- | --- | --- |
| `× 0` or `/ ∞` | **Collapse** | Shift toward zero (→) |
| `× ∞` or `/ 0` | **Expand** | Shift toward infinity (←) |

### Definition 3.2: Multiplication by Symbolic Values

Multiplication by symbolic values decomposes into scalar and basis operations:

- `× a(0) = × a × 0` (scalar multiplication, then collapse)
- `/ a(0) = / a / 0 = × (1/a) × ∞` (scalar division, then expand)

### Definition 3.3: Composite × Composite Multiplication

**Composite multiplication follows standard algebraic rules: distribute, then simplify using order cancellation.**

Given:

- A = `⟨a₁; a₂ | a₃⟩` = a₁×∞ + a₂×1 + a₃×0
- B = `⟨b₁; b₂ | b₃⟩` = b₁×∞ + b₂×1 + b₃×0

**A × B** distributes to 9 terms, then simplifies via order cancellation (`∞×0 = 1`):

```
A × B = a₁b₁(∞²) + (a₁b₂ + a₂b₁)(∞) + (a₁b₃ + a₂b₂ + a₃b₁)(1)
       + (a₂b₃ + a₃b₂)(0) + a₃b₃(0²)
```

**Commutativity:** A × B = B × A ✔

- Follows from commutativity of scalar multiplication and symmetry of order cancellation

**Associativity:** (A × B) × C = A × (B × C) ✔

- Follows from associativity of scalar multiplication and basis element multiplication

**Worked example:**

```
A = ⟨1; 2 | 1⟩,  B = ⟨0; 3 | 2⟩

A × B = (∞ + 2 + 0) × (3 + 2×0)
      = 3∞ + 2(∞×0) + 6 + 4×0 + 3×0 + 2×0²
      = 3∞ + 2 + 6 + 7×0 + 2×0²
      = ⟨3; 8 | 7 || 2⟩
```

---

## 4. Step-by-Step Evaluation Process

### Rule 4.1: Two-Phase Process

Collapse/expansion proceeds in two phases:

**Phase 1: Multiplication**

Multiply all components by the trigger (`0` or `∞`), keeping results in their current positions.

**Phase 2: Migration**

Evaluate dimensional interactions step-by-step, migrating values toward their natural segments.

### Rule 4.2: Processing Order

| Operation | Processing Order | Rationale |
| --- | --- | --- |
| **Collapse** | Left → Right (∞ → r → 0) | Catch values as they fall |
| **Expand** | Right → Left (0 → r → ∞) | Lift values as they rise |

### Rule 4.3: Dimensional Interaction Rules

**Core principle:** Since `∞ = 1/0`, multiplying by infinity is equivalent to dividing by zero. Orders cancel arithmetically.

**Expansion (×∞):**

```
a(0ⁿ) × ∞ = a(0ⁿ) × (1/0) = a(0ⁿ) / 0 = a(0ⁿ⁻¹)
```

The zero in the numerator cancels one zero from the denominator, reducing the order by 1.

*Example:* `2(0) × ∞ = 2(0)/0 = 2` → plain coefficient, naturally belongs in rational

**Collapse (×0):**

```
a × 0 = a(0)
```

Multiplying by zero increases order by 1.

*Example:* `5 × 0 = 5(0)` → symbolic zero, naturally belongs in zero segment

**Order cancellation formula:**

- `0ⁿ × ∞ᵐ = 0ⁿ⁻ᵐ` if n > m
- `0ⁿ × ∞ᵐ = ∞ᵐ⁻ⁿ` if m > n
- `0ⁿ × ∞ᵐ = 1` if n = m (orders fully cancel → plain rational)

### Rule 4.5: Order Zero (Identity Order)

**Order zero is the identity/exit point where zero and infinity families converge to 1.**

- `0⁰ = 1` (any number to the zero power is 1)
- `∞⁰ = 1`
- `a(0⁰) = a(1) = a × 1 = a` (collapses to plain number)

**Order spectrum:**

| Order | Zero family | Infinity family |
| --- | --- | --- |
| 3 | 0³ | ∞³ |
| 2 | 0² | ∞² |
| 1 | 0 | ∞ |
| **0** | **1** | **1** |
| -1 | ∞ | 0 |
| -2 | ∞² | 0² |

**Key insight:** When dimensional interaction reduces order to 0, the value exits the symbolic family and becomes a plain rational number.

**Example:** `a(0¹) × ∞ = a(0⁰) = a` → plain number, migrates to rational segment

### Rule 4.4: Mixed State Persistence

**Values remain in mixed states until they reach their natural segment.**

During step-by-step evaluation, a symbolic zero landing in rational position stays there temporarily until the next collapse step moves it to the zero segment.

---

## 5. Reversibility Conditions

### Rule 5.1: Preservation of Expressions

**To maintain reversibility, do not evaluate additions prematurely.**

Keep expressions like `5(0)+2(0)` in symbolic form rather than simplifying to `7(0)`.

**Rationale:** Addition is compressive—it loses the information about original components.

**Important clarification:** Distribution over multiplication is still mandatory (standard algebra):

```
(5(0)+2(0)) × ∞ = 5(0)×∞ + 2(0)×∞ = 5 + 2
```

The result `5 + 2` can remain unevaluated to preserve provenance, or collapse to `7` when reversal is no longer needed.

**What to preserve vs. what to evaluate:**

| Expression | Preserve? | Reason |
| --- | --- | --- |
| `5(0) + 2(0)` | Yes | Tracks original segment sources |
| `(a+b) × c` | No → distribute | Standard algebra required |
| `5 + 2` (post-operation) | Optional | Preserve if reversal needed |

### Rule 5.2: Provenance Tracking

**Provenance is preserved per-step, not globally — exactly like standard algebra.**

In standard algebra:

- `5 × 3 = 15` → reversible: `15 ÷ 3 = 5`
- But after `15 + 1 = 16`, you can't recover `5` without additional information

This system extends the same principle to zero:

- `5 × 0 = 5(0)` → reversible: `5(0) × ∞ = 5`
- But after further operations, the original structure may be lost

**What this achieves:** Operations with zero become reversible in the same local sense that operations with any other number are reversible. Information loss over multiple steps is expected behavior, not a flaw.

### Rule 5.3: Step Reversibility

**Each individual step is reversible when expressions are preserved.**

| Forward Step | Reverse Step |
| --- | --- |
| Multiply by 0 | Multiply by ∞ |
| Multiply by ∞ | Multiply by 0 |
| Collapse ∞ → r | Expand r → ∞ |
| Collapse r → 0 | Expand 0 → r |

### Rule 5.4: Global vs Local Reversibility

- **Local (step-by-step):** Each step is reversible if expressions are preserved
- **Global (full collapse then expand):** NOT reversible—structure is lost when all values migrate to the same segment

---

## 6. Worked Examples

### Example 6.1: Full Collapse with Preserved Expressions

**Forward:** `⟨2(0);5|0⟩ × 0`

**Phase 1: Multiply all by 0**

- Infinity: `2(0) × 0 = 2(0²)`
- Rational: `5 × 0 = 5(0)`
- Zero: `0 × 0 = 1(0²)` *(since 0 ≡ 1(0))*

State: `⟨2(0²); 5(0) | 1(0²)⟩`

**Phase 2: Collapse step-by-step (left to right)**

*Step 1: ∞ → rational*

- `2(0²) × ∞ = 2(0)` → moves to rational
- State: `⟨_; 5(0)+2(0) | 1(0²)⟩`

*Step 2: rational → zero*

- `5(0)` → coefficient 5 in zero
- `2(0)` → coefficient 2 in zero
- State: `⟨_; _ | 5+2 || 1⟩`

### Example 6.2: Step Reversal Verification

**After multiply step:**

State: `⟨2(0²); 5(0) | 1(0²)⟩`

**Reverse (multiply by ∞):**

- `2(0²) × ∞ = 2(0)` ✓
- `5(0) × ∞ = 5` ✓
- `1(0²) × ∞ = 1(0) = 0` ✓

**Result:** `⟨2(0); 5 | 0⟩` ✓ **Matches original!**

### Example 6.3: Expansion

**Forward:** `⟨2(0);5|0⟩ × ∞`

**Phase 1: Multiply all by ∞**

- Infinity: `2(0) × ∞ = 2`
- Rational: `5 × ∞ = 5(∞)`
- Zero: `0 × ∞ = 1(0) × ∞ = 1`

State: `⟨2; 5(∞)+1 | _⟩`

**Phase 2: Expand step-by-step (right to left)**

*Step 1: Rational → infinity*

- `5(∞)` is symbolic infinity → migrates to infinity
- `1` and `2` are plain → stay in rational
- State: `⟨5(∞)+2; 1 | _⟩`

*(Note: keeping expressions unevaluated)*

---

## 7. Summary of Key Principles

1. **Natural Segment Rule:** Values migrate toward their natural dimensional home
2. **Step-by-Step Processing:** Collapse left-to-right, expand right-to-left
3. **Expression Preservation:** Avoid addition to maintain reversibility
4. **Provenance Tracking:** Track origin segment for each value
5. **Local Reversibility:** Each step is reversible; full round-trips are not
6. **Canonical Form:** A composite is canonical when all segments contain only plain coefficients

© Toni Milovan. Documentation licensed under CC BY-SA 4.0. Code licensed under AGPL-3.0.
