# Revised Notation System — Formal Rules - Current!

This document defines the revised composite notation using subscript dimensions, which resolves the structural absence vs zero-value problem.

---

## Core Principles

### Principle 1: Constructed Dimensions

**Derived dimensions (infinities and zeroes) don't exist unless explicitly created by multiplication or division with zero (or infinity).**

This is not a special case — it's the fundamental construction rule of the system.

### Principle 2: Only Existing Dimensions Participate

**If a dimension is mentioned in the composite, we calculate with it. If not mentioned, that dimension does not exist and nothing is calculated there.**

No phantom expansion from non-existent dimensions.

### Principle 3: Dimensional Shift Requires ×0 or ×∞

**Values cannot change dimensions unless the whole number is multiplied or divided by zero(es) or infinity(ies).**

### Principle 4: Shift Magnitude Matches Multiplier Order

**Numbers shift exactly by the order of the multiplier:**

- `×0` = shift down 1 dimension
- `×0²` = shift down 2 dimensions
- `×∞` = shift up 1 dimension
- `×∞²` = shift up 2 dimensions

---

## Notation

### The Dimension Scale

Dimensions are indexed on an integer scale with **rational as origin (0)**:

```
... |_|₃  |_|₂  |_|₁  |r|  |_|₋₁  |_|₋₂  |_|₋₃ ...
     ∞³    ∞²    ∞     R     0      0²     0³
```

| Dimension | Subscript | Contains |
| --- | --- | --- |
| ∞³ | ₃ | Third-order infinities |
| ∞² | ₂ | Second-order infinities |
| ∞ | ₁ | First-order infinities |
| **Rational** | *(none)* | Plain numbers |
| 0 | ₋₁ | First-order zeroes |
| 0² | ₋₂ | Second-order zeroes |
| 0³ | ₋₃ | Third-order zeroes |

### Writing Composites

**Format:** `|coeff|ₙ` where n is the dimension subscript (omitted for rational)

**Examples:**

- `|5|` = rational 5 (only rational dimension exists)
- `|5|₋₁` = 5 first-order zeroes
- `|3|₁` = 3 first-order infinities
- `|2|₁ |5| |1|₋₁` = 2∞ + 5 + 1(0)

### The Value Zero and Infinity

**`0 = 1(0) = |0| = |1|₋₁`** — these are the **same value** in two equivalent representations (the Ladder of Absences: `|0|ₙ = |1|ₙ₋₁`).

- `|0|₀` — zero expressed as coefficient 0 at dimension 0
- `|1|₋₁` — zero expressed as coefficient 1 at dimension -1

Both follow the same multiplication rule: `|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ`. There is no annihilation — only standard coefficient arithmetic and dimension addition.

**`∞ = |1|₁`** — one first-order infinity (the "unit infinity").

**Shift operators:** `×0` and `×∞` are dimension-shift operators equivalent to `×|1|₋₁` and `×|1|₁` respectively.

### Notation Conventions

- **Ordering:** Terms are written high-to-low by dimension: `|3|₂ |5|₁ |2| |1|₋₁` (not `|1|₋₁ |2| |5|₁ |3|₂`)
- **Spacing:** Terms are separated by spaces for readability
- **Zero coefficients:** Per Decision 2, `|0|ₙ` is preserved for provenance tracking but may be omitted in simplified display when provenance is not needed

---

## Operations

### Multiplication by 0ⁿ (Collapse)

**Effect:** Shift all existing dimensions down by n

```
|a|ₖ × 0ⁿ = |a|ₖ₋ₙ
```

**Example:** `|2|₁ |5| |1|₋₁ × 0`

- `|2|₁` → `|2|₀` = `|2|` (rational)
- `|5|` → `|5|₋₁`
- `|1|₋₁` → `|1|₋₂`
- **Result:** `|2| |5|₋₁ |1|₋₂`

### Multiplication by ∞ⁿ (Expansion)

**Effect:** Shift all existing dimensions up by n

```
|a|ₖ × ∞ⁿ = |a|ₖ₊ₙ
```

**Example:** `|2| |5|₋₁ |1|₋₂ × ∞`

- `|2|` → `|2|₁`
- `|5|₋₁` → `|5|₀` = `|5|`
- `|1|₋₂` → `|1|₋₁`
- **Result:** `|2|₁ |5| |1|₋₁` ✓ Matches original!

### Composite × Composite

Uses distribution. Each term's dimension is the **sum** of the two dimensions:

```
|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ
```

**Key cases:**

- `|a|₁ × |b|₋₁ = |a×b|₀` (∞ × 0 → rational)
- `|a|₋₁ × |b|₋₁ = |a×b|₋₂` (0 × 0 → 0²)

### Addition

Only same-dimension terms combine:

```jsx
|a|ₙ + |b|ₙ = |a+b|ₙ
```

Terms in different dimensions remain separate — they represent different orders of magnitude.

**Example:** `|3|₁ + |5| + |2|₋₁ + |4|₁ = |7|₁ |5| |2|₋₁`

### Division

Division follows the inverse of multiplication:

```jsx
|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ
```

**Key cases:**

- `|6|₁ / |2|₋₁ = |3|₂` (dividing by zero → multiplying by infinity)
- `|6| / |2|₁ = |3|₋₁` (dividing by infinity → multiplying by zero)

**Note:** Division by multi-term composites: exact division returns finite result; non-exact kept as rational expression.

### Exponentiation

For single-term composites:

```jsx
(|a|ₙ)^k = |a^k|ₙₖ
```

**Example:** `(|2|₁)² = |4|₂` and `√(|4|₂) = |2|₁`

Multi-term exponentiation uses distribution (polynomial expansion).

---

## Comparison: Old vs New Notation

| Old Notation | New Notation | Meaning |
| --- | --- | --- |
| `⟨_; 5 | _⟩` | `|5|` | Plain rational 5 |
| `⟨0; 5 | 0⟩` | `|5|` | Same — phantom zeros don't exist |
| `⟨2(0); 5 | 1(0)⟩` | `|2|₁ |5| |1|₋₁` | 2∞ + 5 + 1 zero |

**The key difference:** In new notation, dimensions only exist if written. You can't accidentally have phantom `|0|₁` or `|0|₋₁` — they must be explicitly constructed.

---

## Tests

### Test Suite 1: Basic Operations

#### Test 1.1: Plain Rational × 0

**Input:** `|5| × 0`

**Expected:** Shift down 1 → `|5|₋₁`

**Process:**

- Only rational dimension exists
- `|5|` shifts to dimension -1
- **Result:** `|5|₋₁` ✓

#### Test 1.2: Plain Rational × ∞

**Input:** `|5| × ∞`

**Expected:** Shift up 1 → `|5|₁`

**Process:**

- Only rational dimension exists
- `|5|` shifts to dimension +1
- **Result:** `|5|₁` ✓

#### Test 1.3: Reversal (Collapse then Expand)

**Input:** `|5| × 0 × ∞`

**Process:**

- `|5| × 0 = |5|₋₁`
- `|5|₋₁ × ∞ = |5|₀ = |5|`
- **Result:** `|5|` ✓ **Matches original!**

#### Test 1.4: Reversal (Expand then Collapse)

**Input:** `|5| × ∞ × 0`

**Process:**

- `|5| × ∞ = |5|₁`
- `|5|₁ × 0 = |5|₀ = |5|`
- **Result:** `|5|` ✓ **Matches original!**

#### Test 1.5: Multi-component Collapse

**Input:** `|2|₁ |5| |1|₋₁ × 0`

**Expected:** All shift down 1

**Process:**

- `|2|₁` → `|2|₀` = `|2|`
- `|5|` → `|5|₋₁`
- `|1|₋₁` → `|1|₋₂`
- **Result:** `|2| |5|₋₁ |1|₋₂` ✓

#### Test 1.6: Multi-component Expansion

**Input:** `|2| |5|₋₁ |1|₋₂ × ∞`

**Expected:** All shift up 1

**Process:**

- `|2|` → `|2|₁`
- `|5|₋₁` → `|5|₀` = `|5|`
- `|1|₋₂` → `|1|₋₁`
- **Result:** `|2|₁ |5| |1|₋₁` ✓ **Reversal works!**

---

### Test Suite 2: Multi-Order Operations

#### Test 2.1: × 0² (Shift down 2)

**Input:** `|5| × 0²`

**Expected:** Shift down 2 → `|5|₋₂`

**Process:**

- `|5|` shifts from dim 0 to dim -2
- **Result:** `|5|₋₂` ✓

#### Test 2.2: × ∞² (Shift up 2)

**Input:** `|5| × ∞²`

**Expected:** Shift up 2 → `|5|₂`

**Process:**

- `|5|` shifts from dim 0 to dim +2
- **Result:** `|5|₂` ✓

#### Test 2.3: Reversal with Order 2

**Input:** `|5| × 0² × ∞²`

**Process:**

- `|5| × 0² = |5|₋₂`
- `|5|₋₂ × ∞² = |5|₀ = |5|`
- **Result:** `|5|` ✓ **Matches original!**

#### Test 2.4: Mixed Orders

**Input:** `|3|₂ |5| |2|₋₁ × 0`

**Process:**

- `|3|₂` → `|3|₁`
- `|5|` → `|5|₋₁`
- `|2|₋₁` → `|2|₋₂`
- **Result:** `|3|₁ |5|₋₁ |2|₋₂` ✓

#### Test 2.5: Order Mismatch Reversal

**Input:** `|5|₋₂ × ∞` (only shift up 1, not 2)

**Process:**

- `|5|₋₂` → `|5|₋₁`
- **Result:** `|5|₋₁` ✓ (partial reversal, as expected)

---

### Test Suite 3: Composite × Composite

#### Test 3.1: Rational × Rational

**Input:** `|5| × |3|`

**Process:**

- `|5|₀ × |3|₀ = |15|₀₊₀ = |15|`
- **Result:** `|15|` ✓

#### Test 3.2: Rational × Zero-dimension

**Input:** `|5| × |3|₋₁`

**Process:**

- `|5|₀ × |3|₋₁ = |15|₀₊₍₋₁₎ = |15|₋₁`
- **Result:** `|15|₋₁` ✓

#### Test 3.3: Zero × Infinity (Order Cancellation)

**Input:** `|2|₁ × |3|₋₁`

**Process:**

- `|2|₁ × |3|₋₁ = |6|₁₊₍₋₁₎ = |6|₀ = |6|`
- **Result:** `|6|` ✓ (∞ × 0 = rational)

#### Test 3.4: Full Composite × Full Composite

**Input:** `|2|₁ |3| × |1|₁ |4|`

**Process (distribute):**

- `|2|₁ × |1|₁ = |2|₂`
- `|2|₁ × |4|₀ = |8|₁`
- `|3|₀ × |1|₁ = |3|₁`
- `|3|₀ × |4|₀ = |12|₀`
- **Collect by dimension:**
    - Dim 2: `|2|₂`
    - Dim 1: `|8+3|₁ = |11|₁`
    - Dim 0: `|12|`
- **Result:** `|2|₂ |11|₁ |12|` ✓

#### Test 3.5: Composite × Composite with Zero Dimensions

**Input:** `|2|₁ |5| |1|₋₁ × |1| |1|₋₁`

**Process (distribute all 6 terms):**

- `|2|₁ × |1|₀ = |2|₁`
- `|2|₁ × |1|₋₁ = |2|₀ = |2|`
- `|5|₀ × |1|₀ = |5|₀`
- `|5|₀ × |1|₋₁ = |5|₋₁`
- `|1|₋₁ × |1|₀ = |1|₋₁`
- `|1|₋₁ × |1|₋₁ = |1|₋₂`
- **Collect by dimension:**
    - Dim 1: `|2|₁`
    - Dim 0: `|2+5|` = `|7|`
    - Dim -1: `|5+1|₋₁` = `|6|₋₁`
    - Dim -2: `|1|₋₂`
- **Result:** `|2|₁ |7| |6|₋₁ |1|₋₂` ✓

---

### Test Suite 4: Edge Cases

#### Test 4.1: The Original Problem Case

**Old problem:** `⟨0; 5 | 0⟩ × 0` created phantom `1` and `0²`

**New system:** Starting with plain `|5|`

- `|5| × 0 = |5|₋₁`
- **Result:** `|5|₋₁` ✓ **No phantom terms!**

#### Test 4.2: What if we explicitly have zero-valued dimensions?

**Input:** `|0|₁ |5| |0|₋₁ × 0`

**Process:**

- `|0|₁` → `|0|₀` = contributes 0 to rational
- `|5|` → `|5|₋₁`
- `|0|₋₁` → `|0|₋₂` = contributes 0 to dim -2
- **Result:** `|0| |5|₋₁ |0|₋₂`

**Ladder equivalence:** By `|0|ₙ = |1|ₙ₋₁`, we have `|0|₀ = |1|₋₁` and `|0|₋₂ = |1|₋₃`, so the full result is equivalently `|1|₋₁ |5|₋₁ |1|₋₃`. Zero coefficients are **not omitted** — they carry structural meaning via the Ladder.

✓ **Consistent!** Explicit zeros are preserved and interpretable.

#### Test 4.3: Zero as a Value (REVISED)

**Definition:** `0 = |0| = |1|₋₁` (two equivalent single-term forms)

**Test A:** `|5| × |0|` (multiply by value-zero)

**Process:** `|5|₀ × |0|₀ = |0|₀ = |0|`

**Result:** `|0|` ✓ (value becomes zero)

**Test B:** `|5| × |1|₋₁` (multiply by structural-zero)

**Process:** `|5|₀ × |1|₋₁ = |5|₋₁`

**Result:** `|5|₋₁` ✓ (shifts to zero dimension with coefficient preserved)

**Test C:** `|5| × 0` (using ×0 as shift operator)

**Result:** `|5|₋₁` ✓

**Key insight:** Both `|0|₀` and `|1|₋₁` are the same value (the Ladder of Absences: `|0|ₙ = |1|ₙ₋₁`). Both follow the same rule `|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ`:

- `|5|₀ × |0|₀ = |0|₀` — coefficients multiply (5×0=0), dimensions add (0+0=0)
- `|5|₀ × |1|₋₁ = |5|₋₁` — coefficients multiply (5×1=5), dimensions add (0+(-1)=-1)

The different-looking results `|0|₀` and `|5|₋₁` are themselves equivalent via the Ladder: `|0|₀ = |1|₋₁` and `|5×0|₀ = |5|₋₁`.

#### Test 4.4: Commutativity

**Test:** `|2|₁ × |3|₋₁` vs `|3|₋₁ × |2|₁`

- First: `|2|₁ × |3|₋₁ = |6|₀ = |6|`
- Second: `|3|₋₁ × |2|₁ = |6|₀ = |6|`
- ✓ **Commutative!**

#### Test 4.5: Associativity

**Test:** `(|2| × |3|₋₁) × |4|₁` vs `|2| × (|3|₋₁ × |4|₁)`

- First: `|6|₋₁ × |4|₁ = |24|₀ = |24|`
- Second: `|2| × |12|₀ = |24|₀ = |24|`
- ✓ **Associative!**

---

## Summary

| Test Category | Tests | Passed |
| --- | --- | --- |
| Basic Operations | 6 | 6 ✓ |
| Multi-Order | 5 | 5 ✓ |
| Composite × Composite | 5 | 5 ✓ |
| Edge Cases | 5 | 5 ✓ |
| **Total** | **21** | **21 ✓** |

**All tests pass.** The new notation system appears consistent and solves the structural absence problem.

---

## Test Suite 5: Composite × Composite — Trap Hunting

The basic tests pass, but let's deliberately look for hidden contradictions.

### Test 5.1: Full 3×3 Distribution

**Input:** `(|2|₁ |3| |1|₋₁) × (|1|₁ |2| |1|₋₁)`

**Process (9 terms):**

| Term A | × | Term B | = | Result |
| --- | --- | --- | --- | --- |
| `|2|₁` | × | `|1|₁` | = | `|2|₂` |
| `|2|₁` | × | `|2|` | = | `|4|₁` |
| `|2|₁` | × | `|1|₋₁` | = | `|2|` |
| `|3|` | × | `|1|₁` | = | `|3|₁` |
| `|3|` | × | `|2|` | = | `|6|` |
| `|3|` | × | `|1|₋₁` | = | `|3|₋₁` |
| `|1|₋₁` | × | `|1|₁` | = | `|1|` |
| `|1|₋₁` | × | `|2|` | = | `|2|₋₁` |
| `|1|₋₁` | × | `|1|₋₁` | = | `|1|₋₂` |

**Collect by dimension:**

- Dim 2: `|2|₂`
- Dim 1: `|4+3|₁` = `|7|₁`
- Dim 0: `|2+6+1|` = `|9|`
- Dim -1: `|3+2|₋₁` = `|5|₋₁`
- Dim -2: `|1|₋₂`

**Result:** `|2|₂ |7|₁ |9| |5|₋₁ |1|₋₂` ✓

---

### Test 5.2: Negative Coefficients

**Question:** What does `|-5|₋₁` mean? Is "negative 5 zeroes" valid?

**Input:** `|3| × |-2|₋₁`

**Process:**

- `|3|₀ × |-2|₋₁ = |-6|₋₁`

**Result:** `|-6|₋₁`

**Interpretation:** -6 zeroes. Mathematically: `-6 × 0 = 0` as a value.

**⚠️ POTENTIAL ISSUE:** Negative coefficients work algebraically, but what do they *mean*?

- In provenance terms: "I owe 6 zeroes" or "6 anti-zeroes"?
- Does `|6|₋₁ + |-6|₋₁ = |0|₋₁`? (zero zeroes = no zero dimension?)

**Status:** 🟡 Algebraically consistent, semantically unclear

---

### Test 5.3: Coefficient Cancellation Within a Dimension

**Input:** `(|3|₁ |2|) × (|-1|₁ |4|)`

**Process:**

- `|3|₁ × |-1|₁ = |-3|₂`
- `|3|₁ × |4|₀ = |12|₁`
- `|2|₀ × |-1|₁ = |-2|₁`
- `|2|₀ × |4|₀ = |8|₀`

**Collect:**

- Dim 2: `|-3|₂`
- Dim 1: `|12-2|₁ = |10|₁`
- Dim 0: `|8|`

**Result:** `|-3|₂ |10|₁ |8|` ✓

---

### Test 5.4: Complete Cancellation to Zero Coefficient

**Input:** `(|2|₁ |-1|) × (|1|₁ |2|)`

**Process:**

- `|2|₁ × |1|₁ = |2|₂`
- `|2|₁ × |2|₀ = |4|₁`
- `|-1|₀ × |1|₁ = |-1|₁`
- `|-1|₀ × |2|₀ = |-2|₀`

**Collect:**

- Dim 2: `|2|₂`
- Dim 1: `|4-1|₁ = |3|₁`
- Dim 0: `|-2|`

**Result:** `|2|₂ |3|₁ |-2|` ✓

**Now try for exact cancellation:**

**Input:** `(|2|₁ |-4|) × (|1|₁ |2|)`

**Process:**

- `|2|₁ × |1|₁ = |2|₂`
- `|2|₁ × |2|₀ = |4|₁`
- `|-4|₀ × |1|₁ = |-4|₁`
- `|-4|₀ × |2|₀ = |-8|₀`

**Collect:**

- Dim 2: `|2|₂`
- Dim 1: `|4-4|₁ = |0|₁` ← **Zero coefficient!**
- Dim 0: `|-8|`

**Result:** `|2|₂ |0|₁ |-8|`

**Result:** `|2|₂ |0|₁ |-8|`

**✅ Resolved:** By the Ladder of Absences, `|0|ₙ = |1|ₙ₋₁`. A zero coefficient at dimension n is equivalent to coefficient 1 at dimension n-1. So `|0|₁` = `|1|₀` = `|1|` — the dimension exists and carries meaning. Zero coefficients are always preserved.

**Status:** ✅ Resolved via Ladder of Absences

---

### Test 5.5: Multiplication by Zero Forms (REVISED)

**Definition:** `0 = |0| = |1|₋₁`

**Test A:** `(|2|₁ |3|) × |0|` (multiply by value-zero)

**Process:** By the Ladder, `|0|₀ = |1|₋₁`, so normalize first:

- `|2|₁ × |1|₋₁ = |2|₀ = |2|`
- `|3|₀ × |1|₋₁ = |3|₋₁`

**Result:** `|2| |3|₋₁` ✓

**Interpretation:** Identical to Test B — because `|0|` and `|1|₋₁` are the same value, they must produce the same result. The Ladder normalization is always applied before arithmetic.

**Test B:** `(|2|₁ |3|) × |1|₋₁` (multiply by structural-zero)

**Process:**

- `|2|₁ × |1|₋₁ = |2|₀ = |2|`
- `|3|₀ × |1|₋₁ = |3|₋₁`

**Result:** `|2| |3|₋₁` ✓

**Test C:** `(|2|₁ |3|) × 0` (using ×0 as shift operator)

- `|2|₁` → `|2|₀`
- `|3|₀` → `|3|₋₁`

**Result:** `|2| |3|₋₁` ✓

**Key finding:** `×|1|₋₁` and `×0` (shift operator) produce **identical results**.

Since `|0|₀ = |1|₋₁` (Ladder of Absences), `×|0|` also produces an equivalent result — the different-looking outputs are the same values expressed in different dimensional representations. All three follow the single rule `|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ`.

---

### Test 5.6: Division — Does the Rule Extend?

**Hypothesis:** If `|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ`, then `|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ`

**Test:** `|6|₁ / |2|₋₁`

**Expected:** `|6/2|₁₋₍₋₁₎ = |3|₂`

**Verification via inverse:**

- If `|3|₂ × |2|₋₁ = |6|₂₊₍₋₁₎ = |6|₁` ✓

**Result:** Division rule `|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ` appears consistent ✓

---

### Test 5.7: Division by Zero-Dimension Value

**Input:** `|6| / |2|₋₁`

**Process:** `|6/2|₀₋₍₋₁₎ = |3|₁`

**Interpretation:** Dividing by a "zero" promotes to infinity dimension.

**Verification via inverse:**

- `|3|₁ × |2|₋₁ = |6|₁₊₍₋₁₎ = |6|₀ = |6|` ✓

**This is interesting:** `/|n|₋₁` acts like `×∞` but scaled by 1/n.

---

### Test 5.8: Self-Multiplication (Squaring)

**Input:** `(|2|₁ |3|)²`

**Process:**

- `|2|₁ × |2|₁ = |4|₂`
- `|2|₁ × |3|₀ = |6|₁`
- `|3|₀ × |2|₁ = |6|₁`
- `|3|₀ × |3|₀ = |9|₀`

**Collect:**

- Dim 2: `|4|₂`
- Dim 1: `|6+6|₁ = |12|₁`
- Dim 0: `|9|`

**Result:** `|4|₂ |12|₁ |9|` ✓

**Sanity check:** This should equal `(2∞ + 3)²`

- In limit terms: `(2∞ + 3)² = 4∞² + 12∞ + 9` ✓ **Matches!**

---

### Test 5.9: Gaps in Dimensions

**Input:** `(|2|₂ |5|₋₂) × (|3|₁ |1|₋₁)`

**Process:**

- `|2|₂ × |3|₁ = |6|₃`
- `|2|₂ × |1|₋₁ = |2|₁`
- `|5|₋₂ × |3|₁ = |15|₋₁`
- `|5|₋₂ × |1|₋₁ = |5|₋₃`

**Collect:**

- Dim 3: `|6|₃`
- Dim 1: `|2|₁`
- Dim -1: `|15|₋₁`
- Dim -3: `|5|₋₃`

**Result:** `|6|₃ |2|₁ |15|₋₁ |5|₋₃` ✓

**Note:** Gaps are preserved. No phantom middle dimensions appear.

---

### Test 5.10: Power of Zero/Infinity Dimension Values

**Input:** `(|2|₋₁)²`

**Process:**

- `|2|₋₁ × |2|₋₁ = |4|₋₂`

**Result:** `|4|₋₂` ✓

**Interpretation:** `(2×0)² = 4×0²` — consistent with `0² = 0×0`

---

## Summary of Traps Found

| Issue | Status | Notes |
| --- | --- | --- |
| Negative coefficients | ✅ | Algebraically sound; semantics (negative infinitesimals) to be developed later |
| Zero coefficients `\|0\|ₙ` | ✅ | **Keep them.** If written or resulting from operations, respect and calculate with them |
| Division rule | ✅ | Extends naturally: `\|a\|ₘ / \|b\|ₙ = \|a/b\|ₘ₋ₙ` |
| Self-multiplication | ✅ | Works, matches polynomial expansion |
| Dimension gaps | ✅ | Preserved correctly, no phantom fill |

---

## Design Decisions (Resolved)

### Decision 1: Negative Coefficients

**Rule:** Negative coefficients are algebraically valid.

**Interpretation:** `|-5|₋₁` means an infinitely small but specified *negative* value. Full semantic development deferred.

**Example:** `|-6|₋₁` = -6 zeroes = a negative infinitesimal with magnitude 6.

### Decision 2: Zero Coefficients

**Rule:** Zero coefficients (`|0|ₙ`) are preserved if they result from operations on previously existing dimensions.

**Principle:** "If zero is written, we respect it."

**Implications:**

- `|4|₁ + |-4|₁ = |0|₁` — dimension 1 exists but has zero coefficient
- `|0|₁` ≠ (no dimension 1) — structural difference preserved
- A dimension with `|0|ₙ` still participates in operations

**Example:** `|0|₁ × |3|₋₁ = |3|₋₁` — by the Ladder, `|0|₁ = |1|₀`, so `|1|₀ × |3|₋₁ = |3|₋₁`. The Ladder normalization applies before coefficient multiplication.

---

**All traps resolved.** The system is now fully specified for multiplication and division.

---

## Test Suite 6: Addition of Composites

Addition should be simpler than multiplication — only same-dimension terms combine.

### Addition Rule

```
|a|ₙ + |b|ₙ = |a+b|ₙ
```

Terms in different dimensions remain separate (they represent different "orders of magnitude").

---

### Test 6.1: Same-Dimension Addition

**Input:** `|3|₁ + |2|₁`

**Process:** Same dimension (1), add coefficients

**Result:** `|5|₁` ✓

---

### Test 6.2: Different-Dimension Addition

**Input:** `|3|₁ + |2|₋₁`

**Process:** Different dimensions, cannot combine

**Result:** `|3|₁ |2|₋₁` ✓ (remains as two-term composite)

---

### Test 6.3: Addition with Rational

**Input:** `|5| + |3|₁`

**Process:** Dim 0 and dim 1, cannot combine

**Result:** `|3|₁ |5|` ✓ (written high-to-low by convention)

---

### Test 6.4: Full Composite + Full Composite

**Input:** `(|2|₁ |5| |1|₋₁) + (|3|₁ |2| |4|₋₁)`

**Process:** Combine matching dimensions:

- Dim 1: `|2+3|₁ = |5|₁`
- Dim 0: `|5+2| = |7|`
- Dim -1: `|1+4|₋₁ = |5|₋₁`

**Result:** `|5|₁ |7| |5|₋₁` ✓

---

### Test 6.5: Additive Identity

**Input:** `|5|₁ + |0|₁`

**Process:** `|5+0|₁ = |5|₁`

**Result:** `|5|₁` ✓

**Note:** Per Decision 2, if we started with explicit `|0|₁`, the dimension existed. But `|5|₁` already has dim 1, so result just has `|5|₁`.

---

### Test 6.6: Negative Coefficient Addition

**Input:** `|5|₁ + |-3|₁`

**Process:** `|5-3|₁ = |2|₁`

**Result:** `|2|₁` ✓

---

### Test 6.7: Cancellation to Zero Coefficient

**Input:** `|5|₁ + |-5|₁`

**Process:** `|5-5|₁ = |0|₁`

**Result:** `|0|₁` ✓

**Per Decision 2:** Dimension 1 existed in both operands, so `|0|₁` is preserved. The dimension exists but has zero coefficient.

---

### Test 6.8: Adding Zero Forms (REVISED)

**Definition:** `0 = |0|₀ = |1|₋₁` (same value, two representations — Ladder of Absences)

**Test A:** `|5| + |0|` (add zero as `|0|₀`)

**Process:** Dim 0: `|5+0| = |5|`

**Result:** `|5|` ✓

**Note:** Addition operates on the representation as written — `|0|₀` shares dim 0 with `|5|₀`, so coefficients combine (5+0=5).

**Test B:** `|5| + |1|₋₁` (add zero as `|1|₋₁`)

**Process:**

- Dim 0: `|5|` (only in first operand)
- Dim -1: `|1|₋₁` (only in second operand)

**Result:** `|5| |1|₋₁`

**Note:** `|1|₋₁` occupies dim -1, which doesn't overlap with dim 0, so both terms remain.

**Key finding:** Since `|0|₀ = |1|₋₁` (Ladder of Absences), these two additions are adding the **same value** — the different results (`|5|` vs `|5| |1|₋₁`) reflect the choice of dimensional representation, not different algebraic behavior. Both results are equivalent via the Ladder: `|5|₀` = `|5|₀ |1|₋₁` (the `|1|₋₁` is the zero that's always implicitly present at dim -1).

**Status:** ✅ Consistent with the Ladder of Absences

---

### Test 6.9: Commutativity

**Test:** `(|2|₁ |3|) + (|4|₁ |5|)` vs `(|4|₁ |5|) + (|2|₁ |3|)`

- First: `|6|₁ |8|`
- Second: `|6|₁ |8|`

✓ **Commutative!**

---

### Test 6.10: Associativity

**Test:** `((|2|₁ |3|) + (|1|₁ |4|)) + (|3|₁ |2|)` vs `(|2|₁ |3|) + ((|1|₁ |4|) + (|3|₁ |2|))`

**First path:**

- `(|2|₁ |3|) + (|1|₁ |4|) = |3|₁ |7|`
- `|3|₁ |7| + |3|₁ |2| = |6|₁ |9|`

**Second path:**

- `(|1|₁ |4|) + (|3|₁ |2|) = |4|₁ |6|`
- `|2|₁ |3| + |4|₁ |6| = |6|₁ |9|`

✓ **Associative!**

---

### Test 6.11: Distributivity (× over +)

**Test:** `|2|₋₁ × (|3|₁ + |4|)` vs `(|2|₋₁ × |3|₁) + (|2|₋₁ × |4|)`

**First path:**

- `|3|₁ + |4| = |3|₁ |4|`
- `|2|₋₁ × (|3|₁ |4|)` — distribute:
    - `|2|₋₁ × |3|₁ = |6|₀ = |6|`
    - `|2|₋₁ × |4|₀ = |8|₋₁`
- Result: `|6| |8|₋₁`

**Second path:**

- `|2|₋₁ × |3|₁ = |6|₀ = |6|`
- `|2|₋₁ × |4|₀ = |8|₋₁`
- Sum: `|6| + |8|₋₁ = |6| |8|₋₁`

✓ **Distributive!**

---

### Test 6.12: Mixed Dimension Sets

**Input:** `(|2|₂ |3|) + (|5|₁ |1|₋₁)`

**Process:** No overlapping dimensions

- Dim 2: `|2|₂` (first only)
- Dim 1: `|5|₁` (second only)
- Dim 0: `|3|` (first only)
- Dim -1: `|1|₋₁` (second only)

**Result:** `|2|₂ |5|₁ |3| |1|₋₁` ✓

**Note:** Union of dimension sets.

---

### Test 6.13: Partial Overlap

**Input:** `(|2|₂ |3|₁ |5|) + (|1|₁ |4| |2|₋₁)`

**Process:**

- Dim 2: `|2|₂` (first only)
- Dim 1: `|3+1|₁ = |4|₁` (both)
- Dim 0: `|5+4| = |9|` (both)
- Dim -1: `|2|₋₁` (second only)

**Result:** `|2|₂ |4|₁ |9| |2|₋₁` ✓

---

### Test 6.14: Subtraction (Addition of Negative)

**Input:** `(|5|₁ |3|) - (|2|₁ |1|)`

**Rewrite as:** `(|5|₁ |3|) + (|-2|₁ |-1|)`

**Process:**

- Dim 1: `|5-2|₁ = |3|₁`
- Dim 0: `|3-1| = |2|`

**Result:** `|3|₁ |2|` ✓

---

## Summary: Addition Tests

| Test | Result | Notes |
| --- | --- | --- |
| 6.1 Same-dimension | ✅ | Coefficients add |
| 6.2 Different-dimension | ✅ | Terms stay separate |
| 6.3 With rational | ✅ | No cross-dim mixing |
| 6.4 Full + Full | ✅ | Matching dims combine |
| 6.5 Additive identity | ✅ | `+|0|ₙ` preserves value |
| 6.6 Negative coefficients | ✅ | Works as expected |
| 6.7 Cancellation → |0|ₙ | ✅ | Zero coefficient preserved |
| 6.8 Adding value zero | ✅ | **Provenance preserved** (not identity!) |
| 6.9 Commutativity | ✅ | A + B = B + A |
| 6.10 Associativity | ✅ | (A+B)+C = A+(B+C) |
| 6.11 Distributivity | ✅ | A×(B+C) = A×B + A×C |
| 6.12 Mixed dimension sets | ✅ | Union of dimensions |
| 6.13 Partial overlap | ✅ | Combine where matching |
| 6.14 Subtraction | ✅ | Works via negative coefficients |

**All 14 addition tests pass.**

---

## Key Finding: Zero and the Ladder of Absences

`|0|₀ = |1|₋₁` — these are the **same value** in two equivalent representations (Ladder of Absences: `|0|ₙ = |1|ₙ₋₁`).

All operations follow the same rules regardless of representation:

- Multiplication: `|a|ₘ × |b|ₙ = |a×b|ₘ₊ₙ`
- Addition: `|a|ₙ + |b|ₙ = |a+b|ₙ` (same-dimension terms combine)
- Division: `|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ`

There is no "annihilation" and no special-cased additive identity among composites. The system has **one zero** with multiple dimensional representations, all connected by the Ladder.

---

## Test Suite 7: Deep Edge Cases

Now let's stress-test with indeterminate forms, identities, and pathological cases.

---

### Test 7.1: The Multiplicative Identity

**Question:** What is `1` in this system?

**Answer:** `|1|` — just 1 in the rational dimension.

**Test:** `|3|₁ |5| |2|₋₁ × |1|`

**Process:**

- `|3|₁ × |1|₀ = |3|₁`
- `|5|₀ × |1|₀ = |5|₀`
- `|2|₋₁ × |1|₀ = |2|₋₁`

**Result:** `|3|₁ |5| |2|₋₁` ✓ **Identity preserved!**

---

### Test 7.2: 0 × ∞ — The Classic Indeterminate (REVISED)

**In standard math:** `0 × ∞` is indeterminate.

**In our system:**

- `0 = |0| = |1|₋₁` (two forms)
- `∞ = |1|₁` (one infinity)

**Test A:** `|0| × |1|₁` (value-zero × infinity)

**Process:** `|0|₀ × |1|₁ = |0|₁`

**Result:** `|0|₁` (zero infinities)

**Value:** By the Ladder, `|0|₁ = |1|₀ = |1|` — so this also equals 1

**Test B:** `|1|₋₁ × |1|₁` (structural-zero × infinity)

**Process:** `|1|₋₁ × |1|₁ = |1|₀ = |1|`

**Result:** `|1|` ✓

**Value:** 1 — dimensions cancel!

**Key insight:** The "indeterminacy" of 0×∞ in standard math comes from not knowing *which* zero and *which* infinity.

- `|0| × |1|₁ = |0|₁` (value = 0)
- `|1|₋₁ × |1|₁ = |1|` (value = 1)
- `|2|₋₁ × |1|₁ = |2|` (value = 2)
- `|1|₋₁ × |3|₁ = |3|` (value = 3)

✓ **Always determinate** when zeros and infinities are specific!

---

### Test 7.3: ∞ × 0 (Reverse Order) (REVISED)

**Test A:** `|1|₁ × |0|`

**Process:** `|1|₁ × |0|₀ = |0|₁`

**Result:** `|0|₁` ✓ Same as 7.2A

**Test B:** `|1|₁ × |1|₋₁`

**Process:** `|1|₁ × |1|₋₁ = |1|₀ = |1|`

**Result:** `|1|` ✓ Same as 7.2B

✓ **Commutative!**

---

### Test 7.4: 0/0 — Reconsidered

**Initial error:** I tested the multi-term composite `(|0||1|₋₁) / (|0||1|₋₁)` which falls into the "division by multi-term" problem (Test 7.10).

**Correct approach:** Use the structural zero `|1|₋₁` (one zero, single term).

**Test:** `|1|₋₁ / |1|₋₁`

**Process:** Using division rule `|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ`:

- `|1|₋₁ / |1|₋₁ = |1/1|₋₁₋₍₋₁₎ = |1|₀ = |1|`

**Result:** `|1|` ✓

**Interpretation:** 0/0 = 1 when using structural zeroes of the same order!

This is **consistent with the fundamental rule**: anything divided by itself equals 1.

**More examples:**

- `|5|₋₁ / |5|₋₁ = |1|` ✓ (5 zeroes / 5 zeroes = 1)
- `|3|₋₁ / |1|₋₁ = |3|` ✓ (3 zeroes / 1 zero = 3)
- `|1|₋₂ / |1|₋₁ = |1|₋₁` ✓ (0² / 0 = 0)

**Key insight:** There are NO undefined values in this system. The apparent indeterminacy of 0/0 in standard math comes from unspecified limits. Our structural zeroes are *specific*, so division is always determinate.

**Status:** ✅ **0/0 = 1** (for same-order structural zeroes)

---

### Test 7.5: ∞/∞

**Test:** `|1|₁ / |1|₁`

**Process:** `|1/1|₁₋₁ = |1|₀ = |1|`

**Result:** `|1|` ✓

**Hmm:** Unlike standard math where ∞/∞ is indeterminate, here we get a clean answer.

**Why?** Because our "infinity" `|1|₁` is a *specific* infinity (1 first-order infinity), not a vague "goes to infinity." Dividing it by itself gives 1.

**Different infinities:**

- `|2|₁ / |1|₁ = |2|₀ = |2|` ✓
- `|1|₂ / |1|₁ = |1|₁` (∞²/∞ = ∞) ✓

---

### Test 7.6: ∞ - ∞

**Test:** `|1|₁ + |-1|₁`

**Process:** `|1-1|₁ = |0|₁`

**Result:** `|0|₁`

**Interpretation:** Zero infinities — the dimension exists but has coefficient 0.

Per Decision 2, we keep `|0|₁`. This is NOT the same as "no infinity dimension."

**Different infinities:**

- `|3|₁ + |-2|₁ = |1|₁` (determinate)
- `|2|₂ |3|₁ + |-3|₁ = |2|₂ |0|₁` (partial cancellation)

✓ **Always determinate** — no true "∞ - ∞" indeterminacy.

---

### Test 7.7: Chained Dimension Operations

**Test:** `|5| × 0 × ∞ × 0 × ∞ × 0`

**Process:**

- `|5| × 0 = |5|₋₁`
- `|5|₋₁ × ∞ = |5|₀ = |5|`
- `|5| × 0 = |5|₋₁`
- `|5|₋₁ × ∞ = |5|₀ = |5|`
- `|5| × 0 = |5|₋₁`

**Result:** `|5|₋₁`

**Verification:** 3 zeros, 2 infinities → net 1 zero → dim -1 ✓

---

### Test 7.8: Zero Coefficient in Zero Dimension

**Input:** `|0|₋₁`

**Interpretation:** Zero zeroes. The dimension exists but is empty.

**Test:** `|0|₋₁ × |3|₁`

**Process:** `|0×3|₋₁₊₁ = |0|₀ = |0|`

**Result:** `|0|`

✓ **Zero propagates correctly**

**Test:** `|5| |0|₋₁ × ∞`

**Process:**

- `|5|₀ × ∞ = |5|₁`
- `|0|₋₁ × ∞ = |0|₀ = |0|`

**Result:** `|5|₁ |0|`

Per Decision 2, keep `|0|` since it resulted from operation on existing `|0|₋₁`.

✓ **Consistent**

---

### Test 7.9: Very Deep Dimensions

**Test:** `|7|₋₁₀ × |3|₁₀`

**Process:** `|21|₋₁₀₊₁₀ = |21|₀ = |21|`

**Result:** `|21|` ✓

**Interpretation:** 10th-order zero times 10th-order infinity = rational. The orders cancel.

---

### Test 7.10: Division by Multi-Term Composite

**Input:** `|12| / (|2||1|₋₁)`

**Problem:** How do we divide by a sum?

In standard algebra: `a / (b + c) ≠ a/b + a/c`

**Approach 1:** Leave as unevaluated expression `|12| / (|2||1|₋₁)`

**Approach 2:** If `|2||1|₋₁ = 2 + 1(0) = 2` as a value, then `|12| / 2 = |6|`

But this loses the `|1|₋₁` provenance.

**Status:** ⚠️ **Division by multi-term composite is problematic**

We can define:

- `|a|ₘ / |b|ₙ = |a/b|ₘ₋ₙ` (single term by single term) ✓
- Division by multi-term composite: **undefined** or requires special handling

---

### Test 7.11: Square Root of Composite

**Test:** `√(|4|₂)` — square root of 4∞²

**If** exponentiation follows dimension rules:

- `(|a|ₙ)^k = |a^k|ₙₖ`

**Then:** `(|2|₁)² = |4|₂` ✓

**And:** `√(|4|₂) = (|4|₂)^(1/2) = |4^(1/2)|₂ₓ₍₁/₂₎ = |2|₁` ✓

**Test:** `√(|9| |6|₋₁ |1|₋₂)` — can we take square root of multi-term?

This would require `(|3| |1|₋₁)² = |9| |6|₋₁ |1|₋₂`

**Verify:**

- `|3|² = |9|`
- `|3| × |1|₋₁ = |3|₋₁` (twice) → `|6|₋₁`
- `|1|₋₁ × |1|₋₁ = |1|₋₂`

**Yes!** `(|3| |1|₋₁)² = |9| |6|₋₁ |1|₋₂` ✓

**So:** `√(|9| |6|₋₁ |1|₋₂) = |3| |1|₋₁`

✓ **Square roots work for perfect squares**

---

### Test 7.12: Non-Perfect Square Root

**Test:** `√(|5|₋₁)` — square root of 5 zeroes

**If:** `(|a|ₙ)^(1/2) = |a^(1/2)|ₙ/₂`

**Then:** `√(|5|₋₁) = |√5|₋₁/₂`

**Problem:** Dimension -1/2 is not an integer!

**Options:**

1. Allow fractional dimensions (extends the system)
2. Leave as unevaluated `√(|5|₋₁)`
3. Reject — only integer dimensions allowed

**Status:** ⚠️ **Fractional dimensions question** — design decision needed

---

### Test 7.13: Negative Base in Zero Dimension

**Test:** `|-3|₋₁ × |-2|₋₁`

**Process:** `|(-3)×(-2)|₋₂ = |6|₋₂`

**Result:** `|6|₋₂` ✓

**Interpretation:** Negative times negative in any dimension = positive. Standard sign rules apply.

---

### Test 7.14: What is |1|₋₁?

**Interpretation:** `|1|₋₁` = 1 zero = the "unit zero" = 0 (as a value)

By the Ladder of Absences, `|1|₋₁ = |0|₀` — same value, different dimensional representations.

**Test:** `|1|₋₁ × ∞`

**Process:** `|1|₋₁₊₁ = |1|₀ = |1|`

**Result:** `|1|` ✓

**This confirms:** `|1|₋₁ × ∞ = 1`, which aligns with `0 × ∞ = 1` when both are "unit" sized.

**Compare:** `|5|₋₁ × ∞ = |5|₀ = |5|` — the coefficient survives!

---

## Summary: Edge Cases

| Test | Result | Notes |
| --- | --- | --- |
| 7.1 Multiplicative identity | ✅ | `\|1\|` works as identity |
| 7.2 0 × ∞ | ✅ | Determinate: `\|0\| × ∞ = 0`, `\|1\|₋₁ × ∞ = 1` |
| 7.3 ∞ × 0 | ✅ | Commutative with 7.2 |
| 7.4 0/0 | ✅ | **0/0 = 1** (structural zeroes, same order) |
| 7.5 ∞/∞ | ✅ | Determinate: `\|1\|` (specific infinities) |
| 7.6 ∞ - ∞ | ✅ | Determinate: `\|0\|₁` (zero coefficient) |
| 7.7 Chained ops | ✅ | Net dimension shift works |
| 7.8 Zero coeff in zero dim | ✅ | Propagates correctly |
| 7.9 Deep dimensions | ✅ | Orders cancel as expected |
| 7.10 Divide by multi-term | ⚠️ | Undefined for now |
| 7.11 √ of composite | ✅ | Works for perfect squares |
| 7.12 Non-perfect √ | ⚠️ | Fractional dimensions? Design decision |
| 7.13 Negative × negative | ✅ | Standard sign rules |
| 7.14 Unit zero `\|1\|₋₁` | ✅ | `\|1\|₋₁ × ∞ = \|1\|` |

**12 passed, 2 flagged:**

1. Division by multi-term composite → see exploration below
2. Fractional dimensions → **EXPERIMENTAL** (may be valid, needs exploration)

---

## Exploration: Division by Multi-Term Composite

### The Polynomial Analogy

Our composites behave like polynomials where the "variable" is dimension shift:

- `|a|₁` is like `a·x` (where x = ∞)
- `|b|` is like `b·x⁰ = b`
- `|c|₋₁` is like `c·x⁻¹` (where x⁻¹ = 0)

So `|2|₁ |5| |3|₋₁` corresponds to the polynomial `2x + 5 + 3x⁻¹` or equivalently `2x + 5 + 3/x`.

### Polynomial Long Division

**Example:** Divide `|6|₁ |11| |6|₋₁` by `|2|₁ |3|`

In polynomial form: `(6x + 11 + 6/x) ÷ (2x + 3)`

**Step 1:** Divide leading terms

- `|6|₁ ÷ |2|₁ = |3|₀ = |3|`
- Quotient so far: `|3|`

**Step 2:** Multiply back and subtract

- `|3| × (|2|₁ |3|) = |6|₁ |9|`
- Subtract from dividend: `(|6|₁ |11| |6|₋₁) - (|6|₁ |9|)`
- = `|0|₁ |2| |6|₋₁`
- = `|2| |6|₋₁` (dropping zero coefficient)

**Step 3:** Divide leading terms of remainder

- `|2| ÷ |2|₁ = |1|₋₁`
- Quotient so far: `|3| |1|₋₁`

**Step 4:** Multiply back and subtract

- `|1|₋₁ × (|2|₁ |3|) = |2|₀ |3|₋₁ = |2| |3|₋₁`
- Subtract from remainder: `(|2| |6|₋₁) - (|2| |3|₋₁)`
- = `|0| |3|₋₁`
- = `|3|₋₁`

**Step 5:** Divide leading terms of remainder

- `|3|₋₁ ÷ |2|₁ = |3/2|₋₂`
- Quotient: `|3| |1|₋₁ |3/2|₋₂`

**Step 6:** This continues infinitely...

- We get an infinite series: `|3| |1|₋₁ |3/2|₋₂ |9/4|₋₃ ...`

⚠️ **Problem:** Unlike polynomial division over integers, this doesn't terminate!

### Why Polynomial Division Can Be Problematic

In standard polynomial division, we require:

- The divisor's leading term divides into everything
- We eventually reach degree 0 or get a "proper" remainder

But in our system:

- Dimensions extend infinitely in both directions (∞³, ∞², ∞, R, 0, 0², 0³...)
- Division can push terms into ever-deeper dimensions
- No natural "bottom" to stop at

### When Division DOES Terminate

**Case 1: Exact factorization**

If the dividend is an exact multiple of the divisor, division terminates.

**Test:** `(|4|₂ |12|₁ |9|) ÷ (|2|₁ |3|)`

**Check:** Is `|4|₂ |12|₁ |9|` equal to `(|2|₁ |3|)²`?

- `(|2|₁ |3|)² = |4|₂ |12|₁ |9|` ✓ (we verified this in Test 5.8)

**So:** `(|4|₂ |12|₁ |9|) ÷ (|2|₁ |3|) = |2|₁ |3|` ✓

**Case 2: Single-term divisor**

Division by single terms always works:

- `(|6|₁ |10| |4|₋₁) ÷ |2|₋₁`
- = `|6|₁/|2|₋₁ + |10|/|2|₋₁ + |4|₋₁/|2|₋₁`
- = `|3|₂ + |5|₁ + |2|`
- = `|3|₂ |5|₁ |2|` ✓

### Proposed Rule for Multi-Term Division

**Option A: Exact Division Only**

Allow division only when the result is exact (no remainder, finite terms).

This is like saying: `a ÷ b` is defined iff `∃c` such that `b × c = a`.

To check: try polynomial division; if it terminates with zero remainder, the result is valid.

**Option B: Allow Infinite Series**

Allow division to produce infinite series representations.

`|12| ÷ (|2||1|₋₁)` = infinite series `|6| |-3|₋₁ |3/2|₋₂ ...`

This is like writing `12 / (2 + ε) = 6 - 3ε + 3ε²/2 - ...` (Taylor expansion)

**Option C: Rational Expressions**

Keep unevaluated as `|12| / (|2||1|₋₁)` — a "rational composite."

Like how we write `(x+1)/(x-1)` without expanding.

### Decision: Options A + C

**Rule:** Division by multi-term composites is always defined:

1. **If exact** (terminates with zero remainder) → return finite composite
2. **If non-exact** → keep as **rational expression** `A / B`

Rational expressions can later be expanded to infinite series if needed (like `1/(1-x) = 1 + x + x² + ...`), but the primary representation preserves structure.

**Rationale:** The system has no undefined values. Division always has a result — either a finite composite or a rational expression.

**Future development:** 

- Formalize rational expression notation and simplification rules
- Define when/how to expand to infinite series

---

### Test: Exact Division

**Test 7.15:** `(|9| |6|₋₁ |1|₋₂) ÷ (|3| |1|₋₁)`

**Check if exact:** Does `(|3| |1|₋₁)² = |9| |6|₋₁ |1|₋₂`?

- `|3|² = |9|`
- `|3| × |1|₋₁ = |3|₋₁` (×2) → `|6|₋₁`
- `|1|₋₁ × |1|₋₁ = |1|₋₂`
- Result: `|9| |6|₋₁ |1|₋₂` ✓

**Answer:** `(|9| |6|₋₁ |1|₋₂) ÷ (|3| |1|₋₁) = |3| |1|₋₁` ✓

---

**Status:** Multi-term division works for **exact factors**. Non-exact division produces rational expressions or infinite series (design choice).

---

## Exploratory: Division as Dimension Shifting

*To be developed further.*

**Core intuition:** Division by a zero-dimension value is equivalent to multiplication by an infinity-dimension value.

**Example:** `|6|₁ / |2|₋₁`

- 6 infinities ÷ 2 zeroes
- Algebraically: `6(1/0) / 2(0) = (6/2) × (1/0) × (1/0) = 3/0² = 3∞²`
- Using our rule: `|6/2|₁₋₍₋₁₎ = |3|₂` ✓

**General principle:**

- `/|n|₋₁` acts like `×|1/n|₁` (dividing by zero → multiplying by infinity)
- `/|n|₁` acts like `×|1/n|₋₁` (dividing by infinity → multiplying by zero)

This reinforces that all operations are fundamentally about **dimension shifting** along the scale:

```
... ∞³ — ∞² — ∞ — R — 0 — 0² — 0³ ...
```

**Open questions:**

- How does this help with multi-term division?
- Can we express rational expressions as dimension-shift operators?