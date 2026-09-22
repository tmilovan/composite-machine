# Zero Rules — Formal Specification (DRAFT)

**Status:** awaiting review. **Not implemented.**
Derived from the working session of 2026-08-19. Every numeric claim in
§7–§11 was measured against a working prototype, not reasoned about.

**Structure.** §1–§7 are the rules. §8 is the measured consequence of applying
them. §9–§11 are the open items and the plan.

---

## 0. Purpose

This document fixes the semantics of a **zero coefficient at a dimension** —
what `|0|_d` means, when it shifts dimension, and what it does under each
arithmetic operation. It supersedes the informal rationale currently written
in the `Composite.__sub__` source comment (see §6.1).

Notation: `|a|_d` is coefficient `a` at dimension `d`; the shorthand `a_d`
is used in worked examples. Recall `|a|_d = a·h^(-d)`, so dimension `-1` is
the infinitesimal and dimension `+1` is the infinity.

---

## 1. Representation

A composite is a sparse map from integer dimensions to real coefficients.

**1.1 — Nothing is not zero.**
A dimension that is not present **does not exist**. This is *nothing* (`∅`).

**1.2 — Zero exists.**
A dimension present with coefficient `0` is **the zero at that dimension**,
written `|0|_d`. It exists, and it is retained in the representation.

**1.3 — There is exactly one kind of zero.**
No provenance, tag, order field, or flag is carried. Any coefficient equal
to `0` is the zero, regardless of how it arose — written by hand, computed
(`ln(1)`, `cos(π/2)`), or produced by cancellation.

> This is the decision that drives everything in §8. See §9.1.

---

## 2. R1 — The zero identity

```
|0|_d  ≡  |1|_(d-1)
```

A zero at dimension `d` is one unit at dimension `d-1`.

**2.1 — Latency.** The zero is stored as `|0|_d` and is *read as* `|1|_(d-1)`
**only at the moment of a multiplication or a division**. It is not rewritten
eagerly, and the identity is never applied repeatedly or to a fixpoint.

**2.2 — Only `×` and `÷` shift dimensions.** No other operation moves a
coefficient between dimensions.

---

## 3. R2 — Addition and subtraction

**3.1** Coefficients add per dimension. Dimensions never shift.

**3.2** A zero contributes nothing and does not disturb other dimensions:

```
|0|_3 + |5|_3  =  |5|_3
|0|_0 + |3|_0  =  |3|_0
```

**3.3 — Cancellation yields a zero, not nothing.** For `a ≠ 0`:

```
|a|_d − |a|_d  =  |0|_d
```

The dimension remains expressed. It does **not** become `∅`.

**3.4** The result's dimension set is the union of the operands'.

---

## 4. R3 — Multiplication

**4.1** At the moment of multiplication, every zero coefficient in either
operand is read per R1: `|0|_d → |1|_(d-1)`. Applied once, term-wise, to
each operand independently.

**4.2** Then ordinary convolution: dimensions add, coefficients multiply.

**4.3 — Information preservation (Theorem 1).** Multiplying by a zero does
not annihilate the other operand; it shifts it.

```
|0|_0 × |5|_0  =  |1|_-1 × |5|_0  =  |5|_-1
|5|_0 × |0|_3  =  |5|_0 × |1|_2   =  |5|_2
|0|_3 × |2|_0  =  |1|_2 × |2|_0   =  |2|_2
```

**4.4 — Products of zeros.**

```
|0|_0 × |0|_0        =  |1|_-1 × |1|_-1            =  |1|_-2
|0|_0 × |0|_0 × |0|_0 =  |1|_-1 × |1|_-1 × |1|_-1  =  |1|_-3
|0|_2 × |0|_1        =  |1|_1 × |1|_0              =  |1|_1
```

**4.5 — Distributivity holds.** Because R1 is applied term-wise, regrouping
cannot change the result:

```
(a+b)(c+d)  =  ac + ad + bc + bd
```

**4.6 — Constructed dimensions.** The dimensions a product constructs are
exactly the **Minkowski sum** of the operands' dimension sets, computed
*after* the R1 reading:

```
dims(A×B) = { da + db : da ∈ dims(A'), db ∈ dims(B') }
```

Zero-valued results at those dimensions are **retained** (per 1.2). No
dimension outside that set is created.

> This corrects a current backend defect: `_merge_cluster_outputs` prunes
> with `np.abs(vals) > zero_tol` (`zero_tol=0.0`), discarding every zero a
> product constructs. Note the fix cannot simply stop pruning — `_cluster_terms`
> densifies gaps before convolving, so the output must be masked to the
> Minkowski sum or phantom dimensions appear.

---

## 5. R4 — Division

Identical to R3: zeros are read per R1 at the point of division, then
dimensions subtract and coefficients divide.

```
|0|_0 / |0|_0  =  |1|_-1 / |1|_-1  =  |1|_0
```

**5.1 — Implementation note.** `Composite.__truediv__` currently has a
single-term-divisor fast path that bypasses the backend entirely. R1 must be
applied there too, or division and multiplication will disagree.

---

## 6. R5 — Derived results

These are **consequences** of R1–R4, not additional rules.

**6.1 — Subtraction of equal zeros.**

```
0**a  =  |1|_-a                       (canonical form)
|1|_-a − |1|_-a  =  |0|_-a            (R2.3, cancellation of equal values)
|0|_-a           =  |1|_(-a-1)        (R1)
                 =  0**(a+1)
```

So `0**a − 0**a = 0**(a+1)`, uniformly, for every `a`:

| | result |
|---|---|
| `0**1 − 0**1` | `0**2` = `1_-2` |
| `0**2 − 0**2` | `0**3` = `1_-3` |
| `0**3 − 0**3` | `0**4` = `1_-4` |

This matches the library's current behaviour. It does **not** match the
rationale in the `__sub__` source comment, which reads:

```
#   R(0)-R(0) = ZERO-ZERO = 0·0 = 0². Multiplication with zero → shift.
#   ZERO²-ZERO² = 0·(0²) = 0³. Shift via multiplication.
```

That justification is incoherent (line 1 multiplies both operands, line 2
multiplies by one) and, taken seriously, would give `0**2 − 0**2 = 0**4`.
**Action: rewrite that comment to the R2.3 + R1 derivation above.** The
`+1` shift is a consequence, not a special case, so the bespoke cancellation
branch in `__sub__` may be removable — to be confirmed during implementation.

---

## 7. Worked examples (all measured)

| expression | result |
|---|---|
| `0_3 + 5_3` | `5_3` |
| `0_0 + 3_0` | `3_0` |
| `0_0 × 0_0` | `1_-2` |
| `0_0 × 5_0` | `5_-1` |
| `0_2 × 0_1` | `1_1` |
| `0_0 × 0_0 × 0_0` | `1_-3` |
| `5_0 × 0_3` | `5_2` |
| `0_3 × 2_0` | `2_2` |
| `(0_2 + 3_0) × (0_1 + 5_0)` | `6_1 + 18_0` |

The last one term by term:

```
(0*0)_3 + (0*5)_2 + (3*0)_1 + 15_0
  = 1_1  +   5_1  +   3_0   + 15_0
  = 6_1 + 18_0                        "six infinities and eighteen reals"
```

Randomised checks on the prototype: **400/400** regroupings agree
(distributivity), **600/600** associativity and commutativity checks agree.

---

## 8. Consequences of the rules — measured

R1 fires on every `×` and `÷` without exception (§2.1–2.2). A zero produced by
cancellation is the same object as a written zero (§1.3), so `9 − 9 = |0|_0`,
and that zero converts like any other. The following are **measured outputs of
a prototype**, not predictions.

### 8.1 Difference quotients gain `+1`

| | classical | this spec |
|---|---|---|
| `d/dx x²` at 3 | 6 | **7** |
| `d/dx x³` at 2 | 12 | **13** |
| `d/dx x⁴` at 2 | 32 | **33** |
| `d/dx x⁵` at 1 | 5 | **6** |
| `d/dx 3x+2` | 3 | **4** |
| `d/dx eˣ` at 0 | 1 | **2** |
| `d/dx cos x` at 0 | 0 | **1** |

### 8.2 Indeterminate limits become `(a₁+1)/(b₁+1)`

| | classical | this spec |
|---|---|---|
| `lim (x²−4)/(x−2)` at 2 | 4 | **2.5** |
| `lim (x³−1)/(x−1)` at 1 | 3 | **2** |
| `lim (eˣ−1)/x` at 0 | 1 | **2** |
| `lim ln(1+x)/x` at 0 | 1 | **3** |
| `lim (1+x)^(1/x)` at 0 | e ≈ 2.718 | **20.086** (e³) |
| `lim (1+2x)^(1/x)` at 0 | e² ≈ 7.389 | **54.598** (e⁴) |

### 8.3 Divergence from real arithmetic

At `x = 2 + h`, the spec gives `x² − 4 = 5h + h²`. The real function gives
`4h + h²` at every `h` tested (1e-1 … 1e-6). The gap is exactly one `h` per
cancellation. **The system is no longer a conservative extension of ℝ on
expressions containing a cancelling subtraction.**

### 8.4 What is unaffected

- **All eight paper theorems: 100%.** T1 Information Preservation, T2
  Zero-Infinity Duality, T3 Provenance Non-Uniqueness, T4 Reversibility,
  T5 Coefficient Cancellation, T6 Identity Elements, T7 Fractional Orders,
  T8 Total Ordering.
- Subtraction Rules 11/11 · Zero Division 7/7 · Multi-Term Division 7/7 ·
  Integration 4/4 · Algebraic Properties 7/7 · Edge Cases 10/10 ·
  Multivariate 6/6.
- **Coefficient-read derivatives are unchanged**: `derivative(x⁴, at=3) = 108`,
  `all_derivatives(exp, 0)`, `taylor_coefficients(cos, 0)` all correct. These
  seed `a+h`, evaluate once, and read dimension `−n`; they never subtract.

### 8.5 Suite impact

| suite | current | this spec |
|---|---|---|
| test_standalone | 167/167 | **155/167** |
| test_stress | 20/20 | **15/20** |
| test_stress_hard_edge | 20/20 | **11/20** |
| test_integration_comprehensive | 54/54 | **53/54** |

Failing categories are exactly two: *Calculus: Derivatives* (2/7) and
*Calculus: Limits* (2/6), plus *Transcendental Functions* (5/7).

---

## 9. Open decisions for review

**9.1 — Is §8 accepted?**
§8 is what the rules produce. Difference quotients read `+1` and indeterminate
limits read `(a₁+1)/(b₁+1)`. Coefficient-read derivatives are unaffected
(§8.4), so the machine's native route is intact; what shifts is the classical
difference-quotient and limit constructions layered on top.

If accepted, the affected assertions in `test_standalone.py` and
`test_limits.py` are rewritten to the §8 values, and §8.2–§8.3 need an explicit
position in the paper — §8.3 in particular, since the algebra ceases to agree
with ℝ on any expression containing a cancelling subtraction.

**9.2 — What counts as `0`?**
Exact `0.0` only, or a tolerance? A tolerance is unsafe: `exp(-100) ≈ 3.7e-44`
read as a zero becomes a unit infinitesimal, turning a `1e-44` integrand tail
into `50` and hanging `∫₀^∞ x·e⁻ˣ dx` indefinitely. **Recommendation: exact
`0.0` only.**

**9.3 — Serialization.** `to_dict` / `to_bytes` / `to_json` / `from_array`
must round-trip expressed zeros, or a save/load cycle silently drops
dimensions the computation constructed. `from_array` currently filters
`if v != 0`.

**9.4 — Scope.** Does this apply to `MC` (multivariable, tuple dimensions)
and the complex composites in `composite_extended.py`, or to `Composite` only?
Both have their own arithmetic and would need the same treatment separately.

---

## 10. Implementation plan

1. **Backend `convolve`** — mask the output to the Minkowski sum of the
   operands' dimension sets, retaining zero-valued results (§4.6). Replaces the
   `zero_tol` pruning in `_merge_cluster_outputs`. Note the mask is required:
   `_cluster_terms` densifies gaps before convolving, so simply not pruning
   would create phantom dimensions.

2. **Backend `convolve` / `deconvolve` — R1** — apply R1 to both operands of
   every `×` and `÷`: each `|0|_d` term is read as `|1|_(d-1)`, once, term-wise,
   before the convolution or division runs (§4.1, §5).

3. **`Composite.__truediv__`** — the single-term-divisor fast path bypasses the
   backend entirely and must be brought in line, or division and multiplication
   will disagree (§5.1).

4. **`Composite.__add__` / `__sub__`** — retain expressed zeros, never shift.
   Evaluate whether the bespoke cancellation branch in `__sub__` can be deleted
   now that §6 derives its behaviour.

5. **Serialization** — preserve expressed zeros (§9.3).

6. **`__sub__` source comment** — replace with the §6 derivation.

A working prototype backend is in the session scratchpad and can be lifted
directly.

---

## 11. Test plan

- New file `tests/test_zero_rules.py`:
  - the nine identities of §7
  - distributivity fuzz — 400 random regroupings of `(Σaᵢ)(Σbⱼ)` vs `Σᵢⱼ aᵢbⱼ`
  - associativity and commutativity fuzz — 600 checks
- Regression guard for §9.2: `∫₀^∞ x·e⁻ˣ dx` must terminate and return 1.
- Round-trip guard for §9.3: a composite containing expressed zeros must
  survive `to_json` / `from_json` unchanged.

- Rewrite the affected assertions in `test_standalone.py` (Calculus:
  Derivatives, Calculus: Limits, Transcendental Functions) and `test_limits.py`
  to the §8.1 / §8.2 values, once §9.1 is settled.
