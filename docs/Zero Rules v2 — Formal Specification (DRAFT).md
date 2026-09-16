# Zero Rules v2 — Formal Specification (DRAFT)

**Status:** reviewed and **implemented**. See §7 for the decisions taken and
§8 for what changed in the code.
Every suite in the repository passes, and all six algebraic laws hold (§3).
Supersedes *Zero Rules — Formal Specification (DRAFT)*, whose §8 recorded
consequences that these rules do not produce. Every numeric claim below was
measured against a working prototype.

Notation: `|a|_d`, shortened to `a_d`, is coefficient `a` at dimension `d`.
`|a|_d = a·h^(-d)`, so dimension −1 is the infinitesimal and +1 the infinity.

---

## 0. Notation, and the distinction it must carry

**Write a composite with angle brackets and commas, never with `+`.**

```
<0_0, 6_-1, 1_-2>          one number, three terms
<0_0> + <6_-1>             an operation between two numbers
```

This is not cosmetic. Using `+` for both the terms of a number and the
addition of two numbers is a genuine ambiguity, and it is the single thing
that makes these rules hard to reason about. `0_3 + -4_1` has two readings:

```
<0_3, -4_1>          one number whose dimension-3 term is empty   ->  stays as written
<0_3> + <-4_1>       two numbers being added                      ->  <-4_1, 1_2>
```

Both are legitimate; they are different expressions. With `+` overloaded they
look identical, and every apparent inconsistency in these rules traces back to
reading one as the other.

**The distinction that follows.** A composite entering an operation is an
**operand**. The pieces a composite is written from are **terms**. R1 speaks
only of operands. Terms are notation; there is nothing for a rule to act on.

```
<0_0>                     an operand — a number that is a zero
<0_0, 6_-1, 1_-2>         one operand whose first term happens to be zero
```

---

## 1. The rules

**R1 — a zero operand converts.**
A composite that is **wholly zero** (every expressed coefficient is 0), used as
an operand of any operation, converts: `0_d -> 1_(d-1)`.
If it carries several zeros, convert the **lowest dimension only**. Once that
one has converted the composite holds a nonzero, so by R2 every remaining zero
is inert — and, by R2, retained.

```
0_3 + 0_-1 + 0_-3
  convert the lowest      0_-3 -> 1_-4
  a nonzero now exists    0_3 + 0_-1 + 1_-4
  R2: the rest are inert, and kept
                          =  0_3 + 0_-1 + 1_-4
```

Exactly one conversion happens. The other zeros are not discarded — they record
that those dimensions cancelled too.

Lowest-first is forced by the object being a zero: converting highest-first
would yield `1_2`, an infinity, which a zero is not.

**R2 — a zero term is inert and is kept.** *(a consequence of R1, not a
separate rule)*

R1 acts on operands. A zero that is a **term** of a composite is not an
operand — it is part of how that number is written (§0). So there is nothing
for R1 to do, and the term is retained, contributing **0** to every operation
exactly as a zero coefficient does in ordinary polynomial arithmetic. It never
merges into another coefficient and never promotes.

```
<0_0>                  an operand  ->  R1 applies   ->  <1_-1>
<0_0, 6_-1, 1_-2>      one operand ->  R1 does not apply to its terms
   ÷ <1_-1>            all terms shift together     ->  <0_1, 6_0, 1_-1>
```

Nothing here is stipulated. Once `+` stops doing two jobs, R1 alone decides
every case.

**R3 — `× 1` and `/ 1` are identities.** They return the operand untouched.

**R4 — addition and subtraction never shift dimensions.** Coefficients add per
dimension. `a − a` leaves `0_d`: a zero at that dimension, per R2.

**R6 — there is no additive identity.**
Adding zero is not a no-op. `0_d` is a number, so by R1 it converts and
contributes `1_(d-1)`:

```
R(5) + 0_0   ->  5_0 + 1_-1        not 5_0
0_0 + 0_0    ->  2_-1
```

This is the `1 − 1 ≠ 0` thesis applied consistently — if subtracting a value
from itself leaves an infinitesimal, adding a zero must deposit one too.

`∅` (nothing) is **not** an identity element, because it is not a number.
`a + ∅ = a` holds because adding nothing is a no-op, and nothing is not zero
(§0, Principle 2). The system has a multiplicative identity (R3) and no
additive one; the asymmetry is deliberate.

**Consequence for accumulators.** A summation that has not yet added a term
holds *nothing*, not zero. Seeding one with `Composite({0: 0.0})` asserts that
a zero exists there, and that assertion costs one `h`:

```python
total = Composite({})        # correct — nothing yet
total = Composite({0: 0.0})  # wrong — asserts a zero, adds an h
```

The same applies to `x + 0` with a Python literal anywhere in library or user
code: the scalar is lifted to `0_0` and contributes an infinitesimal.

**R5 — products retain the dimensions they construct.** The dimensions of a
product are the Minkowski sum of the operands' dimension sets, and
zero-valued results at those dimensions are kept (R2). Note the current
backend violates this: `_merge_cluster_outputs` prunes with
`abs(v) > zero_tol` and `zero_tol = 0.0`. The mask is required — `_cluster_terms`
densifies gaps before convolving, so simply not pruning would invent
dimensions nobody built.

---

## 2. Worked examples — all measured

| expression | result |
|---|---|
| `0_0 × 0_0` | `1_-2` |
| `0_0 × 0_0 × 0_0` | `1_-3` |
| `0_0 × 5_0` | `5_-1` |
| `0_2 × 0_1` | `1_1` |
| `0_3 × 2_0` | `2_2` |
| `5_0 × 0_3` | `5_2` |
| `0_0 / 0_0` | `1_0` |
| `0_3 + 5_3` | `5_3 + 1_2` (an addition — R1 applies to `0_3`) |
| `(0_3 + 0_-1 + 0_-3) × 2_0` | `0_3 + 0_-1 + 2_-4` |
| `x − x` for `x = 3_3+5_-1+2_-3`, then `× 2_0` | `0_3 + 0_-1 + 2_-4` |
| `(0_2 + 3_0)(0_1 + 5_0)` | `6_1 + 18_0` |

The last one term by term, as written with `+` so each zero is an operand:

```
(0*0)_3 + (0*5)_2 + (3*0)_1 + 15_0
  =  1_1  +   5_1  +   3_0  + 15_0
  =  6_1 + 18_0                        six infinities and eighteen reals
```

---

## 2b. The derivation that most needs the notation

```
x = 3 + h        <3_0, 1_-1>
x²               <9_0, 6_-1, 1_-2>
x² − 9           <0_0, 6_-1, 1_-2>     the subtraction is finished here;
                                        the 0_0 is a TERM of this number
÷ h              <0_1, 6_0, 1_-1>      one operand, all terms shift together
st                6
```

Written with `+`, the third line reads `0_0 + 6_-1 + 1_-2`, and it is natural
to see a pending addition and resolve the `0_0` into the `6_-1`, giving `7_-1`
and a derivative of 7. There is no pending addition. The subtraction completed
when the number was formed.

This is the whole of the 6-versus-7 question, and it is settled by §0 rather
than by any choice between rules.

---

## 3. Algebraic laws — measured

Operands built as sums of single terms with zeros included, N = 1500–2000,
compared as numbers:

| law | failures |
|---|---|
| `a × 1 = a` | **0** |
| `(a+b)·c = a·c + b·c` | **0** |
| `a × b = b × a` | **0** |
| `(a·b)·c = a·(b·c)` | **0** |
| `a + b = b + a` | **0** |
| `(a+b)+c = a+(b+c)` | **0** |

This is the first rule set in the investigation for which all of them hold.

---

## 4. Calculus — measured

| | result |
|---|---|
| `d/dx x²` at 3, by difference quotient | **6** |
| `d/dx x³` at 2, `d/dx x¹⁰` at 2 | 12, 5120 |
| `derivative(x⁴, 3)`, `taylor(cos, 0, 4)` | 108, correct |
| `lim (x²−4)/(x−2)` at 2 | **4** |
| `lim (x³−1)/(x−1)` at 1 | 3 |
| `lim (eˣ−1)/x` at 0, `lim sin(x)/x` at 0 | 1, 1 |
| `lim (1−cos x)/x²` at 0 | 0.5 |
| `∫x² dx` over [0,1], `∫sin x dx` over [0,π] | 1/3, 2 |
| `ZERO/ZERO`, `5·ZERO/ZERO`, `7·ZERO²/ZERO²` | 1, 5, 7 |

---

## 5. Provenance

R2 keeps the zero rather than discarding it, so the cancellation record
survives arithmetic:

```
(3+h)² − 9      ->  0_0 + 6_-1 + 1_-2
÷ h             ->  0_1 + 6_0 + 1_-1        st = 6, and the 0_1 records the 9−9
(0_2+3_0) × 2_0 ->  0_2 + 6_0               the product keeps the dimension it built
```

Read the markers with `[d for d, v in c.c.items() if v == 0.0]`.

**What is preserved**

- **Theorem 1.** `5_0 × 0_0 = 5_-1` — multiplying by zero shifts the value to
  another dimension instead of destroying it, and `5_0 × 0_0 / 0_0 = 5_0`
  recovers it exactly.
- **Order of a zero.** `0_0 × 0_0 = 1_-2`, `0_0³ = 1_-3`. A zero carries its
  dimensional order through every operation; `0_2` and `0_0` are different
  numbers with different futures.
- **The derivative tower.** `(3+h)² = 9_0 + 6_-1 + 1_-2` — value and all
  derivatives, from one evaluation.
- **The cancellation record**, at its own dimension, carried through
  subsequent arithmetic.

**A correction to the paper's framing**

The README states that the residue of `1 − 1` "contains the derivative of every
operation that produced it." That is not what happens:

```
(3+h)²      =  9_0 + 6_-1 + 1_-2      the 6 is already present
minus 9     =  0_0 + 6_-1 + 1_-2      the subtraction only adds the 0_0
```

The derivative is in the tower **before** any subtraction. The residue records
only that a subtraction occurred. This matters: earlier rule sets let the
residue merge into dimension −1, where the derivative lives, and that is
precisely what produced `d/dx x² = 7`. Under R2 the residue keeps its own
dimension and the derivative is untouched.

The accurate claim is: **`×0` is information-preserving and reversible**, and
**derivatives arise from dimensional convolution** — both untouched here.

---

## 5b. Verification

All fifteen headline results, measured together, 0 failures:

| | result | |
|---|---|---|
| `d/dx x²` @3, `x³` @2, `x¹⁰` @2 (difference quotient) | 6, 12, 5120 | ✓ |
| `derivative(x⁴, 3)`, `taylor(cos,0,4)[2]` | 108, −0.5 | ✓ |
| `lim (x²−4)/(x−2)` @2, `(x³−1)/(x−1)` @1 | 4, 3 | ✓ |
| `lim (eˣ−1)/x`, `sin(x)/x`, `(1−cos x)/x²` @0 | 1, 1, 0.5 | ✓ |
| `lim ln(1+x)/x`, `(1+x)^(1/x)` @0 | 1, e | ✓ |
| `∫x²` over [0,1] | 1/3 | ✓ |
| `ZERO/ZERO`, `5·ZERO/ZERO` | 1, 5 | ✓ |
| `(0_2+3_0)(0_1+5_0)` | `<6_1, 18_0>` | ✓ |

---

## 6. Suite results

Measured against the implementation, not a prototype.

| suite | before | after |
|---|---|---|
| test_standalone | 167/167 | **167/167** |
| test_limits | 104/105 | **105/105** |
| test_stress | 20/20 | **20/20** |
| test_stress_hard_edge | 20/20 | **20/20** |
| test_integration_comprehensive | 54/54 | **54/54** |
| test_composite_vector | 25/25 | **25/25** |
| test_multivar_extended | *crashed at test 17* | **50/50** |
| test_multivar_disprove | 68 pass, 1 skip | 68 pass, 1 skip |
| turing_completeness | clean | clean |

Six algebraic laws over 2000 randomised cases: **0 failures**.
Performance unchanged (692 ms on a 200-evaluation mixed workload).

Three library defects surfaced while implementing these rules. None was caused
by them; each was pre-existing and hidden.

**6.1 — `ln` seeded its accumulator with a zero.** `Composite({0: math.log(a)})`
is `{0: 0.0}` at `a = 1` -- a zero, which R1 converts on first use, adding a
spurious `h` and doubling the series. This is R6: a summation that has not yet
added a term holds *nothing*, not zero. `atan`, `asin` and `acos` seeded the
same way. Every accumulator in the library should be audited against R6.

**6.2 — transcendentals built series with nothing to expand in.** `sqrt`, `sin`,
`cos`, `ln`, `atan`, `asin` now return their value directly when the argument
carries no infinitesimal part. Otherwise `x - R(a)` is a genuine zero, and R1
turns it into `|1|_-1`, manufacturing an infinitesimal that is not there.

**6.3 — FFT convolution silently zeroed products of large composites.** Above
128 combined terms, `convolve` used an FFT, which mixes every input coefficient
into every output bin -- so one overflow destroyed all of them, including
dimension 0, whose direct product is finite. Squaring `sin(1/x)` (coefficients
to 1e165) returned all zeros instead of `sin(100)^2`. It now falls back to
direct convolution when the FFT result is not finite. Reachable from any deep
Taylor tower; the oscillatory limit test was simply the only thing that noticed.

---

## 7. Decisions

**7.1 — what counts as `0`: exact `0.0` only.** *Decided; implemented.*
No tolerance decides zero-ness anywhere in the arithmetic. Term existence
(`exp`'s infinitesimal filter, `deconvolve`'s leading-term and dividend
selection in both backends, `_mc_deconvolve`, the `MC` transcendental guards),
long-division residue cleaning, and the reciprocal guards in `_reciprocal` and
`_mc_reciprocal` are all exact comparisons. A tolerance is unsafe in both
directions: `exp(-100) = 3.7e-44` read as zero becomes a unit infinitesimal,
and a legitimately tiny standard part was being refused inversion.

Five `1e-100` guards remain in `_detect_singularity` and the improper-integral
machinery. They test whether a *function value* is large enough to divide by,
not whether a coefficient is a zero -- a different question, left alone.

**7.2 — no discarding in serialization.** *Decided; implemented.*
`to_dict`/`from_dict`, `to_json`/`from_json`, `to_bytes`/`from_bytes` and
`to_array`/`from_array` all round-trip zero coefficients. `from_array` was
filtering `if v != 0`; that is gone.

**7.3 — cancellation above dimension 0.** *Resolved; no change needed.*
`3_3 - 3_3` gives `<0_3>` -- the dimension was constructed by both operands, so
it exists and holds zero, exactly as at any other dimension. When that zero is
later used as an operand R1 converts it, and the result is the mirror image of
the zero case rather than an anomaly:

```
ZERO - ZERO  ->  <0_-1>  = ZERO**2     one step further from dimension 0
INF  - INF   ->  <0_1>   -> <1_0>      one step nearer dimension 0
```

Cancellation always moves one step toward the less-structured end. Two equal
infinitesimals cancel to a deeper infinitesimal; two equal infinities cancel to
a smaller one.

**7.4 — marker accumulation.** *Left as is.* Zeros are never discarded, so long
chains accumulate them. Measured cost is ~1.2x with no growth in
active-dimension count. Revisit if a workload shows otherwise.

**7.5 — scope.** *Left for now.* The rules apply to `Composite`. `MC` and the
complex composites in `composite_extended.py` have their own arithmetic. `MC`
did receive the division fix below, but not R1--R6.

---

## 8. What changed

**`composite/backends/sparse_dense_backend.py`**
- `convolve` masks its output to the Minkowski sum of the operands' dimension
  sets, retaining zero-valued coefficients (R5). Replaces the `zero_tol`
  pruning, which discarded every zero a product built. The mask is required:
  `_cluster_terms` densifies gaps before convolving, so simply not pruning
  would invent dimensions nobody constructed.
- FFT overflow fallback (§6.3).
- Leading-term and dividend selection exact (7.1).

**`composite/backends/dict_backend.py`** -- the same zero retention. Its double
loop already produces the Minkowski sum, so the fix was to stop pruning.
Selection predicates exact. The two backends had drifted apart; they now agree
term for term, which matters because `test_standalone` cross-checks them.

**`composite/backends/base_backend.py`** -- the contract documented on `add`,
`convolve` and `deconvolve`, so a third backend implementer has it in writing.

**`composite/backends/config.py`** -- `set_backend` rebuilds `ZERO`/`INF`/`h`,
which were pinned to the import-time backend. `use_dict()` previously raised
`AttributeError` from anything touching them.

**`composite/composite_lib.py`**
- `_is_wholly_zero`, `_is_unit`, `_r1`, `_operands` added. `_operands` also
  aligns two composites onto one backend, so a reference captured before a
  backend switch still works.
- `__add__`, `__sub__`, `__mul__`, `__truediv__` and reflected forms rewritten
  to R1--R6. `__rtruediv__` must not short-circuit on a unit: its operands are
  reversed, and doing so turned `1/x` into `x`.
- `_expressed_zero` removed entirely -- attribute, `__slots__`, the `__sub__`
  cancellation branch and its self-contradicting `0.0 = 0**2` comment. R1 and
  R2 derive all of it.
- `_compare` reads both sides through R1, so `<0_-1> == ZERO**2`.
- `_has_infinitesimal_part` guards (§6.2); `ln` accumulator (§6.1);
  serialization (7.2); exact predicates (7.1).
- Dead code removed: `_poly_divide` (unreferenced -- division goes through the
  backend) and `_is_all_zero` (left over from `_expressed_zero`).

**`composite/composite_multivar.py`** -- `_mc_order_key` and `_mc_deconvolve`
added. `_mc_reciprocal` expands `1/B` around `B`'s real part, so it cannot run
when that part is zero, which is exactly a limit like `(x^2+y^2)/(x^2+y^2)` at
the origin. Long division under a monomial order handles it, mirroring the
scalar backend. `test_multivar_extended.py` now runs to completion for the
first time.

**`tests/test_standalone.py`** -- four subtraction-canon assertions compared
`.c` dicts, which asserts a representation rather than a value. They now
compare numbers, with a comment explaining why `ZERO - ZERO` is `<0_-1>` and
why that is `ZERO**2`.

---

## 9. Test plan

- New file `tests/test_zero_rules.py`:
  - the nine identities of §2
  - the six algebraic laws of §3, fuzzed over operands **built by addition** of
    single terms with zeros included, ≥1500 cases each
  - the provenance assertions of §5 — `((3+h)²−9)/h` must be
    `0_1 + 6_0 + 1_-1`, not `6_0 + 1_-1`
- Regression guard for 7.3: `∫₀^∞ x·e⁻ˣ dx` must terminate and return 1.
- Round-trip guard for 7.4: a composite carrying zero coefficients must survive
  `to_json` / `from_json` unchanged.
- Guards for §6.1 and §6.2: `ln(1+h)` must have no dimension-0 term, and
  `1/x` with a Python literal must equal `R(1)/x`.
- No xfail markers are needed — every suite passes in full.
