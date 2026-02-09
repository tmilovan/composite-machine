# Turing Completeness — Evidence and Open Questions

**Toni Milovan**

Independent Researcher, Pula, Croatia

<aside>
⚠️

**Status: Seeking Review**

This page presents exploration (evidence?) of idea that composite arithmetic (as implemented in `composite_lib.py`) constitutes a Turing-complete mathematical system. We publish these tests and arguments openly, inviting the community to **reinforce or refute** this claim. If our reasoning contains errors, we want to find them.

</aside>

---

## The Claim

Composite arithmetic — a ℤ-graded sparse coefficient algebra based on Laurent polynomials with provenance-preserving semantics — appears to satisfy the requirements for Turing completeness.

If correct, this would place it among a very small set of mathematical or number systems known to be Turing complete:

| System | Type | Year |
| --- | --- | --- |
| μ-Recursive Functions | Functions on ℕ | 1936 |
| Lambda Calculus | Formal system on terms | 1936 |
| SKI Combinator Calculus | Combinatory algebra | 1930s |
| FRACTRAN | Arithmetic on rationals | 1987 |
| Register Machines (Minsky) | Arithmetic on ℕ | 1961 |
| Generalized Collatz functions | Functions on ℕ | 2000s |
| **Composite Arithmetic** | **Graded ring (Laurent polynomial)** | **2026** |

Standard arithmetic on ℝ, ℤ, or ℚ is **not** Turing complete. Neither are hyperreals, wheel theory, p-adics, quaternions, or surreals. The relevant comparison is FRACTRAN (Conway, 1987), which uses prime factorization as "dimensions" and fraction multiplication as state transformation — structurally similar to our use of dimensional coefficients and `read_dim`/`write_dim`.

**What would make this distinctive:** unlike every other system on this list, composite arithmetic simultaneously functions as a calculus machine (derivatives, limits, integrals via coefficient reads). No other TC mathematical system has this dual interpretation.

---

## The Argument

A computational model requires three primitives for Turing completeness:

### 1. Unbounded Memory

Composite numbers are ℤ-indexed sparse coefficient maps with no bound on the number of active dimensions. This provides:

- **Random access** — `read_dim(n)` and `write_dim(n, v)` at any integer dimension, O(1)
- **Sparse storage** — only occupied dimensions consume space, O(k) for k entries
- **Unbounded addressing** — dimensions extend infinitely in both directions

This satisfies the TM tape requirement.

### 2. Conditional Branching

**Our key argument:** coefficient extraction (`read_dim`) *is* conditional branching.

A transition function δ(state, symbol) → (new_state, new_symbol, direction) can be encoded as three Composite numbers where `dimension = state × ALPHABET_SIZE + symbol` and the coefficient at that dimension encodes the output:

```python
# 2-state busy beaver encoded as Composite numbers
T_state     = Composite({0: 1, 1: 1, 2: 0, 3: 2})  # new state per key
T_symbol    = Composite({0: 1, 1: 1, 2: 1, 3: 1})   # new symbol per key
T_direction = Composite({0: 1, 1: -1, 2: -1, 3: 0}) # direction per key

# The "if/else" is just coefficient extraction:
new_state = T_state.read_dim(key)  # dispatches based on (state, symbol)
```

No Python dictionary lookup. No if/else. The Composite number holds all possible transitions, and `read_dim` selects the correct one algebraically.

### 3. Iteration

Repeated application of a fixed composite step function constitutes iteration. The `×ZERO` dimensional shift provides a natural "tick" — each multiplication shifts the entire state down one dimension, encoding one computational step.

The execution loop (analogous to beta-reduction in lambda calculus, or the clock cycle in hardware) applies this step repeatedly. This is the execution engine, not part of the computational model itself.

### 4. Composition

Composite operations chain naturally: the output of one operation is a Composite that can be input to the next. Function composition `f(g(x))` works natively through the Taylor series propagation.

---

## The Evidence: Test Suite

The following self-contained script tests the claim. It requires no external dependencies beyond Python 3.6+.

**Run:** `python test_turing_completeness.py`

```python
"""
test_turing_completeness.py
============================
Evidence for Turing completeness of Composite arithmetic.

Tests:
  1. Composite tape correctly runs 2-state busy beaver (6 steps, 4 ones)
  2. Composite tape correctly runs 3-state busy beaver (14 steps, 6 ones)
  3. Multi-symbol alphabet works (values 1-5 at distinct positions)
  4. Composite arithmetic (×, +, ×ZERO, ÷ZERO) operates on the tape
  5. Tape is simultaneously a Laurent polynomial
  6. Transition function encoded as Composite numbers (no dict/if-else)

We publish these tests seeking external validation. If the tests pass
but the reasoning is flawed, we want to know.

Author: Toni Milovan
License: AGPL
"""

import sys

# =====================================================================
# COMPOSITE NUMBER (minimal self-contained implementation)
# =====================================================================

class Composite:
    """ℤ-graded sparse coefficient algebra (Laurent polynomial ring)."""

    def __init__(self, coefficients=None):
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            self.c = {0: coefficients} if coefficients != 0 else {}
        else:
            self.c = {k: v for k, v in coefficients.items() if abs(v) > 1e-12}

    @classmethod
    def real(cls, value):
        return cls({0: value})

    @classmethod
    def zero(cls):
        """Structural zero: |1|₋₁ (infinitesimal)"""
        return cls({-1: 1})

    @classmethod
    def infinity(cls):
        """Structural infinity: |1|₁"""
        return cls({1: 1})

    def st(self):
        """Standard part: coefficient at dimension 0."""
        return self.c.get(0, 0)

    def read_dim(self, dim):
        """Read coefficient at a specific dimension."""
        return self.c.get(dim, 0)

    def write_dim(self, dim, value):
        """Write a value at a specific dimension. Returns NEW Composite."""
        new_c = dict(self.c)
        if value == 0:
            new_c.pop(dim, None)
        else:
            new_c[dim] = value
        return Composite(new_c)

    def dims(self):
        return sorted(self.c.keys())

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return Composite(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Composite({k: v * other for k, v in self.c.items()})
        result = {}
        for d1, c1 in self.c.items():
            for d2, c2 in other.c.items():
                dim = d1 + d2
                result[dim] = result.get(dim, 0) + c1 * c2
        return Composite(result)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Composite({k: v / other for k, v in self.c.items()})
        if len(other.c) == 1:
            div_dim, div_coeff = list(other.c.items())[0]
            return Composite({d - div_dim: c / div_coeff for d, c in self.c.items()})
        raise NotImplementedError("Multi-term division")

    def __repr__(self):
        if not self.c:
            return "|empty|"
        sub = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
        def fmt(n):
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            return "\u208b" + ''.join(sub[int(d)] for d in str(-n))
        terms = sorted(self.c.items())
        return " + ".join(f"|{int(c) if c == int(c) else c:.6g}|{fmt(d)}" for d, c in terms)

ZERO = Composite.zero()
INF = Composite.infinity()

# =====================================================================
# TEST 1: 2-State Busy Beaver (dict-based transitions)
# =====================================================================

def test_1_busy_beaver_2state():
    """Tape as Composite. Transitions as Python dict (baseline)."""
    print("TEST 1: 2-State Busy Beaver (Composite tape, dict transitions)")
    bb2 = {
        ('A', 0): ('B', 1, +1),
        ('A', 1): ('B', 1, -1),
        ('B', 0): ('A', 1, -1),
        ('B', 1): ('HALT', 1, 0),
    }
    tape, state, head, steps = Composite({}), 'A', 0, 0
    while state != 'HALT' and steps < 100:
        symbol = int(tape.read_dim(head))
        new_state, new_symbol, direction = bb2[(state, symbol)]
        tape = tape.write_dim(head, new_symbol)
        head += direction
        state = new_state
        steps += 1
    assert state == 'HALT', f"Did not halt"
    assert steps == 6, f"Expected 6 steps, got {steps}"
    assert len(tape.c) == 4, f"Expected 4 ones, got {len(tape.c)}"
    for d in tape.c:
        assert tape.c[d] == 1, f"Cell {d} = {tape.c[d]}, expected 1"
    print(f"  ✅ Halted in {steps} steps, tape = {tape}")
    return True

# =====================================================================
# TEST 2: 3-State Busy Beaver
# =====================================================================

def test_2_busy_beaver_3state():
    """3-state busy beaver: 14 steps, 6 ones."""
    print("TEST 2: 3-State Busy Beaver")
    bb3 = {
        ('A', 0): ('B', 1, +1), ('A', 1): ('HALT', 1, +1),
        ('B', 0): ('C', 0, +1), ('B', 1): ('B', 1, +1),
        ('C', 0): ('C', 1, -1), ('C', 1): ('A', 1, -1),
    }
    tape, state, head, steps = Composite({}), 'A', 0, 0
    while state != 'HALT' and steps < 100:
        symbol = int(tape.read_dim(head))
        new_state, new_symbol, direction = bb3[(state, symbol)]
        tape = tape.write_dim(head, new_symbol)
        head += direction
        state = new_state
        steps += 1
    assert state == 'HALT' and steps == 14 and len(tape.c) == 6
    print(f"  ✅ Halted in {steps} steps, {len(tape.c)} ones on tape")
    return True

# =====================================================================
# TEST 3: Multi-Symbol Alphabet
# =====================================================================

def test_3_multi_symbol():
    """Counter writing values 1-5 at positions 0-4."""
    print("TEST 3: Multi-Symbol Counter")
    counter = {
        ('W1', 0): ('W2', 1, +1), ('W2', 0): ('W3', 2, +1),
        ('W3', 0): ('W4', 3, +1), ('W4', 0): ('W5', 4, +1),
        ('W5', 0): ('HALT', 5, 0),
    }
    tape, state, head, steps = Composite({}), 'W1', 0, 0
    while state != 'HALT' and steps < 100:
        symbol = int(tape.read_dim(head))
        new_state, new_symbol, direction = counter[(state, symbol)]
        tape = tape.write_dim(head, new_symbol)
        head += direction
        state = new_state
        steps += 1
    for i in range(5):
        assert int(tape.read_dim(i)) == i + 1
    print(f"  ✅ Tape = {tape}")
    return True

# =====================================================================
# TEST 4: Composite Arithmetic on Tape
# =====================================================================

def test_4_arithmetic_on_tape():
    """Tape supports ×, +, ×ZERO, ÷ZERO as algebraic operations."""
    print("TEST 4: Composite Arithmetic on Tape")
    tape = Composite({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})

    # Scale
    doubled = tape * 2
    assert int(doubled.read_dim(0)) == 2 and int(doubled.read_dim(4)) == 10

    # ×ZERO shifts dimensions down
    shifted = tape * ZERO
    assert shifted.read_dim(-1) == 1 and shifted.read_dim(3) == 5

    # ÷ZERO recovers original
    recovered = shifted / ZERO
    for d in tape.c:
        assert abs(tape.read_dim(d) - recovered.read_dim(d)) < 1e-10

    # Addition merges tapes
    tape2 = Composite({0: 10, 1: 20})
    combined = tape + tape2
    assert int(combined.read_dim(0)) == 11 and int(combined.read_dim(1)) == 22

    print("  ✅ ×2, ×ZERO, ÷ZERO, + all work on tape")
    return True

# =====================================================================
# TEST 5: Tape as Laurent Polynomial
# =====================================================================

def test_5_polynomial():
    """Tape is simultaneously a Laurent polynomial. Multiplication = convolution."""
    print("TEST 5: Tape as Polynomial")
    tape = Composite({-2: 1, -1: 1, 0: 1, 1: 1})  # Busy beaver result
    factor = Composite({0: 1, 1: 1})  # 1 + x
    product = tape * factor
    # Polynomial multiplication should expand dimensions
    assert len(product.c) > len(tape.c)
    assert product.read_dim(-2) == 1  # lowest term unchanged
    assert product.read_dim(2) == 1   # new highest term
    print(f"  ✅ tape × (1+x) = {product}")
    return True

# =====================================================================
# TEST 6: Composite-Native Control Flow (THE KEY TEST)
# =====================================================================

def test_6_composite_native_control_flow():
    """
    Transition function encoded as Composite numbers.
    Branching = coefficient extraction. No Python dict. No if/else.

    This is the test that matters for the TC claim.
    If branching can be done via read_dim on Composite-encoded
    transition tables, then control flow is native to the algebra.
    """
    print("TEST 6: Composite-Native Control Flow")
    print("  Encoding: key = state × ALPHABET_SIZE + symbol")
    print("  Branching: T_state.read_dim(key), T_symbol.read_dim(key), ...")
    print()

    HALT_CODE = 2
    ALPHABET_SIZE = 2

    # Transition function AS Composite numbers
    # (A=0, B=1, HALT=2) × (blank=0, marked=1) → key ∈ {0,1,2,3}
    T_state     = Composite({0: 1, 1: 1, 2: 0, 3: HALT_CODE})
    T_symbol    = Composite({0: 1, 1: 1, 2: 1, 3: 1})
    T_direction = Composite({0: 1, 1: -1, 2: -1, 3: 0})

    def composite_step(tape, state, head):
        """One TM step. Every operation is a composite primitive."""
        symbol = int(tape.read_dim(head))               # coefficient extraction
        key = state * ALPHABET_SIZE + symbol             # integer arithmetic
        new_state = int(T_state.read_dim(key))           # BRANCHING via read_dim
        new_symbol = int(T_symbol.read_dim(key))         # BRANCHING via read_dim
        direction = int(T_direction.read_dim(key))       # BRANCHING via read_dim
        tape = tape.write_dim(head, new_symbol)          # coefficient mutation
        head = head + direction                          # integer arithmetic
        return tape, new_state, head

    # Execute
    tape, state, head, steps = Composite({}), 0, 0, 0
    names = {0: 'A', 1: 'B', 2: 'HALT'}

    while state != HALT_CODE and steps < 100:
        old_state, old_sym = state, int(tape.read_dim(head))
        key = old_state * ALPHABET_SIZE + old_sym
        tape, state, head = composite_step(tape, state, head)
        steps += 1
        print(f"    Step {steps}: {names[old_state]},read={old_sym}"
              f" → key={key} → read_dim({key})"
              f" → {names[state]},write={int(T_symbol.read_dim(key))}"
              f",move={int(T_direction.read_dim(key)):+d}")

    # Verify: identical result to Test 1
    assert state == HALT_CODE, f"Did not halt"
    assert steps == 6, f"Expected 6 steps, got {steps}"
    assert len(tape.c) == 4, f"Expected 4 ones, got {len(tape.c)}"
    for d in tape.c:
        assert tape.c[d] == 1, f"Cell {d} = {tape.c[d]}, expected 1"

    print()
    print("  ✅ IDENTICAL to Test 1 — no dict, no if/else")
    print("  ✅ Branching = coefficient extraction (read_dim)")
    print("  ✅ Transition function = Composite numbers")
    return True

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPOSITE ARITHMETIC: TURING COMPLETENESS EVIDENCE")
    print("=" * 60)
    print()
    print("These tests seek to demonstrate that composite arithmetic")
    print("satisfies the requirements for Turing completeness.")
    print("We invite review, critique, and counterexamples.")
    print()

    tests = [
        test_1_busy_beaver_2state,
        test_2_busy_beaver_3state,
        test_3_multi_symbol,
        test_4_arithmetic_on_tape,
        test_5_polynomial,
        test_6_composite_native_control_flow,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
        print()

    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    print()
    if passed == len(tests):
        print("All tests pass. The evidence is consistent with the")
        print("claim that composite arithmetic is Turing complete.")
        print()
        print("This does NOT constitute a formal proof. We seek:")
        print("  • Peer review of the argument structure")
        print("  • Identification of gaps in the reasoning")
        print("  • Counterexamples that would refute the claim")
        print("  • Comparison with existing TC proofs (esp. FRACTRAN)")
    else:
        print("Some tests failed. The claim requires further work.")

    sys.exit(0 if passed == len(tests) else 1)
```

---

## How to Read the Tests

| Test | What It Shows | Why It Matters |
| --- | --- | --- |
| **1–2** | Busy beaver TMs run correctly on a Composite tape | Composite numbers work as TM memory |
| **3** | Multi-symbol alphabet (values 1–5) | Not limited to binary |
| **4** | ×, +, ×ZERO, ÷ZERO operate on the tape | Tape is a real algebraic object, not just a dict |
| **5** | Tape is a Laurent polynomial | Same structure encodes calculus (dual interpretation) |
| **6** | **Transition lookup via coefficient extraction** | **Control flow is native to the algebra** |

Test 6 is the critical one. Tests 1–5 show the tape works. Test 6 shows the *branching* works — without Python dicts or if/else.

---

## The Argument We're Making

### Memory

Composite numbers provide unbounded, sparse, ℤ-indexed random-access storage. `read_dim(n)` and `write_dim(n, v)` are O(1) operations on this storage. This maps directly to a TM tape where dimension = cell position and coefficient = symbol.

### Branching

A transition function δ(state, symbol) → (new_state, new_symbol, direction) is encoded as Composite numbers where each dimension corresponds to a (state, symbol) pair. Coefficient extraction (`read_dim(key)`) selects the correct transition — this IS conditional branching, expressed algebraically.

### Iteration

Repeated application of a fixed `composite_step()` function constitutes iteration. The `×ZERO` dimensional shift provides a natural step operator. The execution loop is the engine (like beta-reduction for lambda calculus), not part of the model.

### Composition

Composite operations chain: the output of one is input to the next. Function composition `f(g(x))` propagates through Taylor series automatically.

---

## What We're NOT Claiming

- ❌ That this is a formal proof (it's evidence + argument, seeking review)
- ❌ That TC gives faster algorithms (same asymptotic complexity as standard TMs)
- ❌ That the halting problem is solved (it remains undecidable, as expected)
- ❌ That this replaces existing computation models

## What We ARE Claiming

- ✅ Composite arithmetic has the primitives needed for universal computation
- ✅ The same algebraic structure that does calculus can also do computation
- ✅ This dual interpretation (calculus + computation) appears to be unique among TC systems
- ✅ The tests pass and are reproducible

---

## Open Questions

We actively seek answers to these:

1. **Is coefficient extraction sufficient for branching?** We argue `read_dim(key)` on a Composite-encoded transition table is algebraic branching. Is there a formal objection to this?
2. **How does this compare to FRACTRAN?** Conway's system uses prime factorization as "dimensions" and multiplication as state transformation. Our system uses explicit dimensional coefficients. Are these structurally equivalent? Is one a special case of the other?
3. **Can the execution loop be eliminated?** Currently, a Python `while` loop drives the computation. Can this be expressed purely in composite operations (e.g., via a fixed-point construction)?
4. **What is the formal algebraic classification?** We describe the system as a commutative monoid under addition (no additive identity) paired with standard ring multiplication on ℤ-graded sparse coefficients. Is there a better characterization?
5. **Are there counterexamples?** Is there a computable function that composite arithmetic *cannot* express with its current primitives?

---

## Related Pages

- Provenance-Preserving Arithmetic — foundational theory and algebraic structure
- composite_lib.py — the library implementation
- Standalone Test Suite — 100+ tests validating the core arithmetic

---

## How to Contribute

If you find an error in the reasoning, a gap in the tests, or a counterexample:

1. Open an issue on the repository
2. Reference the specific test or argument section
3. Provide a concrete counterexample if possible

We value honest critique over agreement.
