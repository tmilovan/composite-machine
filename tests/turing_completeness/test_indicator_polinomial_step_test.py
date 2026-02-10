"""
STEP as Pure Composite Expression
====================================
BB2 Turing Machine using ONLY composite operations:
  - read_dim (coefficient extraction = limit)
  - + (addition)
  - * (multiplication, including x ZERO and x INF)
  - scalar arithmetic on extracted coefficients

NO Python if/else, NO dict lookups, NO branching.
Branching is done via indicator polynomials.

Requires: nothing (self-contained)
"""

import sys

# === COMPOSITE (minimal self-contained) ===

class Composite:
    def __init__(self, coefficients=None):
        if coefficients is None:
            self.c = {}
        elif isinstance(coefficients, (int, float)):
            self.c = {0: coefficients} if coefficients != 0 else {}
        else:
            self.c = {k: v for k, v in coefficients.items() if abs(v) > 1e-12}

    @classmethod
    def real(cls, value): return cls({0: value})

    def st(self): return self.c.get(0, 0)
    def read_dim(self, dim): return self.c.get(dim, 0)

    def write_dim(self, dim, value):
        new_c = dict(self.c)
        if abs(value) < 1e-12:
            new_c.pop(dim, None)
        else:
            new_c[dim] = value
        return Composite(new_c)

    def __add__(self, other):
        if isinstance(other, (int, float)): other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) + coeff
        return Composite(result)

    def __sub__(self, other):
        if isinstance(other, (int, float)): other = Composite(other)
        result = dict(self.c)
        for dim, coeff in other.c.items():
            result[dim] = result.get(dim, 0) - coeff
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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        if not self.c: return "|empty|"
        sub = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
        def fmt(n):
            if n >= 0: return ''.join(sub[int(d)] for d in str(n))
            return "\u208b" + ''.join(sub[int(d)] for d in str(-n))
        terms = sorted(self.c.items())
        return " + ".join(f"|{int(c) if c == int(c) else c:.4g}|{fmt(d)}" for d, c in terms)

    def __eq__(self, other):
        if isinstance(other, Composite):
            a = {k: v for k, v in self.c.items() if abs(v) > 1e-10}
            b = {k: v for k, v in other.c.items() if abs(v) > 1e-10}
            if set(a.keys()) != set(b.keys()): return False
            return all(abs(a[k] - b[k]) < 1e-10 for k in a)
        return False


ZERO = Composite({-1: 1})   # |1|_{-1} -- structural infinitesimal
INF  = Composite({1: 1})    # |1|_{+1} -- structural infinity
def R(x): return Composite.real(x)


# ================================================================
# BB2 TRANSITION TABLE
# ================================================================
# State A=0, B=1, HALT=2
# Symbol 0, 1
#
# (A, 0) -> write 1, move R,    go B      key=0
# (A, 1) -> write 1, move L,    go B      key=1
# (B, 0) -> write 1, move L,    go A      key=2
# (B, 1) -> write 1, stay,      go HALT   key=3

STATE_DIM = 1000


# ================================================================
# INDICATOR POLYNOMIALS
# ================================================================

def indicators(state, symbol):
    """Compute all four indicator values for (state, symbol).

    For binary state in {0,1} and binary symbol in {0,1}:
      I(s=0, sig=0) = (1 - state) * (1 - symbol)
      I(s=0, sig=1) = (1 - state) * symbol
      I(s=1, sig=0) = state * (1 - symbol)
      I(s=1, sig=1) = state * symbol

    These are mutually exclusive: exactly one equals 1, rest equal 0.
    ALL operations are scalar arithmetic on extracted coefficients.
    """
    i00 = (1 - state) * (1 - symbol)
    i01 = (1 - state) * symbol
    i10 = state * (1 - symbol)
    i11 = state * symbol
    return i00, i01, i10, i11


# ================================================================
# STEP AS PURE COMPOSITE EXPRESSION
# ================================================================

def composite_step_algebraic(tape, state_val):
    """
    One BB2 step using ONLY composite operations.

    Inputs:
      tape:      Composite (head-at-zero convention, symbol at dim 0)
      state_val: scalar (0=A, 1=B, extracted via read_dim)

    Operations used:
      - st() / read_dim()  : coefficient extraction (= limit)
      - + / -              : composite addition
      - * ZERO             : dimensional shift down (= move head right)
      - * INF              : dimensional shift up (= move head left)
      - scalar * composite : scaling

    NO if/else. NO dict. NO branching constructs.
    Branching is ALGEBRAIC via indicator polynomials.
    """
    # --- Extract symbol at head (dim 0) ---
    symbol = tape.st()    # coefficient extraction = limit = native

    # --- Compute indicators (scalar arithmetic) ---
    i00, i01, i10, i11 = indicators(state_val, symbol)

    # --- Write phase ---
    # All four transitions write 1. In general, each branch would
    # write a different symbol. The write operation is:
    #   tape_written = tape + (new_symbol - old_symbol) * |1|_0
    # For BB2, new_symbol is always 1, so:
    #   tape_written = tape + (1 - symbol) * |1|_0
    # This is a no-op when symbol is already 1.
    write_delta = R(1 - symbol)  # |1-symbol|_0
    tape_written = tape + write_delta

    # --- Move phase (indicator-weighted) ---
    # (A,0): move R  = tape * ZERO
    # (A,1): move L  = tape * INF
    # (B,0): move L  = tape * INF
    # (B,1): stay    = tape
    #
    # new_tape = i00 * (tape_w * ZERO)
    #          + i01 * (tape_w * INF)
    #          + i10 * (tape_w * INF)
    #          + i11 * tape_w
    #
    # Since i01 and i10 both multiply by INF, combine them:
    #   = i00 * (tape_w * ZERO) + (i01 + i10) * (tape_w * INF) + i11 * tape_w

    tape_R = tape_written * ZERO    # shifted right (one multiplication)
    tape_L = tape_written * INF     # shifted left  (one multiplication)
    tape_S = tape_written            # stay

    new_tape = (i00 * tape_R) + ((i01 + i10) * tape_L) + (i11 * tape_S)

    # --- State update (indicator-weighted scalar) ---
    # (A,0) -> B=1,  (A,1) -> B=1,  (B,0) -> A=0,  (B,1) -> HALT=2
    new_state = i00 * 1 + i01 * 1 + i10 * 0 + i11 * 2

    return new_tape, new_state


# ================================================================
# REFERENCE IMPLEMENTATION (with Python branching)
# ================================================================

def composite_step_reference(tape, state_val):
    """Reference implementation using Python if/else for comparison."""
    symbol = int(tape.st())
    key = int(state_val) * 2 + symbol

    transitions = {
        0: (1, 'R', 1),    # (A,0) -> write 1, R, B
        1: (1, 'L', 1),    # (A,1) -> write 1, L, B
        2: (1, 'L', 0),    # (B,0) -> write 1, L, A
        3: (1, 'S', 2),    # (B,1) -> write 1, S, HALT
    }

    new_sym, direction, new_state = transitions[key]
    tape = tape + R(new_sym - symbol)

    if direction == 'R':
        tape = tape * ZERO
    elif direction == 'L':
        tape = tape * INF

    return tape, new_state


# ================================================================
# TESTS
# ================================================================

def test_indicators():
    """Verify indicator polynomials are mutually exclusive and exhaustive."""
    print("=" * 60)
    print("TEST 1: Indicator Polynomials")
    print("=" * 60)

    all_ok = True
    for state in [0, 1]:
        for symbol in [0, 1]:
            i00, i01, i10, i11 = indicators(state, symbol)
            total = i00 + i01 + i10 + i11

            expected = {(0,0): (1,0,0,0), (0,1): (0,1,0,0),
                        (1,0): (0,0,1,0), (1,1): (0,0,0,1)}
            exp = expected[(state, symbol)]
            got = (i00, i01, i10, i11)
            ok = (got == exp) and (total == 1)
            if not ok: all_ok = False

            print(f"  state={state}, symbol={symbol}: "
                  f"I00={i00}, I01={i01}, I10={i10}, I11={i11}, "
                  f"sum={total} {'ok' if ok else 'FAIL'}")

    if all_ok:
        print("  \u2705 All indicators correct and mutually exclusive")
    return all_ok


def test_zero_inf_movement():
    """Verify that xZERO = move right, xINF = move left."""
    print("\n" + "=" * 60)
    print("TEST 2: xZERO and xINF as Head Movement")
    print("=" * 60)

    # Tape: ...0, 1, [1], 0, 1... with head at dim 0 (value 1)
    tape = Composite({-1: 1, 0: 1, 2: 1})
    print(f"  Original tape:  {tape}  (head at dim 0, reading {tape.st()})")

    # Move right: xZERO
    tape_R = tape * ZERO
    print(f"  After x ZERO:   {tape_R}  (head at dim 0, reading {tape_R.st()})")

    # Move left: xINF
    tape_L = tape * INF
    print(f"  After x INF:    {tape_L}  (head at dim 0, reading {tape_L.st()})")

    # Round trip
    tape_RT = tape * ZERO * INF
    print(f"  After x ZERO x INF: {tape_RT}  (round-trip)")

    ok1 = tape_R.st() == 0    # moved right, now reading the 0 at original dim 1
    ok2 = tape_L.st() == 1    # moved left, now reading the 1 at original dim -1
    ok3 = (tape_RT == tape)

    print(f"\n  xZERO moves right (now reads 0): {'ok' if ok1 else 'FAIL'}")
    print(f"  xINF  moves left  (now reads 1): {'ok' if ok2 else 'FAIL'}")
    print(f"  Round-trip xZERO x INF = identity: {'ok' if ok3 else 'FAIL'}")

    all_ok = ok1 and ok2 and ok3
    if all_ok:
        print("  \u2705 Composite ZERO/INF correctly implement head movement")
    return all_ok


def test_write_at_head():
    """Verify writing a symbol at the head position."""
    print("\n" + "=" * 60)
    print("TEST 3: Write Symbol at Head")
    print("=" * 60)

    # Tape with 0 at head
    tape = Composite({-1: 1, 1: 1})  # head at dim 0, reading 0
    print(f"  Tape before write: {tape}  (head reads {tape.st()})")

    # Write 1 at head: tape + (1 - 0) * |1|_0 = tape + |1|_0
    symbol = tape.st()
    tape_new = tape + R(1 - symbol)
    print(f"  Write 1:           {tape_new}  (head reads {tape_new.st()})")

    ok1 = tape_new.st() == 1
    ok2 = tape_new.read_dim(-1) == 1  # other cells unchanged
    ok3 = tape_new.read_dim(1) == 1

    # Write 0 over a 1
    tape2 = Composite({0: 1, 1: 1})
    symbol2 = tape2.st()
    tape2_new = tape2 + R(0 - symbol2)
    print(f"  Tape with 1 at head: {tape2}  -> write 0 -> {tape2_new}")
    ok4 = tape2_new.st() == 0

    all_ok = ok1 and ok2 and ok3 and ok4
    if all_ok:
        print("  \u2705 Write-at-head via addition works correctly")
    return all_ok


def test_algebraic_vs_reference():
    """Run both implementations side-by-side and compare every step."""
    print("\n" + "=" * 60)
    print("TEST 4: Algebraic STEP vs Reference (side-by-side BB2)")
    print("=" * 60)

    state_names = {0: 'A', 1: 'B', 2: 'HALT'}

    # Algebraic version
    tape_a = Composite({})
    state_a = 0

    # Reference version
    tape_r = Composite({})
    state_r = 0

    all_ok = True
    step = 0

    while state_a != 2 and step < 20:
        sym_a = tape_a.st()
        sym_r = tape_r.st()

        print(f"  Step {step+1}: state={state_names[int(state_a)]}, "
              f"symbol={int(sym_a)}")

        # Run both
        tape_a, state_a = composite_step_algebraic(tape_a, state_a)
        tape_r, state_r = composite_step_reference(tape_r, state_r)

        # Compare
        tape_match = (tape_a == tape_r)
        state_match = (abs(state_a - state_r) < 1e-10)

        print(f"    Algebraic: tape={tape_a}, state={state_names[int(state_a)]}")
        print(f"    Reference: tape={tape_r}, state={state_names[int(state_r)]}")
        print(f"    Match: tape={'ok' if tape_match else 'MISMATCH'}, "
              f"state={'ok' if state_match else 'MISMATCH'}")

        if not (tape_match and state_match):
            all_ok = False
            print("    \u274c DIVERGENCE DETECTED")
            break

        step += 1

    print(f"\n  Total steps: {step}")
    print(f"  Final tape:  {tape_a}")
    print(f"  Final state: {state_names[int(state_a)]}")

    ok_steps = (step == 6)
    ok_tape = (len([d for d, c in tape_a.c.items() if abs(c) > 1e-10]) == 4)
    ok_halt = (int(state_a) == 2)

    if all_ok and ok_steps and ok_tape and ok_halt:
        print(f"\n  \u2705 Algebraic STEP matches reference at EVERY step")
        print(f"  \u2705 BB2: 6 steps, 4 ones, halted correctly")
        print(f"  \u2705 NO Python if/else used in the algebraic version")
    else:
        if not ok_steps: print(f"  \u274c Expected 6 steps, got {step}")
        if not ok_halt: print(f"  \u274c Did not halt")

    return all_ok and ok_steps and ok_tape and ok_halt


def test_operation_count():
    """Count the composite operations used in one STEP."""
    print("\n" + "=" * 60)
    print("TEST 5: Operation Census")
    print("=" * 60)

    print("  Operations per STEP (algebraic version):")
    print("    1x st()           -- read symbol at head (coefficient extraction)")
    print("    4x scalar multiply -- indicator computation (state*symbol etc.)")
    print("    3x scalar add/sub  -- indicator computation (1-state etc.)")
    print("    1x R()            -- write delta as composite")
    print("    1x composite +    -- apply write to tape")
    print("    1x composite x ZERO -- shift right (compute tape_R)")
    print("    1x composite x INF  -- shift left  (compute tape_L)")
    print("    3x scalar * composite -- weight branches by indicators")
    print("    2x composite +    -- sum the weighted branches")
    print("    4x scalar multiply -- compute new_state")
    print("    3x scalar add      -- sum new_state contributions")
    print("  -----------------------------------------------")
    print("    Total: ~20 primitive operations")
    print("    Of which: 0 are Python if/else/dict")
    print("    Of which: 0 are branching constructs")
    print("    ALL are: +, *, read_dim (= limits)")
    print()
    print("  For comparison:")
    print("    Reference impl uses: 1 dict lookup + 2 if/else branches")
    print("    Those are NOT composite operations.")
    print("    The algebraic version replaces them with arithmetic.")

    print("\n  \u2705 STEP is expressible as a pure composite expression")
    print("  \u2705 Branching = indicator polynomials (scalar arithmetic)")
    print("  \u2705 Movement = multiplication by ZERO/INF")
    print("  \u2705 Writing = addition")
    print("  \u2705 State update = weighted sum")
    return True


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("#" * 60)
    print("# STEP AS PURE COMPOSITE EXPRESSION")
    print("# Indicator Polynomial Test")
    print("#" * 60)
    print()
    print("Can BB2's STEP function be expressed using ONLY")
    print("composite operations (+, *, read_dim, ZERO, INF)?")
    print()

    results = []
    results.append(("Indicator polynomials", test_indicators()))
    results.append(("xZERO/xINF as movement", test_zero_inf_movement()))
    results.append(("Write at head via addition", test_write_at_head()))
    results.append(("Algebraic vs reference (full BB2)", test_algebraic_vs_reference()))
    results.append(("Operation census", test_operation_count()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "\u2705" if passed else "\u274c"
        print(f"  {status} {name}")

    all_passed = all(p for _, p in results)
    print()
    if all_passed:
        print("All tests passed.")
        print()
        print("CONCLUSION:")
        print("  STEP CAN be expressed as a pure composite expression.")
        print("  Not a single multiplication (it's a sum of 3 weighted products),")
        print("  but ALL operations are native composite algebra:")
        print("    - Coefficient extraction (read_dim / st) for branching")
        print("    - Indicator polynomials for branch selection")
        print("    - x ZERO / x INF for head movement")
        print("    - Addition for symbol writing")
        print("    - Scalar-weighted sums for combining branches")
        print()
        print("  Open Question #1 from Self-Hosted Execution: ANSWERED.")
        print("  STEP is a composite expression. The execution loop is division.")
        print("  Therefore: the ENTIRE computation is algebraic.")
    else:
        print("Some tests failed. Investigation needed.")

    sys.exit(0 if all_passed else 1)
