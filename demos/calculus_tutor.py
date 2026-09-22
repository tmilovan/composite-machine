#!/usr/bin/env python3
# Composite Machine — an interactive calculus tutor for the console
# Author: Toni Milovan <tmilovan@fwd.hr>
# License: AGPL-3.0
"""Learn calculus by predicting, then checking against the arithmetic.

    $ PYTHONPATH=. python3 demos/calculus_tutor.py          # all lessons
    $ PYTHONPATH=. python3 demos/calculus_tutor.py 3        # one lesson

Each lesson has three parts: THE IDEA, a WORKED EXAMPLE with real numbers,
then your turn -- you pick the numbers and say what you think the answer is,
and the arithmetic tells you whether you were right.

  choice prompts   show a suggestion in [brackets]; Enter accepts it
  PREDICTION prompts show nothing -- a default would be the answer.
                   Enter there means "show me", and is not scored correct.
  'q' at any prompt moves on to the next lesson.
"""
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composite.composite_lib import (Composite, R, ZERO, INF, sin, cos, exp,
                                     ln, sqrt, antiderivative, integrate,
                                     limit, taylor_coefficients)

# COMPOSITE transcendentals only.  This used to pair each one with its
# math.* twin, and the twin was never read -- dead weight that would have
# handed anyone writing FUNCS[name][1] the float path silently, which is
# the exact hole lesson 10 warns about.
FUNCS = {"sin": sin, "cos": cos, "exp": exp, "ln": ln, "sqrt": sqrt,
         "x": lambda t: t, "x*x": lambda t: t * t}


# ------------------------------------------------------------------ input
class Quit(Exception):
    pass


def _raw(prompt, default):
    try:
        s = input(f"    {prompt} [{default}]: ").strip()
    except EOFError:
        print(f"    {prompt} [{default}]: (no input -- using {default})")
        return ""
    if s.lower() in ("q", "quit", "done", "n"):
        raise Quit
    return s


def ask_num(prompt, default):
    while True:
        s = _raw(prompt, default)
        if s == "":
            return float(default)
        try:
            return float(eval(s, {"__builtins__": {}, "pi": math.pi,
                                  "e": math.e, "sqrt": math.sqrt}))
        except Exception:
            print("      not a number -- try again (or q)")


def ask_int(prompt, default, lo=None, hi=None):
    """An integer choice, with the allowed range shown IN the prompt.

    'pick an n NOT in the table [9]' sat directly above a prediction prompt
    that also wanted a bare number, so the coefficient got typed into the n
    slot: n = 589824, which overflowed to inf.
    """
    if lo is not None and hi is not None:
        prompt = f"{prompt} ({lo}-{hi})"
    while True:
        v = int(ask_num(prompt, default))
        if lo is not None and v < lo:
            print(f"      too small -- {lo} or more"); continue
        if hi is not None and v > hi:
            print(f"      too big -- {hi} or less (it would overflow)"); continue
        return v


def ask_word(prompt, default, allowed):
    while True:
        s = _raw(prompt, default) or str(default)
        if s in allowed:
            return s
        print(f"      pick one of: {', '.join(allowed)}")


def ask_guess(prompt):
    """Ask for a PREDICTION.  No default, because a default is the answer.

    Every prediction prompt here used to carry the right value in brackets,
    so pressing Enter scored you correct and typing anything meant you had
    already read it.  That makes being wrong impossible, which makes the
    exercise worthless -- the same way a test that prints a tick and hides
    the numbers cannot tell you anything.

    Blank means "I do not know, show me" and is NOT counted as right.
    """
    try:
        s = input(f"    {prompt} (blank = show me): ").strip()
    except EOFError:
        print(f"    {prompt} (blank = show me): (no input)")
        return None
    if s.lower() in ("q", "quit", "done"):
        raise Quit
    if s == "":
        return None
    try:
        return float(eval(s, {"__builtins__": {}, "pi": math.pi,
                              "e": math.e, "sqrt": math.sqrt}))
    except Exception:
        print("      could not read that as a number")
        return None


def ask_guess_word(prompt, allowed):
    """Same, for a word answer."""
    try:
        s = input(f"    {prompt} (blank = show me): ").strip().lower()
    except EOFError:
        print(f"    {prompt} (blank = show me): (no input)")
        return None
    if s in ("q", "quit", "done"):
        raise Quit
    return s if s in allowed else None


def ask_pick(prompt, options):
    """A categorical prediction -- the kind a learner can actually answer.

    'so (f*g)' is?' wants sixteen digits and the only rational reply is to
    skip.  A question with three named answers can be got right or wrong,
    and being wrong there teaches something.
    """
    try:
        a = input(f"    {prompt} {'/'.join(options)} (blank = show me): ")
    except EOFError:
        print(f"    {prompt} (no input)"); return None
    a = a.strip().lower()
    if a in ("q", "quit", "done"):
        raise Quit
    return a if a in options else None


def score_pick(guess, actual, why=""):
    if guess is None:
        print(f"      it is {actual}{'  -- ' + why if why else ''}")
        return False
    if guess == actual:
        print(f"      RIGHT -- {actual}{'  ' + why if why else ''}")
        return True
    print(f"      no -- you said {guess}, it is {actual}"
          f"{'  ' + why if why else ''}")
    return False


def score(guess, actual, tol=5e-3):
    """Mark a prediction.  BOTH sides must be finite.

    With actual = inf, `abs(g - inf) <= tol * max(1, inf)` is `inf <= inf`,
    which Python evaluates TRUE -- so the tutor printed
    "RIGHT.  589824.0 == inf" and confirmed a wrong answer.  Confirming a
    wrong answer is the one failure mode that actively damages a learner, so
    non-finite values are rejected before any comparison happens.

    The default tolerance is RELATIVE and loose on purpose: nobody can
    predict sixteen digits, so 4.5 must score right against 4.517113399272802.
    Two significant figures is the bar.
    """
    if not math.isfinite(actual):
        print(f"      that overflowed -- the arithmetic gave {actual!r}, which")
        print("      is not a number anything can be checked against.")
        return False
    if guess is None:
        print(f"      the arithmetic says {actual!r}")
        return False
    if not math.isfinite(guess):
        print(f"      {guess!r} is not a finite number")
        return False
    # "two significant figures is enough" has to MEAN that.  A relative
    # bound of 5e-3 rejected 5.4 against 5.43656365691809 -- the answer the
    # prompt asked for, refused by the prompt's own checker.  Round both to
    # sig figs and compare, which is exactly what the instruction says.
    def _sig(v, n=2):
        if v == 0.0:
            return 0.0
        from math import floor, log10
        return round(v, -int(floor(log10(abs(v)))) + (n - 1))

    if guess == actual:
        print(f"      RIGHT.  {guess!r} exactly")
        return True
    if abs(guess - actual) <= tol * max(1.0, abs(actual)):
        print(f"      RIGHT.  {guess!r} vs {actual!r}"
              f"   (within {tol:g} relative)")
        return True
    if _sig(guess) == _sig(actual):
        # Say WHICH rule admitted it.  Reporting the rounded pair for an
        # exact match read as though the checker had rounded 196830 to
        # 200000 in order to accept it.
        print(f"      RIGHT.  {guess!r} vs {actual!r}"
              f"   (both round to {_sig(actual):g} at 2 s.f.)")
        return True
    print(f"      no -- you said {guess!r}, the arithmetic says {actual!r}"
          f"   (off by {abs(guess-actual):.4g})")
    return False


def verdict_on_guess(guess, actual, tol=1e-6):
    if abs(guess - actual) <= tol * max(1.0, abs(actual)):
        print(f"      RIGHT.  {guess!r} == {actual!r}")
    else:
        print(f"      not quite.  you said {guess!r}, the arithmetic says "
              f"{actual!r}  (off by {abs(guess-actual):.4g})")


def clip(v, n=48):
    """Print a composite without cutting a number in half."""
    t = str(v)
    return t if len(t) <= n else t[:n] + " ..."


def nudge(a):
    return R(float(a)) + ZERO


def head(n, title):
    print("\n" + "=" * 68)
    print(f"LESSON {n}  —  {title}")
    print("=" * 68 + "\n")


def practice(fn):
    """Run a practice round until the learner says stop."""
    print("\n  --- your turn " + "-" * 50)
    while True:
        try:
            fn()
        except Quit:
            print("    (moving on)\n")
            return
        try:
            again = _raw("another? y/n", "n")
        except Quit:
            print()
            return
        if again.lower() not in ("y", "yes"):
            print()
            return


# ------------------------------------------------------------------- 1
def lesson1():
    head(1, "A derivative is a coefficient you can read off")
    print("""    THE IDEA
    An ordinary number tells you where a function is.  A NUDGED number
    carries a second piece, and it matters what that piece IS.

        h is ZERO.

    Not "a little bit more", not a small real number -- the composite
    zero, |1|_-1, a first-order zero that the arithmetic keeps instead
    of discarding.  Everywhere below, h is shorthand for ZERO, because
    9 + 6*ZERO + ZERO**2 is unreadable and 9 + 6h + h^2 is not.

    That distinction is the whole point, and it is easy to lose.  In
    ordinary calculus the h in

        (f(a+h) - f(a)) / h

    is a small REAL number, and the derivative is what that quotient
    approaches as h shrinks -- a limit, never reached.  Here nothing
    shrinks and no limit is taken.  h is a number in the system, the
    division is performed, and the answer is read off.  Lesson 7 is
    where that difference stops being philosophical.

    A smooth f, expanded about a point a, looks like

        f(a + h) = f(a)  +  f'(a).h  +  f''(a)/2 . h^2  +  ...

    so the number multiplying h IS the derivative.  Not a rule -- a
    position in the answer.

    WORKED EXAMPLE:  f(x) = x*x  at  a = 3""")
    x = nudge(3)
    print(f"\n      x   = nudge(3)          = {x}        (3 + h)")
    print(f"      x*x                     = {x*x}")
    print("""
      By hand:  (3 + h)(3 + h) = 9 + 3h + 3h + h*h
                              = 9 + 6h + h^2

      dimension  0 : 9   the value      f(3)  = 9
      dimension -1 : 6   the h term     f'(3) = 6      (2*3, as expected)
      dimension -2 : 1   the h^2 term   f''(3)/2 = 1,  so f''(3) = 2

    Note the last one: coefficient k is f^(k)(a) divided by k!, because
    h^k can be assembled k! different ways.  That is why 1 means 2 here.""")

    def round_():
        a = ask_num("pick a point a", 5)
        n = ask_int("pick a power n", 3, 1, 12)
        # ASK BEFORE COMPUTING.  Printing the composite first puts the answer
        # on the screen, and then the prediction tests nothing.
        print(f"\n      so f(x) = x**{n}, and we want f'({a:g})")
        g = ask_guess(f"before we compute it: what is f'({a:g})?")
        y = nudge(a) ** n
        print(f"\n      nudge({a:g})**{n} = {y}")
        score(g, y.coeff(-1))
        print(f"      and f''({a:g}) = 2 x {y.coeff(-2)!r} = {2*y.coeff(-2)!r}\n")
    practice(round_)


# ------------------------------------------------------------------- 2
def lesson2():
    head(2, "Find the power rule yourself")
    print("""    THE IDEA
    You are usually TOLD that the derivative of x^n is n*x^(n-1).  Here
    you can watch it appear instead.  Expand (a + h)^n by the binomial
    theorem:

        (a + h)^n = a^n  +  n*a^(n-1)*h  +  C(n,2)*a^(n-2)*h^2  +  ...

    There are n brackets, and to get a single h you take the h from
    exactly one of them and the a from all the rest.  There are n ways
    to choose which bracket gives up its h, so the h term is n*a^(n-1).
    The power rule is a counting fact about brackets.

    WORKED EXAMPLE:  a = 2, powers 1 to 6.  Watch the third column.""")
    a = 2
    print(f"\n      {'n':>3} {'(2+h)^n value':>14} {'h coeff':>9}   n x a^(n-1)")
    print("      " + "-" * 48)
    for n in range(1, 7):
        y = nudge(a) ** n
        print(f"      {n:>3} {y.st():>14.6g} {y.coeff(-1):>9.6g}   "
              f"{n} x 2^{n-1} = {n*a**(n-1)}")
    print("""
      The h column and the last column are the same number every time.
      Nothing enforced that -- it is what multiplying the brackets out
      leaves behind.""")

    def round_():
        a = ask_num("pick a point a", 3)
        print(f"\n      {'n':>3} {'value':>12} {'h coeff':>12}   n x a^(n-1)")
        print("      " + "-" * 46)
        for n in range(1, 7):
            y = nudge(a) ** n
            print(f"      {n:>3} {y.st():>12.6g} {y.coeff(-1):>12.6g}"
                  f"   {n*a**(n-1):>12.6g}")
        n = ask_int("now predict for an n NOT in the table", 9, 7, 15)
        g = ask_guess(f"what will the h coefficient be at n={n}?")
        score(g, (nudge(a) ** n).coeff(-1))
        print()
    practice(round_)


# ------------------------------------------------------------------- 3
def lesson3():
    head(3, "The product rule is one multiplication")
    print("""    THE IDEA
    You have two nudged numbers.  Each is a value plus a slope:

        f = f0 + f1.h      f0 is f(a),  f1 is f'(a)
        g = g0 + g1.h      g0 is g(a),  g1 is g'(a)

    Multiply them the way you would multiply any two brackets -- every
    term on the left against every term on the right:

        f0.g0        no h at all        -> the value of the product
        f0.(g1.h)    one h, from g      -> an h term
        (f1.h).g0    one h, from f      -> an h term
        (f1.h).(g1.h) two h's           -> an h^2 term

    There are exactly TWO ways to end up with a single h: take it from f
    and not from g, or from g and not from f.  Add them and you have

        f0.g1 + f1.g0        which is    f.g' + f'.g

    That is the product rule.  Nobody decided it -- it is a count of how
    many ways one h can come out of two brackets.

    WORKED EXAMPLE 1 -- small whole numbers, so you can follow every term

      f = x  and  g = x*x,  both at a = 2.  Their product is x^3, whose
      derivative at 2 we already know: 3 * 2^2 = 12.""")
    x = nudge(2)
    f, g = x, x * x
    print(f"\n      f = x   = {f}          f0 = 2, f1 = 1")
    print(f"      g = x*x = {g}   g0 = 4, g1 = 4")
    print("""
      Multiply (2 + h)(4 + 4h + h^2) out in full:

           2 * 4      =  8
           2 * 4h     =  8h        <- f0 * g1, the h came from g
           2 * h^2    =  2h^2
           h * 4      =  4h        <- f1 * g0, the h came from f
           h * 4h     =  4h^2
           h * h^2    =  h^3
        ------------------------------
                      =  8 + 12h + 6h^2 + h^3
                             ^^^
                         8 + 4, the two ways of getting one h""")
    print(f"\n      and the arithmetic agrees:  f*g = {f*g}")
    print(f"      h coefficient             = {(f*g).coeff(-1)!r}")
    print(f"      f0 x g1 + f1 x g0 = 2x4 + 1x4 = {2*4 + 1*4}")
    print(f"      textbook 3 x x^2 at x=2   = {3*2**2}")
    print("""
      Four numbers, all 12.  The expansion, the composite, the rule, and
      the answer you already knew.

    WORKED EXAMPLE 2 -- the same thing where you cannot do it in your head

      f = sin, g = exp, at a = 1.3.  Nothing changes except that f0, f1,
      g0 and g1 are no longer small integers.""")
    a = 1.3
    x = nudge(a)
    F, G = sin(x), exp(x)
    f0, f1, g0, g1 = F.st(), F.coeff(-1), G.st(), G.coeff(-1)
    print(f"\n      f = sin(x) = {clip(F, 40)}")
    print(f"          f0 = sin(1.3) = {f0!r}")
    print(f"          f1 = cos(1.3) = {f1!r}    <- the h coefficient IS f'")
    print(f"      g = exp(x) = {clip(G, 40)}")
    print(f"          g0 = {g0!r}")
    print(f"          g1 = {g1!r}    <- exp is its own derivative")
    print(f"\n      f0 x g1 = {f0!r}  x  {g1!r}")
    print(f"            = {f0*g1!r}")
    print(f"      f1 x g0 = {f1!r}  x  {g0!r}")
    print(f"            = {f1*g0!r}")
    print(f"      sum   = {f0*g1 + f1*g0!r}")
    print(f"\n      (f*g).coeff(-1), from the multiply alone:")
    print(f"            = {(F*G).coeff(-1)!r}")
    print("""
      The rule and the multiplication give the same number because they
      ARE the same operation, written twice.

    AND THE QUOTIENT RULE, which nobody enjoys memorising

        (f/g)' = (f'.g - f.g') / g^2

      It comes out of division for the same reason: it is what the h
      term of the quotient has to be.""")
    print(f"      by hand   : {(f1*g0 - f0*g1)/(g0*g0)!r}")
    print(f"      by divide : {(F/G).coeff(-1)!r}")

    def round_():
        fn = ask_word("f", "sin", list(FUNCS))
        gn = ask_word("g", "exp", list(FUNCS))
        a = ask_num("at a", 1.3)
        F, G = FUNCS[fn](nudge(a)), FUNCS[gn](nudge(a))
        f0, f1, g0, g1 = F.st(), F.coeff(-1), G.st(), G.coeff(-1)
        print(f"\n      f={fn}: f0={f0:.8g} f1={f1:.8g}")
        print(f"      g={gn}: g0={g0:.8g} g1={g1:.8g}")
        want = "positive" if f0 * g1 + f1 * g0 > 0 else "negative"
        score_pick(ask_pick("first: is (f*g)' positive or negative?",
                            ["positive", "negative"]), want,
                   f"(f0*g1 = {f0*g1:.4g}, f1*g0 = {f1*g0:.4g})")
        g = ask_guess("now the value -- two significant figures is enough")
        score(g, (F * G).coeff(-1))
        try:
            print(f"      and (f/g)' = {(F/G).coeff(-1)!r}")
        except Exception as e:
            print(f"      (f/g) undefined here: {type(e).__name__}")
        print()
    practice(round_)


# ------------------------------------------------------------------- 4
def lesson4():
    head(4, "The chain rule is the inner nudge, passed along")
    print("""    THE IDEA
    Two functions, one inside the other: f(g(x)).  Students are told to
    "differentiate the outside, then multiply by the derivative of the
    inside", and the multiply is the bit that gets forgotten.  Here it
    cannot be forgotten, because nobody performs it on purpose.

    Send a nudged x into g:

        x = a + 1.h                 x moves by h
        u = g(a) + g'(a).h          u moves by g'(a) times h

    The nudge came out RESCALED.  g stretched it by its own slope.

    Now hand u to f.  f has no idea where u came from and does not care.
    It does what it always does: reports its value at u's value, plus its
    own slope times whatever nudge u is carrying:

        f(u) = f(u0) + f'(u0) * (the nudge u carries)
             = f(u0) + f'(u0) * g'(a) * h

    There is the chain rule, and no step was applied.  It is arithmetic
    on a number that arrived already stretched.

    WORKED EXAMPLE 1 -- whole numbers again, so the terms are visible

      inner  g(x) = x*x  at a = 3
      outer  f(u) = u*u
      together    f(g(x)) = x^4, whose derivative at 3 is 4 * 27 = 108.""")
    y = nudge(3)
    u = y * y
    print(f"\n      x        = {y}")
    print(f"      u = x*x  = {u}")
    print(f"\n      x carried a nudge of 1.  u carries a nudge of "
          f"{u.coeff(-1)!r}.")
    print(f"      That {u.coeff(-1):g} is g'(3) = 2 x 3.  The nudge was stretched by 6.")
    print("""
      Now square u.  f does not know 'u came from x^2' -- it just
      multiplies the number it was handed by itself:

           (9 + 6h + h^2)(9 + 6h + h^2)

           9 * 9        =  81
           9 * 6h       =  54h     <- u0 times u's nudge
           6h * 9       =  54h     <- u's nudge times u0
           ... and the h^2 and higher terms
        -------------------------------
                        =  81 + 108h + ...
                                ^^^^
                          54 + 54 = 2 * 9 * 6 = f'(u0) * g'(3)""")
    print(f"\n      the arithmetic: u*u = {clip(u*u, 44)}")
    print(f"      h coefficient       = {(u*u).coeff(-1)!r}")
    print(f"      f'(u0) x g'(a)      = 2 x {u.st():g} x {u.coeff(-1):g} "
          f"= {2*u.st()*u.coeff(-1):g}")
    print(f"      textbook 4 x x^3 at 3 = {4*3**3}")
    print("""
      The 108 was never assembled by a rule.  It is 54 + 54: u0 meeting
      u's nudge, twice, exactly as in lesson 3.  With THIS outer function
      -- f(u) = u*u -- the chain rule is literally the product rule with
      both brackets the same number.  That is special to squaring; for a
      general f the outer slope is f'(u0) rather than 2.u0.  What carries
      over is the part that matters: whatever f does, it multiplies by
      the nudge it was handed.

    WORKED EXAMPLE 2 -- a transcendental outside, where you cannot expand by hand

      f = sin, g = x*x, at a = 1.7.""")
    a = 1.7
    x = nudge(a)
    u = x * x
    z = sin(u)
    u0, u1 = u.st(), u.coeff(-1)
    print(f"\n      u = x*x = {u}")
    print(f"        u0 = {u0!r}")
    print(f"        u1 = {u1!r}     <- g'(1.7) = 2 x 1.7")
    print(f"\n      sin now expands about u0, and multiplies by the nudge:")
    print(f"        f'(u0) = cos({u0!r})")
    print(f"               = {math.cos(u0)!r}")
    print(f"        times u1 = {u1!r}")
    print(f"               = {math.cos(u0)*u1!r}")
    print(f"\n      sin(u).coeff(-1), straight from the arithmetic:")
    print(f"               = {z.coeff(-1)!r}")
    print("""
      Stack a third function on and nothing new happens -- each one
      receives the accumulated nudge and multiplies its own slope in.""")
    w = exp(sin(x * x))
    print(f"      exp(sin(x*x)).coeff(-1)    = {w.coeff(-1)!r}")
    print(f"      e^sin(a^2).cos(a^2).2a     = "
          f"{math.exp(math.sin(a*a))*math.cos(a*a)*2*a!r}")

    def round_():
        outer = ask_word("outer f", "sin", list(FUNCS))
        a = ask_num("at a", 1.7)
        p = ask_int("inner is x**p, pick p", 2, 1, 6)
        u = nudge(a) ** p
        y = FUNCS[outer](u)
        print(f"\n      u = x**{p} = {u}")
        print(f"      u carries a nudge of {u.coeff(-1)!r}   <- u'({a:g})")
        print(f"      f will expand about u0 = {u.st()!r}")
        u1 = u.coeff(-1)
        want = "bigger" if abs(u1) > 1 else "smaller"
        score_pick(ask_pick(f"u carries a nudge of {u1:g}.  Is |f(u)'| bigger "
                            f"or smaller than |f'(u0)|?", ["bigger", "smaller"]),
                   want, f"the nudge is multiplied in, and |{u1:g}| "
                         f"{'>' if abs(u1) > 1 else '<'} 1")
        g = ask_guess("now the value -- two significant figures is enough")
        score(g, y.coeff(-1))
        print(f"      f'(u0) * u'(a) = "
              f"{FUNCS[outer](nudge(u.st())).coeff(-1) * u.coeff(-1)!r}\n")
    practice(round_)


# ------------------------------------------------------------------- 5
def lesson5():
    head(5, "Integration is the same shift, run backwards")
    print("""    THE IDEA
    Differentiating h^k gives k*h^(k-1): the exponent drops by one and
    the old exponent comes down as a factor.  In this notation that is
    a term moving UP one dimension and being multiplied by its order.

    Integration is the same move in reverse.  A term at dimension -k
    becomes a term at dimension -(k+1), divided by the new order:

        |c|_-k   ->   |c/(k+1)|_-(k+1)

    Written in ordinary symbols that reads

        c*h^k   ->   c*h^(k+1) / (k+1)

    which is 'raise the power, divide by the new power' -- the first
    integration rule anyone is taught, and it is a bookkeeping move.

    WORKED EXAMPLE:  f = 1 + 2h + 3h^2""")
    f = Composite({0: 1.0, -1: 2.0, -2: 3.0})
    F = antiderivative(f)
    print(f"\n      f = {f}")
    print(f"      F = {F}\n")
    for k, c in ((0, 1.0), (1, 2.0), (2, 3.0)):
        print(f"        |{c:g}|_{-k:<3} ->  |{c/(k+1):g}|_{-(k+1):<3}"
              f"     {c:g}h^{k} -> {c:g}h^{k+1}/{k+1}")
    print("""
      Differentiate F back and you land exactly on f again -- the two
      operations are one table read in opposite directions.""")
    back = Composite({-(-d - 1): c * (-d) for d, c in F.c.items() if -d >= 1})
    print(f"      d/dh of F = {back}")
    print(f"      original  = {f}")

    def round_():
        print("      enter a short series, e.g.  1,2,3  meaning 1 + 2h + 3h^2")
        s = _raw("coefficients", "1,2,3") or "1,2,3"
        try:
            cs = [float(t) for t in s.replace(" ", "").split(",") if t]
        except ValueError:
            print("      could not read those\n"); return
        f = Composite({-k: c for k, c in enumerate(cs)})
        F = antiderivative(f)
        print(f"\n      f = {f}")
        for k, c in enumerate(cs):
            print(f"        |{c:g}|_{-k:<3} ->  |{c/(k+1):g}|_{-(k+1):<3}"
                  f"    divide by {k+1}")
        print(f"      F = {F}\n")
    practice(round_)


# ------------------------------------------------------------------- 6
def lesson6():
    head(6, "The Fundamental Theorem, one panel at a time")
    print("""    THE IDEA
    The area under a curve is usually introduced as a limit of thinner
    and thinner rectangles, and then -- as a separate miracle -- you are
    told it equals F(b) - F(a).  Here the two are the same act.

    Expand f about the MIDPOINT of the interval and antidifferentiate
    the series (lesson 5, one shift).  Then evaluate at the two ends of
    the panel and subtract.

    AND THERE IS A CROSSING HERE THAT HAS TO BE SAID OUT LOUD.

    f at the midpoint is a composite -- a series in h, with coefficients
    sitting at dimensions 0, -1, -2.  Its antiderivative F is too.  But
    the panel has a real width, and you cannot substitute a real number
    for h: h is ZERO, and putting 0.5 where a structural zero sits is the
    bug class this library actually had (a real panel width was written
    into an infinitesimal's slot and manufactured a real number out of a
    zero).

    What the panel really does is PROJECT.  It reads F's coefficients out
    of the composite and uses them as the coefficients of an ordinary
    real polynomial in a real variable t:

        F = |c0|_0 + |c1|_-1 + |c2|_-2 + ...     lives in h
              |      |         |
              v      v         v
        P(t) = c0 + c1.t + c2.t^2 + ...          lives in the reals

    then evaluates P(+half) - P(-half).  That is a deliberate exit from
    the composite, exactly like st() or d(k) -- one explicit operation
    crossing between two number systems.  eval_taylor() is that exit.

    So: the antiderivative is computed in h, the panel arithmetic happens
    in the reals, and the coefficients are what pass between them.  t is
    never h and h is never substituted.

    No rectangles are involved, and for a polynomial the series is
    finite -- so the answer is not an approximation, it is exact.

    WORKED EXAMPLE:  the area under x*x from 0 to 1""")
    a, b = 0.0, 1.0
    mid, half = (a + b) / 2, (b - a) / 2
    fx = nudge(mid) ** 2
    F = antiderivative(fx)
    print(f"\n      midpoint = {mid:g}, half-width = {half:g}")
    print(f"\n      f at the midpoint : {fx}")
    print(f"        reading those coefficients into a real polynomial:")
    print(f"          P(t) = 0.25 + 1.t + 1.t^2      t real, h nowhere in it")
    r, l = F.eval_taylor(half), F.eval_taylor(-half)
    print(f"      antiderivative F  : {F}")
    print(f"        as a real polynomial, Q(t) = "
          f"{F.coeff(-1):g}.t + {F.coeff(-2):g}.t^2 + {F.coeff(-3):g}.t^3")
    print(f"\n      Q(+{half:g})           : {r!r}")
    print(f"      Q(-{half:g})           : {l!r}")
    print(f"      Q(+{half:g}) - Q(-{half:g}) : {r - l!r}")
    print(f"      true value 1/3    : {1/3!r}")
    print("""
      t is the offset from the MIDPOINT, not x itself -- the panel runs
      from t = -0.5 to t = +0.5, which is x from 0 to 1.  And t is an
      ordinary real number, which is why it is not called h: F lives in
      h, Q lives in the reals, and reading the coefficients across is the
      one step that connects them.

      You will see three spellings of 1/3 above, differing in the last
      bit or two.  The METHOD is exact -- the series terminates, nothing
      is truncated -- but the coefficients are float64, so 1/3 is not
      representable and the additions round differently depending on the
      order they happen in.  That is float arithmetic, not approximation
      in the calculus.""")

    def round_():
        n = ask_int("integrate x**n, pick n", 2, 0, 12)
        a = ask_num("from a", 0)
        b = ask_num("to b", 1)
        mid, half = (a + b) / 2, (b - a) / 2
        F = antiderivative(nudge(mid) ** n)
        panel = F.eval_taylor(half) - F.eval_taylor(-half)
        rect = (mid ** n) * (b - a)
        want = "more" if panel > rect else "less"
        score_pick(ask_pick(f"the midpoint rectangle is {rect:.6g}.  Is the "
                            f"true area more or less?", ["more", "less"]),
                   want, "x**%d curves %s over this interval"
                         % (n, "up" if panel > rect else "down"))
        g = ask_guess("now the area -- two significant figures is enough")
        score(g, panel)
        print(f"      one panel      : {panel!r}")
        print(f"      integrate()    : {integrate(lambda t: t**n, a, b)!r}")
        print(f"      (b^(n+1)-a^(n+1))/(n+1) = "
              f"{(b**(n+1)-a**(n+1))/(n+1)!r}\n")
    practice(round_)


# ------------------------------------------------------------------- 7
def lesson7():
    head(7, "0/0 is not indeterminate -- the zeros carry their order")
    print("""    THE IDEA
    A calculator reduces sin(0)/0 to 0/0 and stops, because both zeros
    have been flattened to the same thing.  But they are not the same
    thing: one of them came from sin, and it vanishes at the SAME RATE
    as the h underneath it.

    Here a zero keeps its order.  h is a first-order zero.  sin(h) is
    also first-order, because its series starts at h.  So sin(h)/h has
    one h on top and one underneath: they cancel like any common factor
    and leave a perfectly ordinary number.

    Change the orders and the answer changes with them:

        order on top  =  order below   ->  finite
        order on top  <  order below   ->  diverges
        order on top  >  order below   ->  vanishes

    WORKED EXAMPLE""")
    h = ZERO
    print(f"\n      h          = {h}        a first-order zero")
    print(f"      sin(h)     = {str(sin(h))[:46]} ...")
    print(f"                   = h - h^3/6 + h^5/120 - ...,  starts at h")
    print(f"      sin(h)/h   = {str(sin(h)/h)[:46]} ...")
    print(f"      its value  = {(sin(h)/h).st()!r}\n")
    print("      The same four expressions a calculator calls '0/0':\n")
    for nm, v, why in (("sin(h)/h     ", sin(h) / h, "order 1 / order 1"),
                       ("sin(h)/h**2  ", sin(h) / (h * h), "order 1 / order 2"),
                       ("sin(h)**2/h**2", sin(h) ** 2 / (h * h), "order 2 / order 2"),
                       ("(1-cos h)/h**2", (1 - cos(h)) / (h * h), "order 2 / order 2")):
        nz = [d for d, c in v.c.items() if c != 0.0]
        lead = max(nz) if nz else None
        tag = ("diverges, order %d" % lead if lead > 0 else
               "finite, value %g" % v.st() if lead == 0 else
               "vanishes, order %d" % -lead)
        print(f"        {nm}  {tag:<22} {why}")

    def round_():
        p = ask_int("numerator: sin(h)**p, pick p", 1, 1, 5)
        q = ask_int("denominator: h**q, pick q", 2, 1, 5)
        print(f"\n      sin(h)**{p} starts at order {p};  h**{q} is order {q}")
        g = ask_guess_word("finite / diverges / vanishes?",
                           ["finite", "diverges", "vanishes"])
        v = sin(h) ** p / h ** q
        nz = [d for d, c in v.c.items() if c != 0.0]
        lead = max(nz) if nz else None
        want = ("diverges" if lead > 0 else
                "finite" if lead == 0 else "vanishes")
        if g == want:
            print(f"      RIGHT -- leading dimension {lead}, so it {want}")
        elif g is None:
            print(f"      leading dimension is {lead}, so it {want}")
        else:
            print(f"      no -- you said {g}; leading dimension is {lead},"
                  f" so it {want}")
        if lead == 0:
            print(f"      value = {v.st()!r}")
        print()
    practice(round_)


# ------------------------------------------------------------------- 8
def lesson8():
    head(8, "Which end you read decides the answer")
    print("""    THE IDEA
    A composite holds a whole row of dimensions at once.  Two of them
    matter when you want to know what a number IS:

      the LEADING term -- the largest dimension with a NONZERO
      coefficient.  This decides everything: positive means the number
      diverges, zero means it is finite, negative means it vanishes.

      the DEEPEST term -- the smallest dimension present.  This is
      usually the tail of a series, and often just rounding noise.
      It tells you nothing about convergence.

    Two traps, and both have caught this library in real code:

      1.  taking min() instead of max() -- reading the tail and calling
          it the answer.
      2.  taking max() over the KEYS rather than over the nonzero ones.
          A key can be stored holding 0.0, and a stored zero is not a
          term.  (1-cos h)/h^2 keeps a key at +2 with coefficient 0.0,
          so plain max() calls a convergent expression divergent.

    WORKED EXAMPLE""")
    h = ZERO
    v = (1 - cos(h)) / (h * h)
    nz = [d for d, c in v.c.items() if c != 0.0]
    print(f"\n      (1-cos h)/h^2 = {str(v)[:46]} ...")
    print(f"\n        all keys        : {sorted(v.c, reverse=True)[:5]} ...")
    print(f"        max(keys)       : {max(v.c)}   -> would say 'diverges'")
    print(f"        coefficient there: {v.coeff(max(v.c))!r}   <- it is ZERO")
    print(f"        max(nonzero)    : {max(nz)}   -> finite, value {v.st()!r}")
    print(f"        min(keys)       : {min(v.c)}  <- series tail, noise")
    print(f"\n      limit() agrees with the second reading:"
          f" {float(limit(lambda t: (1-cos(t))/(t*t), 0.0))!r}")

    def round_():
        expr = ask_word("try", "sin(h)/h**2",
                        ["sin(h)/h", "sin(h)/h**2", "h/sin(h)", "sin(h)*h",
                         "(1-cos(h))/h**2"])
        v = {"sin(h)/h": sin(h) / h, "sin(h)/h**2": sin(h) / (h * h),
             "h/sin(h)": h / sin(h), "sin(h)*h": sin(h) * h,
             "(1-cos(h))/h**2": (1 - cos(h)) / (h * h)}[expr]
        nz = [d for d, c in v.c.items() if c != 0.0]
        print(f"\n      all keys      : max={max(v.c)}  min={min(v.c)}")
        print(f"      NONZERO keys  : max={max(nz)}  <- the leading term")
        if max(v.c) != max(nz):
            print(f"      key {max(v.c)} holds coefficient "
                  f"{v.coeff(max(v.c))!r} -- a stored zero, not a term")
        print(f"      min {min(v.c)} is the series tail, not mathematics\n")
    practice(round_)


# ------------------------------------------------------------------- 9
def lesson9():
    head(9, "What this is NOT: dual numbers and truncated jets")
    print("""    THE IDEA
    The fair objection to everything above is that it looks like
    forward-mode automatic differentiation in nicer notation.  Two
    things separate it, and both are visible.

    DUAL NUMBERS define eps^2 = 0.  That gives you first derivatives
    and stops.  Everything past dimension -1 is thrown away.

    A TRUNCATED JET of order N keeps dimensions 0 down to -N, which
    fixes that -- lessons 1 to 6 are exactly a jet.  But a jet's slots
    only run DOWNWARDS.  There is no slot for a positive dimension, so
    1/h is not a number it can hold at all, and 'divide by the thing
    that went to zero' has to be a special case outside the arithmetic.

    Here dimensions run both ways, so the indeterminate forms are
    ordinary results of ordinary convolution.

    WORKED EXAMPLE""")
    x = nudge(2)
    cube = x ** 3
    dual = Composite({k: c for k, c in cube.c.items() if k >= -1})
    print(f"\n      (2+h)**3 here    = {cube}")
    print(f"      as a dual number = {dual}")
    print(f"        f'(2)  = {cube.coeff(-1)!r}   kept by both")
    print(f"        f''(2) = {2*cube.coeff(-2)!r}   kept here, GONE in a dual")
    print(f"\n      1/h    = {R(1)/ZERO}     dimension +1")
    print(f"      1/h^2  = {R(1)/(ZERO**2)}     dimension +2")
    print("        a jet has no slot for either\n")
    for nm, v in (("ZERO/ZERO", ZERO / ZERO), ("ZERO*INF ", ZERO * INF),
                  ("INF-INF  ", INF - INF), ("1/ZERO   ", R(1) / ZERO)):
        print(f"      {nm} = {v}")
    print("""
      Read ZERO*INF = |1|_0 carefully: it does NOT say 0 times infinity
      is 1.  ZERO is a FIRST-ORDER zero and INF a first-order infinite,
      so their orders cancel and what survives is the product of the
      coefficients, which both happen to be 1.  Scale either one and the
      answer scales with it:""")
    print(f"        (2*ZERO)*INF     = {(2*ZERO)*INF}")
    print(f"        (2*ZERO)*(3*INF) = {(2*ZERO)*(3*INF)}")
    print(f"        ZERO*(ZERO*INF)  = {ZERO*(ZERO*INF)}   "
          f"<- orders no longer cancel")
    print("""
      Every one of those is an error in float arithmetic and undefined
      in a jet.  None is a special case here.

      One of them looks like it breaks lesson 8.  INF - INF prints as
      |0|_1, and lesson 8 said a stored zero is not a term -- so is this
      just zero?  No, and the difference matters.  Lesson 8 was about a
      number that has OTHER terms, where a key holding 0.0 must not be
      mistaken for the leading one.  Here there are no other terms: the
      whole number is a zero, and the dimension it sits at is the only
      thing distinguishing it from every other zero.  INF - INF vanishes
      at the infinite scale; ZERO - ZERO vanishes at the infinitesimal
      one, and they are not interchangeable.""")
    print(f"      INF - INF   = {INF-INF}     a zero at dimension +1")
    print(f"      ZERO - ZERO = {ZERO-ZERO}    a zero at dimension -1")

    def round_():
        q = ask_int("build 1/h**q, pick q", 3, 1, 8)
        v = R(1) / (ZERO ** q)
        print(f"\n      1/h**{q} = {v}   -- dimension +{q}, no jet slot for it\n")
    practice(round_)


# ------------------------------------------------------------------ 10
def lesson10():
    head(10, "Where it stops: the cases that do NOT work")
    print("""    THE IDEA
    Everything so far worked because the function was smooth and was
    built out of arithmetic the composite understands.  A tutor that
    only shows the wins teaches you to trust it too far, so here are
    the five ways it breaks -- try them yourself in the REPL.

    1.  nudge(0) is NOT 0 + h.
    2.  abs() and any plain-float exit throw the nudge away.
    3.  a Python 'if' picks one branch, and the answer is that branch's.
    4.  a value too small for float64 underflows, and R1 revives it with
        the wrong magnitude.
    5.  an accumulator started at 0 is not empty.

    WORKED EXAMPLE 1 -- the zero that is not zero""")
    print(f"\n      R(0)            = {R(0)}    R1: a zero OPERAND converts")
    print(f"      nudge(0)        = {R(0)+ZERO}    so this is 2h, not h")
    print(f"      nudge(0)**2     = {(R(0)+ZERO)**2}    = (2h)^2 = 4h^2, not h^2")
    print(f"      ZERO**2         = {ZERO**2}    <- what you probably meant")
    print("""
      Nothing is wrong here -- R(0) really is |1|_-1 by the zero rules --
      but 'nudge(0)' reads like '0 + h' and is not.  Use ZERO directly
      when you want a bare first-order zero.

    WORKED EXAMPLE 2 -- abs() drops everything""")
    x = nudge(3)
    print(f"\n      x        = {x}")
    print(f"      abs(x)   = {abs(x)!r}   <- a plain float.  The nudge is gone.")
    print("""
      Composite defines __float__, so every function that wants a number
      gets one silently: math.sin, abs, round, comparisons.  There is no
      error -- just an answer with no derivative in it.  If a result
      comes back as a bare float, something exited the composite.

    WORKED EXAMPLE 3 -- a Python branch picks ONE side""")
    def pw(t):
        return t * t if t > 0 else -(t * t)
    z = R(0) + ZERO
    print(f"\n      def pw(t): return t*t if t > 0 else -(t*t)")
    print(f"      at x = 0.5 : {pw(nudge(0.5))}")
    print(f"      at nudge(0): {pw(z)}          because (nudge(0) > 0) is {z > 0}")
    print(f"      and note the 4 -- that is example 1 firing inside this one:")
    print(f"      nudge(0) is {z}, so squaring it gives (2h)^2, not h^2.")
    print("""
      The comparison resolved to a single True, so one branch ran and the
      other never existed.  |x| at 0 has no derivative, and no arithmetic
      can invent one -- but nothing warned you either.

    WORKED EXAMPLE 4 -- a value too small for float64, resurrected""")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = exp(R(-1) / (nudge(1e-3) ** 2))
    print(f"\n      f(x) = exp(-1/x^2) at x = 1e-3.")
    print(f"        -1/x^2          = {-1/(1e-3)**2:g}")
    print(f"        math.exp(-1e6)  = {math.exp(-1e6)!r}   <- UNDERFLOWS to zero")
    print(f"        R1 then converts that bare zero to {R(0.0)}")
    print(f"\n      so the composite reports:")
    for k in sorted(v.c, reverse=True)[:4]:
        print(f"        dimension {k:>3} : {v.coeff(k):.6g}")
    print(f"""
      Those look like derivatives exploding.  They are not.  The true
      derivatives of exp(-1/x^2) here are all of order e^-1000000, which
      in float64 is exactly 0 -- they VANISH, they do not grow.  What is
      on screen is the derivative chain of the EXPONENT:

        d/dx (-1/x^2) = 2/x^3 = {2/(1e-3)**3:g}    <- the dimension -2 row

      multiplied by a value that underflowed to zero and was then brought
      back as a unit zero by R1 instead of annihilating the product.  The
      |1|_-1 at the top of that table IS the underflow.

      This is not a fact about calculus -- exp(-1/x^2) being non-analytic
      is true everywhere.  It is a fact about REPRESENTATION, and it is a
      failure only this library can have: float64 lost the value, and the
      zero rules gave it back with the wrong magnitude.  When a composite
      comes back with enormous coefficients for a function you know is
      tiny, suspect an underflow that R1 revived.

      One real limitation does survive underneath all that, and it is a
      fact about calculus rather than floats: at x = 0 EXACTLY, every
      derivative of exp(-1/x^2) is zero, so its Taylor series there is
      identically zero while the function is not.  No number of extra
      orders repairs that, because a local series only determines an
      ANALYTIC function.  The series is correct; it simply does not
      describe f away from the point.

    WORKED EXAMPLE 5 -- the accumulator that is not empty""")
    print("\n      The classic way to lose a derivative is to start a sum at 0.")
    print("      Here is the whole example -- x = nudge(2), summing x + x*x:")
    print("\n        x       = " + str(nudge(2)))
    print("        x*x     = " + str(nudge(2) * nudge(2)))
    print("\n        acc = 0                 |  acc = Composite({})")
    print("        for t in (x, x*x):      |  for t in (x, x*x):")
    print("            acc = acc + t       |      acc = acc + t")
    print("\n      A bare Python 0 meeting a composite CONVERTS (R1), so the")
    print("      accumulator is |1|_-1 rather than nothing, and that unit")
    print("      infinitesimal is carried into every term that follows.")
    x2 = nudge(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad = 0
        for t in (x2, x2 * x2):
            bad = bad + t
        first_warning = (str(caught[0].message).split(". ")[0] + "."
                         if caught else "")
    good = Composite({})
    for t in (x2, x2 * x2):
        good = good + t
    print(f"\n        acc = 0          -> {bad}")
    print(f"        acc = Composite() -> {good}")
    print(f"        f' differs by     : {bad.coeff(-1) - good.coeff(-1)!r}")
    print(f"\n      f'(2) should be 5: d/dx(x + x^2) at 2 is 1 + 4.  The bare 0")
    print("      added a spurious 1.  The library does warn -- its first line is:")
    print(f"        \"{first_warning}\"" if first_warning else
          "        (no warning captured)")
    print("      Start an empty sum with Composite({}), which is NOTHING --")
    print("      not zero.  R6: an unstarted sum holds nothing.")
    print("""
    AND THE ONE THAT LOOKS LIKE A FAILURE AND IS NOT""")
    print(f"\n      sqrt(ZERO) = {sqrt(ZERO)}     a HALF dimension")
    print("""
      The square root of a first-order zero is a zero of order one half.
      That is the clearest sign that the dimension index is not a counter
      for "how many derivatives" -- it is a scale, and scales can be
      halved.  A jet has no slot for a fractional order, and neither does
      a dual number; here it is just another dimension, and sqrt(sqrt(ZERO))
      would be order one quarter.""")

    def round_():
        # eval() on user input is acceptable HERE -- a local tutor the
        # learner runs on their own machine, where they already have a REPL.
        # Do not lift this into anything that takes input from elsewhere.
        print("      type a Python expression in x, e.g.  abs(x)  or  x*x")
        print("      (x is nudge(2); the question is whether the nudge survives)")
        src = _raw("expression", "x*x")or "x*x"
        x = nudge(2)
        try:
            r = eval(src, {"__builtins__": {"abs": abs, "round": round},
                           "x": x, "sin": sin, "cos": cos, "exp": exp,
                           "ln": ln, "sqrt": sqrt, "R": R, "ZERO": ZERO})
        except Exception as e:
            print(f"      raised {type(e).__name__}: {str(e)[:50]}\n"); return
        if isinstance(r, Composite):
            d1 = r.coeff(-1)
            print(f"      -> {r}")
            print(f"      nudge SURVIVED: f'(2) = {d1!r}\n" if d1 else
                  f"      composite, but the h term is 0 -- no first derivative\n")
        else:
            print(f"      -> {r!r}  ({type(r).__name__})")
            print("      the nudge was DROPPED -- this exited the composite\n")
    practice(round_)


LESSONS = [lesson1, lesson2, lesson3, lesson4, lesson5,
           lesson6, lesson7, lesson8, lesson9,
           lesson10]

if __name__ == "__main__":
    print(__doc__)
    try:
        todo = ([LESSONS[int(sys.argv[1]) - 1]] if len(sys.argv) > 1
                else LESSONS)
        for fn in todo:
            fn()
    except (KeyboardInterrupt, Quit):
        print("\n  bye\n")
