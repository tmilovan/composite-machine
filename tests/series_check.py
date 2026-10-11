"""Judge a series grade by grade, and tell an isolated coefficient from degradation.

Every grade is measured by its OWN relative error -- a coefficient is a real
number at its own grade, and its digits are right or they are not.  A grade
whose value is formed by a large cancellation keeps fewer digits than its
neighbours (ln(2 + sin x) at 1.1: grade 24 is 7.3e-16, formed from terms near
1e-12, and is good to 6.6e-13 while 23 and 25 are good to 3e-15).  That is a
property of that one coefficient and nothing after it inherits it.
Degradation is different: error that grows with order lifts whole runs of
grades (the old ln: 2e-12, 7e-11, 3e-9 as band medians).

  pass            every grade <= tol
  pass, warn      grades over tol, each ISOLATED (both neighbours <= tol) and
                  each <= cap -- printed with value and error, and warned
  fail            two consecutive grades over tol, a 10-grade band whose
                  median is over tol, or any grade over cap (a single wrong
                  coefficient looks isolated too; cap is the derivative
                  battery's 1e-9)
"""
import statistics
import warnings


class IsolatedCoefficientWarning(UserWarning):
    pass


def check_series(label, got, want, tol, cap=1e-9):
    order = lambda g: ((-float(g[0]), float(g[1])) if isinstance(g, tuple)
                       else (-float(g), 0.0))
    grades = sorted(want, key=order)                         # order 0 first
    errs = []
    for g in grades:
        w, v = want[g], got.get(g, 0.0)
        errs.append(abs(v - w) / abs(w) if w != 0.0 else abs(v))
    over = [i for i, e in enumerate(errs) if e > tol]
    consecutive = [i for i in over if i + 1 in over]
    bands = [(i, statistics.median(errs[i:i + 10])) for i in range(0, len(errs), 10)]
    bad_bands = [(i, m) for i, m in bands if m > tol]
    too_big = [i for i in over if errs[i] > cap]

    print(f"\n  {label}: {len(grades)} grades, worst {max(errs):.1e}  tol {tol:.0e}  cap {cap:.0e}")
    print("    band medians " + "  ".join(f"{i}-{min(i + 9, len(errs) - 1)}:{m:.0e}" for i, m in bands))
    for i in over:
        print(f"    grade {grades[i]}: got {got.get(grades[i], 0.0)!r}  want {want[grades[i]]!r}  rel err {errs[i]:.1e}")

    assert not consecutive, f"{label}: consecutive grades over tol at {[grades[i] for i in consecutive]}"
    assert not bad_bands, f"{label}: band median over tol: {bad_bands}"
    assert not too_big, f"{label}: grades over cap {cap:.0e}: {[grades[i] for i in too_big]}"
    if over:
        warnings.warn(f"{label}: isolated coefficient(s) at grade(s) "
                      f"{[grades[i] for i in over]}, rel err "
                      f"{', '.join(f'{errs[i]:.1e}' for i in over)} (neighbours within {tol:.0e})",
                      IsolatedCoefficientWarning)
