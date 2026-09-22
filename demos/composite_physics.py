#!/usr/bin/env python3
"""Physics with composites: expansions, divergences and singular points from one evaluation.

Five textbook problems, each chosen because a composite answers it in a single
evaluation where ordinary numerics need hand expansion, regularisation, or give up:

  1. DIRAC HYDROGEN.  Set Z*alpha to the infinitesimal and evaluate the exact Dirac
     formula once.  Each grade is one layer of atomic physics: rest energy,
     Bohr/Schrodinger levels, fine structure, then alpha^6 and alpha^8.
  2. CRITICAL COUPLING.  At Z*alpha = 1 the 1s energy sqrt(1 - (Z alpha)^2) has a
     branch point.  The composite returns it as a half grade, and gives exact
     derivatives arbitrarily close to it, where a finite difference drifts with
     its step.
  3. ZERO-POINT MODE SUMS.  sum n e^{-n h} and sum n^3 e^{-n h} with the cutoff h an
     infinitesimal.  The divergent part and the finite part (zeta(-1) = -1/12,
     zeta(-3) = 1/120, the Casimir coefficients) come out on separate grades.
  4. SCHWARZSCHILD.  The acceleration needed to hover is the derivative of the clock
     rate, exact in the strong field; the clock rate at the horizon vanishes as a
     half grade; along a radial infall the curvature near r = 0 comes out as a
     graded infinity with its exact coefficient.
  5. RELATIVISTIC KINEMATICS.  A massive particle given infinite energy: its speed
     falls short of c by an infinitesimal, gamma is a first-order infinity, the
     rapidity sits on the log axis.  A massless particle has no rest frame: its mass
     is absent, and gamma = E/m is refused.

Units: c = G = 1 (and m = 1, M = 1 where stated).  Every check prints got, want,
error and the fixed tolerance it is held to.

One rule matters throughout: a quantity that is exactly absent is written
Composite({}) (nothing), not R(0).  A written zero is an expressed zero -- an
infinitesimal -- and converts.  Section 1 shows what that difference does.
"""
import math
import warnings

from composite.composite_lib import R, ZERO, Composite, exp, ln, sqrt, cos, sin

warnings.filterwarnings("ignore")        # R1 conversions are expected where shown

NOTHING = Composite({})
h = ZERO
_results = []


def coeff(c, grade):
    return c.coeffs_dict().get(grade, 0.0)


def check(label, got, want, tol):
    err = abs(got - want)
    ok = err <= tol
    _results.append(ok)
    print(f"  {'OK ' if ok else 'BAD'} {label:46s} got {got: .15g}  want {want: .15g}  "
          f"err {err:.1e}  tol {tol:.0e}")


def section(title):
    print("\n" + title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# 1. Dirac hydrogen
# ---------------------------------------------------------------------------
def dirac(za, n, j, written_zero=False):
    """E / mc^2 for the Dirac-Coulomb level (n, j) at coupling za = Z*alpha."""
    k = j + 0.5
    shift = n - k
    if shift == 0 and not written_zero:
        base = NOTHING                    # the term is absent for these states
    else:
        base = R(shift)
    return (R(1) + (za / (base + sqrt(R(k*k) - za*za)))**2) ** (-0.5)


def dirac_float(za, n, j):
    k = j + 0.5
    return (1 + (za / (n - k + math.sqrt(k*k - za*za)))**2) ** -0.5


def demo_dirac():
    section("1. Dirac hydrogen: one evaluation, every layer of the fine structure")
    alpha, mc2 = 1/137.035999084, 510998.95000     # fine-structure constant, electron rest energy in eV
    for n, j in ((1, 0.5), (2, 0.5), (2, 1.5), (3, 2.5)):
        k = j + 0.5
        E = dirac(h, n, j)
        print(f" n={n} j={j}:  E/mc^2 = {str(E)[:78]}")
        check("grade  0: rest energy", coeff(E, 0), 1.0, 1e-15)
        check("grade -2: Bohr -1/(2n^2)", coeff(E, -2), -1/(2*n*n), 1e-15)
        check("grade -4: fine structure -(n/k - 3/4)/(2n^4)", coeff(E, -4), -(n/k - 0.75)/(2*n**4), 1e-15)
        check("odd grades absent (alpha^3)", coeff(E, -3), 0.0, 1e-15)
        series = sum(coeff(E, -g) * alpha**g for g in range(2, 11, 2))
        exact = dirac_float(alpha, n, j) - 1
        check("binding energy at alpha = 1/137.036, eV", series*mc2, exact*mc2, 1e-9)
    E2p1, E2p3 = (sum(coeff(dirac(h, 2, jj), -g) * alpha**g for g in range(2, 11, 2)) for jj in (0.5, 1.5))
    split_GHz = (E2p3 - E2p1) * mc2 / 4.135667696e-15 / 1e9
    print(f" 2p fine-structure splitting: {split_GHz:.4f} GHz  (Dirac; measured 10.969 GHz includes QED)")
    wrong = dirac(h, 1, 0.5, written_zero=True)
    print(f" the same 1s level with the absent term WRITTEN as R(0):  alpha^3 coefficient = "
          f"{coeff(wrong, -3):+.3f},  alpha^4 = {coeff(wrong, -4):+.4f}  (want 0 and -0.125)")
    print("   a written zero is an infinitesimal; here the term does not exist, so it must be absent.")


# ---------------------------------------------------------------------------
# 2. Critical coupling: a branch point
# ---------------------------------------------------------------------------
def demo_critical():
    section("2. Critical coupling Z*alpha -> 1: the 1s energy has a square-root branch point")
    E = sqrt(R(1) - (R(1) - h)**2)                     # at Z*alpha = 1 - h
    print(f" E/mc^2 at Z*alpha = 1 - h:  {str(E)[:78]}")
    check("half grade: coefficient of h^(1/2) = sqrt2", coeff(E, -0.5), math.sqrt(2), 1e-14)
    check("next: coefficient of h^(3/2) = -sqrt2/4", coeff(E, -1.5), -math.sqrt(2)/4, 1e-14)
    print(" dE/d(Z alpha) approaching the branch point (exact: -x/sqrt(1-x^2)):")
    f = lambda x: math.sqrt(max(0.0, 1 - x*x))
    for d in (1e-4, 1e-8, 1e-12):
        x = 1 - d
        Ec = sqrt(R(1) - (R(x) + h)**2)
        want = -x / math.sqrt(1 - x*x)
        check(f"composite, 1 - x = {d:g}", coeff(Ec, -1), want, 1e-9*abs(want))
        fd = (f(x + 1e-8) - f(x)) / 1e-8
        print(f"      finite difference (step 1e-8) at 1 - x = {d:g}: {fd: .6g}   (relative error {abs(fd-want)/abs(want):.1e})")


# ---------------------------------------------------------------------------
# 3. Zero-point mode sums
# ---------------------------------------------------------------------------
def demo_casimir():
    section("3. Zero-point mode sums: divergent and finite parts on separate grades")
    q = exp(R(-1) * h)
    S1 = q / (R(1) - q)**2                                      # sum n e^{-n h}
    S3 = q * (R(1) + R(4)*q + q*q) / (R(1) - q)**4              # sum n^3 e^{-n h}
    print(f" sum n e^(-nh)   = {str(S1)[:78]}")
    check("divergent part 1/h^2", coeff(S1, 2), 1.0, 1e-14)
    check("finite part = zeta(-1) = -1/12", coeff(S1, 0), -1/12, 1e-14)
    check("next: h^2/240", coeff(S1, -2), 1/240, 1e-14)
    print(f" sum n^3 e^(-nh) = {str(S3)[:78]}")
    check("divergent part 6/h^4", coeff(S3, 4), 6.0, 1e-13)
    check("finite part = zeta(-3) = 1/120", coeff(S3, 0), 1/120, 1e-14)
    check("next: -h^2/504", coeff(S3, -2), -1/504, 1e-13)
    print(" the finite parts are the Casimir coefficients: 1+1D plates give E = (pi/2a) * (-1/12) = -pi/(24a);")
    print(" the divergent parts are kept, not subtracted, on grades 2 and 4.")


# ---------------------------------------------------------------------------
# 4. Schwarzschild
# ---------------------------------------------------------------------------
def demo_schwarzschild():
    section("4. Schwarzschild (M = 1): hovering, the horizon, and curvature near r = 0")
    print(" acceleration needed to hover = d(clock rate)/dr, clock rate = sqrt(1 - 2M/r):")
    for r0 in (3.0, 10.0, 1000.0):
        rate = sqrt(R(1) - R(2) / (R(r0) + h))
        want = (1/r0**2) / math.sqrt(1 - 2/r0)
        check(f"r = {r0:g}", coeff(rate, -1), want, 1e-13*want)
    rate_h = sqrt(R(1) - R(2) / (R(2) + h))
    print(f" clock rate at the horizon, r = 2M + h: {str(rate_h)[:60]}")
    check("vanishes as h^(1/2), coefficient 1/sqrt2", coeff(rate_h, -0.5), 1/math.sqrt(2), 1e-14)
    print(" radial infall from rest at r = 2M: r = 1 - cos(d), proper time d - sin(d), near r = 0 (d = h):")
    r = R(1) - cos(h)
    tau = h - sin(h)
    K = R(48) / r**6                                            # Kretschmann curvature invariant
    check("r ~ h^2/2", coeff(r, -2), 0.5, 1e-15)
    check("proper time to r = 0 ~ h^3/6", coeff(tau, -3), 1/6, 1e-15)
    check("curvature K = 48/r^6 ~ 3072/h^12", coeff(K, 12), 3072.0, 1e-9)
    check("tidal stretching 2M/r^3 ~ 16/h^6", coeff(R(2)/r**3, 6), 16.0, 1e-11)


# ---------------------------------------------------------------------------
# 5. Relativistic kinematics
# ---------------------------------------------------------------------------
def demo_kinematics():
    section("5. Relativistic kinematics: infinite energy, and a particle with no rest frame")
    m, E = R(1), R(1) / h                                       # mass 1, energy |1|_1
    v = sqrt(R(1) - (m*m)/(E*E))
    gamma = E / m
    rapidity = ln(E/m + sqrt((E/m)**2 - R(1)))
    print(f" speed    v/c   = {str(v)[:60]}")
    check("falls short of c by (m/E)^2/2", coeff(R(1) - v, -2), 0.5, 1e-15)
    print(f" gamma          = {gamma}")
    check("gamma is a first-order infinity", coeff(gamma, 1), 1.0, 1e-15)
    print(f" rapidity       = {str(rapidity)[:60]}")
    check("rapidity = ln(1/h) + ln 2: log-axis term", rapidity.coeffs_dict().get((0, 1), 0.0), 1.0, 1e-15)
    check("rapidity finite part ln 2", rapidity.coeffs_dict().get((0, 0), coeff(rapidity, 0)), math.log(2), 1e-14)
    try:
        R(1) / NOTHING
        print("  BAD massless particle: gamma = E/m should be refused")
        _results.append(False)
    except Exception as e:
        print(f"  OK  massless particle (mass absent): gamma = E/m refused -- {type(e).__name__}: no rest frame")
        _results.append(True)


def main():
    demo_dirac()
    demo_critical()
    demo_casimir()
    demo_schwarzschild()
    demo_kinematics()
    print(f"\nRESULTS: {sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
