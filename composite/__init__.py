"""Composite Calculus — Provenance-Preserving Arithmetic"""
from composite.composite_lib import *

# Resummation is a LAYER ON TOP of the arithmetic, not part of it: it needs
# integrate() and polynomial multiply/divide, and nothing in composite_lib
# needs it back.  Kept in its own module for that reason, and imported after
# composite_lib so the one-way dependency stays one-way.
from composite.resummation import (
    borel,              # b_n = c_n / n!
    pade,               # [L/M] by extended Euclid on convolve/deconvolve
    resum,              # Borel-Pade sum of a divergent asymptotic series
    resum_lateral,     # Borel sum with the contour swung off a blocked ray
    resum_median,      # both laterals: the value, and the flat term between them
    borel_singularity,  # where the Borel transform blows up, and its residue
    flat_term,          # size of the exponentially small ambiguity
    poles,              # roots of the Pade denominator
    classify_singularity,  # pole, branch point, or not yet resolved -- and why
    SingularityKind,
    poly, polydiv, degree, dpoly,   # the polynomial helpers underneath
)

from composite.transseries import (
    Transseries,        # a sparse map sector -> Composite; sector n is exp(-n/h)
    from_series,        # build one FROM a divergent series: the problem does it
    action_from_growth, # the action from the coefficient ratio, stride-aware
    resum_sector,       # the bridge: a sector's series -> a number
    flat, sector, ts_exp, ts_ln, ts_d, ts_st,
    is_infinitesimal, is_infinite,
)

# Forensics is a layer on top too, and a consumer of the arithmetic rather than
# part of it: it seeds a point with an infinitesimal, reads back the exact
# derivative for the condition number, and watches the operators for lost
# significance.  Nothing in composite_lib needs it back.
#
# `F` (the elementary functions a formula is written against) and the renderers
# `table`/`report` stay on the module rather than coming into the top-level
# namespace, where names that short would collide with a caller's own.  Reach
# them as `from composite import forensics` / `forensics.F`.
from composite import forensics
from composite.forensics import (
    audit,              # one formula at one point -> value, f'(x), kappa, verdict
    compare,            # several spellings of one function, side by side
    Audit, Finding,
    STABLE, ILL_CONDITIONED, UNSTABLE, DERIVATIVE_LOST, REFUSED,
)

# Singularity analysis sits on top of both: it needs the composite's series
# arithmetic to build and divide the series, and resummation's Pade to
# cross-check the differential approximant.  Given coefficients it returns
# where the series stops converging and what it does there -- which is the
# blow-up time and rate of an ODE, the critical point and exponent of a
# lattice model, and the asymptotic growth of a counting sequence, all at once.
from composite import singularity as singularity_analysis
from composite.singularity import (
    analyse,                # coefficients -> (location, exponent, confidence)
    blowup,                 # y' = f(y) -> when it blows up, and how fast
    series_solve,           # Taylor coefficients of y' = f(y), by convolution
    coefficient_asymptotics,  # a_n ~ C n^(-beta-1) z0^-n
    radius,                 # radius of convergence from coefficient growth
    Singularity,
)
