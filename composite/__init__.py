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
