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
    borel_singularity,  # where the Borel transform blows up, and its residue
    flat_term,          # size of the exponentially small ambiguity
    poles,              # roots of the Pade denominator
    poly, polydiv, degree, dpoly,   # the polynomial helpers underneath
)
