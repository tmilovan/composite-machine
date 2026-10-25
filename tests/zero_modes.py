"""Run a block under one cancellation rule (Zero Rules v2, sections 1b and 1c).

    from tests.zero_modes import RULES, cancellation

    with cancellation("mechanical"):
        assert (R(6) - R(6)).coeffs_dict() == {-1: 6.0}

The rule is restored on exit, also when the block raises, so a failing check
cannot leak its rule into the next test.  The standalone suites run as
subprocesses with the repository root on PYTHONPATH, so they import this the
same way pytest does.
"""

import contextlib

import composite.composite_lib as cl

RULES = ("convert_and_keep", "natural", "mechanical")
# "mirrored" and "to_nothing" are experimental: cancellation() accepts them,
# but the tests parametrized over RULES do not pin values for them yet.
ACCEPTED = RULES + ("mirrored", "to_nothing")


@contextlib.contextmanager
def cancellation(rule):
    if rule not in ACCEPTED:
        raise ValueError(f"rule must be one of {ACCEPTED}, not {rule!r}")
    old = cl.CANCELLATION
    cl.CANCELLATION = rule
    try:
        yield rule
    finally:
        cl.CANCELLATION = old
