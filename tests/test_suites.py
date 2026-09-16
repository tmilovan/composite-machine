"""Run the standalone suites under pytest.

Most suites in this directory are scripts rather than pytest modules: they
execute on import and call sys.exit(), so pytest cannot collect them directly
(see conftest.py).  Each is run here as a subprocess and judged by its exit
code, which every suite sets from its own pass/fail tally.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SUITES = [
    "tests/test_standalone.py",
    "tests/test_limits.py",
    "tests/test_stress.py",
    "tests/test_stress_hard_edge.py",
    "tests/test_integration_comprehensive.py",
    "tests/test_composite_vector.py",
    "tests/test_dimension_scales.py",
    "tests/test_multivar_extended.py",
    "tests/turing_completeness/test_turing_completeness.py",
    "tests/turing_completeness/test_indicator_polinomial_step_test.py",
    "tests/turing_completeness/test_self_hosted_execution.py",
]


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: Path(s).stem)
def test_suite(suite):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / suite)],
        cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "{} failed (exit {})\n\n{}{}".format(
                suite, proc.returncode, proc.stdout[-4000:], proc.stderr[-2000:]
            )
        )
