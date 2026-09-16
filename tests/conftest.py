"""pytest configuration for the composite-machine test suite."""

import os
import sys

# Resolve `import composite` to this checkout rather than whatever is installed,
# so the tests always exercise the working copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Most suites here are standalone scripts: they run their tests at import time
# and finish with sys.exit(), which pytest cannot collect.  test_suites.py runs
# each of them as a subprocess instead and judges it by its exit code.
collect_ignore = [
    "test_standalone.py",
    "test_limits.py",
    "test_stress.py",
    "test_stress_hard_edge.py",
    "test_integration_comprehensive.py",
    "test_composite_vector.py",
    "test_multivar_extended.py",
]
collect_ignore_glob = ["turing_completeness/*.py"]
