"""The isolated-versus-degradation rule of tests/series_check.py, on synthetic
series: want is all ones, got perturbs chosen grades by a chosen relative error."""
import pytest

from tests.series_check import check_series, IsolatedCoefficientWarning

TOL = 1e-13


def _series(errors, n=30):
    want = {-k: 1.0 for k in range(n)}
    got = {-k: 1.0 + errors.get(k, 0.0) for k in range(n)}
    return got, want


def test_clean_series_passes_without_warning(recwarn):
    check_series("clean", *_series({}), TOL)
    assert not [w for w in recwarn if w.category is IsolatedCoefficientWarning]


def test_isolated_grade_passes_and_warns():
    with pytest.warns(IsolatedCoefficientWarning):
        check_series("isolated", *_series({24: 6.6e-13}), TOL)


def test_consecutive_grades_fail():
    with pytest.raises(AssertionError, match="consecutive"):
        check_series("consecutive", *_series({24: 5e-13, 25: 5e-13}), TOL)


def test_rising_band_median_fails():
    # every other grade from 20 up: no two adjacent, but the band median rises
    errs = {k: 1e-12 for k in range(20, 30, 2)}
    with pytest.raises(AssertionError, match="band median"):
        check_series("degrading band", *_series(errs), TOL)


def test_isolated_but_over_cap_fails():
    with pytest.raises(AssertionError, match="cap"):
        check_series("wrong coefficient", *_series({24: 0.5}), TOL)
