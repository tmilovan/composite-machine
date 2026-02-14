import numpy as np
from composite.backends import SparseDenseBackend, DictBackend

def test_backends_agree():
    """Both backends must produce identical results."""
    sd = SparseDenseBackend(gap_threshold=64)
    db = DictBackend()

    # Extreme sparsity: terms 10M apart
    a_sd = sd.create_from_terms(
        np.array([-10_000_000, 0, 7]),
        np.array([0.5, 2.0, 1.0])
    )
    a_db = db.create_from_terms(
        np.array([-10_000_000, 0, 7]),
        np.array([0.5, 2.0, 1.0])
    )

    b_sd = sd.create_from_terms(
        np.array([0, 3]),
        np.array([4.0, 1.0])
    )
    b_db = db.create_from_terms(
        np.array([0, 3]),
        np.array([4.0, 1.0])
    )

    # Multiply
    result_sd = sd.convolve(a_sd, b_sd)
    result_db = db.convolve(a_db, b_db)

    # Compare
    sd_dims, sd_vals = sd.to_arrays(result_sd)
    db_dims, db_vals = db.to_arrays(result_db)

    assert np.array_equal(sd_dims, db_dims), \
        f"Dims differ: {sd_dims} vs {db_dims}"
    assert np.allclose(sd_vals, db_vals), \
        f"Vals differ: {sd_vals} vs {db_vals}"

    # Verify specific terms
    # -10M * 0 = -10M:  0.5 * 4.0 = 2.0
    # -10M * 3 = -9999997:  0.5 * 1.0 = 0.5
    # 0 * 0 = 0:  2.0 * 4.0 = 8.0
    # 0 * 3 = 3:  2.0 * 1.0 = 2.0
    # 7 * 0 = 7:  1.0 * 4.0 = 4.0
    # 7 * 3 = 10:  1.0 * 1.0 = 1.0
    assert sd.read_dim(result_sd, -10_000_000) == 2.0
    assert sd.read_dim(result_sd, -9_999_997) == 0.5
    assert sd.read_dim(result_sd, 0) == 8.0
    assert sd.read_dim(result_sd, 3) == 2.0
    assert sd.read_dim(result_sd, 7) == 4.0
    assert sd.read_dim(result_sd, 10) == 1.0

    # Must NOT have any terms between -9,999,997 and 0
    all_dims = sd.active_dims(result_sd)
    between = all_dims[(all_dims > -9_999_997) & (all_dims < 0)]
    assert len(between) == 0, \
        f"Ghost dimensions found: {between}"

    print("PASS: both backends agree, no ghost dimensions")


def test_cluster_convolution_exact():
    """Verify partial convolution produces same result as full."""
    sd = SparseDenseBackend(gap_threshold=64)

    # Dense-ish case: all terms close together
    a = sd.create_from_terms(
        np.array([-3, -2, -1, 0, 1]),
        np.array([0.5, 0.1, 0.3, 2.0, 0.7])
    )
    b = sd.create_from_terms(
        np.array([0, 1, 2]),
        np.array([1.0, -0.5, 0.2])
    )

    result = sd.convolve(a, b)

    # Cross-check with DictBackend
    db = DictBackend()
    a_db = db.create_from_terms(
        np.array([-3, -2, -1, 0, 1]),
        np.array([0.5, 0.1, 0.3, 2.0, 0.7])
    )
    b_db = db.create_from_terms(
        np.array([0, 1, 2]),
        np.array([1.0, -0.5, 0.2])
    )
    expected = db.convolve(a_db, b_db)

    sd_dims, sd_vals = sd.to_arrays(result)
    ex_dims, ex_vals = db.to_arrays(expected)

    assert np.array_equal(sd_dims, ex_dims)
    assert np.allclose(sd_vals, ex_vals)
    print("PASS: clustered convolution matches brute-force")


if __name__ == "__main__":
    test_backends_agree()
    test_cluster_convolution_exact()
    print("All tests passed.")
