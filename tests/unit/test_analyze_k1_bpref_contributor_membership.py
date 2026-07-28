import numpy as np

from scripts.analyze_k1_bpref_contributor_membership import (
    compare_particle_membership,
    match_rotations,
    same_identity_set,
)


def _matrices(values):
    matrices = np.zeros((len(values), 3, 3), dtype=np.float32)
    for index, value in enumerate(values):
        matrices[index] = np.eye(3, dtype=np.float32)
        matrices[index, 0, 0] = value
    return matrices


def test_same_identity_set_is_order_independent_and_duplicate_strict():
    assert same_identity_set(np.array([3, 1, 2]), np.array([2, 3, 1]))
    assert not same_identity_set(np.array([1, 1, 2]), np.array([1, 2, 3]))
    assert not same_identity_set(np.array([1, 2]), np.array([1, 3]))


def test_match_rotations_uses_unique_geometry_with_declared_tolerance():
    relion = _matrices([1.0, 2.0, 3.0])
    recovar = _matrices([3.0, 1.0, 4.0])
    recovar[1, 0, 0] += np.float32(5.0e-7)
    matches = match_rotations(relion, recovar, tolerance=1.0e-6)
    assert matches.pairs.tolist() == [[0, 1], [2, 0]]
    assert matches.relion_unmatched.tolist() == [1]
    assert matches.recovar_unmatched.tolist() == [2]
    assert matches.relion_ambiguous == 0
    assert matches.recovar_ambiguous == 0


def test_compare_particle_separates_candidate_and_significance_membership():
    report, arrays = compare_particle_membership(
        relion_rotations=_matrices([1.0, 2.0, 3.0]),
        relion_positive=np.array([True, True, False]),
        recovar_rotations=_matrices([1.0, 2.0, 4.0]),
        recovar_positive=np.array([True, False, True]),
        recovar_posterior_mass=np.array([0.7, 0.01, 0.29]),
        recovar_reconstruction_mass=np.array([0.7, 0.0, 0.29]),
        recovar_max_sample_posterior=np.array([0.5, 0.004, 0.2]),
        recovar_reconstruction_threshold=0.005,
    )
    assert report["candidate_unique_match_count"] == 2
    assert report["relion_candidate_unmatched_count"] == 1
    assert report["recovar_candidate_unmatched_count"] == 1
    assert report["both_positive_matched_count"] == 1
    assert report["relion_positive_recovar_nonpositive_matched_count"] == 1
    assert report["recovar_positive_relion_nonpositive_matched_count"] == 0
    assert report["relion_positive_unmatched_candidate_count"] == 0
    assert report["recovar_positive_unmatched_candidate_count"] == 1
    assert not report["candidate_sets_exact_at_tolerance"]
    assert not report["positive_contributor_sets_exact_at_tolerance"]
    assert report["recovar_reconstruction_threshold"] == 0.005
    assert report["recovar_reconstruction_threshold_positive"]
    np.testing.assert_array_equal(
        arrays["recovar_preprune_mass_relion_positive_recovar_nonpositive"],
        np.array([0.01]),
    )
    np.testing.assert_array_equal(
        arrays["recovar_preprune_mass_recovar_positive_unmatched"],
        np.array([0.29]),
    )


def test_zero_reconstruction_threshold_is_explicit_and_ratio_is_undefined():
    report, arrays = compare_particle_membership(
        relion_rotations=_matrices([1.0]),
        relion_positive=np.array([True]),
        recovar_rotations=_matrices([1.0]),
        recovar_positive=np.array([False]),
        recovar_posterior_mass=np.array([0.1]),
        recovar_reconstruction_mass=np.array([0.0]),
        recovar_max_sample_posterior=np.array([0.1]),
        recovar_reconstruction_threshold=0.0,
    )
    assert report["recovar_reconstruction_threshold"] == 0.0
    assert not report["recovar_reconstruction_threshold_positive"]
    assert np.isnan(
        arrays["recovar_max_over_threshold_relion_positive_recovar_nonpositive"]
    ).all()
