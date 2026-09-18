import numpy as np
import pandas as pd
import pytest

from scripts.diff_relion_recovar_per_iter import (
    _load_current_npz_artifact,
    compare_direction_priors,
    extract_recovar_direction_prior,
    extract_relion_direction_prior,
    relion_star_round_double,
    relion_star_round_scaled,
)


def test_cached_diagnostic_older_than_refinement_is_ignored(tmp_path):
    artifact = tmp_path / "pose_comparison_iter000.npz"
    refinement = tmp_path / "refinement_results.npz"
    np.savez(artifact, value=np.asarray([1.0]))
    np.savez(refinement, value=np.asarray([2.0]))

    assert _load_current_npz_artifact(artifact, refinement, label="pose") is None


def test_cached_diagnostic_newer_than_refinement_is_loaded(tmp_path):
    refinement = tmp_path / "refinement_results.npz"
    artifact = tmp_path / "pose_comparison_iter000.npz"
    np.savez(refinement, value=np.asarray([2.0]))
    np.savez(artifact, value=np.asarray([1.0]))

    loaded = _load_current_npz_artifact(artifact, refinement, label="pose")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["value"], [1.0])
    loaded.close()


def test_extract_relion_direction_prior_is_half_specific():
    relion_iter = {
        "model_h1": {
            "model_pdf_orient_class_1": pd.DataFrame(
                {"rlnOrientationDistribution": [0.2, 0.8]}
            )
        },
        "model_h2": {
            "model_pdf_orient_class_1": pd.DataFrame(
                {"rlnOrientationDistribution": [0.6, 0.4]}
            )
        },
    }
    np.testing.assert_allclose(extract_relion_direction_prior(relion_iter, 1), [0.2, 0.8])
    np.testing.assert_allclose(extract_relion_direction_prior(relion_iter, 2), [0.6, 0.4])


def test_extract_recovar_direction_prior_supports_ragged_object_trajectory(tmp_path):
    path = tmp_path / "result.npz"
    trajectory = np.empty((2, 2), dtype=object)
    trajectory[0] = [np.array([0.2, 0.8]), np.array([0.6, 0.4])]
    trajectory[1] = [np.array([0.1, 0.2, 0.7]), np.array([0.3, 0.3, 0.4])]
    np.savez(path, direction_prior_trajectory_per_half=trajectory)
    with np.load(path, allow_pickle=True) as result:
        half1, half2 = extract_recovar_direction_prior(result, 1)
    np.testing.assert_allclose(half1, [0.1, 0.2, 0.7])
    np.testing.assert_allclose(half2, [0.3, 0.3, 0.4])


def test_direction_prior_primary_diagnostics_are_direct_array_errors():
    stats = compare_direction_priors(np.array([0.25, 0.75]), np.array([0.20, 0.80]))
    assert not stats["mismatch"]
    assert stats["max_abs_diff"] == pytest.approx(0.05)
    assert stats["l1_diff"] == pytest.approx(0.10)
    assert stats["relative_l1_diff"] == pytest.approx(0.10)
    assert stats["mass_diff"] == pytest.approx(0.0)
    assert "corr_auxiliary" in stats


def test_direction_prior_size_mismatch_is_explicit():
    stats = compare_direction_priors(np.ones(2), np.ones(3))
    assert stats == {"mismatch": True, "n_relion": 2, "n_recovar": 3}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.00133367308, 0.001334),
        (0.000388976888, 0.0003889769),
        (-0.0479863286, -0.04799),
        (100001.234567, 100001.2),
    ],
)
def test_relion_star_round_double_matches_metadata_table_format(value, expected):
    assert relion_star_round_double(value) == pytest.approx(expected)


def test_relion_star_round_scaled_rounds_in_native_units():
    n4 = 128**4
    scaled = 0.00133367308 * n4
    assert relion_star_round_scaled(scaled, n4) == pytest.approx(0.001334 * n4)
