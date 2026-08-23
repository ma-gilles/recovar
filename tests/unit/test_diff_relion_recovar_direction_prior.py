import numpy as np
import pandas as pd
import pytest

from scripts.diff_relion_recovar_per_iter import (
    compare_direction_priors,
    extract_recovar_direction_prior,
    extract_relion_direction_prior,
)


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
