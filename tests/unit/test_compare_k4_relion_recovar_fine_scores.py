import numpy as np

from scripts.compare_k4_relion_recovar_fine_scores import _counterfactual_residuals, _metric


def test_fine_score_counterfactual_identifies_data_component():
    components = {
        "data_log_score_centered": np.asarray([1.0, -2.0, 3.0]),
        "orientation_log_prior": np.asarray([0.01, -0.01, 0.0]),
        "translation_log_prior": np.asarray([0.0, 0.01, -0.01]),
    }

    report = _counterfactual_residuals(components)

    assert report["strongest_single_component"] == "data_log_score_centered"
    assert report["strongest_residual_energy_removed_fraction"] > 0.999


def test_fine_score_metric_is_directional_and_exact_aware():
    relion = np.asarray([1.0, 2.0], dtype=np.float32)
    recovar = np.asarray([1.0, 2.25], dtype=np.float64)

    report = _metric(relion, recovar)

    assert report["exact_equal"] is False
    assert report["mismatch_count"] == 1
    assert report["max_abs"] == 0.25
    assert report["relative_l2_over_relion"] > 0
