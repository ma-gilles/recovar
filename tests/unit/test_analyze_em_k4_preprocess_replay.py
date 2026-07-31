import numpy as np
import pytest

from scripts import analyze_em_k4_preprocess_replay as analyzer


def _runs():
    normalized = np.ones((analyzer.REPLAY_COUNT, 1, 4, 4), dtype=np.float32)
    masked = normalized.copy()
    fourier = np.fft.rfft2(masked).astype(np.complex64)
    return normalized, masked, fourier


def _analyze(normalized, masked, fourier):
    return analyzer.analyze_replay_arrays(
        normalized_runs=normalized,
        masked_runs=masked,
        masked_fourier_runs=fourier,
    )


def test_all_exact_replays_have_fixed_denominator():
    normalized, masked, fourier = _runs()

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "preprocessing_replays_bitwise_exact"
    assert report["fixed_metric"] == {
        "evaluated_comparisons": 9,
        "expected_comparisons": 9,
        "bitwise_equal_comparisons": 9,
        "within_fixed_material_floor_comparisons": 9,
    }
    assert report["scorecard_change_admissible"] is False
    assert report["correlation_used"] is False


def test_softmask_roundoff_below_fixed_floor_is_localized():
    normalized, masked, fourier = _runs()
    masked[2, 0, 0, 0] += np.float32(1.0e-7)
    fourier[2] = np.fft.rfft2(masked[2]).astype(np.complex64)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "softmask_background_reduction_drift_within_fixed_material_floor"
    assert report["stages"]["normalized_shifted_real"]["bitwise_equal_comparison_count"] == 3
    assert report["stages"]["masked_real"]["bitwise_equal_comparison_count"] == 2
    assert report["fixed_metric"]["within_fixed_material_floor_comparisons"] == 9


def test_normalization_roundoff_precedes_softmask():
    normalized, masked, fourier = _runs()
    normalized[1, 0, 0, 0] += np.float32(1.0e-7)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "normalization_or_translation_roundoff_within_fixed_material_floor"


def test_material_softmask_drift_is_not_hidden_by_roundoff_label():
    normalized, masked, fourier = _runs()
    masked[3, 0, 0, 0] += np.float32(1.0e-3)
    fourier[3] = np.fft.rfft2(masked[3]).astype(np.complex64)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "material_drift_begins_at_softmask_background"
    assert report["fixed_metric"]["within_fixed_material_floor_comparisons"] < 9


def test_wrong_replay_count_is_rejected():
    normalized, masked, fourier = _runs()

    with pytest.raises(ValueError, match="exactly 4 executions"):
        _analyze(normalized[:3], masked, fourier)
