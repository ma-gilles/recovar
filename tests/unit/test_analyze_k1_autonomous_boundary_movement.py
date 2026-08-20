import numpy as np
import pytest

from scripts.analyze_k1_autonomous_boundary_movement import analyze


def _state(pmax, support, *, identity="same"):
    n = len(pmax)
    return {
        "identity_sha256": identity,
        "rows": np.arange(n, dtype=np.int64) + 10,
        "pmax_recovar": np.asarray(pmax, dtype=np.float64),
        "pmax_relion": np.asarray([0.5, 0.6, 0.7], dtype=np.float64),
        "support_recovar": np.asarray(support, dtype=np.int64),
        "support_relion": np.asarray([2, 3, 4], dtype=np.int64),
        "rotation_error_deg": np.zeros(n, dtype=np.float64),
        "translation_error_angstrom": np.zeros(n, dtype=np.float64),
    }


def test_analyze_reports_improvement_and_support_row_movement():
    baseline = _state([0.4, 0.5, 0.6], [1, 3, 4])
    candidate = _state([0.49, 0.59, 0.69], [2, 3, 4])

    report = analyze(baseline, candidate, 0.99, 0.995)

    assert report["classification"] == "moves_toward_relion_without_measured_regression"
    assert report["movement"]["support"] == {
        "fixed_source_rows": [10],
        "new_source_rows": [],
        "retained_source_rows": [],
    }
    assert report["arms"]["candidate"]["pmax"]["relative_l2"] < report["arms"]["baseline"]["pmax"]["relative_l2"]
    assert report["movement"]["merged_fsc_deficit_ratio_candidate_over_baseline"] == pytest.approx(0.5)


def test_analyze_rejects_new_support_mismatch_and_identity_mismatch():
    baseline = _state([0.4, 0.5, 0.6], [2, 3, 4])
    candidate = _state([0.49, 0.59, 0.69], [2, 2, 4])

    report = analyze(baseline, candidate, 0.99, 0.995)
    assert report["classification"] == "mixed_or_regressive_boundary_result"
    assert report["movement"]["support"]["new_source_rows"] == [11]

    mismatched = _state([0.49, 0.59, 0.69], [2, 3, 4], identity="other")
    with pytest.raises(ValueError, match="identity hashes differ"):
        analyze(baseline, mismatched, 0.99, 0.995)


def test_analyze_accepts_exact_pmax_preservation_with_fewer_pose_outliers():
    baseline = _state([0.5, 0.6, 0.7], [2, 3, 4])
    candidate = _state([0.5, 0.6, 0.7], [2, 3, 4])
    baseline["translation_error_angstrom"] = np.asarray([1.0, 1.0, 0.0])
    candidate["translation_error_angstrom"] = np.asarray([1.0, 0.0, 0.0])

    report = analyze(baseline, candidate, 0.99, 0.995)

    assert report["classification"] == "moves_toward_relion_without_measured_regression"
    assert report["gates"]["pmax_relative_l2_not_worse"]
    assert report["arms"]["baseline"]["translation_error_gt_0p01_angstrom_count"] == 2
    assert report["arms"]["candidate"]["translation_error_gt_0p01_angstrom_count"] == 1
