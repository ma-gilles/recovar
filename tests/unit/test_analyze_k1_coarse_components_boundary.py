from __future__ import annotations

import numpy as np

from scripts.analyze_k1_coarse_components_boundary import _cutoff_boundary


def test_cutoff_boundary_reports_opposite_winner_threshold_margins() -> None:
    native_probability = np.asarray([[0.99899995, 0.00096878, 0.00003127]], dtype=np.float64)
    recovar_probability = np.asarray([[0.99900019, 0.00096854, 0.00003127]], dtype=np.float64)
    report = _cutoff_boundary(
        native_probability=native_probability,
        recovar_probability=recovar_probability,
        native_significant=np.asarray([[True, True, False]]),
        recovar_significant=np.asarray([[True, False, False]]),
        native_raw=np.asarray([[3634.6006, 3636.4414, 3640.0]], dtype=np.float32),
        recovar_scores_pre=np.asarray([[-3634.6003, -3636.4414, -3640.0]]),
        recovar_scores_with=np.asarray([[-3641.9202, -3648.8589, -3655.0]]),
        adaptive_fraction=0.999,
    )

    assert report["relion_winner_margin_above_cutoff"] < 0.0
    assert report["recovar_winner_margin_above_cutoff"] > 0.0
    assert len(report["mismatch_candidates"]) == 1
    mismatch = report["mismatch_candidates"][0]
    assert mismatch["flat_recovar"] == 1
    assert mismatch["relion_significant"] is True
    assert mismatch["recovar_significant"] is False
    expected_delta = -np.float32(3636.4414) + np.float32(3634.6006)
    assert mismatch["relion_raw_score_delta_from_winner"] == expected_delta
