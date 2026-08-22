import numpy as np

from scripts.analyze_k1_coarse_map_score_counterfactual import _score_margin


def test_score_margin_returns_target_minus_winner_log_score():
    diff2 = np.asarray([[10.0, 11.5], [9.0, 12.0]], dtype=np.float32)

    assert _score_margin(diff2, winner=(0, 0), target=(1, 1)) == -2.0
