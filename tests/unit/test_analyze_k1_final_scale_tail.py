import numpy as np
import pytest

from scripts.analyze_k1_final_scale_tail import exposure_panel


def test_exposure_panel_reports_tail_enrichment_without_correlation():
    values = np.arange(1000, dtype=np.float64)
    pose_tail = values >= 990
    support_changed = values >= 900

    panel = exposure_panel(
        values,
        {"pose_tail": pose_tail, "support_changed": support_changed},
    )

    assert panel["q0.99"]["exposed_count"] == 10
    assert panel["q0.99"]["outcomes"]["pose_tail"]["exposed_fraction"] == 1.0
    assert panel["q0.99"]["outcomes"]["pose_tail"]["remainder_fraction"] == 0.0
    assert panel["q0.99"]["outcomes"]["pose_tail"]["enrichment"] is None
    assert panel["q0.9"]["outcomes"]["support_changed"]["exposed_fraction"] == 1.0


def test_exposure_panel_fails_closed_on_misaligned_target():
    with pytest.raises(ValueError, match="target pose shape differs"):
        exposure_panel(np.arange(10), {"pose": np.zeros(9, dtype=bool)})
