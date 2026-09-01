import numpy as np
import pytest

from scripts.analyze_k1_native_serialized_direction_prior import _error_stats


def test_error_stats_reports_signed_mean_and_absolute_tail():
    stats = _error_stats(np.asarray([-2.0, 0.0, 1.0, 3.0]))

    assert stats["count"] == 4
    assert stats["max_abs"] == 3.0
    assert stats["mean"] == 0.5
    assert stats["p95_abs"] == pytest.approx(2.85)
