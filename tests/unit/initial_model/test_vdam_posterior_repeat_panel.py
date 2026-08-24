from __future__ import annotations

import math

import pytest

from scripts.analyze_vdam_posterior_repeat_panel import (
    SCHEMA,
    _centered_positive_logs,
    _distance_ratio,
    _normalize_positive_weights,
    _ratio_summary,
    _rms,
)

pytestmark = pytest.mark.unit


def test_posterior_repeat_schema_is_explicit():
    assert SCHEMA == "recovar.vdam_posterior_repeat_panel.v2"


def test_distance_ratio_uses_nearest_native_repeat():
    assert _distance_ratio(3.0, 2.0, 4.0) == pytest.approx(0.5)


def test_distance_ratio_handles_exact_native_repeats():
    assert _distance_ratio(0.0, 0.0, 0.0) == 0.0
    assert math.isinf(_distance_ratio(1.0, 2.0, 0.0))


def test_ratio_summary_reports_infinite_repeat_outliers_without_json_infinity():
    result = _ratio_summary([0.0, 0.5, float("inf")])

    assert result["finite"]["max"] == pytest.approx(0.5)
    assert result["infinite_count"] == 1


def test_normalize_positive_weights_excludes_native_negative_sentinel():
    result = _normalize_positive_weights([-3.4028235e38, 2.0, 6.0])

    assert result.tolist() == pytest.approx([0.0, 0.25, 0.75])


def test_centered_positive_logs_remove_arbitrary_exponent_frame():
    support, logs = _centered_positive_logs([-1.0, 2.0, 8.0])

    assert support.tolist() == [False, True, True]
    assert logs.tolist() == pytest.approx([-math.log(4.0), 0.0])


def test_rms_is_absolute_score_residual_scale():
    assert _rms([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
