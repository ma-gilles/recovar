from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.analyze_vdam_posterior_ensemble import SCHEMA, _ensemble_metrics, _ratio_summary

pytestmark = pytest.mark.unit


def test_posterior_ensemble_schema_is_explicit():
    assert SCHEMA == "recovar.vdam_posterior_ensemble.v1"


def test_ensemble_metric_accepts_candidate_inside_native_span():
    native = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 3.0]])
    result = _ensemble_metrics(native, np.asarray([1.0, 3.5]), mode="relative_l2")

    assert result["candidate_within_native_pair_max"] is True
    assert result["candidate_inside_coordinate_envelope_fraction"] == 1.0
    assert result["candidate_nearest_over_native_pair_max"] < 1.0


def test_ensemble_rms_metric_rejects_outside_candidate():
    native = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    result = _ensemble_metrics(native, np.asarray([3.0, 0.0]), mode="rms")

    assert result["candidate_within_native_pair_max"] is False
    assert result["candidate_inside_coordinate_envelope_fraction"] == 0.5
    assert result["candidate_nearest_over_native_pair_max"] == pytest.approx(2.0)


def test_ratio_summary_counts_infinite_zero_repeat_cases():
    result = _ratio_summary([0.5, float("inf")])

    assert result["finite"]["max"] == pytest.approx(0.5)
    assert result["infinite_count"] == 1
    assert math.isfinite(result["finite"]["mean"])
