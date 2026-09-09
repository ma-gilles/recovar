from __future__ import annotations

import json

import pytest

from scripts.summarize_vdam_iteration_profiles import summarize


pytestmark = pytest.mark.unit


def _write_meta(root, iteration: int, *, current_size: int, expectation_s: float) -> None:
    payload = {
        "current_size": current_size,
        "healpix_order": 3,
        "n_rotations": 100,
        "n_translations": 20,
        "vdam_iteration_profile_summary": {
            "schedule_time_s": 0.1,
            "expectation_time_s": expectation_s,
            "pre_artifact_time_s": expectation_s + 0.2,
        },
        "sparse_pass2_profile_summary": {
            "pass1_time_s": expectation_s / 4,
            "pass2_time_s": expectation_s / 2,
            "mean_significant_samples": 12.0,
        },
    }
    (root / f"run_it{iteration:03d}_recovar_meta.json").write_text(json.dumps(payload))


def test_summarize_profiles_aggregates_transition_phases(tmp_path):
    _write_meta(tmp_path, 1, current_size=32, expectation_s=2.0)
    _write_meta(tmp_path, 70, current_size=44, expectation_s=4.0)
    _write_meta(tmp_path, 90, current_size=64, expectation_s=8.0)

    report = summarize(tmp_path)

    assert report["all_iterations"]["iteration_count"] == 3
    assert report["all_iterations"]["timings"]["expectation_time_s"]["sum_s"] == pytest.approx(14.0)
    assert report["phases"]["pre_transition_1_69"]["iteration_count"] == 1
    assert report["phases"]["accuracy_transition_70_89"]["first_iteration"] == 70
    late = report["phases"]["adaptive_and_late_90_plus"]
    assert late["timings"]["sparse_pass2_time_s"]["mean_s"] == pytest.approx(4.0)


def test_summarize_profiles_rejects_unprofiled_directory(tmp_path):
    (tmp_path / "run_it001_recovar_meta.json").write_text("{}")
    with pytest.raises(ValueError, match="no profiled"):
        summarize(tmp_path)
