from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_vdam_live_posterior_repeat_panel as audit
from scripts.audit_vdam_repeat_panel import RepeatPanelError


def _rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float32)[None, :, :]


def _native(probabilities: tuple[float, float], weights: tuple[float, float]):
    return {
        "probabilities": np.asarray([probabilities], dtype=np.float32),
        "rotations": _rotation(),
        "raw_weights": np.asarray([weights], dtype=np.float64),
    }


def _candidate(probabilities: tuple[float, float]):
    values = np.asarray([probabilities], dtype=np.float32)
    scores = np.log(values.astype(np.float64))
    return {
        "posterior": values,
        "rotations": _rotation(),
        "centered_scores": scores - np.max(scores),
    }


def test_pairwise_rms_is_offset_invariant_after_centering():
    result = audit._pairwise_rms(
        [
            np.asarray([0.0, -1.0]),
            np.asarray([0.0, -1.5]),
            np.asarray([0.0, -2.0]),
        ]
    )
    assert result["minimum_pairwise_rms"] == pytest.approx(np.sqrt(0.125))
    assert result["maximum_pairwise_rms"] == pytest.approx(np.sqrt(0.5))
    assert len(result["pairwise"]) == 3


def test_diameter_ratio_handles_exact_repeat_panels():
    assert audit._diameter_ratio(0.0, 0.0) == 0.0
    assert np.isinf(audit._diameter_ratio(1.0, 0.0))
    assert audit._diameter_ratio(0.5, 2.0) == 0.25


def test_load_candidate_capture_selects_active_rotation(tmp_path: Path):
    path = tmp_path / "local_fused_posterior_it001_image_7.npz"
    np.savez(
        path,
        selected_global_image_indices=np.asarray([7], dtype=np.int64),
        local_rotation_matrices=np.stack(
            [np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32)]
        ),
        translations=np.zeros((2, 2), dtype=np.float32),
        posterior=np.asarray([[[0.7, 0.3], [0.0, 0.0]]], dtype=np.float32),
        reconstruction_rotation_mask=np.asarray([[True, False]]),
        pass2_scores_total=np.asarray([[[5.0, 4.0], [-np.inf, -np.inf]]], dtype=np.float32),
        debug_iteration=np.asarray([1], dtype=np.int32),
    )
    result = audit._load_candidate_capture(path, iteration=1)
    assert result["original_index"] == 7
    assert result["posterior"].shape == (1, 2)
    np.testing.assert_array_equal(result["centered_scores"], [[0.0, -1.0]])


def test_load_candidate_capture_rejects_missing_scores(tmp_path: Path):
    path = tmp_path / "local_fused_posterior_it001_image_7.npz"
    np.savez(
        path,
        selected_global_image_indices=np.asarray([7], dtype=np.int64),
    )
    with pytest.raises(RepeatPanelError, match="lacks fields"):
        audit._load_candidate_capture(path, iteration=1)


def test_load_native_repeat_uses_complete_panel_capture(monkeypatch, tmp_path: Path):
    panel = {
        "orientation_count": 1,
        "eulers": np.eye(3, dtype=np.float32).reshape(1, 9),
        "weights": np.asarray([[3.0, 1.0]], dtype=np.float32),
        "part_id": 0,
        "path": str(tmp_path / "panel.bin"),
    }
    monkeypatch.setattr(
        audit,
        "_load_native_panels",
        lambda *_args, **_kwargs: {7: panel},
    )
    result = audit._load_native_repeat(tmp_path, iteration=1)
    assert set(result) == {7}
    np.testing.assert_array_equal(result[7]["rotations"], _rotation())
    np.testing.assert_allclose(result[7]["probabilities"], [[0.75, 0.25]])


def test_matched_audit_reports_candidate_underdispersion(monkeypatch, tmp_path: Path):
    native_repeats = [
        {7: _native((0.60, 0.40), (60.0, 40.0))},
        {7: _native((0.61, 0.39), (61.0, 39.0))},
    ]
    candidate_repeats = [
        {7: _candidate((0.6000, 0.4000))},
        {7: _candidate((0.6001, 0.3999))},
    ]

    monkeypatch.setattr(
        audit,
        "_load_json",
        lambda *_args, **_kwargs: {
            "execution_policy": (
                "one allocation, one physical H100, sequential native and live candidate arms"
            ),
            "physical_gpu_uuid": "GPU-test",
            "science_head": "abc123",
        },
    )
    native_iter = iter(native_repeats)
    candidate_iter = iter(candidate_repeats)
    monkeypatch.setattr(audit, "_load_native_repeat", lambda *_args, **_kwargs: next(native_iter))
    monkeypatch.setattr(
        audit,
        "_load_candidate_repeat",
        lambda *_args, **_kwargs: next(candidate_iter),
    )

    result = audit.audit_live_posterior_repeat_panel(tmp_path, repeat_count=2)
    assert result["result"] == "complete"
    assert result["particle_count"] == 1
    assert result["physical_gpu_uuid"] == "GPU-test"
    assert result["posterior"]["candidate_over_native_maximum_diameter"] < 0.02
    assert result["centered_score"]["candidate_over_native_maximum_diameter"] < 0.02


def test_matched_audit_rejects_identity_drift(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        audit,
        "_load_json",
        lambda *_args, **_kwargs: {
            "execution_policy": (
                "one allocation, one physical H100, sequential native and live candidate arms"
            )
        },
    )
    native_iter = iter([{7: _native((0.6, 0.4), (6.0, 4.0))}] * 2)
    candidate_iter = iter(
        [
            {7: _candidate((0.6, 0.4))},
            {8: _candidate((0.6, 0.4))},
        ]
    )
    monkeypatch.setattr(audit, "_load_native_repeat", lambda *_args, **_kwargs: next(native_iter))
    monkeypatch.setattr(
        audit,
        "_load_candidate_repeat",
        lambda *_args, **_kwargs: next(candidate_iter),
    )
    with pytest.raises(RepeatPanelError, match="identities differ"):
        audit.audit_live_posterior_repeat_panel(tmp_path, repeat_count=2)
