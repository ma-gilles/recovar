from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import compact_candidate_capture as capture
from scripts.analyze_k1_native_production_boundary import analyze


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.int32).tofile(stream)
        values.reshape(-1).tofile(stream)


def _make_fixture(tmp_path: Path, monkeypatch, *, score_delta: float = 0.0):
    native_dir = tmp_path / "native"
    capture_dir = tmp_path / "capture"
    native_dir.mkdir()
    capture_dir.mkdir()
    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
        ]
    )
    rotation = np.asarray([0, 0, 1, 1], dtype=np.int32)
    translation = np.asarray([0, 1, 0, 1], dtype=np.int32)
    native_raw = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    native_prior = np.asarray([-1.0, -2.0, -1.0, -2.0], dtype=np.float64)
    native_combined = -native_raw + native_prior
    native_combined_f32 = native_combined.astype(np.float32)
    posterior = np.exp(
        native_combined_f32 - native_combined_f32.max(), dtype=np.float32
    )
    posterior /= posterior.sum(dtype=np.float32)
    significant = posterior >= np.partition(posterior, -2)[-2]
    _write_flat(native_dir / "pass1_class0_fine_eulers.bin", rotations.astype(np.float64))
    _write_flat(native_dir / "pass1_acc_rot_idx.bin", rotation)
    _write_flat(native_dir / "pass1_acc_trans_idx.bin", translation)
    _write_flat(native_dir / "pass1_exp_Mweight_raw_preprior.bin", native_raw)
    _write_flat(native_dir / "pass1_candidate_combined_log_prior.bin", native_prior)
    _write_flat(
        native_dir / "pass1_candidate_weight_normalized.bin",
        posterior.astype(np.float64),
    )
    _write_flat(
        native_dir / "pass1_candidate_in_reconstruction_set.bin",
        significant.astype(np.int32),
    )

    production_scores = native_combined.reshape(1, 2, 2).astype(np.float32)
    production_scores[0, 1, 1] += np.float32(score_delta)
    production_posterior = np.exp(
        production_scores - production_scores.max(axis=(1, 2), keepdims=True),
        dtype=np.float32,
    )
    production_posterior /= production_posterior.sum(
        axis=(1, 2), keepdims=True, dtype=np.float32
    )
    production_significant = significant.reshape(1, 2, 2).copy()
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(capture_dir))
    monkeypatch.setenv(capture.CAPTURE_ITERATION_ENV, "2")
    monkeypatch.setattr(capture, "_capture_counter", 0)
    capture.maybe_capture_k1_production_bucket(
        iteration=2,
        half=1,
        image_indices=np.asarray([0], dtype=np.int64),
        original_indices=np.asarray([84206], dtype=np.int64),
        per_image_inputs={
            "oversampled_rots": [rotations],
            "oversampled_rot_indices": [np.asarray([20, 21], dtype=np.int64)],
            "parent_map": [np.asarray([0, 1], dtype=np.int32)],
            "unique_rot": [np.asarray([100, 101], dtype=np.int32)],
        },
        current_size=58,
        fine_translations=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        fine_translation_parent=np.asarray([0, 0], dtype=np.int32),
        scores=production_scores,
        probs=production_posterior,
        rotation_log_prior=np.asarray([[-1.0, -1.0]], dtype=np.float32),
        translation_log_prior=np.asarray([[0.0, -1.0]], dtype=np.float32),
        candidate_mask=np.ones_like(production_scores, dtype=bool),
        reconstruction_mask=production_significant,
        log_z=np.log(
            np.exp(production_scores, dtype=np.float32).sum(
                axis=(1, 2), dtype=np.float32
            )
        ),
        best_log_score=production_scores.max(axis=(1, 2)),
        best_argmax=production_scores.reshape(1, -1).argmax(axis=1),
        max_posterior=production_posterior.max(axis=(1, 2)),
    )
    return native_dir, next(capture_dir.glob("raw_k1_*.npz"))


@pytest.mark.unit
def test_exact_native_production_boundary_closes(tmp_path, monkeypatch):
    native_dir, production = _make_fixture(tmp_path, monkeypatch)

    report = analyze(native_dir, production)

    assert report["first_nonidentical_boundary"] is None
    assert report["candidate_set"]["exact"]
    assert report["significant_support"]["exact"]
    assert report["winner"]["native"] == report["winner"]["recovar"]


@pytest.mark.unit
def test_preprior_score_is_first_reported_mismatch(tmp_path, monkeypatch):
    native_dir, production = _make_fixture(tmp_path, monkeypatch, score_delta=0.25)

    report = analyze(native_dir, production)

    assert report["first_nonidentical_boundary"] == "preprior_score"
    assert report["preprior_score_centered_residual_recovar_minus_native"]["max_abs"] > 0
