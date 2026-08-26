import numpy as np

from scripts.analyze_k1_pass2_state_ab import _metric, analyze


def _write_capture(path, *, image_correction, projection_delta=0.0):
    np.savez(
        path,
        iteration=np.int64(2),
        half=np.int64(2),
        original_index=np.int64(28262),
        local_index=np.int64(5),
        current_size=np.int64(58),
        n_fine_trans=np.int64(2),
        fine_translations=np.zeros((2, 2), dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        oversampled_rot_indices=np.asarray([3], dtype=np.int64),
        parent_map=np.asarray([0], dtype=np.int32),
        candidate_mask=np.asarray([[True, True]]),
        window_indices=np.asarray([0, 1], dtype=np.int32),
        recon_window_indices=np.asarray([0, 1], dtype=np.int32),
        relion_integer_pre_shift=np.asarray([0, 0], dtype=np.int32),
        batch_image_correction=np.float32(image_correction),
        proj_half=np.asarray([[1 + projection_delta, 2]], dtype=np.complex64),
        raw_operand_raw_diff2=np.asarray([[2.0, 3.0]], dtype=np.float32),
        scores_pre_prior=np.asarray([[2.0, 3.0]], dtype=np.float64),
        rotation_log_prior=np.asarray([0.0]),
        translation_log_prior=np.asarray([0.0, 0.0]),
        scores_with_prior=np.asarray([[2.0, 3.0]], dtype=np.float64),
        probs=np.asarray([[0.25, 0.75]], dtype=np.float64),
        reconstruction_mask=np.asarray([[True, False]]),
        reconstruction_probs=np.asarray([[0.25, 0.0]], dtype=np.float64),
        reconstruction_n_significant=np.int64(1),
    )


def test_metric_handles_matching_infinities_without_nan_metrics():
    metric = _metric(
        np.asarray([1.0, -np.inf]),
        np.asarray([1.25, -np.inf]),
    )
    assert metric["finite_mask_equal"]
    assert metric["value_mismatch_count"] == 1
    assert metric["max_abs"] == 0.25


def test_analyze_reports_first_unequal_causal_stage(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_capture(reference, image_correction=1.0)
    _write_capture(candidate, image_correction=1.25, projection_delta=0.5)

    report = analyze(reference_path=reference, candidate_path=candidate)

    assert report["summary"]["first_unequal"] == {
        "stage": "particle_score_inputs",
        "field": "batch_image_correction",
    }
    assert report["stages"]["projected_reference"]["proj_half"]["max_abs"] == 0.5
    assert report["summary"]["reference"]["pmax"] == 0.75
