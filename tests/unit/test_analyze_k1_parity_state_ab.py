import numpy as np

from scripts.analyze_k1_parity_state_ab import _array_metrics, _f32, analyze


def test_f32_reports_exact_bits():
    assert _f32(1.0) == {"value": 1.0, "bits_hex": "0x3f800000"}


def test_array_metrics_reports_equality_and_delta():
    control = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1.0, 3.0], dtype=np.float32)

    result = _array_metrics(control, candidate)

    assert result["shape"] == [2]
    assert result["bit_equal_fraction"] == 0.5
    assert result["max_abs_delta"] == 1.0
    assert result["relative_l2"] == np.sqrt(1.0 / 5.0)


def _write_dump(path, *, log_evidence, rotation_posterior_sums):
    particle_values = np.asarray([1.0, 2.0], dtype=np.float64)
    np.savez(
        path,
        iteration=np.int32(1),
        relion_iteration=np.int32(2),
        current_size=np.int32(64),
        sigma2_noise=np.asarray([3.0, 4.0]),
        half1_original_image_indices=np.asarray([7, 9], dtype=np.int64),
        half1_log_evidence=np.asarray(log_evidence, dtype=np.float64),
        half1_wsum_norm_correction=particle_values,
        half1_norm_corrections=particle_values,
        half1_image_corrections=particle_values,
        half1_scale_corrections=particle_values,
        half1_max_posterior=np.asarray([0.5, 0.75], dtype=np.float32),
        half1_best_log_score=particle_values,
        half1_hard_assignment=np.asarray([10, 11], dtype=np.int64),
        half1_coarse_hard_assignment=np.asarray([5, 6], dtype=np.int64),
        half1_avg_norm_correction=np.float64(1.5),
        half1_sumw=np.float64(2.0),
        half1_Ft_y_total=np.float64(3.0),
        half1_Ft_ctf_total=np.float64(4.0),
        half1_rotation_posterior_sums=np.asarray(rotation_posterior_sums, dtype=np.float32),
        half1_wsum_sigma2_noise=np.asarray([1.0, 2.0]),
        half1_wsum_img_power=np.asarray([3.0, 4.0]),
        half1_mean_real_ds=np.ones((2, 2), dtype=np.float32),
        half1_unreg_mean_real_ds=np.ones((2, 2), dtype=np.float32),
        half1_extra_particle_vector=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_analyze_covers_log_evidence_rotation_sums_and_all_particle_fields(tmp_path):
    control = tmp_path / "control.npz"
    candidate = tmp_path / "candidate.npz"
    _write_dump(control, log_evidence=[10.0, 20.0], rotation_posterior_sums=[1.0, 2.0, 3.0])
    _write_dump(candidate, log_evidence=[10.0, 20.25], rotation_posterior_sums=[1.0, 2.5, 3.0])

    report = analyze(control_path=control, candidate_path=candidate, half=1, original_index=9)

    assert report["target"]["log_evidence"]["candidate_minus_control"] == 0.25
    assert report["arrays"]["half1_rotation_posterior_sums"]["max_abs_delta"] == 0.5
    assert (
        report["target_all_particle_aligned_fields"]["half1_extra_particle_vector"]["control"]
        == [3.0, 4.0]
    )
    assert report["all_common_numeric_fields"]["half1_log_evidence"]["bit_equal_fraction"] == 0.5
