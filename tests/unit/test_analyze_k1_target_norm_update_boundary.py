import numpy as np

from scripts.analyze_k1_target_norm_update_boundary import analyze


def test_target_norm_update_boundary_closes_atomic_total(tmp_path):
    capture_path = tmp_path / "scale_aa_chunked_orig000066_half2_cs056.npz"
    pixels = np.asarray([1.0, 2.0, 99.0], dtype=np.float32)
    shells = np.asarray([0, 1, -1], dtype=np.int32)
    divisor = 4**4
    current = 3.0
    high = 5.0
    native_wsum = (current + high) / divisor
    native_sqrt = np.sqrt(2.0 * native_wsum)
    np.savez(
        capture_path,
        schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v4"),
        iteration=np.int64(1),
        original_index=np.int64(66),
        wavg_diff2_atomic_rectangle_per_image=np.float64(current),
        relion_norm_high_shell=np.float64(high),
        wavg_diff2_atomic_rectangle_per_pixel=pixels,
        wavg_diff2_atomic_rectangle_shell_indices=shells,
        candidate_posterior_probs=np.asarray([[0.25, 0.75]], dtype=np.float32),
        posterior_mass_per_chunk=np.asarray([0.25, 0.75], dtype=np.float32),
    )
    native_log = tmp_path / "runner.stderr"
    native_log.write_text(
        "RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=7 "
        f"previous_norm={float(1.0).hex()} previous_avg={float(1.0).hex()} "
        f"old_norm_over_avg={float(1.0).hex()} wsum_norm={native_wsum.hex()} "
        f"sqrt_2_wsum={native_sqrt.hex()} new_norm={native_sqrt.hex()}\n"
        "RELION_P1_NORM_SPLIT_OPERANDS_V1 iter=1 part_id=7 "
        f"current_size={(current / divisor).hex()} high_shell={(high / divisor).hex()} "
        f"total={native_wsum.hex()}\n"
    )

    report = analyze(
        capture_path,
        native_log,
        image_size=4,
        iteration=1,
        part_id=7,
        source_index=66,
    )

    assert report["comparison"]["bit_exact_float64"]
    assert report["comparison"]["wsum_norm_delta"] == 0.0
    assert report["recovar"]["posterior_mass_float64_sum"] == 1.0
    assert report["recovar"]["posterior_chunk_mass_float64_sum"] == 1.0
    assert report["comparison"]["native_split"]["stopped_high_shell_delta"] == 0.0
    assert report["comparison"]["native_split"]["stopped_atomic_current_size_delta"] == 0.0


def test_target_norm_update_boundary_accepts_production_algebraic_capture(tmp_path):
    capture_path = tmp_path / "capture.npz"
    divisor = 4**4
    total = 8.0
    native_wsum = total / divisor
    native_sqrt = np.sqrt(2.0 * native_wsum)
    np.savez(
        capture_path,
        schema=np.asarray("recovar-k1-norm-residual-inputs-v3"),
        iteration=np.int64(1),
        original_index=np.int64(66),
        relion_norm_high_shell=np.float64(5.0),
        weighted_img_per_image=np.float64(9.0),
        block_norm_residual=np.float64(-1.0),
        posterior_probs=np.asarray([[0.25, 0.75]], dtype=np.float32),
        wavg_diff2_atomic_rectangle_per_image=np.float64(3.0),
        wavg_diff2_atomic_rectangle_per_pixel=np.asarray([1.0, 2.0], dtype=np.float32),
        wavg_diff2_atomic_rectangle_shell_indices=np.asarray([0, 1], dtype=np.int32),
    )
    native_log = tmp_path / "runner.stderr"
    native_log.write_text(
        "RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=7 "
        "previous_norm=0x1p+0 previous_avg=0x1p+0 old_norm_over_avg=0x1p+0 "
        f"wsum_norm={native_wsum.hex()} sqrt_2_wsum={native_sqrt.hex()} "
        f"new_norm={native_sqrt.hex()}\n"
    )

    report = analyze(
        capture_path,
        native_log,
        image_size=4,
        iteration=1,
        part_id=7,
        source_index=66,
    )

    assert report["comparison"]["bit_exact_float64"]
    assert report["recovar"]["norm_path"] == "production_algebraic_weighted_image_plus_residual"
    assert report["recovar"]["current_size_norm_internal"] == 3.0
    assert report["recovar"]["posterior_chunk_mass_float64_sum"] is None
    assert report["comparison"]["production_algebraic_wsum_delta"] == 0.0
    assert report["comparison"]["direct_atomic_wsum_delta"] == 0.0


def test_target_norm_update_boundary_prefers_full_iteration_state(tmp_path):
    capture_path = tmp_path / "capture.npz"
    np.savez(
        capture_path,
        schema=np.asarray("recovar-k1-norm-residual-inputs-v3"),
        iteration=np.int64(1),
        original_index=np.int64(66),
        relion_norm_high_shell=np.float64(5.0),
        weighted_img_per_image=np.float64(9.0),
        block_norm_residual=np.float64(-1.0),
        posterior_probs=np.asarray([[1.0]], dtype=np.float32),
    )
    state_path = tmp_path / "iter_001.npz"
    np.savez(
        state_path,
        half1_original_image_indices=np.asarray([1], dtype=np.int64),
        half2_original_image_indices=np.asarray([66], dtype=np.int64),
        half2_wsum_norm_correction=np.asarray([9.0], dtype=np.float64),
        half2_norm_corrections=np.asarray([3.0], dtype=np.float64),
        half2_avg_norm_correction=np.float64(4.0),
        half2_image_corrections=np.asarray([1.25], dtype=np.float32),
        half2_scale_corrections=np.asarray([1.0], dtype=np.float32),
    )
    native_wsum = 8.0 / 4**4
    native_sqrt = np.sqrt(2.0 * native_wsum)
    native_log = tmp_path / "runner.stderr"
    native_log.write_text(
        "RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=7 "
        "previous_norm=0x1p+0 previous_avg=0x1p+0 old_norm_over_avg=0x1p+0 "
        f"wsum_norm={native_wsum.hex()} sqrt_2_wsum={native_sqrt.hex()} "
        f"new_norm={native_sqrt.hex()}\n"
    )

    report = analyze(
        capture_path,
        native_log,
        image_size=4,
        iteration=1,
        part_id=7,
        source_index=66,
        iteration_state_path=state_path,
    )

    assert report["comparison"]["authoritative_recovar_source"] == "full_iteration_state"
    assert report["comparison"]["wsum_norm_delta"] == 1.0 / 4**4
    assert report["recovar"]["stopped_capture_wsum_norm_relion_units"] == native_wsum
