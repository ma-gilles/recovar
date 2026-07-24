from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import validate_bpref_device_signature as validator

pytestmark = pytest.mark.unit


def _save(path: Path, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


def _rewrite_signature(path: Path, updates: dict):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(updates)
    _save(path, values)


def _make_nonexact_coefficient_compare(signature: Path, panel: Path):
    with np.load(signature, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    coefficients = values["neighbor_coefficients"].copy()
    coefficients[0, 0, 0] = np.float32(2.0)
    values["neighbor_coefficients"] = coefficients
    _save(signature, values)

    with np.load(panel, allow_pickle=False) as archive:
        panel_values = {key: archive[key] for key in archive.files}
    data = panel_values["data_accumulator"].copy()
    weight = panel_values["weight_accumulator"].copy()
    data[97] += np.complex64(1.0 + 1.0j)
    weight[97] += np.float32(1.0)
    panel_values["data_accumulator"] = data
    panel_values["weight_accumulator"] = weight
    _save(panel, panel_values)


def _artifacts(
    tmp_path: Path,
    corruption: str | None = None,
    *,
    high_precision_operand_bundle: bool = False,
):
    stack_sha = "a" * 64
    summed = np.asarray(
        [
            [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j],
            [0, 0, 0, 0],
        ],
        dtype=np.complex64,
    )
    weights = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.float32)
    if corruption == "zero_contributor":
        weights[0] = 0
    contribution_path = tmp_path / "contribution.npz"
    contribution = {
        "magic": np.asarray(validator.CONTRIBUTION_MAGIC),
        "schema": np.asarray(validator.CONTRIBUTION_SCHEMA),
        "schema_version": np.int32(3),
        "run_id": np.asarray("synthetic"),
        "iteration": np.int32(5),
        "half": np.int32(1),
        "rank": np.int32(0),
        "pass_index": np.int32(2),
        "class_index": np.int32(1),
        "call_index": np.int64(0),
        "dump_index": np.int64(0),
        "source_stack_sha256": np.asarray(stack_sha),
        "shadow_only_mode": np.bool_(True),
        "shadow_score_bitwise_equal": np.bool_(True),
        "shadow_reduction_data_rel_l1": np.float64(2e-4),
        "shadow_reduction_data_normalized_max": np.float64(3e-4),
        "shadow_reduction_weight_rel_l1": np.float64(1e-7),
        "shadow_reduction_weight_normalized_max": np.float64(1e-7),
        "shadow_reduction_rel_l1_bound": np.float64(1e-3),
        "shadow_reduction_normalized_max_bound": np.float64(1e-3),
        "current_size": np.int64(2),
        "image_shape": np.asarray([4, 4], dtype=np.int32),
        "volume_shape": np.asarray([7, 7, 7], dtype=np.int32),
        "local_indices": np.asarray([0], dtype=np.int64),
        "image_identities": np.asarray(["1@/tmp/stack.mrcs"]),
        "original_indices": np.asarray([0], dtype=np.int64),
        "star_rows": np.asarray([0], dtype=np.int64),
        "stack_indices_1based": np.asarray([1], dtype=np.int64),
        "resolved_stack_paths": np.asarray(["/tmp/stack.mrcs"]),
        "high_precision_operand_bundle": np.bool_(high_precision_operand_bundle),
        "raw_real_images": np.empty((0,), dtype=np.float32),
        "raw_source_dtype": np.asarray(""),
        "raw_source_shape": np.asarray([0], dtype=np.int64),
        "ctf_params": np.empty((0,), dtype=np.float32),
        "ctf_parameter_convention": np.asarray("recovar.CTFParamIndex-v1"),
        "noise_variance_half": np.empty((0,), dtype=np.float32),
        "integer_pre_shifts": np.empty((0, 2), dtype=np.int32),
        "image_corrections": np.empty((0,), dtype=np.float32),
        "scale_corrections": np.empty((0,), dtype=np.float32),
        "relion_preprocess_normalization_factors": np.empty((0,), dtype=np.float32),
        "relion_cuda_preprocess": np.bool_(False),
        "preprocess_backend": np.asarray("dataset_native"),
        "preprocess_convention": np.asarray("recovar-half-preprocess-v1"),
        "score_with_masked_images": np.bool_(False),
        "image_mask": np.empty((0,), dtype=np.float32),
        "image_mask_mode": np.asarray("none"),
        "voxel_size": np.float64(1.5),
        "ctf_mode": np.asarray("cryoem"),
        "ctf_dose_per_tilt": np.float64(0.0),
        "ctf_angle_per_tilt": np.float64(0.0),
        "disc_type": np.asarray("linear_interp"),
        "projection_padding_factor": np.int32(2),
        "reconstruction_padding_factor": np.int32(2),
        "actual_counts": np.asarray([2], dtype=np.int64),
        "oversampled_rotation_indices": np.asarray([[10, 11]], dtype=np.int64),
        "fine_translations": np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        "candidate_preprior_scores": np.asarray([[[-1.0, -2.0], [-3.0, -4.0]]]),
        "candidate_rotation_log_prior": np.zeros((1, 2), dtype=np.float64),
        "candidate_translation_log_prior": np.zeros((1, 2), dtype=np.float64),
        "candidate_combined_scores": np.asarray([[[-1.0, -2.0], [-3.0, -4.0]]]),
        "candidate_best_log_score": np.asarray([-1.0], dtype=np.float64),
        "candidate_log_z": np.asarray([-0.5598103], dtype=np.float64),
        "candidate_normalized_sum_exp": np.asarray([1.552], dtype=np.float64),
        "candidate_exponent_shift_f32": np.asarray([51.0], dtype=np.float32),
        "candidate_raw_exp_weights_f32": np.ones((1, 2, 2), dtype=np.float32),
        "posterior_probs": np.asarray([[[0.5, 0.3], [0.2, 0.0]]]),
        "reconstruction_probs": np.asarray([[[0.625, 0.375], [0.0, 0.0]]]),
        "reconstruction_mask": np.asarray([[[True, True], [False, False]]]),
        "reconstruction_sum_weight": np.asarray([0.8], dtype=np.float64),
        "reconstruction_threshold": np.asarray([0.2], dtype=np.float64),
        "candidate_mask": np.ones((1, 2, 2), dtype=bool),
        "window_indices": np.asarray([0, 1, 9, 10], dtype=np.int32),
        "active_particle_rows": np.asarray([0, 0], dtype=np.int32),
        "active_rotation_rows": np.asarray([0, 1], dtype=np.int32),
        "active_summed": summed,
        "active_ctf_probs": weights,
    }
    if high_precision_operand_bundle:
        contribution.update(
            raw_real_images=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            raw_source_dtype=np.asarray("float32"),
            raw_source_shape=np.asarray([1, 4, 4], dtype=np.int64),
            ctf_params=np.ones((1, 11), dtype=np.float32),
            noise_variance_half=np.ones(12, dtype=np.float64),
            integer_pre_shifts=np.zeros((1, 2), dtype=np.int32),
            image_corrections=np.ones(1, dtype=np.float32),
            scale_corrections=np.ones(1, dtype=np.float32),
            relion_preprocess_normalization_factors=np.ones(1, dtype=np.float32),
            image_mask=np.ones((4, 4), dtype=np.float32),
        )
    if corruption == "candidate_dtype":
        contribution["candidate_combined_scores"] = contribution[
            "candidate_combined_scores"
        ].astype(np.float32)
    elif corruption == "raw_shape":
        contribution["raw_source_shape"] = np.asarray([1, 16], dtype=np.int64)
    elif corruption == "contribution_identity":
        contribution["resolved_stack_paths"] = np.asarray(["/tmp/wrong.mrcs"])
    elif corruption == "shadow_gate":
        contribution["shadow_score_bitwise_equal"] = np.bool_(False)
    _save(contribution_path, contribution)
    contribution_sha = validator._sha256_file(contribution_path)

    omitted_rows = np.asarray([1], dtype=np.int32)
    omitted_keys = np.asarray([11], dtype=np.int32)
    omitted_digest = validator._noncontributor_digest(
        omitted_rows,
        omitted_keys,
        np.ascontiguousarray(summed[1:2]),
        np.ascontiguousarray(weights[1:2]),
    )
    source = np.zeros((1, 4, 6), dtype=np.float32)
    source[0, :, 0] = summed[0].real
    source[0, :, 1] = summed[0].imag
    source[0, :, 2] = weights[0]
    neighbor_indices = np.full((1, 4, 8), -1, dtype=np.int32)
    neighbor_coefficients = np.zeros((1, 4, 8), dtype=np.float32)
    neighbor_flags = np.full((1, 4, 8), 8, dtype=np.int32)
    neighbor_indices[0, :, 0] = 97
    neighbor_coefficients[0, :, 0] = 1
    neighbor_flags[0, :, 0] = 1
    signature_path = tmp_path / "signature.device.npz"
    signature = {
        "magic": np.asarray(validator.SIGNATURE_MAGIC),
        "schema": np.asarray(validator.SIGNATURE_SCHEMA),
        "schema_version": np.int32(1),
        "run_id": np.asarray("synthetic"),
        "iteration": np.int32(5),
        "half": np.int32(1),
        "rank": np.int32(0),
        "pass_index": np.int32(2),
        "class_index": np.int32(1),
        "call_index": np.int64(0),
        "dump_index": np.int64(0),
        "source_stack_sha256": np.asarray(stack_sha),
        "companion_contribution_path": np.asarray(str(contribution_path.resolve())),
        "companion_contribution_sha256": np.asarray(contribution_sha),
        "image_shape": np.asarray([4, 4], dtype=np.int32),
        "volume_shape": np.asarray([7, 7, 7], dtype=np.int32),
        "current_size": np.int32(2),
        "max_r": np.float32(1),
        "reconstruction_padding_factor": np.int32(2),
        "causal_arm": np.asarray("soft-posterior-per-particle-fused-xhalf"),
        "winner_take_all": np.bool_(False),
        "topology_claim": np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
        "signature_inertness_gate": np.asarray(
            "bitwise-post-accum-shadow-and-operand-exact"
        ),
        "signature_inertness_gate_passed": np.bool_(True),
        "signature_accumulator_shadow_bitwise_equal": np.bool_(True),
        "signature_prepared_operands_bitwise_equal": np.bool_(True),
        "signature_kernel_accumulate": np.bool_(False),
        "particle_launch_ordinals": np.asarray([0], dtype=np.int64),
        "particle_total_row_counts": np.asarray([2], dtype=np.int32),
        "particle_contributor_row_counts": np.asarray([1], dtype=np.int32),
        "particle_noncontributor_row_counts": np.asarray([1], dtype=np.int32),
        "particle_noncontributor_exact_zero": np.asarray([True]),
        "particle_noncontributor_zero_sha256": np.asarray([omitted_digest]),
        "particle_image_identities": np.asarray(["1@/tmp/stack.mrcs"]),
        "particle_original_indices": np.asarray([0], dtype=np.int64),
        "signature_bytes_per_dense_row_pixel": np.int32(132),
        "signature_estimated_uncompressed_bytes": np.int64(4 * 132),
        "launch_ordinal": np.asarray([0], dtype=np.int64),
        "particle_local_row": np.asarray([0], dtype=np.int32),
        "program_row": np.asarray([0], dtype=np.int32),
        "image_identity": np.asarray(["1@/tmp/stack.mrcs"]),
        "original_indices": np.asarray([0], dtype=np.int64),
        "contributor_canonical_rotation_keys": np.asarray([10], dtype=np.int32),
        "canonical_rotation_keys": np.full((1, 4), 10, dtype=np.int32),
        "canonical_pixel_indices": np.arange(4, dtype=np.int32)[None, :],
        "row_flags": np.full((1, 4), 64, dtype=np.int32),
        "source_values": source,
        "neighbor_indices": neighbor_indices,
        "neighbor_coefficients": neighbor_coefficients,
        "neighbor_flags": neighbor_flags,
        "program_axis_sizes": np.asarray([1, 4, 8], dtype=np.int64),
        "program_lane": np.arange(4, dtype=np.int32),
        "program_serial_pass": np.zeros(4, dtype=np.int32),
        "program_neighbor": np.arange(8, dtype=np.int32),
        "signature_tensor_axis_legend": np.asarray(
            "row-major [contributor_row,dense_pixel,neighbor]; program_row is the "
            "particle-local source rotation row; lane=dense_pixel%128; "
            "serial_pass=dense_pixel//128; neighbor=d0*4+d1*2+d2"
        ),
        "atomic_component_program_order_legend": np.asarray(
            "for each valid neighbor: atomicAdd(data_real), then atomicAdd(data_imag), "
            "then atomicAdd(weight)"
        ),
    }
    if corruption == "key":
        signature["canonical_rotation_keys"][0, 1] = 11
    elif corruption == "pixel":
        signature["canonical_pixel_indices"][0, 1] = 0
    elif corruption == "digest":
        signature["particle_noncontributor_zero_sha256"] = np.asarray(["0" * 64])
    elif corruption == "sequence":
        signature["particle_launch_ordinals"] = np.asarray([1], dtype=np.int64)
        signature["launch_ordinal"] = np.asarray([1], dtype=np.int64)
    elif corruption == "flags":
        signature["row_flags"][0, 0] = 64 | 4
    elif corruption == "source":
        signature["source_values"][0, 0, 0] += 1
    elif corruption == "orientation_early":
        signature["row_flags"][0, 0] = 1 | 16
        signature["source_values"][0, 0] = np.nan
        signature["neighbor_indices"][0, 0] = -1
        signature["neighbor_coefficients"][0, 0] = 0
        signature["neighbor_flags"][0, 0] = 8
    elif corruption == "inertness":
        signature["signature_accumulator_shadow_bitwise_equal"] = np.bool_(False)
    _save(signature_path, signature)

    panel_path = tmp_path / "panel.npz"
    data_accumulator = np.zeros(7 * 7 * 4, dtype=np.complex64)
    weight_accumulator = np.zeros(7 * 7 * 4, dtype=np.float32)
    data_accumulator[97] = np.sum(summed[0], dtype=np.complex64)
    weight_accumulator[97] = np.sum(weights[0], dtype=np.float32)
    panel = {
        "magic": np.asarray(validator.PANEL_MAGIC),
        "schema": np.asarray(validator.PANEL_SCHEMA),
        "schema_version": np.int32(1),
        "run_id": np.asarray("synthetic"),
        "iteration": np.int32(5),
        "half": np.int32(1),
        "rank": np.int32(0),
        "launch_count": np.int64(1),
        "current_size": np.int32(2),
        "max_r": np.float32(1),
        "image_shape": np.asarray([4, 4], dtype=np.int32),
        "volume_shape": np.asarray([7, 7, 7], dtype=np.int32),
        "reconstruction_padding_factor": np.int32(2),
        "source_stack_sha256": np.asarray(stack_sha),
        "causal_arm": np.asarray("soft-posterior-per-particle-fused-xhalf"),
        "winner_take_all": np.bool_(False),
        "topology_claim": np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
        "accumulator_field_legend": np.asarray(
            "data=complex64 x-half;weight=float32 x-half;flat C order"
        ),
        "data_accumulator": data_accumulator,
        "weight_accumulator": weight_accumulator,
    }
    if corruption == "panel_dtype":
        panel["data_accumulator"] = data_accumulator.astype(np.complex128)
    elif corruption == "topology":
        panel["current_size"] = np.int32(4)
    _save(panel_path, panel)
    return signature_path, panel_path


def test_valid_signature_and_panel_replay(tmp_path, capsys):
    signature, panel = _artifacts(tmp_path)
    validator.main([str(signature), "--panel-native", str(panel)])
    assert '"status": "PASS"' in capsys.readouterr().out


def test_zero_contributor_class_shard_retains_particle_manifest(tmp_path):
    signature_path, _ = _artifacts(tmp_path)
    with np.load(signature_path, allow_pickle=False) as archive:
        signature = {key: archive[key] for key in archive.files}
    contribution_path = Path(str(signature["companion_contribution_path"]))
    with np.load(contribution_path, allow_pickle=False) as archive:
        contribution = {key: archive[key] for key in archive.files}

    contribution["active_summed"] = np.zeros((2, 4), dtype=np.complex64)
    contribution["active_ctf_probs"] = np.zeros((2, 4), dtype=np.float32)
    _save(contribution_path, contribution)

    dense_pixels = 4
    empty_shapes = {
        "canonical_rotation_keys": ((0, dense_pixels), np.int32),
        "canonical_pixel_indices": ((0, dense_pixels), np.int32),
        "row_flags": ((0, dense_pixels), np.int32),
        "source_values": ((0, dense_pixels, 6), np.float32),
        "neighbor_indices": ((0, dense_pixels, 8), np.int32),
        "neighbor_coefficients": ((0, dense_pixels, 8), np.float32),
        "neighbor_flags": ((0, dense_pixels, 8), np.int32),
    }
    for key, (shape, dtype) in empty_shapes.items():
        signature[key] = np.empty(shape, dtype=dtype)
    for key, dtype in (
        ("launch_ordinal", np.int64),
        ("particle_local_row", np.int32),
        ("program_row", np.int32),
        ("original_indices", np.int64),
        ("contributor_canonical_rotation_keys", np.int32),
    ):
        signature[key] = np.empty((0,), dtype=dtype)
    signature["image_identity"] = np.empty(
        (0,), dtype=signature["particle_image_identities"].dtype
    )
    omitted_rows = np.asarray([0, 1], dtype=np.int32)
    omitted_keys = np.asarray([10, 11], dtype=np.int32)
    signature["particle_contributor_row_counts"] = np.asarray([0], dtype=np.int32)
    signature["particle_noncontributor_row_counts"] = np.asarray([2], dtype=np.int32)
    signature["particle_noncontributor_zero_sha256"] = np.asarray(
        [
            validator._noncontributor_digest(
                omitted_rows,
                omitted_keys,
                np.zeros((2, 4), dtype=np.complex64),
                np.zeros((2, 4), dtype=np.float32),
            )
        ]
    )
    signature["program_axis_sizes"] = np.asarray([0, dense_pixels, 8], dtype=np.int64)
    signature["signature_estimated_uncompressed_bytes"] = np.int64(0)
    signature["companion_contribution_sha256"] = np.asarray(
        validator._sha256_file(contribution_path)
    )
    _save(signature_path, signature)

    result = validator._validate_signature(signature_path)

    assert result["particle_count"] == 1
    assert result["contributor_rows"] == 0
    assert result["omitted_rows"] == 2
    assert result["contribution_records"].size == 0


def test_panel_class_identity_mismatch_fails_closed(tmp_path):
    signature, panel = _artifacts(tmp_path)
    with np.load(panel, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["class_index"] = np.int32(7)
    _save(panel, values)

    with pytest.raises(ValueError, match="panel/signature class_index mismatch"):
        validator.main([str(signature), "--panel-native", str(panel)])


def test_cross_engine_self_compare_is_exact(tmp_path, capsys):
    signature, panel = _artifacts(tmp_path)

    validator.main(
        [
            str(signature),
            "--panel-native",
            str(panel),
            "--compare-signatures",
            str(signature),
            "--compare-panel-native",
            str(panel),
        ]
    )

    output = capsys.readouterr().out
    assert '"classification": "exact"' in output


def test_cross_engine_compare_requires_same_frozen_boundary(tmp_path):
    signature, panel = _artifacts(tmp_path)
    with np.load(signature, allow_pickle=False) as archive:
        compare_values = {key: archive[key] for key in archive.files}
    compare_values["volume_shape"] = np.asarray([8, 8, 8], dtype=np.int32)
    compare_signature = tmp_path / "compare.device.npz"
    np.savez(compare_signature, **compare_values)

    with pytest.raises(ValueError, match="boundary mismatch for volume_shape"):
        validator.main(
            [
                str(signature),
                "--panel-native",
                str(panel),
                "--compare-signatures",
                str(compare_signature),
                "--compare-panel-native",
                str(panel),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pass_index", np.int32(1)),
        ("class_index", np.int32(2)),
        ("window_indices", np.asarray([6, 7, 3, 4], dtype=np.int32)),
    ],
)
def test_cross_engine_compare_rejects_wrong_pass_class_or_window_boundary(
    tmp_path, field, value
):
    primary_signature, primary_panel = _artifacts(tmp_path / "primary")
    compare_signature, compare_panel = _artifacts(tmp_path / "compare")
    if field in {"pass_index", "class_index", "window_indices"}:
        with np.load(compare_signature, allow_pickle=False) as archive:
            signature_values = {key: archive[key] for key in archive.files}
        contribution_path = Path(str(signature_values["companion_contribution_path"]))
        with np.load(contribution_path, allow_pickle=False) as archive:
            contribution_values = {key: archive[key] for key in archive.files}
        contribution_values[field] = value
        _save(contribution_path, contribution_values)
        if field != "window_indices":
            signature_values[field] = value
        signature_values["companion_contribution_sha256"] = np.asarray(
            validator._sha256_file(contribution_path)
        )
        _save(compare_signature, signature_values)
    else:
        _rewrite_signature(compare_signature, {field: value})

    expected = (
        f"companion boundary mismatch for {field}"
        if field == "window_indices"
        else f"boundary mismatch for {field}"
    )
    with pytest.raises(ValueError, match=expected):
        validator.main(
            [
                str(primary_signature),
                "--panel-native",
                str(primary_panel),
                "--compare-signatures",
                str(compare_signature),
                "--compare-panel-native",
                str(compare_panel),
            ]
        )


@pytest.mark.parametrize("coverage", ["incomplete", "overlap"])
def test_compare_shard_set_must_be_complete_and_nonoverlapping(tmp_path, coverage):
    primary_signature, primary_panel = _artifacts(tmp_path / "primary")
    compare_signature, compare_panel = _artifacts(tmp_path / "compare")
    compare_args = [str(compare_signature)]
    if coverage == "incomplete":
        _rewrite_signature(
            compare_signature,
            {
                "particle_launch_ordinals": np.asarray([1], dtype=np.int64),
                "launch_ordinal": np.asarray([1], dtype=np.int64),
            },
        )
        message = "begin at zero"
    else:
        compare_args.append(str(compare_signature))
        message = "ranges overlap"

    with pytest.raises(ValueError, match=message):
        validator.main(
            [
                str(primary_signature),
                "--panel-native",
                str(primary_panel),
                "--compare-signatures",
                *compare_args,
                "--compare-panel-native",
                str(compare_panel),
            ]
        )


def test_nonexact_cross_compare_fails_by_default_and_diagnostic_is_labeled(
    tmp_path, capsys
):
    primary_signature, primary_panel = _artifacts(tmp_path / "primary")
    compare_signature, compare_panel = _artifacts(tmp_path / "compare")
    _make_nonexact_coefficient_compare(compare_signature, compare_panel)
    args = [
        str(primary_signature),
        "--panel-native",
        str(primary_panel),
        "--compare-signatures",
        str(compare_signature),
        "--compare-panel-native",
        str(compare_panel),
    ]

    with pytest.raises(SystemExit) as error:
        validator.main(args)
    assert error.value.code == 1
    failed = capsys.readouterr().out
    assert '"status": "FAIL"' in failed
    assert '"artifact_validation_status": "PASS"' in failed
    assert '"cross_comparison_status": "FAIL"' in failed

    validator.main([*args, "--allow-nonexact-cross-diagnostic"])
    diagnostic = capsys.readouterr().out
    assert '"status": "DIAGNOSTIC_NONEXACT"' in diagnostic
    assert '"cross_comparison_status": "DIAGNOSTIC_NONEXACT"' in diagnostic
    assert '"status": "PASS"' not in diagnostic


def test_v3_high_precision_replay_bundle_is_validated_and_reported(tmp_path, capsys):
    signature, panel = _artifacts(tmp_path, high_precision_operand_bundle=True)
    validator.main([str(signature), "--panel-native", str(panel)])
    output = capsys.readouterr().out
    assert '"schema_replay_ready": true' in output
    assert '"high_precision_operand_bundle": true' in output
    assert '"raw_source_dtype": "float32"' in output


def test_v3_native_float32_reconstruction_probabilities_are_validated(tmp_path, capsys):
    signature, panel = _artifacts(tmp_path, high_precision_operand_bundle=True)
    with np.load(signature, allow_pickle=False) as archive:
        signature_values = {key: archive[key] for key in archive.files}
    contribution_path = Path(str(signature_values["companion_contribution_path"]))
    with np.load(contribution_path, allow_pickle=False) as archive:
        contribution = {key: archive[key] for key in archive.files}
    probabilities = contribution["reconstruction_probs"].astype(np.float32)
    contribution.update(
        reconstruction_probs=probabilities,
        reconstruction_probs_native_dtype=np.asarray("float32"),
        reconstruction_probs_native_itemsize=np.int32(probabilities.dtype.itemsize),
        reconstruction_probs_native_nbytes=np.int64(probabilities.nbytes),
        reconstruction_probs_storage_policy=np.asarray(
            "native-dtype-preserved;dtype-itemsize-nbytes-bound"
        ),
    )
    _save(contribution_path, contribution)
    _rewrite_signature(
        signature,
        {
            "companion_contribution_sha256": np.asarray(
                validator._sha256_file(contribution_path)
            )
        },
    )

    validator.main([str(signature), "--panel-native", str(panel)])
    output = capsys.readouterr().out
    assert '"status": "PASS"' in output
    assert '"reconstruction_probs_dtype": "float32"' in output
    assert '"reconstruction_probs_dtype_metadata_bound": true' in output


@pytest.mark.parametrize(
    ("metadata_update", "message"),
    [
        ({}, "requires bound dtype/itemsize/nbytes metadata"),
        (
            {
                "reconstruction_probs_native_dtype": np.asarray("float32"),
                "reconstruction_probs_native_itemsize": np.int32(4),
                "reconstruction_probs_native_nbytes": np.int64(1),
                "reconstruction_probs_storage_policy": np.asarray(
                    "native-dtype-preserved;dtype-itemsize-nbytes-bound"
                ),
            },
            "nbytes conflicts",
        ),
    ],
)
def test_v3_native_float32_probability_metadata_fails_closed(
    tmp_path, metadata_update, message
):
    signature, panel = _artifacts(tmp_path, high_precision_operand_bundle=True)
    with np.load(signature, allow_pickle=False) as archive:
        signature_values = {key: archive[key] for key in archive.files}
    contribution_path = Path(str(signature_values["companion_contribution_path"]))
    with np.load(contribution_path, allow_pickle=False) as archive:
        contribution = {key: archive[key] for key in archive.files}
    contribution["reconstruction_probs"] = contribution[
        "reconstruction_probs"
    ].astype(np.float32)
    contribution.update(metadata_update)
    _save(contribution_path, contribution)
    _rewrite_signature(
        signature,
        {
            "companion_contribution_sha256": np.asarray(
                validator._sha256_file(contribution_path)
            )
        },
    )

    with pytest.raises(ValueError, match=message):
        validator.main([str(signature), "--panel-native", str(panel)])


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_accumulator_fsc_is_exact_for_bitwise_identical_float32_inputs(dtype):
    rng = np.random.default_rng(20260715)
    values = rng.normal(size=7 * 7 * 4).astype(np.float32)
    if np.issubdtype(dtype, np.complexfloating):
        values = (values + 1j * rng.normal(size=values.size).astype(np.float32)).astype(dtype)
    else:
        values = values.astype(dtype)

    auc, min_shell, fsc = validator._accumulator_fsc(values, values.copy(), (7, 7, 7))

    assert auc == pytest.approx(1.0, abs=1e-15)
    assert min_shell == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(fsc[1:][np.isfinite(fsc[1:])], 1.0, atol=1e-15)
    assert auc <= 1.0
    assert min_shell <= 1.0


@pytest.mark.parametrize("volume_size", [7, 8])
def test_accumulator_fsc_uses_x_half_hermitian_multiplicity(volume_size):
    shape = (volume_size, volume_size, volume_size // 2 + 1)
    a = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    center = volume_size // 2
    # Both samples occupy shell 1.  The kz=1 pair represents +/-kz and has
    # multiplicity two; the kz=0 sample is self-conjugate and has weight one.
    a[center, center, 1] = 1.0
    b[center, center, 1] = 1.0
    a[center + 1, center, 0] = 1.0
    b[center + 1, center, 0] = -1.0

    fsc = validator._accumulator_fsc_curve(a, b, (volume_size,) * 3)

    assert fsc[1] == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_even_x_half_nyquist_plane_has_unit_multiplicity():
    volume_size = 8
    shape = (volume_size, volume_size, volume_size // 2 + 1)
    a = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    center = volume_size // 2
    # Shell 4 contains kz=Nyquist (unit multiplicity) and kx=0,kz=4.
    # Pair it with kx=4,kz=0 (also unit multiplicity): opposite signs cancel.
    a[center, center, volume_size // 2] = 1.0
    b[center, center, volume_size // 2] = 1.0
    a[0, center, 0] = 1.0
    b[0, center, 0] = -1.0

    fsc = validator._accumulator_fsc_curve(a, b, (volume_size,) * 3)

    assert fsc[volume_size // 2] == pytest.approx(0.0, abs=1e-15)


def test_normalized_fsc_auc_excludes_dc_and_integrates_shell_axis():
    fsc = np.asarray([100.0, 0.0, 1.0, 0.0], dtype=np.float64)
    assert validator._normalized_fsc_auc(fsc) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("key", "canonical rotation keys"),
        ("pixel", "canonical dense pixels"),
        ("digest", "digest mismatch"),
        ("sequence", "begin at zero"),
        ("flags", "exactly one primary"),
        ("source", "source values do not close"),
        ("zero_contributor", "exact positive-weight companion rows"),
        ("orientation_early", "orientation-fold flag"),
        ("inertness", "accumulator shadow is not bitwise equal"),
        ("panel_dtype", "must be complex64"),
        ("topology", "identity/topology mismatch"),
        ("candidate_dtype", "candidate_combined_scores.*dtype"),
        ("raw_shape", "raw_source_shape does not match"),
        ("contribution_identity", "stack-path/image identity mismatch"),
        ("shadow_gate", "did not certify exact score agreement"),
    ],
)
def test_corrupt_signature_or_panel_fails_closed(tmp_path, corruption, message):
    signature, panel = _artifacts(tmp_path, corruption)
    with pytest.raises(ValueError, match=message):
        validator.main([str(signature), "--panel-native", str(panel)])
