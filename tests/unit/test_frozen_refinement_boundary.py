from __future__ import annotations

import hashlib

import numpy as np
import pytest

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
    FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
    FROZEN_BOUNDARY_MANIFEST,
    FROZEN_BOUNDARY_SCHEMA,
    FROZEN_BOUNDARY_SCHEMA_V3,
    V3_REQUIRED_SOURCE_NAMES,
    load_frozen_refinement_boundary,
    validate_fixed_diagnostic_boundary_runtime_config,
    validate_fixed_diagnostic_boundary_sampling_state,
    verify_fixed_diagnostic_boundary_sources,
    v3_source_role,
)


def _write_boundary(root, **overrides):
    root.mkdir(exist_ok=True)
    values = {
        "schema": np.asarray(FROZEN_BOUNDARY_SCHEMA),
        "completed_relion_iteration": np.int32(2),
        "volume_shape": np.asarray([2, 2, 2], dtype=np.int32),
        "current_size": np.int32(92),
        "healpix_order": np.int32(3),
        "relion_incr_size": np.int32(10),
        "has_high_fsc_at_limit": np.bool_(False),
        "half1_mean_ft": np.arange(8, dtype=np.float32).astype(np.complex64),
        "half2_mean_ft": (np.arange(8, dtype=np.float32) + 1j).astype(np.complex64),
        "mean_variance": np.ones(8, dtype=np.float32),
        "half1_noise_radial": np.asarray([1.0, 2.0], dtype=np.float64),
        "half2_noise_radial": np.asarray([1.5, 2.5], dtype=np.float64),
        "fsc": np.asarray([1.0, 0.5], dtype=np.float32),
        "ave_pmax": np.float64(0.25),
        "half1_previous_best_rotation_eulers": np.zeros((2, 3), dtype=np.float32),
        "half2_previous_best_rotation_eulers": np.ones((2, 3), dtype=np.float32),
        "half1_previous_best_translations": np.zeros((2, 2), dtype=np.float32),
        "half2_previous_best_translations": np.ones((2, 2), dtype=np.float32),
        "half1_image_corrections": np.asarray([0.9, 1.1], dtype=np.float32),
        "half2_image_corrections": np.asarray([0.8, 1.2], dtype=np.float32),
        "half1_scale_corrections": np.ones(2, dtype=np.float32),
        "half2_scale_corrections": np.ones(2, dtype=np.float32),
        "half1_direction_prior": np.full(768, 1.0 / 768.0, dtype=np.float32),
        "half2_direction_prior": np.full(768, 1.0 / 768.0, dtype=np.float32),
        "half1_translation_sigma_angstrom": np.float64(16.8),
        "half2_translation_sigma_angstrom": np.float64(17.0),
        "half1_image_name": np.asarray(["1@a.mrcs", "2@a.mrcs"]),
        "half2_image_name": np.asarray(["3@a.mrcs", "4@a.mrcs"]),
        "half1_source_row": np.asarray([0, 2], dtype=np.int64),
        "half2_source_row": np.asarray([1, 3], dtype=np.int64),
        "half1_random_subset": np.asarray([1, 1], dtype=np.int8),
        "half2_random_subset": np.asarray([2, 2], dtype=np.int8),
        "half1_half_index": np.asarray([0, 0], dtype=np.int8),
        "half2_half_index": np.asarray([1, 1], dtype=np.int8),
        "half1_half_local_index": np.asarray([0, 1], dtype=np.int64),
        "half2_half_local_index": np.asarray([0, 1], dtype=np.int64),
        "state_current_resolution": np.float64(14.97),
        "state_previous_resolution": np.float64(29.94),
        "state_nr_iter_wo_resol_gain": np.int32(0),
        "state_nr_iter_wo_assignment_changes": np.int32(0),
        "state_nr_iter_wo_large_hidden_variable_changes": np.int32(0),
        "state_ave_Pmax": np.float64(0.25),
        "state_current_changes_optimal_orientations": np.float64(9.0),
        "state_current_changes_optimal_offsets_angstrom": np.float64(1.5),
        "state_smallest_changes_optimal_orientations": np.float64(9.0),
        "state_smallest_changes_optimal_offsets_angstrom": np.float64(1.5),
        "state_acc_rot": np.float64(2.0),
        "state_acc_trans": np.float64(1.0),
        "state_has_converged": np.bool_(False),
        "source_job_id": np.int64(12345),
        "source_arm": np.asarray("control"),
        "source_map_serialization": np.asarray("in_memory_complex64"),
        "bitwise_identity_to_original_in_memory_means": np.bool_(True),
        "correction_state_owner": np.asarray("sealed_boundary"),
        "identity_schema": np.asarray("five_field.v1"),
        "source_star_sha256": np.asarray("a" * 64),
        "relion_half_star_sha256": np.asarray("b" * 64),
    }
    values.update(overrides)
    path = root / FROZEN_BOUNDARY_FILENAME
    np.savez(path, **values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{digest}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )
    return path


def _v3_overrides(source_digests=None):
    names = sorted(
        {
            *V3_REQUIRED_SOURCE_NAMES,
            "particle_stack:0",
            "consumer_map:half1:class1",
            "consumer_map:half2:class1",
        }
    )
    digests = source_digests or {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    half1_tau2 = np.zeros(8, dtype=np.float32)
    half2_tau2 = np.full(8, 2.0, dtype=np.float32)
    half1_mean = np.arange(8, dtype=np.float32).astype(np.complex64)
    half2_mean = (np.arange(8, dtype=np.float32) + 1j).astype(np.complex64)
    return {
        "schema": np.asarray(FROZEN_BOUNDARY_SCHEMA_V3),
        "completed_relion_iteration": np.int32(1),
        "consumer_relion_iteration": np.int32(2),
        "half1_mean_ft": half1_mean,
        "half2_mean_ft": half2_mean,
        "mean_variance": np.ones(8, dtype=np.float32),
        "half1_mean_variance": half1_tau2,
        "half2_mean_variance": half2_tau2,
        "source_sha256_names": np.asarray(names),
        "source_sha256_digests": np.asarray([digests[name] for name in names]),
        "source_sha256_roles": np.asarray([v3_source_role(name) for name in names]),
        "sampling_directions_ipix": np.arange(768, dtype=np.int64),
        "sampling_rot_angles_deg": np.linspace(0.0, 359.0, 768, dtype=np.float64),
        "sampling_tilt_angles_deg": np.linspace(1.0, 179.0, 768, dtype=np.float64),
        "sampling_psi_angles_deg": np.arange(48, dtype=np.float64) * 7.5,
        "sampling_translations_x_angstrom": np.arange(29, dtype=np.float64),
        "sampling_translations_y_angstrom": -np.arange(29, dtype=np.float64),
        "sampling_translations_z_angstrom": np.empty(0, dtype=np.float64),
        "sampling_healpix_order": np.int32(3),
        "sampling_healpix_order_original": np.int32(3),
        "sampling_psi_step_deg": np.float64(7.5),
        "sampling_offset_range_angstrom": np.float64(4.9125001430511475),
        "sampling_offset_step_angstrom": np.float64(1.6375000476837158),
        "sampling_perturbation_factor": np.float64(0.5),
        "sampling_random_perturbation": np.float64(0.4052000939846039),
        "sampling_sigma_rot_deg": np.float64(0.0),
        "sampling_sigma_psi_deg": np.float64(0.0),
        "sampling_is_3d": np.bool_(True),
        "sampling_is_3d_trans": np.bool_(False),
        "sampling_point_group": np.int32(202),
        "sampling_point_group_order": np.int32(1),
        "sampling_coarse_size": np.int32(56),
        "sampling_full_size": np.int32(256),
        "config_adaptive_oversampling": np.int32(1),
        "config_diagnostic_arm_id": np.asarray(
            "real10076.k1.physical_it2.reconstructed_projector.v1"
        ),
        "config_max_iter": np.int32(1),
        "config_skip_final_iteration": np.bool_(True),
        "config_init_resolution_angstrom": np.float64(30.0),
        "config_offset_range_pixels": np.float64(3.0),
        "config_offset_step_pixels": np.float64(1.0),
        "config_perturb_factor": np.float64(0.5),
        "config_fsc_threshold": np.float64(1.0 / 7.0),
        "config_jax_enable_x64": np.bool_(True),
        "config_provenance_verification_scope": np.asarray(
            FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE
        ),
        "config_numerical_classification_scope": np.asarray(
            FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE
        ),
        "config_auto_local_healpix_order": np.int32(4),
        "config_max_healpix_order": np.int32(7),
        "config_max_significants": np.int32(-1),
        "config_particle_diameter_angstrom": np.float64(280.0),
        "config_width_mask_edge_px": np.float64(5.0),
        "config_tau2_fudge": np.float64(1.0),
        "config_low_resol_join_halves_angstrom": np.float64(40.0),
        "config_image_batch_size": np.int32(187),
        "config_rotation_block_size": np.int32(8192),
        "config_random_seed": np.int64(20260712),
        "config_perturb_seed": np.int64(20260712),
        "config_n_classes": np.int32(1),
        "config_grid_size": np.int32(256),
        "config_voxel_size_angstrom": np.float64(1.6375000476837158),
        "config_projection_padding_factor": np.int32(2),
        "config_backprojection_padding_factor": np.int32(2),
        "config_do_ctf_correction": np.bool_(True),
        "config_firstiter_cc": np.bool_(True),
        "config_do_norm_correction": np.bool_(True),
        "config_do_scale_correction": np.bool_(False),
        "config_refs_are_ctf_corrected": np.bool_(True),
        "config_disc_type": np.asarray("linear_interp"),
        "config_image_fourier_backend": np.asarray("relion_cuda"),
        "config_local_search_translation_prior_mode": np.asarray("coarse"),
        "config_declared_relion_command_line": np.asarray("relion_refine --continue run_it001_optimiser.star"),
        "config_declared_relion_base_git_commit": np.asarray("1" * 40),
        "config_recovar_git_commit": np.asarray("2" * 40),
        "config_declared_relion_build_id": np.asarray("relion-5.0-test"),
        "config_projector_boundary_kind": np.asarray("reconstructed-projector boundary"),
        "config_replay_prefix": np.asarray("run"),
        "source_map_serialization": np.asarray(
            "captured_relion_iref_transformed_to_complex64"
        ),
        "bitwise_identity_to_original_in_memory_means": np.bool_(False),
        "map_transform_id": np.asarray("relion_iref_to_recovar_complex64.v1"),
        "half1_captured_iref_sha256": np.asarray("c" * 64),
        "half2_captured_iref_sha256": np.asarray("d" * 64),
        "half1_transformed_mean_sha256": np.asarray(
            hashlib.sha256(half1_mean.tobytes(order="C")).hexdigest()
        ),
        "half2_transformed_mean_sha256": np.asarray(
            hashlib.sha256(half2_mean.tobytes(order="C")).hexdigest()
        ),
    }


def _write_v3_boundary(root, **overrides):
    values = _v3_overrides()
    values.update(overrides)
    return _write_boundary(root, **values)


def test_frozen_boundary_loader_round_trips_primitive_state(tmp_path):
    _write_boundary(tmp_path)

    boundary = load_frozen_refinement_boundary(tmp_path)

    assert boundary.completed_relion_iteration == 2
    assert boundary.volume_shape == (2, 2, 2)
    assert boundary.current_size == 92
    assert boundary.means[0].dtype == np.complex64
    np.testing.assert_array_equal(
        boundary.image_corrections[0],
        np.asarray([0.9, 1.1], dtype=np.float32),
    )
    assert boundary.direction_prior_per_half[0].shape == (768,)
    assert boundary.translation_sigma_angstrom_per_half == pytest.approx((16.8, 17.0))
    np.testing.assert_array_equal(boundary.source_rows_per_half[0], [0, 2])
    assert set(boundary.refinement_state_fields) == {
        "current_resolution",
        "previous_resolution",
        "nr_iter_wo_resol_gain",
        "nr_iter_wo_assignment_changes",
        "nr_iter_wo_large_hidden_variable_changes",
        "ave_Pmax",
        "current_changes_optimal_orientations",
        "current_changes_optimal_offsets_angstrom",
        "smallest_changes_optimal_orientations",
        "smallest_changes_optimal_offsets_angstrom",
        "acc_rot",
        "acc_trans",
        "has_converged",
    }
    assert boundary.refinement_state_fields["current_resolution"] == pytest.approx(14.97)
    assert boundary.refinement_state_fields["has_converged"] is False
    assert len(boundary.boundary_sha256) == 64
    assert len(boundary.source_manifest_sha256) == 64
    assert boundary.source_job_id == 12345
    assert boundary.bitwise_identity_to_original_in_memory_means is True
    assert boundary.source_star_sha256 == "a" * 64
    assert boundary.fixed_diagnostic_arm is False


def test_v3_round_trips_fixed_arm_sources_sampling_config_and_per_half_tau2(tmp_path):
    _write_v3_boundary(tmp_path)

    boundary = load_frozen_refinement_boundary(tmp_path)

    assert boundary.fixed_diagnostic_arm is True
    assert set(boundary.source_sha256) == {
        *V3_REQUIRED_SOURCE_NAMES,
        "particle_stack:0",
        "consumer_map:half1:class1",
        "consumer_map:half2:class1",
    }
    assert boundary.source_roles["particle_stack:0"] == "input_particle_stack_bytes"
    assert boundary.map_lineage["map_transform_id"] == "relion_iref_to_recovar_complex64.v1"
    assert boundary.sampling_state["directions_ipix"].dtype == np.int64
    assert boundary.runtime_config["image_fourier_backend"] == "relion_cuda"
    assert not np.array_equal(
        boundary.mean_variance_per_half[0],
        boundary.mean_variance_per_half[1],
    )


def test_v3_rejects_collapsed_unequal_half_tau2(tmp_path):
    _write_v3_boundary(tmp_path, mean_variance=np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="explicit float32 average"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_accepts_equal_current_and_coarse_sizes_and_restricted_direction_ids(tmp_path):
    direction_ids = np.asarray([7, 19, 503], dtype=np.int64)
    _write_v3_boundary(
        tmp_path,
        current_size=np.int32(56),
        sampling_coarse_size=np.int32(56),
        sampling_directions_ipix=direction_ids,
        sampling_rot_angles_deg=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        sampling_tilt_angles_deg=np.asarray([40.0, 50.0, 60.0], dtype=np.float64),
    )

    boundary = load_frozen_refinement_boundary(tmp_path)

    assert boundary.current_size == 56
    np.testing.assert_array_equal(boundary.sampling_state["directions_ipix"], direction_ids)


def test_v3_rejects_captured_active_healpix_order_drift(tmp_path):
    _write_v3_boundary(tmp_path, sampling_healpix_order=np.int32(2))

    with pytest.raises(ValueError, match="captured/current HEALPix orders differ"):
        load_frozen_refinement_boundary(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sampling_is_3d", np.bool_(False), "requires captured 3-D orientations"),
        ("sampling_is_3d_trans", np.bool_(True), "requires captured 2-D translations"),
        ("sampling_point_group", np.int32(0), "C1 point-group semantics"),
        ("sampling_point_group_order", np.int32(2), "C1 point-group semantics"),
        ("sampling_psi_step_deg", np.float64(15.0), "psi count/step"),
        (
            "sampling_psi_angles_deg",
            np.roll(np.arange(48, dtype=np.float64) * 7.5, 1),
            "canonical psi-index order",
        ),
    ],
)
def test_v3_rejects_sampling_semantics_the_fixed_scorer_would_reinterpret(
    tmp_path,
    field,
    value,
    message,
):
    _write_v3_boundary(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=message):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_false_transformed_map_lineage(tmp_path):
    _write_v3_boundary(tmp_path, half1_transformed_mean_sha256=np.asarray("e" * 64))

    with pytest.raises(ValueError, match="does not bind half1_mean_ft"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_v2_bitwise_identity_claim(tmp_path):
    _write_v3_boundary(tmp_path, bitwise_identity_to_original_in_memory_means=np.bool_(True))

    with pytest.raises(ValueError, match="must not claim bitwise identity"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_wrong_source_role(tmp_path):
    values = _v3_overrides()
    roles = values["source_sha256_roles"].copy()
    roles[0] = "wrong_role"
    _write_v3_boundary(tmp_path, source_sha256_roles=roles)

    with pytest.raises(ValueError, match="role must equal"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_noncontiguous_particle_stack_names(tmp_path):
    values = _v3_overrides()
    names = values["source_sha256_names"].copy()
    names[names == "particle_stack:0"] = "particle_stack:1"
    _write_v3_boundary(tmp_path, source_sha256_names=names)

    with pytest.raises(ValueError, match="source closure failed"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_inferred_consumer_iteration(tmp_path):
    _write_v3_boundary(tmp_path, consumer_relion_iteration=np.int32(3))

    with pytest.raises(ValueError, match="consumer_relion_iteration"):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_rejects_exact_projector_claim_without_direct_capture_consumption(tmp_path):
    _write_v3_boundary(
        tmp_path,
        config_projector_boundary_kind=np.asarray("exact captured-projector boundary"),
    )

    with pytest.raises(ValueError, match="reconstructed-projector boundary"):
        load_frozen_refinement_boundary(tmp_path)


@pytest.mark.parametrize(
    "missing_source",
    sorted(
        {
            *V3_REQUIRED_SOURCE_NAMES,
            "particle_stack:0",
            "consumer_map:half1:class1",
            "consumer_map:half2:class1",
        }
    ),
)
def test_v3_source_verifier_requires_every_source_class(tmp_path, missing_source):
    source_paths = {}
    source_digests = {}
    for name in _v3_overrides()["source_sha256_names"]:
        path = tmp_path / name
        path.write_bytes(name.encode())
        source_paths[name] = path
        source_digests[name] = hashlib.sha256(name.encode()).hexdigest()
    _write_v3_boundary(tmp_path / "boundary", **_v3_overrides(source_digests))
    boundary = load_frozen_refinement_boundary(tmp_path / "boundary")
    source_paths.pop(missing_source)

    with pytest.raises(ValueError, match="source path closure"):
        verify_fixed_diagnostic_boundary_sources(boundary, source_paths)


def test_v3_source_verifier_rejects_tampering(tmp_path):
    source_paths = {}
    source_digests = {}
    for name in _v3_overrides()["source_sha256_names"]:
        path = tmp_path / name
        path.write_bytes(name.encode())
        source_paths[name] = path
        source_digests[name] = hashlib.sha256(name.encode()).hexdigest()
    _write_v3_boundary(tmp_path / "boundary", **_v3_overrides(source_digests))
    boundary = load_frozen_refinement_boundary(tmp_path / "boundary")
    source_paths["consumer_validation_sampling"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="consumer_validation_sampling SHA-256 mismatch"):
        verify_fixed_diagnostic_boundary_sources(boundary, source_paths)


def test_v3_source_verifier_rejects_consumer_data_tampering(tmp_path):
    source_paths = {}
    source_digests = {}
    for name in _v3_overrides()["source_sha256_names"]:
        path = tmp_path / name.replace(":", "_")
        path.write_bytes(name.encode())
        source_paths[name] = path
        source_digests[name] = hashlib.sha256(name.encode()).hexdigest()
    _write_v3_boundary(tmp_path / "boundary", **_v3_overrides(source_digests))
    boundary = load_frozen_refinement_boundary(tmp_path / "boundary")
    source_paths["consumer_validation_data"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="consumer_validation_data SHA-256 mismatch"):
        verify_fixed_diagnostic_boundary_sources(boundary, source_paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_directions_ipix", np.arange(768, dtype=np.int32)),
        ("sampling_rot_angles_deg", np.zeros(768, dtype=np.float32)),
        ("sampling_psi_angles_deg", np.zeros(0, dtype=np.float64)),
        ("sampling_translations_y_angstrom", np.zeros(28, dtype=np.float64)),
        ("half2_mean_variance", np.ones(7, dtype=np.float32)),
        ("config_random_seed", np.int32(20260712)),
        ("config_disc_type", np.asarray(b"linear_interp")),
    ],
)
def test_v3_rejects_wrong_exact_dtype_or_shape(tmp_path, field, value):
    _write_v3_boundary(tmp_path, **{field: value})

    with pytest.raises(ValueError):
        load_frozen_refinement_boundary(tmp_path)


def test_v3_runtime_config_and_sampling_mismatch_fail_closed(tmp_path):
    _write_v3_boundary(tmp_path)
    boundary = load_frozen_refinement_boundary(tmp_path)
    observed_config = dict(boundary.runtime_config)
    observed_config["image_batch_size"] += 1
    with pytest.raises(ValueError, match="runtime config mismatch"):
        validate_fixed_diagnostic_boundary_runtime_config(boundary, observed_config)

    observed_sampling = dict(boundary.sampling_state)
    observed_sampling["random_perturbation"] += 1e-12
    with pytest.raises(ValueError, match="sampling state mismatch"):
        validate_fixed_diagnostic_boundary_sampling_state(boundary, observed_sampling)


def test_v2_cannot_support_complete_claim(tmp_path):
    _write_boundary(tmp_path)
    boundary = load_frozen_refinement_boundary(tmp_path)
    with pytest.raises(ValueError, match="cannot support the fixed schema-v3 diagnostic arm"):
        verify_fixed_diagnostic_boundary_sources(boundary, {})


def test_frozen_boundary_loader_rejects_modified_payload(tmp_path):
    path = _write_boundary(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"modified")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_duplicate_half_identity(tmp_path):
    _write_boundary(
        tmp_path,
        half2_image_name=np.asarray(["2@a.mrcs", "4@a.mrcs"]),
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_duplicate_source_row_across_halves(tmp_path):
    _write_boundary(
        tmp_path,
        half2_source_row=np.asarray([0, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="source rows must be globally unique"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_correction_row_mismatch(tmp_path):
    _write_boundary(
        tmp_path,
        half1_image_corrections=np.ones(3, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="image-correction row count"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_nonpositive_correction(tmp_path):
    _write_boundary(
        tmp_path,
        half1_image_corrections=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="image corrections must be positive"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_wrong_direction_prior_shape(tmp_path):
    _write_boundary(tmp_path, half2_direction_prior=np.ones(12, dtype=np.float32))

    with pytest.raises(ValueError, match="direction-prior shape"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_nonpositive_translation_sigma(tmp_path):
    _write_boundary(tmp_path, half1_translation_sigma_angstrom=np.float64(0.0))

    with pytest.raises(ValueError, match="translation sigma"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_five_field_identity_mismatch(tmp_path):
    _write_boundary(
        tmp_path,
        half2_half_local_index=np.asarray([1, 0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="local identity order"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_out_of_range_fsc(tmp_path):
    _write_boundary(tmp_path, fsc=np.asarray([1.0, 1.01], dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_nonfinite_state(tmp_path):
    _write_boundary(tmp_path, state_acc_rot=np.float64(np.inf))

    with pytest.raises(ValueError, match="must be finite"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_incomplete_state(tmp_path):
    path = _write_boundary(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files if key != "state_acc_rot"}
    np.savez(path, **values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{digest}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required schema-v2 keys"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_unknown_payload_key(tmp_path):
    _write_boundary(tmp_path, unvalidated_extra=np.asarray([object()], dtype=object))

    with pytest.raises(ValueError, match="unknown schema-v2 keys"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_string_schedule_scalar(tmp_path):
    _write_boundary(tmp_path, current_size=np.asarray("92"))

    with pytest.raises(ValueError, match="current_size has dtype"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_complex128_map_before_cast(tmp_path):
    _write_boundary(tmp_path, half1_mean_ft=np.arange(8, dtype=np.complex128))

    with pytest.raises(ValueError, match="half1_mean_ft has dtype complex128; expected complex64"):
        load_frozen_refinement_boundary(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_relion_iteration", np.int64(2)),
        ("ave_pmax", np.float32(0.25)),
        ("source_job_id", np.int32(12345)),
    ],
)
def test_frozen_boundary_loader_rejects_wrong_scalar_dtype(tmp_path, field, value):
    _write_boundary(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=rf"{field} has dtype"):
        load_frozen_refinement_boundary(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("half1_source_row", np.asarray([0.0, 2.5], dtype=np.float64)),
        ("half1_random_subset", np.asarray([1, 1], dtype=np.int64)),
        ("half1_half_index", np.asarray([0, 0], dtype=np.int32)),
        ("half1_half_local_index", np.asarray([0, 1], dtype=np.int32)),
    ],
)
def test_frozen_boundary_loader_rejects_wrong_identity_dtype(tmp_path, field, value):
    _write_boundary(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=rf"{field} has dtype"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_nonunicode_image_identity(tmp_path):
    _write_boundary(tmp_path, half1_image_name=np.asarray([b"1@a.mrcs", b"2@a.mrcs"]))

    with pytest.raises(ValueError, match="image names must be Unicode strings"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_requires_complete_provenance(tmp_path):
    path = _write_boundary(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files if key != "source_star_sha256"}
    np.savez(path, **values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{digest}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required schema-v2 keys"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_requires_bitwise_original_maps(tmp_path):
    _write_boundary(
        tmp_path,
        bitwise_identity_to_original_in_memory_means=np.bool_(False),
    )

    with pytest.raises(ValueError, match="bitwise-identical"):
        load_frozen_refinement_boundary(tmp_path)


@pytest.mark.parametrize(
    "field",
    ["source_map_serialization", "correction_state_owner", "identity_schema"],
)
def test_frozen_boundary_loader_rejects_false_provenance_semantics(tmp_path, field):
    _write_boundary(tmp_path, **{field: np.asarray("plausible_but_unverified")})

    with pytest.raises(ValueError, match=rf"provenance scalar {field} must equal"):
        load_frozen_refinement_boundary(tmp_path)
