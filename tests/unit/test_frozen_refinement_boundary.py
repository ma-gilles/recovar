from __future__ import annotations

import hashlib

import numpy as np
import pytest

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_MANIFEST,
    FROZEN_BOUNDARY_SCHEMA,
    load_frozen_refinement_boundary,
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


def test_frozen_boundary_loader_round_trips_primitive_state(tmp_path):
    _write_boundary(tmp_path)

    boundary = load_frozen_refinement_boundary(tmp_path)

    assert boundary.completed_relion_iteration == 2
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
