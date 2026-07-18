from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_MANIFEST,
    FROZEN_BOUNDARY_SCHEMA,
)
from recovar.em.dense_single_volume.frozen_boundary_variant import (
    FROZEN_BOUNDARY_VARIANT_ATTESTATION,
    FROZEN_BOUNDARY_VARIANT_MANIFEST,
    build_frozen_boundary_variant,
    validate_frozen_boundary_variant,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base(root: Path) -> Path:
    root.mkdir()
    values = {
        "schema": np.asarray(FROZEN_BOUNDARY_SCHEMA),
        "completed_relion_iteration": np.int32(1),
        "volume_shape": np.asarray([2, 2, 2], dtype=np.int32),
        "current_size": np.int32(56),
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
        "half1_translation_sigma_angstrom": np.float64(6.3),
        "half2_translation_sigma_angstrom": np.float64(6.4),
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
        "state_current_resolution": np.float64(30.2),
        "state_previous_resolution": np.float64(30.2),
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
        "source_job_id": np.int64(123),
        "source_arm": np.asarray("all_relion_control"),
        "source_map_serialization": np.asarray("in_memory_complex64"),
        "bitwise_identity_to_original_in_memory_means": np.bool_(True),
        "correction_state_owner": np.asarray("sealed_boundary"),
        "identity_schema": np.asarray("five_field.v1"),
        "source_star_sha256": np.asarray("a" * 64),
        "relion_half_star_sha256": np.asarray("b" * 64),
    }
    boundary = root / FROZEN_BOUNDARY_FILENAME
    np.savez(boundary, **values)
    (root / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{_sha256(boundary)}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )
    return boundary


def _write_source(root: Path, *, tau2=None, commit="c" * 40, dirty=0, iteration=0):
    run = root / "run"
    intermediates = run / "intermediates"
    intermediates.mkdir(parents=True)
    source = intermediates / f"it{iteration:03d}_tau2.npy"
    np.save(
        source,
        np.arange(2, 10, dtype=np.float32) if tau2 is None else tau2,
    )
    results = run / "refinement_results.npz"
    np.savez(
        results,
        git_commit=np.asarray(commit),
        git_dirty_count=np.int64(dirty),
    )
    return source, results


def _build(tmp_path: Path):
    base = tmp_path / "base"
    _write_base(base)
    source, results = _write_source(tmp_path / "source")
    output = tmp_path / "variant"
    attestation = build_frozen_boundary_variant(
        base_boundary_dir=base,
        output_dir=output,
        component="tau2",
        component_source=source,
        source_results=results,
    )
    return base, source, results, output, attestation


def test_tau2_variant_changes_only_mean_variance(tmp_path):
    base, source, _, output, attestation = _build(tmp_path)

    assert attestation["actual_changed_payload_keys"] == ["mean_variance"]
    with np.load(base / FROZEN_BOUNDARY_FILENAME, allow_pickle=False) as control:
        with np.load(output / FROZEN_BOUNDARY_FILENAME, allow_pickle=False) as variant:
            assert control.files == variant.files
            for key in control.files:
                if key == "mean_variance":
                    np.testing.assert_array_equal(variant[key], np.load(source, allow_pickle=False))
                else:
                    assert control[key].dtype == variant[key].dtype
                    assert control[key].shape == variant[key].shape
                    np.testing.assert_array_equal(control[key], variant[key])


def test_tau2_variant_rejects_nonsealed_base(tmp_path):
    base = tmp_path / "base"
    boundary = _write_base(base)
    with boundary.open("ab") as stream:
        stream.write(b"tampered")
    source, results = _write_source(tmp_path / "source")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_frozen_boundary_variant(
            base_boundary_dir=base,
            output_dir=tmp_path / "variant",
            component="tau2",
            component_source=source,
            source_results=results,
        )


@pytest.mark.parametrize(
    ("tau2", "message"),
    [
        (np.ones(8, dtype=np.float64), "dtype float64"),
        (np.ones(7, dtype=np.float32), "shape"),
        (np.asarray([1, 2, 3, 4, 5, 6, 7, np.nan], dtype=np.float32), "finite"),
        (np.asarray([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.float32), "nonnegative"),
    ],
)
def test_tau2_variant_rejects_invalid_exact_source(tmp_path, tau2, message):
    base = tmp_path / "base"
    _write_base(base)
    source, results = _write_source(tmp_path / "source", tau2=tau2)

    with pytest.raises(ValueError, match=message):
        build_frozen_boundary_variant(
            base_boundary_dir=base,
            output_dir=tmp_path / "variant",
            component="tau2",
            component_source=source,
            source_results=results,
        )


def test_tau2_variant_rejects_clean_control_noop(tmp_path):
    base = tmp_path / "base"
    _write_base(base)
    source, results = _write_source(tmp_path / "source", tau2=np.ones(8, dtype=np.float32))

    with pytest.raises(ValueError, match=r"actual=\[\]"):
        build_frozen_boundary_variant(
            base_boundary_dir=base,
            output_dir=tmp_path / "variant",
            component="tau2",
            component_source=source,
            source_results=results,
        )


def test_tau2_variant_rejects_dirty_source_provenance(tmp_path):
    base = tmp_path / "base"
    _write_base(base)
    source, results = _write_source(tmp_path / "source", dirty=1)

    with pytest.raises(ValueError, match="clean worktree"):
        build_frozen_boundary_variant(
            base_boundary_dir=base,
            output_dir=tmp_path / "variant",
            component="tau2",
            component_source=source,
            source_results=results,
        )


def test_tau2_variant_rejects_wrong_iteration_source(tmp_path):
    base = tmp_path / "base"
    _write_base(base)
    source, results = _write_source(tmp_path / "source", iteration=1)

    with pytest.raises(ValueError, match="source iteration does not match"):
        build_frozen_boundary_variant(
            base_boundary_dir=base,
            output_dir=tmp_path / "variant",
            component="tau2",
            component_source=source,
            source_results=results,
        )


def test_variant_validator_rejects_tampered_component_source(tmp_path):
    _, source, _, output, _ = _build(tmp_path)
    np.save(source, np.full(8, 3.0, dtype=np.float32))

    with pytest.raises(ValueError, match="component source SHA-256"):
        validate_frozen_boundary_variant(output)


def test_variant_validator_rejects_tampered_attestation(tmp_path):
    _, _, _, output, _ = _build(tmp_path)
    attestation = output / FROZEN_BOUNDARY_VARIANT_ATTESTATION
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["component"] = "noise"
    attestation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="attestation SHA-256"):
        validate_frozen_boundary_variant(output)


def test_variant_validator_rejects_second_payload_change(tmp_path):
    _, _, _, output, _ = _build(tmp_path)
    boundary = output / FROZEN_BOUNDARY_FILENAME
    with np.load(boundary, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    values["fsc"] = values["fsc"].copy()
    values["fsc"][1] = np.float32(0.4)
    np.savez(boundary, **values)
    (output / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{_sha256(boundary)}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="variant boundary SHA-256"):
        validate_frozen_boundary_variant(output)


def test_variant_attestation_manifest_is_single_file_and_sealed(tmp_path):
    _, _, _, output, _ = _build(tmp_path)
    manifest = output / FROZEN_BOUNDARY_VARIANT_MANIFEST
    fields = manifest.read_text(encoding="utf-8").strip().split(maxsplit=1)

    assert fields[1] == FROZEN_BOUNDARY_VARIANT_ATTESTATION
    assert fields[0] == _sha256(output / FROZEN_BOUNDARY_VARIANT_ATTESTATION)
