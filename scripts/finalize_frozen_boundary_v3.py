#!/usr/bin/env python3
"""Finalize the fixed real-10076 schema-v3 diagnostic boundary.

This intentionally upgrades an already validated schema-v2 state bundle: v2
owns the compact particle/correction/convergence census, while the schema-4
live capture supplies exact per-half model state and sampling.  Every external
source is supplied through a semantic source-path JSON table and byte-hashed.
The reconstructed-projector output is diagnostic-only. It does not claim
identity to RELION's full in-memory physical iteration and is never a
production restart checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import starfile

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM,
    FROZEN_BOUNDARY_FIXED_MATH_ENVIRONMENT_CONTRACT,
    FROZEN_BOUNDARY_MANIFEST,
    FROZEN_BOUNDARY_SCHEMA_V2,
    FROZEN_BOUNDARY_SCHEMA_V3,
    V3_REQUIRED_FIXED_SOURCE_NAMES,
    _V3_MAP_TRANSFORM_ID,
    _V3_SCALAR_DTYPES,
    load_frozen_refinement_boundary,
    v3_source_role,
)
from recovar.utils import helpers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class _ValidatedCapture:
    """A capture whose every read is closed over a validated manifest row."""

    manifest_path: Path
    member_sha256: dict[Path, str]

    @property
    def capture_dir(self) -> Path:
        return self.manifest_path.parent

    def path(self, iteration: int, rank: int, name: str) -> Path:
        path = (
            self.capture_dir
            / f"state_iter{iteration}_rank{rank}_device0_class0_{name}.bin"
        ).resolve()
        if path not in self.member_sha256:
            raise ValueError(f"capture operand is not bound by the live manifest: {path}")
        return path

    def scalar(self, iteration: int, rank: int, name: str) -> float:
        path = self.path(iteration, rank, name)
        value = np.fromfile(path, dtype="<f8")
        if value.shape != (1,) or not np.isfinite(value[0]):
            raise ValueError(f"invalid captured scalar: {path}")
        return float(value[0])

    def vector(
        self,
        iteration: int,
        rank: int,
        name: str,
        dtype="<f8",
    ) -> np.ndarray:
        path = self.path(iteration, rank, name)
        itemsize = np.dtype(dtype).itemsize
        if path.stat().st_size < 8 or (path.stat().st_size - 8) % itemsize:
            raise ValueError(f"invalid captured vector byte count: {path}")
        with path.open("rb") as stream:
            count = np.fromfile(stream, dtype="<i8", count=1)
            value = np.fromfile(stream, dtype=dtype)
        if count.shape != (1,) or int(count[0]) != value.size:
            raise ValueError(f"invalid captured vector: {path}")
        return value


def _validate_capture_manifest(path: Path) -> _ValidatedCapture:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty live-capture manifest: {path}")
    members: dict[Path, str] = {}
    for row in rows:
        digest, filename = row.split(maxsplit=1)
        source = Path(filename.lstrip("*"))
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if source in members:
            raise ValueError(f"duplicate live-capture manifest member: {source}")
        if not source.is_file() or _sha256(source) != digest:
            raise ValueError(f"live-capture manifest verification failed: {source}")
        members[source] = digest
    return _ValidatedCapture(path.resolve(), members)


def _load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_source_paths(source_paths: dict[str, Path], *, live_manifest: Path) -> None:
    names = set(source_paths)
    stack_names = [name for name in names if name.startswith("particle_stack:")]
    map_names = {name for name in names if name.startswith("consumer_map:")}
    missing = V3_REQUIRED_FIXED_SOURCE_NAMES - names
    for name in stack_names:
        suffix = name.removeprefix("particle_stack:")
        if not suffix.isdigit():
            raise ValueError(f"invalid v3 particle-stack source name: {name}")
    stack_names.sort(key=lambda name: int(name.removeprefix("particle_stack:")))
    stack_indices = [int(name.removeprefix("particle_stack:")) for name in stack_names]
    expected_stack_names = [f"particle_stack:{index}" for index in range(len(stack_names))]
    expected_map_names = {
        "consumer_map:half1:class1",
        "consumer_map:half2:class1",
    }
    expected_names = V3_REQUIRED_FIXED_SOURCE_NAMES | set(expected_stack_names) | expected_map_names
    unknown = names - expected_names
    if (
        missing
        or not stack_names
        or stack_indices != list(range(len(stack_names)))
        or stack_names != expected_stack_names
        or map_names != expected_map_names
        or unknown
    ):
        raise ValueError(
            f"v3 source table incomplete: missing={sorted(missing)} "
            f"stacks={stack_names} expected_stacks={expected_stack_names} "
            f"maps={sorted(map_names)} unknown={sorted(unknown)}"
        )
    if source_paths["live_capture_manifest"].resolve() != live_manifest.resolve():
        raise ValueError("source table live_capture_manifest differs from --capture-manifest")
    for name, path in source_paths.items():
        v3_source_role(name)
        if not path.is_file():
            raise ValueError(f"v3 source is not a file: {name}={path}")

    particles, _ = _particle_tables(source_paths["particle_star"])
    if "rlnImageName" not in particles:
        raise ValueError("particle STAR lacks rlnImageName")
    referenced_stacks = set()
    for image_name in particles["rlnImageName"]:
        _, stack_text = _image_identity(image_name)
        stack_path = Path(stack_text).expanduser()
        if not stack_path.is_absolute():
            stack_path = source_paths["particle_star"].parent / stack_path
        stack_path = stack_path.resolve()
        referenced_stacks.add(stack_path)
    referenced_stacks = sorted(referenced_stacks, key=str)
    sealed_stacks = [source_paths[name].resolve() for name in expected_stack_names]
    if referenced_stacks != sealed_stacks:
        raise ValueError(
            "particle STAR stack order differs from sealed particle_stack sources: "
            f"referenced={referenced_stacks} sealed={sealed_stacks}"
        )

    for half in (1, 2):
        model_path = source_paths[f"consumer_validation_half{half}_model"]
        document = starfile.read(model_path, always_dict=True)
        classes = document.get("model_classes")
        if classes is None or "rlnReferenceImage" not in classes or len(classes) != 1:
            raise ValueError(f"consumer model lacks exactly one K=1 map: {model_path}")
        referenced = Path(str(classes["rlnReferenceImage"].iloc[0]))
        if not referenced.is_absolute():
            referenced = model_path.parent / referenced
        sealed = source_paths[f"consumer_map:half{half}:class1"]
        if referenced.resolve() != sealed.resolve():
            raise ValueError(
                f"consumer half-{half} model map differs from sealed map: "
                f"referenced={referenced.resolve()} sealed={sealed.resolve()}"
            )


def _runtime_payload(runtime_config: dict) -> dict[str, np.ndarray]:
    expected_names = {
        key.removeprefix("config_")
        for key in _V3_SCALAR_DTYPES
        if key.startswith("config_")
    }
    if set(runtime_config) != expected_names:
        raise ValueError(
            "runtime config closure failed: "
            f"missing={sorted(expected_names - set(runtime_config))}, "
            f"unknown={sorted(set(runtime_config) - expected_names)}"
        )
    payload = {}
    for field, dtype in _V3_SCALAR_DTYPES.items():
        if not field.startswith("config_"):
            continue
        value = runtime_config[field.removeprefix("config_")]
        if dtype is None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"runtime config {field} must be a nonempty string")
            payload[field] = np.asarray(value)
        else:
            expected = np.dtype(dtype)
            if expected == np.dtype(np.bool_) and type(value) is not bool:
                raise ValueError(f"runtime config {field} must be a JSON boolean")
            if np.issubdtype(expected, np.integer) and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"runtime config {field} must be a JSON integer")
            if np.issubdtype(expected, np.floating) and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ValueError(f"runtime config {field} must be a JSON number")
            payload[field] = np.asarray(value, dtype=dtype)
    return payload


def _validate_runtime_environment_manifest(path: Path, runtime_config: dict) -> None:
    expected = {
        "schema": "recovar.em.runtime_environment.v1",
        "diagnostic_arm_id": runtime_config["diagnostic_arm_id"],
        "math_environment_contract": FROZEN_BOUNDARY_FIXED_MATH_ENVIRONMENT_CONTRACT,
        "jax_enable_x64": runtime_config["jax_enable_x64"],
        "provenance_verification_scope": runtime_config["provenance_verification_scope"],
        "numerical_classification_scope": runtime_config["numerical_classification_scope"],
        "declared_relion_command_line": runtime_config["declared_relion_command_line"],
        "declared_relion_base_git_commit": runtime_config["declared_relion_base_git_commit"],
        "declared_relion_build_id": runtime_config["declared_relion_build_id"],
        "recovar_git_commit": runtime_config["recovar_git_commit"],
        "projector_boundary_kind": runtime_config["projector_boundary_kind"],
    }
    if _load_json_object(path) != expected:
        raise ValueError("runtime environment manifest differs from fixed diagnostic arm")


def _validate_recovar_source_manifest(path: Path, runtime_config: dict) -> None:
    expected = {
        "schema": "recovar.em.source_manifest.v1",
        "recovar_git_commit": runtime_config["recovar_git_commit"],
        "worktree_clean": True,
    }
    if _load_json_object(path) != expected:
        raise ValueError("RECOVAR source manifest differs from fixed diagnostic arm")


def _assert_exact(label: str, observed, expected) -> None:
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    if observed_array.dtype.kind in "fc" or expected_array.dtype.kind in "fc":
        equal = np.array_equal(observed_array, expected_array)
    else:
        equal = np.array_equal(observed_array, expected_array)
    if not equal:
        raise ValueError(
            f"exact boundary mismatch for {label}: "
            f"observed_shape={observed_array.shape} expected_shape={expected_array.shape}"
        )


def _shared_capture_scalar(
    capture: _ValidatedCapture,
    iteration: int,
    name: str,
) -> float:
    half1 = capture.scalar(iteration, 1, name)
    half2 = capture.scalar(iteration, 2, name)
    if half1 != half2:
        raise ValueError(f"captured shared scalar differs by half: {name}")
    return half1


def _shared_capture_vector(
    capture: _ValidatedCapture,
    iteration: int,
    name: str,
    dtype="<f8",
) -> np.ndarray:
    half1 = capture.vector(iteration, 1, name, dtype)
    half2 = capture.vector(iteration, 2, name, dtype)
    if not np.array_equal(half1, half2):
        raise ValueError(f"captured shared vector differs by half: {name}")
    return half1


def _image_identity(value: str) -> tuple[int, str]:
    index, separator, stack = str(value).partition("@")
    if separator != "@" or not index.isdigit() or not stack:
        raise ValueError(f"invalid RELION image identity {value!r}")
    return int(index), stack


def _particle_tables(path: Path):
    document = starfile.read(path, always_dict=True)
    particles = document.get("particles")
    if particles is None:
        raise ValueError(f"RELION STAR lacks particles table: {path}")
    return particles.reset_index(drop=True), document.get("optics")


def _star_mapping(path: Path, block: str) -> dict:
    document = starfile.read(path, always_dict=True)
    value = document.get(block)
    if not isinstance(value, dict):
        raise ValueError(f"RELION STAR lacks scalar block {block!r}: {path}")
    return value


def _validate_base_v2_state(
    base,
    *,
    capture: _ValidatedCapture,
    source_paths: dict[str, Path],
    consumer: int,
    grid_size: int,
) -> None:
    """Prove that every v2-owned scorer/state field belongs to this capture."""

    particle_star = source_paths["particle_star"]
    completed_data = source_paths["completed_data"]
    if base.source_star_sha256 != _sha256(particle_star):
        raise ValueError("base v2 particle-STAR provenance differs from v3 source")
    if base.relion_half_star_sha256 != _sha256(source_paths["relion_half_star"]):
        raise ValueError("base v2 RELION half-STAR provenance differs from v3 source")

    fixture, fixture_optics = _particle_tables(particle_star)
    completed, completed_optics = _particle_tables(completed_data)
    fixture_ids = [_image_identity(value) for value in fixture["rlnImageName"]]
    completed_ids = [_image_identity(value) for value in completed["rlnImageName"]]
    if len(set(fixture_ids)) != len(fixture_ids) or len(set(completed_ids)) != len(completed_ids):
        raise ValueError("particle/completed STAR contains duplicate image identities")
    completed_row = {identity: row for row, identity in enumerate(completed_ids)}
    if set(fixture_ids) != set(completed_ids):
        raise ValueError("particle/completed STAR image identity sets differ")
    permutation = np.asarray([completed_row[identity] for identity in fixture_ids], dtype=np.int64)
    if fixture_optics is None or completed_optics is None or not fixture_optics.equals(completed_optics):
        raise ValueError("particle/completed STAR optics tables differ")

    subsets = np.asarray(fixture["rlnRandomSubset"], dtype=np.int8)
    completed_subsets = np.asarray(completed["rlnRandomSubset"], dtype=np.int8)[permutation]
    if not np.array_equal(subsets, completed_subsets) or set(subsets.tolist()) != {1, 2}:
        raise ValueError("particle/completed STAR random-half identities differ")
    half_rows = tuple(np.flatnonzero(subsets == half).astype(np.int64) for half in (1, 2))

    scoring_columns = (
        "rlnDefocusU",
        "rlnDefocusV",
        "rlnDefocusAngle",
        "rlnPhaseShift",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnOriginXAngst",
        "rlnOriginYAngst",
        "rlnNormCorrection",
        "rlnGroupNumber",
        "rlnClassNumber",
    )
    source_order = {}
    for name in scoring_columns:
        if name not in completed:
            raise ValueError(f"completed particle STAR lacks {name}")
        source_order[name] = np.asarray(completed[name])[permutation]
    for name in ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift"):
        _assert_exact(
            f"particle/completed CTF column {name}",
            np.asarray(fixture[name], dtype=np.float64),
            np.asarray(source_order[name], dtype=np.float64),
        )
    if not np.all(np.asarray(source_order["rlnClassNumber"], dtype=np.int64) == 1):
        raise ValueError("fixed schema-v3 diagnostic arm requires K=1 particle state")

    voxel_size = _shared_capture_scalar(capture, consumer, "model_pixel_size")
    eulers = np.stack(
        [np.asarray(source_order[name], dtype=np.float64) for name in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")],
        axis=1,
    )
    translations = np.stack(
        [
            np.asarray(source_order[name], dtype=np.float64) / voxel_size
            for name in ("rlnOriginXAngst", "rlnOriginYAngst")
        ],
        axis=1,
    )
    norm_correction = np.asarray(source_order["rlnNormCorrection"], dtype=np.float64)
    group = np.asarray(source_order["rlnGroupNumber"], dtype=np.int64)
    image_names = np.asarray(fixture["rlnImageName"], dtype=np.str_)

    n4 = float(grid_size) ** 4
    tau2_profile_half1 = capture.vector(consumer, 1, "model_tau2_class")
    inherited_tau2 = np.asarray(
        utils.make_radial_image(
            tau2_profile_half1 * n4,
            tuple(base.volume_shape),
            extend_last_frequency=True,
        ),
        dtype=np.float32,
    ).reshape(-1)
    _assert_exact("base v2 mean_variance", base.mean_variance, inherited_tau2)

    fsc_half1 = capture.vector(consumer, 1, "model_fsc_halves_class")
    fsc_half2 = capture.vector(consumer, 2, "model_fsc_halves_class")
    _assert_exact("captured per-half FSC", fsc_half1, fsc_half2)
    _assert_exact("base v2 FSC", base.fsc, np.asarray(fsc_half1, dtype=np.float32))
    _assert_exact(
        "base v2 ave_pmax",
        base.ave_pmax,
        _shared_capture_scalar(capture, consumer, "model_ave_pmax"),
    )

    for half, rows in enumerate(half_rows, start=1):
        half_index = half - 1
        _assert_exact(
            f"base v2 half{half} image names",
            base.image_names_per_half[half_index],
            image_names[rows],
        )
        _assert_exact(f"base v2 half{half} source rows", base.source_rows_per_half[half_index], rows)
        _assert_exact(
            f"base v2 half{half} random subset",
            base.random_subsets_per_half[half_index],
            np.full(rows.shape, half, dtype=np.int8),
        )
        _assert_exact(
            f"base v2 half{half} half index",
            base.half_indices_per_half[half_index],
            np.full(rows.shape, half_index, dtype=np.int8),
        )
        _assert_exact(
            f"base v2 half{half} local index",
            base.half_local_indices_per_half[half_index],
            np.arange(rows.size, dtype=np.int64),
        )
        _assert_exact(
            f"base v2 half{half} poses",
            base.previous_best_rotation_eulers[half_index],
            np.asarray(eulers[rows], dtype=np.float32),
        )
        _assert_exact(
            f"base v2 half{half} translations",
            base.previous_best_translations[half_index],
            np.asarray(translations[rows], dtype=np.float32),
        )

        noise = capture.vector(consumer, half, "model_sigma2_noise_optics0")
        _assert_exact(
            f"base v2 half{half} noise",
            base.noise_radial_per_half[half_index],
            np.asarray(noise * n4, dtype=np.float64),
        )
        direction_prior = capture.vector(consumer, half, "model_pdf_direction")
        _assert_exact(
            f"base v2 half{half} direction prior",
            base.direction_prior_per_half[half_index],
            np.asarray(direction_prior, dtype=np.float32),
        )
        sigma_offset = np.sqrt(capture.scalar(consumer, half, "model_sigma2_offset"))
        _assert_exact(
            f"base v2 half{half} translation sigma",
            base.translation_sigma_angstrom_per_half[half_index],
            sigma_offset,
        )
        scales = capture.vector(consumer, half, "model_scale_correction")
        if np.any(group[rows] < 1) or np.any(group[rows] > scales.size):
            raise ValueError(f"completed STAR half{half} group number exceeds captured scales")
        per_image_scale = scales[group[rows] - 1]
        expected_correction = (
            capture.scalar(consumer, half, "model_avg_norm_correction")
            / norm_correction[rows]
            * per_image_scale
        )
        _assert_exact(
            f"base v2 half{half} scale corrections",
            base.scale_corrections[half_index],
            np.asarray(per_image_scale, dtype=np.float32),
        )
        _assert_exact(
            f"base v2 half{half} image corrections",
            base.image_corrections[half_index],
            np.asarray(expected_correction, dtype=np.float32),
        )

    shared_scalar_fields = {
        "relion_incr_size": "control_incr_size",
        "has_high_fsc_at_limit": "control_has_high_fsc_at_limit",
        "state_nr_iter_wo_resol_gain": "control_nr_iter_wo_resol_gain",
        "state_nr_iter_wo_large_hidden_variable_changes": "control_nr_iter_wo_large_hidden_variable_changes",
        "state_current_changes_optimal_orientations": "control_current_changes_optimal_orientations",
        "state_current_changes_optimal_offsets_angstrom": "control_current_changes_optimal_offsets",
        "state_smallest_changes_optimal_orientations": "control_smallest_changes_optimal_orientations",
        "state_smallest_changes_optimal_offsets_angstrom": "control_smallest_changes_optimal_offsets",
        "state_acc_rot": "control_acc_rot",
        "state_acc_trans": "control_acc_trans",
        "state_has_converged": "control_has_converged",
        "state_ave_Pmax": "model_ave_pmax",
    }
    for boundary_name, capture_name in shared_scalar_fields.items():
        observed = (
            getattr(base, boundary_name)
            if hasattr(base, boundary_name)
            else base.refinement_state_fields[boundary_name.removeprefix("state_")]
        )
        expected = _shared_capture_scalar(capture, consumer, capture_name)
        if isinstance(observed, bool):
            expected = bool(int(expected))
        elif isinstance(observed, (int, np.integer)):
            expected = int(expected)
        _assert_exact(f"base v2 {boundary_name}", observed, expected)

    captured_resolution = 1.0 / _shared_capture_scalar(
        capture,
        consumer,
        "model_current_resolution_inverse_angstrom",
    )
    _assert_exact(
        "base v2 current resolution",
        base.refinement_state_fields["current_resolution"],
        captured_resolution,
    )
    _assert_exact(
        "base v2 previous resolution",
        base.refinement_state_fields["previous_resolution"],
        captured_resolution,
    )
    completed_optimiser = _star_mapping(
        source_paths["completed_optimiser"],
        "optimiser_general",
    )
    _assert_exact(
        "base v2 assignment-change counter",
        base.refinement_state_fields["nr_iter_wo_assignment_changes"],
        int(completed_optimiser["rlnNumberOfIterWithoutChangingAssignments"]),
    )


def _validate_runtime_config_against_capture(
    runtime_config: dict,
    *,
    capture: _ValidatedCapture,
    source_paths: dict[str, Path],
    consumer: int,
) -> None:
    """Reject runtime JSON that is not the captured RELION control state."""

    optimiser = _star_mapping(source_paths["completed_optimiser"], "optimiser_general")
    voxel_size = _shared_capture_scalar(capture, consumer, "model_pixel_size")
    direct = {
        "adaptive_oversampling": int(_shared_capture_scalar(capture, consumer, "control_adaptive_oversampling")),
        "max_significants": int(_shared_capture_scalar(capture, consumer, "control_maximum_significants")),
        "width_mask_edge_px": _shared_capture_scalar(capture, consumer, "control_width_mask_edge"),
        "tau2_fudge": _shared_capture_scalar(capture, consumer, "model_tau2_fudge_factor"),
        "random_seed": int(_shared_capture_scalar(capture, consumer, "control_random_seed")),
        "perturb_seed": int(_shared_capture_scalar(capture, consumer, "control_random_seed")),
        "n_classes": int(_shared_capture_scalar(capture, consumer, "model_nr_classes")),
        "grid_size": int(_shared_capture_scalar(capture, consumer, "model_ori_size")),
        "voxel_size_angstrom": voxel_size,
        "projection_padding_factor": int(_shared_capture_scalar(capture, consumer, "projector_padding_factor")),
        "backprojection_padding_factor": int(_shared_capture_scalar(capture, consumer, "projector_padding_factor")),
        "do_ctf_correction": bool(int(_shared_capture_scalar(capture, consumer, "control_do_ctf_correction"))),
        "firstiter_cc": bool(int(_shared_capture_scalar(capture, consumer, "control_do_firstiter_cc"))),
        "do_norm_correction": bool(int(_shared_capture_scalar(capture, consumer, "control_do_norm_correction"))),
        "do_scale_correction": bool(int(_shared_capture_scalar(capture, consumer, "control_do_scale_correction"))),
        "refs_are_ctf_corrected": bool(int(_shared_capture_scalar(capture, consumer, "control_refs_are_ctf_corrected"))),
        "auto_local_healpix_order": int(optimiser["rlnAutoLocalSearchesHealpixOrder"]),
        "particle_diameter_angstrom": float(optimiser["rlnParticleDiameter"]),
        "low_resol_join_halves_angstrom": float(optimiser["rlnJoinHalvesUntilThisResolution"]),
        "offset_range_pixels": _shared_capture_scalar(
            capture,
            consumer,
            "sampling_offset_range",
        )
        / voxel_size,
        "offset_step_pixels": _shared_capture_scalar(
            capture,
            consumer,
            "sampling_offset_step",
        )
        / voxel_size,
        "perturb_factor": _shared_capture_scalar(
            capture,
            consumer,
            "sampling_perturbation_factor",
        ),
    }
    for name, expected in direct.items():
        _assert_exact(f"runtime config {name}", runtime_config[name], expected)

    active_healpix_order = int(
        _shared_capture_scalar(capture, consumer, "sampling_healpix_order")
    )
    if int(runtime_config["max_healpix_order"]) < active_healpix_order:
        raise ValueError("runtime max_healpix_order is below captured active sampling order")
    full_size = _shared_capture_vector(
        capture,
        consumer,
        "control_image_full_size",
        "<i8",
    )
    if full_size.shape != (1,) or int(full_size[0]) != int(runtime_config["grid_size"]):
        raise ValueError("runtime grid size differs from captured full image size")
    for name, expected in {
        "model_data_dim": 2,
        "model_ref_dim": 3,
        "model_nr_bodies": 1,
        "model_nr_optics_groups": 1,
    }.items():
        if int(_shared_capture_scalar(capture, consumer, name)) != expected:
            raise ValueError(f"unsupported captured RELION model control: {name}")
    expected_strings = {
        "diagnostic_arm_id": FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM,
        "disc_type": "linear_interp",
        "image_fourier_backend": "relion_cuda",
        "local_search_translation_prior_mode": "coarse",
        "projector_boundary_kind": "reconstructed-projector boundary",
    }
    for name, expected in expected_strings.items():
        _assert_exact(f"runtime config {name}", runtime_config[name], expected)
    for name, expected in {
        "max_iter": 1,
        "skip_final_iteration": True,
        "init_resolution_angstrom": 30.0,
        "fsc_threshold": 1.0 / 7.0,
        "jax_enable_x64": True,
    }.items():
        _assert_exact(f"runtime config {name}", runtime_config[name], expected)
    completed_model = _star_mapping(
        source_paths["completed_half1_model"],
        "model_general",
    )
    if int(completed_model["rlnFourierSpaceInterpolator"]) != 1:
        raise ValueError("captured boundary does not use RELION trilinear interpolation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v2-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--source-paths-json", type=Path, required=True)
    parser.add_argument("--runtime-config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consumer-iteration", type=int, required=True)
    args = parser.parse_args()

    base = load_frozen_refinement_boundary(args.base_v2_dir)
    if base.schema != FROZEN_BOUNDARY_SCHEMA_V2:
        raise ValueError("--base-v2-dir must contain historical schema-v2 state")
    consumer = int(args.consumer_iteration)
    if consumer != base.completed_relion_iteration + 1:
        raise ValueError("consumer iteration must equal completed iteration + 1")

    capture_manifest = args.capture_manifest.expanduser().resolve()
    capture = _validate_capture_manifest(capture_manifest)
    for rank in (1, 2):
        if capture.scalar(consumer, rank, "state_schema_version") != 4.0:
            raise ValueError("live capture is not schema 4")
        if capture.scalar(consumer, rank, "iteration") != float(consumer):
            raise ValueError("live capture iteration mismatch")
        if capture.scalar(consumer, rank, "mpi_rank") != float(rank):
            raise ValueError("live capture MPI-rank identity mismatch")
        if capture.scalar(consumer, rank, "control_my_halfset") != float(rank):
            raise ValueError("live capture half identity mismatch")

    raw_source_paths = _load_json_object(args.source_paths_json)
    source_paths = {
        str(name): Path(value).expanduser().resolve()
        for name, value in raw_source_paths.items()
    }
    _validate_source_paths(source_paths, live_manifest=capture_manifest)
    source_names = sorted(source_paths)

    runtime_config = _load_json_object(args.runtime_config_json)
    runtime_payload = _runtime_payload(runtime_config)
    _validate_runtime_environment_manifest(
        source_paths["runtime_environment_manifest"],
        runtime_config,
    )
    _validate_recovar_source_manifest(
        source_paths["recovar_source_manifest"],
        runtime_config,
    )
    grid_size = int(runtime_config["grid_size"])
    if tuple(base.volume_shape) != (grid_size, grid_size, grid_size):
        raise ValueError("base volume shape and runtime grid size differ")
    captured_current_size = int(
        _shared_capture_scalar(capture, consumer, "model_current_size")
    )
    if captured_current_size != int(base.current_size):
        raise ValueError(
            f"captured/base current_size mismatch: capture={captured_current_size} "
            f"base={base.current_size}"
        )
    captured_active_order = int(
        _shared_capture_scalar(capture, consumer, "sampling_healpix_order")
    )
    if captured_active_order != int(base.healpix_order):
        raise ValueError(
            f"captured/base HEALPix-order mismatch: capture={captured_active_order} "
            f"base={base.healpix_order}"
        )
    captured_sizes = {
        name: _shared_capture_vector(capture, consumer, name, "<i8")
        for name in (
            "control_image_coarse_size",
            "control_image_current_size",
            "control_image_full_size",
        )
    }
    if any(value.shape != (1,) for value in captured_sizes.values()):
        raise ValueError("captured coarse/current/full sizes must be scalar vectors")
    if int(captured_sizes["control_image_current_size"][0]) != captured_current_size:
        raise ValueError("captured model/control current sizes differ")
    if int(captured_sizes["control_image_full_size"][0]) != grid_size:
        raise ValueError("captured full size and runtime grid size differ")

    _validate_runtime_config_against_capture(
        runtime_config,
        capture=capture,
        source_paths=source_paths,
        consumer=consumer,
    )
    _validate_base_v2_state(
        base,
        capture=capture,
        source_paths=source_paths,
        consumer=consumer,
        grid_size=grid_size,
    )

    means = []
    captured_iref_sha256 = []
    tau2_per_half = []
    n4 = float(grid_size) ** 4
    for rank in (1, 2):
        dims = tuple(
            int(capture.scalar(consumer, rank, f"iref_{axis}dim"))
            for axis in ("z", "y", "x")
        )
        if dims != tuple(base.volume_shape):
            raise ValueError(f"captured Iref shape mismatch for rank {rank}: {dims}")
        iref = capture.vector(consumer, rank, "iref").reshape(dims)
        captured_iref_sha256.append(_array_sha256(iref))
        converted = np.asarray(helpers.relion_volume_to_recovar(iref), dtype=np.float64)
        mean = np.asarray(ftu.get_dft3(jnp.asarray(converted)), dtype=np.complex64).reshape(-1)
        if not np.array_equal(mean, base.means[rank - 1]):
            raise ValueError(f"captured Iref transform differs from base half-{rank} mean")
        means.append(mean)
        tau2_profile = capture.vector(consumer, rank, "model_tau2_class")
        tau2 = np.asarray(
            utils.make_radial_image(
                tau2_profile * n4,
                tuple(base.volume_shape),
                extend_last_frequency=True,
            ),
            dtype=np.float32,
        ).reshape(-1)
        tau2_per_half.append(tau2)

    shared_vector_fields = {
        "directions_ipix": ("sampling_directions_ipix", "<i8"),
        "rot_angles_deg": ("sampling_rot_angles", "<f8"),
        "tilt_angles_deg": ("sampling_tilt_angles", "<f8"),
        "psi_angles_deg": ("sampling_psi_angles", "<f8"),
        "translations_x_angstrom": ("sampling_translations_x", "<f8"),
        "translations_y_angstrom": ("sampling_translations_y", "<f8"),
        "translations_z_angstrom": ("sampling_translations_z", "<f8"),
    }
    sampling_arrays = {}
    for output_name, (capture_name, dtype) in shared_vector_fields.items():
        half1 = capture.vector(consumer, 1, capture_name, dtype)
        half2 = capture.vector(consumer, 2, capture_name, dtype)
        if not np.array_equal(half1, half2):
            raise ValueError(f"captured sampling vector differs by half: {capture_name}")
        sampling_arrays[f"sampling_{output_name}"] = half1

    def shared_scalar(name: str) -> float:
        half1 = capture.scalar(consumer, 1, name)
        half2 = capture.scalar(consumer, 2, name)
        if half1 != half2:
            raise ValueError(f"captured scalar differs by half: {name}")
        return half1

    coarse = captured_sizes["control_image_coarse_size"]
    full = captured_sizes["control_image_full_size"]

    with np.load(Path(args.base_v2_dir) / FROZEN_BOUNDARY_FILENAME, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload.update(
        {
            "schema": np.asarray(FROZEN_BOUNDARY_SCHEMA_V3),
            "consumer_relion_iteration": np.int32(consumer),
            "half1_mean_variance": tau2_per_half[0],
            "half2_mean_variance": tau2_per_half[1],
            "mean_variance": np.asarray(
                0.5 * (tau2_per_half[0].astype(np.float64) + tau2_per_half[1].astype(np.float64)),
                dtype=np.float32,
            ),
            "source_sha256_names": np.asarray(source_names),
            "source_sha256_digests": np.asarray([_sha256(source_paths[name]) for name in source_names]),
            "source_sha256_roles": np.asarray([v3_source_role(name) for name in source_names]),
            "sampling_healpix_order": np.int32(shared_scalar("sampling_healpix_order")),
            "sampling_healpix_order_original": np.int32(shared_scalar("sampling_healpix_order_original")),
            "sampling_psi_step_deg": np.float64(shared_scalar("sampling_psi_step")),
            "sampling_offset_range_angstrom": np.float64(shared_scalar("sampling_offset_range")),
            "sampling_offset_step_angstrom": np.float64(shared_scalar("sampling_offset_step")),
            "sampling_perturbation_factor": np.float64(shared_scalar("sampling_perturbation_factor")),
            "sampling_random_perturbation": np.float64(shared_scalar("sampling_random_perturbation")),
            "sampling_sigma_rot_deg": np.float64(np.sqrt(max(0.0, shared_scalar("model_sigma2_rot")))),
            "sampling_sigma_psi_deg": np.float64(np.sqrt(max(0.0, shared_scalar("model_sigma2_psi")))),
            "sampling_is_3d": np.bool_(bool(int(shared_scalar("sampling_is_3d")))),
            "sampling_is_3d_trans": np.bool_(bool(int(shared_scalar("sampling_is_3d_trans")))),
            "sampling_point_group": np.int32(shared_scalar("sampling_point_group")),
            "sampling_point_group_order": np.int32(shared_scalar("sampling_point_group_order")),
            "sampling_coarse_size": np.int32(coarse[0]),
            "sampling_full_size": np.int32(full[0]),
            "source_map_serialization": np.asarray("captured_relion_iref_transformed_to_complex64"),
            "bitwise_identity_to_original_in_memory_means": np.bool_(False),
            "map_transform_id": np.asarray(_V3_MAP_TRANSFORM_ID),
            "half1_captured_iref_sha256": np.asarray(captured_iref_sha256[0]),
            "half2_captured_iref_sha256": np.asarray(captured_iref_sha256[1]),
            "half1_transformed_mean_sha256": np.asarray(_array_sha256(means[0])),
            "half2_transformed_mean_sha256": np.asarray(_array_sha256(means[1])),
            "source_star_sha256": np.asarray(_sha256(source_paths["particle_star"])),
            "relion_half_star_sha256": np.asarray(_sha256(source_paths["relion_half_star"])),
            **sampling_arrays,
            **runtime_payload,
        }
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / FROZEN_BOUNDARY_FILENAME
    with output_path.open("xb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    digest = _sha256(output_path)
    manifest = output_dir / FROZEN_BOUNDARY_MANIFEST
    manifest.write_text(f"{digest}  {FROZEN_BOUNDARY_FILENAME}\n", encoding="utf-8")
    loaded = load_frozen_refinement_boundary(output_dir)
    report = {
        "schema": loaded.schema,
        "completed_relion_iteration": loaded.completed_relion_iteration,
        "consumer_relion_iteration": loaded.consumer_relion_iteration,
        "boundary_sha256": loaded.boundary_sha256,
        "source_count": len(loaded.source_sha256),
        "live_capture_manifest": str(capture_manifest),
        "live_capture_manifest_sha256": _sha256(capture_manifest),
        "map_lineage": loaded.map_lineage,
        "source_roles": loaded.source_roles,
        "projector_boundary_kind": loaded.runtime_config["projector_boundary_kind"],
        "quality_metric_policy": "exact arrays/posteriors for intermediates; FSC/FSC-AUC for maps; no correlation gate",
    }
    (output_dir / "FROZEN_BOUNDARY_V3_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
