"""Fail-closed loader for diagnostic frozen EM iteration boundaries.

The production refinement path does not resume from this format.  It exists
only for causal parity experiments where several one-iteration arms must start
from byte-identical, explicitly sealed state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FROZEN_BOUNDARY_SCHEMA_V2 = "recovar.em.frozen_boundary.v2"
FROZEN_BOUNDARY_SCHEMA_V3 = "recovar.em.frozen_boundary.v3"
FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM = (
    "real10076.k1.physical_it2.reconstructed_projector.v1"
)
FROZEN_BOUNDARY_FIXED_MATH_ENVIRONMENT_CONTRACT = (
    "no_unsealed_recovar_jax_xla_overrides.v1"
)
FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE = (
    "declared-source-command-build; source/runtime hardware-toolchain unverified"
)
FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE = (
    "cross-device-unverified; same-device/numerical-noise classification forbidden"
)
# Historical public name. The fixed real-10076 diagnostic arm uses v3.
FROZEN_BOUNDARY_SCHEMA = FROZEN_BOUNDARY_SCHEMA_V2
FROZEN_BOUNDARY_FILENAME = "frozen_boundary_v2.npz"
FROZEN_BOUNDARY_MANIFEST = "FROZEN_BOUNDARY_SHA256SUMS"

_REFINEMENT_STATE_FIELD_DTYPES = {
    "current_resolution": np.dtype(np.float64),
    "previous_resolution": np.dtype(np.float64),
    "nr_iter_wo_resol_gain": np.dtype(np.int32),
    "nr_iter_wo_assignment_changes": np.dtype(np.int32),
    "nr_iter_wo_large_hidden_variable_changes": np.dtype(np.int32),
    "ave_Pmax": np.dtype(np.float64),
    "current_changes_optimal_orientations": np.dtype(np.float64),
    "current_changes_optimal_offsets_angstrom": np.dtype(np.float64),
    "smallest_changes_optimal_orientations": np.dtype(np.float64),
    "smallest_changes_optimal_offsets_angstrom": np.dtype(np.float64),
    "acc_rot": np.dtype(np.float64),
    "acc_trans": np.dtype(np.float64),
    "has_converged": np.dtype(np.bool_),
}
_PROVENANCE_PAYLOAD_KEYS = {
    "source_job_id",
    "source_arm",
    "source_map_serialization",
    "bitwise_identity_to_original_in_memory_means",
    "correction_state_owner",
    "identity_schema",
    "source_star_sha256",
    "relion_half_star_sha256",
}
_V2_EXPECTED_PROVENANCE_STRINGS = {
    "source_map_serialization": "in_memory_complex64",
    "correction_state_owner": "sealed_boundary",
    "identity_schema": "five_field.v1",
}
_V3_EXPECTED_PROVENANCE_STRINGS = {
    "source_map_serialization": "captured_relion_iref_transformed_to_complex64",
    "correction_state_owner": "sealed_boundary",
    "identity_schema": "five_field.v1",
}
_V3_MAP_TRANSFORM_ID = "relion_iref_to_recovar_complex64.v1"
_REQUIRED_PAYLOAD_KEYS = {
    "schema",
    "completed_relion_iteration",
    "volume_shape",
    "current_size",
    "healpix_order",
    "relion_incr_size",
    "has_high_fsc_at_limit",
    "half1_mean_ft",
    "half2_mean_ft",
    "mean_variance",
    "half1_noise_radial",
    "half2_noise_radial",
    "fsc",
    "ave_pmax",
    "half1_previous_best_rotation_eulers",
    "half2_previous_best_rotation_eulers",
    "half1_previous_best_translations",
    "half2_previous_best_translations",
    "half1_image_corrections",
    "half2_image_corrections",
    "half1_scale_corrections",
    "half2_scale_corrections",
    "half1_direction_prior",
    "half2_direction_prior",
    "half1_translation_sigma_angstrom",
    "half2_translation_sigma_angstrom",
    "half1_image_name",
    "half2_image_name",
    "half1_source_row",
    "half2_source_row",
    "half1_random_subset",
    "half2_random_subset",
    "half1_half_index",
    "half2_half_index",
    "half1_half_local_index",
    "half2_half_local_index",
    *(f"state_{key}" for key in _REFINEMENT_STATE_FIELD_DTYPES),
    *_PROVENANCE_PAYLOAD_KEYS,
}
_ALLOWED_PAYLOAD_KEYS = _REQUIRED_PAYLOAD_KEYS

V3_REQUIRED_FIXED_SOURCE_NAMES = frozenset(
    {
        "particle_star",
        "relion_half_star",
        "completed_data",
        "completed_optimiser",
        "completed_sampling",
        "completed_half1_model",
        "completed_half2_model",
        "consumer_validation_optimiser",
        "consumer_validation_data",
        "consumer_validation_sampling",
        "consumer_validation_half1_model",
        "consumer_validation_half2_model",
        "live_capture_manifest",
        "runtime_environment_manifest",
        "recovar_source_manifest",
    }
)
# Compatibility alias for callers that only need the fixed-name subset. Fixed
# v3 bundles additionally contain one or more ``particle_stack:<index>`` rows.
V3_REQUIRED_SOURCE_NAMES = V3_REQUIRED_FIXED_SOURCE_NAMES

_V3_SOURCE_ROLES = {
    "particle_star": "input_metadata_and_optics",
    "relion_half_star": "input_half_identity_validation",
    "completed_data": "incoming_particle_state_validation",
    "completed_optimiser": "incoming_control_validation",
    "completed_sampling": "incoming_sampling_validation",
    "completed_half1_model": "incoming_half1_model_validation",
    "completed_half2_model": "incoming_half2_model_validation",
    "consumer_validation_optimiser": "post_consumer_validation",
    "consumer_validation_data": "post_consumer_particle_state_validation",
    "consumer_validation_sampling": "post_consumer_validation",
    "consumer_validation_half1_model": "post_consumer_validation",
    "consumer_validation_half2_model": "post_consumer_validation",
    "live_capture_manifest": "live_boundary_owner",
    "runtime_environment_manifest": "runtime_build_environment",
    "recovar_source_manifest": "consumer_source_tree",
}


def v3_source_role(source_name: str) -> str:
    """Return the required semantic role for one schema-v3 source name."""

    if source_name.startswith("particle_stack:"):
        suffix = source_name.removeprefix("particle_stack:")
        if suffix.isdigit():
            return "input_particle_stack_bytes"
    if source_name in {"consumer_map:half1:class1", "consumer_map:half2:class1"}:
        return "post_consumer_map_bytes"
    try:
        return _V3_SOURCE_ROLES[source_name]
    except KeyError as exc:
        raise ValueError(f"unknown frozen-boundary v3 source name {source_name!r}") from exc


_V3_ARRAY_DTYPES = {
    "source_sha256_names": None,
    "source_sha256_digests": None,
    "source_sha256_roles": None,
    "sampling_directions_ipix": np.dtype(np.int64),
    "sampling_rot_angles_deg": np.dtype(np.float64),
    "sampling_tilt_angles_deg": np.dtype(np.float64),
    "sampling_psi_angles_deg": np.dtype(np.float64),
    "sampling_translations_x_angstrom": np.dtype(np.float64),
    "sampling_translations_y_angstrom": np.dtype(np.float64),
    "sampling_translations_z_angstrom": np.dtype(np.float64),
    "half1_mean_variance": np.dtype(np.float32),
    "half2_mean_variance": np.dtype(np.float32),
}
_V3_SCALAR_DTYPES = {
    "consumer_relion_iteration": np.dtype(np.int32),
    "sampling_healpix_order": np.dtype(np.int32),
    "sampling_healpix_order_original": np.dtype(np.int32),
    "sampling_psi_step_deg": np.dtype(np.float64),
    "sampling_offset_range_angstrom": np.dtype(np.float64),
    "sampling_offset_step_angstrom": np.dtype(np.float64),
    "sampling_perturbation_factor": np.dtype(np.float64),
    "sampling_random_perturbation": np.dtype(np.float64),
    "sampling_sigma_rot_deg": np.dtype(np.float64),
    "sampling_sigma_psi_deg": np.dtype(np.float64),
    "sampling_is_3d": np.dtype(np.bool_),
    "sampling_is_3d_trans": np.dtype(np.bool_),
    "sampling_point_group": np.dtype(np.int32),
    "sampling_point_group_order": np.dtype(np.int32),
    "sampling_coarse_size": np.dtype(np.int32),
    "sampling_full_size": np.dtype(np.int32),
    "config_adaptive_oversampling": np.dtype(np.int32),
    "config_diagnostic_arm_id": None,
    "config_max_iter": np.dtype(np.int32),
    "config_skip_final_iteration": np.dtype(np.bool_),
    "config_init_resolution_angstrom": np.dtype(np.float64),
    "config_offset_range_pixels": np.dtype(np.float64),
    "config_offset_step_pixels": np.dtype(np.float64),
    "config_perturb_factor": np.dtype(np.float64),
    "config_fsc_threshold": np.dtype(np.float64),
    "config_jax_enable_x64": np.dtype(np.bool_),
    "config_provenance_verification_scope": None,
    "config_numerical_classification_scope": None,
    "config_auto_local_healpix_order": np.dtype(np.int32),
    "config_max_healpix_order": np.dtype(np.int32),
    "config_max_significants": np.dtype(np.int32),
    "config_particle_diameter_angstrom": np.dtype(np.float64),
    "config_width_mask_edge_px": np.dtype(np.float64),
    "config_tau2_fudge": np.dtype(np.float64),
    "config_low_resol_join_halves_angstrom": np.dtype(np.float64),
    "config_image_batch_size": np.dtype(np.int32),
    "config_rotation_block_size": np.dtype(np.int32),
    "config_random_seed": np.dtype(np.int64),
    "config_perturb_seed": np.dtype(np.int64),
    "config_n_classes": np.dtype(np.int32),
    "config_grid_size": np.dtype(np.int32),
    "config_voxel_size_angstrom": np.dtype(np.float64),
    "config_projection_padding_factor": np.dtype(np.int32),
    "config_backprojection_padding_factor": np.dtype(np.int32),
    "config_do_ctf_correction": np.dtype(np.bool_),
    "config_firstiter_cc": np.dtype(np.bool_),
    "config_do_norm_correction": np.dtype(np.bool_),
    "config_do_scale_correction": np.dtype(np.bool_),
    "config_refs_are_ctf_corrected": np.dtype(np.bool_),
    "config_disc_type": None,
    "config_image_fourier_backend": None,
    "config_local_search_translation_prior_mode": None,
    "config_declared_relion_command_line": None,
    "config_declared_relion_base_git_commit": None,
    "config_recovar_git_commit": None,
    "config_declared_relion_build_id": None,
    "config_projector_boundary_kind": None,
    "config_replay_prefix": None,
}
_V3_MAP_LINEAGE_DTYPES = {
    "map_transform_id": None,
    "half1_captured_iref_sha256": None,
    "half2_captured_iref_sha256": None,
    "half1_transformed_mean_sha256": None,
    "half2_transformed_mean_sha256": None,
}
_V3_REQUIRED_PAYLOAD_KEYS = (
    set(_V3_ARRAY_DTYPES) | set(_V3_SCALAR_DTYPES) | set(_V3_MAP_LINEAGE_DTYPES)
)
_V3_ALLOWED_PAYLOAD_KEYS = _REQUIRED_PAYLOAD_KEYS | _V3_REQUIRED_PAYLOAD_KEYS


@dataclass(frozen=True)
class FrozenRefinementBoundary:
    source_dir: Path
    source_manifest: Path
    source_manifest_sha256: str
    boundary_sha256: str
    completed_relion_iteration: int
    consumer_relion_iteration: int
    volume_shape: tuple[int, int, int]
    current_size: int
    healpix_order: int
    relion_incr_size: int
    has_high_fsc_at_limit: bool
    means: tuple[np.ndarray, np.ndarray]
    mean_variance: np.ndarray
    mean_variance_per_half: tuple[np.ndarray, np.ndarray]
    noise_radial_per_half: tuple[np.ndarray, np.ndarray]
    fsc: np.ndarray
    ave_pmax: float
    previous_best_rotation_eulers: tuple[np.ndarray, np.ndarray]
    previous_best_translations: tuple[np.ndarray, np.ndarray]
    image_names_per_half: tuple[np.ndarray, np.ndarray]
    source_rows_per_half: tuple[np.ndarray, np.ndarray]
    random_subsets_per_half: tuple[np.ndarray, np.ndarray]
    half_indices_per_half: tuple[np.ndarray, np.ndarray]
    half_local_indices_per_half: tuple[np.ndarray, np.ndarray]
    image_corrections: tuple[np.ndarray, np.ndarray]
    scale_corrections: tuple[np.ndarray, np.ndarray]
    direction_prior_per_half: tuple[np.ndarray, np.ndarray]
    translation_sigma_angstrom_per_half: tuple[float, float]
    refinement_state_fields: dict[str, float | int | bool]
    source_job_id: int
    source_arm: str
    source_map_serialization: str
    bitwise_identity_to_original_in_memory_means: bool
    correction_state_owner: str
    identity_schema: str
    source_star_sha256: str
    relion_half_star_sha256: str
    schema: str
    fixed_diagnostic_arm: bool
    source_sha256: dict[str, str]
    sampling_state: dict[str, np.ndarray | float | int | bool]
    runtime_config: dict[str, str | float | int | bool]
    source_roles: dict[str, str]
    map_lineage: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_bytes_sha256(value: np.ndarray) -> str:
    """Hash the exact C-order payload bytes used by the v3 map transform."""

    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _parse_single_file_manifest(path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            f"frozen-boundary manifest must contain exactly one entry, found {len(lines)}"
        )
    fields = lines[0].split(maxsplit=1)
    if len(fields) != 2 or len(fields[0]) != 64:
        raise ValueError("invalid frozen-boundary SHA-256 manifest entry")
    digest, relative_name = fields
    relative_name = relative_name.lstrip("*")
    if relative_name.startswith("./"):
        relative_name = relative_name[2:]
    if relative_name != FROZEN_BOUNDARY_FILENAME:
        raise ValueError(
            f"frozen-boundary manifest must seal {FROZEN_BOUNDARY_FILENAME}, got {relative_name}"
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("invalid frozen-boundary SHA-256 digest") from exc
    return digest.lower(), relative_name


def verify_fixed_diagnostic_boundary_sources(
    boundary: FrozenRefinementBoundary,
    source_paths: dict[str, str | Path],
) -> dict[str, Path]:
    """Verify every fixed-arm v3 semantic source against its sealed bytes."""

    if not boundary.fixed_diagnostic_arm:
        raise ValueError(
            "frozen-boundary v2 is historical diagnostic state and cannot support "
            "the fixed schema-v3 diagnostic arm"
        )
    observed_names = set(source_paths)
    expected_names = set(boundary.source_sha256)
    if observed_names != expected_names:
        raise ValueError(
            "fixed diagnostic boundary source path closure failed: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unknown={sorted(observed_names - expected_names)}"
        )
    resolved = {}
    for source_name in sorted(expected_names):
        source_path = Path(source_paths[source_name]).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(
                f"fixed diagnostic boundary source {source_name} is not a file: {source_path}"
            )
        observed_sha256 = _sha256(source_path)
        expected_sha256 = boundary.source_sha256[source_name]
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"fixed diagnostic boundary source {source_name} SHA-256 mismatch: "
                f"expected {expected_sha256}, got {observed_sha256}"
            )
        resolved[source_name] = source_path
    return resolved


def validate_fixed_diagnostic_boundary_runtime_config(
    boundary: FrozenRefinementBoundary,
    observed_config: dict[str, str | float | int | bool],
) -> None:
    """Fail closed unless the in-flight runtime config equals sealed v3 state."""

    if not boundary.fixed_diagnostic_arm:
        raise ValueError(
            "frozen-boundary v2 cannot validate the fixed schema-v3 diagnostic arm"
        )
    expected = boundary.runtime_config
    if set(observed_config) != set(expected):
        raise ValueError(
            "fixed diagnostic boundary runtime-config closure failed: "
            f"missing={sorted(set(expected) - set(observed_config))}, "
            f"unknown={sorted(set(observed_config) - set(expected))}"
        )
    mismatches = []
    for key in sorted(expected):
        expected_value = expected[key]
        observed_value = observed_config[key]
        if isinstance(expected_value, float):
            equal = np.asarray(observed_value).dtype == np.asarray(expected_value).dtype and (
                float(observed_value) == expected_value
            )
        else:
            equal = type(observed_value) is type(expected_value) and observed_value == expected_value
        if not equal:
            mismatches.append(
                f"{key}: observed={observed_value!r} ({type(observed_value).__name__}) "
                f"expected={expected_value!r} ({type(expected_value).__name__})"
            )
    if mismatches:
        raise ValueError("fixed diagnostic boundary runtime config mismatch: " + "; ".join(mismatches))


def validate_fixed_diagnostic_boundary_sampling_state(
    boundary: FrozenRefinementBoundary,
    observed_sampling: dict[str, np.ndarray | float | int | bool],
) -> None:
    """Fail closed unless exact live/runtime sampling equals sealed schema-v3 state."""

    if not boundary.fixed_diagnostic_arm:
        raise ValueError("frozen-boundary v2 has no fixed-arm sampling-state contract")
    expected = boundary.sampling_state
    if set(observed_sampling) != set(expected):
        raise ValueError(
            "fixed diagnostic boundary sampling-state closure failed: "
            f"missing={sorted(set(expected) - set(observed_sampling))}, "
            f"unknown={sorted(set(observed_sampling) - set(expected))}"
        )
    mismatches = []
    for key in sorted(expected):
        expected_value = expected[key]
        observed_value = observed_sampling[key]
        if isinstance(expected_value, np.ndarray):
            observed_array = np.asarray(observed_value)
            equal = (
                observed_array.dtype == expected_value.dtype
                and observed_array.shape == expected_value.shape
                and np.array_equal(observed_array, expected_value)
            )
        else:
            equal = type(observed_value) is type(expected_value) and observed_value == expected_value
        if not equal:
            observed_shape = getattr(np.asarray(observed_value), "shape", None)
            expected_shape = getattr(np.asarray(expected_value), "shape", None)
            mismatches.append(
                f"{key}: observed_type={type(observed_value).__name__} "
                f"observed_shape={observed_shape} expected_type={type(expected_value).__name__} "
                f"expected_shape={expected_shape}"
            )
    if mismatches:
        raise ValueError("fixed diagnostic boundary sampling state mismatch: " + "; ".join(mismatches))


def _required_array(npz, key: str, *, ndim: int | None = None) -> np.ndarray:
    if key not in npz.files:
        raise ValueError(f"frozen boundary is missing required array {key}")
    value = np.asarray(npz[key])
    if value.dtype.hasobject:
        raise ValueError(f"frozen boundary array {key} must not have object dtype")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"frozen boundary array {key} must have ndim={ndim}, got {value.ndim}")
    return value


def _exact_array(npz, key: str, *, ndim: int, dtype) -> np.ndarray:
    value = _required_array(npz, key, ndim=ndim)
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"frozen boundary array {key} has dtype {value.dtype}; expected {expected_dtype}"
        )
    return value


def _finite_array(npz, key: str, *, ndim: int, dtype) -> np.ndarray:
    value = _exact_array(npz, key, ndim=ndim, dtype=dtype)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"frozen boundary array {key} contains non-finite values")
    return value


def _scalar(npz, key: str, dtype):
    value = _required_array(npz, key)
    if value.shape != ():
        raise ValueError(f"frozen boundary scalar {key} must have shape (), got {value.shape}")
    if dtype is str:
        if value.dtype.kind != "U":
            raise ValueError(
                f"frozen boundary scalar {key} has dtype {value.dtype}; expected Unicode dtype"
            )
        return str(value.item())
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"frozen boundary scalar {key} has dtype {value.dtype}; expected {expected_dtype}"
        )
    return value.item()


def load_frozen_refinement_boundary(
    boundary_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> FrozenRefinementBoundary:
    """Load and validate one sealed primitive-only boundary bundle."""

    source_dir = Path(boundary_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"frozen-boundary directory does not exist: {source_dir}")
    source_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else source_dir / FROZEN_BOUNDARY_MANIFEST
    )
    if not source_manifest.is_file():
        raise ValueError(f"frozen-boundary manifest does not exist: {source_manifest}")
    expected_sha256, relative_name = _parse_single_file_manifest(source_manifest)
    boundary_path = source_dir / relative_name
    if not boundary_path.is_file():
        raise ValueError(f"frozen-boundary NPZ does not exist: {boundary_path}")
    actual_sha256 = _sha256(boundary_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "frozen-boundary SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    with np.load(boundary_path, allow_pickle=False) as npz:
        schema = _scalar(npz, "schema", str)
        if schema not in {FROZEN_BOUNDARY_SCHEMA_V2, FROZEN_BOUNDARY_SCHEMA_V3}:
            raise ValueError(
                f"unsupported frozen-boundary schema {schema!r}; expected one of "
                f"{[FROZEN_BOUNDARY_SCHEMA_V2, FROZEN_BOUNDARY_SCHEMA_V3]!r}"
            )
        payload_keys = set(npz.files)
        required_payload_keys = (
            _V3_ALLOWED_PAYLOAD_KEYS
            if schema == FROZEN_BOUNDARY_SCHEMA_V3
            else _REQUIRED_PAYLOAD_KEYS
        )
        allowed_payload_keys = (
            _V3_ALLOWED_PAYLOAD_KEYS
            if schema == FROZEN_BOUNDARY_SCHEMA_V3
            else _ALLOWED_PAYLOAD_KEYS
        )
        missing_keys = sorted(required_payload_keys - payload_keys)
        if missing_keys:
            schema_label = "schema-v3" if schema == FROZEN_BOUNDARY_SCHEMA_V3 else "schema-v2"
            raise ValueError(
                f"frozen boundary is missing required {schema_label} keys: "
                f"{missing_keys}"
            )
        unknown_keys = sorted(payload_keys - allowed_payload_keys)
        if unknown_keys:
            schema_label = "schema-v3" if schema == FROZEN_BOUNDARY_SCHEMA_V3 else "schema-v2"
            raise ValueError(
                f"frozen boundary contains unknown {schema_label} keys: "
                f"{unknown_keys}"
            )
        completed_iteration = _scalar(npz, "completed_relion_iteration", np.int32)
        if completed_iteration < 1:
            raise ValueError("completed_relion_iteration must be positive")
        volume_shape = _exact_array(npz, "volume_shape", ndim=1, dtype=np.int32)
        if volume_shape.shape != (3,) or np.any(volume_shape <= 0):
            raise ValueError(f"invalid frozen-boundary volume_shape {volume_shape.tolist()}")
        volume_size = int(np.prod(volume_shape, dtype=np.int64))

        means = tuple(
            _finite_array(npz, f"half{half}_mean_ft", ndim=1, dtype=np.complex64)
            for half in (1, 2)
        )
        if any(value.shape != (volume_size,) for value in means):
            raise ValueError(
                "frozen-boundary half-map Fourier arrays must match volume size "
                f"{volume_size}, got {[value.shape for value in means]}"
            )
        mean_variance = _finite_array(npz, "mean_variance", ndim=1, dtype=np.float32)
        if mean_variance.shape != (volume_size,) or np.any(mean_variance < 0.0):
            raise ValueError("frozen-boundary mean_variance has invalid shape or negative values")

        noise_radial_per_half = tuple(
            _finite_array(npz, f"half{half}_noise_radial", ndim=1, dtype=np.float64)
            for half in (1, 2)
        )
        if any(value.size < 2 or np.any(value <= 0.0) for value in noise_radial_per_half):
            raise ValueError("frozen-boundary noise spectra must be positive nontrivial arrays")
        fsc = _finite_array(npz, "fsc", ndim=1, dtype=np.float32)
        if fsc.size < 2:
            raise ValueError("frozen-boundary FSC must contain at least two shells")
        if np.any(fsc < -1.0) or np.any(fsc > 1.0):
            raise ValueError("frozen-boundary FSC values must lie in [-1, 1]")

        eulers = tuple(
            _finite_array(npz, f"half{half}_previous_best_rotation_eulers", ndim=2, dtype=np.float32)
            for half in (1, 2)
        )
        translations = tuple(
            _finite_array(npz, f"half{half}_previous_best_translations", ndim=2, dtype=np.float32)
            for half in (1, 2)
        )
        image_names = tuple(
            _required_array(npz, f"half{half}_image_name", ndim=1)
            for half in (1, 2)
        )
        source_rows = tuple(
            _exact_array(npz, f"half{half}_source_row", ndim=1, dtype=np.int64)
            for half in (1, 2)
        )
        random_subsets = tuple(
            _exact_array(npz, f"half{half}_random_subset", ndim=1, dtype=np.int8)
            for half in (1, 2)
        )
        half_indices = tuple(
            _exact_array(npz, f"half{half}_half_index", ndim=1, dtype=np.int8)
            for half in (1, 2)
        )
        half_local_indices = tuple(
            _exact_array(npz, f"half{half}_half_local_index", ndim=1, dtype=np.int64)
            for half in (1, 2)
        )
        for half, (euler, translation, names, rows, subsets, half_ids, local_ids) in enumerate(
            zip(
                eulers,
                translations,
                image_names,
                source_rows,
                random_subsets,
                half_indices,
                half_local_indices,
                strict=True,
            ),
            start=1,
        ):
            if euler.shape[1:] != (3,) or translation.shape[1:] != (2,):
                raise ValueError(f"frozen-boundary half-{half} pose arrays have invalid shape")
            if euler.shape[0] != translation.shape[0] or euler.shape[0] != names.shape[0]:
                raise ValueError(f"frozen-boundary half-{half} pose/identity row counts differ")
            if any(value.shape != names.shape for value in (rows, subsets, half_ids, local_ids)):
                raise ValueError(f"frozen-boundary half-{half} five-field identity row counts differ")
            if names.dtype.kind != "U":
                raise ValueError(f"frozen-boundary half-{half} image names must be Unicode strings")
            if np.any(rows < 0) or np.unique(rows).size != rows.size:
                raise ValueError(f"frozen-boundary half-{half} source rows must be unique nonnegative values")
            if not np.array_equal(subsets, np.full(names.shape, half, dtype=np.int8)):
                raise ValueError(f"frozen-boundary half-{half} random-subset identity is inconsistent")
            if not np.array_equal(half_ids, np.full(names.shape, half - 1, dtype=np.int8)):
                raise ValueError(f"frozen-boundary half-{half} zero-based half identity is inconsistent")
            if not np.array_equal(local_ids, np.arange(names.size, dtype=np.int64)):
                raise ValueError(f"frozen-boundary half-{half} local identity order is inconsistent")
        all_names = np.concatenate([np.asarray(value, dtype=str) for value in image_names])
        if np.unique(all_names).size != all_names.size:
            raise ValueError("frozen-boundary image names must be globally unique")
        all_source_rows = np.concatenate(source_rows)
        if np.unique(all_source_rows).size != all_source_rows.size:
            raise ValueError("frozen-boundary source rows must be globally unique")

        image_corrections = tuple(
            _finite_array(npz, f"half{half}_image_corrections", ndim=1, dtype=np.float32)
            for half in (1, 2)
        )
        scale_corrections = tuple(
            _finite_array(npz, f"half{half}_scale_corrections", ndim=1, dtype=np.float32)
            for half in (1, 2)
        )
        direction_prior_per_half = tuple(
            _finite_array(npz, f"half{half}_direction_prior", ndim=1, dtype=np.float32)
            for half in (1, 2)
        )
        translation_sigma_angstrom_per_half = tuple(
            _scalar(npz, f"half{half}_translation_sigma_angstrom", np.float64)
            for half in (1, 2)
        )
        expected_direction_count = 12 * (4 ** _scalar(npz, "healpix_order", np.int32))
        for half in range(2):
            expected_rows = eulers[half].shape[0]
            if image_corrections[half].shape != (expected_rows,):
                raise ValueError("frozen-boundary image-correction row count mismatch")
            if scale_corrections[half].shape != (expected_rows,):
                raise ValueError("frozen-boundary scale-correction row count mismatch")
            if np.any(image_corrections[half] <= 0.0):
                raise ValueError("frozen-boundary image corrections must be positive")
            if np.any(scale_corrections[half] <= 0.0):
                raise ValueError("frozen-boundary scale corrections must be positive")
            prior = direction_prior_per_half[half]
            if prior.shape != (expected_direction_count,):
                raise ValueError(
                    "frozen-boundary direction-prior shape does not match healpix_order"
                )
            if np.any(prior < 0.0) or not np.any(prior > 0.0):
                raise ValueError("frozen-boundary direction prior must be nonnegative and nonzero")
            if not np.isfinite(translation_sigma_angstrom_per_half[half]) or (
                translation_sigma_angstrom_per_half[half] <= 0.0
            ):
                raise ValueError("frozen-boundary translation sigma must be finite and positive")

        refinement_state_fields = {
            key: _scalar(npz, f"state_{key}", field_dtype)
            for key, field_dtype in _REFINEMENT_STATE_FIELD_DTYPES.items()
        }
        if refinement_state_fields.get("has_converged", False):
            raise ValueError("a frozen numbered-iteration boundary must not already be converged")
        for key, value in refinement_state_fields.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"frozen-boundary state scalar {key} must be finite")
        for key in (
            "nr_iter_wo_resol_gain",
            "nr_iter_wo_assignment_changes",
            "nr_iter_wo_large_hidden_variable_changes",
        ):
            if key in refinement_state_fields and refinement_state_fields[key] < 0:
                raise ValueError(f"frozen-boundary state counter {key} must be nonnegative")
        for key in (
            "current_resolution",
            "previous_resolution",
            "current_changes_optimal_orientations",
            "current_changes_optimal_offsets_angstrom",
            "smallest_changes_optimal_orientations",
            "smallest_changes_optimal_offsets_angstrom",
            "acc_rot",
            "acc_trans",
        ):
            if key in refinement_state_fields and refinement_state_fields[key] < 0.0:
                raise ValueError(f"frozen-boundary state scalar {key} must be nonnegative")
        if "ave_Pmax" in refinement_state_fields and not 0.0 <= refinement_state_fields["ave_Pmax"] <= 1.0:
            raise ValueError("frozen-boundary state ave_Pmax must lie in [0, 1]")

        source_job_id = _scalar(npz, "source_job_id", np.int64)
        if source_job_id < 0:
            raise ValueError("frozen-boundary source_job_id must be nonnegative")
        provenance_strings = {
            key: _scalar(npz, key, str)
            for key in (
                "source_arm",
                "source_map_serialization",
                "correction_state_owner",
                "identity_schema",
            )
        }
        for key, value in provenance_strings.items():
            if not value.strip():
                raise ValueError(f"frozen-boundary provenance scalar {key} must be nonempty")
        expected_provenance_strings = (
            _V3_EXPECTED_PROVENANCE_STRINGS
            if schema == FROZEN_BOUNDARY_SCHEMA_V3
            else _V2_EXPECTED_PROVENANCE_STRINGS
        )
        for key, expected_value in expected_provenance_strings.items():
            if provenance_strings[key] != expected_value:
                raise ValueError(
                    f"frozen-boundary provenance scalar {key} must equal "
                    f"{expected_value!r}, got {provenance_strings[key]!r}"
                )
        bitwise_identity = _scalar(
            npz,
            "bitwise_identity_to_original_in_memory_means",
            np.bool_,
        )
        if schema == FROZEN_BOUNDARY_SCHEMA_V2 and not bitwise_identity:
            raise ValueError(
                "frozen-boundary maps must be bitwise-identical to the original in-memory means"
            )
        if schema == FROZEN_BOUNDARY_SCHEMA_V3 and bitwise_identity:
            raise ValueError(
                "frozen-boundary v3 transformed RELION Iref maps must not claim bitwise "
                "identity to the original in-memory means"
            )
        provenance_hashes = {
            key: _scalar(npz, key, str)
            for key in ("source_star_sha256", "relion_half_star_sha256")
        }
        for key, digest in provenance_hashes.items():
            if not _is_lower_sha256(digest):
                raise ValueError(
                    f"frozen-boundary provenance scalar {key} must be lowercase SHA-256"
                )

        source_sha256 = {}
        source_roles = {}
        sampling_state = {}
        runtime_config = {}
        map_lineage = {}
        if schema == FROZEN_BOUNDARY_SCHEMA_V3:
            source_names = _required_array(npz, "source_sha256_names", ndim=1)
            source_digests = _required_array(npz, "source_sha256_digests", ndim=1)
            source_role_values = _required_array(npz, "source_sha256_roles", ndim=1)
            if any(value.dtype.kind != "U" for value in (source_names, source_digests, source_role_values)):
                raise ValueError(
                    "frozen-boundary v3 source names/digests/roles must be Unicode arrays"
                )
            if source_names.shape != source_digests.shape or source_names.shape != source_role_values.shape:
                raise ValueError("frozen-boundary v3 source name/digest/role row counts differ")
            if np.unique(source_names).size != source_names.size:
                raise ValueError("frozen-boundary v3 source names must be unique")
            source_sha256 = dict(zip(source_names.tolist(), source_digests.tolist(), strict=True))
            source_roles = dict(zip(source_names.tolist(), source_role_values.tolist(), strict=True))
            observed_source_names = set(source_sha256)
            stack_names = [
                name for name in observed_source_names if name.startswith("particle_stack:")
            ]
            invalid_stack_names = [
                name
                for name in stack_names
                if not name.removeprefix("particle_stack:").isdigit()
            ]
            stack_names.sort(
                key=lambda name: (
                    int(name.removeprefix("particle_stack:"))
                    if name not in invalid_stack_names
                    else -1
                )
            )
            consumer_map_names = {
                name for name in observed_source_names if name.startswith("consumer_map:")
            }
            required_consumer_map_names = {
                "consumer_map:half1:class1",
                "consumer_map:half2:class1",
            }
            expected_stack_names = [f"particle_stack:{index}" for index in range(len(stack_names))]
            unknown_names = (
                observed_source_names
                - V3_REQUIRED_FIXED_SOURCE_NAMES
                - set(stack_names)
                - consumer_map_names
            )
            missing_fixed = V3_REQUIRED_FIXED_SOURCE_NAMES - observed_source_names
            if (
                missing_fixed
                or unknown_names
                or invalid_stack_names
                or stack_names != expected_stack_names
                or not stack_names
                or consumer_map_names != required_consumer_map_names
            ):
                raise ValueError(
                    "frozen-boundary v3 source closure failed: "
                    f"missing={sorted(missing_fixed)}, unknown={sorted(unknown_names)}, "
                    f"particle_stacks={stack_names} invalid={invalid_stack_names} "
                    f"expected={expected_stack_names or ['particle_stack:0']}, "
                    f"consumer_maps={sorted(consumer_map_names)} "
                    f"expected_consumer_maps={sorted(required_consumer_map_names)}"
                )
            for source_name, digest in source_sha256.items():
                if not _is_lower_sha256(digest):
                    raise ValueError(
                        f"frozen-boundary v3 source {source_name} must be a lowercase SHA-256"
                    )
                expected_role = v3_source_role(source_name)
                if source_roles[source_name] != expected_role:
                    raise ValueError(
                        f"frozen-boundary v3 source {source_name} role must equal "
                        f"{expected_role!r}, got {source_roles[source_name]!r}"
                    )

            for key, dtype in _V3_ARRAY_DTYPES.items():
                if key.startswith("source_sha256_"):
                    continue
                value = _exact_array(npz, key, ndim=1, dtype=dtype)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"frozen-boundary v3 array {key} contains non-finite values")
                if key.startswith("sampling_"):
                    sampling_state[key.removeprefix("sampling_")] = value
            direction_grid_size = 12 * (4 ** int(_scalar(npz, "healpix_order", np.int32)))
            direction_count = np.asarray(npz["sampling_directions_ipix"]).size
            for key in ("sampling_directions_ipix", "sampling_rot_angles_deg", "sampling_tilt_angles_deg"):
                if np.asarray(npz[key]).shape != (direction_count,) or direction_count < 1:
                    raise ValueError(
                        f"frozen-boundary v3 {key} must have shape {(direction_count,)}"
                    )
            direction_ids = np.asarray(npz["sampling_directions_ipix"])
            if (
                np.unique(direction_ids).size != direction_ids.size
                or np.any(direction_ids < 0)
                or np.any(direction_ids >= direction_grid_size)
            ):
                raise ValueError(
                    "frozen-boundary v3 direction indices must be unique and lie inside "
                    f"[0, {direction_grid_size})"
                )
            psi_angles = np.asarray(npz["sampling_psi_angles_deg"])
            if psi_angles.size < 1:
                raise ValueError("frozen-boundary v3 psi-angle vector must be nonempty")
            tx = np.asarray(npz["sampling_translations_x_angstrom"])
            ty = np.asarray(npz["sampling_translations_y_angstrom"])
            tz = np.asarray(npz["sampling_translations_z_angstrom"])
            if tx.shape != ty.shape or tx.size < 1 or tz.size != 0:
                raise ValueError(
                    "frozen-boundary v3 2-D translation vectors require equal nonempty x/y "
                    "and empty z"
                )

            for key, dtype in _V3_SCALAR_DTYPES.items():
                if dtype is None:
                    value = _scalar(npz, key, str)
                    if not value.strip():
                        raise ValueError(f"frozen-boundary v3 config scalar {key} must be nonempty")
                else:
                    value = _scalar(npz, key, dtype)
                    if isinstance(value, float) and not np.isfinite(value):
                        raise ValueError(f"frozen-boundary v3 scalar {key} must be finite")
                destination = runtime_config if key.startswith("config_") else sampling_state
                destination[key.removeprefix("config_").removeprefix("sampling_")] = value
            if not bool(sampling_state["is_3d"]):
                raise ValueError("fixed diagnostic arm requires captured 3-D orientations")
            if bool(sampling_state["is_3d_trans"]):
                raise ValueError("fixed diagnostic arm requires captured 2-D translations")
            if (
                int(sampling_state["point_group"]) != 202
                or int(sampling_state["point_group_order"]) != 1
            ):
                raise ValueError(
                    "fixed diagnostic arm requires captured RELION C1 point-group semantics"
                )
            psi_step = float(sampling_state["psi_step_deg"])
            expected_psi_count = 6 * (2 ** int(sampling_state["healpix_order_original"]))
            if psi_step <= 0.0 or psi_angles.size != expected_psi_count:
                raise ValueError(
                    "fixed diagnostic arm psi count/step is inconsistent with HEALPix order"
                )
            expected_psi = np.arange(expected_psi_count, dtype=np.float64) * psi_step
            if 360.0 / float(expected_psi_count) != psi_step or not np.array_equal(
                psi_angles,
                expected_psi,
            ):
                raise ValueError(
                    "fixed diagnostic arm psi count/step and rows must use canonical "
                    "psi-index order"
                )
            for key, dtype in _V3_MAP_LINEAGE_DTYPES.items():
                value = _scalar(npz, key, str)
                if not value.strip():
                    raise ValueError(f"frozen-boundary v3 map-lineage scalar {key} must be nonempty")
                map_lineage[key] = value
            if map_lineage["map_transform_id"] != _V3_MAP_TRANSFORM_ID:
                raise ValueError(
                    "frozen-boundary v3 map_transform_id must equal "
                    f"{_V3_MAP_TRANSFORM_ID!r}"
                )
            for key in (
                "half1_captured_iref_sha256",
                "half2_captured_iref_sha256",
                "half1_transformed_mean_sha256",
                "half2_transformed_mean_sha256",
            ):
                if not _is_lower_sha256(map_lineage[key]):
                    raise ValueError(
                        f"frozen-boundary v3 map-lineage scalar {key} must be lowercase SHA-256"
                    )
            for half, mean in enumerate(means, start=1):
                key = f"half{half}_transformed_mean_sha256"
                observed_digest = _array_bytes_sha256(mean)
                if map_lineage[key] != observed_digest:
                    raise ValueError(
                        f"frozen-boundary v3 {key} does not bind half{half}_mean_ft: "
                        f"expected {map_lineage[key]}, got {observed_digest}"
                    )
            sampling_state["current_size"] = int(_scalar(npz, "current_size", np.int32))
            consumer_iteration = int(sampling_state.pop("consumer_relion_iteration"))
            if consumer_iteration != completed_iteration + 1:
                raise ValueError(
                    "frozen-boundary v3 consumer_relion_iteration must equal "
                    "completed_relion_iteration + 1"
                )
            sampling_state["consumer_relion_iteration"] = consumer_iteration
            sealed_healpix_order = int(_scalar(npz, "healpix_order", np.int32))
            if sampling_state["healpix_order"] != sealed_healpix_order:
                raise ValueError(
                    "frozen-boundary v3 captured/current HEALPix orders differ"
                )
            if sampling_state["healpix_order_original"] != sealed_healpix_order:
                raise ValueError("frozen-boundary v3 original/current HEALPix orders differ")
            if int(sampling_state["coarse_size"]) > int(sampling_state["current_size"]):
                raise ValueError("frozen-boundary v3 coarse_size must not exceed current_size")
            if int(sampling_state["full_size"]) != int(runtime_config["grid_size"]):
                raise ValueError("frozen-boundary v3 full-size/grid-size mismatch")
            if int(runtime_config["n_classes"]) != 1:
                raise ValueError("frozen-boundary v3 currently supports exactly K=1")
            if runtime_config["diagnostic_arm_id"] != FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM:
                raise ValueError(
                    "frozen-boundary v3 must name the fixed real-10076 physical-it2 "
                    "diagnostic arm"
                )
            fixed_runtime_values = {
                "max_iter": 1,
                "skip_final_iteration": True,
                "init_resolution_angstrom": 30.0,
                "offset_range_pixels": 3.0,
                "offset_step_pixels": 1.0,
                "perturb_factor": 0.5,
                "fsc_threshold": 1.0 / 7.0,
                "jax_enable_x64": True,
            }
            for key, expected in fixed_runtime_values.items():
                if runtime_config[key] != expected:
                    raise ValueError(
                        f"frozen-boundary v3 fixed diagnostic arm requires "
                        f"{key}={expected!r}"
                    )
            if (
                runtime_config["provenance_verification_scope"]
                != FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE
                or runtime_config["numerical_classification_scope"]
                != FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE
            ):
                raise ValueError(
                    "frozen-boundary v3 must forbid same-device/numerical-noise "
                    "classification while hardware/toolchain identity is unverified"
                )
            if runtime_config["projector_boundary_kind"] != "reconstructed-projector boundary":
                raise ValueError(
                    "frozen-boundary v3 control must be labelled "
                    "'reconstructed-projector boundary'; exact captured-projector claims "
                    "require direct captured-projector consumption"
                )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", runtime_config["replay_prefix"]):
                raise ValueError("frozen-boundary v3 replay_prefix is invalid")
            for key in ("declared_relion_base_git_commit", "recovar_git_commit"):
                commit = runtime_config[key]
                if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
                    raise ValueError(f"frozen-boundary v3 {key} must be a lowercase 40-hex commit")
            if int(runtime_config["adaptive_oversampling"]) < 0:
                raise ValueError("frozen-boundary v3 adaptive oversampling must be nonnegative")
            if float(sampling_state["sigma_rot_deg"]) < 0.0 or float(sampling_state["sigma_psi_deg"]) < 0.0:
                raise ValueError("frozen-boundary v3 angular prior sigmas must be nonnegative")
            mean_variance_per_half = tuple(
                _finite_array(npz, f"half{half}_mean_variance", ndim=1, dtype=np.float32)
                for half in (1, 2)
            )
            if any(value.shape != (volume_size,) or np.any(value < 0.0) for value in mean_variance_per_half):
                raise ValueError("frozen-boundary v3 per-half mean_variance has invalid shape or values")
            mean_variance_average = np.asarray(
                0.5
                * (
                    mean_variance_per_half[0].astype(np.float64)
                    + mean_variance_per_half[1].astype(np.float64)
                ),
                dtype=np.float32,
            )
            if not np.array_equal(mean_variance, mean_variance_average):
                raise ValueError(
                    "frozen-boundary v3 shared mean_variance must be the explicit float32 "
                    "average of the two exact half priors"
                )
        else:
            mean_variance_per_half = (mean_variance, mean_variance)
            source_roles = {}
            map_lineage = {}

        result = FrozenRefinementBoundary(
            source_dir=source_dir,
            source_manifest=source_manifest,
            source_manifest_sha256=_sha256(source_manifest),
            boundary_sha256=actual_sha256,
            completed_relion_iteration=completed_iteration,
            consumer_relion_iteration=(
                int(sampling_state["consumer_relion_iteration"])
                if schema == FROZEN_BOUNDARY_SCHEMA_V3
                else completed_iteration + 1
            ),
            volume_shape=tuple(int(value) for value in volume_shape),
            current_size=_scalar(npz, "current_size", np.int32),
            healpix_order=_scalar(npz, "healpix_order", np.int32),
            relion_incr_size=_scalar(npz, "relion_incr_size", np.int32),
            has_high_fsc_at_limit=_scalar(npz, "has_high_fsc_at_limit", np.bool_),
            means=means,
            mean_variance=mean_variance,
            mean_variance_per_half=mean_variance_per_half,
            noise_radial_per_half=noise_radial_per_half,
            fsc=fsc,
            ave_pmax=_scalar(npz, "ave_pmax", np.float64),
            previous_best_rotation_eulers=eulers,
            previous_best_translations=translations,
            image_names_per_half=image_names,
            source_rows_per_half=source_rows,
            random_subsets_per_half=random_subsets,
            half_indices_per_half=half_indices,
            half_local_indices_per_half=half_local_indices,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            direction_prior_per_half=direction_prior_per_half,
            translation_sigma_angstrom_per_half=translation_sigma_angstrom_per_half,
            refinement_state_fields=refinement_state_fields,
            source_job_id=int(source_job_id),
            source_arm=provenance_strings["source_arm"],
            source_map_serialization=provenance_strings["source_map_serialization"],
            bitwise_identity_to_original_in_memory_means=bool(bitwise_identity),
            correction_state_owner=provenance_strings["correction_state_owner"],
            identity_schema=provenance_strings["identity_schema"],
            source_star_sha256=provenance_hashes["source_star_sha256"],
            relion_half_star_sha256=provenance_hashes["relion_half_star_sha256"],
            schema=schema,
            fixed_diagnostic_arm=(schema == FROZEN_BOUNDARY_SCHEMA_V3),
            source_sha256=source_sha256,
            sampling_state=sampling_state,
            runtime_config=runtime_config,
            source_roles=source_roles,
            map_lineage=map_lineage,
        )

    if result.current_size <= 0 or result.healpix_order < 0 or result.relion_incr_size <= 0:
        raise ValueError("frozen-boundary schedule scalars are invalid")
    if not np.isfinite(result.ave_pmax) or not 0.0 <= result.ave_pmax <= 1.0:
        raise ValueError("frozen-boundary ave_pmax must be finite and in [0, 1]")
    return result
