"""Fail-closed loader for diagnostic frozen EM iteration boundaries.

The production refinement path does not resume from this format.  It exists
only for causal parity experiments where several one-iteration arms must start
from byte-identical, explicitly sealed state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FROZEN_BOUNDARY_SCHEMA = "recovar.em.frozen_boundary.v2"
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
_EXPECTED_PROVENANCE_STRINGS = {
    "source_map_serialization": "in_memory_complex64",
    "correction_state_owner": "sealed_boundary",
    "identity_schema": "five_field.v1",
}
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


@dataclass(frozen=True)
class FrozenRefinementBoundary:
    source_dir: Path
    source_manifest: Path
    source_manifest_sha256: str
    boundary_sha256: str
    completed_relion_iteration: int
    volume_shape: tuple[int, int, int]
    current_size: int
    healpix_order: int
    relion_incr_size: int
    has_high_fsc_at_limit: bool
    means: tuple[np.ndarray, np.ndarray]
    mean_variance: np.ndarray
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if schema != FROZEN_BOUNDARY_SCHEMA:
            raise ValueError(
                f"unsupported frozen-boundary schema {schema!r}; expected {FROZEN_BOUNDARY_SCHEMA!r}"
            )
        payload_keys = set(npz.files)
        missing_keys = sorted(_REQUIRED_PAYLOAD_KEYS - payload_keys)
        if missing_keys:
            raise ValueError(f"frozen boundary is missing required schema-v2 keys: {missing_keys}")
        unknown_keys = sorted(payload_keys - _ALLOWED_PAYLOAD_KEYS)
        if unknown_keys:
            raise ValueError(f"frozen boundary contains unknown schema-v2 keys: {unknown_keys}")
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
        for key, expected_value in _EXPECTED_PROVENANCE_STRINGS.items():
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
        if not bitwise_identity:
            raise ValueError(
                "frozen-boundary maps must be bitwise-identical to the original in-memory means"
            )
        provenance_hashes = {
            key: _scalar(npz, key, str)
            for key in ("source_star_sha256", "relion_half_star_sha256")
        }
        for key, digest in provenance_hashes.items():
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(
                    f"frozen-boundary provenance scalar {key} must be lowercase SHA-256"
                )

        result = FrozenRefinementBoundary(
            source_dir=source_dir,
            source_manifest=source_manifest,
            source_manifest_sha256=_sha256(source_manifest),
            boundary_sha256=actual_sha256,
            completed_relion_iteration=completed_iteration,
            volume_shape=tuple(int(value) for value in volume_shape),
            current_size=_scalar(npz, "current_size", np.int32),
            healpix_order=_scalar(npz, "healpix_order", np.int32),
            relion_incr_size=_scalar(npz, "relion_incr_size", np.int32),
            has_high_fsc_at_limit=_scalar(npz, "has_high_fsc_at_limit", np.bool_),
            means=means,
            mean_variance=mean_variance,
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
        )

    if result.current_size <= 0 or result.healpix_order < 0 or result.relion_incr_size <= 0:
        raise ValueError("frozen-boundary schedule scalars are invalid")
    if not np.isfinite(result.ave_pmax) or not 0.0 <= result.ave_pmax <= 1.0:
        raise ValueError("frozen-boundary ave_pmax must be finite and in [0, 1]")
    return result
