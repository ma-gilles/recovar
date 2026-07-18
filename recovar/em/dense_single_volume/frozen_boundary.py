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

FROZEN_BOUNDARY_SCHEMA = "recovar.em.frozen_boundary.v1"
FROZEN_BOUNDARY_FILENAME = "frozen_boundary_v1.npz"
FROZEN_BOUNDARY_MANIFEST = "FROZEN_BOUNDARY_SHA256SUMS"


@dataclass(frozen=True)
class FrozenRefinementBoundary:
    source_dir: Path
    source_manifest: Path
    source_manifest_sha256: str
    boundary_sha256: str
    completed_relion_iteration: int
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
    image_corrections: tuple[np.ndarray, np.ndarray] | None
    scale_corrections: tuple[np.ndarray, np.ndarray] | None
    refinement_state_fields: dict[str, float | int | bool]


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


def _finite_array(npz, key: str, *, ndim: int, dtype) -> np.ndarray:
    value = _required_array(npz, key, ndim=ndim).astype(dtype, copy=False)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"frozen boundary array {key} contains non-finite values")
    return value


def _scalar(npz, key: str, dtype):
    value = _required_array(npz, key)
    if value.shape != ():
        raise ValueError(f"frozen boundary scalar {key} must have shape (), got {value.shape}")
    return dtype(value.item())


def _load_optional_half_pair(npz, prefix: str, dtype) -> tuple[np.ndarray, np.ndarray] | None:
    keys = (f"half1_{prefix}", f"half2_{prefix}")
    present = [key in npz.files for key in keys]
    if any(present) and not all(present):
        raise ValueError(f"frozen boundary must provide both or neither of {keys}")
    if not any(present):
        return None
    return tuple(_finite_array(npz, key, ndim=1, dtype=dtype) for key in keys)


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
        completed_iteration = _scalar(npz, "completed_relion_iteration", int)
        if completed_iteration < 1:
            raise ValueError("completed_relion_iteration must be positive")
        volume_shape = _required_array(npz, "volume_shape", ndim=1).astype(np.int64, copy=False)
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
        for half, (euler, translation, names) in enumerate(
            zip(eulers, translations, image_names, strict=True), start=1
        ):
            if euler.shape[1:] != (3,) or translation.shape[1:] != (2,):
                raise ValueError(f"frozen-boundary half-{half} pose arrays have invalid shape")
            if euler.shape[0] != translation.shape[0] or euler.shape[0] != names.shape[0]:
                raise ValueError(f"frozen-boundary half-{half} pose/identity row counts differ")
            if names.dtype.kind not in {"U", "S"}:
                raise ValueError(f"frozen-boundary half-{half} image names must be fixed-width strings")
        all_names = np.concatenate([np.asarray(value, dtype=str) for value in image_names])
        if np.unique(all_names).size != all_names.size:
            raise ValueError("frozen-boundary image names must be globally unique")

        image_corrections = _load_optional_half_pair(npz, "image_corrections", np.float32)
        scale_corrections = _load_optional_half_pair(npz, "scale_corrections", np.float32)
        if (image_corrections is None) != (scale_corrections is None):
            raise ValueError("frozen boundary must provide image and scale corrections together")
        if image_corrections is not None:
            for half in range(2):
                expected_rows = eulers[half].shape[0]
                if image_corrections[half].shape != (expected_rows,):
                    raise ValueError("frozen-boundary image-correction row count mismatch")
                if scale_corrections[half].shape != (expected_rows,):
                    raise ValueError("frozen-boundary scale-correction row count mismatch")

        state_field_types = {
            "current_resolution": float,
            "previous_resolution": float,
            "nr_iter_wo_resol_gain": int,
            "nr_iter_wo_assignment_changes": int,
            "nr_iter_wo_large_hidden_variable_changes": int,
            "ave_Pmax": float,
            "current_changes_optimal_orientations": float,
            "current_changes_optimal_offsets_angstrom": float,
            "smallest_changes_optimal_orientations": float,
            "smallest_changes_optimal_offsets_angstrom": float,
            "acc_rot": float,
            "acc_trans": float,
            "has_converged": bool,
        }
        refinement_state_fields = {
            key: _scalar(npz, f"state_{key}", field_type)
            for key, field_type in state_field_types.items()
            if f"state_{key}" in npz.files
        }
        if refinement_state_fields.get("has_converged", False):
            raise ValueError("a frozen numbered-iteration boundary must not already be converged")

        result = FrozenRefinementBoundary(
            source_dir=source_dir,
            source_manifest=source_manifest,
            source_manifest_sha256=_sha256(source_manifest),
            boundary_sha256=actual_sha256,
            completed_relion_iteration=completed_iteration,
            current_size=_scalar(npz, "current_size", int),
            healpix_order=_scalar(npz, "healpix_order", int),
            relion_incr_size=_scalar(npz, "relion_incr_size", int),
            has_high_fsc_at_limit=_scalar(npz, "has_high_fsc_at_limit", bool),
            means=means,
            mean_variance=mean_variance,
            noise_radial_per_half=noise_radial_per_half,
            fsc=fsc,
            ave_pmax=_scalar(npz, "ave_pmax", float),
            previous_best_rotation_eulers=eulers,
            previous_best_translations=translations,
            image_names_per_half=tuple(np.asarray(value, dtype=str) for value in image_names),
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            refinement_state_fields=refinement_state_fields,
        )

    if result.current_size <= 0 or result.healpix_order < 0 or result.relion_incr_size <= 0:
        raise ValueError("frozen-boundary schedule scalars are invalid")
    if not np.isfinite(result.ave_pmax) or not 0.0 <= result.ave_pmax <= 1.0:
        raise ValueError("frozen-boundary ave_pmax must be finite and in [0, 1]")
    return result
