"""Opt-in CPU rigid GT registration; never part of scientific EM execution.

Forward coordinates are y = c + R M (x - c) + t, where c=(shape-1)/2,
M optionally reflects array axis 0, and t is in full-resolution array-axis
voxels. Contrast sign is fixed at +1. Fit a chosen reference once and reuse
``apply_rigid_volume_transform`` when comparing a trajectory in one frame.
The lowpass fitting objective is not a map-quality metric. The continuous
objective lowpasses on the full grid, evaluates fixed reference coordinates
and recomputes the transformed-volume norm for each proposed rigid transform.
RigidVolumeTransform stores one fitted frame for reuse without refitting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize, signal
from scipy.spatial.transform import Rotation

from recovar.em.diagnostics.gt_metrics import centered_correlation, lowpass_volume_by_shell


@dataclass(frozen=True)
class RigidFitControls:
    score_max_shell: int = 8
    coarse_sample_size: int = 17
    objective_sample_size: int = 25
    allow_mirror: bool = True
    final_interpolation_order: int = 1
    maxiter: int = 35
    maxfev: int = 1600
    xtol: float = 1e-5
    ftol: float = 1e-8


@dataclass(frozen=True)
class RigidFitReceipt:
    controls: RigidFitControls
    method: str
    objective_interpolation_order: int
    boundary_mode: str
    rotation_count: int
    rotation_grid_sha256: str
    coarse_score: float
    coarse_rotation_index: int
    coarse_translation_voxels: tuple[float, float, float]
    optimizer_success: bool
    optimizer_message: str
    optimizer_evaluations: int
    used_seed: bool
    coarse_shift_at_linear_boundary: bool
    translation_outside_quarter_box: bool


@dataclass(frozen=True)
class RigidVolumeAlignment:
    aligned_volume: np.ndarray
    corr: float
    score: float
    rotation_index: int
    rotation_matrix: np.ndarray
    mirror_x: bool
    sign: int
    translation_voxels: np.ndarray
    receipt: RigidFitReceipt


_TRANSFORM_METADATA = {
    "schema": "recovar.rigid_volume_transform.v1",
    "coordinate_frame": "recovar_array_axes_0_1_2",
    "translation_units": "full_resolution_voxels",
    "geometry": "y=c+R*M*(x-c)+t; c=(shape-1)/2; M reflects axis0",
}


def _voxel_size(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Voxel size must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Voxel size must be a finite positive real number")
    return result


@dataclass(frozen=True)
class RigidVolumeTransform:
    """Immutable fit-once frame with exact grid/GT checks, never a fitter.

    Geometry uses the module's forward coordinate convention. ``voxel_size``
    is isotropic Angstrom per voxel; ``gt_sha256`` identifies the GT artifact
    used for the fit. Sign is fixed at +1. Transform identity hashes canonical
    JSON and is separate from the fitting receipt and scientific quality.
    """

    rotation_matrix: tuple[tuple[float, float, float], ...]
    translation_voxels: tuple[float, float, float]
    mirror_x: bool
    volume_shape: tuple[int, int, int]
    voxel_size: float
    gt_sha256: str
    interpolation_order: int = 1
    sign: int = 1

    def __post_init__(self):
        rotation = np.asarray(self.rotation_matrix)
        translation = np.asarray(self.translation_voxels)
        if rotation.dtype.kind not in "fiu" or translation.dtype.kind not in "fiu":
            raise ValueError("Transform geometry must be real numeric values")
        rotation = _rotation(rotation)
        translation = np.asarray(translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Translation must contain three finite real values")
        shape_array = np.asarray(self.volume_shape)
        if shape_array.shape != (3,) or shape_array.dtype.kind not in "iu":
            raise ValueError("Transform shape must be a cubic three-integer grid of size >=3")
        shape = tuple(int(n) for n in shape_array)
        if min(shape) < 3 or len(set(shape)) != 1:
            raise ValueError("Transform shape must be a cubic three-integer grid of size >=3")
        if not isinstance(self.mirror_x, (bool, np.bool_)):
            raise ValueError("mirror_x must be boolean")
        if isinstance(self.sign, (bool, np.bool_)) or not isinstance(self.sign, (int, np.integer)) or self.sign != 1:
            raise ValueError("Rigid transform sign must be integer +1")
        order = self.interpolation_order
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 5:
            raise ValueError("Interpolation order must be an integer in [0,5]")
        digest = self.gt_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("gt_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "rotation_matrix", tuple(tuple(float(v) for v in row) for row in rotation))
        object.__setattr__(self, "translation_voxels", tuple(float(v) for v in translation))
        object.__setattr__(self, "volume_shape", tuple(int(n) for n in shape))
        object.__setattr__(self, "voxel_size", _voxel_size(self.voxel_size))
        object.__setattr__(self, "mirror_x", bool(self.mirror_x))
        object.__setattr__(self, "interpolation_order", int(order))
        object.__setattr__(self, "sign", 1)

    @classmethod
    def from_alignment(cls, alignment: RigidVolumeAlignment, *, volume_shape, voxel_size, gt_sha256):
        """Copy a fitted result's geometry; fitting controls remain separate."""
        if tuple(volume_shape) != alignment.aligned_volume.shape:
            raise ValueError("Declared grid does not match the fitted volume")
        return cls(
            rotation_matrix=alignment.rotation_matrix,
            translation_voxels=alignment.translation_voxels,
            mirror_x=alignment.mirror_x,
            volume_shape=volume_shape,
            voxel_size=voxel_size,
            gt_sha256=gt_sha256,
            interpolation_order=alignment.receipt.controls.final_interpolation_order,
            sign=alignment.sign,
        )

    def to_dict(self) -> dict:
        """Return the complete versioned JSON schema, without an embedded hash."""
        return {
            **_TRANSFORM_METADATA,
            "rotation_matrix": [list(row) for row in self.rotation_matrix],
            "translation_voxels": list(self.translation_voxels),
            "mirror_x": self.mirror_x,
            "volume_shape": list(self.volume_shape),
            "voxel_size": self.voxel_size,
            "gt_sha256": self.gt_sha256,
            "interpolation_order": self.interpolation_order,
            "sign": self.sign,
        }

    @classmethod
    def from_dict(cls, value: dict):
        """Reject missing/unknown fields or changed coordinate conventions."""
        fields = {
            "rotation_matrix",
            "translation_voxels",
            "mirror_x",
            "volume_shape",
            "voxel_size",
            "gt_sha256",
            "interpolation_order",
            "sign",
        }
        if not isinstance(value, dict) or set(value) != fields | set(_TRANSFORM_METADATA):
            raise ValueError("Rigid transform payload has unknown or missing fields")
        if any(value[key] != expected for key, expected in _TRANSFORM_METADATA.items()):
            raise ValueError("Rigid transform schema or coordinate convention differs")
        return cls(**{key: value[key] for key in fields})

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def apply(self, volume: np.ndarray, *, voxel_size: float, gt_sha256: str) -> np.ndarray:
        """Validate the original isotropic grid/GT identity and apply once."""
        if np.shape(volume) != self.volume_shape or _voxel_size(voxel_size) != self.voxel_size:
            raise ValueError("Volume shape or voxel size differs from the fitted grid")
        if not isinstance(gt_sha256, str) or gt_sha256 != self.gt_sha256:
            raise ValueError("GT identity differs from the fitted transform")
        return apply_rigid_volume_transform(
            volume,
            self.rotation_matrix,
            self.translation_voxels,
            mirror_x=self.mirror_x,
            order=self.interpolation_order,
        )


def _volume(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("Registration requires real volumes")
    vol = np.asarray(raw, dtype=np.float64)
    if vol.ndim != 3 or len(set(vol.shape)) != 1 or vol.shape[0] < 3:
        raise ValueError("Registration requires cubic 3D volumes of size >= 3")
    if not np.isfinite(vol).all():
        raise ValueError("Registration volume must be finite")
    return vol


def _rotation(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation must be a finite 3x3 matrix")
    # Permit float32 rotation-grid serialization; this is input validation,
    # not a tolerance on the accuracy of the fitted geometric transform.
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("Rotation must be proper and orthogonal")
    return matrix


def apply_rigid_volume_transform(
    volume: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_voxels: np.ndarray,
    *,
    mirror_x: bool = False,
    order: int = 1,
) -> np.ndarray:
    """Apply y=c+R M(x-c)+t once, with zero fill and fixed contrast sign +1."""
    vol = _volume(volume)
    matrix = _rotation(rotation_matrix)
    translation = np.asarray(translation_voxels, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Translation must contain three finite full-resolution voxel shifts")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 5:
        raise ValueError("Interpolation order must be an integer in [0, 5]")
    transform = matrix @ np.diag([-1.0 if mirror_x else 1.0, 1.0, 1.0])
    center = (np.asarray(vol.shape) - 1) / 2
    return ndimage.affine_transform(
        vol,
        transform.T,
        offset=center - transform.T @ (center + translation),
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def _coordinates(shape: tuple[int, ...], size: int) -> np.ndarray:
    return np.asarray(np.meshgrid(*[np.linspace(0, n - 1, size) for n in shape], indexing="ij"))


def _continuous_score(moving: np.ndarray, reference: np.ndarray, shell: int, size: int):
    """Fixed reference coordinates; transformed norm recomputed at every trial."""
    a = lowpass_volume_by_shell(moving - moving.mean(), shell)
    b = lowpass_volume_by_shell(reference - reference.mean(), shell)
    ac = ndimage.spline_filter(a, order=3, mode="constant")
    bc = ndimage.spline_filter(b, order=3, mode="constant")
    coords = _coordinates(moving.shape, size).reshape(3, -1)
    center = ((np.asarray(moving.shape) - 1) / 2)[:, None]
    target = ndimage.map_coordinates(bc, coords, order=3, mode="constant", cval=0, prefilter=False)

    def score(transform: np.ndarray, translation: np.ndarray) -> float:
        inverse = center + transform.T @ (coords - center - np.asarray(translation)[:, None])
        values = ndimage.map_coordinates(ac, inverse, order=3, mode="constant", cval=0, prefilter=False)
        value = centered_correlation(values, target)
        if not np.isfinite(value):
            raise ValueError("Unidentifiable transformed registration norm")
        return float(value)

    return score


def align_volume_rigid_to_reference(
    volume: np.ndarray,
    reference: np.ndarray,
    rotations: np.ndarray,
    *,
    controls: RigidFitControls = RigidFitControls(),
) -> RigidVolumeAlignment:
    """Fit a linear-correlation seed then continuous normalized rigid objective.

    ``rotation_index`` identifies the seed; the fitted matrix/translation are
    authoritative. A visible unsuccessful optimizer receipt is not acceptance.
    No map, source array or caller-provided rotation grid is modified.
    """
    if not isinstance(controls, RigidFitControls):
        raise ValueError("controls must be RigidFitControls")
    if not isinstance(controls.allow_mirror, (bool, np.bool_)):
        raise ValueError("allow_mirror must be boolean")
    moving, target = _volume(volume), _volume(reference)
    if moving.shape != target.shape:
        raise ValueError("Registration volumes must have identical shapes")
    for value in (moving, target):
        norm = np.linalg.norm(value - value.mean())
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Unidentifiable zero/nonfinite centered registration norm")
    grid = np.asarray(rotations, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[1:] != (3, 3) or not len(grid):
        raise ValueError("A nonempty (R,3,3) rotation grid is required")
    for matrix in grid:
        _rotation(matrix)
    for name in ("score_max_shell", "coarse_sample_size", "objective_sample_size", "maxiter", "maxfev"):
        value = getattr(controls, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min(controls.coarse_sample_size, controls.objective_sample_size) < 3:
        raise ValueError("Sampling boxes must be at least 3")
    for name in ("xtol", "ftol"):
        if not np.isfinite(getattr(controls, name)) or getattr(controls, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    # Validate final interpolation before doing any expensive fitting.
    order = controls.final_interpolation_order
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 5:
        raise ValueError("Final interpolation order must be an integer in [0, 5]")
    size = controls.coarse_sample_size
    coords = _coordinates(moving.shape, size)
    a, b = [
        ndimage.map_coordinates(
            lowpass_volume_by_shell(v - v.mean(), controls.score_max_shell),
            coords,
            order=1,
            mode="constant",
            prefilter=False,
        )
        for v in (moving, target)
    ]
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Unidentifiable sampled registration norm")
    best = None
    for hand in (False, True) if controls.allow_mirror else (False,):
        for index, matrix in enumerate(grid):
            candidate = apply_rigid_volume_transform(a, matrix, np.zeros(3), mirror_x=hand)
            correlation = signal.fftconvolve(b, candidate[::-1, ::-1, ::-1], mode="full") / norm
            peak = np.unravel_index(np.argmax(correlation), correlation.shape)
            value = float(correlation[peak])
            shift = np.asarray(peak) - (size - 1)
            if best is None or value > best[0]:
                best = (value, hand, index, shift)
    coarse_score, hand, index, shift = best
    mirror = np.diag([-1.0 if hand else 1.0, 1.0, 1.0])
    translation = shift * (moving.shape[0] - 1) / (size - 1)
    seed = np.r_[Rotation.from_matrix(grid[index]).as_rotvec(), translation]
    score = _continuous_score(moving, target, controls.score_max_shell, controls.objective_sample_size)

    def objective(x):
        return -score(Rotation.from_rotvec(x[:3]).as_matrix() @ mirror, x[3:])

    initial = objective(seed)
    fit = optimize.minimize(
        objective,
        seed,
        method="Powell",
        options={
            "maxiter": controls.maxiter,
            "maxfev": controls.maxfev,
            "xtol": controls.xtol,
            "ftol": controls.ftol,
        },
    )
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise ValueError("Optimizer returned a nonfinite rigid fit")
    used_seed = bool(fit.fun > initial)
    x = seed if used_seed else fit.x
    matrix = Rotation.from_rotvec(x[:3]).as_matrix()
    aligned = apply_rigid_volume_transform(moving, matrix, x[3:], mirror_x=hand, order=order)
    receipt = RigidFitReceipt(
        controls=controls,
        method="full_grid_lowpass_normalized_rigid_powell.v1",
        objective_interpolation_order=3,
        boundary_mode="constant_zero",
        rotation_count=len(grid),
        rotation_grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
        coarse_score=coarse_score,
        coarse_rotation_index=int(index),
        coarse_translation_voxels=tuple(float(v) for v in translation),
        optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),
        optimizer_evaluations=int(fit.nfev),
        used_seed=used_seed,
        coarse_shift_at_linear_boundary=bool(np.any(np.abs(shift) == size - 1)),
        translation_outside_quarter_box=bool(np.any(np.abs(x[3:]) > moving.shape[0] / 4)),
    )
    return RigidVolumeAlignment(
        aligned_volume=aligned,
        corr=centered_correlation(aligned, target),
        score=-objective(x),
        rotation_index=int(index),
        rotation_matrix=matrix.copy(),
        mirror_x=bool(hand),
        sign=1,
        translation_voxels=x[3:].copy(),
        receipt=receipt,
    )
