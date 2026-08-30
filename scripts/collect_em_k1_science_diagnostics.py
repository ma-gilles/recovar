#!/usr/bin/env python3
"""Collect proper-rigid and common-mask FSC diagnostics for a K=1 comparison.

One continuous proper rotation and subpixel translation are fitted from the
low-frequency merged maps.  That transform is then frozen and applied
unchanged to the RECOVAR merged map and both RECOVAR half maps.  Reflections,
density-sign changes, and scale fitting are never searched.  A single soft
mask, constructed symmetrically from the two aligned merged maps, is applied
to both engines and is diagnostic only.

The primary raw FSC curves remain the responsibility of the independently
pinned real-data collector.  This script emits only the explicitly auditable
alignment fallback and supporting masked FSC artifacts consumed by
``summarize_em_k1_realdata_science_equivalence.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import fft as scipy_fft
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation

from recovar.core.mask import make_mask
from recovar.em.initial_model.gt_metrics import (
    align_volume_to_reference,
    lowpass_volume_by_shell,
    relion_alignment_rotations,
)
from recovar.utils import helpers

SCHEMA = "recovar.em_k1_science_diagnostics.v1"
ALIGNED_CURVE_KEYS = (
    "final_cross_engine_proper_aligned",
    "final_cross_engine_half1_proper_aligned",
    "final_cross_engine_half2_proper_aligned",
)
MASKED_CURVE_KEYS = (
    "relion_final_half_fsc_common_masked",
    "recovar_final_half_fsc_common_masked",
    "final_cross_engine_proper_aligned_common_masked",
    "final_cross_engine_half1_proper_aligned_common_masked",
    "final_cross_engine_half2_proper_aligned_common_masked",
)
DEFAULT_FIT_MAX_SHELL = 32
DEFAULT_COARSE_HEALPIX_ORDER = 1
DEFAULT_REFINE_HEALPIX_ORDERS = (2,)
DEFAULT_MAX_CONTINUOUS_ROTATION_DEGREES = 20.0
DEFAULT_MAX_TRANSLATION_FIT_VOXELS = 6.0
DEFAULT_MASK_LOWPASS_DIVISOR = 128
DEFAULT_MASK_EXTEND_DIVISOR = 32
DEFAULT_MASK_SOFT_EDGE_DIVISOR = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _proper_rotation_metrics(matrix: np.ndarray) -> tuple[float, float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    _require(rotation.shape == (3, 3), "rotation must have shape (3, 3)")
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    return determinant, orthogonality_error


def apply_proper_rigid_transform(
    volume: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_zyx: Sequence[float],
    *,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Rotate about the array center, then translate in output ZYX voxels."""

    source = np.asarray(volume, dtype=np.float32)
    _require(source.ndim == 3 and len(set(source.shape)) == 1, f"expected cubic volume, got {source.shape}")
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    translation = np.asarray(translation_zyx, dtype=np.float64)
    determinant, orthogonality_error = _proper_rotation_metrics(rotation)
    _require(determinant > 0.0 and abs(determinant - 1.0) <= 1.0e-6, "transform is not a proper rotation")
    _require(orthogonality_error <= 1.0e-6, "transform is not orthogonal")
    _require(translation.shape == (3,) and np.all(np.isfinite(translation)), "translation must be three finite values")

    center = (np.asarray(source.shape, dtype=np.float64) - 1.0) * 0.5
    inverse = rotation.T
    offset = center - inverse @ (center + translation)
    return ndimage.affine_transform(
        source,
        inverse,
        offset=offset,
        output_shape=source.shape,
        output=np.float32,
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
        prefilter=bool(interpolation_order > 1),
    )


def _centered_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=np.float64).reshape(-1)
    right = np.asarray(rhs, dtype=np.float64).reshape(-1)
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return float(np.dot(left, right) / denominator)


def fit_proper_rigid_transform(
    recovar_merged: np.ndarray,
    relion_merged: np.ndarray,
    *,
    fit_max_shell: int = DEFAULT_FIT_MAX_SHELL,
    coarse_healpix_order: int = DEFAULT_COARSE_HEALPIX_ORDER,
    refine_healpix_orders: tuple[int, ...] = DEFAULT_REFINE_HEALPIX_ORDERS,
    max_continuous_rotation_degrees: float = DEFAULT_MAX_CONTINUOUS_ROTATION_DEGREES,
    max_translation_fit_voxels: float = DEFAULT_MAX_TRANSLATION_FIT_VOXELS,
    interpolation_order: int = 1,
    seed_rotation_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one six-DOF proper transform on compact low-frequency merged maps."""

    recovar = np.asarray(recovar_merged, dtype=np.float32)
    relion = np.asarray(relion_merged, dtype=np.float32)
    _require(recovar.shape == relion.shape, "merged-map shapes differ")
    _require(recovar.ndim == 3 and len(set(recovar.shape)) == 1, "merged maps must be cubic")
    _require(np.all(np.isfinite(recovar)) and np.all(np.isfinite(relion)), "merged maps contain non-finite values")
    full_size = int(recovar.shape[0])
    max_shell = int(fit_max_shell)
    _require(4 <= max_shell < full_size // 2, "fit_max_shell must be in [4, box_size/2)")
    fit_size = min(full_size, 2 * max_shell + 1)

    recovar_fit = lowpass_volume_by_shell(recovar, max_shell, output_size=fit_size).astype(np.float32)
    relion_fit = lowpass_volume_by_shell(relion, max_shell, output_size=fit_size).astype(np.float32)
    if seed_rotation_matrix is None:
        coarse_rotations = relion_alignment_rotations(int(coarse_healpix_order))
        seed = align_volume_to_reference(
            recovar_fit,
            relion_fit,
            coarse_rotations,
            score_max_shell=max_shell,
            allow_mirror=False,
            allow_sign=False,
            interpolation_order=int(interpolation_order),
            refine_orders=tuple(int(order) for order in refine_healpix_orders),
        )
        seed_rotation = np.asarray(seed.rotation_matrix, dtype=np.float64)
        seed_source = "RELION_HEALPix_grid"
    else:
        seed_rotation = np.asarray(seed_rotation_matrix, dtype=np.float64)
        determinant, orthogonality = _proper_rotation_metrics(seed_rotation)
        _require(abs(determinant - 1.0) <= 1.0e-6, "seed matrix is not a proper rotation")
        _require(orthogonality <= 1.0e-6, "seed matrix is not orthogonal")
        seed_source = "explicit_test_or_replay_seed"
    seed_rotated = apply_proper_rigid_transform(
        recovar_fit,
        seed_rotation,
        (0.0, 0.0, 0.0),
        interpolation_order=interpolation_order,
    )
    phase_shift, phase_error, _ = phase_cross_correlation(
        relion_fit,
        seed_rotated,
        upsample_factor=10,
        normalization="phase",
    )
    initial_shift = np.clip(
        np.asarray(phase_shift, dtype=np.float64),
        -float(max_translation_fit_voxels),
        float(max_translation_fit_voxels),
    )
    initial = np.concatenate([np.zeros(3, dtype=np.float64), initial_shift])
    rotation_bound = np.deg2rad(float(max_continuous_rotation_degrees))
    shift_bound = float(max_translation_fit_voxels)
    bounds = [(-rotation_bound, rotation_bound)] * 3 + [(-shift_bound, shift_bound)] * 3

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = Rotation.from_rotvec(np.asarray(parameters[:3], dtype=np.float64)).as_matrix()
        rotation = delta @ seed_rotation
        return rotation, np.asarray(parameters[3:], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        rotation, translation = unpack(parameters)
        candidate = apply_proper_rigid_transform(
            recovar_fit,
            rotation,
            translation,
            interpolation_order=interpolation_order,
        )
        correlation = _centered_correlation(candidate, relion_fit)
        return 1.0 if not math.isfinite(correlation) else -correlation

    result = optimize.minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 120, "xtol": 1.0e-4, "ftol": 1.0e-8},
    )
    _require(np.all(np.isfinite(result.x)), "continuous rigid optimizer returned non-finite parameters")
    rotation_fit, translation_fit = unpack(result.x)
    determinant, orthogonality_error = _proper_rotation_metrics(rotation_fit)
    _require(determinant > 0.0 and abs(determinant - 1.0) <= 1.0e-8, "fitted matrix is not a proper rotation")
    _require(orthogonality_error <= 1.0e-8, "fitted matrix is not orthogonal")
    full_translation = translation_fit * (float(full_size) / float(fit_size))
    return {
        "rotation_matrix_recovar_to_relion": rotation_fit,
        "translation_recovar_to_relion_zyx": full_translation,
        "determinant": determinant,
        "orthogonality_frobenius": orthogonality_error,
        "fit_box_size": fit_size,
        "fit_max_shell_full_box": max_shell,
        "seed_healpix_order": int(coarse_healpix_order),
        "seed_source": seed_source,
        "refine_healpix_orders": [int(order) for order in refine_healpix_orders],
        "seed_rotation_matrix": seed_rotation,
        "phase_correlation_seed_shift_fit_zyx": np.asarray(phase_shift, dtype=np.float64),
        "phase_correlation_error": float(phase_error),
        "continuous_delta_rotvec_radians": np.asarray(result.x[:3], dtype=np.float64),
        "continuous_translation_fit_zyx": translation_fit,
        "fit_correlation": float(-objective(result.x)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_function_evaluations": int(result.nfev),
        "interpolation_order": int(interpolation_order),
    }


class ShellFscCalculator:
    """Memory-conscious signed shellwise FSC calculator for one cubic box."""

    def __init__(self, box_size: int):
        self.box_size = int(box_size)
        _require(self.box_size >= 8, "box size must be at least 8")
        frequencies = (np.fft.fftfreq(self.box_size) * self.box_size).astype(np.float32)
        xy_squared = frequencies[:, None] ** 2 + frequencies[None, :] ** 2
        shells = np.empty((self.box_size, self.box_size, self.box_size), dtype=np.int16)
        for z_index, z_frequency in enumerate(frequencies):
            shells[z_index] = np.rint(np.sqrt(xy_squared + z_frequency * z_frequency)).astype(np.int16)
        self._shells = shells.reshape(-1)

    def fourier(self, volume: np.ndarray) -> np.ndarray:
        array = np.asarray(volume, dtype=np.float32)
        _require(array.shape == (self.box_size,) * 3, f"unexpected FSC volume shape {array.shape}")
        _require(np.all(np.isfinite(array)), "FSC volume contains non-finite values")
        return scipy_fft.fftn(array, workers=-1)

    def curve_from_fourier(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        _require(lhs.shape == rhs.shape == (self.box_size,) * 3, "Fourier array shapes differ")
        product = lhs * np.conj(rhs)
        numerator = np.bincount(self._shells, weights=np.real(product).reshape(-1))
        lhs_power = np.bincount(self._shells, weights=(np.abs(lhs) ** 2).reshape(-1))
        rhs_power = np.bincount(self._shells, weights=(np.abs(rhs) ** 2).reshape(-1))
        denominator = np.sqrt(lhs_power * rhs_power)
        output = np.full(numerator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=output, where=denominator > 0.0)
        return output[: self.box_size // 2 - 1]

    def curves(self, volumes: Mapping[str, np.ndarray], pairs: Mapping[str, tuple[str, str]]) -> dict[str, np.ndarray]:
        transforms = {name: self.fourier(volume) for name, volume in volumes.items()}
        return {
            curve_name: self.curve_from_fourier(transforms[left], transforms[right])
            for curve_name, (left, right) in pairs.items()
        }


def construct_common_soft_mask(
    aligned_recovar_merged: np.ndarray,
    relion_merged: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct one symmetric supporting mask from the two aligned merged maps."""

    recovar = np.asarray(aligned_recovar_merged, dtype=np.float32)
    relion = np.asarray(relion_merged, dtype=np.float32)
    _require(recovar.shape == relion.shape, "mask source shapes differ")

    def unit_rms(volume: np.ndarray) -> np.ndarray:
        centered = volume - np.float32(np.mean(volume, dtype=np.float64))
        rms = float(np.sqrt(np.mean(np.asarray(centered, dtype=np.float64) ** 2)))
        _require(rms > 0.0 and math.isfinite(rms), "mask source has zero or invalid RMS")
        return centered / np.float32(rms)

    consensus = np.float32(0.5) * (unit_rms(recovar) + unit_rms(relion))
    box_size = int(consensus.shape[0])
    lowpass_sigma = max(2, int(math.ceil(box_size / DEFAULT_MASK_LOWPASS_DIVISOR)))
    extend = max(1, int(math.ceil(box_size / DEFAULT_MASK_EXTEND_DIVISOR)))
    soft_edge = max(1, int(math.ceil(box_size / DEFAULT_MASK_SOFT_EDGE_DIVISOR)))
    mask = make_mask(
        consensus,
        threshold="auto",
        lowpass_sigma=lowpass_sigma,
        extend=extend,
        soft_edge=soft_edge,
        cleanup=True,
    )
    _require(mask.shape == consensus.shape, "common mask shape changed")
    _require(np.all(np.isfinite(mask)), "common mask contains non-finite values")
    _require(float(np.min(mask)) >= 0.0 and float(np.max(mask)) <= 1.0, "common mask lies outside [0, 1]")
    _require(np.any(mask > 0.5) and np.any(mask < 0.5), "common mask is empty or all-ones")
    return np.asarray(mask, dtype=np.float32), {
        "construction": "Otsu soft mask from equal-weight unit-RMS aligned merged-map consensus",
        "engine_symmetric": True,
        "threshold": "auto_otsu",
        "lowpass_sigma_voxels": lowpass_sigma,
        "extend_voxels": extend,
        "soft_edge_voxels": soft_edge,
        "cleanup_fill_holes_keep_largest": True,
        "support_fraction_gt_0p5": float(np.mean(mask > 0.5)),
        "mean_value": float(np.mean(mask, dtype=np.float64)),
    }


def _load_volume(path: Path, frame: str) -> tuple[np.ndarray, float]:
    if frame == "recovar":
        volume, voxel_size = helpers.load_mrc(str(path), return_voxel_size=True)
    elif frame == "relion":
        volume, voxel_size = helpers.load_relion_volume(str(path), return_voxel_size=True)
    else:
        raise ValueError(f"unknown frame {frame!r}")
    array = np.asarray(volume, dtype=np.float32)
    voxel = np.asarray(voxel_size, dtype=np.float64).reshape(-1)
    _require(voxel.size >= 1 and np.all(np.isfinite(voxel)), f"invalid voxel size in {path}")
    return array, float(voxel[0])


def collect_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fit and write the sealed JSON, NPZ, and common-mask artifacts."""

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "recovar_merged": args.recovar_merged.resolve(),
        "recovar_half1": args.recovar_half1.resolve(),
        "recovar_half2": args.recovar_half2.resolve(),
        "relion_merged": args.relion_merged.resolve(),
        "relion_half1": args.relion_half1.resolve(),
        "relion_half2": args.relion_half2.resolve(),
    }
    for label, path in paths.items():
        _require(path.is_file(), f"missing {label}: {path}")

    recovar_merged, recovar_voxel = _load_volume(paths["recovar_merged"], "recovar")
    recovar_half1, recovar_half1_voxel = _load_volume(paths["recovar_half1"], "recovar")
    recovar_half2, recovar_half2_voxel = _load_volume(paths["recovar_half2"], "recovar")
    relion_merged, relion_voxel = _load_volume(paths["relion_merged"], "relion")
    relion_half1, relion_half1_voxel = _load_volume(paths["relion_half1"], "relion")
    relion_half2, relion_half2_voxel = _load_volume(paths["relion_half2"], "relion")
    volumes = (recovar_merged, recovar_half1, recovar_half2, relion_merged, relion_half1, relion_half2)
    shapes = {volume.shape for volume in volumes}
    _require(len(shapes) == 1, f"final-map shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))
    _require(len(shape) == 3 and len(set(shape)) == 1, f"final maps are not cubic: {shape}")
    voxel_sizes = (
        recovar_voxel,
        recovar_half1_voxel,
        recovar_half2_voxel,
        relion_voxel,
        relion_half1_voxel,
        relion_half2_voxel,
    )
    _require(max(voxel_sizes) - min(voxel_sizes) <= 1.0e-5, f"final-map voxel sizes differ: {voxel_sizes}")

    fit = fit_proper_rigid_transform(
        recovar_merged,
        relion_merged,
        fit_max_shell=int(args.fit_max_shell),
        coarse_healpix_order=int(args.coarse_healpix_order),
        refine_healpix_orders=tuple(args.refine_healpix_order),
        interpolation_order=int(args.interpolation_order),
    )
    rotation = np.asarray(fit["rotation_matrix_recovar_to_relion"], dtype=np.float64)
    translation = np.asarray(fit["translation_recovar_to_relion_zyx"], dtype=np.float64)
    aligned_recovar_merged = apply_proper_rigid_transform(
        recovar_merged, rotation, translation, interpolation_order=args.interpolation_order
    )
    aligned_recovar_half1 = apply_proper_rigid_transform(
        recovar_half1, rotation, translation, interpolation_order=args.interpolation_order
    )
    aligned_recovar_half2 = apply_proper_rigid_transform(
        recovar_half2, rotation, translation, interpolation_order=args.interpolation_order
    )

    fsc = ShellFscCalculator(int(shape[0]))
    aligned_volumes = {
        "recovar_merged": aligned_recovar_merged,
        "recovar_half1": aligned_recovar_half1,
        "recovar_half2": aligned_recovar_half2,
        "relion_merged": relion_merged,
        "relion_half1": relion_half1,
        "relion_half2": relion_half2,
    }
    curves = fsc.curves(
        aligned_volumes,
        {
            ALIGNED_CURVE_KEYS[0]: ("recovar_merged", "relion_merged"),
            ALIGNED_CURVE_KEYS[1]: ("recovar_half1", "relion_half1"),
            ALIGNED_CURVE_KEYS[2]: ("recovar_half2", "relion_half2"),
        },
    )
    common_mask, mask_metadata = construct_common_soft_mask(aligned_recovar_merged, relion_merged)
    mask_path = output_dir / "common_soft_mask.mrc"
    helpers.write_mrc(str(mask_path), common_mask, voxel_size=relion_voxel)
    masked_volumes = {name: volume * common_mask for name, volume in aligned_volumes.items()}
    curves.update(
        fsc.curves(
            masked_volumes,
            {
                MASKED_CURVE_KEYS[0]: ("relion_half1", "relion_half2"),
                MASKED_CURVE_KEYS[1]: ("recovar_half1", "recovar_half2"),
                MASKED_CURVE_KEYS[2]: ("recovar_merged", "relion_merged"),
                MASKED_CURVE_KEYS[3]: ("recovar_half1", "relion_half1"),
                MASKED_CURVE_KEYS[4]: ("recovar_half2", "relion_half2"),
            },
        )
    )

    curves_path = output_dir / "science_diagnostic_curves.npz"
    np.savez_compressed(curves_path, **curves)
    producer_path = Path(__file__).resolve()
    fit_json = {
        key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in fit.items()
    }
    report = {
        "schema": SCHEMA,
        "case_id": str(args.case_id),
        "producer": {"path": str(producer_path), "sha256": sha256_file(producer_path)},
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "box_size": int(shape[0]),
        "voxel_size_angstrom": float(relion_voxel),
        "diagnostics": {
            "proper_so3_alignment": {
                "fit_source": "merged_low_frequency",
                "method": "HEALPix proper-rotation seed plus continuous scipy rotvec Powell and subpixel translation",
                "continuous_so3_refinement": True,
                "translation_subpixel": True,
                "applied_unchanged_to": ["merged", "half1", "half2"],
                "rotation_matrix_recovar_to_relion": rotation.tolist(),
                "translation_recovar_to_relion_zyx": translation.tolist(),
                "no_reflection": True,
                "sign_fit": False,
                "scale_fit": False,
                "symmetry_label": str(args.symmetry_label),
                "symmetry_operators_sha256": str(args.symmetry_operators_sha256),
                **fit_json,
            },
            "common_mask": {
                "path": str(mask_path),
                "sha256": sha256_file(mask_path),
                "applied_identically_to_both_engines": True,
                "acceptance_metric": False,
                **mask_metadata,
            },
        },
        "artifacts": {
            "curve_archive": {
                "path": str(curves_path),
                "sha256": sha256_file(curves_path),
                "fields": sorted(curves),
            },
            "common_mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
        },
        "acceptance_policy": {
            "proper_alignment_can_replace_only_failed_raw_cross_engine_gates": True,
            "within_engine_half_map_quality_remains_mandatory": True,
            "common_mask_can_rescue": False,
        },
    }
    report_path = output_dir / "science_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--recovar-merged", type=Path, required=True)
    parser.add_argument("--recovar-half1", type=Path, required=True)
    parser.add_argument("--recovar-half2", type=Path, required=True)
    parser.add_argument("--relion-merged", type=Path, required=True)
    parser.add_argument("--relion-half1", type=Path, required=True)
    parser.add_argument("--relion-half2", type=Path, required=True)
    parser.add_argument("--symmetry-label", required=True)
    parser.add_argument("--symmetry-operators-sha256", required=True)
    parser.add_argument("--fit-max-shell", type=int, default=DEFAULT_FIT_MAX_SHELL)
    parser.add_argument("--coarse-healpix-order", type=int, default=DEFAULT_COARSE_HEALPIX_ORDER)
    parser.add_argument(
        "--refine-healpix-order",
        type=int,
        action="append",
        default=list(DEFAULT_REFINE_HEALPIX_ORDERS),
    )
    parser.add_argument("--interpolation-order", type=int, choices=(1, 3), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = collect_diagnostics(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
