"""Frozen K=4 first-iteration winner score replay.

This diagnostic consumes existing RECOVAR and RELION particle captures.  It
does not run EM or modify production scoring.  High-precision arms are gated
behind source-aware float32 controls so a projection, geometry, sign, or score
formula bug cannot be mislabeled as numerical precision.

Intermediate comparisons use exact/array metrics.  This module makes no map
quality claim; map conclusions remain gated by shellwise FSC/FSC-AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from recovar.em.normalized_cc_replay import (
    canonical_float32_reduce,
    canonical_float64_reduce,
    normalized_cc_pixel_contributions,
    relion_coarse_128lane_float32_reduce,
)

SCHEMA = "recovar-k4-firstiter-winner-operand-replay-v1"
SEAL_SCHEMA = "recovar-k4-firstiter-winner-operand-replay-seal-v1"
DEFAULT_RECOVAR_MAX_ULP = 8
DEFAULT_RELION_MAX_ABS = 1.0e-6
DEFAULT_RELION_CENTERED_MAX_ABS = 5.0e-7


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_vector(path: Path, dtype) -> np.ndarray:
    with path.open("rb") as stream:
        count = np.fromfile(stream, np.int64, 1)
        values = np.fromfile(stream, dtype)
    if count.shape != (1,) or values.size != int(count[0]):
        raise ValueError(f"invalid vector serialization: {path}")
    return values


def _read_double_scalar(path: Path) -> float:
    values = np.fromfile(path, np.float64)
    if values.shape != (1,) or not np.isfinite(values[0]):
        raise ValueError(f"invalid double scalar serialization: {path}")
    return float(values[0])


def _array_metrics(left, right) -> dict[str, object]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = np.linalg.norm(left.astype(np.complex128).ravel())
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "relative_l2": float(np.linalg.norm(delta.ravel()) / denominator) if denominator else None,
    }


def _canonical_reorder(values, identities) -> np.ndarray:
    """Return values in deterministic centered-grid identity order."""

    values = np.asarray(values)
    identities = np.asarray(identities, dtype=np.int64)
    if values.ndim != 1 or identities.shape != values.shape:
        raise ValueError(
            f"canonical reorder requires matching vectors, got {values.shape} and {identities.shape}"
        )
    if np.unique(identities).size != identities.size:
        raise ValueError("canonical identities must be unique")
    return values[np.argsort(identities, kind="stable")]


def float32_ulp_distance(left, right) -> np.ndarray:
    """Return elementwise ULP distance for finite float32 arrays."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("ULP comparison requires same-shape finite float32 arrays")

    def ordered(values):
        signed = values.view(np.int32).astype(np.int64)
        return np.where(signed < 0, np.int64(0x80000000) - signed, signed)

    return np.abs(ordered(left) - ordered(right))


def score_control_metrics(production, replay) -> dict[str, object]:
    production = np.asarray(production, dtype=np.float32)
    replay = np.asarray(replay, dtype=np.float32)
    if production.shape != replay.shape:
        raise ValueError(f"score control shape mismatch: {production.shape} != {replay.shape}")
    residual = replay.astype(np.float64) - production.astype(np.float64)
    centered = residual - np.mean(residual, dtype=np.float64)
    ulp = float32_ulp_distance(production, replay)
    return {
        "production": production.astype(np.float64).tolist(),
        "replay": replay.astype(np.float64).tolist(),
        "residual": residual.tolist(),
        "centered_residual": centered.tolist(),
        "ulp_distance": ulp.tolist(),
        "max_abs": float(np.max(np.abs(residual))),
        "centered_max_abs": float(np.max(np.abs(centered))),
        "max_ulp": int(np.max(ulp)),
        "exact_equal": bool(np.array_equal(production, replay)),
    }


def controls_close(
    recovar_metrics: dict[str, object],
    relion_metrics: dict[str, object],
    *,
    recovar_max_ulp: int = DEFAULT_RECOVAR_MAX_ULP,
    relion_max_abs: float = DEFAULT_RELION_MAX_ABS,
    relion_centered_max_abs: float = DEFAULT_RELION_CENTERED_MAX_ABS,
) -> tuple[bool, dict[str, object]]:
    thresholds = {
        "recovar_max_ulp": int(recovar_max_ulp),
        "relion_max_abs": float(relion_max_abs),
        "relion_centered_max_abs": float(relion_centered_max_abs),
    }
    checks = {
        "recovar_exact_production_path": int(recovar_metrics["max_ulp"]) <= int(recovar_max_ulp),
        "relion_captured_float32_absolute": float(relion_metrics["max_abs"]) <= float(relion_max_abs),
        "relion_captured_float32_centered": float(relion_metrics["centered_max_abs"])
        <= float(relion_centered_max_abs),
    }
    return bool(all(checks.values())), {"thresholds": thresholds, "checks": checks}


def _canonical_identities_from_recovar(window_indices, *, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return common centered-grid identities for RECOVAR and RELION row orders."""

    recovar = np.asarray(window_indices, dtype=np.int64)
    current_pixels = int(recovar.size)
    # For even N, packed rFFT pixels P=N*(N/2+1), hence
    # N=-1+sqrt(1+2P).  This is not the triangular-number quadratic.
    current_size = int(round(-1.0 + np.sqrt(1.0 + 2.0 * current_pixels)))
    if current_size * (current_size // 2 + 1) != current_pixels or current_size % 2:
        raise ValueError(f"cannot infer even current size from {current_pixels} packed pixels")
    rows = np.arange(current_size, dtype=np.int64)
    ky = np.where(rows <= current_size // 2, rows, rows - current_size)
    cols = np.arange(current_size // 2 + 1, dtype=np.int64)
    relion = ((ky[:, None] + image_size // 2) * (image_size // 2 + 1) + cols[None, :]).reshape(-1)
    if not np.array_equal(np.sort(recovar), np.sort(relion)):
        raise ValueError("RECOVAR window and RELION current-image identities are not bijective")
    return recovar, relion


def _score_from_reductions(numerator, norm, *, dtype) -> float:
    dtype = np.dtype(dtype)
    numerator = dtype.type(numerator)
    norm = dtype.type(norm)
    if not np.isfinite(numerator) or not np.isfinite(norm) or norm <= dtype.type(0.0):
        raise ValueError(f"invalid normalized-CC reduction numerator={numerator} norm={norm}")
    return float(dtype.type(numerator / np.sqrt(norm, dtype=dtype)))


def _reduction_dict(contributions, *, mode: str, identities=None) -> dict[str, object]:
    if mode == "canonical_float32":
        reducer = canonical_float32_reduce
        dtype = np.float32
    elif mode == "canonical_float64":
        reducer = canonical_float64_reduce
        dtype = np.float64
    elif mode == "original_order_float64":
        if identities is not None:
            raise ValueError("original-order float64 replay cannot reorder contributions")
        reducer = canonical_float64_reduce
        dtype = np.float64
    elif mode == "relion_coarse_128lane_float32":
        if identities is not None:
            raise ValueError("RELION device-order replay cannot reorder contributions")
        reducer = relion_coarse_128lane_float32_reduce
        dtype = np.float32
    else:
        raise ValueError(f"unknown reduction mode {mode!r}")
    if identities is None:
        numerator = reducer(contributions.numerator)
        norm = reducer(contributions.norm)
    else:
        numerator = reducer(contributions.numerator, identities)
        norm = reducer(contributions.norm, identities)
    return {
        "mode": mode,
        "numerator": float(numerator),
        "norm": float(norm),
        "score": _score_from_reductions(numerator, norm, dtype=dtype),
        "dtype_provenance": contributions.provenance.to_dict(),
    }


def _winner_summary(scores) -> dict[str, object]:
    scores = np.asarray(scores)
    winner = int(np.argmax(scores))
    order = np.argsort(scores, kind="stable")
    runner_up = int(order[-2])
    return {
        "scores": scores.astype(np.float64).tolist(),
        "winner_class_zero_based": winner,
        "runner_up_class_zero_based": runner_up,
        "winner_margin": float(scores[winner] - scores[runner_up]),
        "class0_minus_class1": float(scores[0] - scores[1]),
    }


def _discover_relion_prefix(payload: Path) -> str:
    matches = sorted(payload.glob("iter1_rank*_part*_pass0_raw_scores.bin"))
    if len(matches) != 1:
        raise ValueError(f"expected one RELION pass-0 raw-score capture in {payload}, found {len(matches)}")
    return matches[0].name.removesuffix("raw_scores.bin")


def _load_relion_projector(payload: Path, *, rank: int, class_index: int) -> tuple[np.ndarray, int, int]:
    state = f"state_iter1_rank{rank}_device0_class{class_index}_"
    zdim = int(round(_read_double_scalar(payload / f"{state}projector_zdim.bin")))
    ydim = int(round(_read_double_scalar(payload / f"{state}projector_ydim.bin")))
    xdim = int(round(_read_double_scalar(payload / f"{state}projector_xdim.bin")))
    real = _read_vector(payload / f"{state}projector_real.bin", np.float32)
    imag = _read_vector(payload / f"{state}projector_imag.bin", np.float32)
    projector = (real + 1j * imag).reshape(zdim, ydim, xdim).astype(np.complex64)
    r_max = int(round(_read_double_scalar(payload / f"{state}projector_r_max.bin")))
    padding_factor = int(round(_read_double_scalar(payload / f"{state}projector_padding_factor.bin")))
    return projector, r_max, padding_factor


def _regenerate_recovar_projectors(fixture_root: Path, *, n_classes: int, current_size: int, padding_factor: int):
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps
    from recovar.utils.helpers import load_mrc

    references = []
    reference_paths = []
    for class_index in range(n_classes):
        path = fixture_root / f"reference_init_class{class_index + 1:03d}.mrc"
        real = load_mrc(path).astype(np.float32)
        fourier = np.asarray(ftu.get_dft3(jnp.asarray(real))).astype(np.complex64)
        references.append(np.asarray(ftu.get_idft3(jnp.asarray(fourier))).real)
        reference_paths.append(path)
    projectors, r_max = reference_to_relion_projector_half_maps(
        np.asarray(references, dtype=np.float64),
        current_size=int(current_size),
        padding_factor=int(padding_factor),
    )
    return np.asarray(projectors, dtype=np.complex64), int(r_max), reference_paths


def _recovar_float32_arms(
    projectors,
    *,
    rotation,
    shifted_data,
    score_weight,
    half_weights,
    window_indices,
    production_scores,
    image_size: int,
    current_size: int,
    r_max: int,
    padding_factor: int,
    translation_index: int,
    canonical_identities,
) -> tuple[dict[str, object], list[np.ndarray]]:
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.projection import compute_relion_projector_projections_block
    from recovar.em.dense_single_volume.helpers.scoring import _e_step_block_scores_windowed_normalized_cc

    replay_scores = []
    canonical_f32 = []
    promoted_f64 = []
    promoted_original_f64 = []
    projections = []
    for projector in np.asarray(projectors):
        projection, projection_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(np.asarray(rotation, dtype=np.float32)[None]),
            (image_size, image_size),
            r_max=int(r_max),
            padding_factor=int(padding_factor),
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(current_size),
            pixel_indices=np.asarray(window_indices, dtype=np.int32),
            relion_texture_interp=True,
        )
        score = _e_step_block_scores_windowed_normalized_cc(
            jnp.asarray(shifted_data),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.asarray(score_weight),
            projection * jnp.asarray(half_weights)[None],
            projection_abs2 * jnp.asarray(half_weights)[None],
            1,
            int(shifted_data.shape[0]),
            int(window_indices.size),
            (image_size, image_size),
            (image_size, image_size, image_size),
        )
        projection_np = np.asarray(projection[0], dtype=np.complex64)
        replay_scores.append(np.asarray(score, dtype=np.float32)[0, 0, translation_index])
        projections.append(projection_np)

        captured = normalized_cc_pixel_contributions(
            projection_np,
            np.asarray(shifted_data[translation_index], dtype=np.complex64),
            np.asarray(score_weight[0], dtype=np.float32),
            np.asarray(half_weights, dtype=np.float32),
        )
        canonical_f32.append(_reduction_dict(captured, mode="canonical_float32", identities=canonical_identities))
        promoted = normalized_cc_pixel_contributions(
            projection_np.astype(np.complex128),
            np.asarray(shifted_data[translation_index], dtype=np.complex64).astype(np.complex128),
            np.asarray(score_weight[0], dtype=np.float32).astype(np.float64),
            np.asarray(half_weights, dtype=np.float32).astype(np.float64),
            arithmetic_dtype=np.float64,
            precision_origin="promoted_captured",
        )
        promoted_f64.append(_reduction_dict(promoted, mode="canonical_float64", identities=canonical_identities))
        promoted_original_f64.append(_reduction_dict(promoted, mode="original_order_float64"))

    replay_scores = np.asarray(replay_scores, dtype=np.float32)
    return {
        "production": _winner_summary(np.asarray(production_scores, dtype=np.float32)),
        "production_helper_float32": _winner_summary(replay_scores),
        "control_metrics": score_control_metrics(production_scores, replay_scores),
        "canonical_float32": _winner_summary([entry["score"] for entry in canonical_f32]),
        "canonical_float32_reductions": canonical_f32,
        "promoted_captured_float64": _winner_summary([entry["score"] for entry in promoted_f64]),
        "promoted_captured_float64_reductions": promoted_f64,
        "promoted_captured_float64_original_order": _winner_summary(
            [entry["score"] for entry in promoted_original_f64]
        ),
        "promoted_captured_float64_original_order_reductions": promoted_original_f64,
    }, projections


def _relion_shifted_image(fimg, phase_x: float, phase_y: float, *, current_size: int) -> np.ndarray:
    rows = np.arange(current_size, dtype=np.int32)
    y = np.where(rows <= current_size // 2, rows, rows - current_size).astype(np.float32)
    x = np.arange(current_size // 2 + 1, dtype=np.float32)
    angle = np.add(
        np.multiply(y[:, None], np.float32(phase_y), dtype=np.float32),
        np.multiply(x[None, :], np.float32(phase_x), dtype=np.float32),
        dtype=np.float32,
    )
    sine = np.sin(angle, dtype=np.float32)
    cosine = np.cos(angle, dtype=np.float32)
    fimg = np.asarray(fimg, dtype=np.complex64).reshape(current_size, current_size // 2 + 1)
    real = np.subtract(
        np.multiply(cosine, fimg.real, dtype=np.float32),
        np.multiply(sine, fimg.imag, dtype=np.float32),
        dtype=np.float32,
    )
    imag = np.add(
        np.multiply(cosine, fimg.imag, dtype=np.float32),
        np.multiply(sine, fimg.real, dtype=np.float32),
        dtype=np.float32,
    )
    return (real + 1j * imag).astype(np.complex64).reshape(-1)


def _relion_cc_contributions(
    projection,
    shifted_image,
    score_weight,
    half_weights,
    *,
    arithmetic_dtype=np.float32,
    precision_origin="captured_production",
):
    """Build RELION CC terms, including its numerator ``corr_img`` factor.

    RELION's coarse CUDA kernel accumulates ``dot(ref, shifted) * corr_img``
    for the numerator and ``abs2(ref) * corr_img`` for the norm.  The generic
    normalized-CC helper expects that numerator weight to already be folded
    into its image operand.
    """

    arithmetic_dtype = np.dtype(arithmetic_dtype)
    complex_dtype = np.complex64 if arithmetic_dtype == np.dtype(np.float32) else np.complex128
    weighted_image = np.multiply(
        np.asarray(shifted_image, dtype=complex_dtype),
        np.asarray(score_weight, dtype=arithmetic_dtype),
        dtype=complex_dtype,
    )
    return normalized_cc_pixel_contributions(
        np.asarray(projection, dtype=complex_dtype),
        weighted_image,
        np.asarray(score_weight, dtype=arithmetic_dtype),
        np.asarray(half_weights, dtype=arithmetic_dtype),
        arithmetic_dtype=arithmetic_dtype,
        precision_origin=precision_origin,
    )


def _relion_float32_arms(
    projectors,
    *,
    matrix,
    fimg,
    corr,
    phase_x: float,
    phase_y: float,
    production_scores,
    current_size: int,
    r_max: int,
    padding_factor: int,
    canonical_identities,
) -> tuple[dict[str, object], list[np.ndarray]]:
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.projection import compute_relion_projector_projections_block

    shifted = _relion_shifted_image(fimg, phase_x, phase_y, current_size=current_size)
    score_weight = np.asarray(corr, dtype=np.float32)
    half_weights = np.ones(score_weight.size, dtype=np.float32)
    tree_scores = []
    canonical_f32 = []
    promoted_f64 = []
    promoted_original_f64 = []
    projections = []
    for projector in projectors:
        projection, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(np.asarray(matrix, dtype=np.float32).T[None]),
            (current_size, current_size),
            r_max=int(r_max),
            padding_factor=int(padding_factor),
            centered_rows=False,
            dense_scale=False,
            projector_output_size=int(current_size),
            relion_texture_interp=True,
        )
        projection_np = np.asarray(projection[0], dtype=np.complex64)
        projections.append(projection_np)
        captured = _relion_cc_contributions(projection_np, shifted, score_weight, half_weights)
        tree = _reduction_dict(captured, mode="relion_coarse_128lane_float32")
        canonical = _reduction_dict(captured, mode="canonical_float32", identities=canonical_identities)
        promoted = _relion_cc_contributions(
            projection_np.astype(np.complex128),
            shifted.astype(np.complex128),
            score_weight.astype(np.float64),
            half_weights.astype(np.float64),
            arithmetic_dtype=np.float64,
            precision_origin="promoted_captured",
        )
        tree_scores.append(tree["score"])
        canonical_f32.append(canonical)
        promoted_f64.append(_reduction_dict(promoted, mode="canonical_float64", identities=canonical_identities))
        promoted_original_f64.append(_reduction_dict(promoted, mode="original_order_float64"))

    tree_scores = np.asarray(tree_scores, dtype=np.float32)
    return {
        "production": _winner_summary(np.asarray(production_scores, dtype=np.float32)),
        "coarse_128lane_float32": _winner_summary(tree_scores),
        "control_metrics": score_control_metrics(production_scores, tree_scores),
        "canonical_float32": _winner_summary([entry["score"] for entry in canonical_f32]),
        "canonical_float32_reductions": canonical_f32,
        "promoted_captured_float64": _winner_summary([entry["score"] for entry in promoted_f64]),
        "promoted_captured_float64_reductions": promoted_f64,
        "promoted_captured_float64_original_order": _winner_summary(
            [entry["score"] for entry in promoted_original_f64]
        ),
        "promoted_captured_float64_original_order_reductions": promoted_original_f64,
        "translation_arithmetic": "float32 sin/cos and explicit real/imag products in FFTW current-image order",
        "score_formula": (
            "numerator=sum(dot(projector,translated_Fimg)*corr_img); "
            "norm=sum(abs2(projector)*corr_img)"
        ),
    }, projections


def _soft_mask_average_background(image: np.ndarray, *, radius: float, edge: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    n = int(image.shape[0])
    yy, xx = np.mgrid[:n, :n]
    radius_grid = np.sqrt((yy - n // 2) ** 2 + (xx - n // 2) ** 2, dtype=np.float64)
    raised = np.zeros((n, n), dtype=np.float64)
    outer = float(radius + edge)
    raised[radius_grid > outer] = 1.0
    transition = (radius_grid >= radius) & (radius_grid <= outer)
    raised[transition] = 0.5 + 0.5 * np.cos(np.pi * (outer - radius_grid[transition]) / edge)
    background = np.sum(raised * image, dtype=np.float64) / np.sum(raised, dtype=np.float64)
    return (1.0 - raised) * image + raised * background


def _project_half_float64(projector, matrix, *, current_size: int, r_max: int, padding_factor: int) -> np.ndarray:
    projector = np.asarray(projector, dtype=np.complex128)
    matrix = np.asarray(matrix, dtype=np.float64) * float(padding_factor)
    zdim, ydim, xdim = projector.shape
    zinit = -(zdim // 2)
    yinit = -(ydim // 2)
    result = np.zeros((current_size, current_size // 2 + 1), dtype=np.complex128)
    for row in range(current_size):
        y = row if row <= current_size // 2 else row - current_size
        for x in range(current_size // 2 + 1):
            if x * x + y * y > (current_size // 2) ** 2:
                continue
            xp = matrix[0, 0] * x + matrix[0, 1] * y
            yp = matrix[1, 0] * x + matrix[1, 1] * y
            zp = matrix[2, 0] * x + matrix[2, 1] * y
            negative_x = xp < 0
            if negative_x:
                xp, yp, zp = -xp, -yp, -zp
            if xp * xp + yp * yp + zp * zp > float(r_max * r_max):
                continue
            x0, y0, z0 = int(np.floor(xp)), int(np.floor(yp)), int(np.floor(zp))
            fx, fy, fz = xp - x0, yp - y0, zp - z0
            yr, zr = y0 - yinit, z0 - zinit
            if x0 < 0 or x0 + 1 >= xdim or yr < 0 or yr + 1 >= ydim or zr < 0 or zr + 1 >= zdim:
                continue
            d000 = projector[zr, yr, x0]
            d001 = projector[zr, yr, x0 + 1]
            d010 = projector[zr, yr + 1, x0]
            d011 = projector[zr, yr + 1, x0 + 1]
            d100 = projector[zr + 1, yr, x0]
            d101 = projector[zr + 1, yr, x0 + 1]
            d110 = projector[zr + 1, yr + 1, x0]
            d111 = projector[zr + 1, yr + 1, x0 + 1]
            dx00 = d000 + fx * (d001 - d000)
            dx01 = d100 + fx * (d101 - d100)
            dx10 = d010 + fx * (d011 - d010)
            dx11 = d110 + fx * (d111 - d110)
            dxy0 = dx00 + fy * (dx10 - dx00)
            dxy1 = dx01 + fy * (dx11 - dx01)
            value = dxy0 + fz * (dxy1 - dxy0)
            result[row, x] = np.conj(value) if negative_x else value
    return result.reshape(-1)


def _genuine_upstream_float64_arm(
    *,
    payload: Path,
    relion_prefix: str,
    fixture_root: Path,
    particle_index: int,
    matrix,
    phase_x: float,
    phase_y: float,
    n_classes: int,
    image_size: int,
    current_size: int,
    r_max: int,
    padding_factor: int,
    canonical_identities,
) -> tuple[dict[str, object], list[Path]]:
    from recovar.relion_bind import _relion_bind_core as bind

    raw_path = payload / f"{relion_prefix}pre_raw.bin"
    raw = _read_vector(raw_path, np.float64).reshape(image_size, image_size)
    masked = _soft_mask_average_background(raw, radius=380.0 / (2.0 * 4.25), edge=5.0)
    fourier = np.fft.rfft2(masked) / float(image_size * image_size)
    fourier = np.asarray(bind.window_fourier_transform_2d(fourier, current_size), dtype=np.complex128)
    center_sign = (-1.0) ** np.indices(fourier.shape).sum(axis=0)
    image = fourier * center_sign

    ctf_path = fixture_root / "ctf.pkl"
    ctf_params = np.asarray(pickle.load(ctf_path.open("rb")))[particle_index]
    ctf_full = bind.get_ctf_image(
        float(ctf_params[2]),
        float(ctf_params[3]),
        float(ctf_params[4]),
        float(ctf_params[5]),
        float(ctf_params[6]),
        float(ctf_params[7]),
        0.0,
        float(ctf_params[1]),
        image_size,
        image_size,
        False,
        False,
        False,
        float(ctf_params[8]),
        1.0,
    )
    ctf = np.asarray(
        bind.window_fourier_transform_2d(np.asarray(ctf_full, dtype=np.complex128), current_size).real,
        dtype=np.float64,
    )
    rows = np.arange(current_size, dtype=np.float64)
    y = np.where(rows <= current_size // 2, rows, rows - current_size)
    x = np.arange(current_size // 2 + 1, dtype=np.float64)
    phase = np.exp(1j * (y[:, None] * float(phase_y) + x[None, :] * float(phase_x)))
    image_power = float(np.sum(np.abs(image) ** 2, dtype=np.float64))
    shifted = (image * ctf * phase / image_power).reshape(-1)
    score_weight = (ctf * ctf / image_power).reshape(-1)
    half_weights = np.ones(score_weight.size, dtype=np.float64)

    scores = []
    reductions = []
    source_paths = [raw_path, ctf_path]
    rank_match = re.search(r"rank(\d+)_", relion_prefix)
    if rank_match is None:
        raise ValueError(f"cannot infer RELION rank from prefix {relion_prefix!r}")
    rank = int(rank_match.group(1))
    for class_index in range(n_classes):
        state = f"state_iter1_rank{rank}_device0_class{class_index}_"
        iref_path = payload / f"{state}iref.bin"
        reference = _read_vector(iref_path, np.float64).reshape(image_size, image_size, image_size)
        projector = np.asarray(
            bind.compute_fourier_transform_map(
                reference,
                image_size,
                padding_factor,
                1,
                2 * r_max,
                True,
                2,
            )[0],
            dtype=np.complex128,
        )
        projection = _project_half_float64(
            projector,
            matrix,
            current_size=current_size,
            r_max=r_max,
            padding_factor=padding_factor,
        )
        contributions = normalized_cc_pixel_contributions(
            projection,
            shifted,
            score_weight,
            half_weights,
            arithmetic_dtype=np.float64,
            precision_origin="recomputed_high_precision",
        )
        original_order = _reduction_dict(contributions, mode="original_order_float64")
        canonical = _reduction_dict(contributions, mode="canonical_float64", identities=canonical_identities)
        reductions.append(
            {
                "real_space_reference_sha256": sha256_file(iref_path),
                "original_order": original_order,
                "canonical": canonical,
            }
        )
        scores.append(canonical["score"])
        source_paths.append(iref_path)
    return {
        **_winner_summary(scores),
        "reductions": reductions,
        "source_precision_caveat": (
            "operands are recomputed in float64/complex128 from frozen real-space image/reference captures; "
            "precision lost before those real-space captures cannot be recovered"
        ),
    }, source_paths


def _gpu_provenance() -> dict[str, object]:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
        ).splitlines()
    except Exception:
        rows = []
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_visible_rows": rows,
        "RECOVAR_CUDA_LIB": os.environ.get("RECOVAR_CUDA_LIB"),
    }


def run_replay(
    *,
    capture_root: Path,
    fixture_root: Path,
    particle_index: int,
    recovar_max_ulp: int = DEFAULT_RECOVAR_MAX_ULP,
    relion_max_abs: float = DEFAULT_RELION_MAX_ABS,
    relion_centered_max_abs: float = DEFAULT_RELION_CENTERED_MAX_ABS,
) -> dict[str, object]:
    from recovar.utils.parity_provenance import assert_parity_ancestors, git_worktree_provenance

    assert_parity_ancestors()
    capture_root = capture_root.resolve()
    fixture_root = fixture_root.resolve()
    payload = capture_root / "relion_capture" / "payload"
    recovar_path = capture_root / "recovar_capture" / "payload" / (
        f"significance_orig{particle_index:06d}_it001_cs014.npz"
    )
    relion_prefix = _discover_relion_prefix(payload)
    rank_match = re.search(r"rank(\d+)_", relion_prefix)
    if rank_match is None:
        raise ValueError(f"cannot infer RELION rank from prefix {relion_prefix!r}")
    rank = int(rank_match.group(1))

    with np.load(recovar_path, allow_pickle=False) as recovar:
        n_classes = int(recovar["n_classes"])
        n_rot = int(recovar["n_rot"])
        n_trans = int(recovar["n_trans"])
        current_size = int(recovar["current_size"])
        rotations = np.asarray(recovar["rotations"], dtype=np.float32)
        shifted_data = np.asarray(recovar["shifted_data"], dtype=np.complex64)
        score_weight = np.asarray(recovar["ctf2_data"], dtype=np.float32)
        window_indices = np.asarray(recovar["window_indices"], dtype=np.int32)
        half_weights = np.asarray(recovar["half_weights"], dtype=np.float32)
        score_surface = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)
    if n_classes != 4 or rotations.shape != (n_rot, 3, 3) or score_surface.shape != (4, n_rot, n_trans):
        raise ValueError("unexpected RECOVAR K=4 capture topology")
    if not np.array_equal(half_weights, np.ones_like(half_weights)):
        raise ValueError("firstiter normalized-CC replay requires captured unit half weights")
    best_poses = [np.unravel_index(np.argmax(score_surface[c]), score_surface[c].shape) for c in range(n_classes)]
    if len(set(best_poses)) != 1:
        raise ValueError(f"per-class RECOVAR winner geometry is not common: {best_poses}")
    recovar_rotation_index, translation_index = (int(value) for value in best_poses[0])
    recovar_production = score_surface[np.arange(n_classes), recovar_rotation_index, translation_index]

    first_projector, r_max, padding_factor = _load_relion_projector(payload, rank=rank, class_index=0)
    relion_projectors = [first_projector]
    for class_index in range(1, n_classes):
        projector, class_r_max, class_padding = _load_relion_projector(
            payload, rank=rank, class_index=class_index
        )
        if (class_r_max, class_padding) != (r_max, padding_factor):
            raise ValueError("RELION projector metadata differs across classes")
        relion_projectors.append(projector)
    relion_projectors = np.asarray(relion_projectors, dtype=np.complex64)
    # The frozen reference volume dimension is the authoritative image box.
    # In particular, pre_normshift is an image normalization scalar and must
    # never be interpreted as a dimension.
    iref_dim_path = payload / f"state_iter1_rank{rank}_device0_class0_iref_xdim.bin"
    image_size = int(round(_read_double_scalar(iref_dim_path)))
    recovar_identities, relion_identities = _canonical_identities_from_recovar(
        window_indices, image_size=image_size
    )

    recovar_projectors, regenerated_r_max, reference_paths = _regenerate_recovar_projectors(
        fixture_root,
        n_classes=n_classes,
        current_size=2 * r_max,
        padding_factor=padding_factor,
    )
    if regenerated_r_max != r_max:
        raise ValueError(f"regenerated RECOVAR r_max {regenerated_r_max} != captured RELION r_max {r_max}")
    recovar_arms, recovar_projections = _recovar_float32_arms(
        recovar_projectors,
        rotation=rotations[recovar_rotation_index],
        shifted_data=shifted_data,
        score_weight=score_weight,
        half_weights=half_weights,
        window_indices=window_indices,
        production_scores=recovar_production,
        image_size=image_size,
        current_size=current_size,
        r_max=r_max,
        padding_factor=padding_factor,
        translation_index=translation_index,
        canonical_identities=recovar_identities,
    )

    nr_dir = int(round(_read_double_scalar(payload / f"{relion_prefix}nr_dir.bin")))
    nr_psi = int(round(_read_double_scalar(payload / f"{relion_prefix}nr_psi.bin")))
    relion_n_trans = int(round(_read_double_scalar(payload / f"{relion_prefix}nr_trans.bin")))
    raw_scores = _read_vector(payload / f"{relion_prefix}raw_scores.bin", np.float64).reshape(
        n_classes, nr_dir, nr_psi, relion_n_trans
    )
    relion_best = [np.unravel_index(np.argmin(raw_scores[c]), raw_scores[c].shape) for c in range(n_classes)]
    if len(set(relion_best)) != 1:
        raise ValueError(f"per-class RELION winner geometry is not common: {relion_best}")
    direction_index, psi_index, relion_translation_index = (int(value) for value in relion_best[0])
    candidate_flat = (direction_index * nr_psi + psi_index) * relion_n_trans + relion_translation_index
    rot_ids = _read_vector(payload / f"{relion_prefix}rot_idx.bin", np.int64).reshape(
        n_classes, nr_dir * nr_psi, relion_n_trans
    )
    relion_rotation_index = int(rot_ids[0].reshape(-1)[candidate_flat])
    plan_eulers = _read_vector(payload / f"{relion_prefix}plan_eulers.bin", np.float64).reshape(n_rot, 3, 3)
    relion_matrix = plan_eulers[relion_rotation_index]
    if not np.array_equal(rotations[recovar_rotation_index], relion_matrix.T.astype(np.float32)):
        raise ValueError("RECOVAR/RELION winning rotation bijection is not bitwise exact")
    relion_production = -raw_scores[np.arange(n_classes), direction_index, psi_index, relion_translation_index]
    fimg = _read_vector(payload / f"{relion_prefix}Fimg_real.bin", np.float64).astype(np.float32)
    fimg = fimg + 1j * _read_vector(payload / f"{relion_prefix}Fimg_imag.bin", np.float64).astype(np.float32)
    corr = _read_vector(payload / f"{relion_prefix}corr_img.bin", np.float64).astype(np.float32)
    phase_x = _read_vector(payload / f"{relion_prefix}trans_phase_x.bin", np.float64)[relion_translation_index]
    phase_y = _read_vector(payload / f"{relion_prefix}trans_phase_y.bin", np.float64)[relion_translation_index]
    relion_arms, relion_projections = _relion_float32_arms(
        relion_projectors,
        matrix=relion_matrix,
        fimg=fimg,
        corr=corr,
        phase_x=float(phase_x),
        phase_y=float(phase_y),
        production_scores=relion_production,
        current_size=current_size,
        r_max=r_max,
        padding_factor=padding_factor,
        canonical_identities=relion_identities,
    )

    close, control_policy = controls_close(
        recovar_arms["control_metrics"],
        relion_arms["control_metrics"],
        recovar_max_ulp=recovar_max_ulp,
        relion_max_abs=relion_max_abs,
        relion_centered_max_abs=relion_centered_max_abs,
    )
    base_report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "float32_controls_closed" if close else "control_failed",
        "classification": "pending_high_precision" if close else "unresolved_float32_control_mismatch",
        "particle_original_index_zero_based": int(particle_index),
        "fixed_geometry": {
            "recovar_rotation_index": recovar_rotation_index,
            "relion_rotation_index": relion_rotation_index,
            "translation_index": translation_index,
            "rotation_bijection_bitwise_exact": True,
            "current_size": current_size,
            "projector_r_max": r_max,
            "projector_padding_factor": padding_factor,
        },
        "float32_controls": {
            "policy": control_policy,
            "recovar": recovar_arms,
            "relion": relion_arms,
        },
        "projector_source_metrics": [
            _array_metrics(relion_projectors[class_index], recovar_projectors[class_index])
            for class_index in range(n_classes)
        ],
        "projection_metrics_at_winner": [
            _array_metrics(
                _canonical_reorder(relion_projections[class_index], relion_identities),
                _canonical_reorder(
                    -recovar_projections[class_index] / image_size**2,
                    recovar_identities,
                ),
            )
            for class_index in range(n_classes)
        ],
        "quality_metric_policy": "exact/array metrics only; map conclusions require shellwise FSC/FSC-AUC; correlation prohibited",
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "git": git_worktree_provenance(),
            "gpu": _gpu_provenance(),
            "capture_root": str(capture_root),
            "fixture_root": str(fixture_root),
        },
    }
    if not close:
        return base_report

    genuine_float64, high_precision_paths = _genuine_upstream_float64_arm(
        payload=payload,
        relion_prefix=relion_prefix,
        fixture_root=fixture_root,
        particle_index=particle_index,
        matrix=relion_matrix,
        phase_x=float(phase_x),
        phase_y=float(phase_y),
        n_classes=n_classes,
        image_size=image_size,
        current_size=current_size,
        r_max=r_max,
        padding_factor=padding_factor,
        canonical_identities=relion_identities,
    )
    production_winners = {
        "recovar": int(recovar_arms["production"]["winner_class_zero_based"]),
        "relion": int(relion_arms["production"]["winner_class_zero_based"]),
    }
    promoted_winners = {
        "recovar": int(recovar_arms["promoted_captured_float64"]["winner_class_zero_based"]),
        "relion": int(relion_arms["promoted_captured_float64"]["winner_class_zero_based"]),
    }
    genuine_winner = int(genuine_float64["winner_class_zero_based"])
    if len(set(production_winners.values())) > 1 and len(set(promoted_winners.values())) == 1 and genuine_winner in set(
        promoted_winners.values()
    ):
        classification = "float32_order_or_operand_generation_near_tie_resolved_by_common_float64"
    elif len(set(production_winners.values())) == 1:
        classification = "production_float32_decision_agrees"
    else:
        classification = "high_precision_replay_unresolved"
    base_report.update(
        {
            "status": "pass",
            "classification": classification,
            "high_precision_arms": {
                "recovar_promoted_captured_float64": recovar_arms["promoted_captured_float64"],
                "relion_promoted_captured_float64": relion_arms["promoted_captured_float64"],
                "recovar_promoted_captured_float64_original_order": recovar_arms[
                    "promoted_captured_float64_original_order"
                ],
                "relion_promoted_captured_float64_original_order": relion_arms[
                    "promoted_captured_float64_original_order"
                ],
                "common_upstream_recomputed_float64": genuine_float64,
            },
            "precision_interpretation": {
                "production_winners": production_winners,
                "promoted_captured_winners": promoted_winners,
                "common_upstream_float64_winner": genuine_winner,
                "promoted_is_not_genuine": True,
            },
        }
    )
    input_paths = [
        recovar_path,
        payload / f"{relion_prefix}raw_scores.bin",
        payload / f"{relion_prefix}plan_eulers.bin",
        payload / f"{relion_prefix}Fimg_real.bin",
        payload / f"{relion_prefix}Fimg_imag.bin",
        payload / f"{relion_prefix}corr_img.bin",
        *reference_paths,
        *high_precision_paths,
    ]
    for class_index in range(n_classes):
        state = f"state_iter1_rank{rank}_device0_class{class_index}_"
        input_paths.extend(
            [payload / f"{state}projector_real.bin", payload / f"{state}projector_imag.bin"]
        )
    cuda_lib = os.environ.get("RECOVAR_CUDA_LIB")
    if cuda_lib and Path(cuda_lib).is_file():
        input_paths.append(Path(cuda_lib))
    base_report["input_artifact_sha256"] = {
        str(path.resolve()): sha256_file(path) for path in dict.fromkeys(Path(path) for path in input_paths)
    }
    return base_report


def write_report_and_seal(report: dict[str, object], output: Path, seal_output: Path) -> None:
    output = output.resolve()
    seal_output = seal_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    seal = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "status": report["status"],
        "classification": report["classification"],
        "report_path": str(output),
        "report_sha256": sha256_file(output),
        "input_artifact_sha256": report.get("input_artifact_sha256", {}),
    }
    seal_output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--particle-index", type=int, default=7915)
    parser.add_argument("--recovar-max-ulp", type=int, default=DEFAULT_RECOVAR_MAX_ULP)
    parser.add_argument("--relion-max-abs", type=float, default=DEFAULT_RELION_MAX_ABS)
    parser.add_argument("--relion-centered-max-abs", type=float, default=DEFAULT_RELION_CENTERED_MAX_ABS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    report = run_replay(
        capture_root=args.capture_root,
        fixture_root=args.fixture_root,
        particle_index=args.particle_index,
        recovar_max_ulp=args.recovar_max_ulp,
        relion_max_abs=args.relion_max_abs,
        relion_centered_max_abs=args.relion_centered_max_abs,
    )
    write_report_and_seal(report, args.output, args.seal_output)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "classification": report["classification"]}))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
