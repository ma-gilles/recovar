#!/usr/bin/env python3
"""Compare one native RELION VDAM StoreWavg call with RECOVAR pre-scatter rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "recovar.vdam_storewavg_boundary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated flat array: {path}")
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {path}")
    return float(struct.unpack("<d", payload)[0])


def _real_2d(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated real 2-D array: {path}")
    rows, columns = struct.unpack_from("<ii", payload)
    values = np.frombuffer(payload, dtype="<f8", offset=8).copy()
    _require(values.size == rows * columns, f"real 2-D size mismatch: {path}")
    return values.reshape(rows, columns)


def _complex_2d(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated complex 2-D array: {path}")
    rows, columns = struct.unpack_from("<ii", payload)
    values = np.frombuffer(payload, dtype="<c16", offset=8).copy()
    _require(values.size == rows * columns, f"complex 2-D size mismatch: {path}")
    return values.reshape(rows, columns)


def _complex_long_3d(path: Path) -> np.ndarray:
    """Read a RELION MultidimArray dump with three native ``long`` dimensions."""

    payload = path.read_bytes()
    header_bytes = 3 * np.dtype(np.int_).itemsize
    _require(len(payload) >= header_bytes, f"truncated complex 3-D array: {path}")
    dimensions = np.frombuffer(payload, dtype=np.int_, count=3).astype(np.int64)
    _require(np.all(dimensions > 0), f"invalid complex 3-D dimensions: {path}")
    values = np.frombuffer(payload, dtype=np.complex128, offset=header_bytes).copy()
    _require(values.size == int(np.prod(dimensions)), f"complex 3-D size mismatch: {path}")
    return values.reshape(tuple(int(value) for value in dimensions))


def _load_unmasked_image(path: Path) -> np.ndarray:
    """Load only a dump whose name explicitly identifies the unmasked operand."""

    _require(path.is_file(), f"unmasked StoreWavg image is unavailable: {path}")
    _require("nomask" in path.name, f"refusing masked StoreWavg image: {path}")
    if path.name.endswith("Fimg_unweighted_nomask.bin"):
        return _complex_2d(path)
    if path.name.endswith(("Fimg_nomask.bin", "Fimg_shifted_t0_nomask.bin")):
        return _complex_long_3d(path)
    raise ValueError(f"unrecognized unmasked StoreWavg image dump: {path}")


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _require(reference.shape == candidate.shape and reference.size > 0, "metric topology mismatch")
    ref = reference.astype(np.complex128, copy=False).reshape(-1)
    cand = candidate.astype(np.complex128, copy=False).reshape(-1)
    residual = cand - ref
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    denominator = max(ref_norm, np.finfo(np.float64).tiny)
    cosine_denominator = max(ref_norm * cand_norm, np.finfo(np.float64).tiny)
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "cosine": float(np.real(np.vdot(ref, cand)) / cosine_denominator),
        "max_abs": float(np.max(np.abs(residual), initial=0.0)),
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
    }


def _posterior_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    result = _metric(reference, candidate)
    result.update(
        {
            "reference_retained_mass": float(np.sum(reference, dtype=np.float64)),
            "candidate_retained_mass": float(np.sum(candidate, dtype=np.float64)),
            "l1": float(np.sum(np.abs(candidate - reference), dtype=np.float64)),
            "support_mismatch_count": int(
                np.count_nonzero((reference > 0.0) != (candidate > 0.0))
            ),
            "reference_positive_count": int(np.count_nonzero(reference > 0.0)),
            "candidate_positive_count": int(np.count_nonzero(candidate > 0.0)),
        }
    )
    return result


def _inline_projector_comparisons(
    recovar: dict[str, np.ndarray],
    particle_slot: int,
    native_bpref_data: np.ndarray,
    native_bpref_weight: np.ndarray,
    controlled_bpref_data: np.ndarray,
    controlled_bpref_weight: np.ndarray,
    native_gpu_bpref_data: np.ndarray | None = None,
    native_gpu_bpref_weight: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Compare an optional production-projector particle capture by identity."""

    if "inline_projector_data_volumes" not in recovar:
        return {}
    inline_original_indices = np.asarray(
        recovar["inline_projector_original_indices"],
        dtype=np.int64,
    )
    target_original_index = int(np.asarray(recovar["original_indices"])[particle_slot])
    inline_matches = np.flatnonzero(inline_original_indices == target_original_index)
    if not inline_matches.size:
        return {}
    _require(
        inline_matches.size == 1,
        "inline-projector capture does not uniquely identify the selected particle",
    )
    inline_slot = int(inline_matches[0])
    inline_data = np.asarray(
        recovar["inline_projector_data_volumes"][inline_slot],
        dtype=np.complex64,
    ).reshape(native_bpref_data.shape)
    inline_weight = np.asarray(
        recovar["inline_projector_weight_volumes"][inline_slot],
        dtype=np.float32,
    ).reshape(native_bpref_weight.shape)
    result = {
        "inline_projector_bpref_data": _metric(native_bpref_data, inline_data),
        "inline_projector_bpref_weight": _metric(native_bpref_weight, inline_weight),
        "inline_projector_bpref_data_same_posterior_control": _metric(
            controlled_bpref_data,
            inline_data,
        ),
        "inline_projector_bpref_weight_same_posterior_control": _metric(
            controlled_bpref_weight,
            inline_weight,
        ),
    }
    if native_gpu_bpref_data is not None or native_gpu_bpref_weight is not None:
        _require(
            native_gpu_bpref_data is not None and native_gpu_bpref_weight is not None,
            "native GPU particle BPref data and weight must be supplied together",
        )
        result.update(
            {
                "inline_projector_vs_native_gpu_bpref_data": _metric(
                    np.asarray(native_gpu_bpref_data).reshape(inline_data.shape),
                    inline_data,
                ),
                "inline_projector_vs_native_gpu_bpref_weight": _metric(
                    np.asarray(native_gpu_bpref_weight).reshape(inline_weight.shape),
                    inline_weight,
                ),
            }
        )
    return result


def _match_rotations(native: np.ndarray, recovar: np.ndarray, tolerance: float) -> np.ndarray:
    native = np.asarray(native, dtype=np.float32).reshape(-1, 3, 3)
    recovar = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    distances = np.max(np.abs(native[:, None] - recovar[None, :]), axis=(2, 3))
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(native.shape[0]), mapping]
    _require(float(np.max(nearest, initial=0.0)) <= tolerance, "native rotation is absent from RECOVAR")
    _require(np.unique(mapping).size == mapping.size, "native-to-RECOVAR rotation mapping is not one-to-one")
    return mapping


def _select_recovar_particle_rows(
    recovar: dict[str, np.ndarray], original_index: int | None
) -> tuple[int, np.ndarray]:
    original_indices = np.asarray(recovar["original_indices"], dtype=np.int64)
    _require(original_indices.size > 0, "RECOVAR capture contains no particles")
    if original_index is None:
        _require(original_indices.size == 1, "--recovar-original-index is required for a panel capture")
        particle_slot = 0
    else:
        matches = np.flatnonzero(original_indices == int(original_index))
        _require(matches.size == 1, f"RECOVAR capture does not uniquely contain original index {original_index}")
        particle_slot = int(matches[0])
    active_particle_rows = np.asarray(recovar["active_particle_rows"], dtype=np.int64)
    row_mask = active_particle_rows == particle_slot
    _require(np.any(row_mask), f"RECOVAR capture has no active rows for original index {original_indices[particle_slot]}")
    return particle_slot, row_mask


def _fftw_window_to_native_crop(
    window_indices: np.ndarray,
    *,
    physical_image_size: int,
    current_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map full-box FFTW-half rows to native current-size and centered indices."""

    indices = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    full_half_width = physical_image_size // 2 + 1
    rows = indices // full_half_width
    x = indices - rows * full_half_width
    ky = np.where(rows <= physical_image_size // 2, rows, rows - physical_image_size)
    _require(np.all(x <= current_size // 2), "window x coordinate exceeds native crop")
    _require(np.all(np.abs(ky) <= current_size // 2), "window y coordinate exceeds native crop")
    crop_rows = np.where(ky >= 0, ky, ky + current_size)
    crop_indices = crop_rows * (current_size // 2 + 1) + x
    centered_indices = (ky + physical_image_size // 2) * full_half_width + x
    return crop_indices.astype(np.int32), centered_indices.astype(np.int32)


def _native_gradient_rows(
    probabilities: np.ndarray,
    translated_images: np.ndarray,
    projections: np.ndarray,
    ctf: np.ndarray,
    inverse_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay RELION ``cuda_kernel_backproject3D_SGD`` before scatter."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    translated_images = np.asarray(translated_images, dtype=np.complex64)
    projections = np.asarray(projections, dtype=np.complex64)
    ctf = np.asarray(ctf, dtype=np.float32).reshape(-1)
    inverse_noise = np.asarray(inverse_noise, dtype=np.float32).reshape(-1)
    _require(probabilities.ndim == 2, "probabilities must have shape (rotation, translation)")
    _require(
        translated_images.shape == (probabilities.shape[1], ctf.size),
        "translated-image topology changed",
    )
    _require(projections.shape == (probabilities.shape[0], ctf.size), "projection topology changed")
    rotation_mass = np.sum(probabilities, axis=1, dtype=np.float32)
    ctf_inverse_noise = (ctf * inverse_noise).astype(np.float32)
    ctf2_inverse_noise = (ctf * ctf * inverse_noise).astype(np.float32)
    weighted_images = (probabilities @ translated_images).astype(np.complex64)
    data = (
        weighted_images * ctf_inverse_noise[None, :]
        - projections * (rotation_mass[:, None] * ctf2_inverse_noise[None, :])
    ).astype(np.complex64)
    weight = (rotation_mass[:, None] * ctf2_inverse_noise[None, :]).astype(np.float32)
    return data, weight


def _restore_storewavg_inverse_noise_dc(
    inverse_noise: np.ndarray,
    crop_indices: np.ndarray,
    sigma2_noise: np.ndarray,
    sigma2_fudge: float,
) -> np.ndarray:
    """Restore the DC lane that RELION sets immediately inside StoreWavg."""

    inverse_noise = np.asarray(inverse_noise, dtype=np.float32).copy()
    crop_indices = np.asarray(crop_indices, dtype=np.int32).reshape(-1)
    dc_rows = np.flatnonzero(crop_indices == 0)
    _require(dc_rows.size == 1, "StoreWavg crop must contain exactly one DC lane")
    sigma2 = np.asarray(sigma2_noise, dtype=np.float64).reshape(-1)
    _require(sigma2.size > 0 and sigma2[0] > 0.0, "native sigma2_noise DC must be positive")
    _require(float(sigma2_fudge) > 0.0, "native sigma2_fudge must be positive")
    inverse_noise[dc_rows[0]] = np.float32(1.0 / (float(sigma2_fudge) * float(sigma2[0])))
    return inverse_noise


def _production_score_gradient_rows(
    score_dump: dict[str, np.ndarray],
    reconstruction_probs_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the fused production M-step rows from a local score dump.

    The score dump operands are already in RECOVAR's reconstruction frame:
    ``debug_shifted_recon`` contains translated ``Fimg * CTF / sigma2`` and
    ``debug_ctf2_over_nv_recon`` contains ``CTF**2 / sigma2``.  Keep the
    contraction on JAX so its float32 GEMM reduction matches production.
    """

    import jax
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.local_backprojection import (
        compute_local_weighted_sums,
    )

    required = {
        "posterior",
        "reconstruction_sample_mask",
        "debug_shifted_recon",
        "debug_ctf2_over_nv_recon",
        "debug_proj_for_recon",
    }
    missing = sorted(required.difference(score_dump))
    _require(not missing, f"production score dump is missing operands: {missing}")
    posterior = np.asarray(score_dump["posterior"], dtype=np.float32)
    sample_mask = np.asarray(score_dump["reconstruction_sample_mask"], dtype=bool)
    _require(posterior.ndim == 3 and posterior.shape[0] == 1, "score posterior topology changed")
    _require(sample_mask.shape == posterior.shape, "score reconstruction mask topology changed")
    if reconstruction_probs_override is None:
        reconstruction_probs = jnp.asarray(posterior * sample_mask, dtype=jnp.float32)
    else:
        override = np.asarray(reconstruction_probs_override, dtype=np.float32)
        _require(
            override.shape == posterior.shape[1:],
            "score reconstruction-probability override topology changed",
        )
        reconstruction_probs = jnp.asarray(override[None, ...], dtype=jnp.float32)
    shifted = jnp.asarray(
        np.asarray(score_dump["debug_shifted_recon"], dtype=np.complex64)[None, :, :]
    )
    ctf2 = jnp.asarray(
        np.asarray(score_dump["debug_ctf2_over_nv_recon"], dtype=np.float32)[None, :]
    )
    projections = jnp.asarray(
        np.asarray(score_dump["debug_proj_for_recon"], dtype=np.complex64)[None, :, :]
    )
    _require(shifted.shape[:2] == (1, posterior.shape[2]), "score shifted-image topology changed")
    _require(projections.shape[:2] == posterior.shape[:2], "score projection topology changed")
    _require(shifted.shape[-1] == ctf2.shape[-1] == projections.shape[-1], "score pixel topology changed")
    rotation_mass = jnp.sum(reconstruction_probs, axis=-1, dtype=jnp.float32)
    numerator = compute_local_weighted_sums(reconstruction_probs, shifted)
    denominator = jnp.where(
        rotation_mass[..., None] != 0.0,
        rotation_mass[..., None] * ctf2[:, None, :],
        0.0,
    )
    numerator = numerator - jnp.where(
        rotation_mass[..., None] != 0.0,
        rotation_mass[..., None] * projections * ctf2[:, None, :],
        0.0,
    )
    numerator, denominator = jax.block_until_ready((numerator, denominator))
    return (
        np.asarray(numerator[0], dtype=np.complex64),
        np.asarray(denominator[0], dtype=np.float32),
        np.asarray(reconstruction_probs[0], dtype=np.float32),
    )


def _scatter_relion_rows(
    data_rows: np.ndarray,
    weight_rows: np.ndarray,
    rotations: np.ndarray,
    window_indices: np.ndarray,
    *,
    physical_image_size: int,
    current_size: int,
    padding_factor: int,
    get_backprojector_data,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter pre-StoreWavg rows through RELION's CPU BackProjector binding."""

    from recovar.em.bpref_contribution_replay import dense_fftw_half_rows

    image_shape = (physical_image_size, physical_image_size)
    dense_data = dense_fftw_half_rows(
        np.asarray(data_rows),
        np.asarray(window_indices, dtype=np.int32),
        image_shape,
        dtype=np.complex128,
    )
    dense_weight = dense_fftw_half_rows(
        np.asarray(weight_rows),
        np.asarray(window_indices, dtype=np.int32),
        image_shape,
        dtype=np.float64,
    )
    data, weight = get_backprojector_data(
        dense_data,
        np.asarray(rotations, dtype=np.float64),
        dense_weight,
        ori_size=physical_image_size,
        padding_factor=padding_factor,
        interpolator=1,
        current_size=current_size,
    )
    return np.asarray(data), np.asarray(weight)


def _load_native(
    native_directory: Path,
    prefix: str,
    *,
    projector_prefix: str | None = None,
    load_projector: bool = True,
) -> dict[str, np.ndarray | float | int]:
    root = native_directory / prefix
    orientation_count = int(round(_scalar(Path(f"{root}orientation_num.bin"))))
    translation_count = int(round(_scalar(Path(f"{root}translation_num.bin"))))
    raw = _flat(Path(f"{root}sorted_weights.bin"), np.dtype("<f8")).astype(np.float32)
    raw = raw.reshape(orientation_count, translation_count)
    sum_weight = np.float32(_scalar(Path(f"{root}sum_weight.bin")))
    threshold = np.float32(_scalar(Path(f"{root}significant_weight.bin")))
    probabilities = np.where(raw >= threshold, raw / sum_weight, np.float32(0.0)).astype(np.float32)
    rotations = _flat(Path(f"{root}eulers.bin"), np.dtype("<f8")).astype(np.float32)
    rotations = rotations.reshape(orientation_count, 3, 3).transpose(0, 2, 1)
    phases = _flat(Path(f"{root}trans_xyz.bin"), np.dtype("<f8")).astype(np.float32)
    _require(phases.size == 3 * translation_count, "native translation phase topology changed")
    translation_angles = np.stack(
        (phases[:translation_count], phases[translation_count : 2 * translation_count]),
        axis=1,
    )
    ctf = _flat(Path(f"{root}ctfs.bin"), np.dtype("<f8")).astype(np.float32)
    result = {
        "orientation_count": orientation_count,
        "translation_count": translation_count,
        "probabilities": probabilities,
        "rotations": rotations,
        "translation_angles": translation_angles,
        "ctf": ctf,
        "retained_mass": float(np.sum(probabilities, dtype=np.float64)),
    }
    if load_projector:
        projector_root = (
            Path(f"{root}wavg_ppref_")
            if projector_prefix is None
            else native_directory / projector_prefix
        )
        dims = _flat(Path(f"{projector_root}dims.bin"), np.dtype("<i4")).astype(np.int64)
        _require(dims.size == 7, "native Projector dimensions changed")
        xdim, ydim, zdim, _xinit, _yinit, _zinit, r_max = (int(value) for value in dims)
        real = _flat(Path(f"{projector_root}real.bin"), np.dtype("<f8"))
        imag = _flat(Path(f"{projector_root}imag.bin"), np.dtype("<f8"))
        _require(real.size == imag.size == xdim * ydim * zdim, "native Projector payload changed")
        result.update(
            {
                "projector": (real + 1j * imag).astype(np.complex64).reshape(zdim, ydim, xdim),
                "r_max": r_max,
            }
        )
    return result


def _load_native_particle_bpref(
    native_directory: Path,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load an isolated native CUDA backprojector accumulator capture."""

    root = native_directory / prefix
    dims_path = Path(f"{root}bp_dims.bin")
    real_path = Path(f"{root}bp_real.bin")
    imag_path = Path(f"{root}bp_imag.bin")
    weight_path = Path(f"{root}bp_weight.bin")
    dims = _flat(dims_path, np.dtype("<i4")).astype(np.int64)
    _require(dims.size == 6, "native particle BPref dimensions changed")
    voxel_count = int(np.prod(dims[:3], dtype=np.int64))
    real = _flat(real_path, np.dtype("<f8"))
    imag = _flat(imag_path, np.dtype("<f8"))
    weight = _flat(weight_path, np.dtype("<f8"))
    _require(
        real.size == imag.size == weight.size == voxel_count,
        "native particle BPref payload changed",
    )
    artifacts = {
        "native_particle_bpref_prefix": prefix,
        "native_particle_bpref_dims": dims.tolist(),
        "native_particle_bpref_dims_sha256": _sha256(dims_path),
        "native_particle_bpref_real_sha256": _sha256(real_path),
        "native_particle_bpref_imag_sha256": _sha256(imag_path),
        "native_particle_bpref_weight_sha256": _sha256(weight_path),
    }
    return (real + 1j * imag).astype(np.complex64), weight.astype(np.float32), artifacts


def analyze(
    native_directory: Path,
    recovar_capture: Path,
    *,
    native_image_path: Path | None,
    native_inverse_noise_path: Path,
    native_sigma2_noise_path: Path,
    native_sigma2_fudge_path: Path,
    recovar_original_index: int | None,
    native_prefix: str,
    native_projector_prefix: str | None,
    native_particle_bpref_prefix: str | None,
    recovar_score_dump: Path | None,
    physical_image_size: int,
    current_size: int,
    rotation_tolerance: float,
    posterior_only: bool = False,
    adaptive_fraction: float = 0.999,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from recovar.relion_bind._relion_bind_core import get_backprojector_data

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "VDAM StoreWavg replay requires a GPU")
    native = _load_native(
        native_directory,
        native_prefix,
        projector_prefix=native_projector_prefix,
        load_projector=not posterior_only,
    )
    with np.load(recovar_capture, allow_pickle=False) as archive:
        recovar = {name: archive[name] for name in archive.files}
    _require(int(recovar["current_size"]) == current_size, "RECOVAR current size changed")
    particle_slot, recovar_row_mask = _select_recovar_particle_rows(recovar, recovar_original_index)

    score_dump = None
    if recovar_score_dump is not None:
        with np.load(recovar_score_dump, allow_pickle=False) as archive:
            score_dump = {name: archive[name] for name in archive.files}
        score_original_indices = np.asarray(score_dump["selected_global_image_indices"], dtype=np.int64)
        _require(
            score_original_indices.size == 1
            and int(score_original_indices[0]) == int(np.asarray(recovar["original_indices"])[particle_slot]),
            "production score dump particle identity differs from contribution capture",
        )

    rotations = np.asarray(native["rotations"], dtype=np.float32)
    recovar_rotations = (
        np.asarray(score_dump["local_rotation_matrices"], dtype=np.float32)
        if score_dump is not None
        else np.asarray(recovar["active_rotations"])[recovar_row_mask]
    )
    rotation_map = _match_rotations(rotations, recovar_rotations, rotation_tolerance)

    active_rotation_rows = np.asarray(recovar["active_rotation_rows"], dtype=np.int64)[
        recovar_row_mask
    ]
    if score_dump is not None:
        production_data, production_weight, production_posterior = _production_score_gradient_rows(
            score_dump
        )
        current_posterior = production_posterior[rotation_map]
    else:
        production_data = None
        production_weight = None
        current_posterior = np.asarray(recovar["reconstruction_probs"])[particle_slot][
            active_rotation_rows
        ][rotation_map]
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_f32_fine_reconstruction_probs,
    )

    replay_posterior, *_ = _relion_f32_fine_reconstruction_probs(
        jnp.asarray(np.asarray(recovar["candidate_combined_scores"])[particle_slot : particle_slot + 1]),
        adaptive_fraction=float(adaptive_fraction),
    )
    replay_posterior = np.asarray(jax.block_until_ready(replay_posterior))[0][
        active_rotation_rows
    ][rotation_map]
    posterior_comparisons = {
        "fine_posterior_current": _posterior_metric(native["probabilities"], current_posterior),
        "fine_posterior_relion_f32_replay": _posterior_metric(
            native["probabilities"], replay_posterior
        ),
    }
    if posterior_only:
        return {
            "schema": SCHEMA,
            "identity": {
                "original_index": int(np.asarray(recovar["original_indices"])[particle_slot]),
                "image_identity": str(np.asarray(recovar["image_identities"])[particle_slot]),
                "iteration": int(recovar["iteration"]),
                "half": int(recovar["half"]),
                "physical_image_size": physical_image_size,
                "current_size": current_size,
                "orientation_count": int(native["orientation_count"]),
                "translation_count": int(native["translation_count"]),
            },
            "comparisons": posterior_comparisons,
            "artifacts": {
                "native_directory": str(native_directory.resolve()),
                "recovar_capture": str(recovar_capture.resolve()),
                "recovar_capture_sha256": _sha256(recovar_capture),
            },
            "device": str(jax.devices()[0]),
        }

    _require(int(native["r_max"]) == current_size // 2, "native current size changed")
    _require(native_image_path is not None, "native unmasked image is required for gradient replay")
    crop_indices, centered_indices = _fftw_window_to_native_crop(
        recovar["window_indices"],
        physical_image_size=physical_image_size,
        current_size=current_size,
    )
    native_ctf = np.asarray(native["ctf"], dtype=np.float32)[crop_indices]
    inverse_noise = _real_2d(native_inverse_noise_path).astype(np.float32).reshape(-1)[crop_indices]
    inverse_noise = _restore_storewavg_inverse_noise_dc(
        inverse_noise,
        crop_indices,
        _real_2d(native_sigma2_noise_path),
        _scalar(native_sigma2_fudge_path),
    )
    native_image = _load_unmasked_image(native_image_path).astype(np.complex64).reshape(-1)[crop_indices]

    translated = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(native_image[None, :]),
        jnp.asarray(native["translation_angles"], dtype=jnp.float32),
        jnp.asarray(centered_indices, dtype=jnp.int32),
        (physical_image_size, physical_image_size),
    )
    translated = np.asarray(jax.block_until_ready(translated), dtype=np.complex64)
    projections, _ = compute_relion_projector_projections_block(
        jnp.asarray(native["projector"]),
        jnp.asarray(rotations),
        (physical_image_size, physical_image_size),
        r_max=current_size // 2,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=False,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(centered_indices, dtype=jnp.int32),
        relion_texture_interp=True,
    )
    projections = np.asarray(jax.block_until_ready(projections), dtype=np.complex64)
    native_data, native_weight = _native_gradient_rows(
        native["probabilities"],
        translated,
        projections,
        native_ctf,
        inverse_noise,
    )
    controlled_data, controlled_weight = _native_gradient_rows(
        current_posterior,
        translated,
        projections,
        native_ctf,
        inverse_noise,
    )
    native_ctf_inverse_noise = (native_ctf * inverse_noise).astype(np.float32)
    controlled_image_term = (
        (current_posterior @ translated).astype(np.complex64)
        * native_ctf_inverse_noise[None, :]
    ).astype(np.complex64)
    controlled_projection_term = (
        projections * controlled_weight
    ).astype(np.complex64)
    recovar_data = (
        production_data[rotation_map]
        if production_data is not None
        else np.asarray(recovar["active_summed"], dtype=np.complex64)[recovar_row_mask][rotation_map]
    )
    recovar_weight = (
        production_weight[rotation_map]
        if production_weight is not None
        else np.asarray(recovar["active_ctf_probs"], dtype=np.float32)[recovar_row_mask][rotation_map]
    )
    data_scale = -float(physical_image_size) ** -2
    weight_scale = float(physical_image_size) ** -4
    native_gpu_bpref_data = None
    native_gpu_bpref_weight = None
    native_gpu_bpref_artifacts = {}
    if native_particle_bpref_prefix is not None:
        native_gpu_bpref_data, native_gpu_bpref_weight, native_gpu_bpref_artifacts = (
            _load_native_particle_bpref(
                native_directory,
                native_particle_bpref_prefix,
            )
        )
        native_gpu_bpref_data = native_gpu_bpref_data * np.float32(data_scale)
        native_gpu_bpref_weight = native_gpu_bpref_weight * np.float32(weight_scale)
    operand_comparisons = {}
    if score_dump is not None:
        recovar_shifted = np.asarray(score_dump["debug_shifted_recon"], dtype=np.complex64)
        recovar_ctf2 = np.asarray(score_dump["debug_ctf2_over_nv_recon"], dtype=np.float32)
        recovar_projection = np.asarray(score_dump["debug_proj_for_recon"], dtype=np.complex64)[
            rotation_map
        ]
        recovar_image_term = (
            (current_posterior @ recovar_shifted).astype(np.complex64)
        )
        recovar_projection_term = (
            recovar_projection
            * (np.sum(current_posterior, axis=1, dtype=np.float32)[:, None] * recovar_ctf2[None, :])
        ).astype(np.complex64)
        operand_comparisons = {
            "translated_image_ctf_inverse_noise_operand": _metric(
                translated * native_ctf_inverse_noise[None, :] * data_scale,
                recovar_shifted,
            ),
            "ctf2_inverse_noise_operand": _metric(
                (native_ctf * native_ctf * inverse_noise).astype(np.float32) * weight_scale,
                recovar_ctf2,
            ),
            "projection_operand": _metric(
                projections * (data_scale / weight_scale),
                recovar_projection,
            ),
            "gradient_image_term_same_posterior_control": _metric(
                controlled_image_term * data_scale,
                recovar_image_term,
            ),
            "gradient_projection_term_same_posterior_control": _metric(
                controlled_projection_term * data_scale,
                recovar_projection_term,
            ),
        }
    padding_factor = int(recovar["reconstruction_padding_factor"])
    native_bpref_data, native_bpref_weight = _scatter_relion_rows(
        native_data * data_scale,
        native_weight * weight_scale,
        rotations,
        recovar["window_indices"],
        physical_image_size=physical_image_size,
        current_size=current_size,
        padding_factor=padding_factor,
        get_backprojector_data=get_backprojector_data,
    )
    recovar_bpref_data, recovar_bpref_weight = _scatter_relion_rows(
        recovar_data,
        recovar_weight,
        rotations,
        recovar["window_indices"],
        physical_image_size=physical_image_size,
        current_size=current_size,
        padding_factor=padding_factor,
        get_backprojector_data=get_backprojector_data,
    )
    controlled_bpref_data, controlled_bpref_weight = _scatter_relion_rows(
        controlled_data * data_scale,
        controlled_weight * weight_scale,
        rotations,
        recovar["window_indices"],
        physical_image_size=physical_image_size,
        current_size=current_size,
        padding_factor=padding_factor,
        get_backprojector_data=get_backprojector_data,
    )
    inline_projector_comparisons = _inline_projector_comparisons(
        recovar,
        particle_slot,
        native_bpref_data,
        native_bpref_weight,
        controlled_bpref_data,
        controlled_bpref_weight,
        native_gpu_bpref_data,
        native_gpu_bpref_weight,
    )
    return {
        "schema": SCHEMA,
        "identity": {
            "original_index": int(np.asarray(recovar["original_indices"])[particle_slot]),
            "image_identity": str(np.asarray(recovar["image_identities"])[particle_slot]),
            "iteration": int(recovar["iteration"]),
            "half": int(recovar["half"]),
            "physical_image_size": physical_image_size,
            "current_size": current_size,
            "orientation_count": int(native["orientation_count"]),
            "translation_count": int(native["translation_count"]),
            "retained_mass": float(native["retained_mass"]),
            "reconstruction_padding_factor": padding_factor,
        },
        "frame_scales": {
            "native_data_to_recovar": data_scale,
            "native_weight_to_recovar": weight_scale,
        },
        "comparisons": {
            **posterior_comparisons,
            **operand_comparisons,
            **inline_projector_comparisons,
            "gradient_numerator": _metric(native_data * data_scale, recovar_data),
            "gradient_denominator": _metric(native_weight * weight_scale, recovar_weight),
            "gradient_numerator_same_posterior_control": _metric(
                controlled_data * data_scale,
                recovar_data,
            ),
            "gradient_denominator_same_posterior_control": _metric(
                controlled_weight * weight_scale,
                recovar_weight,
            ),
            "bpref_data_after_relion_scatter": _metric(native_bpref_data, recovar_bpref_data),
            "bpref_weight_after_relion_scatter": _metric(native_bpref_weight, recovar_bpref_weight),
            "bpref_data_same_posterior_control": _metric(
                controlled_bpref_data,
                recovar_bpref_data,
            ),
            "bpref_weight_same_posterior_control": _metric(
                controlled_bpref_weight,
                recovar_bpref_weight,
            ),
        },
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "native_unmasked_image": str(native_image_path.resolve()),
            "native_unmasked_image_sha256": _sha256(native_image_path),
            "native_inverse_noise": str(native_inverse_noise_path.resolve()),
            "native_inverse_noise_sha256": _sha256(native_inverse_noise_path),
            "native_sigma2_noise": str(native_sigma2_noise_path.resolve()),
            "native_sigma2_noise_sha256": _sha256(native_sigma2_noise_path),
            "native_sigma2_fudge": str(native_sigma2_fudge_path.resolve()),
            "native_sigma2_fudge_sha256": _sha256(native_sigma2_fudge_path),
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            **(
                {
                    "recovar_production_score_dump": str(recovar_score_dump.resolve()),
                    "recovar_production_score_dump_sha256": _sha256(recovar_score_dump),
                }
                if recovar_score_dump is not None
                else {}
            ),
            **native_gpu_bpref_artifacts,
        },
        "device": str(jax.devices()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-image", type=Path)
    parser.add_argument("--native-inverse-noise", type=Path)
    parser.add_argument("--native-sigma2-noise", type=Path)
    parser.add_argument("--native-sigma2-fudge", type=Path)
    parser.add_argument("--recovar-original-index", type=int)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument(
        "--recovar-score-dump",
        type=Path,
        help=(
            "Optional production fused-score dump. When supplied, posterior and pre-scatter "
            "M-step rows are replayed from its exact production operands; --recovar-capture "
            "continues to provide reconstruction window/scatter metadata."
        ),
    )
    parser.add_argument("--native-prefix", default="img0_part0_storeWavg_")
    parser.add_argument(
        "--native-projector-prefix",
        help=(
            "Optional independent prefix for projector dims/real/imag files, relative to "
            "--native-directory (for example pass1_class0_ppref_). By default they are "
            "read from <native-prefix>wavg_ppref_."
        ),
    )
    parser.add_argument(
        "--native-particle-bpref-prefix",
        help=(
            "Optional prefix for an isolated native CUDA per-particle backprojector capture, "
            "relative to --native-directory (for example img0_part0_backproject_)."
        ),
    )
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--current-size", type=int, default=38)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--posterior-only", action="store_true")
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.recovar_capture,
        native_image_path=args.native_image,
        native_inverse_noise_path=(
            args.native_inverse_noise
            if args.native_inverse_noise is not None
            else args.native_directory / "Minvsigma2.bin"
        ),
        native_sigma2_noise_path=(
            args.native_sigma2_noise
            if args.native_sigma2_noise is not None
            else args.native_directory / "sigma2_noise.bin"
        ),
        native_sigma2_fudge_path=(
            args.native_sigma2_fudge
            if args.native_sigma2_fudge is not None
            else args.native_directory / "sigma2_fudge.bin"
        ),
        recovar_original_index=args.recovar_original_index,
        native_prefix=args.native_prefix,
        native_projector_prefix=args.native_projector_prefix,
        native_particle_bpref_prefix=args.native_particle_bpref_prefix,
        recovar_score_dump=args.recovar_score_dump,
        physical_image_size=args.physical_image_size,
        current_size=args.current_size,
        rotation_tolerance=args.rotation_tolerance,
        posterior_only=args.posterior_only,
        adaptive_fraction=args.adaptive_fraction,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
