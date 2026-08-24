#!/usr/bin/env python3
"""Locate the first K=1 BPref operand mismatch at primitive level.

The comparison is observational.  It combines a native RELION factor capture,
the matching RECOVAR pass-2 dump, and a RECOVAR high-precision operand bundle.
All arrays are joined by immutable particle identity and Fourier coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar.data_io.image_backends import (
    _centered_rfft2_jax,
    _centered_rfft2_jax_per_image,
    _centered_rfft2_numpy,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_translation_angles_f32,
)

if __package__:
    from .analyze_k1_bpref_factor_boundary import _pixel_coordinates, _translation_map
    from .parse_relion_dump_dir import _read_real_2d
    from .validate_relion_bpref_factor_capture import load_factor_capture
else:
    from analyze_k1_bpref_factor_boundary import (  # type: ignore[no-redef]
        _pixel_coordinates,
        _translation_map,
    )
    from parse_relion_dump_dir import _read_real_2d  # type: ignore[no-redef]
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )


SCHEMA = "recovar.em.k1_bpref_primitive_boundary.v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_live_initial_sigma2(path: Path) -> np.ndarray:
    """Load the sealed RELION spectrum or its lossless NumPy equivalent."""

    if path.suffix == ".npy":
        sigma2 = np.load(path, allow_pickle=False)
    elif path.suffix == ".bin":
        matrix = _read_real_2d(path)
        _require(
            matrix.ndim == 2 and matrix.shape[0] == 1,
            "RELION sigma2_noise.bin must contain exactly one spectrum row",
        )
        sigma2 = matrix[0]
    else:
        raise ValueError("live initial sigma2 must be a .npy or RELION .bin file")
    sigma2 = np.asarray(sigma2)
    _require(
        sigma2.ndim == 1 and sigma2.dtype == np.float64 and sigma2.size > 0,
        "live initial sigma2 must be a nonempty one-dimensional float64 array",
    )
    _require(np.all(np.isfinite(sigma2)), "live initial sigma2 must be finite")
    _require(np.all(sigma2 > 0.0), "live initial sigma2 must be strictly positive")
    return sigma2


def _load_selection(path: Path) -> tuple[dict[str, Any], int, float, int]:
    """Load the pinned geometry required to replay production preprocessing."""

    selection = json.loads(path.read_text())
    image_size = int(selection["physical_image_size"])
    particle_diameter_ang = float(selection["particle_diameter_ang"])
    width_mask_edge_px = int(selection["width_mask_edge_px"])
    _require(image_size > 0, "physical image size must be positive")
    _require(particle_diameter_ang > 0.0, "particle diameter must be positive")
    _require(width_mask_edge_px > 0, "mask-edge width must be positive")
    return selection, image_size, particle_diameter_ang, width_mask_edge_px


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "comparison shape changed or is empty")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _standard_half_indices(centered_indices: np.ndarray, image_size: int) -> np.ndarray:
    indices = np.asarray(centered_indices, dtype=np.int64)
    half_width = image_size // 2 + 1
    centered_rows = indices // half_width
    return (
        ((centered_rows - image_size // 2) % image_size) * half_width
        + indices % half_width
    ).astype(np.int32, copy=False)


def _star_column(table, name: str):
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return table[candidate]
    raise ValueError(f"STAR column {name} is missing")


def _load_relion_ctf_inputs(source_star: Path, relion_bind_directory: Path):
    os.environ["RECOVAR_RELION_BIND_BUILD_DIR"] = str(relion_bind_directory.resolve())
    from recovar.data_io.starfile import read_star
    from recovar.relion_bind import _relion_bind_core as relion_bind

    particles, optics = read_star(str(source_star))
    _require(optics is not None, "source STAR has no optics table")
    optics_ids = np.asarray(_star_column(optics, "rlnOpticsGroup"), dtype=np.int64)
    _require(np.unique(optics_ids).size == optics_ids.size, "duplicate optics-group IDs")
    optics_by_id = {int(group): optics.iloc[row] for row, group in enumerate(optics_ids)}
    return particles, optics_by_id, relion_bind


def _native_ctf_image(
    *,
    particle_row,
    optics_row,
    relion_bind,
    image_size: int,
) -> np.ndarray:
    def particle_value(name: str) -> float:
        return float(particle_row[name] if name in particle_row else particle_row[f"_{name}"])

    def optics_value(name: str) -> float:
        return float(optics_row[name] if name in optics_row else optics_row[f"_{name}"])

    return np.asarray(
        relion_bind.get_ctf_image(
            particle_value("rlnDefocusU"),
            particle_value("rlnDefocusV"),
            particle_value("rlnDefocusAngle"),
            optics_value("rlnVoltage"),
            optics_value("rlnSphericalAberration"),
            optics_value("rlnAmplitudeContrast"),
            0.0,
            optics_value("rlnImagePixelSize"),
            image_size,
            image_size,
            False,
            False,
            False,
            particle_value("rlnPhaseShift"),
            1.0,
        ),
        dtype=np.float64,
    )


def _load_contribution_particle(
    directory: Path, original_index: int
) -> tuple[Path, dict[str, np.ndarray], int]:
    matches = []
    for path in sorted(directory.glob("bpref_contribution_rows_it001_h*_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            rows = np.flatnonzero(
                np.asarray(archive["original_indices"], dtype=np.int64) == original_index
            )
            if rows.size:
                _require(rows.size == 1, f"duplicate particle {original_index} in {path}")
                matches.append((path, {name: archive[name] for name in archive.files}, int(rows[0])))
    _require(len(matches) == 1, f"expected one contribution bundle for particle {original_index}")
    return matches[0]


def _first_nonzero_boundary(comparisons: dict[str, dict[str, Any]]) -> str:
    for name in (
        "ctf_with_scale",
        "inverse_noise",
        "weighted_ctf",
        "processed_fourier_image",
        "translated_fourier_image",
        "numerator_operand",
        "denominator_operand",
    ):
        if not comparisons[name]["exact_equal"]:
            return name
    return "all_compared_primitives_exact"


def _compare_particle(
    *,
    target: dict[str, Any],
    factor,
    pass2_path: Path,
    contribution_path: Path,
    contribution: dict[str, np.ndarray],
    contribution_row: int,
    image_size: int,
    particle_diameter_ang: float,
    width_mask_edge_px: int,
    dump_directory: Path,
    source_particles,
    source_optics_by_id: dict[int, Any],
    relion_bind,
    live_initial_sigma2: np.ndarray,
) -> dict[str, Any]:
    stack = int(target["stack_index_one_based"])
    original = int(target["original_index_zero_based"])
    with np.load(pass2_path, allow_pickle=False) as archive:
        pass2 = {name: archive[name] for name in archive.files}
    _require(int(pass2["original_index"]) == original, f"stack {stack}: pass-2 identity changed")
    _require(
        int(contribution["stack_indices_1based"][contribution_row]) == stack,
        f"stack {stack}: contribution identity changed",
    )
    _require(bool(contribution["high_precision_operand_bundle"]), f"stack {stack}: operand bundle missing")
    _require(int(contribution["iteration"]) == 1, f"stack {stack}: iteration changed")

    centered = np.asarray(pass2["recon_window_indices"], dtype=np.int32)
    standard = np.asarray(contribution["window_indices"], dtype=np.int32)
    coordinates = _pixel_coordinates(centered, image_size)
    coordinates_array = np.asarray(coordinates, dtype=np.int64)
    half_width = image_size // 2 + 1
    standard_rows = standard // half_width
    standard_coordinates = np.column_stack(
        (
            standard % half_width,
            np.where(
                standard_rows <= image_size // 2,
                standard_rows,
                standard_rows - image_size,
            ),
        )
    )
    _require(
        np.array_equal(coordinates_array, standard_coordinates),
        f"stack {stack}: pass-2/contribution physical Fourier window changed",
    )
    native_lookup = {
        (int(x), int(y)): row
        for row, (x, y) in enumerate(zip(factor.pixels["x"], factor.pixels["y"], strict=True))
    }
    _require(all(key in native_lookup for key in coordinates), f"stack {stack}: native pixels missing")
    native_rows = np.asarray([native_lookup[key] for key in coordinates], dtype=np.int64)

    accepted_rows = np.flatnonzero((factor.hypotheses["flags"] & 1) != 0)
    _require(accepted_rows.size == 1, f"stack {stack}: expected one firstiter-CC winner")
    accepted = factor.hypotheses[int(accepted_rows[0])]
    translation_map, translation_error = _translation_map(
        factor.translations,
        pass2["fine_translations"],
        physical_image_size=image_size,
    )
    rec_translation = int(translation_map[int(accepted["translation"])])
    terms = factor.terms.reshape(accepted_rows.size, factor.pixels.size)[0][native_rows]
    _require(
        np.all(terms["translation"] == int(accepted["translation"])),
        f"stack {stack}: native term translation changed",
    )
    posterior = np.float32(accepted["posterior_over_weight_norm"])
    _require(posterior == np.float32(1.0), f"stack {stack}: firstiter posterior is not exactly one")

    particle_row = source_particles.iloc[original]
    optics_group = int(
        particle_row["rlnOpticsGroup"]
        if "rlnOpticsGroup" in particle_row
        else particle_row["_rlnOpticsGroup"]
    )
    optics_row = source_optics_by_id[optics_group]
    native_ctf_fftw = _native_ctf_image(
        particle_row=particle_row,
        optics_row=optics_row,
        relion_bind=relion_bind,
        image_size=image_size,
    )
    native_ctf_replay = native_ctf_fftw.reshape(-1)[standard].astype(np.float32)
    rec_noise = np.asarray(contribution["noise_variance_half"], dtype=np.float32)[centered]
    rec_scale = np.float32(contribution["scale_corrections"][contribution_row])
    n2 = np.float32(image_size**2)
    n4 = np.float32(image_size**4)
    # The active exact-BPref path evaluates the CTF from the immutable source
    # STAR with RELION's scalar implementation, then casts once to float32.
    # Do not replay the rounded diagnostic ctf_params array here.
    rec_ctf_aligned = (native_ctf_replay * rec_scale).astype(np.float32)
    rec_inverse_noise_native = (
        (np.float32(1.0) / rec_noise).astype(np.float32) * n4
    ).astype(np.float32)
    rec_weighted_ctf = (
        -rec_ctf_aligned * rec_inverse_noise_native / n4
    ).astype(np.float32)
    rec_denominator = np.asarray(pass2["ctf2_over_nv_recon"], dtype=np.float32)
    rec_shifted_weighted = np.asarray(pass2["shifted_recon"], dtype=np.complex64)[rec_translation]

    valid_weight = np.abs(rec_weighted_ctf) > np.float32(1.0e-20)
    _require(np.count_nonzero(valid_weight) > 0, f"stack {stack}: no valid weighted-CTF pixels")
    rec_translated = np.zeros_like(rec_shifted_weighted)
    rec_translated[valid_weight] = (
        rec_shifted_weighted[valid_weight] / rec_weighted_ctf[valid_weight]
    ).astype(np.complex64)

    raw_real_image = np.asarray(
        contribution["raw_real_images"][contribution_row], dtype=np.float32
    )
    host_fft_replay = _centered_rfft2_numpy(raw_real_image)[0].reshape(-1)[centered].astype(
        np.complex64
    )
    cuda_fft_full = _centered_rfft2_jax(jnp.asarray(raw_real_image, dtype=jnp.float32))
    cuda_fft_replay = np.asarray(jax.block_until_ready(cuda_fft_full)).reshape(1, -1)[
        0, centered
    ].astype(np.complex64)
    preprocess_backend = str(np.asarray(contribution["preprocess_backend"]).item())
    _require(
        preprocess_backend in {"dataset_native", "relion_cuda"},
        f"stack {stack}: unsupported preprocessing backend {preprocess_backend!r}",
    )
    production_preprocessed_real = raw_real_image[None, ...]
    if preprocess_backend == "relion_cuda":
        from recovar import cuda_backproject

        def optics_value(name: str) -> float:
            return float(optics_row[name] if name in optics_row else optics_row[f"_{name}"])

        pixel_size = optics_value("rlnImagePixelSize")
        _require(pixel_size > 0.0, f"stack {stack}: invalid source pixel size")
        radius = float(particle_diameter_ang) / (2.0 * pixel_size)
        normalization_factors = np.asarray(
            contribution["relion_preprocess_normalization_factors"][
                contribution_row : contribution_row + 1
            ],
            dtype=np.float32,
        )
        integer_shifts = np.asarray(
            contribution["integer_pre_shifts"][contribution_row : contribution_row + 1],
            dtype=np.int32,
        )
        _, production_preprocessed_device = cuda_backproject.relion_preprocess_real_f32(
            jnp.asarray(raw_real_image[None, ...], dtype=jnp.float32),
            jnp.asarray(normalization_factors, dtype=jnp.float32),
            jnp.asarray(integer_shifts, dtype=jnp.int32),
            radius,
            float(width_mask_edge_px),
            False,
            native_lane_reduction=bool(contribution["relion_native_lane_reduction"]),
        )
        production_preprocessed_real = np.asarray(
            jax.block_until_ready(production_preprocessed_device),
            dtype=np.float32,
        )
        production_fft_device = _centered_rfft2_jax_per_image(
            production_preprocessed_device
        )
        production_fft_full = np.asarray(
            jax.block_until_ready(production_fft_device),
            dtype=np.complex64,
        )
        rec_processed = production_fft_full.reshape(1, -1)[0, centered]
    else:
        rec_processed = host_fft_replay

    rel_processed = (
        factor.pixels["image_re"][native_rows]
        + np.complex64(1j) * factor.pixels["image_im"][native_rows]
    ).astype(np.complex64) * n2
    rel_ctf_aligned = factor.pixels["ctf"][native_rows].astype(np.float32)
    rel_inverse_noise_native = factor.pixels["minvsigma2"][native_rows].astype(np.float32)
    rel_translated = (
        terms["translated_re"] + np.complex64(1j) * terms["translated_im"]
    ).astype(np.complex64) * n2
    rel_weighted_ctf = (-terms["weighted_ctf"] / n4).astype(np.float32)
    rel_numerator = (
        -(terms["term_re"] + np.complex64(1j) * terms["term_im"]) / n2
    ).astype(np.complex64)
    rel_denominator = (terms["weight_term"] / n4).astype(np.float32)

    shells = np.rint(np.linalg.norm(coordinates_array, axis=1)).astype(np.int64)
    _require(int(shells.max(initial=0)) < live_initial_sigma2.size, "noise shell is out of range")
    native_inverse_noise_replay = (
        np.float64(1.0) / live_initial_sigma2[shells]
    ).astype(np.float32)
    # The active circular window contains only nonzero RELION weights.  Keeping
    # this assertion here prevents an out-of-window zero from looking like a
    # radial-spectrum mismatch.
    _require(np.all(rel_inverse_noise_native != 0.0), f"stack {stack}: inactive pixel leaked into window")
    native_replay_weighted_ctf = (
        -native_ctf_replay * native_inverse_noise_replay / n4
    ).astype(np.float32)
    # RELION's BPref kernel evaluates weight*CTF*CTF from left to right in
    # float32.  Squaring CTF first changes hundreds of pixels by one ULP.
    native_replay_denominator = (
        (native_inverse_noise_replay * native_ctf_replay) * native_ctf_replay / n4
    ).astype(np.float32)
    from recovar import cuda_backproject

    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(pass2["fine_translations"], (image_size, image_size)),
        dtype=jnp.float32,
    )
    cuda_translated_all = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(rec_processed[None, :], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(centered, dtype=jnp.int32),
        (image_size, image_size),
    )
    cuda_translated_replay = np.asarray(jax.block_until_ready(cuda_translated_all))[
        rec_translation
    ].astype(np.complex64)
    native_translation_angles = np.stack(
        (factor.translations["x"], factor.translations["y"]),
        axis=1,
    ).astype(np.float32)
    cuda_native_angle_translated_all = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(rec_processed[None, :], dtype=jnp.complex64),
        jnp.asarray(native_translation_angles, dtype=jnp.float32),
        jnp.asarray(centered, dtype=jnp.int32),
        (image_size, image_size),
    )
    cuda_native_angle_translated_replay = np.asarray(
        jax.block_until_ready(cuda_native_angle_translated_all)
    )[int(accepted["translation"])].astype(np.complex64)
    relion_normalized_rec_processed = (
        rec_processed * np.float32(1.0 / float(image_size**2))
    ).astype(np.complex64)
    relion_normalized_native_angle_translated_all = (
        cuda_backproject.relion_translate_score_f32(
            jnp.asarray(relion_normalized_rec_processed[None, :], dtype=jnp.complex64),
            jnp.asarray(native_translation_angles, dtype=jnp.float32),
            jnp.asarray(centered, dtype=jnp.int32),
            (image_size, image_size),
        )
    )
    relion_normalized_native_angle_translated_replay = (
        np.asarray(
            jax.block_until_ready(relion_normalized_native_angle_translated_all)
        )[int(accepted["translation"])].astype(np.complex64)
        * n2
    )
    relion_normalized_native_weight_bpref_all = (
        cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(relion_normalized_rec_processed[None, :], dtype=jnp.complex64),
            jnp.asarray((rel_weighted_ctf * n2)[None, :], dtype=jnp.float32),
            jnp.asarray(native_translation_angles, dtype=jnp.float32),
            jnp.asarray(centered, dtype=jnp.int32),
            (image_size, image_size),
        )
    )
    relion_normalized_native_weight_bpref_replay = np.asarray(
        jax.block_until_ready(relion_normalized_native_weight_bpref_all)
    )[int(accepted["translation"])].astype(np.complex64)
    native_raw_weight_bpref_all = cuda_backproject.relion_translate_bpref_f32(
        jnp.asarray(relion_normalized_rec_processed[None, :], dtype=jnp.complex64),
        jnp.asarray(
            (-native_ctf_replay * native_inverse_noise_replay)[None, :],
            dtype=jnp.float32,
        ),
        jnp.asarray(native_translation_angles, dtype=jnp.float32),
        jnp.asarray(centered, dtype=jnp.int32),
        (image_size, image_size),
    )
    native_raw_weight_then_scaled_bpref_replay = (
        np.asarray(jax.block_until_ready(native_raw_weight_bpref_all))[
            int(accepted["translation"])
        ].astype(np.complex64)
        / n2
    ).astype(np.complex64)
    native_scaled_weight_bpref_all = cuda_backproject.relion_translate_bpref_f32(
        jnp.asarray(relion_normalized_rec_processed[None, :], dtype=jnp.complex64),
        jnp.asarray(
            (
                (-native_ctf_replay * native_inverse_noise_replay) / n2
            )[None, :],
            dtype=jnp.float32,
        ),
        jnp.asarray(native_translation_angles, dtype=jnp.float32),
        jnp.asarray(centered, dtype=jnp.int32),
        (image_size, image_size),
    )
    native_scaled_weight_bpref_replay = np.asarray(
        jax.block_until_ready(native_scaled_weight_bpref_all)
    )[int(accepted["translation"])].astype(np.complex64)
    relion_normalized_recovar_weight_bpref_all = (
        cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(relion_normalized_rec_processed[None, :], dtype=jnp.complex64),
            jnp.asarray((rec_weighted_ctf * n2)[None, :], dtype=jnp.float32),
            jnp.asarray(native_translation_angles, dtype=jnp.float32),
            jnp.asarray(centered, dtype=jnp.int32),
            (image_size, image_size),
        )
    )
    relion_normalized_recovar_weight_bpref_replay = np.asarray(
        jax.block_until_ready(relion_normalized_recovar_weight_bpref_all)
    )[int(accepted["translation"])].astype(np.complex64)
    phase_valid = (
        (np.abs(rel_processed) > np.float32(1.0e-12))
        & (np.abs(rec_processed) > np.float32(1.0e-12))
        & valid_weight
    )
    relion_phase_observed = (
        rel_translated[phase_valid] / rel_processed[phase_valid]
    ).astype(np.complex64)
    recovar_phase_observed = (
        rec_translated[phase_valid] / rec_processed[phase_valid]
    ).astype(np.complex64)

    rec_numerator = rec_shifted_weighted
    comparisons = {
        "ctf_with_scale": _metric(rel_ctf_aligned, rec_ctf_aligned),
        "inverse_noise": _metric(rel_inverse_noise_native, rec_inverse_noise_native),
        "weighted_ctf": _metric(rel_weighted_ctf, rec_weighted_ctf),
        "processed_fourier_image": _metric(rel_processed[valid_weight], rec_processed[valid_weight]),
        "processed_fourier_image_relion_normalization_roundtrip": _metric(
            rel_processed[valid_weight],
            (relion_normalized_rec_processed * n2)[valid_weight],
        ),
        "translated_fourier_image": _metric(rel_translated[valid_weight], rec_translated[valid_weight]),
        "numerator_operand": _metric(rel_numerator, rec_numerator),
        "denominator_operand": _metric(rel_denominator, rec_denominator),
        "relion_internal_weighted_ctf": _metric(
            rel_weighted_ctf,
            (-rel_ctf_aligned * rel_inverse_noise_native / n4).astype(np.float32),
        ),
        "relion_internal_numerator": _metric(
            rel_numerator,
            (rel_translated * rel_weighted_ctf).astype(np.complex64),
        ),
        "relion_internal_denominator": _metric(
            rel_denominator,
            (
                (rel_inverse_noise_native * rel_ctf_aligned) * rel_ctf_aligned / n4
            ).astype(np.float32),
        ),
        "recovar_internal_denominator": _metric(
            rec_denominator,
            (
                (rec_inverse_noise_native * rec_ctf_aligned) * rec_ctf_aligned / n4
            ).astype(np.float32),
        ),
        "relion_ctf_replayed_from_source_star": _metric(
            rel_ctf_aligned,
            native_ctf_replay,
        ),
        "relion_inverse_noise_replayed_from_float64_sigma2": _metric(
            rel_inverse_noise_native,
            native_inverse_noise_replay,
        ),
        "recovar_processed_image_vs_host_numpy_fft": _metric(
            rec_processed,
            host_fft_replay,
        ),
        "relion_processed_image_vs_batched_cuda_fft": _metric(
            rel_processed,
            cuda_fft_replay,
        ),
        "translation_phase": _metric(
            relion_phase_observed,
            recovar_phase_observed,
        ),
        "relion_translated_image_replayed_with_cuda_sincosf": _metric(
            rel_translated,
            cuda_translated_replay,
        ),
        "relion_translated_image_replayed_with_native_angles": _metric(
            rel_translated,
            cuda_native_angle_translated_replay,
        ),
        "relion_translated_image_replayed_with_normalization_and_native_angles": _metric(
            rel_translated,
            relion_normalized_native_angle_translated_replay,
        ),
    }
    counterfactuals = {
        "native_translated_with_recovar_weighted_ctf": _metric(
            rel_numerator,
            (rel_translated * rec_weighted_ctf).astype(np.complex64),
        ),
        "recovar_translated_with_native_weighted_ctf": _metric(
            rel_numerator,
            (rec_translated * rel_weighted_ctf).astype(np.complex64),
        ),
        "native_ctf_with_recovar_inverse_noise": _metric(
            rel_weighted_ctf,
            (-rel_ctf_aligned * rec_inverse_noise_native / n4).astype(np.float32),
        ),
        "recovar_ctf_with_native_inverse_noise": _metric(
            rel_weighted_ctf,
            (-rec_ctf_aligned * rel_inverse_noise_native / n4).astype(np.float32),
        ),
        "native_ctf_and_inverse_noise_replay_weighted_ctf": _metric(
            rel_weighted_ctf,
            native_replay_weighted_ctf,
        ),
        "native_ctf_and_inverse_noise_replay_denominator": _metric(
            rel_denominator,
            native_replay_denominator,
        ),
        "recovar_translated_with_native_ctf_and_inverse_noise": _metric(
            rel_numerator,
            (rec_translated * native_replay_weighted_ctf).astype(np.complex64),
        ),
        "recovar_processed_with_recovar_phase_and_native_ctf_noise": _metric(
            rel_numerator[phase_valid],
            (
                (rec_processed[phase_valid] * recovar_phase_observed).astype(np.complex64)
                * native_replay_weighted_ctf[phase_valid]
            ).astype(np.complex64),
        ),
        "cuda_fft_sincosf_with_native_ctf_noise": _metric(
            rel_numerator,
            (
                cuda_translated_replay
                * native_replay_weighted_ctf
            ).astype(np.complex64),
        ),
        "cuda_fft_native_angles_with_native_ctf_noise": _metric(
            rel_numerator,
            (
                cuda_native_angle_translated_replay
                * native_replay_weighted_ctf
            ).astype(np.complex64),
        ),
        "relion_normalized_cuda_fft_native_angles_with_native_ctf_noise": _metric(
            rel_numerator,
            (
                relion_normalized_native_angle_translated_replay
                * native_replay_weighted_ctf
            ).astype(np.complex64),
        ),
        "relion_normalized_fft_native_angles_native_weight_bpref_kernel": _metric(
            rel_numerator,
            relion_normalized_native_weight_bpref_replay,
        ),
        "relion_normalized_fft_native_raw_weight_then_output_scale": _metric(
            rel_numerator,
            native_raw_weight_then_scaled_bpref_replay,
        ),
        "relion_normalized_fft_direct_scaled_native_weight": _metric(
            rel_numerator,
            native_scaled_weight_bpref_replay,
        ),
        "relion_normalized_fft_native_angles_recovar_weight_bpref_kernel": _metric(
            rel_numerator,
            relion_normalized_recovar_weight_bpref_replay,
        ),
    }

    dump_path = dump_directory / f"stack{stack}_it1_bpref_primitives.npz"
    np.savez_compressed(
        dump_path,
        centered_packed_indices=centered,
        standard_half_indices=standard,
        valid_weight=valid_weight,
        relion_ctf_with_scale=rel_ctf_aligned,
        recovar_ctf_with_scale=rec_ctf_aligned,
        relion_inverse_noise=rel_inverse_noise_native,
        recovar_inverse_noise=rec_inverse_noise_native,
        relion_weighted_ctf=rel_weighted_ctf,
        recovar_weighted_ctf=rec_weighted_ctf,
        relion_processed_fourier_image=rel_processed,
        recovar_processed_fourier_image=rec_processed,
        relion_translated_fourier_image=rel_translated,
        recovar_translated_fourier_image=rec_translated,
        relion_numerator_operand=rel_numerator,
        recovar_numerator_operand=rec_numerator,
        relion_denominator_operand=rel_denominator,
        recovar_denominator_operand=rec_denominator,
        relion_ctf_replay_from_source_star=native_ctf_replay,
        relion_inverse_noise_replay_from_float64_sigma2=native_inverse_noise_replay,
        native_replay_weighted_ctf=native_replay_weighted_ctf,
        native_replay_denominator=native_replay_denominator,
        raw_real_image=raw_real_image,
        production_preprocessed_real_image=production_preprocessed_real,
        host_numpy_fft_replay=host_fft_replay,
        cuda_fft_replay=cuda_fft_replay,
        production_backend_fft_replay=rec_processed,
        cuda_sincosf_translated_replay=cuda_translated_replay,
        cuda_native_angle_translated_replay=cuda_native_angle_translated_replay,
        relion_normalized_rec_processed=relion_normalized_rec_processed,
        relion_normalized_native_angle_translated_replay=(
            relion_normalized_native_angle_translated_replay
        ),
        relion_normalized_native_weight_bpref_replay=(
            relion_normalized_native_weight_bpref_replay
        ),
        relion_normalized_recovar_weight_bpref_replay=(
            relion_normalized_recovar_weight_bpref_replay
        ),
        native_translation_angles=native_translation_angles,
        phase_valid=phase_valid,
        relion_translation_phase_observed=relion_phase_observed,
        recovar_translation_phase_observed=recovar_phase_observed,
    )
    return {
        "stack_index_one_based": stack,
        "original_index_zero_based": original,
        "factor_capture": str(factor.path.resolve()),
        "factor_capture_sha256": factor.sha256,
        "pass2_capture": str(pass2_path.resolve()),
        "pass2_capture_sha256": _sha256(pass2_path),
        "contribution_bundle": str(contribution_path.resolve()),
        "contribution_bundle_sha256": _sha256(contribution_path),
        "operand_dump": str(dump_path.resolve()),
        "operand_dump_sha256": _sha256(dump_path),
        "translation_map_max_abs": translation_error,
        "recovar_scale_correction": float(rec_scale),
        "recovar_preprocess_backend": preprocess_backend,
        "recovar_preprocess_normalization_factor_field": float(
            contribution["relion_preprocess_normalization_factors"][contribution_row]
        ),
        "comparisons": comparisons,
        "counterfactuals": counterfactuals,
        "first_exact_unequal_boundary": _first_nonzero_boundary(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--pass2-directory", type=Path, required=True)
    parser.add_argument("--contribution-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dump-directory", type=Path, required=True)
    parser.add_argument("--source-star", type=Path, required=True)
    parser.add_argument("--relion-bind-directory", type=Path, required=True)
    parser.add_argument(
        "--live-initial-sigma2",
        type=Path,
        required=True,
        help="Sealed RELION sigma2_noise.bin or its lossless float64 .npy equivalent",
    )
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    selection, image_size, particle_diameter_ang, width_mask_edge_px = _load_selection(
        args.selection_json
    )
    factors = {
        capture.stack_index: capture
        for capture in (
            load_factor_capture(path)
            for path in args.factor_directory.glob("*.bpre-v2.bin")
        )
    }
    source_particles, source_optics_by_id, relion_bind = _load_relion_ctf_inputs(
        args.source_star, args.relion_bind_directory
    )
    live_initial_sigma2 = _load_live_initial_sigma2(args.live_initial_sigma2)
    args.dump_directory.mkdir(parents=True, exist_ok=False)
    particles = []
    for target in selection["targets"]:
        stack = int(target["stack_index_one_based"])
        original = int(target["original_index_zero_based"])
        contribution_path, contribution, contribution_row = _load_contribution_particle(
            args.contribution_directory, original
        )
        pass2_matches = sorted(args.pass2_directory.glob(f"pass2_orig{original:06d}_cs*.npz"))
        _require(len(pass2_matches) == 1, f"stack {stack}: expected one pass-2 dump")
        particles.append(
            _compare_particle(
                target=target,
                factor=factors[stack],
                pass2_path=pass2_matches[0],
                contribution_path=contribution_path,
                contribution=contribution,
                contribution_row=contribution_row,
                image_size=image_size,
                particle_diameter_ang=particle_diameter_ang,
                width_mask_edge_px=width_mask_edge_px,
                dump_directory=args.dump_directory,
                source_particles=source_particles,
                source_optics_by_id=source_optics_by_id,
                relion_bind=relion_bind,
                live_initial_sigma2=live_initial_sigma2,
            )
        )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact and relative-L2 intermediates; no correlation",
        "device": str(jax.devices()[0]),
        "selection_json": str(args.selection_json.resolve()),
        "selection_sha256": _sha256(args.selection_json),
        "source_star": str(args.source_star.resolve()),
        "source_star_sha256": _sha256(args.source_star),
        "relion_bind_module": str(Path(relion_bind.__file__).resolve()),
        "relion_bind_module_sha256": _sha256(Path(relion_bind.__file__)),
        "live_initial_sigma2": str(args.live_initial_sigma2.resolve()),
        "live_initial_sigma2_sha256": _sha256(args.live_initial_sigma2),
        "particle_count": len(particles),
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
