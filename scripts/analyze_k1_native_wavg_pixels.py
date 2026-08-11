#!/usr/bin/env python3
"""Compare native RELION and RECOVAR iteration-1 Wavg pixels for a fixed panel."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path

import numpy as np

if __package__:
    from scripts.compare_k1_relion_recovar_bpref_primitives import (
        _load_relion_ctf_inputs,
        _native_ctf_image,
        _standard_half_indices,
    )
    from scripts.validate_relion_preprocess_capture import load_artifact
else:
    from compare_k1_relion_recovar_bpref_primitives import (  # type: ignore[no-redef]
        _load_relion_ctf_inputs,
        _native_ctf_image,
        _standard_half_indices,
    )
    from validate_relion_preprocess_capture import load_artifact  # type: ignore[no-redef]


_WDIFF_RE = re.compile(r"img(?P<img>\d+)_part(?P<part>\d+)_storeWavg_wdiff2_pixels\.bin")
_PREPROCESS_RE = re.compile(r"part(?P<part>\d+)_stack(?P<stack>\d+)\.preprocess-v1\.bin")


def _load_counted(path: Path, dtype: np.dtype | type) -> np.ndarray:
    """Load the diagnostic format: uint64 count followed by a flat native array."""

    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated counted array: {path}")
    count = struct.unpack_from("<Q", raw)[0]
    array = np.frombuffer(raw, dtype=dtype, offset=8)
    if array.size != count:
        raise ValueError(f"count mismatch in {path}: header={count}, payload={array.size}")
    return array.copy()


def _native_standard_half_indices(current_size: int, image_size: int) -> np.ndarray:
    """Map RELION's current-size packed Wavg rectangle into the full packed FFT."""

    current_half_width = current_size // 2 + 1
    native_rows = np.arange(current_size, dtype=np.int64)
    logical_rows = np.where(
        native_rows <= current_size // 2,
        native_rows,
        native_rows - current_size,
    )
    full_half_width = image_size // 2 + 1
    return (
        (logical_rows[:, None] % image_size) * full_half_width
        + np.arange(current_half_width, dtype=np.int64)[None, :]
    ).reshape(-1)


def _recovar_rows_in_native_order(
    window_indices: np.ndarray,
    *,
    current_size: int,
    image_size: int,
) -> np.ndarray:
    recovar_standard = _standard_half_indices(window_indices, image_size).astype(np.int64)
    native_standard = _native_standard_half_indices(current_size, image_size)
    if np.unique(recovar_standard).size != recovar_standard.size:
        raise ValueError("RECOVAR score-window indices are not unique")
    row_by_standard = {int(value): row for row, value in enumerate(recovar_standard)}
    try:
        result = np.asarray([row_by_standard[int(value)] for value in native_standard], dtype=np.int32)
    except KeyError as error:
        raise ValueError(f"native Wavg pixel is missing from RECOVAR's score window: {error}") from error
    if result.size != recovar_standard.size:
        raise ValueError("native and RECOVAR Wavg rectangles differ in size")
    return result


def _wavg_components(
    projections: np.ndarray,
    translated_images: np.ndarray,
    ctf: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay the deployed float32 Wavg loop and its per-orientation atomics."""

    proj = np.asarray(projections, dtype=np.complex64)
    images = np.asarray(translated_images, dtype=np.complex64)
    ctf_f32 = np.asarray(ctf, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)
    if proj.ndim != 2 or images.ndim != 2 or ctf_f32.ndim != 1:
        raise ValueError("Wavg projections, images, and CTF must be flat pixel panels")
    if probs.shape != (proj.shape[0], images.shape[0]):
        raise ValueError("Wavg probability shape does not match rotations and translations")
    if proj.shape[1] != images.shape[1] or proj.shape[1] != ctf_f32.size:
        raise ValueError("Wavg pixel dimensions do not match")

    totals = {
        name: np.zeros(ctf_f32.size, dtype=np.float32)
        for name in ("wdiff2", "aa", "xa", "image_power")
    }
    for rotation_row in range(proj.shape[0]):
        ref_real = np.asarray(proj[rotation_row].real * ctf_f32, dtype=np.float32)
        ref_imag = np.asarray(proj[rotation_row].imag * ctf_f32, dtype=np.float32)
        rotation = {name: np.zeros(ctf_f32.size, dtype=np.float32) for name in totals}
        for translation_row in range(images.shape[0]):
            weight = probs[rotation_row, translation_row]
            if weight == 0:
                continue
            image_real = images[translation_row].real
            image_imag = images[translation_row].imag
            diff_real = np.asarray(ref_real - image_real, dtype=np.float32)
            diff_imag = np.asarray(ref_imag - image_imag, dtype=np.float32)
            terms = {
                "wdiff2": np.asarray(
                    np.asarray(diff_real * diff_real, dtype=np.float32)
                    + np.asarray(diff_imag * diff_imag, dtype=np.float32),
                    dtype=np.float32,
                ),
                "aa": np.asarray(
                    np.asarray(ref_real * ref_real, dtype=np.float32)
                    + np.asarray(ref_imag * ref_imag, dtype=np.float32),
                    dtype=np.float32,
                ),
                "xa": np.asarray(
                    np.asarray(ref_real * image_real, dtype=np.float32)
                    + np.asarray(ref_imag * image_imag, dtype=np.float32),
                    dtype=np.float32,
                ),
                "image_power": np.asarray(
                    np.asarray(image_real * image_real, dtype=np.float32)
                    + np.asarray(image_imag * image_imag, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
            for name, term in terms.items():
                rotation[name] = np.asarray(
                    rotation[name] + np.asarray(weight * term, dtype=np.float32),
                    dtype=np.float32,
                )
        for name in totals:
            totals[name] = np.asarray(totals[name] + rotation[name], dtype=np.float32)
    return totals


def _replace_window_with_native_preprocess(
    processed_score: np.ndarray,
    window_indices: np.ndarray,
    native_rows: np.ndarray,
    native_masked_fourier: np.ndarray,
    fourier_scale: np.float32,
) -> np.ndarray:
    """Inject native RELION preprocessing bytes into RECOVAR coordinates."""

    result = np.asarray(processed_score, dtype=np.complex64).copy()
    window = np.asarray(window_indices, dtype=np.int32)
    rows = np.asarray(native_rows, dtype=np.int32)
    native = np.asarray(native_masked_fourier, dtype=np.complex64)
    if native.shape != rows.shape:
        raise ValueError("native preprocessing and Wavg row mapping shapes differ")
    # ``rows`` gathers a RECOVAR-window array into native Wavg order.  Scatter
    # the native bytes through the same mapping and undo RELION's exact 1/N^2
    # FFT scale so the result has RECOVAR's internal Fourier convention.
    result[window[rows]] = np.asarray(native / fourier_scale, dtype=np.complex64)
    return result


def _translate_native_preprocess_hybrid(
    processed_score: np.ndarray,
    fine_translations: np.ndarray,
    window_indices: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Apply RECOVAR's exact RELION CUDA translation to injected native bytes."""

    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    translation_angles = _relion_translation_angles_f32(
        np.asarray(fine_translations, dtype=np.float32),
        (image_size, image_size),
    )
    translated = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(processed_score[window_indices][None, :], dtype=jnp.complex64),
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(window_indices, dtype=jnp.int32),
        (image_size, image_size),
    )
    return np.asarray(jax.block_until_ready(translated), dtype=np.complex64)


def _translate_native_wavg_inputs(
    native_image: np.ndarray,
    native_translation_angles: np.ndarray,
    native_centered_indices: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Translate bytes captured at RELION's Wavg kernel-input boundary."""

    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    translated = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(np.asarray(native_image, dtype=np.complex64)[None, :]),
        jnp.asarray(native_translation_angles, dtype=jnp.float32),
        jnp.asarray(native_centered_indices, dtype=jnp.int32),
        (image_size, image_size),
    )
    return np.asarray(jax.block_until_ready(translated), dtype=np.complex64)


def _normalise_native_weights(
    raw_weights: np.ndarray,
    orientation_num: int,
    translation_num: int,
) -> np.ndarray:
    """Convert RELION's dense lowest-float/sparse-weight table to probabilities."""

    weights = np.asarray(raw_weights, dtype=np.float32).reshape(
        orientation_num,
        translation_num,
    )
    present = weights != np.finfo(np.float32).min
    if np.any(weights[present] < 0):
        raise ValueError("native Wavg weights contain a negative non-sentinel value")
    weight_sum = np.sum(weights[present], dtype=np.float64)
    if not weight_sum > 0:
        raise ValueError("native Wavg weights have zero total mass")
    return np.where(present, weights / np.float32(weight_sum), 0.0).astype(np.float32)


def _float32_ordered_int(values: np.ndarray) -> np.ndarray:
    signed = np.asarray(values, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(0x80000000) - signed, signed)


def _comparison(native: np.ndarray, recovar: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    native_f32 = np.asarray(native, dtype=np.float32)[valid]
    recovar_f32 = np.asarray(recovar, dtype=np.float32)[valid]
    difference = recovar_f32.astype(np.float64) - native_f32.astype(np.float64)
    mismatch = np.flatnonzero(recovar_f32.view(np.uint32) != native_f32.view(np.uint32))
    ulp = np.abs(_float32_ordered_int(recovar_f32) - _float32_ordered_int(native_f32))
    native_norm = np.linalg.norm(native_f32.astype(np.float64))
    return {
        "valid_pixel_count": int(valid.sum()),
        "bit_exact_count": int(valid.sum() - mismatch.size),
        "mismatch_count": int(mismatch.size),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": float(np.linalg.norm(difference) / native_norm) if native_norm else 0.0,
        "max_ulp": int(np.max(ulp, initial=0)),
        "first_mismatch_valid_offset": int(mismatch[0]) if mismatch.size else None,
        "native_sum_float64": float(np.sum(native_f32, dtype=np.float64)),
        "recovar_sum_float64": float(np.sum(recovar_f32, dtype=np.float64)),
    }


def _complex_comparison(
    native: np.ndarray,
    recovar: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    native_complex = np.asarray(native, dtype=np.complex64)[valid]
    recovar_complex = np.asarray(recovar, dtype=np.complex64)[valid]
    difference = recovar_complex.astype(np.complex128) - native_complex.astype(np.complex128)
    native_norm = np.linalg.norm(native_complex.astype(np.complex128))
    return {
        "complex_bit_exact_count": int(
            np.count_nonzero(
                (native_complex.real.view(np.uint32) == recovar_complex.real.view(np.uint32))
                & (native_complex.imag.view(np.uint32) == recovar_complex.imag.view(np.uint32))
            )
        ),
        "relative_l2": float(np.linalg.norm(difference) / native_norm) if native_norm else 0.0,
        "max_complex_abs": float(np.max(np.abs(difference), initial=0.0)),
        "real": _comparison(native_complex.real, recovar_complex.real, np.ones(valid.sum(), bool)),
        "imag": _comparison(native_complex.imag, recovar_complex.imag, np.ones(valid.sum(), bool)),
    }


def _source_by_part(preprocess_dir: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in preprocess_dir.glob("*.preprocess-v1.bin"):
        match = _PREPROCESS_RE.fullmatch(path.name)
        if match:
            result[int(match.group("part"))] = int(match.group("stack")) - 1
    if not result:
        raise ValueError(f"no native preprocess identity files in {preprocess_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-pass2-dir", type=Path, required=True)
    parser.add_argument("--native-wavg-dir", type=Path, required=True)
    parser.add_argument("--native-preprocess-dir", type=Path, required=True)
    parser.add_argument("--source-star", type=Path, required=True)
    parser.add_argument("--relion-bind-dir", type=Path, required=True)
    parser.add_argument(
        "--cuda-lib",
        type=Path,
        help="optional custom CUDA library for the native-preprocessing hybrid",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["RECOVAR_RELION_BIND_BUILD_DIR"] = str(args.relion_bind_dir.resolve())
    if args.cuda_lib is not None:
        os.environ["RECOVAR_CUDA_LIB"] = str(args.cuda_lib.resolve())
    particles, optics_by_id, relion_bind = _load_relion_ctf_inputs(
        args.source_star,
        args.relion_bind_dir,
    )
    preprocess_by_part = {
        artifact.part_id: artifact
        for artifact in (
            load_artifact(path)
            for path in sorted(args.native_preprocess_dir.glob("*.preprocess-v1.bin"))
        )
    }
    source_by_part = {
        part_id: int(artifact.stack_index) - 1
        for part_id, artifact in preprocess_by_part.items()
    }
    if not source_by_part:
        raise ValueError(f"no native preprocess identity files in {args.native_preprocess_dir}")
    records = []
    for native_path in sorted(args.native_wavg_dir.glob("*_storeWavg_wdiff2_pixels.bin")):
        match = _WDIFF_RE.fullmatch(native_path.name)
        if match is None:
            continue
        part_id = int(match.group("part"))
        source_index = source_by_part[part_id]
        prefix = native_path.name.removesuffix("wdiff2_pixels.bin")
        native = {
            name: _load_counted(args.native_wavg_dir / f"{prefix}{suffix}.bin", dtype)
            for name, suffix, dtype in (
                ("wdiff2", "wdiff2_pixels", "<f8"),
                ("aa", "aa_pixels", "<f8"),
                ("xa", "xa_pixels", "<f8"),
                ("mresol", "mresol", "<i4"),
            )
        }
        native_scalar = {
            name: float(_load_counted(args.native_wavg_dir / f"{prefix}{name}.bin", "<f8")[0])
            for name in ("valid_wdiff2_sum", "image_size", "current_size")
        }
        has_native_inputs = (args.native_wavg_dir / f"{prefix}weights.bin").is_file()
        if has_native_inputs:
            native.update(
                {
                    name: _load_counted(args.native_wavg_dir / f"{prefix}{name}.bin", "<f4")
                    for name in (
                        "fimg_real",
                        "fimg_imag",
                        "ctf",
                        "trans_x",
                        "trans_y",
                        "weights",
                        "eulers",
                    )
                }
            )
            native_scalar.update(
                {
                    name: float(
                        _load_counted(args.native_wavg_dir / f"{prefix}{name}.bin", "<f8")[0]
                    )
                    for name in ("orientation_num", "translation_num")
                }
            )
        current_size = int(native_scalar["current_size"])
        pass2_path = args.recovar_pass2_dir / f"pass2_orig{source_index:06d}_cs{current_size:03d}.npz"
        norm_paths = list(
            args.recovar_pass2_dir.glob(
                f"norm_residual_orig{source_index:06d}_half*_cs{current_size:03d}.npz"
            )
        )
        if len(norm_paths) != 1:
            raise ValueError(f"expected one RECOVAR norm capture for source {source_index}")
        with np.load(pass2_path, allow_pickle=False) as pass2:
            window_indices = np.asarray(pass2["window_indices"], dtype=np.int32)
            projections = np.asarray(pass2["proj_half"], dtype=np.complex64)
            probabilities = np.asarray(pass2["reconstruction_probs"], dtype=np.float32)
            fine_translations = np.asarray(pass2["fine_translations"], dtype=np.float32)
            rotations = np.asarray(pass2["rotations"], dtype=np.float32)
        with np.load(norm_paths[0], allow_pickle=False) as norm:
            half = int(norm["half"])
            images = np.asarray(norm["raw_translated_wavg"], dtype=np.complex64)
            captured_indices = np.asarray(norm["wavg_window_indices"], dtype=np.int32)
            processed_score = np.asarray(
                norm["processed_score_half_for_noise"],
                dtype=np.complex64,
            )
        if not np.array_equal(window_indices, captured_indices):
            raise ValueError(f"RECOVAR Wavg windows disagree for source {source_index}")

        particle = particles.iloc[source_index]
        optics_group = int(
            particle["rlnOpticsGroup"]
            if "rlnOpticsGroup" in particle
            else particle["_rlnOpticsGroup"]
        )
        full_ctf = _native_ctf_image(
            particle_row=particle,
            optics_row=optics_by_id[optics_group],
            relion_bind=relion_bind,
            image_size=128,
        ).reshape(-1)
        standard_indices = _standard_half_indices(window_indices, 128)
        ctf = full_ctf[standard_indices].astype(np.float32)
        native_rows = _recovar_rows_in_native_order(
            window_indices,
            current_size=current_size,
            image_size=128,
        )
        # RELION's accelerator Fourier arrays include the 1/N^2 forward-FFT
        # normalization. RECOVAR's diagnostic arrays intentionally retain its
        # unnormalised convention; the power-of-two division is exact in f32.
        fourier_scale = np.float32(1.0 / (128 * 128))
        relion_scaled_projections = np.asarray(projections * fourier_scale, np.complex64)
        relion_scaled_images = np.asarray(images * fourier_scale, np.complex64)
        valid = np.asarray(native["mresol"] >= 0)
        native_preprocess = preprocess_by_part[part_id]
        native_masked_fourier = np.asarray(
            native_preprocess.masked_fourier_post_optics,
            dtype=np.complex64,
        ).reshape(-1)
        if native_masked_fourier.size != native_rows.size:
            raise ValueError(f"native preprocessing/Wavg pixel sizes disagree for part {part_id}")
        candidates = {
            "native_ctf_sign": _wavg_components(
                relion_scaled_projections,
                relion_scaled_images,
                ctf,
                probabilities,
            ),
            "flipped_ctf_sign": _wavg_components(
                relion_scaled_projections,
                relion_scaled_images,
                -ctf,
                probabilities,
            ),
        }
        native_order_candidates: set[str] = set()
        native_input_comparisons = None
        if has_native_inputs:
            from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
                _relion_translation_angles_f32,
            )

            orientation_num = int(native_scalar["orientation_num"])
            translation_num = int(native_scalar["translation_num"])
            if orientation_num != projections.shape[0] or translation_num != fine_translations.shape[0]:
                raise ValueError(f"native/RECOVAR candidate dimensions disagree for part {part_id}")
            native_fimg = np.asarray(
                native["fimg_real"] + np.complex64(1j) * native["fimg_imag"],
                dtype=np.complex64,
            )
            native_probabilities = _normalise_native_weights(
                native["weights"],
                orientation_num,
                translation_num,
            )
            expected_translation_angles = _relion_translation_angles_f32(
                fine_translations,
                (128, 128),
            )
            native_translation_angles = np.stack(
                (native["trans_x"], native["trans_y"]),
                axis=1,
            ).astype(np.float32)
            native_eulers = np.asarray(native["eulers"], dtype=np.float32).reshape(
                orientation_num,
                3,
                3,
            )
            native_input_comparisons = {
                "fimg_vs_preprocess": _complex_comparison(
                    native_masked_fourier,
                    native_fimg,
                    np.ones(native_fimg.size, dtype=bool),
                ),
                "ctf_vs_recovar_native_sign": _comparison(
                    native["ctf"],
                    ctf[native_rows],
                    np.ones(native["ctf"].size, dtype=bool),
                ),
                "ctf_vs_recovar_flipped_sign": _comparison(
                    native["ctf"],
                    -ctf[native_rows],
                    np.ones(native["ctf"].size, dtype=bool),
                ),
                "translation_angles": _comparison(
                    native_translation_angles,
                    expected_translation_angles,
                    np.ones(native_translation_angles.shape, dtype=bool),
                ),
                "weights": _comparison(
                    native_probabilities,
                    probabilities,
                    np.ones(native_probabilities.shape, dtype=bool),
                ),
                "eulers_vs_recovar_transpose": _comparison(
                    native_eulers,
                    np.transpose(rotations, (0, 2, 1)),
                    np.ones(native_eulers.shape, dtype=bool),
                ),
            }
            if args.cuda_lib is not None:
                native_translated_images = _translate_native_wavg_inputs(
                    native_fimg,
                    native_translation_angles,
                    window_indices[native_rows],
                    128,
                )
                candidates["native_inputs_recovar_projection"] = _wavg_components(
                    relion_scaled_projections[:, native_rows],
                    native_translated_images,
                    -native["ctf"],
                    native_probabilities,
                )
                native_order_candidates.add("native_inputs_recovar_projection")
        if args.cuda_lib is not None:
            hybrid_processed_score = _replace_window_with_native_preprocess(
                processed_score,
                window_indices,
                native_rows,
                native_masked_fourier,
                fourier_scale,
            )
            hybrid_images = _translate_native_preprocess_hybrid(
                hybrid_processed_score,
                fine_translations,
                window_indices,
                128,
            )
            candidates["native_preprocess_hybrid_flipped_ctf_sign"] = _wavg_components(
                relion_scaled_projections,
                np.asarray(hybrid_images * fourier_scale, dtype=np.complex64),
                -ctf,
                probabilities,
            )
        recovar_masked_fourier = np.asarray(
            processed_score[window_indices[native_rows]] * fourier_scale,
            dtype=np.complex64,
        )
        if native_input_comparisons is not None:
            native_input_comparisons["fimg_vs_recovar_pretranslation"] = _complex_comparison(
                native_fimg,
                recovar_masked_fourier,
                valid,
            )
        preprocessing_comparison = _complex_comparison(
            native_masked_fourier,
            recovar_masked_fourier,
            valid,
        )
        comparisons = {
            sign: {
                name: _comparison(
                    native[name],
                    (
                        components[name]
                        if sign in native_order_candidates
                        else components[name][native_rows]
                    ),
                    valid,
                )
                for name in ("wdiff2", "aa", "xa")
            }
            for sign, components in candidates.items()
        }
        inferred_native_image_power = np.asarray(
            native["wdiff2"] - native["aa"] + 2.0 * native["xa"],
            dtype=np.float32,
        )
        for sign, components in candidates.items():
            comparisons[sign]["image_power"] = _comparison(
                inferred_native_image_power,
                (
                    components["image_power"]
                    if sign in native_order_candidates
                    else components["image_power"][native_rows]
                ),
                valid,
            )
        best_sign = min(
            comparisons,
            key=lambda sign: float(comparisons[sign]["wdiff2"]["relative_l2"]),
        )
        native_host_sum = float(np.sum(native["wdiff2"][valid], dtype=np.float64))
        records.append(
            {
                "part_id": part_id,
                "source_index": source_index,
                "half": half,
                "current_size": current_size,
                "recovar_to_relion_fourier_operand_scale": float(fourier_scale),
                "valid_pixel_count": int(valid.sum()),
                "best_ctf_sign": best_sign,
                "native_captured_valid_sum": native_scalar["valid_wdiff2_sum"],
                "native_recomputed_valid_sum": native_host_sum,
                "native_sum_exact": native_scalar["valid_wdiff2_sum"] == native_host_sum,
                "masked_fourier_pretranslation": preprocessing_comparison,
                "native_input_comparisons": native_input_comparisons,
                "comparisons": comparisons,
                "native_capture": str(native_path.resolve()),
                "recovar_pass2_capture": str(pass2_path.resolve()),
                "recovar_norm_capture": str(norm_paths[0].resolve()),
            }
        )

    if len(records) != 17:
        raise ValueError(f"expected 17 Wavg panel records, found {len(records)}")
    fields = ("wdiff2", "aa", "xa", "image_power")
    half_summaries = {}
    candidate_summaries = {}
    for half in (1, 2):
        selected_records = [record for record in records if int(record["half"]) == half]
        summary = {
            "masked_fourier_pretranslation": {
                "particle_count": len(selected_records),
                "complex_bit_exact_pixels": sum(
                    int(record["masked_fourier_pretranslation"]["complex_bit_exact_count"])
                    for record in selected_records
                ),
                "valid_pixels": sum(
                    int(record["valid_pixel_count"])
                    for record in selected_records
                ),
                "max_relative_l2": max(
                    float(record["masked_fourier_pretranslation"]["relative_l2"])
                    for record in selected_records
                ),
                "max_component_ulp": max(
                    max(
                        int(record["masked_fourier_pretranslation"][component]["max_ulp"])
                        for component in ("real", "imag")
                    )
                    for record in selected_records
                ),
            },
        }
        summary.update(
            {
                field: {
                    "particle_count": len(selected_records),
                "bit_exact_pixels": sum(
                    int(record["comparisons"][record["best_ctf_sign"]][field]["bit_exact_count"])
                        for record in selected_records
                ),
                "valid_pixels": sum(
                    int(record["comparisons"][record["best_ctf_sign"]][field]["valid_pixel_count"])
                        for record in selected_records
                ),
                "max_relative_l2": max(
                    float(record["comparisons"][record["best_ctf_sign"]][field]["relative_l2"])
                        for record in selected_records
                ),
                "max_ulp": max(
                    int(record["comparisons"][record["best_ctf_sign"]][field]["max_ulp"])
                        for record in selected_records
                ),
                }
                for field in fields
            }
        )
        half_summaries[f"half{half}"] = summary
        candidate_summaries[f"half{half}"] = {
            method: {
                field: {
                    "bit_exact_pixels": sum(
                        int(record["comparisons"][method][field]["bit_exact_count"])
                        for record in selected_records
                    ),
                    "valid_pixels": sum(
                        int(record["comparisons"][method][field]["valid_pixel_count"])
                        for record in selected_records
                    ),
                    "max_relative_l2": max(
                        float(record["comparisons"][method][field]["relative_l2"])
                        for record in selected_records
                    ),
                    "max_ulp": max(
                        int(record["comparisons"][method][field]["max_ulp"])
                        for record in selected_records
                    ),
                }
                for field in fields
            }
            for method in selected_records[0]["comparisons"]
        }
    report = {
        "schema": "recovar.em.k1_native_wavg_pixel_comparison.v1",
        "particle_count": len(records),
        "half_summaries": half_summaries,
        "candidate_summaries": candidate_summaries,
        "particles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(candidate_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
