#!/usr/bin/env python3
"""Audit the live VDAM fine-score image/translation boundary against RELION."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject
from recovar.em.dense_single_volume.local_big_jit import _centered_rfft2_per_image
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_scoring_half_image_weights,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
    _relion_cuda_powerclass_highres_xi2_half,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_native_fine_operands import (
    _center,
    _flat_memmap,
    _stats,
)
from scripts.analyze_vdam_storewavg_boundary import _complex_2d, _real_2d, _scalar
from scripts.compare_relion_recovar_estep_dump import (
    _nearest_rotation_rows_by_matrix,
)
from scripts.validate_relion_fine_operand_capture import load_fine_operand_capture  # noqa: E402
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture  # noqa: E402


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | list[int]]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError(f"metric shape mismatch: {reference.shape} != {candidate.shape}")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = float(np.linalg.norm(reference.astype(np.complex128).reshape(-1)))
    return {
        "shape": list(reference.shape),
        "exact_count": int(np.count_nonzero(candidate == reference)),
        "value_count": int(reference.size),
        "relative_l2": float(np.linalg.norm(residual.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _positive_weight_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    """Separate a common multiplicative scale from positive score weights."""

    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    valid = (reference > 0.0) & (candidate > 0.0)
    if reference.shape != candidate.shape or not np.any(valid):
        raise ValueError("positive weight metric requires aligned positive values")
    ref = reference[valid]
    cand = candidate[valid]
    ratio = cand / ref
    scale = float(np.dot(ref, cand) / np.dot(ref, ref))
    scaled_residual = cand - scale * ref
    denominator = max(float(np.linalg.norm(cand)), np.finfo(np.float64).tiny)
    return {
        "positive_count": int(ref.size),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_std": float(np.std(ratio)),
        "least_squares_scale": scale,
        "relative_l2_after_common_scale": float(np.linalg.norm(scaled_residual) / denominator),
    }


def _centered_diff2_replay_stats(
    native_raw: np.ndarray,
    replay_raw: np.ndarray,
) -> dict[str, float | int]:
    """Compare positive diff2 arrays while factoring out native high-res Xi2."""

    native_f32 = np.asarray(native_raw, dtype=np.float32)
    replay_f32 = np.asarray(replay_raw, dtype=np.float32)
    result = _stats(_center(replay_f32 - native_f32))
    inferred_highres = np.subtract(native_f32, replay_f32, dtype=np.float32)
    inferred_bits, inferred_counts = np.unique(inferred_highres.view(np.uint32), return_counts=True)
    inferred_mode_index = int(np.argmax(inferred_counts))
    result.update(
        {
            "raw_exact_count": int(np.count_nonzero(replay_f32 == native_f32)),
            "inferred_highres_mode": float(inferred_bits[inferred_mode_index].view(np.float32)),
            "inferred_highres_mode_count": int(inferred_counts[inferred_mode_index]),
            "inferred_highres_unique_count": int(inferred_bits.size),
        }
    )
    return result


def _diff2_replay_boundary(
    native_raw_diff2: np.ndarray,
    replay_diff2: np.ndarray,
    native_posterior: np.ndarray,
) -> dict[str, float | int]:
    """Report centered and native-top-pair errors for one diff2 replay."""

    native_raw = np.asarray(native_raw_diff2, dtype=np.float32).reshape(-1)
    replay_raw = np.asarray(replay_diff2, dtype=np.float32).reshape(-1)
    native_prob = np.asarray(native_posterior, dtype=np.float64).reshape(-1)
    if native_raw.size < 2 or replay_raw.shape != native_raw.shape:
        raise ValueError("diff2 replay boundary requires at least two aligned scores")
    if native_prob.shape != native_raw.shape or not np.all(np.isfinite(native_prob)):
        raise ValueError("diff2 replay boundary requires aligned finite probabilities")
    first, second = np.argsort(-native_prob, kind="stable")[:2]
    native_delta = float(native_raw[first] - native_raw[second])
    replay_delta = float(replay_raw[first] - replay_raw[second])
    result = _centered_diff2_replay_stats(native_raw, replay_raw)
    result.update(
        {
            "native_best_row": int(first),
            "native_second_row": int(second),
            "native_best_minus_second": native_delta,
            "replay_best_minus_second": replay_delta,
            "replay_minus_native_pair_delta": replay_delta - native_delta,
        }
    )
    return result


def _top_pair_score_boundary(
    *,
    native_raw_diff2: np.ndarray,
    recovar_raw_score: np.ndarray,
    native_posterior: np.ndarray,
    recovar_posterior: np.ndarray,
    mapped_rotations: np.ndarray,
    translation_ids: np.ndarray,
) -> dict[str, object]:
    """Report the native top-two odds in one common candidate order."""

    arrays = [
        np.asarray(native_raw_diff2, dtype=np.float64).reshape(-1),
        np.asarray(recovar_raw_score, dtype=np.float64).reshape(-1),
        np.asarray(native_posterior, dtype=np.float64).reshape(-1),
        np.asarray(recovar_posterior, dtype=np.float64).reshape(-1),
        np.asarray(mapped_rotations, dtype=np.int64).reshape(-1),
        np.asarray(translation_ids, dtype=np.int64).reshape(-1),
    ]
    if arrays[0].size < 2 or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("top-pair score boundary requires at least two aligned candidates")
    native_raw, recovar_raw, native_prob, recovar_prob, rotations, translations = arrays
    if not np.all(np.isfinite(native_raw)) or not np.all(np.isfinite(recovar_raw)):
        raise ValueError("top-pair raw scores must be finite")
    if (
        not np.all(np.isfinite(native_prob))
        or not np.all(np.isfinite(recovar_prob))
        or np.any(native_prob < 0.0)
        or np.any(recovar_prob < 0.0)
    ):
        raise ValueError("top-pair posterior comparison requires finite nonnegative probabilities")

    first, second = np.argsort(-native_prob, kind="stable")[:2]
    top_probabilities = (
        native_prob[first],
        native_prob[second],
        recovar_prob[first],
        recovar_prob[second],
    )
    if min(top_probabilities) <= 0.0:
        raise ValueError("top-pair posterior probabilities must be positive in both engines")

    def _row(index: int) -> dict[str, object]:
        return {
            "native_candidate_row": int(index),
            "mapped_key": [int(rotations[index]), int(translations[index])],
            "native_probability": float(native_prob[index]),
            "recovar_probability": float(recovar_prob[index]),
            "native_raw_diff2": float(native_raw[index]),
            "recovar_raw_score": float(recovar_raw[index]),
            "raw_sign_convention_residual": float(recovar_raw[index] + native_raw[index]),
        }

    native_log_odds = float(np.log(native_prob[first]) - np.log(native_prob[second]))
    recovar_log_odds = float(np.log(recovar_prob[first]) - np.log(recovar_prob[second]))
    native_raw_delta = float(native_raw[first] - native_raw[second])
    recovar_raw_delta = float(recovar_raw[first] - recovar_raw[second])
    return {
        "native_best": _row(int(first)),
        "native_second": _row(int(second)),
        "native_log_odds_best_over_second": native_log_odds,
        "recovar_log_odds_same_order": recovar_log_odds,
        "recovar_minus_native_log_odds": recovar_log_odds - native_log_odds,
        "native_raw_diff2_best_minus_second": native_raw_delta,
        "recovar_raw_score_best_minus_second": recovar_raw_delta,
        "raw_pair_delta_sign_convention_residual": recovar_raw_delta + native_raw_delta,
    }


def _flat_real_dump(path: Path) -> np.ndarray:
    """Read RELION's count-prefixed ``__relion_acc_dump_flat_real`` format."""

    path = Path(path)
    with path.open("rb") as stream:
        count = np.fromfile(stream, dtype=np.int32, count=1)
        if count.size != 1 or int(count[0]) < 0:
            raise ValueError(f"invalid RELION flat-real header: {path}")
        values = np.fromfile(stream, dtype=np.float64, count=int(count[0]))
        trailing = stream.read(1)
    if values.size != int(count[0]) or trailing:
        raise ValueError(f"invalid RELION flat-real payload: {path}")
    return values


def _captured_native_current_size(native_dir: Path) -> int:
    """Return the fine-pass image size sealed by RELION's verbose capture."""

    value = _scalar(Path(native_dir) / "pass1_img0_exp_current_image_size.bin")
    rounded = int(round(value))
    if rounded <= 0 or float(rounded) != value:
        raise ValueError(f"invalid native fine current size: {value}")
    return rounded


def _preprocess_capture(
    preprocess_dir: Path,
    *,
    full_size: int,
    current_size: int,
) -> dict[str, np.ndarray | float]:
    prefix = Path(preprocess_dir) / "preprocess_img0_"
    normalized = _flat_real_dump(Path(f"{prefix}normalized_shifted_real.bin"))
    masked = _flat_real_dump(Path(f"{prefix}masked_real.bin"))
    fourier_real = _flat_real_dump(Path(f"{prefix}masked_fourier_pre_optics_real.bin"))
    fourier_imag = _flat_real_dump(Path(f"{prefix}masked_fourier_pre_optics_imag.bin"))
    post_optics_real = _flat_real_dump(Path(f"{prefix}masked_fourier_post_optics_real.bin"))
    post_optics_imag = _flat_real_dump(Path(f"{prefix}masked_fourier_post_optics_imag.bin"))
    expected_real = full_size * full_size
    expected_fourier = current_size * (current_size // 2 + 1)
    if normalized.size != expected_real or masked.size != expected_real:
        raise ValueError("RELION preprocessing real-space capture has the wrong image size")
    if any(
        values.size != expected_fourier
        for values in (fourier_real, fourier_imag, post_optics_real, post_optics_imag)
    ):
        raise ValueError("RELION preprocessing Fourier capture has the wrong image size")
    background_path = Path(f"{prefix}softmask_background.bin")
    background = np.fromfile(background_path, dtype=np.float64)
    if background.shape != (1,):
        raise ValueError(f"invalid RELION scalar payload: {background_path}")
    return {
        "normalized_shifted": normalized.astype(np.float32).reshape(full_size, full_size),
        "masked": masked.astype(np.float32).reshape(full_size, full_size),
        "masked_fourier_pre_optics": (
            fourier_real.astype(np.float32) + np.complex64(1j) * fourier_imag.astype(np.float32)
        ).astype(np.complex64),
        "masked_fourier_post_optics": (
            post_optics_real.astype(np.float32)
            + np.complex64(1j) * post_optics_imag.astype(np.float32)
        ).astype(np.complex64),
        "background": float(np.float32(background[0])),
    }


def _native_current_fft_rows(*, full_size: int, current_size: int) -> np.ndarray:
    """Map RELION's standard current-size FFTW grid into a centered full FFT."""

    logical_y = np.where(
        np.arange(current_size) <= current_size // 2,
        np.arange(current_size),
        np.arange(current_size) - current_size,
    )
    return (
        (logical_y[:, None] + full_size // 2) * (full_size // 2 + 1)
        + np.arange(current_size // 2 + 1)[None, :]
    ).astype(np.int32).reshape(-1)


def _native_crop_rows(score_indices: np.ndarray, full_size: int, current_size: int) -> np.ndarray:
    score_indices = np.asarray(score_indices, dtype=np.int64)
    half_width = full_size // 2 + 1
    centered_row = score_indices // half_width
    x = score_indices - centered_row * half_width
    ky = centered_row - full_size // 2
    native_row = np.where(ky >= 0, ky, ky + current_size)
    crop = native_row * (current_size // 2 + 1) + x
    if np.unique(crop).size != crop.size:
        raise ValueError("score window does not map bijectively into the native crop")
    return crop.astype(np.int32)


def _current_crop_to_compact(crop: np.ndarray, current_size: int) -> np.ndarray:
    """Map RELION's full current-size FFTW grid to the radial compact rows."""

    crop = np.asarray(crop, dtype=np.int64)
    current_pixel_count = current_size * (current_size // 2 + 1)
    if np.any(crop < 0) or np.any(crop >= current_pixel_count):
        raise ValueError("native crop index lies outside the current-size Fourier grid")
    lookup = np.full(current_pixel_count, -1, dtype=np.int32)
    lookup[crop] = np.arange(crop.size, dtype=np.int32)
    return lookup


def analyze(
    native_dir: Path,
    live_score_path: Path,
    *,
    full_size: int,
    particle_stack: Path | None = None,
    particle_index: int | None = None,
    normalization_factor: float = 1.0,
    integer_pre_shift: tuple[int, int] = (0, 0),
    mask_radius: float | None = None,
    mask_cosine_width: float = 5.0,
    native_preprocess_dir: Path | None = None,
    native_fine_score_path: Path | None = None,
    native_fine_operand_path: Path | None = None,
) -> dict[str, object]:
    native_dir = Path(native_dir)
    with np.load(live_score_path, allow_pickle=False) as payload:
        live = {name: np.array(payload[name]) for name in payload.files}

    current_size = int(np.asarray(live["current_size"]).reshape(-1)[0])
    native_current_size = _captured_native_current_size(native_dir)
    if native_current_size != current_size:
        return {
            "schema": "recovar.vdam_native_translation_boundary.v1",
            "status": "current_size_mismatch",
            "device": str(jax.devices()[0]),
            "identity": {
                "recovar_current_size": current_size,
                "native_current_size": native_current_size,
            },
            "comparisons": {},
            "artifacts": {
                "native_directory": str(native_dir.resolve()),
                "live_score_dump": str(Path(live_score_path).resolve()),
                "native_preprocess_directory": (
                    str(Path(native_preprocess_dir).resolve())
                    if native_preprocess_dir is not None
                    else None
                ),
            },
        }
    score_indices, _ = make_fourier_window_indices_np(
        (full_size, full_size),
        current_size,
        square=False,
        include_dc=False,
    )
    if score_indices.size != live["debug_ctf2_over_nv"].shape[-1]:
        raise ValueError("live score dump has a different radial window")
    crop = _native_crop_rows(score_indices, full_size, current_size)
    native_pixel_count = current_size * (current_size // 2 + 1)

    native_rotation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int32,
    )
    translation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_eulers = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_fine_eulers.bin")
    ).reshape(-1, 3, 3)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        native_eulers,
        live["local_rotation_matrices"],
    )
    rotation = nearest[native_rotation]
    native_raw = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float64,
    )
    native_fine_score = None
    if native_fine_score_path is not None:
        native_fine_score = load_fine_score_capture(native_fine_score_path)
        selected = native_fine_score.candidates[
            (native_fine_score.candidates["flags"] & ACTIVE) != 0
        ]
        if selected.size != native_rotation.size:
            raise ValueError("formal fine-score candidate count differs from verbose capture")
        if not np.array_equal(
            selected["rotation_local"], native_rotation.astype(np.uint64)
        ):
            raise ValueError("formal fine-score rotation order differs from verbose capture")
        if not np.array_equal(
            selected["translation_id"], translation.astype(np.uint64)
        ):
            raise ValueError("formal fine-score translation order differs from verbose capture")
        native_raw = np.asarray(selected["raw_diff2"], dtype=np.float64)
    live_raw = np.asarray(live["pass2_scores_raw"])[0, rotation, translation]
    native_posterior = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_posterior.bin"),
        dtype=np.float64,
    ) / float(_scalar(native_dir / "pass1_exp_sum_weight.bin"))
    live_posterior = np.asarray(live["posterior"], dtype=np.float64)[
        0, rotation, translation
    ]

    native_fine_operand = None
    captured_highres_xi2_half = None
    if native_fine_operand_path is not None:
        native_fine_operand = load_fine_operand_capture(native_fine_operand_path)
        if (
            native_fine_score is not None
            and native_fine_operand.stack_index != native_fine_score.stack_index
        ):
            raise ValueError("formal fine-score and fine-operand particle identities differ")
        captured_sum_init = np.asarray(
            native_fine_operand.candidates["sum_init"], dtype=np.float32
        )
        if captured_sum_init.size == 0 or not np.all(captured_sum_init == captured_sum_init[0]):
            raise ValueError("formal fine operand does not seal one high-resolution accumulator")
        captured_highres_xi2_half = captured_sum_init[0]

    scale = np.float32(full_size**2)
    native_weight_full = np.asarray(
        _flat_memmap(native_dir / "pass1_img0_corr_img.bin"),
        dtype=np.float32,
    ) / np.float32(full_size**4)
    native_weight = native_weight_full[crop]
    live_weight = np.asarray(live["debug_ctf2_over_nv"], dtype=np.float32)

    candidate_count = native_raw.size
    shape = (candidate_count, native_pixel_count)
    native_reference = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_ref_real.bin"),
            dtype=np.float32,
        ).reshape(shape)
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_ref_imag.bin"),
            dtype=np.float32,
        ).reshape(shape)
    )
    native_shifted = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_shifted_real.bin"),
            dtype=np.float32,
        ).reshape(shape)
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_shifted_imag.bin"),
            dtype=np.float32,
        ).reshape(shape)
    )
    native_reference = native_reference[:, crop]
    native_shifted = native_shifted[:, crop]
    live_reference = np.asarray(live["debug_proj_weighted"], dtype=np.complex64)[rotation]
    live_shifted_weighted = np.asarray(live["debug_shifted_score"], dtype=np.complex64)[translation]

    base_corrected = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_img0_Fimg_corrected_real.bin"),
            dtype=np.float32,
        )[crop]
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_img0_Fimg_corrected_imag.bin"),
            dtype=np.float32,
        )[crop]
    )
    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(live["translations"], (full_size, full_size)),
        dtype=jnp.float32,
    )
    translated_unweighted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(base_corrected[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(score_indices, dtype=jnp.int32),
        (full_size, full_size),
    )
    translated_preweighted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray((base_corrected * native_weight)[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(score_indices, dtype=jnp.int32),
        (full_size, full_size),
    )
    translated_unweighted, translated_preweighted = (
        np.asarray(value, dtype=np.complex64)
        for value in jax.block_until_ready(
            (translated_unweighted, translated_preweighted)
        )
    )

    # Replay the production fine-score reduction twice from the same native
    # operands.  The pair replay consumes RELION's already-translated image;
    # the fused replay starts from the unshifted corrected image and performs
    # translation inside RECOVAR's production score kernel.  Their centered
    # residuals separate translation arithmetic from the reduction tree while
    # ignoring the candidate-independent high-resolution Xi2 addend.
    score_half_weights = np.asarray(
        make_scoring_half_image_weights(
            (full_size, full_size),
            relion_half_sum=True,
        ),
        dtype=np.float32,
    )[score_indices]
    direct_reference = np.where(
        score_half_weights[None] > 0.0,
        native_reference,
        np.complex64(0.0),
    ).astype(np.complex64)
    direct_weight = np.multiply(
        native_weight,
        score_half_weights,
        dtype=np.float32,
    )
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (full_size, full_size),
        current_size,
        score_indices,
    )
    native_shifted_pair_diff2 = cuda_backproject.relion_fine_diff2_pairs_f32(
        jnp.asarray(direct_reference[None], dtype=jnp.complex64),
        jnp.asarray(native_shifted[None], dtype=jnp.complex64),
        jnp.asarray(direct_weight[None], dtype=jnp.float32),
        jnp.asarray(full_to_compact, dtype=jnp.int32),
    )[0]
    fused_translate_diff2 = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
        jnp.asarray(direct_reference[None], dtype=jnp.complex64),
        jnp.asarray(base_corrected[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(direct_weight[None], dtype=jnp.float32),
        jnp.asarray(full_to_compact, dtype=jnp.int32),
        current_size=current_size,
    )[0]
    native_shifted_pair_diff2, fused_translate_diff2 = (
        np.asarray(value, dtype=np.float32)
        for value in jax.block_until_ready(
            (native_shifted_pair_diff2, fused_translate_diff2)
        )
    )
    fused_translate_selected = fused_translate_diff2[
        np.arange(candidate_count, dtype=np.int64),
        translation,
    ]

    # Reconstruct the exact-path live operands from the generic debug values.
    # ``debug_shifted_score`` is the translated corrected image multiplied by
    # ctf^2/noise, while the exact fine kernel receives those two factors
    # separately.  Rows excluded from RELION's half-plane have zero direct
    # weight, so their arbitrary quotient is immaterial.
    live_shifted_unweighted = np.zeros_like(live_shifted_weighted)
    np.divide(
        live_shifted_weighted,
        live_weight[None],
        out=live_shifted_unweighted,
        where=live_weight[None] != 0.0,
    )
    live_direct_weight = np.multiply(
        live_weight,
        score_half_weights,
        dtype=np.float32,
    )

    def _pair_diff2(reference, shifted_image, weight):
        value = cuda_backproject.relion_fine_diff2_pairs_f32(
            jnp.asarray(reference[None], dtype=jnp.complex64),
            jnp.asarray(shifted_image[None], dtype=jnp.complex64),
            jnp.asarray(weight[None], dtype=jnp.float32),
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )[0]
        return np.asarray(jax.block_until_ready(value), dtype=np.float32)

    live_operand_diff2 = _pair_diff2(
        live_reference,
        live_shifted_unweighted,
        live_direct_weight,
    )
    live_reference_only_diff2 = _pair_diff2(
        live_reference,
        native_shifted,
        direct_weight,
    )
    live_image_only_diff2 = _pair_diff2(
        direct_reference,
        live_shifted_unweighted,
        direct_weight,
    )
    live_weight_only_diff2 = _pair_diff2(
        direct_reference,
        native_shifted,
        live_direct_weight,
    )

    expected_weighted = (native_shifted * native_weight[None]).astype(np.complex64)
    comparisons = {
        "top_pair_score_boundary": _top_pair_score_boundary(
            native_raw_diff2=native_raw,
            recovar_raw_score=live_raw,
            native_posterior=native_posterior,
            recovar_posterior=live_posterior,
            mapped_rotations=rotation,
            translation_ids=translation,
        ),
        "live_centered_raw_score_residual": _stats(_center(live_raw + native_raw)),
        "native_raw_diff2_vs_native_shifted_pair_replay": _centered_diff2_replay_stats(
            native_raw,
            native_shifted_pair_diff2,
        ),
        "native_raw_diff2_vs_fused_translate_replay": _centered_diff2_replay_stats(
            native_raw,
            fused_translate_selected,
        ),
        "native_shifted_pair_vs_fused_translate_replay": _stats(
            _center(fused_translate_selected - native_shifted_pair_diff2)
        ),
        "live_raw_score_vs_live_operand_pair_replay": _diff2_replay_boundary(
            -live_raw,
            live_operand_diff2,
            native_posterior,
        ),
        "native_raw_diff2_with_live_reference_only": _diff2_replay_boundary(
            native_raw,
            live_reference_only_diff2,
            native_posterior,
        ),
        "native_raw_diff2_with_live_image_only": _diff2_replay_boundary(
            native_raw,
            live_image_only_diff2,
            native_posterior,
        ),
        "native_raw_diff2_with_live_weight_only": _diff2_replay_boundary(
            native_raw,
            live_weight_only_diff2,
            native_posterior,
        ),
        "live_projected_reference": _metric(native_reference, live_reference),
        "live_score_weight": _metric(native_weight, live_weight),
        "live_score_weight_scale": _positive_weight_metric(native_weight, live_weight),
        "live_weighted_shifted_image": _metric(expected_weighted, live_shifted_weighted),
        "native_base_then_translate": _metric(
            native_shifted,
            translated_unweighted[translation],
        ),
        "native_base_preweighted_then_translate": _metric(
            expected_weighted,
            translated_preweighted[translation],
        ),
        "native_translate_then_weight_vs_live": _metric(
            expected_weighted,
            live_shifted_weighted,
        ),
        "native_preweight_then_translate_vs_live": _metric(
            translated_preweighted[translation],
            live_shifted_weighted,
        ),
    }
    storewavg_paths = {
        "image": native_dir / "Fimg_unweighted.bin",
        "ctf": native_dir / "Fctf.bin",
        "inverse_noise": native_dir / "Minvsigma2.bin",
    }
    storewavg_available = all(path.is_file() for path in storewavg_paths.values())
    native_processed = None
    if storewavg_available:
        native_processed = (
            -scale
            * np.asarray(_complex_2d(storewavg_paths["image"]), dtype=np.complex64)
            .reshape(-1)[crop]
        ).astype(np.complex64)
        native_ctf = np.asarray(
            _real_2d(storewavg_paths["ctf"]), dtype=np.float64
        ).astype(np.float32).reshape(-1)[crop]
        native_inverse_noise_engine = (
            np.asarray(
                _real_2d(storewavg_paths["inverse_noise"]),
                dtype=np.float64,
            ).reshape(-1)[crop]
            / float(full_size**4)
        ).astype(np.float32)
        native_production_weighted = (
            native_processed * (native_ctf * native_inverse_noise_engine).astype(np.float32)
        ).astype(np.complex64)
        translated_native_production = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(native_production_weighted[None], dtype=jnp.complex64),
            translation_angles,
            jnp.asarray(score_indices, dtype=jnp.int32),
            (full_size, full_size),
        )
        translated_native_production = np.asarray(
            jax.block_until_ready(translated_native_production),
            dtype=np.complex64,
        )
        comparisons["native_production_product_then_translate_vs_live"] = _metric(
            translated_native_production[translation],
            live_shifted_weighted,
        )
        comparisons[
            "native_corrected_weight_product_vs_native_production_product"
        ] = _metric(
            (base_corrected * native_weight).astype(np.complex64),
            native_production_weighted,
        )
    if particle_stack is not None:
        if native_processed is None and native_preprocess_dir is None:
            raise ValueError(
                "particle preprocessing replay requires either a verbose preprocessing "
                "capture or Fimg_unweighted.bin, Fctf.bin, and Minvsigma2.bin"
            )
        if particle_index is None or mask_radius is None:
            raise ValueError("particle preprocessing replay requires particle_index and mask_radius")
        with mrcfile.open(particle_stack, permissive=False) as stack_file:
            raw_image = np.asarray(stack_file.data[int(particle_index)], dtype=np.float32)
        captured_preprocess = (
            _preprocess_capture(
                native_preprocess_dir,
                full_size=full_size,
                current_size=current_size,
            )
            if native_preprocess_dir is not None
            else None
        )
        current_fft_rows = _native_current_fft_rows(
            full_size=full_size,
            current_size=current_size,
        )
        for mode, native_lane_reduction, native_atomic_reduction in (
            ("block_first", False, False),
            ("native_lane", True, False),
            ("native_atomic", False, True),
        ):
            normalized_real, masked_real = cuda_backproject.relion_preprocess_real_f32(
                jnp.asarray(raw_image[None], dtype=jnp.float32),
                jnp.asarray([normalization_factor], dtype=jnp.float32),
                jnp.asarray([integer_pre_shift], dtype=jnp.int32),
                float(mask_radius),
                float(mask_cosine_width),
                apply_mask=True,
                native_lane_reduction=native_lane_reduction,
                native_atomic_reduction=native_atomic_reduction,
            )
            replay_half = _centered_rfft2_per_image(masked_real)
            normalized_real, masked_real, replay_half = (
                np.asarray(value)
                for value in jax.block_until_ready((normalized_real, masked_real, replay_half))
            )
            normalized_real = np.asarray(normalized_real, dtype=np.float32)[0]
            masked_real = np.asarray(masked_real, dtype=np.float32)[0]
            replay_half = np.asarray(replay_half, dtype=np.complex64)[0]
            if native_processed is not None:
                comparisons[f"native_masked_fourier_vs_{mode}_preprocess_replay"] = _metric(
                    -native_processed,
                    replay_half[score_indices],
                )
            if captured_preprocess is not None:
                comparisons[f"native_normalized_real_vs_{mode}_preprocess_replay"] = _metric(
                    captured_preprocess["normalized_shifted"],
                    normalized_real,
                )
                comparisons[f"native_masked_real_vs_{mode}_preprocess_replay"] = _metric(
                    captured_preprocess["masked"],
                    masked_real,
                )
                native_background = np.float32(captured_preprocess["background"])
                replay_background = np.float32(masked_real[0, 0])
                comparisons[f"native_background_vs_{mode}_preprocess_replay"] = {
                    "native": float(native_background),
                    "replay": float(replay_background),
                    "difference": float(np.float32(replay_background - native_background)),
                    "exact": bool(replay_background == native_background),
                    "native_bits": int(native_background.view(np.uint32)),
                    "replay_bits": int(replay_background.view(np.uint32)),
                }
                native_preoptics = np.asarray(
                    captured_preprocess["masked_fourier_pre_optics"], dtype=np.complex64
                )
                native_postoptics = np.asarray(
                    captured_preprocess["masked_fourier_post_optics"], dtype=np.complex64
                )
                comparisons["native_preoptics_vs_postoptics_fourier"] = _metric(
                    native_preoptics,
                    native_postoptics,
                )
                comparisons[f"native_preoptics_fourier_vs_{mode}_preprocess_replay"] = _metric(
                    native_preoptics,
                    (replay_half.reshape(-1)[current_fft_rows] / scale).astype(np.complex64),
                )
                native_masked_fft = _centered_rfft2_per_image(
                    jnp.asarray(captured_preprocess["masked"][None], dtype=jnp.float32)
                )
                native_masked_fft = np.asarray(
                    jax.block_until_ready(native_masked_fft), dtype=np.complex64
                )[0]
                comparisons[f"native_preoptics_fourier_vs_native_real_recovar_fft_{mode}"] = _metric(
                    native_preoptics,
                    (
                        native_masked_fft.reshape(-1)[current_fft_rows] / scale
                    ).astype(np.complex64),
                )
            if not (native_dir / "Fctf.bin").is_file():
                continue
            pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                jnp.asarray([1.0], dtype=jnp.float32),
                jnp.asarray(
                    -np.asarray(_real_2d(native_dir / "Fctf.bin"), dtype=np.float64)
                    .reshape(-1)[crop][None],
                    dtype=jnp.float64,
                ),
            )
            corrected = (
                jnp.asarray(replay_half[score_indices][None], dtype=jnp.complex64)
                * pixel_correction
            )
            translated_corrected = cuda_backproject.relion_translate_score_f32(
                corrected,
                translation_angles,
                jnp.asarray(score_indices, dtype=jnp.int32),
                (full_size, full_size),
            )
            lookup = _relion_cuda_fine_full_to_compact_lookup(
                (full_size, full_size),
                current_size,
                score_indices,
            )
            highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
                jnp.asarray(replay_half[None], dtype=jnp.complex64),
                image_shape=(full_size, full_size),
                current_size=current_size,
            )[0]
            production_highres_xi2_half = (
                np.float32(highres_xi2_half)
                if captured_highres_xi2_half is None
                else np.float32(captured_highres_xi2_half)
            )
            lookup_variants = {
                "physical_grid": lookup,
                "current_grid": _current_crop_to_compact(crop, current_size),
                "radial_compact": np.arange(score_indices.size, dtype=np.int32),
            }
            for lookup_name, lookup_variant in lookup_variants.items():
                direct_diff2 = cuda_backproject.relion_fine_diff2_rectangular_f32(
                    jnp.asarray(live["debug_proj_weighted"][None], dtype=jnp.complex64),
                    translated_corrected[None],
                    jnp.asarray(live_weight[None], dtype=jnp.float32),
                    jnp.asarray(lookup_variant, dtype=jnp.int32),
                )[0]
                direct_lowres = np.asarray(
                    jax.block_until_ready(direct_diff2), dtype=np.float32
                )[rotation, translation]
                inferred_highres = np.subtract(
                    native_raw.astype(np.float32), direct_lowres, dtype=np.float32
                )
                inferred_bits, inferred_counts = np.unique(
                    inferred_highres.view(np.uint32), return_counts=True
                )
                inferred_mode_index = int(np.argmax(inferred_counts))
                inferred_mode = inferred_bits[inferred_mode_index].view(np.float32)
                direct_cost = np.add(
                    direct_lowres, production_highres_xi2_half, dtype=np.float32
                )
                centered_direct_residual = _stats(_center(direct_cost - native_raw))
                centered_direct_residual.update(
                    {
                        "raw_exact_count": int(
                            np.count_nonzero(direct_cost == native_raw.astype(np.float32))
                        ),
                        "highres_xi2_half": float(highres_xi2_half),
                        "production_highres_xi2_half": float(
                            production_highres_xi2_half
                        ),
                        "inferred_highres_mode": float(inferred_mode),
                        "inferred_highres_mode_count": int(
                            inferred_counts[inferred_mode_index]
                        ),
                        "inferred_highres_unique_count": int(inferred_bits.size),
                    }
                )
                comparisons[
                    f"native_raw_diff2_vs_{mode}_{lookup_name}_direct_replay"
                ] = centered_direct_residual

    return {
        "schema": "recovar.vdam_native_translation_boundary.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "identity": {
            "current_size": current_size,
            "candidate_count": candidate_count,
            "score_pixel_count": int(score_indices.size),
            "rotation_matrix_orientation": orientation,
            "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
            "storewavg_operands_available": storewavg_available,
        },
        "comparisons": comparisons,
        "artifacts": {
            "native_directory": str(native_dir.resolve()),
            "live_score_dump": str(Path(live_score_path).resolve()),
            "native_preprocess_directory": (
                str(Path(native_preprocess_dir).resolve())
                if native_preprocess_dir is not None
                else None
            ),
            "native_fine_score": (
                str(Path(native_fine_score_path).resolve())
                if native_fine_score_path is not None
                else None
            ),
            "native_fine_operand": (
                str(Path(native_fine_operand_path).resolve())
                if native_fine_operand_path is not None
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--live-score", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--particle-stack", type=Path)
    parser.add_argument("--particle-index", type=int)
    parser.add_argument("--normalization-factor", type=float, default=1.0)
    parser.add_argument("--integer-pre-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--mask-radius", type=float)
    parser.add_argument("--mask-cosine-width", type=float, default=5.0)
    parser.add_argument("--native-preprocess-directory", type=Path)
    parser.add_argument("--native-fine-score", type=Path)
    parser.add_argument("--native-fine-operand", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.live_score,
        full_size=args.full_image_size,
        particle_stack=args.particle_stack,
        particle_index=args.particle_index,
        normalization_factor=args.normalization_factor,
        integer_pre_shift=tuple(args.integer_pre_shift),
        mask_radius=args.mask_radius,
        mask_cosine_width=args.mask_cosine_width,
        native_preprocess_dir=args.native_preprocess_directory,
        native_fine_score_path=args.native_fine_score,
        native_fine_operand_path=args.native_fine_operand,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
