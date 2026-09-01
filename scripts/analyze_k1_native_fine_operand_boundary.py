#!/usr/bin/env python3
"""Join native RELION verbose fine operands to one RECOVAR K=1 capture.

The comparison is identity- and candidate-aligned.  It reports the first
unequal native boundary among corr_img, projected references, translated
images, source-like 256-lane Gaussian reductions, and centered raw fine scores.
No map surrogate or correlation metric is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_k1_fine_score_boundary import _rotation_map, _translation_map
    from scripts.validate_relion_bpref_factor_capture import load_factor_capture
    from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from analyze_k1_fine_score_boundary import _rotation_map, _translation_map
    from validate_relion_bpref_factor_capture import load_factor_capture
    from validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


REPORT_SCHEMA = "recovar.em.k1_native_fine_operand_boundary.v1"
BLOCK_SIZE = 256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    _require(len(matches) == 1, f"expected one {pattern!r} file, found {len(matches)}")
    return matches[0]


def _load_flat_real(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count_raw = stream.read(4)
        _require(len(count_raw) == 4, f"truncated flat-real header: {path}")
        count = int(np.frombuffer(count_raw, dtype="<i4", count=1)[0])
        _require(count >= 0, f"negative flat-real count in {path}")
        payload = stream.read()
    _require(len(payload) == count * 8, f"flat-real payload size mismatch in {path}")
    return np.frombuffer(payload, dtype="<f8", count=count).copy()


def _load_flat_int(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count_raw = stream.read(4)
        _require(len(count_raw) == 4, f"truncated flat-int header: {path}")
        count = int(np.frombuffer(count_raw, dtype="<i4", count=1)[0])
        _require(count >= 0, f"negative flat-int count in {path}")
        payload = stream.read()
    _require(len(payload) == count * 4, f"flat-int payload size mismatch in {path}")
    return np.frombuffer(payload, dtype="<i4", count=count).copy()


def _load_flat_complex(real_path: Path, imag_path: Path) -> np.ndarray:
    real = _load_flat_real(real_path).astype(np.float32)
    imag = _load_flat_real(imag_path).astype(np.float32)
    _require(real.shape == imag.shape, "complex operand components are misaligned")
    return (real + np.complex64(1j) * imag).astype(np.complex64)


def _load_scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar payload size mismatch in {path}")
    return float(np.frombuffer(payload, dtype="<f8", count=1)[0])


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    # Boolean panel selection commonly produces strided arrays.  Normalize
    # layout before byte views so exact-bit telemetry works for every slice.
    ref = np.ascontiguousarray(reference)
    cand = np.ascontiguousarray(candidate)
    shape_equal = ref.shape == cand.shape
    result: dict[str, Any] = {
        "shape_equal": bool(shape_equal),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
    }
    if not shape_equal:
        return result
    exact = ref.view(np.uint8).reshape(-1) == cand.view(np.uint8).reshape(-1)
    ref64 = ref.astype(np.float64, copy=False)
    cand64 = cand.astype(np.float64, copy=False)
    delta = cand64 - ref64
    denominator = float(np.linalg.norm(ref64.reshape(-1)))
    result.update(
        {
            "bit_equal_fraction": float(np.mean(exact)) if exact.size else 1.0,
            "value_mismatch_count": int(np.count_nonzero(ref != cand)),
            "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "relative_l2_over_reference": (
                float(np.linalg.norm(delta.reshape(-1)) / denominator)
                if denominator > 0.0
                else float(np.linalg.norm(delta.reshape(-1)))
            ),
        }
    )
    mismatch = np.flatnonzero(ref.reshape(-1) != cand.reshape(-1))
    if mismatch.size:
        flat = int(mismatch[0])
        index = np.unravel_index(flat, ref.shape)
        result["first_mismatch"] = {
            "flat_index": flat,
            "index": [int(value) for value in index],
            "reference": float(ref[index]),
            "candidate": float(cand[index]),
            "absolute_delta": float(cand64[index] - ref64[index]),
        }
    else:
        result["first_mismatch"] = None
    return result


def _complex_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.complex64)
    cand = np.asarray(candidate, dtype=np.complex64)
    packed_ref = np.stack((ref.real, ref.imag), axis=-1)
    packed_cand = np.stack((cand.real, cand.imag), axis=-1)
    return _metric(packed_ref, packed_cand)


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.subtract(values, np.max(values), dtype=np.float32)


def _relion_tree_sum(terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terms = np.asarray(terms, dtype=np.float32)
    _require(terms.ndim == 2, "tree-reduction terms must be candidate by pixel")
    pixel_count = terms.shape[1]
    pass_count = (pixel_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = pass_count * BLOCK_SIZE
    terms_padded = np.pad(terms, ((0, 0), (0, padded - pixel_count)))
    lanes = np.zeros((terms.shape[0], BLOCK_SIZE), dtype=np.float32)
    passes = terms_padded.reshape(terms.shape[0], pass_count, BLOCK_SIZE)
    for pass_index in range(pass_count):
        lanes = np.add(lanes, passes[:, pass_index, :], dtype=np.float32)
    pre_tree_lanes = lanes.copy()
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = np.add(lanes[:, :width], lanes[:, width : 2 * width], dtype=np.float32)
    return pre_tree_lanes, lanes[:, 0]


def _native_gaussian_components(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr_img: np.ndarray,
    highres_xi2_half: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply RELION's source-ordered float32 Gaussian fine-score reduction."""

    ref = np.asarray(reference, dtype=np.complex64)
    img = np.asarray(shifted, dtype=np.complex64)
    corr = np.asarray(corr_img, dtype=np.float32)
    _require(ref.shape == img.shape, "reference and shifted arrays are misaligned")
    _require(ref.ndim == 2 and corr.shape == (ref.shape[1],), "invalid fine operand shapes")

    delta_real = np.subtract(ref.real, img.real, dtype=np.float32)
    delta_imag = np.subtract(ref.imag, img.imag, dtype=np.float32)
    squared_real = np.multiply(delta_real, delta_real, dtype=np.float32)
    squared_imag = np.multiply(delta_imag, delta_imag, dtype=np.float32)
    squared_norm = np.add(squared_real, squared_imag, dtype=np.float32)
    half_squared_norm = np.multiply(squared_norm, np.float32(0.5), dtype=np.float32)
    pixel_terms = np.multiply(half_squared_norm, corr[None, :], dtype=np.float32)
    pre_tree_lanes, tree_sum = _relion_tree_sum(pixel_terms)
    raw_score = np.add(
        tree_sum,
        np.asarray(highres_xi2_half, dtype=np.float32),
        dtype=np.float32,
    )
    return pixel_terms, pre_tree_lanes, raw_score


def _counterfactual_metrics(
    *,
    native_raw: np.ndarray,
    native_ref: np.ndarray,
    native_shifted: np.ndarray,
    native_corr: np.ndarray,
    recovar_ref: np.ndarray,
    recovar_shifted: np.ndarray,
    recovar_corr: np.ndarray,
    native_highres_xi2_half: np.float32,
    recovar_highres_xi2_half: np.float32,
) -> dict[str, Any]:
    arms = {
        "recovar_all": (recovar_ref, recovar_shifted, recovar_corr),
        "native_corr_only": (recovar_ref, recovar_shifted, native_corr),
        "native_reference_only": (native_ref, recovar_shifted, recovar_corr),
        "native_shifted_only": (recovar_ref, native_shifted, recovar_corr),
        "native_reference_and_shifted": (native_ref, native_shifted, recovar_corr),
        "native_all_operands": (native_ref, native_shifted, native_corr),
    }
    centered_native = _center(native_raw)
    reports: dict[str, Any] = {}
    for name, operands in arms.items():
        highres = (
            native_highres_xi2_half
            if name == "native_all_operands"
            else recovar_highres_xi2_half
        )
        *_, score = _native_gaussian_components(*operands, highres)
        reports[name] = _metric(centered_native, _center(score))
    return reports


def _native_to_recovar_compact(
    *,
    native_image_size: int,
    recovar_full_to_compact: np.ndarray,
) -> np.ndarray:
    native_xdim = int(round(np.sqrt(2 * native_image_size))) // 2 + 1
    native_ydim = native_image_size // native_xdim
    _require(native_xdim * native_ydim == native_image_size, "invalid native rFFT shape")
    _require(native_ydim == 2 * (native_xdim - 1), "native rFFT is not square")
    recovar_full_size = int(recovar_full_to_compact.size)
    recovar_xdim = int(round(np.sqrt(2 * recovar_full_size))) // 2 + 1
    recovar_ydim = recovar_full_size // recovar_xdim
    _require(recovar_xdim * recovar_ydim == recovar_full_size, "invalid RECOVAR rFFT shape")
    _require(recovar_ydim == 2 * (recovar_xdim - 1), "RECOVAR rFFT is not square")
    mapping = np.full(native_image_size, -1, dtype=np.int64)
    for native_row in range(native_ydim):
        signed_y = native_row if native_row <= native_ydim // 2 else native_row - native_ydim
        recovar_row = signed_y if signed_y >= 0 else signed_y + recovar_ydim
        for x_coord in range(native_xdim):
            native_flat = native_row * native_xdim + x_coord
            recovar_flat = recovar_row * recovar_xdim + x_coord
            mapping[native_flat] = int(recovar_full_to_compact[recovar_flat])
    return mapping


def analyze(
    *,
    native_verbose_dir: Path,
    native_factor: Path,
    native_fine_score: Path,
    recovar_capture: Path,
    physical_image_size: int,
) -> dict[str, Any]:
    factor = load_factor_capture(native_factor)
    fine = load_fine_score_capture(native_fine_score)
    _require(factor.stack_index == fine.stack_index, "native identity changed")
    active_mask = (fine.candidates["flags"] & ACTIVE) != 0
    legacy_candidates = fine.candidates[active_mask]
    legacy_active_count = int(legacy_candidates.size)

    corr_path = _one(native_verbose_dir, "pass1_img*_corr_img.bin")
    prefix = corr_path.name[: -len("corr_img.bin")]
    image_size_path = native_verbose_dir / f"{prefix}image_size.bin"
    native_corr = _load_flat_real(corr_path).astype(np.float32)
    image_size = int(round(_load_scalar(image_size_path)))
    _require(native_corr.shape == (image_size,), "native corr_img size mismatch")
    native_preprocessed = _load_flat_complex(
        native_verbose_dir / "preprocess_img0_masked_fourier_post_optics_real.bin",
        native_verbose_dir / "preprocess_img0_masked_fourier_post_optics_imag.bin",
    )
    native_corrected = _load_flat_complex(
        native_verbose_dir / "pass1_img0_Fimg_corrected_real.bin",
        native_verbose_dir / "pass1_img0_Fimg_corrected_imag.bin",
    )
    _require(
        native_preprocessed.shape == native_corrected.shape == (image_size,),
        "native preprocessing operands changed topology",
    )

    verbose_raw = _load_flat_real(
        native_verbose_dir / "pass1_exp_Mweight_raw_preprior.bin"
    ).astype(np.float32)
    verbose_rotation_indices = _load_flat_int(
        native_verbose_dir / "pass1_acc_rot_idx.bin"
    )
    verbose_translation_ids = _load_flat_int(native_verbose_dir / "pass1_acc_trans_idx.bin")
    verbose_count = int(verbose_raw.size)
    _require(
        verbose_rotation_indices.shape
        == verbose_translation_ids.shape
        == (verbose_count,),
        "verbose candidate identity arrays are misaligned",
    )
    verbose_keys = list(
        zip(verbose_rotation_indices.tolist(), verbose_translation_ids.tolist())
    )
    legacy_keys = list(
        zip(
            np.asarray(legacy_candidates["rotation_local"], dtype=np.int64).tolist(),
            np.asarray(legacy_candidates["translation_id"], dtype=np.int64).tolist(),
        )
    )
    _require(len(set(verbose_keys)) == verbose_count, "verbose candidate keys are not unique")
    _require(
        len(set(legacy_keys)) == legacy_active_count,
        "fine-score candidate keys are not unique",
    )
    verbose_key_set = set(verbose_keys)
    legacy_key_set = set(legacy_keys)
    verbose_by_key = {key: index for index, key in enumerate(verbose_keys)}
    legacy_by_key = {key: index for index, key in enumerate(legacy_keys)}
    verbose_only = sorted(verbose_key_set - legacy_key_set)
    legacy_only = sorted(legacy_key_set - verbose_key_set)
    common_keys = [key for key in verbose_keys if key in legacy_key_set]
    verbose_common_order = np.asarray(
        [verbose_by_key[key] for key in common_keys], dtype=np.int64
    )
    legacy_common_order = np.asarray(
        [legacy_by_key[key] for key in common_keys], dtype=np.int64
    )

    def fine_array(name: str) -> np.ndarray:
        path = native_verbose_dir / f"pass1_class0_{name}.bin"
        values = _load_flat_real(path).astype(np.float32)
        _require(
            values.size == verbose_count * image_size,
            f"native {name} candidate/pixel size mismatch",
        )
        return values.reshape(verbose_count, image_size)

    native_ref = fine_array("fine_ref_real") + 1j * fine_array("fine_ref_imag")
    native_shifted = fine_array("fine_shifted_real") + 1j * fine_array("fine_shifted_imag")
    native_ref = native_ref.astype(np.complex64)
    native_shifted = native_shifted.astype(np.complex64)
    legacy_raw = np.asarray(legacy_candidates["raw_diff2"], dtype=np.float32)

    with np.load(recovar_capture, allow_pickle=False) as archive:
        rotations = np.asarray(archive["rotations"])
        translations = np.asarray(archive["fine_translations"])
        full_to_compact = np.asarray(
            archive["raw_operand_relion_full_to_compact"], dtype=np.int64
        )
        recovar_ref_compact = np.asarray(archive["raw_operand_proj_half"], dtype=np.complex64)
        recovar_shifted_compact = np.asarray(
            archive["raw_operand_shifted_corrected"], dtype=np.complex64
        )
        recovar_corr_compact = np.asarray(
            archive["raw_operand_corr_img_score"], dtype=np.float32
        )
        recovar_raw_dense = np.asarray(archive["raw_operand_raw_diff2"], dtype=np.float32)
        half_weights = np.asarray(archive["raw_operand_half_weights"], dtype=np.float32)
        recovar_highres_xi2_half = np.asarray(
            archive["raw_operand_highres_xi2_half"], dtype=np.float32
        )
        recovar_preprocessed_compact = np.asarray(
            archive["direct_preprocessed_score_input"], dtype=np.complex64
        )
        recovar_corrected_compact = np.asarray(
            archive["direct_score_input"], dtype=np.complex64
        )

    native_to_compact = _native_to_recovar_compact(
        native_image_size=image_size,
        recovar_full_to_compact=full_to_compact,
    )
    valid = native_to_compact >= 0
    native_xdim = int(round(np.sqrt(2 * image_size))) // 2 + 1
    native_ydim = image_size // native_xdim
    native_rows = np.arange(image_size, dtype=np.int64) // native_xdim
    native_nyquist_row = native_ydim // 2
    nyquist_valid = valid & (native_rows == native_nyquist_row)
    ordinary_valid = valid & ~nyquist_valid
    compact_rows = native_to_compact[valid]
    _require(np.unique(compact_rows).size == compact_rows.size, "compact lookup is not injective")
    _require(np.all(native_corr[~valid] == 0.0), "unmapped native pixels contribute to scoring")
    _require(np.all(half_weights == np.float32(1.0)), "Gaussian half weights are not unity")

    rotation_map, rotation_error = _rotation_map(factor.rotations, rotations)
    translation_map, translation_error = _translation_map(
        factor.translations,
        translations,
        physical_image_size=physical_image_size,
    )
    mapped_rotation = rotation_map[verbose_rotation_indices]
    mapped_translation = translation_map[verbose_translation_ids]
    tuple_keys = np.column_stack((mapped_rotation, mapped_translation))
    _require(
        np.unique(tuple_keys, axis=0).shape[0] == verbose_count,
        "mapped tuples are not unique",
    )

    recovar_ref = np.zeros((verbose_count, image_size), dtype=np.complex64)
    recovar_shifted = np.zeros_like(recovar_ref)
    recovar_corr = np.zeros((image_size,), dtype=np.float32)
    recovar_ref[:, valid] = recovar_ref_compact[mapped_rotation][:, compact_rows]
    recovar_shifted[:, valid] = recovar_shifted_compact[mapped_translation][:, compact_rows]
    recovar_corr[valid] = recovar_corr_compact[compact_rows]
    recovar_raw = recovar_raw_dense[mapped_rotation, mapped_translation]

    # RECOVAR's full-image Fourier convention is larger than RELION's by N^2.
    # The captured RELION CTF has the opposite sign, so both the CTF-weighted
    # reference and corrected image share a harmless global minus sign.  Align
    # that common gauge before comparing either operand.  Xi^-2 carries the
    # compensating N^4 factor in native RELION.
    fft_scale = np.float32(physical_image_size * physical_image_size)
    recovar_preprocessed = np.zeros((image_size,), dtype=np.complex64)
    recovar_corrected = np.zeros((image_size,), dtype=np.complex64)
    recovar_preprocessed[valid] = np.divide(
        recovar_preprocessed_compact[compact_rows],
        fft_scale,
        dtype=np.complex64,
    )
    recovar_corrected[valid] = np.negative(
        np.divide(
            recovar_corrected_compact[compact_rows],
            fft_scale,
            dtype=np.complex64,
        ),
        dtype=np.complex64,
    )
    recovar_ref = np.negative(
        np.divide(recovar_ref, fft_scale, dtype=np.complex64),
        dtype=np.complex64,
    )
    recovar_shifted = np.negative(
        np.divide(recovar_shifted, fft_scale, dtype=np.complex64),
        dtype=np.complex64,
    )
    recovar_corr = np.multiply(
        recovar_corr,
        np.multiply(fft_scale, fft_scale, dtype=np.float32),
        dtype=np.float32,
    )

    mapped_compact = np.zeros(recovar_corr_compact.size, dtype=bool)
    mapped_compact[compact_rows] = True
    extra_compact = ~mapped_compact
    extra_reference_max_abs = float(
        np.max(
            np.abs(recovar_ref_compact[mapped_rotation][:, extra_compact]),
            initial=0.0,
        )
    )
    _require(extra_reference_max_abs == 0.0, "RECOVAR-only score pixels have nonzero reference")

    native_without_highres = _native_gaussian_components(
        native_ref,
        native_shifted,
        native_corr,
        np.float32(0.0),
    )[-1]
    inferred_highres = np.subtract(verbose_raw, native_without_highres, dtype=np.float32)
    native_highres_xi2_half = np.float32(np.median(inferred_highres))
    native_components = _native_gaussian_components(
        native_ref,
        native_shifted,
        native_corr,
        native_highres_xi2_half,
    )
    recovar_components = _native_gaussian_components(
        recovar_ref,
        recovar_shifted,
        recovar_corr,
        recovar_highres_xi2_half,
    )
    component_names = (
        "gaussian_pixel_terms",
        "gaussian_pre_tree_lanes",
        "source_order_gaussian_raw_score",
    )
    components = {
        name: _metric(native_value, recovar_value)
        for name, native_value, recovar_value in zip(
            component_names, native_components, recovar_components, strict=True
        )
    }
    native_nyquist_reference_max_abs = float(
        np.max(np.abs(native_ref[:, nyquist_valid]), initial=0.0)
    )

    legacy_repeat_diagnostics = {
        "verbose_raw_vs_legacy_fine_score_common_candidates": _metric(
            legacy_raw[legacy_common_order], verbose_raw[verbose_common_order]
        )
    }
    boundaries = {
        "corr_img_valid_pixels": _metric(native_corr[valid], recovar_corr[valid]),
        "masked_fourier_post_optics_valid_pixels": _complex_metric(
            native_preprocessed[valid], recovar_preprocessed[valid]
        ),
        "corrected_image_before_translation_valid_pixels": _complex_metric(
            native_corrected[valid], recovar_corrected[valid]
        ),
        "projected_reference_valid_pixels": _complex_metric(
            native_ref[:, valid], recovar_ref[:, valid]
        ),
        "shifted_image_valid_pixels": _complex_metric(
            native_shifted[:, valid], recovar_shifted[:, valid]
        ),
        "shifted_image_ordinary_rows": _complex_metric(
            native_shifted[:, ordinary_valid],
            recovar_shifted[:, ordinary_valid],
        ),
        "shifted_image_positive_nyquist_row": _complex_metric(
            native_shifted[:, nyquist_valid],
            recovar_shifted[:, nyquist_valid],
        ),
        "gaussian_pixel_terms_positive_nyquist_row": _metric(
            native_components[0][:, nyquist_valid],
            recovar_components[0][:, nyquist_valid],
        ),
        **components,
        "centered_raw_fine_score": _metric(_center(verbose_raw), _center(recovar_raw)),
    }
    ordered = (
        "corr_img_valid_pixels",
        "masked_fourier_post_optics_valid_pixels",
        "corrected_image_before_translation_valid_pixels",
        "projected_reference_valid_pixels",
        "shifted_image_valid_pixels",
        "gaussian_pixel_terms",
        "gaussian_pre_tree_lanes",
        "source_order_gaussian_raw_score",
        "centered_raw_fine_score",
    )
    first_unequal = next(
        (
            name
            for name in ordered
            if boundaries[name].get("bit_equal_fraction") != 1.0
        ),
        None,
    )
    counterfactuals = _counterfactual_metrics(
        native_raw=verbose_raw,
        native_ref=native_ref,
        native_shifted=native_shifted,
        native_corr=native_corr,
        recovar_ref=recovar_ref,
        recovar_shifted=recovar_shifted,
        recovar_corr=recovar_corr,
        native_highres_xi2_half=native_highres_xi2_half,
        recovar_highres_xi2_half=recovar_highres_xi2_half,
    )
    nyquist_corrected_shifted = recovar_shifted.copy()
    nyquist_corrected_shifted[:, nyquist_valid] = native_shifted[:, nyquist_valid]
    nyquist_corrected_score = _native_gaussian_components(
        recovar_ref,
        nyquist_corrected_shifted,
        recovar_corr,
        recovar_highres_xi2_half,
    )[-1]
    counterfactuals["native_positive_nyquist_shifted_only"] = _metric(
        _center(verbose_raw),
        _center(nyquist_corrected_score),
    )

    files = sorted(native_verbose_dir.glob("pass1_*.bin"))
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": "exact bytes and relative L2; no correlation",
        "stack_index_one_based": int(fine.stack_index),
        "active_candidate_count": verbose_count,
        "verbose_candidate_count": verbose_count,
        "legacy_active_candidate_count": legacy_active_count,
        "candidate_set_comparison": {
            "common_count": len(common_keys),
            "verbose_only_count": len(verbose_only),
            "legacy_only_count": len(legacy_only),
            "verbose_only_first": [list(key) for key in verbose_only[:16]],
            "legacy_only_first": [list(key) for key in legacy_only[:16]],
        },
        "native_image_size": image_size,
        "native_positive_nyquist_row": native_nyquist_row,
        "native_positive_nyquist_score_pixel_count": int(
            np.count_nonzero(nyquist_valid)
        ),
        # RELION's current-size kernel replaces x by maxR on these six
        # out-of-radius sentinel pixels.  Their projected reference is zero,
        # so only the phase-invariant shifted-image power remains.  Report the
        # actual rounded Gaussian terms separately above instead of treating
        # the visibly different translated-image words as a scoring boundary.
        "native_positive_nyquist_reference_max_abs": native_nyquist_reference_max_abs,
        "native_positive_nyquist_reference_cross_term_zero": bool(
            native_nyquist_reference_max_abs == 0.0
        ),
        "valid_score_pixel_count": int(np.count_nonzero(valid)),
        "recovar_only_zero_reference_pixel_count": int(np.count_nonzero(extra_compact)),
        "recovar_only_reference_max_abs": extra_reference_max_abs,
        "native_highres_xi2_half_inferred_median": float(native_highres_xi2_half),
        "native_highres_xi2_half_inferred_range": [
            float(np.min(inferred_highres)),
            float(np.max(inferred_highres)),
        ],
        "recovar_highres_xi2_half": float(recovar_highres_xi2_half),
        "recovar_to_native_fourier_scale": float(fft_scale),
        "recovar_to_native_common_ctf_sign": -1,
        "first_non_bit_exact_boundary": first_unequal,
        "ordered_boundaries": list(ordered),
        "rotation_map_max_abs": float(rotation_error),
        "translation_map_max_abs": float(translation_error),
        "boundaries": boundaries,
        "legacy_repeat_diagnostics": legacy_repeat_diagnostics,
        "counterfactual_centered_score_metrics": counterfactuals,
        "artifacts": {
            "native_verbose_dir": str(native_verbose_dir.resolve()),
            "native_verbose_files": [
                {"path": str(path.resolve()), "sha256": _sha256(path)} for path in files
            ],
            "native_factor": str(native_factor.resolve()),
            "native_factor_sha256": _sha256(native_factor),
            "native_fine_score": str(native_fine_score.resolve()),
            "native_fine_score_sha256": _sha256(native_fine_score),
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-verbose-dir", type=Path, required=True)
    parser.add_argument("--native-factor", type=Path, required=True)
    parser.add_argument("--native-fine-score", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_verbose_dir=args.native_verbose_dir,
        native_factor=args.native_factor,
        native_fine_score=args.native_fine_score,
        recovar_capture=args.recovar_capture,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
