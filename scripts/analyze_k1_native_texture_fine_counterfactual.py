#!/usr/bin/env python3
"""Score one captured K=1 fine table with RELION's native texture topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
    relion_projector_half_to_texture_full,
)
from recovar.em.dense_single_volume.helpers.significance import _dense_projection_scale
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_diff2_min,
    _relion_cuda_fine_diff2_to_scores,
    _relion_cuda_fine_pixel_weights,
    _relion_translation_angles_f32,
)
from recovar.em.initial_model.dense_adapter import (
    reference_to_relion_projector_half_maps,
)
from recovar.utils import helpers
from scripts.analyze_k1_fine_score_boundary import _rotation_map, _translation_map
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

SCHEMA = "recovar.em.k1_native_texture_fine_counterfactual.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    _require(reference.shape == candidate.shape, "metric operands have different shapes")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    mismatch = np.flatnonzero(reference != candidate)
    return {
        "shape": list(reference.shape),
        "mismatch_count": int(mismatch.size),
        "exact_equal": bool(mismatch.size == 0),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_reference": (
            float(np.linalg.norm(delta) / denominator)
            if denominator > 0.0
            else float(np.linalg.norm(delta))
        ),
        "first_mismatch_flat": int(mismatch[0]) if mismatch.size else None,
    }


def _center_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float32)
    return np.subtract(np.min(cost), cost, dtype=np.float32)


def _pmax_from_raw_diff2(
    raw_diff2: np.ndarray,
    rotation_log_prior: np.ndarray,
    translation_log_prior: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    """Replay the production XFLOAT score conversion and float64 softmax."""

    raw = jnp.asarray(raw_diff2[None, ...], dtype=jnp.float32)
    mask = jnp.asarray(candidate_mask[None, ...], dtype=bool)
    minimum = _relion_cuda_fine_diff2_min(raw, mask)
    scores = np.asarray(
        _relion_cuda_fine_diff2_to_scores(
            raw,
            jnp.asarray(rotation_log_prior[None, :, None], dtype=jnp.float32),
            jnp.asarray(translation_log_prior[None, None, :], dtype=jnp.float32),
            mask,
            min_diff2=minimum,
        )[0],
        dtype=np.float64,
    )
    finite = np.isfinite(scores)
    _require(np.any(finite), "fine posterior has no finite candidates")
    maximum = float(np.max(scores[finite]))
    weights = np.zeros_like(scores, dtype=np.float64)
    weights[finite] = np.exp(scores[finite] - maximum)
    normalization = float(np.sum(weights, dtype=np.float64))
    _require(np.isfinite(normalization) and normalization > 0.0, "invalid fine posterior mass")
    return float(np.max(weights) / normalization)


def analyze(
    *,
    recovar_capture: Path,
    reference_map: Path,
    native_factor: Path,
    native_fine_score: Path,
    physical_image_size: int,
    padding_factor: int,
    projector_current_size: int,
) -> dict[str, object]:
    factor = load_factor_capture(native_factor)
    fine = load_fine_score_capture(native_fine_score)
    _require(factor.stack_index == fine.stack_index, "native particle identity changed")

    with np.load(recovar_capture, allow_pickle=False) as archive:
        capture = {name: np.asarray(archive[name]) for name in archive.files}
    current_size = int(capture["current_size"])
    _require(int(capture["half"]) in (1, 2), "captured half must be 1 or 2")
    _require(int(capture["original_index"]) + 1 == fine.stack_index, "particle identity changed")

    rotations = np.asarray(capture["rotations"], dtype=np.float32)
    translations = np.asarray(capture["fine_translations"], dtype=np.float32)
    rotation_map, rotation_error = _rotation_map(factor.rotations, rotations)
    translation_map, translation_error = _translation_map(
        factor.translations,
        translations,
        physical_image_size=physical_image_size,
    )
    native_candidates = fine.candidates
    active = (native_candidates["flags"] & ACTIVE) != 0
    selected = native_candidates[active]
    _require(selected.size > 0, "native fine capture has no active candidates")
    mapped_rotation = rotation_map[
        np.asarray(selected["rotation_local"], dtype=np.int64)
    ]
    mapped_translation = translation_map[
        np.asarray(selected["translation_id"], dtype=np.int64)
    ]
    mapped_keys = np.column_stack((mapped_rotation, mapped_translation))
    _require(
        np.unique(mapped_keys, axis=0).shape[0] == mapped_keys.shape[0],
        "mapped native candidate tuples are not unique",
    )
    native_raw = np.asarray(selected["raw_diff2"], dtype=np.float32)

    reference_real = np.asarray(helpers.load_mrc(str(reference_map)), dtype=np.float64)
    _require(
        reference_real.ndim == 3 and len(set(reference_real.shape)) == 1,
        "reference map must be cubic",
    )
    projector_half, projector_r_max = reference_to_relion_projector_half_maps(
        reference_real[None],
        current_size=projector_current_size,
        padding_factor=padding_factor,
    )
    texture_full = relion_projector_half_to_texture_full(
        jnp.asarray(projector_half[0], dtype=jnp.complex64)
    )
    texture_full = texture_full * np.float32(
        _dense_projection_scale((physical_image_size, physical_image_size))
    )
    rebuilt_generic_projection = np.asarray(
        compute_relion_projector_projections_block(
            jnp.asarray(projector_half[0], dtype=jnp.complex64),
            jnp.asarray(rotations, dtype=jnp.float32),
            (physical_image_size, physical_image_size),
            r_max=int(projector_r_max),
            padding_factor=padding_factor,
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            pixel_indices=np.asarray(capture["window_indices"], dtype=np.int32),
            relion_texture_interp=True,
        )[0],
        dtype=np.complex64,
    )

    image_scale = np.float32(physical_image_size * physical_image_size)
    image = (
        np.asarray(capture["direct_score_input"], dtype=np.complex64) / image_scale
    )[None, :]
    corr = np.asarray(capture["raw_operand_corr_img_score"], dtype=np.float32)
    half_weights = np.asarray(capture["raw_operand_half_weights"], dtype=np.float32)
    weights = np.asarray(
        _relion_cuda_fine_pixel_weights(
            jnp.asarray(corr),
            jnp.asarray(half_weights),
        ),
        dtype=np.float32,
    )[None, :]
    translation_angles = _relion_translation_angles_f32(
        translations,
        (physical_image_size, physical_image_size),
    )
    initial_diff2 = np.asarray(
        [capture["raw_operand_highres_xi2_half"]], dtype=np.float32
    )
    lookup = np.asarray(
        capture["raw_operand_relion_full_to_compact"], dtype=np.int32
    )

    def texture_score(rotation_operand: np.ndarray) -> np.ndarray:
        result = np.asarray(
            cuda_backproject.relion_fine_diff2_native_texture_rectangular_f32(
                texture_full,
                jnp.asarray(rotation_operand[None, ...], dtype=jnp.float32),
                jnp.asarray(image),
                jnp.asarray(translation_angles),
                jnp.asarray(weights),
                jnp.asarray(initial_diff2),
                jnp.asarray(lookup),
                current_size=current_size,
                padding_factor=padding_factor,
                projector_max_r=int(projector_r_max),
            )
        )[0]
        _require(
            result.shape == (rotations.shape[0], translations.shape[0]),
            "native texture output shape changed",
        )
        _require(np.all(np.isfinite(result)), "native texture output is nonfinite")
        return result

    texture_raw_direct = texture_score(rotations)
    # RECOVAR's generic projector converts scorer matrices to two compact
    # rows before the CUDA texture launch. Fine/M-step host matrices use the
    # opposite storage convention from coarse AccProjectorPlan matrices, so
    # preserve both arms until the same-input score proves the required map.
    texture_raw_transposed = texture_score(
        np.ascontiguousarray(rotations.transpose(0, 2, 1))
    )

    # Recover individual complex texture samples through the already-qualified
    # native normalized-CC pair kernel. A one-hot real/imaginary image makes
    # its numerator equal the projected reference's corresponding component.
    # This localizes a score failure to projection coordinates without adding
    # another diagnostic CUDA kernel.
    probe_rotation_ids = np.unique(mapped_rotation)[:4]
    probe_pixel_ids = np.linspace(
        0,
        image.shape[1] - 1,
        num=min(16, image.shape[1]),
        dtype=np.int64,
    )
    probe_rotation_grid, probe_pixel_grid = np.meshgrid(
        probe_rotation_ids,
        probe_pixel_ids,
        indexing="ij",
    )
    probe_rotation_rows = probe_rotation_grid.reshape(-1)
    probe_pixel_rows = probe_pixel_grid.reshape(-1)
    probe_count = probe_rotation_rows.size
    probe_selector = np.zeros((probe_count, image.shape[1]), dtype=np.float32)
    probe_selector[np.arange(probe_count), probe_pixel_rows] = 1.0
    probe_real_image = probe_selector.astype(np.complex64)
    probe_imag_image = (np.complex64(1j) * probe_selector).astype(np.complex64)
    probe_zero_score_weight = np.zeros_like(probe_selector)
    probe_angles = np.zeros((probe_count, 2), dtype=np.float32)

    def probe_projected_reference(rotation_operand: np.ndarray) -> np.ndarray:
        common = {
            "projector_full": texture_full,
            "rotation_matrices": jnp.asarray(
                rotation_operand[probe_rotation_rows], dtype=jnp.float32
            ),
            "score_weight": jnp.asarray(probe_zero_score_weight),
            "half_weights": jnp.ones(image.shape[1], dtype=jnp.float32),
            "packed_to_compact": jnp.asarray(lookup),
            "current_size": current_size,
            "padding_factor": padding_factor,
            "projector_max_r": int(projector_r_max),
            "return_components": True,
            "translation_angles": jnp.asarray(probe_angles),
            "numerator_weight": jnp.asarray(probe_selector),
        }
        real = np.asarray(
            cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32(
                shifted_image=jnp.asarray(probe_real_image),
                **common,
            )
        )[:, 1]
        imag = np.asarray(
            cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32(
                shifted_image=jnp.asarray(probe_imag_image),
                **common,
            )
        )[:, 1]
        return np.asarray(real + np.complex64(1j) * imag, dtype=np.complex64)

    probe_expected = np.asarray(
        capture["raw_operand_proj_half"], dtype=np.complex64
    )[probe_rotation_rows, probe_pixel_rows]
    probe_original_input = probe_projected_reference(rotations)
    probe_transposed_input = probe_projected_reference(
        np.ascontiguousarray(rotations.transpose(0, 2, 1))
    )
    current_raw = np.asarray(capture["raw_operand_raw_diff2"], dtype=np.float32)
    fused_preshifted_raw = np.asarray(
        cuda_backproject.relion_fine_diff2_rectangular_f32(
            jnp.asarray(capture["raw_operand_proj_half"][None, ...], dtype=jnp.complex64),
            jnp.asarray(capture["raw_operand_shifted_corrected"][None, ...], dtype=jnp.complex64),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.asarray(lookup, dtype=jnp.int32),
        )
    )[0]
    fused_preshifted_raw = np.add(
        fused_preshifted_raw,
        initial_diff2[0],
        dtype=np.float32,
    )

    # Counterfactual for the immediately preceding production implementation:
    # keep the algebraically equivalent normalized-FFT image/reference/corr_img
    # operands instead of consuming RELION's native-unit corr_img directly.
    normalized_weights = np.asarray(
        _relion_cuda_fine_pixel_weights(
            jnp.asarray(capture["ctf2_over_nv_score"], dtype=jnp.float32),
            jnp.asarray(capture["half_weights"], dtype=jnp.float32),
        ),
        dtype=np.float32,
    )[None, :]
    normalized_unit_raw = np.asarray(
        cuda_backproject.relion_fine_diff2_rectangular_f32(
            jnp.asarray(capture["proj_half"][None, ...], dtype=jnp.complex64),
            jnp.asarray(capture["shifted_corrected"][None, ...], dtype=jnp.complex64),
            jnp.asarray(normalized_weights, dtype=jnp.float32),
            jnp.asarray(lookup, dtype=jnp.int32),
        )
    )[0]
    normalized_unit_raw = np.add(
        normalized_unit_raw,
        initial_diff2[0],
        dtype=np.float32,
    )
    fused_preprojected_raw = np.asarray(
        cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            jnp.asarray(capture["raw_operand_proj_half"][None, ...], dtype=jnp.complex64),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weights),
            jnp.asarray(lookup),
            current_size=current_size,
        )
    )[0]
    fused_preprojected_raw = np.add(
        fused_preprojected_raw,
        initial_diff2[0],
        dtype=np.float32,
    )
    current_aligned = current_raw[mapped_rotation, mapped_translation]
    fused_preprojected_aligned = fused_preprojected_raw[
        mapped_rotation, mapped_translation
    ]
    fused_preshifted_aligned = fused_preshifted_raw[
        mapped_rotation, mapped_translation
    ]
    normalized_unit_aligned = normalized_unit_raw[
        mapped_rotation, mapped_translation
    ]
    texture_direct_aligned = texture_raw_direct[mapped_rotation, mapped_translation]
    texture_transposed_aligned = texture_raw_transposed[
        mapped_rotation, mapped_translation
    ]

    current_centered = _center_cost(current_aligned)
    texture_direct_centered = _center_cost(texture_direct_aligned)
    texture_transposed_centered = _center_cost(texture_transposed_aligned)
    native_centered = _center_cost(native_raw)
    current_metric = _metric(native_centered, current_centered)
    normalized_unit_metric = _metric(
        native_centered,
        _center_cost(normalized_unit_aligned),
    )
    texture_direct_metric = _metric(native_centered, texture_direct_centered)
    texture_transposed_metric = _metric(native_centered, texture_transposed_centered)
    current_l2 = float(current_metric["relative_l2_over_reference"])
    texture_l2 = float(texture_transposed_metric["relative_l2_over_reference"])
    return {
        "schema": SCHEMA,
        "status": "ok",
        "metric_policy": "float32 centered fine pre-prior score on native active tuples",
        "particle": {
            "stack_index_one_based": int(fine.stack_index),
            "original_index_zero_based": int(capture["original_index"]),
            "half": int(capture["half"]),
            "current_size": current_size,
            "active_tuple_count": int(native_raw.size),
        },
        "projector": {
            "padding_factor": int(padding_factor),
            "projector_current_size": int(projector_current_size),
            "projector_r_max": int(projector_r_max),
            "projector_half_shape": list(projector_half[0].shape),
            "dense_projection_scale": float(
                _dense_projection_scale((physical_image_size, physical_image_size))
            ),
        },
        "mapping": {
            "rotation_max_abs": float(rotation_error),
            "translation_max_abs": float(translation_error),
        },
        "projection_pixel_probe": {
            "probe_count": int(probe_count),
            "rotation_ids": probe_rotation_ids.astype(int).tolist(),
            "pixel_ids": probe_pixel_ids.astype(int).tolist(),
            "original_rotation_input_vs_captured_reference": _metric(
                np.stack((probe_expected.real, probe_expected.imag), axis=-1),
                np.stack(
                    (probe_original_input.real, probe_original_input.imag), axis=-1
                ),
            ),
            "transposed_rotation_input_vs_captured_reference": _metric(
                np.stack((probe_expected.real, probe_expected.imag), axis=-1),
                np.stack(
                    (probe_transposed_input.real, probe_transposed_input.imag),
                    axis=-1,
                ),
            ),
        },
        "projector_rebuild_guard": {
            "generic_texture_projection_vs_captured_reference": _metric(
                np.stack(
                    (
                        np.asarray(capture["raw_operand_proj_half"]).real,
                        np.asarray(capture["raw_operand_proj_half"]).imag,
                    ),
                    axis=-1,
                ),
                np.stack(
                    (
                        rebuilt_generic_projection.real,
                        rebuilt_generic_projection.imag,
                    ),
                    axis=-1,
                ),
            ),
        },
        "comparisons": {
            "current_preprojected_vs_native": current_metric,
            "previous_normalized_units_vs_native": normalized_unit_metric,
            "previous_normalized_units_vs_current_native_units_raw": _metric(
                current_aligned,
                normalized_unit_aligned,
            ),
            "fused_preprojected_vs_native": _metric(
                native_centered,
                _center_cost(fused_preprojected_aligned),
            ),
            "fused_preshifted_vs_native": _metric(
                native_centered,
                _center_cost(fused_preshifted_aligned),
            ),
            "fused_preshifted_vs_current_raw": _metric(
                current_aligned,
                fused_preshifted_aligned,
            ),
            "fused_preprojected_vs_current_raw": _metric(
                current_aligned,
                fused_preprojected_aligned,
            ),
            "native_texture_direct_rotation_vs_native": texture_direct_metric,
            "native_texture_transposed_rotation_vs_native": texture_transposed_metric,
            "native_texture_transposed_vs_current_preprojected_raw": _metric(
                current_aligned, texture_transposed_aligned
            ),
        },
        "posterior_counterfactual": {
            "current_native_units_pmax": _pmax_from_raw_diff2(
                current_raw,
                np.asarray(capture["rotation_log_prior"], dtype=np.float32),
                np.asarray(capture["translation_log_prior"], dtype=np.float32),
                np.asarray(capture["candidate_mask"], dtype=bool),
            ),
            "previous_normalized_units_pmax": _pmax_from_raw_diff2(
                normalized_unit_raw,
                np.asarray(capture["rotation_log_prior"], dtype=np.float32),
                np.asarray(capture["translation_log_prior"], dtype=np.float32),
                np.asarray(capture["candidate_mask"], dtype=bool),
            ),
            "captured_current_probability_max": float(
                np.max(np.asarray(capture["probs"], dtype=np.float64))
            ),
        },
        "improvement": {
            "relative_l2_ratio_texture_over_current": (
                texture_l2 / current_l2 if current_l2 > 0.0 else None
            ),
            "relative_l2_fraction_reduced": (
                1.0 - texture_l2 / current_l2 if current_l2 > 0.0 else None
            ),
            "max_abs_fraction_reduced": (
                1.0
                - float(texture_transposed_metric["max_abs"])
                / float(current_metric["max_abs"])
                if float(current_metric["max_abs"]) > 0.0
                else None
            ),
        },
        "inputs": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "reference_map": str(reference_map.resolve()),
            "reference_map_sha256": _sha256(reference_map),
            "native_factor": str(native_factor.resolve()),
            "native_factor_sha256": _sha256(native_factor),
            "native_fine_score": str(native_fine_score.resolve()),
            "native_fine_score_sha256": _sha256(native_fine_score),
            "cuda_library": str(Path(os.environ["RECOVAR_CUDA_LIB"]).resolve()),
            "jax_backend": jax.default_backend(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--native-factor", type=Path, required=True)
    parser.add_argument("--native-fine-score", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--projector-current-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        recovar_capture=args.recovar_capture,
        reference_map=args.reference_map,
        native_factor=args.native_factor,
        native_fine_score=args.native_fine_score,
        physical_image_size=args.physical_image_size,
        padding_factor=args.padding_factor,
        projector_current_size=args.projector_current_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
