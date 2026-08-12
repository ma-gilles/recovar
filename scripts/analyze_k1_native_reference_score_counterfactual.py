#!/usr/bin/env python3
"""Score a bounded K=1 coarse panel after substituting native RELION references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.analyze_k1_coarse_operand_boundary_v3 import (
    _rotation_key_to_recovar,
)
from scripts.validate_relion_coarse_operand_capture import (
    load_artifact as load_operand,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    load_artifact as load_components,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _centered_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    _require(candidate.shape == reference.shape and candidate.size > 0, "score topology mismatch")
    candidate_centered = (candidate - np.max(candidate)).astype(np.float32)
    reference_centered = (reference - np.max(reference)).astype(np.float32)
    residual = candidate_centered.astype(np.float64) - reference_centered.astype(np.float64)
    return {
        "count": int(candidate.size),
        "bitwise_equal_count": int(
            np.count_nonzero(candidate_centered.view(np.uint32) == reference_centered.view(np.uint32))
        ),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate).reshape(-1).astype(np.complex128)
    reference = np.asarray(reference).reshape(-1).astype(np.complex128)
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    _require(jax.default_backend() == "gpu", "counterfactual requires a GPU")
    _require(cuda_backproject.cuda_available(), "custom CUDA extension is unavailable")
    components = load_components(args.components)
    operand = load_operand(args.operands)
    with np.load(args.recovar, allow_pickle=False) as payload:
        recovar = {name: np.asarray(payload[name]) for name in payload.files}

    _require(components.part_id == operand.part_id, "native particle mismatch")
    _require(components.stack_index == operand.stack_index, "native stack mismatch")
    _require(int(recovar["original_index"]) == components.stack_index - 1, "cross-engine identity mismatch")
    current_size = int(recovar["current_size"])
    score_indices = np.asarray(recovar["coarse_gaussian_score_indices"], dtype=np.int32)
    unshifted = np.asarray(recovar["coarse_gaussian_unshifted_corrected"], dtype=np.complex64)
    pixel_weight = np.asarray(recovar["coarse_gaussian_pixel_weight"], dtype=np.float32).reshape(1, -1)
    initial_diff2 = np.asarray(recovar["coarse_gaussian_initial_diff2"], dtype=np.float32).reshape(1)
    translation_source = np.asarray(recovar["translation_phase_source"], dtype=np.float64)
    image_shape = (128, 128)

    angles = _relion_translation_angles_f32(translation_source, image_shape)
    shifted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(unshifted.reshape(1, -1)),
        jnp.asarray(angles, dtype=jnp.float32),
        jnp.asarray(score_indices, dtype=jnp.int32),
        image_shape,
    ).reshape(1, translation_source.shape[0], -1)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        score_indices,
    )

    native_reference = relion_reference_on_recovar_window(
        operand.reference_real.astype(np.float32) + 1j * operand.reference_imag.astype(np.float32),
        score_indices,
        full_image_size=image_shape[0],
        current_size=current_size,
    ).astype(np.complex64)
    native_diff2_via_recovar = np.asarray(
        cuda_backproject.relion_coarse_diff2_rectangular_f32(
            jnp.asarray(native_reference),
            shifted,
            jnp.asarray(pixel_weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )[0],
        dtype=np.float32,
    )

    n_directions, n_psi, _ = components.header[10:13]
    translation_permutation, translation_report = _translation_permutation(
        components.translations,
        np.asarray(recovar["translations"], dtype=np.float64),
    )
    mapped_native_diff2 = _map_relion_table(
        components.raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float32)
    rotation_ids = np.asarray(
        [_rotation_key_to_recovar(key, n_directions, n_psi) for key in operand.rotation_keys],
        dtype=np.int64,
    )
    selected_native_diff2 = mapped_native_diff2[rotation_ids]
    _require(np.all(selected_native_diff2 != RELION_INVALID_DIFF2), "captured rotations contain inactive candidates")
    selected_recovar_diff2 = -np.asarray(
        recovar["scores_pre_prior_per_class"][0, rotation_ids],
        dtype=np.float32,
    )

    native_shifted = relion_values_on_recovar_window(
        operand.shifted_real.astype(np.float32) + 1j * operand.shifted_imag.astype(np.float32),
        score_indices,
        full_image_size=image_shape[0],
        current_size=current_size,
    )
    native_shifted_ordered = np.empty_like(native_shifted)
    native_shifted_ordered[translation_permutation] = native_shifted
    expected_shifted = (-float(image_shape[0] ** 2) * native_shifted_ordered).astype(np.complex64)
    native_correction = relion_values_on_recovar_window(
        operand.correction[np.newaxis, :],
        score_indices,
        full_image_size=image_shape[0],
        current_size=current_size,
    )[0].real.astype(np.float32)
    expected_weight = (native_correction / np.float32(image_shape[0] ** 4)).astype(np.float32)

    active_pixels = pixel_weight[0] > 0.0
    shifted_active = np.asarray(shifted[0])[:, active_pixels]
    expected_shifted_active = expected_shifted[:, active_pixels]
    report = {
        "schema": "recovar.em.k1_native_reference_score_counterfactual.v1",
        "particle": {
            "part_id": components.part_id,
            "stack_index_one_based": components.stack_index,
            "original_index_zero_based": int(recovar["original_index"]),
        },
        "rotation_ids_recovar": rotation_ids.tolist(),
        "translation_mapping": translation_report,
        "operand_equality": {
            "active_translated_image_bitwise_equal_count": int(
                np.count_nonzero(
                    shifted_active.view(np.uint64)
                    == expected_shifted_active.view(np.uint64)
                )
            ),
            "active_translated_image_count": int(expected_shifted_active.size),
            "active_translated_image_relative_l2": _relative_l2(
                shifted_active,
                expected_shifted_active,
            ),
            "pixel_weight_bitwise_equal_count": int(
                np.count_nonzero(pixel_weight[0].view(np.uint32) == expected_weight.view(np.uint32))
            ),
            "pixel_weight_count": int(expected_weight.size),
            "pixel_weight_relative_l2": _relative_l2(pixel_weight[0], expected_weight),
        },
        "score_comparison": {
            "recovar_reference_vs_native": _centered_stats(
                selected_recovar_diff2,
                selected_native_diff2,
            ),
            "native_reference_counterfactual_vs_native": _centered_stats(
                native_diff2_via_recovar,
                selected_native_diff2,
            ),
            "native_reference_counterfactual_vs_recovar": _centered_stats(
                native_diff2_via_recovar,
                selected_recovar_diff2,
            ),
        },
        "artifacts": {
            "components": str(args.components.resolve()),
            "components_sha256": _sha256(args.components),
            "operands": str(args.operands.resolve()),
            "operands_sha256": _sha256(args.operands),
            "recovar": str(args.recovar.resolve()),
            "recovar_sha256": _sha256(args.recovar),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
