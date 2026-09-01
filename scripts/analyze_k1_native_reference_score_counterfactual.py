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
from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture
from scripts.analyze_k1_coarse_operand_boundary_v3 import (
    _rotation_key_to_recovar,
)
from scripts.validate_relion_coarse_operand_capture import (
    load_artifact as load_operand,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)
from scripts.validate_relion_coarse_pass1_components import (
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


def _complex_operand_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Compare same-shaped complex64 operands without discarding their bit patterns."""

    candidate = np.asarray(candidate, dtype=np.complex64)
    reference = np.asarray(reference, dtype=np.complex64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "complex operand topology mismatch")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    return {
        "count": int(candidate.size),
        "bitwise_equal_count": int(
            np.count_nonzero(candidate.view(np.uint64) == reference.view(np.uint64))
        ),
        "relative_l2": _relative_l2(candidate, reference),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _rotation_matrix_stats(native_projectors: np.ndarray, recovar_rotations: np.ndarray) -> dict[str, float | int]:
    """Compare RELION projector matrices with RECOVAR's transposed convention."""

    native = np.asarray(native_projectors, dtype=np.float32)
    recovar = np.asarray(recovar_rotations, dtype=np.float32)
    _require(native.shape == recovar.shape and native.ndim == 3, "rotation topology mismatch")
    expected_native = np.swapaxes(recovar, -1, -2)
    residual = native.astype(np.float64) - expected_native.astype(np.float64)
    return {
        "matrix_count": int(native.shape[0]),
        "element_count": int(native.size),
        "bitwise_equal_count": int(
            np.count_nonzero(native.view(np.uint32) == expected_native.view(np.uint32))
        ),
        "relative_l2": _relative_l2(native, expected_native),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _pixel_weight_shell_stats(
    candidate: np.ndarray,
    reference: np.ndarray,
    score_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    """Summarize a packed-half pixel-weight residual by rounded RELION shell."""

    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    _require(
        candidate.shape == reference.shape == indices.shape,
        "pixel-weight shell topology mismatch",
    )
    height, width = (int(value) for value in image_shape)
    half_width = width // 2 + 1
    rows, x = np.divmod(indices, half_width)
    y = rows - height // 2
    shells = np.rint(np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2)).astype(np.int64)
    active = (candidate > 0.0) & (reference > 0.0)
    _require(np.any(active), "pixel-weight comparison has no active values")
    support_mismatch = (candidate > 0.0) != (reference > 0.0)
    unequal = candidate.view(np.uint32) != reference.view(np.uint32)
    mismatch_positions = np.flatnonzero(unequal)

    shell_rows = []
    for shell in np.unique(shells[active]):
        selected = active & (shells == shell)
        ratios = candidate[selected].astype(np.float64) / reference[selected].astype(np.float64)
        shell_rows.append(
            {
                "shell": int(shell),
                "count": int(np.count_nonzero(selected)),
                "candidate_over_reference_min": float(np.min(ratios)),
                "candidate_over_reference_median": float(np.median(ratios)),
                "candidate_over_reference_max": float(np.max(ratios)),
                "candidate_over_reference_std": float(np.std(ratios)),
            }
        )

    first_mismatch = None
    if mismatch_positions.size:
        position = int(mismatch_positions[0])
        first_mismatch = {
            "score_position": position,
            "full_half_index": int(indices[position]),
            "x": int(x[position]),
            "centered_y": int(y[position]),
            "rounded_shell": int(shells[position]),
            "candidate": float(candidate[position]),
            "reference": float(reference[position]),
            "candidate_bits": f"0x{int(candidate[position].view(np.uint32)):08x}",
            "reference_bits": f"0x{int(reference[position].view(np.uint32)):08x}",
        }
    shell_medians = np.asarray(
        [row["candidate_over_reference_median"] for row in shell_rows],
        dtype=np.float64,
    )
    max_within_shell_span = max(
        row["candidate_over_reference_max"] - row["candidate_over_reference_min"]
        for row in shell_rows
    )
    return {
        "active_count": int(np.count_nonzero(active)),
        "support_mismatch_count": int(np.count_nonzero(support_mismatch)),
        "bitwise_mismatch_count": int(np.count_nonzero(unequal)),
        "first_bitwise_mismatch": first_mismatch,
        "shell_count": len(shell_rows),
        "shell_ratio_median_span": float(np.ptp(shell_medians)),
        "max_within_shell_ratio_span": float(max_within_shell_span),
        "shells": shell_rows,
    }


def _apply_high_shell_retained_fraction(
    pixel_weight: np.ndarray,
    score_indices: np.ndarray,
    image_shape: tuple[int, int],
    *,
    shell_cutoff: int,
    retained_fraction: float,
) -> tuple[np.ndarray, int]:
    """Correct high-shell weights for RELION's unweighted power tail."""

    _require(0.0 < retained_fraction <= 1.0, "retained fraction must be in (0, 1]")
    candidate = np.asarray(pixel_weight, dtype=np.float32).reshape(-1).copy()
    indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    _require(candidate.shape == indices.shape, "high-shell intervention topology mismatch")
    height, width = (int(value) for value in image_shape)
    half_width = width // 2 + 1
    rows, x = np.divmod(indices, half_width)
    y = rows - height // 2
    shells = np.rint(np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2)).astype(np.int64)
    selected = (shells > int(shell_cutoff)) & (candidate > 0.0)
    candidate[selected] = (
        candidate[selected] * np.asarray(retained_fraction, dtype=np.float32)
    ).astype(np.float32)
    return candidate, int(np.count_nonzero(selected))


def _apply_uniform_pixel_weight_factor(pixel_weight: np.ndarray, factor: float) -> np.ndarray:
    """Apply a stopped scale-correction intervention to active score pixels."""

    _require(np.isfinite(factor) and factor > 0.0, "uniform pixel-weight factor must be positive")
    candidate = np.asarray(pixel_weight, dtype=np.float32).reshape(-1).copy()
    active = candidate > 0.0
    candidate[active] = (candidate[active] * np.asarray(factor, dtype=np.float32)).astype(np.float32)
    return candidate


def _apply_shell_factors(
    pixel_weight: np.ndarray,
    score_indices: np.ndarray,
    image_shape: tuple[int, int],
    shell_factors: dict[int, float],
) -> tuple[np.ndarray, int]:
    """Multiply active score weights by explicitly supplied shell factors."""

    candidate = np.asarray(pixel_weight, dtype=np.float32).reshape(-1).copy()
    indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    _require(candidate.shape == indices.shape, "shell-factor intervention topology mismatch")
    height, width = (int(value) for value in image_shape)
    rows, x = np.divmod(indices, width // 2 + 1)
    y = rows - height // 2
    shells = np.rint(np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2)).astype(np.int64)
    corrected = 0
    for shell, factor in shell_factors.items():
        _require(np.isfinite(factor) and factor > 0.0, f"invalid factor for shell {shell}")
        selected = (shells == int(shell)) & (candidate > 0.0)
        candidate[selected] = (candidate[selected] * np.asarray(factor, dtype=np.float32)).astype(np.float32)
        corrected += int(np.count_nonzero(selected))
    return candidate, corrected


def _raw_parent_margin(
    diff2: np.ndarray,
    rotation_ids: np.ndarray,
    *,
    native_parent: tuple[int, int],
    recovar_parent: tuple[int, int],
) -> float:
    """Return native-parent minus RECOVAR-parent raw log-score margin."""

    values = np.asarray(diff2, dtype=np.float32)
    rotations = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    _require(values.ndim == 2 and values.shape[0] == rotations.size, "parent score panel changed")
    rows = {int(rotation): row for row, rotation in enumerate(rotations)}
    _require(len(rows) == rotations.size, "parent score panel contains duplicate rotations")
    native_rotation, native_translation = native_parent
    recovar_rotation, recovar_translation = recovar_parent
    _require(
        native_rotation in rows and recovar_rotation in rows,
        "requested parent rotation is absent from the capture",
    )
    _require(
        0 <= native_translation < values.shape[1]
        and 0 <= recovar_translation < values.shape[1],
        "requested parent translation is out of range",
    )
    return float(
        values[rows[recovar_rotation], recovar_translation]
        - values[rows[native_rotation], native_translation]
    )


def _parse_parent(value: str) -> tuple[int, int]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("parent must be ROTATION,TRANSLATION")
    try:
        parent = (int(fields[0]), int(fields[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("parent must contain integer rows") from error
    if parent[0] < 0 or parent[1] < 0:
        raise argparse.ArgumentTypeError("parent rows must be non-negative")
    return parent


def _authoritative_native_raw_diff2(components, operand, native_coarse) -> np.ndarray:
    """Choose raw scores without treating a rejected component capture as an oracle."""

    if native_coarse is None:
        return components.raw_diff2
    _require(int(native_coarse.header[2]) == operand.stack_index, "coarse stack mismatch")
    _require(int(native_coarse.header[3]) == operand.part_id, "coarse particle mismatch")
    _require(
        native_coarse.raw_diff2.size == components.raw_diff2.size,
        "coarse/component score topology mismatch",
    )
    return native_coarse.raw_diff2.reshape(components.raw_diff2.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument(
        "--native-coarse",
        type=Path,
        help="Authoritative coarse-v1 scores; component capture remains metadata-only",
    )
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native-parent", type=_parse_parent)
    parser.add_argument("--recovar-parent", type=_parse_parent)
    parser.add_argument("--high-shell-cutoff", type=int)
    parser.add_argument("--high-shell-retained-fraction", type=float)
    parser.add_argument("--uniform-pixel-weight-factor", type=float)
    parser.add_argument("--noise-report-json", type=Path)
    parser.add_argument("--noise-shell-max", type=int)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        (args.native_parent is None) == (args.recovar_parent is None),
        "--native-parent and --recovar-parent must be supplied together",
    )
    _require(
        (args.high_shell_cutoff is None) == (args.high_shell_retained_fraction is None),
        "--high-shell-cutoff and --high-shell-retained-fraction must be supplied together",
    )

    _require(jax.default_backend() == "gpu", "counterfactual requires a GPU")
    _require(cuda_backproject.cuda_available(), "custom CUDA extension is unavailable")
    components = load_components(args.components)
    operand = load_operand(args.operands)
    native_coarse = (
        None if args.native_coarse is None else load_native_coarse_capture(args.native_coarse)
    )
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
    _require(args.physical_image_size > 0, "physical image size must be positive")
    image_shape = (args.physical_image_size, args.physical_image_size)
    _require(
        current_size <= args.physical_image_size,
        "current score size exceeds the physical image size",
    )

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
    native_raw_diff2 = _authoritative_native_raw_diff2(
        components, operand, native_coarse
    )
    mapped_native_diff2 = _map_relion_table(
        native_raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float32)
    rotation_ids = np.asarray(
        [_rotation_key_to_recovar(key, n_directions, n_psi) for key in operand.rotation_keys],
        dtype=np.int64,
    )
    rotation_matrix_stats = _rotation_matrix_stats(
        operand.euler_matrices,
        np.asarray(recovar["rotations"], dtype=np.float32)[rotation_ids],
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
    native_unshifted = relion_values_on_recovar_window(
        (operand.image_real.astype(np.float32) + 1j * operand.image_imag.astype(np.float32))[
            np.newaxis,
            :,
        ],
        score_indices,
        full_image_size=image_shape[0],
        current_size=current_size,
    )[0]
    expected_unshifted = (-float(image_shape[0] ** 2) * native_unshifted).astype(np.complex64)
    native_unshifted_translated_via_recovar = np.asarray(
        cuda_backproject.relion_translate_score_f32(
            jnp.asarray(expected_unshifted.reshape(1, -1)),
            jnp.asarray(angles, dtype=jnp.float32),
            jnp.asarray(score_indices, dtype=jnp.int32),
            image_shape,
        ),
        dtype=np.complex64,
    ).reshape(1, translation_source.shape[0], -1)[0]
    native_correction = relion_values_on_recovar_window(
        operand.correction[np.newaxis, :],
        score_indices,
        full_image_size=image_shape[0],
        current_size=current_size,
    )[0].real.astype(np.float32)
    expected_weight = (native_correction / np.float32(image_shape[0] ** 4)).astype(np.float32)

    def score_native_reference(
        image_values: np.ndarray,
        weight_values: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            cuda_backproject.relion_coarse_diff2_rectangular_f32(
                jnp.asarray(native_reference),
                jnp.asarray(image_values, dtype=jnp.complex64).reshape(
                    1,
                    translation_source.shape[0],
                    -1,
                ),
                jnp.asarray(weight_values, dtype=jnp.float32).reshape(1, -1),
                jnp.asarray(initial_diff2),
                jnp.asarray(full_to_compact, dtype=jnp.int32),
            )[0],
            dtype=np.float32,
        )

    native_reference_native_image = score_native_reference(
        expected_shifted,
        pixel_weight,
    )
    native_reference_native_weight = score_native_reference(
        np.asarray(shifted[0]),
        expected_weight,
    )
    native_reference_native_image_and_weight = score_native_reference(
        expected_shifted,
        expected_weight,
    )
    native_reference_native_unshifted_recovar_translation = score_native_reference(
        native_unshifted_translated_via_recovar,
        pixel_weight,
    )
    native_reference_native_unshifted_recovar_translation_and_weight = score_native_reference(
        native_unshifted_translated_via_recovar,
        expected_weight,
    )
    high_shell_corrected_weight = None
    native_reference_high_shell_corrected_weight = None
    high_shell_corrected_count = 0
    if args.high_shell_cutoff is not None:
        high_shell_corrected_weight, high_shell_corrected_count = _apply_high_shell_retained_fraction(
            pixel_weight[0],
            score_indices,
            image_shape,
            shell_cutoff=args.high_shell_cutoff,
            retained_fraction=args.high_shell_retained_fraction,
        )
        native_reference_high_shell_corrected_weight = score_native_reference(
            np.asarray(shifted[0]),
            high_shell_corrected_weight,
        )
    uniform_corrected_weight = None
    native_reference_uniform_corrected_weight = None
    if args.uniform_pixel_weight_factor is not None:
        uniform_corrected_weight = _apply_uniform_pixel_weight_factor(
            pixel_weight[0],
            args.uniform_pixel_weight_factor,
        )
        native_reference_uniform_corrected_weight = score_native_reference(
            np.asarray(shifted[0]),
            uniform_corrected_weight,
        )
    noise_corrected_weight = None
    native_reference_noise_corrected_weight = None
    combined_scale_noise_corrected_weight = None
    native_reference_combined_scale_noise_corrected_weight = None
    noise_corrected_count = 0
    noise_shell_factors = None
    if args.noise_report_json is not None:
        noise_report = json.loads(args.noise_report_json.read_text())
        noise_shell_factors = {
            int(row["shell"]): float(row["recovar_new_relion_units"]) / float(row["native_new"])
            for row in noise_report["shells"]
            if float(row["native_new"]) > 0.0
            and float(row["recovar_new_relion_units"]) > 0.0
            and (args.noise_shell_max is None or int(row["shell"]) <= int(args.noise_shell_max))
        }
        noise_corrected_weight, noise_corrected_count = _apply_shell_factors(
            pixel_weight[0], score_indices, image_shape, noise_shell_factors
        )
        native_reference_noise_corrected_weight = score_native_reference(
            np.asarray(shifted[0]), noise_corrected_weight
        )
        if args.uniform_pixel_weight_factor is not None:
            combined_scale_noise_corrected_weight = _apply_uniform_pixel_weight_factor(
                noise_corrected_weight, args.uniform_pixel_weight_factor
            )
            native_reference_combined_scale_noise_corrected_weight = score_native_reference(
                np.asarray(shifted[0]), combined_scale_noise_corrected_weight
            )

    active_pixels = pixel_weight[0] > 0.0
    shifted_active = np.asarray(shifted[0])[:, active_pixels]
    expected_shifted_active = expected_shifted[:, active_pixels]
    unshifted_active = unshifted[active_pixels]
    expected_unshifted_active = expected_unshifted[active_pixels]
    native_unshifted_translated_active = native_unshifted_translated_via_recovar[:, active_pixels]
    report = {
        "schema": "recovar.em.k1_native_reference_score_counterfactual.v1",
        "particle": {
            "part_id": components.part_id,
            "stack_index_one_based": components.stack_index,
            "original_index_zero_based": int(recovar["original_index"]),
            "physical_image_size": int(args.physical_image_size),
        },
        "rotation_ids_recovar": rotation_ids.tolist(),
        "translation_mapping": translation_report,
        "operand_equality": {
            "rotation_matrices": rotation_matrix_stats,
            "active_unshifted_image": _complex_operand_stats(
                unshifted_active,
                expected_unshifted_active,
            ),
            "active_recovar_translated_image": _complex_operand_stats(
                shifted_active,
                expected_shifted_active,
            ),
            "active_native_unshifted_recovar_translation": _complex_operand_stats(
                native_unshifted_translated_active,
                expected_shifted_active,
            ),
            "pixel_weight_bitwise_equal_count": int(
                np.count_nonzero(pixel_weight[0].view(np.uint32) == expected_weight.view(np.uint32))
            ),
            "pixel_weight_count": int(expected_weight.size),
            "pixel_weight_relative_l2": _relative_l2(pixel_weight[0], expected_weight),
            "pixel_weight_shell_boundary": _pixel_weight_shell_stats(
                pixel_weight[0],
                expected_weight,
                score_indices,
                image_shape,
            ),
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
            "native_reference_native_image_vs_native": _centered_stats(
                native_reference_native_image,
                selected_native_diff2,
            ),
            "native_reference_native_weight_vs_native": _centered_stats(
                native_reference_native_weight,
                selected_native_diff2,
            ),
            "native_reference_native_image_and_weight_vs_native": _centered_stats(
                native_reference_native_image_and_weight,
                selected_native_diff2,
            ),
            "native_reference_native_unshifted_recovar_translation_vs_native": _centered_stats(
                native_reference_native_unshifted_recovar_translation,
                selected_native_diff2,
            ),
            "native_reference_native_unshifted_recovar_translation_and_weight_vs_native": _centered_stats(
                native_reference_native_unshifted_recovar_translation_and_weight,
                selected_native_diff2,
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
    if native_coarse is not None:
        report["artifacts"]["native_coarse"] = str(args.native_coarse.resolve())
        report["artifacts"]["native_coarse_sha256"] = _sha256(args.native_coarse)
        report["artifacts"]["components_semantics"] = (
            "metadata_and_translation_grid_only; raw scores supplied by native_coarse"
        )
    if high_shell_corrected_weight is not None:
        report["high_shell_noise_intervention"] = {
            "shell_cutoff_inclusive": int(args.high_shell_cutoff),
            "retained_fraction": float(args.high_shell_retained_fraction),
            "corrected_pixel_count": high_shell_corrected_count,
        }
        report["score_comparison"]["native_reference_high_shell_corrected_weight_vs_native"] = (
            _centered_stats(
                native_reference_high_shell_corrected_weight,
                selected_native_diff2,
            )
        )
    if uniform_corrected_weight is not None:
        report["uniform_pixel_weight_intervention"] = {
            "factor": float(args.uniform_pixel_weight_factor),
            "corrected_pixel_count": int(np.count_nonzero(pixel_weight[0] > 0.0)),
        }
        report["score_comparison"]["native_reference_uniform_corrected_weight_vs_native"] = (
            _centered_stats(native_reference_uniform_corrected_weight, selected_native_diff2)
        )
    if noise_corrected_weight is not None:
        report["noise_shell_intervention"] = {
            "report": str(args.noise_report_json.resolve()),
            "report_sha256": _sha256(args.noise_report_json),
            "shell_max_inclusive": args.noise_shell_max,
            "shell_factors": {str(key): value for key, value in sorted(noise_shell_factors.items())},
            "corrected_pixel_count": noise_corrected_count,
        }
        report["score_comparison"]["native_reference_noise_corrected_weight_vs_native"] = (
            _centered_stats(native_reference_noise_corrected_weight, selected_native_diff2)
        )
        if native_reference_combined_scale_noise_corrected_weight is not None:
            report["score_comparison"]["native_reference_combined_scale_noise_weight_vs_native"] = (
                _centered_stats(native_reference_combined_scale_noise_corrected_weight, selected_native_diff2)
            )
    if args.native_parent is not None:
        report["parent_margin_counterfactual"] = {
            "margin_semantics": "native_parent_minus_recovar_parent_raw_log_score",
            "native_parent_recovar_coordinates": list(args.native_parent),
            "recovar_parent_recovar_coordinates": list(args.recovar_parent),
            "native": _raw_parent_margin(
                selected_native_diff2,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "recovar_reference": _raw_parent_margin(
                selected_recovar_diff2,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_counterfactual": _raw_parent_margin(
                native_diff2_via_recovar,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_native_image": _raw_parent_margin(
                native_reference_native_image,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_native_weight": _raw_parent_margin(
                native_reference_native_weight,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_native_image_and_weight": _raw_parent_margin(
                native_reference_native_image_and_weight,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_native_unshifted_recovar_translation": _raw_parent_margin(
                native_reference_native_unshifted_recovar_translation,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
            "native_reference_native_unshifted_recovar_translation_and_weight": _raw_parent_margin(
                native_reference_native_unshifted_recovar_translation_and_weight,
                rotation_ids,
                native_parent=args.native_parent,
                recovar_parent=args.recovar_parent,
            ),
        }
        if native_reference_high_shell_corrected_weight is not None:
            report["parent_margin_counterfactual"]["native_reference_high_shell_corrected_weight"] = (
                _raw_parent_margin(
                    native_reference_high_shell_corrected_weight,
                    rotation_ids,
                    native_parent=args.native_parent,
                    recovar_parent=args.recovar_parent,
                )
            )
        if native_reference_uniform_corrected_weight is not None:
            report["parent_margin_counterfactual"]["native_reference_uniform_corrected_weight"] = (
                _raw_parent_margin(
                    native_reference_uniform_corrected_weight,
                    rotation_ids,
                    native_parent=args.native_parent,
                    recovar_parent=args.recovar_parent,
                )
            )
        if native_reference_noise_corrected_weight is not None:
            report["parent_margin_counterfactual"]["native_reference_noise_corrected_weight"] = (
                _raw_parent_margin(
                    native_reference_noise_corrected_weight,
                    rotation_ids,
                    native_parent=args.native_parent,
                    recovar_parent=args.recovar_parent,
                )
            )
        if native_reference_combined_scale_noise_corrected_weight is not None:
            report["parent_margin_counterfactual"]["native_reference_combined_scale_noise_weight"] = (
                _raw_parent_margin(
                    native_reference_combined_scale_noise_corrected_weight,
                    rotation_ids,
                    native_parent=args.native_parent,
                    recovar_parent=args.recovar_parent,
                )
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
