#!/usr/bin/env python3
"""Compare bounded RELION fine-score operands with RECOVAR's exact replay path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.ctf import _compute_spa_ctf
from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _half_translation_phase_table_for_indices,
    _relion_cuda_fine_full_to_compact_lookup,
)
from recovar.em.initial_model.dense_adapter import (
    reference_to_relion_projector_half_maps,
)
from recovar.utils import helpers

if __package__:
    from .compare_k4_relion_recovar_bpref_factors import _compact_indices
    from .validate_relion_fine_operand_capture import (
        _cuda_fine_contribution,
        _reduce_lanes,
        _replay_lanes,
        load_fine_operand_capture,
        validate_capture,
    )
else:
    from compare_k4_relion_recovar_bpref_factors import (  # type: ignore[no-redef]
        _compact_indices,
    )
    from validate_relion_fine_operand_capture import (  # type: ignore[no-redef]
        _cuda_fine_contribution,
        _reduce_lanes,
        _replay_lanes,
        load_fine_operand_capture,
        validate_capture,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(relion: np.ndarray, recovar: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(relion)
    rhs = np.asarray(recovar)
    _require(lhs.shape == rhs.shape, f"operand shape changed: {lhs.shape} != {rhs.shape}")
    promoted_lhs = lhs.astype(np.complex128 if np.iscomplexobj(lhs) else np.float64, copy=False)
    promoted_rhs = rhs.astype(np.complex128 if np.iscomplexobj(rhs) else np.float64, copy=False)
    delta = promoted_rhs - promoted_lhs
    denominator = max(float(np.linalg.norm(promoted_lhs.reshape(-1))), np.finfo(np.float64).tiny)
    return {
        "shape": list(lhs.shape),
        "relion_dtype": str(lhs.dtype),
        "recovar_dtype": str(rhs.dtype),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "mismatch_count": int(np.count_nonzero(lhs != rhs)),
        "relative_l2_over_relion": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _metric_up_to_global_sign(
    relion: np.ndarray,
    recovar: np.ndarray,
) -> dict[str, object]:
    raw = _metric(relion, recovar)
    sign_flipped = _metric(relion, -np.asarray(recovar))
    use_sign_flip = (
        sign_flipped["relative_l2_over_relion"] < raw["relative_l2_over_relion"]
    )
    return {
        "raw": raw,
        "recovar_alignment_multiplier": -1 if use_sign_flip else 1,
        "sign_aligned": sign_flipped if use_sign_flip else raw,
    }


def _zero_dc_compact_score_weight(
    score_weight: np.ndarray,
    compact_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score_weight)
    indices = np.asarray(compact_indices, dtype=np.int64)
    dc_mask = np.asarray(make_shell_indices_half(image_shape)).reshape(-1)[indices] == 0
    result = values.copy()
    result[..., dc_mask] = np.asarray(0, dtype=result.dtype)
    return result, dc_mask


def _infer_current_size(image_size: int) -> int:
    matches = [size for size in range(2, 4097, 2) if size * (size // 2 + 1) == image_size]
    _require(len(matches) == 1, f"cannot infer even current_size from packed image_size={image_size}")
    return matches[0]


def _translation_alignment(
    relion_translation: np.ndarray,
    fine_translations: np.ndarray,
    physical_image_size: int,
) -> tuple[int, float]:
    relion = np.asarray(relion_translation, dtype=np.float64)[:2]
    recovar = -2 * np.pi * np.asarray(fine_translations, dtype=np.float64) / physical_image_size
    error = np.max(np.abs(recovar - relion[None, :]), axis=1)
    index = int(np.argmin(error))
    return index, float(error[index])


def _tree_raw_diff2(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.float32,
) -> tuple[np.float32, np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.complex64)
    shifted = np.asarray(shifted, dtype=np.complex64)
    corr = np.asarray(corr, dtype=np.float32)
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float32)
    contribution = _cuda_fine_contribution(diff_real, diff_imag, corr)
    lanes = _replay_lanes(contribution)
    raw_diff2 = np.float32(_reduce_lanes(lanes) + np.float32(sum_init))
    return raw_diff2, contribution, lanes


def _component_counterfactual(
    relion_raw: np.ndarray,
    all_recovar_raw: np.ndarray,
    substituted_raw: dict[str, np.ndarray],
    *,
    center_deltas: bool = False,
) -> dict[str, object]:
    target_delta = np.asarray(all_recovar_raw, dtype=np.float64) - np.asarray(
        relion_raw, dtype=np.float64
    )
    if center_deltas:
        target_delta = target_delta - np.mean(target_delta)
    baseline_energy = float(np.vdot(target_delta, target_delta).real)
    records = {}
    for name, values in substituted_raw.items():
        component_delta = np.asarray(values, dtype=np.float64) - np.asarray(
            relion_raw, dtype=np.float64
        )
        if center_deltas:
            component_delta = component_delta - np.mean(component_delta)
        residual = target_delta - component_delta
        residual_energy = float(np.vdot(residual, residual).real)
        records[name] = {
            "target_all_recovar_delta_l2": float(np.sqrt(baseline_energy)),
            "after_single_component_substitution_l2": float(np.sqrt(residual_energy)),
            "target_delta_energy_removed_fraction": (
                float(1.0 - residual_energy / baseline_energy) if baseline_energy > 0 else 0.0
            ),
        }
    strongest = max(
        records,
        key=lambda name: records[name]["target_delta_energy_removed_fraction"],
    )
    return {
        "deltas_centered": center_deltas,
        "single_component_substitution": records,
        "strongest_single_component": strongest,
        "strongest_target_delta_energy_removed_fraction": records[strongest][
            "target_delta_energy_removed_fraction"
        ],
    }


def compare(
    capture_path: Path,
    contribution_path: Path,
    reference_path: Path,
    *,
    recovar_global_rotation: int,
    particle_diameter_angstrom: float,
    mask_edge_pixels: float,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "fine-operand comparison requires a JAX GPU")
    capture = load_fine_operand_capture(capture_path)
    validation = validate_capture(capture)
    with np.load(contribution_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    particle_rows = np.flatnonzero(
        np.asarray(values["stack_indices_1based"], dtype=np.int64) == capture.stack_index
    )
    _require(particle_rows.size == 1, "captured stack is not unique in the contribution shard")
    particle = int(particle_rows[0])
    image_shape = tuple(int(value) for value in values["image_shape"])
    physical_image_size = int(image_shape[0])
    _require(
        image_shape == (physical_image_size, physical_image_size),
        f"fine-operand comparison requires square images, got {image_shape}",
    )
    current_size = _infer_current_size(capture.image_size)
    compact_indices = _compact_indices(values)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        compact_indices,
    )
    supported_full = np.flatnonzero(full_to_compact >= 0)
    supported_compact = full_to_compact[supported_full]
    _require(
        np.unique(supported_compact).size == compact_indices.size,
        "RELION full-to-compact score support changed",
    )

    original_index = int(values["original_indices"][particle])
    active_rows = np.flatnonzero(
        (np.asarray(values["active_original_indices"], dtype=np.int64) == original_index)
        & (
            np.asarray(values["active_global_rotation_indices"], dtype=np.int64)
            == recovar_global_rotation
        )
    )
    _require(active_rows.size == 1, "target RECOVAR scoring rotation is not unique")
    recovar_rotation = np.asarray(values["active_rotations"][int(active_rows[0])], dtype=np.float32)

    reference_real = np.asarray(helpers.load_relion_volume(str(reference_path)), dtype=np.float64)
    projector_half, projector_r_max = reference_to_relion_projector_half_maps(
        reference_real[None, ...],
        current_size=current_size,
        padding_factor=int(np.asarray(values["projection_padding_factor"]).item()),
    )
    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half[0]),
        jnp.asarray(recovar_rotation[None, ...]),
        image_shape,
        r_max=int(projector_r_max),
        padding_factor=int(np.asarray(values["projection_padding_factor"]).item()),
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
    )
    projected = np.asarray(jax.block_until_ready(projected), dtype=np.complex64)[0]
    projected_compact = projected[compact_indices]

    raw = jnp.asarray(values["raw_real_images"], dtype=jnp.float32)
    normalization = jnp.asarray(
        values["relion_preprocess_normalization_factors"], dtype=jnp.float32
    )
    integer_shifts = jnp.asarray(values["integer_pre_shifts"], dtype=jnp.int32)
    radius = float(particle_diameter_angstrom) / (
        2.0 * float(np.asarray(values["voxel_size"]).item())
    )
    _, masked = relion_preprocess_real_f32(
        raw,
        normalization,
        integer_shifts,
        radius,
        float(mask_edge_pixels),
        True,
    )
    processed = _centered_rfft2_jax(masked).reshape(masked.shape[0], -1).astype(jnp.complex64)
    ctf = _compute_spa_ctf(
        jnp.asarray(values["ctf_params"], dtype=jnp.float32),
        image_shape,
        float(np.asarray(values["voxel_size"]).item()),
        half_image=True,
    ).astype(jnp.float32)
    compact_device = jnp.asarray(compact_indices, dtype=jnp.int32)
    processed_compact = processed[:, compact_device]
    ctf_compact = ctf[:, compact_device]
    noise_compact = jnp.asarray(values["noise_variance_half"], dtype=jnp.float32)[
        compact_device
    ]
    scale = jnp.asarray(values["scale_corrections"], dtype=jnp.float32)
    base_shifted = processed_compact[particle] / scale[particle]
    base_shifted = jnp.where(
        jnp.abs(ctf_compact[particle]) > jnp.float32(1e-8),
        base_shifted / ctf_compact[particle],
        base_shifted,
    )
    phases = _half_translation_phase_table_for_indices(
        jnp.asarray(values["fine_translations"], dtype=jnp.float32),
        image_shape,
        compact_device,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=True,
        exclude_relion_redundant_x0=True,
    )[compact_device]
    recovar_corr_compact = (
        ctf_compact[particle] ** 2
        / noise_compact
        * scale[particle] ** 2
        * half_weights
    ).astype(jnp.float32)
    base_shifted = np.asarray(jax.block_until_ready(base_shifted), dtype=np.complex64)
    phases = np.asarray(jax.block_until_ready(phases), dtype=np.complex64)
    recovar_corr_compact = np.asarray(
        jax.block_until_ready(recovar_corr_compact), dtype=np.float32
    )
    recovar_corr_compact, compact_dc_mask = _zero_dc_compact_score_weight(
        recovar_corr_compact,
        compact_indices,
        image_shape,
    )

    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    captured_pixels = capture.pixels.reshape(capture.candidates.size, capture.image_size)
    recovar_reference_native_full = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_reference_native_full[supported_full] = (
        projected_compact[supported_compact] / n2
    ).astype(np.complex64)
    recovar_reference_aligned_full = -recovar_reference_native_full
    recovar_corr_full = np.zeros(capture.image_size, dtype=np.float32)
    recovar_corr_full[supported_full] = (
        recovar_corr_compact[supported_compact] * n4
    ).astype(np.float32)

    operands: dict[str, list[np.ndarray]] = {
        f"{name}_{engine}": []
        for name in ("reference", "shifted_image", "corr", "contribution", "lane_partial")
        for engine in ("relion", "recovar")
    }
    candidate_rows = []
    relion_raw = []
    all_recovar_raw = []
    substituted_raw: dict[str, list[np.float32]] = {
        "reference": [],
        "shifted_image": [],
        "corr": [],
    }
    alignment_errors = []
    for target, candidate in enumerate(capture.candidates):
        translation_index, translation_error = _translation_alignment(
            candidate["translation"],
            values["fine_translations"],
            physical_image_size,
        )
        alignment_errors.append(translation_error)
        recovar_shifted_native_full = np.zeros(capture.image_size, dtype=np.complex64)
        recovar_shifted_native_full[supported_full] = (
            base_shifted[supported_compact] * phases[translation_index, supported_compact] / n2
        ).astype(np.complex64)
        recovar_shifted_aligned_full = -recovar_shifted_native_full
        relion_reference = (
            captured_pixels[target]["reference_real"]
            + 1j * captured_pixels[target]["reference_imag"]
        ).astype(np.complex64)
        relion_shifted = (
            captured_pixels[target]["shifted_real"]
            + 1j * captured_pixels[target]["shifted_imag"]
        ).astype(np.complex64)
        relion_corr = np.asarray(captured_pixels[target]["corr"], dtype=np.float32)
        relion_contribution = np.asarray(
            captured_pixels[target]["contribution"], dtype=np.float32
        )
        sum_init = np.float32(candidate["sum_init"])
        recovar_score, recovar_contribution, recovar_lanes = _tree_raw_diff2(
            recovar_reference_aligned_full,
            recovar_shifted_aligned_full,
            recovar_corr_full,
            sum_init,
        )
        reference_score, _, _ = _tree_raw_diff2(
            recovar_reference_aligned_full,
            relion_shifted,
            relion_corr,
            sum_init,
        )
        shifted_score, _, _ = _tree_raw_diff2(
            relion_reference,
            recovar_shifted_aligned_full,
            relion_corr,
            sum_init,
        )
        corr_score, _, _ = _tree_raw_diff2(
            relion_reference,
            relion_shifted,
            recovar_corr_full,
            sum_init,
        )
        relion_raw.append(np.float32(candidate["production_raw_diff2"]))
        all_recovar_raw.append(recovar_score)
        substituted_raw["reference"].append(reference_score)
        substituted_raw["shifted_image"].append(shifted_score)
        substituted_raw["corr"].append(corr_score)
        operands["reference_relion"].append(relion_reference[supported_full] * n2)
        operands["reference_recovar"].append(projected_compact[supported_compact])
        operands["shifted_image_relion"].append(relion_shifted[supported_full] * n2)
        operands["shifted_image_recovar"].append(
            recovar_shifted_native_full[supported_full] * n2
        )
        operands["corr_relion"].append(relion_corr[supported_full] / n4)
        operands["corr_recovar"].append(recovar_corr_compact[supported_compact])
        operands["contribution_relion"].append(relion_contribution)
        operands["contribution_recovar"].append(recovar_contribution)
        operands["lane_partial_relion"].append(candidate["lane_partials"])
        operands["lane_partial_recovar"].append(recovar_lanes)
        candidate_rows.append(
            {
                "target_index": target,
                "translation_id_relion": int(candidate["translation_id"]),
                "translation_index_recovar": translation_index,
                "translation_alignment_max_abs": translation_error,
                "relion_raw_diff2": float(candidate["production_raw_diff2"]),
                "recovar_operand_replay_raw_diff2_with_relion_sum_init": float(
                    recovar_score
                ),
                "raw_diff2_delta_recovar_minus_relion": float(
                    np.float64(recovar_score)
                    - np.float64(candidate["production_raw_diff2"])
                ),
            }
        )
    _require(max(alignment_errors) <= 1e-6, "fine translation alignment changed")

    metrics = {}
    for name in ("reference", "shifted_image", "corr", "contribution", "lane_partial"):
        relion_operand = np.concatenate(operands[f"{name}_relion"])
        recovar_operand = np.concatenate(operands[f"{name}_recovar"])
        metrics[name] = (
            _metric_up_to_global_sign(relion_operand, recovar_operand)
            if name in {"reference", "shifted_image"}
            else _metric(relion_operand, recovar_operand)
        )
    relion_raw_array = np.asarray(relion_raw, dtype=np.float32)
    all_recovar_raw_array = np.asarray(all_recovar_raw, dtype=np.float32)
    raw_delta = all_recovar_raw_array.astype(np.float64) - relion_raw_array.astype(
        np.float64
    )
    raw_delta_centered = raw_delta - np.mean(raw_delta)
    for row, centered_delta in zip(candidate_rows, raw_delta_centered, strict=True):
        row["raw_diff2_delta_centered"] = float(centered_delta)
        row["implied_centered_data_score_delta_recovar_minus_relion"] = float(
            -centered_delta
        )

    relion_corr_supported = np.asarray(
        captured_pixels[0]["corr"][supported_full], dtype=np.float32
    ) / n4
    recovar_corr_supported = recovar_corr_compact[supported_compact]
    corr_delta = recovar_corr_supported.astype(np.float64) - relion_corr_supported.astype(
        np.float64
    )
    largest_corr_delta_row = int(np.argmax(np.abs(corr_delta)))
    dc_compact_rows = np.flatnonzero(compact_dc_mask)
    _require(dc_compact_rows.size == 1, "compact score support must contain one DC pixel")
    dc_compact_row = int(dc_compact_rows[0])
    dc_supported_rows = np.flatnonzero(supported_compact == dc_compact_row)
    _require(dc_supported_rows.size == 1, "RELION score support must contain one DC pixel")
    dc_supported_row = int(dc_supported_rows[0])
    dc_full_row = int(supported_full[dc_supported_row])

    raw_counterfactual = _component_counterfactual(
        relion_raw_array,
        all_recovar_raw_array,
        {
            name: np.asarray(records, dtype=np.float32)
            for name, records in substituted_raw.items()
        },
    )
    centered_counterfactual = _component_counterfactual(
        relion_raw_array,
        all_recovar_raw_array,
        {
            name: np.asarray(records, dtype=np.float32)
            for name, records in substituted_raw.items()
        },
        center_deltas=True,
    )
    rotation_direct = _metric(capture.candidates[0]["matrix"], recovar_rotation.reshape(-1))
    rotation_transpose = _metric(
        capture.candidates[0]["matrix"], recovar_rotation.T.reshape(-1)
    )
    classification = (
        f"{centered_counterfactual['strongest_single_component']}"
        "_dominates_centered_fine_operand_residual"
    )
    return {
        "schema": "k4_relion_recovar_fine_operand_comparison_v3",
        "status": "complete",
        "classification": classification,
        "capture_validation": validation,
        "inputs": {
            "capture": str(capture_path.resolve()),
            "capture_sha256": _sha256(capture_path),
            "contribution": str(contribution_path.resolve()),
            "contribution_sha256": _sha256(contribution_path),
            "reference": str(reference_path.resolve()),
            "reference_sha256": _sha256(reference_path),
        },
        "scope": {
            "stack_index_one_based": capture.stack_index,
            "particle_row_in_contribution_shard": particle,
            "original_index_zero_based": original_index,
            "recovar_global_rotation": recovar_global_rotation,
            "relion_rotation_local": int(capture.candidates[0]["rotation_local"]),
            "physical_image_size": physical_image_size,
            "current_size": current_size,
            "compact_pixel_count": int(compact_indices.size),
            "relion_full_pixel_count": capture.image_size,
            "projector_r_max": int(projector_r_max),
            "particle_diameter_angstrom": particle_diameter_angstrom,
            "mask_edge_pixels": mask_edge_pixels,
            "rotation_matrix_direct": rotation_direct,
            "rotation_matrix_transpose": rotation_transpose,
            "max_translation_alignment_abs": max(alignment_errors),
        },
        "fourier_sign_convention": {
            "description": (
                "RELION and RECOVAR reference/image operands use opposite global "
                "Fourier signs; both operands flip, so squared score contributions "
                "are invariant."
            ),
            "reference_recovar_alignment_multiplier": metrics["reference"][
                "recovar_alignment_multiplier"
            ],
            "shifted_image_recovar_alignment_multiplier": metrics["shifted_image"][
                "recovar_alignment_multiplier"
            ],
        },
        "dc_score_weight": {
            "production_rule": "zero shell zero before direct Gaussian scoring",
            "compact_row": dc_compact_row,
            "relion_full_row": dc_full_row,
            "relion_scaled_value": float(relion_corr_supported[dc_supported_row]),
            "recovar_scaled_value_after_production_dc_zero": float(
                recovar_corr_supported[dc_supported_row]
            ),
            "largest_remaining_abs_delta": float(
                np.abs(corr_delta[largest_corr_delta_row])
            ),
            "largest_remaining_delta_relion_full_row": int(
                supported_full[largest_corr_delta_row]
            ),
        },
        "operands": metrics,
        "raw_diff2_delta_centering": {
            "mean_raw_delta_recovar_minus_relion": float(np.mean(raw_delta)),
            "max_abs_centered_raw_delta": float(
                np.max(np.abs(raw_delta_centered), initial=0.0)
            ),
            "sign_relation": (
                "centered Gaussian data score is the negative of centered raw diff2"
            ),
        },
        "raw_diff2_component_counterfactual": raw_counterfactual,
        "centered_raw_diff2_component_counterfactual": centered_counterfactual,
        "candidates": candidate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--contribution", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recovar-global-rotation", type=int, required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(
        args.capture,
        args.contribution,
        args.reference,
        recovar_global_rotation=args.recovar_global_rotation,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        mask_edge_pixels=args.mask_edge_pixels,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
