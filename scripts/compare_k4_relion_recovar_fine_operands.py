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
from recovar.data_io.image_backends import (
    _apply_relion_soft_image_mask_numpy,
    _centered_rfft2_jax,
    _centered_rfft2_numpy,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _half_translation_phase_table_for_indices,
    _relion_cuda_fine_diff2_sum,
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_cuda_fine_pixel_weights,
)

if __package__:
    from .validate_relion_fine_operand_capture import (
        _cuda_fine_contribution,
        _reduce_lanes,
        _replay_lanes,
        load_fine_operand_capture,
        validate_capture,
    )
else:
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


def _direct_score_image_factor(
    *,
    relion_cuda_preprocess: bool,
    image_correction: np.float32,
    scale_correction: np.float32,
) -> np.float32:
    scale = np.float32(scale_correction)
    factor = np.float32(1.0) / scale
    if not relion_cuda_preprocess:
        factor = np.float32(
            factor * np.float32(np.float32(image_correction) / scale)
        )
    return factor


def _relion_cuda_normalization_factors(
    values: dict[str, np.ndarray],
    *,
    captured_backend_is_relion_cuda: bool,
) -> np.ndarray:
    if captured_backend_is_relion_cuda:
        return np.asarray(
            values["relion_preprocess_normalization_factors"], dtype=np.float32
        )
    image_correction = np.asarray(values["image_corrections"], dtype=np.float32)
    scale_correction = np.asarray(values["scale_corrections"], dtype=np.float32)
    return np.asarray(image_correction / scale_correction, dtype=np.float32)


def _is_relion_cuda_replay_mode(mode: str) -> bool:
    return mode in {"relion_cuda", "relion_cuda_native_lane"}


def _preprocess_normalization_source(
    mode: str, *, captured_backend_is_relion_cuda: bool
) -> str:
    if not _is_relion_cuda_replay_mode(mode):
        return "not_applied"
    if captured_backend_is_relion_cuda:
        return "captured"
    return "derived_image_correction_over_scale"


def _reconstruct_processed_score_half(
    values: dict[str, np.ndarray],
    *,
    particle_diameter_angstrom: float,
    mask_edge_pixels: float,
    mode_override: str | None = None,
) -> tuple[np.ndarray | jax.Array, str]:
    raw = np.asarray(values["raw_real_images"], dtype=np.float32)
    captured_normalization = np.asarray(
        values["relion_preprocess_normalization_factors"], dtype=np.float32
    )
    integer_shifts = np.asarray(values["integer_pre_shifts"], dtype=np.int32)
    relion_cuda = bool(np.asarray(values["relion_cuda_preprocess"]).item())
    captured_native_lane = bool(
        np.asarray(values.get("relion_native_lane_reduction", False)).item()
    )
    backend = str(np.asarray(values["preprocess_backend"]).item())
    _require(
        relion_cuda == (backend == "relion_cuda"),
        "RECOVAR preprocessing flag and backend disagree",
    )
    _require(
        not captured_native_lane or relion_cuda,
        "native-lane capture requires RELION CUDA preprocessing",
    )
    captured_mode = "relion_cuda_native_lane" if captured_native_lane else backend
    mode = captured_mode if mode_override is None else mode_override
    _require(
        mode
        in {
            "dataset_native",
            "dataset_native_jax_fft",
            "relion_cuda",
            "relion_cuda_native_lane",
        },
        f"unsupported preprocessing replay mode {mode!r}",
    )
    score_with_mask = bool(np.asarray(values["score_with_masked_images"]).item())
    if _is_relion_cuda_replay_mode(mode):
        normalization = _relion_cuda_normalization_factors(
            values,
            captured_backend_is_relion_cuda=relion_cuda,
        )
        radius = float(particle_diameter_angstrom) / (
            2.0 * float(np.asarray(values["voxel_size"]).item())
        )
        _, processed_real = relion_preprocess_real_f32(
            jnp.asarray(raw),
            jnp.asarray(normalization),
            jnp.asarray(integer_shifts),
            radius,
            float(mask_edge_pixels),
            score_with_mask,
            native_lane_reduction=(mode == "relion_cuda_native_lane"),
        )
        processed = _centered_rfft2_jax(processed_real)
        return processed.reshape(processed.shape[0], -1).astype(jnp.complex64), mode

    _require(
        relion_cuda
        or np.array_equal(
            captured_normalization, np.ones_like(captured_normalization)
        ),
        "dataset-native capture unexpectedly stored active RELION normalization",
    )
    processed_real = apply_relion_integer_pre_shifts(raw, integer_shifts)
    if score_with_mask:
        image_mask = np.asarray(values["image_mask"], dtype=np.float32)
        mask_mode = str(np.asarray(values["image_mask_mode"]).item())
        if mask_mode == "relion_background_fill":
            processed_real = _apply_relion_soft_image_mask_numpy(
                processed_real,
                image_mask,
            )
        elif mask_mode == "multiply":
            processed_real = processed_real * image_mask[None, :, :]
        else:
            raise ValueError(f"unsupported captured image mask mode {mask_mode!r}")
    if mode == "dataset_native_jax_fft":
        processed = _centered_rfft2_jax(jnp.asarray(processed_real, dtype=jnp.float32))
        return processed.reshape(processed.shape[0], -1).astype(jnp.complex64), mode
    processed = _centered_rfft2_numpy(processed_real)
    return processed.reshape(processed.shape[0], -1).astype(np.complex64), mode


def _infer_current_size(image_size: int) -> int:
    matches = [size for size in range(2, 4097, 2) if size * (size // 2 + 1) == image_size]
    _require(len(matches) == 1, f"cannot infer even current_size from packed image_size={image_size}")
    return matches[0]


def _compact_indices_from_full_lookup(
    image_shape: tuple[int, int],
    current_size: int,
    full_to_compact: np.ndarray,
) -> np.ndarray:
    """Invert RECOVAR's score lookup into centered physical half-image rows."""

    lookup = np.asarray(full_to_compact, dtype=np.int32).reshape(-1)
    current_half_width = int(current_size) // 2 + 1
    _require(
        lookup.shape == (int(current_size) * current_half_width,),
        "raw score full-to-compact lookup shape changed",
    )
    valid = lookup >= 0
    compact_count = int(np.count_nonzero(valid))
    _require(
        np.array_equal(np.sort(lookup[valid]), np.arange(compact_count)),
        "raw score full-to-compact lookup is not a complete gather",
    )
    full_rows = np.flatnonzero(valid)
    fftw_rows = full_rows // current_half_width
    columns = full_rows % current_half_width
    ky = np.where(
        fftw_rows < int(current_size) // 2,
        fftw_rows,
        fftw_rows - int(current_size),
    )
    physical_half_width = int(image_shape[1]) // 2 + 1
    centered_rows = ky + int(image_shape[0]) // 2
    _require(
        np.all((centered_rows >= 0) & (centered_rows < int(image_shape[0]))),
        "raw score lookup contains rows outside the physical image",
    )
    compact_indices = np.empty(compact_count, dtype=np.int32)
    compact_indices[lookup[valid]] = (
        centered_rows * physical_half_width + columns
    ).astype(np.int32)
    roundtrip = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        compact_indices,
    )
    _require(
        np.array_equal(roundtrip, lookup),
        "raw score full-to-compact gather direction changed",
    )
    return compact_indices


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


@jax.jit
def _jax_tree_raw_diff2_device(
    reference,
    shifted,
    corr,
    sum_init,
    relion_full_to_compact,
):
    raw = _relion_cuda_fine_diff2_sum(
        reference,
        shifted,
        corr,
        relion_full_to_compact,
    )
    return raw + sum_init


def _jax_tree_raw_diff2(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.ndarray,
    relion_full_to_compact: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate RECOVAR's fused JAX/XLA tree on aligned native operands."""

    if relion_full_to_compact is None:
        relion_full_to_compact = np.arange(reference.shape[-1], dtype=np.int32)
    raw = _jax_tree_raw_diff2_device(
        jnp.asarray(reference, dtype=jnp.complex64),
        jnp.asarray(shifted, dtype=jnp.complex64),
        jnp.asarray(corr, dtype=jnp.float32),
        jnp.asarray(sum_init, dtype=jnp.float32),
        jnp.asarray(relion_full_to_compact, dtype=jnp.int32),
    )
    return np.asarray(jax.block_until_ready(raw), dtype=np.float32)


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
    strongest_fraction = max(
        record["target_delta_energy_removed_fraction"]
        for record in records.values()
    )
    strongest_components = [
        name
        for name, record in records.items()
        if record["target_delta_energy_removed_fraction"] == strongest_fraction
    ]
    strongest = strongest_components[0]
    return {
        "deltas_centered": center_deltas,
        "target_all_recovar_delta_l2": float(np.sqrt(baseline_energy)),
        "informative": baseline_energy > 0,
        "single_component_substitution": records,
        "strongest_single_component": strongest,
        "strongest_components": strongest_components,
        "strongest_is_unique": len(strongest_components) == 1,
        "strongest_target_delta_energy_removed_fraction": strongest_fraction,
    }


def _select_component_classification(
    raw_counterfactual: dict[str, object],
    centered_counterfactual: dict[str, object],
    *,
    candidate_count: int,
) -> tuple[str, str]:
    """Select component attribution without centering away one sample."""

    if candidate_count >= 2 and centered_counterfactual["informative"]:
        if (
            centered_counterfactual[
                "strongest_target_delta_energy_removed_fraction"
            ]
            <= 0
        ):
            return (
                "no_single_component_reduces_centered_fine_operand_residual",
                "centered_raw_diff2",
            )
        if not centered_counterfactual["strongest_is_unique"]:
            return (
                "multiple_fine_operand_components_tie_for_largest_centered_"
                "single_substitution_effect",
                "centered_raw_diff2",
            )
        component = centered_counterfactual["strongest_single_component"]
        return (
            f"{component}_has_largest_centered_fine_operand_single_substitution_effect",
            "centered_raw_diff2",
        )
    if raw_counterfactual["informative"]:
        if raw_counterfactual["strongest_target_delta_energy_removed_fraction"] <= 0:
            return (
                "no_single_component_reduces_raw_fine_operand_residual",
                "raw_diff2",
            )
        if not raw_counterfactual["strongest_is_unique"]:
            return (
                "multiple_fine_operand_components_tie_for_largest_raw_"
                "single_substitution_effect",
                "raw_diff2",
            )
        component = raw_counterfactual["strongest_single_component"]
        return (
            f"{component}_has_largest_raw_fine_operand_single_substitution_effect",
            "raw_diff2",
        )
    return "no_nonzero_fine_operand_residual", "none"


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - np.mean(array)


def compare(
    capture_path: Path,
    pass2_path: Path,
    contribution_path: Path,
    *,
    recovar_global_rotation: int,
    particle_diameter_angstrom: float,
    mask_edge_pixels: float,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "fine-operand comparison requires a JAX GPU")
    capture = load_fine_operand_capture(capture_path)
    validation = validate_capture(capture)
    with np.load(pass2_path, allow_pickle=False) as archive:
        score_values = {name: np.asarray(archive[name]) for name in archive.files}
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
    _require(
        int(np.asarray(score_values["current_size"]).item()) == current_size,
        "RECOVAR pass-2 and native current sizes differ",
    )
    full_to_compact = np.asarray(
        score_values["raw_operand_relion_full_to_compact"],
        dtype=np.int32,
    )
    compact_indices = _compact_indices_from_full_lookup(
        image_shape,
        current_size,
        full_to_compact,
    )
    supported_full = np.flatnonzero(full_to_compact >= 0)
    supported_compact = full_to_compact[supported_full]
    _require(
        np.unique(supported_compact).size == compact_indices.size,
        "RELION full-to-compact score support changed",
    )

    original_index = int(np.asarray(score_values["original_index"]).item())
    _require(
        int(values["original_indices"][particle]) == original_index,
        "RECOVAR pass-2 and contribution particle identities differ",
    )
    _require(
        int(np.asarray(score_values["class_index"]).item()) == 0,
        "fine-operand comparator requires the admitted RECOVAR class-1 shard",
    )
    recovar_rotation_rows = np.flatnonzero(
        np.asarray(score_values["oversampled_rot_indices"], dtype=np.int64)
        == recovar_global_rotation
    )
    _require(
        recovar_rotation_rows.size == 1,
        "target RECOVAR pass-2 scoring rotation is not unique",
    )
    recovar_rotation_local = int(recovar_rotation_rows[0])
    recovar_rotation = np.asarray(
        score_values["rotations"][recovar_rotation_local],
        dtype=np.float32,
    )
    actual_rotation_count = int(
        np.asarray(score_values["raw_operand_actual_rotation_count"]).item()
    )
    _require(
        actual_rotation_count == score_values["oversampled_rot_indices"].size
        and recovar_rotation_local < actual_rotation_count,
        "RECOVAR raw projection rotation axis changed",
    )
    raw_projection = np.asarray(
        score_values["raw_operand_proj_half"],
        dtype=np.complex64,
    )
    raw_shifted_corrected = np.asarray(
        score_values["raw_operand_shifted_corrected"],
        dtype=np.complex64,
    )
    raw_corr_img_score = np.asarray(
        score_values["raw_operand_corr_img_score"],
        dtype=np.float32,
    )
    raw_half_weights = np.asarray(
        score_values["raw_operand_half_weights"],
        dtype=np.float32,
    )
    _require(
        raw_projection.ndim == 2
        and raw_projection.shape[0] >= actual_rotation_count
        and raw_projection.shape[1] == compact_indices.size
        and raw_shifted_corrected.shape
        == (score_values["fine_translations"].shape[0], compact_indices.size)
        and raw_corr_img_score.shape == (compact_indices.size,)
        and raw_half_weights.shape == (compact_indices.size,),
        "RECOVAR pass-2 raw operand shapes changed",
    )
    recovar_reference_compact = raw_projection[recovar_rotation_local]
    recovar_score_weight_compact = np.asarray(
        jax.block_until_ready(
            _relion_cuda_fine_pixel_weights(
                jnp.asarray(raw_corr_img_score),
                jnp.asarray(raw_half_weights),
            )
        ),
        dtype=np.float32,
    )
    recovar_score_weight_compact, compact_dc_mask = (
        _zero_dc_compact_score_weight(
            recovar_score_weight_compact,
            compact_indices,
            image_shape,
        )
    )
    recovar_sum_init = np.float32(
        np.asarray(score_values["raw_operand_highres_xi2_half"]).item()
    )

    processed, preprocess_backend = _reconstruct_processed_score_half(
        values,
        particle_diameter_angstrom=particle_diameter_angstrom,
        mask_edge_pixels=mask_edge_pixels,
    )
    ctf = _compute_spa_ctf(
        jnp.asarray(values["ctf_params"], dtype=jnp.float32),
        image_shape,
        float(np.asarray(values["voxel_size"]).item()),
        half_image=True,
    ).astype(jnp.float32)
    compact_device = jnp.asarray(compact_indices, dtype=jnp.int32)
    processed_compact = jnp.asarray(processed)[:, compact_device]
    ctf_compact = ctf[:, compact_device]
    noise_compact = jnp.asarray(values["noise_variance_half"], dtype=jnp.float32)[
        compact_device
    ]
    scale = np.asarray(values["scale_corrections"], dtype=np.float32)
    image_correction = np.asarray(values["image_corrections"], dtype=np.float32)
    direct_score_factor = _direct_score_image_factor(
        relion_cuda_preprocess=bool(
            np.asarray(values["relion_cuda_preprocess"]).item()
        ),
        image_correction=np.float32(image_correction[particle]),
        scale_correction=np.float32(scale[particle]),
    )
    base_shifted = processed_compact[particle] * jnp.asarray(
        direct_score_factor, dtype=jnp.float32
    )
    base_shifted = jnp.where(
        jnp.abs(ctf_compact[particle]) > jnp.float32(1e-8),
        base_shifted / ctf_compact[particle],
        base_shifted,
    )
    phases = _half_translation_phase_table_for_indices(
        jnp.asarray(score_values["fine_translations"], dtype=jnp.float32),
        image_shape,
        compact_device,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=True,
        exclude_relion_redundant_x0=True,
    )[compact_device]
    reconstructed_corr_compact = (
        ctf_compact[particle] ** 2
        / noise_compact
        * jnp.asarray(scale[particle], dtype=jnp.float32) ** 2
        * half_weights
    ).astype(jnp.float32)
    base_shifted = np.asarray(jax.block_until_ready(base_shifted), dtype=np.complex64)
    phases = np.asarray(jax.block_until_ready(phases), dtype=np.complex64)
    reconstructed_corr_compact = np.asarray(
        jax.block_until_ready(reconstructed_corr_compact), dtype=np.float32
    )
    reconstructed_corr_compact, _ = _zero_dc_compact_score_weight(
        reconstructed_corr_compact,
        compact_indices,
        image_shape,
    )

    preprocessing_counterfactuals: dict[str, dict[str, object]] = {}
    for mode in (
        "dataset_native_jax_fft",
        "relion_cuda",
        "relion_cuda_native_lane",
    ):
        alternate_processed, _ = _reconstruct_processed_score_half(
            values,
            particle_diameter_angstrom=particle_diameter_angstrom,
            mask_edge_pixels=mask_edge_pixels,
            mode_override=mode,
        )
        alternate_factor = _direct_score_image_factor(
            relion_cuda_preprocess=_is_relion_cuda_replay_mode(mode),
            image_correction=np.float32(image_correction[particle]),
            scale_correction=np.float32(scale[particle]),
        )
        alternate_base = (
            jnp.asarray(alternate_processed)[particle, compact_device]
            * jnp.asarray(alternate_factor, dtype=jnp.float32)
        )
        alternate_base = jnp.where(
            jnp.abs(ctf_compact[particle]) > jnp.float32(1e-8),
            alternate_base / ctf_compact[particle],
            alternate_base,
        )
        preprocessing_counterfactuals[mode] = {
            "direct_score_image_factor": float(alternate_factor),
            "normalization_source": _preprocess_normalization_source(
                mode,
                captured_backend_is_relion_cuda=bool(
                    np.asarray(values["relion_cuda_preprocess"]).item()
                ),
            ),
            "base_shifted": np.asarray(
                jax.block_until_ready(alternate_base), dtype=np.complex64
            ),
            "raw_diff2": [],
            "shifted_image_relion": [],
            "shifted_image_counterfactual": [],
        }

    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    captured_pixels = capture.pixels.reshape(capture.candidates.size, capture.image_size)
    recovar_reference_native_full = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_reference_native_full[supported_full] = (
        recovar_reference_compact[supported_compact] / n2
    ).astype(np.complex64)
    recovar_reference_aligned_full = -recovar_reference_native_full
    recovar_corr_full = np.zeros(capture.image_size, dtype=np.float32)
    recovar_corr_full[supported_full] = (
        recovar_score_weight_compact[supported_compact] * n4
    ).astype(np.float32)

    operands: dict[str, list[np.ndarray]] = {
        f"{name}_{engine}": []
        for name in ("reference", "shifted_image", "corr", "contribution", "lane_partial")
        for engine in ("relion", "recovar")
    }
    candidate_rows = []
    relion_raw = []
    all_recovar_raw = []
    recovar_operand_replay_raw = []
    substituted_raw: dict[str, list[np.float32]] = {
        "reference": [],
        "shifted_image": [],
        "corr": [],
        "highres_xi2": [],
    }
    pair_mask = np.asarray(score_values["raw_operand_pair_mask"], dtype=bool)
    pair_rotation_row = np.asarray(
        score_values["raw_operand_pair_rotation_row"],
        dtype=np.int32,
    )
    pair_translation_idx = np.asarray(
        score_values["raw_operand_pair_translation_idx"],
        dtype=np.int32,
    )
    pair_raw_diff2 = np.asarray(
        score_values["raw_operand_raw_diff2"],
        dtype=np.float32,
    )
    _require(
        pair_mask.shape
        == pair_rotation_row.shape
        == pair_translation_idx.shape
        == pair_raw_diff2.shape,
        "RECOVAR raw compact-pair arrays changed shape",
    )
    alignment_errors = []
    for target, candidate in enumerate(capture.candidates):
        translation_index, translation_error = _translation_alignment(
            candidate["translation"],
            score_values["fine_translations"],
            physical_image_size,
        )
        alignment_errors.append(translation_error)
        recovar_shifted_native_full = np.zeros(capture.image_size, dtype=np.complex64)
        recovar_shifted_native_full[supported_full] = (
            raw_shifted_corrected[translation_index, supported_compact] / n2
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
        native_sum_init = np.float32(candidate["sum_init"])
        recovar_score, recovar_contribution, recovar_lanes = _tree_raw_diff2(
            recovar_reference_aligned_full,
            recovar_shifted_aligned_full,
            recovar_corr_full,
            recovar_sum_init,
        )
        reference_score, _, _ = _tree_raw_diff2(
            recovar_reference_aligned_full,
            relion_shifted,
            relion_corr,
            native_sum_init,
        )
        shifted_score, _, _ = _tree_raw_diff2(
            relion_reference,
            recovar_shifted_aligned_full,
            relion_corr,
            native_sum_init,
        )
        corr_score, _, _ = _tree_raw_diff2(
            relion_reference,
            relion_shifted,
            recovar_corr_full,
            native_sum_init,
        )
        highres_score, _, _ = _tree_raw_diff2(
            relion_reference,
            relion_shifted,
            relion_corr,
            recovar_sum_init,
        )
        for counterfactual in preprocessing_counterfactuals.values():
            alternate_shifted_native_full = np.zeros(
                capture.image_size, dtype=np.complex64
            )
            alternate_shifted_native_full[supported_full] = (
                np.asarray(counterfactual["base_shifted"])[supported_compact]
                * phases[translation_index, supported_compact]
                / n2
            ).astype(np.complex64)
            alternate_shifted_aligned_full = -alternate_shifted_native_full
            alternate_score, _, _ = _tree_raw_diff2(
                relion_reference,
                alternate_shifted_aligned_full,
                relion_corr,
                native_sum_init,
            )
            counterfactual["raw_diff2"].append(alternate_score)
            counterfactual["shifted_image_relion"].append(
                relion_shifted[supported_full] * n2
            )
            counterfactual["shifted_image_counterfactual"].append(
                alternate_shifted_native_full[supported_full] * n2
            )
        relion_raw.append(np.float32(candidate["production_raw_diff2"]))
        recovar_production_raw = np.float32(
            score_values["relion_raw_diff2"][
                recovar_rotation_local,
                translation_index,
            ]
        )
        selected_pair = np.flatnonzero(
            pair_mask
            & (pair_rotation_row == recovar_rotation_local)
            & (pair_translation_idx == translation_index)
        )
        _require(
            selected_pair.size == 1,
            "target RECOVAR raw compact-pair tuple is not unique",
        )
        _require(
            pair_raw_diff2[int(selected_pair[0])] == recovar_production_raw,
            "RECOVAR raw compact-pair value and dense score table differ",
        )
        all_recovar_raw.append(recovar_production_raw)
        recovar_operand_replay_raw.append(recovar_score)
        substituted_raw["reference"].append(reference_score)
        substituted_raw["shifted_image"].append(shifted_score)
        substituted_raw["corr"].append(corr_score)
        substituted_raw["highres_xi2"].append(highres_score)
        operands["reference_relion"].append(relion_reference[supported_full] * n2)
        operands["reference_recovar"].append(
            recovar_reference_compact[supported_compact]
        )
        operands["shifted_image_relion"].append(relion_shifted[supported_full] * n2)
        operands["shifted_image_recovar"].append(
            recovar_shifted_native_full[supported_full] * n2
        )
        operands["corr_relion"].append(relion_corr[supported_full] / n4)
        operands["corr_recovar"].append(
            recovar_score_weight_compact[supported_compact]
        )
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
                "recovar_production_raw_diff2": float(recovar_production_raw),
                "recovar_operand_replay_raw_diff2_with_recovar_sum_init": float(
                    recovar_score
                ),
                "raw_diff2_delta_recovar_minus_relion": float(
                    np.float64(recovar_production_raw)
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
    recovar_operand_replay_raw_array = np.asarray(
        recovar_operand_replay_raw,
        dtype=np.float32,
    )
    native_reference_full = (
        captured_pixels["reference_real"]
        + np.complex64(1j) * captured_pixels["reference_imag"]
    ).astype(np.complex64)
    native_shifted_full = (
        captured_pixels["shifted_real"]
        + np.complex64(1j) * captured_pixels["shifted_imag"]
    ).astype(np.complex64)
    native_corr_full = np.asarray(captured_pixels["corr"], dtype=np.float32)
    native_contribution_full = np.asarray(
        captured_pixels["contribution"],
        dtype=np.float32,
    )
    unsupported_full = full_to_compact < 0
    _require(
        not np.any(native_corr_full[:, unsupported_full])
        and not np.any(native_contribution_full[:, unsupported_full]),
        "native score support contains nonzero terms outside RECOVAR's raw score lookup",
    )
    native_reference_compact = np.zeros(
        (capture.candidates.size, compact_indices.size),
        dtype=np.complex64,
    )
    native_shifted_compact = np.zeros_like(native_reference_compact)
    native_corr_compact = np.zeros(
        (capture.candidates.size, compact_indices.size),
        dtype=np.float32,
    )
    native_reference_compact[:, supported_compact] = native_reference_full[
        :, supported_full
    ]
    native_shifted_compact[:, supported_compact] = native_shifted_full[
        :, supported_full
    ]
    native_corr_compact[:, supported_compact] = native_corr_full[:, supported_full]
    native_sum_init = np.asarray(
        [candidate["sum_init"] for candidate in capture.candidates],
        dtype=np.float32,
    )
    jax_arithmetic_native_raw = _jax_tree_raw_diff2(
        native_reference_compact,
        native_shifted_compact,
        native_corr_compact,
        native_sum_init,
        full_to_compact,
    )
    substitution_arrays = {
        name: np.asarray(records, dtype=np.float32)
        for name, records in substituted_raw.items()
    }
    substitution_arrays["jax_arithmetic_on_native_operands"] = (
        jax_arithmetic_native_raw
    )
    raw_delta = all_recovar_raw_array.astype(np.float64) - relion_raw_array.astype(
        np.float64
    )
    raw_delta_centered = _center(raw_delta)
    jax_arithmetic_delta = (
        jax_arithmetic_native_raw.astype(np.float64)
        - relion_raw_array.astype(np.float64)
    )
    jax_arithmetic_delta_centered = _center(jax_arithmetic_delta)
    for row, centered_delta, arithmetic_raw, arithmetic_delta, arithmetic_centered in zip(
        candidate_rows,
        raw_delta_centered,
        jax_arithmetic_native_raw,
        jax_arithmetic_delta,
        jax_arithmetic_delta_centered,
        strict=True,
    ):
        row["raw_diff2_delta_centered"] = float(centered_delta)
        row["implied_centered_data_score_delta_recovar_minus_relion"] = float(
            -centered_delta
        )
        row["jax_arithmetic_on_native_operands_raw_diff2"] = float(arithmetic_raw)
        row["jax_arithmetic_on_native_operands_delta"] = float(arithmetic_delta)
        row["jax_arithmetic_on_native_operands_delta_centered"] = float(
            arithmetic_centered
        )

    relion_corr_supported = np.asarray(
        captured_pixels[0]["corr"][supported_full], dtype=np.float32
    ) / n4
    recovar_corr_supported = recovar_score_weight_compact[supported_compact]
    corr_delta = recovar_corr_supported.astype(np.float64) - relion_corr_supported.astype(
        np.float64
    )
    largest_corr_delta_row = int(np.argmax(np.abs(corr_delta)))
    dc_compact_rows = np.flatnonzero(compact_dc_mask)
    _require(
        dc_compact_rows.size <= 1,
        "compact score support contains duplicate DC pixels",
    )
    if dc_compact_rows.size:
        dc_compact_row = int(dc_compact_rows[0])
        dc_supported_rows = np.flatnonzero(supported_compact == dc_compact_row)
        _require(
            dc_supported_rows.size == 1,
            "RELION score support DC mapping is not unique",
        )
        dc_supported_row = int(dc_supported_rows[0])
        dc_full_row = int(supported_full[dc_supported_row])
        relion_dc_value = float(relion_corr_supported[dc_supported_row])
        recovar_dc_value = float(recovar_corr_supported[dc_supported_row])
    else:
        dc_compact_row = None
        dc_full_row = None
        relion_dc_value = None
        recovar_dc_value = None

    raw_counterfactual = _component_counterfactual(
        relion_raw_array,
        all_recovar_raw_array,
        substitution_arrays,
    )
    centered_counterfactual = _component_counterfactual(
        relion_raw_array,
        all_recovar_raw_array,
        substitution_arrays,
        center_deltas=True,
    )
    production_centered_data_delta = -raw_delta_centered
    replay_centered_data_delta = -_center(
        recovar_operand_replay_raw_array.astype(np.float64)
        - relion_raw_array.astype(np.float64)
    )
    production_replay_exact_mask = np.asarray(
        [
            bool(candidate_validation["production_replay_exact"])
            for candidate_validation in validation["candidates"]
        ],
        dtype=bool,
    )
    _require(
        production_replay_exact_mask.shape == production_centered_data_delta.shape,
        "capture validation candidate order changed",
    )
    production_replay_exact_count = int(
        np.count_nonzero(production_replay_exact_mask)
    )
    exact_production_centered = _center(
        production_centered_data_delta[production_replay_exact_mask]
    )
    exact_replay_centered = _center(
        replay_centered_data_delta[production_replay_exact_mask]
    )
    exact_replay_metric = _metric(exact_production_centered, exact_replay_centered)
    production_exact_centered_counterfactual = _component_counterfactual(
        relion_raw_array[production_replay_exact_mask],
        all_recovar_raw_array[production_replay_exact_mask],
        {
            name: records[production_replay_exact_mask]
            for name, records in substitution_arrays.items()
        },
        center_deltas=True,
    )
    production_exact_raw_counterfactual = _component_counterfactual(
        relion_raw_array[production_replay_exact_mask],
        all_recovar_raw_array[production_replay_exact_mask],
        {
            name: records[production_replay_exact_mask]
            for name, records in substitution_arrays.items()
        },
    )
    production_exact_jax_arithmetic_delta_centered = _center(
        jax_arithmetic_delta[production_replay_exact_mask]
    )
    production_centered_energy = float(
        np.vdot(production_centered_data_delta, production_centered_data_delta).real
    )
    production_exact_centered_energy = float(
        np.vdot(exact_production_centered, exact_production_centered).real
    )
    preprocessing_reports = {}
    for mode, counterfactual in preprocessing_counterfactuals.items():
        counterfactual_raw = np.asarray(counterfactual["raw_diff2"], dtype=np.float32)
        centered_data_delta = -_center(
            counterfactual_raw.astype(np.float64)
            - relion_raw_array.astype(np.float64)
        )
        exact_centered_data_delta = _center(
            centered_data_delta[production_replay_exact_mask]
        )
        energy = float(np.vdot(centered_data_delta, centered_data_delta).real)
        exact_energy = float(
            np.vdot(exact_centered_data_delta, exact_centered_data_delta).real
        )
        preprocessing_reports[mode] = {
            "direct_score_image_factor": counterfactual[
                "direct_score_image_factor"
            ],
            "normalization_source": counterfactual["normalization_source"],
            "shifted_image_operand": _metric_up_to_global_sign(
                np.concatenate(counterfactual["shifted_image_relion"]),
                np.concatenate(counterfactual["shifted_image_counterfactual"]),
            ),
            "centered_data_score_delta_recovar_minus_relion": (
                centered_data_delta.tolist()
            ),
            "centered_data_score_delta_l2": float(np.sqrt(energy)),
            "centered_data_score_delta_max_abs": float(
                np.max(np.abs(centered_data_delta), initial=0.0)
            ),
            "residual_energy_change_vs_captured_production": (
                float(energy / production_centered_energy - 1.0)
                if production_centered_energy > 0
                else 0.0
            ),
            "production_exact_candidates_recentered": {
                "centered_data_score_delta_recovar_minus_relion": (
                    exact_centered_data_delta.tolist()
                ),
                "l2": float(np.sqrt(exact_energy)),
                "max_abs": float(
                    np.max(np.abs(exact_centered_data_delta), initial=0.0)
                ),
                "residual_energy_change_vs_captured_production": (
                    float(exact_energy / production_exact_centered_energy - 1.0)
                    if production_exact_centered_energy > 0
                    else 0.0
                ),
            },
        }
    rotation_direct = _metric(capture.candidates[0]["matrix"], recovar_rotation.reshape(-1))
    rotation_transpose = _metric(
        capture.candidates[0]["matrix"], recovar_rotation.T.reshape(-1)
    )
    classification, selected_basis = _select_component_classification(
        production_exact_raw_counterfactual,
        production_exact_centered_counterfactual,
        candidate_count=production_replay_exact_count,
    )
    classification_basis = (
        "production_exact_candidates_centered_raw_diff2"
        if selected_basis == "centered_raw_diff2"
        else selected_basis
    )
    return {
        "schema": "k4_relion_recovar_fine_operand_comparison_v10",
        "status": "complete",
        "classification": classification,
        "classification_basis": classification_basis,
        "capture_validation": validation,
        "inputs": {
            "capture": str(capture_path.resolve()),
            "capture_sha256": _sha256(capture_path),
            "pass2": str(pass2_path.resolve()),
            "pass2_sha256": _sha256(pass2_path),
            "contribution": str(contribution_path.resolve()),
            "contribution_sha256": _sha256(contribution_path),
        },
        "scope": {
            "stack_index_one_based": capture.stack_index,
            "particle_row_in_contribution_shard": particle,
            "original_index_zero_based": original_index,
            "recovar_global_rotation": recovar_global_rotation,
            "relion_rotation_local": int(capture.candidates[0]["rotation_local"]),
            "physical_image_size": physical_image_size,
            "current_size": current_size,
            "score_compact_pixel_count": int(compact_indices.size),
            "contribution_reconstruction_pixel_count": int(
                np.asarray(values["window_indices"]).size
            ),
            "relion_full_pixel_count": capture.image_size,
            "particle_diameter_angstrom": particle_diameter_angstrom,
            "mask_edge_pixels": mask_edge_pixels,
            "recovar_preprocess_backend": preprocess_backend,
            "recovar_direct_score_image_factor": float(direct_score_factor),
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
            "production_rule": (
                "exclude DC from the compact score lookup, or zero it before "
                "direct Gaussian scoring when present"
            ),
            "present_in_score_support": bool(dc_compact_rows.size),
            "compact_row": dc_compact_row,
            "relion_full_row": dc_full_row,
            "relion_scaled_value": relion_dc_value,
            "recovar_scaled_value_after_production_dc_zero": recovar_dc_value,
            "largest_remaining_abs_delta": float(
                np.abs(corr_delta[largest_corr_delta_row])
            ),
            "largest_remaining_delta_relion_full_row": int(
                supported_full[largest_corr_delta_row]
            ),
        },
        "operands": metrics,
        "pass2_raw_operand_crosschecks": {
            "source": "RECOVAR pass-2 effective raw operand v2 archive",
            "raw_operand_schema": str(
                np.asarray(score_values["raw_operand_schema"]).item()
            ),
            "score_weight_reconstructed_from_contribution": _metric(
                recovar_score_weight_compact,
                reconstructed_corr_compact,
            ),
            "native_sum_init": _metric(
                native_sum_init,
                np.full(native_sum_init.shape, recovar_sum_init, dtype=np.float32),
            ),
            "native_nonzero_corr_outside_score_lookup": int(
                np.count_nonzero(native_corr_full[:, unsupported_full])
            ),
            "native_nonzero_contribution_outside_score_lookup": int(
                np.count_nonzero(native_contribution_full[:, unsupported_full])
            ),
            "selected_compact_pair_count": len(candidate_rows),
        },
        "raw_diff2_delta_centering": {
            "mean_raw_delta_recovar_minus_relion": float(np.mean(raw_delta)),
            "max_abs_centered_raw_delta": float(
                np.max(np.abs(raw_delta_centered), initial=0.0)
            ),
            "sign_relation": (
                "centered Gaussian data score is the negative of centered raw diff2"
            ),
        },
        "candidate_score_boundary_closure": {
            "recovar_rotation_local": recovar_rotation_local,
            "production_centered_data_score_delta_recovar_minus_relion": (
                production_centered_data_delta.tolist()
            ),
            "operand_replay_centered_data_score_delta_recovar_minus_relion": (
                replay_centered_data_delta.tolist()
            ),
            "all_candidates_replay_vs_production": _metric(
                production_centered_data_delta,
                replay_centered_data_delta,
            ),
            "relion_production_replay_exact_mask": (
                production_replay_exact_mask.tolist()
            ),
            "production_exact_candidates_recentered_replay_vs_production": (
                exact_replay_metric
            ),
            "production_exact_candidates_centered_component_counterfactual": (
                production_exact_centered_counterfactual
            ),
            "production_exact_candidates_raw_component_counterfactual": (
                production_exact_raw_counterfactual
            ),
            "classification_candidate_count": production_replay_exact_count,
            "classification": (
                "production target comes directly from the admitted pass-2 "
                "raw score table; the separate operand replay metric retains "
                "any JAX/XLA arithmetic difference"
            ),
        },
        "jax_arithmetic_on_native_operands": {
            "description": (
                "RECOVAR's production JAX/XLA direct-Gaussian reduction tree "
                "evaluated with the captured native RELION reference, shifted "
                "image, correction weights, and sum_init through the admitted "
                "pass-2 full-to-compact score lookup"
            ),
            "raw_diff2_vs_native_production": _metric(
                relion_raw_array,
                jax_arithmetic_native_raw,
            ),
            "raw_diff2_delta_recovar_minus_relion": (
                jax_arithmetic_delta.tolist()
            ),
            "raw_diff2_delta_centered": jax_arithmetic_delta_centered.tolist(),
            "centered_delta_l2": float(
                np.linalg.norm(jax_arithmetic_delta_centered)
            ),
            "centered_delta_max_abs": float(
                np.max(np.abs(jax_arithmetic_delta_centered), initial=0.0)
            ),
            "production_exact_candidates_recentered": {
                "raw_diff2_delta_centered": (
                    production_exact_jax_arithmetic_delta_centered.tolist()
                ),
                "exact_zero": bool(
                    np.all(production_exact_jax_arithmetic_delta_centered == 0)
                ),
                "mismatch_count_vs_zero": int(
                    np.count_nonzero(production_exact_jax_arithmetic_delta_centered)
                ),
                "l2": float(
                    np.linalg.norm(production_exact_jax_arithmetic_delta_centered)
                ),
                "max_abs": float(
                    np.max(
                        np.abs(production_exact_jax_arithmetic_delta_centered),
                        initial=0.0,
                    )
                ),
            },
        },
        "preprocessing_counterfactuals": preprocessing_reports,
        "raw_diff2_component_counterfactual": raw_counterfactual,
        "centered_raw_diff2_component_counterfactual": centered_counterfactual,
        "candidates": candidate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--contribution", type=Path, required=True)
    parser.add_argument("--recovar-global-rotation", type=int, required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(
        args.capture,
        args.pass2,
        args.contribution,
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
