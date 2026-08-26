"""Env-gated diagnostic dumps for the dense single-volume iteration loop.

Extracted from ``iteration_loop.py``:

- ``_maybe_dump_noise_update_debug`` writes per-iteration RELION-parity
  noise-update sufficient statistics when ``RECOVAR_NOISE_DEBUG_DUMP_DIR``
  is set.
- ``_save_iteration_intermediates`` writes per-iteration regularized and
  unregularized volumes, FSC, noise, tau2, hard assignments, and metadata
  when ``--save_intermediates_dir`` is provided.
- ``_save_iteration_particle_states`` writes resolved particle poses,
  posterior maxima, significance counts, and immutable source indices.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_int_set
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.relion_metadata import _relion_half_plane_shell_counts
from recovar.em.sampling import rotation_grid_size

logger = logging.getLogger(__name__)

_K1_INTERNAL_FOURIER_DUMP_DIR_ENV = "RECOVAR_K1_INTERNAL_FOURIER_DUMP_DIR"
_K1_INTERNAL_FOURIER_DUMP_ITERATION_ENV = "RECOVAR_K1_INTERNAL_FOURIER_DUMP_ITERATION"


def _dump_array_or_empty(arr):
    if arr is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(arr)


def _maybe_save_k1_internal_fourier_references(*, iteration: int, means, k_class_enabled: bool) -> None:
    """Save one exact in-memory K=1 half-map boundary when explicitly requested."""

    output_dir = os.environ.get(_K1_INTERNAL_FOURIER_DUMP_DIR_ENV, "").strip()
    if not output_dir:
        return
    iteration_text = os.environ.get(_K1_INTERNAL_FOURIER_DUMP_ITERATION_ENV, "").strip()
    if not iteration_text:
        raise ValueError(
            f"{_K1_INTERNAL_FOURIER_DUMP_ITERATION_ENV} is required when "
            f"{_K1_INTERNAL_FOURIER_DUMP_DIR_ENV} is set"
        )
    try:
        target_iteration = int(iteration_text)
    except ValueError as exc:
        raise ValueError(
            f"{_K1_INTERNAL_FOURIER_DUMP_ITERATION_ENV} must be an integer"
        ) from exc
    if target_iteration < 0:
        raise ValueError(f"{_K1_INTERNAL_FOURIER_DUMP_ITERATION_ENV} must be non-negative")
    if int(iteration) != target_iteration:
        return
    if k_class_enabled:
        raise ValueError("exact internal Fourier reference dumping is restricted to K=1")
    if len(means) != 2:
        raise ValueError(f"expected two K=1 half references, got {len(means)}")

    os.makedirs(output_dir, exist_ok=True)
    for half_index, mean in enumerate(means, start=1):
        value = np.asarray(mean)
        if not np.issubdtype(value.dtype, np.complexfloating) or not np.isfinite(value).all():
            raise ValueError(f"half {half_index} internal reference must be finite complex data")
        output_path = os.path.join(
            output_dir,
            f"it{int(iteration):03d}_half{half_index}_mean_ft.npz",
        )
        if os.path.exists(output_path):
            raise FileExistsError(f"refusing to overwrite internal Fourier reference: {output_path}")
        temporary_path = output_path + ".tmp.npz"
        if os.path.exists(temporary_path):
            raise FileExistsError(f"refusing to overwrite temporary Fourier reference: {temporary_path}")
        np.savez(temporary_path, mean_vol_ft=value)
        os.replace(temporary_path, output_path)
    logger.info(
        "Saved exact internal K=1 Fourier references to %s (iteration %d)",
        output_dir,
        iteration,
    )


def _save_iteration_intermediates(
    save_dir: str,
    *,
    iteration: int,
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    means,
    unreg_means,
    fsc,
    noise_variance,
    noise_variance_per_half,
    mean_variance,
    hard_assignments,
    coarse_ha,
    effective_rotations,
    current_translations,
    use_local: bool,
    local_search_order: int,
    cs: int,
    state,
    n_classes: int,
    k_class_enabled: bool,
    volume_shape,
    voxel_size: float,
) -> None:
    """Write per-iteration intermediate volumes + diagnostics to ``save_dir``."""
    from recovar.output.output import save_volume

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_y_0.npy"), _dump_array_or_empty(Ft_y_0))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_y_1.npy"), _dump_array_or_empty(Ft_y_1))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_ctf_0.npy"), _dump_array_or_empty(Ft_ctf_0))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_ctf_1.npy"), _dump_array_or_empty(Ft_ctf_1))
    for k_half in range(2):
        class_indices_to_save = range(n_classes) if k_class_enabled else (None,)
        for class_idx in class_indices_to_save:
            suffix = f"_class{class_idx + 1}" if class_idx is not None else ""
            mean_to_save = means[k_half][class_idx] if class_idx is not None else means[k_half]
            save_volume(
                np.asarray(mean_to_save).reshape(-1),
                os.path.join(save_dir, f"it{iteration:03d}_half{k_half + 1}{suffix}_reg"),
                volume_shape=volume_shape,
                from_ft=True,
                voxel_size=voxel_size,
            )
            if unreg_means[k_half] is not None:
                unreg_to_save = unreg_means[k_half][class_idx] if class_idx is not None else unreg_means[k_half]
                save_volume(
                    np.asarray(unreg_to_save).reshape(-1),
                    os.path.join(save_dir, f"it{iteration:03d}_half{k_half + 1}{suffix}_unreg"),
                    volume_shape=volume_shape,
                    from_ft=True,
                    voxel_size=voxel_size,
                )
    _maybe_save_k1_internal_fourier_references(
        iteration=iteration,
        means=means,
        k_class_enabled=k_class_enabled,
    )
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_fsc.npy"),
        np.asarray(fsc) if fsc is not None else np.array([], dtype=np.float32),
    )
    np.save(os.path.join(save_dir, f"it{iteration:03d}_noise.npy"), np.asarray(noise_variance))
    for k_half, noise_k in enumerate(noise_variance_per_half):
        np.save(
            os.path.join(save_dir, f"it{iteration:03d}_noise_half{k_half + 1}.npy"),
            np.asarray(noise_k),
        )
    np.save(os.path.join(save_dir, f"it{iteration:03d}_tau2.npy"), np.asarray(mean_variance))
    for k_half in range(2):
        if hard_assignments[k_half] is not None:
            np.save(
                os.path.join(save_dir, f"it{iteration:03d}_ha_half{k_half + 1}.npy"),
                hard_assignments[k_half],
            )
    iter_meta = {
        "iteration": iteration,
        "current_size": int(cs),
        "n_rotations": int(rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]),
        "n_translations": int(current_translations.shape[0]),
        "healpix_order": int(state.healpix_order),
        "local_search": bool(use_local),
        "sigma_rot": float(state.sigma_rot),
    }
    np.save(os.path.join(save_dir, f"it{iteration:03d}_meta.npy"), iter_meta)
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_rotations.npy"),
        (np.asarray(effective_rotations) if not use_local else np.empty((0, 3, 3), dtype=np.float32)),
    )
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_translations.npy"),
        np.asarray(current_translations),
    )
    for k_half in range(2):
        if coarse_ha[k_half] is not None:
            np.save(
                os.path.join(save_dir, f"it{iteration:03d}_coarse_ha_half{k_half + 1}.npy"),
                np.asarray(coarse_ha[k_half], dtype=np.int32),
            )
    logger.info("Saved intermediate volumes to %s (iteration %d)", save_dir, iteration)


def _save_iteration_particle_states(
    save_dir: str,
    *,
    iteration: int,
    rotation_matrices_per_half,
    rotation_eulers_deg_per_half,
    relative_translations_pixels_per_half,
    absolute_translations_pixels_per_half,
    max_posterior_per_half,
    significant_counts_per_half,
    hard_assignments_per_half,
    coarse_hard_assignments_per_half,
    original_image_indices_per_half,
) -> None:
    """Write source-aligned resolved particle state for one sealed iteration."""

    os.makedirs(save_dir, exist_ok=True)
    per_half_fields = {
        "rotation_matrices": rotation_matrices_per_half,
        "rotation_eulers_deg": rotation_eulers_deg_per_half,
        "relative_translations_pixels": relative_translations_pixels_per_half,
        "absolute_translations_pixels": absolute_translations_pixels_per_half,
        "max_posterior": max_posterior_per_half,
        "significant_counts": significant_counts_per_half,
        "fine_hard_assignment": hard_assignments_per_half,
        "coarse_hard_assignment": coarse_hard_assignments_per_half,
        "original_image_indices": original_image_indices_per_half,
    }
    for half_index in range(2):
        payload = {
            name: _dump_array_or_empty(values[half_index])
            for name, values in per_half_fields.items()
        }
        populated_lengths = {
            name: int(value.shape[0])
            for name, value in payload.items()
            if value.ndim > 0 and value.size > 0
        }
        if populated_lengths and len(set(populated_lengths.values())) != 1:
            raise ValueError(
                "particle-state fields are not source-aligned for "
                f"half {half_index + 1}: {populated_lengths}"
            )
        particle_count = next(iter(populated_lengths.values()), 0)
        payload["half_local_indices"] = np.arange(particle_count, dtype=np.int64)
        payload["zero_based_iteration"] = np.asarray([iteration], dtype=np.int32)
        payload["one_based_iteration"] = np.asarray([iteration + 1], dtype=np.int32)
        payload["half"] = np.asarray([half_index + 1], dtype=np.int32)
        output_path = os.path.join(
            save_dir,
            f"it{iteration:03d}_particle_state_half{half_index + 1}.npz",
        )
        np.savez(output_path, **payload)
        logger.info("Saved particle-state diagnostics to %s", output_path)


def _maybe_dump_noise_update_debug(
    *,
    iteration: int,
    current_size: int | None,
    image_shape,
    noise_stats_per_half,
    previous_noise_radial_per_half,
    noise_from_res_per_half,
    noise_from_res,
):
    """Write raw noise M-step terms for RELION parity debugging when requested."""

    dump_dir = os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
    if not dump_dir:
        return
    requested_iterations = parse_int_set(os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_ITERATION"))
    if requested_iterations is not None and int(iteration) not in requested_iterations:
        return

    os.makedirs(dump_dir, exist_ok=True)
    n_shells = int(image_shape[0]) // 2 + 1
    shell_indices_half = np.asarray(make_shell_indices_half(image_shape), dtype=np.int64)
    half_counts = np.bincount(shell_indices_half, minlength=n_shells).astype(np.float64)[:n_shells]
    half_weights = np.asarray(make_half_image_weights(image_shape), dtype=np.float64)
    half_weighted_counts = np.bincount(shell_indices_half, weights=half_weights, minlength=n_shells).astype(
        np.float64,
    )[:n_shells]

    payload = {
        "zero_based_iteration": np.array([int(iteration)], dtype=np.int32),
        "one_based_iteration": np.array([int(iteration) + 1], dtype=np.int32),
        "current_size": np.array([-1 if current_size is None else int(current_size)], dtype=np.int32),
        "image_shape": np.asarray(image_shape, dtype=np.int32),
        "shell_index_half": shell_indices_half.astype(np.int32),
        "half_shell_counts": half_counts,
        "half_weighted_shell_counts": half_weighted_counts,
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts(image_shape),
        "mean_sigma2_noise": np.asarray(noise_from_res, dtype=np.float64),
    }
    for half_id, stats_k in enumerate(noise_stats_per_half, start=1):
        prefix = f"half{half_id}"
        wsum_sigma2 = np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64)
        img_power = np.asarray(stats_k.wsum_img_power, dtype=np.float64)
        payload[f"{prefix}_wsum_sigma2_noise"] = wsum_sigma2
        payload[f"{prefix}_wsum_img_power"] = img_power
        payload[f"{prefix}_wsum_total"] = wsum_sigma2 + img_power
        payload[f"{prefix}_sumw"] = np.array([float(stats_k.sumw)], dtype=np.float64)
        if getattr(stats_k, "wsum_norm_correction", None) is not None:
            payload[f"{prefix}_wsum_norm_correction"] = np.asarray(
                stats_k.wsum_norm_correction,
                dtype=np.float64,
            )
        payload[f"{prefix}_sigma2_noise"] = np.asarray(noise_from_res_per_half[half_id - 1], dtype=np.float64)
        payload[f"{prefix}_previous_sigma2_noise"] = np.asarray(
            previous_noise_radial_per_half[half_id - 1],
            dtype=np.float64,
        )
        if getattr(stats_k, "wsum_noise_a2", None) is not None:
            payload[f"{prefix}_wsum_noise_a2"] = np.asarray(stats_k.wsum_noise_a2, dtype=np.float64)
        if getattr(stats_k, "wsum_noise_xa", None) is not None:
            payload[f"{prefix}_wsum_noise_xa"] = np.asarray(stats_k.wsum_noise_xa, dtype=np.float64)

    path = os.path.join(dump_dir, f"recovar_noise_update_it{int(iteration) + 1:03d}.npz")
    np.savez_compressed(path, **payload)
    logger.info("Wrote RECOVAR noise update debug dump: %s", path)
