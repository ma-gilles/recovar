"""Diagnostic capture policy and writers for dense single-volume refinement.

Extracted from ``iteration_loop.py``:

- ``_maybe_dump_noise_update_debug`` writes per-iteration RELION-parity
  noise-update sufficient statistics when ``RECOVAR_NOISE_DEBUG_DUMP_DIR``
  is set.
- ``_save_iteration_intermediates`` writes per-iteration regularized and
  unregularized volumes, FSC, noise, tau2, hard assignments, and metadata
  when ``--save_intermediates_dir`` is provided.
- ``_save_bpref_accumulators`` writes the same accumulator schema before or
  after the low-resolution half-map join. The controller chooses the boundary.
- Half selectors validate terminating significance/noise captures and explicit
  numbered-half device captures before the controller dispatches scoring.
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


_SIGNIFICANCE_DUMP_TARGET_HALF_ENV = "RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF"
_PASS2_NORM_DUMP_TARGET_HALF_ENV = "RECOVAR_PASS2_DUMP_TARGET_HALF"


def _significance_dump_half_indices(
    *,
    numbered_iteration: int,
    n_classes: int,
    experiment_datasets,
    environ=None,
) -> tuple[int, ...]:
    """Select one half only at an explicitly terminating diagnostic boundary."""

    env = os.environ if environ is None else environ
    raw_significance_half = str(env.get(_SIGNIFICANCE_DUMP_TARGET_HALF_ENV, "")).strip()
    raw_pass2_half = str(env.get(_PASS2_NORM_DUMP_TARGET_HALF_ENV, "")).strip()
    if raw_significance_half and raw_pass2_half:
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} and "
            f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} are mutually exclusive"
        )
    if not raw_significance_half and not raw_pass2_half:
        return (0, 1)
    pass2_norm_mode = bool(raw_pass2_half)
    target_half_env = (
        _PASS2_NORM_DUMP_TARGET_HALF_ENV
        if pass2_norm_mode
        else _SIGNIFICANCE_DUMP_TARGET_HALF_ENV
    )
    raw_half = raw_pass2_half if pass2_norm_mode else raw_significance_half
    if pass2_norm_mode:
        if str(env.get("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS=1"
            )
        if str(env.get("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET=1"
            )
    elif str(env.get("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "")).strip() != "1":
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} requires "
            "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1"
        )
    if int(n_classes) != 1:
        raise RuntimeError(f"{target_half_env} is K=1 diagnostic-only")
    try:
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError(f"{target_half_env} must be 1 or 2") from exc
    if target_half not in {1, 2}:
        raise ValueError(f"{target_half_env} must be 1 or 2")

    prefix = "RECOVAR_PASS2_DUMP" if pass2_norm_mode else "RECOVAR_SIGNIFICANCE_DUMP"
    raw_iteration = str(env.get(f"{prefix}_ITERATION", "")).strip()
    raw_targets = str(env.get(f"{prefix}_ORIGINAL_INDICES", "")).strip()
    dump_dir = str(env.get(f"{prefix}_DIR", "")).strip()
    if not raw_iteration or not raw_targets or not dump_dir:
        raise RuntimeError(
            f"{target_half_env} requires an explicit dump directory, "
            "iteration, and original-index target set"
        )
    try:
        target_iteration = int(raw_iteration)
        target_indices = {
            int(token) for token in raw_targets.replace(",", " ").split()
        }
    except ValueError as exc:
        raise ValueError("significance dump iteration and original indices must be integers") from exc
    if target_iteration <= 0 or not target_indices:
        raise ValueError("significance dump iteration and original-index target set must be nonempty")
    if int(numbered_iteration) != target_iteration:
        return (0, 1)

    selected_half_indices = set(
        np.asarray(experiment_datasets[target_half - 1].dataset_indices, dtype=np.int64).tolist()
    )
    missing = sorted(target_indices - selected_half_indices)
    if missing:
        raise RuntimeError(
            "significance dump targets are not all present in the selected half: "
            f"half={target_half} missing={missing}"
        )
    logger.info(
        "RECOVAR %s diagnostic: iteration=%d entering only half=%d "
        "for %d target particles; completion is mandatory before returning from the half",
        "pass-2 norm operand" if pass2_norm_mode else "coarse-significance",
        int(numbered_iteration),
        target_half,
        len(target_indices),
    )
    return (target_half - 1,)


def _bpref_device_signature_active_for_numbered_half(
    *,
    iteration: int,
    half: int,
    final_all_data: bool = False,
    environ=None,
) -> bool:
    """Resolve one explicit numbered half boundary for device capture."""

    env = os.environ if environ is None else environ
    if not str(env.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "")).strip():
        return False
    raw_iteration = str(env.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "")).strip()
    raw_half = str(env.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "")).strip()
    if not raw_iteration or not raw_half:
        raise RuntimeError(
            "Scoped BPref device capture requires explicit positive "
            "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION and half 1 or 2"
        )
    try:
        target_iteration = int(raw_iteration)
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError("BPref device capture iteration/half targets must be integers") from exc
    if target_iteration <= 0 or target_half not in {1, 2}:
        raise ValueError("BPref device capture requires target iteration > 0 and half 1 or 2")
    if int(iteration) <= 0 or int(half) not in {1, 2}:
        raise ValueError("BPref device capture context requires numbered iteration > 0 and half 1 or 2")
    if final_all_data:
        return False
    return int(iteration) == target_iteration and int(half) == target_half


def _save_bpref_accumulators(
    dump_dir: str,
    *,
    stage: str,
    iteration: int,
    current_size: int,
    padding_factor: int,
    grid_size: int,
    voxel_size: float,
    volume_shape,
    accumulator_shape,
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
) -> None:
    """Save K1 accumulators at the controller's ``prejoin`` or ``accum`` stage.

    ``iteration`` is zero-based; filenames and payload metadata are one-based.
    Preserve the numerator dtype and save the real part of each weight array.
    These captures locate a state boundary, without establishing that every
    upstream scoring input matched between runs.
    """
    import pathlib

    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        pathlib.Path(dump_dir) / f"recovar_bpref_{stage}_it{iteration + 1:03d}.npz",
        schema=np.asarray(f"recovar-bpref-{stage}-v2"),
        run_id=np.asarray(os.environ.get("RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID", "unset")),
        iteration=np.int32(iteration + 1),
        current_size=np.int32(current_size),
        padding_factor=np.int32(padding_factor),
        grid_size=np.int32(grid_size),
        voxel_size=np.float32(voxel_size),
        volume_shape=np.asarray(volume_shape, dtype=np.int32),
        mstep_accumulator_shape=np.asarray(accumulator_shape, dtype=np.int32),
        Ft_y_0=np.asarray(Ft_y_0),
        Ft_y_1=np.asarray(Ft_y_1),
        Ft_ctf_0=np.asarray(Ft_ctf_0).real,
        Ft_ctf_1=np.asarray(Ft_ctf_1).real,
    )


def _dump_array_or_empty(arr):
    if arr is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(arr)


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
