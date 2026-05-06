"""RELION parity dump: per-iter and per-particle observables for diff comparison.

Activated by env var ``RECOVAR_PARITY_DUMP_DIR``. When unset, all hook calls
no-op so the dump has zero behavioral effect. Optional env vars:

- ``RECOVAR_PARITY_DUMP_TAG`` — per-particle full-tensor capture, comma-separated
  global indices (default: empty).
- ``RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE`` — int factor to shrink half volumes
  before saving (default 2 → 64³ for 128³ box).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_E_STEP: dict[int, dict[str, Any]] = {}

# Per-iter wall-time tracking. Keyed by iteration index (the ``iteration``
# variable inside ``_run_relion_iteration_loop``), holds:
#   {"t0": float, "stages": {stage_name: cumulative_seconds_since_t0}}
# All hooks no-op when ``RECOVAR_PARITY_DUMP_DIR`` is unset.
_ITER_TIMERS: dict[int, dict[str, Any]] = {}


def is_active() -> bool:
    return bool(os.environ.get("RECOVAR_PARITY_DUMP_DIR"))


def start_iteration(iteration: int) -> None:
    """Stamp the start of an iteration so ``mark_stage`` can record cumulative time.

    No-op when the parity dump is not active.
    """
    if not is_active():
        return
    _ITER_TIMERS[int(iteration)] = {"t0": time.time(), "stages": {}}


def mark_stage(iteration: int, stage: str) -> None:
    """Record cumulative seconds since ``start_iteration(iteration)`` for a stage.

    Stage names mirror the production perf baseline vocabulary plus EM-specific
    stages: ``"e_step"``, ``"m_step"``, ``"recon"``, ``"fsc"``, ``"noise_update"``.
    Calling with the same stage name twice overwrites; the cumulative-since-t0
    semantics make this safe — later calls dominate. No-op when inactive or when
    ``start_iteration`` was never called for that iteration.
    """
    if not is_active():
        return
    timer = _ITER_TIMERS.get(int(iteration))
    if timer is None:
        return
    timer["stages"][str(stage)] = float(time.time() - timer["t0"])


def get_iteration_timing(iteration: int) -> tuple[float | None, dict[str, float]]:
    """Return ``(wall_time_s, stage_seconds)`` for an iteration. Wall is None if untracked."""
    timer = _ITER_TIMERS.get(int(iteration))
    if timer is None:
        return None, {}
    wall = float(time.time() - timer["t0"])
    stages = dict(timer["stages"])
    return wall, stages


def reset_iteration_timer(iteration: int) -> None:
    """Drop the timer state for a given iteration (called by ``dump_iteration``)."""
    _ITER_TIMERS.pop(int(iteration), None)


def dump_dir() -> Path | None:
    raw = os.environ.get("RECOVAR_PARITY_DUMP_DIR")
    if not raw:
        return None
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_iteration() -> None:
    _E_STEP.clear()


def collect_e_step(
    *,
    half: int,
    em_stats,
    hard_assignment,
    coarse_hard_assignment,
    noise_stats,
    Ft_y,
    Ft_ctf,
    pose_rotation_eulers,
    best_pose_rotation_eulers,
    best_pose_translations,
    translation_search_base,
    original_image_indices=None,
    class_assignments=None,
    class_responsibilities=None,
    class_posterior_sums=None,
) -> None:
    """Snapshot per-half E-step outputs. Called once per half inside the iter."""

    if not is_active():
        return
    _E_STEP[int(half)] = {
        "log_evidence": np.asarray(em_stats.log_evidence_per_image, dtype=np.float64),
        "best_log_score": np.asarray(em_stats.best_log_score_per_image, dtype=np.float64),
        "max_posterior": np.asarray(em_stats.max_posterior_per_image, dtype=np.float32),
        "rotation_posterior_sums": np.asarray(em_stats.rotation_posterior_sums, dtype=np.float32),
        "hard_assignment": np.asarray(hard_assignment, dtype=np.int64),
        "coarse_hard_assignment": (
            np.asarray(coarse_hard_assignment, dtype=np.int64) if coarse_hard_assignment is not None else None
        ),
        "wsum_sigma2_noise": np.asarray(noise_stats.wsum_sigma2_noise, dtype=np.float64)
        if noise_stats is not None
        else None,
        "wsum_img_power": np.asarray(noise_stats.wsum_img_power, dtype=np.float64) if noise_stats is not None else None,
        "wsum_sigma2_offset": float(noise_stats.wsum_sigma2_offset) if noise_stats is not None else 0.0,
        "sumw": float(noise_stats.sumw) if noise_stats is not None else 0.0,
        "Ft_y_norm_per_voxel": _voxel_magnitude(Ft_y),
        "Ft_ctf_per_voxel": _voxel_magnitude(Ft_ctf),
        "pose_eulers": (
            np.asarray(pose_rotation_eulers, dtype=np.float32) if pose_rotation_eulers is not None else None
        ),
        "best_eulers": (
            np.asarray(best_pose_rotation_eulers, dtype=np.float32) if best_pose_rotation_eulers is not None else None
        ),
        "best_translations": (
            np.asarray(best_pose_translations, dtype=np.float32) if best_pose_translations is not None else None
        ),
        "translation_search_base": (
            np.asarray(translation_search_base, dtype=np.float32) if translation_search_base is not None else None
        ),
        "original_image_indices": (
            np.asarray(original_image_indices, dtype=np.int64) if original_image_indices is not None else None
        ),
        "class_assignments": (
            np.asarray(class_assignments, dtype=np.int32) if class_assignments is not None else None
        ),
        "class_responsibilities": (
            np.asarray(class_responsibilities, dtype=np.float32)
            if class_responsibilities is not None
            else None
        ),
        "class_posterior_sums": (
            np.asarray(class_posterior_sums, dtype=np.float64) if class_posterior_sums is not None else None
        ),
    }


def _voxel_magnitude(arr) -> np.ndarray:
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        return np.abs(a).astype(np.float32)
    return np.abs(np.asarray(a, dtype=np.float64)).astype(np.float32)


def dump_iteration(
    *,
    iteration: int,
    init_relion_iteration: int,
    current_size: int,
    sigma_offset: float,
    translation_step: float,
    translation_range: float,
    random_perturbation: float,
    random_perturbation_instance: int,
    tau2_fudge: float,
    voxel_size: float,
    grid_size: int,
    volume_shape,
    ave_pmax: float,
    fsc: np.ndarray,
    sigma2_noise: np.ndarray,
    means: list,
    unreg_means: list,
    new_iter_best_rotation_eulers: list,
    new_iter_best_translations: list,
    iteration_start: float | None = None,
) -> None:
    """Write one .npz per iteration combining both halves with E-step snapshots.

    If ``start_iteration(iteration)`` was called, the dump records ``wall_time_s``
    plus a ``stage_seconds_<name>`` field per stage and clears the timer entry.
    If the timer was not registered but ``iteration_start`` is provided, falls
    back to ``time.time() - iteration_start`` for the wall time only.
    """

    out = dump_dir()
    if out is None:
        return

    payload: dict[str, Any] = {
        "iteration": np.int32(iteration),
        "init_relion_iteration": np.int32(init_relion_iteration),
        "relion_iteration": np.int32(int(init_relion_iteration) + int(iteration) + 1),
        "current_size": np.int32(current_size),
        "sigma_offset": np.float64(sigma_offset),
        "translation_step": np.float64(translation_step),
        "translation_range": np.float64(translation_range),
        "random_perturbation": np.float64(random_perturbation),
        "random_perturbation_instance": np.int64(random_perturbation_instance),
        "tau2_fudge": np.float64(tau2_fudge),
        "voxel_size": np.float64(voxel_size),
        "grid_size": np.int32(grid_size),
        "ave_pmax": np.float64(ave_pmax),
        "fsc": np.asarray(fsc, dtype=np.float64),
        "sigma2_noise": np.asarray(sigma2_noise, dtype=np.float64),
    }

    # --- Wall-time / per-stage timing ---
    wall_time_s, stage_seconds = get_iteration_timing(iteration)
    if wall_time_s is None and iteration_start is not None:
        wall_time_s = float(time.time() - iteration_start)
    if wall_time_s is not None:
        payload["wall_time_s"] = np.float64(wall_time_s)
    for stage_name, stage_t in stage_seconds.items():
        payload[f"stage_seconds_{stage_name}"] = np.float64(stage_t)

    for k in (0, 1):
        snap = _E_STEP.get(k)
        if snap is None:
            continue
        for key, val in snap.items():
            if val is None:
                continue
            payload[f"half{k + 1}_{key}"] = val

        # Per-shell reduction needs to know the layout of Ft_y/Ft_ctf (full N^3 vs
        # half-spectrum N^2 * (N/2+1)). Skip the reduction; record summary scalars only.
        Ft_y_per_voxel = snap.get("Ft_y_norm_per_voxel")
        Ft_ctf_per_voxel = snap.get("Ft_ctf_per_voxel")
        if Ft_y_per_voxel is not None:
            payload[f"half{k + 1}_Ft_y_total"] = float(np.sum(Ft_y_per_voxel.astype(np.float64)))
            payload[f"half{k + 1}_Ft_y_max"] = float(np.max(Ft_y_per_voxel))
            payload[f"half{k + 1}_Ft_y_size"] = int(Ft_y_per_voxel.size)
        if Ft_ctf_per_voxel is not None:
            payload[f"half{k + 1}_Ft_ctf_total"] = float(np.sum(Ft_ctf_per_voxel.astype(np.float64)))
            payload[f"half{k + 1}_Ft_ctf_max"] = float(np.max(Ft_ctf_per_voxel))
            payload[f"half{k + 1}_Ft_ctf_size"] = int(Ft_ctf_per_voxel.size)
        payload.pop(f"half{k + 1}_Ft_y_norm_per_voxel", None)
        payload.pop(f"half{k + 1}_Ft_ctf_per_voxel", None)

        if means[k] is not None:
            payload[f"half{k + 1}_mean_real_ds"] = _downsample_volume_real(means[k], volume_shape)
        if unreg_means[k] is not None:
            payload[f"half{k + 1}_unreg_mean_real_ds"] = _downsample_volume_real(unreg_means[k], volume_shape)
        if new_iter_best_rotation_eulers[k] is not None:
            payload[f"half{k + 1}_best_eulers_total"] = np.asarray(new_iter_best_rotation_eulers[k], dtype=np.float32)
        if new_iter_best_translations[k] is not None:
            payload[f"half{k + 1}_best_translations_total"] = np.asarray(
                new_iter_best_translations[k], dtype=np.float32
            )

    relion_iter = int(init_relion_iteration) + int(iteration) + 1
    np.savez_compressed(out / f"iter_{relion_iter:03d}.npz", **payload)
    reset_iteration()
    reset_iteration_timer(iteration)


def _downsample_volume_real(volume_ft_flat, volume_shape) -> np.ndarray:
    """Downsample by a stored env factor, returning a real-space crop."""
    factor = int(os.environ.get("RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE", "2"))
    factor = max(1, factor)
    from recovar.core import fourier_transform_utils as ftu

    arr = np.asarray(volume_ft_flat)
    n_voxels = int(np.prod(volume_shape))
    if arr.size != n_voxels:
        if arr.size % n_voxels != 0:
            raise ValueError(
                f"volume has {arr.size} coefficients, not a multiple of {volume_shape}",
            )
        leading = int(arr.size // n_voxels)
        return np.stack(
            [
                _downsample_volume_real(arr.reshape(leading, n_voxels)[i], volume_shape)
                for i in range(leading)
            ],
            axis=0,
        )

    arr = arr.reshape(volume_shape)
    if factor == 1:
        return np.real(np.asarray(ftu.get_idft3(arr))).astype(np.float32).reshape(-1)
    nz = volume_shape[0]
    crop = nz // factor
    if crop < 4:
        return np.real(np.asarray(ftu.get_idft3(arr))).astype(np.float32).reshape(-1)
    real = np.real(np.asarray(ftu.get_idft3(arr))).astype(np.float32)
    start = (nz - crop) // 2
    end = start + crop
    return real[start:end, start:end, start:end].reshape(-1)
