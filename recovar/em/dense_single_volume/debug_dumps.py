"""Env-gated diagnostic dumps for the dense single-volume iteration loop.

Extracted from ``iteration_loop.py``: ``_maybe_dump_noise_update_debug``
writes per-iteration RELION-parity noise-update sufficient statistics
when ``RECOVAR_NOISE_DEBUG_DUMP_DIR`` is set.
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

logger = logging.getLogger(__name__)


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
