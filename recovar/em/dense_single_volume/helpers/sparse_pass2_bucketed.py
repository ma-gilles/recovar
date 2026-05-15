"""Bucketed batched implementation of sparse pass-2 oversampling.

Replaces the per-image Python loop in
``compute_pass2_stats_sparse`` with a shape-bucketed batched evaluation.

Background
----------
RELION's adaptive pass-2 evaluates the oversampled children of each
image's significant coarse (rotation, translation) pairs.  Because the
number of significant coarse rotations differs per image, a naive per-
image evaluation produces a different XLA shape for every call, leading
to catastrophic JIT recompilation when there are thousands of images.

This helper groups images by ``oversampled_rots.shape[0]`` (quantized
to a small set of bucket sizes via
``local_layout._exact_bucket_rotation_size``), pads each image's
oversampled rotations / log-priors / candidate masks to the bucket size,
and evaluates each bucket as a single GPU call with per-image
projections (analogous to the local-search exact engine).

The numerical contract matches the per-image reference path exactly:
identity-padded rotations are masked out via ``-inf`` log-prior and
``False`` (rot, trans) mask, so they contribute zero posterior mass and
do not perturb the M-step accumulators.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    half_image_phase_factors,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
    process_half_image,
)
from recovar.em.dense_single_volume.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size

logger = logging.getLogger(__name__)

_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH = 1_000_000
_DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH = 1_250_000
_DEFAULT_MAX_TRANSLATION_TILE_BYTES = 384 * 1024**2
# Scale sparse pass-2 bucket sizes from physical GPU memory and active score
# pixels. The fused K-class path is launch-bound at 100k/256 unless it uses
# larger chunks; these fractions still scale down on smaller GPUs.
_AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION = 0.640
_AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION = 0.305
_AUTO_FUSED_KCLASS_FULL_HYPOTHESIS_DEVICE_FRACTION = 0.610
_AUTO_TRANSLATION_TILE_DEVICE_FRACTION = 0.020
_AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION = 0.014
_AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION = 0.007
_AUTO_PROJECTION_CACHE_DEVICE_FRACTION = 0.040
_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES"
_SCORE_ONLY_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES"
_MAX_TRANSLATION_TILE_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES"
_PROJECTION_CACHE_MAX_BYTES_ENV = "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES"
_DEFAULT_PROJECTION_CACHE_MAX_BYTES = 3 * 1024**3

_native_mstep_dump_counter = 0


class SparseKClassPass2FusedResult(NamedTuple):
    """K-class sparse pass-2 result normalized over the joint class x pose grid."""

    class_log_evidence: np.ndarray
    class_score_log_z: np.ndarray
    Ft_y: tuple[np.ndarray, ...]
    Ft_ctf: tuple[np.ndarray, ...]
    per_class_hard_assignments: np.ndarray
    per_class_stats: tuple
    noise_stats: tuple | None
    per_class_best_pose_rotations: tuple[np.ndarray, ...] | None
    per_class_best_pose_translations: tuple[np.ndarray, ...] | None
    per_class_best_pose_rotation_ids: tuple[np.ndarray, ...] | None
    profile_summary: dict


def _maybe_dump_native_half_mstep(
    Ft_y_total,
    Ft_ctf_total,
    *,
    current_size,
    n_images,
    recon_volume_shape,
    stage,
):
    dump_dir = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR")
    if not dump_dir:
        return

    global _native_mstep_dump_counter
    dump_idx = _native_mstep_dump_counter
    _native_mstep_dump_counter += 1

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / f"native_half_mstep_{dump_idx:03d}_{stage}_n{int(n_images):04d}_cs{int(current_size):03d}.npz",
        Ft_y=np.asarray(Ft_y_total),
        Ft_ctf=np.asarray(Ft_ctf_total),
        current_size=np.int32(current_size),
        n_images=np.int32(n_images),
        recon_volume_shape=np.asarray(recon_volume_shape, dtype=np.int32),
        stage=np.asarray(stage),
    )


# ---------------------------------------------------------------------------
# Per-image hypothesis preparation
# ---------------------------------------------------------------------------


def _prepare_per_image_pass2_inputs(
    significant_sample_indices,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation,
    fine_rotations_override=None,
    fine_rotation_parent_override=None,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Mirrors the per-image branch in the reference implementation in
    :func:`compute_pass2_stats_sparse_perimage_reference` exactly so the
    batched path is a strict per-image equivalent.
    """
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    n_images = len(significant_sample_indices)
    per_image_oversampled_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=np.float32)
    else:
        rotation_log_prior_np = None

    fine_rotations_np = None
    fine_parent_np = None
    if fine_rotations_override is None and fine_rotation_parent_override is None:
        pass
    elif fine_rotations_override is not None and fine_rotation_parent_override is not None:
        fine_rotations_np = np.asarray(fine_rotations_override, dtype=np.float32)
        fine_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int64)
        if fine_parent_np.ndim != 1:
            raise ValueError("fine_rotation_parent_override must be a 1D array")
        if fine_rotations_np.shape[0] != fine_parent_np.shape[0]:
            raise ValueError(
                "fine_rotations_override and fine_rotation_parent_override disagree on rotation count: "
                f"{fine_rotations_np.shape[0]} vs {fine_parent_np.shape[0]}",
            )
        if int(fine_parent_np.min(initial=0)) < 0 or int(fine_parent_np.max(initial=-1)) >= int(n_coarse_rot):
            raise ValueError("fine_rotation_parent_override values must be in [0, n_coarse_rot)")
    else:
        raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
            use_full_candidate_mask = True
            coarse_rot = unique_rot
            coarse_trans = None
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                coarse_rot = np.empty(0, dtype=np.int32)
                coarse_trans = np.empty(0, dtype=np.int32)
                unique_rot = np.array([0], dtype=np.int32)
                use_full_candidate_mask = False
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot)
                use_full_candidate_mask = False

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        if fine_rotations_override is None and fine_rotation_parent_override is None:
            oversampled_rots, parent_map, oversampled_rot_indices = get_oversampled_rotation_grid_from_samples(
                unique_rot,
                nside_level,
                oversampling_order=oversampling_order,
                random_perturbation=random_perturbation,
                return_rotation_indices=True,
            )
            oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
            parent_map = np.asarray(parent_map, dtype=np.int32)
            oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
        elif fine_rotations_np is not None and fine_parent_np is not None:
            selected_parent = np.zeros(n_coarse_rot, dtype=bool)
            selected_parent[unique_rot] = True
            child_mask = selected_parent[fine_parent_np]
            oversampled_rot_indices = np.flatnonzero(child_mask).astype(np.int64)
            oversampled_rots = fine_rotations_np[oversampled_rot_indices]
            parent_map = np.searchsorted(unique_rot, fine_parent_np[oversampled_rot_indices]).astype(np.int32)
        else:
            raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

        if rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=np.float32)

        if use_full_candidate_mask:
            candidate_mask = np.ones((oversampled_rots.shape[0], n_fine_trans), dtype=bool)
        elif coarse_trans.size == 0:
            candidate_mask = np.zeros((oversampled_rots.shape[0], n_fine_trans), dtype=bool)
        else:
            coarse_valid = np.zeros((unique_rot.size, n_coarse_trans), dtype=bool)
            coarse_valid[np.searchsorted(unique_rot, coarse_rot), coarse_trans] = True
            candidate_mask = coarse_valid[:, fine_translation_parent][parent_map]

        per_image_oversampled_rots.append(oversampled_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(np.float32, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    return {
        "oversampled_rots": per_image_oversampled_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }


# ---------------------------------------------------------------------------
# Bucket spec
# ---------------------------------------------------------------------------


def _bucket_pass2_inputs(
    per_image_inputs,
    n_fine_trans,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
):
    """Group images into buckets that share a padded rotation count.

    Returns a list of dicts; each contains the padded per-image arrays
    needed to evaluate the bucket as one batched call.

    To avoid OOM when one bucket is very large
    (``bucket_size * n_images_in_bucket * n_fine_trans`` is the (B, R, T)
    score tensor footprint), we split each per-quantization-size group
    into chunks of at most ``max_hypotheses_per_microbatch /
    (bucket_size * n_fine_trans)`` images.
    """
    n_images = len(per_image_inputs["oversampled_rots"])
    rotation_counts = np.array(
        [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
        dtype=np.int64,
    )
    if n_images == 0:
        return []

    bucket_sizes = np.array(
        [_exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization) for count in rotation_counts],
        dtype=np.int64,
    )

    # Group by bucket size, smaller buckets first
    processing_order = np.lexsort((rotation_counts, bucket_sizes)).astype(np.int64)
    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])

    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
        bucket_image_indices = processing_order[bucket_sizes[processing_order] == bucket_size]
        # Chunk by max_hypotheses_per_microbatch and max_images_per_microbatch
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch) // max(1, bucket_size * int(n_fine_trans)),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_hypotheses))
        for start in range(0, bucket_image_indices.shape[0], max_per_chunk):
            chunk = bucket_image_indices[start : start + max_per_chunk]
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(chunk, dtype=np.int64),
                }
            )
    return buckets


def _bucket_sparse_k_class_pass2_inputs(
    per_image_inputs_by_class,
    n_fine_trans,
    *,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
):
    """Group images by the largest padded class support in a fused K-class pass."""

    n_classes = len(per_image_inputs_by_class)
    if n_classes == 0:
        return []
    n_images = len(per_image_inputs_by_class[0]["oversampled_rots"])
    if n_images == 0:
        return []
    bucket_sizes_by_class = []
    for per_image_inputs in per_image_inputs_by_class:
        if len(per_image_inputs["oversampled_rots"]) != n_images:
            raise ValueError("All classes must have the same image count for fused sparse pass-2")
        counts = np.asarray(
            [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
            dtype=np.int64,
        )
        bucket_sizes_by_class.append(
            np.asarray(
                [
                    _exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization)
                    for count in counts
                ],
                dtype=np.int64,
            )
        )
    fused_bucket_sizes = np.max(np.stack(bucket_sizes_by_class, axis=0), axis=0)
    processing_order = np.argsort(fused_bucket_sizes, kind="stable").astype(np.int64)
    unique_bucket_sizes = np.unique(fused_bucket_sizes[processing_order])

    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
        bucket_image_indices = processing_order[fused_bucket_sizes[processing_order] == bucket_size]
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch)
            // max(1, int(n_classes) * bucket_size * int(n_fine_trans)),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_hypotheses))
        for start in range(0, bucket_image_indices.shape[0], max_per_chunk):
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(
                        bucket_image_indices[start : start + max_per_chunk],
                        dtype=np.int64,
                    ),
                }
            )
    return buckets


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _parse_nvidia_smi_memory_rows(output: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, uuid, memory_mib = parts[:3]
        try:
            memory_bytes = int(memory_mib.split()[0]) * 1024**2
        except (ValueError, IndexError):
            continue
        if memory_bytes <= 0:
            continue
        rows[index] = memory_bytes
        rows[uuid] = memory_bytes
        if uuid.startswith("GPU-"):
            rows[uuid[4:]] = memory_bytes
    return rows


def _nvidia_smi_visible_device_memory_bytes(output: str, visible_devices: str | None) -> int | None:
    rows = _parse_nvidia_smi_memory_rows(output)
    if not rows:
        return None
    if visible_devices:
        tokens = [
            part.strip()
            for part in visible_devices.split(",")
            if part.strip() and part.strip() not in {"-1", "none", "NoDevFiles"}
        ]
        if not tokens:
            return None
        for token in tokens:
            if token in rows:
                return rows[token]
        return None
    return next(iter(rows.values()))


def _device_memory_limit_bytes() -> int | None:
    """Return selected accelerator memory, preferring physical GPU memory."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            memory_bytes = _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            if memory_bytes is not None:
                return memory_bytes
    except Exception:
        pass
    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory"):
        value = stats.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _auto_hypotheses_per_microbatch(
    *,
    score_only: bool,
    fused_k_class: bool = False,
    n_score_pixels: int | None,
    device_memory_bytes: int | None,
) -> int | None:
    if device_memory_bytes is None or n_score_pixels is None or int(n_score_pixels) <= 0:
        return None
    if score_only:
        fraction = _AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION
    elif fused_k_class:
        fraction = _AUTO_FUSED_KCLASS_FULL_HYPOTHESIS_DEVICE_FRACTION
    else:
        fraction = _AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION
    # The score kernel's dominant live block scales with candidate count times
    # active Fourier pixels. This keeps larger windows and smaller GPUs from
    # inheriting the same candidate cap as low-resolution H100 runs.
    bytes_per_score_pixel = np.dtype(np.complex64).itemsize
    return max(1, int(float(device_memory_bytes) * fraction / (int(n_score_pixels) * bytes_per_score_pixel)))


def _max_hypotheses_per_microbatch_for_pass(
    *,
    score_only: bool,
    use_window: bool,
    has_external_normalization: bool,
    dump_pass2_operands: bool,
    fused_k_class: bool = False,
    n_score_pixels: int | None = None,
    device_memory_bytes: int | None = None,
) -> int:
    if score_only and use_window and not has_external_normalization and not dump_pass2_operands:
        override = _optional_positive_int_env(_SCORE_ONLY_MAX_HYPOTHESES_ENV)
        if override is not None:
            return override
        auto = _auto_hypotheses_per_microbatch(
            score_only=True,
            fused_k_class=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory_bytes,
        )
        return int(auto) if auto is not None else _DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH
    override = _optional_positive_int_env(_MAX_HYPOTHESES_ENV)
    if override is not None:
        return override
    auto = _auto_hypotheses_per_microbatch(
        score_only=False,
        fused_k_class=fused_k_class,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory_bytes,
    )
    return int(auto) if auto is not None else _DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH


def _max_translation_tile_bytes_for_pass(
    device_memory_bytes: int | None = None,
    *,
    has_external_normalization: bool = False,
    fused_k_class: bool = False,
) -> int:
    override = _optional_positive_int_env(_MAX_TRANSLATION_TILE_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_MAX_TRANSLATION_TILE_BYTES
    if fused_k_class:
        fraction = _AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION
    elif has_external_normalization:
        fraction = _AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION
    else:
        fraction = _AUTO_TRANSLATION_TILE_DEVICE_FRACTION
    return max(1, int(float(device_memory_bytes) * fraction))


def _projection_cache_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_CACHE_DEVICE_FRACTION))


def _bucket_summary(buckets) -> str:
    if not buckets:
        return "empty"
    sizes = np.asarray([int(bucket["bucket_size"]) for bucket in buckets], dtype=np.int64)
    image_counts = np.asarray([len(bucket["image_indices"]) for bucket in buckets], dtype=np.int64)
    unique, counts = np.unique(sizes, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist(), strict=True), key=lambda item: item[1], reverse=True)[:8]
    return (
        f"bucket_size min/med/mean/max={int(sizes.min())}/{int(np.median(sizes))}/"
        f"{float(np.mean(sizes)):.1f}/{int(sizes.max())}, "
        f"images_per_bucket med/max={int(np.median(image_counts))}/{int(image_counts.max())}, "
        f"top_bucket_counts={top}"
    )


def _bucket_group_stats(buckets) -> dict[int, tuple[int, int]]:
    stats: dict[int, list[int]] = {}
    for bucket in buckets:
        bucket_size = int(bucket["bucket_size"])
        entry = stats.setdefault(bucket_size, [0, 0])
        entry[0] += 1
        entry[1] += len(bucket["image_indices"])
    return {bucket_size: (counts[0], counts[1]) for bucket_size, counts in stats.items()}


def _max_images_for_translation_tile(
    image_shape,
    n_fine_trans,
    *,
    max_tile_bytes=384 * 1024**2,
):
    """Limit one translated-image tile allocation to a bounded size."""
    half_image_size = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    bytes_per_complex_value = np.dtype(np.complex64).itemsize
    bytes_per_image = int(n_fine_trans) * half_image_size * bytes_per_complex_value
    return max(1, int(max_tile_bytes) // max(1, bytes_per_image))


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
):
    """Stack/pad per-image arrays into batched bucket tensors."""
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    # padded_rotations: identity-fill — projection of identity is harmless
    # because we mask via candidate_mask=False everywhere for padded rows.
    padded_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (batch, bucket_size, 3, 3),
    ).copy()
    padded_log_prior = np.full((batch, bucket_size), -1e30, dtype=np.float32)
    padded_candidate_mask = np.zeros((batch, bucket_size, n_fine_trans), dtype=bool)
    padded_parent_map = np.full((batch, bucket_size), -1, dtype=np.int32)
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        padded_rotations[row, :cnt] = rots
        padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
        padded_candidate_mask[row, :cnt, :] = per_image_inputs["candidate_mask"][image_idx]
        padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]

    return {
        "image_indices": image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "rotation_indices": padded_rotation_indices,
        "log_prior": padded_log_prior,
        "candidate_mask": padded_candidate_mask,
        "parent_map": padded_parent_map,
    }


# ---------------------------------------------------------------------------
# Scoring + normalization (per-bucket, supports (B, R, T) mask)
# ---------------------------------------------------------------------------


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
):
    """RELION GPU-style direct ``diff2`` scoring for pass-2 diagnostics.

    RELION's CUDA fine-search kernel first corrects the image by dividing by
    CTF and scale, then accumulates ``|Fref - Fimg_corrected_shift|^2 *
    corr_img`` where ``corr_img = Minvsigma2 * CTF^2 * scale^2``.  This is
    algebraically equivalent to the CPU ``Frefctf - Fimg`` expression but has
    different float32 rounding.  We remove the image-only constant so the
    existing relative-score/log-evidence contract is unchanged.

    Extremely small CTF/noise combinations can still overflow the direct form
    on long 256px runs.  Treat non-finite candidates as impossible hypotheses
    rather than letting NaNs enter posterior and noise accumulators.
    """

    weights = corr_img_score * half_weights[None, :]
    cross = jnp.einsum(
        "btn,bn,brn->brt",
        jnp.conj(shifted_corrected),
        weights,
        proj_half,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    proj_norm = 0.5 * jnp.einsum(
        "bn,brn->br",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = cross - proj_norm[:, :, None] + rotation_log_prior[:, :, None] + translation_log_prior[:, None, :]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_bucket_normalized_cc(
    shifted_score,  # (B, T, N) complex, image * CTF * shift / Xi2
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (B, R, T) bool
):
    """RELION iter-1 normalized-CC scoring for sparse pass-2 buckets."""

    proj_weighted = proj_half * half_weights[None, None, :]
    cross = -2.0 * jnp.einsum(
        "btn,brn->brt",
        jnp.conj(shifted_score),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2_weighted = (proj_half.real * proj_half.real + proj_half.imag * proj_half.imag) * half_weights[
        None, None, :
    ]
    norms = jnp.einsum(
        "bn,brn->br",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom[:, :, None]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _normalize_pass2_bucket(scores):
    """Compute per-image normalization stats from (B, R, T) scores."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    log_shift = safe_best_log_score[:, None, None]
    shifted = jnp.where(has_finite_score[:, None, None], scores - log_shift, -jnp.inf)
    probs = jnp.exp(shifted.astype(jnp.float64))
    probs = jnp.where(jnp.isfinite(probs), probs, 0.0)
    sum_exp = jnp.sum(probs.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    probs = probs / safe_sum_exp[:, None, None]
    probs = jnp.where(has_mass[:, None, None], probs, 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.where(has_mass, jnp.max(probs.reshape(scores.shape[0], -1), axis=1), 0.0)
    best_log_score = jnp.where(has_mass, best_log_score, -jnp.inf)
    return log_Z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def _normalize_pass2_bucket_score_only(scores):
    """Compute sparse pass-2 score stats without materializing posteriors."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None, None], scores - safe_best_log_score[:, None, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - log_Z)
    max_posterior = jnp.where(has_mass & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_mass, best_log_score, -jnp.inf)
    return log_Z, best_log_score, best_argmax, max_posterior


@jax.jit
def _logsumexp_pass2_bucket_score_only(scores):
    """Compute per-image sparse pass-2 logZ only."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None, None], scores - safe_best_log_score[:, None, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms.reshape(scores.shape[0], -1), axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    return jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), -jnp.inf)


@jax.jit
def _logsumexp_class_log_z(class_log_z):
    """Stable logsumexp over class-local sparse score normalizers."""

    finite = jnp.isfinite(class_log_z)
    max_value = jnp.max(jnp.where(finite, class_log_z, -jnp.inf), axis=0)
    has_finite = jnp.isfinite(max_value)
    shifted = jnp.where(finite & has_finite[None, :], class_log_z - max_value[None, :], -jnp.inf)
    exp_terms = jnp.exp(shifted)
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=0)
    return jnp.where(has_finite & (sum_exp > 0.0), max_value + jnp.log(sum_exp), -jnp.inf)


@jax.jit
def _winner_take_all_bucket_probs(scores, best_argmax, best_log_score):
    """One-hot sparse bucket probabilities for RELION firstiter_cc."""

    flat_size = scores.shape[1] * scores.shape[2]
    valid = jnp.isfinite(best_log_score)
    probs = jax.nn.one_hot(best_argmax, flat_size, dtype=scores.real.dtype).reshape(scores.shape)
    return probs * valid[:, None, None].astype(probs.dtype)


@jax.jit
def _normalize_pass2_bucket_with_log_z(scores, log_z):
    """Normalize sparse candidate scores with a precomputed full-grid log-Z."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    safe_log_z = jnp.where(has_finite_score, log_z, 0.0)
    probs = jnp.exp(scores - safe_log_z[:, None, None])
    probs = jnp.where(has_finite_score[:, None, None] & jnp.isfinite(probs), probs, 0.0)
    best_argmax = jnp.where(has_finite_score, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, probs, best_log_score, best_argmax, max_posterior


# ---------------------------------------------------------------------------
# Main bucketed driver
# ---------------------------------------------------------------------------


def _reorder_to_indices(image_indices_returned, requested_image_indices, *arrays):
    """Reorder per-image arrays so they match the order returned by the dataset."""
    if np.array_equal(image_indices_returned, requested_image_indices):
        return arrays
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(requested_image_indices).tolist())}
    order = np.array([position[int(idx)] for idx in np.asarray(image_indices_returned).tolist()], dtype=np.int64)
    return tuple(arr[order] for arr in arrays)


def _maybe_dump_pass2_bucket(
    *,
    experiment_dataset,
    image_indices,
    per_image_inputs,
    current_size,
    n_fine_trans,
    fine_translations,
    scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    ctf2_over_nv_score,
    proj_half,
    half_weights_used,
    window_indices,
    shifted_corrected_score_split=None,
):
    """Env-gated sparse pass-2 dump for RELION operand parity debugging."""
    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        original_indices = local_indices
    else:
        original_indices = np.asarray(original_indices_all, dtype=np.int64)[local_indices]

    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    rot_prior_np = np.asarray(rotation_log_prior, dtype=np.float64)
    trans_prior_np = np.asarray(translation_log_prior, dtype=np.float64)
    mask_np = np.asarray(candidate_mask, dtype=bool)
    ctf2_np = np.asarray(ctf2_over_nv_score, dtype=np.float64)
    proj_np = np.asarray(proj_half)
    shifted_corrected_np = (
        None if shifted_corrected_score_split is None else np.asarray(shifted_corrected_score_split)
    )

    for row in wanted_rows:
        image_idx = int(local_indices[row])
        original_idx = int(original_indices[row])
        cnt = int(per_image_inputs["oversampled_rots"][image_idx].shape[0])
        scores_row = scores_np[row, :cnt, :]
        pre_prior = scores_row - rot_prior_np[row, :cnt, None] - trans_prior_np[row, None, :]
        out_path = os.path.join(
            dump_dir,
            f"pass2_orig{original_idx:06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_idx),
            local_index=np.int64(image_idx),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            n_fine_trans=np.int64(n_fine_trans),
            fine_translations=np.asarray(fine_translations, dtype=np.float32),
            rotations=np.asarray(per_image_inputs["oversampled_rots"][image_idx], dtype=np.float32),
            oversampled_rot_indices=np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64),
            parent_map=np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32),
            candidate_mask=mask_np[row, :cnt, :],
            scores_with_prior=scores_row,
            scores_pre_prior=pre_prior,
            probs=probs_np[row, :cnt, :],
            rotation_log_prior=rot_prior_np[row, :cnt],
            translation_log_prior=trans_prior_np[row],
            shifted_corrected=(
                shifted_corrected_np[row] if shifted_corrected_np is not None else np.empty((0,), dtype=np.complex64)
            ),
            ctf2_over_nv_score=ctf2_np[row],
            proj_half=proj_np[row, :cnt, :],
            half_weights=np.asarray(half_weights_used, dtype=np.float64),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32) if window_indices is not None else np.empty((0,), dtype=np.int32)
            ),
        )


def _prepare_bucket_io(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    noise_variance_half,
    fine_translations,
    config,
    n_trans,
    score_with_masked_images,
    half_spectrum_scoring,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    return_direct_scoring_io=False,
    score_only=False,
    score_mode="gaussian",
    window_indices=None,
):
    """Run preprocessing for a batch of images (translations tiled, CTF/noise ratios).

    Mirrors the ``run_em``/``_preprocess_batch`` pipeline exactly so the
    bucketed sparse pass-2 path is bit-for-bit identical to calling
    ``run_em`` per image.
    """
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")

    image_shape = config.image_shape
    use_normalized_cc = score_mode == "normalized_cc"
    batch_size = int(batch.shape[0])
    integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices, batch=batch)
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied:
        batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half = config.compute_ctf_half(ctf_params)
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half
    ctf2_score_half = ctf_half**2

    # Raw processed half-spectrum images (BEFORE any per-image correction).
    # The score path uses masked images iff ``score_with_masked_images`` is True,
    # while the reconstruction path always uses the unmasked (raw) images.
    processed_score_half_raw = process_half_image(experiment_dataset, batch, score_with_masked_images)
    if score_with_masked_images:
        processed_recon_half_raw = process_half_image(experiment_dataset, batch, False)
    else:
        processed_recon_half_raw = processed_score_half_raw

    if use_normalized_cc:
        # RELION firstiter_cc uses unweighted image power over the same Fourier
        # window as the score denominator, with no Hermitian doubling.
        abs2_half = jnp.abs(processed_score_half_raw) ** 2
        if window_indices is not None:
            abs2_half = abs2_half[:, window_indices]
        batch_norm = jnp.sum(abs2_half, axis=-1, keepdims=True).real
    else:
        # batch_norm starts from raw processed-score images, then follows dense
        # run_em's image-only correction convention below.
        norm_half_weights = make_half_image_weights(image_shape)
        batch_norm = jnp.sum(
            (jnp.abs(processed_score_half_raw) ** 2 / noise_variance_half) * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real

    score_weighted_half = processed_score_half_raw * ctf_half / noise_variance_half
    recon_weighted_half = processed_recon_half_raw * ctf_half / noise_variance_half
    sparse_score_input_half = processed_score_half_raw * ctf_half if use_normalized_cc else processed_score_half_raw
    processed_score_half_for_noise = processed_score_half_raw

    if scale_corrections is not None:
        batch_scale = jnp.asarray(np.asarray(scale_corrections)[np.asarray(image_indices)])
    else:
        batch_scale = jnp.ones(batch_size, dtype=ctf_half.dtype)

    # Per-image image corrections follow dense run_em's image-only convention.
    if image_corrections is not None:
        batch_corr = jnp.asarray(np.asarray(image_corrections)[np.asarray(image_indices)])
        image_only_corr = batch_corr / batch_scale
        # Note: corrections are applied to the per-translation-tiled arrays in
        # run_em, but multiplication by a per-image scalar commutes with the
        # tiling and shifting so we apply it before tiling for efficiency.
        score_weighted_half = score_weighted_half * batch_corr[:, None]
        recon_weighted_half = recon_weighted_half * batch_corr[:, None]
        if return_direct_scoring_io:
            direct_raw_corr = batch_corr / batch_scale
            if use_normalized_cc:
                sparse_score_input_half = sparse_score_input_half * batch_corr[:, None]
            else:
                sparse_score_input_half = sparse_score_input_half * direct_raw_corr[:, None]
        batch_norm = batch_norm * (image_only_corr**2)[:, None]
        processed_score_half_for_noise = processed_score_half_for_noise * image_only_corr[:, None]

    # Per-image scale correction on CTF^2/noise.
    if scale_corrections is not None:
        ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
        ctf2_score_half = ctf2_score_half * (batch_scale**2)[:, None]
        if return_direct_scoring_io:
            if not use_normalized_cc:
                sparse_score_input_half = sparse_score_input_half / batch_scale[:, None]

    if return_direct_scoring_io and not use_normalized_cc:
        ctf_safe = jnp.abs(ctf_half) > 1e-8
        sparse_score_input_half = jnp.where(
            ctf_safe,
            sparse_score_input_half / ctf_half,
            sparse_score_input_half,
        )
    if score_only and not return_direct_scoring_io:
        raise ValueError("score-only sparse pass-2 requires direct scoring I/O")

    # Per-image pre-centering: phase shift in Fourier space after scalar corrections.
    if image_pre_shifts is not None and not real_space_pre_shift_applied:
        batch_shifts = jnp.asarray(np.asarray(image_pre_shifts)[np.asarray(image_indices)])
        phase_factors = half_image_phase_factors(image_shape, batch_shifts)
        if not score_only:
            score_weighted_half = score_weighted_half * phase_factors
            recon_weighted_half = recon_weighted_half * phase_factors
        if return_direct_scoring_io:
            sparse_score_input_half = sparse_score_input_half * phase_factors

    translation_phases_half = half_translation_phase_table(fine_translations, image_shape)
    if score_only:
        shifted_score_half = None
        shifted_recon_half = None
        shifted_score_half_with_dc = None
        ctf2_over_nv_half_with_dc = None
        processed_score_half_for_noise = None
    else:
        shifted_score_half = apply_half_translation_phases(score_weighted_half, translation_phases_half)
        if score_with_masked_images:
            shifted_recon_half = apply_half_translation_phases(recon_weighted_half, translation_phases_half)
        else:
            shifted_recon_half = shifted_score_half
        shifted_score_half_with_dc = shifted_score_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half

    shifted_corrected_score_half = None
    if return_direct_scoring_io:
        shifted_corrected_score_half = apply_half_translation_phases(
            sparse_score_input_half,
            translation_phases_half,
        )

    if half_spectrum_scoring and not use_normalized_cc:
        dc_shell_idx = make_shell_indices_half(image_shape)
        dc_mask = dc_shell_idx == 0
        if not score_only:
            shifted_score_half = jnp.where(dc_mask[None, :], 0.0, shifted_score_half)
        ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    if return_direct_scoring_io and use_normalized_cc:
        inv_xi2 = (1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))).astype(
            precision_policy.score_real_dtype,
        )
        shifted_corrected_score_half = shifted_corrected_score_half * jnp.repeat(inv_xi2, n_trans, axis=0)
        ctf2_over_nv_half = ctf2_score_half * inv_xi2
    if score_only:
        ctf2_over_nv_half = ctf2_over_nv_half.astype(precision_policy.score_real_dtype)
    else:
        (
            shifted_score_half,
            shifted_recon_half,
            shifted_score_half_with_dc,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
        ) = precision_policy.cast_local_preprocessed_inputs(
            shifted_score_half,
            shifted_recon_half,
            shifted_score_half_with_dc,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
        )
    if return_direct_scoring_io:
        shifted_corrected_score_half = shifted_corrected_score_half.astype(
            precision_policy.score_complex_dtype,
        )

    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        shifted_corrected_score_half,
    )


def compute_pass2_stats_sparse_bucketed(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    oversampling_order,
    current_size,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation,
    normalization_log_z=None,
    normalization_other_score_log_z=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    rotation_block_size_for_quantization=5000,
    fine_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
):
    """Bucketed batched implementation of sparse pass-2 oversampling.

    Returns the same tuple as ``compute_pass2_stats_sparse``.
    """
    from recovar.em.sampling import (
        get_oversampled_translation_grid,
        rotation_grid_size,
    )

    if relion_firstiter_score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            "relion_firstiter_score_mode must be 'gaussian' or 'normalized_cc', "
            f"got {relion_firstiter_score_mode!r}",
        )
    winner_take_all = bool(relion_firstiter_winner_take_all)
    if bool(disable_adjoint_y) != bool(disable_adjoint_ctf):
        raise NotImplementedError("Sparse pass-2 currently supports disabling both M-step adjoints together")
    score_only = bool(disable_adjoint_y and disable_adjoint_ctf)
    if return_score_log_z_only:
        if not score_only:
            raise ValueError("return_score_log_z_only requires both M-step adjoints to be disabled")
        if normalization_log_z is not None:
            raise ValueError("return_score_log_z_only cannot be combined with normalization_log_z")
        if normalization_other_score_log_z is not None:
            raise ValueError("return_score_log_z_only cannot be combined with normalization_other_score_log_z")
        if accumulate_noise:
            raise ValueError("return_score_log_z_only cannot accumulate noise")
    if normalization_log_z is not None and normalization_other_score_log_z is not None:
        raise ValueError("normalization_log_z and normalization_other_score_log_z are mutually exclusive")
    if normalization_other_score_log_z is not None and not return_score_log_z:
        raise ValueError("normalization_other_score_log_z requires return_score_log_z=True")
    if score_only and accumulate_noise:
        raise ValueError("Sparse pass-2 score-only mode is incompatible with accumulate_noise=True")

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)

    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    window_spec_kwargs = {}
    if relion_firstiter_score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    budget_window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    device_memory_bytes = _device_memory_limit_bytes()

    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    use_half_volume_mstep = bool(relion_half_volume_mstep)
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape) if use_half_volume_mstep else recon_volume_shape
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_accum_dtype = experiment_dataset.dtype

    # Projection volume + padding
    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            volume,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = volume
        proj_volume_shape = volume_shape

    # Fine translations and prior mapping
    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    if fine_translations_override is None and fine_translation_parent_override is None:
        fine_translations, fine_translation_parent = get_oversampled_translation_grid(
            translations_np,
            translation_step,
            oversampling_order=oversampling_order,
        )
        fine_translations = np.asarray(fine_translations, dtype=np.float32)
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations = np.asarray(fine_translations_override, dtype=np.float32)
        fine_translation_parent = np.asarray(fine_translation_parent_override, dtype=np.int32)
        if fine_translations.ndim != 2 or fine_translations.shape[1] != translations_np.shape[1]:
            raise ValueError(
                "fine_translations_override must have shape "
                f"(n_fine_trans, {translations_np.shape[1]}), got {fine_translations.shape}",
            )
        if fine_translation_parent.shape != (fine_translations.shape[0],):
            raise ValueError(
                "fine_translation_parent_override must have shape "
                f"({fine_translations.shape[0]},), got {fine_translation_parent.shape}",
            )
        if int(fine_translation_parent.max(initial=-1)) >= n_coarse_trans:
            raise ValueError("fine_translation_parent_override values must be < n_coarse_trans")
    else:
        raise ValueError(
            "fine_translations_override and fine_translation_parent_override must be provided together",
        )
    n_fine_trans = fine_translations.shape[0]

    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations_np.shape[1],
    )

    # Translation prior in the fine grid
    if translation_log_prior is None:
        fine_translation_prior_2d = None
    else:
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior_np.ndim == 1:
            fine_tp = translation_log_prior_np[fine_translation_parent]
            fine_translation_prior_2d = np.broadcast_to(fine_tp, (n_images, n_fine_trans)).astype(
                np.float32, copy=False
            )
        elif translation_log_prior_np.ndim == 2:
            fine_translation_prior_2d = translation_log_prior_np[:, fine_translation_parent].astype(
                np.float32, copy=False
            )
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions",
            )

    # Per-image hypothesis prep
    prep_t0 = time.time()
    per_image_inputs = _prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_rotations_override=fine_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
    )
    prep_s = time.time() - prep_t0

    local_rot_counts = [int(rots.shape[0]) for rots in per_image_inputs["oversampled_rots"]]
    valid_candidate_counts = [int(np.asarray(m).sum()) for m in per_image_inputs["candidate_mask"]]

    # Bucket.  The default cap intentionally allows multi-image buckets for
    # broad soft posteriors; the old 100k cap fragmented 100k/256 K=4 into
    # tens of thousands of one-image launches on A100.
    max_hypotheses_per_microbatch = _max_hypotheses_per_microbatch_for_pass(
        score_only=score_only,
        use_window=budget_window_spec.use_window,
        has_external_normalization=normalization_log_z is not None or normalization_other_score_log_z is not None,
        dump_pass2_operands=bool(os.environ.get("RECOVAR_PASS2_DUMP_DIR")),
        n_score_pixels=budget_window_spec.n_score,
        device_memory_bytes=device_memory_bytes,
    )
    has_external_normalization = normalization_log_z is not None or normalization_other_score_log_z is not None
    max_translation_tile_bytes = _max_translation_tile_bytes_for_pass(
        device_memory_bytes,
        has_external_normalization=has_external_normalization,
    )
    max_images_per_microbatch = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_translation_tile_bytes,
    )
    bucket_t0 = time.time()
    buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
    )
    bucket_s = time.time() - bucket_t0

    logger.info(
        "Sparse pass-2 bucketing: %d images -> %d buckets (%s; "
        "max_hypotheses_per_microbatch=%d, max_images_per_microbatch=%d, "
        "max_translation_tile_bytes=%d, n_score_pixels=%d, device_memory_gib=%.2f)",
        n_images,
        len(buckets),
        _bucket_summary(buckets),
        max_hypotheses_per_microbatch,
        max_images_per_microbatch,
        max_translation_tile_bytes,
        int(budget_window_spec.n_score),
        (-1.0 if device_memory_bytes is None else device_memory_bytes / float(1024**3)),
    )
    logger.info("Sparse pass-2 setup timing: hypothesis_prep=%.2fs bucket=%.2fs", prep_s, bucket_s)
    logger.info(
        "Sparse pass-2 M-step: using %s backprojection",
        "native half-volume" if use_half_volume_mstep else "full-volume",
    )

    # Output accumulators (volume_size matches what original returned: full N**3)
    if return_score_log_z_only:
        Ft_y_total = None
        Ft_ctf_total = None
        hard_assignment = None
        best_rotations = None
        best_rotation_indices = None
    else:
        Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_accum_dtype)
        Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_accum_dtype)
        hard_assignment = np.empty(n_images, dtype=np.int32)
        best_rotations = np.empty((n_images, 3, 3), dtype=np.float32)
        best_rotation_indices = np.empty(n_images, dtype=np.int64)

    # K-class assignment depends on small inter-class score deltas after adding
    # a large image-power offset. Keep these in float64 like dense run_em.
    log_evidence = np.empty(n_images, dtype=np.float64) if (return_stats or return_score_log_z_only) else None
    best_log_score = np.empty(n_images, dtype=np.float64) if return_stats else None
    max_posterior = np.empty(n_images, dtype=np.float32) if return_stats else None
    rotation_posterior_sums = np.zeros(n_coarse_rot, dtype=np.float64) if return_stats else None
    score_log_z = (
        np.empty(n_images, dtype=np.float64)
        if ((return_stats and return_score_log_z) or return_score_log_z_only)
        else None
    )

    noise_wsum_total = None
    noise_img_power_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        noise_wsum_total = np.zeros(n_shells, dtype=np.float64)
        noise_img_power_total = np.zeros(n_shells, dtype=np.float64)

    # Forward-model config & half/window precomputes
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    window_indices_np = window_spec.score_indices_np
    window_indices = window_spec.score_indices
    recon_window_indices = window_spec.recon_indices
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    if accumulate_noise:
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)

    normalization_log_z_np = None
    if normalization_log_z is not None:
        normalization_log_z_np = np.asarray(normalization_log_z, dtype=np.float64)
        if normalization_log_z_np.shape != (n_images,):
            raise ValueError(
                "normalization_log_z must have shape "
                f"({n_images},), got {normalization_log_z_np.shape}",
            )
    normalization_other_score_log_z_np = None
    if normalization_other_score_log_z is not None:
        normalization_other_score_log_z_np = np.asarray(normalization_other_score_log_z, dtype=np.float64)
        if normalization_other_score_log_z_np.shape != (n_images,):
            raise ValueError(
                "normalization_other_score_log_z must have shape "
                f"({n_images},), got {normalization_other_score_log_z_np.shape}",
            )
    dump_pass2_operands = bool(os.environ.get("RECOVAR_PASS2_DUMP_DIR"))

    projection_cache = None
    if fine_rotations_override is not None and not dump_pass2_operands:
        n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
        transient_projection_bytes = n_fine_rot * n_half * np.dtype(np.complex64).itemsize
        if not score_only and not use_window:
            transient_projection_bytes += n_fine_rot * n_half * np.dtype(np.float32).itemsize
        max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
        if transient_projection_bytes <= max_projection_cache_bytes:
            cache_t0 = time.time()
            projection_kwargs = window_spec.projection_kwargs(return_abs2=False if (use_window or score_only) else None)
            proj_half_cache_flat, proj_abs2_cache_flat = _compute_projections_block(
                mean_for_proj,
                jnp.asarray(fine_rotations_override, dtype=jnp.float32),
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
            if use_window:
                projection_cache = {
                    "score": proj_half_cache_flat[:, window_indices],
                    "recon": None if score_only else proj_half_cache_flat[:, recon_window_indices],
                    "recon_abs2": None,
                }
                if not score_only:
                    projection_cache["recon_abs2"] = jnp.abs(projection_cache["recon"]) ** 2
                del proj_half_cache_flat
            else:
                projection_cache = {
                    "score": proj_half_cache_flat,
                    "recon": None if score_only else proj_half_cache_flat,
                    "recon_abs2": None if score_only else proj_abs2_cache_flat,
                }
            logger.info(
                "Sparse pass-2 projection cache: cached %d fine rotations in %.2fs (estimated transient %.2f GiB)",
                n_fine_rot,
                time.time() - cache_t0,
                transient_projection_bytes / float(1024**3),
            )
        else:
            logger.info(
                "Sparse pass-2 projection cache skipped: estimated transient %.2f GiB exceeds cap %.2f GiB",
                transient_projection_bytes / float(1024**3),
                max_projection_cache_bytes / float(1024**3),
            )
    overall_t0 = time.time()

    bucket_group_stats = _bucket_group_stats(buckets)
    last_bucket_size_logged = None
    group_t0 = None
    for bucket_meta in buckets:
        bucket_arrays = _build_bucket_arrays(
            bucket_meta,
            per_image_inputs,
            n_fine_trans,
        )
        image_indices = bucket_arrays["image_indices"]
        bucket_size = int(bucket_arrays["bucket_size"])
        if bucket_size != last_bucket_size_logged:
            if last_bucket_size_logged is not None and group_t0 is not None:
                prev_chunks, prev_images = bucket_group_stats[last_bucket_size_logged]
                prev_wall = time.time() - group_t0
                logger.info(
                    "Sparse pass-2 bucket group done: bucket_size=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
                    last_bucket_size_logged,
                    prev_chunks,
                    prev_images,
                    prev_wall,
                    prev_images / max(prev_wall, 1e-9),
                )
            group_chunks, group_images = bucket_group_stats[bucket_size]
            logger.info(
                "Sparse pass-2 bucket group start: bucket_size=%d chunks=%d images=%d",
                bucket_size,
                group_chunks,
                group_images,
            )
            last_bucket_size_logged = bucket_size
            group_t0 = time.time()
        batch = int(image_indices.shape[0])

        # Fetch images (the dataset may reorder; we reorder our padded arrays
        # to match.)
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
        batch_data = jnp.asarray(batch_data)
        # Reorder bucket arrays to match fetched_indices
        if not np.array_equal(np.asarray(fetched_indices), image_indices):
            (
                rotations,
                rotation_indices,
                log_prior,
                candidate_mask,
                parent_map_padded,
                actual_counts,
            ) = _reorder_to_indices(
                np.asarray(fetched_indices),
                image_indices,
                bucket_arrays["rotations"],
                bucket_arrays["rotation_indices"],
                bucket_arrays["log_prior"],
                bucket_arrays["candidate_mask"],
                bucket_arrays["parent_map"],
                bucket_arrays["actual_counts"],
            )
            image_indices = np.asarray(fetched_indices)
        else:
            rotations = bucket_arrays["rotations"]
            rotation_indices = bucket_arrays["rotation_indices"]
            log_prior = bucket_arrays["log_prior"]
            candidate_mask = bucket_arrays["candidate_mask"]
            parent_map_padded = bucket_arrays["parent_map"]
            actual_counts = bucket_arrays["actual_counts"]

        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                image_indices,
                batch_size=batch,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                fine_translations,
                centers,
                experiment_dataset.voxel_size,
            )

        # Translation prior for this bucket (per-image)
        if fine_translation_prior_2d is None:
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=jnp.float32)
        else:
            bucket_translation_prior = jnp.asarray(fine_translation_prior_2d[image_indices], dtype=jnp.float32)

        # Preprocess
        (
            shifted_score_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
            shifted_score_half_with_dc,
            processed_score_half_for_noise,
            shifted_corrected_score_half,
        ) = _prepare_bucket_io(
            experiment_dataset,
            batch_data,
            ctf_params,
            image_indices,
            noise_variance_half,
            fine_translations,
            config,
            n_fine_trans,
            score_with_masked_images,
            half_spectrum_scoring,
            image_corrections,
            scale_corrections,
            image_pre_shifts,
            use_float64_scoring,
            return_direct_scoring_io=True,
            score_only=score_only,
            score_mode=relion_firstiter_score_mode,
            window_indices=window_indices,
        )

        # Window gather (if applicable)
        if use_window:
            ctf2_over_nv_score = ctf2_over_nv_half[:, window_indices]
            shifted_corrected_score = shifted_corrected_score_half[:, window_indices]
            if score_only:
                shifted_score = None
                shifted_recon = None
                ctf2_over_nv_recon = None
                shifted_noise = None
            else:
                shifted_score = shifted_score_half[:, window_indices]
                shifted_recon = shifted_recon_half[:, recon_window_indices]
                ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, recon_window_indices]
                shifted_noise = shifted_score_half_with_dc[:, recon_window_indices]
        else:
            ctf2_over_nv_score = ctf2_over_nv_half
            shifted_corrected_score = shifted_corrected_score_half
            if score_only:
                shifted_score = None
                shifted_recon = None
                ctf2_over_nv_recon = None
                shifted_noise = None
            else:
                shifted_score = shifted_score_half
                shifted_recon = shifted_recon_half
                ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
                shifted_noise = shifted_score_half_with_dc

        flat_rotations = flatten_bucket_rotations(jnp.asarray(rotations))
        flat_backproject_rotations = flat_rotations
        if projection_cache is not None:
            rotation_indices_jax = jnp.asarray(rotation_indices, dtype=jnp.int32)
            proj_half = projection_cache["score"][rotation_indices_jax]
            if score_only:
                proj_for_noise = None
                proj_abs2_for_noise = None
            else:
                proj_for_noise = projection_cache["recon"][rotation_indices_jax]
                proj_abs2_for_noise = projection_cache["recon_abs2"][rotation_indices_jax]
        else:
            # Project (B*R, 3, 3) -> (B*R, n_half) -> reshape (B, R, n_half)
            projection_kwargs = window_spec.projection_kwargs(return_abs2=False if (use_window or score_only) else None)
            proj_half_flat, proj_abs2_half_flat = _compute_projections_block(
                mean_for_proj,
                flat_rotations,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
            if use_window:
                proj_half = proj_half_flat[:, window_indices].reshape(batch, bucket_size, n_windowed)
                if score_only:
                    proj_for_noise = None
                    proj_abs2_for_noise = None
                else:
                    proj_for_noise = proj_half_flat[:, recon_window_indices].reshape(
                        batch,
                        bucket_size,
                        n_recon_windowed,
                    )
                    proj_abs2_for_noise = jnp.abs(proj_for_noise) ** 2
            else:
                proj_half = proj_half_flat.reshape(batch, bucket_size, n_half)
                if score_only:
                    proj_for_noise = None
                    proj_abs2_for_noise = None
                else:
                    proj_abs2_for_noise = proj_abs2_half_flat.reshape(batch, bucket_size, n_half)
                    proj_for_noise = proj_half

        if not score_only:
            proj_for_noise, proj_abs2_for_noise = precision_policy.cast_local_noise_projection_scores(
                proj_for_noise,
                proj_abs2_for_noise,
            )

        # Score: (B, R, T)
        shifted_corrected_score_split = shifted_corrected_score.reshape(batch, n_fine_trans, -1)
        direct_half_weights = half_weights_windowed if use_window else half_weights
        if relion_firstiter_score_mode == "normalized_cc":
            scores = _score_pass2_bucket_normalized_cc(
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                jnp.asarray(candidate_mask),
            )
        else:
            scores = _score_pass2_bucket_relion_gpu_diff2(
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                jnp.asarray(log_prior),
                bucket_translation_prior,
                jnp.asarray(candidate_mask),
            )

        probs = None
        if return_score_log_z_only:
            log_Z = _logsumexp_pass2_bucket_score_only(scores)
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_Z_np = np.asarray(log_Z, dtype=np.float64)
            for row, image_idx in enumerate(image_indices.tolist()):
                if np.isfinite(log_Z_np[row]):
                    log_evidence[image_idx] = float(log_Z_np[row] + log_score_offset[row])
                    score_log_z[image_idx] = float(log_Z_np[row])
                else:
                    log_evidence[image_idx] = -np.inf
                    score_log_z[image_idx] = -np.inf
            continue
        local_score_log_z = None
        if (
            score_only
            and normalization_log_z_np is None
            and normalization_other_score_log_z_np is None
            and not dump_pass2_operands
        ):
            log_Z, best_log_score_bucket, best_argmax, max_posterior_bucket = _normalize_pass2_bucket_score_only(
                scores,
            )
        elif normalization_log_z_np is None and normalization_other_score_log_z_np is None:
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = _normalize_pass2_bucket(scores)
        elif normalization_log_z_np is not None:
            bucket_log_z = jnp.asarray(normalization_log_z_np[image_indices], dtype=scores.real.dtype)
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                _normalize_pass2_bucket_with_log_z(scores, bucket_log_z)
            )
        else:
            local_score_log_z = _logsumexp_pass2_bucket_score_only(scores)
            bucket_other_log_z = jnp.asarray(
                normalization_other_score_log_z_np[image_indices],
                dtype=local_score_log_z.dtype,
            )
            bucket_log_z = jnp.logaddexp(local_score_log_z, bucket_other_log_z).astype(scores.real.dtype)
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                _normalize_pass2_bucket_with_log_z(scores, bucket_log_z)
            )
        if winner_take_all:
            if probs is not None:
                probs = _winner_take_all_bucket_probs(scores, best_argmax, best_log_score_bucket)
            max_posterior_bucket = jnp.where(
                jnp.isfinite(best_log_score_bucket),
                jnp.ones_like(max_posterior_bucket),
                jnp.zeros_like(max_posterior_bucket),
            )

        actual_counts_arr = np.asarray(actual_counts, dtype=np.int64)
        if probs is not None:
            _maybe_dump_pass2_bucket(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                per_image_inputs=per_image_inputs,
                current_size=current_size,
                n_fine_trans=n_fine_trans,
                fine_translations=fine_translations,
                scores=scores,
                probs=probs,
                rotation_log_prior=jnp.asarray(log_prior),
                translation_log_prior=bucket_translation_prior,
                candidate_mask=jnp.asarray(candidate_mask),
                ctf2_over_nv_score=ctf2_over_nv_score,
                proj_half=proj_half,
                half_weights_used=half_weights_windowed if use_window else half_weights,
                window_indices=window_indices_np,
                shifted_corrected_score_split=shifted_corrected_score_split,
            )

        ctf_probs = None
        if not score_only:
            # M-step accumulation: posterior-weighted sums per (image, rot).
            shifted_recon_split = shifted_recon.reshape(batch, n_fine_trans, -1)
            summed = compute_local_weighted_sums(probs, shifted_recon_split)  # (B, R, N)
            ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)  # (B, R, N)

            # Backproject (use flat_rotations + flat summed/ctf_probs).
            # Padded rotations contribute zero because their probs == 0
            # (candidate_mask=False -> score=-inf -> exp(-inf)=0).
            if use_window:
                Ft_y_total = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(summed),
                    recon_window_indices,
                    flat_backproject_rotations,
                    Ft_y_total,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                    float(current_size // 2),
                )
                Ft_ctf_total = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(ctf_probs),
                    recon_window_indices,
                    flat_backproject_rotations,
                    Ft_ctf_total,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                    float(current_size // 2),
                )
            else:
                Ft_y_total = _adjoint_slice_volume_half(
                    flatten_bucket_rows(summed),
                    flat_backproject_rotations,
                    Ft_y_total,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                )
                Ft_ctf_total = _adjoint_slice_volume_half(
                    flatten_bucket_rows(ctf_probs),
                    flat_backproject_rotations,
                    Ft_ctf_total,
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                )

        # Noise accumulation
        if accumulate_noise:
            if translation_sqdist_ang is not None:
                translation_posterior = np.asarray(jnp.sum(probs, axis=1), dtype=np.float64)
                noise_sigma2_offset_total += float(
                    np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                )
            # ``processed_score_half_for_noise`` is already adjusted for dense
            # run_em's image-only correction convention when applicable.
            batch_img_power = jnp.sum(jnp.abs(processed_score_half_for_noise) ** 2, axis=0).astype(jnp.float32)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            noise_img_power_total += np.asarray(batch_img_power_shells, dtype=np.float64)
            noise_sumw_total += float(batch)

            if half_spectrum_scoring:
                shifted_noise_split = shifted_noise.reshape(batch, n_fine_trans, -1)
            else:
                shifted_noise_split = shifted_score.reshape(batch, n_fine_trans, -1)
            summed_masked_noise = compute_local_weighted_sums(probs, shifted_noise_split)
            block_noise_shells, _, _ = _compute_noise_block(
                flatten_bucket_rows(proj_for_noise),
                flatten_bucket_rows(proj_abs2_for_noise),
                flatten_bucket_rows(summed_masked_noise),
                flatten_bucket_rows(ctf_probs),
                noise_variance_for_noise,
                shell_indices_noise,
                n_shells,
            )
            noise_wsum_total += np.asarray(block_noise_shells, dtype=np.float64)

        # Decode best assignment and write per-image stats
        best_argmax_np = np.asarray(best_argmax, dtype=np.int64)
        best_rot_idx = best_argmax_np // n_fine_trans
        best_trans_idx = best_argmax_np % n_fine_trans

        # Sanity check: padded rotations should never be chosen (probs == 0 there).
        if np.any(best_rot_idx >= actual_counts_arr):
            bad = np.flatnonzero(best_rot_idx >= actual_counts_arr)
            raise RuntimeError(
                f"Bucket pass-2: best rotation index points into padding for images {bad.tolist()} "
                f"(best_rot_idx={best_rot_idx[bad].tolist()}, actual_counts={actual_counts_arr[bad].tolist()})"
            )

        for row, image_idx in enumerate(image_indices.tolist()):
            r = int(best_rot_idx[row])
            t = int(best_trans_idx[row])
            hard_assignment[image_idx] = r * n_fine_trans + t
            best_rotations[image_idx] = per_image_inputs["oversampled_rots"][image_idx][r]
            best_rotation_indices[image_idx] = per_image_inputs["oversampled_rot_indices"][image_idx][r]

        if return_stats:
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_Z_np = np.asarray(log_Z, dtype=np.float64)
            class_log_Z_np = (
                np.asarray(local_score_log_z, dtype=np.float64) if local_score_log_z is not None else log_Z_np
            )
            best_log_score_np = np.asarray(best_log_score_bucket, dtype=np.float64)
            max_posterior_np = np.asarray(max_posterior_bucket, dtype=np.float32)
            for row, image_idx in enumerate(image_indices.tolist()):
                if np.isfinite(best_log_score_np[row]):
                    log_evidence[image_idx] = float(class_log_Z_np[row] + log_score_offset[row])
                    if score_log_z is not None:
                        score_log_z[image_idx] = float(class_log_Z_np[row])
                else:
                    log_evidence[image_idx] = -np.inf
                    if score_log_z is not None:
                        score_log_z[image_idx] = -np.inf
                best_log_score[image_idx] = float(best_log_score_np[row] + log_score_offset[row])
                max_posterior[image_idx] = float(max_posterior_np[row])

            # rotation_posterior_sums: scatter per (image, rot) probability mass back
            # to the parent coarse rotation indices.
            if probs is not None:
                probs_sum_t = np.asarray(jnp.sum(probs, axis=-1), dtype=np.float64)  # (B, R)
                for row, image_idx in enumerate(image_indices.tolist()):
                    cnt = int(actual_counts[row])
                    if cnt == 0:
                        continue
                    unique_rot_image = per_image_inputs["unique_rot"][image_idx]
                    parent_map_image = per_image_inputs["parent_map"][image_idx]
                    # Map each oversampled rot back to its coarse-grid rotation index.
                    coarse_rot_indices = unique_rot_image[parent_map_image]
                    np.add.at(rotation_posterior_sums, coarse_rot_indices, probs_sum_t[row, :cnt])

    if last_bucket_size_logged is not None and group_t0 is not None:
        group_chunks, group_images = bucket_group_stats[last_bucket_size_logged]
        group_wall = time.time() - group_t0
        logger.info(
            "Sparse pass-2 bucket group done: bucket_size=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
            last_bucket_size_logged,
            group_chunks,
            group_images,
            group_wall,
            group_images / max(group_wall, 1e-9),
        )

    em_wall = time.time() - overall_t0
    logger.info(
        "Sparse pass-2 (bucketed): %d images, %d buckets, %.2fs E+M; "
        "median local rot=%d, mean local rot=%.1f, median valid candidates/image=%d",
        n_images,
        len(buckets),
        em_wall,
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
    )

    if return_score_log_z_only:
        return log_evidence, score_log_z

    if score_only:
        full_volume_size = int(np.prod(recon_volume_shape))
        Ft_y_total = jnp.zeros(full_volume_size, dtype=recon_accum_dtype)
        Ft_ctf_total = jnp.zeros(full_volume_size, dtype=recon_accum_dtype)
    elif use_half_volume_mstep:
        _maybe_dump_native_half_mstep(
            Ft_y_total,
            Ft_ctf_total,
            current_size=current_size,
            n_images=n_images,
            recon_volume_shape=recon_volume_shape,
            stage="pre_x0",
        )
        Ft_y_total, Ft_ctf_total = enforce_half_volume_x0(
            Ft_y_total,
            Ft_ctf_total,
            recon_volume_shape,
            logger=logger,
            label="Sparse pass-2",
        )
        _maybe_dump_native_half_mstep(
            Ft_y_total,
            Ft_ctf_total,
            current_size=current_size,
            n_images=n_images,
            recon_volume_shape=recon_volume_shape,
            stage="post_x0",
        )
        Ft_y_total, Ft_ctf_total = half_volume_accumulators_to_full(
            Ft_y_total,
            Ft_ctf_total,
            recon_volume_shape,
        )

    best_translations = fine_translations[hard_assignment % n_fine_trans]

    merged_noise_stats = None
    if accumulate_noise:
        merged_noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum_total,
            wsum_img_power=noise_img_power_total,
            wsum_sigma2_offset=noise_sigma2_offset_total,
            sumw=noise_sumw_total,
        )

    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=log_evidence,
            best_log_score_per_image=best_log_score,
            max_posterior_per_image=max_posterior,
            rotation_posterior_sums=rotation_posterior_sums,
        )
        result = (
            Ft_y_total,
            Ft_ctf_total,
            hard_assignment,
            best_rotations,
            best_translations,
            best_rotation_indices,
            relion_stats,
        )
        if return_score_log_z:
            result = result + (score_log_z,)
        if accumulate_noise:
            result = result + (merged_noise_stats,)
        return result

    result = (
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_indices,
    )
    if accumulate_noise:
        result = result + (merged_noise_stats,)
    return result


def _shared_k_class_noise_variance(noise_variance, n_classes: int):
    noise_np = np.asarray(noise_variance)
    if noise_np.ndim >= 2 and int(noise_np.shape[0]) == int(n_classes):
        first = noise_np[0]
        if not np.allclose(noise_np, first[None, ...], rtol=0.0, atol=0.0):
            return None
        return first
    return noise_variance


def compute_k_class_pass2_stats_sparse_fused(
    experiment_dataset,
    volumes,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices_by_class,
    *,
    rotation_log_priors_by_class,
    nside_level,
    disc_type,
    oversampling_order,
    current_size,
    translation_step=None,
    score_with_masked_images=False,
    return_stats=True,
    accumulate_noise=False,
    translation_log_prior=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    use_float64_scoring=False,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    rotation_block_size_for_quantization=5000,
    fine_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
) -> SparseKClassPass2FusedResult:
    """Evaluate K-class sparse pass-2 in one joint class-normalized sweep.

    This mirrors RELION's fine-pass semantics: all class-local scores are
    normalized by one per-image class x pose denominator before M-step
    accumulation.  The exact fused implementation currently requires a shared
    class noise model; callers should fall back to the existing per-class path
    when noise differs by class.
    """

    from recovar.em.sampling import (
        get_oversampled_translation_grid,
        rotation_grid_size,
    )

    if not return_stats:
        raise ValueError("fused sparse K-class pass-2 requires return_stats=True")
    if relion_firstiter_score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            "relion_firstiter_score_mode must be 'gaussian' or 'normalized_cc', "
            f"got {relion_firstiter_score_mode!r}",
        )

    volumes = jnp.asarray(volumes)
    n_classes = int(volumes.shape[0])
    if len(significant_sample_indices_by_class) != n_classes:
        raise ValueError("significant_sample_indices_by_class must match class count")
    if len(rotation_log_priors_by_class) != n_classes:
        raise ValueError("rotation_log_priors_by_class must match class count")
    shared_noise_variance = _shared_k_class_noise_variance(noise_variance, n_classes)
    if shared_noise_variance is None:
        raise NotImplementedError("fused sparse K-class pass-2 requires shared class noise variance")

    n_images = int(experiment_dataset.n_units)
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    winner_take_all = bool(relion_firstiter_winner_take_all)
    window_spec_kwargs = {}
    if relion_firstiter_score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    budget_window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    device_memory_bytes = _device_memory_limit_bytes()

    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    use_half_volume_mstep = bool(relion_half_volume_mstep)
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape) if use_half_volume_mstep else recon_volume_shape
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_accum_dtype = experiment_dataset.dtype

    mean_for_proj_by_class = []
    proj_volume_shape = volume_shape
    for class_index in range(n_classes):
        class_volume = volumes[class_index]
        if projection_padding_factor > 1:
            from recovar.reconstruction.relion_functions import pad_volume_for_projection

            mean_for_proj, proj_volume_shape = pad_volume_for_projection(
                class_volume,
                volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
        else:
            mean_for_proj = class_volume
        mean_for_proj_by_class.append(mean_for_proj)

    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    if fine_translations_override is None and fine_translation_parent_override is None:
        fine_translations, fine_translation_parent = get_oversampled_translation_grid(
            translations_np,
            translation_step,
            oversampling_order=oversampling_order,
        )
        fine_translations = np.asarray(fine_translations, dtype=np.float32)
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations = np.asarray(fine_translations_override, dtype=np.float32)
        fine_translation_parent = np.asarray(fine_translation_parent_override, dtype=np.int32)
    else:
        raise ValueError(
            "fine_translations_override and fine_translation_parent_override must be provided together",
        )
    n_fine_trans = int(fine_translations.shape[0])

    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations_np.shape[1],
    )
    if translation_log_prior is None:
        fine_translation_prior_2d = None
    else:
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior_np.ndim == 1:
            fine_tp = translation_log_prior_np[fine_translation_parent]
            fine_translation_prior_2d = np.broadcast_to(fine_tp, (n_images, n_fine_trans)).astype(
                np.float32,
                copy=False,
            )
        elif translation_log_prior_np.ndim == 2:
            fine_translation_prior_2d = translation_log_prior_np[:, fine_translation_parent].astype(
                np.float32,
                copy=False,
            )
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions",
            )

    prep_t0 = time.time()
    per_image_inputs_by_class = [
        _prepare_per_image_pass2_inputs(
            significant_sample_indices_by_class[class_index],
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            nside_level=nside_level,
            oversampling_order=oversampling_order,
            n_fine_trans=n_fine_trans,
            fine_translation_parent=fine_translation_parent,
            rotation_log_prior=rotation_log_priors_by_class[class_index],
            random_perturbation=random_perturbation,
            fine_rotations_override=fine_rotations_override,
            fine_rotation_parent_override=fine_rotation_parent_override,
        )
        for class_index in range(n_classes)
    ]
    prep_s = time.time() - prep_t0
    local_rot_counts = [
        int(rots.shape[0])
        for per_image_inputs in per_image_inputs_by_class
        for rots in per_image_inputs["oversampled_rots"]
    ]
    valid_candidate_counts = [
        int(np.asarray(mask).sum())
        for per_image_inputs in per_image_inputs_by_class
        for mask in per_image_inputs["candidate_mask"]
    ]

    max_hypotheses_per_microbatch = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=budget_window_spec.use_window,
        has_external_normalization=False,
        dump_pass2_operands=bool(os.environ.get("RECOVAR_PASS2_DUMP_DIR")),
        fused_k_class=True,
        n_score_pixels=budget_window_spec.n_score,
        device_memory_bytes=device_memory_bytes,
    )
    max_translation_tile_bytes = _max_translation_tile_bytes_for_pass(
        device_memory_bytes,
        fused_k_class=True,
    )
    max_images_per_microbatch = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_translation_tile_bytes,
    )
    bucket_t0 = time.time()
    buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
    )
    bucket_s = time.time() - bucket_t0
    logger.info(
        "Sparse fused K-class pass-2 bucketing: %d images x %d classes -> %d buckets (%s; "
        "max_hypotheses_per_microbatch=%d, max_images_per_microbatch=%d, "
        "max_translation_tile_bytes=%d, n_score_pixels=%d, device_memory_gib=%.2f)",
        n_images,
        n_classes,
        len(buckets),
        _bucket_summary(buckets),
        max_hypotheses_per_microbatch,
        max_images_per_microbatch,
        max_translation_tile_bytes,
        int(budget_window_spec.n_score),
        (-1.0 if device_memory_bytes is None else device_memory_bytes / float(1024**3)),
    )
    logger.info(
        "Sparse fused K-class pass-2 setup timing: hypothesis_prep=%.2fs bucket=%.2fs",
        prep_s,
        bucket_s,
    )

    Ft_y_total = [jnp.zeros(recon_volume_size, dtype=recon_accum_dtype) for _ in range(n_classes)]
    Ft_ctf_total = [jnp.zeros(recon_volume_size, dtype=recon_accum_dtype) for _ in range(n_classes)]
    class_hard_assignments = np.empty((n_classes, n_images), dtype=np.int32)
    best_rotations = [np.empty((n_images, 3, 3), dtype=np.float32) for _ in range(n_classes)]
    best_rotation_indices = [np.empty(n_images, dtype=np.int64) for _ in range(n_classes)]
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_score_log_z = np.empty((n_classes, n_images), dtype=np.float64)
    best_log_score = np.empty((n_classes, n_images), dtype=np.float64)
    max_posterior = np.empty((n_classes, n_images), dtype=np.float32)
    rotation_posterior_sums = np.zeros((n_classes, n_coarse_rot), dtype=np.float64)

    noise_wsum_total = [None] * n_classes
    noise_img_power_total = [None] * n_classes
    noise_sumw_total = np.zeros(n_classes, dtype=np.float64)
    noise_sigma2_offset_total = np.zeros(n_classes, dtype=np.float64)
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        noise_wsum_total = [np.zeros(n_shells, dtype=np.float64) for _ in range(n_classes)]
        noise_img_power_total = [np.zeros(n_shells, dtype=np.float64) for _ in range(n_classes)]

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    window_indices_np = window_spec.score_indices_np
    window_indices = window_spec.score_indices
    recon_window_indices = window_spec.recon_indices
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)
    direct_half_weights = half_weights_windowed if use_window else half_weights

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(shared_noise_variance, image_shape).squeeze()
    if accumulate_noise:
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)

    projection_cache_by_class = [None] * n_classes
    dump_pass2_operands = bool(os.environ.get("RECOVAR_PASS2_DUMP_DIR"))
    if fine_rotations_override is not None and not dump_pass2_operands:
        n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
        transient_projection_bytes = n_fine_rot * n_half * np.dtype(np.complex64).itemsize
        if not use_window:
            transient_projection_bytes += n_fine_rot * n_half * np.dtype(np.float32).itemsize
        max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
        if transient_projection_bytes * n_classes <= max_projection_cache_bytes * n_classes:
            for class_index in range(n_classes):
                cache_t0 = time.time()
                projection_kwargs = window_spec.projection_kwargs(return_abs2=False if use_window else None)
                proj_half_cache_flat, proj_abs2_cache_flat = _compute_projections_block(
                    mean_for_proj_by_class[class_index],
                    jnp.asarray(fine_rotations_override, dtype=jnp.float32),
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    **projection_kwargs,
                )
                if use_window:
                    recon_cache = proj_half_cache_flat[:, recon_window_indices]
                    projection_cache_by_class[class_index] = {
                        "score": proj_half_cache_flat[:, window_indices],
                        "recon": recon_cache,
                        "recon_abs2": jnp.abs(recon_cache) ** 2,
                    }
                    del proj_half_cache_flat
                else:
                    projection_cache_by_class[class_index] = {
                        "score": proj_half_cache_flat,
                        "recon": proj_half_cache_flat,
                        "recon_abs2": proj_abs2_cache_flat,
                    }
                logger.info(
                    "Sparse fused K-class pass-2 projection cache: class %d cached %d fine rotations in %.2fs "
                    "(estimated transient %.2f GiB)",
                    class_index + 1,
                    n_fine_rot,
                    time.time() - cache_t0,
                    transient_projection_bytes / float(1024**3),
                )
        else:
            logger.info(
                "Sparse fused K-class pass-2 projection cache skipped: estimated class transient %.2f GiB "
                "exceeds cap %.2f GiB",
                transient_projection_bytes / float(1024**3),
                max_projection_cache_bytes / float(1024**3),
            )

    bucket_group_stats = _bucket_group_stats(buckets)
    last_bucket_size_logged = None
    group_t0 = None
    overall_t0 = time.time()
    for bucket_meta in buckets:
        image_indices = np.asarray(bucket_meta["image_indices"], dtype=np.int64)
        class_bucket_arrays = [
            _build_bucket_arrays(bucket_meta, per_image_inputs_by_class[class_index], n_fine_trans)
            for class_index in range(n_classes)
        ]
        bucket_size = int(bucket_meta["bucket_size"])
        if bucket_size != last_bucket_size_logged:
            if last_bucket_size_logged is not None and group_t0 is not None:
                prev_chunks, prev_images = bucket_group_stats[last_bucket_size_logged]
                prev_wall = time.time() - group_t0
                logger.info(
                    "Sparse fused K-class pass-2 bucket group done: bucket_size=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
                    last_bucket_size_logged,
                    prev_chunks,
                    prev_images,
                    prev_wall,
                    prev_images / max(prev_wall, 1e-9),
                )
            group_chunks, group_images = bucket_group_stats[bucket_size]
            logger.info(
                "Sparse fused K-class pass-2 bucket group start: bucket_size=%d chunks=%d images=%d",
                bucket_size,
                group_chunks,
                group_images,
            )
            last_bucket_size_logged = bucket_size
            group_t0 = time.time()
        batch = int(image_indices.shape[0])
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
        batch_data = jnp.asarray(batch_data)
        if not np.array_equal(np.asarray(fetched_indices), image_indices):
            fetched_indices_np = np.asarray(fetched_indices)
            reordered = []
            for arrays in class_bucket_arrays:
                (
                    rotations,
                    rotation_indices,
                    log_prior,
                    candidate_mask,
                    parent_map_padded,
                    actual_counts,
                ) = _reorder_to_indices(
                    fetched_indices_np,
                    image_indices,
                    arrays["rotations"],
                    arrays["rotation_indices"],
                    arrays["log_prior"],
                    arrays["candidate_mask"],
                    arrays["parent_map"],
                    arrays["actual_counts"],
                )
                reordered.append(
                    {
                        "image_indices": fetched_indices_np,
                        "bucket_size": arrays["bucket_size"],
                        "actual_counts": actual_counts,
                        "rotations": rotations,
                        "rotation_indices": rotation_indices,
                        "log_prior": log_prior,
                        "candidate_mask": candidate_mask,
                        "parent_map": parent_map_padded,
                    }
                )
            class_bucket_arrays = reordered
            image_indices = fetched_indices_np

        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                image_indices,
                batch_size=batch,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                fine_translations,
                centers,
                experiment_dataset.voxel_size,
            )
        if fine_translation_prior_2d is None:
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=jnp.float32)
        else:
            bucket_translation_prior = jnp.asarray(fine_translation_prior_2d[image_indices], dtype=jnp.float32)

        (
            shifted_score_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
            shifted_score_half_with_dc,
            processed_score_half_for_noise,
            shifted_corrected_score_half,
        ) = _prepare_bucket_io(
            experiment_dataset,
            batch_data,
            ctf_params,
            image_indices,
            noise_variance_half,
            fine_translations,
            config,
            n_fine_trans,
            score_with_masked_images,
            half_spectrum_scoring,
            image_corrections,
            scale_corrections,
            image_pre_shifts,
            use_float64_scoring,
            return_direct_scoring_io=True,
            score_only=False,
            score_mode=relion_firstiter_score_mode,
            window_indices=window_indices,
        )
        if use_window:
            ctf2_over_nv_score = ctf2_over_nv_half[:, window_indices]
            shifted_corrected_score = shifted_corrected_score_half[:, window_indices]
            shifted_score = shifted_score_half[:, window_indices]
            shifted_recon = shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_score_half_with_dc[:, recon_window_indices]
        else:
            ctf2_over_nv_score = ctf2_over_nv_half
            shifted_corrected_score = shifted_corrected_score_half
            shifted_score = shifted_score_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
            shifted_noise = shifted_score_half_with_dc

        shifted_corrected_score_split = shifted_corrected_score.reshape(batch, n_fine_trans, -1)
        scores_by_class = []
        class_score_log_z_bucket = []
        flat_rotations_by_class = []
        proj_for_noise_by_class = []
        proj_abs2_by_class = []
        for class_index, arrays in enumerate(class_bucket_arrays):
            class_bucket_size = int(arrays["bucket_size"])
            flat_rotations = flatten_bucket_rotations(jnp.asarray(arrays["rotations"]))
            flat_rotations_by_class.append(flat_rotations)
            cache = projection_cache_by_class[class_index]
            if cache is not None:
                rotation_indices_jax = jnp.asarray(arrays["rotation_indices"], dtype=jnp.int32)
                proj_half = cache["score"][rotation_indices_jax]
                proj_for_noise = cache["recon"][rotation_indices_jax]
                proj_abs2_for_noise = cache["recon_abs2"][rotation_indices_jax]
            else:
                projection_kwargs = window_spec.projection_kwargs(return_abs2=False if use_window else None)
                proj_half_flat, proj_abs2_half_flat = _compute_projections_block(
                    mean_for_proj_by_class[class_index],
                    flat_rotations,
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    **projection_kwargs,
                )
                if use_window:
                    proj_half = proj_half_flat[:, window_indices].reshape(batch, class_bucket_size, n_windowed)
                    proj_for_noise = proj_half_flat[:, recon_window_indices].reshape(
                        batch,
                        class_bucket_size,
                        n_recon_windowed,
                    )
                    proj_abs2_for_noise = jnp.abs(proj_for_noise) ** 2
                else:
                    proj_half = proj_half_flat.reshape(batch, class_bucket_size, n_half)
                    proj_abs2_for_noise = proj_abs2_half_flat.reshape(batch, class_bucket_size, n_half)
                    proj_for_noise = proj_half
            proj_for_noise, proj_abs2_for_noise = precision_policy.cast_local_noise_projection_scores(
                proj_for_noise,
                proj_abs2_for_noise,
            )
            if relion_firstiter_score_mode == "normalized_cc":
                scores = _score_pass2_bucket_normalized_cc(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(arrays["candidate_mask"]),
                )
            else:
                scores = _score_pass2_bucket_relion_gpu_diff2(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(arrays["log_prior"]),
                    bucket_translation_prior,
                    jnp.asarray(arrays["candidate_mask"]),
                )
            scores_by_class.append(scores)
            class_score_log_z_bucket.append(_logsumexp_pass2_bucket_score_only(scores))
            proj_for_noise_by_class.append(proj_for_noise)
            proj_abs2_by_class.append(proj_abs2_for_noise)

        global_score_log_z_bucket = _logsumexp_class_log_z(jnp.stack(class_score_log_z_bucket, axis=0))
        log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        shifted_recon_split = shifted_recon.reshape(batch, n_fine_trans, -1)
        if accumulate_noise:
            batch_img_power = jnp.sum(jnp.abs(processed_score_half_for_noise) ** 2, axis=0).astype(jnp.float32)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            batch_img_power_shells_np = np.asarray(batch_img_power_shells, dtype=np.float64)
            shifted_noise_split = (
                shifted_noise.reshape(batch, n_fine_trans, -1)
                if half_spectrum_scoring
                else shifted_score.reshape(batch, n_fine_trans, -1)
            )

        for class_index, arrays in enumerate(class_bucket_arrays):
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                _normalize_pass2_bucket_with_log_z(
                    scores_by_class[class_index],
                    global_score_log_z_bucket.astype(scores_by_class[class_index].real.dtype),
                )
            )
            if winner_take_all:
                probs = _winner_take_all_bucket_probs(
                    scores_by_class[class_index],
                    best_argmax,
                    best_log_score_bucket,
                )
                max_posterior_bucket = jnp.where(
                    jnp.isfinite(best_log_score_bucket),
                    jnp.ones_like(max_posterior_bucket),
                    jnp.zeros_like(max_posterior_bucket),
                )
            summed = compute_local_weighted_sums(probs, shifted_recon_split)
            ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)
            if use_window:
                Ft_y_total[class_index] = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(summed),
                    recon_window_indices,
                    flat_rotations_by_class[class_index],
                    Ft_y_total[class_index],
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                    float(current_size // 2),
                )
                Ft_ctf_total[class_index] = _adjoint_slice_volume_windowed(
                    flatten_bucket_rows(ctf_probs),
                    recon_window_indices,
                    flat_rotations_by_class[class_index],
                    Ft_ctf_total[class_index],
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                    float(current_size // 2),
                )
            else:
                Ft_y_total[class_index] = _adjoint_slice_volume_half(
                    flatten_bucket_rows(summed),
                    flat_rotations_by_class[class_index],
                    Ft_y_total[class_index],
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                )
                Ft_ctf_total[class_index] = _adjoint_slice_volume_half(
                    flatten_bucket_rows(ctf_probs),
                    flat_rotations_by_class[class_index],
                    Ft_ctf_total[class_index],
                    image_shape,
                    recon_volume_shape,
                    "linear_interp",
                    True,
                    use_half_volume_mstep,
                )
            if accumulate_noise:
                if translation_sqdist_ang is not None:
                    translation_posterior = np.asarray(jnp.sum(probs, axis=1), dtype=np.float64)
                    noise_sigma2_offset_total[class_index] += float(
                        np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                    )
                noise_img_power_total[class_index] += batch_img_power_shells_np
                noise_sumw_total[class_index] += float(batch)
                summed_masked_noise = compute_local_weighted_sums(probs, shifted_noise_split)
                block_noise_shells, _, _ = _compute_noise_block(
                    flatten_bucket_rows(proj_for_noise_by_class[class_index]),
                    flatten_bucket_rows(proj_abs2_by_class[class_index]),
                    flatten_bucket_rows(summed_masked_noise),
                    flatten_bucket_rows(ctf_probs),
                    noise_variance_for_noise,
                    shell_indices_noise,
                    n_shells,
                )
                noise_wsum_total[class_index] += np.asarray(block_noise_shells, dtype=np.float64)

            actual_counts_arr = np.asarray(arrays["actual_counts"], dtype=np.int64)
            best_argmax_np = np.asarray(best_argmax, dtype=np.int64)
            best_rot_idx = best_argmax_np // n_fine_trans
            best_trans_idx = best_argmax_np % n_fine_trans
            if np.any(best_rot_idx >= actual_counts_arr):
                bad = np.flatnonzero(best_rot_idx >= actual_counts_arr)
                raise RuntimeError(
                    "Fused sparse K-class pass-2: best rotation index points into padding for "
                    f"class {class_index + 1}, images {bad.tolist()}",
                )
            best_log_score_np = np.asarray(best_log_score_bucket, dtype=np.float64)
            max_posterior_np = np.asarray(max_posterior_bucket, dtype=np.float32)
            class_log_z_np = np.asarray(class_score_log_z_bucket[class_index], dtype=np.float64)
            probs_sum_t = np.asarray(jnp.sum(probs, axis=-1), dtype=np.float64)
            for row, image_idx in enumerate(image_indices.tolist()):
                r = int(best_rot_idx[row])
                t = int(best_trans_idx[row])
                fine_rot_idx = int(per_image_inputs_by_class[class_index]["oversampled_rot_indices"][image_idx][r])
                class_hard_assignments[class_index, image_idx] = fine_rot_idx * n_fine_trans + t
                best_rotations[class_index][image_idx] = per_image_inputs_by_class[class_index]["oversampled_rots"][
                    image_idx
                ][r]
                best_rotation_indices[class_index][image_idx] = fine_rot_idx
                if np.isfinite(class_log_z_np[row]):
                    class_log_evidence[class_index, image_idx] = float(class_log_z_np[row] + log_score_offset[row])
                    class_score_log_z[class_index, image_idx] = float(class_log_z_np[row])
                else:
                    class_log_evidence[class_index, image_idx] = -np.inf
                    class_score_log_z[class_index, image_idx] = -np.inf
                best_log_score[class_index, image_idx] = float(best_log_score_np[row] + log_score_offset[row])
                max_posterior[class_index, image_idx] = float(max_posterior_np[row])
                cnt = int(actual_counts_arr[row])
                if cnt == 0:
                    continue
                unique_rot_image = per_image_inputs_by_class[class_index]["unique_rot"][image_idx]
                parent_map_image = per_image_inputs_by_class[class_index]["parent_map"][image_idx]
                coarse_rot_indices = unique_rot_image[parent_map_image]
                np.add.at(
                    rotation_posterior_sums[class_index],
                    coarse_rot_indices,
                    probs_sum_t[row, :cnt],
                )

    if last_bucket_size_logged is not None and group_t0 is not None:
        group_chunks, group_images = bucket_group_stats[last_bucket_size_logged]
        group_wall = time.time() - group_t0
        logger.info(
            "Sparse fused K-class pass-2 bucket group done: bucket_size=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
            last_bucket_size_logged,
            group_chunks,
            group_images,
            group_wall,
            group_images / max(group_wall, 1e-9),
        )

    em_wall = time.time() - overall_t0
    logger.info(
        "Sparse fused K-class pass-2: %d images, %d classes, %d buckets, %.2fs E+M; "
        "median local rot=%d, mean local rot=%.1f, median valid candidates/image=%d",
        n_images,
        n_classes,
        len(buckets),
        em_wall,
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
    )

    Ft_y_out = []
    Ft_ctf_out = []
    for class_index in range(n_classes):
        class_Ft_y = Ft_y_total[class_index]
        class_Ft_ctf = Ft_ctf_total[class_index]
        if use_half_volume_mstep:
            _maybe_dump_native_half_mstep(
                class_Ft_y,
                class_Ft_ctf,
                current_size=current_size,
                n_images=n_images,
                recon_volume_shape=recon_volume_shape,
                stage=f"fused_class{class_index + 1}_pre_x0",
            )
            class_Ft_y, class_Ft_ctf = enforce_half_volume_x0(
                class_Ft_y,
                class_Ft_ctf,
                recon_volume_shape,
                logger=logger,
                label=f"Sparse fused K-class pass-2 class {class_index + 1}",
            )
            _maybe_dump_native_half_mstep(
                class_Ft_y,
                class_Ft_ctf,
                current_size=current_size,
                n_images=n_images,
                recon_volume_shape=recon_volume_shape,
                stage=f"fused_class{class_index + 1}_post_x0",
            )
            class_Ft_y, class_Ft_ctf = half_volume_accumulators_to_full(
                class_Ft_y,
                class_Ft_ctf,
                recon_volume_shape,
            )
        Ft_y_out.append(np.asarray(jax.device_get(class_Ft_y)))
        Ft_ctf_out.append(np.asarray(jax.device_get(class_Ft_ctf)))

    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=class_log_evidence[class_index],
            best_log_score_per_image=best_log_score[class_index],
            max_posterior_per_image=max_posterior[class_index],
            rotation_posterior_sums=rotation_posterior_sums[class_index],
        )
        for class_index in range(n_classes)
    )
    noise_stats = None
    if accumulate_noise:
        noise_stats = tuple(
            make_noise_stats(
                wsum_sigma2_noise=noise_wsum_total[class_index],
                wsum_img_power=noise_img_power_total[class_index],
                wsum_sigma2_offset=float(noise_sigma2_offset_total[class_index]),
                sumw=float(noise_sumw_total[class_index]),
            )
            for class_index in range(n_classes)
        )
    best_translations = tuple(
        fine_translations[class_hard_assignments[class_index] % n_fine_trans]
        for class_index in range(n_classes)
    )
    return SparseKClassPass2FusedResult(
        class_log_evidence=class_log_evidence,
        class_score_log_z=class_score_log_z,
        Ft_y=tuple(Ft_y_out),
        Ft_ctf=tuple(Ft_ctf_out),
        per_class_hard_assignments=class_hard_assignments,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        per_class_best_pose_rotations=tuple(best_rotations),
        per_class_best_pose_translations=best_translations,
        per_class_best_pose_rotation_ids=tuple(best_rotation_indices),
        profile_summary={"sparse_kclass_fused_s": np.float64(em_wall)},
    )
