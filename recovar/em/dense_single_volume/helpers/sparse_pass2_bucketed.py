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
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.normalization_inputs import optional_normalization_vector
from recovar.em.dense_single_volume.helpers.scale_groups import prepare_scale_correction_groups

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers import relion_ctf
from recovar.em.dense_single_volume.helpers import bpref_diagnostics, norm_scale_diagnostics, pass2_diagnostics
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
)
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch, original_image_indices
from recovar.em.dense_single_volume.helpers.compact_candidate_capture import (
    compact_capture_requested_for_original_indices,
    compact_capture_requested_particle_count,
    maybe_capture_k1_production_bucket,
    maybe_capture_k1_production_bucket_chunked,
    require_chunked_capture_capacity,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.env_flags import (
    parse_env_flag,
    parse_env_int_set,
    parse_env_nonnegative_int,
)
from recovar.em.dense_single_volume.helpers.fourier_window import (
    centered_half_indices_to_fftw_half_indices,
    make_fourier_window_indices_np,
    make_fourier_window_spec,
    relion_fftw_order_for_square_score_window,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    bin_shell_values_jax,
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    half_image_phase_factors,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    _find_significant_mask_full_sort,
    _relion_cuda_f32_tail_target,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
    prepare_batch_preprocess_operands,
    process_half_image,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_noise_block as _compute_noise_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_norm_residual_per_image as _compute_norm_residual_per_image,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
)
from recovar.em.dense_single_volume.helpers.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from recovar.em.dense_single_volume.helpers.compact_candidates import (
    SparseCandidateMask,
    _candidate_mask_count,
    _candidate_mask_is_full,
)
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _compact_pair_counts_from_inputs,
    _compact_pair_image_mask_for_threshold,
    _bucket_sparse_k_class_compact_pair_counts,
    _DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION,
    _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE,
    _bucket_pass2_inputs,
    _bucket_sparse_k_class_pass2_inputs,
    _prepare_per_image_compact_candidate_pairs,
    _prepare_per_image_pass2_inputs,
    _build_compact_pair_bucket_arrays,
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
    _build_bucket_arrays,
    _compact_bucket_size_for_class,
    _build_k_class_bucket_arrays,
)
from recovar.em.dense_single_volume.helpers.translation_prior import (
    expand_fine_translation_prior,
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.local_backprojection import (
    relion_x_half_sequential_translation_reduction_enabled,
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger(__name__)

_RELION_WAVG_ATOMIC_SCALE_AA_ENV = "RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA"
_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV = (
    "RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL"
)
_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV = (
    "RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY"
)
_RELION_FINE_ROTATION_EXECUTION_ORDER_ENV = (
    "RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER"
)
_DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH = 1_250_000
_DEFAULT_MAX_TRANSLATION_TILE_BYTES = 384 * 1024**2
# Scale sparse pass-2 bucket sizes from physical GPU memory and active score
# pixels. The fused K-class path is launch-bound at 100k/256 unless it uses
# larger chunks; these fractions still scale down on smaller GPUs.
_AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION = 0.640
_AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION = 0.305
# Compact K-class scoring materializes two complex candidate-by-pixel gathers
# for one class at a time while projections and M-step operands remain live.
# Keep those two gathers within 10% of physical memory.  A K=4 cap that allowed
# 6,587,373 total candidates formed two 8 GiB gathers and requested a 17.04 GiB
# compiled temporary on the 100k/256 fixture after earlier JIT fragmentation.
_AUTO_FUSED_KCLASS_SCORE_GATHER_DEVICE_FRACTION = 0.100
_AUTO_FUSED_KCLASS_LIVE_COMPLEX_GATHERS = 2
_AUTO_TRANSLATION_TILE_DEVICE_FRACTION = 0.020
_AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION = 0.014
_AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION = 0.007
_AUTO_PROJECTION_CACHE_DEVICE_FRACTION = 0.100
_AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION = 0.040
_AUTO_PROJECTION_GATHER_DEVICE_FRACTION = 0.020
_AUTO_NOISE_BLOCK_DEVICE_FRACTION = 0.0125
_AUTO_ADJOINT_BLOCK_DEVICE_FRACTION = 0.006
_DEFAULT_SMALL_BUCKET_COALESCE_SIZE = 128
_DEFAULT_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES = 5_000
_DEFAULT_TAIL_BUCKET_COALESCE_MAX_IMAGES_FUSED_KCLASS = 0
_DEFAULT_PROJECTION_GATHER_MAX_BYTES = 1024 * 1024**2
_DEFAULT_NOISE_BLOCK_MAX_BYTES = 512 * 1024**2
_DEFAULT_ADJOINT_BLOCK_MAX_BYTES = 512 * 1024**2
_EXACT_RAW_DIFF2_CACHE_MAX_BYTES = 512 * 1024**2
_EXACT_RAW_DIFF2_CACHE_DEVICE_FRACTION = 0.01
_EXACT_RAW_DIFF2_CACHE_FREE_FRACTION = 0.25
_EXACT_RAW_DIFF2_CACHE_MAX_BYTES_ENV = "RECOVAR_SPARSE_PASS2_EXACT_RAW_DIFF2_CACHE_MAX_BYTES"
_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES"
_SCORE_ONLY_MAX_HYPOTHESES_ENV = "RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES"
_MAX_TRANSLATION_TILE_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES"
_MAX_PROJECTION_GATHER_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES"
_MAX_NOISE_BLOCK_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES"
_MAX_ADJOINT_BLOCK_BYTES_ENV = "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES"
_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES"
_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES"
_SMALL_BUCKET_THRESHOLD_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD"
_SMALL_BUCKET_COALESCE_SIZE_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_COALESCE_SIZE"
_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES_ENV = "RECOVAR_SPARSE_PASS2_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES"
_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES"
_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION"
_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE"
_MAX_PROJECTED_ROTATIONS_ENV = "RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS"
_PROJECTION_CACHE_MAX_BYTES_ENV = "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES"
_COMPACT_KCLASS_PAIR_STATS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS"
_COMPACT_KCLASS_PAIRS_CHECK_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK"
_SPARSE_KCLASS_COMPACT_PAIRS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS"
_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS"
_SPARSE_KCLASS_COMPACT_BUCKETS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS"
_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH"
_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE"
_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES"
)
_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION"
)
_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE"
)
_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV = "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS"
_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT"
)
_SPARSE_KCLASS_GROUP_TIMING_ENV = "RECOVAR_SPARSE_KCLASS_GROUP_TIMING"
_SPARSE_KCLASS_EXECUTION_SIGNATURES_ENV = (
    "RECOVAR_SPARSE_KCLASS_EXECUTION_SIGNATURES"
)
_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV = (
    "RECOVAR_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE"
)
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_ENV = "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS"
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE_ENV = (
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE"
)
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_ENV = "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL"
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO_ENV = (
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO"
)
_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV = "RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE"
_SPARSE_KCLASS_FUSED_NOISE_NORM_ENV = "RECOVAR_SPARSE_KCLASS_FUSED_NOISE_NORM"
_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV = "RECOVAR_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED"
_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS_ENV = "RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS"
_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV = (
    "RECOVAR_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS"
)
_SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV = "RECOVAR_SPARSE_KCLASS_FUSED_MSTEP_NOISE"
_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP"
_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV = "RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE"
_RELION_X_HALF_F32_FINE_POSTERIOR_ENV = "RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR"
_RELION_FINE_DIFF2_FUSED_FFI_ENV = "RECOVAR_RELION_FINE_DIFF2_FUSED_FFI"
_RELION_X_HALF_BP_PARTICLE_POOL_SIZE_ENV = (
    "RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE"
)
_RELION_POWERCLASS_SPECTRUM_NORM_ENV = "RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM"
_RELION_EXACT_BPREF_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS"
_RELION_TRANSLATED_WAVG_NORM_ENV = "RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM"
_RELION_WAVG_SEQUENTIAL_CUDA_ENV = "RECOVAR_K1_RELION_WAVG_SEQUENTIAL_CUDA"
_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE"
_BPREF_REVERSE_PHYSICAL_ORDER_ENV = "RECOVAR_K1_BPREF_REVERSE_PHYSICAL_ORDER"
_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE"
_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV = (
    "RECOVAR_K1_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT"
)
_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV = (
    "RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE"
)
_DEFAULT_FRESH_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE = 220
_PASS2_DUMP_CONSERVATIVE_EXECUTION_ENV = "RECOVAR_PASS2_DUMP_CONSERVATIVE_EXECUTION"
_PASS2_DUMP_STOP_AFTER_TARGET_ENV = "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET"
_NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET"
)
_NORM_RESIDUAL_DUMP_ONLY_ENV = "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_ONLY"
_SPARSE_PASS2_PROJECTION_CACHE_ENV = "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE"
_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_JOINT_MODES = {"joint", "global", "class_pose", "class-pose"}
_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK_ENV = "RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK"
_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS_ENV = "RECOVAR_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS"
_SPARSE_PASS2_GROUP_PROGRESS_SECONDS_ENV = "RECOVAR_SPARSE_PASS2_GROUP_PROGRESS_SECONDS"
_SPARSE_PASS2_WINDOWED_PREPARE_ENV = "RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE"
_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP_ENV = (
    "RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP"
)
_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES_ENV = (
    "RECOVAR_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES"
)
_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER_ENV = (
    "RECOVAR_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER"
)
_DEFAULT_PROJECTION_CACHE_MAX_BYTES = 3 * 1024**3
_DEFAULT_COMPACT_PAIR_THRESHOLD_REPORT = (8192, 16384, 32768, 65536, 131072)
_DEFAULT_COMPACT_PAIR_MIN_BUCKET_SIZE = 512
_DEFAULT_COMPACT_PAIR_TAIL_BUCKET_COALESCE_MAX_IMAGES = 19
_DEFAULT_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE = 4096
_DEFAULT_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO = 0.05
_DEFAULT_ACTIVE_ROW_PAD_MULTIPLE = 1024
_DEFAULT_CACHED_SCORE_ROT_CHUNK_SIZE = 8192
_DEFAULT_PASS2_GROUP_PROGRESS_CHUNKS = 1000
_DEFAULT_PASS2_GROUP_PROGRESS_SECONDS = 300
_DEFAULT_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER = 4
_DEFAULT_KCLASS_RAW_HOST_STAGING_MAX_BYTES = 8 * 1024**3


_noise_block_chunk_log_keys: set[tuple[int, int, int, int]] = set()
_active_noise_gather_chunk_log_keys: set[tuple[int, int, int, int]] = set()
_active_flat_gather_chunk_log_keys: set[tuple[str, int, int, int, int]] = set()
_adjoint_block_chunk_log_keys: set[tuple[str, int, int, int, int]] = set()
_cached_score_chunk_log_keys: set[tuple[str, int, int, int]] = set()
_relion_wavg_direct_noise_log_keys: set[int] = set()


class RelionWavgRectangle(NamedTuple):
    """Static mapping for RELION's full cropped Wavg CUDA pixel stream."""

    centered_indices: np.ndarray
    exact_positions: np.ndarray
    shell_indices: np.ndarray


class Pass2DumpComplete(RuntimeError):
    """Raised by explicit diagnostic runs after requested pass-2 dump files are written."""

    def __init__(self, *, dump_count: int, current_size: int | None):
        self.dump_count = int(dump_count)
        self.current_size = None if current_size is None else int(current_size)
        super().__init__(
            "requested RECOVAR pass-2 dump target set was written "
            f"(dump_count={self.dump_count}, current_size={self.current_size})"
        )


def _k_class_pass2_dump_progress(
    *,
    dump_dir: str | Path,
    target_original_indices,
    target_classes_one_based,
    current_size: int | None,
) -> tuple[int, int]:
    """Return written and expected file counts for a K-class dump target set."""

    target_indices = {int(value) for value in target_original_indices}
    target_classes = {int(value) for value in target_classes_one_based}
    if not target_indices:
        raise ValueError("K-class pass-2 dump completion requires at least one target particle")
    if not target_classes or min(target_classes) < 1:
        raise ValueError("K-class pass-2 dump completion requires positive one-based classes")
    size_label = -1 if current_size is None else int(current_size)
    root = Path(dump_dir)
    expected_paths = [
        root / f"pass2_orig{original_index:06d}_class{class_one_based:03d}_cs{size_label:03d}.npz"
        for original_index in sorted(target_indices)
        for class_one_based in sorted(target_classes)
    ]
    return sum(path.is_file() for path in expected_paths), len(expected_paths)


def _k1_pass2_dump_progress(
    *,
    dump_dir: str | Path,
    target_original_indices,
    current_size: int | None,
) -> tuple[int, int]:
    """Return written and expected file counts for a K=1 dump target set."""

    target_indices = {int(value) for value in target_original_indices}
    if not target_indices:
        raise ValueError("K=1 pass-2 dump completion requires at least one target particle")
    size_label = -1 if current_size is None else int(current_size)
    root = Path(dump_dir)
    expected_paths = [
        root / f"pass2_orig{original_index:06d}_cs{size_label:03d}.npz"
        for original_index in sorted(target_indices)
    ]
    return sum(path.is_file() for path in expected_paths), len(expected_paths)


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
    class_posterior_sums: np.ndarray | None = None

    per_class_best_pose_eulers_deg: tuple[np.ndarray, ...] | None = None


class SparseKClassCompactPairPlanStats(NamedTuple):
    """Host-side accounting for the compact K-class pass-2 planner."""

    buckets: tuple[dict, ...]
    valid_pair_candidates: int
    padded_pair_candidates: int
    rectangular_candidates: int
    reduction_factor: float
    padded_reduction_factor: float
    median_valid_pairs_per_image: int
    mean_valid_pairs_per_image: float
    max_valid_pairs_per_image: int
    max_images_per_microbatch: int


def _candidate_mask_prefers_rectangular_execution(candidate_mask) -> bool:
    if isinstance(candidate_mask, SparseCandidateMask):
        return candidate_mask.mode in {"full", "coarse_exclude"}
    return _candidate_mask_is_full(candidate_mask)


def _compact_pair_execution_mask_excluding_full_support(per_image_inputs_by_class, image_mask):
    """Filter compact-pair execution away from masks with no sparse reduction.

    Compact pairs are a memory win only when they represent a strict subset of
    the rectangular rotation x translation tile.  A full-support mask would
    materialize the same candidate set in larger host-side arrays, which is
    pathological for the first global RELION-style iteration.
    """

    if not per_image_inputs_by_class:
        return image_mask, 0
    n_images = len(per_image_inputs_by_class[0]["candidate_mask"])
    for per_image_inputs in per_image_inputs_by_class[1:]:
        if len(per_image_inputs["candidate_mask"]) != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
    if image_mask is None:
        filtered = np.ones(n_images, dtype=bool)
        had_input_mask = False
    else:
        filtered = np.asarray(image_mask, dtype=bool).copy()
        if filtered.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {filtered.shape} vs {(n_images,)}")
        had_input_mask = True

    excluded = 0
    for image_idx in np.flatnonzero(filtered):
        if any(
            _candidate_mask_prefers_rectangular_execution(per_image_inputs["candidate_mask"][int(image_idx)])
            for per_image_inputs in per_image_inputs_by_class
        ):
            filtered[int(image_idx)] = False
            excluded += 1

    if excluded == 0 and not had_input_mask:
        return None, 0
    return filtered, excluded


# ---------------------------------------------------------------------------
# Per-image hypothesis preparation
# ---------------------------------------------------------------------------


def _half_translation_phase_table_for_indices(translations, image_shape, pixel_indices):
    lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
        image_shape,
        voxel_size=1,
        scaled=True,
    )
    lattice_half = jnp.asarray(lattice_half)
    lattice_window = lattice_half[jnp.asarray(pixel_indices, dtype=jnp.int32)]
    phase_arg = jnp.einsum(
        "td,pd->tp",
        jnp.asarray(translations, dtype=jnp.float32),
        lattice_window,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.exp(-2j * jnp.pi * phase_arg)


def _translation_phase_table_for_indices(
    translations,
    image_shape,
    pixel_indices,
    translation_phases_half,
):
    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    if translation_phases_half is None:
        return _half_translation_phase_table_for_indices(translations, image_shape, pixel_indices)
    return translation_phases_half[:, pixel_indices]


def _relion_translation_angles_f32(translations, image_shape):
    """Return RELION fine-score ``(tx, ty)`` radians with host rounding."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError(
            "RELION score translations must have shape (T, 2), got "
            f"{translations_f64.shape}"
        )
    return np.asarray(
        -2.0 * np.pi * translations_f64 / float(image_size),
        dtype=np.float32,
    )


def _relion_translation_angles_f64(translations, image_shape):
    """Return RELION double-ACC ``(tx, ty)`` translation radians."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError(
            "RELION score translations must have shape (T, 2), got "
            f"{translations_f64.shape}"
        )
    return -2.0 * np.pi * translations_f64 / float(image_size)


def _relion_cuda_score_translation_angles_if_available(
    translations,
    image_shape,
    *,
    enabled,
    dtype=np.float32,
):
    """Prepare exact score-translation angles or retain the JAX fallback."""

    if not enabled or jax.default_backend() != "gpu":
        return None
    from recovar import cuda_backproject

    if not cuda_backproject.cuda_available():
        logger.warning(
            "Exact RELION fine Gaussian scoring is retaining JAX translation "
            "phase arithmetic because custom CUDA is unavailable"
        )
        return None
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"RELION CUDA translation dtype must be float32 or float64, got {dtype}")
    logger.info(
        "Exact RELION fine Gaussian scoring: using CUDA %s score/M-step translation",
        "sincosf" if dtype == np.dtype(np.float32) else "sincos",
    )
    return jnp.asarray(
        np.asarray(
            -2.0 * np.pi * np.asarray(translations, dtype=np.float64) / float(image_shape[0]),
            dtype=dtype,
        ),
        dtype=dtype,
    )


def _compact_k_class_pair_plan_stats(
    per_image_inputs_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    image_mask=None,
) -> SparseKClassCompactPairPlanStats:
    """Compute compact-pair work counters without changing pass-2 execution."""

    compact_inputs_by_class = tuple(
        _prepare_per_image_compact_candidate_pairs(per_image_inputs)
        for per_image_inputs in per_image_inputs_by_class
    )
    return _compact_k_class_pair_plan_stats_from_counts(
        _compact_pair_counts_from_inputs(compact_inputs_by_class),
        dense_buckets,
        n_fine_trans,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        image_mask=image_mask,
    )


def _compact_k_class_pair_plan_stats_from_counts(
    pair_counts_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    image_mask=None,
) -> SparseKClassCompactPairPlanStats:
    """Compute compact-pair work counters from per-image valid-pair counts."""

    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        n_images = int(np.asarray(pair_counts_by_class[0]).shape[0]) if pair_counts_by_class else 0
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")

    compact_buckets = _bucket_sparse_k_class_compact_pair_counts(
        pair_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        image_mask=image_mask,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
    )

    n_classes = len(pair_counts_by_class)
    valid_count_arrays = []
    for pair_counts in pair_counts_by_class:
        pair_counts = np.asarray(pair_counts, dtype=np.int64)
        if image_mask is not None:
            pair_counts = pair_counts[image_mask]
        valid_count_arrays.append(pair_counts)
    valid_counts = np.concatenate(valid_count_arrays) if valid_count_arrays else np.zeros(0, dtype=np.int64)
    valid_pair_candidates = int(valid_counts.sum(dtype=np.int64))
    padded_pair_candidates = int(
        sum(
            n_classes * len(bucket["image_indices"]) * int(bucket["pair_bucket_size"])
            for bucket in compact_buckets
        )
    )
    rectangular_candidates = int(
        sum(
            n_classes
            * (
                int(np.count_nonzero(image_mask[np.asarray(bucket["image_indices"], dtype=np.int64)]))
                if image_mask is not None
                else len(bucket["image_indices"])
            )
            * int(bucket["bucket_size"])
            * int(n_fine_trans)
            for bucket in dense_buckets
        )
    )
    reduction_factor = (
        float(rectangular_candidates) / float(valid_pair_candidates)
        if valid_pair_candidates > 0
        else float("inf")
    )
    padded_reduction_factor = (
        float(rectangular_candidates) / float(padded_pair_candidates)
        if padded_pair_candidates > 0
        else float("inf")
    )

    return SparseKClassCompactPairPlanStats(
        buckets=tuple(compact_buckets),
        valid_pair_candidates=valid_pair_candidates,
        padded_pair_candidates=padded_pair_candidates,
        rectangular_candidates=rectangular_candidates,
        reduction_factor=reduction_factor,
        padded_reduction_factor=padded_reduction_factor,
        median_valid_pairs_per_image=int(np.median(valid_counts)) if valid_counts.size else 0,
        mean_valid_pairs_per_image=float(np.mean(valid_counts)) if valid_counts.size else 0.0,
        max_valid_pairs_per_image=int(valid_counts.max(initial=0)) if valid_counts.size else 0,
        max_images_per_microbatch=max(1, int(max_images_per_microbatch)),
    )


def _maybe_prepare_sparse_k_class_compact_pair_plan(
    per_image_inputs_by_class,
    dense_buckets,
    n_fine_trans,
    *,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
) -> SparseKClassCompactPairPlanStats | None:
    """Return compact-pair planner stats only when explicitly enabled.

    The planner is diagnostic only: dense rectangular scoring/M-step remains
    authoritative.
    """

    stats_flag = os.environ.get(_COMPACT_KCLASS_PAIR_STATS_ENV)
    if stats_flag is None or stats_flag.strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    compact_pair_max_images_per_microbatch = _compact_pair_max_images_per_microbatch_for_pass(
        max_images_per_microbatch,
    )
    return _compact_k_class_pair_plan_stats(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=compact_pair_max_images_per_microbatch,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        image_mask=None,
    )


# ---------------------------------------------------------------------------
# Bucket spec
# ---------------------------------------------------------------------------


def _load_bpref_execution_order_local_override(n_images: int) -> np.ndarray | None:
    """Load a fail-closed diagnostic K=1 particle execution permutation."""

    raw_path = os.environ.get(_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must name an existing absolute file",
        )
    order = np.asarray(np.loadtxt(path, dtype=np.int64, ndmin=1), dtype=np.int64).reshape(-1)
    if order.shape != (int(n_images),):
        raise ValueError(
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must contain {int(n_images)} rows, "
            f"got {order.shape[0]}",
        )
    if not np.array_equal(np.sort(order), np.arange(int(n_images), dtype=np.int64)):
        raise ValueError(f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must contain a permutation")
    return order


def _resolve_bpref_processing_order(
    n_images: int,
    *,
    preserve_bpref_particle_order: bool,
) -> np.ndarray | None:
    """Resolve production or diagnostic K=1 BPref execution ordering."""

    diagnostic_order = _load_bpref_execution_order_local_override(n_images)
    reverse_physical_order = parse_env_flag(
        _BPREF_REVERSE_PHYSICAL_ORDER_ENV,
        default=False,
    )
    if reverse_physical_order and not preserve_bpref_particle_order:
        raise ValueError(
            f"{_BPREF_REVERSE_PHYSICAL_ORDER_ENV}=1 requires the guarded fresh "
            "K=1 physical-order path"
        )
    if reverse_physical_order and diagnostic_order is not None:
        raise ValueError(
            f"{_BPREF_REVERSE_PHYSICAL_ORDER_ENV}=1 cannot be combined with "
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV}"
        )
    if preserve_bpref_particle_order and diagnostic_order is not None:
        raise ValueError(
            "preserve_bpref_particle_order cannot be combined with "
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV}"
        )
    if preserve_bpref_particle_order:
        order = np.arange(int(n_images), dtype=np.int64)
        return order[::-1].copy() if reverse_physical_order else order
    return diagnostic_order


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


def _resolve_bpref_execution_bucket_policy(
    *,
    preserve_bpref_particle_order: bool,
    processing_order_group_by_bucket_size: bool,
) -> tuple[int, bool]:
    """Resolve strict-order batching for the fresh-K=1 physical sequence.

    The production default pads consecutive mixed-support particles in bounded
    chunks.  An explicit chunk size overrides that bound.  The older adjacent
    equal-support batching remains available only as an explicit diagnostic.
    Every mode in this helper retains the exact particle sequence.
    """

    explicit_chunk_size = _optional_positive_int_env(
        _BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV,
    )
    batch_consecutive_requested = parse_env_flag(
        _BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV,
        default=False,
    )
    if batch_consecutive_requested and explicit_chunk_size is not None:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} cannot be "
            f"combined with {_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV}"
        )
    if batch_consecutive_requested and not preserve_bpref_particle_order:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} requires "
            "the guarded fresh K=1 physical particle order"
        )
    if batch_consecutive_requested and processing_order_group_by_bucket_size:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} cannot be "
            f"combined with {_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV}"
        )
    batch_consecutive_bucket_sizes = bool(
        batch_consecutive_requested
        and preserve_bpref_particle_order
        and not processing_order_group_by_bucket_size
    )
    processing_order_chunk_size = explicit_chunk_size or (
        _DEFAULT_FRESH_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE
        if preserve_bpref_particle_order
        and not processing_order_group_by_bucket_size
        and not batch_consecutive_bucket_sizes
        else 1
    )
    return processing_order_chunk_size, batch_consecutive_bucket_sizes


def _optional_positive_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive float, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive float, got {raw!r}")
    return value


def _native_dual_weighted_sums_enabled_for_pass(
    *,
    use_exact_relion_gaussian: bool,
    use_relion_x_half_mstep: bool,
    accumulate_noise: bool,
) -> bool:
    """Select the qualified native reduction only on its exact GPU contract."""

    if os.environ.get(_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV) is not None:
        return parse_env_flag(
            _SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV,
            default=False,
        )
    if not (use_exact_relion_gaussian and use_relion_x_half_mstep and accumulate_noise):
        return False
    from recovar.cuda_backproject import custom_cuda_requested

    return bool(jax.default_backend() == "gpu" and custom_cuda_requested())


def _fused_mstep_noise_enabled_for_pass(
    *,
    native_dual_weighted_sums: bool,
    use_exact_relion_gaussian: bool,
    use_relion_x_half_mstep: bool,
    accumulate_noise: bool,
    compact_noise_sums_match_mstep: bool,
) -> bool:
    """Select the fused reduction only on its qualified exact-GPU contract."""

    if not (
        native_dual_weighted_sums
        and use_exact_relion_gaussian
        and use_relion_x_half_mstep
        and accumulate_noise
        and not compact_noise_sums_match_mstep
        and parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
    ):
        return False
    return parse_env_flag(_SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV, default=True)


def _fresh_k1_direct_noise_default(
    *,
    preserve_bpref_particle_order: bool,
    relion_exact_bpref_operands: bool,
) -> bool:
    """Enable the accepted noise path only inside the fresh K=1 guard."""

    return bool(preserve_bpref_particle_order and relion_exact_bpref_operands)


def _relion_powerclass_spectrum_norm_enabled(
    *,
    fresh_k1_guard: bool,
) -> bool:
    """Use RELION's shell spectrum by default only in the fresh K=1 guard."""

    return parse_env_flag(
        _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
        default=bool(fresh_k1_guard),
    )


def _relion_exact_bpref_operands_enabled(
    *,
    fresh_k1_guard: bool,
    source_faithful_spectrum_norm: bool,
) -> bool:
    """Pair exact BPref with the qualified fresh-K=1 spectrum path."""

    return parse_env_flag(
        _RELION_EXACT_BPREF_OPERANDS_ENV,
        default=bool(fresh_k1_guard and source_faithful_spectrum_norm),
    )


def _relion_wavg_direct_modes(
    *,
    accumulate_noise: bool,
    scale_groups_available: bool,
    scale_aa_enabled: bool,
    direct_noise_only_default: bool = False,
) -> tuple[bool, bool]:
    """Resolve the stopped direct-Wavg noise/norm factorial arms.

    ``DIRECT_RESIDUAL`` preserves the existing coupled treatment: the native
    Wavg ``diff2`` stream supplies both shell noise and per-particle norm.
    ``DIRECT_NOISE_ONLY`` supplies only shell noise, leaving normalization on
    the production algebraic path.  The latter isolates the already-localized
    radial-noise boundary without silently changing a second state variable.
    """

    direct_residual_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV,
            default=False,
        )
    )
    direct_noise_only_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV,
            default=direct_noise_only_default,
        )
    )
    if direct_residual_requested and direct_noise_only_requested:
        raise ValueError(
            f"{_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV}=1 and "
            f"{_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV}=1 are mutually exclusive"
        )
    direct_noise = direct_residual_requested or direct_noise_only_requested
    # Fresh iteration 1 intentionally has no scale-group accumulator.  The
    # established coupled diagnostic is dormant there and activates once the
    # scale state exists; preserve that lifecycle for the isolated arm.
    if direct_noise and not scale_groups_available:
        return False, False
    if direct_noise and not scale_aa_enabled:
        requested_name = (
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV
            if direct_residual_requested
            else _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV
        )
        raise ValueError(
            f"{requested_name}=1 requires "
            f"{_RELION_WAVG_ATOMIC_SCALE_AA_ENV}=1 and scale groups"
        )
    return direct_noise, direct_residual_requested
_PASS2_TOP2_DEBUG_INDICES_ENV = "RECOVAR_PASS2_TOP2_DEBUG_INDICES"


def _pass2_top2_debug_target_indices() -> tuple[int, ...]:
    """Diagnostic only: original (combined, pre-half-split) dataset image
    indices to log the fine (pass-2) top-2 candidate score margin for,
    mirroring ``k_class._pass1_top2_debug_target_indices`` but for the
    oversampled fine-grid decision within pass-1's surviving coarse
    cell(s), where the per-particle candidate set actually differs
    (children of that particle's own coarse winner). Resolved to this
    call's local (within-half) index space via
    ``_resolve_local_target_indices`` before use -- a half-1 and a half-2
    particle can share the same local position, so matching on the raw
    env value directly would silently also hit an unrelated particle in
    the other half.
    """

    raw = os.environ.get(_PASS2_TOP2_DEBUG_INDICES_ENV, "").strip()
    if not raw:
        return ()
    return tuple(int(token) for token in raw.split(",") if token.strip())


def _resolve_local_target_indices(experiment_dataset, original_targets: tuple[int, ...]) -> tuple[int, ...]:
    """Map original (combined dataset) indices to this half's local indices.

    Only returns the subset of ``original_targets`` actually present in
    ``experiment_dataset`` (e.g. the half this call is scoring). Required
    because pass-1/pass-2 debug/override target indices are specified in
    original-dataset space but ``image_indices`` inside the per-half
    scoring functions is local (within-half) space, and two different
    halves' particles can land on the same local position.
    """

    if not original_targets:
        return ()
    resolver = getattr(experiment_dataset, "local_image_indices_from_original", None)
    if not callable(resolver):
        raise RuntimeError(
            "pass1/pass2 top-2 debug/override requires "
            "experiment_dataset.local_image_indices_from_original()"
        )
    local = np.asarray(
        resolver(np.asarray(original_targets, dtype=np.int64), allow_missing=True)
    )
    return tuple(int(v) for v in local if v >= 0)


def _log_pass2_top2_debug(scores, image_indices, targets: tuple[int, ...], *, dataset_tag=None) -> None:
    image_indices_np = np.asarray(image_indices, dtype=np.int64).reshape(-1)
    for target in targets:
        rows = np.flatnonzero(image_indices_np == target)
        if rows.size == 0:
            continue
        row = int(rows[0])
        flat = np.asarray(scores[row], dtype=np.float64).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size < 1:
            logger.warning("PASS2_TOP2_DEBUG dataset=%s image_idx=%d: no finite fine candidates", dataset_tag, target)
            continue
        order = np.argsort(finite)
        best = float(finite[order[-1]])
        second = float(finite[order[-2]]) if finite.size >= 2 else float("-inf")
        n_row_trans = int(np.asarray(scores).shape[-1])
        best_flat_id = int(np.flatnonzero(flat == best)[0])
        second_candidates = np.flatnonzero(flat == second) if finite.size >= 2 else np.array([], dtype=np.int64)
        second_flat_id = int(second_candidates[0]) if second_candidates.size else -1
        logger.warning(
            "PASS2_TOP2_DEBUG dataset=%s image_idx=%d n_candidates=%d best_score=%.8f second_score=%.8f "
            "margin=%.8g best_flat_id=%d(rot=%d,trans=%d) second_flat_id=%d(rot=%d,trans=%d)",
            dataset_tag,
            target,
            finite.size,
            best,
            second,
            best - second,
            best_flat_id,
            best_flat_id // n_row_trans,
            best_flat_id % n_row_trans,
            second_flat_id,
            second_flat_id // n_row_trans if second_flat_id >= 0 else -1,
            second_flat_id % n_row_trans if second_flat_id >= 0 else -1,
        )


def _pass2_dump_enabled() -> bool:
    return bool(os.environ.get(pass2_diagnostics._PASS2_DUMP_DIR_ENV)) and not parse_env_flag(
        _NORM_RESIDUAL_DUMP_ONLY_ENV,
        default=False,
    )


def _pass2_conservative_dump_execution_enabled() -> bool:
    """Keep dump-only planner changes behind an explicit diagnostic opt-in."""

    return _pass2_dump_enabled() and parse_env_flag(
        _PASS2_DUMP_CONSERVATIVE_EXECUTION_ENV,
        default=False,
    )


def _projection_cache_enabled_for_pass(
    *,
    fine_rotations_override,
    dump_pass2_operands: bool,
) -> bool:
    """Resolve the diagnostic projection-cache override without changing defaults."""

    if fine_rotations_override is None:
        return False
    raw = os.environ.get(_SPARSE_PASS2_PROJECTION_CACHE_ENV)
    mode = "auto" if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode == "auto":
        # Preserve the currently qualified paths while cache-on/cache-off is
        # adjudicated: production uses the cache and operand dumps do not.
        return not bool(dump_pass2_operands)
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_SPARSE_PASS2_PROJECTION_CACHE_ENV} must be 'auto', 'on', or 'off', got {raw!r}",
    )


def _cached_score_rotation_chunk_size_for_pass(bucket_size: int) -> int:
    override = _optional_positive_int_env(_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK_ENV)
    chunk_size = _DEFAULT_CACHED_SCORE_ROT_CHUNK_SIZE if override is None else int(override)
    return max(1, min(int(bucket_size), int(chunk_size)))


def _compact_pair_mstep_mode_for_pass() -> str:
    """Return the compact-pair M-step reduction mode for this process."""

    raw = os.environ.get(_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV)
    if raw is None or raw.strip() == "":
        return "dense"
    mode = raw.strip().lower()
    if mode in {"dense", "default"}:
        return "dense"
    if mode == "pair_sparse":
        return mode
    raise ValueError(
        f"{_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV} must be 'dense' or 'pair_sparse', got {raw!r}",
    )


def _compact_pair_pair_sparse_mstep_enabled_for_pass(*, allow_pair_sparse: bool = True) -> bool:
    return bool(allow_pair_sparse) and _compact_pair_mstep_mode_for_pass() == "pair_sparse"


def _windowed_prepare_enabled_for_pass(use_window: bool) -> bool:
    """Return whether sparse pass-2 should materialize only active Fourier windows."""

    return bool(
        use_window
        and parse_env_flag(
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            default=True,
        )
    )


class _SparsePass2WindowSetup(NamedTuple):
    """Forward-model configuration and Fourier windows of one sparse pass-2 scorer."""

    config: object
    window_spec: object
    use_window: bool
    window_indices_np: object
    window_indices: object
    recon_window_indices: object
    relion_x_half_recon_indices: object
    windowed_prepare: bool
    n_windowed: int
    n_recon_windowed: int


def _sparse_pass2_window_setup(
    experiment_dataset,
    *,
    disc_type,
    image_shape,
    current_size,
    n_half,
    mstep_current_size,
    square_window,
    window_spec_kwargs,
    use_relion_x_half_mstep,
    log_label,
) -> _SparsePass2WindowSetup:
    """Build the forward model and score/reconstruction windows of a sparse pass 2.

    The RELION x-half M-step addresses its reconstruction window in FFTW half
    order, so the centred reconstruction indices are converted when that
    layout is active. Windowed prepare is logged once per pass with the
    caller's label. The single-class and fused K-class sparse scorers share
    this setup.
    """

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    recon_window_indices = window_spec.recon_indices
    relion_x_half_recon_indices = None
    if use_relion_x_half_mstep:
        centered_recon_indices = (
            recon_window_indices
            if recon_window_indices is not None
            else jnp.arange(int(n_half), dtype=jnp.int32)
        )
        relion_x_half_recon_indices = centered_half_indices_to_fftw_half_indices(
            image_shape,
            centered_recon_indices,
        )
    windowed_prepare = _windowed_prepare_enabled_for_pass(use_window)
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon
    if windowed_prepare:
        logger.info(
            "%s windowed prepare enabled; set %s=0 to disable "
            "(score_pixels=%d recon_pixels=%d full_half_pixels=%d)",
            log_label,
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            int(n_windowed),
            int(n_recon_windowed),
            int(n_half),
        )
    return _SparsePass2WindowSetup(
        config,
        window_spec,
        use_window,
        window_spec.score_indices_np,
        window_spec.score_indices,
        recon_window_indices,
        relion_x_half_recon_indices,
        windowed_prepare,
        n_windowed,
        n_recon_windowed,
    )


def _windowed_translation_tile_cap_enabled_for_pass() -> bool:
    """Return whether K-class sparse pass-2 should budget translation tiles on active windows."""

    return parse_env_flag(
        _SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP_ENV,
        default=True,
    )


def _translation_tile_half_pixels_for_budget(
    *,
    use_window: bool,
    n_score_pixels: int,
    n_recon_pixels: int,
) -> int | None:
    """Return active half-pixel count for translation-tile budgeting."""

    if not _windowed_prepare_enabled_for_pass(bool(use_window)):
        return None
    if not _windowed_translation_tile_cap_enabled_for_pass():
        return None
    return max(int(n_score_pixels), int(n_recon_pixels))


def _windowed_translation_tile_max_multiplier_for_pass() -> int:
    explicit = _optional_positive_int_env(_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER_ENV)
    if explicit is not None:
        return int(explicit)
    return int(_DEFAULT_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER)


def _max_images_for_sparse_pass2_translation_tile(
    image_shape,
    n_fine_trans,
    *,
    max_tile_bytes: int,
    complex_dtype,
    translation_tile_half_pixels: int | None,
) -> tuple[int, int, int | None, int | None]:
    full_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=complex_dtype,
    )
    if translation_tile_half_pixels is None:
        return full_cap, full_cap, None, None
    window_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=complex_dtype,
        n_half_pixels=translation_tile_half_pixels,
    )
    multiplier = _windowed_translation_tile_max_multiplier_for_pass()
    bounded_window_cap = max(full_cap, int(full_cap) * int(multiplier))
    return min(window_cap, bounded_window_cap), full_cap, window_cap, multiplier


def _compact_pair_execution_enabled_for_pass() -> bool:
    """Return whether fused K-class pass-2 should use compact-pair execution."""

    compact_pair_check = parse_env_flag(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False)
    return parse_env_flag(
        _SPARSE_KCLASS_COMPACT_PAIRS_ENV,
        default=not compact_pair_check,
    )


def _compact_pair_min_bucket_size_for_pass(default_value: int | None = None) -> int:
    """Return the hybrid threshold for compact-pair execution buckets.

    An explicit environment setting wins over a caller-specific default so
    benchmark and diagnostic jobs retain their existing override behavior.
    """

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV)
    if explicit is not None:
        return int(explicit)
    if default_value is not None:
        if int(default_value) <= 0:
            raise ValueError("compact-pair minimum bucket size must be positive")
        return int(default_value)
    return _DEFAULT_COMPACT_PAIR_MIN_BUCKET_SIZE


def _compact_pair_max_images_per_microbatch_for_pass(default_max_images_per_microbatch: int) -> int:
    """Return the compact-pair chunk cap, guarded by an explicit env override."""

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_ENV)
    if explicit is not None:
        return explicit
    return max(1, int(default_max_images_per_microbatch))


def _compact_pair_prepare_max_images_per_microbatch(
    *,
    dense_max_images_per_microbatch: int,
    compact_pair_max_images_per_microbatch: int,
) -> int:
    """Return the compact-pair prepare cap used for execution bucket splitting."""

    return max(
        1,
        min(
            int(dense_max_images_per_microbatch),
            int(compact_pair_max_images_per_microbatch),
        ),
    )


def _active_row_pad_multiple_for_pass() -> int:
    """Return active-row gather padding multiple for stable JIT shapes."""

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV)
    if explicit is not None:
        return int(explicit)
    return _DEFAULT_ACTIVE_ROW_PAD_MULTIPLE


def _small_bucket_coalesce_size_for_pass(n_images: int) -> int | None:
    explicit = _optional_positive_int_env(_SMALL_BUCKET_COALESCE_SIZE_ENV)
    if explicit is not None:
        return explicit
    max_images = parse_env_nonnegative_int(_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = _DEFAULT_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES
    if int(n_images) > int(max_images):
        return None
    return _DEFAULT_SMALL_BUCKET_COALESCE_SIZE


def _tail_bucket_coalesce_params_for_pass(*, fused_k_class: bool) -> tuple[int | None, float | None, int | None]:
    """Return conservative tail-coalescing controls for sparse pass-2 buckets.

    Tail coalescing stays opt-in because the fused K-class 100k/256 probe
    showed that the old fused default could merge too many medium tail groups
    into 4096-row buckets and slow the sparse pass-2 path. Explicit env
    settings keep the diagnostic behavior available when a dataset has a true
    tiny high-rotation tail.
    """

    explicit_max_images = parse_env_nonnegative_int(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if explicit_max_images is None:
        max_images = _DEFAULT_TAIL_BUCKET_COALESCE_MAX_IMAGES_FUSED_KCLASS if fused_k_class else 0
    else:
        max_images = explicit_max_images
    if max_images <= 1:
        return None, None, None

    max_inflation = _optional_positive_float_env(_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION

    min_bucket_size = _optional_positive_int_env(_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
    return int(max_images), float(max_inflation), int(min_bucket_size)


def _compact_pair_tail_bucket_coalesce_params_for_pass(
    *,
    default_max_images: int | None = None,
    default_max_inflation: float | None = None,
    default_min_bucket_size: int | None = None,
) -> tuple[int | None, float | None, int | None]:
    """Return bounded tail-coalescing controls for compact-pair K-class buckets."""

    max_images = parse_env_nonnegative_int(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = parse_env_nonnegative_int(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = (
            _DEFAULT_COMPACT_PAIR_TAIL_BUCKET_COALESCE_MAX_IMAGES
            if default_max_images is None
            else int(default_max_images)
        )
    if int(max_images) <= 1:
        return None, None, None

    max_inflation = _optional_positive_float_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _optional_positive_float_env(_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = (
            _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION
            if default_max_inflation is None
            else float(default_max_inflation)
        )
    if float(max_inflation) <= 0.0:
        raise ValueError("compact-pair tail coalescing inflation must be positive")

    min_bucket_size = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _optional_positive_int_env(_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = (
            _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
            if default_min_bucket_size is None
            else int(default_min_bucket_size)
        )
    if int(min_bucket_size) <= 0:
        raise ValueError("compact-pair tail coalescing minimum bucket size must be positive")
    return int(max_images), float(max_inflation), int(min_bucket_size)


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

    # ``RECOVAR_SPARSE_PASS2_DEVICE_MEMORY_GB`` overrides the nvidia-smi probe.
    # Keep this as a manual escape hatch for reserving headroom on shared GPUs
    # or working around inaccurate allocator/device probes.
    _override = os.environ.get("RECOVAR_SPARSE_PASS2_DEVICE_MEMORY_GB")
    if _override is not None:
        try:
            override_gb = float(_override.strip())
            if override_gb > 0:
                return int(override_gb * (1024 ** 3))
        except ValueError:
            pass

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


def _device_free_memory_bytes() -> int | None:
    """Return current free memory for the selected physical GPU, if known."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            return _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
    except Exception:
        pass
    return None


def _jax_allocator_free_memory_bytes() -> int | None:
    """Return unused bytes in the active JAX GPU allocator, if reported."""

    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None

    limit = next(
        (
            int(stats[key])
            for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory")
            if stats.get(key) is not None and int(stats[key]) > 0
        ),
        None,
    )
    bytes_in_use = next(
        (
            int(stats[key])
            for key in ("bytes_in_use", "bytesInUse", "memory_in_use")
            if stats.get(key) is not None and int(stats[key]) >= 0
        ),
        None,
    )
    if limit is None or bytes_in_use is None:
        return None
    return max(0, limit - bytes_in_use)


def _exact_raw_diff2_cache_limit_bytes(
    device_memory_bytes: int | None,
    free_device_memory_bytes: int | None,
    allocator_free_memory_bytes: int | None,
    *,
    max_cache_bytes: int = _EXACT_RAW_DIFF2_CACHE_MAX_BYTES,
) -> int:
    """Return the strict per-bucket cap for exact fine-score reuse."""

    if (
        device_memory_bytes is None
        or free_device_memory_bytes is None
        or allocator_free_memory_bytes is None
        or int(device_memory_bytes) <= 0
        or int(free_device_memory_bytes) <= 0
        or int(allocator_free_memory_bytes) <= 0
        or int(max_cache_bytes) <= 0
    ):
        return 0
    return min(
        int(max_cache_bytes),
        int(int(device_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_DEVICE_FRACTION),
        int(int(free_device_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_FREE_FRACTION),
        int(int(allocator_free_memory_bytes) * _EXACT_RAW_DIFF2_CACHE_FREE_FRACTION),
    )


def _exact_raw_diff2_cache_estimated_bytes(
    batch_size: int,
    bucket_size: int,
    n_fine_translations: int,
    dtype=np.float32,
) -> int:
    return (
        int(batch_size)
        * int(bucket_size)
        * int(n_fine_translations)
        * np.dtype(dtype).itemsize
    )


def _exact_raw_diff2_cache_fits_budget(estimated_bytes: int, cache_limit_bytes: int) -> bool:
    return int(estimated_bytes) > 0 and int(estimated_bytes) <= int(cache_limit_bytes)


def _dtype_itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def _complex_counterpart_real_dtype(complex_dtype):
    complex_dtype = np.dtype(complex_dtype)
    if complex_dtype.itemsize <= np.dtype(np.complex64).itemsize:
        return np.float32
    return np.float64


def _auto_hypotheses_per_microbatch(
    *,
    score_only: bool,
    fused_k_class: bool = False,
    fused_k_class_count: int | None = None,
    n_score_pixels: int | None,
    device_memory_bytes: int | None,
    score_complex_dtype=np.complex64,
) -> int | None:
    if device_memory_bytes is None or n_score_pixels is None or int(n_score_pixels) <= 0:
        return None
    if score_only:
        fraction = _AUTO_SCORE_ONLY_HYPOTHESIS_DEVICE_FRACTION
    elif fused_k_class:
        if fused_k_class_count is None or int(fused_k_class_count) <= 0:
            raise ValueError("fused_k_class_count must be positive for fused K-class planning")
        bytes_per_score_pixel = _dtype_itemsize(score_complex_dtype)
        return max(
            1,
            int(
                float(device_memory_bytes)
                * _AUTO_FUSED_KCLASS_SCORE_GATHER_DEVICE_FRACTION
                * int(fused_k_class_count)
                / (
                    int(n_score_pixels)
                    * bytes_per_score_pixel
                    * _AUTO_FUSED_KCLASS_LIVE_COMPLEX_GATHERS
                )
            ),
        )
    else:
        fraction = _AUTO_FULL_HYPOTHESIS_DEVICE_FRACTION
    # The score kernel's dominant live block scales with candidate count times
    # active Fourier pixels. This keeps larger windows and smaller GPUs from
    # inheriting the same candidate cap as low-resolution H100 runs.
    bytes_per_score_pixel = _dtype_itemsize(score_complex_dtype)
    return max(1, int(float(device_memory_bytes) * fraction / (int(n_score_pixels) * bytes_per_score_pixel)))


def _max_hypotheses_per_microbatch_for_pass(
    *,
    score_only: bool,
    use_window: bool,
    has_external_normalization: bool,
    conservative_dump_execution: bool,
    fused_k_class: bool = False,
    fused_k_class_count: int | None = None,
    n_score_pixels: int | None = None,
    device_memory_bytes: int | None = None,
    score_complex_dtype=np.complex64,
) -> int:
    if score_only and use_window and not has_external_normalization and not conservative_dump_execution:
        override = _optional_positive_int_env(_SCORE_ONLY_MAX_HYPOTHESES_ENV)
        auto = _auto_hypotheses_per_microbatch(
            score_only=True,
            fused_k_class=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory_bytes,
            score_complex_dtype=score_complex_dtype,
        )
        if override is not None:
            if auto is not None and int(override) < int(auto):
                logger.warning(
                    "%s=%d is below the auto sparse pass-2 score-only cap %d; "
                    "this can fragment buckets and slow pass-2.",
                    _SCORE_ONLY_MAX_HYPOTHESES_ENV,
                    int(override),
                    int(auto),
                )
            return override
        return int(auto) if auto is not None else _DEFAULT_SCORE_ONLY_MAX_HYPOTHESES_PER_MICROBATCH
    override = _optional_positive_int_env(_MAX_HYPOTHESES_ENV)
    auto = _auto_hypotheses_per_microbatch(
        score_only=False,
        fused_k_class=fused_k_class,
        fused_k_class_count=fused_k_class_count,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory_bytes,
        score_complex_dtype=score_complex_dtype,
    )
    if override is not None:
        if auto is not None and int(override) < int(auto):
            logger.warning(
                "%s=%d is below the auto sparse pass-2 cap %d; "
                "this can fragment buckets and slow pass-2.",
                _MAX_HYPOTHESES_ENV,
                int(override),
                int(auto),
            )
        return override
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


def _max_projection_gather_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_PROJECTION_GATHER_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_GATHER_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_GATHER_DEVICE_FRACTION))


def _max_noise_block_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_NOISE_BLOCK_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_NOISE_BLOCK_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_NOISE_BLOCK_DEVICE_FRACTION))


def _max_adjoint_block_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_ADJOINT_BLOCK_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_ADJOINT_BLOCK_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_ADJOINT_BLOCK_DEVICE_FRACTION))


def _compact_pair_dense_mstep_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES_ENV)
    if override is not None:
        return int(override)
    return _max_adjoint_block_bytes_for_pass(device_memory_bytes)


def _projection_cache_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_CACHE_DEVICE_FRACTION))


def _projection_call_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION))


def _bucket_summary(buckets, size_key: str = "bucket_size") -> str:
    if not buckets:
        return "empty"
    sizes = np.asarray([int(bucket[size_key]) for bucket in buckets], dtype=np.int64)
    image_counts = np.asarray([len(bucket["image_indices"]) for bucket in buckets], dtype=np.int64)
    unique, counts = np.unique(sizes, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist(), strict=True), key=lambda item: item[1], reverse=True)[:8]
    return (
        f"bucket_size min/med/mean/max={int(sizes.min())}/{int(np.median(sizes))}/"
        f"{float(np.mean(sizes)):.1f}/{int(sizes.max())}, "
        f"images_per_bucket med/max={int(np.median(image_counts))}/{int(image_counts.max())}, "
        f"top_bucket_counts={top}"
    )


def _bucket_group_stats(buckets, size_key: str = "bucket_size") -> dict[int, tuple[int, int]]:
    stats: dict[int, list[int]] = {}
    for bucket in buckets:
        bucket_size = int(bucket[size_key])
        entry = stats.setdefault(bucket_size, [0, 0])
        entry[0] += 1
        entry[1] += len(bucket["image_indices"])
    return {bucket_size: (counts[0], counts[1]) for bucket_size, counts in stats.items()}


def _tag_k_class_execution_bucket(bucket, *, mode: str):
    tagged = dict(bucket)
    tagged["_execution_mode"] = mode
    if mode == "compact_pair":
        tagged["_execution_size_key"] = "pair_bucket_size"
    elif mode == "rectangular":
        tagged["_execution_size_key"] = "bucket_size"
    else:
        raise ValueError(f"Unknown sparse K-class execution mode {mode!r}")
    tagged["_execution_bucket_size"] = int(tagged[tagged["_execution_size_key"]])
    return tagged


def _k_class_execution_bucket_group_stats(buckets):
    stats: dict[tuple[str, str, int], list[int]] = {}
    for bucket in buckets:
        mode = str(bucket.get("_execution_mode", "rectangular"))
        size_key = str(bucket.get("_execution_size_key", "bucket_size"))
        bucket_size = int(bucket.get("_execution_bucket_size", bucket[size_key]))
        key = (mode, size_key, bucket_size)
        entry = stats.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += len(bucket["image_indices"])
    return {key: (counts[0], counts[1]) for key, counts in stats.items()}


def _hybrid_k_class_compact_pair_execution_buckets(
    dense_buckets,
    compact_pair_buckets,
    *,
    min_pair_bucket_size: int,
):
    """Route low pair-count images through rectangular buckets and high tails through compact pairs."""

    min_pair_bucket_size = int(min_pair_bucket_size)
    if min_pair_bucket_size <= 0:
        raise ValueError("min_pair_bucket_size must be positive")

    compact_execution_buckets = []
    compact_image_indices = set()
    for bucket in compact_pair_buckets:
        if int(bucket["pair_bucket_size"]) < min_pair_bucket_size:
            continue
        tagged = _tag_k_class_execution_bucket(bucket, mode="compact_pair")
        compact_execution_buckets.append(tagged)
        compact_image_indices.update(int(idx) for idx in np.asarray(bucket["image_indices"], dtype=np.int64))

    rectangular_execution_buckets = []
    for bucket in dense_buckets:
        image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
        if compact_image_indices:
            keep_mask = np.asarray([int(idx) not in compact_image_indices for idx in image_indices], dtype=bool)
            image_indices = image_indices[keep_mask]
        if image_indices.size == 0:
            continue
        rectangular_bucket = dict(bucket)
        rectangular_bucket["image_indices"] = np.asarray(image_indices, dtype=np.int64)
        rectangular_execution_buckets.append(
            _tag_k_class_execution_bucket(rectangular_bucket, mode="rectangular"),
        )

    return rectangular_execution_buckets + compact_execution_buckets


def _compact_pair_buckets_for_execution_threshold(compact_pair_buckets, min_pair_bucket_size: int | None):
    """Return compact buckets eligible for execution under the hybrid threshold."""

    if min_pair_bucket_size is None:
        return list(compact_pair_buckets)
    min_pair_bucket_size = int(min_pair_bucket_size)
    if min_pair_bucket_size <= 0:
        raise ValueError("min_pair_bucket_size must be positive")
    return [
        bucket
        for bucket in compact_pair_buckets
        if int(bucket["pair_bucket_size"]) >= min_pair_bucket_size
    ]


def _validate_k_class_execution_bucket_partition(execution_buckets, *, n_images: int) -> None:
    """Validate that execution buckets cover each image exactly once."""

    n_images = int(n_images)
    if n_images < 0:
        raise ValueError("n_images must be non-negative")
    if n_images == 0:
        if execution_buckets:
            raise ValueError("Execution buckets must be empty when n_images=0")
        return
    if not execution_buckets:
        raise ValueError(f"Execution buckets are empty for {n_images} images")

    image_indices = np.concatenate(
        [
            np.asarray(bucket["image_indices"], dtype=np.int64).reshape(-1)
            for bucket in execution_buckets
        ],
    )
    if image_indices.size != n_images:
        raise ValueError(
            "Execution bucket image coverage count mismatch: "
            f"got {image_indices.size}, expected {n_images}",
        )
    if int(image_indices.min(initial=0)) < 0 or int(image_indices.max(initial=-1)) >= n_images:
        raise ValueError(
            "Execution bucket image indices out of range for "
            f"{n_images} images",
        )

    counts = np.bincount(image_indices, minlength=n_images)
    missing = np.flatnonzero(counts == 0)
    duplicated = np.flatnonzero(counts > 1)
    if missing.size or duplicated.size:
        raise ValueError(
            "Execution buckets must partition images exactly once "
            f"(missing={missing[:8].tolist()}, duplicated={duplicated[:8].tolist()})",
        )


def _compact_pair_threshold_report_thresholds() -> tuple[int, ...]:
    raw_thresholds = parse_env_int_set(_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT_ENV)
    if raw_thresholds is None:
        return _DEFAULT_COMPACT_PAIR_THRESHOLD_REPORT
    return tuple(sorted(int(value) for value in raw_thresholds if int(value) > 0))


def _compact_pair_hybrid_threshold_reports(
    dense_buckets,
    compact_pair_buckets,
    *,
    thresholds: tuple[int, ...],
    n_classes: int,
    n_fine_trans: int,
):
    """Estimate hybrid compact-pair routing cost for candidate thresholds."""

    baseline_rectangular_candidates = int(
        sum(
            int(n_classes) * len(bucket["image_indices"]) * int(bucket["bucket_size"]) * int(n_fine_trans)
            for bucket in dense_buckets
        )
    )
    reports = []
    for threshold in thresholds:
        threshold = int(threshold)
        compact_buckets_for_threshold = [
            bucket
            for bucket in compact_pair_buckets
            if int(bucket["pair_bucket_size"]) >= threshold
        ]
        compact_image_indices = {
            int(idx)
            for bucket in compact_buckets_for_threshold
            for idx in np.asarray(bucket["image_indices"], dtype=np.int64)
        }
        compact_candidates = int(
            sum(
                int(n_classes) * len(bucket["image_indices"]) * int(bucket["pair_bucket_size"])
                for bucket in compact_buckets_for_threshold
            )
        )

        rectangular_buckets = 0
        rectangular_images = 0
        rectangular_candidates = 0
        for bucket in dense_buckets:
            image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
            if compact_image_indices:
                keep_mask = np.asarray(
                    [int(idx) not in compact_image_indices for idx in image_indices],
                    dtype=bool,
                )
                image_indices = image_indices[keep_mask]
            if image_indices.size == 0:
                continue
            rectangular_buckets += 1
            rectangular_images += int(image_indices.size)
            rectangular_candidates += (
                int(n_classes)
                * int(image_indices.size)
                * int(bucket["bucket_size"])
                * int(n_fine_trans)
            )

        total_candidates = int(rectangular_candidates + compact_candidates)
        reports.append(
            {
                "threshold": threshold,
                "compact_buckets": len(compact_buckets_for_threshold),
                "compact_images": len(compact_image_indices),
                "rectangular_buckets": rectangular_buckets,
                "rectangular_images": rectangular_images,
                "rectangular_candidate_slots": rectangular_candidates,
                "compact_candidate_slots": compact_candidates,
                "total_candidate_slots": total_candidates,
                "slot_reduction": (
                    float(baseline_rectangular_candidates) / float(total_candidates)
                    if total_candidates > 0
                    else float("inf")
                ),
            }
        )
    return reports


def _add_sparse_group_timing(group_timing: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if group_timing is None:
        return
    group_timing[key] = group_timing.get(key, 0.0) + float(elapsed_s)


def _log_sparse_kclass_group_timing(
    group_key: tuple[str, str, int],
    group_timing: dict[str, float] | None,
    *,
    wall_s: float,
) -> None:
    if group_timing is None:
        return
    build_s = group_timing.get("build", 0.0)
    fetch_s = group_timing.get("fetch", 0.0)
    prepare_s = group_timing.get("prepare", 0.0)
    score_s = group_timing.get("score", 0.0)
    mstep_noise_stats_s = group_timing.get("mstep_noise_stats", 0.0)
    mstep_weighted_sums_s = group_timing.get("mstep_weighted_sums", 0.0)
    mstep_adjoint_s = group_timing.get("mstep_adjoint", 0.0)
    noise_s = group_timing.get("noise", 0.0)
    stats_s = group_timing.get("stats", 0.0)
    total_profiled_s = build_s + fetch_s + prepare_s + score_s + mstep_noise_stats_s
    logger.info(
        "Sparse fused K-class pass-2 bucket group timing: mode=%s %s=%d "
        "build=%.2fs fetch=%.2fs prepare=%.2fs score=%.2fs "
        "mstep_noise_stats=%.2fs mstep_weighted_sums=%.2fs "
        "mstep_adjoint=%.2fs noise=%.2fs stats=%.2fs "
        "total_profiled=%.2fs wall=%.2fs",
        group_key[0],
        group_key[1],
        group_key[2],
        build_s,
        fetch_s,
        prepare_s,
        score_s,
        mstep_noise_stats_s,
        mstep_weighted_sums_s,
        mstep_adjoint_s,
        noise_s,
        stats_s,
        total_profiled_s,
        float(wall_s),
    )


def _max_images_for_translation_tile(
    image_shape,
    n_fine_trans,
    *,
    max_tile_bytes=384 * 1024**2,
    complex_dtype=np.complex64,
    n_half_pixels: int | None = None,
):
    """Limit one translated-image tile allocation to a bounded size."""
    half_image_size = (
        max(1, int(n_half_pixels))
        if n_half_pixels is not None
        else int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    )
    bytes_per_complex_value = _dtype_itemsize(complex_dtype)
    bytes_per_image = int(n_fine_trans) * half_image_size * bytes_per_complex_value
    return max(1, int(max_tile_bytes) // max(1, bytes_per_image))


def _projection_cache_transient_bytes(
    n_rotations: int,
    n_half_pixels: int,
    *,
    projection_complex_dtype=np.complex64,
    include_abs2: bool,
) -> int:
    complex_bytes = _dtype_itemsize(projection_complex_dtype)
    total = int(n_rotations) * int(n_half_pixels) * complex_bytes
    if include_abs2:
        real_dtype = _complex_counterpart_real_dtype(projection_complex_dtype)
        total += int(n_rotations) * int(n_half_pixels) * _dtype_itemsize(real_dtype)
    return int(total)


def _projection_cache_budget_complex_dtype(
    projection_source_dtype,
    score_complex_dtype,
    *,
    use_relion_projector: bool = False,
):
    dtype = np.promote_types(np.dtype(projection_source_dtype), np.dtype(score_complex_dtype))
    if use_relion_projector:
        # RELION Projector parity uses float64 interpolation weights, which
        # promotes complex64 projector data to complex128 before the caller's
        # output cast. Budget the transient allocation, not the retained cache.
        dtype = np.promote_types(dtype, np.dtype(np.complex128))
    return dtype


def _projection_cache_fits_budget(transient_bytes: int, max_bytes: int, *, n_classes: int = 1) -> bool:
    return int(transient_bytes) * max(1, int(n_classes)) <= int(max_bytes)


def _max_projected_rotations_per_call_for_pass(
    *,
    device_memory_bytes: int | None,
    n_projection_pixels: int,
    projection_complex_dtype,
    include_abs2: bool,
) -> int | None:
    override = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None or int(n_projection_pixels) <= 0:
        return None
    max_bytes = _projection_call_max_bytes_for_pass(device_memory_bytes)
    bytes_per_rotation = _projection_cache_transient_bytes(
        1,
        int(n_projection_pixels),
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=bool(include_abs2),
    )
    if max_bytes <= 0 or bytes_per_rotation <= 0:
        return None
    return max(1, int(max_bytes) // int(bytes_per_rotation))


def _projection_budget_pixels_for_pass(
    n_half_pixels: int,
    *,
    use_window: bool,
    use_relion_projector: bool,
) -> int:
    """Effective projection pixels for the sparse pass-2 projection cap.

    Windowed sparse pass-2 only keeps score/reconstruction rows after
    projection, but RELION's centered Projector handoff currently materializes
    full-half intermediates before gathering the requested windows. Budget that
    path with extra headroom for the centered-row scatter, dense scaling, and
    other live pass-2 buffers so huge one-image compact-pair buckets still split
    before the projection helper allocates.
    """

    pixels = int(n_half_pixels)
    if bool(use_window) and bool(use_relion_projector):
        return max(1, 8 * pixels)
    return max(1, pixels)


def _compute_sparse_pass2_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    projector_output_size: int | None = None,
    **projection_kwargs,
):
    projection_kwargs = dict(projection_kwargs)
    return_abs2 = projection_kwargs.pop("return_abs2", True)
    projection_max_r = projection_kwargs.pop("max_r", None)
    projection_relion_texture_interp = projection_kwargs.get("relion_texture_interp")
    projection_mask_current_image_disk = bool(
        projection_kwargs.pop("mask_current_image_disk", True)
    )
    if projector_output_size is None and projection_max_r is not None:
        projector_output_size = int(2 * float(projection_max_r))
    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")

    def _project(rotations):
        if use_relion_projector:
            return _compute_relion_projector_projections_block(
                relion_projector_half,
                rotations,
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                return_abs2=bool(return_abs2),
                centered_rows=True,
                dense_scale=True,
                relion_texture_interp=projection_relion_texture_interp,
                projector_output_size=projector_output_size,
                mask_current_image_disk=projection_mask_current_image_disk,
            )
        return _compute_projections_block(
            mean_for_proj,
            rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=bool(return_abs2),
            **projection_kwargs,
        )

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if max_projected_rotations is None:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    n_rotations = int(rotations_block.shape[0])
    max_projected_rotations = max(1, int(max_projected_rotations))
    if n_rotations <= max_projected_rotations:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    proj_chunks = []
    abs2_chunks = []
    for start in range(0, n_rotations, max_projected_rotations):
        stop = min(start + max_projected_rotations, n_rotations)
        proj_chunk, abs2_chunk = _project(rotations_block[start:stop])
        if output_complex_dtype is not None:
            proj_chunk = proj_chunk.astype(output_complex_dtype)
        if abs2_chunk is not None and output_abs2_dtype is not None:
            abs2_chunk = abs2_chunk.astype(output_abs2_dtype)
        proj_chunks.append(proj_chunk)
        abs2_chunks.append(abs2_chunk)

    proj_half = jnp.concatenate(proj_chunks, axis=0)
    if all(abs2_chunk is None for abs2_chunk in abs2_chunks):
        return proj_half, None
    if any(abs2_chunk is None for abs2_chunk in abs2_chunks):
        raise RuntimeError("Inconsistent projection abs2 chunks")
    return proj_half, jnp.concatenate(abs2_chunks, axis=0)


def _projection_kwargs_for_relion_score_window(
    projection_kwargs,
    *,
    use_relion_projector: bool,
    current_size: int | None,
):
    """Keep the RELION projector crop large enough for the particle image.

    ``r_max`` describes the model sphere, but it does not always describe the
    particle-image crop.  In particular, fresh first-iteration CC can score a
    size-58 particle image from a projector whose model ``r_max`` is 28.  A
    crop inferred as ``2 * r_max == 56`` drops the valid ``ky=-28`` row before
    the score window gathers it.  RELION projects into the particle-image box
    and clips samples independently to the model sphere, so preserve that
    distinction here.
    """

    kwargs = dict(projection_kwargs)
    if use_relion_projector:
        if current_size is None:
            raise ValueError("windowed RELION projection requires current_size")
        kwargs["projector_output_size"] = int(current_size)
    return kwargs


def _compute_sparse_pass2_windowed_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    score_indices,
    recon_indices=None,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    **projection_kwargs,
):
    """Project in capped chunks and retain only score/reconstruction windows."""

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)

    projection_kwargs = dict(projection_kwargs)
    projection_kwargs["return_abs2"] = False
    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    recon_indices = None if recon_indices is None else jnp.asarray(recon_indices, dtype=jnp.int32)

    n_rotations = int(rotations_block.shape[0])
    if max_projected_rotations is None:
        chunk_ranges = [(0, n_rotations)]
    else:
        max_projected_rotations = max(1, int(max_projected_rotations))
        chunk_ranges = [
            (start, min(start + max_projected_rotations, n_rotations))
            for start in range(0, n_rotations, max_projected_rotations)
        ]

    score_chunks = []
    recon_chunks = []
    for start, stop in chunk_ranges:
        proj_chunk, _ = _compute_sparse_pass2_projections_block(
            mean_for_proj,
            rotations_block[start:stop],
            image_shape,
            proj_volume_shape,
            disc_type,
            max_projected_rotations=None,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            **projection_kwargs,
        )
        score_chunk = proj_chunk[:, score_indices]
        if output_complex_dtype is not None:
            score_chunk = score_chunk.astype(output_complex_dtype)
        score_chunks.append(score_chunk)
        if recon_indices is not None:
            recon_chunk = proj_chunk[:, recon_indices]
            if output_complex_dtype is not None:
                recon_chunk = recon_chunk.astype(output_complex_dtype)
            recon_chunks.append(recon_chunk)
        del proj_chunk

    score_proj = jnp.concatenate(score_chunks, axis=0)
    if recon_indices is None:
        return score_proj, None, None
    recon_proj = jnp.concatenate(recon_chunks, axis=0)
    recon_abs2 = jnp.abs(recon_proj) ** 2
    if output_abs2_dtype is not None:
        recon_abs2 = recon_abs2.astype(output_abs2_dtype)
    return score_proj, recon_proj, recon_abs2


def _compute_noise_block_chunked(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    shell_count,
    *,
    max_block_bytes: int | None,
):
    """Run ``compute_noise_block`` in row chunks when one bucket is too large."""

    n_rows = int(proj_half.shape[0])
    if n_rows <= 0:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )
    if max_block_bytes is None:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )

    n_pixels = int(proj_half.shape[1])
    complex_bytes = max(
        _dtype_itemsize(proj_half.dtype),
        _dtype_itemsize(summed_masked.dtype),
    )
    real_bytes = max(
        _dtype_itemsize(proj_abs2_half.dtype),
        _dtype_itemsize(ctf_probs.dtype),
        _dtype_itemsize(noise_variance_half.dtype),
    )
    # compute_noise_block's live temporaries include complex cross terms and
    # several real products. Keep the estimate conservative because this path
    # is only needed for pathological sparse tail buckets.
    bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
    max_rows = max(1, int(max_block_bytes) // bytes_per_row)
    if n_rows <= max_rows:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _noise_block_chunk_log_keys:
        _noise_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 noise block chunking: rows=%d pixels=%d max_rows=%d chunks=%d max_block_bytes=%.2f GiB",
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            int(max_block_bytes) / float(1024**3),
        )
    accumulator_dtype = jnp.result_type(
        proj_abs2_half.dtype,
        ctf_probs.dtype,
        noise_variance_half.dtype,
    )
    noise_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    a2_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    xa_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, a2_chunk, xa_chunk = _compute_noise_block(
            proj_half[start:stop],
            proj_abs2_half[start:stop],
            summed_masked[start:stop],
            ctf_probs[start:stop],
            noise_variance_half,
            shell_indices,
            shell_count,
        )
        noise_total = noise_total + noise_chunk
        a2_total = a2_total + a2_chunk
        xa_total = xa_total + xa_chunk
    return noise_total, a2_total, xa_total


@partial(jax.jit, static_argnames=("shell_count", "batch_size"))
def _compute_noise_block_and_norm_residual_from_flat_rows(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
):
    """Return shell-binned noise and per-image norm residuals for active rows."""

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2 = jnp.sum(a2_terms, axis=0)
    a2_per_row = jnp.sum(a2_terms, axis=1)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(cross_terms, axis=0)
    xa = jnp.where(cross.real != 0.0, noise_variance_half * cross.real, 0.0)
    xa_per_row = jnp.sum(noise_variance_half[None, :] * cross_terms.real, axis=1)

    block_noise = a2 - 2.0 * xa
    # No explicit dtype cast: preserve whatever real dtype the inputs
    # naturally promote to (float64 under double-precision scoring), same
    # as compute_noise_block/compute_norm_residual_per_image. The zero-init
    # container below must match residual_per_row's dtype exactly --
    # jnp .at[].add() requires an exact dtype match, unlike plain addition.
    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)

    residual_per_row = a2_per_row - 2.0 * xa_per_row
    norm_residual = jnp.zeros(int(batch_size), dtype=residual_per_row.dtype).at[flat_image_indices].add(
        residual_per_row
    )
    return noise_shells, norm_residual


@partial(jax.jit, static_argnames=("shell_count", "batch_size"))
def _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
):
    """Real-valued residual term form for K-class noise/norm accumulation."""

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    summed_has_mass = summed_masked != 0.0
    cross_real = (proj_half.real * summed_masked.real) + (proj_half.imag * summed_masked.imag)
    cross_real = jnp.where(summed_has_mass, cross_real, 0.0)
    xa_terms = noise_variance_half[None, :] * cross_real
    residual_terms = a2_terms - 2.0 * xa_terms

    block_noise = jnp.sum(residual_terms, axis=0)
    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)

    residual_per_row = jnp.sum(residual_terms, axis=1)
    norm_residual = jnp.zeros(int(batch_size), dtype=residual_per_row.dtype).at[flat_image_indices].add(
        residual_per_row
    )
    return noise_shells, norm_residual


def _compute_noise_block_and_norm_residual_chunked(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
    max_block_bytes: int | None,
):
    """Run fused noise shell / norm-residual accumulation in row chunks."""

    compute_block = (
        _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms
        if parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
        else _compute_noise_block_and_norm_residual_from_flat_rows
    )
    n_rows = int(proj_half.shape[0])
    if n_rows <= 0:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    if max_block_bytes is None:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )

    n_pixels = int(proj_half.shape[1])
    complex_bytes = max(
        _dtype_itemsize(proj_half.dtype),
        _dtype_itemsize(summed_masked.dtype),
    )
    real_bytes = max(
        _dtype_itemsize(proj_abs2_half.dtype),
        _dtype_itemsize(ctf_probs.dtype),
        _dtype_itemsize(noise_variance_half.dtype),
    )
    bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
    max_rows = max(1, int(max_block_bytes) // bytes_per_row)
    if n_rows <= max_rows:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _noise_block_chunk_log_keys:
        _noise_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 fused noise/norm block chunking: rows=%d pixels=%d max_rows=%d "
            "chunks=%d max_block_bytes=%.2f GiB",
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            int(max_block_bytes) / float(1024**3),
        )
    accumulator_dtype = jnp.result_type(
        proj_abs2_half.dtype,
        ctf_probs.dtype,
        noise_variance_half.dtype,
    )
    noise_total = jnp.zeros(int(shell_count), dtype=accumulator_dtype)
    norm_total = jnp.zeros(int(batch_size), dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, norm_chunk = compute_block(
            proj_half[start:stop],
            proj_abs2_half[start:stop],
            summed_masked[start:stop],
            ctf_probs[start:stop],
            noise_variance_half,
            shell_indices,
            flat_image_indices[start:stop],
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
        noise_total = noise_total + noise_chunk
        norm_total = norm_total + norm_chunk
    return noise_total, norm_total


def _weighted_image_power_shells_and_per_image(
    processed_half,
    shell_indices_half,
    support_mass,
    *,
    shell_count: int,
    norm_unweighted_shell_cutoff: int | None = None,
    norm_unweighted_high_shell=None,
    include_unweighted_high_shell: bool = True,
    valid_image_mask=None,
    source_faithful_spectrum_norm: bool | None = None,
):
    """Accumulate image power for noise shells and per-image norm correction.

    Inside the current model size, shell sums use the same significant-support
    mass as the A2/XA residual terms.  Above it, RELION adds ``power_img`` once
    per particle outside the class/posterior loop, so both the noise spectrum
    and norm-correction tail are unweighted.  Fourier pixels assigned to the
    shell-binning sentinel are excluded.
    """

    pixel_power = jnp.abs(processed_half) ** 2
    mass = jnp.asarray(support_mass, dtype=pixel_power.dtype)
    shell_indices_half = jnp.asarray(shell_indices_half)
    valid_norm_shell = (shell_indices_half >= 0) & (shell_indices_half < int(shell_count))
    shell_mass = jnp.where(valid_norm_shell[None, :], mass[:, None], 0.0)
    norm_mass = jnp.where(valid_norm_shell[None, :], mass[:, None], 0.0)
    if norm_unweighted_shell_cutoff is not None:
        full_mass = (
            jnp.ones_like(mass)
            if valid_image_mask is None
            else jnp.asarray(valid_image_mask, dtype=pixel_power.dtype)
        )
        unweighted_shell = valid_norm_shell & (shell_indices_half > int(norm_unweighted_shell_cutoff))
        high_shell_mass = full_mass if include_unweighted_high_shell else jnp.zeros_like(full_mass)
        shell_mass = jnp.where(unweighted_shell[None, :], high_shell_mass[:, None], shell_mass)
        norm_mass = jnp.where(unweighted_shell[None, :], high_shell_mass[:, None], norm_mass)
    weighted_pixel_power = pixel_power * shell_mass
    # Keep the image-power reduction in the producer precision.  RELION's
    # RFLOAT path is binary64 for the double-precision oracle, and narrowing
    # here perturbs both the shell noise statistics and the per-image norm
    # correction before either is accumulated into the host float64 totals.
    weighted_half = jnp.sum(weighted_pixel_power, axis=0)
    if parse_env_flag("RECOVAR_DISABLE_CUDA", default=False):
        # ``bins.at[indices].add`` lowers to unordered GPU atomics even when
        # RECOVAR's custom CUDA path is explicitly disabled.  Keep this
        # fallback repeatable by reducing each shell independently instead.
        shell_ids = jnp.arange(int(shell_count), dtype=shell_indices_half.dtype)
        weighted_shells = jnp.sum(
            jnp.where(
                shell_indices_half[None, :] == shell_ids[:, None],
                weighted_half[None, :],
                0.0,
            ),
            axis=1,
        )
    else:
        weighted_shells = bin_shell_values_jax(weighted_half, shell_indices_half, shell_count)
    if source_faithful_spectrum_norm is None:
        source_faithful_spectrum_norm = parse_env_flag(
            _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
            default=False,
        )
    source_faithful_spectrum_norm = bool(source_faithful_spectrum_norm)
    deterministic_norm_reduction = source_faithful_spectrum_norm or parse_env_flag(
        "RECOVAR_K1_RELION_DETERMINISTIC_NORM_REDUCTION",
        default=False,
    )
    norm_reduction_dtype = jnp.float64 if deterministic_norm_reduction else pixel_power.dtype
    weighted_per_image = jnp.sum(
        (pixel_power * norm_mass).astype(norm_reduction_dtype),
        axis=-1,
    )
    if norm_unweighted_high_shell is not None and include_unweighted_high_shell:
        if norm_unweighted_shell_cutoff is None:
            raise ValueError("a replacement high-shell norm term requires a shell cutoff")
        replacement_high = jnp.asarray(norm_unweighted_high_shell, dtype=norm_reduction_dtype)
        if replacement_high.shape != mass.shape:
            raise ValueError(
                "replacement high-shell norm term must match the particle axis, got "
                f"{replacement_high.shape} for {mass.shape}"
            )
        generic_high = jnp.sum(
            jnp.where(unweighted_shell[None, :], pixel_power, 0.0).astype(
                norm_reduction_dtype
            ),
            axis=-1,
        )
        # Preserve the current-size norm path and the separate shell/noise
        # reduction. Replace only the unweighted high-shell norm term
        # with RELION powerClass's divide-before-square float32 arithmetic.
        weighted_per_image = jax.lax.optimization_barrier(weighted_per_image)
        weighted_per_image = weighted_per_image + full_mass * (replacement_high - generic_high)
    return weighted_shells, weighted_per_image.astype(norm_reduction_dtype)


def _make_relion_wavg_rectangle(
    image_shape,
    current_size,
    recon_window_indices,
    *,
    reconstruction_current_size=None,
):
    """Map active reconstruction pixels into RELION's complete Wavg crop.

    ``exact_positions`` retains its historical field name, but may describe
    either the exact BackProjector disk or RELION InitialModel's rounded-shell
    Wavg support. The supplied reconstruction indices select that contract.

    RELION may remap the particle-image ``current_size`` for an optics group
    while retaining the model-coordinate radius for Projector/BackProjector.
    The rectangle and its rounded noise-shell mask therefore use the particle
    size, while the exact projected terms use ``reconstruction_current_size``.
    """

    image_shape = tuple(int(value) for value in image_shape)
    current_size = int(current_size)
    model_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if model_current_size > current_size:
        raise ValueError(
            "RELION Wavg model support cannot exceed the particle-image crop: "
            f"model={model_current_size}, image={current_size}"
        )
    rectangle_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=True,
        include_dc=True,
    )
    rectangle_order = relion_fftw_order_for_square_score_window(
        image_shape,
        current_size,
        rectangle_indices,
    )
    rectangle_indices = rectangle_indices[rectangle_order]
    rounded_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=False,
    )
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        model_current_size,
        include_dc=True,
        exact_radius=True,
    )
    recon_indices = np.asarray(recon_window_indices, dtype=np.int32).reshape(-1)
    exact_support = np.array_equal(np.sort(recon_indices), exact_indices)
    rounded_support = np.array_equal(np.sort(recon_indices), rounded_indices)
    if not (exact_support or rounded_support):
        raise ValueError(
            "RELION Wavg rectangle requires a complete exact-radius or rounded-shell "
            "reconstruction window: "
            f"got {recon_indices.size} pixels, expected {exact_indices.size} or "
            f"{rounded_indices.size}"
        )

    rectangle_position = {
        int(centered_index): position
        for position, centered_index in enumerate(rectangle_indices.tolist())
    }
    try:
        exact_positions = np.asarray(
            [rectangle_position[int(index)] for index in recon_indices],
            dtype=np.int32,
        )
        rounded_positions = np.asarray(
            [rectangle_position[int(index)] for index in rounded_indices],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError("RELION Wavg support is not contained in its square crop") from error

    shell_indices_half = np.asarray(
        make_relion_noise_shell_indices_half(image_shape),
        dtype=np.int32,
    )
    rectangle_shells = shell_indices_half[rectangle_indices]
    valid = np.zeros(rectangle_indices.size, dtype=bool)
    valid[rounded_positions] = True
    rectangle_shells = np.where(valid, rectangle_shells, -1).astype(np.int32)
    expected_rectangle_size = current_size * (current_size // 2 + 1)
    if rectangle_indices.size != expected_rectangle_size:
        raise ValueError(
            "RELION Wavg square crop topology changed: "
            f"got {rectangle_indices.size}, expected {expected_rectangle_size}"
        )
    if np.unique(exact_positions).size != exact_positions.size:
        raise ValueError("RELION Wavg reconstruction position mapping is not bijective")
    return RelionWavgRectangle(
        centered_indices=rectangle_indices.astype(np.int32, copy=False),
        exact_positions=exact_positions,
        shell_indices=rectangle_shells,
    )


def _make_stable_relion_wavg_rectangle(image_shape, shape_plan):
    """Pack a logical Wavg rectangle before its physical-capacity tail.

    RELION's Wavg kernel walks the dense FFTW rectangle, whereas RECOVAR's
    shared projection path stores a compact disk.  Stable shapes are exact
    only when both streams retain their original logical order.  The first
    ``logical_rectangle_pixels`` entries below are therefore byte-for-byte the
    ordinary logical rectangle.  Physical-only pixels follow it and receive a
    sentinel shell; runtime CUDA bounds must never issue that tail.
    """

    logical_recon = shape_plan.packed_indices_np("recon")[
        : shape_plan.logical_reconstruction_pixels
    ]
    physical_recon = shape_plan.packed_indices_np("recon")
    logical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.logical_current_size,
        logical_recon,
    )
    physical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.physical_current_size,
        physical_recon,
    )

    logical_set = set(map(int, logical.centered_indices.tolist()))
    physical_tail = np.asarray(
        [
            int(index)
            for index in physical.centered_indices.tolist()
            if int(index) not in logical_set
        ],
        dtype=np.int32,
    )
    packed_rectangle = np.concatenate(
        (logical.centered_indices, physical_tail),
    ).astype(np.int32, copy=False)
    if packed_rectangle.size != shape_plan.physical_rectangle_pixels:
        raise ValueError(
            "stable RELION Wavg rectangle does not fill its physical capacity: "
            f"got {packed_rectangle.size}, expected {shape_plan.physical_rectangle_pixels}"
        )
    logical_count = shape_plan.logical_rectangle_pixels
    recon_tail_count = (
        shape_plan.physical_reconstruction_pixels
        - shape_plan.logical_reconstruction_pixels
    )
    rectangle_tail_count = packed_rectangle.size - logical_count
    if recon_tail_count > rectangle_tail_count:
        raise ValueError(
            "stable RELION Wavg rectangle tail cannot hold its reconstruction "
            f"tail: recon={recon_tail_count}, rectangle={rectangle_tail_count}"
        )
    # Some pixels newly admitted by the larger physical radius still lie in
    # the *logical* square rectangle (for example, immediately outside its
    # exact-radius disk).  Mapping those pixels by coordinate would make the
    # logical native Wavg/BPref loops consume physical-only values.  Logical
    # reconstruction rows retain their exact FFTW positions; all capacity-only
    # rows instead receive arbitrary unique storage in the inert rectangle
    # tail, whose coordinates are intentionally never issued.
    recon_positions = np.concatenate(
        (
            logical.exact_positions,
            np.arange(
                logical_count,
                logical_count + recon_tail_count,
                dtype=np.int32,
            ),
        )
    ).astype(np.int32, copy=False)
    rectangle_shells = np.full(packed_rectangle.size, -1, dtype=np.int32)
    rectangle_shells[:logical_count] = logical.shell_indices
    return RelionWavgRectangle(
        centered_indices=packed_rectangle,
        exact_positions=recon_positions,
        shell_indices=rectangle_shells,
    )


def _select_optional_wavg_exact_pixels(values, rectangle):
    """Select exact-radius Wavg pixels when the atomic diagnostic is active."""

    if values is None or rectangle is None:
        return None
    return values[:, rectangle.exact_positions]


def _relion_wavg_rectangle_image_power(raw_shifted, posterior):
    """Shared rectangle power contraction; keep the original F32/barrier order."""
    raw_shifted = jnp.asarray(raw_shifted, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    shifted_power = (raw_shifted.real * raw_shifted.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    shifted_power = (shifted_power + raw_shifted.imag * raw_shifted.imag).astype(jnp.float32)
    image_power = jnp.einsum(
        "brt,btp->brp",
        posterior,
        shifted_power,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    return image_power


@jax.jit
def _relion_wavg_rectangle_triplet_terms(
    exact_triplet_terms,
    raw_shifted_rectangle,
    posterior,
    exact_positions,
):
    """Embed exact projection terms in the full native Wavg issue stream."""

    exact_terms = jnp.asarray(exact_triplet_terms, dtype=jnp.float32)
    raw_shifted = jnp.asarray(raw_shifted_rectangle, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    exact_positions = jnp.asarray(exact_positions, dtype=jnp.int32)
    if exact_terms.ndim != 4 or exact_terms.shape[-1] != 3:
        raise ValueError(f"exact Wavg terms must have shape (B,R,P,3), got {exact_terms.shape}")
    if raw_shifted.ndim != 3 or posterior.shape != exact_terms.shape[:2] + (raw_shifted.shape[1],):
        raise ValueError(
            "Wavg rectangle translations/posteriors do not match exact terms: "
            f"terms={exact_terms.shape}, shifted={raw_shifted.shape}, posterior={posterior.shape}"
        )
    if exact_positions.shape != (exact_terms.shape[2],):
        raise ValueError(
            "Wavg exact-position mapping does not match the projected pixel axis: "
            f"positions={exact_positions.shape}, terms={exact_terms.shape}"
        )

    image_power = _relion_wavg_rectangle_image_power(raw_shifted, posterior)
    rectangle_terms = jnp.zeros(
        exact_terms.shape[:2] + (raw_shifted.shape[-1], 3),
        dtype=jnp.float32,
    )
    rectangle_terms = rectangle_terms.at[..., 2].set(image_power)
    return rectangle_terms.at[:, :, exact_positions, :].set(exact_terms)


def _relion_wavg_direct_norm_per_image(
    atomic_diff2_per_pixel,
    shell_indices,
    high_shell_power,
):
    """Reproduce RELION's sequential host norm sum from the direct Wavg buffer."""

    atomic_diff2 = np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    shells = np.asarray(shell_indices, dtype=np.int32).reshape(-1)
    high_shell = np.asarray(high_shell_power, dtype=np.float64).reshape(-1)
    if atomic_diff2.ndim != 2 or atomic_diff2.shape[1] != shells.size:
        raise ValueError(
            "atomic Wavg diff2 must have shape (images, rectangle pixels), got "
            f"{atomic_diff2.shape} for {shells.shape}"
        )
    if high_shell.shape != (atomic_diff2.shape[0],):
        raise ValueError(
            "RELION high-shell norm term must match the image axis, got "
            f"{high_shell.shape} for {atomic_diff2.shape}"
        )
    valid = shells >= 0
    output = np.zeros(atomic_diff2.shape[0], dtype=np.float64)
    for image_row in range(atomic_diff2.shape[0]):
        current_size_sum = np.float64(0.0)
        for value in atomic_diff2[image_row, valid]:
            current_size_sum += np.float64(value)
        output[image_row] = current_size_sum + high_shell[image_row]
    return output


def _relion_wavg_atomic_triplet_terms(
    proj,
    proj_abs2,
    summed_shifted,
    ctf_posterior,
    noise_variance,
    scale,
    raw_shifted_images,
    posterior,
):
    """Form per-rotation Wavg ``[XA, AA, diff2]`` float32 atomic operands.

    RELION accumulates all three quantities in one CUDA thread after its
    translation loop. XA and AA are returned in scale-correction units;
    diff2 stays in the raw residual units used by ``wsum_sigma2_noise``.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    proj_abs2 = jnp.asarray(proj_abs2, dtype=jnp.float32)
    summed_shifted = jnp.asarray(summed_shifted, dtype=jnp.complex64)
    ctf_posterior = jnp.asarray(ctf_posterior, dtype=jnp.float32)
    noise_variance = jnp.asarray(noise_variance, dtype=jnp.float32).reshape(-1)
    scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)
    raw_shifted_images = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)

    ctf_has_mass = ctf_posterior != 0.0
    ctf_posterior_raw = jnp.where(
        ctf_has_mass,
        ctf_posterior * noise_variance[None, None, :],
        0.0,
    )
    aa_raw = jnp.where(ctf_has_mass, proj_abs2 * ctf_posterior_raw, 0.0).astype(
        jnp.float32
    )
    cross_has_mass = summed_shifted != 0.0
    cross = jnp.where(cross_has_mass, proj * jnp.conj(summed_shifted), 0.0)
    xa_raw = (noise_variance[None, None, :] * cross.real).astype(jnp.float32)
    safe_scale = jnp.maximum(scale, jnp.asarray(1e-30, dtype=jnp.float32))
    # Wavg emits all three atomics at every pixel inside current_size. RELION
    # applies its lower-resolution scale-correction cutoff only when the host
    # later consumes XA/AA; masking here changes the CUDA issue stream.
    xa = (xa_raw / safe_scale[:, None, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None, None] ** 2)).astype(jnp.float32)

    # RELION's g_img input is the raw translated preprocessed image, not the
    # CTF/noise-weighted BPref numerator used by RECOVAR's adjoint path.
    shifted_power = (raw_shifted_images.real * raw_shifted_images.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    shifted_power = (
        shifted_power + raw_shifted_images.imag * raw_shifted_images.imag
    ).astype(jnp.float32)
    image_power = jnp.einsum(
        "brt,btp->brp",
        posterior,
        shifted_power,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    diff2 = (
        (image_power + aa_raw)
        - jnp.asarray(2.0, dtype=jnp.float32) * xa_raw
    ).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


@jax.jit
def _relion_wavg_sequential_triplet_terms_jax(
    proj,
    raw_ctf,
    scale,
    raw_shifted_images,
    posterior,
):
    """Reproduce RELION Wavg's translation-loop float32 accumulators.

    RELION forms the CTF-and-scale-corrected reference once per rotation and
    pixel, then visits translations in storage order.  It accumulates squared
    residual, XA, and AA separately in float32 before issuing the three
    rotation-level atomics.  Forming ``image_power + AA - 2 * XA`` after three
    independent reductions is algebraically equivalent but not numerically
    equivalent at this boundary.

    XA and AA are returned in RELION's host scale-correction units.  The
    residual remains in the raw units consumed by ``wsum_sigma2_noise``.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    raw_ctf = jnp.asarray(raw_ctf, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)
    raw_shifted_images = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError(f"Wavg projections must have shape (B,R,P), got {proj.shape}")
    if raw_ctf.shape != (proj.shape[0], proj.shape[2]):
        raise ValueError(
            "Wavg raw CTF must match projection batch/pixel axes, got "
            f"{raw_ctf.shape} for {proj.shape}"
        )
    if raw_shifted_images.ndim != 3:
        raise ValueError(
            "Wavg shifted images must have shape (B,T,P), got "
            f"{raw_shifted_images.shape}"
        )
    if raw_shifted_images.shape[0] != proj.shape[0] or raw_shifted_images.shape[2] != proj.shape[2]:
        raise ValueError(
            "Wavg shifted-image batch/pixel axes must match projections, got "
            f"{raw_shifted_images.shape} for {proj.shape}"
        )
    if posterior.shape != proj.shape[:2] + (raw_shifted_images.shape[1],):
        raise ValueError(
            "Wavg posterior must match projection rotations and image translations, got "
            f"{posterior.shape} for proj={proj.shape}, shifted={raw_shifted_images.shape}"
        )

    ctf_with_scale = (raw_ctf * scale[:, None]).astype(jnp.float32)
    ref_real = (proj.real * ctf_with_scale[:, None, :]).astype(jnp.float32)
    ref_imag = (proj.imag * ctf_with_scale[:, None, :]).astype(jnp.float32)
    zeros = jnp.zeros_like(ref_real, dtype=jnp.float32)

    def add_translation(translation_index, accumulators):
        xa_acc, aa_acc, diff2_acc = accumulators
        weight = posterior[:, :, translation_index]
        trans_real = raw_shifted_images[:, translation_index, :].real
        trans_imag = raw_shifted_images[:, translation_index, :].imag
        diff_real = (ref_real - trans_real[:, None, :]).astype(jnp.float32)
        diff_imag = (ref_imag - trans_imag[:, None, :]).astype(jnp.float32)
        diff_abs2 = (diff_real * diff_real).astype(jnp.float32)
        diff_abs2 = jax.lax.optimization_barrier(diff_abs2)
        diff_abs2 = (diff_abs2 + diff_imag * diff_imag).astype(jnp.float32)
        cross = (ref_real * trans_real[:, None, :]).astype(jnp.float32)
        cross = jax.lax.optimization_barrier(cross)
        cross = (cross + ref_imag * trans_imag[:, None, :]).astype(jnp.float32)
        ref_abs2 = (ref_real * ref_real).astype(jnp.float32)
        ref_abs2 = jax.lax.optimization_barrier(ref_abs2)
        ref_abs2 = (ref_abs2 + ref_imag * ref_imag).astype(jnp.float32)
        weighted = weight[:, :, None]
        return (
            (xa_acc + weighted * cross).astype(jnp.float32),
            (aa_acc + weighted * ref_abs2).astype(jnp.float32),
            (diff2_acc + weighted * diff_abs2).astype(jnp.float32),
        )

    xa_raw, aa_raw, diff2 = jax.lax.fori_loop(
        0,
        raw_shifted_images.shape[1],
        add_translation,
        (zeros, zeros, zeros),
    )
    safe_scale = jnp.maximum(scale, jnp.asarray(1e-30, dtype=jnp.float32))
    xa = (xa_raw / safe_scale[:, None, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None, None] ** 2)).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


def _relion_wavg_sequential_triplet_terms(
    proj,
    raw_ctf,
    scale,
    raw_shifted_images,
    posterior,
    *,
    relion_wavg_sequential_cuda: bool | None = None,
    logical_pixel_count=None,
):
    """Dispatch the shared Wavg translation-order reduction.

    The CUDA path is an explicit performance discriminator.  It keeps the
    same image/rotation/pixel ownership and sequential translation arithmetic
    as the JAX reference while avoiding one XLA loop-body launch per
    translation.  Both local EM and VDAM reach this helper through the shared
    exact-local pass-2 implementation.  A typed policy overrides the legacy
    environment gate; ``None`` preserves its existing behavior.
    """

    use_cuda = (
        parse_env_flag(_RELION_WAVG_SEQUENTIAL_CUDA_ENV, default=False)
        if relion_wavg_sequential_cuda is None
        else bool(relion_wavg_sequential_cuda)
    )
    if use_cuda:
        from recovar import cuda_backproject

        if logical_pixel_count is not None:
            return cuda_backproject.relion_wavg_sequential_runtime_triplet_f32(
                jnp.asarray(proj, dtype=jnp.complex64),
                jnp.asarray(raw_ctf, dtype=jnp.float32),
                jnp.asarray(scale, dtype=jnp.float32).reshape(-1),
                jnp.asarray(raw_shifted_images, dtype=jnp.complex64),
                jnp.asarray(posterior, dtype=jnp.float32),
                jnp.asarray(logical_pixel_count, dtype=jnp.int32),
            )
        return cuda_backproject.relion_wavg_sequential_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            jnp.asarray(raw_ctf, dtype=jnp.float32),
            jnp.asarray(scale, dtype=jnp.float32).reshape(-1),
            jnp.asarray(raw_shifted_images, dtype=jnp.complex64),
            jnp.asarray(posterior, dtype=jnp.float32),
        )
    if logical_pixel_count is not None:
        raise ValueError(
            "stable Fourier-window Wavg requires the runtime-bound CUDA reducer"
        )
    return _relion_wavg_sequential_triplet_terms_jax(
        proj,
        raw_ctf,
        scale,
        raw_shifted_images,
        posterior,
    )


def _replace_low_shell_noise_with_relion_wavg_direct_residual(
    residual_shells,
    image_power_shells,
    atomic_diff2_per_pixel,
    shell_indices,
    *,
    exclusive_shell_stop: int,
):
    """Replace complete low shells with fused Wavg ``diff2`` atomics.

    The fused value already contains image power, A2, and -2*XA.  Therefore
    its covered shells replace both RECOVAR noise-stat components.
    ``exclusive_shell_stop`` is expressed in shell-number coordinates. The
    caller supplies RELION's complete rectangular Wavg buffer, so the rounded
    cutoff shell is replaced in full even though the reconstruction window
    contains only its exact-radius subset.
    """

    residual = np.asarray(residual_shells, dtype=np.float64).copy()
    image_power = np.asarray(image_power_shells, dtype=np.float64).copy()
    atomic_diff2 = np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    shells = np.asarray(shell_indices, dtype=np.int32).reshape(-1)
    if residual.ndim != 1 or image_power.shape != residual.shape:
        raise ValueError(
            "noise residual and image-power shells must be matching vectors, got "
            f"{residual.shape} and {image_power.shape}"
        )
    if atomic_diff2.ndim != 2 or atomic_diff2.shape[1] != shells.size:
        raise ValueError(
            "atomic Wavg diff2 must have shape (images, pixels) matching shell indices, got "
            f"{atomic_diff2.shape} and {shells.shape}"
        )
    shell_stop = min(max(0, int(exclusive_shell_stop)), residual.size)
    valid = (shells >= 0) & (shells < shell_stop)
    direct_shells = np.zeros_like(residual)
    if np.any(valid):
        # Preserve physical particle order, then reconstruction-window pixel
        # order.  The per-pixel values have already undergone RELION-style
        # float32 rotation atomics on device.
        for image_row in range(atomic_diff2.shape[0]):
            np.add.at(
                direct_shells,
                shells[valid],
                atomic_diff2[image_row, valid].astype(np.float64),
            )
    residual[:shell_stop] = direct_shells[:shell_stop]
    image_power[:shell_stop] = 0.0
    return residual, image_power


@jax.jit
def _translated_wavg_low_shell_power_pixels(
    shifted_score,
    translation_posterior,
    shell_indices,
    shell_cutoff,
):
    """Return RELION-Wavg-style low-shell image-power pixels per image.

    RELION forms ``wdiff2`` after translating each image and preserves one
    float32 accumulator per Fourier pixel until the host-side normalization
    sum.  Computing image power from the untranslated image is algebraically
    equivalent only in exact arithmetic; the CUDA translation phase makes the
    distinction observable in float32.  Keep the per-pixel boundary here so
    callers can reproduce RELION's host float64/RFLOAT summation order.
    """

    shifted_score = jnp.asarray(shifted_score, dtype=jnp.complex64)
    translation_posterior = jnp.asarray(translation_posterior, dtype=jnp.float32)
    shell_indices = jnp.asarray(shell_indices, dtype=jnp.int32)
    if shifted_score.ndim != 3:
        raise ValueError(f"translated Wavg images must have shape (B,T,P), got {shifted_score.shape}")
    if translation_posterior.shape != shifted_score.shape[:2]:
        raise ValueError(
            "translation posterior must match translated Wavg batch/translation axes, got "
            f"{translation_posterior.shape} for {shifted_score.shape}"
        )
    if shell_indices.shape != (shifted_score.shape[-1],):
        raise ValueError(
            "translated Wavg shell indices must match the pixel axis, got "
            f"{shell_indices.shape} for {shifted_score.shape}"
        )

    pixel_power = shifted_score.real * shifted_score.real
    pixel_power = jax.lax.optimization_barrier(pixel_power)
    pixel_power = pixel_power + shifted_score.imag * shifted_score.imag
    weighted_pixels = jnp.sum(
        translation_posterior[:, :, None] * pixel_power,
        axis=1,
        dtype=jnp.float32,
    )
    valid_low_shell = (shell_indices >= 0) & (shell_indices <= jnp.asarray(shell_cutoff))
    return jnp.where(valid_low_shell[None, :], weighted_pixels, 0.0).astype(jnp.float32)


def _relion_cuda_translate_wavg_norm_images(
    processed_score_half,
    translation_angles,
    score_window_indices,
    image_shape,
):
    """Translate the raw masked image at RELION's Wavg input boundary."""

    from recovar import cuda_backproject

    processed_score_half = jnp.asarray(processed_score_half, dtype=jnp.complex64)
    score_window_indices = jnp.asarray(score_window_indices, dtype=jnp.int32)
    translation_angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    translated = cuda_backproject.relion_translate_score_f32(
        processed_score_half[:, score_window_indices],
        translation_angles,
        score_window_indices,
        image_shape,
    )
    return translated.reshape(
        processed_score_half.shape[0],
        translation_angles.shape[0],
        score_window_indices.shape[0],
    )


def _replace_untranslated_low_shell_norm_power(
    weighted_img_per_image,
    processed_score_half,
    shifted_score,
    translation_posterior,
    shell_indices_half,
    score_window_indices,
    *,
    shell_cutoff: int,
):
    """Replace RECOVAR's untranslated low-shell norm power with Wavg power."""

    score_window_indices = jnp.asarray(score_window_indices, dtype=jnp.int32)
    window_shell_indices = jnp.asarray(shell_indices_half, dtype=jnp.int32)[score_window_indices]
    shifted_score = jnp.asarray(shifted_score, dtype=jnp.complex64)
    translated_pixels = _translated_wavg_low_shell_power_pixels(
        shifted_score,
        translation_posterior,
        window_shell_indices,
        jnp.asarray(shell_cutoff, dtype=jnp.int32),
    )

    processed_window = jnp.asarray(processed_score_half, dtype=jnp.complex64)[:, score_window_indices]
    untranslated_power = jnp.abs(processed_window) ** 2
    support_mass = jnp.sum(
        jnp.asarray(translation_posterior, dtype=jnp.float32),
        axis=1,
        dtype=jnp.float32,
    )
    valid_low_shell = (window_shell_indices >= 0) & (window_shell_indices <= int(shell_cutoff))
    untranslated_pixels = jnp.where(
        valid_low_shell[None, :],
        untranslated_power * support_mass[:, None],
        0.0,
    ).astype(jnp.float32)

    # RELION copies its per-pixel float32 Wavg accumulators to the host and
    # adds them into an RFLOAT normalization scalar in pixel order.
    translated_host = np.asarray(jax.block_until_ready(translated_pixels), dtype=np.float32)
    untranslated_host = np.asarray(jax.block_until_ready(untranslated_pixels), dtype=np.float32)
    adjustment = np.sum(translated_host, axis=-1, dtype=np.float64) - np.sum(
        untranslated_host,
        axis=-1,
        dtype=np.float64,
    )
    return jnp.asarray(weighted_img_per_image, dtype=jnp.float64) + jnp.asarray(
        adjustment,
        dtype=jnp.float64,
    )


def _flat_block_row_bytes(flat_block) -> int:
    if flat_block is None or len(flat_block.shape) == 0:
        return 1
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    return max(1, n_pixels * _dtype_itemsize(flat_block.dtype))


def _adjoint_block_chunk_rows(flat_block, *, max_block_bytes: int) -> int:
    if flat_block is None:
        return 1
    row_bytes = _flat_block_row_bytes(flat_block)
    return max(1, int(max_block_bytes) // row_bytes)


def _accumulate_relion_x_half_per_particle_launches(
    values,
    ctf_values,
    rotations,
    actual_counts,
    y_volume,
    ctf_volume,
    *,
    window_indices,
    image_shape,
    volume_shape,
    disc_type,
    half_volume,
    max_r,
    log_label_prefix: str,
    winner_take_all: bool = False,
    strict_particle_order: bool = False,
):
    """Accumulate one particle-owned orientation grid per FFI launch.

    This preserves a RELION-like particle launch boundary and local orientation
    order.  In soft-posterior mode it remains a causal arm rather than strict
    RELION closure because RECOVAR reduces translations before this boundary.
    The optional fused-atomics diagnostic interleaves each neighbor's real,
    imaginary, and weight atomics in one kernel.
    """

    diagnostic_fused_atomics = bpref_diagnostics.relion_x_half_bp_fused_atomics_enabled()
    # Fresh K=1 --firstiter_cc contributes exactly one winning hypothesis per
    # particle. Unlike later soft-posterior iterations, no translation
    # reduction changes the contributor stream. RELION still dispatches each
    # MPI pool concurrently through per-thread CUDA streams, so the optional
    # pool diagnostic below changes only that outer launch grouping.
    production_firstiter_fused_atomics = bool(winner_take_all and strict_particle_order)
    use_fused_atomics = bool(
        diagnostic_fused_atomics or production_firstiter_fused_atomics
    )
    if not winner_take_all and not strict_particle_order:
        logger.warning(
            "RECOVAR soft-posterior per-particle x-half causal arm enabled; "
            "this does not claim RELION hypothesis-arithmetic closure"
        )
    if use_fused_atomics:
        import recovar.cuda_backproject as cuda_backproject

        if (
            diagnostic_fused_atomics
            and not production_firstiter_fused_atomics
            and not bpref_diagnostics.relion_x_half_bp_per_particle_launch_enabled()
        ):
            raise RuntimeError(
                "RELION fused-atomics diagnostic requires "
                "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1"
            )
        if (
            diagnostic_fused_atomics
            and not production_firstiter_fused_atomics
            and not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        ):
            raise RuntimeError(
                "RELION fused-atomics diagnostic requires "
                "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1"
            )
        if production_firstiter_fused_atomics:
            logger.info(
                "STRICT-PARITY: fresh K=1 firstiter-CC uses RELION fused "
                "real/imaginary/weight atomics for particle-owned launches "
                "(label=%s)",
                log_label_prefix,
            )
        else:
            logger.info(
                "RELION x-half diagnostic: fused real/imaginary/weight atomics enabled "
                "for particle-owned launches (label=%s)",
                log_label_prefix,
            )

        # The fused CUDA target is specialized by the BPref accumulator
        # dtype. Scoring and M-step precision are independently gated, so
        # convert reduced rows once at this boundary rather than assuming that
        # scoring precision matches accumulator precision.
        fused_complex_dtype = (
            jnp.complex128
            if y_volume.dtype == jnp.dtype(jnp.complex128)
            else jnp.complex64
        )
        values = jnp.asarray(values, dtype=fused_complex_dtype)
        ctf_values = jnp.asarray(ctf_values, dtype=ctf_volume.dtype)

    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    if values.shape[:2] != rotations.shape[:2] or ctf_values.shape[:2] != values.shape[:2]:
        raise ValueError("per-particle x-half diagnostic requires matching (particle, rotation) axes")
    if actual_counts.shape != (int(values.shape[0]),):
        raise ValueError(
            f"per-particle x-half actual_counts shape mismatch: {actual_counts.shape} vs {(int(values.shape[0]),)}"
        )
    particle_pool_size = (
        _optional_positive_int_env(_RELION_X_HALF_BP_PARTICLE_POOL_SIZE_ENV) or 1
    )
    if particle_pool_size > 1:
        if not (winner_take_all and strict_particle_order and use_fused_atomics):
            raise RuntimeError(
                "RELION particle-pool diagnostic requires fresh K=1 winner-take-all, "
                "strict particle order, and fused atomics"
            )
        logger.info(
            "STRICT-PARITY diagnostic: grouping consecutive fresh K=1 BPref "
            "particles into RELION-sized launch pools (pool_size=%d, particles=%d, "
            "label=%s)",
            particle_pool_size,
            int(values.shape[0]),
            log_label_prefix,
        )
    else:
        logger.info(
            "RELION x-half diagnostic: one adjoint launch per particle "
            "(particles=%d, rotations min/median/max=%d/%d/%d, label=%s)",
            int(values.shape[0]),
            int(actual_counts.min()) if actual_counts.size else 0,
            int(np.median(actual_counts)) if actual_counts.size else 0,
            int(actual_counts.max()) if actual_counts.size else 0,
            log_label_prefix,
        )
    for pool_start in range(0, actual_counts.size, particle_pool_size):
        pool_stop = min(pool_start + particle_pool_size, actual_counts.size)
        value_rows = []
        ctf_rows = []
        rotation_rows = []
        for particle_index in range(pool_start, pool_stop):
            count = int(actual_counts[particle_index])
            if count <= 0:
                continue
            particle_slice = (
                slice(particle_index, particle_index + 1),
                slice(0, count),
            )
            value_rows.append(values[particle_slice].reshape(count, values.shape[-1]))
            ctf_rows.append(
                ctf_values[particle_slice].reshape(count, ctf_values.shape[-1])
            )
            rotation_rows.append(rotations[particle_slice].reshape(count, 3, 3))
        if not value_rows:
            continue
        particle_values = jnp.concatenate(value_rows, axis=0)
        particle_ctf_values = jnp.concatenate(ctf_rows, axis=0)
        particle_rotations = jnp.concatenate(rotation_rows, axis=0)
        if use_fused_atomics:
            if disc_type != "linear_interp" or not half_volume:
                raise RuntimeError(
                    "RELION fused-atomics diagnostic requires linear interpolation and a half-volume accumulator"
                )
            y_volume, ctf_volume = cuda_backproject.relion_fused_x_half_backproject_indexed(
                y_volume,
                ctf_volume,
                particle_values,
                particle_ctf_values,
                window_indices,
                particle_rotations,
                image_shape=image_shape,
                volume_shape=volume_shape,
                max_r=max_r,
            )
        else:
            y_volume = _adjoint_slice_volume_windowed(
                particle_values,
                window_indices,
                particle_rotations,
                y_volume,
                image_shape,
                volume_shape,
                disc_type,
                True,
                half_volume,
                max_r,
                True,
            )
            ctf_volume = _adjoint_slice_volume_windowed(
                particle_ctf_values,
                window_indices,
                particle_rotations,
                ctf_volume,
                image_shape,
                volume_shape,
                disc_type,
                True,
                half_volume,
                max_r,
                True,
            )
    return y_volume, ctf_volume


def _accumulate_adjoint_block_chunked(
    flat_block,
    flat_rotations,
    volume,
    *,
    window_indices=None,
    use_windowed_adjoint: bool,
    image_shape,
    volume_shape,
    disc_type,
    half_image: bool,
    half_volume: bool,
    max_r,
    relion_x_half: bool,
    max_block_bytes: int,
    log_label: str,
):
    """Accumulate adjoint-slice rows in capped chunks for pathological tails."""

    if flat_block is None:
        return volume
    n_rows = int(flat_block.shape[0])
    if n_rows == 0:
        return volume
    max_rows = _adjoint_block_chunk_rows(flat_block, max_block_bytes=max_block_bytes)
    if n_rows <= max_rows:
        if use_windowed_adjoint:
            return _adjoint_slice_volume_windowed(
                flat_block,
                window_indices,
                flat_rotations,
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
                max_r,
                relion_x_half,
            )
        return _adjoint_slice_volume_half(
            flat_block,
            flat_rotations,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    log_key = (str(log_label), n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _adjoint_block_chunk_log_keys:
        _adjoint_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 adjoint block chunking: %s rows=%d pixels=%d max_rows=%d chunks=%d max_block_bytes=%.2f GiB",
            log_label,
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            float(max_block_bytes) / float(1024**3),
        )

    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        if use_windowed_adjoint:
            volume = _adjoint_slice_volume_windowed(
                flat_block[start:stop],
                window_indices,
                flat_rotations[start:stop],
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
                max_r,
                relion_x_half,
            )
        else:
            volume = _adjoint_slice_volume_half(
                flat_block[start:stop],
                flat_rotations[start:stop],
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
            )
    return volume


def _bucket_row_bytes(values) -> int:
    if values is None or len(values.shape) == 0:
        return 1
    n_pixels = int(values.shape[-1]) if len(values.shape) > 1 else 1
    return max(1, n_pixels * _dtype_itemsize(values.dtype))


def _active_flat_gather_chunk_rows(values, ctf_values, flat_rotations, *, max_block_bytes: int | None) -> int:
    if max_block_bytes is None:
        return 2**62
    row_bytes = _bucket_row_bytes(values) + _bucket_row_bytes(ctf_values)
    if flat_rotations is not None and len(flat_rotations.shape) > 1:
        rotation_items = int(np.prod(tuple(int(dim) for dim in flat_rotations.shape[1:])))
        row_bytes += max(1, rotation_items * _dtype_itemsize(flat_rotations.dtype))
    return max(1, int(max_block_bytes) // max(1, row_bytes))


def _accumulate_active_flat_rows_adjoint_chunked(
    values,
    ctf_values,
    flat_rotations,
    active_indices,
    active_mask,
    y_volume,
    ctf_volume,
    *,
    window_indices=None,
    use_windowed_adjoint: bool,
    image_shape,
    volume_shape,
    disc_type,
    half_image: bool,
    half_volume: bool,
    max_r,
    relion_x_half: bool,
    max_block_bytes: int,
    log_label_prefix: str,
):
    """Gather active M-step rows in bounded chunks before adjoint accumulation."""

    n_rows = int(active_indices.size)
    if n_rows == 0:
        return y_volume, ctf_volume
    max_rows = _active_flat_gather_chunk_rows(
        values,
        ctf_values,
        flat_rotations,
        max_block_bytes=max_block_bytes,
    )
    if n_rows > max_rows:
        n_pixels = int(values.shape[-1]) if len(values.shape) > 1 else 1
        n_chunks = (n_rows + max_rows - 1) // max_rows
        log_key = (str(log_label_prefix), n_rows, n_pixels, max_rows, int(max_block_bytes or 0))
        if log_key not in _active_flat_gather_chunk_log_keys:
            _active_flat_gather_chunk_log_keys.add(log_key)
            logger.info(
                "Sparse pass-2 active flat-row gather chunking: %s rows=%d pixels=%d max_rows=%d "
                "chunks=%d max_block_bytes=%.2f GiB",
                log_label_prefix,
                n_rows,
                n_pixels,
                max_rows,
                n_chunks,
                int(max_block_bytes or 0) / float(1024**3),
            )

    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        chunk_indices = active_indices[start:stop]
        chunk_mask = None if active_mask is None else active_mask[start:stop]
        flat_summed, active_flat_rotations = _select_active_flat_rows(
            values,
            flat_rotations,
            chunk_indices,
            chunk_mask,
        )
        flat_ctf_probs = _select_active_flat_values(
            ctf_values,
            chunk_indices,
            chunk_mask,
        )
        y_volume = _accumulate_adjoint_block_chunked(
            flat_summed,
            active_flat_rotations,
            y_volume,
            window_indices=window_indices,
            use_windowed_adjoint=use_windowed_adjoint,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            half_image=half_image,
            half_volume=half_volume,
            max_r=max_r,
            relion_x_half=relion_x_half,
            max_block_bytes=max_block_bytes,
            log_label=f"{log_label_prefix}-y",
        )
        ctf_volume = _accumulate_adjoint_block_chunked(
            flat_ctf_probs,
            active_flat_rotations,
            ctf_volume,
            window_indices=window_indices,
            use_windowed_adjoint=use_windowed_adjoint,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            half_image=half_image,
            half_volume=half_volume,
            max_r=max_r,
            relion_x_half=relion_x_half,
            max_block_bytes=max_block_bytes,
            log_label=f"{log_label_prefix}-ctf",
        )
    return y_volume, ctf_volume


def _projection_gather_bytes_per_rotation_row(
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    include_recon_noise: bool,
) -> int:
    complex_bytes = _dtype_itemsize(projection_complex_dtype)
    real_dtype = _complex_counterpart_real_dtype(projection_complex_dtype)
    row_bytes = int(n_score_pixels) * complex_bytes
    if include_recon_noise:
        row_bytes += int(n_recon_pixels) * (complex_bytes + _dtype_itemsize(real_dtype))
    return max(1, int(row_bytes))


def _projection_rotation_chunk_size(
    *,
    batch_size: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    include_recon_noise: bool,
    max_gather_bytes: int | None,
    max_projected_rotations: int | None,
) -> int | None:
    """Return a per-image rotation chunk cap for bounded projection gathers."""

    if max_gather_bytes is None:
        return max_projected_rotations
    max_gather_bytes = int(max_gather_bytes)
    if max_gather_bytes <= 0:
        return max_projected_rotations
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_recon_noise=include_recon_noise,
    )
    max_flat_rows = max(1, max_gather_bytes // row_bytes)
    max_rows = max(1, max_flat_rows // max(1, int(batch_size)))
    if max_projected_rotations is not None:
        max_rows = min(max_rows, max(1, int(max_projected_rotations)))
    return max_rows


def _split_compact_pair_buckets_by_projection_gather_budget(
    compact_buckets,
    per_image_inputs_by_class,
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    max_gather_bytes: int | None,
    max_dense_mstep_bytes: int | None = None,
    n_fine_trans: int | None = None,
    prob_dtype=np.float64,
    max_prepare_images_per_microbatch: int | None = None,
    rotation_block_size_for_quantization: int,
):
    """Split compact-pair execution buckets by gather/prep/dense-M-step memory."""

    group_by_rotation_signature = parse_env_flag(
        _SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV,
        default=True,
    )
    if (
        not group_by_rotation_signature
        and max_gather_bytes is None
        and max_prepare_images_per_microbatch is None
        and max_dense_mstep_bytes is None
    ):
        return list(compact_buckets)
    ungrouped_bucket_count = len(compact_buckets)
    original_pair_bucket_max_images: dict[int, int] = {}
    for bucket in compact_buckets:
        pair_bucket_size = int(bucket["pair_bucket_size"])
        original_pair_bucket_max_images[pair_bucket_size] = max(
            original_pair_bucket_max_images.get(pair_bucket_size, 0),
            len(bucket["image_indices"]),
        )
    if group_by_rotation_signature:
        image_indices_by_pair_bucket: dict[int, list[int]] = {}
        for bucket in compact_buckets:
            image_indices_by_pair_bucket.setdefault(
                int(bucket["pair_bucket_size"]),
                [],
            ).extend(
                np.asarray(bucket["image_indices"], dtype=np.int64).tolist(),
            )
        signature_floors_by_pair_bucket: dict[int, int] = {}
        for pair_bucket_size, pair_image_indices in image_indices_by_pair_bucket.items():
            percentile_index = max(
                0,
                int(np.ceil(0.95 * len(pair_image_indices))) - 1,
            )
            signature_floors_by_pair_bucket[pair_bucket_size] = sorted(
                max(
                    _exact_bucket_rotation_size(
                        int(per_image_inputs["oversampled_rots"][image_idx].shape[0]),
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
                for image_idx in pair_image_indices
            )[percentile_index]
        grouped_image_indices: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for bucket in compact_buckets:
            pair_bucket_size = int(bucket["pair_bucket_size"])
            signature_floor = signature_floors_by_pair_bucket[pair_bucket_size]
            for image_idx in np.asarray(bucket["image_indices"], dtype=np.int64).tolist():
                exact_rotation_signature = tuple(
                    _exact_bucket_rotation_size(
                        int(per_image_inputs["oversampled_rots"][image_idx].shape[0]),
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
                outlier_bucket_size = max(exact_rotation_signature)
                rotation_signature = (
                    (outlier_bucket_size,) * len(exact_rotation_signature)
                    if outlier_bucket_size > signature_floor
                    else (signature_floor,) * len(exact_rotation_signature)
                )
                grouped_image_indices.setdefault(
                    (pair_bucket_size, rotation_signature),
                    [],
                ).append(int(image_idx))
        compact_buckets = [
            {
                "pair_bucket_size": pair_bucket_size,
                "image_indices": np.asarray(image_indices, dtype=np.int64),
                "class_bucket_sizes": _rotation_signature,
            }
            for (pair_bucket_size, _rotation_signature), image_indices in grouped_image_indices.items()
        ]
        logger.info(
            "Sparse fused K-class compact-pair rotation-signature grouping: "
            "buckets %d -> %d (default on; set %s=0 to opt out)",
            ungrouped_bucket_count,
            len(compact_buckets),
            _SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV,
        )
    max_gather_bytes = None if max_gather_bytes is None else int(max_gather_bytes)
    if max_gather_bytes is not None and max_gather_bytes <= 0:
        max_gather_bytes = None
    max_dense_mstep_bytes = None if max_dense_mstep_bytes is None else int(max_dense_mstep_bytes)
    if max_dense_mstep_bytes is not None and max_dense_mstep_bytes <= 0:
        max_dense_mstep_bytes = None
    if max_dense_mstep_bytes is not None and n_fine_trans is None:
        raise ValueError("n_fine_trans is required when max_dense_mstep_bytes is set")
    n_fine_trans_int = 0 if n_fine_trans is None else int(n_fine_trans)
    if max_dense_mstep_bytes is not None and n_fine_trans_int <= 0:
        raise ValueError("n_fine_trans must be positive when max_dense_mstep_bytes is set")
    max_prepare_images = (
        None
        if max_prepare_images_per_microbatch is None
        else max(1, int(max_prepare_images_per_microbatch))
    )
    if max_gather_bytes is None and max_prepare_images is None and max_dense_mstep_bytes is None:
        return list(compact_buckets)
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_recon_noise=True,
    )
    prob_item_bytes = _dtype_itemsize(prob_dtype)
    split_buckets = []
    split_bucket_count = 0
    original_bucket_count = len(compact_buckets)
    original_max_images = 0
    split_max_images = 0
    max_dense_bytes_per_image = 0
    for bucket in compact_buckets:
        image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
        original_max_images = max(original_max_images, int(image_indices.size))
        if image_indices.size <= 1:
            split_buckets.append(bucket)
            split_max_images = max(split_max_images, int(image_indices.size))
            continue
        max_images = int(image_indices.size)
        max_images = min(
            max_images,
            original_pair_bucket_max_images[int(bucket["pair_bucket_size"])],
        )
        if max_gather_bytes is not None or max_dense_mstep_bytes is not None:
            if "class_bucket_sizes" in bucket:
                max_class_bucket_size = max(int(value) for value in bucket["class_bucket_sizes"])
            else:
                max_class_bucket_size = max(
                    _compact_bucket_size_for_class(
                        bucket,
                        per_image_inputs,
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
        if max_gather_bytes is not None:
            per_image_bytes = max(1, int(max_class_bucket_size) * row_bytes)
            max_images = min(max_images, max(1, max_gather_bytes // per_image_bytes))
        if max_dense_mstep_bytes is not None:
            dense_bytes_per_image = max(1, int(max_class_bucket_size) * n_fine_trans_int * prob_item_bytes)
            max_dense_bytes_per_image = max(max_dense_bytes_per_image, dense_bytes_per_image)
            max_images = min(max_images, max(1, max_dense_mstep_bytes // dense_bytes_per_image))
        if max_prepare_images is not None:
            max_images = min(max_images, max_prepare_images)
        if image_indices.size <= max_images:
            split_buckets.append(bucket)
            split_max_images = max(split_max_images, int(image_indices.size))
            continue
        split_bucket_count += 1
        for start in range(0, image_indices.size, max_images):
            chunk = image_indices[start : start + max_images]
            chunk_bucket = dict(bucket)
            chunk_bucket["image_indices"] = np.asarray(chunk, dtype=np.int64)
            split_buckets.append(chunk_bucket)
            split_max_images = max(split_max_images, int(chunk.size))
    if split_bucket_count:
        logger.info(
            "Sparse fused K-class compact-pair execution split: buckets %d -> %d; "
            "split_source_buckets=%d, max_images %d -> %d, max_gather_bytes=%s, "
            "row_bytes=%d, max_dense_mstep_bytes=%s, dense_prob_item_bytes=%d, "
            "max_dense_mstep_bytes_per_image=%s, max_prepare_images=%s",
            original_bucket_count,
            len(split_buckets),
            split_bucket_count,
            original_max_images,
            split_max_images,
            "unset" if max_gather_bytes is None else f"{max_gather_bytes / float(1024**3):.2f} GiB",
            row_bytes,
            "unset" if max_dense_mstep_bytes is None else f"{max_dense_mstep_bytes / float(1024**3):.2f} GiB",
            prob_item_bytes,
            "unset" if max_dense_mstep_bytes is None else f"{max_dense_bytes_per_image / float(1024**3):.2f} GiB",
            "unset" if max_prepare_images is None else str(max_prepare_images),
        )
    return split_buckets


# ---------------------------------------------------------------------------
# Scoring + normalization (per-bucket, supports (B, R, T) mask)
# ---------------------------------------------------------------------------


_RELION_CUDA_FINE_REF3D_BLOCK_SIZE = 256


@jax.jit
def _score_pass2_bucket_gaussian_algebraic_components(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
):
    """Return historical algebraic Gaussian scores and their pre-prior terms.

    The extra output is used only by scoped BPref diagnostics. Production exact
    RELION scoring uses the direct ``diff2`` tree below.
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
    preprior_scores = cross - proj_norm[:, :, None]
    scores = preprior_scores + rotation_log_prior[:, :, None] + translation_log_prior[:, None, :]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    preprior_scores = jnp.where(candidate_mask & jnp.isfinite(preprior_scores), preprior_scores, -jnp.inf)
    return scores, preprior_scores


@jax.jit
def _score_pass2_bucket_gaussian_algebraic(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
):
    """Historical algebraic Gaussian scorer used outside exact CUDA mode."""

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
    scores = (
        cross
        - proj_norm[:, :, None]
        + rotation_log_prior[:, :, None]
        + translation_log_prior[:, None, :]
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_bucket_gaussian_algebraic_single_cached(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
):
    """Single-image cached variant of the historical algebraic scorer."""

    weights = corr_img_score * half_weights
    cross = jnp.einsum(
        "tn,n,rn->rt",
        jnp.conj(shifted_corrected),
        weights,
        proj_half,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    proj_norm = 0.5 * jnp.einsum(
        "n,rn->r",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = (
        cross
        - proj_norm[:, None]
        + rotation_log_prior[:, None]
        + translation_log_prior[None, :]
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


def _relion_cuda_fine_reduce_lanes(lanes):
    """Reduce the 256 shared-memory lanes used by RELION CUDA REF3D fine diff2.

    Preserves the caller's dtype (promoted against float32 as a floor) instead
    of forcing float32, so a caller that has already promoted its operands to
    float64 (to match a ``ACC_DOUBLE_PRECISION`` RELION oracle) is not
    silently narrowed back down here.
    """

    lanes = jnp.asarray(lanes, dtype=jnp.result_type(lanes, jnp.float32))
    if lanes.shape[-1] != _RELION_CUDA_FINE_REF3D_BLOCK_SIZE:
        raise ValueError(
            "RELION CUDA fine reduction needs exactly "
            f"{_RELION_CUDA_FINE_REF3D_BLOCK_SIZE} lanes, got {lanes.shape[-1]}"
        )
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    return lanes[..., 0]


def _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, compact_indices):
    """Map RELION's full current-size packed pixel order to compact score rows."""

    image_size = int(image_shape[0])
    current_size = image_size if current_size is None else int(current_size)
    original_half_width = int(image_shape[1]) // 2 + 1
    current_half_width = current_size // 2 + 1
    compact_indices = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    centered_rows = compact_indices // original_half_width
    columns = compact_indices % original_half_width
    ky = centered_rows - image_size // 2
    fftw_rows = np.where(ky < 0, ky + current_size, ky)
    if np.any(fftw_rows < 0) or np.any(fftw_rows >= current_size):
        raise ValueError("compact score indices contain rows outside the RELION current-size crop")
    if np.any(columns < 0) or np.any(columns >= current_half_width):
        raise ValueError("compact score indices contain columns outside the RELION current-size crop")
    relion_flat_indices = fftw_rows * current_half_width + columns
    if np.unique(relion_flat_indices).size != relion_flat_indices.size:
        raise ValueError("compact score indices do not map uniquely into RELION's current-size layout")
    lookup = np.full(current_size * current_half_width, -1, dtype=np.int32)
    lookup[relion_flat_indices] = np.arange(compact_indices.size, dtype=np.int32)
    return lookup


def _relion_cuda_fine_diff2_sum(
    reference,
    shifted_image,
    pixel_weight,
    relion_full_to_compact=None,
    *,
    use_fused_ffi=False,
):
    """Accumulate direct Gaussian diff2 without materializing ``(..., N)``.

    Pinned RELION CUDA's ``REF3D=true, DATA3D=false`` path uses
    ``D2F_BLOCK_SIZE_REF3D=256`` and ``D2F_CHUNK_REF3D=7``. The chunk controls
    translation batching; thread ``tid`` still accumulates pixels
    ``tid + pass * 256`` sequentially in XFLOAT (float32), followed by the
    shared-memory 128,64,...,1 reduction tree. Keeping only the 256 lanes per
    hypothesis avoids the much larger hypothesis-by-pixel temporary.
    """

    complex_dtype = jnp.result_type(reference, shifted_image, jnp.complex64)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_image = jnp.asarray(shifted_image, dtype=complex_dtype)
    pixel_weight = jnp.asarray(pixel_weight, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if shifted_image.shape[-1] != n_values or pixel_weight.shape[-1] != n_values:
        raise ValueError(
            "RELION CUDA fine diff2 operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_image.shape[-1]}, "
            f"weight={pixel_weight.shape[-1]}"
        )
    output_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_image.shape[:-1], pixel_weight.shape[:-1]
    )
    if n_values == 0:
        return jnp.zeros(output_shape, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )
    if bool(use_fused_ffi) or parse_env_flag(
        _RELION_FINE_DIFF2_FUSED_FFI_ENV,
        default=False,
    ):
        from recovar import cuda_backproject

        if reference.ndim == 4:
            if (
                shifted_image.ndim != 4
                or pixel_weight.ndim != 4
                or reference.shape[2] != 1
                or shifted_image.shape[1] != 1
                or pixel_weight.shape[1:3] != (1, 1)
                or reference.shape[0] != shifted_image.shape[0]
                or reference.shape[0] != pixel_weight.shape[0]
            ):
                raise ValueError(
                    "fused rectangular fine diff2 received unsupported broadcast shapes: "
                    f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
                )
            fine_diff2_rectangular = (
                cuda_backproject.relion_fine_diff2_rectangular_f64
                if real_dtype == jnp.float64
                else cuda_backproject.relion_fine_diff2_rectangular_f32
            )
            return fine_diff2_rectangular(
                reference[:, :, 0, :],
                shifted_image[:, 0, :, :],
                pixel_weight[:, 0, 0, :],
                relion_full_to_compact,
            )
        if reference.ndim == 3 and shifted_image.ndim == 3 and pixel_weight.ndim == 3:
            if (
                reference.shape == shifted_image.shape
                and pixel_weight.shape[1] == 1
                and reference.shape[0] == pixel_weight.shape[0]
            ):
                fine_diff2_pairs = (
                    cuda_backproject.relion_fine_diff2_pairs_f64
                    if real_dtype == jnp.float64
                    else cuda_backproject.relion_fine_diff2_pairs_f32
                )
                return fine_diff2_pairs(
                    reference,
                    shifted_image,
                    pixel_weight[:, 0, :],
                    relion_full_to_compact,
                )
            if (
                reference.shape[1] == 1
                and shifted_image.shape[0] == 1
                and pixel_weight.shape[:2] == (1, 1)
            ):
                fine_diff2_rectangular = (
                    cuda_backproject.relion_fine_diff2_rectangular_f64
                    if real_dtype == jnp.float64
                    else cuda_backproject.relion_fine_diff2_rectangular_f32
                )
                return fine_diff2_rectangular(
                    reference[:, 0, :][None, :, :],
                    shifted_image[0, :, :][None, :, :],
                    pixel_weight[0, 0, :][None, :],
                    relion_full_to_compact,
                )[0]
        raise ValueError(
            "fused fine diff2 received unsupported operand ranks/shapes: "
            f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
        )
    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    lanes = jnp.zeros(output_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        img_pass = jnp.take(shifted_image, safe_rows, axis=-1)
        weight_pass = jnp.take(pixel_weight, safe_rows, axis=-1)
        diff_real = ref_pass.real - img_pass.real
        diff_imag = ref_pass.imag - img_pass.imag
        terms = (
            (diff_real * diff_real + diff_imag * diff_imag)
            * jnp.asarray(0.5, dtype=real_dtype)
            * weight_pass
        )
        terms = jnp.where(valid_pixel, terms, jnp.asarray(0.0, dtype=real_dtype))
        return lane_values + terms

    lanes = jax.lax.fori_loop(0, n_passes, accumulate_pass, lanes)
    return _relion_cuda_fine_reduce_lanes(lanes)


def _relion_cuda_fine_normalized_cc_score(
    reference,
    shifted_score,
    score_weight,
    half_weights,
    relion_full_to_compact=None,
):
    """Reproduce RELION CUDA's 256-lane fine normalized-CC reduction.

    The pinned ``cuda_kernel_diff2_CC_fine<REF3D=true>`` accumulates numerator
    and reference norm over pixels ``tid + pass * 256`` in RELION's XFLOAT,
    then uses the same shared-memory tree as fine Gaussian ``diff2``. XFLOAT is
    float32 in RELION's default accelerated build but float64 whenever
    ``ACC_DOUBLE_PRECISION`` is set (our double-precision oracle build); this
    reduction follows the caller's operand dtype instead of hardcoding
    float32, so it matches whichever precision the RELION oracle actually
    used. RECOVAR stores the score window in centered compact order, so
    ``relion_full_to_compact`` restores RELION's packed current-size FFTW
    pixel order before accumulation.
    """

    complex_dtype = jnp.result_type(reference, shifted_score, jnp.complex64)
    real_dtype = jnp.result_type(score_weight, half_weights, jnp.float32)
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_score = jnp.asarray(shifted_score, dtype=complex_dtype)
    score_weight = jnp.asarray(score_weight, dtype=real_dtype)
    half_weights = jnp.asarray(half_weights, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if (
        shifted_score.shape[-1] != n_values
        or score_weight.shape[-1] != n_values
        or half_weights.shape != (n_values,)
    ):
        raise ValueError(
            "RELION CUDA fine normalized-CC operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_score.shape[-1]}, "
            f"score_weight={score_weight.shape[-1]}, half_weights={half_weights.shape}"
        )
    numerator_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_score.shape[:-1], score_weight.shape[:-1]
    )
    norm_shape = jnp.broadcast_shapes(reference.shape[:-1], score_weight.shape[:-1])
    if n_values == 0:
        return jnp.full(numerator_shape, -jnp.inf, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )

    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    numerator_lanes = jnp.zeros(numerator_shape + (block_size,), dtype=real_dtype)
    norm_lanes = jnp.zeros(norm_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        numerator, norm = lane_values
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        shifted_pass = jnp.take(shifted_score, safe_rows, axis=-1)
        score_weight_pass = jnp.take(score_weight, safe_rows, axis=-1)
        half_weight_pass = jnp.take(half_weights, safe_rows, axis=-1)
        numerator_terms = (
            ref_pass.real * shifted_pass.real + ref_pass.imag * shifted_pass.imag
        ) * score_weight_pass * half_weight_pass
        norm_terms = (
            ref_pass.real * ref_pass.real + ref_pass.imag * ref_pass.imag
        ) * score_weight_pass * half_weight_pass
        zero = jnp.asarray(0.0, dtype=real_dtype)
        numerator_terms = jnp.where(valid_pixel, numerator_terms, zero)
        norm_terms = jnp.where(valid_pixel, norm_terms, zero)
        return numerator + numerator_terms, norm + norm_terms

    numerator_lanes, norm_lanes = jax.lax.fori_loop(
        0,
        n_passes,
        accumulate_pass,
        (numerator_lanes, norm_lanes),
    )
    numerator = _relion_cuda_fine_reduce_lanes(numerator_lanes)
    norm = _relion_cuda_fine_reduce_lanes(norm_lanes)
    return numerator / jnp.sqrt(
        jnp.maximum(norm, jnp.asarray(1e-30, dtype=real_dtype))
    )


def _relion_cuda_fine_pixel_weights(corr_img_score, half_weights):
    """Form RELION XFLOAT pixel weights in the active ACC precision."""

    real_dtype = jnp.result_type(corr_img_score, half_weights, jnp.float32)
    return jnp.asarray(corr_img_score, dtype=real_dtype) * jnp.asarray(
        half_weights, dtype=real_dtype
    )


def _relion_cuda_corr_img_from_rfloat_ctf(
    inverse_noise,
    ctf_rfloat,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form XFLOAT ``corr_img`` after RELION's RFLOAT CTF square.

    The deployed mixed-precision build stores ``Minvsigma2`` and ``corr_img``
    as float32 (XFLOAT), but evaluates the CTF and ``CTF * CTF`` as float64
    (RFLOAT).  The compound multiplication promotes Minvsigma2 to float64 and
    casts the product back to float32 before the optional float32 scale square.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    inverse_noise_rfloat = jnp.asarray(inverse_noise, dtype=output_dtype).astype(jnp.float64)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    ctf_squared_rfloat = jax.lax.optimization_barrier(ctf_rfloat * ctf_rfloat)
    corr_img = jax.lax.optimization_barrier(
        inverse_noise_rfloat * ctf_squared_rfloat
    ).astype(output_dtype)
    if scale is not None:
        scale = jnp.asarray(scale, dtype=output_dtype)
        scale_squared = jax.lax.optimization_barrier(scale * scale)
        corr_img = corr_img * scale_squared
    return corr_img


def _relion_cuda_corr_img_from_native_noise_variance(
    noise_variance,
    ctf_rfloat,
    image_shape,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form score-unit ``corr_img`` with RELION's native-FFT cast order.

    RECOVAR stores the noise variance in its normalized-FFT units, larger than
    RELION's variance by ``N**4``.  Reciprocating that value into float32 and
    then applying the compensating Fourier scale is algebraically correct but
    changes ``Minvsigma2`` by one ULP on real parity fixtures.  RELION first
    reciprocates its native-unit binary64 variance into XFLOAT, forms the
    CTF-square product, and only then does RECOVAR need to convert the completed
    XFLOAT operand back to normalized-FFT score units. ``output_dtype`` selects
    the score dtype of that final conversion (float32 production, float64 for
    double-precision scoring); the RELION-side casts above it are unchanged.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    image_size = int(image_shape[0])
    if tuple(image_shape) != (image_size, image_size):
        raise ValueError(f"RELION corr_img requires a square image, got {image_shape}")
    native_fourier_scale_rfloat = jnp.asarray(image_size**4, dtype=jnp.float64)
    native_variance = (
        jnp.asarray(noise_variance, dtype=jnp.float64)
        / native_fourier_scale_rfloat
    )
    native_inverse_noise = jnp.reciprocal(native_variance).astype(jnp.float32)
    native_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
        native_inverse_noise,
        ctf_rfloat,
        scale,
    )
    # XLA's float32 division may lower to a reciprocal multiply and differs
    # from correctly rounded division by one ULP.  This conversion is not a
    # RELION operation, so perform it in binary64 and cast once to preserve the
    # native XFLOAT operand under RECOVAR's Fourier normalization.
    return (
        native_corr_img.astype(jnp.float64) / native_fourier_scale_rfloat
    ).astype(output_dtype)


def _relion_cuda_pixel_correction_from_rfloat_ctf(
    scale,
    ctf_rfloat,
    *,
    output_dtype=jnp.float32,
):
    """Form RELION's XFLOAT score-image correction from an RFLOAT CTF."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    scale = jnp.asarray(scale, dtype=output_dtype)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    pixel_correction = jax.lax.optimization_barrier(jnp.reciprocal(scale))
    corrected = jax.lax.optimization_barrier(
        pixel_correction.astype(jnp.float64) / ctf_rfloat
    ).astype(output_dtype)
    return jnp.where(jnp.abs(ctf_rfloat) > 1e-8, corrected, pixel_correction)


_RELION_CUDA_POWERCLASS_BLOCK_SIZE = 128


class _RelionPowerClassOperands(NamedTuple):
    """RELION ``powerClass`` operands of one flattened centred rfft image batch."""

    relion_image: jnp.ndarray
    real_dtype: object
    image_height: int
    image_width: int
    half_width: int
    resolution_limit: object
    shell: np.ndarray
    valid: np.ndarray


def _relion_powerclass_resolution_limit(current_size, runtime_current_size, *, image_width):
    """RELION's first high-resolution shell, ``current_size / 2 + 1`` (static or traced)."""

    if runtime_current_size is None:
        if current_size is None:
            current_size = image_width
        return int(current_size) // 2 + 1
    return jnp.asarray(runtime_current_size, dtype=jnp.int32) // 2 + 1


def _relion_powerclass_packed_image(processed_score_half, *, image_shape, dtype=None):
    """Repack centred rfft images into RELION's unshifted ``Faux`` layout.

    RELION's packed rows are ``0, +1, ..., +Nyquist, -Nyquist+1, ..., -1`` and
    its FFT amplitudes are smaller than RECOVAR's by the real-space pixel
    count. ``dtype`` forces the complex working precision (the native kernels
    take complex64); otherwise complex128 inputs keep double precision.
    Returns ``(relion_image, real_dtype, image_height, image_width, half_width)``.
    """

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    if dtype is None:
        processed_score_half = jnp.asarray(processed_score_half)
        complex_dtype = (
            jnp.complex128
            if processed_score_half.dtype == jnp.dtype(jnp.complex128)
            else jnp.complex64
        )
        processed_score_half = processed_score_half.astype(complex_dtype)
    else:
        complex_dtype = dtype
        processed_score_half = jnp.asarray(processed_score_half, dtype=complex_dtype)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=real_dtype)
    return relion_image, real_dtype, image_height, image_width, half_width


def _relion_powerclass_operands(processed_score_half, *, image_shape, current_size, runtime_current_size):
    """RELION ``powerClass`` operands for the JAX reproductions.

    Besides the packed image this supplies the integer shell of every packed
    pixel (CUDA ``__float2int_rn(sqrtf(...))``, or the double variant under
    ``ACC_DOUBLE_PRECISION``) and the kernel's pixel validity: shells 1 to
    ``half_width - 1`` excluding the redundant negative-row DC column.
    """

    relion_image, real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
        processed_score_half, image_shape=image_shape
    )
    resolution_limit = _relion_powerclass_resolution_limit(
        current_size, runtime_current_size, image_width=image_width
    )
    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    shell_real_dtype = np.float64 if real_dtype == jnp.float64 else np.float32
    shell = np.rint(np.sqrt(radius_squared.astype(shell_real_dtype))).astype(np.int32)
    valid = (
        (shell > 0)
        & (shell < half_width)
        & ~((columns == 0) & (signed_rows < 0))
    ).reshape(-1)
    return _RelionPowerClassOperands(
        relion_image, real_dtype, image_height, image_width, half_width, resolution_limit, shell, valid
    )


def _relion_powerclass_native_spectrum_highres(processed_score_half, *, image_shape, current_size, runtime_current_size):
    """Run the native CUDA ``powerClass`` atomics on complex64 packed images.

    Returns the per-image spectrum with the block-tree ``highres_Xi2`` scalar
    appended, together with ``(image_height, image_width, half_width)``.
    """

    from recovar import cuda_backproject

    relion_image, _real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
        processed_score_half, image_shape=image_shape, dtype=jnp.complex64
    )
    relion_image = relion_image.astype(jnp.complex64)
    if runtime_current_size is None:
        spectrum_and_highres = cuda_backproject.relion_powerclass_spectrum_highres_f32(
            relion_image,
            xdim=half_width,
            ydim=image_height,
            resolution_limit=int(current_size) // 2 + 1,
        )
    else:
        spectrum_and_highres = (
            cuda_backproject.relion_powerclass_spectrum_highres_runtime_f32(
                relion_image,
                jnp.asarray(runtime_current_size, dtype=jnp.int32) // 2 + 1,
                xdim=half_width,
                ydim=image_height,
            )
        )
    return spectrum_and_highres, image_height, image_width, half_width


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_xi2_half(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the class-power high-resolution image tail used by fine diff2.

    RELION's CUDA ``powerClass`` kernel (``cuda_kernels/helper.cuh``) bins the
    unshifted, unnormalised ``Faux`` image in XFLOAT, reduces each contiguous
    128-pixel block with a shared-memory tree, and atomically accumulates bins
    at or above ``current_size / 2 + 1``. XFLOAT is float32 normally and
    float64 under ``ACC_DOUBLE_PRECISION``. ``diff2_fine`` then adds half of
    that scalar to every fine-search hypothesis (``cuda_kernels/diff2.cuh``).

    RECOVAR stores the y axis centred and its FFT amplitudes are larger by the
    real-space pixel count. Convert both conventions before reproducing the
    matching XFLOAT power and block reduction. The final cross-block
    accumulation uses ascending block order; RELION's atomic arrival order is
    not specified, so its last bit may vary between launches while the
    per-block arithmetic is fixed.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    real_dtype = operands.real_dtype
    valid = jnp.asarray(operands.valid) & (
        jnp.asarray(operands.shell.reshape(-1), dtype=jnp.int32) >= operands.resolution_limit
    )

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    power = jnp.where(jnp.asarray(valid)[None, :], power, jnp.asarray(0.0, dtype=real_dtype))

    block_size = _RELION_CUDA_POWERCLASS_BLOCK_SIZE
    n_blocks = (power.shape[-1] + block_size - 1) // block_size
    power = jnp.pad(power, ((0, 0), (0, n_blocks * block_size - power.shape[-1])))
    block_lanes = power.reshape((power.shape[0], n_blocks, block_size))
    for width in (64, 32, 16, 8, 4, 2, 1):
        block_lanes = block_lanes[..., :width] + block_lanes[..., width : 2 * width]
        block_lanes = jax.lax.optimization_barrier(block_lanes)
    block_sums = block_lanes[..., 0]

    def add_block(block_index, total):
        total = total + block_sums[:, block_index]
        return jax.lax.optimization_barrier(total)

    highres_xi2 = jax.lax.fori_loop(
        0,
        n_blocks,
        add_block,
        jnp.zeros((relion_image.shape[0],), dtype=real_dtype),
    )
    return highres_xi2 * jnp.asarray(0.5, dtype=real_dtype)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_xi2_half_atomic(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Run the native CUDA powerClass atomics used by exact fine scoring."""

    spectrum_and_highres, _image_height, _image_width, _half_width = _relion_powerclass_native_spectrum_highres(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    return spectrum_and_highres[:, -1] * jnp.asarray(0.5, dtype=jnp.float32)


def _relion_powerclass_highres_xi2_half_to_norm_units(highres_xi2_half, image_shape):
    """Convert RELION's half-Xi2 FFT units to RECOVAR norm N^4 units."""

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    highres = jnp.asarray(highres_xi2_half)
    highres = highres * jnp.asarray(2.0, dtype=highres.dtype)
    highres = jax.lax.optimization_barrier(highres)
    return highres * jnp.asarray((image_height * image_width) ** 2, dtype=highres.dtype)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Return source-faithful powerClass high-shell power in RECOVAR N^4 units."""

    return _relion_powerclass_highres_xi2_half_to_norm_units(
        _relion_cuda_powerclass_highres_xi2_half(
            processed_score_half,
            image_shape=image_shape,
            current_size=current_size,
            runtime_current_size=runtime_current_size,
        ),
        image_shape,
    )


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the high-shell norm term from RELION's power spectrum.

    RELION's ``powerClass`` kernel produces two independently reduced values:
    a block-tree ``highres_Xi2`` scalar used by fine scoring, and an
    atomically binned shell spectrum.  Norm correction consumes the latter,
    summing its high shells sequentially in host RFLOAT.  These reductions are
    numerically distinct, so the fine-score scalar cannot be reused here.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    image_height, image_width, half_width = operands.image_height, operands.image_width, operands.half_width
    resolution_limit = operands.resolution_limit
    shell = np.where(operands.valid, operands.shell.reshape(-1), half_width).astype(np.int32)

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    spectrum = jax.vmap(
        lambda row: bin_shell_values_jax(row, jnp.asarray(shell), half_width)
    )(power)

    # RELION copies the float32 spectrum to the host and adds the selected
    # shells into an RFLOAT accumulator in increasing shell order.
    def add_shell(shell_index, total):
        return total + spectrum[:, shell_index].astype(jnp.float64)

    high_shell = jax.lax.fori_loop(
        resolution_limit,
        half_width,
        add_shell,
        jnp.zeros((relion_image.shape[0],), dtype=jnp.float64),
    )
    return high_shell * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float64)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Return RELION's atomically binned per-image power spectrum in N^4 units."""

    spectrum_and_highres, image_height, image_width, half_width = _relion_powerclass_native_spectrum_highres(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    return spectrum_and_highres[:, :half_width] * jnp.asarray(
        (image_height * image_width) ** 2,
        dtype=jnp.float32,
    )


def _relion_powerclass_noise_terms(
    processed_score_half_for_noise,
    *,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
):
    """RELION ``powerClass`` terms one sparse pass-2 batch needs.

    Exact fine Gaussian scoring adds half of the block-tree ``highres_Xi2``
    to every hypothesis, and norm correction at a current size consumes the
    high-shell power: the atomically binned spectrum in source-faithful mode,
    otherwise the same ``highres_Xi2`` converted to RECOVAR's N^4 units
    (``ml_optimiser.cpp`` ``storeWeightedSums``). Returns
    ``(highres_xi2_half, norm_high_shell)`` with ``None`` for terms the batch
    does not need.
    """

    relion_highres_xi2_half = None
    if use_exact_relion_gaussian or (accumulate_noise and current_size is not None):
        relion_highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
        )
    if accumulate_noise and current_size is not None and relion_highres_xi2_half is not None:
        if source_faithful_spectrum_norm:
            relion_norm_high_shell = _relion_cuda_powerclass_spectrum_highres_norm_units(
                processed_score_half_for_noise,
                image_shape=image_shape,
                current_size=current_size,
            )
        else:
            relion_norm_high_shell = _relion_powerclass_highres_xi2_half_to_norm_units(
                relion_highres_xi2_half,
                image_shape,
            )
    else:
        relion_norm_high_shell = None
    return relion_highres_xi2_half, relion_norm_high_shell


def _relion_cuda_fine_diff2_min(diff2, candidate_mask):
    """Return one finite XFLOAT minimum per image over a raw diff2 tensor."""

    minimum = _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask)
    return jnp.where(
        jnp.isfinite(minimum),
        minimum,
        jnp.asarray(0.0, dtype=minimum.dtype),
    )


def _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask):
    """Reduce one partition, retaining ``+inf`` for all-invalid images."""

    diff2 = jnp.asarray(diff2)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    if diff2.shape != candidate_mask.shape:
        raise ValueError(
            "RELION raw diff2 and candidate mask shapes must match: "
            f"diff2={diff2.shape}, mask={candidate_mask.shape}"
        )
    if diff2.ndim < 2:
        raise ValueError(f"RELION raw diff2 needs a leading image axis, got {diff2.shape}")
    valid = candidate_mask & jnp.isfinite(diff2)
    reduction_axes = tuple(range(1, diff2.ndim))
    return jnp.min(jnp.where(valid, diff2, jnp.inf), axis=reduction_axes)


def _relion_cuda_fine_global_diff2_min(raw_diff2_by_partition, masks_by_partition):
    """Return the common per-image minimum spanning chunks and/or classes."""

    if len(raw_diff2_by_partition) != len(masks_by_partition):
        raise ValueError("RELION raw diff2 partitions and masks must have equal lengths")
    if not raw_diff2_by_partition:
        raise ValueError("RELION common-min reduction needs at least one partition")
    partition_minima = []
    for raw_diff2, mask in zip(raw_diff2_by_partition, masks_by_partition, strict=True):
        host_staged_partition = isinstance(raw_diff2, np.ndarray)
        raw_diff2_device = jnp.asarray(raw_diff2)
        mask_device = jnp.asarray(mask, dtype=bool)
        partition_minimum = _relion_cuda_fine_partition_diff2_min_or_inf(
            raw_diff2_device,
            mask_device,
        )
        if host_staged_partition:
            # K-class staging deliberately serializes each D2H-staged class
            # partition back through the device. Synchronize the tiny reduced
            # result before releasing the raw upload so successive classes
            # cannot become simultaneously resident through async dispatch.
            partition_minimum = jax.block_until_ready(partition_minimum)
        partition_minima.append(partition_minimum)
        del raw_diff2_device
    common_min = jnp.min(jnp.stack(partition_minima, axis=0), axis=0)
    return jnp.where(
        jnp.isfinite(common_min),
        common_min,
        jnp.asarray(0.0, dtype=common_min.dtype),
    )


def _relion_cuda_fine_log_evidence_offset(min_diff2):
    """Undo RELION's common-min score centering for absolute log evidence."""

    return -jnp.asarray(min_diff2)


def _relion_cuda_fine_diff2_to_scores(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    *,
    min_diff2=None,
):
    """Apply RELION's XFLOAT fine diff2-to-log-weight conversion order.

    RELION first finds one common minimum over the full valid fine candidate
    set for each image. Its CUDA conversion kernel then evaluates, in XFLOAT,
    ``((orientation_log_prior + translation_log_prior) + min_diff2) - diff2``.
    The common-min term cancels algebraically in normalized probabilities but
    its placement changes float32 tie-breaking at diff2 magnitudes around 1e3.

    ``min_diff2`` may be supplied by a caller that splits one image's full
    candidate set across score chunks or classes. Otherwise it is computed
    over the candidate set represented by ``diff2``. K-class callers must
    therefore supply an external minimum spanning every class; a per-class
    call is not a claim of K-class bit parity.
    """

    diff2 = jnp.asarray(diff2)
    real_dtype = diff2.dtype
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=real_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=real_dtype)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    valid = candidate_mask & jnp.isfinite(diff2)
    if min_diff2 is None:
        local_min = _relion_cuda_fine_diff2_min(diff2, candidate_mask)
    else:
        local_min = jnp.asarray(min_diff2, dtype=real_dtype)
    has_valid = jnp.any(valid, axis=tuple(range(1, diff2.ndim)))
    local_min = jnp.where(has_valid, local_min, jnp.asarray(0.0, dtype=real_dtype))
    min_shape = (diff2.shape[0],) + (1,) * (diff2.ndim - 1)
    # RELION's exponentiation kernel rejects candidates below the supplied
    # global minimum. This is normally impossible for a self-consistent
    # partition, but is observable at cross-partition float32 boundaries.
    valid = valid & (diff2 >= local_min.reshape(min_shape))

    scores = rotation_log_prior + translation_log_prior
    scores = jax.lax.optimization_barrier(scores)
    scores = scores + local_min.reshape(min_shape)
    scores = jax.lax.optimization_barrier(scores)
    scores = scores - diff2
    scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)
    return scores


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Return positive float32 RELION fine-pass costs without priors or centering."""

    weights = _relion_cuda_fine_pixel_weights(
        corr_img_score, jnp.asarray(half_weights)[None, :]
    )
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, :, None, :],
        shifted_corrected[:, None, :, :],
        weights[:, None, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)[:, None, None]
    return diff2


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2_from_raw(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    min_diff2,
):
    """Convert retained raw costs with the same jitted exact score arithmetic."""

    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # (B,) optional external common minimum across chunks/classes
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """RELION GPU-style direct ``diff2`` scoring for pass-2 diagnostics.

    RELION's CUDA fine-search kernel first corrects the image by the same
    scalar factors carried by the projection-norm weight, then accumulates a
    direct ``|Fref - Fimg_corrected_shift|^2 * corr_img`` form.  This is
    algebraically equivalent to the dense cross-minus-norm expression but has
    different float32 rounding. The positive diff2 values are converted with
    RELION's common-min and prior-addition order below.

    Extremely small CTF/noise combinations can still overflow the direct form
    on long 256px runs.  Treat non-finite candidates as impossible hypotheses
    rather than letting NaNs enter posterior and noise accumulators.
    """

    score_dtype = (
        jnp.float64
        if jnp.result_type(shifted_corrected, corr_img_score) == jnp.complex128
        or jnp.asarray(corr_img_score).dtype == jnp.float64
        else jnp.float32
    )
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
    shifted_corrected,  # (T, N) complex
    corr_img_score,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # scalar float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Single-image cached positive-cost variant without priors or centering."""

    weights = _relion_cuda_fine_pixel_weights(corr_img_score, half_weights)
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, None, :],
        shifted_corrected[None, :, :],
        weights[None, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)
    return diff2


@jax.jit
def _score_pass2_bucket_relion_gpu_normalized_cc(
    shifted_score,  # (B, T, N) complex, RELION-corrected image after shift
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (B, R, T) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """RELION iter-1 normalized-CC scoring for sparse pass-2 buckets."""

    scores = _relion_cuda_fine_normalized_cc_score(
        proj_half[:, :, None, :],
        shifted_score[:, None, :, :],
        score_weight[:, None, None, :],
        half_weights,
        relion_full_to_compact,
    )
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
    """Historical algebraic normalized-CC scorer retained outside K=1."""

    cross_products = (
        proj_half[:, :, None, :].real * shifted_score[:, None, :, :].real
        + proj_half[:, :, None, :].imag * shifted_score[:, None, :, :].imag
    ) * jnp.asarray(half_weights, dtype=proj_half.real.dtype)[None, None, None, :]
    # Sum in the operands' own precision (natural promotion) rather than
    # forcing float32 -- this term is the CC numerator and must match the
    # denominator's precision (below, via Precision.HIGHEST) or a genuine
    # near-tie candidate ranking can flip relative to RELION's RFLOAT/XFLOAT
    # arithmetic. See docs/math/relion_parity_agent_notes.md.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    ) * half_weights[None, None, :]
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
def _score_pass2_bucket_normalized_cc_single_cached(
    shifted_score,  # (T, N) complex
    score_weight,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (R, T) bool
):
    """Historical cached normalized-CC scorer retained outside K=1."""

    cross_products = (
        proj_half[:, None, :].real * shifted_score[None, :, :].real
        + proj_half[:, None, :].imag * shifted_score[None, :, :].imag
    ) * jnp.asarray(half_weights, dtype=proj_half.real.dtype)[None, None, :]
    # See _score_pass2_bucket_normalized_cc: sum in the operands' own
    # precision instead of forcing float32, to match the einsum denominator's
    # precision below and avoid spurious near-tie ranking flips.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    ) * half_weights[None, :]
    norms = jnp.einsum(
        "n,rn->r",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom[:, None]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_pairs_gaussian_algebraic(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    pair_rotation_log_prior,
    translation_log_prior,
    local_rotation_row,
    translation_idx,
    pair_mask,
):
    """Compact-pair variant of the historical algebraic Gaussian scorer."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    shifted_pair = shifted_corrected[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    weights = corr_img_score * half_weights[None, :]
    cross = jnp.einsum(
        "bpn,bn,bpn->bp",
        jnp.conj(shifted_pair),
        weights,
        proj_pair,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_pair.real * proj_pair.real + proj_pair.imag * proj_pair.imag
    proj_norm = 0.5 * jnp.einsum(
        "bn,bpn->bp",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    translation_prior = translation_log_prior[row, safe_translation_idx]
    scores = cross - proj_norm + pair_rotation_log_prior + translation_prior
    scores = jnp.where(pair_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_pairs_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Return positive float32 RELION costs for compact candidate pairs."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)

    shifted_pair = shifted_corrected[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    weights = _relion_cuda_fine_pixel_weights(
        corr_img_score, jnp.asarray(half_weights)[None, :]
    )
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_pair,
        shifted_pair,
        weights[:, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)[:, None]
    return diff2


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_pairs_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    pair_rotation_log_prior,  # (B, P) real
    translation_log_prior,  # (B, T) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # (B,) optional external common minimum across classes
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """RELION GPU-style Gaussian scoring for compact pass-2 pairs."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    score_dtype = jnp.float64 if jnp.asarray(corr_img_score).dtype == jnp.float64 else jnp.float32
    pair_rotation_log_prior = jnp.asarray(pair_rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    trans_prior = jnp.asarray(
        translation_log_prior[row, safe_translation_idx], dtype=score_dtype
    )
    diff2 = _score_pass2_pairs_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        local_rotation_row,
        translation_idx,
        pair_mask,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        pair_rotation_log_prior,
        trans_prior,
        pair_mask,
        min_diff2=min_diff2,
    )


@jax.jit
def _score_pass2_pairs_normalized_cc(
    shifted_score,  # (B, T, N) complex, image * CTF * shift / Xi2
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
):
    """Historical compact-pair normalized-CC scorer retained outside K=1."""

    batch = shifted_score.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    shifted_pair = shifted_score[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    cross_products = (
        proj_pair.real * shifted_pair.real + proj_pair.imag * shifted_pair.imag
    ) * jnp.asarray(half_weights, dtype=proj_pair.real.dtype)[None, None, :]
    # See _score_pass2_bucket_normalized_cc: sum in the operands' own
    # precision instead of forcing float32.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_pair.real * proj_pair.real + proj_pair.imag * proj_pair.imag
    ) * half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,bpn->bp",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom
    scores = jnp.where(pair_mask, scores, -jnp.inf)
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
def _normalize_pass2_pairs_with_log_z(pair_scores, pair_mask, global_log_z):
    """Normalize compact pair scores against a precomputed joint log-Z.

    ``global_log_z`` may include other classes, so the returned pair
    probabilities need not sum to one within this class. Padded pairs and
    non-finite scores always contribute zero probability.
    """

    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(global_log_z)
    safe_log_z = jnp.where(has_finite_score, global_log_z, 0.0)
    pair_probs = jnp.exp(pair_scores - safe_log_z[:, None])
    pair_probs = jnp.where(has_finite_score[:, None] & jnp.isfinite(pair_probs), pair_probs, 0.0)
    best_pair_argmax = jnp.where(has_finite_score, jnp.argmax(pair_scores, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, pair_probs, best_log_score, best_pair_argmax, max_posterior


def _compact_pair_valid_weights_and_indices(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    *,
    n_rotation_rows,
    n_trans,
):
    finite_pair_probs = jnp.isfinite(pair_probs)
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    valid_pair = (
        pair_mask
        & finite_pair_probs
        & (safe_rotation_row >= 0)
        & (safe_rotation_row < int(n_rotation_rows))
        & (safe_translation_idx >= 0)
        & (safe_translation_idx < int(n_trans))
    )
    weights = jnp.where(valid_pair, pair_probs, 0.0)
    safe_rotation_row = jnp.where(valid_pair, safe_rotation_row, 0)
    safe_translation_idx = jnp.where(valid_pair, safe_translation_idx, 0)
    return weights, safe_rotation_row, safe_translation_idx, valid_pair


@partial(jax.jit, static_argnames=("n_rotation_rows", "n_trans"))
def _compact_pair_dense_probs_and_reductions(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    n_rotation_rows,
    n_trans,
):
    """Scatter compact pair probabilities into dense scalar row probabilities."""

    batch, n_pairs = pair_probs.shape
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None], (batch, n_pairs))

    finite_pair_probs = jnp.isfinite(pair_probs)
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    valid_pair = (
        pair_mask
        & finite_pair_probs
        & (safe_rotation_row >= 0)
        & (safe_rotation_row < int(n_rotation_rows))
        & (safe_translation_idx >= 0)
        & (safe_translation_idx < int(n_trans))
    )
    weights = jnp.where(valid_pair, pair_probs, 0.0)
    scatter_rotation_row = jnp.where(valid_pair, safe_rotation_row, 0)
    scatter_translation_idx = jnp.where(valid_pair, safe_translation_idx, 0)

    dense_probs = jnp.zeros((batch, int(n_rotation_rows), int(n_trans)), dtype=weights.dtype)
    dense_probs = dense_probs.at[batch_idx, scatter_rotation_row, scatter_translation_idx].add(weights)
    return dense_probs


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_sparse_weighted_image_and_prob_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    """Reduce compact pairs in pair order, then translations in dense order."""

    batch, n_pairs = pair_probs.shape
    n_trans = shifted_recon_split.shape[1]
    n_pixels = shifted_recon_split.shape[-1]
    weights, safe_rotation_row, safe_translation_idx, valid_pair = _compact_pair_valid_weights_and_indices(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=n_trans,
    )
    batch_idx = jnp.arange(batch, dtype=jnp.int32)
    pair_indices = jnp.arange(n_pairs, dtype=jnp.int32)

    summed_dtype = jnp.result_type(weights, shifted_recon_split)
    summed0 = jnp.zeros((batch, int(n_rotation_rows), n_pixels), dtype=summed_dtype)
    probs_sum_t0 = jnp.zeros((batch, int(n_rotation_rows)), dtype=weights.dtype)
    translation_posterior0 = jnp.zeros((batch, n_trans), dtype=weights.dtype)

    def translation_body(carry, trans_idx):
        summed, probs_sum_t, translation_posterior = carry

        def pair_body(row_probs, pair_idx):
            pair_valid = valid_pair[:, pair_idx] & (safe_translation_idx[:, pair_idx] == trans_idx)
            pair_weights = jnp.where(pair_valid, weights[:, pair_idx], 0.0)
            pair_rows = jnp.where(pair_valid, safe_rotation_row[:, pair_idx], 0)
            row_probs = row_probs.at[batch_idx, pair_rows].add(pair_weights)
            return row_probs, None

        row_probs0 = jnp.zeros((batch, int(n_rotation_rows)), dtype=weights.dtype)
        row_probs, _ = jax.lax.scan(pair_body, row_probs0, pair_indices)
        summed = summed + row_probs[:, :, None] * shifted_recon_split[:, trans_idx, :][:, None, :]
        probs_sum_t = probs_sum_t + row_probs
        translation_posterior = translation_posterior.at[:, trans_idx].set(jnp.sum(row_probs, axis=1))
        return (summed, probs_sum_t, translation_posterior), None

    (summed, probs_sum_t, translation_posterior), _ = jax.lax.scan(
        translation_body,
        (summed0, probs_sum_t0, translation_posterior0),
        jnp.arange(n_trans, dtype=jnp.int32),
    )
    return summed, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows", "relion_x_half"))
def _compact_pair_weighted_rotation_sums_dense(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    relion_x_half=False,
):
    """Accumulate compact-pair M-step stats without forming dense ``(B,R,T)``.

    Returns the dense helper equivalents:
    ``summed = compute_local_weighted_sums(probs, shifted_recon_split)``,
    ``ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)``,
    plus ``probs_sum_t`` and ``translation_posterior``.
    """

    dense_probs = _compact_pair_dense_probs_and_reductions(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    # Use the same dense weighted-sum primitive as the rectangular path after
    # compacting the scalar probabilities. Scattering complex image rows directly
    # changes GPU accumulation order enough to break x-half M-step parity.
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    summed, ctf_probs = compute_local_mstep_sums(
        dense_probs,
        shifted_recon_split,
        ctf2_over_nv_recon,
        relion_x_half=bool(relion_x_half),
        default_probs_sum_t=probs_sum_t,
    )
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Experimental compact-pair M-step reduction without dense ``(B,R,T)``."""

    summed, probs_sum_t, translation_posterior = _compact_pair_sparse_weighted_image_and_prob_sums(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    return summed, ctf_probs, probs_sum_t, translation_posterior


def _compact_pair_weighted_rotation_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
    relion_x_half=False,
):
    use_sequential_relion_reduction = bool(
        relion_x_half and relion_x_half_sequential_translation_reduction_enabled()
    )
    impl = (
        _compact_pair_weighted_rotation_sums_pair_sparse
        if (
            not use_sequential_relion_reduction
            and _compact_pair_pair_sparse_mstep_enabled_for_pass(
                allow_pair_sparse=allow_pair_sparse
            )
        )
        else _compact_pair_weighted_rotation_sums_dense
    )
    kwargs = dict(
        n_rotation_rows=n_rotation_rows,
    )
    if impl is _compact_pair_weighted_rotation_sums_dense:
        kwargs["relion_x_half"] = bool(relion_x_half)
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        ctf2_over_nv_recon,
        **kwargs,
    )


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_image_sums_dense(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    """Accumulate compact-pair image weighted sums only.

    This is used when masked scoring makes image sums differ from the M-step
    reconstruction sums, while the CTF/probability reductions can still be
    reused from the M-step.
    """

    batch, n_pairs = pair_probs.shape
    n_trans = shifted_recon_split.shape[1]
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None], (batch, n_pairs))

    finite_pair_probs = jnp.isfinite(pair_probs)
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    valid_pair = (
        pair_mask
        & finite_pair_probs
        & (safe_rotation_row >= 0)
        & (safe_rotation_row < int(n_rotation_rows))
        & (safe_translation_idx >= 0)
        & (safe_translation_idx < n_trans)
    )
    weights = jnp.where(valid_pair, pair_probs, 0.0)
    scatter_rotation_row = jnp.where(valid_pair, safe_rotation_row, 0)
    scatter_translation_idx = jnp.where(valid_pair, safe_translation_idx, 0)

    dense_probs = jnp.zeros((batch, int(n_rotation_rows), n_trans), dtype=weights.dtype)
    dense_probs = dense_probs.at[batch_idx, scatter_rotation_row, scatter_translation_idx].add(weights)
    return compute_local_weighted_sums(dense_probs, shifted_recon_split)


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_image_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    summed, _probs_sum_t, _translation_posterior = _compact_pair_sparse_weighted_image_and_prob_sums(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )
    return summed


def _compact_pair_weighted_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
):
    impl = (
        _compact_pair_weighted_image_sums_pair_sparse
        if _compact_pair_pair_sparse_mstep_enabled_for_pass(allow_pair_sparse=allow_pair_sparse)
        else _compact_pair_weighted_image_sums_dense
    )
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_legacy(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Accumulate compact-pair M-step sums plus an alternate image sum.

    The default RELION K-class path scores with masked images but reconstructs
    from unmasked images. Build the compact dense probability tensor once and
    reuse it for both image sums, while keeping the CTF/probability reductions
    identical to ``_compact_pair_weighted_rotation_sums``.
    """

    dense_probs = _compact_pair_dense_probs_and_reductions(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    summed = compute_local_weighted_sums(dense_probs, shifted_recon_split)
    summed_image = compute_local_weighted_sums(dense_probs, shifted_image_split)
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


def _compact_pair_weighted_rotation_and_image_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    summed, ctf_probs, probs_sum_t, translation_posterior = _compact_pair_weighted_rotation_sums_pair_sparse(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )
    summed_image = _compact_pair_weighted_image_sums_pair_sparse(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_image_split,
        n_rotation_rows=n_rotation_rows,
    )
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_fused_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Accumulate compact-pair M-step and image sums in one weighted reduction."""

    dense_probs = _compact_pair_dense_probs_and_reductions(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    recon_n_pixels = shifted_recon_split.shape[-1]
    combined_shifted = jnp.concatenate((shifted_recon_split, shifted_image_split), axis=-1)
    combined_summed = compute_local_weighted_sums(dense_probs, combined_shifted)
    summed = combined_summed[..., :recon_n_pixels]
    summed_image = combined_summed[..., recon_n_pixels:]
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_native(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Use one native launch boundary for two independent weighted sums."""

    from recovar.cuda_backproject import dual_weighted_sums_f32

    dense_probs = _compact_pair_dense_probs_and_reductions(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )
    summed, summed_image = dual_weighted_sums_f32(
        dense_probs,
        shifted_recon_split,
        shifted_image_split,
    )
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(
    jax.jit,
    static_argnames=("n_rotation_rows", "shell_count", "batch_size"),
)
def _compact_pair_weighted_sums_and_noise_native(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    proj_for_noise,
    proj_abs2_for_noise,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
):
    """Fuse compact weighted sums with dense noise/norm sufficient statistics."""

    (
        summed,
        summed_image,
        ctf_probs,
        probs_sum_t,
        translation_posterior,
    ) = _compact_pair_weighted_rotation_and_image_sums_native(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        shifted_image_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )
    flat_image_indices = jnp.broadcast_to(
        jnp.arange(int(batch_size), dtype=jnp.int32)[:, None],
        (int(batch_size), int(n_rotation_rows)),
    ).reshape(-1)
    block_noise_shells, block_norm_residual = (
        _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            proj_for_noise.reshape((-1, proj_for_noise.shape[-1])),
            proj_abs2_for_noise.reshape((-1, proj_abs2_for_noise.shape[-1])),
            summed_image.reshape((-1, summed_image.shape[-1])),
            ctf_probs.reshape((-1, ctf_probs.shape[-1])),
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    )
    return (
        summed,
        summed_image,
        ctf_probs,
        probs_sum_t,
        translation_posterior,
        block_noise_shells,
        block_norm_residual,
    )


def _compact_pair_weighted_rotation_and_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
):
    """Accumulate compact-pair M-step sums plus an alternate image sum."""

    if _compact_pair_pair_sparse_mstep_enabled_for_pass(allow_pair_sparse=allow_pair_sparse):
        return _compact_pair_weighted_rotation_and_image_sums_pair_sparse(
            pair_probs,
            local_rotation_row,
            translation_idx,
            pair_mask,
            shifted_recon_split,
            shifted_image_split,
            ctf2_over_nv_recon,
            n_rotation_rows=n_rotation_rows,
        )

    impl = (
        _compact_pair_weighted_rotation_and_image_sums_fused_image_sums
        if parse_env_flag(_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS_ENV, default=True)
        else _compact_pair_weighted_rotation_and_image_sums_legacy
    )
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        shifted_image_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )


def _active_flat_row_indices_from_probs_sum_t(
    probs_sum_t,
    *,
    pad_multiple: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return stable active row indices, padded to reduce active-path JIT churn."""

    probs_sum_t_np = np.asarray(jax.device_get(probs_sum_t))
    total_rows = int(probs_sum_t_np.size)
    active_indices = np.flatnonzero(probs_sum_t_np.reshape(-1) != 0.0).astype(np.int32, copy=False)
    active_count = int(active_indices.size)
    if active_count == 0:
        return active_indices, np.zeros((0,), dtype=np.float32), 0

    pad_multiple = max(1, int(pad_multiple))
    if pad_multiple <= 1:
        return active_indices, np.ones((active_count,), dtype=np.float32), active_count

    padded_count = min(
        total_rows,
        ((active_count + pad_multiple - 1) // pad_multiple) * pad_multiple,
    )
    if padded_count <= active_count:
        return active_indices, np.ones((active_count,), dtype=np.float32), active_count

    padded_indices = np.empty((padded_count,), dtype=np.int32)
    padded_indices[:active_count] = active_indices
    padded_indices[active_count:] = active_indices[0]
    active_mask = np.zeros((padded_count,), dtype=np.float32)
    active_mask[:active_count] = 1.0
    return padded_indices, active_mask, active_count


def _apply_active_row_mask(values, active_mask):
    if active_mask is None:
        return values
    mask = jnp.asarray(active_mask, dtype=jnp.asarray(values).real.dtype)
    while mask.ndim < values.ndim:
        mask = mask[:, None]
    return values * mask


def _select_active_flat_rows(values, flat_rotations, active_indices, active_mask=None):
    """Gather active flattened rows with matching rotations."""

    if active_indices.size == 0:
        return None, None
    active_indices_jax = jnp.asarray(active_indices, dtype=jnp.int32)
    active_values = _gather_active_flat_bucket_rows(values, active_indices_jax)
    active_values = _apply_active_row_mask(active_values, active_mask)
    return active_values, flat_rotations[active_indices_jax]


def _select_active_flat_values(values, active_indices, active_mask=None):
    """Gather active flattened rows."""

    if active_indices.size == 0:
        return None
    active_values = _gather_active_flat_bucket_rows(values, jnp.asarray(active_indices, dtype=jnp.int32))
    return _apply_active_row_mask(active_values, active_mask)


def _gather_active_flat_bucket_rows(values, active_indices):
    """Gather flat row indices without materializing the full flattened bucket."""

    values = jnp.asarray(values)
    if values.ndim >= 3:
        n_rotation_rows = int(values.shape[1])
        image_indices = active_indices // n_rotation_rows
        rotation_row_indices = active_indices - image_indices * n_rotation_rows
        return values[image_indices, rotation_row_indices]
    if values.ndim == 2:
        return values[active_indices]
    return flatten_bucket_rows(values)[active_indices]


def _active_image_indices_for_rotation_rows(active_indices, active_mask, n_rotation_rows: int):
    """Return image ids for active flattened ``(batch, rotation_row)`` rows."""

    if active_indices.size == 0:
        return None
    active_indices_jax = jnp.asarray(active_indices, dtype=jnp.int32)
    image_indices = active_indices_jax // int(n_rotation_rows)
    if active_mask is None:
        return image_indices
    active_mask_jax = jnp.asarray(active_mask, dtype=jnp.int32)
    return jnp.where(active_mask_jax != 0, image_indices, 0)


@partial(
    jax.jit,
    static_argnames=("n_rotation_rows", "shell_count", "batch_size", "use_residual_terms"),
)
def _compute_active_noise_rows_block(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    use_residual_terms: bool,
):
    """Gather one active-row chunk and accumulate its noise/norm residuals."""

    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_mask = jnp.asarray(active_mask)
    row_mask = active_mask.astype(jnp.asarray(summed_masked_noise).real.dtype)
    active_image_indices = jnp.where(active_mask.astype(jnp.int32) != 0, active_indices // int(n_rotation_rows), 0)

    def gather(values):
        flat_values = values.reshape((values.shape[0] * values.shape[1], values.shape[-1]))
        gathered = flat_values[active_indices]
        return gathered * row_mask[:, None]

    flat_proj_for_noise = gather(proj_for_noise)
    flat_proj_abs2_for_noise = gather(proj_abs2_for_noise)
    flat_summed_masked_noise = gather(summed_masked_noise)
    flat_ctf_probs_for_noise = gather(ctf_probs_for_noise)
    if use_residual_terms:
        return _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            flat_proj_for_noise,
            flat_proj_abs2_for_noise,
            flat_summed_masked_noise,
            flat_ctf_probs_for_noise,
            noise_variance_half,
            shell_indices,
            active_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    return _compute_noise_block_and_norm_residual_from_flat_rows(
        flat_proj_for_noise,
        flat_proj_abs2_for_noise,
        flat_summed_masked_noise,
        flat_ctf_probs_for_noise,
        noise_variance_half,
        shell_indices,
        active_image_indices,
        shell_count=int(shell_count),
        batch_size=int(batch_size),
    )


def _compute_active_noise_rows_chunked(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    max_block_bytes: int | None,
):
    """Gather compact active noise rows in row chunks before accumulation."""

    n_rows = int(active_indices.size)
    accumulator_dtype = jnp.result_type(
        proj_abs2_for_noise.dtype,
        ctf_probs_for_noise.dtype,
        noise_variance_half.dtype,
    )
    if n_rows <= 0:
        return (
            jnp.zeros(int(shell_count), dtype=accumulator_dtype),
            jnp.zeros(int(batch_size), dtype=accumulator_dtype),
        )

    use_residual_terms = parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
    if max_block_bytes is None:
        max_rows = n_rows
    else:
        n_pixels = int(proj_for_noise.shape[-1])
        complex_bytes = max(
            _dtype_itemsize(proj_for_noise.dtype),
            _dtype_itemsize(summed_masked_noise.dtype),
        )
        real_bytes = max(
            _dtype_itemsize(proj_abs2_for_noise.dtype),
            _dtype_itemsize(ctf_probs_for_noise.dtype),
            _dtype_itemsize(noise_variance_half.dtype),
        )
        bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
        max_rows = max(1, int(max_block_bytes) // bytes_per_row)

    if n_rows > max_rows:
        n_pixels = int(proj_for_noise.shape[-1])
        n_chunks = (n_rows + max_rows - 1) // max_rows
        log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes or 0))
        if log_key not in _active_noise_gather_chunk_log_keys:
            _active_noise_gather_chunk_log_keys.add(log_key)
            logger.info(
                "Sparse pass-2 compact active noise gather chunking: rows=%d pixels=%d max_rows=%d "
                "chunks=%d max_block_bytes=%.2f GiB",
                n_rows,
                n_pixels,
                max_rows,
                n_chunks,
                int(max_block_bytes or 0) / float(1024**3),
            )

    noise_total = jnp.zeros(int(shell_count), dtype=accumulator_dtype)
    norm_total = jnp.zeros(int(batch_size), dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, norm_chunk = _compute_active_noise_rows_block(
            proj_for_noise,
            proj_abs2_for_noise,
            summed_masked_noise,
            ctf_probs_for_noise,
            active_indices[start:stop],
            active_mask[start:stop],
            noise_variance_half,
            shell_indices,
            n_rotation_rows=int(n_rotation_rows),
            shell_count=int(shell_count),
            batch_size=int(batch_size),
            use_residual_terms=bool(use_residual_terms),
        )
        noise_total = noise_total + noise_chunk
        norm_total = norm_total + norm_chunk
    return noise_total, norm_total


def _active_row_grouping_shape(active_indices, active_mask, n_images, n_rotation_rows):
    """Return active count, max active rows per image, and grouped dense rows."""

    active_indices = np.asarray(active_indices, dtype=np.int32)
    active_mask = np.asarray(active_mask, dtype=np.float32)
    valid = active_mask != 0.0
    active_count = int(np.count_nonzero(valid))
    if active_count == 0:
        return 0, 1, int(n_images)
    image_indices = active_indices[valid] // int(n_rotation_rows)
    counts = np.bincount(image_indices, minlength=int(n_images))
    active_row_slots = max(1, int(np.max(counts, initial=0)))
    return active_count, active_row_slots, int(n_images) * active_row_slots


def _active_row_grouping_for_canonical_matmul(active_indices, active_mask, n_images, n_rotation_rows):
    """Group flat active rows by image while preserving flat active-row order."""

    active_indices = np.asarray(active_indices, dtype=np.int32)
    active_mask = np.asarray(active_mask, dtype=np.float32)
    image_indices = (active_indices // int(n_rotation_rows)).astype(np.int32, copy=False)
    active_slots = np.zeros(active_indices.shape, dtype=np.int32)
    valid = active_mask != 0.0
    valid_positions = np.flatnonzero(valid).astype(np.int32, copy=False)
    if valid_positions.size == 0:
        return image_indices, active_slots, np.zeros((int(n_images), 1), dtype=np.int32)

    valid_image_indices = image_indices[valid_positions]
    counts = np.bincount(valid_image_indices, minlength=int(n_images))
    active_row_slots = max(1, int(np.max(counts, initial=0)))
    grouped_rotation_rows = np.zeros((int(n_images), active_row_slots), dtype=np.int32)

    order = np.lexsort((valid_positions, valid_image_indices))
    sorted_positions = valid_positions[order]
    sorted_image_indices = valid_image_indices[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_image_indices)) + 1]
    group_lengths = np.diff(np.r_[group_starts, sorted_image_indices.size])
    sorted_slots = np.arange(sorted_image_indices.size, dtype=np.int32) - np.repeat(
        group_starts.astype(np.int32, copy=False),
        group_lengths,
    )
    active_slots[sorted_positions] = sorted_slots
    grouped_rotation_rows[sorted_image_indices, sorted_slots] = (
        active_indices[sorted_positions] % int(n_rotation_rows)
    )
    return image_indices, active_slots, grouped_rotation_rows


def _rectangular_active_prematmul_is_efficient(
    active_indices,
    active_mask,
    *,
    n_images: int,
    n_rotation_rows: int,
    max_grouped_dense_ratio: float,
):
    active_count, active_slots, grouped_rows = _active_row_grouping_shape(
        active_indices,
        active_mask,
        n_images=n_images,
        n_rotation_rows=n_rotation_rows,
    )
    dense_rows = int(n_images) * int(n_rotation_rows)
    grouped_dense_ratio = float(grouped_rows) / float(dense_rows) if dense_rows > 0 else 1.0
    use_prematmul = active_count > 0 and grouped_dense_ratio <= float(max_grouped_dense_ratio)
    return use_prematmul, active_count, active_slots, grouped_rows, dense_rows, grouped_dense_ratio


@jax.jit
def _rectangular_active_weighted_image_sums_grouped(
    probs,
    shifted,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute active rows with the same per-image matmul shape as the dense path."""

    grouped_probs = jnp.take_along_axis(
        probs,
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32)[:, :, None],
        axis=1,
    )
    grouped_summed = compute_local_weighted_sums(grouped_probs, shifted)
    active_summed = grouped_summed[
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
    ]
    active_mask = jnp.asarray(active_mask, dtype=active_summed.real.dtype)
    return active_summed * active_mask[:, None]


@jax.jit
def _rectangular_active_weighted_sums(
    probs,
    probs_sum_t,
    shifted,
    ctf2_over_nv,
    active_indices,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute rectangular M-step rows after gathering active ``(image, rotation)`` rows."""

    n_rotation_rows = probs.shape[1]
    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_summed = _rectangular_active_weighted_image_sums_grouped(
        probs,
        shifted,
        active_image_indices,
        active_slots,
        grouped_rotation_rows,
        active_mask,
    )
    active_image_indices = jnp.asarray(active_image_indices, dtype=jnp.int32)
    active_ctf_probs = probs_sum_t.reshape((probs.shape[0] * n_rotation_rows,))[active_indices, None]
    active_ctf_probs = active_ctf_probs * ctf2_over_nv[active_image_indices]
    active_mask = jnp.asarray(active_mask, dtype=active_ctf_probs.real.dtype)
    active_ctf_probs = active_ctf_probs * active_mask[:, None]
    return active_summed, active_ctf_probs


@jax.jit
def _rectangular_active_weighted_image_sums(
    probs,
    shifted,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute active rectangular weighted image rows without recomputing CTF sums."""

    return _rectangular_active_weighted_image_sums_grouped(
        probs,
        shifted,
        active_image_indices,
        active_slots,
        grouped_rotation_rows,
        active_mask,
    )


def _rectangular_active_weighted_sums_or_none(
    probs,
    probs_sum_t,
    shifted,
    ctf2_over_nv,
    flat_rotations,
    active_indices,
    active_mask,
):
    if active_indices.size == 0:
        return None, None, None
    active_image_indices, active_slots, grouped_rotation_rows = _active_row_grouping_for_canonical_matmul(
        active_indices,
        active_mask,
        n_images=probs.shape[0],
        n_rotation_rows=probs.shape[1],
    )
    active_indices_jax = jnp.asarray(active_indices, dtype=jnp.int32)
    active_summed, active_ctf_probs = _rectangular_active_weighted_sums(
        probs,
        probs_sum_t,
        shifted,
        ctf2_over_nv,
        active_indices_jax,
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32),
        jnp.asarray(active_mask),
    )
    return active_summed, active_ctf_probs, flat_rotations[active_indices_jax]


def _rectangular_active_weighted_image_sums_or_none(
    probs,
    shifted,
    active_indices,
    active_mask,
):
    if active_indices.size == 0:
        return None
    active_image_indices, active_slots, grouped_rotation_rows = _active_row_grouping_for_canonical_matmul(
        active_indices,
        active_mask,
        n_images=probs.shape[0],
        n_rotation_rows=probs.shape[1],
    )
    return _rectangular_active_weighted_image_sums(
        probs,
        shifted,
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32),
        jnp.asarray(active_mask),
    )


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
def _logsumexp_pass2_pairs_score_only(pair_scores, pair_mask):
    """Compute per-image compact pass-2 logZ over valid pairs only."""
    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None], pair_scores - safe_best_log_score[:, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    return jnp.where(has_mass, safe_best_log_score + jnp.log(sum_exp), -jnp.inf)


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
def _winner_take_all_bucket_probs_from_global_argmax(scores, global_argmax, chunk_rotation_start, best_log_score):
    """One-hot probabilities for a rotation chunk using full-bucket argmax."""

    flat_size = scores.shape[1] * scores.shape[2]
    local_argmax = global_argmax - chunk_rotation_start * scores.shape[2]
    valid = (local_argmax >= 0) & (local_argmax < flat_size) & jnp.isfinite(best_log_score)
    safe_argmax = jnp.where(valid, local_argmax, 0)
    probs = jax.nn.one_hot(safe_argmax, flat_size, dtype=scores.real.dtype).reshape(scores.shape)
    return probs * valid[:, None, None].astype(probs.dtype)


@jax.jit
def _winner_take_all_pair_probs(pair_scores, best_pair_argmax, best_log_score):
    """One-hot compact pair probabilities for RELION firstiter_cc."""

    valid = jnp.isfinite(best_log_score)
    probs = jax.nn.one_hot(best_pair_argmax, pair_scores.shape[1], dtype=pair_scores.real.dtype)
    return probs * valid[:, None].astype(probs.dtype)


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


@jax.jit
def _diagnostics_from_normalized_pass2_probs(scores, probs, log_z):
    """Return normalization diagnostics without recomputing probabilities."""

    flat_scores = jnp.asarray(scores).reshape(scores.shape[0], -1)
    flat_probs = jnp.asarray(probs).reshape(probs.shape[0], -1)
    best_log_score = jnp.max(flat_scores, axis=1)
    has_finite = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    best_argmax = jnp.where(has_finite, jnp.argmax(flat_scores, axis=1), 0)
    max_posterior = jnp.where(
        has_finite,
        jnp.max(flat_probs, axis=1),
        jnp.asarray(0.0, dtype=flat_probs.dtype),
    )
    return (
        jnp.where(has_finite, log_z, 0.0),
        jnp.where(has_finite, best_log_score, -jnp.inf),
        best_argmax,
        max_posterior,
    )


def _relion_pass2_reconstruction_probs(probs, *, adaptive_fraction: float):
    """Apply RELION's fine-pass significant threshold before M-step sums."""

    flat_probs = probs.reshape(probs.shape[0], -1)
    mask_flat, n_significant = _find_significant_mask_full_sort(
        flat_probs,
        float(adaptive_fraction),
        -1,
    )
    mask = mask_flat.reshape(probs.shape)
    return jnp.where(mask, probs, 0.0), mask, n_significant


@partial(jax.jit, static_argnames=("adaptive_fraction", "keep_all"))
def _relion_f32_fine_posterior(
    scores,
    *,
    adaptive_fraction: float,
    normalization_sum_weight=None,
    keep_all: bool = False,
):
    """Build full and pruned fine probabilities with RELION GPU arithmetic.

    The reference GPU path shifts its float32 log weights so the maximum is
    50, applies ``expf``, sorts the raw weights in ascending order, and obtains
    both ``sum_weight`` and the lower-tail significance cutoff from a float32
    cumulative scan.  Surviving weights are divided by the full pre-pruning
    ``sum_weight``; they are intentionally not renormalized afterward.
    """

    scores_f32 = jnp.asarray(scores, dtype=jnp.float32)
    flat_scores = scores_f32.reshape(scores_f32.shape[0], -1)
    finite = jnp.isfinite(flat_scores)
    best = jnp.max(jnp.where(finite, flat_scores, -jnp.inf), axis=1)
    has_finite = jnp.isfinite(best)
    safe_best = jnp.where(has_finite, best, jnp.float32(0.0))
    exponent_add = jnp.float32(50.0) - safe_best
    use_native_cuda = False
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        use_native_cuda = cuda_backproject.custom_cuda_requested()
    if use_native_cuda:
        # RELION computes ``50 - weights_max`` once in XFLOAT, then its CUDA
        # kernel evaluates ``expf(score + add)``.  Reassociating this as
        # ``exp(score - best + 50)`` changes hundreds of thousands of raw
        # weights at the iteration-2 case-22 boundary.  Its deployed sm_80
        # scan policy also remains observable when the same binary is JITed
        # on Hopper, so the CUDA primitive pins that policy explicitly.
        finite_scores = jnp.where(finite, flat_scores, -jnp.inf)
        batched_primitives = (
            cuda_backproject.relion_batched_posterior_primitives_requested()
        )
        if batched_primitives:
            raw_weights = cuda_backproject.relion_exponentiate_batched_f32(
                finite_scores,
                exponent_add,
            )
            sorted_weights, cumulative = (
                cuda_backproject.relion_cub_sort_scan_batched_f32(raw_weights)
            )
        else:
            raw_weights = jax.vmap(cuda_backproject.relion_exponentiate_f32)(
                finite_scores,
                exponent_add,
            )
            sorted_weights, cumulative = jax.vmap(
                cuda_backproject.relion_cub_sort_scan_f32,
            )(raw_weights)
    else:
        shifted = jnp.where(
            finite,
            flat_scores + exponent_add[:, None],
            -jnp.inf,
        )
        raw_weights = jnp.where(
            shifted < jnp.float32(-88.0),
            jnp.float32(0.0),
            jnp.exp(shifted),
        )
        raw_weights = jnp.where(
            finite & jnp.isfinite(raw_weights),
            raw_weights,
            jnp.float32(0.0),
        )
        sorted_weights = jnp.sort(raw_weights, axis=1)
        cumulative = jnp.cumsum(sorted_weights, axis=1, dtype=jnp.float32)
    fine_sum_weight = cumulative[:, -1]
    if normalization_sum_weight is None:
        sum_weight = fine_sum_weight
    else:
        # Gradient InitialModel with adaptive_oversampling==0 does not update
        # op.sum_weight in the symbolic fine pass.  The CUDA fine weights are
        # shifted by their own maximum, then divided directly by the numeric
        # float32 denominator retained from the coarse pass.
        sum_weight = jnp.asarray(normalization_sum_weight, dtype=jnp.float32)
    has_mass = has_finite & jnp.isfinite(sum_weight) & (sum_weight > jnp.float32(0.0))
    if keep_all:
        threshold = jnp.zeros_like(sum_weight)
        mask_flat = has_mass[:, None] & finite & (raw_weights > jnp.float32(0.0))
    else:
        tail_target = _relion_cuda_f32_tail_target(fine_sum_weight, adaptive_fraction)
        threshold_idx = jax.vmap(
            lambda row, target: jnp.searchsorted(row, target, side="right")
        )(
            cumulative,
            tail_target,
        )
        threshold_idx = jnp.minimum(threshold_idx, cumulative.shape[1] - 1)
        threshold = sorted_weights[jnp.arange(flat_scores.shape[0]), threshold_idx]
        mask_flat = has_mass[:, None] & finite & (raw_weights >= threshold[:, None])
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    if use_native_cuda:
        if batched_primitives:
            normalized_weights = cuda_backproject.relion_divide_batched_f32(
                raw_weights,
                safe_sum_weight,
            )
        else:
            normalized_weights = jax.vmap(cuda_backproject.relion_divide_f32)(
                raw_weights,
                safe_sum_weight,
            )
    else:
        normalized_weights = raw_weights / safe_sum_weight[:, None]
    reconstruction_probs_flat = jnp.where(
        mask_flat,
        normalized_weights,
        jnp.float32(0.0),
    )
    n_significant = jnp.sum(mask_flat, axis=1).astype(jnp.int32)
    output_shape = scores_f32.shape
    return (
        normalized_weights.reshape(output_shape),
        reconstruction_probs_flat.reshape(output_shape),
        mask_flat.reshape(output_shape),
        n_significant,
        sum_weight,
        threshold,
    )


def _relion_f32_fine_reconstruction_probs(scores, *, adaptive_fraction: float):
    """Return the legacy pruned view of :func:`_relion_f32_fine_posterior`."""

    full = _relion_f32_fine_posterior(
        scores,
        adaptive_fraction=adaptive_fraction,
    )
    return full[1:]


def relion_x_half_f32_fine_posterior_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION float32 fine posterior is enabled.

    Preserve the PR179 production default; enable this separately qualified
    arithmetic path explicitly for boundary experiments.
    """

    return parse_env_flag(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, default=default)


def _relion_fine_parent_execution_order_enabled(
    *,
    use_relion_f32_fine_posterior: bool,
) -> bool:
    """Keep exact fine-posterior diagnostics in RELION's candidate order."""

    return bool(use_relion_f32_fine_posterior) or parse_env_flag(
        _RELION_FINE_ROTATION_EXECUTION_ORDER_ENV,
        default=False,
    )


def _relion_pass2_reconstruction_probs_for_mstep(
    scores,
    probs,
    *,
    adaptive_fraction: float,
    use_relion_x_half_mstep: bool,
    use_relion_f32_fine_posterior: bool = False,
    winner_take_all: bool = False,
    return_diagnostics: bool = False,
):
    """Select the default or diagnostic fine-posterior reconstruction path."""

    if (
        use_relion_x_half_mstep
        and not winner_take_all
        and (
            bool(use_relion_f32_fine_posterior)
            or relion_x_half_f32_fine_posterior_enabled()
        )
    ):
        reconstruction_probs, mask, n_significant, sum_weight, threshold = (
            _relion_f32_fine_reconstruction_probs(
                scores,
                adaptive_fraction=float(adaptive_fraction),
            )
        )
        if return_diagnostics:
            return reconstruction_probs, mask, n_significant, sum_weight, threshold
        return reconstruction_probs, mask, n_significant
    reconstruction_probs, mask, n_significant = _relion_pass2_reconstruction_probs(
        probs,
        adaptive_fraction=float(adaptive_fraction),
    )
    if not return_diagnostics:
        return reconstruction_probs, mask, n_significant
    flat_probs = jnp.asarray(probs, dtype=jnp.float64).reshape(probs.shape[0], -1)
    sum_weight = jnp.sum(flat_probs, axis=1, dtype=jnp.float64)
    threshold = jnp.min(
        jnp.where(mask.reshape(mask.shape[0], -1), flat_probs, jnp.inf),
        axis=1,
    )
    threshold = jnp.where(jnp.isfinite(threshold), threshold, 0.0)
    return reconstruction_probs, mask, n_significant, sum_weight, threshold


def _relion_pass2_reconstruction_pair_probs(pair_probs, pair_mask, *, adaptive_fraction: float):
    """Apply RELION's fine-pass significant threshold to compact pair probs."""

    pair_probs = jnp.where(pair_mask, pair_probs, 0.0)
    mask, n_significant = _find_significant_mask_full_sort(
        pair_probs,
        float(adaptive_fraction),
        -1,
    )
    mask = mask & pair_mask
    return jnp.where(mask, pair_probs, 0.0), mask, n_significant


def _relion_fine_mstep_prune_mode(*, use_relion_x_half_mstep: bool, mode_override: str | None = None) -> str:
    """Return the diagnostic fine-pass M-step pruning mode.

    ``per_class`` preserves the original opt-in diagnostic. ``joint`` matches
    RELION Class3D storeWeightedSums: threshold one flattened class x pose
    posterior list per image before accumulating M-step sums.
    ``joint_keep_all`` preserves every admitted coarse candidate while still
    using the joint RELION float32 posterior arithmetic.
    """

    value = mode_override
    if value is None:
        value = os.environ.get(_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV)
    if value is None or not value.strip():
        return "per_class" if use_relion_x_half_mstep else "none"
    mode = value.strip().lower()
    if mode in {"joint_keep_all", "joint-keep-all"}:
        return "joint_keep_all"
    if mode in {"all", "keep_all", "keep-all", "no_prune", "no-prune"}:
        return "none"
    if mode in _SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_JOINT_MODES:
        return "joint"
    if mode in {"1", "true", "yes", "on", "class", "per_class", "per-class"}:
        return "per_class"
    if mode in {"0", "false", "no", "off", "none"}:
        return "per_class" if use_relion_x_half_mstep else "none"
    raise ValueError(
        f"{_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV} must be one of "
        "0/1/per_class/joint/joint_keep_all"
    )


def _relion_pass2_reconstruction_joint_masks(flat_probs_by_class, *, adaptive_fraction: float):
    """Threshold a per-image flattened class x pose posterior list."""

    if not flat_probs_by_class:
        return []
    flat_sizes = [int(probs.shape[1]) for probs in flat_probs_by_class]
    joint_probs = jnp.concatenate(flat_probs_by_class, axis=1)
    joint_mask, _n_significant = _find_significant_mask_full_sort(
        joint_probs,
        float(adaptive_fraction),
        -1,
    )
    split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
    return list(jnp.split(joint_mask, split_points, axis=1))


def _relion_joint_winner_take_all_masks(flat_scores_by_class):
    """Return one global class x pose winner mask per image."""

    if not flat_scores_by_class:
        return []
    flat_sizes = [int(scores.shape[1]) for scores in flat_scores_by_class]
    joint_scores = jnp.concatenate(flat_scores_by_class, axis=1)
    finite = jnp.isfinite(joint_scores)
    safe_scores = jnp.where(finite, joint_scores, -jnp.inf)
    best_idx = jnp.argmax(safe_scores, axis=1)
    valid = jnp.any(finite, axis=1)
    joint_mask = (jnp.arange(joint_scores.shape[1])[None, :] == best_idx[:, None]) & valid[:, None]
    split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
    return list(jnp.split(joint_mask, split_points, axis=1))


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


def _pass2_dump_requested_for_bucket(
    *,
    experiment_dataset,
    image_indices,
    current_size,
) -> bool:
    """Return whether this bucket must stay materialized for a pass-2 dump."""

    return bool(
        pass2_diagnostics._pass2_dump_target_rows(
            experiment_dataset=experiment_dataset,
            image_indices=image_indices,
            current_size=current_size,
        ).size
    )


def _prioritize_stopped_pass2_dump_buckets(
    buckets,
    *,
    experiment_dataset,
    current_size,
):
    """Move explicitly requested dump buckets first in a stopped diagnostic.

    A stop-after-target capture consumes no M-step result, so unrelated
    particles cannot affect the requested particle's fine-score operands.
    Normal refinement and non-stopped dumps retain their original physical
    execution order.
    """

    stopped_pass2_dump = _pass2_dump_enabled() and parse_env_flag(
        _PASS2_DUMP_STOP_AFTER_TARGET_ENV, default=False
    )
    stopped_norm_dump = parse_env_flag(
        "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", default=False
    ) and parse_env_flag(
        _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV, default=False
    )
    if not (stopped_pass2_dump or stopped_norm_dump):
        return buckets

    requested = []
    remaining = []
    for bucket in buckets:
        destination = (
            requested
            if _pass2_dump_requested_for_bucket(
                experiment_dataset=experiment_dataset,
                image_indices=bucket["image_indices"],
                current_size=current_size,
            )
            else remaining
        )
        destination.append(bucket)
    if not requested:
        return buckets
    logger.info(
        "Sparse K=1 pass-2 stopped diagnostic: moving %d requested dump "
        "bucket(s) before %d unrelated bucket(s)",
        len(requested),
        len(remaining),
    )
    return requested + remaining


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
    recon_window_indices=None,
    translation_phases_half=None,
    score_translation_phases=None,
    recon_translation_phases=None,
    relion_score_translation_angles=None,
    return_windowed_shifted=False,
    return_shifted_score=True,
    relion_exact_normalized_cc_operands=False,
    relion_exact_bpref_operands=False,
):
    """Run preprocessing for a batch of images (translations tiled, CTF/noise ratios).

    Mirrors the ``run_em``/``_preprocess_batch`` pipeline exactly so the
    bucketed sparse pass-2 path is bit-for-bit identical to calling
    ``run_em`` per image.
    """
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    if return_windowed_shifted:
        if window_indices is None:
            raise ValueError("return_windowed_shifted requires window_indices")
        if recon_window_indices is None:
            recon_window_indices = window_indices

    image_shape = config.image_shape
    use_normalized_cc = score_mode == "normalized_cc"
    (
        relion_cuda_preprocess,
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        relion_preprocess_kwargs,
    ) = prepare_batch_preprocess_operands(
        experiment_dataset,
        batch,
        image_indices,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        dtype=(np.float64 if use_float64_scoring else np.float32),
    )
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied and not relion_cuda_preprocess:
        batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half_rfloat = (
        relion_ctf._relion_exact_ctf_half_from_source_star(
            experiment_dataset,
            image_indices,
            image_shape,
        )
        if relion_exact_bpref_operands
        else None
    )
    acc_real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    ctf_half = (
        jnp.asarray(ctf_half_rfloat, dtype=acc_real_dtype)
        if ctf_half_rfloat is not None
        else config.compute_ctf_half(jnp.asarray(ctf_params, dtype=acc_real_dtype))
    )
    batch_scale = jnp.asarray(batch_scale_np, dtype=ctf_half.dtype)
    relion_score_corr_img_half = None
    direct_pixel_correction_full = None
    if relion_exact_bpref_operands:
        if relion_preprocess_kwargs is None:
            raise ValueError(
                "exact RELION BPref operands require RELION CUDA preprocessing"
            )
        relion_preprocess_kwargs = dict(relion_preprocess_kwargs)
        relion_preprocess_kwargs["relion_fft_per_image"] = True
        # RELION computes minvsigma2 from its binary64 sigma2 spectrum, then
        # stores the reciprocal as XFLOAT. Preserve that cast boundary and
        # its scalar multiplication order instead of dividing by an already
        # rounded variance. XFLOAT is float64 under ACC_DOUBLE_PRECISION.
        inverse_noise_half = jnp.reciprocal(
            jnp.asarray(noise_variance_half, dtype=jnp.float64)
        ).astype(acc_real_dtype)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        ctf2_over_nv_half = weighted_ctf_half * ctf_half
        relion_score_corr_img_half = _relion_cuda_corr_img_from_native_noise_variance(
            noise_variance_half[None, :],
            ctf_half_rfloat,
            image_shape,
            batch_scale[:, None] if scale_corrections is not None else None,
            output_dtype=acc_real_dtype,
        )
    else:
        inverse_noise_half = None
        weighted_ctf_half = None
        ctf2_over_nv_half = ctf_half**2 / noise_variance_half
    ctf2_score_half = ctf_half**2

    # Raw processed half-spectrum images (BEFORE any per-image correction).
    # The score path uses masked images iff ``score_with_masked_images`` is True,
    # while the reconstruction path always uses the unmasked (raw) images.
    processed_score_half_raw = process_half_image(
        experiment_dataset,
        batch,
        score_with_masked_images,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
    )
    if score_with_masked_images:
        processed_recon_half_raw = process_half_image(
            experiment_dataset,
            batch,
            False,
            relion_preprocess_kwargs=relion_preprocess_kwargs,
        )
    else:
        processed_recon_half_raw = processed_score_half_raw

    if use_normalized_cc:
        # RELION firstiter_cc uses unweighted image power over the same Fourier
        # window as the score denominator, with no Hermitian doubling.
        if relion_exact_normalized_cc_operands:
            # ``exp_local_sqrtXi2`` is formed from an RFLOAT (float64) serial
            # sum before RELION casts its reciprocal to XFLOAT.  A float32 XLA
            # reduction can move that reciprocal by one ULP, which is enough
            # to erase demonstrated fine-translation score margins.
            norm_real = processed_score_half_raw.real.astype(jnp.float64)
            norm_imag = processed_score_half_raw.imag.astype(jnp.float64)
            abs2_half = norm_real * norm_real + norm_imag * norm_imag
        else:
            abs2_half = jnp.abs(processed_score_half_raw) ** 2
        if window_indices is not None:
            abs2_half = abs2_half[:, window_indices]
        batch_norm = jnp.sum(abs2_half, axis=-1, keepdims=True).real
    else:
        # batch_norm starts from raw processed-score images, then follows dense
        # run_em's image-only correction convention below.
        norm_half_weights = make_half_image_weights(image_shape)
        score_power_over_noise = (
            jnp.abs(processed_score_half_raw) ** 2 * inverse_noise_half[None, :]
            if relion_exact_bpref_operands
            else jnp.abs(processed_score_half_raw) ** 2 / noise_variance_half
        )
        batch_norm = jnp.sum(
            score_power_over_noise * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real

    if relion_exact_bpref_operands:
        score_weighted_half = processed_score_half_raw * weighted_ctf_half
        recon_weighted_half = processed_recon_half_raw * weighted_ctf_half
        recon_bpref_input_half = processed_recon_half_raw
    else:
        score_weighted_half = processed_score_half_raw * ctf_half / noise_variance_half
        recon_weighted_half = processed_recon_half_raw * ctf_half / noise_variance_half
        recon_bpref_input_half = None
    folded_normalized_cc_operands = (
        use_normalized_cc and not relion_exact_normalized_cc_operands
    )
    sparse_score_input_half = (
        processed_score_half_raw * ctf_half
        if folded_normalized_cc_operands
        else processed_score_half_raw
    )
    processed_score_half_for_noise = processed_score_half_raw

    # Per-image image corrections follow dense run_em's image-only convention.
    if image_corrections is not None:
        batch_corr = jnp.asarray(batch_corr_np)
        image_only_corr = batch_corr / batch_scale
        # Note: corrections are applied to the per-translation-tiled arrays in
        # run_em, but multiplication by a per-image scalar commutes with the
        # tiling and shifting so we apply it before tiling for efficiency.
        applied_corr = batch_scale if relion_cuda_preprocess else batch_corr
        score_weighted_half = score_weighted_half * applied_corr[:, None]
        recon_weighted_half = recon_weighted_half * applied_corr[:, None]
        if relion_exact_bpref_operands:
            recon_bpref_input_half = recon_bpref_input_half * applied_corr[:, None]
        if return_direct_scoring_io:
            direct_raw_corr = batch_corr / batch_scale
            if folded_normalized_cc_operands:
                sparse_score_input_half = sparse_score_input_half * applied_corr[:, None]
            elif not relion_cuda_preprocess:
                sparse_score_input_half = sparse_score_input_half * direct_raw_corr[:, None]
        if not relion_cuda_preprocess:
            batch_norm = batch_norm * (image_only_corr**2)[:, None]
            processed_score_half_for_noise = processed_score_half_for_noise * image_only_corr[:, None]

    # Per-image scale correction on CTF^2/noise.
    if scale_corrections is not None:
        ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
        ctf2_score_half = ctf2_score_half * (batch_scale**2)[:, None]
        if return_direct_scoring_io:
            if not folded_normalized_cc_operands and not relion_exact_bpref_operands:
                sparse_score_input_half = sparse_score_input_half / batch_scale[:, None]

    # BPref operands remain in their demonstrated native XFLOAT order. Only
    # fine-score corr_img uses RELION's distinct RFLOAT-square construction.
    ctf2_over_nv_recon_half = ctf2_over_nv_half
    if relion_score_corr_img_half is not None:
        ctf2_over_nv_half = relion_score_corr_img_half

    if return_direct_scoring_io and not folded_normalized_cc_operands:
        if relion_exact_bpref_operands:
            pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                batch_scale[:, None],
                ctf_half_rfloat,
                output_dtype=acc_real_dtype,
            )
            direct_pixel_correction_full = pixel_correction
            sparse_score_input_half = sparse_score_input_half * pixel_correction
        else:
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
        phase_factors = half_image_phase_factors(image_shape, batch_shifts, dtype=batch_shifts.dtype)
        if not score_only:
            score_weighted_half = score_weighted_half * phase_factors
            recon_weighted_half = recon_weighted_half * phase_factors
            if relion_exact_bpref_operands:
                recon_bpref_input_half = recon_bpref_input_half * phase_factors
        if return_direct_scoring_io:
            sparse_score_input_half = sparse_score_input_half * phase_factors

    score_weighted_half_for_score = score_weighted_half

    if translation_phases_half is None and not return_windowed_shifted:
        translation_phases_half = half_translation_phase_table(fine_translations, image_shape)

    def _cuda_translate_score(values, pixel_indices):
        if relion_score_translation_angles is None:
            return None
        from recovar import cuda_backproject

        values = jnp.asarray(values)
        pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
        if values.dtype == jnp.complex128:
            return cuda_backproject.relion_translate_score_f64(
                values,
                jnp.asarray(relion_score_translation_angles, dtype=jnp.float64),
                pixel_indices,
                image_shape,
            )
        return cuda_backproject.relion_translate_score_f32(
            jnp.asarray(values, dtype=jnp.complex64),
            jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            pixel_indices,
            image_shape,
        )
    if score_only:
        shifted_score_half = None
        shifted_recon_half = None
        shifted_score_half_with_dc = None
        ctf2_over_nv_half_with_dc = None
    else:
        if return_windowed_shifted:
            score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
            recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
            score_phase = (
                score_translation_phases
                if score_translation_phases is not None
                else _translation_phase_table_for_indices(
                    fine_translations,
                    image_shape,
                    score_indices,
                    translation_phases_half,
                )
            )
            recon_phase = (
                recon_translation_phases
                if recon_translation_phases is not None
                else _translation_phase_table_for_indices(
                    fine_translations,
                    image_shape,
                    recon_indices,
                    translation_phases_half,
                )
            )
            shifted_score_half = None
            if return_shifted_score:
                shifted_score_half = _cuda_translate_score(
                    score_weighted_half_for_score[:, score_indices],
                    score_indices,
                )
                if shifted_score_half is None:
                    shifted_score_half = apply_half_translation_phases(
                        score_weighted_half_for_score[:, score_indices],
                        score_phase,
                    )
            if relion_exact_bpref_operands:
                if relion_score_translation_angles is None:
                    raise ValueError(
                        "exact RELION BPref operands require RELION translation angles"
                    )
                from recovar import cuda_backproject

                translate_bpref = (
                    cuda_backproject.relion_translate_bpref_f64
                    if use_float64_scoring
                    else cuda_backproject.relion_translate_bpref_f32
                )
                complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
                shifted_recon_half = translate_bpref(
                    jnp.asarray(recon_bpref_input_half[:, recon_indices], dtype=complex_dtype),
                    jnp.asarray(weighted_ctf_half[:, recon_indices], dtype=acc_real_dtype),
                    jnp.asarray(relion_score_translation_angles, dtype=acc_real_dtype),
                    recon_indices,
                    image_shape,
                )
            else:
                shifted_recon_half = _cuda_translate_score(
                    recon_weighted_half[:, recon_indices],
                    recon_indices,
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_half[:, recon_indices],
                        recon_phase,
                    )
            if score_with_masked_images:
                shifted_score_half_with_dc = _cuda_translate_score(
                    score_weighted_half[:, recon_indices],
                    recon_indices,
                )
                if shifted_score_half_with_dc is None:
                    shifted_score_half_with_dc = apply_half_translation_phases(
                        score_weighted_half[:, recon_indices],
                        recon_phase,
                    )
            else:
                shifted_score_half_with_dc = shifted_recon_half
        else:
            if relion_exact_bpref_operands:
                if relion_score_translation_angles is None:
                    raise ValueError(
                        "exact RELION BPref operands require RELION translation angles"
                    )
                from recovar import cuda_backproject

                translate_bpref = (
                    cuda_backproject.relion_translate_bpref_f64
                    if use_float64_scoring
                    else cuda_backproject.relion_translate_bpref_f32
                )
                complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
                exact_shifted_recon_half = translate_bpref(
                    jnp.asarray(recon_bpref_input_half, dtype=complex_dtype),
                    jnp.asarray(weighted_ctf_half, dtype=acc_real_dtype),
                    jnp.asarray(relion_score_translation_angles, dtype=acc_real_dtype),
                    jnp.arange(recon_bpref_input_half.shape[1], dtype=jnp.int32),
                    image_shape,
                )
            else:
                exact_shifted_recon_half = None
            shifted_score_half = None
            if return_shifted_score:
                full_pixel_indices = jnp.arange(score_weighted_half_for_score.shape[1], dtype=jnp.int32)
                shifted_score_half = _cuda_translate_score(
                    score_weighted_half_for_score,
                    full_pixel_indices,
                )
                if shifted_score_half is None:
                    shifted_score_half = apply_half_translation_phases(
                        score_weighted_half_for_score,
                        translation_phases_half,
                    )
            if score_with_masked_images:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else _cuda_translate_score(
                        recon_weighted_half,
                        jnp.arange(recon_weighted_half.shape[1], dtype=jnp.int32),
                    )
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_half,
                        translation_phases_half,
                    )
                shifted_score_half_with_dc = _cuda_translate_score(
                    score_weighted_half,
                    jnp.arange(score_weighted_half.shape[1], dtype=jnp.int32),
                )
                if shifted_score_half_with_dc is None:
                    shifted_score_half_with_dc = apply_half_translation_phases(
                        score_weighted_half,
                        translation_phases_half,
                    )
            else:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else _cuda_translate_score(
                        recon_weighted_half,
                        jnp.arange(recon_weighted_half.shape[1], dtype=jnp.int32),
                    )
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_half,
                        translation_phases_half,
                    )
                shifted_score_half_with_dc = shifted_recon_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_recon_half

    shifted_corrected_score_half = None
    direct_score_input = None
    direct_preprocessed_score_input = None
    direct_pixel_correction = None
    if return_direct_scoring_io:
        if return_windowed_shifted:
            score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
            direct_score_input = sparse_score_input_half[:, score_indices]
            direct_preprocessed_score_input = processed_score_half_raw[:, score_indices]
            if direct_pixel_correction_full is not None:
                direct_pixel_correction = direct_pixel_correction_full[:, score_indices]
            direct_score_pixel_indices = score_indices
        else:
            direct_score_input = sparse_score_input_half
            direct_preprocessed_score_input = processed_score_half_raw
            direct_pixel_correction = direct_pixel_correction_full
            direct_score_pixel_indices = jnp.arange(
                sparse_score_input_half.shape[1],
                dtype=jnp.int32,
            )
        if relion_score_translation_angles is not None:
            from recovar import cuda_backproject

            shifted_corrected_score_half = _cuda_translate_score(
                direct_score_input,
                direct_score_pixel_indices,
            )
        else:
            if return_windowed_shifted:
                score_phase = (
                    score_translation_phases
                    if score_translation_phases is not None
                    else _translation_phase_table_for_indices(
                        fine_translations,
                        image_shape,
                        direct_score_pixel_indices,
                        translation_phases_half,
                    )
                )
            else:
                score_phase = translation_phases_half
            shifted_corrected_score_half = apply_half_translation_phases(
                direct_score_input,
                score_phase,
            )

    if half_spectrum_scoring and not use_normalized_cc:
        dc_shell_idx = make_shell_indices_half(image_shape)
        dc_mask = dc_shell_idx == 0
        if not score_only and shifted_score_half is not None:
            if return_windowed_shifted:
                score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
                shifted_score_half = jnp.where(dc_mask[score_indices][None, :], 0.0, shifted_score_half)
            else:
                shifted_score_half = jnp.where(dc_mask[None, :], 0.0, shifted_score_half)
        ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    if return_direct_scoring_io and use_normalized_cc:
        inv_xi2 = (1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))).astype(
            precision_policy.score_real_dtype,
        )
        if folded_normalized_cc_operands:
            shifted_corrected_score_half = shifted_corrected_score_half * jnp.repeat(inv_xi2, n_trans, axis=0)
            ctf2_over_nv_half = ctf2_score_half * inv_xi2
        else:
            # buildCorrImage evaluates CTF * CTF in RFLOAT and casts the full
            # product to XFLOAT once.  Reusing the generic float32 CTF-square
            # path changes most corr_img pixels by one or two ULPs.  Preserve
            # the source order here; the existing RECOVAR FFT normalization
            # is already encoded in ``inv_xi2``.
            corr_ctf_rfloat = (
                ctf_half.astype(jnp.float64)
                if ctf_half_rfloat is None
                else ctf_half_rfloat
            )
            ctf2_over_nv_half = _relion_cuda_corr_img_from_rfloat_ctf(
                inv_xi2,
                corr_ctf_rfloat,
                batch_scale[:, None] if scale_corrections is not None else None,
            )
    if return_windowed_shifted:
        score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
        ctf2_over_nv_half = ctf2_over_nv_half[:, score_indices]
        if ctf2_over_nv_half_with_dc is not None:
            recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
            ctf2_over_nv_half_with_dc = ctf2_over_nv_half_with_dc[:, recon_indices]
    if score_only:
        ctf2_over_nv_half = ctf2_over_nv_half.astype(precision_policy.score_real_dtype)
    else:
        if shifted_score_half is not None:
            shifted_score_half = shifted_score_half.astype(precision_policy.score_complex_dtype)
        ctf2_over_nv_half = ctf2_over_nv_half.astype(precision_policy.score_real_dtype)
        if precision_policy.use_float64_scoring:
            shifted_recon_half = shifted_recon_half.astype(precision_policy.score_complex_dtype)
            shifted_score_half_with_dc = shifted_score_half_with_dc.astype(precision_policy.score_complex_dtype)
            ctf2_over_nv_half_with_dc = ctf2_over_nv_half_with_dc.astype(precision_policy.score_real_dtype)
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
        direct_score_input,
        direct_preprocessed_score_input,
        direct_pixel_correction,
        (
            None
            if relion_preprocess_kwargs is None
            else relion_preprocess_kwargs.get("relion_normalization_factors")
        ),
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        inverse_noise_half,
        ctf_half_rfloat,
    )


def subtract_projected_reference_from_sparse_mstep_sums(
    summed,
    reconstruction_probs,
    projected_reference,
    ctf2_over_noise,
):
    """Form RELION VDAM's residual backprojection operand on compact support."""

    reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
    return subtract_projected_reference_from_sparse_mstep_rotation_sums(
        summed,
        reconstruction_probs_sum_t,
        projected_reference,
        ctf2_over_noise,
    )


def subtract_projected_reference_from_sparse_mstep_rotation_sums(
    summed,
    posterior_mass_by_rotation,
    projected_reference,
    ctf2_over_noise,
):
    """Subtract reference signal after dense or pair-sparse translation sums."""

    projected_reference_weighted = projected_reference * ctf2_over_noise[:, None, :]
    projected_reference_delta = jnp.where(
        posterior_mass_by_rotation[..., None] != 0.0,
        posterior_mass_by_rotation[..., None] * projected_reference_weighted,
        0.0,
    )
    return summed - projected_reference_delta


def compute_pass2_stats_sparse_bucketed(
    experiment_dataset,
    volume,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    oversampling_order,
    current_size,
    reconstruction_current_size=None,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    projection_mask_current_image_disk=True,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    normalization_log_z=None,
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    rotation_block_size_for_quantization=5000,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
):
    """Bucketed batched implementation of sparse pass-2 oversampling.

    Returns the same tuple as ``compute_pass2_stats_sparse``.

    ``relion_exact_fine_gaussian`` enables RELION's direct fine-search
    diff2/minimum ordering in the active ACC precision (float32 or float64).
    """
    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    device_signature_requested = bool(
        device_signature_configured and bpref_device_signature_active
    )
    contribution_diagnostics_active = bool(
        os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip()
        and (bpref_device_signature_active or not device_signature_configured)
    )
    membership_diagnostics_active = bpref_diagnostics._bpref_membership_dump_requested()
    scoped_diagnostic_flags = bpref_diagnostics._scoped_bpref_diagnostic_flags(
        active=bpref_device_signature_active
    )
    production_firstiter_xhalf_topology = bool(
        relion_x_half_mstep and relion_firstiter_winner_take_all
    )
    execution_modes = bpref_diagnostics._resolve_bpref_execution_modes(
        scoped_diagnostic_flags,
        device_signature_requested=device_signature_requested,
        production_firstiter_xhalf_topology=production_firstiter_xhalf_topology,
    )
    diagnostic_sequential_translation_reduction = execution_modes[
        "diagnostic_sequential_translation_reduction"
    ]
    diagnostic_per_particle_launches = execution_modes["diagnostic_per_particle_launches"]
    fused_atomics_requested = scoped_diagnostic_flags["fused_atomics"]
    shadow_only_mode_requested = execution_modes["shadow_only"]
    # A scoped device capture is observational: ordinary score/reduction/
    # adjoint outputs remain authoritative and the requested RELION-order
    # variants execute only as checked diagnostic shadows.
    use_sequential_translation_reduction = execution_modes[
        "live_sequential_translation_reduction"
    ]
    use_per_particle_launches = execution_modes["live_per_particle_launches"]
    if preserve_bpref_particle_order:
        if not relion_x_half_mstep:
            raise ValueError(
                "preserve_bpref_particle_order requires the RELION x-half M-step"
            )
        # Scoring may batch adjacent particles with the same padded support,
        # but RELION contributes one particle at a time to BPref.  Keep that
        # launch boundary authoritative even when no diagnostic flag is set.
        use_per_particle_launches = True
    if device_signature_requested:
        if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"):
            raise RuntimeError(
                "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR requires "
                "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"
            )
        bpref_diagnostics._require_bpref_device_soft_particle_arm(
            use_relion_x_half_mstep=bool(relion_x_half_mstep),
        )
    from recovar.em.sampling import (
        get_oversampled_translation_grid,
        rotation_grid_size,
    )

    if relion_firstiter_score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            "relion_firstiter_score_mode must be 'gaussian' or 'normalized_cc', "
            f"got {relion_firstiter_score_mode!r}",
        )
    use_exact_relion_gaussian = bool(
        relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
    )
    use_relion_fine_diff2_fused_ffi = bool(
        relion_fine_diff2_fused_ffi
        or parse_env_flag(_RELION_FINE_DIFF2_FUSED_FFI_ENV, default=False)
    )
    use_relion_f32_fine_posterior = bool(
        relion_f32_fine_posterior
        or relion_x_half_f32_fine_posterior_enabled()
    )
    winner_take_all = bool(relion_firstiter_winner_take_all)
    if bool(disable_adjoint_y) != bool(disable_adjoint_ctf):
        raise NotImplementedError("Sparse pass-2 currently supports disabling both M-step adjoints together")
    score_only = bool(disable_adjoint_y and disable_adjoint_ctf)
    if score_only and mstep_subtract_ctf_projection:
        raise ValueError("score-only sparse pass 2 cannot subtract the projected reference")
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
    has_external_score_normalization = (
        normalization_log_z is not None or normalization_other_score_log_z is not None
    )
    if normalization_score_mode is not None and normalization_score_mode not in {
        "gaussian",
        "normalized_cc",
    }:
        raise ValueError(
            "normalization_score_mode must be 'gaussian' or 'normalized_cc', "
            f"got {normalization_score_mode!r}",
        )
    if has_external_score_normalization and normalization_score_mode is None:
        raise ValueError(
            "external sparse pass-2 score normalization requires normalization_score_mode; "
            "Gaussian logZ is absolute while normalized-CC logZ is centered"
        )
    if normalization_score_mode is not None and not has_external_score_normalization:
        raise ValueError("normalization_score_mode requires an external score normalization")
    if (
        has_external_score_normalization
        and normalization_score_mode is not None
        and normalization_score_mode != relion_firstiter_score_mode
    ):
        raise ValueError(
            "external score normalization mode does not match this pass: "
            f"external={normalization_score_mode!r}, pass={relion_firstiter_score_mode!r}",
        )
    if normalization_other_score_log_z is not None and not return_score_log_z:
        raise ValueError("normalization_other_score_log_z requires return_score_log_z=True")
    if score_only and accumulate_noise:
        raise ValueError("Sparse pass-2 score-only mode is incompatible with accumulate_noise=True")
    use_relion_projector = relion_projector_half is not None
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim != 3:
            raise ValueError(
                "relion_projector_half must be a single-class Projector::data slab "
                f"with shape (z, y, x_half), got {relion_projector_half.shape}",
            )

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)

    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    mstep_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if (
        use_exact_relion_gaussian
        and current_size is not None
        and int(current_size) < int(W)
        and (not half_spectrum_scoring or square_window)
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian high-resolution scoring requires "
            "half_spectrum_scoring=True and square_window=False"
        )
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
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    device_memory_bytes = _device_memory_limit_bytes()
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)

    if bool(relion_x_half_mstep):
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max.  The reconstruction boundary then crops the output
        # back to ``volume_shape``.
        recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=mstep_current_size,
        )
    elif reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    use_relion_x_half_mstep = bool(relion_x_half_mstep)
    use_relion_fine_mstep_prune = bool(relion_fine_mstep_prune) or use_relion_x_half_mstep
    use_relion_f32_fine_posterior = (
        use_relion_x_half_mstep
        and not winner_take_all
        and (
            bool(relion_f32_fine_posterior)
            or relion_x_half_f32_fine_posterior_enabled()
        )
    )
    use_half_volume_mstep = bool(relion_half_volume_mstep) or use_relion_x_half_mstep
    # Preserve early mode validation; compact-pair dispatch is owned by the K-class path.
    _compact_pair_mstep_mode_for_pass()
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape) if use_half_volume_mstep else recon_volume_shape
    recon_volume_size = int(np.prod(recon_accum_shape))
    if use_relion_x_half_mstep:
        logger.info(
            "Sparse pass-2 RELION x-half current-size BPref accumulator shape: "
            "volume_shape=%s score_current_size=%s model_current_size=%s padding_factor=%s recon_volume_shape=%s half_accum_shape=%s voxels=%d",
            tuple(volume_shape),
            current_size,
            mstep_current_size,
            reconstruction_padding_factor,
            tuple(recon_volume_shape),
            tuple(recon_accum_shape),
            recon_volume_size,
        )
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
    )

    # Projection volume + padding
    if projection_padding_factor > 1 and not use_relion_projector:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            volume,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=mstep_current_size,
        )
    else:
        mean_for_proj = volume
        proj_volume_shape = volume_shape

    # Fine translations and prior mapping
    translations_source_np = np.asarray(translations)
    translations_np = np.asarray(translations_source_np, dtype=precision_policy.score_real_dtype)
    if translation_step is None:
        unique_vals = np.unique(translations_np)
        diffs = np.diff(np.sort(unique_vals))
        diffs = diffs[diffs > 1e-6]
        translation_step = float(diffs.min()) if diffs.size else 1.0
    if fine_translations_override is None and fine_translation_parent_override is None:
        fine_translations_source, fine_translation_parent = get_oversampled_translation_grid(
            translations_source_np,
            translation_step,
            oversampling_order=oversampling_order,
        )
        fine_translations = np.asarray(fine_translations_source, dtype=precision_policy.score_real_dtype)
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations_source = np.asarray(fine_translations_override)
        fine_translations = np.asarray(fine_translations_source, dtype=precision_policy.score_real_dtype)
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
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=precision_policy.score_real_dtype)
        fine_translation_prior_2d = expand_fine_translation_prior(
            translation_log_prior_np,
            fine_translation_parent,
            n_images=n_images,
            n_fine_trans=n_fine_trans,
            dtype=precision_policy.score_real_dtype,
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
        fine_source_eulers_override=fine_source_eulers_override,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=_relion_fine_parent_execution_order_enabled(
            use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        ),
        dtype=precision_policy.score_real_dtype,
    )
    prep_s = time.time() - prep_t0

    local_rot_counts = [int(rots.shape[0]) for rots in per_image_inputs["oversampled_rots"]]
    valid_candidate_counts = [_candidate_mask_count(m) for m in per_image_inputs["candidate_mask"]]

    # Bucket.  The default cap intentionally allows multi-image buckets for
    # broad soft posteriors; the old 100k cap fragmented 100k/256 K=4 into
    # tens of thousands of one-image launches on A100.
    max_hypotheses_per_microbatch = _max_hypotheses_per_microbatch_for_pass(
        score_only=score_only,
        use_window=budget_window_spec.use_window,
        has_external_normalization=normalization_log_z is not None or normalization_other_score_log_z is not None,
        conservative_dump_execution=_pass2_conservative_dump_execution_enabled(),
        n_score_pixels=budget_window_spec.n_score,
        device_memory_bytes=device_memory_bytes,
        score_complex_dtype=precision_policy.score_complex_dtype,
    )
    has_external_normalization = normalization_log_z is not None or normalization_other_score_log_z is not None
    max_translation_tile_bytes = _max_translation_tile_bytes_for_pass(
        device_memory_bytes,
        has_external_normalization=has_external_normalization,
    )
    max_noise_block_bytes = _max_noise_block_bytes_for_pass(device_memory_bytes)
    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)
    translation_tile_half_pixels = _translation_tile_half_pixels_for_budget(
        use_window=budget_window_spec.use_window,
        n_score_pixels=budget_window_spec.n_score,
        n_recon_pixels=budget_window_spec.n_recon,
    )
    (
        max_images_per_microbatch,
        full_translation_tile_max_images,
        window_translation_tile_max_images,
        window_translation_tile_max_multiplier,
    ) = _max_images_for_sparse_pass2_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_translation_tile_bytes,
        complex_dtype=precision_policy.score_complex_dtype,
        translation_tile_half_pixels=translation_tile_half_pixels,
    )
    small_bucket_coalesce_size = _small_bucket_coalesce_size_for_pass(n_images)
    (
        tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size,
    ) = _tail_bucket_coalesce_params_for_pass(fused_k_class=False)
    projection_complex_dtype = _projection_cache_budget_complex_dtype(
        jnp.asarray(mean_for_proj).dtype,
        precision_policy.score_complex_dtype,
        use_relion_projector=use_relion_projector,
    )
    projection_budget_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=budget_window_spec.use_window,
        use_relion_projector=use_relion_projector,
    )
    max_projected_rotations_per_projection_call = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=device_memory_bytes,
        n_projection_pixels=projection_budget_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=not (budget_window_spec.use_window or score_only),
    )
    max_projection_gather_bytes = _max_projection_gather_bytes_for_pass(device_memory_bytes)
    processing_order_override = _resolve_bpref_processing_order(
        n_images,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
    )
    processing_order_chunk_size = 1
    processing_order_group_by_bucket_size = False
    processing_order_batch_consecutive_bucket_sizes = False
    if processing_order_override is not None:
        processing_order_group_by_bucket_size = parse_env_flag(
            _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV,
            default=False,
        )
        (
            processing_order_chunk_size,
            processing_order_batch_consecutive_bucket_sizes,
        ) = _resolve_bpref_execution_bucket_policy(
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            processing_order_group_by_bucket_size=processing_order_group_by_bucket_size,
        )
        if preserve_bpref_particle_order:
            if parse_env_flag(_BPREF_REVERSE_PHYSICAL_ORDER_ENV, default=False):
                logger.warning(
                    "STRICT-PARITY diagnostic: reversing fresh RELION physical "
                    "BPref particle order; scoring inputs and output identities remain aligned"
                )
            else:
                logger.info(
                    "STRICT-PARITY: preserving fresh RELION physical BPref particle "
                    "order (%s)",
                    "stable within support-size buckets"
                    if processing_order_group_by_bucket_size
                    else (
                        "global; batching only consecutive equal-size supports"
                        if processing_order_batch_consecutive_bucket_sizes
                        else f"global; {processing_order_chunk_size} consecutive "
                        "mixed-support particles per diagnostic bucket"
                    ),
                )
        else:
            logger.info(
                "STRICT-PARITY diagnostic: executing K=1 BPref particles in the "
                "explicit local order from %s (%s; %d contiguous particles per bucket call)",
                os.environ[_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV],
                "stable within support-size buckets"
                if processing_order_group_by_bucket_size
                else "global",
                processing_order_chunk_size,
            )
    bucket_t0 = time.time()
    buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        small_bucket_coalesce_size=small_bucket_coalesce_size,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        processing_order_override=processing_order_override,
        processing_order_chunk_size=processing_order_chunk_size,
        processing_order_group_by_bucket_size=processing_order_group_by_bucket_size,
        processing_order_batch_consecutive_bucket_sizes=(
            processing_order_batch_consecutive_bucket_sizes
        ),
    )
    buckets = _prioritize_stopped_pass2_dump_buckets(
        buckets,
        experiment_dataset=experiment_dataset,
        current_size=current_size,
    )
    bucket_s = time.time() - bucket_t0

    logger.info(
        "Sparse pass-2 bucketing: %d images -> %d buckets (%s; "
        "max_hypotheses_per_microbatch=%d, max_images_per_microbatch=%d, "
        "translation_tile_half_pixels=%s, windowed_translation_tile_cap=%s, "
        "full_tile_max_images=%d, window_tile_max_images=%s, window_tile_max_multiplier=%s, "
        "small_bucket_coalesce_size=%s, tail_bucket_coalesce=%s/%s/%s, "
        "max_projected_rotations_per_projection_call=%s, max_translation_tile_bytes=%d, "
        "max_projection_gather_bytes=%d, max_noise_block_bytes=%d, max_adjoint_block_bytes=%d, "
        "n_score_pixels=%d, device_memory_gib=%.2f)",
        n_images,
        len(buckets),
        _bucket_summary(buckets),
        max_hypotheses_per_microbatch,
        max_images_per_microbatch,
        "full" if translation_tile_half_pixels is None else str(int(translation_tile_half_pixels)),
        str(_windowed_translation_tile_cap_enabled_for_pass()),
        int(full_translation_tile_max_images),
        "unset" if window_translation_tile_max_images is None else str(int(window_translation_tile_max_images)),
        "unset"
        if window_translation_tile_max_multiplier is None
        else str(int(window_translation_tile_max_multiplier)),
        "unset" if small_bucket_coalesce_size is None else str(int(small_bucket_coalesce_size)),
        "unset"
        if tail_bucket_coalesce_max_images is None
        else str(int(tail_bucket_coalesce_max_images)),
        "unset"
        if tail_bucket_coalesce_max_inflation is None
        else f"{float(tail_bucket_coalesce_max_inflation):.3g}",
        "unset"
        if tail_bucket_coalesce_min_bucket_size is None
        else str(int(tail_bucket_coalesce_min_bucket_size)),
        "unset"
        if max_projected_rotations_per_projection_call is None
        else str(int(max_projected_rotations_per_projection_call)),
        max_translation_tile_bytes,
        max_projection_gather_bytes,
        max_noise_block_bytes,
        max_adjoint_block_bytes,
        int(budget_window_spec.n_score),
        (-1.0 if device_memory_bytes is None else device_memory_bytes / float(1024**3)),
    )
    logger.info("Sparse pass-2 setup timing: hypothesis_prep=%.2fs bucket=%.2fs", prep_s, bucket_s)
    if use_relion_x_half_mstep:
        mstep_layout_label = "RELION x-half BPref-layout"
    elif use_half_volume_mstep:
        mstep_layout_label = "native half-volume"
    else:
        mstep_layout_label = "full-volume"
    logger.info(
        "Sparse pass-2 M-step: using %s backprojection",
        mstep_layout_label,
    )
    if production_firstiter_xhalf_topology:
        logger.info(
            "STRICT-PARITY: fresh K=1 firstiter-CC uses sequential XFLOAT "
            "translation reduction and particle-owned fused BPref launches"
        )
    elif use_relion_x_half_mstep and diagnostic_sequential_translation_reduction:
        logger.info(
            "Sparse pass-2 RELION x-half M-step diagnostic: sequential XFLOAT-precision "
            "translation reduction runs as %s",
            "a checked shadow" if shadow_only_mode_requested else "the standalone diagnostic path",
        )
    if use_relion_f32_fine_posterior:
        logger.info(
            "Sparse pass-2 RELION x-half M-step: using float32 fine-posterior "
            "normalization and significance pruning"
        )
    if use_relion_fine_mstep_prune and not use_relion_x_half_mstep:
        logger.info("Sparse pass-2 M-step: applying RELION fine-pass significant-weight pruning")

    best_eulers = (
        np.empty((n_images, 3), dtype=np.float64)
        if return_source_eulers and all(x is not None for x in per_image_inputs["source_eulers"])
        else None
    )

    # Output accumulators (volume_size matches what original returned: full N**3)
    if return_score_log_z_only:
        Ft_y_total = None
        Ft_ctf_total = None
        hard_assignment = None
        best_rotations = None
        best_rotation_indices = None
    else:
        Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype)
        Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype)
        hard_assignment = np.empty(n_images, dtype=np.int32)
        best_rotations = np.empty((n_images, 3, 3), dtype=precision_policy.score_real_dtype)
        best_rotation_indices = np.empty(n_images, dtype=np.int64)

    # K-class assignment depends on small inter-class score deltas after adding
    # a large image-power offset. Keep these in float64 like dense run_em.
    log_evidence = np.empty(n_images, dtype=np.float64) if (return_stats or return_score_log_z_only) else None
    best_log_score = np.empty(n_images, dtype=np.float64) if return_stats else None
    max_posterior = np.empty(n_images, dtype=precision_policy.score_real_dtype) if return_stats else None
    rotation_posterior_sums = np.zeros(n_coarse_rot, dtype=np.float64) if return_stats else None
    score_log_z = (
        np.empty(n_images, dtype=np.float64)
        if ((return_stats and return_score_log_z) or return_score_log_z_only)
        else None
    )

    noise_wsum_total = None
    noise_img_power_total = None
    noise_norm_correction_total = None
    noise_wavg_direct_norm_current_total = None
    noise_wavg_direct_norm_high_total = None
    noise_scale_correction_xa_total = None
    noise_scale_correction_aa_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )
    if group_ids is not None:
        noise_scale_correction_xa_total = np.zeros(n_scale_groups, dtype=np.float64)
        noise_scale_correction_aa_total = np.zeros(n_scale_groups, dtype=np.float64)
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        noise_wsum_total = np.zeros(n_shells, dtype=np.float64)
        noise_img_power_total = np.zeros(n_shells, dtype=np.float64)
        noise_norm_correction_total = np.zeros(n_images, dtype=np.float64)
        noise_wavg_direct_norm_current_total = np.zeros(n_images, dtype=np.float64)
        noise_wavg_direct_norm_high_total = np.zeros(n_images, dtype=np.float64)

    # Forward-model config & half/window precomputes
    window_setup = _sparse_pass2_window_setup(
        experiment_dataset,
        disc_type=disc_type,
        image_shape=image_shape,
        current_size=current_size,
        n_half=n_half,
        mstep_current_size=mstep_current_size,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        log_label="Sparse pass-2",
    )
    config = window_setup.config
    window_spec = window_setup.window_spec
    use_window = window_setup.use_window
    window_indices_np = window_setup.window_indices_np
    window_indices = window_setup.window_indices
    recon_window_indices = window_setup.recon_window_indices
    relion_x_half_recon_indices = window_setup.relion_x_half_recon_indices
    windowed_prepare = window_setup.windowed_prepare
    n_windowed = window_setup.n_windowed
    n_recon_windowed = window_setup.n_recon_windowed

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=relion_firstiter_score_mode != "normalized_cc",
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            image_shape,
            current_size,
            window_indices_np if use_window else np.arange(int(n_half), dtype=np.int32),
        ),
        dtype=jnp.int32,
    )

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    fresh_k1_guard = bool(source_faithful_spectrum_norm)
    source_faithful_spectrum_norm = _relion_powerclass_spectrum_norm_enabled(
        fresh_k1_guard=fresh_k1_guard,
    )
    relion_exact_bpref_operands = _relion_exact_bpref_operands_enabled(
        fresh_k1_guard=fresh_k1_guard,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    if relion_exact_bpref_operands:
        logger.info(
            "STRICT-PARITY: using RELION binary64-to-%s inverse-noise and "
            "fused translate-then-weight BPref operands",
            "float64" if use_float64_scoring else "float32",
        )
    if source_faithful_spectrum_norm:
        logger.info(
            "STRICT-PARITY: normalization high shell consumes RELION's "
            "powerClass shell spectrum"
        )

    if accumulate_noise:
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        if use_window:
            shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
                shell_indices_half,
                image_shape,
                current_size,
                window_indices,
            )
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        scale_correction_pixel_mask = _relion_scale_correction_pixel_mask(
            scale_correction_data_vs_prior,
            shell_indices_noise,
            n_shells=n_shells,
        )

    normalization_log_z_np = optional_normalization_vector(
        normalization_log_z, name="normalization_log_z", n_images=n_images,
    )
    normalization_other_score_log_z_np = optional_normalization_vector(
        normalization_other_score_log_z, name="normalization_other_score_log_z", n_images=n_images,
    )
    dump_pass2_operands = _pass2_dump_enabled()

    projection_cache = None
    if _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations_override,
        dump_pass2_operands=dump_pass2_operands,
    ):
        n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
        if use_window:
            # The RELION centered projector materializes a larger full-half
            # transient, but _compute_sparse_pass2_windowed_projections_block
            # chunks that transient with max_projected_rotations.  Cache
            # admission should therefore be based on retained window rows.
            transient_projection_bytes = _projection_cache_transient_bytes(
                n_fine_rot,
                n_windowed,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_abs2=False,
            )
            if not score_only:
                transient_projection_bytes += _projection_cache_transient_bytes(
                    n_fine_rot,
                    n_recon_windowed,
                    projection_complex_dtype=precision_policy.score_complex_dtype,
                    include_abs2=True,
                )
        else:
            transient_projection_bytes = _projection_cache_transient_bytes(
                n_fine_rot,
                n_half,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_abs2=not score_only,
            )
        max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
        if _projection_cache_fits_budget(transient_projection_bytes, max_projection_cache_bytes):
            cache_t0 = time.time()
            if use_window:
                projection_kwargs = _projection_kwargs_for_relion_score_window(
                    window_spec.projection_kwargs(return_abs2=False),
                    use_relion_projector=use_relion_projector,
                    current_size=current_size,
                )
                projection_kwargs["mask_current_image_disk"] = bool(
                    projection_mask_current_image_disk
                )
                score_cache, recon_cache, recon_abs2_cache = _compute_sparse_pass2_windowed_projections_block(
                    mean_for_proj,
                    jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype),
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    score_indices=window_indices,
                    recon_indices=None if score_only else recon_window_indices,
                    max_projected_rotations=max_projected_rotations_per_projection_call,
                    output_complex_dtype=precision_policy.score_complex_dtype,
                    output_abs2_dtype=precision_policy.score_real_dtype,
                    relion_projector_half=relion_projector_half,
                    relion_projector_r_max=relion_projector_r_max,
                    projection_padding_factor=projection_padding_factor,
                    **projection_kwargs,
                )
                projection_cache = {
                    "score": score_cache,
                    "recon": recon_cache,
                    "recon_abs2": recon_abs2_cache,
                }
            else:
                projection_kwargs = window_spec.projection_kwargs(return_abs2=None if not score_only else False)
                projection_kwargs["mask_current_image_disk"] = bool(
                    projection_mask_current_image_disk
                )
                proj_half_cache_flat, proj_abs2_cache_flat = _compute_sparse_pass2_projections_block(
                    mean_for_proj,
                    jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype),
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    max_projected_rotations=max_projected_rotations_per_projection_call,
                    output_complex_dtype=precision_policy.score_complex_dtype,
                    output_abs2_dtype=precision_policy.score_real_dtype,
                    relion_projector_half=relion_projector_half,
                    relion_projector_r_max=relion_projector_r_max,
                    projection_padding_factor=projection_padding_factor,
                    **projection_kwargs,
                )
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
    relion_score_translation_angles = (
        _relion_cuda_score_translation_angles_if_available(
            fine_translations_source,
            image_shape,
            enabled=use_exact_relion_gaussian or relion_exact_bpref_operands,
            dtype=np.float64 if use_float64_scoring else np.float32,
        )
    )
    translation_phases_half = None if windowed_prepare else half_translation_phase_table(fine_translations, image_shape)

    exact_raw_diff2_cache_limit_bytes = 0
    exact_raw_diff2_cache_admission_logged = False
    if use_exact_relion_gaussian:
        free_device_memory_bytes = _device_free_memory_bytes()
        allocator_free_memory_bytes = _jax_allocator_free_memory_bytes()
        exact_raw_diff2_cache_max_bytes = parse_env_nonnegative_int(
            _EXACT_RAW_DIFF2_CACHE_MAX_BYTES_ENV,
        )
        if exact_raw_diff2_cache_max_bytes is None:
            exact_raw_diff2_cache_max_bytes = _EXACT_RAW_DIFF2_CACHE_MAX_BYTES
        exact_raw_diff2_cache_limit_bytes = _exact_raw_diff2_cache_limit_bytes(
            device_memory_bytes,
            free_device_memory_bytes,
            allocator_free_memory_bytes,
            max_cache_bytes=exact_raw_diff2_cache_max_bytes,
        )
        logger.info(
            "Sparse pass-2 exact raw-diff2 reuse cap: %.2f MiB "
            "(device=%.2f GiB physical_free=%s allocator_free=%s configured_max=%.2f MiB)",
            exact_raw_diff2_cache_limit_bytes / float(1024**2),
            0.0 if device_memory_bytes is None else device_memory_bytes / float(1024**3),
            "unknown"
            if free_device_memory_bytes is None
            else f"{free_device_memory_bytes / float(1024**3):.2f} GiB",
            "unknown"
            if allocator_free_memory_bytes is None
            else f"{allocator_free_memory_bytes / float(1024**3):.2f} GiB",
            exact_raw_diff2_cache_max_bytes / float(1024**2),
        )

    bucket_group_stats = _bucket_group_stats(buckets)
    last_bucket_size_logged = None
    group_t0 = None
    group_completed_chunks = 0
    group_completed_images = 0
    group_last_progress_t = None
    progress_chunks_override = parse_env_nonnegative_int(_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS_ENV)
    progress_seconds_override = parse_env_nonnegative_int(_SPARSE_PASS2_GROUP_PROGRESS_SECONDS_ENV)
    group_progress_chunks = (
        _DEFAULT_PASS2_GROUP_PROGRESS_CHUNKS
        if progress_chunks_override is None
        else int(progress_chunks_override)
    )
    group_progress_seconds = (
        _DEFAULT_PASS2_GROUP_PROGRESS_SECONDS
        if progress_seconds_override is None
        else int(progress_seconds_override)
    )

    def _mark_bucket_group_chunk_done(bucket_size: int, image_count: int) -> None:
        nonlocal group_completed_chunks, group_completed_images, group_last_progress_t
        group_completed_chunks += 1
        group_completed_images += int(image_count)
        if group_t0 is None:
            return
        group_chunks, group_images = bucket_group_stats[int(bucket_size)]
        if group_chunks < 100:
            return
        now = time.time()
        chunk_due = group_progress_chunks > 0 and group_completed_chunks % group_progress_chunks == 0
        time_due = (
            group_progress_seconds > 0
            and group_last_progress_t is not None
            and now - group_last_progress_t >= float(group_progress_seconds)
        )
        if not (chunk_due or time_due):
            return
        group_wall = now - group_t0
        logger.info(
            "Sparse pass-2 bucket group progress: bucket_size=%d chunks=%d/%d images=%d/%d "
            "wall=%.1fs images/s=%.1f",
            int(bucket_size),
            group_completed_chunks,
            group_chunks,
            group_completed_images,
            group_images,
            group_wall,
            group_completed_images / max(group_wall, 1e-9),
        )
        group_last_progress_t = now

    for bucket_meta in buckets:
        bucket_arrays = _build_bucket_arrays(
            bucket_meta,
            per_image_inputs,
            n_fine_trans,
        )
        image_indices = bucket_arrays["image_indices"]
        bucket_size = int(bucket_arrays["bucket_size"])
        dump_this_bucket = bool(
            dump_pass2_operands
            and _pass2_dump_requested_for_bucket(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
            )
        )
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
            group_completed_chunks = 0
            group_completed_images = 0
            group_last_progress_t = group_t0
        batch = int(image_indices.shape[0])

        # Fetch images (the dataset may reorder; we reorder our padded arrays
        # to match.)
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
        batch_data = jnp.asarray(batch_data)
        # Reorder bucket arrays to match fetched_indices
        if not np.array_equal(np.asarray(fetched_indices), image_indices):
            if bucket_arrays["mstep_rotations"] is bucket_arrays["rotations"]:
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
                mstep_rotations = rotations
            else:
                (
                    rotations,
                    mstep_rotations,
                    rotation_indices,
                    log_prior,
                    candidate_mask,
                    parent_map_padded,
                    actual_counts,
                ) = _reorder_to_indices(
                    np.asarray(fetched_indices),
                    image_indices,
                    bucket_arrays["rotations"],
                    bucket_arrays["mstep_rotations"],
                    bucket_arrays["rotation_indices"],
                    bucket_arrays["log_prior"],
                    bucket_arrays["candidate_mask"],
                    bucket_arrays["parent_map"],
                    bucket_arrays["actual_counts"],
                )
            image_indices = np.asarray(fetched_indices)
        else:
            rotations = bucket_arrays["rotations"]
            mstep_rotations = bucket_arrays["mstep_rotations"]
            rotation_indices = bucket_arrays["rotation_indices"]
            log_prior = bucket_arrays["log_prior"]
            candidate_mask = bucket_arrays["candidate_mask"]
            parent_map_padded = bucket_arrays["parent_map"]
            actual_counts = bucket_arrays["actual_counts"]
        target_particle_rows = (
            bpref_diagnostics._bpref_contribution_target_rows(experiment_dataset, image_indices)
            if device_signature_requested
            else np.empty((0,), dtype=np.int64)
        )
        bucket_diagnostic_modes = bpref_diagnostics._resolve_bpref_bucket_diagnostic_modes(
            device_signature_requested=device_signature_requested,
            contribution_diagnostics_active=contribution_diagnostics_active,
            target_particle_rows=target_particle_rows,
            high_precision_operand_bundle_requested=scoped_diagnostic_flags[
                "high_precision_operand_bundle"
            ],
        )
        bucket_device_signature_requested = bucket_diagnostic_modes[
            "device_signature_requested"
        ]
        bucket_contribution_diagnostics_active = bucket_diagnostic_modes[
            "contribution_diagnostics_active"
        ]
        bucket_shadow_only_mode = bucket_diagnostic_modes["shadow_only"]
        bucket_group_ids = (
            jnp.asarray(group_ids_np[image_indices], dtype=jnp.int32)
            if group_ids_np is not None
            else None
        )
        bucket_scale_for_stats = (
            jnp.asarray(np.asarray(scale_corrections, dtype=precision_policy.score_real_dtype)[image_indices])
            if scale_corrections is not None
            else jnp.ones(batch, dtype=precision_policy.score_real_dtype)
        )

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
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=precision_policy.score_real_dtype)
        else:
            bucket_translation_prior = jnp.asarray(
                fine_translation_prior_2d[image_indices], dtype=precision_policy.score_real_dtype
            )

        contribution_preprocess_operands = None
        high_precision_operand_bundle = bucket_diagnostic_modes[
            "high_precision_operand_bundle"
        ]
        if high_precision_operand_bundle:
            diagnostic_preprocess_operands = prepare_batch_preprocess_operands(
                experiment_dataset,
                batch_data,
                image_indices,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
            )
            contribution_preprocess_operands = bpref_diagnostics.build_bpref_preprocess_capture(
                experiment_dataset,
                image_shape,
                diagnostic_preprocess_operands,
                batch=batch,
                score_with_masked_images=score_with_masked_images,
            )

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
            direct_score_input,
            direct_preprocessed_score_input,
            direct_pixel_correction,
            direct_preprocess_normalization_factors,
            direct_integer_pre_shifts,
            direct_batch_image_corrections,
            direct_batch_scale_corrections,
            direct_inverse_noise_half,
            direct_ctf_rfloat_half,
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
            recon_window_indices=recon_window_indices,
            translation_phases_half=translation_phases_half,
            relion_score_translation_angles=relion_score_translation_angles,
            return_windowed_shifted=windowed_prepare,
            relion_exact_normalized_cc_operands=relion_exact_fine_normalized_cc,
            relion_exact_bpref_operands=relion_exact_bpref_operands,
        )
        if use_window:
            direct_inverse_noise_score = (
                None
                if direct_inverse_noise_half is None
                else direct_inverse_noise_half[jnp.asarray(window_indices, dtype=jnp.int32)]
            )
            direct_ctf_rfloat_score = (
                None
                if direct_ctf_rfloat_half is None
                else direct_ctf_rfloat_half[:, jnp.asarray(window_indices, dtype=jnp.int32)]
            )
            direct_ctf_rfloat_recon = (
                None
                if direct_ctf_rfloat_half is None
                else direct_ctf_rfloat_half[
                    :, jnp.asarray(recon_window_indices, dtype=jnp.int32)
                ]
            )
        else:
            direct_inverse_noise_score = direct_inverse_noise_half
            direct_ctf_rfloat_score = direct_ctf_rfloat_half
            direct_ctf_rfloat_recon = direct_ctf_rfloat_half
        relion_highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        translated_wavg_norm = bool(
            accumulate_noise
            and current_size is not None
            and parse_env_flag(_RELION_TRANSLATED_WAVG_NORM_ENV, default=False)
        )
        raw_translated_wavg_for_norm = None
        if translated_wavg_norm:
            if relion_score_translation_angles is None:
                raise ValueError(
                    "translated Wavg norm parity requires RELION translation angles"
                )
            raw_translated_wavg_for_norm = _relion_cuda_translate_wavg_norm_images(
                processed_score_half_for_noise,
                relion_score_translation_angles,
                window_indices,
                image_shape,
            )
        raw_translated_wavg_for_atomic = None
        raw_translated_wavg_rectangle = None
        relion_wavg_rectangle = None
        diagnostic_wavg_atomic_capture = bool(
            accumulate_noise
            and parse_env_flag(
                "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS",
                default=False,
            )
        )
        relion_wavg_atomic_scale_aa = bool(
            accumulate_noise
            and (
                noise_scale_correction_aa_total is not None
                or diagnostic_wavg_atomic_capture
            )
            and parse_env_flag(
                _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
                default=_fresh_k1_direct_noise_default(
                    preserve_bpref_particle_order=preserve_bpref_particle_order,
                    relion_exact_bpref_operands=relion_exact_bpref_operands,
                ),
            )
        )
        relion_wavg_atomic_direct_noise, relion_wavg_atomic_direct_norm = (
            _relion_wavg_direct_modes(
                accumulate_noise=bool(accumulate_noise),
                scale_groups_available=noise_scale_correction_aa_total is not None,
                scale_aa_enabled=bool(relion_wavg_atomic_scale_aa),
                direct_noise_only_default=_fresh_k1_direct_noise_default(
                    preserve_bpref_particle_order=preserve_bpref_particle_order,
                    relion_exact_bpref_operands=relion_exact_bpref_operands,
                ),
            )
        )
        if relion_wavg_atomic_direct_noise and current_size is None:
            raise ValueError("direct Wavg noise replacement requires current_size")
        if relion_wavg_atomic_scale_aa:
            if relion_score_translation_angles is None:
                raise ValueError(
                    "Wavg atomic parity requires RELION translation angles"
                )
            if current_size is None:
                raise ValueError("Wavg atomic parity requires current_size")
            relion_wavg_rectangle = _make_relion_wavg_rectangle(
                image_shape,
                current_size,
                recon_window_indices,
                reconstruction_current_size=mstep_current_size,
            )
            raw_translated_wavg_rectangle = _relion_cuda_translate_wavg_norm_images(
                processed_score_half_for_noise,
                relion_score_translation_angles,
                relion_wavg_rectangle.centered_indices,
                image_shape,
            )
            raw_translated_wavg_for_atomic = raw_translated_wavg_rectangle[
                :, :, relion_wavg_rectangle.exact_positions
            ]
        if relion_wavg_atomic_direct_noise:
            direct_noise_log_key = int(current_size)
            if direct_noise_log_key not in _relion_wavg_direct_noise_log_keys:
                _relion_wavg_direct_noise_log_keys.add(direct_noise_log_key)
                logger.info(
                    "Sparse pass-2 RELION Wavg parity: issuing the full "
                    "%d-pixel FFTW rectangle and replacing current-size noise "
                    "shells [0, %d] with direct residual atomics; per-particle "
                    "norm mode=%s",
                    int(relion_wavg_rectangle.centered_indices.size),
                    int(current_size // 2),
                    "direct" if relion_wavg_atomic_direct_norm else "production-algebraic",
                )

        # Window gather (if applicable)
        if use_window:
            ctf2_over_nv_score = ctf2_over_nv_half if windowed_prepare else ctf2_over_nv_half[:, window_indices]
            shifted_corrected_score = (
                shifted_corrected_score_half if windowed_prepare else shifted_corrected_score_half[:, window_indices]
            )
            if score_only:
                shifted_score = None
                shifted_recon = None
                ctf2_over_nv_recon = None
                shifted_noise = None
            elif windowed_prepare:
                shifted_score = shifted_score_half
                shifted_recon = shifted_recon_half
                ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
                shifted_noise = shifted_score_half_with_dc
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
        flat_backproject_rotations = (
            flat_rotations
            if mstep_rotations is rotations
            else flatten_bucket_rotations(jnp.asarray(mstep_rotations))
        )
        rotation_chunk_size = None
        identity_full_projection_cache_rows = False
        if projection_cache is not None:
            rotation_indices_np = np.asarray(rotation_indices, dtype=np.int64)
            cache_rows = int(projection_cache["score"].shape[0])
            identity_full_projection_cache_rows = (
                int(batch) == 1
                and rotation_indices_np.shape == (1, cache_rows)
                and int(bucket_size) == cache_rows
                and np.array_equal(rotation_indices_np[0], np.arange(cache_rows, dtype=np.int64))
            )
        if use_window and projection_cache is None and not dump_this_bucket and not score_only:
            rotation_chunk_size = _projection_rotation_chunk_size(
                batch_size=batch,
                n_score_pixels=n_windowed,
                n_recon_pixels=0 if score_only else n_recon_windowed,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_recon_noise=not score_only,
                max_gather_bytes=max_projection_gather_bytes,
                max_projected_rotations=max_projected_rotations_per_projection_call,
            )
        elif use_window and projection_cache is not None and not dump_this_bucket and not score_only:
            rotation_chunk_size = _cached_score_rotation_chunk_size_for_pass(bucket_size)
        rotation_chunk_size = bpref_diagnostics._guard_bpref_target_rotation_chunking(
            rotation_chunk_size,
            bucket_size=bucket_size,
            target_particle_rows=target_particle_rows,
        )
        if bucket_device_signature_requested:
            logger.info(
                "Scoped BPref device capture preserves production rotation planning: "
                "target_particles=%d bucket_size=%d rotation_chunk_size=%s naturally_unchunked=true",
                int(target_particle_rows.size),
                int(bucket_size),
                "unset" if rotation_chunk_size is None else str(int(rotation_chunk_size)),
            )
        if (
            rotation_chunk_size is not None
            and int(rotation_chunk_size) < bucket_size
            and use_relion_x_half_mstep
            and diagnostic_per_particle_launches
            and not device_signature_requested
        ):
            raise RuntimeError(
                "RELION per-particle launch diagnostic does not support rotation-chunked pass 2"
            )
        if rotation_chunk_size is not None and int(rotation_chunk_size) < bucket_size:
            rotation_chunk_size = max(1, int(rotation_chunk_size))
            if projection_cache is not None:
                log_key = ("single-cached", int(bucket_size), int(batch), int(rotation_chunk_size))
                if log_key not in _cached_score_chunk_log_keys:
                    _cached_score_chunk_log_keys.add(log_key)
                    logger.info(
                        "Sparse pass-2 cached rotation chunking: bucket_size=%d batch=%d chunk_size=%d",
                        bucket_size,
                        batch,
                        rotation_chunk_size,
                    )
            else:
                logger.info(
                    "Sparse pass-2 rotation chunking: bucket_size=%d batch=%d chunk_size=%d "
                    "max_projection_gather_bytes=%.2f GiB",
                    bucket_size,
                    batch,
                    rotation_chunk_size,
                    max_projection_gather_bytes / float(1024**3),
                )
            chunk_ranges = [
                (start, min(start + rotation_chunk_size, bucket_size))
                for start in range(0, bucket_size, rotation_chunk_size)
            ]
            shifted_corrected_score_split = shifted_corrected_score.reshape(batch, n_fine_trans, -1)
            direct_half_weights = half_weights_windowed

            def _score_rotation_chunk(start, stop, *, need_recon, raw_diff2=False, min_diff2=None):
                rot_count = int(stop - start)
                if projection_cache is None:
                    rotations_chunk = jnp.asarray(rotations[:, start:stop])
                    flat_rotations_chunk = flatten_bucket_rotations(rotations_chunk)
                    projection_kwargs = _projection_kwargs_for_relion_score_window(
                        window_spec.projection_kwargs(return_abs2=False),
                        use_relion_projector=use_relion_projector,
                        current_size=current_size,
                    )
                    projection_kwargs["mask_current_image_disk"] = bool(
                        projection_mask_current_image_disk
                    )
                    score_flat, recon_flat, recon_abs2_flat = _compute_sparse_pass2_windowed_projections_block(
                        mean_for_proj,
                        flat_rotations_chunk,
                        image_shape,
                        proj_volume_shape,
                        disc_type,
                        score_indices=window_indices,
                        recon_indices=recon_window_indices if need_recon else None,
                        max_projected_rotations=max_projected_rotations_per_projection_call,
                        output_complex_dtype=precision_policy.score_complex_dtype,
                        output_abs2_dtype=precision_policy.score_real_dtype,
                        relion_projector_half=relion_projector_half,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                        **projection_kwargs,
                    )
                    proj_chunk = score_flat.reshape(batch, rot_count, n_windowed)
                else:
                    if identity_full_projection_cache_rows:
                        # Full-support first-iteration K=1 buckets use cache
                        # order directly. Slice the retained cache instead of
                        # gathering a duplicate ``(1, R, N)`` projection slab.
                        proj_chunk = projection_cache["score"][start:stop][jnp.newaxis, :, :]
                    else:
                        rotation_indices_chunk = jnp.asarray(rotation_indices[:, start:stop], dtype=jnp.int32)
                        proj_chunk = projection_cache["score"][rotation_indices_chunk]
                if relion_firstiter_score_mode == "normalized_cc":
                    if raw_diff2:
                        raise ValueError("normalized-CC scoring has no raw Gaussian diff2 tensor")
                    score_args = (
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        proj_chunk,
                        direct_half_weights,
                        jnp.asarray(candidate_mask[:, start:stop, :]),
                    )
                    if relion_exact_fine_normalized_cc:
                        score_chunk = _score_pass2_bucket_relion_gpu_normalized_cc(
                            *score_args,
                            relion_score_full_to_compact,
                        )
                    else:
                        score_chunk = _score_pass2_bucket_normalized_cc(*score_args)
                elif use_exact_relion_gaussian:
                    if raw_diff2:
                        score_chunk = _score_pass2_bucket_relion_gpu_diff2_raw(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_chunk,
                            direct_half_weights,
                            relion_score_full_to_compact,
                            relion_highres_xi2_half,
                            use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                        )
                    else:
                        score_chunk = _score_pass2_bucket_relion_gpu_diff2(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_chunk,
                            direct_half_weights,
                            jnp.asarray(log_prior[:, start:stop]),
                            bucket_translation_prior,
                            jnp.asarray(candidate_mask[:, start:stop, :]),
                            relion_score_full_to_compact,
                            min_diff2,
                            relion_highres_xi2_half,
                            use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                        )
                else:
                    if raw_diff2:
                        raise ValueError("algebraic Gaussian scoring has no raw RELION diff2 tensor")
                    score_chunk = _score_pass2_bucket_gaussian_algebraic(
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        proj_chunk,
                        direct_half_weights,
                        jnp.asarray(log_prior[:, start:stop]),
                        bucket_translation_prior,
                        jnp.asarray(candidate_mask[:, start:stop, :]),
                    )
                if not need_recon:
                    return score_chunk, proj_chunk, None, None
                if projection_cache is None:
                    proj_noise = recon_flat.reshape(batch, rot_count, n_recon_windowed)
                    proj_abs2_noise = recon_abs2_flat.reshape(batch, rot_count, n_recon_windowed)
                else:
                    if identity_full_projection_cache_rows:
                        proj_noise = projection_cache["recon"][start:stop][jnp.newaxis, :, :]
                        proj_abs2_noise = projection_cache["recon_abs2"][start:stop][jnp.newaxis, :, :]
                    else:
                        rotation_indices_chunk = jnp.asarray(rotation_indices[:, start:stop], dtype=jnp.int32)
                        proj_noise = projection_cache["recon"][rotation_indices_chunk]
                        proj_abs2_noise = projection_cache["recon_abs2"][rotation_indices_chunk]
                proj_noise, proj_abs2_noise = precision_policy.cast_local_noise_projection_scores(
                    proj_noise,
                    proj_abs2_noise,
                )
                return score_chunk, proj_chunk, proj_noise, proj_abs2_noise

            global_log_z = jnp.full((batch,), -jnp.inf, dtype=jnp.float64)
            global_best_log_score = jnp.full((batch,), -jnp.inf, dtype=jnp.float64)
            global_best_argmax = jnp.zeros((batch,), dtype=jnp.int32)
            global_min_diff2 = None
            cached_raw_diff2_chunks = None
            if use_exact_relion_gaussian:
                raw_diff2_cache_bytes = _exact_raw_diff2_cache_estimated_bytes(
                    batch,
                    bucket_size,
                    n_fine_trans,
                    dtype=precision_policy.score_real_dtype,
                )
                if _exact_raw_diff2_cache_fits_budget(
                    raw_diff2_cache_bytes,
                    exact_raw_diff2_cache_limit_bytes,
                ):
                    cached_raw_diff2_chunks = []
                    if not exact_raw_diff2_cache_admission_logged:
                        logger.info(
                            "Sparse pass-2 exact raw-diff2 reuse enabled: "
                            "estimated=%.2f MiB cap=%.2f MiB bucket_size=%d batch=%d chunks=%d",
                            raw_diff2_cache_bytes / float(1024**2),
                            exact_raw_diff2_cache_limit_bytes / float(1024**2),
                            bucket_size,
                            batch,
                            len(chunk_ranges),
                        )
                        exact_raw_diff2_cache_admission_logged = True
                global_min_diff2 = jnp.full(
                    (batch,),
                    jnp.inf,
                    dtype=precision_policy.score_real_dtype,
                )
                for start, stop in chunk_ranges:
                    raw_diff2_chunk = _score_rotation_chunk(
                        start,
                        stop,
                        need_recon=False,
                        raw_diff2=True,
                    )[0]
                    chunk_min = _relion_cuda_fine_partition_diff2_min_or_inf(
                        raw_diff2_chunk,
                        jnp.asarray(candidate_mask[:, start:stop, :]),
                    )
                    global_min_diff2 = jnp.minimum(global_min_diff2, chunk_min)
                    if cached_raw_diff2_chunks is None:
                        del raw_diff2_chunk
                    else:
                        cached_raw_diff2_chunks.append(raw_diff2_chunk)
                global_min_diff2 = jnp.where(
                    jnp.isfinite(global_min_diff2),
                    global_min_diff2,
                    jnp.asarray(0.0, dtype=precision_policy.score_real_dtype),
                )
            for chunk_idx, (start, stop) in enumerate(chunk_ranges):
                if cached_raw_diff2_chunks is None:
                    scores_chunk = _score_rotation_chunk(
                        start,
                        stop,
                        need_recon=False,
                        min_diff2=global_min_diff2,
                    )[0]
                else:
                    scores_chunk = _score_pass2_bucket_relion_gpu_diff2_from_raw(
                        cached_raw_diff2_chunks[chunk_idx],
                        jnp.asarray(log_prior[:, start:stop]),
                        bucket_translation_prior,
                        jnp.asarray(candidate_mask[:, start:stop, :]),
                        global_min_diff2,
                    )
                    cached_raw_diff2_chunks[chunk_idx] = None
                chunk_log_z, chunk_best_log_score, chunk_best_argmax, _ = _normalize_pass2_bucket_score_only(
                    scores_chunk,
                )
                chunk_best_log_score = chunk_best_log_score.astype(global_best_log_score.dtype)
                chunk_log_z = jnp.where(
                    jnp.isfinite(chunk_best_log_score),
                    chunk_log_z,
                    -jnp.inf,
                )
                chunk_global_argmax = chunk_best_argmax + int(start) * int(n_fine_trans)
                take_chunk_best = chunk_best_log_score > global_best_log_score
                global_best_log_score = jnp.where(take_chunk_best, chunk_best_log_score, global_best_log_score)
                global_best_argmax = jnp.where(take_chunk_best, chunk_global_argmax, global_best_argmax)
                global_log_z = jnp.logaddexp(global_log_z, chunk_log_z.astype(global_log_z.dtype))
                del scores_chunk
            del cached_raw_diff2_chunks

            score_log_offset_jax = (
                _relion_cuda_fine_log_evidence_offset(global_min_diff2).astype(jnp.float64)
                if use_exact_relion_gaussian
                else -0.5 * jnp.squeeze(batch_norm, axis=1).astype(jnp.float64)
            )
            if normalization_log_z_np is not None:
                bucket_log_z = jnp.asarray(normalization_log_z_np[image_indices], dtype=jnp.float64)
                if use_exact_relion_gaussian:
                    bucket_log_z = bucket_log_z - score_log_offset_jax
                local_score_log_z = None
            elif normalization_other_score_log_z_np is not None:
                local_score_log_z = global_log_z
                bucket_other_log_z = jnp.asarray(
                    normalization_other_score_log_z_np[image_indices],
                    dtype=jnp.float64,
                )
                if not use_exact_relion_gaussian:
                    bucket_log_z = jnp.logaddexp(local_score_log_z, bucket_other_log_z)
                else:
                    bucket_log_z_absolute = jnp.logaddexp(
                        local_score_log_z + score_log_offset_jax,
                        bucket_other_log_z,
                    )
                    bucket_log_z = bucket_log_z_absolute - score_log_offset_jax
            else:
                bucket_log_z = global_log_z
                local_score_log_z = None

            if return_score_log_z_only:
                log_score_offset = np.asarray(score_log_offset_jax, dtype=np.float64)
                log_Z_np = np.asarray(global_log_z, dtype=np.float64)
                for row, image_idx in enumerate(image_indices.tolist()):
                    if np.isfinite(log_Z_np[row]):
                        log_evidence[image_idx] = float(log_Z_np[row] + log_score_offset[row])
                        score_log_z[image_idx] = float(
                            log_Z_np[row] + log_score_offset[row]
                            if use_exact_relion_gaussian
                            else log_Z_np[row]
                        )
                    else:
                        log_evidence[image_idx] = -np.inf
                        score_log_z[image_idx] = -np.inf
                _mark_bucket_group_chunk_done(bucket_size, batch)
                continue

            global_max_posterior = jnp.exp(global_best_log_score - bucket_log_z)
            global_max_posterior = jnp.where(
                jnp.isfinite(global_max_posterior),
                global_max_posterior,
                0.0,
            )
            if winner_take_all:
                global_max_posterior = jnp.where(
                    jnp.isfinite(global_best_log_score),
                    jnp.ones_like(global_max_posterior),
                    jnp.zeros_like(global_max_posterior),
                )
            bucket_original_indices = original_image_indices(
                experiment_dataset,
                image_indices,
            )
            capture_chunked_particle_count = (
                compact_capture_requested_particle_count(
                    int(bpref_diagnostics._bpref_contribution_context["iteration"]),
                    bucket_original_indices,
                )
                if not score_only
                else 0
            )
            capture_chunked_bucket = capture_chunked_particle_count > 0
            contribution_chunked_bucket = bool(
                not score_only and bucket_contribution_diagnostics_active
            )
            membership_chunked_bucket = bool(
                not score_only and membership_diagnostics_active
            )
            if capture_chunked_bucket:
                require_chunked_capture_capacity(
                    capture_chunked_particle_count,
                    bucket_size,
                    n_fine_trans,
                )
            capture_score_chunks = [] if capture_chunked_bucket else None
            capture_prob_chunks = [] if capture_chunked_bucket else None
            capture_reconstruction_mask_chunks = [] if capture_chunked_bucket else None
            contribution_score_chunks = [] if contribution_chunked_bucket else None
            contribution_preprior_score_chunks = [] if contribution_chunked_bucket else None
            contribution_prob_chunks = [] if contribution_chunked_bucket else None
            contribution_reconstruction_prob_chunks = [] if contribution_chunked_bucket else None
            contribution_reconstruction_mask_chunks = [] if contribution_chunked_bucket else None
            contribution_summed_chunks = [] if contribution_chunked_bucket else None
            contribution_ctf_prob_chunks = [] if contribution_chunked_bucket else None
            contribution_authoritative_summed_chunks = (
                [] if contribution_chunked_bucket and bucket_shadow_only_mode else None
            )
            contribution_authoritative_ctf_prob_chunks = (
                [] if contribution_chunked_bucket and bucket_shadow_only_mode else None
            )
            membership_candidate_count_chunks = [] if membership_chunked_bucket else None
            membership_posterior_mass_chunks = [] if membership_chunked_bucket else None
            membership_reconstruction_mass_chunks = [] if membership_chunked_bucket else None
            membership_significant_count_chunks = [] if membership_chunked_bucket else None
            chunk_shadow_score_bitwise_equal = False

            ctf_probs = None
            reconstruction_mask_chunks = None
            reconstruction_prob_chunks = None
            chunk_reconstruction_sum_weight = None
            chunk_reconstruction_threshold = None
            if use_relion_fine_mstep_prune and not score_only:
                score_chunks = [
                    _score_rotation_chunk(
                        start,
                        stop,
                        need_recon=False,
                        min_diff2=global_min_diff2,
                    )[0]
                    for start, stop in chunk_ranges
                ]
                if use_relion_f32_fine_posterior:
                    score_flat_chunks = []
                    for scores_chunk in score_chunks:
                        score_flat_chunks.append(scores_chunk.reshape(batch, -1))
                    all_scores_flat = jnp.concatenate(score_flat_chunks, axis=1)
                    (
                        reconstruction_probs_flat,
                        reconstruction_mask_flat,
                        _reconstruction_n_significant,
                        chunk_reconstruction_sum_weight,
                        chunk_reconstruction_threshold,
                    ) = _relion_f32_fine_reconstruction_probs(
                        all_scores_flat,
                        adaptive_fraction=float(adaptive_fraction),
                    )
                else:
                    mask_flat_chunks = []
                    for scores_chunk in score_chunks:
                        normalize_log_z = bucket_log_z
                        (
                            _chunk_log_z,
                            probs_chunk,
                            _chunk_best_log_score,
                            _chunk_best_argmax,
                            _chunk_max_posterior,
                        ) = _normalize_pass2_bucket_with_log_z(scores_chunk, normalize_log_z)
                        mask_flat_chunks.append(probs_chunk.reshape(batch, -1))
                    all_probs_flat = jnp.concatenate(mask_flat_chunks, axis=1)
                    chunk_reconstruction_sum_weight = jnp.sum(
                        all_probs_flat,
                        axis=1,
                        dtype=jnp.float64,
                    )
                    reconstruction_mask_flat, _reconstruction_n_significant = _find_significant_mask_full_sort(
                        all_probs_flat,
                        float(adaptive_fraction),
                        -1,
                    )
                    chunk_reconstruction_threshold = jnp.min(
                        jnp.where(
                            reconstruction_mask_flat,
                            all_probs_flat,
                            jnp.inf,
                        ),
                        axis=1,
                    )
                    chunk_reconstruction_threshold = jnp.where(
                        jnp.isfinite(chunk_reconstruction_threshold),
                        chunk_reconstruction_threshold,
                        0.0,
                    )
                    reconstruction_probs_flat = None
                reconstruction_mask_chunks = []
                if reconstruction_probs_flat is not None:
                    reconstruction_prob_chunks = []
                offset = 0
                for start, stop in chunk_ranges:
                    width = int(stop - start) * int(n_fine_trans)
                    reconstruction_mask_chunks.append(
                        reconstruction_mask_flat[:, offset : offset + width].reshape(
                            batch,
                            int(stop - start),
                            n_fine_trans,
                        )
                    )
                    if reconstruction_prob_chunks is not None:
                        reconstruction_prob_chunks.append(
                            reconstruction_probs_flat[:, offset : offset + width].reshape(
                                batch,
                                int(stop - start),
                                n_fine_trans,
                            )
                        )
                    offset += width
                del score_chunks
            if accumulate_noise:
                bucket_block_noise_shells = (
                    np.zeros(n_shells, dtype=np.float64)
                    if relion_wavg_atomic_direct_noise
                    else None
                )
                relion_wavg_atomic_scale_triplet_pixels = (
                    jnp.zeros(
                        (
                            batch,
                            int(relion_wavg_rectangle.centered_indices.size),
                            3,
                        ),
                        dtype=jnp.float32,
                    )
                    if relion_wavg_atomic_scale_aa
                    else None
                )
                relion_wavg_atomic_scale_xa_pixels_np = None
                relion_wavg_atomic_scale_aa_pixels_np = None
                relion_wavg_atomic_diff2_pixels_np = None
                if translation_sqdist_ang is not None or translated_wavg_norm:
                    chunk_translation_posterior_total = np.zeros((batch, n_fine_trans), dtype=np.float64)
                chunk_support_mass = np.zeros((batch,), dtype=np.float64)
                shifted_noise_split = (
                    shifted_noise.reshape(batch, n_fine_trans, -1)
                    if half_spectrum_scoring
                    else shifted_score.reshape(batch, n_fine_trans, -1)
                )
                chunked_scale_aa_target_rows = (
                    pass2_diagnostics._pass2_dump_target_rows(
                        experiment_dataset=experiment_dataset,
                        image_indices=image_indices,
                        current_size=current_size,
                    )
                    if parse_env_flag(
                        "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS",
                        default=False,
                    )
                    else np.empty((0,), dtype=np.int64)
                )
                if chunked_scale_aa_target_rows.size:
                    chunked_scale_aa_posterior_mass = []
                    chunked_scale_aa_proj_abs2_sum = []
                    chunked_scale_aa_ctf_probs_raw_sum = []
                    chunked_scale_xa_per_pixel = []
                    chunked_scale_xa_per_image = []
                    chunked_norm_a2_per_pixel = []
                    chunked_norm_a2_per_image = []
                    chunked_norm_xa_per_pixel = []
                    chunked_norm_xa_per_image = []
                    chunked_scale_aa_before_scale_per_pixel = []
                    chunked_scale_aa_per_pixel = []
                    chunked_scale_aa_per_image = []
                    chunked_scale_aa_posterior_probs = []
                    chunked_scale_aa_rotation_matrices = []
                    chunked_scale_aa_feature_per_shell = []
                    scale_shell_indices_np = np.asarray(
                        shell_indices_noise,
                        dtype=np.int32,
                    ).reshape(-1)
                    scale_mask_np = np.asarray(
                        scale_correction_pixel_mask,
                        dtype=bool,
                    ).reshape(-1)
                    chunked_scale_aa_feature_shell_ids = np.unique(
                        scale_shell_indices_np[
                            scale_mask_np & (scale_shell_indices_np >= 0)
                        ]
                    ).astype(np.int32)
            if not score_only:
                shifted_recon_split = shifted_recon.reshape(batch, n_fine_trans, -1)
                mstep_window_indices = relion_x_half_recon_indices if use_relion_x_half_mstep else recon_window_indices

            for chunk_idx, (start, stop) in enumerate(chunk_ranges):
                _rescored_chunk, proj_half_chunk, proj_for_noise_chunk, proj_abs2_for_noise_chunk = (
                    _score_rotation_chunk(
                        start,
                        stop,
                        need_recon=not score_only,
                        min_diff2=global_min_diff2,
                    )
                )
                scores_chunk = _rescored_chunk
                normalize_log_z = bucket_log_z
                _chunk_log_z, probs, _chunk_best_log_score, _chunk_best_argmax, _chunk_max_posterior = (
                    _normalize_pass2_bucket_with_log_z(scores_chunk, normalize_log_z)
                )
                if winner_take_all:
                    probs = _winner_take_all_bucket_probs_from_global_argmax(
                        scores_chunk,
                        global_best_argmax,
                        jnp.asarray(start, dtype=jnp.int32),
                        global_best_log_score.astype(scores_chunk.real.dtype),
                    )
                if contribution_chunked_bucket:
                    contribution_score_chunks.append(scores_chunk)
                    if relion_firstiter_score_mode == "normalized_cc":
                        contribution_preprior_score_chunks.append(scores_chunk)
                    elif use_exact_relion_gaussian:
                        contribution_raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_half_chunk,
                            direct_half_weights,
                            relion_score_full_to_compact,
                            relion_highres_xi2_half,
                            use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                        )
                        contribution_preprior_score_chunks.append(
                            _relion_cuda_fine_diff2_to_scores(
                                contribution_raw_diff2,
                                jnp.zeros_like(jnp.asarray(log_prior[:, start:stop]))[
                                    :,
                                    :,
                                    None,
                                ],
                                jnp.zeros_like(bucket_translation_prior)[:, None, :],
                                jnp.asarray(candidate_mask[:, start:stop, :]),
                                min_diff2=global_min_diff2,
                            )
                        )
                    else:
                        _, contribution_preprior_scores = (
                            _score_pass2_bucket_gaussian_algebraic_components(
                                shifted_corrected_score_split,
                                ctf2_over_nv_score,
                                proj_half_chunk,
                                direct_half_weights,
                                jnp.asarray(log_prior[:, start:stop]),
                                bucket_translation_prior,
                                jnp.asarray(candidate_mask[:, start:stop, :]),
                            )
                        )
                        contribution_preprior_score_chunks.append(
                            contribution_preprior_scores
                        )
                    contribution_prob_chunks.append(probs)
                if capture_chunked_bucket:
                    capture_score_chunks.append(scores_chunk)
                    capture_prob_chunks.append(probs)
                    if winner_take_all:
                        capture_reconstruction_mask_chunks.append(probs > 0)
                    elif use_relion_fine_mstep_prune:
                        if reconstruction_mask_chunks is None:
                            raise RuntimeError("chunked fine-prune capture has no reconstruction support")
                        capture_reconstruction_mask_chunks.append(
                            reconstruction_mask_chunks[chunk_idx]
                        )
                    else:
                        # Without fine M-step pruning production uses ``probs``
                        # on the entire candidate support, so candidate_mask is
                        # exactly the reconstruction-significance support.
                        capture_reconstruction_mask_chunks.append(
                            jnp.asarray(candidate_mask[:, start:stop, :])
                        )

                if not score_only:
                    mstep_probs = probs
                    if use_relion_fine_mstep_prune:
                        if reconstruction_prob_chunks is None:
                            mstep_probs = jnp.where(reconstruction_mask_chunks[chunk_idx], probs, 0.0)
                        else:
                            mstep_probs = reconstruction_prob_chunks[chunk_idx]
                    if membership_chunked_bucket:
                        membership_candidate_count_chunks.append(
                            np.asarray(
                                jnp.sum(
                                    jnp.asarray(candidate_mask[:, start:stop, :]),
                                    axis=-1,
                                    dtype=jnp.int32,
                                ),
                                dtype=np.int32,
                            )
                        )
                        membership_posterior_mass_chunks.append(
                            np.asarray(jnp.sum(probs, axis=-1))
                        )
                        membership_reconstruction_mass_chunks.append(
                            np.asarray(jnp.sum(mstep_probs, axis=-1))
                        )
                        membership_significant_count_chunks.append(
                            np.asarray(
                                jnp.sum(mstep_probs > 0, axis=-1, dtype=jnp.int32),
                                dtype=np.int32,
                            )
                        )
                    summed, ctf_probs = compute_local_mstep_sums(
                        mstep_probs,
                        shifted_recon_split,
                        ctf2_over_nv_recon,
                        relion_x_half=use_relion_x_half_mstep,
                        sequential_translation_reduction=use_sequential_translation_reduction,
                    )
                    if mstep_subtract_ctf_projection:
                        summed = subtract_projected_reference_from_sparse_mstep_sums(
                            summed,
                            mstep_probs,
                            proj_for_noise_chunk,
                            ctf2_over_nv_recon,
                        )
                    if contribution_chunked_bucket:
                        dump_summed = summed
                        dump_ctf_probs = ctf_probs
                        if bucket_shadow_only_mode:
                            shadow_scores = _score_rotation_chunk(
                                start,
                                stop,
                                need_recon=False,
                                min_diff2=global_min_diff2,
                            )[0]
                            bpref_diagnostics._require_bpref_shadow_exact(
                                "chunked score",
                                scores_chunk,
                                shadow_scores,
                            )
                            chunk_shadow_score_bitwise_equal = True
                            shadow_summed, shadow_ctf_probs = (
                                compute_local_mstep_sums(
                                    mstep_probs,
                                    shifted_recon_split,
                                    ctf2_over_nv_recon,
                                    relion_x_half=use_relion_x_half_mstep,
                                    sequential_translation_reduction=(
                                        diagnostic_sequential_translation_reduction
                                    ),
                                )
                            )
                            contribution_authoritative_summed_chunks.append(
                                summed
                            )
                            contribution_authoritative_ctf_prob_chunks.append(
                                ctf_probs
                            )
                            dump_summed = shadow_summed
                            dump_ctf_probs = shadow_ctf_probs
                        contribution_reconstruction_prob_chunks.append(mstep_probs)
                        if winner_take_all:
                            contribution_reconstruction_mask_chunks.append(probs > 0)
                        elif use_relion_fine_mstep_prune:
                            contribution_reconstruction_mask_chunks.append(
                                reconstruction_mask_chunks[chunk_idx]
                            )
                        else:
                            contribution_reconstruction_mask_chunks.append(
                                mstep_probs > 0
                            )
                        contribution_summed_chunks.append(dump_summed)
                        contribution_ctf_prob_chunks.append(dump_ctf_probs)
                    flat_chunk_rotations = flatten_bucket_rotations(jnp.asarray(mstep_rotations[:, start:stop]))
                    if use_window:
                        Ft_y_total = _accumulate_adjoint_block_chunked(
                            flatten_bucket_rows(summed),
                            flat_chunk_rotations,
                            Ft_y_total,
                            window_indices=mstep_window_indices,
                            use_windowed_adjoint=True,
                            image_shape=image_shape,
                            volume_shape=recon_volume_shape,
                            disc_type="linear_interp",
                            half_image=True,
                            half_volume=use_half_volume_mstep,
                            max_r=float(mstep_current_size // 2),
                            relion_x_half=use_relion_x_half_mstep,
                            max_block_bytes=max_adjoint_block_bytes,
                            log_label="single-y-window-chunk",
                        )
                        Ft_ctf_total = _accumulate_adjoint_block_chunked(
                            flatten_bucket_rows(ctf_probs),
                            flat_chunk_rotations,
                            Ft_ctf_total,
                            window_indices=mstep_window_indices,
                            use_windowed_adjoint=True,
                            image_shape=image_shape,
                            volume_shape=recon_volume_shape,
                            disc_type="linear_interp",
                            half_image=True,
                            half_volume=use_half_volume_mstep,
                            max_r=float(mstep_current_size // 2),
                            relion_x_half=use_relion_x_half_mstep,
                            max_block_bytes=max_adjoint_block_bytes,
                            log_label="single-ctf-window-chunk",
                        )

                if accumulate_noise:
                    noise_probs = mstep_probs if use_relion_fine_mstep_prune and not score_only else probs
                    if translation_sqdist_ang is not None or translated_wavg_norm:
                        chunk_translation_posterior_total += np.asarray(
                            jnp.sum(noise_probs, axis=1),
                            dtype=np.float64,
                        )
                    chunk_support_mass += np.asarray(jnp.sum(noise_probs, axis=(1, 2)), dtype=np.float64)
                    summed_masked_noise = compute_local_weighted_sums(noise_probs, shifted_noise_split)
                    if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                        logger.info(
                            "RECOVAR_NOISE_DTYPE_DEBUG: proj_for_noise_chunk=%s proj_abs2_for_noise_chunk=%s "
                            "summed_masked_noise=%s ctf_probs=%s noise_variance_for_noise=%s",
                            proj_for_noise_chunk.dtype,
                            proj_abs2_for_noise_chunk.dtype,
                            summed_masked_noise.dtype,
                            ctf_probs.dtype,
                            noise_variance_for_noise.dtype,
                        )
                    block_noise_shells, _, _ = _compute_noise_block_chunked(
                        flatten_bucket_rows(proj_for_noise_chunk),
                        flatten_bucket_rows(proj_abs2_for_noise_chunk),
                        flatten_bucket_rows(summed_masked_noise),
                        flatten_bucket_rows(ctf_probs),
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_shells,
                        max_block_bytes=max_noise_block_bytes,
                    )
                    if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                        logger.info(
                            "RECOVAR_NOISE_DTYPE_DEBUG: block_noise_shells=%s",
                            block_noise_shells.dtype,
                        )
                    block_noise_shells_np = np.asarray(
                        block_noise_shells,
                        dtype=np.float64,
                    )
                    if relion_wavg_atomic_direct_noise:
                        bucket_block_noise_shells += block_noise_shells_np
                    else:
                        noise_wsum_total += block_noise_shells_np
                    block_norm_residual = _compute_norm_residual_per_image(
                        proj_for_noise_chunk,
                        proj_abs2_for_noise_chunk,
                        summed_masked_noise,
                        ctf_probs,
                        noise_variance_for_noise,
                    )
                    if not relion_wavg_atomic_direct_norm:
                        noise_norm_correction_total[image_indices] += np.asarray(
                            block_norm_residual,
                            dtype=np.float64,
                        )
                    if noise_scale_correction_xa_total is not None:
                        scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                            proj_for_noise_chunk,
                            proj_abs2_for_noise_chunk,
                            summed_masked_noise,
                            ctf_probs,
                            noise_variance_for_noise,
                            bucket_scale_for_stats,
                            scale_correction_pixel_mask,
                        )
                        if relion_wavg_atomic_scale_aa:
                            from recovar.cuda_backproject import (
                                relion_wavg_rotation_atomic_triplet_add_f32,
                            )

                            if direct_ctf_rfloat_recon is None:
                                atomic_triplet_terms = _relion_wavg_atomic_triplet_terms(
                                    proj_for_noise_chunk,
                                    proj_abs2_for_noise_chunk,
                                    summed_masked_noise,
                                    ctf_probs,
                                    noise_variance_for_noise,
                                    bucket_scale_for_stats,
                                    raw_translated_wavg_for_atomic,
                                    noise_probs,
                                )
                            else:
                                atomic_triplet_terms = _relion_wavg_sequential_triplet_terms(
                                    proj_for_noise_chunk,
                                    direct_ctf_rfloat_recon,
                                    bucket_scale_for_stats,
                                    raw_translated_wavg_for_atomic,
                                    noise_probs,
                                )
                            atomic_triplet_terms = _relion_wavg_rectangle_triplet_terms(
                                atomic_triplet_terms,
                                raw_translated_wavg_rectangle,
                                noise_probs,
                                relion_wavg_rectangle.exact_positions,
                            )
                            relion_wavg_atomic_scale_triplet_pixels = (
                                relion_wavg_rotation_atomic_triplet_add_f32(
                                    atomic_triplet_terms,
                                    relion_wavg_atomic_scale_triplet_pixels,
                                )
                            )
                        if chunked_scale_aa_target_rows.size:
                            selected = jnp.asarray(
                                chunked_scale_aa_target_rows,
                                dtype=jnp.int32,
                            )
                            selected_proj_abs2 = jnp.asarray(proj_abs2_for_noise_chunk)[selected]
                            selected_proj = jnp.asarray(proj_for_noise_chunk)[selected]
                            selected_summed_masked = jnp.asarray(summed_masked_noise)[selected]
                            selected_ctf_probs = jnp.asarray(ctf_probs)[selected]
                            selected_noise = jnp.asarray(noise_variance_for_noise)
                            norm_ctf_has_mass = selected_ctf_probs != 0.0
                            norm_ctf_probs_raw = jnp.where(
                                norm_ctf_has_mass,
                                selected_ctf_probs * selected_noise[None, None, :],
                                0.0,
                            )
                            norm_a2_terms = jnp.where(
                                norm_ctf_has_mass,
                                selected_proj_abs2 * norm_ctf_probs_raw,
                                0.0,
                            )
                            norm_cross_terms = jnp.where(
                                selected_summed_masked != 0.0,
                                selected_proj * jnp.conj(selected_summed_masked),
                                0.0,
                            )
                            norm_xa_terms = (
                                selected_noise[None, None, :] * norm_cross_terms.real
                            )
                            scale_mask = jnp.asarray(
                                scale_correction_pixel_mask,
                                dtype=bool,
                            ).reshape(-1)
                            ctf_has_mass = (
                                (selected_ctf_probs != 0.0)
                                & scale_mask[None, None, :]
                            )
                            selected_ctf_probs_raw = jnp.where(
                                ctf_has_mass,
                                selected_ctf_probs * selected_noise[None, None, :],
                                0.0,
                            )
                            aa_before_scale = jnp.where(
                                ctf_has_mass,
                                selected_proj_abs2 * selected_ctf_probs_raw,
                                0.0,
                            )
                            selected_scale = jnp.maximum(
                                jnp.asarray(bucket_scale_for_stats)[selected].astype(
                                    selected_proj_abs2.real.dtype
                                ),
                                1e-30,
                            )
                            aa_terms = aa_before_scale / (
                                selected_scale[:, None, None] ** 2
                            )
                            cross_has_mass = (
                                (selected_summed_masked != 0.0)
                                & scale_mask[None, None, :]
                            )
                            cross_terms = jnp.where(
                                cross_has_mass,
                                selected_proj * jnp.conj(selected_summed_masked),
                                0.0,
                            )
                            xa_terms = (
                                selected_noise[None, None, :]
                                * cross_terms.real
                                / selected_scale[:, None, None]
                            )
                            selected_rotation_mass = jnp.sum(
                                jnp.asarray(noise_probs)[selected],
                                axis=2,
                                dtype=jnp.float32,
                            )
                            aa_feature_per_pixel = jnp.where(
                                selected_rotation_mass[:, :, None] > 0.0,
                                aa_terms / selected_rotation_mass[:, :, None],
                                0.0,
                            )
                            aa_feature_per_shell = jnp.stack(
                                [
                                    jnp.sum(
                                        jnp.where(
                                            scale_mask
                                            & (
                                                jnp.asarray(shell_indices_noise).reshape(-1)
                                                == int(shell)
                                            ),
                                            aa_feature_per_pixel,
                                            0.0,
                                        ),
                                        axis=2,
                                        dtype=jnp.float32,
                                    )
                                    for shell in chunked_scale_aa_feature_shell_ids.tolist()
                                ],
                                axis=2,
                            )
                            staged_scale_aa = jax.block_until_ready(
                                (
                                    jnp.sum(
                                        jnp.asarray(noise_probs)[selected],
                                        axis=(1, 2),
                                        dtype=jnp.float32,
                                    ),
                                    jnp.sum(selected_proj_abs2, axis=1, dtype=jnp.float32),
                                    jnp.sum(selected_ctf_probs_raw, axis=1, dtype=jnp.float32),
                                    jnp.sum(xa_terms, axis=1, dtype=jnp.float32),
                                    jnp.asarray(scale_xa_per_image)[selected],
                                    jnp.sum(norm_a2_terms, axis=1, dtype=jnp.float32),
                                    jnp.sum(norm_a2_terms, axis=(1, 2), dtype=jnp.float32),
                                    jnp.sum(norm_xa_terms, axis=1, dtype=jnp.float32),
                                    jnp.sum(norm_xa_terms, axis=(1, 2), dtype=jnp.float32),
                                    jnp.sum(aa_before_scale, axis=1, dtype=jnp.float32),
                                    jnp.sum(aa_terms, axis=1, dtype=jnp.float32),
                                    jnp.asarray(scale_aa_per_image)[selected],
                                    jnp.asarray(noise_probs)[selected],
                                    jnp.asarray(mstep_rotations)[selected, start:stop],
                                    aa_feature_per_shell,
                                )
                            )
                            (
                                staged_posterior_mass,
                                staged_proj_abs2_sum,
                                staged_ctf_probs_raw_sum,
                                staged_xa_per_pixel,
                                staged_xa_per_image,
                                staged_norm_a2_per_pixel,
                                staged_norm_a2_per_image,
                                staged_norm_xa_per_pixel,
                                staged_norm_xa_per_image,
                                staged_aa_before_scale,
                                staged_aa_per_pixel,
                                staged_aa_per_image,
                                staged_posterior_probs,
                                staged_rotation_matrices,
                                staged_aa_feature_per_shell,
                            ) = (np.asarray(value) for value in staged_scale_aa)
                            chunked_scale_aa_posterior_mass.append(staged_posterior_mass)
                            chunked_scale_aa_proj_abs2_sum.append(staged_proj_abs2_sum)
                            chunked_scale_aa_ctf_probs_raw_sum.append(staged_ctf_probs_raw_sum)
                            chunked_scale_xa_per_pixel.append(staged_xa_per_pixel)
                            chunked_scale_xa_per_image.append(staged_xa_per_image)
                            chunked_norm_a2_per_pixel.append(staged_norm_a2_per_pixel)
                            chunked_norm_a2_per_image.append(staged_norm_a2_per_image)
                            chunked_norm_xa_per_pixel.append(staged_norm_xa_per_pixel)
                            chunked_norm_xa_per_image.append(staged_norm_xa_per_image)
                            chunked_scale_aa_before_scale_per_pixel.append(staged_aa_before_scale)
                            chunked_scale_aa_per_pixel.append(staged_aa_per_pixel)
                            chunked_scale_aa_per_image.append(staged_aa_per_image)
                            chunked_scale_aa_posterior_probs.append(staged_posterior_probs)
                            chunked_scale_aa_rotation_matrices.append(staged_rotation_matrices)
                            chunked_scale_aa_feature_per_shell.append(
                                staged_aa_feature_per_shell
                            )
                        if not relion_wavg_atomic_scale_aa:
                            np.add.at(
                                noise_scale_correction_xa_total,
                                np.asarray(bucket_group_ids, dtype=np.int64),
                                np.asarray(scale_xa_per_image, dtype=np.float64),
                            )
                            np.add.at(
                                noise_scale_correction_aa_total,
                                np.asarray(bucket_group_ids, dtype=np.int64),
                                np.asarray(scale_aa_per_image, dtype=np.float64),
                            )

                if return_stats and probs is not None:
                    # RELION's pdf_direction update is accumulated from the
                    # same significant-pruned weights produced by
                    # collect2jobs for storeWeightedSums.  Keep the chunked
                    # path consistent with the unchunked path and BPref.
                    stats_probs = mstep_probs if use_relion_fine_mstep_prune and not score_only else probs
                    probs_sum_t = np.asarray(jnp.sum(stats_probs, axis=-1), dtype=np.float64)
                    parent_map_chunk = np.asarray(parent_map_padded[:, start:stop], dtype=np.int32)
                    for row, image_idx in enumerate(image_indices.tolist()):
                        cnt = max(0, min(int(actual_counts[row]), int(stop)) - int(start))
                        if cnt == 0:
                            continue
                        unique_rot_image = per_image_inputs["unique_rot"][image_idx]
                        parent_rows = parent_map_chunk[row, :cnt]
                        valid_parent_rows = parent_rows >= 0
                        if not np.any(valid_parent_rows):
                            continue
                        coarse_rot_indices = unique_rot_image[parent_rows[valid_parent_rows]]
                        np.add.at(rotation_posterior_sums, coarse_rot_indices, probs_sum_t[row, :cnt][valid_parent_rows])

            if membership_chunked_bucket:
                if chunk_reconstruction_sum_weight is None:
                    chunk_reconstruction_sum_weight = np.sum(
                        np.concatenate(membership_posterior_mass_chunks, axis=1),
                        axis=1,
                        dtype=np.float64,
                    )
                if chunk_reconstruction_threshold is None:
                    chunk_reconstruction_threshold = np.zeros(
                        (batch,),
                        dtype=np.float64,
                    )
                bpref_diagnostics._maybe_dump_k1_bpref_rotation_mass(
                    experiment_dataset=experiment_dataset,
                    image_indices=image_indices,
                    current_size=current_size,
                    actual_counts=actual_counts,
                    rotations=mstep_rotations,
                    rotation_indices=rotation_indices,
                    candidate_translation_count=np.concatenate(
                        membership_candidate_count_chunks,
                        axis=1,
                    ),
                    posterior_rotation_mass=np.concatenate(
                        membership_posterior_mass_chunks,
                        axis=1,
                    ),
                    reconstruction_rotation_mass=np.concatenate(
                        membership_reconstruction_mass_chunks,
                        axis=1,
                    ),
                    significant_translation_count=np.concatenate(
                        membership_significant_count_chunks,
                        axis=1,
                    ),
                    reconstruction_sum_weight=chunk_reconstruction_sum_weight,
                    reconstruction_threshold=chunk_reconstruction_threshold,
                )

            if contribution_chunked_bucket:
                contribution_scores = jnp.concatenate(
                    contribution_score_chunks,
                    axis=1,
                )
                contribution_preprior_scores = jnp.concatenate(
                    contribution_preprior_score_chunks,
                    axis=1,
                )
                contribution_probs = jnp.concatenate(
                    contribution_prob_chunks,
                    axis=1,
                )
                contribution_reconstruction_probs = jnp.concatenate(
                    contribution_reconstruction_prob_chunks,
                    axis=1,
                )
                contribution_reconstruction_mask = jnp.concatenate(
                    contribution_reconstruction_mask_chunks,
                    axis=1,
                )
                contribution_summed = jnp.concatenate(
                    contribution_summed_chunks,
                    axis=1,
                )
                contribution_ctf_probs = jnp.concatenate(
                    contribution_ctf_prob_chunks,
                    axis=1,
                )
                chunk_shadow_reduction_agreement = None
                if bucket_shadow_only_mode:
                    contribution_authoritative_summed = jnp.concatenate(
                        contribution_authoritative_summed_chunks,
                        axis=1,
                    )
                    contribution_authoritative_ctf_probs = jnp.concatenate(
                        contribution_authoritative_ctf_prob_chunks,
                        axis=1,
                    )
                    chunk_shadow_reduction_agreement = (
                        bpref_diagnostics._require_bpref_reduction_shadow_agreement(
                            contribution_authoritative_summed,
                            contribution_authoritative_ctf_probs,
                            contribution_summed,
                            contribution_ctf_probs,
                        )
                    )
                if chunk_reconstruction_sum_weight is None:
                    chunk_reconstruction_sum_weight = jnp.sum(
                        contribution_probs.reshape(batch, -1),
                        axis=1,
                        dtype=jnp.float64,
                    )
                if chunk_reconstruction_threshold is None:
                    chunk_reconstruction_threshold = jnp.zeros(
                        (batch,),
                        dtype=jnp.float64,
                    )
                bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                    experiment_dataset=experiment_dataset,
                    image_indices=image_indices,
                    current_size=current_size,
                    summed=contribution_summed,
                    ctf_probs=contribution_ctf_probs,
                    rotations=mstep_rotations,
                    actual_counts=actual_counts,
                    rotation_indices=rotation_indices,
                    fine_translations=fine_translations,
                    scores=contribution_scores,
                    preprior_scores=contribution_preprior_scores,
                    probs=contribution_probs,
                    rotation_log_prior=log_prior,
                    translation_log_prior=bucket_translation_prior,
                    log_z=bucket_log_z,
                    best_log_score=global_best_log_score,
                    reconstruction_probs=contribution_reconstruction_probs,
                    reconstruction_mask=contribution_reconstruction_mask,
                    reconstruction_sum_weight=chunk_reconstruction_sum_weight,
                    reconstruction_threshold=chunk_reconstruction_threshold,
                    candidate_mask=candidate_mask,
                    high_precision_operand_bundle=high_precision_operand_bundle,
                    raw_batch_data=(
                        batch_data if high_precision_operand_bundle else None
                    ),
                    ctf_params=(
                        ctf_params if high_precision_operand_bundle else None
                    ),
                    noise_variance_half=(
                        noise_variance_half if high_precision_operand_bundle else None
                    ),
                    integer_pre_shifts=(
                        contribution_preprocess_operands["integer_pre_shifts"]
                        if high_precision_operand_bundle else None
                    ),
                    batch_image_corrections=(
                        contribution_preprocess_operands["batch_image_corrections"]
                        if high_precision_operand_bundle else None
                    ),
                    batch_scale_corrections=(
                        contribution_preprocess_operands["batch_scale_corrections"]
                        if high_precision_operand_bundle else None
                    ),
                    relion_preprocess_normalization_factors=(
                        contribution_preprocess_operands[
                            "relion_preprocess_normalization_factors"
                        ]
                        if high_precision_operand_bundle else None
                    ),
                    relion_cuda_preprocess=(
                        contribution_preprocess_operands["relion_cuda_preprocess"]
                        if high_precision_operand_bundle else False
                    ),
                    score_with_masked_images=score_with_masked_images,
                    image_mask=(
                        contribution_preprocess_operands["image_mask"]
                        if high_precision_operand_bundle else None
                    ),
                    image_mask_mode=(
                        contribution_preprocess_operands["image_mask_mode"]
                        if high_precision_operand_bundle else "not-captured"
                    ),
                    voxel_size=experiment_dataset.voxel_size,
                    ctf_mode=getattr(
                        getattr(config.ctf, "mode", "legacy"),
                        "name",
                        "legacy",
                    ),
                    ctf_dose_per_tilt=getattr(config.ctf, "dose_per_tilt", 0.0),
                    ctf_angle_per_tilt=getattr(config.ctf, "angle_per_tilt", 0.0),
                    disc_type=disc_type,
                    projection_padding_factor=projection_padding_factor,
                    reconstruction_padding_factor=reconstruction_padding_factor,
                    use_relion_x_half_mstep=use_relion_x_half_mstep,
                    winner_take_all=winner_take_all,
                    max_r=float(mstep_current_size // 2) if use_window else None,
                    window_indices=(
                        relion_x_half_recon_indices
                        if use_relion_x_half_mstep
                        else recon_window_indices
                    ),
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    shadow_only_mode=bucket_shadow_only_mode,
                    shadow_score_bitwise_equal=(
                        chunk_shadow_score_bitwise_equal
                    ),
                    shadow_reduction_agreement=(
                        chunk_shadow_reduction_agreement
                    ),
                    device_signature_active=bucket_device_signature_requested,
                    class_index=int(bpref_class_index),
                    mstep_shifted_recon=(
                        shifted_recon_split
                        if high_precision_operand_bundle
                        else None
                    ),
                    mstep_ctf2_over_nv=(
                        ctf2_over_nv_recon
                        if high_precision_operand_bundle
                        else None
                    ),
                )

            if capture_chunked_bucket:
                maybe_capture_k1_production_bucket_chunked(
                    iteration=int(bpref_diagnostics._bpref_contribution_context["iteration"]),
                    half=int(bpref_diagnostics._bpref_contribution_context["half"]),
                    image_indices=image_indices,
                    original_indices=bucket_original_indices,
                    per_image_inputs=per_image_inputs,
                    current_size=current_size,
                    fine_translations=fine_translations,
                    fine_translation_parent=fine_translation_parent,
                    score_chunks=capture_score_chunks,
                    prob_chunks=capture_prob_chunks,
                    rotation_log_prior=jnp.asarray(log_prior),
                    translation_log_prior=bucket_translation_prior,
                    candidate_mask=jnp.asarray(candidate_mask),
                    reconstruction_mask_chunks=capture_reconstruction_mask_chunks,
                    log_z=bucket_log_z,
                    best_log_score=global_best_log_score,
                    best_argmax=global_best_argmax,
                    max_posterior=global_max_posterior,
                )

            if accumulate_noise:
                if relion_wavg_atomic_scale_aa:
                    relion_wavg_atomic_scale_triplet_pixels_np = np.asarray(
                        jax.block_until_ready(relion_wavg_atomic_scale_triplet_pixels),
                        dtype=np.float32,
                    )
                    relion_wavg_atomic_scale_xa_pixels_np = (
                        relion_wavg_atomic_scale_triplet_pixels_np[:, :, 0]
                    )
                    relion_wavg_atomic_scale_aa_pixels_np = (
                        relion_wavg_atomic_scale_triplet_pixels_np[:, :, 1]
                    )
                    relion_wavg_atomic_diff2_pixels_np = (
                        relion_wavg_atomic_scale_triplet_pixels_np[:, :, 2]
                    )
                    scale_pixel_mask_np = np.zeros(
                        relion_wavg_rectangle.centered_indices.size,
                        dtype=bool,
                    )
                    scale_pixel_mask_np[relion_wavg_rectangle.exact_positions] = np.asarray(
                        scale_correction_pixel_mask,
                        dtype=bool,
                    )
                    scale_pixel_mask_np = scale_pixel_mask_np.reshape(1, -1)
                    atomic_xa_per_image = np.sum(
                        np.where(
                            scale_pixel_mask_np,
                            relion_wavg_atomic_scale_xa_pixels_np,
                            np.float32(0.0),
                        ),
                        axis=1,
                        dtype=np.float64,
                    )
                    atomic_aa_per_image = np.sum(
                        np.where(
                            scale_pixel_mask_np,
                            relion_wavg_atomic_scale_aa_pixels_np,
                            np.float32(0.0),
                        ),
                        axis=1,
                        dtype=np.float64,
                    )
                    np.add.at(
                        noise_scale_correction_xa_total,
                        np.asarray(bucket_group_ids, dtype=np.int64),
                        atomic_xa_per_image,
                    )
                    np.add.at(
                        noise_scale_correction_aa_total,
                        np.asarray(bucket_group_ids, dtype=np.int64),
                        atomic_aa_per_image,
                    )
                weighted_img_shells, weighted_img_per_image = _weighted_image_power_shells_and_per_image(
                    processed_score_half_for_noise,
                    shell_indices_half,
                    jnp.asarray(chunk_support_mass, dtype=jnp.float32),
                    shell_count=n_shells,
                    norm_unweighted_shell_cutoff=None if current_size is None else int(current_size // 2),
                    norm_unweighted_high_shell=relion_norm_high_shell,
                    include_unweighted_high_shell=include_unweighted_norm_high_shell,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                )
                if translated_wavg_norm and not relion_wavg_atomic_direct_norm:
                    weighted_img_per_image = _replace_untranslated_low_shell_norm_power(
                        weighted_img_per_image,
                        processed_score_half_for_noise,
                        raw_translated_wavg_for_norm,
                        jnp.asarray(chunk_translation_posterior_total, dtype=jnp.float32),
                        shell_indices_half,
                        window_indices,
                        shell_cutoff=int(current_size // 2),
                    )
                weighted_img_shells_np = np.asarray(
                    weighted_img_shells,
                    dtype=np.float64,
                )
                if relion_wavg_atomic_direct_noise:
                    direct_residual_shells, direct_image_power_shells = (
                        _replace_low_shell_noise_with_relion_wavg_direct_residual(
                            bucket_block_noise_shells,
                            weighted_img_shells_np,
                            relion_wavg_atomic_diff2_pixels_np,
                            relion_wavg_rectangle.shell_indices,
                            exclusive_shell_stop=int(current_size // 2) + 1,
                        )
                    )
                    noise_wsum_total += direct_residual_shells
                    noise_img_power_total += direct_image_power_shells
                else:
                    noise_img_power_total += weighted_img_shells_np
                if relion_wavg_atomic_direct_norm:
                    direct_norm_current = _relion_wavg_direct_norm_per_image(
                        relion_wavg_atomic_diff2_pixels_np,
                        relion_wavg_rectangle.shell_indices,
                        np.zeros(batch, dtype=np.float64),
                    )
                    direct_norm_high = np.asarray(relion_norm_high_shell, dtype=np.float64)
                    noise_wavg_direct_norm_current_total[image_indices] += direct_norm_current
                    noise_wavg_direct_norm_high_total[image_indices] += direct_norm_high
                    noise_norm_correction_total[image_indices] += direct_norm_current + direct_norm_high
                else:
                    noise_norm_correction_total[image_indices] += np.asarray(
                        weighted_img_per_image,
                        dtype=np.float64,
                    )
                noise_sumw_total += float(np.sum(chunk_support_mass, dtype=np.float64))

                if chunked_scale_aa_target_rows.size:
                    dump_dir = os.environ.get(pass2_diagnostics._PASS2_DUMP_DIR_ENV)
                    if not dump_dir:
                        raise ValueError(
                            "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS requires "
                            "RECOVAR_PASS2_DUMP_DIR"
                        )
                    chunked_scale_aa_dump_count = norm_scale_diagnostics._write_chunked_scale_aa_dump(
                        dump_dir=dump_dir,
                        experiment_dataset=experiment_dataset,
                        image_indices=image_indices,
                        target_rows=chunked_scale_aa_target_rows,
                        current_size=current_size,
                        bucket_group_ids=bucket_group_ids,
                        bucket_scale_for_stats=bucket_scale_for_stats,
                        scale_correction_pixel_mask=scale_correction_pixel_mask,
                        scale_shell_indices=shell_indices_noise,
                        chunk_ranges=chunk_ranges,
                        posterior_mass_chunks=chunked_scale_aa_posterior_mass,
                        proj_abs2_sum_chunks=chunked_scale_aa_proj_abs2_sum,
                        ctf_probs_raw_sum_chunks=chunked_scale_aa_ctf_probs_raw_sum,
                        xa_per_pixel_chunks=chunked_scale_xa_per_pixel,
                        xa_per_image_chunks=chunked_scale_xa_per_image,
                        norm_a2_per_pixel_chunks=chunked_norm_a2_per_pixel,
                        norm_a2_per_image_chunks=chunked_norm_a2_per_image,
                        norm_xa_per_pixel_chunks=chunked_norm_xa_per_pixel,
                        norm_xa_per_image_chunks=chunked_norm_xa_per_image,
                        aa_before_scale_per_pixel_chunks=chunked_scale_aa_before_scale_per_pixel,
                        aa_per_pixel_chunks=chunked_scale_aa_per_pixel,
                        aa_per_image_chunks=chunked_scale_aa_per_image,
                        posterior_probs_chunks=chunked_scale_aa_posterior_probs,
                        rotation_matrix_chunks=chunked_scale_aa_rotation_matrices,
                        fine_translations=fine_translations,
                        aa_feature_per_shell_chunks=chunked_scale_aa_feature_per_shell,
                        aa_feature_shell_ids=chunked_scale_aa_feature_shell_ids,
                        atomic_xa_per_pixel=_select_optional_wavg_exact_pixels(
                            relion_wavg_atomic_scale_xa_pixels_np,
                            relion_wavg_rectangle,
                        ),
                        atomic_aa_per_pixel=_select_optional_wavg_exact_pixels(
                            relion_wavg_atomic_scale_aa_pixels_np,
                            relion_wavg_rectangle,
                        ),
                        atomic_diff2_per_pixel=_select_optional_wavg_exact_pixels(
                            relion_wavg_atomic_diff2_pixels_np,
                            relion_wavg_rectangle,
                        ),
                        atomic_diff2_rectangle=relion_wavg_atomic_diff2_pixels_np,
                        atomic_diff2_rectangle_shell_indices=relion_wavg_rectangle.shell_indices,
                        noise_variance_for_noise=noise_variance_for_noise,
                        weighted_img_per_image=weighted_img_per_image,
                        relion_norm_high_shell=relion_norm_high_shell,
                        norm_shifted_images=np.asarray(
                            jax.block_until_ready(
                                jnp.asarray(shifted_noise_split)[
                                    chunked_scale_aa_target_rows
                                ]
                            ),
                            dtype=np.complex64,
                        ),
                    )
                    if chunked_scale_aa_dump_count and parse_env_flag(
                        _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV,
                        default=False,
                    ):
                        logger.info(
                            "Sparse K=1 chunked scale-AA stop-after-dump wrote %d "
                            "requested file(s) at current_size=%s",
                            int(chunked_scale_aa_dump_count),
                            "None" if current_size is None else str(int(current_size)),
                        )
                        raise Pass2DumpComplete(
                            dump_count=chunked_scale_aa_dump_count,
                            current_size=current_size,
                        )

            if accumulate_noise and translation_sqdist_ang is not None:
                noise_sigma2_offset_total += float(
                    np.sum(chunk_translation_posterior_total * translation_sqdist_ang, dtype=np.float64),
                )

            actual_counts_arr = np.asarray(actual_counts, dtype=np.int64)
            best_argmax_np = np.asarray(global_best_argmax, dtype=np.int64)
            best_rot_idx = best_argmax_np // n_fine_trans
            best_trans_idx = best_argmax_np % n_fine_trans
            if np.any(best_rot_idx >= actual_counts_arr):
                bad = np.flatnonzero(best_rot_idx >= actual_counts_arr)
                raise RuntimeError(
                    f"Bucket pass-2 rotation chunking: best rotation index points into padding for images {bad.tolist()} "
                    f"(best_rot_idx={best_rot_idx[bad].tolist()}, actual_counts={actual_counts_arr[bad].tolist()})"
                )
            for row, image_idx in enumerate(image_indices.tolist()):
                r = int(best_rot_idx[row])
                t = int(best_trans_idx[row])
                hard_assignment[image_idx] = r * n_fine_trans + t
                best_rotations[image_idx] = per_image_inputs["oversampled_rots"][image_idx][r]
                if best_eulers is not None:
                    best_eulers[image_idx] = per_image_inputs["source_eulers"][image_idx][r]
                best_rotation_indices[image_idx] = per_image_inputs["oversampled_rot_indices"][image_idx][r]

            if return_stats:
                log_score_offset = (
                    -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
                    if not use_exact_relion_gaussian
                    else np.asarray(
                        _relion_cuda_fine_log_evidence_offset(global_min_diff2),
                        dtype=np.float64,
                    )
                )
                log_Z_np = np.asarray(bucket_log_z, dtype=np.float64)
                class_log_Z_np = (
                    np.asarray(local_score_log_z, dtype=np.float64)
                    if local_score_log_z is not None
                    else log_Z_np
                )
                best_log_score_np = np.asarray(global_best_log_score, dtype=np.float64)
                max_posterior_np = np.asarray(
                    global_max_posterior,
                    dtype=precision_policy.score_real_dtype,
                )
                for row, image_idx in enumerate(image_indices.tolist()):
                    if np.isfinite(best_log_score_np[row]):
                        log_evidence[image_idx] = float(class_log_Z_np[row] + log_score_offset[row])
                        if score_log_z is not None:
                            score_log_z[image_idx] = float(
                                class_log_Z_np[row] + log_score_offset[row]
                                if use_exact_relion_gaussian
                                else class_log_Z_np[row]
                            )
                    else:
                        log_evidence[image_idx] = -np.inf
                        if score_log_z is not None:
                            score_log_z[image_idx] = -np.inf
                    best_log_score[image_idx] = float(best_log_score_np[row] + log_score_offset[row])
                    max_posterior[image_idx] = float(max_posterior_np[row])
            _mark_bucket_group_chunk_done(bucket_size, batch)
            continue

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
            projection_kwargs = window_spec.projection_kwargs(
                return_abs2=False if (use_window or score_only) else None
            )
            projection_kwargs["mask_current_image_disk"] = bool(
                projection_mask_current_image_disk
            )
            if use_window:
                projection_kwargs = _projection_kwargs_for_relion_score_window(
                    projection_kwargs,
                    use_relion_projector=use_relion_projector,
                    current_size=current_size,
                )
                proj_half_flat, proj_for_noise_flat, proj_abs2_for_noise_flat = (
                    _compute_sparse_pass2_windowed_projections_block(
                        mean_for_proj,
                        flat_rotations,
                        image_shape,
                        proj_volume_shape,
                        disc_type,
                        score_indices=window_indices,
                        recon_indices=None if score_only else recon_window_indices,
                        max_projected_rotations=max_projected_rotations_per_projection_call,
                        output_complex_dtype=precision_policy.score_complex_dtype,
                        output_abs2_dtype=precision_policy.score_real_dtype,
                        relion_projector_half=relion_projector_half,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                        **projection_kwargs,
                    )
                )
                proj_half = proj_half_flat.reshape(batch, bucket_size, n_windowed)
                if score_only:
                    proj_for_noise = None
                    proj_abs2_for_noise = None
                else:
                    proj_for_noise = proj_for_noise_flat.reshape(batch, bucket_size, n_recon_windowed)
                    proj_abs2_for_noise = proj_abs2_for_noise_flat.reshape(batch, bucket_size, n_recon_windowed)
            else:
                proj_half_flat, proj_abs2_half_flat = _compute_sparse_pass2_projections_block(
                    mean_for_proj,
                    flat_rotations,
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    max_projected_rotations=max_projected_rotations_per_projection_call,
                    output_complex_dtype=precision_policy.score_complex_dtype,
                    output_abs2_dtype=precision_policy.score_real_dtype,
                    relion_projector_half=relion_projector_half,
                    relion_projector_r_max=relion_projector_r_max,
                    projection_padding_factor=projection_padding_factor,
                    **projection_kwargs,
                )
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
        shadow_score_bitwise_equal = False
        raw_diff2 = None
        if relion_firstiter_score_mode == "normalized_cc":
            min_diff2 = None
            score_args = (
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                jnp.asarray(candidate_mask),
            )
            if relion_exact_fine_normalized_cc:
                scores = _score_pass2_bucket_relion_gpu_normalized_cc(
                    *score_args,
                    relion_score_full_to_compact,
                )
            else:
                scores = _score_pass2_bucket_normalized_cc(*score_args)
            preprior_scores = scores
            _pass2_top2_targets = _resolve_local_target_indices(
                experiment_dataset, _pass2_top2_debug_target_indices()
            )
            if _pass2_top2_targets:
                _log_pass2_top2_debug(
                    scores, image_indices, _pass2_top2_targets, dataset_tag=id(experiment_dataset)
                )
            if bucket_shadow_only_mode:
                if relion_exact_fine_normalized_cc:
                    shadow_scores = _score_pass2_bucket_relion_gpu_normalized_cc(
                        *score_args,
                        relion_score_full_to_compact,
                    )
                else:
                    shadow_scores = _score_pass2_bucket_normalized_cc(*score_args)
                bpref_diagnostics._require_bpref_shadow_exact("normalized-CC score", scores, shadow_scores)
                shadow_score_bitwise_equal = True
        elif use_exact_relion_gaussian:
            raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                relion_score_full_to_compact,
                relion_highres_xi2_half,
                use_fused_ffi=use_relion_fine_diff2_fused_ffi,
            )
            min_diff2 = _relion_cuda_fine_diff2_min(
                raw_diff2,
                jnp.asarray(candidate_mask),
            )
            scores = _relion_cuda_fine_diff2_to_scores(
                raw_diff2,
                jnp.asarray(log_prior)[:, :, None],
                jnp.asarray(bucket_translation_prior)[:, None, :],
                jnp.asarray(candidate_mask),
                min_diff2=min_diff2,
            )
            preprior_scores = None
            if bucket_contribution_diagnostics_active or membership_diagnostics_active:
                zero_rotation_prior = jnp.zeros_like(jnp.asarray(log_prior))[:, :, None]
                zero_translation_prior = jnp.zeros_like(bucket_translation_prior)[:, None, :]
                preprior_scores = _relion_cuda_fine_diff2_to_scores(
                    raw_diff2,
                    zero_rotation_prior,
                    zero_translation_prior,
                    jnp.asarray(candidate_mask),
                    min_diff2=min_diff2,
                )
                shadow_scores = _score_pass2_bucket_relion_gpu_diff2(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(log_prior),
                    bucket_translation_prior,
                    jnp.asarray(candidate_mask),
                    relion_score_full_to_compact,
                    min_diff2,
                    relion_highres_xi2_half,
                    use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                )
                bpref_diagnostics._require_bpref_shadow_exact("exact Gaussian score", scores, shadow_scores)
                shadow_score_bitwise_equal = True
        else:
            min_diff2 = None
            if bucket_contribution_diagnostics_active:
                scores, preprior_scores = _score_pass2_bucket_gaussian_algebraic_components(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(log_prior),
                    bucket_translation_prior,
                    jnp.asarray(candidate_mask),
                )
                shadow_scores = _score_pass2_bucket_gaussian_algebraic(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(log_prior),
                    bucket_translation_prior,
                    jnp.asarray(candidate_mask),
                )
                bpref_diagnostics._require_bpref_shadow_exact("algebraic Gaussian score", scores, shadow_scores)
                shadow_score_bitwise_equal = True
            elif identity_full_projection_cache_rows and projection_cache is not None:
                scores = _score_pass2_bucket_gaussian_algebraic_single_cached(
                    shifted_corrected_score_split[0],
                    ctf2_over_nv_score[0],
                    projection_cache["score"],
                    direct_half_weights,
                    jnp.asarray(log_prior[0]),
                    bucket_translation_prior[0],
                    jnp.asarray(candidate_mask[0]),
                )[jnp.newaxis, :, :]
                preprior_scores = None
            else:
                scores = _score_pass2_bucket_gaussian_algebraic(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(log_prior),
                    bucket_translation_prior,
                    jnp.asarray(candidate_mask),
                )
                preprior_scores = None

        score_log_offset_jax = (
            _relion_cuda_fine_log_evidence_offset(min_diff2).astype(jnp.float64)
            if use_exact_relion_gaussian
            else -0.5 * jnp.squeeze(batch_norm, axis=1).astype(jnp.float64)
        )
        probs = None
        if return_score_log_z_only:
            log_Z = _logsumexp_pass2_bucket_score_only(scores)
            log_score_offset = np.asarray(score_log_offset_jax, dtype=np.float64)
            log_Z_np = np.asarray(log_Z, dtype=np.float64)
            for row, image_idx in enumerate(image_indices.tolist()):
                if np.isfinite(log_Z_np[row]):
                    log_evidence[image_idx] = float(log_Z_np[row] + log_score_offset[row])
                    score_log_z[image_idx] = float(
                        log_Z_np[row] + log_score_offset[row]
                        if use_exact_relion_gaussian
                        else log_Z_np[row]
                    )
                else:
                    log_evidence[image_idx] = -np.inf
                    score_log_z[image_idx] = -np.inf
            _mark_bucket_group_chunk_done(bucket_size, batch)
            continue
        local_score_log_z = None
        if (
            score_only
            and normalization_log_z_np is None
            and normalization_other_score_log_z_np is None
            and not dump_this_bucket
        ):
            log_Z, best_log_score_bucket, best_argmax, max_posterior_bucket = _normalize_pass2_bucket_score_only(
                scores,
            )
        elif normalization_log_z_np is None and normalization_other_score_log_z_np is None:
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = _normalize_pass2_bucket(scores)
        elif normalization_log_z_np is not None:
            bucket_log_z = jnp.asarray(
                normalization_log_z_np[image_indices],
                dtype=precision_policy.normalization_real_dtype,
            )
            if use_exact_relion_gaussian:
                bucket_log_z = bucket_log_z - score_log_offset_jax
            log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                _normalize_pass2_bucket_with_log_z(scores, bucket_log_z)
            )
        else:
            local_score_log_z = _logsumexp_pass2_bucket_score_only(scores)
            bucket_other_log_z = jnp.asarray(
                normalization_other_score_log_z_np[image_indices],
                dtype=local_score_log_z.dtype,
            )
            if not use_exact_relion_gaussian:
                bucket_log_z = jnp.logaddexp(local_score_log_z, bucket_other_log_z)
            else:
                bucket_log_z_absolute = jnp.logaddexp(
                    local_score_log_z + score_log_offset_jax,
                    bucket_other_log_z,
                )
                bucket_log_z = bucket_log_z_absolute - score_log_offset_jax
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
        ctf_probs = None
        reconstruction_probs = None
        reconstruction_mask = None
        reconstruction_n_significant = None
        reconstruction_sum_weight = None
        reconstruction_threshold = None
        if probs is not None and use_relion_fine_mstep_prune and not score_only:
            reconstruction_probs, reconstruction_mask, reconstruction_n_significant = (
                _relion_pass2_reconstruction_probs_for_mstep(
                    scores,
                    probs,
                    adaptive_fraction=float(adaptive_fraction),
                    use_relion_x_half_mstep=use_relion_x_half_mstep,
                    use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
                    winner_take_all=winner_take_all,
                )
            )
            if use_relion_f32_fine_posterior:
                # RELION reports Pmax from the same float32 exponentiation,
                # scan sum, and CUDA division used by its M-step weights.
                # The generic log-sum-exp posterior is only a mathematical
                # equivalent and differs at this observable controller input.
                max_posterior_bucket = jnp.max(
                    reconstruction_probs.reshape(reconstruction_probs.shape[0], -1),
                    axis=1,
                )
            if bucket_contribution_diagnostics_active:
                (
                    shadow_reconstruction_probs,
                    shadow_reconstruction_mask,
                    shadow_reconstruction_n_significant,
                    reconstruction_sum_weight,
                    reconstruction_threshold,
                ) = _relion_pass2_reconstruction_probs_for_mstep(
                    scores,
                    probs,
                    adaptive_fraction=float(adaptive_fraction),
                    use_relion_x_half_mstep=use_relion_x_half_mstep,
                    use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
                    winner_take_all=winner_take_all,
                    return_diagnostics=True,
                )
                bpref_diagnostics._require_bpref_shadow_exact(
                    "reconstruction probabilities",
                    reconstruction_probs,
                    shadow_reconstruction_probs,
                )
                bpref_diagnostics._require_bpref_shadow_exact(
                    "reconstruction mask",
                    reconstruction_mask,
                    shadow_reconstruction_mask,
                )
                bpref_diagnostics._require_bpref_shadow_exact(
                    "reconstruction significant counts",
                    reconstruction_n_significant,
                    shadow_reconstruction_n_significant,
                )
        shifted_recon_split_for_dump = None
        ctf2_over_nv_recon_for_dump = None
        recon_window_indices_for_dump = None
        if probs is not None and not score_only:
            shifted_recon_split_for_dump = shifted_recon.reshape(batch, n_fine_trans, -1)
            ctf2_over_nv_recon_for_dump = ctf2_over_nv_recon
            if use_window:
                recon_window_indices_for_dump = recon_window_indices
            elif use_relion_x_half_mstep:
                recon_window_indices_for_dump = jnp.arange(int(n_half), dtype=jnp.int32)
        if probs is not None:
            bucket_original_indices = original_image_indices(
                experiment_dataset,
                image_indices,
            )
            if compact_capture_requested_for_original_indices(
                int(bpref_diagnostics._bpref_contribution_context["iteration"]),
                bucket_original_indices,
            ):
                maybe_capture_k1_production_bucket(
                    iteration=int(bpref_diagnostics._bpref_contribution_context["iteration"]),
                    half=int(bpref_diagnostics._bpref_contribution_context["half"]),
                    image_indices=image_indices,
                    original_indices=bucket_original_indices,
                    per_image_inputs=per_image_inputs,
                    current_size=current_size,
                    fine_translations=fine_translations,
                    fine_translation_parent=fine_translation_parent,
                    scores=scores,
                    probs=probs,
                    rotation_log_prior=jnp.asarray(log_prior),
                    translation_log_prior=bucket_translation_prior,
                    candidate_mask=candidate_mask,
                    reconstruction_mask=(
                        probs > 0
                        if winner_take_all
                        else (
                            reconstruction_mask
                            if use_relion_fine_mstep_prune
                            # Without pruning, production reconstructs from every
                            # candidate with nonzero posterior support.
                            else candidate_mask
                        )
                    ),
                    log_z=log_Z,
                    best_log_score=best_log_score_bucket,
                    best_argmax=best_argmax,
                    max_posterior=max_posterior_bucket,
                )
            pass2_dump_count = pass2_diagnostics._maybe_dump_pass2_bucket(
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
                reconstruction_mask=reconstruction_mask,
                reconstruction_probs=reconstruction_probs,
                reconstruction_n_significant=reconstruction_n_significant,
                ctf2_over_nv_score=ctf2_over_nv_score,
                proj_half=proj_half,
                half_weights_used=half_weights_windowed if use_window else half_weights,
                window_indices=window_indices_np,
                shifted_corrected_score_split=shifted_corrected_score_split,
                direct_score_input=direct_score_input,
                direct_preprocessed_score_input=direct_preprocessed_score_input,
                direct_pixel_correction=direct_pixel_correction,
                direct_inverse_noise_score=direct_inverse_noise_score,
                direct_ctf_rfloat_score=direct_ctf_rfloat_score,
                direct_preprocess_normalization_factors=(
                    direct_preprocess_normalization_factors
                ),
                direct_integer_pre_shifts=direct_integer_pre_shifts,
                direct_batch_image_corrections=direct_batch_image_corrections,
                direct_batch_scale_corrections=direct_batch_scale_corrections,
                shifted_recon_split=shifted_recon_split_for_dump,
                ctf2_over_nv_recon=ctf2_over_nv_recon_for_dump,
                recon_window_indices=recon_window_indices_for_dump,
                relion_highres_xi2_half=relion_highres_xi2_half,
                relion_min_diff2=min_diff2,
                relion_raw_diff2=raw_diff2,
                relion_full_to_compact=relion_score_full_to_compact,
                raw_score_mode=relion_firstiter_score_mode,
            )
            if pass2_dump_count and parse_env_flag(
                _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
                default=False,
            ):
                target_original_indices = parse_env_int_set(
                    "RECOVAR_PASS2_DUMP_ORIGINAL_INDICES"
                )
                if not target_original_indices:
                    target_original_indices = parse_env_int_set(
                        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
                    )
                completed_dump_count, expected_dump_count = _k1_pass2_dump_progress(
                    dump_dir=os.environ[pass2_diagnostics._PASS2_DUMP_DIR_ENV],
                    target_original_indices=target_original_indices,
                    current_size=current_size,
                )
                logger.info(
                    "Sparse K=1 pass-2 stop-after-dump requested via %s=1; "
                    "target-set progress %d/%d file(s) at current_size=%s",
                    _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
                    int(completed_dump_count),
                    int(expected_dump_count),
                    "None" if current_size is None else str(int(current_size)),
                )
                if completed_dump_count == expected_dump_count:
                    raise Pass2DumpComplete(
                        dump_count=completed_dump_count,
                        current_size=current_size,
                    )

        if not score_only:
            # M-step accumulation: posterior-weighted sums per (image, rot).
            if use_relion_fine_mstep_prune:
                mstep_probs = reconstruction_probs
            else:
                mstep_probs = probs
            if membership_diagnostics_active:
                bpref_diagnostics._maybe_dump_k1_bpref_membership(
                    experiment_dataset=experiment_dataset,
                    image_indices=image_indices,
                    current_size=current_size,
                    actual_counts=actual_counts,
                    rotations=mstep_rotations,
                    rotation_indices=rotation_indices,
                    fine_translations=fine_translations,
                    candidate_mask=candidate_mask,
                    posterior_probs=probs,
                    reconstruction_probs=mstep_probs,
                    reconstruction_mask=(
                        reconstruction_mask
                        if reconstruction_mask is not None
                        else jnp.asarray(mstep_probs) > 0
                    ),
                    reconstruction_sum_weight=(
                        reconstruction_sum_weight
                        if reconstruction_sum_weight is not None
                        else jnp.sum(jnp.asarray(probs).reshape(batch, -1), axis=1)
                    ),
                    reconstruction_threshold=(
                        reconstruction_threshold
                        if reconstruction_threshold is not None
                        else jnp.zeros((batch,), dtype=jnp.float64)
                    ),
                )
            shifted_recon_split = shifted_recon_split_for_dump
            summed, ctf_probs = compute_local_mstep_sums(
                mstep_probs,
                shifted_recon_split,
                ctf2_over_nv_recon,
                relion_x_half=use_relion_x_half_mstep,
                sequential_translation_reduction=use_sequential_translation_reduction,
            )
            if mstep_subtract_ctf_projection:
                summed = subtract_projected_reference_from_sparse_mstep_sums(
                    summed,
                    mstep_probs,
                    proj_for_noise,
                    ctf2_over_nv_recon,
                )
            dump_summed = summed
            dump_ctf_probs = ctf_probs
            shadow_reduction_agreement = None
            if bucket_shadow_only_mode:
                shadow_summed, shadow_ctf_probs = compute_local_mstep_sums(
                    mstep_probs,
                    shifted_recon_split,
                    ctf2_over_nv_recon,
                    relion_x_half=use_relion_x_half_mstep,
                    sequential_translation_reduction=diagnostic_sequential_translation_reduction,
                )
                shadow_reduction_agreement = bpref_diagnostics._require_bpref_reduction_shadow_agreement(
                    summed,
                    ctf_probs,
                    shadow_summed,
                    shadow_ctf_probs,
                )
                dump_summed = shadow_summed
                dump_ctf_probs = shadow_ctf_probs
            bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
                summed=dump_summed,
                ctf_probs=dump_ctf_probs,
                rotations=mstep_rotations,
                actual_counts=actual_counts,
                rotation_indices=rotation_indices,
                fine_translations=fine_translations,
                scores=scores,
                preprior_scores=preprior_scores,
                probs=probs,
                rotation_log_prior=log_prior,
                translation_log_prior=bucket_translation_prior,
                log_z=log_Z,
                best_log_score=best_log_score_bucket,
                reconstruction_probs=mstep_probs,
                reconstruction_mask=(
                    reconstruction_mask
                    if reconstruction_mask is not None
                    else jnp.asarray(mstep_probs) > 0
                ),
                reconstruction_sum_weight=(
                    reconstruction_sum_weight
                    if reconstruction_sum_weight is not None
                    else jnp.sum(jnp.asarray(probs).reshape(batch, -1), axis=1)
                ),
                reconstruction_threshold=(
                    reconstruction_threshold
                    if reconstruction_threshold is not None
                    else jnp.zeros((batch,), dtype=jnp.float64)
                ),
                candidate_mask=candidate_mask,
                high_precision_operand_bundle=high_precision_operand_bundle,
                raw_batch_data=batch_data if high_precision_operand_bundle else None,
                ctf_params=ctf_params if high_precision_operand_bundle else None,
                noise_variance_half=noise_variance_half if high_precision_operand_bundle else None,
                integer_pre_shifts=(
                    contribution_preprocess_operands["integer_pre_shifts"]
                    if high_precision_operand_bundle else None
                ),
                batch_image_corrections=(
                    contribution_preprocess_operands["batch_image_corrections"]
                    if high_precision_operand_bundle else None
                ),
                batch_scale_corrections=(
                    contribution_preprocess_operands["batch_scale_corrections"]
                    if high_precision_operand_bundle else None
                ),
                relion_preprocess_normalization_factors=(
                    contribution_preprocess_operands["relion_preprocess_normalization_factors"]
                    if high_precision_operand_bundle else None
                ),
                relion_cuda_preprocess=(
                    contribution_preprocess_operands["relion_cuda_preprocess"]
                    if high_precision_operand_bundle else False
                ),
                score_with_masked_images=score_with_masked_images,
                image_mask=(contribution_preprocess_operands["image_mask"] if high_precision_operand_bundle else None),
                image_mask_mode=(
                    contribution_preprocess_operands["image_mask_mode"]
                    if high_precision_operand_bundle else "not-captured"
                ),
                voxel_size=experiment_dataset.voxel_size,
                ctf_mode=getattr(getattr(config.ctf, "mode", "legacy"), "name", "legacy"),
                ctf_dose_per_tilt=getattr(config.ctf, "dose_per_tilt", 0.0),
                ctf_angle_per_tilt=getattr(config.ctf, "angle_per_tilt", 0.0),
                disc_type=disc_type,
                projection_padding_factor=projection_padding_factor,
                reconstruction_padding_factor=reconstruction_padding_factor,
                use_relion_x_half_mstep=use_relion_x_half_mstep,
                winner_take_all=winner_take_all,
                max_r=float(mstep_current_size // 2) if use_window else None,
                window_indices=(
                    relion_x_half_recon_indices
                    if use_relion_x_half_mstep
                    else recon_window_indices
                ),
                image_shape=image_shape,
                volume_shape=recon_volume_shape,
                shadow_only_mode=bucket_shadow_only_mode,
                shadow_score_bitwise_equal=shadow_score_bitwise_equal,
                shadow_reduction_agreement=shadow_reduction_agreement,
                device_signature_active=bucket_device_signature_requested,
                class_index=int(bpref_class_index),
                mstep_shifted_recon=(
                    shifted_recon_split
                    if high_precision_operand_bundle
                    else None
                ),
                mstep_ctf2_over_nv=(
                    ctf2_over_nv_recon
                    if high_precision_operand_bundle
                    else None
                ),
            )

            diagnostic_particle_launches_effective = bool(
                use_relion_x_half_mstep
                and diagnostic_per_particle_launches
                and (not device_signature_requested or bucket_device_signature_requested)
            )
            live_per_particle_launches = bool(
                use_relion_x_half_mstep and use_per_particle_launches
            )
            bucket_fused_atomics_requested = bool(
                fused_atomics_requested
                and (not device_signature_requested or bucket_device_signature_requested)
            )
            if bucket_fused_atomics_requested and not diagnostic_particle_launches_effective:
                raise RuntimeError(
                    "RELION fused-atomics diagnostic requires the x-half M-step and "
                    "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1"
                )
            if diagnostic_particle_launches_effective:
                positive_rotation_rows = np.count_nonzero(
                    np.asarray(jnp.sum(mstep_probs, axis=-1)) > 0,
                    axis=1,
                )
                bpref_diagnostics._validate_bpref_positive_rotation_rows(
                    positive_rotation_rows,
                    target_particle_rows,
                    device_signature_requested=device_signature_requested,
                    winner_take_all=winner_take_all,
                )
                diagnostic_owners = bpref_diagnostics._bpref_diagnostic_ownership_indices(
                    image_indices,
                    target_particle_rows,
                    device_signature_requested=device_signature_requested,
                )
                bpref_diagnostics._validate_bpref_diagnostic_ownership(
                    diagnostic_owners,
                    device_signature_requested=bucket_device_signature_requested,
                )
            if live_per_particle_launches:
                mstep_window_indices = (
                    relion_x_half_recon_indices if use_relion_x_half_mstep else recon_window_indices
                )
                Ft_y_total, Ft_ctf_total = _accumulate_relion_x_half_per_particle_launches(
                    summed,
                    ctf_probs,
                    jnp.asarray(mstep_rotations),
                    actual_counts,
                    Ft_y_total,
                    Ft_ctf_total,
                    window_indices=mstep_window_indices,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_volume=use_half_volume_mstep,
                    max_r=float(mstep_current_size // 2) if use_window else None,
                    winner_take_all=winner_take_all,
                    strict_particle_order=preserve_bpref_particle_order,
                    log_label_prefix="single-particle-xhalf",
                )

            # Backproject (use flat_rotations + flat summed/ctf_probs).
            # Padded rotations contribute zero because their probs == 0
            # (candidate_mask=False -> score=-inf -> exp(-inf)=0).
            flat_summed = flatten_bucket_rows(summed)
            flat_ctf_probs = flatten_bucket_rows(ctf_probs)
            mstep_window_indices = relion_x_half_recon_indices if use_relion_x_half_mstep else recon_window_indices
            if not live_per_particle_launches and use_window:
                Ft_y_total = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    flat_backproject_rotations,
                    Ft_y_total,
                    window_indices=mstep_window_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=float(mstep_current_size // 2),
                    relion_x_half=use_relion_x_half_mstep,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-y-window",
                )
                Ft_ctf_total = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    flat_backproject_rotations,
                    Ft_ctf_total,
                    window_indices=mstep_window_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=float(mstep_current_size // 2),
                    relion_x_half=use_relion_x_half_mstep,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-ctf-window",
                )
            elif not live_per_particle_launches and use_relion_x_half_mstep:
                Ft_y_total = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    flat_backproject_rotations,
                    Ft_y_total,
                    window_indices=relion_x_half_recon_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=True,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-y-xhalf",
                )
                Ft_ctf_total = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    flat_backproject_rotations,
                    Ft_ctf_total,
                    window_indices=relion_x_half_recon_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=True,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-ctf-xhalf",
                )
            elif not live_per_particle_launches:
                Ft_y_total = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    flat_backproject_rotations,
                    Ft_y_total,
                    use_windowed_adjoint=False,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=False,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-y-half",
                )
                Ft_ctf_total = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    flat_backproject_rotations,
                    Ft_ctf_total,
                    use_windowed_adjoint=False,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=False,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label="single-ctf-half",
                )

        # Noise accumulation
        if accumulate_noise:
            noise_probs = reconstruction_probs if use_relion_fine_mstep_prune else probs
            if translation_sqdist_ang is not None:
                translation_posterior = np.asarray(jnp.sum(noise_probs, axis=1), dtype=np.float64)
                noise_sigma2_offset_total += float(
                    np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                )
            # RELION support-weights image power inside current_size, while
            # its power_img tail is unweighted above current_size.
            support_mass = jnp.sum(noise_probs, axis=(1, 2))
            weighted_img_shells, weighted_img_per_image = _weighted_image_power_shells_and_per_image(
                processed_score_half_for_noise,
                shell_indices_half,
                support_mass,
                shell_count=n_shells,
                norm_unweighted_shell_cutoff=None if current_size is None else int(current_size // 2),
                norm_unweighted_high_shell=relion_norm_high_shell,
                include_unweighted_high_shell=include_unweighted_norm_high_shell,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            )
            if translated_wavg_norm and not relion_wavg_atomic_direct_norm:
                weighted_img_per_image = _replace_untranslated_low_shell_norm_power(
                    weighted_img_per_image,
                    processed_score_half_for_noise,
                    raw_translated_wavg_for_norm,
                    jnp.sum(noise_probs, axis=1, dtype=jnp.float32),
                    shell_indices_half,
                    window_indices,
                    shell_cutoff=int(current_size // 2),
                )
            support_mass_np = np.asarray(support_mass, dtype=np.float64)
            weighted_img_shells_np = np.asarray(weighted_img_shells, dtype=np.float64)
            if not relion_wavg_atomic_direct_norm:
                noise_norm_correction_total[image_indices] += np.asarray(
                    weighted_img_per_image,
                    dtype=np.float64,
                )
            noise_sumw_total += float(np.sum(support_mass_np, dtype=np.float64))

            if half_spectrum_scoring:
                shifted_noise_split = shifted_noise.reshape(batch, n_fine_trans, -1)
            else:
                shifted_noise_split = shifted_score.reshape(batch, n_fine_trans, -1)
            summed_masked_noise = compute_local_weighted_sums(noise_probs, shifted_noise_split)
            if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                logger.info(
                    "RECOVAR_NOISE_DTYPE_DEBUG(unchunked): proj_for_noise=%s proj_abs2_for_noise=%s "
                    "summed_masked_noise=%s ctf_probs=%s noise_variance_for_noise=%s",
                    proj_for_noise.dtype,
                    proj_abs2_for_noise.dtype,
                    summed_masked_noise.dtype,
                    ctf_probs.dtype,
                    noise_variance_for_noise.dtype,
                )
            block_noise_shells, _, _ = _compute_noise_block_chunked(
                flatten_bucket_rows(proj_for_noise),
                flatten_bucket_rows(proj_abs2_for_noise),
                flatten_bucket_rows(summed_masked_noise),
                flatten_bucket_rows(ctf_probs),
                noise_variance_for_noise,
                shell_indices_noise,
                n_shells,
                max_block_bytes=max_noise_block_bytes,
            )
            if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                logger.info(
                    "RECOVAR_NOISE_DTYPE_DEBUG(unchunked): block_noise_shells=%s",
                    block_noise_shells.dtype,
                )
            block_noise_shells_np = np.asarray(block_noise_shells, dtype=np.float64)
            relion_wavg_atomic_scale_triplet_pixels_np = None
            if relion_wavg_atomic_scale_aa:
                from recovar.cuda_backproject import (
                    relion_wavg_rotation_atomic_triplet_add_f32,
                )

                if direct_ctf_rfloat_recon is None:
                    atomic_triplet_terms = _relion_wavg_atomic_triplet_terms(
                        proj_for_noise,
                        proj_abs2_for_noise,
                        summed_masked_noise,
                        ctf_probs,
                        noise_variance_for_noise,
                        bucket_scale_for_stats,
                        raw_translated_wavg_for_atomic,
                        noise_probs,
                    )
                else:
                    atomic_triplet_terms = _relion_wavg_sequential_triplet_terms(
                        proj_for_noise,
                        direct_ctf_rfloat_recon,
                        bucket_scale_for_stats,
                        raw_translated_wavg_for_atomic,
                        noise_probs,
                    )
                atomic_triplet_terms = _relion_wavg_rectangle_triplet_terms(
                    atomic_triplet_terms,
                    raw_translated_wavg_rectangle,
                    noise_probs,
                    relion_wavg_rectangle.exact_positions,
                )
                atomic_triplet_pixels = jnp.zeros(
                    (
                        batch,
                        int(relion_wavg_rectangle.centered_indices.size),
                        3,
                    ),
                    dtype=jnp.float32,
                )
                relion_wavg_atomic_scale_triplet_pixels_np = np.asarray(
                    jax.block_until_ready(
                        relion_wavg_rotation_atomic_triplet_add_f32(
                            atomic_triplet_terms,
                            atomic_triplet_pixels,
                        )
                    ),
                    dtype=np.float32,
                )
            if relion_wavg_atomic_direct_noise:
                direct_residual_shells, direct_image_power_shells = (
                    _replace_low_shell_noise_with_relion_wavg_direct_residual(
                        block_noise_shells_np,
                        weighted_img_shells_np,
                        relion_wavg_atomic_scale_triplet_pixels_np[:, :, 2],
                        relion_wavg_rectangle.shell_indices,
                        exclusive_shell_stop=int(current_size // 2) + 1,
                    )
                )
                noise_wsum_total += direct_residual_shells
                noise_img_power_total += direct_image_power_shells
            else:
                noise_wsum_total += block_noise_shells_np
                noise_img_power_total += weighted_img_shells_np
            if relion_wavg_atomic_direct_norm:
                direct_norm_current = _relion_wavg_direct_norm_per_image(
                    relion_wavg_atomic_scale_triplet_pixels_np[:, :, 2],
                    relion_wavg_rectangle.shell_indices,
                    np.zeros(batch, dtype=np.float64),
                )
                direct_norm_high = np.asarray(relion_norm_high_shell, dtype=np.float64)
                noise_wavg_direct_norm_current_total[image_indices] += direct_norm_current
                noise_wavg_direct_norm_high_total[image_indices] += direct_norm_high
                noise_norm_correction_total[image_indices] += direct_norm_current + direct_norm_high
            block_norm_residual = _compute_norm_residual_per_image(
                proj_for_noise,
                proj_abs2_for_noise,
                summed_masked_noise,
                ctf_probs,
                noise_variance_for_noise,
            )
            norm_residual_dump_count = norm_scale_diagnostics._maybe_dump_norm_residual_inputs(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
                proj_for_noise=proj_for_noise,
                proj_abs2_for_noise=proj_abs2_for_noise,
                summed_masked_noise=summed_masked_noise,
                ctf_probs=ctf_probs,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                posterior_probs=noise_probs,
                rotations_for_noise=mstep_rotations,
                noise_variance_for_noise=noise_variance_for_noise,
                block_norm_residual=block_norm_residual,
                processed_score_half_for_noise=processed_score_half_for_noise,
                shell_indices_half=shell_indices_half,
                support_mass=support_mass,
                relion_norm_high_shell=relion_norm_high_shell,
                weighted_img_per_image=weighted_img_per_image,
                relion_score_translation_angles=relion_score_translation_angles,
                recon_window_indices=recon_window_indices,
                score_window_indices=window_indices,
                image_shape=image_shape,
                bucket_scale_for_stats=bucket_scale_for_stats,
                scale_correction_pixel_mask=scale_correction_pixel_mask,
                scale_shell_indices=shell_indices_noise,
                bucket_group_ids=bucket_group_ids,
                relion_wavg_atomic_diff2_rectangle=(
                    None
                    if relion_wavg_atomic_scale_triplet_pixels_np is None
                    else relion_wavg_atomic_scale_triplet_pixels_np[:, :, 2]
                ),
                relion_wavg_atomic_rectangle_shell_indices=(
                    None
                    if relion_wavg_atomic_scale_triplet_pixels_np is None
                    else relion_wavg_rectangle.shell_indices
                ),
            )
            if norm_residual_dump_count and parse_env_flag(
                _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV,
                default=False,
            ):
                logger.info(
                    "Sparse K=1 norm/scale operand stop-after-dump requested via %s=1; "
                    "wrote %d requested file(s) at current_size=%s",
                    _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV,
                    int(norm_residual_dump_count),
                    "None" if current_size is None else str(int(current_size)),
                )
                raise Pass2DumpComplete(
                    dump_count=norm_residual_dump_count,
                    current_size=current_size,
                )
            if not relion_wavg_atomic_direct_norm:
                noise_norm_correction_total[image_indices] += np.asarray(
                    block_norm_residual,
                    dtype=np.float64,
                )
            if noise_scale_correction_xa_total is not None:
                scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                    proj_for_noise,
                    proj_abs2_for_noise,
                    summed_masked_noise,
                    ctf_probs,
                    noise_variance_for_noise,
                    bucket_scale_for_stats,
                    scale_correction_pixel_mask,
                )
                if relion_wavg_atomic_scale_aa:
                    scale_pixel_mask_np = np.zeros(
                        relion_wavg_rectangle.centered_indices.size,
                        dtype=bool,
                    )
                    scale_pixel_mask_np[relion_wavg_rectangle.exact_positions] = np.asarray(
                        scale_correction_pixel_mask,
                        dtype=bool,
                    )
                    scale_pixel_mask_np = scale_pixel_mask_np.reshape(1, -1)
                    scale_xa_per_image = np.sum(
                        np.where(
                            scale_pixel_mask_np,
                            relion_wavg_atomic_scale_triplet_pixels_np[:, :, 0],
                            np.float32(0.0),
                        ),
                        axis=1,
                        dtype=np.float64,
                    )
                    scale_aa_per_image = np.sum(
                        np.where(
                            scale_pixel_mask_np,
                            relion_wavg_atomic_scale_triplet_pixels_np[:, :, 1],
                            np.float32(0.0),
                        ),
                        axis=1,
                        dtype=np.float64,
                    )
                np.add.at(
                    noise_scale_correction_xa_total,
                    np.asarray(bucket_group_ids, dtype=np.int64),
                    np.asarray(scale_xa_per_image, dtype=np.float64),
                )
                np.add.at(
                    noise_scale_correction_aa_total,
                    np.asarray(bucket_group_ids, dtype=np.int64),
                    np.asarray(scale_aa_per_image, dtype=np.float64),
                )

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
            if best_eulers is not None:
                best_eulers[image_idx] = per_image_inputs["source_eulers"][image_idx][r]
            best_rotation_indices[image_idx] = per_image_inputs["oversampled_rot_indices"][image_idx][r]

        if return_stats:
            log_score_offset = (
                np.asarray(
                    _relion_cuda_fine_log_evidence_offset(min_diff2),
                    dtype=np.float64,
                )
                if use_exact_relion_gaussian
                else -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            )
            log_Z_np = np.asarray(log_Z, dtype=np.float64)
            class_log_Z_np = (
                np.asarray(local_score_log_z, dtype=np.float64) if local_score_log_z is not None else log_Z_np
            )
            best_log_score_np = np.asarray(best_log_score_bucket, dtype=np.float64)
            max_posterior_np = np.asarray(
                max_posterior_bucket,
                dtype=precision_policy.score_real_dtype,
            )
            for row, image_idx in enumerate(image_indices.tolist()):
                if np.isfinite(best_log_score_np[row]):
                    log_evidence[image_idx] = float(class_log_Z_np[row] + log_score_offset[row])
                    if score_log_z is not None:
                        score_log_z[image_idx] = float(
                            class_log_Z_np[row] + log_score_offset[row]
                            if use_exact_relion_gaussian
                            else class_log_Z_np[row]
                        )
                else:
                    log_evidence[image_idx] = -np.inf
                    if score_log_z is not None:
                        score_log_z[image_idx] = -np.inf
                best_log_score[image_idx] = float(best_log_score_np[row] + log_score_offset[row])
                max_posterior[image_idx] = float(max_posterior_np[row])

            # rotation_posterior_sums: scatter per (image, rot) probability mass back
            # to the parent coarse rotation indices.
            if probs is not None:
                stats_probs = reconstruction_probs if use_relion_fine_mstep_prune else probs
                probs_sum_t = np.asarray(jnp.sum(stats_probs, axis=-1), dtype=np.float64)  # (B, R)
                for row, image_idx in enumerate(image_indices.tolist()):
                    cnt = int(actual_counts[row])
                    if cnt == 0:
                        continue
                    unique_rot_image = per_image_inputs["unique_rot"][image_idx]
                    parent_map_image = per_image_inputs["parent_map"][image_idx]
                    # Map each oversampled rot back to its coarse-grid rotation index.
                    coarse_rot_indices = unique_rot_image[parent_map_image]
                    np.add.at(rotation_posterior_sums, coarse_rot_indices, probs_sum_t[row, :cnt])
        _mark_bucket_group_chunk_done(bucket_size, batch)

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
        Ft_y_total = jnp.zeros(full_volume_size, dtype=recon_y_accum_dtype)
        Ft_ctf_total = jnp.zeros(full_volume_size, dtype=recon_ctf_accum_dtype)
    elif use_half_volume_mstep:
        bpref_diagnostics._maybe_dump_native_half_mstep(
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
        bpref_diagnostics._maybe_dump_native_half_mstep(
            Ft_y_total,
            Ft_ctf_total,
            current_size=current_size,
            n_images=n_images,
            recon_volume_shape=recon_volume_shape,
            stage="post_x0",
        )
        if use_relion_x_half_mstep:
            Ft_y_total, Ft_ctf_total = relion_x_half_accumulators_to_public_layout(
                Ft_y_total,
                Ft_ctf_total,
                recon_volume_shape,
            )
        else:
            Ft_y_total, Ft_ctf_total = half_volume_accumulators_to_full(
                Ft_y_total,
                Ft_ctf_total,
                recon_volume_shape,
            )

    best_translations = fine_translations[hard_assignment % n_fine_trans]

    merged_noise_stats = None
    if accumulate_noise:
        if relion_wavg_atomic_direct_norm:
            norm_dump_dir = os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
            if norm_dump_dir:
                os.makedirs(norm_dump_dir, exist_ok=True)
                context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
                context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
                local_rows = np.arange(n_images, dtype=np.int64)
                original_rows = original_image_indices(experiment_dataset, local_rows)
                norm_dump_path = os.path.join(
                    norm_dump_dir,
                    f"recovar_wavg_norm_it{context_iteration:03d}_half{context_half}.npz",
                )
                np.savez_compressed(
                    norm_dump_path,
                    schema=np.asarray("recovar-k1-wavg-direct-norm-v1"),
                    one_based_iteration=np.int64(context_iteration),
                    half=np.int64(context_half),
                    local_row=local_rows,
                    original_row=original_rows,
                    current_size=np.int64(current_size),
                    direct_current_size=np.asarray(
                        noise_wavg_direct_norm_current_total,
                        dtype=np.float64,
                    ),
                    powerclass_high_shell=np.asarray(
                        noise_wavg_direct_norm_high_total,
                        dtype=np.float64,
                    ),
                    total=np.asarray(noise_norm_correction_total, dtype=np.float64),
                )
                logger.info("Wrote RECOVAR direct Wavg norm debug dump: %s", norm_dump_path)
        merged_noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum_total,
            wsum_img_power=noise_img_power_total,
            wsum_sigma2_offset=noise_sigma2_offset_total,
            sumw=noise_sumw_total,
            wsum_norm_correction=noise_norm_correction_total,
            wsum_scale_correction_xa=noise_scale_correction_xa_total,
            wsum_scale_correction_aa=noise_scale_correction_aa_total,
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
        return result + (best_eulers,) if return_source_eulers else result

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
    return result + (best_eulers,) if return_source_eulers else result


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
    noise_variance,
    translations,
    significant_sample_indices_by_class,
    *,
    rotation_log_priors_by_class,
    nside_level,
    disc_type,
    oversampling_order,
    current_size,
    reconstruction_current_size=None,
    translation_step=None,
    score_with_masked_images=False,
    return_stats=True,
    accumulate_noise=False,
    translation_log_prior=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    projection_mask_current_image_disk=True,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    image_pre_shifts=None,
    use_float64_scoring=False,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    rotation_block_size_for_quantization=5000,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune_mode: str | None = None,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    normalization_log_evidence=None,
    relion_f32_normalization_sum_weight=None,
    compact_pair_min_bucket_size_default: int | None = None,
    compact_pair_tail_coalesce_max_images_default: int | None = None,
    compact_pair_tail_coalesce_max_inflation_default: float | None = None,
    compact_pair_tail_coalesce_min_bucket_size_default: int | None = None,
    source_faithful_spectrum_norm: bool = False,
) -> SparseKClassPass2FusedResult:
    """Evaluate K-class sparse pass-2 in one joint class-normalized sweep.

    This mirrors RELION's fine-pass semantics: all class-local scores are
    normalized by one per-image class x pose denominator before M-step
    accumulation.  The exact fused implementation currently requires a shared
    class noise model; callers should fall back to the existing per-class path
    when noise differs by class.
    """

    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    device_signature_requested = bool(
        device_signature_configured and bpref_device_signature_active
    )
    scoped_diagnostic_flags = bpref_diagnostics._scoped_bpref_diagnostic_flags(
        active=bpref_device_signature_active
    )
    execution_modes = bpref_diagnostics._resolve_bpref_execution_modes(
        scoped_diagnostic_flags,
        device_signature_requested=device_signature_requested,
    )
    use_per_particle_launches = execution_modes["live_per_particle_launches"]
    if device_signature_requested:
        if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"):
            raise RuntimeError(
                "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR requires "
                "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"
            )
        bpref_diagnostics._require_bpref_device_soft_particle_arm(
            use_relion_x_half_mstep=bool(relion_x_half_mstep),
        )

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
    use_exact_relion_gaussian = bool(
        relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
    )
    use_relion_fine_diff2_fused_ffi = bool(
        relion_fine_diff2_fused_ffi
        or parse_env_flag(_RELION_FINE_DIFF2_FUSED_FFI_ENV, default=False)
    )
    use_relion_f32_fine_posterior = bool(
        relion_f32_fine_posterior
        or relion_x_half_f32_fine_posterior_enabled()
    )
    relion_exact_bpref_operands = parse_env_flag(
        "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS",
        default=False,
    )
    if relion_exact_bpref_operands:
        if use_float64_scoring:
            raise ValueError("exact RELION BPref operands require the native float32 path")
        logger.info(
            "Sparse fused K-class pass-2: using RELION binary64-to-float32 "
            "inverse-noise and fused translate-then-weight BPref operands"
        )
    volumes = jnp.asarray(volumes)
    n_classes = int(volumes.shape[0])
    if source_faithful_spectrum_norm and n_classes != 1:
        raise ValueError("source-faithful powerClass normalization is K=1-only")
    if device_signature_requested and not any(
        bpref_diagnostics._bpref_contribution_class_enabled(class_index)
        for class_index in range(n_classes)
    ):
        raise ValueError(
            f"{bpref_diagnostics._BPREF_CONTRIBUTION_DUMP_CLASS_ENV} selects no class in a {n_classes}-class run"
        )
    if len(significant_sample_indices_by_class) != n_classes:
        raise ValueError("significant_sample_indices_by_class must match class count")
    if len(rotation_log_priors_by_class) != n_classes:
        raise ValueError("rotation_log_priors_by_class must match class count")
    use_relion_projector = relion_projector_half is not None
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim == 3 and n_classes == 1:
            relion_projector_half = relion_projector_half[None, ...]
        if relion_projector_half.ndim != 4 or int(relion_projector_half.shape[0]) != n_classes:
            raise ValueError(
                "relion_projector_half must have shape "
                f"({n_classes}, z, y, x_half), got {relion_projector_half.shape}",
            )
    shared_noise_variance = _shared_k_class_noise_variance(noise_variance, n_classes)
    if shared_noise_variance is None:
        raise NotImplementedError("fused sparse K-class pass-2 requires shared class noise variance")

    n_images = int(experiment_dataset.n_units)
    normalization_log_evidence_np = None
    if normalization_log_evidence is not None:
        normalization_log_evidence_np = np.asarray(
            normalization_log_evidence,
            dtype=np.float64,
        )
        if normalization_log_evidence_np.shape != (n_images,):
            raise ValueError(
                "normalization_log_evidence must have shape "
                f"({n_images},), got {normalization_log_evidence_np.shape}",
            )
    relion_f32_normalization_sum_weight_np = None
    if relion_f32_normalization_sum_weight is not None:
        relion_f32_normalization_sum_weight_np = np.asarray(
            relion_f32_normalization_sum_weight,
            dtype=np.float32,
        )
        if relion_f32_normalization_sum_weight_np.shape != (n_images,):
            raise ValueError(
                "relion_f32_normalization_sum_weight must have shape "
                f"({n_images},), got {relion_f32_normalization_sum_weight_np.shape}",
            )
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    if not hasattr(experiment_dataset, "image_shape") or not hasattr(experiment_dataset, "volume_shape"):
        raise NotImplementedError("fused sparse K-class pass-2 requires dataset image_shape and volume_shape")
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    mstep_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if (
        use_exact_relion_gaussian
        and current_size is not None
        and int(current_size) < int(W)
        and (not half_spectrum_scoring or square_window)
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian high-resolution scoring requires "
            "half_spectrum_scoring=True and square_window=False"
        )
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
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    device_memory_bytes = _device_memory_limit_bytes()
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    # The fused K-class route deliberately excludes the K=1-only
    # source-faithful spectrum-normalization option at its caller boundary.
    source_faithful_spectrum_norm = False

    use_relion_x_half_mstep = bool(relion_x_half_mstep)
    if use_relion_x_half_mstep:
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max.  The reconstruction boundary then crops the output
        # back to ``volume_shape``.
        recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=mstep_current_size,
        )
    elif reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
    else:
        recon_volume_shape = volume_shape
    relion_fine_mstep_prune_mode = _relion_fine_mstep_prune_mode(
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        mode_override=relion_fine_mstep_prune_mode,
    )
    relion_fine_mstep_prune = relion_fine_mstep_prune_mode != "none"
    relion_fine_mstep_joint = relion_fine_mstep_prune_mode in {
        "joint",
        "joint_keep_all",
    }
    use_half_volume_mstep = bool(relion_half_volume_mstep) or use_relion_x_half_mstep
    compact_pair_mstep_mode_requested = _compact_pair_mstep_mode_for_pass()
    compact_pair_pair_sparse_requested = compact_pair_mstep_mode_requested == "pair_sparse"
    # RELION x-half M-step parity depends on the dense probability tensor plus
    # the same GPU matmul order as rectangular pass-2. Sparse pair-order image
    # reductions are mathematically equivalent but not arithmetic-equivalent
    # enough for the strict x-half guard.
    compact_pair_pair_sparse_effective = bool(
        compact_pair_pair_sparse_requested
        and not use_relion_x_half_mstep
    )
    compact_pair_pair_sparse_xhalf_fallback = bool(
        compact_pair_pair_sparse_requested
        and use_relion_x_half_mstep
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape) if use_half_volume_mstep else recon_volume_shape
    recon_volume_size = int(np.prod(recon_accum_shape))
    if use_relion_x_half_mstep:
        logger.info(
            "Sparse fused K-class RELION x-half current-size BPref accumulator shape: "
            "volume_shape=%s current_size=%s padding_factor=%s recon_volume_shape=%s half_accum_shape=%s voxels=%d",
            tuple(volume_shape),
            mstep_current_size,
            reconstruction_padding_factor,
            tuple(recon_volume_shape),
            tuple(recon_accum_shape),
            recon_volume_size,
        )
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
    )

    mean_for_proj_by_class = []
    proj_volume_shape = volume_shape
    for class_index in range(n_classes):
        class_volume = volumes[class_index]
        if projection_padding_factor > 1 and not use_relion_projector:
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

    translations_np = np.asarray(translations, dtype=precision_policy.score_real_dtype)
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
        fine_translations = np.asarray(fine_translations, dtype=precision_policy.score_real_dtype)
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations = np.asarray(fine_translations_override, dtype=precision_policy.score_real_dtype)
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
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=precision_policy.score_real_dtype)
        fine_translation_prior_2d = expand_fine_translation_prior(
            translation_log_prior_np,
            fine_translation_parent,
            n_images=n_images,
            n_fine_trans=n_fine_trans,
            dtype=precision_policy.score_real_dtype,
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
            fine_source_eulers_override=fine_source_eulers_override,
            fine_rotations_override=fine_rotations_override,
            fine_mstep_rotations_override=fine_mstep_rotations_override,
            fine_rotation_parent_override=fine_rotation_parent_override,
            dtype=precision_policy.score_real_dtype,
        )
        for class_index in range(n_classes)
    ]
    prep_s = time.time() - prep_t0
    local_rot_counts = [
        int(rots.shape[0])
        for per_image_inputs in per_image_inputs_by_class
        for rots in per_image_inputs["oversampled_rots"]
    ]
    candidate_counts_by_class = tuple(
        np.asarray(
            [_candidate_mask_count(mask) for mask in per_image_inputs["candidate_mask"]],
            dtype=np.int64,
        )
        for per_image_inputs in per_image_inputs_by_class
    )
    valid_candidate_counts = [
        int(count)
        for candidate_counts in candidate_counts_by_class
        for count in candidate_counts.tolist()
    ]

    max_hypotheses_per_microbatch = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=budget_window_spec.use_window,
        has_external_normalization=False,
        conservative_dump_execution=_pass2_conservative_dump_execution_enabled(),
        fused_k_class=True,
        fused_k_class_count=n_classes,
        n_score_pixels=budget_window_spec.n_score,
        device_memory_bytes=device_memory_bytes,
        score_complex_dtype=precision_policy.score_complex_dtype,
    )
    max_translation_tile_bytes = _max_translation_tile_bytes_for_pass(
        device_memory_bytes,
        fused_k_class=True,
    )
    max_projection_gather_bytes = _max_projection_gather_bytes_for_pass(device_memory_bytes)
    max_noise_block_bytes = _max_noise_block_bytes_for_pass(device_memory_bytes)
    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)
    compact_pair_dense_mstep_max_bytes = _compact_pair_dense_mstep_max_bytes_for_pass(device_memory_bytes)
    translation_tile_half_pixels = _translation_tile_half_pixels_for_budget(
        use_window=budget_window_spec.use_window,
        n_score_pixels=budget_window_spec.n_score,
        n_recon_pixels=budget_window_spec.n_recon,
    )
    max_images_per_microbatch = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_translation_tile_bytes,
        complex_dtype=precision_policy.score_complex_dtype,
        n_half_pixels=translation_tile_half_pixels,
    )
    small_bucket_threshold = _optional_positive_int_env(_SMALL_BUCKET_THRESHOLD_ENV)
    small_bucket_coalesce_size = _small_bucket_coalesce_size_for_pass(n_images)
    (
        tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size,
    ) = _tail_bucket_coalesce_params_for_pass(fused_k_class=True)
    small_bucket_max_images_per_microbatch = None
    small_bucket_max_translation_tile_bytes = _optional_positive_int_env(
        _SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES_ENV,
    )
    if small_bucket_max_translation_tile_bytes is not None:
        if small_bucket_threshold is None:
            small_bucket_threshold = 128
        small_bucket_max_images_per_microbatch = _max_images_for_translation_tile(
            image_shape,
            n_fine_trans,
            max_tile_bytes=small_bucket_max_translation_tile_bytes,
            complex_dtype=precision_policy.score_complex_dtype,
            n_half_pixels=translation_tile_half_pixels,
        )
    projection_complex_dtype = _projection_cache_budget_complex_dtype(
        jnp.asarray(mean_for_proj_by_class[0]).dtype,
        precision_policy.score_complex_dtype,
        use_relion_projector=use_relion_projector,
    )
    projection_budget_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=budget_window_spec.use_window,
        use_relion_projector=use_relion_projector,
    )
    max_projected_rotations_per_projection_call = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=device_memory_bytes,
        n_projection_pixels=projection_budget_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=not budget_window_spec.use_window,
    )
    bucket_t0 = time.time()
    buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        small_bucket_threshold=small_bucket_threshold,
        small_bucket_max_images_per_microbatch=small_bucket_max_images_per_microbatch,
        small_bucket_coalesce_size=small_bucket_coalesce_size,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
    )
    bucket_s = time.time() - bucket_t0
    logger.info(
        "Sparse fused K-class pass-2 bucketing: %d images x %d classes -> %d buckets (%s; "
        "max_hypotheses_per_microbatch=%d, max_images_per_microbatch=%d, "
        "translation_tile_half_pixels=%s, windowed_translation_tile_cap=%s, "
        "small_bucket_threshold=%s, small_bucket_max_images_per_microbatch=%s, "
        "small_bucket_coalesce_size=%s, tail_bucket_coalesce=%s/%s/%s, "
        "max_projected_rotations_per_projection_call=%s, max_translation_tile_bytes=%d, "
        "max_projection_gather_bytes=%d, max_compact_pair_dense_mstep_bytes=%d, "
        "max_noise_block_bytes=%d, max_adjoint_block_bytes=%d, "
        "n_score_pixels=%d, device_memory_gib=%.2f)",
        n_images,
        n_classes,
        len(buckets),
        _bucket_summary(buckets),
        max_hypotheses_per_microbatch,
        max_images_per_microbatch,
        int(translation_tile_half_pixels) if translation_tile_half_pixels is not None else int(n_half),
        int(bool(_windowed_translation_tile_cap_enabled_for_pass())),
        "unset" if small_bucket_threshold is None else str(int(small_bucket_threshold)),
        "unset"
        if small_bucket_max_images_per_microbatch is None
        else str(int(small_bucket_max_images_per_microbatch)),
        "unset" if small_bucket_coalesce_size is None else str(int(small_bucket_coalesce_size)),
        "unset"
        if tail_bucket_coalesce_max_images is None
        else str(int(tail_bucket_coalesce_max_images)),
        "unset"
        if tail_bucket_coalesce_max_inflation is None
        else f"{float(tail_bucket_coalesce_max_inflation):.3g}",
        "unset"
        if tail_bucket_coalesce_min_bucket_size is None
        else str(int(tail_bucket_coalesce_min_bucket_size)),
        "unset"
        if max_projected_rotations_per_projection_call is None
        else str(max_projected_rotations_per_projection_call),
        max_translation_tile_bytes,
        max_projection_gather_bytes,
        compact_pair_dense_mstep_max_bytes,
        max_noise_block_bytes,
        max_adjoint_block_bytes,
        int(budget_window_spec.n_score),
        (-1.0 if device_memory_bytes is None else device_memory_bytes / float(1024**3)),
    )
    compact_pairs_env = os.environ.get(_SPARSE_KCLASS_COMPACT_PAIRS_ENV)
    compact_pairs = _compact_pair_execution_enabled_for_pass()
    if use_per_particle_launches:
        if not relion_x_half_mstep:
            raise ValueError(
                "fused K-class per-particle launches require the RELION x-half M-step"
            )
        if not compact_pairs:
            raise ValueError(
                "fused K-class per-particle launches require compact-pair execution"
            )
    compact_active_rows_env = os.environ.get(_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV)
    compact_active_rows = (
        compact_pairs
        and parse_env_flag(_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV, default=True)
    )
    reuse_compact_noise_sums = parse_env_flag(
        _SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV,
        default=False,
    )
    native_dual_weighted_sums = _native_dual_weighted_sums_enabled_for_pass(
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        accumulate_noise=accumulate_noise,
    )
    compact_noise_sums_match_mstep = bool(
        reuse_compact_noise_sums
        and half_spectrum_scoring
        and not score_with_masked_images
    )
    fused_mstep_noise = _fused_mstep_noise_enabled_for_pass(
        native_dual_weighted_sums=native_dual_weighted_sums,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        accumulate_noise=accumulate_noise,
        compact_noise_sums_match_mstep=compact_noise_sums_match_mstep,
    )
    rectangular_active_rows = parse_env_flag(_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_ENV, default=True)
    rectangular_active_prematmul = (
        rectangular_active_rows
        and parse_env_flag(_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_ENV, default=False)
    )
    rectangular_active_prematmul_max_grouped_dense_ratio = _optional_positive_float_env(
        _SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO_ENV,
    )
    if rectangular_active_prematmul_max_grouped_dense_ratio is None:
        rectangular_active_prematmul_max_grouped_dense_ratio = (
            _DEFAULT_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO
        )
    fused_noise_norm = parse_env_flag(_SPARSE_KCLASS_FUSED_NOISE_NORM_ENV, default=True)
    rectangular_active_rows_min_bucket_size = _optional_positive_int_env(
        _SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE_ENV,
    )
    if rectangular_active_rows_min_bucket_size is None:
        rectangular_active_rows_min_bucket_size = _DEFAULT_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE
    active_row_pad_multiple = _active_row_pad_multiple_for_pass()
    compact_pair_buckets = None
    compact_pair_report_buckets = None
    compact_pair_min_bucket_size = None
    compact_plan_t0 = time.time()
    compact_pair_plan_stats = None
    compact_pair_threshold_reports = []
    (
        compact_pair_tail_coalesce_max_images,
        compact_pair_tail_coalesce_max_inflation,
        compact_pair_tail_coalesce_min_bucket_size,
    ) = _compact_pair_tail_bucket_coalesce_params_for_pass(
        default_max_images=compact_pair_tail_coalesce_max_images_default,
        default_max_inflation=compact_pair_tail_coalesce_max_inflation_default,
        default_min_bucket_size=compact_pair_tail_coalesce_min_bucket_size_default,
    )
    if compact_pairs:
        compact_pair_min_bucket_size = _compact_pair_min_bucket_size_for_pass(
            compact_pair_min_bucket_size_default,
        )
        compact_pair_max_images_per_microbatch = _compact_pair_max_images_per_microbatch_for_pass(
            max_images_per_microbatch,
        )
        compact_pair_counts_by_class = candidate_counts_by_class
        compact_pair_execution_image_mask = _compact_pair_image_mask_for_threshold(
            compact_pair_counts_by_class,
            compact_pair_min_bucket_size,
        )
        if compact_pair_execution_image_mask is not None:
            logger.info(
                "Sparse fused K-class compact-pair execution prefilter: "
                "threshold=%d compact_images=%d/%d",
                int(compact_pair_min_bucket_size),
                int(np.count_nonzero(compact_pair_execution_image_mask)),
                int(compact_pair_execution_image_mask.shape[0]),
            )
        (
            compact_pair_execution_image_mask,
            compact_pair_full_support_excluded,
        ) = _compact_pair_execution_mask_excluding_full_support(
            per_image_inputs_by_class,
            compact_pair_execution_image_mask,
        )
        if compact_pair_full_support_excluded:
            logger.info(
                "Sparse fused K-class compact-pair full-support filter: "
                "excluded=%d compact_images=%d/%d",
                int(compact_pair_full_support_excluded),
                0
                if compact_pair_execution_image_mask is None
                else int(np.count_nonzero(compact_pair_execution_image_mask)),
                int(n_images),
            )
        compact_pair_plan_stats = _compact_k_class_pair_plan_stats_from_counts(
            compact_pair_counts_by_class,
            buckets,
            n_fine_trans,
            max_pair_candidates_per_microbatch=max_hypotheses_per_microbatch,
            max_images_per_microbatch=compact_pair_max_images_per_microbatch,
            image_mask=compact_pair_execution_image_mask,
            tail_bucket_coalesce_max_images=compact_pair_tail_coalesce_max_images,
            tail_bucket_coalesce_max_inflation=compact_pair_tail_coalesce_max_inflation,
            tail_bucket_coalesce_min_bucket_size=compact_pair_tail_coalesce_min_bucket_size,
        )
        compact_pair_report_buckets = list(compact_pair_plan_stats.buckets)
        compact_pair_buckets_for_split = _compact_pair_buckets_for_execution_threshold(
            compact_pair_report_buckets,
            compact_pair_min_bucket_size,
        )
        if compact_pair_min_bucket_size is not None:
            logger.info(
                "Sparse fused K-class compact-pair pre-split threshold filter: "
                "threshold=%d buckets %d -> %d images %d -> %d",
                int(compact_pair_min_bucket_size),
                len(compact_pair_report_buckets),
                len(compact_pair_buckets_for_split),
                sum(len(bucket["image_indices"]) for bucket in compact_pair_report_buckets),
                sum(len(bucket["image_indices"]) for bucket in compact_pair_buckets_for_split),
        )
        compact_pair_buckets = _split_compact_pair_buckets_by_projection_gather_budget(
            compact_pair_buckets_for_split,
            per_image_inputs_by_class,
            n_score_pixels=int(budget_window_spec.n_score),
            n_recon_pixels=int(budget_window_spec.n_recon),
            projection_complex_dtype=projection_complex_dtype,
            max_gather_bytes=max_projection_gather_bytes,
            max_dense_mstep_bytes=None
            if compact_pair_pair_sparse_effective
            else compact_pair_dense_mstep_max_bytes,
            n_fine_trans=n_fine_trans,
            prob_dtype=precision_policy.normalization_real_dtype,
            max_prepare_images_per_microbatch=_compact_pair_prepare_max_images_per_microbatch(
                dense_max_images_per_microbatch=max_images_per_microbatch,
                compact_pair_max_images_per_microbatch=compact_pair_max_images_per_microbatch,
            ),
            rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        )
    else:
        compact_pair_plan_stats = _maybe_prepare_sparse_k_class_compact_pair_plan(
            per_image_inputs_by_class,
            buckets,
            n_fine_trans,
            max_pair_candidates_per_microbatch=max_hypotheses_per_microbatch,
            max_images_per_microbatch=max_images_per_microbatch,
            tail_bucket_coalesce_max_images=compact_pair_tail_coalesce_max_images,
            tail_bucket_coalesce_max_inflation=compact_pair_tail_coalesce_max_inflation,
            tail_bucket_coalesce_min_bucket_size=compact_pair_tail_coalesce_min_bucket_size,
        )
    compact_plan_s = time.time() - compact_plan_t0
    if compact_pair_plan_stats is not None:
        logger.info(
            "Sparse fused K-class compact-pair planner: dense scoring unchanged. "
            "valid_pair_candidates=%d, padded_pair_candidates=%d, rectangular_candidates=%d, "
            "valid_reduction=%.1fx, padded_reduction=%.1fx, compact_buckets=%d, "
            "median_valid_pairs/image=%d, mean_valid_pairs/image=%.1f, max_valid_pairs/image=%d, "
            "compact_pair_max_images_per_microbatch=%d, dense_max_images_per_microbatch=%d, "
            "compact_tail_coalesce=%s/%s/%s, plan_time=%.2fs",
            compact_pair_plan_stats.valid_pair_candidates,
            compact_pair_plan_stats.padded_pair_candidates,
            compact_pair_plan_stats.rectangular_candidates,
            compact_pair_plan_stats.reduction_factor,
            compact_pair_plan_stats.padded_reduction_factor,
            len(compact_pair_plan_stats.buckets),
            compact_pair_plan_stats.median_valid_pairs_per_image,
            compact_pair_plan_stats.mean_valid_pairs_per_image,
            compact_pair_plan_stats.max_valid_pairs_per_image,
            compact_pair_plan_stats.max_images_per_microbatch,
            max_images_per_microbatch,
            "unset"
            if compact_pair_tail_coalesce_max_images is None
            else str(int(compact_pair_tail_coalesce_max_images)),
            "unset"
            if compact_pair_tail_coalesce_max_inflation is None
            else f"{float(compact_pair_tail_coalesce_max_inflation):.3g}",
            "unset"
            if compact_pair_tail_coalesce_min_bucket_size is None
            else str(int(compact_pair_tail_coalesce_min_bucket_size)),
            compact_plan_s,
        )
        threshold_report_buckets = (
            compact_pair_report_buckets
            if compact_pair_report_buckets is not None
            else list(compact_pair_plan_stats.buckets)
        )
        compact_pair_threshold_reports = _compact_pair_hybrid_threshold_reports(
            buckets,
            threshold_report_buckets,
            thresholds=_compact_pair_threshold_report_thresholds(),
            n_classes=n_classes,
            n_fine_trans=n_fine_trans,
        )
        for threshold_report in compact_pair_threshold_reports:
            logger.info(
                "Sparse fused K-class compact-pair hybrid threshold plan: "
                "threshold=%d compact_buckets=%d compact_images=%d "
                "rectangular_buckets=%d rectangular_images=%d "
                "rectangular_candidate_slots=%d compact_candidate_slots=%d "
                "total_candidate_slots=%d slot_reduction=%.3fx",
                threshold_report["threshold"],
                threshold_report["compact_buckets"],
                threshold_report["compact_images"],
                threshold_report["rectangular_buckets"],
                threshold_report["rectangular_images"],
                threshold_report["rectangular_candidate_slots"],
                threshold_report["compact_candidate_slots"],
                threshold_report["total_candidate_slots"],
                threshold_report["slot_reduction"],
            )
    compact_pair_inputs_by_class_for_check = None
    if compact_pairs:
        logger.info(
            "Sparse fused K-class compact-pair execution enabled (%s=%s); "
            "planned %d compact pair buckets before hybrid routing (%s)",
            _SPARSE_KCLASS_COMPACT_PAIRS_ENV,
            "auto" if compact_pairs_env is None or compact_pairs_env == "" else compact_pairs_env,
            0 if compact_pair_buckets is None else len(compact_pair_buckets),
            "empty"
            if not compact_pair_buckets
            else _bucket_summary(compact_pair_buckets, "pair_bucket_size"),
        )
        if compact_pair_min_bucket_size is not None:
            logger.info(
                "Sparse fused K-class compact-pair hybrid threshold enabled via %s=%d",
                _SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV,
                int(compact_pair_min_bucket_size),
            )
        if compact_pair_pair_sparse_xhalf_fallback:
            logger.warning(
                "Sparse fused K-class compact-pair M-step mode %s=pair_sparse requested, "
                "but RELION x-half M-step requires dense matmul-order reductions; using dense reductions",
                _SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV,
            )
        elif compact_pair_pair_sparse_effective:
            logger.info(
                "Sparse fused K-class compact-pair M-step mode enabled via %s=pair_sparse",
                _SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV,
            )
        if compact_active_rows:
            logger.info(
                "Sparse fused K-class compact active rows enabled (%s=%s)",
                _SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV,
                "auto" if compact_active_rows_env is None or compact_active_rows_env == "" else compact_active_rows_env,
            )
        if relion_fine_mstep_prune:
            logger.info(
                "Sparse fused K-class RELION fine-pass M-step pruning enabled via %s=%s",
                _SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV,
                relion_fine_mstep_prune_mode,
            )
        if compact_noise_sums_match_mstep:
            logger.info(
                "Sparse fused K-class compact-pair noise sums reuse M-step weighted sums "
                "(%s=1, half_spectrum_scoring=True, score_with_masked_images=False)",
                _SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV,
            )
        elif reuse_compact_noise_sums and accumulate_noise:
            logger.info(
                "Sparse fused K-class compact-pair noise path reuses compact probability/CTF sums "
                "and fused image sums when possible (%s=1, half_spectrum_scoring=%s, "
                "score_with_masked_images=%s)",
                _SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV,
                bool(half_spectrum_scoring),
                bool(score_with_masked_images),
            )
        if native_dual_weighted_sums:
            logger.info(
                "Sparse fused K-class compact-pair M-step/noise image sums use the "
                "guarded native dual reduction (%s=1)",
                _SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV,
            )
        if fused_mstep_noise:
            logger.info(
                "Sparse fused K-class compact-pair weighted sums and dense "
                "noise/norm statistics share one JIT boundary (%s=1)",
                _SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV,
            )
        if parse_env_flag(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False):
            raise ValueError(
                f"{_COMPACT_KCLASS_PAIRS_CHECK_ENV}=1 cannot be combined with "
                f"{_SPARSE_KCLASS_COMPACT_PAIRS_ENV}=1",
            )
    elif parse_env_flag(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False):
        if relion_firstiter_score_mode != "gaussian":
            logger.warning(
                "Sparse fused K-class compact-pair check skipped for score mode %s; "
                "only Gaussian compact-pair scoring is implemented",
                relion_firstiter_score_mode,
            )
        else:
            compact_pair_inputs_by_class_for_check = tuple(
                _prepare_per_image_compact_candidate_pairs(per_image_inputs)
                for per_image_inputs in per_image_inputs_by_class
            )
            logger.info(
                "Sparse fused K-class compact-pair score check enabled via %s=1; "
                "dense scoring/M-step remains authoritative",
                _COMPACT_KCLASS_PAIRS_CHECK_ENV,
            )
    if rectangular_active_rows:
        logger.info(
            "Sparse fused K-class rectangular active rows enabled via %s=1 "
            "(min_bucket_size=%d via %s, prematmul=%s via %s, "
            "prematmul_max_grouped_dense_ratio=%.3g via %s, pad_multiple=%d via %s)",
            _SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_ENV,
            int(rectangular_active_rows_min_bucket_size),
            _SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE_ENV,
            "1" if rectangular_active_prematmul else "0",
            _SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_ENV,
            float(rectangular_active_prematmul_max_grouped_dense_ratio),
            _SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO_ENV,
            int(active_row_pad_multiple),
            _SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV,
        )
    logger.info(
        "Sparse fused K-class pass-2 setup timing: hypothesis_prep=%.2fs bucket=%.2fs",
        prep_s,
        bucket_s,
    )
    if use_relion_x_half_mstep:
        mstep_layout_label = "RELION x-half BPref-layout"
    elif use_half_volume_mstep:
        mstep_layout_label = "native half-volume"
    else:
        mstep_layout_label = "full-volume"
    logger.info(
        "Sparse fused K-class pass-2 M-step: using %s backprojection",
        mstep_layout_label,
    )
    compact_buckets = parse_env_flag(_SPARSE_KCLASS_COMPACT_BUCKETS_ENV, default=False)
    if compact_buckets:
        logger.info(
            "Sparse fused K-class compact buckets enabled via %s=1; default rectangular fused path unchanged",
            _SPARSE_KCLASS_COMPACT_BUCKETS_ENV,
        )

    Ft_y_total = [jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype) for _ in range(n_classes)]
    Ft_ctf_total = [jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype) for _ in range(n_classes)]
    class_hard_assignments = np.empty((n_classes, n_images), dtype=np.int32)
    best_rotations = [
        np.empty((n_images, 3, 3), dtype=precision_policy.score_real_dtype) for _ in range(n_classes)
    ]
    best_eulers = (
        [np.empty((n_images, 3), dtype=np.float64) for _ in range(n_classes)]
        if return_source_eulers
        and all(x is not None for inputs in per_image_inputs_by_class for x in inputs["source_eulers"])
        else None
    )
    best_rotation_indices = [np.empty(n_images, dtype=np.int64) for _ in range(n_classes)]
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_score_log_z = np.empty((n_classes, n_images), dtype=np.float64)
    best_log_score = np.empty((n_classes, n_images), dtype=np.float64)
    max_posterior = np.empty((n_classes, n_images), dtype=precision_policy.score_real_dtype)
    rotation_posterior_sums = np.zeros((n_classes, n_coarse_rot), dtype=np.float64)
    class_posterior_sums_mstep = np.zeros(n_classes, dtype=np.float64)
    compact_pair_check_max_abs_diff = 0.0
    compact_pair_check_rows = 0
    compact_pair_check_finite_mismatches = 0
    compact_pair_noise_sum_reuses = 0
    compact_pair_noise_ctf_sum_reuses = 0
    compact_pair_noise_image_sum_precomputes = 0
    compact_pair_noise_fused_active_gathers = 0

    noise_scale_correction_xa_total = None
    noise_scale_correction_aa_total = None
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )
    if group_ids is not None:
        noise_scale_correction_xa_total = np.zeros((n_classes, n_scale_groups), dtype=np.float64)
        noise_scale_correction_aa_total = np.zeros((n_classes, n_scale_groups), dtype=np.float64)

    noise_wsum_total = [None] * n_classes
    noise_img_power_total = [None] * n_classes
    noise_norm_correction_total = [None] * n_classes
    noise_sumw_total = np.zeros(n_classes, dtype=np.float64)
    noise_sigma2_offset_total = np.zeros(n_classes, dtype=np.float64)
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        noise_wsum_total = [np.zeros(n_shells, dtype=np.float64) for _ in range(n_classes)]
        noise_img_power_total = [np.zeros(n_shells, dtype=np.float64) for _ in range(n_classes)]
        noise_norm_correction_total = [np.zeros(n_images, dtype=np.float64) for _ in range(n_classes)]

    window_setup = _sparse_pass2_window_setup(
        experiment_dataset,
        disc_type=disc_type,
        image_shape=image_shape,
        current_size=current_size,
        n_half=n_half,
        mstep_current_size=mstep_current_size,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        log_label="Sparse fused K-class pass-2",
    )
    config = window_setup.config
    window_spec = window_setup.window_spec
    use_window = window_setup.use_window
    window_indices_np = window_setup.window_indices_np
    window_indices = window_setup.window_indices
    recon_window_indices = window_setup.recon_window_indices
    relion_x_half_recon_indices = window_setup.relion_x_half_recon_indices
    windowed_prepare = window_setup.windowed_prepare
    n_windowed = window_setup.n_windowed
    n_recon_windowed = window_setup.n_recon_windowed

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=relion_firstiter_score_mode != "normalized_cc",
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)
    direct_half_weights = half_weights_windowed if use_window else half_weights
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            image_shape,
            current_size,
            window_indices_np if use_window else np.arange(int(n_half), dtype=np.int32),
        ),
        dtype=jnp.int32,
    )

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(shared_noise_variance, image_shape).squeeze()
    if accumulate_noise:
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        if use_window:
            shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
                shell_indices_half,
                image_shape,
                current_size,
                window_indices,
            )
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        scale_dvp = scale_correction_data_vs_prior
        if scale_dvp is None:
            scale_dvp_per_class = [None] * n_classes
        else:
            scale_dvp_array = np.asarray(scale_dvp)
            if scale_dvp_array.ndim == 1:
                scale_dvp_per_class = [scale_dvp_array] * n_classes
            elif scale_dvp_array.ndim == 2 and scale_dvp_array.shape[0] == n_classes:
                scale_dvp_per_class = [scale_dvp_array[k] for k in range(n_classes)]
            else:
                raise ValueError(
                    "scale_correction_data_vs_prior must be one shell vector or have "
                    f"shape ({n_classes}, n_shells), got {scale_dvp_array.shape}"
                )
        scale_correction_pixel_masks = [
            _relion_scale_correction_pixel_mask(dvp_k, shell_indices_noise, n_shells=n_shells)
            for dvp_k in scale_dvp_per_class
        ]

    projection_cache_by_class = [None] * n_classes
    dump_pass2_operands = _pass2_dump_enabled()
    if _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations_override,
        dump_pass2_operands=dump_pass2_operands,
    ):
        n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
        if use_window:
            # See the single-class cache above: cap the full-half RELION
            # projector transient per call, but admit the retained cache by
            # the stored windowed projection rows.
            transient_projection_bytes = _projection_cache_transient_bytes(
                n_fine_rot,
                n_windowed,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_abs2=False,
            )
            transient_projection_bytes += _projection_cache_transient_bytes(
                n_fine_rot,
                n_recon_windowed,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_abs2=True,
            )
        else:
            transient_projection_bytes = _projection_cache_transient_bytes(
                n_fine_rot,
                n_half,
                projection_complex_dtype=precision_policy.score_complex_dtype,
                include_abs2=True,
            )
        max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
        if _projection_cache_fits_budget(
            transient_projection_bytes,
            max_projection_cache_bytes,
            n_classes=n_classes,
        ):
            for class_index in range(n_classes):
                cache_t0 = time.time()
                if use_window:
                    projection_kwargs = window_spec.projection_kwargs(return_abs2=False)
                    projection_kwargs["mask_current_image_disk"] = bool(
                        projection_mask_current_image_disk
                    )
                    score_cache, recon_cache, recon_abs2_cache = _compute_sparse_pass2_windowed_projections_block(
                        mean_for_proj_by_class[class_index],
                        jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype),
                        image_shape,
                        proj_volume_shape,
                        disc_type,
                        score_indices=window_indices,
                        recon_indices=recon_window_indices,
                        max_projected_rotations=max_projected_rotations_per_projection_call,
                        output_complex_dtype=precision_policy.score_complex_dtype,
                        output_abs2_dtype=precision_policy.score_real_dtype,
                        relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                        **projection_kwargs,
                    )
                    projection_cache_by_class[class_index] = {
                        "score": score_cache,
                        "recon": recon_cache,
                        "recon_abs2": recon_abs2_cache,
                    }
                else:
                    projection_kwargs = window_spec.projection_kwargs(return_abs2=None)
                    projection_kwargs["mask_current_image_disk"] = bool(
                        projection_mask_current_image_disk
                    )
                    proj_half_cache_flat, proj_abs2_cache_flat = _compute_sparse_pass2_projections_block(
                        mean_for_proj_by_class[class_index],
                        jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype),
                        image_shape,
                        proj_volume_shape,
                        disc_type,
                        max_projected_rotations=max_projected_rotations_per_projection_call,
                        output_complex_dtype=precision_policy.score_complex_dtype,
                        output_abs2_dtype=precision_policy.score_real_dtype,
                        relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                        **projection_kwargs,
                    )
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
                "Sparse fused K-class pass-2 projection cache skipped: estimated total transient %.2f GiB "
                "exceeds cap %.2f GiB",
                (transient_projection_bytes * n_classes) / float(1024**3),
                max_projection_cache_bytes / float(1024**3),
            )

    if compact_pairs:
        if compact_pair_min_bucket_size is not None or compact_pair_execution_image_mask is not None:
            execution_buckets = _hybrid_k_class_compact_pair_execution_buckets(
                buckets,
                compact_pair_buckets or [],
                min_pair_bucket_size=1
                if compact_pair_min_bucket_size is None
                else compact_pair_min_bucket_size,
            )
        else:
            execution_buckets = [
                _tag_k_class_execution_bucket(bucket, mode="compact_pair")
                for bucket in (compact_pair_buckets or [])
            ]
    else:
        execution_buckets = [
            _tag_k_class_execution_bucket(bucket, mode="rectangular")
            for bucket in buckets
        ]
    if not execution_buckets:
        execution_buckets = [
            _tag_k_class_execution_bucket(bucket, mode="rectangular")
            for bucket in buckets
        ]
    _validate_k_class_execution_bucket_partition(execution_buckets, n_images=n_images)
    relion_score_translation_angles = (
        _relion_cuda_score_translation_angles_if_available(
            fine_translations,
            image_shape,
            enabled=use_exact_relion_gaussian,
            dtype=np.float64 if use_float64_scoring else np.float32,
        )
    )
    translation_phases_half = None if windowed_prepare else half_translation_phase_table(fine_translations, image_shape)
    score_translation_phases = None
    recon_translation_phases = None
    if windowed_prepare:
        score_translation_phases = _translation_phase_table_for_indices(
            fine_translations,
            image_shape,
            window_indices,
            None,
        )
        recon_translation_phases = _translation_phase_table_for_indices(
            fine_translations,
            image_shape,
            recon_window_indices,
            None,
        )
        logger.info(
            "Sparse fused K-class pass-2 windowed translation phases cached "
            "(score_pixels=%d recon_pixels=%d translations=%d)",
            int(n_windowed),
            int(n_recon_windowed),
            int(n_fine_trans),
        )
    compact_pair_execution_buckets = [
        bucket for bucket in execution_buckets if bucket["_execution_mode"] == "compact_pair"
    ]
    rectangular_execution_buckets = [
        bucket for bucket in execution_buckets if bucket["_execution_mode"] == "rectangular"
    ]
    if compact_pairs:
        logger.info(
            "Sparse fused K-class compact-pair execution routing: compact_pair_buckets=%d images=%d; "
            "rectangular_buckets=%d images=%d",
            len(compact_pair_execution_buckets),
            sum(len(bucket["image_indices"]) for bucket in compact_pair_execution_buckets),
            len(rectangular_execution_buckets),
            sum(len(bucket["image_indices"]) for bucket in rectangular_execution_buckets),
        )
    bucket_group_stats = _k_class_execution_bucket_group_stats(execution_buckets)
    profile_group_timing = os.environ.get(_SPARSE_KCLASS_GROUP_TIMING_ENV) == "1"
    last_bucket_size_logged = None
    group_t0 = None
    group_timing = None
    overall_t0 = time.time()
    rectangular_rotation_slots = 0
    compact_rotation_slots = 0
    compact_mstep_active_rows = 0
    compact_mstep_padded_active_rows = 0
    compact_mstep_rectangular_rows = 0
    compact_noise_active_rows = 0
    compact_noise_padded_active_rows = 0
    compact_noise_rectangular_rows = 0
    rectangular_mstep_active_rows = 0
    rectangular_mstep_padded_active_rows = 0
    rectangular_mstep_rectangular_rows = 0
    rectangular_active_prematmul_attempts = 0
    rectangular_active_prematmul_used = 0
    rectangular_active_prematmul_skipped = 0
    rectangular_active_prematmul_grouped_rows = 0
    rectangular_active_prematmul_dense_rows = 0
    rectangular_noise_active_rows = 0
    rectangular_noise_padded_active_rows = 0
    rectangular_noise_rectangular_rows = 0
    raw_host_staging_max_bytes = _optional_positive_int_env(
        _SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES_ENV,
    )
    if raw_host_staging_max_bytes is None:
        raw_host_staging_max_bytes = _DEFAULT_KCLASS_RAW_HOST_STAGING_MAX_BYTES
    raw_host_staging_total_bytes = 0
    raw_host_staging_peak_bytes = 0
    raw_host_staging_s = 0.0
    raw_host_staging_dtype = np.dtype(
        np.float64 if precision_policy.use_float64_scoring else np.float32
    )

    def _stage_raw_diff2_on_host(raw_diff2, current_bucket_bytes):
        nonlocal raw_host_staging_total_bytes
        nonlocal raw_host_staging_peak_bytes
        nonlocal raw_host_staging_s

        raw_nbytes = int(raw_diff2.size) * raw_host_staging_dtype.itemsize
        next_bucket_bytes = int(current_bucket_bytes) + raw_nbytes
        if next_bucket_bytes > int(raw_host_staging_max_bytes):
            raise MemoryError(
                "fused K-class raw diff2 host staging would exceed its hard cap: "
                f"requested={next_bucket_bytes} bytes, cap={raw_host_staging_max_bytes} bytes. "
                f"Increase {_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES_ENV} or lower the "
                "sparse pass-2 hypothesis microbatch cap."
            )
        stage_t0 = time.time()
        raw_host = np.asarray(raw_diff2, dtype=raw_host_staging_dtype)
        raw_host_staging_s += time.time() - stage_t0
        raw_host_staging_total_bytes += raw_nbytes
        raw_host_staging_peak_bytes = max(raw_host_staging_peak_bytes, next_bucket_bytes)
        return raw_host, next_bucket_bytes

    for bucket_meta in execution_buckets:
        bucket_raw_host_staging_bytes = 0
        execution_mode = str(bucket_meta["_execution_mode"])
        execution_bucket_size_key = str(bucket_meta["_execution_size_key"])
        bucket_uses_compact_pairs = execution_mode == "compact_pair"
        image_indices = np.asarray(bucket_meta["image_indices"], dtype=np.int64)
        bucket_size = int(bucket_meta["_execution_bucket_size"])
        bucket_uses_rectangular_active_rows = (
            rectangular_active_rows
            and not bucket_uses_compact_pairs
            and bucket_size >= int(rectangular_active_rows_min_bucket_size)
        )
        bucket_uses_active_rows = (
            compact_active_rows and bucket_uses_compact_pairs
        ) or bucket_uses_rectangular_active_rows
        if fused_mstep_noise and bucket_uses_compact_pairs:
            # The fused wrapper consumes dense rotation rows before any host
            # materialization of the active-row index set.
            bucket_uses_active_rows = False
        if mstep_subtract_ctf_projection:
            # Residual VDAM accumulation needs the dense projected-reference
            # row tensor before adjoint packing.
            bucket_uses_active_rows = False
        group_key = (execution_mode, execution_bucket_size_key, bucket_size)
        if group_key != last_bucket_size_logged:
            if last_bucket_size_logged is not None and group_t0 is not None:
                prev_chunks, prev_images = bucket_group_stats[last_bucket_size_logged]
                prev_wall = time.time() - group_t0
                logger.info(
                    "Sparse fused K-class pass-2 bucket group done: mode=%s %s=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
                    last_bucket_size_logged[0],
                    last_bucket_size_logged[1],
                    last_bucket_size_logged[2],
                    prev_chunks,
                    prev_images,
                    prev_wall,
                    prev_images / max(prev_wall, 1e-9),
                )
                _log_sparse_kclass_group_timing(
                    last_bucket_size_logged,
                    group_timing,
                    wall_s=prev_wall,
                )
            group_chunks, group_images = bucket_group_stats[group_key]
            logger.info(
                "Sparse fused K-class pass-2 bucket group start: mode=%s %s=%d chunks=%d images=%d",
                execution_mode,
                execution_bucket_size_key,
                bucket_size,
                group_chunks,
                group_images,
            )
            last_bucket_size_logged = group_key
            group_t0 = time.time()
            group_timing = {} if profile_group_timing else None
        stage_t0 = time.time()
        class_bucket_arrays = _build_k_class_bucket_arrays(
            bucket_meta,
            per_image_inputs_by_class,
            n_fine_trans,
            compact_buckets=bucket_uses_compact_pairs or compact_buckets,
            include_dense_score_fields=not bucket_uses_compact_pairs,
            rotation_block_size_for_quantization=rotation_block_size_for_quantization,
        )
        if parse_env_flag(
            _SPARSE_KCLASS_EXECUTION_SIGNATURES_ENV,
            default=False,
        ):
            print(
                "VDAM_EXECUTION_SIGNATURE "
                f"mode={execution_mode} {execution_bucket_size_key}={bucket_size} "
                f"batch={int(image_indices.shape[0])} "
                "class_bucket_sizes="
                f"{tuple(int(arrays['bucket_size']) for arrays in class_bucket_arrays)}",
                flush=True,
            )
        compact_pair_arrays_by_class = None
        if bucket_uses_compact_pairs:
            compact_pair_arrays_by_class = [
                _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket_meta, per_image_inputs)
                for per_image_inputs in per_image_inputs_by_class
            ]
        batch = int(image_indices.shape[0])
        _add_sparse_group_timing(group_timing, "build", time.time() - stage_t0)
        if bucket_uses_compact_pairs:
            rectangular_rotation_slots += (
                int(n_classes)
                * max(int(arrays["bucket_size"]) for arrays in class_bucket_arrays)
                * batch
            )
        else:
            rectangular_rotation_slots += int(n_classes) * int(bucket_size) * batch
        compact_rotation_slots += sum(int(arrays["bucket_size"]) for arrays in class_bucket_arrays) * batch
        stage_t0 = time.time()
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
        batch_data = jnp.asarray(batch_data)
        if not np.array_equal(np.asarray(fetched_indices), image_indices):
            fetched_indices_np = np.asarray(fetched_indices)
            reordered = []
            for arrays in class_bucket_arrays:
                shared_mstep_rotations = arrays["mstep_rotations"] is arrays["rotations"]
                if arrays["log_prior"] is None:
                    if shared_mstep_rotations:
                        rotations, rotation_indices, actual_counts = _reorder_to_indices(
                            fetched_indices_np,
                            image_indices,
                            arrays["rotations"],
                            arrays["rotation_indices"],
                            arrays["actual_counts"],
                        )
                        mstep_rotations = rotations
                    else:
                        rotations, mstep_rotations, rotation_indices, actual_counts = _reorder_to_indices(
                            fetched_indices_np,
                            image_indices,
                            arrays["rotations"],
                            arrays["mstep_rotations"],
                            arrays["rotation_indices"],
                            arrays["actual_counts"],
                        )
                    log_prior = None
                    candidate_mask = None
                    parent_map_padded = None
                else:
                    if shared_mstep_rotations:
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
                        mstep_rotations = rotations
                    else:
                        (
                            rotations,
                            mstep_rotations,
                            rotation_indices,
                            log_prior,
                            candidate_mask,
                            parent_map_padded,
                            actual_counts,
                        ) = _reorder_to_indices(
                            fetched_indices_np,
                            image_indices,
                            arrays["rotations"],
                            arrays["mstep_rotations"],
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
                        "mstep_rotations": mstep_rotations,
                        "rotation_indices": rotation_indices,
                        "log_prior": log_prior,
                        "candidate_mask": candidate_mask,
                        "parent_map": parent_map_padded,
                    }
                )
            class_bucket_arrays = reordered
            if compact_pair_arrays_by_class is not None:
                reordered_compact_pairs = []
                for pair_arrays in compact_pair_arrays_by_class:
                    (
                        pair_counts,
                        local_rotation_row,
                        translation_idx,
                        rotation_index,
                        pair_log_prior,
                        pair_mask,
                    ) = _reorder_to_indices(
                        fetched_indices_np,
                        image_indices,
                        pair_arrays["pair_counts"],
                        pair_arrays["local_rotation_row"],
                        pair_arrays["translation_idx"],
                        pair_arrays["rotation_index"],
                        pair_arrays["log_prior"],
                        pair_arrays["pair_mask"],
                    )
                    reordered_compact_pairs.append(
                        {
                            "image_indices": fetched_indices_np,
                            "pair_bucket_size": pair_arrays["pair_bucket_size"],
                            "pair_counts": pair_counts,
                            "local_rotation_row": local_rotation_row,
                            "translation_idx": translation_idx,
                            "rotation_index": rotation_index,
                            "log_prior": pair_log_prior,
                            "pair_mask": pair_mask,
                        }
                    )
                compact_pair_arrays_by_class = reordered_compact_pairs
            image_indices = fetched_indices_np
        pass2_dump_rows = (
            pass2_diagnostics._pass2_dump_target_rows(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
            )
            if dump_pass2_operands
            else np.empty((0,), dtype=np.int64)
        )
        target_particle_rows = (
            bpref_diagnostics._bpref_contribution_target_rows(experiment_dataset, image_indices)
            if device_signature_requested
            else np.empty((0,), dtype=np.int64)
        )
        bucket_diagnostic_modes = bpref_diagnostics._resolve_bpref_bucket_diagnostic_modes(
            device_signature_requested=device_signature_requested,
            contribution_diagnostics_active=bool(
                os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip()
                and bpref_device_signature_active
            ),
            target_particle_rows=target_particle_rows,
            high_precision_operand_bundle_requested=scoped_diagnostic_flags[
                "high_precision_operand_bundle"
            ],
        )
        bucket_device_signature_requested = bucket_diagnostic_modes[
            "device_signature_requested"
        ]
        high_precision_operand_bundle = bucket_diagnostic_modes[
            "high_precision_operand_bundle"
        ]
        contribution_preprocess_operands = None
        if high_precision_operand_bundle:
            diagnostic_preprocess_operands = prepare_batch_preprocess_operands(
                experiment_dataset,
                batch_data,
                image_indices,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
            )
            contribution_preprocess_operands = bpref_diagnostics.build_bpref_preprocess_capture(
                experiment_dataset,
                image_shape,
                diagnostic_preprocess_operands,
                batch=batch,
                score_with_masked_images=score_with_masked_images,
            )
        bucket_group_ids = (
            jnp.asarray(group_ids_np[image_indices], dtype=jnp.int32)
            if group_ids_np is not None
            else None
        )
        bucket_scale_for_stats = (
            jnp.asarray(np.asarray(scale_corrections, dtype=precision_policy.score_real_dtype)[image_indices])
            if scale_corrections is not None
            else jnp.ones(batch, dtype=precision_policy.score_real_dtype)
        )
        _add_sparse_group_timing(group_timing, "fetch", time.time() - stage_t0)

        stage_t0 = time.time()
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
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=precision_policy.score_real_dtype)
        else:
            bucket_translation_prior = jnp.asarray(
                fine_translation_prior_2d[image_indices], dtype=precision_policy.score_real_dtype
            )

        (
            shifted_score_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            ctf2_over_nv_half_with_dc,
            shifted_score_half_with_dc,
            processed_score_half_for_noise,
            shifted_corrected_score_half,
            direct_score_input,
            _direct_preprocessed_score_input,
            _direct_pixel_correction,
            _direct_preprocess_normalization_factors,
            _direct_integer_pre_shifts,
            _direct_batch_image_corrections,
            _direct_batch_scale_corrections,
            _direct_inverse_noise_half,
            _direct_ctf_rfloat_half,
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
            recon_window_indices=recon_window_indices,
            translation_phases_half=translation_phases_half,
            score_translation_phases=score_translation_phases,
            recon_translation_phases=recon_translation_phases,
            relion_score_translation_angles=relion_score_translation_angles,
            return_windowed_shifted=windowed_prepare,
            return_shifted_score=not half_spectrum_scoring,
            relion_exact_bpref_operands=relion_exact_bpref_operands,
        )
        relion_highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        if use_window:
            ctf2_over_nv_score = ctf2_over_nv_half if windowed_prepare else ctf2_over_nv_half[:, window_indices]
            shifted_corrected_score = (
                shifted_corrected_score_half if windowed_prepare else shifted_corrected_score_half[:, window_indices]
            )
            shifted_score = (
                None
                if half_spectrum_scoring
                else shifted_score_half if windowed_prepare else shifted_score_half[:, window_indices]
            )
            shifted_recon = shifted_recon_half if windowed_prepare else shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc if windowed_prepare else ctf2_over_nv_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_score_half_with_dc if windowed_prepare else shifted_score_half_with_dc[:, recon_window_indices]
        else:
            ctf2_over_nv_score = ctf2_over_nv_half
            shifted_corrected_score = shifted_corrected_score_half
            shifted_score = None if half_spectrum_scoring else shifted_score_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
            shifted_noise = shifted_score_half_with_dc

        shifted_corrected_score_split = shifted_corrected_score.reshape(batch, n_fine_trans, -1)
        _add_sparse_group_timing(group_timing, "prepare", time.time() - stage_t0)
        scores_by_class = []
        class_score_log_z_bucket = []
        raw_diff2_by_class = []
        raw_diff2_masks_by_class = []
        raw_diff2_rotation_priors_by_class = []
        raw_diff2_translation_priors_by_class = []
        raw_diff2_dump_by_class = [None] * n_classes
        raw_operand_dump_by_class = [None] * n_classes
        score_projection_for_compact_check_by_class = []
        flat_backproject_rotations_by_class = []
        proj_for_noise_by_class = []
        proj_abs2_by_class = []
        stage_t0 = time.time()
        for class_index, arrays in enumerate(class_bucket_arrays):
            class_bucket_size = int(arrays["bucket_size"])
            flat_rotations = flatten_bucket_rotations(jnp.asarray(arrays["rotations"]))
            flat_backproject_rotations_by_class.append(
                flat_rotations
                if arrays["mstep_rotations"] is arrays["rotations"]
                else flatten_bucket_rotations(jnp.asarray(arrays["mstep_rotations"]))
            )
            cache = projection_cache_by_class[class_index]
            defer_compact_recon_projection = False
            identity_full_cache_rows = False
            cached_score_2d = None
            if cache is not None:
                rotation_indices_np = np.asarray(arrays["rotation_indices"], dtype=np.int64)
                cache_score = cache["score"]
                cache_recon = cache["recon"]
                cache_recon_abs2 = cache["recon_abs2"]
                identity_full_cache_rows = (
                    int(batch) == 1
                    and rotation_indices_np.shape == (1, int(cache_score.shape[0]))
                    and int(class_bucket_size) == int(cache_score.shape[0])
                    and np.array_equal(rotation_indices_np[0], np.arange(int(cache_score.shape[0]), dtype=np.int64))
                )
                if identity_full_cache_rows:
                    # Full-support K=1/firstiter buckets already have all fine
                    # rotations in cache order.  Gathering with an explicit
                    # ``(1, R)`` index array duplicates the multi-GiB projection
                    # cache and can OOM before scoring starts.
                    cached_score_2d = cache_score
                    proj_half = cache_score[jnp.newaxis, :, :]
                    proj_for_noise = cache_recon[jnp.newaxis, :, :]
                    proj_abs2_for_noise = cache_recon_abs2[jnp.newaxis, :, :]
                else:
                    rotation_indices_jax = jnp.asarray(rotation_indices_np, dtype=jnp.int32)
                    proj_half = cache_score[rotation_indices_jax]
                    proj_for_noise = cache_recon[rotation_indices_jax]
                    proj_abs2_for_noise = cache_recon_abs2[rotation_indices_jax]
            else:
                projection_kwargs = window_spec.projection_kwargs(return_abs2=False if use_window else None)
                projection_kwargs["mask_current_image_disk"] = bool(
                    projection_mask_current_image_disk
                )
                if use_window:
                    retained_window_projection_bytes = (
                        int(flat_rotations.shape[0])
                        * (int(n_windowed) + int(n_recon_windowed))
                        * _dtype_itemsize(precision_policy.score_complex_dtype)
                    )
                    defer_compact_recon_projection = bool(
                        bucket_uses_compact_pairs
                        and retained_window_projection_bytes > int(max_projection_gather_bytes)
                    )
                    if defer_compact_recon_projection:
                        proj_half_flat, _, _ = _compute_sparse_pass2_windowed_projections_block(
                            mean_for_proj_by_class[class_index],
                            flat_rotations,
                            image_shape,
                            proj_volume_shape,
                            disc_type,
                            score_indices=window_indices,
                            recon_indices=None,
                            max_projected_rotations=max_projected_rotations_per_projection_call,
                            output_complex_dtype=precision_policy.score_complex_dtype,
                            output_abs2_dtype=None,
                            relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                            relion_projector_r_max=relion_projector_r_max,
                            projection_padding_factor=projection_padding_factor,
                            **projection_kwargs,
                        )
                        proj_half = proj_half_flat.reshape(batch, class_bucket_size, n_windowed)
                        proj_for_noise = None
                        proj_abs2_for_noise = None
                    else:
                        proj_half_flat, proj_for_noise_flat, proj_abs2_for_noise_flat = (
                            _compute_sparse_pass2_windowed_projections_block(
                                mean_for_proj_by_class[class_index],
                                flat_rotations,
                                image_shape,
                                proj_volume_shape,
                                disc_type,
                                score_indices=window_indices,
                                recon_indices=recon_window_indices,
                                max_projected_rotations=max_projected_rotations_per_projection_call,
                                output_complex_dtype=precision_policy.score_complex_dtype,
                                output_abs2_dtype=precision_policy.score_real_dtype,
                                relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                                relion_projector_r_max=relion_projector_r_max,
                                projection_padding_factor=projection_padding_factor,
                                **projection_kwargs,
                            )
                        )
                        proj_half = proj_half_flat.reshape(batch, class_bucket_size, n_windowed)
                        proj_for_noise = proj_for_noise_flat.reshape(batch, class_bucket_size, n_recon_windowed)
                        proj_abs2_for_noise = proj_abs2_for_noise_flat.reshape(batch, class_bucket_size, n_recon_windowed)
                else:
                    proj_half_flat, proj_abs2_half_flat = _compute_sparse_pass2_projections_block(
                        mean_for_proj_by_class[class_index],
                        flat_rotations,
                        image_shape,
                        proj_volume_shape,
                        disc_type,
                        max_projected_rotations=max_projected_rotations_per_projection_call,
                        output_complex_dtype=precision_policy.score_complex_dtype,
                        output_abs2_dtype=precision_policy.score_real_dtype,
                        relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                        **projection_kwargs,
                    )
                    proj_half = proj_half_flat.reshape(batch, class_bucket_size, n_half)
                    proj_abs2_for_noise = proj_abs2_half_flat.reshape(batch, class_bucket_size, n_half)
                    proj_for_noise = proj_half
            if not (cache is None and use_window and defer_compact_recon_projection):
                proj_for_noise, proj_abs2_for_noise = precision_policy.cast_local_noise_projection_scores(
                    proj_for_noise,
                    proj_abs2_for_noise,
                )
            compact_arrays = (
                None
                if compact_pair_arrays_by_class is None
                else compact_pair_arrays_by_class[class_index]
            )
            if bucket_uses_compact_pairs:
                pair_mask = jnp.asarray(compact_arrays["pair_mask"])
                if relion_firstiter_score_mode == "normalized_cc":
                    scores = _score_pass2_pairs_normalized_cc(
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        proj_half,
                        direct_half_weights,
                        jnp.asarray(compact_arrays["local_rotation_row"]),
                        jnp.asarray(compact_arrays["translation_idx"]),
                        pair_mask,
                    )
                    class_log_z_for_bucket = _logsumexp_pass2_pairs_score_only(scores, pair_mask)
                elif use_exact_relion_gaussian:
                    local_rotation_row = jnp.asarray(compact_arrays["local_rotation_row"])
                    translation_idx = jnp.asarray(compact_arrays["translation_idx"])
                    raw_diff2 = _score_pass2_pairs_relion_gpu_diff2_raw(
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        proj_half,
                        direct_half_weights,
                        local_rotation_row,
                        translation_idx,
                        pair_mask,
                        relion_score_full_to_compact,
                        relion_highres_xi2_half,
                        use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                    )
                    row = jnp.arange(batch)[:, None]
                    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
                    # The joint minimum is not known until every class has
                    # scored. Offload each raw partition immediately so K
                    # device-resident raw tensors cannot overlap the K score
                    # tensors built below.
                    raw_host, bucket_raw_host_staging_bytes = _stage_raw_diff2_on_host(
                        raw_diff2,
                        bucket_raw_host_staging_bytes,
                    )
                    raw_diff2_by_class.append(raw_host)
                    raw_diff2_masks_by_class.append(pair_mask)
                    raw_diff2_rotation_priors_by_class.append(
                        jnp.asarray(
                            compact_arrays["log_prior"],
                            dtype=precision_policy.score_real_dtype,
                        )
                    )
                    raw_diff2_translation_priors_by_class.append(
                        jnp.asarray(
                            bucket_translation_prior[row, safe_translation_idx],
                            dtype=precision_policy.score_real_dtype,
                        )
                    )
                    scores = None
                    class_log_z_for_bucket = None
                else:
                    scores = _score_pass2_pairs_gaussian_algebraic(
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        proj_half,
                        direct_half_weights,
                        jnp.asarray(compact_arrays["log_prior"]),
                        bucket_translation_prior,
                        jnp.asarray(compact_arrays["local_rotation_row"]),
                        jnp.asarray(compact_arrays["translation_idx"]),
                        pair_mask,
                    )
                    class_log_z_for_bucket = _logsumexp_pass2_pairs_score_only(scores, pair_mask)
            else:
                if relion_firstiter_score_mode == "normalized_cc":
                    if identity_full_cache_rows and cached_score_2d is not None:
                        scores = _score_pass2_bucket_normalized_cc_single_cached(
                            shifted_corrected_score_split[0],
                            ctf2_over_nv_score[0],
                            cached_score_2d,
                            direct_half_weights,
                            jnp.asarray(arrays["candidate_mask"][0]),
                        )[jnp.newaxis, :, :]
                    else:
                        scores = _score_pass2_bucket_normalized_cc(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_half,
                            direct_half_weights,
                            jnp.asarray(arrays["candidate_mask"]),
                        )
                elif use_exact_relion_gaussian:
                    if identity_full_cache_rows and cached_score_2d is not None:
                        raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
                            shifted_corrected_score_split[0],
                            ctf2_over_nv_score[0],
                            cached_score_2d,
                            direct_half_weights,
                            relion_score_full_to_compact,
                            relion_highres_xi2_half[0],
                            use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                        )[jnp.newaxis, :, :]
                    else:
                        raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_half,
                            direct_half_weights,
                            relion_score_full_to_compact,
                            relion_highres_xi2_half,
                            use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                        )
                    # Keep the inter-class staging on the host. The score
                    # microbatch cap applies to device residency; retaining
                    # all raw class tensors here would nearly double its peak.
                    raw_host, bucket_raw_host_staging_bytes = _stage_raw_diff2_on_host(
                        raw_diff2,
                        bucket_raw_host_staging_bytes,
                    )
                    raw_diff2_by_class.append(raw_host)
                    raw_diff2_masks_by_class.append(jnp.asarray(arrays["candidate_mask"]))
                    raw_diff2_rotation_priors_by_class.append(
                        jnp.asarray(
                            arrays["log_prior"],
                            dtype=precision_policy.score_real_dtype,
                        )[:, :, None]
                    )
                    raw_diff2_translation_priors_by_class.append(
                        jnp.asarray(
                            bucket_translation_prior,
                            dtype=precision_policy.score_real_dtype,
                        )[:, None, :]
                    )
                    scores = None
                    class_log_z_for_bucket = None
                else:
                    if identity_full_cache_rows and cached_score_2d is not None:
                        scores = _score_pass2_bucket_gaussian_algebraic_single_cached(
                            shifted_corrected_score_split[0],
                            ctf2_over_nv_score[0],
                            cached_score_2d,
                            direct_half_weights,
                            jnp.asarray(arrays["log_prior"][0]),
                            bucket_translation_prior[0],
                            jnp.asarray(arrays["candidate_mask"][0]),
                        )[jnp.newaxis, :, :]
                    else:
                        scores = _score_pass2_bucket_gaussian_algebraic(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_half,
                            direct_half_weights,
                            jnp.asarray(arrays["log_prior"]),
                            bucket_translation_prior,
                            jnp.asarray(arrays["candidate_mask"]),
                        )
                if not use_exact_relion_gaussian:
                    class_log_z_for_bucket = _logsumexp_pass2_bucket_score_only(scores)
            target_dump_class = os.environ.get("RECOVAR_PASS2_DUMP_CLASS")
            if (
                use_exact_relion_gaussian
                and pass2_dump_rows.size
                and parse_env_flag(
                    pass2_diagnostics._PASS2_DUMP_RAW_OPERANDS_ENV,
                    default=False,
                )
                and (
                    not target_dump_class
                    or int(target_dump_class) == class_index + 1
                )
            ):
                raw_operand_dump_by_class[class_index] = (
                    pass2_diagnostics._capture_k_class_pass2_raw_operands(
                        raw_diff2=raw_diff2,
                        target_rows=pass2_dump_rows,
                        actual_counts=arrays["actual_counts"],
                        shifted_corrected=shifted_corrected_score_split,
                        corr_img_score=ctf2_over_nv_score,
                        proj_half=proj_half,
                        half_weights=direct_half_weights,
                        relion_full_to_compact=relion_score_full_to_compact,
                        highres_xi2_half=relion_highres_xi2_half,
                        pair_mask=(
                            compact_pair_arrays_by_class[class_index][
                                "pair_mask"
                            ]
                            if bucket_uses_compact_pairs
                            else None
                        ),
                        pair_rotation_row=(
                            compact_pair_arrays_by_class[class_index][
                                "local_rotation_row"
                            ]
                            if bucket_uses_compact_pairs
                            else None
                        ),
                        pair_translation_idx=(
                            compact_pair_arrays_by_class[class_index][
                                "translation_idx"
                            ]
                            if bucket_uses_compact_pairs
                            else None
                        ),
                    )
                )
            scores_by_class.append(scores)
            score_projection_for_compact_check_by_class.append(
                proj_half if compact_pair_inputs_by_class_for_check is not None else None
            )
            if (
                compact_pair_inputs_by_class_for_check is not None
                and not use_exact_relion_gaussian
            ):
                compact_inputs = compact_pair_inputs_by_class_for_check[class_index]
                pair_counts = np.asarray(compact_inputs["pair_counts"], dtype=np.int64)[image_indices]
                pair_bucket_size = max(1, int(pair_counts.max(initial=0)))
                compact_arrays = _build_compact_pair_bucket_arrays(
                    {
                        "pair_bucket_size": pair_bucket_size,
                        "image_indices": image_indices,
                    },
                    compact_inputs,
                )
                compact_scores = _score_pass2_pairs_relion_gpu_diff2(
                    shifted_corrected_score_split,
                    ctf2_over_nv_score,
                    proj_half,
                    direct_half_weights,
                    jnp.asarray(compact_arrays["log_prior"]),
                    bucket_translation_prior,
                    jnp.asarray(compact_arrays["local_rotation_row"]),
                    jnp.asarray(compact_arrays["translation_idx"]),
                    jnp.asarray(compact_arrays["pair_mask"]),
                    relion_score_full_to_compact,
                )
                compact_log_z = _logsumexp_pass2_pairs_score_only(
                    compact_scores,
                    jnp.asarray(compact_arrays["pair_mask"]),
                )
                dense_log_z_np = np.asarray(class_log_z_for_bucket, dtype=np.float64)
                compact_log_z_np = np.asarray(compact_log_z, dtype=np.float64)
                dense_finite = np.isfinite(dense_log_z_np)
                compact_finite = np.isfinite(compact_log_z_np)
                both_finite = dense_finite & compact_finite
                if np.any(both_finite):
                    compact_pair_check_max_abs_diff = max(
                        compact_pair_check_max_abs_diff,
                        float(np.max(np.abs(dense_log_z_np[both_finite] - compact_log_z_np[both_finite]))),
                    )
                compact_pair_check_finite_mismatches += int(np.count_nonzero(dense_finite != compact_finite))
                compact_pair_check_rows += int(dense_log_z_np.size)
            class_score_log_z_bucket.append(class_log_z_for_bucket)
            if cache is None and use_window and defer_compact_recon_projection:
                try:
                    ready_value = (
                        raw_diff2
                        if use_exact_relion_gaussian
                        else class_log_z_for_bucket
                    )
                    ready_value.block_until_ready()
                except AttributeError:
                    pass
                del proj_half
                proj_for_noise_flat, _, _ = _compute_sparse_pass2_windowed_projections_block(
                    mean_for_proj_by_class[class_index],
                    flat_rotations,
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    score_indices=recon_window_indices,
                    recon_indices=None,
                    max_projected_rotations=max_projected_rotations_per_projection_call,
                    output_complex_dtype=precision_policy.score_complex_dtype,
                    output_abs2_dtype=None,
                    relion_projector_half=relion_projector_half[class_index] if use_relion_projector else None,
                    relion_projector_r_max=relion_projector_r_max,
                    projection_padding_factor=projection_padding_factor,
                    **projection_kwargs,
                )
                proj_for_noise = proj_for_noise_flat.reshape(batch, class_bucket_size, n_recon_windowed)
                proj_abs2_for_noise = jnp.abs(proj_for_noise) ** 2
                if precision_policy.score_real_dtype is not None:
                    proj_abs2_for_noise = proj_abs2_for_noise.astype(precision_policy.score_real_dtype)
                proj_for_noise, proj_abs2_for_noise = precision_policy.cast_local_noise_projection_scores(
                    proj_for_noise,
                    proj_abs2_for_noise,
                )
            proj_for_noise_by_class.append(proj_for_noise)
            proj_abs2_by_class.append(proj_abs2_for_noise)

        global_min_diff2 = None
        relion_min_diff2_dump = None
        if use_exact_relion_gaussian:
            if len(raw_diff2_by_class) != n_classes:
                raise RuntimeError(
                    "RELION Gaussian K-class scoring did not retain one raw diff2 tensor per class"
                )
            global_min_diff2 = _relion_cuda_fine_global_diff2_min(
                raw_diff2_by_class,
                raw_diff2_masks_by_class,
            )
            scores_by_class = []
            class_score_log_z_bucket = []
            for class_index, raw_diff2 in enumerate(raw_diff2_by_class):
                target_dump_class = os.environ.get(
                    "RECOVAR_PASS2_DUMP_CLASS"
                )
                if (
                    pass2_dump_rows.size
                    and (
                        not target_dump_class
                        or int(target_dump_class) == class_index + 1
                    )
                ):
                    raw_diff2_np = np.asarray(raw_diff2, dtype=raw_host_staging_dtype)
                    raw_diff2_dump_by_class[class_index] = {
                        int(row): np.array(raw_diff2_np[int(row)], copy=True)
                        for row in pass2_dump_rows
                    }
                score = _relion_cuda_fine_diff2_to_scores(
                    jnp.asarray(raw_diff2, dtype=precision_policy.score_real_dtype),
                    raw_diff2_rotation_priors_by_class[class_index],
                    raw_diff2_translation_priors_by_class[class_index],
                    raw_diff2_masks_by_class[class_index],
                    min_diff2=global_min_diff2,
                )
                scores_by_class.append(score)
                bucket_raw_host_staging_bytes -= int(raw_diff2.nbytes)
                raw_diff2_by_class[class_index] = None
                if bucket_uses_compact_pairs:
                    class_log_z_for_bucket = _logsumexp_pass2_pairs_score_only(
                        score,
                        raw_diff2_masks_by_class[class_index],
                    )
                else:
                    class_log_z_for_bucket = _logsumexp_pass2_bucket_score_only(score)
                class_score_log_z_bucket.append(class_log_z_for_bucket)

                if compact_pair_inputs_by_class_for_check is not None:
                    compact_inputs = compact_pair_inputs_by_class_for_check[class_index]
                    pair_counts = np.asarray(compact_inputs["pair_counts"], dtype=np.int64)[image_indices]
                    pair_bucket_size = max(1, int(pair_counts.max(initial=0)))
                    compact_arrays = _build_compact_pair_bucket_arrays(
                        {
                            "pair_bucket_size": pair_bucket_size,
                            "image_indices": image_indices,
                        },
                        compact_inputs,
                    )
                    pair_mask = jnp.asarray(compact_arrays["pair_mask"])
                    local_rotation_row = jnp.asarray(compact_arrays["local_rotation_row"])
                    translation_idx = jnp.asarray(compact_arrays["translation_idx"])
                    compact_raw_diff2 = _score_pass2_pairs_relion_gpu_diff2_raw(
                        shifted_corrected_score_split,
                        ctf2_over_nv_score,
                        score_projection_for_compact_check_by_class[class_index],
                        direct_half_weights,
                        local_rotation_row,
                        translation_idx,
                        pair_mask,
                        relion_score_full_to_compact,
                        relion_highres_xi2_half,
                        use_fused_ffi=use_relion_fine_diff2_fused_ffi,
                    )
                    row = jnp.arange(batch)[:, None]
                    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
                    compact_scores = _relion_cuda_fine_diff2_to_scores(
                        compact_raw_diff2,
                        jnp.asarray(
                            compact_arrays["log_prior"],
                            dtype=precision_policy.score_real_dtype,
                        ),
                        jnp.asarray(
                            bucket_translation_prior[row, safe_translation_idx],
                            dtype=precision_policy.score_real_dtype,
                        ),
                        pair_mask,
                        min_diff2=global_min_diff2,
                    )
                    compact_log_z = _logsumexp_pass2_pairs_score_only(compact_scores, pair_mask)
                    dense_log_z_np = np.asarray(class_log_z_for_bucket, dtype=np.float64)
                    compact_log_z_np = np.asarray(compact_log_z, dtype=np.float64)
                    dense_finite = np.isfinite(dense_log_z_np)
                    compact_finite = np.isfinite(compact_log_z_np)
                    both_finite = dense_finite & compact_finite
                    if np.any(both_finite):
                        compact_pair_check_max_abs_diff = max(
                            compact_pair_check_max_abs_diff,
                            float(
                                np.max(
                                    np.abs(
                                        dense_log_z_np[both_finite]
                                        - compact_log_z_np[both_finite]
                                    )
                                )
                            ),
                        )
                    compact_pair_check_finite_mismatches += int(
                        np.count_nonzero(dense_finite != compact_finite)
                    )
                    compact_pair_check_rows += int(dense_log_z_np.size)
            if any(rows is not None for rows in raw_diff2_dump_by_class):
                relion_min_diff2_dump = np.asarray(
                    global_min_diff2,
                    dtype=raw_host_staging_dtype,
                )
            del raw_diff2_by_class
            if bucket_raw_host_staging_bytes != 0:
                raise RuntimeError(
                    "fused K-class raw diff2 host staging accounting did not return to zero: "
                    f"{bucket_raw_host_staging_bytes} bytes"
                )
        _add_sparse_group_timing(group_timing, "score", time.time() - stage_t0)

        log_score_offset = (
            np.asarray(
                _relion_cuda_fine_log_evidence_offset(global_min_diff2),
                dtype=np.float64,
            )
            if use_exact_relion_gaussian
            else -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        )
        if normalization_log_evidence_np is None:
            global_score_log_z_bucket = _logsumexp_class_log_z(
                jnp.stack(class_score_log_z_bucket, axis=0)
            )
        else:
            # RELION's oversampling-zero symbolic second pass reuses the
            # coarse pass sum_weight. Convert its absolute log evidence into
            # this pass's common-min-centered score frame.
            global_score_log_z_bucket = jnp.asarray(
                normalization_log_evidence_np[image_indices] - log_score_offset,
                dtype=jnp.float64,
            )
        joint_mstep_masks_by_class = None
        joint_mstep_probs_by_class = None
        joint_full_probs_by_class = None
        if relion_fine_mstep_joint:
            flat_joint_probs_by_class = []
            flat_joint_scores_by_class = []
            joint_prob_shapes = []
            for class_index, arrays in enumerate(class_bucket_arrays):
                if bucket_uses_compact_pairs:
                    pair_arrays = compact_pair_arrays_by_class[class_index]
                    pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                    if use_relion_f32_fine_posterior and not winner_take_all:
                        # The native RELION fine-posterior path rebuilds both
                        # full and pruned probabilities directly from the
                        # concatenated float32 scores below.  Computing the
                        # generic float64 exp/log-Z probabilities here only
                        # to discard them caused one shape-specific XLA
                        # compilation per class and bucket group.
                        pair_probs = None
                    else:
                        (
                            _log_Z,
                            pair_probs,
                            best_log_score_bucket,
                            best_argmax,
                            _max_posterior_bucket,
                        ) = _normalize_pass2_pairs_with_log_z(
                            scores_by_class[class_index],
                            pair_mask,
                            global_score_log_z_bucket,
                        )
                    if winner_take_all:
                        pair_probs = _winner_take_all_pair_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    if pair_probs is not None:
                        pair_probs = jnp.where(pair_mask, pair_probs, 0.0)
                        flat_joint_probs_by_class.append(pair_probs.reshape(batch, -1))
                    flat_joint_scores_by_class.append(
                        jnp.where(pair_mask, scores_by_class[class_index], -jnp.inf).reshape(batch, -1)
                    )
                    joint_prob_shapes.append(scores_by_class[class_index].shape)
                else:
                    if use_relion_f32_fine_posterior and not winner_take_all:
                        probs = None
                    else:
                        (
                            _log_Z,
                            probs,
                            best_log_score_bucket,
                            best_argmax,
                            _max_posterior_bucket,
                        ) = _normalize_pass2_bucket_with_log_z(
                            scores_by_class[class_index],
                            global_score_log_z_bucket,
                        )
                    if winner_take_all:
                        probs = _winner_take_all_bucket_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    if probs is not None:
                        flat_joint_probs_by_class.append(probs.reshape(batch, -1))
                    flat_joint_scores_by_class.append(scores_by_class[class_index].reshape(batch, -1))
                    joint_prob_shapes.append(scores_by_class[class_index].shape)
            if winner_take_all:
                flat_joint_masks = _relion_joint_winner_take_all_masks(flat_joint_scores_by_class)
            elif use_relion_f32_fine_posterior:
                flat_sizes = [int(scores.shape[1]) for scores in flat_joint_scores_by_class]
                joint_scores = jnp.concatenate(flat_joint_scores_by_class, axis=1)
                (
                    joint_full_probs,
                    joint_reconstruction_probs,
                    joint_mask,
                    *_diagnostics,
                ) = (
                    _relion_f32_fine_posterior(
                        joint_scores,
                        adaptive_fraction=float(adaptive_fraction),
                        normalization_sum_weight=(
                            None
                            if relion_f32_normalization_sum_weight_np is None
                            else jnp.asarray(
                                relion_f32_normalization_sum_weight_np[image_indices],
                                dtype=jnp.float32,
                            )
                        ),
                        keep_all=relion_fine_mstep_prune_mode == "joint_keep_all",
                    )
                )
                split_points = np.cumsum(flat_sizes[:-1], dtype=np.int64).tolist()
                flat_joint_masks = list(jnp.split(joint_mask, split_points, axis=1))
                flat_joint_full_probs = list(
                    jnp.split(joint_full_probs, split_points, axis=1)
                )
                flat_joint_reconstruction_probs = list(
                    jnp.split(joint_reconstruction_probs, split_points, axis=1)
                )
                joint_full_probs_by_class = [
                    flat_probs.reshape(shape)
                    for flat_probs, shape in zip(
                        flat_joint_full_probs,
                        joint_prob_shapes,
                        strict=True,
                    )
                ]
                joint_mstep_probs_by_class = [
                    flat_probs.reshape(shape)
                    for flat_probs, shape in zip(
                        flat_joint_reconstruction_probs,
                        joint_prob_shapes,
                        strict=True,
                    )
                ]
            else:
                flat_joint_masks = _relion_pass2_reconstruction_joint_masks(
                    flat_joint_probs_by_class,
                    adaptive_fraction=float(adaptive_fraction),
                )
            joint_mstep_masks_by_class = [
                flat_mask.reshape(shape)
                for flat_mask, shape in zip(flat_joint_masks, joint_prob_shapes, strict=True)
            ]
        if dump_pass2_operands and parse_env_flag(_PASS2_DUMP_STOP_AFTER_TARGET_ENV, default=False):
            bucket_dump_count = 0
            for class_index, arrays in enumerate(class_bucket_arrays):
                if bucket_uses_compact_pairs:
                    pair_arrays = compact_pair_arrays_by_class[class_index]
                    pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                    if joint_full_probs_by_class is None:
                        _log_Z, pair_probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                            _normalize_pass2_pairs_with_log_z(
                                scores_by_class[class_index],
                                pair_mask,
                                global_score_log_z_bucket,
                            )
                        )
                    else:
                        pair_probs = joint_full_probs_by_class[class_index]
                        (
                            _log_Z,
                            best_log_score_bucket,
                            best_argmax,
                            _max_posterior_bucket,
                        ) = _diagnostics_from_normalized_pass2_probs(
                            scores_by_class[class_index],
                            pair_probs,
                            global_score_log_z_bucket,
                        )
                    if winner_take_all:
                        pair_probs = _winner_take_all_pair_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    dump_reconstruction_mask = (
                        None
                        if joint_mstep_masks_by_class is None
                        else joint_mstep_masks_by_class[class_index]
                    )
                    dump_reconstruction_probs = (
                        pair_probs
                        if dump_reconstruction_mask is None
                        else jnp.where(dump_reconstruction_mask, pair_probs, 0.0)
                    )
                    bucket_dump_count += pass2_diagnostics._maybe_dump_k_class_pass2_bucket(
                        experiment_dataset=experiment_dataset,
                        image_indices=image_indices,
                        class_index=class_index,
                        per_image_inputs=per_image_inputs_by_class[class_index],
                        class_bucket_arrays=arrays,
                        compact_pair_arrays=pair_arrays,
                        current_size=current_size,
                        n_fine_trans=n_fine_trans,
                        fine_translations=fine_translations,
                        fine_translation_parent=fine_translation_parent,
                        scores=scores_by_class[class_index],
                        probs=pair_probs,
                        bucket_translation_prior=bucket_translation_prior,
                        compact_pairs=True,
                        reconstruction_mask=dump_reconstruction_mask,
                        reconstruction_probs=dump_reconstruction_probs,
                        raw_diff2_by_batch_row=raw_diff2_dump_by_class[
                            class_index
                        ],
                        raw_operands_by_batch_row=raw_operand_dump_by_class[
                            class_index
                        ],
                        relion_min_diff2=relion_min_diff2_dump,
                    )
                else:
                    if joint_full_probs_by_class is None:
                        _log_Z, probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                            _normalize_pass2_bucket_with_log_z(
                                scores_by_class[class_index],
                                global_score_log_z_bucket,
                            )
                        )
                    else:
                        probs = joint_full_probs_by_class[class_index]
                        (
                            _log_Z,
                            best_log_score_bucket,
                            best_argmax,
                            _max_posterior_bucket,
                        ) = _diagnostics_from_normalized_pass2_probs(
                            scores_by_class[class_index],
                            probs,
                            global_score_log_z_bucket,
                        )
                    if winner_take_all:
                        probs = _winner_take_all_bucket_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    dump_reconstruction_mask = (
                        None
                        if joint_mstep_masks_by_class is None
                        else joint_mstep_masks_by_class[class_index]
                    )
                    dump_reconstruction_probs = (
                        probs
                        if dump_reconstruction_mask is None
                        else jnp.where(dump_reconstruction_mask, probs, 0.0)
                    )
                    bucket_dump_count += pass2_diagnostics._maybe_dump_k_class_pass2_bucket(
                        experiment_dataset=experiment_dataset,
                        image_indices=image_indices,
                        class_index=class_index,
                        per_image_inputs=per_image_inputs_by_class[class_index],
                        class_bucket_arrays=arrays,
                        compact_pair_arrays=None,
                        current_size=current_size,
                        n_fine_trans=n_fine_trans,
                        fine_translations=fine_translations,
                        fine_translation_parent=fine_translation_parent,
                        scores=scores_by_class[class_index],
                        probs=probs,
                        bucket_translation_prior=bucket_translation_prior,
                        compact_pairs=False,
                        reconstruction_mask=dump_reconstruction_mask,
                        reconstruction_probs=dump_reconstruction_probs,
                        raw_diff2_by_batch_row=raw_diff2_dump_by_class[
                            class_index
                        ],
                        raw_operands_by_batch_row=raw_operand_dump_by_class[
                            class_index
                        ],
                        relion_min_diff2=relion_min_diff2_dump,
                    )
            if bucket_dump_count:
                target_original_indices = parse_env_int_set(
                    "RECOVAR_PASS2_DUMP_ORIGINAL_INDICES"
                )
                if not target_original_indices:
                    target_original_indices = parse_env_int_set(
                        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
                    )
                target_class = os.environ.get("RECOVAR_PASS2_DUMP_CLASS")
                target_classes_one_based = (
                    {int(target_class)}
                    if target_class
                    else range(1, len(class_bucket_arrays) + 1)
                )
                completed_dump_count, expected_dump_count = _k_class_pass2_dump_progress(
                    dump_dir=os.environ[pass2_diagnostics._PASS2_DUMP_DIR_ENV],
                    target_original_indices=target_original_indices,
                    target_classes_one_based=target_classes_one_based,
                    current_size=current_size,
                )
                logger.info(
                    "Sparse fused K-class pass-2 stop-after-dump requested via %s=1; "
                    "target-set progress %d/%d file(s) at current_size=%s",
                    _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
                    int(completed_dump_count),
                    int(expected_dump_count),
                    "None" if current_size is None else str(int(current_size)),
                )
                if completed_dump_count == expected_dump_count:
                    raise Pass2DumpComplete(
                        dump_count=completed_dump_count,
                        current_size=current_size,
                    )
        shifted_recon_split = shifted_recon.reshape(batch, n_fine_trans, -1)
        if accumulate_noise:
            shifted_noise_split = (
                shifted_noise.reshape(batch, n_fine_trans, -1)
                if half_spectrum_scoring
                else shifted_score.reshape(batch, n_fine_trans, -1)
            )

        stage_t0 = time.time()
        for class_index, arrays in enumerate(class_bucket_arrays):
            substage_t0 = time.time()
            class_bucket_size = int(arrays["bucket_size"])
            active_rows_precomputed = False
            active_flat_rows_chunked = False
            flat_summed = None
            flat_ctf_probs = None
            active_flat_rotations = None
            mstep_active_indices = None
            mstep_active_mask = None
            mstep_active_count = 0
            summed_masked_noise_precomputed = None
            block_noise_shells_precomputed = None
            block_norm_residual_precomputed = None
            if bucket_uses_compact_pairs:
                pair_arrays = compact_pair_arrays_by_class[class_index]
                pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                if joint_full_probs_by_class is None:
                    log_Z, pair_probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                        _normalize_pass2_pairs_with_log_z(
                            scores_by_class[class_index],
                            pair_mask,
                            global_score_log_z_bucket,
                        )
                    )
                else:
                    pair_probs = joint_full_probs_by_class[class_index]
                    (
                        log_Z,
                        best_log_score_bucket,
                        best_argmax,
                        max_posterior_bucket,
                    ) = _diagnostics_from_normalized_pass2_probs(
                        scores_by_class[class_index],
                        pair_probs,
                        global_score_log_z_bucket,
                    )
                if winner_take_all:
                    pair_probs = _winner_take_all_pair_probs(
                        scores_by_class[class_index],
                        best_argmax,
                        best_log_score_bucket,
                    )
                    max_posterior_bucket = jnp.where(
                        jnp.isfinite(best_log_score_bucket),
                        jnp.ones_like(max_posterior_bucket),
                        jnp.zeros_like(max_posterior_bucket),
                    )
                if relion_fine_mstep_joint:
                    reconstruction_mask = joint_mstep_masks_by_class[class_index]
                    reconstruction_probs = (
                        joint_mstep_probs_by_class[class_index]
                        if joint_mstep_probs_by_class is not None
                        else jnp.where(reconstruction_mask, pair_probs, 0.0)
                    )
                    mstep_probs = reconstruction_probs
                elif relion_fine_mstep_prune:
                    reconstruction_probs, reconstruction_mask, _reconstruction_n_significant = (
                        _relion_pass2_reconstruction_pair_probs(
                            pair_probs,
                            pair_mask,
                            adaptive_fraction=float(adaptive_fraction),
                        )
                    )
                    mstep_probs = reconstruction_probs
                else:
                    reconstruction_probs = None
                    mstep_probs = pair_probs
                pass2_diagnostics._maybe_dump_k_class_pass2_bucket(
                    experiment_dataset=experiment_dataset,
                    image_indices=image_indices,
                    class_index=class_index,
                    per_image_inputs=per_image_inputs_by_class[class_index],
                    class_bucket_arrays=arrays,
                    compact_pair_arrays=pair_arrays,
                    current_size=current_size,
                    n_fine_trans=n_fine_trans,
                    fine_translations=fine_translations,
                    fine_translation_parent=fine_translation_parent,
                    scores=scores_by_class[class_index],
                    probs=pair_probs,
                    bucket_translation_prior=bucket_translation_prior,
                    compact_pairs=True,
                    reconstruction_mask=reconstruction_mask if relion_fine_mstep_prune else None,
                    reconstruction_probs=reconstruction_probs,
                    raw_diff2_by_batch_row=raw_diff2_dump_by_class[
                        class_index
                    ],
                    raw_operands_by_batch_row=raw_operand_dump_by_class[
                        class_index
                    ],
                    relion_min_diff2=relion_min_diff2_dump,
                )
                if (
                    accumulate_noise
                    and (reuse_compact_noise_sums or native_dual_weighted_sums)
                    and not compact_noise_sums_match_mstep
                ):
                    compact_pair_noise_image_sum_precomputes += 1
                    if fused_mstep_noise:
                        (
                            summed,
                            summed_masked_noise_precomputed,
                            ctf_probs,
                            probs_sum_t_jax,
                            translation_posterior_jax,
                            block_noise_shells_precomputed,
                            block_norm_residual_precomputed,
                        ) = _compact_pair_weighted_sums_and_noise_native(
                            mstep_probs,
                            jnp.asarray(pair_arrays["local_rotation_row"]),
                            jnp.asarray(pair_arrays["translation_idx"]),
                            pair_mask,
                            shifted_recon_split,
                            shifted_noise_split,
                            ctf2_over_nv_recon,
                            proj_for_noise_by_class[class_index],
                            proj_abs2_by_class[class_index],
                            noise_variance_for_noise,
                            shell_indices_noise,
                            n_rotation_rows=class_bucket_size,
                            shell_count=n_shells,
                            batch_size=batch,
                        )
                    else:
                        (
                            summed,
                            summed_masked_noise_precomputed,
                            ctf_probs,
                            probs_sum_t_jax,
                            translation_posterior_jax,
                        ) = (
                            _compact_pair_weighted_rotation_and_image_sums_native(
                                mstep_probs,
                                jnp.asarray(pair_arrays["local_rotation_row"]),
                                jnp.asarray(pair_arrays["translation_idx"]),
                                pair_mask,
                                shifted_recon_split,
                                shifted_noise_split,
                                ctf2_over_nv_recon,
                                n_rotation_rows=class_bucket_size,
                            )
                            if native_dual_weighted_sums
                            else _compact_pair_weighted_rotation_and_image_sums(
                                mstep_probs,
                                jnp.asarray(pair_arrays["local_rotation_row"]),
                                jnp.asarray(pair_arrays["translation_idx"]),
                                pair_mask,
                                shifted_recon_split,
                                shifted_noise_split,
                                ctf2_over_nv_recon,
                                n_rotation_rows=class_bucket_size,
                                allow_pair_sparse=compact_pair_pair_sparse_effective,
                            )
                        )
                else:
                    summed, ctf_probs, probs_sum_t_jax, translation_posterior_jax = (
                        _compact_pair_weighted_rotation_sums(
                            mstep_probs,
                            jnp.asarray(pair_arrays["local_rotation_row"]),
                            jnp.asarray(pair_arrays["translation_idx"]),
                            pair_mask,
                            shifted_recon_split,
                            ctf2_over_nv_recon,
                            n_rotation_rows=class_bucket_size,
                            allow_pair_sparse=compact_pair_pair_sparse_effective,
                            relion_x_half=use_relion_x_half_mstep,
                        )
                    )
            else:
                if joint_full_probs_by_class is None:
                    log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                        _normalize_pass2_bucket_with_log_z(
                            scores_by_class[class_index],
                            global_score_log_z_bucket,
                        )
                    )
                else:
                    probs = joint_full_probs_by_class[class_index]
                    (
                        log_Z,
                        best_log_score_bucket,
                        best_argmax,
                        max_posterior_bucket,
                    ) = _diagnostics_from_normalized_pass2_probs(
                        scores_by_class[class_index],
                        probs,
                        global_score_log_z_bucket,
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
                if relion_fine_mstep_joint:
                    reconstruction_mask = joint_mstep_masks_by_class[class_index]
                    reconstruction_probs = (
                        joint_mstep_probs_by_class[class_index]
                        if joint_mstep_probs_by_class is not None
                        else jnp.where(reconstruction_mask, probs, 0.0)
                    )
                    mstep_probs = reconstruction_probs
                elif relion_fine_mstep_prune:
                    reconstruction_probs, reconstruction_mask, _reconstruction_n_significant = (
                        _relion_pass2_reconstruction_probs(
                            probs,
                            adaptive_fraction=float(adaptive_fraction),
                        )
                    )
                    mstep_probs = reconstruction_probs
                else:
                    reconstruction_probs = None
                    mstep_probs = probs
                pass2_diagnostics._maybe_dump_k_class_pass2_bucket(
                    experiment_dataset=experiment_dataset,
                    image_indices=image_indices,
                    class_index=class_index,
                    per_image_inputs=per_image_inputs_by_class[class_index],
                    class_bucket_arrays=arrays,
                    compact_pair_arrays=None,
                    current_size=current_size,
                    n_fine_trans=n_fine_trans,
                    fine_translations=fine_translations,
                    fine_translation_parent=fine_translation_parent,
                    scores=scores_by_class[class_index],
                    probs=probs,
                    bucket_translation_prior=bucket_translation_prior,
                    compact_pairs=False,
                    reconstruction_mask=reconstruction_mask if relion_fine_mstep_prune else None,
                    reconstruction_probs=reconstruction_probs,
                    raw_diff2_by_batch_row=raw_diff2_dump_by_class[
                        class_index
                    ],
                    raw_operands_by_batch_row=raw_operand_dump_by_class[
                        class_index
                    ],
                    relion_min_diff2=relion_min_diff2_dump,
                )
                probs_sum_t_jax = jnp.sum(mstep_probs, axis=-1)
                translation_posterior_jax = jnp.sum(mstep_probs, axis=1)
                if bucket_uses_active_rows and rectangular_active_prematmul:
                    rectangular_active_prematmul_attempts += 1
                    mstep_active_indices, mstep_active_mask, mstep_active_count = (
                        _active_flat_row_indices_from_probs_sum_t(
                            probs_sum_t_jax,
                            pad_multiple=active_row_pad_multiple,
                        )
                    )
                    (
                        prematmul_is_efficient,
                        _active_count,
                        _active_slots,
                        grouped_rows,
                        dense_rows,
                        _grouped_dense_ratio,
                    ) = _rectangular_active_prematmul_is_efficient(
                        mstep_active_indices,
                        mstep_active_mask,
                        n_images=batch,
                        n_rotation_rows=class_bucket_size,
                        max_grouped_dense_ratio=rectangular_active_prematmul_max_grouped_dense_ratio,
                    )
                    rectangular_active_prematmul_grouped_rows += int(grouped_rows)
                    rectangular_active_prematmul_dense_rows += int(dense_rows)
                    if prematmul_is_efficient:
                        rectangular_active_prematmul_used += 1
                        rectangular_mstep_active_rows += int(mstep_active_count)
                        rectangular_mstep_padded_active_rows += int(mstep_active_indices.size)
                        rectangular_mstep_rectangular_rows += int(batch * class_bucket_size)
                        flat_summed, flat_ctf_probs, active_flat_rotations = (
                            _rectangular_active_weighted_sums_or_none(
                                mstep_probs,
                                probs_sum_t_jax,
                                shifted_recon_split,
                                ctf2_over_nv_recon,
                                flat_backproject_rotations_by_class[class_index],
                                mstep_active_indices,
                                mstep_active_mask,
                            )
                        )
                        active_rows_precomputed = True
                    else:
                        rectangular_active_prematmul_skipped += 1
                        summed, ctf_probs = compute_local_mstep_sums(
                            mstep_probs,
                            shifted_recon_split,
                            ctf2_over_nv_recon,
                            relion_x_half=use_relion_x_half_mstep,
                            default_probs_sum_t=probs_sum_t_jax,
                        )
                else:
                    summed, ctf_probs = compute_local_mstep_sums(
                        mstep_probs,
                        shifted_recon_split,
                        ctf2_over_nv_recon,
                        relion_x_half=use_relion_x_half_mstep,
                        default_probs_sum_t=probs_sum_t_jax,
                    )
            if mstep_subtract_ctf_projection:
                summed = subtract_projected_reference_from_sparse_mstep_rotation_sums(
                    summed,
                    probs_sum_t_jax,
                    proj_for_noise_by_class[class_index],
                    ctf2_over_nv_recon,
                )
            if (
                bucket_device_signature_requested
                and bpref_diagnostics._bpref_contribution_class_enabled(class_index)
            ):
                capture = bpref_diagnostics._materialize_k_class_capture_rows(
                    image_indices=image_indices,
                    target_particle_rows=target_particle_rows,
                    per_image_inputs=per_image_inputs_by_class[class_index],
                    class_bucket_arrays=arrays,
                    compact_pair_arrays=(
                        compact_pair_arrays_by_class[class_index]
                        if bucket_uses_compact_pairs
                        else None
                    ),
                    scores=scores_by_class[class_index],
                    probs=(pair_probs if bucket_uses_compact_pairs else probs),
                    reconstruction_mask=(
                        reconstruction_mask if relion_fine_mstep_prune else None
                    ),
                    reconstruction_probs=(
                        reconstruction_probs if relion_fine_mstep_prune else None
                    ),
                    bucket_translation_prior=bucket_translation_prior,
                    n_fine_trans=n_fine_trans,
                )
                capture_rows_jax = jnp.asarray(capture["batch_rows"], dtype=jnp.int32)
                capture_shifted_recon = shifted_recon_split[capture_rows_jax]
                capture_ctf2_over_nv = ctf2_over_nv_recon[capture_rows_jax]
                ordinary_capture_summed, ordinary_capture_ctf = compute_local_mstep_sums(
                    jnp.asarray(capture["reconstruction_probs"]),
                    capture_shifted_recon,
                    capture_ctf2_over_nv,
                    relion_x_half=use_relion_x_half_mstep,
                    sequential_translation_reduction=False,
                )
                shadow_capture_summed, shadow_capture_ctf = compute_local_mstep_sums(
                    jnp.asarray(capture["reconstruction_probs"]),
                    capture_shifted_recon,
                    capture_ctf2_over_nv,
                    relion_x_half=use_relion_x_half_mstep,
                    sequential_translation_reduction=True,
                )
                shadow_reduction_agreement = bpref_diagnostics._require_bpref_reduction_shadow_agreement(
                    ordinary_capture_summed,
                    ordinary_capture_ctf,
                    shadow_capture_summed,
                    shadow_capture_ctf,
                )
                positive_rotation_rows = np.count_nonzero(
                    np.sum(np.asarray(capture["reconstruction_probs"]), axis=-1) > 0,
                    axis=1,
                )
                bpref_diagnostics._validate_bpref_positive_rotation_rows(
                    positive_rotation_rows,
                    np.arange(capture["image_indices"].size, dtype=np.int64),
                    device_signature_requested=True,
                    winner_take_all=winner_take_all,
                    posterior_partitioned_across_classes=True,
                )
                diagnostic_owners = bpref_diagnostics._bpref_diagnostic_ownership_indices(
                    capture["image_indices"],
                    np.arange(capture["image_indices"].size, dtype=np.int64),
                    device_signature_requested=True,
                )
                bpref_diagnostics._validate_bpref_diagnostic_ownership(
                    diagnostic_owners,
                    device_signature_requested=True,
                )
                rotation_log_prior = np.asarray(capture["rotation_log_prior"])
                translation_log_prior_capture = np.asarray(capture["translation_log_prior"])
                preprior_scores = (
                    np.asarray(capture["scores"])
                    - rotation_log_prior[:, :, None]
                    - translation_log_prior_capture[:, None, :]
                )
                selected_rows = capture["batch_rows"]

                def _capture_preprocess_value(name):
                    if not high_precision_operand_bundle:
                        return None
                    return np.asarray(contribution_preprocess_operands[name])[selected_rows]

                bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                    experiment_dataset=experiment_dataset,
                    image_indices=capture["image_indices"],
                    current_size=current_size,
                    summed=shadow_capture_summed,
                    ctf_probs=shadow_capture_ctf,
                    rotations=capture["rotations"],
                    actual_counts=capture["actual_counts"],
                    rotation_indices=capture["rotation_indices"],
                    fine_translations=fine_translations,
                    scores=capture["scores"],
                    preprior_scores=preprior_scores,
                    probs=capture["probs"],
                    rotation_log_prior=rotation_log_prior,
                    translation_log_prior=translation_log_prior_capture,
                    log_z=np.asarray(log_Z)[selected_rows],
                    best_log_score=np.asarray(best_log_score_bucket)[selected_rows],
                    reconstruction_probs=capture["reconstruction_probs"],
                    reconstruction_mask=capture["reconstruction_mask"],
                    reconstruction_sum_weight=np.sum(
                        np.asarray(capture["probs"]).reshape(capture["image_indices"].size, -1),
                        axis=1,
                    ),
                    reconstruction_threshold=np.zeros(capture["image_indices"].size, dtype=np.float64),
                    candidate_mask=capture["candidate_mask"],
                    high_precision_operand_bundle=high_precision_operand_bundle,
                    raw_batch_data=(
                        np.asarray(batch_data)[selected_rows]
                        if high_precision_operand_bundle
                        else None
                    ),
                    ctf_params=(
                        np.asarray(ctf_params)[selected_rows]
                        if high_precision_operand_bundle
                        else None
                    ),
                    noise_variance_half=(
                        noise_variance_half if high_precision_operand_bundle else None
                    ),
                    integer_pre_shifts=_capture_preprocess_value("integer_pre_shifts"),
                    batch_image_corrections=_capture_preprocess_value("batch_image_corrections"),
                    batch_scale_corrections=_capture_preprocess_value("batch_scale_corrections"),
                    relion_preprocess_normalization_factors=_capture_preprocess_value(
                        "relion_preprocess_normalization_factors"
                    ),
                    relion_cuda_preprocess=(
                        contribution_preprocess_operands["relion_cuda_preprocess"]
                        if high_precision_operand_bundle
                        else False
                    ),
                    score_with_masked_images=score_with_masked_images,
                    image_mask=(
                        contribution_preprocess_operands["image_mask"]
                        if high_precision_operand_bundle
                        else None
                    ),
                    image_mask_mode=(
                        contribution_preprocess_operands["image_mask_mode"]
                        if high_precision_operand_bundle
                        else "not-captured"
                    ),
                    voxel_size=experiment_dataset.voxel_size,
                    ctf_mode=getattr(getattr(config.ctf, "mode", "legacy"), "name", "legacy"),
                    ctf_dose_per_tilt=getattr(config.ctf, "dose_per_tilt", 0.0),
                    ctf_angle_per_tilt=getattr(config.ctf, "angle_per_tilt", 0.0),
                    disc_type=disc_type,
                    projection_padding_factor=projection_padding_factor,
                    reconstruction_padding_factor=reconstruction_padding_factor,
                    use_relion_x_half_mstep=use_relion_x_half_mstep,
                    winner_take_all=winner_take_all,
                    max_r=float(current_size // 2) if use_window else None,
                    window_indices=(
                        relion_x_half_recon_indices
                        if use_relion_x_half_mstep
                        else recon_window_indices
                    ),
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    shadow_only_mode=True,
                    shadow_score_bitwise_equal=True,
                    shadow_reduction_agreement=shadow_reduction_agreement,
                    device_signature_active=True,
                    class_index=class_index,
                )

            if active_rows_precomputed:
                pass
            elif bucket_uses_active_rows:
                if mstep_active_indices is None:
                    mstep_active_indices, mstep_active_mask, mstep_active_count = (
                        _active_flat_row_indices_from_probs_sum_t(
                            probs_sum_t_jax,
                            pad_multiple=active_row_pad_multiple,
                        )
                    )
                if bucket_uses_compact_pairs:
                    compact_mstep_active_rows += int(mstep_active_count)
                    compact_mstep_padded_active_rows += int(mstep_active_indices.size)
                    compact_mstep_rectangular_rows += int(batch * class_bucket_size)
                else:
                    rectangular_mstep_active_rows += int(mstep_active_count)
                    rectangular_mstep_padded_active_rows += int(mstep_active_indices.size)
                    rectangular_mstep_rectangular_rows += int(batch * class_bucket_size)
                active_flat_rows_chunked = int(mstep_active_indices.size) > _active_flat_gather_chunk_rows(
                    summed,
                    ctf_probs,
                    flat_backproject_rotations_by_class[class_index],
                    max_block_bytes=max_adjoint_block_bytes,
                )
                if not active_flat_rows_chunked:
                    flat_summed, active_flat_rotations = _select_active_flat_rows(
                        summed,
                        flat_backproject_rotations_by_class[class_index],
                        mstep_active_indices,
                        mstep_active_mask,
                    )
                    flat_ctf_probs = _select_active_flat_values(
                        ctf_probs,
                        mstep_active_indices,
                        mstep_active_mask,
                    )
            else:
                flat_summed = flatten_bucket_rows(summed)
                flat_ctf_probs = flatten_bucket_rows(ctf_probs)
                active_flat_rotations = flat_backproject_rotations_by_class[class_index]
            _add_sparse_group_timing(group_timing, "mstep_weighted_sums", time.time() - substage_t0)
            substage_t0 = time.time()
            mstep_window_indices = relion_x_half_recon_indices if use_relion_x_half_mstep else recon_window_indices
            live_per_particle_launches = bool(
                use_relion_x_half_mstep and use_per_particle_launches
            )
            if live_per_particle_launches:
                Ft_y_total[class_index], Ft_ctf_total[class_index] = (
                    _accumulate_relion_x_half_per_particle_launches(
                        jnp.asarray(summed, dtype=jnp.complex64),
                        jnp.asarray(ctf_probs, dtype=jnp.float32),
                        jnp.asarray(arrays["mstep_rotations"]),
                        arrays["actual_counts"],
                        Ft_y_total[class_index],
                        Ft_ctf_total[class_index],
                        window_indices=mstep_window_indices,
                        image_shape=image_shape,
                        volume_shape=recon_volume_shape,
                        disc_type="linear_interp",
                        half_volume=use_half_volume_mstep,
                        max_r=float(current_size // 2) if use_window else None,
                        winner_take_all=winner_take_all,
                        strict_particle_order=False,
                        log_label_prefix=f"kclass{class_index + 1}-particle-xhalf",
                    )
                )
            elif active_flat_rows_chunked:
                if use_window:
                    Ft_y_total[class_index], Ft_ctf_total[class_index] = (
                        _accumulate_active_flat_rows_adjoint_chunked(
                            summed,
                            ctf_probs,
                            flat_backproject_rotations_by_class[class_index],
                            mstep_active_indices,
                            mstep_active_mask,
                            Ft_y_total[class_index],
                            Ft_ctf_total[class_index],
                            window_indices=mstep_window_indices,
                            use_windowed_adjoint=True,
                            image_shape=image_shape,
                            volume_shape=recon_volume_shape,
                            disc_type="linear_interp",
                            half_image=True,
                            half_volume=use_half_volume_mstep,
                            max_r=float(current_size // 2),
                            relion_x_half=use_relion_x_half_mstep,
                            max_block_bytes=max_adjoint_block_bytes,
                            log_label_prefix=f"kclass{class_index + 1}-active-window",
                        )
                    )
                elif use_relion_x_half_mstep:
                    Ft_y_total[class_index], Ft_ctf_total[class_index] = (
                        _accumulate_active_flat_rows_adjoint_chunked(
                            summed,
                            ctf_probs,
                            flat_backproject_rotations_by_class[class_index],
                            mstep_active_indices,
                            mstep_active_mask,
                            Ft_y_total[class_index],
                            Ft_ctf_total[class_index],
                            window_indices=relion_x_half_recon_indices,
                            use_windowed_adjoint=True,
                            image_shape=image_shape,
                            volume_shape=recon_volume_shape,
                            disc_type="linear_interp",
                            half_image=True,
                            half_volume=use_half_volume_mstep,
                            max_r=None,
                            relion_x_half=True,
                            max_block_bytes=max_adjoint_block_bytes,
                            log_label_prefix=f"kclass{class_index + 1}-active-xhalf",
                        )
                    )
                else:
                    Ft_y_total[class_index], Ft_ctf_total[class_index] = (
                        _accumulate_active_flat_rows_adjoint_chunked(
                            summed,
                            ctf_probs,
                            flat_backproject_rotations_by_class[class_index],
                            mstep_active_indices,
                            mstep_active_mask,
                            Ft_y_total[class_index],
                            Ft_ctf_total[class_index],
                            use_windowed_adjoint=False,
                            image_shape=image_shape,
                            volume_shape=recon_volume_shape,
                            disc_type="linear_interp",
                            half_image=True,
                            half_volume=use_half_volume_mstep,
                            max_r=None,
                            relion_x_half=False,
                            max_block_bytes=max_adjoint_block_bytes,
                            log_label_prefix=f"kclass{class_index + 1}-active-half",
                        )
                    )
            elif use_window:
                Ft_y_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    active_flat_rotations,
                    Ft_y_total[class_index],
                    window_indices=mstep_window_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=float(current_size // 2),
                    relion_x_half=use_relion_x_half_mstep,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-y-window",
                )
                Ft_ctf_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    active_flat_rotations,
                    Ft_ctf_total[class_index],
                    window_indices=mstep_window_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=float(current_size // 2),
                    relion_x_half=use_relion_x_half_mstep,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-ctf-window",
                )
            elif use_relion_x_half_mstep:
                Ft_y_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    active_flat_rotations,
                    Ft_y_total[class_index],
                    window_indices=relion_x_half_recon_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=True,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-y-xhalf",
                )
                Ft_ctf_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    active_flat_rotations,
                    Ft_ctf_total[class_index],
                    window_indices=relion_x_half_recon_indices,
                    use_windowed_adjoint=True,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=True,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-ctf-xhalf",
                )
            else:
                Ft_y_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_summed,
                    active_flat_rotations,
                    Ft_y_total[class_index],
                    use_windowed_adjoint=False,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=False,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-y-half",
                )
                Ft_ctf_total[class_index] = _accumulate_adjoint_block_chunked(
                    flat_ctf_probs,
                    active_flat_rotations,
                    Ft_ctf_total[class_index],
                    use_windowed_adjoint=False,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    disc_type="linear_interp",
                    half_image=True,
                    half_volume=use_half_volume_mstep,
                    max_r=None,
                    relion_x_half=False,
                    max_block_bytes=max_adjoint_block_bytes,
                    log_label=f"kclass{class_index + 1}-ctf-half",
                )
            _add_sparse_group_timing(group_timing, "mstep_adjoint", time.time() - substage_t0)
            class_posterior_sums_mstep[class_index] += float(
                np.sum(np.asarray(probs_sum_t_jax, dtype=np.float64))
            )
            if (
                accumulate_noise
                and bucket_uses_compact_pairs
                and not reuse_compact_noise_sums
                and not native_dual_weighted_sums
                and not compact_noise_sums_match_mstep
            ):
                # The compact-pair noise path recomputes weighted image sums with
                # masked scoring data.  Release the M-step dense weighted-sum
                # buffers before launching that second matmul; otherwise large
                # one-image RELION buckets can transiently hold both copies.
                try:
                    Ft_y_total[class_index].block_until_ready()
                    Ft_ctf_total[class_index].block_until_ready()
                except AttributeError:
                    pass
                summed = None
                ctf_probs = None
                flat_summed = None
                flat_ctf_probs = None
            if accumulate_noise:
                substage_t0 = time.time()
                if bucket_uses_compact_pairs:
                    noise_probs = mstep_probs
                    translation_posterior = np.asarray(translation_posterior_jax, dtype=np.float64)
                    if compact_noise_sums_match_mstep:
                        summed_masked_noise = summed
                        ctf_probs_for_noise = ctf_probs
                        noise_probs_sum_t = probs_sum_t_jax
                        compact_pair_noise_sum_reuses += 1
                        compact_pair_noise_ctf_sum_reuses += 1
                    elif reuse_compact_noise_sums or native_dual_weighted_sums:
                        if summed_masked_noise_precomputed is None:
                            summed_masked_noise = _compact_pair_weighted_image_sums(
                                noise_probs,
                                jnp.asarray(pair_arrays["local_rotation_row"]),
                                jnp.asarray(pair_arrays["translation_idx"]),
                                pair_mask,
                                shifted_noise_split,
                                n_rotation_rows=class_bucket_size,
                                allow_pair_sparse=compact_pair_pair_sparse_effective,
                                relion_x_half=use_relion_x_half_mstep,
                            )
                        else:
                            summed_masked_noise = summed_masked_noise_precomputed
                        ctf_probs_for_noise = ctf_probs
                        noise_probs_sum_t = probs_sum_t_jax
                        compact_pair_noise_ctf_sum_reuses += 1
                    else:
                        summed_masked_noise, ctf_probs_for_noise, noise_probs_sum_t, _noise_translation_posterior = (
                            _compact_pair_weighted_rotation_sums(
                                noise_probs,
                                jnp.asarray(pair_arrays["local_rotation_row"]),
                                jnp.asarray(pair_arrays["translation_idx"]),
                                pair_mask,
                                shifted_noise_split,
                                ctf2_over_nv_recon,
                                n_rotation_rows=class_bucket_size,
                                allow_pair_sparse=compact_pair_pair_sparse_effective,
                            )
                        )
                else:
                    noise_probs = reconstruction_probs if relion_fine_mstep_prune else probs
                    translation_posterior = np.asarray(translation_posterior_jax, dtype=np.float64)
                    noise_probs_sum_t = probs_sum_t_jax
                    if bucket_uses_active_rows and active_rows_precomputed:
                        summed_masked_noise = _rectangular_active_weighted_image_sums_or_none(
                            noise_probs,
                            shifted_noise_split,
                            mstep_active_indices,
                            mstep_active_mask,
                        )
                        ctf_probs_for_noise = None
                    else:
                        summed_masked_noise = compute_local_weighted_sums(noise_probs, shifted_noise_split)
                        ctf_probs_for_noise = ctf_probs
                if translation_sqdist_ang is not None:
                    noise_sigma2_offset_total[class_index] += float(
                        np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                    )
                support_mass = jnp.sum(noise_probs_sum_t, axis=1)
                # RELION adds power_img outside the class loop, once per image.
                # Keep the shared high-shell term on class zero so downstream
                # summation of class-local statistics reproduces that ordering.
                weighted_img_shells, weighted_img_per_image = _weighted_image_power_shells_and_per_image(
                    processed_score_half_for_noise,
                    shell_indices_half,
                    support_mass,
                    shell_count=n_shells,
                    norm_unweighted_shell_cutoff=None if current_size is None else int(current_size // 2),
                    norm_unweighted_high_shell=relion_norm_high_shell,
                    include_unweighted_high_shell=class_index == 0,
                )
                support_mass_np = np.asarray(support_mass, dtype=np.float64)
                noise_img_power_total[class_index] += np.asarray(weighted_img_shells, dtype=np.float64)
                noise_norm_correction_total[class_index][image_indices] += np.asarray(
                    weighted_img_per_image,
                    dtype=np.float64,
                )
                noise_sumw_total[class_index] += float(np.sum(support_mass_np, dtype=np.float64))
                if noise_scale_correction_xa_total is not None:
                    if ctf_probs_for_noise is None:
                        scale_summed_masked = compute_local_weighted_sums(noise_probs, shifted_noise_split)
                        scale_ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
                            noise_probs_sum_t,
                            ctf2_over_nv_recon,
                        )
                    else:
                        scale_summed_masked = summed_masked_noise
                        scale_ctf_probs = ctf_probs_for_noise
                    scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                        proj_for_noise_by_class[class_index],
                        proj_abs2_by_class[class_index],
                        scale_summed_masked,
                        scale_ctf_probs,
                        noise_variance_for_noise,
                        bucket_scale_for_stats,
                        scale_correction_pixel_masks[class_index],
                    )
                    np.add.at(
                        noise_scale_correction_xa_total[class_index],
                        np.asarray(bucket_group_ids, dtype=np.int64),
                        np.asarray(scale_xa_per_image, dtype=np.float64),
                    )
                    np.add.at(
                        noise_scale_correction_aa_total[class_index],
                        np.asarray(bucket_group_ids, dtype=np.int64),
                        np.asarray(scale_aa_per_image, dtype=np.float64),
                    )
                if block_noise_shells_precomputed is not None:
                    flat_proj_for_noise = None
                    flat_proj_abs2_for_noise = None
                    flat_summed_masked_noise = None
                    flat_ctf_probs_for_noise = None
                elif bucket_uses_active_rows:
                    flat_image_indices = None
                    if mstep_active_indices is None:
                        noise_active_indices, noise_active_mask, noise_active_count = (
                            _active_flat_row_indices_from_probs_sum_t(
                                noise_probs_sum_t,
                                pad_multiple=active_row_pad_multiple,
                            )
                        )
                    else:
                        noise_active_indices = mstep_active_indices
                        noise_active_mask = mstep_active_mask
                        noise_active_count = mstep_active_count
                    if bucket_uses_compact_pairs:
                        compact_noise_active_rows += int(noise_active_count)
                        compact_noise_padded_active_rows += int(noise_active_indices.size)
                        compact_noise_rectangular_rows += int(batch * class_bucket_size)
                    else:
                        rectangular_noise_active_rows += int(noise_active_count)
                        rectangular_noise_padded_active_rows += int(noise_active_indices.size)
                        rectangular_noise_rectangular_rows += int(batch * class_bucket_size)
                    if bucket_uses_compact_pairs:
                        flat_proj_for_noise = None
                        flat_proj_abs2_for_noise = None
                        flat_summed_masked_noise = None
                        flat_ctf_probs_for_noise = None
                        if noise_active_indices.size != 0:
                            compact_pair_noise_fused_active_gathers += 1
                    elif active_rows_precomputed:
                        flat_summed_masked_noise = summed_masked_noise
                        flat_ctf_probs_for_noise = flat_ctf_probs
                        flat_proj_for_noise = _select_active_flat_values(
                            proj_for_noise_by_class[class_index],
                            noise_active_indices,
                            noise_active_mask,
                        )
                        flat_proj_abs2_for_noise = _select_active_flat_values(
                            proj_abs2_by_class[class_index],
                            noise_active_indices,
                            noise_active_mask,
                        )
                    else:
                        flat_proj_for_noise = _select_active_flat_values(
                            proj_for_noise_by_class[class_index],
                            noise_active_indices,
                            noise_active_mask,
                        )
                        flat_proj_abs2_for_noise = _select_active_flat_values(
                            proj_abs2_by_class[class_index],
                            noise_active_indices,
                            noise_active_mask,
                        )
                        if active_rows_precomputed:
                            flat_summed_masked_noise = summed_masked_noise
                            flat_ctf_probs_for_noise = flat_ctf_probs
                        else:
                            flat_summed_masked_noise = _select_active_flat_values(
                                summed_masked_noise,
                                noise_active_indices,
                                noise_active_mask,
                            )
                            flat_ctf_probs_for_noise = flat_ctf_probs
                else:
                    flat_proj_for_noise = flatten_bucket_rows(proj_for_noise_by_class[class_index])
                    flat_proj_abs2_for_noise = flatten_bucket_rows(proj_abs2_by_class[class_index])
                    flat_summed_masked_noise = flatten_bucket_rows(summed_masked_noise)
                    flat_ctf_probs_for_noise = flatten_bucket_rows(ctf_probs_for_noise)
                if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                    logger.info(
                        "RECOVAR_NOISE_DTYPE_DEBUG(fused): bucket_uses_active_rows=%s "
                        "bucket_uses_compact_pairs=%s fused_noise_norm=%s "
                        "proj_for_noise=%s proj_abs2=%s ctf_probs_for_noise=%s "
                        "noise_variance_for_noise=%s summed_masked_noise=%s",
                        bucket_uses_active_rows,
                        bucket_uses_compact_pairs,
                        fused_noise_norm,
                        proj_for_noise_by_class[class_index].dtype,
                        proj_abs2_by_class[class_index].dtype,
                        ctf_probs_for_noise.dtype,
                        noise_variance_for_noise.dtype,
                        summed_masked_noise.dtype,
                    )
                if block_noise_shells_precomputed is not None:
                    noise_wsum_total[class_index] += np.asarray(
                        block_noise_shells_precomputed,
                        dtype=np.float64,
                    )
                    noise_norm_correction_total[class_index][image_indices] += np.asarray(
                        block_norm_residual_precomputed,
                        dtype=np.float64,
                    )
                elif bucket_uses_active_rows and bucket_uses_compact_pairs:
                    block_noise_shells, block_norm_residual = _compute_active_noise_rows_chunked(
                        proj_for_noise_by_class[class_index],
                        proj_abs2_by_class[class_index],
                        summed_masked_noise,
                        ctf_probs_for_noise,
                        noise_active_indices,
                        noise_active_mask,
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_rotation_rows=class_bucket_size,
                        shell_count=n_shells,
                        batch_size=batch,
                        max_block_bytes=max_noise_block_bytes,
                    )
                    if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                        logger.info(
                            "RECOVAR_NOISE_DTYPE_DEBUG(fused): block_noise_shells=%s",
                            block_noise_shells.dtype,
                        )
                    noise_wsum_total[class_index] += np.asarray(block_noise_shells, dtype=np.float64)
                    noise_norm_correction_total[class_index][image_indices] += np.asarray(
                        block_norm_residual,
                        dtype=np.float64,
                    )
                elif flat_summed_masked_noise is not None:
                    if bucket_uses_active_rows:
                        if flat_image_indices is None:
                            flat_image_indices = _active_image_indices_for_rotation_rows(
                                noise_active_indices,
                                noise_active_mask,
                                class_bucket_size,
                            )
                        block_noise_shells, block_norm_residual = (
                            _compute_noise_block_and_norm_residual_chunked(
                                flat_proj_for_noise,
                                flat_proj_abs2_for_noise,
                                flat_summed_masked_noise,
                                flat_ctf_probs_for_noise,
                                noise_variance_for_noise,
                                shell_indices_noise,
                                flat_image_indices,
                                shell_count=n_shells,
                                batch_size=batch,
                                max_block_bytes=max_noise_block_bytes,
                            )
                        )
                    elif fused_noise_norm and not bucket_uses_compact_pairs:
                        flat_image_indices = jnp.broadcast_to(
                            jnp.arange(batch, dtype=jnp.int32)[:, None],
                            (batch, class_bucket_size),
                        ).reshape(-1)
                        block_noise_shells, block_norm_residual = (
                            _compute_noise_block_and_norm_residual_chunked(
                                flat_proj_for_noise,
                                flat_proj_abs2_for_noise,
                                flat_summed_masked_noise,
                                flat_ctf_probs_for_noise,
                                noise_variance_for_noise,
                                shell_indices_noise,
                                flat_image_indices,
                                shell_count=n_shells,
                                batch_size=batch,
                                max_block_bytes=max_noise_block_bytes,
                            )
                        )
                    else:
                        block_noise_shells, _, _ = _compute_noise_block_chunked(
                            flat_proj_for_noise,
                            flat_proj_abs2_for_noise,
                            flat_summed_masked_noise,
                            flat_ctf_probs_for_noise,
                            noise_variance_for_noise,
                            shell_indices_noise,
                            n_shells,
                            max_block_bytes=max_noise_block_bytes,
                        )
                        block_norm_residual = _compute_norm_residual_per_image(
                            proj_for_noise_by_class[class_index],
                            proj_abs2_by_class[class_index],
                            summed_masked_noise,
                            ctf_probs_for_noise,
                            noise_variance_for_noise,
                        )
                    if parse_env_flag("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                        logger.info(
                            "RECOVAR_NOISE_DTYPE_DEBUG(fused): block_noise_shells=%s",
                            block_noise_shells.dtype,
                        )
                    noise_wsum_total[class_index] += np.asarray(block_noise_shells, dtype=np.float64)
                    noise_norm_correction_total[class_index][image_indices] += np.asarray(
                        block_norm_residual,
                        dtype=np.float64,
                    )
                _add_sparse_group_timing(group_timing, "noise", time.time() - substage_t0)

            substage_t0 = time.time()
            actual_counts_arr = np.asarray(arrays["actual_counts"], dtype=np.int64)
            best_argmax_np = np.asarray(best_argmax, dtype=np.int64)
            best_log_score_np = np.asarray(best_log_score_bucket, dtype=np.float64)
            has_best_pose_np = np.isfinite(best_log_score_np)
            if bucket_uses_compact_pairs:
                safe_best_argmax_np = np.where(has_best_pose_np, best_argmax_np, 0)
                row_index_np = np.arange(batch, dtype=np.int64)
                pair_local_rotation_row = np.asarray(pair_arrays["local_rotation_row"], dtype=np.int32)
                pair_translation_idx = np.asarray(pair_arrays["translation_idx"], dtype=np.int32)
                pair_rotation_index = np.asarray(pair_arrays["rotation_index"], dtype=np.int64)
                best_rot_idx = np.where(
                    has_best_pose_np,
                    pair_local_rotation_row[row_index_np, safe_best_argmax_np],
                    0,
                ).astype(np.int64, copy=False)
                best_trans_idx = np.where(
                    has_best_pose_np,
                    pair_translation_idx[row_index_np, safe_best_argmax_np],
                    0,
                ).astype(np.int64, copy=False)
                best_fine_rot_idx = np.where(
                    has_best_pose_np,
                    pair_rotation_index[row_index_np, safe_best_argmax_np],
                    np.asarray(arrays["rotation_indices"], dtype=np.int64)[:, 0],
                ).astype(np.int64, copy=False)
            else:
                best_rot_idx = best_argmax_np // n_fine_trans
                best_trans_idx = best_argmax_np % n_fine_trans
                row_index_np = np.arange(batch, dtype=np.int64)
                best_fine_rot_idx = np.asarray(arrays["rotation_indices"], dtype=np.int64)[
                    row_index_np,
                    best_rot_idx,
                ]
            if np.any(best_rot_idx >= actual_counts_arr):
                bad = np.flatnonzero(best_rot_idx >= actual_counts_arr)
                raise RuntimeError(
                    "Fused sparse K-class pass-2: best rotation index points into padding for "
                    f"class {class_index + 1}, images {bad.tolist()}",
                )
            max_posterior_np = np.asarray(
                max_posterior_bucket,
                dtype=precision_policy.score_real_dtype,
            )
            class_log_z_np = np.asarray(class_score_log_z_bucket[class_index], dtype=np.float64)
            probs_sum_t = np.asarray(probs_sum_t_jax, dtype=np.float64)
            for row, image_idx in enumerate(image_indices.tolist()):
                r = int(best_rot_idx[row])
                t = int(best_trans_idx[row])
                fine_rot_idx = int(best_fine_rot_idx[row])
                class_hard_assignments[class_index, image_idx] = fine_rot_idx * n_fine_trans + t
                best_rotations[class_index][image_idx] = per_image_inputs_by_class[class_index]["oversampled_rots"][
                    image_idx
                ][r]
                if best_eulers is not None:
                    best_eulers[class_index][image_idx] = per_image_inputs_by_class[class_index]["source_eulers"][
                        image_idx
                    ][r]
                best_rotation_indices[class_index][image_idx] = fine_rot_idx
                if np.isfinite(class_log_z_np[row]):
                    class_log_evidence[class_index, image_idx] = float(class_log_z_np[row] + log_score_offset[row])
                    class_score_log_z[class_index, image_idx] = float(
                        class_log_z_np[row] + log_score_offset[row]
                        if use_exact_relion_gaussian
                        else class_log_z_np[row]
                    )
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
            _add_sparse_group_timing(group_timing, "stats", time.time() - substage_t0)
        _add_sparse_group_timing(group_timing, "mstep_noise_stats", time.time() - stage_t0)

    if last_bucket_size_logged is not None and group_t0 is not None:
        group_chunks, group_images = bucket_group_stats[last_bucket_size_logged]
        group_wall = time.time() - group_t0
        logger.info(
            "Sparse fused K-class pass-2 bucket group done: mode=%s %s=%d chunks=%d images=%d wall=%.1fs images/s=%.1f",
            last_bucket_size_logged[0],
            last_bucket_size_logged[1],
            last_bucket_size_logged[2],
            group_chunks,
            group_images,
            group_wall,
            group_images / max(group_wall, 1e-9),
        )
        _log_sparse_kclass_group_timing(
            last_bucket_size_logged,
            group_timing,
            wall_s=group_wall,
        )

    if compact_pair_inputs_by_class_for_check is not None:
        logger.info(
            "Sparse fused K-class compact-pair score check: rows=%d, finite_mismatches=%d, "
            "max_abs_log_z_diff=%.6g",
            compact_pair_check_rows,
            compact_pair_check_finite_mismatches,
            compact_pair_check_max_abs_diff,
        )

    em_wall = time.time() - overall_t0
    compact_slot_ratio = (
        float(compact_rotation_slots) / float(rectangular_rotation_slots)
        if rectangular_rotation_slots > 0
        else 1.0
    )
    logger.info(
        "Sparse fused K-class pass-2: %d images, %d classes, %d buckets, %.2fs E+M; "
        "median local rot=%d, mean local rot=%.1f, median valid candidates/image=%d, "
        "padded_rotation_slots=%d/%d (ratio=%.3f)",
        n_images,
        n_classes,
        len(buckets),
        em_wall,
        int(np.median(local_rot_counts)) if local_rot_counts else 0,
        float(np.mean(local_rot_counts)) if local_rot_counts else 0.0,
        int(np.median(valid_candidate_counts)) if valid_candidate_counts else 0,
        compact_rotation_slots,
        rectangular_rotation_slots,
        compact_slot_ratio,
    )
    if raw_host_staging_total_bytes:
        logger.info(
            "Sparse fused K-class raw diff2 host staging: transferred=%.3f GiB "
            "peak_per_bucket=%.3f GiB cap=%.3f GiB d2h=%.3fs",
            raw_host_staging_total_bytes / float(1024**3),
            raw_host_staging_peak_bytes / float(1024**3),
            raw_host_staging_max_bytes / float(1024**3),
            raw_host_staging_s,
        )
    if compact_active_rows:
        mstep_active_ratio = (
            float(compact_mstep_active_rows) / float(compact_mstep_rectangular_rows)
            if compact_mstep_rectangular_rows > 0
            else 1.0
        )
        noise_active_ratio = (
            float(compact_noise_active_rows) / float(compact_noise_rectangular_rows)
            if compact_noise_rectangular_rows > 0
            else 1.0
        )
        logger.info(
            "Sparse fused K-class compact active rows: mstep=%d/%d (ratio=%.3f), "
            "noise=%d/%d (ratio=%.3f)",
            compact_mstep_active_rows,
            compact_mstep_rectangular_rows,
            mstep_active_ratio,
            compact_noise_active_rows,
            compact_noise_rectangular_rows,
            noise_active_ratio,
        )
        mstep_padded_active_ratio = (
            float(compact_mstep_padded_active_rows) / float(compact_mstep_rectangular_rows)
            if compact_mstep_rectangular_rows > 0
            else 1.0
        )
        noise_padded_active_ratio = (
            float(compact_noise_padded_active_rows) / float(compact_noise_rectangular_rows)
            if compact_noise_rectangular_rows > 0
            else 1.0
        )
        logger.info(
            "Sparse fused K-class compact active padded rows: mstep=%d/%d (ratio=%.3f), "
            "noise=%d/%d (ratio=%.3f)",
            compact_mstep_padded_active_rows,
            compact_mstep_rectangular_rows,
            mstep_padded_active_ratio,
            compact_noise_padded_active_rows,
            compact_noise_rectangular_rows,
            noise_padded_active_ratio,
        )
    if rectangular_active_rows:
        mstep_active_ratio = (
            float(rectangular_mstep_active_rows) / float(rectangular_mstep_rectangular_rows)
            if rectangular_mstep_rectangular_rows > 0
            else 1.0
        )
        noise_active_ratio = (
            float(rectangular_noise_active_rows) / float(rectangular_noise_rectangular_rows)
            if rectangular_noise_rectangular_rows > 0
            else 1.0
        )
        logger.info(
            "Sparse fused K-class rectangular active rows: min_bucket_size=%d, "
            "mstep=%d/%d (ratio=%.3f), noise=%d/%d (ratio=%.3f)",
            int(rectangular_active_rows_min_bucket_size),
            rectangular_mstep_active_rows,
            rectangular_mstep_rectangular_rows,
            mstep_active_ratio,
            rectangular_noise_active_rows,
            rectangular_noise_rectangular_rows,
            noise_active_ratio,
        )
        mstep_padded_active_ratio = (
            float(rectangular_mstep_padded_active_rows) / float(rectangular_mstep_rectangular_rows)
            if rectangular_mstep_rectangular_rows > 0
            else 1.0
        )
        noise_padded_active_ratio = (
            float(rectangular_noise_padded_active_rows) / float(rectangular_noise_rectangular_rows)
            if rectangular_noise_rectangular_rows > 0
            else 1.0
        )
        logger.info(
            "Sparse fused K-class rectangular active padded rows: min_bucket_size=%d, "
            "mstep=%d/%d (ratio=%.3f), noise=%d/%d (ratio=%.3f)",
            int(rectangular_active_rows_min_bucket_size),
            rectangular_mstep_padded_active_rows,
            rectangular_mstep_rectangular_rows,
            mstep_padded_active_ratio,
            rectangular_noise_padded_active_rows,
            rectangular_noise_rectangular_rows,
            noise_padded_active_ratio,
        )
        prematmul_grouped_dense_ratio = (
            float(rectangular_active_prematmul_grouped_rows) / float(rectangular_active_prematmul_dense_rows)
            if rectangular_active_prematmul_dense_rows > 0
            else 0.0
        )
        logger.info(
            "Sparse fused K-class rectangular active prematmul: enabled=%s, max_grouped_dense_ratio=%.3g, "
            "attempts=%d, used=%d, skipped=%d, grouped_rows=%d, dense_rows=%d, grouped_dense_ratio=%.3f",
            "1" if rectangular_active_prematmul else "0",
            float(rectangular_active_prematmul_max_grouped_dense_ratio),
            rectangular_active_prematmul_attempts,
            rectangular_active_prematmul_used,
            rectangular_active_prematmul_skipped,
            rectangular_active_prematmul_grouped_rows,
            rectangular_active_prematmul_dense_rows,
            prematmul_grouped_dense_ratio,
        )

    Ft_y_out = []
    Ft_ctf_out = []
    for class_index in range(n_classes):
        class_Ft_y = Ft_y_total[class_index]
        class_Ft_ctf = Ft_ctf_total[class_index]
        if use_half_volume_mstep:
            bpref_diagnostics._maybe_dump_native_half_mstep(
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
            bpref_diagnostics._maybe_dump_native_half_mstep(
                class_Ft_y,
                class_Ft_ctf,
                current_size=current_size,
                n_images=n_images,
                recon_volume_shape=recon_volume_shape,
                stage=f"fused_class{class_index + 1}_post_x0",
            )
            if use_relion_x_half_mstep:
                class_Ft_y, class_Ft_ctf = relion_x_half_accumulators_to_public_layout(
                    class_Ft_y,
                    class_Ft_ctf,
                    recon_volume_shape,
                )
            else:
                logger.info(
                    "Sparse fused K-class pass-2 class %d M-step: keeping native half-volume accumulators",
                    class_index + 1,
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
                wsum_norm_correction=noise_norm_correction_total[class_index],
                wsum_scale_correction_xa=None
                if noise_scale_correction_xa_total is None
                else noise_scale_correction_xa_total[class_index],
                wsum_scale_correction_aa=None
                if noise_scale_correction_aa_total is None
                else noise_scale_correction_aa_total[class_index],
            )
            for class_index in range(n_classes)
        )
    best_translations = tuple(
        fine_translations[class_hard_assignments[class_index] % n_fine_trans]
        for class_index in range(n_classes)
    )
    profile_summary = {
        "sparse_kclass_fused_s": np.float64(em_wall),
        "sparse_kclass_buckets": np.int64(len(buckets)),
        "sparse_kclass_max_hypotheses_per_microbatch": np.int64(max_hypotheses_per_microbatch),
        "sparse_kclass_max_images_per_microbatch": np.int64(max_images_per_microbatch),
        "sparse_kclass_max_translation_tile_bytes": np.int64(max_translation_tile_bytes),
        "sparse_kclass_translation_tile_half_pixels": np.int64(
            int(translation_tile_half_pixels) if translation_tile_half_pixels is not None else int(n_half),
        ),
        "sparse_kclass_windowed_translation_tile_cap": bool(_windowed_translation_tile_cap_enabled_for_pass()),
        "sparse_kclass_max_projection_gather_bytes": np.int64(max_projection_gather_bytes),
        "sparse_kclass_compact_pair_dense_mstep_max_bytes": np.int64(compact_pair_dense_mstep_max_bytes),
        "sparse_kclass_max_noise_block_bytes": np.int64(max_noise_block_bytes),
        "sparse_kclass_max_adjoint_block_bytes": np.int64(max_adjoint_block_bytes),
        "sparse_kclass_raw_host_staging_max_bytes": np.int64(raw_host_staging_max_bytes),
        "sparse_kclass_raw_host_staging_total_bytes": np.int64(raw_host_staging_total_bytes),
        "sparse_kclass_raw_host_staging_peak_bytes": np.int64(raw_host_staging_peak_bytes),
        "sparse_kclass_raw_host_staging_s": np.float64(raw_host_staging_s),
        "sparse_kclass_exact_relion_gaussian": bool(use_exact_relion_gaussian),
        "sparse_kclass_relion_fine_diff2_fused_ffi": bool(
            use_relion_fine_diff2_fused_ffi
        ),
        "sparse_kclass_relion_f32_fine_posterior": bool(
            use_relion_f32_fine_posterior
        ),
        "sparse_kclass_compact_pair_check_rows": np.int64(compact_pair_check_rows),
        "sparse_kclass_compact_pair_check_finite_mismatches": np.int64(
            compact_pair_check_finite_mismatches,
        ),
        "sparse_kclass_compact_pair_check_max_abs_log_z_diff": np.float64(
            compact_pair_check_max_abs_diff,
        ),
        "sparse_kclass_score_pixels": np.int64(int(budget_window_spec.n_score)),
        "sparse_kclass_device_memory_bytes": np.int64(-1 if device_memory_bytes is None else int(device_memory_bytes)),
        "sparse_kclass_windowed_prepare": bool(windowed_prepare),
        "sparse_kclass_fused_noise_norm": bool(fused_noise_norm),
        "sparse_kclass_relion_fine_mstep_prune": bool(relion_fine_mstep_prune),
        "sparse_kclass_relion_fine_mstep_prune_mode": relion_fine_mstep_prune_mode,
        "sparse_kclass_mstep_class_posterior_sums": class_posterior_sums_mstep.astype(np.float64, copy=True),
        "sparse_kclass_mstep_class_posterior_sum_total": np.float64(np.sum(class_posterior_sums_mstep)),
        "sparse_kclass_compact_pairs": bool(compact_pairs),
        "sparse_kclass_native_dual_weighted_sums": bool(native_dual_weighted_sums),
        "sparse_kclass_fused_mstep_noise": bool(fused_mstep_noise),
        "sparse_kclass_compact_pair_mstep_pair_sparse_requested": bool(
            compact_pair_pair_sparse_requested,
        ),
        "sparse_kclass_compact_pair_mstep_pair_sparse_effective": bool(
            compact_pair_pair_sparse_effective,
        ),
        "sparse_kclass_compact_pair_mstep_pair_sparse_xhalf_fallback": bool(
            compact_pair_pair_sparse_xhalf_fallback,
        ),
        "sparse_kclass_compact_pair_min_bucket_size": np.int64(
            0 if compact_pair_min_bucket_size is None else int(compact_pair_min_bucket_size),
        ),
        "sparse_kclass_compact_pairs_min_bucket_size": np.int64(
            0 if compact_pair_min_bucket_size is None else int(compact_pair_min_bucket_size),
        ),
        "sparse_kclass_compact_pair_tail_coalesce_max_images": np.int64(
            0 if compact_pair_tail_coalesce_max_images is None else int(compact_pair_tail_coalesce_max_images),
        ),
        "sparse_kclass_compact_pair_tail_coalesce_max_inflation": np.float64(
            0.0
            if compact_pair_tail_coalesce_max_inflation is None
            else float(compact_pair_tail_coalesce_max_inflation),
        ),
        "sparse_kclass_compact_pair_tail_coalesce_min_bucket_size": np.int64(
            0
            if compact_pair_tail_coalesce_min_bucket_size is None
            else int(compact_pair_tail_coalesce_min_bucket_size),
        ),
        "sparse_kclass_compact_pair_execution_buckets": np.int64(len(compact_pair_execution_buckets)),
        "sparse_kclass_compact_pair_execution_images": np.int64(
            sum(len(bucket["image_indices"]) for bucket in compact_pair_execution_buckets),
        ),
        "sparse_kclass_rectangular_execution_buckets": np.int64(len(rectangular_execution_buckets)),
        "sparse_kclass_rectangular_execution_images": np.int64(
            sum(len(bucket["image_indices"]) for bucket in rectangular_execution_buckets),
        ),
        "sparse_kclass_hybrid_compact_pair_buckets": np.int64(len(compact_pair_execution_buckets)),
        "sparse_kclass_hybrid_compact_pair_images": np.int64(
            sum(len(bucket["image_indices"]) for bucket in compact_pair_execution_buckets),
        ),
        "sparse_kclass_hybrid_rectangular_buckets": np.int64(len(rectangular_execution_buckets)),
        "sparse_kclass_hybrid_rectangular_images": np.int64(
            sum(len(bucket["image_indices"]) for bucket in rectangular_execution_buckets),
        ),
        "sparse_kclass_compact_active_rows": bool(compact_active_rows),
        "sparse_kclass_rectangular_active_rows": bool(rectangular_active_rows),
        "sparse_kclass_rectangular_active_prematmul": bool(rectangular_active_prematmul),
        "sparse_kclass_rectangular_active_prematmul_max_grouped_dense_ratio": np.float64(
            rectangular_active_prematmul_max_grouped_dense_ratio,
        ),
        "sparse_kclass_rectangular_active_prematmul_attempts": np.int64(
            rectangular_active_prematmul_attempts,
        ),
        "sparse_kclass_rectangular_active_prematmul_used": np.int64(rectangular_active_prematmul_used),
        "sparse_kclass_rectangular_active_prematmul_skipped": np.int64(
            rectangular_active_prematmul_skipped,
        ),
        "sparse_kclass_rectangular_active_prematmul_grouped_rows": np.int64(
            rectangular_active_prematmul_grouped_rows,
        ),
        "sparse_kclass_rectangular_active_prematmul_dense_rows": np.int64(
            rectangular_active_prematmul_dense_rows,
        ),
        "sparse_kclass_rectangular_active_prematmul_grouped_dense_ratio": np.float64(
            float(rectangular_active_prematmul_grouped_rows) / float(rectangular_active_prematmul_dense_rows)
            if rectangular_active_prematmul_dense_rows > 0
            else 0.0,
        ),
        "sparse_kclass_rectangular_active_rows_min_bucket_size": np.int64(
            int(rectangular_active_rows_min_bucket_size),
        ),
        "sparse_kclass_compact_buckets": bool(compact_buckets),
        "sparse_kclass_compact_rotation_slots": np.int64(compact_rotation_slots),
        "sparse_kclass_rectangular_rotation_slots": np.int64(rectangular_rotation_slots),
        "sparse_kclass_compact_slot_ratio": np.float64(compact_slot_ratio),
        "sparse_kclass_compact_mstep_active_rows": np.int64(compact_mstep_active_rows),
        "sparse_kclass_compact_mstep_padded_active_rows": np.int64(compact_mstep_padded_active_rows),
        "sparse_kclass_compact_mstep_rectangular_rows": np.int64(compact_mstep_rectangular_rows),
        "sparse_kclass_compact_mstep_active_ratio": np.float64(
            float(compact_mstep_active_rows) / float(compact_mstep_rectangular_rows)
            if compact_mstep_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_compact_mstep_padded_active_ratio": np.float64(
            float(compact_mstep_padded_active_rows) / float(compact_mstep_rectangular_rows)
            if compact_mstep_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_compact_noise_active_rows": np.int64(compact_noise_active_rows),
        "sparse_kclass_compact_noise_padded_active_rows": np.int64(compact_noise_padded_active_rows),
        "sparse_kclass_compact_noise_rectangular_rows": np.int64(compact_noise_rectangular_rows),
        "sparse_kclass_compact_noise_sum_reuses": np.int64(compact_pair_noise_sum_reuses),
        "sparse_kclass_compact_noise_ctf_sum_reuses": np.int64(compact_pair_noise_ctf_sum_reuses),
        "sparse_kclass_compact_noise_image_sum_precomputes": np.int64(
            compact_pair_noise_image_sum_precomputes,
        ),
        "sparse_kclass_compact_noise_fused_active_gathers": np.int64(compact_pair_noise_fused_active_gathers),
        "sparse_kclass_compact_noise_active_ratio": np.float64(
            float(compact_noise_active_rows) / float(compact_noise_rectangular_rows)
            if compact_noise_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_compact_noise_padded_active_ratio": np.float64(
            float(compact_noise_padded_active_rows) / float(compact_noise_rectangular_rows)
            if compact_noise_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_rectangular_mstep_active_rows": np.int64(rectangular_mstep_active_rows),
        "sparse_kclass_rectangular_mstep_padded_active_rows": np.int64(
            rectangular_mstep_padded_active_rows,
        ),
        "sparse_kclass_rectangular_mstep_rectangular_rows": np.int64(rectangular_mstep_rectangular_rows),
        "sparse_kclass_rectangular_mstep_active_ratio": np.float64(
            float(rectangular_mstep_active_rows) / float(rectangular_mstep_rectangular_rows)
            if rectangular_mstep_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_rectangular_mstep_padded_active_ratio": np.float64(
            float(rectangular_mstep_padded_active_rows) / float(rectangular_mstep_rectangular_rows)
            if rectangular_mstep_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_rectangular_noise_active_rows": np.int64(rectangular_noise_active_rows),
        "sparse_kclass_rectangular_noise_padded_active_rows": np.int64(
            rectangular_noise_padded_active_rows,
        ),
        "sparse_kclass_rectangular_noise_rectangular_rows": np.int64(rectangular_noise_rectangular_rows),
        "sparse_kclass_rectangular_noise_active_ratio": np.float64(
            float(rectangular_noise_active_rows) / float(rectangular_noise_rectangular_rows)
            if rectangular_noise_rectangular_rows > 0
            else 1.0,
        ),
        "sparse_kclass_rectangular_noise_padded_active_ratio": np.float64(
            float(rectangular_noise_padded_active_rows) / float(rectangular_noise_rectangular_rows)
            if rectangular_noise_rectangular_rows > 0
            else 1.0,
        ),
    }
    if compact_pair_plan_stats is not None:
        profile_summary.update(
            {
                "sparse_kclass_compact_pair_plan_s": np.float64(compact_plan_s),
                "sparse_kclass_compact_pair_buckets": np.int64(len(compact_pair_plan_stats.buckets)),
                "sparse_kclass_compact_pair_max_images_per_microbatch": np.int64(
                    compact_pair_plan_stats.max_images_per_microbatch,
                ),
                "sparse_kclass_valid_pair_candidates": np.int64(
                    compact_pair_plan_stats.valid_pair_candidates,
                ),
                "sparse_kclass_padded_pair_candidates": np.int64(
                    compact_pair_plan_stats.padded_pair_candidates,
                ),
                "sparse_kclass_rectangular_pair_candidates": np.int64(
                    compact_pair_plan_stats.rectangular_candidates,
                ),
                "sparse_kclass_valid_pair_reduction": np.float64(
                    compact_pair_plan_stats.reduction_factor,
                ),
                "sparse_kclass_padded_pair_reduction": np.float64(
                    compact_pair_plan_stats.padded_reduction_factor,
                ),
                "sparse_kclass_median_valid_pairs_per_image": np.int64(
                    compact_pair_plan_stats.median_valid_pairs_per_image,
                ),
                "sparse_kclass_mean_valid_pairs_per_image": np.float64(
                    compact_pair_plan_stats.mean_valid_pairs_per_image,
                ),
                "sparse_kclass_max_valid_pairs_per_image": np.int64(
                    compact_pair_plan_stats.max_valid_pairs_per_image,
                ),
            },
        )
    return SparseKClassPass2FusedResult(
        class_log_evidence=class_log_evidence,
        class_score_log_z=class_score_log_z,
        Ft_y=tuple(Ft_y_out),
        Ft_ctf=tuple(Ft_ctf_out),
        per_class_hard_assignments=class_hard_assignments,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        per_class_best_pose_eulers_deg=None if best_eulers is None else tuple(best_eulers),
        per_class_best_pose_rotations=tuple(best_rotations),
        per_class_best_pose_translations=best_translations,
        per_class_best_pose_rotation_ids=tuple(best_rotation_indices),
        profile_summary=profile_summary,
        class_posterior_sums=class_posterior_sums_mstep if relion_fine_mstep_prune else None,
    )
