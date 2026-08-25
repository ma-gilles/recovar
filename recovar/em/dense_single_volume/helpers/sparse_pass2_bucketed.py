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

import hashlib
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

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
)
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
)
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.compact_candidate_capture import (
    compact_capture_requested_for_original_indices,
    compact_capture_requested_particle_count,
    maybe_capture_k1_production_bucket,
    maybe_capture_k1_production_bucket_chunked,
    require_chunked_capture_capacity,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
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
    image_preprocess_backend,
    prepare_batch_preprocess_operands,
    process_half_image,
    resolve_image_mask_for_half_preprocess,
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
from recovar.em.dense_single_volume.helpers.significance import (
    ComplementSignificantSampleIndices,
)
from recovar.em.dense_single_volume.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
    relion_x_half_sequential_translation_reduction_enabled,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger(__name__)

_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH = 1_000_000
_RELION_WAVG_ATOMIC_SCALE_AA_ENV = "RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA"
_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV = (
    "RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL"
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
_DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION = 2.0
_DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE = 4096
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
_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP"
_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV = "RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE"
_RELION_X_HALF_F32_FINE_POSTERIOR_ENV = "RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR"
_RELION_FINE_DIFF2_FUSED_FFI_ENV = "RECOVAR_RELION_FINE_DIFF2_FUSED_FFI"
_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV = "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH"
_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS"
_RELION_POWERCLASS_SPECTRUM_NORM_ENV = "RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM"
_RELION_TRANSLATED_WAVG_NORM_ENV = "RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM"
_BPREF_CONTRIBUTION_DUMP_CLASS_ENV = "RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS"
_BPREF_CONTRIBUTION_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET"
)
_BPREF_MEMBERSHIP_DUMP_DIR_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR"
_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_ITERATION"
_BPREF_MEMBERSHIP_DUMP_HALF_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_HALF"
_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE"
_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE"
_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV = (
    "RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE"
)
_PASS2_DUMP_DIR_ENV = "RECOVAR_PASS2_DUMP_DIR"
_PASS2_DUMP_CONSERVATIVE_EXECUTION_ENV = "RECOVAR_PASS2_DUMP_CONSERVATIVE_EXECUTION"
_PASS2_DUMP_STOP_AFTER_TARGET_ENV = "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET"
_NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET"
)
_NORM_RESIDUAL_DUMP_ONLY_ENV = "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_ONLY"
_PASS2_DUMP_RAW_OPERANDS_ENV = "RECOVAR_PASS2_DUMP_RAW_OPERANDS"
_PASS2_DUMP_ROTATION_ROWS_ENV = "RECOVAR_PASS2_DUMP_ROTATION_ROWS"
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

_native_mstep_dump_counter = 0
_bpref_contribution_dump_counter = 0
_bpref_contribution_call_counter = 0
_bpref_membership_dump_counter = 0
_bpref_contribution_context = {"iteration": -1, "half": -1}
_bpref_image_identity_cache: dict[str, np.ndarray] = {}
_BPrefPanelKey = tuple[int, int, str, int]
_bpref_device_panel_accumulators: dict[_BPrefPanelKey, tuple[jax.Array, jax.Array]] = {}
_bpref_device_panel_launch_counters: dict[_BPrefPanelKey, int] = {}
_bpref_device_panel_metadata: dict[_BPrefPanelKey, dict[str, object]] = {}


def set_bpref_contribution_dump_context(*, iteration: int, half: int) -> None:
    """Set explicit one-based iteration/half labels for diagnostic row dumps."""

    _bpref_contribution_context["iteration"] = int(iteration)
    _bpref_contribution_context["half"] = int(half)


def clear_bpref_contribution_dump_context() -> None:
    """Mark contribution and native M-step dumps as outside a numbered half."""

    _bpref_contribution_context["iteration"] = -1
    _bpref_contribution_context["half"] = -1
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


class BPrefContributionDumpComplete(RuntimeError):
    """Raised after an explicitly targeted BPref diagnostic bundle is written."""

    def __init__(
        self,
        *,
        contribution_path: str | Path,
        device_signature_path: str | Path | None,
    ):
        self.contribution_path = Path(contribution_path)
        self.device_signature_path = (
            None if device_signature_path is None else Path(device_signature_path)
        )
        message = (
            "requested RECOVAR BPref contribution target was written "
            f"(contribution_path={self.contribution_path}"
        )
        if self.device_signature_path is not None:
            message += f", device_signature_path={self.device_signature_path}"
        super().__init__(message + ")")


def _maybe_stop_after_bpref_contribution_dump(
    *,
    contribution_path: str | Path,
    device_signature_path: str | Path | None,
) -> None:
    """Stop an explicit diagnostic only after all requested files exist."""

    if os.environ.get(_BPREF_CONTRIBUTION_STOP_AFTER_TARGET_ENV) != "1":
        return
    contribution_path = Path(contribution_path)
    if not contribution_path.is_file():
        raise RuntimeError(
            "RECOVAR BPref contribution stop target is missing its contribution file: "
            f"{contribution_path}"
        )
    device_dump_requested = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    resolved_device_path = (
        None if device_signature_path is None else Path(device_signature_path)
    )
    if device_dump_requested and (
        resolved_device_path is None or not resolved_device_path.is_file()
    ):
        raise RuntimeError(
            "RECOVAR BPref contribution stop target is missing its requested "
            f"device-signature file: {resolved_device_path}"
        )
    raise BPrefContributionDumpComplete(
        contribution_path=contribution_path,
        device_signature_path=resolved_device_path,
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


def _original_indices_for_local(experiment_dataset, local_indices) -> np.ndarray:
    """Map local batch image indices to original image ids for debug dumps."""
    local_indices = np.asarray(local_indices, dtype=np.int64)
    mapper = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if mapper is not None:
        return np.asarray(mapper(local_indices), dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        return local_indices
    return np.asarray(original_indices_all, dtype=np.int64)[local_indices]


def _bpref_contribution_target_rows(experiment_dataset, image_indices) -> np.ndarray:
    """Return bucket rows selected by the optional frozen original-index target."""

    local_indices = np.asarray(image_indices, dtype=np.int64)
    target_raw = os.environ.get(
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES",
        "",
    ).strip()
    if not target_raw:
        return np.arange(local_indices.size, dtype=np.int64)
    targets = np.asarray(
        [int(value.strip()) for value in target_raw.split(",") if value.strip()],
        dtype=np.int64,
    )
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    return np.flatnonzero(np.isin(original_indices, targets)).astype(np.int64, copy=False)


def _bpref_diagnostic_ownership_indices(
    image_indices,
    target_particle_rows,
    *,
    device_signature_requested: bool,
) -> np.ndarray:
    """Return particle owners relevant to the requested diagnostic.

    A scoped device signature captures only the configured target rows.  The
    surrounding sparse bucket may be ordered by support size rather than by
    particle id, so requiring every unrelated bucket row to be monotone can
    abort an otherwise target-only observational capture.  Unscoped
    per-particle diagnostics retain the original full-bucket ordering gate.
    """

    owners = np.asarray(image_indices, dtype=np.int64)
    if not device_signature_requested:
        return owners
    rows = np.asarray(target_particle_rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= owners.size):
        raise RuntimeError("BPref device signature target row is outside the sparse bucket")
    return owners[rows]


def _validate_bpref_diagnostic_ownership(
    owners,
    *,
    device_signature_requested: bool,
) -> None:
    """Validate ownership without imposing particle-id order on scoped captures."""

    owners = np.asarray(owners, dtype=np.int64)
    if owners.size < 2:
        return
    if device_signature_requested:
        if np.unique(owners).size != owners.size:
            raise RuntimeError("Scoped BPref device signature requires unique particle ownership")
    elif not np.all(np.diff(owners) > 0):
        raise RuntimeError(
            "RELION per-particle launch diagnostic requires strictly increasing "
            "particle ownership order"
        )


def _resolve_bpref_bucket_diagnostic_modes(
    *,
    device_signature_requested: bool,
    contribution_diagnostics_active: bool,
    target_particle_rows,
    high_precision_operand_bundle_requested: bool,
) -> dict[str, bool]:
    """Limit scoped device diagnostics to buckets containing a target row."""

    target_bucket_active = bool(
        device_signature_requested and np.asarray(target_particle_rows).size
    )
    bucket_contribution_diagnostics_active = bool(
        contribution_diagnostics_active
        and (not device_signature_requested or target_bucket_active)
    )
    return {
        "device_signature_requested": target_bucket_active,
        "contribution_diagnostics_active": bucket_contribution_diagnostics_active,
        "shadow_only": target_bucket_active,
        "high_precision_operand_bundle": bool(
            bucket_contribution_diagnostics_active
            and high_precision_operand_bundle_requested
        ),
    }


def _bpref_contribution_class_enabled(class_index: int) -> bool:
    """Return whether a zero-based class belongs to the scoped capture.

    The environment value is one-based to match RELION's class numbering and
    the class labels used by the pre-scatter comparison scripts.
    """

    value = os.environ.get(_BPREF_CONTRIBUTION_DUMP_CLASS_ENV, "").strip()
    if not value:
        return True
    try:
        requested = int(value)
    except ValueError as exc:
        raise ValueError(f"{_BPREF_CONTRIBUTION_DUMP_CLASS_ENV} must be a positive integer") from exc
    if requested <= 0:
        raise ValueError(f"{_BPREF_CONTRIBUTION_DUMP_CLASS_ENV} must be a positive integer")
    return int(class_index) + 1 == requested


def _validate_bpref_positive_rotation_rows(
    positive_rotation_rows,
    target_particle_rows,
    *,
    device_signature_requested: bool,
    winner_take_all: bool,
    posterior_partitioned_across_classes: bool = False,
) -> None:
    """Validate positive-row support for owners represented by a diagnostic.

    A soft posterior is allowed to leave one positive rotation row for every
    particle after reconstruction pruning.  This check runs independently for
    each sparse bucket, so requiring a multi-row witness here would incorrectly
    reject a valid bucket even when other buckets contain soft multi-row
    particles.  A fused K-class capture is a slice of a jointly normalized
    posterior: a particle may therefore have zero rows in the requested class
    while retaining support in another class.
    """

    counts = np.asarray(positive_rotation_rows, dtype=np.int64)
    if device_signature_requested:
        rows = np.asarray(target_particle_rows, dtype=np.int64)
        if rows.size == 0:
            return
        if np.any(rows < 0) or np.any(rows >= counts.size):
            raise RuntimeError("BPref device signature target row is outside the sparse bucket")
        counts = counts[rows]
    if np.any(counts < 0):
        raise RuntimeError("BPref positive rotation-row count cannot be negative")
    if posterior_partitioned_across_classes:
        if winner_take_all and np.any(counts > 1):
            raise RuntimeError(
                "RELION K-class WTA diagnostic permits at most one positive "
                "rotation row per particle and class"
            )
        return
    if winner_take_all:
        if not np.all(counts == 1):
            raise RuntimeError(
                "RELION WTA per-particle diagnostic requires exactly one positive rotation row per particle"
            )
    elif np.any(counts < 1):
        raise RuntimeError(
            "RECOVAR soft-particle causal arm requires at least one positive row per particle"
        )


def _empty_bpref_device_signature_arrays(
    dense_pixel_count: int,
    *,
    image_identity_dtype,
) -> dict[str, np.ndarray]:
    """Return a schema-valid signature payload for an all-zero class slice."""

    pixels = int(dense_pixel_count)
    if pixels <= 0:
        raise ValueError("BPref device signature dense pixel count must be positive")
    return {
        "rotation_keys": np.empty((0, pixels), dtype=np.int32),
        "pixel_indices": np.empty((0, pixels), dtype=np.int32),
        "row_flags": np.empty((0, pixels), dtype=np.int32),
        "source_values": np.empty((0, pixels, 6), dtype=np.float32),
        "neighbor_indices": np.empty((0, pixels, 8), dtype=np.int32),
        "neighbor_coefficients": np.empty((0, pixels, 8), dtype=np.float32),
        "neighbor_flags": np.empty((0, pixels, 8), dtype=np.int32),
        "launch_ordinals": np.empty((0,), dtype=np.int64),
        "particle_local_rows": np.empty((0,), dtype=np.int32),
        "image_identities": np.empty((0,), dtype=np.dtype(image_identity_dtype)),
        "original_indices": np.empty((0,), dtype=np.int64),
        "contributor_rotation_keys": np.empty((0,), dtype=np.int32),
    }


def _guard_bpref_target_rotation_chunking(
    rotation_chunk_size,
    *,
    bucket_size: int,
    target_particle_rows,
):
    """Preserve live chunk planning and reject only a genuinely chunked target."""

    target_count = int(np.asarray(target_particle_rows).size)
    if (
        target_count
        and rotation_chunk_size is not None
        and int(rotation_chunk_size) < int(bucket_size)
    ):
        raise RuntimeError(
            "BPref device signature target bucket is rotation-chunked in the "
            "authoritative production plan; capture refuses to change that plan "
            f"(bucket_size={int(bucket_size)}, rotation_chunk_size={int(rotation_chunk_size)}, "
            f"target_particles={target_count})"
        )
    return rotation_chunk_size


def _bpref_image_identities_for_original_indices(original_indices: np.ndarray) -> np.ndarray:
    """Return exact ``rlnImageName`` identities for diagnostic particles.

    The explicit mapping is required for cross-engine diagnostics because a
    local dataset row or original integer index is not, by itself, a stable
    identity across STAR readers.  Object arrays are deliberately rejected so
    the diagnostic never needs pickle.
    """

    mapping_path = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY", "").strip()
    if not mapping_path:
        raise RuntimeError(
            "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY is required when "
            "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR is enabled"
        )
    resolved = str(Path(mapping_path).expanduser().resolve())
    identities = _bpref_image_identity_cache.get(resolved)
    if identities is None:
        identities = np.load(resolved, allow_pickle=False)
        if identities.ndim != 1 or identities.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "BPref image identity mapping must be a rank-1 fixed-width string NPY, "
                f"got shape={identities.shape} dtype={identities.dtype}"
            )
        identities = identities.astype(str, copy=False)
        _bpref_image_identity_cache[resolved] = identities
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if original_indices.size and (
        int(original_indices.min()) < 0 or int(original_indices.max()) >= identities.size
    ):
        raise IndexError(
            "BPref original particle index is outside the explicit image identity mapping: "
            f"range=[{int(original_indices.min())}, {int(original_indices.max())}] "
            f"mapping_size={identities.size}"
        )
    selected = identities[original_indices]
    if np.any(np.char.find(selected, "@") <= 0):
        raise ValueError("Every BPref image identity must be an exact 1-based-index@stack-path string")
    for identity in selected.tolist():
        _, stack_path = identity.split("@", 1)
        if not Path(stack_path).is_absolute():
            raise ValueError(f"BPref image identity stack path must be absolute, got {identity!r}")
    return selected


def _bpref_required_stack_checksum() -> str:
    checksum = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256", "").strip().lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise RuntimeError(
            "RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256 must contain the frozen source stack SHA256"
        )
    return checksum


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flush_bpref_device_panel_accumulator(*, iteration: int, half: int) -> None:
    """Write and release every exact native class panel for one half."""

    dump_dir = os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    run_id = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "unset")
    prefix = (int(iteration), int(half), run_id)
    keys = sorted(key for key in _bpref_device_panel_metadata if key[:3] == prefix)
    if not keys:
        raise RuntimeError(f"No RECOVAR device panel metadata exists for {prefix}")
    output = Path(dump_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        accumulators = _bpref_device_panel_accumulators.pop(key, None)
        launch_count = _bpref_device_panel_launch_counters.pop(key, 0)
        metadata = _bpref_device_panel_metadata.pop(key)
        if accumulators is None:
            raise RuntimeError(f"No RECOVAR device panel accumulator exists for {key}")
        data_accumulator, weight_accumulator = accumulators
        class_index = int(metadata["class_index"])
        np.savez(
            output
            / (
                f"recovar_device_panel_native_it{int(iteration):03d}_h{int(half)}"
                f"_class{class_index + 1:03d}_rank{int(metadata['rank']):03d}.npz"
            ),
            magic=np.asarray("RECOVAR_DEVICE_PANEL_NATIVE"),
            schema=np.asarray("recovar-device-panel-native-v1"),
            schema_version=np.int32(1),
            run_id=np.asarray(run_id),
            iteration=np.int32(iteration),
            half=np.int32(half),
            class_index=np.int32(class_index),
            rank=np.int32(metadata["rank"]),
            launch_count=np.int64(launch_count),
            current_size=np.int32(metadata["current_size"]),
            max_r=np.float32(metadata["max_r"]),
            image_shape=np.asarray(metadata["image_shape"], dtype=np.int32),
            volume_shape=np.asarray(metadata["volume_shape"], dtype=np.int32),
            reconstruction_padding_factor=np.int32(metadata["reconstruction_padding_factor"]),
            source_stack_sha256=np.asarray(metadata["source_stack_sha256"]),
            causal_arm=np.asarray(metadata["causal_arm"]),
            winner_take_all=np.bool_(metadata["winner_take_all"]),
            topology_claim=np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
            accumulator_field_legend=np.asarray("data=complex64 x-half;weight=float32 x-half;flat C order"),
            data_accumulator=np.asarray(data_accumulator),
            weight_accumulator=np.asarray(weight_accumulator),
        )


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


class SparseCandidateMask:
    """Compact host representation of one image's pass-2 candidate mask."""

    __slots__ = (
        "mode",
        "n_rows",
        "n_fine_trans",
        "parent_map",
        "coarse_valid",
        "coarse_excluded",
        "fine_translation_parent",
        "count",
    )

    def __init__(
        self,
        *,
        mode: str,
        n_rows: int,
        n_fine_trans: int,
        parent_map=None,
        coarse_valid=None,
        coarse_excluded=None,
        fine_translation_parent=None,
        count: int | None = None,
    ):
        self.mode = str(mode)
        self.n_rows = int(n_rows)
        self.n_fine_trans = int(n_fine_trans)
        self.parent_map = None if parent_map is None else np.asarray(parent_map, dtype=np.int32)
        self.coarse_valid = None if coarse_valid is None else np.asarray(coarse_valid, dtype=bool)
        self.coarse_excluded = None if coarse_excluded is None else np.asarray(coarse_excluded, dtype=np.int32)
        self.fine_translation_parent = (
            None if fine_translation_parent is None else np.asarray(fine_translation_parent, dtype=np.int32)
        )
        self.count = int(_dense_candidate_mask_from_spec(self).sum()) if count is None else int(count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_fine_trans)

    def __array__(self, dtype=None, copy=None):
        dense = _dense_candidate_mask_from_spec(self)
        if dtype is not None:
            return dense.astype(dtype, copy=False if copy is None else bool(copy))
        if copy:
            return dense.copy()
        return dense


def _dense_candidate_mask_from_spec(mask: SparseCandidateMask) -> np.ndarray:
    if mask.mode == "full":
        return np.ones(mask.shape, dtype=bool)
    if mask.mode == "empty":
        return np.zeros(mask.shape, dtype=bool)
    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing parent/coarse arrays")
        return mask.coarse_valid[:, mask.fine_translation_parent][mask.parent_map]
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        dense = np.ones(mask.shape, dtype=bool)
        excluded = np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1)
        if excluded.size:
            n_coarse_trans = int(mask.fine_translation_parent.max(initial=-1) + 1)
            if n_coarse_trans <= 0:
                raise ValueError("coarse_exclude candidate mask has empty translation parent map")
            excluded_rot = excluded // n_coarse_trans
            excluded_trans = excluded % n_coarse_trans
            for coarse_rot, coarse_trans in zip(excluded_rot.tolist(), excluded_trans.tolist(), strict=False):
                rows = np.flatnonzero(mask.parent_map == int(coarse_rot))
                cols = np.flatnonzero(mask.fine_translation_parent == int(coarse_trans))
                if rows.size and cols.size:
                    dense[np.ix_(rows, cols)] = False
        return dense
    raise ValueError(f"Unknown sparse candidate mask mode {mask.mode!r}")


def _candidate_mask_to_dense(candidate_mask) -> np.ndarray:
    if isinstance(candidate_mask, SparseCandidateMask):
        return _dense_candidate_mask_from_spec(candidate_mask)
    return np.asarray(candidate_mask, dtype=bool)


def _candidate_mask_count(candidate_mask) -> int:
    if isinstance(candidate_mask, SparseCandidateMask):
        return int(candidate_mask.count)
    return int(np.asarray(candidate_mask, dtype=bool).sum())


def _candidate_mask_is_full(candidate_mask) -> bool:
    if isinstance(candidate_mask, SparseCandidateMask):
        total = int(candidate_mask.n_rows) * int(candidate_mask.n_fine_trans)
        return total > 0 and int(candidate_mask.count) >= total
    dense = np.asarray(candidate_mask, dtype=bool)
    return dense.size > 0 and bool(np.all(dense))


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


def _candidate_mask_nonzero(candidate_mask):
    if isinstance(candidate_mask, SparseCandidateMask):
        if candidate_mask.mode == "empty":
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if candidate_mask.mode == "full":
            rows = np.repeat(np.arange(candidate_mask.n_rows, dtype=np.int64), candidate_mask.n_fine_trans)
            trans = np.tile(np.arange(candidate_mask.n_fine_trans, dtype=np.int64), candidate_mask.n_rows)
            return rows, trans
        if candidate_mask.mode == "coarse_exclude":
            dense = _dense_candidate_mask_from_spec(candidate_mask)
            return np.nonzero(dense)
    return np.nonzero(_candidate_mask_to_dense(candidate_mask))


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
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return

    global _native_mstep_dump_counter
    dump_idx = _native_mstep_dump_counter
    _native_mstep_dump_counter += 1

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_RUN_ID", "unset")
    np.savez_compressed(
        path
        / (
            f"native_half_mstep_it{context_iteration:03d}_h{context_half}"
            f"_dump{dump_idx:03d}_{stage}_n{int(n_images):04d}_cs{int(current_size):03d}.npz"
        ),
        schema=np.asarray("recovar-native-half-mstep-v2"),
        dump_index=np.int64(dump_idx),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        run_id=np.asarray(run_id),
        Ft_y=np.asarray(Ft_y_total),
        Ft_ctf=np.asarray(Ft_ctf_total),
        current_size=np.int32(current_size),
        n_images=np.int32(n_images),
        recon_volume_shape=np.asarray(recon_volume_shape, dtype=np.int32),
        stage=np.asarray(stage),
    )


def _maybe_dump_bpref_contribution_rows(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    summed,
    ctf_probs,
    rotations,
    actual_counts,
    rotation_indices,
    fine_translations,
    scores,
    preprior_scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    log_z,
    best_log_score,
    reconstruction_probs,
    reconstruction_mask,
    reconstruction_sum_weight,
    reconstruction_threshold,
    candidate_mask,
    high_precision_operand_bundle,
    raw_batch_data,
    ctf_params,
    noise_variance_half,
    integer_pre_shifts,
    batch_image_corrections,
    batch_scale_corrections,
    relion_preprocess_normalization_factors,
    relion_cuda_preprocess,
    score_with_masked_images,
    image_mask,
    image_mask_mode,
    voxel_size,
    ctf_mode,
    ctf_dose_per_tilt,
    ctf_angle_per_tilt,
    disc_type,
    projection_padding_factor,
    reconstruction_padding_factor,
    use_relion_x_half_mstep,
    winner_take_all,
    max_r,
    window_indices,
    image_shape,
    volume_shape,
    shadow_only_mode,
    shadow_score_bitwise_equal,
    shadow_reduction_agreement,
    device_signature_active: bool | None = None,
    class_index: int = 0,
    mstep_shifted_recon=None,
    mstep_ctf2_over_nv=None,
):
    """Dump posterior-reduced active rows for whole-accumulator scatter replay.

    This diagnostic boundary is immediately before the x-half backprojection.
    Files retain bucket execution order, particle ownership, and every valid
    rotation row, including exact-zero rows.  The companion device signature
    limits only its signature-only output arrays to exact positive-weight
    contributors; its native accumulator launch still receives every row.
    Replaying every contribution file in counter order therefore permits a
    streaming closure check without materializing one 3-D accumulator per
    particle.
    """

    if (
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
        and device_signature_active is not True
    ):
        return
    dump_dir = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR")
    if not dump_dir:
        return
    class_index = int(class_index)
    if class_index < 0:
        raise ValueError("BPref contribution class_index must be non-negative")
    global _bpref_contribution_call_counter
    call_idx = _bpref_contribution_call_counter
    _bpref_contribution_call_counter += 1
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return
    target_half = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF")
    if target_half:
        if int(target_half) not in {1, 2}:
            raise ValueError("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF must be 1 or 2")
        if context_half != int(target_half):
            return
    target_current_size = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return

    preprocess_backend_object = image_preprocess_backend(experiment_dataset)
    relion_native_lane_reduction = bool(
        getattr(preprocess_backend_object, "relion_native_lane_reduction", False)
    )
    if relion_native_lane_reduction and not relion_cuda_preprocess:
        raise ValueError(
            "native-lane preprocessing telemetry requires the RELION CUDA backend"
        )

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    image_identities = _bpref_image_identities_for_original_indices(original_indices)
    stack_sha256 = _bpref_required_stack_checksum()
    selected_particle_rows = _bpref_contribution_target_rows(
        experiment_dataset,
        local_indices,
    )
    if os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES", "").strip():
        if selected_particle_rows.size == 0:
            return
        local_indices = local_indices[selected_particle_rows]
        original_indices = original_indices[selected_particle_rows]
        image_identities = image_identities[selected_particle_rows]

    def _select_particle_axis(values):
        values_np = np.asarray(values)
        if values_np.ndim > 0 and values_np.shape[0] == np.asarray(image_indices).size:
            return values_np[selected_particle_rows]
        return values_np

    actual_counts_np = _select_particle_axis(actual_counts).astype(np.int64, copy=False)
    summed_np = _select_particle_axis(summed)
    ctf_probs_np = _select_particle_axis(ctf_probs)
    rotations_np = _select_particle_axis(rotations)
    if summed_np.shape[:2] != ctf_probs_np.shape[:2] or summed_np.shape[:2] != rotations_np.shape[:2]:
        raise ValueError("BPref contribution dump requires matching particle/rotation axes")
    if actual_counts_np.shape != (summed_np.shape[0],):
        raise ValueError("BPref contribution dump actual_counts shape mismatch")

    rotation_rows = np.arange(summed_np.shape[1], dtype=np.int64)[None, :]
    valid = rotation_rows < actual_counts_np[:, None]
    # Preserve every valid rotation row, including exact-zero rows.  A strict
    # RELION/RECOVAR four-arm replay must distinguish a genuine support/value
    # difference from a row silently omitted by the diagnostic writer.
    active = valid
    active_particle_rows, active_rotation_rows = np.nonzero(active)
    rotation_indices_np = _select_particle_axis(rotation_indices).astype(np.int64, copy=False)
    if rotation_indices_np.ndim == 1:
        rotation_indices_np = np.broadcast_to(rotation_indices_np[None, :], summed_np.shape[:2])
    if rotation_indices_np.shape[:2] != summed_np.shape[:2]:
        raise ValueError("BPref contribution dump rotation_indices shape mismatch")

    global _bpref_contribution_dump_counter
    dump_idx = _bpref_contribution_dump_counter
    _bpref_contribution_dump_counter += 1
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "unset")
    stack_indices = np.asarray([int(value.split("@", 1)[0]) for value in image_identities], dtype=np.int64)
    stack_paths = np.asarray([value.split("@", 1)[1] for value in image_identities])
    if high_precision_operand_bundle:
        raw_real_images = _select_particle_axis(raw_batch_data)
        if np.iscomplexobj(raw_real_images):
            raise ValueError("BPref raw source images must be real, not Fourier/complex samples")
        expected_raw_shape = (raw_real_images.shape[0], int(image_shape[0]), int(image_shape[1]))
        if raw_real_images.ndim == 2 and raw_real_images.shape[1] == int(np.prod(image_shape)):
            raw_real_images = raw_real_images.reshape(expected_raw_shape)
        if raw_real_images.shape != expected_raw_shape:
            raise ValueError(
                "BPref raw source images must have shape (B,H,W) before FFT/preprocessing, "
                f"got {raw_real_images.shape}, expected {expected_raw_shape}"
            )
        raw_source_dtype = str(raw_real_images.dtype)
        raw_real_images = raw_real_images.astype(np.float32, copy=False)
        captured_ctf_params = _select_particle_axis(ctf_params)
        captured_noise_variance_half = np.asarray(noise_variance_half)
        captured_integer_pre_shifts = _select_particle_axis(integer_pre_shifts).astype(np.int32, copy=False)
        captured_image_corrections = _select_particle_axis(batch_image_corrections).astype(
            np.float32, copy=False
        )
        captured_scale_corrections = _select_particle_axis(batch_scale_corrections).astype(
            np.float32, copy=False
        )
        captured_normalization_factors = _select_particle_axis(
            relion_preprocess_normalization_factors
        ).astype(np.float32, copy=False)
        captured_image_mask = np.asarray(image_mask, dtype=np.float32)
        captured_mstep_shifted_recon = (
            np.empty((0,), dtype=np.complex64)
            if mstep_shifted_recon is None
            else _select_particle_axis(mstep_shifted_recon)
        )
        captured_mstep_ctf2_over_nv = (
            np.empty((0,), dtype=np.float32)
            if mstep_ctf2_over_nv is None
            else _select_particle_axis(mstep_ctf2_over_nv)
        )
    else:
        raw_real_images = np.empty((0,), dtype=np.float32)
        raw_source_dtype = ""
        captured_ctf_params = np.empty((0,), dtype=np.float32)
        captured_noise_variance_half = np.empty((0,), dtype=np.float32)
        captured_integer_pre_shifts = np.empty((0, 2), dtype=np.int32)
        captured_image_corrections = np.empty((0,), dtype=np.float32)
        captured_scale_corrections = np.empty((0,), dtype=np.float32)
        captured_normalization_factors = np.empty((0,), dtype=np.float32)
        captured_image_mask = np.empty((0,), dtype=np.float32)
        captured_mstep_shifted_recon = np.empty((0,), dtype=np.complex64)
        captured_mstep_ctf2_over_nv = np.empty((0,), dtype=np.float32)
    rotation_log_prior_np = _select_particle_axis(rotation_log_prior).astype(np.float64, copy=False)
    translation_log_prior_np = _select_particle_axis(translation_log_prior).astype(np.float64, copy=False)
    combined_scores_np = _select_particle_axis(scores).astype(np.float64, copy=False)
    preprior_scores_np = _select_particle_axis(preprior_scores).astype(np.float64, copy=False)
    best_log_score_np = _select_particle_axis(best_log_score).astype(np.float64, copy=False)
    log_z_np = _select_particle_axis(log_z).astype(np.float64, copy=False)
    normalized_sum_exp = np.exp(log_z_np - best_log_score_np)
    captured_reconstruction_probs = _select_particle_axis(reconstruction_probs)
    if captured_reconstruction_probs.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(
            "BPref reconstruction probabilities must retain native float32/float64 dtype, "
            f"got {captured_reconstruction_probs.dtype}"
        )
    scores_f32 = combined_scores_np.astype(np.float32)
    best_f32 = np.max(np.where(np.isfinite(scores_f32), scores_f32, -np.inf), axis=(1, 2))
    exponent_shift_f32 = np.float32(50.0) - best_f32
    shifted_f32 = scores_f32 + exponent_shift_f32[:, None, None]
    raw_exp_weights_f32 = np.where(
        np.isfinite(shifted_f32) & (shifted_f32 >= np.float32(-88.0)),
        np.exp(shifted_f32, dtype=np.float32),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    contribution_path = path / (
            f"bpref_contribution_rows_it{context_iteration:03d}_h{context_half}"
            f"_call{call_idx:06d}_dump{dump_idx:06d}_cs{int(current_size):03d}.npz"
        )
    np.savez(
        contribution_path,
        magic=np.asarray("RECOVAR_BPREF_CONTRIBUTION_ROWS"),
        schema=np.asarray("recovar-bpref-contribution-rows-v3"),
        schema_version=np.int32(3),
        dump_index=np.int64(dump_idx),
        call_index=np.int64(call_idx),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        rank=np.int32(int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0"))),
        pass_index=np.int32(2),
        class_index=np.int32(class_index),
        run_id=np.asarray(run_id),
        current_size=np.int64(current_size),
        image_shape=np.asarray(image_shape, dtype=np.int32),
        volume_shape=np.asarray(volume_shape, dtype=np.int32),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        local_indices=local_indices,
        original_indices=original_indices,
        star_rows=original_indices,
        image_identities=image_identities,
        stack_indices_1based=stack_indices,
        resolved_stack_paths=stack_paths,
        source_stack_sha256=np.asarray(stack_sha256),
        shadow_only_mode=np.bool_(shadow_only_mode),
        shadow_score_bitwise_equal=np.bool_(shadow_score_bitwise_equal),
        shadow_reduction_data_rel_l1=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["data_rel_l1"]
        ),
        shadow_reduction_data_normalized_max=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["data_normalized_max"]
        ),
        shadow_reduction_weight_rel_l1=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["weight_rel_l1"]
        ),
        shadow_reduction_weight_normalized_max=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["weight_normalized_max"]
        ),
        shadow_reduction_rel_l1_bound=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["rel_l1_bound"]
        ),
        shadow_reduction_normalized_max_bound=np.float64(
            np.nan if shadow_reduction_agreement is None
            else shadow_reduction_agreement["normalized_max_bound"]
        ),
        high_precision_operand_bundle=np.bool_(high_precision_operand_bundle),
        raw_real_images=raw_real_images,
        raw_source_dtype=np.asarray(raw_source_dtype),
        raw_source_shape=np.asarray(raw_real_images.shape, dtype=np.int64),
        ctf_params=captured_ctf_params,
        ctf_parameter_convention=np.asarray(
            "recovar.CTFParamIndex-v1:DFU[A],DFV[A],DFANG[deg],VOLT[kV],CS[mm],"
            "W[amplitude_fraction],PHASE_SHIFT[deg],BFACTOR[A^2],CONTRAST,DOSE[e-/A^2],TILT_ANGLE[deg]"
        ),
        noise_variance_half=captured_noise_variance_half,
        integer_pre_shifts=captured_integer_pre_shifts,
        image_corrections=captured_image_corrections,
        scale_corrections=captured_scale_corrections,
        relion_preprocess_normalization_factors=captured_normalization_factors,
        relion_cuda_preprocess=np.bool_(relion_cuda_preprocess),
        relion_native_lane_reduction=np.bool_(relion_native_lane_reduction),
        preprocess_backend=np.asarray("relion_cuda" if relion_cuda_preprocess else "dataset_native"),
        preprocess_convention=np.asarray("recovar-half-preprocess-v1"),
        score_with_masked_images=np.bool_(score_with_masked_images),
        image_mask=captured_image_mask,
        image_mask_mode=np.asarray(image_mask_mode),
        voxel_size=np.float64(voxel_size),
        ctf_mode=np.asarray(ctf_mode),
        ctf_dose_per_tilt=np.float64(ctf_dose_per_tilt),
        ctf_angle_per_tilt=np.float64(ctf_angle_per_tilt),
        disc_type=np.asarray(disc_type),
        projection_padding_factor=np.int32(projection_padding_factor),
        reconstruction_padding_factor=np.int32(reconstruction_padding_factor),
        actual_counts=actual_counts_np,
        oversampled_rotation_indices=rotation_indices_np,
        fine_translations=np.asarray(fine_translations, dtype=np.float32),
        candidate_preprior_scores=preprior_scores_np,
        candidate_rotation_log_prior=rotation_log_prior_np,
        candidate_translation_log_prior=translation_log_prior_np,
        candidate_combined_scores=combined_scores_np,
        candidate_best_log_score=best_log_score_np,
        candidate_log_z=log_z_np,
        candidate_normalized_sum_exp=normalized_sum_exp,
        candidate_exponent_shift_f32=exponent_shift_f32,
        candidate_raw_exp_weights_f32=raw_exp_weights_f32,
        posterior_probs=_select_particle_axis(probs).astype(np.float64, copy=False),
        reconstruction_probs=captured_reconstruction_probs,
        reconstruction_probs_native_dtype=np.asarray(
            str(captured_reconstruction_probs.dtype)
        ),
        reconstruction_probs_native_itemsize=np.int32(
            captured_reconstruction_probs.dtype.itemsize
        ),
        reconstruction_probs_native_nbytes=np.int64(
            captured_reconstruction_probs.nbytes
        ),
        # Additive v3 fields: old readers ignore unknown NPZ members, while
        # new high-precision replay fails closed if any member is missing.
        reconstruction_probs_storage_policy=np.asarray(
            "native-dtype-preserved;dtype-itemsize-nbytes-bound"
        ),
        mstep_shifted_recon=captured_mstep_shifted_recon,
        mstep_ctf2_over_nv=captured_mstep_ctf2_over_nv,
        reconstruction_mask=_select_particle_axis(reconstruction_mask).astype(bool, copy=False),
        reconstruction_sum_weight=_select_particle_axis(reconstruction_sum_weight).astype(np.float64, copy=False),
        reconstruction_threshold=_select_particle_axis(reconstruction_threshold).astype(np.float64, copy=False),
        candidate_mask=_select_particle_axis(candidate_mask).astype(bool, copy=False),
        active_particle_rows=active_particle_rows.astype(np.int32, copy=False),
        active_rotation_rows=active_rotation_rows.astype(np.int32, copy=False),
        active_original_indices=original_indices[active_particle_rows],
        active_global_rotation_indices=rotation_indices_np[active_particle_rows, active_rotation_rows],
        active_summed=summed_np[active_particle_rows, active_rotation_rows],
        active_ctf_probs=ctf_probs_np[active_particle_rows, active_rotation_rows],
        active_rotations=rotations_np[active_particle_rows, active_rotation_rows],
    )

    device_signature_path = None
    device_dump_dir = os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    if device_dump_dir:
        from recovar import cuda_backproject

        _require_bpref_device_soft_particle_arm(
            use_relion_x_half_mstep=bool(use_relion_x_half_mstep),
        )
        if max_r is None:
            raise RuntimeError("RECOVAR device signature requires the explicit production support radius")
        if context_iteration <= 0 or context_half not in {1, 2}:
            raise RuntimeError("RECOVAR device signature requires explicit positive iteration/half context")
        accumulator_key = (context_iteration, context_half, run_id, int(class_index))
        accumulators = _bpref_device_panel_accumulators.get(accumulator_key)
        accumulator_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
        if accumulators is None:
            accumulators = (
                jnp.zeros((accumulator_size,), dtype=jnp.complex64),
                jnp.zeros((accumulator_size,), dtype=jnp.float32),
            )
        launch_ordinal = _bpref_device_panel_launch_counters.get(accumulator_key, 0)
        signature_chunks = [[] for _ in range(7)]
        signature_launch_ordinals = []
        signature_particle_local_rows = []
        signature_image_identities = []
        signature_original_indices = []
        signature_contributor_rotation_keys = []
        particle_launch_ordinals = []
        particle_total_row_counts = []
        particle_contributor_row_counts = []
        particle_noncontributor_row_counts = []
        particle_noncontributor_zero_sha256 = []
        particle_image_identities = []
        particle_original_indices = []
        # Production uses one stream-ordered CUDA launch per particle.  Keep
        # those launch boundaries exact; flattening several particles into one
        # grid changes inter-particle atomic scheduling.
        for particle_row in range(summed_np.shape[0]):
            row_count = int(actual_counts_np[particle_row])
            if row_count <= 0:
                continue
            particle_rotation_keys = np.asarray(rotation_indices_np[particle_row, :row_count], dtype=np.int64)
            if particle_rotation_keys.size and (
                int(particle_rotation_keys.min()) < np.iinfo(np.int32).min
                or int(particle_rotation_keys.max()) > np.iinfo(np.int32).max
            ):
                raise OverflowError("RECOVAR canonical rotation key exceeds device int32 range")
            particle_summed = np.asarray(summed_np[particle_row, :row_count], dtype=np.complex64)
            particle_weights = np.asarray(ctf_probs_np[particle_row, :row_count], dtype=np.float32)
            if not np.all(np.isfinite(particle_summed)) or not np.all(np.isfinite(particle_weights)):
                raise RuntimeError("RECOVAR soft-particle causal arm encountered a nonfinite scatter operand")
            if np.any(particle_weights < 0):
                raise RuntimeError("RECOVAR soft-particle causal arm encountered a negative scatter weight")
            contributor_rows = np.flatnonzero(np.any(particle_weights > 0, axis=1)).astype(
                np.int32, copy=False
            )
            noncontributor_mask = np.ones(row_count, dtype=bool)
            noncontributor_mask[contributor_rows] = False
            noncontributor_summed = np.ascontiguousarray(particle_summed[noncontributor_mask])
            noncontributor_weights = np.ascontiguousarray(particle_weights[noncontributor_mask])
            noncontributor_rows = np.flatnonzero(noncontributor_mask).astype(np.int32, copy=False)
            noncontributor_rotation_keys = particle_rotation_keys[noncontributor_mask].astype(
                np.int32, copy=False
            )
            if np.any(noncontributor_summed != 0) or np.any(noncontributor_weights != 0):
                raise RuntimeError(
                    "RECOVAR device signature requires every omitted signature row to be exactly zero"
                )
            zero_digest = hashlib.sha256()
            zero_digest.update(noncontributor_rows.tobytes(order="C"))
            zero_digest.update(noncontributor_rotation_keys.tobytes(order="C"))
            zero_digest.update(str(noncontributor_summed.dtype).encode("ascii"))
            zero_digest.update(np.asarray(noncontributor_summed.shape, dtype=np.int64).tobytes())
            zero_digest.update(noncontributor_summed.tobytes(order="C"))
            zero_digest.update(str(noncontributor_weights.dtype).encode("ascii"))
            zero_digest.update(np.asarray(noncontributor_weights.shape, dtype=np.int64).tobytes())
            zero_digest.update(noncontributor_weights.tobytes(order="C"))

            ffi_args = (
                accumulators[0],
                accumulators[1],
                jnp.asarray(particle_summed, dtype=jnp.complex64),
                jnp.asarray(particle_weights, dtype=jnp.float32),
                jnp.asarray(window_indices, dtype=jnp.int32),
                jnp.asarray(rotations_np[particle_row, :row_count], dtype=jnp.float32),
            )
            if contributor_rows.size:
                signature_outputs = cuda_backproject.relion_fused_x_half_backproject_signature_indexed(
                    *ffi_args,
                    jnp.asarray(particle_rotation_keys, dtype=jnp.int32),
                    jnp.asarray(contributor_rows, dtype=jnp.int32),
                    tuple(int(value) for value in image_shape),
                    tuple(int(value) for value in volume_shape),
                    float(max_r),
                )
                accumulators = signature_outputs[:2]
                for output_index, output in enumerate(signature_outputs[2:]):
                    signature_chunks[output_index].append(np.asarray(output))
                signature_launch_ordinals.append(
                    np.full(contributor_rows.size, launch_ordinal, dtype=np.int64)
                )
                signature_particle_local_rows.append(contributor_rows)
                signature_image_identities.append(
                    np.full(contributor_rows.size, image_identities[particle_row])
                )
                signature_original_indices.append(
                    np.full(contributor_rows.size, original_indices[particle_row], dtype=np.int64)
                )
                signature_contributor_rotation_keys.append(
                    particle_rotation_keys[contributor_rows].astype(np.int32, copy=False)
                )
            else:
                # Preserve the native all-row launch even when no row passes
                # its Fweight>0 gate; only the signature-only launch is absent.
                accumulators = cuda_backproject.relion_fused_x_half_backproject_indexed(
                    *ffi_args,
                    tuple(int(value) for value in image_shape),
                    tuple(int(value) for value in volume_shape),
                    float(max_r),
                )
            particle_launch_ordinals.append(launch_ordinal)
            particle_total_row_counts.append(row_count)
            particle_contributor_row_counts.append(int(contributor_rows.size))
            particle_noncontributor_row_counts.append(int(row_count - contributor_rows.size))
            particle_noncontributor_zero_sha256.append(zero_digest.hexdigest())
            particle_image_identities.append(image_identities[particle_row])
            particle_original_indices.append(original_indices[particle_row])
            launch_ordinal += 1
        if not particle_launch_ordinals:
            raise RuntimeError("RECOVAR device signature selected no particle launches")
        _bpref_device_panel_accumulators[accumulator_key] = accumulators
        _bpref_device_panel_launch_counters[accumulator_key] = launch_ordinal
        metadata = {
            "current_size": int(current_size),
            "max_r": float(max_r),
            "image_shape": tuple(int(value) for value in image_shape),
            "volume_shape": tuple(int(value) for value in volume_shape),
            "reconstruction_padding_factor": int(reconstruction_padding_factor),
            "source_stack_sha256": stack_sha256,
            "rank": int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0")),
            "causal_arm": (
                "winner-take-all-per-particle-fused-xhalf"
                if winner_take_all
                else "soft-posterior-per-particle-fused-xhalf"
            ),
            "winner_take_all": bool(winner_take_all),
            "class_index": int(class_index),
        }
        previous_metadata = _bpref_device_panel_metadata.setdefault(accumulator_key, metadata)
        if previous_metadata != metadata:
            raise RuntimeError("RECOVAR device panel metadata changed within one half")
        dense_height = 2 * int(round(float(max_r)))
        dense_pixel_count = dense_height * (dense_height // 2 + 1)
        if signature_chunks[0]:
            (
                signature_rotation_keys,
                signature_pixel_indices,
                signature_row_flags,
                signature_source_values,
                signature_neighbor_indices,
                signature_neighbor_coefficients,
                signature_neighbor_flags,
            ) = (np.concatenate(chunks, axis=0) for chunks in signature_chunks)
            signature_launch_ordinals = np.concatenate(signature_launch_ordinals)
            signature_particle_local_rows = np.concatenate(signature_particle_local_rows)
            signature_image_identities = np.concatenate(signature_image_identities)
            signature_original_indices = np.concatenate(signature_original_indices)
            signature_contributor_rotation_keys = np.concatenate(signature_contributor_rotation_keys)
        else:
            empty_signature = _empty_bpref_device_signature_arrays(
                dense_pixel_count,
                image_identity_dtype=np.asarray(image_identities).dtype,
            )
            signature_rotation_keys = empty_signature["rotation_keys"]
            signature_pixel_indices = empty_signature["pixel_indices"]
            signature_row_flags = empty_signature["row_flags"]
            signature_source_values = empty_signature["source_values"]
            signature_neighbor_indices = empty_signature["neighbor_indices"]
            signature_neighbor_coefficients = empty_signature["neighbor_coefficients"]
            signature_neighbor_flags = empty_signature["neighbor_flags"]
            signature_launch_ordinals = empty_signature["launch_ordinals"]
            signature_particle_local_rows = empty_signature["particle_local_rows"]
            signature_image_identities = empty_signature["image_identities"]
            signature_original_indices = empty_signature["original_indices"]
            signature_contributor_rotation_keys = empty_signature[
                "contributor_rotation_keys"
            ]
        device_path = Path(device_dump_dir)
        device_path.mkdir(parents=True, exist_ok=True)
        contribution_sha256 = _sha256_file(contribution_path)
        device_signature_path = device_path / f"{contribution_path.stem}.device.npz"
        np.savez(
            device_signature_path,
            magic=np.asarray("RECOVAR_DEVICE_SCATTER_SIGNATURE"),
            schema=np.asarray("recovar-device-scatter-signature-v1"),
            schema_version=np.int32(1),
            run_id=np.asarray(run_id),
            iteration=np.int32(context_iteration),
            half=np.int32(context_half),
            rank=np.int32(int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0"))),
            pass_index=np.int32(2),
            class_index=np.int32(class_index),
            call_index=np.int64(call_idx),
            dump_index=np.int64(dump_idx),
            source_stack_sha256=np.asarray(stack_sha256),
            companion_contribution_path=np.asarray(str(contribution_path.resolve())),
            companion_contribution_sha256=np.asarray(contribution_sha256),
            image_shape=np.asarray(image_shape, dtype=np.int32),
            volume_shape=np.asarray(volume_shape, dtype=np.int32),
            current_size=np.int32(current_size),
            max_r=np.float32(max_r),
            causal_arm=np.asarray(
                "winner-take-all-per-particle-fused-xhalf"
                if winner_take_all
                else "soft-posterior-per-particle-fused-xhalf"
            ),
            winner_take_all=np.bool_(winner_take_all),
            topology_claim=np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
            signature_inertness_gate=np.asarray(
                "bitwise-post-accum-shadow-and-operand-exact"
            ),
            signature_inertness_gate_passed=np.bool_(True),
            signature_accumulator_shadow_bitwise_equal=np.bool_(True),
            signature_prepared_operands_bitwise_equal=np.bool_(True),
            signature_kernel_accumulate=np.bool_(False),
            reconstruction_padding_factor=np.int32(reconstruction_padding_factor),
            particle_launch_ordinals=np.asarray(particle_launch_ordinals, dtype=np.int64),
            particle_total_row_counts=np.asarray(particle_total_row_counts, dtype=np.int32),
            particle_contributor_row_counts=np.asarray(
                particle_contributor_row_counts, dtype=np.int32
            ),
            particle_noncontributor_row_counts=np.asarray(
                particle_noncontributor_row_counts, dtype=np.int32
            ),
            particle_noncontributor_exact_zero=np.ones(
                len(particle_launch_ordinals), dtype=bool
            ),
            particle_noncontributor_zero_sha256=np.asarray(
                particle_noncontributor_zero_sha256
            ),
            particle_image_identities=np.asarray(particle_image_identities),
            particle_original_indices=np.asarray(particle_original_indices, dtype=np.int64),
            signature_bytes_per_dense_row_pixel=np.int32(132),
            signature_estimated_uncompressed_bytes=np.int64(
                int(signature_contributor_rotation_keys.size) * dense_pixel_count * 132
            ),
            launch_ordinal=signature_launch_ordinals,
            particle_local_row=signature_particle_local_rows,
            image_identity=signature_image_identities,
            original_indices=signature_original_indices,
            contributor_canonical_rotation_keys=signature_contributor_rotation_keys,
            canonical_rotation_keys=signature_rotation_keys,
            canonical_pixel_indices=signature_pixel_indices,
            row_flags=signature_row_flags,
            source_values=signature_source_values,
            neighbor_indices=signature_neighbor_indices,
            neighbor_coefficients=signature_neighbor_coefficients,
            neighbor_flags=signature_neighbor_flags,
            program_row=signature_particle_local_rows,
            program_lane=np.arange(dense_pixel_count, dtype=np.int32) % np.int32(128),
            program_serial_pass=np.arange(dense_pixel_count, dtype=np.int32) // np.int32(128),
            program_neighbor=np.arange(8, dtype=np.int32),
            program_axis_sizes=np.asarray(
                [
                    int(signature_contributor_rotation_keys.size),
                    dense_pixel_count,
                    8,
                ],
                dtype=np.int64,
            ),
            signature_tensor_axis_legend=np.asarray(
                "row-major [contributor_row,dense_pixel,neighbor]; program_row is the "
                "particle-local source rotation row; lane=dense_pixel%128; "
                "serial_pass=dense_pixel//128; neighbor=d0*4+d1*2+d2"
            ),
            atomic_component_program_order_legend=np.asarray(
                "for each valid neighbor: atomicAdd(data_real), then atomicAdd(data_imag), "
                "then atomicAdd(weight)"
            ),
            row_flag_legend=np.asarray(
                "1=redundant-x0;2=2d-radius;4=nonpositive-weight;8=3d-radius;"
                "16=orientation-fold;32=compact-oob;64=reached-scatter"
            ),
            neighbor_flag_legend=np.asarray("1=valid;2=hermitian-fold;4=nyquist;8=oob"),
            source_value_legend=np.asarray("data_re,data_im,Fweight,rk0,rk1,rk2 (pre-orientation-fold)"),
        )
    _maybe_stop_after_bpref_contribution_dump(
        contribution_path=contribution_path,
        device_signature_path=device_signature_path,
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
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    relion_parent_execution_order=False,
    dtype: np.dtype = np.float32,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Mirrors the per-image branch in the reference implementation in
    :func:`compute_pass2_stats_sparse_perimage_reference` exactly so the
    batched path is a strict per-image equivalent.

    ``dtype`` controls the precision of the RELION-supplied fine rotation
    override (``fine_rotations_override`` / ``fine_mstep_rotations_override``).
    RELION's own fine-search rotation matrices stay ``RFLOAT`` (double) end to
    end in a double-precision build; pass ``precision_policy.score_real_dtype``
    from the caller so this matches ``use_float64_scoring`` instead of always
    narrowing to float32.
    """
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    n_images = len(significant_sample_indices)
    per_image_oversampled_rots = []
    per_image_oversampled_mstep_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []
    full_unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
    full_support_rotation_cache = None
    full_support_log_prior_cache = None
    full_support_zero_log_prior_cache = None
    full_support_candidate_mask_cache = None

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=np.float32)
    else:
        rotation_log_prior_np = None

    fine_rotations_np = None
    fine_mstep_rotations_np = None
    fine_parent_np = None
    if fine_rotations_override is None and fine_rotation_parent_override is None:
        pass
    elif fine_rotations_override is not None and fine_rotation_parent_override is not None:
        fine_rotations_np = np.asarray(fine_rotations_override, dtype=dtype)
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

    if fine_mstep_rotations_override is not None:
        if fine_rotations_np is None:
            raise ValueError("fine_mstep_rotations_override requires fine_rotations_override")
        fine_mstep_rotations_np = np.asarray(fine_mstep_rotations_override, dtype=dtype)
        if fine_mstep_rotations_np.shape != fine_rotations_np.shape:
            raise ValueError(
                "fine_mstep_rotations_override must match fine_rotations_override shape: "
                f"{fine_mstep_rotations_np.shape} vs {fine_rotations_np.shape}",
            )

    def _reorder_children(rotations, parent_map, rotation_indices, parent_ids):
        if not relion_parent_execution_order:
            return rotations, parent_map, rotation_indices
        parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
        n_pixels = 12 * (2 ** int(nside_level)) ** 2
        n_psi = 6 * 2 ** int(nside_level)
        if parent_ids.shape != np.asarray(parent_map).shape:
            raise ValueError("RELION parent execution keys must match fine rotations")
        if parent_ids.size and (
            int(parent_ids.min(initial=0)) < 0
            or int(parent_ids.max(initial=-1)) >= int(n_pixels * n_psi)
        ):
            raise ValueError("RELION parent execution key is outside the coarse grid")
        relion_parent_key = (parent_ids % n_pixels) * n_psi + parent_ids // n_pixels
        order = np.argsort(relion_parent_key, kind="stable")
        return (
            np.asarray(rotations)[order],
            np.asarray(parent_map)[order],
            np.asarray(rotation_indices)[order],
        )

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        coarse_excluded = None
        if sig_samples is None:
            unique_rot = full_unique_rot
            use_full_candidate_mask = True
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
        elif isinstance(sig_samples, ComplementSignificantSampleIndices):
            if int(sig_samples.total_size) != int(n_coarse_rot * n_coarse_trans):
                raise ValueError(
                    "Complement significant sample mask total_size does not match coarse pose grid: "
                    f"{int(sig_samples.total_size)} vs {int(n_coarse_rot * n_coarse_trans)}",
                )
            unique_rot = full_unique_rot
            use_full_candidate_mask = False
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
            coarse_excluded = np.asarray(sig_samples.excluded_indices, dtype=np.int32).reshape(-1)
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                coarse_rot = np.empty(0, dtype=np.int32)
                coarse_trans = np.empty(0, dtype=np.int32)
                unique_rot = np.array([0], dtype=np.int32)
                use_full_candidate_mask = False
                use_full_rotation_support = False
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot)
                use_full_candidate_mask = False
                use_full_rotation_support = False

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        if use_full_rotation_support:
            if full_support_rotation_cache is None:
                if fine_rotations_override is None and fine_rotation_parent_override is None:
                    full_rots, full_parent_map, full_rot_indices = get_oversampled_rotation_grid_from_samples(
                        full_unique_rot,
                        nside_level,
                        oversampling_order=oversampling_order,
                        random_perturbation=random_perturbation,
                        return_rotation_indices=True,
                    )
                    full_support_rotation_cache = (
                        np.asarray(full_rots, dtype=np.float32),
                        np.asarray(full_parent_map, dtype=np.int32),
                        np.asarray(full_rot_indices, dtype=np.int64),
                    )
                elif fine_rotations_np is not None and fine_parent_np is not None:
                    full_support_rotation_cache = (
                        fine_rotations_np,
                        fine_parent_np.astype(np.int32, copy=False),
                        np.arange(fine_rotations_np.shape[0], dtype=np.int64),
                    )
                else:
                    raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")
                full_support_rotation_cache = _reorder_children(
                    *full_support_rotation_cache,
                    parent_ids=full_unique_rot[full_support_rotation_cache[1]],
                )
            oversampled_rots, parent_map, oversampled_rot_indices = full_support_rotation_cache
        elif fine_rotations_override is None and fine_rotation_parent_override is None:
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
            oversampled_rots, parent_map, oversampled_rot_indices = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                unique_rot[parent_map],
            )
        elif fine_rotations_np is not None and fine_parent_np is not None:
            selected_parent = np.zeros(n_coarse_rot, dtype=bool)
            selected_parent[unique_rot] = True
            child_mask = selected_parent[fine_parent_np]
            oversampled_rot_indices = np.flatnonzero(child_mask).astype(np.int64)
            oversampled_rots = fine_rotations_np[oversampled_rot_indices]
            parent_map = np.searchsorted(unique_rot, fine_parent_np[oversampled_rot_indices]).astype(np.int32)
            oversampled_rots, parent_map, oversampled_rot_indices = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                fine_parent_np[oversampled_rot_indices],
            )
        else:
            raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

        oversampled_mstep_rots = (
            oversampled_rots
            if fine_mstep_rotations_np is None
            else fine_mstep_rotations_np[oversampled_rot_indices]
        )

        if use_full_rotation_support:
            if rotation_log_prior_np is not None:
                if full_support_log_prior_cache is None:
                    full_support_log_prior_cache = rotation_log_prior_np[full_unique_rot][parent_map].astype(
                        np.float32,
                        copy=False,
                    )
                local_rotation_log_prior = full_support_log_prior_cache
            else:
                if full_support_zero_log_prior_cache is None:
                    full_support_zero_log_prior_cache = np.zeros(oversampled_rots.shape[0], dtype=np.float32)
                local_rotation_log_prior = full_support_zero_log_prior_cache
        elif rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=np.float32)

        if use_full_candidate_mask:
            if full_support_candidate_mask_cache is None:
                full_support_candidate_mask_cache = SparseCandidateMask(
                    mode="full",
                    n_rows=oversampled_rots.shape[0],
                    n_fine_trans=n_fine_trans,
                count=int(oversampled_rots.shape[0]) * int(n_fine_trans),
            )
            candidate_mask = full_support_candidate_mask_cache
        elif coarse_excluded is not None:
            excluded = np.unique(coarse_excluded.astype(np.int64, copy=False))
            if excluded.size and (
                int(excluded.min(initial=0)) < 0
                or int(excluded.max(initial=-1)) >= int(n_coarse_rot * n_coarse_trans)
            ):
                raise ValueError("Complement significant sample exclusions must index the coarse pose grid")
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            fine_rot_children = np.bincount(np.asarray(parent_map, dtype=np.int64), minlength=int(n_coarse_rot))
            fine_trans_children = np.bincount(
                np.asarray(fine_translation_parent, dtype=np.int64),
                minlength=int(n_coarse_trans),
            )
            excluded_fine_count = int(
                np.sum(fine_rot_children[excluded_rot] * fine_trans_children[excluded_trans], dtype=np.int64),
            )
            candidate_mask = SparseCandidateMask(
                mode="coarse_exclude",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_excluded=excluded.astype(np.int32, copy=False),
                fine_translation_parent=fine_translation_parent,
                count=int(oversampled_rots.shape[0]) * int(n_fine_trans) - excluded_fine_count,
            )
        elif coarse_trans.size == 0:
            candidate_mask = SparseCandidateMask(
                mode="empty",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                count=0,
            )
        else:
            coarse_valid = np.zeros((unique_rot.size, n_coarse_trans), dtype=bool)
            coarse_valid[np.searchsorted(unique_rot, coarse_rot), coarse_trans] = True
            translated_valid = coarse_valid[:, fine_translation_parent]
            candidate_mask = SparseCandidateMask(
                mode="coarse",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid,
                fine_translation_parent=fine_translation_parent,
                count=int(translated_valid[parent_map].sum()),
            )

        per_image_oversampled_rots.append(oversampled_rots)
        per_image_oversampled_mstep_rots.append(oversampled_mstep_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(np.float32, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    return {
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }


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


def _relion_cuda_score_translation_angles_if_available(
    translations,
    image_shape,
    *,
    enabled,
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
    logger.info(
        "Exact RELION fine Gaussian scoring: using CUDA sincosf score translation"
    )
    return jnp.asarray(
        _relion_translation_angles_f32(translations, image_shape),
        dtype=jnp.float32,
    )


def _prepare_per_image_compact_candidate_pairs(per_image_inputs, *, image_mask=None):
    """Flatten per-image sparse pass-2 masks into valid candidate pairs.

    The current fused K-class path pads rotations and then scores every
    translation in the rectangular ``R_bucket x T`` tile.  These arrays are a
    host-side representation of only the valid ``(rotation, translation)``
    candidates; the default scoring path does not consume them yet.
    """

    n_images = len(per_image_inputs["candidate_mask"])
    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")
    compact_local_rotation_row = []
    compact_translation_idx = []
    compact_rotation_index = []
    compact_log_prior = []
    compact_pair_mask = []
    pair_counts = np.zeros(n_images, dtype=np.int32)

    for image_idx in range(n_images):
        if image_mask is not None and not bool(image_mask[image_idx]):
            compact_local_rotation_row.append(np.zeros(0, dtype=np.int32))
            compact_translation_idx.append(np.zeros(0, dtype=np.int32))
            compact_rotation_index.append(np.zeros(0, dtype=np.int64))
            compact_log_prior.append(np.zeros(0, dtype=np.float32))
            compact_pair_mask.append(np.zeros(0, dtype=bool))
            continue
        local_rot_rows, translation_idx = _candidate_mask_nonzero(per_image_inputs["candidate_mask"][image_idx])
        local_rot_rows = local_rot_rows.astype(np.int32, copy=False)
        translation_idx = translation_idx.astype(np.int32, copy=False)
        rotation_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=np.float32)

        compact_local_rotation_row.append(local_rot_rows)
        compact_translation_idx.append(translation_idx)
        compact_rotation_index.append(rotation_indices[local_rot_rows].astype(np.int64, copy=False))
        compact_log_prior.append(rotation_log_prior[local_rot_rows].astype(np.float32, copy=False))
        compact_pair_mask.append(np.ones(local_rot_rows.shape[0], dtype=bool))
        pair_counts[image_idx] = int(local_rot_rows.shape[0])

    return {
        "local_rotation_row": compact_local_rotation_row,
        "translation_idx": compact_translation_idx,
        "rotation_index": compact_rotation_index,
        "log_prior": compact_log_prior,
        "pair_mask": compact_pair_mask,
        "pair_counts": pair_counts,
    }


def _compact_pair_counts_from_inputs(compact_inputs_by_class):
    pair_counts_by_class = []
    n_images = None
    for compact_inputs in compact_inputs_by_class:
        pair_counts = np.asarray(compact_inputs["pair_counts"], dtype=np.int64)
        if n_images is None:
            n_images = int(pair_counts.shape[0])
        elif pair_counts.shape[0] != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
        pair_counts_by_class.append(pair_counts)
    return tuple(pair_counts_by_class)


def _compact_pair_counts_from_candidate_masks(per_image_inputs_by_class):
    pair_counts_by_class = []
    n_images = None
    for per_image_inputs in per_image_inputs_by_class:
        candidate_masks = per_image_inputs["candidate_mask"]
        if n_images is None:
            n_images = len(candidate_masks)
        elif len(candidate_masks) != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
        pair_counts_by_class.append(
            np.asarray(
                [_candidate_mask_count(candidate_mask) for candidate_mask in candidate_masks],
                dtype=np.int64,
            ),
        )
    return tuple(pair_counts_by_class)


def _compact_pair_fused_bucket_sizes(pair_counts_by_class, *, pair_block_size_for_quantization=5000):
    if not pair_counts_by_class:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    fused_pair_counts = np.max(np.stack(pair_counts_by_class, axis=0), axis=0)
    pair_bucket_sizes = np.asarray(
        [
            _exact_bucket_rotation_size(int(count), pair_block_size_for_quantization)
            for count in fused_pair_counts
        ],
        dtype=np.int64,
    )
    return fused_pair_counts, pair_bucket_sizes


def _compact_pair_image_mask_for_threshold(
    pair_counts_by_class,
    min_pair_bucket_size: int | None,
    *,
    pair_block_size_for_quantization=5000,
):
    if min_pair_bucket_size is None:
        return None
    _, pair_bucket_sizes = _compact_pair_fused_bucket_sizes(
        pair_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
    )
    return pair_bucket_sizes >= int(min_pair_bucket_size)


def _bucket_sparse_k_class_compact_pair_counts(
    pair_counts_by_class,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    image_mask=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
):
    """Group images by padded valid-pair count for a compact K-class pass-2."""

    n_classes = len(pair_counts_by_class)
    if n_classes == 0:
        return []
    n_images = int(np.asarray(pair_counts_by_class[0]).shape[0])
    if n_images == 0:
        return []
    if image_mask is not None:
        image_mask = np.asarray(image_mask, dtype=bool)
        if image_mask.shape != (n_images,):
            raise ValueError(f"compact pair image mask shape mismatch: {image_mask.shape} vs {(n_images,)}")
    normalized_counts_by_class = []
    for pair_counts in pair_counts_by_class:
        pair_counts = np.asarray(pair_counts, dtype=np.int64)
        if pair_counts.shape[0] != n_images:
            raise ValueError("All classes must have the same image count for compact sparse pass-2")
        normalized_counts_by_class.append(pair_counts)

    _, pair_bucket_sizes = _compact_pair_fused_bucket_sizes(
        normalized_counts_by_class,
        pair_block_size_for_quantization=pair_block_size_for_quantization,
    )
    if image_mask is not None:
        masked_indices = np.nonzero(image_mask)[0]
        if masked_indices.size:
            pair_bucket_sizes = pair_bucket_sizes.copy()
            pair_bucket_sizes[masked_indices] = _coalesce_tail_bucket_sizes(
                pair_bucket_sizes[masked_indices],
                max_images=tail_bucket_coalesce_max_images,
                max_inflation=tail_bucket_coalesce_max_inflation,
                min_bucket_size=tail_bucket_coalesce_min_bucket_size,
                max_hypotheses_per_microbatch=max_pair_candidates_per_microbatch,
                max_images_per_microbatch=max_images_per_microbatch,
                n_fine_trans=1,
                n_classes=n_classes,
            )
    else:
        pair_bucket_sizes = _coalesce_tail_bucket_sizes(
            pair_bucket_sizes,
            max_images=tail_bucket_coalesce_max_images,
            max_inflation=tail_bucket_coalesce_max_inflation,
            min_bucket_size=tail_bucket_coalesce_min_bucket_size,
            max_hypotheses_per_microbatch=max_pair_candidates_per_microbatch,
            max_images_per_microbatch=max_images_per_microbatch,
            n_fine_trans=1,
            n_classes=n_classes,
        )
    processing_order = np.argsort(pair_bucket_sizes, kind="stable").astype(np.int64)
    if image_mask is not None:
        processing_order = processing_order[image_mask[processing_order]]
    unique_bucket_sizes = np.unique(pair_bucket_sizes[processing_order])

    buckets = []
    for pair_bucket_size in unique_bucket_sizes:
        pair_bucket_size = int(pair_bucket_size)
        bucket_image_indices = processing_order[pair_bucket_sizes[processing_order] == pair_bucket_size]
        cap_by_pairs = max(
            1,
            int(max_pair_candidates_per_microbatch) // max(1, int(n_classes) * pair_bucket_size),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_pairs))
        for start in range(0, bucket_image_indices.shape[0], max_per_chunk):
            buckets.append(
                {
                    "pair_bucket_size": pair_bucket_size,
                    "image_indices": np.asarray(
                        bucket_image_indices[start : start + max_per_chunk],
                        dtype=np.int64,
                    ),
                }
            )
    return buckets


def _bucket_sparse_k_class_compact_pair_inputs(
    compact_inputs_by_class,
    *,
    pair_block_size_for_quantization=5000,
    max_pair_candidates_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
):
    return _bucket_sparse_k_class_compact_pair_counts(
        _compact_pair_counts_from_inputs(compact_inputs_by_class),
        pair_block_size_for_quantization=pair_block_size_for_quantization,
        max_pair_candidates_per_microbatch=max_pair_candidates_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        tail_bucket_coalesce_max_images=tail_bucket_coalesce_max_images,
        tail_bucket_coalesce_max_inflation=tail_bucket_coalesce_max_inflation,
        tail_bucket_coalesce_min_bucket_size=tail_bucket_coalesce_min_bucket_size,
    )


def _build_compact_pair_bucket_arrays(bucket, compact_inputs):
    """Stack/pad compact candidate pairs for one class and bucket."""

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    padded_local_rotation_row = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_translation_idx = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_rotation_index = np.zeros((batch, pair_bucket_size), dtype=np.int64)
    padded_log_prior = np.full((batch, pair_bucket_size), -1e30, dtype=np.float32)
    padded_pair_mask = np.zeros((batch, pair_bucket_size), dtype=bool)
    pair_counts = np.zeros(batch, dtype=np.int32)

    for row, image_idx in enumerate(image_indices.tolist()):
        count = int(compact_inputs["pair_counts"][image_idx])
        pair_counts[row] = count
        if count == 0:
            continue
        padded_local_rotation_row[row, :count] = compact_inputs["local_rotation_row"][image_idx]
        padded_translation_idx[row, :count] = compact_inputs["translation_idx"][image_idx]
        padded_rotation_index[row, :count] = compact_inputs["rotation_index"][image_idx]
        padded_log_prior[row, :count] = compact_inputs["log_prior"][image_idx]
        padded_pair_mask[row, :count] = compact_inputs["pair_mask"][image_idx]

    return {
        "image_indices": image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": padded_local_rotation_row,
        "translation_idx": padded_translation_idx,
        "rotation_index": padded_rotation_index,
        "log_prior": padded_log_prior,
        "pair_mask": padded_pair_mask,
    }


def _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, per_image_inputs):
    """Stack/pad compact candidate pairs for one class and bucket on demand."""

    pair_bucket_size = int(bucket["pair_bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    batch = int(image_indices.shape[0])

    padded_local_rotation_row = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_translation_idx = np.full((batch, pair_bucket_size), -1, dtype=np.int32)
    padded_rotation_index = np.zeros((batch, pair_bucket_size), dtype=np.int64)
    padded_log_prior = np.full((batch, pair_bucket_size), -1e30, dtype=np.float32)
    padded_pair_mask = np.zeros((batch, pair_bucket_size), dtype=bool)
    pair_counts = np.zeros(batch, dtype=np.int32)

    for row, image_idx in enumerate(image_indices.tolist()):
        local_rot_rows, translation_idx = _candidate_mask_nonzero(per_image_inputs["candidate_mask"][image_idx])
        count = int(local_rot_rows.shape[0])
        if count > pair_bucket_size:
            raise RuntimeError(
                "Compact K-class sparse pass-2 bucket is too small for image "
                f"{int(image_idx)}: pair_count={count}, bucket_size={pair_bucket_size}",
            )
        pair_counts[row] = count
        if count == 0:
            continue

        local_rot_rows = local_rot_rows.astype(np.int32, copy=False)
        translation_idx = translation_idx.astype(np.int32, copy=False)
        rotation_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotation_log_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=np.float32)

        padded_local_rotation_row[row, :count] = local_rot_rows
        padded_translation_idx[row, :count] = translation_idx
        padded_rotation_index[row, :count] = rotation_indices[local_rot_rows]
        padded_log_prior[row, :count] = rotation_log_prior[local_rot_rows]
        padded_pair_mask[row, :count] = True

    return {
        "image_indices": image_indices,
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": padded_local_rotation_row,
        "translation_idx": padded_translation_idx,
        "rotation_index": padded_rotation_index,
        "log_prior": padded_log_prior,
        "pair_mask": padded_pair_mask,
    }


def _best_compact_pair_from_scores(
    pair_scores,
    pair_mask,
    local_rotation_row,
    translation_idx,
    rotation_index,
):
    """Select best compact candidates without allowing padded pairs to win."""

    scores = np.asarray(pair_scores, dtype=np.float64)
    mask = np.asarray(pair_mask, dtype=bool)
    if scores.shape != mask.shape:
        raise ValueError(f"pair_scores and pair_mask shape mismatch: {scores.shape} vs {mask.shape}")
    masked_scores = np.where(mask, scores, -np.inf)
    has_valid = np.any(np.isfinite(masked_scores), axis=1)
    best_pair = np.argmax(masked_scores, axis=1).astype(np.int32, copy=False)
    safe_best_pair = np.where(has_valid, best_pair, 0)
    row_index = np.arange(scores.shape[0])

    best_score = masked_scores[row_index, safe_best_pair]
    return {
        "pair_index": np.where(has_valid, best_pair, -1).astype(np.int32, copy=False),
        "local_rotation_row": np.where(
            has_valid,
            np.asarray(local_rotation_row, dtype=np.int32)[row_index, safe_best_pair],
            -1,
        ).astype(np.int32, copy=False),
        "translation_idx": np.where(
            has_valid,
            np.asarray(translation_idx, dtype=np.int32)[row_index, safe_best_pair],
            -1,
        ).astype(np.int32, copy=False),
        "rotation_index": np.where(
            has_valid,
            np.asarray(rotation_index, dtype=np.int64)[row_index, safe_best_pair],
            -1,
        ).astype(np.int64, copy=False),
        "score": best_score,
        "has_valid": has_valid,
    }


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
    return _compact_k_class_pair_plan_stats_from_inputs(
        compact_inputs_by_class,
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


def _compact_k_class_pair_plan_stats_from_inputs(
    compact_inputs_by_class,
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
    """Compute compact-pair work counters from prebuilt compact inputs."""

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


def _bucket_pass2_inputs(
    per_image_inputs,
    n_fine_trans,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    small_bucket_coalesce_size=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    processing_order_override=None,
    processing_order_chunk_size=1,
    processing_order_group_by_bucket_size=False,
    processing_order_batch_consecutive_bucket_sizes=False,
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
    if small_bucket_coalesce_size is not None:
        bucket_sizes = _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size)
    bucket_sizes = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        n_fine_trans=n_fine_trans,
        n_classes=1,
    )

    if processing_order_override is not None:
        processing_order = np.asarray(processing_order_override, dtype=np.int64).reshape(-1)
        if processing_order.shape != (n_images,):
            raise ValueError(
                "processing_order_override must have shape "
                f"({n_images},), got {processing_order.shape}",
            )
        if not np.array_equal(np.sort(processing_order), np.arange(n_images, dtype=np.int64)):
            raise ValueError("processing_order_override must be a permutation of image indices")
        if not processing_order_group_by_bucket_size:
            if processing_order_batch_consecutive_bucket_sizes:
                buckets = []
                run_start = 0
                while run_start < n_images:
                    run_bucket_size = int(bucket_sizes[processing_order[run_start]])
                    run_end = run_start + 1
                    while (
                        run_end < n_images
                        and int(bucket_sizes[processing_order[run_end]]) == run_bucket_size
                    ):
                        run_end += 1
                    cap_by_hypotheses = max(
                        1,
                        int(max_hypotheses_per_microbatch)
                        // max(1, run_bucket_size * int(n_fine_trans)),
                    )
                    max_per_chunk = max(
                        1,
                        min(int(max_images_per_microbatch), cap_by_hypotheses),
                    )
                    for start in range(run_start, run_end, max_per_chunk):
                        chunk = processing_order[start : min(start + max_per_chunk, run_end)]
                        buckets.append(
                            {
                                "bucket_size": run_bucket_size,
                                "image_indices": np.asarray(chunk, dtype=np.int64),
                            }
                        )
                    run_start = run_end
                return buckets
            ordered_chunk_size = int(processing_order_chunk_size)
            if ordered_chunk_size <= 0:
                raise ValueError("processing_order_chunk_size must be positive")
            return [
                {
                    "bucket_size": int(np.max(bucket_sizes[chunk])),
                    "image_indices": np.asarray(chunk, dtype=np.int64),
                }
                for start in range(0, n_images, ordered_chunk_size)
                for chunk in (processing_order[start : start + ordered_chunk_size],)
            ]
    else:
        # Group by bucket size, smaller buckets first. The secondary rotation
        # count key is historical RECOVAR behavior; an explicit order keeps
        # RELION order stable within each equal padded-size bucket.
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
    if preserve_bpref_particle_order and diagnostic_order is not None:
        raise ValueError(
            "preserve_bpref_particle_order cannot be combined with "
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV}"
        )
    if preserve_bpref_particle_order:
        return np.arange(int(n_images), dtype=np.int64)
    return diagnostic_order


def _bucket_sparse_k_class_pass2_inputs(
    per_image_inputs_by_class,
    n_fine_trans,
    *,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    small_bucket_threshold=None,
    small_bucket_max_images_per_microbatch=None,
    small_bucket_coalesce_size=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
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
    if small_bucket_coalesce_size is not None:
        fused_bucket_sizes = _coalesce_small_bucket_sizes(fused_bucket_sizes, small_bucket_coalesce_size)
    fused_bucket_sizes = _coalesce_tail_bucket_sizes(
        fused_bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        max_images_per_microbatch=max_images_per_microbatch,
        n_fine_trans=n_fine_trans,
        n_classes=n_classes,
    )
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
        image_cap = int(max_images_per_microbatch)
        if (
            small_bucket_threshold is not None
            and small_bucket_max_images_per_microbatch is not None
            and bucket_size <= int(small_bucket_threshold)
        ):
            image_cap = max(image_cap, int(small_bucket_max_images_per_microbatch))
        max_per_chunk = max(
            1,
            min(
                image_cap,
                cap_by_hypotheses,
            ),
        )
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


def _optional_nonnegative_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


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


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_PASS2_TOP2_DEBUG_INDICES_ENV = "RECOVAR_PASS2_TOP2_DEBUG_INDICES"


def _pass2_top2_debug_target_indices() -> tuple[int, ...]:
    """Diagnostic only: original-dataset image indices to log the fine (pass-2)
    top-2 candidate score margin for, mirroring
    ``k_class._pass1_top2_debug_target_indices`` but for the oversampled
    fine-grid decision within pass-1's surviving coarse cell(s), where the
    per-particle candidate set actually differs (children of that
    particle's own coarse winner).
    """

    raw = os.environ.get(_PASS2_TOP2_DEBUG_INDICES_ENV, "").strip()
    if not raw:
        return ()
    return tuple(int(token) for token in raw.split(",") if token.strip())


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
    return bool(os.environ.get(_PASS2_DUMP_DIR_ENV)) and not _env_flag_enabled(
        _NORM_RESIDUAL_DUMP_ONLY_ENV,
        default=False,
    )


def _pass2_conservative_dump_execution_enabled() -> bool:
    """Keep dump-only planner changes behind an explicit diagnostic opt-in."""

    return _pass2_dump_enabled() and _env_flag_enabled(
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
        and _env_flag_enabled(
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            default=True,
        )
    )


def _windowed_translation_tile_cap_enabled_for_pass() -> bool:
    """Return whether K-class sparse pass-2 should budget translation tiles on active windows."""

    return _env_flag_enabled(
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

    compact_pair_check = _env_flag_enabled(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False)
    return _env_flag_enabled(
        _SPARSE_KCLASS_COMPACT_PAIRS_ENV,
        default=not compact_pair_check,
    )


def _compact_pair_min_bucket_size_for_pass() -> int:
    """Return the hybrid threshold for compact-pair execution buckets."""

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV)
    if explicit is not None:
        return int(explicit)
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
    max_images = _optional_nonnegative_int_env(_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES_ENV)
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

    explicit_max_images = _optional_nonnegative_int_env(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
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


def _compact_pair_tail_bucket_coalesce_params_for_pass() -> tuple[int | None, float | None, int | None]:
    """Return bounded tail-coalescing controls for compact-pair K-class buckets."""

    max_images = _optional_nonnegative_int_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = _optional_nonnegative_int_env(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = _DEFAULT_COMPACT_PAIR_TAIL_BUCKET_COALESCE_MAX_IMAGES
    if int(max_images) <= 1:
        return None, None, None

    max_inflation = _optional_positive_float_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _optional_positive_float_env(_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION

    min_bucket_size = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _optional_positive_int_env(_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
    return int(max_images), float(max_inflation), int(min_bucket_size)


def _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size):
    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    coalesce_size = int(small_bucket_coalesce_size)
    if coalesce_size <= 1:
        return bucket_sizes
    small_or_target_mask = bucket_sizes <= coalesce_size
    if np.unique(bucket_sizes[small_or_target_mask]).size <= 1:
        return bucket_sizes
    return np.where(bucket_sizes < coalesce_size, coalesce_size, bucket_sizes)


def _coalesce_tail_bucket_sizes(
    bucket_sizes,
    *,
    max_images,
    max_inflation,
    min_bucket_size,
    max_hypotheses_per_microbatch,
    max_images_per_microbatch,
    n_fine_trans,
    n_classes,
):
    """Merge only tiny adjacent high-bucket groups under strict padding caps."""

    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    if bucket_sizes.size == 0 or max_images is None:
        return bucket_sizes
    max_images = int(max_images)
    if max_images <= 1:
        return bucket_sizes
    max_inflation = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION
        if max_inflation is None
        else float(max_inflation)
    )
    min_bucket_size = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
        if min_bucket_size is None
        else int(min_bucket_size)
    )
    if max_inflation < 1.0:
        return bucket_sizes

    unique_sizes, inverse, counts = np.unique(
        bucket_sizes,
        return_inverse=True,
        return_counts=True,
    )
    if unique_sizes.size <= 1:
        return bucket_sizes

    assigned_sizes = unique_sizes.copy()
    group_count = unique_sizes.size
    i = 0
    while i < group_count:
        size_i = int(unique_sizes[i])
        count_i = int(counts[i])
        if size_i < min_bucket_size or count_i > max_images:
            i += 1
            continue

        best_j = i
        total_images = 0
        total_rows = 0
        for j in range(i, group_count):
            size_j = int(unique_sizes[j])
            count_j = int(counts[j])
            if size_j < min_bucket_size or count_j > max_images:
                break
            total_images += count_j
            total_rows += size_j * count_j
            if total_images > max_images or total_images > int(max_images_per_microbatch):
                break
            target_size = size_j
            padded_rows = target_size * total_images
            if total_rows <= 0:
                continue
            inflation = float(padded_rows) / float(total_rows)
            # The bucket builder chunks each coalesced size by
            # max_hypotheses_per_microbatch before execution.  Applying the
            # same cap here prevents adjacent tiny high-tail groups from
            # sharing one padded size even when every eventual chunk remains
            # within the score-tensor budget.
            if inflation <= max_inflation:
                best_j = j

        if best_j > i:
            assigned_sizes[i : best_j + 1] = unique_sizes[best_j]
            i = best_j + 1
        else:
            i += 1

    if np.array_equal(assigned_sizes, unique_sizes):
        return bucket_sizes
    return assigned_sizes[inverse]


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
) -> int:
    return (
        int(batch_size)
        * int(bucket_size)
        * int(n_fine_translations)
        * np.dtype(np.float32).itemsize
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
    override = _optional_nonnegative_int_env(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_CACHE_DEVICE_FRACTION))


def _projection_call_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_nonnegative_int_env(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION))


def _bucket_summary(buckets) -> str:
    return _bucket_summary_by_key(buckets, "bucket_size")


def _bucket_summary_by_key(buckets, size_key: str) -> str:
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


def _bucket_group_stats(buckets) -> dict[int, tuple[int, int]]:
    return _bucket_group_stats_by_key(buckets, "bucket_size")


def _bucket_group_stats_by_key(buckets, size_key: str) -> dict[int, tuple[int, int]]:
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
                projector_output_size=projector_output_size,
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
    noise_total = jnp.zeros(shell_count, dtype=jnp.float32)
    a2_total = jnp.zeros(shell_count, dtype=jnp.float32)
    xa_total = jnp.zeros(shell_count, dtype=jnp.float32)
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
    noise_shells = bin_shell_values_jax(block_noise.astype(jnp.float32), shell_indices, shell_count)

    residual_per_row = (a2_per_row - 2.0 * xa_per_row).astype(jnp.float32)
    norm_residual = jnp.zeros(int(batch_size), dtype=jnp.float32).at[flat_image_indices].add(residual_per_row)
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
    noise_shells = bin_shell_values_jax(block_noise.astype(jnp.float32), shell_indices, shell_count)

    residual_per_row = jnp.sum(residual_terms, axis=1).astype(jnp.float32)
    norm_residual = jnp.zeros(int(batch_size), dtype=jnp.float32).at[flat_image_indices].add(residual_per_row)
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
        if _env_flag_enabled(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
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
    noise_total = jnp.zeros(int(shell_count), dtype=jnp.float32)
    norm_total = jnp.zeros(int(batch_size), dtype=jnp.float32)
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
    weighted_half = jnp.sum(weighted_pixel_power, axis=0).astype(jnp.float32)
    weighted_shells = bin_shell_values_jax(weighted_half, shell_indices_half, shell_count)
    source_faithful_spectrum_norm = _env_flag_enabled(
        _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
        default=False,
    )
    deterministic_norm_reduction = source_faithful_spectrum_norm or _env_flag_enabled(
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
    output_dtype = norm_reduction_dtype if source_faithful_spectrum_norm else jnp.float32
    return weighted_shells, weighted_per_image.astype(output_dtype)


def _make_relion_wavg_rectangle(
    image_shape,
    current_size,
    recon_window_indices,
):
    """Map exact BPref pixels into RELION's complete FFTW-ordered Wavg crop."""

    image_shape = tuple(int(value) for value in image_shape)
    current_size = int(current_size)
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
        current_size,
        include_dc=True,
        exact_radius=True,
    )
    recon_indices = np.asarray(recon_window_indices, dtype=np.int32).reshape(-1)
    if not np.array_equal(np.sort(recon_indices), exact_indices):
        raise ValueError(
            "RELION Wavg rectangle requires the complete exact-radius BPref window: "
            f"got {recon_indices.size} pixels, expected {exact_indices.size}"
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
        raise ValueError("RELION Wavg exact-radius position mapping is not bijective")
    return RelionWavgRectangle(
        centered_indices=rectangle_indices.astype(np.int32, copy=False),
        exact_positions=exact_positions,
        shell_indices=rectangle_shells,
    )


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

    shifted_power = (raw_shifted.real * raw_shifted.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    shifted_power = (shifted_power + raw_shifted.imag * raw_shifted.imag).astype(jnp.float32)
    image_power = jnp.einsum(
        "brt,btp->brp",
        posterior,
        shifted_power,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
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
    its covered shells replace both RECOVAR noise-stat components.  The
    exclusive boundary is intentional: the exact-radius reconstruction window
    only contains part of RELION's cutoff shell, so that shell must remain on
    the existing algebraic path until the full rectangular Wavg window is
    reproduced.
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


@partial(jax.jit, static_argnames=("batch_size",))
def _compute_norm_residual_per_image_from_flat_rows(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    flat_image_indices,
    *,
    batch_size: int,
):
    """Return norm-correction residuals per image from flattened active rows."""

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2_per_row = jnp.sum(a2_terms, axis=1)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance_half[None, :] * cross_terms.real
    xa_per_row = jnp.sum(xa_terms, axis=1)
    residual_per_row = (a2_per_row - 2.0 * xa_per_row).astype(jnp.float32)
    return jnp.zeros(int(batch_size), dtype=jnp.float32).at[flat_image_indices].add(residual_per_row)


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


def relion_x_half_bp_per_particle_launch_enabled() -> bool:
    """Return whether the diagnostic x-half path launches once per particle."""

    return _env_flag_enabled(_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV, default=False)


def relion_x_half_bp_fused_atomics_enabled() -> bool:
    """Return whether the diagnostic fused data/weight scatter is enabled."""

    return _env_flag_enabled(_RELION_X_HALF_BP_FUSED_ATOMICS_ENV, default=False)


def _scoped_bpref_diagnostic_flags(*, active: bool) -> dict[str, bool]:
    """Resolve process flags against an explicit device-capture boundary."""

    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    scope_active = bool(active or not device_signature_configured)
    return {
        "device_signature_configured": device_signature_configured,
        "sequential_translation_reduction": bool(
            scope_active and relion_x_half_sequential_translation_reduction_enabled()
        ),
        "per_particle_launches": bool(
            scope_active and relion_x_half_bp_per_particle_launch_enabled()
        ),
        "fused_atomics": bool(
            scope_active and relion_x_half_bp_fused_atomics_enabled()
        ),
        "high_precision_operand_bundle": bool(
            scope_active
            and _env_flag_enabled(
                "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
                default=False,
            )
        ),
    }


def _resolve_bpref_execution_modes(
    scoped_diagnostic_flags: dict[str, bool],
    *,
    device_signature_requested: bool,
) -> dict[str, bool]:
    """Separate requested diagnostic shadows from authoritative live modes."""

    shadow_only = bool(device_signature_requested)
    diagnostic_sequential = bool(scoped_diagnostic_flags["sequential_translation_reduction"])
    diagnostic_per_particle = bool(scoped_diagnostic_flags["per_particle_launches"])
    return {
        "shadow_only": shadow_only,
        "diagnostic_sequential_translation_reduction": diagnostic_sequential,
        "diagnostic_per_particle_launches": diagnostic_per_particle,
        "live_sequential_translation_reduction": bool(diagnostic_sequential and not shadow_only),
        "live_per_particle_launches": bool(diagnostic_per_particle and not shadow_only),
    }


def _require_bpref_shadow_exact(label: str, authoritative, shadow) -> None:
    """Fail closed unless a target-only shadow exactly matches live output."""

    authoritative_np = np.asarray(authoritative)
    shadow_np = np.asarray(shadow)
    if authoritative_np.shape != shadow_np.shape or authoritative_np.dtype != shadow_np.dtype:
        raise RuntimeError(
            f"BPref {label} shadow shape/dtype mismatch: "
            f"{authoritative_np.shape}/{authoritative_np.dtype} vs "
            f"{shadow_np.shape}/{shadow_np.dtype}"
        )
    if not np.array_equal(authoritative_np, shadow_np):
        mismatch_count = int(np.count_nonzero(authoritative_np != shadow_np))
        raise RuntimeError(
            f"BPref {label} shadow is not bitwise equal to the authoritative path "
            f"({mismatch_count}/{authoritative_np.size} elements differ)"
        )


def _require_bpref_reduction_shadow_agreement(
    authoritative_summed,
    authoritative_weights,
    shadow_summed,
    shadow_weights,
    *,
    rel_l1_bound: float = 1e-3,
    normalized_max_bound: float = 1e-3,
) -> dict[str, float]:
    """Gate the sequential-f32 diagnostic rows against ordinary live rows."""

    metrics: dict[str, float] = {}
    for label, authoritative, shadow in (
        ("data", authoritative_summed, shadow_summed),
        ("weight", authoritative_weights, shadow_weights),
    ):
        authoritative_np = np.asarray(authoritative)
        shadow_np = np.asarray(shadow)
        if authoritative_np.shape != shadow_np.shape:
            raise RuntimeError(
                f"BPref {label} reduction shadow shape mismatch: "
                f"{authoritative_np.shape} vs {shadow_np.shape}"
            )
        metric_dtype = np.complex128 if (
            np.iscomplexobj(authoritative_np) or np.iscomplexobj(shadow_np)
        ) else np.float64
        authoritative_metric = authoritative_np.astype(metric_dtype, copy=False)
        shadow_metric = shadow_np.astype(metric_dtype, copy=False)
        if not np.all(np.isfinite(authoritative_metric)) or not np.all(np.isfinite(shadow_metric)):
            raise RuntimeError(f"BPref {label} reduction shadow contains nonfinite values")
        difference = np.abs(authoritative_metric - shadow_metric)
        scale_l1 = max(float(np.sum(np.abs(authoritative_metric))), np.finfo(np.float64).tiny)
        scale_max = max(float(np.max(np.abs(authoritative_metric), initial=0.0)), np.finfo(np.float64).tiny)
        rel_l1 = float(np.sum(difference) / scale_l1)
        normalized_max = float(np.max(difference, initial=0.0) / scale_max)
        metrics[f"{label}_rel_l1"] = rel_l1
        metrics[f"{label}_normalized_max"] = normalized_max
        if rel_l1 > rel_l1_bound or normalized_max > normalized_max_bound:
            raise RuntimeError(
                f"BPref {label} reduction shadow exceeds the ordinary-path envelope: "
                f"rel_l1={rel_l1:.6g} (bound={rel_l1_bound:.6g}), "
                f"normalized_max={normalized_max:.6g} "
                f"(bound={normalized_max_bound:.6g})"
            )
    metrics["rel_l1_bound"] = float(rel_l1_bound)
    metrics["normalized_max_bound"] = float(normalized_max_bound)
    return metrics


def _require_bpref_device_soft_particle_arm(*, use_relion_x_half_mstep: bool) -> None:
    """Fail closed unless capture shares the explicit soft-particle causal arm.

    This arm is deliberately not called baseline production parity: RECOVAR
    first reduces translations into one row per orientation, whereas RELION may
    scatter orientation-by-translation hypotheses.  It is useful only as a
    controlled causal arm, with ordinary-vs-arm and plain-vs-instrumented
    controls recorded separately.
    """

    if not use_relion_x_half_mstep:
        raise RuntimeError("RECOVAR device signature requires the RELION x-half M-step")
    if not relion_x_half_bp_per_particle_launch_enabled():
        raise RuntimeError(
            "RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1"
        )
    if not relion_x_half_sequential_translation_reduction_enabled():
        raise RuntimeError(
            "RECOVAR device signature requires "
            "RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION=1"
        )
    if not relion_x_half_bp_fused_atomics_enabled():
        raise RuntimeError(
            "RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=1"
        )
    from recovar import cuda_backproject

    if not cuda_backproject.relion_x_half_bp_block_topology_requested():
        raise RuntimeError(
            "RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1"
        )


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

    diagnostic_fused_atomics = relion_x_half_bp_fused_atomics_enabled()
    # Fresh K=1 --firstiter_cc contributes exactly one winning hypothesis per
    # particle.  At this boundary the native RELION data/weight atomic stream
    # is reproduced exactly by the fused target; unlike later soft-posterior
    # iterations, no translation reduction changes the contributor stream.
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
            and not relion_x_half_bp_per_particle_launch_enabled()
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

    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    if values.shape[:2] != rotations.shape[:2] or ctf_values.shape[:2] != values.shape[:2]:
        raise ValueError("per-particle x-half diagnostic requires matching (particle, rotation) axes")
    if actual_counts.shape != (int(values.shape[0]),):
        raise ValueError(
            f"per-particle x-half actual_counts shape mismatch: {actual_counts.shape} vs {(int(values.shape[0]),)}"
        )
    logger.info(
        "RELION x-half diagnostic: one adjoint launch per particle "
        "(particles=%d, rotations min/median/max=%d/%d/%d, label=%s)",
        int(values.shape[0]),
        int(actual_counts.min()) if actual_counts.size else 0,
        int(np.median(actual_counts)) if actual_counts.size else 0,
        int(actual_counts.max()) if actual_counts.size else 0,
        log_label_prefix,
    )
    for particle_index, count in enumerate(actual_counts.tolist()):
        if count <= 0:
            continue
        particle_slice = (slice(particle_index, particle_index + 1), slice(0, int(count)))
        particle_values = values[particle_slice].reshape(int(count), values.shape[-1])
        particle_ctf_values = ctf_values[particle_slice].reshape(int(count), ctf_values.shape[-1])
        particle_rotations = rotations[particle_slice].reshape(int(count), 3, 3)
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

    if max_gather_bytes is None and max_prepare_images_per_microbatch is None and max_dense_mstep_bytes is None:
        return list(compact_buckets)
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
        if max_gather_bytes is not None or max_dense_mstep_bytes is not None:
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


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
    *,
    include_dense_score_fields: bool = True,
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
    separate_mstep_rotations = any(
        per_image_inputs["oversampled_mstep_rots"][int(image_idx)]
        is not per_image_inputs["oversampled_rots"][int(image_idx)]
        for image_idx in image_indices.tolist()
    )
    padded_mstep_rotations = (
        np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (batch, bucket_size, 3, 3),
        ).copy()
        if separate_mstep_rotations
        else padded_rotations
    )
    padded_log_prior = (
        np.full((batch, bucket_size), -1e30, dtype=np.float32)
        if include_dense_score_fields
        else None
    )
    padded_candidate_mask = (
        np.zeros((batch, bucket_size, n_fine_trans), dtype=bool)
        if include_dense_score_fields
        else None
    )
    padded_parent_map = (
        np.full((batch, bucket_size), -1, dtype=np.int32)
        if include_dense_score_fields
        else None
    )
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        padded_rotations[row, :cnt] = rots
        if separate_mstep_rotations:
            padded_mstep_rotations[row, :cnt] = per_image_inputs["oversampled_mstep_rots"][image_idx]
        if include_dense_score_fields:
            padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
            padded_candidate_mask[row, :cnt, :] = _candidate_mask_to_dense(per_image_inputs["candidate_mask"][image_idx])
            padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]

    return {
        "image_indices": image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "mstep_rotations": padded_mstep_rotations,
        "rotation_indices": padded_rotation_indices,
        "log_prior": padded_log_prior,
        "candidate_mask": padded_candidate_mask,
        "parent_map": padded_parent_map,
    }


def _compact_bucket_size_for_class(
    bucket,
    per_image_inputs,
    rotation_block_size_for_quantization,
) -> int:
    """Return the padded class-local bucket size for a fused K-class chunk."""

    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    if image_indices.size == 0:
        return 1
    max_count = max(
        int(per_image_inputs["oversampled_rots"][int(image_idx)].shape[0])
        for image_idx in image_indices.tolist()
    )
    return _exact_bucket_rotation_size(int(max_count), rotation_block_size_for_quantization)


def _build_k_class_bucket_arrays(
    bucket,
    per_image_inputs_by_class,
    n_fine_trans,
    *,
    compact_buckets: bool = False,
    include_dense_score_fields: bool = True,
    rotation_block_size_for_quantization=5000,
):
    """Build per-class padded arrays for fused sparse K-class pass 2.

    Default fused scoring keeps one rectangular bucket size shared by every
    class. The opt-in compact path keeps the same image chunk and joint
    class x pose normalization, but pads each class only to its class-local
    maximum rotation support inside that chunk.
    """

    if not compact_buckets:
        return [
            _build_bucket_arrays(
                bucket,
                per_image_inputs,
                n_fine_trans,
                include_dense_score_fields=include_dense_score_fields,
            )
            for per_image_inputs in per_image_inputs_by_class
        ]

    class_arrays = []
    for per_image_inputs in per_image_inputs_by_class:
        class_bucket = dict(bucket)
        class_bucket["bucket_size"] = _compact_bucket_size_for_class(
            bucket,
            per_image_inputs,
            rotation_block_size_for_quantization,
        )
        class_arrays.append(
            _build_bucket_arrays(
                class_bucket,
                per_image_inputs,
                n_fine_trans,
                include_dense_score_fields=include_dense_score_fields,
            )
        )
    return class_arrays


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
    """Historical algebraic Gaussian scorer used outside exact CUDA-f32 mode.

    In particular, this preserves the documented float64 scoring diagnostic.
    The exact RELION CUDA scorer below intentionally casts to XFLOAT and must
    therefore never be selected when ``use_float64_scoring=True``.
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
    """Reduce the 256 shared-memory lanes used by RELION CUDA REF3D fine diff2."""

    lanes = jnp.asarray(lanes, dtype=jnp.float32)
    if lanes.shape[-1] != _RELION_CUDA_FINE_REF3D_BLOCK_SIZE:
        raise ValueError(
            "RELION CUDA fine reduction needs exactly "
            f"{_RELION_CUDA_FINE_REF3D_BLOCK_SIZE} lanes, got {lanes.shape[-1]}"
        )
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    return lanes[..., 0]


def _relion_cuda_fine_tree_sum(values):
    """Reproduce RELION CUDA's 256-lane pixel-pass accumulation and tree."""

    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_values = int(values.shape[-1])
    if n_values == 0:
        return jnp.zeros(values.shape[:-1], dtype=jnp.float32)
    n_passes = (n_values + block_size - 1) // block_size
    padded_size = n_passes * block_size
    values = jnp.asarray(values, dtype=jnp.float32)
    values = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, padded_size - n_values)])
    pass_values = values.reshape(values.shape[:-1] + (n_passes, block_size))
    lanes = jnp.zeros(values.shape[:-1] + (block_size,), dtype=jnp.float32)
    for pass_index in range(n_passes):
        lanes = lanes + pass_values[..., pass_index, :]
    return _relion_cuda_fine_reduce_lanes(lanes)


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
):
    """Accumulate direct Gaussian diff2 without materializing ``(..., N)``.

    Pinned RELION CUDA's ``REF3D=true, DATA3D=false`` path uses
    ``D2F_BLOCK_SIZE_REF3D=256`` and ``D2F_CHUNK_REF3D=7``. The chunk controls
    translation batching; thread ``tid`` still accumulates pixels
    ``tid + pass * 256`` sequentially in XFLOAT (float32), followed by the
    shared-memory 128,64,...,1 reduction tree. Keeping only the 256 lanes per
    hypothesis avoids the much larger hypothesis-by-pixel temporary.
    """

    reference = jnp.asarray(reference, dtype=jnp.complex64)
    shifted_image = jnp.asarray(shifted_image, dtype=jnp.complex64)
    pixel_weight = jnp.asarray(pixel_weight, dtype=jnp.float32)
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
        return jnp.zeros(output_shape, dtype=jnp.float32)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )
    if _env_flag_enabled(_RELION_FINE_DIFF2_FUSED_FFI_ENV, default=False):
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
            return cuda_backproject.relion_fine_diff2_rectangular_f32(
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
                return cuda_backproject.relion_fine_diff2_pairs_f32(
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
                return cuda_backproject.relion_fine_diff2_rectangular_f32(
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
    lanes = jnp.zeros(output_shape + (block_size,), dtype=jnp.float32)

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
            * jnp.asarray(0.5, dtype=jnp.float32)
            * weight_pass
        )
        terms = jnp.where(valid_pixel, terms, jnp.asarray(0.0, dtype=jnp.float32))
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
    and reference norm over pixels ``tid + pass * 256`` in float32, then uses
    the same shared-memory tree as fine Gaussian ``diff2``.  RECOVAR stores
    the score window in centered compact order, so ``relion_full_to_compact``
    restores RELION's packed current-size FFTW pixel order before accumulation.
    """

    reference = jnp.asarray(reference, dtype=jnp.complex64)
    shifted_score = jnp.asarray(shifted_score, dtype=jnp.complex64)
    score_weight = jnp.asarray(score_weight, dtype=jnp.float32)
    half_weights = jnp.asarray(half_weights, dtype=jnp.float32)
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
        return jnp.full(numerator_shape, -jnp.inf, dtype=jnp.float32)

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
    numerator_lanes = jnp.zeros(numerator_shape + (block_size,), dtype=jnp.float32)
    norm_lanes = jnp.zeros(norm_shape + (block_size,), dtype=jnp.float32)

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
        zero = jnp.asarray(0.0, dtype=jnp.float32)
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
        jnp.maximum(norm, jnp.asarray(1e-30, dtype=jnp.float32))
    )


def _relion_cuda_fine_pixel_weights(corr_img_score, half_weights):
    """Form RELION XFLOAT pixel weights without a float64 intermediate."""

    return jnp.asarray(corr_img_score, dtype=jnp.float32) * jnp.asarray(
        half_weights, dtype=jnp.float32
    )


def _relion_cuda_corr_img_from_rfloat_ctf(inverse_noise, ctf_rfloat, scale=None):
    """Form XFLOAT ``corr_img`` after RELION's RFLOAT CTF square.

    The deployed mixed-precision build stores ``Minvsigma2`` and ``corr_img``
    as float32 (XFLOAT), but evaluates the CTF and ``CTF * CTF`` as float64
    (RFLOAT).  The compound multiplication promotes Minvsigma2 to float64 and
    casts the product back to float32 before the optional float32 scale square.
    """

    inverse_noise_rfloat = jnp.asarray(inverse_noise, dtype=jnp.float32).astype(
        jnp.float64
    )
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    ctf_squared_rfloat = jax.lax.optimization_barrier(ctf_rfloat * ctf_rfloat)
    corr_img = jax.lax.optimization_barrier(
        inverse_noise_rfloat * ctf_squared_rfloat
    ).astype(jnp.float32)
    if scale is not None:
        scale = jnp.asarray(scale, dtype=jnp.float32)
        scale_squared = jax.lax.optimization_barrier(scale * scale)
        corr_img = corr_img * scale_squared
    return corr_img


def _relion_cuda_pixel_correction_from_rfloat_ctf(scale, ctf_rfloat):
    """Form RELION's XFLOAT score-image correction from an RFLOAT CTF."""

    scale = jnp.asarray(scale, dtype=jnp.float32)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    pixel_correction = jax.lax.optimization_barrier(jnp.reciprocal(scale))
    corrected = jax.lax.optimization_barrier(
        pixel_correction.astype(jnp.float64) / ctf_rfloat
    ).astype(jnp.float32)
    return jnp.where(jnp.abs(ctf_rfloat) > 1e-8, corrected, pixel_correction)


_RELION_CUDA_POWERCLASS_BLOCK_SIZE = 128


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_xi2_half(
    processed_score_half,
    *,
    image_shape,
    current_size,
):
    """Reproduce the class-power high-resolution image tail used by fine diff2.

    RELION's CUDA ``powerClass`` kernel (``cuda_kernels/helper.cuh``) bins the
    unshifted, unnormalised ``Faux`` image in float32, reduces each contiguous
    128-pixel block with a shared-memory tree, and atomically accumulates bins
    at or above ``current_size / 2 + 1``. ``diff2_fine`` then adds half of that
    scalar to every fine-search hypothesis (``cuda_kernels/diff2.cuh``).

    RECOVAR stores the y axis centred and its FFT amplitudes are larger by the
    real-space pixel count. Convert both conventions before reproducing the
    float32 power and block reduction. The final cross-block accumulation uses
    ascending block order; RELION's atomic arrival order is not specified, so
    its last bit may vary between launches while the per-block arithmetic is
    fixed.
    """

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    processed_score_half = jnp.asarray(processed_score_half, dtype=jnp.complex64)
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    if current_size is None:
        current_size = image_width
    resolution_limit = int(current_size) // 2 + 1

    # RELION's packed rows are 0,+1,...,+Nyquist,-Nyquist+1,...,-1. RECOVAR's
    # rows are fftshift-centred, so move the first non-negative row to index 0.
    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=jnp.float32)

    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    # CUDA __float2int_rn(sqrtf(...)): nearest-even float32 conversion.
    shell = np.rint(np.sqrt(radius_squared.astype(np.float32))).astype(np.int32)
    valid = (
        (shell > 0)
        & (shell < half_width)
        & ~((columns == 0) & (signed_rows < 0))
        & (shell >= resolution_limit)
    ).reshape(-1)

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    power = jnp.where(jnp.asarray(valid)[None, :], power, jnp.asarray(0.0, dtype=jnp.float32))

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
        jnp.zeros((processed_score_half.shape[0],), dtype=jnp.float32),
    )
    return highres_xi2 * jnp.asarray(0.5, dtype=jnp.float32)


def _relion_powerclass_highres_xi2_half_to_norm_units(highres_xi2_half, image_shape):
    """Convert RELION's half-Xi2 FFT units to RECOVAR norm N^4 units."""

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    highres = jnp.asarray(highres_xi2_half, dtype=jnp.float32)
    highres = highres * jnp.asarray(2.0, dtype=jnp.float32)
    highres = jax.lax.optimization_barrier(highres)
    return highres * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
):
    """Return source-faithful powerClass high-shell power in RECOVAR N^4 units."""

    return _relion_powerclass_highres_xi2_half_to_norm_units(
        _relion_cuda_powerclass_highres_xi2_half(
            processed_score_half,
            image_shape=image_shape,
            current_size=current_size,
        ),
        image_shape,
    )


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
):
    """Reproduce the high-shell norm term from RELION's power spectrum.

    RELION's ``powerClass`` kernel produces two independently reduced values:
    a block-tree ``highres_Xi2`` scalar used by fine scoring, and an
    atomically binned shell spectrum.  Norm correction consumes the latter,
    summing its high shells sequentially in host RFLOAT.  These reductions are
    numerically distinct, so the fine-score scalar cannot be reused here.
    """

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    processed_score_half = jnp.asarray(processed_score_half, dtype=jnp.complex64)
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    resolution_limit = int(current_size) // 2 + 1
    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=jnp.float32)

    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    shell = np.rint(np.sqrt(radius_squared.astype(np.float32))).astype(np.int32)
    valid = (
        (shell > 0)
        & (shell < half_width)
        & ~((columns == 0) & (signed_rows < 0))
    ).reshape(-1)
    shell = np.where(valid, shell.reshape(-1), half_width).astype(np.int32)

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
        jnp.zeros((processed_score_half.shape[0],), dtype=jnp.float64),
    )
    return high_shell * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float64)


def _relion_cuda_fine_diff2_min(diff2, candidate_mask):
    """Return one finite float32 minimum per image over a raw diff2 tensor."""

    minimum = _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask)
    return jnp.where(
        jnp.isfinite(minimum),
        minimum,
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask):
    """Reduce one partition, retaining ``+inf`` for all-invalid images."""

    diff2 = jnp.asarray(diff2, dtype=jnp.float32)
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
        raw_diff2_device = jnp.asarray(raw_diff2, dtype=jnp.float32)
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
    return jnp.where(jnp.isfinite(common_min), common_min, jnp.asarray(0.0, dtype=jnp.float32))


def _relion_cuda_fine_log_evidence_offset(min_diff2):
    """Undo RELION's common-min score centering for absolute log evidence."""

    return -jnp.asarray(min_diff2, dtype=jnp.float32)


def _relion_cuda_fine_diff2_to_scores(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    *,
    min_diff2=None,
):
    """Apply RELION's float32 fine diff2-to-log-weight conversion order.

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

    diff2 = jnp.asarray(diff2, dtype=jnp.float32)
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    valid = candidate_mask & jnp.isfinite(diff2)
    if min_diff2 is None:
        local_min = _relion_cuda_fine_diff2_min(diff2, candidate_mask)
    else:
        local_min = jnp.asarray(min_diff2, dtype=jnp.float32)
    has_valid = jnp.any(valid, axis=tuple(range(1, diff2.ndim)))
    local_min = jnp.where(has_valid, local_min, jnp.asarray(0.0, dtype=jnp.float32))
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


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
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
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=jnp.float32)[:, None, None]
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


@jax.jit
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

    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
    shifted_corrected,  # (T, N) complex
    corr_img_score,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # scalar float32 powerClass tail already divided by two
):
    """Single-image cached positive-cost variant without priors or centering."""

    weights = _relion_cuda_fine_pixel_weights(corr_img_score, half_weights)
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, None, :],
        shifted_corrected[None, :, :],
        weights[None, None, :],
        relion_full_to_compact,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=jnp.float32)
    return diff2


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2_single_cached(
    shifted_corrected,  # (T, N) complex
    corr_img_score,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (R,) real
    translation_log_prior,  # (T,) real
    candidate_mask,  # (R, T) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # scalar or (1,) optional external common minimum
    highres_xi2_half=None,  # scalar float32 powerClass tail already divided by two
):
    """Single-image cached-projection variant that avoids a ``(1, R, N)`` copy."""

    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2[jnp.newaxis, :, :],
        rotation_log_prior[jnp.newaxis, :, None],
        translation_log_prior[jnp.newaxis, None, :],
        candidate_mask[jnp.newaxis, :, :],
        min_diff2=min_diff2,
    )[0]


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
def _score_pass2_bucket_relion_gpu_normalized_cc_single_cached(
    shifted_score,  # (T, N) complex, RELION-corrected image after shift
    score_weight,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (R, T) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """Single-image normalized-CC scorer for cached ``(R, N)`` projections."""

    scores = _relion_cuda_fine_normalized_cc_score(
        proj_half[:, None, :],
        shifted_score[None, :, :],
        score_weight[None, None, :],
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
    ) * jnp.asarray(half_weights, dtype=jnp.float32)[None, None, None, :]
    cross = -2.0 * jnp.sum(cross_products, axis=-1, dtype=jnp.float32)
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
    ) * jnp.asarray(half_weights, dtype=jnp.float32)[None, None, :]
    cross = -2.0 * jnp.sum(cross_products, axis=-1, dtype=jnp.float32)
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


@jax.jit
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
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=jnp.float32)[:, None]
    return diff2


@jax.jit
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
):
    """RELION GPU-style Gaussian scoring for compact pass-2 pairs."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    pair_rotation_log_prior = jnp.asarray(pair_rotation_log_prior, dtype=jnp.float32)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
    trans_prior = jnp.asarray(
        translation_log_prior[row, safe_translation_idx], dtype=jnp.float32
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
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        pair_rotation_log_prior,
        trans_prior,
        pair_mask,
        min_diff2=min_diff2,
    )


@jax.jit
def _score_pass2_pairs_relion_gpu_normalized_cc(
    shifted_score,  # (B, T, N) complex, RELION-corrected image after shift
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """RELION iter-1 normalized-CC scoring for compact pass-2 pairs."""

    batch = shifted_score.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)

    shifted_pair = shifted_score[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    scores = _relion_cuda_fine_normalized_cc_score(
        proj_pair,
        shifted_pair,
        score_weight[:, None, :],
        half_weights,
        relion_full_to_compact,
    )
    scores = jnp.where(pair_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


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
    ) * jnp.asarray(half_weights, dtype=jnp.float32)[None, None, :]
    cross = -2.0 * jnp.sum(cross_products, axis=-1, dtype=jnp.float32)
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
def _normalize_pass2_pairs_score_only(pair_scores, pair_mask):
    """Compute compact pair score stats without materializing dense posteriors."""
    pair_scores = jnp.where(pair_mask & jnp.isfinite(pair_scores), pair_scores, -jnp.inf)
    best_log_score = jnp.max(pair_scores, axis=1)
    has_finite_score = jnp.isfinite(best_log_score)
    safe_best_log_score = jnp.where(has_finite_score, best_log_score, 0.0)
    shifted = jnp.where(has_finite_score[:, None], pair_scores - safe_best_log_score[:, None], -jnp.inf)
    exp_terms = jnp.exp(shifted.astype(jnp.float64))
    exp_terms = jnp.where(jnp.isfinite(exp_terms), exp_terms, 0.0)
    sum_exp = jnp.sum(exp_terms, axis=1)
    has_mass = has_finite_score & (sum_exp > 0) & jnp.isfinite(sum_exp)
    safe_sum_exp = jnp.where(has_mass, sum_exp, 1.0)
    log_Z = jnp.where(has_mass, safe_best_log_score + jnp.log(safe_sum_exp), 0.0)
    best_argmax = jnp.where(has_mass, jnp.argmax(pair_scores, axis=1), 0)
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


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_sums_dense(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
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
    summed = compute_local_weighted_sums(dense_probs, shifted_recon_split)
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
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
):
    impl = (
        _compact_pair_weighted_rotation_sums_pair_sparse
        if _compact_pair_pair_sparse_mstep_enabled_for_pass(allow_pair_sparse=allow_pair_sparse)
        else _compact_pair_weighted_rotation_sums_dense
    )
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
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
        if _env_flag_enabled(_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS_ENV, default=True)
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


def _flat_image_indices_for_rotation_rows(batch: int, n_rotation_rows: int):
    """Return image ids for flattened ``(batch, rotation_row)`` arrays."""

    return jnp.broadcast_to(
        jnp.arange(int(batch), dtype=jnp.int32)[:, None],
        (int(batch), int(n_rotation_rows)),
    )


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


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _select_active_noise_rows(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    *,
    n_rotation_rows: int,
):
    """Gather compact active noise rows and their image ids in one launch."""

    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_mask = jnp.asarray(active_mask)
    row_mask = active_mask.astype(jnp.asarray(summed_masked_noise).real.dtype)
    active_image_indices = jnp.where(active_mask.astype(jnp.int32) != 0, active_indices // int(n_rotation_rows), 0)

    def gather(values):
        flat_values = values.reshape((values.shape[0] * values.shape[1], values.shape[-1]))
        gathered = flat_values[active_indices]
        return gathered * row_mask[:, None]

    return (
        gather(proj_for_noise),
        gather(proj_abs2_for_noise),
        gather(summed_masked_noise),
        gather(ctf_probs_for_noise),
        active_image_indices,
    )


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
    if n_rows <= 0:
        return jnp.zeros(int(shell_count), dtype=jnp.float32), jnp.zeros(int(batch_size), dtype=jnp.float32)

    use_residual_terms = _env_flag_enabled(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
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

    noise_total = jnp.zeros(int(shell_count), dtype=jnp.float32)
    norm_total = jnp.zeros(int(batch_size), dtype=jnp.float32)
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


@partial(jax.jit, static_argnames=("adaptive_fraction",))
def _relion_f32_fine_reconstruction_probs(scores, *, adaptive_fraction: float):
    """Build fine M-step probabilities with RELION GPU float32 arithmetic.

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
    shifted = jnp.where(finite, flat_scores - safe_best[:, None] + jnp.float32(50.0), -jnp.inf)
    raw_weights = jnp.where(shifted < jnp.float32(-88.0), jnp.float32(0.0), jnp.exp(shifted))
    raw_weights = jnp.where(finite & jnp.isfinite(raw_weights), raw_weights, jnp.float32(0.0))

    sorted_weights = jnp.sort(raw_weights, axis=1)
    cumulative = jnp.cumsum(sorted_weights, axis=1, dtype=jnp.float32)
    sum_weight = cumulative[:, -1]
    has_mass = has_finite & jnp.isfinite(sum_weight) & (sum_weight > jnp.float32(0.0))
    tail_target = _relion_cuda_f32_tail_target(sum_weight, adaptive_fraction)
    threshold_idx = jax.vmap(lambda row, target: jnp.searchsorted(row, target, side="right"))(
        cumulative,
        tail_target,
    )
    threshold_idx = jnp.minimum(threshold_idx, cumulative.shape[1] - 1)
    threshold = sorted_weights[jnp.arange(flat_scores.shape[0]), threshold_idx]
    mask_flat = has_mass[:, None] & finite & (raw_weights >= threshold[:, None])
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    reconstruction_probs_flat = jnp.where(mask_flat, raw_weights / safe_sum_weight[:, None], jnp.float32(0.0))
    n_significant = jnp.sum(mask_flat, axis=1).astype(jnp.int32)
    output_shape = scores_f32.shape
    return (
        reconstruction_probs_flat.reshape(output_shape),
        mask_flat.reshape(output_shape),
        n_significant,
        sum_weight,
        threshold,
    )


def relion_x_half_f32_fine_posterior_enabled() -> bool:
    """Return whether the opt-in RELION float32 fine posterior is enabled."""

    return _env_flag_enabled(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, default=False)


def _relion_pass2_reconstruction_probs_for_mstep(
    scores,
    probs,
    *,
    adaptive_fraction: float,
    use_relion_x_half_mstep: bool,
    winner_take_all: bool = False,
    return_diagnostics: bool = False,
):
    """Select the default or diagnostic fine-posterior reconstruction path."""

    if (
        use_relion_x_half_mstep
        and not winner_take_all
        and relion_x_half_f32_fine_posterior_enabled()
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
    """

    value = mode_override
    if value is None:
        value = os.environ.get(_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV)
    if value is None or not value.strip():
        return "per_class" if use_relion_x_half_mstep else "none"
    mode = value.strip().lower()
    if mode in _SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_JOINT_MODES:
        return "joint"
    if mode in {"1", "true", "yes", "on", "class", "per_class", "per-class"}:
        return "per_class"
    if mode in {"0", "false", "no", "off", "none"}:
        return "per_class" if use_relion_x_half_mstep else "none"
    raise ValueError(
        f"{_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV} must be one of "
        "0/1/per_class/joint"
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


def _bpref_membership_dump_requested():
    dump_dir = os.environ.get(_BPREF_MEMBERSHIP_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return False
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get(_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV)
    if target_iteration and context_iteration != int(target_iteration):
        return False
    target_half = os.environ.get(_BPREF_MEMBERSHIP_DUMP_HALF_ENV)
    if target_half:
        if int(target_half) not in {1, 2}:
            raise ValueError(f"{_BPREF_MEMBERSHIP_DUMP_HALF_ENV} must be 1 or 2")
        if context_half != int(target_half):
            return False
    return True


def _maybe_dump_k1_bpref_rotation_mass(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    actual_counts,
    rotations,
    rotation_indices,
    candidate_translation_count,
    posterior_rotation_mass,
    reconstruction_rotation_mass,
    significant_translation_count,
    reconstruction_sum_weight,
    reconstruction_threshold,
):
    """Dump the sufficient per-rotation inputs to the BPref denominator."""

    if not _bpref_membership_dump_requested():
        return
    dump_dir = os.environ[_BPREF_MEMBERSHIP_DUMP_DIR_ENV].strip()
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    counts = np.asarray(actual_counts, dtype=np.int64)
    rotations_np = np.asarray(rotations, dtype=np.float32)
    rotation_indices_np = np.asarray(rotation_indices, dtype=np.int64)
    candidate_count_np = np.asarray(candidate_translation_count, dtype=np.int32)
    posterior_mass_np = np.asarray(posterior_rotation_mass)
    reconstruction_mass_np = np.asarray(reconstruction_rotation_mass)
    significant_count_np = np.asarray(significant_translation_count, dtype=np.int32)
    sum_weight_np = np.asarray(reconstruction_sum_weight)
    threshold_np = np.asarray(reconstruction_threshold)

    batch = local_indices.size
    topology = posterior_mass_np.shape
    if counts.shape != (batch,) or len(topology) != 2 or topology[0] != batch:
        raise ValueError("BPref rotation-mass topology mismatch")
    if (
        reconstruction_mass_np.shape != topology
        or candidate_count_np.shape != topology
        or significant_count_np.shape != topology
    ):
        raise ValueError("BPref rotation-mass arrays have inconsistent topology")
    if rotations_np.shape != (*topology, 3, 3):
        raise ValueError("BPref rotation-mass rotation topology mismatch")
    if rotation_indices_np.ndim == 1:
        rotation_indices_np = np.broadcast_to(rotation_indices_np[None, :], topology)
    if rotation_indices_np.shape != topology:
        raise ValueError("BPref rotation-mass index topology mismatch")
    if np.any(counts < 0) or np.any(counts > topology[1]):
        raise ValueError("BPref rotation counts are outside the padded rotation axis")
    if np.any(candidate_count_np < 0) or np.any(significant_count_np < 0):
        raise ValueError("BPref translation counts are negative")
    if np.any(significant_count_np > candidate_count_np):
        raise ValueError("BPref significant translations exceed candidate translations")
    if np.any(posterior_mass_np < 0) or np.any(reconstruction_mass_np < 0):
        raise ValueError("BPref rotation masses are negative")
    if np.any(reconstruction_mass_np > posterior_mass_np + np.finfo(np.float32).eps):
        raise ValueError("BPref reconstruction mass exceeds posterior mass")
    padded = np.arange(topology[1])[None, :] >= counts[:, None]
    if (
        np.any(candidate_count_np[padded])
        or np.any(significant_count_np[padded])
        or np.any(posterior_mass_np[padded])
        or np.any(reconstruction_mass_np[padded])
    ):
        raise ValueError("BPref padded rotations carry membership or mass")
    if np.max(candidate_count_np, initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("BPref candidate translation count exceeds uint16")

    global _bpref_membership_dump_counter
    dump_index = _bpref_membership_dump_counter
    _bpref_membership_dump_counter += 1
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / (
        f"bpref_membership_it{context_iteration:03d}_h{context_half}"
        f"_dump{dump_index:06d}_cs{int(current_size):03d}.npz"
    )
    np.savez(
        output,
        schema=np.asarray("recovar-bpref-rotation-mass-v2"),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        current_size=np.int32(current_size),
        local_indices=local_indices,
        original_indices=original_indices,
        stack_indices_1based=original_indices + 1,
        actual_counts=counts,
        rotations=rotations,
        rotation_indices=rotation_indices_np,
        candidate_translation_count=candidate_count_np.astype(np.uint16),
        posterior_rotation_mass=posterior_mass_np,
        reconstruction_rotation_mass=reconstruction_mass_np,
        significant_translation_count=significant_count_np.astype(np.uint16),
        reconstruction_sum_weight=sum_weight_np,
        reconstruction_threshold=threshold_np,
    )


def _maybe_dump_k1_bpref_membership(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    actual_counts,
    rotations,
    rotation_indices,
    fine_translations,
    candidate_mask,
    posterior_probs,
    reconstruction_probs,
    reconstruction_mask,
    reconstruction_sum_weight,
    reconstruction_threshold,
):
    """Collapse a rectangular fine posterior to sufficient rotation masses."""

    if not _bpref_membership_dump_requested():
        return
    candidate_mask_np = np.asarray(candidate_mask, dtype=bool)
    posterior_np = np.asarray(posterior_probs)
    reconstruction_np = np.asarray(reconstruction_probs)
    reconstruction_mask_np = np.asarray(reconstruction_mask, dtype=bool)
    if posterior_np.ndim != 3:
        raise ValueError("BPref membership posterior topology mismatch")
    if reconstruction_np.shape != posterior_np.shape:
        raise ValueError("BPref membership reconstruction-posterior shape mismatch")
    if candidate_mask_np.shape != posterior_np.shape:
        raise ValueError("BPref membership candidate-mask shape mismatch")
    if reconstruction_mask_np.shape != posterior_np.shape:
        raise ValueError("BPref membership reconstruction-mask shape mismatch")
    if not np.array_equal(reconstruction_mask_np, reconstruction_np > 0):
        raise ValueError("BPref membership mask does not equal positive reconstruction posterior")
    _maybe_dump_k1_bpref_rotation_mass(
        experiment_dataset=experiment_dataset,
        image_indices=image_indices,
        current_size=current_size,
        actual_counts=actual_counts,
        rotations=rotations,
        rotation_indices=rotation_indices,
        candidate_translation_count=np.sum(candidate_mask_np, axis=-1, dtype=np.int32),
        posterior_rotation_mass=np.sum(posterior_np, axis=-1),
        reconstruction_rotation_mass=np.sum(reconstruction_np, axis=-1),
        significant_translation_count=np.sum(reconstruction_mask_np, axis=-1, dtype=np.int32),
        reconstruction_sum_weight=reconstruction_sum_weight,
        reconstruction_threshold=reconstruction_threshold,
    )


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
    direct_score_input=None,
    direct_preprocessed_score_input=None,
    direct_pixel_correction=None,
    direct_inverse_noise_score=None,
    direct_ctf_rfloat_score=None,
    direct_preprocess_normalization_factors=None,
    direct_integer_pre_shifts=None,
    direct_batch_image_corrections=None,
    direct_batch_scale_corrections=None,
    shifted_recon_split=None,
    ctf2_over_nv_recon=None,
    recon_window_indices=None,
    reconstruction_mask=None,
    reconstruction_probs=None,
    reconstruction_n_significant=None,
    relion_highres_xi2_half=None,
    relion_min_diff2=None,
):
    """Env-gated sparse pass-2 dump for RELION operand parity debugging."""
    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return 0
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return 0
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return 0
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return 0
    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)

    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return 0

    requested_rotation_rows = parse_env_int_set(_PASS2_DUMP_ROTATION_ROWS_ENV)
    if requested_rotation_rows:
        rotation_rows = np.asarray(sorted(requested_rotation_rows), dtype=np.int64)
        os.makedirs(dump_dir, exist_ok=True)
        dump_count = 0
        for row in wanted_rows:
            image_idx = int(local_indices[row])
            original_idx = int(original_indices[row])
            cnt = int(per_image_inputs["oversampled_rots"][image_idx].shape[0])
            if np.any(rotation_rows < 0) or np.any(rotation_rows >= cnt):
                raise ValueError(
                    f"{_PASS2_DUMP_ROTATION_ROWS_ENV} contains a row outside "
                    f"[0, {cnt}) for original particle {original_idx}",
                )
            selected_scores = np.asarray(
                jnp.take(scores[row], jnp.asarray(rotation_rows), axis=0),
                dtype=np.float64,
            )
            selected_rotation_prior = np.asarray(
                jnp.take(rotation_log_prior[row], jnp.asarray(rotation_rows), axis=0),
                dtype=np.float64,
            )
            translation_prior = np.asarray(translation_log_prior[row], dtype=np.float64)
            pre_prior = (
                selected_scores
                - selected_rotation_prior[:, None]
                - translation_prior[None, :]
            )
            full_mask = jnp.asarray(candidate_mask[row], dtype=bool)
            full_scores = jnp.asarray(scores[row])
            full_probs = jnp.asarray(probs[row])
            masked_scores = jnp.where(full_mask, full_scores, -jnp.inf)
            masked_probs = jnp.where(full_mask, full_probs, 0.0)
            score_argmax_flat = int(np.asarray(jnp.argmax(masked_scores)))
            prob_argmax_flat = int(np.asarray(jnp.argmax(masked_probs)))
            score_argmax_rotation, score_argmax_translation = divmod(
                score_argmax_flat, int(n_fine_trans)
            )
            prob_argmax_rotation, prob_argmax_translation = divmod(
                prob_argmax_flat, int(n_fine_trans)
            )
            selected_reconstruction_fields = {}
            if reconstruction_mask is not None:
                selected_reconstruction_fields["reconstruction_mask"] = np.asarray(
                    jnp.take(reconstruction_mask[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=bool,
                )
            if reconstruction_probs is not None:
                selected_reconstruction_fields["reconstruction_probs"] = np.asarray(
                    jnp.take(reconstruction_probs[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=np.float64,
                )
            if reconstruction_n_significant is not None:
                recon_n_sig = np.asarray(reconstruction_n_significant, dtype=np.int64)
                selected_reconstruction_fields["reconstruction_n_significant"] = (
                    recon_n_sig[row] if recon_n_sig.ndim else recon_n_sig
                )
            if relion_highres_xi2_half is not None:
                selected_reconstruction_fields["relion_highres_xi2_half"] = np.float32(
                    np.asarray(relion_highres_xi2_half, dtype=np.float32)[row]
                )
            if relion_min_diff2 is not None:
                selected_reconstruction_fields["relion_min_diff2"] = np.float32(
                    np.asarray(relion_min_diff2, dtype=np.float32)[row]
                )
            out_path = os.path.join(
                dump_dir,
                f"pass2_orig{original_idx:06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
            )
            np.savez_compressed(
                out_path,
                schema=np.asarray("recovar.em.k1_pass2_selected_rotations.v1"),
                iteration=np.int64(context_iteration),
                half=np.int64(context_half),
                original_index=np.int64(original_idx),
                local_index=np.int64(image_idx),
                current_size=np.int64(-1 if current_size is None else int(current_size)),
                n_fine_trans=np.int64(n_fine_trans),
                rotation_rows_global=rotation_rows,
                fine_translations=np.asarray(fine_translations, dtype=np.float32),
                rotations=np.asarray(
                    per_image_inputs["oversampled_rots"][image_idx],
                    dtype=np.float32,
                )[rotation_rows],
                oversampled_rot_indices=np.asarray(
                    per_image_inputs["oversampled_rot_indices"][image_idx],
                    dtype=np.int64,
                )[rotation_rows],
                parent_map=np.asarray(
                    per_image_inputs["parent_map"][image_idx],
                    dtype=np.int32,
                )[rotation_rows],
                candidate_mask=np.asarray(
                    jnp.take(candidate_mask[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=bool,
                ),
                candidate_rotation_count=np.int64(cnt),
                candidate_mask_total_count=np.int64(
                    np.asarray(jnp.count_nonzero(full_mask))
                ),
                score_max=np.float64(np.asarray(jnp.max(masked_scores))),
                score_argmax_rotation=np.int64(score_argmax_rotation),
                score_argmax_translation=np.int64(score_argmax_translation),
                posterior_sum=np.float64(np.asarray(jnp.sum(masked_probs))),
                posterior_max=np.float64(np.asarray(jnp.max(masked_probs))),
                posterior_argmax_rotation=np.int64(prob_argmax_rotation),
                posterior_argmax_translation=np.int64(prob_argmax_translation),
                scores_with_prior=selected_scores,
                scores_pre_prior=pre_prior,
                probs=np.asarray(
                    jnp.take(probs[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=np.float64,
                ),
                rotation_log_prior=selected_rotation_prior,
                translation_log_prior=translation_prior,
                shifted_corrected=(
                    np.asarray(shifted_corrected_score_split[row])
                    if shifted_corrected_score_split is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_score_input=(
                    np.asarray(direct_score_input[row])
                    if direct_score_input is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_preprocessed_score_input=(
                    np.asarray(direct_preprocessed_score_input[row])
                    if direct_preprocessed_score_input is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_pixel_correction=(
                    np.asarray(direct_pixel_correction[row])
                    if direct_pixel_correction is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                direct_inverse_noise_score=(
                    np.asarray(direct_inverse_noise_score, dtype=np.float32)
                    if direct_inverse_noise_score is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                direct_ctf_rfloat_score=(
                    np.asarray(direct_ctf_rfloat_score[row], dtype=np.float64)
                    if direct_ctf_rfloat_score is not None
                    else np.empty((0,), dtype=np.float64)
                ),
                relion_preprocess_normalization_factor=(
                    np.float32(np.asarray(direct_preprocess_normalization_factors)[row])
                    if direct_preprocess_normalization_factors is not None
                    else np.float32(np.nan)
                ),
                relion_integer_pre_shift=(
                    np.asarray(direct_integer_pre_shifts, dtype=np.int32)[row]
                    if direct_integer_pre_shifts is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                batch_image_correction=(
                    np.float32(np.asarray(direct_batch_image_corrections)[row])
                    if direct_batch_image_corrections is not None
                    else np.float32(np.nan)
                ),
                batch_scale_correction=(
                    np.float32(np.asarray(direct_batch_scale_corrections)[row])
                    if direct_batch_scale_corrections is not None
                    else np.float32(np.nan)
                ),
                ctf2_over_nv_score=np.asarray(ctf2_over_nv_score[row], dtype=np.float64),
                shifted_recon=(
                    np.asarray(shifted_recon_split[row])
                    if shifted_recon_split is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                ctf2_over_nv_recon=(
                    np.asarray(ctf2_over_nv_recon[row], dtype=np.float64)
                    if ctf2_over_nv_recon is not None
                    else np.empty((0,), dtype=np.float64)
                ),
                proj_half=np.asarray(
                    jnp.take(proj_half[row], jnp.asarray(rotation_rows), axis=0),
                ),
                half_weights=np.asarray(half_weights_used, dtype=np.float64),
                window_indices=(
                    np.asarray(window_indices, dtype=np.int32)
                    if window_indices is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                recon_window_indices=(
                    np.asarray(recon_window_indices, dtype=np.int32)
                    if recon_window_indices is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                **selected_reconstruction_fields,
            )
            dump_count += 1
        return dump_count

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    rot_prior_np = np.asarray(rotation_log_prior, dtype=np.float64)
    trans_prior_np = np.asarray(translation_log_prior, dtype=np.float64)
    mask_np = np.asarray(candidate_mask, dtype=bool)
    recon_mask_np = None if reconstruction_mask is None else np.asarray(reconstruction_mask, dtype=bool)
    recon_probs_np = None if reconstruction_probs is None else np.asarray(reconstruction_probs, dtype=np.float64)
    recon_n_sig_np = (
        None if reconstruction_n_significant is None else np.asarray(reconstruction_n_significant, dtype=np.int64)
    )
    ctf2_np = np.asarray(ctf2_over_nv_score, dtype=np.float64)
    proj_np = np.asarray(proj_half)
    shifted_corrected_np = (
        None if shifted_corrected_score_split is None else np.asarray(shifted_corrected_score_split)
    )
    direct_score_input_np = (
        None if direct_score_input is None else np.asarray(direct_score_input)
    )
    direct_preprocessed_score_input_np = (
        None
        if direct_preprocessed_score_input is None
        else np.asarray(direct_preprocessed_score_input)
    )
    direct_pixel_correction_np = (
        None if direct_pixel_correction is None else np.asarray(direct_pixel_correction)
    )
    direct_inverse_noise_score_np = (
        None
        if direct_inverse_noise_score is None
        else np.asarray(direct_inverse_noise_score, dtype=np.float32)
    )
    direct_ctf_rfloat_score_np = (
        None
        if direct_ctf_rfloat_score is None
        else np.asarray(direct_ctf_rfloat_score, dtype=np.float64)
    )
    shifted_recon_np = None if shifted_recon_split is None else np.asarray(shifted_recon_split)
    ctf2_recon_np = None if ctf2_over_nv_recon is None else np.asarray(ctf2_over_nv_recon, dtype=np.float64)
    recon_window_indices_np = None if recon_window_indices is None else np.asarray(recon_window_indices, dtype=np.int32)
    highres_np = (
        None if relion_highres_xi2_half is None else np.asarray(relion_highres_xi2_half, dtype=np.float32)
    )
    min_diff2_np = None if relion_min_diff2 is None else np.asarray(relion_min_diff2, dtype=np.float32)

    dump_count = 0
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
        reconstruction_fields = {}
        if recon_mask_np is not None:
            reconstruction_fields["reconstruction_mask"] = recon_mask_np[row, :cnt, :]
        if recon_probs_np is not None:
            reconstruction_fields["reconstruction_probs"] = recon_probs_np[row, :cnt, :]
        if recon_n_sig_np is not None:
            reconstruction_fields["reconstruction_n_significant"] = (
                recon_n_sig_np[row] if recon_n_sig_np.ndim else recon_n_sig_np
            )
        if highres_np is not None:
            reconstruction_fields["relion_highres_xi2_half"] = np.float32(highres_np[row])
        if min_diff2_np is not None:
            reconstruction_fields["relion_min_diff2"] = np.float32(min_diff2_np[row])
        np.savez_compressed(
            out_path,
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
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
            direct_score_input=(
                direct_score_input_np[row]
                if direct_score_input_np is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            direct_preprocessed_score_input=(
                direct_preprocessed_score_input_np[row]
                if direct_preprocessed_score_input_np is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            direct_pixel_correction=(
                direct_pixel_correction_np[row]
                if direct_pixel_correction_np is not None
                else np.empty((0,), dtype=np.float32)
            ),
            direct_inverse_noise_score=(
                direct_inverse_noise_score_np
                if direct_inverse_noise_score_np is not None
                else np.empty((0,), dtype=np.float32)
            ),
            direct_ctf_rfloat_score=(
                direct_ctf_rfloat_score_np[row]
                if direct_ctf_rfloat_score_np is not None
                else np.empty((0,), dtype=np.float64)
            ),
            relion_preprocess_normalization_factor=(
                np.float32(np.asarray(direct_preprocess_normalization_factors)[row])
                if direct_preprocess_normalization_factors is not None
                else np.float32(np.nan)
            ),
            relion_integer_pre_shift=(
                np.asarray(direct_integer_pre_shifts, dtype=np.int32)[row]
                if direct_integer_pre_shifts is not None
                else np.empty((0,), dtype=np.int32)
            ),
            batch_image_correction=(
                np.float32(np.asarray(direct_batch_image_corrections)[row])
                if direct_batch_image_corrections is not None
                else np.float32(np.nan)
            ),
            batch_scale_correction=(
                np.float32(np.asarray(direct_batch_scale_corrections)[row])
                if direct_batch_scale_corrections is not None
                else np.float32(np.nan)
            ),
            ctf2_over_nv_score=ctf2_np[row],
            shifted_recon=(
                shifted_recon_np[row] if shifted_recon_np is not None else np.empty((0,), dtype=np.complex64)
            ),
            ctf2_over_nv_recon=(
                ctf2_recon_np[row] if ctf2_recon_np is not None else np.empty((0,), dtype=np.float64)
            ),
            proj_half=proj_np[row, :cnt, :],
            half_weights=np.asarray(half_weights_used, dtype=np.float64),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32) if window_indices is not None else np.empty((0,), dtype=np.int32)
            ),
            recon_window_indices=(
                recon_window_indices_np if recon_window_indices_np is not None else np.empty((0,), dtype=np.int32)
            ),
            **reconstruction_fields,
        )
        dump_count += 1
    return dump_count


def _maybe_dump_norm_residual_inputs(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs,
    ctf2_over_nv_recon,
    posterior_probs,
    noise_variance_for_noise,
    block_norm_residual,
    processed_score_half_for_noise,
    shell_indices_half,
    support_mass,
    relion_norm_high_shell,
    weighted_img_per_image,
    relion_score_translation_angles,
    recon_window_indices,
    score_window_indices,
    image_shape,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    bucket_group_ids,
):
    """Capture norm and group-scale AA inputs before any global reduction.

    This diagnostic is deliberately separate from the ordinary pass-2 dump:
    that dump records score-window projections, while the normalization update
    consumes the reconstruction/noise window and its squared projections.
    """

    if not _env_flag_enabled("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", default=False):
        return 0
    dump_dir = os.environ.get(_PASS2_DUMP_DIR_ENV)
    if not dump_dir:
        raise ValueError(
            "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS requires RECOVAR_PASS2_DUMP_DIR"
        )
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    context_iteration = int(_bpref_contribution_context["iteration"])
    if target_iteration and context_iteration != int(target_iteration):
        return 0
    target_rows = _pass2_dump_target_rows(
        experiment_dataset=experiment_dataset,
        image_indices=image_indices,
        current_size=current_size,
    )
    if target_rows.size == 0:
        return 0

    selected = jnp.asarray(target_rows, dtype=jnp.int32)
    raw_translated_recon = None
    raw_translated_wavg = None
    if relion_score_translation_angles is not None:
        from recovar import cuda_backproject

        recon_indices_jax = jnp.asarray(recon_window_indices, dtype=jnp.int32)
        translation_angles_jax = jnp.asarray(
            relion_score_translation_angles,
            dtype=jnp.float32,
        )
        raw_translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, recon_indices_jax],
            translation_angles_jax,
            recon_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
        score_indices_jax = jnp.asarray(score_window_indices, dtype=jnp.int32)
        raw_translated_wavg = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, score_indices_jax],
            translation_angles_jax,
            score_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
    selected_proj_abs2 = jnp.asarray(proj_abs2_for_noise)[selected]
    selected_ctf_probs = jnp.asarray(ctf_probs)[selected]
    selected_posterior_probs = jnp.asarray(posterior_probs)[selected]
    selected_ctf2_over_nv = jnp.asarray(ctf2_over_nv_recon)[selected]
    selected_scale = jnp.asarray(bucket_scale_for_stats)[selected]
    scale_mask = jnp.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    selected_noise = jnp.asarray(noise_variance_for_noise)
    ctf_has_mass = (selected_ctf_probs != 0.0) & scale_mask[None, None, :]
    ctf_probs_raw = jnp.where(
        ctf_has_mass,
        selected_ctf_probs * selected_noise[None, None, :],
        0.0,
    )
    aa_terms_before_scale = jnp.where(
        ctf_has_mass,
        selected_proj_abs2 * ctf_probs_raw,
        0.0,
    )
    safe_scale = jnp.maximum(
        selected_scale.astype(selected_proj_abs2.real.dtype),
        1e-30,
    )
    aa_terms = aa_terms_before_scale / (safe_scale[:, None, None] ** 2)
    aa_per_pixel = jnp.sum(aa_terms, axis=1)
    aa_per_image = jnp.sum(aa_terms, axis=(1, 2)).astype(jnp.float32)

    staged = jax.block_until_ready(
        (
            jnp.asarray(proj_for_noise)[selected],
            selected_proj_abs2,
            jnp.asarray(summed_masked_noise)[selected],
            selected_ctf_probs,
            selected_ctf2_over_nv,
            selected_posterior_probs,
            jnp.asarray(block_norm_residual)[selected],
            jnp.asarray(processed_score_half_for_noise)[selected],
            jnp.asarray(support_mass)[selected],
            jnp.asarray(weighted_img_per_image)[selected],
            selected_scale,
            ctf_has_mass,
            ctf_probs_raw,
            aa_terms_before_scale,
            aa_terms,
            aa_per_pixel,
            aa_per_image,
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_recon is None
                else raw_translated_recon
            ),
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_wavg is None
                else raw_translated_wavg
            ),
        )
    )
    (
        proj_np,
        proj_abs2_np,
        summed_np,
        ctf_probs_np,
        ctf2_over_nv_np,
        posterior_probs_np,
        residual_np,
        processed_image_np,
        support_mass_np,
        weighted_image_power_np,
        scale_np,
        ctf_has_mass_np,
        ctf_probs_raw_np,
        aa_terms_before_scale_np,
        aa_terms_np,
        aa_per_pixel_np,
        aa_per_image_np,
        raw_translated_recon_np,
        raw_translated_wavg_np,
    ) = (np.asarray(value) for value in staged)
    noise_np = np.asarray(jax.block_until_ready(noise_variance_for_noise))
    shell_indices_np = np.asarray(jax.block_until_ready(shell_indices_half), dtype=np.int32)
    scale_shell_indices_np = np.asarray(
        jax.block_until_ready(scale_shell_indices),
        dtype=np.int32,
    )
    scale_mask_np = np.asarray(jax.block_until_ready(scale_mask), dtype=bool)
    relion_high_shell_np = (
        np.empty((0,), dtype=np.float32)
        if relion_norm_high_shell is None
        else np.asarray(jax.block_until_ready(relion_norm_high_shell), dtype=np.float32)
    )
    local_indices = np.asarray(image_indices, dtype=np.int64)
    group_ids_np = np.asarray(bucket_group_ids, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    os.makedirs(dump_dir, exist_ok=True)
    context_half = int(_bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"norm_residual_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        aa_shells = np.zeros(int(np.max(scale_shell_indices_np, initial=-1)) + 1, dtype=np.float64)
        valid_scale_shell = (
            (scale_shell_indices_np >= 0)
            & scale_mask_np
            & (scale_shell_indices_np < aa_shells.size)
        )
        np.add.at(
            aa_shells,
            scale_shell_indices_np[valid_scale_shell],
            aa_per_pixel_np[selected_row, valid_scale_shell].astype(np.float64),
        )
        np.savez_compressed(
            out_path,
            schema=np.asarray("recovar-k1-norm-residual-inputs-v2"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            proj_for_noise=proj_np[selected_row],
            proj_abs2_for_noise=proj_abs2_np[selected_row],
            summed_masked_noise=summed_np[selected_row],
            ctf_probs=ctf_probs_np[selected_row],
            ctf2_over_nv_recon=ctf2_over_nv_np[selected_row],
            posterior_probs=posterior_probs_np[selected_row],
            noise_variance_for_noise=noise_np,
            block_norm_residual=np.asarray(residual_np[selected_row]),
            processed_score_half_for_noise=processed_image_np[selected_row],
            shell_indices_half=shell_indices_np,
            support_mass=np.asarray(support_mass_np[selected_row]),
            relion_norm_high_shell=(
                relion_high_shell_np
                if relion_high_shell_np.size == 0
                else np.asarray(relion_high_shell_np[bucket_row])
            ),
            weighted_img_per_image=np.asarray(weighted_image_power_np[selected_row]),
            group_id=np.int64(group_ids_np[bucket_row]),
            scale_for_stats=np.asarray(scale_np[selected_row]),
            scale_correction_pixel_mask=scale_mask_np,
            scale_shell_indices=scale_shell_indices_np,
            scale_ctf_has_mass=ctf_has_mass_np[selected_row],
            scale_ctf_probs_raw=ctf_probs_raw_np[selected_row],
            scale_aa_terms_before_scale=aa_terms_before_scale_np[selected_row],
            scale_aa_terms=aa_terms_np[selected_row],
            scale_aa_per_pixel=aa_per_pixel_np[selected_row],
            scale_aa_per_shell=aa_shells,
            scale_aa_per_image=np.asarray(aa_per_image_np[selected_row]),
            raw_translated_recon=raw_translated_recon_np[selected_row],
            raw_translated_wavg=raw_translated_wavg_np[selected_row],
            wavg_window_indices=np.asarray(score_window_indices, dtype=np.int32),
        )
    return int(target_rows.size)


def _write_chunked_scale_aa_dump(
    *,
    dump_dir,
    experiment_dataset,
    image_indices,
    target_rows,
    current_size,
    bucket_group_ids,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    chunk_ranges,
    posterior_mass_chunks,
    proj_abs2_sum_chunks,
    ctf_probs_raw_sum_chunks,
    xa_per_pixel_chunks,
    xa_per_image_chunks,
    aa_before_scale_per_pixel_chunks,
    aa_per_pixel_chunks,
    aa_per_image_chunks,
    posterior_probs_chunks=None,
    rotation_matrix_chunks=None,
    fine_translations=None,
    aa_feature_per_shell_chunks=None,
    aa_feature_shell_ids=None,
    atomic_xa_per_pixel=None,
    atomic_aa_per_pixel=None,
    atomic_diff2_per_pixel=None,
):
    """Write compact Wavg XA/AA/diff2 intermediates for a target bucket."""

    target_rows = np.asarray(target_rows, dtype=np.int64)
    if target_rows.size == 0:
        return 0
    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    group_ids = np.asarray(bucket_group_ids, dtype=np.int64)[target_rows]
    scales = np.asarray(bucket_scale_for_stats, dtype=np.float32)[target_rows]
    mask = np.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    shells = np.asarray(scale_shell_indices, dtype=np.int32).reshape(-1)
    chunk_ranges_np = np.asarray(chunk_ranges, dtype=np.int64)
    posterior_mass = np.stack(posterior_mass_chunks, axis=1)
    proj_abs2_sum = np.stack(proj_abs2_sum_chunks, axis=1)
    ctf_probs_raw_sum = np.stack(ctf_probs_raw_sum_chunks, axis=1)
    xa_per_pixel_by_chunk = np.stack(xa_per_pixel_chunks, axis=1)
    xa_per_image_by_chunk = np.stack(xa_per_image_chunks, axis=1)
    aa_before_scale = np.stack(aa_before_scale_per_pixel_chunks, axis=1)
    aa_per_pixel_by_chunk = np.stack(aa_per_pixel_chunks, axis=1)
    aa_per_image_by_chunk = np.stack(aa_per_image_chunks, axis=1)
    atomic_aa_per_pixel_np = (
        None
        if atomic_aa_per_pixel is None
        else np.asarray(atomic_aa_per_pixel, dtype=np.float32)
    )
    atomic_diff2_per_pixel_np = (
        None
        if atomic_diff2_per_pixel is None
        else np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    )
    atomic_xa_per_pixel_np = (
        None
        if atomic_xa_per_pixel is None
        else np.asarray(atomic_xa_per_pixel, dtype=np.float32)
    )
    if atomic_xa_per_pixel_np is not None and atomic_xa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-XA atomic pixel topology changed")
    if atomic_aa_per_pixel_np is not None and atomic_aa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-AA atomic pixel topology changed")
    if atomic_diff2_per_pixel_np is not None and atomic_diff2_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked Wavg diff2 atomic pixel topology changed")
    candidate_arrays_present = posterior_probs_chunks is not None or rotation_matrix_chunks is not None
    if candidate_arrays_present:
        if posterior_probs_chunks is None or rotation_matrix_chunks is None or fine_translations is None:
            raise ValueError("chunked scale-AA candidate capture is incomplete")
        posterior_probs = np.concatenate(posterior_probs_chunks, axis=1)
        rotation_matrices = np.concatenate(rotation_matrix_chunks, axis=1)
        fine_translations_np = np.asarray(fine_translations, dtype=np.float32)
        if posterior_probs.shape[:2] != rotation_matrices.shape[:2]:
            raise ValueError("chunked scale-AA candidate rotation topology changed")
        if posterior_probs.shape[2] != fine_translations_np.shape[0]:
            raise ValueError("chunked scale-AA candidate translation topology changed")
        if aa_feature_per_shell_chunks is None or aa_feature_shell_ids is None:
            raise ValueError("chunked scale-AA candidate shell features are missing")
        aa_feature_per_shell = np.concatenate(aa_feature_per_shell_chunks, axis=1)
        aa_feature_shell_ids_np = np.asarray(aa_feature_shell_ids, dtype=np.int32)
        if aa_feature_per_shell.shape[:2] != posterior_probs.shape[:2]:
            raise ValueError("chunked scale-AA candidate shell-feature topology changed")
        if aa_feature_per_shell.shape[2] != aa_feature_shell_ids_np.size:
            raise ValueError("chunked scale-AA candidate shell labels changed")
    if not (
        posterior_mass.shape[:2]
        == proj_abs2_sum.shape[:2]
        == ctf_probs_raw_sum.shape[:2]
        == xa_per_pixel_by_chunk.shape[:2]
        == xa_per_image_by_chunk.shape[:2]
        == aa_before_scale.shape[:2]
        == aa_per_pixel_by_chunk.shape[:2]
        == aa_per_image_by_chunk.shape[:2]
        == (target_rows.size, chunk_ranges_np.shape[0])
    ):
        raise ValueError("chunked scale-AA capture topology changed")

    os.makedirs(dump_dir, exist_ok=True)
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    shell_count = int(np.max(shells, initial=-1)) + 1
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        xa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_before_scale_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        ctf_probs_raw_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        proj_abs2_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        for chunk_index in range(chunk_ranges_np.shape[0]):
            xa_per_pixel = (
                xa_per_pixel + xa_per_pixel_by_chunk[selected_row, chunk_index]
            ).astype(np.float32)
            aa_per_pixel = (aa_per_pixel + aa_per_pixel_by_chunk[selected_row, chunk_index]).astype(np.float32)
            aa_before_scale_per_pixel = (
                aa_before_scale_per_pixel + aa_before_scale[selected_row, chunk_index]
            ).astype(np.float32)
            ctf_probs_raw_per_pixel = (
                ctf_probs_raw_per_pixel + ctf_probs_raw_sum[selected_row, chunk_index]
            ).astype(np.float32)
            proj_abs2_per_pixel = (
                proj_abs2_per_pixel + proj_abs2_sum[selected_row, chunk_index]
            ).astype(np.float32)
        aa_per_shell = np.zeros(shell_count, dtype=np.float64)
        valid = mask & (shells >= 0) & (shells < shell_count)
        np.add.at(
            aa_per_shell,
            shells[valid],
            aa_per_pixel[valid].astype(np.float64),
        )
        production_aa_total = float(
            np.sum(aa_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        production_xa_total = float(
            np.sum(xa_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"scale_aa_chunked_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        payload = dict(
            schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v2"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            group_id=np.int64(group_ids[selected_row]),
            scale_for_stats=np.float32(scales[selected_row]),
            chunk_ranges=chunk_ranges_np,
            posterior_mass_per_chunk=posterior_mass[selected_row],
            proj_abs2_sum_per_pixel_by_chunk=proj_abs2_sum[selected_row],
            ctf_probs_raw_sum_per_pixel_by_chunk=ctf_probs_raw_sum[selected_row],
            aa_before_scale_per_pixel_by_chunk=aa_before_scale[selected_row],
            aa_per_pixel_by_chunk=aa_per_pixel_by_chunk[selected_row],
            aa_per_image_by_chunk=aa_per_image_by_chunk[selected_row],
            scale_correction_pixel_mask=mask,
            scale_shell_indices=shells,
            proj_abs2_sum_per_pixel=proj_abs2_per_pixel,
            ctf_probs_raw_sum_per_pixel=ctf_probs_raw_per_pixel,
            scale_xa_per_pixel_by_chunk=xa_per_pixel_by_chunk[selected_row],
            scale_xa_per_image_by_chunk=xa_per_image_by_chunk[selected_row],
            scale_xa_per_pixel=xa_per_pixel,
            scale_xa_per_image=np.float64(production_xa_total),
            xa_pixel_sum_minus_production_total=np.float64(
                np.sum(xa_per_pixel, dtype=np.float64) - production_xa_total
            ),
            scale_aa_terms_before_scale_per_pixel=aa_before_scale_per_pixel,
            scale_aa_per_pixel=aa_per_pixel,
            scale_aa_per_shell=aa_per_shell,
            scale_aa_per_image=np.float64(production_aa_total),
            pixel_sum_minus_production_total=np.float64(
                np.sum(aa_per_pixel, dtype=np.float64) - production_aa_total
            ),
        )
        if atomic_xa_per_pixel_np is not None:
            atomic_xa_pixels = atomic_xa_per_pixel_np[bucket_row]
            atomic_xa_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_xa_shells,
                shells[valid],
                atomic_xa_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_xa_atomic_per_pixel=atomic_xa_pixels,
                scale_xa_atomic_per_shell=atomic_xa_shells,
                scale_xa_atomic_per_image=np.float64(
                    np.sum(atomic_xa_pixels, dtype=np.float64)
                ),
            )
        if atomic_aa_per_pixel_np is not None:
            atomic_pixels = atomic_aa_per_pixel_np[bucket_row]
            atomic_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_shells,
                shells[valid],
                atomic_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_aa_atomic_per_pixel=atomic_pixels,
                scale_aa_atomic_per_shell=atomic_shells,
                scale_aa_atomic_per_image=np.float64(
                    np.sum(atomic_pixels, dtype=np.float64)
                ),
            )
        if atomic_diff2_per_pixel_np is not None:
            atomic_diff2_pixels = atomic_diff2_per_pixel_np[bucket_row]
            atomic_diff2_shells = np.zeros(shell_count, dtype=np.float64)
            valid_current_size = (shells >= 0) & (shells < shell_count)
            np.add.at(
                atomic_diff2_shells,
                shells[valid_current_size],
                atomic_diff2_pixels[valid_current_size].astype(np.float64),
            )
            payload.update(
                wavg_diff2_atomic_per_pixel=atomic_diff2_pixels,
                wavg_diff2_atomic_per_shell=atomic_diff2_shells,
                wavg_diff2_atomic_per_image=np.float64(
                    np.sum(atomic_diff2_pixels, dtype=np.float64)
                ),
            )
        if candidate_arrays_present:
            payload.update(
                candidate_posterior_probs=np.asarray(posterior_probs[selected_row], dtype=np.float32),
                candidate_rotation_matrices=np.asarray(rotation_matrices[selected_row], dtype=np.float32),
                fine_translations=fine_translations_np,
                candidate_aa_feature_per_shell=np.asarray(
                    aa_feature_per_shell[selected_row],
                    dtype=np.float32,
                ),
                candidate_aa_feature_shell_ids=aa_feature_shell_ids_np,
            )
        np.savez_compressed(out_path, **payload)
    return int(target_rows.size)


def _maybe_dump_k_class_pass2_bucket(
    *,
    experiment_dataset,
    image_indices,
    class_index,
    per_image_inputs,
    class_bucket_arrays,
    compact_pair_arrays,
    current_size,
    n_fine_trans,
    fine_translations,
    scores,
    probs,
    bucket_translation_prior,
    compact_pairs,
    fine_translation_parent=None,
    reconstruction_mask=None,
    reconstruction_probs=None,
    raw_diff2_by_batch_row=None,
    raw_operands_by_batch_row=None,
    relion_min_diff2=None,
):
    """Env-gated K-class sparse pass-2 dump for RELION parity debugging."""

    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return 0
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return 0
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return 0
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return 0
    target_class = os.environ.get("RECOVAR_PASS2_DUMP_CLASS")
    if target_class and int(target_class) != int(class_index) + 1:
        return 0

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return 0

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    trans_prior_np = np.asarray(bucket_translation_prior, dtype=np.float64)
    fine_translation_parent_np = (
        np.arange(int(n_fine_trans), dtype=np.int32)
        if fine_translation_parent is None
        else np.asarray(fine_translation_parent, dtype=np.int32)
    )
    recon_mask_np = None if reconstruction_mask is None else np.asarray(reconstruction_mask, dtype=bool)
    recon_probs_np = None if reconstruction_probs is None else np.asarray(reconstruction_probs, dtype=np.float64)
    min_diff2_np = (
        None
        if relion_min_diff2 is None
        else np.asarray(relion_min_diff2, dtype=np.float32)
    )
    if raw_diff2_by_batch_row is not None:
        if min_diff2_np is None:
            raise ValueError("raw diff2 pass-2 dump requires the common RELION minimum")
        if min_diff2_np.shape != local_indices.shape:
            raise ValueError(
                "RELION minimum shape differs from the pass-2 dump batch: "
                f"{min_diff2_np.shape} != {local_indices.shape}"
            )

    dump_count = 0
    for row in wanted_rows:
        image_idx = int(local_indices[row])
        original_idx = int(original_indices[row])
        rot_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotations = np.asarray(per_image_inputs["oversampled_rots"][image_idx], dtype=np.float32)
        parent_map = np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32)
        rotation_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=np.float64)
        n_rot = int(rot_indices.shape[0])

        if compact_pairs:
            pair_mask = np.asarray(compact_pair_arrays["pair_mask"][row], dtype=bool)
            pair_rot_rows = np.asarray(compact_pair_arrays["local_rotation_row"][row], dtype=np.int64)
            pair_trans = np.asarray(compact_pair_arrays["translation_idx"][row], dtype=np.int64)
            pair_scores = scores_np[row]
            pair_probs = probs_np[row]
            valid = (
                pair_mask
                & (pair_rot_rows >= 0)
                & (pair_rot_rows < n_rot)
                & (pair_trans >= 0)
                & (pair_trans < int(n_fine_trans))
            )
            scores_with = np.full((n_rot, int(n_fine_trans)), -np.inf, dtype=np.float64)
            prob_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=np.float64)
            candidate_mask = np.zeros((n_rot, int(n_fine_trans)), dtype=bool)
            reconstruction_mask_dense = None
            reconstruction_probs_dense = None
            raw_diff2_dense = None
            if np.any(valid):
                rr = pair_rot_rows[valid]
                tt = pair_trans[valid]
                scores_with[rr, tt] = pair_scores[valid]
                prob_dense[rr, tt] = pair_probs[valid]
                candidate_mask[rr, tt] = True
                if recon_mask_np is not None:
                    reconstruction_mask_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=bool)
                    reconstruction_mask_dense[rr, tt] = recon_mask_np[row][valid]
                if recon_probs_np is not None:
                    reconstruction_probs_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=np.float64)
                    reconstruction_probs_dense[rr, tt] = recon_probs_np[row][valid]
                if raw_diff2_by_batch_row is not None:
                    if row not in raw_diff2_by_batch_row:
                        raise ValueError(
                            f"raw diff2 pass-2 dump is missing batch row {row}"
                        )
                    raw_diff2_pair = np.asarray(
                        raw_diff2_by_batch_row[row],
                        dtype=np.float32,
                    )
                    if raw_diff2_pair.shape != pair_scores.shape:
                        raise ValueError(
                            "compact raw diff2 shape differs from scores: "
                            f"{raw_diff2_pair.shape} != {pair_scores.shape}"
                        )
                    raw_diff2_dense = np.full(
                        (n_rot, int(n_fine_trans)),
                        np.nan,
                        dtype=np.float32,
                    )
                    raw_diff2_dense[rr, tt] = raw_diff2_pair[valid]
        else:
            scores_with = scores_np[row, :n_rot, :]
            prob_dense = probs_np[row, :n_rot, :]
            candidate_mask = np.asarray(class_bucket_arrays["candidate_mask"][row, :n_rot, :], dtype=bool)
            reconstruction_mask_dense = (
                None if recon_mask_np is None else np.asarray(recon_mask_np[row, :n_rot, :], dtype=bool)
            )
            reconstruction_probs_dense = (
                None if recon_probs_np is None else np.asarray(recon_probs_np[row, :n_rot, :], dtype=np.float64)
            )
            raw_diff2_dense = None
            if raw_diff2_by_batch_row is not None:
                if row not in raw_diff2_by_batch_row:
                    raise ValueError(
                        f"raw diff2 pass-2 dump is missing batch row {row}"
                    )
                raw_diff2_dense = np.asarray(
                    raw_diff2_by_batch_row[row],
                    dtype=np.float32,
                )
                if raw_diff2_dense.shape != scores_with.shape:
                    raise ValueError(
                        "dense raw diff2 shape differs from scores: "
                        f"{raw_diff2_dense.shape} != {scores_with.shape}"
                    )

        scores_pre = (
            scores_with
            - rotation_prior[:, None]
            - trans_prior_np[row, None, :]
        )
        reconstruction_fields = {}
        if reconstruction_mask_dense is not None:
            reconstruction_fields["reconstruction_mask"] = reconstruction_mask_dense
            reconstruction_fields["reconstruction_n_significant"] = np.int64(np.count_nonzero(reconstruction_mask_dense))
        if reconstruction_probs_dense is not None:
            reconstruction_fields["reconstruction_probs"] = reconstruction_probs_dense
        raw_diff2_fields = {}
        if raw_diff2_dense is not None:
            raw_diff2_fields = {
                "relion_raw_diff2": raw_diff2_dense,
                "relion_min_diff2": np.float32(min_diff2_np[row]),
            }
        raw_operand_fields = {}
        if raw_operands_by_batch_row is not None:
            if row not in raw_operands_by_batch_row:
                raise ValueError(
                    f"raw operand pass-2 dump is missing batch row {row}"
                )
            raw_operands = raw_operands_by_batch_row[row]
            raw_operand_fields = {
                "raw_operand_schema": np.asarray(
                    "recovar-kclass-pass2-effective-raw-operands-v2"
                ),
                "raw_operand_actual_rotation_count": np.int64(
                    raw_operands["actual_rotation_count"]
                ),
                "raw_operand_raw_diff2": np.asarray(
                    raw_operands["raw_diff2"],
                    dtype=np.float32,
                ),
                "raw_operand_shifted_corrected": np.asarray(
                    raw_operands["shifted_corrected"],
                    dtype=np.complex64,
                ),
                "raw_operand_corr_img_score": np.asarray(
                    raw_operands["corr_img_score"],
                    dtype=np.float32,
                ),
                "raw_operand_proj_half": np.asarray(
                    raw_operands["proj_half"],
                    dtype=np.complex64,
                ),
                "raw_operand_half_weights": np.asarray(
                    raw_operands["half_weights"],
                    dtype=np.float32,
                ),
                "raw_operand_relion_full_to_compact": np.asarray(
                    raw_operands["relion_full_to_compact"],
                    dtype=np.int32,
                ),
                "raw_operand_highres_xi2_half": np.float32(
                    raw_operands["highres_xi2_half"]
                ),
                "raw_operand_pair_mask": np.asarray(
                    raw_operands["pair_mask"],
                    dtype=bool,
                ),
                "raw_operand_pair_rotation_row": np.asarray(
                    raw_operands["pair_rotation_row"],
                    dtype=np.int32,
                ),
                "raw_operand_pair_translation_idx": np.asarray(
                    raw_operands["pair_translation_idx"],
                    dtype=np.int32,
                ),
            }
        out_path = os.path.join(
            dump_dir,
            f"pass2_orig{original_idx:06d}_class{int(class_index) + 1:03d}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_idx),
            local_index=np.int64(image_idx),
            class_index=np.int64(class_index),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            n_fine_trans=np.int64(n_fine_trans),
            fine_translations=np.asarray(fine_translations, dtype=np.float32),
            fine_translation_parent=fine_translation_parent_np,
            rotations=rotations,
            oversampled_rot_indices=rot_indices,
            parent_map=parent_map,
            candidate_mask=candidate_mask,
            scores_with_prior=scores_with,
            scores_pre_prior=scores_pre,
            probs=prob_dense,
            rotation_log_prior=rotation_prior,
            translation_log_prior=trans_prior_np[row],
            compact_pair_dump=np.bool_(compact_pairs),
            **reconstruction_fields,
            **raw_diff2_fields,
            **raw_operand_fields,
        )
        dump_count += 1
    return dump_count


def _capture_k_class_pass2_raw_operands(
    *,
    raw_diff2,
    target_rows,
    actual_counts,
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    relion_full_to_compact,
    highres_xi2_half,
    pair_mask=None,
    pair_rotation_row=None,
    pair_translation_idx=None,
):
    """Stage the effective float32 raw-diff2 operands after scoring completes."""

    raw_diff2 = np.asarray(jax.block_until_ready(raw_diff2), dtype=np.float32)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    shifted_corrected = np.asarray(shifted_corrected, dtype=np.complex64)
    corr_img_score = np.asarray(corr_img_score, dtype=np.float32)
    proj_half = np.asarray(proj_half, dtype=np.complex64)
    half_weights = np.asarray(half_weights, dtype=np.float32)
    if relion_full_to_compact is None:
        relion_full_to_compact = np.arange(
            proj_half.shape[-1],
            dtype=np.int32,
        )
    else:
        relion_full_to_compact = np.asarray(
            relion_full_to_compact,
            dtype=np.int32,
        )
    if highres_xi2_half is None:
        highres_xi2_half = np.zeros(shifted_corrected.shape[0], dtype=np.float32)
    else:
        highres_xi2_half = np.asarray(highres_xi2_half, dtype=np.float32)
    if pair_mask is None:
        pair_mask = np.empty((shifted_corrected.shape[0], 0), dtype=bool)
        pair_rotation_row = np.empty(
            (shifted_corrected.shape[0], 0),
            dtype=np.int32,
        )
        pair_translation_idx = np.empty(
            (shifted_corrected.shape[0], 0),
            dtype=np.int32,
        )
    else:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        pair_rotation_row = np.asarray(pair_rotation_row, dtype=np.int32)
        pair_translation_idx = np.asarray(pair_translation_idx, dtype=np.int32)

    captured = {}
    for row in target_rows:
        row = int(row)
        n_rot = int(actual_counts[row])
        captured[row] = {
            "actual_rotation_count": np.int64(n_rot),
            "raw_diff2": np.array(raw_diff2[row], copy=True),
            "shifted_corrected": np.array(
                shifted_corrected[row],
                copy=True,
            ),
            "corr_img_score": np.array(corr_img_score[row], copy=True),
            "proj_half": np.array(proj_half[row], copy=True),
            "half_weights": np.array(half_weights, copy=True),
            "relion_full_to_compact": np.array(
                relion_full_to_compact,
                copy=True,
            ),
            "highres_xi2_half": np.float32(highres_xi2_half[row]),
            "pair_mask": np.array(pair_mask[row], copy=True),
            "pair_rotation_row": np.array(pair_rotation_row[row], copy=True),
            "pair_translation_idx": np.array(
                pair_translation_idx[row],
                copy=True,
            ),
        }
    return captured


def _materialize_k_class_capture_rows(
    *,
    image_indices,
    target_particle_rows,
    per_image_inputs,
    class_bucket_arrays,
    compact_pair_arrays,
    scores,
    probs,
    reconstruction_mask,
    reconstruction_probs,
    bucket_translation_prior,
    n_fine_trans: int,
):
    """Materialize only selected fused-K rows in rectangular diagnostic form."""

    rows = np.asarray(target_particle_rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("fused K-class capture requires at least one target particle row")
    image_indices_np = np.asarray(image_indices, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= image_indices_np.size):
        raise ValueError("fused K-class capture target row is outside the bucket")

    selected_image_indices = image_indices_np[rows]
    n_selected = int(rows.size)
    n_rot = int(class_bucket_arrays["bucket_size"])
    n_trans = int(n_fine_trans)

    def _selected(values):
        return np.asarray(jnp.asarray(values)[jnp.asarray(rows, dtype=jnp.int32)])

    selected_scores = _selected(scores)
    selected_probs = _selected(probs)
    selected_reconstruction_mask = (
        None if reconstruction_mask is None else _selected(reconstruction_mask).astype(bool, copy=False)
    )
    selected_reconstruction_probs = (
        None if reconstruction_probs is None else _selected(reconstruction_probs)
    )

    rotation_log_prior = np.zeros((n_selected, n_rot), dtype=np.float32)
    for selected_row, image_index in enumerate(selected_image_indices.tolist()):
        prior = np.asarray(per_image_inputs["log_prior"][int(image_index)], dtype=np.float32)
        if prior.size > n_rot:
            raise ValueError("fused K-class capture rotation prior exceeds its bucket")
        rotation_log_prior[selected_row, : prior.size] = prior

    if compact_pair_arrays is None:
        candidate_mask = _selected(class_bucket_arrays["candidate_mask"]).astype(bool, copy=False)
        dense_scores = selected_scores
        dense_probs = selected_probs
        dense_reconstruction_mask = selected_reconstruction_mask
        dense_reconstruction_probs = selected_reconstruction_probs
    else:
        pair_rows = _selected(compact_pair_arrays["local_rotation_row"]).astype(np.int64, copy=False)
        pair_translations = _selected(compact_pair_arrays["translation_idx"]).astype(np.int64, copy=False)
        pair_mask = _selected(compact_pair_arrays["pair_mask"]).astype(bool, copy=False)
        dense_scores = np.full((n_selected, n_rot, n_trans), -np.inf, dtype=selected_scores.dtype)
        dense_probs = np.zeros((n_selected, n_rot, n_trans), dtype=selected_probs.dtype)
        candidate_mask = np.zeros((n_selected, n_rot, n_trans), dtype=bool)
        dense_reconstruction_mask = (
            None
            if selected_reconstruction_mask is None
            else np.zeros((n_selected, n_rot, n_trans), dtype=bool)
        )
        dense_reconstruction_probs = (
            None
            if selected_reconstruction_probs is None
            else np.zeros((n_selected, n_rot, n_trans), dtype=selected_reconstruction_probs.dtype)
        )
        for selected_row in range(n_selected):
            valid = (
                pair_mask[selected_row]
                & (pair_rows[selected_row] >= 0)
                & (pair_rows[selected_row] < n_rot)
                & (pair_translations[selected_row] >= 0)
                & (pair_translations[selected_row] < n_trans)
            )
            rr = pair_rows[selected_row, valid]
            tt = pair_translations[selected_row, valid]
            if np.unique(rr * n_trans + tt).size != rr.size:
                raise RuntimeError("fused K-class capture encountered duplicate compact candidate pairs")
            dense_scores[selected_row, rr, tt] = selected_scores[selected_row, valid]
            dense_probs[selected_row, rr, tt] = selected_probs[selected_row, valid]
            candidate_mask[selected_row, rr, tt] = True
            if dense_reconstruction_mask is not None:
                dense_reconstruction_mask[selected_row, rr, tt] = selected_reconstruction_mask[
                    selected_row, valid
                ]
            if dense_reconstruction_probs is not None:
                dense_reconstruction_probs[selected_row, rr, tt] = selected_reconstruction_probs[
                    selected_row, valid
                ]

    if dense_reconstruction_probs is None:
        mstep_probs = dense_probs
    else:
        mstep_probs = dense_reconstruction_probs
    if dense_reconstruction_mask is None:
        dense_reconstruction_mask = mstep_probs > 0

    return {
        "image_indices": selected_image_indices,
        "batch_rows": rows,
        "scores": dense_scores,
        "probs": dense_probs,
        "candidate_mask": candidate_mask,
        "reconstruction_mask": dense_reconstruction_mask,
        "reconstruction_probs": mstep_probs,
        "rotation_log_prior": rotation_log_prior,
        "translation_log_prior": _selected(bucket_translation_prior),
        "rotations": _selected(class_bucket_arrays["mstep_rotations"]),
        "rotation_indices": _selected(class_bucket_arrays["rotation_indices"]),
        "actual_counts": _selected(class_bucket_arrays["actual_counts"]).astype(np.int64, copy=False),
    }


def _pass2_dump_target_rows(
    *,
    experiment_dataset,
    image_indices,
    current_size,
) -> np.ndarray:
    """Return batch rows selected by the explicit pass-2 dump contract."""

    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return np.empty((0,), dtype=np.int64)
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return np.empty((0,), dtype=np.int64)
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return np.empty((0,), dtype=np.int64)

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)
    return np.flatnonzero(
        np.isin(
            original_indices,
            np.fromiter(target_original_indices, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)


def _pass2_dump_requested_for_bucket(
    *,
    experiment_dataset,
    image_indices,
    current_size,
) -> bool:
    """Return whether this bucket must stay materialized for a pass-2 dump."""

    return bool(
        _pass2_dump_target_rows(
            experiment_dataset=experiment_dataset,
            image_indices=image_indices,
            current_size=current_size,
        ).size
    )


_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}


def _star_column(table, name: str):
    """Return a RELION STAR column with or without its legacy underscore."""

    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return table[candidate]
    raise ValueError(f"RELION source STAR column {name} is missing")


def _relion_exact_ctf_half_from_source_star(
    experiment_dataset,
    image_indices,
    image_shape,
):
    """Evaluate source-precision SPA CTFs with RELION's scalar implementation.

    The result uses RECOVAR's centered-y half-spectrum coordinates and sign.
    The source STAR is mandatory because the ordinary dataset metadata has
    already been rounded to float32 before pass 2.
    """

    source_star = os.environ.get("RECOVAR_K1_RELION_EXACT_CTF_STAR", "").strip()
    if not source_star:
        raise ValueError(
            "exact RELION BPref operands require RECOVAR_K1_RELION_EXACT_CTF_STAR"
        )
    source_path = Path(source_star).expanduser().resolve()
    cache_key = (str(source_path), tuple(int(size) for size in image_shape))
    cache = _RELION_EXACT_CTF_SOURCE_CACHE.get(cache_key)
    if cache is None:
        from recovar.data_io.starfile import read_star
        from recovar.relion_bind import _relion_bind_core as relion_bind

        particles, optics = read_star(str(source_path))
        if optics is None:
            raise ValueError(f"RELION source STAR has no optics table: {source_path}")
        optics_ids = np.asarray(_star_column(optics, "rlnOpticsGroup"), dtype=np.int64)
        if np.unique(optics_ids).size != optics_ids.size:
            raise ValueError(f"RELION source STAR has duplicate optics groups: {source_path}")
        cache = {
            "particles": particles,
            "optics": {
                int(group): optics.iloc[row]
                for row, group in enumerate(optics_ids)
            },
            "relion_bind": relion_bind,
            "images": {},
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    original_indices = _original_indices_for_local(
        experiment_dataset,
        np.asarray(image_indices, dtype=np.int64),
    )
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    ctf_rows = []
    for original_index in original_indices:
        original_index = int(original_index)
        cached_image = cache["images"].get(original_index)
        if cached_image is None:
            particle = cache["particles"].iloc[original_index]
            optics_group = int(
                particle["rlnOpticsGroup"]
                if "rlnOpticsGroup" in particle
                else particle["_rlnOpticsGroup"]
            )
            optics = cache["optics"][optics_group]

            def particle_value(name: str) -> float:
                return float(
                    particle[name] if name in particle else particle[f"_{name}"]
                )

            def optics_value(name: str) -> float:
                return float(optics[name] if name in optics else optics[f"_{name}"])

            native = np.asarray(
                cache["relion_bind"].get_ctf_image(
                    particle_value("rlnDefocusU"),
                    particle_value("rlnDefocusV"),
                    particle_value("rlnDefocusAngle"),
                    optics_value("rlnVoltage"),
                    optics_value("rlnSphericalAberration"),
                    optics_value("rlnAmplitudeContrast"),
                    0.0,
                    optics_value("rlnImagePixelSize"),
                    image_w,
                    image_h,
                    False,
                    False,
                    False,
                    particle_value("rlnPhaseShift"),
                    1.0,
                ),
                dtype=np.float64,
            )
            # RELION/FFTW stores y in standard order and uses the opposite CTF
            # sign from RECOVAR's forward-model convention.
            cached_image = (-np.fft.fftshift(native, axes=0)).reshape(-1)
            cache["images"][original_index] = cached_image
        ctf_rows.append(cached_image)
    return jnp.asarray(np.stack(ctf_rows, axis=0), dtype=jnp.float64)


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
    batch_size = int(batch.shape[0])
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
    )
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied and not relion_cuda_preprocess:
        batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half_rfloat = (
        _relion_exact_ctf_half_from_source_star(
            experiment_dataset,
            image_indices,
            image_shape,
        )
        if relion_exact_bpref_operands
        else None
    )
    ctf_half = (
        jnp.asarray(ctf_half_rfloat, dtype=jnp.float32)
        if ctf_half_rfloat is not None
        else config.compute_ctf_half(ctf_params)
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
        # stores the reciprocal as float32.  Preserve that cast boundary and
        # its scalar multiplication order instead of dividing by an already
        # rounded float32 variance.
        inverse_noise_half = jnp.reciprocal(
            jnp.asarray(noise_variance_half, dtype=jnp.float64)
        ).astype(jnp.float32)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        ctf2_over_nv_half = weighted_ctf_half * ctf_half
        relion_score_corr_img_half = _relion_cuda_corr_img_from_rfloat_ctf(
            inverse_noise_half[None, :],
            ctf_half_rfloat,
            batch_scale[:, None] if scale_corrections is not None else None,
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

    # BPref operands remain in their demonstrated native float32 order.  Only
    # fine-score corr_img uses RELION's distinct RFLOAT-square construction.
    ctf2_over_nv_recon_half = ctf2_over_nv_half
    if relion_score_corr_img_half is not None:
        ctf2_over_nv_half = relion_score_corr_img_half

    if return_direct_scoring_io and not folded_normalized_cc_operands:
        if relion_exact_bpref_operands:
            pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                batch_scale[:, None],
                ctf_half_rfloat,
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
        phase_factors = half_image_phase_factors(image_shape, batch_shifts)
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
            shifted_score_half = (
                apply_half_translation_phases(
                    score_weighted_half_for_score[:, score_indices],
                    score_phase,
                )
                if return_shifted_score
                else None
            )
            if relion_exact_bpref_operands:
                if relion_score_translation_angles is None:
                    raise ValueError(
                        "exact RELION BPref operands require RELION translation angles"
                    )
                from recovar import cuda_backproject

                shifted_recon_half = cuda_backproject.relion_translate_bpref_f32(
                    jnp.asarray(recon_bpref_input_half[:, recon_indices], dtype=jnp.complex64),
                    jnp.asarray(weighted_ctf_half[:, recon_indices], dtype=jnp.float32),
                    jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
                    recon_indices,
                    image_shape,
                )
            else:
                shifted_recon_half = apply_half_translation_phases(
                    recon_weighted_half[:, recon_indices],
                    recon_phase,
                )
            if score_with_masked_images:
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

                exact_shifted_recon_half = cuda_backproject.relion_translate_bpref_f32(
                    jnp.asarray(recon_bpref_input_half, dtype=jnp.complex64),
                    jnp.asarray(weighted_ctf_half, dtype=jnp.float32),
                    jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
                    jnp.arange(recon_bpref_input_half.shape[1], dtype=jnp.int32),
                    image_shape,
                )
            else:
                exact_shifted_recon_half = None
            shifted_score_half = (
                apply_half_translation_phases(score_weighted_half_for_score, translation_phases_half)
                if return_shifted_score
                else None
            )
            if score_with_masked_images:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else apply_half_translation_phases(recon_weighted_half, translation_phases_half)
                )
                shifted_score_half_with_dc = apply_half_translation_phases(
                    score_weighted_half,
                    translation_phases_half,
                )
            else:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else apply_half_translation_phases(recon_weighted_half, translation_phases_half)
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

            shifted_corrected_score_half = (
                cuda_backproject.relion_translate_score_f32(
                    jnp.asarray(direct_score_input, dtype=jnp.complex64),
                    jnp.asarray(
                        relion_score_translation_angles,
                        dtype=jnp.float32,
                    ),
                    direct_score_pixel_indices,
                    image_shape,
                )
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
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
):
    """Bucketed batched implementation of sparse pass-2 oversampling.

    Returns the same tuple as ``compute_pass2_stats_sparse``.

    ``relion_exact_fine_gaussian`` enables RELION's float32 fine-search
    diff2/minimum ordering. Float64 scoring deliberately uses the legacy
    algebraic expression as a high-precision diagnostic route.
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
    membership_diagnostics_active = _bpref_membership_dump_requested()
    scoped_diagnostic_flags = _scoped_bpref_diagnostic_flags(
        active=bpref_device_signature_active
    )
    execution_modes = _resolve_bpref_execution_modes(
        scoped_diagnostic_flags,
        device_signature_requested=device_signature_requested,
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
        _require_bpref_device_soft_particle_arm(
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
        and not use_float64_scoring
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
            current_size=current_size,
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
        and relion_x_half_f32_fine_posterior_enabled()
    )
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
            "Sparse pass-2 RELION x-half current-size BPref accumulator shape: "
            "volume_shape=%s current_size=%s padding_factor=%s recon_volume_shape=%s half_accum_shape=%s voxels=%d",
            tuple(volume_shape),
            current_size,
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
            current_size=current_size,
        )
    else:
        mean_for_proj = volume
        proj_volume_shape = volume_shape

    # Fine translations and prior mapping
    translations_source_np = np.asarray(translations)
    translations_np = np.asarray(translations_source_np, dtype=np.float32)
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
        fine_translations = np.asarray(fine_translations_source, dtype=np.float32)
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations_source = np.asarray(fine_translations_override)
        fine_translations = np.asarray(fine_translations_source, dtype=np.float32)
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
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=_env_flag_enabled(
            _RELION_FINE_ROTATION_EXECUTION_ORDER_ENV,
            default=False,
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
        processing_order_chunk_size = _optional_positive_int_env(
            _BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV,
        ) or 1
        processing_order_group_by_bucket_size = _env_flag_enabled(
            _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV,
            default=False,
        )
        processing_order_batch_consecutive_bucket_sizes = bool(
            preserve_bpref_particle_order and not processing_order_group_by_bucket_size
        )
        if preserve_bpref_particle_order:
            logger.info(
                "STRICT-PARITY: preserving fresh RELION physical BPref particle "
                "order (%s)",
                "stable within support-size buckets"
                if processing_order_group_by_bucket_size
                else "global; batching only consecutive equal-size supports",
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
    if use_relion_x_half_mstep and diagnostic_sequential_translation_reduction:
        logger.info(
            "Sparse pass-2 RELION x-half M-step diagnostic: sequential float32 "
            "translation reduction runs as %s",
            "a checked shadow" if shadow_only_mode_requested else "the standalone diagnostic path",
        )
    if use_relion_f32_fine_posterior:
        logger.info(
            "Sparse pass-2 RELION x-half M-step diagnostic: using float32 fine-posterior "
            "normalization and significance pruning"
        )
    if use_relion_fine_mstep_prune and not use_relion_x_half_mstep:
        logger.info("Sparse pass-2 M-step: applying RELION fine-pass significant-weight pruning")

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
    noise_norm_correction_total = None
    noise_wavg_direct_norm_current_total = None
    noise_wavg_direct_norm_high_total = None
    noise_scale_correction_xa_total = None
    noise_scale_correction_aa_total = None
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    group_ids_np = None
    explicit_scale_group_count = 0
    if scale_correction_group_count is not None:
        explicit_scale_group_count = int(scale_correction_group_count)
        if (
            explicit_scale_group_count < 0
            or not np.isfinite(float(scale_correction_group_count))
            or float(scale_correction_group_count) != float(explicit_scale_group_count)
        ):
            raise ValueError(
                "scale_correction_group_count must be a non-negative integer, "
                f"got {scale_correction_group_count!r}"
            )
    if group_ids is not None:
        group_ids_np = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if group_ids_np.shape != (n_images,):
            raise ValueError(f"group_ids must have shape ({n_images},), got {group_ids_np.shape}")
        if group_ids_np.size and int(np.min(group_ids_np)) < 0:
            raise ValueError("group_ids must be non-negative")
        inferred_scale_group_count = int(np.max(group_ids_np)) + 1 if group_ids_np.size else 1
        n_scale_groups = max(explicit_scale_group_count, inferred_scale_group_count)
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
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
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
            "Sparse pass-2 windowed prepare enabled; set %s=0 to disable "
            "(score_pixels=%d recon_pixels=%d full_half_pixels=%d)",
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            int(n_windowed),
            int(n_recon_windowed),
            int(n_half),
        )

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
    relion_exact_bpref_operands = _env_flag_enabled(
        "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS",
        default=False,
    )
    if relion_exact_bpref_operands:
        if use_float64_scoring:
            raise ValueError("exact RELION BPref operands require the native float32 path")
        logger.info(
            "STRICT-PARITY: using RELION binary64-to-float32 inverse-noise and "
            "fused translate-then-weight BPref operands"
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
                projection_kwargs = window_spec.projection_kwargs(return_abs2=False)
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
            "Sparse pass-2 windowed translation phases cached "
            "(score_pixels=%d recon_pixels=%d translations=%d)",
            int(n_windowed),
            int(n_recon_windowed),
            int(n_fine_trans),
        )

    exact_raw_diff2_cache_limit_bytes = 0
    exact_raw_diff2_cache_admission_logged = False
    if use_exact_relion_gaussian:
        free_device_memory_bytes = _device_free_memory_bytes()
        allocator_free_memory_bytes = _jax_allocator_free_memory_bytes()
        exact_raw_diff2_cache_max_bytes = _optional_nonnegative_int_env(
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
    progress_chunks_override = _optional_nonnegative_int_env(_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS_ENV)
    progress_seconds_override = _optional_nonnegative_int_env(_SPARSE_PASS2_GROUP_PROGRESS_SECONDS_ENV)
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
            _bpref_contribution_target_rows(experiment_dataset, image_indices)
            if device_signature_requested
            else np.empty((0,), dtype=np.int64)
        )
        bucket_diagnostic_modes = _resolve_bpref_bucket_diagnostic_modes(
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
            jnp.asarray(np.asarray(scale_corrections, dtype=np.float32)[image_indices])
            if scale_corrections is not None
            else jnp.ones(batch, dtype=jnp.float32)
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
            bucket_translation_prior = jnp.zeros((batch, n_fine_trans), dtype=jnp.float32)
        else:
            bucket_translation_prior = jnp.asarray(fine_translation_prior_2d[image_indices], dtype=jnp.float32)

        contribution_preprocess_operands = None
        high_precision_operand_bundle = bucket_diagnostic_modes[
            "high_precision_operand_bundle"
        ]
        if high_precision_operand_bundle:
            (
                diagnostic_relion_cuda_preprocess,
                diagnostic_integer_pre_shifts,
                diagnostic_batch_corr,
                diagnostic_batch_scale,
                diagnostic_relion_preprocess_kwargs,
            ) = prepare_batch_preprocess_operands(
                experiment_dataset,
                batch_data,
                image_indices,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
            )
            if diagnostic_integer_pre_shifts is None:
                diagnostic_integer_pre_shifts = np.zeros((batch, 2), dtype=np.int32)
            if diagnostic_batch_corr is None:
                diagnostic_batch_corr = np.ones(batch, dtype=np.float32)
            if diagnostic_relion_preprocess_kwargs is None:
                diagnostic_normalization_factors = np.ones(batch, dtype=np.float32)
            else:
                diagnostic_normalization_factors = np.asarray(
                    diagnostic_relion_preprocess_kwargs["relion_normalization_factors"],
                    dtype=np.float32,
                )
            diagnostic_image_mask, diagnostic_image_mask_mode = resolve_image_mask_for_half_preprocess(
                experiment_dataset,
                image_shape,
                require_mask=bool(score_with_masked_images),
            )
            contribution_preprocess_operands = {
                "integer_pre_shifts": diagnostic_integer_pre_shifts,
                "batch_image_corrections": diagnostic_batch_corr,
                "batch_scale_corrections": diagnostic_batch_scale,
                "relion_preprocess_normalization_factors": diagnostic_normalization_factors,
                "relion_cuda_preprocess": diagnostic_relion_cuda_preprocess,
                "image_mask": diagnostic_image_mask,
                "image_mask_mode": diagnostic_image_mask_mode,
            }

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
        else:
            direct_inverse_noise_score = direct_inverse_noise_half
            direct_ctf_rfloat_score = direct_ctf_rfloat_half
        relion_highres_xi2_half = None
        if (
            use_exact_relion_gaussian
            or (accumulate_noise and current_size is not None)
        ):
            relion_highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
                processed_score_half_for_noise,
                image_shape=image_shape,
                current_size=current_size,
            )
        if accumulate_noise and current_size is not None and relion_highres_xi2_half is not None:
            if _env_flag_enabled(
                _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
                default=False,
            ):
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
        translated_wavg_norm = bool(
            accumulate_noise
            and current_size is not None
            and _env_flag_enabled(_RELION_TRANSLATED_WAVG_NORM_ENV, default=False)
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
        relion_wavg_atomic_scale_aa = bool(
            accumulate_noise
            and noise_scale_correction_aa_total is not None
            and _env_flag_enabled(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=False)
        )
        relion_wavg_atomic_direct_residual_requested = bool(
            accumulate_noise
            and _env_flag_enabled(
                _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV,
                default=False,
            )
        )
        if (
            relion_wavg_atomic_direct_residual_requested
            and noise_scale_correction_aa_total is not None
            and not relion_wavg_atomic_scale_aa
        ):
            raise ValueError(
                f"{_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV}=1 requires "
                f"{_RELION_WAVG_ATOMIC_SCALE_AA_ENV}=1 and scale groups"
            )
        relion_wavg_atomic_direct_residual = bool(
            relion_wavg_atomic_direct_residual_requested
            and relion_wavg_atomic_scale_aa
        )
        if relion_wavg_atomic_direct_residual and current_size is None:
            raise ValueError("direct Wavg residual replacement requires current_size")
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
        if relion_wavg_atomic_direct_residual:
            direct_noise_log_key = int(current_size)
            if direct_noise_log_key not in _relion_wavg_direct_noise_log_keys:
                _relion_wavg_direct_noise_log_keys.add(direct_noise_log_key)
                logger.info(
                    "Sparse pass-2 RELION Wavg diagnostic: issuing the full "
                    "%d-pixel FFTW rectangle and replacing current-size noise "
                    "shells [0, %d] plus per-particle norm with direct residual atomics",
                    int(relion_wavg_rectangle.centered_indices.size),
                    int(current_size // 2),
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
        rotation_chunk_size = _guard_bpref_target_rotation_chunking(
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
                    projection_kwargs = window_spec.projection_kwargs(return_abs2=False)
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
                global_min_diff2 = jnp.full((batch,), jnp.inf, dtype=jnp.float32)
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
                    jnp.asarray(0.0, dtype=jnp.float32),
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
            bucket_original_indices = _original_indices_for_local(
                experiment_dataset,
                image_indices,
            )
            capture_chunked_particle_count = (
                compact_capture_requested_particle_count(
                    int(_bpref_contribution_context["iteration"]),
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
                    if relion_wavg_atomic_direct_residual
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
                    _pass2_dump_target_rows(
                        experiment_dataset=experiment_dataset,
                        image_indices=image_indices,
                        current_size=current_size,
                    )
                    if _env_flag_enabled(
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
                            _require_bpref_shadow_exact(
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
                            max_r=float(current_size // 2),
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
                            max_r=float(current_size // 2),
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
                    if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
                    if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
                        logger.info(
                            "RECOVAR_NOISE_DTYPE_DEBUG: block_noise_shells=%s",
                            block_noise_shells.dtype,
                        )
                    block_noise_shells_np = np.asarray(
                        block_noise_shells,
                        dtype=np.float64,
                    )
                    if relion_wavg_atomic_direct_residual:
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
                    if not relion_wavg_atomic_direct_residual:
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
                _maybe_dump_k1_bpref_rotation_mass(
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
                        _require_bpref_reduction_shadow_agreement(
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
                _maybe_dump_bpref_contribution_rows(
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
                    max_r=float(current_size // 2) if use_window else None,
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
                    iteration=int(_bpref_contribution_context["iteration"]),
                    half=int(_bpref_contribution_context["half"]),
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
                )
                if translated_wavg_norm and not relion_wavg_atomic_direct_residual:
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
                if relion_wavg_atomic_direct_residual:
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
                if relion_wavg_atomic_direct_residual:
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
                    dump_dir = os.environ.get(_PASS2_DUMP_DIR_ENV)
                    if not dump_dir:
                        raise ValueError(
                            "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS requires "
                            "RECOVAR_PASS2_DUMP_DIR"
                        )
                    chunked_scale_aa_dump_count = _write_chunked_scale_aa_dump(
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
                        aa_before_scale_per_pixel_chunks=chunked_scale_aa_before_scale_per_pixel,
                        aa_per_pixel_chunks=chunked_scale_aa_per_pixel,
                        aa_per_image_chunks=chunked_scale_aa_per_image,
                        posterior_probs_chunks=chunked_scale_aa_posterior_probs,
                        rotation_matrix_chunks=chunked_scale_aa_rotation_matrices,
                        fine_translations=fine_translations,
                        aa_feature_per_shell_chunks=chunked_scale_aa_feature_per_shell,
                        aa_feature_shell_ids=chunked_scale_aa_feature_shell_ids,
                        atomic_xa_per_pixel=relion_wavg_atomic_scale_xa_pixels_np[
                            :, relion_wavg_rectangle.exact_positions
                        ],
                        atomic_aa_per_pixel=relion_wavg_atomic_scale_aa_pixels_np[
                            :, relion_wavg_rectangle.exact_positions
                        ],
                        atomic_diff2_per_pixel=relion_wavg_atomic_diff2_pixels_np[
                            :, relion_wavg_rectangle.exact_positions
                        ],
                    )
                    if chunked_scale_aa_dump_count and _env_flag_enabled(
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
                max_posterior_np = np.asarray(global_max_posterior, dtype=np.float32)
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
            projection_kwargs = window_spec.projection_kwargs(return_abs2=False if (use_window or score_only) else None)
            if use_window:
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
            _pass2_top2_targets = _pass2_top2_debug_target_indices()
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
                _require_bpref_shadow_exact("normalized-CC score", scores, shadow_scores)
                shadow_score_bitwise_equal = True
        elif use_exact_relion_gaussian:
            raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
                shifted_corrected_score_split,
                ctf2_over_nv_score,
                proj_half,
                direct_half_weights,
                relion_score_full_to_compact,
                relion_highres_xi2_half,
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
                )
                _require_bpref_shadow_exact("exact Gaussian score", scores, shadow_scores)
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
                _require_bpref_shadow_exact("algebraic Gaussian score", scores, shadow_scores)
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
                    winner_take_all=winner_take_all,
                )
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
                    winner_take_all=winner_take_all,
                    return_diagnostics=True,
                )
                _require_bpref_shadow_exact(
                    "reconstruction probabilities",
                    reconstruction_probs,
                    shadow_reconstruction_probs,
                )
                _require_bpref_shadow_exact(
                    "reconstruction mask",
                    reconstruction_mask,
                    shadow_reconstruction_mask,
                )
                _require_bpref_shadow_exact(
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
            bucket_original_indices = _original_indices_for_local(
                experiment_dataset,
                image_indices,
            )
            if compact_capture_requested_for_original_indices(
                int(_bpref_contribution_context["iteration"]),
                bucket_original_indices,
            ):
                maybe_capture_k1_production_bucket(
                    iteration=int(_bpref_contribution_context["iteration"]),
                    half=int(_bpref_contribution_context["half"]),
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
            pass2_dump_count = _maybe_dump_pass2_bucket(
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
            )
            if pass2_dump_count and _env_flag_enabled(
                _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
                default=False,
            ):
                logger.info(
                    "Sparse K=1 pass-2 stop-after-dump requested via %s=1; "
                    "wrote %d requested file(s) at current_size=%s",
                    _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
                    int(pass2_dump_count),
                    "None" if current_size is None else str(int(current_size)),
                )
                raise Pass2DumpComplete(
                    dump_count=pass2_dump_count,
                    current_size=current_size,
                )

        if not score_only:
            # M-step accumulation: posterior-weighted sums per (image, rot).
            if use_relion_fine_mstep_prune:
                mstep_probs = reconstruction_probs
            else:
                mstep_probs = probs
            if membership_diagnostics_active:
                _maybe_dump_k1_bpref_membership(
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
                shadow_reduction_agreement = _require_bpref_reduction_shadow_agreement(
                    summed,
                    ctf_probs,
                    shadow_summed,
                    shadow_ctf_probs,
                )
                dump_summed = shadow_summed
                dump_ctf_probs = shadow_ctf_probs
            _maybe_dump_bpref_contribution_rows(
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
                max_r=float(current_size // 2) if use_window else None,
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
                _validate_bpref_positive_rotation_rows(
                    positive_rotation_rows,
                    target_particle_rows,
                    device_signature_requested=device_signature_requested,
                    winner_take_all=winner_take_all,
                )
                diagnostic_owners = _bpref_diagnostic_ownership_indices(
                    image_indices,
                    target_particle_rows,
                    device_signature_requested=device_signature_requested,
                )
                _validate_bpref_diagnostic_ownership(
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
                    max_r=float(current_size // 2) if use_window else None,
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
                    max_r=float(current_size // 2),
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
                    max_r=float(current_size // 2),
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
            )
            if translated_wavg_norm and not relion_wavg_atomic_direct_residual:
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
            if not relion_wavg_atomic_direct_residual:
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
            if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
            if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
            if relion_wavg_atomic_direct_residual:
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
                direct_norm_current = _relion_wavg_direct_norm_per_image(
                    relion_wavg_atomic_scale_triplet_pixels_np[:, :, 2],
                    relion_wavg_rectangle.shell_indices,
                    np.zeros(batch, dtype=np.float64),
                )
                direct_norm_high = np.asarray(relion_norm_high_shell, dtype=np.float64)
                noise_wavg_direct_norm_current_total[image_indices] += direct_norm_current
                noise_wavg_direct_norm_high_total[image_indices] += direct_norm_high
                noise_norm_correction_total[image_indices] += direct_norm_current + direct_norm_high
            else:
                noise_wsum_total += block_noise_shells_np
                noise_img_power_total += weighted_img_shells_np
            block_norm_residual = _compute_norm_residual_per_image(
                proj_for_noise,
                proj_abs2_for_noise,
                summed_masked_noise,
                ctf_probs,
                noise_variance_for_noise,
            )
            norm_residual_dump_count = _maybe_dump_norm_residual_inputs(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
                proj_for_noise=proj_for_noise,
                proj_abs2_for_noise=proj_abs2_for_noise,
                summed_masked_noise=summed_masked_noise,
                ctf_probs=ctf_probs,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                posterior_probs=noise_probs,
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
            )
            if norm_residual_dump_count and _env_flag_enabled(
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
            if not relion_wavg_atomic_direct_residual:
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
            max_posterior_np = np.asarray(max_posterior_bucket, dtype=np.float32)
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
        if relion_wavg_atomic_direct_residual:
            norm_dump_dir = os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
            if norm_dump_dir:
                os.makedirs(norm_dump_dir, exist_ok=True)
                context_iteration = int(_bpref_contribution_context["iteration"])
                context_half = int(_bpref_contribution_context["half"])
                local_rows = np.arange(n_images, dtype=np.int64)
                original_rows = _original_indices_for_local(experiment_dataset, local_rows)
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
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    relion_fine_mstep_prune_mode: str | None = None,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
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
    scoped_diagnostic_flags = _scoped_bpref_diagnostic_flags(
        active=bpref_device_signature_active
    )
    if device_signature_requested:
        if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"):
            raise RuntimeError(
                "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR requires "
                "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"
            )
        _require_bpref_device_soft_particle_arm(
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
        and not use_float64_scoring
    )

    volumes = jnp.asarray(volumes)
    n_classes = int(volumes.shape[0])
    if device_signature_requested and not any(
        _bpref_contribution_class_enabled(class_index)
        for class_index in range(n_classes)
    ):
        raise ValueError(
            f"{_BPREF_CONTRIBUTION_DUMP_CLASS_ENV} selects no class in a {n_classes}-class run"
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
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    if not hasattr(experiment_dataset, "image_shape") or not hasattr(experiment_dataset, "volume_shape"):
        raise NotImplementedError("fused sparse K-class pass-2 requires dataset image_shape and volume_shape")
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
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
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    device_memory_bytes = _device_memory_limit_bytes()
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)

    use_relion_x_half_mstep = bool(relion_x_half_mstep)
    if use_relion_x_half_mstep:
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max.  The reconstruction boundary then crops the output
        # back to ``volume_shape``.
        recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=current_size,
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
            current_size,
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
    compact_active_rows_env = os.environ.get(_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV)
    compact_active_rows = (
        compact_pairs
        and _env_flag_enabled(_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV, default=True)
    )
    reuse_compact_noise_sums = _env_flag_enabled(
        _SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV,
        default=False,
    )
    compact_noise_sums_match_mstep = bool(
        reuse_compact_noise_sums
        and half_spectrum_scoring
        and not score_with_masked_images
    )
    rectangular_active_rows = _env_flag_enabled(_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_ENV, default=True)
    rectangular_active_prematmul = (
        rectangular_active_rows
        and _env_flag_enabled(_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_ENV, default=False)
    )
    rectangular_active_prematmul_max_grouped_dense_ratio = _optional_positive_float_env(
        _SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO_ENV,
    )
    if rectangular_active_prematmul_max_grouped_dense_ratio is None:
        rectangular_active_prematmul_max_grouped_dense_ratio = (
            _DEFAULT_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO
        )
    fused_noise_norm = _env_flag_enabled(_SPARSE_KCLASS_FUSED_NOISE_NORM_ENV, default=True)
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
    ) = _compact_pair_tail_bucket_coalesce_params_for_pass()
    if compact_pairs:
        compact_pair_min_bucket_size = _compact_pair_min_bucket_size_for_pass()
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
            else _bucket_summary_by_key(compact_pair_buckets, "pair_bucket_size"),
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
        if _env_flag_enabled(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False):
            raise ValueError(
                f"{_COMPACT_KCLASS_PAIRS_CHECK_ENV}=1 cannot be combined with "
                f"{_SPARSE_KCLASS_COMPACT_PAIRS_ENV}=1",
            )
    elif _env_flag_enabled(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False):
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
    compact_buckets = _env_flag_enabled(_SPARSE_KCLASS_COMPACT_BUCKETS_ENV, default=False)
    if compact_buckets:
        logger.info(
            "Sparse fused K-class compact buckets enabled via %s=1; default rectangular fused path unchanged",
            _SPARSE_KCLASS_COMPACT_BUCKETS_ENV,
        )

    Ft_y_total = [jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype) for _ in range(n_classes)]
    Ft_ctf_total = [jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype) for _ in range(n_classes)]
    class_hard_assignments = np.empty((n_classes, n_images), dtype=np.int32)
    best_rotations = [np.empty((n_images, 3, 3), dtype=np.float32) for _ in range(n_classes)]
    best_rotation_indices = [np.empty(n_images, dtype=np.int64) for _ in range(n_classes)]
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_score_log_z = np.empty((n_classes, n_images), dtype=np.float64)
    best_log_score = np.empty((n_classes, n_images), dtype=np.float64)
    max_posterior = np.empty((n_classes, n_images), dtype=np.float32)
    rotation_posterior_sums = np.zeros((n_classes, n_coarse_rot), dtype=np.float64)
    class_posterior_sums_mstep = np.zeros(n_classes, dtype=np.float64)
    compact_pair_check_max_abs_diff = 0.0
    compact_pair_check_rows = 0
    compact_pair_check_finite_mismatches = 0
    compact_pair_noise_sum_reuses = 0
    compact_pair_noise_ctf_sum_reuses = 0
    compact_pair_noise_image_sum_precomputes = 0
    compact_pair_noise_fused_active_gathers = 0

    group_ids_np = None
    noise_scale_correction_xa_total = None
    noise_scale_correction_aa_total = None
    explicit_scale_group_count = 0
    if scale_correction_group_count is not None:
        explicit_scale_group_count = int(scale_correction_group_count)
        if (
            explicit_scale_group_count < 0
            or not np.isfinite(float(scale_correction_group_count))
            or float(scale_correction_group_count) != float(explicit_scale_group_count)
        ):
            raise ValueError(
                "scale_correction_group_count must be a non-negative integer, "
                f"got {scale_correction_group_count!r}"
            )
    if group_ids is not None:
        group_ids_np = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if group_ids_np.shape != (n_images,):
            raise ValueError(f"group_ids must have shape ({n_images},), got {group_ids_np.shape}")
        if group_ids_np.size and int(np.min(group_ids_np)) < 0:
            raise ValueError("group_ids must be non-negative")
        inferred_scale_group_count = int(np.max(group_ids_np)) + 1 if group_ids_np.size else 1
        n_scale_groups = max(explicit_scale_group_count, inferred_scale_group_count)
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

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
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
            "Sparse fused K-class pass-2 windowed prepare enabled; set %s=0 to disable "
            "(score_pixels=%d recon_pixels=%d full_half_pixels=%d)",
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            int(n_windowed),
            int(n_recon_windowed),
            int(n_half),
        )

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

    def _stage_raw_diff2_on_host(raw_diff2, current_bucket_bytes):
        nonlocal raw_host_staging_total_bytes
        nonlocal raw_host_staging_peak_bytes
        nonlocal raw_host_staging_s

        raw_nbytes = int(raw_diff2.size) * np.dtype(np.float32).itemsize
        next_bucket_bytes = int(current_bucket_bytes) + raw_nbytes
        if next_bucket_bytes > int(raw_host_staging_max_bytes):
            raise MemoryError(
                "fused K-class raw diff2 host staging would exceed its hard cap: "
                f"requested={next_bucket_bytes} bytes, cap={raw_host_staging_max_bytes} bytes. "
                f"Increase {_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES_ENV} or lower the "
                "sparse pass-2 hypothesis microbatch cap."
            )
        stage_t0 = time.time()
        raw_host = np.asarray(raw_diff2, dtype=np.float32)
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
            _pass2_dump_target_rows(
                experiment_dataset=experiment_dataset,
                image_indices=image_indices,
                current_size=current_size,
            )
            if dump_pass2_operands
            else np.empty((0,), dtype=np.int64)
        )
        target_particle_rows = (
            _bpref_contribution_target_rows(experiment_dataset, image_indices)
            if device_signature_requested
            else np.empty((0,), dtype=np.int64)
        )
        bucket_diagnostic_modes = _resolve_bpref_bucket_diagnostic_modes(
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
            (
                diagnostic_relion_cuda_preprocess,
                diagnostic_integer_pre_shifts,
                diagnostic_batch_corr,
                diagnostic_batch_scale,
                diagnostic_relion_preprocess_kwargs,
            ) = prepare_batch_preprocess_operands(
                experiment_dataset,
                batch_data,
                image_indices,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
            )
            if diagnostic_integer_pre_shifts is None:
                diagnostic_integer_pre_shifts = np.zeros((batch, 2), dtype=np.int32)
            if diagnostic_batch_corr is None:
                diagnostic_batch_corr = np.ones(batch, dtype=np.float32)
            if diagnostic_relion_preprocess_kwargs is None:
                diagnostic_normalization_factors = np.ones(batch, dtype=np.float32)
            else:
                diagnostic_normalization_factors = np.asarray(
                    diagnostic_relion_preprocess_kwargs["relion_normalization_factors"],
                    dtype=np.float32,
                )
            diagnostic_image_mask, diagnostic_image_mask_mode = resolve_image_mask_for_half_preprocess(
                experiment_dataset,
                image_shape,
                require_mask=bool(score_with_masked_images),
            )
            contribution_preprocess_operands = {
                "integer_pre_shifts": diagnostic_integer_pre_shifts,
                "batch_image_corrections": diagnostic_batch_corr,
                "batch_scale_corrections": diagnostic_batch_scale,
                "relion_preprocess_normalization_factors": diagnostic_normalization_factors,
                "relion_cuda_preprocess": diagnostic_relion_cuda_preprocess,
                "image_mask": diagnostic_image_mask,
                "image_mask_mode": diagnostic_image_mask_mode,
            }
        bucket_group_ids = (
            jnp.asarray(group_ids_np[image_indices], dtype=jnp.int32)
            if group_ids_np is not None
            else None
        )
        bucket_scale_for_stats = (
            jnp.asarray(np.asarray(scale_corrections, dtype=np.float32)[image_indices])
            if scale_corrections is not None
            else jnp.ones(batch, dtype=jnp.float32)
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
        )
        relion_highres_xi2_half = None
        if use_exact_relion_gaussian or (accumulate_noise and current_size is not None):
            relion_highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
                processed_score_half_for_noise,
                image_shape=image_shape,
                current_size=current_size,
            )
        if accumulate_noise and current_size is not None and relion_highres_xi2_half is not None:
            if _env_flag_enabled(
                _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
                default=False,
            ):
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
                        jnp.asarray(compact_arrays["log_prior"], dtype=jnp.float32)
                    )
                    raw_diff2_translation_priors_by_class.append(
                        jnp.asarray(bucket_translation_prior[row, safe_translation_idx], dtype=jnp.float32)
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
                        )[jnp.newaxis, :, :]
                    else:
                        raw_diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
                            shifted_corrected_score_split,
                            ctf2_over_nv_score,
                            proj_half,
                            direct_half_weights,
                            relion_score_full_to_compact,
                            relion_highres_xi2_half,
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
                        jnp.asarray(arrays["log_prior"], dtype=jnp.float32)[:, :, None]
                    )
                    raw_diff2_translation_priors_by_class.append(
                        jnp.asarray(bucket_translation_prior, dtype=jnp.float32)[:, None, :]
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
                and _env_flag_enabled(
                    _PASS2_DUMP_RAW_OPERANDS_ENV,
                    default=False,
                )
                and (
                    not target_dump_class
                    or int(target_dump_class) == class_index + 1
                )
            ):
                raw_operand_dump_by_class[class_index] = (
                    _capture_k_class_pass2_raw_operands(
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
                    raw_diff2_np = np.asarray(raw_diff2, dtype=np.float32)
                    raw_diff2_dump_by_class[class_index] = {
                        int(row): np.array(raw_diff2_np[int(row)], copy=True)
                        for row in pass2_dump_rows
                    }
                score = _relion_cuda_fine_diff2_to_scores(
                    jnp.asarray(raw_diff2, dtype=jnp.float32),
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
                    )
                    row = jnp.arange(batch)[:, None]
                    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
                    compact_scores = _relion_cuda_fine_diff2_to_scores(
                        compact_raw_diff2,
                        jnp.asarray(compact_arrays["log_prior"], dtype=jnp.float32),
                        jnp.asarray(
                            bucket_translation_prior[row, safe_translation_idx],
                            dtype=jnp.float32,
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
                    dtype=np.float32,
                )
            del raw_diff2_by_class
            if bucket_raw_host_staging_bytes != 0:
                raise RuntimeError(
                    "fused K-class raw diff2 host staging accounting did not return to zero: "
                    f"{bucket_raw_host_staging_bytes} bytes"
                )
        _add_sparse_group_timing(group_timing, "score", time.time() - stage_t0)

        global_score_log_z_bucket = _logsumexp_class_log_z(jnp.stack(class_score_log_z_bucket, axis=0))
        joint_mstep_masks_by_class = None
        if relion_fine_mstep_prune_mode == "joint":
            flat_joint_probs_by_class = []
            flat_joint_scores_by_class = []
            joint_prob_shapes = []
            for class_index, arrays in enumerate(class_bucket_arrays):
                if bucket_uses_compact_pairs:
                    pair_arrays = compact_pair_arrays_by_class[class_index]
                    pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                    _log_Z, pair_probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                        _normalize_pass2_pairs_with_log_z(
                            scores_by_class[class_index],
                            pair_mask,
                            global_score_log_z_bucket,
                        )
                    )
                    if winner_take_all:
                        pair_probs = _winner_take_all_pair_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    pair_probs = jnp.where(pair_mask, pair_probs, 0.0)
                    flat_joint_probs_by_class.append(pair_probs.reshape(batch, -1))
                    flat_joint_scores_by_class.append(
                        jnp.where(pair_mask, scores_by_class[class_index], -jnp.inf).reshape(batch, -1)
                    )
                    joint_prob_shapes.append(pair_probs.shape)
                else:
                    _log_Z, probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                        _normalize_pass2_bucket_with_log_z(
                            scores_by_class[class_index],
                            global_score_log_z_bucket,
                        )
                    )
                    if winner_take_all:
                        probs = _winner_take_all_bucket_probs(
                            scores_by_class[class_index],
                            best_argmax,
                            best_log_score_bucket,
                        )
                    flat_joint_probs_by_class.append(probs.reshape(batch, -1))
                    flat_joint_scores_by_class.append(scores_by_class[class_index].reshape(batch, -1))
                    joint_prob_shapes.append(probs.shape)
            if winner_take_all:
                flat_joint_masks = _relion_joint_winner_take_all_masks(flat_joint_scores_by_class)
            else:
                flat_joint_masks = _relion_pass2_reconstruction_joint_masks(
                    flat_joint_probs_by_class,
                    adaptive_fraction=float(adaptive_fraction),
                )
            joint_mstep_masks_by_class = [
                flat_mask.reshape(shape)
                for flat_mask, shape in zip(flat_joint_masks, joint_prob_shapes, strict=True)
            ]
        if dump_pass2_operands and _env_flag_enabled(_PASS2_DUMP_STOP_AFTER_TARGET_ENV, default=False):
            bucket_dump_count = 0
            for class_index, arrays in enumerate(class_bucket_arrays):
                if bucket_uses_compact_pairs:
                    pair_arrays = compact_pair_arrays_by_class[class_index]
                    pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                    _log_Z, pair_probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                        _normalize_pass2_pairs_with_log_z(
                            scores_by_class[class_index],
                            pair_mask,
                            global_score_log_z_bucket,
                        )
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
                    bucket_dump_count += _maybe_dump_k_class_pass2_bucket(
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
                    _log_Z, probs, best_log_score_bucket, best_argmax, _max_posterior_bucket = (
                        _normalize_pass2_bucket_with_log_z(
                            scores_by_class[class_index],
                            global_score_log_z_bucket,
                        )
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
                    bucket_dump_count += _maybe_dump_k_class_pass2_bucket(
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
                    dump_dir=os.environ[_PASS2_DUMP_DIR_ENV],
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
        log_score_offset = (
            np.asarray(
                _relion_cuda_fine_log_evidence_offset(global_min_diff2),
                dtype=np.float64,
            )
            if use_exact_relion_gaussian
            else -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
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
            if bucket_uses_compact_pairs:
                pair_arrays = compact_pair_arrays_by_class[class_index]
                pair_mask = jnp.asarray(pair_arrays["pair_mask"])
                log_Z, pair_probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                    _normalize_pass2_pairs_with_log_z(
                        scores_by_class[class_index],
                        pair_mask,
                        global_score_log_z_bucket,
                    )
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
                if relion_fine_mstep_prune_mode == "joint":
                    reconstruction_mask = joint_mstep_masks_by_class[class_index]
                    reconstruction_probs = jnp.where(reconstruction_mask, pair_probs, 0.0)
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
                _maybe_dump_k_class_pass2_bucket(
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
                if accumulate_noise and reuse_compact_noise_sums and not compact_noise_sums_match_mstep:
                    compact_pair_noise_image_sum_precomputes += 1
                    (
                        summed,
                        summed_masked_noise_precomputed,
                        ctf_probs,
                        probs_sum_t_jax,
                        translation_posterior_jax,
                    ) = _compact_pair_weighted_rotation_and_image_sums(
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
                        )
                    )
            else:
                log_Z, probs, best_log_score_bucket, best_argmax, max_posterior_bucket = (
                    _normalize_pass2_bucket_with_log_z(
                        scores_by_class[class_index],
                        global_score_log_z_bucket,
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
                if relion_fine_mstep_prune_mode == "joint":
                    reconstruction_mask = joint_mstep_masks_by_class[class_index]
                    reconstruction_probs = jnp.where(reconstruction_mask, probs, 0.0)
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
                _maybe_dump_k_class_pass2_bucket(
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
            if (
                bucket_device_signature_requested
                and _bpref_contribution_class_enabled(class_index)
            ):
                capture = _materialize_k_class_capture_rows(
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
                shadow_reduction_agreement = _require_bpref_reduction_shadow_agreement(
                    ordinary_capture_summed,
                    ordinary_capture_ctf,
                    shadow_capture_summed,
                    shadow_capture_ctf,
                )
                positive_rotation_rows = np.count_nonzero(
                    np.sum(np.asarray(capture["reconstruction_probs"]), axis=-1) > 0,
                    axis=1,
                )
                _validate_bpref_positive_rotation_rows(
                    positive_rotation_rows,
                    np.arange(capture["image_indices"].size, dtype=np.int64),
                    device_signature_requested=True,
                    winner_take_all=winner_take_all,
                    posterior_partitioned_across_classes=True,
                )
                diagnostic_owners = _bpref_diagnostic_ownership_indices(
                    capture["image_indices"],
                    np.arange(capture["image_indices"].size, dtype=np.int64),
                    device_signature_requested=True,
                )
                _validate_bpref_diagnostic_ownership(
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

                _maybe_dump_bpref_contribution_rows(
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
            if active_flat_rows_chunked:
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
                    elif reuse_compact_noise_sums:
                        if summed_masked_noise_precomputed is None:
                            summed_masked_noise = _compact_pair_weighted_image_sums(
                                noise_probs,
                                jnp.asarray(pair_arrays["local_rotation_row"]),
                                jnp.asarray(pair_arrays["translation_idx"]),
                                pair_mask,
                                shifted_noise_split,
                                n_rotation_rows=class_bucket_size,
                                allow_pair_sparse=compact_pair_pair_sparse_effective,
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
                if bucket_uses_active_rows:
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
                if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
                if bucket_uses_active_rows and bucket_uses_compact_pairs:
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
                    if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
                    if _env_flag_enabled("RECOVAR_NOISE_DTYPE_DEBUG", default=False):
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
            max_posterior_np = np.asarray(max_posterior_bucket, dtype=np.float32)
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
        per_class_best_pose_rotations=tuple(best_rotations),
        per_class_best_pose_translations=best_translations,
        per_class_best_pose_rotation_ids=tuple(best_rotation_indices),
        profile_summary=profile_summary,
        class_posterior_sums=class_posterior_sums_mstep if relion_fine_mstep_prune else None,
    )
