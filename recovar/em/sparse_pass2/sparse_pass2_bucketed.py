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
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics import bpref_diagnostics
from recovar.em.diagnostics import norm_scale as norm_scale_diagnostics
from recovar.em.diagnostics import pass2 as pass2_diagnostics
from recovar.em.diagnostics.compact_candidate_capture import (
    compact_capture_requested_for_original_indices,
    compact_capture_requested_particle_count,
    maybe_capture_k1_production_bucket,
    maybe_capture_k1_production_bucket_chunked,
    require_chunked_capture_capacity,
)
from recovar.em.diagnostics.sparse_pass2_dump import (
    _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV,
    _PASS2_DUMP_STOP_AFTER_TARGET_ENV,
    Pass2DumpComplete,
    _add_sparse_group_timing,
    _k1_pass2_dump_progress,
    _k_class_pass2_dump_progress,
    _log_pass2_top2_debug,
    _log_sparse_kclass_group_timing,
    _pass2_dump_requested_for_bucket,
    _pass2_top2_debug_target_indices,
    _prioritize_stopped_pass2_dump_buckets,
    _resolve_local_target_indices,
)
from recovar.em.helpers.batch_fetch import fetch_indexed_batch, original_image_indices
from recovar.em.helpers.env_flags import parse_env_flag, parse_env_int_set, parse_env_nonnegative_int
from recovar.em.helpers.half_spectrum import (
    make_relion_noise_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from recovar.em.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from recovar.em.helpers.normalization_inputs import optional_normalization_vector
from recovar.em.helpers.oversampling import _find_significant_mask_full_sort
from recovar.em.helpers.preprocessing import half_translation_phase_table, prepare_batch_preprocess_operands
from recovar.em.helpers.projection import compute_norm_residual_per_image as _compute_norm_residual_per_image
from recovar.em.helpers.projection import (
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
)
from recovar.em.helpers.projection import relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask
from recovar.em.helpers.scale_groups import prepare_scale_correction_groups
from recovar.em.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.helpers.types import OMITTED, make_noise_stats, make_relion_stats, sparse_pass2_result
from recovar.em.local.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.scoring.compact_candidates import (  # noqa: F401 - SparseCandidateMask.__module__ is pinned to this module, so legacy pickles resolve it here
    SparseCandidateMask,
    _candidate_mask_count,
)
from recovar.em.scoring.sparse_bucket_arrays import (
    _bucket_pass2_inputs,
    _bucket_sparse_k_class_pass2_inputs,
    _build_bucket_arrays,
    _build_compact_pair_bucket_arrays,
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
    _build_k_class_bucket_arrays,
    _compact_pair_image_mask_for_threshold,
    _prepare_per_image_compact_candidate_pairs,
    _prepare_per_image_pass2_inputs,
)
from recovar.em.sparse_pass2.sparse_pass2_adjoint import (
    _accumulate_active_flat_rows_adjoint_chunked,
    _accumulate_adjoint_block_chunked,
    _accumulate_relion_x_half_per_particle_launches,
    _projection_rotation_chunk_size,
    _split_compact_pair_buckets_by_projection_gather_budget,
)
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
    _reorder_to_indices,
    _translation_phase_table_for_indices,
)
from recovar.em.sparse_pass2.sparse_pass2_bucket_plan import (
    _bucket_group_stats,
    _bucket_summary,
    _compact_k_class_pair_plan_stats_from_counts,
    _compact_pair_buckets_for_execution_threshold,
    _compact_pair_execution_mask_excluding_full_support,
    _compact_pair_hybrid_threshold_reports,
    _compact_pair_threshold_report_thresholds,
    _hybrid_k_class_compact_pair_execution_buckets,
    _k_class_execution_bucket_group_stats,
    _maybe_prepare_sparse_k_class_compact_pair_plan,
    _tag_k_class_execution_bucket,
    _validate_k_class_execution_bucket_partition,
)
from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _EXACT_RAW_DIFF2_CACHE_MAX_BYTES,
    _compact_pair_dense_mstep_max_bytes_for_pass,
    _device_free_memory_bytes,
    _dtype_itemsize,
    _exact_raw_diff2_cache_estimated_bytes,
    _exact_raw_diff2_cache_fits_budget,
    _exact_raw_diff2_cache_limit_bytes,
    _jax_allocator_free_memory_bytes,
    _max_adjoint_block_bytes_for_pass,
    _max_hypotheses_per_microbatch_for_pass,
    _max_images_for_translation_tile,
    _max_noise_block_bytes_for_pass,
    _max_projection_gather_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _optional_positive_float_env,
    _optional_positive_int_env,
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
)
from recovar.em.sparse_pass2.sparse_pass2_compact_pair_sums import (
    _active_flat_gather_chunk_rows,
    _active_flat_row_indices_from_probs_sum_t,
    _active_image_indices_for_rotation_rows,
    _compact_pair_weighted_image_sums,
    _compact_pair_weighted_rotation_and_image_sums,
    _compact_pair_weighted_rotation_and_image_sums_native,
    _compact_pair_weighted_rotation_sums,
    _compact_pair_weighted_sums_and_noise_native,
    _compute_active_noise_rows_chunked,
    _rectangular_active_prematmul_is_efficient,
    _rectangular_active_weighted_image_sums_or_none,
    _rectangular_active_weighted_sums_or_none,
    _select_active_flat_rows,
    _select_active_flat_values,
)
from recovar.em.sparse_pass2.sparse_pass2_noise_blocks import (
    _compute_noise_block_and_norm_residual_chunked,
    _compute_noise_block_chunked,
)
from recovar.em.sparse_pass2.sparse_pass2_policy import (
    _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV,
    _BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV,
    _BPREF_REVERSE_PHYSICAL_ORDER_ENV,
    _COMPACT_KCLASS_PAIRS_CHECK_ENV,
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    _SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV,
    _SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV,
    _SPARSE_KCLASS_COMPACT_PAIRS_ENV,
    _SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV,
    _SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV,
    _SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV,
    _active_row_pad_multiple_for_pass,
    _cached_score_rotation_chunk_size_for_pass,
    _compact_pair_execution_enabled_for_pass,
    _compact_pair_max_images_per_microbatch_for_pass,
    _compact_pair_min_bucket_size_for_pass,
    _compact_pair_mstep_mode_for_pass,
    _compact_pair_prepare_max_images_per_microbatch,
    _compact_pair_tail_bucket_coalesce_params_for_pass,
    _fresh_k1_direct_noise_default,
    _fused_mstep_noise_enabled_for_pass,
    _max_images_for_sparse_pass2_translation_tile,
    _native_dual_weighted_sums_enabled_for_pass,
    _pass2_conservative_dump_execution_enabled,
    _pass2_dump_enabled,
    _projection_cache_enabled_for_pass,
    _relion_exact_bpref_operands_enabled,
    _relion_powerclass_spectrum_norm_enabled,
    _relion_wavg_direct_modes,
    _resolve_bpref_execution_bucket_policy,
    _resolve_bpref_processing_order,
    _small_bucket_coalesce_size_for_pass,
    _tail_bucket_coalesce_params_for_pass,
    _translation_tile_half_pixels_for_budget,
    _windowed_translation_tile_cap_enabled_for_pass,
)
from recovar.em.sparse_pass2.sparse_pass2_posterior import (
    _SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE_ENV,
    _diagnostics_from_normalized_pass2_probs,
    _logsumexp_class_log_z,
    _logsumexp_pass2_bucket_score_only,
    _logsumexp_pass2_pairs_score_only,
    _normalize_pass2_bucket,
    _normalize_pass2_bucket_score_only,
    _normalize_pass2_bucket_with_log_z,
    _normalize_pass2_pairs_with_log_z,
    _relion_f32_fine_posterior_by_class,
    _relion_f32_fine_reconstruction_probs,
    _relion_fine_mstep_prune_mode,
    _relion_fine_parent_execution_order_enabled,
    _relion_joint_winner_take_all_masks,
    _relion_pass2_reconstruction_joint_masks,
    _relion_pass2_reconstruction_pair_probs,
    _relion_pass2_reconstruction_probs,
    _relion_pass2_reconstruction_probs_for_mstep,
    _winner_take_all_bucket_probs,
    _winner_take_all_bucket_probs_from_global_argmax,
    _winner_take_all_pair_probs,
    relion_x_half_f32_fine_posterior_enabled,
)
from recovar.em.sparse_pass2.sparse_pass2_projection_blocks import (
    _compute_sparse_pass2_projections_block,
    _compute_sparse_pass2_windowed_projections_block,
    _projection_kwargs_for_relion_score_window,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _gather_pair_translation_log_prior,
    _gather_projection_cache_rows,
    _relion_cuda_fine_diff2_min,
    _relion_cuda_fine_diff2_to_scores,
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_cuda_fine_global_diff2_min,
    _relion_cuda_fine_log_evidence_offset,
    _relion_cuda_fine_partition_diff2_min_or_inf,
    _relion_powerclass_noise_terms,
    _score_pass2_bucket_gaussian_algebraic,
    _score_pass2_bucket_gaussian_algebraic_components,
    _score_pass2_bucket_gaussian_algebraic_single_cached,
    _score_pass2_bucket_normalized_cc,
    _score_pass2_bucket_normalized_cc_single_cached,
    _score_pass2_bucket_relion_gpu_diff2,
    _score_pass2_bucket_relion_gpu_diff2_from_raw,
    _score_pass2_bucket_relion_gpu_diff2_raw,
    _score_pass2_bucket_relion_gpu_diff2_single_cached_raw,
    _score_pass2_bucket_relion_gpu_normalized_cc,
    _score_pass2_pairs_gaussian_algebraic,
    _score_pass2_pairs_normalized_cc,
    _score_pass2_pairs_relion_gpu_diff2,
    _score_pass2_pairs_relion_gpu_diff2_raw,
)
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
    _relion_cuda_translate_wavg_norm_images,
    _relion_wavg_atomic_triplet_terms,
    _relion_wavg_direct_norm_per_image,
    _relion_wavg_rectangle_triplet_terms,
    _relion_wavg_sequential_triplet_terms,
    _replace_low_shell_noise_with_relion_wavg_direct_residual,
    _replace_untranslated_low_shell_norm_power,
    _select_optional_wavg_exact_pixels,
    _weighted_image_power_shells_and_per_image,
)
from recovar.em.sparse_pass2.sparse_pass2_window import (
    _fine_translation_prior_2d,
    _pass2_half_weights,
    _pass2_projection_budget,
    _pass2_relion_flags,
    _pass2_window_setup,
    _shared_k_class_noise_variance,
    _sparse_pass2_window_setup,
    subtract_projected_reference_from_sparse_mstep_rotation_sums,
    subtract_projected_reference_from_sparse_mstep_sums,
)
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger("recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed")

# Scale sparse pass-2 bucket sizes from physical GPU memory and active score
# pixels. The fused K-class path is launch-bound at 100k/256 unless it uses
# larger chunks; these fractions still scale down on smaller GPUs.
# Compact K-class scoring materializes two complex candidate-by-pixel gathers
# for one class at a time while projections and M-step operands remain live.
# Keep those two gathers within 10% of physical memory.  A K=4 cap that allowed
# 6,587,373 total candidates formed two 8 GiB gathers and requested a 17.04 GiB
# compiled temporary on the 100k/256 fixture after earlier JIT fragmentation.
_EXACT_RAW_DIFF2_CACHE_MAX_BYTES_ENV = "RECOVAR_SPARSE_PASS2_EXACT_RAW_DIFF2_CACHE_MAX_BYTES"
_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES"
_SMALL_BUCKET_THRESHOLD_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD"
_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS"
_SPARSE_KCLASS_COMPACT_BUCKETS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS"
_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS_ENV = "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS"
_SPARSE_KCLASS_GROUP_TIMING_ENV = "RECOVAR_SPARSE_KCLASS_GROUP_TIMING"
_SPARSE_KCLASS_EXECUTION_SIGNATURES_ENV = (
    "RECOVAR_SPARSE_KCLASS_EXECUTION_SIGNATURES"
)
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_ENV = "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS"
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE_ENV = (
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE"
)
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_ENV = "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL"
_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO_ENV = (
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO"
)
_SPARSE_KCLASS_FUSED_NOISE_NORM_ENV = "RECOVAR_SPARSE_KCLASS_FUSED_NOISE_NORM"
_RELION_TRANSLATED_WAVG_NORM_ENV = "RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM"
_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS_ENV = "RECOVAR_SPARSE_PASS2_GROUP_PROGRESS_CHUNKS"
_SPARSE_PASS2_GROUP_PROGRESS_SECONDS_ENV = "RECOVAR_SPARSE_PASS2_GROUP_PROGRESS_SECONDS"
_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES_ENV = (
    "RECOVAR_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES"
)
_DEFAULT_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE = 4096
_DEFAULT_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO = 0.05
_DEFAULT_PASS2_GROUP_PROGRESS_CHUNKS = 1000
_DEFAULT_PASS2_GROUP_PROGRESS_SECONDS = 300
_DEFAULT_KCLASS_RAW_HOST_STAGING_MAX_BYTES = 8 * 1024**3


_cached_score_chunk_log_keys: set[tuple[str, int, int, int]] = set()
_relion_wavg_direct_noise_log_keys: set[int] = set()


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


# ---------------------------------------------------------------------------
# Per-image hypothesis preparation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bucket spec
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scoring + normalization (per-bucket, supports (B, R, T) mask)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main bucketed driver
# ---------------------------------------------------------------------------


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
    (
        use_exact_relion_gaussian,
        use_relion_fine_diff2_fused_ffi,
        use_relion_f32_fine_posterior,
    ) = _pass2_relion_flags(
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
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
    (
        mstep_current_size,
        n_half,
        window_spec_kwargs,
        budget_window_spec,
        device_memory_bytes,
        precision_policy,
    ) = _pass2_window_setup(
        image_shape,
        current_size=current_size,
        reconstruction_current_size=reconstruction_current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_float64_scoring=use_float64_scoring,
    )

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
    fine_translation_prior_2d = _fine_translation_prior_2d(
        translation_log_prior,
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
    (
        projection_complex_dtype,
        projection_budget_pixels,
        max_projected_rotations_per_projection_call,
    ) = _pass2_projection_budget(
        jnp.asarray(mean_for_proj).dtype,
        precision_policy,
        n_half=n_half,
        use_relion_projector=use_relion_projector,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_bytes,
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

    half_weights, half_weights_windowed = _pass2_half_weights(
        image_shape,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
    )
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

    relion_stats = OMITTED
    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=log_evidence,
            best_log_score_per_image=best_log_score,
            max_posterior_per_image=max_posterior,
            rotation_posterior_sums=rotation_posterior_sums,
        )
    return sparse_pass2_result(
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_indices,
        relion_stats=relion_stats,
        # RELION's score-only log partition function rides with the statistics:
        # without them the historical tuple never carried it.
        score_log_z=score_log_z if (return_stats and return_score_log_z) else OMITTED,
        noise_stats=merged_noise_stats if accumulate_noise else OMITTED,
        source_eulers=best_eulers if return_source_eulers else OMITTED,
    )


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
    (
        use_exact_relion_gaussian,
        use_relion_fine_diff2_fused_ffi,
        use_relion_f32_fine_posterior,
    ) = _pass2_relion_flags(
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
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
    (
        mstep_current_size,
        n_half,
        window_spec_kwargs,
        budget_window_spec,
        device_memory_bytes,
        precision_policy,
    ) = _pass2_window_setup(
        image_shape,
        current_size=current_size,
        reconstruction_current_size=reconstruction_current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_float64_scoring=use_float64_scoring,
    )
    winner_take_all = bool(relion_firstiter_winner_take_all)
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
    fine_translation_prior_2d = _fine_translation_prior_2d(
        translation_log_prior,
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
    (
        projection_complex_dtype,
        projection_budget_pixels,
        max_projected_rotations_per_projection_call,
    ) = _pass2_projection_budget(
        jnp.asarray(mean_for_proj_by_class[0]).dtype,
        precision_policy,
        n_half=n_half,
        use_relion_projector=use_relion_projector,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_bytes,
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

    half_weights, half_weights_windowed = _pass2_half_weights(
        image_shape,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
    )
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
                    projection_kwargs = _projection_kwargs_for_relion_score_window(
                        window_spec.projection_kwargs(return_abs2=False),
                        use_relion_projector=use_relion_projector,
                        current_size=current_size,
                    )
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
                    proj_half, proj_for_noise, proj_abs2_for_noise = _gather_projection_cache_rows(
                        cache_score,
                        cache_recon,
                        cache_recon_abs2,
                        jnp.asarray(rotation_indices_np, dtype=jnp.int32),
                    )
            else:
                projection_kwargs = window_spec.projection_kwargs(return_abs2=False if use_window else None)
                projection_kwargs["mask_current_image_disk"] = bool(
                    projection_mask_current_image_disk
                )
                if use_window:
                    projection_kwargs = _projection_kwargs_for_relion_score_window(
                        projection_kwargs,
                        use_relion_projector=use_relion_projector,
                        current_size=current_size,
                    )
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
                        _gather_pair_translation_log_prior(
                            bucket_translation_prior,
                            translation_idx,
                            pair_mask,
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
                    compact_scores = _relion_cuda_fine_diff2_to_scores(
                        compact_raw_diff2,
                        jnp.asarray(
                            compact_arrays["log_prior"],
                            dtype=precision_policy.score_real_dtype,
                        ),
                        _gather_pair_translation_log_prior(
                            bucket_translation_prior,
                            translation_idx,
                            pair_mask,
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
            joint_masks_by_class = []
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
                    joint_masks_by_class.append(pair_mask)
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
                    joint_masks_by_class.append(None)
                    joint_prob_shapes.append(scores_by_class[class_index].shape)
            if winner_take_all:
                flat_joint_masks = _relion_joint_winner_take_all_masks(
                    scores_by_class,
                    joint_masks_by_class,
                )
            elif use_relion_f32_fine_posterior:
                (
                    joint_mstep_masks_by_class,
                    joint_full_probs_by_class,
                    joint_mstep_probs_by_class,
                ) = _relion_f32_fine_posterior_by_class(
                    tuple(scores_by_class),
                    tuple(joint_masks_by_class),
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
                flat_joint_masks = None
            else:
                flat_joint_masks = _relion_pass2_reconstruction_joint_masks(
                    flat_joint_probs_by_class,
                    adaptive_fraction=float(adaptive_fraction),
                )
            if flat_joint_masks is not None:
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
