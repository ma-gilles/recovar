"""Exact per-image local EM engine for RELION-mode local search."""

from __future__ import annotations

import gc
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense.deferred_noise_pack import pack_noise_pixel_capacity
from recovar.em.diagnostics import bpref_diagnostics, vdam_replay
from recovar.em.diagnostics.local_bpref_capture import (
    _bpref_capture_priors,
    _bucket_contains_debug_target,
    _exact_local_bpref_capture_static_kwargs,
    _exact_local_bpref_contribution_capture_for_call,
    _exact_local_bpref_reconstruction_probs_for_capture,
    _filter_buckets_to_debug_targets,
    _maybe_dump_exact_local_bpref_contribution_rows,
)
from recovar.em.diagnostics.local_debug import (
    current_size_matches_request,
    iteration_matches_request,
    maybe_write_debug_fused_posterior_dump,
    maybe_write_debug_noise_component_dump,
    maybe_write_debug_score_dump,
    noise_split_diagnostics_requested,
    parse_debug_fused_posterior_dump_request,
    parse_debug_noise_component_dump_request,
    parse_debug_score_dump_request,
)
from recovar.em.helpers.adjoint import adjoint_slice_volume_maybe_windowed as _adjoint_slice_volume_maybe_windowed
from recovar.em.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.helpers.deterministic_reduce import add_segment_sum
from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.env_flags import parse_env_binary_flag, parse_env_nonnegative_int
from recovar.em.helpers.fourier_window import (
    DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_quantum,
)
from recovar.em.helpers.fourier_window import STABLE_FOURIER_WINDOW_QUANTUM_ENV as STABLE_FOURIER_WINDOW_QUANTUM_ENV
from recovar.em.helpers.half_spectrum import (
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from recovar.em.helpers.half_volume_mstep import (
    crop_relion_x_half_accumulator,
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from recovar.em.helpers.image_shifts import integer_pre_shifts_or_none, tiled_half_image_phase_factors
from recovar.em.helpers.normalization_inputs import prepare_local_normalization_inputs
from recovar.em.helpers.preprocessing import half_translation_phase_table as _half_translation_phase_table
from recovar.em.helpers.preprocessing import (
    relion_preprocess_backend,
    resolve_image_mask_for_half_preprocess,
    uses_relion_cuda_image_preprocessing,
)
from recovar.em.helpers.projection import compute_noise_block as _compute_noise_block
from recovar.em.helpers.projection import compute_norm_residual_per_image as _compute_norm_residual_per_image
from recovar.em.helpers.projection import (
    compute_scale_correction_terms_per_image as _compute_scale_correction_terms_per_image,
)
from recovar.em.helpers.projection import indexed_projection_available as _indexed_projection_available
from recovar.em.helpers.projection import relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask
from recovar.em.helpers.scale_groups import prepare_scale_correction_groups
from recovar.em.helpers.shape_buckets import pad_axis
from recovar.em.helpers.timing import block_until_ready as _block_until_ready
from recovar.em.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.helpers.types import LocalEMResult, make_noise_stats, make_relion_stats
from recovar.em.local import fixed_capacity_local, local_preprocessing
from recovar.em.local import local_projection_cache as projection_cache
from recovar.em.local.fixed_capacity_local import _FixedCapacityLocalExecutionBundle
from recovar.em.local.flat_local_rows import (
    build_dense_to_flat_local_row_lookup,
    map_dense_local_rows_to_flat_rows,
    scatter_flat_local_rows,
)
from recovar.em.local.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_noise_scalar_terms,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.local.local_batch_planning import (
    _exact_local_effective_max_hypotheses_per_microbatch,
    _exact_local_microbatch_env_overridden,
    _exact_local_xhalf_auto_microbatch_boost,
    _exact_local_xhalf_projection_microbatch_cap,
    _exact_local_xhalf_projection_target_row_pixels,
    _exact_local_xhalf_tail_microbatch_cap,
)
from recovar.em.local.local_big_jit import (
    _LocalBigJitDebug,
    _LocalMstepAccumulators,
    _LocalNoiseAccumulators,
    _noise_image_power_shells_and_per_image,
    _partition_uniform_fixed_capacity_calls,
    _prepare_fixed_capacity_local_call,
    _relion_wavg_direct_triplet_shells,
    run_deferred_local_exact_noise_core_jit,
    run_deferred_local_exact_noise_jit,
    run_fixed_capacity_segmented_local_scan,
)
from recovar.em.local.local_bucket_stages import (
    _accumulate_packed_noise_chunk,
    _adjoint_slice_volume_maybe_windowed_row_chunks,
    _build_flat_local_row_argument,
    _build_local_fused_pair_fine_arguments,
    _build_nonzero_reconstruction_pack_indices,
    _build_reconstruction_pack_indices,
    _FixedCapacityWholeScoreCallContext,
    _invoke_local_bucket_big_jit,
    _local_mstep_adjoint_window,
    _local_projection_mode,
    _LocalPostprocessBuffers,
    _noise_norm_capacity,
    _noise_wsum_initial_dtype,
    _packed_bucket_rotations,
    _packed_noise_projection_chunk_rows,
    _packed_reconstruction_rows,
    _pad_local_big_jit_image_axis,
    _plan_flat_local_row_capacities,
    _plan_local_fine_job_capacities,
    _postprocess_fixed_capacity_whole_score_calls,
    _postprocess_local_bucket,
    _project_local_bucket,
    _project_packed_noise_rows,
    _relion_exact_fine_full_to_compact_lookup,
    _reorder_bucket_to_indices,
    _return_local_big_jit_mstep_tensors,
    _unpadded_bucket_rows,
    validate_local_relion_projector_window,
)
from recovar.em.local.local_caches import (
    _all_integer_pre_shifts_or_none,
    _build_local_processed_half_cache,
    _build_local_raw_cache,
    _local_processed_half_cache_enabled,
    _local_raw_cache_enabled,
    _sparse_big_jit_mstep_tensors_memory_gb,
    _validate_native_half_batch,
)
from recovar.em.local.local_layout import (
    LocalHypothesisLayout,
    _local_mstep_rotations,
    _resolve_exact_local_bucket_radix,
    bucket_local_hypothesis_layout,
)
from recovar.em.local.local_physical_grid import (
    _accumulate_relion_physical_particle_grid,
    _accumulate_relion_vdam_physical_particle_grid,
    _source_faithful_bpref_particle_chunk_cap,
    _source_faithful_bpref_particle_chunk_size,
    _source_faithful_bpref_particle_slices,
)
from recovar.em.local.local_score_pass import (
    compute_reconstruction_support,
    compute_reconstruction_support_from_threshold,
    fused_score_normalize_mstep_abs2_on_demand,
    fused_score_normalize_support_abs2_on_demand,
    fused_score_normalize_support_probs_abs2_on_demand,
    fused_score_normalize_support_probs_abs2_with_log_z_on_demand,
    normalize_local_scores,
    normalize_local_scores_float32,
    normalize_local_scores_with_log_z,
    normalize_local_scores_with_log_z_float32,
    score_local_bucket_abs2_on_demand,
    score_local_bucket_abs2_weighted_on_demand,
)
from recovar.em.local.local_timing import (
    LocalBucketProgress,
    _local_timing_profile,
    _LocalTiming,
    _new_local_preprocess_timer,
    _new_local_transfer_timer,
    _prefixed_timer_profile,
)
from recovar.em.refinement.projector_preparation import prepare_local_projector_slab
from recovar.em.relion import relion_ctf
from recovar.em.sparse_pass2 import sparse_pass2_bucketed
from recovar.em.sparse_pass2.sparse_pass2_wavg import _make_relion_wavg_rectangle, _make_stable_relion_wavg_rectangle
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger(__name__)


# Keep the deferred exact-local noise projection chunks small enough for
# low-image-count/high-candidate outlier cases that otherwise fragment H100
# memory. This cap is total projected row-pixels across the active image batch,
# i.e. about a 512 MB complex64 projection temporary before JAX overhead.
EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES_ENV = (
    "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES"
)
EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS_ENV = (
    "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS"
)
EXACT_LOCAL_SOURCE_BPREF_LAUNCH_SERIAL_ROTATIONS_ENV = (
    "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_LAUNCH_SERIAL_ROTATIONS"
)
EXACT_LOCAL_DEFER_PACKED_MSTEP_ENV = "RECOVAR_EXACT_LOCAL_DEFER_PACKED_MSTEP"
EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV = "RECOVAR_EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP"
EXACT_LOCAL_PROJECTOR_CAPACITY_ENV = "RECOVAR_EXACT_LOCAL_PROJECTOR_CAPACITY"
EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV = "RECOVAR_EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY"
EXACT_LOCAL_NOISE_STABLE_CORE_ENV = "RECOVAR_EXACT_LOCAL_NOISE_STABLE_CORE"
EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV = "RECOVAR_EXACT_LOCAL_NOISE_NATIVE_RESIDUAL"
EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV = "RECOVAR_EXACT_LOCAL_NOISE_NORM_CAPACITY"
EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV = "RECOVAR_EXACT_LOCAL_NOISE_PIXEL_CAPACITY"
EXACT_LOCAL_HOST_PLAN_PACK_ENV = "RECOVAR_EXACT_LOCAL_HOST_PLAN_PACK"
EXACT_LOCAL_HOST_PUBLICATION_ENV = "RECOVAR_EXACT_LOCAL_HOST_PUBLICATION"
EXACT_LOCAL_BPREF_TRANSACTION_ENV = "RECOVAR_EXACT_LOCAL_BPREF_TRANSACTION"
EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV = "RECOVAR_EXACT_LOCAL_BPREF_PARTICLE_CAPACITY"
EXACT_LOCAL_BPREF_CUDA_PACKING_ENV = "RECOVAR_EXACT_LOCAL_BPREF_CUDA_PACKING"
EXACT_LOCAL_HOST_PLAN_CUDA_ENV = "RECOVAR_EXACT_LOCAL_HOST_PLAN_CUDA"
EXACT_LOCAL_SKIP_DEFERRED_ZERO_NORM_ENV = "RECOVAR_EXACT_LOCAL_SKIP_DEFERRED_ZERO_NORM"
LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV = "RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT"
LOCAL_SCORE_DUMP_OPERANDS_ENV = "RECOVAR_LOCAL_SCORE_DUMP_OPERANDS"
LOCAL_SCORE_DUMP_TARGET_ONLY_ENV = "RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY"
EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS_ENV = "RECOVAR_EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
# Disabled by default: on the 50k/256 local-search target this cache made the
# iteration slower by precomputing more spectra than the bucket schedule reuses.
# Upper bound for the extra M-step tensors materialized by the sparse big-JIT
# hybrid path. This path still packs rows before backprojection; the cap only
# guards the temporary fused summed/ctf tensor outputs.
EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION = 0.25


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def run_local_em_exact(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str,
    *,
    image_batch_size: int,
    rotation_block_size: int,
    current_size: int | None,
    reconstruction_current_size: int | None = None,
    accumulate_noise: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    score_with_masked_images: bool = True,
    half_spectrum_scoring: bool = False,
    relion_exact_score_translation: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    use_float64_projections: bool = False,
    projection_relion_texture_interp: bool = False,
    projection_relion_acc_double_floorf_quirk: bool = False,
    projection_force_jax: bool = False,
    projection_mask_current_image_disk: bool = True,
    relion_exact_bpref_operands: bool = False,
    relion_exact_fine_diff2: bool = False,
    relion_wavg_sequential_cuda: bool | None = None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    recon_exact_radius: bool = True,
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    reconstruction_group_ids: np.ndarray | None = None,
    reconstruction_group_count: int | None = None,
    scale_correction_group_count: int | None = None,
    scale_correction_data_vs_prior: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    mstep_subtract_ctf_projection: bool = False,
    mstep_relion_x_half: bool = False,
    host_accumulator_finalize: bool = False,
    host_stats_publication: bool = False,
    return_half_volume_accumulators: bool = False,
    return_profile: bool = False,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
    max_hypotheses_per_microbatch: int | None = None,
    reconstruct_significant_only: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    debug_iteration: int | None = None,
    debug_pass_label: str | None = None,
    return_best_pose_details: bool = False,
    normalization_log_z: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    normalization_log_evidence: np.ndarray | None = None,
    normalization_max_posterior: np.ndarray | None = None,
    translation_prior_centers: np.ndarray | None = None,
    unify_local_bucket_sizes: bool | None = None,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    preserve_bpref_particle_order: bool = False,
    stable_fourier_window_shapes: bool = False,
    stats_use_reconstruction_probs: bool = False,
    relion_f32_fine_posterior: bool = False,
    include_unweighted_norm_high_shell: bool = True,
    unweighted_high_shell_image_power: bool = False,
    source_faithful_spectrum_norm: bool = False,
    reconstruction_probability_threshold: np.ndarray | None = None,
    return_reconstruction_probability_values: bool = False,
    return_reconstruction_sample_indices: bool = False,
    return_significant_counts: bool = False,
    score_only: bool = False,
    _fixed_capacity_bundle: _FixedCapacityLocalExecutionBundle | None = None,
    _fixed_capacity_enabled: bool = False,
    _fixed_capacity_class_count: int | None = None,
    _fixed_capacity_whole_boundary_enabled: bool = False,
    _flat_local_rows_enabled: bool = False,
    _stable_flat_row_capacity_enabled: bool = False,
    _packed_local_projection_enabled: bool = False,
    fused_pair_fine_score: bool = False,
    _defer_packed_vdam_enabled: bool = False,
    _packed_final_noise_enabled: bool = False,
) -> LocalEMResult:
    """Run exact local EM over per-image local hypothesis sets.

    ``debug_pass_label`` is diagnostic-only: it is appended verbatim to
    ``RECOVAR_LOCAL_SCORE_DUMP_*`` filenames (see
    ``local_debug.maybe_write_debug_score_dump``). Callers that invoke this
    function more than once per iteration for the *same* image/current_size/
    debug_iteration (e.g. local search's pass-1 "parent" probe followed by
    its pass-2 fine call) must pass distinct labels, or the later call's
    dump silently overwrites the earlier one at the same path.

    Returns named accumulators, assignments and statistics. Optional pose,
    noise, profile and significant-count fields are None when disabled.
    Requesting reconstruction probabilities or sample IDs also enables the
    profile that carries those captures, as with explicit ``return_profile``.

    ``unweighted_high_shell_image_power`` selects InitialModel's unweighted
    per-particle noise-spectrum tail. The default preserves ordinary local
    EM's posterior-weighted spectrum. Norm correction has its own unchanged
    high-shell ownership policy.
    """

    resolved_exact_local_bucket_radix = _resolve_exact_local_bucket_radix(exact_local_bucket_radix)
    score_only = bool(score_only)
    fixed_capacity_enabled = bool(_fixed_capacity_enabled)
    fixed_capacity_whole_boundary_enabled = bool(
        _fixed_capacity_whole_boundary_enabled
    )
    flat_local_rows_enabled = bool(_flat_local_rows_enabled)
    stable_flat_row_capacity_enabled = bool(
        _stable_flat_row_capacity_enabled
    )
    packed_local_projection_enabled = bool(_packed_local_projection_enabled)
    fused_pair_fine_score_enabled = bool(fused_pair_fine_score)
    defer_packed_vdam_enabled = bool(_defer_packed_vdam_enabled)
    stable_fourier_window_shapes = bool(stable_fourier_window_shapes)
    projector_capacity_enabled = parse_env_binary_flag(EXACT_LOCAL_PROJECTOR_CAPACITY_ENV)
    if projector_capacity_enabled and not stable_fourier_window_shapes:
        raise ValueError("projector capacity requires stable exact-local VDAM Fourier windows")
    bpref_projector_capacity_enabled = parse_env_binary_flag(EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV)
    if bpref_projector_capacity_enabled and not projector_capacity_enabled:
        raise ValueError("BPref projector capacity requires the shared local projector capacity")
    packed_final_noise_enabled = bool(_packed_final_noise_enabled)
    bpref_transaction_enabled = parse_env_binary_flag(EXACT_LOCAL_BPREF_TRANSACTION_ENV)
    bpref_particle_capacity_enabled = parse_env_binary_flag(EXACT_LOCAL_BPREF_PARTICLE_CAPACITY_ENV)
    bpref_cuda_packing_enabled = parse_env_binary_flag(EXACT_LOCAL_BPREF_CUDA_PACKING_ENV)
    if bpref_cuda_packing_enabled and not bpref_particle_capacity_enabled:
        raise ValueError("CUDA BPref packing requires stable particle capacity")
    if bpref_particle_capacity_enabled and (
        not bpref_transaction_enabled or bpref_projector_capacity_enabled
    ):
        raise ValueError("BPref particle capacity requires transactions and no BPref projector capacity")
    if bpref_transaction_enabled and not (
        defer_packed_vdam_enabled and packed_final_noise_enabled
    ):
        raise ValueError("BPref transactions require deferred packed final-noise execution")
    host_plan_pack_enabled = parse_env_binary_flag(EXACT_LOCAL_HOST_PLAN_PACK_ENV)
    host_plan_cuda_enabled = parse_env_binary_flag(EXACT_LOCAL_HOST_PLAN_CUDA_ENV)
    if host_plan_cuda_enabled and not host_plan_pack_enabled:
        raise ValueError("CUDA host-plan packing requires host-plan packing")
    host_publication_enabled = parse_env_binary_flag(EXACT_LOCAL_HOST_PUBLICATION_ENV)
    if type(host_stats_publication) is not bool:
        raise TypeError("host_stats_publication must be a bool")
    if host_stats_publication and not host_accumulator_finalize:
        raise ValueError("host statistics publication requires host accumulator finalization")
    skip_deferred_zero_norm = parse_env_binary_flag(EXACT_LOCAL_SKIP_DEFERRED_ZERO_NORM_ENV)
    if host_publication_enabled and not defer_packed_vdam_enabled:
        raise ValueError("host publication requires deferred packed VDAM execution")
    if host_plan_pack_enabled and not (
        defer_packed_vdam_enabled and packed_final_noise_enabled
    ):
        raise ValueError("host-plan packing requires deferred packed final-noise execution")
    noise_stable_core_enabled = parse_env_binary_flag(EXACT_LOCAL_NOISE_STABLE_CORE_ENV)
    noise_native_residual_enabled = parse_env_binary_flag(EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV)
    if noise_native_residual_enabled and not (
        stable_fourier_window_shapes
        and defer_packed_vdam_enabled
        and packed_final_noise_enabled
        and accumulate_noise
    ):
        raise ValueError("native noise residual requires stable deferred packed final-noise accumulation")
    if noise_stable_core_enabled and not (
        stable_fourier_window_shapes and packed_final_noise_enabled
    ):
        raise ValueError("separate noise core requires stable packed final-noise execution")
    noise_norm_capacity_enabled = parse_env_binary_flag(EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV)
    if noise_norm_capacity_enabled and not (
        stable_fourier_window_shapes
        and defer_packed_vdam_enabled
        and packed_final_noise_enabled
        and accumulate_noise
    ):
        raise ValueError("noise norm capacity requires stable deferred packed final-noise accumulation")
    noise_pixel_capacity_enabled = parse_env_binary_flag(EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV)
    noise_pixel_cuda_enabled = parse_env_binary_flag('RECOVAR_EXACT_LOCAL_NOISE_PIXEL_CUDA')
    if noise_pixel_cuda_enabled and not noise_pixel_capacity_enabled:
        raise ValueError("noise pixel CUDA packing requires noise pixel capacity")
    if noise_pixel_capacity_enabled and not noise_norm_capacity_enabled:
        raise ValueError("noise pixel capacity requires noise norm capacity")
    if fixed_capacity_whole_boundary_enabled and not fixed_capacity_enabled:
        raise ValueError(
            "fixed-capacity whole-local boundary requires fixed-capacity execution"
        )
    use_relion_f32_fine_posterior = bool(
        relion_f32_fine_posterior
        and mstep_relion_x_half
        and reconstruct_significant_only
        and not score_only
    )
    if relion_f32_fine_posterior and not use_relion_f32_fine_posterior:
        raise ValueError(
            "RELION float32 fine posterior requires a significant-only "
            "RELION x-half M-step"
        )
    unweighted_high_shell_image_power = bool(unweighted_high_shell_image_power)
    include_unweighted_norm_high_shell = bool(include_unweighted_norm_high_shell)
    source_faithful_spectrum_norm = bool(source_faithful_spectrum_norm)
    relion_exact_score_translation = bool(relion_exact_score_translation)
    relion_exact_bpref_operands = bool(relion_exact_bpref_operands)
    relion_exact_fine_diff2 = bool(relion_exact_fine_diff2)
    if flat_local_rows_enabled and not relion_exact_fine_diff2:
        raise ValueError(
            "flat local rows require exact RELION fine diff2"
        )
    if stable_flat_row_capacity_enabled and not flat_local_rows_enabled:
        raise ValueError("stable flat-row capacity requires flat local rows")
    if packed_local_projection_enabled and not flat_local_rows_enabled:
        raise ValueError(
            "packed local projection requires flat local rows"
        )
    if fused_pair_fine_score_enabled and not (
        relion_exact_fine_diff2 and flat_local_rows_enabled
    ):
        raise ValueError(
            "fused-pair fine scoring requires exact RELION fine diff2 and flat local rows"
        )
    if defer_packed_vdam_enabled and not packed_local_projection_enabled:
        raise ValueError(
            "deferred packed VDAM requires packed local projection"
        )
    if packed_final_noise_enabled and not defer_packed_vdam_enabled:
        raise ValueError(
            "packed final-support VDAM noise requires deferred packed VDAM"
        )
    if relion_wavg_sequential_cuda is not None:
        relion_wavg_sequential_cuda = bool(relion_wavg_sequential_cuda)
    preserve_bpref_particle_order = bool(preserve_bpref_particle_order)
    source_faithful_bpref = bool(
        preserve_bpref_particle_order and relion_exact_bpref_operands
    )
    if stable_fourier_window_shapes and not (
        relion_exact_fine_diff2
        and relion_exact_bpref_operands
        and relion_wavg_sequential_cuda is True
        and accumulate_noise
        and mstep_relion_x_half
        and source_faithful_bpref
        and relion_projector_half is not None
        and not disable_adjoint_y
        and not disable_adjoint_ctf
        and not score_only
    ):
        raise ValueError(
            "stable Fourier-window shapes require the K=1 exact-local VDAM "
            "topology: exact fine scoring, CUDA Wavg, source-faithful x-half "
            "BPref, a RELION projector, noise accumulation, and a full M-step"
        )
    if stable_fourier_window_shapes and os.environ.get(
        "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY", ""
    ).strip():
        raise ValueError(
            "stable Fourier-window shapes do not support external VDAM host replay"
        )
    if defer_packed_vdam_enabled and not (
        source_faithful_bpref
        and mstep_subtract_ctf_projection
        and mstep_relion_x_half
        and reconstruct_significant_only
        and relion_projector_half is not None
        and not disable_adjoint_y
        and not disable_adjoint_ctf
    ):
        raise ValueError(
            "deferred packed VDAM requires source-faithful significant-only "
            "RELION residual backprojection"
        )
    reconstruction_group_ids_np = None
    resolved_reconstruction_group_count = 1
    if reconstruction_group_ids is not None:
        reconstruction_group_ids_np = np.asarray(reconstruction_group_ids)
        if reconstruction_group_ids_np.dtype != np.int32:
            raise TypeError("reconstruction_group_ids must be int32")
        if reconstruction_group_ids_np.shape != (int(local_layout.n_images),):
            raise ValueError(
                "reconstruction_group_ids must match the local-layout image axis"
            )
        if reconstruction_group_count is None:
            raise ValueError(
                "reconstruction_group_count is required with reconstruction_group_ids"
            )
        resolved_reconstruction_group_count = int(reconstruction_group_count)
        if resolved_reconstruction_group_count <= 0:
            raise ValueError("reconstruction_group_count must be positive")
        if np.any(reconstruction_group_ids_np < 0) or np.any(
            reconstruction_group_ids_np >= resolved_reconstruction_group_count
        ):
            raise ValueError("reconstruction_group_ids contains an out-of-range group")
        if score_only:
            raise ValueError("grouped reconstruction is not supported in score-only mode")
        if not source_faithful_bpref:
            raise ValueError(
                "grouped reconstruction requires source-faithful RELION BPref accumulation"
            )
    elif reconstruction_group_count not in (None, 1):
        raise ValueError(
            "reconstruction_group_ids is required when reconstruction_group_count is not one"
        )
    if preserve_bpref_particle_order and not mstep_relion_x_half:
        raise ValueError("BPref particle-order preservation requires the RELION x-half M-step")
    if preserve_bpref_particle_order and not relion_exact_bpref_operands:
        raise ValueError("BPref particle-order preservation requires exact RELION BPref operands")
    if source_faithful_bpref and (disable_adjoint_y or disable_adjoint_ctf):
        raise ValueError(
            "source-faithful BPref accumulation requires both data and weight adjoints"
        )
    if relion_exact_score_translation and not half_spectrum_scoring:
        raise ValueError("exact RELION score translation requires half_spectrum_scoring=True")
    if relion_exact_fine_diff2 and not relion_exact_bpref_operands:
        raise ValueError("exact RELION fine diff2 requires exact BPref operands")
    if score_only:
        if not (disable_adjoint_y and disable_adjoint_ctf):
            raise ValueError("score_only exact-local EM requires both adjoints disabled")
        if accumulate_noise:
            raise ValueError("score_only exact-local EM does not support noise accumulation")
        if mstep_subtract_ctf_projection:
            raise ValueError("score_only exact-local EM does not support residual M-step subtraction")
        if return_half_volume_accumulators:
            raise ValueError("score_only exact-local EM does not return half-volume accumulators")

    return_profile = bool(
        return_profile or return_reconstruction_probability_values or return_reconstruction_sample_indices
    )
    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    logical_current_size = int(H) if current_size is None else int(current_size)
    mstep_current_size = (
        logical_current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    n_half = H * (W // 2 + 1)
    if stable_fourier_window_shapes and mstep_current_size != logical_current_size:
        raise ValueError(
            "stable Fourier-window VDAM currently requires identical score and "
            "reconstruction current sizes"
        )
    resolved_window_quantum = (
        stable_fourier_window_quantum()
        if stable_fourier_window_shapes
        else DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
    )
    stable_window_plan = make_stable_fourier_window_shape_plan(
        image_shape,
        logical_current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        enabled=stable_fourier_window_shapes,
        quantum=resolved_window_quantum,
        square=square_window,
        recon_exact_radius=bool(recon_exact_radius),
    )
    physical_current_size = stable_window_plan.physical_current_size
    physical_mstep_current_size = (
        stable_window_plan.physical_reconstruction_current_size
    )
    # Every non-full stable window uses a runtime logical bound, even when its
    # logical size equals the physical class boundary. This keeps one JIT
    # topology for all members of a physical capacity class.
    stable_window_active = bool(
        stable_fourier_window_shapes and stable_window_plan.logical_spec.use_window
    )
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    class_log_prior = float(class_log_prior)
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )
    if group_ids_np is None:
        n_scale_groups = 0
    normalization_inputs = prepare_local_normalization_inputs(
        n_images=n_images,
        normalization_log_z=normalization_log_z,
        normalization_log_evidence=normalization_log_evidence,
        normalization_max_posterior=normalization_max_posterior,
        reconstruction_probability_threshold=reconstruction_probability_threshold,
    )
    normalization_log_z_np = normalization_inputs.log_z
    normalization_log_evidence_np = normalization_inputs.log_evidence
    normalization_max_posterior_np = normalization_inputs.max_posterior
    reconstruction_probability_threshold_np = normalization_inputs.reconstruction_threshold
    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=local_layout.translation_grid.shape[1],
    )
    (
        debug_score_dump_dir,
        debug_score_dump_targets,
        debug_score_dump_current_sizes,
        debug_score_dump_iterations,
    ) = parse_debug_score_dump_request()
    (
        debug_fused_posterior_dump_dir,
        debug_fused_posterior_dump_targets,
        debug_fused_posterior_dump_current_sizes,
        debug_fused_posterior_dump_iterations,
    ) = parse_debug_fused_posterior_dump_request()
    (
        debug_noise_dump_dir,
        debug_noise_dump_targets,
        debug_noise_dump_current_sizes,
        debug_noise_dump_iterations,
    ) = parse_debug_noise_component_dump_request()
    debug_score_dump_filter_matches = (
        debug_score_dump_dir is not None
        and current_size_matches_request(debug_score_dump_current_sizes, current_size)
        and iteration_matches_request(debug_score_dump_iterations, debug_iteration)
    )
    debug_fused_posterior_dump_filter_matches = (
        debug_fused_posterior_dump_dir is not None
        and current_size_matches_request(debug_fused_posterior_dump_current_sizes, current_size)
        and iteration_matches_request(debug_fused_posterior_dump_iterations, debug_iteration)
    )
    debug_fused_posterior_dump_scores = bool(
        debug_fused_posterior_dump_filter_matches
        and _env_flag("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES")
    )
    debug_noise_dump_filter_matches = (
        debug_noise_dump_dir is not None
        and current_size_matches_request(debug_noise_dump_current_sizes, current_size)
        and iteration_matches_request(debug_noise_dump_iterations, debug_iteration)
    )
    bpref_contribution_capture_active = _exact_local_bpref_contribution_capture_for_call(
        current_size=current_size,
        debug_iteration=debug_iteration,
        score_only=score_only,
        mstep_relion_x_half=mstep_relion_x_half,
    )
    debug_score_dump_operands = bool(
        debug_score_dump_filter_matches
        and _env_flag(LOCAL_SCORE_DUMP_OPERANDS_ENV)
    )
    debug_score_dump_force_split = bool(
        debug_score_dump_filter_matches
        and _env_flag(LOCAL_SCORE_DUMP_FORCE_SPLIT_ENV)
    )
    debug_score_dump_big_jit = bool(debug_score_dump_filter_matches and not debug_score_dump_force_split)
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    # BigJIT preprocesses raw images itself and never calls process_fn. Exclude
    # that dataset-bound method from its static cache key so equivalent dataset
    # wrappers share an executable. Keep the full config for the split path.
    big_jit_config = config.replace(process_fn=None)
    if return_half_volume_accumulators and mstep_relion_x_half:
        raise ValueError("return_half_volume_accumulators only supports native half-volume accumulators")

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=mstep_current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = volume_shape

    precision_policy = DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        use_float64_normalization=use_float64_normalization,
    )
    mean_for_proj = precision_policy.cast_projection_volume(mean_for_proj)

    if mstep_relion_x_half:
        # RELION BPref::initZeros(current_size) sizes the accumulator from the
        # iteration r_max.  The reconstruction boundary then crops the output
        # back to ``volume_shape``.
        logical_recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=mstep_current_size,
        )
        recon_volume_shape = relion_backprojector_volume_shape(
            volume_shape,
            reconstruction_padding_factor,
            current_size=physical_mstep_current_size,
        )
    elif reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
        logical_recon_volume_shape = recon_volume_shape
    else:
        recon_volume_shape = volume_shape
        logical_recon_volume_shape = recon_volume_shape
    if score_only:
        logger.info("Exact local score-only: M-step accumulators disabled")
    elif mstep_relion_x_half:
        logger.info(
            "Exact local M-step: using RELION x-half BPref-layout backprojection shape=%s",
            recon_volume_shape,
        )
    else:
        logger.info("Exact local M-step: using native half-volume backprojection")
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    score_only_accumulator_size = 1 if score_only else recon_volume_size

    window_spec = (
        stable_window_plan.packed_physical_spec()
        if stable_window_active
        else stable_window_plan.logical_spec
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    if relion_exact_fine_diff2:
        if stable_window_active:
            logical_lookup = _relion_exact_fine_full_to_compact_lookup(
                image_shape,
                logical_current_size,
                n_half,
                stable_window_plan.logical_spec,
            )
            relion_fine_full_to_compact = pad_axis(
                logical_lookup,
                0,
                stable_window_plan.physical_rectangle_pixels,
                value=-1,
            )
        else:
            relion_fine_full_to_compact = _relion_exact_fine_full_to_compact_lookup(
                image_shape,
                logical_current_size,
                n_half,
                window_spec,
            )
    else:
        relion_fine_full_to_compact = np.zeros(1, dtype=np.int32)
    recon_window_indices = window_spec.recon_indices
    mstep_recon_window_indices, mstep_adjoint_max_r = _local_mstep_adjoint_window(
        image_shape,
        n_half,
        physical_mstep_current_size,
        use_window=use_window,
        recon_window_indices=recon_window_indices,
        mstep_relion_x_half=bool(mstep_relion_x_half),
    )
    capture_static_kwargs = (
        _exact_local_bpref_capture_static_kwargs(
            experiment_dataset=experiment_dataset,
            score_with_masked_images=score_with_masked_images,
            disc_type=disc_type,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            mstep_relion_x_half=mstep_relion_x_half,
            mstep_adjoint_max_r=mstep_adjoint_max_r,
            mstep_recon_window_indices=mstep_recon_window_indices,
            image_shape=image_shape,
            recon_volume_shape=recon_volume_shape,
        )
        if bpref_contribution_capture_active
        else None
    )
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    projection_kwargs["relion_texture_interp"] = projection_relion_texture_interp
    projection_kwargs["relion_acc_double_floorf_quirk"] = projection_relion_acc_double_floorf_quirk
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)
    projection_mode = _local_projection_mode(window_spec, projection_kwargs, relion_projector_half)
    if relion_projector_half is not None:
        validate_local_relion_projector_window(window_spec, image_shape)

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    norm_half_weights = make_half_image_weights(image_shape)
    half_weights_windowed = window_spec.score_values(half_weights)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=bool(mstep_relion_x_half),
    )
    accumulator_shape = (
        (resolved_reconstruction_group_count, score_only_accumulator_size)
        if reconstruction_group_ids_np is not None
        else (score_only_accumulator_size,)
    )
    Ft_y = jnp.zeros(accumulator_shape, dtype=recon_y_accum_dtype)
    Ft_ctf = jnp.zeros(accumulator_shape, dtype=recon_ctf_accum_dtype)
    hard_assignment = np.empty(n_images, dtype=np.int64)
    log_evidence_per_image = np.empty(n_images, dtype=precision_policy.score_real_dtype)
    best_log_score_per_image = np.empty(n_images, dtype=precision_policy.score_real_dtype)
    max_posterior_per_image = np.empty(n_images, dtype=precision_policy.score_real_dtype)
    significant_counts = np.empty(n_images, dtype=np.int32) if return_significant_counts else None
    rotation_posterior_sums = np.zeros(int(local_layout.n_global_rotations), dtype=np.float64)
    best_pose_rotations = (
        np.empty((n_images, 3, 3), dtype=precision_policy.score_real_dtype) if return_best_pose_details else None
    )
    best_pose_translations = (
        np.empty((n_images, local_layout.translation_grid.shape[1]), dtype=precision_policy.score_real_dtype)
        if return_best_pose_details
        else None
    )
    best_pose_eulers_deg = (
        np.empty((n_images, 3), dtype=np.float64)
        if return_best_pose_details and local_layout.source_eulers_flat is not None
        else None
    )
    best_pose_rotation_ids = np.empty(n_images, dtype=np.int64) if return_best_pose_details else None

    noise_wsum = None
    noise_img_power = None
    noise_norm_correction = None
    noise_a2 = None
    noise_xa = None
    noise_scale_xa = None
    noise_scale_aa = None
    noise_sigma2_offset = jnp.asarray(0.0, dtype=precision_policy.score_real_dtype)
    noise_sumw = jnp.asarray(0.0, dtype=precision_policy.score_real_dtype)
    return_noise_split = noise_split_diagnostics_requested()
    fixed_capacity_unsupported_diagnostics = tuple(
        name
        for name, requested in (
            ("return_profile", return_profile),
            ("return_best_pose_details", return_best_pose_details),
            ("return_reconstruction_probability_values", return_reconstruction_probability_values),
            ("return_reconstruction_sample_indices", return_reconstruction_sample_indices),
            ("return_significant_counts", return_significant_counts),
            ("normalization_log_z", normalization_log_z_np is not None),
            ("normalization_log_evidence", normalization_log_evidence_np is not None),
            ("normalization_max_posterior", normalization_max_posterior_np is not None),
            ("reconstruction_probability_threshold", reconstruction_probability_threshold_np is not None),
            ("score_dump", debug_score_dump_filter_matches),
            ("fused_posterior_dump", debug_fused_posterior_dump_filter_matches),
            ("noise_dump", debug_noise_dump_filter_matches),
            ("noise_split", return_noise_split),
            ("bpref_contribution_capture", bpref_contribution_capture_active),
        )
        if requested
    )
    fixed_capacity_mstep_requested = bool(
        mstep_subtract_ctf_projection
        or mstep_relion_x_half
        or host_accumulator_finalize
        or return_half_volume_accumulators
        or preserve_bpref_particle_order
    )
    require_materialized_recon_projection = bool(
        mstep_subtract_ctf_projection
        or debug_noise_dump_filter_matches
        or return_noise_split
    )
    can_defer_local_noise_projection = (
        relion_projector_half is None
        and not bool(projection_kwargs.get("relion_texture_interp", False))
        and not bool(projection_kwargs.get("force_jax", False))
        and _indexed_projection_available()
    )
    defer_local_noise_projection = (
        not require_materialized_recon_projection
        and window_spec.use_window
        and can_defer_local_noise_projection
    )
    need_local_recon_projection = require_materialized_recon_projection or (
        accumulate_noise and not defer_local_noise_projection
    )
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        if use_window:
            shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
                shell_indices_half,
                image_shape,
                logical_current_size,
                (
                    stable_window_plan.logical_spec.score_indices
                    if stable_window_active
                    else window_indices
                ),
            )
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        norm_unweighted_shell_cutoff = int(logical_current_size // 2)
        noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
        if stable_window_active:
            logical_recon_mask = jnp.arange(window_spec.n_recon) < int(
                stable_window_plan.logical_reconstruction_pixels
            )
            shell_indices_noise = jnp.where(
                logical_recon_mask,
                shell_indices_noise,
                jnp.asarray(-1, dtype=jnp.int32),
            )
            noise_variance_for_noise = jnp.where(
                logical_recon_mask,
                noise_variance_for_noise,
                jnp.asarray(0, dtype=noise_variance_for_noise.dtype),
            )
        scale_correction_pixel_mask = _relion_scale_correction_pixel_mask(
            scale_correction_data_vs_prior,
            shell_indices_noise,
            n_shells=n_shells,
        )
        # Direct Wavg replaces the cutoff shell with its ordered float64
        # reduction, so the first bucket otherwise promotes this carry and
        # creates a one-off float32 big-JIT ABI.  Starting from float64 zero is
        # numerically identical and matches every subsequent bucket.
        noise_wsum = jnp.zeros(
            n_shells,
            dtype=_noise_wsum_initial_dtype(
                relion_exact_fine_diff2=relion_exact_fine_diff2,
                use_window=use_window,
            ) if not use_float64_scoring else precision_policy.score_real_dtype,
        )
        noise_img_power = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
        noise_norm_correction = jnp.zeros(
            _noise_norm_capacity(n_images, enabled=noise_norm_capacity_enabled),
            dtype=jnp.float64 if source_faithful_spectrum_norm else precision_policy.score_real_dtype,
        )
        noise_a2 = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
        noise_xa = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
        if group_ids_np is not None:
            noise_scale_xa = jnp.zeros(n_scale_groups, dtype=precision_policy.score_real_dtype)
            noise_scale_aa = jnp.zeros(n_scale_groups, dtype=precision_policy.score_real_dtype)

    default_fused_score_mstep = max_significants is None or int(max_significants) <= 0
    fused_score_mstep_enabled = default_fused_score_mstep
    timing = _LocalTiming()
    raw_cache_enabled = False
    processed_half_cache_enabled = False
    preprocess_profile = _new_local_preprocess_timer()
    transfer_profile = _new_local_transfer_timer()
    big_jit_bucket_count = 0
    sparse_big_jit_bucket_count = 0
    big_jit_debug_bucket_count = 0
    sparse_adjoint_chunk_count = 0
    sparse_adjoint_target_rows = parse_env_nonnegative_int(EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS_ENV) or 0
    total_local_rotations = int(local_layout.total_local_rotations)
    logged_deferred_mstep_chunking = False
    logged_deferred_noise_projection_chunking = False
    logged_cached_noise_projection_chunking = False
    logged_sparse_big_jit_deferred_fallback = False
    collect_profile_stats = bool(return_profile or reconstruct_significant_only)
    seen_global_rotations = (
        np.zeros(rotation_posterior_sums.shape[0], dtype=bool)
        if return_profile and rotation_posterior_sums.size
        else np.zeros(0, dtype=bool)
    )
    seen_nonzero_global_rotations = np.zeros_like(seen_global_rotations)
    seen_reconstruction_global_rotations = np.zeros_like(seen_global_rotations)
    total_padded_rotations = 0
    total_planned_padded_rotations = 0
    chunk_sizes = []
    chunk_padded_image_counts = []
    chunk_planned_padded_image_counts = []
    chunk_local_rotations = []
    chunk_padded_rotations = []
    chunk_flat_score_rows = []
    chunk_fused_pair_capacities = []
    chunk_fused_pair_counts = []
    chunk_fused_pair_dense_capacities = []
    chunk_planned_padded_rotations = []
    chunk_unique_rotations = []
    chunk_nonzero_posterior_rows = []
    chunk_reconstruction_rows = []
    chunk_significant_samples = []
    n_chunks = 0
    local_total_hypotheses = 0
    total_significant_samples = 0
    total_reconstruction_rows = 0
    total_flat_score_rows = 0
    total_fused_pair_candidates = 0
    total_fused_pair_capacity = 0
    total_fused_pair_dense_capacity = 0
    total_packed_final_noise_rows = 0
    reconstruction_sample_indices_by_image = (
        [np.zeros(0, dtype=np.int64) for _ in range(n_images)] if return_reconstruction_sample_indices else None
    )
    postprocess_buffers = _LocalPostprocessBuffers(
        hard_assignment=hard_assignment,
        log_evidence_per_image=log_evidence_per_image,
        best_log_score_per_image=best_log_score_per_image,
        max_posterior_per_image=max_posterior_per_image,
        rotation_posterior_sums=rotation_posterior_sums,
        transfer_profile=transfer_profile,
        chunk_nonzero_posterior_rows=chunk_nonzero_posterior_rows,
        chunk_significant_samples=chunk_significant_samples,
        chunk_reconstruction_rows=chunk_reconstruction_rows,
        seen_global_rotations=seen_global_rotations,
        seen_nonzero_global_rotations=seen_nonzero_global_rotations,
        seen_reconstruction_global_rotations=seen_reconstruction_global_rotations,
        significant_counts=significant_counts,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        best_pose_eulers_deg=best_pose_eulers_deg,
        reconstruction_sample_indices_by_image=reconstruction_sample_indices_by_image,
    )
    reconstruction_probability_values_by_image = (
        [[] for _ in range(n_images)] if return_reconstruction_probability_values else None
    )

    def _collect_reconstruction_probability_values(image_indices, posterior_probs):
        """Collect unpruned positive posterior values for global support thresholding."""

        if reconstruction_probability_values_by_image is None:
            return
        image_indices_np = np.asarray(image_indices, dtype=np.int32)
        probs_np = np.asarray(posterior_probs, dtype=precision_policy.score_real_dtype).reshape(
            len(image_indices_np), -1
        )
        for row, image_index in enumerate(image_indices_np):
            values = probs_np[row]
            values = values[values > 0.0]
            if values.size:
                reconstruction_probability_values_by_image[int(image_index)].append(values.copy())

    # The cap model already accounts for the active score/reconstruction
    # windows and the x-half M-step row footprint, but RELION projector x-half
    # buckets at 256 OOMed at both 2x and 1.25x in c180 probes. Keep the default
    # conservative and allow explicit experiments through the x-half env knob.
    allow_microbatch_auto_boost = True
    xhalf_bpref_mstep = bool(relion_projector_half is not None and mstep_relion_x_half and not score_only)
    xhalf_auto_microbatch_boost = _exact_local_xhalf_auto_microbatch_boost() if xhalf_bpref_mstep else None
    xhalf_full_bpref_mstep = bool(
        xhalf_bpref_mstep and int(recon_volume_shape[0]) >= (2 * int(image_shape[0]) + 1)
    )
    if xhalf_bpref_mstep and max_hypotheses_per_microbatch is None and not _exact_local_microbatch_env_overridden():
        bpreftype = "full-BPref" if xhalf_full_bpref_mstep else "current-size BPref"
        logger.info(
            "Exact local RELION x-half %s M-step: using conservative microbatch cap "
            "(image_shape=%s, recon_volume_shape=%s)",
            bpreftype,
            tuple(int(x) for x in image_shape),
            tuple(int(x) for x in recon_volume_shape),
        )
    max_hypotheses_per_microbatch = _exact_local_effective_max_hypotheses_per_microbatch(
        max_hypotheses_per_microbatch,
        n_windowed,
        n_trans=n_trans,
        n_recon_windowed=window_spec.n_recon,
        local_layout=local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        exact_local_bucket_radix=resolved_exact_local_bucket_radix,
        allow_auto_boost=allow_microbatch_auto_boost,
        auto_boost_factor=xhalf_auto_microbatch_boost,
        allow_high_memory_default=not xhalf_bpref_mstep,
        score_only=score_only,
    )
    if xhalf_bpref_mstep:
        uncapped_hypotheses_per_microbatch = int(max_hypotheses_per_microbatch)
        max_hypotheses_per_microbatch = _exact_local_xhalf_tail_microbatch_cap(
            uncapped_hypotheses_per_microbatch,
            local_layout,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
        )
        if max_hypotheses_per_microbatch < uncapped_hypotheses_per_microbatch:
            logger.info(
                "Exact local RELION x-half tail microbatch cap: %d -> %d "
                "(max_local_rotations=%d, planned_image_batch=%d, planned_rotation_block=%d)",
                uncapped_hypotheses_per_microbatch,
                int(max_hypotheses_per_microbatch),
                int(np.max(np.asarray(local_layout.rotation_counts), initial=0)),
                int(image_batch_size),
                int(rotation_block_size),
            )
        tail_capped_hypotheses_per_microbatch = int(max_hypotheses_per_microbatch)
        max_hypotheses_per_microbatch = _exact_local_xhalf_projection_microbatch_cap(
            tail_capped_hypotheses_per_microbatch,
            local_layout,
            n_projection_pixels=int(window_spec.n_projection),
            rotation_block_size=rotation_block_size,
            exact_local_bucket_radix=resolved_exact_local_bucket_radix,
        )
        if max_hypotheses_per_microbatch < tail_capped_hypotheses_per_microbatch:
            logger.info(
                "Exact local RELION x-half projection microbatch cap: %d -> %d "
                "(projection_pixels=%d target_row_pixels=%d)",
                tail_capped_hypotheses_per_microbatch,
                int(max_hypotheses_per_microbatch),
                int(window_spec.n_projection),
                int(_exact_local_xhalf_projection_target_row_pixels()),
            )
    bucket_build_t0 = time.time()
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        unify_bucket_sizes=unify_local_bucket_sizes,
        preserve_image_order=source_faithful_bpref,
        exact_local_bucket_radix=resolved_exact_local_bucket_radix,
        consecutive_mixed_bucket_size=consecutive_mixed_bucket_size,
    )
    timing.bucket_build_s += time.time() - bucket_build_t0
    debug_target_only_targets: set[int] = set()
    if debug_score_dump_filter_matches:
        debug_target_only_targets.update(debug_score_dump_targets)
    if debug_fused_posterior_dump_filter_matches:
        debug_target_only_targets.update(debug_fused_posterior_dump_targets)
    debug_score_dump_target_only = bool(
        score_only
        and debug_target_only_targets
        # ``score_only`` also implements the science-critical local parent
        # pass that supplies pass-2 support.  Filtering it merely because a
        # dump target is configured makes the diagnostic change refinement
        # results.  Keep target-only execution as an explicit opt-in for
        # standalone diagnostics.
        and _env_flag(LOCAL_SCORE_DUMP_TARGET_ONLY_ENV)
    )
    debug_target_only_original_bucket_count = len(bucket_specs)
    debug_target_only_original_image_count = int(
        sum(int(bucket.image_indices.shape[0]) for bucket in bucket_specs)
    )
    if debug_score_dump_target_only:
        filter_t0 = time.time()
        bucket_specs = _filter_buckets_to_debug_targets(
            experiment_dataset,
            bucket_specs,
            debug_target_only_targets,
        )
        timing.bucket_build_s += time.time() - filter_t0
        total_local_rotations = int(
            sum(int(np.sum(bucket.actual_rotation_counts, dtype=np.int64)) for bucket in bucket_specs)
        )
        target_only_images = int(sum(int(bucket.image_indices.shape[0]) for bucket in bucket_specs))
        logger.info(
            "Exact local debug target-only: keeping %d/%d buckets and %d/%d images "
            "for requested original ids %s; unset %s to retain the full score-only computation",
            len(bucket_specs),
            debug_target_only_original_bucket_count,
            target_only_images,
            debug_target_only_original_image_count,
            sorted(int(target) for target in debug_target_only_targets),
            LOCAL_SCORE_DUMP_TARGET_ONLY_ENV,
        )
    if fixed_capacity_enabled:
        if not isinstance(_fixed_capacity_bundle, _FixedCapacityLocalExecutionBundle):
            raise ValueError("fixed-capacity score-only execution requires a bound execution bundle")
        if not bucket_specs:
            raise ValueError("fixed-capacity score-only execution requires authoritative local calls")
        if int(_fixed_capacity_bundle.plan.valid_call_count) != len(bucket_specs):
            raise ValueError(
                "fixed-capacity score-only execution requires every authoritative local call"
            )
    flat_local_row_capacities = (
        _plan_flat_local_row_capacities(
            bucket_specs,
            rotation_block_size=rotation_block_size,
            exact_local_bucket_radix=resolved_exact_local_bucket_radix,
            stable_rectangular_capacity=stable_flat_row_capacity_enabled,
        )
        if flat_local_rows_enabled
        else {}
    )
    fine_job_capacities = (
        _plan_local_fine_job_capacities(bucket_specs)
        if fused_pair_fine_score_enabled
        else {}
    )
    if bucket_specs:
        bucket_rotation_counts = np.asarray(
            [int(bucket.bucket_rotation_count) for bucket in bucket_specs],
            dtype=np.int64,
        )
        bucket_image_counts = np.asarray(
            [int(bucket.image_indices.shape[0]) for bucket in bucket_specs],
            dtype=np.int64,
        )
        unique_bucket_counts, unique_bucket_freq = np.unique(bucket_rotation_counts, return_counts=True)
        top_bucket_counts = sorted(
            (
                (int(bucket_count), int(freq))
                for bucket_count, freq in zip(unique_bucket_counts, unique_bucket_freq)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
        logger.info(
            "Exact local bucketing: %d images -> %d buckets "
            "(bucket_size min/med/mean/max=%d/%d/%.1f/%d, images_per_bucket med/max=%d/%d, "
            "top_bucket_counts=%s; max_hypotheses_per_microbatch=%d, n_score_pixels=%d, "
            "n_recon_pixels=%d, n_trans=%d, score_only=%s, relion_x_half_mstep=%s)",
            n_images,
            len(bucket_specs),
            int(np.min(bucket_rotation_counts)),
            int(np.median(bucket_rotation_counts)),
            float(np.mean(bucket_rotation_counts)),
            int(np.max(bucket_rotation_counts)),
            int(np.median(bucket_image_counts)),
            int(np.max(bucket_image_counts)),
            top_bucket_counts,
            int(max_hypotheses_per_microbatch),
            int(n_windowed),
            int(window_spec.n_recon),
            int(n_trans),
            bool(score_only),
            bool(mstep_relion_x_half),
        )
    local_progress = LocalBucketProgress(
        bucket_specs, total_local_rotations=total_local_rotations, n_trans=n_trans,
    )

    raw_batch_cache = None
    ctf_param_cache = None
    processed_half_cache = None

    phase_t0 = time.time()
    translation_phases_half = _half_translation_phase_table(
        local_layout.translation_grid,
        image_shape,
        dtype=precision_policy.score_real_dtype,
    )
    relion_score_translation_angles = (
        sparse_pass2_bucketed._relion_cuda_score_translation_angles_if_available(
            local_layout.translation_grid,
            image_shape,
            enabled=relion_exact_score_translation,
            dtype=np.float64 if use_float64_scoring else np.float32,
        )
    )
    if return_profile:
        _block_until_ready(translation_phases_half)
    translation_phase_time = time.time() - phase_t0
    timing.preprocess_s += translation_phase_time
    preprocess_profile["translation_phase_s"] += translation_phase_time

    big_jit_image_mask_arg, big_jit_mask_mode = resolve_image_mask_for_half_preprocess(
        experiment_dataset,
        image_shape,
        require_mask=score_with_masked_images,
    )
    big_jit_image_mask_arg = jnp.asarray(big_jit_image_mask_arg)
    relion_cuda_preprocess_radius = 0.0
    relion_cuda_preprocess_cosine_width = 0.0
    if relion_exact_bpref_operands:
        if not uses_relion_cuda_image_preprocessing(experiment_dataset):
            raise ValueError("exact RELION BPref operands require RELION CUDA image preprocessing")
        preprocess_params = getattr(relion_preprocess_backend(experiment_dataset), "_relion_image_mask_params", None)
        if preprocess_params is None:
            raise ValueError("RELION CUDA image preprocessing requires explicit image-mask parameters")
        pixel_size, particle_diameter_ang, cosine_width = preprocess_params
        relion_cuda_preprocess_radius = float(particle_diameter_ang) / (2.0 * float(pixel_size))
        relion_cuda_preprocess_cosine_width = float(cosine_width)

    big_jit_window_indices_arg = window_spec.score_or_full_indices(n_half)
    big_jit_recon_window_indices_arg = window_spec.recon_or_full_indices(n_half)
    stable_bpref_dense_positions = None
    if relion_exact_fine_diff2 and accumulate_noise and use_window:
        if stable_window_active:
            relion_wavg_rectangle = _make_stable_relion_wavg_rectangle(
                image_shape,
                stable_window_plan,
            )
            stable_bpref_dense_positions = relion_wavg_rectangle.exact_positions
        else:
            relion_wavg_rectangle = _make_relion_wavg_rectangle(
                image_shape,
                logical_current_size,
                big_jit_recon_window_indices_arg,
            )
        big_jit_relion_wavg_rectangle_indices_arg = relion_wavg_rectangle.centered_indices
        big_jit_relion_wavg_exact_positions_arg = relion_wavg_rectangle.exact_positions
        big_jit_relion_wavg_rectangle_shell_indices_arg = relion_wavg_rectangle.shell_indices
    else:
        big_jit_relion_wavg_rectangle_indices_arg = np.zeros(1, dtype=np.int32)
        big_jit_relion_wavg_exact_positions_arg = np.zeros(1, dtype=np.int32)
        big_jit_relion_wavg_rectangle_shell_indices_arg = np.zeros(1, dtype=np.int32)
    big_jit_mstep_recon_window_indices_arg = (
        mstep_recon_window_indices if mstep_relion_x_half else big_jit_recon_window_indices_arg
    )
    disabled_noise_wsum = jnp.zeros(1, dtype=precision_policy.score_real_dtype)
    disabled_noise_img_power = jnp.zeros(1, dtype=precision_policy.score_real_dtype)
    disabled_noise_a2 = jnp.zeros(1, dtype=precision_policy.score_real_dtype)
    disabled_noise_xa = jnp.zeros(1, dtype=precision_policy.score_real_dtype)
    disabled_noise_scale = jnp.zeros(1, dtype=precision_policy.score_real_dtype)
    disabled_group_ids = jnp.zeros(1, dtype=jnp.int32)
    disabled_noise_shell_indices = jnp.zeros(n_half, dtype=jnp.int32)

    local_support_rows = int(np.sum(local_layout.rotation_counts))
    significant_backprojection_candidate = (
        reconstruct_significant_only
        and n_images > 0
        and local_support_rows >= int(np.ceil(max(n_images, 1) / EXACT_LOCAL_BIG_JIT_MIN_SIGNIFICANT_ROW_FRACTION))
    )
    use_relion_projector = relion_projector_half is not None
    compact_relion_projector_big_jit = bool(use_relion_projector and window_spec.use_window)
    disable_big_jit_buckets = os.environ.get("RECOVAR_DISABLE_LOCAL_BIG_JIT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    processed_half_cache_preferred = (
        image_pre_shifts is None or _all_integer_pre_shifts_or_none(image_pre_shifts, n_images) is not None
    ) and _local_processed_half_cache_enabled(
        n_images,
        n_half,
        np.complex64,
        store_recon_half=bool(score_with_masked_images),
    )
    use_big_jit_buckets = (
        not disable_big_jit_buckets
        and not return_reconstruction_probability_values
        and not (accumulate_noise and debug_noise_dump_dir is not None)
        and not processed_half_cache_preferred
    )
    if projector_capacity_enabled and use_relion_projector and logical_current_size == H and not use_window:
        if bpref_projector_capacity_enabled:
            raise ValueError("BPref projector capacity is not supported for full-box local projection")
        # The full-box projector already occupies its maximum capacity. Keep
        # the existing full-spectrum projection path and its pixel ordering;
        # scientific window/DC/noise behavior must not change to admit this
        # optimization, which only combines cropped projector shapes.
        projector_capacity_enabled = False
    if projector_capacity_enabled and not (
        stable_window_active and use_big_jit_buckets and compact_relion_projector_big_jit
        and flat_local_rows_enabled and packed_local_projection_enabled
        and not projection_force_jax and not fixed_capacity_enabled
    ):
        raise ValueError("projector capacity requires the stable packed compact local BigJIT VDAM route")
    if projector_capacity_enabled and projection_cache.read_nonnegative_float_env(
        projection_cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, projection_cache.EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB
    ) > 0:
        raise ValueError("projector capacity cannot be combined with the local projection cache")
    if bpref_projector_capacity_enabled and (not source_faithful_bpref or score_only):
        raise ValueError("BPref projector capacity requires the source-faithful VDAM accumulator route")
    bpref_transaction_queue = None
    if bpref_transaction_enabled:
        if (
            not source_faithful_bpref or score_only
            or not use_big_jit_buckets or fixed_capacity_enabled
        ):
            raise ValueError(
                "BPref transactions require the source-faithful packed local BigJIT VDAM route"
            )
        from recovar.em.helpers.bpref_transaction import BprefTransactionQueue

        bpref_transaction_queue = BprefTransactionQueue(
            stable_particle_capacity=bpref_particle_capacity_enabled,
            cuda_packing=bpref_cuda_packing_enabled,
        )
    if fixed_capacity_enabled and not use_big_jit_buckets:
        raise ValueError(
            "fixed-capacity score-only execution requires the mature local big-JIT bucket path",
        )
    if flat_local_rows_enabled and not use_big_jit_buckets:
        raise ValueError("flat local rows require the mature local big-JIT bucket path")
    if relion_exact_fine_diff2 and not use_big_jit_buckets:
        raise ValueError("exact RELION fine diff2 requires the local big-JIT bucket path")
    mean_for_proj_big_jit = mean_for_proj
    projection_half_volume_big_jit = False
    relion_projector_half_big_jit = jnp.zeros((1, 1, 1), dtype=jnp.complex64)
    relion_projector_r_max_big_jit = 0
    big_jit_projection_pixel_indices_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_projection_score_take_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_projection_recon_take_arg = jnp.zeros((1,), dtype=jnp.int32)
    big_jit_relion_projector_output_size = 0
    relion_projection_cache = projection_cache.disabled_cache()
    projection_cache_plan = projection_cache.ProjectionCachePlan(bucket_specs)
    relion_projection_cache_group_cursor = 0
    relion_projection_cache_groups_built = 0
    relion_projection_cache_total_build_s = 0.0
    relion_projection_cache_max_rows = 0
    relion_projection_cache_max_estimated_gb = 0.0
    relion_projection_cache_id_map_rows = 0
    source_vdam_projector_full = None
    big_jit_projection_pixel_count = int(window_spec.n_projection)
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half_big_jit = prepare_local_projector_slab(
            relion_projector_half, path_label="local RELION projector big-JIT path",
        )
        relion_projector_r_max_big_jit = int(relion_projector_r_max)
        if compact_relion_projector_big_jit:
            big_jit_relion_projector_output_size = int(window_spec.relion_projector_output_size() or 0)
            if score_only:
                big_jit_projection_pixel_indices_arg = jnp.asarray(window_spec.score_indices, dtype=jnp.int32)
                big_jit_projection_score_take_arg = jnp.arange(window_spec.n_score, dtype=jnp.int32)
                big_jit_projection_recon_take_arg = jnp.zeros((1,), dtype=jnp.int32)
                big_jit_projection_pixel_count = int(window_spec.n_score)
            else:
                big_jit_projection_pixel_indices_arg = jnp.asarray(window_spec.projection_indices, dtype=jnp.int32)
                big_jit_projection_score_take_arg = jnp.asarray(window_spec.score_projection_take, dtype=jnp.int32)
                big_jit_projection_recon_take_arg = jnp.asarray(window_spec.recon_projection_take, dtype=jnp.int32)
        if source_faithful_bpref and not score_only and not bpref_projector_capacity_enabled:
            from recovar.em.helpers.projection import relion_projector_half_to_texture_full

            source_vdam_projector_full = relion_projector_half_to_texture_full(
                relion_projector_half_big_jit
            )
    # Keep logical inputs above unchanged for BPref and the legacy projection cache.
    local_projection_half_arg = relion_projector_half_big_jit
    local_projection_static_radius = relion_projector_r_max_big_jit
    local_projection_runtime_radius = None
    if projector_capacity_enabled:
        from recovar.em.helpers.projection import prepare_relion_projector_capacity

        local_projection_half_arg, local_projection_runtime_radius = prepare_relion_projector_capacity(
            relion_projector_half_big_jit,
            r_max=relion_projector_r_max_big_jit,
            physical_size=physical_current_size,
            padding_factor=projection_padding_factor,
        )
        local_projection_static_radius = 0
    source_vdam_projector_static_radius = relion_projector_r_max_big_jit
    source_vdam_projector_runtime_radius = None
    if bpref_projector_capacity_enabled:
        # Reuse the exact same physical half slab and device radius as BigJIT.
        # No logical full-cube materialization is needed for this BPref ABI.
        source_vdam_projector_full = local_projection_half_arg
        source_vdam_projector_static_radius = 0
        source_vdam_projector_runtime_radius = local_projection_runtime_radius
    if (
        use_big_jit_buckets
        and compact_relion_projector_big_jit
        and not score_only
    ):
        projection_cache_plan = projection_cache.plan_cache(
            bucket_specs, n_projection_pixels=big_jit_projection_pixel_count,
        )
        bucket_specs = projection_cache_plan.buckets
        if projection_cache_plan.groups:
            valid_layout_ids = np.asarray(local_layout.rotation_ids_flat, dtype=np.int64)
            valid_layout_ids = valid_layout_ids[valid_layout_ids >= 0]
            relion_projection_cache_id_map_rows = int(np.unique(valid_layout_ids).size)
            projection_cache_plan.log_enabled(relion_projection_cache_id_map_rows)
    if use_big_jit_buckets and not use_relion_projector and not projection_relion_texture_interp:
        mean_for_proj_big_jit = fourier_transform_utils.full_volume_to_half_volume(
            mean_for_proj,
            proj_volume_shape,
        ).reshape(-1)
        projection_half_volume_big_jit = True

    can_use_processed_half_cache = not use_big_jit_buckets and processed_half_cache_preferred
    if can_use_processed_half_cache:
        processed_half_cache_t0 = time.time()
        processed_half_cache = _build_local_processed_half_cache(
            experiment_dataset,
            n_images,
            score_with_masked_images=bool(score_with_masked_images),
            image_pre_shifts=image_pre_shifts,
        )
        cache_elapsed = time.time() - processed_half_cache_t0
        timing.preprocess_s += cache_elapsed
        preprocess_profile["cache_build_s"] += cache_elapsed
        processed_half_cache_enabled = True
    else:
        raw_cache_enabled = _local_raw_cache_enabled(
            n_images,
            image_shape,
            getattr(experiment_dataset, "dtype", np.float32),
        )
        if raw_cache_enabled:
            raw_cache_t0 = time.time()
            raw_batch_cache, ctf_param_cache = _build_local_raw_cache(experiment_dataset, n_images)
            timing.raw_cache_build_s = time.time() - raw_cache_t0

    fixed_capacity_whole_call_program = []
    fixed_capacity_whole_call_contexts = []
    fixed_capacity_whole_initial_carry = None
    fixed_capacity_whole_static_options = None
    fixed_capacity_whole_preparation_s = 0.0
    for bucket_index, bucket in enumerate(bucket_specs):
        if (
            projection_cache_plan.groups
            and relion_projection_cache_group_cursor < len(projection_cache_plan.groups)
            and bucket_index == projection_cache_plan.groups[relion_projection_cache_group_cursor][0]
        ):
            group_start, group_stop, _ = projection_cache_plan.groups[relion_projection_cache_group_cursor]
            relion_projection_cache = projection_cache.build_cache(
                bucket_specs[group_start:group_stop],
                relion_projector_half_big_jit,
                image_shape=image_shape,
                n_projection_pixels=big_jit_projection_pixel_count,
                relion_projector_r_max=int(relion_projector_r_max_big_jit),
                projection_padding_factor=int(projection_padding_factor),
                projection_relion_texture_interp=projection_relion_texture_interp,
                projection_relion_acc_double_floorf_quirk=projection_relion_acc_double_floorf_quirk,
                projection_mask_current_image_disk=bool(projection_mask_current_image_disk),
                projection_pixel_indices=big_jit_projection_pixel_indices_arg,
                projector_output_size=int(big_jit_relion_projector_output_size),
                cache_row_capacity=int(projection_cache_plan.capacity_rows),
                group_index=relion_projection_cache_group_cursor,
                n_groups=len(projection_cache_plan.groups),
            )
            relion_projection_cache_group_cursor += 1
            relion_projection_cache_groups_built += int(relion_projection_cache.enabled)
            relion_projection_cache_total_build_s += float(relion_projection_cache.build_s)
            relion_projection_cache_max_rows = max(relion_projection_cache_max_rows, int(relion_projection_cache.row_count))
            relion_projection_cache_max_estimated_gb = max(
                relion_projection_cache_max_estimated_gb,
                float(relion_projection_cache.estimated_gb),
            )
        n_chunks += 1
        if collect_profile_stats:
            chunk_sizes.append(int(bucket.image_indices.shape[0]))
            planned_padded_image_count = max(
                int(bucket.image_indices.shape[0]),
                int(getattr(bucket, "bucket_image_count", bucket.image_indices.shape[0])),
            )
            chunk_planned_padded_image_counts.append(planned_padded_image_count)
            chunk_local_rotations.append(int(np.sum(bucket.actual_rotation_counts)))
            chunk_planned_padded_rotations.append(
                int(planned_padded_image_count * bucket.bucket_rotation_count)
            )
            bucket_valid_rotation_ids = np.asarray(bucket.local_rotation_ids, dtype=np.int64)[
                np.asarray(bucket.local_rotation_mask, dtype=bool)
            ]
            chunk_unique_rotations.append(int(np.unique(bucket_valid_rotation_ids).shape[0]))
            total_planned_padded_rotations += int(
                planned_padded_image_count * bucket.bucket_rotation_count
            )
            local_total_hypotheses += int(np.sum(bucket.actual_rotation_counts) * n_trans)
        fetch_t0 = time.time()
        fixed_capacity_call_view = None
        if fixed_capacity_enabled:
            fixed_capacity_call_view = fixed_capacity_local._select_fixed_capacity_score_only_view(
                _fixed_capacity_bundle,
                bucket,
                call_index=bucket_index,
                n_classes=_fixed_capacity_class_count,
                class_log_prior=class_log_prior,
                image_pre_shifts=image_pre_shifts,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                score_only=score_only,
                disable_adjoint_y=disable_adjoint_y,
                disable_adjoint_ctf=disable_adjoint_ctf,
                accumulate_noise=accumulate_noise,
                mstep_requested=fixed_capacity_mstep_requested,
                unsupported_diagnostics=fixed_capacity_unsupported_diagnostics,
                enabled=True,
            )
            batch_data, ctf_params, fetched_indices = fixed_capacity_local._fetch_and_validate_fixed_capacity_call_operands(
                experiment_dataset,
                fixed_capacity_call_view,
                bucket,
            )
            bucket = fixed_capacity_call_view.bucket
            bucket_image_indices = np.asarray(fetched_indices, dtype=np.int32)
        elif raw_batch_cache is None:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            if processed_half_cache is None:
                batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, bucket.image_indices)
            else:
                batch_data = None
                ctf_params = processed_half_cache.ctf_params[bucket_image_indices]
                fetched_indices = bucket_image_indices
        else:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            batch_data = raw_batch_cache[bucket_image_indices]
            ctf_params = ctf_param_cache[bucket_image_indices]
            fetched_indices = bucket_image_indices
        timing.batch_fetch_s += time.time() - fetch_t0
        bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
        batch_size = int(bucket.image_indices.shape[0])
        bucket_reconstruction_group_ids = (
            None
            if reconstruction_group_ids_np is None
            else reconstruction_group_ids_np[
                np.asarray(bucket.image_indices, dtype=np.int64)
            ]
        )
        debug_fused_posterior_bucket_matches = (
            debug_fused_posterior_dump_filter_matches
            and _bucket_contains_debug_target(
                experiment_dataset,
                bucket.image_indices,
                debug_fused_posterior_dump_targets,
            )
        )
        debug_score_dump_bucket_matches = (
            debug_score_dump_filter_matches
            and _bucket_contains_debug_target(
                experiment_dataset,
                bucket.image_indices,
                debug_score_dump_targets,
            )
        )
        use_big_jit_buckets_for_bucket = bool(
            use_big_jit_buckets
            and not (debug_score_dump_force_split and debug_score_dump_bucket_matches)
        )
        need_local_recon_projection_for_bucket = bool(
            need_local_recon_projection
            or (
                debug_score_dump_bucket_matches
                and debug_score_dump_operands
            )
        )
        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                bucket.image_indices,
                batch_size=batch_size,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                local_layout.translation_grid,
                centers,
                experiment_dataset.voxel_size,
            )

        sparse_big_jit_backprojection = False
        sparse_big_jit_mstep_estimated_gb = 0.0
        sparse_big_jit_mstep_cap_gb = 0.0
        if use_big_jit_buckets_for_bucket and significant_backprojection_candidate:
            sparse_big_jit_mstep_estimated_gb, sparse_big_jit_mstep_cap_gb = (
                _sparse_big_jit_mstep_tensors_memory_gb(
                    image_count=max(batch_size, int(getattr(bucket, "bucket_image_count", batch_size))),
                    rotation_count=int(bucket.bucket_rotation_count),
                    n_recon_windowed=window_spec.n_recon,
                    use_float64_scoring=use_float64_scoring,
                )
            )
            sparse_big_jit_backprojection = (
                sparse_big_jit_mstep_cap_gb > 0.0
                and sparse_big_jit_mstep_estimated_gb <= sparse_big_jit_mstep_cap_gb
            )
        can_defer_big_jit_backprojection = (
            use_big_jit_buckets_for_bucket
            and significant_backprojection_candidate
            and not score_only
            and (
                not mstep_subtract_ctf_projection
                or defer_packed_vdam_enabled
            )
            and (not disable_adjoint_y or not disable_adjoint_ctf or accumulate_noise)
        )
        force_deferred_big_jit_backprojection = bool(
            defer_packed_vdam_enabled
            or _env_flag(EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV)
        )
        deferred_big_jit_backprojection = (
            can_defer_big_jit_backprojection
            and (
                force_deferred_big_jit_backprojection
                or not sparse_big_jit_backprojection
            )
        )
        if (
            deferred_big_jit_backprojection
            and not sparse_big_jit_backprojection
            and not logged_sparse_big_jit_deferred_fallback
        ):
            logger.info(
                "Exact local big-JIT using deferred packed M-step because sparse M-step tensors "
                "estimate %.3f GB exceeds cap %.3f GB "
                "(images=%d, bucket_rot=%d, n_recon_pixels=%d, forced=%s)",
                float(sparse_big_jit_mstep_estimated_gb),
                float(sparse_big_jit_mstep_cap_gb),
                int(max(batch_size, int(getattr(bucket, "bucket_image_count", batch_size)))),
                int(bucket.bucket_rotation_count),
                int(window_spec.n_recon),
                bool(force_deferred_big_jit_backprojection),
            )
            logged_sparse_big_jit_deferred_fallback = True

        execute_big_jit_bucket = bool(
            use_big_jit_buckets_for_bucket
            and (
                not significant_backprojection_candidate
                or sparse_big_jit_backprojection
                or deferred_big_jit_backprojection
                or score_only
            )
        )
        if fixed_capacity_call_view is not None and not execute_big_jit_bucket:
            raise ValueError(
                f"fixed-capacity call {fixed_capacity_call_view.call_index} did not reach "
                "the mature local big-JIT bucket path"
            )
        if flat_local_rows_enabled and not execute_big_jit_bucket:
            raise ValueError("flat local rows did not reach the mature local big-JIT bucket path")
        if collect_profile_stats:
            executed_padded_image_count = (
                max(
                    batch_size,
                    int(getattr(bucket, "bucket_image_count", batch_size)),
                )
                if execute_big_jit_bucket
                else batch_size
            )
            executed_padded_rotations = int(
                executed_padded_image_count * bucket.bucket_rotation_count
            )
            chunk_padded_image_counts.append(executed_padded_image_count)
            chunk_padded_rotations.append(executed_padded_rotations)
            total_padded_rotations += executed_padded_rotations
            flat_score_rows = (
                int(
                    flat_local_row_capacities[
                        (
                            int(executed_padded_image_count),
                            int(bucket.bucket_rotation_count),
                        )
                    ]
                )
                if flat_local_rows_enabled
                else executed_padded_rotations
            )
            chunk_flat_score_rows.append(flat_score_rows)
            total_flat_score_rows += flat_score_rows

        if execute_big_jit_bucket:
            if batch_data is None:
                raise RuntimeError("exact local big-JIT requires fetched native image batches")
            big_jit_t0 = time.time()
            unpadded_bucket = bucket
            unpadded_batch_size = batch_size
            _validate_native_half_batch(batch_data, image_shape)
            integer_pre_shifts = integer_pre_shifts_or_none(
                image_pre_shifts,
                np.asarray(unpadded_bucket.image_indices, dtype=np.int32),
                batch=batch_data,
            )
            bucket, batch_data, ctf_params, valid_image_mask, batch_size = _pad_local_big_jit_image_axis(
                bucket,
                batch_data,
                ctf_params,
            )
            if fixed_capacity_call_view is not None:
                fixed_capacity_local._validate_fixed_capacity_padded_call(
                    fixed_capacity_call_view,
                    bucket,
                    batch_data,
                    ctf_params,
                    valid_image_mask,
                    batch_size,
                )
            bucket_image_indices = np.asarray(unpadded_bucket.image_indices, dtype=np.int32)
            if relion_exact_bpref_operands:
                ctf_rfloat_unpadded = np.asarray(
                    relion_ctf._relion_exact_ctf_half_from_source_star_host(
                        experiment_dataset,
                        bucket_image_indices,
                        image_shape,
                    ),
                    dtype=np.float64,
                )
                ctf_rfloat_half_arg = jnp.asarray(
                    pad_axis(
                        ctf_rfloat_unpadded,
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=jnp.float64,
                )
                inverse_noise_rfloat_cast_np = np.reciprocal(
                    np.asarray(noise_variance_half, dtype=np.float64)
                ).astype(np.float32)
                ctf_squared_rfloat = ctf_rfloat_unpadded * ctf_rfloat_unpadded
                corr_img_rfloat_square_unpadded = (
                    inverse_noise_rfloat_cast_np[None, :].astype(np.float64)
                    * ctf_squared_rfloat
                ).astype(np.float32)
                inverse_noise_rfloat_cast_arg = jnp.asarray(
                    inverse_noise_rfloat_cast_np,
                    dtype=jnp.float32,
                )
                corr_img_rfloat_square_arg = jnp.asarray(
                    pad_axis(
                        corr_img_rfloat_square_unpadded,
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=jnp.float32,
                )
            else:
                ctf_rfloat_half_arg = jnp.zeros(
                    (batch_size, n_half),
                    dtype=jnp.float64,
                )
                inverse_noise_rfloat_cast_arg = jnp.zeros(
                    (n_half,),
                    dtype=jnp.float32,
                )
                corr_img_rfloat_square_arg = jnp.zeros(
                    (batch_size, n_half),
                    dtype=jnp.float32,
                )
            apply_integer_pre_shift = integer_pre_shifts is not None
            if apply_integer_pre_shift:
                integer_pre_shifts_arg = jnp.asarray(
                    pad_axis(integer_pre_shifts, 0, batch_size, value=0),
                    dtype=jnp.int32,
                )
                fourier_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=precision_policy.score_real_dtype)
                apply_fourier_pre_shift = False
            elif image_pre_shifts is not None:
                integer_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.int32)
                fourier_pre_shifts_arg = jnp.asarray(
                    pad_axis(
                        np.asarray(image_pre_shifts)[bucket_image_indices],
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=precision_policy.score_real_dtype,
                )
                apply_fourier_pre_shift = True
            else:
                integer_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=jnp.int32)
                fourier_pre_shifts_arg = jnp.zeros((batch_size, 2), dtype=precision_policy.score_real_dtype)
                apply_fourier_pre_shift = False

            image_corrections_arg = (
                jnp.asarray(
                    pad_axis(
                        np.asarray(image_corrections)[bucket_image_indices],
                        0,
                        batch_size,
                        value=1,
                    ),
                    dtype=precision_policy.score_real_dtype,
                )
                if image_corrections is not None
                else jnp.ones(batch_size, dtype=precision_policy.score_real_dtype)
            )
            scale_corrections_arg = (
                jnp.asarray(
                    pad_axis(
                        np.asarray(scale_corrections)[bucket_image_indices],
                        0,
                        batch_size,
                        value=1,
                    ),
                    dtype=precision_policy.score_real_dtype,
                )
                if scale_corrections is not None
                else jnp.ones(batch_size, dtype=precision_policy.score_real_dtype)
            )
            image_only_corrections_arg = (
                image_corrections_arg / scale_corrections_arg
                if image_corrections is not None
                else jnp.ones(batch_size, dtype=precision_policy.score_real_dtype)
            )
            translation_sqdist_arg = (
                jnp.asarray(
                    pad_axis(translation_sqdist_ang, 0, batch_size, value=0),
                    dtype=precision_policy.score_real_dtype,
                )
                if translation_sqdist_ang is not None
                else jnp.zeros((batch_size, n_trans), dtype=precision_policy.score_real_dtype)
            )
            sample_mask_arg = (
                None
                if bucket.local_sample_mask is None
                else jnp.asarray(bucket.local_sample_mask)
            )
            normalization_log_z_arg = (
                jnp.asarray(
                    pad_axis(normalization_log_z_np[bucket_image_indices], 0, batch_size, value=0),
                    dtype=(jnp.float64 if use_float64_normalization else jnp.float32),
                )
                if normalization_log_z_np is not None
                else jnp.zeros(batch_size, dtype=(jnp.float64 if use_float64_normalization else jnp.float32))
            )
            normalization_log_evidence_arg = (
                jnp.asarray(
                    pad_axis(normalization_log_evidence_np[bucket_image_indices], 0, batch_size, value=0),
                    dtype=(jnp.float64 if use_float64_normalization else jnp.float32),
                )
                if normalization_log_evidence_np is not None
                else jnp.zeros(batch_size, dtype=(jnp.float64 if use_float64_normalization else jnp.float32))
            )
            normalization_max_posterior_arg = (
                jnp.asarray(
                    pad_axis(
                        normalization_max_posterior_np[bucket_image_indices],
                        0,
                        batch_size,
                        value=0,
                    ),
                    dtype=(jnp.float64 if use_float64_normalization else jnp.float32),
                )
                if normalization_max_posterior_np is not None
                else jnp.zeros(batch_size, dtype=jnp.float32)
            )
            local_rotation_log_prior_arg = jnp.asarray(bucket.local_rotation_log_prior)
            if class_log_prior != 0.0:
                local_rotation_log_prior_arg = local_rotation_log_prior_arg + jnp.asarray(
                    class_log_prior,
                    dtype=local_rotation_log_prior_arg.dtype,
                )
            if accumulate_noise:
                noise_wsum_arg = noise_wsum
                noise_img_power_arg = noise_img_power
                noise_a2_arg = noise_a2
                noise_xa_arg = noise_xa
                noise_scale_xa_arg = noise_scale_xa if noise_scale_xa is not None else disabled_noise_scale
                noise_scale_aa_arg = noise_scale_aa if noise_scale_aa is not None else disabled_noise_scale
                group_ids_arg = (
                    jnp.asarray(
                        pad_axis(group_ids_np[bucket_image_indices], 0, batch_size, value=0),
                        dtype=jnp.int32,
                    )
                    if group_ids_np is not None
                    else jnp.zeros(batch_size, dtype=jnp.int32)
                )
                shell_indices_half_arg = shell_indices_half
                shell_indices_noise_arg = shell_indices_noise
                noise_variance_for_noise_arg = noise_variance_for_noise
                scale_correction_pixel_mask_arg = scale_correction_pixel_mask
                n_shells_arg = n_shells
            else:
                noise_wsum_arg = disabled_noise_wsum
                noise_img_power_arg = disabled_noise_img_power
                noise_a2_arg = disabled_noise_a2
                noise_xa_arg = disabled_noise_xa
                noise_scale_xa_arg = disabled_noise_scale
                noise_scale_aa_arg = disabled_noise_scale
                group_ids_arg = disabled_group_ids
                shell_indices_half_arg = disabled_noise_shell_indices
                shell_indices_noise_arg = disabled_noise_shell_indices
                noise_variance_for_noise_arg = noise_variance_half
                scale_correction_pixel_mask_arg = jnp.zeros(n_half, dtype=bool)
                n_shells_arg = 1
            if reconstruction_probability_threshold_np is None:
                reconstruction_probability_threshold_arg = jnp.zeros((batch_size,), dtype=jnp.float64)
                has_reconstruction_probability_threshold = False
            else:
                threshold_values = reconstruction_probability_threshold_np[bucket_image_indices]
                threshold_values = pad_axis(threshold_values, 0, batch_size, value=np.inf)
                reconstruction_probability_threshold_arg = jnp.asarray(threshold_values, dtype=jnp.float64)
                has_reconstruction_probability_threshold = True

            projection_max_r_big_jit = window_spec.dense_big_jit_max_r()
            return_big_jit_mstep_tensors = _return_local_big_jit_mstep_tensors(
                sparse_big_jit_backprojection=sparse_big_jit_backprojection,
                source_faithful_bpref=source_faithful_bpref,
                grouped_reconstruction=reconstruction_group_ids_np is not None,
                reconstruct_significant_only=reconstruct_significant_only,
                disable_adjoint_y=disable_adjoint_y,
                disable_adjoint_ctf=disable_adjoint_ctf,
            )
            if defer_packed_vdam_enabled:
                if not deferred_big_jit_backprojection:
                    raise ValueError(
                        "deferred packed VDAM did not reach the deferred big-JIT path"
                    )
                return_big_jit_mstep_tensors = False
            return_source_vdam_operands = bool(
                return_big_jit_mstep_tensors
                and source_faithful_bpref
                and not disable_adjoint_y
                and not disable_adjoint_ctf
            )
            return_big_jit_deferred_mstep_inputs = (
                deferred_big_jit_backprojection and not return_big_jit_mstep_tensors
            )
            return_deferred_source_vdam_operands = bool(
                return_big_jit_deferred_mstep_inputs
                and defer_packed_vdam_enabled
            )
            if host_plan_pack_enabled and not (
                return_big_jit_deferred_mstep_inputs
                and return_deferred_source_vdam_operands
                and packed_final_noise_enabled
            ):
                raise ValueError("host-plan packing did not reach the deferred source-noise lane")
            if stable_window_active and not (
                return_source_vdam_operands
                or return_deferred_source_vdam_operands
            ):
                raise ValueError(
                    "stable Fourier-window BPref requires the sparse source-operand "
                    "route for every bucket"
                )
            big_jit_disable_adjoint_y = (
                disable_adjoint_y or return_big_jit_mstep_tensors or return_big_jit_deferred_mstep_inputs
            )
            big_jit_disable_adjoint_ctf = (
                disable_adjoint_ctf or return_big_jit_mstep_tensors or return_big_jit_deferred_mstep_inputs
            )
            fused_debug_bucket_matches = debug_fused_posterior_bucket_matches
            score_debug_bucket_matches = bool(debug_score_dump_big_jit and debug_score_dump_bucket_matches)
            return_big_jit_debug_arrays = bool(
                fused_debug_bucket_matches
                or score_debug_bucket_matches
                or bpref_contribution_capture_active
            )
            return_big_jit_debug_scores = bool(
                score_debug_bucket_matches
                or bpref_contribution_capture_active
                or (
                    fused_debug_bucket_matches
                    and debug_fused_posterior_dump_scores
                )
            )
            return_big_jit_debug_operands = bool(debug_score_dump_operands and score_debug_bucket_matches)
            if flat_local_rows_enabled and score_only and return_big_jit_debug_operands:
                raise ValueError("flat local rows do not yet support dense projection operand dumps")
            if return_big_jit_debug_arrays:
                big_jit_debug_bucket_count += 1
            flat_local_row_argument = (
                _build_flat_local_row_argument(
                    unpadded_bucket,
                    flat_local_row_capacities,
                    dense_batch_size=batch_size,
                    rotation_block_size=rotation_block_size,
                    exact_local_bucket_radix=resolved_exact_local_bucket_radix,
                )
                if flat_local_rows_enabled
                else np.zeros((1, 3), dtype=np.int32)
            )
            if fused_pair_fine_score_enabled:
                fine_job_capacity_key = (
                    int(batch_size),
                    int(bucket.bucket_rotation_count),
                )
                fused_pair_arguments = _build_local_fused_pair_fine_arguments(
                    bucket,
                    flat_local_row_argument,
                    valid_image_mask,
                    fine_job_bucket_size=fine_job_capacities[
                        fine_job_capacity_key
                    ],
                )
                fused_fine_job_plan_arg = jnp.asarray(
                    fused_pair_arguments["job_plan"],
                    dtype=jnp.int32,
                )
                pair_capacity = int(fused_pair_arguments["job_bucket_size"])
                valid_pair_count = int(fused_pair_arguments["valid_pair_count"])
                dense_pair_capacity = int(
                    fused_pair_arguments["dense_candidate_capacity"]
                )
                chunk_fused_pair_capacities.append(pair_capacity)
                chunk_fused_pair_counts.append(valid_pair_count)
                chunk_fused_pair_dense_capacities.append(dense_pair_capacity)
                total_fused_pair_capacity += pair_capacity
                total_fused_pair_candidates += valid_pair_count
                total_fused_pair_dense_capacity += dense_pair_capacity
            else:
                fused_fine_job_plan_arg = jnp.full(
                    (1, 4),
                    -1,
                    dtype=jnp.int32,
                )
            big_jit_arguments = (
                jnp.asarray(batch_data),
                jnp.asarray(ctf_params),
                ctf_rfloat_half_arg,
                inverse_noise_rfloat_cast_arg,
                corr_img_rfloat_square_arg,
                mean_for_proj_big_jit,
                local_projection_half_arg,
                _LocalMstepAccumulators(Ft_y, Ft_ctf),
                _LocalNoiseAccumulators(
                    noise_wsum_arg,
                    noise_img_power_arg,
                    noise_a2_arg,
                    noise_xa_arg,
                    noise_scale_xa_arg,
                    noise_scale_aa_arg,
                    noise_sigma2_offset,
                    noise_sumw,
                ),
                big_jit_image_mask_arg,
                integer_pre_shifts_arg,
                fourier_pre_shifts_arg,
                image_corrections_arg,
                image_only_corrections_arg,
                scale_corrections_arg,
                translation_sqdist_arg,
                noise_variance_half,
                translation_phases_half,
                relion_score_translation_angles,
                half_weights,
                norm_half_weights,
                big_jit_window_indices_arg,
                jnp.asarray(relion_fine_full_to_compact, dtype=jnp.int32),
                big_jit_recon_window_indices_arg,
                jnp.asarray(big_jit_relion_wavg_rectangle_indices_arg, dtype=jnp.int32),
                jnp.asarray(big_jit_relion_wavg_exact_positions_arg, dtype=jnp.int32),
                jnp.asarray(big_jit_relion_wavg_rectangle_shell_indices_arg, dtype=jnp.int32),
                big_jit_mstep_recon_window_indices_arg,
                shell_indices_half_arg,
                shell_indices_noise_arg,
                noise_variance_for_noise_arg,
                scale_correction_pixel_mask_arg,
                big_jit_projection_pixel_indices_arg,
                big_jit_projection_score_take_arg,
                big_jit_projection_recon_take_arg,
                relion_projection_cache.projections,
                relion_projection_cache.id_map,
                jnp.asarray(
                    projection_cache.rows_for_bucket(relion_projection_cache, bucket.local_rotation_ids),
                    dtype=jnp.int32,
                ),
                jnp.asarray(bucket.local_rotations),
                jnp.asarray(_local_mstep_rotations(bucket)),
                jnp.asarray(flat_local_row_argument, dtype=jnp.int32),
                fused_fine_job_plan_arg,
                local_rotation_log_prior_arg,
                jnp.asarray(bucket.translation_log_prior),
                jnp.asarray(bucket.local_rotation_mask),
                sample_mask_arg,
                jnp.asarray(valid_image_mask),
                group_ids_arg,
                normalization_log_z_arg,
                normalization_log_evidence_arg,
                normalization_max_posterior_arg,
                reconstruction_probability_threshold_arg,
                jnp.asarray(logical_current_size, dtype=jnp.int32),
                big_jit_config,
                local_projection_runtime_radius,
            )
            big_jit_static_options = dict(
                mask_mode=big_jit_mask_mode,
                score_with_masked_images=score_with_masked_images,
                apply_integer_pre_shift=apply_integer_pre_shift,
                apply_fourier_pre_shift=apply_fourier_pre_shift,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_scoring=use_float64_scoring,
                use_float64_normalization=use_float64_normalization,
                use_window=use_window,
                reconstruct_significant_only=reconstruct_significant_only,
                use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                image_shape=image_shape,
                proj_volume_shape=proj_volume_shape,
                recon_volume_shape=recon_volume_shape,
                disc_type=disc_type,
                projection_half_volume=projection_half_volume_big_jit,
                projection_max_r=projection_max_r_big_jit,
                mstep_max_r=mstep_adjoint_max_r,
                use_compact_relion_projector_projection=bool(compact_relion_projector_big_jit),
                use_relion_projection_cache=bool(relion_projection_cache.enabled),
                relion_projector_output_size=int(big_jit_relion_projector_output_size),
                projection_relion_texture_interp=bool(projection_relion_texture_interp),
                projection_force_jax=bool(projection_force_jax),
                projection_mask_current_image_disk=bool(projection_mask_current_image_disk),
                relion_exact_bpref_operands=relion_exact_bpref_operands,
                relion_exact_fine_diff2=relion_exact_fine_diff2,
                use_flat_local_rows=flat_local_rows_enabled,
                use_packed_local_projection=packed_local_projection_enabled,
                use_fused_pair_fine_score=fused_pair_fine_score_enabled,
                relion_wavg_sequential_cuda=relion_wavg_sequential_cuda,
                stable_fourier_window_shapes=stable_window_active,
                relion_cuda_preprocess_radius=relion_cuda_preprocess_radius,
                relion_cuda_preprocess_cosine_width=relion_cuda_preprocess_cosine_width,
                mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
                mstep_relion_x_half=bool(mstep_relion_x_half),
                relion_sequential_mstep_reduction=source_faithful_bpref,
                disable_adjoint_y=big_jit_disable_adjoint_y,
                disable_adjoint_ctf=big_jit_disable_adjoint_ctf,
                accumulate_noise=accumulate_noise and not return_big_jit_deferred_mstep_inputs,
                accumulate_scale_correction=group_ids_np is not None,
                return_noise_split=return_noise_split,
                return_mstep_tensors=return_big_jit_mstep_tensors,
                return_source_vdam_operands=return_source_vdam_operands,
                return_deferred_mstep_inputs=return_big_jit_deferred_mstep_inputs,
                return_deferred_source_vdam_operands=(
                    return_deferred_source_vdam_operands
                ),
                packed_deferred_source_vdam_noise=bool(
                    return_deferred_source_vdam_operands
                    and packed_final_noise_enabled
                ),
                return_deferred_noise_inputs=bool(return_big_jit_deferred_mstep_inputs and accumulate_noise),
                n_shells=n_shells_arg,
                norm_current_size=physical_current_size,
                include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                has_normalization_log_z=normalization_log_z_np is not None,
                has_normalization_log_evidence=normalization_log_evidence_np is not None,
                has_normalization_max_posterior=normalization_max_posterior_np is not None,
                has_reconstruction_probability_threshold=has_reconstruction_probability_threshold,
                score_only=score_only,
                use_relion_projector=bool(use_relion_projector),
                relion_projector_r_max=local_projection_static_radius,
                projector_capacity=projector_capacity_enabled,
                projection_padding_factor=int(projection_padding_factor),
                return_debug_arrays=return_big_jit_debug_arrays,
                return_debug_scores=return_big_jit_debug_scores,
                return_debug_operands=return_big_jit_debug_operands,
                unweighted_high_shell_image_power=unweighted_high_shell_image_power,
            )
            if fixed_capacity_whole_boundary_enabled:
                fixed_capacity_whole_preparation_s += time.time() - big_jit_t0
                if fixed_capacity_whole_initial_carry is None:
                    fixed_capacity_whole_initial_carry = (
                        *big_jit_arguments[7], *big_jit_arguments[8]
                    )
                    fixed_capacity_whole_static_options = big_jit_static_options
                elif big_jit_static_options != fixed_capacity_whole_static_options:
                    raise ValueError(
                        "fixed-capacity whole-local calls require one static option topology"
                    )
                fixed_capacity_whole_call_program.append(
                    _prepare_fixed_capacity_local_call(*big_jit_arguments)
                )
                fixed_capacity_whole_call_contexts.append(
                    _FixedCapacityWholeScoreCallContext(
                        unpadded_bucket=unpadded_bucket,
                        padded_bucket=bucket,
                        unpadded_batch_size=unpadded_batch_size,
                    )
                )
                if bucket_index + 1 < len(bucket_specs):
                    continue
                fixed_capacity_whole_execution_t0 = time.time()
                fixed_capacity_segments = _partition_uniform_fixed_capacity_calls(
                    fixed_capacity_whole_call_program
                )
                logger.info(
                    "Exact local fixed-capacity score executor: calls=%d "
                    "uniform_scan_segments=%d",
                    len(fixed_capacity_whole_call_program),
                    len(fixed_capacity_segments),
                )
                final_carry, whole_call_outputs = (
                    run_fixed_capacity_segmented_local_scan(
                        fixed_capacity_whole_call_program,
                        *fixed_capacity_whole_initial_carry,
                        **fixed_capacity_whole_static_options,
                    )
                )
                Ft_y, Ft_ctf = final_carry[:2]
                if return_profile:
                    _block_until_ready(Ft_y, Ft_ctf, whole_call_outputs)
                timing.big_jit_bucket_s += (
                    fixed_capacity_whole_preparation_s
                    + time.time()
                    - fixed_capacity_whole_execution_t0
                )
                big_jit_bucket_count += len(fixed_capacity_whole_call_program)
                postprocess_t0 = time.time()
                added_significant, added_reconstruction_rows = (
                    _postprocess_fixed_capacity_whole_score_calls(
                        fixed_capacity_whole_call_contexts,
                        final_carry,
                        whole_call_outputs,
                        n_trans=n_trans,
                        translation_grid=local_layout.translation_grid,
                        stats_use_reconstruction_probs=stats_use_reconstruction_probs,
                        collect_profile_stats=collect_profile_stats,
                        buffers=postprocess_buffers,
                    )
                )
                total_significant_samples += added_significant
                total_reconstruction_rows += added_reconstruction_rows
                timing.postprocess_s += time.time() - postprocess_t0
                for context in fixed_capacity_whole_call_contexts:
                    local_progress.mark_bucket_done(context.padded_bucket)
                continue
            if bpref_transaction_queue is not None:
                big_jit_result = bpref_transaction_queue.run_deferred_scorer(
                    _invoke_local_bucket_big_jit, big_jit_arguments, big_jit_static_options
                )
            else:
                big_jit_result = _invoke_local_bucket_big_jit(
                    *big_jit_arguments,
                    **big_jit_static_options,
                )
            (
                debug_scores, debug_probs, debug_shifted_score_split,
                debug_shifted_recon_split, debug_ctf2_over_nv_score,
                debug_ctf2_over_nv_recon, debug_proj_weighted,
                debug_proj_for_noise, debug_wavg_cutoff_triplet,
            ) = big_jit_result.debug or _LocalBigJitDebug()
            if score_only:
                debug_shifted_recon_split = None
                debug_ctf2_over_nv_recon = None
                debug_proj_for_noise = None
                debug_wavg_cutoff_triplet = None
            (
                Ft_y,
                Ft_ctf,
                noise_wsum,
                noise_img_power,
                noise_a2,
                noise_xa,
                noise_scale_xa,
                noise_scale_aa,
                bucket_norm_correction,
                noise_sigma2_offset,
                noise_sumw,
                batch_norm,
                log_Z,
                best_log_score,
                best_argmax,
                max_posterior,
                probs_sum_t,
                reconstruction_probs_sum_t,
                n_significant_samples,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                reconstruction_row_count_jax,
            ) = big_jit_result.core
            summed = None
            ctf_probs = None
            if return_big_jit_deferred_mstep_inputs:
                (
                    reconstruction_probs,
                    shifted_recon_split,
                    ctf2_over_nv_recon,
                    shifted_noise_split,
                    processed_score_half,
                    deferred_flat_proj_for_noise,
                    deferred_source_vdam_images,
                    deferred_source_vdam_ctf,
                    deferred_source_vdam_minvsigma2,
                    deferred_source_vdam_ctf_probs,
                ) = big_jit_result.deferred_mstep
            elif return_big_jit_mstep_tensors and return_source_vdam_operands:
                (
                    source_vdam_images,
                    source_vdam_ctf,
                    source_vdam_minvsigma2,
                    source_vdam_posterior,
                    source_vdam_reference,
                    ctf_probs,
                ) = big_jit_result.source_vdam
                if bpref_contribution_capture_active:
                    # The source-faithful production route deliberately avoids
                    # materializing the large (particle, rotation, pixel)
                    # numerator tensor.  Contribution capture is diagnostic and
                    # needs those pre-scatter rows, so reproduce them only for
                    # the selected capture bucket in RELION's native statement
                    # order.  Recompute the denominator as well so both captured
                    # operands come from the same reducer.
                    from recovar import cuda_backproject

                    summed, ctf_probs = cuda_backproject.relion_vdam_mstep_sums_f32(
                        source_vdam_images,
                        source_vdam_ctf,
                        source_vdam_minvsigma2,
                        source_vdam_posterior,
                        relion_score_translation_angles,
                        mstep_recon_window_indices,
                        source_vdam_reference,
                        image_shape,
                    )
            elif return_big_jit_mstep_tensors:
                summed, ctf_probs = big_jit_result.mstep_tensors
            if group_ids_np is None:
                noise_scale_xa = None
                noise_scale_aa = None
            if return_profile:
                _block_until_ready(
                    Ft_y,
                    Ft_ctf,
                    batch_norm,
                    log_Z,
                    best_log_score,
                    best_argmax,
                    max_posterior,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    n_significant_samples,
                    reconstruction_sample_mask,
                    reconstruction_rotation_mask,
                    reconstruction_row_count_jax,
                    noise_wsum,
                    noise_img_power,
                    bucket_norm_correction,
                    *(() if summed is None else (summed, ctf_probs)),
                    *(
                        ()
                        if not return_big_jit_deferred_mstep_inputs
                        else (
                            reconstruction_probs,
                            shifted_recon_split,
                            ctf2_over_nv_recon,
                            shifted_noise_split,
                            processed_score_half,
                            deferred_flat_proj_for_noise,
                            deferred_source_vdam_images,
                            deferred_source_vdam_ctf,
                            deferred_source_vdam_minvsigma2,
                            deferred_source_vdam_ctf_probs,
                        )
                    ),
                )
            # Deferred BigJIT returns float32 zero norm rows without computing
            # noise. The later deferred noise path owns the real update.
            if accumulate_noise and not (
                skip_deferred_zero_norm and return_big_jit_deferred_mstep_inputs
            ):
                noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket_image_indices, dtype=jnp.int32)].add(
                    bucket_norm_correction[:unpadded_batch_size].astype(noise_norm_correction.dtype),
                )
            timing.big_jit_bucket_s += time.time() - big_jit_t0
            big_jit_bucket_count += 1
            if sparse_big_jit_backprojection or return_big_jit_deferred_mstep_inputs:
                sparse_big_jit_bucket_count += 1

            if return_big_jit_debug_arrays:
                debug_probs_unpadded = debug_probs[:unpadded_batch_size]
                debug_scores_unpadded = (
                    debug_scores[:unpadded_batch_size]
                    if return_big_jit_debug_scores
                    else None
                )
                log_Z_unpadded = log_Z[:unpadded_batch_size]
                best_log_score_unpadded = best_log_score[:unpadded_batch_size]
                best_argmax_unpadded = best_argmax[:unpadded_batch_size]
                max_posterior_unpadded = max_posterior[:unpadded_batch_size]
                reconstruction_sample_mask_unpadded = reconstruction_sample_mask[:unpadded_batch_size]
                reconstruction_rotation_mask_unpadded = reconstruction_rotation_mask[:unpadded_batch_size]
                n_significant_samples_unpadded = n_significant_samples[:unpadded_batch_size]
                if bpref_contribution_capture_active:
                    if summed is None or ctf_probs is None or debug_scores_unpadded is None:
                        raise RuntimeError(
                            "big-JIT BPref contribution capture requires returned M-step tensors and scores"
                        )
                    inline_projector_data_volumes = None
                    inline_projector_weight_volumes = None
                    if (
                        return_source_vdam_operands
                        and os.environ.get(
                            "RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES",
                            "",
                        ).strip()
                    ):
                        target_particle_rows = (
                            bpref_diagnostics._bpref_contribution_target_rows(
                                experiment_dataset,
                                unpadded_bucket.image_indices,
                            )
                        )
                        inline_data_parts = []
                        inline_weight_parts = []
                        inline_row_mask = (
                            reconstruction_rotation_mask_unpadded
                            & jnp.asarray(unpadded_bucket.local_rotation_mask)
                        )
                        for target_particle_row in target_particle_rows.tolist():
                            if bucket_reconstruction_group_ids is None:
                                zero_data = jnp.zeros_like(Ft_y)
                                zero_weight = jnp.zeros_like(Ft_ctf)
                            else:
                                # The diagnostic replays one particle without a
                                # group-ID operand, so give the ungrouped kernel
                                # one group's native x-half accumulator shape.
                                zero_data = jnp.zeros_like(Ft_y[0])
                                zero_weight = jnp.zeros_like(Ft_ctf[0])
                            target_slice = slice(
                                int(target_particle_row),
                                int(target_particle_row) + 1,
                            )
                            inline_data, inline_weight = (
                                _accumulate_relion_vdam_physical_particle_grid(
                                    source_vdam_images[target_slice],
                                    source_vdam_ctf[target_slice],
                                    source_vdam_minvsigma2[target_slice],
                                    source_vdam_posterior[target_slice],
                                    relion_score_translation_angles,
                                    source_vdam_reference[target_slice],
                                    _local_mstep_rotations(unpadded_bucket)[target_slice],
                                    inline_row_mask[target_slice],
                                    zero_data,
                                    zero_weight,
                                    projector_full=source_vdam_projector_full,
                                    scoring_rotations=unpadded_bucket.local_rotations[
                                        target_slice
                                    ],
                                    projector_r_max=source_vdam_projector_static_radius,
                                    runtime_projector_radius=source_vdam_projector_runtime_radius,
                                    projection_padding_factor=projection_padding_factor,
                                    pixel_indices=mstep_recon_window_indices,
                                    image_shape=image_shape,
                                    volume_shape=recon_volume_shape,
                                    max_r=mstep_adjoint_max_r,
                                    stable_dense_positions=(
                                        stable_bpref_dense_positions
                                        if stable_window_active
                                        else None
                                    ),
                                    logical_current_size=(
                                        mstep_current_size
                                        if stable_window_active
                                        else None
                                    ),
                                )
                            )
                            inline_data_parts.append(inline_data)
                            inline_weight_parts.append(inline_weight)
                        if inline_data_parts:
                            inline_projector_data_volumes = jnp.stack(
                                inline_data_parts,
                                axis=0,
                            )
                            inline_projector_weight_volumes = jnp.stack(
                                inline_weight_parts,
                                axis=0,
                            )
                    capture_priors = _bpref_capture_priors(
                        debug_scores_unpadded,
                        debug_probs_unpadded.shape,
                        bucket=unpadded_bucket,
                        rotation_log_prior=local_rotation_log_prior_arg[:unpadded_batch_size],
                    )
                    reconstruction_probs_for_dump = (
                        _exact_local_bpref_reconstruction_probs_for_capture(
                            debug_scores_unpadded,
                            debug_probs_unpadded,
                            reconstruction_sample_mask_unpadded,
                            use_relion_f32_fine_posterior=(
                                use_relion_f32_fine_posterior
                            ),
                            adaptive_fraction=adaptive_fraction,
                        )
                    )
                    if reconstruction_probability_threshold_np is None:
                        reconstruction_threshold_for_dump = jnp.zeros(
                            (unpadded_batch_size,), dtype=jnp.float64
                        )
                    else:
                        reconstruction_threshold_for_dump = jnp.asarray(
                            reconstruction_probability_threshold_np[
                                np.asarray(unpadded_bucket.image_indices)
                            ],
                            dtype=jnp.float64,
                        )
                    _maybe_dump_exact_local_bpref_contribution_rows(
                        experiment_dataset=experiment_dataset,
                        image_indices=unpadded_bucket.image_indices,
                        current_size=current_size,
                        summed=summed[:unpadded_batch_size],
                        ctf_probs=ctf_probs[:unpadded_batch_size],
                        rotations=_local_mstep_rotations(unpadded_bucket),
                        actual_counts=unpadded_bucket.actual_rotation_counts,
                        rotation_indices=unpadded_bucket.local_rotation_ids,
                        fine_translations=local_layout.translation_grid,
                        scores=debug_scores_unpadded,
                        preprior_scores=capture_priors.preprior_scores,
                        probs=debug_probs_unpadded,
                        rotation_log_prior=capture_priors.rotation_log_prior,
                        translation_log_prior=capture_priors.translation_log_prior,
                        log_z=log_Z_unpadded,
                        best_log_score=best_log_score_unpadded,
                        reconstruction_probs=reconstruction_probs_for_dump,
                        reconstruction_mask=reconstruction_sample_mask_unpadded,
                        reconstruction_sum_weight=jnp.sum(
                            reconstruction_probs_for_dump, axis=(1, 2)
                        ),
                        reconstruction_threshold=reconstruction_threshold_for_dump,
                        candidate_mask=capture_priors.candidate_mask,
                        **capture_static_kwargs,
                        inline_projector_data_volumes=inline_projector_data_volumes,
                        inline_projector_weight_volumes=inline_projector_weight_volumes,
                        reconstruction_group_ids=(
                            None
                            if bucket_reconstruction_group_ids is None
                            else bucket_reconstruction_group_ids[:unpadded_batch_size]
                        ),
                    )
                if fused_debug_bucket_matches and debug_fused_posterior_dump_targets:
                    debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                        experiment_dataset=experiment_dataset,
                        local_layout=local_layout,
                        bucket=unpadded_bucket,
                        image_pre_shifts=image_pre_shifts,
                        scores=(
                            debug_scores_unpadded
                            if debug_fused_posterior_dump_scores
                            else None
                        ),
                        probs=debug_probs_unpadded,
                        log_Z=log_Z_unpadded,
                        best_log_score=best_log_score_unpadded,
                        best_argmax=best_argmax_unpadded,
                        max_posterior=max_posterior_unpadded,
                        reconstruction_sample_mask=reconstruction_sample_mask_unpadded,
                        reconstruction_rotation_mask=reconstruction_rotation_mask_unpadded,
                        n_significant_samples=n_significant_samples_unpadded,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        dump_dir=debug_fused_posterior_dump_dir,
                        pending_targets=debug_fused_posterior_dump_targets,
                        requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                        requested_iterations=debug_fused_posterior_dump_iterations,
                    )
                if score_debug_bucket_matches and debug_score_dump_targets:
                    reconstruction_probs_for_score_dump = (
                        _exact_local_bpref_reconstruction_probs_for_capture(
                            debug_scores_unpadded,
                            debug_probs_unpadded,
                            reconstruction_sample_mask_unpadded,
                            use_relion_f32_fine_posterior=(
                                use_relion_f32_fine_posterior
                            ),
                            adaptive_fraction=adaptive_fraction,
                        )
                    )
                    debug_score_dump_targets = maybe_write_debug_score_dump(
                        experiment_dataset=experiment_dataset,
                        local_layout=local_layout,
                        bucket=unpadded_bucket,
                        image_pre_shifts=image_pre_shifts,
                        scores=debug_scores_unpadded,
                        probs=debug_probs_unpadded,
                        reconstruction_probs=reconstruction_probs_for_score_dump,
                        log_Z=log_Z_unpadded,
                        best_log_score=best_log_score_unpadded,
                        max_posterior=max_posterior_unpadded,
                        reconstruction_sample_mask=reconstruction_sample_mask_unpadded,
                        reconstruction_rotation_mask=reconstruction_rotation_mask_unpadded,
                        n_significant_samples=n_significant_samples_unpadded,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        debug_pass_label=debug_pass_label,
                        shifted_score_split=debug_shifted_score_split,
                        shifted_recon_split=debug_shifted_recon_split,
                        ctf2_over_nv_score=debug_ctf2_over_nv_score,
                        ctf2_over_nv_recon=debug_ctf2_over_nv_recon,
                        proj_weighted=debug_proj_weighted,
                        proj_for_noise=debug_proj_for_noise,
                        proj_abs2_weighted=None,
                        wavg_cutoff_triplet=debug_wavg_cutoff_triplet,
                        dump_dir=debug_score_dump_dir,
                        pending_targets=debug_score_dump_targets,
                        requested_current_sizes=debug_score_dump_current_sizes,
                        requested_iterations=debug_score_dump_iterations,
                    )

            pack_t0 = time.time()
            reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)[:unpadded_batch_size]
            local_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)[:unpadded_batch_size]
            # Pack only outputs explicitly returned by the big JIT.  Besides
            # the sparse M-step path, grouped oversampling-zero VDAM returns
            # physical operands here so the outer group-aware scatter can
            # route each particle to its pseudo-halfset accumulator.
            packed_reconstruction_probs = None
            packed_reconstruction_probs_sum_t = None
            packed_source_vdam_images = None
            packed_source_vdam_ctf = None
            packed_source_vdam_minvsigma2 = None
            packed_source_vdam_posterior = None
            packed_source_vdam_reference = None
            packed_source_vdam_ctf_probs = None
            packed_source_vdam_noise_projection = None
            if return_big_jit_deferred_mstep_inputs:
                probs_sum_t_np = (
                    np.asarray(probs_sum_t, dtype=np.float64)[:unpadded_batch_size]
                    if host_publication_enabled
                    else np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
                )
                (
                    reconstruction_take_indices,
                    reconstruction_pack_mask_np,
                    _,
                    reconstruction_row_count,
                ) = _build_nonzero_reconstruction_pack_indices(
                    reconstruction_rotation_mask_np,
                    local_mask_np,
                    probs_sum_t_np,
                    rotation_block_size,
                    exact_local_bucket_radix=resolved_exact_local_bucket_radix,
                )
                if collect_profile_stats and packed_final_noise_enabled:
                    total_packed_final_noise_rows += int(
                        reconstruction_pack_mask_np.size
                    )
                (
                    reconstruction_take_indices_jnp,
                    reconstruction_pack_mask_jnp,
                    packed_rotations_np,
                    packed_mstep_rotations_np,
                ) = _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np, batch_rows=unpadded_batch_size)
                if not host_plan_pack_enabled:
                    packed_reconstruction_probs = _packed_reconstruction_rows(reconstruction_probs[:unpadded_batch_size], reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
                    packed_reconstruction_probs_sum_t = jnp.take_along_axis(
                        reconstruction_probs_sum_t[:unpadded_batch_size],
                        reconstruction_take_indices_jnp,
                        axis=1,
                    )
                    packed_reconstruction_probs_sum_t = jnp.where(
                        reconstruction_pack_mask_jnp,
                        packed_reconstruction_probs_sum_t,
                        0.0,
                    )
                if return_deferred_source_vdam_operands:
                    if not host_plan_pack_enabled:
                        packed_source_vdam_images = deferred_source_vdam_images[
                            :unpadded_batch_size
                        ]
                        packed_source_vdam_ctf = deferred_source_vdam_ctf[
                            :unpadded_batch_size
                        ]
                        packed_source_vdam_minvsigma2 = (
                            deferred_source_vdam_minvsigma2[:unpadded_batch_size]
                        )
                        packed_source_vdam_posterior = packed_reconstruction_probs
                    if packed_final_noise_enabled:
                        flat_proj_for_noise = jnp.asarray(
                            deferred_flat_proj_for_noise,
                            dtype=jnp.complex64,
                        )
                        expected_flat_projection_shape = (
                            int(flat_local_row_argument.shape[0]),
                            window_spec.n_recon
                            if window_spec.use_window
                            else int(n_half),
                        )
                        if flat_proj_for_noise.shape != expected_flat_projection_shape:
                            raise RuntimeError(
                                "deferred VDAM scoring projection did not retain the "
                                "packed reconstruction layout: "
                                f"{flat_proj_for_noise.shape} vs "
                                f"{expected_flat_projection_shape}",
                            )
                        dense_to_flat_lookup = build_dense_to_flat_local_row_lookup(
                            flat_local_row_argument,
                            batch_size=int(batch_size),
                            dense_rotation_count=int(bucket.bucket_rotation_count),
                        )
                        packed_flat_take_indices = map_dense_local_rows_to_flat_rows(
                            dense_to_flat_lookup,
                            reconstruction_take_indices,
                            reconstruction_pack_mask_np,
                        )
                        if host_plan_pack_enabled:
                            from recovar.em.helpers.deferred_vdam_host_pack import pack_deferred_vdam_host_plan
                            if host_plan_cuda_enabled:
                                from recovar.em.helpers.deferred_vdam_host_pack import (
                                    pack_deferred_vdam_host_plan_cuda as pack_deferred_vdam_host_plan,
                                )

                            (
                                packed_reconstruction_probs,
                                packed_reconstruction_probs_sum_t,
                                packed_source_vdam_images,
                                packed_source_vdam_ctf,
                                packed_source_vdam_minvsigma2,
                                packed_source_vdam_noise_projection,
                                packed_source_vdam_ctf_probs,
                            ) = pack_deferred_vdam_host_plan(
                                reconstruction_probs,
                                reconstruction_probs_sum_t,
                                deferred_source_vdam_images,
                                deferred_source_vdam_ctf,
                                deferred_source_vdam_minvsigma2,
                                flat_proj_for_noise,
                                reconstruction_take_indices_jnp,
                                reconstruction_pack_mask_jnp,
                                jnp.asarray(packed_flat_take_indices, dtype=jnp.int32),
                            )
                            packed_source_vdam_posterior = packed_reconstruction_probs
                        else:
                            packed_source_vdam_noise_projection = jnp.take(
                                flat_proj_for_noise,
                                jnp.asarray(packed_flat_take_indices).reshape(-1),
                                axis=0,
                            ).reshape(
                                (
                                    *packed_flat_take_indices.shape,
                                    flat_proj_for_noise.shape[-1],
                                )
                            )
                            packed_source_vdam_noise_projection = jnp.where(
                                reconstruction_pack_mask_jnp[:, :, None],
                                packed_source_vdam_noise_projection,
                                0.0,
                            )

                            from recovar import cuda_backproject

                            packed_source_vdam_ctf_probs = (
                                cuda_backproject.relion_vdam_mstep_denominator_f32(
                                    packed_source_vdam_ctf,
                                    packed_source_vdam_minvsigma2,
                                    packed_source_vdam_posterior,
                                )
                            )
                    else:
                        packed_source_vdam_ctf_probs = jnp.take_along_axis(
                            deferred_source_vdam_ctf_probs[:unpadded_batch_size],
                            reconstruction_take_indices_jnp[:, :, None],
                            axis=1,
                        )
                    if not host_plan_pack_enabled:
                        packed_source_vdam_ctf_probs = jnp.where(
                            reconstruction_pack_mask_jnp[:, :, None],
                            packed_source_vdam_ctf_probs,
                            0.0,
                        )
                packed_summed = None
                packed_ctf_probs = None
                packed_flat_rotations = None
            elif return_source_vdam_operands:
                probs_sum_t_np = np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
                if vdam_replay._relion_vdam_block_start_replay_active(
                    debug_iteration=debug_iteration
                ):
                    # Native launches every row in its padded significant-
                    # orientation grid, including rows whose posterior is zero.
                    # The production optimization below removes those no-op
                    # blocks; retain them only for exact captured chronology
                    # replay so native row IDs and launch timing remain aligned.
                    (
                        reconstruction_take_indices,
                        reconstruction_pack_mask_np,
                        _,
                        reconstruction_row_count,
                    ) = _build_reconstruction_pack_indices(
                        local_mask_np,
                        local_mask_np,
                        rotation_block_size,
                        exact_local_bucket_radix=resolved_exact_local_bucket_radix,
                    )
                else:
                    (
                        reconstruction_take_indices,
                        reconstruction_pack_mask_np,
                        _,
                        reconstruction_row_count,
                    ) = _build_nonzero_reconstruction_pack_indices(
                        reconstruction_rotation_mask_np,
                        local_mask_np,
                        probs_sum_t_np,
                        rotation_block_size,
                        exact_local_bucket_radix=resolved_exact_local_bucket_radix,
                    )
                (
                    reconstruction_take_indices_jnp,
                    reconstruction_pack_mask_jnp,
                    packed_rotations_np,
                    packed_mstep_rotations_np,
                ) = _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np, batch_rows=unpadded_batch_size, rotations_dtype=np.float32)
                packed_source_vdam_images = source_vdam_images[:unpadded_batch_size]
                packed_source_vdam_ctf = source_vdam_ctf[:unpadded_batch_size]
                packed_source_vdam_minvsigma2 = source_vdam_minvsigma2[:unpadded_batch_size]
                packed_source_vdam_posterior = _packed_reconstruction_rows(source_vdam_posterior[:unpadded_batch_size], reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
                packed_source_vdam_reference = _packed_reconstruction_rows(source_vdam_reference[:unpadded_batch_size], reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
                packed_summed = None
                packed_ctf_probs = None
                packed_flat_rotations = None
            elif sparse_big_jit_backprojection and (not disable_adjoint_y or not disable_adjoint_ctf):
                probs_sum_t_np = np.asarray(probs_sum_t[:unpadded_batch_size], dtype=np.float64)
                (
                    reconstruction_take_indices,
                    reconstruction_pack_mask_np,
                    _,
                    reconstruction_row_count,
                ) = _build_nonzero_reconstruction_pack_indices(
                    reconstruction_rotation_mask_np,
                    local_mask_np,
                    probs_sum_t_np,
                    rotation_block_size,
                    exact_local_bucket_radix=resolved_exact_local_bucket_radix,
                )
                (
                    reconstruction_take_indices_jnp,
                    reconstruction_pack_mask_jnp,
                    packed_rotations_np,
                    packed_mstep_rotations_np,
                ) = _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np, batch_rows=unpadded_batch_size)
                packed_summed = _packed_reconstruction_rows(summed[:unpadded_batch_size], reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
                packed_ctf_probs = _packed_reconstruction_rows(ctf_probs[:unpadded_batch_size], reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
                packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_mstep_rotations_np))
            else:
                probs_sum_t_np = None
                reconstruction_take_indices = np.broadcast_to(
                    np.arange(int(bucket.bucket_rotation_count), dtype=np.int32)[None, :],
                    (unpadded_batch_size, int(bucket.bucket_rotation_count)),
                )
                reconstruction_pack_mask_np = reconstruction_rotation_mask_np & local_mask_np
                reconstruction_row_count = int(np.asarray(reconstruction_row_count_jax, dtype=np.int32))
            timing.pack_s += time.time() - pack_t0

            flat_packed_summed = None
            flat_packed_ctf_probs = None
            if (
                not source_faithful_bpref
                and sparse_big_jit_backprojection
                and (not disable_adjoint_y or not disable_adjoint_ctf)
            ):
                flat_packed_summed = flatten_bucket_rows(packed_summed)
                flat_packed_ctf_probs = flatten_bucket_rows(packed_ctf_probs)

            if (
                return_big_jit_deferred_mstep_inputs
                and source_faithful_bpref
                and not return_deferred_source_vdam_operands
                and (not disable_adjoint_y or not disable_adjoint_ctf)
            ):
                # The fused scorer intentionally did not materialize the
                # oversized (particle, rotation, pixel) BPref tensors. Reduce
                # and scatter consecutive particle groups instead. Splitting
                # the particle axis retains RELION's complete rotation/pixel
                # traversal for one particle before advancing to the next.
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                particle_chunk_size = _source_faithful_bpref_particle_chunk_size(
                    image_count=unpadded_batch_size,
                    rotation_count=int(packed_rotations_np.shape[1]),
                    n_recon_pixels=n_recon_pixels,
                    max_gb=sparse_big_jit_mstep_cap_gb,
                    max_particles=_source_faithful_bpref_particle_chunk_cap(),
                )
                if particle_chunk_size < unpadded_batch_size and not logged_deferred_mstep_chunking:
                    logger.info(
                        "Exact local source-faithful BPref particle chunking: "
                        "particles=%d chunk_particles=%d packed_rows=%d n_recon_pixels=%d cap_gb=%.3f",
                        unpadded_batch_size,
                        particle_chunk_size,
                        int(packed_rotations_np.shape[1]),
                        n_recon_pixels,
                        float(sparse_big_jit_mstep_cap_gb),
                    )
                    logged_deferred_mstep_chunking = True
                shifted_recon_split_unpadded = shifted_recon_split[:unpadded_batch_size]
                ctf2_over_nv_recon_unpadded = ctf2_over_nv_recon[:unpadded_batch_size]
                for particle_start in range(0, unpadded_batch_size, particle_chunk_size):
                    particle_stop = min(
                        unpadded_batch_size,
                        particle_start + particle_chunk_size,
                    )
                    mstep_t0 = time.time()
                    chunk_summed, chunk_ctf_probs = compute_local_mstep_sums(
                        packed_reconstruction_probs[particle_start:particle_stop],
                        shifted_recon_split_unpadded[particle_start:particle_stop],
                        ctf2_over_nv_recon_unpadded[particle_start:particle_stop],
                        relion_x_half=True,
                        sequential_translation_reduction=True,
                    )
                    if return_profile:
                        _block_until_ready(chunk_summed, chunk_ctf_probs)
                    timing.mstep_s += time.time() - mstep_t0

                    adjoint_t0 = time.time()
                    Ft_y, Ft_ctf = _accumulate_relion_physical_particle_grid(
                        chunk_summed,
                        chunk_ctf_probs,
                        packed_mstep_rotations_np[particle_start:particle_stop],
                        reconstruction_pack_mask_jnp[particle_start:particle_stop],
                        Ft_y,
                        Ft_ctf,
                        pixel_indices=mstep_recon_window_indices,
                        image_shape=image_shape,
                        volume_shape=recon_volume_shape,
                        max_r=mstep_adjoint_max_r,
                    )
                    if return_profile:
                        _block_until_ready(Ft_y, Ft_ctf)
                    timing.adjoint_y_s += time.time() - adjoint_t0
            elif (
                return_big_jit_deferred_mstep_inputs
                and not return_deferred_source_vdam_operands
                and (not disable_adjoint_y or not disable_adjoint_ctf)
            ):
                if packed_reconstruction_probs is None:
                    raise RuntimeError("deferred big-JIT local M-step requires packed posterior rows")
                if packed_reconstruction_probs_sum_t is None:
                    raise RuntimeError("deferred big-JIT local M-step requires packed posterior sums")
                packed_rotation_count = int(packed_rotations_np.shape[1])
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(n_recon_pixels, batch_size=unpadded_batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_deferred_mstep_chunking:
                    logger.info(
                        "Exact local big-JIT deferred packed M-step chunking: "
                        "packed_rows=%d chunk_rows=%d n_recon_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        n_recon_pixels,
                        unpadded_batch_size,
                    )
                    logged_deferred_mstep_chunking = True
                shifted_recon_split_unpadded = shifted_recon_split[:unpadded_batch_size]
                ctf2_over_nv_recon_unpadded = ctf2_over_nv_recon[:unpadded_batch_size]
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                    chunk_scoring_rotations = packed_rotations_np[:, chunk_start:chunk_stop]
                    chunk_rotations = packed_mstep_rotations_np[:, chunk_start:chunk_stop]
                    chunk_flat_rotations = flatten_bucket_rotations(jnp.asarray(chunk_rotations))
                    if not disable_adjoint_y:
                        mstep_t0 = time.time()
                        chunk_summed = compute_local_weighted_sums(chunk_probs, shifted_recon_split_unpadded)
                        if mstep_subtract_ctf_projection:
                            # The memory-deferred big-JIT path returns posterior rows instead
                            # of the already reduced residual-image rows. Recreate RELION's
                            # Fimg_store = Fimg - Frefctf operand before backprojection, just
                            # as the ordinary deferred path does below.
                            chunk_proj_for_residual = _project_packed_noise_rows(
                                mean_for_proj=mean_for_proj,
                                packed_flat_rotations=flatten_bucket_rotations(
                                    jnp.asarray(chunk_scoring_rotations)
                                ),
                                packed_rotation_count=chunk_stop - chunk_start,
                                batch_size=unpadded_batch_size,
                                image_shape=image_shape,
                                proj_volume_shape=proj_volume_shape,
                                disc_type=disc_type,
                                projection_kwargs=projection_kwargs,
                                window_spec=window_spec,
                                n_half=n_half,
                                precision_policy=precision_policy,
                                reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[
                                    :, chunk_start:chunk_stop
                                ],
                                relion_projector_half=relion_projector_half,
                                relion_projector_r_max=relion_projector_r_max,
                                projection_padding_factor=projection_padding_factor,
                            )
                            chunk_probs_sum_t = packed_reconstruction_probs_sum_t[
                                :, chunk_start:chunk_stop
                            ]
                            frefctf_weighted = (
                                chunk_proj_for_residual * ctf2_over_nv_recon_unpadded[:, None, :]
                            )
                            chunk_summed = chunk_summed - chunk_probs_sum_t[..., None] * frefctf_weighted
                        if return_profile:
                            _block_until_ready(chunk_summed)
                        timing.mstep_s += time.time() - mstep_t0

                        adjoint_y_t0 = time.time()
                        Ft_y, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                            flatten_bucket_rows(chunk_summed),
                            mstep_recon_window_indices,
                            chunk_flat_rotations,
                            Ft_y,
                            image_shape,
                            recon_volume_shape,
                            "linear_interp",
                            use_window=use_window,
                            max_r=mstep_adjoint_max_r,
                            relion_x_half=bool(mstep_relion_x_half),
                            target_rows=sparse_adjoint_target_rows,
                        )
                        if return_profile:
                            _block_until_ready(Ft_y)
                        sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                        timing.adjoint_y_s += time.time() - adjoint_y_t0

                    if not disable_adjoint_ctf:
                        mstep_t0 = time.time()
                        chunk_probs_sum_t = packed_reconstruction_probs_sum_t[:, chunk_start:chunk_stop]
                        chunk_ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
                            chunk_probs_sum_t,
                            ctf2_over_nv_recon_unpadded,
                        )
                        if return_profile:
                            _block_until_ready(chunk_ctf_probs)
                        timing.mstep_s += time.time() - mstep_t0

                        adjoint_ctf_t0 = time.time()
                        Ft_ctf, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                            flatten_bucket_rows(chunk_ctf_probs),
                            mstep_recon_window_indices,
                            chunk_flat_rotations,
                            Ft_ctf,
                            image_shape,
                            recon_volume_shape,
                            "linear_interp",
                            use_window=use_window,
                            max_r=mstep_adjoint_max_r,
                            relion_x_half=bool(mstep_relion_x_half),
                            target_rows=sparse_adjoint_target_rows,
                        )
                        if return_profile:
                            _block_until_ready(Ft_ctf)
                        sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                        timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            source_vdam_outer_scatter = bool(
                return_source_vdam_operands
                or return_deferred_source_vdam_operands
            )
            if source_vdam_outer_scatter:
                if any(
                    value is None
                    for value in (
                        packed_source_vdam_images,
                        packed_source_vdam_ctf,
                        packed_source_vdam_minvsigma2,
                        packed_source_vdam_posterior,
                    )
                ):
                    raise RuntimeError("source VDAM physical operands were not packed")
                adjoint_t0 = time.time()
                candidate_trace_ids = vdam_replay._relion_vdam_candidate_trace_ids_for_images(
                    experiment_dataset,
                    unpadded_bucket.image_indices,
                    debug_iteration=debug_iteration,
                )
                block_start_order = vdam_replay._relion_vdam_block_start_orders_for_images(
                    experiment_dataset,
                    unpadded_bucket.image_indices,
                    rotation_count=packed_mstep_rotations_np.shape[1],
                    valid_rotation_counts=np.sum(
                        reconstruction_pack_mask_np,
                        axis=1,
                        dtype=np.int64,
                    ),
                    debug_iteration=debug_iteration,
                )
                native_grid_counts = vdam_replay._relion_vdam_native_grid_counts_for_images(
                    experiment_dataset,
                    unpadded_bucket.image_indices,
                    rotation_count=packed_mstep_rotations_np.shape[1],
                    valid_rotation_counts=np.sum(
                        reconstruction_pack_mask_np,
                        axis=1,
                        dtype=np.int64,
                    ),
                    debug_iteration=debug_iteration,
                )
                particle_issue_order = vdam_replay._relion_vdam_particle_issue_order_for_images(
                    experiment_dataset,
                    unpadded_bucket.image_indices,
                    debug_iteration=debug_iteration,
                )
                particle_start_offsets_ns = (
                    vdam_replay._relion_vdam_particle_start_offsets_for_images(
                        experiment_dataset,
                        unpadded_bucket.image_indices,
                        debug_iteration=debug_iteration,
                    )
                )
                if vdam_replay._relion_vdam_identity_native_grid_replay():
                    # RELION launches physical block IDs directly.  Preserve its
                    # sealed per-particle grid cardinality without translating
                    # those IDs through the captured chronology permutation.
                    block_start_order = None
                vdam_replay._maybe_write_vdam_candidate_block_map(
                    particle_ids=candidate_trace_ids,
                    reconstruction_take_indices=reconstruction_take_indices,
                    reconstruction_pack_mask=reconstruction_pack_mask_np,
                    reconstruction_contributing_mask=(
                        np.take_along_axis(
                            reconstruction_rotation_mask_np
                            & local_mask_np
                            & (probs_sum_t_np > 0.0),
                            reconstruction_take_indices,
                            axis=1,
                        )
                        & reconstruction_pack_mask_np
                    ),
                    local_rotation_ids=bucket.local_rotation_ids[:unpadded_batch_size],
                    reconstruction_group_ids=(
                        None
                        if bucket_reconstruction_group_ids is None
                        else bucket_reconstruction_group_ids[:unpadded_batch_size]
                    ),
                    candidate_launch_counts=native_grid_counts,
                    debug_iteration=debug_iteration,
                )
                worker_lane_ids = vdam_replay._relion_vdam_worker_lanes_for_images(
                    experiment_dataset,
                    unpadded_bucket.image_indices,
                    debug_iteration=debug_iteration,
                )
                chronology_serial_rotation_replay = bool(
                    vdam_replay._relion_vdam_serial_rotation_replay()
                    or vdam_replay._relion_vdam_captured_block_serial_replay()
                )
                fused_serial_rotations = _env_flag(
                    EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS_ENV
                )
                launch_serial_rotations = _env_flag(
                    EXACT_LOCAL_SOURCE_BPREF_LAUNCH_SERIAL_ROTATIONS_ENV
                )
                if fused_serial_rotations and launch_serial_rotations:
                    raise ValueError(
                        "persistent and launch-serialized VDAM BPref rotations "
                        "are mutually exclusive"
                    )
                serial_rotation_replay = bool(
                    chronology_serial_rotation_replay
                    or fused_serial_rotations
                    or launch_serial_rotations
                )
                float64_accumulator_replay = vdam_replay._relion_vdam_float64_accumulator_replay()
                reverse_rotation_replay = vdam_replay._relion_vdam_reverse_rotation_replay()
                rotation_replay_stride = vdam_replay._relion_vdam_rotation_replay_stride()
                native_trace_shape_replay = vdam_replay._relion_vdam_native_trace_shape_replay(
                    debug_iteration=debug_iteration
                )
                materialized_rotation_replay = vdam_replay._relion_vdam_materialized_native_grid_replay(
                    debug_iteration=debug_iteration
                )
                candidate_trace_active = vdam_replay._relion_vdam_candidate_trace_active(
                    debug_iteration=debug_iteration
                )
                particle_chunk_cap = _source_faithful_bpref_particle_chunk_cap()
                fused_serial_particles = _env_flag(
                    EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES_ENV
                )
                if (fused_serial_rotations or launch_serial_rotations) and (
                    worker_lane_ids is not None
                    or any(
                        value is not None
                        for value in (
                            block_start_order,
                            native_grid_counts,
                            particle_start_offsets_ns,
                            particle_issue_order,
                        )
                    )
                    or any(
                        (
                            chronology_serial_rotation_replay,
                            float64_accumulator_replay,
                            reverse_rotation_replay,
                            native_trace_shape_replay,
                            materialized_rotation_replay,
                        )
                    )
                ):
                    raise ValueError(
                        "fused serial VDAM BPref rotations cannot be combined "
                        "with VDAM chronology replay"
                    )
                if fused_serial_particles and particle_chunk_cap is not None:
                    raise ValueError(
                        "fused serial VDAM BPref particles cannot be combined with "
                        "host particle chunking"
                    )
                particle_slices = _source_faithful_bpref_particle_slices(
                    unpadded_batch_size,
                    particle_chunk_cap,
                )
                particle_chunk_size = particle_slices[0][1] - particle_slices[0][0]
                split_particle_stream = particle_chunk_size < unpadded_batch_size
                if split_particle_stream or fused_serial_particles:
                    if any(
                        value is not None
                        for value in (
                            block_start_order,
                            native_grid_counts,
                            particle_start_offsets_ns,
                            particle_issue_order,
                        )
                    ) or any(
                        (
                            chronology_serial_rotation_replay,
                            float64_accumulator_replay,
                            reverse_rotation_replay,
                            native_trace_shape_replay,
                            materialized_rotation_replay,
                        )
                    ):
                        raise ValueError(
                            "source-faithful BPref particle ordering cannot be combined "
                            "with VDAM chronology replay"
                        )
                if split_particle_stream:
                    if not logged_deferred_mstep_chunking:
                        logger.info(
                            "Exact local VDAM BPref particle chunking: "
                            "particles=%d chunk_particles=%d packed_rows=%d",
                            unpadded_batch_size,
                            particle_chunk_size,
                            int(packed_mstep_rotations_np.shape[1]),
                        )
                        logged_deferred_mstep_chunking = True
                elif (
                    fused_serial_particles
                    or fused_serial_rotations
                    or launch_serial_rotations
                ) and not logged_deferred_mstep_chunking:
                    logger.info(
                        "Exact local VDAM BPref fused serial ordering: "
                        "particles=%d packed_rows=%d serial_particles=%s "
                        "persistent_serial_rotations=%s "
                        "launch_serial_rotations=%s",
                        unpadded_batch_size,
                        int(packed_mstep_rotations_np.shape[1]),
                        fused_serial_particles,
                        fused_serial_rotations,
                        launch_serial_rotations,
                    )
                    logged_deferred_mstep_chunking = True

                def _particle_slice(value, particle_start, particle_stop):
                    if value is None or not split_particle_stream:
                        return value
                    return value[particle_start:particle_stop]

                for particle_start, particle_stop in particle_slices:
                    Ft_y, Ft_ctf = _accumulate_relion_vdam_physical_particle_grid(
                        _particle_slice(
                            packed_source_vdam_images, particle_start, particle_stop
                        ),
                        _particle_slice(
                            packed_source_vdam_ctf, particle_start, particle_stop
                        ),
                        _particle_slice(
                            packed_source_vdam_minvsigma2, particle_start, particle_stop
                        ),
                        _particle_slice(
                            packed_source_vdam_posterior, particle_start, particle_stop
                        ),
                        relion_score_translation_angles,
                        _particle_slice(
                            packed_source_vdam_reference, particle_start, particle_stop
                        ),
                        _particle_slice(
                            packed_mstep_rotations_np, particle_start, particle_stop
                        ),
                        _particle_slice(
                            reconstruction_pack_mask_jnp, particle_start, particle_stop
                        ),
                        Ft_y,
                        Ft_ctf,
                        transaction_queue=bpref_transaction_queue,
                        projector_full=source_vdam_projector_full,
                        scoring_rotations=_particle_slice(
                            packed_rotations_np, particle_start, particle_stop
                        ),
                        projector_r_max=source_vdam_projector_static_radius,
                        runtime_projector_radius=source_vdam_projector_runtime_radius,
                        projection_padding_factor=projection_padding_factor,
                        pixel_indices=mstep_recon_window_indices,
                        image_shape=image_shape,
                        volume_shape=recon_volume_shape,
                        max_r=mstep_adjoint_max_r,
                        stable_dense_positions=(
                            stable_bpref_dense_positions
                            if stable_window_active
                            else None
                        ),
                        logical_current_size=(
                            mstep_current_size if stable_window_active else None
                        ),
                        reconstruction_group_ids=_particle_slice(
                            bucket_reconstruction_group_ids,
                            particle_start,
                            particle_stop,
                        ),
                        worker_lane_ids=_particle_slice(
                            worker_lane_ids, particle_start, particle_stop
                        ),
                        particle_trace_ids=_particle_slice(
                            candidate_trace_ids, particle_start, particle_stop
                        ),
                        serial_rotation_replay=serial_rotation_replay,
                        persistent_serial_rotation_replay=fused_serial_rotations,
                        float64_accumulator_replay=float64_accumulator_replay,
                        reverse_rotation_replay=reverse_rotation_replay,
                        rotation_replay_stride=rotation_replay_stride,
                        rotation_replay_order=_particle_slice(
                            block_start_order, particle_start, particle_stop
                        ),
                        rotation_replay_counts=_particle_slice(
                            native_grid_counts, particle_start, particle_stop
                        ),
                        particle_start_offsets_ns=_particle_slice(
                            particle_start_offsets_ns, particle_start, particle_stop
                        ),
                        native_trace_shape_replay=native_trace_shape_replay,
                        materialized_rotation_replay=materialized_rotation_replay,
                        particle_replay_order=_particle_slice(
                            particle_issue_order, particle_start, particle_stop
                        ),
                        candidate_trace_active=candidate_trace_active,
                        serial_particle_accumulation=fused_serial_particles,
                    )
                if return_profile:
                    _block_until_ready(Ft_y, Ft_ctf)
                timing.adjoint_y_s += time.time() - adjoint_t0
            elif source_faithful_bpref and sparse_big_jit_backprojection:
                adjoint_t0 = time.time()
                Ft_y, Ft_ctf = _accumulate_relion_physical_particle_grid(
                    packed_summed,
                    packed_ctf_probs,
                    packed_mstep_rotations_np,
                    reconstruction_pack_mask_jnp,
                    Ft_y,
                    Ft_ctf,
                    pixel_indices=mstep_recon_window_indices,
                    image_shape=image_shape,
                    volume_shape=recon_volume_shape,
                    max_r=mstep_adjoint_max_r,
                )
                if return_profile:
                    _block_until_ready(Ft_y, Ft_ctf)
                timing.adjoint_y_s += time.time() - adjoint_t0
            elif sparse_big_jit_backprojection and (
                not disable_adjoint_y or not disable_adjoint_ctf
            ):
                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                        flat_packed_summed,
                        mstep_recon_window_indices,
                        packed_flat_rotations,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                        target_rows=sparse_adjoint_target_rows,
                    )
                    if return_profile:
                        _block_until_ready(Ft_y)
                    sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf, n_adjoint_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
                        flat_packed_ctf_probs,
                        mstep_recon_window_indices,
                        packed_flat_rotations,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                        target_rows=sparse_adjoint_target_rows,
                    )
                    if return_profile:
                        _block_until_ready(Ft_ctf)
                    sparse_adjoint_chunk_count += int(n_adjoint_chunks)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            use_packed_final_noise = bool(
                return_big_jit_deferred_mstep_inputs
                and accumulate_noise
                and return_deferred_source_vdam_operands
                and packed_final_noise_enabled
            )
            if noise_stable_core_enabled and not use_packed_final_noise:
                raise ValueError("separate noise core requires a deferred VDAM noise bucket")
            if noise_native_residual_enabled and not use_packed_final_noise:
                raise ValueError("native noise residual requires a deferred VDAM noise bucket")
            if use_packed_final_noise:
                if packed_reconstruction_probs is None:
                    raise RuntimeError(
                        "packed final-support VDAM noise is missing posterior rows"
                    )
                if (
                    packed_source_vdam_noise_projection is None
                    or packed_source_vdam_ctf_probs is None
                ):
                    raise RuntimeError(
                        "packed final-support VDAM noise operands were not built"
                    )
                packed_noise_scale_xa = (
                    noise_scale_xa
                    if noise_scale_xa is not None
                    else disabled_noise_scale
                )
                packed_noise_scale_aa = (
                    noise_scale_aa
                    if noise_scale_aa is not None
                    else disabled_noise_scale
                )
                noise_t0 = time.time()
                prepared_noise_core = None
                noise_pixel_probs = packed_reconstruction_probs
                noise_pixel_projection = packed_source_vdam_noise_projection
                noise_pixel_ctf_probs = packed_source_vdam_ctf_probs
                noise_image_indices = jnp.asarray(bucket_image_indices, dtype=jnp.int32)
                if noise_pixel_capacity_enabled:
                    (
                        noise_pixel_probs,
                        noise_pixel_projection,
                        noise_pixel_ctf_probs,
                        noise_image_indices,
                    ) = pack_noise_pixel_capacity(
                        noise_pixel_probs,
                        noise_pixel_projection,
                        noise_pixel_ctf_probs,
                        noise_image_indices,
                        target_batch=reconstruction_probs.shape[0],
                        n_images=n_images,
                        norm_capacity=noise_norm_correction.shape[0],
                        cuda_packing=noise_pixel_cuda_enabled,
                    )
                deferred_noise_shared_kwargs = dict(
                    image_shape=image_shape,
                    shell_count=n_shells,
                    norm_current_size=(
                        physical_current_size
                        if stable_window_active
                        else current_size
                    ),
                    stable_fourier_window_shapes=stable_window_active,
                    include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
                    use_relion_cuda_powerclass_spectrum=bool(
                        relion_exact_fine_diff2
                        and return_deferred_source_vdam_operands
                    ),
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    unweighted_high_shell_image_power=unweighted_high_shell_image_power,
                )
                if noise_stable_core_enabled:
                    prepared_noise_core = run_deferred_local_exact_noise_core_jit(
                        reconstruction_probs,
                        processed_score_half,
                        image_only_corrections_arg,
                        translation_sqdist_arg,
                        valid_image_mask,
                        shell_indices_half,
                        jnp.asarray(logical_current_size, dtype=jnp.int32),
                        **deferred_noise_shared_kwargs,
                    )
                (
                    noise_wsum,
                    noise_img_power,
                    noise_a2,
                    noise_xa,
                    packed_noise_scale_xa,
                    packed_noise_scale_aa,
                    noise_norm_correction,
                    noise_sigma2_offset,
                    noise_sumw,
                ) = run_deferred_local_exact_noise_jit(
                    noise_wsum,
                    noise_img_power,
                    noise_a2,
                    noise_xa,
                    packed_noise_scale_xa,
                    packed_noise_scale_aa,
                    noise_norm_correction,
                    noise_sigma2_offset,
                    noise_sumw,
                    reconstruction_probs,
                    noise_pixel_probs,
                    noise_pixel_projection,
                    noise_pixel_ctf_probs,
                    shifted_noise_split,
                    processed_score_half,
                    image_only_corrections_arg,
                    translation_sqdist_arg,
                    valid_image_mask,
                    shell_indices_half,
                    shell_indices_noise,
                    noise_variance_for_noise,
                    scale_correction_pixel_mask,
                    group_ids_arg,
                    ctf_rfloat_half_arg,
                    scale_corrections_arg,
                    relion_score_translation_angles,
                    big_jit_relion_wavg_rectangle_indices_arg,
                    big_jit_relion_wavg_exact_positions_arg,
                    big_jit_relion_wavg_rectangle_shell_indices_arg,
                    big_jit_recon_window_indices_arg,
                    noise_image_indices,
                    jnp.asarray(logical_current_size, dtype=jnp.int32),
                    **deferred_noise_shared_kwargs,
                    accumulate_scale_correction=group_ids_np is not None,
                    return_noise_split=return_noise_split,
                    use_relion_wavg_cutoff=bool(
                        relion_exact_fine_diff2
                        and use_window
                        and logical_current_size is not None
                    ),
                    relion_wavg_sequential_cuda=relion_wavg_sequential_cuda,
                    prepared_core=prepared_noise_core,
                    native_residual_statistics=noise_native_residual_enabled,
                )
                if noise_scale_xa is not None:
                    noise_scale_xa = packed_noise_scale_xa
                    noise_scale_aa = packed_noise_scale_aa
                if return_profile:
                    _block_until_ready(
                        noise_wsum,
                        noise_img_power,
                        noise_a2,
                        noise_xa,
                        noise_norm_correction,
                        noise_sigma2_offset,
                        noise_sumw,
                    )
                timing.noise_s += time.time() - noise_t0

            if (
                return_big_jit_deferred_mstep_inputs
                and accumulate_noise
                and not use_packed_final_noise
            ):
                noise_t0 = time.time()
                reconstruction_probs_unpadded = reconstruction_probs[:unpadded_batch_size]
                preserve_dense_noise_reduction = bool(
                    return_deferred_source_vdam_operands
                )
                preserve_dense_scalar_reduction = bool(
                    return_deferred_source_vdam_operands
                )
                if preserve_dense_scalar_reduction:
                    scalar_noise_reconstruction_probs = reconstruction_probs
                    scalar_noise_batch_size = int(batch_size)
                    scalar_noise_valid_image_mask = valid_image_mask
                else:
                    scalar_noise_reconstruction_probs = reconstruction_probs_unpadded
                    scalar_noise_batch_size = int(unpadded_batch_size)
                    scalar_noise_valid_image_mask = jnp.ones(
                        scalar_noise_batch_size,
                        dtype=bool,
                    )
                (
                    support_mass,
                    _translation_posterior,
                    noise_sumw_offset,
                ) = compute_local_noise_scalar_terms(
                    scalar_noise_reconstruction_probs,
                    translation_sqdist_arg[:scalar_noise_batch_size],
                    scalar_noise_valid_image_mask,
                )
                processed_noise_power_half = processed_score_half[
                    :scalar_noise_batch_size
                ]
                processed_noise_power_half = (
                    processed_noise_power_half
                    * image_only_corrections_arg[:scalar_noise_batch_size, None]
                )
                batch_img_power_shells, batch_img_power_per_image = _noise_image_power_shells_and_per_image(
                    processed_noise_power_half,
                    support_mass,
                    shell_indices_half,
                    scalar_noise_valid_image_mask,
                    norm_unweighted_shell_cutoff,
                    shell_count=n_shells,
                    image_shape=image_shape,
                    current_size=(
                        physical_current_size
                        if stable_window_active
                        else current_size
                    ),
                    runtime_current_size=(
                        jnp.asarray(logical_current_size, dtype=jnp.int32)
                        if stable_window_active
                        else None
                    ),
                    include_unweighted_high_shell=include_unweighted_norm_high_shell,
                    use_relion_cuda_powerclass_spectrum=bool(
                        relion_exact_fine_diff2
                        and return_deferred_source_vdam_operands
                    ),
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    unweighted_high_shell_image_power=unweighted_high_shell_image_power,
                )
                noise_sumw = noise_sumw + jnp.sum(support_mass)

                block_noise_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
                block_a2_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
                block_xa_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
                direct_wavg_triplet_shells = jnp.zeros(
                    (3, n_shells),
                    dtype=jnp.float64,
                )
                use_relion_wavg_cutoff = bool(
                    return_deferred_source_vdam_operands
                    and relion_exact_fine_diff2
                    and use_window
                    and logical_current_size is not None
                )
                bucket_group_ids = (
                    group_ids_arg[:unpadded_batch_size] if group_ids_np is not None else None
                )
                if preserve_dense_noise_reduction:
                    flat_row_plan = jnp.asarray(
                        flat_local_row_argument,
                        dtype=jnp.int32,
                    )
                    flat_proj_for_noise = jnp.asarray(
                        deferred_flat_proj_for_noise,
                        dtype=jnp.complex64,
                    )
                    expected_flat_projection_shape = (
                        int(flat_row_plan.shape[0]),
                        window_spec.n_recon
                        if window_spec.use_window
                        else int(n_half),
                    )
                    if flat_proj_for_noise.shape != expected_flat_projection_shape:
                        raise RuntimeError(
                            "deferred VDAM scoring projection did not retain the "
                            "packed reconstruction layout: "
                            f"{flat_proj_for_noise.shape} vs "
                            f"{expected_flat_projection_shape}",
                        )
                    proj_for_noise_reduction = scatter_flat_local_rows(
                        flat_proj_for_noise,
                        flat_row_plan[:, 0],
                        flat_row_plan[:, 1],
                        flat_row_plan[:, 2] != 0,
                        batch_size=int(batch_size),
                        dense_rotation_count=int(bucket.bucket_rotation_count),
                        fill_value=0.0,
                    )
                    probs_for_noise_reduction = reconstruction_probs
                    ctf_probs_for_noise_reduction = (
                        deferred_source_vdam_ctf_probs
                    )
                    processed_score_for_wavg = processed_score_half
                    ctf_rfloat_for_wavg = ctf_rfloat_half_arg
                    scale_for_noise_reduction = scale_corrections_arg
                    pixel_noise_batch_size = int(batch_size)
                    pixel_support_mass = support_mass
                    pixel_noise_valid_image_mask = valid_image_mask

                    shifted_for_noise_reduction = jnp.where(
                        pixel_support_mass[:, None, None] != 0.0,
                        shifted_noise_split[:pixel_noise_batch_size],
                        0.0,
                    )
                    summed_masked_for_noise_reduction = compute_local_weighted_sums(
                        probs_for_noise_reduction,
                        shifted_for_noise_reduction,
                    )
                    flat_proj_for_noise_reduction = flatten_bucket_rows(
                        proj_for_noise_reduction
                    )
                    proj_abs2_for_noise_reduction = (
                        jnp.abs(proj_for_noise_reduction) ** 2
                    )
                    (
                        block_noise_shells,
                        block_a2_shells,
                        block_xa_shells,
                    ) = _compute_noise_block(
                        flat_proj_for_noise_reduction,
                        flatten_bucket_rows(proj_abs2_for_noise_reduction),
                        flatten_bucket_rows(summed_masked_for_noise_reduction),
                        flatten_bucket_rows(ctf_probs_for_noise_reduction),
                        noise_variance_for_noise,
                        shell_indices_noise,
                        n_shells,
                        return_noise_split,
                    )
                    block_norm_residual = _compute_norm_residual_per_image(
                        proj_for_noise_reduction,
                        proj_abs2_for_noise_reduction,
                        summed_masked_for_noise_reduction,
                        ctf_probs_for_noise_reduction,
                        noise_variance_for_noise,
                    )
                    if use_relion_wavg_cutoff:
                        direct_wavg_triplet_shells, _ = (
                            _relion_wavg_direct_triplet_shells(
                                processed_score_for_wavg,
                                relion_score_translation_angles,
                                big_jit_relion_wavg_rectangle_indices_arg,
                                big_jit_relion_wavg_exact_positions_arg,
                                big_jit_relion_wavg_rectangle_shell_indices_arg,
                                big_jit_recon_window_indices_arg,
                                proj_for_noise_reduction,
                                ctf_rfloat_for_wavg,
                                scale_for_noise_reduction,
                                probs_for_noise_reduction,
                                pixel_noise_valid_image_mask,
                                image_shape=image_shape,
                                shell_count=n_shells,
                                cutoff_shell=int(logical_current_size) // 2,
                                relion_wavg_sequential_cuda=(
                                    relion_wavg_sequential_cuda
                                ),
                                logical_recon_pixel_count=(
                                    stable_window_plan.logical_reconstruction_pixels
                                    if stable_window_active
                                    else None
                                ),
                                logical_rectangle_pixel_count=(
                                    stable_window_plan.logical_rectangle_pixels
                                    if stable_window_active
                                    else None
                                ),
                            )
                        )
                    if noise_scale_xa is not None:
                        scale_xa_per_image, scale_aa_per_image = _compute_scale_correction_terms_per_image(
                            proj_for_noise_reduction,
                            proj_abs2_for_noise_reduction,
                            summed_masked_for_noise_reduction,
                            ctf_probs_for_noise_reduction,
                            noise_variance_for_noise,
                            scale_for_noise_reduction,
                            scale_correction_pixel_mask,
                        )
                        noise_scale_xa = add_segment_sum(
                            noise_scale_xa,
                            bucket_group_ids,
                            scale_xa_per_image[:unpadded_batch_size].astype(noise_scale_xa.dtype),
                        )
                        noise_scale_aa = add_segment_sum(
                            noise_scale_aa,
                            bucket_group_ids,
                            scale_aa_per_image[:unpadded_batch_size].astype(noise_scale_aa.dtype),
                        )
                else:
                    shifted_noise_split_unpadded = shifted_noise_split[
                        :unpadded_batch_size
                    ]
                    ctf2_over_nv_recon_unpadded = ctf2_over_nv_recon[
                        :unpadded_batch_size
                    ]
                    packed_rotation_count = int(packed_rotations_np.shape[1])
                    n_recon_pixels = (
                        window_spec.n_recon
                        if window_spec.use_window
                        else int(n_half)
                    )
                    noise_projection_pixels = (
                        int(n_half)
                        if relion_projector_half is not None
                        else int(n_recon_pixels)
                    )
                    chunk_rows = min(
                        packed_rotation_count,
                        _packed_noise_projection_chunk_rows(
                            noise_projection_pixels,
                            batch_size=unpadded_batch_size,
                        ),
                    )
                    if (
                        chunk_rows < packed_rotation_count
                        and not logged_deferred_noise_projection_chunking
                    ):
                        logger.info(
                            "Exact local big-JIT deferred noise projection chunking: "
                            "packed_rows=%d chunk_rows=%d n_recon_pixels=%d "
                            "projection_pixels=%d batch_size=%d",
                            packed_rotation_count,
                            chunk_rows,
                            n_recon_pixels,
                            noise_projection_pixels,
                            unpadded_batch_size,
                        )
                        logged_deferred_noise_projection_chunking = True
                    block_norm_residual = jnp.zeros(
                        unpadded_batch_size,
                        dtype=precision_policy.score_real_dtype,
                    )
                    batch_scale_unpadded = scale_corrections_arg[
                        :unpadded_batch_size
                    ]
                    for chunk_start in range(
                        0,
                        packed_rotation_count,
                        chunk_rows,
                    ):
                        chunk_stop = min(
                            packed_rotation_count,
                            chunk_start + chunk_rows,
                        )
                        chunk_rotations = packed_rotations_np[
                            :,
                            chunk_start:chunk_stop,
                        ]
                        chunk_proj_for_noise = _project_packed_noise_rows(
                            mean_for_proj=mean_for_proj,
                            packed_flat_rotations=flatten_bucket_rotations(
                                jnp.asarray(chunk_rotations)
                            ),
                            packed_rotation_count=chunk_stop - chunk_start,
                            batch_size=unpadded_batch_size,
                            image_shape=image_shape,
                            proj_volume_shape=proj_volume_shape,
                            disc_type=disc_type,
                            projection_kwargs=projection_kwargs,
                            window_spec=window_spec,
                            n_half=n_half,
                            precision_policy=precision_policy,
                            reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[
                                :,
                                chunk_start:chunk_stop,
                            ],
                            relion_projector_half=relion_projector_half,
                            relion_projector_r_max=relion_projector_r_max,
                            projection_padding_factor=projection_padding_factor,
                        )
                        chunk_probs = packed_reconstruction_probs[
                            :,
                            chunk_start:chunk_stop,
                        ]
                        chunk_probs_sum_t = packed_reconstruction_probs_sum_t[
                            :,
                            chunk_start:chunk_stop,
                        ]
                        chunk_summed_masked_noise = compute_local_weighted_sums(
                            chunk_probs,
                            shifted_noise_split_unpadded,
                        )
                        chunk_ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
                            chunk_probs_sum_t,
                            ctf2_over_nv_recon_unpadded,
                        )
                        flat_proj_for_noise = flatten_bucket_rows(
                            chunk_proj_for_noise
                        )
                        flat_proj_abs2_for_noise = (
                            jnp.abs(flat_proj_for_noise) ** 2
                        )
                        (
                            chunk_noise_shells,
                            chunk_a2_shells,
                            chunk_xa_shells,
                        ) = _compute_noise_block(
                            flat_proj_for_noise,
                            flat_proj_abs2_for_noise,
                            flatten_bucket_rows(chunk_summed_masked_noise),
                            flatten_bucket_rows(chunk_ctf_probs),
                            noise_variance_for_noise,
                            shell_indices_noise,
                            n_shells,
                            return_noise_split,
                        )
                        block_noise_shells = (
                            block_noise_shells + chunk_noise_shells
                        )
                        block_a2_shells = block_a2_shells + chunk_a2_shells
                        block_xa_shells = block_xa_shells + chunk_xa_shells
                        chunk_proj_abs2_for_norm = (
                            jnp.abs(chunk_proj_for_noise) ** 2
                        )
                        block_norm_residual = (
                            block_norm_residual
                            + _compute_norm_residual_per_image(
                                chunk_proj_for_noise,
                                chunk_proj_abs2_for_norm,
                                chunk_summed_masked_noise,
                                chunk_ctf_probs,
                                noise_variance_for_noise,
                            )
                        )
                        if noise_scale_xa is not None:
                            (
                                scale_xa_per_image,
                                scale_aa_per_image,
                            ) = _compute_scale_correction_terms_per_image(
                                chunk_proj_for_noise,
                                chunk_proj_abs2_for_norm,
                                chunk_summed_masked_noise,
                                chunk_ctf_probs,
                                noise_variance_for_noise,
                                batch_scale_unpadded,
                                scale_correction_pixel_mask,
                            )
                            noise_scale_xa = noise_scale_xa.at[
                                bucket_group_ids
                            ].add(scale_xa_per_image.astype(noise_scale_xa.dtype))
                            noise_scale_aa = noise_scale_aa.at[
                                bucket_group_ids
                            ].add(scale_aa_per_image.astype(noise_scale_aa.dtype))
                if use_relion_wavg_cutoff:
                    cutoff_mask = (
                        jnp.arange(n_shells, dtype=jnp.int32)
                        == int(current_size) // 2
                    )
                    block_noise_shells = jnp.where(
                        cutoff_mask,
                        direct_wavg_triplet_shells[2],
                        block_noise_shells,
                    )
                    batch_img_power_shells = jnp.where(
                        cutoff_mask,
                        0.0,
                        batch_img_power_shells,
                    )
                    if return_noise_split:
                        block_a2_shells = jnp.where(
                            cutoff_mask,
                            direct_wavg_triplet_shells[1],
                            block_a2_shells,
                        )
                        block_xa_shells = jnp.where(
                            cutoff_mask,
                            direct_wavg_triplet_shells[0],
                            block_xa_shells,
                        )
                if return_profile:
                    _block_until_ready(block_noise_shells, block_norm_residual)
                noise_wsum = noise_wsum + block_noise_shells
                noise_img_power = noise_img_power + batch_img_power_shells
                if return_noise_split:
                    noise_a2 = noise_a2 + block_a2_shells
                    noise_xa = noise_xa + block_xa_shells
                bucket_norm_rows = batch_img_power_per_image + block_norm_residual
                if preserve_dense_noise_reduction:
                    bucket_norm_rows = bucket_norm_rows[:unpadded_batch_size]
                noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket_image_indices, dtype=jnp.int32)].add(
                    bucket_norm_rows.astype(noise_norm_correction.dtype),
                )
                noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
                timing.noise_s += time.time() - noise_t0

            postprocess_t0 = time.time()
            stats_probs_sum_t = reconstruction_probs_sum_t if stats_use_reconstruction_probs else probs_sum_t
            # The postprocessor already needs these small arrays on the host.
            # Preserve their physical shapes until that consumer when enabled.
            def postprocess_rows(value):
                return value if host_publication_enabled else value[:unpadded_batch_size]

            stats_probs_sum_t_np = (
                None
                if probs_sum_t_np is None
                else np.asarray(postprocess_rows(stats_probs_sum_t), dtype=np.float64)[:unpadded_batch_size]
            )
            significant_sample_count, reconstruction_row_count = _postprocess_local_bucket(
                image_indices=unpadded_bucket.image_indices,
                **_unpadded_bucket_rows(bucket, unpadded_batch_size),
                translation_grid=local_layout.translation_grid,
                n_trans=n_trans,
                best_argmax=postprocess_rows(best_argmax),
                batch_norm=postprocess_rows(batch_norm),
                log_Z=postprocess_rows(log_Z),
                best_log_score=postprocess_rows(best_log_score),
                max_posterior=postprocess_rows(max_posterior),
                probs_sum_t=(
                    postprocess_rows(stats_probs_sum_t) if stats_probs_sum_t_np is None else stats_probs_sum_t_np
                ),
                n_significant_samples=postprocess_rows(n_significant_samples),
                reconstruction_sample_mask=(
                    None
                    if host_publication_enabled and postprocess_buffers.reconstruction_sample_indices_by_image is None
                    else postprocess_rows(reconstruction_sample_mask)
                ),
                collect_profile_stats=collect_profile_stats,
                reconstruction_row_count=reconstruction_row_count,
                reconstruction_take_indices=reconstruction_take_indices,
                reconstruction_pack_mask=reconstruction_pack_mask_np,
                buffers=postprocess_buffers,
                host_prefix=host_publication_enabled,
            )
            if collect_profile_stats:
                total_significant_samples += significant_sample_count
                total_reconstruction_rows += int(reconstruction_row_count)
            timing.postprocess_s += time.time() - postprocess_t0

            host_stats_t0 = time.time()
            logger.debug(
                "Exact local big-JIT bucket: %d images, bucket_rot=%d, total_local_rot=%d",
                unpadded_batch_size,
                int(bucket.bucket_rotation_count),
                int(np.sum(unpadded_bucket.actual_rotation_counts)),
            )
            timing.host_stats_s += time.time() - host_stats_t0
            local_progress.mark_bucket_done(bucket)
            continue

        preprocess_t0 = time.time()
        (
            shifted_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_score_half,
            ctf2_over_nv_recon_half,
            processed_score_half,
            real_space_pre_shift_applied,
        ) = local_preprocessing.prepare_local_bucket(
            experiment_dataset,
            batch_data,
            ctf_params,
            bucket.image_indices,
            noise_variance_half,
            translation_phases_half,
            config,
            norm_half_weights,
            score_with_masked_images,
            relion_score_translation_angles=relion_score_translation_angles,
            image_pre_shifts=image_pre_shifts,
            processed_half_cache=processed_half_cache,
            timer=preprocess_profile if return_profile else None,
            synchronize_profile=return_profile,
            score_complex_dtype=precision_policy.score_complex_dtype,
            score_real_dtype=precision_policy.score_real_dtype,
            norm_real_dtype=precision_policy.normalization_real_dtype,
            relion_exact_bpref_operands=relion_exact_bpref_operands,
        )
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(bucket.image_indices)])
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)
        bucket_group_ids = (
            jnp.asarray(group_ids_np[np.asarray(bucket.image_indices)], dtype=jnp.int32)
            if group_ids_np is not None
            else None
        )

        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(bucket.image_indices)])
            image_only_corr = batch_corr / batch_scale
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
            batch_norm = batch_norm * (image_only_corr**2)[:, None]
        else:
            batch_corr = None
            image_only_corr = None

        if scale_corrections is not None:
            ctf2_over_nv_score_half = ctf2_over_nv_score_half * (batch_scale**2)[:, None]
            ctf2_over_nv_recon_half = ctf2_over_nv_recon_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(bucket.image_indices)])
            phase_expanded = tiled_half_image_phase_factors(
                image_shape, batch_shifts, n_trans, dtype=batch_shifts.dtype
            )
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded
        shifted_half_with_dc = shifted_half
        ctf2_over_nv_recon_half_with_dc = ctf2_over_nv_recon_half

        if half_spectrum_scoring:
            dc_mask = make_shell_indices_half(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_score_half = jnp.where(
                dc_mask[None, :], 0.0, ctf2_over_nv_score_half
            )

        if use_window:
            shifted_score = shifted_half[:, window_indices]
            shifted_recon = shifted_recon_half[:, recon_window_indices]
            ctf2_over_nv_score = ctf2_over_nv_score_half[:, window_indices]
            ctf2_over_nv_recon = ctf2_over_nv_recon_half_with_dc[:, recon_window_indices]
            shifted_noise = shifted_half_with_dc[:, recon_window_indices]
        else:
            shifted_score = shifted_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_score = ctf2_over_nv_score_half
            ctf2_over_nv_recon = ctf2_over_nv_recon_half_with_dc
            shifted_noise = shifted_half_with_dc

        (
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        ) = precision_policy.cast_local_preprocessed_inputs(
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        )
        timing.preprocess_s += time.time() - preprocess_t0

        projection_t0 = time.time()
        projection_block = _project_local_bucket(
            mean_for_proj=mean_for_proj,
            bucket=bucket,
            image_shape=image_shape,
            proj_volume_shape=proj_volume_shape,
            disc_type=disc_type,
            projection_kwargs=projection_kwargs,
            window_spec=window_spec,
            n_half=n_half,
            half_weights=half_weights,
            precision_policy=precision_policy,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            materialize_recon_projection=need_local_recon_projection_for_bucket,
        )
        proj_weighted = projection_block.proj_weighted
        proj_for_noise = projection_block.proj_for_noise
        if return_profile:
            _block_until_ready(proj_weighted)
        timing.projection_s += time.time() - projection_t0

        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
        local_rotation_log_prior = jnp.asarray(bucket.local_rotation_log_prior)
        bucket_translation_log_prior = jnp.asarray(bucket.translation_log_prior)
        bucket_local_rotation_mask = jnp.asarray(bucket.local_rotation_mask)
        bucket_local_sample_mask = None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask)
        if class_log_prior != 0.0:
            local_rotation_log_prior = local_rotation_log_prior + jnp.asarray(
                class_log_prior,
                dtype=local_rotation_log_prior.dtype,
            )
        defer_packed_mstep_reduction = False
        has_external_normalization = (
            normalization_log_z_np is not None
            or normalization_log_evidence_np is not None
            or normalization_max_posterior_np is not None
        )
        can_use_fused_score_mstep = (
            fused_score_mstep_enabled
            and normalization_max_posterior_np is None
            and reconstruction_probability_threshold_np is None
            and not debug_score_dump_bucket_matches
            and not bpref_contribution_capture_active
        )
        defer_packed_mstep_requested = _env_flag(EXACT_LOCAL_DEFER_PACKED_MSTEP_ENV)
        threshold_for_bucket = (
            None
            if reconstruction_probability_threshold_np is None
            else jnp.asarray(reconstruction_probability_threshold_np[np.asarray(bucket.image_indices)], dtype=jnp.float64)
        )
        used_fused_score_mstep = True
        fused_score_operands = (
            shifted_score_split,
            ctf2_over_nv_score,
            proj_weighted,
            half_weights_windowed if use_window else half_weights,
            local_rotation_log_prior,
            bucket_translation_log_prior,
            bucket_local_rotation_mask,
            bucket_local_sample_mask,
        )
        if can_use_fused_score_mstep and score_only and not has_external_normalization:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_abs2_on_demand(
                *fused_score_operands,
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            reconstruction_probs = None
            summed = None
            ctf_probs = None
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
        elif can_use_fused_score_mstep and not has_external_normalization and defer_packed_mstep_requested:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_probs_abs2_on_demand(
                *fused_score_operands,
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            summed = None
            ctf_probs = None
            defer_packed_mstep_reduction = True
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
        elif can_use_fused_score_mstep and not has_external_normalization:
            fused_t0 = time.time()
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
                summed,
                ctf_probs,
            ) = fused_score_normalize_mstep_abs2_on_demand(
                *fused_score_operands,
                shifted_recon_split,
                ctf2_over_nv_recon,
                None,
                half_spectrum_scoring=half_spectrum_scoring,
                use_float64_normalization=use_float64_normalization,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            if mstep_subtract_ctf_projection:
                # RELION's VDAM/--grad path backprojects the residual image,
                # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                if proj_for_noise is None:
                    raise RuntimeError("Residual local M-step requires materialized recon projections")
                frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
                summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted
            if return_profile:
                _block_until_ready(
                    summed,
                    ctf_probs,
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
        elif can_use_fused_score_mstep and not score_only:
            fused_t0 = time.time()
            if normalization_log_evidence_np is None:
                bucket_log_z = jnp.asarray(
                    normalization_log_z_np[np.asarray(bucket.image_indices)],
                    dtype=precision_policy.normalization_real_dtype,
                )
            else:
                normalization_dtype = precision_policy.normalization_real_dtype
                log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
                bucket_log_z = (
                    jnp.asarray(
                        normalization_log_evidence_np[np.asarray(bucket.image_indices)],
                        dtype=normalization_dtype,
                    )
                    - log_score_offset
                )
            (
                log_Z,
                probs,
                best_log_score,
                best_argmax,
                max_posterior,
                reconstruction_sample_mask,
                reconstruction_rotation_mask,
                n_significant_samples,
                reconstruction_probs,
                probs_sum_t,
                reconstruction_probs_sum_t,
            ) = fused_score_normalize_support_probs_abs2_with_log_z_on_demand(
                *fused_score_operands,
                bucket_log_z,
                half_spectrum_scoring=half_spectrum_scoring,
                reconstruct_significant_only=reconstruct_significant_only,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            summed = None
            ctf_probs = None
            defer_packed_mstep_reduction = True
            if return_profile:
                _block_until_ready(
                    probs_sum_t,
                    reconstruction_probs_sum_t,
                    reconstruction_probs,
                    reconstruction_rotation_mask,
                    n_significant_samples,
                    best_argmax,
                    log_Z,
                    best_log_score,
                    max_posterior,
                )
        else:
            used_fused_score_mstep = False
            score_t0 = time.time()
            if half_spectrum_scoring:
                scores = score_local_bucket_abs2_on_demand(
                    shifted_score_split,
                    ctf2_over_nv_score,
                    proj_weighted,
                    local_rotation_log_prior,
                    bucket_translation_log_prior,
                    bucket_local_rotation_mask,
                    bucket_local_sample_mask,
                )
            else:
                score_half_weights = half_weights_windowed if use_window else half_weights
                scores = score_local_bucket_abs2_weighted_on_demand(
                    shifted_score_split,
                    ctf2_over_nv_score,
                    proj_weighted,
                    score_half_weights,
                    local_rotation_log_prior,
                    bucket_translation_log_prior,
                    bucket_local_rotation_mask,
                    bucket_local_sample_mask,
                )
            if return_profile:
                _block_until_ready(scores)
            timing.score_s += time.time() - score_t0

            normalize_t0 = time.time()
            if (
                normalization_log_z_np is None
                and normalization_log_evidence_np is None
                and normalization_max_posterior_np is None
            ):
                if use_float64_normalization:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores(scores)
                else:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores_float32(scores)
            else:
                if normalization_max_posterior_np is not None:
                    normalization_dtype = precision_policy.normalization_real_dtype
                    fine_best = jnp.max(scores.reshape(scores.shape[0], -1), axis=1).astype(
                        normalization_dtype,
                    )
                    bucket_pmax = jnp.asarray(
                        normalization_max_posterior_np[np.asarray(bucket.image_indices)],
                        dtype=normalization_dtype,
                    )
                    bucket_log_z = fine_best - jnp.log(bucket_pmax)
                elif normalization_log_evidence_np is None:
                    bucket_log_z = jnp.asarray(
                        normalization_log_z_np[np.asarray(bucket.image_indices)],
                        dtype=scores.real.dtype,
                    )
                else:
                    normalization_dtype = precision_policy.normalization_real_dtype
                    log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
                    bucket_log_z = (
                        jnp.asarray(
                            normalization_log_evidence_np[np.asarray(bucket.image_indices)],
                            dtype=normalization_dtype,
                        )
                        - log_score_offset
                    )
                if use_float64_normalization:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores_with_log_z(
                        scores,
                        bucket_log_z,
                    )
                else:
                    log_Z, probs, best_log_score, best_argmax, max_posterior = (
                        normalize_local_scores_with_log_z_float32(
                            scores,
                            bucket_log_z,
                        )
                    )
            if return_profile:
                _block_until_ready(log_Z, probs, best_log_score, best_argmax, max_posterior)
            timing.normalize_s += time.time() - normalize_t0

            significance_t0 = time.time()
            if reconstruct_significant_only and use_relion_f32_fine_posterior:
                if threshold_for_bucket is not None:
                    raise ValueError(
                        "RELION float32 fine posterior does not accept an external "
                        "reconstruction threshold"
                    )
                (
                    reconstruction_probs,
                    reconstruction_sample_mask,
                    n_significant_samples,
                    _relion_sum_weight,
                    _relion_significant_weight,
                ) = sparse_pass2_bucketed._relion_f32_fine_reconstruction_probs(
                    scores,
                    adaptive_fraction=adaptive_fraction,
                )
                reconstruction_rotation_mask = jnp.any(
                    reconstruction_sample_mask,
                    axis=-1,
                )
                max_posterior = jnp.max(
                    reconstruction_probs.reshape(reconstruction_probs.shape[0], -1),
                    axis=1,
                )
            elif reconstruct_significant_only:
                if threshold_for_bucket is None:
                    reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = (
                        compute_reconstruction_support(
                            probs,
                            adaptive_fraction=adaptive_fraction,
                            max_significants=max_significants,
                        )
                    )
                else:
                    reconstruction_sample_mask, reconstruction_rotation_mask, n_significant_samples = (
                        compute_reconstruction_support_from_threshold(
                            probs,
                            threshold_for_bucket,
                        )
                    )
                reconstruction_probs = jnp.where(reconstruction_sample_mask, probs, 0.0)
            else:
                reconstruction_rotation_mask = bucket_local_rotation_mask
                reconstruction_sample_mask = jnp.broadcast_to(
                    reconstruction_rotation_mask[:, :, None],
                    probs.shape,
                )
                n_significant_samples = jnp.sum(reconstruction_rotation_mask, axis=1).astype(jnp.int32) * n_trans
                reconstruction_probs = probs
            if return_profile:
                _block_until_ready(reconstruction_probs, reconstruction_rotation_mask, n_significant_samples)
            timing.significance_s += time.time() - significance_t0

            debug_score_dump_targets = maybe_write_debug_score_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                scores=scores,
                probs=probs,
                reconstruction_probs=reconstruction_probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                debug_pass_label=debug_pass_label,
                shifted_score_split=shifted_score.reshape(batch_size, n_trans, -1),
                shifted_recon_split=shifted_recon_split,
                ctf2_over_nv_score=ctf2_over_nv_score,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                proj_weighted=proj_weighted,
                proj_for_noise=proj_for_noise,
                proj_abs2_weighted=None,
                dump_dir=debug_score_dump_dir,
                pending_targets=debug_score_dump_targets,
                requested_current_sizes=debug_score_dump_current_sizes,
                requested_iterations=debug_score_dump_iterations,
            )

            mstep_t0 = time.time()
            probs_sum_t = jnp.sum(probs, axis=-1)
            reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
            if score_only:
                summed = None
                ctf_probs = None
                if return_profile:
                    _block_until_ready(probs_sum_t, reconstruction_probs_sum_t)
            else:
                summed = compute_local_weighted_sums(reconstruction_probs, shifted_recon_split)
                ctf_probs = compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
                if mstep_subtract_ctf_projection:
                    # RELION's VDAM/--grad path backprojects the residual image,
                    # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                    if proj_for_noise is None:
                        raise RuntimeError("Residual local M-step requires materialized recon projections")
                    frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
                    summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted
                if return_profile:
                    _block_until_ready(summed, ctf_probs, probs_sum_t, reconstruction_probs_sum_t)
            timing.mstep_s += time.time() - mstep_t0

            if bpref_contribution_capture_active and not score_only:
                capture_priors = _bpref_capture_priors(
                    scores,
                    probs.shape,
                    bucket=bucket,
                    rotation_log_prior=local_rotation_log_prior,
                )
                reconstruction_threshold_for_dump = (
                    jnp.zeros((batch_size,), dtype=jnp.float64)
                    if threshold_for_bucket is None
                    else threshold_for_bucket
                )
                _maybe_dump_exact_local_bpref_contribution_rows(
                    experiment_dataset=experiment_dataset,
                    image_indices=bucket.image_indices,
                    current_size=current_size,
                    summed=summed,
                    ctf_probs=ctf_probs,
                    rotations=_local_mstep_rotations(bucket),
                    actual_counts=bucket.actual_rotation_counts,
                    rotation_indices=bucket.local_rotation_ids,
                    fine_translations=local_layout.translation_grid,
                    scores=scores,
                    preprior_scores=capture_priors.preprior_scores,
                    probs=probs,
                    rotation_log_prior=capture_priors.rotation_log_prior,
                    translation_log_prior=capture_priors.translation_log_prior,
                    log_z=log_Z,
                    best_log_score=best_log_score,
                    reconstruction_probs=reconstruction_probs,
                    reconstruction_mask=reconstruction_sample_mask,
                    reconstruction_sum_weight=jnp.sum(reconstruction_probs, axis=(1, 2)),
                    reconstruction_threshold=reconstruction_threshold_for_dump,
                    candidate_mask=capture_priors.candidate_mask,
                    **capture_static_kwargs,
                    reconstruction_group_ids=bucket_reconstruction_group_ids,
                )
            scores = None

        if used_fused_score_mstep:
            debug_fused_posterior_dump_targets = maybe_write_debug_fused_posterior_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                probs=probs,
                log_Z=log_Z,
                best_log_score=best_log_score,
                best_argmax=best_argmax,
                max_posterior=max_posterior,
                reconstruction_sample_mask=reconstruction_sample_mask,
                reconstruction_rotation_mask=reconstruction_rotation_mask,
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                dump_dir=debug_fused_posterior_dump_dir,
                pending_targets=debug_fused_posterior_dump_targets,
                requested_current_sizes=debug_fused_posterior_dump_current_sizes,
                requested_iterations=debug_fused_posterior_dump_iterations,
            )
            fused_elapsed = time.time() - fused_t0
            timing.fused_score_mstep_s += fused_elapsed

        if source_faithful_bpref and not score_only:
            # RELION carries one float32 numerator and denominator through the
            # translation loop for every orientation/pixel. Recompute this
            # narrow boundary after any fused score path so GEMM/reduce_sum
            # ordering cannot leak into the physical particle scatter.
            mstep_t0 = time.time()
            summed, ctf_probs = compute_local_mstep_sums(
                reconstruction_probs,
                shifted_recon_split,
                ctf2_over_nv_recon,
                relion_x_half=True,
                sequential_translation_reduction=True,
            )
            if mstep_subtract_ctf_projection:
                if proj_for_noise is None:
                    raise RuntimeError(
                        "Residual local M-step requires materialized recon projections"
                    )
                reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
                frefctf_weighted = proj_for_noise * ctf2_over_nv_recon[:, None, :]
                summed = summed - reconstruction_probs_sum_t[..., None] * frefctf_weighted
            defer_packed_mstep_reduction = False
            if return_profile:
                _block_until_ready(summed, ctf_probs)
            timing.mstep_s += time.time() - mstep_t0

        _collect_reconstruction_probability_values(bucket.image_indices, probs)

        pack_t0 = time.time()
        probs_sum_t_np = None
        if reconstruct_significant_only:
            reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
            transfer_profile["reconstruction_mask_to_host_s"] += time.time() - pack_t0
        else:
            reconstruction_rotation_mask_np = np.asarray(bucket.local_rotation_mask, dtype=bool)
        transfer_t0 = time.time()
        probs_sum_t_np = np.asarray(probs_sum_t, dtype=np.float64)
        stats_probs_sum_t_np = (
            np.asarray(reconstruction_probs_sum_t, dtype=np.float64)
            if stats_use_reconstruction_probs
            else probs_sum_t_np
        )
        transfer_profile["mstep_posterior_sum_to_host_s"] += time.time() - transfer_t0
        (
            reconstruction_take_indices,
            reconstruction_pack_mask_np,
            reconstruction_counts_np,
            reconstruction_row_count,
        ) = _build_nonzero_reconstruction_pack_indices(
            reconstruction_rotation_mask_np,
            np.asarray(bucket.local_rotation_mask, dtype=bool),
            probs_sum_t_np,
            rotation_block_size,
            exact_local_bucket_radix=resolved_exact_local_bucket_radix,
        )
        (
            reconstruction_take_indices_jnp,
            reconstruction_pack_mask_jnp,
            packed_rotations_np,
            packed_mstep_rotations_np,
        ) = _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np)
        packed_reconstruction_probs = None
        if score_only:
            packed_summed = None
            packed_ctf_probs = None
        elif defer_packed_mstep_reduction:
            packed_reconstruction_probs = _packed_reconstruction_rows(reconstruction_probs, reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
            if mstep_subtract_ctf_projection:
                # RELION's VDAM/--grad path backprojects the residual image,
                # not the raw unmasked image: Fimg_store = Fimg - Frefctf.
                if proj_for_noise is None:
                    raise RuntimeError("Residual local M-step requires materialized recon projections")
            packed_summed = None
            packed_ctf_probs = None
        else:
            packed_summed = _packed_reconstruction_rows(summed, reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
            packed_ctf_probs = _packed_reconstruction_rows(ctf_probs, reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
        packed_flat_rotations = None
        if (not defer_packed_mstep_reduction) and (
            not disable_adjoint_y or not disable_adjoint_ctf or (accumulate_noise and proj_for_noise is None)
        ):
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_mstep_rotations_np))
        timing.pack_s += time.time() - pack_t0

        if source_faithful_bpref and not score_only:
            adjoint_t0 = time.time()
            Ft_y, Ft_ctf = _accumulate_relion_physical_particle_grid(
                packed_summed,
                packed_ctf_probs,
                packed_mstep_rotations_np,
                reconstruction_pack_mask_jnp,
                Ft_y,
                Ft_ctf,
                pixel_indices=mstep_recon_window_indices,
                image_shape=image_shape,
                volume_shape=recon_volume_shape,
                max_r=mstep_adjoint_max_r,
            )
            if return_profile:
                _block_until_ready(Ft_y, Ft_ctf)
            timing.adjoint_y_s += time.time() - adjoint_t0
        elif defer_packed_mstep_reduction and not score_only and (
            not disable_adjoint_y or not disable_adjoint_ctf
        ):
            if packed_reconstruction_probs is None:
                raise RuntimeError("packed posterior rows are required for deferred local M-step")
            packed_rotation_count = int(packed_rotations_np.shape[1])
            n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
            chunk_rows = min(
                packed_rotation_count,
                _packed_noise_projection_chunk_rows(n_recon_pixels, batch_size=batch_size),
            )
            if chunk_rows < packed_rotation_count and not logged_deferred_mstep_chunking:
                logger.info(
                    "Exact local packed M-step chunking: packed_rows=%d chunk_rows=%d n_recon_pixels=%d batch_size=%d",
                    packed_rotation_count,
                    chunk_rows,
                    n_recon_pixels,
                    batch_size,
                )
                logged_deferred_mstep_chunking = True
            for chunk_start in range(0, packed_rotation_count, chunk_rows):
                chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                chunk_probs = packed_reconstruction_probs[:, chunk_start:chunk_stop]
                chunk_rotations = packed_mstep_rotations_np[:, chunk_start:chunk_stop]
                chunk_flat_rotations = flatten_bucket_rotations(jnp.asarray(chunk_rotations))
                chunk_summed = None
                chunk_ctf_probs = None
                if not disable_adjoint_y:
                    mstep_t0 = time.time()
                    chunk_summed = compute_local_weighted_sums(chunk_probs, shifted_recon_split)
                    if mstep_subtract_ctf_projection:
                        chunk_take_indices = reconstruction_take_indices_jnp[:, chunk_start:chunk_stop]
                        chunk_pack_mask = reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop]
                        chunk_proj_for_residual = _packed_reconstruction_rows(proj_for_noise, chunk_take_indices, chunk_pack_mask)
                        chunk_probs_sum_t = jnp.sum(chunk_probs, axis=-1)
                        frefctf_weighted = chunk_proj_for_residual * ctf2_over_nv_recon[:, None, :]
                        chunk_summed = chunk_summed - chunk_probs_sum_t[..., None] * frefctf_weighted
                    if return_profile:
                        _block_until_ready(chunk_summed)
                    timing.mstep_s += time.time() - mstep_t0

                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_maybe_windowed(
                        flatten_bucket_rows(chunk_summed),
                        mstep_recon_window_indices,
                        chunk_flat_rotations,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        True,
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                    )
                    if return_profile:
                        _block_until_ready(Ft_y)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    mstep_t0 = time.time()
                    chunk_ctf_probs = compute_local_ctf_sums(chunk_probs, ctf2_over_nv_recon)
                    if return_profile:
                        _block_until_ready(chunk_ctf_probs)
                    timing.mstep_s += time.time() - mstep_t0

                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                        flatten_bucket_rows(chunk_ctf_probs),
                        mstep_recon_window_indices,
                        chunk_flat_rotations,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        True,
                        use_window=use_window,
                        max_r=mstep_adjoint_max_r,
                        relion_x_half=bool(mstep_relion_x_half),
                    )
                    if return_profile:
                        _block_until_ready(Ft_ctf)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

        elif not disable_adjoint_y:
            adjoint_y_t0 = time.time()
            Ft_y = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_summed),
                mstep_recon_window_indices,
                packed_flat_rotations,
                Ft_y,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=mstep_adjoint_max_r,
                relion_x_half=bool(mstep_relion_x_half),
            )
            if return_profile:
                _block_until_ready(Ft_y)
            timing.adjoint_y_s += time.time() - adjoint_y_t0

        if (
            not source_faithful_bpref
            and not defer_packed_mstep_reduction
            and not disable_adjoint_ctf
        ):
            adjoint_ctf_t0 = time.time()
            Ft_ctf = _adjoint_slice_volume_maybe_windowed(
                flatten_bucket_rows(packed_ctf_probs),
                mstep_recon_window_indices,
                packed_flat_rotations,
                Ft_ctf,
                image_shape,
                recon_volume_shape,
                "linear_interp",
                True,
                True,
                use_window=use_window,
                max_r=mstep_adjoint_max_r,
                relion_x_half=bool(mstep_relion_x_half),
            )
            if return_profile:
                _block_until_ready(Ft_ctf)
            timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

        if accumulate_noise:
            noise_t0 = time.time()
            support_mass = jnp.sum(reconstruction_probs.reshape(batch_size, -1), axis=1).astype(
                precision_policy.score_real_dtype
            )
            if translation_sqdist_ang is not None:
                translation_posterior = jnp.sum(reconstruction_probs, axis=1).astype(precision_policy.score_real_dtype)
                noise_sumw_offset = jnp.sum(
                    translation_posterior * jnp.asarray(translation_sqdist_ang, dtype=precision_policy.score_real_dtype),
                )
            else:
                noise_sumw_offset = jnp.asarray(0.0, dtype=precision_policy.score_real_dtype)
            processed_noise_power_half = processed_score_half
            if image_only_corr is not None:
                processed_noise_power_half = processed_noise_power_half * image_only_corr[:, None]
            batch_img_power_shells, batch_img_power_per_image = _noise_image_power_shells_and_per_image(
                processed_noise_power_half,
                support_mass,
                shell_indices_half,
                jnp.ones_like(support_mass, dtype=bool),
                norm_unweighted_shell_cutoff,
                shell_count=n_shells,
                image_shape=image_shape,
                current_size=current_size,
                include_unweighted_high_shell=include_unweighted_norm_high_shell,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                unweighted_high_shell_image_power=unweighted_high_shell_image_power,
            )
            noise_img_power = noise_img_power + batch_img_power_shells
            noise_sumw = noise_sumw + jnp.sum(support_mass)

            shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)
            packed_summed_masked_noise = None
            if defer_packed_mstep_reduction:
                if packed_reconstruction_probs is None:
                    raise RuntimeError("packed posterior rows are required for deferred local noise accumulation")
                if debug_noise_dump_dir is not None and debug_noise_dump_targets:
                    summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
                    ctf_probs_for_debug = (
                        ctf_probs
                        if ctf_probs is not None
                        else compute_local_ctf_sums(reconstruction_probs, ctf2_over_nv_recon)
                    )
                    debug_noise_dump_targets = maybe_write_debug_noise_component_dump(
                        experiment_dataset=experiment_dataset,
                        bucket=bucket,
                        support_mass=support_mass,
                        processed_noise_power_half=processed_noise_power_half,
                        proj_for_noise=proj_for_noise,
                        proj_abs2_for_noise=None,
                        summed_masked_noise=summed_masked_noise,
                        ctf_probs=ctf_probs_for_debug,
                        noise_variance_for_noise=noise_variance_for_noise,
                        shell_indices_half=shell_indices_half,
                        shell_indices_noise=shell_indices_noise,
                        n_shells=n_shells,
                        current_size=current_size,
                        debug_iteration=debug_iteration,
                        reconstruction_sample_mask=reconstruction_sample_mask,
                        n_significant_samples=n_significant_samples,
                        dump_dir=debug_noise_dump_dir,
                        pending_targets=debug_noise_dump_targets,
                        requested_current_sizes=debug_noise_dump_current_sizes,
                        requested_iterations=debug_noise_dump_iterations,
                    )
            else:
                summed_masked_noise = compute_local_weighted_sums(reconstruction_probs, shifted_noise_split)
                debug_noise_dump_targets = maybe_write_debug_noise_component_dump(
                    experiment_dataset=experiment_dataset,
                    bucket=bucket,
                    support_mass=support_mass,
                    processed_noise_power_half=processed_noise_power_half,
                    proj_for_noise=proj_for_noise,
                    proj_abs2_for_noise=None,
                    summed_masked_noise=summed_masked_noise,
                    ctf_probs=ctf_probs,
                    noise_variance_for_noise=noise_variance_for_noise,
                    shell_indices_half=shell_indices_half,
                    shell_indices_noise=shell_indices_noise,
                    n_shells=n_shells,
                    current_size=current_size,
                    debug_iteration=debug_iteration,
                    reconstruction_sample_mask=reconstruction_sample_mask,
                    n_significant_samples=n_significant_samples,
                    dump_dir=debug_noise_dump_dir,
                    pending_targets=debug_noise_dump_targets,
                    requested_current_sizes=debug_noise_dump_current_sizes,
                    requested_iterations=debug_noise_dump_iterations,
                )
                packed_summed_masked_noise = _packed_reconstruction_rows(summed_masked_noise, reconstruction_take_indices_jnp, reconstruction_pack_mask_jnp)
            block_noise_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
            block_a2_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
            block_xa_shells = jnp.zeros(n_shells, dtype=precision_policy.score_real_dtype)
            block_norm_residual = jnp.zeros(batch_size, dtype=precision_policy.score_real_dtype)
            packed_noise_chunk_static_kwargs = dict(
                defer_packed_mstep_reduction=defer_packed_mstep_reduction,
                packed_reconstruction_probs=packed_reconstruction_probs,
                shifted_noise_split=shifted_noise_split,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                packed_summed_masked_noise=packed_summed_masked_noise,
                packed_ctf_probs=packed_ctf_probs,
                noise_variance_for_noise=noise_variance_for_noise,
                shell_indices_noise=shell_indices_noise,
                n_shells=n_shells,
                return_noise_split=return_noise_split,
                batch_scale=batch_scale,
                scale_correction_pixel_mask=scale_correction_pixel_mask,
                bucket_group_ids=bucket_group_ids,
            )
            if proj_for_noise is None:
                packed_rotation_count = int(packed_rotations_np.shape[1])
                n_recon_pixels = window_spec.n_recon if window_spec.use_window else int(n_half)
                noise_projection_pixels = int(n_half) if relion_projector_half is not None else int(n_recon_pixels)
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_deferred_noise_projection_chunking:
                    logger.info(
                        "Exact local noise projection chunking: packed_rows=%d chunk_rows=%d n_recon_pixels=%d projection_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        n_recon_pixels,
                        noise_projection_pixels,
                        batch_size,
                    )
                    logged_deferred_noise_projection_chunking = True
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_rotations = packed_rotations_np[:, chunk_start:chunk_stop]
                    chunk_proj_for_noise = _project_packed_noise_rows(
                        mean_for_proj=mean_for_proj,
                        packed_flat_rotations=flatten_bucket_rotations(jnp.asarray(chunk_rotations)),
                        packed_rotation_count=chunk_stop - chunk_start,
                        batch_size=batch_size,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        disc_type=disc_type,
                        projection_kwargs=projection_kwargs,
                        window_spec=window_spec,
                        n_half=n_half,
                        precision_policy=precision_policy,
                        reconstruction_pack_mask_jnp=reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop],
                        relion_projector_half=relion_projector_half,
                        relion_projector_r_max=relion_projector_r_max,
                        projection_padding_factor=projection_padding_factor,
                    )
                    (
                        block_noise_shells,
                        block_a2_shells,
                        block_xa_shells,
                        block_norm_residual,
                        noise_scale_xa,
                        noise_scale_aa,
                    ) = _accumulate_packed_noise_chunk(
                        chunk_proj_for_noise,
                        chunk_start=chunk_start,
                        chunk_stop=chunk_stop,
                        **packed_noise_chunk_static_kwargs,
                        block_noise_shells=block_noise_shells,
                        block_a2_shells=block_a2_shells,
                        block_xa_shells=block_xa_shells,
                        block_norm_residual=block_norm_residual,
                        noise_scale_xa=noise_scale_xa,
                        noise_scale_aa=noise_scale_aa,
                    )
            else:
                packed_rotation_count = int(reconstruction_take_indices_jnp.shape[1])
                noise_projection_pixels = int(proj_for_noise.shape[-1])
                chunk_rows = min(
                    packed_rotation_count,
                    _packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=batch_size),
                )
                if chunk_rows < packed_rotation_count and not logged_cached_noise_projection_chunking:
                    logger.info(
                        "Exact local cached noise projection chunking: packed_rows=%d chunk_rows=%d projection_pixels=%d batch_size=%d",
                        packed_rotation_count,
                        chunk_rows,
                        noise_projection_pixels,
                        batch_size,
                    )
                    logged_cached_noise_projection_chunking = True
                for chunk_start in range(0, packed_rotation_count, chunk_rows):
                    chunk_stop = min(packed_rotation_count, chunk_start + chunk_rows)
                    chunk_take_indices = reconstruction_take_indices_jnp[:, chunk_start:chunk_stop]
                    chunk_pack_mask = reconstruction_pack_mask_jnp[:, chunk_start:chunk_stop]
                    chunk_proj_for_noise = _packed_reconstruction_rows(proj_for_noise, chunk_take_indices, chunk_pack_mask)
                    (
                        block_noise_shells,
                        block_a2_shells,
                        block_xa_shells,
                        block_norm_residual,
                        noise_scale_xa,
                        noise_scale_aa,
                    ) = _accumulate_packed_noise_chunk(
                        chunk_proj_for_noise,
                        chunk_start=chunk_start,
                        chunk_stop=chunk_stop,
                        **packed_noise_chunk_static_kwargs,
                        block_noise_shells=block_noise_shells,
                        block_a2_shells=block_a2_shells,
                        block_xa_shells=block_xa_shells,
                        block_norm_residual=block_norm_residual,
                        noise_scale_xa=noise_scale_xa,
                        noise_scale_aa=noise_scale_aa,
                    )
            if return_profile:
                _block_until_ready(block_noise_shells, block_norm_residual)
            noise_wsum = noise_wsum + block_noise_shells
            if return_noise_split:
                noise_a2 = noise_a2 + block_a2_shells
                noise_xa = noise_xa + block_xa_shells
            noise_norm_correction = noise_norm_correction.at[jnp.asarray(bucket.image_indices, dtype=jnp.int32)].add(
                (batch_img_power_per_image + block_norm_residual).astype(noise_norm_correction.dtype),
            )
            noise_sigma2_offset = noise_sigma2_offset + noise_sumw_offset
            timing.noise_s += time.time() - noise_t0

        postprocess_t0 = time.time()
        significant_sample_count, reconstruction_row_count = _postprocess_local_bucket(
            image_indices=bucket.image_indices,
            local_rotation_ids=bucket.local_rotation_ids,
            local_rotation_mask=bucket.local_rotation_mask,
            local_rotations=bucket.local_rotations,
            local_source_eulers=bucket.local_source_eulers,
            local_rotation_posterior_ids=bucket.local_rotation_posterior_ids,
            translation_grid=local_layout.translation_grid,
            n_trans=n_trans,
            best_argmax=best_argmax,
            batch_norm=batch_norm,
            log_Z=log_Z,
            best_log_score=best_log_score,
            max_posterior=max_posterior,
            probs_sum_t=stats_probs_sum_t_np,
            n_significant_samples=n_significant_samples,
            reconstruction_sample_mask=reconstruction_sample_mask,
            collect_profile_stats=collect_profile_stats,
            reconstruction_row_count=reconstruction_row_count,
            reconstruction_take_indices=reconstruction_take_indices,
            reconstruction_pack_mask=reconstruction_pack_mask_np,
            buffers=postprocess_buffers,
        )
        if collect_profile_stats:
            total_significant_samples += significant_sample_count
            total_reconstruction_rows += int(reconstruction_row_count)
        timing.postprocess_s += time.time() - postprocess_t0

        host_stats_t0 = time.time()
        logger.debug(
            "Exact local bucket: %d images, bucket_rot=%d, total_local_rot=%d",
            batch_size,
            int(bucket.bucket_rotation_count),
            int(np.sum(bucket.actual_rotation_counts)),
        )
        timing.host_stats_s += time.time() - host_stats_t0
        local_progress.mark_bucket_done(bucket)
        if debug_score_dump_force_split and debug_score_dump_bucket_matches:
            cleanup_t0 = time.time()
            shifted_half = None
            shifted_recon_half = None
            shifted_score = None
            shifted_recon = None
            shifted_noise = None
            ctf2_over_nv_score = None
            ctf2_over_nv_recon = None
            projection_block = None
            proj_weighted = None
            proj_for_noise = None
            shifted_score_split = None
            shifted_recon_split = None
            scores = None
            probs = None
            reconstruction_probs = None
            reconstruction_sample_mask = None
            reconstruction_rotation_mask = None
            gc.collect()
            jax.clear_caches()
            timing.host_stats_s += time.time() - cleanup_t0

    if bpref_transaction_queue is not None:
        adjoint_t0 = time.time()
        Ft_y, Ft_ctf, _ = bpref_transaction_queue.flush(Ft_y, Ft_ctf)
        if return_profile:
            _block_until_ready(Ft_y, Ft_ctf)
        timing.adjoint_y_s += time.time() - adjoint_t0
    local_progress.log(force=True, done=True)
    final_accumulator_t0 = time.time()
    if not score_only:
        def _finalize_accumulator(data, weight, *, label):
            final_recon_volume_shape = recon_volume_shape
            if stable_window_active and mstep_relion_x_half:
                data = crop_relion_x_half_accumulator(
                    data,
                    recon_volume_shape,
                    logical_recon_volume_shape,
                )
                weight = crop_relion_x_half_accumulator(
                    weight,
                    recon_volume_shape,
                    logical_recon_volume_shape,
                )
                final_recon_volume_shape = logical_recon_volume_shape
            data, weight = enforce_half_volume_x0(
                data,
                weight,
                final_recon_volume_shape,
                logger=logger,
                label=label,
                force_host=host_accumulator_finalize,
            )
            if return_half_volume_accumulators:
                return data, weight
            if mstep_relion_x_half:
                return relion_x_half_accumulators_to_public_layout(
                    data,
                    weight,
                    final_recon_volume_shape,
                    force_host=host_accumulator_finalize,
                )
            return half_volume_accumulators_to_full(
                data,
                weight,
                final_recon_volume_shape,
            )

        if reconstruction_group_ids_np is None:
            Ft_y, Ft_ctf = _finalize_accumulator(
                Ft_y,
                Ft_ctf,
                label="Exact local",
            )
        else:
            grouped = [
                _finalize_accumulator(
                    Ft_y[group_index],
                    Ft_ctf[group_index],
                    label=f"Exact local reconstruction group {group_index}",
                )
                for group_index in range(resolved_reconstruction_group_count)
            ]
            stack = np.stack if host_accumulator_finalize else jnp.stack
            Ft_y = stack([value[0] for value in grouped], axis=0)
            Ft_ctf = stack([value[1] for value in grouped], axis=0)
        if return_half_volume_accumulators:
            logger.info("Exact local M-step: keeping native half-volume accumulators for downstream reconstruction")

        if return_profile:
            _block_until_ready(Ft_y, Ft_ctf)
    timing.final_accumulator_s += time.time() - final_accumulator_t0

    stats_finalize_t0 = time.time()
    relion_stats = make_relion_stats(
        log_evidence_per_image=log_evidence_per_image,
        best_log_score_per_image=best_log_score_per_image,
        max_posterior_per_image=max_posterior_per_image,
        rotation_posterior_sums=rotation_posterior_sums,
        host_arrays=host_stats_publication,
    )
    noise_stats = None
    if accumulate_noise:
        transfer_t0 = time.time()
        noise_sigma2_offset_value = float(np.asarray(noise_sigma2_offset, dtype=np.float64))
        noise_sumw_value = float(np.asarray(noise_sumw, dtype=np.float64))
        transfer_profile["final_noise_to_host_s"] += time.time() - transfer_t0
        if host_stats_publication:
            # Publish the physical carry before taking the logical host prefix;
            # do not compile a fresh device slice for every subset length.
            noise_norm_correction = np.asarray(noise_norm_correction)
        noise_stats = make_noise_stats(
            wsum_sigma2_noise=noise_wsum,
            wsum_img_power=noise_img_power,
            wsum_sigma2_offset=noise_sigma2_offset_value,
            sumw=noise_sumw_value,
            wsum_noise_a2=(noise_a2 if return_noise_split else None),
            wsum_noise_xa=(noise_xa if return_noise_split else None),
            wsum_norm_correction=(
                noise_norm_correction[:n_images]
                if noise_norm_capacity_enabled
                else noise_norm_correction
            ),
            wsum_scale_correction_xa=noise_scale_xa,
            wsum_scale_correction_aa=noise_scale_aa,
            host_arrays=host_stats_publication,
        )
    timing.stats_finalize_s += time.time() - stats_finalize_t0

    if debug_score_dump_filter_matches and debug_score_dump_targets and debug_score_dump_iterations is None:
        logger.warning(
            "Requested local score dump indices were not observed in this dataset view: %s",
            sorted(debug_score_dump_targets),
        )
    if (
        debug_fused_posterior_dump_filter_matches
        and debug_fused_posterior_dump_targets
        and debug_fused_posterior_dump_iterations is None
    ):
        logger.warning(
            "Requested fused posterior dump indices were not observed in this dataset view: %s",
            sorted(debug_fused_posterior_dump_targets),
        )

    if reconstruct_significant_only:
        logger.info(
            "Exact local significant-support summary: chunks=%d big_jit_buckets=%d "
            "sparse_big_jit_buckets=%d reconstruction_rows=%d padded_rows=%d "
            "significant_samples=%d mean_reconstruction_rows_per_image=%.2f "
            "mean_significant_samples_per_image=%.2f",
            n_chunks,
            big_jit_bucket_count,
            sparse_big_jit_bucket_count,
            total_reconstruction_rows,
            total_padded_rotations,
            total_significant_samples,
            0.0 if n_images == 0 else total_reconstruction_rows / n_images,
            0.0 if n_images == 0 else total_significant_samples / n_images,
        )

    profile_summary = None
    if return_profile:
        _block_until_ready(Ft_y, Ft_ctf)
        total_wall_time = time.time() - overall_t0
        profile_summary = {
            "big_jit_bucket_count": np.int32(big_jit_bucket_count),
            "sparse_big_jit_bucket_count": np.int32(sparse_big_jit_bucket_count),
            "big_jit_debug_bucket_count": np.int32(big_jit_debug_bucket_count),
            "score_only": np.asarray(score_only),
            "fused_score_mstep_enabled": np.asarray(fused_score_mstep_enabled),
            "defer_local_noise_projection": np.asarray(defer_local_noise_projection),
            "bucket_build_time_s": np.float64(timing.bucket_build_s),
            "raw_cache_build_time_s": np.float64(timing.raw_cache_build_s),
            "raw_cache_enabled": np.asarray(raw_cache_enabled),
            "processed_half_cache_enabled": np.asarray(processed_half_cache_enabled),
            "relion_projection_cache_enabled": np.asarray(relion_projection_cache_groups_built > 0),
            "relion_projection_cache_groups": np.int64(len(projection_cache_plan.groups)),
            "relion_projection_cache_groups_built": np.int64(relion_projection_cache_groups_built),
            "relion_projection_cache_rows": np.int64(relion_projection_cache_max_rows),
            "relion_projection_cache_capacity_rows": np.int64(projection_cache_plan.capacity_rows),
            "relion_projection_cache_id_map_rows": np.int64(relion_projection_cache_id_map_rows),
            "relion_projection_cache_pixels": np.int64(projection_cache_plan.projection_pixels),
            "relion_projection_cache_estimated_gb": np.float64(relion_projection_cache_max_estimated_gb),
            "relion_projection_cache_build_s": np.float64(relion_projection_cache_total_build_s),
            "relion_projection_cache_cap_gb": np.float64(projection_cache_plan.budget_gb),
            "batch_fetch_time_s": np.float64(timing.batch_fetch_s),
            "preprocess_time_s": np.float64(timing.preprocess_s),
            **_prefixed_timer_profile("preprocess_", preprocess_profile),
            **_prefixed_timer_profile("transfer_", transfer_profile),
            "transfer_total_to_host_s": np.float64(sum(transfer_profile.values())),
            **_local_timing_profile(timing),
            "em_time_s": np.float64(total_wall_time),
            "accounted_em_time_s": np.float64(timing.accounted_s()),
            "unattributed_em_time_s": np.float64(max(total_wall_time - timing.accounted_s(), 0.0)),
            "n_chunks": np.int32(n_chunks),
            "projection_mode": np.asarray(projection_mode),
            "n_projection_windowed": np.int32(window_spec.n_projection),
            "big_jit_projection_pixels": np.int32(big_jit_projection_pixel_count),
            "chunk_sizes": np.asarray(chunk_sizes, dtype=np.int32),
            "chunk_padded_image_counts": np.asarray(
                chunk_padded_image_counts,
                dtype=np.int32,
            ),
            "chunk_planned_padded_image_counts": np.asarray(
                chunk_planned_padded_image_counts,
                dtype=np.int32,
            ),
            "chunk_local_rotations": np.asarray(chunk_local_rotations, dtype=np.int32),
            "chunk_padded_rotations": np.asarray(chunk_padded_rotations, dtype=np.int32),
            "flat_local_rows_enabled": np.asarray(flat_local_rows_enabled),
            "stable_flat_row_capacity_enabled": np.asarray(
                stable_flat_row_capacity_enabled
            ),
            "packed_local_projection_enabled": np.asarray(
                packed_local_projection_enabled
            ),
            "fused_pair_fine_score_enabled": np.asarray(
                fused_pair_fine_score_enabled
            ),
            "fused_pair_fine_score_default_enabled": np.asarray(False),
            "fused_pair_fine_uses_shared_compact_order": np.asarray(
                fused_pair_fine_score_enabled
            ),
            "fused_pair_fine_avoids_pair_pixel_gathers": np.asarray(
                fused_pair_fine_score_enabled
            ),
            "fused_pair_fine_restores_dense_posterior_order": np.asarray(
                fused_pair_fine_score_enabled
            ),
            "defer_packed_vdam_enabled": np.asarray(defer_packed_vdam_enabled),
            "stable_fourier_window_shapes": np.asarray(stable_window_active),
            "stable_fourier_window_quantum": np.int32(resolved_window_quantum),
            "logical_current_size": np.int32(logical_current_size),
            "physical_current_size": np.int32(physical_current_size),
            "logical_reconstruction_pixels": np.int32(
                stable_window_plan.logical_reconstruction_pixels
            ),
            "physical_reconstruction_pixels": np.int32(window_spec.n_recon),
            "packed_final_noise_enabled": np.asarray(packed_final_noise_enabled),
            "packed_vdam_reuses_flat_score_projection": np.asarray(
                defer_packed_vdam_enabled and accumulate_noise
            ),
            "packed_vdam_avoids_dense_noise_rows": np.asarray(
                packed_final_noise_enabled and accumulate_noise
            ),
            "packed_final_noise_preserves_dense_scalar_order": np.asarray(
                packed_final_noise_enabled and accumulate_noise
            ),
            "chunk_flat_score_rows": np.asarray(chunk_flat_score_rows, dtype=np.int32),
            "chunk_fused_pair_capacities": np.asarray(
                chunk_fused_pair_capacities,
                dtype=np.int32,
            ),
            "chunk_fused_pair_counts": np.asarray(
                chunk_fused_pair_counts,
                dtype=np.int64,
            ),
            "chunk_fused_pair_dense_capacities": np.asarray(
                chunk_fused_pair_dense_capacities,
                dtype=np.int64,
            ),
            "chunk_planned_padded_rotations": np.asarray(
                chunk_planned_padded_rotations,
                dtype=np.int32,
            ),
            "chunk_unique_rotations": np.asarray(chunk_unique_rotations, dtype=np.int32),
            "chunk_nonzero_posterior_rows": np.asarray(chunk_nonzero_posterior_rows, dtype=np.int32),
            "chunk_reconstruction_rows": np.asarray(chunk_reconstruction_rows, dtype=np.int32),
            "chunk_significant_samples": np.asarray(chunk_significant_samples, dtype=np.int32),
            "sum_union_rows": np.int64(total_local_rotations),
            "sum_padded_rows": np.int64(total_padded_rotations),
            "sum_flat_score_rows": np.int64(total_flat_score_rows),
            "sum_fused_pair_candidates": np.int64(total_fused_pair_candidates),
            "sum_fused_pair_capacity": np.int64(total_fused_pair_capacity),
            "sum_fused_pair_dense_capacity": np.int64(
                total_fused_pair_dense_capacity
            ),
            "fused_pair_valid_fraction_of_dense": np.float64(
                0.0
                if total_fused_pair_dense_capacity == 0
                else total_fused_pair_candidates / total_fused_pair_dense_capacity
            ),
            "fused_pair_padded_fraction_of_dense": np.float64(
                0.0
                if total_fused_pair_dense_capacity == 0
                else total_fused_pair_capacity / total_fused_pair_dense_capacity
            ),
            "sum_planned_padded_rows": np.int64(total_planned_padded_rotations),
            "sum_nonzero_posterior_rows": np.int64(np.sum(chunk_nonzero_posterior_rows)),
            "sum_reconstruction_rows": np.int64(total_reconstruction_rows),
            "sum_packed_final_noise_rows": np.int64(
                total_packed_final_noise_rows
            ),
            "sum_significant_samples": np.int64(total_significant_samples),
            "unique_global_rotations": np.int64(np.count_nonzero(seen_global_rotations)),
            "unique_nonzero_global_rotations": np.int64(np.count_nonzero(seen_nonzero_global_rotations)),
            "unique_reconstruction_global_rotations": np.int64(np.count_nonzero(seen_reconstruction_global_rotations)),
            "duplicate_rotation_factor": np.float64(
                0.0
                if not np.any(seen_global_rotations)
                else total_local_rotations / np.count_nonzero(seen_global_rotations)
            ),
            "reconstruction_duplicate_rotation_factor": np.float64(
                0.0
                if not np.any(seen_reconstruction_global_rotations)
                else total_reconstruction_rows / np.count_nonzero(seen_reconstruction_global_rotations)
            ),
            "local_total_hypotheses": np.int64(local_total_hypotheses),
            "local_mean_rotations_per_image": np.float64(0.0 if n_images == 0 else total_local_rotations / n_images),
            "local_mean_reconstruction_rows_per_image": np.float64(
                0.0 if n_images == 0 else total_reconstruction_rows / n_images
            ),
            "local_mean_significant_samples_per_image": np.float64(
                0.0 if n_images == 0 else total_significant_samples / n_images
            ),
            "local_num_buckets": np.int32(n_chunks),
            "max_hypotheses_per_microbatch": np.int64(max_hypotheses_per_microbatch),
            "sparse_adjoint_target_rows": np.int64(sparse_adjoint_target_rows),
            "sparse_adjoint_chunk_count": np.int64(sparse_adjoint_chunk_count),
            "local_pad_fraction": np.float64(
                0.0 if total_padded_rotations == 0 else 1.0 - total_local_rotations / total_padded_rotations
            ),
            "flat_score_row_reduction_fraction": np.float64(
                0.0
                if total_padded_rotations == 0
                else 1.0 - total_flat_score_rows / total_padded_rotations
            ),
            "n_windowed": np.int32(n_windowed),
        }
        if reconstruction_probability_values_by_image is not None:
            profile_summary["reconstruction_probability_values_by_image"] = tuple(
                np.concatenate(values).astype(precision_policy.score_real_dtype, copy=False)
                if values
                else np.zeros(0, dtype=precision_policy.score_real_dtype)
                for values in reconstruction_probability_values_by_image
            )
        if reconstruction_sample_indices_by_image is not None:
            profile_summary["reconstruction_sample_indices_by_image"] = tuple(reconstruction_sample_indices_by_image)
    return LocalEMResult(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        hard_assignments=hard_assignment,
        stats=relion_stats,
        best_pose_rotations=best_pose_rotations if return_best_pose_details else None,
        best_pose_translations=best_pose_translations if return_best_pose_details else None,
        best_pose_rotation_ids=best_pose_rotation_ids if return_best_pose_details else None,
        best_pose_eulers_deg=best_pose_eulers_deg,
        noise_stats=noise_stats if accumulate_noise else None,
        significant_counts=significant_counts if return_significant_counts else None,
        profile=profile_summary,
    )
