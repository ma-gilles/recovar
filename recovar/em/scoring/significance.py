"""Coarse significance pruning for adaptive class and pose searches.

The shared scorer serves K=1, multiclass refinement and initial-model searches.
It selects per-image significant samples from one class/rotation/translation
posterior, with optional diagnostic capture of the coarse scoring boundary.
"""

import logging
import operator
import os
from enum import Enum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics.coarse_gaussian_diagnostics import (
    _COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV,
    CoarseGaussianGemmDiagnosticScope,
    _coarse_gaussian_gemm_diagnostic_request,
    _coarse_gaussian_gemm_streaming_diagnostic_request,
    _coarse_runtime_prefix_dump_request,
    _maybe_dump_coarse_runtime_prefix_operands,
    _maybe_dump_k_class_significance_batch,
    _maybe_dump_tree_rescore_batch,
    _resolve_coarse_gaussian_gemm_diagnostic_scope,
    _seal_coarse_gaussian_gemm_diagnostic_scope,
    _seal_coarse_gaussian_gemm_streaming_scope,
    _significance_debug_dump_matches,
    _write_coarse_gaussian_gemm_diagnostic,
)
from recovar.em.diagnostics.coarse_score_diagnostics import (
    _build_coarse_significance_support_audit,
    _validate_coarse_selector_audit,
)
from recovar.em.helpers.batch_fetch import original_image_indices
from recovar.em.helpers.env_flags import parse_env_int_set
from recovar.em.helpers.projection import compute_projections_block
from recovar.em.relion.relion_coarse_operands import (
    _K1_RELION_EXACT_COMPACT_PREPROCESS_ENV,
    _assemble_relion_exact_coarse_gaussian_operands,
    _infer_relion_coarse_healpix_order,
    _k1_relion_exact_coarse_assembly_profile_enabled,
    _k1_relion_exact_coarse_operands_enabled,
    _k1_relion_exact_coarse_skip_generic_operands_enabled,
    _k1_relion_exact_compact_preprocess_enabled,
    _k1_relion_f32_coarse_support_enabled,
    _process_relion_exact_coarse_half_image,
    _relion_acc_double_floorf_quirk_enabled,
    _relion_cc_inverse_power_from_processed,
    _relion_coarse_gaussian_square_operands,
    _relion_coarse_gaussian_square_operands_sincosf,
    _repeat_pad_batch_axis,
    _resolve_k1_relion_exact_coarse_skip_generic_operands,
    _resolve_k1_relion_exact_compact_preprocess,
    _select_relion_coarse_rescore_winner_slots,
)
from recovar.em.scoring.coarse_gaussian_gemm import (
    _COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV,
    _COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION_ENV,
    _COARSE_GAUSSIAN_GEMM_HYBRID_ENV,
    _COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV,
    _COARSE_GAUSSIAN_GEMM_MACRO_ENV,
    _K1_RELION_EXACT_COARSE_OPERANDS_ENV,
    _K1_RELION_F32_COARSE_SUPPORT_ENV,
    _build_coarse_gaussian_gemm_projection_cache,
    _coarse_gaussian_gemm_compact_posterior_enabled,
    _coarse_gaussian_gemm_device_transaction_enabled,
    _coarse_gaussian_gemm_hybrid_block_capacity,
    _coarse_gaussian_gemm_hybrid_enabled,
    _coarse_gaussian_gemm_hybrid_image_batch_size_request,
    _coarse_gaussian_gemm_macro_enabled,
    _coarse_gaussian_gemm_projected_transient_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_enabled,
    _coarse_gaussian_gemm_projection_cache_stats,
    _coarse_gaussian_gemm_real_cross_enabled,
    _coarse_gaussian_gemm_resources,
    _compute_coarse_gaussian_gemm_hybrid_batch,
    _plan_coarse_gaussian_gemm_projection_cache,
    _project_coarse_gaussian_gemm_projection_cache_block_once,
    _resolve_coarse_gaussian_gemm_hybrid_image_batch_size,
    _score_relion_coarse_gaussian_gemm_macro,
    _select_coarse_gaussian_gemm_score_representation,
    _validate_coarse_gaussian_gemm_compact_posterior_request,
    _validate_coarse_gaussian_gemm_hybrid_request,
    _validate_coarse_gaussian_gemm_projection_cache_request,
)
from recovar.em.scoring.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    SOURCE_ROTATION_BLOCK_SIZE,
    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids,
    plan_coarse_gemm_certificate_topology,
)
from recovar.em.scoring.coarse_gemm_streaming import (
    coarse_gemm_streaming_dual_state_bytes,
    coarse_gemm_streaming_state_bytes,
    initialize_coarse_gemm_streaming_state,
    update_coarse_gemm_streaming_state,
    write_coarse_gemm_streaming_summary,
)
from recovar.em.scoring.scoring import _e_step_block_scores, _e_step_block_scores_windowed, _update_logsumexp
from recovar.em.scoring.significant_samples import compact_significant_sample_indices_from_mask
from recovar.utils.nvtx_shim import nvtx

_SIGNIFICANCE_SCORE_CACHE_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE"
_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE_MAX_GB"
_SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB = 2.0
_SIGNIFICANCE_FUSED_PASS1_ENV = "RECOVAR_PASS1_FUSED"
_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV = "RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"
_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV = (
    "RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN"
)
_K1_COARSE_GAUSSIAN_FFI_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_FFI"
_K1_COARSE_GAUSSIAN_SINCOSF_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF"
_K1_COARSE_FUSED_PROJECTOR_ENV = "RECOVAR_K1_COARSE_FUSED_PROJECTOR"
_RELION_COARSE_CANONICAL_REDUCTION_ENV = (
    "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION"
)
_K1_COARSE_SINGLE_LANE_CANONICAL_ENV = (
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"
)
_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV = (
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"
)
_K1_COARSE_PREHALF_WEIGHT_ENV = "RECOVAR_K1_COARSE_PREHALF_WEIGHT"
_K1_COARSE_MULTISTREAM_WORKERS_ENV = "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS"
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
)
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"
)
_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV = (
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"
)
_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE"
)
NVTX_DOMAIN_EM = "recovar_em"
logger = logging.getLogger("recovar.em.dense_single_volume.helpers.significance")


def _pad_significance_preprocess_inputs(
    batch_data,
    ctf_params,
    integer_pre_shifts,
    batch_corr,
    batch_scale,
    relion_preprocess_kwargs,
    *,
    target_size: int,
):
    """Give a tail significance batch the same compiled image shape as full batches."""

    actual_size = int(np.asarray(batch_data).shape[0])
    target_size = max(actual_size, int(target_size))
    if actual_size == target_size:
        return (
            batch_data,
            ctf_params,
            integer_pre_shifts,
            batch_corr,
            batch_scale,
            relion_preprocess_kwargs,
        )
    padded_kwargs = None
    if relion_preprocess_kwargs is not None:
        padded_kwargs = {
            key: jnp.asarray(_repeat_pad_batch_axis(value, target_size))
            for key, value in relion_preprocess_kwargs.items()
        }
    return (
        _repeat_pad_batch_axis(batch_data, target_size),
        _repeat_pad_batch_axis(ctf_params, target_size),
        (
            None
            if integer_pre_shifts is None
            else _repeat_pad_batch_axis(integer_pre_shifts, target_size)
        ),
        None if batch_corr is None else _repeat_pad_batch_axis(batch_corr, target_size),
        _repeat_pad_batch_axis(batch_scale, target_size),
        padded_kwargs,
    )


def _compact_projection_window_positions(compact_indices, window_indices) -> np.ndarray:
    """Map full-image Fourier indices to positions in a compact projection."""

    compact = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    window = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    if np.unique(compact).size != compact.size:
        raise ValueError("compact projection indices must be unique")
    position_by_index = {int(index): position for position, index in enumerate(compact)}
    missing = [int(index) for index in window if int(index) not in position_by_index]
    if missing:
        raise ValueError(
            "projection window contains indices absent from the compact projection: "
            f"{missing[:8]}"
        )
    return np.asarray([position_by_index[int(index)] for index in window], dtype=np.int32)


class CoarseGaussianSquareLayout(NamedTuple):
    """Logical RELION square issue stream inside a stable physical capacity."""

    logical_current_size: int
    physical_current_size: int
    logical_square_count: int
    physical_square_count: int
    score_indices_np: np.ndarray
    score_active_mask_np: np.ndarray
    logical_projector_mask_np: np.ndarray
    full_to_compact_np: np.ndarray


def _plan_coarse_gaussian_square_layout(
    image_shape,
    logical_current_size: int,
    active_score_indices,
    *,
    stable_fourier_window_shapes: bool,
) -> CoarseGaussianSquareLayout:
    """Plan coarse storage without changing RELION's logical pixel traversal.

    The physical compact table is ordered as the complete logical square
    followed by physical-only capacity rows.  The direct CUDA lookup retains
    the old logical full-pixel permutation as its prefix; its tail visits only
    zero-weight capacity rows.  Consequently exact rescoring keeps every
    logical lane assignment and appends only fused multiply-adds by zero.
    """

    from recovar.em.helpers.fourier_window import (
        make_fourier_window_indices_np,
        make_frequency_coords_half_np,
        stable_fourier_window_current_size,
        stable_fourier_window_quantum,
    )
    from recovar.em.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_full_to_compact_lookup

    image_shape = tuple(int(value) for value in image_shape)
    logical_current_size = int(logical_current_size)
    physical_current_size = (
        stable_fourier_window_current_size(
            logical_current_size,
            image_shape[0],
            quantum=stable_fourier_window_quantum(),
        )
        if stable_fourier_window_shapes
        else logical_current_size
    )
    logical_indices, logical_count = make_fourier_window_indices_np(
        image_shape,
        logical_current_size,
        square=True,
        include_dc=True,
    )
    physical_indices, physical_count = make_fourier_window_indices_np(
        image_shape,
        physical_current_size,
        square=True,
        include_dc=True,
    )
    logical_indices = np.asarray(logical_indices, dtype=np.int32)
    physical_indices = np.asarray(physical_indices, dtype=np.int32)
    logical_count = int(logical_count)
    physical_count = int(physical_count)
    expected_logical_count = logical_current_size * (logical_current_size // 2 + 1)
    expected_physical_count = physical_current_size * (physical_current_size // 2 + 1)
    if logical_count != expected_logical_count or physical_count != expected_physical_count:
        raise ValueError(
            "RELION coarse Gaussian square crop has an unexpected size: "
            f"logical={logical_count}/{expected_logical_count}, "
            f"physical={physical_count}/{expected_physical_count}"
        )
    if np.setdiff1d(logical_indices, physical_indices, assume_unique=True).size:
        raise ValueError("stable coarse physical square does not contain logical support")

    physical_tail = np.setdiff1d(
        physical_indices,
        logical_indices,
        assume_unique=True,
    ).astype(np.int32, copy=False)
    score_indices_np = np.concatenate((logical_indices, physical_tail)).astype(
        np.int32,
        copy=False,
    )
    if score_indices_np.size != physical_count or np.unique(score_indices_np).size != physical_count:
        raise ValueError("stable coarse compact score rows are not a unique physical square")

    logical_lookup = np.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            image_shape,
            logical_current_size,
            logical_indices,
        ),
        dtype=np.int32,
    )
    full_to_compact_np = np.concatenate(
        (
            logical_lookup,
            np.arange(logical_count, physical_count, dtype=np.int32),
        )
    )
    active_score_indices = np.asarray(active_score_indices, dtype=np.int32).reshape(-1)
    score_active_mask_np = np.isin(score_indices_np, active_score_indices)
    if physical_count > logical_count:
        score_active_mask_np[logical_count:] = False
    compact_coords = np.rint(
        make_frequency_coords_half_np(image_shape)[score_indices_np],
    ).astype(np.int64)
    logical_projector_mask_np = np.sum(compact_coords**2, axis=1) <= (
        logical_current_size // 2
    ) ** 2
    if physical_count > logical_count:
        logical_projector_mask_np[logical_count:] = False

    return CoarseGaussianSquareLayout(
        logical_current_size=logical_current_size,
        physical_current_size=physical_current_size,
        logical_square_count=logical_count,
        physical_square_count=physical_count,
        score_indices_np=score_indices_np,
        score_active_mask_np=np.asarray(score_active_mask_np, dtype=np.bool_),
        logical_projector_mask_np=np.asarray(
            logical_projector_mask_np,
            dtype=np.bool_,
        ),
        full_to_compact_np=full_to_compact_np,
    )


def _coarse_gaussian_fused_logical_lookup(
    full_to_compact,
    square_layout: CoarseGaussianSquareLayout,
    *,
    current_size: int,
):
    """Return the logical fused-ABI prefix from a stable physical lookup."""

    current_size = operator.index(current_size)
    logical_count = current_size * (current_size // 2 + 1)
    if (
        int(square_layout.logical_current_size) != current_size
        or int(square_layout.logical_square_count) != logical_count
    ):
        raise ValueError(
            "fused coarse logical lookup does not match current_size: "
            f"layout={square_layout.logical_current_size}/"
            f"{square_layout.logical_square_count}, expected={current_size}/"
            f"{logical_count}",
        )
    lookup = jnp.asarray(full_to_compact)
    if lookup.ndim != 1 or lookup.dtype != jnp.int32:
        raise TypeError("fused coarse full_to_compact lookup must be rank-1 int32")
    if int(lookup.shape[0]) < logical_count:
        raise ValueError(
            "fused coarse full_to_compact lookup is shorter than its logical prefix",
        )
    return lookup[:logical_count]


def _coarse_rotated_radius_enabled() -> bool:
    """Opt-in qualification of RELION's rotated image-radius clipping."""
    token = os.environ.get("RECOVAR_K1_COARSE_ROTATED_RADIUS", "0")
    if token not in {"0", "1"}:
        raise ValueError("RECOVAR_K1_COARSE_ROTATED_RADIUS must be 0 or 1")
    return token == "1"


def _k1_coarse_gaussian_ffi_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION coarse Gaussian FFI is active."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_FFI_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {_K1_COARSE_GAUSSIAN_FFI_ENV}={token!r}")


def _coarse_gaussian_ffi_default(
    relion_coarse_gaussian_default: bool,
    *,
    use_relion_projector: bool,
    use_float64_scoring: bool,
    coarse_texture_interp: bool,
) -> bool:
    """Whether the fresh-InitialModel coarse Gaussian FFI default applies.

    The FFI scores from the supplied RELION projector with texture
    interpolation (or float64 operands), so the guarded default is active only
    when those operands exist; a dense pass without a supplied projector keeps
    the JAX coarse path. An explicit environment request still fails closed at
    the operand check.
    """

    return bool(
        relion_coarse_gaussian_default
        and use_relion_projector
        and (use_float64_scoring or coarse_texture_interp)
    )


def _k1_coarse_gaussian_sincosf_enabled(*, default: bool = False) -> bool:
    """Return whether exact RELION coarse score translation is active."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_SINCOSF_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_GAUSSIAN_SINCOSF_ENV}={token!r}",
    )


def _k1_coarse_gaussian_native_texture_enabled(*, default: bool = False) -> bool:
    """Return whether projection and coarse scoring run in one RELION kernel."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV}={token!r}",
    )


def _k1_coarse_fused_projector_enabled(*, default: bool = False) -> bool:
    """Return whether coarse projection and diff2 use RELION's fused topology."""

    token = os.environ.get(
        _K1_COARSE_FUSED_PROJECTOR_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_FUSED_PROJECTOR_ENV}={token!r}",
    )


def _relion_coarse_canonical_reduction_enabled(*, default: bool = False) -> bool:
    """Whether the shared fused coarse scorer reduces lanes in index order."""

    token = os.environ.get(
        _RELION_COARSE_CANONICAL_REDUCTION_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_RELION_COARSE_CANONICAL_REDUCTION_ENV}={token!r}",
    )


def _k1_coarse_single_lane_canonical_enabled(*, default: bool = False) -> bool:
    """Whether 65--128 translations use the single-lane CUDA specialization."""

    token = os.environ.get(
        _K1_COARSE_SINGLE_LANE_CANONICAL_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}={token!r}",
    )


def _k1_coarse_single_lane_canonical_selected(
    *,
    requested: bool,
    score_mode: str,
    translation_count: int,
) -> bool:
    """Select the specialization only where one thread owns a translation."""

    return bool(
        requested
        and score_mode == "gaussian"
        and 65 <= int(translation_count) <= 128
    )


def _k1_coarse_native_atomic_reduction_enabled(*, default: bool = False) -> bool:
    """Whether the fused coarse scorer uses RELION's native atomic lane adds."""

    token = os.environ.get(
        _K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}={token!r}",
    )


def _k1_coarse_native_atomic_reduction_selected(
    *,
    requested: bool,
    score_mode: str,
    translation_count: int,
) -> bool:
    """Select only the T=29 Gaussian topology covered by the live H100 gate."""

    return bool(
        requested
        and score_mode == "gaussian"
        and int(translation_count) == 29
    )


def _k1_coarse_prehalf_weight_enabled(*, default: bool = False) -> bool:
    """Whether coarse pixel weights are halved once before rotation FMAs.

    RELION stores ``corr_img / 2`` in shared memory once per pixel.  The
    historical RECOVAR kernel instead multiplied every rotation contribution
    by ``0.5``.  The specialization is mathematically equivalent, but changes
    float32 operation order and is therefore opt-in until its trajectory gate
    is complete.
    """

    token = os.environ.get(
        _K1_COARSE_PREHALF_WEIGHT_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_PREHALF_WEIGHT_ENV}={token!r}",
    )


def _k1_coarse_multistream_worker_count(*, default: int = 0) -> int:
    """Return the default-off RELION coarse particle-stream count.

    The first performance gate is deliberately restricted to the proven
    GUI-default ``--j 8`` ownership topology.  Zero keeps the accepted
    single-stream batch grid unchanged.
    """

    token = os.environ.get(
        _K1_COARSE_MULTISTREAM_WORKERS_ENV,
        str(int(default)),
    ).strip()
    try:
        count = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV} must be 0 or 8, got {token!r}",
        ) from error
    if count not in {0, 8}:
        raise ValueError(
            f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV} must be 0 or 8, got {token!r}",
        )
    return count


def _coarse_max_posterior_physical_batch_enabled() -> bool:
    """Keep the physical row count through coarse Pmax publication."""
    name = "RECOVAR_COARSE_MAX_POSTERIOR_PHYSICAL_BATCH"
    token = os.environ.get(name, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"Unsupported {name}={token!r}")
    return token == "1"


def _coarse_max_posterior_for_host(
    batch_weights, actual_batch_size, *, physical_batch=False,
):
    """Publish active row maxima without specializing on fringe batch sizes.

    Rows are independent. Reducing the physical table before slicing the
    compact host vector keeps padded rows out of the published statistics.
    """
    if physical_batch:
        return np.asarray(jnp.max(batch_weights, axis=1), dtype=np.float32)[
            :actual_batch_size
        ]
    return np.asarray(
        jnp.max(batch_weights[:actual_batch_size], axis=1), dtype=np.float32,
    )


def _coarse_significance_support_audit_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve exact, diagnostic-only coarse-support hashing."""

    token = os.environ.get(
        _COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}={token!r}",
    )


def _coarse_significance_support_audit_ids_enabled() -> bool:
    """Whether a support audit also retains its exact selected IDs."""

    token = os.environ.get(
        _COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV,
        "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV}={token!r}",
    )


class _CoarseGaussianScoreBackend(str, Enum):
    """One resolved coarse Gaussian score/reduction implementation."""

    RECTANGULAR = "rectangular"
    FUSED = "fused"
    FUSED_CANONICAL = "fused_canonical"
    FUSED_NATIVE_ATOMIC = "fused_native_atomic"
    FUSED_SINGLE_LANE = "fused_single_lane"
    FUSED_MULTISTREAM = "fused_multistream"
    NATIVE_TEXTURE = "native_texture"
    GEMM_MACRO = "gemm_macro"


_COARSE_GAUSSIAN_FUSED_SCORE_BACKENDS = frozenset(
    {
        _CoarseGaussianScoreBackend.FUSED,
        _CoarseGaussianScoreBackend.FUSED_CANONICAL,
        _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC,
        _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE,
        _CoarseGaussianScoreBackend.FUSED_MULTISTREAM,
    }
)


def _resolve_coarse_gaussian_score_backend(
    *,
    gemm_macro_requested: bool,
    gemm_hybrid_requested: bool,
    score_mode: str,
    fused_projector_requested: bool,
    fused_projector_enabled: bool,
    canonical_reduction_requested: bool,
    canonical_reduction_enabled: bool,
    native_atomic_reduction_requested: bool,
    native_atomic_reduction_enabled: bool,
    single_lane_canonical_requested: bool,
    single_lane_canonical_enabled: bool,
    multistream_requested: bool,
    multistream_enabled: bool,
    native_texture_requested: bool,
    native_texture_enabled: bool,
) -> _CoarseGaussianScoreBackend:
    """Resolve the primary score backend and reject ambiguous selectors.

    The exact-operand, CUDA-sincosf, and texture-projection flags are operands
    or prerequisites rather than competing score/reduction selectors.  All
    selectors that the expanded GEMM branch would otherwise silently override
    are represented here.  The mature fused selector family is permitted only
    when the certified hybrid is also requested, where it is a secondary full-
    dense fallback rather than the primary backend.
    """

    if gemm_macro_requested:
        if score_mode != "gaussian":
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 requires "
                "score_mode='gaussian'",
            )
        fused_fallback_selectors = (
            (_K1_COARSE_FUSED_PROJECTOR_ENV, "1", fused_projector_requested),
            (_RELION_COARSE_CANONICAL_REDUCTION_ENV, "1", canonical_reduction_requested),
            (_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV, "1", native_atomic_reduction_requested),
            (_K1_COARSE_SINGLE_LANE_CANONICAL_ENV, "1", single_lane_canonical_requested),
            (_K1_COARSE_MULTISTREAM_WORKERS_ENV, "8", multistream_requested),
        )
        conflicts = [
            f"{name}={requested_value}"
            for name, requested_value, selected in (
                (_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV, "1", native_texture_requested),
                *(() if gemm_hybrid_requested else fused_fallback_selectors),
            )
            if selected
        ]
        if conflicts:
            rendered = ", ".join(conflicts)
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 conflicts with {rendered}",
            )
        return _CoarseGaussianScoreBackend.GEMM_MACRO
    if native_texture_enabled:
        return _CoarseGaussianScoreBackend.NATIVE_TEXTURE
    if multistream_enabled:
        return _CoarseGaussianScoreBackend.FUSED_MULTISTREAM
    if single_lane_canonical_enabled:
        return _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE
    if native_atomic_reduction_enabled:
        return _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC
    if canonical_reduction_enabled:
        return _CoarseGaussianScoreBackend.FUSED_CANONICAL
    if fused_projector_enabled:
        return _CoarseGaussianScoreBackend.FUSED
    return _CoarseGaussianScoreBackend.RECTANGULAR


def _k1_coarse_fused_projector_supports_padding(padding_factor: int) -> bool:
    """Whether the fused CUDA projector implements this RELION padding."""

    # The fused kernel stages the compact pad-1 Projector texture and samples
    # unscaled Fourier coordinates.  The general texture-projector path below
    # infers and applies larger padding factors from the projector shape.
    return int(padding_factor) == 1


def _capture_offset_free_and_absolute_float32_scores(scores, log_score_offset):
    """Capture native score margins before adding a large common offset."""

    offset_free = np.asarray(scores, dtype=np.float32)
    absolute = (
        np.asarray(scores, dtype=np.float64) + np.asarray(log_score_offset, dtype=np.float64)
    ).astype(np.float32)
    return offset_free, absolute


def _global_pass1_relion_projector_texture_enabled() -> bool:
    """Whether dense/global pass-1 significance uses texture arithmetic.

    Coarse significance defaults to RELION's texture projector.  Set the
    environment flag to false to force the manual/JAX diagnostic fallback.
    """
    token = os.environ.get(_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV, "1").strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV}={token!r}")


def _firstiter_cc_tree_top2_rescore_max_margin() -> float | None:
    """Return the near-tie margin for RELION coarse-tree replay."""

    token = os.environ.get(_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV, "").strip()
    if not token:
        return None
    if token.lower() in {"off", "none", "disable", "disabled"}:
        return None
    margin = float(token)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} must be a finite "
            f"non-negative float, got {token!r}",
        )
    return margin


def _dense_projection_scale(image_shape) -> float:
    """Match the dense E-step projection scaling used by the shared helper."""

    token = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
    n = int(image_shape[0])
    scale = {"-N2": -(n**2), "N2": float(n**2)}.get(token)
    if scale is None:
        raise ValueError(f"Unsupported RECOVAR_DENSE_MEANS_SCALE={token!r}")
    return scale


def _pass1_fused_enabled() -> bool:
    """Whether the fused-pass1 fast path is enabled.

    Off by default while the path is being validated. Set
    ``RECOVAR_PASS1_FUSED=1`` to opt in. Bit-identical to the unfused path
    when active (same ops, same order, same dtypes).
    """
    mode = os.environ.get(_SIGNIFICANCE_FUSED_PASS1_ENV, "0").strip().lower()
    return mode in {"1", "true", "yes", "on"}


@partial(
    jax.jit,
    static_argnames=(
        "image_shape",
        "proj_volume_shape",
        "volume_shape",
        "disc_type",
        "use_window",
        "use_float64_scoring",
        "rotation_block_size",
        "batch_size",
        "n_trans",
        "n_windowed",
        "max_r_static",
    ),
)
def _fused_score_priors_logsumexp_block(
    mean_for_proj,
    rots_b,
    shifted_data,
    batch_norm,
    ctf2_data,
    half_weights_for_score,
    window_indices,
    rotation_log_prior_block,
    translation_log_prior_per_image,
    class_log_prior_scalar,
    valid_count,
    class_max,
    class_sum,
    global_max,
    global_sum,
    *,
    image_shape: tuple,
    proj_volume_shape: tuple,
    volume_shape: tuple,
    disc_type: str,
    use_window: bool,
    use_float64_scoring: bool,
    rotation_block_size: int,
    batch_size: int,
    n_trans: int,
    n_windowed: int,
    max_r_static,
):
    """Fused pass-1 inner block: project + score + pad-mask + priors + 2× logsumexp.

    Replaces 4-5 separate JIT dispatches per (image_batch, class, rotation_block)
    with a single compiled boundary. JAX inlines the @jit-d leaf functions
    (compute_projections_block, _e_step_block_scores_windowed, _update_logsumexp)
    when traced from inside another @jit, so this is bit-identical to the
    unfused path while saving ~150ms of per-batch host-side dispatch at
    50k/256 K=1 (~16s/iter → ~2-4s/iter for pass1).
    """
    proj_kwargs = {}
    if use_window and max_r_static is not None:
        proj_kwargs["max_r"] = max_r_static
    proj_half_b, proj_abs2_half_b = compute_projections_block(
        mean_for_proj,
        rots_b,
        image_shape,
        proj_volume_shape,
        disc_type,
        **proj_kwargs,
    )

    if use_window:
        proj_w = proj_half_b[:, window_indices]
        proj_abs2_w = proj_abs2_half_b[:, window_indices]
        if not use_float64_scoring:
            proj_w = proj_w.astype(jnp.complex64)
            proj_abs2_w = proj_abs2_w.astype(jnp.float32)
        scores = _e_step_block_scores_windowed(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_w * half_weights_for_score,
            proj_abs2_w * half_weights_for_score,
            half_weights_for_score,
            batch_size,
            n_trans,
            n_windowed,
            image_shape,
            volume_shape,
        )
    else:
        if not use_float64_scoring:
            proj_half_b = proj_half_b.astype(jnp.complex64)
            proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
        scores = _e_step_block_scores(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_half_b * half_weights_for_score,
            proj_abs2_half_b * half_weights_for_score,
            half_weights_for_score,
            batch_size,
            n_trans,
            image_shape,
            volume_shape,
        )

    # Padding mask: -inf for rotations beyond valid_count (= n_rot - r0).
    pad_mask = jnp.arange(rotation_block_size)[None, :, None] < valid_count
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    scores = jnp.where(pad_mask, scores, neg_inf)

    # Priors: class scalar + rotation block + per-image translation.
    scores = scores + jnp.asarray(class_log_prior_scalar, dtype=scores.real.dtype)
    scores = scores + rotation_log_prior_block[None, :, None]
    scores = scores + translation_log_prior_per_image[:, None, :]

    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
    return scores, class_max, class_sum, global_max, global_sum


def _significance_score_cache_enabled(n_images, n_classes, n_rot, n_trans, *, use_float64_scoring: bool) -> bool:
    """Whether to keep pass-1 score blocks for reuse in pass 2.

    The cache is exact: it stores the already-prior-adjusted score tensors
    computed for the streaming logsumexp pass and reuses them when forming
    posterior weights/significance masks.  If the estimated tensor footprint is
    too large, callers fall back to the previous recompute path.
    """

    mode = os.environ.get(_SIGNIFICANCE_SCORE_CACHE_ENV, "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    force = mode in {"1", "true", "yes", "on", "force", "always"}
    itemsize = 8 if use_float64_scoring else 4
    estimated_bytes = int(n_images) * int(n_classes) * int(n_rot) * int(n_trans) * itemsize
    max_gb = float(os.environ.get(_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV, _SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB))
    return force or estimated_bytes <= int(max_gb * (1024**3))


def _uses_relion_background_fill(experiment_dataset) -> bool:
    image_source = getattr(experiment_dataset, "image_source", None)
    while hasattr(image_source, "parent"):
        image_source = image_source.parent
    backend = getattr(image_source, "backend", image_source)
    return getattr(backend, "image_mask_mode", None) == "relion_background_fill"


@nvtx.annotate("kclass.adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _compute_k_class_significance_batched(
    experiment_dataset,
    means,
    noise_variance,
    rotations,
    translations,
    disc_type,
    *,
    class_log_priors,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    score_with_masked_images=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    do_gridding_correction=False,
    square_window=False,
    use_float64_scoring=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    relion_projector_texture_interp: bool | None = None,
    score_mode: str = "gaussian",
    collect_significance: bool = True,
    return_class_best: bool = False,
    return_class_second: bool = False,
    debug_iteration: int | None = None,
    coarse_healpix_order: int | None = None,
    coarse_rotation_ids=None,
    translation_phase_source=None,
    relion_coarse_gaussian_default: bool = False,
    relion_f32_coarse_tie_ulps: int = 0,
    pad_final_image_batch: bool = False,
    stable_fourier_window_shapes: bool = False,
    coarse_gemm_diagnostic_scope: CoarseGaussianGemmDiagnosticScope | None = None,
):
    """Find significant samples from one posterior over ``class x rotation x translation``."""

    if return_class_second and not return_class_best:
        raise ValueError("return_class_second requires return_class_best")

    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.helpers.fourier_window import make_fourier_window_spec, relion_fftw_order_for_square_score_window
    from recovar.em.helpers.half_spectrum import make_half_image_weights, make_scoring_half_image_weights
    from recovar.em.helpers.image_shifts import apply_relion_integer_pre_shifts, tiled_half_image_phase_factors
    from recovar.em.helpers.oversampling import find_significant_rotations as _find_sig
    from recovar.em.helpers.preprocessing import prepare_batch_preprocess_operands, process_half_image
    from recovar.em.helpers.preprocessing import preprocess_batch as _preprocess_batch
    from recovar.em.helpers.preprocessing import preprocess_batch_firstiter_cc as _preprocess_batch_firstiter_cc
    from recovar.em.helpers.projection import compute_projections_block as _compute_projections_block
    from recovar.em.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from recovar.em.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows as _project_relion_projector_manual,
    )
    from recovar.em.scoring.scoring import (
        _e_step_block_scores,
        _e_step_block_scores_normalized_cc,
        _e_step_block_scores_windowed,
        _e_step_block_scores_windowed_normalized_cc,
        _relion_coarse_normalized_cc_rescore,
        _update_logsumexp,
    )
    from recovar.reconstruction import noise as noise_utils

    score_mode = str(score_mode)
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    # RELION's pdf_orientation/pdf_offset priors and score/evidence/Pmax
    # outputs are RFLOAT (double), never narrowed -- derive from the
    # caller's own use_float64_scoring instead of hardcoding float32.
    score_real_dtype = np.float64 if use_float64_scoring else np.float32
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    n_classes = int(means_array.shape[0])
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    if class_log_priors_np.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {class_log_priors_np.shape}")

    translations_source = np.asarray(
        translations if translation_phase_source is None else translation_phase_source,
    )
    if translations_source.shape != translations.shape:
        raise ValueError(
            "translation_phase_source must match translations: "
            f"{translations_source.shape} != {translations.shape}",
        )
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    n_images = int(experiment_dataset.n_units)
    input_image_batch_size = operator.index(image_batch_size)
    if input_image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    image_batch_size = input_image_batch_size
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (n_rot,):
            raise ValueError(
                f"coarse_rotation_ids must have shape ({n_rot},), got {coarse_rotation_ids.shape}",
            )
    if coarse_healpix_order is None:
        coarse_healpix_order = _infer_relion_coarse_healpix_order(n_rot)
    elif int(coarse_healpix_order) < 0:
        raise ValueError(f"coarse_healpix_order must be non-negative, got {coarse_healpix_order}")

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim != 4 or int(relion_projector_half.shape[0]) != n_classes:
            raise ValueError(
                "relion_projector_half must have shape "
                f"({n_classes}, z, y, x_half), got {relion_projector_half.shape}",
            )
    if projection_padding_factor > 1 and not use_relion_projector:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        means_for_proj = []
        proj_volume_shape = None
        for class_index in range(n_classes):
            mean_for_proj, proj_volume_shape = pad_volume_for_projection(
                means_array[class_index],
                experiment_dataset.volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
            means_for_proj.append(mean_for_proj)
    else:
        means_for_proj = [means_array[class_index] for class_index in range(n_classes)]
        proj_volume_shape = experiment_dataset.volume_shape

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=score_mode != "normalized_cc",
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    score_size = int(image_shape[0]) if current_size is None else int(current_size)
    window_indices = window_spec.score_indices
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    coarse_texture_interp = (
        _global_pass1_relion_projector_texture_enabled()
        if relion_projector_texture_interp is None
        else bool(relion_projector_texture_interp)
    )
    coarse_floorf_quirk = bool(
        use_float64_scoring
        and not coarse_texture_interp
        and _relion_acc_double_floorf_quirk_enabled()
    )
    tree_rescore_max_margin = _firstiter_cc_tree_top2_rescore_max_margin()
    # The environment setting spans the full process, while only iteration 1
    # uses normalized CC.  Later Gaussian iterations must remain unaffected.
    tree_rescore_enabled = (
        tree_rescore_max_margin is not None and score_mode == "normalized_cc"
    )
    # The environment flag spans the complete refinement process. Iteration 1
    # may use normalized CC, while this intervention applies only to later
    # Gaussian coarse passes. Keep the flag dormant for the CC call instead of
    # rejecting the process before it reaches the intended boundary.
    coarse_gaussian_ffi_requested = _k1_coarse_gaussian_ffi_enabled(
        default=_coarse_gaussian_ffi_default(
            relion_coarse_gaussian_default,
            use_relion_projector=use_relion_projector,
            use_float64_scoring=use_float64_scoring,
            coarse_texture_interp=coarse_texture_interp,
        ),
    )
    coarse_gaussian_ffi_enabled = (
        coarse_gaussian_ffi_requested and score_mode == "gaussian"
    )
    coarse_gaussian_sincosf_requested = _k1_coarse_gaussian_sincosf_enabled(
        default=relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled,
    )
    coarse_gaussian_sincosf_enabled = (
        coarse_gaussian_sincosf_requested and score_mode == "gaussian"
    )
    if coarse_gaussian_sincosf_enabled and not coarse_gaussian_ffi_enabled:
        raise ValueError(
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1",
        )
    coarse_fused_projector_requested = _k1_coarse_fused_projector_enabled(
        default=(
            relion_coarse_gaussian_default
            and coarse_gaussian_ffi_enabled
            and coarse_gaussian_sincosf_enabled
        ),
    )
    coarse_fused_projector_enabled = (
        coarse_fused_projector_requested
        and score_mode == "gaussian"
        and _k1_coarse_fused_projector_supports_padding(projection_padding_factor)
    )
    coarse_canonical_reduction_requested = (
        _relion_coarse_canonical_reduction_enabled(
            default=(
                relion_coarse_gaussian_default
                and coarse_fused_projector_enabled
            ),
        )
    )
    coarse_native_atomic_reduction_requested = (
        _k1_coarse_native_atomic_reduction_enabled()
    )
    coarse_native_atomic_reduction_enabled = (
        _k1_coarse_native_atomic_reduction_selected(
            requested=coarse_native_atomic_reduction_requested,
            score_mode=score_mode,
            translation_count=n_trans,
        )
    )
    if (
        coarse_native_atomic_reduction_enabled
        and _RELION_COARSE_CANONICAL_REDUCTION_ENV in os.environ
        and coarse_canonical_reduction_requested
    ):
        raise ValueError(
            f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 conflicts with "
            f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV}=1",
        )
    coarse_canonical_reduction_enabled = bool(
        coarse_canonical_reduction_requested
        and score_mode == "gaussian"
        and not coarse_native_atomic_reduction_enabled
    )
    if coarse_canonical_reduction_enabled and not coarse_fused_projector_enabled:
        raise ValueError(
            f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV} requires "
            f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
        )
    if coarse_native_atomic_reduction_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 currently "
                "supports K=1 only",
            )
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 requires "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
            )
    elif coarse_native_atomic_reduction_requested and score_mode == "gaussian":
        logger.debug(
            "RELION native atomic coarse reduction unavailable for %d "
            "translations; retaining the configured canonical reduction",
            n_trans,
        )
    coarse_prehalf_weight_requested = _k1_coarse_prehalf_weight_enabled()
    coarse_prehalf_weight_enabled = bool(
        coarse_prehalf_weight_requested and coarse_native_atomic_reduction_enabled
    )
    if coarse_prehalf_weight_requested and not coarse_prehalf_weight_enabled:
        raise ValueError(
            f"{_K1_COARSE_PREHALF_WEIGHT_ENV}=1 requires the effective "
            f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 K=1 Gaussian T=29 path",
        )
    coarse_single_lane_canonical_requested = (
        _k1_coarse_single_lane_canonical_enabled()
    )
    coarse_single_lane_canonical_enabled = (
        _k1_coarse_single_lane_canonical_selected(
            requested=coarse_single_lane_canonical_requested,
            score_mode=score_mode,
            translation_count=n_trans,
        )
    )
    if coarse_single_lane_canonical_enabled:
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}=1 requires "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
            )
        if not coarse_canonical_reduction_enabled:
            raise ValueError(
                f"{_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}=1 requires "
                f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV}=1",
            )
    elif coarse_single_lane_canonical_requested and score_mode == "gaussian":
        logger.debug(
            "RELION coarse single-lane specialization unavailable for %d "
            "translations; retaining generic canonical reduction",
            n_trans,
        )
    coarse_multistream_worker_count = _k1_coarse_multistream_worker_count()
    coarse_multistream_enabled = (
        coarse_multistream_worker_count > 0 and score_mode == "gaussian"
    )
    if coarse_multistream_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV}=8 currently supports K=1 only",
            )
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV}=8 requires the accepted "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1 path",
            )
    if (
        coarse_fused_projector_requested
        and score_mode == "gaussian"
        and not _k1_coarse_fused_projector_supports_padding(projection_padding_factor)
    ):
        logger.warning(
            "K=1 RELION fused coarse projector disabled for padding_factor=%d; "
            "using the padding-aware projector and exact diff2 path",
            int(projection_padding_factor),
        )
    if coarse_fused_projector_enabled and not (
        coarse_gaussian_ffi_enabled and coarse_gaussian_sincosf_enabled
    ):
        raise ValueError(
            f"{_K1_COARSE_FUSED_PROJECTOR_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1 and "
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV}=1",
        )
    exact_coarse_operands_requested = _k1_relion_exact_coarse_operands_enabled(
        default=(
            relion_coarse_gaussian_default
            and coarse_gaussian_ffi_enabled
            and coarse_gaussian_sincosf_enabled
        ),
    )
    exact_coarse_operands_enabled = (
        exact_coarse_operands_requested and score_mode == "gaussian"
    )
    if exact_coarse_operands_enabled and not coarse_gaussian_sincosf_enabled:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1 and "
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV}=1",
        )
    exact_coarse_skip_generic_operands_requested = (
        _k1_relion_exact_coarse_skip_generic_operands_enabled()
    )
    exact_coarse_skip_generic_operands_enabled = (
        _resolve_k1_relion_exact_coarse_skip_generic_operands(
            requested=exact_coarse_skip_generic_operands_requested,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
        )
    )
    exact_coarse_assembly_profile_enabled = (
        _k1_relion_exact_coarse_assembly_profile_enabled()
    )
    coarse_gaussian_native_texture_requested = (
        _k1_coarse_gaussian_native_texture_enabled(
            # Keep the fused texture scorer as an explicit diagnostic.  The
            # preprojected rectangular FFI consumes the same exact image/CTF
            # operands but matches RELION's coarse support boundary more
            # closely; the fused scorer can move marginal parents across the
            # adaptive-significance cutoff.
            default=False,
        )
    )
    coarse_gaussian_native_texture_enabled = (
        coarse_gaussian_native_texture_requested and score_mode == "gaussian"
    )
    coarse_gaussian_gemm_macro_requested = _coarse_gaussian_gemm_macro_enabled()
    coarse_gaussian_gemm_projection_cache_requested = (
        _coarse_gaussian_gemm_projection_cache_enabled()
    )
    coarse_gaussian_gemm_hybrid_requested = (
        _coarse_gaussian_gemm_hybrid_enabled()
    )
    coarse_gaussian_gemm_compact_posterior_requested = (
        _coarse_gaussian_gemm_compact_posterior_enabled()
    )
    coarse_gaussian_gemm_device_transaction_requested = (
        _coarse_gaussian_gemm_device_transaction_enabled()
    )
    coarse_gaussian_gemm_real_cross_requested = _coarse_gaussian_gemm_real_cross_enabled()
    coarse_max_posterior_physical_batch = _coarse_max_posterior_physical_batch_enabled()
    partition_token = os.environ.get("RECOVAR_COARSE_ROW_PARTITION", "0")
    if partition_token not in {"0", "1"}:
        raise ValueError("RECOVAR_COARSE_ROW_PARTITION must be 0 or 1")
    coarse_row_partition_requested = partition_token == "1"
    posterior_token = os.environ.get("RECOVAR_COARSE_POSTERIOR_TRANSACTION", "0")
    if posterior_token not in {"0", "1"}:
        raise ValueError("RECOVAR_COARSE_POSTERIOR_TRANSACTION must be 0 or 1")
    coarse_cuda_posterior_requested = posterior_token == "1"
    if coarse_cuda_posterior_requested and (
        not coarse_row_partition_requested or relion_f32_coarse_tie_ulps != 0
    ):
        raise ValueError("CUDA coarse posterior requires row partition and exact ties")
    if coarse_gaussian_gemm_real_cross_requested and (
        not coarse_gaussian_gemm_hybrid_requested or coarse_gaussian_gemm_device_transaction_requested
    ):
        raise ValueError("Real-cross certificate requires host-orchestrated coarse GEMM hybrid")
    if coarse_gaussian_gemm_device_transaction_requested and not (
        coarse_gaussian_gemm_hybrid_requested
        and coarse_gaussian_gemm_compact_posterior_requested
    ):
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION_ENV}=1 requires "
            "the compact coarse GEMM hybrid",
        )
    coarse_gaussian_gemm_hybrid_image_batch_size_request = (
        _coarse_gaussian_gemm_hybrid_image_batch_size_request()
    )
    if coarse_gaussian_gemm_compact_posterior_requested:
        _validate_coarse_gaussian_gemm_compact_posterior_request(
            hybrid_enabled=coarse_gaussian_gemm_hybrid_requested,
            return_class_best=return_class_best,
            return_class_second=return_class_second,
        )
    coarse_gaussian_gemm_hybrid_capacity = (
        _coarse_gaussian_gemm_hybrid_block_capacity()
        if coarse_gaussian_gemm_hybrid_requested
        else DEFAULT_ROTATION_BLOCK_CAPACITY
    )
    coarse_gaussian_score_backend = _resolve_coarse_gaussian_score_backend(
        gemm_macro_requested=coarse_gaussian_gemm_macro_requested,
        gemm_hybrid_requested=coarse_gaussian_gemm_hybrid_requested,
        score_mode=score_mode,
        fused_projector_requested=coarse_fused_projector_requested,
        fused_projector_enabled=coarse_fused_projector_enabled,
        canonical_reduction_requested=coarse_canonical_reduction_requested,
        canonical_reduction_enabled=coarse_canonical_reduction_enabled,
        native_atomic_reduction_requested=coarse_native_atomic_reduction_requested,
        native_atomic_reduction_enabled=coarse_native_atomic_reduction_enabled,
        single_lane_canonical_requested=coarse_single_lane_canonical_requested,
        single_lane_canonical_enabled=coarse_single_lane_canonical_enabled,
        multistream_requested=coarse_multistream_worker_count > 0,
        multistream_enabled=coarse_multistream_enabled,
        native_texture_requested=coarse_gaussian_native_texture_requested,
        native_texture_enabled=coarse_gaussian_native_texture_enabled,
    )
    coarse_gaussian_gemm_macro_enabled = (
        coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.GEMM_MACRO
    )
    coarse_gaussian_fused_full_fallback_armed = bool(
        coarse_gaussian_gemm_hybrid_requested
        and coarse_fused_projector_enabled
    )
    (
        coarse_gaussian_gemm_diagnostic_dir,
        coarse_gaussian_gemm_diagnostic_targets,
    ) = _coarse_gaussian_gemm_diagnostic_request()
    (
        coarse_gaussian_gemm_stream_diagnostic_dir,
        coarse_gaussian_gemm_stream_topk,
    ) = _coarse_gaussian_gemm_streaming_diagnostic_request(
        max_significants=max_significants,
    )
    (
        coarse_runtime_prefix_dump_dir,
        coarse_runtime_prefix_dump_targets,
        coarse_runtime_prefix_dump_label,
    ) = _coarse_runtime_prefix_dump_request()
    if coarse_runtime_prefix_dump_dir is not None and not (
        coarse_gaussian_gemm_hybrid_requested
        and coarse_gaussian_gemm_compact_posterior_requested
    ):
        raise ValueError(
            "coarse runtime-prefix operand capture requires the compact "
            "certified coarse GEMM hybrid",
        )
    coarse_gaussian_gemm_requested_targets = coarse_gaussian_gemm_diagnostic_targets
    coarse_gaussian_gemm_diagnostic_scope = None
    coarse_gaussian_gemm_diagnostic_selection_policy = None
    coarse_gaussian_gemm_any_diagnostic = bool(
        coarse_gaussian_gemm_diagnostic_dir is not None
        or coarse_gaussian_gemm_stream_diagnostic_dir is not None
    )
    if coarse_gaussian_gemm_any_diagnostic:
        if not coarse_gaussian_gemm_macro_enabled:
            raise ValueError(
                "coarse GEMM diagnostics require "
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
            )
        if not collect_significance:
            raise ValueError(
                "coarse GEMM diagnostics require "
                "collect_significance=True so exact support can be compared",
            )
        (
            coarse_gaussian_gemm_diagnostic_scope,
            explicitly_scoped_diagnostic,
        ) = _resolve_coarse_gaussian_gemm_diagnostic_scope(
            coarse_gemm_diagnostic_scope,
            debug_iteration=debug_iteration,
            current_size=current_size,
        )
    if coarse_gaussian_gemm_diagnostic_dir is not None:
        logger.warning(
            "coarse GEMM paired capture is requested; calls containing target "
            "particles execute both direct and GEMM scorers, and the configured "
            "run is excluded from runtime qualification"
        )
        if explicitly_scoped_diagnostic:
            available_original_indices = set(
                int(value)
                for value in original_image_indices(
                    experiment_dataset,
                    np.arange(n_images, dtype=np.int64),
                )
            )
            coarse_gaussian_gemm_diagnostic_targets = (
                coarse_gaussian_gemm_requested_targets & available_original_indices
            )
            coarse_gaussian_gemm_diagnostic_selection_policy = (
                "explicit_call_scope_intersection"
            )
        else:
            coarse_gaussian_gemm_diagnostic_selection_policy = "strict_single_call"
    if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
        logger.warning(
            "coarse GEMM all-particle streaming exact-rescore diagnostic is "
            "requested (topk=%d); every score block executes both scorers, only "
            "bounded summaries are retained, and the run is excluded from "
            "runtime qualification",
            coarse_gaussian_gemm_stream_topk,
        )
    if coarse_gaussian_gemm_macro_enabled and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 requires "
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if coarse_gaussian_gemm_projection_cache_requested:
        _validate_coarse_gaussian_gemm_projection_cache_request(
            macro_enabled=coarse_gaussian_gemm_macro_enabled,
            n_classes=n_classes,
            n_rotations=n_rot,
            coarse_gaussian_ffi_enabled=coarse_gaussian_ffi_enabled,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            use_relion_projector=use_relion_projector,
            relion_texture_interp_enabled=coarse_texture_interp,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_scoring=use_float64_scoring,
            relion_projector_dtype=(
                relion_projector_half[0].dtype if use_relion_projector else None
            ),
        )
    if coarse_gaussian_native_texture_enabled and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV} requires "
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    relion_f32_coarse_support_requested = _k1_relion_f32_coarse_support_enabled(
        default=relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled,
    )
    relion_f32_coarse_support_enabled = (
        relion_f32_coarse_support_requested and score_mode == "gaussian"
    )
    if coarse_gaussian_gemm_hybrid_requested:
        _validate_coarse_gaussian_gemm_hybrid_request(
            macro_enabled=coarse_gaussian_gemm_macro_enabled,
            projection_cache_enabled=coarse_gaussian_gemm_projection_cache_requested,
            n_classes=n_classes,
            n_rotations=n_rot,
            score_mode=score_mode,
            coarse_gaussian_ffi_enabled=coarse_gaussian_ffi_enabled,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            relion_f32_coarse_support_enabled=relion_f32_coarse_support_enabled,
            collect_significance=collect_significance,
            any_diagnostic_requested=coarse_gaussian_gemm_any_diagnostic,
        )
    if coarse_row_partition_requested and not (
        coarse_gaussian_gemm_hybrid_requested
        and coarse_gaussian_gemm_compact_posterior_requested
        and relion_f32_coarse_support_enabled
        and coarse_gaussian_ffi_enabled
        and n_classes == 1
        and collect_significance
        and not return_class_best
        and not return_class_second
        and not coarse_gaussian_gemm_device_transaction_requested
        and not coarse_gaussian_fused_full_fallback_armed
        and not coarse_gaussian_gemm_any_diagnostic
        and coarse_runtime_prefix_dump_dir is None
    ):
        raise ValueError(
            "RECOVAR_COARSE_ROW_PARTITION requires the K=1 compact host hybrid "
            "with ordinary full scoring, float32 posterior, and no score dumps/class diagnostics"
        )
    exact_compact_preprocess_requested = (
        _k1_relion_exact_compact_preprocess_enabled()
    )
    exact_compact_preprocess_enabled = (
        _resolve_k1_relion_exact_compact_preprocess(
            requested=exact_compact_preprocess_requested,
            exact_coarse_skip_generic_operands_enabled=(
                exact_coarse_skip_generic_operands_enabled
            ),
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            coarse_gaussian_gemm_hybrid_requested=(
                coarse_gaussian_gemm_hybrid_requested
            ),
            coarse_gaussian_gemm_compact_posterior_requested=(
                coarse_gaussian_gemm_compact_posterior_requested
            ),
            score_mode=score_mode,
            any_diagnostic_requested=coarse_gaussian_gemm_any_diagnostic,
        )
    )
    if exact_compact_preprocess_enabled and (
        collect_significance
        and _significance_debug_dump_matches(
            current_size=current_size,
            debug_iteration=debug_iteration,
        )
    ):
        raise ValueError(
            f"{_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV}=1 does not support "
            "raw significance score dumps",
        )
    if coarse_gaussian_gemm_hybrid_image_batch_size_request is not None:
        prefix = (
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} requires"
        )
        if not coarse_gaussian_gemm_hybrid_requested:
            raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
        if not coarse_gaussian_gemm_compact_posterior_requested:
            raise ValueError(
                f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
            )
    if relion_f32_coarse_support_enabled:
        if use_float64_scoring:
            raise ValueError(
                f"{_K1_RELION_F32_COARSE_SUPPORT_ENV} requires production float32 scoring",
            )
        logger.warning(
            "RELION CUDA float32 coarse support enabled (%s): current_size=%d",
            (
                "guarded fresh InitialModel default"
                if relion_coarse_gaussian_default
                and _K1_RELION_F32_COARSE_SUPPORT_ENV not in os.environ
                else "environment override"
            ),
            int(image_shape[0]) if current_size is None else int(current_size),
        )
    coarse_gaussian_full_to_compact = None
    coarse_gaussian_full_to_compact_np = None
    coarse_gaussian_score_indices = None
    coarse_gaussian_score_indices_np = None
    coarse_gaussian_score_active_mask = None
    coarse_gaussian_window_positions = None
    coarse_gaussian_powerclass = None
    coarse_gaussian_projector_full = None
    coarse_gaussian_gemm_resource_estimate = None
    coarse_gaussian_gemm_projection_cache_plan = None
    coarse_gaussian_gemm_certificate_topology = None
    coarse_gaussian_square_layout = None
    if coarse_gaussian_ffi_enabled:
        if use_float64_scoring and n_classes != 1:
            raise ValueError("Diagnostic float64 coarse Gaussian FFI is restricted to K=1")
        if not use_relion_projector or (not use_float64_scoring and not coarse_texture_interp):
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires the supplied RELION "
                "projector"
            )
        if not half_spectrum_scoring:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires half-spectrum scoring"
            )
        if n_trans > 128:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} supports at most 128 "
                f"translations, got {n_trans}"
            )
        from recovar import cuda_backproject
        from recovar.em.helpers.projection import relion_projector_half_to_texture_full
        from recovar.em.relion.relion_ctf import _relion_exact_ctf_half_from_source_star
        from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32
        from recovar.em.sparse_pass2.sparse_pass2_scoring import (
            _relion_cuda_corr_img_from_rfloat_ctf,
            _relion_cuda_pixel_correction_from_rfloat_ctf,
            _relion_cuda_powerclass_highres_xi2_half,
        )

        if jax.default_backend() != "gpu" or not cuda_backproject.cuda_available():
            raise RuntimeError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires the custom CUDA backend"
            )
        active_score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else np.asarray(window_spec.score_indices_np, dtype=np.int32)
        )
        coarse_gaussian_square_layout = _plan_coarse_gaussian_square_layout(
            image_shape,
            score_size,
            active_score_indices_np,
            stable_fourier_window_shapes=bool(stable_fourier_window_shapes),
        )
        square_score_indices_np = coarse_gaussian_square_layout.score_indices_np
        square_score_count = coarse_gaussian_square_layout.physical_square_count
        coarse_gaussian_projector_output_size = (
            coarse_gaussian_square_layout.physical_current_size
        )
        coarse_gaussian_score_indices_np = np.asarray(
            square_score_indices_np,
            dtype=np.int32,
        )
        coarse_gaussian_score_indices = jnp.asarray(
            coarse_gaussian_score_indices_np,
            dtype=jnp.int32,
        )
        coarse_gaussian_score_active_mask = jnp.asarray(
            coarse_gaussian_square_layout.score_active_mask_np,
            dtype=jnp.bool_,
        )
        coarse_gaussian_window_positions = jnp.asarray(
            _compact_projection_window_positions(
                square_score_indices_np,
                active_score_indices_np,
            ),
            dtype=jnp.int32,
        )
        coarse_gaussian_full_to_compact_np = np.asarray(
            coarse_gaussian_square_layout.full_to_compact_np,
            dtype=np.int32,
        )
        coarse_gaussian_full_to_compact = jnp.asarray(
            coarse_gaussian_full_to_compact_np,
            dtype=jnp.int32,
        )
        if coarse_gaussian_gemm_hybrid_requested:
            coarse_gaussian_gemm_certificate_topology = (
                plan_coarse_gemm_certificate_topology(
                    coarse_gaussian_full_to_compact_np,
                    compact_pixel_count=int(square_score_count),
                    translation_count=n_trans,
                )
            )
        coarse_gaussian_powerclass = _relion_cuda_powerclass_highres_xi2_half
        if coarse_gaussian_gemm_macro_enabled:
            coarse_gaussian_gemm_resource_estimate = _coarse_gaussian_gemm_resources(
                rotation_block_size=int(rotation_block_size),
                image_shape=image_shape,
                compact_pixel_count=int(square_score_count),
                budget_bytes=_coarse_gaussian_gemm_projected_transient_budget_bytes(),
            )
        if coarse_gaussian_gemm_projection_cache_requested:
            coarse_gaussian_gemm_projection_cache_plan = (
                _plan_coarse_gaussian_gemm_projection_cache(
                    n_rotations=n_rot,
                    compact_pixel_count=int(square_score_count),
                    image_shape=image_shape,
                    budget_bytes=(
                        _coarse_gaussian_gemm_projection_cache_budget_bytes()
                    ),
                )
            )
        if coarse_gaussian_gemm_hybrid_image_batch_size_request is not None:
            image_batch_size = _resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
                input_image_batch_size,
                requested_batch_size=(
                    coarse_gaussian_gemm_hybrid_image_batch_size_request
                ),
                n_images=n_images,
                certificate_chunk_rows=(
                    1
                    if coarse_gaussian_gemm_projection_cache_plan is None
                    else coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                ),
                n_translations=n_trans,
                hybrid_enabled=coarse_gaussian_gemm_hybrid_requested,
                compact_posterior_enabled=(
                    coarse_gaussian_gemm_compact_posterior_requested
                ),
            )
            if coarse_gaussian_gemm_projection_cache_plan is None:
                raise RuntimeError(
                    "compact-hybrid image-batch override is missing its "
                    "projection-cache plan",
                )
            logger.warning(
                "Opt-in shared compact-hybrid image batching enabled: "
                "input=%d requested=%d effective=%d images=%d; each streamed "
                "certificate tile retains at most %d candidate values",
                input_image_batch_size,
                coarse_gaussian_gemm_hybrid_image_batch_size_request,
                image_batch_size,
                n_images,
                image_batch_size
                * coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                * n_trans,
            )
        coarse_gaussian_projector_full_by_class = None
        coarse_gaussian_translation_angles = None
        if (
            coarse_gaussian_score_backend
            in _COARSE_GAUSSIAN_FUSED_SCORE_BACKENDS
            or coarse_gaussian_fused_full_fallback_armed
        ):
            coarse_gaussian_projector_full_by_class = [
                relion_projector_half_to_texture_full(
                    relion_projector_half[class_index],
                )
                for class_index in range(n_classes)
            ]
            coarse_gaussian_translation_angles = jnp.asarray(
                _relion_translation_angles_f32(translations_source, image_shape),
                dtype=jnp.float32,
            )
            if (
                coarse_gaussian_score_backend
                in _COARSE_GAUSSIAN_FUSED_SCORE_BACKENDS
            ):
                # One all-rotation launch preserves RELION's 128-orientation
                # main segment followed by its one-orientation tail.  It also
                # avoids projecting a memory-planner padding block.
                rotation_block_size = n_rot
        logger.warning(
            "RELION coarse Gaussian FFI enabled (%s, %s): "
            "classes=%d current_size=%d physical_size=%d square_pixels=%d "
            "translations=%d stable_shapes=%s",
            (
                "guarded fresh InitialModel default"
                if relion_coarse_gaussian_default
                and _K1_COARSE_GAUSSIAN_FFI_ENV not in os.environ
                else "environment override"
            ),
            "float64" if use_float64_scoring else "float32",
            n_classes,
            score_size,
            coarse_gaussian_projector_output_size,
            square_score_count,
            n_trans,
            bool(stable_fourier_window_shapes),
        )
        if coarse_gaussian_sincosf_enabled:
            logger.warning(
                "RELION coarse CUDA sincosf translation enabled (%s): "
                "classes=%d current_size=%d square_pixels=%d translations=%d",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_GAUSSIAN_SINCOSF_ENV not in os.environ
                    else "environment override"
                ),
                n_classes,
                score_size,
                square_score_count,
                n_trans,
            )
        if exact_coarse_operands_enabled:
            logger.warning(
                "RELION exact coarse operands enabled (%s): per-image FFT, "
                "RFLOAT CTF division/square, and binary64-to-XFLOAT inverse noise",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_RELION_EXACT_COARSE_OPERANDS_ENV not in os.environ
                    else "environment override"
                ),
            )
            if exact_coarse_skip_generic_operands_enabled:
                logger.warning(
                    "Opt-in exact-coarse single-translation path enabled: "
                    "skipping generic square operands that the exact-source "
                    "assembly replaces before first use",
                )
            if exact_compact_preprocess_enabled:
                logger.warning(
                    "Opt-in exact+compact preprocessing enabled: skipping the "
                    "unused generic half-image FFT, CTF evaluation, and full "
                    "translated score image",
                )
        if (
            coarse_gaussian_score_backend
            in _COARSE_GAUSSIAN_FUSED_SCORE_BACKENDS
        ):
            logger.warning(
                "RELION fused coarse projector/diff2 enabled (%s): "
                "classes=%d rotations=%d translations=%d",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_FUSED_PROJECTOR_ENV not in os.environ
                    else "environment override"
                ),
                n_classes,
                n_rot,
                n_trans,
            )
            if coarse_canonical_reduction_enabled:
                logger.warning(
                    "RELION shared coarse canonical lane reduction enabled (%s)",
                    (
                        "guarded exact-path default"
                        if _RELION_COARSE_CANONICAL_REDUCTION_ENV not in os.environ
                        else "environment override"
                    ),
                )
            if coarse_native_atomic_reduction_enabled:
                logger.warning(
                    "Opt-in native RELION atomic coarse lane reduction enabled: "
                    "translations=%d",
                    n_trans,
                )
            if coarse_prehalf_weight_enabled:
                logger.warning(
                    "Opt-in RELION source-ordered coarse prehalf weight enabled: "
                    "translations=%d",
                    n_trans,
                )
            if coarse_single_lane_canonical_enabled:
                logger.warning(
                    "Opt-in RELION coarse single-lane canonical specialization "
                    "enabled: translations=%d",
                    n_trans,
                )
            if coarse_multistream_enabled:
                logger.warning(
                    "Opt-in shared K=1 coarse multistream dispatcher enabled: "
                    "workers=%d actual image rows only",
                    coarse_multistream_worker_count,
                )
        if coarse_gaussian_fused_full_fallback_armed:
            logger.warning(
                "Opt-in certified hybrid native fused full fallback armed: "
                "the fused scorer executes only for whole-batch full-dense exits",
            )
        if coarse_gaussian_gemm_macro_enabled:
            logger.warning(
                "Opt-in shared coarse projection-once/GEMM macro enabled: "
                "classes=%d rotations=%d image_lanes=%d translations=%d",
                n_classes,
                n_rot,
                int(image_batch_size),
                n_trans,
            )
        if coarse_gaussian_gemm_hybrid_requested:
            logger.warning(
                "Opt-in certified K=1 coarse GEMM/source16 hybrid enabled: "
                "rotations=%d translations=%d selected_block_capacity=%d; "
                "all ineligible batches use full %s direct fallback",
                n_rot,
                n_trans,
                coarse_gaussian_gemm_hybrid_capacity,
                (
                    "native fused"
                    if coarse_gaussian_fused_full_fallback_armed
                    else "rectangular"
                ),
            )
            if coarse_gaussian_gemm_compact_posterior_requested:
                logger.warning(
                    "Opt-in compact certified-hybrid posterior enabled: "
                    "selected scores remain in fixed-capacity source16 order; "
                    "the dense selected-score table remains the fail-closed oracle",
                )
        if coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.NATIVE_TEXTURE:
            from recovar.em.helpers.projection import relion_projector_half_to_texture_full

            coarse_gaussian_projector_full = jnp.asarray(
                relion_projector_half_to_texture_full(relion_projector_half[0])
                * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
                dtype=jnp.complex64,
            )
            logger.warning(
                "K=1 RELION native texture coarse scoring enabled (%s): "
                "projection, translation, and score reduction share one kernel",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV not in os.environ
                    else "environment override"
                ),
            )
    tree_rescore_fftw_order = None
    tree_rescore_translation_angles = None
    if tree_rescore_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} currently "
                "supports K=1 only",
            )
        if not return_class_best:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "return_class_best=True",
            )
        if use_float64_scoring:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "production float32 scoring",
            )
        if not use_relion_projector or not coarse_texture_interp:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "the supplied RELION projector with texture interpolation",
            )
        if not half_spectrum_scoring:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "half-spectrum scoring",
            )
        from recovar import cuda_backproject
        from recovar.em.helpers.projection import relion_projector_half_to_texture_full
        from recovar.em.relion.relion_ctf import _relion_exact_ctf_half_from_source_star
        from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32
        from recovar.em.sparse_pass2.sparse_pass2_scoring import (
            _relion_cuda_corr_img_from_rfloat_ctf,
            _relion_cuda_pixel_correction_from_rfloat_ctf,
        )

        if (
            jax.default_backend() != "gpu"
            or not cuda_backproject.custom_cuda_requested()
            or not cuda_backproject.cuda_available()
        ):
            raise RuntimeError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "the custom CUDA backend",
            )
        coarse_gaussian_projector_full = jnp.asarray(
            relion_projector_half_to_texture_full(relion_projector_half[0])
            * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
            dtype=jnp.complex64,
        )
        score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else window_spec.score_indices_np
        )
        tree_rescore_fftw_order = jnp.asarray(
            relion_fftw_order_for_square_score_window(
                image_shape,
                score_size,
                score_indices_np,
            ),
            dtype=jnp.int32,
        )
        tree_rescore_translation_angles = jnp.asarray(
            _relion_translation_angles_f32(translations_source, image_shape),
            dtype=jnp.float32,
        )
        logger.warning(
            "Opt-in RELION coarse-tree top-2 rescore enabled: max_margin=%g current_size=%d",
            tree_rescore_max_margin,
            score_size,
        )
    track_class_second = return_class_second or tree_rescore_enabled
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = window_spec.score_values(half_weights)

    if coarse_gaussian_native_texture_enabled:
        # RELION's SPA coarse scorer launches one particle over the complete
        # orientation set.  Splitting rotations changes the float32 atomic
        # accumulation schedule at adaptive-significance boundaries.
        rotation_block_size = n_rot
        logger.warning(
            "K=1 RELION native texture coarse diagnostic: one particle per "
            "full orientation grid (%d rotations)",
            n_rot,
        )

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [
                rotations,
                np.tile(np.eye(3, dtype=np.asarray(rotations).dtype), (pad_size, 1, 1)),
            ],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=score_real_dtype)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=score_real_dtype)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
                )
        else:
            raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    norm_half_weights = make_half_image_weights(image_shape)
    use_relion_numpy_preprocess = _uses_relion_background_fill(experiment_dataset)

    def _preprocess_batch_relion_numpy(
        batch_data,
        ctf_params,
        batch_size,
        *,
        return_unshifted_score_weighted=False,
    ):
        processed_half = experiment_dataset.process_images_half(
            np.asarray(batch_data),
            apply_image_mask=score_with_masked_images,
        )
        processed_half = jnp.asarray(processed_half)
        ctf_half = config.compute_ctf_half(jnp.asarray(ctf_params, dtype=score_real_dtype))
        ctf2_over_nv_half = ctf_half**2 / noise_variance_half
        ctf_weighted = processed_half * ctf_half / noise_variance_half
        translations_tiled = jnp.repeat(jnp.asarray(translations)[None], batch_size, axis=0).reshape(
            batch_size * n_trans,
            -1,
        )
        weighted_tiled = jnp.repeat(ctf_weighted[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans,
            -1,
        )
        shifted_half = core.translate_images(
            weighted_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )
        batch_norm = jnp.sum(
            (jnp.abs(processed_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real
        result = (shifted_half, batch_norm, ctf2_over_nv_half)
        if return_unshifted_score_weighted:
            return (*result, ctf_weighted)
        return result

    coarse_gaussian_shifted_corrected = None
    coarse_gaussian_unshifted_corrected = None
    coarse_gaussian_translation_angles = None
    coarse_gaussian_pixel_weight = None
    coarse_gaussian_initial_diff2 = None

    # The texture projector naturally produces a centered current-size crop.
    # Ask it only for the rows consumed by the scorer instead of scattering the
    # crop into a full image and immediately gathering the same rows again.
    # This is an exact index remapping and avoids a large transient scatter for
    # global rotation blocks.
    projector_compact_indices_np = None
    projector_output_size = None
    if use_relion_projector and coarse_texture_interp:
        if coarse_gaussian_ffi_enabled:
            projector_compact_indices_np = coarse_gaussian_score_indices_np
            projector_output_size = coarse_gaussian_projector_output_size
        elif use_window:
            projector_compact_indices_np = window_spec.score_indices_np
            projector_output_size = score_size
    projector_returns_compact = projector_compact_indices_np is not None

    coarse_rotated_radius = _coarse_rotated_radius_enabled()
    if coarse_rotated_radius and not (
        use_relion_projector and coarse_texture_interp and projector_returns_compact
    ):
        raise ValueError("rotated coarse radius requires the compact RELION texture projector")

    def _project_relion_compact_score_rows(
        class_index,
        rots_b,
        *,
        return_abs2: bool,
    ):
        """Project the exact compact rows shared by direct and GEMM scoring."""

        if not (
            use_relion_projector
            and coarse_texture_interp
            and projector_returns_compact
        ):
            raise RuntimeError(
                "compact RELION score-row projection requires the texture "
                "projector and an explicit score-row table",
            )
        projected, projected_abs2 = _compute_relion_projector_projections_block(
            relion_projector_half[class_index],
            rots_b,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=return_abs2,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(projector_output_size),
            # Keep the already-host-resident table on the host so validation
            # cannot materialize its JAX mirror once per score block.
            pixel_indices=projector_compact_indices_np,
            relion_texture_interp=True,
            # Certificate and exact scorer share this table. The qualified
            # legacy convention masks source pixels; the opt-in correction
            # uses RELION's rotated float32 radius without changing storage.
            mask_current_image_disk=not coarse_rotated_radius,
            image_r_max=(
                jnp.asarray(score_size // 2, dtype=jnp.int32)
                if coarse_rotated_radius else None
            ),
            current_image_mask_size=(
                jnp.asarray(score_size, dtype=jnp.int32)
                if stable_fourier_window_shapes
                else None
            ),
        )
        return projected, projected_abs2

    def _project_block(class_index, mean_for_proj, rots_b):
        if use_relion_projector:
            projector_kwargs = {}
            if current_size is not None:
                projector_kwargs["projector_output_size"] = int(current_size)
            if projector_returns_compact:
                projector_kwargs["pixel_indices"] = projector_compact_indices_np
            if coarse_texture_interp:
                if projector_returns_compact:
                    proj_half_b, proj_abs2_half_b = (
                        _project_relion_compact_score_rows(
                            class_index,
                            rots_b,
                            return_abs2=True,
                        )
                    )
                else:
                    proj_half_b, proj_abs2_half_b = (
                        _compute_relion_projector_projections_block(
                            relion_projector_half[class_index],
                            rots_b,
                            image_shape,
                            r_max=int(relion_projector_r_max),
                            padding_factor=int(projection_padding_factor),
                            centered_rows=True,
                            dense_scale=True,
                            relion_texture_interp=True,
                            **projector_kwargs,
                        )
                    )
            else:
                proj_half_b = _project_relion_projector_manual(
                    relion_projector_half[class_index],
                    rots_b,
                    image_shape,
                    int(relion_projector_r_max),
                    int(projection_padding_factor),
                    projector_kwargs.get("projector_output_size"),
                    coarse_floorf_quirk,
                )
                proj_half_b = proj_half_b * _dense_projection_scale(image_shape)
                proj_abs2_half_b = jnp.abs(proj_half_b) ** 2
        else:
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
        return proj_half_b, proj_abs2_half_b

    coarse_selector_execution = {
        "wrapper": None,
        "target": None,
        "fused_calls": 0,
        "actual_rows": 0,
        "multistream_calls": 0,
        "native_atomic_selected_calls": 0,
        "prehalf_selected_calls": 0,
    }

    def _score_coarse_fused_full_diff2(
        class_index,
        rots_b,
        *,
        actual_image_count,
    ):
        """Invoke the mature fused scorer and record observed execution."""

        from recovar import cuda_backproject

        if not coarse_fused_projector_enabled:
            raise RuntimeError("fused coarse scorer was not enabled")
        if coarse_gaussian_projector_full_by_class is None:
            raise RuntimeError("fused coarse scorer is missing its projector")
        if coarse_gaussian_translation_angles is None:
            raise RuntimeError("fused coarse scorer is missing translation angles")
        if coarse_gaussian_square_layout is None:
            raise RuntimeError("fused coarse scorer is missing its square layout")
        # Stable shapes append physical-only zero-weight rows after the exact
        # logical lookup.  The fused ABI is keyed by logical current_size, so
        # it must consume only that unchanged prefix.
        fused_full_to_compact = _coarse_gaussian_fused_logical_lookup(
            coarse_gaussian_full_to_compact,
            coarse_gaussian_square_layout,
            current_size=score_size,
        )
        actual_image_count = operator.index(actual_image_count)
        if actual_image_count <= 0:
            raise ValueError("fused coarse scorer requires actual image rows")
        coarse_projector = (
            cuda_backproject.relion_coarse_diff2_projector_multistream_f32
            if coarse_multistream_enabled
            else cuda_backproject.relion_coarse_diff2_projector_f32
        )
        coarse_projector_kwargs = {}
        if coarse_multistream_enabled:
            coarse_projector_kwargs["actual_batch_size"] = jnp.asarray(
                actual_image_count,
                dtype=jnp.int32,
            )
        selected_wrapper = getattr(coarse_projector, "__name__", None)
        selected_target = (
            cuda_backproject._TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32
            if coarse_multistream_enabled
            else cuda_backproject._TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32
        )
        previous_wrapper = coarse_selector_execution["wrapper"]
        previous_target = coarse_selector_execution["target"]
        if previous_wrapper is None:
            coarse_selector_execution["wrapper"] = selected_wrapper
            coarse_selector_execution["target"] = selected_target
        elif previous_wrapper != selected_wrapper or previous_target != selected_target:
            raise RuntimeError(
                "coarse selector changed wrapper/target within one significance pass"
            )
        coarse_selector_execution["fused_calls"] += 1
        coarse_selector_execution["actual_rows"] += actual_image_count
        if coarse_multistream_enabled:
            coarse_selector_execution["multistream_calls"] += 1
        if coarse_native_atomic_reduction_enabled:
            coarse_selector_execution["native_atomic_selected_calls"] += 1
        if coarse_prehalf_weight_enabled:
            coarse_selector_execution["prehalf_selected_calls"] += 1
        coarse_projector_kwargs["prehalf_weight"] = coarse_prehalf_weight_enabled
        return coarse_projector(
            coarse_gaussian_projector_full_by_class[class_index],
            jnp.asarray(rots_b, dtype=jnp.float32),
            jnp.asarray(coarse_gaussian_unshifted_corrected, dtype=jnp.complex64),
            coarse_gaussian_translation_angles,
            jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
            jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
            fused_full_to_compact,
            current_size=score_size,
            physical_image_size=int(image_shape[0]),
            model_max_r=int(relion_projector_r_max),
            canonical_reduction=coarse_canonical_reduction_enabled,
            single_lane_canonical=coarse_single_lane_canonical_enabled,
            **coarse_projector_kwargs,
        )

    def _project_coarse_gemm_rows(class_index, rots_b, *, return_abs2: bool):
        return _project_relion_compact_score_rows(
            class_index,
            rots_b,
            return_abs2=return_abs2,
        )

    def _project_coarse_gemm_block_once(class_index, mean_for_proj, rots_b):
        del mean_for_proj
        return _project_coarse_gemm_rows(
            class_index,
            rots_b,
            return_abs2=True,
        )

    coarse_gaussian_gemm_projection_cache = None
    if coarse_gaussian_gemm_projection_cache_plan is not None:

        def _project_coarse_gemm_cache_build_block(table_index, start, stop):
            projected_reference, projected_reference_abs2 = (
                _project_coarse_gemm_rows(
                    table_index,
                    rotations[start:stop],
                    return_abs2=False,
                )
            )
            if projected_reference_abs2 is not None:
                raise RuntimeError(
                    "coarse GEMM cache build unexpectedly materialized abs2",
                )
            return projected_reference

        coarse_gaussian_gemm_projection_cache = (
            _build_coarse_gaussian_gemm_projection_cache(
                coarse_gaussian_gemm_projection_cache_plan,
                _project_coarse_gemm_cache_build_block,
            )
        )
        logger.warning(
            "Opt-in call-scoped coarse GEMM C64 projection cache built: "
            "shape=%s chunks=%d conservative_peak_bytes=%d budget_bytes=%d",
            coarse_gaussian_gemm_projection_cache_plan.cache_shape,
            coarse_gaussian_gemm_projection_cache_plan.chunk_count_per_table,
            coarse_gaussian_gemm_projection_cache_plan.predicted_peak_bytes,
            coarse_gaussian_gemm_projection_cache_plan.budget_bytes,
        )

    coarse_gemm_diagnostic_positions = None
    coarse_gemm_direct_capture = {"scores": None}
    coarse_gemm_stream_capture_control = {"enabled": False}
    coarse_gaussian_gemm_hybrid_batch_result = None

    def _score_block(
        class_index,
        mean_for_proj,
        rots_b,
        shifted_data,
        batch_norm,
        ctf2_data,
        batch_size,
        *,
        rotation_start,
    ):
        if coarse_gaussian_gemm_hybrid_batch_result is not None:
            if int(class_index) != 0:
                raise RuntimeError("certified coarse GEMM hybrid is restricted to K=1")
            start = int(rotation_start)
            requested_rows = int(rots_b.shape[0])
            dense_scores = coarse_gaussian_gemm_hybrid_batch_result.scores
            if dense_scores is None:
                raise RuntimeError(
                    "compact hybrid scores must bypass dense rotation-block replay",
                )
            if (
                start < 0
                or start >= int(dense_scores.shape[1])
                or requested_rows <= 0
            ):
                raise IndexError("hybrid score block is outside the dense source table")
            stop = min(start + requested_rows, int(dense_scores.shape[1]))
            scores = dense_scores[:, start:stop, :]
            padding_rows = requested_rows - int(scores.shape[1])
            if padding_rows:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, padding_rows), (0, 0)),
                    constant_values=-jnp.inf,
                )
            return scores
        if coarse_gaussian_gemm_macro_enabled:
            coarse_gemm_direct_capture["scores"] = None
            project_block_once = _project_coarse_gemm_block_once
            if coarse_gaussian_gemm_projection_cache is not None:
                project_block_once = partial(
                    _project_coarse_gaussian_gemm_projection_cache_block_once,
                    coarse_gaussian_gemm_projection_cache,
                    rotation_start=int(rotation_start),
                )
            score_result = _score_relion_coarse_gaussian_gemm_macro(
                project_block_once,
                class_index,
                mean_for_proj,
                rots_b,
                jnp.asarray(
                    coarse_gaussian_shifted_corrected,
                    dtype=jnp.complex64,
                ),
                jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                actual_batch_size,
                image_shape=image_shape,
                volume_shape=volume_shape,
                return_projected=bool(
                    coarse_gemm_diagnostic_positions is not None
                    or coarse_gemm_stream_capture_control["enabled"]
                ),
            )
            if (
                coarse_gemm_diagnostic_positions is None
                and not coarse_gemm_stream_capture_control["enabled"]
            ):
                return score_result

            from recovar import cuda_backproject

            macro_scores, projected_reference, _projected_reference_abs2 = score_result
            active = jnp.arange(batch_size, dtype=jnp.int32) < jnp.asarray(
                actual_batch_size,
                dtype=jnp.int32,
            )
            direct_shifted = jnp.where(
                active[:, None, None],
                jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                jnp.zeros((), dtype=jnp.complex64),
            )
            direct_weight = jnp.where(
                active[:, None],
                jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32),
            )
            direct_initial = jnp.where(
                active,
                jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32),
            )
            direct_diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
                jnp.asarray(projected_reference, dtype=jnp.complex64),
                direct_shifted,
                direct_weight,
                direct_initial,
                coarse_gaussian_full_to_compact,
            )
            coarse_gemm_direct_capture["scores"] = jnp.where(
                active[:, None, None],
                -direct_diff2,
                jnp.zeros((), dtype=jnp.float32),
            )
            return macro_scores
        if (
            coarse_gaussian_score_backend
            in _COARSE_GAUSSIAN_FUSED_SCORE_BACKENDS
        ):
            return -_score_coarse_fused_full_diff2(
                class_index,
                rots_b,
                actual_image_count=actual_batch_size,
            )

        if coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.NATIVE_TEXTURE:
            from recovar import cuda_backproject

            diff2 = cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
                coarse_gaussian_projector_full,
                jnp.asarray(rots_b, dtype=jnp.float32),
                coarse_gaussian_unshifted_corrected,
                coarse_gaussian_translation_angles,
                coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2,
                coarse_gaussian_full_to_compact,
                int(image_shape[0]) if current_size is None else int(current_size),
                int(projection_padding_factor),
                int(relion_projector_r_max),
            )
            return -diff2
        proj_half_b, proj_abs2_half_b = _project_block(class_index, mean_for_proj, rots_b)
        if coarse_gaussian_ffi_enabled:
            from recovar import cuda_backproject

            proj_score = (
                proj_half_b
                if projector_returns_compact
                else proj_half_b[:, coarse_gaussian_score_indices]
            )
            coarse_complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
            proj_score = jnp.asarray(proj_score, dtype=coarse_complex_dtype)
            coarse_diff2 = (
                cuda_backproject.relion_coarse_diff2_rectangular_f64
                if use_float64_scoring
                else cuda_backproject.relion_coarse_diff2_rectangular_f32
            )
            diff2 = coarse_diff2(
                proj_score,
                coarse_gaussian_shifted_corrected,
                coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2,
                coarse_gaussian_full_to_compact,
            )
            return -diff2
        if use_window:
            if projector_returns_compact:
                proj_w = proj_half_b
                proj_abs2_w = proj_abs2_half_b
            else:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
            if not use_float64_scoring:
                proj_w = proj_w.astype(jnp.complex64)
                proj_abs2_w = proj_abs2_w.astype(jnp.float32)
            if score_mode == "normalized_cc":
                return _e_step_block_scores_windowed_normalized_cc(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            return _e_step_block_scores_windowed(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_w * half_weights_windowed,
                proj_abs2_w * half_weights_windowed,
                half_weights_windowed,
                batch_size,
                n_trans,
                n_windowed,
                image_shape,
                volume_shape,
            )
        if not use_float64_scoring:
            proj_half_b = proj_half_b.astype(jnp.complex64)
            proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
        if score_mode == "normalized_cc":
            return _e_step_block_scores_normalized_cc(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_half_b * half_weights,
                proj_abs2_half_b * half_weights,
                batch_size,
                n_trans,
                image_shape,
                volume_shape,
            )
        return _e_step_block_scores(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_half_b * half_weights,
            proj_abs2_half_b * half_weights,
            half_weights,
            batch_size,
            n_trans,
            image_shape,
            volume_shape,
        )

    def _add_priors(scores, class_index, r0, r1, batch_translation_log_prior):
        if score_mode == "normalized_cc":
            return scores
        scores = scores + jnp.asarray(class_log_priors_np[class_index], dtype=scores.real.dtype)
        if rotation_log_prior_padded is not None:
            scores = scores + jnp.asarray(rotation_log_prior_padded[class_index, r0:r1])[None, :, None]
        if batch_translation_log_prior is not None:
            if translation_log_prior.ndim == 1:
                scores = scores + batch_translation_log_prior[None, None, :]
            else:
                scores = scores + batch_translation_log_prior[:, None, :]
        return scores

    sig_rot_any = np.zeros((n_classes, n_rot), dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    cutoff_count_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    class_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [[None] * n_images for _ in range(n_classes)] if collect_significance else None
    normalization_log_z = np.empty(n_images, dtype=np.float64)
    normalization_log_evidence = np.empty(n_images, dtype=np.float64)
    log_evidence = np.empty(n_images, dtype=score_real_dtype)
    best_log_score = np.empty(n_images, dtype=score_real_dtype)
    max_posterior = np.empty(n_images, dtype=score_real_dtype)
    relion_f32_sum_weight = (
        np.empty(n_images, dtype=np.float32)
        if relion_f32_coarse_support_enabled and collect_significance
        else None
    )
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_best_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_best else None
    )
    class_second_best_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_second else None
    )
    # Diagnostic-only native scores before the large, class-common image
    # normalization offset.  The offset is useful for absolute log evidence,
    # but adding it before a float32 cast can erase class and pose margins.
    class_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_best else None
    )
    class_second_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_second else None
    )
    class_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_best else None
    )
    class_second_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_second else None
    )
    tree_rescore_examined = 0
    tree_rescore_ambiguous = 0
    tree_rescore_winner_changes = 0
    tree_rescore_exact_ties = 0
    coarse_gaussian_gemm_diagnostic_paths = []
    coarse_gaussian_gemm_diagnostic_found_targets = set()
    coarse_gaussian_gemm_diagnostic_target_counts = {}
    coarse_gaussian_gemm_stream_paths = []
    coarse_gaussian_gemm_stream_original_indices = []
    coarse_gaussian_gemm_hybrid_batch_count = 0
    coarse_partition_mixed_batch_count = 0
    coarse_cuda_posterior_batch_count = 0
    coarse_cuda_posterior_group_count = 0
    coarse_cuda_posterior_image_count = 0
    coarse_partition_actual_group_sizes = []
    coarse_partition_physical_group_sizes = []
    coarse_gaussian_gemm_hybrid_selected_batch_count = 0
    coarse_gaussian_gemm_hybrid_static_dense_batch_count = 0
    coarse_gaussian_gemm_hybrid_fallback_batch_count = 0
    coarse_gaussian_gemm_hybrid_selected_image_count = 0
    coarse_gaussian_gemm_hybrid_static_dense_image_count = 0
    coarse_gaussian_gemm_hybrid_fallback_image_count = 0
    coarse_gaussian_gemm_hybrid_fused_full_batch_count = 0
    coarse_gaussian_gemm_hybrid_fused_full_image_count = 0
    coarse_gaussian_gemm_hybrid_rectangular_full_batch_count = 0
    coarse_gaussian_gemm_hybrid_rectangular_full_image_count = 0
    coarse_full_kernel_calls = {}
    coarse_full_kernel_images = {}
    coarse_gaussian_gemm_hybrid_selected_block_count = 0
    coarse_gaussian_gemm_hybrid_max_blocks_per_image = 0
    coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates = 0
    coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates = 0
    coarse_gaussian_gemm_hybrid_fallback_reasons = {}
    coarse_gaussian_gemm_hybrid_score_representation_batch_counts = {}
    coarse_gaussian_gemm_hybrid_actual_image_batch_sizes = []
    coarse_gaussian_gemm_hybrid_physical_image_batch_sizes = []
    # Every batch in this significance call shares the exact projection
    # cache, topology, rotation/translation geometry, and selected-block
    # capacity.  Once one certified selection overflows, later batches can
    # safely skip that certificate and use the mature exact rectangular
    # scorer.  Keep this latch local so no posterior-width assumption crosses
    # an iteration, dataset, or independently constructed score call.
    coarse_gaussian_gemm_hybrid_overflow_latched = False
    coarse_gaussian_gemm_hybrid_overflow_latch_activation_count = 0
    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count = 0
    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count = 0
    generic_coarse_operand_assembly_count = 0
    exact_coarse_operand_assembly_count = 0
    generic_score_preprocess_count = 0
    exact_source_preprocess_count = 0

    start_idx = 0
    image_indices = np.arange(n_images)
    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        actual_batch_size = len(indices)
        end_idx = start_idx + actual_batch_size
        coarse_gaussian_gemm_hybrid_batch_result = None
        (
            relion_cuda_preprocess,
            integer_pre_shifts,
            batch_corr_np,
            batch_scale_np,
            relion_preprocess_kwargs,
        ) = prepare_batch_preprocess_operands(
            experiment_dataset,
            batch_data,
            indices,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            dtype=score_real_dtype,
        )
        if pad_final_image_batch and actual_batch_size < int(image_batch_size):
            (
                batch_data,
                ctf_params,
                integer_pre_shifts,
                batch_corr_np,
                batch_scale_np,
                relion_preprocess_kwargs,
            ) = _pad_significance_preprocess_inputs(
                batch_data,
                ctf_params,
                integer_pre_shifts,
                batch_corr_np,
                batch_scale_np,
                relion_preprocess_kwargs,
                target_size=int(image_batch_size),
            )
        batch_size = int(np.asarray(batch_data).shape[0])
        if coarse_gaussian_gemm_hybrid_requested:
            coarse_gaussian_gemm_hybrid_actual_image_batch_sizes.append(
                int(actual_batch_size),
            )
            coarse_gaussian_gemm_hybrid_physical_image_batch_sizes.append(
                int(batch_size),
            )
        local_indices_np = np.asarray(indices, dtype=np.int64)
        coarse_runtime_prefix_dump_positions = np.empty((0,), dtype=np.int64)
        if coarse_runtime_prefix_dump_dir is not None:
            coarse_runtime_prefix_original_indices = original_image_indices(
                experiment_dataset,
                local_indices_np,
            )
            coarse_runtime_prefix_dump_positions = np.flatnonzero(
                np.isin(
                    coarse_runtime_prefix_original_indices,
                    np.fromiter(
                        coarse_runtime_prefix_dump_targets,
                        dtype=np.int64,
                    ),
                ),
            ).astype(np.int64)
        batch_original_indices_np = None
        if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
            batch_original_indices_np = original_image_indices(
                experiment_dataset,
                local_indices_np,
            )
        coarse_gemm_diagnostic_positions = None
        coarse_gemm_diagnostic_original_indices = None
        if coarse_gaussian_gemm_diagnostic_targets is not None:
            original_indices_np = original_image_indices(
                experiment_dataset,
                local_indices_np,
            )
            positions = np.flatnonzero(
                np.isin(
                    original_indices_np,
                    np.fromiter(
                        coarse_gaussian_gemm_diagnostic_targets,
                        dtype=np.int64,
                    ),
                )
            ).astype(np.int64)
            if positions.size:
                coarse_gemm_diagnostic_positions = positions
                coarse_gemm_diagnostic_original_indices = original_indices_np[positions]
        coarse_gemm_stream_capture_control["enabled"] = bool(
            coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        coarse_gemm_stream_state = (
            initialize_coarse_gemm_streaming_state(
                batch_size,
                coarse_gaussian_gemm_stream_topk,
                score_dtype=jnp.float32,
            )
            if coarse_gemm_stream_capture_control["enabled"]
            else None
        )
        coarse_gemm_pre_prior_stream_state = (
            initialize_coarse_gemm_streaming_state(
                batch_size,
                coarse_gaussian_gemm_stream_topk,
                score_dtype=jnp.float32,
            )
            if coarse_gemm_stream_capture_control["enabled"]
            else None
        )
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied and not relion_cuda_preprocess:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
        batch_data = jnp.asarray(batch_data)
        if translation_log_prior is None:
            batch_translation_log_prior = None
        elif translation_log_prior.ndim == 1:
            batch_translation_log_prior = jnp.asarray(translation_log_prior)
        else:
            batch_translation_log_prior_np = np.asarray(translation_log_prior[start_idx:end_idx])
            if batch_size > actual_batch_size:
                batch_translation_log_prior_np = _repeat_pad_batch_axis(
                    batch_translation_log_prior_np,
                    batch_size,
                )
            batch_translation_log_prior = jnp.asarray(batch_translation_log_prior_np)

        if exact_compact_preprocess_enabled:
            shifted_half = None
            batch_norm = None
            ctf2_over_nv_half = None
            coarse_gaussian_unshifted_score_weighted = None
        elif score_mode == "normalized_cc":
            cc_window_indices = window_indices if use_window else None
            score_complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
            score_real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
            cc_preprocess_result = _preprocess_batch_firstiter_cc(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                window_indices=cc_window_indices,
                score_complex_dtype=score_complex_dtype,
                score_real_dtype=score_real_dtype,
                norm_real_dtype=jnp.float64,
                relion_preprocess_kwargs=relion_preprocess_kwargs,
                return_unshifted_score_weighted=tree_rescore_enabled,
            )
            if tree_rescore_enabled:
                (
                    shifted_half,
                    batch_norm,
                    ctf2_half_score,
                    ctf2_over_nv_half,
                    tree_rescore_unshifted_half,
                ) = cc_preprocess_result
            else:
                shifted_half, batch_norm, ctf2_half_score, ctf2_over_nv_half = (
                    cc_preprocess_result
                )
        elif use_relion_numpy_preprocess and not relion_cuda_preprocess:
            preprocess_result = _preprocess_batch_relion_numpy(
                batch_data,
                ctf_params,
                batch_size,
                return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled,
            )
            if coarse_gaussian_sincosf_enabled:
                (
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    coarse_gaussian_unshifted_score_weighted,
                ) = preprocess_result
            else:
                shifted_half, batch_norm, ctf2_over_nv_half = preprocess_result
        else:
            if exact_coarse_assembly_profile_enabled:
                generic_score_preprocess_count += 1
            preprocess_result = _preprocess_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                relion_preprocess_kwargs=relion_preprocess_kwargs,
                return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled,
            )
            if coarse_gaussian_sincosf_enabled:
                (
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    coarse_gaussian_unshifted_score_weighted,
                ) = preprocess_result
            else:
                shifted_half, batch_norm, ctf2_over_nv_half = preprocess_result
        batch_scale = jnp.asarray(batch_scale_np)
        if image_corrections is not None and not exact_compact_preprocess_enabled:
            batch_corr = jnp.asarray(batch_corr_np)
            applied_corr = batch_scale if relion_cuda_preprocess else batch_corr
            corr_expanded = jnp.repeat(applied_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half * applied_corr[:, None]
                )
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = (
                    coarse_gaussian_unshifted_score_weighted
                    * applied_corr[:, None]
                )
            # ``image_corrections`` carries ``(avg_norm/normcorr) * scale``;
            # ``scale_corrections`` carries ``scale``. The image-only
            # ``|F_img|^2`` term must be weighted by ``(avg_norm/normcorr)^2``
            # alone — divide ``batch_corr`` by ``batch_scale`` to isolate it.
            # Otherwise ``batch_norm`` picks up an extra ``scale^2`` that is
            # already accounted for on the reference side via
            # ``ctf2_over_nv_half *= batch_scale^2`` below, double-counting
            # ``scale^2`` in the Wiener score offset. See
            # ``em_engine._relion_image_correction_factors`` and
            # ``ml_optimiser.cpp:6240,7298,8516``.
            if not relion_cuda_preprocess:
                norm_corr = batch_corr / batch_scale
                batch_norm = batch_norm * (norm_corr**2)[:, None]
        if scale_corrections is not None and not exact_compact_preprocess_enabled:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if score_mode == "normalized_cc":
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]
        if (
            image_pre_shifts is not None
            and not real_space_pre_shift_applied
            and not exact_compact_preprocess_enabled
        ):
            batch_shifts_np = np.asarray(image_pre_shifts)[np.asarray(indices)]
            if batch_size > actual_batch_size:
                batch_shifts_np = _repeat_pad_batch_axis(batch_shifts_np, batch_size)
            batch_shifts = jnp.asarray(batch_shifts_np)
            shifted_half = shifted_half * tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans, dtype=batch_shifts.dtype)
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half
                    * tiled_half_image_phase_factors(
                        image_shape,
                        batch_shifts,
                        1,
                        dtype=batch_shifts.dtype,
                    )
                )
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = (
                    coarse_gaussian_unshifted_score_weighted
                    * tiled_half_image_phase_factors(
                        image_shape,
                        batch_shifts,
                        1,
                        dtype=batch_shifts.dtype,
                    )
                )
        if score_mode == "normalized_cc":
            inv_xi2 = 1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))
            score_weight_half = ctf2_half_score * inv_xi2
            shifted_half = shifted_half * jnp.repeat(inv_xi2, n_trans, axis=0)
            if tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half * inv_xi2
                )
        elif not exact_compact_preprocess_enabled:
            score_weight_half = ctf2_over_nv_half
        else:
            score_weight_half = None
        if (
            half_spectrum_scoring
            and score_mode != "normalized_cc"
            and not exact_compact_preprocess_enabled
        ):
            from recovar.em.helpers.half_spectrum import make_shell_indices_half as _mshi

            dc_mask = _mshi(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            score_weight_half = jnp.where(dc_mask[None, :], 0.0, score_weight_half)
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = jnp.where(
                    dc_mask[None, :],
                    0.0,
                    coarse_gaussian_unshifted_score_weighted,
                )
        if exact_compact_preprocess_enabled:
            shifted_data = None
            ctf2_data = None
        elif use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = score_weight_half[:, window_indices]
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_half[
                    :, window_indices
                ]
        else:
            shifted_data = shifted_half
            ctf2_data = score_weight_half
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_half
        if exact_compact_preprocess_enabled:
            pass
        elif use_float64_scoring:
            shifted_data = shifted_data.astype(jnp.complex128)
            ctf2_data = ctf2_data.astype(jnp.float64)
        else:
            shifted_data = shifted_data.astype(jnp.complex64)
            ctf2_data = ctf2_data.astype(jnp.float32)
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_data.astype(
                    jnp.complex64
                )

        if score_mode == "normalized_cc" and tree_rescore_enabled:
            if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
                raise ValueError(
                    f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                    "RELION CUDA image preprocessing",
                )
            exact_cc_preprocess_kwargs = dict(relion_preprocess_kwargs)
            exact_cc_preprocess_kwargs["relion_fft_per_image"] = True
            exact_cc_processed = process_half_image(
                experiment_dataset,
                batch_data,
                score_with_masked_images,
                relion_preprocess_kwargs=exact_cc_preprocess_kwargs,
            )
            exact_cc_inv_xi2 = _relion_cc_inverse_power_from_processed(
                exact_cc_processed,
                window_indices if use_window else None,
            )
            exact_cc_ctf_rfloat = _relion_exact_ctf_half_from_source_star(
                experiment_dataset,
                indices,
                image_shape,
            )
            batch_scale_f32 = jnp.asarray(batch_scale_np, dtype=jnp.float32)
            exact_cc_pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                batch_scale_f32[:, None],
                exact_cc_ctf_rfloat,
            )
            exact_cc_unshifted_corrected = jnp.asarray(
                exact_cc_processed * exact_cc_pixel_correction,
                dtype=jnp.complex64,
            )
            if image_pre_shifts is not None and not real_space_pre_shift_applied:
                exact_cc_unshifted_corrected = (
                    exact_cc_unshifted_corrected
                    * tiled_half_image_phase_factors(image_shape, batch_shifts, 1)
                )
            exact_cc_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
                exact_cc_inv_xi2,
                exact_cc_ctf_rfloat,
                batch_scale_f32[:, None] if scale_corrections is not None else None,
            )
            if use_window:
                tree_rescore_unshifted_data = exact_cc_unshifted_corrected[
                    :, window_indices
                ]
                tree_rescore_corr_img_data = exact_cc_corr_img[:, window_indices]
            else:
                tree_rescore_unshifted_data = exact_cc_unshifted_corrected
                tree_rescore_corr_img_data = exact_cc_corr_img

        if coarse_gaussian_ffi_enabled:
            coarse_preprocess_kwargs = relion_preprocess_kwargs
            if exact_coarse_operands_enabled:
                if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
                    raise ValueError(
                        f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires "
                        "RELION CUDA image preprocessing",
                    )
                if exact_coarse_assembly_profile_enabled:
                    exact_source_preprocess_count += 1
                processed_direct = _process_relion_exact_coarse_half_image(
                    experiment_dataset,
                    batch_data,
                    score_with_masked_images,
                    relion_preprocess_kwargs=relion_preprocess_kwargs,
                )
            else:
                processed_direct = process_half_image(
                    experiment_dataset,
                    batch_data,
                    score_with_masked_images,
                    relion_preprocess_kwargs=coarse_preprocess_kwargs,
                )
            processed_for_powerclass = processed_direct
            if image_corrections is not None and not relion_cuda_preprocess:
                image_only_correction = batch_corr / batch_scale
                processed_for_powerclass = (
                    processed_for_powerclass * image_only_correction[:, None]
                )
            # Reuse the exact operands of RECOVAR's accepted Gaussian score
            # boundary. Reprocessing and translating a second image copy here
            # changed the operands as well as the reduction. Algebraically,
            # ``shifted_half / score_weight_half`` is the shifted image divided
            # by CTF, and RELION's square-difference weight is
            # ``score_weight_half * half_weights``.
            if not exact_coarse_skip_generic_operands_enabled:
                if exact_coarse_assembly_profile_enabled:
                    generic_coarse_operand_assembly_count += 1
                if coarse_gaussian_sincosf_enabled:
                    (
                        coarse_gaussian_shifted_corrected,
                        coarse_gaussian_pixel_weight,
                        coarse_gaussian_unshifted_corrected,
                    ) = _relion_coarse_gaussian_square_operands_sincosf(
                        coarse_gaussian_unshifted_score_weighted,
                        score_weight_half,
                        half_weights,
                        coarse_gaussian_score_indices,
                        coarse_gaussian_score_active_mask,
                        translations,
                        image_shape,
                        translation_phase_source=translations_source,
                        return_unshifted=True,
                    )
                else:
                    (
                        coarse_gaussian_shifted_corrected,
                        coarse_gaussian_pixel_weight,
                    ) = _relion_coarse_gaussian_square_operands(
                        shifted_half,
                        score_weight_half,
                        half_weights,
                        coarse_gaussian_score_indices,
                        coarse_gaussian_score_active_mask,
                        batch_size=batch_size,
                        n_trans=n_trans,
                    )
            if exact_coarse_operands_enabled:
                if exact_coarse_assembly_profile_enabled:
                    exact_coarse_operand_assembly_count += 1
                exact_operands = _assemble_relion_exact_coarse_gaussian_operands(
                    experiment_dataset,
                    processed_direct,
                    indices,
                    use_float64_scoring=use_float64_scoring,
                    batch_scale_np=batch_scale_np,
                    actual_batch_size=actual_batch_size,
                    batch_size=batch_size,
                    score_indices=coarse_gaussian_score_indices,
                    score_indices_np=coarse_gaussian_score_indices_np,
                    score_active_mask=coarse_gaussian_score_active_mask,
                    translations_source=translations_source,
                    image_shape=image_shape,
                    noise_variance_half=noise_variance_half,
                    scale_corrections_enabled=scale_corrections is not None,
                    half_weights=half_weights,
                    powerclass=coarse_gaussian_powerclass,
                    current_size=(
                        coarse_gaussian_square_layout.physical_current_size
                        if stable_fourier_window_shapes
                        else current_size
                    ),
                    runtime_current_size=(
                        jnp.asarray(score_size, dtype=jnp.int32) if stable_fourier_window_shapes else None
                    ),
                )
                coarse_gaussian_shifted_corrected = exact_operands.shifted_corrected
                coarse_gaussian_pixel_weight = exact_operands.pixel_weight
                coarse_gaussian_unshifted_corrected = exact_operands.unshifted_corrected
                coarse_gaussian_initial_diff2 = exact_operands.initial_diff2
                coarse_gaussian_translation_angles = exact_operands.translation_angles
            else:
                coarse_gaussian_initial_diff2 = coarse_gaussian_powerclass(
                    processed_for_powerclass,
                    image_shape=image_shape,
                    current_size=(
                        coarse_gaussian_square_layout.physical_current_size
                        if stable_fourier_window_shapes
                        else current_size
                    ),
                    runtime_current_size=(
                        jnp.asarray(score_size, dtype=jnp.int32)
                        if stable_fourier_window_shapes
                        else None
                    ),
                )

        # Identify per-batch dump target rows so we can record raw scores
        # (pre-prior) for each target image inside the per-class block loop.
        # This enables direct diff against RELION's exp_Mweight_diff2
        # without needing the full (batch, n_classes, n_rot*n_trans) cache.
        debug_dump_enabled = collect_significance and _significance_debug_dump_matches(
            current_size=current_size,
            debug_iteration=debug_iteration,
        )
        dump_target_local_positions = None
        if debug_dump_enabled:
            _dump_targets = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if _dump_targets:
                _local_for_dump = np.asarray(indices, dtype=np.int64)
                _orig = original_image_indices(experiment_dataset, _local_for_dump)
                _positions = np.flatnonzero(np.isin(_orig, np.fromiter(_dump_targets, dtype=np.int64)))
                if _positions.size:
                    dump_target_local_positions = _positions.astype(np.int64)
        # Per-class collectors for raw (pre-prior) score blocks at target rows.
        # Shape after concat per class: (n_targets, n_rot, n_trans)
        dump_target_pre_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )
        dump_target_with_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )
        passive_score_dump = bool(
            dump_target_local_positions is not None
            and os.environ.get(_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV) == "1"
        )
        passive_raw_score_blocks_per_class = (
            [[] for _ in range(n_classes)] if passive_score_dump else None
        )
        coarse_gemm_macro_pre_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_direct_pre_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_macro_with_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_direct_with_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        if passive_score_dump:
            # Do not materialize score blocks while production support is
            # being computed. Near an atomic cutoff that observation can
            # perturb the execution under diagnosis. Retain device buffers
            # and write them only after cached-score support selection.
            dump_target_pre_prior_blocks_per_class = None
            dump_target_with_prior_blocks_per_class = None

        if coarse_gaussian_gemm_hybrid_requested:
            if coarse_gaussian_gemm_compact_posterior_requested and debug_dump_enabled:
                raise ValueError(
                    f"{_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1 does not "
                    "support raw significance score dumps",
                )
            if dump_target_local_positions is not None:
                raise ValueError(
                    f"{_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1 does not support "
                    "raw significance score dumps",
                )
            if (
                coarse_gaussian_gemm_projection_cache is None
                or coarse_gaussian_gemm_projection_cache_plan is None
                or coarse_gaussian_gemm_certificate_topology is None
            ):
                raise RuntimeError(
                    "certified coarse GEMM hybrid is missing its projection "
                    "cache, cache plan, or sealed topology",
                )
            force_static_dense_after_overflow = bool(
                coarse_gaussian_gemm_hybrid_overflow_latched
            )
            coarse_partition_plan = None
            coarse_partition_groups = None
            if (
                coarse_row_partition_requested
                and not force_static_dense_after_overflow
                and n_rot > coarse_gaussian_gemm_hybrid_capacity * SOURCE_ROTATION_BLOCK_SIZE
            ):
                from recovar.em.scoring.coarse_partition import compute_partitioned_coarse_batch

                partition_translation_prior = batch_translation_log_prior
                if partition_translation_prior is not None and partition_translation_prior.ndim == 1:
                    partition_translation_prior = jnp.broadcast_to(partition_translation_prior, (batch_size, n_trans))
                coarse_partition_plan, coarse_partition_groups = compute_partitioned_coarse_batch(
                    coarse_gaussian_gemm_projection_cache,
                    jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                    jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                    jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                    topology=coarse_gaussian_gemm_certificate_topology,
                    actual_image_count=actual_batch_size,
                    class_log_prior=class_log_priors_np[0],
                    rotation_log_prior=(None if rotation_log_prior_padded is None else rotation_log_prior_padded[0, :n_rot]),
                    translation_log_prior=partition_translation_prior,
                    certificate_chunk_rows=coarse_gaussian_gemm_projection_cache_plan.chunk_rows,
                    block_capacity=coarse_gaussian_gemm_hybrid_capacity,
                    logical_full_pixel_count=(coarse_gaussian_square_layout.logical_square_count if stable_fourier_window_shapes else None),
                    real_cross=coarse_gaussian_gemm_real_cross_requested,
                )
            else:
                full_dense_diff2_fn = None
                if coarse_gaussian_fused_full_fallback_armed:
                    full_dense_diff2_fn = partial(
                        _score_coarse_fused_full_diff2,
                        0,
                        rotations,
                        actual_image_count=actual_batch_size,
                    )
                coarse_gaussian_gemm_hybrid_batch_result = (
                    _compute_coarse_gaussian_gemm_hybrid_batch(
                        coarse_gaussian_gemm_projection_cache,
                        jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                        jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                        jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                        topology=coarse_gaussian_gemm_certificate_topology,
                        actual_image_count=actual_batch_size,
                        class_log_prior=class_log_priors_np[0],
                        rotation_log_prior=(
                            None
                            if rotation_log_prior_padded is None
                            else rotation_log_prior_padded[0, :n_rot]
                        ),
                        translation_log_prior=batch_translation_log_prior,
                        certificate_chunk_rows=(
                            coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                        ),
                        block_capacity=coarse_gaussian_gemm_hybrid_capacity,
                        compact_posterior=(
                            coarse_gaussian_gemm_compact_posterior_requested
                        ),
                        force_static_dense_after_overflow=(
                            force_static_dense_after_overflow
                        ),
                        logical_full_pixel_count=(
                            coarse_gaussian_square_layout.logical_square_count
                            if stable_fourier_window_shapes
                            else None
                        ),
                        capture_selected_diff2=bool(
                            coarse_runtime_prefix_dump_positions.size
                        ),
                        full_dense_diff2_fn=full_dense_diff2_fn,
                        device_transaction=coarse_gaussian_gemm_device_transaction_requested,
                        real_cross=coarse_gaussian_gemm_real_cross_requested,
                    )
                )
            coarse_gaussian_gemm_hybrid_batch_count += 1
            group_results = (
                [(group.result, len(group.image_indices), len(group.result.raw_score_max)) for group in coarse_partition_groups]
                if coarse_partition_groups is not None
                else [(coarse_gaussian_gemm_hybrid_batch_result, actual_batch_size, batch_size)]
            )
            for coarse_gaussian_gemm_hybrid_batch_result, group_actual_size, group_physical_size in group_results:
                if force_static_dense_after_overflow:
                    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count += 1
                    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count += (
                        group_actual_size
                    )
                score_representation = str(
                    coarse_gaussian_gemm_hybrid_batch_result.score_representation,
                )
                coarse_gaussian_gemm_hybrid_score_representation_batch_counts[
                    score_representation
                ] = (
                    coarse_gaussian_gemm_hybrid_score_representation_batch_counts.get(
                        score_representation,
                        0,
                    )
                    + 1
                )
                if coarse_gaussian_gemm_hybrid_batch_result.used_selected_rescore:
                    if (
                        coarse_gaussian_gemm_hybrid_batch_result.full_dense_backend
                        is not None
                    ):
                        raise RuntimeError(
                            "selected hybrid batch unexpectedly reports a full-dense backend",
                        )
                    coarse_gaussian_gemm_hybrid_selected_batch_count += 1
                    coarse_gaussian_gemm_hybrid_selected_image_count += group_actual_size
                    selected_count = coarse_gaussian_gemm_hybrid_batch_result.selected_block_count
                    max_selected = coarse_gaussian_gemm_hybrid_batch_result.max_selected_blocks
                    if selected_count is None or max_selected is None:
                        selected_counts = np.asarray(
                            coarse_gaussian_gemm_hybrid_batch_result.selection.block_count,
                            dtype=np.int32,
                        )[:group_actual_size]
                        selected_count = int(np.sum(selected_counts, dtype=np.int64))
                        max_selected = int(np.max(selected_counts))
                    coarse_gaussian_gemm_hybrid_selected_block_count += selected_count
                    coarse_gaussian_gemm_hybrid_max_blocks_per_image = max(
                        coarse_gaussian_gemm_hybrid_max_blocks_per_image,
                        max_selected,
                    )
                    coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates += (
                        group_physical_size
                        * coarse_gaussian_gemm_hybrid_capacity
                        * SOURCE_ROTATION_BLOCK_SIZE
                        * n_trans
                    )
                    coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates += (
                        group_physical_size * n_rot * n_trans
                    )
                else:
                    full_dense_backend = (
                        coarse_gaussian_gemm_hybrid_batch_result.full_dense_backend
                    )
                    kernel = coarse_gaussian_gemm_hybrid_batch_result.full_dense_kernel
                    expected_family = {
                        "rectangular": "rectangular",
                        "rectangular_runtime": "rectangular",
                        "shared_pretranslated": "rectangular",
                        "fused_projector": "fused_projector",
                    }.get(kernel)
                    if expected_family != full_dense_backend or expected_family is None:
                        raise RuntimeError("full-dense kernel disagrees with executed backend")
                    coarse_full_kernel_calls[kernel] = (
                        coarse_full_kernel_calls.get(kernel, 0) + 1
                    )
                    coarse_full_kernel_images[kernel] = (
                        coarse_full_kernel_images.get(kernel, 0) + group_actual_size
                    )
                    if full_dense_backend == "fused_projector":
                        coarse_gaussian_gemm_hybrid_fused_full_batch_count += 1
                        coarse_gaussian_gemm_hybrid_fused_full_image_count += (
                            group_actual_size
                        )
                    elif full_dense_backend == "rectangular":
                        coarse_gaussian_gemm_hybrid_rectangular_full_batch_count += 1
                        coarse_gaussian_gemm_hybrid_rectangular_full_image_count += (
                            group_actual_size
                        )
                    else:
                        raise RuntimeError(
                            "full-dense hybrid batch is missing its executed backend",
                        )
                if (
                    not coarse_gaussian_gemm_hybrid_batch_result.used_selected_rescore
                    and score_representation == "dense_full_direct_static_capacity"
                ):
                    coarse_gaussian_gemm_hybrid_static_dense_batch_count += 1
                    coarse_gaussian_gemm_hybrid_static_dense_image_count += (
                        group_actual_size
                    )
                elif not coarse_gaussian_gemm_hybrid_batch_result.used_selected_rescore:
                    coarse_gaussian_gemm_hybrid_fallback_batch_count += 1
                    coarse_gaussian_gemm_hybrid_fallback_image_count += group_actual_size
                    fallback_reason = str(
                        coarse_gaussian_gemm_hybrid_batch_result.fallback_reason,
                    )
                    coarse_gaussian_gemm_hybrid_fallback_reasons[fallback_reason] = (
                        coarse_gaussian_gemm_hybrid_fallback_reasons.get(
                            fallback_reason,
                            0,
                        )
                        + 1
                    )
                    if (
                        fallback_reason == "block_capacity_overflow"
                        and not coarse_gaussian_gemm_hybrid_overflow_latched
                    ):
                        coarse_gaussian_gemm_hybrid_overflow_latched = True
                        coarse_gaussian_gemm_hybrid_overflow_latch_activation_count += 1

            if coarse_partition_groups is not None:
                coarse_partition_mixed_batch_count += int(coarse_partition_plan.partitioned)
                coarse_partition_actual_group_sizes.extend(len(group.image_indices) for group in coarse_partition_groups)
                coarse_partition_physical_group_sizes.extend(len(group.result.raw_score_max) for group in coarse_partition_groups)
                # Preserve the ordinary latch only when every actual image
                # exceeded capacity. One outlier must not promote later batches.
                if not coarse_partition_plan.partitioned and coarse_partition_plan.fallback_reason == "block_capacity_overflow":
                    coarse_gaussian_gemm_hybrid_overflow_latched = True
                    coarse_gaussian_gemm_hybrid_overflow_latch_activation_count += 1
            if coarse_partition_groups is not None or coarse_cuda_posterior_requested:
                from recovar.em.scoring.coarse_publication import publish_coarse_rows

                publication_groups = coarse_partition_groups
                if publication_groups is None:
                    from recovar.em.scoring.coarse_partition import CoarseRowResult

                    publication_prior = batch_translation_log_prior
                    if publication_prior is not None and publication_prior.ndim == 1:
                        publication_prior = jnp.broadcast_to(publication_prior, (batch_size, n_trans))
                    publication_groups = (CoarseRowResult(
                        np.arange(actual_batch_size, dtype=np.int32),
                        coarse_gaussian_gemm_hybrid_batch_result,
                        publication_prior,
                    ),)
                published = publish_coarse_rows(
                    publication_groups,
                    actual_image_count=actual_batch_size,
                    n_rotations=n_rot,
                    n_translations=n_trans,
                    class_log_prior=class_log_priors_np[0],
                    rotation_log_prior=(None if rotation_log_prior_padded is None else rotation_log_prior_padded[0, :n_rot]),
                    rotation_chunk_rows=rotation_block_size,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    tie_score_ulps=relion_f32_coarse_tie_ulps,
                    posterior_backend="cuda" if coarse_cuda_posterior_requested else "jax",
                )
                if coarse_cuda_posterior_requested:
                    coarse_cuda_posterior_batch_count += 1
                    coarse_cuda_posterior_group_count += len(publication_groups)
                    coarse_cuda_posterior_image_count += actual_batch_size
                target = slice(start_idx, end_idx)
                hard_assignment[target] = published["best_pose"]
                class_assignment[target] = 0
                normalization_log_z[target] = published["global_log_z"]
                normalization_log_evidence[target] = published["global_log_z"]
                log_evidence[target] = published["global_log_z"].astype(np.float32)
                best_log_score[target] = published["best_score"].astype(np.float32)
                max_posterior[target] = published["pmax"]
                class_log_evidence[0, target] = published["global_log_z"]
                relion_f32_sum_weight[target] = published["sum_weight"]
                n_sig_all[target] = published["n_significant"]
                cutoff_count_all[target] = published["cutoff_count"]
                for local, image in enumerate(indices):
                    begin, stop = published["support_offsets"][local : local + 2]
                    pose_ids = published["support_ids"][begin:stop].astype(np.int32)
                    significant_sample_indices[0][int(image)] = pose_ids
                    sig_rot_any[0, pose_ids // n_trans] = True
                start_idx = end_idx
                continue

        compact_hybrid_scores = (
            None
            if coarse_gaussian_gemm_hybrid_batch_result is None
            else coarse_gaussian_gemm_hybrid_batch_result.compact_scores
        )
        global_max = jnp.full(batch_size, -jnp.inf)
        global_sum = jnp.zeros(batch_size, dtype=jnp.float64)
        class_max_values = []
        class_sum_values = []
        best_score_batch = jnp.full(batch_size, -jnp.inf)
        best_argmax_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        best_class_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        class_best_scores = [jnp.full(batch_size, -jnp.inf) for _ in range(n_classes)] if return_class_best else None
        class_best_argmaxes = [jnp.zeros(batch_size, dtype=jnp.int32) for _ in range(n_classes)] if return_class_best else None
        class_second_best_scores = (
            [jnp.full(batch_size, -jnp.inf) for _ in range(n_classes)] if track_class_second else None
        )
        class_second_best_argmaxes = (
            [jnp.zeros(batch_size, dtype=jnp.int32) for _ in range(n_classes)] if track_class_second else None
        )
        cache_score_blocks = compact_hybrid_scores is None and collect_significance and (
            coarse_gemm_diagnostic_positions is not None
            or _significance_score_cache_enabled(
                batch_size,
                n_classes,
                n_rot_padded,
                n_trans,
                use_float64_scoring=use_float64_scoring,
            )
        )
        cached_class_score_blocks = [] if cache_score_blocks else None
        if passive_score_dump and cached_class_score_blocks is None:
            raise RuntimeError(
                f"{_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV}=1 requires the "
                "production significance score cache"
            )

        # ``RECOVAR_PASS1_FUSED=1`` swaps the per-block 4-5 separate JIT
        # dispatches (project/score, padding-mask, add-priors, 2× logsumexp)
        # for one fused @jit call. Bit-identical when active; disabled if any
        # debug-dump path is on so per-block pre/post-prior captures still
        # have access to intermediate scores.
        use_fused_pass1 = (
            _pass1_fused_enabled()
            and score_mode == "gaussian"
            and not use_relion_projector
            and dump_target_pre_prior_blocks_per_class is None
            and dump_target_with_prior_blocks_per_class is None
        )
        if passive_score_dump and use_fused_pass1:
            raise RuntimeError(
                f"{_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV}=1 does not support "
                "the fused pass-1 diagnostic path"
            )
        if relion_f32_coarse_support_enabled and use_fused_pass1:
            raise RuntimeError(
                "RELION float32 coarse support requires access to pre-prior "
                "scores for min_diff2 offset reconstruction"
            )

        if not relion_f32_coarse_support_enabled:
            relion_raw_score_max = None
        elif coarse_gaussian_gemm_hybrid_batch_result is not None:
            relion_raw_score_max = (
                coarse_gaussian_gemm_hybrid_batch_result.raw_score_max
            )
        else:
            relion_raw_score_max = jnp.full(
                batch_size,
                -jnp.inf,
                dtype=jnp.float32,
            )

        # Precompute fused-path inputs once per batch (constant across class/block).
        if use_fused_pass1:
            _fused_half_weights = half_weights_windowed if use_window else half_weights
            _fused_max_r_static = projection_kwargs.get("max_r", None) if use_window else None
            _fused_window_indices = window_indices if use_window else jnp.zeros(0, dtype=jnp.int32)
            if translation_log_prior is None:
                _fused_trans_lp_per_image = jnp.zeros((batch_size, n_trans), dtype=score_real_dtype)
            elif translation_log_prior.ndim == 1:
                _fused_trans_lp_per_image = jnp.broadcast_to(
                    jnp.asarray(batch_translation_log_prior, dtype=score_real_dtype),
                    (batch_size, n_trans),
                )
            else:
                _fused_trans_lp_per_image = jnp.asarray(batch_translation_log_prior, dtype=score_real_dtype)

        for class_index, mean_for_proj in enumerate(means_for_proj):
            class_max = jnp.full(batch_size, -jnp.inf)
            class_sum = jnp.zeros(batch_size, dtype=jnp.float64)
            cached_score_blocks = [] if cached_class_score_blocks is not None else None
            score_block_count = n_blocks if compact_hybrid_scores is None else 0
            for block_index in range(score_block_count):
                r0 = block_index * rotation_block_size
                r1 = r0 + rotation_block_size
                if use_fused_pass1:
                    valid_count = jnp.asarray(min(rotation_block_size, n_rot - r0), dtype=jnp.int32)
                    if rotation_log_prior_padded is None:
                        rot_lp_block = jnp.zeros(rotation_block_size, dtype=score_real_dtype)
                    else:
                        rot_lp_block = jnp.asarray(
                            rotation_log_prior_padded[class_index, r0:r1], dtype=score_real_dtype
                        )
                    scores, class_max, class_sum, global_max, global_sum = _fused_score_priors_logsumexp_block(
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        _fused_half_weights,
                        _fused_window_indices,
                        rot_lp_block,
                        _fused_trans_lp_per_image,
                        float(class_log_priors_np[class_index]),
                        valid_count,
                        class_max,
                        class_sum,
                        global_max,
                        global_sum,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        volume_shape=volume_shape,
                        disc_type=disc_type,
                        use_window=use_window,
                        use_float64_scoring=use_float64_scoring,
                        rotation_block_size=int(rotation_block_size),
                        batch_size=int(batch_size),
                        n_trans=int(n_trans),
                        n_windowed=int(n_windowed) if use_window else 0,
                        max_r_static=_fused_max_r_static,
                    )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                else:
                    scores = _score_block(
                        class_index,
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        batch_size,
                        rotation_start=r0,
                    )
                    direct_scores_for_diagnostic = coarse_gemm_direct_capture["scores"]
                    if (
                        coarse_gemm_diagnostic_positions is not None
                        and direct_scores_for_diagnostic is None
                    ):
                        raise RuntimeError(
                            "coarse GEMM diagnostic did not capture the paired "
                            "direct-square score block",
                        )
                    if r1 > n_rot:
                        valid = n_rot - r0
                        valid_rotation = (
                            jnp.arange(rotation_block_size)[None, :, None] < valid
                        )
                        scores = jnp.where(valid_rotation, scores, -jnp.inf)
                        if direct_scores_for_diagnostic is not None:
                            direct_scores_for_diagnostic = jnp.where(
                                valid_rotation,
                                direct_scores_for_diagnostic,
                                -jnp.inf,
                            )
                    if passive_raw_score_blocks_per_class is not None:
                        passive_raw_score_blocks_per_class[class_index].append(scores)
                    if (
                        relion_raw_score_max is not None
                        and coarse_gaussian_gemm_hybrid_batch_result is None
                    ):
                        relion_raw_score_max = jnp.maximum(
                            relion_raw_score_max,
                            jnp.max(scores.reshape(batch_size, -1), axis=1),
                        )
                    # Capture pre-prior raw scores for dump targets BEFORE _add_priors.
                    # scores shape: (batch_size, rotation_block_size, n_trans).
                    # For comparison vs RELION exp_Mweight_diff2, recovar's score is
                    # -0.5 * residual where residual = sum_pixel((proj*ctf - shifted_img)² - |img|²)
                    # / sigma² × half_weights. RELION's diff2 has the same core term
                    # plus the per-image Xi2/2 constant. Per-pose RELATIVE differences
                    # cancel the constant, so direct diff is meaningful.
                    if dump_target_pre_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_pre_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    if coarse_gemm_macro_pre_prior_blocks is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        target_rows = coarse_gemm_diagnostic_positions
                        coarse_gemm_macro_pre_prior_blocks[class_index].append(
                            scores[target_rows, :actual_rot, :]
                        )
                        coarse_gemm_direct_pre_prior_blocks[class_index].append(
                            direct_scores_for_diagnostic[
                                target_rows,
                                :actual_rot,
                                :,
                            ]
                        )
                    if coarse_gemm_pre_prior_stream_state is not None:
                        if direct_scores_for_diagnostic is None:
                            raise RuntimeError(
                                "coarse GEMM pre-prior streaming diagnostic "
                                "requires the paired direct-square score block",
                            )
                        coarse_gemm_pre_prior_stream_state = (
                            update_coarse_gemm_streaming_state(
                                coarse_gemm_pre_prior_stream_state,
                                direct_scores_for_diagnostic,
                                scores,
                                candidate_offset=(
                                    class_index * n_rot * n_trans
                                    + r0 * n_trans
                                ),
                                actual_image_count=actual_batch_size,
                            )
                        )
                    if not (
                        coarse_gaussian_gemm_hybrid_batch_result is not None
                        and coarse_gaussian_gemm_hybrid_batch_result.scores_include_priors
                    ):
                        scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    if direct_scores_for_diagnostic is not None:
                        direct_scores_for_diagnostic = _add_priors(
                            direct_scores_for_diagnostic,
                            class_index,
                            r0,
                            r1,
                            batch_translation_log_prior,
                        )
                        if coarse_gemm_stream_state is not None:
                            coarse_gemm_stream_state = update_coarse_gemm_streaming_state(
                                coarse_gemm_stream_state,
                                direct_scores_for_diagnostic,
                                scores,
                                candidate_offset=(
                                    class_index * n_rot * n_trans + r0 * n_trans
                                ),
                                actual_image_count=actual_batch_size,
                            )
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        target_rows = coarse_gemm_diagnostic_positions
                        if target_rows is not None:
                            coarse_gemm_macro_with_prior_blocks[class_index].append(
                                scores[target_rows, :actual_rot, :]
                            )
                            coarse_gemm_direct_with_prior_blocks[class_index].append(
                                direct_scores_for_diagnostic[
                                    target_rows,
                                    :actual_rot,
                                    :,
                                ]
                            )
                    if dump_target_with_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_with_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
                    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
                flat_scores = scores.reshape(batch_size, -1)
                block_best = jnp.max(flat_scores, axis=1)
                block_argmax = jnp.argmax(flat_scores, axis=1)
                improved = block_best > best_score_batch
                best_score_batch = jnp.where(improved, block_best, best_score_batch)
                best_argmax_batch = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax_batch)
                best_class_batch = jnp.where(improved, class_index, best_class_batch)
                if return_class_best:
                    previous_best = class_best_scores[class_index]
                    previous_best_argmax = class_best_argmaxes[class_index]
                    class_improved = block_best > previous_best
                    if track_class_second:
                        if flat_scores.shape[1] < 2:
                            raise RuntimeError("class runner-up diagnostic requires at least two poses per block")
                        rows = jnp.arange(batch_size)
                        block_without_best = flat_scores.at[rows, block_argmax].set(-jnp.inf)
                        block_second = jnp.max(block_without_best, axis=1)
                        block_second_argmax = jnp.argmax(block_without_best, axis=1)
                        previous_second = class_second_best_scores[class_index]
                        previous_second_argmax = class_second_best_argmaxes[class_index]

                        improved_second_from_previous = previous_best >= block_second
                        improved_second = jnp.where(
                            improved_second_from_previous,
                            previous_best,
                            block_second,
                        )
                        improved_second_argmax = jnp.where(
                            improved_second_from_previous,
                            previous_best_argmax,
                            block_second_argmax + r0 * n_trans,
                        )
                        retained_second_from_previous = previous_second >= block_best
                        retained_second = jnp.where(
                            retained_second_from_previous,
                            previous_second,
                            block_best,
                        )
                        retained_second_argmax = jnp.where(
                            retained_second_from_previous,
                            previous_second_argmax,
                            block_argmax + r0 * n_trans,
                        )
                        class_second_best_scores[class_index] = jnp.where(
                            class_improved,
                            improved_second,
                            retained_second,
                        )
                        class_second_best_argmaxes[class_index] = jnp.where(
                            class_improved,
                            improved_second_argmax,
                            retained_second_argmax,
                        )
                    class_best_scores[class_index] = jnp.where(
                        class_improved,
                        block_best,
                        previous_best,
                    )
                    class_best_argmaxes[class_index] = jnp.where(
                        class_improved,
                        block_argmax + r0 * n_trans,
                        previous_best_argmax,
                    )
            if cached_class_score_blocks is not None:
                cached_class_score_blocks.append(cached_score_blocks)
            class_max_values.append(class_max)
            class_sum_values.append(class_sum)

        if compact_hybrid_scores is not None:
            # Active compact candidates are ordered by ascending source16
            # block, rotation, then translation.  One device reduction avoids
            # replaying every empty global rotation block; omitted candidates
            # are certified at least 138 score units below the maximum and
            # therefore contribute exact zero to RELION's float32 posterior.
            global_max, global_sum = _update_logsumexp(
                jnp.full(batch_size, -jnp.inf),
                jnp.zeros(batch_size, dtype=jnp.float64),
                compact_hybrid_scores.posterior_scores_flat,
            )
            class_max_values = [global_max]
            class_sum_values = [global_sum]
            best_score_batch = compact_hybrid_scores.best_score
            best_argmax_batch = compact_hybrid_scores.best_pose
            best_class_batch = jnp.zeros(batch_size, dtype=jnp.int32)

        # The second significance pass may recompute score blocks when the
        # production cache is disabled.  The all-particle diagnostic is a
        # first-pass online reduction; do not execute or count direct scores a
        # second time merely because support probabilities need replay.
        coarse_gemm_stream_capture_control["enabled"] = False

        if tree_rescore_enabled:
            tree_score_dtype = np.float64 if use_float64_scoring else np.float32
            best_scores_np = np.asarray(class_best_scores[0], dtype=tree_score_dtype)
            second_scores_np = np.asarray(class_second_best_scores[0], dtype=tree_score_dtype)
            score_margins = best_scores_np - second_scores_np
            ambiguous_rows = np.flatnonzero(
                np.isfinite(score_margins) & (score_margins <= tree_rescore_max_margin)
            ).astype(np.int32)
            tree_rescore_examined += int(batch_size)
            tree_rescore_ambiguous += int(ambiguous_rows.size)
            if ambiguous_rows.size:
                best_pose_np = np.asarray(class_best_argmaxes[0], dtype=np.int32)[ambiguous_rows]
                second_pose_np = np.asarray(class_second_best_argmaxes[0], dtype=np.int32)[
                    ambiguous_rows
                ]
                candidate_pose_ids = np.sort(
                    np.stack([best_pose_np, second_pose_np], axis=1),
                    axis=1,
                )
                candidate_rotation_ids = candidate_pose_ids // n_trans
                candidate_translation_ids = candidate_pose_ids % n_trans
                candidate_rotations = jnp.asarray(
                    rotations[candidate_rotation_ids.reshape(-1)],
                    dtype=jnp.float32,
                ).reshape(
                    ambiguous_rows.size,
                    2,
                    3,
                    3,
                )
                unshifted_candidates = jnp.broadcast_to(
                    tree_rescore_unshifted_data[
                        jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                    ],
                    (
                        ambiguous_rows.size,
                        2,
                        tree_rescore_unshifted_data.shape[-1],
                    ),
                )
                candidate_translation_angles = tree_rescore_translation_angles[
                    jnp.asarray(candidate_translation_ids, dtype=jnp.int32)
                ]
                score_weight_candidates = jnp.broadcast_to(
                    tree_rescore_corr_img_data[
                        jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                    ],
                    unshifted_candidates.shape,
                )
                rescored_candidates = _relion_coarse_normalized_cc_rescore(
                    unshifted_candidates,
                    score_weight_candidates,
                    None,
                    half_weights_windowed if use_window else half_weights,
                    tree_rescore_fftw_order,
                    projector_full=coarse_gaussian_projector_full,
                    rotation_matrices=candidate_rotations,
                    translation_angles=candidate_translation_angles,
                    current_size=score_size,
                    padding_factor=projection_padding_factor,
                    projector_max_r=relion_projector_r_max,
                    numerator_weight_candidates=score_weight_candidates,
                )
                rescored_scores_np = np.asarray(rescored_candidates, dtype=tree_score_dtype)
                rescored_winner_slot, exact_ties = _select_relion_coarse_rescore_winner_slots(
                    rescored_scores_np,
                    candidate_pose_ids,
                    n_trans=n_trans,
                    healpix_order=coarse_healpix_order,
                    coarse_rotation_ids=coarse_rotation_ids,
                    score_dtype=tree_score_dtype,
                )
                _maybe_dump_tree_rescore_batch(
                    experiment_dataset=experiment_dataset,
                    indices=indices,
                    ambiguous_rows=ambiguous_rows,
                    candidate_pose_ids=candidate_pose_ids,
                    original_best_pose=best_pose_np,
                    original_best_score=best_scores_np[ambiguous_rows],
                    original_second_pose=second_pose_np,
                    original_second_score=second_scores_np[ambiguous_rows],
                    rescored_scores=rescored_scores_np,
                    rescored_winner_slot=rescored_winner_slot,
                    shifted_candidates=unshifted_candidates,
                    score_weight_candidates=score_weight_candidates,
                    numerator_weight_candidates=score_weight_candidates,
                    rotation_matrices=candidate_rotations,
                    translation_angles=candidate_translation_angles,
                    n_trans=n_trans,
                    half_weights=(
                        half_weights_windowed if use_window else half_weights
                    ),
                    packed_to_compact=tree_rescore_fftw_order,
                    projector_full=coarse_gaussian_projector_full,
                    current_size=score_size,
                    padding_factor=projection_padding_factor,
                    projector_max_r=relion_projector_r_max,
                    debug_iteration=debug_iteration,
                )
                tree_rescore_exact_ties += exact_ties
                row_ids = np.arange(ambiguous_rows.size, dtype=np.int32)
                rescored_runner_slot = 1 - rescored_winner_slot
                rescored_winner_pose = candidate_pose_ids[row_ids, rescored_winner_slot]
                rescored_runner_pose = candidate_pose_ids[row_ids, rescored_runner_slot]
                rescored_winner_score = rescored_scores_np[row_ids, rescored_winner_slot]
                rescored_runner_score = rescored_scores_np[row_ids, rescored_runner_slot]
                tree_rescore_winner_changes += int(
                    np.count_nonzero(rescored_winner_pose != best_pose_np)
                )
                applied_rows = np.arange(ambiguous_rows.size, dtype=np.int32)
                if applied_rows.size:
                    rows_jax = jnp.asarray(ambiguous_rows[applied_rows], dtype=jnp.int32)
                    best_argmax_batch = best_argmax_batch.at[rows_jax].set(
                        rescored_winner_pose[applied_rows]
                    )
                    best_score_batch = best_score_batch.at[rows_jax].set(
                        rescored_winner_score[applied_rows]
                    )
                    class_best_argmaxes[0] = class_best_argmaxes[0].at[rows_jax].set(
                        rescored_winner_pose[applied_rows]
                    )
                    class_best_scores[0] = class_best_scores[0].at[rows_jax].set(
                        rescored_winner_score[applied_rows]
                    )
                    class_second_best_argmaxes[0] = class_second_best_argmaxes[0].at[
                        rows_jax
                    ].set(rescored_runner_pose[applied_rows])
                    class_second_best_scores[0] = class_second_best_scores[0].at[
                        rows_jax
                    ].set(rescored_runner_score[applied_rows])

        global_log_z = global_max + jnp.log(global_sum)
        class_log_z_values = [
            class_max + jnp.log(class_sum) for class_max, class_sum in zip(class_max_values, class_sum_values)
        ]

        class_weight_mats = []
        compact_support_pose_ids = None
        if collect_significance:
            if compact_hybrid_scores is not None:
                batch_values = compact_hybrid_scores.posterior_scores_flat
            else:
                for class_index, mean_for_proj in enumerate(means_for_proj):
                    class_weight_blocks = []
                    for block_index in range(n_blocks):
                        r0 = block_index * rotation_block_size
                        r1 = r0 + rotation_block_size
                        if cached_class_score_blocks is None:
                            scores = _score_block(
                                class_index,
                                mean_for_proj,
                                rotations_padded[r0:r1],
                                shifted_data,
                                batch_norm,
                                ctf2_data,
                                batch_size,
                                rotation_start=r0,
                            )
                            if r1 > n_rot:
                                valid = n_rot - r0
                                scores = jnp.where(
                                    jnp.arange(rotation_block_size)[None, :, None]
                                    < valid,
                                    scores,
                                    -jnp.inf,
                                )
                            if not (
                                coarse_gaussian_gemm_hybrid_batch_result is not None
                                and coarse_gaussian_gemm_hybrid_batch_result.scores_include_priors
                            ):
                                scores = _add_priors(
                                    scores,
                                    class_index,
                                    r0,
                                    r1,
                                    batch_translation_log_prior,
                                )
                        else:
                            scores = cached_class_score_blocks[class_index][block_index]
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        if relion_f32_coarse_support_enabled:
                            class_weight_blocks.append(
                                scores[:, :actual_rot, :].reshape(batch_size, -1),
                            )
                        else:
                            probs = jnp.exp(scores - global_log_z[:, None, None])
                            class_weight_blocks.append(
                                probs[:, :actual_rot, :].reshape(batch_size, -1),
                            )
                    class_weight_mats.append(
                        jnp.concatenate(class_weight_blocks, axis=1),
                    )

                batch_values = jnp.concatenate(class_weight_mats, axis=1)
            if relion_f32_coarse_support_enabled:
                from recovar.em.helpers.oversampling import relion_cuda_f32_coarse_posterior

                posterior_kwargs = {}
                if compact_hybrid_scores is not None:
                    # Stage 1's positive-only primitive is the exact
                    # correctness oracle, but its current sequential vmap
                    # performs one device-to-host count synchronization per
                    # image.  The intended runtime path instead scans this
                    # much smaller fixed-capacity table in one shape-stable
                    # computation; positive-count clamping below the CUB call
                    # still excludes exact-zero candidates from support.
                    posterior_kwargs["filter_positive_before_sort"] = False
                (
                    batch_weights,
                    batch_sig_mask,
                    batch_n_sig,
                    batch_cutoff_count,
                    _batch_sum_weight,
                    _batch_significant_weight,
                ) = relion_cuda_f32_coarse_posterior(
                    batch_values,
                    adaptive_fraction=float(adaptive_fraction),
                    max_significants=max_significants,
                    tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    min_diff2_offsets=-relion_raw_score_max,
                    **posterior_kwargs,
                )
                relion_f32_sum_weight[start_idx:end_idx] = np.asarray(
                    _batch_sum_weight[:actual_batch_size],
                    dtype=np.float32,
                )
                if compact_hybrid_scores is None:
                    batch_sig_rot_mask = jnp.any(
                        batch_sig_mask.reshape(
                            batch_size,
                            n_classes * n_rot,
                            n_trans,
                        ),
                        axis=2,
                    )
            else:
                batch_weights = batch_values
                (
                    batch_sig_mask,
                    batch_sig_rot_mask,
                    batch_n_sig,
                    batch_cutoff_count,
                ) = _find_sig(
                    batch_weights,
                    n_classes * n_rot,
                    n_trans,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    return_cutoff_count=True,
                )
            batch_sig_mask_np = np.array(batch_sig_mask, dtype=bool, copy=True)
            if compact_hybrid_scores is None:
                sig_rot_any |= np.asarray(
                    jnp.any(batch_sig_rot_mask[:actual_batch_size], axis=0),
                    dtype=bool,
                ).reshape(n_classes, n_rot)
            else:
                compact_support_pose_ids = (
                    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
                        compact_hybrid_scores,
                        batch_sig_mask_np,
                    )
                )
                for selected_pose_ids in compact_support_pose_ids[
                    :actual_batch_size
                ]:
                    if selected_pose_ids.size:
                        sig_rot_any[0, selected_pose_ids // n_trans] = True
                _maybe_dump_coarse_runtime_prefix_operands(
                    dump_dir=coarse_runtime_prefix_dump_dir,
                    dump_label=coarse_runtime_prefix_dump_label,
                    target_original_indices=coarse_runtime_prefix_dump_targets,
                    experiment_dataset=experiment_dataset,
                    indices=indices,
                    compact_result=coarse_gaussian_gemm_hybrid_batch_result,
                    support_pose_ids=compact_support_pose_ids,
                    batch_weights=batch_weights,
                    batch_sig_mask=batch_sig_mask_np,
                    batch_n_sig=batch_n_sig,
                    batch_cutoff_count=batch_cutoff_count,
                    batch_sum_weight=_batch_sum_weight,
                    batch_significant_weight=_batch_significant_weight,
                    projection_cache=coarse_gaussian_gemm_projection_cache,
                    shifted_corrected=coarse_gaussian_shifted_corrected,
                    pixel_weight=coarse_gaussian_pixel_weight,
                    initial_diff2=coarse_gaussian_initial_diff2,
                    full_to_compact=coarse_gaussian_gemm_certificate_topology.full_to_compact,
                    logical_full_pixel_count=(
                        coarse_gaussian_square_layout.logical_square_count
                    ),
                    class_log_prior=class_log_priors_np[0],
                    rotation_log_prior=(
                        None
                        if rotation_log_prior_padded is None
                        else rotation_log_prior_padded[0, :n_rot]
                    ),
                    translation_log_prior=batch_translation_log_prior,
                    current_size=current_size,
                    physical_current_size=(
                        coarse_gaussian_square_layout.physical_current_size
                    ),
                    debug_iteration=debug_iteration,
                )
            n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig[:actual_batch_size], dtype=np.int32)
            cutoff_count_all[start_idx:end_idx] = np.asarray(
                batch_cutoff_count[:actual_batch_size],
                dtype=np.int32,
            )
        else:
            batch_sig_mask_np = None
            n_sig_all[start_idx:end_idx] = 0
            cutoff_count_all[start_idx:end_idx] = 0

        if coarse_gemm_diagnostic_positions is not None:
            if batch_sig_mask_np is None or coarse_gaussian_gemm_resource_estimate is None:
                raise RuntimeError(
                    "coarse GEMM paired diagnostic requires production support "
                    "and a sealed resource estimate",
                )

            def _stack_coarse_gemm_diagnostic_blocks(blocks_by_class):
                return jnp.stack(
                    [jnp.concatenate(blocks, axis=1) for blocks in blocks_by_class],
                    axis=1,
                )

            direct_pre_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_direct_pre_prior_blocks,
            )
            macro_pre_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_macro_pre_prior_blocks,
            )
            direct_with_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_direct_with_prior_blocks,
            )
            macro_with_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_macro_with_prior_blocks,
            )
            direct_values = direct_with_prior.reshape(
                direct_with_prior.shape[0],
                -1,
            )
            if relion_f32_coarse_support_enabled:
                from recovar.em.helpers.oversampling import relion_cuda_f32_coarse_posterior

                direct_raw_max = jnp.max(
                    direct_pre_prior.reshape(direct_pre_prior.shape[0], -1),
                    axis=1,
                )
                _, direct_sig_mask, *_ = relion_cuda_f32_coarse_posterior(
                    direct_values,
                    adaptive_fraction=float(adaptive_fraction),
                    max_significants=max_significants,
                    tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    min_diff2_offsets=-direct_raw_max,
                )
            else:
                direct_max = jnp.max(direct_values, axis=1, keepdims=True)
                direct_exp = jnp.exp(direct_values - direct_max)
                direct_weights = direct_exp / jnp.sum(
                    direct_exp,
                    axis=1,
                    keepdims=True,
                )
                direct_sig_mask, *_ = _find_sig(
                    direct_weights,
                    n_classes * n_rot,
                    n_trans,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    return_cutoff_count=True,
                )
            macro_support = batch_sig_mask_np[coarse_gemm_diagnostic_positions]
            iteration_label = -1 if debug_iteration is None else int(debug_iteration)
            size_label = -1 if current_size is None else int(current_size)
            diagnostic_path = os.path.join(
                coarse_gaussian_gemm_diagnostic_dir,
                "coarse_gemm_ab_"
                f"{coarse_gaussian_gemm_diagnostic_scope.run_id}_"
                f"{coarse_gaussian_gemm_diagnostic_scope.call_id}_"
                f"it{iteration_label:04d}_cs{size_label:04d}_"
                f"batch{start_idx:08d}_{end_idx:08d}.npz",
            )
            local_indices_np = np.asarray(indices, dtype=np.int64)
            _write_coarse_gaussian_gemm_diagnostic(
                diagnostic_path,
                direct_scores_pre_prior=np.asarray(direct_pre_prior),
                macro_scores_pre_prior=np.asarray(macro_pre_prior),
                direct_scores_with_prior=np.asarray(direct_with_prior),
                macro_scores_with_prior=np.asarray(macro_with_prior),
                direct_support=np.asarray(direct_sig_mask, dtype=bool),
                macro_support=macro_support,
                original_indices=coarse_gemm_diagnostic_original_indices,
                local_indices=local_indices_np[coarse_gemm_diagnostic_positions],
                actual_batch_size=actual_batch_size,
                padded_batch_size=batch_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                resource_estimate=coarse_gaussian_gemm_resource_estimate,
                diagnostic_scope=coarse_gaussian_gemm_diagnostic_scope,
                diagnostic_selection_policy=(
                    coarse_gaussian_gemm_diagnostic_selection_policy
                ),
                debug_iteration=debug_iteration,
                current_size=current_size,
            )
            coarse_gaussian_gemm_diagnostic_paths.append(diagnostic_path)
            for value in coarse_gemm_diagnostic_original_indices:
                target = int(value)
                coarse_gaussian_gemm_diagnostic_found_targets.add(target)
                coarse_gaussian_gemm_diagnostic_target_counts[target] = (
                    coarse_gaussian_gemm_diagnostic_target_counts.get(target, 0) + 1
                )

        if (coarse_gemm_stream_state is None) != (
            coarse_gemm_pre_prior_stream_state is None
        ):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic requires paired pre-prior "
                "and posterior states",
            )
        if coarse_gemm_stream_state is not None:
            if (
                batch_sig_mask_np is None
                or batch_original_indices_np is None
                or coarse_gaussian_gemm_diagnostic_scope is None
            ):
                raise RuntimeError(
                    "coarse GEMM streaming diagnostic requires production support, "
                    "particle identity, and a resolved call scope",
                )
            iteration_label = -1 if debug_iteration is None else int(debug_iteration)
            size_label = -1 if current_size is None else int(current_size)
            stream_path = os.path.join(
                coarse_gaussian_gemm_stream_diagnostic_dir,
                "coarse_gemm_rescore_"
                f"{coarse_gaussian_gemm_diagnostic_scope.run_id}_"
                f"{coarse_gaussian_gemm_diagnostic_scope.call_id}_"
                f"it{iteration_label:04d}_cs{size_label:04d}_"
                f"batch{start_idx:08d}_{end_idx:08d}.npz",
            )
            write_coarse_gemm_streaming_summary(
                stream_path,
                coarse_gemm_stream_state,
                coarse_gemm_pre_prior_stream_state,
                batch_sig_mask_np,
                original_indices=batch_original_indices_np,
                local_indices=local_indices_np,
                actual_image_count=actual_batch_size,
                padded_image_count=batch_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                diagnostic_run_id=coarse_gaussian_gemm_diagnostic_scope.run_id,
                diagnostic_call_id=coarse_gaussian_gemm_diagnostic_scope.call_id,
                debug_iteration=debug_iteration,
                current_size=current_size,
                n_rotations=n_rot,
                n_translations=n_trans,
            )
            coarse_gaussian_gemm_stream_paths.append(stream_path)
            coarse_gaussian_gemm_stream_original_indices.extend(
                int(value) for value in batch_original_indices_np
            )

        hard_assignment[start_idx:end_idx] = np.asarray(
            best_argmax_batch[:actual_batch_size],
            dtype=np.int32,
        )
        class_assignment[start_idx:end_idx] = np.asarray(
            best_class_batch[:actual_batch_size],
            dtype=np.int32,
        )

        log_score_offset = (
            np.zeros(batch_size, dtype=np.float64)
            if coarse_gaussian_ffi_enabled
            else -0.5
            * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        )
        global_log_z_np = np.asarray(global_log_z, dtype=np.float64)
        best_score_np = np.asarray(best_score_batch, dtype=np.float64)
        output_slice = slice(0, actual_batch_size)
        normalization_log_z[start_idx:end_idx] = global_log_z_np[output_slice]
        normalization_log_evidence[start_idx:end_idx] = (
            global_log_z_np[output_slice] + log_score_offset[output_slice]
        )
        log_evidence[start_idx:end_idx] = normalization_log_evidence[start_idx:end_idx].astype(score_real_dtype)
        best_log_score[start_idx:end_idx] = (
            best_score_np[output_slice] + log_score_offset[output_slice]
        ).astype(score_real_dtype)
        if relion_f32_coarse_support_enabled and collect_significance:
            max_posterior[start_idx:end_idx] = _coarse_max_posterior_for_host(
                batch_weights, actual_batch_size,
                physical_batch=coarse_max_posterior_physical_batch,
            )
        else:
            max_posterior[start_idx:end_idx] = np.exp(
                best_score_np[output_slice] - global_log_z_np[output_slice]
            ).astype(score_real_dtype)
        for class_index, class_log_z in enumerate(class_log_z_values):
            class_log_evidence[class_index, start_idx:end_idx] = (
                np.asarray(class_log_z, dtype=np.float64)[output_slice]
                + log_score_offset[output_slice]
            )
        if return_class_best:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_best_scores[class_index],
                    log_score_offset,
                )
                class_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free[output_slice]
                class_best_log_score[class_index, start_idx:end_idx] = absolute[output_slice]
                class_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_best_argmaxes[class_index][output_slice],
                    dtype=np.int32,
                )
        if return_class_second:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_second_best_scores[class_index],
                    log_score_offset,
                )
                class_second_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free[output_slice]
                class_second_best_log_score[class_index, start_idx:end_idx] = absolute[output_slice]
                class_second_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_second_best_argmaxes[class_index][output_slice],
                    dtype=np.int32,
                )

        if debug_dump_enabled:
            # Concatenate per-class per-block raw scores for the dump targets
            # into per-class arrays of shape (n_targets, n_rot, n_trans).
            target_scores_pre_prior_per_class = None
            target_scores_with_prior_per_class = None
            target_local_positions_for_dump = None
            score_capture_mode = "intrusive_per_block_host_materialization"
            if passive_raw_score_blocks_per_class is not None:
                if cached_class_score_blocks is None:
                    raise RuntimeError("passive significance dump lost cached scores")
                target_scores_pre_prior_per_class = []
                target_scores_with_prior_per_class = []
                for class_index in range(n_classes):
                    raw_blocks = []
                    with_prior_blocks = []
                    for block_index in range(n_blocks):
                        r0 = block_index * rotation_block_size
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        raw_blocks.append(
                            np.asarray(
                                passive_raw_score_blocks_per_class[class_index][block_index][
                                    dump_target_local_positions, :actual_rot, :
                                ],
                                dtype=np.float64,
                            )
                        )
                        with_prior_blocks.append(
                            np.asarray(
                                cached_class_score_blocks[class_index][block_index][
                                    dump_target_local_positions, :actual_rot, :
                                ],
                                dtype=np.float64,
                            )
                        )
                    target_scores_pre_prior_per_class.append(
                        np.concatenate(raw_blocks, axis=1)
                    )
                    target_scores_with_prior_per_class.append(
                        np.concatenate(with_prior_blocks, axis=1)
                    )
                target_local_positions_for_dump = dump_target_local_positions
                score_capture_mode = "passive_cached_after_support"
            elif dump_target_pre_prior_blocks_per_class is not None:
                target_scores_pre_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_pre_prior_blocks_per_class
                ]
                target_scores_with_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_with_prior_blocks_per_class
                ]
                target_local_positions_for_dump = dump_target_local_positions
            projected_reference_rotation_ids = None
            projected_reference_per_class = None
            projected_reference_norm_score_per_class = None
            projected_cross_score_per_class = None
            requested_projection_rotations = sorted(
                parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS") or (),
            )
            if requested_projection_rotations and target_local_positions_for_dump is not None:
                projected_reference_rotation_ids = np.asarray(requested_projection_rotations, dtype=np.int32)
                if (
                    int(projected_reference_rotation_ids[0]) < 0
                    or int(projected_reference_rotation_ids[-1]) >= n_rot
                ):
                    raise ValueError(
                        "RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS contains an out-of-range rotation",
                    )
                projection_rotations = jnp.asarray(rotations[projected_reference_rotation_ids])
                projection_values = []
                projection_norm_scores = []
                projection_cross_scores = []
                for class_index, mean_for_proj in enumerate(means_for_proj):
                    projected_half, projected_abs2 = _project_block(
                        class_index,
                        mean_for_proj,
                        projection_rotations,
                    )
                    if use_window:
                        if projector_returns_compact:
                            if coarse_gaussian_window_positions is not None:
                                projected_half = projected_half[:, coarse_gaussian_window_positions]
                                projected_abs2 = projected_abs2[:, coarse_gaussian_window_positions]
                        else:
                            projected_half = projected_half[:, window_indices]
                            projected_abs2 = projected_abs2[:, window_indices]
                    if not use_float64_scoring:
                        projected_half = projected_half.astype(jnp.complex64)
                        projected_abs2 = projected_abs2.astype(jnp.float32)
                    score_weights = half_weights_windowed if use_window else half_weights
                    weighted_projected = projected_half * score_weights
                    weighted_projected_abs2 = projected_abs2 * score_weights
                    component_cross = (
                        -2.0
                        * jnp.matmul(
                            jnp.conj(shifted_data),
                            weighted_projected.T,
                            precision=jax.lax.Precision.HIGHEST,
                        ).real
                    )
                    component_cross = component_cross.reshape(
                        batch_size,
                        n_trans,
                        projected_reference_rotation_ids.size,
                    ).swapaxes(1, 2)
                    component_norm = jnp.matmul(
                        ctf2_data,
                        weighted_projected_abs2.T,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    component_norm = jnp.broadcast_to(
                        component_norm[..., None],
                        component_cross.shape,
                    )
                    projection_values.append(np.asarray(projected_half, dtype=np.complex128))
                    projection_norm_scores.append(
                        np.asarray(-0.5 * component_norm, dtype=np.float64)
                    )
                    projection_cross_scores.append(
                        np.asarray(-0.5 * component_cross, dtype=np.float64)
                    )
                projected_reference_per_class = np.stack(projection_values, axis=0)
                projected_reference_norm_score_per_class = np.stack(
                    projection_norm_scores,
                    axis=0,
                )
                projected_cross_score_per_class = np.stack(
                    projection_cross_scores,
                    axis=0,
                )
            _maybe_dump_k_class_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                n_classes=n_classes,
                rotations=rotations,
                translations=translations,
                class_weight_mats=(
                    [
                        np.asarray(
                            batch_weights.reshape(
                                batch_size,
                                n_classes,
                                n_rot * n_trans,
                            )[:, class_index, :],
                            dtype=np.float64,
                        )
                        for class_index in range(n_classes)
                    ]
                    if relion_f32_coarse_support_enabled
                    else [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats]
                ),
                batch_sig_mask=batch_sig_mask_np,
                batch_n_sig=np.asarray(batch_n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(best_argmax_batch, dtype=np.int64),
                class_assignment_batch=np.asarray(best_class_batch, dtype=np.int64),
                global_log_z=global_log_z_np,
                class_log_z_values=class_log_z_values,
                best_score=best_score_np,
                max_posterior=max_posterior[start_idx:end_idx],
                rotation_log_prior_padded=rotation_log_prior_padded,
                batch_translation_log_prior=batch_translation_log_prior,
                class_log_priors=class_log_priors_np,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                target_local_positions=target_local_positions_for_dump,
                target_scores_pre_prior_per_class=target_scores_pre_prior_per_class,
                target_scores_with_prior_per_class=target_scores_with_prior_per_class,
                projected_reference_rotation_ids=projected_reference_rotation_ids,
                projected_reference_per_class=projected_reference_per_class,
                projected_reference_norm_score_per_class=(
                    projected_reference_norm_score_per_class
                ),
                projected_cross_score_per_class=projected_cross_score_per_class,
                shifted_data=shifted_data,
                ctf2_data=ctf2_data,
                window_indices=window_indices,
                half_weights_used=half_weights_windowed if use_window else half_weights,
                coarse_gaussian_shifted_corrected=coarse_gaussian_shifted_corrected,
                coarse_gaussian_unshifted_corrected=coarse_gaussian_unshifted_corrected,
                coarse_gaussian_pixel_weight=coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2=coarse_gaussian_initial_diff2,
                coarse_gaussian_score_indices=coarse_gaussian_score_indices,
                translation_phase_source=translations_source,
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=projection_padding_factor,
                relion_f32_sum_weight=(
                    _batch_sum_weight if relion_f32_coarse_support_enabled else None
                ),
                relion_f32_significant_weight=(
                    _batch_significant_weight
                    if relion_f32_coarse_support_enabled
                    else None
                ),
                relion_f32_cutoff_count=(
                    batch_cutoff_count if relion_f32_coarse_support_enabled else None
                ),
                score_capture_mode=score_capture_mode,
                debug_iteration=debug_iteration,
            )

        if collect_significance:
            if compact_hybrid_scores is not None:
                if compact_support_pose_ids is None:
                    raise RuntimeError("compact hybrid support mapping was not produced")
                for local_idx, global_idx in enumerate(indices):
                    significant_sample_indices[0][int(global_idx)] = (
                        compact_support_pose_ids[local_idx].copy()
                    )
            else:
                samples_per_class = n_rot * n_trans
                for local_idx, global_idx in enumerate(indices):
                    global_idx = int(global_idx)
                    for class_index in range(n_classes):
                        c0 = class_index * samples_per_class
                        c1 = c0 + samples_per_class
                        mask = batch_sig_mask_np[local_idx, c0:c1]
                        significant_sample_indices[class_index][global_idx] = compact_significant_sample_indices_from_mask(
                            mask,
                        )
        start_idx = end_idx

    coarse_gaussian_gemm_hybrid_full_dense_batch_count = (
        coarse_gaussian_gemm_hybrid_static_dense_batch_count
        + coarse_gaussian_gemm_hybrid_fallback_batch_count
    )
    coarse_gaussian_gemm_hybrid_full_dense_image_count = (
        coarse_gaussian_gemm_hybrid_static_dense_image_count
        + coarse_gaussian_gemm_hybrid_fallback_image_count
    )
    if coarse_gaussian_gemm_hybrid_requested:
        if (
            coarse_gaussian_gemm_hybrid_fused_full_batch_count
            + coarse_gaussian_gemm_hybrid_rectangular_full_batch_count
            != coarse_gaussian_gemm_hybrid_full_dense_batch_count
            or coarse_gaussian_gemm_hybrid_fused_full_image_count
            + coarse_gaussian_gemm_hybrid_rectangular_full_image_count
            != coarse_gaussian_gemm_hybrid_full_dense_image_count
        ):
            raise RuntimeError(
                "hybrid full-dense backend counts do not cover every full-dense batch",
            )
        if coarse_gaussian_fused_full_fallback_armed:
            if (
                coarse_gaussian_gemm_hybrid_rectangular_full_batch_count
                or coarse_gaussian_gemm_hybrid_rectangular_full_image_count
            ):
                raise RuntimeError(
                    "armed fused hybrid fallback executed the rectangular scorer",
                )
        elif (
            coarse_gaussian_gemm_hybrid_fused_full_batch_count
            or coarse_gaussian_gemm_hybrid_fused_full_image_count
        ):
            raise RuntimeError(
                "unarmed fused hybrid fallback recorded fused execution",
            )
        if (
            int(coarse_selector_execution["fused_calls"])
            != coarse_gaussian_gemm_hybrid_fused_full_batch_count
            or int(coarse_selector_execution["actual_rows"])
            != coarse_gaussian_gemm_hybrid_fused_full_image_count
        ):
            raise RuntimeError(
                "hybrid fused fallback counts disagree with selector execution",
            )

    coarse_selector_audit = _validate_coarse_selector_audit(
        {
            "score_mode": score_mode,
            "translation_count": int(n_trans),
            "requested_fused": bool(coarse_fused_projector_requested),
            "effective_fused": bool(coarse_selector_execution["fused_calls"]),
            "requested_workers": int(coarse_multistream_worker_count),
            "effective_workers": (
                int(coarse_multistream_worker_count)
                if coarse_selector_execution["multistream_calls"]
                else 0
            ),
            "requested_atomic": bool(coarse_native_atomic_reduction_requested),
            "effective_atomic": bool(
                coarse_selector_execution["native_atomic_selected_calls"]
            ),
            "requested_prehalf": bool(coarse_prehalf_weight_requested),
            "effective_prehalf": bool(
                coarse_selector_execution["prehalf_selected_calls"]
            ),
            "wrapper": coarse_selector_execution["wrapper"],
            "target": coarse_selector_execution["target"],
            "counts": {
                "fused_calls": int(coarse_selector_execution["fused_calls"]),
                "actual_rows": int(coarse_selector_execution["actual_rows"]),
                "multistream_calls": int(
                    coarse_selector_execution["multistream_calls"]
                ),
                "native_atomic_selected_calls": int(
                    coarse_selector_execution["native_atomic_selected_calls"]
                ),
                "prehalf_selected_calls": int(
                    coarse_selector_execution["prehalf_selected_calls"]
                ),
            },
        }
    )
    coarse_gaussian_gemm_scope_manifest_path = None
    coarse_gaussian_gemm_aggregate_manifest_path = None
    coarse_gaussian_gemm_stream_scope_manifest_path = None
    coarse_gaussian_gemm_stream_aggregate_manifest_path = None
    if coarse_gaussian_gemm_diagnostic_targets is not None:
        missing_targets = (
            coarse_gaussian_gemm_diagnostic_targets
            - coarse_gaussian_gemm_diagnostic_found_targets
        )
        if missing_targets:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} contains "
                "targets that were not captured in this diagnostic call "
                f"scope: {sorted(missing_targets)}",
            )
        (
            coarse_gaussian_gemm_scope_manifest_path,
            coarse_gaussian_gemm_aggregate_manifest_path,
        ) = _seal_coarse_gaussian_gemm_diagnostic_scope(
            coarse_gaussian_gemm_diagnostic_dir,
            scope=coarse_gaussian_gemm_diagnostic_scope,
            selection_policy=coarse_gaussian_gemm_diagnostic_selection_policy,
            requested_targets=coarse_gaussian_gemm_requested_targets,
            targets_in_scope=coarse_gaussian_gemm_diagnostic_targets,
            captured_target_counts=coarse_gaussian_gemm_diagnostic_target_counts,
            artifact_paths=coarse_gaussian_gemm_diagnostic_paths,
        )
    if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
        if len(coarse_gaussian_gemm_stream_original_indices) != n_images:
            raise RuntimeError(
                "coarse GEMM streaming diagnostic did not capture every particle "
                f"in its call scope: {len(coarse_gaussian_gemm_stream_original_indices)} "
                f"!= {n_images}",
            )
        (
            coarse_gaussian_gemm_stream_scope_manifest_path,
            coarse_gaussian_gemm_stream_aggregate_manifest_path,
        ) = _seal_coarse_gaussian_gemm_streaming_scope(
            coarse_gaussian_gemm_stream_diagnostic_dir,
            scope=coarse_gaussian_gemm_diagnostic_scope,
            retained_topk=coarse_gaussian_gemm_stream_topk,
            artifact_paths=coarse_gaussian_gemm_stream_paths,
            original_indices=coarse_gaussian_gemm_stream_original_indices,
        )
    full_stats = {
        "normalization_log_z": normalization_log_z,
        "normalization_log_evidence": normalization_log_evidence,
        "log_evidence_per_image": log_evidence,
        "best_log_score_per_image": best_log_score,
        "max_posterior_per_image": max_posterior,
        "class_log_evidence_per_image": class_log_evidence,
        "class_assignments": class_assignment,
        # RELION serializes the cutoff rank before inclusive threshold ties
        # expand the pass-2/M-step support represented by ``n_sig_all``.
        "significant_cutoff_counts": cutoff_count_all,
        "coarse_selector_audit": coarse_selector_audit,
    }
    if coarse_gaussian_square_layout is not None:
        full_stats["coarse_gaussian_square_layout"] = {
            "stable_fourier_window_shapes_requested": bool(
                stable_fourier_window_shapes
            ),
            "stable_fourier_window_shapes_effective": bool(
                stable_fourier_window_shapes
                and coarse_gaussian_square_layout.physical_current_size
                != coarse_gaussian_square_layout.logical_current_size
            ),
            "logical_current_size": int(
                coarse_gaussian_square_layout.logical_current_size
            ),
            "physical_current_size": int(
                coarse_gaussian_square_layout.physical_current_size
            ),
            "logical_square_pixels": int(
                coarse_gaussian_square_layout.logical_square_count
            ),
            "physical_square_pixels": int(
                coarse_gaussian_square_layout.physical_square_count
            ),
            "executed_square_pixels": int(
                coarse_gaussian_square_layout.logical_square_count
                if stable_fourier_window_shapes
                else coarse_gaussian_square_layout.physical_square_count
            ),
            "logical_issue_stream_is_prefix": True,
            "physical_tail_zero_weighted": True,
            "physical_tail_skipped_by_runtime_count": bool(
                stable_fourier_window_shapes
                and coarse_gaussian_square_layout.physical_square_count
                != coarse_gaussian_square_layout.logical_square_count
            ),
        }
    if exact_coarse_assembly_profile_enabled:
        full_stats["exact_coarse_operand_assembly"] = {
            "skip_generic_default_enabled": False,
            "skip_generic_requested": bool(
                exact_coarse_skip_generic_operands_requested,
            ),
            "skip_generic_effective": bool(
                exact_coarse_skip_generic_operands_enabled,
            ),
            "exact_coarse_operands_effective": bool(exact_coarse_operands_enabled),
            "exact_compact_preprocess_default_enabled": False,
            "exact_compact_preprocess_requested": bool(
                exact_compact_preprocess_requested,
            ),
            "exact_compact_preprocess_effective": bool(
                exact_compact_preprocess_enabled,
            ),
            "generic_score_preprocess_count": int(generic_score_preprocess_count),
            "exact_source_preprocess_count": int(exact_source_preprocess_count),
            "generic_ctf_evaluation_count": int(generic_score_preprocess_count),
            "generic_full_translation_count": int(generic_score_preprocess_count),
            "generic_assembly_count": int(generic_coarse_operand_assembly_count),
            "exact_assembly_count": int(exact_coarse_operand_assembly_count),
            "translate_score_call_site_count": int(
                generic_coarse_operand_assembly_count
                * int(coarse_gaussian_sincosf_enabled)
                + exact_coarse_operand_assembly_count
            ),
            "translate_score_call_count": int(
                generic_coarse_operand_assembly_count
                * int(coarse_gaussian_sincosf_enabled)
                + exact_coarse_operand_assembly_count
            ),
            "downstream_operand_source": (
                "exact_source_star"
                if exact_coarse_operands_enabled
                else "generic"
            ),
            "diagnostic_operand_source": (
                "exact_source_star"
                if exact_coarse_operands_enabled
                else "generic"
            ),
            "raw_score_capture_changed": False,
            "generic_fallback_policy": (
                "exact_full_direct_scores_available"
                if exact_compact_preprocess_enabled
                else "available"
            ),
            "skipped_generic_outputs": (
                [
                    "coarse_gaussian_shifted_corrected",
                    "coarse_gaussian_pixel_weight",
                    "coarse_gaussian_unshifted_corrected",
                ]
                if exact_coarse_skip_generic_operands_enabled
                else []
            ),
        }
    if coarse_gaussian_gemm_resource_estimate is not None:
        full_stats["coarse_gaussian_gemm_resources"] = {
            field: int(value)
            for field, value in coarse_gaussian_gemm_resource_estimate._asdict().items()
        }
        paired_capture_requested = bool(
            coarse_gaussian_gemm_diagnostic_dir is not None
            or coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        paired_capture_active = bool(
            coarse_gaussian_gemm_diagnostic_targets
            or coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        full_stats["coarse_gaussian_gemm_qualification"] = {
            "paired_capture_requested": paired_capture_requested,
            "paired_capture_active": paired_capture_active,
            "clean_timing_eligible": not paired_capture_requested,
            "numerical_qualification_status": "NO_GO_UNQUALIFIED",
            "requires_bitwise_score_identity": False,
            "requires_exact_discrete_identity": True,
            "timing_policy": (
                "paired capture executes both scorers; use a separate "
                "diagnostic-off timing arm"
            ),
            "numerical_policy": (
                "mathematically equivalent score noise may pass only when "
                "repeat-stable, unbiased/non-directional, bounded/non-growing, "
                "discrete-identical, basin/quality-neutral, and materially faster; "
                "scale amplification or negative implied diff2 is NO-GO"
            ),
        }
    if coarse_gaussian_gemm_projection_cache_plan is not None:
        full_stats["coarse_gaussian_gemm_projection_cache"] = (
            _coarse_gaussian_gemm_projection_cache_stats(
                coarse_gaussian_gemm_projection_cache_plan,
                enabled=coarse_gaussian_gemm_projection_cache is not None,
            )
        )
    if coarse_gaussian_gemm_hybrid_requested:
        selected_candidate_count = (
            coarse_gaussian_gemm_hybrid_selected_block_count
            * SOURCE_ROTATION_BLOCK_SIZE
            * n_trans
        )
        full_selected_image_candidate_count = (
            coarse_gaussian_gemm_hybrid_selected_image_count * n_rot * n_trans
        )
        compact_table_capacity = (
            coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates
        )
        dense_table_capacity = (
            coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates
        )
        full_stats["coarse_gaussian_gemm_hybrid"] = {
            "enabled": True,
            "default_enabled": False,
            "coarse_square_layout": dict(
                full_stats["coarse_gaussian_square_layout"]
            ),
            "compact_posterior_enabled": bool(
                coarse_gaussian_gemm_compact_posterior_requested,
            ),
            "compact_posterior_default_enabled": False,
            "selected_score_layout": (
                "fixed_capacity_source16"
                if coarse_gaussian_gemm_compact_posterior_requested
                else "dense_global"
            ),
            "positive_only_scan_role": (
                "correctness_oracle_not_runtime"
                if coarse_gaussian_gemm_compact_posterior_requested
                else None
            ),
            "published_score_source": (
                "exact_relion_source16_or_full_fused_projector"
                if coarse_gaussian_fused_full_fallback_armed
                else "exact_relion_source16_or_full_rectangular"
            ),
            "expanded_gemm_scores_published": False,
            "whole_batch_fail_closed_fallback": not coarse_row_partition_requested,
            "full_fallback_backend_requested": (
                "fused_projector"
                if coarse_fused_projector_requested
                else "rectangular"
            ),
            "full_fallback_backend_armed": (
                "fused_projector"
                if coarse_gaussian_fused_full_fallback_armed
                else "rectangular"
            ),
            "full_fallback_backend_effective": (
                None
                if not coarse_gaussian_gemm_hybrid_full_dense_batch_count
                else (
                    "fused_projector"
                    if coarse_gaussian_gemm_hybrid_fused_full_batch_count
                    else "rectangular"
                )
            ),
            "all_full_dense_batches_used_fused": (
                None
                if not coarse_gaussian_gemm_hybrid_full_dense_batch_count
                else (
                    coarse_gaussian_gemm_hybrid_fused_full_batch_count
                    == coarse_gaussian_gemm_hybrid_full_dense_batch_count
                )
            ),
            "full_dense_kernel_calls": dict(coarse_full_kernel_calls),
            "full_dense_kernel_images": dict(coarse_full_kernel_images),
            "full_dense_batch_count": int(
                coarse_gaussian_gemm_hybrid_full_dense_batch_count,
            ),
            "full_dense_image_count": int(
                coarse_gaussian_gemm_hybrid_full_dense_image_count,
            ),
            "fused_full_fallback_batch_count": int(
                coarse_gaussian_gemm_hybrid_fused_full_batch_count,
            ),
            "fused_full_fallback_image_count": int(
                coarse_gaussian_gemm_hybrid_fused_full_image_count,
            ),
            "rectangular_full_fallback_batch_count": int(
                coarse_gaussian_gemm_hybrid_rectangular_full_batch_count,
            ),
            "rectangular_full_fallback_image_count": int(
                coarse_gaussian_gemm_hybrid_rectangular_full_image_count,
            ),
            "score_representation_policy": (
                "compact_only_when_fixed_physical_capacity_is_smaller_than_dense"
            ),
            "static_preferred_score_representation": (
                _select_coarse_gaussian_gemm_score_representation(
                    n_rotations=n_rot,
                    block_capacity=coarse_gaussian_gemm_hybrid_capacity,
                    compact_posterior=(
                        coarse_gaussian_gemm_compact_posterior_requested
                    ),
                )
            ),
            "score_representation_batch_counts": dict(
                sorted(
                    coarse_gaussian_gemm_hybrid_score_representation_batch_counts.items(),
                ),
            ),
            "batch_count": int(coarse_gaussian_gemm_hybrid_batch_count),
            "actual_image_batch_sizes": list(
                coarse_gaussian_gemm_hybrid_actual_image_batch_sizes,
            ),
            "physical_image_batch_sizes": list(
                coarse_gaussian_gemm_hybrid_physical_image_batch_sizes,
            ),
            "input_image_batch_size": int(input_image_batch_size),
            "requested_hybrid_image_batch_size": (
                None
                if coarse_gaussian_gemm_hybrid_image_batch_size_request is None
                else int(coarse_gaussian_gemm_hybrid_image_batch_size_request)
            ),
            "effective_image_batch_size": int(image_batch_size),
            "streamed_certificate_candidate_count_at_effective_batch": int(
                image_batch_size
                * coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                * n_trans
            ),
            "selected_rescore_batch_count": int(
                coarse_gaussian_gemm_hybrid_selected_batch_count,
            ),
            "static_dense_batch_count": int(
                coarse_gaussian_gemm_hybrid_static_dense_batch_count,
            ),
            "fallback_batch_count": int(
                coarse_gaussian_gemm_hybrid_fallback_batch_count,
            ),
            "selected_rescore_image_count": int(
                coarse_gaussian_gemm_hybrid_selected_image_count,
            ),
            "static_dense_image_count": int(
                coarse_gaussian_gemm_hybrid_static_dense_image_count,
            ),
            "fallback_image_count": int(
                coarse_gaussian_gemm_hybrid_fallback_image_count,
            ),
            "overflow_latch_scope": (
                "current_significance_call_exact_geometry_and_capacity"
            ),
            "overflow_latch_active_at_return": bool(
                coarse_gaussian_gemm_hybrid_overflow_latched,
            ),
            "overflow_latch_activation_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_activation_count,
            ),
            "overflow_latch_static_dense_batch_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count,
            ),
            "overflow_latch_static_dense_image_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count,
            ),
            "selected_source16_block_count": int(
                coarse_gaussian_gemm_hybrid_selected_block_count,
            ),
            "selected_exact_candidate_count": int(selected_candidate_count),
            "full_candidate_count_for_selected_images": int(
                full_selected_image_candidate_count,
            ),
            "selected_exact_candidate_fraction": (
                float(selected_candidate_count / full_selected_image_candidate_count)
                if full_selected_image_candidate_count
                else None
            ),
            "selected_score_table_capacity_candidates": int(
                compact_table_capacity,
            ),
            "dense_global_score_table_capacity_candidates": int(
                dense_table_capacity,
            ),
            "selected_score_table_capacity_bytes_f32": int(
                compact_table_capacity * np.dtype(np.float32).itemsize,
            ),
            "dense_global_score_table_capacity_bytes_f32": int(
                dense_table_capacity * np.dtype(np.float32).itemsize,
            ),
            "selected_to_dense_score_table_capacity_fraction": (
                float(compact_table_capacity / dense_table_capacity)
                if dense_table_capacity
                else None
            ),
            "static_compact_to_dense_capacity_fraction": float(
                coarse_gaussian_gemm_hybrid_capacity
                * SOURCE_ROTATION_BLOCK_SIZE
                / n_rot
            ),
            "max_selected_blocks_per_image": int(
                coarse_gaussian_gemm_hybrid_max_blocks_per_image,
            ),
            "selected_block_capacity": int(
                coarse_gaussian_gemm_hybrid_capacity,
            ),
            "certificate_chunk_rows": int(
                coarse_gaussian_gemm_projection_cache_plan.chunk_rows,
            ),
            "certificate_chunk_count_per_batch": int(
                coarse_gaussian_gemm_projection_cache_plan.chunk_count_per_table,
            ),
            "topology_full_to_compact_sha256": (
                coarse_gaussian_gemm_certificate_topology.full_to_compact_sha256
            ),
            "fallback_reasons": dict(
                sorted(coarse_gaussian_gemm_hybrid_fallback_reasons.items()),
            ),
        }
    if coarse_gaussian_gemm_diagnostic_paths:
        full_stats["coarse_gaussian_gemm_diagnostic_paths"] = tuple(
            coarse_gaussian_gemm_diagnostic_paths,
        )
    if coarse_gaussian_gemm_scope_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_scope_manifest_path"] = (
            coarse_gaussian_gemm_scope_manifest_path
        )
    if coarse_gaussian_gemm_aggregate_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_aggregate_manifest_path"] = (
            coarse_gaussian_gemm_aggregate_manifest_path
        )
    if coarse_gaussian_gemm_stream_paths:
        full_stats["coarse_gaussian_gemm_stream_diagnostic"] = {
            "artifact_paths": tuple(coarse_gaussian_gemm_stream_paths),
            "retained_topk": int(coarse_gaussian_gemm_stream_topk),
            "persistent_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_dual_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "posterior_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "pre_prior_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "stores_score_cube": False,
            "clean_timing_eligible": False,
            "production_behavior_changed": False,
        }
    if coarse_gaussian_gemm_stream_scope_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_stream_scope_manifest_path"] = (
            coarse_gaussian_gemm_stream_scope_manifest_path
        )
    if coarse_gaussian_gemm_stream_aggregate_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_stream_aggregate_manifest_path"] = (
            coarse_gaussian_gemm_stream_aggregate_manifest_path
        )
    if _coarse_significance_support_audit_enabled():
        if significant_sample_indices is None:
            raise RuntimeError(
                f"{_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}=1 requires "
                "collect_significance=True",
            )
        full_stats["coarse_significance_support_audit"] = (
            _build_coarse_significance_support_audit(
                significant_sample_indices,
                samples_per_class=n_rot * n_trans,
                include_ids=_coarse_significance_support_audit_ids_enabled(),
            )
        )
    if coarse_row_partition_requested:
        full_stats["coarse_gaussian_gemm_hybrid"]["row_partition"] = {
            "requested": True,
            "default_enabled": False,
            "mixed_input_batch_count": coarse_partition_mixed_batch_count,
            "transaction_actual_group_sizes": coarse_partition_actual_group_sizes,
            "transaction_physical_group_sizes": coarse_partition_physical_group_sizes,
            "input_batch_count": coarse_gaussian_gemm_hybrid_batch_count,
            "execution_group_count": coarse_gaussian_gemm_hybrid_selected_batch_count + coarse_gaussian_gemm_hybrid_full_dense_batch_count,
            "invalid_certificate_whole_batch_fallback": True,
            "overflow_latch_policy": "all_actual_rows_overflow",
        }
    if coarse_cuda_posterior_requested:
        full_stats["coarse_gaussian_gemm_hybrid"]["posterior_transaction"] = {
            "requested": True,
            "default_enabled": False,
            "backend": "cuda",
            "input_batch_count": coarse_cuda_posterior_batch_count,
            "execution_group_count": coarse_cuda_posterior_group_count,
            "actual_image_count": coarse_cuda_posterior_image_count,
        }
    if relion_f32_sum_weight is not None:
        # RELION's oversampling-zero second pass deliberately reuses this
        # coarse, maximum-shifted float32 denominator numerically.  It is not
        # interchangeable with a log-evidence value because the fine pass
        # independently shifts its own maximum to 50 before division.
        full_stats["relion_f32_sum_weight"] = relion_f32_sum_weight
    if return_class_best:
        full_stats["class_best_log_score_per_image"] = class_best_log_score
        full_stats["class_best_offset_free_log_score_per_image"] = class_best_offset_free_log_score
        full_stats["class_hard_assignments"] = class_hard_assignment
    if return_class_second:
        full_stats["class_second_best_log_score_per_image"] = class_second_best_log_score
        full_stats["class_second_hard_assignments"] = class_second_hard_assignment
        full_stats["class_second_best_offset_free_log_score_per_image"] = class_second_best_offset_free_log_score
    if tree_rescore_enabled:
        full_stats["firstiter_cc_tree_top2_rescore"] = {
            "max_margin": float(tree_rescore_max_margin),
            "examined_images": int(tree_rescore_examined),
            "ambiguous_images": int(tree_rescore_ambiguous),
            "exact_score_ties": int(tree_rescore_exact_ties),
            "winner_changes": int(tree_rescore_winner_changes),
        }
        logger.warning(
            "RELION coarse-tree top-2 rescore complete: "
            "examined=%d ambiguous=%d exact_ties=%d winner_changes=%d",
            tree_rescore_examined,
            tree_rescore_ambiguous,
            tree_rescore_exact_ties,
            tree_rescore_winner_changes,
        )
    return sig_rot_any, n_sig_all, hard_assignment, class_assignment, significant_sample_indices, full_stats
