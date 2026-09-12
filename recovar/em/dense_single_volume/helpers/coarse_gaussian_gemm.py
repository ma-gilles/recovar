"""Coarse Gaussian GEMM backend of the K-class significance pass.

The env-gated GEMM macro and its opt-ins, the device-memory budgets and
projection cache, the hybrid image-batch plan and its request validation,
and the hybrid batch itself. ``significance`` selects this backend once per
pass and calls it per image batch.
"""

import operator
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import projection_cache as projection_cache_helpers
from recovar.em.dense_single_volume.helpers.coarse_device_selection import DeviceCoarseBlockSelection
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    SOURCE_ROTATION_BLOCK_SIZE,
    CoarseGemmHybridBlockSelection,
    CoarseGemmHybridCompactScores,
    assemble_coarse_gemm_hybrid_compact_scores_f32,
    assemble_coarse_gemm_hybrid_dense_scores_f32,
    initialize_coarse_gemm_hybrid_interval_state,
    select_coarse_gemm_hybrid_rotation_blocks,
)
from recovar.em.dense_single_volume.helpers.scoring import (
    _prepare_relion_coarse_gaussian_gemm_f64_image_batch,
    _relion_coarse_diff2_rotation_blocks_from_topology_f32,
    _relion_coarse_gaussian_gemm_scores,
    _relion_coarse_gaussian_gemm_update_certificate_state,
)

_COARSE_GAUSSIAN_GEMM_MACRO_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"


_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB = 4.0


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS = 4_608


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT = 16


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB = 13_332_001


_COARSE_GAUSSIAN_GEMM_HYBRID_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID"


_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
)


_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"
)


_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION"
)


_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE"
)


_K1_RELION_EXACT_COARSE_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS"


_K1_RELION_F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"


def _coarse_gaussian_gemm_macro_enabled(*, default: bool = False) -> bool:
    """Whether one coarse projection feeds the shared multi-image GEMMs.

    The exact-arithmetic objective is unchanged, but expanding the direct
    square changes operation order and can amplify cancellation.  Keep it
    default-off until same-H100 repeats establish stable numerical noise,
    unchanged discrete choices/support and quality, and a material end-to-end
    speedup.
    """

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_MACRO_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_projection_cache_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the explicit call-scoped coarse-projection cache toggle."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_hybrid_enabled(*, default: bool = False) -> bool:
    """Resolve the default-off certified GEMM/exact-source16 hybrid."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_HYBRID_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_compact_posterior_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the experimental fixed-capacity selected-score posterior."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_device_transaction_enabled() -> bool:
    """Opt into device certificate/selection/exact rescore composition."""
    token = os.environ.get(_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION_ENV, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(
            f"Unsupported {_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION_ENV}={token!r}",
        )
    return token == "1"


def _coarse_shared_pretranslated_enabled() -> bool:
    """Opt into shared staging only for full runtime-prefix coarse scoring."""
    name = "RECOVAR_COARSE_SHARED_PRETRANSLATED"
    token = os.environ.get(name, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"Unsupported {name}={token!r}")
    return token == "1"


def _coarse_gaussian_gemm_real_cross_enabled() -> bool:
    """Use FP64 real-component GEMM only in the coarse certificate."""
    name = "RECOVAR_COARSE_GAUSSIAN_GEMM_REAL_CROSS"
    token = os.environ.get(name, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"Unsupported {name}={token!r}")
    return token == "1"


def _coarse_gaussian_gemm_hybrid_image_batch_size_request() -> int | None:
    """Return the explicit compact-hybrid image-batch override, if any."""

    token = os.environ.get(_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV)
    if token is None:
        return None
    token = token.strip()
    try:
        batch_size = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {token!r}",
        ) from error
    if batch_size <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {token!r}",
        )
    return batch_size


def _resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
    input_batch_size: int,
    *,
    requested_batch_size: int | None,
    n_images: int,
    certificate_chunk_rows: int,
    n_translations: int,
    hybrid_enabled: bool,
    compact_posterior_enabled: bool,
    score_float_budget: int | None = None,
) -> int:
    """Resolve a guarded shared image batch for the compact score hybrid.

    InitialModel currently caps pass-1 batches against the full
    ``B*R*T`` score cube.  The compact hybrid never materializes that cube: its
    largest score-dependent tile is streamed over ``certificate_chunk_rows``.
    An explicit override can therefore coalesce logical image batches while
    retaining the mature shared matrix-matrix scorer and every per-image
    arithmetic/reduction order.
    """

    input_size = operator.index(input_batch_size)
    image_count = operator.index(n_images)
    chunk_rows = operator.index(certificate_chunk_rows)
    translation_count = operator.index(n_translations)
    if min(input_size, image_count, chunk_rows, translation_count) <= 0:
        raise ValueError("compact-hybrid image-batch dimensions must be positive")
    if requested_batch_size is None:
        return input_size
    requested = operator.index(requested_batch_size)
    prefix = f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} requires"
    if not hybrid_enabled:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if not compact_posterior_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
        )
    if requested <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {requested!r}",
        )
    if score_float_budget is None:
        from recovar.em.dense_single_volume.batch_planning import (
            RELION_SCORE_TENSOR_FLOAT_BUDGET,
        )

        score_float_budget = RELION_SCORE_TENSOR_FLOAT_BUDGET
    score_budget = operator.index(score_float_budget)
    if score_budget <= 0:
        raise ValueError("compact-hybrid score-float budget must be positive")

    effective = min(requested, image_count)
    streamed_candidate_count = effective * chunk_rows * translation_count
    if streamed_candidate_count > score_budget:
        raise MemoryError(
            "compact-hybrid image-batch override exceeds the mature EM score "
            f"tile budget: {streamed_candidate_count} > {score_budget} floats",
        )
    return effective


def _validate_coarse_gaussian_gemm_compact_posterior_request(
    *,
    hybrid_enabled: bool,
    return_class_best: bool,
    return_class_second: bool,
) -> None:
    """Keep the first compact-posterior boundary narrow and fail closed."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1 requires"
    if not hybrid_enabled:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if return_class_best or return_class_second:
        raise ValueError(
            f"{prefix} return_class_best=False and return_class_second=False",
        )


def _coarse_gaussian_gemm_hybrid_block_capacity(
    *,
    default: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
) -> int:
    """Return the fixed selected-source16 capacity for one image row."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV,
        str(default),
    ).strip()
    try:
        capacity = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV} must be a "
            f"positive integer, got {token!r}",
        ) from error
    if capacity <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV} must be a "
            f"positive integer, got {token!r}",
        )
    return capacity


class CoarseGaussianGemmResources(NamedTuple):
    """Conservative device-transient accounting for one GEMM rotation block."""

    full_centered_projection_bytes: int
    compact_projection_bytes: int
    compact_projection_abs2_bytes: int
    predicted_peak_projection_bytes: int
    projected_transient_budget_bytes: int
    pixel_index_device_to_host_materializations: int


def _coarse_gaussian_gemm_projected_transient_budget_bytes(
    *,
    default_gb: float = 2.0,
) -> int:
    """Return the explicit conservative projection-transient budget."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


def _coarse_gaussian_gemm_projection_cache_budget_bytes(
    *,
    default_gb: float = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB,
) -> int:
    """Return the explicit conservative call-scoped cache budget."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


def _coarse_gaussian_gemm_resources(
    *,
    rotation_block_size: int,
    image_shape,
    compact_pixel_count: int,
    budget_bytes: int,
) -> CoarseGaussianGemmResources:
    """Estimate and gate projection buffers before the first GPU launch.

    The mature texture projector currently creates a full physical centered
    half-image before gathering the current-size compact pixels.  Count that
    buffer plus the compact complex projection and its real abs2 companion.
    The pixel-index table is deliberately kept as a NumPy host array, so this
    path performs no device-to-host index materialization.
    """

    block_size = int(rotation_block_size)
    image_height, image_width = (int(value) for value in image_shape)
    compact_count = int(compact_pixel_count)
    if block_size <= 0 or image_height <= 0 or image_width <= 0 or compact_count <= 0:
        raise ValueError("coarse GEMM resource dimensions must all be positive")
    full_count = image_height * (image_width // 2 + 1)
    full_bytes = block_size * full_count * np.dtype(np.complex64).itemsize
    compact_bytes = block_size * compact_count * np.dtype(np.complex64).itemsize
    abs2_bytes = block_size * compact_count * np.dtype(np.float32).itemsize
    predicted_peak = full_bytes + compact_bytes + abs2_bytes
    resources = CoarseGaussianGemmResources(
        full_centered_projection_bytes=int(full_bytes),
        compact_projection_bytes=int(compact_bytes),
        compact_projection_abs2_bytes=int(abs2_bytes),
        predicted_peak_projection_bytes=int(predicted_peak),
        projected_transient_budget_bytes=int(budget_bytes),
        pixel_index_device_to_host_materializations=0,
    )
    if resources.predicted_peak_projection_bytes > resources.projected_transient_budget_bytes:
        raise MemoryError(
            "coarse GEMM predicted projection transient exceeds "
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV}: "
            f"{resources.predicted_peak_projection_bytes} > "
            f"{resources.projected_transient_budget_bytes} bytes",
        )
    return resources


def _validate_coarse_gaussian_gemm_projection_cache_request(
    *,
    macro_enabled: bool,
    n_classes: int,
    n_rotations: int,
    coarse_gaussian_ffi_enabled: bool,
    exact_coarse_operands_enabled: bool,
    use_relion_projector: bool,
    relion_texture_interp_enabled: bool,
    half_spectrum_scoring: bool,
    use_float64_scoring: bool,
    relion_projector_dtype,
) -> None:
    """Fail closed unless the first cache seam's exact K=1 contract holds."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1 requires"
    if not macro_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
        )
    if int(n_classes) != 1:
        raise ValueError(f"{prefix} K=1, got K={int(n_classes)}")
    if int(n_rotations) <= 0 or int(n_rotations) % int(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT
    ):
        raise ValueError(
            f"{prefix} a positive rotation count divisible by "
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT}, "
            f"got {int(n_rotations)}",
        )
    if not coarse_gaussian_ffi_enabled:
        raise ValueError(
            f"{prefix} the exact RELION coarse Gaussian FFI path",
        )
    if not exact_coarse_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if not use_relion_projector or not relion_texture_interp_enabled:
        raise ValueError(
            f"{prefix} the supplied RELION texture projector",
        )
    if not half_spectrum_scoring:
        raise ValueError(f"{prefix} half-spectrum scoring")
    if use_float64_scoring:
        raise ValueError(f"{prefix} production float32/complex64 scoring")
    if relion_projector_dtype is None or np.dtype(relion_projector_dtype) != np.dtype(
        np.complex64
    ):
        raise TypeError(
            f"{prefix} a complex64 RELION projector, got "
            f"{relion_projector_dtype}",
        )


def _plan_coarse_gaussian_gemm_projection_cache(
    *,
    n_rotations: int,
    compact_pixel_count: int,
    image_shape,
    budget_bytes: int,
) -> projection_cache_helpers.ProjectionCachePlan:
    """Plan one conservative C64 K=1 cache in qualified 4,608-row chunks."""

    image_height, image_width = (int(value) for value in image_shape)
    return projection_cache_helpers.plan_projection_cache(
        table_count=1,
        row_count=int(n_rotations),
        pixel_count=int(compact_pixel_count),
        cache_dtype=np.complex64,
        requested_max_chunk_rows=(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        row_alignment=_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT,
        transient_specs=(
            projection_cache_helpers.ProjectionCacheTransientSpec(
                name="full_centered_projection",
                elements_per_row=image_height * (image_width // 2 + 1),
                dtype=np.complex64,
            ),
        ),
        budget_bytes=int(budget_bytes),
        # Job 13332001 observed donation aliasing on one H100 lowering.  The
        # production admission remains conservative and correct if that
        # informational hardware-specific observation does not generalize.
        destination_alias_proven=False,
    )


def _build_coarse_gaussian_gemm_projection_cache(plan, project_block):
    """Build one private call-scoped cache through the shared exact builder."""

    return projection_cache_helpers.build_projection_cache(plan, project_block)


def _coarse_gaussian_gemm_projection_cache_stats(plan, *, enabled: bool):
    """Describe conservative admission and narrowly scoped alias evidence."""

    h100_alias_evidence_applies = bool(
        plan.cache_shape == (1, 36_864, 5_100)
        and plan.chunk_rows
        == _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
    )
    return {
        "enabled": bool(enabled),
        "scope": "one significance call",
        "cache_shape": tuple(int(value) for value in plan.cache_shape),
        "cache_dtype": plan.cache_dtype.name,
        "stores_projection_abs2": False,
        "chunk_rows": int(plan.chunk_rows),
        "chunk_count": int(plan.chunk_count_per_table),
        "retained_bytes": int(plan.retained_bytes),
        "conservative_predicted_peak_bytes": int(plan.predicted_peak_bytes),
        "budget_bytes": int(plan.budget_bytes),
        "admission_destination_alias_proven": bool(
            plan.destination_alias_proven
        ),
        # Informational only: deterministic job 13332001 observed donated
        # insert aliasing for exactly (1, 36864, 5100) with 4608-row chunks
        # on one H100. Admission always reserves a non-aliased copy.
        "h100_alias_evidence_applies_to_plan": h100_alias_evidence_applies,
        "h100_alias_evidence_cache_shape": (1, 36_864, 5_100),
        "h100_alias_evidence_chunk_rows": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        "h100_observed_donated_insert_alias": (
            True if h100_alias_evidence_applies else None
        ),
        "h100_alias_evidence_job_id": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB
        ),
        "h100_observed_alias_peak_bytes": (
            int(plan.predicted_peak_bytes - plan.destination_copy_bytes)
            if h100_alias_evidence_applies
            else None
        ),
        "h100_alias_evidence_used_for_admission": False,
    }


def _project_coarse_gaussian_gemm_projection_cache_block_once(
    cache,
    class_index,
    mean_for_proj,
    rotations_block,
    *,
    rotation_start: int,
):
    """Serve one existing macro block from C64 cache without reprojection."""

    del mean_for_proj
    cache = jnp.asarray(cache)
    if cache.ndim != 3 or np.dtype(cache.dtype) != np.dtype(np.complex64):
        raise TypeError(
            "coarse GEMM projection cache must have shape "
            "(table, rotation, pixel) and dtype complex64",
        )
    table_index = int(class_index)
    if table_index < 0 or table_index >= int(cache.shape[0]):
        raise IndexError(
            f"coarse GEMM projection-cache table {table_index} is out of range",
        )
    start = int(rotation_start)
    requested_rows = int(rotations_block.shape[0])
    if start < 0 or requested_rows <= 0 or start >= int(cache.shape[1]):
        raise IndexError(
            "coarse GEMM projection-cache block must start inside the cache "
            "and contain at least one row",
        )
    stop = min(start + requested_rows, int(cache.shape[1]))
    projected_reference = cache[table_index, start:stop]
    padding_rows = requested_rows - int(projected_reference.shape[0])
    if padding_rows:
        # The shared significance loop masks these physical tail rows to -inf.
        # Zero padding preserves its fixed score-block shape without projecting
        # synthetic identity rotations or changing any valid cached row.
        projected_reference = jnp.pad(
            projected_reference,
            ((0, padding_rows), (0, 0)),
        )
    # This is the same C64 expression used by the mature projector callback.
    # The promoted certificate deliberately ignores this companion and forms
    # its two component squares explicitly after conversion to FP64.
    projected_reference_abs2 = jnp.abs(projected_reference) ** 2
    return projected_reference, projected_reference_abs2


def _validate_coarse_gaussian_gemm_hybrid_request(
    *,
    macro_enabled: bool,
    projection_cache_enabled: bool,
    n_classes: int,
    n_rotations: int,
    score_mode: str,
    coarse_gaussian_ffi_enabled: bool,
    exact_coarse_operands_enabled: bool,
    relion_f32_coarse_support_enabled: bool,
    collect_significance: bool,
    any_diagnostic_requested: bool,
) -> None:
    """Fail closed unless the first production hybrid contract is complete."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1 requires"
    if not macro_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
        )
    if not projection_cache_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1",
        )
    if int(n_classes) != 1:
        raise ValueError(f"{prefix} K=1, got K={int(n_classes)}")
    if int(n_rotations) <= 0 or int(n_rotations) % SOURCE_ROTATION_BLOCK_SIZE:
        raise ValueError(
            f"{prefix} a rotation count divisible by "
            f"{SOURCE_ROTATION_BLOCK_SIZE}, got {int(n_rotations)}",
        )
    if score_mode != "gaussian":
        raise ValueError(f"{prefix} score_mode='gaussian'")
    if not coarse_gaussian_ffi_enabled:
        raise ValueError(f"{prefix} the exact RELION coarse Gaussian FFI path")
    if not exact_coarse_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if not relion_f32_coarse_support_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_F32_COARSE_SUPPORT_ENV}=1",
        )
    if not collect_significance:
        raise ValueError(f"{prefix} collect_significance=True")
    if any_diagnostic_requested:
        raise ValueError(
            f"{prefix} paired coarse GEMM diagnostics to be disabled",
        )


class CoarseGaussianGemmHybridBatchResult(NamedTuple):
    """One batch's exact selected score layout or full-direct fallback."""

    scores: jax.Array | None
    raw_score_max: jax.Array
    scores_include_priors: bool
    used_selected_rescore: bool
    fallback_reason: str | None
    selection: CoarseGemmHybridBlockSelection | DeviceCoarseBlockSelection
    compact_scores: CoarseGemmHybridCompactScores | None = None
    score_representation: str = "compact_selected_exact"
    diagnostic_selected_diff2: jax.Array | None = None
    full_dense_backend: str | None = None
    selected_block_count: int | None = None
    max_selected_blocks: int | None = None
    full_dense_kernel: str | None = None


def _select_coarse_gaussian_gemm_score_representation(
    *,
    n_rotations: int,
    block_capacity: int,
    compact_posterior: bool,
) -> str:
    """Choose the smaller physical exact-score layout before certification."""

    n_rotations = operator.index(n_rotations)
    block_capacity = operator.index(block_capacity)
    if n_rotations <= 0 or block_capacity <= 0:
        raise ValueError("hybrid score-layout dimensions must be positive")
    if (
        compact_posterior
        and block_capacity * SOURCE_ROTATION_BLOCK_SIZE >= n_rotations
    ):
        return "dense_full_direct_static_capacity"
    if compact_posterior:
        return "compact_selected_exact"
    return "dense_selected_exact"


def _compute_coarse_gaussian_gemm_hybrid_batch(
    projection_cache,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    *,
    topology,
    actual_image_count: int,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    certificate_chunk_rows: int = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS,
    block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
    compact_posterior: bool = False,
    force_static_dense_after_overflow: bool = False,
    logical_full_pixel_count=None,
    capture_selected_diff2: bool = False,
    full_dense_diff2_fn=None,
    device_transaction: bool = False,
    real_cross: bool = False,
) -> CoarseGaussianGemmHybridBatchResult:
    """Certify, exactly rescore, and restore one K=1 coarse score table.

    The expanded FP64 GEMMs only choose complete source-16 rotation blocks.
    Published values always come from the mature direct CUDA arithmetic.  Any
    incomplete certificate, capacity overflow, or invalid selected output
    routes the whole padded image batch through one lazy full-direct call.
    The default remains the rectangular scorer; callers may supply the mature
    fused projector/scorer without paying for it on selected-rescore batches.
    """

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
        validate_coarse_gemm_certificate_topology,
    )

    validate_coarse_gemm_certificate_topology(topology)
    shared_pretranslated = _coarse_shared_pretranslated_enabled()
    cache = jnp.asarray(projection_cache)
    shifted = jnp.asarray(shifted_corrected)
    weight = jnp.asarray(pixel_weight)
    initial = jnp.asarray(initial_diff2)
    if cache.ndim != 3 or tuple(cache.shape[:1]) != (1,):
        raise ValueError(
            "certified coarse GEMM hybrid requires one [1,R,F] projection table",
        )
    if cache.dtype != jnp.complex64:
        raise TypeError("certified coarse GEMM hybrid projection cache must be complex64")
    if shifted.ndim != 3:
        raise ValueError(
            "certified coarse GEMM hybrid shifted images must have shape [B,T,F]",
        )
    batch_size, n_translations, n_pixels = map(int, shifted.shape)
    n_rotations = int(cache.shape[1])
    if (
        shifted.dtype != jnp.complex64
        or weight.dtype != jnp.float32
        or initial.dtype != jnp.float32
        or tuple(weight.shape) != (batch_size, n_pixels)
        or tuple(initial.shape) != (batch_size,)
        or int(cache.shape[2]) != n_pixels
        or n_rotations <= 0
        or n_rotations % SOURCE_ROTATION_BLOCK_SIZE
        or topology.compact_pixel_count != n_pixels
        or topology.translation_count != n_translations
    ):
        raise ValueError(
            "certified coarse GEMM hybrid operands have inconsistent shapes, "
            "dtypes, or topology",
        )
    try:
        chunk_rows = operator.index(certificate_chunk_rows)
        capacity = operator.index(block_capacity)
        actual_count = operator.index(actual_image_count)
    except TypeError as error:
        raise ValueError("certified coarse GEMM hybrid counts must be integers") from error
    if (
        chunk_rows <= 0
        or chunk_rows % SOURCE_ROTATION_BLOCK_SIZE
        or capacity <= 0
        or actual_count <= 0
        or actual_count > batch_size
    ):
        raise ValueError(
            "certified coarse GEMM hybrid requires positive aligned chunk, "
            "capacity, and image counts",
        )
    if not isinstance(force_static_dense_after_overflow, (bool, np.bool_)):
        raise ValueError("force_static_dense_after_overflow must be boolean")
    if full_dense_diff2_fn is not None and not callable(full_dense_diff2_fn):
        raise TypeError("full_dense_diff2_fn must be callable or None")
    if not isinstance(device_transaction, (bool, np.bool_)):
        raise TypeError("device_transaction must be boolean")
    if not isinstance(real_cross, (bool, np.bool_)):
        raise TypeError("real_cross must be boolean")
    if real_cross and device_transaction:
        raise ValueError("real_cross with device_transaction is not yet qualified")
    if device_transaction and not compact_posterior:
        raise ValueError("device_transaction requires compact_posterior")

    rotation_prior = None
    if rotation_log_prior is not None:
        rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
        if tuple(rotation_prior.shape) != (n_rotations,):
            raise ValueError(
                "hybrid rotation_log_prior must have one value per source rotation",
            )
    score_representation = (
        "dense_full_direct_static_capacity"
        if force_static_dense_after_overflow
        else _select_coarse_gaussian_gemm_score_representation(
            n_rotations=n_rotations,
            block_capacity=capacity,
            compact_posterior=compact_posterior,
        )
    )
    device_result = None
    device_fallback_reason = None
    if score_representation == "dense_full_direct_static_capacity":
        empty_counts = np.zeros(batch_size, dtype=np.int32)
        selection = CoarseGemmHybridBlockSelection(
            eligible=False,
            fallback_reason=(
                "prior_batch_block_capacity_overflow"
                if force_static_dense_after_overflow
                else "compact_physical_capacity_not_smaller_than_dense"
            ),
            block_ids=np.full((batch_size, capacity), -1, dtype=np.int32),
            block_count=empty_counts,
            posterior_block_count=empty_counts.copy(),
            raw_max_block_count=empty_counts.copy(),
        )
    elif device_transaction:
        from recovar.em.dense_single_volume.helpers.coarse_device_rescore import (
            RESCORE_REASONS,
            rescore_coarse_rotation_blocks,
        )

        device_result = rescore_coarse_rotation_blocks(
            cache[0], shifted, weight, initial, jnp.int32(actual_count),
            topology=topology, class_log_prior=class_log_prior,
            rotation_log_prior=rotation_prior,
            translation_log_prior=translation_log_prior,
            chunk_rows=chunk_rows, block_capacity=capacity,
            logical_full_pixel_count=logical_full_pixel_count,
            capture_selected_diff2=capture_selected_diff2,
        )
        # One compact transfer governs publication and supplies telemetry.
        # Interval tables, ordered block IDs and per-image counts stay on device.
        status = np.asarray(device_result.status)
        if status.dtype != np.dtype(np.int64) or status.shape != (5,):
            raise RuntimeError("Invalid device coarse transaction status")
        valid, reason, used_selected, selected_count, max_selected = map(int, status)
        if (
            valid not in (0, 1) or used_selected not in (0, 1)
            or not 0 <= reason < len(RESCORE_REASONS)
        ):
            raise RuntimeError("Invalid device coarse transaction status values")
        if bool(valid) != (reason == 0):
            raise RuntimeError("Device coarse validity and reason disagree")
        if valid and (
            not used_selected or not 0 < max_selected <= capacity
            or not max_selected <= selected_count <= actual_count * capacity
        ):
            raise RuntimeError("Device coarse selected counts are inconsistent")
        selection = device_result.selection
        if valid:
            assembled = device_result.compact_scores
            return CoarseGaussianGemmHybridBatchResult(
                scores=None,
                raw_score_max=jnp.where(
                    jnp.arange(batch_size, dtype=jnp.int32) < actual_count,
                    assembled.raw_score_max, jnp.float32(0.0),
                ),
                scores_include_priors=True, used_selected_rescore=True,
                fallback_reason=None, selection=selection,
                compact_scores=assembled, score_representation=score_representation,
                diagnostic_selected_diff2=device_result.selected_diff2,
                selected_block_count=selected_count, max_selected_blocks=max_selected,
            )
        device_fallback_reason = RESCORE_REASONS[reason]
    else:
        image_batch = _prepare_relion_coarse_gaussian_gemm_f64_image_batch(
            shifted,
            weight,
            initial,
            actual_count,
        )
        state = initialize_coarse_gemm_hybrid_interval_state(
            batch_size,
            n_rotations,
        )
        for rotation_start in range(0, n_rotations, chunk_rows):
            rotation_stop = min(rotation_start + chunk_rows, n_rotations)
            state = _relion_coarse_gaussian_gemm_update_certificate_state(
                state,
                cache[0, rotation_start:rotation_stop],
                image_batch,
                topology=topology,
                rotation_offset=rotation_start,
                class_log_prior=class_log_prior,
                rotation_log_prior=(
                    None
                    if rotation_prior is None
                    else rotation_prior[rotation_start:rotation_stop]
                ),
                translation_log_prior=translation_log_prior,
                real_cross=real_cross,
            )
        selection = select_coarse_gemm_hybrid_rotation_blocks(
            state,
            actual_image_count=actual_count,
            n_rotations=n_rotations,
            n_translations=n_translations,
            certificate_valid=True,
            block_capacity=capacity,
        )
    fallback_reason = (
        device_fallback_reason if device_result is not None else selection.fallback_reason
    )
    if device_result is None and selection.eligible:
        selected_rescore_kwargs = {"topology": topology}
        if logical_full_pixel_count is not None:
            selected_rescore_kwargs["logical_full_pixel_count"] = (
                logical_full_pixel_count
            )
        selected_diff2 = _relion_coarse_diff2_rotation_blocks_from_topology_f32(
            cache[0],
            shifted,
            weight,
            initial,
            jnp.asarray(selection.block_ids, dtype=jnp.int32),
            **selected_rescore_kwargs,
        )
        assemble = (
            assemble_coarse_gemm_hybrid_compact_scores_f32
            if compact_posterior
            else assemble_coarse_gemm_hybrid_dense_scores_f32
        )
        assembled = assemble(
            selected_diff2,
            selection,
            actual_image_count=actual_count,
            n_rotations=n_rotations,
            class_log_prior=class_log_prior,
            rotation_log_prior=rotation_prior,
            translation_log_prior=translation_log_prior,
        )
        if np.all(np.asarray(assembled.selected_output_valid, dtype=bool)):
            compact_scores = assembled if compact_posterior else None
            return CoarseGaussianGemmHybridBatchResult(
                scores=(
                    None
                    if compact_posterior
                    else assembled.posterior_scores_flat.reshape(
                        batch_size,
                        n_rotations,
                        n_translations,
                    )
                ),
                raw_score_max=jnp.where(
                    jnp.arange(batch_size, dtype=jnp.int32) < actual_count,
                    assembled.raw_score_max,
                    jnp.float32(0.0),
                ),
                scores_include_priors=True,
                used_selected_rescore=True,
                fallback_reason=None,
                selection=selection,
                compact_scores=compact_scores,
                score_representation=score_representation,
                diagnostic_selected_diff2=(
                    selected_diff2 if capture_selected_diff2 else None
                ),
            )
        fallback_reason = "invalid_selected_exact_output"
        score_representation = "dense_full_direct_dynamic_fallback"
    elif score_representation != "dense_full_direct_static_capacity":
        score_representation = "dense_full_direct_dynamic_fallback"

    if full_dense_diff2_fn is not None:
        full_diff2 = jnp.asarray(full_dense_diff2_fn())
        full_dense_backend = "fused_projector"
        full_dense_kernel = "fused_projector"
    else:
        full_operands = (
            cache[0],
            shifted,
            weight,
            initial,
            jnp.asarray(topology.full_to_compact),
        )
        if logical_full_pixel_count is None:
            full_diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
                *full_operands,
            )
            full_dense_kernel = "rectangular"
        else:
            scorer = (
                cuda_backproject.relion_coarse_diff2_shared_pretranslated_runtime_f32
                if shared_pretranslated
                else cuda_backproject.relion_coarse_diff2_rectangular_runtime_f32
            )
            full_diff2 = scorer(
                *full_operands,
                jnp.asarray(logical_full_pixel_count, dtype=jnp.int32),
            )
            full_dense_kernel = (
                "shared_pretranslated" if shared_pretranslated else "rectangular_runtime"
            )
        full_dense_backend = "rectangular"
    expected_full_shape = (batch_size, n_rotations, n_translations)
    if tuple(full_diff2.shape) != expected_full_shape:
        raise ValueError(
            "full dense hybrid fallback returned shape "
            f"{tuple(full_diff2.shape)}, expected {expected_full_shape}",
        )
    if np.dtype(full_diff2.dtype) != np.dtype(np.float32):
        raise TypeError(
            "full dense hybrid fallback must return float32, got "
            f"{full_diff2.dtype}",
        )
    raw_scores = -full_diff2
    return CoarseGaussianGemmHybridBatchResult(
        scores=raw_scores,
        raw_score_max=jnp.max(raw_scores.reshape(batch_size, -1), axis=1),
        scores_include_priors=False,
        used_selected_rescore=False,
        fallback_reason=fallback_reason or "unspecified_fail_closed_fallback",
        selection=selection,
        score_representation=score_representation,
        diagnostic_selected_diff2=None,
        full_dense_backend=full_dense_backend,
        full_dense_kernel=full_dense_kernel,
    )


def _score_relion_coarse_gaussian_gemm_macro(
    project_block_once,
    class_index,
    mean_for_proj,
    rotations_block,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    image_shape,
    volume_shape,
    return_projected: bool = False,
):
    """Project once, then score every physical image lane through shared GEMMs."""

    projected_reference, projected_reference_abs2 = project_block_once(
        class_index,
        mean_for_proj,
        rotations_block,
    )
    scores = _relion_coarse_gaussian_gemm_scores(
        projected_reference,
        projected_reference_abs2,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )
    if return_projected:
        return scores, projected_reference, projected_reference_abs2
    return scores
