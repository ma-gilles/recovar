"""Per-image local hypothesis layout and bucketization helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from recovar import utils
from recovar.em.dense_single_volume.batch_planning import (
    _FixedCapacityLocalCall,
    _FixedCapacityPhysicalOrder,
    _FixedCapacityWholeLocalPlan,
    _pack_fixed_capacity_local_candidate_rows,
    _pack_fixed_capacity_local_images,
    _plan_consecutive_padded_batches,
)
from recovar.em.dense_single_volume.helpers.local_search import _local_search_engine_rotation_block_size
from recovar.em.dense_single_volume.helpers.orientation_priors import make_relion_translation_log_prior
from recovar.em.dense_single_volume.shape_buckets import coarse_bucket, power_bucket
from recovar.em.sampling import (
    _normalized_log_weights,
    _wrapped_abs_diff_deg,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    rotation_grid_n_in_planes,
    rotation_grid_size,
    rotation_indices_to_relion_eulers,
)

EXACT_LOCAL_BUCKET_QUANTUM_ENV = "RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM"
EXACT_LOCAL_BUCKET_RADIX_ENV = "RECOVAR_EXACT_LOCAL_BUCKET_RADIX"
EXACT_LOCAL_BUCKET_MIN_QUANTUM = 256


def _resolve_exact_local_bucket_radix(explicit: int | None = None) -> int:
    """Resolve and validate the exact-local small-bucket radix."""

    source = "exact_local_bucket_radix"
    raw_value = explicit
    if raw_value is None:
        source = EXACT_LOCAL_BUCKET_RADIX_ENV
        raw_value = os.environ.get(EXACT_LOCAL_BUCKET_RADIX_ENV, "2")
    try:
        bucket_radix = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer at least 2") from exc
    if bucket_radix < 2:
        raise ValueError(f"{source} must be at least 2")
    return bucket_radix


def _exact_bucket_rotation_size(
    local_rotation_count: int,
    rotation_block_size: int,
    *,
    large_bucket_quantum: int | None = None,
    exact_local_bucket_radix: int | None = None,
) -> int:
    """Return a compile-friendly padded size for one exact local neighborhood.

    The exact local engine cannot safely cap the bucket size below the true
    per-image neighborhood cardinality. Use power-of-two style padding for
    smaller neighborhoods. For larger exact neighborhoods, round up to a
    coarse fixed quantum so nearby local-support
    sizes reuse the same compiled shapes instead of each exact count generating
    its own XLA program.
    """
    local_rotation_count = int(local_rotation_count)
    if local_rotation_count <= 0:
        return 1
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    if local_rotation_count <= engine_cap:
        bucket_radix = _resolve_exact_local_bucket_radix(exact_local_bucket_radix)
        if bucket_radix != 2:
            return int(
                power_bucket(
                    local_rotation_count,
                    base=bucket_radix,
                    minimum=16,
                    maximum=engine_cap,
                ),
            )
        return int(
            coarse_bucket(
                local_rotation_count,
                small_power2_max=engine_cap,
                large_multiple=engine_cap,
                minimum=16,
            ),
        )
    # ``RECOVAR_LOCAL_BUCKET_QUANTUM`` lets callers override the large-bucket
    # quantization. The default is deliberately coarser than the exact-local
    # engine cap: outlier-heavy/local-search tails otherwise generate hundreds
    # of near-duplicate XLA shapes. Hypothesis/tile caps still chunk each
    # bucket, so this changes padding/shape reuse rather than the candidate set.
    env_quantum = os.environ.get("RECOVAR_LOCAL_BUCKET_QUANTUM", "")
    if env_quantum:
        large_bucket_quantum = max(1, int(env_quantum))
    elif large_bucket_quantum is None:
        large_bucket_quantum = max(4096, engine_cap)
    else:
        large_bucket_quantum = max(1, int(large_bucket_quantum))
    return int(
        coarse_bucket(
            local_rotation_count,
            small_power2_max=engine_cap,
            large_multiple=large_bucket_quantum,
            minimum=16,
        ),
    )


def _exact_local_large_bucket_quantum(rotation_block_size: int, explicit: int | None = None) -> int:
    """Return the large-neighborhood bucket quantum for exact local search."""

    if explicit is not None:
        return max(1, int(explicit))
    env_quantum = os.environ.get(EXACT_LOCAL_BUCKET_QUANTUM_ENV, "")
    if env_quantum:
        return max(1, int(env_quantum))
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    return max(EXACT_LOCAL_BUCKET_MIN_QUANTUM, engine_cap)


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
    n_pixels: int
    n_psi: int
    rotation_offsets: np.ndarray
    rotation_ids_flat: np.ndarray
    rotations_flat: np.ndarray
    rotation_log_priors_flat: np.ndarray
    rotation_counts: np.ndarray
    translation_grid: np.ndarray
    translation_log_priors: np.ndarray
    rotation_posterior_ids_flat: np.ndarray | None = None
    sample_mask_flat: np.ndarray | None = None
    mstep_rotations_flat: np.ndarray | None = None

    @property
    def n_images(self) -> int:
        return int(self.rotation_counts.shape[0])

    @property
    def total_local_rotations(self) -> int:
        return int(self.rotation_ids_flat.shape[0])


@dataclass(frozen=True)
class LocalBucketSpec:
    """Static-shape padded execution batch for the exact local engine."""

    image_indices: np.ndarray
    bucket_image_count: int
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    local_rotation_mask: np.ndarray
    translation_log_prior: np.ndarray
    local_rotation_posterior_ids: np.ndarray | None = None
    local_sample_mask: np.ndarray | None = None
    local_mstep_rotations: np.ndarray | None = None


@dataclass(frozen=True)
class _FixedCapacityLocalHypothesisProgram:
    """Immutable host payloads for one sealed fixed-capacity local program."""

    physical_image_capacity: int
    physical_row_capacity: int
    valid_image_count: int
    valid_row_count: int
    image_indices: np.ndarray
    row_offsets: np.ndarray
    valid_image_mask: np.ndarray
    valid_candidate_row_mask: np.ndarray
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_mstep_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    translation_log_prior: np.ndarray
    local_rotation_posterior_ids: np.ndarray | None
    local_sample_mask: np.ndarray | None
    mstep_rotations_fall_back_to_score: bool


def _fixed_capacity_calls_from_local_buckets(
    bucket_specs: Sequence[LocalBucketSpec],
    *,
    expected_order: _FixedCapacityPhysicalOrder,
) -> tuple[_FixedCapacityLocalCall, ...]:
    """Convert authoritative local buckets without changing their topology."""

    if not isinstance(expected_order, _FixedCapacityPhysicalOrder):
        raise ValueError("fixed-capacity local bucket conversion requires an independently sealed physical order")
    bucket_specs = tuple(bucket_specs)
    if not bucket_specs:
        raise ValueError("fixed-capacity local bucket sequence cannot be empty")

    calls = []
    for bucket_index, bucket in enumerate(bucket_specs):
        raw_image_indices = np.asarray(bucket.image_indices)
        raw_row_counts = np.asarray(bucket.actual_rotation_counts)
        if raw_image_indices.ndim != 1 or not np.issubdtype(raw_image_indices.dtype, np.integer):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} image_indices must be a one-dimensional integer array",
            )
        if raw_row_counts.ndim != 1 or not np.issubdtype(raw_row_counts.dtype, np.integer):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} actual_rotation_counts must be a one-dimensional integer array",
            )

        image_indices = raw_image_indices.astype(np.int64, copy=True)
        row_counts = raw_row_counts.astype(np.int64, copy=True)
        if image_indices.size == 0:
            raise ValueError(f"fixed-capacity local bucket {bucket_index} cannot be empty")
        if row_counts.shape != image_indices.shape:
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} row counts must match its real image axis",
            )

        for field_name, raw_value in (
            ("bucket_image_count", bucket.bucket_image_count),
            ("bucket_rotation_count", bucket.bucket_rotation_count),
        ):
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Integral):
                raise ValueError(
                    f"fixed-capacity local bucket {bucket_index} {field_name} must be an integer",
                )
        image_capacity = int(bucket.bucket_image_count)
        radix_bucket = int(bucket.bucket_rotation_count)
        if image_capacity < image_indices.size:
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} image capacity is smaller than its real image count",
            )
        if radix_bucket <= 0:
            raise ValueError(f"fixed-capacity local bucket {bucket_index} radix must be positive")
        if np.any(row_counts <= 0) or np.any(row_counts > radix_bucket):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} row count is outside its radix",
            )

        rotation_mask = np.asarray(bucket.local_rotation_mask)
        if rotation_mask.dtype != np.bool_ or rotation_mask.shape != (image_indices.size, radix_bucket):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} rotation mask must match its real image/radix axes",
            )
        expected_mask = np.arange(radix_bucket, dtype=np.int64)[None, :] < row_counts[:, None]
        if not np.array_equal(rotation_mask, expected_mask):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} candidate membership must be a dense ordered prefix",
            )

        calls.append(
            _FixedCapacityLocalCall(
                image_indices=image_indices,
                row_counts=row_counts,
                radix_bucket=radix_bucket,
                image_capacity=image_capacity,
            )
        )
    chronological_indices = np.concatenate([call.image_indices for call in calls])
    if not np.array_equal(chronological_indices, expected_order.image_indices):
        raise ValueError("fixed-capacity local bucket chronology does not match the sealed physical order")
    return tuple(calls)


def _validate_fixed_capacity_plan_matches_local_calls(
    plan: _FixedCapacityWholeLocalPlan,
    calls: Sequence[_FixedCapacityLocalCall],
    expected_order: _FixedCapacityPhysicalOrder,
) -> None:
    """Fail closed unless ``plan`` is the exact fixed arena for ``calls``."""

    if not isinstance(plan, _FixedCapacityWholeLocalPlan):
        raise ValueError("fixed-capacity hypothesis packing requires a fixed-capacity local plan")

    for field_name in (
        "physical_image_capacity",
        "physical_row_capacity",
        "physical_call_capacity",
        "valid_image_count",
        "valid_row_count",
        "valid_call_count",
    ):
        raw_value = getattr(plan, field_name)
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Integral):
            raise ValueError(f"fixed-capacity hypothesis plan {field_name} must be an integer")
    physical_image_capacity = int(plan.physical_image_capacity)
    physical_row_capacity = int(plan.physical_row_capacity)
    physical_call_capacity = int(plan.physical_call_capacity)
    if min(physical_image_capacity, physical_row_capacity, physical_call_capacity) <= 0:
        raise ValueError("fixed-capacity hypothesis plan capacities must be positive")

    calls = tuple(calls)
    row_counts = np.concatenate([np.asarray(call.row_counts, dtype=np.int64) for call in calls])
    valid_image_count = int(row_counts.size)
    valid_row_count = int(np.sum(row_counts, dtype=np.int64))
    valid_call_count = len(calls)
    if valid_image_count > physical_image_capacity:
        raise ValueError(
            "fixed-capacity hypothesis image capacity overflow: "
            f"valid={valid_image_count}, capacity={physical_image_capacity}",
        )
    if valid_row_count > physical_row_capacity:
        raise ValueError(
            "fixed-capacity hypothesis candidate-row capacity overflow: "
            f"valid={valid_row_count}, capacity={physical_row_capacity}",
        )
    if valid_call_count > physical_call_capacity:
        raise ValueError(
            "fixed-capacity hypothesis call capacity overflow: "
            f"valid={valid_call_count}, capacity={physical_call_capacity}",
        )
    if (
        int(plan.valid_image_count) != valid_image_count
        or int(plan.valid_row_count) != valid_row_count
        or int(plan.valid_call_count) != valid_call_count
    ):
        raise ValueError("fixed-capacity hypothesis plan/bucket valid counts do not match")

    expected_image_indices = np.full(physical_image_capacity, -1, dtype=np.int64)
    expected_image_indices[:valid_image_count] = expected_order.image_indices
    expected_row_offsets = np.full(
        physical_image_capacity + 1,
        valid_row_count,
        dtype=np.int64,
    )
    expected_row_offsets[0] = 0
    expected_row_offsets[1 : valid_image_count + 1] = np.cumsum(row_counts, dtype=np.int64)

    expected_call_valid_mask = np.zeros(physical_call_capacity, dtype=bool)
    expected_call_valid_mask[:valid_call_count] = True
    expected_call_image_offsets = np.full(
        physical_call_capacity,
        valid_image_count,
        dtype=np.int64,
    )
    expected_call_row_offsets = np.full(
        physical_call_capacity,
        valid_row_count,
        dtype=np.int64,
    )
    expected_call_valid_images = np.zeros(physical_call_capacity, dtype=np.int64)
    expected_call_valid_rows = np.zeros(physical_call_capacity, dtype=np.int64)
    expected_call_image_capacities = np.zeros(physical_call_capacity, dtype=np.int64)
    expected_call_radix_buckets = np.zeros(physical_call_capacity, dtype=np.int64)

    running_images = 0
    running_rows = 0
    for call_index, call in enumerate(calls):
        call_valid_images = int(np.asarray(call.image_indices).size)
        call_valid_rows = int(np.sum(np.asarray(call.row_counts), dtype=np.int64))
        expected_call_image_offsets[call_index] = running_images
        expected_call_row_offsets[call_index] = running_rows
        expected_call_valid_images[call_index] = call_valid_images
        expected_call_valid_rows[call_index] = call_valid_rows
        expected_call_image_capacities[call_index] = int(call.image_capacity)
        expected_call_radix_buckets[call_index] = int(call.radix_bucket)
        running_images += call_valid_images
        running_rows += call_valid_rows

    expected_arrays = {
        "image_indices": expected_image_indices,
        "row_offsets": expected_row_offsets,
        "call_valid_mask": expected_call_valid_mask,
        "call_image_offsets": expected_call_image_offsets,
        "call_row_offsets": expected_call_row_offsets,
        "call_valid_images": expected_call_valid_images,
        "call_valid_rows": expected_call_valid_rows,
        "call_image_capacities": expected_call_image_capacities,
        "call_radix_buckets": expected_call_radix_buckets,
    }
    for field_name, expected in expected_arrays.items():
        actual = np.asarray(getattr(plan, field_name))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"fixed-capacity hypothesis plan/bucket mismatch in {field_name}",
            )


def _fixed_capacity_local_payload_array(
    bucket: LocalBucketSpec,
    bucket_index: int,
    field_name: str,
    expected_shape: tuple[int, ...],
    *,
    dtype_kind: str,
) -> np.ndarray:
    value = np.asarray(getattr(bucket, field_name))
    if value.shape != expected_shape:
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} {field_name} has shape "
            f"{value.shape}; expected {expected_shape}",
        )
    if dtype_kind == "integer" and not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"fixed-capacity local bucket {bucket_index} {field_name} must be integer")
    if dtype_kind == "floating" and not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"fixed-capacity local bucket {bucket_index} {field_name} must be floating point")
    if dtype_kind == "boolean" and value.dtype != np.bool_:
        raise ValueError(f"fixed-capacity local bucket {bucket_index} {field_name} must be boolean")
    return value


def _fixed_capacity_optional_topology(bucket_specs: Sequence[LocalBucketSpec], field_name: str) -> bool:
    present = tuple(getattr(bucket, field_name) is not None for bucket in bucket_specs)
    if any(present) and not all(present):
        raise ValueError(
            f"fixed-capacity hypothesis buckets have mixed optional topology for {field_name}",
        )
    return all(present)


def _validate_fixed_capacity_bucket_poison_tails(
    *,
    bucket_index: int,
    rotation_mask: np.ndarray,
    rotation_ids: np.ndarray,
    rotations: np.ndarray,
    mstep_rotations: np.ndarray,
    rotation_log_prior: np.ndarray,
    posterior_ids: np.ndarray | None,
    sample_mask: np.ndarray | None,
) -> None:
    """Verify that discarded rectangular radix tails carry canonical sentinels."""

    inactive = ~rotation_mask
    if np.any(rotation_ids[inactive] != -1):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_rotation_ids poison tails",
        )
    expected_identity = np.eye(3, dtype=rotations.dtype)
    if not np.array_equal(
        rotations[inactive],
        np.broadcast_to(expected_identity, rotations[inactive].shape),
    ):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_rotations poison tails",
        )
    expected_mstep_identity = np.eye(3, dtype=mstep_rotations.dtype)
    if not np.array_equal(
        mstep_rotations[inactive],
        np.broadcast_to(expected_mstep_identity, mstep_rotations[inactive].shape),
    ):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_mstep_rotations poison tails",
        )
    log_prior_poison = np.asarray(-1e30, dtype=rotation_log_prior.dtype)
    if np.any(rotation_log_prior[inactive] != log_prior_poison):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_rotation_log_prior poison tails",
        )
    if posterior_ids is not None and np.any(posterior_ids[inactive] != -1):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_rotation_posterior_ids poison tails",
        )
    if sample_mask is not None and np.any(sample_mask[inactive]):
        raise ValueError(
            f"fixed-capacity local bucket {bucket_index} has malformed local_sample_mask poison tails",
        )


def _pack_fixed_capacity_local_hypothesis_program(
    bucket_specs: Sequence[LocalBucketSpec],
    plan: _FixedCapacityWholeLocalPlan,
    expected_order: _FixedCapacityPhysicalOrder,
    *,
    enabled: bool = False,
) -> _FixedCapacityLocalHypothesisProgram | None:
    """Pack one authoritative bucket program without carrying radix padding.

    This host-only seam is deliberately inert by default.  When enabled, it
    requires the independently sealed physical order and the exact plan made
    from the same buckets.  All payload arrays are snapshots; their active
    rows preserve source bits while inactive fixed-capacity tails use field-
    specific sentinels guarded by explicit valid masks.
    """

    if not enabled:
        return None
    if not isinstance(expected_order, _FixedCapacityPhysicalOrder):
        raise ValueError("fixed-capacity hypothesis packing requires an independently sealed physical order")

    bucket_specs = tuple(bucket_specs)
    calls = _fixed_capacity_calls_from_local_buckets(
        bucket_specs,
        expected_order=expected_order,
    )
    _validate_fixed_capacity_plan_matches_local_calls(plan, calls, expected_order)

    has_mstep_rotations = _fixed_capacity_optional_topology(bucket_specs, "local_mstep_rotations")
    has_posterior_ids = _fixed_capacity_optional_topology(bucket_specs, "local_rotation_posterior_ids")
    has_sample_mask = _fixed_capacity_optional_topology(bucket_specs, "local_sample_mask")

    rotation_ids_by_call = []
    rotations_by_call = []
    mstep_rotations_by_call = []
    rotation_log_prior_by_call = []
    translation_log_prior_by_call = []
    posterior_ids_by_call = []
    sample_mask_by_call = []

    for bucket_index, (bucket, call) in enumerate(zip(bucket_specs, calls, strict=True)):
        valid_images = int(np.asarray(call.image_indices).size)
        radix = int(call.radix_bucket)
        candidate_shape = (valid_images, radix)
        rotation_mask = np.asarray(bucket.local_rotation_mask)
        rotation_ids = _fixed_capacity_local_payload_array(
            bucket,
            bucket_index,
            "local_rotation_ids",
            candidate_shape,
            dtype_kind="integer",
        )
        rotations = _fixed_capacity_local_payload_array(
            bucket,
            bucket_index,
            "local_rotations",
            candidate_shape + (3, 3),
            dtype_kind="floating",
        )
        if has_mstep_rotations:
            mstep_rotations = _fixed_capacity_local_payload_array(
                bucket,
                bucket_index,
                "local_mstep_rotations",
                candidate_shape + (3, 3),
                dtype_kind="floating",
            )
        else:
            mstep_rotations = rotations
        rotation_log_prior = _fixed_capacity_local_payload_array(
            bucket,
            bucket_index,
            "local_rotation_log_prior",
            candidate_shape,
            dtype_kind="floating",
        )

        translation_log_prior = np.asarray(bucket.translation_log_prior)
        if (
            translation_log_prior.ndim != 2
            or translation_log_prior.shape[0] != valid_images
            or translation_log_prior.shape[1] <= 0
            or not np.issubdtype(translation_log_prior.dtype, np.floating)
        ):
            raise ValueError(
                f"fixed-capacity local bucket {bucket_index} translation_log_prior must be a "
                "nonempty floating-point matrix with one row per real image",
            )

        posterior_ids = None
        if has_posterior_ids:
            posterior_ids = _fixed_capacity_local_payload_array(
                bucket,
                bucket_index,
                "local_rotation_posterior_ids",
                candidate_shape,
                dtype_kind="integer",
            )
        sample_mask = None
        if has_sample_mask:
            sample_mask = _fixed_capacity_local_payload_array(
                bucket,
                bucket_index,
                "local_sample_mask",
                candidate_shape + (translation_log_prior.shape[1],),
                dtype_kind="boolean",
            )

        if np.any(rotation_ids[rotation_mask] < 0):
            raise ValueError(f"fixed-capacity local bucket {bucket_index} has negative active rotation IDs")
        if posterior_ids is not None and np.any(posterior_ids[rotation_mask] < 0):
            raise ValueError(f"fixed-capacity local bucket {bucket_index} has negative active posterior IDs")
        _validate_fixed_capacity_bucket_poison_tails(
            bucket_index=bucket_index,
            rotation_mask=rotation_mask,
            rotation_ids=rotation_ids,
            rotations=rotations,
            mstep_rotations=mstep_rotations,
            rotation_log_prior=rotation_log_prior,
            posterior_ids=posterior_ids,
            sample_mask=sample_mask,
        )

        rotation_ids_by_call.append(rotation_ids)
        rotations_by_call.append(rotations)
        mstep_rotations_by_call.append(mstep_rotations)
        rotation_log_prior_by_call.append(rotation_log_prior)
        translation_log_prior_by_call.append(translation_log_prior)
        if posterior_ids is not None:
            posterior_ids_by_call.append(posterior_ids)
        if sample_mask is not None:
            sample_mask_by_call.append(sample_mask)

    local_rotation_ids = _pack_fixed_capacity_local_candidate_rows(
        plan,
        rotation_ids_by_call,
        fill_value=-1,
    )
    local_rotations = _pack_fixed_capacity_local_candidate_rows(
        plan,
        rotations_by_call,
        fill_value=np.nan,
    )
    local_mstep_rotations = _pack_fixed_capacity_local_candidate_rows(
        plan,
        mstep_rotations_by_call,
        fill_value=np.nan,
    )
    local_rotation_log_prior = _pack_fixed_capacity_local_candidate_rows(
        plan,
        rotation_log_prior_by_call,
        fill_value=-np.inf,
    )
    translation_log_prior = _pack_fixed_capacity_local_images(
        plan,
        translation_log_prior_by_call,
        fill_value=-np.inf,
    )
    local_rotation_posterior_ids = (
        _pack_fixed_capacity_local_candidate_rows(
            plan,
            posterior_ids_by_call,
            fill_value=-1,
        )
        if has_posterior_ids
        else None
    )
    local_sample_mask = (
        _pack_fixed_capacity_local_candidate_rows(
            plan,
            sample_mask_by_call,
            fill_value=False,
        )
        if has_sample_mask
        else None
    )

    image_indices = np.asarray(plan.image_indices).copy()
    row_offsets = np.asarray(plan.row_offsets).copy()
    valid_image_mask = np.arange(plan.physical_image_capacity, dtype=np.int64) < int(plan.valid_image_count)
    valid_candidate_row_mask = np.arange(plan.physical_row_capacity, dtype=np.int64) < int(plan.valid_row_count)
    arrays = [
        image_indices,
        row_offsets,
        valid_image_mask,
        valid_candidate_row_mask,
        local_rotation_ids,
        local_rotations,
        local_mstep_rotations,
        local_rotation_log_prior,
        translation_log_prior,
    ]
    if local_rotation_posterior_ids is not None:
        arrays.append(local_rotation_posterior_ids)
    if local_sample_mask is not None:
        arrays.append(local_sample_mask)
    for value in arrays:
        value.setflags(write=False)

    return _FixedCapacityLocalHypothesisProgram(
        physical_image_capacity=int(plan.physical_image_capacity),
        physical_row_capacity=int(plan.physical_row_capacity),
        valid_image_count=int(plan.valid_image_count),
        valid_row_count=int(plan.valid_row_count),
        image_indices=image_indices,
        row_offsets=row_offsets,
        valid_image_mask=valid_image_mask,
        valid_candidate_row_mask=valid_candidate_row_mask,
        local_rotation_ids=local_rotation_ids,
        local_rotations=local_rotations,
        local_mstep_rotations=local_mstep_rotations,
        local_rotation_log_prior=local_rotation_log_prior,
        translation_log_prior=translation_log_prior,
        local_rotation_posterior_ids=local_rotation_posterior_ids,
        local_sample_mask=local_sample_mask,
        mstep_rotations_fall_back_to_score=not has_mstep_rotations,
    )


def _resolve_prior_rotations(prior_rotations: np.ndarray, healpix_order: int, grid_metadata):
    """Return RELION eulers and rotation matrices for local-support construction."""

    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 0:
        prior_rotations = prior_rotations.reshape(1)

    if prior_rotations.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float32)[prior_rotations.astype(np.int64)]
        else:
            prior_eulers = rotation_indices_to_relion_eulers(prior_rotations.astype(np.int64), healpix_order)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return np.asarray(prior_eulers, dtype=np.float32), np.asarray(prior_rotation_mats, dtype=np.float64)
    if prior_rotations.ndim == 2 and prior_rotations.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotations, dtype=np.float32).reshape(-1, 3)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return prior_eulers, np.asarray(prior_rotation_mats, dtype=np.float64)
    prior_rotation_mats = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3, 3)
    prior_eulers = utils.R_to_relion(prior_rotation_mats, degrees=True).astype(np.float32)
    return prior_eulers, prior_rotation_mats


def _local_selector_chunk_size(n_images: int, n_pixels: int, n_psi: int, use_direction: bool, use_psi: bool) -> int:
    explicit = os.environ.get("RECOVAR_LOCAL_SELECTOR_CHUNK_SIZE", "")
    if explicit:
        return max(1, min(int(n_images), int(explicit)))

    max_elements = int(os.environ.get("RECOVAR_LOCAL_SELECTOR_MAX_ELEMENTS", "16000000"))
    per_image_elements = 0
    if use_direction:
        per_image_elements += int(n_pixels)
    if use_psi:
        per_image_elements += int(n_psi)
    if per_image_elements <= 0:
        return max(1, int(n_images))
    return max(1, min(int(n_images), max_elements // per_image_elements))


def _build_factorized_local_entries(
    prior_rotations: np.ndarray,
    healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    grid_metadata,
):
    """Build exact per-image local supports for factorized HEALPix x psi grids."""

    prior_eulers, prior_rotation_mats = _resolve_prior_rotations(prior_rotations, healpix_order, grid_metadata)
    dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
    psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)
    n_pixels = int(grid_metadata["n_pixels"])

    prior_dir_vecs = np.asarray(prior_rotation_mats[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm
    prior_psi_deg = np.mod(np.asarray(prior_eulers[:, 2], dtype=np.float64), 360.0)

    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    # RELION widens the direction cone with max(sigma_rot, sigma_tilt).
    # In this SPA path sigma_tilt == sigma_rot; sigma_psi only controls
    # in-plane support and must not widen directions.
    biggest_sigma_deg = sigma_rot_deg
    cutoff_dir_deg = 3.0 * biggest_sigma_deg
    cutoff_psi_deg = 3.0 * sigma_psi_deg

    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    n_images = int(prior_eulers.shape[0])
    counts = np.zeros(n_images, dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    chunk_size = _local_selector_chunk_size(
        n_images,
        n_pixels,
        int(grid_metadata["n_psi"]),
        sigma_rot_deg > 0.0,
        sigma_psi_deg > 0.0,
    )
    running_offset = 0

    for chunk_start in range(0, n_images, chunk_size):
        chunk_stop = min(n_images, chunk_start + chunk_size)
        if sigma_rot_deg > 0.0:
            dots = np.clip(prior_dir_vecs[chunk_start:chunk_stop] @ dir_vecs.T, -1.0, 1.0)
            diffang_chunk = np.rad2deg(np.arccos(dots))
        else:
            diffang_chunk = None

        if sigma_psi_deg > 0.0:
            diffpsi_chunk = _wrapped_abs_diff_deg(
                psi_deg_grid[None, :],
                prior_psi_deg[chunk_start:chunk_stop, None],
            )
        else:
            diffpsi_chunk = None

        for local_idx, image_idx in enumerate(range(chunk_start, chunk_stop)):
            if sigma_rot_deg > 0.0:
                diffang_i = diffang_chunk[local_idx]
                dir_mask = diffang_i < cutoff_dir_deg
                dir_indices = np.flatnonzero(dir_mask).astype(np.int64)
                if dir_indices.size == 0:
                    dir_indices = np.array([int(np.argmin(diffang_i))], dtype=np.int64)
                    dir_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    dir_log_prior = _normalized_log_weights(diffang_i[dir_indices], biggest_sigma_deg)
            else:
                dir_indices = np.arange(n_pixels, dtype=np.int64)
                dir_log_prior = np.full(n_pixels, -np.log(max(n_pixels, 1)), dtype=np.float32)

            if sigma_psi_deg > 0.0:
                diffpsi_i = diffpsi_chunk[local_idx]
                psi_mask = diffpsi_i < cutoff_psi_deg
                psi_indices = np.flatnonzero(psi_mask).astype(np.int64)
                if psi_indices.size == 0:
                    psi_indices = np.array([int(np.argmin(diffpsi_i))], dtype=np.int64)
                    psi_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    psi_log_prior = _normalized_log_weights(diffpsi_i[psi_indices], sigma_psi_deg)
            else:
                psi_indices = np.arange(int(grid_metadata["n_psi"]), dtype=np.int64)
                psi_log_prior = np.full(
                    psi_indices.shape[0],
                    -np.log(max(psi_indices.shape[0], 1)),
                    dtype=np.float32,
                )

            local_ids = (psi_indices[:, None] * n_pixels + dir_indices[None, :]).reshape(-1).astype(np.int32)
            local_log_prior = (psi_log_prior[:, None] + dir_log_prior[None, :]).reshape(-1).astype(np.float32)
            counts[image_idx] = int(local_ids.shape[0])
            running_offset += int(local_ids.shape[0])
            offsets[image_idx + 1] = running_offset
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
    )
    rotation_log_priors_flat = (
        np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
    )
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat


def _build_parent_expanded_local_entries(
    prior_rotations: np.ndarray,
    fine_healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    *,
    oversampling_order: int,
    rotation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    generate_relion_mstep_rotations: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Build RELION-style local support by expanding selected coarse parents.

    RELION local search first calls
    ``selectOrientationsWithNonZeroPriorProbability`` on the current coarse
    sampling object, then ``getOrientations`` expands those selected coarse
    direction/psi parents into oversampled children. The child orientations
    inherit the parent prior; the prior is not redistributed over children.
    """

    oversampling_order = int(oversampling_order)
    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for parent-expanded local support")
    fine_healpix_order = int(fine_healpix_order)
    parent_order = fine_healpix_order - oversampling_order
    if parent_order < 0:
        raise ValueError(
            "fine_healpix_order must be >= oversampling_order for parent-expanded local support; "
            f"got fine_healpix_order={fine_healpix_order}, oversampling_order={oversampling_order}"
        )

    parent_metadata = build_local_search_grid_metadata(parent_order)
    rotation_log_prior_np = None
    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=np.float32)
        expected_parent_size = rotation_grid_size(parent_order)
        if rotation_log_prior_np.shape[0] != expected_parent_size:
            raise ValueError(
                "rotation_log_prior must have one value per parent-grid rotation "
                f"({expected_parent_size}) for parent-expanded local search; got {rotation_log_prior_np.shape}"
            )
    parent_offsets, parent_counts, parent_ids_flat, parent_log_priors_flat = _build_factorized_local_entries(
        prior_rotations,
        parent_order,
        sigma_rot,
        sigma_psi,
        parent_metadata,
    )

    n_images = int(parent_counts.shape[0])
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    mstep_rotations_parts: list[np.ndarray] = []
    running_offset = 0

    for image_idx in range(n_images):
        start = int(parent_offsets[image_idx])
        stop = int(parent_offsets[image_idx + 1])
        parent_ids = np.asarray(parent_ids_flat[start:stop], dtype=np.int64)
        parent_log_prior = np.asarray(parent_log_priors_flat[start:stop], dtype=np.float32)
        if rotation_log_prior_np is not None:
            parent_log_prior = parent_log_prior + rotation_log_prior_np[parent_ids]
        oversampled = get_oversampled_rotation_grid_from_samples(
            parent_ids,
            parent_order,
            oversampling_order=oversampling_order,
            random_perturbation=float(random_perturbation),
            return_rotation_indices=True,
            return_mstep_rotations=bool(generate_relion_mstep_rotations),
            rotation_index_order="recovar",
        )
        child_rotations, parent_map, child_ids = oversampled[:3]
        child_mstep_rotations = oversampled[3] if bool(generate_relion_mstep_rotations) else None
        parent_map = np.asarray(parent_map, dtype=np.int64)
        child_ids = np.asarray(child_ids, dtype=np.int32)
        child_log_prior = parent_log_prior[parent_map].astype(np.float32, copy=False)

        counts[image_idx] = int(child_ids.shape[0])
        running_offset += int(child_ids.shape[0])
        offsets[image_idx + 1] = running_offset
        rotation_ids_parts.append(child_ids)
        log_prior_parts.append(child_log_prior)
        rotations_parts.append(np.asarray(child_rotations, dtype=np.float32))
        if child_mstep_rotations is not None:
            mstep_rotations_parts.append(np.asarray(child_mstep_rotations, dtype=np.float32))

    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
    )
    rotation_log_priors_flat = (
        np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
    )
    rotations_flat = (
        np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
    )
    mstep_rotations_flat = (
        np.concatenate(mstep_rotations_parts, axis=0)
        if mstep_rotations_parts
        else (np.zeros((0, 3, 3), dtype=np.float32) if generate_relion_mstep_rotations else None)
    )
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat, rotations_flat, mstep_rotations_flat


def _rotation_eulers_from_grid_metadata(
    rotation_ids: np.ndarray,
    grid_metadata,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return canonical RELION Euler angles for selected rotation ids only."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    if "eulers_full" in grid_metadata:
        return np.asarray(grid_metadata["eulers_full"], dtype=dtype)[rotation_ids]
    if str(grid_metadata["mode"]) != "factorized":
        raise ValueError("Selected rotation eulers require factorized metadata or eulers_full")
    n_pixels = int(grid_metadata["n_pixels"])
    pixel_idx = rotation_ids % n_pixels
    psi_idx = rotation_ids // n_pixels
    return np.stack(
        [
            np.asarray(grid_metadata["rot_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["tilt_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["psi_deg"], dtype=dtype)[psi_idx],
        ],
        axis=1,
    ).astype(dtype, copy=False)


def _selected_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
) -> np.ndarray:
    """Build matrices for selected local ids without materializing the full grid."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    if rotation_grid_rotations is not None:
        return np.asarray(rotation_grid_rotations, dtype=np.float32).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = np.unique(rotation_ids, return_inverse=True)
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata)
    if abs(float(random_perturbation)) > 1e-12:
        if angular_sampling_deg is None:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            float(random_perturbation),
            float(angular_sampling_deg),
        )
    else:
        # Preserve RELION's accelerated-path handoff: host RFLOAT inverse
        # matrices are cast to XFLOAT before scoring on the device.
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            0.0,
            0.0,
        )
    return rotations.astype(np.float32, copy=False)[inverse]


def _selected_mstep_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_mstep_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
) -> np.ndarray:
    """Build RELION host-path adjoint matrices for selected local ids."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    if rotation_grid_mstep_rotations is not None:
        return np.asarray(rotation_grid_mstep_rotations, dtype=np.float32).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = np.unique(rotation_ids, return_inverse=True)
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=np.float64)
    if angular_sampling_deg is None:
        if abs(float(random_perturbation)) > 1e-12:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        angular_sampling_deg = 0.0
    _, _, mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
        selected_eulers,
        float(random_perturbation),
        float(angular_sampling_deg),
        return_mstep_rotations=True,
    )
    return np.asarray(mstep_rotations, dtype=np.float32)[inverse]


def build_local_hypothesis_layout(
    prior_rotations: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    translations: np.ndarray,
    prior_translations: np.ndarray,
    sigma_offset_angstrom: float,
    offset_range_pixels: float | None,
    voxel_size: float,
    *,
    grid_metadata,
    translation_prior_reference_translations: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    rotation_grid_random_perturbation: float = 0.0,
    rotation_grid_angular_sampling_deg: float | None = None,
    local_parent_oversampling_order: int = 0,
    rotation_grid_mstep_rotations: np.ndarray | None = None,
    generate_relion_mstep_rotations: bool = False,
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors."""

    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    if rotation_grid_rotations is not None:
        rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
    if rotation_grid_mstep_rotations is not None:
        rotation_grid_mstep_rotations = np.asarray(rotation_grid_mstep_rotations, dtype=np.float32).reshape(-1, 3, 3)
        expected_rotation_count = (
            int(rotation_grid_rotations.shape[0])
            if rotation_grid_rotations is not None
            else int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])
        )
        if int(rotation_grid_mstep_rotations.shape[0]) != expected_rotation_count:
            raise ValueError(
                "rotation_grid_mstep_rotations must match the scoring grid size: "
                f"{rotation_grid_mstep_rotations.shape[0]} vs {expected_rotation_count}",
            )
    generate_relion_mstep_rotations = bool(generate_relion_mstep_rotations or rotation_grid_mstep_rotations is not None)
    translations = np.asarray(translations, dtype=np.float32)
    prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(-1, translations.shape[1])
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=np.float32)

    rotations_flat_override = None
    mstep_rotations_flat_override = None
    if int(local_parent_oversampling_order) > 0:
        (
            offsets,
            counts,
            rotation_ids_flat,
            rotation_log_priors_flat,
            rotations_flat_override,
            mstep_rotations_flat_override,
        ) = _build_parent_expanded_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            oversampling_order=int(local_parent_oversampling_order),
            rotation_log_prior=rotation_log_prior_np,
            random_perturbation=float(rotation_grid_random_perturbation),
            generate_relion_mstep_rotations=generate_relion_mstep_rotations,
        )
    elif str(grid_metadata["mode"]) == "factorized":
        offsets, counts, rotation_ids_flat, rotation_log_priors_flat = _build_factorized_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            grid_metadata,
        )
    else:
        n_images = int(prior_rotations.shape[0])
        offsets = np.zeros(n_images + 1, dtype=np.int64)
        counts = np.zeros(n_images, dtype=np.int32)
        rotation_ids_parts: list[np.ndarray] = []
        log_prior_parts: list[np.ndarray] = []

        for image_idx in range(n_images):
            local_ids, local_log_prior = get_local_rotation_grid_fast(
                prior_rotations[image_idx : image_idx + 1],
                sigma_rot,
                sigma_psi,
                healpix_order,
                sigma_cutoff=3.0,
                per_image=True,
                grid_metadata=grid_metadata,
            )
            local_ids = np.asarray(local_ids, dtype=np.int32).reshape(-1)
            local_log_prior = np.asarray(local_log_prior[0], dtype=np.float32).reshape(-1)
            counts[image_idx] = int(local_ids.shape[0])
            offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

        rotation_ids_flat = (
            np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
        )
        rotation_log_priors_flat = (
            np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
        )
    if rotation_log_prior_np is not None and int(local_parent_oversampling_order) <= 0:
        if rotation_log_prior_np.shape[0] != int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"]):
            raise ValueError(
                "rotation_log_prior must have one value per local-grid rotation "
                f"({int(grid_metadata['n_pixels']) * int(grid_metadata['n_psi'])}); "
                f"got {rotation_log_prior_np.shape}"
            )
        rotation_log_priors_flat = (
            rotation_log_priors_flat + rotation_log_prior_np[np.asarray(rotation_ids_flat, dtype=np.int64)]
        )
    rotations_flat = (
        rotations_flat_override
        if rotations_flat_override is not None
        else _selected_rotation_matrices(
            rotation_ids_flat,
            rotation_grid_rotations,
            grid_metadata,
            random_perturbation=rotation_grid_random_perturbation,
            angular_sampling_deg=rotation_grid_angular_sampling_deg,
        )
    )
    mstep_rotations_flat = None
    if generate_relion_mstep_rotations:
        mstep_rotations_flat = (
            mstep_rotations_flat_override
            if mstep_rotations_flat_override is not None
            else _selected_mstep_rotation_matrices(
                rotation_ids_flat,
                rotation_grid_mstep_rotations,
                grid_metadata,
                random_perturbation=rotation_grid_random_perturbation,
                angular_sampling_deg=rotation_grid_angular_sampling_deg,
            )
        )
    translation_grid = translations
    translation_parent = None
    if int(local_parent_oversampling_order) > 0:
        translation_grid, translation_parent = get_oversampled_translation_grid(
            translations,
            _infer_translation_step(translations),
            oversampling_order=int(local_parent_oversampling_order),
        )
        translation_grid = np.asarray(translation_grid, dtype=np.float32)
        translation_parent = np.asarray(translation_parent, dtype=np.int32)

    reference_translations = (
        np.asarray(translation_prior_reference_translations, dtype=np.float32)
        if translation_prior_reference_translations is not None
        else translations
    )

    coarse_translation_log_priors = make_relion_translation_log_prior(
        reference_translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
    ).astype(np.float32, copy=False)
    if translation_parent is None:
        translation_log_priors = coarse_translation_log_priors
    else:
        translation_log_priors = _fine_translation_log_prior(
            coarse_translation_log_priors,
            translation_parent,
            int(prior_translations.shape[0]),
            int(translation_grid.shape[0]),
        )

    if rotation_grid_rotations is not None:
        n_global_rotations = int(rotation_grid_rotations.shape[0])
    else:
        n_global_rotations = int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])

    return LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=int(grid_metadata["n_pixels"]),
        n_psi=int(grid_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=translation_grid,
        translation_log_priors=np.asarray(translation_log_priors, dtype=np.float32),
        mstep_rotations_flat=mstep_rotations_flat,
    )


def build_local_adaptive_pass2_hypothesis_layout(
    parent_layout: LocalHypothesisLayout,
    significant_sample_indices,
    parent_healpix_order: int,
    *,
    oversampling_order: int,
    random_perturbation: float = 0.0,
    translation_step: float | None = None,
) -> LocalHypothesisLayout:
    """Expand local adaptive parent support while preserving significant pairs.

    RELION's adaptive local pass 2 first expands significant coarse orientation
    parents, then only scores fine translation children for coarse
    ``(orientation, translation)`` pairs that survived pass 1. ``parent_layout``
    carries the image-specific local Gaussian priors from that coarse pass.
    """

    oversampling_order = int(oversampling_order)
    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for adaptive local pass 2")
    parent_healpix_order = int(parent_healpix_order)
    fine_healpix_order = parent_healpix_order + oversampling_order
    n_images = int(parent_layout.n_images)
    if len(significant_sample_indices) != n_images:
        raise ValueError(
            "significant_sample_indices must have one entry per image; "
            f"got {len(significant_sample_indices)} for {n_images} images",
        )

    coarse_translations = np.asarray(parent_layout.translation_grid, dtype=np.float32)
    n_coarse_trans = int(coarse_translations.shape[0])
    if translation_step is None:
        translation_step = _infer_translation_step(coarse_translations)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = int(fine_translations.shape[0])

    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotations_parts: list[np.ndarray] = []
    mstep_rotations_parts: list[np.ndarray] = []
    rotation_ids_parts: list[np.ndarray] = []
    posterior_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    sample_mask_parts: list[np.ndarray | None] = []

    n_parent_global = int(parent_layout.n_global_rotations)
    running_offset = 0
    for image_idx, sig_samples in enumerate(significant_sample_indices):
        parent_start = int(parent_layout.rotation_offsets[image_idx])
        parent_stop = int(parent_layout.rotation_offsets[image_idx + 1])
        local_parent_ids = np.asarray(parent_layout.rotation_ids_flat[parent_start:parent_stop], dtype=np.int32)
        local_parent_log_prior = np.asarray(
            parent_layout.rotation_log_priors_flat[parent_start:parent_stop],
            dtype=np.float32,
        )
        if local_parent_ids.size == 0:
            raise ValueError(f"Image {image_idx} has no local parent rotations for adaptive pass 2")

        if sig_samples is None:
            unique_rot = local_parent_ids
            coarse_rot = local_parent_ids
            coarse_trans = np.tile(np.arange(n_coarse_trans, dtype=np.int32), local_parent_ids.size)
            use_full_candidate_mask = True
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int64).reshape(-1)
            if sig_samples.size == 0:
                unique_rot = local_parent_ids
                coarse_rot = local_parent_ids
                coarse_trans = np.tile(np.arange(n_coarse_trans, dtype=np.int32), local_parent_ids.size)
                use_full_candidate_mask = True
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
                use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= n_parent_global):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the parent grid")
        selected_parent_log_prior, matched_parent_ids = _lookup_values_by_id(
            local_parent_ids,
            local_parent_log_prior,
            unique_rot,
        )
        if not np.all(matched_parent_ids):
            missing = unique_rot[~matched_parent_ids]
            raise ValueError(
                f"Image {image_idx} has significant rotations outside its local parent support: {missing[:8].tolist()}"
            )

        oversampled_rots, parent_map, oversampled_rot_indices, oversampled_mstep_rots = (
            get_oversampled_rotation_grid_from_samples(
                unique_rot,
                parent_healpix_order,
                oversampling_order=oversampling_order,
                random_perturbation=float(random_perturbation),
                return_rotation_indices=True,
                return_mstep_rotations=True,
                rotation_index_order="recovar",
            )
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
        oversampled_mstep_rots = np.asarray(oversampled_mstep_rots, dtype=np.float32)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int32)
        parent_posterior_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if use_full_candidate_mask:
            sample_mask = None
        else:
            local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
            if not np.all(matched_coarse_rot):
                raise ValueError(f"Image {image_idx} has significant samples outside unique parent rotations")
            significance_mask_coarse = np.zeros((unique_rot.shape[0], n_coarse_trans), dtype=bool)
            significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
            sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if sample_mask is not None and not np.any(sample_mask):
            raise ValueError(f"Image {image_idx} has no valid adaptive local pass-2 candidates")

        counts[image_idx] = int(oversampled_rots.shape[0])
        running_offset += int(oversampled_rots.shape[0])
        offsets[image_idx + 1] = running_offset
        rotations_parts.append(oversampled_rots)
        mstep_rotations_parts.append(oversampled_mstep_rots)
        rotation_ids_parts.append(oversampled_rot_indices)
        posterior_ids_parts.append(parent_posterior_ids)
        log_prior_parts.append(selected_parent_log_prior[parent_map].astype(np.float32, copy=False))
        sample_mask_parts.append(sample_mask)

    fine_metadata = build_local_search_grid_metadata(fine_healpix_order)
    rotations_flat = (
        np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
    )
    mstep_rotations_flat = (
        np.concatenate(mstep_rotations_parts, axis=0)
        if mstep_rotations_parts
        else np.zeros((0, 3, 3), dtype=np.float32)
    )
    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
    )
    posterior_ids_flat = (
        np.concatenate(posterior_ids_parts, axis=0) if posterior_ids_parts else np.zeros(0, dtype=np.int32)
    )
    rotation_log_priors_flat = (
        np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
    )
    if not sample_mask_parts:
        sample_mask_flat = np.zeros((0, n_fine_trans), dtype=bool)
    elif all(sample_mask is None for sample_mask in sample_mask_parts):
        # ``None`` is the exact-local engine's compact representation of full
        # per-rotation/per-translation support. Avoid materializing massive
        # all-ones masks for RELION full-parent local pass 2.
        sample_mask_flat = None
    else:
        sample_mask_flat = np.concatenate(
            [
                np.ones((int(count), n_fine_trans), dtype=bool) if sample_mask is None else sample_mask
                for sample_mask, count in zip(sample_mask_parts, counts, strict=True)
            ],
            axis=0,
        )
    return LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_healpix_order),
        n_pixels=int(fine_metadata["n_pixels"]),
        n_psi=int(fine_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=np.asarray(parent_layout.translation_log_priors, dtype=np.float32)[
            :, fine_translation_parent
        ],
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_flat=sample_mask_flat,
        mstep_rotations_flat=mstep_rotations_flat,
    )


def _infer_translation_step(translations: np.ndarray) -> float:
    unique_vals = np.unique(np.asarray(translations, dtype=np.float32))
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 1e-6]
    return float(diffs.min()) if diffs.size else 1.0


def _lookup_values_by_id(ids: np.ndarray, values: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``values`` for integer ids without allocating a global id table."""

    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    values_np = np.asarray(values)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return values_np[:0], np.ones(0, dtype=bool)
    if ids_np.size == 0:
        return values_np[:0], np.zeros(query_np.shape, dtype=bool)

    order = np.argsort(ids_np, kind="stable")
    sorted_ids = ids_np[order]
    # Match the previous dense table behavior for duplicate ids: later writes
    # won, so search to the right and take the last matching entry.
    pos = np.searchsorted(sorted_ids, query_np, side="right") - 1
    valid = pos >= 0
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_ids[pos[valid]] == query_np[valid]
    if not np.all(matched):
        return values_np[:0], matched
    return values_np[order[pos]], matched


def _positions_in_sorted_unique_ids(sorted_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map query ids to row positions in a sorted unique id array."""

    sorted_np = np.asarray(sorted_ids, dtype=np.int64).reshape(-1)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return np.zeros(0, dtype=np.int64), np.ones(0, dtype=bool)
    pos = np.searchsorted(sorted_np, query_np)
    valid = pos < sorted_np.size
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_np[pos[valid]] == query_np[valid]
    return pos.astype(np.int64, copy=False), matched


def _fine_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
) -> np.ndarray:
    if translation_log_prior is None:
        return np.zeros((n_images, n_fine_translations), dtype=np.float32)
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=np.float32)
    if translation_log_prior_np.ndim == 1:
        fine = translation_log_prior_np[fine_translation_parent]
        return np.broadcast_to(fine[None, :], (n_images, n_fine_translations)).astype(np.float32, copy=False)
    if translation_log_prior_np.ndim == 2:
        if translation_log_prior_np.shape[0] != n_images:
            raise ValueError(
                "translation_log_prior must have one row per image when 2D; "
                f"got {translation_log_prior_np.shape[0]} rows for {n_images} images",
            )
        return translation_log_prior_np[:, fine_translation_parent].astype(np.float32, copy=False)
    raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions")


def _pass2_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
) -> np.ndarray:
    if fine_translation_log_prior is None:
        return _fine_translation_log_prior(
            translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
        )
    if translation_log_prior is not None:
        raise ValueError("translation_log_prior and fine_translation_log_prior are mutually exclusive")

    prior_np = np.asarray(fine_translation_log_prior, dtype=np.float32)
    if prior_np.ndim == 1:
        if prior_np.shape[0] != n_fine_translations:
            raise ValueError(
                "fine_translation_log_prior must have one value per fine translation; "
                f"got {prior_np.shape[0]} values for {n_fine_translations} translations",
            )
        return np.broadcast_to(prior_np[None, :], (n_images, n_fine_translations)).astype(np.float32, copy=False)
    if prior_np.ndim == 2:
        if prior_np.shape != (n_images, n_fine_translations):
            raise ValueError(
                f"fine_translation_log_prior must have shape ({n_images}, {n_fine_translations}); got {prior_np.shape}",
            )
        return prior_np.astype(np.float32, copy=False)
    raise ValueError(f"fine_translation_log_prior must be 1D or 2D, got {prior_np.ndim} dimensions")


def build_pass2_hypothesis_layout(
    significant_sample_indices,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    nside_level: int,
    translations: np.ndarray,
    *,
    oversampling_order: int,
    translation_step: float | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    fine_translation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    rotation_index_order: str = "recovar",
    allow_empty: bool = False,
) -> LocalHypothesisLayout:
    """Build exact-local layout for RELION adaptive pass-2 hypotheses.

    Pass 2 is not a Gaussian local search around one previous best pose. RELION
    oversamples the coarse ``(rotation, translation)`` samples that survived
    pass 1. The exact local engine can score the same structure if each image
    carries its own oversampled rotations plus a sparse ``(R, T)`` mask.
    """

    translations_np = np.asarray(translations, dtype=np.float32)
    if translation_step is None:
        translation_step = _infer_translation_step(translations_np)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_translations = int(fine_translations.shape[0])
    n_images = len(significant_sample_indices)
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=np.float32)

    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotations_parts: list[np.ndarray] = []
    rotation_ids_parts: list[np.ndarray] = []
    posterior_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    sample_mask_parts: list[np.ndarray] = []

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rotations, dtype=np.int32)
            coarse_rot = unique_rot
            coarse_trans = None
            use_full_candidate_mask = True
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int64).reshape(-1)
            if sig_samples.size == 0:
                if not allow_empty:
                    raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
                unique_rot = np.zeros(1, dtype=np.int64)
                coarse_rot = unique_rot
                coarse_trans = np.zeros(0, dtype=np.int64)
            else:
                coarse_rot = sig_samples // int(n_coarse_translations)
                coarse_trans = sig_samples % int(n_coarse_translations)
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
            use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= int(n_coarse_rotations)):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the coarse grid")

        oversampled_rots, parent_map, oversampled_rot_indices = get_oversampled_rotation_grid_from_samples(
            unique_rot,
            int(nside_level),
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            rotation_index_order=rotation_index_order,
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int32)
        coarse_parent_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if rotation_log_prior_np is None:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=np.float32)
        else:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map].astype(np.float32, copy=False)

        if use_full_candidate_mask:
            sample_mask = np.ones((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
        else:
            sample_mask = np.zeros((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
            if coarse_trans.size:
                # Vectorized replacement of the inner-image Python loop over
                # unique_rot. Build a (n_unique_rot, n_coarse_trans) mask of
                # significant (rot, trans) pairs, then expand by parent_map
                # and fine_translation_parent. At 50k/256 K=1 this cuts pass2
                # layout-build time from ~10 s/iter to <1 s.
                significance_mask_coarse = np.zeros(
                    (unique_rot.shape[0], int(n_coarse_translations)),
                    dtype=bool,
                )
                local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
                if not np.all(matched_coarse_rot):
                    raise ValueError(f"Image {image_idx} has significant samples outside unique coarse rotations")
                significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
                sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if not np.any(sample_mask) and not allow_empty:
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        counts[image_idx] = int(oversampled_rots.shape[0])
        offsets[image_idx + 1] = offsets[image_idx] + oversampled_rots.shape[0]
        rotations_parts.append(oversampled_rots)
        rotation_ids_parts.append(oversampled_rot_indices)
        posterior_ids_parts.append(coarse_parent_ids)
        log_prior_parts.append(local_rotation_log_prior)
        sample_mask_parts.append(sample_mask)

    rotations_flat = (
        np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
    )
    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0) if rotation_ids_parts else np.zeros(0, dtype=np.int32)
    )
    posterior_ids_flat = (
        np.concatenate(posterior_ids_parts, axis=0) if posterior_ids_parts else np.zeros(0, dtype=np.int32)
    )
    rotation_log_priors_flat = (
        np.concatenate(log_prior_parts, axis=0) if log_prior_parts else np.zeros(0, dtype=np.float32)
    )
    sample_mask_flat = (
        np.concatenate(sample_mask_parts, axis=0)
        if sample_mask_parts
        else np.zeros((0, n_fine_translations), dtype=bool)
    )
    n_pixels = 12 * (2 ** int(nside_level)) ** 2

    return LocalHypothesisLayout(
        n_global_rotations=int(n_coarse_rotations),
        n_pixels=int(n_pixels),
        n_psi=int(rotation_grid_n_in_planes(int(nside_level))),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=_pass2_translation_log_prior(
            translation_log_prior,
            fine_translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
        ),
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_flat=sample_mask_flat,
    )


def bucket_local_hypothesis_layout(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
) -> list[LocalBucketSpec]:
    """Bucket images by exact local-rotation count for static-shape execution."""

    image_batch_size = int(max(1, image_batch_size))
    max_hypotheses_per_microbatch = int(max(1, max_hypotheses_per_microbatch))
    mstep_rotations_flat = (
        np.asarray(layout.rotations_flat, dtype=np.float32)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat, dtype=np.float32)
    )
    if mstep_rotations_flat.shape != np.asarray(layout.rotations_flat).shape:
        raise ValueError(
            "mstep_rotations_flat must match rotations_flat shape: "
            f"{mstep_rotations_flat.shape} vs {np.asarray(layout.rotations_flat).shape}",
        )
    resolved_large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size, large_bucket_quantum)
    bucket_sizes = np.asarray(
        [
            _exact_bucket_rotation_size(
                int(count),
                rotation_block_size,
                large_bucket_quantum=resolved_large_bucket_quantum,
                exact_local_bucket_radix=exact_local_bucket_radix,
            )
            for count in layout.rotation_counts
        ],
        dtype=np.int32,
    )
    # ``RECOVAR_LOCAL_BUCKET_UNIFY=1`` forces all images to share a single
    # rotation-count bucket class so the JIT only compiles one shape per
    # layout. At 50k/256 K=1 this collapses ~13 unique shapes (per-image
    # significant rotation counts vary widely across iters) into 1, removing
    # per-bucket JIT compilation overhead. Memory cost: smaller-significance
    # images carry extra rotation padding.
    if unify_bucket_sizes is None:
        unify_bucket_sizes = os.environ.get("RECOVAR_LOCAL_BUCKET_UNIFY", "").lower() in {"1", "true", "yes", "on"}
    if consecutive_mixed_bucket_size is not None:
        consecutive_mixed_bucket_size = int(consecutive_mixed_bucket_size)
        if consecutive_mixed_bucket_size <= 0:
            raise ValueError("consecutive_mixed_bucket_size must be positive")
        if not preserve_image_order:
            raise ValueError("consecutive mixed buckets require preserved image order")
        if bool(unify_bucket_sizes):
            raise ValueError("consecutive mixed buckets cannot use run-global bucket unification")
    if bucket_sizes.size and bool(unify_bucket_sizes):
        bucket_sizes = np.full_like(bucket_sizes, int(bucket_sizes.max()))
    processing_order = (
        np.arange(layout.n_images, dtype=np.int32)
        if preserve_image_order
        else np.lexsort((layout.rotation_counts, bucket_sizes)).astype(np.int32)
    )
    bucket_specs: list[LocalBucketSpec] = []

    if processing_order.size == 0:
        return bucket_specs

    planned_groups: list[tuple[np.ndarray, int, int]] = []
    if consecutive_mixed_bucket_size is not None:
        for plan in _plan_consecutive_padded_batches(
            bucket_sizes,
            processing_order=processing_order,
            target_items_per_batch=consecutive_mixed_bucket_size,
            max_items_per_batch=image_batch_size,
            max_padded_values_per_batch=max_hypotheses_per_microbatch,
            item_alignment=3,
        ):
            planned_groups.append(
                (
                    np.asarray(plan.item_indices, dtype=np.int32),
                    int(plan.padded_size),
                    int(plan.padded_item_capacity),
                )
            )
    elif preserve_image_order:
        boundaries = np.flatnonzero(
            np.r_[True, bucket_sizes[processing_order][1:] != bucket_sizes[processing_order][:-1], True]
        )
        bucket_groups = [
            processing_order[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    else:
        bucket_groups = [
            processing_order[bucket_sizes[processing_order] == bucket_size]
            for bucket_size in np.unique(bucket_sizes[processing_order])
        ]
    if consecutive_mixed_bucket_size is None:
        for bucket_images in bucket_groups:
            bucket_size = int(bucket_sizes[int(bucket_images[0])])
            max_images = max(1, min(image_batch_size, max_hypotheses_per_microbatch // int(bucket_size)))
            if preserve_image_order and max_images >= 3:
                # RELION's InitialModel default processes pools of three particles.
                # Keep static-shape FFI boundaries on pool boundaries so a new call
                # never changes which physical particles may update BPref together.
                max_images = max(3, (max_images // 3) * 3)
            for start in range(0, bucket_images.shape[0], max_images):
                planned_groups.append(
                    (
                        np.asarray(bucket_images[start : start + max_images], dtype=np.int32),
                        bucket_size,
                        max_images,
                    )
                )

    for image_indices, bucket_size, max_images in planned_groups:
        actual_counts = layout.rotation_counts[image_indices].astype(np.int32, copy=False)
        batch_size = int(image_indices.shape[0])
        padded_rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (batch_size, int(bucket_size), 3, 3),
        ).copy()
        padded_mstep_rotations = padded_rotations.copy()
        padded_rotation_ids = np.full((batch_size, int(bucket_size)), -1, dtype=np.int32)
        padded_log_prior = np.full((batch_size, int(bucket_size)), -1e30, dtype=np.float32)
        padded_mask = np.zeros((batch_size, int(bucket_size)), dtype=bool)
        padded_posterior_ids = (
            None
            if layout.rotation_posterior_ids_flat is None
            else np.full((batch_size, int(bucket_size)), -1, dtype=np.int32)
        )
        padded_sample_mask = (
            None
            if layout.sample_mask_flat is None
            else np.zeros(
                (batch_size, int(bucket_size), int(layout.translation_grid.shape[0])),
                dtype=bool,
            )
        )

        for row, image_idx in enumerate(image_indices.tolist()):
            start_off = int(layout.rotation_offsets[image_idx])
            end_off = int(layout.rotation_offsets[image_idx + 1])
            count = end_off - start_off
            padded_rotations[row, :count] = layout.rotations_flat[start_off:end_off]
            padded_mstep_rotations[row, :count] = mstep_rotations_flat[start_off:end_off]
            padded_rotation_ids[row, :count] = layout.rotation_ids_flat[start_off:end_off]
            padded_log_prior[row, :count] = layout.rotation_log_priors_flat[start_off:end_off]
            padded_mask[row, :count] = True
            if padded_posterior_ids is not None:
                padded_posterior_ids[row, :count] = layout.rotation_posterior_ids_flat[start_off:end_off]
            if padded_sample_mask is not None:
                padded_sample_mask[row, :count, :] = layout.sample_mask_flat[start_off:end_off]

        bucket_specs.append(
            LocalBucketSpec(
                image_indices=image_indices,
                bucket_image_count=int(max_images),
                bucket_rotation_count=int(bucket_size),
                actual_rotation_counts=actual_counts,
                local_rotation_ids=padded_rotation_ids,
                local_rotations=padded_rotations,
                local_rotation_log_prior=padded_log_prior,
                local_rotation_mask=padded_mask,
                translation_log_prior=np.asarray(layout.translation_log_priors[image_indices], dtype=np.float32),
                local_mstep_rotations=padded_mstep_rotations,
                local_rotation_posterior_ids=padded_posterior_ids,
                local_sample_mask=padded_sample_mask,
            )
        )

    return bucket_specs
