"""Host-only binding contracts for the default-off whole-local executor seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from recovar.em.dense_single_volume.batch_planning import (
    _fixed_capacity_plan_descriptor_fingerprint,
    _FixedCapacityLocalGenerationToken,
    _FixedCapacityWholeLocalPlan,
)
from recovar.em.dense_single_volume.local_caches import _FixedCapacityLocalOperands
from recovar.em.dense_single_volume.local_layout import _FixedCapacityLocalHypothesisProgram


@dataclass(frozen=True)
class _FixedCapacityLocalExecutionBundle:
    """One immutable, fingerprint-bound plan/operand/hypothesis triple."""

    plan: _FixedCapacityWholeLocalPlan
    operands: _FixedCapacityLocalOperands
    hypotheses: _FixedCapacityLocalHypothesisProgram
    descriptor_fingerprint: str
    generation_token: _FixedCapacityLocalGenerationToken


@dataclass(frozen=True)
class _FixedCapacityActiveLocalRows:
    """Poison-free active prefixes structurally ready for a future executor."""

    descriptor_fingerprint: str
    generation_token: _FixedCapacityLocalGenerationToken
    excluded_image_tail_count: int
    excluded_candidate_tail_count: int
    image_indices: np.ndarray
    row_offsets: np.ndarray
    raw_images: np.ndarray
    ctf_params: np.ndarray
    metadata_by_name: Mapping[str, np.ndarray]
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_mstep_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    translation_log_prior: np.ndarray
    local_rotation_posterior_ids: np.ndarray | None
    local_sample_mask: np.ndarray | None


_PLAN_ARRAY_FIELDS = (
    "image_indices",
    "row_offsets",
    "call_valid_mask",
    "call_image_offsets",
    "call_row_offsets",
    "call_valid_images",
    "call_valid_rows",
    "call_image_capacities",
    "call_radix_buckets",
    "logical_cutoff",
)

_PROGRAM_ARRAY_FIELDS = (
    "image_indices",
    "row_offsets",
    "valid_image_mask",
    "valid_candidate_row_mask",
    "local_rotation_ids",
    "local_rotations",
    "local_mstep_rotations",
    "local_rotation_log_prior",
    "translation_log_prior",
)


def _fixed_capacity_array_must_be_read_only(owner_name: str, field_name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.flags.writeable:
        raise ValueError(f"fixed-capacity {owner_name} {field_name} must be a read-only snapshot")
    return array


def _validate_fixed_capacity_local_execution_components(
    plan: _FixedCapacityWholeLocalPlan,
    operands: _FixedCapacityLocalOperands,
    hypotheses: _FixedCapacityLocalHypothesisProgram,
) -> str:
    if not isinstance(plan, _FixedCapacityWholeLocalPlan):
        raise ValueError("fixed-capacity execution binding requires a fixed-capacity local plan")
    if not isinstance(operands, _FixedCapacityLocalOperands):
        raise ValueError("fixed-capacity execution binding requires fixed-capacity cached operands")
    if not isinstance(hypotheses, _FixedCapacityLocalHypothesisProgram):
        raise ValueError("fixed-capacity execution binding requires a fixed-capacity hypothesis program")

    fingerprint = _fixed_capacity_plan_descriptor_fingerprint(plan)
    if fingerprint != plan.descriptor_fingerprint:
        raise ValueError("fixed-capacity plan descriptors changed after fingerprint sealing")
    if operands.plan_fingerprint != fingerprint or hypotheses.plan_fingerprint != fingerprint:
        raise ValueError("fixed-capacity execution components are cross-paired across plan fingerprints")
    if (
        operands.plan_generation_token is not plan.generation_token
        or hypotheses.plan_generation_token is not plan.generation_token
    ):
        raise ValueError("fixed-capacity execution components are cross-paired across plan generations")

    for field_name in _PLAN_ARRAY_FIELDS:
        _fixed_capacity_array_must_be_read_only("plan", field_name, getattr(plan, field_name))

    if (
        int(operands.physical_image_capacity) != plan.physical_image_capacity
        or int(operands.valid_image_count) != plan.valid_image_count
        or int(hypotheses.physical_image_capacity) != plan.physical_image_capacity
        or int(hypotheses.physical_row_capacity) != plan.physical_row_capacity
        or int(hypotheses.valid_image_count) != plan.valid_image_count
        or int(hypotheses.valid_row_count) != plan.valid_row_count
    ):
        raise ValueError("fixed-capacity execution component counts do not match the bound plan")

    expected_image_mask = np.arange(plan.physical_image_capacity, dtype=np.int64) < plan.valid_image_count
    expected_row_mask = np.arange(plan.physical_row_capacity, dtype=np.int64) < plan.valid_row_count
    if (
        not np.array_equal(operands.image_indices, plan.image_indices)
        or not np.array_equal(hypotheses.image_indices, plan.image_indices)
        or not np.array_equal(hypotheses.row_offsets, plan.row_offsets)
        or not np.array_equal(operands.valid_image_mask, expected_image_mask)
        or not np.array_equal(hypotheses.valid_image_mask, expected_image_mask)
        or not np.array_equal(hypotheses.valid_candidate_row_mask, expected_row_mask)
    ):
        raise ValueError("fixed-capacity execution component topology does not match the bound plan")

    expected_positions = {
        int(image_id): position
        for position, image_id in enumerate(plan.image_indices[: plan.valid_image_count].tolist())
    }
    if dict(operands.physical_position_by_image_id) != expected_positions:
        raise ValueError("fixed-capacity operand image-position mapping does not match the bound plan")

    for field_name in ("image_indices", "valid_image_mask", "raw_images", "ctf_params"):
        value = _fixed_capacity_array_must_be_read_only("operands", field_name, getattr(operands, field_name))
        if value.shape[0] != plan.physical_image_capacity:
            raise ValueError(f"fixed-capacity operand {field_name} has the wrong physical image axis")
    if operands.image_indices.dtype != np.dtype(np.int64) or operands.image_indices.shape != (
        plan.physical_image_capacity,
    ):
        raise ValueError("fixed-capacity operand image indices have a noncanonical dtype or shape")
    if operands.valid_image_mask.dtype != np.bool_ or operands.valid_image_mask.shape != (
        plan.physical_image_capacity,
    ):
        raise ValueError("fixed-capacity operand image mask has a noncanonical dtype or shape")
    if not isinstance(operands.metadata_by_name, Mapping):
        raise ValueError("fixed-capacity operand metadata must remain a mapping")
    for name, value in operands.metadata_by_name.items():
        value = _fixed_capacity_array_must_be_read_only("operand metadata", name, value)
        if value.shape[0] != plan.physical_image_capacity:
            raise ValueError(f"fixed-capacity operand metadata {name!r} has the wrong physical image axis")

    for field_name in _PROGRAM_ARRAY_FIELDS:
        _fixed_capacity_array_must_be_read_only("hypotheses", field_name, getattr(hypotheses, field_name))
    if hypotheses.image_indices.dtype != np.dtype(np.int64) or hypotheses.image_indices.shape != (
        plan.physical_image_capacity,
    ):
        raise ValueError("fixed-capacity hypothesis image indices have a noncanonical dtype or shape")
    if hypotheses.row_offsets.dtype != np.dtype(np.int64) or hypotheses.row_offsets.shape != (
        plan.physical_image_capacity + 1,
    ):
        raise ValueError("fixed-capacity hypothesis row offsets have a noncanonical dtype or shape")
    if hypotheses.valid_image_mask.dtype != np.bool_ or hypotheses.valid_image_mask.shape != (
        plan.physical_image_capacity,
    ):
        raise ValueError("fixed-capacity hypothesis image mask has a noncanonical dtype or shape")
    if hypotheses.valid_candidate_row_mask.dtype != np.bool_ or hypotheses.valid_candidate_row_mask.shape != (
        plan.physical_row_capacity,
    ):
        raise ValueError("fixed-capacity hypothesis candidate mask has a noncanonical dtype or shape")
    if hypotheses.local_rotation_ids.shape != (plan.physical_row_capacity,) or not np.issubdtype(
        hypotheses.local_rotation_ids.dtype,
        np.integer,
    ):
        raise ValueError("fixed-capacity hypothesis rotation IDs have an invalid row axis or dtype")
    for field_name in ("local_rotations", "local_mstep_rotations"):
        value = np.asarray(getattr(hypotheses, field_name))
        if value.shape != (plan.physical_row_capacity, 3, 3) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"fixed-capacity hypothesis {field_name} has an invalid row axis or dtype")
    if hypotheses.local_rotation_log_prior.shape != (plan.physical_row_capacity,) or not np.issubdtype(
        hypotheses.local_rotation_log_prior.dtype,
        np.floating,
    ):
        raise ValueError("fixed-capacity hypothesis rotation priors have an invalid row axis or dtype")
    if (
        hypotheses.translation_log_prior.ndim != 2
        or hypotheses.translation_log_prior.shape[0] != plan.physical_image_capacity
        or not np.issubdtype(hypotheses.translation_log_prior.dtype, np.floating)
    ):
        raise ValueError("fixed-capacity hypothesis translation priors have an invalid image axis or dtype")
    n_translations = int(hypotheses.translation_log_prior.shape[1])

    if hypotheses.local_rotation_posterior_ids is not None:
        posterior_ids = _fixed_capacity_array_must_be_read_only(
            "hypotheses",
            "local_rotation_posterior_ids",
            hypotheses.local_rotation_posterior_ids,
        )
        if posterior_ids.shape != (plan.physical_row_capacity,) or not np.issubdtype(
            posterior_ids.dtype,
            np.integer,
        ):
            raise ValueError("fixed-capacity hypothesis posterior IDs have an invalid row axis or dtype")
    if hypotheses.local_sample_mask is not None:
        sample_mask = _fixed_capacity_array_must_be_read_only(
            "hypotheses",
            "local_sample_mask",
            hypotheses.local_sample_mask,
        )
        if sample_mask.shape != (plan.physical_row_capacity, n_translations) or sample_mask.dtype != np.bool_:
            raise ValueError("fixed-capacity hypothesis sample mask has an invalid row axis or dtype")
    return fingerprint


def _bind_fixed_capacity_local_execution(
    plan: _FixedCapacityWholeLocalPlan,
    operands: _FixedCapacityLocalOperands,
    hypotheses: _FixedCapacityLocalHypothesisProgram,
    *,
    enabled: bool = False,
) -> _FixedCapacityLocalExecutionBundle | None:
    """Bind matching immutable components; remain inert unless explicitly enabled."""

    if not enabled:
        return None
    fingerprint = _validate_fixed_capacity_local_execution_components(plan, operands, hypotheses)
    return _FixedCapacityLocalExecutionBundle(
        plan=plan,
        operands=operands,
        hypotheses=hypotheses,
        descriptor_fingerprint=fingerprint,
        generation_token=plan.generation_token,
    )


def _validate_fixed_capacity_hypothesis_poison_exclusion(
    plan: _FixedCapacityWholeLocalPlan,
    hypotheses: _FixedCapacityLocalHypothesisProgram,
) -> None:
    valid_images = plan.valid_image_count
    valid_rows = plan.valid_row_count
    image_tail = slice(valid_images, plan.physical_image_capacity)
    row_tail = slice(valid_rows, plan.physical_row_capacity)

    if np.any(hypotheses.valid_image_mask[image_tail]) or np.any(
        hypotheses.valid_candidate_row_mask[row_tail],
    ):
        raise ValueError("fixed-capacity inactive arena tails must be excluded by false masks")
    if np.any(hypotheses.local_rotation_ids[row_tail] != -1):
        raise ValueError("fixed-capacity inactive rotation-ID tail lost its poison sentinel")
    if not np.all(np.isnan(hypotheses.local_rotations[row_tail])) or not np.all(
        np.isnan(hypotheses.local_mstep_rotations[row_tail]),
    ):
        raise ValueError("fixed-capacity inactive rotation tails lost their NaN poison")
    if not np.all(np.isneginf(hypotheses.local_rotation_log_prior[row_tail])) or not np.all(
        np.isneginf(hypotheses.translation_log_prior[image_tail]),
    ):
        raise ValueError("fixed-capacity inactive prior tails lost their negative-infinity poison")
    if hypotheses.local_rotation_posterior_ids is not None and np.any(
        hypotheses.local_rotation_posterior_ids[row_tail] != -1,
    ):
        raise ValueError("fixed-capacity inactive posterior-ID tail lost its poison sentinel")
    if hypotheses.local_sample_mask is not None and np.any(hypotheses.local_sample_mask[row_tail]):
        raise ValueError("fixed-capacity inactive sample-mask tail must remain false")

    if np.any(hypotheses.local_rotation_ids[:valid_rows] < 0):
        raise ValueError("fixed-capacity active rotation rows contain a poison ID")
    if not np.all(np.isfinite(hypotheses.local_rotations[:valid_rows])) or not np.all(
        np.isfinite(hypotheses.local_mstep_rotations[:valid_rows]),
    ):
        raise ValueError("fixed-capacity active rotation rows contain NaN/infinite poison")
    # Active -inf priors can legitimately encode zero probability.  Tail
    # exclusion is structural, so only NaN is poison within an active prefix.
    if np.any(np.isnan(hypotheses.local_rotation_log_prior[:valid_rows])) or np.any(
        np.isnan(hypotheses.translation_log_prior[:valid_images]),
    ):
        raise ValueError("fixed-capacity active prior rows contain NaN poison")
    if hypotheses.local_rotation_posterior_ids is not None and np.any(
        hypotheses.local_rotation_posterior_ids[:valid_rows] < 0,
    ):
        raise ValueError("fixed-capacity active posterior rows contain a poison ID")


def _materialize_fixed_capacity_active_local_rows(
    bundle: _FixedCapacityLocalExecutionBundle,
    *,
    enabled: bool = False,
) -> _FixedCapacityActiveLocalRows | None:
    """Expose only active prefixes after validating that poison tails stay out."""

    if not enabled:
        return None
    if not isinstance(bundle, _FixedCapacityLocalExecutionBundle):
        raise ValueError("fixed-capacity active-row materialization requires a bound execution bundle")
    fingerprint = _validate_fixed_capacity_local_execution_components(
        bundle.plan,
        bundle.operands,
        bundle.hypotheses,
    )
    if bundle.descriptor_fingerprint != fingerprint:
        raise ValueError("fixed-capacity execution bundle fingerprint does not match its components")
    if bundle.generation_token is not bundle.plan.generation_token:
        raise ValueError("fixed-capacity execution bundle generation does not match its components")
    _validate_fixed_capacity_hypothesis_poison_exclusion(bundle.plan, bundle.hypotheses)

    valid_images = bundle.plan.valid_image_count
    valid_rows = bundle.plan.valid_row_count
    metadata = MappingProxyType(
        {name: np.asarray(value)[:valid_images] for name, value in bundle.operands.metadata_by_name.items()}
    )
    return _FixedCapacityActiveLocalRows(
        descriptor_fingerprint=fingerprint,
        generation_token=bundle.generation_token,
        excluded_image_tail_count=bundle.plan.physical_image_capacity - valid_images,
        excluded_candidate_tail_count=bundle.plan.physical_row_capacity - valid_rows,
        image_indices=bundle.plan.image_indices[:valid_images],
        row_offsets=bundle.plan.row_offsets[: valid_images + 1],
        raw_images=bundle.operands.raw_images[:valid_images],
        ctf_params=bundle.operands.ctf_params[:valid_images],
        metadata_by_name=metadata,
        local_rotation_ids=bundle.hypotheses.local_rotation_ids[:valid_rows],
        local_rotations=bundle.hypotheses.local_rotations[:valid_rows],
        local_mstep_rotations=bundle.hypotheses.local_mstep_rotations[:valid_rows],
        local_rotation_log_prior=bundle.hypotheses.local_rotation_log_prior[:valid_rows],
        translation_log_prior=bundle.hypotheses.translation_log_prior[:valid_images],
        local_rotation_posterior_ids=(
            None
            if bundle.hypotheses.local_rotation_posterior_ids is None
            else bundle.hypotheses.local_rotation_posterior_ids[:valid_rows]
        ),
        local_sample_mask=(
            None if bundle.hypotheses.local_sample_mask is None else bundle.hypotheses.local_sample_mask[:valid_rows]
        ),
    )
