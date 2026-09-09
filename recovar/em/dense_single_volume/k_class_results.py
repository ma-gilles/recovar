"""K-class result assembly, joint summaries and host/device publication.

Scheduling and accumulator offloading remain in k_class. This module combines
completed class outputs in their existing order, preserving dtypes, aliases,
canonical Euler metadata and the selected publication policy. It imports no
scoring engine or controller.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .helpers.types import NoiseStats, RelionStats, make_noise_stats, make_relion_stats

_K1_POSE_PUBLISH_DIRECT_ENV = "RECOVAR_K1_POSE_PUBLISH_DIRECT"


class KClassEMResult(NamedTuple):
    """K-class EM sufficient statistics normalized over class and pose axes."""

    new_means: jax.Array | None
    Ft_y: jax.Array
    Ft_ctf: jax.Array
    per_class_hard_assignments: jax.Array
    class_assignments: jax.Array
    pose_assignments: jax.Array
    class_responsibilities: jax.Array
    class_posterior_sums: jax.Array
    stats: RelionStats
    per_class_stats: tuple[RelionStats, ...]
    noise_stats: tuple[NoiseStats, ...] | None
    aggregate_noise_stats: NoiseStats | None
    per_class_best_pose_rotations: tuple[jax.Array, ...] | None = None
    per_class_best_pose_translations: tuple[jax.Array, ...] | None = None
    per_class_best_pose_rotation_ids: tuple[jax.Array, ...] | None = None
    best_pose_rotations: jax.Array | None = None
    best_pose_translations: jax.Array | None = None
    best_pose_rotation_ids: jax.Array | None = None
    profile_summary: dict | None = None
    significant_counts: jax.Array | None = None
    class_mstep_posterior_sums: jax.Array | None = None
    mstep_full_half_axis: int | None = None
    mstep_accumulator_shape: tuple[int, int, int] | None = None

    per_class_best_pose_eulers_deg: tuple[np.ndarray, ...] | None = None
    best_pose_eulers_deg: np.ndarray | None = None


# Retain the historical pickle GLOBAL name. k_class re-exports this same type;
# assigning its module string does not import that execution module here.
KClassEMResult.__module__ = "recovar.em.dense_single_volume.k_class"


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    # Guard against the all-(-inf) case which would otherwise propagate NaN
    # through ``values - max_value`` (since -inf - (-inf) = NaN).
    safe_max = np.where(np.isfinite(max_value), max_value, np.zeros_like(max_value))
    diff = np.where(np.isfinite(values), values - safe_max, -np.inf)
    sum_exp = np.sum(np.exp(diff), axis=axis)
    with np.errstate(divide="ignore"):
        log_sum = np.log(sum_exp)
    return np.squeeze(max_value, axis=axis) + log_sum



def _stack_or_none(values):
    if not values:
        return None
    return jnp.stack([jnp.asarray(value) for value in values], axis=0)



def _k1_pose_publish_direct_requested() -> bool:
    token = os.environ.get(_K1_POSE_PUBLISH_DIRECT_ENV, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"{_K1_POSE_PUBLISH_DIRECT_ENV} must be 0 or 1")
    return token == "1"



def _selected_by_class(per_class_values, class_assignments: np.ndarray, *, direct_single_class: bool = False):
    if (
        direct_single_class
        and per_class_values is not None
        and len(per_class_values) == 1
        and isinstance(class_assignments, np.ndarray)
        and class_assignments.ndim == 1
        and np.all(class_assignments == 0)
    ):
        # The sole class already has the requested image order. Avoid stacking,
        # index normalization and a device gather for each changing subset N.
        value = jnp.asarray(per_class_values[0])
        if value.ndim > 0 and value.shape[0] == class_assignments.shape[0]:
            return value
    stacked = _stack_or_none(per_class_values)
    if stacked is None:
        return None
    image_indices = jnp.arange(class_assignments.shape[0])
    return stacked[jnp.asarray(class_assignments, dtype=jnp.int32), image_indices]



def _sum_noise_stats(noise_stats: tuple[NoiseStats, ...] | None, *, host_arrays=False) -> NoiseStats | None:
    if not noise_stats:
        return None
    if host_arrays:
        if len(noise_stats) != 1:
            raise ValueError("host noise publication requires exactly one class")
        # A one-element class-axis reduction has no additions. Preserve the
        # already published bytes rather than recreating device arrays.
        fields = noise_stats[0]._asdict()
        # Preserve Python sum's initial-zero semantics, including signed zero.
        fields["wsum_sigma2_offset"] = sum(float(stats.wsum_sigma2_offset) for stats in noise_stats)
        fields["sumw"] = sum(float(stats.sumw) for stats in noise_stats)
        return make_noise_stats(**fields, host_arrays=True)

    def _sum_field(name: str):
        values = [getattr(stats, name) for stats in noise_stats]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Cannot aggregate mixed missing/present noise field {name}")
        return jnp.sum(jnp.stack([jnp.asarray(value) for value in values], axis=0), axis=0)

    summed_sigma2_noise = _sum_field("wsum_sigma2_noise")
    return make_noise_stats(
        wsum_sigma2_noise=summed_sigma2_noise,
        wsum_img_power=_sum_field("wsum_img_power"),
        wsum_sigma2_offset=sum(float(stats.wsum_sigma2_offset) for stats in noise_stats),
        sumw=sum(float(stats.sumw) for stats in noise_stats),
        wsum_noise_a2=_sum_field("wsum_noise_a2"),
        wsum_noise_xa=_sum_field("wsum_noise_xa"),
        wsum_norm_correction=_sum_field("wsum_norm_correction"),
        wsum_scale_correction_xa=_sum_field("wsum_scale_correction_xa"),
        wsum_scale_correction_aa=_sum_field("wsum_scale_correction_aa"),
        # make_noise_stats defaults array_dtype=jnp.float32; the per-class
        # noise_stats[i] arrays already carry whatever dtype their own
        # producer correctly chose (float64 under double-precision scoring),
        # and _sum_field's plain jnp.sum preserves it -- derive from that
        # instead of silently narrowing the aggregate back down.
        array_dtype=(np.float32 if summed_sigma2_noise is None else summed_sigma2_noise.dtype),
    )



def _stack_accumulators(values, *, host: bool):
    if host:
        return np.stack([np.asarray(value) for value in values], axis=0)
    return jnp.stack([jnp.asarray(value) for value in values], axis=0)



def _sum_k_class_noise_stats(
    noise_stats: tuple[NoiseStats, ...] | None,
    class_posterior_sums: np.ndarray,
    *,
    host_arrays=False,
) -> NoiseStats | None:
    """Aggregate Class3D noise stats with RELION's single global sum_weight.

    Each per-class ``run_em`` call normalizes posteriors over poses within one
    class and reports ``sumw == n_images``.  RELION normalizes over the joint
    class x pose grid, so ``sum_weight`` is the sum of class responsibilities
    over images, not ``n_classes * n_images``.  Newer fused K-class paths
    already return per-class image-power stats weighted by that same class
    mass; legacy paths still return raw image power and need total-mass
    rescaling.
    """

    aggregate = _sum_noise_stats(noise_stats, host_arrays=host_arrays)
    if aggregate is None:
        return None
    responsibilities = np.asarray(class_posterior_sums, dtype=np.float64).reshape(-1)
    relion_sumw = float(np.sum(responsibilities))
    raw_sumw = np.asarray([float(stats.sumw) for stats in noise_stats], dtype=np.float64)
    if responsibilities.shape != raw_sumw.shape:
        raise ValueError(
            "class_posterior_sums and noise_stats disagree on class count: "
            f"{responsibilities.shape[0]} vs {raw_sumw.shape[0]}",
        )
    image_power = np.asarray(aggregate.wsum_img_power, dtype=np.float64)
    if not np.allclose(raw_sumw, responsibilities, rtol=1e-4, atol=1e-4):
        image_power = np.zeros_like(image_power)
        for stats, responsibility, class_sumw in zip(noise_stats, responsibilities, raw_sumw, strict=True):
            if class_sumw <= 0.0:
                continue
            image_power += np.asarray(stats.wsum_img_power, dtype=np.float64) * (responsibility / class_sumw)
    return aggregate._replace(
        wsum_img_power=(np.asarray if host_arrays else jnp.asarray)(image_power, dtype=aggregate.wsum_img_power.dtype), sumw=relion_sumw
    )



def _resolve_class_mstep_posterior_sums(
    *,
    noise_stats: tuple[NoiseStats, ...] | None,
    class_posterior_sums_full: np.ndarray,
    class_posterior_sums_override,
) -> np.ndarray:
    """Choose RELION's retained class/pose mass for M-step normalization.

    Independent multi-class probes normalize each class over its own pose
    grid, so their raw ``NoiseStats.sumw`` values cannot be combined and the
    full class responsibilities remain the fallback.  K=1 has no class-axis
    renormalization: its single noise statistic already contains RELION's
    significant-support mass and must not be replaced by the image count.
    Fused/local K-class callers can provide the corresponding retained
    per-class masses explicitly.
    """

    if class_posterior_sums_override is not None:
        resolved = np.asarray(class_posterior_sums_override, dtype=np.float64)
    elif noise_stats is not None and len(noise_stats) == 1:
        resolved = np.asarray([float(noise_stats[0].sumw)], dtype=np.float64)
    else:
        resolved = np.asarray(class_posterior_sums_full, dtype=np.float64)
    if resolved.shape != np.asarray(class_posterior_sums_full).shape:
        raise ValueError(
            "class_posterior_sums_override must have shape "
            f"{np.asarray(class_posterior_sums_full).shape}, got {resolved.shape}",
        )
    if not np.all(np.isfinite(resolved)) or np.any(resolved < 0.0):
        raise ValueError("class M-step posterior sums must be finite and non-negative")
    return resolved



def _assemble_result(
    *,
    class_log_evidence: np.ndarray,
    new_means,
    Ft_y,
    Ft_ctf,
    per_class_hard_assignments,
    per_class_stats: tuple[RelionStats, ...],
    noise_stats: tuple[NoiseStats, ...] | None,
    per_class_best_pose_eulers_deg=None,
    per_class_best_pose_rotations=None,
    per_class_best_pose_translations=None,
    per_class_best_pose_rotation_ids=None,
    profile_summary: dict | None = None,
    class_posterior_sums_override=None,
    firstiter_winner_take_all: bool = False,
    host_accumulators: bool = False,
    host_stats_publication: bool = False,
    mstep_full_half_axis: int | None = None,
    mstep_accumulator_shape: tuple[int, int, int] | None = None,
) -> KClassEMResult:
    # Derive the output dtype from the per-class stats' own precision rather
    # than hardcoding float32: threading an explicit dtype/precision_policy
    # parameter through this function's several callers (each several
    # layers removed from iteration_loop.py's precision switches) would be
    # invasive, and per_class_stats[i].best_log_score_per_image already
    # carries whatever dtype its own caller correctly chose (float64 under
    # double-precision scoring) -- forcing float32 here discards that
    # upstream precision at this universal per-image aggregation step.
    output_dtype = (
        np.asarray(per_class_stats[0].best_log_score_per_image).dtype if per_class_stats else np.float32
    )
    if type(host_stats_publication) is not bool:
        raise TypeError("host_stats_publication must be a bool")
    if host_stats_publication and (len(per_class_stats) != 1 or not host_accumulators):
        raise ValueError("host statistics publication requires one class with host accumulators")
    global_log_evidence = _logsumexp_np(class_log_evidence, axis=0).astype(np.float64)
    # Guard against -inf - (-inf) = NaN when an entire (image, class) had all
    # poses masked out (e.g., RELION firstiter_cc_pass2_only_best_coarse where
    # the losing class is fully excluded by the significance mask). Treat
    # those entries as zero responsibility, matching RELION's binarized
    # weight pattern.
    diff = np.where(
        np.isfinite(global_log_evidence)[None, :] & np.isfinite(class_log_evidence),
        class_log_evidence - global_log_evidence[None, :],
        -np.inf,
    )
    class_responsibilities = np.exp(diff)
    class_posterior_sums_full = np.sum(class_responsibilities, axis=1)
    class_posterior_sums = class_posterior_sums_full
    class_mstep_posterior_sums = _resolve_class_mstep_posterior_sums(
        noise_stats=noise_stats,
        class_posterior_sums_full=class_posterior_sums_full,
        class_posterior_sums_override=class_posterior_sums_override,
    )

    best_scores = np.stack(
        [np.asarray(stats.best_log_score_per_image, dtype=np.float64) for stats in per_class_stats],
        axis=0,
    )
    class_assignments = np.argmax(best_scores, axis=0).astype(np.int32)
    image_indices = np.arange(class_assignments.shape[0])
    pose_assignments = np.asarray(per_class_hard_assignments)[class_assignments, image_indices]
    global_best_scores = np.max(best_scores, axis=0)
    joint_pmax = np.zeros_like(global_best_scores, dtype=np.float64)
    if firstiter_winner_take_all:
        # RELION binarizes the firstiter-CC weights after each pass, so every
        # valid image has Pmax=1. The coarse class log-evidence and fine-pass
        # best scores above come from different score surfaces and cannot be
        # subtracted to reconstruct this probability.
        joint_pmax[np.isfinite(global_best_scores)] = 1.0
    elif len(per_class_stats) == 1:
        # The single-class kernel is the authoritative source for Pmax.  In
        # RELION-exact mode it obtains this value from the float32
        # exp/sort/scan/divide path, before converting scores back to absolute
        # log-evidence coordinates.  Recomputing ``best - logZ`` here loses
        # that arithmetic boundary (and at real-data score magnitudes visibly
        # quantizes Pmax to exp(-n / 128)).  K>1 still needs the joint-class
        # normalization below.
        joint_pmax = np.asarray(
            per_class_stats[0].max_posterior_per_image,
            dtype=np.float64,
        ).copy()
    else:
        finite_joint_best = np.isfinite(global_best_scores) & np.isfinite(global_log_evidence)
        joint_log_pmax = global_best_scores[finite_joint_best] - global_log_evidence[finite_joint_best]
        joint_pmax[finite_joint_best] = np.exp(np.minimum(joint_log_pmax, 0.0))
    # Pmax is a probability; clip tiny numerical overshoots or inconsistent
    # synthetic fixtures while preserving all valid joint posterior values.
    joint_pmax = np.clip(joint_pmax, 0.0, 1.0)
    if host_stats_publication:
        rotation_posterior_sums = np.asarray(per_class_stats[0].rotation_posterior_sums)
    else:
        rotation_posterior_sums = jnp.sum(
            jnp.stack([jnp.asarray(stats.rotation_posterior_sums) for stats in per_class_stats], axis=0),
            axis=0,
        )
    stats = make_relion_stats(
        log_evidence_per_image=global_log_evidence,
        best_log_score_per_image=global_best_scores,
        max_posterior_per_image=joint_pmax,
        rotation_posterior_sums=rotation_posterior_sums,
        image_dtype=output_dtype,
        # make_relion_stats defaults rotation_dtype=jnp.float32; override with
        # rotation_posterior_sums' own (already correctly precision-derived)
        # dtype instead of silently narrowing it back down.
        rotation_dtype=rotation_posterior_sums.dtype,
        host_arrays=host_stats_publication,
    )
    direct_single_class = _k1_pose_publish_direct_requested()
    best_pose_rotations = _selected_by_class(
        per_class_best_pose_rotations, class_assignments, direct_single_class=direct_single_class
    )
    best_pose_translations = _selected_by_class(
        per_class_best_pose_translations, class_assignments, direct_single_class=direct_single_class
    )
    best_pose_rotation_ids = _selected_by_class(
        per_class_best_pose_rotation_ids, class_assignments, direct_single_class=direct_single_class
    )
    best_pose_eulers_deg = None
    if per_class_best_pose_eulers_deg is not None:
        selected_classes = np.asarray(class_assignments)
        active_classes = np.unique(selected_classes)
        if all(per_class_best_pose_eulers_deg[int(k)] is not None for k in active_classes):
            best_pose_eulers_deg = np.empty((selected_classes.size, 3), dtype=np.float64)
            for k in active_classes:
                rows = selected_classes == k
                best_pose_eulers_deg[rows] = np.asarray(per_class_best_pose_eulers_deg[int(k)])[rows]
    if new_means is None or all(mean is None for mean in new_means):
        stacked_new_means = None
    elif any(mean is None for mean in new_means):
        raise ValueError("Cannot stack mixed missing/present per-class new_means")
    else:
        stacked_new_means = jnp.stack([jnp.asarray(mean) for mean in new_means], axis=0)

    aggregate_noise_stats = _sum_k_class_noise_stats(
        noise_stats, class_mstep_posterior_sums, host_arrays=host_stats_publication
    )
    profile_summary_out = None
    if profile_summary is not None:
        profile_summary_out = dict(profile_summary)
        profile_summary_out["class_posterior_sums_full"] = class_posterior_sums_full.astype(
            np.float64,
            copy=True,
        )
        profile_summary_out["class_posterior_sums_returned"] = class_posterior_sums.astype(
            np.float64,
            copy=True,
        )
        profile_summary_out["class_mstep_posterior_sums_returned"] = class_mstep_posterior_sums.astype(
            np.float64,
            copy=True,
        )
        profile_summary_out["class_posterior_sums_used_override"] = class_posterior_sums_override is not None

    result_array = np.asarray if host_stats_publication else jnp.asarray
    return KClassEMResult(
        new_means=stacked_new_means,
        Ft_y=_stack_accumulators(Ft_y, host=host_accumulators),
        Ft_ctf=_stack_accumulators(Ft_ctf, host=host_accumulators),
        per_class_hard_assignments=result_array(per_class_hard_assignments, dtype=jnp.int32),
        class_assignments=result_array(class_assignments, dtype=jnp.int32),
        pose_assignments=result_array(pose_assignments, dtype=jnp.int32),
        class_responsibilities=result_array(class_responsibilities, dtype=output_dtype),
        class_posterior_sums=result_array(class_posterior_sums, dtype=output_dtype),
        stats=stats,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        aggregate_noise_stats=aggregate_noise_stats,
        per_class_best_pose_rotations=(
            None if per_class_best_pose_rotations is None else tuple(per_class_best_pose_rotations)
        ),
        per_class_best_pose_translations=(
            None if per_class_best_pose_translations is None else tuple(per_class_best_pose_translations)
        ),
        per_class_best_pose_rotation_ids=(
            None if per_class_best_pose_rotation_ids is None else tuple(per_class_best_pose_rotation_ids)
        ),
        per_class_best_pose_eulers_deg=(
            None if per_class_best_pose_eulers_deg is None else tuple(per_class_best_pose_eulers_deg)
        ),
        best_pose_eulers_deg=best_pose_eulers_deg,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        profile_summary=profile_summary_out,
        class_mstep_posterior_sums=result_array(class_mstep_posterior_sums, dtype=output_dtype),
        mstep_full_half_axis=mstep_full_half_axis,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _expand_subset_noise_stats(
    noise: NoiseStats,
    image_indices: np.ndarray,
    n_images: int,
    *,
    full_group_count: int | None,
) -> NoiseStats:
    """Expand subset-only optional noise fields to the parent image/group axes."""

    image_indices = np.asarray(image_indices, dtype=np.int64).reshape(-1)
    n_images = int(n_images)
    if image_indices.size:
        if int(np.min(image_indices)) < 0 or int(np.max(image_indices)) >= n_images:
            raise ValueError("subset image indices are out of bounds")

    def _image_field(value, name: str):
        if value is None:
            return None
        array = np.asarray(value)
        flat = array.reshape(-1)
        if flat.shape[0] == n_images:
            return jnp.asarray(array)
        if flat.shape[0] != image_indices.size:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {image_indices.size} subset values "
                f"or {n_images} full-image values",
            )
        out = np.zeros(n_images, dtype=flat.dtype)
        out[image_indices] = flat
        return jnp.asarray(out)

    def _group_field(value, name: str):
        if value is None:
            return None
        array = np.asarray(value)
        flat = array.reshape(-1)
        if full_group_count is None or flat.shape[0] == int(full_group_count):
            return jnp.asarray(array)
        if flat.shape[0] > int(full_group_count):
            raise ValueError(
                f"{name} has shape {array.shape}; expected no more than {full_group_count} groups",
            )
        out = np.zeros(int(full_group_count), dtype=flat.dtype)
        out[: flat.shape[0]] = flat
        return jnp.asarray(out)

    return noise._replace(
        wsum_norm_correction=_image_field(noise.wsum_norm_correction, "wsum_norm_correction"),
        wsum_scale_correction_xa=_group_field(noise.wsum_scale_correction_xa, "wsum_scale_correction_xa"),
        wsum_scale_correction_aa=_group_field(noise.wsum_scale_correction_aa, "wsum_scale_correction_aa"),
    )



def _zero_subset_noise_stats(
    noise_variance,
    *,
    n_images: int,
    full_group_count: int | None,
) -> NoiseStats:
    class_noise = np.asarray(noise_variance)
    stats_dtype = class_noise.dtype
    return make_noise_stats(
        wsum_sigma2_noise=np.zeros_like(class_noise),
        wsum_img_power=np.zeros_like(class_noise),
        wsum_sigma2_offset=0.0,
        sumw=0.0,
        wsum_norm_correction=np.zeros(int(n_images), dtype=stats_dtype),
        wsum_scale_correction_xa=(
            None if full_group_count is None else np.zeros(int(full_group_count), dtype=stats_dtype)
        ),
        wsum_scale_correction_aa=(
            None if full_group_count is None else np.zeros(int(full_group_count), dtype=stats_dtype)
        ),
    )



def _full_stats_from_subset(
    subset_stats: RelionStats,
    image_indices: np.ndarray,
    n_images: int,
    *,
    class_log_evidence: np.ndarray,
) -> RelionStats:
    image_indices = np.asarray(image_indices, dtype=np.int64)
    stats_dtype = np.asarray(subset_stats.best_log_score_per_image).dtype
    best = np.full(int(n_images), -np.inf, dtype=stats_dtype)
    pmax = np.zeros(int(n_images), dtype=stats_dtype)
    best[image_indices] = np.asarray(subset_stats.best_log_score_per_image, dtype=stats_dtype)
    pmax[image_indices] = np.asarray(subset_stats.max_posterior_per_image, dtype=stats_dtype)
    return make_relion_stats(
        log_evidence_per_image=np.asarray(class_log_evidence, dtype=stats_dtype),
        best_log_score_per_image=best,
        max_posterior_per_image=pmax,
        rotation_posterior_sums=subset_stats.rotation_posterior_sums,
    )



def _expand_subset_pose_details(best_rots, best_trans, best_rot_ids, image_indices, n_images):
    """Scatter scored poses to parent image rows, leaving unvisited rows zero.

    Preserve the rotation/translation dtypes and the int32 published pose IDs.
    The caller retains the returned arrays in its per-class output slots.
    """
    best_rots_full = np.zeros((n_images, 3, 3), dtype=np.asarray(best_rots).dtype)
    best_trans_full = np.zeros((n_images, 2), dtype=np.asarray(best_trans).dtype)
    best_rot_ids_full = np.zeros(n_images, dtype=np.int32)
    best_rots_full[image_indices] = best_rots
    best_trans_full[image_indices] = best_trans
    best_rot_ids_full[image_indices] = best_rot_ids
    return best_rots_full, best_trans_full, best_rot_ids_full
