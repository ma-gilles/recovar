"""K-class EM orchestration for dense and exact-local single-volume engines."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .em_engine import run_em
from .helpers.types import NoiseStats, RelionStats, make_noise_stats, make_relion_stats
from .local_em_engine import run_local_em_exact
from .local_layout import LocalHypothesisLayout


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


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_value, axis=axis) + np.log(
        np.sum(np.exp(values - max_value), axis=axis),
    )


def _class_log_priors(n_classes: int, class_log_priors) -> np.ndarray:
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    priors = np.asarray(class_log_priors, dtype=np.float64)
    if priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {priors.shape}")
    return priors


def _as_class_means(means) -> jax.Array:
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    if int(means_array.shape[0]) < 1:
        raise ValueError("means must contain at least one class")
    return means_array


def _select_class_value(value, class_index: int, n_classes: int):
    value_array = jnp.asarray(value)
    if value_array.ndim >= 2 and int(value_array.shape[0]) == n_classes:
        return value_array[class_index]
    return value


def _reject_kwargs(kwargs: dict, names: tuple[str, ...], caller: str) -> None:
    present = sorted(name for name in names if name in kwargs)
    if present:
        raise ValueError(f"{caller} controls these arguments directly: {', '.join(present)}")


def _dense_outputs(output, *, accumulate_noise: bool):
    new_mean, hard_assignment, Ft_y, Ft_ctf, stats = output[:5]
    noise_stats = output[5] if accumulate_noise else None
    return new_mean, hard_assignment, Ft_y, Ft_ctf, stats, noise_stats


def _local_outputs(output, *, accumulate_noise: bool, return_best_pose_details: bool):
    Ft_y, Ft_ctf, hard_assignment = output[:3]
    next_index = 3
    best_pose_rotations = None
    best_pose_translations = None
    best_pose_rotation_ids = None
    if return_best_pose_details:
        best_pose_rotations = output[next_index]
        best_pose_translations = output[next_index + 1]
        best_pose_rotation_ids = output[next_index + 2]
        next_index += 3
    stats = output[next_index]
    next_index += 1
    noise_stats = output[next_index] if accumulate_noise else None
    return (
        Ft_y,
        Ft_ctf,
        hard_assignment,
        best_pose_rotations,
        best_pose_translations,
        best_pose_rotation_ids,
        stats,
        noise_stats,
    )


def _stack_or_none(values):
    if not values:
        return None
    return jnp.stack([jnp.asarray(value) for value in values], axis=0)


def _selected_by_class(per_class_values, class_assignments: np.ndarray):
    stacked = _stack_or_none(per_class_values)
    if stacked is None:
        return None
    image_indices = jnp.arange(class_assignments.shape[0])
    return stacked[jnp.asarray(class_assignments, dtype=jnp.int32), image_indices]


def _sum_noise_stats(noise_stats: tuple[NoiseStats, ...] | None) -> NoiseStats | None:
    if not noise_stats:
        return None

    def _sum_field(name: str):
        values = [getattr(stats, name) for stats in noise_stats]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Cannot aggregate mixed missing/present noise field {name}")
        return jnp.sum(jnp.stack([jnp.asarray(value) for value in values], axis=0), axis=0)

    return make_noise_stats(
        wsum_sigma2_noise=_sum_field("wsum_sigma2_noise"),
        wsum_img_power=_sum_field("wsum_img_power"),
        wsum_sigma2_offset=sum(float(stats.wsum_sigma2_offset) for stats in noise_stats),
        sumw=sum(float(stats.sumw) for stats in noise_stats),
        wsum_noise_a2=_sum_field("wsum_noise_a2"),
        wsum_noise_xa=_sum_field("wsum_noise_xa"),
    )


def _assemble_result(
    *,
    class_log_evidence: np.ndarray,
    new_means,
    Ft_y,
    Ft_ctf,
    per_class_hard_assignments,
    per_class_stats: tuple[RelionStats, ...],
    noise_stats: tuple[NoiseStats, ...] | None,
    per_class_best_pose_rotations=None,
    per_class_best_pose_translations=None,
    per_class_best_pose_rotation_ids=None,
) -> KClassEMResult:
    global_log_evidence = _logsumexp_np(class_log_evidence, axis=0).astype(np.float64)
    class_responsibilities = np.exp(class_log_evidence - global_log_evidence[None, :])
    class_posterior_sums = np.sum(class_responsibilities, axis=1)

    best_scores = np.stack(
        [np.asarray(stats.best_log_score_per_image, dtype=np.float64) for stats in per_class_stats],
        axis=0,
    )
    pmax = np.stack(
        [np.asarray(stats.max_posterior_per_image, dtype=np.float64) for stats in per_class_stats],
        axis=0,
    )
    class_assignments = np.argmax(class_responsibilities, axis=0).astype(np.int32)
    image_indices = np.arange(class_assignments.shape[0])
    pose_assignments = np.asarray(per_class_hard_assignments)[class_assignments, image_indices]
    rotation_posterior_sums = jnp.sum(
        jnp.stack([jnp.asarray(stats.rotation_posterior_sums) for stats in per_class_stats], axis=0),
        axis=0,
    )
    stats = make_relion_stats(
        log_evidence_per_image=global_log_evidence,
        best_log_score_per_image=np.max(best_scores, axis=0),
        max_posterior_per_image=np.max(pmax, axis=0),
        rotation_posterior_sums=rotation_posterior_sums,
        image_dtype=jnp.float32,
    )
    best_pose_rotations = _selected_by_class(per_class_best_pose_rotations, class_assignments)
    best_pose_translations = _selected_by_class(per_class_best_pose_translations, class_assignments)
    best_pose_rotation_ids = _selected_by_class(per_class_best_pose_rotation_ids, class_assignments)
    if new_means is None or all(mean is None for mean in new_means):
        stacked_new_means = None
    elif any(mean is None for mean in new_means):
        raise ValueError("Cannot stack mixed missing/present per-class new_means")
    else:
        stacked_new_means = jnp.stack([jnp.asarray(mean) for mean in new_means], axis=0)

    return KClassEMResult(
        new_means=stacked_new_means,
        Ft_y=jnp.stack([jnp.asarray(value) for value in Ft_y], axis=0),
        Ft_ctf=jnp.stack([jnp.asarray(value) for value in Ft_ctf], axis=0),
        per_class_hard_assignments=jnp.asarray(per_class_hard_assignments, dtype=jnp.int32),
        class_assignments=jnp.asarray(class_assignments, dtype=jnp.int32),
        pose_assignments=jnp.asarray(pose_assignments, dtype=jnp.int32),
        class_responsibilities=jnp.asarray(class_responsibilities, dtype=jnp.float32),
        class_posterior_sums=jnp.asarray(class_posterior_sums, dtype=jnp.float32),
        stats=stats,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        aggregate_noise_stats=_sum_noise_stats(noise_stats),
        per_class_best_pose_rotations=(
            None if per_class_best_pose_rotations is None else tuple(per_class_best_pose_rotations)
        ),
        per_class_best_pose_translations=(
            None if per_class_best_pose_translations is None else tuple(per_class_best_pose_translations)
        ),
        per_class_best_pose_rotation_ids=(
            None if per_class_best_pose_rotation_ids is None else tuple(per_class_best_pose_rotation_ids)
        ),
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
    )


def run_dense_k_class_em(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    **engine_kwargs,
) -> KClassEMResult:
    """Run dense K-class EM using ``run_em`` as the only scoring/M-step kernel."""

    _reject_kwargs(
        engine_kwargs,
        (
            "return_stats",
            "accumulate_noise",
            "class_log_prior",
            "normalization_log_evidence",
            "disable_adjoint_y",
            "disable_adjoint_ctf",
            "return_profile",
        ),
        "run_dense_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)

    class_log_evidence = []
    for class_index in range(n_classes):
        probe = run_em(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            rotations,
            translations,
            disc_type,
            return_stats=True,
            accumulate_noise=False,
            class_log_prior=float(log_priors[class_index]),
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            **engine_kwargs,
        )
        class_log_evidence.append(np.asarray(probe[4].log_evidence_per_image, dtype=np.float64))

    class_log_evidence_np = np.stack(class_log_evidence, axis=0)
    global_log_evidence = _logsumexp_np(class_log_evidence_np, axis=0)

    new_means = []
    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    for class_index in range(n_classes):
        output = run_em(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            rotations,
            translations,
            disc_type,
            return_stats=True,
            accumulate_noise=accumulate_noise,
            class_log_prior=float(log_priors[class_index]),
            normalization_log_evidence=global_log_evidence,
            **engine_kwargs,
        )
        new_mean, hard_assignment, class_Ft_y, class_Ft_ctf, stats, noise = _dense_outputs(
            output,
            accumulate_noise=accumulate_noise,
        )
        new_means.append(new_mean)
        Ft_y.append(class_Ft_y)
        Ft_ctf.append(class_Ft_ctf)
        hard_assignments.append(np.asarray(hard_assignment, dtype=np.int32))
        per_class_stats.append(stats)
        if per_class_noise is not None:
            per_class_noise.append(noise)

    return _assemble_result(
        class_log_evidence=class_log_evidence_np,
        new_means=new_means,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
    )


def run_local_k_class_em(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    return_best_pose_details: bool = False,
    **engine_kwargs,
) -> KClassEMResult:
    """Run exact-local K-class EM using ``run_local_em_exact`` for all kernels."""

    _reject_kwargs(
        engine_kwargs,
        (
            "accumulate_noise",
            "class_log_prior",
            "normalization_log_evidence",
            "normalization_log_z",
            "disable_adjoint_y",
            "disable_adjoint_ctf",
            "return_profile",
            "return_best_pose_details",
        ),
        "run_local_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)

    class_log_evidence = []
    for class_index in range(n_classes):
        probe = run_local_em_exact(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            local_layout,
            disc_type,
            accumulate_noise=False,
            return_best_pose_details=False,
            class_log_prior=float(log_priors[class_index]),
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            **engine_kwargs,
        )
        class_log_evidence.append(np.asarray(probe[3].log_evidence_per_image, dtype=np.float64))

    class_log_evidence_np = np.stack(class_log_evidence, axis=0)
    global_log_evidence = _logsumexp_np(class_log_evidence_np, axis=0)

    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    per_class_best_pose_rotations = [] if return_best_pose_details else None
    per_class_best_pose_translations = [] if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [] if return_best_pose_details else None
    for class_index in range(n_classes):
        output = run_local_em_exact(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            local_layout,
            disc_type,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            class_log_prior=float(log_priors[class_index]),
            normalization_log_evidence=global_log_evidence,
            **engine_kwargs,
        )
        (
            class_Ft_y,
            class_Ft_ctf,
            hard_assignment,
            best_pose_rotations,
            best_pose_translations,
            best_pose_rotation_ids,
            stats,
            noise,
        ) = _local_outputs(
            output,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
        )
        Ft_y.append(class_Ft_y)
        Ft_ctf.append(class_Ft_ctf)
        hard_assignments.append(np.asarray(hard_assignment, dtype=np.int32))
        per_class_stats.append(stats)
        if per_class_noise is not None:
            per_class_noise.append(noise)
        if return_best_pose_details:
            per_class_best_pose_rotations.append(best_pose_rotations)
            per_class_best_pose_translations.append(best_pose_translations)
            per_class_best_pose_rotation_ids.append(best_pose_rotation_ids)

    return _assemble_result(
        class_log_evidence=class_log_evidence_np,
        new_means=None,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
    )
