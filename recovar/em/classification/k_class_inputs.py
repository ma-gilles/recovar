"""Validate and select inputs for dense and local K-class EM.

Class scheduling and engine dispatch stay in ``k_class``. These helpers own
class-axis validation, shared/per-class selection and local prior layouts;
they do not import execution engines or change scoring precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.local.local_layout import LocalHypothesisLayout


def _class_log_priors(n_classes: int, class_log_priors) -> np.ndarray:
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    priors = np.asarray(class_log_priors, dtype=np.float64)
    if priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {priors.shape}")
    if not np.all(np.isfinite(priors)):
        raise ValueError("class_log_priors must be finite")
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


def _select_projector_half_for_class(value, class_index: int, n_classes: int):
    if value is None:
        return None
    value_array = jnp.asarray(value)
    if value_array.ndim >= 4 and int(value_array.shape[0]) == n_classes:
        return value_array[class_index]
    return value


def _select_required_class_value(value, class_index: int, n_classes: int, name: str):
    value_array = jnp.asarray(value)
    if value_array.ndim < 2 or int(value_array.shape[0]) != n_classes:
        raise ValueError(f"{name} must have leading class axis of length {n_classes}, got {value_array.shape}")
    return value_array[class_index]


def _local_engine_kwargs_for_class(engine_kwargs: dict, class_index: int, n_classes: int) -> dict:
    """Select class-indexed local-engine kwargs before calling the single-class kernel."""

    kwargs = dict(engine_kwargs)
    # RELION adds the unweighted high-shell power_img term once per particle,
    # outside the class loop. Local K-class runs return class-local noise
    # statistics which are summed downstream, so assign the shared term to one
    # class while leaving the single-class route unchanged.
    kwargs["include_unweighted_norm_high_shell"] = class_index == 0
    projector_half = kwargs.get("relion_projector_half")
    if projector_half is not None:
        projector_half_arr = jnp.asarray(projector_half)
        if projector_half_arr.ndim >= 4 and int(projector_half_arr.shape[0]) == n_classes:
            kwargs["relion_projector_half"] = projector_half_arr[class_index]
    scale_dvp = kwargs.get("scale_correction_data_vs_prior")
    if scale_dvp is not None:
        kwargs["scale_correction_data_vs_prior"] = _select_class_value(
            scale_dvp,
            class_index,
            n_classes,
        )
    return kwargs


def _local_layout_for_class(
    local_layout: LocalHypothesisLayout,
    class_local_rotation_log_prior,
    class_index: int,
    n_classes: int,
) -> LocalHypothesisLayout:
    if class_local_rotation_log_prior is None:
        return local_layout
    # Preserve local_layout.rotation_log_priors_flat's own dtype (float64
    # under double-precision scoring) rather than hardcoding float32 --
    # this field is being replaced in place, so forcing a fixed dtype here
    # would silently downgrade the layout bucket_local_hypothesis_layout
    # was built to preserve.
    class_prior = np.asarray(
        _select_required_class_value(
            class_local_rotation_log_prior,
            class_index,
            n_classes,
            "class_local_rotation_log_prior",
        ),
        dtype=np.asarray(local_layout.rotation_log_priors_flat).dtype,
    ).reshape(-1)
    if class_prior.shape != local_layout.rotation_log_priors_flat.shape:
        raise ValueError(
            "class_local_rotation_log_prior per-class values must have shape "
            f"{local_layout.rotation_log_priors_flat.shape}, got {class_prior.shape}",
        )
    return LocalHypothesisLayout(
        n_global_rotations=local_layout.n_global_rotations,
        n_pixels=local_layout.n_pixels,
        n_psi=local_layout.n_psi,
        rotation_offsets=local_layout.rotation_offsets,
        rotation_ids_flat=local_layout.rotation_ids_flat,
        rotations_flat=local_layout.rotations_flat,
        rotation_log_priors_flat=class_prior,
        rotation_counts=local_layout.rotation_counts,
        translation_grid=local_layout.translation_grid,
        translation_log_priors=local_layout.translation_log_priors,
        rotation_posterior_ids_flat=local_layout.rotation_posterior_ids_flat,
        sample_mask_flat=local_layout.sample_mask_flat,
        source_eulers_flat=local_layout.source_eulers_flat,
    )


def _select_local_layout_for_class(
    local_layout,
    class_local_rotation_log_prior,
    class_index: int,
    n_classes: int,
) -> LocalHypothesisLayout:
    if isinstance(local_layout, (list, tuple)):
        if len(local_layout) != n_classes:
            raise ValueError(f"local_layout must contain {n_classes} per-class layouts, got {len(local_layout)}")
        if class_local_rotation_log_prior is not None:
            raise ValueError("class_local_rotation_log_prior is redundant with per-class local layouts")
        return local_layout[class_index]
    return _local_layout_for_class(
        local_layout,
        class_local_rotation_log_prior,
        class_index,
        n_classes,
    )
