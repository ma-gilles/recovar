"""Validate and select inputs for dense and local K-class EM.

Class scheduling and engine dispatch stay in ``k_class``. These helpers own
class-axis validation, shared/per-class selection and local prior layouts;
they do not import execution engines or change scoring precision.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    """Select one RELION projector before transferring it to the device.

    Production projector slabs are NumPy arrays.  Preserve a host view for
    both singleton reshape and K-class indexing so only the selected 3-D slab
    reaches ``jnp.asarray`` in its consumer.  A traced JAX value may reshape
    inside its enclosing compilation.  Reject eager 4-D JAX arrays because
    even a logically shape-only reshape can allocate a second device buffer.
    """
    if value is None:
        return None
    value_array = value if isinstance(value, (np.ndarray, jax.Array, jax.core.Tracer)) else np.asarray(value)
    if value_array.ndim >= 4 and int(value_array.shape[0]) == int(n_classes):
        if isinstance(value_array, jax.Array) and not isinstance(value_array, jax.core.Tracer):
            raise ValueError(
                "RELION projector class selection must occur before eager "
                "device transfer; pass the NumPy host array or select inside "
                "an enclosing jax.jit trace",
            )
        if int(n_classes) == 1:
            if int(class_index) != 0:
                raise IndexError("a singleton RELION projector only has class index 0")
            if isinstance(value_array, jax.core.Tracer):
                return jnp.reshape(value_array, value_array.shape[1:])
            return np.reshape(value_array, value_array.shape[1:])
        return value_array[int(class_index)]
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
        kwargs["relion_projector_half"] = _select_projector_half_for_class(
            projector_half,
            class_index,
            n_classes,
        )
    scale_dvp = kwargs.get("scale_correction_data_vs_prior")
    if scale_dvp is not None:
        kwargs["scale_correction_data_vs_prior"] = _select_class_value(
            scale_dvp,
            class_index,
            n_classes,
        )
    return kwargs


def _class_local_layouts(
    local_layout,
    n_classes: int,
) -> Sequence[LocalHypothesisLayout]:
    if isinstance(local_layout, (list, tuple)):
        if len(local_layout) != n_classes:
            raise ValueError(f"local_layout must contain {n_classes} per-class layouts, got {len(local_layout)}")
        return local_layout
    return (local_layout,) * n_classes
