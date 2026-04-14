import logging

import jax.numpy as jnp
import numpy as np
import scipy.stats

from recovar import core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.configs import ForwardModelConfig
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)
## TODO: clean this up, document it (should be very non probably since no one has asked for this, but nevertheless an interesting feature)


def _process_images_if_available(experiment_dataset, images):
    process_images = getattr(experiment_dataset, "process_images", None)
    if process_images is None:
        return images
    return process_images(images)


def compute_residual(
    images,
    ctf_params,
    rotation_matrices,
    translations,
    mean_estimate,
    config,
    noise_variance,
):
    if not isinstance(mean_estimate, (core.Volume, core.CubicVolume)):
        raise TypeError("compute_residual requires Volume(...) or CubicVolume(...)")
    images = core.translate_images(images, translations, config.image_shape)

    projected_mean = core_forward.forward_model(
        config,
        mean_estimate,
        ctf_params,
        rotation_matrices,
    )
    difference = images - projected_mean
    difference /= jnp.sqrt(noise_variance)[None]

    return jnp.linalg.norm(difference, axis=-1) ** 2


def _require_batched_projection_volumes(volumes, *, volume_shape, function_name):
    if not isinstance(volumes, (core.Volume, core.CubicVolume)):
        raise TypeError(f"{function_name} requires batched Volume(...) or CubicVolume(...)")
    arrays = jnp.asarray(volumes.array)
    full_shape = tuple(int(s) for s in volume_shape)
    half_shape = tuple(int(s) for s in ftu.volume_shape_to_half_volume_shape(volume_shape))
    if arrays.ndim == 1 or tuple(arrays.shape) in (full_shape, half_shape):
        raise ValueError(f"{function_name} requires a batch of candidate volumes, not a single volume")
    return volumes, arrays


def compute_image_assignment(experiment_dataset, volumes, noise_variance, batch_size):
    projection_volumes, projection_arrays = _require_batched_projection_volumes(
        volumes,
        volume_shape=experiment_dataset.volume_shape,
        function_name="compute_image_assignment",
    )
    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=projection_volumes.disc_type)

    logger.info(
        "Computing image assignment: %d volumes, %d images, batch_size=%d",
        projection_arrays.shape[0],
        experiment_dataset.n_units,
        batch_size,
    )
    projection_arrays = jnp.asarray(projection_arrays, dtype=experiment_dataset.dtype)
    projection_volumes = projection_volumes.replace_array(projection_arrays)
    residuals = np.zeros((projection_arrays.shape[0], experiment_dataset.n_units), dtype=experiment_dataset.dtype_real)
    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(batch_size):
        images = _process_images_if_available(experiment_dataset, images)
        for volume_ind in range(projection_arrays.shape[0]):
            projection_volume = projection_volumes.replace_array(projection_arrays[volume_ind])
            residuals[volume_ind, particle_indices] = compute_residual(
                images,
                ctf_params,
                rotation_matrices,
                translations,
                projection_volume,
                config,
                np.array(noise_variance),
            )
    return residuals


def estimate_false_positive_rate(experiment_dataset, volumes, noise_variance, batch_size):
    projection_volumes, projection_arrays = _require_batched_projection_volumes(
        volumes,
        volume_shape=experiment_dataset.volume_shape,
        function_name="estimate_false_positive_rate",
    )
    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=projection_volumes.disc_type)

    projection_arrays = jnp.asarray(projection_arrays, dtype=experiment_dataset.dtype)
    alphas = np.zeros((experiment_dataset.n_units,), dtype=experiment_dataset.dtype_real)
    if projection_arrays.shape[0] != 2:
        raise ValueError(f"Only two volumes are supported, got {projection_arrays.shape[0]}")
    projection_volumes = projection_volumes.replace_array(projection_arrays)
    difference = projection_volumes.replace_array(projection_arrays[0] - projection_arrays[1])
    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(batch_size):
        images = _process_images_if_available(experiment_dataset, images)
        res = compute_residual(
            images * 0,
            ctf_params,
            rotation_matrices,
            translations,
            difference,
            config,
            np.array(noise_variance),
        )
        alphas[particle_indices] = 0.5 * jnp.sqrt(res)
    gamma = 1 - np.mean(scipy.stats.norm.cdf(alphas))

    return gamma
