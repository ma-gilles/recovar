import inspect
import logging
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import scipy.stats

from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.data_io.batch_iterator import coerce_batch_fields, iter_batch_fields
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)
## TODO: clean this up, document it (should be very non probably since no one has asked for this, but nevertheless an interesting feature)


def _process_images_if_available(experiment_dataset, images):
    process_images = getattr(experiment_dataset, "process_images", None)
    if process_images is None:
        return images
    return process_images(images)

def compute_residual(
    images=None,
    ctf_params=None,
    rotation_matrices=None,
    translations=None,
    mean_estimate=None,
    config=None,
    noise_variance=None,
    **kwargs,
):
    batch_data = kwargs.pop("batch_data", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if batch_data is not None:
        images, rotation_matrices, translations, ctf_params, batch_noise, _, _ = coerce_batch_fields(batch_data)
        if noise_variance is None:
            noise_variance = batch_noise
    else:
        images, rotation_matrices, translations, ctf_params, batch_noise, _, _ = coerce_batch_fields(
            images,
            rotation_matrices=rotation_matrices,
            translations=translations,
            ctf_params=ctf_params,
            noise_variance=noise_variance,
        )
        if noise_variance is None:
            noise_variance = batch_noise

    images = core.translate_images(images, translations, config.image_shape)

    projected_mean = core_forward.forward_model(
        config, mean_estimate, ctf_params, rotation_matrices,
    )
    difference = images - projected_mean
    difference /= jnp.sqrt(noise_variance)[None]

    return jnp.linalg.norm(difference, axis=-1)**2


def _call_compute_residual(
    images,
    ctf_params,
    rotation_matrices,
    translations,
    mean_estimate,
    config,
    noise_variance,
    *,
    particle_indices=None,
    image_indices=None,
):
    params = inspect.signature(compute_residual).parameters
    if "images" in params:
        return compute_residual(
            images=images,
            ctf_params=ctf_params,
            rotation_matrices=rotation_matrices,
            translations=translations,
            mean_estimate=mean_estimate,
            config=config,
            noise_variance=noise_variance,
        )

    legacy_batch = SimpleNamespace(
        images=images,
        rotation_matrices=rotation_matrices,
        translations=translations,
        ctf_params=ctf_params,
        noise_variance=noise_variance,
        particle_indices=particle_indices,
        image_indices=image_indices,
    )
    return compute_residual(
        batch_data=legacy_batch,
        mean_estimate=mean_estimate,
        config=config,
        noise_variance=noise_variance,
    )


def compute_image_assignment(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    logger.info("Computing image assignment: %d volumes, %d images, batch_size=%d",
                volumes.shape[0], experiment_dataset.n_units, batch_size)
    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    residuals = np.zeros((volumes.shape[0], experiment_dataset.n_units), dtype=experiment_dataset.dtype_real)
    for images, rotation_matrices, translations, ctf_params, _noise_variance, particle_indices, image_indices in iter_batch_fields(experiment_dataset.iterate(batch_size)):
        images = _process_images_if_available(experiment_dataset, images)
        for volume_ind in range(volumes.shape[0]):
            residuals[volume_ind, particle_indices] = _call_compute_residual(
                images,
                ctf_params,
                rotation_matrices,
                translations,
                volumes[volume_ind],
                config,
                np.array(noise_variance),
                particle_indices=particle_indices,
                image_indices=image_indices,
            )
    return residuals


def estimate_false_positive_rate(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    alphas = np.zeros((experiment_dataset.n_units,), dtype=experiment_dataset.dtype_real)
    if volumes.shape[0] != 2:
        raise ValueError(f"Only two volumes are supported, got {volumes.shape[0]}")
    difference = volumes[0] - volumes[1]
    for images, rotation_matrices, translations, ctf_params, _noise_variance, particle_indices, image_indices in iter_batch_fields(experiment_dataset.iterate(batch_size)):
        images = _process_images_if_available(experiment_dataset, images)
        res = _call_compute_residual(
            images * 0,
            ctf_params,
            rotation_matrices,
            translations,
            difference,
            config,
            np.array(noise_variance),
            particle_indices=particle_indices,
            image_indices=image_indices,
        )
        alphas[particle_indices] = 0.5 * jnp.sqrt(res)
    gamma = 1 - np.mean(scipy.stats.norm.cdf(alphas))

    return gamma
