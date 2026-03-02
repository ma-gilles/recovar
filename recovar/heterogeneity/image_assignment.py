import logging

import jax.numpy as jnp
import numpy as np
import scipy.stats

from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)


def compute_residual(batch, mean_estimate, CTF_params, rotation_matrices, translations,
                     config, noise_variance, process_fn):

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations, config.image_shape)

    projected_mean = core_forward.forward_model(
        config, mean_estimate, CTF_params, rotation_matrices,
    )
    difference = batch - projected_mean
    difference /= jnp.sqrt(noise_variance)[None]

    return jnp.linalg.norm(difference, axis=-1)**2


def compute_image_assignment(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    logger.info("Computing image assignment: %d volumes, %d images, batch_size=%d",
                volumes.shape[0], experiment_dataset.n_units, batch_size)
    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    residuals = np.zeros((volumes.shape[0], experiment_dataset.n_units), dtype=experiment_dataset.dtype_real)
    for batch, particles_ind, batch_image_ind in data_generator:
        for volume_ind in range(volumes.shape[0]):
            residuals[volume_ind, particles_ind] = compute_residual(
                batch, volumes[volume_ind],
                experiment_dataset.CTF_params[batch_image_ind],
                experiment_dataset.rotation_matrices[batch_image_ind],
                experiment_dataset.translations[batch_image_ind],
                config,
                np.array(noise_variance),
                experiment_dataset.image_stack.process_images,
            )
    return residuals


def estimate_false_positive_rate(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    alphas = np.zeros((experiment_dataset.n_units,), dtype=experiment_dataset.dtype_real)
    if volumes.shape[0] != 2:
        raise ValueError(f"Only two volumes are supported, got {volumes.shape[0]}")
    difference = volumes[0] - volumes[1]
    for batch, particles_ind, batch_image_ind in data_generator:
        res = compute_residual(
            batch * 0, difference,
            experiment_dataset.CTF_params[batch_image_ind],
            experiment_dataset.rotation_matrices[batch_image_ind],
            experiment_dataset.translations[batch_image_ind],
            config,
            np.array(noise_variance),
            experiment_dataset.image_stack.process_images,
        )
        alphas[particles_ind] = 0.5 * jnp.sqrt(res)
    gamma = 1 - np.mean(scipy.stats.norm.cdf(alphas))

    return gamma
