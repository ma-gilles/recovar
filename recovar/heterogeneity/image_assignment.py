import logging

import jax.numpy as jnp
import numpy as np
import scipy.stats

from recovar import core
from recovar.core.configs import DataIterator, ForwardModelConfig
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)
## TODO: clean this up, document it (should be very non probably since no one has asked for this, but nevertheless an interesting feature)

def compute_residual(batch_data, mean_estimate, config, noise_variance):
    images = core.translate_images(batch_data.images, batch_data.translations, config.image_shape)

    projected_mean = core_forward.forward_model(
        config, mean_estimate, batch_data.ctf_params, batch_data.rotation_matrices,
    )
    difference = images - projected_mean
    difference /= jnp.sqrt(noise_variance)[None]

    return jnp.linalg.norm(difference, axis=-1)**2


def compute_image_assignment(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, (experiment_dataset.grid_size,)*3, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    logger.info("Computing image assignment: %d volumes, %d images, batch_size=%d",
                volumes.shape[0], experiment_dataset.n_units, batch_size)
    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    residuals = np.zeros((volumes.shape[0], experiment_dataset.n_units), dtype=experiment_dataset.dtype_real)
    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        apply_process_images=True,
    ):
        for volume_ind in range(volumes.shape[0]):
            residuals[volume_ind, batch_data.particle_indices] = compute_residual(
                batch_data, volumes[volume_ind], config, np.array(noise_variance),
            )
    return residuals


def estimate_false_positive_rate(experiment_dataset, volumes, noise_variance, batch_size, disc_type='cubic'):

    if disc_type == 'cubic':
        from recovar.heterogeneity import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, (experiment_dataset.grid_size,)*3, gpu_memory=None)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    volumes = jnp.asarray(volumes, dtype=experiment_dataset.dtype)
    alphas = np.zeros((experiment_dataset.n_units,), dtype=experiment_dataset.dtype_real)
    if volumes.shape[0] != 2:
        raise ValueError(f"Only two volumes are supported, got {volumes.shape[0]}")
    difference = volumes[0] - volumes[1]
    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        apply_process_images=True,
    ):
        # Zero out images — compute residual of the difference volume alone
        from recovar.core.configs import BatchData
        zero_batch = BatchData(
            images=batch_data.images * 0,
            rotation_matrices=batch_data.rotation_matrices,
            translations=batch_data.translations,
            ctf_params=batch_data.ctf_params,
            particle_indices=batch_data.particle_indices,
            image_indices=batch_data.image_indices,
        )
        res = compute_residual(zero_batch, difference, config, np.array(noise_variance))
        alphas[batch_data.particle_indices] = 0.5 * jnp.sqrt(res)
    gamma = 1 - np.mean(scipy.stats.norm.cdf(alphas))

    return gamma
