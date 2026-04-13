import logging

import jax.numpy as jnp
import numpy as np
import scipy.stats

from recovar import core
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


def compute_image_assignment(experiment_dataset, volumes, noise_variance, batch_size, disc_type="cubic"):

    projection_coeffs = volumes
    if disc_type == "cubic":
        from recovar.heterogeneity import covariance_estimation

        projection_coeffs = covariance_estimation.compute_spline_coeffs_in_batch(
            projection_coeffs, experiment_dataset.volume_shape, gpu_memory=None
        )

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    logger.info(
        "Computing image assignment: %d volumes, %d images, batch_size=%d",
        projection_coeffs.shape[0],
        experiment_dataset.n_units,
        batch_size,
    )
    projection_coeffs = jnp.asarray(projection_coeffs, dtype=experiment_dataset.dtype)
    residuals = np.zeros((projection_coeffs.shape[0], experiment_dataset.n_units), dtype=experiment_dataset.dtype_real)
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
        for volume_ind in range(projection_coeffs.shape[0]):
            projection_volume = (
                core.CubicVolume.from_coeffs(projection_coeffs[volume_ind])
                if disc_type == "cubic"
                else core.Volume(projection_coeffs[volume_ind], disc_type=disc_type)
            )
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


def estimate_false_positive_rate(experiment_dataset, volumes, noise_variance, batch_size, disc_type="cubic"):

    projection_coeffs = volumes
    if disc_type == "cubic":
        from recovar.heterogeneity import covariance_estimation

        projection_coeffs = covariance_estimation.compute_spline_coeffs_in_batch(
            projection_coeffs, experiment_dataset.volume_shape, gpu_memory=None
        )

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    projection_coeffs = jnp.asarray(projection_coeffs, dtype=experiment_dataset.dtype)
    alphas = np.zeros((experiment_dataset.n_units,), dtype=experiment_dataset.dtype_real)
    if projection_coeffs.shape[0] != 2:
        raise ValueError(f"Only two volumes are supported, got {projection_coeffs.shape[0]}")
    difference_coeffs = projection_coeffs[0] - projection_coeffs[1]
    if disc_type == "cubic":
        difference = core.CubicVolume.from_coeffs(difference_coeffs)
    else:
        difference = core.Volume(difference_coeffs, disc_type=disc_type)
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
