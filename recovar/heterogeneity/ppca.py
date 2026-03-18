import logging

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools

from recovar import core
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig
from recovar.data_io.batch_iterator import iter_batch_fields
from recovar.heterogeneity import embedding

logger = logging.getLogger(__name__)

def M_step_batch(images, lhs_summed, rhs_summed, mean_batch, covariance_batch, CTF_params, rotation_matrices, translations, image_shape, volume_shape, grid_size, voxel_size, noise_variance,  ctf):

    # Precomp piece
    CTF = ctf( CTF_params, image_shape, voxel_size)
    ctf_over_noise_variance = CTF**2 / noise_variance

    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape)
    volume_size = np.prod(volume_shape)

    # Second moments: per-image (basis, basis) weighted per-pixel by CTF^2/noise.
    second_moments = covariance_batch + linalg.broadcast_outer(mean_batch, mean_batch)  # (n_images, basis, basis)
    second_moments = second_moments.reshape(second_moments.shape[0], 1, -1)  # (n_images, 1, basis*basis)
    second_moments = second_moments * ctf_over_noise_variance[:, :, None]   # (n_images, pixels, basis*basis)

    lhs_summed = lhs_summed.at[grid_point_indices.reshape(-1)].add(second_moments.reshape(-1, second_moments.shape[-1]))

    images = core.translate_images(images, translations, image_shape)
    images = images * CTF / noise_variance
    images_means_h = linalg.broadcast_outer(images, mean_batch)  # (n_images, pixels, basis)

    rhs_summed = rhs_summed.at[grid_point_indices.reshape(-1)].add(images_means_h.reshape(-1, images_means_h.shape[-1]))

    return lhs_summed, rhs_summed



def M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size ):


    basis_size = latent_means.shape[-1]
    rhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype)
    lhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size *  basis_size), dtype = experiment_dataset.dtype)

    for images, rotation_matrices, translations, ctf_params, _noise_variance, _particle_indices, image_indices in iter_batch_fields(experiment_dataset.iterate(batch_size)):
        lhs_summed, rhs_summed = M_step_batch(
            images,
            lhs_summed,
            rhs_summed,
            latent_means[image_indices],
            latent_covariances[image_indices],
            ctf_params,
            rotation_matrices,
            translations,
            experiment_dataset.image_shape,
            experiment_dataset.volume_shape,
            experiment_dataset.grid_size,
            experiment_dataset.voxel_size,
            noise_variance,
            experiment_dataset.ctf_evaluator,
        )
        
    # Solve least squares
    lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)
    W = linalg.batch_solve(lhs_summed, rhs_summed)
    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    W = U @ jnp.diag(S)
    
    return W


def EM(experiment_dataset, mean_estimate, noise_variance, EM_iter = 20, basis_size = 10):

    # Initialize
    matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype_real)
    W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    eigenvalue = np.ones(basis_size)
    volume_mask = np.ones(experiment_dataset.volume_shape)
    contrast_grid = np.ones([1])
    batch_size = 1000
    disc_type = 'nearest'
    for iter_i in range(EM_iter):
        # E-step
        latent_means, latent_covariances, _ = embedding.get_coords_in_basis_and_contrast_3(experiment_dataset, mean_estimate, W, eigenvalue, volume_mask, noise_variance, contrast_grid, batch_size, disc_type, compute_covariances = True )

        # M-step
        W = M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size)


    return W
