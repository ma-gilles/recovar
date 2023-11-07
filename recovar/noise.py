import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)
## There is currently two ways to estimate noise:
# White and radial
# From my observations, white seems fine for most datasets but some need some other noise distribution
# Neither solution implemented here are very satisfying. Guessing noise in presence of heterogeneity is not trivial, since the residual doesn't seem like the correct way to do it.
# It makes me think we should have "noise pickers".

# Assume white noise, and guess variance of noise by looking at high frequency, assuming noise dominates over there

# Perhaps it should be mean at low freq and median at high freq?
mean_fn = np.mean

def estimate_noise_variance(experiment_dataset, batch_size):
    sum_sq = 0
    
    
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []

    for batch, _ in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch)
        sum_sq += np.sum(np.abs(batch)**2, axis =0)


    mean_PS =  sum_sq / experiment_dataset.n_images    
    cov_noise_mask = np.median(mean_PS)

    average_image_PS = regularization.average_over_shells(mean_PS, experiment_dataset.image_shape)

    return cov_noise_mask.astype(experiment_dataset.dtype_real), np.array(average_image_PS).astype(experiment_dataset.dtype_real)
    

def estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []
    image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    masked_image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    for batch, batch_ind in data_generator:
        masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask_inner(batch, 
                    volume_mask, experiment_dataset.rotation_matrices[batch_ind], 
                    experiment_dataset.translations[batch_ind], 
                    image_mask, 
                    experiment_dataset.volume_mask_threshold, 
                    experiment_dataset.image_shape, 
                    experiment_dataset.volume_shape, 
                    experiment_dataset.grid_size, 
                    experiment_dataset.padding, 
                    disc_type, 
                    experiment_dataset.image_stack.process_images)
        image_PSs[batch_ind] = np.array(image_PS)
        masked_image_PSs[batch_ind] = np.array(masked_image_PS)

    return masked_image_PSs, image_PSs


def upper_bound_noise_by_reprojected_mean(experiment_dataset, mean_estimate, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []
    image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    masked_image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    soften_mask =-1 # no softening
    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    for batch, batch_ind in data_generator:

        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              experiment_dataset.rotation_matrices[batch_ind], 
                                              experiment_dataset.image_stack.mask, 
                                              experiment_dataset.volume_mask_threshold, 
                                              experiment_dataset.image_shape, 
                                              experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                            experiment_dataset.padding, disc_type, soften = soften_mask )

        images = experiment_dataset.image_stack.process_images(batch)
        images = covariance_core.get_centered_images(images, mean_estimate,
                                     experiment_dataset.CTF_params[batch_ind],
                                     experiment_dataset.rotation_matrices[batch_ind],
                                     experiment_dataset.translations[batch_ind],
                                     experiment_dataset.image_shape, 
                                     experiment_dataset.volume_shape,
                                     experiment_dataset.grid_size, 
                                     experiment_dataset.voxel_size,
                                     experiment_dataset.CTF_fun,
                                     disc_type )
        image_PSs[batch_ind] = regularization.batch_average_over_shells(jnp.abs(images)**2, experiment_dataset.image_shape, 0)
        masked_images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  
        # image_size = batch.shape[-1]
        # # Integral of mask:
        image_mask_sums = jnp.sum(image_mask, axis =(-2, -1)) / experiment_dataset.image_size
        masked_image_PSs[batch_ind] = regularization.batch_average_over_shells(jnp.abs(masked_images)**2, experiment_dataset.image_shape, 0) / image_mask_sums[:,None]


    return masked_image_PSs, image_PSs


@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10,11])    
def estimate_noise_variance_from_outside_mask_inner(batch, volume_mask, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, process_fn):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften =10)
    # image_mask = image_mask > 0
    
    # Invert mask
    image_mask = 1 - image_mask

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    image_size = batch.shape[-1]
    # Integral of mask:
    image_mask_2 = ftu.get_dft2(image_mask)
    image_mask_sums = jnp.sum(jnp.abs(image_mask_2)**2, axis =(-2, -1)) / image_size**2 
    masked_image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0) / image_mask_sums[:,None]

    # masked_image_PS = masked_image_PS.at[:,0].set(masked_image_PS[:,0] *  image_mask_sums[:] )
    # import pdb; pdb.set_trace()

    return masked_image_PS, image_PS
    

# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_radial_noise_upper_bound_from_inside_mask(experiment_dataset, mean_estimate, volume_mask, batch_size):
    masked_image_PS, image_PS = upper_bound_noise_by_reprojected_mean(experiment_dataset, mean_estimate , volume_mask, batch_size, disc_type = 'linear_interp')
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0), mean_fn(image_PS, axis =0), np.std(image_PS, axis =0)


# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_radial_noise_statistic_from_outside_mask(experiment_dataset, volume_mask, batch_size):
    masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp')
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0), mean_fn(image_PS, axis =0), np.std(image_PS, axis =0)



def make_radial_noise(average_image_PS, image_shape):
    # If you pass a scalar, return a constant
    if average_image_PS.size == 1:
        return np.ones(image_shape, dtype =average_image_PS.dtype ) * average_image_PS
    
    return utils.make_radial_image(average_image_PS, image_shape, extend_last_frequency = True)




# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_noise_from_heterogeneity_residuals_inside_mask(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    masked_image_PS =  get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type )
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0)

# @functools.partial(jax.jit, static_argnums = [5])    
def get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    
    # basis_size = basis.shape[-1]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    residual_squared = jnp.zeros(experiment_dataset.image_stack.image_size, dtype = basis.dtype)
    # residuals_squared_per_image = jnp.zeros_like(residual_squared, shape = experiment_dataset.n_images, dtype = basis.dtype)
    all_averaged_residual_squared = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)
    # all_averaged_residual_squared = 
    # soften_mask = -1
    basis = jnp.asarray(basis.T)
    for batch, batch_image_ind in data_generator:
        # batch = experiment_dataset.image_stack.process_images(batch)
        averaged_residual_squared = get_average_residual_square_inner(batch, mean_estimate, volume_mask, 
                                                                        basis,
                                                                        experiment_dataset.CTF_params[batch_image_ind],
                                                                        experiment_dataset.rotation_matrices[batch_image_ind],
                                                                        experiment_dataset.translations[batch_image_ind],
                                                                        experiment_dataset.image_stack.mask,
                                                                        experiment_dataset.volume_mask_threshold,
                                                                        experiment_dataset.image_shape, 
                                                                        experiment_dataset.volume_shape, 
                                                                        experiment_dataset.grid_size, 
                                                                        experiment_dataset.voxel_size, 
                                                                        experiment_dataset.padding, 
                                                                        disc_type, 
                                                                        experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun, 
                                                                        contrasts[batch_image_ind], basis_coordinates[batch_image_ind])
        all_averaged_residual_squared[batch_image_ind] = np.array(averaged_residual_squared)

    return all_averaged_residual_squared



def get_average_residual_square_inner(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, process_fn, CTF_fun, contrasts,basis_coordinates):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type )
    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    # jnp.

    projected_mean = covariance_core.get_projected_image(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                           
                                          )

    projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
    # projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)
    AUs = covariance_core.batch_over_vol_forward_model(basis,
                                         CTF_params, 
                                         rotation_matrices,
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type )    
    # Apply mask on operator
    AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)
    image_mask_sums = jnp.sum(image_mask, axis =(-2, -1)) / batch.shape[-1]

    predicted_images = contrasts[...,None] * (jax.lax.batch_matmul(AUs, basis_coordinates[...,None])[...,0] + projected_mean)
    residual_squared = jnp.abs(batch - predicted_images)**2    / image_mask_sums[...,None]
    averaged_residual_squared = regularization.batch_average_over_shells(residual_squared, image_shape,0) 

    return averaged_residual_squared#, averaged_residual_squared
    