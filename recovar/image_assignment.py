import logging
import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

# @functools.partial(jax.jit, static_argnums = [9,10,11,12,13,14,15,16,18, 19, 23, 24])    
def compute_residual(batch, mean_estimate,  CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, disc_type,  noise_variance, process_fn, CTF_fun):

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    projected_mean = core.forward_model_from_map(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type              
                                          )
    difference = batch - projected_mean
    difference /= jnp.sqrt(noise_variance)[None]

    # import matplotlib.pyplot as plt
    # plt.imshow(ftu.get_idft2(projected_mean[0].reshape(image_shape)).real)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(ftu.get_idft2(batch[0].reshape(image_shape)).real)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(ftu.get_idft2((batch[0] - projected_mean[0]).reshape(image_shape)).real)
    # plt.colorbar()
    # plt.show()

    # import pdb; pdb.set_trace()

    return jnp.linalg.norm(difference, axis = -1)**2  




# @functools.partial(jax.jit, static_argnums = [5])    
def compute_image_assignment(experiment_dataset, volumes,  noise_variance, batch_size, disc_type = 'cubic'):


    if disc_type == 'cubic':
        from recovar import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory= None)


    volumes = jnp.array(volumes).astype(experiment_dataset.dtype)
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    residuals = np.zeros((volumes.shape[0], experiment_dataset.n_units), dtype = experiment_dataset.dtype_real)
    for batch, particles_ind, batch_image_ind in data_generator:
        for volume_ind in range(volumes.shape[0]):
            residuals[volume_ind,particles_ind] = compute_residual(batch, volumes[volume_ind], 
                                                                            experiment_dataset.CTF_params[batch_image_ind],
                                                                            experiment_dataset.rotation_matrices[batch_image_ind],
                                                                            experiment_dataset.translations[batch_image_ind],
                                                                            experiment_dataset.image_shape, 
                                                                            experiment_dataset.volume_shape, 
                                                                            experiment_dataset.voxel_size, 
                                                                            disc_type,
                                                                            np.array(noise_variance),
                                                                            experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun )             
    # assignment = jnp.argmin(residuals, axis = 0)
    return residuals


# @functools.partial(jax.jit, static_argnums = [5])    
def estimate_false_positive_rate(experiment_dataset, volumes,  noise_variance, batch_size, disc_type = 'cubic'):

    if disc_type == 'cubic':
        from recovar import covariance_estimation
        volumes = covariance_estimation.compute_spline_coeffs_in_batch(volumes, experiment_dataset.volume_shape, gpu_memory= None)

    volumes = jnp.array(volumes).astype(experiment_dataset.dtype)
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    alphas = np.zeros(( experiment_dataset.n_units), dtype = experiment_dataset.dtype_real)
    assert volumes.shape[0] ==2, 'Only two volumes are supported'
    difference = volumes[0] - volumes[1]
    for batch, particles_ind, batch_image_ind in data_generator:
        res = compute_residual(batch * 0, difference,experiment_dataset.CTF_params[batch_image_ind],
                                                                        experiment_dataset.rotation_matrices[batch_image_ind],
                                                                        experiment_dataset.translations[batch_image_ind],
                                                                        experiment_dataset.image_shape, 
                                                                        experiment_dataset.volume_shape, 
                                                                        experiment_dataset.voxel_size, 
                                                                        disc_type,
                                                                        np.array(noise_variance),
                                                                        experiment_dataset.image_stack.process_images,
                                                                    experiment_dataset.CTF_fun )
        alphas[particles_ind] = 0.5 * jnp.sqrt(res)
    import scipy.stats
    gamma = 1 - np.mean(scipy.stats.norm.cdf(alphas))

    return gamma