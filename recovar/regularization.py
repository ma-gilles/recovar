import jax.numpy as jnp
import numpy as np
import jax, functools

from recovar import core, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

## Mean prior computation

#@functools.partial(jax.jit, static_argnums = [7,8,9,10, 11, 12,13])    
def compute_batch_prior_quantities(rotation_matrices, translations, CTF_params, noise_variance, voxel_size, dtype, volume_shape, image_shape, grid_size, CTF_fun , for_whitening = False):
    volume_size = np.prod(np.array(volume_shape))
    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    all_one_volume = jnp.ones(volume_size, dtype = dtype)    
    
    # if for_whitening:
    #     ones_CTF_mapped = jnp.sqrt(core.forward_model(all_one_volume, CTF, grid_point_indices) * CTF )
    # else:
    ones_CTF_mapped = core.forward_model(all_one_volume, CTF, grid_point_indices) * CTF / noise_variance[None] 
    diag_mean = core.sum_adj_forward_model(volume_size, ones_CTF_mapped, jnp.ones_like(CTF), grid_point_indices)
    
    return diag_mean

def compute_prior_quantites(experiment_datasets, cov_noise, batch_size, for_whitening = False ):
    bottom_of_fraction = jnp.zeros(experiment_datasets[0].volume_size, dtype = experiment_datasets[0].image_stack.dtype)
    for experiment_dataset in experiment_datasets:
        n_images = experiment_dataset.n_images
        # Compute the bottom of fraction.
        for k in range(0, int(np.ceil(n_images/batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min( [(k+1) * batch_size, n_images]))
            indices = jnp.arange(batch_st, batch_end)
            bottom_of_fraction_this = compute_batch_prior_quantities(
                                             experiment_dataset.rotation_matrices[indices], 
                                             experiment_dataset.translations[indices], 
                                             experiment_dataset.CTF_params[indices], 
                                             cov_noise,
                                             experiment_dataset.voxel_size, 
                                             experiment_dataset.dtype,
                                             experiment_dataset.volume_shape, 
                                             experiment_dataset.image_shape, 
                                             experiment_dataset.grid_size, 
                                             experiment_dataset.CTF_fun, 
                                             for_whitening)
            
            bottom_of_fraction += bottom_of_fraction_this
        
    bottom_of_fraction = bottom_of_fraction.real / len(experiment_datasets)
    return bottom_of_fraction 
    

def compute_relion_prior(experiment_datasets, cov_noise, image0, image1, batch_size, estimate_merged_SNR = False, noise_level = None):
    
    if noise_level is not None:
        bottom_of_fraction = noise_level
        from_noise_level = True
    else:
        bottom_of_fraction = compute_prior_quantites(experiment_datasets, cov_noise, batch_size, for_whitening = False )
        from_noise_level = False
    
    return compute_fsc_prior_gpu(experiment_datasets[0].volume_shape, image0, image1, bottom_of_fraction, estimate_merged_SNR = estimate_merged_SNR, from_noise_level = from_noise_level )

# @functools.partial(jax.jit, static_argnums = [7,8,9,10, 11, 12,13])    
def get_fsc_gpu(vol1, vol2, volume_shape, substract_shell_mean = False, frequency_shift = 0):
    
    if substract_shell_mean:
        # Center two volumes.
        vol1_avg = average_over_shells(vol1, volume_shape, frequency_shift = frequency_shift)
        vol2_avg = average_over_shells(vol2, volume_shape, frequency_shift = frequency_shift)
        radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
        vol1 -= vol1_avg[radial_distances]
        vol2 -= vol2_avg[radial_distances]

    top = jnp.conj(vol1) * vol2    
    top_avg = average_over_shells(top.real, volume_shape, frequency_shift = frequency_shift)
    bot1 = average_over_shells(jnp.abs(vol1)**2, volume_shape, frequency_shift = frequency_shift)
    bot2 = average_over_shells(jnp.abs(vol2)**2, volume_shape, frequency_shift = frequency_shift)    
    bot = jnp.sqrt(bot1 * bot2)
    fsc = jnp.where(bot  > constants.EPSILON , top_avg / bot, constants.EPSILON)
    fsc = fsc.at[0].set(fsc[1]) # Always set this 1st shell?
    return fsc


# @functools.partial(jax.jit, static_argnums = [1,2])
def average_over_shells(input_vec, volume_shape, frequency_shift = 0 ):
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1) 
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0]//2 - 1)        
    return jax_scipy_nd_image_mean(input_vec.reshape(-1), labels = labels, index = indices)    


def jax_scipy_nd_image_mean(input, labels=None, index=None):
    if input.dtype == 'complex64':
        input = input.astype('complex128') #jax.numpy.bincount complex64 version seems to be bugged.
        return jax_scipy_nd_image_mean(input.reshape(-1), labels = labels, index = index).astype('complex64')
    return jax_scipy_nd_image_mean_inner(input, labels = labels, index = index)

def jax_scipy_nd_image_mean_inner(input, labels=None, index=None):
    # A jittable simplified scipy.ndimage.mean method
    # numpy = np
    # unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    # new_labels = labels
    
    # # counts = numpy.bincount(new_labels,length = index.size )
    # counts = numpy.bincount(new_labels)#,length = index.size )

    # # sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    # sums = numpy.bincount(new_labels, weights=input.ravel())#,length = index.size )


    numpy = jnp
    unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    new_labels = labels
    
    # counts = numpy.bincount(new_labels,length = index.size )
    counts = numpy.bincount(new_labels,length = index.size )

    # sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )


    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = numpy.where( idxs >= int(unique_labels.size), 0, idxs)

    found = (unique_labels[idxs] == index)
    counts = counts[idxs]
    counts = numpy.where(found, counts, 0)
    sums = sums[idxs]

    sums = numpy.where(sums, sums, 0)
    return sums / counts


def sum_over_shells(input_vec, volume_shape, frequency_shift = 0 ):
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1) 
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0]//2 - 1)
    return jax_scipy_nd_image_sum(input_vec.reshape(-1), labels = labels, index = indices)    

def jax_scipy_nd_image_sum(input, labels=None, index=None):
    # A jittable simplified scipy.ndimage.mean method
    numpy = jnp
    unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    new_labels = labels
    
    counts = numpy.bincount(new_labels,length = index.size )
    sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    
    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = jnp.where( idxs >= int(unique_labels.size), 0, idxs)

    found = (unique_labels[idxs] == index)
    counts = counts[idxs]
    counts = jnp.where(found, counts, 0)
    sums = sums[idxs]

    sums = jnp.where(sums, sums, 0)
    return sums


    
def compute_fsc_prior_gpu(volume_shape, image0, image1, bottom_of_fraction = None, estimate_merged_SNR = False, substract_shell_mean = False, frequency_shift = 0, from_noise_level = False):
    epsilon = constants.EPSILON
    # FSC top:
    fsc = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)
    fsc = jnp.where(fsc > epsilon , fsc, epsilon )
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon )
    if estimate_merged_SNR:
        fsc = 2 * fsc / ( 1 + fsc )
        
    # SNR = jnp.where(fsc < 1 - epsilon, fsc / ( 1 - fsc), jnp.inf)
    SNR = fsc / (1 - fsc)
    
    # Bottom of fraction
    if from_noise_level:        
        # bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)        
        prior_avg = SNR * bottom_of_fraction #jnp.where( bottom_avg > 0 , SNR * bottom_avg, epsilon )
        print("used outdated prior!!!! Change this!")
    else:
        bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)        
        prior_avg = jnp.where( bottom_avg > 0 , SNR / bottom_avg, epsilon )
    
    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]
    
    return prior, fsc, prior_avg


@functools.partial(jax.jit, static_argnums = [0,6])    
def compute_fsc_prior_gpu_v2(volume_shape, image0, image1, lhs , prior, frequency_shift , substract_shell_mean = False ):
    epsilon = constants.EPSILON
    # FSC top:
    fsc = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)
    fsc = jnp.where(fsc > epsilon , fsc, epsilon )
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon )
        
    # SNR = jnp.where(fsc < 1 - epsilon, fsc / ( 1 - fsc), jnp.inf)
    SNR = fsc / (1 - fsc)
    
    top = lhs**2 / (lhs + 1/prior)**2
    sum_top = average_over_shells(top,  volume_shape, frequency_shift)    
    
    bot = lhs / (lhs + 1/prior)**2
    sum_bot = average_over_shells(bot,  volume_shape, frequency_shift)    
    
    prior_avg = jnp.where( sum_top > 0 , SNR * sum_bot / sum_top , epsilon )
    
    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]
    
    return prior, fsc, prior_avg



def covariance_update_col(H, B, prior, epsilon = constants.EPSILON):
    # H is not divided by sigma.
    cov = jnp.where( jnp.abs(H) < epsilon , 0,  B / ( H + (1 / prior) ) )
    return cov

def covariance_update_col_with_mask(H, B, prior, volume_mask, valid_idx, volume_shape, epsilon = constants.EPSILON):
    # H is not divided by sigma.
    cov = (jnp.where( jnp.abs(H) < epsilon , 0,  B / ( H + (1 / prior) ) ) * valid_idx).reshape(volume_shape)
    cov = ftu.get_dft3( ftu.get_idft3(cov ) * volume_mask ).reshape(-1)
    return cov

@functools.partial(jax.jit, static_argnums = [6,7,8])    
def prior_iteration(H0, H1, B0, B1, shift, init_regularization, substract_shell_mean, volume_shape, prior_iterations ):
    
    H_comb = (H0 +  H1)/2
    prior = init_regularization

    # Harcoded iterations because I couldn't figure out to make jit the loop properly...
    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)
    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)
    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)
    
    return prior, fsc

prior_iteration_batch = jax.vmap(prior_iteration, in_axes = (0,0,0,0, 0, 0, None, None, None) )


def compute_masked_fscs(H0, B0, H1, B1, prior, volume_shape, volume_mask):
    volumes1 =  covariance_update_col(H0,B0, prior)
    volumes2 =  covariance_update_col(H1,B1, prior)
    
    def apply_masks(volumes):
        vols_real = ftu.get_idft3(volumes.reshape([-1, *volume_shape])) * volume_mask
        return ftu.get_dft3(vols_real).reshape([vols_real.shape[0], -1])
    
    volumes1_masked = apply_masks(volumes1)
    volumes2_masked = apply_masks(volumes2)
    
    _, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, volumes1_masked, volumes2_masked, H0 , prior, frequency_shift = 0)

    return fsc

batch_average_over_shells = jax.vmap(average_over_shells, in_axes = (0,None,None))


