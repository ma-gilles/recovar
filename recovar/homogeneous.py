import logging
import jax.numpy as jnp
import numpy as np
import jax, functools, time

from recovar import core, regularization, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def image_weight_parse(image_weights, dtype = np.float32):
    if image_weights is None:
        return [None, None]
    else:
        return [w.astype(dtype) for w in image_weights]



def get_mean_conformation(cryos, batch_size, cov_noise = None, valid_idx = None, disc_type ='linear_interp', use_noise_level_prior = False, grad_n_iter = 5, image_weights = None):
    
    cryo = cryos[0]
    valid_idx = cryo.get_valid_frequency_indices() if valid_idx is None else valid_idx
    if cov_noise is None:
        cov_noise, _ = estimate_noise_variance(cryos, batch_size) 
    
    image_weights = image_weight_parse(image_weights, dtype = cryos[0].dtype_real)
    means = {}

    # This is kind of a stupid way to code this. Should probably rewrite next 3 forloops
    st_time = time.time()
    for cryo_idx,cryo in enumerate(cryos):
        means["init" +str(cryo_idx) ] = np.array(compute_mean_volume(cryo, cov_noise,  jnp.ones(cryo.volume_size, dtype = cryo.dtype_real) * 1e18, batch_size, disc_type = disc_type, n_iter = 1, image_weights = image_weights[cryo_idx]))
        
    mean_prior, fsc, _ = regularization.compute_relion_prior(cryos, cov_noise, means["init0"], means["init1"], batch_size)
    for cryo_idx, cryo in enumerate(cryos):
        means["corrected" +str(cryo_idx) ] = np.array(compute_mean_volume(cryo, cov_noise,  mean_prior, batch_size, disc_type = disc_type, n_iter = grad_n_iter, image_weights = image_weights[cryo_idx]))
        
    mean_prior, fsc, _ = regularization.compute_relion_prior(cryos, cov_noise, means["corrected0"], means["corrected1"], batch_size)        
    lhs = 0; rhs = 0 
    for cryo_idx, cryo in enumerate(cryos):
        lhs_t, rhs_t = compute_mean_volume(cryo, cov_noise,  mean_prior, batch_size, disc_type = disc_type, n_iter = 1, return_lhs_rhs = True, mean_estimate = (means["corrected0"] + means["corrected1"])/2 , image_weights = image_weights[cryo_idx] )
        lhs +=  lhs_t
        rhs +=  rhs_t
            
    # Probably should be rewritten around here.
    if use_noise_level_prior:
        key = "corrected"
        mean_prior, fsc, prior_avg = regularization.compute_relion_prior(cryos, cov_noise, means["corrected0"], means["corrected1"], batch_size, estimate_merged_SNR = False)
        logger.info(f"Using RELION prior")

    else:    
        mean_prior, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, means["corrected0"], means["corrected1"], lhs/2 , mean_prior, frequency_shift = jnp.array([0,0,0]))
        logger.info(f"Using new prior")

    mean_prior = np.array(mean_prior)
    means["combined"] = np.array( jnp.where(jnp.abs(lhs) < 1e-8, 0, rhs / (lhs + 1 / mean_prior ) ))
    means["prior"] = mean_prior
    
    end_time = time.time()
    logger.info(f"time to compute means: {end_time- st_time}")

    return means, mean_prior, fsc, lhs


def get_multiple_conformations(cryos, cov_noise, disc_type, batch_size, mean_prior, mean ,image_weights, recompute_prior = True, volume_mask = None ):
    image_weights = image_weight_parse(image_weights, dtype = cryos[0].dtype_real)

    st_time = time.time()

    num_reconstructions = image_weights[0].shape[0]
    lhs = 0; rhs = 0 
    lhs_l = []
    rhs_l = []
    for cryo_idx, cryo in enumerate(cryos):
        lhs_t, rhs_t = compute_mean_volume(cryo, cov_noise,  mean_prior, batch_size, disc_type = disc_type, n_iter = 1, return_lhs_rhs = True, mean_estimate = mean , image_weights = image_weights[cryo_idx].astype(cryo.dtype_real) )
        lhs_l.append(jnp.real(lhs_t).astype(cryo.dtype_real))
        rhs_l.append(rhs_t)
        # lhs +=  lhs_t
        # rhs +=  rhs_t
        
    
    if recompute_prior:
        priors, fscs_this = regularization.prior_iteration_batch(lhs_l[0], lhs_l[1], rhs_l[0], rhs_l[1], np.zeros((num_reconstructions,3)), mean_prior[None].repeat(num_reconstructions,0) , False,  cryos[0].volume_shape, 3 ) 
    else:
        priors = mean_prior[None]
        logger.info("Not recomputing prior?? Make sure this is what you want")

    # Dump on CPU

    # I have no idea why all of the sudden this broke. Something about JAX -> np?
    priors = np.abs(np.array(priors)) + constants.EPSILON

    lhs_l[0] = np.array(lhs_l[0], dtype = lhs_l[0].dtype )
    lhs_l[1] = np.array(lhs_l[1], dtype = lhs_l[1].dtype )
    rhs_l[0] = np.array(rhs_l[0], dtype = rhs_l[0].dtype )
    rhs_l[1] = np.array(rhs_l[1], dtype = rhs_l[1].dtype )

    half_maps = []

    for s in range(2):
        half_maps.append(np.array( np.where(np.abs(lhs_l[s]) < constants.ROOT_EPSILON, 0, rhs_l[s] / (lhs_l[s] + 1 / priors ) )))

    # Add second to first
    lhs_l[0] += lhs_l[1]
    rhs_l[0] += rhs_l[1]
    reconstructions = np.array( np.where( np.abs(lhs_l[0]) < constants.ROOT_EPSILON, 0, rhs_l[0] / (lhs_l[0] + 1 / priors ) ))
    end_time = time.time()
    logger.info(f"time to compute reweighted conformations: {end_time- st_time}")
    
    return reconstructions, half_maps



## Mean functions
def compute_mean_volume(experiment_dataset, cov_noise, cov_diag_prior, batch_size, disc_type, n_iter = 2, image_weights = None, return_lhs_rhs = False, mean_estimate = None ):

    if disc_type =="nearest":
        n_iter = 1        
        
    if return_lhs_rhs:
        n_iter =1
    
    for _ in range(n_iter):
        # Make sure mean_estimate is size # volume_size ?
        if mean_estimate is not None:
            mean_estimate = mean_estimate.reshape(experiment_dataset.volume_size)
        mean_estimate = solve_least_squares_mean_iteration(experiment_dataset , 
                                            cov_diag_prior,
                                            cov_noise, 
                                            batch_size, 
                                            mean_estimate = mean_estimate, 
                                            image_weights = image_weights,
                                            disc_type = disc_type,
                                            return_lhs_rhs = return_lhs_rhs)
    return mean_estimate


# Solves the linear system Dx = b.
def solve_least_squares_mean_iteration(experiment_dataset , cov_diag_prior, cov_noise,  batch_size, mean_estimate, image_weights = None, disc_type = None, return_lhs_rhs = False ):
    # all_one_volume = jnp.ones(volume_size)
    
    mean_rhs = jnp.zeros(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
    diag_mean = jnp.zeros(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for batch, indices in data_generator:
        
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = True)
        
        if image_weights is None:
            image_weights_batch = jnp.ones((1,batch.shape[0]), dtype = experiment_dataset.dtype_real )
        else:
            image_weights_batch = image_weights[...,indices]
            
        mean_rhs_this, diag_mean_this = compute_batch_mean_rhs(batch,
                                         experiment_dataset.rotation_matrices[indices], 
                                         experiment_dataset.translations[indices], 
                                         experiment_dataset.CTF_params[indices], 
                                         mean_estimate,
                                         cov_noise,
                                         image_weights_batch,
                                         experiment_dataset.voxel_size, 
                                         experiment_dataset.volume_shape, 
                                         experiment_dataset.image_shape, 
                                         experiment_dataset.grid_size, 
                                         disc_type,
                                         experiment_dataset.CTF_fun)
        mean_rhs += mean_rhs_this
        diag_mean += diag_mean_this

    if return_lhs_rhs:
        return diag_mean, mean_rhs

    X_mean = jnp.where(jnp.abs(diag_mean) < 1e-8, 0, mean_rhs / (diag_mean + 1 / cov_diag_prior ) )
    return X_mean

def estimate_noise_variance(experiment_dataset, batch_size):
    sum_sq = 0
    
    
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []

    for batch, _ in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch)
        sum_sq += np.sum(np.abs(batch)**2, axis =0)

        # shell_avgs = np.abs(batch)**2
        # batch_average_over_shells
        # shell_avgs = regularization.batch_average_over_shells(shell_avgs, experiment_dataset.image_shape)
        # all_shell_avgs.append(shell_avgs)


    mean_PS =  sum_sq / experiment_dataset.n_images    
    cov_noise_mask = np.median(mean_PS)

    average_image_PS = regularization.average_over_shells(mean_PS, experiment_dataset.image_shape)

    return cov_noise_mask.astype(experiment_dataset.dtype_real), np.array(average_image_PS).astype(experiment_dataset.dtype_real)
    

@functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
def compute_batch_mean_rhs(images, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    volume_size = np.prod(np.array(volume_shape))
    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    translated_images = core.translate_images(images, translations, image_shape)
    if disc_type != "nearest" and mean_estimate is not None:
        grad_correction = (core.get_slices(mean_estimate, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) \
                        - core.get_slices(mean_estimate, rotation_matrices, image_shape, volume_shape, grid_size, "nearest")) * CTF
        corrected_images = translated_images - grad_correction
    else:
        corrected_images = translated_images
    
    all_one_volume = jnp.ones(volume_size, dtype = images.dtype)    
    ones_mapped = core.forward_model(all_one_volume, CTF, grid_point_indices) / noise_variance[None] * image_weights[...,None]
    # diag_mean = core.sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)
    diag_mean = batch_over_weights_sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)

    corrected_images = corrected_images * image_weights[...,None] / noise_variance[None]
    # mean_rhs = core.sum_adj_forward_model(volume_size, corrected_images  / noise_variance[None] , CTF, grid_point_indices)
    mean_rhs = batch_over_weights_sum_adj_forward_model(volume_size, corrected_images , CTF, grid_point_indices)
    return mean_rhs, diag_mean

batch_over_weights_sum_adj_forward_model = jax.vmap(core.sum_adj_forward_model, in_axes = (None,0, None,None))
