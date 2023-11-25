import logging
import jax.numpy as jnp
import numpy as np
import jax, functools, time

from recovar import core, regularization, constants, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils

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
        cov_noise, signal_var = noise.estimate_noise_variance(cryo, batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(cryo, batch_size)
        signal_var = np.max(signal_var)

    regularization_init = jnp.ones(cryo.volume_size, dtype = cryo.dtype_real) * signal_var
    logger.info("regularization init done")
    image_weights = image_weight_parse(image_weights, dtype = cryos[0].dtype_real)
    means = {}

    # This is kind of a stupid way to code this. Should probably rewrite next 3 forloops
    st_time = time.time()
    for cryo_idx,cryo in enumerate(cryos):
        means["init" +str(cryo_idx) ] = np.array(compute_mean_volume(cryo, cov_noise,  regularization_init, batch_size, disc_type = disc_type, n_iter = 1, image_weights = image_weights[cryo_idx]))
        
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


def estimate_derivative_norm(volume):
    # Compute gradient of volume
    grad = jnp.gradient(volume)
    grad_norm = jnp.linalg.norm(grad, axis = 0)
    return grad




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
        print('TOOK OUT IMAGE MASK IN MEAN!!! PUT IT BACK??')
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        
        if image_weights is None:
            image_weights_batch = jnp.ones((1,batch.shape[0]), dtype = experiment_dataset.dtype_real )
        else:
            image_weights_batch = image_weights[...,indices]
            
        mean_rhs_this, diag_mean_this = compute_mean_least_squares_rhs_lhs(batch,
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
    

@functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
def compute_mean_least_squares_rhs_lhs(images, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
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

# def compute_weight_matrix_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun ):
#     volume_size = np.prod(np.array(volume_shape))
#     C_mat, grid_point_vec_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
#     # This is going to be stroed twice as much stuff as it needs to be
#     C_mat_outer = batch_batch_outer(C_mat).reshape([C_mat.shape[0], C_mat.shape[1], C_mat.shape[2]*C_mat.shape[2]])#.transpose([2,0,1])
#     RR = core.batch_over_vol_summed_adjoint_projections_nearest(volume_size, C_mat_outer, grid_point_vec_indices)
#     return RR

def compute_weight_matrix_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist ):
    volume_size = np.prod(np.array(volume_shape))
    # C_mat, grid_point_vec_indices = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)
    # This is going to be stroed twice as much stuff as it needs to be
    if grid_dist is None:
        C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
    else:
        C_mat, grid_point_indices = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)


    C_mat_outer = broadcast_outer(C_mat)


    RR = core.batch_over_vol_summed_adjoint_projections_nearest(volume_size, C_mat_outer.reshape(-1,C_mat.shape[-2]*C_mat.shape[-1] ), grid_point_indices.reshape(-1))
    return RR



def make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun):

    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    # Discretized grid points
    # This could be done more efficiently
    grid_points_coords_nearest = core.round_to_int(grid_points_coords)
    differences = grid_points_coords - grid_points_coords_nearest
    C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    C_mat *= CTF[...,None]

    return C_mat, grid_point_vec_indices


def make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist):
    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
    differences = near_frequencies - grid_points_coords[...,None,:]
    distances = jnp.linalg.norm(differences, axis = -1)
    valid_points = (distances <= max_dist) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])
    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
    
    # This could be done more efficiently

    C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    C_mat *= CTF[...,None,None]
    C_mat = C_mat * valid_points[...,None]

    C_mat_veced = C_mat#.reshape([C_mat.shape[0]* C_mat.shape[1]*C_mat.shape[2], C_mat.shape[3]]) 
    
    near_frequencies_vec_indices = near_frequencies_vec_indices#.reshape(-1)

    return C_mat_veced, near_frequencies_vec_indices


batch_outer = jax.vmap(lambda x: jnp.outer(x,x), in_axes = (0))
batch_batch_outer = jax.vmap(batch_outer, in_axes = (0))

## Mean functions
def compute_discretization_weights(experiment_dataset, prior, batch_size, order=1, grid_dist = None, max_dist = None ):
    order = 1   # Only implemented for order 1 but could generalize
    rr_size = 3*order + 1
    RR = jnp.zeros((experiment_dataset.volume_size, (rr_size)**2 ))
    # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()) * 3

    for i in range(utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)):
        batch_st, batch_end = utils.get_batch_of_indices(experiment_dataset.n_images, batch_size, i)
        # Make sure mean_estimate is size # volume_size ?
        RR_this = compute_weight_matrix_inner(experiment_dataset.rotation_matrices[batch_st:batch_end], experiment_dataset.CTF_params[batch_st:batch_end], experiment_dataset.voxel_size, experiment_dataset.volume_shape, experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, grid_dist, max_dist)        
        RR += RR_this

    RR = RR.reshape([experiment_dataset.volume_size, rr_size, rr_size])
    weights, good_weights = batch_solve_for_weights(RR, prior)
    # If bad weights, just do Weiner filtering with 0th order disc
    # good_weights = (good_weights*0).astype(bool)

    other_weights = jnp.zeros_like(weights)
    weiner_weights = 1 / (RR[...,0,0] + prior)
    other_weights = other_weights.at[...,0].set(weiner_weights)
    weights = weights.at[~good_weights].set(other_weights[~good_weights])
    return weights, good_weights, RR

def solve_for_weights(RR, prior):
    e1 = jnp.zeros(RR.shape[0])
    e1 = e1.at[0].set(RR[0,0] / (RR[0,0] + prior))
    # Maybe could just check for conditioning
    v = jax.scipy.linalg.solve(RR, e1, lower=False, assume_a='pos')
    # Probably should replace with numpy.linalg.eigvalsh
    good_v = jnp.linalg.cond(RR) < 1e2
    # good_v = jnp.min(jnp.diag(RR)) > constants.ROOT_EPSILON
    return v, good_v


batch_solve_for_weights = jax.vmap(solve_for_weights, in_axes = (0,0))

broadcas = jax.vmap(jnp.dot, in_axes = (0,0))
batch_batch_dot = jax.vmap(batch_dot, in_axes = (0,0))


def broadcast_dot(x,y):
    return jax.lax.batch_matmul(jnp.conj(x[...,None,:]),y[...,:,None])[...,0,0]

def broadcast_outer(x,y):
    return jax.lax.batch_matmul(x[...,:,None],jnp.conj(y[...,None,:]))


# def broadcast_dot(x,y):
#     return jnp.dot(x,y)


# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
def compute_mean_least_squares_rhs_lhs_with_weights(images, precomp_weights, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun, grid_dist, max_dist ):

    # Now use weights: w_i = C.^T v
    if grid_dist is None:
        C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
    else:
        C_mat, grid_point_indices = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)

    precomp_weights_on_pixel = core.batch_slice_volume(precomp_weights,grid_point_indices)

    weights = broadcast_dot(C_mat, precomp_weights_on_pixel)

    volume_size = np.prod(np.array(volume_shape))
    # grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    corrected_images = core.translate_images(images, translations, image_shape)    
    # corrected_images = corrected_images * image_weights[...,None] / noise_variance[None]

    estimate = core.summed_adjoint_projections_nearest(volume_size, corrected_images * weights, grid_point_indices)
    lhs = core.summed_adjoint_projections_nearest(volume_size, weights, grid_point_indices)

    return estimate, lhs

# Solves the linear system Dx = b.
def solve_least_squares_mean_iteration_second_order(experiment_dataset , prior, cov_noise,  batch_size, mean_estimate, image_weights = None, disc_type = None, return_lhs_rhs = False, grid_dist = None, max_dist = None ):
    # all_one_volume = jnp.ones(volume_size)
    estimate =0;  lhs = 0 
    # Need to take out RR
    weights, good_weights, RR = compute_discretization_weights(experiment_dataset, prior, batch_size, order=1, grid_dist = grid_dist, max_dist = max_dist )

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for batch, indices in data_generator:
        
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
                    
        estimate_this, lhs_this = compute_mean_least_squares_rhs_lhs_with_weights(batch,
                                         weights,
                                         experiment_dataset.rotation_matrices[indices], 
                                         experiment_dataset.translations[indices], 
                                         experiment_dataset.CTF_params[indices], 
                                         mean_estimate,
                                         cov_noise,
                                         None,
                                         experiment_dataset.voxel_size, 
                                         experiment_dataset.volume_shape, 
                                         experiment_dataset.image_shape, 
                                         experiment_dataset.grid_size, 
                                         disc_type,
                                         experiment_dataset.CTF_fun,
                                         grid_dist,max_dist)
        estimate += estimate_this
        lhs += lhs_this

    return estimate, good_weights, lhs