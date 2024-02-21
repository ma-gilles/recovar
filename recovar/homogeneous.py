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
    means["lhs"] = lhs/2

    end_time = time.time()
    logger.info(f"time to compute means: {end_time- st_time}")

    return means, mean_prior, fsc, lhs


def predict_optimal_h_value(summed_grad_terms ,summed_squared_weights, noise_variance, grad_norm, max_dist=1,order =1 ):
    
    # A random divided by 10 term here...
    normalized_c1 = (grad_norm * summed_grad_terms )**2 / 10
    normalized_c2 =  noise_variance* (summed_squared_weights * max_dist**3)
    h = 1/((2*normalized_c1/ (3* normalized_c2)))**(1/5)
    lower_bound = 0.5
    h = np.where(h > lower_bound, h,  np.ones_like(h)* lower_bound)
    return h

def compute_gradient_norm(x):
    gradients = jnp.gradient(x)
    gradients = jnp.stack(gradients, axis=0)
    grad_norm = jnp.linalg.norm(gradients, axis = (0), ord=2)
    # grad_norm2 = scipy.ndimage.maximum_filter(grad_norm, size =2)
    return grad_norm

batch_compute_gradient_norm = jax.vmap(compute_gradient_norm, in_axes = (0,))
batch_make_radial_image = jax.vmap(utils.make_radial_image, in_axes = (0, None, None))

def compute_with_adaptive_discretization(cryo, mean_lhs, mean_prior, mean_estimate, noise_variance, batch_size, image_weights):

    num_reconstructions = image_weights.shape[0]
    num_recon_by_shape = [num_reconstructions, *cryo.volume_shape]

    # lhs, rhs = solve_least_squares_mean_iteration(cryo , np.inf, 1,  1000, None, image_weights = None, disc_type = 'nearest', return_lhs_rhs = True )
    # x = (rhs[0]/(lhs[0] + 1 /tau_prior))

    bias_fac = (mean_lhs/ (mean_lhs + 1/mean_prior)).real
    grad_norm = batch_compute_gradient_norm(mean_estimate.reshape(num_recon_by_shape)).reshape(num_reconstructions, -1)
    grad_norm_radial_mod = regularization.batch_average_over_shells(grad_norm**1, cryo.volume_shape, 0)/ np.sqrt(regularization.batch_average_over_shells(bias_fac, cryo.volume_shape, 0))
    grad_norm_est = batch_make_radial_image(grad_norm_radial_mod, cryo.volume_shape, True)
    vol, h_adapt = adaptive_discretization(cryo, grad_norm_est, mean_prior/ noise_variance, noise_variance,batch_size, image_weights)
    # vol, h_adapt, prior = adaptive_discretization(cryo, grad_norm_est, mean_prior, noise_variance)
    return vol, h_adapt


def adaptive_discretization(cryo, grad_norm, prior, noise_variance, batch_size_inp, image_weights):
    
    num_reconstructions = image_weights.shape[0]
    summed_squared_weights = {}
    xs = {}
    summed_grad_terms = {}
    order =0
    dist = 1
    print("here BEFORE,", batch_size_inp)
    print("here BEFORE,", utils.report_memory_device())

    memory_to_use = utils.get_gpu_memory_total() -  utils.get_size_in_gb(prior) * 20 
    print("mem to use,", memory_to_use)
    assert memory_to_use > 0, "reduce number of volumes computed at once"
    batch_size_inp = 2 * utils.get_image_batch_size(cryo.grid_size, memory_to_use) * 20

    batch_size = np.ceil(batch_size_inp/ (1*2 + 1)**3 /3 / num_reconstructions).astype(int)
    logger.info(f"batch size in reweighting: {batch_size}")
    print("here 1,batch_size_inp", batch_size_inp)
    print("here 1,batch_size", batch_size)
    utils.report_memory_device()

    # A first pass to compute optimal h
    xs[(order, dist)], _, summed_squared_weights[(order, dist)], _, _, summed_grad_terms[(order, dist)], bias_fac = solve_least_squares_mean_iteration_second_order(
    cryo ,prior * 0, 1e-8,  batch_size, None, image_weights = image_weights, disc_type = None, return_lhs_rhs = False, 
    grid_dist = np.ceil(dist), max_dist=dist*np.ones([num_reconstructions, cryo.volume_size], dtype = cryo.dtype_real), order = order  )
        
    h_adapt = predict_optimal_h_value(summed_grad_terms[(order, dist)] ,summed_squared_weights[(order, dist)], noise_variance, grad_norm, max_dist=1,order =order )
    # regularization.average_over_shells()

    print("weight,", batch_size)

    # The actual computation
    max_grid_dist = 2#3 if cryo.grid_size < 256 else 2

    batch_size = np.ceil(batch_size_inp/ (max_grid_dist*2 + 1)**3 /3/ num_reconstructions).astype(int)
    logger.info(f"batch size in reweighting: {batch_size}")
    print("here 3,", batch_size)

    dist = 'a'
    xs[(order, dist)], _, summed_squared_weights[(order, dist)], _, _ , summed_grad_terms[(order, dist)], _ = solve_least_squares_mean_iteration_second_order(
    cryo ,1/prior, 1e-8,  batch_size, None, image_weights = image_weights, disc_type = None, return_lhs_rhs = False, 
    grid_dist = max_grid_dist, max_dist=h_adapt, order = order  )

    print("here 4 ,", batch_size)
    print("est 4 ")
    utils.report_memory_device()
    return xs[(order, dist)], h_adapt


def estimate_derivative_norm(volume):
    # Compute gradient of volume
    grad = jnp.gradient(volume)
    grad_norm = jnp.linalg.norm(grad, axis = 0)
    return grad


def get_multiple_conformations(cryos, cov_noise, disc_type, batch_size, mean_prior, mean ,image_weights, recompute_prior = True, volume_mask = None, adaptive = True ):
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
    # lhs_l[0] += lhs_l[1]
    # rhs_l[0] += rhs_l[1]
    reconstructions = np.array( np.where( np.abs(lhs_l[0] + lhs_l[1]) < constants.ROOT_EPSILON, 0, (rhs_l[0] + rhs_l[1])/ ((lhs_l[0] + lhs_l[1]) + 1 / priors ) ))
    end_time = time.time()
    logger.info(f"time to compute reweighted conformations: {end_time- st_time}")
    
    if adaptive:
        adaptive_reconstructions_halfmaps = np.zeros([2 , *reconstructions.shape], dtype = cryo.dtype)
        # Do one by one for now...
        # Initial contrasts
        # initial_contrasts = [ cryo.CTF_params[:,8]  for cryo in cryos]

        # for k in range(num_reconstructions):
        #     for cryo_idx, cryo in enumerate(cryos):
        #         # Instead of passing weights... this should do the same.
        #         cryo.CTF_params[:,8] = initial_contrasts[cryo_idx] * image_weights[cryo_idx][k]
        #         adaptive_reconstructions_halfmaps[cryo_idx,k],_ = compute_with_adaptive_discretization(cryo, lhs_l[cryo_idx][k], priors[k], reconstructions[k], cov_noise, batch_size)

        for cryo_idx, cryo in enumerate(cryos):
            adaptive_reconstructions_halfmaps[cryo_idx],_ = compute_with_adaptive_discretization(cryo, lhs_l[cryo_idx], priors, reconstructions, cov_noise, batch_size, image_weights[cryo_idx])

        reconstructions = np.mean(adaptive_reconstructions_halfmaps, axis=0)

        return reconstructions, [adaptive_reconstructions_halfmaps[0], adaptive_reconstructions_halfmaps[1]]

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
    logger.warning('TOOK OUT IMAGE MASK IN MEAN!!! PUT IT BACK??')

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for batch, indices in data_generator:
        
        # Only place where image mask is used ?
        # print('TOOK OUT IMAGE MASK IN MEAN!!! PUT IT BACK??')
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
        grad_correction = (core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) \
                        - core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, grid_size, "nearest")) * CTF
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
#     RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer, grid_point_vec_indices)
#     return RR

@functools.partial(jax.jit, static_argnums = [3,4,5,6,7])    
def compute_weight_matrix_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist, image_weights ):
    volume_size = np.prod(np.array(volume_shape))
    # C_mat, grid_point_vec_indices = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)
    # This is going to be stroed twice as much stuff as it needs to be
    if grid_dist is None:
        C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
    else:
        C_mat, grid_point_indices, valid_points = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)

    C_mat_outer = broadcast_outer(C_mat, C_mat)


    # RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer.reshape(-1,C_mat_outer.shape[-2]*C_mat_outer.shape[-1] ), grid_point_indices.reshape(-1))
    # return RR
    return batch_compute_weight_matrix_inner_last_step(C_mat_outer, grid_point_indices, image_weights, valid_points, volume_size)

@functools.partial(jax.jit, static_argnums = [4])    
def compute_weight_matrix_inner_last_step( C_mat_outer, grid_point_indices, image_weights, valid_points, volume_size ):
    C_mat_outer = multiply_along_axis(C_mat_outer,image_weights, 0 ) * valid_points[...,None,None]

    # C_mat_outer *= image_weights
    RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer.reshape(-1,C_mat_outer.shape[-2]*C_mat_outer.shape[-1] ), grid_point_indices.reshape(-1))
    return RR

batch_compute_weight_matrix_inner_last_step = jax.vmap(compute_weight_matrix_inner_last_step, in_axes = (None,None,0, 0, None))



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


def make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist, order=2):
    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
    differences =  grid_points_coords[...,None,:] - near_frequencies
    distances = jnp.linalg.norm(differences, axis = -1)
    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

    # if a
    max_dist_this = max_dist[...,near_frequencies_vec_indices.reshape(-1)].reshape([max_dist.shape[0], *near_frequencies_vec_indices.shape])
    # print(jnp.std(max_dist_this),jnp.mean(max_dist_this) )
    # import pdb; pdb.set_trace()
    valid_points = (distances < max_dist_this) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
    
    # This could be done more efficiently
    if order==2:
        C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
        CTF = CTF_fun( CTF_params, image_shape, voxel_size)
        C_mat *= CTF[...,None,None]
        C_mat = C_mat #* valid_points[...,None]
    else:
        CTF = CTF_fun( CTF_params, image_shape, voxel_size)
        C_mat = CTF[...,None]
        C_mat = C_mat #* valid_points[...]

    # C_mat_veced = C_mat#.reshape([C_mat.shape[0]* C_mat.shape[1]*C_mat.shape[2], C_mat.shape[3]]) 
    
    near_frequencies_vec_indices = near_frequencies_vec_indices#.reshape(-1)

    return C_mat, near_frequencies_vec_indices, valid_points


# def make_C_mat_many_gridpoints_first_order(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist):
#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
#     differences =  grid_points_coords[...,None,:] - near_frequencies
#     distances = jnp.linalg.norm(differences, axis = -1)
#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

#     # if a
#     max_dist_this = max_dist[near_frequencies_vec_indices.reshape(-1)].reshape(near_frequencies_vec_indices.shape)
#     # print(jnp.std(max_dist_this),jnp.mean(max_dist_this) )
#     # import pdb; pdb.set_trace()
#     valid_points = (distances < max_dist_this) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
    
#     # This could be done more efficiently

#     C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     C_mat *= CTF[...,None,None]
#     C_mat = C_mat * valid_points[...,None]

#     # C_mat_veced = C_mat#.reshape([C_mat.shape[0]* C_mat.shape[1]*C_mat.shape[2], C_mat.shape[3]]) 
    
#     near_frequencies_vec_indices = near_frequencies_vec_indices#.reshape(-1)

#     return C_mat, near_frequencies_vec_indices


batch_outer = jax.vmap(lambda x: jnp.outer(x,x), in_axes = (0))
batch_batch_outer = jax.vmap(batch_outer, in_axes = (0))

## Mean functions
def compute_discretization_weights(experiment_dataset, prior, batch_size, order=1, grid_dist = None, max_dist = None, image_weights = None ):

    order_t = 1   # Only implemented for order 1 but could generalize
    rr_size = 3*order_t + 1
    RR = jnp.zeros((experiment_dataset.volume_size, (rr_size)**2 ), dtype = experiment_dataset.dtype_real)
    # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()) * 3

    for i in range(utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)):
        batch_st, batch_end = utils.get_batch_of_indices(experiment_dataset.n_images, batch_size, i)
        # Make sure mean_estimate is size # volume_size ?
        RR_this = compute_weight_matrix_inner(experiment_dataset.rotation_matrices[batch_st:batch_end], experiment_dataset.CTF_params[batch_st:batch_end], experiment_dataset.voxel_size, experiment_dataset.volume_shape, experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, grid_dist, max_dist, image_weights[...,batch_st:batch_end])        
        RR += RR_this

    RR = RR.reshape([RR_this.shape[0], RR_this.shape[1], rr_size, rr_size])
    # weights, good_weights = batch_solve_for_weights(RR, prior)
    weights = jnp.zeros_like(RR, shape = RR.shape[0:-1])
    # weights, good_weights = batch_solve_for_weights(RR, prior)
    # If bad weights, just do Weiner filtering with 0th order disc
    good_weights = None#(good_weights*0).astype(bool)
    bias_multiple = RR[...,0,0] /(RR[...,0,0] + prior)
    other_weights = jnp.zeros_like(weights)
    # weights = jnp.zeros_like(prior)

    weiner_weights = 1 / (RR[...,0,0] + prior)
    other_weights = other_weights.at[...,0].set(weiner_weights)
    if order ==0:
        weights = other_weights
        good_weights = None#(good_weights*0).astype(bool)
    else:
        weights, good_weights = batch_solve_for_weights(RR, prior)
        weights = weights.at[~good_weights].set(other_weights[~good_weights])

    ### THROW AWAY RR
    RR = None
    return weights, good_weights, RR, bias_multiple

@jax.jit
def solve_for_weights(RR, prior):
    e1 = jnp.zeros(RR.shape[0])
    e1 = e1.at[0].set(RR[0,0] / (RR[0,0] + prior))
    # Maybe could just check for conditioning
    v = jax.scipy.linalg.solve(RR, e1, lower=False, assume_a='pos')
    # Probably should replace with numpy.linalg.eigvalsh
    good_v = jnp.linalg.cond(RR) < 1e4
    # good_v = jnp.min(jnp.diag(RR)) > constants.ROOT_EPSILON
    return v, good_v

batch_solve_for_weights = jax.vmap(solve_for_weights, in_axes = (0,0))

def broadcast_dot(x,y):
    return jax.lax.batch_matmul(jnp.conj(x[...,None,:]),y[...,:,None])[...,0,0]

def broadcast_outer(x,y):
    return jax.lax.batch_matmul(x[...,:,None],jnp.conj(y[...,None,:]))


batch_batch_slice_volume_by_nearest = jax.vmap(core.batch_slice_volume_by_nearest, in_axes = (0, None))

bcast0_broadcast_dot = jax.vmap(broadcast_dot, in_axes = (None,0))
@functools.partial(jax.jit, static_argnums = [8,9,10,11,12,13,14])    
def compute_mean_least_squares_rhs_lhs_with_weights(images, precomp_weights, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun, grid_dist, max_dist ):


    corrected_images = core.translate_images(images, translations, image_shape)    

    # Now use weights: w_i = C.^T v
    if grid_dist is None:
        C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
    else:
        C_mat, grid_point_indices, valid_points = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)
        corrected_images = corrected_images[...,None]

    precomp_weights_on_pixel = batch_batch_slice_volume_by_nearest(precomp_weights,grid_point_indices)

    # precomp_weights_on_pixel = core.batch_slice_volume_by_nearest(precomp_weights,grid_point_indices)
    # weights = broadcast_dot(C_mat[None], precomp_weights_on_pixel)
    weights = bcast0_broadcast_dot(C_mat, precomp_weights_on_pixel)

    residuals = weights[...,None] * C_mat[None]
    C_squared = jnp.abs(C_mat[...,0])**2

    volume_size = np.prod((volume_shape))


    # res_summed = jnp.linalg.norm(residuals[...,1:], axis=-1).reshape(-1)
    # residuals_summed2 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))

    # res_summed = jnp.abs(residuals[...,0])

    # residuals_summed1 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))
    # C_squared_summed = core.summed_adjoint_slice_by_nearest(volume_size, C_squared.reshape(-1)**1, grid_point_indices.reshape(-1))

    # estimate = core.summed_adjoint_slice_by_nearest(volume_size, corrected_images * weights, grid_point_indices)
    # summed_weights_squared = core.summed_adjoint_slice_by_nearest(volume_size, weights**2, grid_point_indices)

    # return estimate, summed_weights_squared, residuals_summed1, residuals_summed2, C_squared_summed
    # return compute_mean_least_squares_rhs_lhs_with_weights(image_weights, weights, residuals, C_squared, grid_point_indices, corrected_images, volume_size )
    return batch_compute_mean_least_squares_rhs_lhs_with_weights_last_step(image_weights, weights, valid_points, residuals, C_squared, grid_point_indices, corrected_images, volume_size )

def multiply_along_axis(A, B, axis):
    return jnp.swapaxes(jnp.swapaxes(A, axis, -1) * B, -1, axis)


@functools.partial(jax.jit, static_argnums = [7])    
def compute_mean_least_squares_rhs_lhs_with_weights_last_step(image_weights, weights, valid_points, residuals, C_squared, grid_point_indices, corrected_images, volume_size ):
    
    weights = multiply_along_axis(weights,image_weights, 0 ) * valid_points
    residuals = multiply_along_axis(residuals,image_weights, 0 ) * valid_points[...,None]
    C_squared = multiply_along_axis(C_squared,image_weights, 0 ) * valid_points
    corrected_images = multiply_along_axis(corrected_images,image_weights, 0 ) * valid_points

    # weights *= image_weights
    # residuals *= image_weights
    # C_squared *= image_weights
    # corrected_images *= image_weights


    res_summed = jnp.linalg.norm(residuals[...,1:], axis=-1).reshape(-1)
    residuals_summed2 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))

    res_summed = jnp.abs(residuals[...,0]) 

    residuals_summed1 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))

    C_squared_summed = core.summed_adjoint_slice_by_nearest(volume_size, C_squared.reshape(-1)**1, grid_point_indices.reshape(-1))

    estimate = core.summed_adjoint_slice_by_nearest(volume_size, corrected_images * weights, grid_point_indices)
    summed_weights_squared = core.summed_adjoint_slice_by_nearest(volume_size, weights**2, grid_point_indices)
    return estimate, summed_weights_squared, residuals_summed1, residuals_summed2, C_squared_summed

batch_compute_mean_least_squares_rhs_lhs_with_weights_last_step = jax.vmap(compute_mean_least_squares_rhs_lhs_with_weights_last_step, in_axes = (0, 0, 0, 0, None, None, None, None))





# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
# def compute_mean_least_squares_rhs_lhs_nn_adaptive(images, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):

#     ### UNFINISHED !!!!

    
#     volume_size = np.prod(np.array(volume_shape))
#     grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )

#     ## 
#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
#     differences =  grid_points_coords[...,None,:] - near_frequencies
#     distances = jnp.linalg.norm(differences, axis = -1)
#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

#     # if a
#     max_dist_this = max_dist[near_frequencies_vec_indices.reshape(-1)].reshape(near_frequencies_vec_indices.shape)
#     # print(jnp.std(max_dist_this),jnp.mean(max_dist_this) )
#     # import pdb; pdb.set_trace()
#     valid_points = (distances < max_dist_this) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
    
#     # This could be done more efficiently

#     C_mat = 1#jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     C_mat *= CTF[...,None]
#     C_mat = C_mat * valid_points[...,None]


#     translated_images = core.translate_images(images, translations, image_shape)
#     corrected_images = translated_images
    
#     all_one_volume = jnp.ones(volume_size, dtype = images.dtype)    
#     ones_mapped = core.forward_model(all_one_volume, CTF, grid_point_indices) / noise_variance[None] * image_weights[...,None]
#     # diag_mean = core.sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)
#     diag_mean = batch_over_weights_sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)

#     corrected_images = corrected_images * image_weights[...,None] / noise_variance[None]
#     # mean_rhs = core.sum_adj_forward_model(volume_size, corrected_images  / noise_variance[None] , CTF, grid_point_indices)
#     mean_rhs = batch_over_weights_sum_adj_forward_model(volume_size, corrected_images , CTF, grid_point_indices)
#     return mean_rhs, diag_mean


# def compute_residuals():


# Solves the linear system Dx = b.
def solve_least_squares_mean_iteration_second_order(experiment_dataset , prior, cov_noise,  batch_size, mean_estimate, image_weights = None, disc_type = None, return_lhs_rhs = False, grid_dist = None, max_dist = None, order=1 ):
    # all_one_volume = jnp.ones(volume_size)
    estimate =0;  summed_weights_squared = 0 ; residuals_summed1 = 0; residuals_summed2 = 0; residuals_summed3 = 0
    # Need to take out RR
    logger.info(f"batch size in second order: {batch_size}")

    weights, good_weights, RR, bias_multiple = compute_discretization_weights(experiment_dataset, prior, batch_size, order=order, grid_dist = grid_dist, max_dist = max_dist, image_weights=image_weights )
    logger.info(f"time done with weights")
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    for batch, indices in data_generator:
        
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
                    
        estimate_this, summed_weights_squared_this, residuals_summed_this1, residuals_summed_this2, residuals_summed_this3  = compute_mean_least_squares_rhs_lhs_with_weights(batch,
                                         weights,
                                         experiment_dataset.rotation_matrices[indices], 
                                         experiment_dataset.translations[indices], 
                                         experiment_dataset.CTF_params[indices], 
                                         mean_estimate,
                                         cov_noise,
                                         image_weights[...,indices],
                                         experiment_dataset.voxel_size, 
                                         experiment_dataset.volume_shape, 
                                         experiment_dataset.image_shape, 
                                         experiment_dataset.grid_size, 
                                         disc_type,
                                         experiment_dataset.CTF_fun,
                                         grid_dist,max_dist)
        
        estimate += estimate_this
        summed_weights_squared += summed_weights_squared_this
        residuals_summed1 += residuals_summed_this1
        residuals_summed2 += residuals_summed_this2
        residuals_summed3 += residuals_summed_this3
    logger.info(f"time done with estimate")

    # residuals_summed3 = jnp.linalg.norm(residuals_summed3[:,1:], axis=-1)

    return estimate, good_weights, summed_weights_squared, weights, residuals_summed1, residuals_summed2, bias_multiple