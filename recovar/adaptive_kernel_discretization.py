import logging
import jax.numpy as jnp
import numpy as np
import jax, functools, time

from recovar import core, regularization, constants, noise, linalg
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils

logger = logging.getLogger(__name__)

## Low level functions
def make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = 0, dtype = np.float32):

    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    if pol_degree ==0:
        return jnp.ones(grid_point_vec_indices.shape, dtype = dtype )[...,None], grid_point_vec_indices

    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size ).astype(dtype)
    # Discretized grid points
    # This could be done more efficiently
    grid_points_coords_nearest = core.round_to_int(grid_points_coords)
    differences = grid_points_coords - grid_points_coords_nearest

    if pol_degree==1:
        X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences**2], axis = -1)
        return X_mat, grid_point_vec_indices

    differences_squared = linalg.broadcast_outer(differences, differences)
    differences_squared = keep_upper_triangular(differences_squared)
    X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_squared], axis = -1)

    return X_mat, grid_point_vec_indices


## Handling triangular to full matrix representation

def keep_upper_triangular(XWX):
    iu1 = np.triu_indices(XWX.shape[-1])
    return XWX[...,iu1[0], iu1[1]]

def undo_keep_upper_triangular_one(XWX):
    # m = n(n+1)/2
    # n = (sqrt(8m+1) -1)/2
    m = XWX.shape[-1]
    n = np.round((np.sqrt(8*m+1) -1)/2).astype(int)

    triu_indices = jnp.triu_indices(n)
    matrix = jnp.empty((n,n), dtype = XWX.dtype)
    matrix = matrix.at[triu_indices[0], triu_indices[1]].set(XWX)

    i_lower = jnp.tril_indices(n, -1)
    matrix = matrix.at[i_lower[0], i_lower[1]].set(matrix.T[i_lower[0], i_lower[1]])
    return matrix

def find_smaller_pol_indices(max_pol_degree, target_pol_degree):
    max_num_pol_params = get_feature_size(max_pol_degree)
    max_triu_indices = np.triu_indices(max_num_pol_params)
    target_num_pol_params = get_feature_size(target_pol_degree)
    target_triu_indices = np.triu_indices(target_num_pol_params)

    target_triu_indices = np.array(target_triu_indices).T
    max_triu_indices = np.array(max_triu_indices).T

    indices = np.zeros(target_triu_indices.shape[0], dtype = int)
    for k , searchval in enumerate(target_triu_indices):
        indices[k] = np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0]

    return indices

def find_diagonal_pol_indices(max_pol_degree):
    max_num_pol_params = get_feature_size(max_pol_degree)
    max_triu_indices = np.triu_indices(max_num_pol_params)
    max_triu_indices = np.array(max_triu_indices).T
    diagonal_indices = np.concatenate([np.arange(max_num_pol_params)[...,None],np.arange(max_num_pol_params)[...,None]], axis=-1)
    indices = np.zeros(diagonal_indices.shape[0], dtype = int)
    for k, searchval in enumerate(diagonal_indices):
        indices[k] = np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0]
    return indices

undo_keep_upper_triangular = jax.vmap(undo_keep_upper_triangular_one, in_axes = (0), out_axes = 0)


def volume_shape_to_half_volume_shape(volume_shape):
    return (volume_shape[0]//2 + 1, volume_shape[1], volume_shape[2] )

def half_volume_shape_to_volume_shape(volume_shape): 
    volume_shape[0] = volume_shape[0] * 2
    return ((volume_shape[0]-1)*2, *volume_shape[1:])

## NOTE There is something weird and jitting this. Compilation takes a very long time. Check it out.
@functools.partial(jax.jit, static_argnums = [1,2])
def vec_index_to_half_vec_index(indices, volume_shape, flip_positive = False):
    vol_indices_full = core.vec_indices_to_vol_indices(indices, volume_shape)

    grid_size = volume_shape[0]
    negative_frequencies = vol_indices_full[...,0] < (grid_size // 2 + 1) 
    if flip_positive:
        frequencies = core.vol_indices_to_frequencies(vol_indices_full, volume_shape)
        # flipped_first = jnp.where(frequencies, -frequencies , frequencies)
        frequencies_flipped = jnp.where(frequencies[...,0:1] > 0, -frequencies , frequencies)
        vol_indices_full_flipped = core.frequencies_to_vol_indices(frequencies_flipped, volume_shape)
        vol_indices_full = vol_indices_full_flipped

    in_bound = core.check_vol_indices_in_bound(vol_indices_full, grid_size)

    vec_indices = core.vol_indices_to_vec_indices(vol_indices_full, volume_shape)
    vec_indices = jnp.where(in_bound, vec_indices, -1*jnp.ones_like(vec_indices))
    return vec_indices, negative_frequencies

@functools.partial(jax.jit, static_argnums = [1])
def half_volume_to_full_volume(half_volume, volume_shape):
    volume_size = np.prod(volume_shape)
    indices = jnp.arange(volume_size) # will this JIT this whole thing? ugh
    half_indices, negative_frequencies = vec_index_to_half_vec_index(indices, volume_shape, flip_positive = True)
    volume = jnp.zeros(volume_size, dtype = half_volume.dtype)
    volume = half_volume[half_indices]
    volume = jnp.where(negative_frequencies, volume, jnp.conj(volume))
    return volume

batch_half_volume_to_full_volume = jax.vmap(half_volume_to_full_volume, in_axes = (0,None), out_axes = 0)

def full_volume_to_half_volume(volume, volume_shape):
    half_volume_shape = volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = np.prod(half_volume_shape)
    return volume[:half_volume_size]

def half_vec_index_to_vec_index(indices_half, volume_shape):
    # For indices with negative frequencies, return -1 (out of bound, not used)
    vol_indices_half = core.vec_indices_to_vol_indices(indices_half, volume_shape_to_half_volume_shape(volume_shape))
    indices = core.vol_indices_to_vec_indices(vol_indices_half, volume_shape)
    bad_indices = indices_half == -1
    indices = indices.at[bad_indices].set(-1)
    return indices

## Precompute functions

@functools.partial(jax.jit, static_argnums = [5,6,7,8, 10])    
def precompute_kernel_one_batch(images, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, noise_variance, pol_degree =0, XWX = None, F = None):

    # Precomp piece
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    volume_size = np.prod(volume_shape)
    ctf_over_noise_variance = CTF**2 / noise_variance

    X, grid_point_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = pol_degree)
    grid_point_indices, good_idx = vec_index_to_half_vec_index(grid_point_indices, volume_shape, flip_positive = True)
    X = X * good_idx[...,None]

    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))

    # XWX
    XWX_b = linalg.broadcast_outer(X * ctf_over_noise_variance[...,None] , X)
    XWX_b = keep_upper_triangular(XWX_b)
    XWX = core.batch_over_vol_summed_adjoint_slice_by_nearest(
        half_volume_size, XWX_b,
        grid_point_indices.reshape(-1), XWX)
    
    # F
    images = core.translate_images(images, translations, image_shape) 
    # In my formalism, images === images / CTF soo I guess this is right
    F_b = X * (images * CTF / noise_variance)[...,None] 

    F = core.batch_over_vol_summed_adjoint_slice_by_nearest(half_volume_size, F_b, grid_point_indices.reshape(-1),F)
    return XWX, F

def get_differences_zero(pol_degree, differences):
    if pol_degree ==0:
        differences_zero = jnp.zeros_like(differences[...,0:1])
    elif pol_degree==1:
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences**2], axis = -1)
    elif pol_degree==2:
        differences_squared = linalg.broadcast_outer(differences, differences)
        differences_squared = keep_upper_triangular(differences_squared)
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences, differences_squared], axis = -1)
    else:
        assert(NotImplementedError)
    return differences_zero


# Should this all just be vmapped instead of vmapping each piece? Not really sure.
# It allocate XWX a bunch of time if I do?
@functools.partial(jax.jit, static_argnums = [2,5,6,8])
def compute_estimate_from_precompute_one(XWX, F, max_grid_dist, grid_distances, frequencies, volume_shape, pol_degree, signal_variance, use_regularization):

    if use_regularization:
        regularization = 1 / signal_variance
    else:
        dim = pol_degree + 1
        regularization = jnp.zeros((frequencies.shape[0], dim), dtype = XWX.dtype)

    
    # if max_grid_dist == -1:
    #     print("here")
    #     frequencies_vec_indices = core.vol_indices_to_vec_indices(frequencies.astype(int), volume_shape)
    #     # frequencies_vec_indices, negative_frequencies = vec_index_to_half_vec_index(frequencies_vec_indices, volume_shape, flip_positive = True)

    #     XWX_undo = undo_keep_upper_triangular(XWX[frequencies_vec_indices])

    #     if use_regularization:
    #         regularization_expanded = make_regularization_from_reduced(regularization)
    #         XWX_undo += jnp.diag(regularization_expanded)

    #     f = F[frequencies_vec_indices]
    #     vreal = jnp.linalg.solve(XWX_undo, f.real)#, lower=False, assume_a='pos')
    #     vimag = jnp.linalg.solve(XWX_undo, f.imag)#, lower=False, assume_a='pos')

    #     y_and_deriv = vreal + 1j * vimag

    #     # y_and_deriv, good_v, problems = batch_solve_for_m(XWX_undo, F[frequencies_vec_indices], regularization )
    #     # y_and_deriv = jnp.where(negative_frequencies[...,None], y_and_deriv, jnp.conj(y_and_deriv))
    #     good_v = jnp.ones_like(y_and_deriv[...,0], dtype = bool)
    #     problems = jnp.zeros_like(y_and_deriv[...,0], dtype = bool)

    #     return y_and_deriv, good_v, problems


    # Might have to cast this back to frequencies vs indices frequencies
    near_frequencies = core.find_frequencies_within_grid_dist(frequencies, max_grid_dist)
    differences =  near_frequencies - frequencies[...,None,:]
    # This is just storing the same array many times over...
    differences_zero = get_differences_zero(pol_degree, differences)
    # L_inf norm distances
    distances = jnp.max(jnp.abs(differences), axis = -1)
    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

    ## TODO SHOULD I PUT THIS BACK?
    # valid_points = (distances <= (grid_distances[...,None] + 0.5) )* core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])
    valid_points = core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])


    near_frequencies_vec_indices, negative_frequencies = vec_index_to_half_vec_index(near_frequencies_vec_indices, volume_shape, flip_positive = True)

    XWX_summed_neighbor = batch_summed_over_indices(XWX, near_frequencies_vec_indices, valid_points)
    XWX_summed_neighbor = undo_keep_upper_triangular(XWX_summed_neighbor)

    Z = XWX[...,:differences_zero.shape[-1]]
    Z_summed_neighbor = batch_Z_grab(Z, near_frequencies_vec_indices, valid_points, differences_zero)

    # Rank 3 update. Z is real so no need for np.conj
    XWX_summed_neighbor += Z_summed_neighbor + jnp.swapaxes(Z_summed_neighbor, -1, -2 )

    alpha = XWX[...,0]
    XWX_summed_neighbor += batch_summed_scaled_outer(alpha, near_frequencies_vec_indices, differences_zero, valid_points)

    # F terms involve conjugates so need to be careful
    F_summed_neighbor = batch_summed_over_indices(F, near_frequencies_vec_indices, valid_points * negative_frequencies)
    F_summed_neighbor_conj = batch_summed_over_indices(jnp.conj(F), near_frequencies_vec_indices, valid_points * ~negative_frequencies)
    F_summed_neighbor += F_summed_neighbor_conj

    F0_summed_neighbor = batch_Z_grab(F[...,0:1], near_frequencies_vec_indices, valid_points * negative_frequencies, differences_zero)[...,0,:]
    F0_summed_neighbor_conj = batch_Z_grab(jnp.conj(F[...,0:1]), near_frequencies_vec_indices, valid_points * ~negative_frequencies, differences_zero)[...,0,:]
    F_summed_neighbor += F0_summed_neighbor + F0_summed_neighbor_conj


    y_and_deriv, good_v, problems = batch_solve_for_m(XWX_summed_neighbor,F_summed_neighbor, regularization )

    return y_and_deriv, good_v, problems


def summed_scaled_outer(alpha, indices, differences_zero, valid_points):
    alpha_slices = alpha[indices] * valid_points
    return jnp.sum(linalg.multiply_along_axis(linalg.broadcast_outer(differences_zero, differences_zero), alpha_slices, 0 ), axis=0)

batch_summed_scaled_outer = jax.vmap(summed_scaled_outer, in_axes = (None,0,0, 0))

# Idk why I can't find a nice syntax to do this.
slice_first_axis = jax.vmap(lambda vec, indices: vec[indices], in_axes = (-1,None), out_axes=(-1))

def summed_over_indices(vec, indices, valid):
    sliced = slice_first_axis(vec,indices)
    return jnp.sum(sliced * valid[...,None], axis = -2)

batch_summed_over_indices = jax.vmap(summed_over_indices, in_axes = (None,0,0))

# It feels very silly to have to do this. But I guess JAX will clean up?
def Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
    sliced = slice_first_axis(Z,near_frequencies_vec_indices)
    return  (sliced * valid_points[...,None]).T @ differences

batch_Z_grab = jax.vmap(Z_grab, in_axes = (None,0,0,0))

def one_Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
    sliced = Z[near_frequencies_vec_indices]#slice_first_axis(Z,near_frequencies_vec_indices)
    return  (sliced * valid_points) * differences

batch_one_Z_grab = jax.vmap(one_Z_grab, in_axes = (None,0,0,0))

@jax.jit
def cond_from_flat_one(XWX):
    XWX = undo_keep_upper_triangular_one(XWX)
    return jnp.linalg.cond(XWX)

cond_from_flat = jax.vmap(cond_from_flat_one, in_axes = (0))


@jax.jit
def solve_for_m(XWX, f, regularization):

    regularization_expanded = make_regularization_from_reduced(regularization)
    XWX += jnp.diag(regularization_expanded)
    # XWX = XWX.at[0,0].add(regularization)
    
    dtype_to_solve = np.float64
    vreal = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.real.astype(dtype_to_solve), lower=False, assume_a='pos')
    vimag = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.imag.astype(dtype_to_solve), lower=False, assume_a='pos')
    # vreal = jnp.linalg.solve(XWX, f.real)#, lower=False, assume_a='pos')
    # vimag = jnp.linalg.solve(XWX, f.imag)#, lower=False, assume_a='pos')

    v = vreal + 1j * vimag
    good_v = jnp.linalg.cond(XWX) < 1e4

    e1 = jnp.zeros_like(v)
    e1 = e1.at[0].set(f[0]/XWX[0,0])

    v = jnp.where(good_v, v, e1)
    v = jnp.where(XWX[0,0] > 0, v, jnp.zeros_like(v))
    v = v.astype(np.complex64) ## 128!!?!? CHANGE MAYBE??
    # v = v.astype(np.complex128) ## 128!!?!? CHANGE MAYBE??

    # If condition number is bad, do degree 0 approx?
    # v = jnp.where(good_v, v, jnp.zeros_like(v))
    # res = jnp.linalg.norm(XWX @v - f)**1 / jnp.linalg.norm(f)**1#, axis = -1)

    bad_indices = jnp.isnan(v).any(axis=-1) + jnp.isinf(v).any(axis=-1)
    problem = bad_indices * good_v

    # problem = (res > 1e-6) * good_v
    v = jnp.where(bad_indices, jnp.zeros_like(v), v)

    return v, good_v, problem

batch_solve_for_m = jax.vmap(solve_for_m, in_axes = (0,0,0))


# Should this be set by cross validation?

def precompute_kernel(experiment_dataset, cov_noise, pol_degree=0):    
    XWX, F = 0,0
    # print(utils.report_memory_device())
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(experiment_dataset.volume_shape))

    XWX = jnp.zeros((half_volume_size, small_gram_matrix_size(pol_degree) ), dtype = np.float32)
    F = jnp.zeros((half_volume_size, get_feature_size(pol_degree) ), dtype = np.complex64)

    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - 2* utils.get_size_in_gb(XWX) - 2*utils.get_size_in_gb(F)  ) )

    # Need to take out RR
    logger.info(f"batch size in precompute kernel: {batch_size}")
    # batch_size = 1
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    cov_noise_image = noise.make_radial_noise(cov_noise, experiment_dataset.image_shape)

    idx = 0 
    for batch, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        XWX, F = precompute_kernel_one_batch(batch,
                                experiment_dataset.rotation_matrices[indices], 
                                experiment_dataset.translations[indices], 
                                experiment_dataset.CTF_params[indices], 
                                experiment_dataset.voxel_size, 
                                experiment_dataset.volume_shape, 
                                experiment_dataset.image_shape, 
                                experiment_dataset.grid_size, 
                                experiment_dataset.CTF_fun,
                                cov_noise_image, pol_degree = pol_degree, XWX = XWX, F = F)
        idx+=1

    logger.info(f"Done with precompute of kernel")
    return np.asarray(XWX), np.asarray(F)

# Should pass a list of triplets (pol_degree : int, h : float, regularization : bool)

def get_default_discretization_params(grid_size):
    params = []
    pol_degrees = [0,1] if grid_size <= 512 else [0,]

    for pol_degree in pol_degrees:
        for h in [0,1]:
            params.append((pol_degree, h, True))
    params.append((0,1,False))
    params.append((0,2,True))

    return params
            
def small_gram_matrix_size(pol_degree):
    feature_size = get_feature_size(pol_degree)
    return (feature_size * (feature_size+1))//2

def big_gram_matrix_size(pol_degree):
    return get_feature_size(pol_degree)**2

def get_feature_size(pol_degree):
    if pol_degree==0:
        return 1
    if pol_degree==1:
        return 4
    if pol_degree==2:
        return 10
    return pol_degree


def estimate_signal_variance(experiment_datasets, cov_noise, discretization_params, signal_variance = None, return_all = False):
    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         cov_noise.astype(np.float32), pol_degree=max_pol_degree)
    
    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
    if cov_noise is None:
        cov_noise, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    
    signal_variance = jnp.ones((experiment_datasets[0].volume_size), dtype = np.float32) * signal_var
    # First estimate signal variance with h = 0, p =0 
    first_estimates = [None,None]
    for k in range(2):
        first_estimates[k] = np.array( np.where(np.abs(XWXs[k][...,0]) < constants.ROOT_EPSILON, 0, Fs[k][...,0] / (XWXs[k][...,0] + 1/ signal_var ) ))
        first_estimates[k] = half_volume_to_full_volume(first_estimates[k], experiment_datasets[k].volume_shape)


    lhs = (XWXs[0][...,0] + XWXs[1][...,0]) /2
    lhs = half_volume_to_full_volume(lhs, experiment_datasets[0].volume_shape)
    signal_variance, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(experiment_datasets[0].volume_shape, first_estimates[0], first_estimates[1], lhs , signal_variance, frequency_shift = jnp.array([0,0,0]))

    # Set all regularization params to signal_variance
    signal_variance = np.repeat(signal_variance[...,None], max_pol_degree+1, axis=-1)

    # Polynomial estimates
    pol_estimates = [None,None]
    for k in range(2):
        pol_estimates[k], valid_weights_this = compute_weights_from_precompute(experiment_datasets[k], XWXs[k], Fs[k], signal_variance, max_pol_degree, max_pol_degree, h = 0)

    diag_indices = find_diagonal_pol_indices(max_pol_degree)
    degrees = get_degree_of_each_term(max_pol_degree)
    diagonal_M_terms = (XWXs[0][...,diag_indices] + XWXs[1][...,diag_indices]) /2
    diagonal_M_terms = batch_half_volume_to_full_volume(diagonal_M_terms.T, experiment_datasets[0].volume_shape)

    # Compute the variance of that term
    num_params = Fs[0].shape[-1]
    signal_variance_final = np.zeros((experiment_datasets[0].volume_size, num_params), dtype = np.float32)
    for i in range(Fs[0].shape[-1]):
        # signal_variance_final[:,i], _ , _  = regularization.compute_fsc_prior_gpu_v2(experiment_datasets[0].volume_shape, pol_estimates[0][...,i], pol_estimates[1][...,i], diagonal_M_terms[i] , signal_variance[:,degrees[i]], frequency_shift = jnp.array([0,0,0]))
        signal_variance_final[:,i] = estimate_signal_variance_from_correlation(pol_estimates[0][...,i], pol_estimates[1][...,i], diagonal_M_terms[i], signal_variance[:,degrees[i]], experiment_datasets[0].volume_shape)

    return signal_variance_final, signal_variance

def estimate_signal_variance_from_correlation(vol1, vol2, lhs, prior, volume_shape):
    correlation = jnp.conj(vol1) * vol2
    correlation_avg = regularization.average_over_shells(correlation.real, volume_shape, frequency_shift = np.array([0,0,0]))

    top = lhs**2 / (lhs + 1/prior)**2
    sum_top = regularization.average_over_shells(top,  volume_shape, frequency_shift = np.array([0,0,0]))
    prior_avg = jnp.where( sum_top > 0 , correlation_avg / sum_top , constants.ROOT_EPSILON )
    # prior_avg = jnp.where( sum_top > 0 , correlation_avg  , constants.ROOT_EPSILON )

    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = np.array([0,0,0])).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]

    return prior


def get_degree_of_each_term(max_pol_degree):
    # signal variance, deriv_variance, hessian_variance
    if max_pol_degree==0:
        return np.array([0])
    if max_pol_degree==1:
        return np.array([0,1,1,1])
    if max_pol_degree==2:
        return np.array([0,1,1,1,2,2,2,2,2,2])
    assert(NotImplementedError)



def test_multiple_disc(experiment_dataset, cross_validation_dataset, cov_noise,  batch_size, discretization_params, signal_variance, return_all = False):

    discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWX, F = precompute_kernel(experiment_dataset, cov_noise.astype(np.float32),  batch_size, pol_degree=max_pol_degree)
    n_disc_test = len(discretization_params)


    # Compute weights for each discretization
    weights = np.zeros((n_disc_test, experiment_dataset.volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    valid_weights = np.zeros((n_disc_test, experiment_dataset.volume_size), dtype = bool)
    utils.report_memory_device(logger=logger)
    XWX = np.asarray(XWX)
    F = np.asarray(F)
    for idx, (pol_degree, h, reg) in enumerate(discretization_params):
        logger.info(f"computing discretization with params: degree={pol_degree}, h={h}, reg={reg}")
        reg_used = signal_variance if reg else None
        weights_this, valid_weights_this = compute_weights_from_precompute(experiment_dataset, XWX, F, reg_used, pol_degree, max_pol_degree, h)
        weights[idx,:,:weights_this.shape[-1]] = weights_this
        valid_weights[idx] = valid_weights_this


    logger.info(f"Done computing params")
    weights = weights.swapaxes(0,1)
    utils.report_memory_device(logger=logger)
    del XWX, F

    # residuals to pick best one
    residuals, _ = compute_residuals_many_weights_in_weight_batch(cross_validation_dataset, weights, max_pol_degree )
    residuals_averaged = regularization.batch_average_over_shells(residuals.T, experiment_dataset.volume_shape,0)

    # Make choice. Impose that h must be increasing
    index_array = jnp.argmin(residuals_averaged, axis = 0)
    index_array_vol = utils.make_radial_image(index_array, experiment_dataset.volume_shape)

    disc_choices = np.array(discretization_params)[index_array]
    hs_choices = disc_choices[:,1]
    hs_choices = np.maximum.accumulate(hs_choices)
    disc_choices[:,1] = hs_choices

    weights_opt = jnp.take_along_axis(weights[...,0], np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)

    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(weights), discretization_params # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)


def make_regularization_from_reduced(regularization_reduced):
    # signal variance, deriv_variance, hessian_variance
    if regularization_reduced.shape[-1] == 1:
        return regularization_reduced
    if regularization_reduced.shape[-1] == 2:
        return jnp.concatenate([regularization_reduced[...,0:1], 
                                jnp.repeat(regularization_reduced[...,1:2], repeats =3, axis=-1, total_repeat_length=3 ) ], axis = -1)
    if regularization_reduced.shape[-1] == 3:
        return jnp.concatenate([regularization_reduced[...,0:1], 
                                jnp.repeat(regularization_reduced[...,1:2], repeats =3, axis=-1,total_repeat_length=3),
                                jnp.repeat(regularization_reduced[...,2:3], repeats =6, axis=-1,total_repeat_length=6) ],
                                axis = -1)



def compute_weights_from_precompute(experiment_dataset, XWX, F, signal_variance, pol_degree, max_pol_degree, h):
    use_regularization = signal_variance is not None

    # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(experiment_dataset.volume_size), experiment_dataset.volume_shape ) * 1.0

    if type(h) == float or type(h) == int:
        h_max = int(h)
        h_ar = h*np.ones(experiment_dataset.volume_size)
    else:
        h_max = int(np.max(h))
        h_ar = h.astype(int) * 1.0

    feature_size = get_feature_size(pol_degree)

    if XWX.shape[-1] != small_gram_matrix_size(pol_degree):
        triu_indices = find_smaller_pol_indices(max_pol_degree, pol_degree)
        XWX = XWX[...,triu_indices]
        F = F[...,:feature_size]

    # n_pol_param = small_gram_matrix_size(pol_degree) + 2 * get_feature_size(pol_degree) + 1
    memory_per_pixel = (2*h_max +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    ## There seems to be a strange bug with JAX. If it just barely runs out of memory, it won't throw an error but the memory will get corrupted and the answer is nonsense. This is an incredibly difficult thing to debug. 
    if h_max == 0:
        memory_per_pixel = (2*1 +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4

    # batch_size = int(utils.get_gpu_memory_total() / (memory_per_pixel * 16 * 20/ 1e9))
    # n_batches = np.ceil(experiment_dataset.volume_size / batch_size).astype(int)


    # print('CHANGE BACK TYPE?')
    # XWX = jnp.asarray(XWX).astype(np.float64)
    # F = jnp.asarray(F).astype(np.complex128)

    XWX = jnp.asarray(XWX)
    F = jnp.asarray(F)
    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX) - utils.get_size_in_gb(F)  )/ (memory_per_pixel   /1e9  )  ) 
    n_batches = np.ceil(experiment_dataset.volume_size / batch_size).astype(int)

    reconstruction = np.zeros((experiment_dataset.volume_size, feature_size), dtype = np.complex64)
    good_pixels = np.zeros((experiment_dataset.volume_size), dtype = bool)
    logger.info(f"KE batch size: {batch_size}")
    msgs = 0

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(experiment_dataset.volume_size, batch_size, k)
        signal_variance_this = signal_variance[ind_st:ind_end, :pol_degree+1] if use_regularization else None
        reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems = compute_estimate_from_precompute_one(XWX, F, h_max, h_ar[ind_st:ind_end], threed_frequencies[ind_st:ind_end], experiment_dataset.volume_shape, pol_degree =pol_degree, signal_variance=signal_variance_this, use_regularization = use_regularization)
        if k < 3:
            utils.report_memory_device(logger=logger)
            print(k)
        # if problems.any():   
        #     logger.warning(f"Issues in linalg solve? Problems for {problems.sum() / problems.size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")

        if jnp.isnan(reconstruction[ind_st:ind_end]).any():
            logger.warning(f"IsNAN {jnp.isnan(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")
        # if jnp.isinf(reconstruction[ind_st:ind_end]).any():
        if problems.any():
            logger.warning(f"Issues in linalg solve? Problems for {problems.sum() / problems.size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")
            if msgs < 10: 
                msgs +=1
                logger.warning(f"isinf {jnp.isinf(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")

            # XWX_undo = undo_keep_upper_triangular(XWX, reconstruction.shape[-1])

            # recon2 , good_v_v, problems_v = batch_solve_for_m(XWX_undo[ind_st:ind_end], F[ind_st:ind_end], 1/signal_variance_this )

            # bad_idx = jnp.isinf(reconstruction[ind_st:ind_end]).any(axis=-1)
            # bad_indices = np.arange(ind_st, ind_end)[bad_idx]
            # # bad_indices = np.nonzero(bad_idx)[0]
            # # import pdb; pdb.set_trace()
            # vmapcond = jax.vmap(jnp.linalg.cond, in_axes = 0)

            # # print(np.max(vmapcond(XWX_undo[bad_indices])))
            # x = core.vec_indices_to_frequencies(bad_indices, experiment_dataset.volume_shape)
            # print(x)
            # print(np.min(np.linalg.norm(x, axis=-1)))
            # import pdb; pdb.set_trace()
    logger.info(f"Done with kernel estimate")
    return reconstruction, good_pixels#, np.asarray(density_over_noise), XWX, F


# def adaptive_disc(experiment_dataset, cov_noise,  batch_size, pol_degree=0, h =1, signal_variance=None):
#     use_regularization = signal_variance is not None
#     XWX, F = precompute_kernel(experiment_dataset, cov_noise,  batch_size, pol_degree=pol_degree)
#     return compute_weights_from_precompute(experiment_dataset, XWX, F, signal_variance, pol_degree, h)


@functools.partial(jax.jit, static_argnums = [5,6,7,8,9, 10])    
def compute_residuals_batch_many_weights(images, weights, rotation_matrices, translations, CTF_params, volume_shape, image_shape, grid_size, CTF_fun, voxel_size, pol_degree ):

    X_mat, gridpoint_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = pol_degree)
    weights_on_grid = weights[gridpoint_indices]

    X_mat = jnp.repeat(X_mat[...,None,:], axis = -2, repeats = weights_on_grid.shape[-2])

    predicted_phi = linalg.broadcast_dot(X_mat, weights_on_grid)

    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    translated_images = core.translate_images(images, translations, image_shape)
    residuals = jnp.abs(translated_images[...,None] - predicted_phi * CTF[...,None])**2
    
    volume_size = np.prod(volume_shape)
    summed_residuals = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, residuals, gridpoint_indices.reshape(-1), None)

    summed_n = core.summed_adjoint_slice_by_nearest(volume_size, jnp.ones_like(residuals[...,0]), gridpoint_indices.reshape(-1))

    return summed_residuals, summed_n


def compute_residuals_many_weights(experiment_dataset, weights , pol_degree ):
    
    batch_size =int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - utils.get_size_in_gb(weights) ) * 5 / np.prod(weights.shape[1:]))

    residuals, summed_n =0, 0
    logger.info(f"batch size in residual computation: {batch_size}")
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    weights = jnp.asarray(weights)
    for batch, indices in data_generator:
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)

        residuals_t, summed_n_t = compute_residuals_batch_many_weights(batch, weights,
                                            experiment_dataset.rotation_matrices[indices],
                                            experiment_dataset.translations[indices], 
                                            experiment_dataset.CTF_params[indices], 
                                            experiment_dataset.volume_shape, 
                                            experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, experiment_dataset.voxel_size, pol_degree )
        residuals += residuals_t
        summed_n += summed_n_t

    return residuals , summed_n


def compute_residuals_many_weights_in_weight_batch(experiment_dataset, weights, pol_degree ):
    
    # Keep half memory free for other stuff

    n_batches = utils.get_size_in_gb(weights) / (0.5 * utils.get_gpu_memory_total() )
    weight_batch_size = np.floor(weights.shape[-2] / n_batches).astype(int)
    n_batches = np.ceil(weights.shape[-2] / weight_batch_size).astype(int)

    logger.info(f"number of batches in residual computation: {n_batches}, batch size: {weight_batch_size}")

    residuals = []
    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(weights.shape[-2], weight_batch_size, k)
        res, _ = compute_residuals_many_weights(experiment_dataset,
                                                weights[:,ind_st:ind_end],
                                                pol_degree )
        residuals.append(res)

    return np.concatenate(residuals, axis = -1), 0




# ### PICK H BASED ON ASYMPTOTICS

# ## High level functions
# # Integral of u^2 K(u) du over R^3
# mu_2_kernel_statistics = { 'cube' : 1/27, 'Epanechnikov' : 3/5 } 

# # Integral of K(u)^2 du over R^3 . K(u) is normalized such that integral K(u) du = 1 
# R_kernel_statistics = { 'cube': 1/8,  'ball' : 4/3 * np.pi , 'Epanechnikov' : 1/5 } 

# # mu_2_kernel_statistics = { 'uniform' : 1/2, 'Epanechnikov' : 3/5 } 
# def Epanechnikov_kernel(dist_squared, h=1):
#     return 3/4 * jnp.where( dist_squared < h,  1- dist_squared/h, jnp.zeros_like(dist_squared) )

# def uniform_kernel(dist_squared, h=1):
#     return jnp.where( dist_squared < h,  jnp.ones_like(dist_squared), jnp.zeros_like(dist_squared) )

# def predict_optimal_h_value(noise_variance_over_density, hessian_norm, kernel = 'cube' ):
#     R = R_kernel_statistics[kernel]
#     mu_2 = mu_2_kernel_statistics[kernel]
#     # h = (R * noise_variance_over_density / mu_2 / hessian_norm_squared)**(1/5)
#     # Theorem 6.1 https://bookdown.org/egarpor/PM-UC3M/npreg-kre.html
#     B_p_squared =  ( mu_2 *  hessian_norm /2)**2
#     d = 3
#     h = (d * R * noise_variance_over_density / ( 4 * B_p_squared))**(1/(4 + d))
#     # B_p^2 * h^4  + noise_variance_over_density * R / h^d 
#     # Deriv is :
#     # B_p^2 * 4 h^3  + noise_variance_over_density * R * (-d) * h^(-d-1) = 0 
#     # Solving for h:
#     # h = (d * R * density_over_variance / (B_p^2 * 4) )**(1/(4 + d)) 

#     return h

def compute_gradient(x):
    gradients = jnp.gradient(x)
    gradients = jnp.stack(gradients, axis=0)
    return gradients

def compute_gradient_norm_squared(x):
    gradients = compute_gradient(x)
    grad_norm = jnp.linalg.norm(gradients, axis = (0), ord=2)**2
    return grad_norm

def compute_hessian(x):
    gradients = jnp.gradient(x)
    hessians = [jnp.gradient(dx) for dx in gradients ]
    hessians = np.stack(hessians, axis=0)
    return hessians

def compute_hessian_norm_squared(x):
    hessians = compute_hessian(x)
    return jnp.linalg.norm(hessians, axis = (0,1), ord =2)**2

