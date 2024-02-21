import logging
import jax.numpy as jnp
import numpy as np
import jax, time

from recovar import core, covariance_estimation, embedding, plot_utils, linalg, constants, utils, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def estimate_principal_components(cryos, options,  means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use, noise_model,  disc_type = 'linear_interp', radius = 5):

    volume_shape = cryos[0].volume_shape
    vol_batch_size = utils.get_vol_batch_size(cryos[0].grid_size, gpu_memory_to_use)

    covariance_cols, picked_frequencies, column_fscs = covariance_estimation.compute_regularized_covariance_columns(cryos, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory_to_use, noise_model, disc_type = disc_type, radius = constants.COLUMN_RADIUS)
    logger.info("memory after covariance estimation")
    utils.report_memory_device(logger=logger)
    

    if options['ignore_zero_frequency']:
        zero_freq_index = np.asarray(core.frequencies_to_vec_indices( np.array([0,0,0]), cryos[0].volume_shape)).astype(int)
        zero_freq_in_picked_freq = np.where(picked_frequencies == zero_freq_index)[0].astype(int)

        # Set covariances with frequency 0 to 0.
        # I am not this if this is a good idea...
        # covariance_cols['est_mask'][:,zero_freq_in_picked_freq] *= 0 
        # covariance_cols['est_mask'][zero_freq_index,:] *= 0 

    # First approximation of eigenvalue decomposition
    u,s = get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, options['ignore_zero_frequency'])
    


    # # Let's see?
    # if noise_model == "white":
    #     cov_noise = cov_noise
    # else:
    #     # This probably should be moved into embedding
    if options['ignore_zero_frequency']:
        # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
        logger.info('ignoring zero frequency')
        cov_noise[0] *=1e16
        
    image_cov_noise = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].image_shape))

    u['rescaled'],s['rescaled'] = pca_by_projected_covariance(cryos, u['real'], means['combined'], image_cov_noise, dilated_volume_mask, disc_type ='linear_interp', gpu_memory_to_use= gpu_memory_to_use, use_mask = True, parallel_analysis = False ,ignore_zero_frequency = False)

    if options['ignore_zero_frequency']:
        logger.warning("FIX THIS OPTION!! NOT SURE IT WILL STILL WORK")
    # u['rescaled_old'],s['rescaled_old'], zss['init'] = rescale_eigs(cryos, u['real'],s['real'], means['combined'], volume_mask, image_cov_noise, basis_size = constants.N_PCS_TO_COMPUTE, gpu_memory_to_use = gpu_memory_to_use, use_mask = True, ignore_zero_frequency = options['ignore_zero_frequency'])

    # logger.info(f"u after rescale dtype: {u['rescaled'].dtype}")
    logger.info("memory after rescaling")
    utils.report_memory_device(logger=logger)
    if (options['contrast'] == "contrast_qr"):
        c_time = time.time()
        # Going to keep a copy around for debugging purposes. Probably should delete at some point to reduce memory 
        u['rescaled_no_contrast'] = u['rescaled'].copy()
        s['rescaled_no_contrast'] = s['rescaled'].copy()

        mean_used = means['combined']
        # if options['ignore_zero_frequency']:
        #     mean_used[...,zero_freq_index] = 0
        u['rescaled'],s['rescaled'] = knock_out_mean_component_2(u['rescaled'], s['rescaled'],mean_used, volume_mask, volume_shape, vol_batch_size, options['ignore_zero_frequency'])

        logger.info(f"knock out time: {time.time() - c_time}")
        if u['rescaled'].dtype != cryos[0].dtype:
            logger.warning(f"u['rescaled'].dtype: {u['rescaled'].dtype}")

            
    return u, s, covariance_cols, picked_frequencies, column_fscs


def get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape,  vol_batch_size, gpu_memory_to_use, ignore_zero_frequency ):
    u = {}; s = {}    

    u['real'], s['real'],_ = randomized_real_svd_of_columns(covariance_cols["est_mask"], picked_frequencies, volume_mask, volume_shape, vol_batch_size, test_size=constants.RANDOMIZED_SKETCH_SIZE, gpu_memory_to_use=gpu_memory_to_use, ignore_zero_frequency=ignore_zero_frequency)

    # u['real'], s['real'] = randomized_real_svd_of_columns_with_s_guess(covariance_cols["est_mask"],  picked_frequencies, volume_mask, volume_shape, vol_batch_size, test_size = constants.RANDOMIZED_SKETCH_SIZE, gpu_memory_to_use = gpu_memory_to_use, ignore_zero_frequency = ignore_zero_frequency )

    # n_components = covariance_cols["est_mask"].shape[-1]
    # u['real'], s['real'] = real_svd3(covariance_cols["est_mask"],  picked_frequencies, volume_shape, vol_batch_size, n_components = n_components)
    # Will cause out of memory problems.
    # if volume_shape[0] <= 128:
    #     u['real4'], s['real4'] = real_svd4(covariance_cols["est_mask"],  picked_frequencies, volume_shape, vol_batch_size, n_components = n_components)
    # plot_utils.plot_cov_results(u,s)
    return u, s


def pca_by_projected_covariance(cryos, basis, mean, noise_variance, volume_mask, disc_type ='linear_interp', gpu_memory_to_use= 40, use_mask = True, parallel_analysis = False ,ignore_zero_frequency = False):

    # basis_size = basis.shape[-1]
    basis_size = constants.N_PCS_TO_COMPUTE
    basis = basis[:,:basis_size]

    ####
    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
    batch_size = utils.get_embedding_batch_size(basis, cryos[0].image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )

    # memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() 
    # batch_size = utils.get_embedding_batch_size(basis, cryos[0].image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )
    # batch_size = batch_size//3

    logger.info('batch size for covariance computation: ' + str(batch_size))

    covariance = covariance_estimation.compute_projected_covariance(cryos, mean, basis, volume_mask, noise_variance, batch_size,  disc_type, parallel_analysis = False )

    ss, u = np.linalg.eigh(covariance)
    u =  np.fliplr(u)
    s = np.flip(ss)
    u = basis @ u 

    s = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)
    
    return u , s

## EVERYTHING BELOW HERE IS NOT USED IN CURRENT VERSION OF THE CODE. DELETE?



def knock_out_mean_component_2(u,s, mean, volume_mask, volume_shape, vol_batch_size,ignore_zero_frequency):
    # This assumes s has been kept around
    # cov == u s u^*
    # Want to compute eigendecomposition of the projection onto complement of mean:
    # (I - qq^*) cov ( I - q q^*)
    
    volume_size = np.prod(volume_shape)
    u_real = linalg.batch_idft3(u, volume_shape, vol_batch_size).real
    u_real /= np.linalg.norm(u_real, axis =0)
    # u2_norm = np.linalg.norm(u_real, axis =0)

    # Mask mean
    masked_mean = ( ftu.get_idft3(mean.reshape(volume_shape)) * volume_mask.reshape(volume_shape) ).reshape(-1).real
    masked_mean /= np.linalg.norm(masked_mean)
    
    # Make it orthogonal to mask
    if ignore_zero_frequency:
        # knockout volume_mask direction stuff?
        norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
        # substract component in direction of mask?
        # Apply matrix (I - mask mask.T / \|mask^2\| ) 
        masked_mean -= norm_volume_mask * (norm_volume_mask.T @ masked_mean)


    # Project out the mean
    u_m_proj = u_real - masked_mean[:,None] @  (np.conj(masked_mean).T @ u_real )[None]
    cov_chol = u_m_proj * np.sqrt(s)

    # Reorthogonalize
    # Replaced by a slower but stable.
    # new_u, new_s, _ = linalg.thin_svd_in_blocks(cov_chol)
    cov_chol = jax.device_put(cov_chol, device=jax.devices("cpu")[0])
    new_u, new_s, _ = jnp.linalg.svd(cov_chol, full_matrices = False)

    new_u = np.array(new_u) 
    new_s = np.array(new_s)


    ones_vol = np.ones_like(masked_mean)
    ones_vol /= np.linalg.norm(ones_vol)

    # Align to positive. Not really necessary, but 
    ip = ones_vol.T @ new_u
    ip = np.where(np.abs(ip) > constants.ROOT_EPSILON, ip / np.abs(ip) , np.ones_like(ip))
    new_u *= ip

    # back to Fourier domain
    new_u = linalg.batch_dft3(new_u, volume_shape, vol_batch_size)
    new_u /= np.linalg.norm(new_u, axis =0)
    
    return np.array(new_u.astype(u.dtype)), np.array(new_s.astype(s.dtype)**2)


# A lot of implementation of the same things, having to do with taking the real SVD of
# the columns of Sigma_col
# - First is doubline Sigma in the Fourier domain (using the symmetry), then doing a normal SVD
# in the Fourier domain, then casting to spatial domain and try to align to make it real
# - Second explicitely computed Sigma in the spatial domain then does a real SVD (very memory expensive)
# - Third does a randomized SVD in the spatial domain.

def flip_vec(column, volume_shape):
    column = column.reshape(volume_shape)
    column_flipped = jnp.zeros_like(column)
    column_flipped = column_flipped.at[1:,1:,1:].set(jnp.conj(jnp.flip(column[1:,1:,1:])))
    return column_flipped.reshape(-1)

def get_zero_boundary_mask(volume_shape, dtype):
    ones = np.zeros(volume_shape, dtype = dtype)
    ones[1:,1:,1:] = 1
    return ones.reshape(-1)


def get_minus_vec_index(picked_v_idx,volume_shape):
        # Get - vec
        freq = core.vec_indices_to_frequencies(picked_v_idx, volume_shape)
        minus_idx = core.frequencies_to_vec_indices(-freq, volume_shape)
        return minus_idx
    
flip_vec_cpu = jax.jit(flip_vec, backend = 'cpu', static_argnums = (1,))
batch_flip_vec = jax.vmap(flip_vec_cpu, in_axes = (1, None))

def make_symmetric_columns(columns, picked_frequencies, volume_shape):
    freqs = core.vec_indices_to_frequencies(picked_frequencies, volume_shape)
    
    good_idx = freqs[:,0] > 0
    # freqs = freqs.at[good_idx].get()
    minus_freqs = -freqs
    minus_indices = core.frequencies_to_vec_indices(minus_freqs, volume_shape)
    columns_flipped = batch_flip_vec(columns, volume_shape)
    
    return columns_flipped.T, minus_indices, good_idx

make_symmetric_columns_cpu = jax.jit(make_symmetric_columns, backend = 'cpu', static_argnums = (2))

# to check if there is. abug... delete?
def make_symmetric_columns_np(columns, picked_frequencies, volume_shape):
    freqs = np.array(core.vec_indices_to_frequencies(picked_frequencies, volume_shape))
    
    good_idx = freqs[:,0] > 0
    # freqs = freqs.at[good_idx].get()
    minus_freqs = -freqs
    minus_indices = np.array(core.frequencies_to_vec_indices(minus_freqs, volume_shape))
    columns_flipped = batch_flip_vec2(columns, volume_shape)
    return columns_flipped.T, minus_indices, good_idx
 

def batch_flip_vec2(columns,volume_shape):
    mapped_idx = np.array(get_minus_vec_index(np.arange(np.prod(volume_shape)),volume_shape))
    one_mask = get_zero_boundary_mask(volume_shape, columns.dtype)
    return np.conj(columns[mapped_idx,:] * one_mask[...,None]).T



def IDFT_from_both_sides(cube_smaller_matrix, left_volume_shape, right_volume_shape, vol_batch_size_left, vol_batch_size_right):
    # Apply fft along rows
    # Compute C = F Sigma
    # Then C^*
    cube_smaller_matrix = np.conj(linalg.batch_idft3(cube_smaller_matrix, left_volume_shape, vol_batch_size_left).T)
    # Apply fft along cols
    
    # Computes C F^* by (F C^*)^*
    cube_smaller_matrix = np.conj(linalg.batch_idft3(cube_smaller_matrix, right_volume_shape, vol_batch_size_right).T)
    return cube_smaller_matrix

def get_all_copied_columns(columns, picked_frequencies, volume_shape):
    
    # Make symmetric columns
    columns_flipped, minus_indices, good_idx = make_symmetric_columns_np(columns, picked_frequencies, volume_shape)
    all_frequencies = np.concatenate([picked_frequencies, minus_indices[good_idx]])
    all_columns = np.concatenate([columns, columns_flipped[:,good_idx]], axis =-1)
    return all_columns, all_frequencies


# IMPLEMENTS THE TWO MATVECS WE NEED TO RUN THE RANDOMIZED SVD.

def right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time = time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    logger.info(f"make big mat {time.time() - st_time}")
    # columns_flipped = np.array(columns_flipped)
    utils.report_memory_device(logger=logger)

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(all_frequencies) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # F_2r^* test_mat
    F_t = linalg.batch_dft3(test_mat, smaller_vol_shape, vol_batch_size) / smaller_vol_size
    logger.info(f"DFT time {time.time() - st_time}")

    
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
    utils.report_memory_device(logger=logger)

    C_F_t = linalg.blockwise_A_X(columns, F_t[original_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use)
    logger.info(f"AX: {time.time() - st_time}")

    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))

    columns_flipped = columns_flipped[:,good_idx]
    F_t2 = F_t[flipped_frequencies_indices_in_smaller,:].copy()
    C_F_t_2 = linalg.blockwise_A_X(columns_flipped , F_t2, memory_to_use = memory_to_use) 
    C_F_t += C_F_t_2 
    logger.info(f"AX: {time.time() - st_time}")


    F_C_F_t = linalg.batch_idft3(C_F_t, volume_shape, vol_batch_size)
    logger.info(f"IDFT: {time.time() - st_time}")

    return F_C_F_t

def left_matvec_with_spatial_Sigma(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time =time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    
    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(all_frequencies) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # Now do compute:
    # F = IDFT here
    # so F^* = DFT
    
    # Q should be real I think?
    F_star_Q_star = linalg.batch_dft3( np.conj(Q), volume_shape, vol_batch_size) / np.prod(volume_shape)
    # Q_F = np.conj(F_star_Q_star)
    Q_F = F_star_Q_star
    logger.info(f"DFT: {time.time() - st_time}")

    # Frequencies in new grid
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
        
    Q_F_C = np.zeros((Q.shape[-1], smaller_vol_size), dtype = columns.dtype)
    Q_F_C[:,original_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")

    # Flipped Frequencies in new grid
    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape)
    flipped_frequencies_indices_in_smaller = np.array(flipped_frequencies_indices_in_smaller)
    # Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_A_X(columns_flipped[:,good_idx].T, Q_F.T, memory_to_use = 20).T

    columns_flipped = columns_flipped[:,good_idx]
    Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns_flipped, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")
    
    # DFT back
    # X F^* = (F X^*)^*
    Q_F_C_F = np.conj(linalg.batch_idft3(np.conj(Q_F_C).T, smaller_vol_shape, vol_batch_size)).T
    logger.info(f"DFT2: {time.time() - st_time}")

    return Q_F_C_F


def randomized_real_svd_of_columns(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, test_size = 300, gpu_memory_to_use= 40, ignore_zero_frequency = False):
    st_time = time.time()


    # memory_to_use = utils.get_gpu_memory_total() - 5
    utils.report_memory_device(logger=logger)
    picked_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    smaller_size = int(2 * (np.max(picked_frequencies) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])

    smaller_vol_size = np.prod(smaller_vol_shape)
    test_mat = np.random.randn(smaller_vol_size, test_size).real.astype(np.float32)

    st_time = time.time()
    Q = right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use ).real.astype(np.float32)
    
    ## Do masking here ?
    Q *= volume_mask.reshape(-1,1)

    if ignore_zero_frequency:
        # knockout volume_mask direction stuff?
        norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
        # substract component in direction of mask?
        # Apply matrix (I - mask mask.T / \|mask^2\| ) 
        Q -= np.outer(norm_volume_mask, (norm_volume_mask.T @ Q))
        logger.info('ignoring zero frequency')

    logger.info(f"right matvec {time.time() - st_time}")
    utils.report_memory_device(logger=logger)
    Q = jax.device_put(Q, device=jax.devices("cpu")[0])
    Q,_ = jnp.linalg.qr(Q)
    Q = np.array(Q) # I don't know why but not doing this causes massive slowdowns sometimes?
    logger.info(f"QR time: {time.time() - st_time}")

    # In principle, should apply (I - mask mask.T / \|mask\|^2 )  again, but should already be orthogonal

    # 
    Q_mask = Q*volume_mask.reshape(-1,1)
    
    C_F_t_2 = left_matvec_with_spatial_Sigma(Q_mask, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    del Q_mask
    utils.report_memory_device(logger=logger)
    logger.info(f"left matvec {time.time() - st_time}")


    U,S,V = np.linalg.svd(C_F_t_2)
    logger.info(f"big SVD {time.time() - st_time}")
    
    vol_size = np.prod(volume_shape)
    F_Q = linalg.batch_dft3(Q, volume_shape, vol_batch_size)
    UU = linalg.blockwise_A_X(F_Q, U, memory_to_use = gpu_memory_to_use) / np.sqrt(vol_size)
    logger.info(f"FQU matvec {time.time() - st_time}")

    volume_size = np.prod(volume_shape)
    # Factors due to IDFT on both sides
    S_fd = S * np.sqrt(smaller_vol_size) * np.sqrt(volume_size)
    return np.array(UU), np.array(S_fd), np.array(V)

## everything below is from old versions. DELETE?


# def compute_real_matrix_big(all_columns, all_frequency_indices, volume_shape, vol_batch_size):

#     grid_size = volume_shape[0]

#     # Make symmetric columns
#     all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
#     smaller_size = int(2 * (np.max(all_frequencies) + 1))
#     smaller_vol_shape = tuple(3*[smaller_size])
#     st_time = time.time()
#     # Make a N^3 x d^3 matrix, so that we can 3D-DFT from the right and the left (of size N and d)
#     smaller_freq_indices = core.frequencies_to_vec_indices(all_frequencies, smaller_vol_shape)
#     cube_smaller_matrix = np.zeros( shape = (grid_size**3, np.prod(np.array(smaller_vol_shape))), dtype = 'complex64')
#     cube_smaller_matrix[:,np.array(smaller_freq_indices)] = all_columns
#     del all_columns
#     print('big mat, make matrix', time.time() - st_time )
#     st_time = time.time()

#     vol_batch_size_left = vol_batch_size
#     vol_batch_size_right = vol_batch_size_left * grid_size**3 / np.prod(np.array(smaller_vol_shape))

#     cube_smaller_matrix = IDFT_from_both_sides(cube_smaller_matrix, volume_shape, smaller_vol_shape, vol_batch_size_left, vol_batch_size_right )
#     print('IDFT of big mat', time.time() - st_time)

#     print('imag part norm: ', np.linalg.norm(cube_smaller_matrix.imag) / np.linalg.norm(cube_smaller_matrix))

#     return cube_smaller_matrix.real

# def compute_real_matrix_small(all_columns, all_frequency_indices, volume_shape, vol_batch_size, gpu_memory_to_use =40):

#     all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
#     smaller_size = int(2 * (np.max(all_frequencies) + 1))
#     smaller_vol_shape = tuple(3*[smaller_size])

#     # Make a N^3 x d^3 matrix, so that we can 3D-DFT from the right and the left (of size N and d)
#     smaller_freq_indices = core.frequencies_to_vec_indices(all_frequencies, smaller_vol_shape)
#     smaller_vol_size = np.prod(np.array(smaller_vol_shape))

#     cube_smaller_matrix = np.zeros( shape = (smaller_vol_size, smaller_vol_size), dtype = 'complex64')
#     cube_smaller_matrix[np.ix_(np.array(smaller_freq_indices),np.array(smaller_freq_indices))] = all_columns

#     vol_batch_size_left = utils.get_vol_batch_size(smaller_size, gpu_memory_to_use)
#     vol_batch_size_right = utils.get_vol_batch_size(smaller_size, gpu_memory_to_use)

#     cube_smaller_matrix = IDFT_from_both_sides(cube_smaller_matrix, smaller_vol_shape, smaller_vol_shape, vol_batch_size_left, vol_batch_size_right)

#     # print('imag part norm: ', np.linalg.norm(cube_smaller_matrix.imag) / np.linalg.norm(cube_smaller_matrix))

#     return cube_smaller_matrix.real



# def svd_of_small_matrix(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, ignore_zero_frequency):
    
#     # Explicitely apply mask to columns:
#     columns_real = linalg.batch_idft3(columns, volume_shape, vol_batch_size)
#     columns_real *= volume_mask.reshape(-1,1)#[...,None]
#     if ignore_zero_frequency:
#         # knockout volume_mask direction stuff?
#         norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
#         # substract component in direction of mask?
#         # Apply matrix (I - mask mask.T / \|mask^2\| ) 
#         columns_real -= np.outer(norm_volume_mask, (norm_volume_mask.T @ columns_real))

#     columns = linalg.batch_dft3(columns_real, volume_shape, vol_batch_size)


#     # Make symmetric columns
#     columns_flipped, minus_indices, good_idx = make_symmetric_columns_cpu(columns, picked_frequency_indices, volume_shape)
#     all_frequency_indices = np.concatenate([picked_frequency_indices, minus_indices[good_idx]])


#     picked_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
#     smaller_size = int(2 * (np.max(picked_frequencies) + 1))
#     smaller_vol_shape = tuple(3*[smaller_size])
#     smaller_vol_size = np.prod(smaller_vol_shape)

#     # A lot of repeated, unnecessary work in here. Probably should merge/clean up with other function.
#     all_columns = np.concatenate([columns, columns_flipped[:,good_idx]], axis =-1)
#     cube_smaller_matrix_small = compute_real_matrix_small(all_columns[all_frequency_indices,:], all_frequency_indices, volume_shape, vol_batch_size, gpu_memory_to_use)
    
#     ssmall =  jnp.linalg.svd(cube_smaller_matrix_small, compute_uv = False)
#     # IDFT on both sides
#     ssmall *=  np.sqrt(smaller_vol_size) * np.sqrt(smaller_vol_size)
#     return ssmall.astype(columns.dtype).real


# def randomized_real_svd_of_columns_with_s_guess(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, ignore_zero_frequency,  test_size = 300, gpu_memory_to_use = 40):

#     u,s,_ = randomized_real_svd_of_columns(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, test_size, gpu_memory_to_use, ignore_zero_frequency=ignore_zero_frequency)

#     # Is this a more reasonable estimate?
#     # guess_from_s_factor = False
#     # if guess_from_s_factor:
#     #     # s-factor
#     #     Su_norms = np.linalg.norm(u[:,picked_frequency_indices], axis=0)
#     #     s_guess = s * Su_norms
#     #     s_guess = np.minimum.accumulate(s_guess)
#     #     return u, s_guess

#     # In new version of the code which uses the projected PCA, this is not used
#     s_small = svd_of_small_matrix(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, ignore_zero_frequency)

#     # Guess eigenvalue with rank 1 estimate
#     n_components = s.size
#     s_guess =  s[:n_components]**2 / s_small[:n_components]

#     # Is this a terrible idea?
#     s_guess = np.minimum.accumulate(s_guess)
#     return u , s_guess




# def rescale_eigs(cryos,u,s, mean, volume_mask, cov_noise, basis_size = 200, gpu_memory_to_use= 40, use_mask = True, ignore_zero_frequency = False):
#     # Implements the approximate SVD from the paper

#     rescale_time = time.time()    
#     # basis_size = 200 if basis_size is None else basis_size
#     volume_shape = cryos[0].volume_shape
#     contrast_option = 'None'
#     mask = volume_mask if use_mask is True else np.ones_like(volume_mask)   

#     zs12, _, _ = embedding.get_per_image_embedding(mean, u,s, basis_size,
#                                       cov_noise, cryos, mask,gpu_memory_to_use,  'linear_interp',
#                                     contrast_grid = None, contrast_option = contrast_option,
#                                     to_real = True, parallel_analysis = False, 
#                                     compute_covariances = False, ignore_zero_frequency = ignore_zero_frequency )    
    
#     st_time = time.time()
#     _, sz, vz = np.linalg.svd(zs12, full_matrices = False)
#     logger.info(f"rescale svd time, {time.time() - st_time}")

#     u_rescaled = u[:,:basis_size] @ np.conj(vz.T) # Conj should be useless, I guess?
#     logger.info(f"rescale matvec, {time.time() - st_time}")

#     zero_freq = core.frequencies_to_vec_indices( np.array([[0,0,0]] ), volume_shape)
    
#     # Makes sure they are aligned to vector of all one, or to first frequency if it has 0 sum. 
#     # This isn't strictly necessary, but helps with getting repeatable eigenvectors as they are only unique up to complex sign
#     normalize = lambda x : x /np.abs(x)
#     ip = np.where( np.abs(u_rescaled[zero_freq,:]) > constants.ROOT_EPSILON ,
#                   normalize(u_rescaled[zero_freq,:]),
#                   np.ones_like(u_rescaled[zero_freq,:]))
#     u_rescaled /= ip
    

#     # Estimate eigenvalues from singular values
#     s_rescaled = sz**2/ zs12.shape[0]
#     logger.info(f"rescale time, {time.time() - rescale_time}")
#     return u_rescaled, s_rescaled[:basis_size], zs12


# def nystrom_correction_doubling(columns, picked_frequencies, volume_shape, eigenvalue_bump_mult = 0, epsilon = 1e-8, keep_eigs = 20):
    
#     all_columns, all_frequencies = get_all_copied_columns(columns, picked_frequencies, volume_shape)
#     one_one_block = all_columns[all_frequencies,:].copy()
    

#     # Make sure it is symmetric
#     one_one_block = 0.5 * ( one_one_block + np.conj(one_one_block).T ) 
    
#     # Compute square root
#     bump = np.linalg.norm(one_one_block) * eigenvalue_bump_mult
#     s, v = np.linalg.eigh(one_one_block + bump*np.eye(one_one_block.shape[0]))
#     inv_sqrt_eigs = np.where( s > epsilon , 1 / np.sqrt(s) , 0 ) 
#     inv_square_root = (v * inv_sqrt_eigs[None,:] ) @ np.conj(v).T 
    
#     # 
#     nystrom_factor = all_columns @ inv_square_root
    
#     #st_time = time.time()
#     u,s,_ = linalg.thin_svd_in_blocks(nystrom_factor)
#     #print(time.time() - st_time)

    
#     return u[:,:keep_eigs], s[:keep_eigs]**2


# def real_svd4(columns, picked_frequencies, volume_shape, vol_batch_size, n_components = 200):
#     # This probably the better way to do this. Take the matrix to real domain first, then SVD. But it takes a ton of memory

#     st_time = time.time()
#     st_st_time = time.time()
#     volume_size = np.prod(volume_shape)
#     print("in real svd")
#     import utils
#     print('memory used:', utils.get_process_memory_used())
#     # Make symmetric columns
#     columns_flipped, minus_indices, good_idx = make_symmetric_columns_cpu(columns, picked_frequencies, volume_shape)
#     all_frequency_indices = np.concatenate([picked_frequencies, minus_indices[good_idx]])

#     all_columns = np.concatenate([columns, columns_flipped[:,good_idx]], axis =-1)

#     print("make all columns", time.time() - st_time)
#     st_time = time.time()
#     print('memory used:', utils.get_process_memory_used())

#     del columns, columns_flipped
#     # all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
#     # Compute real matrices
#     cube_smaller_matrix_big = compute_real_matrix_big(all_columns, all_frequency_indices, volume_shape, vol_batch_size)

#     print("make big mat", time.time() - st_time)
#     st_time = time.time()
#     print('memory used:', utils.get_process_memory_used())

#     cube_smaller_matrix_small = compute_real_matrix_small(all_columns[all_frequency_indices,:], all_frequency_indices, volume_shape, vol_batch_size)
#     print("make small mat", time.time() - st_time)
#     st_time = time.time()

#     # SVD them
#     u, s, _ =  linalg.thin_svd_in_blocks(cube_smaller_matrix_big)#, full_matrices = False)
#     del cube_smaller_matrix_big
#     _, ssmall, _ =  linalg.thin_svd_in_blocks(cube_smaller_matrix_small)
#     print("SVD", time.time() - st_time)
#     st_time = time.time()
#     print('memory used:', utils.get_process_memory_used())

#     # Guess eigenvalue with rank 1 estimate
#     s_guess =  s[:n_components]**2 / ssmall[:n_components]
#     s_guess = np.minimum.accumulate(s_guess)


#     # Align eigenvectors to positive
#     signs = np.sum(u, axis = 0) > 0 # sign
#     signs = np.where(signs, 1, -1)
#     u = u * signs
#     u = u[:,:n_components]
#     u = linalg.batch_dft3(u[:,:n_components], volume_shape, vol_batch_size) / np.sqrt(volume_size)
#     print("final DFT", time.time() - st_time)

#     print("svd4 + fft", time.time() - st_st_time)
    
#     return u, s_guess


# def enforce_real_on_eigenvectors(u, volume_shape, vol_batch_size, from_ft = True):
#     # Align to positive axis
#     # u[zero_freq] == np.sum(Fu)== < Fu, 1 > 
#     st_time = time.time()
#     volume_size = np.prod(volume_shape)
#     if from_ft:
#         u_real = linalg.batch_idft3(u, volume_shape, vol_batch_size)
#         u_real *= np.sqrt(volume_size)
#     else:
#         u_real = u

#     # Align to positive. In principle, this should make it real. Except it can fail if sum(u) == 0.
#     # This may be overkill
#     ones_vol = np.ones(volume_size, dtype = u_real.dtype).real
#     ones_vol /= np.linalg.norm(ones_vol)
#     ip = ones_vol.T @ u_real

#     rand_vol = np.abs(np.random.randn(*ones_vol.shape).astype(u_real.dtype).real)
#     rand_vol /= np.linalg.norm(rand_vol)
#     ip2 = rand_vol.T @ u_real

#     ip_bestof = np.where(np.abs(ip) > constants.ROOT_EPSILON , ip, ip2)
#     ip_bestof = ip_bestof / np.abs(ip_bestof)

#     u_real /= ip_bestof
#     if np.any(np.linalg.norm(u_real - u_real.real, axis= 0 )) > 1e-3:
#         print('u_real not very orthogonal:', np.linalg.norm(u_real - u_real.real, axis= 0 ))
#         # print(np.linalg.norm(u_real - u_real.real, axis= 0 ))
#     # back to Fourier domain
#     u_real = linalg.batch_dft3(u_real.real, volume_shape, vol_batch_size)
#     u_real /= np.linalg.norm(u_real, axis =0)

#     print('time to make sure it is real:', time.time() - st_time)
#     return u_real


# def real_svd3(columns, picked_frequencies, volume_shape, vol_batch_size, n_components = 50, do_qr = True):
#     st_time = time.time()
#     volume_size = np.prod(volume_shape)
#     print("in real svd3")

#     # Make symmetric columns
#     columns_flipped, minus_indices, good_idx = make_symmetric_columns_cpu(columns, picked_frequencies, volume_shape)
#     all_frequencies = np.concatenate([picked_frequencies, minus_indices[good_idx]])
#     all_columns = np.concatenate([columns, columns_flipped[:,good_idx]], axis =-1)

#     # Guess eigenvalues, using sigma(C)^2 / sigma(W)
#     _, ssmall, _ =  linalg.thin_svd_in_blocks(all_columns[all_frequencies,:])
#     u, s, _ =  linalg.thin_svd_in_blocks(all_columns)    
#     u = u[:,:n_components]
#     s_guess =  s[:n_components]**2 / ssmall[:n_components]
#     s_guess = np.minimum.accumulate(s_guess)

#     # Align to positive axis
#     # u[zero_freq] == np.sum(Fu)== < Fu, 1 > 
#     # zero_freq = core.frequencies_to_vec_indices( jnp.array([[0,0,0]] ), volume_shape)
#     # ip = u[zero_freq,:] / np.abs(u[zero_freq,:])
#     # u /= ip
#     print("svd stuff", time.time() - st_time)

#     u2 = enforce_real_on_eigenvectors(u, volume_shape, vol_batch_size, from_ft = True)

#     # # go to real space and reorthogonalize
#     # u_real = linalg.batch_idft3(u, volume_shape, vol_batch_size).real
#     # u_real *= np.sqrt(volume_size)
#     # u2_norm = np.linalg.norm(u_real, axis =0)

#     # # This shouldn't be necessary.
#     # if volume_shape[0] > 128:
#     #     u_real, _ = np.linalg.qr(u_real)
#     # else:
#     #     u_real, _ = jnp.linalg.qr(u_real)
            
#     # u2 = linalg.batch_dft3(u_real, volume_shape, vol_batch_size) 

#     # Renormalize for good measure.    
#     u2_norm = np.linalg.norm(u2, axis =0)
#     u2 /= u2_norm

#     print("qr time", time.time() - st_time)
#     print('realsvd2 time:', time.time() - st_time)
#     return u2, s_guess


