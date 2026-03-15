"""Principal component analysis of the estimated covariance operator."""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config, utils
from recovar.core import linalg
from recovar.heterogeneity import covariance_estimation, embedding
from recovar.reconstruction import noise

logger = logging.getLogger(__name__)

# NVTX domain for principal components operations
NVTX_DOMAIN_PCA = "principal_components"


@nvtx.annotate("estimate_principal_components", color="purple", domain=NVTX_DOMAIN_PCA)
def estimate_principal_components(cryos, options,  means, mean_prior, volume_mask,
                                dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use,
                                covariance_options = None, variance_estimate = None, use_reg_mean_in_contrast = False, use_multi_gpu = False, n_gpus = None):
    """Estimate principal components of the covariance operator.

    Computes regularized covariance columns, then extracts their leading
    eigenvectors and eigenvalues via SVD.

    Args:
        cryos: Half-set datasets (``CryoEMHalfsets``).
        options: Pipeline options namespace.
        means: Dict with mean volume estimates.
        mean_prior: Prior mean volume (Fourier coefficients).
        volume_mask: Binary mask selecting valid voxels.
        dilated_volume_mask: Dilated version of *volume_mask*.
        valid_idx: Indices of valid Fourier frequencies.
        batch_size: Image batch size for GPU processing.
        gpu_memory_to_use: Available GPU memory in GB.
        covariance_options: Options dict (default: auto-generated).
        variance_estimate: Pre-computed variance estimate.
        use_reg_mean_in_contrast: Use regularized mean for contrast estimation.
        use_multi_gpu: Distribute across multiple GPUs.
        n_gpus: Number of GPUs (``None`` = auto-detect).

    Returns:
        Tuple ``(u, s, covariance_cols, picked_frequencies, column_fscs)``
        where *u* and *s* are dicts with keys ``'real'`` and ``'rescaled'``
        containing eigenvectors and eigenvalues respectively.
    """

    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options

    volume_shape = cryos.volume_shape
    vol_batch_size = utils.get_vol_batch_size(cryos.grid_size, gpu_memory_to_use)

    # Different way of sampling columns:
    # - from low to high frequencies
    # This is the way it was done in the original code.
    # - Highest SNR columns, computed by lhs of mean estimation. May want to not take frequencies that are too similar
    # - Highest variance columns. Also want want to diversify.
    # For the last one, could also batch by doing randomized-Cholesky like choice

    if covariance_options['column_sampling_scheme'] == 'low_freqs':
        from recovar.heterogeneity import covariance_core
        volume_shape = cryos.volume_shape
        if cryos.grid_size == 16:
            picked_frequencies = np.arange(cryos.volume_size)
        else:
            picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = covariance_options['column_radius'], use_half = True))
    elif covariance_options['column_sampling_scheme'] == 'high_snr' or covariance_options['column_sampling_scheme'] == 'high_lhs' or covariance_options['column_sampling_scheme'] == 'high_snr_p' or covariance_options['column_sampling_scheme'] =='high_snr_from_var_est':
        from recovar.reconstruction import regularization
        upsampling_factor = np.round((means.lhs.size / cryos.volume_size)**(1/3)).astype(int)
        upsampled_volume_shape = tuple(upsampling_factor * np.array(volume_shape))
        lhs = regularization.downsample_lhs(means.lhs.reshape(upsampled_volume_shape), volume_shape, upsampling_factor = upsampling_factor).reshape(-1)
        # At low freqs, signal variance decays as ~1/rad^2

        dist = (fourier_transform_utils.get_grid_of_radial_distances(volume_shape)+1)**2
        if covariance_options['column_sampling_scheme'] == 'high_snr':
            lhs = lhs / dist.reshape(-1)
        if covariance_options['column_sampling_scheme'] == 'high_snr_p':
            lhs = lhs * mean_prior
        if covariance_options['column_sampling_scheme'] == 'high_snr_from_var_est':
            if variance_estimate is None:
                raise ValueError("variance_estimate must be provided")
            lhs = lhs * variance_estimate

        if covariance_options['randomize_column_sampling']:
            picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.randomized_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])
        else:
            picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.greedy_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])

        logger.info("Largest frequency computed: %s", np.max(np.abs(picked_frequencies_in_frequencies_format)))
        if np.max(np.abs(picked_frequencies_in_frequencies_format)) > cryos.grid_size//2-1:
            logger.warning("Largest frequency computed is larger than grid size//2-1. This may cause issues in SVD. Check that variance estimates are correct")
    else:
        raise NotImplementedError('unrecognized column sampling scheme')
    
    covariance_cols, picked_frequencies, column_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory_to_use, covariance_options, picked_frequencies, use_multi_gpu = use_multi_gpu, n_gpus = n_gpus)
    
    # Check for NaN or Inf values in covariance_cols
    for col in covariance_cols.values():
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            raise ValueError("covariance_cols contains NaN or Inf values")
        if col.dtype != np.complex64:
            raise TypeError("covariance_cols is not of type np.complex64")

    # First approximation of eigenvalue decomposition
    u,s = get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, False, covariance_options['randomized_sketch_size'])
    # Check for NaN or Inf values in u and s
    if np.any(np.isnan(u['real'])) or np.any(np.isinf(u['real'])):
        raise ValueError("u['real'] contains NaN or Inf values")
    if np.any(np.isnan(s['real'])) or np.any(np.isinf(s['real'])):
        raise ValueError("s['real'] contains NaN or Inf values")

    # Check if the type of u and s is np.float32 or np.float64
    if u['real'].dtype not in [np.float32, np.complex64]:
        raise TypeError("u['real'] is not of type np.float32 or np.complex64")
    if s['real'].dtype not in [np.float32, np.complex64]:
        raise TypeError("s['real'] is not of type np.float32 or np.complex64")

    if not options.keep_intermediate:
        for key in covariance_cols.keys():
            covariance_cols[key] = None

    u['rescaled'], s['rescaled'] = pca_by_projected_covariance(cryos, u['real'], means.combined, dilated_volume_mask, disc_type = covariance_options['disc_type'], disc_type_u = covariance_options['disc_type_u'], gpu_memory_to_use= gpu_memory_to_use, use_mask = covariance_options['mask_images_in_proj'], ignore_zero_frequency = False, n_pcs_to_compute = covariance_options['n_pcs_to_compute'])

    if not options.keep_intermediate:
        u['real'] = None


    if options.ignore_zero_frequency:
        logger.warning("FIX THIS OPTION!! NOT SURE IT WILL STILL WORK")

    logger.info("memory after rescaling")
    utils.report_memory_device(logger=logger)

    if (options.contrast == "contrast_qr" or options.ignore_zero_frequency):
        c_time = time.time()
        # Keep reference (not copy) for debugging; numpy arrays below are reassigned, not mutated
        u['rescaled_no_contrast'] = u['rescaled']
        s['rescaled_no_contrast'] = s['rescaled']

        mean_used = (means.corrected0reg + means.corrected1reg) / 2 if use_reg_mean_in_contrast else means.combined
        u['rescaled'],s['rescaled'] = knock_out_mean_component_2(u['rescaled'], s['rescaled'],mean_used, volume_mask, volume_shape, vol_batch_size, options.ignore_zero_frequency, options.contrast == "contrast_qr" )

        if not options.keep_intermediate:
            u['rescaled_no_contrast'] = None

        logger.info("knock out time: %s", time.time() - c_time)
        if u['rescaled'].dtype != cryos[0].dtype:
            logger.warning("u['rescaled'].dtype: %s", u['rescaled'].dtype)

            
    return u, s, covariance_cols, picked_frequencies, column_fscs


# TODO: seems like a useless function, just move the inner one to other fn
@nvtx.annotate("get_cov_svds", color="blue", domain=NVTX_DOMAIN_PCA)
def get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape,  vol_batch_size, gpu_memory_to_use, ignore_zero_frequency, randomized_sketch_size ):
    u = {}; s = {}    
    
    u['real'], s['real'],_ = randomized_real_svd_of_columns(covariance_cols["est_mask"], picked_frequencies, volume_mask, volume_shape, vol_batch_size, test_size=randomized_sketch_size, gpu_memory_to_use=gpu_memory_to_use, ignore_zero_frequency=ignore_zero_frequency)

    return u, s


@nvtx.annotate("pca_by_projected_covariance", color="green", domain=NVTX_DOMAIN_PCA)
def pca_by_projected_covariance(cryos, basis, mean, volume_mask, disc_type , disc_type_u, gpu_memory_to_use= 40, use_mask = True, ignore_zero_frequency = False, n_pcs_to_compute = None):

    basis_size = basis.shape[1] if n_pcs_to_compute is None else n_pcs_to_compute
    basis = basis[:,:basis_size]

    ####
    # TODO: Clean up this batch size computation, move elsewhere with other batch size comp. Also does the algebra make sense?
    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
    batch_size = utils.get_embedding_batch_size(basis, cryos[0].image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )

    logger.info('batch size for covariance computation: ' + str(batch_size))

    covariance = covariance_estimation.compute_projected_covariance(cryos, mean, basis, volume_mask, batch_size,  disc_type, disc_type_u, do_mask_images = use_mask)

    if not np.all(np.isfinite(covariance)):
        n_nan = np.sum(np.isnan(covariance))
        n_inf = np.sum(np.isinf(covariance))
        raise ValueError(
            f"projected covariance has {n_nan} NaN, {n_inf} Inf out of {covariance.size} elements"
        )

    ss, u = np.linalg.eigh(covariance)
    u =  np.fliplr(u)
    s = np.flip(ss)
    u = basis @ u 

    s = np.where(s >0 , s, np.ones_like(s)*jax_config.EPSILON)
 
    return u , s


# Legacy diagnostic functions below — retained for debugging

def knock_out_mean_component_2(u,s, mean, volume_mask, volume_shape, vol_batch_size,ignore_zero_frequency, correct_contrast):
    # This assumes s has been kept around
    # cov == u s u^*
    # Want to compute eigendecomposition of the projection onto complement of mean:
    # (I - qq^*) cov ( I - q q^*)
    
    volume_size = np.prod(volume_shape)
    u_real = linalg.batch_idft3(u, volume_shape, vol_batch_size).real
    u_real_norms = np.linalg.norm(u_real, axis=0)
    u_real_norms = np.where(u_real_norms > 0, u_real_norms, 1.0)
    u_real /= u_real_norms
    # u2_norm = np.linalg.norm(u_real, axis =0)

    # Mask mean
    masked_mean = ( fourier_transform_utils.get_idft3(mean.reshape(volume_shape)) * volume_mask.reshape(volume_shape) ).reshape(-1).real
    masked_mean /= np.linalg.norm(masked_mean)
    
    # Make it orthogonal to mask
    if ignore_zero_frequency:
        # knockout volume_mask direction stuff?
        norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
        # substract component in direction of mask?
        # Apply matrix (I - mask mask.T / \|mask^2\| ) 
        masked_mean -= norm_volume_mask * (norm_volume_mask.T @ masked_mean)

    # Project out the mean (and optionally the mask direction).
    # Apply projections sequentially so both take effect when both flags are True.
    u_m_proj = u_real
    if correct_contrast:
        u_m_proj = u_m_proj - masked_mean[:,None] @  (np.conj(masked_mean).T @ u_m_proj )[None]

    if ignore_zero_frequency:
        u_m_proj = u_m_proj - norm_volume_mask[:,None] @  (np.conj(norm_volume_mask).T @ u_m_proj )[None]

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
    ip = np.where(np.abs(ip) > jax_config.ROOT_EPSILON, ip / np.abs(ip) , np.ones_like(ip))
    new_u *= ip

    # back to Fourier domain
    new_u = linalg.batch_dft3(new_u, volume_shape, vol_batch_size)
    norms = np.linalg.norm(new_u, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    new_u /= norms
    
    return np.array(new_u.astype(u.dtype)), np.array(new_s.astype(s.dtype)**2)


# A lot of implementation of the same things, having to do with taking the real SVD of
# the columns of Sigma_col
# - Third does a randomized SVD in the spatial domain.
# TODO: unsured functions (if they can't be called by pipeline.py) should be cleaned out here. There are too many useless ones
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
    
flip_vec_cpu = jax.jit(flip_vec, static_argnums = (1,))
batch_flip_vec = jax.vmap(flip_vec_cpu, in_axes = (1, None))

def make_symmetric_columns(columns, picked_frequencies, volume_shape):
    freqs = core.vec_indices_to_frequencies(picked_frequencies, volume_shape)
    
    good_idx = freqs[:,0] > 0
    minus_freqs = -freqs
    minus_indices = core.frequencies_to_vec_indices(minus_freqs, volume_shape)
    columns_flipped = batch_flip_vec(columns, volume_shape)
    
    return columns_flipped.T, minus_indices, good_idx

make_symmetric_columns_cpu = jax.jit(make_symmetric_columns, static_argnums = (2,))

def make_symmetric_columns_np(columns, picked_frequencies, volume_shape):
    freqs = np.array(core.vec_indices_to_frequencies(picked_frequencies, volume_shape))

    good_idx = freqs[:,0] > 0
    minus_freqs = -freqs
    minus_indices = np.array(core.frequencies_to_vec_indices(minus_freqs, volume_shape))
    columns_flipped = flip_columns_structured(columns, volume_shape)
    return columns_flipped, minus_indices, good_idx


def flip_columns_structured(columns, volume_shape):
    """Hermitian conjugate flip using structured numpy ops instead of fancy indexing.

    For frequency index (x,y,z), the negated frequency maps to ((N-x)%N, (N-y)%N, (N-z)%N).
    Boundary voxels (where any coordinate is 0) are zeroed because the mapping
    is degenerate there (clipping artifact). This is equivalent to the old
    batch_flip_vec2 but uses np.flip (a zero-copy view) instead of random-access
    fancy indexing (columns[mapped_idx,:]), giving better cache behavior and
    lower peak memory on large arrays.
    """
    vol = columns.reshape(*volume_shape, -1)
    result = np.zeros_like(vol)
    result[1:, 1:, 1:] = np.conj(np.flip(vol[1:, 1:, 1:], axis=(0, 1, 2)))
    return result.reshape(columns.shape[0], -1)


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
## TODO: which veriosn is used? Should it be the other? Figure it out. Dlete the other ones
@nvtx.annotate("right_matvec_with_spatial_Sigma", color="orange", domain=NVTX_DOMAIN_PCA)
def right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time = time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]
    minus_frequency_indices = minus_frequency_indices[good_idx]
    logger.info("make big mat 1 %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)

    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)

    # F_2r^* test_mat
    F_t = linalg.batch_dft3(test_mat, smaller_vol_shape, vol_batch_size) / smaller_vol_size
    logger.info("DFT time 1 %s", time.time() - st_time)


    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
    utils.report_memory_device(logger=logger)

    C_F_t = linalg.blockwise_A_X(columns, F_t[original_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use)
    logger.info("AX 1: %s", time.time() - st_time)

    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices, volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))

    F_t2 = F_t[flipped_frequencies_indices_in_smaller,:].copy()
    del F_t  # Free DFT of test_mat — no longer needed
    C_F_t_2 = linalg.blockwise_A_X(columns_flipped, F_t2, memory_to_use = memory_to_use)
    del F_t2, columns_flipped  # Free before accumulation
    C_F_t += C_F_t_2
    del C_F_t_2  # Free — result accumulated into C_F_t
    logger.info("AX 2: %s", time.time() - st_time)


    F_C_F_t = linalg.batch_idft3(C_F_t, volume_shape, vol_batch_size)
    del C_F_t  # Free — IDFT result replaces it
    logger.info("IDFT: %s", time.time() - st_time)

    return F_C_F_t

def right_matvec_with_spatial_Sigma_v2(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time = time.time()
    # Some precompute

    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]

    logger.info("make big mat 1 %s", time.time() - st_time)
    # columns_flipped = np.array(columns_flipped)
    utils.report_memory_device(logger=logger)

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # F_2r^* test_mat
    F_t = linalg.batch_dft3(test_mat, smaller_vol_shape, vol_batch_size) / smaller_vol_size
    logger.info("DFT time 1 %s", time.time() - st_time)

    
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
    utils.report_memory_device(logger=logger)

    C_F_t = linalg.blockwise_A_X(columns, F_t[original_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use)
    logger.info("AX 1: %s", time.time() - st_time)

    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))
    logger.info("C_F_t shape: %s", C_F_t.shape)
    C_F_t += linalg.blockwise_A_X(columns_flipped , F_t[flipped_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use) 
    # C_F_t += C_F_t_2 
    logger.info("AX 2: %s", time.time() - st_time)

    # Store FT in same array
    C_F_t = linalg.batch_idft3(C_F_t, volume_shape, vol_batch_size)
    logger.info("IDFT 1: %s", time.time() - st_time)

    return C_F_t


@nvtx.annotate("left_matvec_with_spatial_Sigma", color="yellow", domain=NVTX_DOMAIN_PCA)
def left_matvec_with_spatial_Sigma(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time =time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]
    minus_frequency_indices = minus_frequency_indices[good_idx]

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)

    # Compute smallest grid that contains all picked frequencies
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)

    # Now do compute:
    # F = IDFT here
    # so F^* = DFT

    # Q should be real I think?
    Q_F = linalg.batch_dft3( np.conj(Q), volume_shape, vol_batch_size) / np.prod(volume_shape)
    logger.info("DFT: %s", time.time() - st_time)

    # Frequencies in new grid
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)

    Q_F_C = np.zeros((Q.shape[-1], smaller_vol_size), dtype = columns.dtype)
    Q_F_C[:,original_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns, memory_to_use = memory_to_use)
    logger.info("Y^T @ X: %s", time.time() - st_time)

    # Flipped Frequencies in new grid
    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices, volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))

    Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns_flipped, memory_to_use = memory_to_use)
    del Q_F, columns_flipped  # Free — no longer needed after both Y^T @ X calls
    logger.info("Y^T @ X: %s", time.time() - st_time)

    # DFT back
    # X F^* = (F X^*)^*
    Q_F_C_F = np.conj(linalg.batch_idft3(np.conj(Q_F_C).T, smaller_vol_shape, vol_batch_size)).T
    del Q_F_C  # Free — transformed result replaces it
    logger.info("DFT2: %s", time.time() - st_time)

    return Q_F_C_F

use_v2_fn = False
report_memory = True

def left_matvec_with_spatial_Sigma_v2(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    if report_memory:
        utils.report_memory_device(logger=logger)

    st_time =time.time()
    Q_shape_inp = Q.shape
    # Some precompute
    # Compute frequencies and all that stuff...
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]

    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    if report_memory:
        logger.info("after make symmetric columns")
        utils.report_memory_device(logger=logger)

    # Compute smallest grid that contains all picked frequencies
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # Now do compute:
    # F = IDFT here
    # so F^* = DFT
    
    # Q should be real I think?
    # 2x memory of Q
    Q = linalg.batch_dft3( np.conj(Q), volume_shape, vol_batch_size) / np.prod(volume_shape)
    logger.info("DFT: %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)


    # Frequencies in new grid
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
        
    Q_F_C = np.zeros((Q_shape_inp[-1], smaller_vol_size), dtype = columns.dtype)
    Q_F_C[:,original_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q, columns, memory_to_use = memory_to_use)
    logger.info("Y^T @ X: %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    # Flipped Frequencies in new grid
    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape)
    flipped_frequencies_indices_in_smaller = np.array(flipped_frequencies_indices_in_smaller)


    Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q, columns_flipped, memory_to_use = memory_to_use)
    logger.info("Y^T @ X: %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    del columns_flipped

    # DFT back
    # X F^* = (F X^*)^*
    Q_F_C_F = np.conj(linalg.batch_idft3(np.conj(Q_F_C).T, smaller_vol_shape, vol_batch_size)).T
    logger.info("DFT2: %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    return Q_F_C_F


@nvtx.annotate("randomized_real_svd_of_columns", color="red", domain=NVTX_DOMAIN_PCA)
def randomized_real_svd_of_columns(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, test_size = 300, gpu_memory_to_use= 40, ignore_zero_frequency = False):
    st_time = time.time()


    # memory_to_use = utils.get_gpu_memory_total() - 5
    utils.report_memory_device(logger=logger)
    picked_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    smaller_size = int(2 * (np.max(np.abs(picked_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])

    smaller_vol_size = np.prod(smaller_vol_shape)
    test_mat = np.random.randn(smaller_vol_size, test_size).real.astype(np.float32)

    st_time = time.time()
    if use_v2_fn:
        Q = right_matvec_with_spatial_Sigma_v2(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    else:
        Q = right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    del test_mat

    ## Do masking here ?


    logger.info("right matvec %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)
    Q = jax.device_put(Q, device=jax.devices("cpu")[0])
    Q,_ = jnp.linalg.qr(Q)
    Q = np.array(Q)  # Force transfer to host to avoid JAX tracing slowdowns
    logger.info("QR time: %s", time.time() - st_time)

    # In principle, should apply (I - mask mask.T / \|mask\|^2 )  again, but should already be orthogonal
    #
    utils.report_memory_device(logger=logger)
    if use_v2_fn:
        C_F_t_2 = left_matvec_with_spatial_Sigma_v2(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    else:
        C_F_t_2 = left_matvec_with_spatial_Sigma(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    utils.report_memory_device(logger=logger)
    logger.info("left matvec %s", time.time() - st_time)


    U,S,V = np.linalg.svd(C_F_t_2, full_matrices = False)
    logger.info("big SVD %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    vol_size = np.prod(volume_shape)
    # To save some memory... Q = FQ
    Q = linalg.batch_dft3(Q, volume_shape, vol_batch_size)
    Q = linalg.blockwise_A_X(Q, U, memory_to_use = gpu_memory_to_use) / np.float32(np.sqrt(vol_size))
    logger.info("FQU matvec %s", time.time() - st_time)
    utils.report_memory_device(logger=logger)

    volume_size = np.prod(volume_shape)
    # Factors due to IDFT on both sides
    S_fd = S * np.float32(np.sqrt(smaller_vol_size) * np.sqrt(volume_size))
    return np.array(Q), np.array(S_fd), np.array(V)


### Legacy diagnostic functions — not used in the current pipeline


def test_different_embeddings(cryos, volume_mask, mean_estimate, basis, eigenvalues, noise_variance, batch_size, disc_type = 'linear_interp', use_contrast = False, zdims= None):
    
    # noise.get_average_residual_square_v2(cryos[0], volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

    # likelihoods = np.array([cryo.likelihood for cryo in cryos])
    residuals = np.zeros(zdims.size)
    cryo = cryos[0]
    contrast_grid = np.ones([1])
    residuals_flipped = np.zeros(zdims.size)
    for zdim_idx, zdim in enumerate(zdims):
        if zdim > 0:
            basis_coordinates, image_latent_covariances, estimated_contrasts = embedding.get_coords_in_basis_and_contrast_3(cryo, mean_estimate, basis[:,:zdim], eigenvalues[:zdim], volume_mask, noise_variance, contrast_grid, batch_size, disc_type, compute_covariances = False )
        else:
            basis_coordinates = np.zeros((cryo.n_images, 0))
        logger.info("embd ")#, end="")
        contrasts = np.ones(cryo.n_images)
        residuals[zdim_idx] = noise.get_average_residual_square_v3(cryos[0], volume_mask, mean_estimate, basis[:,:zdim].T, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

        basis_flipped = basis[:,:zdim].copy()
        basis_flipped[:,zdim-1] = basis[:,zdim-1] * (np.random.randint(0,2, size = basis[:,zdim-1].shape)*2 - 1)

        if zdim > 0:
            basis_coordinates, image_latent_covariances, estimated_contrasts = embedding.get_coords_in_basis_and_contrast_3(cryo, mean_estimate, basis_flipped, eigenvalues[:zdim], volume_mask, noise_variance, contrast_grid, batch_size, disc_type, compute_covariances = False )
        else:
            basis_coordinates = np.zeros((cryo.n_images, 0))
        logger.info("embd ")#, end="")
        contrasts = np.ones(cryo.n_images)
        residuals_flipped[zdim_idx] = noise.get_average_residual_square_v3(cryos[0], volume_mask, mean_estimate, basis_flipped.T, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')


        logger.info("zdim %s", zdim)#, end="")
    return residuals, residuals_flipped


## TODO: move this elsewhere. This is a failed attempt at trying to predict which zs are the best. It's an interesting one but not used because it doesn't work
def test_different_embeddings_from_volumes(cryos, zs, cov_zs, noise_variance, zdims= None, n_images = 1000, volume_mask= None):
    
    # noise.get_average_residual_square_v2(cryos[0], volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

    # likelihoods = np.array([cryo.likelihood for cryo in cryos])
    from recovar.heterogeneity import latent_density
    metrics = {'locerr_median':np.zeros(zdims.size), 'locerr_90':np.zeros(zdims.size), 'locerr_mean':np.zeros(zdims.size)}

    global_likely_choice1 = np.zeros(zdims.size)
    global_likely_choice2 = np.zeros(zdims.size)
    all_estimators = {}
    all_lhs = {}

    # First pick some points?
    zdim_target = 4
    zs_this = zs[zdim_target]#[:cryo.n_images]
    covs_this = cov_zs[zdim_target]#[:cryo.n_images]

    image_likelihoods = latent_density.compute_latent_space_density_at_pts(zs_this,zs_this,  covs_this)
    most_likelihood_idx = np.argmax(image_likelihoods)

    percentages = np.array([ 0, 0.1 ])#, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    index = np.round(percentages * zs_this.shape[0] ).astype(int)
    likelihoods_sorted = np.flip(np.argsort(image_likelihoods))
    images_chosen_index = likelihoods_sorted[index]




    for zdim_idx, zdim in enumerate(zdims):
        zs_this = zs[zdim]#[:cryo.n_images]
        covs_this = cov_zs[zdim]#[:cryo.n_images]
        zs_split = [zs_this[:cryos[0].n_images], zs_this[cryos[0].n_images:]]
        cov_zs_split = [covs_this[:cryos[0].n_images], covs_this[cryos[0].n_images:]]

        for most_likelihood_idx in images_chosen_index:
            best_zs = zs_this[most_likelihood_idx]
            lhs, rhs, estimators = {}, {}, {}
            all_estimators[zdim] = {}
            all_lhs[zdim] = {}
            real_estimators = {}


            for cryo_idx, cryo in enumerate(cryos):
                # Compute likelihood to best_zs
                # Take n_images most likely image for this point
                best_likelihood = latent_density.compute_latent_quadratic_forms_in_batch(best_zs[None], zs_split[cryo_idx], cov_zs_split[cryo_idx]).reshape(-1)
                single_bin = np.array(np.sort(best_likelihood)[n_images-1])[None]
                inds = np.digitize(best_likelihood, single_bin, right = True)
                images_used = inds == 0 

                # Compute two estimators
                from recovar.heterogeneity import adaptive_kernel_discretization
                estimators[cryo_idx], lhs[cryo_idx], rhs[cryo_idx] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[cryo_idx], noise_variance.astype(np.float32), None, best_likelihood, single_bin, tau= None, grid_correct=False, use_spherical_mask=False, return_lhs_rhs=True)
                lhs[cryo_idx] = adaptive_kernel_discretization.half_volume_to_full_volume(lhs[cryo_idx][0], cryos[cryo_idx].volume_shape)
                lhs[cryo_idx] = (lhs[cryo_idx] * cryos[0].get_valid_frequency_indices())
                real_estimators[cryo_idx] = fourier_transform_utils.get_idft3(estimators[cryo_idx].reshape(cryos[0].volume_shape))
                all_estimators[zdim][cryo_idx] = estimators[cryo_idx]
                all_lhs[zdim][cryo_idx] = lhs[cryo_idx]


            weighting1 = 1/(1/lhs[0] + 1/lhs[1])
            global_likely_choice1[zdim_idx] += jnp.linalg.norm(np.sqrt(weighting1) * (estimators[0] - estimators[1]) )**2
            from recovar.heterogeneity import locres
            weighting1 = weighting1.reshape(cryos[0].volume_shape)
            loc_error = locres.expensive_local_error_with_cov(real_estimators[0], real_estimators[1], cryos[0].voxel_size, weighting1.reshape(real_estimators[0].shape), locres_sampling = 25, locres_maskrad= 25, locres_edgwidth= 0)
            volume_mask = volume_mask > 1e-3
            good_errors = loc_error.reshape(-1)[volume_mask.reshape(-1)]
            metrics['locerr_median'][zdim_idx] += np.median(good_errors)
            metrics['locerr_90'][zdim_idx] += np.percentile(good_errors, 90)
            metrics['locerr_mean'][zdim_idx] += np.mean(good_errors)


        logger.info("zdim %s", zdim)#, end="")
    return global_likely_choice1, metrics, all_estimators, all_lhs

## TODO Similar to function above
def test_different_embeddings_from_variance(cryos, zs, cov_zs, noise_variance, zdims= None, n_images = 1000, tau = None):
    
    from recovar.heterogeneity import latent_density
    zdims = np.asarray(zdims)
    metrics = {'variance':np.zeros(zdims.size), 'filt_var': np.zeros(zdims.size) , 'var_avg': {}, 'variance_lp': np.zeros(zdims.size) }
    for k in range(zdims.size):
        metrics['var_avg'][k] = 0

    all_estimators = {}
    all_lhs = {}


    images_chosen_index = np.random.randint(0, zs[zdims[0]].shape[0], 30)
    first = True
    for zdim_idx, zdim in enumerate(zdims):
        zs_this = zs[zdim]#[:cryo.n_images]
        covs_this = cov_zs[zdim]#[:cryo.n_images]
        zs_split = [zs_this[:cryos[0].n_images], zs_this[cryos[0].n_images:]]
        cov_zs_split = [covs_this[:cryos[0].n_images], covs_this[cryos[0].n_images:]]

        for most_likelihood_idx in images_chosen_index:
            best_zs = zs_this[most_likelihood_idx]
            lhs, rhs, estimators = {}, {}, {}
            all_estimators[zdim] = {}
            all_lhs[zdim] = {}
            real_estimators = {}


            for cryo_idx, cryo in enumerate(cryos):
                # Compute likelihood to best_zs
                # Take n_images most likely image for this point
                best_likelihood = latent_density.compute_latent_quadratic_forms_in_batch(best_zs[None], zs_split[cryo_idx], cov_zs_split[cryo_idx]).reshape(-1)
                single_bin = np.array(np.sort(best_likelihood)[n_images-1])[None]
                inds = np.digitize(best_likelihood, single_bin, right = True)
                images_used = inds == 0 
                import matplotlib.pyplot as plt

                # Compute two estimators
                from recovar import utils
                from recovar.reconstruction import relion_functions, regularization
                from recovar.heterogeneity import adaptive_kernel_discretization
                estimators[cryo_idx], lhs[cryo_idx], rhs[cryo_idx] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[cryo_idx], noise_variance.astype(np.float32), None, best_likelihood, single_bin, tau= tau, grid_correct=True, use_spherical_mask=True, return_lhs_rhs=True)

                image_subset = np.argwhere(images_used).reshape(-1)
                Ft_ctf, Ft_y = relion_functions.residual_relion_style_triangular_kernel(cryos[cryo_idx], estimators[cryo_idx][0], noise_variance.astype(np.float32) * 0 ,  100,  index_subset = image_subset)
                Ft_y = (Ft_y * cryos[0].get_valid_frequency_indices())

                if first:
                    Ft_ctf_avg = regularization.average_over_shells(Ft_ctf, cryos[0].volume_shape) / 10
                    Ft_avg_radial = utils.make_radial_image(Ft_ctf_avg, cryos[0].volume_shape)
                    first = False

                variance_estimate = Ft_y / (Ft_ctf + Ft_avg_radial)
                avg_variance = regularization.average_over_shells(variance_estimate, cryos[0].volume_shape)
                plt.plot(avg_variance)


            metrics['var_avg'][zdim_idx] = avg_variance
            metrics['filt_var'][zdim_idx] += np.sum(Ft_y)
            metrics['variance'][zdim_idx] += np.sum(variance_estimate)
            metrics['variance_lp'][zdim_idx] += np.sum(variance_estimate * cryos[0].get_valid_frequency_indices(rad = 20))


        logger.info("zdim %s", zdim)#, end="")
    return metrics, all_estimators, all_lhs

