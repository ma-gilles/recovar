import functools
import logging
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from recovar import core, utils, jax_config
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.reconstruction import relion_functions, noise
from recovar.heterogeneity import covariance_estimation, principal_components
from recovar.core.configs import ForwardModelConfig
from .core import batch_vol_slice_volume
from recovar.heterogeneity.principal_components import get_cov_svds, pca_by_projected_covariance
from recovar.heterogeneity.covariance_estimation import compute_both_H_B, compute_covariance_regularization_relion_style

logger = logging.getLogger(__name__)


'''
This function is used to estimate principal components during EM. It is not relevant for homogeneous EM.
'''

def compute_UPLambdainvPU(u_projections, CTF, noise_variance):
    ## Note there is not Lambda^{-1} here.

    # Form H
    u_projections = u_projections.swapaxes(1,2)
    u_outer_projections = u_projections[...,None] @ jnp.conj(u_projections[...,None,:])
    u_outer_projections = u_outer_projections.reshape(*u_outer_projections.shape[:-1], -1)
    # Now mat vec with CTFs and whatnot
    CTF_squared = CTF**2 / noise_variance

    u_outer_projections = u_outer_projections.transpose(0,2,3,1)
    
    # Now matvec
    u_outer_projections =  (u_outer_projections @ CTF_squared.T).real

    # Now reshape back
    H = u_outer_projections.transpose(0, 3, 1, 2)
    return H

def compute_little_H_b(mean_projections, u_projections, s, batch, translations, CTF_params, ctf, noise_variance, voxel_size, image_shape, process_images):

    # u_projections is n_rotations x n_principal_components  x image_size

    # Often we would n_principal_components ~ 10 
    # n_rotations depends how much we can allocate at once, but hopefully 100s
    n_principal_components = u_projections.shape[1]
    n_rotations = u_projections.shape[0]
    n_images = batch.shape[0]
    n_translations = translations.shape[0]

    CTF = ctf(CTF_params, image_shape, voxel_size)
    H = compute_UPLambdainvPU(u_projections, CTF, noise_variance)
    H += jnp.diag(1/s)

    batch = process_images(batch, apply_image_mask = False)
    batch *= ctf( CTF_params, image_shape, voxel_size) / noise_variance

    b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, noise_variance, image_shape)

    return H, b

batch_batch_diag = jax.vmap(jax.vmap(jnp.diag, in_axes = 0, out_axes = 0), in_axes = 0, out_axes = 0)

@functools.partial(jax.jit, static_argnums=[6,9,10])
def compute_bHb_terms(mean_projections, u_projections, s, batch, translations, CTF_params, ctf, noise_variance, voxel_size, image_shape, process_images):

    H,b = compute_little_H_b(mean_projections, u_projections, s, batch, translations, CTF_params, ctf, noise_variance, voxel_size, image_shape, process_images)

    # Compute bHb
    # Hinvb = jax.scipy.linalg.solve(H, b, assume_a = 'pos')
    # 
    H_chol, low = jax.scipy.linalg.cho_factor(H, lower = True)
    Hinvb = jax.scipy.linalg.cho_solve((H_chol, low), b, overwrite_b=False, check_finite=True)

    bHinvb = jnp.sum(jnp.conj(b) * Hinvb, axis =-2)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(batch_batch_diag(H_chol))), axis = -1)
    half_inv_logdet = - 0.5 * log_det * 2
    log_det_H = half_inv_logdet
    summed = bHinvb + log_det_H[...,None]
    return summed.transpose(1,0,2)

def compute_bLambdainvPU_terms(mean_projections, u_projections, invnoise_CTFed_images, translations, CTF, noise_variance, image_shape):

    n_principal_components = u_projections.shape[1]
    n_rotations = u_projections.shape[0]
    n_images = invnoise_CTFed_images.shape[0]
    n_translations = translations.shape[0]
    n_shifted_images = n_images * n_translations

    u_projections = u_projections.swapaxes(1,2)

    shifted_images = core.batch_trans_translate_images(invnoise_CTFed_images, jnp.repeat(translations[None], invnoise_CTFed_images.shape[0], axis=0), image_shape)
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])
    b1 = (jnp.conj(shifted_images) @ u_projections).real
    b1 = b1.reshape(n_rotations, n_images, n_translations, n_principal_components)
    b1 = b1.transpose(0,1,3,2)

    u_projections_times_mu_projection = u_projections * jnp.conj(mean_projections[...,None])
    b2 = ((CTF**2 / noise_variance) @ u_projections_times_mu_projection).real
    b = 2 * (-b1 + b2[...,None])
    b = -0.5 * b
    return b


# ============================================================================
# Equinox-based EM heterogeneity API
# ============================================================================

@eqx.filter_jit
def compute_bHb_terms_eqx(config: ForwardModelConfig, mean_projections, u_projections, s, batch, translations, ctf_params, noise_variance):
    """Equinox version of compute_bHb_terms (11 → 8 params)."""
    H, b = compute_little_H_b(
        mean_projections, u_projections, s, batch, translations,
        ctf_params, config.ctf, noise_variance, config.voxel_size,
        config.image_shape, config.process_fn,
    )
    H_chol, low = jax.scipy.linalg.cho_factor(H, lower=True)
    Hinvb = jax.scipy.linalg.cho_solve((H_chol, low), b, overwrite_b=False, check_finite=True)
    bHinvb = jnp.sum(jnp.conj(b) * Hinvb, axis=-2)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(batch_batch_diag(H_chol))), axis=-1)
    half_inv_logdet = -0.5 * log_det * 2
    log_det_H = half_inv_logdet
    logger.warning("Make sure this is correct...")
    summed = bHinvb + log_det_H[...,None]
    return summed.transpose(1,0,2)


@eqx.filter_jit
def sum_up_images_fixed_rots_covariance_precompute_eqx(config: ForwardModelConfig, batch, translations, ctf_params):
    """Equinox version of sum_up_images_fixed_rots_covariance_precompute (7 → 4 params)."""
    CTF = config.compute_ctf(ctf_params)
    batch = config.process_fn(batch, apply_image_mask=False) * CTF
    shifted_CTFed_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), config.image_shape)
    return shifted_CTFed_images, CTF


@eqx.filter_jit
def sum_up_images_fixed_rots_covariance_with_precompute_eqx(config: ForwardModelConfig, shifted_CTFed_images, mean_projections, CTF, gridpoints, probabilities, rotations, noise_variance, gridpoint_target, H=0, B=0, right_kernel_width=2, right_kernel="triangular"):
    """Equinox version of sum_up_images_fixed_rots_covariance_with_precompute (14 → 13 params)."""
    n_rotations = rotations.shape[0]
    n_translations = shifted_CTFed_images.shape[1]
    n_images = shifted_CTFed_images.shape[0]
    n_shifted_images = n_images * n_translations
    image_size = shifted_CTFed_images.shape[-1]

    from recovar.heterogeneity import covariance_core
    kernel_vals = covariance_core.evaluate_kernel_on_grid(gridpoints, gridpoint_target, kernel=right_kernel, kernel_width=right_kernel_width)

    e2_p1 = shifted_CTFed_images @ kernel_vals.T
    e2_p2 = (CTF**2) @ (kernel_vals * mean_projections).T
    e2 = e2_p1 - e2_p2[:,None,:]
    e2 = e2.swapaxes(1,2)
    e2 = jnp.conj(e2)

    gamma_2 = probabilities * e2
    gamma_2_summed_over_translations = jnp.sum(gamma_2, axis=-1)
    summed_CTF_squared_gamma2 = (CTF**2).T @ gamma_2_summed_over_translations
    summed_CTF_squared_gamma2 = summed_CTF_squared_gamma2.T
    before_adj_B2 = -summed_CTF_squared_gamma2 * mean_projections

    gamma_2 = gamma_2.swapaxes(1,2).reshape(n_images * n_translations, n_rotations)
    shifted_CTFed_images = shifted_CTFed_images.reshape(n_images * n_translations, image_size)
    before_adj_B2 += gamma_2.T @ shifted_CTFed_images

    probabilties_summed_over_translations = jnp.sum(probabilities, axis=-1)
    CTF_squared_times_noise = (CTF**2 * noise_variance).T @ probabilties_summed_over_translations
    noise_piece = CTF_squared_times_noise.T * kernel_vals
    before_adj_B2 -= noise_piece

    before_adj_B2_half = fourier_transform_utils.full_image_to_half_image(before_adj_B2, config.image_shape)
    B = core.adjoint_slice_volume(before_adj_B2_half, rotations, config.image_shape, config.volume_shape, "linear_interp", volume=B, half_image=True)

    CTF_squared = CTF**2
    CTF_squared_kernel_vals = kernel_vals @ CTF_squared.T
    gamma_3 = probabilties_summed_over_translations.T * CTF_squared_kernel_vals
    H_before_adj = gamma_3 @ CTF_squared
    H_before_adj_half = fourier_transform_utils.full_image_to_half_image(H_before_adj, config.image_shape)
    H = core.adjoint_slice_volume(H_before_adj_half, rotations, config.image_shape, config.volume_shape, "linear_interp", volume=H, half_image=True)

    return H, B


@eqx.filter_jit
def reduce_covariance_est_inner_eqx(config: ForwardModelConfig, mean_projections, u_projections, probabilities, batch, translations, ctf_params, noise_variance):
    """Equinox version of reduce_covariance_est_inner (11 → 8 params)."""
    CTF = config.compute_ctf(ctf_params)
    batch = config.process_fn(batch, apply_image_mask=False)
    batch *= config.compute_ctf(ctf_params)

    probabilities = probabilities.swapaxes(0,1)
    b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, jnp.ones_like(noise_variance), config.image_shape)
    b = b.swapaxes(-1,-2)
    b *= jnp.sqrt(probabilities[...,None])
    outer_products = covariance_estimation.summed_outer_products(b.reshape(-1, b.shape[-1]))

    probabilities_summed_over_translations = jnp.sum(probabilities, axis=-1)
    UALambdaAUs = compute_UPLambdainvPU(u_projections, CTF, 1/noise_variance)
    UALambdaAUs = jnp.sum(probabilities_summed_over_translations[...,None,None] * UALambdaAUs, axis=(0,1))

    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(ctf_params.dtype)

    H = compute_UPLambdainvPU(u_projections, CTF, jnp.ones_like(noise_variance))
    H *= jnp.sqrt(probabilities_summed_over_translations[...,None,None])
    H = H.reshape(-1, H.shape[-2], H.shape[-1])
    lhs = jnp.sum(covariance_estimation.batch_kron(H, H), axis=(0))

    return lhs, rhs


# ============================================================================
# Legacy EM heterogeneity API
# ============================================================================


def compute_H_B(experiment_dataset, mean, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, image_indices, mean_disc):
    # Memory in here scales as O (batch_size )

    logger.warning("Not using mask in compute_H_B. Not implemented yet")

    image_shape = experiment_dataset.image_shape
    image_size = experiment_dataset.image_size
    volume_shape = experiment_dataset.volume_shape
    volume_size = experiment_dataset.volume_size
    n_picked_indices = picked_frequency_indices.size
    n_rotations = rotations.shape[0]
    if n_rotations <= 0:
        raise ValueError("compute_H_B requires at least one rotation")
    if translations.shape[0] <= 0:
        raise ValueError("compute_H_B requires at least one translation")

    H = [jnp.zeros(volume_size, dtype = experiment_dataset.dtype_real )] * n_picked_indices
    B = [jnp.zeros(volume_size, dtype = experiment_dataset.dtype )] * n_picked_indices

    gpu_memory = utils.get_gpu_memory_total()
    # *10: slicing is cheap per image, use larger batches for mean projection precomputation
    batch_size = utils.safe_batch_size(utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) * 10)
    n_batches = utils.get_number_of_index_batch(n_rotations, batch_size)

    mean_projections = np.zeros((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
        mean_projections[rot_indices] = core.slice_volume(mean, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, mean_disc)

    picked_freq_coords = core.vec_indices_to_vol_indices(picked_frequency_indices, volume_shape)

    # Divide by translations to account for per-translation memory in inner loop
    batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(mean_projections)) / translations.shape[0])
    logger.info("Starting H_B, batch size %s. Remaining memory %s", batch_size, gpu_memory - utils.get_size_in_gb(mean_projections))
    utils.report_memory_device(logger=logger)
    
    # Allocate this to GPU.
    mean_projections = jnp.asarray(mean_projections)
    rotation_batch = max(1, rotations.shape[0] // 10)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=mean_disc,
        process_fn=experiment_dataset.process_images,
    )

    start_idx =0
    for images, _rotation_matrices, _translations, ctf_params, _noise_variance, _particle_indices, indices in experiment_dataset.iter_batches(
        batch_size,
        indices=image_indices,
        by_image=False,
    ):
        end_idx = start_idx + len(indices)
        prob_batch = jnp.array(probabilities[start_idx:end_idx])
        shifted_CTFed_images, CTF = sum_up_images_fixed_rots_covariance_precompute_eqx(
            config, images, translations, ctf_params,
        )
        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):

            gridpoints = core.batch_get_gridpoint_coords(
                rotations[rot_indices],
                image_shape, volume_shape )
            for (k, picked_freq_coord) in enumerate(picked_freq_coords):
                H[k], B[k] = sum_up_images_fixed_rots_covariance_with_precompute_eqx(
                    config, shifted_CTFed_images, mean_projections[np.array(rot_indices)],
                    CTF, gridpoints, prob_batch[:,rot_indices], rotations[rot_indices],
                    noise_variance, picked_freq_coord, H=H[k], B=B[k],
                    right_kernel_width=2, right_kernel="triangular",
                )

        start_idx = end_idx

    return H, B


@functools.partial(jax.jit, static_argnums=[3,5,6])
def sum_up_images_fixed_rots_covariance_precompute(batch, translations, CTF_params, ctf, voxel_size, image_shape, process_images):

    CTF = ctf(CTF_params, image_shape, voxel_size)
    batch = process_images(batch, apply_image_mask = False) * CTF
    shifted_CTFed_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)

    return shifted_CTFed_images, CTF


@functools.partial(jax.jit, static_argnums=[7,8,12,13])
def sum_up_images_fixed_rots_covariance_with_precompute(shifted_CTFed_images, mean_projections, CTF, gridpoints, probabilities, rotations, noise_variance, image_shape, volume_shape, gridpoint_target, H=0, B=0, right_kernel_width=2, right_kernel="triangular"):

    n_rotations = rotations.shape[0]
    n_translations = shifted_CTFed_images.shape[1]
    n_images = shifted_CTFed_images.shape[0]
    n_shifted_images = n_images * n_translations
    image_size = shifted_CTFed_images.shape[-1]

    from recovar.heterogeneity import covariance_core
    kernel_vals = covariance_core.evaluate_kernel_on_grid(gridpoints, gridpoint_target, kernel=right_kernel, kernel_width=right_kernel_width)

    e2_p1 = shifted_CTFed_images @ kernel_vals.T
    e2_p2 = (CTF**2) @ (kernel_vals * mean_projections).T
    e2 = e2_p1 - e2_p2[:,None,:]
    e2 = e2.swapaxes(1,2)
    e2 = jnp.conj(e2)

    gamma_2 = probabilities * e2
    gamma_2_summed_over_translations = jnp.sum(gamma_2, axis=-1)
    summed_CTF_squared_gamma2 = (CTF**2).T @ gamma_2_summed_over_translations
    summed_CTF_squared_gamma2 = summed_CTF_squared_gamma2.T
    before_adj_B2 = -summed_CTF_squared_gamma2 * mean_projections

    gamma_2 = gamma_2.swapaxes(1,2).reshape(n_images * n_translations, n_rotations)
    shifted_CTFed_images = shifted_CTFed_images.reshape(n_images * n_translations, image_size)
    before_adj_B2 += gamma_2.T @ shifted_CTFed_images

    probabilties_summed_over_translations = jnp.sum(probabilities, axis=-1)
    CTF_squared_times_noise = (CTF**2 * noise_variance).T @ probabilties_summed_over_translations
    noise_piece = CTF_squared_times_noise.T * kernel_vals
    before_adj_B2 -= noise_piece

    before_adj_B2_half = fourier_transform_utils.full_image_to_half_image(before_adj_B2, image_shape)
    B = core.adjoint_slice_volume(before_adj_B2_half, rotations, image_shape, volume_shape, "linear_interp", volume=B, half_image=True)

    CTF_squared = CTF**2
    CTF_squared_kernel_vals = kernel_vals @ CTF_squared.T
    gamma_3 = probabilties_summed_over_translations.T * CTF_squared_kernel_vals
    H_before_adj = gamma_3 @ CTF_squared

    H_before_adj_half = fourier_transform_utils.full_image_to_half_image(H_before_adj, image_shape)
    H = core.adjoint_slice_volume(H_before_adj_half, rotations, image_shape, volume_shape, "linear_interp", volume=H, half_image=True)
    return H, B





from recovar.heterogeneity import covariance_estimation
def compute_projected_covariance(experiment_datasets, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, disc_type_u, image_indices = None):
    
    lhs, rhs = compute_projected_covariance_rhs_lhs(experiment_datasets, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, disc_type_mean, disc_type_u, image_indices = None)
    covar = solve_covariance(lhs, rhs)
    return covar


def compute_projected_covariance_rhs_lhs(experiment_dataset, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, disc_type_mean, disc_type_u, image_indices = None):
    
    # experiment_dataset = experiment_datasets[0]

    basis = jnp.asarray(basis.T, dtype=experiment_dataset.dtype)
    mean = jnp.asarray(mean, dtype=experiment_dataset.dtype)

    lhs = 0
    rhs = 0

    if disc_type_mean == 'cubic':
        from recovar.core import cubic_interpolation
        mean = cubic_interpolation.calculate_spline_coefficients(mean.reshape(experiment_dataset.volume_shape))

    if disc_type_u == 'cubic':
        basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)
    
    n_rotations = rotations.shape[0]
    if n_rotations <= 0:
        raise ValueError("compute_projected_covariance_rhs_lhs requires at least one rotation")
    if translations.shape[0] <= 0:
        raise ValueError("compute_projected_covariance_rhs_lhs requires at least one translation")
    n_principal_components = basis.shape[0]
    image_size = experiment_dataset.image_size

    batch_size = utils.safe_batch_size(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()))

    u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype = np.complex64)
    # Compute all mean and principal component projections
    mean_projections = np.empty((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size): 
        mean_projections[rot_indices] = core.slice_volume(mean, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_mean)
        u_projections[rot_indices] = batch_vol_slice_volume(basis, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_u)

    
    del basis, mean
    logger.info("done with u_proj %s", batch_size)
    basis_size = u_projections.shape[1]

    rotation_batch = max(1, rotations.shape[0] // 10)

    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  (2*basis_size**4*8/1e9 + utils.get_size_in_gb(mean_projections[:rotation_batch])* ( 1 + basis_size**2) )
    # Divide by translations to account for per-translation memory in inner loop
    batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, memory_left_over_after_kron_allocate) / translations.shape[0])

    logger.info('batch size for projected covariance computation: %s', batch_size)

    rotation_batch = max(1, rotations.shape[0] // 10)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type_mean,
        process_fn=experiment_dataset.process_images,
    )

    start_idx = 0
    for images, _rotation_matrices, _translations, ctf_params, _noise_variance, _particle_indices, batch_image_ind in experiment_dataset.iter_batches(
        batch_size,
        indices=image_indices,
        by_image=False,
    ):

        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):
            end_idx = start_idx + len(batch_image_ind)
            lhs_this, rhs_this = reduce_covariance_est_inner_eqx(
                config, mean_projections[rot_indices], u_projections[rot_indices],
                probabilities[start_idx:end_idx][:,np.array(rot_indices)],
                images, translations,
                ctf_params, noise_variance,
            )
            lhs += lhs_this
            rhs += rhs_this

        del lhs_this, rhs_this
        start_idx = end_idx

    return lhs, rhs

def solve_covariance(lhs, rhs):
    def vec(X):
        return X.T.reshape(-1)

    ## Inverse of vec function.
    def unvec(x):
        n = np.sqrt(x.size).astype(int)
        return x.reshape(n,n).T
    
    logger.info("end of covariance computation - before solve")
    rhs = vec(rhs)

    covar = jax.scipy.linalg.solve(lhs, rhs, assume_a='pos')
    covar = unvec(covar)
    logger.info("end of solve")

    return covar




@functools.partial(jax.jit, static_argnums = [6,9,10])    
def reduce_covariance_est_inner(mean_projections, u_projections, probabilities, batch, translations, CTF_params, ctf, noise_variance, voxel_size, image_shape, process_images):

    CTF = ctf(CTF_params, image_shape, voxel_size)

    batch = process_images(batch, apply_image_mask = False)
    batch *= ctf( CTF_params, image_shape, voxel_size)

    probabilities = probabilities.swapaxes(0,1)

    b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, jnp.ones_like(noise_variance), image_shape)
    b = b.swapaxes(-1,-2)
    b *= jnp.sqrt(probabilities[...,None])
    outer_products = covariance_estimation.summed_outer_products(b.reshape(-1, b.shape[-1]))

    probabilities_summed_over_translations = jnp.sum(probabilities, axis = -1)
    UALambdaAUs = compute_UPLambdainvPU(u_projections, CTF, 1/noise_variance)
    UALambdaAUs = jnp.sum( probabilities_summed_over_translations[...,None,None] * UALambdaAUs, axis=(0,1))

    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(CTF_params.dtype)

    H = compute_UPLambdainvPU(u_projections, CTF, jnp.ones_like(noise_variance))

    H *= jnp.sqrt(probabilities_summed_over_translations[...,None,None])
    H = H.reshape(-1, H.shape[-2], H.shape[-1])
    lhs = jnp.sum(covariance_estimation.batch_kron(H, H), axis=(0))

    return lhs, rhs


def estimate_principal_components_simple(experiment_dataset, mean, mean_signal_variance, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, batch_size, image_indices, disc_type_mean, covariance_options):
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options
    H,B = compute_H_B(experiment_dataset, mean, probabilities, rotations, translations, noise_variance, volume_mask, picked_frequency_indices, image_indices, disc_type_mean)
    H = np.stack(H, axis =1)
    B = np.stack(B, axis =1)
    cov_prior = mean_signal_variance**2     

    # Should change the function get_cov_svds...
    cov = {'est_mask' : B / (H + 0.01 * cov_prior[:,None]) }

    vol_batch_size = 50
    gpu_memory_to_use =  50

    basis,s = principal_components.get_cov_svds(cov, picked_frequency_indices, volume_mask, experiment_dataset.volume_shape, vol_batch_size, gpu_memory_to_use, False, covariance_options['randomized_sketch_size'])
    basis = basis['real']
    # basis_size = basis.shape[-1]
    basis_size = 3
    basis = basis[:,:basis_size]

    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() - 2*basis_size**4*8/1e9
    batch_size = utils.get_embedding_batch_size(basis, experiment_dataset.image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate)
    logger.info('batch size for covariance computation: %s', batch_size)

    covariance = compute_projected_covariance([experiment_dataset], mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, covariance_options['disc_type_u'], image_indices = None)
    ss, u = np.linalg.eigh(covariance)
    u =  np.fliplr(u)
    s = np.flip(ss)
    u = basis @ u 
    s = np.where(s >0 , s, np.ones_like(s)*jax_config.EPSILON)
    return u, s


def estimate_principal_components_halfset(cryos, means, mean_signal_variance, cov_prior, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, batch_size, image_indices, disc_type_mean, covariance_options):
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options

    gpu_memory = utils.report_gpu_memory()
    volume_shape = cryos[0].volume_shape
    Hs, Bs = 2*[None], 2*[None]
    for cryo_idx, cryo in enumerate(cryos):
        Hs[cryo_idx], Bs[cryo_idx] = compute_H_B(cryo, means[cryo_idx], probabilities[cryo_idx], rotations, translations, noise_variance, volume_mask, picked_frequency_indices, image_indices, disc_type_mean)
    
    volume_noise_var = None
    volume_mask = None
    _, covariance_prior, covariance_fscs = principal_components.compute_covariance_regularization_relion_style(Hs, Bs, mean_signal_variance, picked_frequency_indices, volume_noise_var, volume_mask, cryos[0].volume_shape, gpu_memory, reg_init_multiplier = jax_config.REG_INIT_MULTIPLIER, options = covariance_options)

    cov_cols = 2 * [None]
    us = 2 * [None]; ss = 2 * [None]
    for cryo_idx, cryo in enumerate(cryos):
        cov_cols[cryo_idx] = relion_functions.post_process_from_filter_v2(Hs[cryo_idx], Bs[cryo_idx], volume_shape, volume_upsampling_factor = 1, tau = covariance_prior, kernel = covariance_options['left_kernel'], use_spherical_mask = covariance_options['use_spherical_mask'], grid_correct = covariance_options['grid_correct'], gridding_correct = "square", kernel_width = 1, volume_mask = volume_mask )

        vol_batch_size = utils.get_vol_batch_size(cryo.grid_size, gpu_memory)
        orthog_cov_cols,_ = principal_components.get_cov_svds(cov_cols[cryo_idx], picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, gpu_memory, False, covariance_options['randomized_sketch_size'])

        basis_size = cov_cols[cryo_idx].shape[0]
        memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
        batch_size = utils.get_embedding_batch_size(orthog_cov_cols, cryo.image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )
        logger.info('batch size for covariance computation: %s', batch_size)

        covariance = compute_projected_covariance([cryo], means[cryo_idx], orthog_cov_cols, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, covariance_options['disc_type_u'], image_indices = None)
        ss, u = np.linalg.eigh(covariance)
        u =  np.fliplr(u)
        s = np.flip(ss)
        us[cryo_idx] = orthog_cov_cols @ u 
        ss[cryo_idx] = np.where(s >0 , s, np.ones_like(s)*jax_config.EPSILON)


    return us, ss




def estimate_principal_components(cryos, options,  means, mean_signal_variance, cov_noise, volume_mask,
                                dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use,
                                noise_model,  
                                covariance_options = None, variance_estimate = None):
    
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options

    volume_shape = cryos[0].volume_shape
    vol_batch_size = utils.get_vol_batch_size(cryos[0].grid_size, gpu_memory_to_use)

    covariance_cols, picked_frequencies, column_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(cryos, means, mean_signal_variance, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory_to_use, noise_model, covariance_options, picked_frequencies)
    logger.info("memory after covariance estimation")
    utils.report_memory_device(logger=logger)
    

    # First approximation of eigenvalue decomposition
    u,s = get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, options.ignore_zero_frequency, covariance_options['randomized_sketch_size'])

    if not options.keep_intermediate:
        for key in covariance_cols.keys():
            covariance_cols[key] = None
    image_cov_noise = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].image_shape))

    u['rescaled'], s['rescaled'] = pca_by_projected_covariance(cryos, u['real'], means.combined, image_cov_noise, dilated_volume_mask, disc_type = covariance_options['disc_type'], disc_type_u = covariance_options['disc_type_u'], gpu_memory_to_use= gpu_memory_to_use, use_mask = covariance_options['mask_images_in_proj'], ignore_zero_frequency = False, n_pcs_to_compute = covariance_options['n_pcs_to_compute'])

    if not options.keep_intermediate:
        u['real'] = None
            
    return u, s, covariance_cols, picked_frequencies, column_fscs



def compute_regularized_covariance_columns(cryos, means, mean_signal_variance, cov_noise, volume_mask, dilated_volume_mask, gpu_memory, noise_model, options, picked_frequencies):

    volume_shape = cryos[0].volume_shape
    mask_final = volume_mask

    utils.report_memory_device(logger=logger)
    Hs, Bs = compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies,
                               gpu_memory, options=options)
    volume_noise_var = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].volume_shape))
    covariance_cols = {}

    logger.info("using new covariance reg fn")
    utils.report_memory_device(logger=logger)

    covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(
        Hs, Bs, 1/mean_signal_variance, picked_frequencies, volume_noise_var,
        mask_final, volume_shape, gpu_memory,
        reg_init_multiplier=jax_config.REG_INIT_MULTIPLIER, options=options)
    covariance_cols["est_mask"] = covariance_cols["est_mask"].T
    del Hs, Bs
    logger.info("after reg fn")

    utils.report_memory_device(logger=logger)

    return covariance_cols, picked_frequencies, np.asarray(fscs)
