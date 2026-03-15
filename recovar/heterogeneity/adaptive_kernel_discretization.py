"""Adaptive kernel discretization for heterogeneous cryo-EM reconstruction.

Implements the kernel-based volume estimation with cross-validated bandwidth
selection described in the RECOVAR method. Provides both polynomial (nearest-
neighbor) and triangular (linear-interpolation) kernel precomputation, along
with residual-based model selection across discretization parameters.
"""

import logging
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from recovar import core, jax_config, utils
from recovar.reconstruction import noise, regularization, relion_functions
from recovar.data_io import dataset
from recovar.core import linalg
from recovar.core.configs import BatchData, DataIterator, ForwardModelConfig
import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils

logger = logging.getLogger(__name__)


@eqx.filter_jit
def _heterogeneity_kernel_batch_from_fft(
    config: ForwardModelConfig,
    batch: BatchData,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    upsample_ctf: bool = True,
):
    """Backproject half-spectrum images into heterogeneity kernel accumulators.

    Expects ``batch.images`` in half-image (rfft) format.  Accumulates into
    half-volume layout for memory efficiency.

    Parameters
    ----------
    upsample_ctf : bool
        If True, evaluate CTF² on a 2x-upsampled grid and box-filter back to
        native resolution before backprojecting.  Reduces aliasing in the CTF
        weight accumulator.  Default True (matches legacy behaviour).
    """
    from recovar.core.geometry import translate_images
    from recovar.reconstruction import noise as noise_mod

    half_images = translate_images(batch.images, batch.translations, config.image_shape, half_image=True)
    noise_half = noise_mod.to_batched_half_pixel_noise(
        batch.noise_variance, config.image_shape, batch_size=half_images.shape[0]
    )
    half_images = half_images / noise_half

    # Pre-compute CTF once (reused for both image and weight backprojections)
    ctf = config.compute_ctf_half(batch.ctf_params)
    if not config.premultiplied_ctf:
        half_images = half_images * ctf

    # TODO: remove max_r=None once the default max_r clipping is resolved globally.
    # Currently needed because the old code had no sphere clipping and the default
    # max_r=image_shape[0]//2-1 discards the outermost frequency shell.
    Ft_y = core.adjoint_slice_volume(
        half_images, batch.rotation_matrices, config.image_shape, config.volume_shape,
        config.disc_type,
        volume=Ft_y, half_image=True, half_volume=True, max_r=None,
    )

    if upsample_ctf:
        from recovar.core.ctf import compute_antialiased_ctf_squared
        ctf_half = compute_antialiased_ctf_squared(
            config.ctf, batch.ctf_params, config.image_shape, config.voxel_size,
            half_image=True,
        ) / noise_half
    else:
        ctf_half = ctf ** 2 / noise_half

    Ft_ctf = core.adjoint_slice_volume(
        ctf_half, batch.rotation_matrices, config.image_shape, config.volume_shape,
        config.disc_type,
        volume=Ft_ctf, half_image=True, half_volume=True, max_r=None,  # TODO: see above
    )
    return Ft_y, Ft_ctf.real

def make_X_mat(rotation_matrices, volume_shape, image_shape, pol_degree = 0, dtype = np.float32):

    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
    if pol_degree ==0:
        return jnp.ones(grid_point_vec_indices.shape, dtype = dtype )[...,None], grid_point_vec_indices

    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape ).astype(dtype)
    # Discretized grid points
    # This could be done more efficiently
    grid_points_coords_nearest = core.round_to_int(grid_points_coords)
    differences = grid_points_coords - grid_points_coords_nearest

    if pol_degree==1:
        X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences**1], axis = -1)
        return X_mat, grid_point_vec_indices

    differences_squared = linalg.broadcast_outer(differences, differences)
    differences_squared = keep_upper_triangular(differences_squared)
    X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_squared], axis = -1)

    return X_mat, grid_point_vec_indices

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

    target_triu_indices = np.asarray(target_triu_indices).T
    max_triu_indices = np.asarray(max_triu_indices).T

    indices = np.zeros(target_triu_indices.shape[0], dtype = int)
    for k , searchval in enumerate(target_triu_indices):
        indices[k] = int(np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0][0])
    return indices

def find_diagonal_pol_indices(max_pol_degree):
    max_num_pol_params = get_feature_size(max_pol_degree)
    max_triu_indices = np.triu_indices(max_num_pol_params)
    max_triu_indices = np.asarray(max_triu_indices).T
    diagonal_indices = np.concatenate([np.arange(max_num_pol_params)[...,None],np.arange(max_num_pol_params)[...,None]], axis=-1)
    indices = np.zeros(diagonal_indices.shape[0], dtype = int)
    for k, searchval in enumerate(diagonal_indices):
        indices[k] = int(np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0][0])
    return indices

undo_keep_upper_triangular = jax.vmap(undo_keep_upper_triangular_one, in_axes = (0), out_axes = 0)


def volume_shape_to_half_volume_shape(volume_shape):
    return (volume_shape[0]//2 + 1, volume_shape[1], volume_shape[2] )

def half_volume_shape_to_volume_shape(volume_shape): 
    volume_shape[0] = volume_shape[0] * 2
    return ((volume_shape[0]-1)*2, *volume_shape[1:])

@functools.partial(jax.jit, static_argnums = [1,2])
def vec_index_to_half_vec_index(indices, volume_shape, flip_positive = False):
    vol_indices_full = core.vec_indices_to_vol_indices(indices, volume_shape)

    grid_size = volume_shape[0]
    negative_frequencies = vol_indices_full[...,0] < (grid_size // 2 + 1) 
    if flip_positive:
        frequencies = core.vol_indices_to_frequencies(vol_indices_full, volume_shape)
        frequencies_flipped = jnp.where(frequencies[...,0:1] > 0, -frequencies , frequencies)
        vol_indices_full_flipped = core.frequencies_to_vol_indices(frequencies_flipped, volume_shape)
        vol_indices_full = vol_indices_full_flipped

    in_bound = core.check_vol_indices_in_bound(vol_indices_full, grid_size)

    vec_indices = core.vol_indices_to_vec_indices(vol_indices_full, volume_shape)
    vec_indices = jnp.where(in_bound, vec_indices, -1*jnp.ones_like(vec_indices))
    return vec_indices, negative_frequencies

# Use the canonical rfft-aware implementations from fourier_transform_utils.
# Legacy duplicates with different packing conventions were removed to prevent
# packing-mismatch bugs (see commit fixing heterogeneity_volume.py:136).
half_volume_to_full_volume = fourier_transform_utils.half_volume_to_full_volume
full_volume_to_half_volume = fourier_transform_utils.full_volume_to_half_volume
batch_half_volume_to_full_volume = jax.vmap(half_volume_to_full_volume, in_axes=(0, None), out_axes=0)
batch_full_volume_to_half_volume = jax.vmap(full_volume_to_half_volume, in_axes=(0, None), out_axes=0)


def half_vec_index_to_vec_index(indices_half, volume_shape):
    # For indices with negative frequencies, return -1 (out of bound, not used)
    vol_indices_half = core.vec_indices_to_vol_indices(indices_half, volume_shape_to_half_volume_shape(volume_shape))
    indices = core.vol_indices_to_vec_indices(vol_indices_half, volume_shape)
    bad_indices = indices_half == -1
    indices = indices.at[bad_indices].set(-1)
    return indices

# ============================================================================
# Equinox-based adaptive kernel API
# ============================================================================


@eqx.filter_jit
def precompute_kernel_batch(
    config: ForwardModelConfig,
    batch_data: BatchData,
    pol_degree: int = 0,
    XWX=None, F=None,
    heterogeneity_distances=None, heterogeneity_bins=None,
):
    """Precompute kernel for one batch — Equinox API.

    Uses nearest-neighbor scatter so images are full-spectrum (not half-image).
    Noise variance from ``batch_data.noise_variance``.
    """
    from recovar.reconstruction import noise as noise_mod
    noise_variance = noise_mod.to_batched_pixel_noise(
        batch_data.noise_variance, config.image_shape,
    )
    CTF = config.compute_ctf(batch_data.ctf_params)
    ctf_over_noise_variance = CTF ** 2 / noise_variance

    X, grid_point_indices = make_X_mat(
        batch_data.rotation_matrices, config.volume_shape, config.image_shape, pol_degree=pol_degree
    )
    grid_point_indices, good_idx = vec_index_to_half_vec_index(
        grid_point_indices, config.volume_shape, flip_positive=True
    )
    X = X * good_idx[..., None]

    half_volume_size = np.prod(volume_shape_to_half_volume_shape(config.volume_shape))

    XWX_b = linalg.broadcast_outer(X * ctf_over_noise_variance[..., None], X)
    XWX_b = keep_upper_triangular(XWX_b)

    if heterogeneity_bins is not None:
        heterogeneity_bins_this = (heterogeneity_distances[..., None] <= heterogeneity_bins)
        n_bins = heterogeneity_bins.size
    else:
        heterogeneity_bins_this = jnp.ones((batch_data.images.shape[0], 1), dtype=np.bool_)
        n_bins = 1

    XWX_b = XWX_b[..., None] * heterogeneity_bins_this[..., None, None, :]
    XWX_b = XWX_b.reshape(XWX_b.shape[:-2] + (-1,))

    XWX = XWX.at[grid_point_indices.reshape(-1)].add(XWX_b.reshape(-1, XWX_b.shape[-1]))

    images = core.translate_images(batch_data.images, batch_data.translations, config.image_shape)
    F_b = X * (images * CTF / noise_variance)[..., None]
    F_b = F_b[..., None] * heterogeneity_bins_this[..., None, None, :]
    F_b = F_b.reshape(F_b.shape[:-2] + (-1,))

    F = F.at[grid_point_indices.reshape(-1)].add(F_b.reshape(-1, F_b.shape[-1]))
    return XWX, F


@eqx.filter_jit
def precompute_triangular_kernel_batch(
    config: ForwardModelConfig,
    batch_data: BatchData,
    pol_degree: int = 0,
    XWX=None, F=None,
    heterogeneity_distances=None, heterogeneity_bins=None,
):
    """Precompute triangular kernel for one batch — Equinox API.

    Expects ``batch_data.images`` in half-image (rfft) format and
    ``batch_data.noise_variance`` from ``noise.get()`` (full-spectrum OK,
    auto-converted to half-pixel noise internally).
    """
    from recovar.reconstruction import noise as noise_mod

    if pol_degree > 0:
        raise NotImplementedError

    if heterogeneity_bins is not None:
        heterogeneity_bins_this = (heterogeneity_distances[..., None] <= heterogeneity_bins)
        n_bins = heterogeneity_bins.size
    else:
        heterogeneity_bins_this = jnp.ones((batch_data.images.shape[0], 1), dtype=np.bool_)
        n_bins = 1

    half_images = core.translate_images(batch_data.images, batch_data.translations, config.image_shape, half_image=True)
    noise_half = noise_mod.to_batched_half_pixel_noise(
        batch_data.noise_variance, config.image_shape, batch_size=half_images.shape[0]
    )
    images = half_images / noise_half
    images = images[..., None] * heterogeneity_bins_this[..., None, :]

    config_li = config.replace(disc_type='linear_interp')
    Ft_y = batch_im_adjoint_forward(
        config_li, images, batch_data.ctf_params, batch_data.rotation_matrices,
        half_image=True, half_volume=True,
    )

    CTF = config.compute_ctf_half(batch_data.ctf_params) / noise_half
    CTF = CTF[..., None] * heterogeneity_bins_this[..., None, :]

    Ft_ctf = batch_im_adjoint_forward(
        config_li, CTF, batch_data.ctf_params, batch_data.rotation_matrices,
        half_image=True, half_volume=True,
    )

    return (
        XWX + Ft_ctf.T,
        F + Ft_y.T,
    )


@eqx.filter_jit
def compute_residuals_batch(
    config: ForwardModelConfig,
    batch_data: BatchData,
    weights: jax.Array,
    pol_degree: int = 0,
    use_linear_interp: bool = False,
):
    """Compute residuals for many weights — Equinox API."""
    X_mat, gridpoint_indices = make_X_mat(
        batch_data.rotation_matrices, config.volume_shape, config.image_shape, pol_degree=pol_degree
    )
    weights_on_grid = weights[gridpoint_indices]
    if use_linear_interp:
        X_mat = jnp.repeat(X_mat[..., None, :], axis=-2, repeats=weights_on_grid.shape[-2])
        predicted_phi = linalg.broadcast_dot(X_mat, weights_on_grid)
    else:
        predicted_phi = weights_on_grid[..., 0]

    CTF = config.compute_ctf(batch_data.ctf_params)
    translated_images = core.translate_images(batch_data.images, batch_data.translations, config.image_shape)
    residuals = jnp.abs(translated_images[..., None] - predicted_phi * CTF[..., None]) ** 2

    volume_size = np.prod(config.volume_shape)
    summed_residuals = jnp.zeros((volume_size, residuals.shape[-1]), dtype=residuals.dtype).at[gridpoint_indices.reshape(-1)].add(residuals.reshape(-1, residuals.shape[-1]))
    summed_n = jnp.zeros(volume_size, dtype=residuals.real.dtype).at[gridpoint_indices.reshape(-1)].add(jnp.ones_like(residuals[..., 0]).reshape(-1))
    return summed_residuals, summed_n


def precompute_triangular_kernel(experiment_dataset, noise_variance, pol_degree=0, heterogeneity_distances = None, heterogeneity_bins = None):
    pol_degree = int(pol_degree)  # ensure Python int (numpy scalars are traced by eqx.filter_jit)
    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size

    half_volume_size = np.prod(volume_shape_to_half_volume_shape((experiment_dataset.grid_size,)*3))
    XWX = jnp.zeros((half_volume_size, small_gram_matrix_size(pol_degree) * n_bins ), dtype = np.float32)
    F = jnp.zeros((half_volume_size, get_feature_size(pol_degree) *  n_bins), dtype = np.complex64)

    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - 3 * ( utils.get_size_in_gb(XWX)  + utils.get_size_in_gb(F))  ) )

    logger.info("batch size in precompute kernel: %s", batch_size)

    config = ForwardModelConfig(
        image_shape=tuple(experiment_dataset.image_shape),
        volume_shape=tuple((experiment_dataset.grid_size,)*3),
        grid_size=int(experiment_dataset.grid_size),
        voxel_size=float(experiment_dataset.voxel_size),
        padding=int(experiment_dataset.padding),
        disc_type='',
        ctf=experiment_dataset.ctf_evaluator,
        premultiplied_ctf=False,
        volume_mask_threshold=float(experiment_dataset.volume_mask_threshold),
    )

    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        noise_model=experiment_dataset.noise, noise_half=False,
        apply_process_images=True, half_images=True,
    ):
        XWX, F = precompute_triangular_kernel_batch(
            config, batch_data,
            pol_degree=pol_degree, XWX=XWX, F=F,
            heterogeneity_distances=None if heterogeneity_distances is None else heterogeneity_distances[batch_data.image_indices],
            heterogeneity_bins=heterogeneity_bins,
        )

    XWX = XWX.reshape(-1, small_gram_matrix_size(pol_degree), n_bins)
    F = F.reshape(-1, get_feature_size(pol_degree), n_bins)
    logger.info("Done with precompute of kernel")
    return np.asarray(XWX), np.asarray(F)


def batch_im_adjoint_forward(config, slices, ctf_params, rotation_matrices,
                             half_image=False, half_volume=False):
    """Adjoint forward model vmapped over last axis of slices.

    Returns shape (n_bins, volume_size) — batch axis first.
    """
    return jax.vmap(
        lambda s: core_forward.adjoint_forward_model(
            config, s, ctf_params, rotation_matrices,
            half_image=half_image, half_volume=half_volume,
        ),
        in_axes=-1,
    )(slices)


def get_differences_zero(pol_degree, differences):
    if pol_degree ==0:
        differences_zero = jnp.zeros_like(differences[...,0:1])
    elif pol_degree==1:
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences], axis = -1)
    elif pol_degree==2:
        differences_squared = linalg.broadcast_outer(differences, differences)
        differences_squared = keep_upper_triangular(differences_squared)
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences, differences_squared], axis = -1)
    else:
        raise NotImplementedError
    return differences_zero

@functools.partial(jax.jit, static_argnums = [2,5,6])
def compute_summed_XWX_F(XWX, F, max_grid_dist, grid_distances, frequencies, volume_shape, pol_degree, extra_dimensions = None):

    # Might have to cast this back to frequencies vs indices frequencies
    near_frequencies = core.find_frequencies_within_grid_dist(frequencies, max_grid_dist)
    differences =  near_frequencies - frequencies[...,None,:]
    # This is just storing the same array many times over...
    differences_zero = get_differences_zero(pol_degree, differences)
    # L_inf norm distances
    distances = jnp.max(jnp.abs(differences), axis = -1)
    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

    # Include distance threshold in addition to bounds check
    valid_points = (distances <= (grid_distances[...,None] + 0.5) )* core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

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
    return XWX_summed_neighbor, F_summed_neighbor


# Should this all just be vmapped instead of vmapping each piece? Not really sure.
# It allocate XWX a bunch of time if I do?
@functools.partial(jax.jit, static_argnums = [2,5,6,8, 9])
def compute_estimate_from_precompute_one(XWX, F, max_grid_dist, grid_distances, frequencies_vol_indices, volume_shape, pol_degree, prior_inverse_covariance, prior_option, return_XWX_F):
    XWX_summed_neighbor,F_summed_neighbor = compute_summed_XWX_F(XWX, F, max_grid_dist, grid_distances, frequencies_vol_indices, volume_shape, pol_degree)
    y_and_deriv, good_v, problems = compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor, F_summed_neighbor, frequencies_vol_indices, prior_inverse_covariance, prior_option, volume_shape)
    if return_XWX_F:
        return y_and_deriv, good_v, problems, keep_upper_triangular(XWX_summed_neighbor), F_summed_neighbor
    else:
        return y_and_deriv, good_v, problems, None, None

@functools.partial(jax.jit, static_argnums = [4,5,6])
def compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor, F_summed_neighbor, frequencies_vol_indices, prior_inverse_covariance, prior_option, volume_shape, XWX_in_flat = False):
    if XWX_in_flat:
        XWX_summed_neighbor = undo_keep_upper_triangular(XWX_summed_neighbor)
    vec_indices = core.vol_indices_to_vec_indices(frequencies_vol_indices.astype(int), volume_shape)
    prior_inverse_covariance = get_prior_from_options(prior_inverse_covariance, prior_option, vec_indices, volume_shape )
    y_and_deriv, good_v, problems = batch_solve_for_m(XWX_summed_neighbor,F_summed_neighbor, prior_inverse_covariance )
    return y_and_deriv, good_v, problems


def compute_estimate_from_XWX_F_summed(XWX_summed_neighbor, F_summed_neighbor, prior_inverse_covariance, prior_option, volume_shape):

    memory_per_pixel =  800 * utils.get_size_in_gb(XWX_summed_neighbor[0]) * 10
    volume_size = np.prod(volume_shape)
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    logger.debug("mem before anything: %s", utils.get_gpu_memory_used())
    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX_summed_neighbor) - utils.get_size_in_gb(F_summed_neighbor)  )/ (memory_per_pixel )  ) 
    n_batches = np.ceil(half_volume_size / batch_size).astype(int)

    frequencies_vol_indices = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    logger.info("compute_estimate_from_XWX_F_summed with prior option=%s batch size: %s", prior_option, batch_size)

    prior_inverse_covariance = jnp.asarray(prior_inverse_covariance).real.astype(np.float32)

    reconstruction = np.zeros((half_volume_size, F_summed_neighbor.shape[-1]), dtype = np.complex64)
    good_pixels = np.zeros((half_volume_size), dtype=np.bool_)

    logger.info("dtype = %s", prior_inverse_covariance.dtype)

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(half_volume_size, batch_size, k)

        reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems = compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor[ind_st:ind_end], F_summed_neighbor[ind_st:ind_end], frequencies_vol_indices[ind_st:ind_end], prior_inverse_covariance, prior_option, volume_shape, XWX_in_flat = True)
        logger.debug("batch %d...", k)

    utils.report_memory_device(logger=logger)
    logger.info("Done with compute_estimate_from_XWX_F_summed with prior option=%s", prior_option)

    reconstruction = batch_half_volume_to_full_volume(reconstruction.T, volume_shape).T
    good_pixels = half_volume_to_full_volume(good_pixels, volume_shape)

    return np.asarray(reconstruction), np.asarray(good_pixels)


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
def solve_for_m_simple(XWX, f, regularization):
    XWX += regularization
    v = linalg.batch_hermitian_linear_solver(XWX, f)
    good_v = jnp.linalg.cond(XWX) < 1e4
    problem = ~good_v
    return v, good_v, problem


batch_solve_for_m = jax.vmap(solve_for_m_simple, in_axes = (0,0,0))

# Should this be set by cross validation?
def precompute_kernel(experiment_dataset, noise_variance, pol_degree=0, heterogeneity_distances = None, heterogeneity_bins = None):
    pol_degree = int(pol_degree)  # ensure Python int (numpy scalars are traced by eqx.filter_jit)
    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size

    half_volume_size = np.prod(volume_shape_to_half_volume_shape((experiment_dataset.grid_size,)*3))
    XWX = jnp.zeros((half_volume_size, small_gram_matrix_size(pol_degree) * n_bins ), dtype = np.float32)
    F = jnp.zeros((half_volume_size, get_feature_size(pol_degree) *  n_bins), dtype = np.complex64)

    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - 2* utils.get_size_in_gb(XWX) - 2*utils.get_size_in_gb(F)  ) )

    logger.info("batch size in precompute kernel: %s", batch_size)

    config = ForwardModelConfig(
        image_shape=tuple(experiment_dataset.image_shape),
        volume_shape=tuple((experiment_dataset.grid_size,)*3),
        grid_size=int(experiment_dataset.grid_size),
        voxel_size=float(experiment_dataset.voxel_size),
        padding=int(experiment_dataset.padding),
        disc_type='',
        ctf=experiment_dataset.ctf_evaluator,
        premultiplied_ctf=False,
        volume_mask_threshold=float(experiment_dataset.volume_mask_threshold),
    )

    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        noise_model=experiment_dataset.noise, noise_half=False,
        apply_process_images=True,
    ):
        XWX, F = precompute_kernel_batch(
            config, batch_data,
            pol_degree=pol_degree, XWX=XWX, F=F,
            heterogeneity_distances=None if heterogeneity_distances is None else heterogeneity_distances[batch_data.image_indices],
            heterogeneity_bins=heterogeneity_bins,
        )

    XWX = XWX.reshape(-1, small_gram_matrix_size(pol_degree), n_bins)
    F = F.reshape(-1, get_feature_size(pol_degree), n_bins)
    logger.info("Done with precompute of kernel")
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


def estimate_volume_from_covariance_and_precompute(init_variance, discretization_params, XWXs, Fs, volume_shape):

    logger.info("Starting adaptive disc with params = %s", discretization_params)
    h = discretization_params[1]
    max_pol_degree = discretization_params[0]
    volume_size = np.prod(volume_shape)

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]

    # All of this gets overwritten
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    pol_init_prior_inverse_covariance = 1 / init_variance[...,None]

    first_estimates = [None,None]

    for k in range(2):
        first_estimates[k], _, XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_weights_from_precompute(volume_shape, XWXs[k], Fs[k], pol_init_prior_inverse_covariance, max_pol_degree, max_pol_degree, h = h, return_XWX_F = True, prior_option = "by_degree")

    combined, _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[0] + XWX_summed_neighbors[1], F_summed_neighbors[0] + F_summed_neighbors[1], pol_init_prior_inverse_covariance, "by_degree", volume_shape)
        
    return combined, first_estimates


def estimate_from_relion_style(cryos, discretization_params, XWXs, Fs, volume_shape, tau = None, use_spherical_mask = True, grid_correct = True, gridding_correct = "square" ):


    logger.info("Starting adaptive disc with params = %s", discretization_params)
    h = discretization_params[1]
    pol_degree = discretization_params[0]
    if pol_degree != 0:
        raise NotImplementedError("Only p = 0 supported for now")

    volume_size = np.prod(cryos.volume_shape)
    from recovar.reconstruction import relion_functions

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]
    estimates = [None,None]
    for k in range(2):
        # Probably should rewrite this part
        XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_summed_XWX_F_only(volume_shape, XWXs[k], Fs[k], pol_degree, h)
        XWX_this = half_volume_to_full_volume(XWX_summed_neighbors[k][...,0], volume_shape)
        F_this = half_volume_to_full_volume(F_summed_neighbors[k][...,0], volume_shape)

        estimates[k] = relion_functions.post_process_from_filter(cryos[0], XWX_this, F_this, disc_type = 'nearest', use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = gridding_correct, kernel_width=h+1)

    return estimates


def estimate_multiple_disc_relion_style(experiment_datasets, noise_variance, discretization_params, heterogeneity_distances = None, heterogeneity_bins= None, residual_threshold = None, use_spherical_mask = True, grid_correct = True, gridding_correct = "square" ):
    
    cryos = experiment_datasets
    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    # prior_option = discretization_params[0][2]
    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])
    if max_pol_degree != 0:
        raise NotImplementedError("Only p = 0 supported for now")

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        heterogeneity_distances_this = None if heterogeneity_distances is None else heterogeneity_distances[k]
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances_this, heterogeneity_bins = heterogeneity_bins)    
        # For now just throw away this piece so it doesn't break code.
        if heterogeneity_bins is None:
            XWXs[k] = XWXs[k][...,0]
            Fs[k] = Fs[k][...,0]


    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].grid_size**3


    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size
    first_estimates = np.zeros((n_disc_test, n_bins,  2, experiment_datasets[0].volume_size), dtype = np.complex64)


    for idx, disc_params_this in enumerate(discretization_params):
        for b in range(n_bins):
            first, second = estimate_from_relion_style(cryos, disc_params_this, [XWXs[0][...,b], XWXs[1][...,b] ], [Fs[0][...,b], Fs[1][...,b] ], (cryos[0].grid_size,)*3, tau = None, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = gridding_correct )
            first_estimates[idx,b,0] = first.reshape(-1)
            first_estimates[idx,b,1] = second.reshape(-1)


    # n_dataset (2) x volume_size (N^3) x n_disc_test x n_bins x feature_size  
    first_estimates = first_estimates[...,None]
    first_estimates = first_estimates.transpose(2, 3, 0, 1, 4)
    first_estimates = first_estimates.reshape([*first_estimates.shape[:2], -1, first_estimates.shape[-1]])


    index_array_vol, disc_choices, residuals_averaged, summed_residuals = pick_best_heterogeneity_from_residual(first_estimates[0,...], experiment_datasets[1], heterogeneity_distances[1], heterogeneity_bins, discretization_params, residual_threshold = residual_threshold , min_number_of_images_in_bin = 50)

    opt_halfmaps = [None, None]
    opt_halfmaps[0] = jnp.take_along_axis(first_estimates[0,...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)
    opt_halfmaps[1] = jnp.take_along_axis(first_estimates[1,...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)


    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    
    return first_estimates, np.asarray(opt_halfmaps), np.asarray(disc_choices), np.asarray(residuals_averaged)

def pick_best_heterogeneity_from_residual(estimates, full_test_dataset, heterogeneity_distances, heterogeneity_bins, discretization_params = None, residual_threshold = None , min_number_of_images_in_bin = 50):

    # Probably should separate this stuff?
    discretization_params = [(0,0,"")] if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    if residual_threshold is None:
        logger.warning("didn't specify either residual_threshold or residual_num_images, using first bin with at least %s images", min_number_of_images_in_bin)
        residual_threshold = heterogeneity_bins
        n_in_bins = np.zeros(heterogeneity_bins.size)
        for idx, b in enumerate(heterogeneity_bins):
            n_in_bins[idx] = np.sum(heterogeneity_distances < b)
        good_bins = n_in_bins >= min_number_of_images_in_bin
        bin_chosen = np.argmax(good_bins)
        residual_threshold = heterogeneity_bins[bin_chosen]


    # residuals to pick best one
    from recovar.data_io import dataset

    if residual_threshold is not None:
        good_indices = heterogeneity_distances <= residual_threshold
        test_dataset = dataset.subsample_cryoem_dataset(full_test_dataset, good_indices)

        if test_dataset.n_images <= 0:
            raise ValueError("No images in bin after applying residual threshold")

    logger.info("Number of images used for residual computation: %s", test_dataset.n_images)

    residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, estimates, max_pol_degree, use_linear_interp = True)

    all_params = []
    for param in discretization_params:
        for bin in range(heterogeneity_bins.size):
            all_params.append((*param, bin))
            

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, all_params, test_dataset.volume_shape)

    summed_residuals = jnp.sum(residuals, axis = 0)
    return index_array_vol, disc_choices, residuals_averaged, summed_residuals
    

def estimate_optimal_covariance_and_volume(init_variance, init_prior_covariance_option, discretization_params, XWXs, Fs, volume_shape, reg_iters = 1):

    logger.info("Starting adaptive disc with params = %s", discretization_params)
    h = discretization_params[1]
    max_pol_degree = discretization_params[0]
    volume_size = np.prod(volume_shape)

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]

    # All of this gets overwritten
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    if init_prior_covariance_option == "one_fixed":
        init_prior_inverse_covariance = 1 / (jnp.ones((half_volume_size), dtype = np.float32) * init_variance ) #+ (0 if use_reg else np.inf))
    else:
        init_prior_inverse_covariance = 1 / init_variance

    pol_init_prior_inverse_covariance = np.repeat(init_prior_inverse_covariance[...,None] , max_pol_degree+1, axis=-1)

    for k in range(2):
        _, _, XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_weights_from_precompute(volume_shape, XWXs[k], Fs[k], pol_init_prior_inverse_covariance, max_pol_degree, max_pol_degree, h = h, return_XWX_F = True, prior_option = "by_degree")


    # First estimate signal variance with h = 0, p =0 
    first_estimates = [None,None]
    
    for k in range(2):
        first_estimates[k] = np.array( np.where(np.abs(XWX_summed_neighbors[k][...,0]) < jax_config.ROOT_EPSILON, 0, F_summed_neighbors[k][...,0] / (XWX_summed_neighbors[k][...,0] + init_prior_inverse_covariance ) ))
        first_estimates[k] = half_volume_to_full_volume(first_estimates[k], volume_shape)


    # From this, estimate a first prior with correct decay
    lhs = (XWX_summed_neighbors[0][...,0] + XWX_summed_neighbors[1][...,0]) /2
    lhs = half_volume_to_full_volume(lhs, volume_shape)
    pol_init_prior_inverse_covariance = 1/estimate_signal_variance_from_correlation(first_estimates[0], first_estimates[1], lhs, half_volume_to_full_volume(init_prior_inverse_covariance, volume_shape), volume_shape)
    # Now do a p = input, h = input discretization with diagonal variance
    pol_init_prior_inverse_covariance = np.repeat(pol_init_prior_inverse_covariance[...,None] , discretization_params[0]+1, axis=-1)
    init_prior_inverse_covariance_avg = regularization.batch_average_over_shells(pol_init_prior_inverse_covariance.T, volume_shape,0).T
    init_prior_inverse_covariance_avg = batch_diag(make_regularization_from_reduced(init_prior_inverse_covariance_avg))


    current_prior_covariance_option = "by_degree"
    pol_current_prior_inverse_covariance = pol_init_prior_inverse_covariance.copy()
    local_covar_tmp = []
    pol_current_tmp = []
    estimate_tmp = []
    for reg_iter in range(reg_iters):

        first_estimates = np.zeros((2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
        for k in range(2): 
            first_estimates[k], _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[k], F_summed_neighbors[k], pol_current_prior_inverse_covariance , current_prior_covariance_option, volume_shape)

        estimate_tmp.append(np.array(first_estimates))

        local_covariances, bad_covars = estimate_local_covariances(XWX_summed_neighbors[0], XWX_summed_neighbors[1], first_estimates[0], first_estimates[1], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape, max_pol_degree)

        ##
        local_covar_tmp.append(np.array(local_covariances))
        ##
         
        # Project onto Hermitian positive definite matrices
        local_covariances = 0.5*(local_covariances + jnp.conj(local_covariances.swapaxes(1,2))).real # Assume things are real
        eigs , U = jnp.linalg.eigh(local_covariances)

        # Make eigenvalues not too small so that 
        EPS = 1e-4
        # Threshold away negatives
        eigs = jnp.where(eigs > 0, eigs , 0)

        # If eigenvalues are too small, maybe should bump them all the way to signal variance
        def _safe_inv_eigs(eigs):
            max_abs = jnp.max(jnp.abs(eigs))
            # Guard against all-zero eigenvalues: use 1.0 as fallback denominator
            safe_denom = jnp.where(max_abs > 0, EPS * max_abs, jnp.float32(1.0))
            return jnp.where(eigs > EPS * max_abs, 1 / eigs, 1 / safe_denom)
        get_inv_s = jax.vmap(_safe_inv_eigs)
        s =get_inv_s(eigs)
        
        invert_from_svd = jax.vmap( lambda U, s :  (U * s[None]) @ jnp.conj(U).T  , in_axes = (0,0))
        prior_inverse_covariance = invert_from_svd(U,s)

        # If any are bad, revert to previous one?
        bad_prior_inverse_covariance = jnp.isnan(prior_inverse_covariance).any(axis=(-1, -2))  +  np.isinf(prior_inverse_covariance).any(axis=(-1, -2)) + bad_covars
        logger.debug("bad ones: %s", bad_prior_inverse_covariance.sum())
        # Now do a p = input, h = input discretization with covariance
        pol_current_prior_inverse_covariance = jnp.where(bad_prior_inverse_covariance[...,None,None], init_prior_inverse_covariance_avg, prior_inverse_covariance)
        if jnp.isnan(pol_current_prior_inverse_covariance).any():
            bad_prior_inverse_covariance2 = jnp.isnan(pol_current_prior_inverse_covariance).any(axis=(-1, -2)) #* np.isinf(pol_current_prior_inverse_covariance).any(axis=(-1, -2))
            raise RuntimeError(
                f"NaN in prior_inverse_covariance after fallback. "
                f"bad_count={bad_prior_inverse_covariance2.sum()}"
            )

        pol_current_tmp.append(np.array(pol_current_prior_inverse_covariance))
        current_prior_covariance_option = "complete"

    # Now solve again with new covariances
    final_estimates = np.zeros((2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    for k in range(2): 
        final_estimates[k], _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[k], F_summed_neighbors[k], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape)

    combined, _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[0] + XWX_summed_neighbors[1], F_summed_neighbors[0] + F_summed_neighbors[1], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape)

    return final_estimates, first_estimates, [local_covar_tmp, pol_current_tmp, estimate_tmp, init_prior_inverse_covariance_avg, combined,
    batch_half_volume_to_full_volume(XWX_summed_neighbors[0].T, volume_shape).T, batch_half_volume_to_full_volume(F_summed_neighbors[0].T,volume_shape).T]


def heterogeneous_reconstruction_fixed_variance(experiment_datasets, noise_variance, signal_variance, discretization_params, return_all, heterogeneity_distances, heterogeneity_bins, residual_threshold = None, residual_num_images = None ):

    if residual_threshold is None and residual_num_images is None:
        logger.warning("didn't specify either residual_threshold or residual_num_images, using first bin")
        residual_threshold = heterogeneity_bins[0]

    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])
    if max_pol_degree > 0:
        logger.warning("probably not implemented for pol_degree > 0")

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances[k], heterogeneity_bins = heterogeneity_bins)    
        
    # A crude signal variance estimation
    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
    if noise_variance is None:
        noise_variance, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)

    final_estimates = 0
    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].volume_size
    # Compute weights for each discretization
    n_bins = heterogeneity_bins.size
    final_estimates = np.zeros((n_disc_test, n_bins, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    first_estimates = np.zeros((n_disc_test, n_bins,  2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)

    for idx, disc_params_this in enumerate(discretization_params):
        for b in range(n_bins):
            final_estimates[idx,b], first_estimates[idx,b] = estimate_volume_from_covariance_and_precompute(signal_variance, disc_params_this, [XWXs[0][...,b], XWXs[1][...,b] ], [Fs[0][...,b], Fs[1][...,b] ], (experiment_datasets[0].grid_size,)*3)

    # volume_size (N^3) x n_disc_test x n_bins x feature_size  
    final_estimates = final_estimates.transpose(2, 0, 1, 3)
    final_estimates = final_estimates.reshape(final_estimates.shape[0], final_estimates.shape[1] * final_estimates.shape[2], final_estimates.shape[-1])


    # n_dataset (2) x volume_size (N^3) x n_disc_test x n_bins x feature_size  
    first_estimates = first_estimates.transpose(2, 3, 0, 1, 4)
    first_estimates = first_estimates.reshape([*first_estimates.shape[:2], -1, first_estimates.shape[-1]])


    from recovar.data_io import dataset
    if residual_threshold is not None:
        good_indices = heterogeneity_distances[1] < residual_threshold
        test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)
    else:
        good_indices = np.argsort(heterogeneity_distances[1])[:residual_num_images]
        test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)

    logger.info("Number of images used for residual computation: %s", test_dataset.n_images)
    residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, first_estimates[0], max_pol_degree )
    all_params = []
    for param in discretization_params:
        for bin in heterogeneity_bins:
            all_params.append((*param, bin))

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, all_params, (experiment_datasets[0].grid_size,)*3)

    weights_opt = jnp.take_along_axis(final_estimates[...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(final_estimates), np.asarray(first_estimates), all_params # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)


def naive_heterogeneity_scheme_relion_style(experiment_dataset, noise_variance, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = 100, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp'):
    import gc

    estimates = []
    lhs, rhs = [], []
    og_contrast = experiment_dataset.CTF_params[:,8]
    idx =0 
    for residual_threshold in heterogeneity_bins:
        from recovar.data_io import dataset
        good_indices = heterogeneity_distances <= residual_threshold
        # utils.report_memory_device(logger=logger)

        test_dataset = dataset.subsample_cryoem_dataset(experiment_dataset, good_indices)
        utils.report_memory_device(logger=logger)
        from recovar.reconstruction import relion_functions


        if compute_lhs_rhs:

            Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(test_dataset , noise_variance.astype(np.float32),  batch_size,  disc_type = disc_type, upsampling_factor=1)

            kernel_type = 'triangular' if disc_type == 'linear_interp' else 'square'
            estimate = relion_functions.post_process_from_filter_v2(Ft_ctf, F_ty, test_dataset.volume_shape, 1, tau = tau, kernel = kernel_type, use_spherical_mask = False, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1 )

            lhs.append(np.array(Ft_ctf))
            rhs.append(np.array(F_ty))
        else:
            estimate = relion_functions.relion_reconstruct(test_dataset, noise_variance, batch_size, 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = grid_correct, gridding_correct = "square", tau = tau )

        test_dataset.delete()
        del test_dataset
        gc.collect()

        logger.info("Number of images used in estimator %d: %s", idx, np.sum(good_indices))
        idx+=1
        estimates.append(np.array(estimate.reshape(-1)))

    estimates = np.array(estimates)
    if compute_lhs_rhs:
        return estimates, lhs, rhs

    return estimates


def less_naive_heterogeneity_scheme_relion_style(experiment_dataset, noise_variance, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = 100, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp'):

    estimates = []
    XWX, F = precompute_triangular_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=0, heterogeneity_distances= heterogeneity_distances, heterogeneity_bins = heterogeneity_bins)

    XWX = XWX[:,0,:].T
    F = F[:,0,:].T

    kernel_type = 'triangular' if disc_type == 'linear_interp' else 'square'
    for idx in range(heterogeneity_bins.size):
        from recovar.reconstruction import relion_functions

        estimate = relion_functions.post_process_from_filter_v2(
            XWX[idx],
            F[idx],
            experiment_dataset.volume_shape, 2,
            tau = tau, kernel = kernel_type, use_spherical_mask = True,
            grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1,
            input_half_volume=True,
        )
        estimates.append(np.array(estimate.reshape(-1)))
    estimates = np.array(estimates)

    return estimates

def even_less_naive_heterogeneity_scheme_relion_style(experiment_dataset, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = None, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp', use_spherical_mask = True, return_lhs_rhs = False, heterogeneity_kernel = "parabola", upsampling_factor=None, return_real_space=False):
    bins = heterogeneity_bins
    inds = np.digitize(heterogeneity_distances, bins, right = True).astype(np.int32)
    n_bins = bins.size

    if upsampling_factor is not None:
        upsampled_vol_shape = tuple(3 * [experiment_dataset.grid_size * upsampling_factor])
    else:
        upsampled_vol_shape = tuple((experiment_dataset.grid_size,)*3)
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(upsampled_vol_shape))

    # Accumulate on CPU to avoid OOM from JAX immutable-array copies at large
    # grid sizes (256^3 × 50 bins ≈ 5 GB; .at[].set() would double that).
    # Transferred to GPU only for the final matmul below.
    rhs_all = np.zeros((n_bins, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_bins, half_volume_size), dtype=experiment_dataset.dtype_real)

    # Auto-compute batch size based on GPU memory if not specified
    if batch_size is None:
        accum_gb = utils.get_size_in_gb(rhs_all) + utils.get_size_in_gb(lhs_all)
        avail_gb = max(1.0, utils.get_gpu_memory_total() - accum_gb)
        batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, avail_gb))
    logger.info("batch size in heterogeneity kernel: %s", batch_size)

    if upsampling_factor is not None:
        config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, upsampling_factor=upsampling_factor)
    else:
        config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, upsampling_factor=2)

    image_inds_by_bin = [np.flatnonzero(inds == bin_idx).astype(np.int32) for bin_idx in range(n_bins)]
    # Pre-allocate accumulators once (reused across bins)
    Ft_y_acc = jnp.zeros(half_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf_acc = jnp.zeros(half_volume_size, dtype=experiment_dataset.dtype_real)
    for bin_idx, image_inds in enumerate(image_inds_by_bin):
        if image_inds.size == 0:
            continue

        Ft_y_acc = jnp.zeros_like(Ft_y_acc)
        Ft_ctf_acc = jnp.zeros_like(Ft_ctf_acc)
        # image_inds are derived from per-particle embedding distances, so for
        # tilt series they are particle indices.  Use particle-grouped iteration
        # so that all tilts of a particle are yielded together.
        _use_image_gen = image_inds is None or not getattr(experiment_dataset, 'tilt_series_flag', False)
        for batch_data in DataIterator(
            experiment_dataset, batch_size,
            noise_model=experiment_dataset.noise, noise_half=False,
            apply_process_images=True, half_images=True,
            index_subset=image_inds,
            use_image_generator=_use_image_gen,
        ):
            Ft_y_acc, Ft_ctf_acc = _heterogeneity_kernel_batch_from_fft(
                config, batch_data, Ft_y=Ft_y_acc, Ft_ctf=Ft_ctf_acc,
            )

        rhs_all[bin_idx] = np.asarray(Ft_y_acc)
        lhs_all[bin_idx] = np.asarray(Ft_ctf_acc)

    # A slight improvement is an almost triangular kernel/ pyramid kernel
    #    _
    #  _| |_
    #_|     |_ 
    # or almost epachenikov 
    #
    # heterogeneity_kernel
    if heterogeneity_kernel == "parabola" or heterogeneity_kernel == "triangle":
        distances = bins
        h_grid = 2 * bins

        np_to_use = np
        if heterogeneity_kernel == "triangle":
            kernel_fn = lambda dist : np_to_use.where( np_to_use.abs(dist) < 1, 1 - np_to_use.abs(dist), 0)
        else:
            kernel_fn = lambda dist : np_to_use.where( np_to_use.abs(dist) < 1, 3/4 * (1- dist**2), 0)
        weight_matrix = np_to_use.zeros((n_bins, n_bins)).astype(np.float32)
        weight_matrix[0,0] = 1
        for idx in range(1, n_bins):
            weights = kernel_fn(np_to_use.sqrt(distances/h_grid[idx]))
            weight_matrix[:,idx] = weights
        # Matmul on CPU: (50,50) @ (50, half_vol) — fast and avoids GPU OOM.
        # Downstream post_process_from_filter_v2 transfers each row individually.
        rhs_all = np.asarray(weight_matrix.T.astype(rhs_all.real.dtype) @ rhs_all)
        lhs_all = np.asarray(weight_matrix.T @ lhs_all)


    elif heterogeneity_kernel == "square":
        rhs_all = np.cumsum(rhs_all, axis=0)
        lhs_all = np.cumsum(lhs_all, axis=0)
    else:
        raise NotImplementedError

    kernel_type = 'triangular' if disc_type == 'linear_interp' else 'square'
    vol_upsample = upsampling_factor if upsampling_factor is not None else 1
    estimate_bins = []
    for idx in range(heterogeneity_bins.size):
        estimate = relion_functions.post_process_from_filter_v2(
            lhs_all[idx],
            rhs_all[idx],
            experiment_dataset.volume_shape, vol_upsample,
            tau = tau, kernel = kernel_type,
            use_spherical_mask = use_spherical_mask, grid_correct = grid_correct,
            gridding_correct = "square", kernel_width = 1,
            return_real_space=return_real_space,
            input_half_volume=True,
        )
        estimate_bins.append(estimate.reshape(-1))

    estimates = np.asarray(jnp.stack(estimate_bins, axis=0))
    if return_lhs_rhs:
        return estimates, np.asarray(lhs_all), np.asarray(rhs_all)
    
    return estimates


def compute_lhs_rhs(cryo,noise_variance, heterogeneity_distances, residual_threshold, batch_size  = 100, disc_type = 'linear_interp' ):
    
    good_indices = heterogeneity_distances <= residual_threshold
    test_dataset = dataset.subsample_cryoem_dataset(cryo, good_indices)
    Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(test_dataset , noise_variance.astype(np.float32),  batch_size,  disc_type = disc_type, upsampling_factor=1)
    return Ft_ctf, F_ty


def estimate_variance_and_discretization_params(experiment_datasets, noise_variance, discretization_params, return_all, heterogeneity_distances = None, heterogeneity_bins= None ):
    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances, heterogeneity_bins = heterogeneity_bins)    
        # For now just throw away this piece so it doesn't break code.
        XWXs[k] = XWXs[k][...,0]
        Fs[k] = Fs[k][...,0]

    # A crude signal variance estimation
    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
    if noise_variance is None:
        noise_variance, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)

    final_estimates = 0
    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].grid_size**3
    # Compute weights for each discretization
    final_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    first_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)


    for idx, disc_params_this in enumerate(discretization_params):
        final_estimates[idx], first_estimates[idx], prior_inverse_covariance = estimate_optimal_covariance_and_volume(signal_var, "one_fixed", disc_params_this, XWXs, Fs, (experiment_datasets[0].grid_size,)*3)


    final_estimates = final_estimates.transpose(1, 2, 0, 3)
    first_estimates = first_estimates.transpose(1, 2, 0, 3)

    # residuals to pick best one
    residuals, _ = compute_residuals_many_weights_in_weight_batch(experiment_datasets[1], final_estimates[0], max_pol_degree )

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, discretization_params, (experiment_datasets[0].grid_size,)*3)

    weights_opt = jnp.take_along_axis(0.5 * (final_estimates[0][...,0] + final_estimates[1][...,0]), np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(final_estimates), np.asarray(first_estimates), discretization_params, prior_inverse_covariance # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)

def pick_best_params(residuals, discretization_params, volume_shape):

    # residuals to pick best one
    residuals_averaged = regularization.batch_average_over_shells(residuals.T, volume_shape,0)
    # Make choice. Impose that h must be increasing
    index_array = jnp.argmin(residuals_averaged, axis = 0)
    # This assumes that the discretization params are ordered by h

    # Check that all polynomial terms are the same:
    pol_degrees = np.array( [ param[0] for param in discretization_params ])

    if np.isclose(pol_degrees, pol_degrees[0]).all():
        index_array = index_array
    else:
        logger.warning("Non-uniform pol_degrees: %s", pol_degrees)

    disc_choices = np.array(discretization_params)[index_array]
    hs_choices = disc_choices[:,1].astype(int)
    hs_choices = np.maximum.accumulate(hs_choices)
    disc_choices[:,1] = hs_choices

    index_array_vol = utils.make_radial_image(index_array, volume_shape)
    return index_array_vol, disc_choices, residuals_averaged


def get_prior_from_options(prior_inverse_covariance, prior_option, vec_indices, volume_shape ):
    if prior_option == "by_degree":
        regularization_expanded = make_regularization_from_reduced(prior_inverse_covariance[vec_indices])
        regularization = batch_diag(regularization_expanded)
    elif prior_option == "complete":
        frequencies = core.vec_indices_to_frequencies(vec_indices, volume_shape)
        radiuses = jnp.round(jnp.linalg.norm(frequencies, axis=-1)).astype(int)
        # I should probably store the inverse of these of pseudo inverse, which is not great but they are small so should be okay...
        regularization = prior_inverse_covariance[radiuses]
    elif prior_option == "none":
        regularization = jnp.zeros((vec_indices.shape[0],1,1 ), dtype = np.float32)
    return regularization

batch_kron = jax.vmap(jnp.kron, in_axes=(0,0))
batch_diag = jax.vmap(jnp.diag, in_axes=(0))

def batch_vec(x):
    return x.swapaxes(-1,-2).reshape(-1, x.shape[-1]**2)

def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1,n,n).swapaxes(-1,-2)


def estimate_local_pol_covariances_inner(XWX, estimate_0, estimate_1, prior_inverse_covariance, prior_option, frequency_vec_indices, volume_shape):
    frequencies = core.vec_indices_to_frequencies(frequency_vec_indices, volume_shape)
    # if freq is not on the 0-line, can double it (since there is one XWX for each side)
    doubling = jnp.where(frequencies[...,0] == 0, 1, jnp.sqrt(2))
    radiuses = jnp.round(jnp.linalg.norm(frequencies, axis=-1)).astype(int)

    prior_inverse_covariance = get_prior_from_options(prior_inverse_covariance, prior_option, frequency_vec_indices, volume_shape )
    XWX = undo_keep_upper_triangular(XWX)
    U = (XWX + prior_inverse_covariance)
    krons = batch_kron(U,U)
    _n_shells = volume_shape[0] // 2 - 1
    _krons_flat = krons.reshape(krons.shape[0], -1)
    summed_krons = jnp.zeros((_n_shells, _krons_flat.shape[-1]), dtype=_krons_flat.dtype).at[radiuses].add(_krons_flat).reshape([-1, krons.shape[-1], krons.shape[-1]])

    estimate_covariance = linalg.broadcast_outer(estimate_0 , estimate_1 )
    estimate_covariance = 0.5 * (estimate_covariance + jnp.conj(estimate_covariance.swapaxes(-1,-2)))
    estimate_covariance = batch_vec(estimate_covariance)

    _ec_flat = estimate_covariance.reshape(krons.shape[0], -1)
    estimate_covariance_summed = jnp.zeros((_n_shells, _ec_flat.shape[-1]), dtype=_ec_flat.dtype).at[radiuses].add(_ec_flat)

    good_v = jnp.where(frequencies[...,0] == 0, 0, 1)

    # Add the hermitian conjugate
    estimate_covariance = linalg.broadcast_outer(jnp.conj(estimate_0) * good_v[...,None], jnp.conj(estimate_1) * good_v[...,None])
    estimate_covariance = 0.5 * (estimate_covariance + jnp.conj(estimate_covariance.swapaxes(-1,-2)))
    estimate_covariance = batch_vec(estimate_covariance)

    estimate_covariance_summed = estimate_covariance_summed.at[radiuses].add(estimate_covariance.reshape(krons.shape[0], -1))


    ## TODO There is a wild 0.5 here b/c we are treating real and imaginary part as independent
    return summed_krons, 0.5 * batch_unvec(estimate_covariance_summed)


def estimate_local_covariances(XWX_0, XWX_1, estimate_0, estimate_1, prior_inverse_covariance, prior_option, volume_shape, pol_degree):

    logger.info("starting local covariances estimation")
    volume_size = np.prod(volume_shape)
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    num_params = get_feature_size(pol_degree)

    krons = jnp.zeros((volume_shape[0]//2-1, num_params**2, num_params**2), dtype = XWX_0.dtype)
    vec_indices = np.arange(half_volume_size)
    # Covariances of estimates
    estimate_0 = estimate_0.astype(np.complex64)
    estimate_1 = estimate_1.astype(np.complex64)
    utils.report_memory_device(logger=logger)
    # Batch size based on GPU memory and per-pixel Kronecker product size
    batch_size = int((utils.get_gpu_memory_total()  )/ ( 1 * 4 * num_params**4 * 4 / 1e9 )  )
    n_batches = np.ceil(half_volume_size / batch_size).astype(int)
    krons, estimate_covariance_averaged = 0,0
    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(half_volume_size, batch_size, k)
        krons_this, covs_this = estimate_local_pol_covariances_inner((XWX_0[ind_st:ind_end] + XWX_1[ind_st:ind_end])/2, estimate_0[ind_st:ind_end], estimate_1[ind_st:ind_end],  prior_inverse_covariance, prior_option, vec_indices[ind_st:ind_end], volume_shape)

        krons += krons_this
        estimate_covariance_averaged += covs_this
    logger.info("end of local covariances batch")
    utils.report_memory_device(logger=logger)


    estimated_covariances = linalg.batch_hermitian_linear_solver(krons, batch_vec(estimate_covariance_averaged).real)#, assume_a='pos' )

    estimated_covariances = batch_unvec(estimated_covariances)
    logger.info("end of local covariances estimation")

    bad_covars = jnp.linalg.cond(krons) > 1e4

    return estimated_covariances, bad_covars


def estimate_signal_variance_from_correlation(vol1, vol2, lhs, prior, volume_shape):
    correlation = jnp.conj(vol1) * vol2
    correlation_avg = regularization.average_over_shells(correlation.real, volume_shape, frequency_shift = np.array([0,0,0]))
    correlation_avg = jnp.where( correlation_avg > jax_config.ROOT_EPSILON , correlation_avg , jax_config.ROOT_EPSILON )

    top = lhs**2 / (lhs + 1/prior)**2
    sum_top = regularization.average_over_shells(top,  volume_shape, frequency_shift = np.array([0,0,0]))
    prior_avg = jnp.where( sum_top > 0 , correlation_avg / sum_top , jax_config.ROOT_EPSILON )

    # Put back in array
    radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = np.array([0,0,0])).astype(int).reshape(-1)
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

def test_multiple_disc2(experiment_dataset, noise_variance, discretization_params, prior_inverse_covariance):

    discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWX, F = precompute_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=max_pol_degree)
    n_disc_test = len(discretization_params)

    # Compute weights for each discretization
    weights = np.zeros((n_disc_test, experiment_dataset.grid_size**3, get_feature_size(max_pol_degree)), dtype = np.complex64)
    valid_weights = np.zeros((n_disc_test, experiment_dataset.grid_size**3), dtype = bool)
    utils.report_memory_device(logger=logger)
    XWX = np.asarray(XWX)
    F = np.asarray(F)

    XWX_s = dict()
    F_s = dict()
    for idx, (pol_degree, h, reg) in enumerate(discretization_params):
        logger.info("computing discretization with params: degree=%s, h=%s, reg=%s", pol_degree, h, reg)
        weights_this, valid_weights_this, XWX_s[idx],F_s[idx] = compute_weights_from_precompute((experiment_dataset.grid_size,)*3, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, prior_option = reg)
        weights[idx,:,:weights_this.shape[-1]] = weights_this
        valid_weights[idx] = valid_weights_this

    return XWX_s, F_s


def test_multiple_disc(experiment_dataset, cross_validation_dataset, noise_variance,  batch_size, discretization_params, prior_inverse_covariance, return_all = False):

    discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWX, F = precompute_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=max_pol_degree)
    n_disc_test = len(discretization_params)

    # Compute weights for each discretization
    weights = np.zeros((n_disc_test, experiment_dataset.grid_size**3, get_feature_size(max_pol_degree)), dtype = np.complex64)
    valid_weights = np.zeros((n_disc_test, experiment_dataset.grid_size**3), dtype = bool)
    utils.report_memory_device(logger=logger)
    XWX = np.asarray(XWX)
    F = np.asarray(F)
    for idx, (pol_degree, h, reg) in enumerate(discretization_params):
        logger.info("computing discretization with params: degree=%s, h=%s, reg=%s", pol_degree, h, reg)
        weights_this, valid_weights_this, _,_ = compute_weights_from_precompute((experiment_dataset.grid_size,)*3, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, prior_option = reg)
        weights[idx,:,:weights_this.shape[-1]] = weights_this
        valid_weights[idx] = valid_weights_this


    logger.info("Done computing params")
    weights = weights.swapaxes(0,1)
    utils.report_memory_device(logger=logger)
    del XWX, F

    # residuals to pick best one
    residuals, _ = compute_residuals_many_weights_in_weight_batch(cross_validation_dataset, weights, max_pol_degree )
    residuals_averaged = regularization.batch_average_over_shells(residuals.T, (experiment_dataset.grid_size,)*3,0)

    # Make choice. Impose that h must be increasing
    index_array = jnp.argmin(residuals_averaged, axis = 0)
    index_array_vol = utils.make_radial_image(index_array, (experiment_dataset.grid_size,)*3)

    disc_choices = np.array(discretization_params)[index_array]
    hs_choices = disc_choices[:,1].astype(int)
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

def compute_weights_from_precompute(volume_shape, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, return_XWX_F = False, prior_option = "by_degree"):
    volume_size = np.prod(volume_shape)

    # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    if type(h) == float or type(h) == int:
        h_max = int(h)
        h_ar = h*np.ones(volume_size)
    else:
        h_max = int(np.max(h))
        h_ar = h.astype(int) * 1.0

    feature_size = get_feature_size(pol_degree)

    if XWX.shape[-1] != small_gram_matrix_size(pol_degree):
        triu_indices = find_smaller_pol_indices(max_pol_degree, pol_degree)
        XWX = XWX[...,triu_indices]
        F = F[...,:feature_size]

    # take out stuff from prior
    if prior_option == "by_degree":
        prior_inverse_covariance = prior_inverse_covariance[:,:pol_degree+1]   
    elif prior_option == "complete":
        num_params = get_feature_size(pol_degree)
        prior_inverse_covariance = prior_inverse_covariance[...,:num_params, :num_params]
    else:
        prior_inverse_covariance = None

    memory_per_pixel = (2*h_max +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    # JAX can silently corrupt memory near OOM without raising errors.
    if h_max == 0:
        memory_per_pixel = (2*1 +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    # Safety factor: JAX can silently corrupt memory near OOM without raising errors
    memory_per_pixel *= 10

    XWX = jnp.asarray(XWX)
    F = jnp.asarray(F)
    prior_inverse_covariance = jnp.asarray(prior_inverse_covariance)

    reconstruction = np.zeros((volume_size, feature_size), dtype = np.complex64)
    good_pixels = np.zeros((volume_size), dtype = bool)
    msgs = 0
    
    if return_XWX_F:
        XWX_s = np.zeros_like(XWX)
        F_s = np.zeros_like(F)
    else:
        XWX_s = None
        F_s = None

    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX) - utils.get_size_in_gb(F)  )/ (memory_per_pixel   /1e9  )  ) 
    n_batches = np.ceil(volume_size / batch_size).astype(int)
    logger.info("KE batch size: %s", batch_size)

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(volume_size, batch_size, k)
        reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems, XWX_b, F_b = compute_estimate_from_precompute_one(XWX, F, h_max, h_ar[ind_st:ind_end], threed_frequencies[ind_st:ind_end], volume_shape, pol_degree =pol_degree, prior_inverse_covariance=prior_inverse_covariance, prior_option = prior_option, return_XWX_F = return_XWX_F)

        if return_XWX_F and (ind_st < XWX_s.shape[0]):
            ind_end_t = np.min([ind_end, XWX_s.shape[0]])
            # if ind_st <XWX_s.shape[0]
            XWX_s[ind_st:ind_end_t] = XWX_b[:ind_end_t-ind_st]
            F_s[ind_st:ind_end_t] = F_b[:ind_end_t-ind_st]

        if k < 3:
            utils.report_memory_device(logger=logger)
            logger.debug("batch %d", k)

        if jnp.isnan(reconstruction[ind_st:ind_end]).any():
            logger.warning("IsNAN %s pixels, pol_degree=%s, h=%s, reg=%s", jnp.isnan(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size, pol_degree, h, prior_option)

        if problems.any():
            if msgs < 10: 
                logger.warning("Issues in linalg solve? Problems for %s pixels, pol_degree=%s, h=%s, reg=%s", problems.sum() / problems.size, pol_degree, h, prior_option)
                msgs +=1
                logger.warning("isinf %s pixels, pol_degree=%s, h=%s, reg=%s", jnp.isinf(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size, pol_degree, h, prior_option)

    logger.info("Done with kernel estimate")
    return reconstruction, good_pixels, XWX_s, F_s


def compute_summed_XWX_F_only(volume_shape, XWX, F, pol_degree, h):
    volume_size = np.prod(volume_shape)

    # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0
    
    XWX = jnp.asarray(XWX)
    F = jnp.asarray(F)
    # compute_summed_XWX_F requires at least 2-D arrays (last dim = gram/feature size).
    # After callers strip the bin dimension for the single-bin case the arrays can be
    # 1-D (half_vol_size,); reshape them here so the downstream vmap over the last
    # axis works correctly.
    if XWX.ndim == 1:
        XWX = XWX[..., None]
    if F.ndim == 1:
        F = F[..., None]
    XWX_s = np.zeros_like(XWX)[...,None]
    F_s = np.zeros_like(F)
    memory_per_pixel = (2*h +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4

    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX) - utils.get_size_in_gb(F)  )/ (memory_per_pixel   /1e9  )  ) 
    n_batches = np.ceil(volume_size / batch_size).astype(int)
    logger.info("KE batch size: %s", batch_size)
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(volume_size, batch_size, k)
        # probably should rewrite this to not have extra indices
        if (ind_st < XWX_s.shape[0]):
            ind_end_t = np.min([ind_end, XWX_s.shape[0]])

            XWX_s[ind_st:ind_end_t], F_s[ind_st:ind_end_t] = compute_summed_XWX_F(XWX, F, h, h, threed_frequencies[ind_st:ind_end_t], volume_shape, pol_degree)

    logger.info("Done with kernel estimate")
    return XWX_s[...,0], F_s[...]


def compute_residuals_many_weights(experiment_dataset, weights , pol_degree, use_linear_interp ):
    pol_degree = int(pol_degree)  # ensure Python int (numpy scalars are traced by eqx.filter_jit)
    use_linear_interp = bool(use_linear_interp)
    # *5: slicing is cheap; /weights_per_image: each weight adds a volume-sized projection
    batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - utils.get_size_in_gb(weights)) * 5 / np.prod(weights.shape[1:]))

    config = ForwardModelConfig(
        image_shape=tuple(experiment_dataset.image_shape),
        volume_shape=tuple((experiment_dataset.grid_size,)*3),
        grid_size=int(experiment_dataset.grid_size),
        voxel_size=float(experiment_dataset.voxel_size),
        padding=int(experiment_dataset.padding),
        disc_type='',
        ctf=experiment_dataset.ctf_evaluator,
        premultiplied_ctf=False,
        volume_mask_threshold=float(experiment_dataset.volume_mask_threshold),
    )

    residuals, summed_n = 0, 0
    logger.info("batch size in residual computation: %s", batch_size)
    weights = jnp.asarray(weights)
    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        apply_process_images=True,
    ):
        residuals_t, summed_n_t = compute_residuals_batch(
            config, batch_data, weights,
            pol_degree=pol_degree, use_linear_interp=use_linear_interp,
        )
        residuals += residuals_t
        summed_n += summed_n_t

    return residuals, summed_n


def compute_residuals_many_weights_in_weight_batch(experiment_dataset, weights, pol_degree, use_linear_interp = False ):
    
    # Keep half memory free for other stuff

    n_batches = utils.get_size_in_gb(weights) / (0.5 * utils.get_gpu_memory_total() )
    weight_batch_size = np.floor(weights.shape[-2] / n_batches).astype(int)
    n_batches = np.ceil(weights.shape[-2] / weight_batch_size).astype(int)

    logger.info("number of batches in residual computation: %s, batch size: %s", n_batches, weight_batch_size)
    residuals = []
    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(weights.shape[-2], weight_batch_size, k)
        res, _ = compute_residuals_many_weights(experiment_dataset,
                                                weights[:,ind_st:ind_end],
                                                pol_degree, use_linear_interp )
        residuals.append(res)

    return np.concatenate(residuals, axis = -1), 0


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
