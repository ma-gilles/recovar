"""Core EM iteration logic: cross-correlation, residual computation."""

import functools
import jax
import jax.numpy as jnp
import equinox as eqx
from recovar import core
from recovar.core.configs import ForwardModelConfig
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# Probabilities will be 4 dimensional:
IMAGE_AXIS=0
VOL_AXIS=1
ROT_AXIS=2
TRANS_AXIS=3


NORM_FFT = "backward"

# batch volumes
batch_vol_rot_slice_volume = jax.vmap(core.slice_volume, in_axes = (0, VOL_AXIS, None, None, None), out_axes=1 )
batch_vol_slice_volume = jax.vmap(core.slice_volume, in_axes = (0, None, None, None, None), out_axes=1 )



import recovar.core.fourier_transform_utils as fourier_transform_utils
def crosscorr_from_ft(many_images, one_image, image_shape):
    return fourier_transform_utils.get_idft2( jnp.conj(one_image.reshape(1, *image_shape)) * (many_images.reshape(-1, *image_shape)))

def norm_squared_residuals_from_ft_one_image(many_images, one_image, image_shape):
    many_images_of_shape = many_images.shape
    many_images = many_images.reshape(-1, many_images.shape[-1])
    many_images = crosscorr_from_ft(many_images, one_image, image_shape)
    many_images = many_images.reshape(many_images_of_shape)
    return many_images

norm_squared_residuals_from_ft = jax.vmap(norm_squared_residuals_from_ft_one_image, in_axes = (0, 0, None))



@functools.partial(jax.jit, static_argnums=[4,6,8])
def compute_dot_products(projections, batch, translations, CTF_params, CTF_fun, noise_variance, process_images, voxel_size, image_shape):
    '''
    Computes -2 * y_i.T @ (S_s C_i * Proj_j) for i,j,s
    where C_i is CTF, S_s are shifts, and Proj_j are projections, y_i are batch (unprocessed)
    '''
    batch = process_images(batch, apply_image_mask = False)
    batch_norm = jnp.linalg.norm(batch / jnp.sqrt(noise_variance), axis = (-1), keepdims = True)**2

    batch *= CTF_fun( CTF_params, image_shape, voxel_size) / noise_variance
    result = jnp.empty((batch.shape[0], projections.shape[0], translations.shape[0]), dtype = jnp.float32)

    # Compute IP for each shift (memory-efficient over computing all shifted images at once).
    shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    n_shifted_images = np.prod(shifted_images.shape[:-1])
    result = -2 * (jnp.conj(shifted_images).reshape(n_shifted_images, shifted_images.shape[-1] ) @ projections.T).real
    result = result.reshape(batch.shape[0], translations.shape[0], projections.shape[0]) + batch_norm[:,None]
    result = result.swapaxes(1,2)

    return result


# ============================================================================
# Equinox-based EM core API
# ============================================================================

@eqx.filter_jit
def compute_dot_products_eqx(config: ForwardModelConfig, projections, batch, translations, ctf_params, noise_variance):
    """Equinox version of compute_dot_products (9 → 6 params)."""
    batch = config.process_fn(batch, apply_image_mask=False)
    batch_norm = jnp.linalg.norm(batch / jnp.sqrt(noise_variance), axis=(-1), keepdims=True)**2
    batch *= config.compute_ctf(ctf_params) / noise_variance
    shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), config.image_shape)
    n_shifted_images = np.prod(shifted_images.shape[:-1])
    result = -2 * (jnp.conj(shifted_images).reshape(n_shifted_images, shifted_images.shape[-1]) @ projections.T).real
    result = result.reshape(batch.shape[0], translations.shape[0], projections.shape[0]) + batch_norm[:,None]
    result = result.swapaxes(1,2)
    return result


@eqx.filter_jit
def compute_CTFed_proj_norms_eqx(config: ForwardModelConfig, projections, ctf_params, noise_variance):
    """Equinox version of compute_CTFed_proj_norms (6 → 4 params)."""
    CTFs = config.compute_ctf(ctf_params)**2 / noise_variance
    return CTFs @ projections.T


# ============================================================================
# Legacy EM core API
# ============================================================================

@functools.partial(jax.jit, static_argnums=[2,5])
def compute_CTFed_proj_norms(projections, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape):
    '''
    Computes  |C_i Proj_j|^2 for i,j by writing it as a mat-mat
    where C_i is CTF, S_s are shifts, and Proj_j are projections
    '''
    CTFs = CTF_fun( CTF_params, image_shape, voxel_size)**2 / noise_variance

    result = CTFs @ projections.T

    return result




def probabilities_to_hard_assignment_pose(probabilities, rotation_grid, translation_grid):
    idx = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)
    return hard_assignment_idx_to_pose(idx, rotation_grid, translation_grid)

def probabilities_to_hard_assignment_idx(probabilities, rotation_grid, translation_grid):
    idx = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)
    square_shape = (rotation_grid.shape[0], translation_grid.shape[0])
    maxpos_vect = np.column_stack(np.unravel_index(idx,square_shape))
    rot_idx = maxpos_vect[:,0]
    trans_idx = maxpos_vect[:,1]
    return rot_idx, trans_idx

def hard_assignment_idx_to_pose(indices, rotation_grid, translation_grid):
    square_shape = (rotation_grid.shape[0], translation_grid.shape[0])
    maxpos_vect = np.column_stack(np.unravel_index(indices,square_shape))
    predicted_trans = translation_grid[maxpos_vect[:,1]]
    predicted_pose = rotation_grid[maxpos_vect[:,0]]
    return predicted_pose, predicted_trans


def estimate_error_from_hard_assignment(hard_assignment, gt_pose, gt_trans, rotation_grid, translation_grid):
    predicted_pose, predicted_trans = hard_assignment_idx_to_pose(hard_assignment, rotation_grid, translation_grid)
    predicted_pose = R.from_matrix(predicted_pose)
    gt_pose = R.from_matrix(gt_pose)
    error = (predicted_pose * gt_pose.inv()).magnitude() / np.pi * 180
    
    mean_angle_error = np.mean(error)
    mean_trans_error = np.mean(np.linalg.norm(predicted_trans - gt_trans, axis=-1))
    logger.info("mean trans error: %s pixels", mean_trans_error)
    logger.info("mean angle error: %s degrees", mean_angle_error)

    return np.mean(error), np.mean(np.linalg.norm(predicted_trans - gt_trans, axis=-1))
