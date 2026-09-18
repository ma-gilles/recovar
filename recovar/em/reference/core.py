"""Core EM iteration logic: cross-correlation, residual computation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.core import fourier_transform_utils
from recovar.core.configs import ForwardModelConfig

# Probabilities will be 4 dimensional:
IMAGE_AXIS = 0
VOL_AXIS = 1
ROT_AXIS = 2
TRANS_AXIS = 3


NORM_FFT = "backward"

# batch volumes
batch_vol_rot_slice_volume = jax.vmap(core.slice_volume, in_axes=(0, VOL_AXIS, None, None, None), out_axes=1)
batch_vol_slice_volume = jax.vmap(core.slice_volume, in_axes=(0, None, None, None, None), out_axes=1)


def crosscorr_from_ft(many_images, one_image, image_shape):
    return fourier_transform_utils.get_idft2(
        jnp.conj(one_image.reshape(1, *image_shape)) * (many_images.reshape(-1, *image_shape))
    )


def norm_squared_residuals_from_ft_one_image(many_images, one_image, image_shape):
    many_images_of_shape = many_images.shape
    many_images = many_images.reshape(-1, many_images.shape[-1])
    many_images = crosscorr_from_ft(many_images, one_image, image_shape)
    many_images = many_images.reshape(many_images_of_shape)
    return many_images


norm_squared_residuals_from_ft = jax.vmap(norm_squared_residuals_from_ft_one_image, in_axes=(0, 0, None))


@eqx.filter_jit
def compute_dot_products(config: ForwardModelConfig, projections, batch, translations, ctf_params, noise_variance):
    """Compute image/projection dot products over the translation grid."""
    batch = config.process_fn(batch, apply_image_mask=False)
    batch_norm = jnp.linalg.norm(batch / jnp.sqrt(noise_variance), axis=(-1), keepdims=True) ** 2
    batch *= config.compute_ctf(ctf_params) / noise_variance
    shifted_images = core.batch_trans_translate_images(
        batch, jnp.repeat(translations[None], batch.shape[0], axis=0), config.image_shape
    )
    n_shifted_images = np.prod(shifted_images.shape[:-1])
    result = -2 * (jnp.conj(shifted_images).reshape(n_shifted_images, shifted_images.shape[-1]) @ projections.T).real
    result = result.reshape(batch.shape[0], translations.shape[0], projections.shape[0]) + batch_norm[:, None]
    result = result.swapaxes(1, 2)
    return result


@eqx.filter_jit
def compute_ctf_projection_norms(config: ForwardModelConfig, projections, ctf_params, noise_variance):
    """Compute noise-weighted squared-CTF projection norms."""
    CTFs = config.compute_ctf(ctf_params) ** 2 / noise_variance
    return CTFs @ projections.T

def hard_assignment_idx_to_pose(indices, rotation_grid, translation_grid):
    square_shape = (rotation_grid.shape[0], translation_grid.shape[0])
    maxpos_vect = np.column_stack(np.unravel_index(indices, square_shape))
    predicted_trans = translation_grid[maxpos_vect[:, 1]]
    predicted_pose = rotation_grid[maxpos_vect[:, 0]]
    return predicted_pose, predicted_trans
