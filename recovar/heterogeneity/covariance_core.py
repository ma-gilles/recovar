"""Low-level covariance column computation kernels (JAX)."""

import logging
import jax
import jax.numpy as jnp
import numpy as np
import nvtx
import equinox as eqx

import recovar.core.padding as pad
import functools
from recovar import core
from recovar.core import mask
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)

# NVTX domain for covariance core operations
NVTX_DOMAIN_COV_CORE = "covariance_core"

# Covariance computation

def pick_frequencies(freq, radius = 2, use_half = True):
    if use_half:
        return (jnp.linalg.norm(freq, axis = -1) <= radius) * (freq[...,0] >= 0) 
    else:
        return (jnp.linalg.norm(freq, axis = -1) <= radius)

def get_picked_frequencies(volume_shape, radius = 2, use_half = True):
    # Chose the frequency we will use to compute.
    volume_size = np.prod(volume_shape)
    three_d_freqs = core.vec_indices_to_frequencies(jnp.arange(volume_size), volume_shape)
    picked_three_d_freqs_mask = pick_frequencies(three_d_freqs, radius, use_half)
    picked_three_d_freqs = three_d_freqs[picked_three_d_freqs_mask]
    picked_frequency_indices = core.frequencies_to_vec_indices(picked_three_d_freqs, volume_shape)
    return picked_frequency_indices

@functools.partial(jax.jit, static_argnums = [4,5,6,7,8,9,10])    
@nvtx.annotate("get_per_image_tight_mask", color="green", domain=NVTX_DOMAIN_COV_CORE)
def get_per_image_tight_mask(volume_mask, rotation_matrices, image_mask, mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, binary = True, soften = -1):
    
    disc_type = 'linear_interp'
    
    if disc_type == 'cubic':
        extra_padding = 0 
        mask_ft = volume_mask
    else:
        # if padding is already there, do nothing else double image size.
        extra_padding = grid_size if ( padding == 0 ) else 0
        # Do this in half precision? Shouldn't matter much.
        volume_mask = pad.pad_volume_spatial_domain(volume_mask, extra_padding).real
        mask_ft = fourier_transform_utils.get_dft3(volume_mask).reshape(-1)

    padded_image_shape = tuple(np.array(image_shape) + extra_padding)
    padded_volume_shape = tuple(np.array(volume_shape) + extra_padding)
    padded_grid_size = grid_size + extra_padding

    proj_mask = core.slice_volume_by_map(mask_ft, rotation_matrices, padded_image_shape,
                               padded_volume_shape, disc_type)
    
    proj_mask = fourier_transform_utils.get_idft2(proj_mask.reshape([-1] + list(padded_image_shape)))
                             
    if extra_padding > 0:
        proj_mask = pad.unpad_images_spatial_domain(proj_mask, extra_padding)
              
    if padding > 0:
        image_mask = pad.pad_images_spatial_domain(image_mask, padding)[0]

    if binary:
        proj_mask = (proj_mask > mask_threshold)  * ( image_mask[None]  if image_mask is not None else 1)
        
    if soften > 0:
        # Soft mask
        soft_edge_kernel = mask.create_soft_edged_kernel_pxl(soften, image_shape).astype(volume_mask.dtype)
        
        # Convolve
        soft_edge_kernel_ft = fourier_transform_utils.get_dft2(soft_edge_kernel)
        proj_mask_ft = fourier_transform_utils.get_dft2(proj_mask.reshape([-1] + list(image_shape)))

        proj_mask_ft = proj_mask_ft * soft_edge_kernel_ft[None]
        proj_mask = fourier_transform_utils.get_idft2(proj_mask_ft.reshape([-1] + list(image_shape))).real

    return proj_mask


@functools.partial(jax.jit, static_argnums = [2])    
@nvtx.annotate("apply_image_masks", color="cyan", domain=NVTX_DOMAIN_COV_CORE)
def apply_image_masks(images, image_masks, image_shape):
    images = fourier_transform_utils.get_idft2(images.reshape([images.shape[0], *image_shape]))
    images = images * image_masks
    images = fourier_transform_utils.get_dft2(images).reshape([images.shape[0] , -1])
    return images


@functools.partial(jax.jit, static_argnums = [2])    
@nvtx.annotate("apply_image_masks_to_eigen", color="cyan", domain=NVTX_DOMAIN_COV_CORE)
def apply_image_masks_to_eigen(proj_eigen, image_masks, image_shape):
    proj_eigen = fourier_transform_utils.get_idft2(proj_eigen.reshape([*proj_eigen.shape[0:2], *image_shape]))
    proj_eigen = proj_eigen * image_masks
    proj_eigen = fourier_transform_utils.get_dft2(proj_eigen).reshape([*proj_eigen.shape[0:2], -1])
    return proj_eigen


def check_mask(mask):
    no_mask = np.all(np.isclose(mask,1))
    if no_mask:
        logger.info("no mask used")
    return no_mask

batch_forward_model = jax.vmap(core.forward_model, in_axes = (0, None, None))

@functools.partial(jax.jit, static_argnums = [3,4,5,6,7])
@nvtx.annotate("batch_over_vol_forward_model", color="blue", domain=NVTX_DOMAIN_COV_CORE)
def batch_over_vol_forward_model(mean, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):
    batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
    batch_CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    projected_mean =  batch_forward_model(mean, batch_CTF, batch_grid_pt_vec_ind_of_images)
    return projected_mean


def batch_vol_forward_from_map(
    config: ForwardModelConfig,
    volumes: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Forward-model a batch of volumes via slice_volume_by_map (vmap over volume axis)."""
    return jax.vmap(
        lambda vol: core_forward.forward_model(config, vol, ctf_params, rotation_matrices, skip_ctf=skip_ctf),
    )(volumes)


# ============================================================================
# Equinox-based API
# ============================================================================


@eqx.filter_jit
@nvtx.annotate("centered_images", color="yellow", domain=NVTX_DOMAIN_COV_CORE)
def centered_images(
    config: ForwardModelConfig,
    images: jax.Array,
    mean: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    translations: jax.Array,
) -> jax.Array:
    """Compute y_i - A_i mu (centered images) using ForwardModelConfig.

    If ``config.premultiplied_ctf`` is True, computes z_i - CTF_i^2 P_i mu
    where z_i = y_i CTF_i.
    """
    translated = core.translate_images(images, translations, config.image_shape)
    if config.premultiplied_ctf:
        projected = core_forward.forward_model(
            config, mean, ctf_params, rotation_matrices, skip_ctf=True
        )
        centered = translated - projected * config.compute_ctf(ctf_params) ** 2
    else:
        projected = core_forward.forward_model(
            config, mean, ctf_params, rotation_matrices, skip_ctf=False
        )
        centered = translated - projected
    return centered


@eqx.filter_jit
@nvtx.annotate("batch_vol_forward", color="blue", domain=NVTX_DOMAIN_COV_CORE)
def batch_vol_forward(
    config: ForwardModelConfig,
    volumes: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
) -> jax.Array:
    """Forward-model a batch of volumes (vmap over volume axis)."""
    batch_grid_pt_vec_ind = core.batch_get_nearest_gridpoint_indices(
        rotation_matrices, config.image_shape, config.volume_shape
    )
    batch_CTF = config.compute_ctf(ctf_params)
    return batch_forward_model(volumes, batch_CTF, batch_grid_pt_vec_ind)


def triangular_kernel(gridpoints, gridpoint_target, kernel_width = 1):
    weights = jnp.ones(gridpoints.shape[:-1])
    # Note that this is a very small loop (3) so it should be fine to jit this
    for i in range(gridpoint_target.shape[-1]):
        weights *= jnp.where(jnp.abs(gridpoints[...,i] - gridpoint_target[i]) < kernel_width, 1 - jnp.abs(gridpoints[...,i] - gridpoint_target[i]) / kernel_width, 0) #/ kernel_width
    return weights


def square_kernel(gridpoints, gridpoint_target, kernel_width = 1):
    weights = jnp.ones(gridpoints.shape[:-1])
    # Note that this is a very small loop (3) so it should be fine to jit this 
    for i in range(gridpoint_target.shape[-1]):
        weights *= jnp.where(jnp.abs(gridpoints[...,i] - gridpoint_target[i]) < kernel_width/2, 1/ kernel_width, 0) 
    return weights


# Are there at most 4 or 5 within one dist? or 9?
#@jax.vmap(in_axes=[0,0,None])
@nvtx.annotate("sum_up_over_near_grid_points", color="orange", domain=NVTX_DOMAIN_COV_CORE)
def sum_up_over_near_grid_points(image, gridpoints, gridpoint_target, kernel = "triangular", kernel_width = 1):
    # if kernel == "triangular":
    #     kernel_vals = triangular_kernel(gridpoints, gridpoint_target, kernel_width = kernel_width)
    # elif kernel == "square":
    #     kernel_vals = square_kernel(gridpoints, gridpoint_target, kernel_width = kernel_width)
    # else:
    #     raise ValueError("Kernel function not recognized")
    # kernel_vals = triangular_kernel(gridpoints, gridpoint_target, kernel_width = 1)
    kernel_vals = evaluate_kernel_on_grid(gridpoints, gridpoint_target, kernel = kernel, kernel_width = kernel_width)
    kernel_estimated = jnp.sum(kernel_vals * image, axis =-1)
    return kernel_estimated #, jnp.sum(kernel_vals)

@nvtx.annotate("evaluate_kernel_on_grid", color="magenta", domain=NVTX_DOMAIN_COV_CORE)
def evaluate_kernel_on_grid(gridpoints, gridpoint_target, kernel = "triangular", kernel_width = 1):
    if kernel == "triangular":
        kernel_vals = triangular_kernel(gridpoints, gridpoint_target, kernel_width = kernel_width)
    elif kernel == "square":
        kernel_vals = square_kernel(gridpoints, gridpoint_target, kernel_width = kernel_width)
    else:
        raise ValueError("Kernel function not recognized")
    return kernel_vals



