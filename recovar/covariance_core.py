import logging
import jax
import jax.numpy as jnp
import numpy as np

import recovar.padding as pad
import functools
from recovar import core, mask
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

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
def get_per_image_tight_mask(volume_mask, rotation_matrices, image_mask, mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, binary = True, soften = -1):
    
    # if padding is already there, do nothing else double image size.
    extra_padding = grid_size if ( padding == 0 ) else 0
    padded_image_shape = tuple(np.array(image_shape) + extra_padding)
    padded_volume_shape = tuple(np.array(volume_shape) + extra_padding)
    padded_grid_size = grid_size + extra_padding

    # Do this in half precision? Shouldn't matter much.
    volume_mask = pad.pad_volume_spatial_domain(volume_mask, extra_padding).real
    mask_ft = ftu.get_dft3(volume_mask).reshape(-1)
    proj_mask = core.get_slices(mask_ft, rotation_matrices, padded_image_shape,
                               padded_volume_shape, padded_grid_size, disc_type)
    
    proj_mask = ftu.get_idft2(proj_mask.reshape([-1] + list(padded_image_shape)))
                             
    if extra_padding > 0:
        proj_mask = pad.unpad_images_spatial_domain(proj_mask, extra_padding)
              
    if padding > 0:
        image_mask = pad.pad_images_spatial_domain(image_mask, padding)[0]

    if binary:
        proj_mask = (proj_mask > mask_threshold)  * image_mask[None]    
        
    if soften > 0:
        # Soft mask
        soft_edge_kernel = mask.create_soft_edged_kernel_pxl(soften, image_shape).astype(volume_mask.dtype)
        
        # Convolve
        soft_edge_kernel_ft = ftu.get_dft2(soft_edge_kernel)
        proj_mask_ft = ftu.get_dft2(proj_mask.reshape([-1] + list(image_shape)))

        proj_mask_ft = proj_mask_ft * soft_edge_kernel_ft[None]
        proj_mask = ftu.get_idft2(proj_mask_ft.reshape([-1] + list(image_shape))).real

    return proj_mask

def get_per_image_tight_mask_cryo(mask, indices, cryo, disc_type  , binary = True, use_extra_padding =True, soften = 3):
    # pad = 0 if use_extra_padding else 1
    image_masks = get_per_image_tight_mask(mask, cryo.rotation_matrices[indices], cryo.image_stack.mask, cryo.volume_mask_threshold, cryo.image_shape, cryo.volume_shape, cryo.grid_size, cryo.padding, disc_type,  binary = binary, soften = soften)
    return image_masks


@functools.partial(jax.jit, static_argnums = [2])    
def apply_image_masks(images, image_masks, image_shape):
    images = ftu.get_idft2(images.reshape([images.shape[0], *image_shape]))
    images = images * image_masks
    images = ftu.get_dft2(images).reshape([images.shape[0] , -1])
    return images


@functools.partial(jax.jit, static_argnums = [2])    
def apply_image_masks_to_eigen(proj_eigen, image_masks, image_shape):
    proj_eigen = ftu.get_idft2(proj_eigen.reshape([*proj_eigen.shape[0:2], *image_shape]))
    proj_eigen = proj_eigen * image_masks
    proj_eigen = ftu.get_dft2(proj_eigen).reshape([*proj_eigen.shape[0:2], -1])
    return proj_eigen


# Compute y_i - P_i mu terms
@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10])    
def get_centered_images(images, mean, CTF_params, rotation_matrices, translations, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type  ):    
    translated_images = core.translate_images(images, translations, image_shape)
    centered_images = translated_images - get_projected_image(mean, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
    return centered_images

def check_mask(mask):
    no_mask = np.all(np.isclose(mask,1))
    if no_mask:
        logger.info("no mask used")
    return no_mask

batch_forward_model = jax.vmap(core.forward_model, in_axes = (0, None, None))

def get_projected_image(mean, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):
    slices = core.get_slices(mean, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)
    return slices

@functools.partial(jax.jit, static_argnums = [3,4,5,6,7,8])    
def batch_over_vol_forward_model(mean, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):
    
    batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    batch_CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    projected_mean =  batch_forward_model(mean, batch_CTF, batch_grid_pt_vec_ind_of_images) 
        
    return projected_mean

