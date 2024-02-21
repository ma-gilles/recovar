import functools
import jax
import jax.numpy as jnp
from recovar import core
import numpy as np

# TODO: Should it be residual of masked?
# Residual will be 4 dimensional
# volumes_batch x images_batch x pose_batch  
# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])   

def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    indices = translations + image_shape[0]//2
    vec_indices = indices[...,0] * image_shape[1] + indices[...,1]
    return vec_indices

NORM_FFT = "backward"

@functools.partial(jax.jit, static_argnums=[6,7,8,9,10,11])
def compute_residuals_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft" ):
    # Rotations should be vol_batch x images_batch x rotation_batch x 10
    # assert(rotation_matrices.shape[0] == volumes.shape[0])
    # assert((translations.shape[:-1] == rotation_batch.shape[0]).all())
    # assert(rotation_matrices.shape[1] == images.shape[0])
    # assert(translations.shape[0] == volumes.shape[0])
    # assert(translations.shape[1] == images.shape[0])

    # n_vols x rotations x image_size
    projected_volumes = core.batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type).swapaxes(0,1)

    # Broadcast CTF in volumes x rotations
    projected_volumes = (projected_volumes * CTF_fun( CTF_params, image_shape, voxel_size)[:,None,None,:])#[...,None,:]
    # Add axes for volumes and rotations

    # # This seems much faster.
    # Broacast over volumes x rotations
    if translation_fn == "fft":
        images /= jnp.sqrt(noise_variance)
        projected_volumes /= jnp.sqrt(noise_variance)

        # projected_volumes = (projected_volumes * CTF_fun( CTF_params, limage_shape, voxel_size)[None,:,None,:])[...,None,:]

        proj_volume_norm = jnp.linalg.norm(projected_volumes, axis = (-1), keepdims = True)**2

        ## 
        # dots = norm_squared_residuals_from_ft(projected_volumes, images, image_shape) 
        # image_size = np.prod(image_shape)

        # if NORM_FFT != "ortho":
        #     dots = dots * image_size
        # translations_indices = translations_to_indices(translations, image_shape)
        # dots_chosen = batch_take(dots, translations_indices, axis = -1)
        # norm_res_squared = proj_volume_norm - 2 * dots_chosen + jnp.linalg.norm(images, axis = (-1), keepdims = True)[:,None,None]**2

        # Alittle bit of memory saving:
        projected_volumes = norm_squared_residuals_from_ft(projected_volumes, images, image_shape) 
        image_size = np.prod(image_shape)

        if NORM_FFT != "ortho":
            projected_volumes = projected_volumes * image_size
        translations_indices = translations_to_indices(translations, image_shape)
        dots_chosen = batch_take(projected_volumes, translations_indices, axis = -1)
        norm_res_squared = proj_volume_norm - 2 * dots_chosen #+ jnp.linalg.norm(images, axis = (-1), keepdims = True)[:,None,None]**2

        # import pdb; pdb.set_trace()

        # # This seems much faster.
        # projected_volumes = proj_volume_norm**2 -2 * norm_squared_residuals_from_ft(projected_volumes, images, image_shape) - jnp.linalg.norm(images, axis = (-1), keepdims = True)**2

        # # norm_res_squared = projected_volumes[...,:translations.size//2]

        # # import pdb; pdb.set_trace()
        # image_size = np.prod(image_shape)
        # dots = norm_squared_residuals_from_ft(projected_volumes, images, image_shape) 
        # if NORM_FFT != "ortho":
        #     dots = dots * image_size
        
        
        # x1 = dots[0,0,0,0].reshape(image_shape).real
        # x2 = jnp.dot(jnp.conj(projected_volumes[0,0,0,0]), images[0]).real
        
        # # dots3 = 
        # translations_indices = translations_to_indices(translations, image_shape)
        # dots_chosen = batch_take(dots, translations_indices, axis = -1)

        # x1.reshape(-1)
        # # import pdb; pdb.set_trace()
        # translated_images = core.batch_trans_translate_images(images, translations, image_shape)[None,:, None]

        # norm_res_squared2 = jnp.linalg.norm((projected_volumes - translated_images) / jnp.sqrt(noise_variance), axis = (-1))**2
        # import pdb; pdb.set_trace()

    else:
        # add axis for translations
        projected_volumes = projected_volumes[...,None,:]
        translated_images = core.batch_trans_translate_images(images, translations, image_shape)[None,:, None]
        norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images) / jnp.sqrt(noise_variance), axis = (-1))**2


    # norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images) / jnp.sqrt(noise_variance), axis = (-1))**2
    # Output is vol_batch x image_batch x rots_batch x trans_batch
    return norm_res_squared



take_vmap = jax.vmap(lambda x, y: jnp.take(x,y, -1), in_axes = (0, 0))
def batch_take(arr, indices, axis):
    og_shape = arr.shape
    indices_shape = indices.shape
    return take_vmap(arr.reshape(-1, og_shape[-1]), indices.reshape(-1, indices_shape[-1])).reshape(indices_shape)



# Cross image
def cross_image(im1, im2):
   xx = jnp.fft.get_idft2(jnp.conj(im1) * jnp.conj(im2))
   return xx #jax.scipy.signal.fftconvolve(im1, im2[::-1,::-1], mode='same')

cross_one_to_many_images = jax.vmap(cross_image, in_axes = (None, 0))

from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
def crosscorr_from_ft(many_images, one_image, image_shape):
    return ftu.get_idft2( jnp.conj(one_image.reshape(1, *image_shape)) * (many_images.reshape(-1, *image_shape)))#.reshape(-1, *image_shape)).reshape(-1, *image_shape)

def norm_squared_residuals_from_ft_one_image(many_images, one_image, image_shape):
    many_images_of_shape = many_images.shape
    many_images = many_images.reshape(-1, many_images.shape[-1])
    many_images = crosscorr_from_ft(many_images, one_image, image_shape)
    many_images = many_images.reshape(many_images_of_shape)
    return many_images

norm_squared_residuals_from_ft = jax.vmap(norm_squared_residuals_from_ft_one_image, in_axes = (0, 0, None))

    # 
    # # Guess I could do a 2x speed up here for FFTing only one
    # corr = cross_one_to_many_images(im1, im2)

    # # For a single image, and many projections
    # return


def compute_probability_from_residual_normal_squared_one_image(norm_res_squared):
    norm_res_squared -= jnp.min(norm_res_squared)
    exp_res = jnp.exp(- norm_res_squared)
    summed_exp = jnp.sum(exp_res)
    return exp_res / summed_exp

compute_probability_from_residual_normal_squared = jax.vmap(compute_probability_from_residual_normal_squared_one_image)


def compute_adj_slice(one_image, rotation_matrices, translations, CTF_params, probabilities):
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    one_image_summed  = jnp.sum(batch_translate(one_image * CTF * probabilities, translations))
    indices = core.get_nearest_gridpoint_indices(rotation_matrix, image_shape, volume_shape, grid_size)
    volume = core.summed_adjoint_slice_by_nearest(volume_size, one_image_summed, indices)
    return volume
