import functools
import jax
import jax.numpy as jnp
from recovar import core
import numpy as np
import logging

logger = logging.getLogger(__name__)


IMAGE_AXIS=0
VOL_AXIS=1
ROT_AXIS=2
TRANS_AXIS=3

# TODO: Should it be residual of masked?
# Residual will be 4 dimensional
# volumes_batch x images_batch x pose_batch  
# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])   

@functools.partial(jax.jit, static_argnums=[1])
def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    # Does this not work?
    indices = translations + image_shape[0]//2
    vec_indices = indices[...,1] * image_shape[1] + indices[...,0]
    # logger.warning("not sure that this is working as intended")
    return vec_indices


# def translations_to_indices2(translations, image_shape):
#     # Assumes that translations are integers
#     indices = translations + image_shape[0]//2
#     vec_indices = indices[...,0] + image_shape[1] * indices[...,1]
#     return vec_indices

NORM_FFT = "backward"

# batch volumes
batch_vol_rot_slice_volume_by_map = jax.vmap(core.slice_volume_by_map, in_axes = (0, VOL_AXIS, None, None, None), out_axes=1 )

# TODO: Should it be residual of masked?
# Residual will be 4 dimensional
# volumes_batch x images_batch x rotations_batch x translations_batch x  
# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
# def compute_residuals_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun ):
    

#     assert(rotation_matrices.shape[0] == volumes.shape[0])
#     assert(rotation_matrices.shape[1] == images.shape[0])

#     assert(translations.shape[0] == volumes.shape[0])
#     assert(translations.shape[1] == images.shape[0])


#     # n_vols x rotations x image_size
#     projected_volumes = batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type)
#     projected_volumes = projected_volumes * CTF_fun( CTF_params, image_shape, voxel_size)

#     translated_images = translate_images(images, translations, image_shape)

#     norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images) / jnp.sqrt(noise_variance), axis = (-1))
#     return norm_res_squared



@functools.partial(jax.jit, static_argnums=[6,7,8,9,10,11])
def compute_residuals_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft" ):
    # Everything should be stored as:?
    # Rotations should be images_batch x vol_batch x rotation_batch x translation_batch
    # assert(rotation_matrices.shape[0] == volumes.shape[0])
    # assert((translations.shape[:-1] == rotation_batch.shape[0]).all())
    # assert(rotation_matrices.shape[1] == images.shape[0])
    # assert(translations.shape[0] == volumes.shape[0])
    # assert(translations.shape[1] == images.shape[0])

    # n_vols x rotations x image_size
    projected_volumes = batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type)#.swapaxes(0,1)
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # Broadcast CTF in volumes x rotations
    projected_volumes = (projected_volumes * CTF_fun( CTF_params, image_shape, voxel_size)[:,None,None,:])#[...,None,:]
    # Add axes for volumes and rotations

    # # This seems much faster.
    # Broacast over volumes x rotations
    images /= jnp.sqrt(noise_variance)
    projected_volumes /= jnp.sqrt(noise_variance)

    if translation_fn == "fft":

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
        # print(translations_indices[:,None].shape)
        dots_chosen = batch_take(projected_volumes, translations_indices, axis = -1)


        norm_res_squared = proj_volume_norm - 2 * dots_chosen.real #+ jnp.linalg.norm(images, axis = (-1), keepdims = True)[:,None,None]**2

        ## Note that this does not do anything but it is good to to have it match the other implementation

        norm_res_squared += jnp.linalg.norm(images, axis = (-1), keepdims = True)[:,None,None]**2
    else:
        # add axis for translations
        projected_volumes = projected_volumes[...,None,:]
        translated_images = core.batch_trans_translate_images(images, translations, image_shape)[:,None, None]
        norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images), axis = (-1))**2
        # import pdb; pdb.set_trace()

    # Output is image_batch x vol_batch  x rots_batch x trans_batch
    return norm_res_squared


take_vmap = jax.vmap(lambda x, y, axis: jnp.take(x,y, axis), in_axes = (0, 0, None))
def batch_take(arr, indices, axis):
    # og_shape = arr.shape
    # indices_shape = indices.shape
    # arr # swap volume axis
    # arr = arr.swapaxes(1, 2)
    # zz = jnp.take(arr[0], indices, axis=-1)
    # import pdb; pdb.set_trace()
    return take_vmap(arr, indices, axis)#.reshape(indices_shape)



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


@functools.partial(jax.jit)
def compute_probability_from_residual_normal_squared_one_image(norm_res_squared):
    all_axis_but_first = tuple(range(1, norm_res_squared.ndim))
    norm_res_squared -= jnp.min(norm_res_squared, axis= all_axis_but_first, keepdims=True)
    exp_res = jnp.exp(- norm_res_squared)
    summed_exp = jnp.sum(exp_res, axis = all_axis_but_first, keepdims=True)
    return exp_res / summed_exp

compute_probability_from_residual_normal_squared = jax.vmap(compute_probability_from_residual_normal_squared_one_image)

import matplotlib.pyplot as plt

def sum_up_translate_one_image(image, probabilities, translations, image_shape, translation_fn = "fft"):

    if translation_fn == "fft":
        image_size = np.prod(image_shape)
        images_probs = jnp.zeros( (*probabilities.shape[:-1], image_size), dtype = probabilities.dtype)
        translations_indices = translations_to_indices(translations, image_shape)
        # This allows for duplicates which may or may not be good?
        images_probs = images_probs.at[...,translations_indices].add(probabilities)
        images_probs = ftu.get_dft2(images_probs.reshape(*images_probs.shape[:-1], *image_shape))
        summed_up_images = (image) * (images_probs.reshape(*images_probs.shape[:-2], np.prod(image_shape)))
    else:  
        # import pdb; pdb.set_trace()
        translated_images = core.batch_trans_translate_images(image[None], translations[None], image_shape)
        summed_up_images = jnp.sum(translated_images * probabilities[...,None], axis = 2)

    return summed_up_images

sum_up_translations = jax.vmap(sum_up_translate_one_image, in_axes = (0,0,0,None, None))
# def sum_up_translations(images, probabilities, translations, image_shape, translation_fn = "fft"):

#     if translation_fn == "fft":
#         # Compute sum of images by a convolution 
#         image_size = np.prod(image_shape)
#         images_probs = jnp.zeros( [translations.shape[-2] , image_size])
#         translations_indices = translations_to_indices2(translations, image_shape)
#         images_probs = images_probs.at[translations_indices].set(probabilities)
#         images_probs = ftu.get_dft2(images_probs.reshape(image_shape))
#         summed_up_images = crosscorr_from_ft(images_probs, images_probs, image_shape)
#     else:
#         translated_images = core.batch_trans_translate_images(images, translations, image_shape)[None,:, None] 
#         # Summed up?
#         summed_up_images = jnp.sum(translated_images * probabilities, axis = -1)
#         # Multiply by probs
#         # assert(False)

#     return summed_up_images

@functools.partial(jax.jit, static_argnums=[7,8,9,10,11])
def backproject_one_image(probabilities, images_i, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft" ):
    
    # Probability image
    # images3 = sum_up_translations(images, probabilities, translations, image_shape, translation_fn)
    
    # images2 = sum_up_translations(images, probabilities, translations, image_shape, 'nofft')
    # print(jnp.linalg.norm(images3 - images2)/jnp.linalg.norm(images))
    # # print(jnp.linalg.norm(images3 - images2, axis = (0,1,2) )/jnp.linalg.norm(images, axis = (0,1,2) ))
    # kk=0
    # plt.imshow(ftu.get_idft2((images2-images3)[0,0,kk].reshape(image_shape)).real); plt.colorbar(); plt.title(f'diff k {kk}'); plt.show()
    # plt.imshow(ftu.get_idft2((images3)[0,0,kk].reshape(image_shape)).real); plt.title(f'fft k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(ftu.get_idft2((images2)[0,0,kk].reshape(image_shape)).real);  plt.title(f'nofft k {kk}');  plt.colorbar(); plt.show()
    # plt.imshow(ftu.get_idft2((images)[0].reshape(image_shape)).real); plt.title(f'raw k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(ftu.get_idft2((images2)[0,0,kk].reshape(image_shape)).real/ ftu.get_idft2((images3)[0,0,kk].reshape(image_shape)).real);  plt.title(f'ratio k {kk}');  plt.colorbar(); plt.show()


    # kk=1
    # plt.imshow(ftu.get_idft2((images2-images3)[0,0,kk].reshape(image_shape)).real); plt.colorbar(); plt.title(f'diff k {kk}'); plt.show()
    # plt.imshow(ftu.get_idft2((images3)[0,0,kk].reshape(image_shape)).real); plt.title(f'fft k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(ftu.get_idft2((images2)[0,0,kk].reshape(image_shape)).real);  plt.title(f'nofft k {kk}');  plt.colorbar(); plt.show()
    # plt.imshow(ftu.get_idft2((images)[0].reshape(image_shape)).real); plt.title(f'raw k {kk}'); plt.colorbar(); plt.show()
    images = sum_up_translations(images_i, probabilities, translations, image_shape, translation_fn)
    # images = sum_up_translations(images_i, probabilities, translations, image_shape, translation_fn)
    # images = sum_up_translations(images_i, probabilities, translations, image_shape, translation_fn)
    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    # Add volume and rotation axis (CTF vixed over those)
    images *= (CTF[:,None,None] / noise_variance)
    # Ft = F transpose which is probably a confusing name

    # grid_coords, _ = core.rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    # grid_points, weights = core.get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    # grid_vec_indices = core.vol_indices_to_vec_indices(grid_points, volume_shape)
    # Ft_y = core.adjoint_slice_volume_by_trilinear_from_weights(images, grid_vec_indices, weights, volume_shape, None)

    Ft_y = batch_vol_adjoint_slice_volume(images, rotation_matrices, image_shape, volume_shape, None)
    # Ft_y = batch_vol_adjoint_slice_volume(images+0.01, rotation_matrices, image_shape, volume_shape, None)
    # Ft_y = batch_vol_adjoint_slice_volume(images+0.02, rotation_matrices, image_shape, volume_shape, None)


    # Add image axis (not batched image axis, the actual pixel axis)
    probabilites_summed_over_translations = jnp.sum(probabilities, axis = -1)[...,None]
    # Add volume and rotation axis (CTF vixed over those)
    CTF_probs = (CTF**2 / noise_variance)[:,None,None] * probabilites_summed_over_translations
    Ft_ctf = batch_vol_adjoint_slice_volume(CTF_probs, rotation_matrices, image_shape, volume_shape, None)
    # Ft_ctf = core.adjoint_slice_volume_by_trilinear_from_weights(CTF_probs, grid_vec_indices, weights, volume_shape, None)



    return Ft_y, Ft_ctf

# batch_vol_adjoint_slice_volume = jax.vmap(core.adjoint_slice_volume_by_map, in_axes = (VOL_AXIS, VOL_AXIS, None, None, None), out_axes=0 )
batch_vol_adjoint_slice_volume = jax.vmap(core.adjoint_slice_volume_by_trilinear, in_axes = (VOL_AXIS, VOL_AXIS, None, None, None), out_axes=0 )



# def EM_iteration_image_batch(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft"):

#     # E - iter
#     residuals = compute_residuals_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn)
#     probabilities = compute_probability_from_residual_normal_squared_one_image(residuals)
#     summed_images = sum_up_translate_one_image(image, probabilities, translations, image_shape, translation_fn )


#     return 