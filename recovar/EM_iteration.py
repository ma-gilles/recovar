import functools
import jax
import jax.numpy as jnp
from recovar import core
import numpy as np
import logging
import itertools
from recovar import mask as mask_fn
from recovar import relion_functions    
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
batch_vol_slice_volume_by_map = jax.vmap(core.slice_volume_by_map, in_axes = (0, None, None, None, None), out_axes=1 )

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
        

    # Output is image_batch x vol_batch  x rots_batch x trans_batch
    return norm_res_squared


take_vmap = jax.vmap(lambda x, y, axis: jnp.take(x,y, axis), in_axes = (0, 0, None))
def batch_take(arr, indices, axis):
    # og_shape = arr.shape
    # indices_shape = indices.shape
    # arr # swap volume axis
    # arr = arr.swapaxes(1, 2)
    # zz = jnp.take(arr[0], indices, axis=-1)
    
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
    norm_res_squared = 0.5 * norm_res_squared

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
        
        translated_images = core.batch_trans_translate_images(image[None], translations[None], image_shape)
        summed_up_images = jnp.sum(translated_images * probabilities[...,None], axis = 2)

    return summed_up_images

sum_up_translations = jax.vmap(sum_up_translate_one_image, in_axes = (0,0,0,None, None))
sum_up_translations_shared_translations = jax.vmap(sum_up_translate_one_image, in_axes = (0,0,None,None, None))

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



# @functools.partial(jax.jit, static_argnums=[6,7,8,9,10,11])
# def compute_gaussian_residual_integral_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft" ):
#     # Everything should be stored as:?
#     # Rotations should be images_batch x vol_batch x rotation_batch x translation_batch
#     # assert(rotation_matrices.shape[0] == volumes.shape[0])
#     # assert((translations.shape[:-1] == rotation_batch.shape[0]).all())
#     # assert(rotation_matrices.shape[1] == images.shape[0])
#     # assert(translations.shape[0] == volumes.shape[0])
#     # assert(translations.shape[1] == images.shape[0])

#     # n_vols x rotations x image_size
#     projected_volumes = batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type)#.swapaxes(0,1)
#     

#     
#     # Broadcast CTF in volumes x rotations
#     projected_volumes = (projected_volumes * CTF_fun( CTF_params, image_shape, voxel_size)[:,None,None,:])#[...,None,:]
#     # Add axes for volumes and rotations

#     # # This seems much faster.
#     # Broacast over volumes x rotations
#     images /= jnp.sqrt(noise_variance)
#     projected_volumes /= jnp.sqrt(noise_variance)
#     # Output is image_batch x vol_batch  x rots_batch x trans_batch
#     return norm_res_squared

# def optimize_

def E_with_precompute(experiment_dataset, volume, rotations, translations, noise_variance, disc_type, image_indices = None, u = None, s = None):
    # I am not sure this is a reasonable way to be passing things around. 

    logger.info(f"starting precomp proj. Num rotations {rotations.shape[0]}, num translations {translations.shape[0]}. Total = {rotations.shape[0] * translations.shape[0]}")
    # Probably should stop storing rotations as matrices at some point.
    # batch_size = utils.get_num
    image_shape = experiment_dataset.image_shape
    image_size = experiment_dataset.image_size
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)
    use_heterogeneous = u is not None
    n_principal_components = u.shape[0] if use_heterogeneous else 0
    from recovar import utils

    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) * 10
    n_batches = utils.get_number_of_index_batch(n_rotations, batch_size)

    projections = np.zeros((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):    
        projections[rot_indices] = core.slice_volume_by_map(volume, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type)

    logger.info(f"done with precomp proj, batch size {batch_size}")
    projections = jnp.asarray(projections)
    logger.info(f"Allocating proj")

    # Compute \sum_i A_i^T y_i / sigma_i^2
    residuals = np.empty((n_images,  projections.shape[0], n_translations))

    dot_product_batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections)) / translations.shape[0] * 40
    dot_product_batch_size/= 4
    dot_product_batch_size = int(max(1, dot_product_batch_size))
    logger.info(f"Starting IP. Dot product batch size {dot_product_batch_size}. Remaing memory {gpu_memory - utils.get_size_in_gb(projections)}")
    utils.report_memory_device(logger=logger)
    # dot_product_batch_size = batch_size // translations.shape[0]
    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=dot_product_batch_size, subset_indices = image_indices)
    image_indices = np.arange(n_images) if image_indices is None else image_indices

    start_idx = 0
    for batch, _, indices in data_generator:
        # running_idx 
        # Only place where image mask is used ?
        end_idx = start_idx + len(indices)
        residuals[start_idx:end_idx] = compute_dot_products(projections, batch,
                            translations, experiment_dataset.CTF_params[indices], 
                            experiment_dataset.CTF_fun, noise_variance,
                            experiment_dataset.image_stack.process_images,
                            experiment_dataset.voxel_size, image_shape)
        start_idx  = end_idx

    if use_heterogeneous:
        u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype = np.complex64)
        # Compute all mean and principal component projections
        for rot_indices in utils.index_batch_iter(n_rotations, batch_size):    
            u_projections[rot_indices] = batch_vol_slice_volume_by_map(u, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type)

        logger.info(f"done with u_proj {batch_size}")
        data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=dot_product_batch_size, subset_indices = image_indices)
        
        # start_idx = 0
        # for batch, _, indices in data_generator:
        #     # running_idx 
        #     # Only place where image mask is used ?
        #     end_idx = start_idx + len(indices)
        #     # zz1, z_opt = compute_bHb_term(projections, u_projections, s, batch, translations, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.image_stack.process_images)
        #     # print(np.mean(z_opt[0, ..., 0], axis=-0))
            
        #     residuals[start_idx:end_idx] -= compute_bHb_terms(projections, u_projections, s, batch, translations, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.image_stack.process_images)
        #     start_idx = end_idx
        rotation_batch = rotations.shape[0]//10
        start_idx = 0
        for batch, _, indices in data_generator:
            batch = jnp.asarray(batch)
            end_idx = start_idx + len(indices)

            for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):
                # could just not backproject until the end
                # rot_indices = utils.get_batch_of_indices_arange(n_rotations, rotation_batch, k)
                # Hmmm this is a bit of a hack. Indexing is not what I wish it was
                rot_indices = np.array(rot_indices)
                residuals[start_idx:end_idx, rot_indices] -= compute_bHb_terms(projections[rot_indices], u_projections[rot_indices], s, batch, translations, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.image_stack.process_images)
                # Ft_y, Ft_ctf = sum_up_images_fixed_rots(batch, probabilities[start_idx:end_idx, rot_indices[0]:rot_indices[-1]+1], translations, rotations[rot_indices], experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.volume_shape, experiment_dataset.image_stack.process_images,  Ft_y = Ft_y, Ft_ctf = Ft_ctf)

            start_idx = end_idx



    projections = (jnp.abs(projections)**2).block_until_ready()

    logger.info(f"done with IP")
    utils.report_memory_device(logger=logger)
    # For the \|C_i Proj_j\|^2 term

    # norm_batch_size = batch_size // 3
    norm_batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections)) * 3
    
    # n_batches = utils.get_number_of_index_batch(n_images, norm_batch_size)
    start_idx = 0
    # for k in range(n_batches):
    # for indices in utils.subset_batch_iter(image_indices, norm_batch_size):
    #     end_idx = start_idx + len(indices)
    #     # indices = utils.get_batch_of_indices_arange(n_images, norm_batch_size, k)
    #     res = compute_CTFed_proj_norms(projections, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape)
    #     if k == n_batches - 1:
    #         res = res.block_until_ready()
    #     residuals[start_idx:end_idx] += np.array(res[...,None])
    #     start_idx = end_idx

    for array_indices, dataset_indices in utils.subset_and_indices_batch_iter(image_indices, norm_batch_size):
        # indices = utils.get_batch_of_indices_arange(n_images, norm_batch_size, k)
        res = compute_CTFed_proj_norms(projections, experiment_dataset.CTF_params[dataset_indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape)
        if array_indices[-1] == n_images - 1:
            res = res.block_until_ready()
        residuals[array_indices] += np.array(res[...,None])



    del projections
    logger.info(f"done with norms. Batch size {norm_batch_size}")


    n_batches = utils.get_number_of_index_batch(n_images, batch_size)
    start_idx = 0
    for array_indices, _ in utils.subset_and_indices_batch_iter(image_indices, batch_size):
        residuals[array_indices] = compute_probability_from_residual_normal_squared_one_image(residuals[array_indices])

    logger.info(f"done probs. Batch size {batch_size}")

    return residuals

def get_default_sgd_options():
    options = {}
    options['minibatch_size'] = 30
    options['steps_size'] = 'hess'
    options['mu'] = 0.9
    return 


DEBUG = True
from recovar import utils


def E_M_batches_2(experiment_dataset, state_obj, rotations, translations, disc_type, memory_to_use = 128, volume_mask = None):

    total_hidden = rotations.shape[0] * translations.shape[0]
    logger.info(f"starting precomp proj. Num rotations {rotations.shape[0]}, num translations {translations.shape[0]}. Total = {total_hidden}")
    n_images_batch = int(memory_to_use * 1e9 / ( total_hidden * 8  ))
    # If we count the allocated memor

    if n_images_batch < 1:
        n_images_batch = 1
        logger.warning(f"Memory to use is too small. Setting n_images_batch to {n_images_batch}. May run out of memory")
    logger.info(f"n_images_batch {n_images_batch}. Number of batches {int(np.ceil(experiment_dataset.n_units / n_images_batch))}")   
    
    if state_obj.name =='SGD':
        n_images_batch = state_obj.sgd_batchsize

    hard_assignment = np.empty(experiment_dataset.n_units, dtype = int)

    for big_image_batch in utils.index_batch_iter(experiment_dataset.n_units, n_images_batch): 

        probabilities = state_obj.E_step(experiment_dataset, rotations, translations, disc_type, big_image_batch)
        # hard_assignment[big_image_batch] = np.argmax(probabilities, axis = (-1, -2))
        hard_assignment[big_image_batch] = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)
        if np.isnan(probabilities).any():
            print(np.linalg.norm(state_obj.mean))
            import pdb; pdb.set_trace()
        state_obj.M_step(experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch)
    
    return state_obj, hard_assignment


def split_E_M_v2(experiment_datasets, state_objs, rotations, translations, disc_type, average_up_to_angstrom = None,  ):

    # Ft_y, Ft_CTF, H, B, projected_cov_lhs, projected_cov_rhs = tuple(6*[2 *[None]])
    # Ft_y = [None] * len(experiment_datasets)
    # Ft_y, Ft_CTF, Hs, Bs, projected_cov_lhs, projected_cov_rhs = 2 * [None], 2 * [None], 2 * [None], 2 * [None], 2 * [None], 2 * [None]
    # cov_cols = 2 * [None]
    hard_assignments = 2 * [None]
    for i, experiment_dataset in enumerate(experiment_datasets):
        state_objs[i], hard_assignments[i] = E_M_batches_2(experiment_dataset, state_objs[i], rotations, translations, disc_type)
        state_objs[i].finish_up_M_step(experiment_dataset, disc_type)

    ## Update prior and estimate resolution
    from recovar import regularization, locres
    # sgd_updates priors
    cryo = experiment_datasets[0]

    use_fsc_prior= state_objs[0].name == 'SGD'
    means = [state_obj.mean for state_obj in state_objs]
    if use_fsc_prior:
        mean_signal_variance, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, means[0], means[1], (state_objs[0].Ft_CTF + state_objs[i].Ft_CTF[1])/2, mean_signal_variance, frequency_shift = jnp.array([0,0,0]), upsampling_factor = 1)
    else:
        fsc = regularization.get_fsc_gpu(means[0], means[1], cryo.volume_shape, substract_shell_mean = False, frequency_shift = 0 )
        mean_avg = (means[0] + means[1])/2
        PS = regularization.average_over_shells(jnp.abs(mean_avg)**2, cryo.volume_shape)

        T = 4
        mean_signal_variance = T * 1/2 * utils.make_radial_image(PS, cryo.volume_shape, extend_last_frequency = True)
        
        mean_signal_variance += np.max(mean_signal_variance) * 1e-6
        # mean_signal_variance  = 1 /signal_variance

    from recovar import plot_utils
    plot_utils.plot_fsc(cryo, means[0], means[1])
    
    ##  Estimate noise level
    from recovar import noise
    # if heterogeneous:
    # This doesn't really make sense...

    for k in range(2):
        best_rotations, best_translations = hard_assignment_idx_to_pose(hard_assignments[k], rotations, translations)
        experiment_datasets[k].rotation_matrices = best_rotations
        experiment_datasets[k].translations = best_translations

    noise_from_res, _, _ = noise.get_average_residual_square_just_mean(cryo, None, means[0], 100, disc_type = 'linear_interp', subset_indices = np.arange(1000), subset_fn = None)
    noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)#, cryo.voxel_size)
    # In pixel units?
    current_pixel_res = locres.find_fsc_resol(fsc, threshold = 1/7)
    current_res = current_pixel_res / cryo.voxel_size
    # logger.info("Current resolution is", current_res, "pixel resolution: ", current_pixel_res)
    print("Current resolution is ", current_res, "pixel resolution: ", current_pixel_res)

    # [ state_obj.noise_variance for state_obj in state_objs]
    if state_objs[0].name == 'HeterogeneousEM':
        # Downsample to mean resolution
        valid_freqs = np.array(cryo.get_valid_frequency_indices(current_pixel_res))

        if state_objs[0].u is not None:
            for state_obj in state_objs:
                state_obj.u = state_obj.u * valid_freqs[None]
                state_obj.subspace = state_obj.subspace * valid_freqs[...,None]

        covariance_options = state_objs[0].covariance_options
        _, covariance_prior, _ = regularization.prior_iteration_relion_style_batch(state_objs[0].H, state_objs[1].H, state_objs[0].B, state_objs[1].B, np.zeros(state_objs[0].H.shape[0]),
        state_objs[0].covariance_prior, 
        covariance_options['substract_shell_mean'], 
        cryo.volume_shape, covariance_options['left_kernel'], 
        covariance_options['use_spherical_mask'],  covariance_options['grid_correct'],  None, covariance_options["prior_n_iterations"], covariance_options["downsample_from_fsc"])

        for k in range(2):
            state_objs[k].covariance_prior = covariance_prior

    if average_up_to_angstrom is not None:
        low_res_mask = cryo.get_valid_frequency_indices(average_up_to_angstrom)
        logger.info(f"Averaging halfmaps up to {average_up_to_angstrom} pixels")
        means = [np.array(mean) for mean in means ]
        # old_means = means[0].copy()
        means[0][low_res_mask] = (means[0][low_res_mask] + means[1][low_res_mask])/2
        means[1][low_res_mask] = means[0][low_res_mask]
        
    # Update objects
    for k in range(2):
        state_objs[k].noise_variance = noise_variance
        state_objs[k].mean_variance = mean_signal_variance
        state_objs[k].mean = means[k]

    return state_objs, current_pixel_res, hard_assignments

def hard_assignment_idx_to_pose(indices, rotation_grid, translation_grid):
    square_shape = (rotation_grid.shape[0], translation_grid.shape[0])
    maxpos_vect = np.column_stack(np.unravel_index(indices,square_shape))
    predicted_trans = translation_grid[maxpos_vect[:,1]]
    predicted_pose = rotation_grid[maxpos_vect[:,0]]
    return predicted_pose, predicted_trans

## Probably should implement these so we don't have to pass around so many arguments
class EMState():
    mean = None
    mean_variance = None
    noise_variance = None
    name = "EM"
    Ft_CTF = 0
    Ft_y = 0
    def __init__(self, mean, mean_variance, noise_variance):
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        return
    
    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        return probabilities
    
    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):
        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this
        return

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(experiment_dataset, self.Ft_CTF, self.Ft_y, tau = self.mean_variance, disc_type = disc_type).reshape(-1)
        return



class SGDState():
    mean = None
    mean_variance = None
    noise_variance = None
    update = 0
    name = "SGD"
    sgd_projection = lambda x : x
    sgd_batchsize = 100

    def __init__(self, mean, mean_variance, noise_variance):
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        return
    
    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch, iter, volume_mask = None):

        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        n_images_batch = len(big_image_batch)

        mean = self.mean
        mu = 0.9
        grad = 2 * ((Ft_CTF_this) * mean - Ft_y_this) *  experiment_dataset.n_images / n_images_batch + 2/ self.mean_variance * mean

        step = 1 / np.max(np.abs(Ft_CTF_this))
        self.update = mu * self.update + (1 - mu) * step * grad 
        if np.isnan(self.update).any() or np.isinf(self.update).any():
            print(np.linalg.norm(self.update))

        # import pdb; pdb.set_trace()
        if iter%10==0 and DEBUG:
            # print(idx)
            print('|dx| / |x|:', np.linalg.norm(self.update) / np.linalg.norm(mean))
            print('|prior|/ grad:', np.linalg.norm( 2/ self.mean_variance * mean) / np.linalg.norm(grad))
            print('|x|:', np.linalg.norm( mean))
            print('|dx|:', np.linalg.norm( self.update))
            # plot_utils.plot
            first = False
        mean -= self.update * 0.1#0.01 
        mean = self.sgd_projection(mean)

        std_multiplier = 10
        mean = np.clip(mean.real, -std_multiplier * np.sqrt(self.mean_variance), std_multiplier * np.sqrt(self.mean_variance)) + 1j * np.clip(mean.imag, -std_multiplier * np.sqrt(self.mean_variance), std_multiplier * np.sqrt(self.mean_variance))

        if np.isnan(mean).any() or np.isinf(mean).any() or np.isnan(np.linalg.norm(mean)) or np.isinf(np.linalg.norm(mean)):
            print('|dx| / |x|:', np.linalg.norm(self.update) / np.linalg.norm(mean))
            print('|prior|/ grad:', np.linalg.norm( 2/ self.mean_variance * mean) / np.linalg.norm(grad))
            print('|x|:', np.linalg.norm( mean))
            print('|dx|:', np.linalg.norm( self.update))

            print(np.linalg.norm(self.update))
            import pdb; pdb.set_trace()

        logger.warning("There is a necessary 0.1 that shouldn't be there")
        self.mean = mean

        # iter +=1
        return
        
    def finish_up_M_step(self, experiment_dataset, disc_type):
        # nothing
        return


class HeterogeneousEMState():

    # Mean stuff
    mean = None
    mean_prior = None
    Ft_y = 0
    Ft_CTF = 0 

    # Covariance stuff
    H = 0
    B = 0
    cov_cols = None
    covariance_prior = None
    covariance_options = None
    picked_frequency_indices = None

    # Projected covariance stuff
    projected_cov_lhs = 0
    projected_cov_rhs = 0
    subspace = None
    cov_cols = None

    # PCA stuff
    u = None
    s = None

    # Other stuff
    volume_mask = None
    noise_variance = None
    name = "HeterogeneousEM"


    def __init__(self, mean, mean_variance, noise_variance):
        self.grid_size = utils.guess_grid_size_from_vol_size(mean.size)
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        grid_size = utils.guess_grid_size_from_vol_size(mean.size)
        self.volume_mask = mask_fn.raised_cosine_mask(3 * [grid_size], grid_size//2 -3, grid_size//2, -1)  
        return
    


    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch, u = self.u, s = self.s)
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):

        ## Accumulate Ft_y and Ft_CTF
        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this

        ## Accumulate H, B, and covs
        H_this,B_this = compute_H_B(experiment_dataset, self.mean, probabilities, rotations, translations, self.noise_variance, None, self.picked_frequency_indices, big_image_batch, self.covariance_options['disc_type'])
        H_this = np.array(H_this)
        B_this = np.array(B_this)
        self.H += H_this
        self.B += B_this

        if self.subspace is not None:
            projected_cov_lhs_this, projected_cov_rhs_this = compute_projected_covariance_rhs_lhs(experiment_dataset, self.mean, self.subspace, rotations, translations, probabilities, None, self.noise_variance, disc_type_mean = self.covariance_options['disc_type'], disc_type_u = self.covariance_options['disc_type_u'], image_indices = big_image_batch)
            self.projected_cov_lhs += projected_cov_lhs_this
            self.projected_cov_rhs += projected_cov_rhs_this
        return 

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(experiment_dataset, self.Ft_CTF, self.Ft_y, tau = self.mean_variance, disc_type = disc_type).reshape(-1)

        if self.subspace is not None:
            projected_covar = solve_covariance(self.projected_cov_lhs, self.projected_cov_rhs)
            s, u_small = np.linalg.eigh(projected_covar)
            u_small =  np.fliplr(u_small)
            s = np.flip(s)
            self.u = (self.subspace @ u_small).T
            self.s = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)

        post_process_vmap = jax.vmap(relion_functions.post_process_from_filter_v2, in_axes = (0, 0, None, None, 0, None,None, None, None, None, None))
        
        self.cov_cols = post_process_vmap(self.H, self.B, experiment_dataset.volume_shape, 1, self.covariance_prior, self.covariance_options['left_kernel'], False, self.covariance_options['grid_correct'],  "square",  1, self.volume_mask ).reshape(self.H.shape[0], -1).T

        # basis,_ = principal_components.get_cov_svds(cov_col0, picked_frequency_indices)
        # spherical_mask = 
        memory_to_use = utils.get_gpu_memory_total()
        self.subspace, _ , _ = principal_components.randomized_real_svd_of_columns(self.cov_cols, self.picked_frequency_indices, None, experiment_dataset.volume_shape, 50, test_size=self.covariance_options['randomized_sketch_size'], gpu_memory_to_use=memory_to_use)
        # Keep only the first n_pcs_to_compute
        self.subspace = self.subspace[:,:self.covariance_options['n_pcs_to_compute']]
        return



@functools.partial(jax.jit, static_argnums=[5,8,9,10])
def sum_up_images_fixed_rots(batch, probabilities, translations, rotations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, volume_shape, process_images, Ft_y = 0, Ft_ctf = 0):

    # probabilities is shape n_images x n_rotations x n_translations
    assert(probabilities.shape[0] == batch.shape[0])
    assert(probabilities.shape[1] == rotations.shape[0])
    assert(probabilities.shape[2] == translations.shape[0])
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = batch.shape[0]
    n_shifted_images = n_images * n_translations

    # batch_size = utils.get_num
    # image_shape = experiment_dataset.image_shape
    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    batch = process_images(batch, apply_image_mask = False) * CTF / noise_variance
    # This should be a matvec 
    # We are computing \sum_i \sum  S_s y_i p_{i,s,j} over images j .... (Here the direction is fixed)
    # Will first form the S_s y_i
    #  P @ Y
    # Y is n_shifted_images x image_size
    # P is n_rotations x n_shifted_images
    shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])

    # Put n_rotations first, then reshape for mat-mat
    P = probabilities.swapaxes(0,1).reshape(n_rotations, n_shifted_images )
    summed_images = P @ shifted_images

    Ft_y = core.adjoint_slice_volume_by_trilinear(summed_images, rotations, image_shape, volume_shape, Ft_y)

    probabilites_summed_over_translations = jnp.sum(probabilities, axis = -1)

    CTF_probs =  probabilites_summed_over_translations.T @ (CTF**2 / noise_variance)
    # summed_CTF = probabilites_summed_over_translations @ CTF_probs

    Ft_ctf = core.adjoint_slice_volume_by_trilinear(CTF_probs, rotations, image_shape, volume_shape, Ft_ctf)

    return Ft_y, Ft_ctf

def M_with_precompute(experiment_dataset, probabilities, rotations, translations, noise_variance, disc_type, image_indices = None):

    logger.info(f"starting precomp proj. Num rotations {rotations.shape[0]}, num translations {translations.shape[0]}. Total = {rotations.shape[0] * translations.shape[0]}")
    # Probably should stop storing rotations as matrices at some point.
    projections = np.zeros((rotations.shape[0], experiment_dataset.image_size), dtype = np.complex64)

    # batch_size = utils.get_num
    image_shape = experiment_dataset.image_shape
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)
    from recovar import utils

    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) // translations.shape[0] * 20

    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)
    
    Ft_y, Ft_ctf = jnp.zeros((experiment_dataset.volume_size), dtype = experiment_dataset.dtype), jnp.zeros((experiment_dataset.volume_size), experiment_dataset.dtype)
    
    # n_rotation_batch = rotations.shape[0]//10
    mult = 5
    rotation_batch = rotations.shape[0]//mult
    logger.info(f"Starting sum up images. Batch size {batch_size}, rotation batch {rotation_batch}")
    start_idx = 0
    for batch, _, indices in data_generator:
        batch = jnp.asarray(batch)
        end_idx = start_idx + len(indices)

        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):
            # could just not backproject until the end
            # rot_indices = utils.get_batch_of_indices_arange(n_rotations, rotation_batch, k)
            # Hmmm this is a bit of a hack. Indexing is not what I wish it was
            Ft_y, Ft_ctf = sum_up_images_fixed_rots(batch, probabilities[start_idx:end_idx, rot_indices[0]:rot_indices[-1]+1], translations, rotations[rot_indices], experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.volume_shape, experiment_dataset.image_stack.process_images,  Ft_y = Ft_y, Ft_ctf = Ft_ctf)

        start_idx = end_idx

    return Ft_y, Ft_ctf



@functools.partial(jax.jit, static_argnums=[2,5])
def compute_CTFed_proj_norms(projections, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape):
    '''
    Computes  |C_i Proj_j|^2 for i,j by writing it as a mat-mat
    where C_i is CTF, S_s are shifts, and Proj_j are projections
    '''
    CTFs = CTF_fun( CTF_params, image_shape, voxel_size)**2 / noise_variance

    result = CTFs @ projections.T

    return result


def compute_UPLambdainvPU(u_projections, CTF, noise_variance):
    ## Note there is not Lambda^{-1} here.

    # Form H
    u_projections = u_projections.swapaxes(1,2)
    u_outer_projections = u_projections[...,None] @ jnp.conj(u_projections[...,None,:])
    #[:,:,None] @ jnp.conj(u_projections.T[:,None])
    u_outer_projections = u_outer_projections.reshape(*u_outer_projections.shape[:-1], -1)
    # Now mat vec with CTFs and whatnot
    CTF_squared = CTF**2 / noise_variance

    u_outer_projections = u_outer_projections.transpose(0,2,3,1)
    
    # Now matvec
    u_outer_projections =  (u_outer_projections @ CTF_squared.T).real

    # Now reshape back
    H = u_outer_projections.transpose(0, 3, 1, 2)
    return H



def compute_little_H_b(mean_projections, u_projections, s, batch, translations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, process_images):

    # u_projections is n_rotations x n_principal_components  x image_size

    # Often we would n_principal_components ~ 10 
    # n_rotations depends how much we can allocate at once, but hopefully 100s
    n_principal_components = u_projections.shape[1]
    n_rotations = u_projections.shape[0]
    n_images = batch.shape[0]
    n_translations = translations.shape[0]
    # n_shifted_images = n_images * n_translations
    # n_u_projections = n_principal_components * n_rotations
    # image_size = batch.shape[-1]

    # # Form H
    # u_projections = u_projections.swapaxes(1,2)
    # u_outer_projections = u_projections[...,None] @ jnp.conj(u_projections[...,None,:])
    # #[:,:,None] @ jnp.conj(u_projections.T[:,None])
    # u_outer_projections = u_outer_projections.reshape(*u_outer_projections.shape[:-1], -1)
    # # Now mat vec with CTFs and whatnot
    # CTF_squared = CTF_fun(CTF_params, image_shape, voxel_size)**2 / noise_variance

    # u_outer_projections = u_outer_projections.transpose(0,2,3,1)
    # 
    # # Now matvec
    # u_outer_projections =  u_outer_projections @ CTF_squared.T
    # # Now reshape back
    # H = u_outer_projections.transpose(0, 3, 1, 2)

    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    H = compute_UPLambdainvPU(u_projections, CTF, noise_variance)
    
    # These are the H matrices we seek
    
    H += jnp.diag(1/s)
    
    # print(1/s)

    # Form b    
    batch = process_images(batch, apply_image_mask = False) 
    batch *= CTF_fun( CTF_params, image_shape, voxel_size) / noise_variance

    b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, noise_variance, image_shape)

    # # Going to split b into two: 2 (y_i^* \noiseinv S_s C_i) P_j U and 2 \left( C_i P_j \mu \right)^* \noiseinv\left( C_i P_j U \right)
    # # First: 2 (y_i^* \noiseinv S_s C_i) P_j U
    # shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    # # Is reshape necessary?
    # shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])
    # b1 = jnp.conj(shifted_images) @ u_projections#.swapaxes(1,2)
    # 
    # b1 = b1.reshape(n_rotations, n_images, n_translations , n_principal_components)
    # # Should be size n_images x n_rotations x n_principal_components x n_translations 
    # # UGH. GROSS
    # # b1 is stored as rotations, images, principal_components, translations
    # b1 = b1.transpose(0,1,3,2) 

    # Getting lost in the indices... hopefully this is right?

    # Second: 2 \left( C_i P_j \mu \right)^* \noiseinv\left( C_i P_j U \right)
    # First, compute terms like P_j \mu * P_j U
    # u_projections_times_mu_projection = u_projections * jnp.conj(mean_projections[...,None])
    # b2 =  CTF_squared @ u_projections_times_mu_projection 
    # b = 2 * (- b1 +  b2[...,None] )
    # b = - 0.5 * b # To agree with definition of Gaussian integral
     
    # Reshape or vmap? the eternal question
    # b = b.transpose(n_images, n_rotations, n_principal_components, n_translations).reshape(n_images * n_rotations, n_principal_components )
    # b = b.transpose(0, 3, 2, 1).reshape(n_images * n_rotations, n_principal_components )
    return H, b

# diag_vmap = jax.vmap(jnp.diag, in_axes = 0, out_axes = 1)
batch_batch_diag = jax.vmap(jax.vmap(jnp.diag, in_axes = 0, out_axes = 0), in_axes = 0, out_axes = 0)

@functools.partial(jax.jit, static_argnums=[6,9,10])
def compute_bHb_terms(mean_projections, u_projections, s, batch, translations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, process_images):

    H,b = compute_little_H_b(mean_projections, u_projections, s, batch, translations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, process_images)

    # Compute bHb
    # Hinvb = jax.scipy.linalg.solve(H, b, assume_a = 'pos')
    # 
    # import pdb; pdb.set_trace()   
    H_chol, low = jax.scipy.linalg.cho_factor(H, lower = True)
    Hinvb = jax.scipy.linalg.cho_solve((H_chol, low), b, overwrite_b=False, check_finite=True)

    bHinvb = jnp.sum(jnp.conj(b) * Hinvb, axis =-2)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(batch_batch_diag(H_chol))), axis = -1)
    ## THings are off by a factor of 2 everywhere. YUCK. This is why the last 2
    half_inv_logdet = - 0.5 * log_det * 2
    # log_det_H2 =  jnp.log((jnp.linalg.det(H)))

    # This could be done more efficiently by solving Chol of H.
    log_det_H =  half_inv_logdet #jnp.log((1/jnp.linalg.det(H)))
    logger.warning(f"Make sure this is correct...")
    summed = bHinvb + log_det_H[...,None]
    # import pdb; pdb.set_trace()
    return summed.transpose(1,0,2) #, Hinvb # I think also need to compute det(H)??? Check

def compute_bLambdainvPU_terms(mean_projections, u_projections, invnoise_CTFed_images, translations, CTF, noise_variance, image_shape):

    n_principal_components = u_projections.shape[1]
    n_rotations = u_projections.shape[0]
    n_images = invnoise_CTFed_images.shape[0]
    n_translations = translations.shape[0]
    n_shifted_images = n_images * n_translations

    u_projections = u_projections.swapaxes(1,2)
    # Going to split b into two: 2 (y_i^* \noiseinv S_s C_i) P_j U and 2 \left( C_i P_j \mu \right)^* \noiseinv\left( C_i P_j U \right)
    # First: 2 (y_i^* \noiseinv S_s C_i) P_j U
    shifted_images = core.batch_trans_translate_images(invnoise_CTFed_images, jnp.repeat(translations[None], invnoise_CTFed_images.shape[0], axis=0), image_shape)
    # Is reshape necessary?
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])
    b1 = (jnp.conj(shifted_images) @ u_projections).real#.swapaxes(1,2)
    
    b1 = b1.reshape(n_rotations, n_images, n_translations , n_principal_components)
    # Should be size n_images x n_rotations x n_principal_components x n_translations 
    # UGH. GROSS
    # b1 is stored as rotations, images, principal_components, translations
    b1 = b1.transpose(0,1,3,2) 



    # shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    # # Is reshape necessary?
    # shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])
    # b1 = jnp.conj(shifted_images) @ u_projections#.swapaxes(1,2)
    
    # b1 = b1.reshape(n_rotations, n_images, n_translations , n_principal_components)
    # # Should be size n_images x n_rotations x n_principal_components x n_translations 
    # # UGH. GROSS
    # # b1 is stored as rotations, images, principal_components, translations
    # b1 = b1.transpose(0,1,3,2) 

    # Getting lost in the indices... hopefully this is right?

    # Second: 2 \left( C_i P_j \mu \right)^* \noiseinv\left( C_i P_j U \right)
    ## First, compute terms like P_j \mu * P_j U
    u_projections_times_mu_projection = u_projections * jnp.conj(mean_projections[...,None])
    b2 =  (( CTF**2 /noise_variance )  @ u_projections_times_mu_projection ).real
    b = 2 * (- b1 +  b2[...,None] )
    b = - 0.5 * b # To agree with definition of Gaussian integral
    return b #, Hinvb # I think also need to compute det(H)??? Check


@functools.partial(jax.jit, static_argnums=[4,6,8])
def compute_dot_products(projections, batch, translations, CTF_params, CTF_fun, noise_variance, process_images, voxel_size, image_shape):
    '''
    Computes -2 * y_i.T @ (S_s C_i * Proj_j) for i,j,s
    where C_i is CTF, S_s are shifts, and Proj_j are projections, y_i are batch (unprocessed)
    '''
    # proj_norm = jnp.linalg.norm(projections, axis=-1, keepdims=True)**2
    # Technically don't need do to this step. Maybe should delete at some point.

    batch = process_images(batch, apply_image_mask = False) 
    batch_norm = jnp.linalg.norm(batch / jnp.sqrt(noise_variance), axis = (-1), keepdims = True)**2
    # result

    batch *= CTF_fun( CTF_params, image_shape, voxel_size) / noise_variance

    result = jnp.empty((batch.shape[0], projections.shape[0], translations.shape[0]), dtype = jnp.float32)

    # One bad thing about the FFT shifting way is that is involve allocating size(projections) * num_images
    # This way relies a lot on never allocating things of that size, so that we can handle a ton of projections x images at once.
    # It's not very clear whether it's better to do. Compute all shifted images, then compute IP, or compute IP for each shift.
    # The latter is probably better for memory, but the former is probably better for speed. We probably want to do the latter.

    ## IMPLEMENATION 1
    ## I thought this would avoid allocating all the shifted images at once, but it doesn't seem to be the case.
    imp = 3
    if imp == 1:
        for i in range(translations.shape[0]):
            # Probably should put a better fn here for shfiting but oh well.
            shifted_batch = core.translate_images(batch, jnp.repeat(translations[i:i+1], batch.shape[0], axis=0), image_shape)

            result = result.at[:,:,i].set(-2 * (jnp.conj(shifted_batch) @ projections.T).real + batch_norm)
    
    ## IMPLEMENATION 2
    ## This doesn't work. I'm not sure if it should be any different than the typical for loop

    # def body_fun(i, result):
    #     shifted_batch = core.translate_images(batch, jnp.repeat(translations[i:i+1], batch.shape[0], axis=0), image_shape)

    #     result = result.at[:,:,i].set(-2 * (jnp.conj(shifted_batch) @ projections.T).real + batch_norm)

    # result = jax.lax.fori_loop(0, translations.shape[0], body_fun, result)

    # IMPLEMENTATION 3. Just one big matvec.
    # This is probably allocating far more memory than needed.
    elif imp == 3:
        shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
        n_shifted_images = np.prod(shifted_images.shape[:-1])
        result = -2 * (jnp.conj(shifted_images).reshape(n_shifted_images, shifted_images.shape[-1] ) @ projections.T).real 
        result = result.reshape(batch.shape[0], translations.shape[0], projections.shape[0]) + batch_norm[:,None]
        result = result.swapaxes(1,2)
        # May want to swap axes
    

    return result





# def compute_H_B_triangular(centered_images, CTF_val_on_grid_stacked, plane_coords_on_grid_stacked, rotation_matrices,  cov_noise, picked_freq_index, image_mask, image_shape, volume_size, right_kernel = "triangular", left_kernel = "triangular", kernel_width = 2, shared_label = False):
#     # print("Using kernel", right_kernel, left_kernel, kernel_width)

#     volume_shape = utils.guess_vol_shape_from_vol_size(volume_size)
#     picked_freq_coord = core.vec_indices_to_vol_indices(picked_freq_index, volume_shape)

#     # The image term
#     ctfed_images = centered_images * jnp.conj(CTF_val_on_grid_stacked)
    
#     # Between SPA and tomography, the only difference here is that we want to treat all images
#     # (which are assumed to be from same tilt series) as a single measurement. 

#     # Why the hell did I call this images_prod??? 
#     images_prod = covariance_core.sum_up_over_near_grid_points(ctfed_images, plane_coords_on_grid_stacked, picked_freq_coord, kernel = right_kernel, kernel_width = kernel_width)
    
#     if shared_label:
#         # I think this is literally the only change? ( a corresponding one below for lhs)
#         ## TODO: Make sure this is correct.
#         images_prod = jnp.repeat(jnp.sum(images_prod, axis=0, keepdims=True), images_prod.shape[0], axis=0)
#         logger.warning("SHARED LABEL! CHECK NOISE TERM IS CORRECT")
#         logger.warning("SHARED LABEL! CHECK NOISE TERM IS CORRECT")
#         logger.warning("SHARED LABEL! CHECK NOISE TERM IS CORRECT")

#     ctfed_images  *= jnp.conj(images_prod)[...,None]

#     # - noise term
#     ctfed_images -= compute_noise_term(plane_coords_on_grid_stacked, picked_freq_coord, CTF_val_on_grid_stacked, image_shape, image_mask, cov_noise, kernel = right_kernel, kernel_width = kernel_width)

#     # # TODO: put this in a function
#     # if left_kernel == "triangular":
#     #     rhs_summed_up = core.adjoint_slice_volume_by_trilinear(ctfed_images, rotation_matrices,image_shape, volume_shape )
#     # elif left_kernel == "square":
#     #     rhs_summed_up = core.adjoint_slice_volume_by_map(ctfed_images, rotation_matrices,image_shape, volume_shape , 'nearest')
#     # else:
#     #     raise ValueError("Kernel not implemented")
    
#     rhs_summed_up = adjoint_kernel_slice(ctfed_images, rotation_matrices, image_shape, volume_shape, left_kernel)

#     # lhs term 
#     ctf_squared = CTF_val_on_grid_stacked * jnp.conj(CTF_val_on_grid_stacked)

#     ctfs_prods = covariance_core.sum_up_over_near_grid_points(ctf_squared, plane_coords_on_grid_stacked, picked_freq_coord , kernel = right_kernel, kernel_width = kernel_width)

#     if shared_label:
#         ctfs_prods = jnp.repeat(jnp.sum(ctfs_prods, axis=0, keepdims=True), ctfs_prods.shape[0], axis=0) 

#     ctf_squared *= ctfs_prods[...,None]


#     # if left_kernel == "triangular":
#     #     lhs_summed_up = core.adjoint_slice_volume_by_trilinear(ctf_squared, rotation_matrices,image_shape, volume_shape )
#     # elif left_kernel == "square":
#     #     lhs_summed_up = core.adjoint_slice_volume_by_map(ctf_squared, rotation_matrices,image_shape, volume_shape, 'nearest' )
#     # else:
#     #     raise ValueError("Kernel not implemented")
#     lhs_summed_up = adjoint_kernel_slice(ctf_squared, rotation_matrices, image_shape, volume_shape, left_kernel)

#     return lhs_summed_up, rhs_summed_up


def compute_H_B(experiment_dataset, mean, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, image_indices, mean_disc):
    # Memory in here scales as O (batch_size )

    logger.warning("Not using mask in compute_H_B. Not implemented yet")
    # utils.report_memory_device()

    # volume_size = mean_estimate.size
    image_shape = experiment_dataset.image_shape
    image_size = experiment_dataset.image_size
    volume_shape = experiment_dataset.volume_shape
    volume_size = experiment_dataset.volume_size
    n_picked_indices = picked_frequency_indices.size
    n_rotations = rotations.shape[0]

    H = [jnp.zeros(volume_size, dtype = experiment_dataset.dtype_real )] * n_picked_indices
    B = [jnp.zeros(volume_size, dtype = experiment_dataset.dtype )] * n_picked_indices

    # H_B_fn = options["covariance_fn"]
    # f_jit = jax.jit(compute_H_B_triangular, static_argnums = [7,8,9,10,11,12])


    # if experiment_dataset.tilt_series_flag:
    #     assert "kernel" in H_B_fn, "Only kernel implemented for tilt series"

    # if options['disc_type'] == 'cubic':
    #     these_disc = 'cubic'
    #     from recovar import cryojax_map_coordinates
    #     mean_estimate = cryojax_map_coordinates.compute_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))
    # else:
    #     these_disc = 'linear_interp'


    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) * 10
    n_batches = utils.get_number_of_index_batch(n_rotations, batch_size)

    mean_projections = np.zeros((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):    
        mean_projections[rot_indices] = core.slice_volume_by_map(mean, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, mean_disc)

    # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) // translations.shape[0] * 20
    
    picked_freq_coords = core.vec_indices_to_vol_indices(picked_frequency_indices, volume_shape)

    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(mean_projections)) / translations.shape[0] * 1
    batch_size = int(max(1, batch_size))
    logger.info(f"Starting H_B, batch size {batch_size}. Remaing memory {gpu_memory - utils.get_size_in_gb(mean_projections)}")
    utils.report_memory_device(logger=logger)
    
    # Allocate this to GPU.
    mean_projections = jnp.asarray(mean_projections)
    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)
    rotation_batch = rotations.shape[0]//10

    start_idx =0 
    for images, _, indices in data_generator:
        end_idx = start_idx + len(indices)
        prob_batch = jnp.array(probabilities[start_idx:end_idx])
        shifted_CTFed_images, CTF = sum_up_images_fixed_rots_covariance_precompute(images, translations, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, experiment_dataset.voxel_size, experiment_dataset.image_shape, experiment_dataset.image_stack.process_images)
        del images
        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):

            gridpoints = core.batch_get_gridpoint_coords(
                rotations[rot_indices],
                image_shape, volume_shape )
            # There is a lot of compute to win by skipping rotations for which the right kernel will evaluate to zero.
            for (k, picked_freq_coord) in enumerate(picked_freq_coords):
                # picked_freq_coord = core.vec_indices_to_vol_indices(picked_freq_idx, volume_shape)
                H[k], B[k] = sum_up_images_fixed_rots_covariance_with_precompute(shifted_CTFed_images, mean_projections[np.array(rot_indices)], CTF, gridpoints, prob_batch[:,rot_indices], rotations[rot_indices],  noise_variance, image_shape, volume_shape, picked_freq_coord, H = H[k], B = B[k], right_kernel_width = 2, right_kernel = "triangular")

        start_idx = end_idx

        # del image_mask
        # del images, batch_CTF, batch_grid_pt_vec_ind_of_images
        
    # H = np.stack(H, axis =1)
    # B = np.stack(B, axis =1)
    return H, B


@functools.partial(jax.jit, static_argnums=[3,5,6])
def sum_up_images_fixed_rots_covariance_precompute(batch, translations, CTF_params, CTF_fun, voxel_size, image_shape, process_images):

    # probabilities is shape n_images x n_rotations x n_translations
    # assert(probabilities.shape[0] == batch.shape[0])
    # assert(probabilities.shape[1] == rotations.shape[0])
    # assert(probabilities.shape[2] == translations.shape[0])
    n_translations = translations.shape[0]
    n_images = batch.shape[0]
    n_shifted_images = n_images * n_translations

    # batch_size = utils.get_num
    # image_shape = experiment_dataset.image_shape
    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    batch = process_images(batch, apply_image_mask = False) * CTF 
    shifted_CTFed_images= core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)

    return shifted_CTFed_images, CTF


@functools.partial(jax.jit, static_argnums=[7,8,12,13])
def sum_up_images_fixed_rots_covariance_with_precompute(shifted_CTFed_images, mean_projections, CTF, gridpoints, probabilities, rotations,  noise_variance, image_shape, volume_shape, gridpoint_target, H = 0, B = 0, right_kernel_width = 2, right_kernel = "triangular"):

    # probabilities is shape n_images x n_rotations x n_translations
    # assert(probabilities.shape[0] == batch.shape[0])
    # assert(probabilities.shape[1] == rotations.shape[0])
    # assert(probabilities.shape[2] == translations.shape[0])
    n_rotations = rotations.shape[0]
    n_translations = shifted_CTFed_images.shape[1]
    n_images = shifted_CTFed_images.shape[0]
    n_shifted_images = n_images * n_translations
    image_size = shifted_CTFed_images.shape[-1]
    # batch_size = utils.get_num
    # image_shape = experiment_dataset.image_shape
    # CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    # batch = process_images(batch, apply_image_mask = False) * CTF / noise_variance
    # This should be a matvec 
    # We are computing \sum_i \sum  S_s y_i p_{i,s,j} over images j .... (Here the direction is fixed)
    # Will first form the S_s y_i
    #  P @ Y
    # Y is n_shifted_images x image_size
    # P is n_rotations x n_shifted_images
    # shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    # shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])

    # Put n_rotations first, then reshape for mat-mat
    # Alright...
    # P = probabilities.swapaxes(0,1).reshape(n_rotations, n_shifted_images )
    # Compute e_2
    # gridpoints = None## Compute this.

    from recovar import covariance_core
    # One kernel per rotation
    kernel_vals = covariance_core.evaluate_kernel_on_grid(gridpoints, gridpoint_target, kernel = right_kernel, kernel_width = right_kernel_width)
    
    # Kernel vals is size n_rotations x image_size

    e2_p1 = shifted_CTFed_images @ kernel_vals.T
    # Should reshape so that translation is last index
    # e2_p1 is size n_rotations x n_shifted_images 

    # want to compute kernelvals.^T_e2 (CTF_i *  u_proj). CTF_i * u _proj is n_rotations x n_images x image_size which is not what we want to allocate.
    # Can compute  instead C_i ^T (kernelvals * u_proj) . 
    # u_proj* kernel = kernel_vals * CTF

    e2_p2 = (CTF**2) @ (kernel_vals * mean_projections).T 
    e2 = e2_p1 - e2_p2[:,None,:]
    # e2 is size n_images x n_translations x n_rotations
    # Put it to n_images x n_rotations x n_translations
    e2 = e2.swapaxes(1,2)
    e2 = jnp.conj(e2)

    gamma_2 = probabilities * e2

    # Summed over translations
    gamma_2_summed_over_translations = jnp.sum(gamma_2, axis = -1)
    summed_CTF_squared_gamma2 = (CTF**2).T @ gamma_2_summed_over_translations
    summed_CTF_squared_gamma2 = summed_CTF_squared_gamma2.T
    #Piece 1
    before_adj_B2 = -summed_CTF_squared_gamma2 * mean_projections

    # Piece 2
    # gamma_2 is size n_images x n_rotations x n_translations
    # Need to swap axis to make a big matvec with shifted images
    gamma_2 = gamma_2.swapaxes(1,2).reshape(n_images * n_translations, n_rotations)
    shifted_CTFed_images = shifted_CTFed_images.reshape(n_images * n_translations, image_size)

    before_adj_B2 += gamma_2.T @ shifted_CTFed_images

    #Noise piece... No image mask here
    probabilties_summed_over_translations = jnp.sum(probabilities, axis = -1)


    CTF_squared_times_noise = (CTF**2 * noise_variance ).T @ probabilties_summed_over_translations #@ kernel_vals.T
    noise_piece = CTF_squared_times_noise.T * kernel_vals
    before_adj_B2 -= noise_piece

    ## Now adjoint
    B = core.adjoint_slice_volume_by_trilinear(before_adj_B2, rotations, image_shape, volume_shape, B)

    # Now for H
    CTF_squared = CTF**2
    CTF_squared_kernel_vals = kernel_vals @ CTF_squared.T
    # Size n_rotations x n_images
    gamma_3 = probabilties_summed_over_translations.T * CTF_squared_kernel_vals
    # This mat vec should integrate over rotation
    H_before_adj = gamma_3 @ CTF_squared

    H = core.adjoint_slice_volume_by_trilinear(H_before_adj, rotations, image_shape, volume_shape, H)
    return H, B





from recovar import covariance_estimation
def compute_projected_covariance(experiment_datasets, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, disc_type_u, image_indices = None):
    
    # experiment_dataset = experiment_datasets[0]

    # basis = basis.T.astype(experiment_dataset.dtype)
    # # Make sure variables used in every iteration are on gpu.
    # basis = jnp.asarray(basis)
    # # volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    # mean = jnp.array(mean).astype(experiment_dataset.dtype)

    # lhs =0
    # rhs =0 
    # # summed_batch_kron_cpu = jax.jit(summed_batch_kron, backend='cpu')
    # logger.info(f"batch size in compute_projected_covariance {batch_size}")

    # if disc_type_mean == 'cubic':
    #     mean = covariance_estimation.cryojax_map_coordinates.compute_spline_coefficients(mean.reshape(experiment_dataset.volume_shape))

    # if disc_type_u == 'cubic':
    #     basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)
    
    # n_rotations = rotations.shape[0]
    # n_principal_components = basis.shape[0]
    # image_size = experiment_dataset.image_size

    # u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype = np.complex64)
    # # Compute all mean and principal component projections
    # mean_projections = np.empty((rotations.shape[0], image_size), dtype = np.complex64)
    # for rot_indices in utils.index_batch_iter(n_rotations, batch_size): 
    #     mean_projections[rot_indices] = core.slice_volume_by_map(mean, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_mean)
    #     u_projections[rot_indices] = batch_vol_slice_volume_by_map(basis, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_u)

    
    # del basis, mean

    # logger.info(f"done with u_proj {batch_size}")
    
    # # batch_size = 100
    
    # change_device= False

    # for experiment_dataset in experiment_datasets:
    #     data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)
    #     start_idx = 0
    #     for batch, _, batch_image_ind in data_generator:
            
    #         end_idx = start_idx + len(batch_image_ind)
    #         lhs_this, rhs_this = reduce_covariance_est_inner(mean_projections, u_projections, probabilities[start_idx:end_idx], batch, translations, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, experiment_dataset.image_shape, experiment_dataset.image_stack.process_images)
    #         lhs += lhs_this
    #         rhs += rhs_this
            
    #         del lhs_this, rhs_this
    #         start_idx = end_idx

    #     # del lhs_this, rhs_this
    # # del basis
    # # Deallocate some memory?
    lhs, rhs = compute_projected_covariance_rhs_lhs(experiment_datasets, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, disc_type_mean, disc_type_u, image_indices = None)
    # Solve dense least squares?
    # def vec(X):
    #     return X.T.reshape(-1)

    # ## Inverse of vec function.
    # def unvec(x):
    #     n = np.sqrt(x.size).astype(int)
    #     return x.reshape(n,n).T
    
    # logger.info("end of covariance computation - before solve")
    # rhs = vec(rhs)

    # if change_device:
    #     rhs = jax.device_put(rhs, jax.devices("gpu")[0])
    #     lhs = jax.device_put(lhs, jax.devices("gpu")[0])
    # # lhs_this = jax.device_put(lhs_this, jax.devices("gpu")[0])

    # covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    # # covar = linalg.batch_linear_solver(lhs, rhs)
    # covar = unvec(covar)
    # logger.info("end of solve")
    covar = solve_covariance(lhs, rhs)

    return covar #, lhs, rhs


from recovar import covariance_estimation
def compute_projected_covariance_rhs_lhs(experiment_dataset, mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, disc_type_mean, disc_type_u, image_indices = None):
    
    # experiment_dataset = experiment_datasets[0]

    basis = basis.T.astype(experiment_dataset.dtype)
    # Make sure variables used in every iteration are on gpu.
    basis = jnp.asarray(basis)
    # volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    mean = jnp.array(mean).astype(experiment_dataset.dtype)

    lhs =0
    rhs =0 
    # summed_batch_kron_cpu = jax.jit(summed_batch_kron, backend='cpu')
    # logger.info(f"batch size in compute_projected_covariance {batch_size}")

    if disc_type_mean == 'cubic':
        mean = covariance_estimation.cryojax_map_coordinates.compute_spline_coefficients(mean.reshape(experiment_dataset.volume_shape))

    if disc_type_u == 'cubic':
        basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)
    
    n_rotations = rotations.shape[0]
    n_principal_components = basis.shape[0]
    image_size = experiment_dataset.image_size

    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total())

    u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype = np.complex64)
    # Compute all mean and principal component projections
    mean_projections = np.empty((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size): 
        mean_projections[rot_indices] = core.slice_volume_by_map(mean, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_mean)
        u_projections[rot_indices] = batch_vol_slice_volume_by_map(basis, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type_u)

    
    del basis, mean

    logger.info(f"done with u_proj {batch_size}")
    basis_size = u_projections.shape[1]

    # batch_size = 100
    rotation_batch = rotations.shape[0]//10

    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  (2*basis_size**4*8/1e9 + utils.get_size_in_gb(mean_projections[:rotation_batch])* ( 1 + basis_size**2) )
    batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, memory_left_over_after_kron_allocate) / translations.shape[0] * 1
    batch_size = int(max(1, batch_size))

    # batch_size = utils.get_embedding_batch_size(basis, experiment_dataset.image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )
    logger.info('batch size for projected covariance computation: ' + str(batch_size))

    # change_device= False
    rotation_batch = rotations.shape[0]//10

    # for experiment_dataset in experiment_datasets:
    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)
    start_idx = 0
    for batch, _, batch_image_ind in data_generator:

        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):
            # gridpoints = core.batch_get_gridpoint_coords(
            #     rotations[rot_indices],
            #     image_shape, volume_shape )

            end_idx = start_idx + len(batch_image_ind)
            lhs_this, rhs_this = reduce_covariance_est_inner(mean_projections[rot_indices], u_projections[rot_indices], probabilities[start_idx:end_idx][:,np.array(rot_indices)], batch, translations, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, experiment_dataset.image_shape, experiment_dataset.image_stack.process_images)
            lhs += lhs_this
            rhs += rhs_this
        
        del lhs_this, rhs_this
        start_idx = end_idx

    return lhs, rhs

def solve_covariance(lhs, rhs):
        # del lhs_this, rhs_this
    # del basis
    # Deallocate some memory?

    # Solve dense least squares?
    def vec(X):
        return X.T.reshape(-1)

    ## Inverse of vec function.
    def unvec(x):
        n = np.sqrt(x.size).astype(int)
        return x.reshape(n,n).T
    
    logger.info("end of covariance computation - before solve")
    rhs = vec(rhs)

    # if change_device:
    #     rhs = jax.device_put(rhs, jax.devices("gpu")[0])
    #     lhs = jax.device_put(lhs, jax.devices("gpu")[0])
    # lhs_this = jax.device_put(lhs_this, jax.devices("gpu")[0])

    covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    # covar = linalg.batch_linear_solver(lhs, rhs)
    covar = unvec(covar)
    logger.info("end of solve")

    return covar #, lhs, rhs




@functools.partial(jax.jit, static_argnums = [6,9,10])    
def reduce_covariance_est_inner(mean_projections, u_projections, probabilities, batch, translations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, process_images):

    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    # These are the H matrices we seek
    # H = H + jnp.diag(1/s)

    # Form b    
    batch = process_images(batch, apply_image_mask = False) 
    batch *= CTF_fun( CTF_params, image_shape, voxel_size) #/ noise_variance

    probabilities = probabilities.swapaxes(0,1)

    # This is size n_images x n_rotations x n_principal_components x n_translations 
    b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, jnp.ones_like(noise_variance), image_shape)
    # Reshape to n_rotations x n_images x n_translsations x n_principal
    b = b.swapaxes(-1,-2)
    # b = compute_bLambdainvPU_terms(mean_projections, u_projections, batch, translations, CTF, noise_variance, image_shape)
    b *= jnp.sqrt(probabilities[...,None])
    outer_products = covariance_estimation.summed_outer_products(b.reshape(-1, b.shape[-1]))

    probabilities_summed_over_translations = jnp.sum(probabilities, axis = -1)
    UALambdaAUs = compute_UPLambdainvPU(u_projections, CTF, 1/noise_variance)
    UALambdaAUs = jnp.sum( probabilities_summed_over_translations[...,None,None] * UALambdaAUs, axis=(0,1))

    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(CTF_params.dtype)

    H = compute_UPLambdainvPU(u_projections, CTF, jnp.ones_like(noise_variance))

    H *= jnp.sqrt(probabilities_summed_over_translations[...,None,None])
    H = H.reshape(-1, H.shape[-2],  H.shape[-1])
    lhs = jnp.sum(covariance_estimation.batch_kron(H, H), axis=(0))

    # if shared_label:
    #     AU_t_images = jnp.sum(AU_t_images, axis=0,keepdims=True)
    #     AU_t_AU = jnp.sum(AU_t_AU, axis=0,keepdims=True)

    
    # return AU_t_AU, rhs
    # Perhaps this should use: jax.lax.fori_loop. This is a lot of memory.
    # Or maybe jax.lax.reduce ?
    
    return lhs, rhs

from recovar import constants, principal_components

def estimate_principal_components_simple(experiment_dataset, mean, mean_signal_variance, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, batch_size, image_indices, disc_type_mean, covariance_options):
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options
    covariance_options
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

    ####
    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
    batch_size = utils.get_embedding_batch_size(basis, experiment_dataset.image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )
    logger.info('batch size for covariance computation: ' + str(batch_size))

    covariance = compute_projected_covariance([experiment_dataset], mean, basis, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, covariance_options['disc_type_u'], image_indices = None)
    ss, u = np.linalg.eigh(covariance)
    u =  np.fliplr(u)
    s = np.flip(ss)
    u = basis @ u 
    s = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)
    return u, s


def estimate_principal_components_halfset(cryos, means, mean_signal_variance, cov_prior, probabilities, rotations, translations, noise_variance,  volume_mask, picked_frequency_indices, batch_size, image_indices, disc_type_mean, covariance_options):
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options
    covariance_options

    gpu_memory = utils.report_gpu_memory()
    volume_shape = cryos[0].volume_shape
    Hs, Bs = 2*[None], 2*[None]
    for cryo_idx, cryo in enumerate(cryos):
        Hs[cryo_idx], Bs[cryo_idx] = compute_H_B(cryo, means[cryo_idx], probabilities[cryo_idx], rotations, translations, noise_variance, volume_mask, picked_frequency_indices, image_indices, disc_type_mean)
    
    volume_noise_var = None
    volume_mask = None
    _, covariance_prior, covariance_fscs = principal_components.compute_covariance_regularization_relion_style(Hs, Bs, mean_signal_variance, picked_frequency_indices, volume_noise_var, volume_mask, cryos[0].volume_shape,  gpu_memory, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, options = covariance_options)

    from recovar import relion_functions

    cov_cols = 2 * [None]
    us = 2 * [None]; ss = 2 * [None]
    for cryo_idx, cryo in enumerate(cryos):
        cov_cols[cryo_idx] = relion_functions.post_process_from_filter_v2(Hs[cryo_idx], Bs[cryo_idx], volume_shape, volume_upsampling_factor = 1, tau = covariance_prior, kernel = covariance_options['left_kernel'], use_spherical_mask = covariance_options['use_spherical_mask'], grid_correct = covariance_options['grid_correct'], gridding_correct = "square", kernel_width = 1, volume_mask = volume_mask )

        # options['substract_shell_mean'], 
        # volume_shape, options['left_kernel'], 
        # options['use_spherical_mask'],  options['grid_correct'],  volume_mask, options["prior_n_iterations"], options["downsample_from_fsc"])

        # logger.info(f"image batch size: {batch_size}")
        # logger.info(f"volume batch size: {utils.get_vol_batch_size(cryo.grid_size, gpu_memory)}")
        # logger.info(f"column batch size: {utils.get_column_batch_size(cryo.grid_size, gpu_memory)}")
        vol_batch_size = utils.get_vol_batch_size(cryo.grid_size, gpu_memory)
        orthog_cov_cols,_ = principal_components.get_cov_svds(cov_cols[cryo_idx], picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, gpu_memory, False, covariance_options['randomized_sketch_size'])

        basis_size = cov_cols[cryo_idx].shape[0]
        memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
        batch_size = utils.get_embedding_batch_size(orthog_cov_cols, cryo.image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )
        logger.info('batch size for covariance computation: ' + str(batch_size))

        covariance = compute_projected_covariance([cryo], means[cryo_idx], orthog_cov_cols, rotations, translations, probabilities, volume_mask, noise_variance, batch_size, disc_type_mean, covariance_options['disc_type_u'], image_indices = None)
        ss, u = np.linalg.eigh(covariance)
        u =  np.fliplr(u)
        s = np.flip(ss)
        us[cryo_idx] = orthog_cov_cols @ u 
        ss[cryo_idx] = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)


    return us, ss




def estimate_principal_components(cryos, options,  means, mean_signal_variance, cov_noise, volume_mask,
                                dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use,
                                noise_model,  
                                covariance_options = None, variance_estimate = None):
    
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options

    volume_shape = cryos[0].volume_shape
    vol_batch_size = utils.get_vol_batch_size(cryos[0].grid_size, gpu_memory_to_use)

    # Different way of sampling columns: 
    # - from low to high frequencies
    # This is the way it was done in the original code. 
    # - Highest SNR columns, computed by lhs of mean estimation. May want to not take frequencies that are too similar
    # - Highest variance columns. Also want want to diversify.
    # For the last one, could also batch by doing randomized-Cholesky like choice

    # if covariance_options['column_sampling_scheme'] == 'low_freqs':
    #     from recovar import covariance_core
    #     volume_shape = cryos[0].volume_shape
    #     if cryos[0].grid_size == 16:
    #         picked_frequencies = np.arange(cryos[0].volume_size) 
    #     else:
    #         picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = covariance_options['column_radius'], use_half = True))
    # elif covariance_options['column_sampling_scheme'] == 'high_snr' or covariance_options['column_sampling_scheme'] == 'high_lhs' or covariance_options['column_sampling_scheme'] == 'high_snr_p' or covariance_options['column_sampling_scheme'] =='high_snr_from_var_est':
    #     from recovar import regularization
    #     upsampling_factor = np.round((means['lhs'].size / cryos[0].volume_size)**(1/3)).astype(int)
    #     upsampled_volume_shape = tuple(upsampling_factor * np.array(volume_shape))
    #     lhs = regularization.downsample_lhs(means['lhs'].reshape(upsampled_volume_shape), volume_shape, upsampling_factor = upsampling_factor).reshape(-1)
    #     # At low freqs, signal variance decays as ~1/rad^2

    #     dist = (ftu.get_grid_of_radial_distances(volume_shape)+1)**2
    #     if covariance_options['column_sampling_scheme'] == 'high_snr':
    #         lhs = lhs / dist.reshape(-1)
    #     if covariance_options['column_sampling_scheme'] == 'high_snr_p':
    #         lhs = lhs * means['prior']
    #     if covariance_options['column_sampling_scheme'] == 'high_snr_from_var_est':
    #         if variance_estimate is None:
    #             raise ValueError("variance_estimate must be provided")
    #         lhs = lhs * variance_estimate

    #     if covariance_options['randomize_column_sampling']:
    #         picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.randomized_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])
    #     else:
    #         picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.greedy_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])

    #     logger.info(f"Largest frequency computed: {np.max(np.abs(picked_frequencies_in_frequencies_format))}")
    #     if np.max(np.abs(picked_frequencies_in_frequencies_format)) > cryos[0].grid_size//2-1:
    #         logger.warning("Largest frequency computed is larger than grid size//2-1. This may cause big issues in SVD. This probably means variance estimates were wrong")
    #     # print("chosen cols", picked_frequencies_in_frequencies_format.T)
    #     
    # else:
    #     raise NotImplementedError('unrecognized column sampling scheme')
    

    covariance_cols, picked_frequencies, column_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(cryos, means, mean_signal_variance, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory_to_use, noise_model, covariance_options, picked_frequencies)
    logger.info("memory after covariance estimation")
    utils.report_memory_device(logger=logger)
    

    # First approximation of eigenvalue decomposition
    u,s = get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, options['ignore_zero_frequency'], covariance_options['randomized_sketch_size'])
    
    if not options['keep_intermediate']:
        for key in covariance_cols.keys():
            covariance_cols[key] = None
    image_cov_noise = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].image_shape))

    u['rescaled'], s['rescaled'] = pca_by_projected_covariance(cryos, u['real'], means['combined'], image_cov_noise, dilated_volume_mask, disc_type = covariance_options['disc_type'], disc_type_u = covariance_options['disc_type_u'], gpu_memory_to_use= gpu_memory_to_use, use_mask = covariance_options['mask_images_in_proj'], parallel_analysis = False ,ignore_zero_frequency = False, n_pcs_to_compute = covariance_options['n_pcs_to_compute'])

    if not options['keep_intermediate']:
        u['real'] = None
            
    return u, s, covariance_cols, picked_frequencies, column_fscs


# def pca_by_projected_covariance(cryos, basis, mean, noise_variance, volume_mask, disc_type , disc_type_u, gpu_memory_to_use= 40, use_mask = True, parallel_analysis = False ,ignore_zero_frequency = False, n_pcs_to_compute = -1):

#     # basis_size = basis.shape[-1]
#     basis_size = n_pcs_to_compute
#     basis = basis[:,:basis_size]

#     ####
#     memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
#     batch_size = utils.get_embedding_batch_size(basis, cryos[0].image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )

#     logger.info('batch size for covariance computation: ' + str(batch_size))

#     covariance = covariance_estimation.compute_projected_covariance(cryos, mean, basis, volume_mask, noise_variance, batch_size,  disc_type, disc_type_u, parallel_analysis = parallel_analysis, do_mask_images = use_mask )

#     ss, u = np.linalg.eigh(covariance)
#     u =  np.fliplr(u)
#     s = np.flip(ss)
#     u = basis @ u 

#     s = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)
 
#     return u , s



def compute_regularized_covariance_columns(cryos, means, mean_signal_variance, cov_noise, volume_mask, dilated_volume_mask, gpu_memory, noise_model, options, picked_frequencies):

    cryo = cryos[0]
    volume_shape = cryos[0].volume_shape

    # These options should probably be left as is.
    mask_ls = dilated_volume_mask
    mask_final = volume_mask
    # substract_shell_mean = False 
    # shift_fsc = False
    keep_intermediate = False
    image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)

    utils.report_memory_device(logger = logger)
    disc_type = 'nearest'
    Hs, Bs = compute_both_H_B(cryos, means, mask_ls, picked_frequencies, gpu_memory, image_noise_var, disc_type, parallel_analysis = False, options = options)
    st_time = time.time()
    volume_noise_var = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].volume_shape))
    covariance_cols = {}

    logger.info("using new covariance reg fn")
    utils.report_memory_device(logger = logger)

    covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(Hs, Bs, 1/mean_signal_variance, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, options = options)
    covariance_cols["est_mask"] = covariance_cols["est_mask"].T
    del Hs, Bs
    logger.info("after reg fn")
    
    utils.report_memory_device(logger = logger)

    return covariance_cols, picked_frequencies, np.asarray(fscs)




# def E_M_batches(experiment_dataset, mean, rotations, translations, noise_variance, disc_type, memory_to_use = 128, u = None, s = None, subspace = None, heterogenous = False, volume_mask = None, picked_frequency_indices= None, covariance_options = None, sgd = False, sgd_stepsize = 1e-3, sgd_batchsize = 100, mean_signal_variance = None, sgd_update = 0, sgd_projection = None):

#     total_hidden = rotations.shape[0] * translations.shape[0]
#     logger.info(f"starting precomp proj. Num rotations {rotations.shape[0]}, num translations {translations.shape[0]}. Total = {total_hidden}")
#     n_images_batch = int(memory_to_use * 1e9 / ( total_hidden * 8  ))
#     # If we count the allocated memor

#     if n_images_batch < 1:
#         n_images_batch = 1
#         logger.warning(f"Memory to use is too small. Setting n_images_batch to {n_images_batch}. May run out of memory")
#     logger.info(f"n_images_batch {n_images_batch}. Number of batches {int(np.ceil(experiment_dataset.n_units / n_images_batch))}")   
    
#     if sgd:
#         n_images_batch = sgd_batchsize

#     Ft_y, Ft_CTF = 0, 0
#     H,B = 0,0
#     projected_cov_lhs, projected_cov_rhs = 0, 0
#     sgd_iter = 0
#     first= True
#     idx =0 
#     hard_assignment = np.empty(experiment_dataset.n_units)

#     for big_image_batch in utils.index_batch_iter(experiment_dataset.n_units, n_images_batch): 

#         probabilities = E_with_precompute(experiment_dataset, mean, rotations, translations, noise_variance, disc_type, big_image_batch, u = u, s = s)
#         # hard_assignment[big_image_batch] = np.argmax(probabilities, axis = (-1, -2))
#         hard_assignment[big_image_batch] = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)
#         if np.isnan(probabilities).any():
#             print(np.linalg.norm(mean))
#             import pdb; pdb.set_trace()
#         Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, noise_variance, disc_type, big_image_batch)


#         if sgd:
#             mu = 0.9
#             grad = 2 * ((Ft_CTF_this) * mean - Ft_y_this) *  experiment_dataset.n_images / n_images_batch + 2/ mean_signal_variance * mean

#             step = 1 / np.max(np.abs(Ft_CTF_this))
#             sgd_update = mu * sgd_update + (1 - mu) * step * grad 
#             if np.isnan(sgd_update).any() or np.isinf(sgd_update).any():
#                 print(np.linalg.norm(sgd_update))

#             # import pdb; pdb.set_trace()
#             if idx%10==0 and DEBUG:
#                 print(idx)
#                 print('|dx| / |x|:', np.linalg.norm(sgd_update) / np.linalg.norm(mean))
#                 print('|prior|/ grad:', np.linalg.norm( 2/ mean_signal_variance * mean) / np.linalg.norm(grad))
#                 print('|x|:', np.linalg.norm( mean))
#                 print('|dx|:', np.linalg.norm( sgd_update))
#                 # plot_utils.plot
#                 first = False
#             mean -= sgd_update * 0.1#0.01 

#             if sgd_projection is not None:
#                 mean = sgd_projection(mean)

#             std_multiplier = 10
#             mean = np.clip(mean.real, -std_multiplier * np.sqrt(mean_signal_variance), std_multiplier * np.sqrt(mean_signal_variance)) + 1j * np.clip(mean.imag, -std_multiplier * np.sqrt(mean_signal_variance), std_multiplier * np.sqrt(mean_signal_variance))

#             if np.isnan(mean).any() or np.isinf(mean).any() or np.isnan(np.linalg.norm(mean)) or np.isinf(np.linalg.norm(mean)):
#                 print('|dx| / |x|:', np.linalg.norm(sgd_update) / np.linalg.norm(mean))
#                 print('|prior|/ grad:', np.linalg.norm( 2/ mean_signal_variance * mean) / np.linalg.norm(grad))
#                 print('|x|:', np.linalg.norm( mean))
#                 print('|dx|:', np.linalg.norm( sgd_update))

#                 print(np.linalg.norm(sgd_update))
#                 import pdb; pdb.set_trace()

#             logger.warning("There is a necessary 0.1 that shouldn't be there")

#             idx +=1
#             # mean *= experiment_dataset.get_valid_frequency_indices(5)
#         else:
#             Ft_y += Ft_y_this
#             Ft_CTF += Ft_CTF_this

#         if heterogenous:
            
#             ## Accumulate H, B, and covs
#             H_this,B_this = compute_H_B(experiment_dataset, mean, probabilities, rotations, translations, noise_variance, volume_mask, picked_frequency_indices, big_image_batch, covariance_options['disc_type'])
#             H_this = np.array(H_this)
#             B_this = np.array(B_this)
#             H += H_this
#             B += B_this

#             if subspace is not None:
#                 projected_cov_lhs_this, projected_cov_rhs_this = compute_projected_covariance_rhs_lhs(experiment_dataset, mean, subspace, rotations, translations, probabilities, volume_mask, noise_variance, disc_type_mean = covariance_options['disc_type'], disc_type_u = covariance_options['disc_type_u'], image_indices = big_image_batch)
#                 projected_cov_lhs += projected_cov_lhs_this
#                 projected_cov_rhs += projected_cov_rhs_this

#         ## Aculumate projected covariance_matrix

#         del probabilities

#     if sgd:
#         return mean, sgd_update, hard_assignment

#     if heterogenous:
#         # sgd_update U, s ? No need to store projected_cov_lhs/projected_cov_rhs

#         return Ft_y, Ft_CTF, H, B, projected_cov_lhs, projected_cov_rhs, hard_assignment
    
#     return Ft_y, Ft_CTF, hard_assignment





# from recovar import relion_functions

# def split_E_M(experiment_datasets, means, mean_signal_variance, rotations, translations, noise_variance, disc_type, heterogeneous = False, us = None, ss = None, covariance_signal_variance = None, bases = None, sgd=False, sgd_updates = None, average_up_to_angstrom = None, sgd_batchsize = 100, sgd_projection = None, covariance_options = None, picked_frequency_indices = None):

#     # Ft_y, Ft_CTF, H, B, projected_cov_lhs, projected_cov_rhs = tuple(6*[2 *[None]])
#     # Ft_y = [None] * len(experiment_datasets)
#     Ft_y, Ft_CTF, Hs, Bs, projected_cov_lhs, projected_cov_rhs = 2 * [None], 2 * [None], 2 * [None], 2 * [None], 2 * [None], 2 * [None]
#     cov_cols = 2 * [None]
#     hard_assignments = 2 * [None]
#     for i, experiment_dataset in enumerate(experiment_datasets):
#         if sgd:
#             means[i], sgd_updates[i], hard_assignments[i] = E_M_batches(experiment_dataset, means[i], rotations, translations, noise_variance, disc_type, memory_to_use = 128, u = None, s = None, subspace = None, heterogenous = False, volume_mask = None, picked_frequency_indices= None, covariance_options = None, sgd = True, sgd_batchsize = sgd_batchsize, mean_signal_variance = mean_signal_variance, sgd_update = sgd_updates[i], sgd_projection = sgd_projection)
#         elif heterogeneous:
#             Ft_y[i], Ft_CTF[i], Hs[i], Bs[i], projected_cov_lhs[i], projected_cov_rhs[i], hard_assignments[i] = E_M_batches(experiment_dataset, means[i], rotations, translations, noise_variance, disc_type, memory_to_use = 128,  u = us[i], s = ss[i], subspace = bases[i], heterogenous = heterogeneous, volume_mask = None, picked_frequency_indices= picked_frequency_indices, covariance_options = covariance_options, mean_signal_variance = mean_signal_variance)

#             means[i] = relion_functions.post_process_from_filter(experiment_dataset, Ft_CTF[i], Ft_y[i], tau = mean_signal_variance, disc_type = disc_type).reshape(-1)
#         else:
#             Ft_y[i], Ft_CTF[i], hard_assignments[i] = E_M_batches(experiment_dataset, means[i], rotations, translations, noise_variance, disc_type, memory_to_use = 128, u = None, s = None, subspace = None, heterogenous = False, volume_mask = None, picked_frequency_indices= None, options = None, sgd = sgd, sgd_batchsize = sgd_batchsize, mean_signal_variance = mean_signal_variance, sgd_update = None, sgd_projection = sgd_projection, covariance_options = covariance_options)

#             means[i] = relion_functions.post_process_from_filter(experiment_dataset, Ft_CTF[i], Ft_y[i], tau = mean_signal_variance, disc_type = disc_type).reshape(-1)

#             # cov_cols[i] = relion_functions.post_process_from_filter_v2(Hs[i], Bs[i], experiment_dataset.volume_shape, volume_upsampling_factor = 1, tau = covariance_signal_variance, kernel = covariance_options['left_kernel'], use_spherical_mask = True, grid_correct = covariance_options['grid_correct'], gridding_correct = "square", kernel_width = 1, volume_mask = None )

#             # plt.figure()
#             # plt.imshow(experiment_dataset.get_proj(Ft_y[i]))
#             # plt.show()

#             # plt.figure()
#             # plt.imshow(experiment_dataset.get_proj(Ft_CTF[i]))
#             # plt.show()

#             # # import pdb; pdb.set_trace()
#             # zz = Ft_y[i] / ( Ft_CTF[i] + 1 / mean_signal_variance)
#             # means[i] = relion_functions.post_process_from_filter(experiment_dataset, Ft_CTF[i], Ft_y[i], tau = mean_signal_variance, disc_type = disc_type).reshape(-1)
#             # plt.figure()
#             # plt.imshow(experiment_dataset.get_proj(means[i]))
#             # plt.show()

#             # plt.figure()
#             # plt.imshow(experiment_dataset.get_proj(zz)); plt.show()
#             # plt.show()
#             # import pdb; pdb.set_trace()


#         ## sgd_update...
#         if heterogeneous:

#             if bases[i] is not None:
#                 projected_covar = solve_covariance(projected_cov_lhs[i], projected_cov_rhs[i])
#                 s, u = np.linalg.eigh(projected_covar)
#                 u =  np.fliplr(u)
#                 s = np.flip(s)
#                 us[i] = (bases[i] @ u).T
#                 ss[i] = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)
#                 # us[i] = us[i].T

#             post_process_vmap = jax.vmap(relion_functions.post_process_from_filter_v2, in_axes = (0, 0, None, None, 0, None,None, None, None, None, None))
            
#             # cov_cols[i] = post_process_vmap(Hs[i], Bs[i], experiment_dataset.volume_shape, volume_upsampling_factor = 1, tau = covariance_signal_variance, kernel = covariance_options['left_kernel'], use_spherical_mask = True, grid_correct = covariance_options['grid_correct'], gridding_correct = "square", kernel_width = 1, volume_mask = None )

#             cov_cols[i] = post_process_vmap(Hs[i], Bs[i], experiment_dataset.volume_shape, 1, covariance_signal_variance, covariance_options['left_kernel'], True, covariance_options['grid_correct'],  "square",  1, None ).reshape(Hs[i].shape[0], -1).T

#             # basis,_ = principal_components.get_cov_svds(cov_col0, picked_frequency_indices)
#             # spherical_mask = 
#             memory_to_use = utils.get_gpu_memory_total()
#             bases[i], _ , _ = principal_components.randomized_real_svd_of_columns(cov_cols[i], picked_frequency_indices, None, experiment_dataset.volume_shape, 50, test_size=covariance_options['randomized_sketch_size'], gpu_memory_to_use=memory_to_use)
#             # Keep only the first n_pcs_to_compute
#             bases[i] = bases[i][:,:covariance_options['n_pcs_to_compute']]



#     ## Update prior and estimate resolution
#     from recovar import regularization, locres
#     # sgd_updates priors
#     cryo = experiment_datasets[0]
#     use_fsc_prior= not sgd

#     if use_fsc_prior:
#         mean_signal_variance, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, means[0], means[1], (Ft_CTF[0] + Ft_CTF[1])/2, mean_signal_variance, frequency_shift = jnp.array([0,0,0]), upsampling_factor = 1)
#     else:
#         fsc = regularization.get_fsc_gpu(means[0], means[1], cryo.volume_shape, substract_shell_mean = False, frequency_shift = 0 )
#         mean_avg = (means[0] + means[1])/2
#         PS = regularization.average_over_shells(jnp.abs(mean_avg)**2, cryo.volume_shape)

#         T = 4
#         mean_signal_variance = T * 1/2 * utils.make_radial_image(PS, cryo.volume_shape, extend_last_frequency = True)
        
#         mean_signal_variance += np.max(mean_signal_variance) * 1e-6
#         # mean_signal_variance  = 1 /signal_variance

#     from recovar import plot_utils
#     plot_utils.plot_fsc(cryo, means[0], means[1])
    
#     ##  Estimate noise level
#     from recovar import noise
#     # if heterogeneous:
#     # This doesn't really make sense...
#     noise_from_res, _, _ = noise.get_average_residual_square_just_mean(cryo, None, means[0], 100, disc_type = 'linear_interp', subset_indices = np.arange(1000), subset_fn = None)
#     noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)#, cryo.voxel_size)
#     # In pixel units?
#     current_pixel_res = locres.find_fsc_resol(fsc, threshold = 1/7)
#     current_res = current_pixel_res / cryo.voxel_size
#     # logger.info("Current resolution is", current_res, "pixel resolution: ", current_pixel_res)
#     print("Current resolution is ", current_res, "pixel resolution: ", current_pixel_res)

#     if heterogeneous:
#         # Downsample to mean resolution
#         valid_freqs = np.array(cryo.get_valid_frequency_indices(current_pixel_res))
#         if us[0] is not None:
#             us = [u * valid_freqs[None] for u in us]
#         if bases[0] is not None:
#             bases = [basis * valid_freqs[...,None] for basis in bases]
#         # Update covariance prior


#         # H0, H1, B0, B1, frequency_shift, init_regularization, substract_shell_mean, volume_shape, kernel = 'triangular', use_spherical_mask = True, grid_correct = True, volume_mask = None, prior_iterations = 3, downsample_from_fsc_flag = False
#         _, covariance_signal_variance, _ = regularization.prior_iteration_relion_style_batch(Hs[0], Hs[1], Bs[0], Bs[1], np.zeros(Hs[0].shape[0]),
#         covariance_signal_variance, 
#         covariance_options['substract_shell_mean'], 
#         cryo.volume_shape, covariance_options['left_kernel'], 
#         covariance_options['use_spherical_mask'],  covariance_options['grid_correct'],  None, covariance_options["prior_n_iterations"], covariance_options["downsample_from_fsc"])

#     # 
#     if average_up_to_angstrom is not None:
#         low_res_mask = cryo.get_valid_frequency_indices(average_up_to_angstrom)
#         logger.info(f"Averaging halfmaps up to {3/cryo.voxel_size} resolution")
#         means = [np.array(mean) for mean in means ]
#         old_means = means[0].copy()

#         means[0][low_res_mask] = (means[0][low_res_mask] + means[1][low_res_mask])/2
#         means[1][low_res_mask] = means[0][low_res_mask]
        

#     if heterogeneous:
#         return means, mean_signal_variance, current_pixel_res, noise_variance, sgd_updates, us, ss, bases, covariance_signal_variance, hard_assignments

#     return means, mean_signal_variance, current_pixel_res, noise_variance, sgd_updates, hard_assignments



        # probabilities = E_with_precompute(experiment_dataset, mean, rotations, translations, noise_variance, disc_type, big_image_batch, u = u, s = s)

## Probably should implement these so we don't have to pass around so many arguments
# class EMState():
#     mean = None
#     mean_variance = None
#     noise_variance = None
#     name = "EM"
#     Ft_CTF = 0
#     Ft_y = 0
#     def __init__(self, mean, mean_variance, noise_variance):
#         self.mean = mean
#         self.mean_variance = mean_variance
#         self.noise_variance = noise_variance
#         return
    
#     def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
#         probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch)
#         return probabilities
    
#     def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch, volume_mask = None):
#         Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
#         self.Ft_y += Ft_y_this
#         self.Ft_CTF += Ft_CTF_this
#         return
    