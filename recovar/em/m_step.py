import functools
import logging
import numpy as np
import jax
import jax.numpy as jnp
from recovar import core
import recovar.fourier_transform_utils as fourier_transform_utils
from .sampling import translations_to_indices
from .core import VOL_AXIS
logger = logging.getLogger(__name__)

def sum_up_translate_one_image(image, probabilities, translations, image_shape, translation_fn = "fft"):
    if translation_fn == "fft":
        image_size = np.prod(image_shape)
        images_probs = jnp.zeros( (*probabilities.shape[:-1], image_size), dtype = probabilities.dtype)
        translations_indices = translations_to_indices(translations, image_shape)
        # This allows for duplicates which may or may not be good?
        images_probs = images_probs.at[...,translations_indices].add(probabilities)
        images_probs = fourier_transform_utils.get_dft2(images_probs.reshape(*images_probs.shape[:-1], *image_shape))
        summed_up_images = (image) * (images_probs.reshape(*images_probs.shape[:-2], np.prod(image_shape)))
    else:  
        
        translated_images = core.batch_trans_translate_images(image[None], translations[None], image_shape)
        summed_up_images = jnp.sum(translated_images * probabilities[...,None], axis = 2)

    return summed_up_images

sum_up_translations = jax.vmap(sum_up_translate_one_image, in_axes = (0,0,0,None, None))
sum_up_translations_shared_translations = jax.vmap(sum_up_translate_one_image, in_axes = (0,0,None,None, None))



@functools.partial(jax.jit, static_argnums=[7,8,9,10,11])
def backproject_one_image(probabilities, images_i, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun, translation_fn = "fft" ):
    
    # Probability image
    # images3 = sum_up_translations(images, probabilities, translations, image_shape, translation_fn)
    
    # images2 = sum_up_translations(images, probabilities, translations, image_shape, 'nofft')
    # print(jnp.linalg.norm(images3 - images2)/jnp.linalg.norm(images))
    # # print(jnp.linalg.norm(images3 - images2, axis = (0,1,2) )/jnp.linalg.norm(images, axis = (0,1,2) ))
    # kk=0
    # plt.imshow(fourier_transform_utils.get_idft2((images2-images3)[0,0,kk].reshape(image_shape)).real); plt.colorbar(); plt.title(f'diff k {kk}'); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images3)[0,0,kk].reshape(image_shape)).real); plt.title(f'fft k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images2)[0,0,kk].reshape(image_shape)).real);  plt.title(f'nofft k {kk}');  plt.colorbar(); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images)[0].reshape(image_shape)).real); plt.title(f'raw k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images2)[0,0,kk].reshape(image_shape)).real/ fourier_transform_utils.get_idft2((images3)[0,0,kk].reshape(image_shape)).real);  plt.title(f'ratio k {kk}');  plt.colorbar(); plt.show()


    # kk=1
    # plt.imshow(fourier_transform_utils.get_idft2((images2-images3)[0,0,kk].reshape(image_shape)).real); plt.colorbar(); plt.title(f'diff k {kk}'); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images3)[0,0,kk].reshape(image_shape)).real); plt.title(f'fft k {kk}'); plt.colorbar(); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images2)[0,0,kk].reshape(image_shape)).real);  plt.title(f'nofft k {kk}');  plt.colorbar(); plt.show()
    # plt.imshow(fourier_transform_utils.get_idft2((images)[0].reshape(image_shape)).real); plt.title(f'raw k {kk}'); plt.colorbar(); plt.show()
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
    if n_rotations <= 0:
        raise ValueError("M_with_precompute requires at least one rotation")
    if n_translations <= 0:
        raise ValueError("M_with_precompute requires at least one translation")
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)
    from recovar import utils

    gpu_memory = utils.get_gpu_memory_total()
    batch_size = max(1, (utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) // translations.shape[0]) * 20)

    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)
    
    Ft_y, Ft_ctf = jnp.zeros((experiment_dataset.volume_size), dtype = experiment_dataset.dtype), jnp.zeros((experiment_dataset.volume_size), experiment_dataset.dtype)
    
    # n_rotation_batch = rotations.shape[0]//10
    mult = 5
    rotation_batch = max(1, rotations.shape[0] // mult)
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
