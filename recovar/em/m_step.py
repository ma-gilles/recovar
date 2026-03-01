"""M-step: volume update via weighted backprojection."""

import functools
import logging
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as fourier_transform_utils
from .sampling import translations_to_indices
from .core import VOL_AXIS
logger = logging.getLogger(__name__)

def sum_up_translate_one_image(image, probabilities, translations, image_shape, translation_fn = "fft"):
    if translation_fn == "fft":
        image_size = np.prod(image_shape)
        images_probs = jnp.zeros( (*probabilities.shape[:-1], image_size), dtype = probabilities.dtype)
        translations_indices = translations_to_indices(translations, image_shape)
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
    images = sum_up_translations(images_i, probabilities, translations, image_shape, translation_fn)
    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    images *= (CTF[:,None,None] / noise_variance)

    images_half = fourier_transform_utils.full_image_to_half_image(images, image_shape)
    Ft_y = batch_vol_adjoint_slice_volume_half(images_half, rotation_matrices, image_shape, volume_shape, None)

    probabilites_summed_over_translations = jnp.sum(probabilities, axis = -1)[...,None]
    CTF_probs = (CTF**2 / noise_variance)[:,None,None] * probabilites_summed_over_translations
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, image_shape)
    Ft_ctf = batch_vol_adjoint_slice_volume_half(CTF_probs_half, rotation_matrices, image_shape, volume_shape, None)

    return Ft_y, Ft_ctf


batch_vol_adjoint_slice_volume = jax.vmap(core.adjoint_slice_volume_by_trilinear, in_axes = (VOL_AXIS, VOL_AXIS, None, None, None), out_axes=0 )
batch_vol_adjoint_slice_volume_half = jax.vmap(core.adjoint_slice_volume_by_trilinear_from_half_images, in_axes = (VOL_AXIS, VOL_AXIS, None, None, None), out_axes=0 )


# ============================================================================
# Equinox-based M-step API
# ============================================================================

@eqx.filter_jit
def backproject_one_image_eqx(config: ForwardModelConfig, probabilities, images_i, rotation_matrices, translations, ctf_params, noise_variance, translation_fn="fft"):
    """Equinox version of backproject_one_image (12 → 8 params)."""
    images = sum_up_translations(images_i, probabilities, translations, config.image_shape, translation_fn)
    CTF = config.compute_ctf(ctf_params)
    images *= (CTF[:,None,None] / noise_variance)
    images_half = fourier_transform_utils.full_image_to_half_image(images, config.image_shape)
    Ft_y = batch_vol_adjoint_slice_volume_half(images_half, rotation_matrices, config.image_shape, config.volume_shape, None)
    probabilites_summed_over_translations = jnp.sum(probabilities, axis=-1)[...,None]
    CTF_probs = (CTF**2 / noise_variance)[:,None,None] * probabilites_summed_over_translations
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, config.image_shape)
    Ft_ctf = batch_vol_adjoint_slice_volume_half(CTF_probs_half, rotation_matrices, config.image_shape, config.volume_shape, None)
    return Ft_y, Ft_ctf


@eqx.filter_jit
def sum_up_images_fixed_rots_eqx(config: ForwardModelConfig, batch, probabilities, translations, rotations, ctf_params, noise_variance, Ft_y=0, Ft_ctf=0):
    """Equinox version of sum_up_images_fixed_rots (13 → 9 params)."""
    assert(probabilities.shape[0] == batch.shape[0])
    assert(probabilities.shape[1] == rotations.shape[0])
    assert(probabilities.shape[2] == translations.shape[0])
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = batch.shape[0]
    n_shifted_images = n_images * n_translations

    CTF = config.compute_ctf(ctf_params)
    batch = config.process_fn(batch, apply_image_mask=False) * CTF / noise_variance
    shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), config.image_shape)
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])

    P = probabilities.swapaxes(0,1).reshape(n_rotations, n_shifted_images)
    summed_images = P @ shifted_images
    summed_half = fourier_transform_utils.full_image_to_half_image(summed_images, config.image_shape)
    Ft_y = core.adjoint_slice_volume_by_trilinear_from_half_images(summed_half, rotations, config.image_shape, config.volume_shape, Ft_y)

    probabilites_summed_over_translations = jnp.sum(probabilities, axis=-1)
    CTF_probs = probabilites_summed_over_translations.T @ (CTF**2 / noise_variance)
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, config.image_shape)
    Ft_ctf = core.adjoint_slice_volume_by_trilinear_from_half_images(CTF_probs_half, rotations, config.image_shape, config.volume_shape, Ft_ctf)

    return Ft_y, Ft_ctf


# ============================================================================
# Legacy M-step API
# ============================================================================

@functools.partial(jax.jit, static_argnums=[5,8,9,10])
def sum_up_images_fixed_rots(batch, probabilities, translations, rotations, CTF_params, CTF_fun, noise_variance, voxel_size, image_shape, volume_shape, process_images, Ft_y = 0, Ft_ctf = 0):

    assert(probabilities.shape[0] == batch.shape[0])
    assert(probabilities.shape[1] == rotations.shape[0])
    assert(probabilities.shape[2] == translations.shape[0])
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = batch.shape[0]
    n_shifted_images = n_images * n_translations

    CTF = CTF_fun(CTF_params, image_shape, voxel_size)
    batch = process_images(batch, apply_image_mask = False) * CTF / noise_variance
    shifted_images = core.batch_trans_translate_images(batch, jnp.repeat(translations[None], batch.shape[0], axis=0), image_shape)
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])

    P = probabilities.swapaxes(0,1).reshape(n_rotations, n_shifted_images )
    summed_images = P @ shifted_images

    summed_half = fourier_transform_utils.full_image_to_half_image(summed_images, image_shape)
    Ft_y = core.adjoint_slice_volume_by_trilinear_from_half_images(summed_half, rotations, image_shape, volume_shape, Ft_y)

    probabilites_summed_over_translations = jnp.sum(probabilities, axis = -1)

    CTF_probs =  probabilites_summed_over_translations.T @ (CTF**2 / noise_variance)
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, image_shape)
    Ft_ctf = core.adjoint_slice_volume_by_trilinear_from_half_images(CTF_probs_half, rotations, image_shape, volume_shape, Ft_ctf)

    return Ft_y, Ft_ctf

def M_with_precompute(experiment_dataset, probabilities, rotations, translations, noise_variance, disc_type, image_indices = None):

    logger.info("starting precomp proj. Num rotations %s, num translations %s. Total = %s", rotations.shape[0], translations.shape[0], rotations.shape[0] * translations.shape[0])
    projections = np.zeros((rotations.shape[0], experiment_dataset.image_size), dtype = np.complex64)

    image_shape = experiment_dataset.image_shape
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    if n_rotations <= 0:
        raise ValueError("M_with_precompute requires at least one rotation")
    if n_translations <= 0:
        raise ValueError("M_with_precompute requires at least one translation")
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)
    from recovar import utils

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.image_stack.process_images,
    )

    gpu_memory = utils.get_gpu_memory_total()
    # *20: backprojection accumulates into a single volume, so per-image memory is low
    # Divide by translations for per-translation inner loop memory
    batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) // translations.shape[0] * 20)

    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = image_indices)

    Ft_y, Ft_ctf = jnp.zeros((experiment_dataset.volume_size), dtype = experiment_dataset.dtype), jnp.zeros((experiment_dataset.volume_size), experiment_dataset.dtype)

    mult = 5
    rotation_batch = max(1, rotations.shape[0] // mult)
    logger.info("Starting sum up images. Batch size %s, rotation batch %s", batch_size, rotation_batch)
    start_idx = 0
    for batch, _, indices in data_generator:
        batch = jnp.asarray(batch)
        end_idx = start_idx + len(indices)

        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):
            Ft_y, Ft_ctf = sum_up_images_fixed_rots_eqx(
                config, batch,
                probabilities[start_idx:end_idx, rot_indices[0]:rot_indices[-1]+1],
                translations, rotations[rot_indices],
                experiment_dataset.CTF_params[indices], noise_variance,
                Ft_y=Ft_y, Ft_ctf=Ft_ctf,
            )

        start_idx = end_idx

    return Ft_y, Ft_ctf
