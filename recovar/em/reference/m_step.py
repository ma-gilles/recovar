"""M-step: volume update via weighted backprojection."""

import logging

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.sampling import translations_to_indices

logger = logging.getLogger(__name__)


def sum_up_translate_one_image(image, probabilities, translations, image_shape, translation_fn="fft"):
    if translation_fn == "fft":
        image_size = np.prod(image_shape)
        images_probs = jnp.zeros((*probabilities.shape[:-1], image_size), dtype=probabilities.dtype)
        translations_indices = translations_to_indices(translations, image_shape)
        images_probs = images_probs.at[..., translations_indices].add(probabilities)
        images_probs = fourier_transform_utils.get_dft2(images_probs.reshape(*images_probs.shape[:-1], *image_shape))
        summed_up_images = (image) * (images_probs.reshape(*images_probs.shape[:-2], np.prod(image_shape)))
    else:
        translated_images = core.batch_trans_translate_images(image[None], translations[None], image_shape)
        # Sum over the translation axis (axis=-2), not the pixel axis (axis=-1)
        summed_up_images = jnp.sum(translated_images * probabilities[..., None], axis=-2)

    return summed_up_images


@eqx.filter_jit
def accumulate_fixed_rotation_mstep(
    config: ForwardModelConfig,
    batch,
    probabilities,
    translations,
    rotations,
    ctf_params,
    noise_variance,
    Ft_y=0,
    Ft_ctf=0,
):
    """Accumulate fixed-rotation image and CTF terms for the M-step."""
    assert probabilities.shape[0] == batch.shape[0]
    assert probabilities.shape[1] == rotations.shape[0]
    assert probabilities.shape[2] == translations.shape[0]
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = batch.shape[0]
    n_shifted_images = n_images * n_translations

    CTF = config.compute_ctf(ctf_params)
    batch = config.process_fn(batch, apply_image_mask=False) * CTF / noise_variance
    shifted_images = core.batch_trans_translate_images(
        batch, jnp.repeat(translations[None], batch.shape[0], axis=0), config.image_shape
    )
    shifted_images = shifted_images.reshape(n_shifted_images, shifted_images.shape[-1])

    P = probabilities.swapaxes(0, 1).reshape(n_rotations, n_shifted_images)
    summed_images = P @ shifted_images
    summed_half = fourier_transform_utils.full_image_to_half_image(summed_images, config.image_shape)
    Ft_y = core.adjoint_slice_volume(
        summed_half, rotations, config.image_shape, config.volume_shape, "linear_interp", volume=Ft_y, half_image=True
    )

    probabilites_summed_over_translations = jnp.sum(probabilities, axis=-1)
    CTF_probs = probabilites_summed_over_translations.T @ (CTF**2 / noise_variance)
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, config.image_shape)
    Ft_ctf = core.adjoint_slice_volume(
        CTF_probs_half,
        rotations,
        config.image_shape,
        config.volume_shape,
        "linear_interp",
        volume=Ft_ctf,
        half_image=True,
    )

    return Ft_y, Ft_ctf

def M_with_precompute(
    experiment_dataset, probabilities, rotations, translations, noise_variance, disc_type, image_indices=None
):

    logger.info(
        "starting precomp proj. Num rotations %s, num translations %s. Total = %s",
        rotations.shape[0],
        translations.shape[0],
        rotations.shape[0] * translations.shape[0],
    )
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    if n_rotations <= 0:
        raise ValueError("M_with_precompute requires at least one rotation")
    if n_translations <= 0:
        raise ValueError("M_with_precompute requires at least one translation")

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    gpu_memory = utils.get_gpu_memory_total()
    # *20: backprojection accumulates into a single volume, so per-image memory is low
    # Divide by translations for per-translation inner loop memory
    batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) // translations.shape[0] * 20
    )

    Ft_y, Ft_ctf = (
        jnp.zeros((experiment_dataset.volume_size), dtype=experiment_dataset.dtype),
        jnp.zeros((experiment_dataset.volume_size), experiment_dataset.dtype),
    )

    mult = 5
    rotation_batch = max(1, rotations.shape[0] // mult)
    logger.info("Starting sum up images. Batch size %s, rotation batch %s", batch_size, rotation_batch)
    start_idx = 0
    for (
        batch,
        _rotation_matrices,
        _translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        batch_image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        indices=image_indices,
        by_image=False,
    ):
        batch = jnp.asarray(batch)
        end_idx = start_idx + len(batch_image_indices)

        for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):
            Ft_y, Ft_ctf = accumulate_fixed_rotation_mstep(
                config,
                batch,
                probabilities[start_idx:end_idx, rot_indices[0] : rot_indices[-1] + 1],
                translations,
                rotations[rot_indices],
                ctf_params,
                noise_variance,
                Ft_y=Ft_y,
                Ft_ctf=Ft_ctf,
            )

        start_idx = end_idx

    return Ft_y, Ft_ctf
