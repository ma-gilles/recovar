"""E-step: posterior probability computation over poses and translations."""

import functools
import logging
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
from .core import (
    batch_vol_slice_volume,
    batch_vol_rot_slice_volume,
    compute_dot_products,
    compute_dot_products_eqx,
    compute_CTFed_proj_norms,
    compute_CTFed_proj_norms_eqx,
    norm_squared_residuals_from_ft,
    NORM_FFT,
)
from .sampling import translations_to_indices
from .heterogeneity import compute_bHb_terms

logger = logging.getLogger(__name__)


def E_with_precompute(
    experiment_dataset, volume, rotations, translations, noise_variance, disc_type, image_indices=None, u=None, s=None
):
    logger.info(
        "starting precomp proj. Num rotations %s, num translations %s. Total = %s",
        rotations.shape[0],
        translations.shape[0],
        rotations.shape[0] * translations.shape[0],
    )
    image_shape = experiment_dataset.image_shape
    image_size = experiment_dataset.image_size
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    if n_rotations <= 0:
        raise ValueError("E_with_precompute requires at least one rotation")
    if n_translations <= 0:
        raise ValueError("E_with_precompute requires at least one translation")
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)
    use_heterogeneous = u is not None
    n_principal_components = u.shape[0] if use_heterogeneous else 0
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    gpu_memory = utils.get_gpu_memory_total()
    # *5: slicing is cheap per image, use larger batches for projection precomputation
    batch_size = utils.safe_batch_size(utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) * 5)

    projections = np.zeros((rotations.shape[0], image_size), dtype=np.complex64)
    volume_obj = core.to_cubic(volume, experiment_dataset.volume_shape) if disc_type == "cubic" else core.Volume(volume, disc_type=disc_type)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
        projections[rot_indices] = core.slice_volume(
            volume_obj,
            rotations[rot_indices],
            experiment_dataset.image_shape,
            experiment_dataset.volume_shape,
        )

    logger.info("done with precomp proj, batch size %s", batch_size)
    projections = jnp.asarray(projections)
    logger.info("Allocating proj")

    # Compute \sum_i A_i^T y_i / sigma_i^2
    residuals = np.empty((n_images, projections.shape[0], n_translations))

    # *10: dot products use less memory than full forward model; divide by translations for inner loop
    dot_product_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections))
        / translations.shape[0]
        * 10
    )
    logger.info(
        "Starting IP. Dot product batch size %s. Remaining memory %s",
        dot_product_batch_size,
        gpu_memory - utils.get_size_in_gb(projections),
    )
    utils.report_memory_device(logger=logger)
    image_indices = np.arange(n_images) if image_indices is None else image_indices

    start_idx = 0
    for (
        batch,
        _rotation_matrices,
        _translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        indices,
    ) in experiment_dataset.iter_batches(
        dot_product_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        end_idx = start_idx + len(indices)
        residuals[start_idx:end_idx] = compute_dot_products_eqx(
            config,
            projections,
            batch,
            translations,
            ctf_params,
            noise_variance,
        )
        start_idx = end_idx

    if use_heterogeneous:
        u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype=np.complex64)
        # Compute all mean and principal component projections
        for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
            u_projections[rot_indices] = batch_vol_slice_volume(
                u, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type
            )

        logger.info("done with u_proj %s", batch_size)

        rotation_batch = max(1, rotations.shape[0] // 10)
        start_idx = 0
        for (
            batch,
            _rotation_matrices,
            _translations,
            ctf_params,
            _noise_variance,
            _particle_indices,
            indices,
        ) in experiment_dataset.iter_batches(
            dot_product_batch_size,
            indices=image_indices,
            by_image=False,
        ):
            batch = jnp.asarray(batch)
            end_idx = start_idx + len(indices)

            for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):
                rot_indices = np.array(rot_indices)
                residuals[start_idx:end_idx, rot_indices] -= compute_bHb_terms(
                    projections[rot_indices],
                    u_projections[rot_indices],
                    s,
                    batch,
                    translations,
                    ctf_params,
                    experiment_dataset.ctf_evaluator,
                    noise_variance,
                    experiment_dataset.voxel_size,
                    image_shape,
                    experiment_dataset.process_images,
                )

            start_idx = end_idx

    projections = (jnp.abs(projections) ** 2).block_until_ready()

    logger.info("done with IP")
    utils.report_memory_device(logger=logger)
    # For the \|C_i Proj_j\|^2 term

    # *3: norm computation is lighter than full forward model
    norm_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections)) * 3
    )

    for array_indices, dataset_indices in utils.subset_and_indices_batch_iter(image_indices, norm_batch_size):
        res = compute_CTFed_proj_norms_eqx(
            config,
            projections,
            experiment_dataset.CTF_params[dataset_indices],
            noise_variance,
        )
        if array_indices[-1] == n_images - 1:
            res = res.block_until_ready()
        residuals[array_indices] += np.array(res[..., None])

    del projections
    logger.info("done with norms. Batch size %s", norm_batch_size)

    # //10: probability computation is memory-intensive (softmax over rotations)
    prob_batch_size = utils.safe_batch_size(batch_size // 10)
    for array_indices, _ in utils.subset_and_indices_batch_iter(image_indices, prob_batch_size):
        residuals[array_indices] = compute_probability_from_residual_normal_squared_one_image(residuals[array_indices])

    logger.info("done probs. Batch size %s", batch_size)

    return residuals


@functools.partial(jax.jit)
def compute_probability_from_residual_normal_squared_one_image(norm_res_squared):
    all_axis_but_first = tuple(range(1, norm_res_squared.ndim))
    norm_res_squared = 0.5 * norm_res_squared

    norm_res_squared -= jnp.min(norm_res_squared, axis=all_axis_but_first, keepdims=True)
    exp_res = jnp.exp(-norm_res_squared)
    summed_exp = jnp.sum(exp_res, axis=all_axis_but_first, keepdims=True)
    return exp_res / summed_exp


compute_probability_from_residual_normal_squared = jax.vmap(compute_probability_from_residual_normal_squared_one_image)


## This is the version of the code to be used when poses are not the same for all images.

# ============================================================================
# Equinox-based E-step API
# ============================================================================


@eqx.filter_jit
def compute_residuals_many_poses_eqx(
    config: ForwardModelConfig,
    volumes,
    images,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    translation_fn="fft",
):
    """Equinox version of compute_residuals_many_poses (12 → 8 params)."""
    projected_volumes = batch_vol_rot_slice_volume(
        volumes, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
    )
    projected_volumes = projected_volumes * config.compute_ctf(ctf_params)[:, None, None, :]

    images /= jnp.sqrt(noise_variance)
    projected_volumes /= jnp.sqrt(noise_variance)

    if translation_fn == "fft":
        proj_volume_norm = jnp.linalg.norm(projected_volumes, axis=(-1), keepdims=True) ** 2
        projected_volumes = norm_squared_residuals_from_ft(projected_volumes, images, config.image_shape)
        image_size = np.prod(config.image_shape)
        if NORM_FFT != "ortho":
            projected_volumes = projected_volumes * image_size
        translations_indices = translations_to_indices(translations, config.image_shape)
        dots_chosen = batch_take(projected_volumes, translations_indices, axis=-1)
        norm_res_squared = proj_volume_norm - 2 * dots_chosen.real
        norm_res_squared += jnp.linalg.norm(images, axis=(-1), keepdims=True)[:, None, None] ** 2
    else:
        projected_volumes = projected_volumes[..., None, :]
        translated_images = core.batch_trans_translate_images(images, translations, config.image_shape)[:, None, None]
        norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images), axis=(-1)) ** 2

    return norm_res_squared


# ============================================================================
# Legacy E-step API
# ============================================================================


@functools.partial(jax.jit, static_argnums=[6, 7, 8, 9, 10, 11])
def compute_residuals_many_poses(
    volumes,
    images,
    rotation_matrices,
    translations,
    CTF_params,
    noise_variance,
    voxel_size,
    volume_shape,
    image_shape,
    disc_type,
    ctf,
    translation_fn="fft",
):
    # n_vols x rotations x image_size
    projected_volumes = batch_vol_rot_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, disc_type)

    # Broadcast CTF in volumes x rotations
    projected_volumes = projected_volumes * ctf(CTF_params, image_shape, voxel_size)[:, None, None, :]

    # Broadcast over volumes x rotations
    images /= jnp.sqrt(noise_variance)
    projected_volumes /= jnp.sqrt(noise_variance)

    if translation_fn == "fft":
        proj_volume_norm = jnp.linalg.norm(projected_volumes, axis=(-1), keepdims=True) ** 2

        # Memory-saving variant:
        projected_volumes = norm_squared_residuals_from_ft(projected_volumes, images, image_shape)
        image_size = np.prod(image_shape)

        if NORM_FFT != "ortho":
            projected_volumes = projected_volumes * image_size
        translations_indices = translations_to_indices(translations, image_shape)
        dots_chosen = batch_take(projected_volumes, translations_indices, axis=-1)

        norm_res_squared = proj_volume_norm - 2 * dots_chosen.real
        # Match the other implementation
        norm_res_squared += jnp.linalg.norm(images, axis=(-1), keepdims=True)[:, None, None] ** 2
    else:
        # add axis for translations
        projected_volumes = projected_volumes[..., None, :]
        translated_images = core.batch_trans_translate_images(images, translations, image_shape)[:, None, None]
        norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images), axis=(-1)) ** 2

    # Output is image_batch x vol_batch  x rots_batch x trans_batch
    return norm_res_squared


take_vmap = jax.vmap(lambda x, y, axis: jnp.take(x, y, axis), in_axes=(0, 0, None))


def batch_take(arr, indices, axis):
    return take_vmap(arr, indices, axis)
