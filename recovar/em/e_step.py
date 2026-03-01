import functools
import logging
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from recovar import core
from recovar.core.configs import ForwardModelConfig
from .core import (
    batch_vol_slice_volume_by_map,
    batch_vol_rot_slice_volume_by_map,
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

def E_with_precompute(experiment_dataset, volume, rotations, translations, noise_variance, disc_type, image_indices = None, u = None, s = None):
    # I am not sure this is a reasonable way to be passing things around.

    logger.info(f"starting precomp proj. Num rotations {rotations.shape[0]}, num translations {translations.shape[0]}. Total = {rotations.shape[0] * translations.shape[0]}")
    # Probably should stop storing rotations as matrices at some point.
    # batch_size = utils.get_num
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
    from recovar import utils

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.image_stack.process_images,
    )

    gpu_memory = utils.get_gpu_memory_total()
    # *5: slicing is cheap per image, use larger batches for projection precomputation
    batch_size = utils.safe_batch_size(utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory) * 5)
    n_batches = utils.get_number_of_index_batch(n_rotations, batch_size)

    projections = np.zeros((rotations.shape[0], image_size), dtype = np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
        projections[rot_indices] = core.slice_volume_by_map(volume, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type)

    logger.info(f"done with precomp proj, batch size {batch_size}")
    projections = jnp.asarray(projections)
    logger.info(f"Allocating proj")

    # Compute \sum_i A_i^T y_i / sigma_i^2
    residuals = np.empty((n_images,  projections.shape[0], n_translations))

    # *10: dot products use less memory than full forward model; divide by translations for inner loop
    dot_product_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections)) / translations.shape[0] * 10)
    logger.info(f"Starting IP. Dot product batch size {dot_product_batch_size}. Remaining memory {gpu_memory - utils.get_size_in_gb(projections)}")
    utils.report_memory_device(logger=logger)
    # dot_product_batch_size = batch_size // translations.shape[0]
    data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=dot_product_batch_size, subset_indices = image_indices)
    image_indices = np.arange(n_images) if image_indices is None else image_indices

    start_idx = 0
    for batch, _, indices in data_generator:
        # running_idx
        # Only place where image mask is used ?
        end_idx = start_idx + len(indices)
        residuals[start_idx:end_idx] = compute_dot_products_eqx(
            config, projections, batch, translations,
            experiment_dataset.CTF_params[indices], noise_variance,
        )
        start_idx  = end_idx

    if use_heterogeneous:
        u_projections = np.empty((rotations.shape[0], n_principal_components, image_size), dtype = np.complex64)
        # Compute all mean and principal component projections
        for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
            u_projections[rot_indices] = batch_vol_slice_volume_by_map(u, rotations[rot_indices], experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type)

        logger.info(f"done with u_proj {batch_size}")
        data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=dot_product_batch_size, subset_indices = image_indices)

        rotation_batch = max(1, rotations.shape[0] // 10)
        start_idx = 0
        for batch, _, indices in data_generator:
            batch = jnp.asarray(batch)
            end_idx = start_idx + len(indices)

            for rot_indices in utils.index_batch_iter(n_rotations, rotation_batch):# k in range(mult):
                # Hmmm this is a bit of a hack. Indexing is not what I wish it was
                rot_indices = np.array(rot_indices)
                residuals[start_idx:end_idx, rot_indices] -= compute_bHb_terms(projections[rot_indices], u_projections[rot_indices], s, batch, translations, experiment_dataset.CTF_params[indices], experiment_dataset.CTF_fun, noise_variance, experiment_dataset.voxel_size, image_shape, experiment_dataset.image_stack.process_images)

            start_idx = end_idx

    projections = (jnp.abs(projections)**2).block_until_ready()

    logger.info(f"done with IP")
    utils.report_memory_device(logger=logger)
    # For the \|C_i Proj_j\|^2 term

    # *3: norm computation is lighter than full forward model
    norm_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(experiment_dataset.grid_size, gpu_memory - utils.get_size_in_gb(projections)) * 3)

    for array_indices, dataset_indices in utils.subset_and_indices_batch_iter(image_indices, norm_batch_size):
        # indices = utils.get_batch_of_indices_arange(n_images, norm_batch_size, k)
        res = compute_CTFed_proj_norms_eqx(
            config, projections,
            experiment_dataset.CTF_params[dataset_indices], noise_variance,
        )
        if array_indices[-1] == n_images - 1:
            res = res.block_until_ready()
        residuals[array_indices] += np.array(res[...,None])

    del projections
    logger.info(f"done with norms. Batch size {norm_batch_size}")

    # //10: probability computation is memory-intensive (softmax over rotations)
    prob_batch_size = utils.safe_batch_size(batch_size // 10)
    n_batches = utils.get_number_of_index_batch(n_images, prob_batch_size)
    for array_indices, _ in utils.subset_and_indices_batch_iter(image_indices, prob_batch_size):
        residuals[array_indices] = compute_probability_from_residual_normal_squared_one_image(residuals[array_indices])

    logger.info(f"done probs. Batch size {batch_size}")

    return residuals



@functools.partial(jax.jit)
def compute_probability_from_residual_normal_squared_one_image(norm_res_squared):
    all_axis_but_first = tuple(range(1, norm_res_squared.ndim))
    norm_res_squared = 0.5 * norm_res_squared

    norm_res_squared -= jnp.min(norm_res_squared, axis= all_axis_but_first, keepdims=True)
    exp_res = jnp.exp(- norm_res_squared)
    summed_exp = jnp.sum(exp_res, axis = all_axis_but_first, keepdims=True)
    return exp_res / summed_exp

compute_probability_from_residual_normal_squared = jax.vmap(compute_probability_from_residual_normal_squared_one_image)




## This is the version of the code to be used when poses are not the same for all images.

# ============================================================================
# Equinox-based E-step API
# ============================================================================

@eqx.filter_jit
def compute_residuals_many_poses_eqx(config: ForwardModelConfig, volumes, images, rotation_matrices, translations, ctf_params, noise_variance, translation_fn="fft"):
    """Equinox version of compute_residuals_many_poses (12 → 8 params)."""
    projected_volumes = batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type)
    projected_volumes = (projected_volumes * config.compute_ctf(ctf_params)[:,None,None,:])

    images /= jnp.sqrt(noise_variance)
    projected_volumes /= jnp.sqrt(noise_variance)

    if translation_fn == "fft":
        proj_volume_norm = jnp.linalg.norm(projected_volumes, axis=(-1), keepdims=True)**2
        projected_volumes = norm_squared_residuals_from_ft(projected_volumes, images, config.image_shape)
        image_size = np.prod(config.image_shape)
        if NORM_FFT != "ortho":
            projected_volumes = projected_volumes * image_size
        translations_indices = translations_to_indices(translations, config.image_shape)
        dots_chosen = batch_take(projected_volumes, translations_indices, axis=-1)
        norm_res_squared = proj_volume_norm - 2 * dots_chosen.real
        norm_res_squared += jnp.linalg.norm(images, axis=(-1), keepdims=True)[:,None,None]**2
    else:
        projected_volumes = projected_volumes[...,None,:]
        translated_images = core.batch_trans_translate_images(images, translations, config.image_shape)[:,None, None]
        norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images), axis=(-1))**2

    return norm_res_squared


# ============================================================================
# Legacy E-step API
# ============================================================================

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
        dots_chosen = batch_take(projected_volumes, translations_indices, axis = -1)


        norm_res_squared = proj_volume_norm - 2 * dots_chosen.real
        # Match the other implementation
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
    return take_vmap(arr, indices, axis)
