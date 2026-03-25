"""High-level EM loop orchestration and convergence tracking."""

from recovar import utils
from recovar.em.core import hard_assignment_idx_to_pose
import numpy as np
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)


def E_M_batches_2(
    experiment_dataset, state_obj, rotations, translations, disc_type, memory_to_use=128, volume_mask=None
):

    if rotations.shape[0] <= 0:
        raise ValueError("E_M_batches_2 requires at least one rotation")
    if translations.shape[0] <= 0:
        raise ValueError("E_M_batches_2 requires at least one translation")
    total_hidden = rotations.shape[0] * translations.shape[0]
    logger.info(
        "starting precomp proj. Num rotations %s, num translations %s. Total = %s",
        rotations.shape[0],
        translations.shape[0],
        total_hidden,
    )
    n_images_batch = int(memory_to_use * 1e9 / (total_hidden * 8))
    if n_images_batch < 1:
        n_images_batch = 1
        logger.warning(
            "Memory to use is too small. Setting n_images_batch to %s. May run out of memory", n_images_batch
        )
    logger.info(
        "n_images_batch %s. Number of batches %s",
        n_images_batch,
        int(np.ceil(experiment_dataset.n_units / n_images_batch)),
    )

    if state_obj.name == "SGD":
        sgd_batch = int(state_obj.sgd_batchsize)
        if sgd_batch < 1:
            raise ValueError("SGD batch size must be >= 1")
        n_images_batch = sgd_batch

    hard_assignment = np.empty(experiment_dataset.n_units, dtype=int)

    for big_image_batch in utils.index_batch_iter(experiment_dataset.n_units, n_images_batch):
        probabilities = state_obj.E_step(experiment_dataset, rotations, translations, disc_type, big_image_batch)
        hard_assignment[big_image_batch] = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)
        if np.isnan(probabilities).any():
            logger.warning("NaNs detected in probabilities; mean norm=%s", np.linalg.norm(state_obj.mean))
        state_obj.M_step(experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch)

    return state_obj, hard_assignment


def split_E_M_v2(
    experiment_datasets,
    state_objs,
    rotations,
    translations,
    disc_type,
    average_up_to_angstrom=None,
):

    hard_assignments = 2 * [None]
    for i, experiment_dataset in enumerate(experiment_datasets):
        state_objs[i], hard_assignments[i] = E_M_batches_2(
            experiment_dataset, state_objs[i], rotations, translations, disc_type
        )
        state_objs[i].finish_up_M_step(experiment_dataset, disc_type)

    ## Update prior and estimate resolution
    from recovar.reconstruction import regularization
    from recovar.heterogeneity import locres

    cryo = experiment_datasets[0]

    use_fsc_prior = state_objs[0].name != "SGD"
    means = [state_obj.mean for state_obj in state_objs]
    if use_fsc_prior:
        from recovar.reconstruction import relion_functions

        unreg_means = [
            relion_functions.post_process_from_filter(
                experiment_dataset, state_obj.Ft_CTF, state_obj.Ft_y, tau=None, disc_type=disc_type
            )
            for state_obj in state_objs
        ]
        mean_signal_variance, fsc, _ = regularization.compute_relion_prior(
            experiment_datasets, state_objs[0].noise_variance, unreg_means[0], unreg_means[1], 100
        )

    else:
        fsc = regularization.get_fsc_gpu(
            means[0], means[1], cryo.volume_shape, substract_shell_mean=False, frequency_shift=0
        )
        mean_avg = (means[0] + means[1]) / 2
        PS = regularization.average_over_shells(jnp.abs(mean_avg) ** 2, cryo.volume_shape)

        T = 4
        mean_signal_variance = T * 1 / 2 * utils.make_radial_image(PS, cryo.volume_shape, extend_last_frequency=True)

        mean_signal_variance += np.max(mean_signal_variance) * 1e-6

    from recovar.reconstruction import noise

    for k in range(2):
        best_rotations, best_translations = hard_assignment_idx_to_pose(hard_assignments[k], rotations, translations)
        experiment_datasets[k].update_poses(best_rotations, best_translations)

    noise_from_res = noise.estimate_noise_level_no_masks(
        experiment_datasets[0], np.arange(np.min([1000, cryo.n_units])), means[0], 100, disc_type="linear_interp"
    )
    noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)
    current_pixel_res = locres.find_fsc_resol(fsc, threshold=1 / 7)
    current_res = current_pixel_res / cryo.voxel_size
    logger.info("Current resolution is %s, pixel resolution: %s", current_res, current_pixel_res)

    if state_objs[0].name == "HeterogeneousEM":
        # Downsample to mean resolution
        valid_freqs = np.array(cryo.get_valid_frequency_indices(current_pixel_res))

        if state_objs[0].u is not None:
            for state_obj in state_objs:
                state_obj.u = state_obj.u * valid_freqs[None]
                state_obj.subspace = state_obj.subspace * valid_freqs[..., None]

        covariance_options = state_objs[0].covariance_options
        _, covariance_prior, _ = regularization.prior_iteration_relion_style_batch(
            state_objs[0].H,
            state_objs[1].H,
            state_objs[0].B,
            state_objs[1].B,
            np.zeros(state_objs[0].H.shape[0]),
            state_objs[0].covariance_prior,
            covariance_options["substract_shell_mean"],
            cryo.volume_shape,
            covariance_options["left_kernel"],
            covariance_options["use_spherical_mask"],
            covariance_options["grid_correct"],
            None,
            covariance_options["prior_n_iterations"],
            covariance_options["downsample_from_fsc"],
        )

        for k in range(2):
            state_objs[k].covariance_prior = covariance_prior

    if average_up_to_angstrom is not None:
        low_res_mask = cryo.get_valid_frequency_indices(average_up_to_angstrom)
        logger.info("Averaging halfmaps up to %s pixels", average_up_to_angstrom)
        means = [np.array(mean) for mean in means]
        means[0][low_res_mask] = (means[0][low_res_mask] + means[1][low_res_mask]) / 2
        means[1][low_res_mask] = means[0][low_res_mask]

    # Update objects
    for k in range(2):
        state_objs[k].noise_variance = noise_variance
        state_objs[k].mean_variance = mean_signal_variance
        state_objs[k].mean = means[k]

    return state_objs, current_pixel_res, hard_assignments
