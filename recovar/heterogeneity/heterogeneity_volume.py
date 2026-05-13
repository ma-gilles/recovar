"""Kernel-regression volume reconstruction from latent embeddings."""

import logging
import os

import jax.numpy as jnp
import jax.scipy
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.heterogeneity.latent_density
import recovar.jax_config
from recovar import utils
from recovar.core import mask
from recovar.data_io import cryoem_dataset, halfsets
from recovar.heterogeneity import deconvolved_kernel_regression, kernel_regression_reconstruction, locres
from recovar.output import output as output_mod
from recovar.output.output_paths import VolumeOutputPaths, resolve_volume_diag_path
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)


def pick_minimum_discretization_size(ndim, log_likelihoods, q=0.5, min_images=50):
    if ndim > 0:
        disc_latent_dist = recovar.heterogeneity.latent_density.get_log_likelihood_threshold(k=ndim, q=0.5)
    else:
        disc_latent_dist = -1
    if log_likelihoods.size < min_images:
        logger.warning("Not enough images for minimum discretization size. Using %s images.", min_images)
        return disc_latent_dist
    value = np.max([np.sort(log_likelihoods)[min_images], disc_latent_dist])  # Bump a lil bit
    return value * (1 + 1e-8)


def pick_heterogeneity_bins2(ndim, log_likelihoods, q=0.5, min_images=50, n_bins=11):
    if log_likelihoods.size == 0:
        logger.warning("Empty log_likelihoods array; returning default bins.")
        disc_latent_dist = recovar.heterogeneity.latent_density.get_log_likelihood_threshold(k=max(ndim, 1), q=q)
        return np.linspace(np.sqrt(disc_latent_dist), np.sqrt(disc_latent_dist * 10), n_bins) ** 2
    disc_latent_dist = pick_minimum_discretization_size(ndim, log_likelihoods, q, min_images)
    max_latent_dist = np.percentile(log_likelihoods, 95)
    return np.linspace(np.sqrt(disc_latent_dist), np.sqrt(max_latent_dist), n_bins) ** 2


def make_volumes_kernel_estimate_from_results(
    latent_point,
    results,
    ndim,
    cryos=None,
    n_bins=11,
    output_folder=None,
    B_factor=0,
    metric_used="locmost_likely",
    n_min_particles=50,
):
    """Reconstruct a volume at an arbitrary latent-space point.

    Convenience wrapper around :func:`make_volumes_kernel_estimate_local`
    that loads the dataset and computes heterogeneity distances from a
    pipeline results dict.

    Args:
        latent_point: Target point in latent space, shape ``(zdim,)``.
        results: Pipeline output dictionary (loaded from pickle).
        ndim: Latent dimensionality to use.
        cryos: Pre-loaded dataset (``CryoEMDataset`` or ``CryoEMDataset``);
            loaded from *results* if ``None``.
        n_bins: Number of heterogeneity bins for kernel regression.
        output_folder: Directory for output MRC files.
        B_factor: B-factor sharpening in Angstroms squared.
        metric_used: Volume quality metric for selection.
        n_min_particles: Minimum particles per bin.
    """
    ds = halfsets.load_halfset_dataset_from_args(results["input_args"], lazy=False) if cryos is None else cryos
    output_folder = results["input_args"].outdir + "/output/" if output_folder is None else output_folder
    logger.info("Dumping to %s", output_folder)
    output_mod.mkdir_safe(output_folder)
    noise_variance = results["cov_noise"]
    latent_points = latent_point[None]

    coords = results["latent_coords"]
    precision = results["latent_precision"]
    log_likelihoods = recovar.heterogeneity.latent_density.compute_latent_quadratic_forms_in_batch(
        latent_points[:, :ndim], coords[ndim], precision[ndim]
    )[..., 0]
    heterogeneity_distances = ds.split_halfset_array(log_likelihoods, per_particle=ds.tilt_series_flag)
    if metric_used == "global":
        pass
    else:
        logger.warning("Choosing threshold only based on number of images")
        vol_paths = VolumeOutputPaths(output_folder, "state", 0)
        make_volumes_kernel_estimate_local(
            heterogeneity_distances,
            ds,
            vol_paths,
            -1,
            n_bins,
            B_factor,
            tau=None,
            n_min_particles=n_min_particles,
            metric_used=metric_used,
        )


def _standard_kernel_candidates(bins, heterogeneity_distances, n_min_particles):
    if isinstance(bins, int):
        logger.warning("Picking bins based on number of particles only. n_min_particles = %s", n_min_particles)
        bins = pick_heterogeneity_bins2(-1, heterogeneity_distances[1], 0.5, n_min_particles, n_bins=bins)

    n_images_per_bin = [
        int(np.sum(heterogeneity_distances[0] < b) + np.sum(heterogeneity_distances[1] < b))
        for b in bins
    ]
    logger.info("bins %s", bins)
    logger.info("Particles per bin: %s", n_images_per_bin)
    return bins, n_images_per_bin


def _deconvolved_kernel_candidates(ndim, latent_differences, latent_precision, lambda_grid):
    if ndim != 1:
        raise NotImplementedError(f"Deconvolved kernel regression only supports zdim=1; got ndim={ndim}")
    if latent_differences is None or latent_precision is None:
        raise ValueError("deconv_latent_differences and deconv_latent_precision are required for deconvolved mode")

    latent_differences = [
        deconvolved_kernel_regression._coerce_1d_latent_differences(x) for x in latent_differences
    ]
    latent_precision = [
        deconvolved_kernel_regression._coerce_1d_latent_precision(x) for x in latent_precision
    ]
    lambda_grid, _, sigma_ref = deconvolved_kernel_regression.deconvolution_bandwidths_1d(
        np.concatenate(latent_precision), lambda_grid=lambda_grid
    )
    all_latent_differences = np.concatenate(latent_differences)
    all_latent_precision = np.concatenate(latent_precision)
    n_images_per_lambda = [
        int(
            np.count_nonzero(
                deconvolved_kernel_regression.deconvolution_weights_1d(
                    all_latent_differences,
                    all_latent_precision,
                    float(lambda_val) * sigma_ref,
                )
            )
        )
        for lambda_val in lambda_grid
    ]
    logger.info("deconvolution lambda grid %s", lambda_grid)
    logger.info("deconvolution sigma_ref %s", sigma_ref)
    logger.info("Nonzero deconvolution weights per lambda: %s", n_images_per_lambda)
    return lambda_grid, n_images_per_lambda, sigma_ref, latent_differences, latent_precision


@nvtx.annotate("make_volumes_kernel_estimate_local", color="yellow")
def make_volumes_kernel_estimate_local(
    heterogeneity_distances,
    dataset,
    vol_paths,
    ndim,
    bins,
    B_factor,
    tau=None,
    n_min_particles=50,
    metric_used="locshellmost_likely",
    upsampling_for_ests=1,
    use_mask_ests=False,
    grid_correct_ests=False,
    locres_sampling=25,
    locres_maskrad=None,
    locres_edgwidth=None,
    kernel_rad=4,
    save_all_estimates=False,
    heterogeneity_kernel="parabola",
    use_fast_rfft=False,
    kernel_regression_mode="standard",
    deconv_latent_differences=None,
    deconv_latent_precision=None,
    deconv_lambda_grid=None,
):
    """Reconstruct volumes along a heterogeneity path using kernel regression.

    For each bin along the heterogeneity axis, selects nearby images
    weighted by a kernel function and performs a local 3-D reconstruction
    with local-resolution filtering.  Results are written as MRC files.

    Args:
        heterogeneity_distances: Per-half-set log-likelihood distances,
            list of two arrays each of shape ``(n_images_half,)``.
        dataset: ``CryoEMDataset`` with ``halfset_indices`` set.
            Halfset datasets are obtained via ``dataset.get_halfset(k)``.
        vol_paths: :class:`VolumeOutputPaths` defining where to write outputs.
        ndim: Latent dimensionality (``-1`` for automatic).
        bins: Number of bins (int) or explicit bin edges (array).
        B_factor: B-factor sharpening in Angstroms squared.
        tau: Regularization parameter (``None`` = auto).
        n_min_particles: Minimum particles per bin.
        metric_used: Volume quality metric for local-resolution selection.
        upsampling_for_ests: Upsampling factor for estimates.
        use_mask_ests: Apply mask to estimates.
        grid_correct_ests: Apply gridding correction.
        locres_sampling: Number of local-resolution shells.
        locres_maskrad: Local-resolution mask radius.
        locres_edgwidth: Local-resolution mask edge width.
        kernel_rad: Radius of the heterogeneity kernel.
        save_all_estimates: Save all intermediate estimates.
        heterogeneity_kernel: Kernel shape (``'parabola'`` or ``'flat'``).
    """
    vol_paths.ensure_dirs()
    ds = dataset
    if kernel_regression_mode not in ("standard", "deconvolved"):
        raise ValueError(f"Unknown kernel_regression_mode={kernel_regression_mode!r}")

    deconv_sigma_ref = None
    if kernel_regression_mode == "standard":
        heterogeneity_bins, n_images_per_bin = _standard_kernel_candidates(
            bins, heterogeneity_distances, n_min_particles
        )
    else:
        (
            heterogeneity_bins,
            n_images_per_bin,
            deconv_sigma_ref,
            deconv_latent_differences,
            deconv_latent_precision,
        ) = _deconvolved_kernel_candidates(
            ndim,
            deconv_latent_differences,
            deconv_latent_precision,
            deconv_lambda_grid,
        )

    if kernel_regression_mode == "standard":

        def _run_kernel_estimator(
            half_ds,
            half_idx,
            candidate_grid,
            *,
            tau_value,
            grid_correct,
            use_spherical_mask,
            upsampling_factor,
            return_lhs_rhs=False,
        ):
            return kernel_regression_reconstruction.estimate_standard_kernel_volumes(
                half_ds,
                heterogeneity_distances[half_idx],
                candidate_grid,
                tau=tau_value,
                grid_correct=grid_correct,
                use_spherical_mask=use_spherical_mask,
                return_lhs_rhs=return_lhs_rhs,
                heterogeneity_kernel=heterogeneity_kernel,
                upsampling_factor=upsampling_factor,
                return_real_space=True,
                use_fast_rfft=use_fast_rfft,
            )

    else:

        def _run_kernel_estimator(
            half_ds,
            half_idx,
            candidate_grid,
            *,
            tau_value,
            grid_correct,
            use_spherical_mask,
            upsampling_factor,
            return_lhs_rhs=False,
        ):
            return deconvolved_kernel_regression.estimate_deconvolved_kernel_volumes(
                half_ds,
                deconv_latent_differences[half_idx],
                deconv_latent_precision[half_idx],
                candidate_grid,
                tau=tau_value,
                grid_correct=grid_correct,
                use_spherical_mask=use_spherical_mask,
                return_lhs_rhs=return_lhs_rhs,
                upsampling_factor=upsampling_factor,
                return_real_space=True,
                use_fast_rfft=use_fast_rfft,
                sigma_ref=deconv_sigma_ref,
            )

    estimates = [None, None]
    lhs, rhs = [None, None], [None, None]
    cross_validation_estimators = [None, None]
    for k in range(2):
        half_ds = ds.get_halfset(k)
        estimates[k] = _run_kernel_estimator(
            half_ds,
            k,
            heterogeneity_bins,
            tau_value=tau,
            grid_correct=grid_correct_ests,
            use_spherical_mask=use_mask_ests,
            upsampling_factor=upsampling_for_ests,
        )
        estimates[k] = estimates[k].reshape(-1, *ds.volume_shape).astype(np.float32)
        logger.info("Computing estimates done")

        cross_validation_estimators[k], lhs[k], rhs[k] = (
            _run_kernel_estimator(
                half_ds,
                k,
                heterogeneity_bins[0:1],
                tau_value=tau,
                grid_correct=False,
                use_spherical_mask=False,
                return_lhs_rhs=True,
                upsampling_factor=1,
            )
        )

        lhs[k] = fourier_transform_utils.half_volume_to_full_volume(
            lhs[k][0],
            ds.volume_shape,
        )
        # Zero out things after Nyquist - these won't be used in CV
        lhs[k] = (lhs[k] * ds.get_valid_frequency_indices()).reshape(ds.volume_shape)
        cross_validation_estimators[k] = cross_validation_estimators[k].reshape(ds.volume_shape).astype(np.float32)

    logger.info("Computing estimates done")

    do_smooth_error = "smooth" in metric_used
    if metric_used == "locmost_likely":
        ml_choice, ml_errors = choice_most_likely(
            estimates[0],
            estimates[1],
            cross_validation_estimators[0],
            cross_validation_estimators[1],
            lhs[0],
            lhs[1],
            ds.voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
        )
    elif "locshellmost_likely" in metric_used:
        ml_choice, ml_errors = choice_most_likely_split(
            estimates[0],
            estimates[1],
            cross_validation_estimators[0],
            cross_validation_estimators[1],
            lhs[0],
            lhs[1],
            ds.voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            smooth_error=do_smooth_error,
        )
    else:
        raise ValueError("Metric used not recognized")

    estimates = [None, None]

    for k in range(2):
        logger.info("Computing estimates start")
        half_ds = ds.get_halfset(k)
        estimates[k] = _run_kernel_estimator(
            half_ds,
            k,
            heterogeneity_bins,
            tau_value=None,
            grid_correct=True,
            use_spherical_mask=True,
            upsampling_factor=2,
        )
        estimates[k] = estimates[k].reshape(-1, *ds.volume_shape).astype(np.float32)

    def _save_outputs(choice):
        """Select best estimates, filter, and write all output files."""

        opt_halfmaps = [None, None]

        for k in range(2):
            if metric_used == "locmost_likely":
                opt_halfmaps[k] = jnp.take_along_axis(estimates[k], choice[None], axis=0)[0]
                _, smoothed_choice = smoothed_best_choice(estimates[0], choice, kernel_rad=kernel_rad)
            elif "locshellmost_likely" in metric_used:
                opt_halfmaps[k] = locres.recombine_estimates(
                    estimates[k],
                    choice,
                    ds.voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=locres_maskrad,
                    locres_edgwidth=locres_edgwidth,
                )

        best_filtered, best_filtered_res, best_auc, fscs, _ = locres.local_resolution(
            opt_halfmaps[0],
            opt_halfmaps[1],
            B_factor,
            ds.voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=None,
            locres_edgwidth=None,
            locres_minres=50,
            use_filter=True,
            fsc_threshold=1 / 7,
            use_v2=False,
        )

        # Primary outputs
        recovar.utils.write_mrc(vol_paths.filtered, best_filtered, voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.half1_unfil, opt_halfmaps[0], voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.half2_unfil, opt_halfmaps[1], voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.unfil, (opt_halfmaps[0] + opt_halfmaps[1]) / 2, voxel_size=ds.voxel_size)

        # Diagnostics
        recovar.utils.write_mrc(vol_paths.locres, best_filtered_res, voxel_size=ds.voxel_size)

        volume_sampling = locres.make_sampling_volume(ds.grid_size, locres_sampling, ds.voxel_size, locres_maskrad)
        recovar.utils.write_mrc(vol_paths.sampling, volume_sampling, voxel_size=ds.voxel_size)

        if save_all_estimates:
            if metric_used == "locmost_likely":
                # Take best smoothed then filter
                debug_halfmaps = [None, None]
                for k in range(2):
                    debug_halfmaps[k], _ = smoothed_best_choice(estimates[k], choice, kernel_rad=kernel_rad)

                bf_smooth, bfr_smooth, _, _, _ = locres.local_resolution(
                    debug_halfmaps[0],
                    debug_halfmaps[1],
                    B_factor,
                    ds.voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=None,
                    locres_edgwidth=None,
                    locres_minres=50,
                    use_filter=True,
                    fsc_threshold=1 / 7,
                    use_v2=False,
                )

                recovar.utils.write_mrc(vol_paths.filtered_smooth, bf_smooth, voxel_size=ds.voxel_size)
                recovar.utils.write_mrc(vol_paths.locres_smooth, bfr_smooth, voxel_size=ds.voxel_size)

            # Filter then take best
            loc_filtered_estimates = np.zeros_like(estimates[0])
            for i in range(estimates[0].shape[0]):
                loc_filtered_estimates[i], _, _, _, _ = locres.local_resolution(
                    estimates[0][i],
                    estimates[1][i],
                    B_factor,
                    ds.voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=None,
                    locres_edgwidth=None,
                    locres_minres=50,
                    use_filter=True,
                    fsc_threshold=1 / 7,
                    use_v2=True,
                )

            if metric_used == "locmost_likely":
                opt_filtered_before = jnp.take_along_axis(loc_filtered_estimates, choice[None], axis=0)[0]
            elif "locshellmost_likely" in metric_used:
                opt_filtered_before = locres.recombine_estimates(
                    loc_filtered_estimates,
                    choice,
                    ds.voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=locres_maskrad,
                    locres_edgwidth=locres_edgwidth,
                )

            recovar.utils.write_mrc(vol_paths.filtered_before, opt_filtered_before, voxel_size=ds.voxel_size)

            if metric_used == "locmost_likely":
                opt_filtered_before, smoothed_choice = smoothed_best_choice(
                    loc_filtered_estimates, choice, kernel_rad=kernel_rad
                )
                recovar.utils.write_mrc(vol_paths.filtered_before_smooth, opt_filtered_before, voxel_size=ds.voxel_size)

            est_filt_dir = vol_paths.estimates_dir(1, filtered=True)
            os.makedirs(est_filt_dir, exist_ok=True)
            output_mod.save_volumes(
                loc_filtered_estimates,
                os.path.join(est_filt_dir, ""),
                ds.volume_shape,
                voxel_size=ds.voxel_size,
                from_ft=False,
            )

        if "locshellmost_likely" in metric_used:
            recovar.utils.pickle_dump({"split_choice": ml_choice, "ml_errors": ml_errors}, vol_paths.split_choice)
        else:
            recovar.utils.write_mrc(vol_paths.choice, choice, voxel_size=ds.voxel_size)
            recovar.utils.write_mrc(vol_paths.choice_smooth, smoothed_choice, voxel_size=ds.voxel_size)

        output_dict = {
            "kernel_regression_mode": kernel_regression_mode,
            "heterogeneity_bins": heterogeneity_bins,
            "n_images_per_bin": n_images_per_bin,
            "fscs": fscs,
            "locres_sampling": locres_sampling,
            "locres_maskrad": locres_maskrad,
            "voxel_size": ds.voxel_size,
            "ml_choice": ml_choice,
            "ml_errors": ml_errors,
        }
        if kernel_regression_mode == "deconvolved":
            output_dict["lambda_grid"] = heterogeneity_bins
            output_dict["sigma_ref"] = deconv_sigma_ref

        recovar.utils.pickle_dump(output_dict, vol_paths.params)

    distances_reordered = cryoem_dataset.reorder_to_original_indexing(
        heterogeneity_distances, ds, use_tilt_indices=ds.tilt_series_flag
    )
    np.savetxt(vol_paths.heterogeneity_distances, distances_reordered)
    _save_outputs(ml_choice)

    if save_all_estimates:
        for half_idx in (0, 1):
            est_dir = vol_paths.estimates_dir(half_idx + 1)
            os.makedirs(est_dir, exist_ok=True)
            output_mod.save_volumes(
                estimates[half_idx], os.path.join(est_dir, ""), ds.volume_shape, voxel_size=ds.voxel_size, from_ft=False
            )

        recovar.utils.write_mrc(vol_paths.cv_half1_unfil, cross_validation_estimators[0], voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.cv_noise_half1, lhs[0], voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.cv_noise_half2, lhs[1], voxel_size=ds.voxel_size)
        recovar.utils.write_mrc(vol_paths.cv_half2_unfil, cross_validation_estimators[1], voxel_size=ds.voxel_size)

    return


def choice_most_likely(
    estimates0,
    estimates1,
    target0,
    target1,
    noise_variances_target0,
    noise_variances_target1,
    voxel_size,
    locres_sampling,
    locres_maskrad,
    locres_edgwidth,
):

    n_estimators = estimates0.shape[0]
    errors = np.zeros_like(estimates0)
    use_v2 = True
    for k in range(n_estimators):
        errors[k] = locres.expensive_local_error_with_cov(
            target0,
            estimates1[k],
            voxel_size,
            noise_variances_target0.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
        )
        errors[k] += locres.expensive_local_error_with_cov(
            estimates0[k],
            target1,
            voxel_size,
            noise_variances_target1.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
        )

    choice = np.argmin(errors, axis=0)
    return choice, errors


def smooth_shell_error(shell_error, voxel_size, subarray_size, sum_up_up_to_res=50, smooth_mean_filter=3):
    kernel = jnp.ones(smooth_mean_filter, dtype=jnp.float32)
    vmapped_convolve = jax.vmap(jax.scipy.signal.convolve, in_axes=(0, None, None))
    shell_choice_new = vmapped_convolve(shell_error, kernel, "same")

    # For very low frequencies, just sum up
    full_grids = fourier_transform_utils.get_1d_frequency_grid(subarray_size, voxel_size, scaled=True)

    # Exclude last shell (Nyquist) to match smoothing conventions
    grids = full_grids[-shell_error.shape[-1] - 1 : -1]
    low_res_indices = grids <= 1 / sum_up_up_to_res
    logger.info(
        "Averaging first %s shells out of %s until resolution %s. Smoothing shells with kernel size %s",
        jnp.sum(low_res_indices),
        shell_error.shape[-1],
        sum_up_up_to_res,
        smooth_mean_filter,
    )
    shell_choice_new = jnp.where(
        grids <= 1 / sum_up_up_to_res, jnp.sum(shell_error * low_res_indices), shell_choice_new
    )
    return shell_choice_new


batch_smooth_shell_error = jax.vmap(smooth_shell_error, in_axes=(0, None, None, None, None))


def choice_most_likely_split(
    estimates0,
    estimates1,
    target0,
    target1,
    noise_variances_target0,
    noise_variances_target1,
    voxel_size,
    locres_sampling,
    locres_maskrad,
    locres_edgwidth,
    smooth_error=False,
):

    n_estimators = estimates0.shape[0]
    errors = n_estimators * [None]
    use_v2 = True
    for k in range(n_estimators):
        errors[k] = locres.expensive_local_error_with_cov(
            target0,
            estimates1[k],
            voxel_size,
            noise_variances_target0.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
            split_shell=True,
        )
        errors[k] += locres.expensive_local_error_with_cov(
            estimates0[k],
            target1,
            voxel_size,
            noise_variances_target1.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
            split_shell=True,
        )

    errors = np.asarray(errors)
    if smooth_error:
        subarray_size = int((errors.shape[-1] + 1) * 2)
        logger.info("Smoothing shell error with subarray size %s", subarray_size)
        sum_up_up_to_res = 40
        smooth_mean_filter = 3
        logger.info(
            "Grouping first %s shells together, and smoothing with kernel size %s", sum_up_up_to_res, smooth_mean_filter
        )
        errors = batch_smooth_shell_error(errors, voxel_size, subarray_size, sum_up_up_to_res, smooth_mean_filter)

    choice = np.argmin(errors, axis=0)
    return choice, errors


def smoothed_best_choice(estimates, choice, kernel_rad=4):
    smoothed_choice = mask.soften_volume_mask(choice, kernel_rad)
    bot_boundary = jnp.floor(smoothed_choice).astype(int)

    max_choice = np.max(choice)
    min_choice = np.min(choice)
    bot_boundary = np.where(bot_boundary < min_choice, min_choice, bot_boundary)
    bot_boundary = np.where(bot_boundary > max_choice, max_choice, bot_boundary)

    weight = smoothed_choice - bot_boundary
    bot_estimate = jnp.take_along_axis(estimates, bot_boundary[None], axis=0)[0]

    top_boundary = bot_boundary + 1
    top_boundary = np.where(top_boundary < min_choice, min_choice, top_boundary)
    top_boundary = np.where(top_boundary > max_choice, max_choice, top_boundary)

    top_estimate = jnp.take_along_axis(estimates, top_boundary[None], axis=0)[0]

    smoothed_estimate = (1 - weight) * bot_estimate + (weight) * top_estimate
    return smoothed_estimate, smoothed_choice


def get_inds_for_subvolume(path_to_vol_folder, subvolume_idx, prefix=None, index=None):
    """Get image indices contributing to a local subvolume.

    Supports both the new ``diagnostics/{stem}/`` layout and the old
    flat layout where all files live directly in *path_to_vol_folder*.

    Parameters
    ----------
    path_to_vol_folder : str
        Path to the volume output directory.
    subvolume_idx : int
        Index of the local subvolume to query.
    prefix : str, optional
        Volume name prefix for new layout resolution.
    index : int, optional
        Volume index for new layout resolution.
    """
    params_path = resolve_volume_diag_path(path_to_vol_folder, "params.pkl", prefix, index)
    locres_path = resolve_volume_diag_path(path_to_vol_folder, "local_resolution.mrc", prefix, index)
    het_dist_path = resolve_volume_diag_path(path_to_vol_folder, "heterogeneity_distances.txt", prefix, index)

    params = recovar.utils.pickle_load(params_path)
    locres_ar = recovar.utils.load_mrc(locres_path)

    grid_size = locres_ar.shape[0]
    sampling_points = locres.get_sampling_points(
        grid_size, params["locres_sampling"], params["locres_maskrad"], params["voxel_size"]
    )

    point = sampling_points[subvolume_idx].astype(int) + grid_size // 2
    locres_at_point = locres_ar[point[0], point[1], point[2]]
    logger.info("Local resolution at point is %f \\AA", locres_at_point)

    locres_maskrad = 0.5 * params["locres_sampling"] if params["locres_maskrad"] is None else params["locres_maskrad"]
    subvolume_size = locres.get_local_error_subvolume_size(locres_maskrad, params["voxel_size"])
    frequency_shells = fourier_transform_utils.get_1d_frequency_grid(subvolume_size, params["voxel_size"], scaled=True)
    # Exclude last shell (Nyquist) for consistency with smoothing
    frequency_shells = frequency_shells[frequency_shells >= 0][:-1]

    shell_idx = np.argmin(np.abs(frequency_shells - 1 / locres_at_point)) - 1

    if shell_idx < 0:
        shell_idx = 0
        logger.warning(
            "Local resolution is at selected point is very bad, so using the first shell. Probably meaningless results"
        )

    logger.info(
        "This correspond to the %d frequency shell out of %d of the subvolume", shell_idx, len(frequency_shells)
    )

    ml_choice_idx_shell = params["ml_choice"][subvolume_idx][shell_idx]
    upper_bound = params["heterogeneity_bins"][ml_choice_idx_shell]
    logger.info(
        "This was estimated using the %d bin out of %d in the kernel regression",
        ml_choice_idx_shell,
        len(params["heterogeneity_bins"]),
    )
    logger.info("Which contains %d images", params["n_images_per_bin"][ml_choice_idx_shell])

    heterogeneity_distances = np.loadtxt(het_dist_path)
    good_indices = np.where(heterogeneity_distances < upper_bound)[0]
    return good_indices
