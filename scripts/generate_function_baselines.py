#!/usr/bin/env python
"""Generate per-function baseline intermediates and scores from ~/recovar.

This script runs each pipeline function from ~/recovar (published recovar at
ma-gilles/recovar.git) step-by-step on the shared PDB test dataset, saving
intermediate results and per-function metrics.

Usage (from conda env with recovar installed):
    python scripts/generate_function_baselines.py \
        --dataset-dir /path/to/test_dataset \
        --output-dir /path/to/function_baselines

The script must be run with ~/recovar on PYTHONPATH (conda env 'recovar').
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Path to test_dataset")
    parser.add_argument("--output-dir", required=True, help="Where to save intermediates")
    parser.add_argument("--batch-size", type=int, default=512, help="GPU batch size")
    parser.add_argument("--gpu-memory", type=float, default=40.0, help="GPU memory in GB")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    gpu_memory = args.gpu_memory

    os.makedirs(output_dir, exist_ok=True)
    intermediates_dir = os.path.join(output_dir, "intermediates")
    os.makedirs(intermediates_dir, exist_ok=True)

    # Verify recovar is importable
    import recovar

    recovar_path = os.path.abspath(recovar.__file__)
    logger.info("Using recovar at: %s", recovar_path)

    # Import ~/recovar modules
    from recovar import dataset, noise, mask, utils
    from recovar import homogeneous, covariance_estimation, principal_components, embedding
    from recovar import regularization
    from recovar.simulator import load_volumes_from_folder

    # Load simulation info
    sim_path = os.path.join(dataset_dir, "simulation_info.pkl")
    with open(sim_path, "rb") as f:
        sim_info = pickle.load(f)
    gt_noise = np.asarray(sim_info["noise_variance"]).ravel()
    gt_contrasts = np.asarray(sim_info["per_image_contrast"]).ravel()
    pa = np.asarray(sim_info["image_assignment"]).ravel()

    # Load GT volumes for metrics
    from recovar import simulator as sim_module

    volumes_path_root = sim_info["volumes_path_root"]
    grid_size = sim_info["grid_size"]
    trailing = sim_info.get("trailing_zero_format_in_vol_name", True)
    gt_vols = sim_module.load_volumes_from_folder(
        volumes_path_root, grid_size, trailing_zero_format_in_vol_name=trailing
    )
    if "scale_vol" in sim_info:
        gt_vols *= sim_info["scale_vol"]

    from recovar.synthetic_dataset import HeterogeneousVolumeDistribution

    valid_idx = np.ones(int(grid_size**3), dtype=np.float32)
    gt = HeterogeneousVolumeDistribution(gt_vols, pa, gt_contrasts, valid_indices=valid_idx)
    gt_mean = gt.get_mean()
    gt_variance = gt.get_spatial_variances(contrasted=False)
    u_gt, s_gt, _ = gt.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)

    logger.info("GT loaded: %d volumes, grid=%d, %d images", len(gt_vols), grid_size, len(pa))

    scores = {}

    # ========================================================================
    # Step 0: Create dataset (half-sets)
    # ========================================================================
    logger.info("=== Step 0: Load dataset ===")
    particles_file = os.path.join(dataset_dir, f"particles.{grid_size}.mrcs")
    poses_file = os.path.join(dataset_dir, "poses.pkl")
    ctf_file = os.path.join(dataset_dir, "ctf.pkl")

    ind_split = dataset.get_split_indices(particles_file)
    dataset_spec = dataset.HalfsetDatasetSpec(
        particles_file=particles_file,
        poses_file=poses_file,
        ctf_file=ctf_file,
        datadir=None,
    )
    cryos = dataset.load_halfset_dataset(dataset_spec, ind_split=ind_split, lazy=True)
    volume_shape = cryos[0].volume_shape
    volume_size = cryos[0].volume_size
    vol_norm = np.sqrt(np.prod(volume_shape))
    voxel_size = cryos[0].voxel_size
    logger.info("Dataset loaded: volume_shape=%s, voxel_size=%s", volume_shape, voxel_size)

    # Save halfset indices for reproducibility
    np.save(os.path.join(intermediates_dir, "halfset_ind0.npy"), ind_split[0])
    np.save(os.path.join(intermediates_dir, "halfset_ind1.npy"), ind_split[1])

    # ========================================================================
    # Step 1: compute_mean
    # ========================================================================
    logger.info("=== Step 1: compute_mean ===")
    t0 = time.time()

    # Initial noise from half-maps (needed for mean)
    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    means, mean_prior, mean_fsc = homogeneous.get_mean_conformation_relion(
        cryos, batch_size, noise_variance=noise_var_from_hf, use_regularization=False
    )
    t_mean = time.time() - t0
    logger.info("compute_mean took %.1fs", t_mean)

    # Save intermediates
    np.save(os.path.join(intermediates_dir, "mean_combined.npy"), means.combined)
    np.save(os.path.join(intermediates_dir, "mean_prior.npy"), mean_prior)
    np.save(os.path.join(intermediates_dir, "means_lhs.npy"), means.lhs)
    np.save(os.path.join(intermediates_dir, "mean_corrected0.npy"), means.corrected0)
    np.save(os.path.join(intermediates_dir, "mean_corrected1.npy"), means.corrected1)
    np.save(os.path.join(intermediates_dir, "noise_var_from_hf.npy"), noise_var_from_hf)

    # Metric: FSC vs GT mean
    from recovar import plot_utils as pu

    _, mean_fsc_score = pu.plot_fsc_new(
        gt_mean, means.combined, np.array(volume_shape), voxel_size, threshold=0.5, name="Mean FSC"
    )
    scores["mean_fsc"] = float(mean_fsc_score)
    logger.info("mean_fsc: %s", mean_fsc_score)

    # ========================================================================
    # Step 2: masking — use GT union mask instead of from_halfmaps
    # ========================================================================
    logger.info("=== Step 2: GT union masking ===")
    from scipy.ndimage import binary_dilation

    # Build union mask from all GT real-space volumes
    # Use make_mask_from_gt with from_ft=True to avoid needing get_idft3
    # (old ~/recovar has get_idft3 as a class method, not a module function)
    dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))
    union_binary = np.zeros(volume_shape, dtype=bool)
    for i in range(gt_vols.shape[0]):
        per_mask = mask.make_mask_from_gt(gt_vols[i], smax=3, iter=1, from_ft=True)
        union_binary |= per_mask > 0.5
    dilated_binary = binary_dilation(union_binary, iterations=dilation_iters)
    from recovar.mask import soften_volume_mask

    volume_mask = soften_volume_mask(dilated_binary, kern_rad=3).astype(np.float32)
    dilated_volume_mask = volume_mask  # same mask for both

    np.save(os.path.join(intermediates_dir, "volume_mask.npy"), volume_mask)
    np.save(os.path.join(intermediates_dir, "dilated_volume_mask.npy"), dilated_volume_mask)
    logger.info(
        "GT union mask: %d/%d voxels (%.1f%%)",
        int(np.sum(volume_mask > 0.5)),
        volume_mask.size,
        100 * np.sum(volume_mask > 0.5) / volume_mask.size,
    )

    # ========================================================================
    # Step 3: estimate_noise
    # ========================================================================
    logger.info("=== Step 3: estimate_noise ===")
    t0 = time.time()

    # Outside mask noise
    masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(
        cryos[0], dilated_volume_mask, batch_size
    )
    white_noise_var = noise.estimate_white_noise_variance_from_mask(cryos[0], dilated_volume_mask, batch_size)
    _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
        cryos[0], dilated_volume_mask, batch_size
    )
    # Upper bound from inside mask
    radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
        cryos[0], means.combined, dilated_volume_mask, batch_size
    )

    # Take minimum
    noise_var_used = np.where(masked_image_PS > radial_ub_noise_var, radial_ub_noise_var, masked_image_PS)
    # Fix negatives
    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

    t_noise = time.time() - t0
    logger.info("estimate_noise took %.1fs", t_noise)

    # Initialize and set noise model on cryos
    for cryo in cryos:
        cryo.set_radial_noise_model(None)
    noise.update_noise_variance(noise_var_used, cryos)

    np.save(os.path.join(intermediates_dir, "noise_var_used.npy"), noise_var_used)
    np.save(os.path.join(intermediates_dir, "radial_ub_noise_var.npy"), radial_ub_noise_var)
    np.save(os.path.join(intermediates_dir, "masked_image_PS.npy"), masked_image_PS)
    np.save(os.path.join(intermediates_dir, "image_PS.npy"), image_PS)

    # Metric: noise correlation & relative error
    n_shells = min(len(gt_noise), len(noise_var_used))
    noise_corr = float(np.corrcoef(gt_noise[:n_shells], noise_var_used[:n_shells])[0, 1])
    rel_err = np.abs(gt_noise[:n_shells] - noise_var_used[:n_shells]) / (gt_noise[:n_shells] + 1e-12)
    scores["noise_correlation"] = noise_corr
    scores["noise_mean_relative_error"] = float(np.mean(rel_err))
    scores["noise_median_relative_error"] = float(np.median(rel_err))
    logger.info("noise_correlation: %s", noise_corr)

    # ========================================================================
    # Step 4: upper_bound_noise + compute_variance
    # ========================================================================
    logger.info("=== Step 4: compute_variance ===")
    t0 = time.time()

    # Upper bound noise by signal+noise
    variance_ub, ub_noise = noise.upper_bound_noise_by_signal_p_noise_dispatched(
        noise_var_used, cryos, means, batch_size, dilated_volume_mask
    )

    # Compute variance with regularization
    variance_est, variance_prior, variance_fsc_curve, lhs, noise_p_variance = covariance_estimation.compute_variance(
        cryos, means.combined, batch_size // 2, dilated_volume_mask, use_regularization=True, disc_type="cubic"
    )
    t_var = time.time() - t0
    logger.info("compute_variance took %.1fs", t_var)

    np.save(os.path.join(intermediates_dir, "variance_combined.npy"), variance_est["combined"])
    np.save(os.path.join(intermediates_dir, "variance_prior.npy"), variance_prior)
    np.save(os.path.join(intermediates_dir, "variance_lhs.npy"), lhs)

    # Metric: Fourier variance FSC (GT per-Fourier-voxel power vs estimated)
    gt_fourier_variance = np.sum(np.abs(gt.get_covariance_square_root(contrasted=False)) ** 2, axis=-1)
    _, var_fsc_score = pu.plot_fsc_new(
        gt_fourier_variance,
        variance_est["combined"],
        np.array(volume_shape),
        voxel_size,
        threshold=0.5,
        name="Variance Fourier FSC",
    )
    scores["variance_fourier_fsc"] = float(var_fsc_score)
    # Keep old key for backward compat
    scores["variance_fsc"] = float(var_fsc_score)
    logger.info("variance_fourier_fsc: %s", var_fsc_score)

    # ========================================================================
    # Step 5: principal_components (covariance_columns + SVD + projected_cov)
    # ========================================================================
    logger.info("=== Step 5: principal_components ===")
    t0 = time.time()

    # Build options matching pipeline defaults
    options = {
        "zs_dim_to_test": [4, 10],
        "contrast": "contrast_qr",
        "keep_intermediate": True,
        "ignore_zero_frequency": True,
        "use_combined_mean": True,
    }

    valid_idx = cryos[0].get_valid_frequency_indices()

    # Default covariance options
    covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos[0].grid_size)

    u, s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(
        cryos,
        options,
        means,
        mean_prior,
        volume_mask,
        dilated_volume_mask,
        valid_idx,
        batch_size,
        gpu_memory_to_use=gpu_memory,
        covariance_options=covariance_options,
        variance_estimate=variance_est["combined"],
        use_reg_mean_in_contrast=False,
    )
    t_pcs = time.time() - t0
    logger.info("principal_components took %.1fs", t_pcs)

    # Save PCs
    np.save(os.path.join(intermediates_dir, "u_rescaled.npy"), u["rescaled"])
    np.save(os.path.join(intermediates_dir, "s_rescaled.npy"), s["rescaled"])
    np.save(os.path.join(intermediates_dir, "picked_frequencies.npy"), picked_frequencies)
    if "rescaled_no_contrast" in u:
        np.save(os.path.join(intermediates_dir, "u_rescaled_no_contrast.npy"), u["rescaled_no_contrast"])

    # Metric: pcs_relative_variance
    from recovar import metrics
    from recovar import linalg

    # Convert pipeline eigenvectors to real space for comparison
    n_pcs_to_test = min(20, u["rescaled"].shape[1])
    u_real = linalg.batch_idft3(u["rescaled"][:, :n_pcs_to_test], volume_shape, batch_size=2).real
    u_est = np.array(u_real.reshape(volume_size, n_pcs_to_test)) * vol_norm
    variance_all, rel_var, norm_var = metrics.get_all_variance_scores(u_est, u_gt, s_gt)
    if rel_var.size > 4:
        scores["pcs_relative_variance_4"] = float(rel_var[4])
    if rel_var.size > 10:
        scores["pcs_relative_variance_10"] = float(rel_var[10])
    logger.info("pcs_relative_variance_4: %s", scores.get("pcs_relative_variance_4", "N/A"))
    logger.info("pcs_relative_variance_10: %s", scores.get("pcs_relative_variance_10", "N/A"))

    # ========================================================================
    # Step 5b: covariance_columns — save GT covariance comparison
    # ========================================================================
    logger.info("=== Step 5b: covariance_columns FSC ===")
    gt_cov_cols = gt.get_covariance_columns(picked_frequencies, contrasted=False)
    est_cov_cols = covariance_cols.get("est_mask", covariance_cols.get("est"))
    if est_cov_cols is not None and gt_cov_cols is not None:
        # Mean FSC across columns
        col_fscs = []
        for i in range(min(est_cov_cols.shape[1], gt_cov_cols.shape[1])):
            _, col_fsc = pu.plot_fsc_new(
                gt_cov_cols[:, i], est_cov_cols[:, i], np.array(volume_shape), voxel_size, threshold=0.5
            )
            col_fscs.append(float(col_fsc))
        scores["covariance_columns_mean_fsc"] = float(np.mean(col_fscs))
        scores["covariance_columns_median_fsc"] = float(np.median(col_fscs))
        logger.info("covariance_columns_mean_fsc: %s", scores["covariance_columns_mean_fsc"])

    # ========================================================================
    # Step 5c: projected_covariance — with GT eigenvectors as basis
    # ========================================================================
    logger.info("=== Step 5c: projected_covariance with GT basis ===")
    # Get GT eigenvectors in Fourier space
    u_gt_fourier, s_gt_fourier, _ = gt.get_vol_svd(contrasted=False, real_space=False)
    n_gt_pcs = min(10, u_gt_fourier.shape[1])
    gt_basis = u_gt_fourier[:, :n_gt_pcs]  # (vol_size, n_pcs)

    proj_cov = covariance_estimation.compute_projected_covariance(
        cryos, means.combined, gt_basis, volume_mask, batch_size, disc_type="linear_interp", disc_type_u="linear_interp"
    )
    # GT projected covariance = diag(s_gt_fourier[:n_gt_pcs]**2)
    gt_proj_cov = np.diag(s_gt_fourier[:n_gt_pcs] ** 2)
    frob_err = np.linalg.norm(proj_cov - gt_proj_cov) / (np.linalg.norm(gt_proj_cov) + 1e-12)
    scores["projected_covariance_relative_frobenius_error"] = float(frob_err)
    logger.info("projected_covariance_relative_frobenius_error: %s", frob_err)

    np.save(os.path.join(intermediates_dir, "projected_covariance_gt_basis.npy"), proj_cov)

    # ========================================================================
    # Step 6: embedding with contrast
    # ========================================================================
    logger.info("=== Step 6: embedding with contrast ===")
    t0 = time.time()
    embedding_scores = {}

    for zdim in [4, 10]:
        n_pcs_to_use = zdim
        zs, cov_zs, est_contrasts_reg, _ = embedding.get_per_image_embedding(
            means.combined,
            u["rescaled"],
            s["rescaled"],
            n_pcs_to_use,
            cryos,
            volume_mask,
            gpu_memory,
            "linear_interp",
            contrast_grid=None,
            contrast_option="contrast",
            ignore_zero_frequency=options["ignore_zero_frequency"],
        )
        # Save embeddings
        np.save(os.path.join(intermediates_dir, f"embedding_reg_zs_{zdim}.npy"), zs)
        np.save(os.path.join(intermediates_dir, f"embedding_reg_contrasts_{zdim}.npy"), est_contrasts_reg)

        # Metric: embedding squared error
        _, avg_var = metrics.variance_of_zs(zs, pa)
        scores[f"embedding_squared_error_{zdim}"] = float(avg_var)
        logger.info("embedding_squared_error_%d: %s", zdim, avg_var)

        # Metric: contrast MAE
        contrast_mae = float(np.mean(np.abs(gt_contrasts - est_contrasts_reg.ravel())))
        scores[f"contrasts_{zdim}"] = contrast_mae
        logger.info("contrasts_%d: %s", zdim, contrast_mae)

    t_emb = time.time() - t0
    logger.info("embedding with contrast took %.1fs", t_emb)

    # ========================================================================
    # Step 7: embedding without contrast (noreg)
    # ========================================================================
    logger.info("=== Step 7: embedding without contrast (no regularization) ===")
    t0 = time.time()

    for zdim in [4, 10]:
        n_pcs_to_use = zdim
        zs_noreg, cov_zs_noreg, est_contrasts_noreg, _ = embedding.get_per_image_embedding(
            means.combined,
            u["rescaled"],
            s["rescaled"] * 0 + np.inf,
            n_pcs_to_use,
            cryos,
            volume_mask,
            gpu_memory,
            "linear_interp",
            contrast_grid=None,
            contrast_option="contrast",
            ignore_zero_frequency=options["ignore_zero_frequency"],
        )
        np.save(os.path.join(intermediates_dir, f"embedding_noreg_zs_{zdim}.npy"), zs_noreg)
        np.save(os.path.join(intermediates_dir, f"embedding_noreg_contrasts_{zdim}.npy"), est_contrasts_noreg)

        _, avg_var_noreg = metrics.variance_of_zs(zs_noreg, pa)
        scores[f"embedding_squared_error_{zdim}_noreg"] = float(avg_var_noreg)
        logger.info("embedding_squared_error_%d_noreg: %s", zdim, avg_var_noreg)

        contrast_mae_noreg = float(np.mean(np.abs(gt_contrasts - est_contrasts_noreg.ravel())))
        scores[f"contrasts_{zdim}_noreg"] = contrast_mae_noreg
        logger.info("contrasts_%d_noreg: %s", zdim, contrast_mae_noreg)

    t_emb_noreg = time.time() - t0
    logger.info("embedding without contrast took %.1fs", t_emb_noreg)

    # ========================================================================
    # Save scores and metadata
    # ========================================================================
    # Normalize scores
    cleaned = {}
    for k, v in scores.items():
        if isinstance(v, (np.floating, np.integer)):
            cleaned[k] = float(v)
        elif isinstance(v, (int, float)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    scores = cleaned

    scores_path = os.path.join(output_dir, "per_function_scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    logger.info("Scores saved to: %s", scores_path)

    metadata = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_base": "~/recovar (published recovar, ma-gilles/recovar.git)",
        "recovar_path": recovar_path,
        "dataset_dir": dataset_dir,
        "grid_size": int(grid_size),
        "voxel_size": float(voxel_size),
        "n_images": int(len(pa)),
        "batch_size": batch_size,
        "notes": "Per-function baseline for isolated pipeline function tests.",
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    logger.info("\n=== All scores ===")
    for k, v in sorted(scores.items()):
        logger.info("  %s: %s", k, v)

    logger.info("\nDone! Intermediates at: %s", intermediates_dir)


if __name__ == "__main__":
    main()
