#!/usr/bin/env python
"""Run each pipeline function on ET data and report scores.

Usage:
  python et_isolated_test.py --dataset-dir <path> --output <scores.json> [--old]

With --old: inserts ~/recovar on sys.path to use old code.
Without: uses whatever recovar is installed.
"""
import argparse, json, logging, os, pickle, numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--old", action="store_true", help="Use old ~/recovar code")
    ap.add_argument("--halfsets", default=None, help="Pickle file with pre-computed halfset split")
    args = ap.parse_args()

    if args.old:
        import sys
        sys.path.insert(0, "/home/mg6942/recovar")

    import recovar
    logger.info("recovar: %s", recovar.__file__)

    if args.old:
        from recovar import plot_utils
        from recovar import metrics as gt_metrics
        from recovar import synthetic_dataset
        from recovar import dataset, noise, mask, utils, homogeneous
        from recovar import covariance_estimation, principal_components, embedding, linalg
    else:
        from recovar.output import plot_utils, metrics as gt_metrics
        from recovar.simulation import synthetic_dataset
        from recovar import utils
        from recovar.data_io import dataset
        from recovar.reconstruction import homogeneous, noise
        from recovar.heterogeneity import covariance_estimation, principal_components, embedding
        from recovar.core import mask, linalg

    sim_path = os.path.join(args.dataset_dir, "simulation_info.pkl")
    if args.old:
        gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_path)
    else:
        gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_path)
    gt_mean = gt.get_mean()
    u_gt, s_gt, _ = gt.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)
    with open(sim_path, "rb") as f:
        sim_info = pickle.load(f)
    gt_contrasts = np.asarray(sim_info["per_image_contrast"]).ravel()
    gt_noise = np.asarray(sim_info["noise_variance"]).ravel()
    pa = np.asarray(sim_info["image_assignment"]).ravel()
    grid_size = sim_info["grid_size"]
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = grid_size**3
    vol_norm = np.sqrt(np.prod(volume_shape))
    batch_size = 512
    gpu_memory = 40.0

    gt_fourier_var = np.sum(np.abs(gt.get_covariance_square_root(contrasted=False))**2, axis=-1)
    def F(v): return float(np.asarray(v)) if v is not None else None

    scores = {}

    # Load ET dataset — use tilt-aware split (all tilts of a particle in same halfset)
    particles_file = os.path.join(args.dataset_dir, "particles.star")
    poses_file = os.path.join(args.dataset_dir, "poses.pkl")
    ctf_file = os.path.join(args.dataset_dir, "ctf.pkl")
    if args.halfsets:
        with open(args.halfsets, "rb") as f:
            ind_split = pickle.load(f)
        logger.info("Using pre-computed halfsets from %s", args.halfsets)
    elif hasattr(dataset, 'get_split_tilt_indices'):
        ind_split = dataset.get_split_tilt_indices(particles_file)
    else:
        ind_split = dataset.get_split_indices(particles_file)
    logger.info("half0: %d images, half1: %d images", len(ind_split[0]), len(ind_split[1]))
    cryos = dataset.get_split_datasets(particles_file, poses_file, ctf_file,
                                        datadir=None, ind_split=ind_split, lazy=True)
    voxel_size = cryos[0].voxel_size

    # Step 1: Mean
    logger.info("Step 1: Mean")
    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)
    means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
        cryos, batch_size, noise_variance=noise_var_from_hf, use_regularization=False)
    _, mean_fsc = plot_utils.plot_fsc_new(
        gt_mean, means["combined"], np.array(volume_shape), voxel_size, threshold=0.5, name="Mean")
    scores["mean_fsc"] = F(mean_fsc)
    logger.info("  mean_fsc = %.6f", scores["mean_fsc"])

    # Step 2: Mask
    gt_mask_path = os.path.join(args.dataset_dir, "gt_masks", "gt_union_mask.mrc")
    if os.path.exists(gt_mask_path):
        volume_mask, dilated_volume_mask = mask.masking_options(gt_mask_path, means, volume_shape, np.float32)
    else:
        volume_mask, dilated_volume_mask = mask.masking_options("from_halfmaps", means, volume_shape, np.float32)

    # Step 3: Noise
    logger.info("Step 3: Noise")
    for cryo in cryos:
        cryo.set_radial_noise_model(None)
    masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(cryos[0], dilated_volume_mask, batch_size)
    radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
        cryos[0], means["combined"], dilated_volume_mask, batch_size)
    _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(cryos[0], dilated_volume_mask, batch_size)
    noise_var_used = np.where(masked_image_PS > radial_ub_noise_var, radial_ub_noise_var, masked_image_PS)
    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)
    noise.update_noise_variance(noise_var_used, cryos)
    n_sh = min(len(gt_noise), len(noise_var_used))
    scores["noise_correlation"] = F(np.corrcoef(gt_noise[:n_sh], noise_var_used[:n_sh])[0,1])
    logger.info("  noise_correlation = %.6f", scores["noise_correlation"])

    # Step 3b: Upper bound
    noise.upper_bound_noise_by_signal_p_noise_dispatched(noise_var_used, cryos, means, batch_size, dilated_volume_mask)

    # Step 4: Variance
    logger.info("Step 4: Variance")
    variance_est, _, _, _, _ = covariance_estimation.compute_variance(
        cryos, means["combined"], max(1, batch_size // 2),
        dilated_volume_mask, use_regularization=True, disc_type="cubic")
    _, var_fsc = plot_utils.plot_fsc_new(
        gt_fourier_var, variance_est["combined"], np.array(volume_shape), voxel_size,
        threshold=0.5, name="Var FSC")
    scores["variance_fsc"] = F(var_fsc)
    logger.info("  variance_fsc = %.6f", scores["variance_fsc"])

    # Step 5a: Covariance columns
    logger.info("Step 5a: Covariance columns")
    options = {"zs_dim_to_test": [4, 10], "contrast": "contrast_qr",
               "keep_intermediate": True, "ignore_zero_frequency": False, "use_combined_mean": True}
    valid_idx = cryos[0].get_valid_frequency_indices()
    covariance_options = covariance_estimation.get_default_covariance_computation_options(grid_size)

    # Covariance columns — compute H/B norms for comparison
    logger.info("  Computing H, B via compute_both_H_B...")
    try:
        H, B = covariance_estimation.compute_both_H_B(
            cryos, means, dilated_volume_mask, None, gpu_memory, False, covariance_options)
        scores["H_norm"] = float(np.linalg.norm(np.asarray(H)))
        scores["B_norm"] = float(np.linalg.norm(np.asarray(B)))
        scores["H_max"] = float(np.max(np.abs(np.asarray(H))))
        scores["B_max"] = float(np.max(np.abs(np.asarray(B))))
        logger.info("  H_norm=%.6e  B_norm=%.6e", scores["H_norm"], scores["B_norm"])
    except Exception as e:
        logger.warning("  compute_both_H_B failed: %s", e)

    # Step 5b: Projected covariance with GT basis
    logger.info("Step 5b: Projected covariance")
    u_gt_fourier, s_gt_fourier, _ = gt.get_vol_svd(contrasted=False, real_space=False)
    n_gt_pcs = min(10, u_gt_fourier.shape[1])
    gt_basis = u_gt_fourier[:, :n_gt_pcs]

    proj_cov = covariance_estimation.compute_projected_covariance(
        cryos, means["combined"], gt_basis, volume_mask,
        batch_size, disc_type="linear_interp", disc_type_u="linear_interp")
    gt_proj_cov = np.diag(s_gt_fourier[:n_gt_pcs] ** 2)
    frob_err = float(np.linalg.norm(proj_cov - gt_proj_cov) / (np.linalg.norm(gt_proj_cov) + 1e-12))
    scores["proj_cov_frobenius_error"] = frob_err
    logger.info("  proj_cov_frobenius_error = %.6f", frob_err)

    # Step 5c: Full PCA
    logger.info("Step 5c: Full PCA (estimate_principal_components)")
    u, s, _, _, _ = principal_components.estimate_principal_components(
        cryos, options, means, mean_prior, volume_mask, dilated_volume_mask,
        valid_idx, batch_size, gpu_memory_to_use=gpu_memory,
        covariance_options=covariance_options,
        variance_estimate=variance_est["combined"],
        use_reg_mean_in_contrast=True)
    n_pcs = min(20, u["rescaled"].shape[1])
    u_real = linalg.batch_idft3(u["rescaled"][:, :n_pcs], volume_shape, batch_size=2).real
    u_est = np.array(u_real.reshape(volume_size, n_pcs)) * vol_norm
    _, rel_var, _ = gt_metrics.get_all_variance_scores(u_est, u_gt, s_gt)
    scores["svd_relative_variance_4"] = F(rel_var[4]) if rel_var.size > 4 else None
    scores["svd_relative_variance_10"] = F(rel_var[10]) if rel_var.size > 10 else None
    scores["eigenvalue_0"] = float(s["rescaled"][0])
    scores["eigenvalue_1"] = float(s["rescaled"][1]) if len(s["rescaled"]) > 1 else None
    logger.info("  svd_4=%.6f  svd_10=%.6f  s0=%.2f  s1=%.2f",
                scores.get("svd_relative_variance_4", 0), scores.get("svd_relative_variance_10", 0),
                scores["eigenvalue_0"], scores.get("eigenvalue_1", 0))

    # Step 6: Embedding
    logger.info("Step 6: Embedding")
    for zdim in [4, 10]:
        zs, _, est_contrasts, _ = embedding.get_per_image_embedding(
            means["combined"], u["rescaled"], s["rescaled"], zdim,
            cryos, volume_mask, gpu_memory, "linear_interp",
            contrast_grid=None, contrast_option="contrast", ignore_zero_frequency=False)
        pa_use = sim_info.get("tilt_series_assignment", pa)
        if hasattr(pa_use, "__len__") and len(pa_use) == zs.shape[0]:
            _, ratio = gt_metrics.variance_of_zs(zs, np.asarray(pa_use).ravel())
            scores[f"embedding_squared_error_{zdim}"] = F(ratio)
        gt_c = gt_contrasts[:len(est_contrasts.ravel())]
        scores[f"contrast_abs_error_{zdim}"] = F(np.mean(np.abs(gt_c - est_contrasts.ravel())))
    logger.info("  contrast_4=%.6f", scores.get("contrast_abs_error_4", 0))

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    logger.info("Scores saved to %s", args.output)
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
