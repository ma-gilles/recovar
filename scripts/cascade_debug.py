#!/usr/bin/env python
"""Cascade debug: run each pipeline function with NEW code, feeding outputs forward.

Tests whether small per-function drifts compound into the 66% contrast regression.
"""
import json, logging, os, pickle, time, numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=None,
                    help="Override dataset dir (default: PDB baseline or FUNCTION_TEST_DATASET_DIR)")
    ap.add_argument("--baseline-dir", default="/scratch/gpfs/GILLES/mg6942/pdb_baseline_snr01/function_baselines_gtmask")
    ap.add_argument("--swap-after-step", type=int, default=0,
                    help="Use OLD intermediates for steps 1..N, NEW code for N+1..end. "
                         "0=full cascade (all NEW). 6=all OLD inputs to embedding.")
    cli_args = ap.parse_args()

    from recovar import utils
    from recovar.data_io import dataset
    from recovar.reconstruction import homogeneous, noise
    from recovar.heterogeneity import covariance_estimation, principal_components, embedding
    from recovar.output import plot_utils, metrics
    from recovar.simulation import synthetic_dataset
    from recovar.core import mask, linalg
    import recovar.core.fourier_transform_utils as ftu

    BASELINE_DIR = cli_args.baseline_dir
    INTER_DIR = os.path.join(BASELINE_DIR, "intermediates")
    DATASET_DIR = cli_args.dataset_dir or os.environ.get("FUNCTION_TEST_DATASET_DIR",
                                  "/scratch/gpfs/GILLES/mg6942/pdb_baseline_snr01/test_dataset")
    SCORES_PATH = os.path.join(BASELINE_DIR, "per_function_scores.json")

    with open(SCORES_PATH) as f:
        baseline = json.load(f)

    # Load dataset
    sim_path = os.path.join(DATASET_DIR, "simulation_info.pkl")
    with open(sim_path, "rb") as f:
        sim_info = pickle.load(f)
    grid_size = sim_info["grid_size"]
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_path)
    gt_mean = gt.get_mean()
    u_gt, s_gt, _ = gt.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)
    gt_contrasts = np.asarray(sim_info["per_image_contrast"]).ravel()
    gt_noise = np.asarray(sim_info["noise_variance"]).ravel()
    pa = np.asarray(sim_info["image_assignment"]).ravel()

    particles_file = os.path.join(DATASET_DIR, f"particles.{grid_size}.mrcs")
    poses_file = os.path.join(DATASET_DIR, "poses.pkl")
    ctf_file = os.path.join(DATASET_DIR, "ctf.pkl")
    ind_split = dataset.get_split_indices(particles_file)
    cryos = dataset.get_split_datasets(particles_file, poses_file, ctf_file,
                                        datadir=None, ind_split=ind_split, lazy=True)
    volume_shape = cryos[0].volume_shape
    volume_size = cryos[0].volume_size
    vol_norm = np.sqrt(np.prod(volume_shape))
    voxel_size = cryos[0].voxel_size
    batch_size = 512
    gpu_memory = 40.0

    # Load OLD intermediates for comparison
    def load_old(name):
        return np.load(os.path.join(INTER_DIR, f"{name}.npy"))

    swap = cli_args.swap_after_step
    logger.info("swap_after_step=%d (steps 1..%d use OLD intermediates)", swap, swap)
    scores_cascade = {}

    # ==============================
    # STEP 1: Mean
    # ==============================
    logger.info("=== CASCADE STEP 1: Mean ===")
    noise_var_from_hf = load_old("noise_var_from_hf")
    if swap >= 1:
        logger.info("  [USING OLD mean]")
        means = {"combined": load_old("mean_combined"), "lhs": load_old("means_lhs"),
                 "corrected0": load_old("mean_corrected0"), "corrected1": load_old("mean_corrected1")}
        mean_prior = load_old("mean_prior")
    else:
        means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
            cryos, batch_size, noise_variance=noise_var_from_hf, use_regularization=False)
    _, mean_fsc = plot_utils.plot_fsc_new(
        gt_mean, means["combined"], np.array(volume_shape), voxel_size, threshold=0.5, name="Mean FSC")
    scores_cascade["mean_fsc"] = float(np.asarray(mean_fsc))
    logger.info("mean_fsc: cascade=%.8f baseline=%.8f", scores_cascade["mean_fsc"], baseline["mean_fsc"])

    # ==============================
    # STEP 2: Masking (use GT mask from baseline)
    # ==============================
    volume_mask = load_old("volume_mask")
    dilated_volume_mask = load_old("dilated_volume_mask")

    # ==============================
    # STEP 3: Noise + upper-bound
    # ==============================
    logger.info("=== CASCADE STEP 3: Noise ===")
    if swap >= 3:
        logger.info("  [USING OLD noise]")
        noise_var_used = load_old("noise_var_used")
    else:
        masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(
            cryos[0], dilated_volume_mask, batch_size)
        radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
            cryos[0], means["combined"], dilated_volume_mask, batch_size)
        _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
            cryos[0], dilated_volume_mask, batch_size)
        noise_var_used = np.where(masked_image_PS > radial_ub_noise_var, radial_ub_noise_var, masked_image_PS)
        noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

    n_sh = min(len(gt_noise), len(noise_var_used))
    noise_corr = float(np.corrcoef(gt_noise[:n_sh], noise_var_used[:n_sh])[0, 1])
    scores_cascade["noise_correlation"] = noise_corr
    logger.info("noise_correlation: cascade=%.8f baseline=%.8f", noise_corr, baseline["noise_correlation"])

    for cryo in cryos:
        cryo.set_radial_noise_model(None)
    noise.update_noise_variance(noise_var_used, cryos)

    # Upper-bound noise by signal+noise (matches pipeline line 827)
    logger.info("=== CASCADE STEP 3b: Upper-bound noise ===")
    variance_est_ub, ub_noise_var = noise.upper_bound_noise_by_signal_p_noise_dispatched(
        noise_var_used, cryos, means, batch_size, dilated_volume_mask)

    # ==============================
    # STEP 4: Variance (NEW code, CASCADE mean+noise)
    # ==============================
    logger.info("=== CASCADE STEP 4: Variance ===")
    if swap >= 4:
        logger.info("  [USING OLD variance]")
        variance_est = {"combined": load_old("variance_combined")}
    else:
        variance_est, variance_prior, _, _, _ = covariance_estimation.compute_variance(
            cryos, means["combined"], utils.safe_batch_size(batch_size // 2),
            dilated_volume_mask, use_regularization=True, disc_type="cubic")

    gt_variance = gt.get_spatial_variances(contrasted=False)
    _, var_fsc = plot_utils.plot_fsc_new(
        gt_variance, variance_est["combined"], np.array(volume_shape), voxel_size,
        threshold=0.5, name="Variance FSC")
    scores_cascade["variance_fsc"] = float(np.asarray(var_fsc))
    logger.info("variance_fsc: cascade=%.8f baseline=%.8f", scores_cascade["variance_fsc"], baseline["variance_fsc"])

    # ==============================
    # STEP 5: PCA (NEW code, CASCADE everything)
    # ==============================
    logger.info("=== CASCADE STEP 5: PCA ===")
    if swap >= 5:
        logger.info("  [USING OLD PCA]")
        u = {"rescaled": load_old("u_rescaled")}
        s = {"rescaled": load_old("s_rescaled")}
    else:
        options = {
            "zs_dim_to_test": [4, 10],
            "contrast": "contrast_qr",
            "keep_intermediate": True,
            "ignore_zero_frequency": False,
            "use_combined_mean": True,
        }
        valid_idx = cryos[0].get_valid_frequency_indices()
        covariance_options = covariance_estimation.get_default_covariance_computation_options(grid_size)

        u_out, s_out, covariance_cols, picked_frequencies, column_fscs = \
            principal_components.estimate_principal_components(
                cryos, options, means, mean_prior, volume_mask, dilated_volume_mask,
                valid_idx, batch_size, gpu_memory_to_use=gpu_memory,
                covariance_options=covariance_options,
                variance_estimate=variance_est["combined"],
                use_reg_mean_in_contrast=True)
        u = u_out
        s = s_out

    n_pcs_to_test = min(20, u["rescaled"].shape[1])
    u_real = linalg.batch_idft3(u["rescaled"][:, :n_pcs_to_test], volume_shape, batch_size=2).real
    u_est = np.array(u_real.reshape(volume_size, n_pcs_to_test)) * vol_norm
    _, rel_var, _ = metrics.get_all_variance_scores(u_est, u_gt, s_gt)
    scores_cascade["pcs_relative_variance_4"] = float(rel_var[4]) if rel_var.size > 4 else None
    scores_cascade["pcs_relative_variance_10"] = float(rel_var[10]) if rel_var.size > 10 else None
    logger.info("pcs_relative_variance_4: cascade=%.8f baseline=%.8f",
                scores_cascade.get("pcs_relative_variance_4", 0), baseline.get("pcs_relative_variance_4", 0))

    # ==============================
    # STEP 6: Embedding with contrast (NEW code, CASCADE PCA)
    # ==============================
    logger.info("=== CASCADE STEP 6: Embedding with contrast ===")
    for zdim in [4, 10]:
        n_pcs_to_use = zdim
        zs, _, est_contrasts_reg, _ = embedding.get_per_image_embedding(
            means["combined"], u["rescaled"], s["rescaled"], n_pcs_to_use,
            cryos, volume_mask, gpu_memory, "linear_interp",
            contrast_grid=None, contrast_option="contrast",
            ignore_zero_frequency=False)

        _, avg_var = metrics.variance_of_zs(zs, pa)
        scores_cascade[f"embedding_squared_error_{zdim}"] = float(avg_var)

        contrast_mae = float(np.mean(np.abs(gt_contrasts - est_contrasts_reg.ravel())))
        scores_cascade[f"contrasts_{zdim}"] = contrast_mae
        logger.info("contrasts_%d: cascade=%.8f baseline=%.8f", zdim, contrast_mae, baseline.get(f"contrasts_{zdim}", 0))

    # ==============================
    # SUMMARY
    # ==============================
    print("\n" + "=" * 80)
    print("CASCADE DEBUG SUMMARY")
    print("=" * 80)
    print(f"{'Metric':50s}  {'Baseline':>12s}  {'Cascade':>12s}  {'Drift%':>8s}")
    print("-" * 88)
    for k in sorted(scores_cascade.keys()):
        c = scores_cascade[k]
        b = baseline.get(k, 0)
        if c is not None and b != 0:
            pct = 100 * (c - b) / abs(b)
            print(f"{k:50s}  {b:12.8f}  {c:12.8f}  {pct:+7.3f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
